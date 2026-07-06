"""Tests for incremental (delta-per-level) glacier directory creation."""

import json
import os
import tarfile

import shutil

import numpy as np
import pytest
import xarray as xr

import oggm
from oggm import cfg, utils, workflow
from oggm.exceptions import InvalidWorkflowError
from oggm.utils import _downloads

pytestmark = pytest.mark.test_env("utils")


def _make_fake_gdir(path, extra_file=None):
    """A minimal on-disk stand-in for a glacier directory.

    Contains regular files plus a data_store.zarr with one group, which
    is the structure snapshot_gdir_state must understand.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "diagnostics.json"), "w") as f:
        json.dump({"a": 1}, f)
    with open(os.path.join(path, "log.txt"), "w") as f:
        f.write("task log\n")
    if extra_file:
        with open(os.path.join(path, extra_file), "w") as f:
            f.write("data\n")
    ds = xr.Dataset({"thick": ("x", np.arange(5, dtype=float))})
    store = os.path.join(path, "data_store.zarr")
    ds.to_zarr(
        store,
        group="inversion_flowlines",
        mode="a",
        zarr_format=2,
        consolidated=True,
    )
    return path


def test_snapshot_gdir_state(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))

    state = utils.snapshot_gdir_state(gdir_dir)

    # Regular files are keyed by relative path
    assert "diagnostics.json" in state
    assert "log.txt" in state
    # The zarr store is keyed per top-level group, not per chunk file
    assert "data_store.zarr/inversion_flowlines" in state
    assert not any(
        k.startswith("data_store.zarr/inversion_flowlines/") for k in state
    )
    # Root consolidated metadata is not tracked (stale after layering)
    assert "data_store.zarr/.zmetadata" not in state
    assert "data_store.zarr/zarr.json" not in state

    # Unchanged directory -> identical snapshot
    assert utils.snapshot_gdir_state(gdir_dir) == state

    # Content change is detected, everything else is stable
    with open(os.path.join(gdir_dir, "diagnostics.json"), "w") as f:
        json.dump({"a": 2}, f)
    new_state = utils.snapshot_gdir_state(gdir_dir)
    assert new_state["diagnostics.json"] != state["diagnostics.json"]
    assert new_state["log.txt"] == state["log.txt"]
    assert (
        new_state["data_store.zarr/inversion_flowlines"]
        == state["data_store.zarr/inversion_flowlines"]
    )

    # Adding a zarr group shows up as a new key; existing group unchanged
    ds = xr.Dataset({"w": ("x", np.ones(3))})
    ds.to_zarr(
        os.path.join(gdir_dir, "data_store.zarr"),
        group="model_flowlines",
        mode="a",
        zarr_format=2,
        consolidated=True,
    )
    grown = utils.snapshot_gdir_state(gdir_dir)
    assert "data_store.zarr/model_flowlines" in grown
    assert (
        grown["data_store.zarr/inversion_flowlines"]
        == state["data_store.zarr/inversion_flowlines"]
    )


def test_write_level_manifest_schema(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    prev_state = utils.snapshot_gdir_state(gdir_dir)

    # Simulate a level's work: one updated file, one new file, one new
    # zarr group
    with open(os.path.join(gdir_dir, "log.txt"), "a") as f:
        f.write("more work\n")
    with open(os.path.join(gdir_dir, "mb_calib.json"), "w") as f:
        json.dump({"melt_f": 5.0}, f)
    ds = xr.Dataset({"w": ("x", np.ones(3))})
    ds.to_zarr(
        os.path.join(gdir_dir, "data_store.zarr"),
        group="model_flowlines",
        mode="a",
        zarr_format=2,
        consolidated=True,
    )

    manifest_path, changed = utils.write_level_manifest(
        gdir_dir,
        level=3,
        prev_state=prev_state,
        dataset_tag="abc123",
        requires=[0, 1, 2],
        border=80,
        rgi_version="62",
    )

    assert os.path.basename(manifest_path) == "L3.manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "delta"
    assert manifest["rgi_id"] == "RGI60-11.00897"
    assert manifest["level"] == 3
    assert manifest["requires"] == [0, 1, 2]
    assert manifest["includes_levels"] == [3]
    assert manifest["dataset_tag"] == "abc123"
    assert manifest["border"] == 80
    assert manifest["rgi_version"] == "62"
    assert manifest["oggm_version"]
    assert manifest["created"]
    assert manifest["files"]["added"] == ["mb_calib.json"]
    assert manifest["files"]["updated"] == ["log.txt"]
    assert manifest["zarr_groups"] == ["model_flowlines"]

    # changed_paths is what gdir_to_tar(include=...) needs: the changed
    # files, the changed store groups, and the manifest itself
    assert set(changed) == {
        "mb_calib.json",
        "log.txt",
        "data_store.zarr/model_flowlines",
        "L3.manifest.json",
    }


class _FakeGdir:
    def __init__(self, path, base_dir):
        self.dir = path
        self.base_dir = base_dir
        self.rgi_id = os.path.basename(path)


def _simulate_level(gdir_dir, level=3):
    """Take a snapshot, do some 'level work', write its manifest."""
    prev_state = utils.snapshot_gdir_state(gdir_dir)
    with open(os.path.join(gdir_dir, "log.txt"), "a") as f:
        f.write(f"level {level} work\n")
    with open(os.path.join(gdir_dir, "mb_calib.json"), "w") as f:
        json.dump({"melt_f": 5.0, "level": level}, f)
    ds = xr.Dataset({"w": ("x", np.ones(3) * level)})
    ds.to_zarr(
        os.path.join(gdir_dir, "data_store.zarr"),
        group="model_flowlines",
        mode="a",
        zarr_format=2,
        consolidated=True,
    )
    return utils.write_level_manifest(
        gdir_dir,
        level=level,
        prev_state=prev_state,
        dataset_tag="abc123",
        requires=list(range(level)),
        border=80,
        rgi_version="62",
    )


def test_gdir_to_tar_include(tmp_path):
    rid = "RGI60-11.00897"
    gdir_dir = _make_fake_gdir(str(tmp_path / rid), extra_file="dem.tif")
    _, changed = _simulate_level(gdir_dir)
    fake = _FakeGdir(gdir_dir, str(tmp_path))

    # Delta tar: only the changed paths (+ manifest) are members
    opath = utils.gdir_to_tar.unwrapped(fake, delete=False, include=changed)
    with tarfile.open(opath, "r:gz") as tar:
        names = tar.getnames()
    files = {n for n in names if not n.endswith(rid)}
    assert all(n.startswith(rid + "/") for n in files)
    assert f"{rid}/mb_calib.json" in files
    assert f"{rid}/log.txt" in files
    assert f"{rid}/L3.manifest.json" in files
    # zarr group directory is included recursively
    assert any(
        n.startswith(f"{rid}/data_store.zarr/model_flowlines/") for n in files
    )
    # unchanged files are not shipped
    assert not any("dem.tif" in n or "diagnostics.json" in n for n in files)
    assert not any(
        n.startswith(f"{rid}/data_store.zarr/inversion_flowlines")
        for n in files
    )
    os.remove(opath)

    # include=None keeps the full-directory behavior
    opath = utils.gdir_to_tar.unwrapped(fake, delete=False)
    with tarfile.open(opath, "r:gz") as tar:
        names = tar.getnames()
    assert f"{rid}/dem.tif" in names
    assert f"{rid}/diagnostics.json" in names


class TestLayeredGdir:

    def test_glacierdirectory_from_tar_list(self, tmp_path, hef_gdir):
        rid = hef_gdir.rgi_id

        # Work on a copy of the real gdir, never on the shared fixture
        workbase = str(tmp_path / "work")
        workdir = os.path.join(workbase, rid[:-6], rid[:-3], rid)
        shutil.copytree(hef_gdir.dir, workdir)
        assert os.path.isdir(os.path.join(workdir, "data_store.zarr"))

        # Rollup artifact: everything up to L3 in one tar
        utils.write_level_manifest(
            workdir,
            level=3,
            prev_state={},
            dataset_tag="ds1",
            requires=[],
            includes_levels=[0, 1, 2, 3],
            border=80,
            rgi_version="62",
        )
        rollup_tar = utils.gdir_to_tar.unwrapped(
            _FakeGdir(workdir, workbase), delete=False
        )
        rollup_tar = shutil.move(rollup_tar, str(tmp_path / "rollup.tar.gz"))

        # L4 delta: a changed file and a new zarr group, written the way a
        # delta ships it: group subtree only, no root consolidated metadata
        prev = utils.snapshot_gdir_state(workdir)
        with open(os.path.join(workdir, "mb_calib.json"), "w") as f:
            json.dump({"melt_f": 6.0}, f)
        ds = xr.Dataset({"w": ("x_delta", np.ones(4))})
        ds.to_zarr(
            os.path.join(workdir, "data_store.zarr"),
            group="delta_check",
            mode="a",
            zarr_format=2,
            consolidated=False,
        )
        _, changed = utils.write_level_manifest(
            workdir,
            level=4,
            prev_state=prev,
            dataset_tag="ds1",
            requires=[0, 1, 2, 3],
            border=80,
            rgi_version="62",
        )
        # The delta must not ship the store's root metadata
        delta_tar = utils.gdir_to_tar.unwrapped(
            _FakeGdir(workdir, workbase), delete=False, include=changed
        )
        delta_tar = shutil.move(delta_tar, str(tmp_path / "delta.tar.gz"))

        ref_state = utils.snapshot_gdir_state(workdir)

        # Layer rollup + delta into a fresh base dir
        newbase = str(tmp_path / "layered")
        gdir = oggm.GlacierDirectory(
            rid, base_dir=newbase, from_tar=[rollup_tar, delta_tar]
        )

        assert utils.snapshot_gdir_state(gdir.dir) == ref_state
        # Both manifests document the layering
        assert os.path.isfile(os.path.join(gdir.dir, "L3.manifest.json"))
        assert os.path.isfile(os.path.join(gdir.dir, "L4.manifest.json"))
        # Consolidated metadata was rebuilt: the new group is visible
        # through the consolidated read path, and existing groups still
        # read fine
        new_group = gdir.read_zarr("delta_check", consolidated=True)
        np.testing.assert_allclose(new_group["w"].values, np.ones(4))
        assert gdir.read_store("inversion_flowlines") is not None

    def test_convert_pickles_to_zarr(self, tmp_path, hef_gdir):
        """Pickles are rewritten into the zarr store and then removed,
        with the data reading back equivalently."""
        from oggm.utils import compat

        rid = hef_gdir.rgi_id
        workbase = str(tmp_path / "work")
        workdir = os.path.join(workbase, rid[:-6], rid[:-3], rid)
        shutil.copytree(hef_gdir.dir, workdir)
        gdir = oggm.GlacierDirectory(rid, base_dir=workbase)

        # Simulate pickle-only dataset by writing back out as pickles.
        # Write_pickle drops the zarr group
        names = ["inversion_flowlines", "model_flowlines"]
        original = {n: gdir.read_store(n) for n in names}
        for n in names:
            gdir.write_pickle(original[n], n)
            assert os.path.isfile(os.path.join(gdir.dir, f"{n}.pkl"))
            assert not os.path.isdir(
                os.path.join(gdir.dir, "data_store.zarr", n)
            )

        compat._convert_pickles_to_zarr(gdir)

        for n in names:
            assert not os.path.isfile(os.path.join(gdir.dir, f"{n}.pkl"))
            assert os.path.isdir(os.path.join(gdir.dir, "data_store.zarr", n))
            assert len(gdir.read_store(n)) == len(original[n])


class TestDeltaServer:
    """End-to-end: init_glacier_directories against a delta-file server."""

    BASE_URL = "https://delta.invalid/gdirs/"

    @pytest.fixture
    def delta_server(self, tmp_path, hef_gdir):
        """A local server tree with L3 rollup, L4 delta, L5 standalone."""
        rid = hef_gdir.rgi_id
        srcbase = str(tmp_path / "src")
        workdir = os.path.join(srcbase, rid[:-6], rid[:-3], rid)
        shutil.copytree(hef_gdir.dir, workdir)
        server = tmp_path / "server"

        def publish(level, member_tar):
            region = rid[:-6]
            bundle = f"{rid[:-6]}.{rid[-5:-2]}"
            dest = server / "RGI62" / "b_080" / f"L{level}" / region
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(dest / f"{bundle}.tar"), "w") as tf:
                tf.add(member_tar, arcname=f"{bundle}/{rid}.tar.gz")
            os.remove(member_tar)

        # L3 rollup (includes 0..3)
        utils.write_level_manifest(
            workdir,
            level=3,
            prev_state={},
            dataset_tag="ds1",
            requires=[],
            includes_levels=[0, 1, 2, 3],
            border=80,
            rgi_version="62",
        )
        publish(
            3,
            utils.gdir_to_tar.unwrapped(
                _FakeGdir(workdir, srcbase), delete=False
            ),
        )

        # L4 delta (requires 0..3): changed file + new zarr group, no
        # root store metadata shipped
        prev = utils.snapshot_gdir_state(workdir)
        with open(os.path.join(workdir, "mb_calib.json"), "w") as f:
            json.dump({"melt_f": 6.0}, f)
        ds = xr.Dataset({"w": ("x_delta", np.ones(4))})
        ds.to_zarr(
            os.path.join(workdir, "data_store.zarr"),
            group="delta_check",
            mode="a",
            zarr_format=2,
            consolidated=False,
        )
        _, changed = utils.write_level_manifest(
            workdir,
            level=4,
            prev_state=prev,
            dataset_tag="ds1",
            requires=[0, 1, 2, 3],
            border=80,
            rgi_version="62",
        )
        publish(
            4,
            utils.gdir_to_tar.unwrapped(
                _FakeGdir(workdir, srcbase), delete=False, include=changed
            ),
        )

        # L5 standalone
        utils.write_level_manifest(
            workdir,
            level=5,
            prev_state={},
            requires=[],
            includes_levels=[5],
            kind="standalone",
            border=80,
            rgi_version="62",
            dataset_tag="ds1",
        )
        publish(
            5,
            utils.gdir_to_tar.unwrapped(
                _FakeGdir(workdir, srcbase), delete=False
            ),
        )

        return str(server), rid

    @pytest.fixture
    def served_calls(self, delta_server, monkeypatch, tmp_path):
        server, rid = delta_server
        calls = []

        def fake_file_downloader(www_path, **kwargs):
            calls.append(www_path)
            local = os.path.join(server, www_path.replace(self.BASE_URL, ""))
            return local if os.path.isfile(local) else None

        monkeypatch.setattr(_downloads, "file_downloader", fake_file_downloader)
        monkeypatch.setattr(_downloads, "_prepro_bundle_format", {})
        wd = str(tmp_path / "wd")
        os.makedirs(wd, exist_ok=True)
        cfg.PATHS["working_dir"] = wd
        cfg.PARAMS["has_internet"] = False
        return calls, rid

    def test_init_from_delta_server(self, served_calls):
        calls, rid = served_calls

        # Level 4 = the L4 bundle plus the L3 rollup it requires
        gdirs = workflow.init_glacier_directories(
            [rid],
            from_prepro_level=4,
            prepro_border=80,
            prepro_base_url=self.BASE_URL,
        )
        gdir = gdirs[0]
        assert len(calls) == 2
        assert "/L4/" in calls[0]
        assert "/L3/" in calls[1]
        assert os.path.isfile(os.path.join(gdir.dir, "L3.manifest.json"))
        assert os.path.isfile(os.path.join(gdir.dir, "L4.manifest.json"))
        with open(os.path.join(gdir.dir, "mb_calib.json")) as f:
            assert json.load(f)["melt_f"] == 6.0
        new_group = gdir.read_zarr("delta_check", consolidated=True)
        np.testing.assert_allclose(new_group["w"].values, np.ones(4))

        # Level 5 is standalone: one fetch only
        calls.clear()
        gdirs = workflow.init_glacier_directories(
            [rid],
            from_prepro_level=5,
            prepro_border=80,
            prepro_base_url=self.BASE_URL,
        )
        assert len(calls) == 1
        assert "/L5/" in calls[0]

    def test_append_topup(self, served_calls):
        calls, rid = served_calls

        # Start from the L3 rollup: a single fetch
        workflow.init_glacier_directories(
            [rid],
            from_prepro_level=3,
            prepro_border=80,
            prepro_base_url=self.BASE_URL,
        )
        assert len(calls) == 1
        assert "/L3/" in calls[0]

        # Top up to L4: only the L4 bundle is fetched, the existing
        # directory provides levels 0-3
        calls.clear()
        gdirs = workflow.init_glacier_directories(
            [rid],
            from_prepro_level=4,
            prepro_border=80,
            prepro_base_url=self.BASE_URL,
            append=True,
        )
        gdir = gdirs[0]
        assert len(calls) == 1
        assert "/L4/" in calls[0]
        assert os.path.isfile(os.path.join(gdir.dir, "L3.manifest.json"))
        assert os.path.isfile(os.path.join(gdir.dir, "L4.manifest.json"))
        with open(os.path.join(gdir.dir, "mb_calib.json")) as f:
            assert json.load(f)["melt_f"] == 6.0
        new_group = gdir.read_zarr("delta_check", consolidated=True)
        np.testing.assert_allclose(new_group["w"].values, np.ones(4))


L12_BASE_URL = (
    "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/"
    "L1-L2_files/2025.6/elev_bands/"
)


@pytest.mark.download
def test_convert_prepro_to_deltas(tmp_path):
    """
    The test-env allowlist only covers test data, so this test downloads
    the real (small) reference gdirs. Reverted by restore_oggm_cfg.
    """
    from oggm.utils import compat

    cfg.initialize()
    cfg.PATHS["working_dir"] = str(tmp_path / "wd")
    os.makedirs(cfg.PATHS["working_dir"], exist_ok=True)

    cfg.PARAMS["download_url_allowlist"] += [
        "cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/",
    ]

    rgi_ids = ["RGI60-11.00897", "RGI60-01.16195"]
    base_urls = {
        0: L12_BASE_URL,
        1: L12_BASE_URL,
        2: L12_BASE_URL,
        3: utils.DEFAULT_BASE_URL,
        4: utils.DEFAULT_BASE_URL,
        5: utils.DEFAULT_BASE_URL,
    }
    output_dir = str(tmp_path / "out")
    compat.convert_prepro_to_deltas(
        rgi_ids,
        base_urls,
        border=80,
        rgi_version="62",
        workdir=str(tmp_path / "conv"),
        output_dir=output_dir,
        dataset_tag="oggm_v1.6_2025.6_elev_bands_w5e5",
    )

    expected = {
        0: dict(kind="delta", includes=[0], requires=[]),
        1: dict(kind="delta", includes=[1], requires=[0]),
        2: dict(kind="delta", includes=[2], requires=[0, 1]),
        3: dict(kind="delta", includes=[0, 1, 2, 3], requires=[]),
        4: dict(kind="delta", includes=[4], requires=[0, 1, 2, 3]),
        5: dict(kind="standalone", includes=[5], requires=[]),
    }
    dataset_ids = set()
    for rid in rgi_ids:
        region = rid[:-6]
        bundle = f"{region}.{rid[-5:-2]}"
        for lvl, exp in expected.items():
            bpath = os.path.join(
                output_dir, "RGI62", "b_080", f"L{lvl}", region, f"{bundle}.tar"
            )
            assert os.path.isfile(bpath), f"missing bundle L{lvl} for {rid}"
            manifest = workflow._peek_level_manifest(bpath, rid, lvl)
            assert manifest is not None
            assert manifest["kind"] == exp["kind"]
            assert manifest["includes_levels"] == exp["includes"]
            assert manifest["requires"] == exp["requires"]
            dataset_ids.add(manifest["dataset_id"])
    # One logical dataset across both source URLs
    assert len(dataset_ids) == 1


def test_level_consistency_mismatch(tmp_path):
    rid = "RGI60-11.00897"
    base = str(tmp_path / "src")
    gdir_dir = _make_fake_gdir(os.path.join(base, rid))

    utils.write_level_manifest(
        gdir_dir,
        level=3,
        prev_state={},
        dataset_tag="ds1",
        requires=[],
        includes_levels=[0, 1, 2, 3],
        border=80,
        rgi_version="62",
    )
    rollup_tar = utils.gdir_to_tar.unwrapped(
        _FakeGdir(gdir_dir, base), delete=False
    )
    rollup_tar = shutil.move(rollup_tar, str(tmp_path / "rollup.tar.gz"))

    # A level-4 delta from a *different* dataset
    prev = utils.snapshot_gdir_state(gdir_dir)
    with open(os.path.join(gdir_dir, "mb_calib.json"), "w") as f:
        json.dump({"melt_f": 6.0}, f)
    _, changed = utils.write_level_manifest(
        gdir_dir,
        level=4,
        prev_state=prev,
        dataset_tag="OTHER",
        requires=[0, 1, 2, 3],
        border=80,
        rgi_version="62",
    )
    delta_tar = utils.gdir_to_tar.unwrapped(
        _FakeGdir(gdir_dir, base), delete=False, include=changed
    )
    delta_tar = shutil.move(delta_tar, str(tmp_path / "delta.tar.gz"))

    with pytest.raises(InvalidWorkflowError, match="dataset"):
        oggm.GlacierDirectory(
            rid,
            base_dir=str(tmp_path / "layered"),
            from_tar=[rollup_tar, delta_tar],
        )
