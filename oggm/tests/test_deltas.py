"""Tests for incremental (delta-per-level) glacier directory creation."""

import io
import json
import logging
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import shutil

import numpy as np
import pytest

import oggm
from oggm import cfg, utils, workflow
from oggm.exceptions import InvalidWorkflowError
from oggm.utils import _downloads
from oggm.utils.transcoder import encode_npz

pytestmark = pytest.mark.test_env("utils")


def _create_npz_data(name: str = "flux"):
    """Minimal stand-in for an npz data store.

    Contains a single npz file similar to an inversion flowline's flux.
    """
    data = np.array(
        4000 + np.sin(np.linspace(0, np.pi, 20) * 10e5), dtype=np.float64
    )
    arrays, meta = encode_npz(data, name)
    yield arrays, meta


@pytest.fixture(name="npz_data", scope="function")
def fixture_npz_data():
    """Fixture for a minimal npz data store."""
    yield from _create_npz_data()


def _write_npz_store(
    path: str | Path,
    arrays: dict,
    meta: dict,
    name: str = "inversion_flowlines",
):
    """Write a minimal npz data store to disk."""

    store = Path(path) / "data_store"
    store.mkdir(parents=True, exist_ok=True)
    # avoid double suffixes
    fp = store / f"{name.removesuffix(".npz")}.npz"
    tmp_fp = f"{fp}.tmp{os.getpid()}"

    with open(tmp_fp, "wb") as f:
        np.savez(
            f,
            **arrays,
            __meta__=json.dumps(meta),
            allow_pickle=False,
        )
    os.replace(tmp_fp, fp)
    return Path(store)


def _make_fake_gdir(path, extra_file=None):
    """A minimal on-disk stand-in for a glacier directory.

    Contains regular files plus a data_store with an npz, in a structure
    readable by `snapshot_gdir_state`.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(path, "diagnostics.json"), "w") as f:
        json.dump({"a": 1}, f)
    with open(os.path.join(path, "log.txt"), "w") as f:
        f.write("task log\n")
    if extra_file:
        with open(os.path.join(path, extra_file), "w") as f:
            f.write("data\n")
    # leaving this here in case we want to construct a linestring
    # line_data = np.array(np.arange(20).reshape((10, 2)), dtype=np.float64)
    # line_data[:, 1] = 0  # replace second coordinate with zeros
    # match flux in inversion flowlines
    arrays, meta = next(_create_npz_data())
    _write_npz_store(path, arrays, meta)
    return Path(path)


def test_make_fake_dir(tmp_path):
    # use pathlib instead of tmp_path to avoid pytest tmp_path cleanup
    gdir_dir = _make_fake_gdir(tmp_path / "test_gdir", extra_file="dem.tif")
    assert gdir_dir.is_dir()
    assert (gdir_dir / "diagnostics.json").is_file()
    assert (gdir_dir / "log.txt").is_file()
    assert (gdir_dir / "dem.tif").is_file()
    assert (gdir_dir / "data_store").is_dir()
    assert (gdir_dir / "data_store" / "inversion_flowlines.npz").is_file()
    test_npz = np.load(gdir_dir / "data_store" / "inversion_flowlines.npz")
    assert "flux" in test_npz
    test_flux = test_npz["flux"]
    assert test_flux.shape == (20,)
    np.testing.assert_array_less(0.0, test_flux)


def test_snapshot_gdir_state(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))

    state = utils.snapshot_gdir_state(gdir_dir)

    # Regular files are keyed by relative path
    assert "diagnostics.json" in state
    assert "log.txt" in state
    group_key = "data_store/inversion_flowlines.npz"
    assert group_key in state
    assert "data_store" not in state

    # Unchanged directory -> identical snapshot
    assert utils.snapshot_gdir_state(gdir_dir) == state

    # Content change is detected, everything else is stable
    with open(os.path.join(gdir_dir, "diagnostics.json"), "w") as f:
        json.dump({"a": 2}, f)
    new_state = utils.snapshot_gdir_state(gdir_dir)
    assert new_state["diagnostics.json"] != state["diagnostics.json"]
    assert new_state["log.txt"] == state["log.txt"]
    assert new_state[group_key] == state[group_key]

    # Adding an npz group gives a new key, the existing group is unchanged
    store = gdir_dir / "data_store"
    np.savez(store / "model_flowlines.npz", w=np.ones(3), allow_pickle=False)
    grown = utils.snapshot_gdir_state(gdir_dir)
    assert "data_store/model_flowlines.npz" in grown
    assert grown[group_key] == state[group_key]

    # Rewriting a group changes its digest
    np.savez(
        store / "inversion_flowlines.npz", w=np.zeros(3), allow_pickle=False
    )
    rewritten = utils.snapshot_gdir_state(gdir_dir)
    assert rewritten[group_key] != state[group_key]

    # ensure partial npz left by interrupted write_npz is never snapshotted
    (store / "x.npz.tmp123").write_bytes(b"partial")
    assert "data_store/x.npz.tmp123" not in utils.snapshot_gdir_state(gdir_dir)


def test_write_level_manifest_schema(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    prev_state = utils.snapshot_gdir_state(gdir_dir)

    # Simulate a level's work: one updated file, one new file, one new
    # npz store

    with open(os.path.join(gdir_dir, "log.txt"), "a") as f:
        f.write("more work\n")
    with open(os.path.join(gdir_dir, "mb_calib.json"), "w") as f:
        json.dump({"melt_f": 5.0}, f)

    np.savez(
        gdir_dir / "data_store" / "model_flowlines",
        w=np.ones(3),
        allow_pickle=False,
    )
    assert (gdir_dir / "data_store" / "model_flowlines.npz").is_file()

    manifest_path, changed = utils.write_level_manifest(
        gdir_dir,
        level=3,
        prev_state=prev_state,
        artefact_tag="abc123",
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
    assert manifest["artefact_tag"] == "abc123"
    assert manifest["border"] == 80
    assert manifest["rgi_version"] == "62"
    assert manifest["oggm_version"]
    assert manifest["created"]
    assert manifest["files"]["added"] == ["mb_calib.json"]
    assert manifest["files"]["updated"] == ["log.txt"]
    assert manifest["data_store"] == ["model_flowlines.npz"]

    # changed_paths is what gdir_to_tar(include=...) needs: the changed
    # files, the changed store groups, and the manifest itself
    assert set(changed) == {
        "mb_calib.json",
        "log.txt",
        "data_store/model_flowlines.npz",
        "L3.manifest.json",
    }


@pytest.mark.parametrize(
    "level, requires, includes, kind, expected",
    [
        (3, [], [0, 1, 2, 3], "delta", "materialisation"),
        (3, [], [0, 1, 2, 3], "materialisation", "materialisation"),
        (0, [], [0], "delta", "delta"),
        (4, [0, 1, 2, 3], None, "delta", "delta"),
        (5, [], [5], "standalone", "standalone"),
    ],
)
def test_write_level_manifest_kind(
    tmp_path, level, requires, includes, kind, expected
):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    manifest_path, _ = utils.write_level_manifest(
        gdir_dir,
        level=level,
        prev_state={},
        artefact_tag="abc123",
        requires=requires,
        includes_levels=includes,
        kind=kind,
        border=80,
        rgi_version="62",
    )
    with open(manifest_path) as f:
        assert json.load(f)["kind"] == expected


@pytest.mark.parametrize(
    "requires, includes, kind",
    [
        ([0, 1, 2], [0, 1, 2, 3], "materialisation"),
        ([], [3], "materialisation"),
        ([], [3], "delta"),
    ],
)
def test_write_level_manifest_invalid_kind(tmp_path, requires, includes, kind):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    with pytest.raises(ValueError, match="materialisation"):
        utils.write_level_manifest(
            gdir_dir,
            level=3,
            prev_state={},
            artefact_tag="abc123",
            requires=requires,
            includes_levels=includes,
            kind=kind,
            border=80,
            rgi_version="62",
        )


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
    arrays, meta = next(_create_npz_data(name="w"))
    arrays["w"] = arrays["w"] * level  # make it different per level
    _write_npz_store(gdir_dir, arrays, meta, name="model_flowlines")
    return utils.write_level_manifest(
        gdir_dir,
        level=level,
        prev_state=prev_state,
        artefact_tag="abc123",
        requires=list(range(level)),
        border=80,
        rgi_version="62",
    )


def test_simulate_level(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    manifest_path, changed = _simulate_level(gdir_dir, level=3)
    assert os.path.basename(manifest_path) == "L3.manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["level"] == 3
    assert set(changed) == {
        "mb_calib.json",
        "log.txt",
        "data_store/model_flowlines.npz",
        "L3.manifest.json",
    }


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
    # npz group is included
    assert any(n.startswith(f"{rid}/data_store/model_flowlines") for n in files)
    # unchanged files are not shipped
    assert not any("dem.tif" in n or "diagnostics.json" in n for n in files)
    assert not any(
        n.startswith(f"{rid}/data_store/inversion_flowlines") for n in files
    )
    os.remove(opath)

    # include=None keeps the full-directory behavior
    opath = utils.gdir_to_tar.unwrapped(fake, delete=False)
    with tarfile.open(opath, "r:gz") as tar:
        names = tar.getnames()
    assert f"{rid}/dem.tif" in names
    assert f"{rid}/diagnostics.json" in names


def _build_bundle(path, rid, with_manifest=None):
    """Write a bundle-in-tar (bundle/<rid>.tar.gz) as the downloads look."""
    inner_buf = io.BytesIO()
    with tarfile.open(fileobj=inner_buf, mode="w:gz") as inner:
        payload = b'{"a": 1}\n'
        ti = tarfile.TarInfo(f"{rid}/diagnostics.json")
        ti.size = len(payload)
        inner.addfile(ti, io.BytesIO(payload))
        if with_manifest is not None:
            mf = json.dumps(with_manifest).encode()
            level = with_manifest["level"]
            ti = tarfile.TarInfo(f"{rid}/L{level}.manifest.json")
            ti.size = len(mf)
            inner.addfile(ti, io.BytesIO(mf))
        filler = os.urandom(40000)  # large enough to truncate into
        ti = tarfile.TarInfo(f"{rid}/filler.bin")
        ti.size = len(filler)
        inner.addfile(ti, io.BytesIO(filler))
    inner_bytes = inner_buf.getvalue()

    bundle = f"{rid[:-6]}.{rid[-5:-2]}"
    with tarfile.open(path, "w") as tf:
        ti = tarfile.TarInfo(f"{bundle}/{rid}.tar.gz")
        ti.size = len(inner_bytes)
        tf.addfile(ti, io.BytesIO(inner_bytes))


def test_peek_level_manifest_corruption(tmp_path):
    rid = "RGI60-11.00897"
    bundle = f"{rid[:-6]}.{rid[-5:-2]}"
    tar_base = str(tmp_path / f"{bundle}.tar")

    # Legacy bundle without a manifest should return None
    _build_bundle(tar_base, rid)
    assert workflow._peek_level_manifest(tar_base, rid, 1) is None

    # Delta bundle carrying a manifest should be parsed
    manifest = {"level": 1, "kind": "delta", "requires": [0]}
    _build_bundle(tar_base, rid, with_manifest=manifest)
    got = workflow._peek_level_manifest(tar_base, rid, 1)
    assert got == manifest

    # Truncated or corrupt bundle raises ReadError mid-stream.
    # This should return None to avoid poisoning.
    size = os.path.getsize(tar_base)
    with open(tar_base, "r+b") as f:
        f.truncate(size - 20000)
    with pytest.raises(tarfile.ReadError):
        with tarfile.open(tar_base, "r") as tf:
            tf.getmembers()
    assert workflow._peek_level_manifest(tar_base, rid, 1) is None


class TestLayeredGdir:

    def test_glacierdirectory_from_tar_list(self, tmp_path, hef_gdir):
        rid = hef_gdir.rgi_id

        # Work on a copy of the real gdir, never on the shared fixture
        workbase = str(tmp_path / "work")
        workdir = os.path.join(workbase, rid[:-6], rid[:-3], rid)
        shutil.copytree(hef_gdir.dir, workdir)
        assert os.path.isdir(os.path.join(workdir, "data_store"))

        # Materialisation artifact: everything up to L3 in one tar
        utils.write_level_manifest(
            workdir,
            level=3,
            prev_state={},
            artefact_tag="ds1",
            requires=[],
            includes_levels=[0, 1, 2, 3],
            border=80,
            rgi_version="62",
        )
        materialisation_tar = utils.gdir_to_tar.unwrapped(
            _FakeGdir(workdir, workbase), delete=False
        )
        materialisation_tar = shutil.move(
            materialisation_tar, str(tmp_path / "materialisation.tar.gz")
        )

        # L4 delta: a changed file and a new npz group
        prev = utils.snapshot_gdir_state(workdir)
        with open(os.path.join(workdir, "mb_calib.json"), "w") as f:
            json.dump({"melt_f": 6.0}, f)
        arrays, meta = encode_npz(np.ones(4), "delta_check")
        _write_npz_store(workdir, arrays, meta, name="delta_check")
        _, changed = utils.write_level_manifest(
            workdir,
            level=4,
            prev_state=prev,
            artefact_tag="ds1",
            requires=[0, 1, 2, 3],
            border=80,
            rgi_version="62",
        )
        delta_tar = utils.gdir_to_tar.unwrapped(
            _FakeGdir(workdir, workbase), delete=False, include=changed
        )
        delta_tar = shutil.move(delta_tar, str(tmp_path / "delta.tar.gz"))

        ref_state = utils.snapshot_gdir_state(workdir)

        # Layer materialisation + delta into a fresh base dir
        newbase = str(tmp_path / "layered")
        gdir = oggm.GlacierDirectory(
            rid, base_dir=newbase, from_tar=[materialisation_tar, delta_tar]
        )

        assert utils.snapshot_gdir_state(gdir.dir) == ref_state
        # Both manifests document the layering
        assert os.path.isfile(os.path.join(gdir.dir, "L3.manifest.json"))
        assert os.path.isfile(os.path.join(gdir.dir, "L4.manifest.json"))
        np.testing.assert_allclose(gdir.read_npz("delta_check"), np.ones(4))
        assert gdir.read_store("inversion_flowlines") is not None

    @pytest.mark.parametrize("arg_delete", [True, False])
    def test_convert_pickles_to_npz(self, tmp_path, hef_gdir, arg_delete):
        """Pickles are rewritten into the npz store and then removed,
        with the data reading back equivalently."""
        from oggm.utils import _compat

        rid = hef_gdir.rgi_id
        workbase = Path(tmp_path / "work")
        # workdir = os.path.join(workbase, rid[:-6], rid[:-3], rid)
        workdir = workbase / rid[:-6] / rid[:-3] / rid

        # we delete pickles in this test, so work on copy
        shutil.copytree(hef_gdir.dir, workdir)
        gdir = oggm.GlacierDirectory(rid, base_dir=workbase)

        # Simulate pickle-only dataset by writing back out as pickles.
        # Write_pickle drops the npz group
        names = ["inversion_flowlines", "model_flowlines"]
        original = {n: gdir.read_store(n) for n in names}
        for n in names:
            gdir._write_pickle(original[n], n)
            assert Path(gdir.dir, f"{n}.pkl").is_file()
            assert not Path(gdir.dir, "data_store", n).is_dir()

        _compat.convert_pickles_to_npz(gdir, delete=arg_delete)

        # check the pickles are gone
        for n in names:
            assert Path(gdir.dir, "data_store").is_dir()
            assert Path(gdir.dir, "data_store", f"{n}.npz").is_file()
            if arg_delete:
                assert not Path(gdir.dir, f"{n}.pkl").is_file()
            else:
                assert Path(gdir.dir, f"{n}.pkl").is_file()

            assert len(gdir.read_store(n)) == len(original[n])


class TestDeltaServer:
    """End-to-end: init_glacier_directories against a delta-file server."""

    BASE_URL = "https://delta.invalid/gdirs/"

    @pytest.fixture
    def delta_server(self, tmp_path, hef_gdir):
        """A local server tree with L3 materialisation, L4 delta, L5 standalone."""
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

        # L3 materialisation (includes 0..3)
        utils.write_level_manifest(
            workdir,
            level=3,
            prev_state={},
            artefact_tag="ds1",
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

        # L4 delta (requires 0..3): changed file + new npz group
        prev = utils.snapshot_gdir_state(workdir)
        with open(os.path.join(workdir, "mb_calib.json"), "w") as f:
            json.dump({"melt_f": 6.0}, f)
        arrays, meta = encode_npz(np.ones(4), "delta_check")
        _write_npz_store(workdir, arrays, meta, name="delta_check")
        _, changed = utils.write_level_manifest(
            workdir,
            level=4,
            prev_state=prev,
            artefact_tag="ds1",
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
            artefact_tag="ds1",
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

        # Level 4 = the L4 bundle plus the L3 materialisation it requires
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
        np.testing.assert_allclose(gdir.read_npz("delta_check"), np.ones(4))

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

        # Start from the L3 materialisation: a single fetch
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
        np.testing.assert_allclose(gdir.read_npz("delta_check"), np.ones(4))

    def test_init_from_local_delta_tree(self, delta_server, served_calls):
        """Test initialization from a local delta tree.

        Server tree is read from disk, so the L4 delta is missing the
        the grid from the L3 tar.
        """
        server, _ = delta_server
        calls, rid = served_calls
        gdirs = workflow.init_glacier_directories(
            [rid], from_tar=os.path.join(server, "RGI62", "b_080", "L4")
        )
        gdir = gdirs[0]
        assert calls == []
        assert os.path.isfile(os.path.join(gdir.dir, "L3.manifest.json"))
        assert os.path.isfile(os.path.join(gdir.dir, "L4.manifest.json"))
        assert gdir.grid.nx > 0
        with open(os.path.join(gdir.dir, "mb_calib.json")) as f:
            assert json.load(f)["melt_f"] == 6.0

    @pytest.fixture
    def local_l4_only(self, delta_server, tmp_path):
        """A local tree with the L4 delta but none of the levels it requires."""
        server, _ = delta_server
        local = tmp_path / "local" / "RGI62" / "b_080" / "L4"
        shutil.copytree(Path(server, "RGI62", "b_080", "L4"), local)
        return str(local)

    def test_init_from_local_tree_fetches_missing_level(
        self, served_calls, local_l4_only
    ):
        calls, rid = served_calls
        gdirs = workflow.init_glacier_directories(
            [rid],
            from_tar=local_l4_only,
            prepro_base_url=self.BASE_URL,
            prepro_border=80,
        )
        gdir = gdirs[0]
        assert len(calls) == 1
        assert "/L3/" in calls[0]
        assert Path(gdir.dir, "L3.manifest.json").is_file()
        assert Path(gdir.dir, "L4.manifest.json").is_file()
        assert gdir.grid.nx > 0
        mb_calib = json.loads(Path(gdir.dir, "mb_calib.json").read_text())
        assert mb_calib["melt_f"] == 6.0

    def test_local_tree_missing_level_without_base_url(
        self, served_calls, local_l4_only
    ):
        calls, rid = served_calls
        with pytest.raises(FileNotFoundError):
            workflow.gdir_from_tar(rid, local_l4_only)
        assert calls == []


class TestStartState:
    """`_start_state` at a `3a` resume, with and without the L2 state."""

    @pytest.fixture
    def gdir(self, tmp_path):
        return SimpleNamespace(dir=str(tmp_path), rgi_id="RGI60-11.00897")

    def test_missing_l2_state_is_delta(self, gdir, caplog):
        from oggm.cli.prepro_levels import _start_state

        with caplog.at_level(logging.WARNING, logger="oggm.cli.prepro_levels"):
            assert _start_state(gdir, True) is None
        assert "L2.state.json" in caplog.text

    def test_l2_state_is_loaded_and_removed(self, gdir):
        from oggm.cli.prepro_levels import _start_state

        fp = Path(gdir.dir, "L2.state.json")
        fp.write_text(json.dumps({"a.txt": "abc"}))
        assert _start_state(gdir, True) == {"a.txt": "abc"}
        assert not fp.exists()


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
        artefact_tag="oggm_v1.6_2025.6_elev_bands_w5e5",
    )

    expected = {
        0: dict(kind="delta", includes=[0], requires=[]),
        1: dict(kind="delta", includes=[1], requires=[0]),
        2: dict(kind="delta", includes=[2], requires=[0, 1]),
        3: dict(kind="materialisation", includes=[0, 1, 2, 3], requires=[]),
        4: dict(kind="delta", includes=[4], requires=[0, 1, 2, 3]),
        5: dict(kind="standalone", includes=[5], requires=[]),
    }
    artefact_ids = set()
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
            artefact_ids.add(manifest["artefact_id"])
    # One logical dataset across both source URLs
    assert len(artefact_ids) == 1


def test_level_consistency_mismatch(tmp_path):
    rid = "RGI60-11.00897"
    base = str(tmp_path / "src")
    gdir_dir = _make_fake_gdir(os.path.join(base, rid))

    utils.write_level_manifest(
        gdir_dir,
        level=3,
        prev_state={},
        artefact_tag="ds1",
        requires=[],
        includes_levels=[0, 1, 2, 3],
        border=80,
        rgi_version="62",
    )
    materialisation_tar = utils.gdir_to_tar.unwrapped(
        _FakeGdir(gdir_dir, base), delete=False
    )
    materialisation_tar = shutil.move(
        materialisation_tar, str(tmp_path / "materialisation.tar.gz")
    )

    # A level-4 delta from a *different* dataset
    prev = utils.snapshot_gdir_state(gdir_dir)
    with open(os.path.join(gdir_dir, "mb_calib.json"), "w") as f:
        json.dump({"melt_f": 6.0}, f)
    _, changed = utils.write_level_manifest(
        gdir_dir,
        level=4,
        prev_state=prev,
        artefact_tag="OTHER",
        requires=[0, 1, 2, 3],
        border=80,
        rgi_version="62",
    )
    delta_tar = utils.gdir_to_tar.unwrapped(
        _FakeGdir(gdir_dir, base), delete=False, include=changed
    )
    delta_tar = shutil.move(delta_tar, str(tmp_path / "delta.tar.gz"))

    with pytest.raises(InvalidWorkflowError, match="different artefacts"):
        oggm.GlacierDirectory(
            rid,
            base_dir=str(tmp_path / "layered"),
            from_tar=[materialisation_tar, delta_tar],
        )
