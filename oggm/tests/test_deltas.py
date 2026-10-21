"""Tests for incremental (delta-per-level) glacier directory creation."""

import io
import json
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
from oggm.utils import _compat, _downloads
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


def _write_npz_store(
    path: str | Path,
    arrays: dict,
    meta: dict,
    name: str = "inversion_flowlines",
):
    """Write a minimal npz data store to disk."""

    store = Path(path)
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

    Contains regular files plus an npz store group, in a structure
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
    assert (gdir_dir / "inversion_flowlines.npz").is_file()
    test_npz = np.load(gdir_dir / "inversion_flowlines.npz")
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
    group_key = "inversion_flowlines.npz"
    assert group_key in state

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
    store = gdir_dir
    np.savez(store / "model_flowlines.npz", w=np.ones(3), allow_pickle=False)
    grown = utils.snapshot_gdir_state(gdir_dir)
    assert "model_flowlines.npz" in grown
    assert grown[group_key] == state[group_key]

    # Rewriting a group changes its digest
    np.savez(
        store / "inversion_flowlines.npz", w=np.zeros(3), allow_pickle=False
    )
    rewritten = utils.snapshot_gdir_state(gdir_dir)
    assert rewritten[group_key] != state[group_key]

    # ensure partial npz left by interrupted write_npz is never snapshotted
    (store / "x.npz.tmp123").write_bytes(b"partial")
    assert "x.npz.tmp123" not in utils.snapshot_gdir_state(gdir_dir)


def _write_parent_manifest(gdir_dir, label="L2", mid="parent-l2-id"):
    """A stand-in parent manifest: the writer only reads its ``id``."""
    with open(os.path.join(gdir_dir, f"{label}.manifest.json"), "w") as f:
        json.dump({"label": label, "id": mid}, f)


def test_write_level_manifest_schema(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    _write_parent_manifest(gdir_dir)
    prev_state = utils.snapshot_gdir_state(gdir_dir)

    # Simulate a level's work: one updated file, one new file, one new
    # npz store

    with open(os.path.join(gdir_dir, "log.txt"), "a") as f:
        f.write("more work\n")
    with open(os.path.join(gdir_dir, "mb_calib.json"), "w") as f:
        json.dump({"melt_f": 5.0}, f)

    np.savez(
        gdir_dir / "model_flowlines",
        w=np.ones(3),
        allow_pickle=False,
    )
    assert (gdir_dir / "model_flowlines.npz").is_file()

    manifest_path, changed = utils.write_level_manifest(
        gdir_dir,
        label="L3",
        prev_state=prev_state,
        parent="L2",
        parent_base_url="https://example.invalid/L1-L2/",
        border=80,
        rgi_version="62",
    )

    assert os.path.basename(manifest_path) == "L3.manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    state = utils.snapshot_gdir_state(gdir_dir)
    # this needs to be updated whenever the schema changes
    assert manifest["schema_version"] == "1.0"
    assert manifest["kind"] == "delta"
    assert manifest["rgi_id"] == "RGI60-11.00897"
    assert manifest["label"] == "L3"
    assert manifest["parent"] == {
        "base_url": "https://example.invalid/L1-L2/",
        "label": "L2",
        "id": "parent-l2-id",
    }
    assert manifest["border"] == 80
    assert manifest["rgi_version"] == "62"
    assert manifest["oggm_version"]
    assert manifest["created"]
    assert len(manifest["id"]) == 64
    # store groups are plain files, with no section of their own
    assert manifest["files"]["added"] == {
        "mb_calib.json": state["mb_calib.json"],
        "model_flowlines.npz": state["model_flowlines.npz"],
    }
    assert manifest["files"]["updated"] == {"log.txt": state["log.txt"]}
    assert manifest["files"]["removed"] == []

    # changed_paths is what gdir_to_tar(include=...) needs: the changed
    # files, the changed store groups, and the manifest itself
    assert set(changed) == {
        "mb_calib.json",
        "log.txt",
        "model_flowlines.npz",
        "L3.manifest.json",
    }


# Dataset keys, since these tests run without cfg.initialize()
_DS = dict(border=80, rgi_version="62")


def _manifest(path):
    with open(path) as f:
        return json.load(f)


def test_write_level_manifest_parent(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))

    # The parent's id comes from its manifest on disk, which must exist
    with pytest.raises(FileNotFoundError):
        utils.write_level_manifest(gdir_dir, "L3", {}, parent="L2", **_DS)

    # parent_dir points at a directory holding the parent's manifest,
    # e.g. the full gdir behind an L5 mini directory
    other = tmp_path / "full_gdir"
    other.mkdir()
    _write_parent_manifest(other, "L4", "l4-id")
    path, _ = utils.write_level_manifest(
        gdir_dir,
        "L5",
        {},
        parent="L4",
        parent_dir=str(other),
        kind="standalone",
        **_DS,
    )
    assert _manifest(path)["parent"] == {
        "base_url": None,
        "label": "L4",
        "id": "l4-id",
    }


def test_write_level_manifest_removed(tmp_path):
    gdir_dir = _make_fake_gdir(
        str(tmp_path / "RGI60-11.00897"), extra_file="x.txt"
    )
    _write_parent_manifest(gdir_dir)
    prev_state = utils.snapshot_gdir_state(gdir_dir)
    os.remove(gdir_dir / "x.txt")
    os.remove(gdir_dir / "inversion_flowlines.npz")
    os.remove(gdir_dir / "L2.manifest.json")  # manifests are never diffed

    _write_parent_manifest(gdir_dir)
    path, changed = utils.write_level_manifest(
        gdir_dir, "L3", prev_state, parent="L2", **_DS
    )
    files = _manifest(path)["files"]
    assert files["removed"] == ["inversion_flowlines.npz", "x.txt"]
    assert files["added"] == {} and files["updated"] == {}
    assert changed == ["L3.manifest.json"]


def test_manifest_id(tmp_path):
    def build(name, border=80, parent_id="parent-l2-id", url=None, extra=""):
        gdir_dir = _make_fake_gdir(str(tmp_path / name / "RGI60-11.00897"))
        _write_parent_manifest(gdir_dir, mid=parent_id)
        prev = utils.snapshot_gdir_state(gdir_dir)
        with open(gdir_dir / "mb_calib.json", "w") as f:
            f.write("melt_f" + extra)
        path, _ = utils.write_level_manifest(
            gdir_dir,
            "L3",
            prev,
            parent="L2",
            parent_base_url=url,
            border=border,
            rgi_version="62",
        )
        return _manifest(path)

    ref = build("ref")
    assert ref["id"] == utils.manifest_id(ref)
    # Same content, another build time and another mirror: same id
    same = build("same", url="https://mirror.invalid/")
    assert same["id"] == ref["id"]
    # Content, map border and the parent's id all change it
    assert build("content", extra="!")["id"] != ref["id"]
    assert build("border", border=40)["id"] != ref["id"]
    assert build("parent", parent_id="regenerated")["id"] != ref["id"]


def test_write_level_manifest_standalone_ships_every_file(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    _write_parent_manifest(gdir_dir, "L4")
    prev_state = utils.snapshot_gdir_state(gdir_dir)
    path, changed = utils.write_level_manifest(
        gdir_dir, "L5", prev_state, parent="L4", kind="standalone", **_DS
    )
    m = _manifest(path)
    assert m["kind"] == "standalone"
    assert set(m["files"]["added"]) == {
        "diagnostics.json",
        "log.txt",
        "inversion_flowlines.npz",
    }
    assert m["parent"]["label"] == "L4"
    assert "L4.manifest.json" not in changed


@pytest.mark.parametrize(
    "label, parent, kind",
    [
        ("L3", None, "delta"),  # only the root delta has no parent
        ("L3", None, "unknown"),
    ],
)
def test_write_level_manifest_invalid_kind(tmp_path, label, parent, kind):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    _write_parent_manifest(gdir_dir)
    with pytest.raises(ValueError, match="parent|kind"):
        utils.write_level_manifest(
            gdir_dir, label, {}, parent=parent, kind=kind, **_DS
        )


def test_write_level_manifest_root_and_free_label(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    path, _ = utils.write_level_manifest(gdir_dir, "L1", {}, **_DS)
    assert _manifest(path)["parent"] is None

    # A free-form label is its own manifest name and is never diffed
    prev = utils.snapshot_gdir_state(gdir_dir)
    with open(gdir_dir / "mb_calib.json", "w") as f:
        f.write("{}")
    path, changed = utils.write_level_manifest(
        gdir_dir, "L3a", prev, parent="L1", **_DS
    )
    assert os.path.basename(path) == "L3a.manifest.json"
    assert set(changed) == {"mb_calib.json", "L3a.manifest.json"}
    path, _ = utils.write_level_manifest(
        gdir_dir, "L3", utils.snapshot_gdir_state(gdir_dir), parent="L3a", **_DS
    )
    files = _manifest(path)["files"]
    assert files == {"added": {}, "updated": {}, "removed": []}


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
    _write_parent_manifest(gdir_dir, f"L{level - 1}")
    return utils.write_level_manifest(
        gdir_dir,
        label=f"L{level}",
        prev_state=prev_state,
        parent=f"L{level - 1}",
        **_DS,
    )


def test_simulate_level(tmp_path):
    gdir_dir = _make_fake_gdir(str(tmp_path / "RGI60-11.00897"))
    manifest_path, changed = _simulate_level(gdir_dir, level=3)
    assert os.path.basename(manifest_path) == "L3.manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["label"] == "L3"
    assert manifest["parent"]["label"] == "L2"
    assert set(changed) == {
        "mb_calib.json",
        "log.txt",
        "model_flowlines.npz",
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
    assert f"{rid}/model_flowlines.npz" in files
    # unchanged files are not shipped
    assert not any("dem.tif" in n or "diagnostics.json" in n for n in files)
    assert f"{rid}/inversion_flowlines.npz" not in files
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
            label = with_manifest["label"]
            ti = tarfile.TarInfo(f"{rid}/{label}.manifest.json")
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
    assert workflow._peek_level_manifest(tar_base, rid, "L2") is None

    # Delta bundle carrying a manifest should be parsed
    manifest = {"label": "L2", "kind": "delta", "parent": {"label": "L1"}}
    _build_bundle(tar_base, rid, with_manifest=manifest)
    got = workflow._peek_level_manifest(tar_base, rid, "L2")
    assert got == manifest
    # Free-form labels are keyed by name
    _build_bundle(tar_base, rid, with_manifest={**manifest, "label": "L3a"})
    assert workflow._peek_level_manifest(tar_base, rid, "L3a")["label"] == "L3a"

    # Truncated or corrupt bundle raises ReadError mid-stream.
    # This should return None to avoid poisoning.
    size = os.path.getsize(tar_base)
    with open(tar_base, "r+b") as f:
        f.truncate(size - 20000)
    with pytest.raises(tarfile.ReadError):
        with tarfile.open(tar_base, "r") as tf:
            tf.getmembers()
    assert workflow._peek_level_manifest(tar_base, rid, "L3a") is None


class TestLayeredGdir:

    def test_glacierdirectory_from_tar_list(self, tmp_path, hef_gdir):
        rid = hef_gdir.rgi_id

        # Work on a copy of the real gdir, never on the shared fixture
        workbase = str(tmp_path / "work")
        workdir = os.path.join(workbase, rid[:-6], rid[:-3], rid)
        shutil.copytree(hef_gdir.dir, workdir)
        assert os.path.isfile(os.path.join(workdir, "inversion_flowlines.npz"))

        # L1 root: every file in one tar
        utils.write_level_manifest(workdir, "L1", {}, **_DS)
        root_tar = utils.gdir_to_tar.unwrapped(
            _FakeGdir(workdir, workbase), delete=False
        )
        root_tar = shutil.move(root_tar, str(tmp_path / "root.tar.gz"))

        # L2 delta: a changed file and a new npz group
        prev = utils.snapshot_gdir_state(workdir)
        with open(os.path.join(workdir, "mb_calib.json"), "w") as f:
            json.dump({"melt_f": 6.0}, f)
        arrays, meta = encode_npz(np.ones(4), "delta_check")
        _write_npz_store(workdir, arrays, meta, name="delta_check")
        _, changed = utils.write_level_manifest(
            workdir, "L2", prev, parent="L1", **_DS
        )
        delta_tar = utils.gdir_to_tar.unwrapped(
            _FakeGdir(workdir, workbase), delete=False, include=changed
        )
        delta_tar = shutil.move(delta_tar, str(tmp_path / "delta.tar.gz"))

        ref_state = utils.snapshot_gdir_state(workdir)

        # Layer root + delta into a fresh base dir
        newbase = str(tmp_path / "layered")
        gdir = oggm.GlacierDirectory(
            rid, base_dir=newbase, from_tar=[root_tar, delta_tar]
        )

        assert utils.snapshot_gdir_state(gdir.dir) == ref_state
        # Both manifests document the layering
        assert os.path.isfile(os.path.join(gdir.dir, "L1.manifest.json"))
        assert os.path.isfile(os.path.join(gdir.dir, "L2.manifest.json"))
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
            assert not Path(gdir.dir, f"{n}.npz").exists()

        _compat.convert_pickles_to_npz(gdir, delete=arg_delete)

        # check the pickles are gone
        for n in names:
            assert Path(gdir.dir, f"{n}.npz").is_file()
            if arg_delete:
                assert not Path(gdir.dir, f"{n}.pkl").is_file()
            else:
                assert Path(gdir.dir, f"{n}.pkl").is_file()

            assert len(gdir.read_store(n)) == len(original[n])

    def test_convert_pickles_to_npz_suffixes(self, tmp_path, hef_gdir):
        """Each suffixed pickle maps to its own group, and an npz sharing
        the stem is never read as a pickle."""
        from oggm.utils import _compat

        rid = hef_gdir.rgi_id
        workbase = Path(tmp_path / "work")
        shutil.copytree(hef_gdir.dir, workbase / rid[:-6] / rid[:-3] / rid)
        gdir = oggm.GlacierDirectory(rid, base_dir=workbase)

        fls = gdir.read_store("model_flowlines")
        assert len(fls) > 1
        gdir._write_pickle(fls, "model_flowlines")
        gdir._write_pickle(
            fls[-1:], "model_flowlines", filesuffix="_historical"
        )
        # a stale group beside the pickle, holding something else
        np.savez(Path(gdir.dir, "model_flowlines.npz"), v=-np.ones(2))

        _compat.convert_pickles_to_npz(gdir)

        npz = sorted(p.name for p in Path(gdir.dir).glob("model_flowlines*"))
        assert npz == ["model_flowlines.npz", "model_flowlines_historical.npz"]
        assert len(gdir.read_store("model_flowlines")) == len(fls)
        assert len(gdir.read_store("model_flowlines", "_historical")) == 1


def _assert_matches_build(loaded, built):
    """A loaded gdir equals its build, except for the `fetched_from` stamps.

    Since this uses tmp paths, the stamps keep changing so we can't
    really check the `fetched_from` values.
    """

    def split(d):
        state = utils.snapshot_gdir_state(str(d))
        manifests = {}
        for k in [k for k in state if k.endswith(".manifest.json")]:
            del state[k]
            m = json.loads(Path(d, k).read_text())
            m.pop("fetched_from", None)
            manifests[k] = m
        return state, manifests

    assert split(loaded) == split(built)


class TestDeltaServer:
    """End-to-end: init_glacier_directories against a delta-file server."""

    BASE_URL = "https://delta.invalid/gdirs/"

    @pytest.fixture
    def delta_server(self, tmp_path, hef_gdir):
        """A local server tree with an L1 root, L2 delta and L5 standalone."""
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

        # L1 root: every file
        utils.write_level_manifest(workdir, "L1", {}, **_DS)
        publish(
            1,
            utils.gdir_to_tar.unwrapped(
                _FakeGdir(workdir, srcbase), delete=False
            ),
        )

        # L2 delta: changed file + new npz group
        prev = utils.snapshot_gdir_state(workdir)
        with open(os.path.join(workdir, "mb_calib.json"), "w") as f:
            json.dump({"melt_f": 6.0}, f)
        arrays, meta = encode_npz(np.ones(4), "delta_check")
        _write_npz_store(workdir, arrays, meta, name="delta_check")
        _, changed = utils.write_level_manifest(
            workdir, "L2", prev, parent="L1", **_DS
        )
        publish(
            2,
            utils.gdir_to_tar.unwrapped(
                _FakeGdir(workdir, srcbase), delete=False, include=changed
            ),
        )

        # L5 standalone
        utils.write_level_manifest(
            workdir, "L5", {}, parent="L2", kind="standalone", **_DS
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

        # Level 2 = the L2 bundle plus the L1 root it requires
        gdirs = workflow.init_glacier_directories(
            [rid],
            from_prepro_level=2,
            prepro_border=80,
            prepro_base_url=self.BASE_URL,
        )
        gdir = gdirs[0]
        assert len(calls) == 2
        assert "/L2/" in calls[0]
        assert "/L1/" in calls[1]
        assert os.path.isfile(os.path.join(gdir.dir, "L1.manifest.json"))
        assert os.path.isfile(os.path.join(gdir.dir, "L2.manifest.json"))
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

    def test_init_from_local_delta_tree(self, delta_server, served_calls):
        """Test initialization from a local delta tree.

        Server tree is read from disk, so the L2 delta takes the grid
        from the L1 tar in the sibling folder.
        """
        server, _ = delta_server
        calls, rid = served_calls
        gdirs = workflow.init_glacier_directories(
            [rid], from_tar=os.path.join(server, "RGI62", "b_080", "L2")
        )
        gdir = gdirs[0]
        assert calls == []
        assert os.path.isfile(os.path.join(gdir.dir, "L1.manifest.json"))
        assert os.path.isfile(os.path.join(gdir.dir, "L2.manifest.json"))
        assert gdir.grid.nx > 0
        with open(os.path.join(gdir.dir, "mb_calib.json")) as f:
            assert json.load(f)["melt_f"] == 6.0

    @pytest.fixture
    def local_l2_only(self, delta_server, tmp_path):
        """A local tree with the L2 delta but none of the levels it requires."""
        server, _ = delta_server
        local = tmp_path / "local" / "RGI62" / "b_080" / "L2"
        shutil.copytree(Path(server, "RGI62", "b_080", "L2"), local)
        return str(local)

    def test_init_from_local_tree_fetches_missing_level(
        self, served_calls, local_l2_only
    ):
        calls, rid = served_calls
        gdirs = workflow.init_glacier_directories(
            [rid],
            from_tar=local_l2_only,
            prepro_base_url=self.BASE_URL,
            prepro_border=80,
        )
        gdir = gdirs[0]
        assert len(calls) == 1
        assert "/L1/" in calls[0]
        assert Path(gdir.dir, "L1.manifest.json").is_file()
        assert Path(gdir.dir, "L2.manifest.json").is_file()
        assert gdir.grid.nx > 0
        mb_calib = json.loads(Path(gdir.dir, "mb_calib.json").read_text())
        assert mb_calib["melt_f"] == 6.0

    @staticmethod
    def fetched_from(gdir, label):
        m = json.loads(Path(gdir.dir, f"{label}.manifest.json").read_text())
        assert utils.manifest_id(m) == m["id"]
        return m["fetched_from"]

    def test_url_load_records_fetched_from(self, served_calls):
        _, rid = served_calls
        for level, labels in ((2, ("L2", "L1")), (5, ("L5",))):
            gdir = workflow.init_glacier_directories(
                [rid],
                from_prepro_level=level,
                prepro_border=80,
                prepro_base_url=self.BASE_URL,
            )[0]
            for label in labels:
                assert self.fetched_from(gdir, label) == self.BASE_URL

    def test_local_tree_records_tree_root(self, delta_server, served_calls):
        server, rid = delta_server
        gdir = workflow.gdir_from_tar(
            rid, os.path.join(server, "RGI62", "b_080", "L2")
        )
        for label in ("L2", "L1"):
            assert self.fetched_from(gdir, label) == server

    def test_fallback_level_records_base_url(self, served_calls, local_l2_only):
        _, rid = served_calls
        gdir = workflow.gdir_from_tar(
            rid, local_l2_only, base_url=self.BASE_URL, prepro_border=80
        )
        local_root = str(Path(local_l2_only).parents[2])
        assert self.fetched_from(gdir, "L2") == local_root
        assert self.fetched_from(gdir, "L1") == self.BASE_URL

    def test_local_tree_missing_level_without_base_url(
        self, served_calls, local_l2_only
    ):
        calls, rid = served_calls
        with pytest.raises(FileNotFoundError):
            workflow.gdir_from_tar(rid, local_l2_only)
        assert calls == []


class TestParentChain:
    """Experiment trees sharing one L1-L2 tree, linked by parent manifests."""

    HOST = "https://delta.invalid/"

    def url(self, tree):
        return f"{self.HOST}{tree}/"

    @pytest.fixture
    def chain(self, tmp_path, hef_gdir, monkeypatch):
        rid = hef_gdir.rgi_id
        server = tmp_path / "server"
        src = tmp_path / "src"

        def workdir(name):
            return src / name / rid[:-6] / rid[:-3] / rid

        def publish(tree, label, wdir, include):
            fake = _FakeGdir(str(wdir), str(wdir.parents[2]))
            member = utils.gdir_to_tar.unwrapped(
                fake, delete=False, include=include
            )
            bundle = f"{rid[:-6]}.{rid[-5:-2]}"
            dest = server / tree / "RGI62" / "b_080" / label / rid[:-6]
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(dest / f"{bundle}.tar"), "w") as tf:
                tf.add(member, arcname=f"{bundle}/{rid}.tar.gz")
            os.remove(member)

        def level(wdir, label, parent=None, url=None, write=None, rm=()):
            # The root has nothing below it, so ships every file
            prev = utils.snapshot_gdir_state(wdir) if parent else {}
            for name, text in (write or {}).items():
                (wdir / name).write_text(text)
            for name in rm:
                os.remove(wdir / name)
            _, changed = utils.write_level_manifest(
                wdir, label, prev, parent=parent, parent_base_url=url, **_DS
            )
            return changed

        def fork(src_name, name):
            shutil.copytree(workdir(src_name), workdir(name))
            return workdir(name)

        # The shared L1-L2 tree
        base = workdir("base")
        shutil.copytree(hef_gdir.dir, base)
        publish("base", "L1", base, level(base, "L1", write={"l1.txt": "L1"}))
        fork("base", "base_l1")  # kept to regenerate L2 later
        publish(
            "base", "L2", base, level(base, "L2", "L1", write={"l2.txt": "L2"})
        )

        # Two experiments built on it, each in its own tree
        for exp in ("w5e5", "era5"):
            wdir = fork("base", exp)
            publish(
                exp,
                "L3",
                wdir,
                level(
                    wdir,
                    "L3",
                    "L2",
                    self.url("base"),
                    write={"climate.txt": exp},
                ),
            )

        # A removal followed by a re-addition
        wdir = fork("base", "rm")
        publish(
            "rm",
            "L3",
            wdir,
            level(
                wdir,
                "L3",
                "L2",
                self.url("base"),
                rm=["l2.txt", "inversion_flowlines.npz"],
            ),
        )
        publish(
            "rm", "L4", wdir, level(wdir, "L4", "L3", write={"l2.txt": "back"})
        )

        calls = []

        def fake_file_downloader(www_path, **kwargs):
            calls.append(www_path)
            local = server / www_path.replace(self.HOST, "")
            return str(local) if local.is_file() else None

        monkeypatch.setattr(_downloads, "file_downloader", fake_file_downloader)
        monkeypatch.setattr(_downloads, "_prepro_bundle_format", {})
        wd = tmp_path / "wd"
        wd.mkdir()
        cfg.PATHS["working_dir"] = str(wd)
        cfg.PARAMS["has_internet"] = False
        return SimpleNamespace(
            rid=rid,
            calls=calls,
            workdir=workdir,
            publish=publish,
            level=level,
            fork=fork,
        )

    def load(self, chain, tree, level):
        chain.calls.clear()
        return workflow.init_glacier_directories(
            [chain.rid],
            from_prepro_level=level,
            prepro_border=80,
            prepro_base_url=self.url(tree),
        )[0]

    def test_two_experiments_share_one_parent(self, chain):
        states = {}
        for exp in ("w5e5", "era5"):
            gdir = self.load(chain, exp, 3)
            # L2's parent link is null: it resolves against the base tree,
            # not against the experiment URL the user passed
            assert [c.split("/L")[0] for c in chain.calls] == [
                self.url(exp) + "RGI62/b_080",
                *[self.url("base") + "RGI62/b_080"] * 2,
            ]
            assert ["L3", "L2", "L1"] == [
                c.split("b_080/")[1][:2] for c in chain.calls
            ]
            states[exp] = utils.snapshot_gdir_state(gdir.dir)
            _assert_matches_build(gdir.dir, chain.workdir(exp))
        diff = {
            k for k in states["w5e5"] if states["w5e5"][k] != states["era5"][k]
        }
        assert diff == {"climate.txt", "L3.manifest.json"}

    def test_layers_record_fetched_from(self, chain):
        gdir = self.load(chain, "w5e5", 3)
        expected = {
            "L3": self.url("w5e5"),
            "L2": self.url("base"),
            "L1": self.url("base"),
        }
        for label, url in expected.items():
            m = json.loads(Path(gdir.dir, f"{label}.manifest.json").read_text())
            assert m["fetched_from"] == url
            assert utils.manifest_id(m) == m["id"]

    def test_full_archive_strips_fetched_from(self, chain):
        gdir = self.load(chain, "w5e5", 3)
        opath = utils.gdir_to_tar.unwrapped(gdir, delete=False)
        built = chain.workdir("w5e5")
        with tarfile.open(opath, "r:gz") as tar:
            for label in ("L1", "L2", "L3"):
                name = f"{label}.manifest.json"
                with tar.extractfile(f"{chain.rid}/{name}") as f:
                    assert f.read() == (built / name).read_bytes()

    def test_removal_then_readdition(self, chain):
        gdir = self.load(chain, "rm", 4)
        assert Path(gdir.dir, "l2.txt").read_text() == "back"
        assert not Path(gdir.dir, "inversion_flowlines.npz").exists()
        _assert_matches_build(gdir.dir, chain.workdir("rm"))

    def test_regenerated_parent_raises(self, chain):
        wdir = chain.workdir("base_l1")
        chain.publish(
            "base",
            "L2",
            wdir,
            chain.level(wdir, "L2", "L1", write={"l2.txt": "regenerated"}),
        )
        with open(wdir / "L2.manifest.json") as f:
            new_id = json.load(f)["id"]
        with open(chain.workdir("w5e5") / "L2.manifest.json") as f:
            old_id = json.load(f)["id"]

        with pytest.raises(InvalidWorkflowError, match="L3") as exc:
            self.load(chain, "w5e5", 3)
        msg = str(exc.value)
        assert old_id[:8] in msg and new_id[:8] in msg
        assert self.url("base") in msg

    def test_parent_without_manifest_raises(self, chain):
        # A legacy, cumulative L2: every file, no manifests
        wdir = chain.fork("base", "legacy")
        for fp in wdir.glob("*.manifest.json"):
            fp.unlink()
        chain.publish("base", "L2", wdir, None)
        with pytest.raises(InvalidWorkflowError, match="no manifest"):
            self.load(chain, "w5e5", 3)
        assert len(chain.calls) == 2

    def test_legacy_tree_is_one_fetch(self, chain):
        wdir = chain.fork("w5e5", "old")
        for fp in wdir.glob("*.manifest.json"):
            fp.unlink()
        chain.publish("old", "L3", wdir, None)
        gdir = self.load(chain, "old", 3)
        assert len(chain.calls) == 1
        assert Path(gdir.dir, "climate.txt").read_text() == "w5e5"
        assert not list(Path(gdir.dir).glob("*.manifest.json"))


class TestStartState:
    """`_start_state` at a `3a` resume, with and without the L2 state."""

    @pytest.fixture
    def gdir(self, tmp_path):
        return SimpleNamespace(dir=str(tmp_path), rgi_id="RGI60-11.00897")

    def test_missing_l2_state_raises(self, gdir):
        from oggm.cli.prepro_levels import _start_state

        with pytest.raises(InvalidWorkflowError, match="L2.state.json"):
            _start_state(gdir, True)

    def test_l2_state_is_loaded_and_removed(self, gdir):
        from oggm.cli.prepro_levels import _start_state

        fp = Path(gdir.dir, "L2.state.json")
        fp.write_text(json.dumps({"a.txt": "abc"}))
        assert _start_state(gdir, True) == {"a.txt": "abc"}
        assert not fp.exists()


def test_level_consistency_mismatch(tmp_path):
    """Layering tars directly still checks the parent ids."""
    rid = "RGI60-11.00897"
    base = str(tmp_path / "src")
    gdir_dir = _make_fake_gdir(os.path.join(base, rid))

    def tar(name, include=None):
        path = utils.gdir_to_tar.unwrapped(
            _FakeGdir(str(gdir_dir), base), delete=False, include=include
        )
        return shutil.move(path, str(tmp_path / name))

    utils.write_level_manifest(gdir_dir, "L1", {}, **_DS)
    l0_a = tar("l0_a.tar.gz")

    # A level-2 delta built on a *different* L1
    with open(os.path.join(gdir_dir, "dem_source.txt"), "w") as f:
        f.write("other DEM")
    utils.write_level_manifest(gdir_dir, "L1", {}, **_DS)
    prev = utils.snapshot_gdir_state(gdir_dir)
    with open(os.path.join(gdir_dir, "mb_calib.json"), "w") as f:
        json.dump({"melt_f": 6.0}, f)
    _, changed = utils.write_level_manifest(
        gdir_dir, "L2", prev, parent="L1", **_DS
    )
    l1_on_b = tar("l1.tar.gz", changed)

    layered = tmp_path / "layered"
    with pytest.raises(InvalidWorkflowError, match="L2 was built on L1"):
        utils._workflow._extract_tars([l0_a, l1_on_b], str(layered / rid))


def _raw_manifest(gdir_dir, **overrides):
    """Writes a manifest as is, without `write_level_manifest` checks."""
    m = {
        "schema_version": "1.0",
        "kind": "delta",
        "rgi_id": "RGI60-11.00897",
        "label": "L2",
        "id": "0" * 64,
        "parent": None,
        "created": "2999-01-01T00:00:00+00:00",
        "files": {"added": {}, "updated": {}, "removed": []},
    }
    m.update(overrides)
    with open(os.path.join(gdir_dir, f"{m['label']}.manifest.json"), "w") as f:
        json.dump(m, f)


def test_finalize_rejects_unknown_kind(tmp_path):
    gdir_dir = _make_fake_gdir(tmp_path / "RGI60-11.00897")
    utils.write_level_manifest(gdir_dir, "L1", {}, **_DS)
    _raw_manifest(gdir_dir, kind="unknown", label="extra")
    with pytest.raises(InvalidWorkflowError, match="'unknown'"):
        utils._workflow._finalize_merged_dir(str(gdir_dir))


def test_finalize_rejects_two_chains(tmp_path):
    gdir_dir = _make_fake_gdir(tmp_path / "RGI60-11.00897")
    utils.write_level_manifest(gdir_dir, "L1", {}, **_DS)
    _raw_manifest(gdir_dir)  # an L2 that is not built on L1
    with pytest.raises(InvalidWorkflowError, match="one chain"):
        utils._workflow._finalize_merged_dir(str(gdir_dir))


@pytest.mark.parametrize("version", ["2.0", 2])
def test_finalize_rejects_newer_major(tmp_path, version):
    gdir_dir = _make_fake_gdir(tmp_path / "RGI60-11.00897")
    utils.write_level_manifest(gdir_dir, "L1", {}, **_DS)
    _raw_manifest(gdir_dir, schema_version=version)
    with pytest.raises(InvalidWorkflowError, match="schema 2"):
        utils._workflow._finalize_merged_dir(str(gdir_dir))
