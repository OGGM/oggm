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
