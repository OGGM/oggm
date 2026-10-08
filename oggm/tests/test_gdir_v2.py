"""Tests for the v2 glacier directory format (HANDOFF_1896).

Covers dataset group access (``open_group``/``write_group``) and later
steps: gridded/climate funnels, geoparquet vectors, zip containers,
v2 converters and streaming.
"""

import io
import json
import os
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from oggm import cfg, utils, workflow
from oggm.exceptions import InvalidWorkflowError

pytestmark = pytest.mark.test_env("utils")


def _demo_dataset(seed=0):
    """A small gridded-style dataset with CF-ish attrs."""
    rng = np.random.default_rng(seed)
    ds = xr.Dataset(
        {
            "topo": (("y", "x"), rng.random((4, 5))),
            "glacier_mask": (("y", "x"), rng.integers(0, 2, (4, 5))),
        },
        coords={"x": np.arange(5) * 100.0, "y": np.arange(4) * 100.0},
        attrs={"pyproj_srs": "+proj=tmerc", "author": "OGGM"},
    )
    ds["topo"].attrs["units"] = "m"
    return ds


class TestGroupAccess:
    """open_group / write_group on GlacierDirectory."""

    def test_write_group_roundtrip(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        ds = _demo_dataset()

        gdir.write_group(ds, "gridded_data", filesuffix="_v2test", mode="w")

        # v1 netCDF file, not an npz store entry
        fp = gdir.get_filepath("gridded_data", filesuffix="_v2test")
        assert os.path.isfile(fp)
        assert not os.path.exists(
            gdir.get_store_filepath("gridded_data", filesuffix="_v2test")
        )

        with gdir.open_group("gridded_data", filesuffix="_v2test") as back:
            assert_allclose(back["topo"].values, ds["topo"].values)
            assert back.attrs["pyproj_srs"] == "+proj=tmerc"
            assert back["topo"].attrs["units"] == "m"
            # zlib 4 for size, shuffle for speed
            for name in back.data_vars:
                enc = back[name].encoding
                assert enc.get("zlib") is True, name
                assert enc.get("shuffle") is True, name
                assert enc.get("complevel") == 4, name

    def test_write_group_filesuffix_and_has_file(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        ds = _demo_dataset(1)

        gdir.write_group(ds, "gcm_data", filesuffix="_CCSM4_v2test", mode="w")

        # single underscore, matching the read_store convention
        assert os.path.isfile(
            os.path.join(gdir.dir, "gcm_data_CCSM4_v2test.nc")
        )
        assert gdir.has_file("gcm_data", filesuffix="_CCSM4_v2test")
        gdir.delete_group("gcm_data", filesuffix="_CCSM4_v2test")
        assert not gdir.has_file("gcm_data", filesuffix="_CCSM4_v2test")

    def test_write_group_append_adds_variable(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        ds = _demo_dataset(2)
        gdir.write_group(ds, "gridded_data", filesuffix="_v2app", mode="w")

        extra = xr.Dataset(
            {"consensus": (("y", "x"), np.ones((4, 5)))},
            coords={"x": ds.x, "y": ds.y},
        )
        gdir.write_group(extra, "gridded_data", filesuffix="_v2app", mode="a")

        with gdir.open_group("gridded_data", filesuffix="_v2app") as back:
            # old variables survive, new one is there
            assert "topo" in back
            assert "consensus" in back
            assert_allclose(back["consensus"].values, 1.0)

    def test_write_group_w_replaces_group(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        gdir.write_group(
            _demo_dataset(3), "gridded_data", filesuffix="_v2rep", mode="w"
        )
        ds2 = xr.Dataset({"only_var": ("z", np.arange(3.0))})
        gdir.write_group(ds2, "gridded_data", filesuffix="_v2rep", mode="w")

        with gdir.open_group("gridded_data", filesuffix="_v2rep") as back:
            assert "only_var" in back
            assert "topo" not in back

    def test_open_group_reads_v1_netcdf4_file(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        # an uncompressed v1 .nc written by the netCDF4 engine
        ds = _demo_dataset(5)
        fp = gdir.get_filepath("gcm_data", filesuffix="_v2legacy")
        ds.to_netcdf(fp)
        with gdir.open_group("gcm_data", filesuffix="_v2legacy") as back:
            assert "topo" in back
            assert_allclose(back["topo"].values, ds["topo"].values)

    def test_open_group_missing_raises(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            hef_gdir.open_group("gcm_data", filesuffix="_does_not_exist")

    def test_filesuffix_none_never_makes_none_group(self, tmp_path, hef_gdir):
        import glob as _glob

        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir

        gdir.write_group(
            _demo_dataset(4), "gcm_data", filesuffix=None, mode="w"
        )
        gdir.write_store(
            {"a": np.arange(3.0)}, "inversion_input", filesuffix=None
        )

        assert not _glob.glob(os.path.join(gdir.dir, "*None*"))
        assert not _glob.glob(
            os.path.join(gdir.get_filepath("data_store"), "*None*")
        )
        assert gdir.has_file("gcm_data", filesuffix=None)
        with gdir.open_group("gcm_data", filesuffix=None) as back:
            assert "topo" in back
        out = gdir.read_store("inversion_input", filesuffix=None)
        assert_allclose(out["a"], np.arange(3.0))
