"""Tests for the v2 glacier directory format (HANDOFF_1896).

Covers the zarr-group access funnel (``open_group``/``write_group``),
and later steps: gridded/climate funnels, geoparquet vectors, zip
containers, v2 converters and streaming.
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
    """open_group / write_group on GlacierDirectory (step 1)."""

    def test_write_group_roundtrip(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        ds = _demo_dataset()

        gdir.write_group(ds, "gridded_data", filesuffix="_v2test", mode="w")

        zarr_fp = gdir.get_filepath("data_store").replace(".pkl", ".zarr")
        assert os.path.isdir(os.path.join(zarr_fp, "gridded_data_v2test"))

        with gdir.open_group("gridded_data", filesuffix="_v2test") as back:
            assert_allclose(back["topo"].values, ds["topo"].values)
            assert back.attrs["pyproj_srs"] == "+proj=tmerc"
            assert back["topo"].attrs["units"] == "m"

    def test_write_group_filesuffix_and_has_file(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        ds = _demo_dataset(1)

        gdir.write_group(ds, "gcm_data", filesuffix="_CCSM4_v2test", mode="w")

        zarr_fp = gdir.get_filepath("data_store").replace(".pkl", ".zarr")
        # single underscore, matching the read_store convention
        assert os.path.isdir(os.path.join(zarr_fp, "gcm_data_CCSM4_v2test"))
        assert gdir.has_file("gcm_data", filesuffix="_CCSM4_v2test")

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

    def test_open_group_falls_back_to_legacy_nc(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        # a legacy .nc without a matching zarr group (v1 directory)
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

        zarr_fp = gdir.get_filepath("data_store").replace(".pkl", ".zarr")
        assert not _glob.glob(os.path.join(zarr_fp, "*None*"))
        assert gdir.has_file("gcm_data", filesuffix=None)
        with gdir.open_group("gcm_data", filesuffix=None) as back:
            assert "topo" in back
        out = gdir.read_store("inversion_input", filesuffix=None)
        assert_allclose(out[0]["a"], np.arange(3.0))


class TestGriddedFunnel:
    """GriddedNcdfFile keeps its API but syncs to the zarr store (step 2)."""

    def test_gridded_ncdf_file_syncs_group(self, tmp_path, hef_gdir):
        from oggm.core.gis import GriddedNcdfFile

        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir

        # regression reference: the gridded content before the shim run
        with gdir.open_group("gridded_data") as ds:
            ref = ds.load()

        with GriddedNcdfFile(gdir) as nc:
            v = nc.createVariable("v2_shim_test", "f4", ("y", "x"))
            v.units = "m"
            v[:] = 1.5

        zarr_fp = gdir.get_filepath("data_store").replace(".pkl", ".zarr")
        assert os.path.isdir(os.path.join(zarr_fp, "gridded_data"))
        # the scratch netCDF is not persisted any more
        assert not os.path.exists(gdir.get_filepath("gridded_data"))

        # group content == what was there before + the new variable
        with gdir.open_group("gridded_data") as ds:
            for name in ref.data_vars:
                assert_allclose(ds[name].values, ref[name].values, err_msg=name)
            assert ds.attrs["pyproj_srs"] == ref.attrs["pyproj_srs"]
            assert_allclose(ds["v2_shim_test"].values, 1.5)
            assert ds["v2_shim_test"].attrs["units"] == "m"

    def test_task_writes_reach_the_group(self, tmp_path, hef_gdir):
        from oggm import tasks

        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        tasks.glacier_masks(gdir)

        zarr_fp = gdir.get_filepath("data_store").replace(".pkl", ".zarr")
        assert os.path.isdir(os.path.join(zarr_fp, "gridded_data"))
        assert not os.path.exists(gdir.get_filepath("gridded_data"))
        with gdir.open_group("gridded_data") as ds:
            assert "glacier_mask" in ds
            assert ds["glacier_mask"].values.sum() > 0


class TestClimateFunnel:
    """write_monthly_climate_file writes to the store (step 3)."""

    def test_climate_write_creates_group_and_reads_back(
        self, tmp_path, hef_gdir
    ):
        from oggm.core.massbalance import MonthlyTIModel

        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        time = pd.date_range("2000-01-01", periods=36, freq="MS")
        rng = np.random.default_rng(42)
        prcp = rng.random(36).astype(np.float32) * 100
        temp = rng.random(36).astype(np.float32) * 10 - 5

        gdir.write_monthly_climate_file(
            time,
            prcp,
            temp,
            2500.0,
            10.0,
            46.0,
            source="v2test",
            file_name="gcm_data",
            filesuffix="_v2clim",
        )

        zarr_fp = gdir.get_filepath("data_store").replace(".pkl", ".zarr")
        assert os.path.isdir(os.path.join(zarr_fp, "gcm_data_v2clim"))

        mb = MonthlyTIModel(
            gdir, filename="gcm_data", input_filesuffix="_v2clim"
        )
        assert mb.ref_hgt == 2500.0
        assert mb.climate_source == "v2test"
        assert mb.ys == 2000
        assert mb.ye == 2002
        assert_allclose(
            mb.prcp, prcp.astype(np.float64) * mb.prcp_fac, rtol=1e-6
        )

    def test_climate_noleap_calendar_roundtrip(self, tmp_path, hef_gdir):
        from oggm.core.massbalance import MonthlyTIModel

        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        time = xr.date_range(
            "2000-01-01",
            periods=24,
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        prcp = np.ones(24, np.float32) * 42
        temp = np.zeros(24, np.float32)

        gdir.write_monthly_climate_file(
            time,
            prcp,
            temp,
            2000.0,
            10.0,
            46.0,
            calendar="noleap",
            source="v2noleap",
            file_name="gcm_data",
            filesuffix="_v2nl",
        )
        mb = MonthlyTIModel(gdir, filename="gcm_data", input_filesuffix="_v2nl")
        assert mb.ys == 2000
        assert mb.ye == 2001
        assert len(mb.prcp) == 24


class TestVectorFunnel:
    """write_shapefile/read_shapefile use geoparquet (step 4)."""

    def test_legacy_tar_readable_then_parquet_written(self, tmp_path, hef_gdir):
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir

        # the fixture dir ships legacy shapefile-tars: reading must work
        gdf = gdir.read_shapefile("outlines")
        assert gdf.crs is not None

        # writing goes to parquet now
        gdir.write_shapefile(gdf, "outlines")
        fp_parquet = gdir.get_filepath("outlines").replace(".shp", ".parquet")
        assert os.path.exists(fp_parquet)

        # and reading prefers it, CRS preserved
        back = gdir.read_shapefile("outlines")
        assert back.crs == gdf.crs
        assert back.geometry.iloc[0].equals(gdf.geometry.iloc[0])
        assert gdir.has_file("outlines")

    def test_gdir_init_from_parquet_only_dir(self, tmp_path, hef_gdir):
        import glob as _glob

        import oggm

        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        gdf = gdir.read_shapefile("outlines")
        gdir.write_shapefile(gdf, "outlines")
        # drop every legacy outlines artifact
        for f in _glob.glob(os.path.join(gdir.dir, "outlines.tar*")):
            os.remove(f)
        for f in _glob.glob(os.path.join(gdir.dir, "outlines.sh[px]")):
            os.remove(f)

        # re-open the directory from disk: outlines come from parquet
        new_gdir = oggm.GlacierDirectory(gdir.rgi_id, base_dir=gdir.base_dir)
        assert new_gdir.rgi_id == gdir.rgi_id

    def test_write_shape_tar_keeps_unrelated_siblings(self, tmp_path, hef_gdir):
        from oggm.utils._workflow import _write_shape_to_disk

        cfg.PATHS["working_dir"] = str(tmp_path)
        gdf = hef_gdir.read_shapefile("outlines")

        scratch = tmp_path / "shpglob"
        scratch.mkdir()
        fpath = str(scratch / "outlines.shp")
        parquet = str(scratch / "outlines.parquet")
        gdf.to_parquet(parquet)

        _write_shape_to_disk(gdf, fpath, to_tar=True)

        tar_fp = fpath.replace(".shp", ".tar")
        if cfg.PARAMS["use_compression"]:
            tar_fp += ".gz"
        with tarfile.open(tar_fp, "r:*") as tf:
            names = tf.getnames()

        # the parquet sibling is neither tarred nor deleted
        assert "outlines.parquet" not in names
        assert os.path.exists(parquet)
        # the shapefile parts are tarred and removed
        assert "outlines.shp" in names
        assert not os.path.exists(fpath)


class _FakeGdir:
    def __init__(self, path, base_dir):
        self.dir = path
        self.base_dir = base_dir
        self.rgi_id = os.path.basename(path)


def _make_fake_gdir_dir(path):
    """Minimal on-disk glacier dir: two files + one zarr group."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "diagnostics.json"), "w") as f:
        json.dump({"a": 1}, f)
    with open(os.path.join(path, "dem.tif"), "wb") as f:
        f.write(b"not really a tif")
    ds = xr.Dataset({"thick": ("x", np.arange(5, dtype=float))})
    ds.to_zarr(
        os.path.join(path, "data_store.zarr"),
        group="inversion_flowlines",
        mode="a",
        zarr_format=2,
        consolidated=True,
    )
    return path


def _build_zip_bundle(path, rid, with_manifest=None):
    """Write a zip-of-zip bundle (bundle/<rid>.zip) as downloads look."""
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w", zipfile.ZIP_STORED) as inner:
        inner.writestr(f"{rid}/diagnostics.json", '{"a": 1}\n')
        if with_manifest is not None:
            level = with_manifest["level"]
            inner.writestr(
                f"{rid}/L{level}.manifest.json", json.dumps(with_manifest)
            )
        inner.writestr(f"{rid}/filler.bin", os.urandom(40000))
    bundle = f"{rid[:-6]}.{rid[-5:-2]}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as outer:
        outer.writestr(f"{bundle}/{rid}.zip", inner_buf.getvalue())


class TestZipContainers:
    """gdir_to_archive / base_dir_to_bundles / robust_archive_extract
    (step 5 of HANDOFF_1896)."""

    def test_gdir_to_archive_zip(self, tmp_path):
        rid = "RGI60-07.00001"
        gdir_dir = _make_fake_gdir_dir(str(tmp_path / rid))
        fake = _FakeGdir(gdir_dir, str(tmp_path))

        opath = utils.gdir_to_archive.unwrapped(fake, delete=False, fmt="zip")
        assert opath.endswith(rid + ".zip")
        with zipfile.ZipFile(opath) as zf:
            names = zf.namelist()
            assert f"{rid}/diagnostics.json" in names
            assert f"{rid}/dem.tif" in names
            assert any(
                n.startswith(f"{rid}/data_store.zarr/inversion_flowlines")
                for n in names
            )
            # STORED members so HTTP ranges can stream them
            assert all(
                zi.compress_type == zipfile.ZIP_STORED for zi in zf.infolist()
            )

    def test_gdir_to_archive_zip_include(self, tmp_path):
        rid = "RGI60-07.00001"
        gdir_dir = _make_fake_gdir_dir(str(tmp_path / rid))
        fake = _FakeGdir(gdir_dir, str(tmp_path))

        opath = utils.gdir_to_archive.unwrapped(
            fake,
            delete=False,
            fmt="zip",
            include=["diagnostics.json", "data_store.zarr/inversion_flowlines"],
        )
        with zipfile.ZipFile(opath) as zf:
            names = zf.namelist()
        assert f"{rid}/diagnostics.json" in names
        assert not any("dem.tif" in n for n in names)
        assert any(
            n.startswith(f"{rid}/data_store.zarr/inversion_flowlines")
            for n in names
        )

    def test_gdir_to_archive_tar_delegation(self, tmp_path):
        # fmt='tar' keeps the old behavior, but gdir_to_tar still works
        rid = "RGI60-07.00001"
        gdir_dir = _make_fake_gdir_dir(str(tmp_path / rid))
        fake = _FakeGdir(gdir_dir, str(tmp_path))
        opath = utils.gdir_to_archive.unwrapped(fake, delete=False, fmt="tar")
        assert opath.endswith(rid + ".tar.gz")
        with tarfile.open(opath, "r:gz") as tf:
            assert f"{rid}/diagnostics.json" in tf.getnames()

    @pytest.mark.parametrize(
        "rgi_ids",
        [
            ["RGI60-07.00001", "RGI60-07.00099", "RGI60-07.00100"],
            [
                "RGI2000-v7.0-G-07-00001",
                "RGI2000-v7.0-G-07-00099",
                "RGI2000-v7.0-G-07-00100",
            ],
        ],
        ids=["RGI6", "RGI7"],
    )
    def test_zip_bundle_roundtrip(self, tmp_path, rgi_ids):
        # zip analog of TestStartFromTar::test_bundle_tar_roundtrip
        base_dir = str(tmp_path / "per_glacier")

        for rid in rgi_ids:
            gdir_dir = os.path.join(base_dir, rid[:-6], rid[:-3], rid)
            _make_fake_gdir_dir(gdir_dir)
            fake = _FakeGdir(gdir_dir, base_dir)
            utils.gdir_to_archive.unwrapped(fake, delete=True, fmt="zip")

        utils.base_dir_to_bundles(base_dir, fmt="zip")

        for rid in rgi_ids:
            bundle = f"{rid[:-6]}.{rid[-5:-2]}"
            outer = os.path.join(base_dir, rid[:-6], bundle + ".zip")
            assert os.path.isfile(outer)
            with zipfile.ZipFile(outer) as zf:
                assert f"{bundle}/{rid}.zip" in zf.namelist()
                assert all(
                    zi.compress_type == zipfile.ZIP_STORED
                    for zi in zf.infolist()
                )
            # per-glacier zips were consumed into the bundle
            assert not os.path.exists(
                os.path.join(base_dir, rid[:-6], rid[:-3], rid + ".zip")
            )

            # the member path gdir_from_prepro constructs
            member = os.path.join(base_dir, rid[:-6], bundle, rid + ".zip")
            to_dir = str(tmp_path / "extracted" / rid)
            utils.robust_archive_extract(member, to_dir)
            assert os.path.isfile(os.path.join(to_dir, "diagnostics.json"))
            assert os.path.isdir(
                os.path.join(to_dir, "data_store.zarr", "inversion_flowlines")
            )

    def test_robust_archive_extract_delegates_to_tar(self, tmp_path):
        rid = "RGI60-07.00001"
        gdir_dir = _make_fake_gdir_dir(str(tmp_path / rid))
        fake = _FakeGdir(gdir_dir, str(tmp_path))
        opath = utils.gdir_to_tar.unwrapped(fake, delete=False)
        # extraction lands in dirname(to_dir)/<archive prefix>, so
        # to_dir is named after the RGI ID (as GlacierDirectory does)
        to_dir = str(tmp_path / "out" / rid)
        os.makedirs(os.path.dirname(to_dir))
        utils.robust_archive_extract(opath, to_dir)
        assert os.path.isfile(os.path.join(to_dir, "diagnostics.json"))

    def test_robust_archive_extract_direct_zip(self, tmp_path):
        rid = "RGI60-07.00001"
        gdir_dir = _make_fake_gdir_dir(str(tmp_path / rid))
        fake = _FakeGdir(gdir_dir, str(tmp_path))
        opath = utils.gdir_to_archive.unwrapped(fake, delete=False, fmt="zip")
        to_dir = str(tmp_path / "out" / rid)
        os.makedirs(os.path.dirname(to_dir))
        utils.robust_archive_extract(opath, to_dir)
        assert os.path.isfile(os.path.join(to_dir, "diagnostics.json"))

    def test_robust_archive_extract_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            utils.robust_archive_extract(
                str(tmp_path / "nowhere" / "RGI60-07.00001.zip"),
                str(tmp_path / "out"),
            )

    def test_mixed_tar_and_zip_layering(self, tmp_path):
        # a tar materialisation topped by a zip delta merges cleanly
        rid = "RGI60-07.00001"
        gdir_dir = _make_fake_gdir_dir(str(tmp_path / rid))
        fake = _FakeGdir(gdir_dir, str(tmp_path))
        tar_path = utils.gdir_to_tar.unwrapped(fake, delete=False)

        # the "delta": one new file, packaged as zip
        delta_dir = str(tmp_path / "delta" / rid)
        os.makedirs(delta_dir)
        with open(os.path.join(delta_dir, "mb_calib.json"), "w") as f:
            json.dump({"melt_f": 5.0}, f)
        fake_delta = _FakeGdir(delta_dir, str(tmp_path / "delta"))
        zip_path = utils.gdir_to_archive.unwrapped(
            fake_delta, delete=False, fmt="zip"
        )

        to_dir = str(tmp_path / "merged" / rid)
        os.makedirs(os.path.dirname(to_dir))
        from oggm.utils._workflow import _extract_tars

        _extract_tars([tar_path, zip_path], to_dir)
        assert os.path.isfile(os.path.join(to_dir, "diagnostics.json"))
        assert os.path.isfile(os.path.join(to_dir, "mb_calib.json"))


class TestManifestFormatVersion:
    """format_version in manifests + mixed-version refusal (step 5)."""

    def test_write_level_manifest_format_version(self, tmp_path):
        gdir_dir = _make_fake_gdir_dir(str(tmp_path / "RGI60-07.00001"))

        mp, _ = utils.write_level_manifest(
            gdir_dir,
            level=0,
            prev_state={},
            requires=[],
            dataset_tag="t",
            border=80,
            rgi_version="62",
        )
        with open(mp) as f:
            assert json.load(f)["format_version"] == 1

        mp, _ = utils.write_level_manifest(
            gdir_dir,
            level=0,
            prev_state={},
            requires=[],
            dataset_tag="t",
            border=80,
            rgi_version="62",
            format_version=2,
        )
        with open(mp) as f:
            assert json.load(f)["format_version"] == 2

    def test_check_level_compat_mixed_format_version(self):
        from oggm.utils._workflow import _check_level_compat

        base = dict(dataset_id="d", requires=[], includes_levels=None)
        m0 = dict(base, level=0, includes_levels=[0, 1, 2, 3], format_version=1)
        m4 = dict(base, level=4, requires=[3], format_version=2)
        with pytest.raises(InvalidWorkflowError, match="format"):
            _check_level_compat([m0, m4])

        # legacy manifests without the key default to 1 and stay valid
        m0.pop("format_version")
        m4["format_version"] = 1
        _check_level_compat([m0, m4])


class TestZipPrepro:
    """zip probing + manifest peek on the download path (step 5)."""

    def test_peek_level_manifest_zip(self, tmp_path):
        rid = "RGI60-07.00001"
        bundle = f"{rid[:-6]}.{rid[-5:-2]}"
        zip_base = str(tmp_path / f"{bundle}.zip")

        # legacy-ish zip bundle without a manifest -> None
        _build_zip_bundle(zip_base, rid)
        assert workflow._peek_level_manifest(zip_base, rid, 1) is None

        manifest = {
            "level": 1,
            "kind": "delta",
            "requires": [0],
            "format_version": 2,
        }
        _build_zip_bundle(zip_base, rid, with_manifest=manifest)
        assert workflow._peek_level_manifest(zip_base, rid, 1) == manifest

        # truncated zip -> None, not a crash
        size = os.path.getsize(zip_base)
        with open(zip_base, "r+b") as f:
            f.truncate(size - 20000)
        assert workflow._peek_level_manifest(zip_base, rid, 1) is None

    def test_prepro_probes_zip_first(self, tmp_path, monkeypatch):
        from oggm.utils import _downloads

        rid = "RGI60-07.00001"
        bundle = f"{rid[:-6]}.{rid[-5:-2]}"
        zip_base = str(tmp_path / f"{bundle}.zip")
        _build_zip_bundle(zip_base, rid)

        requested = []

        def fake_downloader(url, *args, **kwargs):
            requested.append(url)
            return zip_base if url.endswith(".zip") else None

        monkeypatch.setattr(_downloads, "file_downloader", fake_downloader)
        monkeypatch.setattr(_downloads, "_prepro_bundle_format", {})

        out = _downloads._get_prepro_gdir_unlocked(
            "62", rid, 80, 3, base_url="https://fake.example/gdirs/"
        )
        assert out == zip_base
        assert requested[0].endswith(f"{bundle}.zip")
        # format is cached: a second call goes straight to zip
        requested.clear()
        out = _downloads._get_prepro_gdir_unlocked(
            "62", rid, 80, 3, base_url="https://fake.example/gdirs/"
        )
        assert out == zip_base
        assert len(requested) == 1

    def test_prepro_zip_missing_falls_back_to_tar(self, tmp_path, monkeypatch):
        from oggm.utils import _downloads

        rid = "RGI60-07.00001"
        bundle = f"{rid[:-6]}.{rid[-5:-2]}"
        tar_base = str(tmp_path / f"{bundle}.tar")
        with tarfile.open(tar_base, "w") as tf:
            payload = b"x"
            ti = tarfile.TarInfo(f"{bundle}/{rid}.tar.gz")
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))

        def fake_downloader(url, *args, **kwargs):
            return tar_base if url.endswith(f"{bundle}.tar") else None

        monkeypatch.setattr(_downloads, "file_downloader", fake_downloader)
        monkeypatch.setattr(_downloads, "_prepro_bundle_format", {})

        out = _downloads._get_prepro_gdir_unlocked(
            "62", rid, 80, 3, base_url="https://fake.example/gdirs/"
        )
        assert out == tar_base


# --- step 6: compat.py v2 converters + fixture tree ---

# repo-local sample data (untracked), skip when absent
_EXT_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "ext",
    "data",
)
DELTA_SERVER = os.path.join(_EXT_DATA, "delta_server")
STREAM_SERVER = os.path.join(_EXT_DATA, "stream_server")

needs_delta_server = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DELTA_SERVER, "RGI62")),
    reason="local ext/data/delta_server fixture tree not available",
)


def _extract_v1_member(level, rid, dest_base):
    """Extract one glacier member of a v1 delta_server bundle."""
    region = rid[:-6]
    bundle = f"{region}.{rid[-5:-2]}"
    member = os.path.join(
        DELTA_SERVER,
        "RGI62",
        "b_080",
        f"L{level}",
        region,
        bundle,
        rid + ".tar.gz",
    )
    to_dir = os.path.join(dest_base, region, rid[:-3], rid)
    os.makedirs(os.path.dirname(to_dir), exist_ok=True)
    utils.robust_archive_extract(member, to_dir)
    return to_dir


@needs_delta_server
class TestV2Converters:

    @pytest.fixture(autouse=True)
    def _init_cfg(self, tmp_path):
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path / "wd")
        os.makedirs(cfg.PATHS["working_dir"], exist_ok=True)
        cfg.PARAMS["border"] = 80
        cfg.PARAMS["rgi_version"] = "62"

    def test_convert_gdir_to_v2_and_back(self, tmp_path):
        from oggm.utils import compat

        rid = "RGI60-07.00001"
        gdir_dir = _extract_v1_member(3, rid, str(tmp_path))

        # reference content before conversion
        with xr.open_dataset(os.path.join(gdir_dir, "gridded_data.nc")) as ds:
            ref_grid = ds.load()
        with xr.open_dataset(
            os.path.join(gdir_dir, "climate_historical.nc")
        ) as ds:
            ref_clim = ds.load()

        gdir = compat.convert_gdir_to_v2(gdir_dir)

        # originals gone, groups + parquet there
        assert not os.path.exists(os.path.join(gdir_dir, "gridded_data.nc"))
        assert not os.path.exists(
            os.path.join(gdir_dir, "climate_historical.nc")
        )
        assert not os.path.exists(os.path.join(gdir_dir, "outlines.tar.gz"))
        assert os.path.exists(os.path.join(gdir_dir, "outlines.parquet"))

        with gdir.open_group("gridded_data") as ds:
            assert_allclose(ds["topo"].values, ref_grid["topo"].values)
        with gdir.open_group("climate_historical") as ds:
            assert_allclose(ds["temp"].values, ref_clim["temp"].values)
            assert ds.attrs["ref_hgt"] == ref_clim.attrs["ref_hgt"]
        gdf = gdir.read_shapefile("outlines")
        assert gdf.crs is not None

        # and back to v1.6
        compat.convert_gdir_to_v16(gdir_dir)
        assert os.path.exists(os.path.join(gdir_dir, "gridded_data.nc"))
        with xr.open_dataset(
            os.path.join(gdir_dir, "climate_historical.nc")
        ) as ds:
            assert_allclose(ds["temp"].values, ref_clim["temp"].values)
        with xr.open_dataset(os.path.join(gdir_dir, "gridded_data.nc")) as ds:
            assert_allclose(ds["topo"].values, ref_grid["topo"].values)

    def test_convert_prepro_to_v2_artifacts(self, tmp_path):
        from oggm.utils import compat

        rid = "RGI60-07.00001"
        out = compat.convert_prepro_to_v2_artifacts(
            DELTA_SERVER,
            [rid],
            str(tmp_path / "v2"),
            dataset_tag="v2test",
            workdir=str(tmp_path / "wk"),
        )
        region, bundle = rid[:-6], f"{rid[:-6]}.{rid[-5:-2]}"
        for lvl in (0, 3, 4, 5):
            assert os.path.isfile(
                os.path.join(out, f"L{lvl}", region, bundle + ".zip")
            )

        # the L3 materialisation keeps its kind and gets format_version 2
        m = workflow._peek_level_manifest(
            os.path.join(out, "L3", region, bundle + ".zip"), rid, 3
        )
        assert m["format_version"] == 2
        assert m["includes_levels"] == [0, 1, 2, 3]
        assert m["requires"] == []

        # the L4 delta stays a delta and ships no v1 payload formats
        m4 = workflow._peek_level_manifest(
            os.path.join(out, "L4", region, bundle + ".zip"), rid, 4
        )
        assert m4["format_version"] == 2
        assert m4["requires"] == [0, 1, 2, 3]
        with zipfile.ZipFile(
            os.path.join(out, "L4", region, bundle + ".zip")
        ) as outer:
            with outer.open(f"{bundle}/{rid}.zip") as fobj:
                with zipfile.ZipFile(fobj) as inner:
                    names = inner.namelist()
        # model run NetCDFs are passthrough members - still there
        assert any("model_geometry" in n for n in names)
        # but no legacy shapefile tars or pickles
        assert not any(n.endswith(".tar.gz") for n in names)
        assert not any(n.endswith(".pkl") for n in names)

        # converted L3 member holds zarr + parquet instead of .nc/.tar.gz
        with zipfile.ZipFile(
            os.path.join(out, "L3", region, bundle + ".zip")
        ) as outer:
            with outer.open(f"{bundle}/{rid}.zip") as fobj:
                with zipfile.ZipFile(fobj) as inner:
                    names = inner.namelist()
        assert any("data_store.zarr/gridded_data/" in n for n in names)
        assert any("data_store.zarr/climate_historical/" in n for n in names)
        assert f"{rid}/outlines.parquet" in names
        assert not any(n.endswith(".nc") for n in names)
        assert not any(n.endswith(".tar.gz") for n in names)


@pytest.fixture(scope="session")
def stream_server_tree():
    """The v2 (zip/zarr/parquet) sibling of ext/data/delta_server.

    Generated once from the local v1 tree and kept untracked,
    regenerated only if missing.
    """
    from oggm.utils import compat

    if not os.path.isdir(os.path.join(DELTA_SERVER, "RGI62")):
        pytest.skip("local ext/data/delta_server fixture tree missing")
    cfg.initialize()
    rids = ["RGI60-07.00001", "RGI60-07.00099", "RGI60-07.00100"]
    marker = os.path.join(
        STREAM_SERVER,
        "RGI62",
        "b_080",
        "L5",
        "RGI60-07",
        "RGI60-07.001.zip",
    )
    if not os.path.isfile(marker):
        import tempfile

        with tempfile.TemporaryDirectory() as wk:
            compat.convert_prepro_to_v2_artifacts(
                DELTA_SERVER,
                rids,
                STREAM_SERVER,
                dataset_tag="oggm_v1.6_2025.6_elev_bands_w5e5_v2",
                workdir=wk,
            )
    return STREAM_SERVER


@pytest.mark.download
def test_live_bremen_zip_probe_falls_back_to_tar(tmp_path):
    """Against the real (v1, tar-only) Bremen sample tree, the new
    .zip probe must fall through to the tar bundles transparently.

    The v2 counterpart (zip probe hits) is
    test_live_bremen_v2_zip_server below.
    """
    from oggm.utils import _downloads

    cfg.initialize()
    cfg.PATHS["working_dir"] = str(tmp_path / "wd")
    os.makedirs(cfg.PATHS["working_dir"], exist_ok=True)
    cfg.PARAMS["download_url_allowlist"] += [
        "cluster.klima.uni-bremen.de/~ngampierakis/test_bundles/",
    ]
    _downloads._prepro_bundle_format.clear()

    gdirs = workflow.init_glacier_directories(
        ["RGI60-07.00001"],
        from_prepro_level=3,
        prepro_border=80,
        prepro_rgi_version="62",
        prepro_base_url=(
            "https://cluster.klima.uni-bremen.de/~ngampierakis/" "test_bundles/"
        ),
    )
    assert gdirs[0].has_file("gridded_data")


@pytest.mark.download
def test_live_bremen_v2_zip_server(tmp_path):
    """TestV2Server against the real v2 zip tree on Bremen (no
    monkeypatch): the .zip probe hits live, L4 layers from two real
    fetches into a functional directory, and the streaming peek reads
    a manifest out of the remote bundle via HTTP ranges.
    """
    from oggm import tasks
    from oggm.utils import _downloads
    from oggm.utils._downloads import peek_remote_manifest

    base_url = "https://cluster.klima.uni-bremen.de/~ngampierakis/test_stream/"
    cfg.initialize()
    cfg.PATHS["working_dir"] = str(tmp_path / "wd")
    os.makedirs(cfg.PATHS["working_dir"], exist_ok=True)
    cfg.PARAMS["download_url_allowlist"] += [
        "cluster.klima.uni-bremen.de/~ngampierakis/test_stream/",
    ]
    _downloads._prepro_bundle_format.clear()

    rid = "RGI60-07.00001"
    gdirs = workflow.init_glacier_directories(
        [rid],
        from_prepro_level=4,
        prepro_border=80,
        prepro_rgi_version="62",
        prepro_base_url=base_url,
    )
    gdir = gdirs[0]
    # the probe confirmed zip bundles live (no tar fallback)
    assert "zip" in _downloads._prepro_bundle_format.values()
    assert gdir.has_file("gridded_data")
    assert gdir.has_file("climate_historical")
    fls = gdir.read_store("model_flowlines")
    assert len(fls) >= 1

    tasks.run_from_climate_data(
        gdir, ys=2004, ye=2006, output_filesuffix="_v2live"
    )
    with xr.open_dataset(
        gdir.get_filepath("model_diagnostics", filesuffix="_v2live")
    ) as ds:
        assert ds.volume_m3.isel(time=-1) > 0

    # streaming: one ranged read of the L3 manifest, no download
    m = peek_remote_manifest(
        base_url + "RGI62/b_080/L3/RGI60-07/RGI60-07.000.zip", rid, 3
    )
    assert m["format_version"] == 2


class TestV2Server:
    """End-to-end: init_glacier_directories against a v2 zip server."""

    BASE_URL = (
        "https://cluster.klima.uni-bremen.de/~ngampierakis/test_gdirs/"
        "oggm_v1.6/2026.7/elev_bands/W5E5/"
    )

    @pytest.fixture
    def served_calls(self, stream_server_tree, monkeypatch, tmp_path):
        from oggm.utils import _downloads

        cfg.initialize()
        calls = []

        def fake_file_downloader(www_path, **kwargs):
            local = os.path.join(
                stream_server_tree, www_path.replace(self.BASE_URL, "")
            )
            if not os.path.isfile(local):
                return None
            calls.append(www_path)
            return local

        monkeypatch.setattr(_downloads, "file_downloader", fake_file_downloader)
        monkeypatch.setattr(_downloads, "_prepro_bundle_format", {})
        wd = str(tmp_path / "wd")
        os.makedirs(wd, exist_ok=True)
        cfg.PATHS["working_dir"] = wd
        cfg.PARAMS["has_internet"] = False
        cfg.PARAMS["border"] = 80
        cfg.PARAMS["rgi_version"] = "62"
        return calls

    def test_init_level4_and_run(self, served_calls):
        from oggm import tasks

        calls = served_calls
        rid = "RGI60-07.00001"
        gdirs = workflow.init_glacier_directories(
            [rid],
            from_prepro_level=4,
            prepro_border=80,
            prepro_base_url=self.BASE_URL,
        )
        gdir = gdirs[0]
        # L4 delta + the L3 materialisation it requires, all zip
        assert len(calls) == 2
        assert calls[0].endswith(".zip") and "/L4/" in calls[0]
        assert calls[1].endswith(".zip") and "/L3/" in calls[1]

        # a functional glacier directory: groups + vectors + run
        assert gdir.has_file("gridded_data")
        assert gdir.has_file("climate_historical")
        gdf = gdir.read_shapefile("outlines")
        assert gdf.crs is not None
        fls = gdir.read_store("model_flowlines")
        assert len(fls) >= 1

        tasks.run_from_climate_data(
            gdir, ys=2004, ye=2006, output_filesuffix="_v2run"
        )
        with xr.open_dataset(
            gdir.get_filepath("model_diagnostics", filesuffix="_v2run")
        ) as ds:
            assert ds.volume_m3.isel(time=-1) > 0

    def test_init_level5_single_fetch(self, served_calls):
        calls = served_calls
        rid = "RGI60-07.00100"
        gdirs = workflow.init_glacier_directories(
            [rid],
            from_prepro_level=5,
            prepro_border=80,
            prepro_base_url=self.BASE_URL,
        )
        assert len(calls) == 1
        assert calls[0].endswith(".zip") and "/L5/" in calls[0]
        fls = gdirs[0].read_store("model_flowlines")
        assert len(fls) >= 1


# --- step 7 (Phase B): streaming mock over HTTP ranges ---


@pytest.fixture(scope="class")
def range_server(request, stream_server_tree):
    """A local HTTP server with Range support over ext/data/stream_server.

    SimpleHTTPRequestHandler does NOT support Range, so this implements
    just enough: HEAD (size discovery) and single-range GET, recording
    response statuses and bytes served.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    root = stream_server_tree
    stats = {"statuses": [], "bytes": 0, "ranges": []}

    class RangeHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _path(self):
            fp = os.path.normpath(os.path.join(root, self.path.lstrip("/")))
            return fp if fp.startswith(os.path.normpath(root)) else None

        def do_HEAD(self):
            fp = self._path()
            if not (fp and os.path.isfile(fp)):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(os.path.getsize(fp)))
            self.end_headers()

        def do_GET(self):
            fp = self._path()
            if not (fp and os.path.isfile(fp)):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                stats["statuses"].append(404)
                return
            size = os.path.getsize(fp)
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start_s, _, end_s = rng[6:].partition("-")
                start = int(start_s) if start_s else 0
                end = min(int(end_s) if end_s else size - 1, size - 1)
                with open(fp, "rb") as f:
                    f.seek(start)
                    body = f.read(end - start + 1)
                self.send_response(206)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                stats["statuses"].append(206)
                stats["bytes"] += len(body)
                stats["ranges"].append((self.path, start, end))
            else:
                with open(fp, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                stats["statuses"].append(200)
                stats["bytes"] += len(body)
                stats["ranges"].append((self.path, 0, size - 1))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield url, stats
    finally:
        server.shutdown()
        server.server_close()


class TestStreaming:
    """Phase B: lazy remote reads via HTTP ranges (mock server)."""

    BUNDLE = "RGI62/b_080/L3/RGI60-07/RGI60-07.000.zip"
    RID = "RGI60-07.00001"

    def test_peek_remote_manifest(self, range_server):
        from oggm.utils._downloads import peek_remote_manifest

        url, stats = range_server
        m = peek_remote_manifest(f"{url}/{self.BUNDLE}", self.RID, 3)
        assert m["format_version"] == 2
        assert m["includes_levels"] == [0, 1, 2, 3]
        assert m["rgi_id"] == self.RID
        # missing level -> None, not a crash
        assert peek_remote_manifest(f"{url}/{self.BUNDLE}", self.RID, 9) is None

    def test_open_remote_group_streams_ranges(
        self, range_server, tmp_path, stream_server_tree
    ):
        from oggm.utils._downloads import open_remote_group

        url, stats = range_server
        stats["statuses"].clear()
        stats["bytes"] = 0

        with open_remote_group(
            f"{url}/{self.BUNDLE}", self.RID, "climate_historical"
        ) as ds:
            remote_temp = ds["temp"].values
            remote_ref_hgt = ds.attrs["ref_hgt"]

        # ground truth: extract the same member locally
        member = os.path.join(
            stream_server_tree,
            "RGI62",
            "b_080",
            "L3",
            "RGI60-07",
            "RGI60-07.000",
            self.RID + ".zip",
        )
        to_dir = str(tmp_path / self.RID)
        utils.robust_archive_extract(member, to_dir)
        ref = xr.open_zarr(
            os.path.join(to_dir, "data_store.zarr"),
            group="climate_historical",
            consolidated=True,
        )
        assert_allclose(remote_temp, ref["temp"].values)
        assert remote_ref_hgt == ref.attrs["ref_hgt"]

        # streaming proof: ranged responses only, and far fewer bytes
        # than the whole bundle
        bundle_size = os.path.getsize(
            os.path.join(stream_server_tree, self.BUNDLE)
        )
        assert 206 in stats["statuses"]
        assert 200 not in stats["statuses"]
        assert stats["bytes"] < bundle_size / 2

    def test_peek_remote_manifest_direct_ranges(self, range_server):
        # Phase C: the peek also fetches each byte region at most once
        from oggm.utils._downloads import peek_remote_manifest

        url, stats = range_server
        stats["statuses"].clear()
        stats["ranges"].clear()

        m = peek_remote_manifest(f"{url}/{self.BUNDLE}", self.RID, 3)
        assert m["rgi_id"] == self.RID

        assert 206 in stats["statuses"]
        assert 200 not in stats["statuses"]
        spans = sorted((s, e) for _, s, e in stats["ranges"])
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start > prev_end, f"overlapping range fetches: {spans}"

    def test_remote_bundle_store_direct_ranges(
        self, range_server, tmp_path, stream_server_tree
    ):
        # Phase C: one group open resolves each byte region at most once
        # (the fsspec chain re-read the outer central directory ~3x).
        from oggm.utils._downloads import open_remote_group

        url, stats = range_server
        stats["statuses"].clear()
        stats["ranges"].clear()
        stats["bytes"] = 0

        with open_remote_group(
            f"{url}/{self.BUNDLE}", self.RID, "climate_historical"
        ) as ds:
            remote_temp = ds["temp"].values

        member = os.path.join(
            stream_server_tree,
            "RGI62",
            "b_080",
            "L3",
            "RGI60-07",
            "RGI60-07.000",
            self.RID + ".zip",
        )
        to_dir = str(tmp_path / "direct" / self.RID)
        utils.robust_archive_extract(member, to_dir)
        ref = xr.open_zarr(
            os.path.join(to_dir, "data_store.zarr"),
            group="climate_historical",
            consolidated=True,
        )
        assert_allclose(remote_temp, ref["temp"].values)

        # ranged responses only, no full downloads
        assert 206 in stats["statuses"]
        assert 200 not in stats["statuses"]

        # every byte region fetched at most once: requested ranges are
        # pairwise non-overlapping (covers the outer central directory too)
        spans = sorted((s, e) for _, s, e in stats["ranges"])
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start > prev_end, f"overlapping range fetches: {spans}"

        bundle_size = os.path.getsize(
            os.path.join(stream_server_tree, self.BUNDLE)
        )
        assert stats["bytes"] < bundle_size / 2

    def test_remote_bundle_store_byte_ranges(
        self, range_server, tmp_path, stream_server_tree
    ):
        # store seam: partial reads match slices of the real file bytes
        import asyncio
        from zarr.abc.store import (
            RangeByteRequest,
            OffsetByteRequest,
            SuffixByteRequest,
        )
        from zarr.core.buffer import default_buffer_prototype
        from oggm.utils._downloads import RemoteBundleStore

        url, stats = range_server

        # independent truth: the same file extracted locally
        member = os.path.join(
            stream_server_tree,
            "RGI62",
            "b_080",
            "L3",
            "RGI60-07",
            "RGI60-07.000",
            self.RID + ".zip",
        )
        to_dir = str(tmp_path / "bytes" / self.RID)
        utils.robust_archive_extract(member, to_dir)
        key = ".zmetadata"
        with open(os.path.join(to_dir, "data_store.zarr", key), "rb") as f:
            raw = f.read()

        store = RemoteBundleStore(f"{url}/{self.BUNDLE}", self.RID)
        proto = default_buffer_prototype()

        def get(byte_range=None, k=key):
            buf = asyncio.run(store.get(k, proto, byte_range))
            return None if buf is None else buf.to_bytes()

        assert get() == raw
        assert get(RangeByteRequest(3, 10)) == raw[3:10]
        assert get(OffsetByteRequest(5)) == raw[5:]
        assert get(SuffixByteRequest(7)) == raw[-7:]
        assert get(k="not/a/key") is None
