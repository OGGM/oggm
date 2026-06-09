import os
import pickle
from functools import partial

import pytest
import pyproj
import shapely.geometry as shpg
import numpy as np
import xarray as xr
from pathlib import Path
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from oggm import Centerline

salem = pytest.importorskip("salem")
gpd = pytest.importorskip("geopandas")

# Locals
import oggm.cfg as cfg
from oggm.tests.funcs import get_test_dir
import oggm.utils.geozarr as oggmzarr

# Globals
pytestmark = pytest.mark.test_env("workflow")
_TEST_DIR = os.path.join(get_test_dir(), "tmp_workflow")
CLI_LOGF = os.path.join(_TEST_DIR, "clilog.pkl")


def _make_centerline(n=5):
    """Create a minimal Centerline."""
    coords = np.arange(n, dtype=float)
    line = shpg.LineString(np.vstack([coords, np.zeros(n)]).T)
    surface_h = np.linspace(3000.0, 2000.0, n)
    cl = Centerline(
        line,
        dx=1.0,
        surface_h=surface_h,
        orig_head=shpg.Point(0, 0),
        rgi_id="RGI60-11.00897",
        map_dx=100.0,
    )
    cl.order = 1
    cl._widths = np.ones(n)
    cl.is_rectangular = np.zeros(n, dtype=bool)
    cl.is_trapezoid = np.zeros(n, dtype=bool)
    cl.apparent_mb = np.zeros(n)
    cl.flux = np.zeros(n)
    cl.flux_out = 0.0
    return cl


def _make_mixed_bed_flowline(n=5):
    """Create a minimal MixedBedFlowline."""
    from oggm.core.flowline import MixedBedFlowline

    coords = np.arange(n, dtype=float)
    line = shpg.LineString(np.vstack([coords, np.zeros(n)]).T)
    surface_h = np.linspace(3000.0, 2000.0, n)
    bed_h = surface_h  # zero ice thickness
    bed_shape = np.full(n, 3.0e-3)
    is_trapezoid = np.zeros(n, dtype=bool)
    lambdas = np.zeros(n)
    section = np.zeros(n)

    return MixedBedFlowline(
        line=line,
        dx=1.0,
        map_dx=100.0,
        surface_h=surface_h,
        bed_h=bed_h,
        section=section,
        bed_shape=bed_shape,
        is_trapezoid=is_trapezoid,
        lambdas=lambdas,
        widths_m=np.zeros(n) + 10.0,
        rgi_id="RGI60-11.00897",
    )


class TestZarrUtilities:
    """Tests for any Zarr operations called via workflow or _workflow."""

    # File operations

    def test_get_pickle_paths_returns_only_pkl(self, tmp_path):
        for name in ("data.pkl", "other.pkl", "README.txt", "noextension"):
            (tmp_path / name).touch()

        class _MockGDir:
            dir = str(tmp_path)

        paths = oggmzarr.get_pickle_paths(_MockGDir().dir)
        assert all(isinstance(p, Path) for p in paths)
        assert all(str(p).endswith(".pkl") for p in paths)
        assert len(paths) == 2

    def test_get_pickle_paths_empty_dir(self, tmp_path):
        class _MockGDir:
            dir = str(tmp_path)

        assert not oggmzarr.get_pickle_paths(_MockGDir().dir)

    def test_get_pickle_data_reads_dict_pickle(self, tmp_path):
        payload = {"array": np.array([1.0, 2.0]), "val": 42}
        with open(tmp_path / "mydata.pkl", "wb") as f:
            pickle.dump(payload, f)

        class _MockGDir:
            dir = str(tmp_path)

            def read_pickle(self, stem):
                with open(os.path.join(self.dir, stem + ".pkl"), "rb") as fh:
                    return pickle.load(fh)

        result = oggmzarr.get_pickle_data([Path("mydata.pkl")], _MockGDir())
        assert "mydata" in result
        assert_allclose(result["mydata"]["array"], payload["array"])
        assert result["mydata"]["val"] == 42

    def test_get_pickle_data_type_only(self, tmp_path):
        payload = {"array": np.array([1.0, 2.0]), "val": 42}
        with open(tmp_path / "mydata.pkl", "wb") as f:
            pickle.dump(payload, f)

        class _MockGDir:
            dir = str(tmp_path)

            def read_pickle(self, stem):
                with open(os.path.join(self.dir, stem + ".pkl"), "rb") as fh:
                    return pickle.load(fh)

        result = oggmzarr.get_pickle_data(
            [Path("mydata.pkl")], _MockGDir(), type_only=True
        )
        assert result["mydata"]["array"] is np.ndarray
        assert result["mydata"]["val"] is int

    def test_get_pickle_data_reads_list_of_dicts(self, tmp_path):
        payload = [{"a": 1, "b": np.array([3.0])}, {"c": 2}]
        with open(tmp_path / "listdata.pkl", "wb") as f:
            pickle.dump(payload, f)

        class _MockGDir:
            dir = str(tmp_path)

            def read_pickle(self, stem):
                with open(os.path.join(self.dir, stem + ".pkl"), "rb") as fh:
                    return pickle.load(fh)

        result = oggmzarr.get_pickle_data([Path("listdata.pkl")], _MockGDir())
        assert "listdata" in result
        # Each dict in the list is processed through get_tranche
        assert isinstance(result["listdata"], list)
        assert result["listdata"][0]["a"] == 1

    def test_get_tranche_returns_values(self):
        assert oggmzarr.get_tranche({}) == {}
        data = {"a": 1, "b": "hello", "c": np.array([1, 2, 3])}
        result = oggmzarr.get_tranche(data, type_only=False)
        assert result["a"] == 1
        assert result["b"] == "hello"
        assert_allclose(result["c"], data["c"])

    def test_get_tranche_returns_types(self):
        data = {"a": 1, "b": "hello", "c": np.array([1, 2, 3])}
        result = oggmzarr.get_tranche(data, type_only=True)
        assert result["a"] is int
        assert result["b"] is str
        assert result["c"] is np.ndarray

    def test_filter_arrays_from_dict(self):
        assert oggmzarr.filter_arrays_from_dict({"x": 1, "y": "z"}) == {}
        data = {
            "array": np.array([1, 2, 3]),
            "scalar": 42,
            "string": "hello",
            "lst": [1, 2],
        }
        result = oggmzarr.filter_arrays_from_dict(data)
        assert set(result.keys()) == {"array"}
        assert_allclose(result["array"], data["array"])

    def test_filter_lists_from_dict(self):
        assert (
            oggmzarr.filter_lists_from_dict({"x": 1, "y": np.array([1])}) == {}
        )
        data = {
            "arr": np.array([1, 2, 3]),
            "scalar": 42,
            "lst": [1, 2],
            "tup": (3, 4),
        }
        result = oggmzarr.filter_lists_from_dict(data)
        assert set(result.keys()) == {"lst"}
        assert result["lst"] == [1, 2]

    # Downstream line

    def test_get_downstream_line_from_pkl_convert_linestring(self):
        line = shpg.LineString([(0, 0), (1, 1), (2, 0)])
        data = {"downstream_line": line, "extra": 99}
        result = oggmzarr.get_downstream_line_from_pkl(data)
        assert isinstance(result["downstream_line"], xr.DataArray)
        assert result["extra"] == 99
        expected = np.array(shpg.mapping(line)["coordinates"])
        assert_allclose(result["downstream_line"].values, expected)

    def test_get_downstream_line_from_pkl_skip_non_linestring(self):
        array = np.array([1.0, 2.0, 3.0])
        data = {"downstream_line": array}
        result = oggmzarr.get_downstream_line_from_pkl(data)
        assert_allclose(result["downstream_line"], array)

    def test_get_downstream_line_from_pkl_errors(self):
        with pytest.raises(TypeError):
            oggmzarr.get_downstream_line_from_pkl("not a dict")
        with pytest.raises(KeyError):
            oggmzarr.get_downstream_line_from_pkl({"other_key": 42})

    # Inversion flowlines

    def test_get_inversion_flowlines_extracts_expected_keys(self):
        cl = _make_centerline()
        result = oggmzarr.get_inversion_flowlines_from_pkl([cl])
        assert len(result) == 1
        data = result[0]
        for key in (
            "line",
            "dx",
            "surface_h",
            "orig_head",
            "rgi_id",
            "map_dx",
            "order",
            "_widths",
            "is_rectangular",
            "is_trapezoid",
            "apparent_mb",
            "flux",
            "flux_out",
        ):
            assert key in data, f"Expected key '{key}' missing from result"

    def test_get_inversion_flowlines_correct_values(self):
        assert oggmzarr.get_inversion_flowlines_from_pkl([]) == []
        cl = _make_centerline(n=7)
        result = oggmzarr.get_inversion_flowlines_from_pkl([cl])
        data = result[0]
        assert_allclose(data["surface_h"], cl.surface_h)
        assert data["rgi_id"] == "RGI60-11.00897"
        assert data["dx"] == 1.0
        assert data["map_dx"] == 100.0
        assert data["order"] == 1

    def test_get_inversion_flowlines_raises_on_non_centerline(self):
        with pytest.raises(TypeError):
            oggmzarr.get_inversion_flowlines_from_pkl(["not a centerline"])

    # Datacube operations

    def test_add_datacube_adds_group(self):
        dt = xr.DataTree()
        datacubes = {"var": xr.DataArray([1.0, 2.0])}
        result = oggmzarr.add_datacube(dt, datacubes, "group_01")
        assert "group_01" in result.children

        dt = xr.DataTree()
        datacubes = {"var": xr.DataArray([1.0, 2.0])}
        dt = oggmzarr.add_datacube(dt, datacubes, "group_01")
        dt = oggmzarr.add_datacube(dt, datacubes, "group_01", overwrite=True)
        assert "group_01" in dt.children

        dt = oggmzarr.add_datacube(dt, {"b": xr.DataArray([2.0])}, "group_02")
        assert "group_01" in dt.children
        assert "group_02" in dt.children

    def test_add_datacube_raises_on_non_dict_datacubes(self):
        dt = xr.DataTree()
        with pytest.raises(ValueError, match="dictionary"):
            oggmzarr.add_datacube(dt, [1, 2, 3], "group_01")

        dt = xr.DataTree()
        datacubes = {"var": xr.DataArray([1.0, 2.0])}
        dt = oggmzarr.add_datacube(dt, datacubes, "group_01")
        with pytest.raises(ValueError, match="already exists"):
            oggmzarr.add_datacube(dt, datacubes, "group_01", overwrite=False)

    # Conversion

    def test_convert_pickles_to_datatree_downstream_line(self):
        line = shpg.LineString([(0, 0), (1, 1), (2, 0)])
        pickle_data = {"downstream_line": {"downstream_line": line}}
        result = oggmzarr.convert_pickles_to_datatree(pickle_data)
        assert isinstance(result, xr.DataTree)
        assert "downstream_line" in result.children

    def test_convert_pickles_to_datatree_inversion_flowlines(self):
        cl = _make_centerline()
        pickle_data = {"inversion_flowlines": [cl]}
        result = oggmzarr.convert_pickles_to_datatree(pickle_data)
        assert isinstance(result, xr.DataTree)
        assert "inversion_flowlines" in result.children

    def test_convert_pickles_to_datatree_generic(self):
        pickle_data = {"mydata": {"key": xr.DataArray([1.0, 2.0, 3.0])}}
        result = oggmzarr.convert_pickles_to_datatree(pickle_data)
        assert isinstance(result, xr.DataTree)
        assert "mydata" in result.children

        pickle_data = {"unsupported": 42}
        result = oggmzarr.convert_pickles_to_datatree(pickle_data)
        assert isinstance(result, xr.DataTree)
        assert "unsupported" not in result.children

        result = oggmzarr.convert_pickles_to_datatree({})
        assert isinstance(result, xr.DataTree)
        assert len(result.children) == 0

    def test_convert_linestring_to_dataarray(self):
        line = shpg.LineString([(0, 0), (1, 1), (2, 0)])
        expected = np.array(shpg.mapping(line)["coordinates"])
        result = oggmzarr.convert_linestring_to_dataarray(line)
        assert isinstance(result, xr.DataArray)
        assert_allclose(result.values, expected)
        assert set(result.dims) == {"x", "y"}

    def test_get_datatree_value(self):
        ds = xr.Dataset({"surface_h": xr.DataArray([1.0, 2.0, 3.0])})
        dt = xr.DataTree(dataset=ds)
        result = oggmzarr.get_datatree_value(dt, "surface_h")
        assert_allclose(result, [1.0, 2.0, 3.0])

        # returns None for missing attribute, empty child
        dt = xr.DataTree()
        result = oggmzarr.get_datatree_value(dt, "nonexistent")
        assert result is None
        dt = xr.DataTree()
        dt["child"] = xr.DataTree()
        result = oggmzarr.get_datatree_value(dt, "child")
        assert result is None

    # get_dict_from_datatree

    def test_get_dict_from_datatree(self):

        # empties
        dt = xr.DataTree()
        result = oggmzarr.get_dict_from_datatree(dt)
        assert result == {}
        dt = xr.DataTree()
        dt["child"] = xr.DataTree()
        result = oggmzarr.get_dict_from_datatree(dt)
        assert "child" in result
        assert result["child"] is None

        array = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])
        dt = xr.DataTree(dataset=xr.Dataset({"flux": array}))
        result = oggmzarr.get_dict_from_datatree(dt)
        assert "flux" in result
        assert_allclose(result["flux"], [1.0, 2.0, 3.0])

        array = xr.DataArray([10.0, 20.0], dims=["x"], coords={"x": [0, 1]})
        dt = xr.DataTree(dataset=xr.Dataset({"flux": array}))
        result = oggmzarr.get_dict_from_datatree(dt)
        assert "x" in result
        assert_allclose(result["x"], [0, 1])

    def test_restore_projection_converts_dict_to_proj(self):
        crs = pyproj.CRS.from_epsg(32632)
        dt = xr.DataTree()
        dt.attrs["pyproj_srs"] = crs.to_json_dict()
        assert isinstance(dt.attrs["pyproj_srs"], dict)
        oggmzarr.restore_projection(dt)
        assert isinstance(dt.attrs["pyproj_srs"], pyproj.Proj)

        dt = xr.DataTree()
        dt.attrs["other"] = "value"
        oggmzarr.restore_projection(dt)
        assert "pyproj_srs" not in dt.attrs

        proj = pyproj.Proj("epsg:32632")
        dt = xr.DataTree()
        dt.attrs["pyproj_srs"] = proj
        oggmzarr.restore_projection(dt)
        assert dt.attrs["pyproj_srs"] == proj

    def test_get_grid_params_from_partial(self):
        proj = pyproj.Proj("epsg:32632")
        grid = salem.Grid(
            proj=proj,
            nxny=(10.0, 8.0),
            dxdy=(200.0, 100.0),
            x0y0=(500.0, 300.0),
            pixel_ref="center",
        )
        p = partial(grid.ij_to_crs, crs=salem.wgs84)
        result = oggmzarr.get_grid_params_from_partial(p)
        for key in ("pyproj_srs", "nxny", "dxdy", "x0y0", "pixel_ref"):
            assert key in result, f"Expected key '{key}' missing"

        assert result["nxny"] == (10.0, 8.0)
        assert result["dxdy"] == (200.0, 100.0)
        assert result["x0y0"] == (500.0, 300.0)
        assert result["pixel_ref"] == "center"
        assert isinstance(result["pyproj_srs"], dict)

    def test_get_map_trafo_from_grid(self):
        # Test with data tree
        proj = pyproj.Proj("epsg:32632")
        dt = xr.DataTree()
        dt.attrs["pyproj_srs"] = proj
        dt.attrs["nxny"] = (10.0, 8.0)
        dt.attrs["dxdy"] = (200.0, 100.0)
        dt.attrs["x0y0"] = (500.0, 300.0)
        dt.attrs["pixel_ref"] = "center"
        result = oggmzarr.get_map_trafo_from_grid(dt)
        assert callable(result)
        result = oggmzarr.get_grid_params_from_partial(result)

        assert result["nxny"] == (10.0, 8.0)
        assert result["dxdy"] == (200.0, 100.0)
        assert result["x0y0"] == (500.0, 300.0)
        assert result["pixel_ref"] == "center"
        assert isinstance(result["pyproj_srs"], dict)

        grid = salem.Grid(
            proj=proj,
            nxny=(10.0, 8.0),
            dxdy=(200.0, 100.0),
            x0y0=(500.0, 300.0),
            pixel_ref="center",
        )
        # reconstruction
        p = partial(grid.ij_to_crs, crs=salem.wgs84)
        params = oggmzarr.get_grid_params_from_partial(p)

        dt = xr.DataTree()
        dt.attrs["pyproj_srs"] = proj
        dt.attrs["nxny"] = params["nxny"]
        dt.attrs["dxdy"] = params["dxdy"]
        dt.attrs["x0y0"] = params["x0y0"]
        dt.attrs["pixel_ref"] = params["pixel_ref"]
        result = oggmzarr.get_map_trafo_from_grid(dt)
        assert callable(result)

    def test_get_model_flowlines(self):
        result = oggmzarr.get_model_flowlines_from_pkl([])
        assert not result and isinstance(result, list)

        fl = _make_mixed_bed_flowline()
        result = oggmzarr.get_model_flowlines_from_pkl([fl])
        assert len(result) == 1
        data = result[0]
        for key in (
            "line",
            "dx",
            "map_dx",
            "surface_h",
            "bed_h",
            "section",
            "bed_shape",
            "is_trapezoid",
            "lambdas",
            "rgi_id",
        ):
            assert key in data, f"Expected key '{key}' missing"
        assert isinstance(result[0]["line"], xr.DataArray)

        fl = _make_mixed_bed_flowline(n=7)
        result = oggmzarr.get_model_flowlines_from_pkl([fl])
        data = result[0]
        assert_allclose(data["surface_h"], fl.surface_h)
        assert data["rgi_id"] == "RGI60-11.00897"
        assert data["dx"] == 1.0
        assert data["map_dx"] == 100.0

        cl = _make_centerline()
        with pytest.raises(TypeError):
            oggmzarr.get_model_flowlines_from_pkl([cl])

    def test_get_pickle_data(self, tmp_path):
        payload = 42
        with open(tmp_path / "scalar.pkl", "wb") as f:
            pickle.dump(payload, f)

        class _MockGDir:
            dir = str(tmp_path)

            def read_pickle(self, stem):
                with open(os.path.join(self.dir, stem + ".pkl"), "rb") as fh:
                    return pickle.load(fh)

        result = oggmzarr.get_pickle_data([Path("scalar.pkl")], _MockGDir())
        assert "scalar" not in result
