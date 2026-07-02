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


def _flowline_base_kwargs(n=5):
    """Common kwargs for a minimal Flowline with non-zero thickness."""
    coords = np.arange(n, dtype=float)
    line = shpg.LineString(np.vstack([coords, np.zeros(n)]).T)
    surface_h = np.linspace(3000.0, 2000.0, n)
    bed_h = surface_h - 50.0  # 50 m thick everywhere
    return dict(
        line=line,
        dx=1.0,
        map_dx=100.0,
        surface_h=surface_h,
        bed_h=bed_h,
        rgi_id="RGI60-11.00897",
    )


def _make_parabolic_flowline(n=5):
    from oggm.core.flowline import ParabolicBedFlowline

    return ParabolicBedFlowline(
        bed_shape=np.full(n, 3.0e-3), **_flowline_base_kwargs(n)
    )


def _make_rectangular_flowline(n=5):
    from oggm.core.flowline import RectangularBedFlowline

    return RectangularBedFlowline(
        widths=np.full(n, 2.0), **_flowline_base_kwargs(n)
    )


def _make_trapezoidal_flowline(n=5):
    from oggm.core.flowline import TrapezoidalBedFlowline

    return TrapezoidalBedFlowline(
        widths=np.full(n, 5.0), lambdas=np.full(n, 1.0),
        **_flowline_base_kwargs(n)
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


    """Round-trip ``geometries`` (polygons with holes, MultiPolygons,
    catchment_indices) through the zarr conversion helpers."""

    @pytest.fixture(autouse=True)
    def _holed_polygon(self):
        """A Polygon with a single interior hole (nunatak)."""
        exterior = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
        hole = [(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)]
        return shpg.Polygon(exterior, [hole])

    @pytest.fixture(autouse=True)
    def _multipolygon_two_holes(self):
        """A MultiPolygon whose first part has more than one hole."""
        exterior = [(20, 20), (20, 30), (30, 30), (30, 20), (20, 20)]
        hole_a = [(21, 21), (21, 23), (23, 23), (23, 21), (21, 21)]
        hole_b = [(25, 25), (25, 27), (27, 27), (27, 25), (25, 25)]
        part_a = shpg.Polygon(exterior, [hole_a, hole_b])
        part_b = shpg.Polygon(
            [(40, 40), (40, 45), (45, 45), (45, 40), (40, 40)]
        )
        return shpg.MultiPolygon([part_a, part_b])

    def test_geometries_roundtrip_in_memory(
        self, _holed_polygon, _multipolygon_two_holes
    ):
        poly_hr = _holed_polygon
        poly_pix = _multipolygon_two_holes
        cis = [
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.zeros((0, 2), dtype=np.int64),  # empty catchment
            np.array([[7, 8]]),
        ]
        geom = {
            "polygon_hr": poly_hr,
            "polygon_pix": poly_pix,
            "polygon_area": 123.45,
            "catchment_indices": cis,
        }

        data_tree = oggmzarr.convert_pickles_to_datatree({"geometries": geom})
        assert "geometries" in data_tree.children
        node = data_tree["geometries"]
        # polygons + catchment_indices are child groups, area is a root var
        assert set(node.children) == {
            "polygon_hr",
            "polygon_pix",
            "catchment_indices",
        }
        assert "polygon_area" in node.data_vars

        result = oggmzarr.get_geometries_from_datatree(node)

        # exterior + interior coordinates preserved
        assert result["polygon_hr"].equals(poly_hr)
        assert len(list(result["polygon_hr"].interiors)) == 1
        assert result["polygon_pix"].equals(poly_pix)
        assert result["polygon_pix"].geom_type == "MultiPolygon"

        # the holed MultiPolygon part keeps both holes
        assert len(list(result["polygon_pix"].geoms[0].interiors)) == 2
        assert isinstance(result["polygon_area"], float)
        assert result["polygon_area"] == 123.45
        for got, exp in zip(result["catchment_indices"], cis):
            assert_allclose(got, exp)
            assert got.shape == exp.reshape(-1, 2).shape

    def test_geometries_write_read_store_on_disk(
        self, tmp_path, hef_gdir, _holed_polygon, _multipolygon_two_holes
    ):
        """write_store must store polygon interiors+exteriors to a real
        zarr file on disk, and read_store must reconstruct them."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir

        # Real geometries are shapely polygons read from the store.
        geom = dict(gdir.read_store("geometries"))
        assert isinstance(geom["polygon_hr"], (shpg.Polygon, shpg.MultiPolygon))

        # Guarantee hole / multipolygon coverage regardless of the glacier.
        geom["polygon_hr"] = _holed_polygon
        geom["polygon_pix"] = _multipolygon_two_holes
        geom["polygon_area"] = float(geom["polygon_hr"].area)
        geom["catchment_indices"] = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6]]),
        ]

        gdir.write_store(geom, "geometries", filesuffix="zarrtest")

        # Check it's really zarr, group and flattened ring arrays exist
        zarr_fp = gdir.get_filepath("data_store").replace(".pkl", ".zarr")
        group_dir = os.path.join(zarr_fp, "geometries_zarrtest")
        assert os.path.isdir(group_dir)
        assert os.path.isdir(os.path.join(group_dir, "polygon_hr", "vertices"))

        back = gdir.read_store("geometries", filesuffix="zarrtest")
        assert back["polygon_hr"].equals(geom["polygon_hr"])
        assert len(list(back["polygon_hr"].interiors)) == 1
        assert back["polygon_pix"].equals(geom["polygon_pix"])
        assert len(list(back["polygon_pix"].geoms[0].interiors)) == 2
        assert_allclose(back["polygon_area"], geom["polygon_area"])
        for got, exp in zip(
            back["catchment_indices"], geom["catchment_indices"]
        ):
            assert_allclose(got, exp)


def _write_datatree_to_zarr(data_tree, fp):
    """Mimic write_zarr's per-node write (no gdir needed)."""
    import zarr

    for node in data_tree.subtree:
        ds = node.ds
        if ds is None or (len(ds.data_vars) == 0 and len(ds.coords) == 0):
            continue
        ds.to_zarr(
            fp,
            group=(node.path.lstrip("/") or None),
            mode="a",
            zarr_format=2,
            consolidated=False,
        )
    zarr.consolidate_metadata(fp)


def _reconstruct_downstream_line(data_tree):
    """Replicate the _validate_store downstream_line branch."""
    out = oggmzarr.get_dict_from_datatree(data_tree)
    if "downstream_line" in out:
        out["downstream_line"] = oggmzarr._validate_linestring(
            out["downstream_line"]
        )
    out["full_line"] = oggmzarr._validate_linestring(out.get("full_line"))
    return out


class TestDownstreamLineZarr:
    """downstream_line must round-trip to zarr, including a full_line
    LineString of a different length than downstream_line."""

    def test_downstream_line_with_full_line_on_disk(self, tmp_path):
        dline = shpg.LineString([(0, 0), (1, 1), (2, 2)])
        # Deliberately a different length than downstream_line.
        lline = shpg.LineString([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        out = {
            "full_line": lline,
            "downstream_line": dline,
            "bedshapes": np.arange(3.0),
            "surface_h": np.arange(3.0),
        }

        data_tree = oggmzarr.convert_pickles_to_datatree(
            {"downstream_line": out}
        )
        fp = os.path.join(str(tmp_path), "data_store.zarr")
        _write_datatree_to_zarr(data_tree, fp)

        back = xr.open_datatree(
            fp, group="downstream_line", engine="zarr", consolidated=True
        )
        result = _reconstruct_downstream_line(back)
        assert result["downstream_line"].equals(dline)
        assert result["full_line"].equals(lline)
        assert_allclose(result["bedshapes"], out["bedshapes"])

    def test_downstream_line_full_line_none(self, tmp_path):
        dline = shpg.LineString([(0, 0), (1, 1), (2, 2)])
        out = {"full_line": None, "downstream_line": dline}

        data_tree = oggmzarr.convert_pickles_to_datatree(
            {"downstream_line": out}
        )
        fp = os.path.join(str(tmp_path), "data_store.zarr")
        _write_datatree_to_zarr(data_tree, fp)

        back = xr.open_datatree(
            fp, group="downstream_line", engine="zarr", consolidated=True
        )
        result = _reconstruct_downstream_line(back)
        assert result["downstream_line"].equals(dline)
        assert result["full_line"] is None


def _reconstruct_model_flowlines(data_tree):
    """Replicate the _validate_store model_flowline branch (numbered
    child groups -> list of reconstructed Flowlines)."""
    keys = sorted(data_tree.children, key=int)
    return [oggmzarr.get_flowline_from_datatree(data_tree[k]) for k in keys]


class TestModelFlowlineZarr:
    """model_flowlines must round-trip to zarr for every Flowline subclass,
    not just MixedBedFlowline.

    Uses a function-level round-trip (convert -> on-disk zarr -> reconstruct)
    rather than the hef_gdir fixture: the public write_store/read_store path
    for a non-Mixed flowline is already exercised end-to-end by
    test_prepro.py::TestPyGEM_compat::test_flowlines_from_gmip_data.
    """

    @pytest.mark.parametrize(
        "make_fl, cls_name",
        [
            (_make_parabolic_flowline, "ParabolicBedFlowline"),
            (_make_rectangular_flowline, "RectangularBedFlowline"),
            (_make_trapezoidal_flowline, "TrapezoidalBedFlowline"),
            (_make_mixed_bed_flowline, "MixedBedFlowline"),
        ],
    )
    def test_model_flowline_roundtrip_on_disk(
        self, tmp_path, make_fl, cls_name
    ):
        cfg.initialize()
        fl = make_fl()

        data_tree = oggmzarr.convert_pickles_to_datatree(
            {"model_flowlines": [fl]}
        )
        # The concrete class is recorded as a node attr (not MixedBedFlowline).
        assert data_tree["model_flowlines"]["0"].attrs["_flowline_class"] == (
            cls_name
        )

        fp = os.path.join(str(tmp_path), "data_store.zarr")
        _write_datatree_to_zarr(data_tree, fp)

        back = xr.open_datatree(
            fp, group="model_flowlines", engine="zarr", consolidated=True
        )
        fls = _reconstruct_model_flowlines(back)
        assert len(fls) == 1
        rfl = fls[0]
        assert type(rfl).__name__ == cls_name
        assert_allclose(rfl.surface_h, fl.surface_h)
        assert_allclose(rfl.bed_h, fl.bed_h)
        assert_allclose(rfl.widths_m, fl.widths_m)
        assert_allclose(rfl.section, fl.section)
        assert_allclose(rfl.area_m2, fl.area_m2)
        assert_allclose(rfl.volume_m3, fl.volume_m3)
