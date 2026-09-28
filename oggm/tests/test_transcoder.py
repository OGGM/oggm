import datetime
import os
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pyproj
import pytest
import shapely
import shapely.geometry as shpg
from numpy.testing import assert_allclose

from oggm import Centerline
from oggm.core.flowline import Flowline
from oggm.utils._compat import convert_pickles_to_npz

salem = pytest.importorskip("salem")

# Locals
import oggm.cfg as cfg
import oggm.utils.transcoder as transcoder

# Globals
pytestmark = pytest.mark.test_env("workflow")


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


def _make_junction_pair():
    """Create a tributary joining a twelve-vertex trunk at vertex one.

    The junction sits inside the band that ``check_tail=True`` clips,
    as ``compute_centerlines`` allows with ``check_tail=False``.
    """
    trunk = _make_centerline(12)
    tributary = _make_centerline(3)
    tributary.set_line(shpg.LineString([(1, 2), (1, 1), (1, 0)]))
    tributary.set_flows_to(trunk, check_tail=False)
    return tributary, trunk


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
        widths=np.full(n, 5.0),
        lambdas=np.full(n, 1.0),
        **_flowline_base_kwargs(n),
    )


class TestNpzCodec:
    """Round trips through the codec behind the npz data store."""

    def test_generic_dict_roundtrip(self):
        """A dict of arrays and scalars survives an encode/decode cycle."""
        data = {
            "flux": np.arange(3, dtype=np.float64),
            "dx": 1.5,
            "rgi_id": "RGI60-11.00897",
            "missing": None,
        }

        arrays, meta = transcoder.encode_npz(data, "inversion_input")
        back = transcoder.decode_npz(arrays, meta, "inversion_input")

        assert set(back) == set(data)
        assert_allclose(back["flux"], data["flux"])
        assert back["dx"] == 1.5
        assert back["rgi_id"] == "RGI60-11.00897"
        assert back["missing"] is None

    def test_scalar_types_are_preserved(self):
        """Python and numpy scalars come back as the type they went in as."""
        data = {
            "py_int": 3,
            "py_float": 0.1,
            "py_bool": True,
            "np_float32": np.float32(0.1),
            "np_int64": np.int64(7),
            "np_bool": np.bool_(False),
        }

        arrays, meta = transcoder.encode_npz(data)
        back = transcoder.decode_npz(arrays, meta)

        for key, expected in data.items():
            assert type(back[key]) is type(expected), key
            assert back[key] == expected, key

    def test_nested_lists_and_tuples_roundtrip(self):
        """Nested containers keep their structure and their type."""
        data = [
            {"width": np.ones(2), "nested": {"depth": 2}},
            {"width": np.zeros(3), "shape": (4, 5)},
        ]

        arrays, meta = transcoder.encode_npz(data, "inversion_output")
        back = transcoder.decode_npz(arrays, meta, "inversion_output")

        assert isinstance(back, list) and len(back) == 2
        assert_allclose(back[0]["width"], np.ones(2))
        assert back[0]["nested"] == {"depth": 2}
        assert back[1]["shape"] == (4, 5)

    def test_linestring_and_point_roundtrip(self):
        """Shapely lines and points come back as shapely objects."""
        data = {
            "downstream_line": shpg.LineString([(0, 0), (1, 1), (2, 4)]),
            "orig_head": shpg.Point(3, 4),
        }

        arrays, meta = transcoder.encode_npz(data, "downstream_line")
        back = transcoder.decode_npz(arrays, meta, "downstream_line")

        assert back["downstream_line"].equals(data["downstream_line"])
        assert back["orig_head"].equals(data["orig_head"])

    def test_polygon_with_holes_roundtrip(self):
        """Interior holes and multipart polygons survive the round trip."""
        outer = [(0, 0), (0, 10), (10, 10), (10, 0)]
        hole = [(2, 2), (2, 4), (4, 4), (4, 2)]
        poly = shpg.Polygon(outer, [hole])
        multi = shpg.MultiPolygon(
            [
                shpg.Polygon(outer, [hole]),
                shpg.Polygon([(20, 20), (20, 25), (25, 25), (25, 20)]),
            ]
        )
        data = {"polygon_hr": poly, "polygon_pix": multi}

        arrays, meta = transcoder.encode_npz(data, "geometries")
        back = transcoder.decode_npz(arrays, meta, "geometries")

        assert back["polygon_hr"].equals(poly)
        assert len(back["polygon_hr"].interiors) == 1
        assert back["polygon_pix"].equals(multi)
        assert len(back["polygon_pix"].geoms) == 2

    def test_list_of_arrays_is_packed(self):
        """A ragged list of arrays round trips without one entry per item."""
        indices = [
            np.arange(2 * (i + 1), dtype=np.int64).reshape(-1, 2)
            for i in range(50)
        ]
        data = {"catchment_indices": indices}

        arrays, meta = transcoder.encode_npz(data, "geometries")
        back = transcoder.decode_npz(arrays, meta, "geometries")

        assert len(arrays) <= 4, "ragged lists must be packed into few entries"
        assert len(back["catchment_indices"]) == 50
        for got, expected in zip(back["catchment_indices"], indices):
            assert got.dtype == expected.dtype
            assert_allclose(got, expected)

    def test_multilinestring_list_roundtrip(self):
        """Ragged geometrical widths, including empty ones, round trip."""
        widths = [
            shpg.MultiLineString([[(0, 0), (0, 1)]]),
            shpg.MultiLineString([[(1, 0), (1, 2)], [(1, 3), (1, 4), (1, 5)]]),
            shpg.MultiLineString(),
        ]
        data = {"geometrical_widths": widths}

        arrays, meta = transcoder.encode_npz(data, "centerlines")
        back = transcoder.decode_npz(arrays, meta, "centerlines")

        got = back["geometrical_widths"]
        assert len(got) == 3
        for got_w, expected in zip(got, widths):
            assert got_w.equals(expected)
        assert got[2].is_empty

    def test_centerline_list_roundtrip(self):
        """Centerlines keep their attributes and their flow connections."""
        tributary, trunk = _make_centerline(), _make_centerline()
        tributary.geometrical_widths = [
            shpg.MultiLineString([[(0, 0), (0, 1)]])
        ] * tributary.nx
        tributary.set_flows_to(trunk)

        arrays, meta = transcoder.encode_npz(
            [tributary, trunk], "inversion_flowlines"
        )
        back = transcoder.decode_npz(arrays, meta, "inversion_flowlines")

        assert len(back) == 2
        assert all(isinstance(cl, Centerline) for cl in back)
        assert back[0].flows_to is back[1]
        assert back[1].flows_to is None
        assert back[0].rgi_id == "RGI60-11.00897"
        assert back[0].order == 1
        assert_allclose(back[0].surface_h, tributary.surface_h)
        assert_allclose(back[0].widths, tributary.widths)
        assert back[0].orig_head.equals(tributary.orig_head)
        assert back[0].line.equals(tributary.line)
        assert len(back[0].geometrical_widths) == tributary.nx

    def test_flows_to_point_survives_a_round_trip(self):
        """A junction near the trunk's head must not move on a round trip."""
        tributary, trunk = _make_junction_pair()
        expected_point = tributary.flows_to_point
        expected_indice = tributary.flows_to_indice
        assert expected_indice == 1

        arrays, meta = transcoder.encode_npz([tributary, trunk], "centerlines")
        back = transcoder.decode_npz(arrays, meta, "centerlines")

        assert back[0].flows_to is back[1]
        assert back[0].flows_to_point.equals(expected_point)
        assert back[0].flows_to_indice == expected_indice
        assert back[1].inflow_points[0].equals(expected_point)
        assert back[1].inflows == [back[0]]

    def test_flows_to_point_falls_back_without_the_key(self):
        """Groups written before flows_to_point was stored still decode."""
        tributary, trunk = _make_junction_pair()
        arrays, meta = transcoder.encode_npz([tributary, trunk], "centerlines")
        for item in meta["root"]["items"]:
            item.pop("flows_to_point", None)

        back = transcoder.decode_npz(arrays, meta, "centerlines")

        # The fallback recomputes the junction with check_tail=True,
        # which clips it into [4, n-5].
        assert back[0].flows_to is back[1]
        assert back[0].flows_to_point.equals(shpg.Point(4, 0))
        assert back[0].flows_to_indice == 4
        assert back[1].inflows == [back[0]]

    @pytest.mark.parametrize(
        "factory,cls_name",
        [
            (_make_mixed_bed_flowline, "MixedBedFlowline"),
            (_make_parabolic_flowline, "ParabolicBedFlowline"),
            (_make_rectangular_flowline, "RectangularBedFlowline"),
            (_make_trapezoidal_flowline, "TrapezoidalBedFlowline"),
        ],
    )
    def test_flowline_subclass_roundtrip(self, factory, cls_name):
        """Every Flowline subclass comes back as itself, geometry intact."""
        flowline = factory()

        arrays, meta = transcoder.encode_npz([flowline], "model_flowlines")
        back = transcoder.decode_npz(arrays, meta, "model_flowlines")

        assert len(back) == 1
        got = back[0]
        assert type(got).__name__ == cls_name
        assert_allclose(got.surface_h, flowline.surface_h)
        assert_allclose(got.bed_h, flowline.bed_h)
        assert_allclose(got.widths_m, flowline.widths_m)
        assert_allclose(got.section, flowline.section)
        assert_allclose(got.area_m2, flowline.area_m2)
        assert_allclose(got.volume_m3, flowline.volume_m3)
        assert got.map_trafo is None

    def test_flowline_map_trafo_roundtrip(self):
        """A flowline's map transformation survives as its grid params."""
        grid = salem.Grid(
            proj=pyproj.Proj("epsg:32632"),
            nxny=(10, 8),
            dxdy=(200.0, -100.0),
            x0y0=(500.0, 300.0),
            pixel_ref="center",
        )
        flowline = _make_parabolic_flowline()
        flowline.map_trafo = partial(grid.ij_to_crs, crs=salem.wgs84)

        arrays, meta = transcoder.encode_npz([flowline], "model_flowlines")
        back = transcoder.decode_npz(arrays, meta, "model_flowlines")[0]

        assert callable(back.map_trafo)
        assert_allclose(
            back.map_trafo(np.array([1.0, 2.0]), np.array([3.0, 4.0])),
            flowline.map_trafo(np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        )


class TestCodecDispatch:
    """The order codecs are matched in, which decides the node tags."""

    def test_every_tag_has_a_decoder(self):
        """Every tag an encoder writes must resolve to a decoder."""
        tags = set()
        for entry in transcoder._CODECS:
            names = entry[0]
            tags.update([names] if isinstance(names, str) else names)

        assert tags == set(transcoder._DECODERS)
        assert transcoder._DECODERS["list"] is transcoder._DECODERS["tuple"]

    def test_npscalar_is_matched_before_scalar(self):
        """np.float64 subclasses float, so the order decides the tag."""
        assert transcoder.encode_node(np.float64(0.1), "p", {})["t"] == (
            "npscalar"
        )
        assert transcoder.encode_node(np.str_("a"), "p", {})["t"] == "npscalar"
        assert transcoder.encode_node(np.bool_(True), "p", {})["t"] == (
            "npscalar"
        )
        assert transcoder.encode_node(0.1, "p", {})["t"] == "scalar"

    def test_centerline_lists(self):

        # A list of Centerlines must not encode as a plain list.
        node = transcoder.encode_node([_make_centerline()], "p", {})
        assert node["t"] == "centerline_list"

        with pytest.raises(TypeError, match="Centerlines, pass a list"):
            transcoder.encode_node((_make_centerline(),), "p", {})

        widths = (shpg.MultiLineString([[(0, 0), (1, 1)]]),)
        with pytest.raises(TypeError, match="MultiLineStrings, pass a list"):
            transcoder.encode_node(widths, "p", {})

    @pytest.mark.parametrize(
        "value",
        [
            np.complex128(1j),
            np.datetime64("2020-01-01"),
            np.timedelta64(1, "D"),
            np.datetime64("2020-01-01T00:00", "ns"),
        ],
    )
    def test_npscalar_rejects_non_json_values(self, value):
        """The codec, not json.dumps, must reject unstorable numpy scalars."""
        with pytest.raises(TypeError, match="numpy scalar of dtype"):
            transcoder.encode_node(value, "p", {})

    def test_encode_node_special_cases(self):

        # test an empty list is encoded as a plain list
        node = transcoder.encode_node([], "p", {})
        assert node["t"] == "list"
        assert node["items"] == []

    def test_an_unknown_tag_raises_a_value_error(self):
        """A node carrying a tag the codec does not know is rejected."""
        with pytest.raises(ValueError, match="Unknown node type"):
            transcoder.decode_node({"t": "not_a_real_tag"}, {})


class TestNpzStore:
    """The npz store as it is written into a glacier directory."""

    def test_write_and_read_npz(self, tmp_path, hef_gdir):
        """A group is written under data_store and read back verbatim."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        data = {"flux": np.arange(4, dtype=np.float64), "flux_out": 2.5}

        gdir.write_npz(data, "inversion_output", filesuffix="_test")

        expected = os.path.join(
            gdir.dir, "data_store", "inversion_output_test.npz"
        )
        assert os.path.isfile(expected)
        back = gdir.read_npz("inversion_output", filesuffix="_test")
        assert_allclose(back["flux"], data["flux"])
        assert back["flux_out"] == 2.5

    def test_underscore_prefixed_filesuffix_group_name(
        self, tmp_path, hef_gdir
    ):
        """A '_'-prefixed filesuffix, as OGGM writes them, must yield a
        single underscore in the group's file name."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        data = {"flux": np.array([1.0, 2.0, 3.0])}

        gdir.write_store(data, "inversion_input", filesuffix="_historical")

        store_dir = os.path.join(gdir.dir, "data_store")
        assert os.path.isfile(
            os.path.join(store_dir, "inversion_input_historical.npz")
        )
        assert not os.path.exists(
            os.path.join(store_dir, "inversion_input__historical.npz")
        )
        back = gdir.read_store("inversion_input", filesuffix="_historical")
        assert_allclose(back["flux"], data["flux"])

    def test_read_npz_raises_when_missing(self, tmp_path, hef_gdir):
        """Reading a group that was never written is an error."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)

        with pytest.raises(FileNotFoundError):
            hef_gdir.read_npz("inversion_output", filesuffix="_absent")


def assert_store_equal(actual, expected, context=""):
    """Assert that two stored objects match, whatever they hold."""
    assert type(actual) is type(expected), f"{context}: {type(actual)}"
    if isinstance(expected, dict):
        assert set(actual) == set(expected), context
        for key in expected:
            assert_store_equal(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), context
        for i, item in enumerate(expected):
            assert_store_equal(actual[i], item, f"{context}[{i}]")
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, context
        assert_allclose(actual, expected, err_msg=context)
    elif isinstance(expected, shapely.geometry.base.BaseGeometry):
        assert actual.equals(expected), context
    elif isinstance(expected, Flowline):
        names = (
            transcoder._FLOWLINE_ARGS
            + transcoder._FLOWLINE_ATTRS
            + (
                "widths_m",
                "section",
                "thick",
                "area_m2",
                "volume_m3",
            )
        )
        for name in names:
            assert_store_equal(
                getattr(actual, name),
                getattr(expected, name),
                f"{context}.{name}",
            )
        for name in transcoder._FLOWLINE_BED_ARGS[type(expected).__name__]:
            assert_store_equal(
                transcoder._get_bed_parameter(actual, name),
                transcoder._get_bed_parameter(expected, name),
                f"{context}.{name}",
            )
    elif isinstance(expected, Centerline):
        for name in transcoder._CENTERLINE_ARGS + transcoder._CENTERLINE_ATTRS:
            if hasattr(expected, name):
                assert_store_equal(
                    getattr(actual, name),
                    getattr(expected, name),
                    f"{context}.{name}",
                )
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(actual), context
    else:
        assert actual == expected, context


class TestStoreRoundTrip:
    """write_store and read_store must match what the pickles held."""

    @pytest.mark.parametrize(
        "name",
        [
            "geometries",
            "downstream_line",
            "centerlines",
            "inversion_flowlines",
            "model_flowlines",
            "inversion_input",
        ],
    )
    def test_store_matches_pickle(self, tmp_path, hef_gdir, name):
        """Each registry group round trips exactly as its pickle does."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        if name == "model_flowlines":
            from oggm import tasks

            tasks.init_present_time_glacier(gdir)
        original = gdir.read_store(name)

        gdir.write_store(original, name, filesuffix="_npz")
        gdir._write_pickle(original, name, filesuffix="_pkl")
        back = gdir.read_store(name, filesuffix="_npz")
        expected = gdir.read_pickle(name, filesuffix="_pkl")

        assert os.path.isfile(gdir.get_store_filepath(name, "_npz"))
        assert_store_equal(back, expected, name)

    def test_centerline_junctions_match_the_pickle(self, tmp_path, hef_gdir):
        """HEF's centerline junctions must match what the pickle held."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        original = gdir.read_store("centerlines")

        gdir.write_store(original, "centerlines", filesuffix="_npz")
        gdir._write_pickle(original, "centerlines", filesuffix="_pkl")
        back = gdir.read_store("centerlines", filesuffix="_npz")
        expected = gdir.read_pickle("centerlines", filesuffix="_pkl")

        assert any(cl.flows_to is not None for cl in expected)
        for got, want in zip(back, expected):
            if want.flows_to is None:
                assert got.flows_to is None
                continue
            assert got.flows_to_point.equals(want.flows_to_point)
            assert got.flows_to_indice == want.flows_to_indice
            assert len(got.inflows) == len(want.inflows)
            assert got.inflow_indices == want.inflow_indices


class TestStoreFallback:
    """How the store behaves when npz is absent or cannot hold the data."""

    def test_get_store_paths(self, tmp_path, hef_gdir):
        """get_store_paths returns both npz and pickle files."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        gdir.write_store(
            {"flux": np.ones(2)}, "inversion_input", filesuffix="_default"
        )
        gdir.write_npz(
            {"flux": np.ones(2)}, "inversion_input", filesuffix="_npz"
        )
        gdir._write_pickle(
            {"flux": np.zeros(2)}, "inversion_input", filesuffix="_pkl"
        )

        store_paths_default = transcoder.get_store_paths(gdir.dir)
        store_paths_npz = transcoder.get_store_paths(gdir.dir, pickle=False)
        store_paths_pkl = transcoder.get_store_paths(gdir.dir, pickle=True)

        assert any(f.name.endswith("_pkl.pkl") for f in store_paths_pkl)
        assert any(f.name.endswith(".npz") for f in store_paths_npz)
        assert any(f.name.endswith(".npz") for f in store_paths_default)

    def test_read_store_falls_back_to_pickle_once(self, tmp_path, hef_gdir):
        """A pickled group is still readable, and warns only once."""
        from oggm.utils import _workflow

        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        gdir._write_pickle([1, 2, 3], "inversion_input", filesuffix="_pkl_only")

        _workflow._warn_store_fallback.cache_clear()
        with pytest.warns(Warning, match="Store data not found"):
            back = gdir.read_store("inversion_input", filesuffix="_pkl_only")
        assert back == [1, 2, 3]

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            gdir.read_store("inversion_input", filesuffix="_pkl_only")
        assert not any(
            "Store data not found" in str(r.message) for r in records
        )

    def test_write_pickle_invalidates_the_npz_group(self, tmp_path, hef_gdir):
        """A pickle written over a group must not be shadowed by the npz."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir
        gdir.write_store(
            {"flux": np.ones(2)}, "inversion_input", filesuffix="_stale"
        )

        gdir._write_pickle(
            {"flux": np.zeros(2)}, "inversion_input", filesuffix="_stale"
        )

        assert not os.path.exists(
            gdir.get_store_filepath("inversion_input", "_stale")
        )
        back = gdir.read_store("inversion_input", filesuffix="_stale")
        assert_allclose(back["flux"], np.zeros(2))

    def test_unsupported_data_falls_back_to_pickle(self, tmp_path, hef_gdir):
        """Data the codec cannot hold is pickled instead, with a warning."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir

        with pytest.warns(RuntimeWarning, match="falling back to pickle"):
            gdir.write_store(
                {"when": datetime.datetime(2020, 1, 1)},
                "inversion_input",
                filesuffix="_odd",
            )

        assert not os.path.exists(
            gdir.get_store_filepath("inversion_input", "_odd")
        )
        back = gdir.read_store("inversion_input", filesuffix="_odd")
        assert back["when"] == datetime.datetime(2020, 1, 1)

    def test_has_file_sees_a_store_group(self, tmp_path, hef_gdir):
        """has_file must find a group that only exists in the store."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir

        assert not gdir.has_file("inversion_input", filesuffix="_probe")
        gdir.write_store(
            {"flux": np.ones(2)}, "inversion_input", filesuffix="_probe"
        )
        assert gdir.has_file("inversion_input", filesuffix="_probe")


class TestCompatibility:
    """The npz store must be compatible with the pickles it replaces."""

    @pytest.mark.parametrize("arg_delete", [True, False])
    def test_convert_pickles_to_npz(self, tmp_path, hef_gdir, arg_delete):
        """A glacier directory's pickles are rewritten into npz."""
        cfg.initialize()
        cfg.PATHS["working_dir"] = str(tmp_path)
        gdir = hef_gdir

        # Write a pickle and a store group
        gdir._write_pickle(np.ones(2), "inversion_input", filesuffix="_convert")
        assert gdir.has_file("inversion_input", filesuffix="_convert")
        gdir.write_store(
            {"flux": np.ones(2)}, "inversion_input", filesuffix="_control"
        )
        assert gdir.has_file("inversion_input", filesuffix="_control")

        convert_pickles_to_npz(gdir, delete=arg_delete)

        assert gdir.has_file("inversion_input", filesuffix="_control")
        assert gdir.has_file("inversion_input", filesuffix="_convert")

        pickle_path = Path(gdir.dir) / "inversion_input_convert.pkl"
        assert not pickle_path.exists() if arg_delete else pickle_path.exists()

        back = gdir.read_store("inversion_input", filesuffix="_control")
        np.testing.assert_array_equal(back["flux"], np.ones(2))

        back = gdir.read_store("inversion_input", filesuffix="_convert")
        np.testing.assert_array_equal(back, np.ones(2))
