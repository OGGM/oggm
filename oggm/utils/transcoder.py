"""Data store utilities for OGGM.

Encode and decode glacier directory data stores.
"""

import os
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import shapely
from salem import Grid, wgs84

SCHEMA_VERSION = 2


def get_store_paths(directory: str | Path, pickle: bool = False) -> list[Path]:
    """Get all file paths for all available data stores in a directory.

    Parameters
    ----------
    directory : str or Path
        Path to the directory.
    pickle : bool, optional
        If True, search for  pickle files. Default is False.

    Returns
    -------
    list[Path]
        A list of file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory.")

    # Using glob is slower and doesn't return the same result
    if not pickle:
        return [
            Path(f)
            # is it safe to hardcode this?
            for f in os.listdir(directory / "data_store")
            if f.endswith(".npz")
        ]
    else:
        return [Path(f) for f in os.listdir(directory) if f.endswith(".pkl")]


def _join_path(path: str, key: str) -> str:
    """Join a key onto an array path.

    Parameters
    ----------
    path : str
        Path of the parent node, empty at the root.
    key : str
        Key of the child node.

    Returns
    -------
    str
        The child's array path.
    """
    return f"{path}/{key}" if path else str(key)


def _is_packable_array_list(obj: list) -> bool:
    """Check whether a list of arrays can be packed into one array.

    Parameters
    ----------
    obj : list
        The list to check.

    Returns
    -------
    bool
        True if every item is an array that concatenates along the first
        axis without losing its dtype or trailing shape.
    """
    if not obj or not all(isinstance(a, np.ndarray) for a in obj):
        return False
    first = obj[0]
    if first.ndim == 0 or first.dtype == object:
        return False

    return all(
        a.dtype == first.dtype and a.shape[1:] == first.shape[1:] for a in obj
    )


# Attributes of a Centerline that its constructor does not take
_CENTERLINE_ATTRS = (
    "order",
    "_widths",
    "is_rectangular",
    "is_trapezoid",
    "apparent_mb",
    "flux",
    "flux_out",
    "flux_needs_correction",
    "geometrical_widths",
)

# Constructor arguments shared by Centerline and every Flowline
_CENTERLINE_ARGS = (
    "line",
    "dx",
    "surface_h",
    "orig_head",
    "rgi_id",
    "map_dx",
)


# Constructor arguments shared by every Flowline subclass
_FLOWLINE_ARGS = (
    "line",
    "dx",
    "map_dx",
    "surface_h",
    "bed_h",
    "rgi_id",
    "water_level",
)

# Attributes of a Flowline that its constructor does not take
_FLOWLINE_ATTRS = (
    "order",
    "calving_bucket_m3",
    "min_ice_thick_for_length",
    "glacier_length_method",
)

# Bed parameters, by subclass. The value stored is not always the
# attribute of the same name (see _get_bed_parameter()).
_FLOWLINE_BED_ARGS = {
    "MixedBedFlowline": (
        "section",
        "bed_shape",
        "is_trapezoid",
        "lambdas",
        "widths_m",
    ),
    "ParabolicBedFlowline": ("bed_shape",),
    "RectangularBedFlowline": ("widths",),
    "TrapezoidalBedFlowline": ("widths", "lambdas"),
}

# Cached arrays for MixedBedFlowline
_MIXED_BED_CACHES = ("_sqrt_bed", "_w0_m")


def _get_bed_parameter(flowline, name: str):
    """Get a bed parameter in the form its constructor expects.

    Parameters
    ----------
    flowline : oggm.core.flowline.Flowline
        The flowline to read from.
    name : str
        Name of the constructor argument.

    Returns
    -------
    Any
        The value to store for `name`.
    """
    from oggm.core.flowline import (
        MixedBedFlowline,
        RectangularBedFlowline,
        TrapezoidalBedFlowline,
    )

    if name == "lambdas":
        if isinstance(flowline, MixedBedFlowline):
            # Mixed keeps the lambdas, with NaNs off trapezoids.
            lambdas = getattr(flowline, "lambdas", None)
            return flowline._lambdas if lambdas is None else lambdas
        if isinstance(flowline, TrapezoidalBedFlowline):
            return flowline._lambdas
    if name == "widths" and isinstance(flowline, RectangularBedFlowline):
        return flowline._widths

    return getattr(flowline, name, None)


def _is_centerline_list(obj: list) -> bool:
    """Check whether a list holds only Centerlines or Flowlines.

    Parameters
    ----------
    obj : list
        The list to check.

    Returns
    -------
    bool
        True if every item is a Centerline, which includes every
        Flowline subclass.
    """
    from oggm import Centerline

    return bool(obj) and all(isinstance(item, Centerline) for item in obj)


def encode_centerline_list(obj: list, path: str, arrays: dict) -> dict:
    """Encode a list of Centerlines or Flowlines.

    Parameters
    ----------
    obj : list
        The Centerlines or Flowlines to encode.
    path : str
        Path of this node within the tree, used to key its arrays.
    arrays : dict
        Mapping of array keys to arrays, updated in place.

    Returns
    -------
    dict
        A JSON-compatible description of the list.
    """
    from oggm.core.flowline import Flowline

    indices = {id(item): i for i, item in enumerate(obj)}
    items = []
    for i, item in enumerate(obj):
        cls_name = type(item).__name__
        if isinstance(item, Flowline):
            fields = _encode_flowline_fields(item, cls_name)
        else:
            fields = {
                name: getattr(item, name, None)
                for name in _CENTERLINE_ARGS + _CENTERLINE_ATTRS
            }
        item_path = _join_path(path, str(i))
        flows_to = getattr(item, "flows_to", None)
        node = {
            "class": cls_name,
            "fields": {
                name: encode_node(value, _join_path(item_path, name), arrays)
                for name, value in fields.items()
            },
            "flows_to": (
                indices.get(id(flows_to), -1) if flows_to is not None else -1
            ),
        }
        map_trafo = getattr(item, "map_trafo", None)
        if map_trafo is not None:
            # A partial cannot be serialised, so store the grid it binds.
            node["grid"] = get_grid_params_from_partial(map_trafo)
        # Store, as junction depends on the flags passed
        node["flows_to_point"] = encode_node(
            getattr(item, "flows_to_point", None),
            _join_path(item_path, "flows_to_point"),
            arrays,
        )
        items.append(node)

    return {"t": "centerline_list", "items": items}


def _encode_flowline_fields(flowline, cls_name: str) -> dict:
    """Collect the attributes needed to rebuild a Flowline.

    Parameters
    ----------
    flowline : oggm.core.flowline.Flowline
        The flowline to read from.
    cls_name : str
        Name of its concrete subclass.

    Returns
    -------
    dict
        Attribute names mapped to the values to store.
    """
    fields = {
        name: getattr(flowline, name, None)
        for name in _FLOWLINE_ARGS + _FLOWLINE_ATTRS
    }
    for name in _FLOWLINE_BED_ARGS.get(cls_name, ()):
        fields[name] = _get_bed_parameter(flowline, name)
    if cls_name == "MixedBedFlowline":
        fields.update(
            {name: getattr(flowline, name, None) for name in _MIXED_BED_CACHES}
        )

    return fields


def decode_centerline_list(node: dict, arrays: dict) -> list:
    """Reconstruct a list of Centerlines or Flowlines.

    Inverse of :func:`encode_centerline_list`.

    Parameters
    ----------
    node : dict
        The encoded list node.
    arrays : dict
        Mapping of array keys to arrays, as read from the npz file.

    Returns
    -------
    list
        The reconstructed Centerlines or Flowlines.
    """
    from oggm import Centerline

    lines = []
    for item in node["items"]:
        fields = {
            name: decode_node(child, arrays)
            for name, child in item["fields"].items()
        }
        cls_name = item["class"]
        if cls_name in _FLOWLINE_BED_ARGS:
            line = _decode_flowline(cls_name, fields, item.get("grid"))
        else:
            line = Centerline(
                **{name: fields[name] for name in _CENTERLINE_ARGS}
            )
            for name in _CENTERLINE_ATTRS:
                setattr(line, name, fields[name])
        lines.append(line)

    for line, item in zip(lines, node["items"]):
        index = item["flows_to"]
        if not 0 <= index < len(lines):
            continue
        other = lines[index]
        point = (
            decode_node(item["flows_to_point"], arrays)
            if "flows_to_point" in item
            else None
        )
        if point is None:
            # Written before flows_to_point was stored: recompute it.
            line.set_flows_to(other)
            continue
        line.flows_to = other
        line.flows_to_point = point
        other.inflow_points.append(point)
        other.inflows.append(line)

    return lines


def get_map_trafo_from_grid_params(grid: dict | None) -> Callable | None:
    """Rebuild the map transformation from stored grid parameters.

    Flowlines built without a glacier directory, e.g. in the PyGEM
    sandbox, have no grid to store, so they come back without a
    transformation.

    Parameters
    ----------
    grid : dict or None
        Grid parameters, as collected by
        :func:`get_grid_params_from_partial`.

    Returns
    -------
    Callable or None
        A partial mapping grid coordinates to WGS84, or None when no
        grid was stored.
    """
    if not grid:
        return None
    map_trafo = Grid(
        proj=grid["pyproj_srs"],
        nxny=tuple(grid["nxny"]),
        dxdy=tuple(grid["dxdy"]),
        x0y0=tuple(grid["x0y0"]),
        pixel_ref=grid["pixel_ref"],
    )

    return partial(map_trafo.ij_to_crs, crs=wgs84)


def _decode_flowline(cls_name: str, fields: dict, grid: dict | None):
    """Rebuild a Flowline of the subclass it was stored as.

    Parameters
    ----------
    cls_name : str
        Name of the concrete Flowline subclass.
    fields : dict
        The decoded attributes, as collected by
        :func:`_encode_flowline_fields`.
    grid : dict or None
        Grid parameters for `map_trafo`, absent for flowlines built
        without a glacier directory.

    Returns
    -------
    oggm.core.flowline.Flowline
        The reconstructed flowline.
    """
    from oggm.core import flowline as oggm_flowline

    cls = getattr(oggm_flowline, cls_name)
    kwargs = {name: fields[name] for name in _FLOWLINE_ARGS}
    kwargs.update({name: fields[name] for name in _FLOWLINE_BED_ARGS[cls_name]})
    flowline = cls(**kwargs)
    for name in _FLOWLINE_ATTRS:
        setattr(flowline, name, fields[name])
    if cls_name == "MixedBedFlowline":
        for name in _MIXED_BED_CACHES:
            setattr(flowline, name, fields[name])
    flowline.map_trafo = get_map_trafo_from_grid_params(grid)

    return flowline


def _is_multilinestring_list(obj: list) -> bool:
    """Check whether a list holds only MultiLineStrings.

    Parameters
    ----------
    obj : list
        The list to check.

    Returns
    -------
    bool
        True if every item is a MultiLineString.
    """
    return bool(obj) and all(
        isinstance(item, shapely.MultiLineString) for item in obj
    )


def _encode_polygon(
    obj: shapely.Polygon | shapely.MultiPolygon, path: str, arrays: dict
) -> dict:
    """Encode a (Multi)Polygon as flattened rings.

    Every ring of every part is concatenated into one vertex array,
    alongside the index arrays needed to split it back up.

    Parameters
    ----------
    obj : shapely.Polygon or shapely.MultiPolygon
        The geometry to encode.
    path : str
        Path of this node within the tree, used to key its arrays.
    arrays : dict
        Mapping of array keys to arrays, updated in place.

    Returns
    -------
    dict
        A JSON-compatible description of the geometry.
    """
    rings = _extract_polygon_coords(obj)
    arrays[_join_path(path, "vertices")] = (
        np.concatenate([r[0] for r in rings], axis=0)
        if rings
        else np.zeros((0, 2), dtype=np.float64)
    )
    arrays[_join_path(path, "ring_lengths")] = np.array(
        [len(r[0]) for r in rings], dtype=np.int64
    )
    arrays[_join_path(path, "ring_poly_idx")] = np.array(
        [r[1] for r in rings], dtype=np.int64
    )
    arrays[_join_path(path, "ring_is_exterior")] = np.array(
        [r[2] for r in rings], dtype=bool
    )

    # A MultiPolygon can hold a single part, so record the type.
    return {
        "t": "polygon",
        "k": path,
        "multi": isinstance(obj, shapely.MultiPolygon),
    }


def _encode_multilinestring_list(obj: list, path: str, arrays: dict) -> dict:
    """Encode a list of MultiLineStrings into flat arrays.

    Parameters
    ----------
    obj : list
        The MultiLineStrings to encode.
    path : str
        Path of this node within the tree, used to key its arrays.
    arrays : dict
        Mapping of array keys to arrays, updated in place.

    Returns
    -------
    dict
        A JSON-compatible description of the list.
    """
    coords, line_lengths, member_counts = [], [], []
    for width in obj:
        count = 0
        for member in width.geoms:
            if member.is_empty:
                continue
            member_coords = shapely.get_coordinates(member)
            coords.append(member_coords)
            line_lengths.append(len(member_coords))
            count += 1
        member_counts.append(count)
    arrays[_join_path(path, "vertices")] = (
        np.concatenate(coords, axis=0)
        if coords
        else np.zeros((0, 2), dtype=np.float64)
    )
    arrays[_join_path(path, "line_lengths")] = np.array(
        line_lengths, dtype=np.int64
    )
    arrays[_join_path(path, "member_counts")] = np.array(
        member_counts, dtype=np.int64
    )

    return {"t": "multilinestring_list", "k": path}


def _encode_array_list(obj: list, path: str, arrays: dict) -> dict:
    """Encode a ragged list of arrays as one array plus its lengths.

    Parameters
    ----------
    obj : list
        The arrays to encode, as accepted by
        :func:`_is_packable_array_list`.
    path : str
        Path of this node within the tree, used to key its arrays.
    arrays : dict
        Mapping of array keys to arrays, updated in place.

    Returns
    -------
    dict
        A JSON-compatible description of the list.
    """
    arrays[_join_path(path, "vertices")] = np.concatenate(obj, axis=0)
    arrays[_join_path(path, "lengths")] = np.array(
        [len(a) for a in obj], dtype=np.int64
    )

    return {"t": "array_list", "k": path}


def _encode_none(obj: Any, path: str, arrays: dict) -> dict:
    """Encode None."""
    return {"t": "none"}


def _encode_array(obj: np.ndarray, path: str, arrays: dict) -> dict:
    """Encode an array, storing it under this node's path."""
    if obj.dtype == object:
        raise TypeError("Cannot encode an object-dtype array.")
    arrays[path] = obj

    return {"t": "array", "k": path}


def _encode_npscalar(obj: np.generic, path: str, arrays: dict) -> dict:
    """Encode a numpy scalar, keeping its dtype.

    Raises a TypeError here rather than in ``json.dumps`` when
    ``obj.item()`` is not a JSON scalar, e.g. for complex values. and for
    datetime64 and timedelta64 because at ``ns`` their item is an int
    which can't be decoded without its unit.
    """
    value = obj.item()
    if obj.dtype.kind in "mM" or not isinstance(value, (bool, int, float, str)):
        raise TypeError(f"Cannot encode a numpy scalar of dtype {obj.dtype}.")

    return {"t": "npscalar", "dtype": obj.dtype.str, "v": value}


def _encode_scalar(obj: Any, path: str, arrays: dict) -> dict:
    """Encode a Python scalar."""
    return {"t": "scalar", "v": obj}


def _encode_linestring(obj: Any, path: str, arrays: dict) -> dict:
    """Encode a LineString as its coordinates."""
    arrays[path] = shapely.get_coordinates(obj)

    return {"t": "linestring", "k": path}


def _encode_point(obj: Any, path: str, arrays: dict) -> dict:
    """Encode a Point as a flat pair of coordinates."""
    arrays[path] = shapely.get_coordinates(obj).flatten()

    return {"t": "point", "k": path}


def _encode_dict(obj: dict, path: str, arrays: dict) -> dict:
    """Encode a dict, recursing into each value.

    Keys must be strings, since the node is written out as JSON. On a
    bad key, `arrays` may already hold the earlier values' arrays, so
    callers must discard it when this raises.
    """
    items = {}
    for key, value in obj.items():
        if not isinstance(key, str):
            raise TypeError(f"Cannot encode a dict with key {key!r}.")
        items[key] = encode_node(value, _join_path(path, key), arrays)

    return {"t": "dict", "items": items}


def _encode_sequence(obj: Any, path: str, arrays: dict) -> dict:
    """Encode a list or a tuple, recursing into each item."""
    # Only lists take the specialised list codecs, and the bare items
    # have no codec of their own, so point at the fix.
    if isinstance(obj, tuple) and (
        _is_centerline_list(obj) or _is_multilinestring_list(obj)
    ):
        raise TypeError(
            f"Cannot encode a tuple of {type(obj[0]).__name__}s, "
            "pass a list instead."
        )
    items = [
        encode_node(item, _join_path(path, str(i)), arrays)
        for i, item in enumerate(obj)
    ]

    return {
        "t": "tuple" if isinstance(obj, tuple) else "list",
        "items": items,
    }


def _decode_polygon(
    node: dict, arrays: dict
) -> shapely.Polygon | shapely.MultiPolygon:
    """Reconstruct a (Multi)Polygon from its flattened rings.

    The flat vertex array is split back into rings using
    ``ring_lengths``, and the rings are grouped into parts by
    ``ring_poly_idx`` and ``ring_is_exterior``.

    Parameters
    ----------
    node : dict
        The encoded polygon node.
    arrays : dict
        Mapping of array keys to arrays, as read from the npz file.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        The reconstructed geometry.
    """
    path = node["k"]
    vertices = arrays[_join_path(path, "vertices")]
    ring_lengths = arrays[_join_path(path, "ring_lengths")]
    ring_poly_idx = arrays[_join_path(path, "ring_poly_idx")]
    ring_is_exterior = arrays[_join_path(path, "ring_is_exterior")]

    splits = np.cumsum(ring_lengths)[:-1]
    rings = np.split(vertices, splits) if len(ring_lengths) else []

    parts = {}
    for ring, poly_idx, is_exterior in zip(
        rings, ring_poly_idx, ring_is_exterior
    ):
        part = parts.setdefault(int(poly_idx), {"exterior": None, "holes": []})
        if bool(is_exterior):
            part["exterior"] = ring
        else:
            part["holes"].append(ring)

    polygons = [
        shapely.Polygon(parts[idx]["exterior"], parts[idx]["holes"])
        for idx in sorted(parts)
    ]
    if node.get("multi"):
        return shapely.MultiPolygon(polygons)

    return polygons[0]


def _decode_multilinestring_list(node: dict, arrays: dict) -> list:
    """Reconstruct a list of MultiLineStrings from flat arrays.

    Inverse of :func:`_encode_multilinestring_list`.

    Parameters
    ----------
    node : dict
        The encoded list node.
    arrays : dict
        Mapping of array keys to arrays, as read from the npz file.

    Returns
    -------
    list
        The reconstructed MultiLineStrings.
    """
    path = node["k"]
    lines = np.split(
        arrays[_join_path(path, "vertices")],
        np.cumsum(arrays[_join_path(path, "line_lengths")])[:-1],
    )
    widths, offset = [], 0
    for count in arrays[_join_path(path, "member_counts")]:
        members = [shapely.LineString(lines[offset + i]) for i in range(count)]
        offset += count
        widths.append(shapely.MultiLineString(members))

    return widths


def _decode_array_list(node: dict, arrays: dict) -> list:
    """Reconstruct a ragged list of arrays from one flat array.

    Inverse of :func:`_encode_array_list`.

    Parameters
    ----------
    node : dict
        The encoded list node.
    arrays : dict
        Mapping of array keys to arrays, as read from the npz file.

    Returns
    -------
    list
        The reconstructed arrays.
    """
    path = node["k"]
    lengths = arrays[_join_path(path, "lengths")]

    return np.split(
        arrays[_join_path(path, "vertices")], np.cumsum(lengths)[:-1]
    )


def _decode_none(node: dict, arrays: dict) -> None:
    """Reconstruct None."""
    return None


def _decode_array(node: dict, arrays: dict) -> np.ndarray:
    """Reconstruct an array from its stored path."""
    return arrays[node["k"]]


def _decode_npscalar(node: dict, arrays: dict) -> Any:
    """Reconstruct a numpy scalar, restoring its dtype."""
    return np.dtype(node["dtype"]).type(node["v"])


def _decode_scalar(node: dict, arrays: dict) -> Any:
    """Reconstruct a Python scalar."""
    return node["v"]


def _decode_linestring(node: dict, arrays: dict) -> shapely.LineString:
    """Reconstruct a LineString from its coordinates."""
    return shapely.LineString(arrays[node["k"]])


def _decode_point(node: dict, arrays: dict) -> shapely.Point:
    """Reconstruct a Point from its coordinates."""
    return shapely.Point(arrays[node["k"]])


def _decode_dict(node: dict, arrays: dict) -> dict:
    """Reconstruct a dict, recursing into each value."""
    return {
        key: decode_node(child, arrays) for key, child in node["items"].items()
    }


def _decode_sequence(node: dict, arrays: dict) -> list | tuple:
    """Reconstruct a list or a tuple, recursing into each item."""
    items = [decode_node(child, arrays) for child in node["items"]]

    return tuple(items) if node["t"] == "tuple" else items


def _is_centerline_list_node(obj: Any) -> bool:
    """Check for a list of Centerlines, excluding tuples."""
    return isinstance(obj, list) and _is_centerline_list(obj)


def _is_multilinestring_list_node(obj: Any) -> bool:
    """Check for a list of MultiLineStrings, excluding tuples."""
    return isinstance(obj, list) and _is_multilinestring_list(obj)


def _is_packable_array_list_node(obj: Any) -> bool:
    """Check for a packable list of arrays, excluding tuples."""
    return isinstance(obj, list) and _is_packable_array_list(obj)


"""
NOTE: The order of the codecs matters! `npscalar` must precede `scalar`,
since `np.float64` subclasses `float` and `np.str_` subclasses `str`.
The three list codecs must precede the generic sequence codec, or a list
of Centerlines is encoded item by item and rejected.

The first field is the tag the encoder writes, or a tuple of tags where
one encoder writes more than one. Every encoder takes (obj, path, arrays)
and returns a node; every decoder takes (node, arrays) and returns the
object.
"""

_CODECS = (
    ("none", lambda o: o is None, _encode_none, _decode_none),
    (
        "array",
        lambda o: isinstance(o, np.ndarray),
        _encode_array,
        _decode_array,
    ),
    (
        "npscalar",
        lambda o: isinstance(o, np.generic),
        _encode_npscalar,
        _decode_npscalar,
    ),
    (
        "scalar",
        lambda o: isinstance(o, (bool, int, float, str)),
        _encode_scalar,
        _decode_scalar,
    ),
    (
        "linestring",
        lambda o: isinstance(o, shapely.LineString),
        _encode_linestring,
        _decode_linestring,
    ),
    (
        "point",
        lambda o: isinstance(o, shapely.Point),
        _encode_point,
        _decode_point,
    ),
    (
        "polygon",
        lambda o: isinstance(o, (shapely.Polygon, shapely.MultiPolygon)),
        _encode_polygon,
        _decode_polygon,
    ),
    ("dict", lambda o: isinstance(o, dict), _encode_dict, _decode_dict),
    (
        "centerline_list",
        _is_centerline_list_node,
        encode_centerline_list,
        decode_centerline_list,
    ),
    (
        "multilinestring_list",
        _is_multilinestring_list_node,
        _encode_multilinestring_list,
        _decode_multilinestring_list,
    ),
    (
        "array_list",
        _is_packable_array_list_node,
        _encode_array_list,
        _decode_array_list,
    ),
    (
        ("list", "tuple"),
        lambda o: isinstance(o, (list, tuple)),
        _encode_sequence,
        _decode_sequence,
    ),
)

# The sequence codec writes two tags so the tags field is flattened.
# Keying this on the row's name would make the tag transient.
_DECODERS = {
    tag: decode
    for tags, _, _, decode in _CODECS
    for tag in ((tags,) if isinstance(tags, str) else tags)
}


def encode_node(obj: Any, path: str, arrays: dict) -> dict:
    """Encode an object into a JSON-compatible node.

    Any array encountered is stored in `arrays` under a key derived from
    `path`, so that the node itself stays JSON-serialisable. The codec
    is chosen from the first matching entry in :data:`_CODECS`

    Parameters
    ----------
    obj : Any
        The object to encode.
    path : str
        Path of this node within the tree, used to key its arrays.
    arrays : dict
        Mapping of array keys to arrays, updated in place.

    Returns
    -------
    dict
        A JSON-compatible description of `obj`.

    Raises
    ------
    TypeError
        If `obj` is of a type the codec cannot represent.
    """
    for _, matches, encode, _ in _CODECS:
        if matches(obj):
            return encode(obj, path, arrays)

    raise TypeError(f"Cannot encode an object of type {type(obj).__name__}.")


def decode_node(node: dict, arrays: dict) -> Any:
    """Reconstruct an object from an encoded node.

    Inverse of :func:`encode_node`. The decoder is looked up from the
    first matching entry in :data:`_DECODERS`.

    Parameters
    ----------
    node : dict
        The node to decode.
    arrays : dict
        Mapping of array keys to arrays, as read from the npz file.

    Returns
    -------
    Any
        The reconstructed object.

    Raises
    ------
    ValueError
        If the node carries a type the codec does not know.
    """
    try:
        decode = _DECODERS[node["t"]]
    except KeyError:
        raise ValueError(f"Unknown node type {node['t']!r}.") from None

    return decode(node, arrays)


def encode_npz(data: Any, name: str = "") -> tuple[dict, dict]:
    """Convert data destined for a pickle into npz contents.

    Parameters
    ----------
    data : Any
        The data to convert.
    name : str, optional
        Name of the store group.

    Returns
    -------
    tuple[dict, dict]
        The arrays to write, and the metadata describing them.
    """
    arrays = {}
    root = encode_node(data, path=name, arrays=arrays)

    return arrays, {"schema": SCHEMA_VERSION, "root": root}


def decode_npz(arrays: dict, meta: dict, name: str = "") -> Any:
    """Reconstruct data from npz contents.

    Inverse of :func:`encode_npz`.

    Parameters
    ----------
    arrays : dict
        Mapping of array keys to arrays.
    meta : dict
        Metadata describing the arrays.
    name : str, optional
        Name of the store group. See :func:`encode_npz`.

    Returns
    -------
    Any
        The reconstructed data, matching what the pickle held.
    """
    return decode_node(meta["root"], arrays)


def _extract_polygon_coords(geometry: shapely.Polygon) -> list[tuple]:
    """Extract coordinates of a shapely Polygon or MultiPolygon.

    Polygons may contain interior holes, MultiPolygons may contain
    several parts, each with its own holes. Every ring is returned with
    the index of the part it belongs to.

    Parameters
    ----------
    geometry : shapely.Polygon or shapely.MultiPolygon
        The input geometry from which to extract coordinates.

    Returns
    -------
    list[tuple]
        One ``(coords, poly_idx, is_exterior)`` tuple per ring, where
        ``coords`` is an ``(n, 2)`` numpy array, ``poly_idx`` is the
        part index (0 for a simple Polygon) and ``is_exterior`` is True
        for a part's exterior ring and False for an interior hole. Rings
        are emitted exterior-first within each part.
    """
    rings = []
    if geometry.geom_type == "Polygon":
        rings.append((np.asarray(geometry.exterior.coords), 0, True))
        for interior in geometry.interiors:
            rings.append((np.asarray(interior.coords), 0, False))
    elif geometry.geom_type == "MultiPolygon":
        for poly_idx, part in enumerate(geometry.geoms):
            rings.append((np.asarray(part.exterior.coords), poly_idx, True))
            for interior in part.interiors:
                rings.append((np.asarray(interior.coords), poly_idx, False))
    else:
        raise ValueError("Unhandled geometry type: " + repr(geometry.geom_type))

    return rings


def get_grid_params_from_partial(p: Callable) -> dict:
    """Collect the parameters of the grid a partial binds.

    A partial cannot be serialised, so a flowline's map transformation
    is stored as the parameters needed to rebuild its grid.

    TODO: Convert from glacier_grid instead of partial.

    Parameters
    ----------
    p : Callable
        A partial of ``salem.Grid.ij_to_crs``.

    Returns
    -------
    dict
        The grid's projection, shape, resolution, origin, and pixel
        reference.
    """
    grid = p.func.__self__
    grid_parameters = {
        "pyproj_srs": grid.proj.crs.to_json_dict(),
        "nxny": (grid.nx, grid.ny),
        "dxdy": (grid.dx, grid.dy),
        "x0y0": (grid.x0, grid.y0),
        "pixel_ref": grid.pixel_ref,
    }

    return grid_parameters
