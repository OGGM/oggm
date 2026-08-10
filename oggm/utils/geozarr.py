"""Zarr utilities for OGGM."""

import xarray as xr
from pathlib import Path
import numpy as np
import os
import shapely

import pyproj
from salem import Grid, wgs84
from functools import partial
from typing import Callable, Any


def get_pickle_paths(directory: str | Path) -> list[Path]:
    """Get file paths for all available pickles in a directory.

    Parameters
    ----------
    directory : str or Path
        Path to the directory containing the pickles.

    Returns
    -------
    list[Path]
        A list of file paths to all the pickles in the directory.
    """

    return [Path(f) for f in os.listdir(directory) if f[-4:] == ".pkl"]


def get_tranche(data: dict, type_only: bool = False) -> dict:
    """Extract a tranche of data from a dictionary.

    Parameters
    ----------
    data : dict
        The input dictionary to extract the tranche from.
    type_only : bool, default True
        If True, only the types of the data will be extracted. If False,
        the actual data will be extracted.
    """
    tranche = {}
    for k, v in data.items():
        if not type_only:
            tranche[k] = v
        else:
            tranche[k] = type(v)
    return tranche


def filter_arrays_from_dict(x: dict) -> dict:
    """Get all numpy array-type items from a dictionary."""
    return {k: v for k, v in x.items() if isinstance(v, np.ndarray)}


def filter_lists_from_dict(x: dict) -> dict:
    """Filter list-type items from a dictionary."""
    return {k: v for k, v in x.items() if isinstance(v, list)}


def get_pickle_data(pickle_files: list[Path], gdir, type_only: bool = False):
    """Read pickle files and extract their data into a dictionary.

    Parameters
    ----------
    pickle_files : list[Path]
        Paths to pickle files.
    gdir : oggm.GlacierDirectory
        GlacierDirectory object from which the pickles are read.
    type_only : bool, default False
        If True, only the types of the data will be extracted. If False,
        the actual data will be extracted.

    Returns
    -------
    dict
        A dictionary with the pickle base names as keys and the
        extracted data as values, or their types if `type_only` is True.
    """
    pickle_data = {}
    for pickle in pickle_files:
        try:
            stem = gdir.read_pickle(pickle.stem)
            if isinstance(stem, list):
                slices = []
                for i in stem:
                    if isinstance(i, dict):
                        slices.append(get_tranche(i, type_only=type_only))
                    else:
                        slices.append(type(i))
                pickle_data[pickle.stem] = slices
            elif isinstance(stem, dict):
                pickle_data[pickle.stem] = get_tranche(
                    stem, type_only=type_only
                )
            else:
                print(f"Pickle {pickle.stem} not parseable.")
        except Exception as e:
            print(e)
            print(
                f"Pickle {pickle.stem} of type {type(pickle.stem)} not parseable."
            )

    return pickle_data


"""Convert data into zarr-compatible structures."""


def _validate_linestring(
    line: xr.DataArray | shapely.LineString,
) -> shapely.LineString | None:
    """Coerce an object into a LineString.

    Parameters
    ----------
    line : xr.DataArray or shapely.LineString
        The input object to validate.

    Returns
    -------
    shapely.LineString or None
        A LineString object, or None if the input is None.
    """
    if not isinstance(line, shapely.LineString) and (line is not None):
        line = shapely.LineString(line)
    return line


def _validate_polygon(
    polygon: xr.DataArray | shapely.Polygon,
) -> shapely.Polygon | None:
    """Coerce an object into a Polygon.

    Parameters
    ----------
    polygon : xr.DataArray or shapely.Polygon
        The input object to validate.

    Returns
    -------
    shapely.Polygon or None
        A Polygon, or None if the input is None.
    """
    if not isinstance(polygon, shapely.Polygon) and (polygon is not None):
        polygon = shapely.Polygon(polygon)
    return polygon


def _extract_polygon_coords(geometry: shapely.Polygon) -> list[tuple]:
    """Extract coordinates of a shapely Polygon or MultiPolygon.

    Polygons may contain interior holes, MultiPolygons may contain
    several parts, each with its own holes. Every ring is returned with
    the index of the part it belongs to.

    TODO: This is streamable, but it may be more efficient to use a
    shapefile instead.

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


def _validate_point(
    point: xr.DataArray | np.ndarray | shapely.Point | None,
) -> shapely.Point | None:
    """Coerce a stored DataArray/ndarray back to a shapely Point.

    Parameters
    ----------
    point : xr.DataArray, np.ndarray, shapely.Point, or None
        The input object to validate.

    Returns
    -------
    shapely.Point or None
        A Point object, or None if the input is None.
    """
    if point is None:
        return None
    if isinstance(point, (xr.DataArray, np.ndarray)):
        coords = np.asarray(point).flatten()
        return shapely.geometry.Point(coords)
    return point


def get_datatree_value(
    data_tree: xr.DataTree, attribute: str
) -> xr.DataArray | None:
    """Get a value from a DataTree node.

    Parameters
    ----------
    data_tree : xr.DataTree
        The DataTree node from which to extract the value.
    attribute : str
        The name of the attribute to retrieve.

    Returns
    -------
    xr.DataArray or None
        The value of the specified attribute, or None if the attribute
        does not exist or is empty.
    """
    if hasattr(data_tree, attribute):
        if isinstance(getattr(data_tree, attribute), xr.DataTree):
            if getattr(data_tree, attribute).is_empty:
                return None
        return getattr(data_tree, attribute).values.copy()
    return None


def get_flowline_from_datatree(data_tree: xr.DataTree):
    """Reconstruct a Flowline object from a DataTree.

    The concrete subclass is read from the ``_flowline_class`` node attr
    (defaulting to ``MixedBedFlowline`` for stores written before
    multi-class support, preserving backward compatibility).
    """
    from oggm.core.flowline import (
        MixedBedFlowline,
        ParabolicBedFlowline,
        RectangularBedFlowline,
        TrapezoidalBedFlowline,
    )

    cls_name = data_tree.attrs.get("_flowline_class", "MixedBedFlowline")

    # kwargs common to every Flowline subclass constructor
    base = dict(
        line=_validate_linestring(get_datatree_value(data_tree, "line")),
        dx=get_datatree_value(data_tree, "dx"),
        map_dx=get_datatree_value(data_tree, "map_dx"),
        surface_h=get_datatree_value(data_tree, "surface_h"),
        bed_h=get_datatree_value(data_tree, "bed_h"),
        rgi_id=get_datatree_value(data_tree, "rgi_id"),
        water_level=get_datatree_value(data_tree, "water_level"),
    )

    if cls_name == "ParabolicBedFlowline":
        flowline = ParabolicBedFlowline(
            bed_shape=get_datatree_value(data_tree, "bed_shape"), **base
        )
    elif cls_name == "RectangularBedFlowline":
        flowline = RectangularBedFlowline(
            widths=get_datatree_value(data_tree, "widths"), **base
        )
    elif cls_name == "TrapezoidalBedFlowline":
        flowline = TrapezoidalBedFlowline(
            widths=get_datatree_value(data_tree, "widths"),
            lambdas=get_datatree_value(data_tree, "lambdas"),
            **base,
        )
    else:  # MixedBedFlowline (default / legacy stores)
        flowline = MixedBedFlowline(
            section=get_datatree_value(data_tree, "section"),
            bed_shape=get_datatree_value(data_tree, "bed_shape"),
            is_trapezoid=get_datatree_value(data_tree, "is_trapezoid"),
            lambdas=get_datatree_value(data_tree, "lambdas"),
            widths_m=get_datatree_value(data_tree, "widths_m"),
            gdir=get_datatree_value(data_tree, "gdir"),
            **base,
        )
        # Mixed-only cached arrays; restore exactly as stored.
        for attribute in ["_sqrt_bed", "_w0_m"]:
            setattr(
                flowline, attribute, get_datatree_value(data_tree, attribute)
            )

    setattr(flowline, "order", get_datatree_value(data_tree, "order"))

    # reconstruct Grid partial (None when no grid params were stored)
    setattr(flowline, "map_trafo", get_map_trafo_from_grid(data_tree))
    return flowline


def get_centerline_from_datatree(data_tree: xr.DataTree):
    """Reconstruct a Centerline object from a DataTree.

    Note that it is not possible to reconstruct a Centerline with all
    the same attributes as the original, because some of these cannot be
    passed to the constructor.
    """
    from oggm import Centerline

    centerline = Centerline(
        line=_validate_linestring(get_datatree_value(data_tree, "line")),
        dx=np.array(get_datatree_value(data_tree, "dx")),
        surface_h=get_datatree_value(data_tree, "surface_h"),
        orig_head=_validate_point(get_datatree_value(data_tree, "orig_head")),
        rgi_id=get_datatree_value(data_tree, "rgi_id"),
        map_dx=get_datatree_value(data_tree, "map_dx"),
    )
    for attribute in [
        "order",
        "_widths",
        "is_rectangular",
        "is_trapezoid",
        "apparent_mb",
        "flux",
        "flux_out",
    ]:
        val = get_datatree_value(data_tree, attribute)
        setattr(centerline, attribute, val)

    # Need to rebuild list of MultiLineStrings from flat coord array
    gw_data = get_datatree_value(data_tree, "geometrical_widths")
    if gw_data is not None:
        coords = np.array(gw_data)
        line_lengths = get_datatree_value(data_tree, "gw_line_lengths")
        counts = get_datatree_value(data_tree, "gw_width_line_counts")
        if line_lengths is None:
            # TODO: legacy (N,2,2) stores from before the ragged fix,
            # drop this logic once pre-fix zarr caches are rebuilt
            widths = []
            for i in range(len(coords)):
                if np.any(np.isnan(coords[i])):
                    widths.append(shapely.geometry.MultiLineString())
                else:
                    widths.append(
                        shapely.geometry.MultiLineString(
                            [shapely.geometry.LineString(coords[i])]
                        )
                    )
        else:
            line_lengths = np.asarray(line_lengths, dtype=np.int64)
            counts = np.asarray(counts, dtype=np.int64)
            splits = np.cumsum(line_lengths)[:-1]
            lines = np.split(coords, splits) if len(line_lengths) else []
            widths = []
            li = 0
            for cnt in counts:
                members = [
                    shapely.geometry.LineString(lines[li + j])
                    for j in range(cnt)
                ]
                li += cnt
                widths.append(shapely.geometry.MultiLineString(members))
        centerline.geometrical_widths = widths

    return centerline


def get_polygon_from_datatree(
    data_tree: xr.DataTree,
) -> shapely.Polygon | shapely.MultiPolygon:
    """Reconstruct a (Multi)Polygon from a DataTree node.

    Inverse of :func:`convert_polygon_to_dataarray`: splits the flat
    ``coords`` array into rings using ``ring_lengths`` and groups them
    into parts via ``ring_poly_idx`` / ``ring_is_exterior``.

    Parameters
    ----------
    data_tree : xr.DataTree
        The DataTree node containing the polygon data.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        The reconstructed (Multi)Polygon object.
    """
    coords = np.asarray(get_datatree_value(data_tree, "vertices"))
    ring_lengths = np.asarray(get_datatree_value(data_tree, "ring_lengths"))
    ring_poly_idx = np.asarray(get_datatree_value(data_tree, "ring_poly_idx"))
    ring_is_exterior = np.asarray(
        get_datatree_value(data_tree, "ring_is_exterior")
    )

    # Split flat coords array back into individual rings
    splits = np.cumsum(ring_lengths)[:-1]
    rings = np.split(coords, splits) if len(ring_lengths) else []

    # Preserve order
    parts = {}
    for ring, poly_idx, is_ext in zip(rings, ring_poly_idx, ring_is_exterior):
        part = parts.setdefault(int(poly_idx), {"exterior": None, "holes": []})
        if bool(is_ext):
            part["exterior"] = ring
        else:
            part["holes"].append(ring)

    polygons = [
        shapely.Polygon(parts[idx]["exterior"], parts[idx]["holes"])
        for idx in sorted(parts)
    ]
    if len(polygons) == 1:
        return polygons[0]

    return shapely.MultiPolygon(polygons)


def get_index_list_from_datatree(data_tree: xr.DataTree) -> list:
    """Reconstruct polygon indices from a DataTree node.

    Used for converting polygons. Inverse of
    :func:`convert_index_list_to_dataarrays`.

    Parameters
    ----------
    data_tree : xr.DataTree
        The DataTree node containing the index list data.

    Returns
    -------
    list of np.ndarray
        Polygon indices as ``(n, 2)`` int arrays.
    """

    coords = np.asarray(
        get_datatree_value(data_tree, "vertices"), dtype=np.int64
    )
    lengths = np.asarray(
        get_datatree_value(data_tree, "lengths"), dtype=np.int64
    )
    if not len(lengths):
        return []
    splits = np.cumsum(lengths)[:-1]

    return [a.reshape(-1, 2) for a in np.split(coords, splits)]


def get_geometries_from_datatree(data_tree: xr.DataTree) -> dict:
    """Reconstruct geometries from a DataTree.

    Returns a flat dict matching the original ``geometries`` pickle:
    polygon children become shapely (Multi)Polygons, the
    ``catchment_indices`` child becomes a list of index arrays, scalar
    root variables (e.g. ``polygon_area``) become plain Python scalars,
    and any other child is recursed into.

    Parameters
    ----------
    data_tree : xr.DataTree
        The DataTree node containing the geometries.

    Returns
    -------
    dict
        A dictionary of geometries, with keys matching the original
        ``geometries`` pickle.
    """
    geometries = {}

    # Root-level variables (scalars such as polygon_area).
    for var in data_tree.data_vars:
        val = data_tree[var].values.copy()
        geometries[var] = val.item() if val.ndim == 0 else val
    for coord in data_tree.coords:
        geometries[coord] = data_tree.coords[coord].values.copy()

    for name, child in data_tree.children.items():
        if not isinstance(child, xr.DataTree):
            geometries[name] = child
        elif child.is_empty:
            geometries[name] = None
        elif child.attrs.get("_geom_kind") == "polygon":
            geometries[name] = get_polygon_from_datatree(child)
        elif child.attrs.get("_geom_kind") == "index_list":
            geometries[name] = get_index_list_from_datatree(child)
        else:
            geometries[name] = get_geometries_from_datatree(child)

    return geometries


def restore_projection(root: xr.DataTree) -> None:
    """Restore the pyproj.Proj object from the stored CRS dictionary.

    Parameters
    ----------
    root : xr.DataTree
        The root DataTree node containing the CRS information.
    """
    if "pyproj_srs" in root.attrs:
        if isinstance(root.attrs["pyproj_srs"], dict):
            crs = pyproj.CRS.from_json_dict(root.attrs["pyproj_srs"])
            root.attrs["pyproj_srs"] = pyproj.Proj(crs)


def get_grid_params_from_partial(p: Callable) -> dict:
    # TODO: Convert from glacier_grid instead of partial.
    grid = p.func.__self__
    grid_parameters = {
        "pyproj_srs": grid.proj.crs.to_json_dict(),
        "nxny": (grid.nx, grid.ny),
        "dxdy": (grid.dx, grid.dy),
        "x0y0": (grid.x0, grid.y0),
        "pixel_ref": grid.pixel_ref,
    }

    return grid_parameters


def get_map_trafo_from_grid(data_tree: xr.DataTree) -> Callable | None:
    """Get a partial function for the map transformation from a DataTree.

    Flowlines created without a gdir/grid (e.g. the PyGEM sandbox) have
    no stored grid params, so reconstruct these without map_trafo.

    Parameters
    ----------
    data_tree : xr.DataTree
        The DataTree node containing the grid parameters.

    Returns
    -------
    Callable or None
        A partial function for the map transformation, or None if grid
        parameters are not available.
    """

    # TODO: Move to change flowline object instead to take a Grid object instead of a gdir.
    if "pyproj_srs" not in data_tree.attrs:
        return None
    map_trafo = Grid(
        proj=data_tree.attrs["pyproj_srs"],
        nxny=(data_tree.attrs["nxny"]),
        dxdy=(data_tree.attrs["dxdy"]),
        x0y0=(data_tree.attrs["x0y0"]),
        pixel_ref=data_tree.attrs["pixel_ref"],
    )
    return partial(map_trafo.ij_to_crs, crs=wgs84)


def convert_linestring_to_dataarray(
    line: shapely.LineString, dims: tuple = ("x", "y")
) -> xr.DataArray:
    """Convert a shapely LineString to a DataArray of coordinates.

    Parameters
    ----------
    line : shapely.LineString
        The line to convert. Any non-LineString input is returned
        unchanged.
    dims : tuple, default ("x", "y")
        Dimension names for the (point, coord) axes. Pass distinct names
        when several linestrings of different lengths are stored in the
        same group, to avoid xarray dimension-size clashes.

    Returns
    -------
    xr.DataArray
        A 2-D DataArray of shape (n_points, 2) with the coordinates of
        the LineString, or the input unchanged if it is not a LineString.
    """
    if not isinstance(line, shapely.LineString):
        return line
    else:
        return xr.DataArray(
            np.array(shapely.geometry.mapping(line)["coordinates"]),
            dims=list(dims),
        )


def convert_point_to_dataarray(
    point: shapely.Point | None,
) -> xr.DataArray | None:
    """Convert a shapely Point to a 1-D DataArray of coordinates.

    Parameters
    ----------
    point : shapely.Point or None
        The point to convert. Any non-Point input is returned unchanged.

    Returns
    -------
    xr.DataArray or None
        A 1-D DataArray of shape (2,) with the coordinates of the Point,
        or the input unchanged if it is not a Point.
    """
    if point is None:
        return None
    if not isinstance(point, shapely.Point):
        return point
    coords = np.array(shapely.get_coordinates(point)).flatten()
    return xr.DataArray(coords, dims=["xy"])


def convert_polygon_to_dataarray(
    polygon: shapely.Polygon,
) -> dict | object:
    """Convert a shapely Polygon/MultiPolygon to DataArrays.

    Interior holes and multiple parts are preserved by flattening every
    ring into a single ``coords`` array with index arrays describing how
    to split it back up (see :func:`_extract_polygon_coords`). The
    result is a dict of ``xr.DataArray``, suitable for
    storing as a single DataTree node.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
        The geometry to convert. Any other input is returned unchanged.

    Returns
    -------
    dict or object
        A dict with ``coords``, ``ring_lengths``, ``ring_poly_idx`` and
        ``ring_is_exterior`` DataArrays, or the input unchanged if it is
        not a (Multi)Polygon.
    """
    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        return polygon

    rings = _extract_polygon_coords(polygon)
    coords = (
        np.concatenate([r[0] for r in rings], axis=0)
        if rings
        else np.zeros((0, 2), dtype=np.float64)
    )
    ring_lengths = np.array([len(r[0]) for r in rings], dtype=np.int64)
    ring_poly_idx = np.array([r[1] for r in rings], dtype=np.int64)
    ring_is_exterior = np.array([r[2] for r in rings], dtype=bool)

    return {
        "vertices": xr.DataArray(
            np.asarray(coords, dtype=np.float64), dims=["vertex", "xy"]
        ),
        "ring_lengths": xr.DataArray(ring_lengths, dims=["ring"]),
        "ring_poly_idx": xr.DataArray(ring_poly_idx, dims=["ring"]),
        "ring_is_exterior": xr.DataArray(ring_is_exterior, dims=["ring"]),
    }


def convert_index_list_to_dataarrays(index_list: list) -> dict | object:
    """Convert a list of polygon indices DataArrays.

    Used for ``catchment_indices``, with index arrays for each
    centerline.They are flattened into a single ``coords`` array plus a
    ``lengths`` array so the list can be split back up when read.

    Parameters
    ----------
    index_list : list
        A list of ``(n_k, 2)`` integer arrays. Any other input is
        returned unchanged.

    Returns
    -------
    dict or object
        A dict with ``coords`` and ``lengths`` DataArrays, or the
        unchanged input if it is not a list of arrays.
    """
    if not isinstance(index_list, list) or not all(
        isinstance(a, np.ndarray) for a in index_list
    ):
        return index_list

    arrays = [np.asarray(a, dtype=np.int64).reshape(-1, 2) for a in index_list]
    coords = (
        np.concatenate(arrays, axis=0)
        if arrays
        else np.zeros((0, 2), dtype=np.int64)
    )
    lengths = np.array([len(a) for a in arrays], dtype=np.int64)

    return {
        "vertices": xr.DataArray(coords, dims=["vertex", "xy"]),
        "lengths": xr.DataArray(lengths, dims=["entry"]),
    }


def get_dict_from_datatree(data_tree: xr.DataTree) -> dict:
    """Convert a DataTree back into a dictionary.

    This will flatten a datatree such that all coordinates and data
    variables match what would be expected in the original pickles.
    """
    data = {}
    for coord in data_tree.coords:
        data[coord] = data_tree.coords[coord].values.copy()
    for var in data_tree.data_vars:
        data[var] = data_tree[var].values.copy()
    for name, child in data_tree.children.items():
        if isinstance(child, xr.DataTree):
            if child.is_empty:
                data[name] = None
            else:
                data[name] = get_dict_from_datatree(child)
        else:
            data[name] = child
    return data


def get_downstream_line_from_pkl(pickle: dict) -> dict:
    """Convert ``downstream_line`` pickle into zarr-compatible structure.

    ``downstream_line`` and ``full_line`` generally differ in length, so
    ``full_line`` is given distinct dim names to avoid clashing on a
    shared dimension when both live in the same zarr group.

    Parameters
    ----------
    pickle : dict
        Data loaded directly from the ``downstream_line`` pickle.

    Returns
    -------
    dict
        The same items as the input, but with the ``downstream_line``
        and ``full_line`` keys converted to DataArrays of coordinates if
        they were originally LineStrings.
    """

    try:
        assert isinstance(pickle, dict)
        downstream_line = pickle["downstream_line"]
    except AssertionError:
        raise TypeError(
            "Input data must be a dictionary."
            "Ensure you are loading from a pickle"
        )
    except KeyError:
        raise KeyError(
            "The pickle must contain a 'downstream_line' key."
            "Check the contents of the pickle."
        )

    """
    Work on a shallow copy so the original dict is not mutated.  If we
    modified the caller's dict in-place the original_data reference in
    write_store would also be changed, causing the pickle fallback to
    store a DataArray instead of the original shapely geometry.
    """
    pickle = dict(pickle)
    pickle["downstream_line"] = convert_linestring_to_dataarray(downstream_line)
    if "full_line" in pickle:
        # None passes through unchanged but LineStrings get distinct dims
        pickle["full_line"] = convert_linestring_to_dataarray(
            pickle["full_line"], dims=("full_line_point", "full_line_coord")
        )

    return pickle


def get_model_flowlines_from_pkl(pickle: list) -> list:
    """Convert ``model_flowlines`` pickle into zarr-compatible structure.

    Note that ``map_trafo`` is a partial and cannot be directly
    serialised to zarr; it is stored separately as group attrs.
    ``gdir`` is a GlacierDirectory and cannot be serialised either;
    it is omitted (``map_trafo`` is reconstructed from the stored grid
    params instead).

    Parameters
    ----------
    pickle : list
        Data loaded directly from the ``model_flowlines`` pickle.

    Returns
    -------
    list[dict]
        One dict per flowline with the attributes needed to reconstruct
        the original :py:class:`oggm.core.flowline.Flowline` (any of the
        four subclasses). The concrete class is recorded separately as a
        node attr by :func:`convert_pickles_to_datatree`.
    """
    from oggm.core.flowline import (
        Flowline,
        MixedBedFlowline,
        ParabolicBedFlowline,
        RectangularBedFlowline,
        TrapezoidalBedFlowline,
    )

    new_pickle = []
    if not isinstance(pickle, list) or not pickle:
        return new_pickle
    try:
        assert all(isinstance(fl, Flowline) for fl in pickle)
        fl_id_to_idx = {id(fl): i for i, fl in enumerate(pickle)}
        for fl in pickle:
            # Attributes common to every Flowline subclass.
            data = {
                "line": convert_linestring_to_dataarray(
                    getattr(fl, "line", None)
                ),
                "dx": getattr(fl, "dx", None),
                "map_dx": getattr(fl, "map_dx", None),
                "surface_h": getattr(fl, "surface_h", None),
                "bed_h": getattr(fl, "bed_h", None),
                "rgi_id": getattr(fl, "rgi_id", None),
                "water_level": getattr(fl, "water_level", None),
                "order": getattr(fl, "order", None),
            }
            # Per-class bed parameters (match each constructor's kwargs).
            if isinstance(fl, MixedBedFlowline):
                lambdas = getattr(fl, "lambdas", None)
                data.update(
                    {
                        "section": getattr(fl, "section", None),
                        "bed_shape": getattr(fl, "bed_shape", None),
                        "is_trapezoid": getattr(fl, "is_trapezoid", None),
                        "widths_m": getattr(fl, "widths_m", None),
                        "lambdas": (
                            getattr(fl, "_lambdas", None)
                            if lambdas is None
                            else lambdas
                        ),
                        "_sqrt_bed": getattr(fl, "_sqrt_bed", None),
                        "_w0_m": getattr(fl, "_w0_m", None),
                    }
                )
            elif isinstance(fl, ParabolicBedFlowline):
                data["bed_shape"] = getattr(fl, "bed_shape", None)
            elif isinstance(fl, RectangularBedFlowline):
                data["widths"] = getattr(fl, "_widths", None)
            elif isinstance(fl, TrapezoidalBedFlowline):
                # Trapezoid reconstructs _w0_m from widths and lambdas, so
                # store the width property (= widths_m / map_dx) and lambdas.
                data["widths"] = getattr(fl, "widths", None)
                data["lambdas"] = getattr(fl, "_lambdas", None)
            # Store the index of the flowline this one flows into (-1 = None).
            flows_to = getattr(fl, "flows_to", None)
            data["_flows_to_list_idx"] = np.int64(
                fl_id_to_idx.get(id(flows_to), -1)
                if flows_to is not None
                else -1
            )
            # Drop None values so xarray can infer dtypes.
            data = {k: v for k, v in data.items() if v is not None}
            new_pickle.append(data)
    except AssertionError:
        raise TypeError(
            "All items in the pickle list must be Flowline instances."
        )

    return new_pickle


def get_inversion_flowlines_from_pkl(pickle: list) -> list[dict]:
    """Convert ``inversion_flowlines`` (or ``centerlines``) pickle into
    zarr-compatible structure.

    Parameters
    ----------
    pickle : list
        Data loaded directly from the pickle – a list of
        :py:class:`oggm.Centerline` objects.

    Returns
    -------
    list[dict]
        One dict per flowline with the attributes needed to reconstruct
        the original Centerline objects.
    """
    from oggm import Centerline

    new_pickle = []
    if not isinstance(pickle, list) or not pickle:
        return new_pickle
    try:
        assert all(isinstance(fl, Centerline) for fl in pickle)
        # Build an id-to-index map so we can store the flows_to list index
        fl_id_to_idx = {id(fl): i for i, fl in enumerate(pickle)}
        for fl in pickle:
            data = {
                "line": convert_linestring_to_dataarray(fl.line),
                "dx": fl.dx,
                "surface_h": fl.surface_h,
                # convert orig_head from shapely Point to DataArray
                "orig_head": convert_point_to_dataarray(
                    getattr(fl, "orig_head", None)
                ),
                "rgi_id": getattr(fl, "rgi_id", None),
                "map_dx": getattr(fl, "map_dx", None),
            }
            # These cannot be passed via Centerline.__init__
            for attribute in [
                "order",
                "_widths",
                "is_rectangular",
                "is_trapezoid",
                "apparent_mb",
                "flux",
                "flux_out",
            ]:
                data[attribute] = getattr(fl, attribute, None)
            # Store index of flowline it flows into (-1 = None)
            # Reconstruct flows_to connections after deserialisation.
            flows_to = getattr(fl, "flows_to", None)
            data["_flows_to_list_idx"] = np.int64(
                fl_id_to_idx.get(id(flows_to), -1)
                if flows_to is not None
                else -1
            )
            gw = getattr(fl, "geometrical_widths", None)
            if gw is not None:
                # Widths are ragged MultiLineStrings, so store flat coord array
                # plus counts to preserve ragged shape
                all_coords = []  # flat coords across all member lines
                line_lengths = []  # vertices per member LineString
                width_line_counts = []  # member lines per width
                for w in gw:
                    if w is None:
                        members = []
                    elif hasattr(w, "geoms"):
                        members = list(w.geoms)
                    else:
                        members = [w]
                    cnt = 0
                    for m in members:
                        if m is None or m.is_empty:
                            continue
                        c = np.asarray(m.coords, dtype=np.float64)
                        all_coords.append(c)
                        line_lengths.append(len(c))
                        cnt += 1
                    width_line_counts.append(cnt)
                coords = (
                    np.concatenate(all_coords, axis=0)
                    if all_coords
                    else np.zeros((0, 2), dtype=np.float64)
                )
                data["geometrical_widths"] = xr.DataArray(
                    coords, dims=["gw_vertex", "gw_coord"]
                )
                data["gw_line_lengths"] = xr.DataArray(
                    np.asarray(line_lengths, dtype=np.int64), dims=["gw_line"]
                )
                data["gw_width_line_counts"] = xr.DataArray(
                    np.asarray(width_line_counts, dtype=np.int64),
                    dims=["gw_width"],
                )
            # so xarray can infer dtypes for each variable
            data = {k: v for k, v in data.items() if v is not None}
            new_pickle.append(data)
    except AssertionError:
        raise TypeError(
            "All items in the pickle list must be Centerline instances."
        )

    return new_pickle


def get_centerlines_from_pkl(pickle: list) -> list[dict]:
    """Convert a ``centerlines`` pickle to zarr-compatible structure.

    Reuses the same format as ``inversion_flowlines`` since both store
    lists of :py:class:`oggm.Centerline` objects.
    """
    return get_inversion_flowlines_from_pkl(pickle)


def get_geometries_from_pkl(pickle: dict) -> dict:
    """Convert a ``geometries`` pickle to zarr-compatible structure.

    Parameters
    ----------
    pickle : dict
        Data loaded directly from the ``geometries`` pickle.

    Returns
    -------
    dict
        Identical to the pickle, but with zarr-compatible values for
        ``polygon_hr``, ``polygon_pix``, ``catchment_indices``, and
        ``downstream_line``. Scalars like ``polygon_area``) are left
        unchanged.
    """
    new_pickle = {}
    for name, geom in pickle.items():
        if "polygon_hr" in name or "polygon_pix" in name:
            new_pickle[name] = convert_polygon_to_dataarray(geom)
        elif "catchment_indices" in name:
            new_pickle[name] = convert_index_list_to_dataarrays(geom)
        elif "downstream_line" in name:
            new_pickle[name] = convert_linestring_to_dataarray(geom)
        else:
            new_pickle[name] = geom

    return new_pickle


def convert_pickles_to_datatree(pickle_data: dict) -> xr.DataTree:
    """Convert a dictionary of pickles into an xarray DataTree."""
    data_tree = xr.DataTree()
    for name, pickle in pickle_data.items():
        try:
            # These are the pickles that require special handling.
            if "downstream_line" in name:
                data = get_downstream_line_from_pkl(pickle)
                data_tree = add_datacube(
                    data_tree=data_tree,
                    datacubes=data,
                    datacube_name=name,
                    overwrite=True,
                )
                continue

            # Centerline lists: centerlines, inversion_flowlines
            if "inversion_flowlines" in name or (
                "centerlines" in name and "inversion" not in name
            ):
                dicts = (
                    get_inversion_flowlines_from_pkl(pickle)
                    if "inversion_flowlines" in name
                    else get_centerlines_from_pkl(pickle)
                )
                sub_tree = xr.DataTree()
                for i, d in enumerate(dicts):
                    sub_tree = add_datacube(
                        data_tree=sub_tree,
                        datacubes=d,
                        datacube_name=str(i),
                        overwrite=True,
                    )
                data_tree[name] = sub_tree
                continue

            if "model_flowlines" in name:
                dicts = get_model_flowlines_from_pkl(pickle)
                sub_tree = xr.DataTree()
                for i, d in enumerate(dicts):
                    # map_trafo can't be serialised
                    from oggm.core.flowline import Flowline

                    fl = (
                        pickle[i]
                        if isinstance(pickle, list) and i < len(pickle)
                        else None
                    )
                    map_trafo = (
                        getattr(fl, "map_trafo", None)
                        if isinstance(fl, Flowline)
                        else None
                    )
                    sub_tree = add_datacube(
                        data_tree=sub_tree,
                        datacubes=d,
                        datacube_name=str(i),
                        overwrite=True,
                    )
                    if isinstance(fl, Flowline):
                        sub_tree[str(i)].attrs["_flowline_class"] = type(
                            fl
                        ).__name__
                    if map_trafo is not None:
                        grid_params = get_grid_params_from_partial(map_trafo)
                        sub_tree[str(i)].attrs.update(grid_params)
                data_tree[name] = sub_tree
                continue

            if "geometries" in name and isinstance(pickle, dict):
                converted = get_geometries_from_pkl(pickle)
                root_vars = {}
                children = {}
                for k, v in converted.items():
                    if isinstance(v, dict) and "ring_lengths" in v:
                        children[k] = ("polygon", v)
                    elif isinstance(v, dict) and "lengths" in v:
                        children[k] = ("index_list", v)
                    elif isinstance(v, xr.DataArray):
                        root_vars[k] = v
                    else:
                        # scalar (e.g. polygon_area) -> 0-d DataArray
                        root_vars[k] = xr.DataArray(v)
                node = xr.DataTree(
                    dataset=xr.Dataset(root_vars) if root_vars else None
                )
                for k, (kind, arrays) in children.items():
                    child = xr.DataTree(dataset=xr.Dataset(arrays))
                    child.attrs["_geom_kind"] = kind
                    node[k] = child
                data_tree[name] = node
                continue

            # Fallback for implicitly supported pickles
            if isinstance(pickle, list) and all(
                isinstance(item, dict) for item in pickle
            ):
                # List of dicts (e.g. inversion_input/output with multiple
                # flowlines): store each dict as a numbered child group.
                sub_tree = xr.DataTree()
                for i, item in enumerate(pickle):
                    sub_tree = add_datacube(
                        data_tree=sub_tree,
                        datacubes=item,
                        datacube_name=str(i),
                        overwrite=True,
                    )
                data_tree[name] = sub_tree
                continue

            if isinstance(pickle, list):
                data = pickle[0]
            elif isinstance(pickle, dict):
                data = pickle
            else:
                raise NotImplementedError

            if isinstance(data, dict):
                data_tree = add_datacube(
                    data_tree=data_tree,
                    datacubes=data,
                    datacube_name=name,
                    overwrite=True,
                )
            # if "model_flowlines" in name:
            #     data_tree.model_flowlines.attrs = data_tree.attrs
        except NotImplementedError as e:
            print(f"Pickle '{name}' is unsupported and was skipped: {e}")

    return data_tree


def add_datacube(
    data_tree: xr.DataTree,
    datacubes: dict,
    datacube_name: str,
    overwrite: bool = False,
) -> xr.DataTree:
    """Add a new dataset as a child group of the DataTree at the root.

    .. note:: The arguments should match those in ``dtcg.GeoZarrHandler.

    Parameters
    ----------
    datacubes : dict
        The dataset to be added.
    datacube_name : str
        Layer name to be used for this node of the tree.
    overwrite : bool
        If True, allow a layer of the same name to be overwritten.

    Returns
    -------
    xr.DataTree
        The updated DataTree with the new datasets.
    """

    if datacube_name in data_tree.children and not overwrite:
        raise ValueError(f"Group '{datacube_name}' already exists.")

    if not isinstance(datacubes, dict):
        raise ValueError(f"Datacubes need to be provided within a dictionary.")

    data_tree[datacube_name] = xr.DataTree.from_dict(
        name=datacube_name, data=datacubes
    )

    return data_tree
