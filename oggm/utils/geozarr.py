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
    """Coerce an object into a LineString."""
    if not isinstance(line, shapely.LineString) and (line is not None):
        line = shapely.LineString(line)
    return line


def _validate_polygon(
    polygon: xr.DataArray | shapely.Polygon,
) -> shapely.Polygon | None:
    """Coerce an object into a LineString."""
    if not isinstance(polygon, shapely.Polygon) and (polygon is not None):
        polygon = shapely.Polygon(polygon)
    return polygon


def _validate_point(
    point: xr.DataArray | np.ndarray | shapely.Point | None,
) -> shapely.Point | None:
    """Coerce a stored DataArray/ndarray back to a shapely Point."""
    if point is None:
        return None
    if isinstance(point, (xr.DataArray, np.ndarray)):
        coords = np.asarray(point).flatten()
        return shapely.geometry.Point(coords)
    return point


def get_datatree_value(
    data_tree: xr.DataTree, attribute: str
) -> xr.DataArray | None:
    if hasattr(data_tree, attribute):
        if isinstance(getattr(data_tree, attribute), xr.DataTree):
            if getattr(data_tree, attribute).is_empty:
                return None
        return getattr(data_tree, attribute).values.copy()
    return None


def get_flowline_from_datatree(data_tree: xr.DataTree):
    """Reconstruct a Flowline object from a DataTree."""
    from oggm.core.flowline import MixedBedFlowline

    flowline = MixedBedFlowline(
        line=_validate_linestring(get_datatree_value(data_tree, "line")),
        dx=get_datatree_value(data_tree, "dx"),
        map_dx=get_datatree_value(data_tree, "map_dx"),
        surface_h=get_datatree_value(data_tree, "surface_h"),
        bed_h=get_datatree_value(data_tree, "bed_h"),
        section=get_datatree_value(data_tree, "section"),
        bed_shape=get_datatree_value(data_tree, "bed_shape"),
        is_trapezoid=get_datatree_value(data_tree, "is_trapezoid"),
        lambdas=get_datatree_value(data_tree, "lambdas"),
        widths_m=get_datatree_value(data_tree, "widths_m"),
        rgi_id=get_datatree_value(data_tree, "rgi_id"),
        water_level=get_datatree_value(data_tree, "water_level"),
        gdir=get_datatree_value(data_tree, "gdir"),
    )
    for attribute in [
        "order",
        "_sqrt_bed",
        "_w0_m",
    ]:
        setattr(flowline, attribute, get_datatree_value(data_tree, attribute))

    # reconstruct Grid partial
    map_trafo = get_map_trafo_from_grid(data_tree)
    setattr(flowline, "map_trafo", map_trafo)
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

    # For widths from (N,2,2) array to MultiLineString
    # Matches original structure by converting NaN row to emtpy MLS
    gw_data = get_datatree_value(data_tree, "geometrical_widths")
    if gw_data is not None:
        gw_array = np.array(gw_data)
        widths = []
        for i in range(len(gw_array)):
            if np.any(np.isnan(gw_array[i])):
                widths.append(shapely.geometry.MultiLineString())
            else:
                widths.append(
                    shapely.geometry.MultiLineString(
                        [shapely.geometry.LineString(gw_array[i])]
                    )
                )
        centerline.geometrical_widths = widths

    return centerline


def restore_projection(root: xr.DataTree) -> None:
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


def get_map_trafo_from_grid(data_tree: xr.DataTree) -> Callable:
    # TODO: Move to change flowline object instead to take a Grid object instead of a gdir.
    map_trafo = Grid(
        proj=data_tree.attrs["pyproj_srs"],
        nxny=(data_tree.attrs["nxny"]),
        dxdy=(data_tree.attrs["dxdy"]),
        x0y0=(data_tree.attrs["x0y0"]),
        pixel_ref=data_tree.attrs["pixel_ref"],
    )
    return partial(map_trafo.ij_to_crs, crs=wgs84)


def convert_linestring_to_dataarray(line: shapely.LineString) -> xr.DataArray:
    """Convert a shapely LineString to a DataArray of coordinates."""
    if not isinstance(line, shapely.LineString):
        return line
    else:
        return xr.DataArray(
            np.array(shapely.geometry.mapping(line)["coordinates"]),
            dims=["x", "y"],
        )


def convert_point_to_dataarray(
    point: shapely.Point | None,
) -> xr.DataArray | None:
    """Convert a shapely Point to a 1-D DataArray of coordinates."""
    if point is None:
        return None
    if not isinstance(point, shapely.Point):
        return point
    coords = np.array(shapely.get_coordinates(point)).flatten()
    return xr.DataArray(coords, dims=["xy"])


def convert_polygon_to_dataarray(
    polygon: shapely.Polygon,
) -> xr.DataArray | shapely.Polygon:
    """Convert a shapely Polygon to a DataArray of coordinates.

    WARNING: This function currently only supports simple Polygons
    without holes. This means Polygons cannot be correctly reconstructed.
    """
    if not isinstance(polygon, shapely.Polygon):
        return polygon
    elif not polygon.geom_type == "MultiPolygon":
        raise NotImplementedError("MultiPolygons are not supported.")
    else:
        exterior_coords = shapely.geometry.mapping(polygon)["coordinates"][0]
        try:
            interior_coords = []
            for interior in polygon.interiors:
                interior_coords += [interior.coords[:]]
            raise NotImplementedError("Polygons with holes are not supported.")
        except NotImplementedError as e:
            print(
                "Warning: Currently polygons with holes cannot be reconstructed."
            )
            return xr.DataArray.from_dict(
                np.array(exterior_coords),
                dims=["x", "y"],
            )


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

    Parameters
    ----------
    pickle : dict
        Data loaded directly from the ``downstream_line`` pickle.

    Returns
    -------
    dict
        The same items as the input, but with the ``downstream_line``
        key converted to a DataArray of coordinates if it was originally
        a LineString.
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
    # Work on a shallow copy so the original dict is not mutated.  If we
    # modified the caller's dict in-place the original_data reference in
    # write_store would also be changed, causing the pickle fallback to
    # store a DataArray instead of the original shapely geometry.
    pickle = dict(pickle)
    coordinates = convert_linestring_to_dataarray(downstream_line)
    pickle["downstream_line"] = coordinates

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
        the original :py:class:`oggm.core.flowline.MixedBedFlowline`.
    """
    from oggm.core.flowline import MixedBedFlowline

    new_pickle = []
    if not isinstance(pickle, list) or not pickle:
        return new_pickle
    try:
        assert all(isinstance(fl, MixedBedFlowline) for fl in pickle)
        fl_id_to_idx = {id(fl): i for i, fl in enumerate(pickle)}
        for fl in pickle:
            data = {
                "line": convert_linestring_to_dataarray(
                    getattr(fl, "line", None)
                ),
                "dx": getattr(fl, "dx", None),
                "map_dx": getattr(fl, "map_dx", None),
                "surface_h": getattr(fl, "surface_h", None),
                "bed_h": getattr(fl, "bed_h", None),
                "section": getattr(fl, "section", None),
                "bed_shape": getattr(fl, "bed_shape", None),
                "is_trapezoid": getattr(fl, "is_trapezoid", None),
                "widths_m": getattr(fl, "widths_m", None),
                "rgi_id": getattr(fl, "rgi_id", None),
                "water_level": getattr(fl, "water_level", None),
                "order": getattr(fl, "order", None),
                "map_trafo": getattr(fl, "map_trafo", None),
                "_sqrt_bed": getattr(fl, "_sqrt_bed", None),
                "_w0_m": getattr(fl, "_w0_m", None),
            }
            lambdas = getattr(fl, "lambdas", None)
            data["lambdas"] = (
                getattr(fl, "_lambdas", None) if lambdas is None else lambdas
            )
            # Store the index of the flowline this one flows into (-1 = None).
            flows_to = getattr(fl, "flows_to", None)
            data["_flows_to_list_idx"] = np.int64(
                fl_id_to_idx.get(id(flows_to), -1)
                if flows_to is not None
                else -1
            )
            # Drop non-serialisable and None values so xarray can infer dtypes
            data = {
                k: v
                for k, v in data.items()
                if v is not None and k != "map_trafo"
            }
            new_pickle.append(data)
    except AssertionError:
        raise TypeError(
            "All items in the pickle list must be MixedBedFlowline instances."
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
                nan_row = np.full((2, 2), np.nan, dtype=np.float64)
                gw_rows = []
                for w in gw:
                    if w is None or w.is_empty:
                        gw_rows.append(nan_row)
                    elif hasattr(w, "geoms"):
                        # MultiLineString: take the first (only) member
                        sub = list(w.geoms)
                        if sub:
                            gw_rows.append(
                                np.array(list(sub[0].coords), dtype=np.float64)
                            )
                        else:
                            gw_rows.append(nan_row)
                    else:
                        # Plain LineString
                        gw_rows.append(
                            np.array(list(w.coords), dtype=np.float64)
                        )
                data["geometrical_widths"] = xr.DataArray(
                    np.array(gw_rows, dtype=np.float64),
                    dims=["width_idx", "vertex", "coord"],
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
                    # map_trafo already filtered out
                    # retrieve from original flowline to get grid params.
                    from oggm.core.flowline import MixedBedFlowline
                    if isinstance(pickle, list) and i < len(pickle) and isinstance(pickle[i], MixedBedFlowline):
                        map_trafo = getattr(pickle[i], "map_trafo", None)
                    else:
                        map_trafo = None
                    sub_tree = add_datacube(
                        data_tree=sub_tree,
                        datacubes=d,
                        datacube_name=str(i),
                        overwrite=True,
                    )
                    if map_trafo is not None:
                        grid_params = get_grid_params_from_partial(map_trafo)
                        sub_tree[str(i)].attrs.update(grid_params)
                data_tree[name] = sub_tree
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
