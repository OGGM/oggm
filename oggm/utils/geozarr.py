"""Zarr utilities for OGGM."""

import xarray as xr
from pathlib import Path
import numpy as np
import os
import shapely

import pyproj
from salem import Grid, wgs84
from functools import partial


def get_pickle_paths(gdir) -> list[Path]:
    """Get all available pickles in the glacier directory."""

    return [Path(f) for f in os.listdir(gdir.dir) if f[-4:] == ".pkl"]


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
        GlacierDirectory object to read the pickles from.
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
) -> shapely.LineString:
    if not isinstance(line, shapely.LineString) and (line is not None):
        line = shapely.LineString(line)
    return line


def _validate_polygon(
    polygon: xr.DataArray | shapely.Polygon,
) -> shapely.Polygon:
    if not isinstance(polygon, shapely.Polygon) and (polygon is not None):
        polygon = shapely.Polygon(polygon)
    return polygon


def get_datatree_value(
    data_tree: xr.DataTree, attribute: str
) -> xr.DataArray | None:
    if hasattr(data_tree, attribute):
        if isinstance(getattr(data_tree, attribute), xr.DataTree):
            if getattr(data_tree, attribute).is_empty:
                return None
        return getattr(data_tree, attribute).values
    return None


def get_flowline_from_datatree(data_tree: xr.DataTree):
    """Reconstruct a Flowline object from a DataTree."""
    from oggm.core.flowline import MixedBedFlowline

    flowline = MixedBedFlowline(
        line=_validate_linestring(get_datatree_value(data_tree, "line").values),
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
        setattr(flowline, attribute, getattr(data_tree, attribute))

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
        line=_validate_linestring(get_datatree_value(data_tree, "line").values),
        dx=np.array(get_datatree_value(data_tree, "dx")),
        surface_h=get_datatree_value(data_tree, "surface_h"),
        orig_head=get_datatree_value(data_tree, "orig_head"),
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
        setattr(centerline, attribute, getattr(data_tree, attribute))

    return centerline


def restore_projection(root: xr.DataTree) -> None:
    if "pyproj_srs" in root.attrs:
        if isinstance(root.attrs["pyproj_srs"], dict):
            crs = pyproj.CRS.from_json_dict(root.attrs["pyproj_srs"])
            root.attrs["pyproj_srs"] = pyproj.Proj(crs)


def get_grid_params_from_partial(p: partial) -> dict:
    grid = p.func.__self__
    grid_parameters = {
        "pyproj_srs": grid.proj.crs.to_json_dict(),
        "nxny": (grid.nx, grid.ny),
        "dxdy": (grid.dx, grid.dy),
        "x0y0": (grid.x0, grid.y0),
        "pixel_ref": grid.pixel_ref,
    }

    return grid_parameters


def get_map_trafo_from_grid(data_tree: xr.DataTree) -> Grid:

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


def convert_polygon_to_dataarray(polygon: shapely.Polygon) -> xr.DataArray:
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
        data[coord] = data_tree.coords[coord].values
    for var in data_tree.data_vars:
        data[var] = data_tree[var].values
    for name, child in data_tree.children.items():
        if isinstance(child, xr.DataTree):
            if child.is_empty:
                data[name] = None
            # else:
            #     data[name] = get_dict_from_datatree(child)
        else:
            data[name] = child
    return data


def get_downstream_line_from_pkl(pickle: dict) -> dict:
    """Convert ``downstream_line`` pickle into zarr-compatible structure.

    Parameters
    ----------
    data : dict
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
    coordinates = convert_linestring_to_dataarray(downstream_line)
    pickle["downstream_line"] = coordinates

    return pickle


def get_model_flowlines_from_pkl(pickle: list) -> list:
    """Convert ``model_flowlines`` pickle into zarr-compatible structure.

    Note that ``map_trafo`` is a partial and cannot be directly
    serialised to zarr.

    Parameters
    ----------
    pickle : dict
        Data loaded directly from the ``model_flowlines`` pickle.

    Returns
    -------
    dict
        Contains all the attributes necessary to reconstruct the
        original Flowline objects.
    """
    from oggm.core.flowline import MixedBedFlowline

    new_pickle = []
    if isinstance(pickle, list):
        if not pickle:
            return new_pickle
        try:
            assert all(
                isinstance(flowline, MixedBedFlowline) for flowline in pickle
            )
            # Get all attributes necessary for reconstructing a Flowline
            data = {
                "line": getattr(pickle[0], "line", None),
                "dx": getattr(pickle[0], "dx", None),
                "map_dx": getattr(pickle[0], "map_dx", None),
                "surface_h": getattr(pickle[0], "surface_h", None),
                "bed_h": getattr(pickle[0], "bed_h", None),
                "section": getattr(pickle[0], "section", None),
                "bed_shape": getattr(pickle[0], "bed_shape", None),
                "is_trapezoid": getattr(pickle[0], "is_trapezoid", None),
                "widths_m": getattr(pickle[0], "widths_m", None),
                "rgi_id": getattr(pickle[0], "rgi_id", None),
                "water_level": getattr(pickle[0], "water_level", None),
                "gdir": getattr(pickle[0], "gdir", None),
                "orig_head": getattr(pickle[0], "orig_head", None),
                "order": getattr(pickle[0], "order", None),
                "map_trafo": getattr(pickle[0], "map_trafo", None),
                "_sqrt_bed": getattr(pickle[0], "_sqrt_bed", None),
                "_w0_m": getattr(pickle[0], "_w0_m", None),
            }
            lambdas = getattr(pickle[0], "lambdas", None)
            if lambdas is None:
                # fallback to _lambdas
                data["lambdas"] = getattr(pickle[0], "_lambdas", None)
            else:
                data["lambdas"] = lambdas

            data["line"] = convert_linestring_to_dataarray(data["line"])
            new_pickle.append(data)

        except AssertionError:
            raise TypeError(
                "All items in the pickle list must be of type Centerline."
                "Check the contents of the pickle."
            )

    return new_pickle


def get_inversion_flowlines_from_pkl(pickle: list) -> dict:
    """Convert ``inversion_flowlines`` pickle into zarr-compatible
    structure.

    Parameters
    ----------
    data : dict
        Data loaded directly from the ``inversion_flowlines`` pickle.

    Returns
    -------
    dict
        Contains all the attributes necessary to reconstruct the
        original Centerline objects.
    """
    from oggm import Centerline

    new_pickle = []
    if isinstance(pickle, list):
        if not pickle:
            return new_pickle
        try:
            assert all(isinstance(flowline, Centerline) for flowline in pickle)
            # Get all attributes necessary for reconstructing a Centerline
            data = {
                "line": pickle[0].line,
                "dx": pickle[0].dx,
                "surface_h": pickle[0].surface_h,
                "orig_head": pickle[0].orig_head,
                "rgi_id": pickle[0].rgi_id,
                "map_dx": pickle[0].map_dx,
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
                data[attribute] = getattr(pickle[0], attribute, None)
            new_pickle.append(data)

        except AssertionError:
            raise TypeError(
                "All items in the pickle list must be of type Centerline."
                "Check the contents of the pickle."
            )

    return new_pickle


def convert_pickles_to_datatree(pickle_data: dict) -> xr.DataTree:
    """Convert a dictionary of pickles into an xarray DataTree."""
    data_tree = xr.DataTree()
    for name, pickle in pickle_data.items():
        try:
            # These are the pickles that require special handling.
            if name == "downstream_line":
                data = get_downstream_line_from_pkl(pickle)
            elif name == "inversion_flowlines":
                data = get_inversion_flowlines_from_pkl(pickle)[0]
            elif name == "model_flowlines":
                data = get_model_flowlines_from_pkl(pickle)
                if data["map_trafo"] is not None:
                    for k, v in get_grid_params_from_partial(data["map_trafo"]):
                        data_tree.attrs[k] = v
                    data.pop("map_trafo", None)

            # Fallback for implicitly supported pickles
            elif isinstance(pickle, list):
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
            if name == "model_flowlines":
                data_tree.model_flowline.attrs = data_tree.attrs
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
