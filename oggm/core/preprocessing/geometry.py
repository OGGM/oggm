"""After the centerlines have been computed, they have to be converted to
flowlines in the sense of OGGM.

This module's tasks are a kind of pre-processor for the inversion. The
centerlines are interpolated to regularly spaced intervals. The flowlines
widths are computed based on geometry and altitude-area distributions. This
ensures that the altitude-area distribution is respected and that the
real (geometrical) area of the glacier is restituted (the gridded
representation of the glacier increases area by a factor ~5%).

The tasks in this module initialise and update the list of Flowline objects::

    -


"""
from __future__ import absolute_import, division

from six.moves import zip
# Built ins
import logging
from itertools import groupby
from collections import Counter
# External libs
import numpy as np
from skimage.graph import route_through_array
import netCDF4
import shapely.geometry as shpg
from shapely.ops import linemerge
import matplotlib._cntr as cntr
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.measurements import label, find_objects
import pandas as pd
import geopandas as gpd
import salem
# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm.core.preprocessing.centerlines import Centerline
from oggm.utils import tuple2int
from oggm import entity_task, divide_task

# Module logger
log = logging.getLogger(__name__)

# Variable needed later
LABEL_STRUCT = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]])


def _line_interpol(line, dx):
    """The shapely interpolate function does not guaranty equally
    spaced points in space. This is what this function is for.

    We construct new points on the line but at constant distance from the
    preceding one.

    Parameters
    ----------
    line: a shapely.geometry.LineString instance
    dx: the spacing

    Returns
    -------
    a list of equally distanced points
    """

    # First point is easy
    points = [line.interpolate(dx/2.)]

    # Continue as long as line is not finished
    while True:
        pref = points[-1]
        pbs = pref.buffer(dx).boundary.intersection(line)
        if pbs.type == 'Point':
            pbs = [pbs]
        # Out of the point(s) that we get, take the one farthest from the top
        refdis = line.project(pref)
        tdis = np.array([line.project(pb) for pb in pbs])
        p = np.where(tdis > refdis)[0]
        if len(p) == 0:
            break
        points.append(pbs[int(p[0])])

    return points

def _line_extend(uline, dline, dx):
    """An extension of _line_interpol with a downstream line to add

    Parameters
    ----------
    uline: a shapely.geometry.LineString instance
    dline: a shapely.geometry.LineString instance
    dx: the spacing

    Returns
    -------
    a shapely.geometry.LineString
    """

    # First points is easy
    points = [shpg.Point(c) for c in uline.coords]

    # Continue as long as line is not finished
    while True:
        pref = points[-1]
        pbs = pref.buffer(dx).boundary.intersection(dline)
        if pbs.type == 'Point':
            pbs = [pbs]
        # Out of the point(s) that we get, take the one farthest from the top
        refdis = dline.project(pref)
        tdis = np.array([dline.project(pb) for pb in pbs])
        p = np.where(tdis > refdis)[0]
        if len(p) == 0:
            break
        points.append(pbs[int(p[0])])

    return shpg.LineString(points)


def _mask_to_polygon(mask, x=None, y=None, gdir=None):
    """Converts a mask to a single polygon.

    The mask should be a single entity with nunataks: I didnt test for more
    than one "blob".

    Parameters
    ----------
    mask: 2d array with ones and zeros
        the mask to convert
    x: 2d array with the coordinates
        if not given it will be generated, give it for optimisation
    y: 2d array with the coordinates
        if not given it will be generated, give it for optimisation
    gdir: GlacierDirectory
        for logging

    Returns
    -------
    (poly, poly_no_nunataks) Shapely polygons
    """

    if (x is None) or (y is None):
        # do it yourself
        ny, nx = mask.shape
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        x, y = np.meshgrid(x, y)

    regions, nregions = label(mask, structure=LABEL_STRUCT)
    if nregions > 1:
        log.debug('%s: we had to cut a blob from the catchment', gdir.rgi_id)
        # Check the size of those
        region_sizes = [np.sum(regions == r) for r in np.arange(1, nregions+1)]
        am = np.argmax(region_sizes)
        # Check not a strange glacier
        sr = region_sizes.pop(am)
        for ss in region_sizes:
            if (ss / sr) > 0.2:
                log.warning('%s: this blob was unusually large', gdir.rgi_id)
        mask[:] = 0
        mask[np.where(regions == (am+1))] = 1

    c = cntr.Cntr(x, y, mask)
    nlist = c.trace(0.5)
    if len(nlist) == 0:
        raise RuntimeError('Mask polygon is empty')
    # The first half are the coordinates. The other stuffs I dont know
    ngeoms = len(nlist)//2 - 1

    # First is the exterior, the rest are nunataks
    e_line = shpg.LinearRing(nlist[0])
    i_lines = [shpg.LinearRing(ipoly) for ipoly in nlist[1:ngeoms+1]]

    poly = shpg.Polygon(e_line, i_lines).buffer(0)
    if not poly.is_valid:
        raise RuntimeError('Mask polygon not valid.')
    poly_no = shpg.Polygon(e_line).buffer(0)
    if not poly_no.is_valid:
        raise RuntimeError('Mask polygon not valid.')
    return poly, poly_no


def _point_width(normals, point, centerline, poly, poly_no_nunataks):
    """ Compute the geometrical width on a specific point.

    Called by catchment_width_geom.

    Parameters
    ----------
    normals: normals of the current point, before, and after
    point: the centerline's point
    centerline: Centerline object
    poly, poly_no_nuntaks: subcatchment polygons

    Returns
    -------
    (width, MultiLineString)
    """

    # How far should the normal vector reach? (make it large)
    far_factor = 150.

    normal = shpg.LineString([shpg.Point(point + normals[0] * far_factor),
                              shpg.Point(point + normals[1] * far_factor)])

    # First use the external boundaries only
    line = normal.intersection(poly_no_nunataks)
    if line.type == 'LineString':
        pass  # Nothing to be done
    elif line.type in ['MultiLineString', 'GeometryCollection']:
        # Take the one that contains the centerline
        oline = None
        for l in line:
            if l.type != 'LineString':
                continue
            if l.intersects(centerline.line):
                oline = l
                break
        if oline is None:
            return np.NaN, shpg.MultiLineString()
        line = oline
    else:
        extext = 'Geometry collection not expected: {}'.format(line.type)
        raise RuntimeError(extext)

    # Then take the nunataks into account
    # Make sure we are always returning a MultiLineString for later
    line = line.intersection(poly)
    if line.type == 'LineString':
        line = shpg.MultiLineString([line])
    elif line.type == 'MultiLineString':
        pass  # nothing to be done
    elif line.type == 'GeometryCollection':
        oline = []
        for l in line:
            if l.type != 'LineString':
                continue
            oline.append(l)
        if len(oline) == 0:
            return np.NaN, shpg.MultiLineString()
        line = shpg.MultiLineString(oline)
    else:
        extext = 'Geometry collection not expected: {}'.format(line.type)
        raise NotImplementedError(extext)

    assert line.type == 'MultiLineString'
    width = np.sum([l.length for l in line])

    return width, line


def _filter_for_altitude_range(widths, wlines, topo):
    """Some width lines have unrealistic lenght and go over the whole
    glacier. Filter them out."""

    # altitude range threshold (if range over the line > threshold, filter it)
    alt_range_th = cfg.PARAMS['width_alt_range_thres']

    while True:
        out_width = widths.copy()
        for i, (wi, wl) in enumerate(zip(widths, wlines)):
            if np.isnan(wi):
                continue
            xc = []
            yc = []
            for dwl in wl:
                # we interpolate at high res and take the int coords
                dwl = shpg.LineString([dwl.interpolate(x, normalized=True)
                                       for x in np.linspace(0., 1., num=100)])
                grouped = groupby(map(tuple, np.rint(dwl.coords)))
                dwl = np.array([x[0] for x in grouped], dtype=np.int)
                xc.extend(dwl[:, 0])
                yc.extend(dwl[:, 1])

            altrange = topo[yc, xc]
            if len(np.where(np.isfinite(altrange))[0]) != 0:
                if (np.nanmax(altrange) - np.nanmin(altrange)) > alt_range_th:
                    out_width[i] = np.NaN

        valid = np.where(np.isfinite(out_width))
        if len(valid[0]) > 0:
            break
        else:
            alt_range_th += 20
            log.warning('Set altitude threshold to {}'.format(alt_range_th))
        if alt_range_th > 1000:
            raise RuntimeError('Problem by altitude filter.')

    return out_width


def _filter_grouplen(arr, minsize=3):
    """Filter out the groups of grid points smaller than minsize

    Parameters
    ----------
    arr : the array to filter (should be False and Trues)
    minsize : the minimum size of the group

    Returns
    -------
    the array, with small groups removed
    """

    # Do it with trues
    r, nr = label(arr)
    nr = [i+1 for i, o in enumerate(find_objects(r)) if (len(r[o]) >= minsize)]
    arr = np.asarray([ri in nr for ri in r])

    # and with Falses
    r, nr = label(~ arr)
    nr = [i+1 for i, o in enumerate(find_objects(r)) if (len(r[o]) >= minsize)]
    arr = ~ np.asarray([ri in nr for ri in r])

    return arr


def _width_change_factor(widths):
    fac = widths[:-1] / widths[1:]
    return fac


@entity_task(log, writes=['catchment_indices'])
@divide_task(log, add_0=False)
def catchment_area(gdir, div_id=None):
    """Compute the catchment areas of each tributary line.

    The idea is to compute the route of lowest cost for any point on the
    glacier to rejoin a centerline. These routes are then put together if
    they belong to the same centerline, thus creating "catchment areas" for
    each centerline.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    # Variables
    cls = gdir.read_pickle('centerlines', div_id=div_id)
    geoms = gdir.read_pickle('geometries', div_id=div_id)
    glacier_pix = geoms['polygon_pix']
    fpath = gdir.get_filepath('gridded_data', div_id=div_id)
    with netCDF4.Dataset(fpath) as nc:
        costgrid = nc.variables['cost_grid'][:]
        mask = nc.variables['glacier_mask'][:]

    # If we have only one centerline this is going to be easy: take the
    # mask and return
    if len(cls) == 1:
        cl_catchments = [np.array(np.nonzero(mask == 1)).T]
        gdir.write_pickle(cl_catchments, 'catchment_indices', div_id=div_id)
        return

    # Initialise costgrid and the "catching" dict
    cost_factor = 0.  # Make it cheap
    dic_catch = dict()
    for i, cl in enumerate(cls):
        x, y = tuple2int(cl.line.xy)
        costgrid[y, x] *= cost_factor
        for x, y in [(int(x), int(y)) for x, y in cl.line.coords]:
            assert (y, x) not in dic_catch
            dic_catch[(y, x)] = set([(y, x)])

    # It is much faster to make the array as small as possible (especially
    # with divides). We have to trick:
    pm = np.nonzero(mask == 1)
    ymi, yma = np.min(pm[0])-1, np.max(pm[0])+2
    xmi, xma = np.min(pm[1])-1, np.max(pm[1])+2
    costgrid = costgrid[ymi:yma, xmi:xma]
    mask = mask[ymi:yma, xmi:xma]

    # Where did we compute the path already?
    computed = np.where(mask == 1, 0, np.nan)

    # Coords of Terminus (converted)
    endcoords = np.array(cls[0].tail.coords[0])[::-1].astype(np.int64)
    endcoords -= [ymi, xmi]

    # Start with all the paths at the boundaries, they are more likely
    # to cover much of the glacier
    for headx, heady in tuple2int(glacier_pix.exterior.coords):
        headcoords = np.array([heady-ymi, headx-xmi])  # convert
        indices, _ = route_through_array(costgrid, headcoords, endcoords)
        inds = np.array(indices).T
        computed[inds[0], inds[1]] = 1
        set_dump = set([])
        for y, x in indices:
            y, x = y+ymi, x+xmi  # back to original
            set_dump.add((y, x))
            if (y, x) in dic_catch:
                dic_catch[(y, x)] = dic_catch[(y, x)].union(set_dump)
                break

    # Repeat for the not yet computed pixels
    while True:
        not_computed = np.where(computed == 0)
        if len(not_computed[0]) == 0:  # All points computed !!
            break
        headcoords = np.array([not_computed[0][0], not_computed[1][0]]).astype(np.int64)
        indices, _ = route_through_array(costgrid, headcoords, endcoords)
        inds = np.array(indices).T
        computed[inds[0], inds[1]] = 1
        set_dump = set([])
        for y, x in indices:
            y, x = y+ymi, x+xmi  # back to original
            set_dump.add((y, x))
            if (y, x) in dic_catch:
                dic_catch[(y, x)] = dic_catch[(y, x)].union(set_dump)
                break

    # For each centerline, make a set of all points flowing into it
    cl_catchments = []
    for cl in cls:
        # Union of all
        cl_catch = set()
        for x, y in [(int(x), int(y)) for x, y in cl.line.coords]:
            cl_catch = cl_catch.union(dic_catch[(y, x)])
        cl_catchments.append(cl_catch)

    # The higher order centerlines will get the points from the upstream
    # ones too. The idea is to store the points which are unique to each
    # centerline: now, in decreasing line order we remove the indices from
    # the tributaries
    cl_catchments = cl_catchments[::-1]
    for i, cl in enumerate(cl_catchments):
        cl_catchments[i] = np.array(list(cl.difference(*cl_catchments[i+1:])))
    cl_catchments = cl_catchments[::-1]  # put it back in order

    # Write the data
    gdir.write_pickle(cl_catchments, 'catchment_indices', div_id=div_id)


def polygon_intersections(gdf):
    """Computes the intersections between all polygons in a GeoDataFrame.

    Parameters
    ----------
    gdf : Geopandas.GeoDataFrame

    Returns
    -------
    a Geodataframe containing the intersections
    """

    out_cols = ['id_1', 'id_2', 'geometry']
    out = gpd.GeoDataFrame(columns=out_cols)

    for i, major in gdf.iterrows():

        # Exterior only
        major_poly = major.geometry.exterior

        # Remove the major from the list
        gdf = gdf.loc[gdf.index != i]

        # Keep catchments which intersect
        gdfs = gdf.loc[gdf.intersects(major_poly)]

        for j, neighbor in gdfs.iterrows():

            # No need to check if we already found the intersect
            if j in out.id_1 or j in out.id_2:
                continue

            # Exterior only
            neighbor_poly = neighbor.geometry.exterior

            # Ok, the actual intersection
            mult_intersect = major_poly.intersection(neighbor_poly)

            # All what follows is to catch all possibilities
            # Should no happen in our simple geometries but ya never know
            if isinstance(mult_intersect, shpg.Point):
                continue
            if isinstance(mult_intersect, shpg.linestring.LineString):
                mult_intersect = [mult_intersect]
            if len(mult_intersect) == 0:
                continue
            mult_intersect = [m for m in mult_intersect if
                              not isinstance(m, shpg.Point)]
            if len(mult_intersect) == 0:
                continue
            mult_intersect = linemerge(mult_intersect)
            if isinstance(mult_intersect, shpg.linestring.LineString):
                mult_intersect = [mult_intersect]
            for line in mult_intersect:
                assert isinstance(line, shpg.linestring.LineString)
                line = gpd.GeoDataFrame([[i, j, line]],
                                        columns=out_cols)
                out = out.append(line)

    return out


@entity_task(log, writes=['flowline_catchments', 'catchments_intersects'])
@divide_task(log, add_0=False)
def catchment_intersections(gdir, div_id=None):
    """Computes the intersections between the catchments.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    catchment_indices = gdir.read_pickle('catchment_indices', div_id=div_id)
    xmesh, ymesh = np.meshgrid(np.arange(0, gdir.grid.nx, 1),
                               np.arange(0, gdir.grid.ny, 1))

    # Loop over the lines
    mask = np.zeros((gdir.grid.ny, gdir.grid.nx))

    gdfc = gpd.GeoDataFrame()
    for i, ci in enumerate(catchment_indices):
        # Catchment polygon
        mask[:] = 0
        mask[tuple(ci.T)] = 1
        _, poly_no = _mask_to_polygon(mask, x=xmesh, y=ymesh, gdir=gdir)
        gdfc.loc[i, 'geometry'] = poly_no

    gdfi = polygon_intersections(gdfc)

    # We project them onto the mercator proj before writing. This is a bit
    # inefficient (they'll be projected back later), but it's more sustainable
    gdfc.crs = gdir.grid
    gdfi.crs = gdir.grid
    salem.transform_geopandas(gdfc, gdir.grid.proj, inplace=True)
    salem.transform_geopandas(gdfi, gdir.grid.proj, inplace=True)
    if hasattr(gdfc.crs, 'srs'):
        # salem uses pyproj
        gdfc.crs = gdfc.crs.srs
        gdfi.crs = gdfi.crs.srs
    gdfc.to_file(gdir.get_filepath('flowline_catchments', div_id=div_id))
    if len(gdfi) > 0:
        gdfi.to_file(gdir.get_filepath('catchments_intersects',
                                       div_id=div_id))


@entity_task(log, writes=['inversion_flowlines'])
@divide_task(log, add_0=False)
def initialize_flowlines(gdir, div_id=None):
    """ Transforms the geometrical Centerlines in the more "physical"
    "Inversion Flowlines".

    This interpolates the centerlines on a regular spacing (i.e. not the
    grid's (i, j) indices. Cuts out the tail of the tributaries to make more
    realistic junctions.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    # variables
    cls = gdir.read_pickle('centerlines', div_id=div_id)

    # Initialise the flowlines
    dx = cfg.PARAMS['flowline_dx']
    lid = int(cfg.PARAMS['flowline_junction_pix'])
    fls = []

    # Topo for heights
    fpath = gdir.get_filepath('gridded_data', div_id=div_id)
    with netCDF4.Dataset(fpath) as nc:
        topo = nc.variables['topo_smoothed'][:]

    # Bilinear interpolation
    # Geometries coordinates are in "pixel centered" convention, i.e
    # (0, 0) is also located in the center of the pixel
    xy = (np.arange(0, gdir.grid.ny-0.1, 1),
          np.arange(0, gdir.grid.nx-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo)

    # Smooth window
    sw = cfg.PARAMS['flowline_height_smooth']

    for ic, cl in enumerate(cls):
        points = _line_interpol(cl.line, dx)

        # For tributaries, remove the tail
        if ic < (len(cls)-1):
            points = points[0:-lid]

        new_line = shpg.LineString(points)

        # Interpolate heights
        x, y = new_line.xy
        hgts = interpolator((y, x))

        # If smoothing, this is the moment
        hgts = gaussian_filter1d(hgts, sw)

        l = Centerline(new_line, dx=dx, surface_h=hgts)
        l.order = cl.order
        fls.append(l)

    # All objects are initialized, now we can link them.
    for cl, fl in zip(cls, fls):
        if cl.flows_to is None:
            continue
        fl.set_flows_to(fls[cls.index(cl.flows_to)])

    # Write the data
    gdir.write_pickle(fls, 'inversion_flowlines', div_id=div_id)


@entity_task(log, writes=['inversion_flowlines'])
@divide_task(log, add_0=False)
def catchment_width_geom(gdir, div_id=None):
    """Compute geometrical catchment widths for each point of the flowlines.

    Updates the 'inversion_flowlines' save file.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    # variables
    flowlines = gdir.read_pickle('inversion_flowlines', div_id=div_id)
    catchment_indices = gdir.read_pickle('catchment_indices', div_id=div_id)
    xmesh, ymesh = np.meshgrid(np.arange(0, gdir.grid.nx, 1),
                               np.arange(0, gdir.grid.ny, 1))

    # Topography is to filter the unrealistic lines afterwards.
    # I take the non-smoothed topography
    # I remove the boundary pixs because they are likely to be higher
    fpath = gdir.get_filepath('gridded_data', div_id=div_id)
    with netCDF4.Dataset(fpath) as nc:
        topo = nc.variables['topo'][:]
        mask_ext = nc.variables['glacier_ext'][:]
        mask_glacier = nc.variables['glacier_mask'][:]
    topo[np.where(mask_glacier == 0)] = np.NaN
    topo[np.where(mask_ext == 1)] = np.NaN

    # Intersects between catchments/glaciers
    gdfi = gpd.GeoDataFrame(columns=['geometry'])
    if gdir.has_file('catchments_intersects', div_id=div_id):
        # read and transform to grid
        gdf = gpd.read_file(gdir.get_filepath('catchments_intersects',
                                              div_id=div_id))
        salem.transform_geopandas(gdf, gdir.grid, inplace=True)
        gdfi = pd.concat([gdfi, gdf[['geometry']]])
    if gdir.has_file('intersects', div_id=0):
        # read and transform to grid
        gdf = gpd.read_file(gdir.get_filepath('intersects', div_id=0))
        salem.transform_geopandas(gdf, gdir.grid, inplace=True)
        gdfi = pd.concat([gdfi, gdf[['geometry']]])

    # apply a buffer to be sure we get the intersects right. Be generous
    gdfi = gdfi.buffer(1.5)

    # Filter parameters
    # Number of pixels to arbitrarily remove at junctions
    jpix = int(cfg.PARAMS['flowline_junction_pix'])

    # Loop over the lines
    mask = np.zeros((gdir.grid.ny, gdir.grid.nx))
    for fl, ci in zip(flowlines, catchment_indices):

        n = len(fl.dis_on_line)

        widths = np.zeros(n)
        wlines = []

        # Catchment polygon
        mask[:] = 0
        mask[tuple(ci.T)] = 1
        poly, poly_no = _mask_to_polygon(mask, x=xmesh, y=ymesh, gdir=gdir)

        # First guess widths
        for i, (normal, pcoord) in enumerate(zip(fl.normals, fl.line.coords)):
            width, wline = _point_width(normal, pcoord, fl, poly, poly_no)
            widths[i] = width
            wlines.append(wline)

        valid = np.where(np.isfinite(widths))
        if len(valid[0]) == 0:
            errmsg = '{}: first guess widths went wrong.'.format(gdir.rgi_id)
            raise RuntimeError(errmsg)

        # Ok now the entire centerline is computed.
        # I take all these widths for geometrically valid, and see if they
        # intersect with our buffered catchment/glacier intersections
        touches_border = []
        for wg in wlines:
            touches_border.append(np.any(gdfi.intersects(wg)))
        touches_border = _filter_grouplen(touches_border, minsize=5)

        # we filter the lines which have a large altitude range
        fil_widths = _filter_for_altitude_range(widths, wlines, topo)

        # Filter +- widths at junction points
        for fid in fl.inflow_indices:
            i0 = np.clip(fid-jpix, jpix/2, n-jpix/2).astype(np.int64)
            i1 = np.clip(fid+jpix+1, jpix/2, n-jpix/2).astype(np.int64)
            fil_widths[i0:i1] = np.NaN

        valid = np.where(np.isfinite(fil_widths))
        if len(valid[0]) == 0:
            # This happens very rarely. Just pick the middle and
            # the correction task should do the rest
            log.warning('{}: width filtering too strong.'.format(gdir.rgi_id))
            fil_widths = widths[np.int(len(widths) / 2.)]

        # Write it in the objects attributes
        assert len(fil_widths) == n
        fl.widths = fil_widths
        fl.geometrical_widths = wlines
        fl.touches_border = touches_border

    # Overwrite pickle
    gdir.write_pickle(flowlines, 'inversion_flowlines', div_id=div_id)


@entity_task(log, writes=['inversion_flowlines'])
def catchment_width_correction(gdir, div_id=None):
    """Corrects for NaNs and inconsistencies in the geometrical widths.

    Interpolates missing values, ensures consistency of the
    surface-area distribution AND with the geometrical area of the glacier
    polygon, avoiding errors due to gridded representation.

    Updates the 'inversion_flowlines' save file.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    # The code below makes of this task a "special" divide task.
    # We keep it as is and remove the divide task decorator
    if div_id is None:
        # This is the original call
        # This time instead of just looping over the divides we add a test
        # to check for the conservation of the shapefile's area.
        area = 0.
        divides = []
        for i in gdir.divide_ids:
            log.info('%s: width correction, divide %d', gdir.rgi_id, i)
            fls = catchment_width_correction(gdir, div_id=i)
            for fl in fls:
                area += np.sum(fl.widths) * fl.dx
            divides.append(fls)

        # Final correction - because of the raster, the gridded area of the
        # glacier is not that of the actual geometry. correct for that
        fac = gdir.rgi_area_km2 / (area * gdir.grid.dx**2 * 10**-6)
        log.debug('%s: corrected widths with a factor %.2f', gdir.rgi_id, fac)
        for i in gdir.divide_ids:
            fls = divides[i-1]
            for fl in fls:
                fl.widths *= fac
            # Overwrite centerlines
            gdir.write_pickle(fls, 'inversion_flowlines', div_id=i)
        return None

    # variables
    flowlines = gdir.read_pickle('inversion_flowlines', div_id=div_id)
    catchment_indices = gdir.read_pickle('catchment_indices', div_id=div_id)

    # Topography for altitude-area distribution
    # I take the non-smoothed topography and remove the borders
    fpath = gdir.get_filepath('gridded_data', div_id=div_id)
    with netCDF4.Dataset(fpath) as nc:
        topo = nc.variables['topo'][:]
        ext = nc.variables['glacier_ext'][:]
    topo[np.where(ext==1)] = np.NaN

    # Param
    nmin = int(cfg.PARAMS['min_n_per_bin'])

    # Per flowline (important so that later, the indices can be moved)
    catchment_heights = []
    for ci in catchment_indices:
        _t = topo[tuple(ci.T)][:]
        catchment_heights.append(list(_t[np.isfinite(_t)]))

    # Loop over lines in a reverse order
    for fl, catch_h in zip(flowlines, catchment_heights):

        # Interpolate widths
        widths = utils.interp_nans(fl.widths)
        widths = np.clip(widths, 0.1, np.max(widths))

        # Smooth them
        widths = utils.smooth1d(widths)
        assert np.max(_width_change_factor(widths)[:-5]) < 2.

        # Get topo per catchment and per flowline point
        fhgt = fl.surface_h

        # Sometimes, the centerline does not reach as high as each pix on the
        # glacier. (e.g. RGI40-11.00006)
        catch_h = np.clip(catch_h, 0, np.max(fhgt))

        # Max and mins for the histogram
        maxh = np.max(fhgt)
        if fl.flows_to is None:
            minh = np.min(fhgt)
            catch_h = np.clip(catch_h, minh, np.max(catch_h))
        else:
            minh = np.min(fhgt)  # Min just for flowline (this has reasons)

        # Now decide on a binsize which ensures at least N element per bin
        bsize = cfg.PARAMS['base_binsize']
        while True:
            maxb = utils.nicenumber(maxh, 1)
            minb = utils.nicenumber(minh, 1, lower=True)
            bins = np.arange(minb, maxb+bsize+0.01, bsize)
            minb = np.min(bins)

            # Ignore the topo pixels below the last bin
            tmp_ght = catch_h[np.where(catch_h >= minb)]

            topo_digi = np.digitize(tmp_ght, bins) - 1  # I prefer the left
            fl_digi = np.digitize(fhgt, bins) - 1  # I prefer the left
            if nmin == 1:
                # No need for complicated count
                _c = set(topo_digi)
                _fl = set(fl_digi)
            else:
                # Keep indexes with at least n counts
                _c = Counter(topo_digi.tolist())
                _c = set([k for (k, v) in _c.items() if v >= nmin])
                _fl = Counter(fl_digi.tolist())
                _fl = set([k for (k, v) in _fl.items() if v >= nmin])

            ref_set = set(range(len(bins)-1))
            if (_c == ref_set) and (_fl == ref_set):
                # For each bin, the width(s) have to represent the "real" area
                nw = widths.copy()
                for bi in range(len(bins) - 1):
                    bintopoarea = len(np.where(topo_digi == bi)[0])
                    wherewiths = np.where(fl_digi == bi)
                    binflarea = np.sum(nw[wherewiths]) * fl.dx
                    nw[wherewiths] = (bintopoarea / binflarea) * nw[wherewiths]
                if np.nanmax(_width_change_factor(nw)[:-5]) < 2.:
                    break
            bsize += 5

            # Add a security for infinite loops
            if bsize > 250:
                nmin -= 1
                bsize = cfg.PARAMS['base_binsize']
                log.warning('%s: reduced min n per bin to %d', gdir.rgi_id,
                            nmin)
                if nmin == 0:
                    raise RuntimeError('NO binsize could be chosen for: '
                                       '{}'.format(gdir.rgi_id))
        if bsize > 150:
            log.warning('%s: chosen binsize %d', gdir.rgi_id, bsize)
        else:
            log.debug('%s: chosen binsize %d', gdir.rgi_id, bsize)

        # Now keep the good topo pixels and send the unattributed ones to the
        # next flowline
        tosend = list(catch_h[np.where(catch_h < minb)])
        if (len(tosend) > 0) and (fl.flows_to is not None):
            ide = flowlines.index(fl.flows_to)
            catchment_heights[ide] = np.append(catchment_heights[ide], tosend)
        if (len(tosend) > 0) and (fl.flows_to is None):
            raise RuntimeError('This should not happen')

        # Write it
        fl.widths = nw

    return flowlines
