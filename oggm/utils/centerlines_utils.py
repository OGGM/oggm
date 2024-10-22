"""
centerlines_utils is a utility class that aims to separate/abstract the clutter of the helper
functions that aid the main CenterLines class. This is a first attempt at potentially doing
similar things to other components of this project in order to make the basic functionality
easier to expand upon in the future by separating the public functionality
of Centerlines away from the utilities that it needs. There is also potential for these more
universal funcs to apply to other components, making for easier reuse later as they are not
coupled as private functions within CenterLines.

@Author: hbbaker
"""

# Built ins
import copy
import logging

from itertools import groupby

# External Libs
import shapely.ops
import shapely.geometry as shpg
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import (gaussian_filter1d, distance_transform_edt,
                           label, find_objects)

from oggm import cfg
from oggm.exceptions import InvalidGeometryError
from oggm.utils import line_interpol


def filter_heads(heads, heads_height, radius, polygon):
    """Filter the head candidates following Kienholz et al. (2014), Ch. 4.1.2

    Parameters
    ----------
    heads : list of shapely.geometry.Point instances
        The heads to filter out (in raster coordinates).
    heads_height : list
        The heads altitudes.
    radius : float
        The radius around each head to search for potential challengers
    polygon : shapely.geometry.Polygon class instance
        The glacier geometry (in raster coordinates).

    Returns
    -------
    a list of shapely.geometry.Point instances with the "bad ones" removed
    """

    heads = copy.copy(heads)
    heads_height = copy.copy(heads_height)

    i = 0
    # I think a "while" here is ok: we remove the heads forwards only
    while i < len(heads):
        head = heads[i]
        pbuffer = head.buffer(radius)
        inter_poly = pbuffer.intersection(polygon.exterior)
        if inter_poly.geom_type in ['MultiPolygon',
                                    'GeometryCollection',
                                    'MultiLineString']:
            #  In the case of a junction point, we have to do a check
            # http://lists.gispython.org/pipermail/community/
            # 2015-January/003357.html
            if inter_poly.geom_type == 'MultiLineString':
                inter_poly = shapely.ops.linemerge(inter_poly)

            if inter_poly.geom_type != 'LineString':
                # keep the local polygon only
                for sub_poly in inter_poly.geoms:
                    if sub_poly.intersects(head):
                        inter_poly = sub_poly
                        break
        elif inter_poly.geom_type == 'LineString':
            inter_poly = shpg.Polygon(np.asarray(inter_poly.xy).T)
        elif inter_poly.geom_type == 'Polygon':
            pass
        else:
            extext = ('Geometry collection not expected: '
                      '{}'.format(inter_poly.geom_type))
            raise InvalidGeometryError(extext)

        # Find other points in radius and in polygon
        _heads = [head]
        _z = [heads_height[i]]
        for op, z in zip(heads[i+1:], heads_height[i+1:]):
            if inter_poly.intersects(op):
                _heads.append(op)
                _z.append(z)

        # If alone, go to the next point
        if len(_heads) == 1:
            i += 1
            continue

        # If not, keep the highest
        _w = np.argmax(_z)

        for head in _heads:
            if not (head is _heads[_w]):
                heads_height = np.delete(heads_height, heads.index(head))
                heads.remove(head)

    return heads, heads_height


def filter_lines(lines, heads, k, r):
    """Filter the centerline candidates by length.

    Kienholz et al. (2014), Ch. 4.3.1

    Parameters
    ----------
    lines : list of shapely.geometry.LineString instances
        The lines to filter out (in raster coordinates).
    heads :  list of shapely.geometry.Point instances
        The heads corresponding to the lines.
    k : float
        A buffer (in raster coordinates) to cut around the selected lines
    r : float
        The lines shorter than r will be filtered out.

    Returns
    -------
    (lines, heads) a list of the new lines and corresponding heads
    """

    olines = []
    oheads = []
    ilines = copy.copy(lines)

    lastline = None
    while len(ilines) > 0:  # loop as long as we haven't filtered all lines
        if len(olines) > 0:  # enter this after the first step only

            toremove = lastline.buffer(k)  # buffer centerlines the last line
            tokeep = []
            for l in ilines:
                # loop over all remaining lines and compute their diff
                # to the last longest line
                diff = l.difference(toremove)
                if diff.geom_type == 'MultiLineString':
                    # Remove the lines that have no head
                    diff = list(diff.geoms)
                    for il in diff:
                        hashead = False
                        for h in heads:
                            if il.intersects(h):
                                hashead = True
                                diff = il
                                break
                        if hashead:
                            break
                        else:
                            diff = None
                # keep this head line only if it's long enough
                if diff is not None and diff.length > r:
                    # Fun fact. The heads can be cut by the buffer too
                    diff = shpg.LineString(l.coords[0:2] + diff.coords[2:])
                    tokeep.append(diff)
            ilines = tokeep

        # it could happen that we're done at this point
        if len(ilines) == 0:
            break

        # Otherwise keep the longest one and continue
        lengths = np.array([])
        for l in ilines:
            lengths = np.append(lengths, l.length)
        ll = ilines[np.argmax(lengths)]

        ilines.remove(ll)
        if len(olines) > 0:
            # the cut line's last point is not guaranteed
            # to on straight coordinates. Remove it
            olines.append(shpg.LineString(np.asarray(ll.xy)[:, 0:-1].T))
        else:
            olines.append(ll)
        lastline = ll

    # add the corresponding head to each line
    for l in olines:
        for h in heads:
            if l.intersects(h):
                oheads.append(h)
                break

    return olines, oheads


def filter_lines_slope(lines, heads, topo, gdir, min_slope):
    """Filter the centerline candidates by slope: if they go up, remove

    Kienholz et al. (2014), Ch. 4.3.1

    Parameters
    ----------
    lines : list of shapely.geometry.LineString instances
        The lines to filter out (in raster coordinates).
    topo : the glacier topography
    gdir : the glacier directory for simplicity
    min_slope: rad

    Returns
    -------
    (lines, heads) a list of the new lines and corresponding heads
    """

    dx_cls = cfg.PARAMS['flowline_dx']
    lid = int(cfg.PARAMS['flowline_junction_pix'])
    sw = cfg.PARAMS['flowline_height_smooth']

    # Bilinear interpolation
    # Geometries coordinates are in "pixel centered" convention, i.e
    # (0, 0) is also located in the center of the pixel
    xy = (np.arange(0, gdir.grid.ny-0.1, 1),
          np.arange(0, gdir.grid.nx-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo.astype(np.float64))

    olines = [lines[0]]
    oheads = [heads[0]]
    for line, head in zip(lines[1:], heads[1:]):

        # The code below mimics what initialize_flowlines will do
        # this is a bit smelly but necessary
        points = line_interpol(line, dx_cls)

        # For tributaries, remove the tail
        points = points[0:-lid]

        new_line = shpg.LineString(points)

        # Interpolate heights
        x, y = new_line.xy
        hgts = interpolator((np.array(y), np.array(x)))

        # If smoothing, this is the moment
        hgts = gaussian_filter1d(hgts, sw)

        # Finally slope
        slope = np.arctan(-np.gradient(hgts, dx_cls*gdir.grid.dx))

        # And altitude range
        z_range = np.max(hgts) - np.min(hgts)

        # arbitrary threshold with which we filter the lines, otherwise bye bye
        if np.sum(slope >= min_slope) >= 5 and z_range > 10:
            olines.append(line)
            oheads.append(head)

    return olines, oheads


def width_change_factor(widths):
    fac = widths[:-1] / widths[1:]
    return fac


def filter_grouplen(arr, minsize=3):
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


def line_extend(uline, dline, dx):
    """Adds a downstream line to a flowline

    Parameters
    ----------
    uline: a shapely.geometry.LineString instance
    dline: a shapely.geometry.LineString instance
    dx: the spacing

    Returns
    -------
    (line, line) : two shapely.geometry.LineString instances. The first
    contains the newly created (long) line, the second only the interpolated
    downstream part (useful for other analyses)
    """

    # First points is easy
    points = [shpg.Point(c) for c in uline.coords]

    if len(points) == 0:
        # eb flowline
        dpoints = [shpg.Point(dline.coords[0])]
        points = [shpg.Point(dline.coords[0])]
    else:
        dpoints = []

    # Continue as long as line is not finished
    while True:
        pref = points[-1]
        pbs = pref.buffer(dx).boundary.intersection(dline)
        if pbs.geom_type in ['LineString', 'GeometryCollection']:
            # Very rare
            pbs = pref.buffer(dx+1e-12).boundary.intersection(dline)
        if pbs.geom_type == 'Point':
            pbs = [pbs]

        try:
            # Shapely v2 compat
            pbs = pbs.geoms
        except AttributeError:
            pass

        # Out of the point(s) that we get, take the one farthest from the top
        refdis = dline.project(pref)
        tdis = np.array([dline.project(pb) for pb in pbs])
        p = np.where(tdis > refdis)[0]
        if len(p) == 0:
            break
        points.append(pbs[int(p[0])])
        dpoints.append(pbs[int(p[0])])

    return shpg.LineString(points), shpg.LineString(dpoints)


def projection_point(centerline, point):
    """Projects a point on a line and returns the closest integer point
    guaranteed to be on the line, and guaranteed to be far enough from the
    head and tail.

    Parameters
    ----------
    centerline : Centerline instance
    point : Shapely Point geometry

    Returns
    -------
    (flow_point, ind_closest): Shapely Point and indice in the line
    """
    prdis = centerline.line.project(point, normalized=False)
    ind_closest = np.argmin(np.abs(centerline.dis_on_line - prdis)).item()
    flow_point = shpg.Point(centerline.line.coords[int(ind_closest)])
    return flow_point


def make_costgrid(mask, ext, z):
    """Computes a costgrid following Kienholz et al. (2014) Eq. (2)

    Parameters
    ----------
    mask : numpy.array
        The glacier mask.
    ext : numpy.array
        The glacier boundaries' mask.
    z : numpy.array
        The terrain height.

    Returns
    -------
    numpy.array of the costgrid
    """

    dis = np.where(mask, distance_transform_edt(mask), np.nan)
    z = np.where(mask, z, np.nan)

    dmax = np.nanmax(dis)
    zmax = np.nanmax(z)
    zmin = np.nanmin(z)
    cost = ((dmax - dis) / dmax * cfg.PARAMS['f1']) ** cfg.PARAMS['a'] + \
           ((z - zmin) / (zmax - zmin) * cfg.PARAMS['f2']) ** cfg.PARAMS['b']

    # This is new: we make the cost to go over boundaries
    # arbitrary high to avoid the lines to jump over adjacent boundaries
    cost[np.where(ext)] = np.nanmax(cost[np.where(ext)]) * 50

    return np.where(mask, cost, np.inf)


def get_terminus_coord(gdir, ext_yx, zoutline):
    """This finds the terminus coordinate of the glacier.

     There is a special case for marine terminating glaciers/
     """

    perc = cfg.PARAMS['terminus_search_percentile']
    deltah = cfg.PARAMS['terminus_search_altitude_range']

    if gdir.is_tidewater and (perc > 0):
        # There is calving

        # find the lowest percentile
        plow = np.percentile(zoutline, perc).astype(np.int64)

        # the minimum altitude in the glacier
        mini = np.min(zoutline)

        # indices of where in the outline the altitude is lower than the qth
        # percentile and lower than 100m higher, than the minimum altitude
        ind = np.where((zoutline < plow) & (zoutline < (mini + deltah)))[0]

        # We take the middle of this area
        try:
            ind_term = ind[np.round(len(ind) / 2.).astype(int)]
        except IndexError:
            # Sometimes the default perc is not large enough
            try:
                # Repeat
                perc *= 2
                plow = np.percentile(zoutline, perc).astype(np.int64)
                mini = np.min(zoutline)
                ind = np.where((zoutline < plow) &
                               (zoutline < (mini + deltah)))[0]
                ind_term = ind[np.round(len(ind) / 2.).astype(int)]
            except IndexError:
                # Last resort
                ind_term = np.argmin(zoutline)
    else:
        # easy: just the minimum
        ind_term = np.argmin(zoutline)

    return np.asarray(ext_yx)[:, ind_term].astype(np.int64)


def normalize(n):
    """Computes the normals of a vector n.

    Returns
    -------
    the two normals (n1, n2)
    """
    nn = n / np.sqrt(np.sum(n*n))
    n1 = np.array([-nn[1], nn[0]])
    n2 = np.array([nn[1], -nn[0]])
    return n1, n2
