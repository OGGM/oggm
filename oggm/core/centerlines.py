""" Compute the centerlines according to Kienholz et al (2014) - with
modifications.

The output is a list of Centerline objects, stored as a list in a pickle.
The order of the list is important since the lines are
sorted per order (hydrological flow level), from the lower orders (upstream)
to the higher orders (downstream). Several tools later on rely on this order
so don't mess with it.

References::

    Kienholz, C., Rich, J. L., Arendt, a. a., and Hock, R. (2014).
        A new method for deriving glacier centerlines applied to glaciers in
        Alaska and northwest Canada. The Cryosphere, 8(2), 503-519.
        doi:10.5194/tc-8-503-2014
"""
# Built ins
import logging
import copy
from itertools import groupby
from collections import Counter
# External libs
import numpy as np
import pandas as pd
import salem
import shapely.ops
import geopandas as gpd
import scipy.signal
import shapely.geometry as shpg
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import label, find_objects
from skimage.graph import route_through_array
# Locals
import oggm.cfg as cfg
from oggm.cfg import GAUSSIAN_KERNEL
from salem import lazy_property
from oggm import utils
from oggm.utils import tuple2int, line_interpol, interp_nans
from oggm import entity_task
from oggm.exceptions import (InvalidParamsError, InvalidGeometryError,
                             GeometryError)

# Module logger
log = logging.getLogger(__name__)

# Variable needed later
LABEL_STRUCT = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]])


class Centerline(object):
    """A Centerline has geometrical and flow rooting properties.

    It is instanciated and updated by _join_lines() exclusively
    """

    def __init__(self, line, dx=None, surface_h=None, orig_head=None,
                 rgi_id=None):
        """ Instantiate."""

        self.line = None  # Shapely LineString
        self.head = None  # Shapely Point
        self.tail = None  # Shapely Point
        self.dis_on_line = None  # Shapely Point
        self.nx = None  # Shapely Point
        self.is_glacier = None  # Shapely Point
        self.set_line(line)  # Init all previous properties

        self.order = None  # Hydrological flow level (~ Strahler number)

        # These are computed at run time by compute_centerlines
        self.flows_to = None  # pointer to a Centerline object (when available)
        self.flows_to_point = None  # point of the junction in flows_to
        self.inflows = []  # list of Centerline instances (when available)
        self.inflow_points = []  # junction points

        # Optional attrs
        self.dx = dx  # dx in pixels (assumes the line is on constant dx
        self._surface_h = surface_h
        self._widths = None
        self.is_rectangular = None
        self.orig_head = orig_head  # Useful for debugging and for filtering
        self.geometrical_widths = None  # these are kept for plotting and such
        self.apparent_mb = None  # Apparent MB, NOT weighted by width.
        self.mu_star = None  # the mu* associated with the apparent mb
        self.mu_star_is_valid = False  # if mu* leeds to good flux, keep it
        self.flux = None  # Flux (kg m-2)
        self.flux_needs_correction = False  # whether this branch was baaad
        self.rgi_id = rgi_id  # Usefull if line is used with another glacier

    def set_flows_to(self, other, check_tail=True, last_point=False):
        """Find the closest point in "other" and sets all the corresponding
        attributes. Btw, it modifies the state of "other" too.

        Parameters
        ----------
        other: an other centerline
        """

        self.flows_to = other

        if check_tail:
            # Project the point and Check that its not too close
            prdis = other.line.project(self.tail, normalized=False)
            ind_closest = np.argmin(np.abs(other.dis_on_line - prdis)).item()
            n = len(other.dis_on_line)
            if n >= 9:
                ind_closest = np.clip(ind_closest, 4, n-5)
            elif n >= 7:
                ind_closest = np.clip(ind_closest, 3, n-4)
            elif n >= 5:
                ind_closest = np.clip(ind_closest, 2, n-3)
            p = shpg.Point(other.line.coords[int(ind_closest)])
            self.flows_to_point = p
        elif last_point:
            self.flows_to_point = other.tail
        else:
            # just the closest
            self.flows_to_point = _projection_point(other, self.tail)
        other.inflow_points.append(self.flows_to_point)
        other.inflows.append(self)

    def set_line(self, line):
        """Update the Shapely LineString coordinate.

        Parameters
        ----------
        line: a shapely.geometry.LineString
        """

        self.nx = len(line.coords)
        self.line = line
        dis = [line.project(shpg.Point(co)) for co in line.coords]
        self.dis_on_line = np.array(dis)
        xx, yy = line.xy
        self.head = shpg.Point(xx[0], yy[0])
        self.tail = shpg.Point(xx[-1], yy[-1])

    @lazy_property
    def flows_to_indice(self):
        """Indices instead of geometry"""

        tofind = self.flows_to_point.coords[0]
        for i, p in enumerate(self.flows_to.line.coords):
            if p == tofind:
                ind = i
        assert ind is not None
        return ind

    @lazy_property
    def inflow_indices(self):
        """Indices instead of geometries"""

        inds = []
        for p in self.inflow_points:
            ind = [i for (i, pi) in enumerate(self.line.coords)
                   if (p.coords[0] == pi)]
            inds.append(ind[0])
        assert len(inds) == len(self.inflow_points)
        return inds

    @lazy_property
    def normals(self):
        """List of (n1, n2) normal vectors at each point.

        We use second order derivatives for smoother widths.
        """

        pcoords = np.array(self.line.coords)

        normals = []
        # First
        normal = np.array(pcoords[1, :] - pcoords[0, :])
        normals.append(_normalize(normal))
        # Second
        normal = np.array(pcoords[2, :] - pcoords[0, :])
        normals.append(_normalize(normal))
        # Others
        for (bbef, bef, cur, aft, aaft) in zip(pcoords[:-4, :],
                                               pcoords[1:-3, :],
                                               pcoords[2:-2, :],
                                               pcoords[3:-1, :],
                                               pcoords[4:, :]):
            normal = np.array(aaft + 2*aft - 2*bef - bbef)
            normals.append(_normalize(normal))
        # One before last
        normal = np.array(pcoords[-1, :] - pcoords[-3, :])
        normals.append(_normalize(normal))
        # Last
        normal = np.array(pcoords[-1, :] - pcoords[-2, :])
        normals.append(_normalize(normal))

        return normals

    @property
    def widths(self):
        """Needed for overriding later"""
        return self._widths

    @widths.setter
    def widths(self, value):
        self._widths = value

    @property
    def surface_h(self):
        """Needed for overriding later"""
        return self._surface_h

    @surface_h.setter
    def surface_h(self, value):
        self._surface_h = value

    def set_apparent_mb(self, mb, mu_star=None):
        """Set the apparent mb and flux for the flowline.

        MB is expected in kg m-2 yr-1 (= mm w.e. yr-1)

        This should happen in line order, otherwise it will be wrong.

        Parameters
        ----------
        mu_star : float
            if appropriate, the mu_star associated with this apparent mb
        """

        self.apparent_mb = mb
        self.mu_star = mu_star

        # Add MB to current flux and sum
        # no more changes should happen after that
        flux_needs_correction = False
        flux = np.cumsum(self.flux + mb * self.widths * self.dx)

        # We filter only lines with two negative grid points, the
        # rest we can cope with
        if flux[-2] < 0:
            flux_needs_correction = True

        self.flux = flux
        self.flux_needs_correction = flux_needs_correction

        # Add to outflow. That's why it should happen in order
        if self.flows_to is not None:
            n = len(self.flows_to.line.coords)
            ide = self.flows_to_indice
            if n >= 9:
                gk = GAUSSIAN_KERNEL[9]
                self.flows_to.flux[ide-4:ide+5] += gk * flux[-1]
            elif n >= 7:
                gk = GAUSSIAN_KERNEL[7]
                self.flows_to.flux[ide-3:ide+4] += gk * flux[-1]
            elif n >= 5:
                gk = GAUSSIAN_KERNEL[5]
                self.flows_to.flux[ide-2:ide+3] += gk * flux[-1]


def _filter_heads(heads, heads_height, radius, polygon):
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
        if inter_poly.type in ['MultiPolygon',
                               'GeometryCollection',
                               'MultiLineString']:
            #  In the case of a junction point, we have to do a check
            # http://lists.gispython.org/pipermail/community/
            # 2015-January/003357.html
            if inter_poly.type == 'MultiLineString':
                inter_poly = shapely.ops.linemerge(inter_poly)

            if inter_poly.type is not 'LineString':
                # keep the local polygon only
                for sub_poly in inter_poly:
                    if sub_poly.intersects(head):
                        inter_poly = sub_poly
                        break
        elif inter_poly.type is 'LineString':
            inter_poly = shpg.Polygon(np.asarray(inter_poly.xy).T)
        elif inter_poly.type is 'Polygon':
            pass
        else:
            extext = ('Geometry collection not expected: '
                      '{}'.format(inter_poly.type))
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


def _filter_lines(lines, heads, k, r):
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
                if diff.type is 'MultiLineString':
                    # Remove the lines that have no head
                    diff = list(diff)
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
                            raise GeometryError('Head not found')
                # keep this head line only if it's long enough
                if diff.length > r:
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
            # the cutted line's last point is not guaranteed
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


def _filter_lines_slope(lines, heads, topo, gdir):
    """Filter the centerline candidates by slope: if they go up, remove

    Kienholz et al. (2014), Ch. 4.3.1

    Parameters
    ----------
    lines : list of shapely.geometry.LineString instances
        The lines to filter out (in raster coordinates).
    topo : the glacier topogaphy
    gdir : the glacier directory for simplicity

    Returns
    -------
    (lines, heads) a list of the new lines and corresponding heads
    """

    dx_cls = cfg.PARAMS['flowline_dx']
    lid = int(cfg.PARAMS['flowline_junction_pix'])
    sw = cfg.PARAMS['flowline_height_smooth']

    # Here we use a conservative value
    min_slope = np.deg2rad(cfg.PARAMS['min_slope'])

    # Bilinear interpolation
    # Geometries coordinates are in "pixel centered" convention, i.e
    # (0, 0) is also located in the center of the pixel
    xy = (np.arange(0, gdir.grid.ny-0.1, 1),
          np.arange(0, gdir.grid.nx-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo)

    olines = [lines[0]]
    oheads = [heads[0]]
    for line, head in zip(lines[1:], heads[1:]):

        # The code below mimicks what initialize_flowlines will do
        # this is a bit smelly but necessary
        points = line_interpol(line, dx_cls)

        # For tributaries, remove the tail
        points = points[0:-lid]

        new_line = shpg.LineString(points)

        # Interpolate heights
        x, y = new_line.xy
        hgts = interpolator((y, x))

        # If smoothing, this is the moment
        hgts = gaussian_filter1d(hgts, sw)

        # Finally slope
        slope = np.arctan(-np.gradient(hgts, dx_cls*gdir.grid.dx))

        # arbitrary threshold with which we filter the lines, otherwise bye bye
        if np.sum(slope >= min_slope) >= 5:
            olines.append(line)
            oheads.append(head)

    return olines, oheads


def _projection_point(centerline, point):
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


def _join_lines(lines, heads):
    """Re-joins the lines that have been cut by _filter_lines

     Compute the rooting scheme.

    Parameters
    ----------
    lines: list of shapely lines instances

    Returns
    -------
    Centerline instances, updated with flow routing properties
     """

    olines = [Centerline(l, orig_head=h) for l, h
              in zip(lines[::-1], heads[::-1])]
    nl = len(olines)
    if nl == 1:
        return olines

    # per construction the line cannot flow in a line placed before in the list
    for i, l in enumerate(olines):

        last_point = shpg.Point(*l.line.coords[-1])

        totest = olines[i+1:]
        dis = [last_point.distance(t.line) for t in totest]
        flow_to = totest[np.argmin(dis)]

        flow_point = _projection_point(flow_to, last_point)

        # Interpolate to finish the line, bute force:
        # we interpolate 20 points, round them, remove consecutive duplicates
        endline = shpg.LineString([last_point, flow_point])
        endline = shpg.LineString([endline.interpolate(x, normalized=True)
                                   for x in np.linspace(0., 1., num=20)])
        # we keep all coords without the first AND the last
        grouped = groupby(map(tuple, np.rint(endline.coords)))
        endline = [x[0] for x in grouped][1:-1]

        # We're done
        l.set_line(shpg.LineString(l.line.coords[:] + endline))
        l.set_flows_to(flow_to, check_tail=False)

        # The last one has nowhere to flow
        if i+2 == nl:
            break

    return olines[::-1]


def line_order(line):
    """Recursive search for the line's hydrological level.

    Parameters
    ----------
    line: a Centerline instance

    Returns
    -------
    The line's order
    """

    if len(line.inflows) == 0:
        return 0
    else:
        levels = [line_order(s) for s in line.inflows]
        return np.max(levels) + 1


def line_inflows(line):
    """Recursive search for all inflows of the given line.

    Parameters
    ----------
    line: a Centerline instance

    Returns
    -------
    A list of lines (including the line itself) sorted in order
    """

    out = set([line])
    for l in line.inflows:
        out = out.union(line_inflows(l))

    out = np.array(list(out))
    return list(out[np.argsort([o.order for o in out])])


def _make_costgrid(mask, ext, z):
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

    dis = np.where(mask, distance_transform_edt(mask), np.NaN)
    z = np.where(mask, z, np.NaN)

    dmax = np.nanmax(dis)
    zmax = np.nanmax(z)
    zmin = np.nanmin(z)
    cost = ((dmax - dis) / dmax * cfg.PARAMS['f1']) ** cfg.PARAMS['a'] + \
           ((z - zmin) / (zmax - zmin) * cfg.PARAMS['f2']) ** cfg.PARAMS['b']

    # This is new: we make the cost to go over boundaries
    # arbitrary high to avoid the lines to jump over adjacent boundaries
    cost[np.where(ext)] = np.nanmax(cost[np.where(ext)]) * 50

    return np.where(mask, cost, np.Inf)


def _get_terminus_coord(gdir, ext_yx, zoutline):
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
            ind_term = ind[np.round(len(ind) / 2.).astype(np.int)]
        except IndexError:
            # Sometimes the default perc is not large enough
            try:
                # Repeat
                perc *= 2
                plow = np.percentile(zoutline, perc).astype(np.int64)
                mini = np.min(zoutline)
                ind = np.where((zoutline < plow) &
                               (zoutline < (mini + deltah)))[0]
                ind_term = ind[np.round(len(ind) / 2.).astype(np.int)]
            except IndexError:
                # Last resort
                ind_term = np.argmin(zoutline)
    else:
        # easy: just the minimum
        ind_term = np.argmin(zoutline)

    return np.asarray(ext_yx)[:, ind_term].astype(np.int64)


def _normalize(n):
    """Computes the normals of a vector n.

    Returns
    -------
    the two normals (n1, n2)
    """
    nn = n / np.sqrt(np.sum(n*n))
    n1 = np.array([-nn[1], nn[0]])
    n2 = np.array([nn[1], -nn[0]])
    return n1, n2


def _get_centerlines_heads(gdir, ext_yx, zoutline, single_fl,
                           glacier_mask, topo, geom, poly_pix):

    # Size of the half window to use to look for local maximas
    maxorder = np.rint(cfg.PARAMS['localmax_window'] / gdir.grid.dx)
    maxorder = np.clip(maxorder, 5., np.rint((len(zoutline) / 5.)))
    heads_idx = scipy.signal.argrelmax(zoutline, mode='wrap',
                                       order=maxorder.astype(np.int64))
    if single_fl or len(heads_idx[0]) <= 1:
        # small glaciers with one or less heads: take the absolute max
        heads_idx = (np.atleast_1d(np.argmax(zoutline)),)

    # Remove the heads that are too low
    zglacier = topo[np.where(glacier_mask)]
    head_threshold = np.percentile(zglacier, (1./3.)*100)
    _heads_idx = heads_idx[0][np.where(zoutline[heads_idx] > head_threshold)]
    if len(_heads_idx) == 0:
        # this is for baaad ice caps where the outline is far off in altitude
        _heads_idx = [heads_idx[0][np.argmax(zoutline[heads_idx])]]
    heads_idx = _heads_idx
    heads = np.asarray(ext_yx)[:, heads_idx]
    heads_z = zoutline[heads_idx]
    # careful, the coords are in y, x order!
    heads = [shpg.Point(x, y) for y, x in zip(heads[0, :],
                                              heads[1, :])]

    # get radius of the buffer according to Kienholz eq. (1)
    radius = cfg.PARAMS['q1'] * geom['polygon_area'] + cfg.PARAMS['q2']
    radius = np.clip(radius, 0, cfg.PARAMS['rmax'])
    radius /= gdir.grid.dx  # in raster coordinates
    # Plus our criteria, quite useful to remove short lines:
    radius += cfg.PARAMS['flowline_junction_pix'] * cfg.PARAMS['flowline_dx']
    log.debug('(%s) radius in raster coordinates: %.2f',
              gdir.rgi_id, radius)

    # OK. Filter and see.
    log.debug('(%s) number of heads before radius filter: %d',
              gdir.rgi_id, len(heads))
    heads, heads_z = _filter_heads(heads, heads_z, radius, poly_pix)
    log.debug('(%s) number of heads after radius filter: %d',
              gdir.rgi_id, len(heads))

    return heads


def _line_extend(uline, dline, dx):
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
    dpoints = []

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
        dpoints.append(pbs[int(p[0])])

    return shpg.LineString(points), shpg.LineString(dpoints)


@entity_task(log, writes=['centerlines', 'gridded_data'])
def compute_centerlines(gdir, heads=None):
    """Compute the centerlines following Kienholz et al., (2014).

    They are then sorted according to the modified Strahler number:
    http://en.wikipedia.org/wiki/Strahler_number

    This function does not initialize a :py:class:`oggm.Centerline` but
    calculates routes along the topography and makes a
    :py:class:`shapely.Linestring` object from them.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    heads : list, optional
        list of shapely.geometry.Points to use as line heads (default is to
        compute them like Kienholz did)
    """

    # Params
    single_fl = not cfg.PARAMS['use_multiple_flowlines']
    do_filter_slope = cfg.PARAMS['filter_min_slope']

    # Force single flowline for ice caps
    if gdir.is_icecap:
        single_fl = True

    if 'force_one_flowline' in cfg.PARAMS:
        raise InvalidParamsError('`force_one_flowline` is deprecated')

    # open
    geom = gdir.read_pickle('geometries')
    grids_file = gdir.get_filepath('gridded_data')
    with utils.ncDataset(grids_file) as nc:
        # Variables
        glacier_mask = nc.variables['glacier_mask'][:]
        glacier_ext = nc.variables['glacier_ext'][:]
        topo = nc.variables['topo_smoothed'][:]
    poly_pix = geom['polygon_pix']

    # Find for local maximas on the outline
    x, y = tuple2int(poly_pix.exterior.xy)
    ext_yx = tuple(reversed(poly_pix.exterior.xy))
    zoutline = topo[y[:-1], x[:-1]]  # last point is first point

    # For diagnostics
    is_first_call = False
    if heads is None:
        # This is the default for when no filter is yet applied
        is_first_call = True
        heads = _get_centerlines_heads(gdir, ext_yx, zoutline, single_fl,
                                       glacier_mask, topo, geom, poly_pix)

    # Cost array
    costgrid = _make_costgrid(glacier_mask, glacier_ext, topo)

    # Terminus
    t_coord = _get_terminus_coord(gdir, ext_yx, zoutline)

    # Compute the routes
    lines = []
    for h in heads:
        h_coord = np.asarray(h.xy)[::-1].astype(np.int64)
        indices, _ = route_through_array(costgrid, h_coord, t_coord)
        lines.append(shpg.LineString(np.array(indices)[:, [1, 0]]))
    log.debug('(%s) computed the routes', gdir.rgi_id)

    # Filter the shortest lines out
    dx_cls = cfg.PARAMS['flowline_dx']
    radius = cfg.PARAMS['flowline_junction_pix'] * dx_cls
    radius += 6 * dx_cls
    olines, oheads = _filter_lines(lines, heads, cfg.PARAMS['kbuffer'], radius)
    log.debug('(%s) number of heads after lines filter: %d',
              gdir.rgi_id, len(olines))

    # Filter the lines which are going up instead of down
    if do_filter_slope:
        olines, oheads = _filter_lines_slope(olines, oheads, topo, gdir)
        log.debug('(%s) number of heads after slope filter: %d',
                  gdir.rgi_id, len(olines))

    # And rejoin the cutted tails
    olines = _join_lines(olines, oheads)

    # Adds the line level
    for cl in olines:
        cl.order = line_order(cl)

    # And sort them per order !!! several downstream tasks  rely on this
    cls = []
    for i in np.argsort([cl.order for cl in olines]):
        cls.append(olines[i])

    # Final check
    if len(cls) == 0:
        raise GeometryError('({}) no valid centerline could be '
                            'found!'.format(gdir.rgi_id))

    # Write the data
    gdir.write_pickle(cls, 'centerlines')

    if is_first_call:
        # For diagnostics of filtered centerlines
        gdir.add_to_diagnostics('n_orig_centerlines', len(cls))


@entity_task(log, writes=['downstream_line'])
def compute_downstream_line(gdir):
    """Computes the Flowline along the unglaciated downstream topography

    The idea is simple: starting from the glacier tail, compute all the routes
    to all local minimas found at the domain edge. The cheapest is "The One".

    The rest of the job (merging centerlines + downstream into
    one single glacier is realized by
    :py:func:`~oggm.tasks.init_present_time_glacier`).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # For tidewater glaciers no need for all this
    if gdir.is_tidewater:
        return

    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo_smoothed'][:]
        glacier_ext = nc.variables['glacier_ext'][:]

    # Look for the starting points
    p = gdir.read_pickle('centerlines')[-1].tail
    head = (int(p.y), int(p.x))

    # Make going up very costy
    topo = topo**4

    # We add an artificial cost as distance from the glacier
    # This should have to much influence on mountain glaciers but helps for
    # tidewater-candidates
    topo = topo + distance_transform_edt(1 - glacier_ext)

    # Make going up very costy
    topo = topo**2

    # Variables we gonna need: the outer side of the domain
    xmesh, ymesh = np.meshgrid(np.arange(0, gdir.grid.nx, 1, dtype=np.int64),
                               np.arange(0, gdir.grid.ny, 1, dtype=np.int64))
    _h = [topo[:, 0], topo[0, :], topo[:, -1], topo[-1, :]]
    _x = [xmesh[:, 0], xmesh[0, :], xmesh[:, -1], xmesh[-1, :]]
    _y = [ymesh[:, 0], ymesh[0, :], ymesh[:, -1], ymesh[-1, :]]

    # Find the way out of the domain
    min_cost = np.Inf
    min_len = np.Inf
    line = None
    for h, x, y in zip(_h, _x, _y):
        ids = scipy.signal.argrelmin(h, order=10, mode='wrap')
        if np.all(h == 0):
            # Test every fifth (we don't really care)
            ids = [np.arange(0, len(h), 5)]
        for i in ids[0]:
            lids, cost = route_through_array(topo, head, (y[i], x[i]))
            if ((cost < min_cost) or
                    ((cost == min_cost) and (len(lids) < min_len))):
                min_cost = cost
                min_len = len(lids)
                line = shpg.LineString(np.array(lids)[:, [1, 0]])
    if line is None:
        raise GeometryError('Downstream line not found')

    cl = gdir.read_pickle('inversion_flowlines')[-1]
    lline, dline = _line_extend(cl.line, line, cl.dx)
    out = dict(full_line=lline, downstream_line=dline)
    gdir.write_pickle(out, 'downstream_line')


def _approx_parabola(x, y, y0=0):
    """Fit a parabola to the equation y = a x**2 + y0

    Parameters
    ----------
    x : array
       the x axis variabls
    y : array
       the dependant variable
    y0 : float, optional
       the intercept

    Returns
    -------
    [a, 0, y0]
    """
    # y=ax**2+y0
    x, y = np.array(x), np.array(y)
    a = np.sum(x ** 2 * (y - y0)) / np.sum(x ** 4)
    return np.array([a, 0, y0])


def _parabola_error(x, y, f):
    # f is an array represents polynom
    x, y = np.array(x), np.array(y)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = sum(abs((np.polyval(f, x) - y) / y)) / len(x)
    return out


class HashablePoint(shpg.Point):
    def __hash__(self):
        return hash(tuple((self.x, self.y)))


def _parabolic_bed_from_topo(gdir, idl, interpolator):
    """this returns the parabolic bedhape for all points on idl"""

    # Volume area scaling formula for the probable ice thickness
    h_mean = 0.034 * gdir.rgi_area_km2**0.375 * 1000
    gnx, gny = gdir.grid.nx, gdir.grid.ny

    # Far Factor
    r = 40
    # number of points
    cs_n = 20

    # normals
    ns = [i[0] for i in idl.normals]
    cs = []
    donot_compute = []

    for pcoords, n in zip(idl.line.coords, ns):
        xi, yi = pcoords
        vx, vy = n
        modul = np.sqrt(vx ** 2 + vy ** 2)
        ci = []
        _isborder = False
        for ro in np.linspace(0, r / 2.0, cs_n):
            t = ro / modul
            cp1 = HashablePoint(xi + t * vx, yi + t * vy)
            cp2 = HashablePoint(xi - t * vx, yi - t * vy)

            # check if out of the frame
            if not (0 < cp2.y < gny - 1) or \
                    not (0 < cp2.x < gnx - 1) or \
                    not (0 < cp1.y < gny - 1) or \
                    not (0 < cp1.x < gnx - 1):
                _isborder = True

            ci.append((cp1, ro))
            ci.append((cp2, -ro))

        ci = list(set(ci))
        cs.append(ci)
        donot_compute.append(_isborder)

    bed = []
    for ic, (cc, dontcomp) in enumerate(zip(cs, donot_compute)):

        if dontcomp:
            bed.append(np.NaN)
            continue

        z = []
        ro = []
        for i in cc:
            z.append(interpolator((i[0].y, i[0].x)))
            ro.append(i[1])
        aso = np.argsort(ro)
        ro, z = np.array(ro)[aso], np.array(z)[aso]

        # find top of parabola
        roHead = ro[np.argmin(z)]
        zero = np.argmin(z)  # it is index of roHead/zHead
        zHead = np.amin(z)

        dsts = abs(h_mean + zHead - z)

        # find local minima in set of distances
        extr = scipy.signal.argrelextrema(dsts, np.less, mode='wrap')
        if len(extr[0]) == 0:
            bed.append(np.NaN)
            continue

        # from local minima find that with the minimum |x|
        idx = extr[0][np.argmin(abs(ro[extr]))]

        # x<0 => x=0
        # (|x|+x)/2
        roN = ro[int((abs(zero - abs(zero - idx)) + zero - abs(
            zero - idx)) / 2):zero + abs(zero - idx) + 1]
        zN = z[int((abs(zero - abs(zero - idx)) + zero - abs(
            zero - idx)) / 2):zero + abs(zero - idx) + 1]
        roNx = roN - roHead
        # zN=zN-zHead#

        p = _approx_parabola(roNx, zN, y0=zHead)

        # shift parabola to the ds-line
        p2 = np.copy(p)
        p2[2] = z[ro == 0]

        err = _parabola_error(roN, zN, p2) * 100

        # The original implementation of @anton-ub stored all three parabola
        # params. We just keep the one important here for now
        if err < 1.5:
            bed.append(p2[0])
        else:
            bed.append(np.NaN)

    bed = np.asarray(bed)
    assert len(bed) == idl.nx
    pvalid = np.sum(np.isfinite(bed)) / len(bed) * 100
    log.debug('(%s) percentage of valid parabolas in downstream: %d',
              gdir.rgi_id, int(pvalid))

    # Scale for dx (we worked in grid coords but need meters)
    bed = bed / gdir.grid.dx**2

    # interpolation, filling the gaps
    default = cfg.PARAMS['default_parabolic_bedshape']
    bed_int = interp_nans(bed, default=default)

    # We forbid super small shapes (important! This can lead to huge volumes)
    # Sometimes the parabola fits in flat areas are very good, implying very
    # flat parabolas.
    bed_int = bed_int.clip(cfg.PARAMS['downstream_min_shape'])

    # Smoothing
    bed_ma = pd.Series(bed_int)
    bed_ma = bed_ma.rolling(window=5, center=True, min_periods=1).mean()
    return bed_ma.values


@entity_task(log, writes=['downstream_line'])
def compute_downstream_bedshape(gdir):
    """The bedshape obtained by fitting a parabola to the line's normals.

    Also computes the downstream's altitude.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # For tidewater glaciers no need for all this
    if gdir.is_tidewater:
        return

    # We make a flowline out of the downstream for simplicity
    tpl = gdir.read_pickle('inversion_flowlines')[-1]
    cl = gdir.read_pickle('downstream_line')['downstream_line']
    cl = Centerline(cl, dx=tpl.dx)

    # Topography
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo_smoothed'][:]
        x = nc.variables['x'][:]
        y = nc.variables['y'][:]
    xy = (np.arange(0, len(y)-0.1, 1), np.arange(0, len(x)-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo)

    bs = _parabolic_bed_from_topo(gdir, cl, interpolator)
    assert len(bs) == cl.nx
    assert np.all(np.isfinite(bs))

    # Interpolate heights for later
    xx, yy = cl.line.xy
    hgts = interpolator((yy, xx))
    assert len(hgts) >= 5

    # If smoothing, this is the moment
    hgts = gaussian_filter1d(hgts, cfg.PARAMS['flowline_height_smooth'])

    # write output
    out = gdir.read_pickle('downstream_line')
    out['bedshapes'] = bs
    out['surface_h'] = hgts
    gdir.write_pickle(out, 'downstream_line')


def _mask_to_polygon(mask, gdir=None):
    """Converts a mask to a single polygon.

    The mask should be a single entity with nunataks: I didnt test for more
    than one "blob".

    Parameters
    ----------
    mask: 2d array with ones and zeros
        the mask to convert
    gdir: GlacierDirectory
        for logging

    Returns
    -------
    (poly, poly_no_nunataks) Shapely polygons
    """

    regions, nregions = label(mask, structure=LABEL_STRUCT)
    if nregions > 1:
        rid = ''
        if gdir is not None:
            rid = gdir.rgi_id
        log.debug('(%s) we had to cut a blob from the catchment', rid)
        # Check the size of those
        region_sizes = [np.sum(regions == r) for r in np.arange(1, nregions+1)]
        am = np.argmax(region_sizes)
        # Check not a strange glacier
        sr = region_sizes.pop(am)
        for ss in region_sizes:
            if (ss / sr) > 0.2:
                log.warning('(%s) this blob was unusually large', rid)
        mask[:] = 0
        mask[np.where(regions == (am+1))] = 1

    nlist = measure.find_contours(mask, 0.5)
    # First is the exterior, the rest are nunataks
    e_line = shpg.LinearRing(nlist[0][:, ::-1])
    i_lines = [shpg.LinearRing(ipoly[:, ::-1]) for ipoly in nlist[1:]]

    poly = shpg.Polygon(e_line, i_lines).buffer(0)
    if not poly.is_valid:
        raise GeometryError('Mask to polygon conversion error.')
    poly_no = shpg.Polygon(e_line).buffer(0)
    if not poly_no.is_valid:
        raise GeometryError('Mask to polygon conversion error.')
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
        raise InvalidGeometryError(extext)

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
        raise InvalidGeometryError(extext)

    assert line.type == 'MultiLineString'
    width = np.sum([l.length for l in line])

    return width, line


def _filter_small_slopes(hgt, dx, min_slope=0):
    """Masks out slopes with NaN until the slope if all valid points is at
    least min_slope (in degrees).
    """

    min_slope = np.deg2rad(min_slope)
    slope = np.arctan(-np.gradient(hgt, dx))  # beware the minus sign
    # slope at the end always OK
    slope[-1] = min_slope

    # Find the locs where it doesn't work and expand till we got everything
    slope_mask = np.where(slope >= min_slope, slope, np.NaN)
    r, nr = label(~np.isfinite(slope_mask))
    for objs in find_objects(r):
        obj = objs[0]
        i = 0
        while True:
            i += 1
            i0 = objs[0].start-i
            if i0 < 0:
                break
            ngap = obj.stop - i0 - 1
            nhgt = hgt[[i0, obj.stop]]
            current_slope = np.arctan(-np.gradient(nhgt, ngap * dx))
            if i0 <= 0 or current_slope[0] >= min_slope:
                break
        slope_mask[i0:obj.stop] = np.NaN
    out = hgt.copy()
    out[~np.isfinite(slope_mask)] = np.NaN
    return out


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
        if alt_range_th > 2000:
            raise GeometryError('Problem by altitude filter.')

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


@entity_task(log, writes=['geometries'])
def catchment_area(gdir):
    """Compute the catchment areas of each tributary line.

    The idea is to compute the route of lowest cost for any point on the
    glacier to rejoin a centerline. These routes are then put together if
    they belong to the same centerline, thus creating "catchment areas" for
    each centerline.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # Variables
    cls = gdir.read_pickle('centerlines')
    geom = gdir.read_pickle('geometries')
    glacier_pix = geom['polygon_pix']
    fpath = gdir.get_filepath('gridded_data')
    with utils.ncDataset(fpath) as nc:
        glacier_mask = nc.variables['glacier_mask'][:]
        glacier_ext = nc.variables['glacier_ext'][:]
        topo = nc.variables['topo_smoothed'][:]

    # If we have only one centerline this is going to be easy: take the
    # mask and return
    if len(cls) == 1:
        cl_catchments = [np.array(np.nonzero(glacier_mask == 1)).T]
        geom['catchment_indices'] = cl_catchments
        gdir.write_pickle(geom, 'geometries')
        return

    # Cost array
    costgrid = _make_costgrid(glacier_mask, glacier_ext, topo)

    # Initialise costgrid and the "catching" dict
    cost_factor = 0.  # Make it cheap
    dic_catch = dict()
    for i, cl in enumerate(cls):
        x, y = tuple2int(cl.line.xy)
        costgrid[y, x] = cost_factor
        for x, y in [(int(x), int(y)) for x, y in cl.line.coords]:
            assert (y, x) not in dic_catch
            dic_catch[(y, x)] = set([(y, x)])

    # It is much faster to make the array as small as possible. We trick:
    pm = np.nonzero(glacier_mask == 1)
    ymi, yma = np.min(pm[0])-1, np.max(pm[0])+2
    xmi, xma = np.min(pm[1])-1, np.max(pm[1])+2
    costgrid = costgrid[ymi:yma, xmi:xma]
    mask = glacier_mask[ymi:yma, xmi:xma]

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
        headcoords = np.array([not_computed[0][0], not_computed[1][0]],
                              dtype=np.int64)
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
    geom['catchment_indices'] = cl_catchments
    gdir.write_pickle(geom, 'geometries')


@entity_task(log, writes=['flowline_catchments', 'catchments_intersects'])
def catchment_intersections(gdir):
    """Computes the intersections between the catchments.

    A glacier usually consists of several flowlines and each flowline has a
    distinct catchment area. This function calculates the intersections between
    these areas.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    catchment_indices = gdir.read_pickle('geometries')['catchment_indices']

    # Loop over the lines
    mask = np.zeros((gdir.grid.ny, gdir.grid.nx))

    gdfc = gpd.GeoDataFrame()
    for i, ci in enumerate(catchment_indices):
        # Catchment polygon
        mask[:] = 0
        mask[tuple(ci.T)] = 1
        _, poly_no = _mask_to_polygon(mask, gdir=gdir)
        gdfc.loc[i, 'geometry'] = poly_no

    gdfi = utils.polygon_intersections(gdfc)

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
    gdir.write_shapefile(gdfc, 'flowline_catchments')
    if len(gdfi) > 0:
        gdir.write_shapefile(gdfi, 'catchments_intersects')


@entity_task(log, writes=['inversion_flowlines'])
def initialize_flowlines(gdir):
    """ Computes more physical Inversion Flowlines from geometrical Centerlines

    This interpolates the centerlines on a regular spacing (i.e. not the
    grid's (i, j) indices. Cuts out the tail of the tributaries to make more
    realistic junctions. Also checks for low and negative slopes and corrects
    them by interpolation.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # variables
    cls = gdir.read_pickle('centerlines')

    # Initialise the flowlines
    dx = cfg.PARAMS['flowline_dx']
    do_filter = cfg.PARAMS['filter_min_slope']
    lid = int(cfg.PARAMS['flowline_junction_pix'])
    fls = []

    # Topo for heights
    fpath = gdir.get_filepath('gridded_data')
    with utils.ncDataset(fpath) as nc:
        topo = nc.variables['topo_smoothed'][:]

    # Bilinear interpolation
    # Geometries coordinates are in "pixel centered" convention, i.e
    # (0, 0) is also located in the center of the pixel
    xy = (np.arange(0, gdir.grid.ny-0.1, 1),
          np.arange(0, gdir.grid.nx-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo)

    # Smooth window
    sw = cfg.PARAMS['flowline_height_smooth']
    diag_n_bad_slopes = 0
    diag_n_pix = 0
    for ic, cl in enumerate(cls):
        points = line_interpol(cl.line, dx)

        # For tributaries, remove the tail
        if ic < (len(cls)-1):
            points = points[0:-lid]

        new_line = shpg.LineString(points)

        # Interpolate heights
        xx, yy = new_line.xy
        hgts = interpolator((yy, xx))
        if len(hgts) < 5:
            raise GeometryError('This centerline is too short')

        # If smoothing, this is the moment
        hgts = gaussian_filter1d(hgts, sw)

        # Check for min slope issues and correct if needed
        if do_filter:
            # Correct only where glacier
            hgts = _filter_small_slopes(hgts, dx*gdir.grid.dx)
            isfin = np.isfinite(hgts)
            if not np.any(isfin):
                raise GeometryError('This centerline has no positive slopes')
            diag_n_bad_slopes += np.sum(~isfin)
            diag_n_pix += len(isfin)
            perc_bad = np.sum(~isfin) / len(isfin)
            if perc_bad > 0.8:
                log.warning('({}) more than {:.0%} of the flowline is cropped '
                            'due to negative slopes.'.format(gdir.rgi_id,
                                                             perc_bad))

            sp = np.min(np.where(np.isfinite(hgts))[0])
            while len(hgts[sp:]) < 5:
                sp -= 1
            hgts = utils.interp_nans(hgts[sp:])
            if not (np.all(np.isfinite(hgts)) and len(hgts) >= 5):
                raise GeometryError('Something went wrong in flowline init')
            new_line = shpg.LineString(points[sp:])

        sl = Centerline(new_line, dx=dx, surface_h=hgts,
                        orig_head=cl.orig_head, rgi_id=gdir.rgi_id)
        sl.order = cl.order
        fls.append(sl)

    # All objects are initialized, now we can link them.
    for cl, fl in zip(cls, fls):
        fl.orig_centerline_id = id(cl)
        if cl.flows_to is None:
            continue
        fl.set_flows_to(fls[cls.index(cl.flows_to)])

    # Write the data
    gdir.write_pickle(fls, 'inversion_flowlines')
    if do_filter:
        out = diag_n_bad_slopes/diag_n_pix
        gdir.add_to_diagnostics('perc_invalid_flowline', out)


@entity_task(log, writes=['inversion_flowlines'])
def catchment_width_geom(gdir):
    """Compute geometrical catchment widths for each point of the flowlines.

    Updates the 'inversion_flowlines' save file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # variables
    flowlines = gdir.read_pickle('inversion_flowlines')
    catchment_indices = gdir.read_pickle('geometries')['catchment_indices']

    # Topography is to filter the unrealistic lines afterwards.
    # I take the non-smoothed topography
    # I remove the boundary pixs because they are likely to be higher
    fpath = gdir.get_filepath('gridded_data')
    with utils.ncDataset(fpath) as nc:
        topo = nc.variables['topo'][:]
        mask_ext = nc.variables['glacier_ext'][:]
        mask_glacier = nc.variables['glacier_mask'][:]
    topo[np.where(mask_glacier == 0)] = np.NaN
    topo[np.where(mask_ext == 1)] = np.NaN

    # Intersects between catchments/glaciers
    gdfi = gpd.GeoDataFrame(columns=['geometry'])
    if gdir.has_file('catchments_intersects'):
        # read and transform to grid
        gdf = gdir.read_shapefile('catchments_intersects')
        salem.transform_geopandas(gdf, gdir.grid, inplace=True)
        gdfi = pd.concat([gdfi, gdf[['geometry']]])
    if gdir.has_file('intersects'):
        # read and transform to grid
        gdf = gdir.read_shapefile('intersects')
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
        poly, poly_no = _mask_to_polygon(mask, gdir=gdir)

        # First guess widths
        for i, (normal, pcoord) in enumerate(zip(fl.normals, fl.line.coords)):
            width, wline = _point_width(normal, pcoord, fl, poly, poly_no)
            widths[i] = width
            wlines.append(wline)

        valid = np.where(np.isfinite(widths))
        if len(valid[0]) == 0:
            errmsg = '({}) first guess widths went wrong.'.format(gdir.rgi_id)
            raise GeometryError(errmsg)

        # Ok now the entire centerline is computed.
        # I take all these widths for geometrically valid, and see if they
        # intersect with our buffered catchment/glacier intersections
        is_rectangular = []
        for wg in wlines:
            is_rectangular.append(np.any(gdfi.intersects(wg)))
        is_rectangular = _filter_grouplen(is_rectangular, minsize=5)

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
            log.warning('({}) width filtering too strong.'.format(gdir.rgi_id))
            fil_widths = widths[np.int(len(widths) / 2.)]

        # Special treatment for tidewater glaciers
        if gdir.is_tidewater and fl.flows_to is None:
            is_rectangular[-5:] = True

        # Write it in the objects attributes
        if len(fil_widths) != n:
            raise GeometryError('Something went wrong')
        fl.widths = fil_widths
        fl.geometrical_widths = wlines
        fl.is_rectangular = is_rectangular

    # Overwrite pickle
    gdir.write_pickle(flowlines, 'inversion_flowlines')


@entity_task(log, writes=['inversion_flowlines'])
def catchment_width_correction(gdir):
    """Corrects for NaNs and inconsistencies in the geometrical widths.

    Interpolates missing values, ensures consistency of the
    surface-area distribution AND with the geometrical area of the glacier
    polygon, avoiding errors due to gridded representation.

    Updates the 'inversion_flowlines' save file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # variables
    fls = gdir.read_pickle('inversion_flowlines')
    catchment_indices = gdir.read_pickle('geometries')['catchment_indices']

    # Topography for altitude-area distribution
    # I take the non-smoothed topography and remove the borders
    fpath = gdir.get_filepath('gridded_data')
    with utils.ncDataset(fpath) as nc:
        topo = nc.variables['topo'][:]
        ext = nc.variables['glacier_ext'][:]
    topo[np.where(ext == 1)] = np.NaN

    # Param
    nmin = int(cfg.PARAMS['min_n_per_bin'])
    smooth_ws = int(cfg.PARAMS['smooth_widths_window_size'])

    # Per flowline (important so that later, the indices can be moved)
    catchment_heights = []
    for ci in catchment_indices:
        _t = topo[tuple(ci.T)][:]
        catchment_heights.append(list(_t[np.isfinite(_t)]))

    # Loop over lines in a reverse order
    for fl, catch_h in zip(fls, catchment_heights):

        # Interpolate widths
        widths = utils.interp_nans(fl.widths)
        widths = np.clip(widths, 0.1, np.max(widths))

        # Get topo per catchment and per flowline point
        fhgt = fl.surface_h

        # Max and mins for the histogram
        maxh = np.max(fhgt)
        minh = np.min(fhgt)

        # Sometimes, the centerline does not reach as high as each pix on the
        # glacier. (e.g. RGI40-11.00006)
        catch_h = np.clip(catch_h, np.min(catch_h), maxh)
        # Same for min
        if fl.flows_to is None:
            # We clip only for main flowline (this has reasons)
            catch_h = np.clip(catch_h, minh, np.max(catch_h))

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
                new_widths = widths.copy()
                for bi in range(len(bins) - 1):
                    bintopoarea = len(np.where(topo_digi == bi)[0])
                    wherewiths = np.where(fl_digi == bi)
                    binflarea = np.sum(new_widths[wherewiths]) * fl.dx
                    new_widths[wherewiths] = (bintopoarea / binflarea *
                                              new_widths[wherewiths])
                break
            bsize += 5

            # Add a security for infinite loops
            if bsize > 500:
                nmin -= 1
                bsize = cfg.PARAMS['base_binsize']
                log.warning('(%s) reduced min n per bin to %d', gdir.rgi_id,
                            nmin)
                if nmin == 0:
                    raise GeometryError('({}) no binsize could be chosen '
                                        .format(gdir.rgi_id))
        if bsize > 150:
            log.warning('(%s) chosen binsize %d', gdir.rgi_id, bsize)
        else:
            log.debug('(%s) chosen binsize %d', gdir.rgi_id, bsize)

        # Now keep the good topo pixels and send the unattributed ones to the
        # next flowline
        tosend = list(catch_h[np.where(catch_h < minb)])
        if (len(tosend) > 0) and (fl.flows_to is not None):
            ide = fls.index(fl.flows_to)
            catchment_heights[ide] = np.append(catchment_heights[ide], tosend)
        if (len(tosend) > 0) and (fl.flows_to is None):
            raise RuntimeError('This should not happen')

        # Now we have a width which is the "best" representation of our
        # tributary according to the altitude area distribution.
        # This sometimes leads to abrupt changes in the widths from one
        # grid point to another. I think it's not too harmful to smooth them
        # here, at the cost of a less perfect altitude area distribution
        if smooth_ws != 0:
            if smooth_ws == 1:
                new_widths = utils.smooth1d(new_widths)
            else:
                new_widths = utils.smooth1d(new_widths, window_size=smooth_ws)

        # Write it
        fl.widths = new_widths

    # Final correction - because of the raster, the gridded area of the
    # glacier is not that of the actual geometry. correct for that
    area = 0.
    for fl in fls:
        area += np.sum(fl.widths) * fl.dx

    fac = gdir.rgi_area_m2 / (area * gdir.grid.dx**2)
    log.debug('(%s) corrected widths with a factor %.2f', gdir.rgi_id, fac)
    for fl in fls:
        fl.widths *= fac

    # Overwrite centerlines
    gdir.write_pickle(fls, 'inversion_flowlines')


@entity_task(log, writes=['inversion_flowlines'])
def terminus_width_correction(gdir, new_width=None):
    """Sets a new value for the terminus width.

    This can be useful for e.g. tiddewater glaciers where we know the width
    and don't like the OGGM one.

    This task preserves the glacier area but will change the fit of the
    altitude-area distribution slightly.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    new_width : float
       the new width of the terminus (in meters)
    """

    # variables
    fls = gdir.read_pickle('inversion_flowlines')
    fl = fls[-1]
    mapdx = gdir.grid.dx

    # Change the value and interpolate
    width = copy.deepcopy(fl.widths)
    width[-5:] = np.NaN
    width[-1] = new_width / mapdx
    width = utils.interp_nans(width)

    # Correct for RGI area
    area_to_match = gdir.rgi_area_m2 - np.sum(width[-5:] * mapdx**2 * fl.dx)
    area_before = np.sum(width[:-5] * mapdx**2 * fl.dx)
    for tfl in fls[:-1]:
        area_before += np.sum(tfl.widths * mapdx**2 * fl.dx)
    cor_factor = area_to_match / area_before
    for tfl in fls:
        tfl.widths = tfl.widths * cor_factor
    width[:-5] = width[:-5] * cor_factor
    fl.widths = width

    # Overwrite centerlines
    gdir.write_pickle(fls, 'inversion_flowlines')


@entity_task(log)
def intersect_downstream_lines(gdir, candidates=None):
    """Find tributaries to a main glacier by intersecting downstream lines

    The GlacierDirectories must at least contain a `downstream_line`.
    If you have a lot of candidates, only execute the necessary tasks for that
    and do the rest of the preprocessing after this function identified the
    true tributary glaciers.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
        The main glacier of interest
    candidates: list of oggm.GlacierDirectory
        Possible tributary glaciers to the main glacier

    Returns
    -------
    tributaries: dict
        Key is the main glacier rgi_id, values is a list of tributary rgi_ids
    """

    # make sure tributaries are iteratable
    candidates = utils.tolist(candidates)

    # Buffer in pixels around the flowline
    buffer = cfg.PARAMS['kbuffer']

    # get main glacier downstream line and CRS
    dline = gdir.read_pickle('downstream_line')['full_line']
    crs = gdir.grid

    # return list
    tributaries = []

    # loop over tributaries
    for trib in candidates:
        # skip self
        if gdir.rgi_id == trib.rgi_id:
            continue

        # get tributary glacier downstream line and CRS
        _dline = trib.read_pickle('downstream_line')['full_line']
        _crs = trib.grid

        # use salem to transform the grids
        _trans_dline = salem.transform_geometry(_dline, crs=_crs, to_crs=crs)

        # check for intersection, with a small buffer and add to list
        if dline.intersects(_trans_dline.buffer(buffer)):
            tributaries.append(trib)

    return {gdir.rgi_id: tributaries}
