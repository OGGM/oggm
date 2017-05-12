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
from __future__ import absolute_import, division
from six.moves import zip

# Built ins
import logging
import copy
from itertools import groupby
# External libs
import numpy as np
from pandas import Series as pdSeries
import shapely.ops
import scipy.signal
from scipy.ndimage.morphology import distance_transform_edt
from skimage.graph import route_through_array
import netCDF4
import shapely.geometry as shpg
import scipy.signal
from scipy.interpolate import RegularGridInterpolator
# Locals
import oggm.cfg as cfg
from oggm.cfg import GAUSSIAN_KERNEL
from salem import lazy_property
from oggm.utils import tuple2int
from oggm import entity_task, divide_task

# Module logger
log = logging.getLogger(__name__)


class Centerline(object):
    """A Centerline has geometrical and flow rooting properties.

    It is instanciated and updated by _join_lines() exclusively
    """

    def __init__(self, line, dx=None, surface_h=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """

        self.line = None  # Shapely LineString
        self.head = None  # Shapely Point
        self.tail = None  # Shapely Point
        self.dis_on_line = None  # Shapely Point
        self.nx = None  # Shapely Point
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
        self.touches_border = None

        # Set by external funcs
        self.geometrical_widths = None  # these are kept for plotting and such
        self.apparent_mb = None  # Apparent MB, NOT weighted by width.
        self.flux = None  # Flux (kg m-2)

    def set_flows_to(self, other, check_tail=True):
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
            ind_closest = np.argmin(np.abs(other.dis_on_line - prdis))
            ind_closest = np.asscalar(ind_closest)
            n = len(other.dis_on_line)
            if n >= 9:
                ind_closest = np.clip(ind_closest, 4, n-5)
            elif n >= 7:
                ind_closest = np.clip(ind_closest, 3, n-4)
            elif n >= 5:
                ind_closest = np.clip(ind_closest, 2, n-3)
            p = shpg.Point(other.line.coords[int(ind_closest)])
            self.flows_to_point = p
            other.inflow_points.append(p)
            other.inflows.append(self)
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

    def set_apparent_mb(self, mb):
        """Set the apparent mb and flux for the flowline.

        MB is expected in kg m-2 yr-1 (= mm w.e. yr-1)

        This should happen in line order, otherwise it will be wrong.
        """

        self.apparent_mb = mb

        # Add MB to current flux and sum
        # no more changes should happen after that
        self.flux += mb * self.widths * self.dx
        self.flux = np.cumsum(self.flux)

        # Add to outflow. That's why it should happen in order
        if self.flows_to is not None:
            n = len(self.flows_to.line.coords)
            ide = self.flows_to_indice
            if n >= 9:
                gk = GAUSSIAN_KERNEL[9]
                self.flows_to.flux[ide-4:ide+5] += gk * self.flux[-1]
            elif n >= 7:
                gk = GAUSSIAN_KERNEL[7]
                self.flows_to.flux[ide-3:ide+4] += gk * self.flux[-1]
            elif n >= 5:
                gk = GAUSSIAN_KERNEL[5]
                self.flows_to.flux[ide-2:ide+3] += gk * self.flux[-1]


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
            extext ='Geometry collection not expected: {}'.format(
                inter_poly.type)
            raise NotImplementedError(extext)

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
        The lines shorter than r will be filtered out..

    Returns
    -------
    (lines, heads) a list of the new lines and corresponding heads
    """

    olines = []
    oheads = []
    ilines = copy.copy(lines)

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
                            raise RuntimeError('Head not found')
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
        l = ilines[np.argmax(lengths)]

        ilines.remove(l)
        if len(olines) > 0:
            # the cutted line's last point is not guaranteed
            # to on straight coordinates. Remove it
            olines.append(shpg.LineString(np.asarray(l.xy)[:, 0:-1].T))
        else:
            olines.append(l)
        lastline = l

    # add the corresponding head to each line
    for l in olines:
        for h in heads:
            if l.intersects(h):
                oheads.append(h)
                break

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
    ind_closest = np.argmin(np.abs(centerline.dis_on_line - prdis))
    ind_closest = np.asscalar(ind_closest)
    flow_point = shpg.Point(centerline.line.coords[int(ind_closest)])
    return flow_point


def _join_lines(lines):
    """Re-joins the lines that have been cut by _filter_lines

     Compute the rooting scheme.

    Parameters
    ----------
    lines: list of Centerline instances

    Returns
    -------
    the same Centerline instances, updated with flow routing properties
     """

    olines = [Centerline(l) for l in lines[::-1]]
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


def _line_order(line):
    """Recursive search for the line's hydrological level.

    Parameters
    ----------
    line: a Centerline instance

    Returns
    -------
    The line;s order
    """

    if len(line.inflows) == 0:
        return 0
    else:
        levels = [_line_order(s) for s in line.inflows]
        return np.max(levels) + 1


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
        ind_term = ind[np.round(len(ind) / 2.).astype(np.int)]

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


@entity_task(log, writes=['centerlines', 'gridded_data'])
@divide_task(log, add_0=False)
def compute_centerlines(gdir, div_id=None):
    """Compute the centerlines following Kienholz et al., (2014).

    They are then sorted according to the modified Strahler number:
    http://en.wikipedia.org/wiki/Strahler_number

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    # Params
    single_fl = not cfg.PARAMS['use_multiple_flowlines']

    if 'force_one_flowline' in cfg.PARAMS:
        if gdir.rgi_id in cfg.PARAMS['force_one_flowline']:
            single_fl = True

    # open
    geom = gdir.read_pickle('geometries', div_id=div_id)
    grids_file = gdir.get_filepath('gridded_data', div_id=div_id)
    with netCDF4.Dataset(grids_file) as nc:
        # Variables
        glacier_mask = nc.variables['glacier_mask'][:]
        glacier_ext = nc.variables['glacier_ext'][:]
        topo = nc.variables['topo_smoothed'][:]
        poly_pix = geom['polygon_pix']

    # Find for local maximas on the outline
    x, y = tuple2int(poly_pix.exterior.xy)
    ext_yx = tuple(reversed(poly_pix.exterior.xy))
    zoutline = topo[y[:-1], x[:-1]]  # last point is first point

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
    heads_idx = heads_idx[0][np.where(zoutline[heads_idx] > head_threshold)]
    heads = np.asarray(ext_yx)[:, heads_idx]
    heads_z = zoutline[heads_idx]
    # careful, the coords are in y, x order!
    heads = [shpg.Point(x, y) for y, x in zip(heads[0, :],
                                              heads[1, :])]

    # get radius of the buffer according to Kienholz eq. (1)
    radius = cfg.PARAMS['q1'] * geom['polygon_area'] + cfg.PARAMS['q2']
    radius = np.clip(radius, 0, cfg.PARAMS['rmax'])
    radius /= gdir.grid.dx # in raster coordinates
    # Plus our criteria, quite usefull to remove short lines:
    radius += cfg.PARAMS['flowline_junction_pix'] * cfg.PARAMS['flowline_dx']
    log.debug('%s: radius in raster coordinates: %.2f',
              gdir.rgi_id, radius)

    # OK. Filter and see.
    log.debug('%s: number of heads before radius filter: %d',
              gdir.rgi_id, len(heads))
    heads, heads_z = _filter_heads(heads, heads_z, radius, poly_pix)
    log.debug('%s: number of heads after radius filter: %d',
              gdir.rgi_id, len(heads))

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
    log.debug('%s: computed the routes', gdir.rgi_id)

    # Filter the shortest lines out
    radius = cfg.PARAMS['flowline_junction_pix'] * cfg.PARAMS['flowline_dx']
    radius += 6 * cfg.PARAMS['flowline_dx']
    olines, _ = _filter_lines(lines, heads, cfg.PARAMS['kbuffer'], radius)
    log.debug('%s: number of heads after lines filter: %d',
              gdir.rgi_id, len(olines))

    # And rejoin the cutted tails
    olines = _join_lines(olines)

    # Adds the line level
    for cl in olines:
        cl.order = _line_order(cl)

    # And sort them per order !!! several downstream tasks  rely on this
    cls = []
    for i in np.argsort([cl.order for cl in olines]):
        cls.append(olines[i])

    # Final check
    if len(cls) == 0:
        raise RuntimeError('{} : no centerline found!'.format(gdir.rgi_id))

    # Write the data
    gdir.write_pickle(cls, 'centerlines', div_id=div_id)

    # Netcdf
    with netCDF4.Dataset(grids_file, 'a') as nc:
        if 'cost_grid' in nc.variables:
            # Overwrite
            nc.variables['cost_grid'][:] = costgrid
        else:
            # Create
            v = nc.createVariable('cost_grid', 'f4', ('y', 'x', ), zlib=True)
            v.units = '-'
            v.long_name = 'Centerlines cost grid'
            v[:] = costgrid


@entity_task(log, writes=['downstream_line', 'major_divide'])
def compute_downstream_lines(gdir):
    """Compute the lines continuing the glacier (one per divide).

    The idea is simple: starting from the glacier tail, compute all the routes
    to all local minimas found at the domain edge. The cheapest is "the One".

    The task also determines the so-called "major flowline" which is the
    only line flowing out of the domain. The other ones are flowing into the
    branch. The rest of the job (merging all centerlines + downstreams into
    one single glacier is realized by
    :py:func:`~oggm.tasks.init_present_time_glacier`).

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    with netCDF4.Dataset(gdir.get_filepath('gridded_data', div_id=0)) as nc:
        topo = nc.variables['topo_smoothed'][:]

    # Make going up very costy
    topo = topo**10

    # Variables we gonna need
    xmesh, ymesh = np.meshgrid(np.arange(0, gdir.grid.nx, 1, dtype=np.int64),
                               np.arange(0, gdir.grid.ny, 1, dtype=np.int64))
    _h = [topo[:, 0], topo[0, :], topo[:, -1], topo[-1, :]]
    _x = [xmesh[:, 0], xmesh[0, :], xmesh[:, -1], xmesh[-1, :]]
    _y = [ymesh[:, 0], ymesh[0, :], ymesh[:, -1], ymesh[-1, :]]

    # First we have to find the "major" divide
    heads = []
    heigts = []
    div_ids = list(gdir.divide_ids)
    for div_id in div_ids:
        head = gdir.read_pickle('centerlines', div_id=div_id)[-1].tail
        heigts.append(topo[int(head.y), int(head.x)])
        heads.append((int(head.y), int(head.x)))

    # Now the lowest first
    aso = np.argsort(heigts)
    major_id = aso[0]
    a = aso[0]
    head = heads[a]
    head_h = heigts[a]
    # Find all local minimas and their coordinates
    min_cost = np.Inf
    line = None
    min_thresold = 50
    while True:
        for h, x, y in zip(_h, _x, _y):
            ids = scipy.signal.argrelmin(h, order=10, mode='wrap')
            for i in ids[0]:
                # Prevent very high local minimas
                # if topo[y[i], x[i]] > ((head_h-min_thresold)**10):
                #     continue
                lids, cost = route_through_array(topo, head, (y[i], x[i]),
                                                 geometric=True)
                if cost < min_cost:
                    min_cost = cost
                    line = shpg.LineString(np.array(lids)[:, [1, 0]])
        if line is None:
            min_thresold -= 10
        else:
            break
        if min_thresold < 0:
            raise RuntimeError('Downstream line not found')

    if gdir.is_tidewater:
        # for tidewater glaciers the line can be very long and chaotic.
        # we clip it for now
        cl = gdir.read_pickle('centerlines', div_id=div_ids[major_id])[-1]
        len_gl = cl.dis_on_line[-1]
        dis = [line.project(shpg.Point(co)) for co in line.coords]
        new_coords = np.array(line.coords)[np.nonzero(dis <= 0.5 * len_gl)]
        line = shpg.LineString(new_coords)

    # Write the data
    mdiv = div_ids[major_id]
    gdir.write_pickle(mdiv, 'major_divide', div_id=0)
    gdir.write_pickle(line, 'downstream_line', div_id=div_ids[a])
    if len(div_ids) == 1:
        return

    # If we have divides, there is just one route and we have to interrupt
    # It at the glacier boundary
    fpath = gdir.get_filepath('gridded_data', div_id=div_ids[a])
    with netCDF4.Dataset(fpath) as nc:
        mask = nc.variables['glacier_mask'][:]
        ext = nc.variables['glacier_ext'][:]
    mask[np.where(ext==1)] = 0
    tail = head
    topo[np.where(mask==1)] = 0
    for a in aso[1:]:
        head = heads[a]
        lids, _ = route_through_array(topo, head, tail)
        lids = [l for l in lids if mask[l[0], l[1]] == 0][0:-1]
        line = shpg.LineString(np.array(lids)[:, [1, 0]])
        gdir.write_pickle(line, 'downstream_line', div_id=div_ids[a])


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
    return sum(abs((np.polyval(f, x) - y) / y)) / len(x)


@entity_task(log, writes=['downstream_bed'])
def compute_downstream_bedshape(gdir):
    """The bedshape obtained by fitting a parabola to the line's normals.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """
    # get the major downstream line only
    majid = gdir.read_pickle('major_divide', div_id=0)
    dl = gdir.read_pickle('downstream_line', div_id=majid)
    
    # Volume area scaling formula for the ice thickness
    h_mean = 0.034 * gdir.rgi_area_km2**0.375 * 1000
    log.debug('%s: estimated glacier thickness: %d', gdir.rgi_id, h_mean)

    bed = []
    # Far Factor
    r = 40
    # number of points
    cs_n = 20
    
    # make distance between point the same
    # TODO: use a Centerline class instead
    from .geometry import _line_interpol, InversionFlowline
    l = shpg.LineString(_line_interpol(dl,cfg.PARAMS['flowline_dx']))
    idl = InversionFlowline(l, cfg.PARAMS['flowline_dx'], None)
       
    # Topography
    with netCDF4.Dataset(gdir.get_filepath('gridded_data', div_id=0)) as nc:
        topo = nc.variables['topo_smoothed'][:]
        x = nc.variables['x'][:]
        y = nc.variables['y'][:]
    xy = (np.arange(0, len(y)-0.1, 1), np.arange(0, len(x)-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo)

    # TODO: temporary class
    class MyPoint(shpg.Point):
        def __hash__(self):
            return hash(tuple((self.x, self.y)))

    # normals
    ns = [i[0] for i in idl.normals]
    cs = []
    try:
        for pcoords, n in zip(idl.line.coords, ns):
            xi, yi = pcoords
            vx, vy = n
            modul = np.sqrt(vx**2 + vy**2)
            ci = []
            for ro in np.linspace(0, r / 2.0, cs_n):
                t = ro / modul
                cp1 = MyPoint(xi + t * vx, yi + t * vy)
                cp2 = MyPoint(xi - t * vx, yi - t * vy)

                # check if out of the frame
                if not (0 < cp2.y < len(y) - 1) or \
                   not (0 < cp2.x < len(x) - 1) or \
                   not (0 < cp1.y < len(y) - 1) or \
                   not (0 < cp1.x < len(x) - 1):
                    raise StopIteration()

                ci.append((cp1, ro))
                ci.append((cp2, -ro))

            ci = list(set(ci))
            cs.append(ci)
    except StopIteration:
        # we reached the end of the line
        pass

    log.debug('%s: length of downstream line is: %d', gdir.rgi_id, len(cs))

    good = 0
    for ic, cc in enumerate(cs):
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
        if err < 1.5:
            bed.append(p2)
            good += 1
        else:
            bed.append([None, 0, p2[2]])

    log.debug('%s: percentage of valid parabolas: %d',
              gdir.rgi_id, int(good/len(cs)*100))

    # skip gaps
    x_bed = [i for i, j in enumerate(bed) if j[0]]
    bg = [bed[i][0] for i in x_bed]

    # interpolation, filling the gaps
    bed_i = np.interp(range(0, len(bed)), x_bed, bg)
    bed_int = [[bed_i[j], i[1], i[2]] for j, i in enumerate(bed)]

    # approximation - BED_Moving_Average
    bed_ma = pdSeries([i[0] for i in bed_int])
    bed_ma = bed_ma.rolling(window=5, center=True, min_periods=1).mean()
    bed_ma = [[bed_ma[j], i[1], i[2]] for j, i in enumerate(bed_int)]

    # write output
    gdir.write_pickle(bed_ma, 'downstream_bed')
