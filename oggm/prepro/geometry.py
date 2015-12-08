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
import matplotlib._cntr as cntr
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.measurements import label
from salem import lazy_property
# Locals
import oggm.conf as cfg
from oggm import utils
import oggm.prepro.centerlines
from oggm.utils import tuple2int

# Module logger
log = logging.getLogger(__name__)

# Variable needed later
label_struct = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]])

class InversionFlowline(oggm.prepro.centerlines.Centerline):
    """An advanced centerline, with widths and apparent MB."""

    def __init__(self, line, dx, heights):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """

        super(InversionFlowline, self).__init__(line)

        self._surface_h = heights
        self.dx = dx
        self._widths = None
        self.geometrical_widths = None  # these are kept for plotting and such

        self.apparent_mb = None  # Apparent MB, NOT weighted by width.
        self.flux = None  # Flux (kg m-2)

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
                gk = utils.gaussian_kernel[9]
                self.flows_to.flux[ide-4:ide+5] += gk * self.flux[-1]
            elif n >= 7:
                gk = utils.gaussian_kernel[7]
                self.flows_to.flux[ide-3:ide+4] += gk * self.flux[-1]
            elif n >= 5:
                gk = utils.gaussian_kernel[5]
                self.flows_to.flux[ide-2:ide+3] += gk * self.flux[-1]

    def set_flows_to(self, other):
        """Find the closest point in "other" and sets all the corresponding
        attributes. Btw, it modifies the state of "other" too.

        Had to override this because of junction's safety reasons: we didnt
        want to be too close to the tail

        Parameters
        ----------
        other: an other centerline
        """

        self.flows_to = other

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
        self.flows_to_point = shpg.Point(other.line.coords[int(ind_closest)])
        other.inflow_points.append(self.flows_to_point)
        other.inflows.append(self)

    @lazy_property
    def normals(self):
        """List of (n1, n2) normal vectors at each point.

        We use second order derivatives for smoother widths.
        """

        def _normalize(n):
            nn = n / np.sqrt(np.sum(n*n))
            n1 = np.array([-nn[1], nn[0]])
            n2 = np.array([nn[1], -nn[0]])
            return n1, n2

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


def _mask_to_polygon(mask, x=None, y=None):
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

    regions, nregions = label(mask, structure=label_struct)
    if nregions > 1:
        log.debug('%s: we had to cut a blob from the catchment')
        # Check the size of those
        region_sizes = [np.sum(regions == r) for r in np.arange(1, nregions+1)]
        am = np.argmax(region_sizes)
        # Check not a strange glacier
        sr = region_sizes.pop(am)
        for ss in region_sizes:
            assert (ss / sr) < 0.1
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
    aft, cur, bef: (x, y) coordinates of the current point, before, and after
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
    alt_range_th = cfg.params['width_alt_range_thres']

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


def catchment_area(gdir, div_id=None):
    """Compute the catchment areas of each tributary line.

    The idea is to compute the route of lowest cost for any point on the
    glacier to rejoin a centerline. These routes are then put together if
    they belong to the same centerline, thus creating "catchment areas" for
    each centerline.

    Parameters
    ----------
    gdir: GlacierDir object
    div_id: the divide ID to process (should be left to None)

    I/O
    ---
    catchment_indices.p: a list of
    """

    if div_id is None:
        for i in gdir.divide_ids:
            log.info('%s: catchment areas, divide %d', gdir.rgi_id, i)
            catchment_area(gdir, div_id=i)
        return

    # Variables
    cls = gdir.read_pickle('centerlines', div_id=div_id)
    geoms = gdir.read_pickle('geometries', div_id=div_id)
    glacier_pix = geoms['polygon_pix']
    nc = netCDF4.Dataset(gdir.get_filepath('grids', div_id=div_id))
    costgrid = nc.variables['cost_grid'][:]
    mask = nc.variables['glacier_mask'][:]
    nc.close()

    # If we have only one centerline this is going to be easy: take the
    # mask and return
    if len(cls) == 1:
        cl_catchments = [np.array(np.where(mask == 1)).T]
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

    # Where did we compute the path already?
    computed = np.where(mask == 1, 0, np.nan)

    # Coords of Terminus
    endcoords = np.array(cls[0].tail.coords[0])[::-1].astype(np.int64)

    # Start with all the paths at the boundaries, they are more likely
    # to cover much of the glacier
    for headx, heady in tuple2int(glacier_pix.exterior.coords):
        indices, _ = route_through_array(costgrid, np.array([heady, headx]),
                                         endcoords)
        inds = np.array(indices).T
        computed[inds[0], inds[1]] = 1
        set_dump = set([])
        for y, x in indices:
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


def initialize_flowlines(gdir, div_id=None):
    """Interpolate the centerlines on a regular spacing, cut out the tail of
       the tributaries.

    Parameters
    ----------
    gdir: GlacierDir object
    div_id: the divide ID to process (should be left to None)

    I/O
    ---
    Creates a flowlines.p with the new objects.
    """

    if div_id is None:
        for i in gdir.divide_ids:
            log.info('%s: initialize flowlines, divide %d', gdir.rgi_id, i)
            initialize_flowlines(gdir, div_id=i)
        return

    # variables
    cls = gdir.read_pickle('centerlines', div_id=div_id)

    # Initialise the flowlines
    dx = cfg.params['flowline_dx']
    lid = int(cfg.params['flowline_junction_pix'])
    fls = []

    # Topo for heights
    nc = netCDF4.Dataset(gdir.get_filepath('grids', div_id=div_id))
    topo = nc.variables['topo_smoothed'][:]
    nc.close()
    # Bilinear interpolation
    # Geometries coordinates are in "pixel centered" convention, i.e
    # (0, 0) is also located in the center of the pixel
    xy = (np.arange(0, gdir.grid.ny-0.1, 1),
          np.arange(0, gdir.grid.nx-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo)

    # Smooth window
    sw = cfg.params['flowline_height_smooth']

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

        l = InversionFlowline(new_line, dx, hgts)
        l.order = cl.order
        fls.append(l)

    # All objects are initialized, now we can link them.
    for cl, fl in zip(cls, fls):
        if cl.flows_to is None:
            continue
        fl.set_flows_to(fls[cls.index(cl.flows_to)])

    # Write the data
    gdir.write_pickle(fls, 'inversion_flowlines', div_id=div_id)


def catchment_width_geom(gdir, div_id=None):
    """Compute geometrical catchment widths for each point of the flowlines

    Parameters
    ----------
    gdir: GlacierDir object
    div_id: the divide ID to process (should be left to None)

    I/O
    ---
    Updates::
        - flowlines.p with the new properties
    """

    if div_id is None:
        for i in gdir.divide_ids:
            log.info('%s: catchment widths, divide %d', gdir.rgi_id, i)
            catchment_width_geom(gdir, div_id=i)
        return

    # variables
    flowlines = gdir.read_pickle('inversion_flowlines', div_id=div_id)
    catchment_indices = gdir.read_pickle('catchment_indices', div_id=div_id)
    xmesh, ymesh = np.meshgrid(np.arange(0, gdir.grid.nx, 1),
                               np.arange(0, gdir.grid.ny, 1))

    # Topography is to filter the lines afterwards.
    # I take the non-smoothed topography
    # I remove the boundary pixs because they are likely to be higher
    nc = netCDF4.Dataset(gdir.get_filepath('grids', div_id=div_id))
    topo = nc.variables['topo'][:]
    mask_ext = nc.variables['glacier_ext'][:]
    mask_glacier = nc.variables['glacier_mask'][:]
    topo[np.where(mask_glacier == 0)] = np.NaN
    topo[np.where(mask_ext == 1)] = np.NaN
    nc.close()

    # Filter parameters
    # Number of pixels to arbitrarily remove at junctions
    jpix = int(cfg.params['flowline_junction_pix'])

    # Loop over the lines
    mask = np.zeros((gdir.grid.ny, gdir.grid.nx))
    for fl, ci in zip(flowlines, catchment_indices):

        if flowlines.index(fl) == 3:
            tt = 1

        n = len(fl.dis_on_line)

        widths = np.zeros(n)
        wlines = []

        # Catchment polygon
        mask[:] = 0
        mask[tuple(ci.T)] = 1
        poly, poly_no = _mask_to_polygon(mask, x=xmesh, y=ymesh)

        # First guess widths
        for i, (normal, pcoord) in enumerate(zip(fl.normals, fl.line.coords)):
            width, wline = _point_width(normal, pcoord, fl, poly, poly_no)
            widths[i] = width
            wlines.append(wline)

        valid = np.where(np.isfinite(widths))
        if len(valid[0]) == 0:
            raise RuntimeError('{}: First guess widths went wrong.'.format(gdir.rgi_id))


        # Ok now the entire centerline is computed.
        # we filter the lines which have a large altitude range
        widths = _filter_for_altitude_range(widths, wlines, topo)

        # Filter +- widths at junction points
        for fid in fl.inflow_indices:
            i0 = np.clip(fid-jpix, jpix/2, n-jpix/2).astype(np.int64)
            i1 = np.clip(fid+jpix+1, jpix/2, n-jpix/2).astype(np.int64)
            widths[i0:i1] = np.NaN

        valid = np.where(np.isfinite(widths))
        if len(valid[0]) == 0:
            raise RuntimeError('{}: I filtered too much.'.format(gdir.rgi_id))

        # Write it in the objects attributes
        assert len(widths) == n
        fl.widths = widths
        fl.geometrical_widths = wlines

    # Overwrite pickle
    gdir.write_pickle(flowlines, 'inversion_flowlines', div_id=div_id)


def catchment_width_correction(gdir, div_id=None):
    """Corrects for NaNs and inconsistencies in the geometrical widths.

    Interpolates missing values, ensures consistency of the
    surface-area distribution AND with the geometrical area of the glacier
    polygon, avoiding errors due to gridded representation.

    Parameters
    ----------
    gdir: GlacierDir object
    div_id: the divide ID to process (should be left to None)

    I/O
    ---
    Updates::
        - flowlines.p with the new properties
    """

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
        fac = gdir.glacier_area / (area * gdir.grid.dx**2 * 10**-6)
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
    nc = netCDF4.Dataset(gdir.get_filepath('grids', div_id=div_id))
    topo = nc.variables['topo'][:]
    ext = nc.variables['glacier_ext'][:]
    nc.close()
    topo[np.where(ext==1)] = np.NaN

    # Param
    nmin = int(cfg.params['min_n_per_bin'])

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

        # Get topo per catchment and per flowline point
        fhgt = fl.surface_h

        # Sometimes, the centerline does reach as high as each pix on the
        # glacier. (e.g. RGI40-11.00006)
        catch_h = np.clip(catch_h, 0, np.max(fhgt))

        # Max and mins for the histogram
        maxh = np.max(fhgt)
        if fl.flows_to is None:
            minh = np.min(fhgt)
            catch_h = np.clip(catch_h, minh, np.max(catch_h))
        else:
            minh = np.min(fhgt)  # Min just for flowline (this has reasons)

        # Now decide on a binsize which ensures at least one element per bin
        bsize = 50.
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
                break
            bsize += 5
            # Add a secutity for infinite loops
            if bsize > 250:
                nmin -= 1
                bsize = 50
                log.warning('%s: reduced min n per bin to %d', gdir.rgi_id,
                            nmin)
                if nmin == 0:
                    raise RuntimeError('NO binsize could be chosen for: '
                                       '{}'.format(gdir.rgi_id))
        if bsize > 100:
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

        # For each bin, the width(s) have to represent the "real" area
        for bi in range(len(bins)-1):
            bintopoarea = len(np.where(topo_digi == bi)[0])
            wherewiths = np.where(fl_digi == bi)
            binflarea = np.sum(widths[wherewiths]) * fl.dx
            widths[wherewiths] = (bintopoarea / binflarea) * widths[wherewiths]

        # Write it
        fl.widths = widths

    return flowlines
