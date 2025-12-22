"""Some useful functions that did not fit into the other modules.
"""

# Builtins
import os
import sys
import logging
import warnings
from packaging.version import Version
import calendar
# External libs
import pandas as pd
import numpy as np
try:
    from numpy._core import umath
except ImportError:
    from numpy.core import umath
import xarray as xr
from scipy.ndimage import convolve1d
try:
    from scipy.signal.windows import gaussian
except AttributeError:
    # Old scipy
    from scipy.signal import gaussian
from scipy.interpolate import interp1d
import cftime
import shapely.geometry as shpg
from shapely.ops import linemerge
from shapely.validation import make_valid
from salem.gis import Grid

# Optional libs
try:
    import geopandas as gpd
except ImportError:
    pass

# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_DAY
from oggm.utils._downloads import get_demo_file
from oggm.exceptions import InvalidParamsError, InvalidGeometryError

# Module logger
log = logging.getLogger('.'.join(__name__.split('.')[:-1]))

_RGI_METADATA = dict()

# Shape factors
# TODO: how to handle zeta > 10? at the moment extrapolation
# Table 1 from Adhikari (2012) and corresponding interpolation functions
_ADHIKARI_TABLE_ZETAS = np.array([0.5, 1, 2, 3, 4, 5, 10])
_ADHIKARI_TABLE_RECTANGULAR = np.array([0.313, 0.558, 0.790, 0.884,
                                        0.929, 0.954, 0.990])
_ADHIKARI_TABLE_PARABOLIC = np.array([0.251, 0.448, 0.653, 0.748,
                                      0.803, 0.839, 0.917])
ADHIKARI_FACTORS_RECTANGULAR = interp1d(_ADHIKARI_TABLE_ZETAS,
                                        _ADHIKARI_TABLE_RECTANGULAR,
                                        fill_value='extrapolate')
ADHIKARI_FACTORS_PARABOLIC = interp1d(_ADHIKARI_TABLE_ZETAS,
                                      _ADHIKARI_TABLE_PARABOLIC,
                                      fill_value='extrapolate')

# those constants help for vectorized conversion between dates and float years
_BASE_CUM_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304,
                            334], dtype=np.int64)
_BASE_CUM_END = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334,
                          365], dtype=np.int64)


def parse_rgi_meta(version=None):
    """Read the meta information (region and sub-region names)"""

    global _RGI_METADATA

    if version is None:
        version = cfg.PARAMS['rgi_version']

    if version in _RGI_METADATA:
        return _RGI_METADATA[version]

    # Parse RGI metadata
    if version in ['7', '70G', '70C']:
        rgi7url = 'https://cluster.klima.uni-bremen.de/~oggm/rgi/RGI2000-v7.0-regions.zip'
        reg_names = gpd.read_file(rgi7url, layer='RGI2000-v7.0-o1regions')
        reg_names.index = reg_names['o1region'].astype(int)
        reg_names = reg_names['full_name']
        subreg_names = gpd.read_file(rgi7url, layer='RGI2000-v7.0-o2regions')
        subreg_names.index = subreg_names['o2region']
        subreg_names = subreg_names['full_name']

    elif version in ['4', '5']:
        reg_names = pd.read_csv(get_demo_file('rgi_regions.csv'), index_col=0)
        # The files where different back then
        subreg_names = pd.read_csv(get_demo_file('rgi_subregions_V5.csv'),
                                   index_col=0)
    else:
        reg_names = pd.read_csv(get_demo_file('rgi_regions.csv'), index_col=0)
        f = os.path.join(get_demo_file('rgi_subregions_'
                                       'V{}.csv'.format(version)))
        subreg_names = pd.read_csv(f)
        subreg_names.index = ['{:02d}-{:02d}'.format(s1, s2) for s1, s2 in
                              zip(subreg_names['O1'], subreg_names['O2'])]
        subreg_names = subreg_names[['Full_name']]

    # For idealized
    reg_names.loc[0] = ['None']
    subreg_names.loc['00-00'] = ['None']
    _RGI_METADATA[version] = (reg_names, subreg_names)
    return _RGI_METADATA[version]


def query_yes_no(question, default="yes"):  # pragma: no cover
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Credits: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def tolist(arg, length=None):
    """Makes sure that arg is a list."""

    if isinstance(arg, str):
        arg = [arg]

    try:
        # Shapely stuff
        arg = arg.geoms
    except AttributeError:
        pass

    try:
        (e for e in arg)
    except TypeError:
        arg = [arg]

    arg = list(arg)

    if length is not None:

        if len(arg) == 1:
            arg *= length
        elif len(arg) == length:
            pass
        else:
            raise ValueError('Cannot broadcast len {} '.format(len(arg)) +
                             'to desired length: {}.'.format(length))

    return arg


def set_array_type(array):
    """Convert array to scalar if it contains a single value.

    Converting arrays with ndim > 0 to scalar is deprecated in numpy
    1.25+. Some OGGM functions expect arrays with a single value to be returned
    as scalars.

    Parameters
    ----------
    array : ArrayLike
        A numpy array or list.

    Returns
    -------
    float or np.ndarray
        Scalar if the array contains a single value, otherwise a numpy
        array.
    """
    if len(array) > 1:
        output = np.asanyarray(array)
    else:
        output = array[0]

    return output


def haversine(lon1, lat1, lon2, lat2):
    """Great circle distance between two (or more) points on Earth

    Parameters
    ----------
    lon1 : float
       scalar or array of point(s) longitude
    lat1 : float
       scalar or array of point(s) longitude
    lon2 : float
       scalar or array of point(s) longitude
    lat2 : float
       scalar or array of point(s) longitude

    Returns
    -------
    the distances

    Examples:
    ---------
    >>> haversine(34, 42, 35, 42)
    82633.46475287154
    >>> haversine(34, 42, [35, 36], [42, 42])
    array([ 82633.46475287, 165264.11172113])
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371000  # Radius of earth in meters


def interp_nans(array, default=None):
    """Interpolate nans using np.interp.

    np.interp is reasonable in that it does not extrapolate, it replaces
    nans at the bounds with the closest valid value.
    """

    _tmp = array.copy()
    nans, x = np.isnan(array), lambda z: z.nonzero()[0]
    if np.all(nans):
        # No valid values
        if default is None:
            raise ValueError('No points available to interpolate: '
                             'please set default.')
        _tmp[:] = default
    else:
        _tmp[nans] = np.interp(x(nans), x(~nans), array[~nans])

    return _tmp


def smooth1d(array, window_size=None, kernel='gaussian'):
    """Apply a centered window smoothing to a 1D array.

    Parameters
    ----------
    array : ndarray
        the array to apply the smoothing to
    window_size : int
        the size of the smoothing window
    kernel : str
        the type of smoothing (`gaussian`, `mean`)

    Returns
    -------
    the smoothed array (same dim as input)
    """

    # some defaults
    if window_size is None:
        if len(array) >= 9:
            window_size = 9
        elif len(array) >= 7:
            window_size = 7
        elif len(array) >= 5:
            window_size = 5
        elif len(array) >= 3:
            window_size = 3

    if window_size % 2 == 0:
        raise ValueError('Window should be an odd number.')

    if isinstance(kernel, str):
        if kernel == 'gaussian':
            kernel = gaussian(window_size, 1)
        elif kernel == 'mean':
            kernel = np.ones(window_size)
        else:
            raise NotImplementedError('Kernel: ' + kernel)
    kernel = kernel / np.asarray(kernel).sum()
    return convolve1d(array, kernel, mode='mirror')


def line_interpol(line, dx):
    """Interpolates a shapely LineString to a regularly spaced one.

    Shapely's interpolate function does not guaranty equally
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
    points = [line.interpolate(dx / 2.)]

    # Continue as long as line is not finished
    while True:
        pref = points[-1]
        pbs = pref.buffer(dx).boundary.intersection(line)
        if pbs.geom_type == 'Point':
            pbs = [pbs]
        elif pbs.geom_type == 'LineString':
            # This is rare
            pbs = [shpg.Point(c) for c in pbs.coords]
            assert len(pbs) == 2
        elif pbs.geom_type == 'GeometryCollection':
            # This is rare
            opbs = []
            for p in pbs.geoms:
                if p.geom_type == 'Point':
                    opbs.append(p)
                elif p.geom_type == 'LineString':
                    opbs.extend([shpg.Point(c) for c in p.coords])
            pbs = opbs
        else:
            if pbs.geom_type != 'MultiPoint':
                raise RuntimeError('line_interpol: we expect a MultiPoint '
                                   'but got a {}.'.format(pbs.geom_type))

        try:
            # Shapely v2 compat
            pbs = pbs.geoms
        except AttributeError:
            pass

        # Out of the point(s) that we get, take the one farthest from the top
        refdis = line.project(pref)
        tdis = np.array([line.project(pb) for pb in pbs])
        p = np.where(tdis > refdis)[0]
        if len(p) == 0:
            break
        points.append(pbs[int(p[0])])

    return points


def md(ref, data, axis=None):
    """Mean Deviation."""
    return np.mean(np.asarray(data) - ref, axis=axis)


def mad(ref, data, axis=None):
    """Mean Absolute Deviation."""
    return np.mean(np.abs(np.asarray(data) - ref), axis=axis)


def rmsd(ref, data, axis=None):
    """Root Mean Square Deviation."""
    return np.sqrt(np.mean((np.asarray(ref) - data)**2, axis=axis))


def rmsd_bc(ref, data):
    """Root Mean Squared Deviation of bias-corrected time series.

    I.e: rmsd(ref - mean(ref), data - mean(data)).
    """
    return rmsd(ref - np.mean(ref), data - np.mean(data))


def rel_err(ref, data):
    """Relative error. Ref should be non-zero"""
    return (np.asarray(data) - ref) / ref


def corrcoef(ref, data):
    """Peason correlation coefficient."""
    return np.corrcoef(ref, data)[0, 1]


def clip_scalar(value, vmin, vmax):
    """A faster numpy.clip ON SCALARS ONLY.

    See https://github.com/numpy/numpy/issues/14281
    """
    return vmin if value < vmin else vmax if value > vmax else value


def weighted_average_1d(data, weights):
    """A faster weighted average without dimension checks.

    We use it because it turned out to be quite a bottleneck in calibration
    """
    scl = np.sum(weights)
    if scl == 0:
        raise ZeroDivisionError("Weights sum to zero, can't be normalized")
    return np.multiply(data, weights).sum() / scl


def weighted_average_2d(data, weights):
    """A faster weighted average without dimension checks.

    Parameters
    ----------
    data : ArrayLike
        Must be of shape (n, m).
    weights : ArrayLike
        Must be of shape (n, ).

    We use it because it turned out to be quite a bottleneck in calibration.
    """
    scl = np.sum(weights, axis=0)
    if scl == 0:
        raise ZeroDivisionError(
            "Weights sum to zero, can't be normalized")
    # can't broadcast operands unless transposed
    return np.multiply(np.transpose(data), weights).sum(axis=1) / scl


if Version(np.__version__) < Version('1.17'):
    clip_array = np.clip
else:
    # TODO: reassess this when https://github.com/numpy/numpy/issues/14281
    # is solved
    clip_array = umath.clip


# A faster numpy.clip when only one value is clipped (here: min).
clip_min = umath.maximum

# A faster numpy.clip when only one value is clipped (here: max).
clip_max = umath.minimum


def nicenumber(number, binsize, lower=False):
    """Returns the next higher or lower "nice number", given by binsize.

    Examples:
    ---------
    >>> nicenumber(12, 10)
    20
    >>> nicenumber(19, 50)
    50
    >>> nicenumber(51, 50)
    100
    >>> nicenumber(51, 50, lower=True)
    50
    """

    e, _ = divmod(number, binsize)
    if lower:
        return e * binsize
    else:
        return (e + 1) * binsize


def signchange(ts):
    """Detect sign changes in a time series.

    http://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-
    for-elements-in-a-numpy-array

    Returns
    -------
    An array with 0s everywhere and 1's when the sign changes
    """
    asign = np.sign(ts)
    sz = asign == 0
    while sz.any():
        asign[sz] = np.roll(asign, 1)[sz]
        sz = asign == 0
    out = ((np.roll(asign, 1) - asign) != 0).astype(int)
    if asign.iloc[0] == asign.iloc[1]:
        out.iloc[0] = 0
    return out


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

    gdf = gdf.reset_index()

    for i, major in gdf.iterrows():

        # Exterior only
        major_poly = major.geometry.exterior

        # Remove the major from the list
        agdf = gdf.loc[gdf.index != i]

        # Keep catchments which intersect
        gdfs = agdf.loc[agdf.intersects(major_poly)]

        for j, neighbor in gdfs.iterrows():

            # No need to check if we already found the intersect
            if j in out.id_1 or j in out.id_2:
                continue

            # Exterior only
            neighbor_poly = neighbor.geometry.exterior

            # Ok, the actual intersection
            mult_intersect = major_poly.intersection(neighbor_poly)

            # All what follows is to catch all possibilities
            # Should not happen in our simple geometries but ya never know
            if isinstance(mult_intersect, shpg.Point):
                continue
            if isinstance(mult_intersect, shpg.linestring.LineString):
                mult_intersect = [mult_intersect]

            try:
                # Shapely v2 compat
                mult_intersect = mult_intersect.geoms
            except AttributeError:
                pass

            if len(mult_intersect) == 0:
                continue
            mult_intersect = [m for m in mult_intersect if
                              not isinstance(m, shpg.Point)]
            if len(mult_intersect) == 0:
                continue
            mult_intersect = linemerge(mult_intersect)
            if isinstance(mult_intersect, shpg.linestring.LineString):
                mult_intersect = [mult_intersect]

            try:
                # Compat with shapely > 2.0
                mult_intersect = mult_intersect.geoms
            except AttributeError:
                pass

            for line in mult_intersect:
                if not isinstance(line, shpg.linestring.LineString):
                    raise RuntimeError('polygon_intersections: we expect'
                                       'a LineString but got a '
                                       '{}.'.format(line.geom_type))
                line = gpd.GeoDataFrame([[i, j, line]],
                                        columns=out_cols)
                out = pd.concat([out, line])

    return out


def recursive_valid_polygons(geoms, crs):
    """Given a list of shapely geometries, makes sure all geometries are valid

    All will be valid polygons of area > 10000m2"""
    new_geoms = []
    for geom in geoms:
        new_geom = make_valid(geom)
        try:
            new_geoms.extend(recursive_valid_polygons(list(new_geom.geoms), crs))
        except AttributeError:
            new_s = gpd.GeoSeries(new_geom, crs=crs)
            if new_s.to_crs({'proj': 'cea'}).area.iloc[0] >= 10000:
                new_geoms.append(new_geom)
    assert np.all([type(geom) == shpg.Polygon for geom in new_geoms])
    return new_geoms


def combine_grids(gdirs):
    """ Combines individual grids of different glacier directories. The
    resulting grid extent includes all individual grids completely. The first
    glacier directory in the list defines the projection of the resulting grid.

    Parameters
    ----------
    gdirs : [], required
        A list of GlacierDirectories. The first gdir in the list defines the
        projection of the resulting grid.

    Returns
    -------
    salem.gis.Grid
    """

    new_grid = {
        'proj': None,
        'nxny': None,
        'dxdy': None,
        'x0y0': None,
        'pixel_ref': None
    }

    left_use = None
    right_use = None
    bottom_use = None
    top_use = None
    dx_use = None
    dy_use = None

    for gdir in gdirs:
        # use the first gdir to define some values
        if new_grid['proj'] is None:
            new_grid['proj'] = gdir.grid.proj
        if new_grid['pixel_ref'] is None:
            new_grid['pixel_ref'] = gdir.grid.pixel_ref

        # find largest extend including all grids completely
        (left, right, bottom, top) = gdir.grid.extent_in_crs(new_grid['proj'])
        if (left_use is None) or (left_use > left):
            left_use = left
        if right_use is None or right_use < right:
            right_use = right
        if bottom_use is None or bottom_use > bottom:
            bottom_use = bottom
        if top_use is None or top_use < top:
            top_use = top

        # find smallest dx and dy for the estimation of nx and ny
        dx = gdir.grid.dx
        dy = gdir.grid.dy
        if dx_use is None or dx_use > dx:
            dx_use = dx
        # dy could be negative
        if dy_use is None or abs(dy_use) > abs(dy):
            dy_use = dy

    # calculate nx and ny, the final extend could be one grid point larger or
    # smaller due to round()
    nx_use = round((right_use - left_use) / dx_use)
    ny_use = round((top_use - bottom_use) / abs(dy_use))

    # finally define the last values of the new grid
    if np.sign(dy_use) < 0:
        new_grid['x0y0'] = (left_use, top_use)
    else:
        new_grid['x0y0'] = (left_use, bottom_use)
    new_grid['nxny'] = (nx_use, ny_use)
    new_grid['dxdy'] = (dx_use, dy_use)

    return Grid.from_dict(new_grid)


def multipolygon_to_polygon(geometry, gdir=None):
    """Sometimes an RGI geometry is a multipolygon: this should not happen.

    It was vor vey old versions, pretty sure this is not needed anymore.

    Parameters
    ----------
    geometry : shpg.Polygon or shpg.MultiPolygon
        the geometry to check
    gdir : GlacierDirectory, optional
        for logging

    Returns
    -------
    the corrected geometry
    """

    # Log
    rid = gdir.rgi_id + ': ' if gdir is not None else ''

    if 'Multi' in geometry.geom_type:
        # needed for shapely version > 0.2.0
        # previous code was: parts = np.array(geometry)
        parts = []
        for p in geometry.geoms:
            parts.append(p)
        parts = np.array(parts)

        for p in parts:
            assert p.geom_type == 'Polygon'
        areas = np.array([p.area for p in parts])
        parts = parts[np.argsort(areas)][::-1]
        areas = areas[np.argsort(areas)][::-1]

        # First case (was RGIV4):
        # let's assume that one poly is exterior and that
        # the other polygons are in fact interiors
        exterior = parts[0].exterior
        interiors = []
        was_interior = 0
        for p in parts[1:]:
            if parts[0].contains(p):
                interiors.append(p.exterior)
                was_interior += 1
        if was_interior > 0:
            # We are done here, good
            geometry = shpg.Polygon(exterior, interiors)
        else:
            # This happens for bad geometries. We keep the largest
            geometry = parts[0]
            if np.any(areas[1:] > (areas[0] / 4)):
                log.info('Geometry {} lost quite a chunk.'.format(rid))

    if geometry.geom_type != 'Polygon':
        raise InvalidGeometryError('Geometry {} is not a Polygon.'.format(rid))
    return geometry


def _is_leap_year_vec(y):
    y = np.asarray(y, dtype=np.int64)
    return (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))


def date_to_floatyear(y, m, d=None):
    """Converts an integer (year, month) pair or (year, month, day) to a float
    year.

    This does account for leap years and the individual length of months.
    Tested in a round trip for all days between 1000 - 2500.

    Parameters
    ----------
    y : int or array_like
        the year
    m : int or array_like
        the month
    d : int or array_like or None
        the day. If None the 1. of the month is used
    """
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)

    if d is None:
        y, m = np.broadcast_arrays(y, m)
        d = np.ones_like(y, dtype=np.int64)
    else:
        d = np.asarray(d, dtype=np.int64)
        y, m, d = np.broadcast_arrays(y, m, d)

    leap = _is_leap_year_vec(y)
    days_in_year = 365 + leap.astype(np.int64)

    doy = d + _BASE_CUM_START[(m - 1)] + (leap & (m > 2))
    return y.astype(np.float64) + (doy - 1) / days_in_year


def floatyear_to_date(yr, return_day=False):
    """Converts a float year to an actual (year, month) pair or
    (year, month, day).

    This does account for leap years and the individual length of months.
    Tested in a round trip for all days between 1000 - 2500.

    Parameters
    ----------
    yr : float or array_like
        The floating year
    return_day : bool, optional
        If False a tuple (year, month) will be returned, if True a tuple
        (year, month, day) will be returned. Default is False.
    """
    yr = np.asarray(yr, dtype=np.float64)
    y = np.floor(yr).astype(np.int64)
    f = yr - y
    f = np.clip(f, 0.0, np.nextafter(1.0, 0.0))

    leap = _is_leap_year_vec(y)
    N = 365 + leap.astype(np.int64)

    x = f * N.astype(np.float64)
    doy_minus1 = np.rint(x).astype(np.int64)
    doy_minus1 = np.clip(doy_minus1, 0, N - 1)
    doy = doy_minus1 + 1

    adj = doy - (leap & (doy > 59)).astype(np.int64)  # collapse leap past Feb
    m_idx = np.searchsorted(_BASE_CUM_END, adj, side='left')  # 0..11
    month = (m_idx + 1).astype(np.int64)

    if not return_day:
        return y, month

    prev_end = np.where(m_idx > 0, _BASE_CUM_END[m_idx - 1], 0)
    day = (adj - prev_end).astype(np.int64)

    feb29 = leap & (doy == 60)
    if np.any(feb29):
        if np.size(feb29) == 1:
            # we only have a single date
            month = 2
            day = 29
        else:
            month[feb29] = 2
            day[feb29] = 29

    return y, month, day


def hydrodate_to_calendardate(y, m, start_month=None):
    """Converts a hydrological (year, month) pair to a calendar date.

    Parameters
    ----------
    y : int
        the year
    m : int
        the month
    start_month : int
        the first month of the hydrological year
    """

    if start_month is None:
        raise InvalidParamsError('In order to avoid confusion, we now force '
                                 'callers of this function to specify the '
                                 'hydrological convention they are using.')

    # nothing to do if start_month is 1
    if start_month == 1:
        return y, m

    y = np.array(y)
    m = np.array(m)

    e = 13 - start_month
    out_m = m + np.where(m <= e, start_month - 1, -e)
    out_y = y - np.where(m <= e, 1, 0)
    return out_y, out_m


def calendardate_to_hydrodate(y, m, start_month=None):
    """Converts a calendar (year, month) pair to a hydrological date.

    Parameters
    ----------
    y : int
        the year
    m : int
        the month
    start_month : int
        the first month of the hydrological year
    """

    if isinstance(y, xr.DataArray):
        y = y.values
    if isinstance(m, xr.DataArray):
        m = m.values

    if start_month is None:
        raise InvalidParamsError('In order to avoid confusion, we now force '
                                 'callers of this function to specify the '
                                 'hydrological convention they are using.')

    # nothing to do if start_month is 1
    if start_month == 1:
        return y, m

    y = np.array(y)
    m = np.array(m)

    out_m = m - start_month + np.where(m >= start_month, 1, 13)
    out_y = y + np.where(m >= start_month, 1, 0)
    return out_y, out_m


def calendardate_to_hydrodate_cftime(
    dates: np.ndarray, start_month: int
) -> np.ndarray:
    """Converts an array of Julian datetimes to hydrological dates.

    The offset is calculated based on days, so cftime automatically
    adjusts for different month/year lengths. Beware that a converted
    time series may not necessarily end on the last day of a month.

    Parameters
    ----------
    dates : np.ndarray[cftime.datetime]
        Julian datetimes.
    start_month : int
        The first month of the hydrological/water year.

    Returns
    -------
    np.ndarray[cftime.datetime]
        Dates following the hydrological/water calendar.
    """

    julian_year_start = int(dates[0].year)
    julian_month_start = int(dates[0].month)

    if start_month == 1:
        return dates  # no need to convert if year matches

    if not start_month:
        raise InvalidParamsError(
            "Specify the hydrological convention using the start month."
        )
    else:
        hydro_year_start = julian_year_start
        month_difference = julian_month_start - start_month
        if month_difference >= 0:
            hydro_month_start = 1 + month_difference
        else:
            hydro_year_start = julian_year_start - 1
            hydro_month_start = 13 + month_difference

    if start_month != 1:  # no need to convert if year start matches.
        offset = cftime.datetime(hydro_year_start, hydro_month_start, 1)
        timedelta = dates[0] - offset
        dates = np.vectorize(lambda x: x - timedelta)(dates)

    return dates


def hydrodate_to_calendardate_cftime(
    dates: np.ndarray, start_month: int
) -> np.ndarray:
    """Converts an array of hydrological datetimes to Julian dates.

    The offset is calculated based on days, so cftime automatically
    adjusts for different month/year lengths. Beware that a converted
    time series may not necessarily end on the last day of a month.

    Parameters
    ----------
    dates : np.ndarray[cftime.datetime]
        Hydrological dates.
    start_month : int
        The first month of the hydrological/water year.

    Returns
    -------
    np.ndarray[cftime.datetime]
        Dates following the Julian calendar.
    """

    hydro_year_start = int(dates[0].year)
    hydro_month_start = int(dates[0].month)

    if not start_month:
        raise InvalidParamsError(
            "Specify the hydrological convention using the start month."
        )
    elif start_month == 1:
        return dates  # no need to convert if year matches
    else:
        julian_year_start = hydro_year_start
        month_difference = start_month + hydro_month_start
        if month_difference > 13:
            julian_year_start = hydro_year_start + 1
            julian_month_start = month_difference - 13
        else:
            julian_month_start = month_difference - 1

    if start_month != 1:  # no need to convert if year start matches.
        offset = cftime.datetime(julian_year_start, julian_month_start, 1)
        timedelta = offset - dates[0]
        dates = np.vectorize(lambda x: x + timedelta)(dates)

    return dates


def get_days_of_year(year: float, use_leap_years: bool = False) -> int:
    """Get the number of days of a given year.

    Parameters
    ----------
    year : float or int
        Year in floating year convention.
    use_leap_years : bool
        Define if leap years should be returned.

    Returns
    -------
    int
        The number of days of a given year.
    """
    if use_leap_years:
        # use floatyear_to_date to avoid floating point mistakes (e.g. 1999.9999999)
        yr_int = floatyear_to_date(year)[0]
        return 366 if calendar.isleap(yr_int) else 365
    else:
        return 365


def get_seconds_of_year(year: float = None, use_leap_years: bool = False) -> int:
    """Get the number of seconds in a year.

    Parameters
    ----------
    year : float or int
        Year in floating year convention.
    use_leap_years : bool
        Define if leap years should be returned.

    Returns
    -------
    int
        The number of seconds in a year.
    """
    if use_leap_years:
        return SEC_IN_DAY * get_days_of_year(year, use_leap_years=use_leap_years)
    else:
        return SEC_IN_YEAR


def get_days_of_month(year: float = None, use_leap_years: bool = False) -> int:
    """Get the number of days in a month.

    Parameters
    ----------
    year : float or int
        Year in floating year convention.
    use_leap_years : bool
        Define if leap years should be returned.

    Returns
    -------
    int
        The number of days in the current month.
    """
    yr, mth = floatyear_to_date(year)
    if use_leap_years:
        return calendar.monthrange(yr, mth)[1]
    else:
        return {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30,
                10: 31, 11: 30, 12: 31}[mth]


def get_seconds_of_month(year: float = None, use_leap_years: bool = False) -> int:
    """Get the number of seconds in a year.

    Parameters
    ----------
    year : float or int
        Year in floating year convention.
    use_leap_years : bool
        Define if leap years should be returned.

    Returns
    -------
    int
        The number of seconds in the current month.
    """
    return SEC_IN_DAY * get_days_of_month(year=year,
                                          use_leap_years=use_leap_years)


def float_years_timeseries(y0, y1=None, ny=None, include_last_year=False,
                           daily=False):
    """Creates a timeseries in units of float years in monthly or daily
    resolution.

    Parameters
    ----------
    y0 : int
        The year to start the timeseries
    y1 : int
        The year to end the timeseries. If None you need to set ny.
        Default is None.
    ny : int
        The number of years of the timeseries. If None you need to set y1.
        Default is None.
    include_last_year : bool
        If the last year should be included. Default is False.
    daily : bool
        If True the resulting timeseries will be in daily resolution. If False
        it will be in monthly resolution. Default is False.

    """

    if isinstance(y0, xr.DataArray):
        y0 = y0.values
    if isinstance(y1, xr.DataArray):
        y1 = y1.values

    if y1 is None:
        if ny is not None:
            y1 = y0 + ny - 1
        else:
            raise ValueError("Need at least two positional arguments.")

    # convert both years to int
    y0 = int(np.floor(y0))
    y1 = int(np.floor(y1))

    if daily:
        if include_last_year:
            end_date = np.datetime64(f'{y1 + 1}-01-01', 'D')
        else:
            end_date = np.datetime64(f'{y1}-01-02', 'D')
        dates = np.arange(np.datetime64(f'{y0}-01-01', 'D'),
                          end_date,
                          dtype='datetime64[D]')
        years = dates.astype('datetime64[Y]').astype(int) + 1970
        months = (dates.astype('datetime64[M]').astype(int) % 12) + 1
        mstart = dates.astype('datetime64[M]').astype('datetime64[D]')
        days = (dates - mstart).astype(int) + 1
    else:
        if include_last_year:
            y1 += 1
            end_month = '01'  # last date is ignored by np.arange
        else:
            end_month = '02'  # last date is ignored by np.arange
        dates = np.arange(np.datetime64(f'{y0}-01', 'M'),
                          np.datetime64(f'{y1}-{end_month}', 'M'),
                          dtype='datetime64[M]')
        years = dates.astype('datetime64[Y]').astype(int) + 1970
        months = (dates.astype('datetime64[M]').astype(int) % 12) + 1
        days = None

    out = date_to_floatyear(y=years, m=months, d=days)
    return out


def filter_rgi_name(name):
    """Remove spurious characters and trailing blanks from RGI glacier name.

    This seems to be unnecessary with RGI V6
    """

    try:
        if name is None or len(name) == 0:
            return ''
    except TypeError:
        return ''

    if name[-1] in ['À', 'È', 'è', '\x9c', '3', 'Ð', '°', '¾',
                    '\r', '\x93', '¤', '0', '`', '/', 'C', '@',
                    'Å', '\x06', '\x10', '^', 'å', ';']:
        return filter_rgi_name(name[:-1])

    return name.strip().title()


def shape_factor_huss(widths, heights, is_rectangular):
    """Shape factor for lateral drag according to Huss and Farinotti (2012).

    The shape factor is only applied for parabolic sections.

    Parameters
    ----------
    widths: ndarray of floats
        widths of the sections
    heights: float or ndarray of floats
        height of the sections
    is_rectangular: bool or ndarray of bools
        determines, whether section has a rectangular or parabolic shape

    Returns
    -------
    shape factor (no units)
    """

    # Ensure bool (for masking)
    is_rect = is_rectangular.astype(bool)
    shape_factors = np.ones(widths.shape)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        shape_factors[~is_rect] = (widths / (2 * heights + widths))[~is_rect]

    # For very small thicknesses ignore
    shape_factors[heights <= 1.] = 1.
    # Security
    shape_factors = clip_array(shape_factors, 0.2, 1.)

    return shape_factors


def shape_factor_adhikari(widths, heights, is_rectangular):
    """Shape factor for lateral drag according to Adhikari (2012).

    TODO: other factors could be used when sliding is included

    Parameters
    ----------
    widths: ndarray of floats
        widths of the sections
    heights: ndarray of floats
        heights of the sections
    is_rectangular: ndarray of bools
        determines, whether section has a rectangular or parabolic shape

    Returns
    -------
    shape factors (no units), ndarray of floats
    """

    # Ensure bool (for masking)
    is_rectangular = is_rectangular.astype(bool)

    # Catch for division by 0 (corrected later)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        zetas = widths / 2. / heights

    shape_factors = np.ones(widths.shape)

    # TODO: higher order interpolation? (e.g. via InterpolatedUnivariateSpline)
    shape_factors[is_rectangular] = ADHIKARI_FACTORS_RECTANGULAR(
        zetas[is_rectangular])
    shape_factors[~is_rectangular] = ADHIKARI_FACTORS_PARABOLIC(
        zetas[~is_rectangular])

    shape_factors = clip_array(shape_factors, 0.2, 1.)
    # Set nan values resulting from zero height to a shape factor of 1
    shape_factors[np.isnan(shape_factors)] = 1.

    return shape_factors


def cook_rgidf(gi_gdf, o1_region, o2_region='01', version='60', ids=None,
               bgndate='20009999', id_suffix='', assign_column_values=None):
    """Convert a glacier inventory into a dataset looking like the RGI (OGGM ready).

    Parameters
    ----------
    gi_gdf : :py:geopandas.GeoDataFrame
        the GeoDataFrame of the user's glacier inventory.
    o1_region : str
        Glacier RGI region code, which is important for some OGGM applications.
        For example, oggm.shop.its_live() need it to locate the right dataset.
        Needs to be assigned.
    o2_region : str or list of str
        Glacier RGI subregion code (default: 01)
    bgndate : str or list of str
        The date of the outlines. This is quite important for the glacier
        evolution runs, which start at the RGI date. Format: ``'YYYYMMDD'``,
        (MMDD is not used).
    version : str
        Glacier inventory version code, which is necessary to generate the RGIId.
        The default is '60'.
    ids : list of IDs as integers. The default is None, which generates the
        RGI ids automatically following the glacier order.
    id_suffix : str or None
         Add a suffix to the glacier ids. The default is None, no suffix
    assign_column_values : dict or None
        Assign predefined values from the original data to the RGI dataframe.
        Dict format :
        - key: name of the column in the original dataframe
        - value: name of the column in the RGI-like file to assign the values to

    Returns
    -------
    cooked_rgidf : :py:geopandas.GeoDataFrame
        glacier inventory into a dataset looking like the RGI (OGGM ready)
    """

    # Check input
    if ids is None:
        ids = range(1, len(gi_gdf)+1)

    # Construct a fake rgiid list following RGI format
    str_ids = ['RGI{}-{}.{:05d}{}'.format(version, o1_region, int(i), id_suffix)
               for i in ids]

    # Check the coordination system.
    # RGI use the geographic coordinate.
    # So if the original glacier inventory is not in geographic coordinate,
    # we need to convert both of the central point and the glacier outline
    # to geographic coordinate (WGS 84)
    if gi_gdf.crs != 'epsg:4326':
        gi_gdf = gi_gdf.to_crs('epsg:4326')

    # Calculate the central point of the glaciers
    geom = gi_gdf.geometry.values
    centroids = gi_gdf.geometry.representative_point()
    clon, clat = centroids.x, centroids.y

    # Prepare data for the output GeoDataFrame
    data = {'RGIId': str_ids, 'CenLon': clon, 'CenLat': clat, 'GLIMSId': '',
            'BgnDate': bgndate, 'EndDate': '9999999',
            'O1Region': o1_region, 'O2Region': o2_region,
            'Area': -9999., 'Zmin': -9999., 'Zmax': -9999., 'Zmed': -9999.,
            'Slope': -9999., 'Aspect': -9999, 'Lmax': -9999,
            'Status': 0, 'Connect': 0, 'Form': 0, 'TermType': 0, 'Surging': 0,
            'Linkages': 1, 'check_geom': None, 'Name': ''}

    # Construct the output GeoDataFrame
    cooked_rgidf = gpd.GeoDataFrame(data=data, geometry=geom, crs='epsg:4326')

    # If there are specific column in the original glacier inventory we want to keep
    if assign_column_values is not None:
        for key, val in assign_column_values.items():
            cooked_rgidf[val] = gi_gdf[key].values

    return cooked_rgidf


def get_closest_grid_point_of_dataset(
    dataset: xr.Dataset, latitude: float, longitude: float
) -> xr.Dataset:
    """Get data of closest grid point of dataset."""
    try:
        c = (dataset.longitude - longitude) ** 2 + (
            dataset.latitude - latitude
        ) ** 2
        dataset = dataset.isel(points=np.argmin(c.data))
    except ValueError:
        # this should not occur for flattened data
        dataset = dataset.sel(
            longitude=longitude, latitude=latitude, method="nearest"
        )

    return dataset
