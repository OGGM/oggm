"""Some useful functions that did not fit into the other modules.

Copyright: OGGM developers, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division

import glob
import pickle

import pandas as pd
from salem import lazy_property
from six.moves.urllib.request import urlretrieve

# Builtins
import os
import shutil
import zipfile
import sys
from functools import partial, wraps

# External libs
import numpy as np
import netCDF4
import scipy.stats as stats
from joblib import Memory

# Locals
from oggm import CACHE_DIR
from oggm.conf import PATHS, BASENAMES

gh_zip = 'https://github.com/OGGM/oggm-sample-data/archive/master.zip'

# Joblib
memory = Memory(cachedir=CACHE_DIR, verbose=0)

gaussian_kernel = dict()
gaussian_kernel[9] = np.array([1.33830625e-04, 4.43186162e-03,
                               5.39911274e-02, 2.41971446e-01,
                               3.98943469e-01, 2.41971446e-01,
                               5.39911274e-02, 4.43186162e-03,
                               1.33830625e-04])
gaussian_kernel[7] = np.array([1.78435052e-04, 1.51942011e-02,
                               2.18673667e-01, 5.31907394e-01,
                               2.18673667e-01, 1.51942011e-02,
                               1.78435052e-04])
gaussian_kernel[5] = np.array([2.63865083e-04, 1.06450772e-01,
                               7.86570726e-01, 1.06450772e-01,
                               2.63865083e-04])

tuple2int = partial(np.array, dtype=np.int64)


def empty_cache():  # pragma: no cover
    """Empty oggm's cache directory."""

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)


def _download_demo_files():
    """Checks if the demo data is already on the cache and downloads it.

    TODO: Currently there's no check to see of the server file has changed
    this is bad. In the mean time, with empty_cache() you can ensure that the
    files are up-to-date.
    """

    ofile = os.path.join(CACHE_DIR, 'oggm-sample-data.zip')
    odir = os.path.join(CACHE_DIR)
    if not os.path.exists(ofile):  # pragma: no cover
        urlretrieve(gh_zip, ofile)
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(odir)

    out = dict()
    sdir = os.path.join(CACHE_DIR, 'oggm-sample-data-master')
    for root, directories, filenames in os.walk(sdir):
        for filename in filenames:
            out[filename] = os.path.join(root, filename)
    return out


def get_demo_file(fname):
    """Returns the path to the desired demo file."""

    d = _download_demo_files()
    if fname in d:
        return d[fname]
    else:
        return None


def query_yes_no(question, default="yes"):
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


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between one point
    on the earth and an array of points (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius of earth in meters
    return c * r


def interp_nans(array):
    """Interpolate NaNs using np.interp.

    np.interp is reasonable in that it does not extrapolate, it replaces
    NaNs at the bounds with the closest valid value.
    """

    _tmp = array.copy()
    nans, x = np.isnan(array), lambda z: z.nonzero()[0]
    _tmp[nans] = np.interp(x(nans), x(~nans), array[~nans])

    return _tmp


def md(ref, data, axis=None):
    """Mean Deviation."""
    return np.mean(data-ref, axis=axis)


def mad(ref, data, axis=None):
    """Mean Absolute Deviation."""
    return np.mean(np.abs(data-ref), axis=axis)


def rmsd(ref, data, axis=None):
    """Root Mean Square Deviation."""
    return np.sqrt(np.mean((ref-data)**2, axis=axis))


def rel_err(ref, data):
    """Relative error. Ref should be non-zero"""
    return (data - ref) / ref


def corrcoef(ref, data):
    """Peason correlation coefficient."""
    return np.corrcoef(ref, data)[0, 1]


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
        return (e+1) * binsize

@memory.cache
def joblib_read_climate(ncpath, ilon, ilat, default_grad, minmax_grad,
                        prcp_scaling_factor):
    """Prevent to re-compute a timeserie if it was done before.

    TODO: dirty solution, should be replaced by proper input.
    """

    # read the file and data
    nc = netCDF4.Dataset(ncpath, mode='r')
    temp = nc.variables['temp']
    prcp = nc.variables['prcp']
    hgt = nc.variables['hgt']

    igrad = np.zeros(len(nc.dimensions['time'])) + default_grad

    ttemp = temp[:, ilat-1:ilat+2, ilon-1:ilon+2]
    itemp = ttemp[:, 1, 1]
    thgt = hgt[ilat-1:ilat+2, ilon-1:ilon+2]
    ihgt = hgt[1, 1]
    thgt = thgt.flatten()
    iprcp = prcp[:, ilat, ilon] * prcp_scaling_factor

    # Now the gradient
    for t, loct in enumerate(ttemp):
        slope, _, _, p_val, _ = stats.linregress(thgt,
                                                 loct.flatten())
        igrad[t] = slope if (p_val < 0.01) else default_grad

    # dont exagerate too much
    igrad = np.clip(igrad, minmax_grad[0], minmax_grad[1])

    return iprcp, itemp, igrad, ihgt


class OggmTask(object):
    """Decorator for all OGGM tasks.

    Handles I/O requirements, catches errors and handles them, logs statuses.
    """

    def __init__(self, requires=[], writes=[]):
        self.requires = requires
        self.writes = writes

    def __call__(self, task_func):
        print("inside my_decorator.__call__()")

        @wraps(task_func)
        def wrapped_task(gdir, **kwargs):
            print("Inside wrapped_f()")
            print("Decorator arguments:", self.arg1, self.arg2, self.arg3)
            task_func(gdir, **kwargs)
            print("After f(*args)")

        return wrapped_task


class GlacierDir(object):
    """Facilitates read and write access to the glacier's files.

    It handles a glacier entity directory, created in a base directory (default
    is the "per_glacier" folder in the working dir). The glacier entity has,
    a map and a dem located in the base entity directory (also called divide_00
    internally). It has one or more divide subdirectories divide_01, ... divide_XX.
    Each divide has its own outlines and masks.  In the case the glacier has
    one single divide, the outlines and masks are thus linked to the divide_00 files.

    For memory optimisations it might do a couple of things twice.
    """

    def __init__(self, rgi_entity, base_dir=None, reset=False):
        """Creates a new directory or opens an existing one.

        Parameters
        ----------
        rgi_entity: glacier entity read from the shapefile
        base_dir: path to the directory where to open the directory
            defaults to "conf.PATHPATHS['working_dir'] + /per_glacier/"
        reset: emtpy the directory at construction (careful!)

        Attributes
        ----------
        dir : str
            path to the directory
        rgi_id : str
            The glacier's RGI identifier
        glims_id : str
            The glacier's GLIMS identifier (when available)
        rgi_area_km2 : float
            The glacier's RGI area (km2)
        cenlon : float
            The glacier's RGI central longitude
        rgi_date : datetime
            The RGI's BGNDATE attribute if available. Otherwise, defaults to
            2003-01-01
        """

        if base_dir is None:
            base_dir = os.path.join(PATHS['working_dir'], 'per_glacier')

        self.rgi_id = rgi_entity.RGIID
        self.glims_id = rgi_entity.GLIMSID
        self.rgi_area_km2 = float(rgi_entity.AREA)
        self.cenlon = float(rgi_entity.CENLON)
        self.cenlat = float(rgi_entity.CENLAT)
        try:
            rgi_date = pd.to_datetime(rgi_entity.BGNDATE[0:6],
                                      errors='raise', format='%Y%m')
        except:
            rgi_date = pd.to_datetime('200301', format='%Y%m')
        self.rgi_date = rgi_date

        self.dir = os.path.join(base_dir, self.rgi_id)
        if reset and os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Divides dirs have to be created by gis.define_glacier_region

    @lazy_property
    def grid(self):
        """A ``salem.Grid`` handling the georeferencing of the local grid."""

        return self.read_pickle('glacier_grid')

    @lazy_property
    def rgi_area_m2(self):
        """The glacier's RGI area (m2)."""

        return self.rgi_area_km2 * 10**6

    @property
    def divide_dirs(self):
        """Directory of the glacier divides.

        Must be called after gis.define_glacier_region."""

        dirs = [self.dir] + list(glob.glob(os.path.join(self.dir, 'divide_*')))
        return dirs

    @property
    def n_divides(self):
        """Number of glacier divides.

        Must be called after gis.define_glacier_region."""

        return len(self.divide_dirs)-1

    @property
    def divide_ids(self):
        """Iterator over the glacier divides ids

        Must be called after gis.define_glacier_region."""

        return range(1, self.n_divides+1)

    def get_filepath(self, filename, div_id=0):

        if filename not in BASENAMES:
            raise ValueError(filename + ' not in the dict.')

        dir = self.divide_dirs[div_id]

        return os.path.join(dir, BASENAMES[filename])

    def has_file(self, filename, div_id=0):

        return os.path.exists(self.get_filepath(filename, div_id=div_id))

    def read_pickle(self, filename, div_id=0):
        """Knows how to open pickles."""

        with open(self.get_filepath(filename, div_id), 'rb') as f:
            out = pickle.load(f)

        return out

    def write_pickle(self, var, filename, div_id=0):
        """Knows how to write pickles."""

        with open(self.get_filepath(filename, div_id), 'wb') as f:
            pickle.dump(var, f)

    def create_netcdf_file(self, fname, div_id=0):
        """Creates a netCDF4 file with geoinformation in it, the rest has to be
        filled by the calling routine.

        Returns
        -------
        A netCDF4.Dataset object
        """

        nc = netCDF4.Dataset(self.get_filepath(fname, div_id),
                             'w', format='NETCDF4')

        xd = nc.createDimension('x', self.grid.nx)
        yd = nc.createDimension('y', self.grid.ny)

        nc.author = 'OGGM'
        nc.author_info = 'Open Global Glacier Model'
        nc.proj_srs = self.grid.proj.srs

        lon, lat = self.grid.ll_coordinates
        x = self.grid.x0 + np.arange(self.grid.nx) * self.grid.dx
        y = self.grid.y0 + np.arange(self.grid.ny) * self.grid.dy

        v = nc.createVariable('x', 'f4', ('x',), zlib=True)
        v.units = 'm'
        v.long_name = 'x coordinate of projection'
        v.standard_name = 'projection_x_coordinate'
        v[:] = x

        v = nc.createVariable('y', 'f4', ('y',), zlib=True)
        v.units = 'm'
        v.long_name = 'y coordinate of projection'
        v.standard_name = 'projection_y_coordinate'
        v[:] = y

        v = nc.createVariable('longitude', 'f4', ('y', 'x'), zlib=True)
        v.units = 'degrees_east'
        v.long_name = 'longitude coordinate'
        v.standard_name = 'longitude'
        v[:] = lon

        v = nc.createVariable('latitude', 'f4', ('y', 'x'), zlib=True)
        v.units = 'degrees_north'
        v.long_name = 'latitude coordinate'
        v.standard_name = 'latitude'
        v[:] = lat

        return nc

    def create_monthly_climate_file(self, time, prcp, temp, grad, hgt):
        """Creates a netCDF4 file with climate data.

        See climate.distribute_climate_data"""

        nc = netCDF4.Dataset(self.get_filepath('climate_monthly'),
                             'w', format='NETCDF4')
        nc.ref_hgt = hgt

        dtime = nc.createDimension('time', None)

        nc.author = 'OGGM'
        nc.author_info = 'Open Global Glacier Model'

        timev = nc.createVariable('time','i4',('time',))
        timev.setncatts({'units':'days since 1801-01-01 00:00:00'})
        timev[:] = netCDF4.date2num([t for t in time],
                                 'days since 1801-01-01 00:00:00')

        v = nc.createVariable('prcp', 'f4', ('time',), zlib=True)
        v.units = 'kg m-2'
        v.long_name = 'total precipitation amount'
        v[:] = prcp

        v = nc.createVariable('temp', 'f4', ('time',), zlib=True)
        v.units = 'degC'
        v.long_name = '2m temperature at height ref_hgt'
        v[:] = temp

        v = nc.createVariable('grad', 'f4', ('time',), zlib=True)
        v.units = 'degC m-1'
        v.long_name = 'temperature gradient'
        v[:] = grad

        nc.close()