"""Some useful functions that did not fit into the other modules.

Copyright: OGGM developers, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division

import glob
import six.moves.cPickle as pickle

import pandas as pd
from salem import lazy_property
from six.moves.urllib.request import urlretrieve, urlopen
from six.moves.urllib.error import HTTPError

# Builtins
import os
import gzip
import shutil
import zipfile
import sys
import json
import time
from collections import OrderedDict
from functools import partial, wraps

# External libs
import geopandas as gpd
import numpy as np
import netCDF4
from scipy import stats
from joblib import Memory
from shapely.ops import transform as shp_trafo
from salem import wgs84

# Locals
import oggm.cfg as cfg

SAMPLE_DATA_GH_REPO = 'OGGM/oggm-sample-data'

# Joblib
MEMORY = Memory(cachedir=cfg.CACHE_DIR, verbose=0)

# Function
tuple2int = partial(np.array, dtype=np.int64)


def empty_cache():  # pragma: no cover
    """Empty oggm's cache directory."""

    if os.path.exists(cfg.CACHE_DIR):
        shutil.rmtree(cfg.CACHE_DIR)
    os.makedirs(cfg.CACHE_DIR)


def _download_demo_files():
    """Checks if the demo data is already on the cache and downloads it."""

    master_sha_url = 'https://api.github.com/repos/%s/commits/master' % \
                     SAMPLE_DATA_GH_REPO
    master_zip_url = 'https://github.com/%s/archive/master.zip' % \
                     SAMPLE_DATA_GH_REPO
    ofile = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data.zip')
    shafile = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-commit.txt')
    odir = os.path.join(cfg.CACHE_DIR)

    # a file containing the online's file's hash and the time of last check
    if os.path.exists(shafile):
        with open(shafile, 'r') as sfile:
            local_sha = sfile.read().strip()
        last_mod = os.path.getmtime(shafile)
    else:
        # very first download
        local_sha = '0000'
        last_mod = 0

    # test only every hour
    if time.time() - last_mod > 3600:
        write_sha = True
        try:
            # this might fail with HTTP 403 when server overload
            resp = urlopen(master_sha_url)

            # following try/finally is just for py2/3 compatibility
            # https://mail.python.org/pipermail/python-list/2016-March/704073.html
            try:
                json_str = resp.read().decode('utf-8')
            finally:
                resp.close()
            json_obj = json.loads(json_str)
            master_sha = json_obj['sha']
            # if not same, delete entire dir
            if local_sha != master_sha:
                empty_cache()
        except HTTPError:
            master_sha = 'error'
    else:
        write_sha = False

    # download only if necessary
    if not os.path.exists(ofile):
        urlretrieve(master_zip_url, ofile)
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(odir)

    # sha did change, replace
    if write_sha:
        with open(shafile, 'w') as sfile:
            sfile.write(master_sha)

    # list of files for output
    out = dict()
    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master')
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


def interp_nans(array, default=None):
    """Interpolate NaNs using np.interp.

    np.interp is reasonable in that it does not extrapolate, it replaces
    NaNs at the bounds with the closest valid value.
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


def md(ref, data, axis=None):
    """Mean Deviation."""
    return np.mean(np.asarray(data)-ref, axis=axis)


def mad(ref, data, axis=None):
    """Mean Absolute Deviation."""
    return np.mean(np.abs(np.asarray(data)-ref), axis=axis)


def rmsd(ref, data, axis=None):
    """Root Mean Square Deviation."""
    return np.sqrt(np.mean((np.asarray(ref)-data)**2, axis=axis))


def rel_err(ref, data):
    """Relative error. Ref should be non-zero"""
    return (np.asarray(data) - ref) / ref


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


@MEMORY.cache
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
    ihgt = thgt[1, 1]
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


def pipe_log(gdir, task_func, err=None):
    """Log the error in a specific directory."""

    fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, gdir.rgi_id)

    if err is not None:
        fpath += '.ERROR'
    else:
        return  # for now
        fpath += '.SUCCESS'

    with open(fpath, 'a') as f:
        f.write(task_func.__name__ + ': ')
        if err is not None:
            f.write(err.__class__.__name__ + ': {}'.format(err))


def get_centerline_lonlat(gdir):
    """Quick n dirty solution to write the centerlines as a shapefile"""

    olist = []
    for i in gdir.divide_ids:
        cls = gdir.read_pickle('centerlines', div_id=i)
        for i, cl in enumerate(cls):
            mm = 1 if i==0 else 0
            gs = gpd.GeoSeries()
            gs['RGIID'] = gdir.rgi_id
            gs['DIVIDE'] = i
            gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
            gs['MAIN'] = mm
            tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
            gs['geometry'] = shp_trafo(tra_func, cl.line)
            olist.append(gs)

    return olist

def write_centerlines_to_shape(gdirs, filename):
    """Write centerlines in a shapefile"""

    olist = []
    for gdir in gdirs:
        olist.extend(get_centerline_lonlat(gdir))

    odf = gpd.GeoDataFrame(olist)

    shema = dict()
    props = OrderedDict()
    props['RGIID'] = 'str:14'
    props['DIVIDE'] = 'int:9'
    props['LE_SEGMENT'] = 'int:9'
    props['MAIN'] = 'int:9'
    shema['geometry'] = 'LineString'
    shema['properties'] = props

    crs = {'init': 'epsg:4326'}

    # some writing function from geopandas rep
    from six import iteritems
    from shapely.geometry import mapping
    import fiona

    def feature(i, row):
        return {
            'id': str(i),
            'type': 'Feature',
            'properties':
                dict((k, v) for k, v in iteritems(row) if k != 'geometry'),
            'geometry': mapping(row['geometry'])}

    with fiona.open(filename, 'w', driver='ESRI Shapefile',
                    crs=crs, schema=shema) as c:
        for i, row in odf.iterrows():
            c.write(feature(i, row))


class entity_task(object):
    """Decorator for common job-controlling logic.

    All tasks share common operations. This decorator is here to handle them:
    exceptions, logging, and (some day) database for job-controlling.
    """

    def __init__(self, log, writes=[]):
        """Decorator syntax: ``@oggm_task(writes=['dem', 'outlines'])``

        Parameters
        ----------
        writes: list
            list of files that the task will write down to disk (must be
            available in ``cfg.BASENAMES``)
        """
        self.log = log
        self.writes = writes

        cnt =  ['    Returns']
        cnt += ['    -------']
        cnt += ['    Files writen to the glacier directory:']
        for k in sorted(writes):
            cnt += [cfg.BASENAMES.doc_str(k)]
        self.iodoc = '\n'.join(cnt)

    def __call__(self, task_func):
        """Decorate."""

        # Add to the original docstring
        task_func.__doc__ = '\n'.join((task_func.__doc__, self.iodoc))

        @wraps(task_func)
        def _entity_task(gdir, **kwargs):
            # Log only if needed:
            if not task_func.__dict__.get('divide_task', False):
                self.log.info('%s: %s', gdir.rgi_id, task_func.__name__)

            # Run the task
            if cfg.CONTINUE_ON_ERROR:
                try:
                    out = task_func(gdir, **kwargs)
                    gdir.log(task_func)
                except Exception as err:
                    # Something happened
                    out = None
                    gdir.log(task_func, err=err)
                    pipe_log(gdir, task_func, err=err)
            else:
                out = task_func(gdir, **kwargs)
            return out
        return _entity_task


class divide_task(object):
    """Decorator for common logic on divides.

    Simply calls the decorated task once for each divide.
    """

    def __init__(self, log, add_0=False):
        """Decorator

        Parameters
        ----------
        add_0: bool, default=False
            If the task also needs to be run on divide 0
        """
        self.log = log
        self.add_0 = add_0
        self._cdoc = """"
            div_id : int
                the ID of the divide to process. Should be left to  the default
                ``None`` unless you know what you do.
        """

    def __call__(self, task_func):
        """Decorate."""

        @wraps(task_func)
        def _divide_task(gdir, div_id=None, **kwargs):
            if div_id is None:
                ids = gdir.divide_ids
                if self.add_0:
                    ids = [0] + list(ids)
                for i in ids:
                    self.log.info('%s: %s, divide %d', gdir.rgi_id,
                                  task_func.__name__, i)
                    task_func(gdir, div_id=i, **kwargs)
            else:
                # For testing only
                task_func(gdir, div_id=div_id, **kwargs)

        # For the logger later on
        _divide_task.__dict__['divide_task'] = True
        return _divide_task


class GlacierDirectory(object):
    """Organizes read and write access to the glacier's files.

    It handles a glacier directory created in a base directory (default
    is the "per_glacier" folder in the working directory). The role of a
    GlacierDirectory is to give access to file paths and to I/O operations
    in a transparent way. The user should not care about *where* the files are
    located, but should know their name (see :ref:`basenames`).

    A glacier entity has one or more divides. See :ref:`glacierdir`
    for more information.
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
            base_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')

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

        # The divides dirs are created by gis.define_glacier_region

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
        """list of the glacier divides directories."""
        dirs = [self.dir] + list(glob.glob(os.path.join(self.dir, 'divide_*')))
        return dirs

    @property
    def n_divides(self):
        """Number of glacier divides."""
        return len(self.divide_dirs)-1

    @property
    def divide_ids(self):
        """Iterator over the glacier divides ids."""
        return range(1, self.n_divides+1)

    def get_filepath(self, filename, div_id=0):
        """Absolute path to a specific file.

        Parameters
        ----------
        filename: str
            file name (must be listed in cfg.BASENAME)
        div_id: int
            the divide for which you want to get the file path

        Returns
        -------
        The absolute path to the desired file
        """

        if filename not in cfg.BASENAMES:
            raise ValueError(filename + ' not in cfg.BASENAMES.')

        dir = self.divide_dirs[div_id]

        return os.path.join(dir, cfg.BASENAMES[filename])

    def has_file(self, filename, div_id=0):

        return os.path.exists(self.get_filepath(filename, div_id=div_id))

    def read_pickle(self, filename, div_id=0):
        """ Reads a pickle located in the directory.

        Parameters
        ----------
        filename: str
            file name (must be listed in cfg.BASENAME)
        div_id: int
            the divide for which you want to get the file path

        Returns
        -------
        An object read from the pickle
        """

        _open = gzip.open if cfg.PARAMS['use_compression'] else open
        with _open(self.get_filepath(filename, div_id), 'rb') as f:
            out = pickle.load(f)

        return out

    def write_pickle(self, var, filename, div_id=0):
        """ Writes a variable to a pickle on disk.

        Parameters
        ----------
        var: object
            the variable to write to disk
        filename: str
            file name (must be listed in cfg.BASENAME)
        div_id: int
            the divide for which you want to get the file path
        """

        _open = gzip.open if cfg.PARAMS['use_compression'] else open
        with _open(self.get_filepath(filename, div_id), 'wb') as f:
            pickle.dump(var, f, protocol=-1)

    def create_gridded_ncdf_file(self, fname, div_id=0):
        """Makes a gridded netcdf file template.

        The other variables have to be created and filled by the calling
        routine.

        Parameters
        ----------
        filename: str
            file name (must be listed in cfg.BASENAME)
        div_id: int
            the divide for which you want to get the file path

        Returns
        -------
        a ``netCDF4.Dataset`` object.
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

    def write_monthly_climate_file(self, time, prcp, temp, grad, hgt):
        """Creates a netCDF4 file with climate data.

        See :py:func:`oggm.tasks.distribute_climate_data`.
        """

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

    def log(self, func, err=None):

        fpath = os.path.join(self.dir, 'log')
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fpath = os.path.join(fpath, func.__name__)

        if err is not None:
            fpath += '.ERROR'
        else:
            fpath += '.SUCCESS'

        with open(fpath, 'w') as f:
            f.write(func.__name__ + '\n')
            if err is not None:
                f.write(err.__class__.__name__ + ': {}'.format(err))
