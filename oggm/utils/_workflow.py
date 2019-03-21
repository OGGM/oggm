"""Classes and functions used by the OGGM workflow"""

# Builtins
import glob
import os
import tempfile
import gzip
import json
import time
import random
import shutil
import tarfile
import sys
import datetime
import logging
import pickle
import warnings
from collections import OrderedDict
from functools import partial, wraps
from time import gmtime, strftime
import fnmatch
import platform
import struct
import importlib

# External libs
import geopandas as gpd
import pandas as pd
import salem
from salem import lazy_property, wgs84
import numpy as np
import netCDF4
from scipy import stats
import xarray as xr
import shapely.geometry as shpg
from shapely.ops import transform as shp_trafo

# Locals
from oggm import __version__
from oggm.utils._funcs import (calendardate_to_hydrodate, date_to_floatyear,
                               tolist, filter_rgi_name, parse_rgi_meta,
                               haversine)
from oggm.utils._downloads import (get_demo_file, get_wgms_files,
                                   get_rgi_glacier_entities)
from oggm import cfg
from oggm.exceptions import InvalidParamsError


# Default RGI date (median per region)
RGI_DATE = {'01': 2009,
            '02': 2004,
            '03': 1999,
            '04': 2001,
            '05': 2001,
            '06': 2000,
            '07': 2008,
            '08': 2002,
            '09': 2001,
            '10': 2011,
            '11': 2003,
            '12': 2001,
            '13': 2006,
            '14': 2001,
            '15': 2001,
            '16': 2000,
            '17': 2000,
            '18': 1978,
            '19': 1989,
            }

# Module logger
log = logging.getLogger('.'.join(__name__.split('.')[:-1]))


def empty_cache():
    """Empty oggm's cache directory."""

    if os.path.exists(cfg.CACHE_DIR):
        shutil.rmtree(cfg.CACHE_DIR)
    os.makedirs(cfg.CACHE_DIR)


def expand_path(p):
    """Helper function for os.path.expanduser and os.path.expandvars"""

    return os.path.expandvars(os.path.expanduser(p))


def gettempdir(dirname='', reset=False, home=False):
    """Get a temporary directory.

    The default is to locate it in the system's temporary directory as
    given by python's `tempfile.gettempdir()/OGGM'. You can set `home=True` for
    a directory in the user's `home/tmp` folder instead (this isn't really
    a temporary folder but well...)

    Parameters
    ----------
    dirname : str
        if you want to give it a name
    reset : bool
        if it has to be emptied first.
    home : bool
        if True, returns `HOME/tmp/OGGM` instead

    Returns
    -------
    the path to the temporary directory
    """

    basedir = (os.path.join(os.path.expanduser('~'), 'tmp') if home
               else tempfile.gettempdir())
    return mkdir(os.path.join(basedir, 'OGGM', dirname), reset=reset)


def get_sys_info():
    """Returns system information as a dict"""

    blob = []
    try:
        (sysname, nodename, release,
         version, machine, processor) = platform.uname()
        blob.extend([
            ("python", "%d.%d.%d.%s.%s" % sys.version_info[:]),
            ("python-bits", struct.calcsize("P") * 8),
            ("OS", "%s" % (sysname)),
            ("OS-release", "%s" % (release)),
            ("machine", "%s" % (machine)),
            ("processor", "%s" % (processor)),
        ])
    except BaseException:
        pass

    return blob


def show_versions(logger=None):
    """Prints the OGGM version and other system information.

    Parameters
    ----------
    logger : optional
        the logger you want to send the printouts to. If None, will use stdout

    Returns
    -------
    the output string
    """

    _print = print if logger is None else logger.workflow

    sys_info = get_sys_info()

    deps = [
        # (MODULE_NAME, f(mod) -> mod version)
        ("oggm", lambda mod: mod.__version__),
        ("numpy", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("geopandas", lambda mod: mod.__version__),
        ("netCDF4", lambda mod: mod.__version__),
        ("matplotlib", lambda mod: mod.__version__),
        ("rasterio", lambda mod: mod.__version__),
        ("fiona", lambda mod: mod.__version__),
        ("osgeo.gdal", lambda mod: mod.__version__),
        ("pyproj", lambda mod: mod.__version__),
        ("shapely", lambda mod: mod.__version__),
        ("xarray", lambda mod: mod.__version__),
        ("salem", lambda mod: mod.__version__),
    ]

    deps_blob = list()
    for (modname, ver_f) in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = ver_f(mod)
            deps_blob.append((modname, ver))
        except BaseException:
            deps_blob.append((modname, None))

    out = ["# System info:"]
    for k, stat in sys_info:
        out.append("%s: %s" % (k, stat))
    out.append("# Packages info:")
    for k, stat in deps_blob:
        out.append("%s: %s" % (k, stat))
    for l in out:
        _print(l)
    return '\n'.join(out)


class SuperclassMeta(type):
    """Metaclass for abstract base classes.

    http://stackoverflow.com/questions/40508492/python-sphinx-inherit-
    method-documentation-from-superclass
    """
    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)
        for name, member in cls_dict.items():
            if not getattr(member, '__doc__'):
                try:
                    member.__doc__ = getattr(bases[-1], name).__doc__
                except AttributeError:
                    pass
        return cls


class LRUFileCache():
    """A least recently used cache for temporary files.

    The files which are no longer used are deleted from the disk.
    """

    def __init__(self, l0=None, maxsize=100):
        """Instanciate.

        Parameters
        ----------
        l0 : list
            a list of file paths
        maxsize : int
            the max number of files to keep
        """
        self.files = [] if l0 is None else l0
        self.maxsize = maxsize
        self.purge()

    def purge(self):
        """Remove expired entries."""
        if len(self.files) > self.maxsize:
            fpath = self.files.pop(0)
            if os.path.exists(fpath):
                os.remove(fpath)

    def append(self, fpath):
        """Append a file to the list."""
        if fpath not in self.files:
            self.files.append(fpath)
        self.purge()


def mkdir(path, reset=False):
    """Checks if directory exists and if not, create one.

    Parameters
    ----------
    reset: erase the content of the directory if exists

    Returns
    -------
    the path
    """

    if reset and os.path.exists(path):
        shutil.rmtree(path)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path


def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().

    https://stackoverflow.com/questions/35155382/copying-specific-files-to-a-
    new-folder-while-maintaining-the-original-subdirect
    """

    def _ignore_patterns(path, names):
        # This is our cuisine
        bname = os.path.basename(path)
        if 'divide' in bname or 'log' in bname:
            keep = []
        else:
            keep = set(name for pattern in patterns
                       for name in fnmatch.filter(names, pattern))
        ignore = set(name for name in names
                     if name not in keep and not
                     os.path.isdir(os.path.join(path, name)))
        return ignore

    return _ignore_patterns


class ncDataset(netCDF4.Dataset):
    """Wrapper around netCDF4 setting auto_mask to False"""

    def __init__(self, *args, **kwargs):
        super(ncDataset, self).__init__(*args, **kwargs)
        self.set_auto_mask(False)


def pipe_log(gdir, task_func_name, err=None):
    """Log the error in a specific directory."""

    time_str = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    # Defaults to working directory: it must be set!
    if not cfg.PATHS['working_dir']:
        warnings.warn("Cannot log to file without a valid "
                      "cfg.PATHS['working_dir']!", RuntimeWarning)
        return

    fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
    mkdir(fpath)

    fpath = os.path.join(fpath, gdir.rgi_id)

    sep = '; '

    if err is not None:
        fpath += '.ERROR'
    else:
        return  # for now
        fpath += '.SUCCESS'

    with open(fpath, 'a') as f:
        f.write(time_str + sep + task_func_name + sep)
        if err is not None:
            f.write(err.__class__.__name__ + sep + '{}\n'.format(err))
        else:
            f.write(sep + '\n')


class DisableLogger():
    """Context manager to temporarily disable all loggers."""

    def __enter__(self):
        logging.disable(logging.ERROR)

    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


class entity_task(object):
    """Decorator for common job-controlling logic.

    All tasks share common operations. This decorator is here to handle them:
    exceptions, logging, and (some day) database for job-controlling.
    """

    def __init__(self, log, writes=[], fallback=None):
        """Decorator syntax: ``@oggm_task(writes=['dem', 'outlines'])``

        Parameters
        ----------
        writes: list
            list of files that the task will write down to disk (must be
            available in ``cfg.BASENAMES``)
        fallback: python function
            will be executed on gdir if entity_task fails
        """
        self.log = log
        self.writes = writes
        self.fallback = fallback

        cnt = ['    Notes']
        cnt += ['    -----']
        cnt += ['    Files writen to the glacier directory:']

        for k in sorted(writes):
            cnt += [cfg.BASENAMES.doc_str(k)]
        self.iodoc = '\n'.join(cnt)

    def __call__(self, task_func):
        """Decorate."""

        # Add to the original docstring
        if task_func.__doc__ is None:
            raise RuntimeError('Entity tasks should have a docstring!')

        task_func.__doc__ = '\n'.join((task_func.__doc__, self.iodoc))

        @wraps(task_func)
        def _entity_task(gdir, *, reset=None, print_log=True, **kwargs):

            if reset is None:
                reset = not cfg.PARAMS['auto_skip_task']

            task_name = task_func.__name__

            # Filesuffix are typically used to differentiate tasks
            fsuffix = (kwargs.get('filesuffix', False) or
                       kwargs.get('output_filesuffix', False))
            if fsuffix:
                task_name += fsuffix

            # Do we need to run this task?
            s = gdir.get_task_status(task_name)
            if not reset and s and ('SUCCESS' in s):
                return

            # Log what we are doing
            if print_log:
                self.log.info('(%s) %s', gdir.rgi_id, task_name)

            # Run the task
            try:
                out = task_func(gdir, **kwargs)
                if task_name != 'gdir_to_tar':
                    gdir.log(task_name)
            except Exception as err:
                # Something happened
                out = None
                gdir.log(task_name, err=err)
                pipe_log(gdir, task_name, err=err)
                if print_log:
                    self.log.error('%s occurred during task %s on %s: %s',
                                   type(err).__name__, task_name,
                                   gdir.rgi_id, str(err))
                if not cfg.PARAMS['continue_on_error']:
                    raise

                if self.fallback is not None:
                    self.fallback(gdir)
            return out

        _entity_task.__dict__['is_entity_task'] = True
        return _entity_task


def global_task(task_func):
    """
    Decorator for common job-controlling logic.

    Indicates that this task expects a list of all GlacierDirs as parameter
    instead of being called once per dir.
    """

    task_func.__dict__['global_task'] = True
    return task_func


def _get_centerline_lonlat(gdir):
    """Quick n dirty solution to write the centerlines as a shapefile"""

    cls = gdir.read_pickle('centerlines')
    olist = []
    for j, cl in enumerate(cls[::-1]):
        mm = 1 if j == 0 else 0
        gs = gpd.GeoSeries()
        gs['RGIID'] = gdir.rgi_id
        gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
        gs['MAIN'] = mm
        tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
        gs['geometry'] = shp_trafo(tra_func, cl.line)
        olist.append(gs)

    return olist


def write_centerlines_to_shape(gdirs, filesuffix='', path=True):
    """Write the centerlines in a shapefile.

    Parameters
    ----------
    gdirs: the list of GlacierDir to process.
    filesuffix : str
        add suffix to output file
    path:
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """

    if path is True:
        path = os.path.join(cfg.PATHS['working_dir'],
                            'glacier_centerlines' + filesuffix + '.shp')

    olist = []
    for gdir in gdirs:
        olist.extend(_get_centerline_lonlat(gdir))

    odf = gpd.GeoDataFrame(olist)

    shema = dict()
    props = OrderedDict()
    props['RGIID'] = 'str:14'
    props['LE_SEGMENT'] = 'int:9'
    props['MAIN'] = 'int:9'
    shema['geometry'] = 'LineString'
    shema['properties'] = props

    crs = {'init': 'epsg:4326'}

    # some writing function from geopandas rep
    from shapely.geometry import mapping
    import fiona

    def feature(i, row):
        return {
            'id': str(i),
            'type': 'Feature',
            'properties':
                dict((k, v) for k, v in row.items() if k != 'geometry'),
            'geometry': mapping(row['geometry'])}

    with fiona.open(path, 'w', driver='ESRI Shapefile',
                    crs=crs, schema=shema) as c:
        for i, row in odf.iterrows():
            c.write(feature(i, row))


def demo_glacier_id(key):
    """Get the RGI id of a glacier by name or key: None if not found."""

    df = cfg.DEMO_GLACIERS

    # Is the name in key?
    s = df.loc[df.Key.str.lower() == key.lower()]
    if len(s) == 1:
        return s.index[0]

    # Is the name in name?
    s = df.loc[df.Name.str.lower() == key.lower()]
    if len(s) == 1:
        return s.index[0]

    # Is the name in Ids?
    try:
        s = df.loc[[key]]
        if len(s) == 1:
            return s.index[0]
    except KeyError:
        pass

    return None


def compile_run_output(gdirs, path=True, filesuffix='', use_compression=True):
    """Merge the output of the model runs of several gdirs into one file.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    path : str
        where to store (default is on the working dir).
        Set to `False` to disable disk storage.
    filesuffix : str
        the filesuffix of the run
    use_compression : bool
        use zlib compression on the output netCDF files

    Returns
    -------
    ds : :py:class:`xarray.Dataset`
        compiled output
    """

    # Get the dimensions of all this
    rgi_ids = [gd.rgi_id for gd in gdirs]

    # The first gdir might have blown up, try some others
    i = 0
    while True:
        if i >= len(gdirs):
            raise RuntimeError('Found no valid glaciers!')
        try:
            ppath = gdirs[i].get_filepath('model_diagnostics',
                                          filesuffix=filesuffix)
            with xr.open_dataset(ppath) as ds_diag:
                ds_diag.time.values
            break
        except BaseException:
            i += 1

    # OK found it, open it and prepare the output
    with xr.open_dataset(ppath) as ds_diag:
        time = ds_diag.time.values
        yrs = ds_diag.hydro_year.values
        months = ds_diag.hydro_month.values
        cyrs = ds_diag.calendar_year.values
        cmonths = ds_diag.calendar_month.values

        # Prepare output
        ds = xr.Dataset()

        # Global attributes
        ds.attrs['description'] = 'OGGM model output'
        ds.attrs['oggm_version'] = __version__
        ds.attrs['calendar'] = '365-day no leap'
        ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # Coordinates
        ds.coords['time'] = ('time', time)
        ds.coords['rgi_id'] = ('rgi_id', rgi_ids)
        ds.coords['hydro_year'] = ('time', yrs)
        ds.coords['hydro_month'] = ('time', months)
        ds.coords['calendar_year'] = ('time', cyrs)
        ds.coords['calendar_month'] = ('time', cmonths)
        ds['time'].attrs['description'] = 'Floating hydrological year'
        ds['rgi_id'].attrs['description'] = 'RGI glacier identifier'
        ds['hydro_year'].attrs['description'] = 'Hydrological year'
        ds['hydro_month'].attrs['description'] = 'Hydrological month'
        ds['calendar_year'].attrs['description'] = 'Calendar year'
        ds['calendar_month'].attrs['description'] = 'Calendar month'

    shape = (len(time), len(rgi_ids))
    vol = np.zeros(shape)
    area = np.zeros(shape)
    length = np.zeros(shape)
    ela = np.zeros(shape)
    for i, gdir in enumerate(gdirs):
        try:
            ppath = gdir.get_filepath('model_diagnostics',
                                      filesuffix=filesuffix)
            with xr.open_dataset(ppath) as ds_diag:
                vol[:, i] = ds_diag.volume_m3.values
                area[:, i] = ds_diag.area_m2.values
                length[:, i] = ds_diag.length_m.values
                ela[:, i] = ds_diag.ela_m.values
        except BaseException:
            vol[:, i] = np.NaN
            area[:, i] = np.NaN
            length[:, i] = np.NaN
            ela[:, i] = np.NaN

    ds['volume'] = (('time', 'rgi_id'), vol)
    ds['volume'].attrs['description'] = 'Total glacier volume'
    ds['volume'].attrs['units'] = 'm 3'
    ds['area'] = (('time', 'rgi_id'), area)
    ds['area'].attrs['description'] = 'Total glacier area'
    ds['area'].attrs['units'] = 'm 2'
    ds['length'] = (('time', 'rgi_id'), length)
    ds['length'].attrs['description'] = 'Glacier length'
    ds['length'].attrs['units'] = 'm'
    ds['ela'] = (('time', 'rgi_id'), ela)
    ds['ela'].attrs['description'] = 'Glacier Equilibrium Line Altitude (ELA)'
    ds['ela'].attrs['units'] = 'm a.s.l'

    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                'run_output' + filesuffix + '.nc')

        enc_var = {'dtype': 'float32'}
        if use_compression:
            enc_var['complevel'] = 5
            enc_var['zlib'] = True
        encoding = {v: enc_var for v in ['volume', 'area', 'length', 'ela']}

        ds.to_netcdf(path, encoding=encoding)

    return ds


def compile_climate_input(gdirs, path=True, filename='climate_monthly',
                          filesuffix='', use_compression=True):
    """Merge the climate input files in the glacier directories into one file.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    path : str
        where to store (default is on the working dir).
        Set to `False` to disable disk storage.
    filename : str
        BASENAME of the climate input files
    filesuffix : str
        the filesuffix of the compiled file
    use_compression : bool
        use zlib compression on the output netCDF files

    Returns
    -------
    ds : :py:class:`xarray.Dataset`
        compiled climate data
    """

    # Get the dimensions of all this
    rgi_ids = [gd.rgi_id for gd in gdirs]

    # The first gdir might have blown up, try some others
    i = 0
    while True:
        if i >= len(gdirs):
            raise RuntimeError('Found no valid glaciers!')
        try:
            ppath = gdirs[i].get_filepath(filename=filename,
                                          filesuffix=filesuffix)
            with xr.open_dataset(ppath) as ds_clim:
                ds_clim.time.values
            break
        except BaseException:
            i += 1

    with xr.open_dataset(ppath) as ds_clim:
        cyrs = ds_clim['time.year']
        cmonths = ds_clim['time.month']
        has_grad = 'gradient' in ds_clim.variables

    yrs, months = calendardate_to_hydrodate(cyrs, cmonths)
    time = date_to_floatyear(yrs, months)

    # Prepare output
    ds = xr.Dataset()

    # Global attributes
    ds.attrs['description'] = 'OGGM model output'
    ds.attrs['oggm_version'] = __version__
    ds.attrs['calendar'] = '365-day no leap'
    ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # Coordinates
    ds.coords['time'] = ('time', time)
    ds.coords['rgi_id'] = ('rgi_id', rgi_ids)
    ds.coords['hydro_year'] = ('time', yrs)
    ds.coords['hydro_month'] = ('time', months)
    ds.coords['calendar_year'] = ('time', cyrs)
    ds.coords['calendar_month'] = ('time', cmonths)
    ds['time'].attrs['description'] = 'Floating hydrological year'
    ds['rgi_id'].attrs['description'] = 'RGI glacier identifier'
    ds['hydro_year'].attrs['description'] = 'Hydrological year'
    ds['hydro_month'].attrs['description'] = 'Hydrological month'
    ds['calendar_year'].attrs['description'] = 'Calendar year'
    ds['calendar_month'].attrs['description'] = 'Calendar month'

    shape = (len(time), len(rgi_ids))
    temp = np.zeros(shape) * np.NaN
    prcp = np.zeros(shape) * np.NaN
    if has_grad:
        grad = np.zeros(shape) * np.NaN
    ref_hgt = np.zeros(len(rgi_ids)) * np.NaN
    ref_pix_lon = np.zeros(len(rgi_ids)) * np.NaN
    ref_pix_lat = np.zeros(len(rgi_ids)) * np.NaN

    for i, gdir in enumerate(gdirs):
        try:
            ppath = gdir.get_filepath(filename=filename,
                                      filesuffix=filesuffix)
            with xr.open_dataset(ppath) as ds_clim:
                prcp[:, i] = ds_clim.prcp.values
                temp[:, i] = ds_clim.temp.values
                if has_grad:
                    grad[:, i] = ds_clim.gradient
                ref_hgt[i] = ds_clim.ref_hgt
                ref_pix_lon[i] = ds_clim.ref_pix_lon
                ref_pix_lat[i] = ds_clim.ref_pix_lat
        except BaseException:
            pass

    ds['temp'] = (('time', 'rgi_id'), temp)
    ds['temp'].attrs['units'] = 'DegC'
    ds['temp'].attrs['description'] = '2m Temperature at height ref_hgt'
    ds['prcp'] = (('time', 'rgi_id'), prcp)
    ds['prcp'].attrs['units'] = 'kg m-2'
    ds['prcp'].attrs['description'] = 'total monthly precipitation amount'
    if has_grad:
        ds['grad'] = (('time', 'rgi_id'), grad)
        ds['grad'].attrs['units'] = 'degC m-1'
        ds['grad'].attrs['description'] = 'temperature gradient'
    ds['ref_hgt'] = ('rgi_id', ref_hgt)
    ds['ref_hgt'].attrs['units'] = 'm'
    ds['ref_hgt'].attrs['description'] = 'reference height'
    ds['ref_pix_lon'] = ('rgi_id', ref_pix_lon)
    ds['ref_pix_lon'].attrs['description'] = 'longitude'
    ds['ref_pix_lat'] = ('rgi_id', ref_pix_lat)
    ds['ref_pix_lat'].attrs['description'] = 'latitude'

    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                'climate_input' + filesuffix + '.nc')

        enc_var = {'dtype': 'float32'}
        if use_compression:
            enc_var['complevel'] = 5
            enc_var['zlib'] = True
        vars = ['temp', 'prcp']
        if has_grad:
            vars += ['grad']
        encoding = {v: enc_var for v in vars}

        ds.to_netcdf(path, encoding=encoding)
    return ds


def compile_task_log(gdirs, task_names=[], filesuffix='', path=True,
                     append=True):
    """Gathers the log output for the selected task(s)

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    task_names : list of str
        The tasks to check for
    filesuffix : str
        add suffix to output file
    path:
        Set to `True` in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
        Set to `False` to omit disk storage
    append:
        If a task log file already exists in the working directory, the new
        logs will be added to the existing file

    Returns
    -------
    out : :py:class:`pandas.DataFrame`
        log output
    """

    out_df = []
    for gdir in gdirs:
        d = OrderedDict()
        d['rgi_id'] = gdir.rgi_id
        for task_name in task_names:
            ts = gdir.get_task_status(task_name)
            if ts is None:
                ts = ''
            d[task_name] = ts.replace(',', ' ')
        out_df.append(d)

    out = pd.DataFrame(out_df).set_index('rgi_id')
    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                'task_log' + filesuffix + '.csv')
        if os.path.exists(path) and append:
            odf = pd.read_csv(path, index_col=0)
            out = odf.join(out, rsuffix='_n')
        out.to_csv(path)
    return out


@entity_task(log)
def glacier_statistics(gdir, inversion_only=False):
    """Gather as much statistics as possible about this glacier.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    inversion_only : bool
        if one wants to summarize the inversion output only (including calving)
    """

    d = OrderedDict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['name'] = gdir.name
    d['cenlon'] = gdir.cenlon
    d['cenlat'] = gdir.cenlat
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['glacier_type'] = gdir.glacier_type
    d['terminus_type'] = gdir.terminus_type
    d['status'] = gdir.status

    # The rest is less certain. We put these in a try block and see
    # We're good with any error - we store the dict anyway below
    # TODO: should be done with more preselected errors
    try:
        # Inversion
        if gdir.has_file('inversion_output'):
            vol = []
            cl = gdir.read_pickle('inversion_output')
            for c in cl:
                vol.extend(c['volume'])
            d['inv_volume_km3'] = np.nansum(vol) * 1e-9
            area = gdir.rgi_area_km2
            d['inv_thickness_m'] = d['inv_volume_km3'] / area * 1000
            d['vas_volume_km3'] = 0.034 * (area ** 1.375)
            d['vas_thickness_m'] = d['vas_volume_km3'] / area * 1000
    except BaseException:
        pass

    try:
        # Calving
        all_calving_data = []
        all_width = []
        cl = gdir.read_pickle('calving_output')
        for c in cl:
            all_calving_data = c['calving_fluxes'][-1]
            all_width = c['t_width']
        d['calving_flux'] = all_calving_data
        d['calving_front_width'] = all_width
    except BaseException:
        pass

    if inversion_only:
        return d

    try:
        # Diagnostics
        diags = gdir.get_diagnostics()
        for k, v in diags.items():
            d[k] = v
    except BaseException:
        pass

    try:
        # Error log
        errlog = gdir.get_error_log()
        if errlog is not None:
            d['error_task'] = errlog.split(';')[-2]
            d['error_msg'] = errlog.split(';')[-1]
    except BaseException:
        pass

    try:
        # Masks related stuff
        fpath = gdir.get_filepath('gridded_data')
        with ncDataset(fpath) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo'][:]
        d['dem_mean_elev'] = np.mean(topo[np.where(mask == 1)])
        d['dem_med_elev'] = np.median(topo[np.where(mask == 1)])
        d['dem_min_elev'] = np.min(topo[np.where(mask == 1)])
        d['dem_max_elev'] = np.max(topo[np.where(mask == 1)])
    except BaseException:
        pass

    try:
        # Ext related stuff
        fpath = gdir.get_filepath('gridded_data')
        with ncDataset(fpath) as nc:
            ext = nc.variables['glacier_ext'][:]
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo'][:]
        d['dem_max_elev_on_ext'] = np.max(topo[np.where(ext == 1)])
        d['dem_min_elev_on_ext'] = np.min(topo[np.where(ext == 1)])
        a = np.sum(mask & (topo > d['dem_max_elev_on_ext']))
        d['dem_perc_area_above_max_elev_on_ext'] = a / np.sum(mask)
    except BaseException:
        pass

    try:
        # Centerlines
        cls = gdir.read_pickle('centerlines')
        longuest = 0.
        for cl in cls:
            longuest = np.max([longuest, cl.dis_on_line[-1]])
        d['n_centerlines'] = len(cls)
        d['longuest_centerline_km'] = longuest * gdir.grid.dx / 1000.
    except BaseException:
        pass

    try:
        # Flowline related stuff
        h = np.array([])
        widths = np.array([])
        slope = np.array([])
        fls = gdir.read_pickle('inversion_flowlines')
        dx = fls[0].dx * gdir.grid.dx
        for fl in fls:
            hgt = fl.surface_h
            h = np.append(h, hgt)
            widths = np.append(widths, fl.widths * dx)
            slope = np.append(slope, np.arctan(-np.gradient(hgt, dx)))
        d['flowline_mean_elev'] = np.average(h, weights=widths)
        d['flowline_max_elev'] = np.max(h)
        d['flowline_min_elev'] = np.min(h)
        d['flowline_avg_width'] = np.mean(widths)
        d['flowline_avg_slope'] = np.mean(slope)
    except BaseException:
        pass

    try:
        # MB calib
        df = gdir.read_json('local_mustar')
        d['t_star'] = df['t_star']
        d['mu_star_glacierwide'] = df['mu_star_glacierwide']
        d['mu_star_flowline_avg'] = df['mu_star_flowline_avg']
        d['mu_star_allsame'] = df['mu_star_allsame']
        d['mb_bias'] = df['bias']
    except BaseException:
        pass

    return d


def compile_glacier_statistics(gdirs, filesuffix='', path=True,
                               inversion_only=False):
    """Gather as much statistics as possible about a list of glaciers.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    inversion_only : bool
        if one wants to summarize the inversion output only (including calving)
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(glacier_statistics, gdirs,
                                 inversion_only=inversion_only)

    out = pd.DataFrame(out_df).set_index('rgi_id')
    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('glacier_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)
    return out


@entity_task(log)
def climate_statistics(gdir, add_climate_period=1995):
    """Gather as much statistics as possible about this glacier.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    add_climate_period : int or list of ints
        compile climate statistics for the 30 yrs period around the selected
        date.
    """

    from oggm.core.massbalance import (ConstantMassBalance,
                                       MultipleFlowlineMassBalance)

    d = OrderedDict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['name'] = gdir.name
    d['cenlon'] = gdir.cenlon
    d['cenlat'] = gdir.cenlat
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['glacier_type'] = gdir.glacier_type
    d['terminus_type'] = gdir.terminus_type
    d['status'] = gdir.status

    # The rest is less certain

    try:
        # Flowline related stuff
        h = np.array([])
        widths = np.array([])
        slope = np.array([])
        fls = gdir.read_pickle('inversion_flowlines')
        dx = fls[0].dx * gdir.grid.dx
        for fl in fls:
            hgt = fl.surface_h
            h = np.append(h, hgt)
            widths = np.append(widths, fl.widths * dx)
            slope = np.append(slope, np.arctan(-np.gradient(hgt, dx)))
        d['flowline_mean_elev'] = np.average(h, weights=widths)
        d['flowline_max_elev'] = np.max(h)
        d['flowline_min_elev'] = np.min(h)
    except BaseException:
        pass

    try:
        # Climate and MB at t*
        mbcl = ConstantMassBalance
        mbmod = MultipleFlowlineMassBalance(gdir, mb_model_class=mbcl,
                                            bias=0,
                                            use_inversion_flowlines=True)
        h, w, mbh = mbmod.get_annual_mb_on_flowlines()
        mbh = mbh * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
        pacc = np.where(mbh >= 0)
        pab = np.where(mbh < 0)
        d['tstar_aar'] = np.sum(w[pacc]) / np.sum(w)
        try:
            # Try to get the slope
            mb_slope, _, _, _, _ = stats.linregress(h[pab], mbh[pab])
            d['tstar_mb_grad'] = mb_slope
        except BaseException:
            # we don't mind if something goes wrong
            d['tstar_mb_grad'] = np.NaN
        d['tstar_ela_h'] = mbmod.get_ela()
        # Climate
        t, tm, p, ps = mbmod.flowline_mb_models[0].get_climate(
            [d['tstar_ela_h'],
             d['flowline_mean_elev'],
             d['flowline_max_elev'],
             d['flowline_min_elev']])
        for n, v in zip(['temp', 'tempmelt', 'prcpsol'], [t, tm, ps]):
            d['tstar_avg_' + n + '_ela_h'] = v[0]
            d['tstar_avg_' + n + '_mean_elev'] = v[1]
            d['tstar_avg_' + n + '_max_elev'] = v[2]
            d['tstar_avg_' + n + '_min_elev'] = v[3]
        d['tstar_avg_prcp'] = p[0]
    except BaseException:
        pass

    # Climate and MB at specified dates
    add_climate_period = tolist(add_climate_period)
    for y0 in add_climate_period:
        try:
            fs = '{}-{}'.format(y0 - 15, y0 + 15)

            mbcl = ConstantMassBalance
            mbmod = MultipleFlowlineMassBalance(gdir, mb_model_class=mbcl,
                                                y0=y0,
                                                use_inversion_flowlines=True)
            h, w, mbh = mbmod.get_annual_mb_on_flowlines()
            mbh = mbh * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
            pacc = np.where(mbh >= 0)
            pab = np.where(mbh < 0)
            d[fs + '_aar'] = np.sum(w[pacc]) / np.sum(w)
            try:
                # Try to get the slope
                mb_slope, _, _, _, _ = stats.linregress(h[pab], mbh[pab])
                d[fs + '_mb_grad'] = mb_slope
            except BaseException:
                # we don't mind if something goes wrong
                d[fs + '_mb_grad'] = np.NaN
            d[fs + '_ela_h'] = mbmod.get_ela()
            # Climate
            t, tm, p, ps = mbmod.flowline_mb_models[0].get_climate(
                [d[fs + '_ela_h'],
                 d['flowline_mean_elev'],
                 d['flowline_max_elev'],
                 d['flowline_min_elev']])
            for n, v in zip(['temp', 'tempmelt', 'prcpsol'], [t, tm, ps]):
                d[fs + '_avg_' + n + '_ela_h'] = v[0]
                d[fs + '_avg_' + n + '_mean_elev'] = v[1]
                d[fs + '_avg_' + n + '_max_elev'] = v[2]
                d[fs + '_avg_' + n + '_min_elev'] = v[3]
            d[fs + '_avg_prcp'] = p[0]
        except BaseException:
            pass

    return d


def compile_climate_statistics(gdirs, filesuffix='', path=True,
                               add_climate_period=1995):
    """Gather as much statistics as possible about a list of glaciers.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    gdirs: the list of GlacierDir to process.
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    add_climate_period : int or list of ints
        compile climate statistics for the 30 yrs period around the selected
        date.
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(climate_statistics, gdirs,
                                 add_climate_period=add_climate_period)

    out = pd.DataFrame(out_df).set_index('rgi_id')
    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('climate_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)
    return out


def idealized_gdir(surface_h, widths_m, map_dx, flowline_dx=1,
                   base_dir=None, reset=False):
    """Creates a glacier directory with flowline input data only.

    This is useful for testing, or for idealized experiments.

    Parameters
    ----------
    surface_h : ndarray
        the surface elevation of the flowline's grid points (in m).
    widths_m : ndarray
        the widths of the flowline's grid points (in m).
    map_dx : float
        the grid spacing (in m)
    flowline_dx : int
        the flowline grid spacing (in units of map_dx, often it should be 1)
    base_dir : str
        path to the directory where to open the directory.
        Defaults to `cfg.PATHS['working_dir'] + /per_glacier/`
    reset : bool, default=False
        empties the directory at construction

    Returns
    -------
    a GlacierDirectory instance
    """

    from oggm.core.centerlines import Centerline

    # Area from geometry
    area_km2 = np.sum(widths_m * map_dx * flowline_dx) * 1e-6

    # Dummy entity - should probably also change the geometry
    entity = gpd.read_file(get_demo_file('Hintereisferner_RGI5.shp')).iloc[0]
    entity.Area = area_km2
    entity.CenLat = 0
    entity.CenLon = 0
    entity.Name = ''
    entity.RGIId = 'RGI50-00.00000'
    entity.O1Region = '00'
    entity.O2Region = '0'
    gdir = GlacierDirectory(entity, base_dir=base_dir, reset=reset)
    gdir.write_shapefile(gpd.GeoDataFrame([entity]), 'outlines')

    # Idealized flowline
    coords = np.arange(0, len(surface_h) - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    fl = Centerline(line, dx=flowline_dx, surface_h=surface_h)
    fl.widths = widths_m / map_dx
    fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
    gdir.write_pickle([fl], 'inversion_flowlines')

    # Idealized map
    grid = salem.Grid(nxny=(1, 1), dxdy=(map_dx, map_dx), x0y0=(0, 0))
    grid.to_json(gdir.get_filepath('glacier_grid'))

    return gdir


def _robust_extract(to_dir, *args, **kwargs):
    """For some obscure reason this operation randomly fails.

    Try to make it more robust.
    """
    count = 0
    while count < 5:
        try:
            if count > 1:
                time.sleep(random.uniform(0.05, 0.1))
            with tarfile.open(*args, **kwargs) as tf:
                tf.extractall(os.path.dirname(to_dir))
            break
        except FileExistsError:
            count += 1
            if count == 5:
                raise


def robust_tar_extract(from_tar, to_dir, delete_tar=False):
    """Extract a tar file - also checks for a "tar in tar" situation"""

    if os.path.isfile(from_tar):
        _robust_extract(to_dir, from_tar, 'r')
    else:
        # maybe a tar in tar
        base_tar = os.path.dirname(from_tar) + '.tar'
        if not os.path.isfile(base_tar):
            raise FileNotFoundError('Could not find a tarfile with path: '
                                    '{}'.format(from_tar))
        if delete_tar:
            raise InvalidParamsError('Cannot delete tar in tar.')
        # Open the tar
        bname = os.path.basename(from_tar)
        dirbname = os.path.basename(os.path.dirname(from_tar))
        with tarfile.open(base_tar, 'r') as tf:
            i_from_tar = tf.getmember(os.path.join(dirbname, bname))
            _robust_extract(to_dir, fileobj=tf.extractfile(i_from_tar))

    if delete_tar:
        os.remove(from_tar)


class GlacierDirectory(object):
    """Organizes read and write access to the glacier's files.

    It handles a glacier directory created in a base directory (default
    is the "per_glacier" folder in the working directory). The role of a
    GlacierDirectory is to give access to file paths and to I/O operations.
    The user should not care about *where* the files are
    located, but should know their name (see :ref:`basenames`).

    If the directory does not exist, it will be created.

    See :ref:`glacierdir` for more information.

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
    cenlon, cenlat : float
        The glacier centerpoint's lon/lat
    rgi_date : int
        The RGI's BGNDATE year attribute if available. Otherwise, defaults to
        the median year for the RGI region
    rgi_region : str
        The RGI region name
    name : str
        The RGI glacier name (if Available)
    glacier_type : str
        The RGI glacier type ('Glacier', 'Ice cap', 'Perennial snowfield',
        'Seasonal snowfield')
    terminus_type : str
        The RGI terminus type ('Land-terminating', 'Marine-terminating',
        'Lake-terminating', 'Dry calving', 'Regenerated', 'Shelf-terminating')
    is_tidewater : bool
        Is the glacier a caving glacier?
    inversion_calving_rate : float
        Calving rate used for the inversion
    """

    def __init__(self, rgi_entity, base_dir=None, reset=False,
                 from_tar=False, delete_tar=False):
        """Creates a new directory or opens an existing one.

        Parameters
        ----------
        rgi_entity : a ``geopandas.GeoSeries`` or str
            glacier entity read from the shapefile (or a valid RGI ID if the
            directory exists)
        base_dir : str
            path to the directory where to open the directory.
            Defaults to `cfg.PATHS['working_dir'] + /per_glacier/`
        reset : bool, default=False
            empties the directory at construction (careful!)
        from_tar : str or bool, default=False
            path to a tar file to extract the gdir data from. If set to `True`,
            will check for a tar file at the expected location in `base_dir`.
        delete_tar : bool, default=False
            delete the original tar file after extraction.
        """

        if base_dir is None:
            if not cfg.PATHS.get('working_dir', None):
                raise ValueError("Need a valid PATHS['working_dir']!")
            base_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')

        # RGI IDs are also valid entries
        if isinstance(rgi_entity, str):
            # Get the meta from the shape file directly
            if from_tar:
                _dir = os.path.join(base_dir, rgi_entity[:8], rgi_entity[:11],
                                    rgi_entity)
                if from_tar is True:
                    from_tar = _dir + '.tar.gz'
                robust_tar_extract(from_tar, _dir, delete_tar=delete_tar)
                from_tar = False  # to not re-unpack later below
                _shp = os.path.join(_dir, 'outlines.shp')
            else:
                _shp = os.path.join(base_dir, rgi_entity[:8], rgi_entity[:11],
                                    rgi_entity, 'outlines.shp')
            rgi_entity = self._read_shapefile_from_path(_shp)
            crs = salem.check_crs(rgi_entity.crs)
            rgi_entity = rgi_entity.iloc[0]
            g = rgi_entity['geometry']
            xx, yy = salem.transform_proj(crs, salem.wgs84,
                                          [g.bounds[0], g.bounds[2]],
                                          [g.bounds[1], g.bounds[3]])
        else:
            g = rgi_entity['geometry']
            xx, yy = ([g.bounds[0], g.bounds[2]],
                      [g.bounds[1], g.bounds[3]])

        # Extent of the glacier in lon/lat
        self.extent_ll = [xx, yy]

        try:
            # RGI V4?
            rgi_entity.RGIID
            raise ValueError('RGI Version 4 is not supported anymore')
        except AttributeError:
            pass

        # Should be V5
        self.rgi_id = rgi_entity.RGIId
        self.glims_id = rgi_entity.GLIMSId
        self.cenlon = float(rgi_entity.CenLon)
        self.cenlat = float(rgi_entity.CenLat)
        self.rgi_region = '{:02d}'.format(int(rgi_entity.O1Region))
        self.rgi_subregion = (self.rgi_region + '-' +
                              '{:02d}'.format(int(rgi_entity.O2Region)))
        name = rgi_entity.Name
        rgi_datestr = rgi_entity.BgnDate

        try:
            gtype = rgi_entity.GlacType
        except AttributeError:
            # RGI V6
            gtype = [str(rgi_entity.Form), str(rgi_entity.TermType)]

        try:
            gstatus = rgi_entity.RGIFlag[0]
        except AttributeError:
            # RGI V6
            gstatus = rgi_entity.Status

        # rgi version can be useful
        self.rgi_version = self.rgi_id.split('-')[0][-2:]
        if self.rgi_version not in ['50', '60', '61']:
            raise RuntimeError('RGI Version not supported: '
                               '{}'.format(self.rgi_version))

        # remove spurious characters and trailing blanks
        self.name = filter_rgi_name(name)

        # region
        reg_names, subreg_names = parse_rgi_meta(version=self.rgi_version[0])
        n = reg_names.loc[int(self.rgi_region)].values[0]
        self.rgi_region_name = self.rgi_region + ': ' + n
        try:
            n = subreg_names.loc[self.rgi_subregion].values[0]
            self.rgi_subregion_name = self.rgi_subregion + ': ' + n
        except KeyError:
            self.rgi_subregion_name = self.rgi_subregion + ': NoName'

        # Read glacier attrs
        gtkeys = {'0': 'Glacier',
                  '1': 'Ice cap',
                  '2': 'Perennial snowfield',
                  '3': 'Seasonal snowfield',
                  '9': 'Not assigned',
                  }
        ttkeys = {'0': 'Land-terminating',
                  '1': 'Marine-terminating',
                  '2': 'Lake-terminating',
                  '3': 'Dry calving',
                  '4': 'Regenerated',
                  '5': 'Shelf-terminating',
                  '9': 'Not assigned',
                  }
        stkeys = {'0': 'Glacier or ice cap',
                  '1': 'Glacier complex',
                  '2': 'Nominal glacier',
                  '9': 'Not assigned',
                  }
        self.glacier_type = gtkeys[gtype[0]]
        self.terminus_type = ttkeys[gtype[1]]
        self.status = stkeys['{}'.format(gstatus)]
        self.is_tidewater = self.terminus_type in ['Marine-terminating',
                                                   'Lake-terminating']
        self.is_nominal = self.status == 'Nominal glacier'
        self.inversion_calving_rate = 0.
        self.is_icecap = self.glacier_type == 'Ice cap'

        # Hemisphere
        self.hemisphere = 'sh' if self.cenlat < 0 else 'nh'

        # convert the date
        rgi_date = int(rgi_datestr[0:4])
        if rgi_date < 0:
            rgi_date = RGI_DATE[self.rgi_region]
        self.rgi_date = rgi_date

        # Root directory
        self.base_dir = os.path.normpath(base_dir)
        self.dir = os.path.join(self.base_dir, self.rgi_id[:8],
                                self.rgi_id[:11], self.rgi_id)

        # Do we have to extract the files first?
        if (reset or from_tar) and os.path.exists(self.dir):
            shutil.rmtree(self.dir)

        if from_tar:
            if from_tar is True:
                from_tar = self.dir + '.tar.gz'
            robust_tar_extract(from_tar, self.dir, delete_tar=delete_tar)
        else:
            mkdir(self.dir)

        if not os.path.isdir(self.dir):
            raise RuntimeError('GlacierDirectory %s does not exist!' % self.dir)

        # logging file
        self.logfile = os.path.join(self.dir, 'log.txt')

        # Optimization
        self._mbdf = None
        self._mbprofdf = None

    def __repr__(self):

        summary = ['<oggm.GlacierDirectory>']
        summary += ['  RGI id: ' + self.rgi_id]
        summary += ['  Region: ' + self.rgi_region_name]
        summary += ['  Subregion: ' + self.rgi_subregion_name]
        if self.name:
            summary += ['  Name: ' + self.name]
        summary += ['  Glacier type: ' + str(self.glacier_type)]
        summary += ['  Terminus type: ' + str(self.terminus_type)]
        summary += ['  Area: ' + str(self.rgi_area_km2) + ' km2']
        summary += ['  Lon, Lat: (' + str(self.cenlon) + ', ' +
                    str(self.cenlat) + ')']
        if os.path.isfile(self.get_filepath('glacier_grid')):
            summary += ['  Grid (nx, ny): (' + str(self.grid.nx) + ', ' +
                        str(self.grid.ny) + ')']
            summary += ['  Grid (dx, dy): (' + str(self.grid.dx) + ', ' +
                        str(self.grid.dy) + ')']
        return '\n'.join(summary) + '\n'

    @lazy_property
    def grid(self):
        """A ``salem.Grid`` handling the georeferencing of the local grid"""
        return salem.Grid.from_json(self.get_filepath('glacier_grid'))

    @lazy_property
    def rgi_area_km2(self):
        """The glacier's RGI area (km2)."""
        try:
            _area = self.read_shapefile('outlines')['Area']
            return np.round(float(_area), decimals=3)
        except OSError:
            raise RuntimeError('Please run `define_glacier_region` before '
                               'using this property.')

    @property
    def rgi_area_m2(self):
        """The glacier's RGI area (m2)."""
        return self.rgi_area_km2 * 10**6

    def get_filepath(self, filename, delete=False, filesuffix=''):
        """Absolute path to a specific file.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        delete : bool
            delete the file if exists
        filesuffix : str
            append a suffix to the filename (useful for model runs). Note
            that the BASENAME remains same.

        Returns
        -------
        The absolute path to the desired file
        """

        if filename not in cfg.BASENAMES:
            raise ValueError(filename + ' not in cfg.BASENAMES.')

        fname = cfg.BASENAMES[filename]
        if filesuffix:
            fname = fname.split('.')
            assert len(fname) == 2
            fname = fname[0] + filesuffix + '.' + fname[1]

        out = os.path.join(self.dir, fname)
        if delete and os.path.isfile(out):
            os.remove(out)
        return out

    def has_file(self, filename):
        """Checks if a file exists.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        """
        fp = self.get_filepath(filename)
        if '.shp' in fp and cfg.PARAMS['use_tar_shapefiles']:
            fp = fp.replace('.shp', '.tar')
            if cfg.PARAMS['use_compression']:
                fp += '.gz'
        return os.path.exists(fp)

    def add_to_diagnostics(self, key, value):
        """Write a key, value pair to the gdir's runtime diagnostics.

        Parameters
        ----------
        key : str
            dict entry key
        value : str or number
            dict entry value
        """

        d = self.get_diagnostics()
        d[key] = value
        with open(self.get_filepath('diagnostics'), 'w') as f:
            json.dump(d, f)

    def get_diagnostics(self):
        """Read the gdir's runtime diagnostics.

        Returns
        -------
        the diagnostics dict
        """
        # If not there, create an empty one
        if not self.has_file('diagnostics'):
            with open(self.get_filepath('diagnostics'), 'w') as f:
                json.dump(dict(), f)

        # Read and return
        with open(self.get_filepath('diagnostics'), 'r') as f:
            out = json.load(f)
        return out

    def read_pickle(self, filename, use_compression=None, filesuffix=''):
        """Reads a pickle located in the directory.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        use_compression : bool
            whether or not the file ws compressed. Default is to use
            cfg.PARAMS['use_compression'] for this (recommended)
        filesuffix : str
            append a suffix to the filename (useful for experiments).

        Returns
        -------
        An object read from the pickle
        """
        use_comp = (use_compression if use_compression is not None
                    else cfg.PARAMS['use_compression'])
        _open = gzip.open if use_comp else open
        fp = self.get_filepath(filename, filesuffix=filesuffix)
        with _open(fp, 'rb') as f:
            out = pickle.load(f)

        return out

    def write_pickle(self, var, filename, use_compression=None, filesuffix=''):
        """ Writes a variable to a pickle on disk.

        Parameters
        ----------
        var : object
            the variable to write to disk
        filename : str
            file name (must be listed in cfg.BASENAME)
        use_compression : bool
            whether or not the file ws compressed. Default is to use
            cfg.PARAMS['use_compression'] for this (recommended)
        filesuffix : str
            append a suffix to the filename (useful for experiments).
        """
        use_comp = (use_compression if use_compression is not None
                    else cfg.PARAMS['use_compression'])
        _open = gzip.open if use_comp else open
        fp = self.get_filepath(filename, filesuffix=filesuffix)
        with _open(fp, 'wb') as f:
            pickle.dump(var, f, protocol=-1)

    def read_json(self, filename, filesuffix=''):
        """Reads a JSON file located in the directory.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        filesuffix : str
            append a suffix to the filename (useful for experiments).

        Returns
        -------
        A dictionary read from the JSON file
        """

        fp = self.get_filepath(filename, filesuffix=filesuffix)
        with open(fp, 'r') as f:
            out = json.load(f)
        return out

    def write_json(self, var, filename, filesuffix=''):
        """ Writes a variable to a pickle on disk.

        Parameters
        ----------
        var : object
            the variable to write to JSON (must be a dictionary)
        filename : str
            file name (must be listed in cfg.BASENAME)
        filesuffix : str
            append a suffix to the filename (useful for experiments).
        """
        fp = self.get_filepath(filename, filesuffix=filesuffix)
        with open(fp, 'w') as f:
            json.dump(var, f)

    def read_text(self, filename, filesuffix=''):
        """Reads a text file located in the directory.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        filesuffix : str
            append a suffix to the filename (useful for experiments).

        Returns
        -------
        the text
        """

        fp = self.get_filepath(filename, filesuffix=filesuffix)
        with open(fp, 'r') as f:
            out = f.read()
        return out

    @classmethod
    def _read_shapefile_from_path(cls, fp):
        if '.shp' not in fp:
            raise ValueError('File ending not that of a shapefile')

        if cfg.PARAMS['use_tar_shapefiles']:
            fp = 'tar://' + fp.replace('.shp', '.tar')
            if cfg.PARAMS['use_compression']:
                fp += '.gz'

        return gpd.read_file(fp)

    def read_shapefile(self, filename, filesuffix=''):
        """Reads a shapefile located in the directory.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        filesuffix : str
            append a suffix to the filename (useful for experiments).

        Returns
        -------
        A geopandas.DataFrame
        """
        fp = self.get_filepath(filename, filesuffix=filesuffix)
        return self._read_shapefile_from_path(fp)

    def write_shapefile(self, var, filename, filesuffix=''):
        """ Writes a variable to a shapefile on disk.

        Parameters
        ----------
        var : object
            the variable to write to shapefile (must be a geopandas.DataFrame)
        filename : str
            file name (must be listed in cfg.BASENAME)
        filesuffix : str
            append a suffix to the filename (useful for experiments).
        """
        fp = self.get_filepath(filename, filesuffix=filesuffix)
        if '.shp' not in fp:
            raise ValueError('File ending not that of a shapefile')
        var.to_file(fp)

        if not cfg.PARAMS['use_tar_shapefiles']:
            # Done here
            return

        # Write them in tar
        fp = fp.replace('.shp', '.tar')
        mode = 'w'
        if cfg.PARAMS['use_compression']:
            fp += '.gz'
            mode += ':gz'
        if os.path.exists(fp):
            os.remove(fp)

        # List all files that were written as shape
        fs = glob.glob(fp.replace('.gz', '').replace('.tar', '.*'))
        # Add them to tar
        with tarfile.open(fp, mode=mode) as tf:
            for ff in fs:
                tf.add(ff, arcname=os.path.basename(ff))

        # Delete the old ones
        for ff in fs:
            os.remove(ff)

    def create_gridded_ncdf_file(self, fname):
        """Makes a gridded netcdf file template.

        The other variables have to be created and filled by the calling
        routine.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)

        Returns
        -------
        a ``netCDF4.Dataset`` object.
        """

        # overwrite as default
        fpath = self.get_filepath(fname)
        if os.path.exists(fpath):
            os.remove(fpath)

        nc = ncDataset(fpath, 'w', format='NETCDF4')

        nc.createDimension('x', self.grid.nx)
        nc.createDimension('y', self.grid.ny)

        nc.author = 'OGGM'
        nc.author_info = 'Open Global Glacier Model'
        nc.proj_srs = self.grid.proj.srs

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

        return nc

    def write_monthly_climate_file(self, time, prcp, temp,
                                   ref_pix_hgt, ref_pix_lon, ref_pix_lat, *,
                                   gradient=None,
                                   time_unit=None,
                                   calendar=None,
                                   file_name='climate_monthly',
                                   filesuffix=''):
        """Creates a netCDF4 file with climate data timeseries.

        Parameters
        ----------
        time : ndarray
            the time array, in a format understood by netCDF4
        prcp : ndarray
            the precipitation array (unit: 'kg m-2 month-1')
        temp : ndarray
            the temperature array (unit: 'degC')
        ref_pix_hgt : float
            the elevation of the dataset's reference altitude
            (for correction). In practice it is the same altitude as the
            baseline climate.
        ref_pix_lon : float
            the location of the gridded data's grid point
        ref_pix_lat : float
            the location of the gridded data's grid point
        gradient : ndarray, optional
            whether to use a time varying gradient
        time_unit : str
            the reference time unit for your time array. This should be chosen
            depending on the length of your data. The default is to choose
            it ourselves based on the starting year.
        calendar : str
            If you use an exotic calendar (e.g. 'noleap')
        file_name : str
            How to name the file
        filesuffix : str
            Apply a suffix to the file
        """

        # overwrite as default
        fpath = self.get_filepath(file_name, filesuffix=filesuffix)
        if os.path.exists(fpath):
            os.remove(fpath)

        zlib = cfg.PARAMS['compress_climate_netcdf']
        
        if time_unit is None:
            # http://pandas.pydata.org/pandas-docs/stable/timeseries.html
            # #timestamp-limitations
            if time[0].year > 1800:
                time_unit = 'days since 1801-01-01 00:00:00'
            elif time[0].year >= 0:
                time_unit = ('days since {:04d}-01-01 '
                             '00:00:00'.format(time[0].year))
            else:
                raise InvalidParamsError('Time format not supported')

        with ncDataset(fpath, 'w', format='NETCDF4') as nc:
            nc.ref_hgt = ref_pix_hgt
            nc.ref_pix_lon = ref_pix_lon
            nc.ref_pix_lat = ref_pix_lat
            nc.ref_pix_dis = haversine(self.cenlon, self.cenlat,
                                       ref_pix_lon, ref_pix_lat)

            nc.createDimension('time', None)

            nc.author = 'OGGM'
            nc.author_info = 'Open Global Glacier Model'

            timev = nc.createVariable('time', 'i4', ('time',))
            
            tatts = {'units': time_unit}
            if calendar is None:
                calendar = 'standard'

            tatts['calendar'] = calendar
            try:
                numdate = netCDF4.date2num([t for t in time], time_unit,
                                           calendar=calendar)
            except TypeError:
                # numpy's broken datetime only works for us precision
                time = time.astype('M8[us]').astype(datetime.datetime)
                numdate = netCDF4.date2num(time, time_unit, calendar=calendar)

            timev.setncatts(tatts)
            timev[:] = numdate

            v = nc.createVariable('prcp', 'f4', ('time',), zlib=zlib)
            v.units = 'kg m-2'
            v.long_name = 'total monthly precipitation amount'
            v[:] = prcp

            v = nc.createVariable('temp', 'f4', ('time',), zlib=zlib)
            v.units = 'degC'
            v.long_name = '2m temperature at height ref_hgt'
            v[:] = temp

            if gradient is not None:
                v = nc.createVariable('gradient', 'f4', ('time',), zlib=zlib)
                v.units = 'degC m-1'
                v.long_name = 'temperature gradient from local regression'
                v[:] = gradient

    def get_inversion_flowline_hw(self):
        """ Shortcut function to read the heights and widths of the glacier.

        Parameters
        ----------

        Returns
        -------
        (height, widths) in units of m
        """

        h = np.array([])
        w = np.array([])
        fls = self.read_pickle('inversion_flowlines')
        for fl in fls:
            w = np.append(w, fl.widths)
            h = np.append(h, fl.surface_h)
        return h, w * self.grid.dx

    def get_ref_mb_data(self):
        """Get the reference mb data from WGMS (for some glaciers only!).

        Raises an Error if it isn't a reference glacier at all.
        """

        if self._mbdf is None:
            flink, mbdatadir = get_wgms_files()
            c = 'RGI{}0_ID'.format(self.rgi_version[0])
            wid = flink.loc[flink[c] == self.rgi_id]
            if len(wid) == 0:
                raise RuntimeError('Not a reference glacier!')
            wid = wid.WGMS_ID.values[0]

            # file
            reff = os.path.join(mbdatadir,
                                'mbdata_WGMS-{:05d}.csv'.format(wid))
            # list of years
            self._mbdf = pd.read_csv(reff).set_index('YEAR')

        # logic for period
        if not self.has_file('climate_info'):
            raise RuntimeError('Please process some climate data before call')
        ci = self.read_pickle('climate_info')
        y0 = ci['baseline_hydro_yr_0']
        y1 = ci['baseline_hydro_yr_1']
        if len(self._mbdf) > 1:
            out = self._mbdf.loc[y0:y1]
        else:
            # Some files are just empty
            out = self._mbdf
        return out.dropna(subset=['ANNUAL_BALANCE'])

    def get_ref_mb_profile(self):
        """Get the reference mb profile data from WGMS (if available!).

        Returns None if this glacier has no profile and an Error if it isn't
        a reference glacier at all.
        """

        if self._mbprofdf is None:
            flink, mbdatadir = get_wgms_files()
            c = 'RGI{}0_ID'.format(self.rgi_version[0])
            wid = flink.loc[flink[c] == self.rgi_id]
            if len(wid) == 0:
                raise RuntimeError('Not a reference glacier!')
            wid = wid.WGMS_ID.values[0]

            # file
            mbdatadir = os.path.join(os.path.dirname(mbdatadir), 'mb_profiles')
            reff = os.path.join(mbdatadir,
                                'profile_WGMS-{:05d}.csv'.format(wid))
            if not os.path.exists(reff):
                return None
            # list of years
            self._mbprofdf = pd.read_csv(reff, index_col=0)

        # logic for period
        if not self.has_file('climate_info'):
            raise RuntimeError('Please process some climate data before call')
        ci = self.read_pickle('climate_info')
        y0 = ci['baseline_hydro_yr_0']
        y1 = ci['baseline_hydro_yr_1']
        if len(self._mbprofdf) > 1:
            out = self._mbprofdf.loc[y0:y1]
        else:
            # Some files are just empty
            out = self._mbprofdf
        out.columns = [float(c) for c in out.columns]
        return out.dropna(axis=1, how='all').dropna(axis=0, how='all')

    def get_ref_length_data(self):
        """Get the glacier lenght data from P. Leclercq's data base.

         https://folk.uio.no/paulwl/data.php

         For some glaciers only!
         """

        df = pd.read_csv(get_demo_file('rgi_leclercq_links_2012_RGIV5.csv'))
        df = df.loc[df.RGI_ID == self.rgi_id]
        if len(df) == 0:
            raise RuntimeError('No length data found for this glacier!')
        ide = df.LID.values[0]

        f = get_demo_file('Glacier_Lengths_Leclercq.nc')
        with xr.open_dataset(f) as dsg:
            # The database is not sorted by ID. Don't ask me...
            grp_id = np.argwhere(dsg['index'].values == ide)[0][0] + 1
        with xr.open_dataset(f, group=str(grp_id)) as ds:
            df = ds.to_dataframe()
            df.name = ds.glacier_name
        return df

    def log(self, task_name, err=None):
        """Logs a message to the glacier directory.

        It is usually called by the :py:class:`entity_task` decorator, normally
        you shouldn't take care about that.

        Parameters
        ----------
        func : a function
            the function which wants to log
        err : Exception
            the exception which has been raised by func (if no exception was
            raised, a success is logged)
        """

        # a line per function call
        nowsrt = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        line = nowsrt + ';' + task_name + ';'
        if err is None:
            line += 'SUCCESS'
        else:
            line += err.__class__.__name__ + ': {}'.format(err)

        count = 0
        while count < 5:
            try:
                with open(self.logfile, 'a') as logfile:
                    logfile.write(line + '\n')
                break
            except FileNotFoundError:
                # I really don't know when this error happens
                # In this case sleep and try again
                time.sleep(0.05)
                count += 1

        if count == 5:
            log.warning('Could not write to logfile: ' + line)

    def get_task_status(self, task_name):
        """Opens this directory's log file to see if a task was already run.

        It is usually called by the :py:class:`entity_task` decorator, normally
        you shouldn't take care about that.

        Parameters
        ----------
        task_name : str
            the name of the task which has to be tested for

        Returns
        -------
        The last message for this task (SUCCESS if was successful),
        None if the task was not run yet
        """

        if not os.path.isfile(self.logfile):
            return None

        with open(self.logfile) as logfile:
            lines = logfile.readlines()

        lines = [l.replace('\n', '') for l in lines if task_name in l]
        if lines:
            # keep only the last log
            return lines[-1].split(';')[-1]
        else:
            return None

    def get_error_log(self):
        """Reads the directory's log file to find the invalid task (if any).

        Returns
        -------
        The first error message in this log, None if all good
        """

        if not os.path.isfile(self.logfile):
            return None

        with open(self.logfile) as logfile:
            lines = logfile.readlines()

        for l in lines:
            if 'SUCCESS' in l:
                continue
            return l.replace('\n', '')

        # OK all good
        return None


@entity_task(log)
def copy_to_basedir(gdir, base_dir=None, setup='run'):
    """Copies the glacier directories and their content to a new location.

    This utility function allows to select certain files only, thus
    saving time at copy.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to copy
    base_dir : str
        path to the new base directory (should end with "per_glacier" most
        of the times)
    setup : str
        set up you want the copied directory to be useful for. Currently
        supported are 'all' (copy the entire directory), 'inversion'
        (copy the necessary files for the inversion AND the run)
        and 'run' (copy the necessary files for a dynamical run).

    Returns
    -------
    New glacier directories from the copied folders
    """
    base_dir = os.path.abspath(base_dir)
    new_dir = os.path.join(base_dir, gdir.rgi_id[:8], gdir.rgi_id[:11],
                           gdir.rgi_id)
    if setup == 'run':
        paths = ['model_flowlines', 'inversion_params', 'outlines',
                 'local_mustar', 'climate_monthly',
                 'gcm_data', 'climate_info']
        paths = ('*' + p + '*' for p in paths)
        shutil.copytree(gdir.dir, new_dir,
                        ignore=include_patterns(*paths))
    elif setup == 'inversion':
        paths = ['inversion_params', 'downstream_line', 'outlines',
                 'inversion_flowlines', 'glacier_grid',
                 'local_mustar', 'climate_monthly', 'gridded_data',
                 'gcm_data', 'climate_info']
        paths = ('*' + p + '*' for p in paths)
        shutil.copytree(gdir.dir, new_dir,
                        ignore=include_patterns(*paths))
    elif setup == 'all':
        shutil.copytree(gdir.dir, new_dir)
    else:
        raise ValueError('setup not understood: {}'.format(setup))
    return GlacierDirectory(gdir.rgi_id, base_dir=base_dir)


def initialize_merged_gdir(main, tribs=[], glcdf=None,
                           filename='climate_monthly', input_filesuffix=''):
    """Creats a new GlacierDirectory if tributaries are merged to a glacier

    This function should be called after centerlines.intersect_downstream_lines
    and before flowline.merge_tributary_flowlines.
    It will create a new GlacierDirectory, with a suitable DEM and reproject
    the flowlines of the main glacier.

    Parameters
    ----------
    main : oggm.GlacierDirectory
        the main glacier
    tribs : list or dictionary containing oggm.GlacierDirectories
        true tributary glaciers to the main glacier
    glcdf: geopandas.GeoDataFrame
        which contains the main glacier, will be downloaded if None
    filename: str
        Baseline climate file
    input_filesuffix: str
        Filesuffix to the climate file
    Returns
    -------
    merged : oggm.GlacierDirectory
        the new GDir
    """
    from oggm.core.gis import define_glacier_region, merged_glacier_masks

    # If its a dict, select the relevant ones
    if isinstance(tribs, dict):
        tribs = tribs[main.rgi_id]
    # make sure tributaries are iteratable
    tribs = tolist(tribs)

    # read flowlines of the Main glacier
    mfls = main.read_pickle('model_flowlines')

    # ------------------------------
    # 0. create the new GlacierDirectory from main glaciers GeoDataFrame
    # Should be passed along, if not download it
    if glcdf is None:
        glcdf = get_rgi_glacier_entities([main.rgi_id])
    # Get index location of the specific glacier
    idx = glcdf.loc[glcdf.RGIId == main.rgi_id].index
    maindf = glcdf.loc[idx].copy()

    # add tributary geometries to maindf
    merged_geometry = maindf.loc[idx, 'geometry'].iloc[0]
    for trib in tribs:
        geom = trib.read_pickle('geometries')['polygon_hr']
        geom = salem.transform_geometry(geom, crs=trib.grid)
        merged_geometry = merged_geometry.union(geom)

    # to get the center point, maximal extensions for DEM and single Polygon:
    maindf.loc[idx, 'geometry'] = merged_geometry.convex_hull

    # make some adjustments to the rgi dataframe
    # 1. update central point of new glacier
    # Could also use the mean of the Flowline centroids, to shift it more
    # downstream. But this is more straight forward.
    maindf.loc[idx, 'CenLon'] = maindf.loc[idx, 'geometry'].centroid.x
    maindf.loc[idx, 'CenLat'] = maindf.loc[idx, 'geometry'].centroid.y
    # 2. update names
    maindf.loc[idx, 'RGIId'] += '_merged'
    if maindf.loc[idx, 'Name'].iloc[0] is None:
        maindf.loc[idx, 'Name'] = main.name + ' (merged)'
    else:
        maindf.loc[idx, 'Name'] += ' (merged)'

    # finally create new Glacier Directory
    # 1. set dx spacing to the one used for the flowlines
    dx_method = cfg.PARAMS['grid_dx_method']
    dx_spacing = cfg.PARAMS['fixed_dx']
    cfg.PARAMS['grid_dx_method'] = 'fixed'
    cfg.PARAMS['fixed_dx'] = mfls[-1].map_dx
    merged = GlacierDirectory(maindf.loc[idx].iloc[0])

    # run define_glacier_region to get a fitting DEM and proper grid
    define_glacier_region(merged, entity=maindf.loc[idx].iloc[0])

    # write gridded data and geometries for visualization
    merged_glacier_masks(merged, merged_geometry)

    # reset dx method
    cfg.PARAMS['grid_dx_method'] = dx_method
    cfg.PARAMS['fixed_dx'] = dx_spacing

    # copy main climate file, climate info and local_mustar to new gdir
    climfilename = filename + '_' + main.rgi_id + input_filesuffix + '.nc'
    climfile = os.path.join(merged.dir, climfilename)
    shutil.copyfile(main.get_filepath(filename, filesuffix=input_filesuffix),
                    climfile)
    _mufile = os.path.basename(merged.get_filepath('local_mustar')).split('.')
    mufile = _mufile[0] + '_' + main.rgi_id + '.' + _mufile[1]
    shutil.copyfile(main.get_filepath('local_mustar'),
                    os.path.join(merged.dir, mufile))
    # I think I need the climate_info only for the main glacier
    shutil.copyfile(main.get_filepath('climate_info'),
                    merged.get_filepath('climate_info'))

    # reproject the flowlines to the new grid
    for nr, fl in reversed(list(enumerate(mfls))):

        # 1. Step: Change projection to the main glaciers grid
        _line = salem.transform_geometry(fl.line,
                                         crs=main.grid, to_crs=merged.grid)
        # 2. set new line
        fl.set_line(_line)

        # 3. set flow to attributes
        if fl.flows_to is not None:
            fl.set_flows_to(fl.flows_to)
        # remove inflow points, will be set by other flowlines if need be
        fl.inflow_points = []

        # 5. set grid size attributes
        dx = [shpg.Point(fl.line.coords[i]).distance(
            shpg.Point(fl.line.coords[i+1]))
            for i, pt in enumerate(fl.line.coords[:-1])]  # get distance
        # and check if equally spaced
        if not np.allclose(dx, np.mean(dx), atol=1e-2):
            raise RuntimeError('Flowline is not evenly spaced.')
        # dx might very slightly change, but should not
        fl.dx = np.mean(dx).round(2)
        # map_dx should stay exactly the same
        # fl.map_dx = mfls[-1].map_dx
        fl.dx_meter = fl.map_dx * fl.dx

        # replace flowline within the list
        mfls[nr] = fl

    # Write the reprojecflowlines
    merged.write_pickle(mfls, 'model_flowlines')

    return merged


@entity_task(log)
def gdir_to_tar(gdir, base_dir=None, delete=True):
    """Writes the content of a glacier directory to a tar file.

    The tar file is located at the same location of the original directory.
    The glacier directory objects are useless if deleted!

    Parameters
    ----------
    base_dir : str
        path to the basedir where to write the directory (defaults to the
        same location of the original directory)
    delete : bool
        delete the original directory afterwards (default)

    Returns
    -------
    the path to the tar file
    """

    source_dir = os.path.normpath(gdir.dir)
    opath = source_dir + '.tar.gz'
    if base_dir is not None:
        opath = os.path.join(base_dir, os.path.relpath(opath, gdir.base_dir))
        mkdir(os.path.dirname(opath))

    with tarfile.open(opath, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    if delete:
        shutil.rmtree(source_dir)

    return opath


def base_dir_to_tar(base_dir=None, delete=True):
    """Merge the directories into 1000 bundles as tar files.

    The tar file is located at the same location of the original directory.

    Parameters
    ----------
    base_dir : str
        path to the basedir to parse (defaults to the working directory)
    to_base_dir : str
        path to the basedir where to write the directory (defaults to the
        same location of the original directory)
    delete : bool
        delete the original directory tars afterwards (default)
    """

    if base_dir is None:
        if not cfg.PATHS.get('working_dir', None):
            raise ValueError("Need a valid PATHS['working_dir']!")
        base_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')

    for dirname, subdirlist, filelist in os.walk(base_dir):
        # RGI60-01.00
        bname = os.path.basename(dirname)
        if not (len(bname) == 11 and bname[-3] == '.'):
            continue
        opath = dirname + '.tar'
        with tarfile.open(opath, 'w') as tar:
            tar.add(dirname, arcname=os.path.basename(dirname))
        if delete:
            shutil.rmtree(dirname)
