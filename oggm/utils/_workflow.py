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
import signal
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
import pandas as pd
import numpy as np
from scipy import stats
import xarray as xr
import shapely.geometry as shpg
from shapely.ops import transform as shp_trafo
import netCDF4

# Optional libs
try:
    import geopandas as gpd
except ImportError:
    pass
try:
    import salem
except ImportError:
    pass
try:
    from salem import wgs84
    from salem.gis import transform_proj
except ImportError:
    pass
try:
    import pyproj
except ImportError:
    pass


# Locals
from oggm import __version__
from oggm.utils._funcs import (calendardate_to_hydrodate, date_to_floatyear,
                               tolist, filter_rgi_name, parse_rgi_meta,
                               haversine, multipolygon_to_polygon)
from oggm.utils._downloads import (get_demo_file, get_wgms_files,
                                   get_rgi_glacier_entities)
from oggm import cfg
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError


# Default RGI date (median per region in RGI6)
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


# alias
get_temp_dir = gettempdir


def get_sys_info():
    """Returns system information as a list of tuples"""

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


def get_env_info():
    """Returns env information as a list of tuples"""

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
        ("pyproj", lambda mod: mod.__version__),
        ("shapely", lambda mod: mod.__version__),
        ("xarray", lambda mod: mod.__version__),
        ("dask", lambda mod: mod.__version__),
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

    return deps_blob


def get_git_ident():
    ident_str = '$Id$'
    if ":" not in ident_str:
        return 'no_git_id'
    return ident_str.replace("$", "").replace("Id:", "").replace(" ", "")


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

    sys_info = get_sys_info()
    deps_blob = get_env_info()

    out = ['# OGGM environment: ']
    out.append("## System info:")
    for k, stat in sys_info:
        out.append("    %s: %s" % (k, stat))
    out.append("## Packages info:")
    for k, stat in deps_blob:
        out.append("    %s: %s" % (k, stat))
    out.append("    OGGM git identifier: " + get_git_ident())

    if logger is not None:
        logger.workflow('\n'.join(out))

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

    def __init__(self, l0=None, maxsize=None):
        """Instanciate.

        Parameters
        ----------
        l0 : list
            a list of file paths
        maxsize : int
            the max number of files to keep
        """
        self.files = [] if l0 is None else l0
        # if no maxsize is specified, use value from configuration
        maxsize = cfg.PARAMS['lru_maxsize'] if maxsize is None else maxsize
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


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""

    attr_name = '_lazy_' + fn.__name__

    @property
    @wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


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
        # deleting stuff takes time
        while os.path.exists(path):  # check if it still exists
            pass
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


def _timeout_handler(signum, frame):
    raise TimeoutError('This task was killed because of timeout')


class entity_task(object):
    """Decorator for common job-controlling logic.

    All tasks share common operations. This decorator is here to handle them:
    exceptions, logging, and (some day) database for job-controlling.
    """

    def __init__(self, log, writes=[], fallback=None):
        """Decorator syntax: ``@oggm_task(log, writes=['dem', 'outlines'])``

        Parameters
        ----------
        log: logger
            module logger
        writes: list
            list of files that the task will write down to disk (must be
            available in ``cfg.BASENAMES``)
        fallback: python function
            will be executed on gdir if entity_task fails
        return_value: bool
            whether the return value from the task should be passed over
            to the caller or not. In general you will always want this to
            be true, but sometimes the task return things which are not
            useful in production and my use a lot of memory, etc,
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
        def _entity_task(gdir, *, reset=None, print_log=True,
                         return_value=True, continue_on_error=None,
                         add_to_log_file=True, **kwargs):

            if reset is None:
                reset = not cfg.PARAMS['auto_skip_task']

            if continue_on_error is None:
                continue_on_error = cfg.PARAMS['continue_on_error']

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
                if cfg.PARAMS['task_timeout'] > 0:
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(cfg.PARAMS['task_timeout'])
                ex_t = time.time()
                out = task_func(gdir, **kwargs)
                ex_t = time.time() - ex_t
                if cfg.PARAMS['task_timeout'] > 0:
                    signal.alarm(0)
                if task_name != 'gdir_to_tar':
                    gdir.log(task_name, task_time=ex_t)
            except Exception as err:
                # Something happened
                out = None
                if add_to_log_file:
                    gdir.log(task_name, err=err)
                    pipe_log(gdir, task_name, err=err)
                if print_log:
                    self.log.error('%s occurred during task %s on %s: %s',
                                   type(err).__name__, task_name,
                                   gdir.rgi_id, str(err))
                if not continue_on_error:
                    raise

                if self.fallback is not None:
                    self.fallback(gdir)
            if return_value:
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
        gs = dict()
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
        try:
            olist.extend(_get_centerline_lonlat(gdir))
        except FileNotFoundError:
            pass

    odf = gpd.GeoDataFrame(olist)
    odf = odf.sort_values(by='RGIID')
    odf.crs = {'init': 'epsg:4326'}
    odf.to_file(path)


def demo_glacier_id(key):
    """Get the RGI id of a glacier by name or key: None if not found."""

    df = cfg.DATA['demo_glaciers']

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


class compile_to_netcdf(object):
    """Decorator for common compiling NetCDF files logic.

    All compile_* tasks can be optimized the same way, by using temporary
    files and merging them afterwards.
    """

    def __init__(self, log):
        """Decorator syntax: ``@compile_to_netcdf(log, n_tmp_files=1000)``

        Parameters
        ----------
        log: logger
            module logger
        tmp_file_size: int
            number of glacier directories per temporary files
        """
        self.log = log

    def __call__(self, task_func):
        """Decorate."""

        @wraps(task_func)
        def _compile_to_netcdf(gdirs, filesuffix='', input_filesuffix='',
                               output_filesuffix='', path=True,
                               tmp_file_size=1000,
                               **kwargs):

            # Check input
            if filesuffix:
                warnings.warn('The `filesuffix` kwarg is deprecated for '
                              'compile_* tasks. Use input_filesuffix from '
                              'now on.',
                              DeprecationWarning)
                input_filesuffix = filesuffix

            if not output_filesuffix:
                output_filesuffix = input_filesuffix

            gdirs = tolist(gdirs)

            hemisphere = [gd.hemisphere for gd in gdirs]
            if len(np.unique(hemisphere)) == 2:
                if path is not True:
                    raise InvalidParamsError('With glaciers from both '
                                             'hemispheres, set `path=True`.')
                self.log.workflow('compile_*: you gave me a list of gdirs from '
                                  'both hemispheres. I am going to write two '
                                  'files out of it with _sh and _nh suffixes.')
                _gdirs = [gd for gd in gdirs if gd.hemisphere == 'sh']
                _compile_to_netcdf(_gdirs,
                                   input_filesuffix=input_filesuffix,
                                   output_filesuffix=output_filesuffix + '_sh',
                                   path=True,
                                   tmp_file_size=tmp_file_size,
                                   **kwargs)
                _gdirs = [gd for gd in gdirs if gd.hemisphere == 'nh']
                _compile_to_netcdf(_gdirs,
                                   input_filesuffix=input_filesuffix,
                                   output_filesuffix=output_filesuffix + '_nh',
                                   path=True,
                                   tmp_file_size=tmp_file_size,
                                   **kwargs)
                return

            task_name = task_func.__name__
            output_base = task_name.replace('compile_', '')

            if path is True:
                path = os.path.join(cfg.PATHS['working_dir'],
                                    output_base + output_filesuffix + '.nc')

            self.log.workflow('Applying %s on %d gdirs.',
                              task_name, len(gdirs))

            # Run the task
            # If small gdir size, no need for temporary files
            if len(gdirs) < tmp_file_size or not path:
                return task_func(gdirs, input_filesuffix=input_filesuffix,
                                 path=path, **kwargs)

            # Otherwise, divide and conquer
            sub_gdirs = [gdirs[i: i + tmp_file_size] for i in
                         range(0, len(gdirs), tmp_file_size)]

            tmp_paths = [os.path.join(cfg.PATHS['working_dir'],
                                      'compile_tmp_{:06d}.nc'.format(i))
                         for i in range(len(sub_gdirs))]

            try:
                for spath, sgdirs in zip(tmp_paths, sub_gdirs):
                    task_func(sgdirs, input_filesuffix=input_filesuffix,
                              path=spath, **kwargs)
            except BaseException:
                # If something wrong, delete the tmp files
                for f in tmp_paths:
                    try:
                        os.remove(f)
                    except FileNotFoundError:
                        pass
                raise

            # Ok, now merge and return
            try:
                with xr.open_mfdataset(tmp_paths, combine='nested',
                                       concat_dim='rgi_id') as ds:
                    # the .load() is actually quite uncool here, but it solves
                    # an unbelievable stalling problem in multiproc
                    ds.load().to_netcdf(path)
            except TypeError:
                # xr < v 0.13
                with xr.open_mfdataset(tmp_paths, concat_dim='rgi_id') as ds:
                    # the .load() is actually quite uncool here, but it solves
                    # an unbelievable stalling problem in multiproc
                    ds.load().to_netcdf(path)

            # We can't return the dataset without loading it, so we don't
            return None

        return _compile_to_netcdf


@compile_to_netcdf(log)
def compile_run_output(gdirs, path=True, input_filesuffix='',
                       use_compression=True):
    """Merge the output of the model runs of several gdirs into one file.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    path : str
        where to store (default is on the working dir).
        Set to `False` to disable disk storage.
    input_filesuffix : str
        the filesuffix of the files to be compiled
    use_compression : bool
        use zlib compression on the output netCDF files

    Returns
    -------
    ds : :py:class:`xarray.Dataset`
        compiled output
    """

    # Get the dimensions of all this
    rgi_ids = [gd.rgi_id for gd in gdirs]

    # To find the longest time, sort the gdirs by date
    sorted_gdir = sorted(gdirs, key=lambda gdir: gdir.rgi_date)
    # The first gdir might have blown up, try some others
    i = 0
    while True:
        if i >= len(sorted_gdir):
            raise RuntimeError('Found no valid glaciers!')
        try:
            ppath = sorted_gdir[i].get_filepath('model_diagnostics',
                                                filesuffix=input_filesuffix)
            with xr.open_dataset(ppath) as ds_diag:
                ds_diag.time.values
            break
        except BaseException:
            i += 1

    # OK found it, open it and prepare the output
    with xr.open_dataset(ppath) as ds_diag:

        # Prepare output
        ds = xr.Dataset()

        # Global attributes
        ds.attrs['description'] = 'OGGM model output'
        ds.attrs['oggm_version'] = __version__
        ds.attrs['calendar'] = '365-day no leap'
        ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # Copy coordinates
        time = ds_diag.time.values
        ds.coords['time'] = ('time', time)
        ds['time'].attrs['description'] = 'Floating hydrological year'
        # New coord
        ds.coords['rgi_id'] = ('rgi_id', rgi_ids)
        ds['rgi_id'].attrs['description'] = 'RGI glacier identifier'
        # This is just taken from there
        for cn in ['hydro_year', 'hydro_month',
                   'calendar_year', 'calendar_month']:
            ds.coords[cn] = ('time', ds_diag[cn].values)
            ds[cn].attrs['description'] = ds_diag[cn].attrs['description']

        # Prepare the 2D variables
        shape = (len(time), len(rgi_ids))
        out_2d = dict()
        for vn in ds_diag.data_vars:
            var = dict()
            var['data'] = np.full(shape, np.nan)
            var['attrs'] = ds_diag[vn].attrs
            out_2d[vn] = var

        # 1D Variables
        out_1d = dict()
        for vn, attrs in [('water_level', {'description': 'Calving water level',
                                           'units': 'm'}),
                          ('glen_a', {'description': 'Simulation Glen A',
                                      'units': ''}),
                          ('fs', {'description': 'Simulation sliding parameter',
                                  'units': ''}),
                          ]:
            var = dict()
            var['data'] = np.full(len(rgi_ids), np.nan)
            var['attrs'] = attrs
            out_1d[vn] = var

    # Read out
    for i, gdir in enumerate(gdirs):
        try:
            ppath = gdir.get_filepath('model_diagnostics',
                                      filesuffix=input_filesuffix)
            with xr.open_dataset(ppath) as ds_diag:
                nt = - len(ds_diag.volume_m3.values)
                for vn, var in out_2d.items():
                    var['data'][nt:, i] = ds_diag[vn].values
                for vn, var in out_1d.items():
                    var['data'][i] = ds_diag.attrs[vn]
        except BaseException:
            pass

    # To xarray
    for vn, var in out_2d.items():
        # Backwards compatibility - to remove one day...
        for r in ['_m3', '_m2', '_myr', '_m']:
            # Order matters
            vn = vn.replace(r, '')
        ds[vn] = (('time', 'rgi_id'), var['data'])
        ds[vn].attrs = var['attrs']
    for vn, var in out_1d.items():
        ds[vn] = (('rgi_id', ), var['data'])
        ds[vn].attrs = var['attrs']

    # To file?
    if path:
        enc_var = {'dtype': 'float32'}
        if use_compression:
            enc_var['complevel'] = 5
            enc_var['zlib'] = True
        encoding = {v: enc_var for v in ds.data_vars}
        ds.to_netcdf(path, encoding=encoding)

    return ds


@compile_to_netcdf(log)
def compile_climate_input(gdirs, path=True, filename='climate_historical',
                          input_filesuffix='', use_compression=True):
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
    input_filesuffix : str
        the filesuffix of the files to be compiled
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
            pgdir = gdirs[i]
            ppath = pgdir.get_filepath(filename=filename,
                                       filesuffix=input_filesuffix)
            with xr.open_dataset(ppath) as ds_clim:
                ds_clim.time.values
            # If this worked, we have a valid gdir
            break
        except BaseException:
            i += 1

    with xr.open_dataset(ppath) as ds_clim:
        cyrs = ds_clim['time.year']
        cmonths = ds_clim['time.month']
        has_grad = 'gradient' in ds_clim.variables
        sm = cfg.PARAMS['hydro_month_' + pgdir.hemisphere]
        yrs, months = calendardate_to_hydrodate(cyrs, cmonths, start_month=sm)
        assert months[0] == 1, 'Expected the first hydro month to be 1'
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
                                      filesuffix=input_filesuffix)
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


def compile_task_time(gdirs, task_names=[], filesuffix='', path=True,
                      append=True):
    """Gathers the time needed for the selected task(s) to run

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
            d[task_name] = gdir.get_task_time(task_name)
        out_df.append(d)

    out = pd.DataFrame(out_df).set_index('rgi_id')
    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                'task_time' + filesuffix + '.csv')
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
    d['rgi_year'] = gdir.rgi_date
    d['glacier_type'] = gdir.glacier_type
    d['terminus_type'] = gdir.terminus_type
    d['is_tidewater'] = gdir.is_tidewater
    d['status'] = gdir.status

    # The rest is less certain. We put these in a try block and see
    # We're good with any error - we store the dict anyway below
    # TODO: should be done with more preselected errors
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        try:
            # Inversion
            if gdir.has_file('inversion_output'):
                vol = []
                vol_bsl = []
                vol_bwl = []
                cl = gdir.read_pickle('inversion_output')
                for c in cl:
                    vol.extend(c['volume'])
                    vol_bsl.extend(c.get('volume_bsl', [0]))
                    vol_bwl.extend(c.get('volume_bwl', [0]))
                d['inv_volume_km3'] = np.nansum(vol) * 1e-9
                area = gdir.rgi_area_km2
                d['vas_volume_km3'] = 0.034 * (area ** 1.375)
                # BSL / BWL
                d['inv_volume_bsl_km3'] = np.nansum(vol_bsl) * 1e-9
                d['inv_volume_bwl_km3'] = np.nansum(vol_bwl) * 1e-9
        except BaseException:
            pass

        try:
            # Diagnostics
            diags = gdir.get_diagnostics()
            for k, v in diags.items():
                d[k] = v
        except BaseException:
            pass

        if inversion_only:
            return d

        try:
            # Error log
            errlog = gdir.get_error_log()
            if errlog is not None:
                d['error_task'] = errlog.split(';')[-2]
                d['error_msg'] = errlog.split(';')[-1]
            else:
                d['error_task'] = None
                d['error_msg'] = None
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
                widths = np.append(widths, fl.widths * gdir.grid.dx)
                slope = np.append(slope, np.arctan(-np.gradient(hgt, dx)))
                length = len(hgt) * dx
            d['main_flowline_length'] = length
            d['inv_flowline_glacier_area'] = np.sum(widths * dx)
            d['flowline_mean_elev'] = np.average(h, weights=widths)
            d['flowline_max_elev'] = np.max(h)
            d['flowline_min_elev'] = np.min(h)
            d['flowline_avg_slope'] = np.mean(slope)
            d['flowline_avg_width'] = np.mean(widths)
            d['flowline_last_width'] = fls[-1].widths[-1] * gdir.grid.dx
            d['flowline_last_5_widths'] = np.mean(fls[-1].widths[-5:] *
                                                  gdir.grid.dx)
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


def compile_fixed_geometry_mass_balance(gdirs, filesuffix='', path=True,
                                        use_inversion_flowlines=True,
                                        ys=None, ye=None, years=None):
    """Compiles a table of specific mass-balance timeseries for all glaciers.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    use_inversion_flowlines : bool
        whether to use the inversion flowlines or the model flowlines
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    """
    from oggm.workflow import execute_entity_task
    from oggm.core.massbalance import fixed_geometry_mass_balance

    out_df = execute_entity_task(fixed_geometry_mass_balance, gdirs,
                                 use_inversion_flowlines=use_inversion_flowlines,
                                 ys=ys, ye=ye, years=years)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.NaN)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('fixed_geometry_mass_balance' +
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            # Flowline related stuff
            h = np.array([])
            widths = np.array([])
            fls = gdir.read_pickle('inversion_flowlines')
            dx = fls[0].dx * gdir.grid.dx
            for fl in fls:
                hgt = fl.surface_h
                h = np.append(h, hgt)
                widths = np.append(widths, fl.widths * dx)
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


def extend_past_climate_run(past_run_file=None,
                            fixed_geometry_mb_file=None,
                            glacier_statistics_file=None,
                            path=False,
                            use_compression=True):
    """Utility function to extend past MB runs prior to the RGI date.

    We use a fixed geometry (and a fixed calving rate) for all dates prior
    to the RGI date.

    Parameters
    ----------
    past_run_file : str
        path to the historical run (nc)
    fixed_geometry_mb_file : str
        path to the MB file (csv)
    glacier_statistics_file : str
        path to the glacier stats file (csv)
    path : str
        where to store the file
    use_compression : bool

    Returns
    -------
    the extended dataset
    """

    log.workflow('Applying extend_past_climate_run on '
                 '{}'.format(past_run_file))

    fixed_geometry_mb_df = pd.read_csv(fixed_geometry_mb_file, index_col=0,
                                       low_memory=False)
    stats_df = pd.read_csv(glacier_statistics_file, index_col=0,
                           low_memory=False)

    with xr.open_dataset(past_run_file) as past_ds:

        y0_run = int(past_ds.time[0])
        y1_run = int(past_ds.time[-1])
        if (y1_run - y0_run + 1) != len(past_ds.time):
            raise NotImplementedError('Currently only supporting annual outputs')
        y0_clim = int(fixed_geometry_mb_df.index[0])
        y1_clim = int(fixed_geometry_mb_df.index[-1])
        if y0_clim > y0_run or y1_clim < y0_run:
            raise InvalidWorkflowError('Dates do not match.')
        if y1_clim != y1_run - 1:
            raise InvalidWorkflowError('Dates do not match.')
        if len(past_ds.rgi_id) != len(fixed_geometry_mb_df.columns):
            raise InvalidWorkflowError('Nb of glaciers do not match.')
        if len(past_ds.rgi_id) != len(stats_df.index):
            raise InvalidWorkflowError('Nb of glaciers do not match.')

        # Make sure we agree on order
        df = fixed_geometry_mb_df[past_ds.rgi_id]

        # Output data
        years = np.arange(y0_clim, y1_run+1)
        ods = past_ds.reindex({'time': years})

        # Time
        ods['hydro_year'].data[:] = years
        ods['hydro_month'].data[:] = ods['hydro_month'][-1]
        ods['calendar_year'].data[:] = years - 1
        ods['calendar_month'].data[:] = ods['calendar_month'][-1]
        for vn in ['hydro_year', 'hydro_month',
                   'calendar_year', 'calendar_month']:
            ods[vn] = ods[vn].astype(int)

        # New vars
        for vn in ['volume', 'volume_bsl', 'volume_bwl', 'area', 'calving']:
            ods[vn + '_ext'] = ods[vn].copy(deep=True)
            ods[vn + '_ext'].attrs['description'] += ' (extended with MB data)'

        vn = 'volume_fixed_geom_ext'
        ods[vn] = ods['volume'].copy(deep=True)
        ods[vn].attrs['description'] += ' (replaced with fixed geom data)'

        rho = cfg.PARAMS['ice_density']
        # Loop over the ids
        for i, rid in enumerate(ods.rgi_id.data):
            # Both do not need to be same length but they need to start same
            mb_ts = df.values[:, i]
            orig_vol_ts = ods.volume_ext.data[:, i]
            if not (np.isfinite(mb_ts[-1]) and np.isfinite(orig_vol_ts[-1])):
                # Not a valid glacier
                continue
            if np.isfinite(orig_vol_ts[0]):
                # Nothing to extend, really
                continue
            orig_area_ts = ods.area_ext.data[:, i]
            orig_calv_ts = ods.calving_ext.data[:, i]
            # First valid id
            fid = np.argmax(np.isfinite(orig_vol_ts))
            # Fill area which stays constant
            orig_area_ts[:fid] = orig_area_ts[fid]

            # Add calving flux to the mix
            try:
                calv_flux = stats_df.loc[rid, 'calving_flux'] * 1e9
            except KeyError:
                calv_flux = 0
            if not np.isfinite(calv_flux):
                calv_flux = 0

            # We convert SMB to volume
            mb_vol_ts = (mb_ts / rho * orig_area_ts[fid] - calv_flux).cumsum()
            calv_ts = (mb_ts * 0 + calv_flux).cumsum()

            # The -1 is because the volume change is known at end of year
            mb_vol_ts = mb_vol_ts + orig_vol_ts[fid] - mb_vol_ts[fid-1]
            calv_ts = calv_ts + orig_calv_ts[fid] - calv_ts[fid-1]

            # Now back to netcdf
            ods.volume_fixed_geom_ext.data[1:, i] = mb_vol_ts
            ods.volume_ext.data[1:fid, i] = mb_vol_ts[0:fid-1]
            ods.calving_ext.data[1:fid, i] = calv_ts[0:fid-1]
            ods.area_ext.data[:, i] = orig_area_ts

            # Extend vol bsl by assuming that % stays constant
            bsl = ods.volume_bsl.data[fid, i] / ods.volume.data[fid, i]
            bwl = ods.volume_bwl.data[fid, i] / ods.volume.data[fid, i]
            ods.volume_bsl_ext.data[:fid, i] = bsl * ods.volume_ext.data[:fid, i]
            ods.volume_bwl_ext.data[:fid, i] = bwl * ods.volume_ext.data[:fid, i]

        # Remove old vars
        for vn in list(ods.data_vars):
            if '_ext' not in vn:
                del ods[vn]

        # Remove t0 (which is NaN)
        ods = ods.isel(time=slice(1, None))

        # To file?
        if path:
            enc_var = {'dtype': 'float32'}
            if use_compression:
                enc_var['complevel'] = 5
                enc_var['zlib'] = True
            encoding = {v: enc_var for v in ods.data_vars}
            ods.to_netcdf(path, encoding=encoding)

    return ods


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
    fl = Centerline(line, dx=flowline_dx, surface_h=surface_h, map_dx=map_dx)
    fl.widths = widths_m / map_dx
    fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
    gdir.write_pickle([fl], 'inversion_flowlines')

    # Idealized map
    grid = salem.Grid(nxny=(1, 1), dxdy=(map_dx, map_dx), x0y0=(0, 0))
    grid.to_json(gdir.get_filepath('glacier_grid'))

    return gdir


def _back_up_retry(func, exceptions, max_count=5):
    """Re-Try an action up to max_count times.
    """

    count = 0
    while count < max_count:
        try:
            if count > 0:
                time.sleep(random.uniform(0.05, 0.1))
            return func()
        except exceptions:
            count += 1
            if count >= max_count:
                raise


def _robust_extract(to_dir, *args, **kwargs):
    """For some obscure reason this operation randomly fails.

    Try to make it more robust.
    """

    def func():
        with tarfile.open(*args, **kwargs) as tf:
            if not len(tf.getnames()):
                raise RuntimeError("Empty tarfile")
            tf.extractall(os.path.dirname(to_dir))

    _back_up_retry(func, FileExistsError)


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

        def func():
            with tarfile.open(base_tar, 'r') as tf:
                i_from_tar = tf.getmember(os.path.join(dirbname, bname))
                with tf.extractfile(i_from_tar) as fileobj:
                    _robust_extract(to_dir, fileobj=fileobj)

        _back_up_retry(func, RuntimeError)

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
            write_shp = False
        else:
            g = rgi_entity['geometry']
            xx, yy = ([g.bounds[0], g.bounds[2]],
                      [g.bounds[1], g.bounds[3]])
            write_shp = True

        # Extent of the glacier in lon/lat
        self.extent_ll = [xx, yy]

        try:
            # RGI V4?
            rgi_entity.RGIID
            raise ValueError('RGI Version 4 is not supported anymore')
        except AttributeError:
            pass

        self.rgi_id = rgi_entity.RGIId
        self.glims_id = rgi_entity.GLIMSId

        # Do we want to use the RGI center point or ours?
        if cfg.PARAMS['use_rgi_area']:
            self.cenlon = float(rgi_entity.CenLon)
            self.cenlat = float(rgi_entity.CenLat)
        else:
            cenlon, cenlat = rgi_entity.geometry.centroid.xy
            self.cenlon = float(cenlon[0])
            self.cenlat = float(cenlat[0])

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
                                                   'Lake-terminating',
                                                   'Shelf-terminating']
        self.is_lake_terminating = self.terminus_type == 'Lake-terminating'
        self.is_marine_terminating = self.terminus_type == 'Marine-terminating'
        self.is_shelf_terminating = self.terminus_type == 'Shelf-terminating'
        self.is_nominal = self.status == 'Nominal glacier'
        self.inversion_calving_rate = 0.
        self.is_icecap = self.glacier_type == 'Ice cap'

        # Hemisphere
        if self.cenlat < 0 or self.rgi_region == '16':
            self.hemisphere = 'sh'
        else:
            self.hemisphere = 'nh'

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
            write_shp = False
        else:
            mkdir(self.dir)

        if not os.path.isdir(self.dir):
            raise RuntimeError('GlacierDirectory %s does not exist!' % self.dir)

        # logging file
        self.logfile = os.path.join(self.dir, 'log.txt')

        if write_shp:
            # Write shapefile
            self._reproject_and_write_shapefile(rgi_entity)

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

    def _reproject_and_write_shapefile(self, entity):

        # Make a local glacier map
        params = dict(name='tmerc', lat_0=0., lon_0=self.cenlon,
                      k=0.9996, x_0=0, y_0=0, datum='WGS84')
        proj4_str = "+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} " \
                    "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**params)

        # Reproject
        proj_in = pyproj.Proj("epsg:4326", preserve_units=True)
        proj_out = pyproj.Proj(proj4_str, preserve_units=True)

        # transform geometry to map
        project = partial(transform_proj, proj_in, proj_out)
        geometry = shp_trafo(project, entity['geometry'])
        geometry = multipolygon_to_polygon(geometry, gdir=self)

        # Save transformed geometry to disk
        entity = entity.copy()
        entity['geometry'] = geometry

        # Do we want to use the RGI area or ours?
        if not cfg.PARAMS['use_rgi_area']:
            # Update Area
            area = geometry.area * 1e-6
            entity['Area'] = area

        # Avoid fiona bug: https://github.com/Toblerity/Fiona/issues/365
        for k, s in entity.iteritems():
            if type(s) in [np.int32, np.int64]:
                entity[k] = int(s)
        towrite = gpd.GeoDataFrame(entity).T
        towrite.crs = proj4_str

        # Write shapefile
        self.write_shapefile(towrite, 'outlines')

        # Also transform the intersects if necessary
        gdf = cfg.PARAMS['intersects_gdf']
        if len(gdf) > 0:
            gdf = gdf.loc[((gdf.RGIId_1 == self.rgi_id) |
                           (gdf.RGIId_2 == self.rgi_id))]
            if len(gdf) > 0:
                gdf = salem.transform_geopandas(gdf, to_crs=proj_out)
                if hasattr(gdf.crs, 'srs'):
                    # salem uses pyproj
                    gdf.crs = gdf.crs.srs
                self.write_shapefile(gdf, 'intersects')
        else:
            # Sanity check
            if cfg.PARAMS['use_intersects']:
                raise InvalidParamsError(
                    'You seem to have forgotten to set the '
                    'intersects file for this run. OGGM '
                    'works better with such a file. If you '
                    'know what your are doing, set '
                    "cfg.PARAMS['use_intersects'] = False to "
                    "suppress this error.")

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
            raise RuntimeError('No outlines available')

    @lazy_property
    def intersects_ids(self):
        """The glacier's intersects RGI ids."""
        try:
            gdf = self.read_shapefile('intersects')
            ids = np.append(gdf['RGIId_1'], gdf['RGIId_2'])
            ids = list(np.unique(np.sort(ids)))
            ids.remove(self.rgi_id)
            return ids
        except OSError:
            return []

    @lazy_property
    def dem_daterange(self):
        """Years in which most of the DEM data was acquired"""
        source_txt = self.get_filepath('dem_source')
        if os.path.isfile(source_txt):
            with open(source_txt, 'r') as f:
                for line in f.readlines():
                    if 'Date range:' in line:
                        return tuple(map(int, line.split(':')[1].split('-')))
        # we did not find the information in the dem_source file
        log.warning('No DEM date range specified in `dem_source.txt`')
        return None

    @lazy_property
    def dem_info(self):
        """More detailed information on the acquisition of the DEM data"""
        source_file = self.get_filepath('dem_source')
        source_text = ''
        if os.path.isfile(source_file):
            with open(source_file, 'r') as f:
                for line in f.readlines():
                    source_text += line
        else:
            log.warning('No DEM source file found.')
        return source_text

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

        if filename == 'climate_historical' and not os.path.exists(out):
            # For backwards compatibility, in these cases try climate_monthly
            if self.has_file('climate_monthly'):
                return self.get_filepath('climate_monthly', delete=delete,
                                         filesuffix=filesuffix)

        if delete and os.path.isfile(out):
            os.remove(out)
        return out

    def has_file(self, filename, filesuffix=''):
        """Checks if a file exists.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        """
        fp = self.get_filepath(filename, filesuffix=filesuffix)
        if '.shp' in fp and cfg.PARAMS['use_tar_shapefiles']:
            fp = fp.replace('.shp', '.tar')
            if cfg.PARAMS['use_compression']:
                fp += '.gz'

        out = os.path.exists(fp)

        # Deprecation cycle
        if not out and (filename == 'climate_info'):
            # Try pickle
            out = os.path.exists(fp.replace('.json', '.pkl'))

        return out

    def _read_deprecated_climate_info(self):
        """Temporary fix for climate_info file type change."""
        fp = self.get_filepath('climate_info')
        if not os.path.exists(fp):
            fp = fp.replace('.json', '.pkl')
            if not os.path.exists(fp):
                raise FileNotFoundError('No climate info file available!')
            _open = gzip.open if cfg.PARAMS['use_compression'] else open
            with _open(fp, 'rb') as f:
                out = pickle.load(f)
            return out
        with open(fp, 'r') as f:
            out = json.load(f)
        return out

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

        # Some deprecations
        if filename == 'climate_info':
            return self._read_deprecated_climate_info()

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

        # Some deprecations
        if filename == 'climate_info':
            return self._read_deprecated_climate_info()

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

        def np_convert(o):
            if isinstance(o, np.int64):
                return int(o)
            raise TypeError

        fp = self.get_filepath(filename, filesuffix=filesuffix)
        with open(fp, 'w') as f:
            json.dump(var, f, default=np_convert)

    def get_climate_info(self, input_filesuffix=''):
        """Convenience function handling some backwards compat aspects"""
            
        # I don't know another way right now, 
        # as I was not able to get the input_filesuffix 
        # to those functions that use get_climate_info 
        if cfg.PARAMS['baseline_climate'] == 'ERA5_daily':
             input_filesuffix = '_daily'
        
        try:
            out = self.read_json('climate_info')
        except FileNotFoundError:
            out = {}

        try:
            f = self.get_filepath('climate_historical',
                                  filesuffix=input_filesuffix)
            with ncDataset(f) as nc:
                out['baseline_climate_source'] = nc.climate_source
                out['baseline_hydro_yr_0'] = nc.hydro_yr_0
                out['baseline_hydro_yr_1'] = nc.hydro_yr_1
        except (AttributeError, FileNotFoundError):
            pass

        return out


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

        shp = gpd.read_file(fp)

        # .properties file is created for compressed shapefiles. github: #904
        _properties = fp.replace('tar://', '') + '.properties'
        if os.path.isfile(_properties):
            # remove it, to keep GDir slim
            os.remove(_properties)

        return shp

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

    def write_monthly_climate_file(self, time, prcp, temp,
                                   ref_pix_hgt, ref_pix_lon, ref_pix_lat, *,
                                   gradient=None,
                                   temp_std=None,
                                   time_unit=None,
                                   calendar=None,
                                   source=None,
                                   file_name='climate_historical',
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
        temp_std : ndarray, optional
            the daily standard deviation of temperature (useful for PyGEM)
        time_unit : str
            the reference time unit for your time array. This should be chosen
            depending on the length of your data. The default is to choose
            it ourselves based on the starting year.
        calendar : str
            If you use an exotic calendar (e.g. 'noleap')
        source : str
            the climate data source (required)
        file_name : str
            How to name the file
        filesuffix : str
            Apply a suffix to the file
        """

        # overwrite as default
        fpath = self.get_filepath(file_name, filesuffix=filesuffix)
        if os.path.exists(fpath):
            os.remove(fpath)

        if source is None:
            raise InvalidParamsError('`source` kwarg is required')

        zlib = cfg.PARAMS['compress_climate_netcdf']

        try:
            y0 = time[0].year
            y1 = time[-1].year
        except AttributeError:
            time = pd.DatetimeIndex(time)
            y0 = time[0].year
            y1 = time[-1].year

        if time_unit is None:
            # http://pandas.pydata.org/pandas-docs/stable/timeseries.html
            # #timestamp-limitations
            if y0 > 1800:
                time_unit = 'days since 1801-01-01 00:00:00'
            elif y0 >= 0:
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
            nc.climate_source = source
            # hydro_year corresponds to the last month of the data 
            if time[0].month == 1:
                # if first_month =1, last_month = 12, so y0 is hydro_yr_0
                nc.hydro_yr_0 = y0
            else:
                # if first_month>1, then the last_month is in the next year,
                nc.hydro_yr_0 = y0 + 1
            nc.hydro_yr_1 = y1

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

            # check if prcp has really monthly format 
            if len(prcp)==(nc.hydro_yr_1-nc.hydro_yr_0+1)*12:
                v.long_name = 'total monthly precipitation amount'
            else:
                v.long_name = 'total monthly precipitation amount'
                import warnings
                warnings.warn("there might be a conflict in the prcp"
                              "timeseries, please check!")
            
            v[:] = prcp

            v = nc.createVariable('temp', 'f4', ('time',), zlib=zlib)
            v.units = 'degC'
            v.long_name = '2m temperature at height ref_hgt'
            v[:] = temp

            if gradient is not None:
                v = nc.createVariable('gradient', 'f4', ('time',), zlib=zlib)
                v.units = 'degC m-1'
                v.long_name = ('temperature gradient from local regression or'
                               'lapserates')
                v[:] = gradient

            if temp_std is not None:
                v = nc.createVariable('temp_std', 'f4', ('time',), zlib=zlib)
                v.units = 'degC'
                v.long_name = 'standard deviation of daily temperatures'
                v[:] = temp_std

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

    def set_ref_mb_data(self, mb_df=None):
        """Adds reference mass-balance data to this glacier.

        The format should be a dataframe with the years as index and
        'ANNUAL_BALANCE' as values in mm yr-1.
        """

        if self.is_tidewater:
            log.warning('You are trying to set MB data on a tidewater glacier!'
                        ' These data will be ignored by the MB model '
                        'calibration routine.')

        if mb_df is None:

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
            mb_df = pd.read_csv(reff).set_index('YEAR')

        # Quality checks
        if 'ANNUAL_BALANCE' not in mb_df:
            raise InvalidParamsError('Need an "ANNUAL_BALANCE" column in the '
                                     'dataframe.')
        if not mb_df.index.is_integer():
            raise InvalidParamsError('The index needs to be integer years')

        mb_df.index.name = 'YEAR'
        self._mbdf = mb_df

    def get_ref_mb_data(self, y0=None, y1=None):
        """Get the reference mb data from WGMS (for some glaciers only!).

        Raises an Error if it isn't a reference glacier at all.

        Parameters
        ----------
        y0 : int
            override the default behavior which is to check the available
            climate data (or PARAMS['ref_mb_valid_window']) and decide
        y1 : int
            override the default behavior which is to check the available
            climate data (or PARAMS['ref_mb_valid_window']) and decide
        """

        if self._mbdf is None:
            self.set_ref_mb_data()

        # logic for period
        t0, t1 = cfg.PARAMS['ref_mb_valid_window']
        if t0 > 0 and y0 is None:
            y0 = t0
        if t1 > 0 and y1 is None:
            y1 = t1

        if y0 is None or y1 is None:
            ci = self.get_climate_info()
            if 'baseline_hydro_yr_0' not in ci:
                raise InvalidWorkflowError('Please process some climate data '
                                           'before call')
            y0 = ci['baseline_hydro_yr_0'] if y0 is None else y0
            y1 = ci['baseline_hydro_yr_1'] if y1 is None else y1

        if len(self._mbdf) > 1:
            out = self._mbdf.loc[y0:y1]
        else:
            # Some files are just empty
            out = self._mbdf
        return out.dropna(subset=['ANNUAL_BALANCE'])

    def get_ref_mb_profile(self, check = True):
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
        
        # somehow that does not work with the HUSS flowlines ?!
        # check = False for HUSS flowlines
        if check:
            # logic for period
            if not self.has_file('climate_info'):
                raise RuntimeError('Please process some climate data before call')
        ci = self.get_climate_info()
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
        """Get the glacier length data from P. Leclercq's data base.

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

    def log(self, task_name, *, err=None, task_time=None):
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
        time : float
            the time (in seconds) that the task needed to run
        """

        # a line per function call
        nowsrt = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        line = nowsrt + ';' + task_name + ';'

        if task_time is not None:
            line += 'time:{};'.format(task_time)

        if err is None:
            line += 'SUCCESS'
        else:
            line += err.__class__.__name__ + ': {}'.format(err)\

        line = line.replace('\n', ' ')

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
        """Opens this directory's log file to check for a task's outcome.

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

        lines = [l.replace('\n', '') for l in lines
                 if ';' in l and (task_name == l.split(';')[1])]
        if lines:
            # keep only the last log
            return lines[-1].split(';')[-1]
        else:
            return None

    def get_task_time(self, task_name):
        """Opens this directory's log file to check for a task's run time.

        Parameters
        ----------
        task_name : str
            the name of the task which has to be tested for

        Returns
        -------
        The timing that the last call of this task needed.
        None if the task was not run yet, or if it errored
        """

        if not os.path.isfile(self.logfile):
            return None

        with open(self.logfile) as logfile:
            lines = logfile.readlines()

        lines = [l.replace('\n', '') for l in lines
                 if task_name == l.split(';')[1]]
        if lines:
            line = lines[-1]
            # Last log is message
            if 'ERROR' in line.split(';')[-1] or 'time:' not in line:
                return None
            # Get the time
            return float(line.split('time:')[-1].split(';')[0])
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
                 'local_mustar', 'climate_historical',
                 'gcm_data', 'climate_info', 'diagnostics']
        paths = ('*' + p + '*' for p in paths)
        shutil.copytree(gdir.dir, new_dir,
                        ignore=include_patterns(*paths))
    elif setup == 'inversion':
        paths = ['inversion_params', 'downstream_line', 'outlines',
                 'inversion_flowlines', 'glacier_grid', 'diagnostics',
                 'local_mustar', 'climate_historical', 'gridded_data',
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
                           filename='climate_historical', input_filesuffix=''):
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
    merged_geometry = maindf.loc[idx, 'geometry'].iloc[0].buffer(0)
    for trib in tribs:
        geom = trib.read_pickle('geometries')['polygon_hr']
        geom = salem.transform_geometry(geom, crs=trib.grid)
        merged_geometry = merged_geometry.union(geom).buffer(0)

    # to get the center point, maximal extensions for DEM and single Polygon:
    new_geometry = merged_geometry.convex_hull
    maindf.loc[idx, 'geometry'] = new_geometry

    # make some adjustments to the rgi dataframe
    # 1. calculate central point of new glacier
    #    reproject twice to avoid Warning, first to flat projection
    flat_centroid = salem.transform_geometry(new_geometry,
                                             to_crs=main.grid).centroid
    #    second reprojection of centroid to wgms
    new_centroid = salem.transform_geometry(flat_centroid, crs=main.grid)
    maindf.loc[idx, 'CenLon'] = new_centroid.x
    maindf.loc[idx, 'CenLat'] = new_centroid.y
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
    climateinfo = main.read_json('climate_info')
    merged.write_json(climateinfo, 'climate_info')

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

    to_delete = []
    for dirname, subdirlist, filelist in os.walk(base_dir):
        # RGI60-01.00
        bname = os.path.basename(dirname)
        if not (len(bname) == 11 and bname[-3] == '.'):
            continue
        opath = dirname + '.tar'
        with tarfile.open(opath, 'w') as tar:
            tar.add(dirname, arcname=os.path.basename(dirname))
        if delete:
            to_delete.append(dirname)

    for dirname in to_delete:
        shutil.rmtree(dirname)
