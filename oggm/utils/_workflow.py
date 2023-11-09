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
import itertools
from collections import OrderedDict
from functools import partial, wraps
from time import gmtime, strftime
import fnmatch
import platform
import struct
import importlib
import re as regexp

# External libs
import pandas as pd
import numpy as np
from scipy import stats
import xarray as xr
import shapely.geometry as shpg
import shapely.affinity as shpa
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
                               haversine, multipolygon_to_polygon,
                               recursive_valid_polygons)
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
        """Instantiate.

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
        logging.disable(logging.CRITICAL)

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
        """Decorator syntax: ``@entity_task(log, writes=['dem', 'outlines'])``

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
        cnt += ['    Files written to the glacier directory:']

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
                    if add_to_log_file:
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


class global_task(object):
    """Decorator for common job-controlling logic.

    Indicates that this task expects a list of all GlacierDirs as parameter
    instead of being called once per dir.
    """

    def __init__(self, log):
        """Decorator syntax: ``@global_task(log)``

        Parameters
        ----------
        log: logger
            module logger
        """
        self.log = log

    def __call__(self, task_func):
        """Decorate."""

        @wraps(task_func)
        def _global_task(gdirs, **kwargs):

            # Should be iterable
            gdirs = tolist(gdirs)

            self.log.workflow('Applying global task %s on %s glaciers',
                              task_func.__name__, len(gdirs))

            # Run the task
            return task_func(gdirs, **kwargs)

        _global_task.__dict__['is_global_task'] = True
        return _global_task


def get_ref_mb_glaciers_candidates(rgi_version=None):
    """Reads in the WGMS list of glaciers with available MB data.

    Can be found afterwards (and extended) in cdf.DATA['RGIXX_ref_ids'].
    """

    if rgi_version is None:
        rgi_version = cfg.PARAMS['rgi_version']

    if len(rgi_version) == 2:
        # We might change this one day
        rgi_version = rgi_version[:1]

    key = 'RGI{}0_ref_ids'.format(rgi_version)

    if key not in cfg.DATA:
        flink, _ = get_wgms_files()
        cfg.DATA[key] = flink['RGI{}0_ID'.format(rgi_version)].tolist()

    return cfg.DATA[key]


@global_task(log)
def get_ref_mb_glaciers(gdirs, y0=None, y1=None):
    """Get the list of glaciers we have valid mass balance measurements for.

    To be valid glaciers must have more than 5 years of measurements and
    be land terminating. Therefore, the list depends on the time period of the
    baseline climate data and this method selects them out of a list
    of potential candidates (`gdirs` arg).

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        list of glaciers to check for valid reference mass balance data
    y0 : int
        override the default behavior which is to check the available
        climate data (or PARAMS['ref_mb_valid_window']) and decide
    y1 : int
        override the default behavior which is to check the available
        climate data (or PARAMS['ref_mb_valid_window']) and decide

    Returns
    -------
    ref_gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        list of those glaciers with valid reference mass balance data

    See Also
    --------
    get_ref_mb_glaciers_candidates
    """

    # Get the links
    ref_ids = get_ref_mb_glaciers_candidates(gdirs[0].rgi_version)

    # We remove tidewater glaciers and glaciers with < 5 years
    ref_gdirs = []
    for g in gdirs:
        if g.rgi_id not in ref_ids or g.is_tidewater:
            continue
        try:
            mbdf = g.get_ref_mb_data(y0=y0, y1=y1)
            if len(mbdf) >= 5:
                ref_gdirs.append(g)
        except RuntimeError as e:
            if 'Please process some climate data before call' in str(e):
                raise
    return ref_gdirs


def _chaikins_corner_cutting(line, refinements=5):
    """Some magic here.

    https://stackoverflow.com/questions/47068504/where-to-find-python-
    implementation-of-chaikins-corner-cutting-algorithm
    """
    coords = np.array(line.coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return shpg.LineString(coords)


@entity_task(log)
def get_centerline_lonlat(gdir,
                          keep_main_only=False,
                          flowlines_output=False,
                          ensure_exterior_match=False,
                          geometrical_widths_output=False,
                          corrected_widths_output=False,
                          to_crs='wgs84',
                          simplify_line_before=0,
                          corner_cutting=0,
                          simplify_line_after=0):
    """Helper task to convert the centerlines to a shapefile

    Parameters
    ----------
    gdir : the glacier directory
    flowlines_output : create a shapefile for the flowlines
    ensure_exterior_match : per design, OGGM centerlines match the underlying
    DEM grid. This may imply that they do not "touch" the exterior outlines
    of the glacier in vector space. Set this to True to correct for that.
    geometrical_widths_output : for the geometrical widths
    corrected_widths_output : for the corrected widths

    Returns
    -------
    a shapefile
    """
    if flowlines_output or geometrical_widths_output or corrected_widths_output:
        cls = gdir.read_pickle('inversion_flowlines')
    else:
        cls = gdir.read_pickle('centerlines')

    exterior = None
    if ensure_exterior_match:
        exterior = gdir.read_shapefile('outlines')
        # Transform to grid
        tra_func = partial(gdir.grid.transform, crs=exterior.crs)
        exterior = shpg.Polygon(shp_trafo(tra_func, exterior.geometry[0].exterior))

    tra_func = partial(gdir.grid.ij_to_crs, crs=to_crs)

    olist = []
    for j, cl in enumerate(cls):
        mm = 1 if j == (len(cls)-1) else 0
        if keep_main_only and mm == 0:
            continue
        if corrected_widths_output:
            le_segment = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
            for wi, cur, (n1, n2), wi_m in zip(cl.widths, cl.line.coords,
                                               cl.normals, cl.widths_m):
                _l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                      shpg.Point(cur + wi / 2. * n2)])
                gs = dict()
                gs['RGIID'] = gdir.rgi_id
                gs['SEGMENT_ID'] = j
                gs['LE_SEGMENT'] = le_segment
                gs['MAIN'] = mm
                gs['WIDTH_m'] = wi_m
                gs['geometry'] = shp_trafo(tra_func, _l)
                olist.append(gs)
        elif geometrical_widths_output:
            le_segment = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
            for _l, wi_m in zip(cl.geometrical_widths, cl.widths_m):
                gs = dict()
                gs['RGIID'] = gdir.rgi_id
                gs['SEGMENT_ID'] = j
                gs['LE_SEGMENT'] = le_segment
                gs['MAIN'] = mm
                gs['WIDTH_m'] = wi_m
                gs['geometry'] = shp_trafo(tra_func, _l)
                olist.append(gs)
        else:
            gs = dict()
            gs['RGIID'] = gdir.rgi_id
            gs['SEGMENT_ID'] = j
            gs['STRAHLER'] = cl.order
            if mm == 0:
                gs['OUTFLOW_ID'] = cls.index(cl.flows_to)
            else:
                gs['OUTFLOW_ID'] = -1
            gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
            gs['MAIN'] = mm
            line = cl.line
            if ensure_exterior_match:
                # Extend line at the start by 10
                fs = shpg.LineString(line.coords[:2])
                # First check if this is necessary - this segment should
                # be within the geometry or it's already good to go
                if fs.within(exterior):
                    fs = shpa.scale(fs, xfact=3, yfact=3, origin=fs.boundary.geoms[1])
                    line = shpg.LineString([*fs.coords, *line.coords[2:]])
                # If last also extend at the end
                if mm == 1:
                    ls = shpg.LineString(line.coords[-2:])
                    if ls.within(exterior):
                        ls = shpa.scale(ls, xfact=3, yfact=3, origin=ls.boundary.geoms[0])
                        line = shpg.LineString([*line.coords[:-2], *ls.coords])

                # Simplify and smooth?
                if simplify_line_before:
                    line = line.simplify(simplify_line_before)
                if corner_cutting:
                    line = _chaikins_corner_cutting(line, corner_cutting)
                if simplify_line_after:
                    line = line.simplify(simplify_line_after)

                # Intersect with exterior geom
                line = line.intersection(exterior)
                if line.geom_type in ['MultiLineString', 'GeometryCollection']:
                    # Take the longest
                    lens = [il.length for il in line.geoms]
                    line = line.geoms[np.argmax(lens)]

                # Recompute length
                gs['LE_SEGMENT'] = np.rint(line.length * gdir.grid.dx)
            gs['geometry'] = shp_trafo(tra_func, line)
            olist.append(gs)

    return olist


def _write_shape_to_disk(gdf, fpath, to_tar=False):
    """Write a shapefile to disk with optional compression

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        the data to write
    fpath : str
        where to writ the file - should be ending in shp
    to_tar : bool
        put the files in a .tar file. If cfg.PARAMS['use_compression'],
        also compress to .gz
    """

    if '.shp' not in fpath:
        raise ValueError('File ending should be .shp')

    gdf.to_file(fpath)

    if not to_tar:
        # Done here
        return

    # Write them in tar
    fpath = fpath.replace('.shp', '.tar')
    mode = 'w'
    if cfg.PARAMS['use_compression']:
        fpath += '.gz'
        mode += ':gz'
    if os.path.exists(fpath):
        os.remove(fpath)

    # List all files that were written as shape
    fs = glob.glob(fpath.replace('.gz', '').replace('.tar', '.*'))
    # Add them to tar
    with tarfile.open(fpath, mode=mode) as tf:
        for ff in fs:
            tf.add(ff, arcname=os.path.basename(ff))

    # Delete the old ones
    for ff in fs:
        os.remove(ff)


@global_task(log)
def write_centerlines_to_shape(gdirs, *, path=True, to_tar=False,
                               to_crs='EPSG:4326',
                               filesuffix='', flowlines_output=False,
                               ensure_exterior_match=False,
                               geometrical_widths_output=False,
                               corrected_widths_output=False,
                               keep_main_only=False,
                               simplify_line_before=0,
                               corner_cutting=0,
                               simplify_line_after=0):
    """Write the centerlines to a shapefile.

    Parameters
    ----------
    gdirs:
        the list of GlacierDir to process.
    path: str or bool
        Set to "True" in order  to store the shape in the working directory
        Set to a str path to store the file to your chosen location
    to_tar : bool
        put the files in a .tar file. If cfg.PARAMS['use_compression'],
        also compress to .gz
    filesuffix : str
        add a suffix to the output file
    flowlines_output : bool
        output the OGGM flowlines instead of the centerlines
    geometrical_widths_output : bool
        output the geometrical widths instead of the centerlines
    corrected_widths_output : bool
        output the corrected widths instead of the centerlines
    ensure_exterior_match : bool
        per design, the centerlines will match the underlying DEM grid.
        This may imply that they do not "touch" the exterior outlines of the
        glacier in vector space. Set this to True to correct for that.
    to_crs : str
        write the shape to another coordinate reference system (CRS)
    keep_main_only : bool
        write only the main flowlines to the output files
    simplify_line_before : float
        apply shapely's `simplify` method to the line before corner cutting.
        It is a cosmetic option: it avoids hard "angles" in the centerlines.
        All points in the simplified object will be within the tolerance
        distance of the original geometry (units: grid points). A good
        value to test first is 0.75
    corner_cutting : int
        apply the Chaikin's corner cutting algorithm to the geometry before
        writing. The integer represents the number of refinements to apply.
        A good first value to test is 3.
    simplify_line_after : float
        apply shapely's `simplify` method to the line *after* corner cutting.
        This is to reduce the size of the geometeries after they have been
        smoothed. The default value of 0 is fine if you use corner cutting less
        than 4. Otherwize try a small number, like 0.05 or 0.1.
    """
    from oggm.workflow import execute_entity_task

    if path is True:
        path = os.path.join(cfg.PATHS['working_dir'],
                            'glacier_centerlines' + filesuffix + '.shp')

    _to_crs = salem.check_crs(to_crs)
    if not _to_crs:
        raise InvalidParamsError(f'CRS not understood: {to_crs}')

    log.workflow('write_centerlines_to_shape on {} ...'.format(path))

    olist = execute_entity_task(get_centerline_lonlat, gdirs,
                                flowlines_output=flowlines_output,
                                ensure_exterior_match=ensure_exterior_match,
                                geometrical_widths_output=geometrical_widths_output,
                                corrected_widths_output=corrected_widths_output,
                                keep_main_only=keep_main_only,
                                simplify_line_before=simplify_line_before,
                                corner_cutting=corner_cutting,
                                simplify_line_after=simplify_line_after,
                                to_crs=_to_crs)
    # filter for none
    olist = [o for o in olist if o is not None]
    odf = gpd.GeoDataFrame(itertools.chain.from_iterable(olist))
    odf = odf.sort_values(by=['RGIID', 'SEGMENT_ID'])
    odf.crs = to_crs
    # Sanity checks to avoid bad surprises
    gtype = np.array([g.geom_type for g in odf.geometry])
    if 'GeometryCollection' in gtype:
        errdf = odf.loc[gtype == 'GeometryCollection']
        with warnings.catch_warnings():
            # errdf.length warns because of use of wgs84
            warnings.filterwarnings("ignore", category=UserWarning)
            if not np.all(errdf.length) == 0:
                errdf = errdf.loc[errdf.length > 0]
                raise RuntimeError('Some geometries are non-empty GeometryCollection '
                                   f'at RGI Ids: {errdf.RGIID.values}')
    _write_shape_to_disk(odf, path, to_tar=to_tar)


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
        def _compile_to_netcdf(gdirs, input_filesuffix='',
                               output_filesuffix='',
                               path=True,
                               tmp_file_size=1000,
                               **kwargs):

            if not output_filesuffix:
                output_filesuffix = input_filesuffix

            gdirs = tolist(gdirs)
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


@entity_task(log)
def merge_consecutive_run_outputs(gdir,
                                  input_filesuffix_1=None,
                                  input_filesuffix_2=None,
                                  output_filesuffix=None,
                                  delete_input=False):
    """Merges the output of two model_diagnostics files into one.

    It assumes that the last time of file1 is equal to the first time of file2.

    Parameters
    ----------
    gdir : the glacier directory
    input_filesuffix_1 : str
        how to recognize the first file
    input_filesuffix_2 : str
        how to recognize the second file
    output_filesuffix : str
        where to write the output (default: no suffix)

    Returns
    -------
    The merged dataset
    """

    # Read in the input files and check
    fp1 = gdir.get_filepath('model_diagnostics', filesuffix=input_filesuffix_1)
    with xr.open_dataset(fp1) as ds:
        ds1 = ds.load()
    fp2 = gdir.get_filepath('model_diagnostics', filesuffix=input_filesuffix_2)
    with xr.open_dataset(fp2) as ds:
        ds2 = ds.load()
    if ds1.time[-1] != ds2.time[0]:
        raise InvalidWorkflowError('The two files are incompatible by time')

    # Samity check for all variables as well
    for v in ds1:
        if not np.all(np.isfinite(ds1[v].data[-1])):
            # This is the last year of hydro output - we will discard anyway
            continue
        if np.allclose(ds1[v].data[-1], ds2[v].data[0]):
            # This means that we're OK - the two match
            continue

        # This has to be a bucket of some sort, probably snow or calving
        if len(ds2[v].data.shape) == 1:
            if ds2[v].data[0] != 0:
                raise InvalidWorkflowError('The two files seem incompatible '
                                           f'by data on variable : {v}')
            bucket = ds1[v].data[-1]
        elif len(ds2[v].data.shape) == 2:
            if ds2[v].data[0, 0] != 0:
                raise InvalidWorkflowError('The two files seem incompatible '
                                           f'by data on variable : {v}')
            bucket = ds1[v].data[-1, -1]
        # Carry it to the rest
        ds2[v] = ds2[v] + bucket

    # Merge by removing the last step of file 1 and delete the files if asked
    out_ds = xr.concat([ds1.isel(time=slice(0, -1)), ds2], dim='time')
    if delete_input:
        os.remove(fp1)
        os.remove(fp2)
    # Write out and return
    fp = gdir.get_filepath('model_diagnostics', filesuffix=output_filesuffix)
    out_ds.to_netcdf(fp)
    return out_ds


@global_task(log)
@compile_to_netcdf(log)
def compile_run_output(gdirs, path=True, input_filesuffix='',
                       use_compression=True):
    """Compiles the output of the model runs of several gdirs into one file.

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

    # To find the longest time, we have to open all files unfortunately, we
    # also create a list of all data variables (in case not all files contain
    # the same data variables), and finally we decide on the name of "3d"
    # variables in case we have daily
    time_info = {}
    time_keys = ['hydro_year', 'hydro_month', 'calendar_year', 'calendar_month']
    allowed_data_vars = ['volume_m3', 'volume_bsl_m3', 'volume_bwl_m3',
                         'volume_m3_min_h',  # only here for back compatibility
                         # as it is a variable in gdirs v1.6 2023.1
                         'area_m2', 'area_m2_min_h', 'length_m', 'calving_m3',
                         'calving_rate_myr', 'off_area',
                         'on_area', 'model_mb', 'is_fixed_geometry_spinup']
    for gi in range(10):
        allowed_data_vars += [f'terminus_thick_{gi}']
    # this hydro variables can be _monthly or _daily
    hydro_vars = ['melt_off_glacier', 'melt_on_glacier',
                  'liq_prcp_off_glacier', 'liq_prcp_on_glacier',
                  'snowfall_off_glacier', 'snowfall_on_glacier',
                  'melt_residual_off_glacier', 'melt_residual_on_glacier',
                  'snow_bucket', 'residual_mb']
    for v in hydro_vars:
        allowed_data_vars += [v]
        allowed_data_vars += [v + '_monthly']
        allowed_data_vars += [v + '_daily']
    data_vars = {}
    name_2d_dim = 'month_2d'
    contains_3d_data = False
    for gd in gdirs:
        fp = gd.get_filepath('model_diagnostics', filesuffix=input_filesuffix)
        try:
            with ncDataset(fp) as ds:
                time = ds.variables['time'][:]
                if 'time' not in time_info:
                    time_info['time'] = time
                    for cn in time_keys:
                        time_info[cn] = ds.variables[cn][:]
                else:
                    # Here we may need to append or add stuff
                    ot = time_info['time']
                    if time[0] > ot[-1] or ot[-1] < time[0]:
                        raise InvalidWorkflowError('Trying to compile output '
                                                   'without overlap.')
                    if time[-1] > ot[-1]:
                        p = np.nonzero(time == ot[-1])[0][0] + 1
                        time_info['time'] = np.append(ot, time[p:])
                        for cn in time_keys:
                            time_info[cn] = np.append(time_info[cn],
                                                      ds.variables[cn][p:])
                    if time[0] < ot[0]:
                        p = np.nonzero(time == ot[0])[0][0]
                        time_info['time'] = np.append(time[:p], ot)
                        for cn in time_keys:
                            time_info[cn] = np.append(ds.variables[cn][:p],
                                                      time_info[cn])

                # check if their are new data variables and add them
                for vn in ds.variables:
                    # exclude time variables
                    if vn in ['month_2d', 'calendar_month_2d',
                              'hydro_month_2d']:
                        name_2d_dim = 'month_2d'
                        contains_3d_data = True
                    elif vn in ['day_2d', 'calendar_day_2d', 'hydro_day_2d']:
                        name_2d_dim = 'day_2d'
                        contains_3d_data = True
                    elif vn in allowed_data_vars:
                        # check if data variable is new
                        if vn not in data_vars.keys():
                            data_vars[vn] = dict()
                            data_vars[vn]['dims'] = ds.variables[vn].dimensions
                            data_vars[vn]['attrs'] = dict()
                            for attr in ds.variables[vn].ncattrs():
                                if attr not in ['_FillValue', 'coordinates',
                                                'dtype']:
                                    data_vars[vn]['attrs'][attr] = getattr(
                                        ds.variables[vn], attr)
                    elif vn not in ['time'] + time_keys:
                        # This check has future developments in mind.
                        # If you end here it means the current data variable is
                        # not under the allowed_data_vars OR not under the
                        # defined time dimensions. If it is a new data variable
                        # add it to allowed_data_vars above (also add it to
                        # test_compile_run_output). If it is a new dimension
                        # handle it in the if/elif statements.
                        raise InvalidParamsError(f'The data variable "{vn}" '
                                                 'is not known. Is it new or '
                                                 'is it a new dimension? '
                                                 'Check comment above this '
                                                 'raise for more info!')

            # If this worked, keep it as template
            ppath = fp
        except FileNotFoundError:
            pass

    if 'time' not in time_info:
        raise RuntimeError('Found no valid glaciers!')

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
        time = time_info['time']
        ds.coords['time'] = ('time', time)
        ds['time'].attrs['description'] = 'Floating year'
        # New coord
        ds.coords['rgi_id'] = ('rgi_id', rgi_ids)
        ds['rgi_id'].attrs['description'] = 'RGI glacier identifier'
        # This is just taken from there
        for cn in ['hydro_year', 'hydro_month',
                   'calendar_year', 'calendar_month']:
            ds.coords[cn] = ('time', time_info[cn])
            ds[cn].attrs['description'] = ds_diag[cn].attrs['description']

        # Prepare the 2D variables
        shape = (len(time), len(rgi_ids))
        out_2d = dict()
        for vn in data_vars:
            if name_2d_dim in data_vars[vn]['dims']:
                continue
            var = dict()
            var['data'] = np.full(shape, np.nan)
            var['attrs'] = data_vars[vn]['attrs']
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

        # Maybe 3D?
        out_3d = dict()
        if contains_3d_data:
            # We have some 3d vars
            month_2d = ds_diag[name_2d_dim]
            ds.coords[name_2d_dim] = (name_2d_dim, month_2d.data)
            cn = f'calendar_{name_2d_dim}'
            ds.coords[cn] = (name_2d_dim, ds_diag[cn].values)

            shape = (len(time), len(month_2d), len(rgi_ids))
            for vn in data_vars:
                if name_2d_dim not in data_vars[vn]['dims']:
                    continue
                var = dict()
                var['data'] = np.full(shape, np.nan)
                var['attrs'] = data_vars[vn]['attrs']
                out_3d[vn] = var

    # Read out
    for i, gdir in enumerate(gdirs):
        try:
            ppath = gdir.get_filepath('model_diagnostics',
                                      filesuffix=input_filesuffix)
            with ncDataset(ppath) as ds_diag:
                it = ds_diag.variables['time'][:]
                a = np.nonzero(time == it[0])[0][0]
                b = np.nonzero(time == it[-1])[0][0] + 1
                for vn, var in out_2d.items():
                    # try statement if some data variables not in all files
                    try:
                        var['data'][a:b, i] = ds_diag.variables[vn][:]
                    except KeyError:
                        pass
                for vn, var in out_3d.items():
                    # try statement if some data variables not in all files
                    try:
                        var['data'][a:b, :, i] = ds_diag.variables[vn][:]
                    except KeyError:
                        pass
                for vn, var in out_1d.items():
                    var['data'][i] = ds_diag.getncattr(vn)
        except FileNotFoundError:
            pass

    # To xarray
    for vn, var in out_2d.items():
        # Backwards compatibility - to remove one day...
        for r in ['_m3', '_m2', '_myr', '_m']:
            # Order matters
            vn = regexp.sub(r + '$', '', vn)
        ds[vn] = (('time', 'rgi_id'), var['data'])
        ds[vn].attrs = var['attrs']
    for vn, var in out_3d.items():
        ds[vn] = (('time', name_2d_dim, 'rgi_id'), var['data'])
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


@global_task(log)
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
        sm = cfg.PARAMS['hydro_month_' + pgdir.hemisphere]
        hyrs, hmonths = calendardate_to_hydrodate(cyrs, cmonths, start_month=sm)
        time = date_to_floatyear(cyrs, cmonths)

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
    ds.coords['calendar_year'] = ('time', cyrs.data)
    ds.coords['calendar_month'] = ('time', cmonths.data)
    ds.coords['hydro_year'] = ('time', hyrs)
    ds.coords['hydro_month'] = ('time', hmonths)
    ds['time'].attrs['description'] = 'Floating year'
    ds['rgi_id'].attrs['description'] = 'RGI glacier identifier'
    ds['hydro_year'].attrs['description'] = 'Hydrological year'
    ds['hydro_month'].attrs['description'] = 'Hydrological month'
    ds['calendar_year'].attrs['description'] = 'Calendar year'
    ds['calendar_month'].attrs['description'] = 'Calendar month'

    shape = (len(time), len(rgi_ids))
    temp = np.zeros(shape) * np.NaN
    prcp = np.zeros(shape) * np.NaN

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
        encoding = {v: enc_var for v in vars}
        ds.to_netcdf(path, encoding=encoding)
    return ds


@global_task(log)
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


@global_task(log)
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
def glacier_statistics(gdir, inversion_only=False, apply_func=None):
    """Gather as much statistics as possible about this glacier.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    inversion_only : bool
        if one wants to summarize the inversion output only (including calving)
    apply_func : function
        if one wants to summarize further information about a glacier, set
        this kwarg to a function that accepts a glacier directory as first
        positional argument, and the directory to fill in with data as
        second argument. The directory should only store scalar values (strings,
        float, int)
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
            # Grid stuff
            d['grid_dx'] = gdir.grid.dx
            d['grid_nx'] = gdir.grid.nx
            d['grid_ny'] = gdir.grid.ny
        except BaseException:
            pass

        try:
            # Geom stuff
            outline = gdir.read_shapefile('outlines')
            d['geometry_type'] = outline.type.iloc[0]
            d['geometry_is_valid'] = outline.is_valid.iloc[0]
            d['geometry_area_km2'] = outline.to_crs({'proj': 'cea'}).area.iloc[0] * 1e-6
        except BaseException:
            pass

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
                mask = nc.variables['glacier_mask'][:] == 1
                topo = nc.variables['topo'][:][mask]
            d['dem_mean_elev'] = np.mean(topo)
            d['dem_med_elev'] = np.median(topo)
            d['dem_min_elev'] = np.min(topo)
            d['dem_max_elev'] = np.max(topo)
        except BaseException:
            pass

        try:
            # Ext related stuff
            fpath = gdir.get_filepath('gridded_data')
            with ncDataset(fpath) as nc:
                ext = nc.variables['glacier_ext'][:] == 1
                mask = nc.variables['glacier_mask'][:] == 1
                topo = nc.variables['topo'][:]
            d['dem_max_elev_on_ext'] = np.max(topo[ext])
            d['dem_min_elev_on_ext'] = np.min(topo[ext])
            a = np.sum(mask & (topo > d['dem_max_elev_on_ext']))
            d['dem_perc_area_above_max_elev_on_ext'] = a / np.sum(mask)
            # Terminus loc
            j, i = np.nonzero((topo[ext].min() == topo) & ext)
            lon, lat = gdir.grid.ij_to_crs(i[0], j[0], crs=salem.wgs84)
            d['terminus_lon'] = lon
            d['terminus_lat'] = lat
        except BaseException:
            pass

        try:
            # Centerlines
            cls = gdir.read_pickle('centerlines')
            longest = 0.
            for cl in cls:
                longest = np.max([longest, cl.dis_on_line[-1]])
            d['n_centerlines'] = len(cls)
            d['longest_centerline_km'] = longest * gdir.grid.dx / 1000.
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
            # climate
            info = gdir.get_climate_info()
            for k, v in info.items():
                d[k] = v
        except BaseException:
            pass

        try:
            # MB calib
            mb_calib = gdir.read_json('mb_calib')
            for k, v in mb_calib.items():
                if np.isscalar(v):
                    d[k] = v
                else:
                    for k2, v2 in v.items():
                        d[k2] = v2
        except BaseException:
            pass

        if apply_func:
            # User defined statistics
            try:
                apply_func(gdir, d)
            except BaseException:
                pass

    return d


@global_task(log)
def compile_glacier_statistics(gdirs, filesuffix='', path=True,
                               inversion_only=False, apply_func=None):
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
    apply_func : function
        if one wants to summarize further information about a glacier, set
        this kwarg to a function that accepts a glacier directory as first
        positional argument, and the directory to fill in with data as
        second argument. The directory should only store scalar values (strings,
        float, int).
        !Careful! For multiprocessing, the function cannot be located at the
        top level, i.e. you may need to import it from a module for this to work,
        or from a dummy class (https://stackoverflow.com/questions/8804830)
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(glacier_statistics, gdirs,
                                 apply_func=apply_func,
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
def read_glacier_hypsometry(gdir):
    """Utility function to read the glacier hypsometry in the folder.

    Parameters
    ----------
    gdir :  :py:class:`oggm.GlacierDirectory` object
        the glacier directory to process

    Returns
    -------
    the dataframe
    """
    try:
        out = pd.read_csv(gdir.get_filepath('hypsometry')).iloc[0]
    except:
        out = pd.Series({'rgi_id': gdir.rgi_id})
    return out


@global_task(log)
def compile_glacier_hypsometry(gdirs, filesuffix='', path=True,
                               add_column=None):
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
    add_column : tuple
        if you feel like adding a key - value pair to the compiled dataframe
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(read_glacier_hypsometry, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')
    if add_column is not None:
        out[add_column[0]] = add_column[1]
    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('glacier_hypsometry' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)
    return out


@global_task(log)
def compile_fixed_geometry_mass_balance(gdirs, filesuffix='',
                                        path=True, csv=False,
                                        use_inversion_flowlines=True,
                                        ys=None, ye=None, years=None,
                                        climate_filename='climate_historical',
                                        climate_input_filesuffix='',
                                        temperature_bias=None,
                                        precipitation_factor=None):

    """Compiles a table of specific mass balance timeseries for all glaciers.

    The file is stored in a hdf file (not csv) per default. Use pd.read_hdf
    to open it.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    csv : bool
        Set to store the data in csv instead of hdf.
    use_inversion_flowlines : bool
        whether to use the inversion flowlines or the model flowlines
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        default is None and means that the precipitation factor from the
        calibration is applied which is cfg.PARAMS['prcp_fac']
    """

    from oggm.workflow import execute_entity_task
    from oggm.core.massbalance import fixed_geometry_mass_balance

    out_df = execute_entity_task(fixed_geometry_mass_balance, gdirs,
                                 use_inversion_flowlines=use_inversion_flowlines,
                                 ys=ys, ye=ye, years=years, climate_filename=climate_filename,
                                 climate_input_filesuffix=climate_input_filesuffix,
                                 temperature_bias=temperature_bias,
                                 precipitation_factor=precipitation_factor)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.NaN)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'fixed_geometry_mass_balance' + filesuffix)
            if csv:
                out.to_csv(fpath + '.csv')
            else:
                out.to_hdf(fpath + '.hdf', key='df')
        else:
            ext = os.path.splitext(path)[-1]
            if ext.lower() == '.csv':
                out.to_csv(path)
            elif ext.lower() == '.hdf':
                out.to_hdf(path, key='df')
    return out


@global_task(log)
def compile_ela(gdirs, filesuffix='', path=True, csv=False, ys=None, ye=None,
                years=None, climate_filename='climate_historical', temperature_bias=None,
                precipitation_factor=None, climate_input_filesuffix='',
                mb_model_class=None):
    """Compiles a table of ELA timeseries for all glaciers for a given years,
    using the mb_model_class (default MonthlyTIModel).

    The file is stored in a hdf file (not csv) per default. Use pd.read_hdf
    to open it.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    csv: bool
        Set to store the data in csv instead of hdf.
    ys : int
        start year
    ye : int
        end year
    years : array of ints
        override ys and ye with the years of your choice
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix : str
        filesuffix for the input climate file
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        default is None and means that the precipitation factor from the
        calibration is applied which is cfg.PARAMS['prcp_fac']
    mb_model_class : MassBalanceModel class
        the MassBalanceModel class to use, default is MonthlyTIModel
    """
    from oggm.workflow import execute_entity_task
    from oggm.core.massbalance import compute_ela, MonthlyTIModel

    if mb_model_class is None:
        mb_model_class = MonthlyTIModel

    out_df = execute_entity_task(compute_ela, gdirs, ys=ys, ye=ye, years=years,
                                 climate_filename=climate_filename,
                                 climate_input_filesuffix=climate_input_filesuffix,
                                 temperature_bias=temperature_bias,
                                 precipitation_factor=precipitation_factor,
                                 mb_model_class=mb_model_class)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.NaN)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'ELA' + filesuffix)
            if csv:
                out.to_csv(fpath + '.csv')
            else:
                out.to_hdf(fpath + '.hdf', key='df')
        else:
            ext = os.path.splitext(path)[-1]
            if ext.lower() == '.csv':
                out.to_csv(path)
            elif ext.lower() == '.hdf':
                out.to_hdf(path, key='df')
    return out


@entity_task(log)
def climate_statistics(gdir, add_climate_period=1995, halfsize=15,
                       input_filesuffix=''):
    """Gather as much statistics as possible about this glacier.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Important note: the climate is extracted from the mass-balance model and
    is therefore "corrected" according to the mass-balance calibration scheme
    (e.g. the precipitation factor and the temp bias correction). For more
    flexible information about the raw climate data, use `compile_climate_input`
    or `raw_climate_statistics`.

    Parameters
    ----------
    add_climate_period : int or list of ints
        compile climate statistics for the halfsize*2 + 1 yrs period
        around the selected date.
    halfsize : int
        the half size of the window
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

        # Climate and MB at specified dates
        add_climate_period = tolist(add_climate_period)
        for y0 in add_climate_period:
            try:
                fs = '{}-{}'.format(y0 - halfsize, y0 + halfsize)
                mbcl = ConstantMassBalance
                mbmod = MultipleFlowlineMassBalance(gdir, mb_model_class=mbcl,
                                                    y0=y0, halfsize=halfsize,
                                                    use_inversion_flowlines=True,
                                                    input_filesuffix=input_filesuffix)
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
                t, tm, p, ps = mbmod.flowline_mb_models[0].get_annual_climate(
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


@entity_task(log)
def raw_climate_statistics(gdir, add_climate_period=1995, halfsize=15,
                           input_filesuffix=''):
    """Gather as much statistics as possible about this glacier.

    This is like "climate_statistics" but without relying on the
    mass-balance model, i.e. closer to the actual data (uncorrected)

    Parameters
    ----------
    add_climate_period : int or list of ints
        compile climate statistics for the 30 yrs period around the selected
        date.
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

    # The rest is less certain
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Climate and MB at specified dates
        add_climate_period = tolist(add_climate_period)

        # get non-corrected winter daily mean prcp (kg m-2 day-1) for
        # the chosen time period
        for y0 in add_climate_period:
            fs = '{}-{}'.format(y0 - halfsize, y0 + halfsize)
            try:
                # get non-corrected winter daily mean prcp (kg m-2 day-1)
                # it is easier to get this directly from the raw climate files
                fp = gdir.get_filepath('climate_historical',
                                       filesuffix=input_filesuffix)
                with xr.open_dataset(fp).prcp as ds_pr:
                    # just select winter months
                    if gdir.hemisphere == 'nh':
                        m_winter = [10, 11, 12, 1, 2, 3, 4]
                    else:
                        m_winter = [4, 5, 6, 7, 8, 9, 10]
                    ds_pr_winter = ds_pr.where(ds_pr['time.month'].isin(m_winter), drop=True)
                    # select the correct year time period
                    ds_pr_winter = ds_pr_winter.sel(time=slice(f'{fs[:4]}-01-01',
                                                               f'{fs[-4:]}-12-01'))
                    # check if we have the full time period
                    n_years = int(fs[-4:]) - int(fs[:4]) + 1
                    assert len(ds_pr_winter.time) == n_years * 7, 'chosen time-span invalid'
                    ds_d_pr_winter_mean = (ds_pr_winter / ds_pr_winter.time.dt.daysinmonth).mean()
                    d[f'{fs}_uncorrected_winter_daily_mean_prcp'] = ds_d_pr_winter_mean.values
            except BaseException:
                pass
    return d


@global_task(log)
def compile_climate_statistics(gdirs, filesuffix='', path=True,
                               add_climate_period=1995,
                               halfsize=15,
                               add_raw_climate_statistics=False,
                               input_filesuffix=''):
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
    input_filesuffix : str
        filesuffix of the used climate_historical file, default is no filesuffix
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(climate_statistics, gdirs,
                                 add_climate_period=add_climate_period,
                                 halfsize=halfsize,
                                 input_filesuffix=input_filesuffix)
    out = pd.DataFrame(out_df).set_index('rgi_id')

    if add_raw_climate_statistics:
        out_df = execute_entity_task(raw_climate_statistics, gdirs,
                                     add_climate_period=add_climate_period,
                                     halfsize=halfsize,
                                     input_filesuffix=input_filesuffix)
        out = out.merge(pd.DataFrame(out_df).set_index('rgi_id'))

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

    This is not parallelized, i.e a bit slow.

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

        # We need at least area and vol to do something
        if 'volume' not in past_ds.data_vars or 'area' not in past_ds.data_vars:
            raise InvalidWorkflowError('Need both volume and area to proceed')

        y0_run = int(past_ds.time[0])
        y1_run = int(past_ds.time[-1])
        if (y1_run - y0_run + 1) != len(past_ds.time):
            raise NotImplementedError('Currently only supports annual outputs')
        y0_clim = int(fixed_geometry_mb_df.index[0])
        y1_clim = int(fixed_geometry_mb_df.index[-1])
        if y0_clim > y0_run or y1_clim < y0_run:
            raise InvalidWorkflowError('Dates do not match.')
        if y1_clim != y1_run - 1:
            raise InvalidWorkflowError('Dates do not match.')
        if len(past_ds.rgi_id) != len(fixed_geometry_mb_df.columns):
            # This might happen if we are testing on new directories
            fixed_geometry_mb_df = fixed_geometry_mb_df[past_ds.rgi_id]
        if len(past_ds.rgi_id) != len(stats_df.index):
            stats_df = stats_df.loc[past_ds.rgi_id]

        # Make sure we agree on order
        df = fixed_geometry_mb_df[past_ds.rgi_id]

        # Output data
        years = np.arange(y0_clim, y1_run+1)
        ods = past_ds.reindex({'time': years})

        # Time
        ods['hydro_year'].data[:] = years
        ods['hydro_month'].data[:] = ods['hydro_month'][-1]
        ods['calendar_year'].data[:] = years
        ods['calendar_month'].data[:] = ods['calendar_month'][-1]
        for vn in ['hydro_year', 'hydro_month', 'calendar_year', 'calendar_month']:
            ods[vn] = ods[vn].astype(int)

        # New vars
        for vn in ['volume', 'volume_m3_min_h', 'volume_bsl', 'volume_bwl',
                   'area', 'area_m2_min_h', 'length', 'calving', 'calving_rate']:
            if vn in ods.data_vars:
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

            # First valid id
            fid = np.argmax(np.isfinite(orig_vol_ts))

            # Add calving to the mix
            try:
                calv_flux = stats_df.loc[rid, 'calving_flux'] * 1e9
                calv_rate = stats_df.loc[rid, 'calving_rate_myr']
            except KeyError:
                calv_flux = 0
                calv_rate = 0
            if not np.isfinite(calv_flux):
                calv_flux = 0
            if not np.isfinite(calv_rate):
                calv_rate = 0

            # Fill area and length which stays constant before date
            orig_area_ts = ods.area_ext.data[:, i]
            orig_area_ts[:fid] = orig_area_ts[fid]

            # We convert SMB to volume
            mb_vol_ts = (mb_ts / rho * orig_area_ts[fid] - calv_flux).cumsum()
            calv_ts = (mb_ts * 0 + calv_flux).cumsum()

            # The -1 is because the volume change is known at end of year
            mb_vol_ts = mb_vol_ts + orig_vol_ts[fid] - mb_vol_ts[fid-1]

            # Now back to netcdf
            ods.volume_fixed_geom_ext.data[1:, i] = mb_vol_ts
            ods.volume_ext.data[1:fid, i] = mb_vol_ts[0:fid-1]
            ods.area_ext.data[:, i] = orig_area_ts

            # Optional variables
            if 'length' in ods.data_vars:
                orig_length_ts = ods.length_ext.data[:, i]
                orig_length_ts[:fid] = orig_length_ts[fid]
                ods.length_ext.data[:, i] = orig_length_ts

            if 'calving' in ods.data_vars:
                orig_calv_ts = ods.calving_ext.data[:, i]
                # The -1 is because the volume change is known at end of year
                calv_ts = calv_ts + orig_calv_ts[fid] - calv_ts[fid-1]
                ods.calving_ext.data[1:fid, i] = calv_ts[0:fid-1]

            if 'calving_rate' in ods.data_vars:
                orig_calv_rate_ts = ods.calving_rate_ext.data[:, i]
                # +1 because calving rate at year 0 is unknown from the dyns model
                orig_calv_rate_ts[:fid+1] = calv_rate
                ods.calving_rate_ext.data[:, i] = orig_calv_rate_ts

            # Extend vol bsl by assuming that % stays constant
            if 'volume_bsl' in ods.data_vars:
                bsl = ods.volume_bsl.data[fid, i] / ods.volume.data[fid, i]
                ods.volume_bsl_ext.data[:fid, i] = bsl * ods.volume_ext.data[:fid, i]
            if 'volume_bwl' in ods.data_vars:
                bwl = ods.volume_bwl.data[fid, i] / ods.volume.data[fid, i]
                ods.volume_bwl_ext.data[:fid, i] = bwl * ods.volume_ext.data[:fid, i]

        # Remove old vars
        for vn in list(ods.data_vars):
            if '_ext' not in vn and 'time' in ods[vn].dims:
                del ods[vn]

        # Rename vars to their old names
        ods = ods.rename(dict((o, o.replace('_ext', ''))
                              for o in ods.data_vars))

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
    fl.is_rectangular = np.ones(fl.nx).astype(bool)
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
    base_dir : str
        path to the base directory
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
        The RGI region ID
    rgi_subregion : str
        The RGI subregion ID
    rgi_version : str
        The RGI version name
    rgi_region_name : str
        The RGI region name
    rgi_subregion_name : str
        The RGI subregion name
    name : str
        The RGI glacier name (if available)
    hemisphere : str
        `nh` or `sh`
    glacier_type : str
        The RGI glacier type ('Glacier', 'Ice cap', 'Perennial snowfield',
        'Seasonal snowfield')
    terminus_type : str
        The RGI terminus type ('Land-terminating', 'Marine-terminating',
        'Lake-terminating', 'Dry calving', 'Regenerated', 'Shelf-terminating')
    is_tidewater : bool
        Is the glacier a calving glacier?
    is_lake_terminating : bool
        Is the glacier a lake terminating glacier?
    is_nominal : bool
        Is the glacier an RGI nominal glacier?
    is_icecap : bool
        Is the glacier an ice cap?
    extent_ll : list
        Extent of the glacier in lon/lat
    logfile : str
        Path to the log file (txt)
    inversion_calving_rate : float
        Calving rate used for the inversion
    grid
    dem_info
    dem_daterange
    intersects_ids
    rgi_area_m2
    rgi_area_km2
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
                _dir = os.path.join(base_dir, rgi_entity[:-6], rgi_entity[:-3],
                                    rgi_entity)
                # Avoid bad surprises
                if os.path.exists(_dir):
                    shutil.rmtree(_dir)
                if from_tar is True:
                    from_tar = _dir + '.tar.gz'
                robust_tar_extract(from_tar, _dir, delete_tar=delete_tar)
                from_tar = False  # to not re-unpack later below
                _shp = os.path.join(_dir, 'outlines.shp')
            else:
                _shp = os.path.join(base_dir, rgi_entity[:-6], rgi_entity[:-3],
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

        is_rgi7 = False
        is_glacier_complex = False
        try:
            self.rgi_id = rgi_entity.rgi_id
            is_rgi7 = True
            try:
                self.glims_id = rgi_entity.glims_id
            except AttributeError:
                # Complex product
                self.glims_id = ''
                is_glacier_complex = True
        except AttributeError:
            # RGI V6
            self.rgi_id = rgi_entity.RGIId
            self.glims_id = rgi_entity.GLIMSId

        # Do we want to use the RGI center point or ours?
        if cfg.PARAMS['use_rgi_area']:
            if is_rgi7:
                self.cenlon = float(rgi_entity.cenlon)
                self.cenlat = float(rgi_entity.cenlat)
            else:
                self.cenlon = float(rgi_entity.CenLon)
                self.cenlat = float(rgi_entity.CenLat)
        else:
            cenlon, cenlat = rgi_entity.geometry.representative_point().xy
            self.cenlon = float(cenlon[0])
            self.cenlat = float(cenlat[0])

        if is_glacier_complex:
            rgi_entity['glac_name'] = ''
            rgi_entity['src_date'] = '2000-01-01 00:00:00'
            rgi_entity['dem_source'] = None
            rgi_entity['term_type'] = 9

        if is_rgi7:
            self.rgi_region = rgi_entity.o1region
            self.rgi_subregion = rgi_entity.o2region
            name = rgi_entity.glac_name
            rgi_datestr = rgi_entity.src_date
            self.rgi_version = '70G'
            self.glacier_type = 'Glacier'
            self.status = 'Glacier'
            ttkeys = {0: 'Land-terminating',
                      1: 'Marine-terminating',
                      2: 'Lake-terminating',
                      3: 'Shelf-terminating',
                      9: 'Not assigned',
                      }
            self.terminus_type = ttkeys[rgi_entity['term_type']]
            if is_glacier_complex:
                self.rgi_version = '70C'
                self.glacier_type = 'Glacier complex'
                self.status = 'Glacier complex'
            self.rgi_dem_source = rgi_entity.dem_source
            self.utm_zone = rgi_entity.utm_zone

            # New attrs
            try:
                self.rgi_termlon = rgi_entity.termlon
                self.rgi_termlat = rgi_entity.termlat
            except AttributeError:
                pass
        else:
            self.rgi_region = '{:02d}'.format(int(rgi_entity.O1Region))
            self.rgi_subregion = f'{self.rgi_region}-{int(rgi_entity.O2Region):02d}'
            name = rgi_entity.Name
            rgi_datestr = rgi_entity.BgnDate

            try:
                # RGI5
                gtype = rgi_entity.GlacType
            except AttributeError:
                # RGI V6
                gtype = [str(rgi_entity.Form), str(rgi_entity.TermType)]

            try:
                # RGI5
                gstatus = rgi_entity.RGIFlag[0]
            except AttributeError:
                # RGI V6
                gstatus = rgi_entity.Status

            rgi_version = self.rgi_id.split('-')[0][-2:]
            if rgi_version not in ['50', '60', '61']:
                raise RuntimeError('RGI Version not supported: '
                                   '{}'.format(self.rgi_version))
            self.rgi_version = rgi_version
            self.rgi_dem_source = None

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

        # remove spurious characters and trailing blanks
        self.name = filter_rgi_name(name)

        # RGI region
        reg_names, subreg_names = parse_rgi_meta(version=self.rgi_version[0])
        reg_name = reg_names.loc[int(self.rgi_region)]

        # RGI V6
        if not isinstance(reg_name, str):
            reg_name = reg_name.values[0]

        self.rgi_region_name = self.rgi_region + ': ' + reg_name
        try:
            subreg_name = subreg_names.loc[self.rgi_subregion]
            # RGI V6
            if not isinstance(subreg_name, str):
                subreg_name = subreg_name.values[0]
            self.rgi_subregion_name = self.rgi_subregion + ': ' + subreg_name
        except KeyError:
            self.rgi_subregion_name = self.rgi_subregion + ': NoName'

        # Decide what is a tidewater glacier
        user = cfg.PARAMS['tidewater_type']
        if user == 1:
            sel = ['Marine-terminating']
        elif user == 2:
            sel = ['Marine-terminating', 'Shelf-terminating']
        elif user == 3:
            sel = ['Marine-terminating', 'Lake-terminating']
        elif user == 4:
            sel = ['Marine-terminating', 'Lake-terminating', 'Shelf-terminating']
        else:
            raise InvalidParamsError("PARAMS['tidewater_type'] not understood")
        self.is_tidewater = self.terminus_type in sel
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
        self.dir = os.path.join(self.base_dir, self.rgi_id[:-6],
                                self.rgi_id[:-3], self.rgi_id)

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
        self._mbprofdf_cte_dh = None

    def __repr__(self):

        summary = ['<oggm.GlacierDirectory>']
        summary += ['  RGI id: ' + self.rgi_id]
        summary += ['  Region: ' + self.rgi_region_name]
        summary += ['  Subregion: ' + self.rgi_subregion_name]
        if self.name:
            summary += ['  Name: ' + self.name]
        summary += ['  Glacier type: ' + str(self.glacier_type)]
        summary += ['  Terminus type: ' + str(self.terminus_type)]
        summary += ['  Status: ' + str(self.status)]
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
        if cfg.PARAMS['map_proj'] == 'utm':
            if entity.get('utm_zone', False):
                # RGI7 has an utm zone
                proj4_str = {'proj': 'utm', 'zone': entity['utm_zone']}
            else:
                # Find it out
                from pyproj.aoi import AreaOfInterest
                from pyproj.database import query_utm_crs_info
                utm_crs_list = query_utm_crs_info(
                    datum_name="WGS 84",
                    area_of_interest=AreaOfInterest(
                        west_lon_degree=self.cenlon,
                        south_lat_degree=self.cenlat,
                        east_lon_degree=self.cenlon,
                        north_lat_degree=self.cenlat,
                    ),
                )
                proj4_str = utm_crs_list[0].code
        elif cfg.PARAMS['map_proj'] == 'tmerc':
            params = dict(name='tmerc', lat_0=0., lon_0=self.cenlon,
                          k=0.9996, x_0=0, y_0=0, datum='WGS84')
            proj4_str = ("+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} "
                         "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**params))
        else:
            raise InvalidParamsError("cfg.PARAMS['map_proj'] must be one of "
                                     "'tmerc', 'utm'.")
        # Reproject
        proj_in = pyproj.Proj("epsg:4326", preserve_units=True)
        proj_out = pyproj.Proj(proj4_str, preserve_units=True)

        # transform geometry to map
        project = partial(transform_proj, proj_in, proj_out)
        geometry = shp_trafo(project, entity['geometry'])
        if len(self.rgi_id) == 23 and (not geometry.is_valid or
                                       type(geometry) != shpg.Polygon):
            # In RGI7 we know that the geometries are valid in the source file,
            # so we have to validate them after projection them as well
            # Try buffer first
            geometry = geometry.buffer(0)
            if not geometry.is_valid:
                correct = recursive_valid_polygons([geometry], crs=proj4_str)
                if len(correct) != 1:
                    raise RuntimeError('Cant correct this geometry')
                geometry = correct[0]
            if type(geometry) != shpg.Polygon:
                raise ValueError(f'{self.rgi_id}: geometry not valid')
        elif not cfg.PARAMS['keep_multipolygon_outlines']:
            geometry = multipolygon_to_polygon(geometry, gdir=self)

        # Save transformed geometry to disk
        entity = entity.copy()
        entity['geometry'] = geometry

        # Do we want to use the RGI area or ours?
        if not cfg.PARAMS['use_rgi_area']:
            # Update Area
            try:
                area = geometry.area * 1e-6
            except:
                area = geometry.area_m2 * 1e-6
            entity['Area'] = area

        # Avoid fiona bug: https://github.com/Toblerity/Fiona/issues/365
        for k, s in entity.items():
            if type(s) in [np.int32, np.int64]:
                entity[k] = int(s)
        towrite = gpd.GeoDataFrame(entity).T.set_geometry('geometry')
        towrite.crs = proj4_str

        # Write shapefile
        self.write_shapefile(towrite, 'outlines')

        # Also transform the intersects if necessary
        gdf = cfg.PARAMS['intersects_gdf']
        if len(gdf) > 0:
            try:
                gdf = gdf.loc[((gdf.RGIId_1 == self.rgi_id) |
                               (gdf.RGIId_2 == self.rgi_id))]
            except AttributeError:
                gdf = gdf.loc[((gdf.rgi_g_id_1 == self.rgi_id) |
                               (gdf.rgi_g_id_2 == self.rgi_id))]

            if len(gdf) > 0:
                gdf = salem.transform_geopandas(gdf, to_crs=proj_out)
                if hasattr(gdf.crs, 'srs'):
                    # salem uses pyproj
                    gdf.crs = gdf.crs.srs
                self.write_shapefile(gdf, 'intersects')
        else:
            # Sanity check
            if cfg.PARAMS['use_intersects'] and not self.rgi_version == '70C':
                raise InvalidParamsError(
                    'You seem to have forgotten to set the '
                    'intersects file for this run. OGGM '
                    'works better with such a file. If you '
                    'know what your are doing, set '
                    "cfg.PARAMS['use_intersects'] = False to "
                    "suppress this error.")

    def grid_from_params(self):
        """If the glacier_grid.json file is lost, reconstruct it."""
        from oggm.core.gis import glacier_grid_params
        utm_proj, nx, ny, ulx, uly, dx = glacier_grid_params(self)
        x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
        return salem.Grid(proj=utm_proj, nxny=(nx, ny), dxdy=(dx, -dx),
                          x0y0=x0y0)

    @lazy_property
    def grid(self):
        """A ``salem.Grid`` handling the georeferencing of the local grid"""
        try:
            return salem.Grid.from_json(self.get_filepath('glacier_grid'))
        except FileNotFoundError:
            raise InvalidWorkflowError('This glacier directory seems to '
                                       'have lost its glacier_grid.json file.'
                                       'Use .grid_from_params(), but make sure'
                                       'that the PARAMS are the ones you '
                                       'want.')

    @lazy_property
    def rgi_area_km2(self):
        """The glacier's RGI area (km2)."""
        try:
            _area = self.read_shapefile('outlines')['Area']
        except OSError:
            raise RuntimeError('No outlines available')
        except KeyError:
            # RGI V7
            _area = self.read_shapefile('outlines')['area_km2']
        return float(_area.iloc[0])

    @lazy_property
    def intersects_ids(self):
        """The glacier's intersects RGI ids."""
        try:
            gdf = self.read_shapefile('intersects')
            try:
                ids = np.append(gdf['RGIId_1'], gdf['RGIId_2'])
            except KeyError:
                ids = np.append(gdf['rgi_g_id_1'], gdf['rgi_g_id_2'])
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

    def get_filepath(self, filename, delete=False, filesuffix='',
                     _deprecation_check=True):
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

    def has_file(self, filename, filesuffix='', _deprecation_check=True):
        """Checks if a file exists.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        filesuffix : str
            append a suffix to the filename (useful for model runs). Note
            that the BASENAME remains same.
        """
        fp = self.get_filepath(filename, filesuffix=filesuffix,
                               _deprecation_check=_deprecation_check)
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
            try:
                out = pickle.load(f)
            except ModuleNotFoundError as err:
                if err.name == "shapely.io":
                    err.msg = "You need shapely version 2.0 or higher for this to work."
                raise

        # Some new attrs to add to old pre-processed directories
        if filename == 'model_flowlines':
            if getattr(out[0], 'map_trafo', None) is None:
                try:
                    # This may fail for very old gdirs
                    grid = self.grid
                except InvalidWorkflowError:
                    return out

                # Add the trafo
                trafo = partial(grid.ij_to_crs, crs=salem.wgs84)
                for fl in out:
                    fl.map_trafo = trafo

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
            pickle.dump(var, f, protocol=4)

    def read_json(self, filename, filesuffix='', allow_empty=False):
        """Reads a JSON file located in the directory.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        filesuffix : str
            append a suffix to the filename (useful for experiments).
        allow_empty : bool
            if True, does not raise an error if the file is not there.

        Returns
        -------
        A dictionary read from the JSON file
        """

        fp = self.get_filepath(filename, filesuffix=filesuffix)
        if allow_empty:
            try:
                with open(fp, 'r') as f:
                    out = json.load(f)
            except FileNotFoundError:
                out = {}
        else:
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
        """Convenience function to read attributes of the historical climate.

        Parameters
        ----------
        input_filesuffix : str
            input_filesuffix of the climate_historical that should be used.
        """
        out = {}
        try:
            f = self.get_filepath('climate_historical',
                                  filesuffix=input_filesuffix)
            with ncDataset(f) as nc:
                out['baseline_climate_source'] = nc.climate_source
                try:
                    out['baseline_yr_0'] = nc.yr_0
                except AttributeError:
                    # needed for back-compatibility before v1.6
                    out['baseline_yr_0'] = nc.hydro_yr_0
                try:
                    out['baseline_yr_1'] = nc.yr_1
                except AttributeError:
                    # needed for back-compatibility before v1.6
                    out['baseline_yr_1'] = nc.hydro_yr_1
                out['baseline_climate_ref_hgt'] = nc.ref_hgt
                out['baseline_climate_ref_pix_lon'] = nc.ref_pix_lon
                out['baseline_climate_ref_pix_lat'] = nc.ref_pix_lat
        except FileNotFoundError:
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
        _write_shape_to_disk(var, fp, to_tar=cfg.PARAMS['use_tar_shapefiles'])

    def write_monthly_climate_file(self, time, prcp, temp,
                                   ref_pix_hgt, ref_pix_lon, ref_pix_lat, *,
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
            (for correction). In practice, it is the same altitude as the
            baseline climate.
        ref_pix_lon : float
            the location of the gridded data's grid point
        ref_pix_lat : float
            the location of the gridded data's grid point
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

            nc.yr_0 = y0
            nc.yr_1 = y1

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
        """Adds reference mass balance data to this glacier.

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
        mb_df.index.name = 'YEAR'
        self._mbdf = mb_df

    def get_ref_mb_data(self, y0=None, y1=None, input_filesuffix=''):
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
        input_filesuffix : str
            input_filesuffix of the climate_historical that should be used
            if y0 and y1 are not given. The default is to take the
            climate_historical without input_filesuffix
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
            ci = self.get_climate_info(input_filesuffix=input_filesuffix)
            if 'baseline_yr_0' not in ci:
                raise InvalidWorkflowError('Please process some climate data '
                                           'before call')
            y0 = ci['baseline_yr_0'] if y0 is None else y0
            y1 = ci['baseline_yr_1'] if y1 is None else y1

        if len(self._mbdf) > 1:
            out = self._mbdf.loc[y0:y1]
        else:
            # Some files are just empty
            out = self._mbdf
        return out.dropna(subset=['ANNUAL_BALANCE'])

    def get_ref_mb_profile(self, input_filesuffix='', constant_dh=False, obs_ratio_needed=0):
        """Get the reference mb profile data from WGMS (if available!).

        Returns None if this glacier has no profile and an Error if it isn't
        a reference glacier at all.

        Parameters
        ----------
        input_filesuffix : str
            input_filesuffix of the climate_historical that should be used. The
            default is to take the climate_historical without input_filesuffix
        constant_dh : boolean
            If set to True, it outputs the MB profiles with a constant step size
            of dh=50m by using interpolation. This can be useful for comparisons
            between years. Default is False which gives the raw
            elevation-dependent point MB
        obs_ratio_needed : float
            necessary relative amount of observations per elevation band in order
            to be included in the MB profile (0<=obs_ratio_needed<=1).
            If obs_ratio_needed set to 0, the output shows all elevation-band
            observations (default is 0).
            When estimating mean MB profiles, it is advisable to set obs_ratio_needed
            to 0.6. E.g. if there are in total 5 years of measurements only those elevation
            bands with at least 3 years of measurements are used. If obs_ratio_needed is not
            0, constant_dh has to be set to True.
        """

        if obs_ratio_needed != 0 and constant_dh is False:
            raise InvalidParamsError('If a filter is applied, you have to set'
                                     ' constant_dh to True')
        if obs_ratio_needed < 0 or obs_ratio_needed > 1:
            raise InvalidParamsError('obs_ratio_needed is the ratio of necessary relative amount'
                                     'of observations per elevation band. It has to be between'
                                     '0 and 1!')

        if self._mbprofdf is None and not constant_dh:
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

        if self._mbprofdf_cte_dh is None and constant_dh:
            flink, mbdatadir = get_wgms_files()
            c = 'RGI{}0_ID'.format(self.rgi_version[0])
            wid = flink.loc[flink[c] == self.rgi_id]
            if len(wid) == 0:
                raise RuntimeError('Not a reference glacier!')
            wid = wid.WGMS_ID.values[0]

            # file
            mbdatadir = os.path.join(os.path.dirname(mbdatadir), 'mb_profiles_constant_dh')
            reff = os.path.join(mbdatadir,
                                'profile_constant_dh_WGMS-{:05d}.csv'.format(wid))
            if not os.path.exists(reff):
                return None
            # list of years
            self._mbprofdf_cte_dh = pd.read_csv(reff, index_col=0)

        ci = self.get_climate_info(input_filesuffix=input_filesuffix)
        if 'baseline_yr_0' not in ci:
            raise RuntimeError('Please process some climate data before call')
        y0 = ci['baseline_yr_0']
        y1 = ci['baseline_yr_1']
        if not constant_dh:
            if len(self._mbprofdf) > 1:
                out = self._mbprofdf.loc[y0:y1]
            else:
                # Some files are just empty
                out = self._mbprofdf
        else:
            if len(self._mbprofdf_cte_dh) > 1:
                out = self._mbprofdf_cte_dh.loc[y0:y1]
                if obs_ratio_needed != 0:
                    # amount of years with any observation
                    n_obs = len(out.index)
                    # amount of years with observations for each elevation band
                    n_obs_h = out.describe().loc['count']
                    # relative amount of observations per elevation band
                    rel_obs_h = n_obs_h / n_obs
                    # select only those elevation bands with a specific ratio
                    # of years with available measurements
                    out = out[rel_obs_h[rel_obs_h >= obs_ratio_needed].index]

            else:
                # Some files are just empty
                out = self._mbprofdf_cte_dh
        out.columns = [float(c) for c in out.columns]
        return out.dropna(axis=1, how='all').dropna(axis=0, how='all')

    def get_ref_length_data(self):
        """Get the glacier length data from P. Leclercq's data base.

         https://folk.uio.no/paulwl/data.php

         For some glaciers only!
         """

        df = pd.read_csv(get_demo_file('rgi_leclercq_links_2014_RGIV6.csv'))
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
        path to the new base directory (should end with "per_glacier"
        most of the time)
    setup : str
        set up you want the copied directory to be useful for. Currently
        supported are 'all' (copy the entire directory), 'inversion'
        (copy the necessary files for the inversion AND the run)
        , 'run' (copy the necessary files for a dynamical run) or 'run/spinup'
        (copy the necessary files and all already conducted model runs, e.g.
        from a dynamic spinup).

    Returns
    -------
    New glacier directories from the copied folders
    """
    base_dir = os.path.abspath(base_dir)
    new_dir = os.path.join(base_dir, gdir.rgi_id[:8], gdir.rgi_id[:11],
                           gdir.rgi_id)
    if setup == 'run':
        paths = ['model_flowlines', 'inversion_params', 'outlines',
                 'mb_calib', 'climate_historical', 'glacier_grid',
                 'gcm_data', 'diagnostics', 'log']
        paths = ('*' + p + '*' for p in paths)
        shutil.copytree(gdir.dir, new_dir,
                        ignore=include_patterns(*paths))
    elif setup == 'inversion':
        paths = ['inversion_params', 'downstream_line', 'outlines',
                 'inversion_flowlines', 'glacier_grid', 'diagnostics',
                 'mb_calib', 'climate_historical', 'gridded_data',
                 'gcm_data', 'log']
        paths = ('*' + p + '*' for p in paths)
        shutil.copytree(gdir.dir, new_dir,
                        ignore=include_patterns(*paths))
    elif setup == 'run/spinup':
        paths = ['model_flowlines', 'inversion_params', 'outlines',
                 'mb_calib', 'climate_historical', 'glacier_grid',
                 'gcm_data', 'diagnostics', 'log', 'model_run',
                 'model_diagnostics', 'model_geometry']
        paths = ('*' + p + '*' for p in paths)
        shutil.copytree(gdir.dir, new_dir,
                        ignore=include_patterns(*paths))
    elif setup == 'all':
        shutil.copytree(gdir.dir, new_dir)
    else:
        raise ValueError('setup not understood: {}'.format(setup))
    return GlacierDirectory(gdir.rgi_id, base_dir=base_dir)


def initialize_merged_gdir(main, tribs=[], glcdf=None,
                           filename='climate_historical',
                           input_filesuffix='',
                           dem_source=None):
    """Creates a new GlacierDirectory if tributaries are merged to a glacier

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
    dem_source: str
        the DEM source to use
    Returns
    -------
    merged : oggm.GlacierDirectory
        the new GDir
    """
    from oggm.core.gis import define_glacier_region, merged_glacier_masks

    # If its a dict, select the relevant ones
    if isinstance(tribs, dict):
        tribs = tribs[main.rgi_id]
    # make sure tributaries are iterable
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
    define_glacier_region(merged, entity=maindf.loc[idx].iloc[0],
                          source=dem_source)

    # write gridded data and geometries for visualization
    merged_glacier_masks(merged, merged_geometry)

    # reset dx method
    cfg.PARAMS['grid_dx_method'] = dx_method
    cfg.PARAMS['fixed_dx'] = dx_spacing

    # copy main climate file, climate info and calib to new gdir
    climfilename = filename + '_' + main.rgi_id + input_filesuffix + '.nc'
    climfile = os.path.join(merged.dir, climfilename)
    shutil.copyfile(main.get_filepath(filename, filesuffix=input_filesuffix),
                    climfile)
    _mufile = os.path.basename(merged.get_filepath('mb_calib')).split('.')
    mufile = _mufile[0] + '_' + main.rgi_id + '.' + _mufile[1]
    shutil.copyfile(main.get_filepath('mb_calib'),
                    os.path.join(merged.dir, mufile))

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
        # second argument for RGI7 naming convention
        if not ((len(bname) == 11 and bname[-3] == '.') or
                (len(bname) == 20 and bname[-3] == '-')):
            continue
        opath = dirname + '.tar'
        with tarfile.open(opath, 'w') as tar:
            tar.add(dirname, arcname=os.path.basename(dirname))
        if delete:
            to_delete.append(dirname)

    for dirname in to_delete:
        shutil.rmtree(dirname)
