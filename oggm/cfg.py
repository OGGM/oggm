"""  Configuration file and options

A number of globals are defined here to be available everywhere.
"""
import logging
import os
import shutil
import sys
import glob
from collections import OrderedDict
from distutils.util import strtobool
import warnings

import numpy as np
import pandas as pd
try:
    from scipy.signal.windows import gaussian
except AttributeError:
    # Old scipy
    from scipy.signal import gaussian
from configobj import ConfigObj, ConfigObjError
try:
    import geopandas as gpd
except ImportError:
    pass
try:
    import salem
except ImportError:
    pass

from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

# Local logger
log = logging.getLogger(__name__)

# Path to the cache directory
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.oggm')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
# Path to the config file
CONFIG_FILE = os.path.join(os.path.expanduser('~'), '.oggm_config')

# config was changed, indicates that multiprocessing needs a reset
CONFIG_MODIFIED = False

# Share state across processes
DL_VERIFIED = dict()
DEM_SOURCE_TABLE = dict()
DATA = dict()

# Machine epsilon
FLOAT_EPS = np.finfo(float).eps


class DocumentedDict(dict):
    """Quick "magic" to document the BASENAMES entries."""

    def __init__(self):
        self._doc = dict()

    def _set_key(self, key, value, docstr=''):
        if key in self:
            raise ValueError('Cannot overwrite a key.')
        dict.__setitem__(self, key, value)
        self._doc[key] = docstr

    def __setitem__(self, key, value):
        # Overrides the original dic to separate value and documentation
        global CONFIG_MODIFIED
        try:
            self._set_key(key, value[0], docstr=value[1])
            CONFIG_MODIFIED = True
        except BaseException:
            raise ValueError('DocumentedDict accepts only tuple of len 2')

    def info_str(self, key):
        """Info string for the documentation."""
        return '    {}'.format(self[key]) + '\n' + '        ' + self._doc[key]

    def doc_str(self, key):
        """Info string for the documentation."""
        return '        {}'.format(self[key]) + '\n' + '            ' + \
               self._doc[key]


class ResettingOrderedDict(OrderedDict):
    """OrderedDict wrapper that resets our multiprocessing on set"""

    def __setitem__(self, key, value):
        global CONFIG_MODIFIED
        OrderedDict.__setitem__(self, key, value)
        CONFIG_MODIFIED = True


class PathOrderedDict(ResettingOrderedDict):
    """Quick "magic" to be sure that paths are expanded correctly."""

    def __setitem__(self, key, value):
        # Overrides the original dic to expand the path
        try:
            value = os.path.expanduser(value)
        except AttributeError:
            raise InvalidParamsError('The value you are trying to set does '
                                     'not seem to be a valid path: '
                                     '{}'.format(value))

        ResettingOrderedDict.__setitem__(self, key, value)


class ParamsLoggingDict(ResettingOrderedDict):
    """Quick "magic" to log the parameter changes by the user."""

    do_log = False

    def __setitem__(self, key, value):
        # Overrides the original dic to log the change
        if self.do_log:
            self._log_param_change(key, value)
        ResettingOrderedDict.__setitem__(self, key, value)

    def _log_param_change(self, key, value):

        prev = self.get(key)
        if prev is None:
            if key not in ['prcp_fac']:
                log.workflow('WARNING: adding an unknown parameter '
                             '`{}`:`{}` to PARAMS.'.format(key, value))
            return

        if prev == value:
            return

        if key == 'use_multiprocessing':
            msg = 'ON' if value else 'OFF'
            log.workflow('Multiprocessing switched {} '.format(msg) +
                         'after user settings.')
            return

        if key == 'mp_processes':
            if value == -1:
                import multiprocessing
                value = multiprocessing.cpu_count()
                if PARAMS.get('use_multiprocessing', False):
                    log.workflow('Multiprocessing: using all available '
                                 'processors (N={})'.format(value))
            else:
                if PARAMS.get('use_multiprocessing', False):
                    log.workflow('Multiprocessing: using the requested number '
                                 'of processors (N={})'.format(value))
            return

        log.workflow("PARAMS['{}'] changed from `{}` to `{}`.".format(key,
                                                                      prev,
                                                                      value))


# Globals
IS_INITIALIZED = False
PARAMS = ParamsLoggingDict()
PATHS = PathOrderedDict()
BASENAMES = DocumentedDict()
LRUHANDLERS = ResettingOrderedDict()

# Constants
SEC_IN_YEAR = 365*24*3600
SEC_IN_DAY = 24*3600
SEC_IN_HOUR = 3600
SEC_IN_MONTH = 2628000
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

G = 9.80665  # gravity

GAUSSIAN_KERNEL = dict()
for ks in [5, 7, 9]:
    kernel = gaussian(ks, 1)
    GAUSSIAN_KERNEL[ks] = kernel / kernel.sum()

_doc = ('A geotiff file containing the DEM (reprojected into the local grid).'
        'This DEM is not smoothed or gap files, and is the closest to the '
        'original DEM source.')
BASENAMES['dem'] = ('dem.tif', _doc)

_doc = ('A glacier mask geotiff file with the same extend and projection as '
        'the `dem.tif`. This geotiff has value 1 at glaciated grid points and '
        ' value 0 at unglaciated points.')
BASENAMES['glacier_mask'] = ('glacier_mask.tif', _doc)

_doc = ('The glacier outlines in the local map projection '
        '(Transverse Mercator or UTM).')
BASENAMES['outlines'] = ('outlines.shp', _doc)

_doc = 'The glacier intersects in the local map projection.'
BASENAMES['intersects'] = ('intersects.shp', _doc)

_doc = ('Each flowline has a catchment area computed from flow routing '
        'algorithms: this shapefile stores the catchment outlines (in the '
        'local map projection).')
BASENAMES['flowline_catchments'] = ('flowline_catchments.shp', _doc)

_doc = ('The intersections between catchments (shapefile) in the local map '
        'projection.')
BASENAMES['catchments_intersects'] = ('catchments_intersects.shp', _doc)

_doc = 'A ``salem.Grid`` handling the georeferencing of the local grid.'
BASENAMES['glacier_grid'] = ('glacier_grid.json', _doc)

_doc = ('A dictionary containing runtime diagnostics useful for debugging or '
        'logging of run parameters.')
BASENAMES['diagnostics'] = ('diagnostics.json', _doc)

_doc = ('A netcdf file containing several gridded data variables such as '
        'topography, the glacier masks, the interpolated 2D glacier bed, '
        'and more. This is for static, non time-dependant data.')
BASENAMES['gridded_data'] = ('gridded_data.nc', _doc)

_doc = ('A csv file containing ice thickness observations from the GlaThiDa '
        'database. Only available when added from the shop, and only for about 2800 '
        'glaciers worldwide.')
BASENAMES['glathida_data'] = ('glathida_data.csv', _doc)

_doc = ('A netcdf file containing gridded data variables which are time '
        'dependant. It has the same coordinates as `gridded_data`.')
BASENAMES['gridded_simulation'] = ('gridded_simulation.nc', _doc)

_doc = ('A dictionary containing the shapely.Polygons of a glacier. The '
        '"polygon_hr" entry contains the geometry transformed to the local '
        'grid in (i, j) coordinates, while the "polygon_pix" entry contains '
        'the geometries transformed into the coarse grid (the i, j elements '
        'are integers). The "polygon_area" entry contains the area of the '
        'polygon as computed by Shapely. The "catchment_indices" entry'
        'contains a list of len `n_centerlines`, each element containing '
        'a numpy array of the indices in the glacier grid which represent '
        'the centerlines catchment area.')
BASENAMES['geometries'] = ('geometries.pkl', _doc)

_doc = ('A dictionary containing the downstream line geometry as well as the '
        'bed shape computed from a parabolic fit.')
BASENAMES['downstream_line'] = ('downstream_line.pkl', _doc)

_doc = 'A text file with the source of the topo file (GIMP, SRTM, ...).'
BASENAMES['dem_source'] = ('dem_source.txt', _doc)

_doc = ('A hypsometry file computed by OGGM and provided in the same format '
        'as the RGI (useful for diagnostics).')
BASENAMES['hypsometry'] = ('hypsometry.csv', _doc)

_doc = 'A list of :py:class:`oggm.Centerline` instances, sorted by flow order.'
BASENAMES['centerlines'] = ('centerlines.pkl', _doc)

_doc = ('A "better" version of the centerlines, now on a regular spacing '
        'i.e., not on the gridded (i, j) indices. The tails of the '
        'tributaries are cut out to make more realistic junctions. '
        'They are now "1.5D" i.e., with a width.')
BASENAMES['inversion_flowlines'] = ('inversion_flowlines.pkl', _doc)

_doc = 'The historical monthly climate timeseries stored in a netCDF file.'
BASENAMES['climate_historical'] = ('climate_historical.nc', _doc)

# so far, this is only ERA5 or E5E5 daily and does not work with the default
# OGGM mass balance module, only with sandbox
_doc = ('The historical daily climate timeseries stored in a netCDF file.'
        '(only temperature is really changing on daily basis,'
        'precipitation is just assumed constant for every day')
BASENAMES['climate_historical_daily'] = ('climate_historical_daily.nc', _doc)

_doc = "A dict containing the glacier's mass balance calibration parameters."
BASENAMES['mb_calib'] = ('mb_calib.json', _doc)

_doc = 'The monthly GCM climate timeseries stored in a netCDF file.'
BASENAMES['gcm_data'] = ('gcm_data.nc', _doc)

_doc = 'List of dicts containing the data needed for the inversion.'
BASENAMES['inversion_input'] = ('inversion_input.pkl', _doc)

_doc = 'List of dicts containing the output data from the inversion.'
BASENAMES['inversion_output'] = ('inversion_output.pkl', _doc)

_doc = 'List of flowlines ready to be run by the model.'
BASENAMES['model_flowlines'] = ('model_flowlines.pkl', _doc)

_doc = ('When using a linear mass balance for the inversion, this dict stores '
        'the optimal ela_h and grad.')
BASENAMES['linear_mb_params'] = ('linear_mb_params.pkl', _doc)

_doc = ('A netcdf file containing enough information to reconstruct the '
        'entire flowline glacier geometry along the run (can be expensive '
        'in disk space).')
BASENAMES['model_geometry'] = ('model_geometry.nc', _doc)

_doc = ('A netcdf file containing the model diagnostics (volume, '
        'mass balance, length...).')
BASENAMES['model_diagnostics'] = ('model_diagnostics.nc', _doc)

_doc = ('A group netcdf file containing the model diagnostics (volume, '
        'thickness, velocity...) along the flowlines (thus much heavier).'
        'Netcdf groups = fl_{i}, with i between 0 and n_flowlines - 1')
BASENAMES['fl_diagnostics'] = ('fl_diagnostics.nc', _doc)

_doc = "A table containing the Huss&Farinotti 2012 squeezed flowlines."
BASENAMES['elevation_band_flowline'] = ('elevation_band_flowline.csv', _doc)


def set_logging_config(logging_level='INFO'):
    """Set the global logger parameters.

    Logging levels:

    DEBUG
        Print detailed information, typically of interest only when diagnosing
        problems.
    INFO
        Print confirmation that things are working as expected, e.g. when
        each task is run correctly (this is the default).
    WARNING
        Indication that something unexpected happened on a glacier,
        but that OGGM is still working on this glacier.
    ERROR
        Print workflow messages and errors only, e.g. when a glacier cannot
        run properly.
    WORKFLOW
        Print only high level, workflow information (typically, one message
        per task). Errors and warnings will NOT be printed.
    CRITICAL
        Print nothing but fatal errors.

    Parameters
    ----------
    logging_level : str or None
        the logging level. See description above for a list of options. Setting
        to `None` is equivalent to `'CRITICAL'`, i.e. no log output will be
        generated.
    """

    # Add a custom level - just for us
    logging.addLevelName(45, 'WORKFLOW')

    def workflow(self, message, *args, **kws):
        """Standard log message with a custom level."""
        if self.isEnabledFor(45):
            # Yes, logger takes its '*args' as 'args'.
            self._log(45, message, args, **kws)

    logging.WORKFLOW = 45
    logging.Logger.workflow = workflow

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Spammers
    logging.getLogger("Fiona").setLevel(logging.CRITICAL)
    logging.getLogger("fiona").setLevel(logging.CRITICAL)
    logging.getLogger("shapely").setLevel(logging.CRITICAL)
    logging.getLogger("rasterio").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.getLogger("numexpr").setLevel(logging.CRITICAL)

    # Basic config
    if logging_level is None:
        logging_level = 'CRITICAL'

    logging_level = logging_level.upper()

    logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=getattr(logging, logging_level))


def initialize_minimal(file=None, logging_level='INFO', params=None):
    """Same as initialise() but without requiring any download of data.

    This is useful for "flowline only" OGGM applications

    Parameters
    ----------
    file : str
        path to the configuration file (default: OGGM params.cfg)
    logging_level : str
        set a logging level. See :func:`set_logging_config` for options.
    params : dict
        overrides for specific parameters from the config file
    """
    global IS_INITIALIZED
    global PARAMS
    global PATHS

    set_logging_config(logging_level=logging_level)

    is_default = False
    if file is None:
        file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'params.cfg')
        is_default = True
    try:
        cp = ConfigObj(file, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Config file could not be parsed (%s): %s', file, e)
        sys.exit()

    if is_default:
        log.workflow('Reading default parameters from the OGGM `params.cfg` '
                     'configuration file.')
    else:
        log.workflow('Reading parameters from the user provided '
                     'configuration file: %s', file)

    # Static Paths
    oggm_static_paths()

    # Apply code-side manual params overrides
    if params:
        for k, v in params.items():
            cp[k] = v

    # Paths
    PATHS['working_dir'] = cp['working_dir']
    PATHS['dem_file'] = cp['dem_file']
    PATHS['climate_file'] = cp['climate_file']

    # Ephemeral paths overrides
    env_wd = os.environ.get('OGGM_WORKDIR')
    if env_wd and not PATHS['working_dir']:
        PATHS['working_dir'] = env_wd
        log.workflow("PATHS['working_dir'] set to env variable $OGGM_WORKDIR: "
                     + env_wd)

    # Do not spam
    PARAMS.do_log = False

    # Multiprocessing pool
    try:
        use_mp = bool(int(os.environ['OGGM_USE_MULTIPROCESSING']))
        msg = 'ON' if use_mp else 'OFF'
        log.workflow('Multiprocessing switched {} '.format(msg) +
                     'according to the ENV variable OGGM_USE_MULTIPROCESSING')
    except KeyError:
        use_mp = cp.as_bool('use_multiprocessing')
        msg = 'ON' if use_mp else 'OFF'
        log.workflow('Multiprocessing switched {} '.format(msg) +
                     'according to the parameter file.')
    PARAMS['use_multiprocessing'] = use_mp

    # Spawn
    try:
        use_mp_spawn = bool(int(os.environ['OGGM_USE_MP_SPAWN']))
        msg = 'ON' if use_mp_spawn else 'OFF'
        log.workflow('MP spawn context switched {} '.format(msg) +
                     'according to the ENV variable OGGM_USE_MP_SPAWN')
    except KeyError:
        use_mp_spawn = cp.as_bool('use_mp_spawn')
    PARAMS['use_mp_spawn'] = use_mp_spawn

    # Number of processes
    mpp = cp.as_int('mp_processes')
    if mpp == -1:
        try:
            mpp = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            log.workflow('Multiprocessing: using slurm allocated '
                         'processors (N={})'.format(mpp))
        except (KeyError, ValueError):
            import multiprocessing
            mpp = multiprocessing.cpu_count()
            log.workflow('Multiprocessing: using all available '
                         'processors (N={})'.format(mpp))
    else:
        log.workflow('Multiprocessing: using the requested number of '
                     'processors (N={})'.format(mpp))
    PARAMS['mp_processes'] = mpp

    # Size of LRU cache
    try:
        lru_maxsize = int(os.environ['LRU_MAXSIZE'])
        log.workflow('Size of LRU cache set to {} '.format(lru_maxsize) +
                     'according to the ENV variable LRU_MAXSIZE')
    except KeyError:
        lru_maxsize = cp.as_int('lru_maxsize')
    PARAMS['lru_maxsize'] = lru_maxsize

    # Some non-trivial params
    PARAMS['continue_on_error'] = cp.as_bool('continue_on_error')
    PARAMS['grid_dx_method'] = cp['grid_dx_method']
    PARAMS['map_proj'] = cp['map_proj']
    PARAMS['topo_interp'] = cp['topo_interp']
    PARAMS['clip_dem_to_zero'] = cp.as_bool('clip_dem_to_zero')
    PARAMS['use_intersects'] = cp.as_bool('use_intersects')
    PARAMS['use_compression'] = cp.as_bool('use_compression')
    PARAMS['border'] = cp.as_int('border')
    PARAMS['mpi_recv_buf_size'] = cp.as_int('mpi_recv_buf_size')
    PARAMS['use_multiple_flowlines'] = cp.as_bool('use_multiple_flowlines')
    PARAMS['filter_min_slope'] = cp.as_bool('filter_min_slope')
    PARAMS['downstream_line_shape'] = cp['downstream_line_shape']
    PARAMS['auto_skip_task'] = cp.as_bool('auto_skip_task')
    PARAMS['rgi_version'] = cp['rgi_version']
    PARAMS['use_rgi_area'] = cp.as_bool('use_rgi_area')
    PARAMS['compress_climate_netcdf'] = cp.as_bool('compress_climate_netcdf')
    PARAMS['use_tar_shapefiles'] = cp.as_bool('use_tar_shapefiles')
    PARAMS['keep_multipolygon_outlines'] = cp.as_bool('keep_multipolygon_outlines')
    PARAMS['clip_tidewater_border'] = cp.as_bool('clip_tidewater_border')
    PARAMS['dl_verify'] = cp.as_bool('dl_verify')
    PARAMS['use_kcalving_for_inversion'] = cp.as_bool('use_kcalving_for_inversion')
    PARAMS['use_kcalving_for_run'] = cp.as_bool('use_kcalving_for_run')
    PARAMS['calving_use_limiter'] = cp.as_bool('calving_use_limiter')
    PARAMS['use_inversion_params_for_run'] = cp.as_bool('use_inversion_params_for_run')
    k = 'error_when_glacier_reaches_boundaries'
    PARAMS[k] = cp.as_bool(k)
    PARAMS['store_model_geometry'] = cp.as_bool('store_model_geometry')
    PARAMS['store_fl_diagnostics'] = cp.as_bool('store_fl_diagnostics')

    # Climate
    PARAMS['baseline_climate'] = cp['baseline_climate'].strip().upper()
    PARAMS['hydro_month_nh'] = cp.as_int('hydro_month_nh')
    PARAMS['hydro_month_sh'] = cp.as_int('hydro_month_sh')
    PARAMS['geodetic_mb_period'] = cp['geodetic_mb_period']
    PARAMS['use_winter_prcp_fac'] = cp.as_bool('use_winter_prcp_fac')
    PARAMS['use_temp_bias_from_file'] = cp.as_bool('use_temp_bias_from_file')

    k = 'winter_prcp_fac_ab'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    k = 'ref_mb_valid_window'
    PARAMS[k] = [int(vk) for vk in cp.as_list(k)]
    k = 'free_board_marine_terminating'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    k = 'store_diagnostic_variables'
    PARAMS[k] = [str(vk) for vk in cp.as_list(k)]
    k = 'store_fl_diagnostic_variables'
    PARAMS[k] = [str(vk) for vk in cp.as_list(k)]
    k = 'by_bin_dx'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    k = 'by_bin_bins'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]

    # Flowline model
    k = 'glacier_length_method'
    PARAMS[k] = cp[k]
    k = 'evolution_model'
    PARAMS[k] = cp[k]

    # Others
    PARAMS['tidewater_type'] = cp.as_int('tidewater_type')

    # Precip factor can be none
    try:
        PARAMS['prcp_fac'] = cp.as_float('prcp_fac')
    except ValueError:
        PARAMS['prcp_fac'] = None

    # This also
    try:
        PARAMS['calving_line_extension'] = cp.as_int('calving_line_extension')
    except ValueError:
        PARAMS['calving_line_extension'] = None

    # Delete non-floats
    ltr = ['working_dir', 'dem_file', 'climate_file', 'use_tar_shapefiles',
           'grid_dx_method', 'compress_climate_netcdf', 'by_bin_dx',
           'mp_processes', 'use_multiprocessing', 'clip_dem_to_zero',
           'topo_interp', 'use_compression', 'bed_shape', 'continue_on_error',
           'use_multiple_flowlines', 'border', 'use_temp_bias_from_file',
           'mpi_recv_buf_size', 'map_proj', 'evolution_model',
           'hydro_month_sh', 'hydro_month_nh', 'by_bin_bins',
           'use_intersects', 'filter_min_slope', 'clip_tidewater_border',
           'auto_skip_task', 'ref_mb_valid_window',
           'rgi_version', 'dl_verify', 'use_mp_spawn', 'calving_use_limiter',
           'use_rgi_area', 'baseline_climate',
           'calving_line_extension', 'use_kcalving_for_run', 'lru_maxsize',
           'free_board_marine_terminating', 'use_kcalving_for_inversion',
           'error_when_glacier_reaches_boundaries', 'glacier_length_method',
           'use_inversion_params_for_run',
           'tidewater_type', 'store_model_geometry', 'use_winter_prcp_fac',
           'store_diagnostic_variables', 'store_fl_diagnostic_variables',
           'geodetic_mb_period', 'store_fl_diagnostics', 'winter_prcp_fac_ab',
           'prcp_fac', 'downstream_line_shape', 'keep_multipolygon_outlines']
    for k in ltr:
        cp.pop(k, None)

    # Other params are floats
    for k in cp:
        PARAMS[k] = cp.as_float(k)
    PARAMS.do_log = True

    # Empty defaults
    set_intersects_db()
    IS_INITIALIZED = True


def initialize(file=None, logging_level='INFO', params=None):
    """Read the configuration file containing the run's parameters.

    This should be the first call, before using any of the other OGGM modules
    for most (all?) OGGM simulations.

    Parameters
    ----------
    file : str
        path to the configuration file (default: OGGM params.cfg)
    logging_level : str
        set a logging level. See :func:`set_logging_config` for options.
    params : dict
        overrides for specific parameters from the config file
    """
    global PARAMS
    global DATA

    initialize_minimal(file=file, logging_level=logging_level, params=params)

    # Do not spam
    PARAMS.do_log = False

    # Make sure we have a proper cache dir
    from oggm.utils import download_oggm_files
    download_oggm_files()

    # Read in the demo glaciers
    file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'data', 'demo_glaciers.csv')
    DATA['demo_glaciers'] = pd.read_csv(file, index_col=0)

    # Add other things
    if 'dem_grids' not in DATA:
        grids = {}
        for grid_json in ['gimpdem_90m_v01.1.json',
                          'arcticdem_mosaic_100m_v3.0.json',
                          'Alaska_albers_V3.json',
                          'AntarcticDEM_wgs84.json',
                          'REMA_100m_dem.json']:
            if grid_json not in grids:
                fp = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  'data', grid_json)
                try:
                    grids[grid_json] = salem.Grid.from_json(fp)
                except NameError:
                    pass
        DATA['dem_grids'] = grids

    # Trigger a one time check of the hash file
    from oggm.utils import get_dl_verify_data
    get_dl_verify_data('dummy_section')

    # OK
    PARAMS.do_log = True


def oggm_static_paths():
    """Initialise the OGGM paths from the config file."""

    global PATHS, PARAMS

    # See if the file is there, if not create it
    if not os.path.exists(CONFIG_FILE):
        dldir = os.path.join(os.path.expanduser('~'), 'OGGM')
        config = ConfigObj()
        config['dl_cache_dir'] = os.path.join(dldir, 'download_cache')
        config['dl_cache_readonly'] = False
        config['tmp_dir'] = os.path.join(dldir, 'tmp')
        config['rgi_dir'] = os.path.join(dldir, 'rgi')
        config['test_dir'] = os.path.join(dldir, 'tests')
        config['has_internet'] = True
        config.filename = CONFIG_FILE
        config.write()

    # OK, read in the file
    try:
        config = ConfigObj(CONFIG_FILE, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Config file could not be parsed (%s): %s',
                     CONFIG_FILE, e)
        sys.exit()

    # Check that all keys are here
    for k in ['dl_cache_dir', 'dl_cache_readonly', 'tmp_dir',
              'rgi_dir', 'test_dir', 'has_internet']:
        if k not in config:
            raise InvalidParamsError('The oggm config file ({}) should have '
                                     'an entry for {}.'.format(CONFIG_FILE, k))

    # Override defaults with env variables if available
    if os.environ.get('OGGM_DOWNLOAD_CACHE_RO') is not None:
        ro = bool(strtobool(os.environ.get('OGGM_DOWNLOAD_CACHE_RO')))
        config['dl_cache_readonly'] = ro
    if os.environ.get('OGGM_DOWNLOAD_CACHE') is not None:
        config['dl_cache_dir'] = os.environ.get('OGGM_DOWNLOAD_CACHE')
    if os.environ.get('OGGM_EXTRACT_DIR') is not None:
        # This is for the directories where OGGM needs to extract things
        # On the cluster it might be useful to do it on a fast disc
        edir = os.path.abspath(os.environ.get('OGGM_EXTRACT_DIR'))
        config['tmp_dir'] = os.path.join(edir, 'tmp')
        config['rgi_dir'] = os.path.join(edir, 'rgi')
    if os.environ.get('OGGM_RGI_DIR') is not None:
        config['rgi_dir'] = os.path.abspath(os.environ.get('OGGM_RGI_DIR'))

    # Fill the PATH dict
    for k, v in config.iteritems():
        if not k.endswith('_dir'):
            continue
        PATHS[k] = os.path.abspath(os.path.expanduser(v))

    # Other
    PARAMS.do_log = False
    PARAMS['has_internet'] = config.as_bool('has_internet')
    PARAMS['dl_cache_readonly'] = config.as_bool('dl_cache_readonly')
    PARAMS.do_log = True

    # Create cache dir if possible
    if not os.path.exists(PATHS['dl_cache_dir']):
        if not PARAMS['dl_cache_readonly']:
            os.makedirs(PATHS['dl_cache_dir'])


# Always call this one!
oggm_static_paths()


def get_lru_handler(tmpdir=None, maxsize=None, ending='.tif'):
    """LRU handler for a given temporary directory (singleton).

    Parameters
    ----------
    tmpdir : str
        path to the temporary directory to handle. Default is
        ``cfg.PATHS['tmp_dir']``.
    maxsize : int
        the max number of files to keep in the directory
    ending : str
        consider only the files with a certain ending
    """
    global LRUHANDLERS

    # see if we're set up
    if tmpdir is None:
        tmpdir = PATHS['tmp_dir']
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # one handler per directory and file ending
    # (in practice not very useful, but a dict is easier to handle)
    k = (tmpdir, ending)
    if k in LRUHANDLERS:
        # was already there
        lru = LRUHANDLERS[k]
        # possibility to increase or decrease the cachesize if need be
        if maxsize is not None:
            lru.maxsize = maxsize
            lru.purge()
        return lru
    else:
        # we do a new one
        from oggm.utils import LRUFileCache
        # the files already present have to be counted, too
        l0 = list(glob.glob(os.path.join(tmpdir, '*' + ending)))
        l0.sort(key=os.path.getctime)
        lru = LRUFileCache(l0, maxsize=maxsize)
        LRUHANDLERS[k] = lru
        return lru


def set_intersects_db(path_or_gdf=None):
    """Set the glacier intersection database for OGGM to use.

    It is now set automatically by the
    :func:`oggm.workflow.init_glacier_directories` task, but setting it
    manually can be useful for a slightly faster run initialization.

    See :func:`oggm.utils.get_rgi_intersects_region_file` for how to obtain
    such data.

    Parameters
    ----------
    path_or_gdf : str of geopandas.GeoDataframe
        the intersects file to use
    """

    global PARAMS
    PARAMS.do_log = False

    if PARAMS['use_intersects'] and path_or_gdf is not None:
        if isinstance(path_or_gdf, str):
            PARAMS['intersects_gdf'] = gpd.read_file(path_or_gdf)
        else:
            PARAMS['intersects_gdf'] = path_or_gdf
    else:
        PARAMS['intersects_gdf'] = pd.DataFrame()
    PARAMS.do_log = True


def reset_working_dir():
    """Deletes the content of the working directory. Careful: cannot be undone!
    """
    if PATHS['working_dir']:
        if os.path.exists(PATHS['working_dir']):
            shutil.rmtree(PATHS['working_dir'])
        os.makedirs(PATHS['working_dir'])


def pack_config():
    """Pack the entire configuration in one pickleable dict."""

    return {
        'IS_INITIALIZED': IS_INITIALIZED,
        'PARAMS': dict(PARAMS),
        'PATHS': dict(PATHS),
        'LRUHANDLERS': dict(LRUHANDLERS),
        'DATA': dict(DATA),
        'BASENAMES': dict(BASENAMES),
        'DL_VERIFIED': dict(DL_VERIFIED),
        'DEM_SOURCE_TABLE': dict(DEM_SOURCE_TABLE)
    }


def unpack_config(cfg_dict):
    """Unpack and apply the config packed via pack_config."""

    global IS_INITIALIZED, PARAMS, PATHS, BASENAMES, LRUHANDLERS, DATA
    global DL_VERIFIED, DEM_SOURCE_TABLE

    IS_INITIALIZED = cfg_dict['IS_INITIALIZED']

    prev_log = PARAMS.do_log
    PARAMS.do_log = False

    PARAMS.clear()
    PATHS.clear()
    LRUHANDLERS.clear()
    DATA.clear()
    DL_VERIFIED.clear()
    DEM_SOURCE_TABLE.clear()

    PARAMS.update(cfg_dict['PARAMS'])
    PATHS.update(cfg_dict['PATHS'])
    LRUHANDLERS.update(cfg_dict['LRUHANDLERS'])
    DATA.update(cfg_dict['DATA'])
    DL_VERIFIED.update(cfg_dict['DL_VERIFIED'])
    DEM_SOURCE_TABLE.update(cfg_dict['DEM_SOURCE_TABLE'])

    PARAMS.do_log = prev_log

    # BASENAMES is a DocumentedDict, which cannot be pickled because
    # set intentionally mismatches with get
    BASENAMES = DocumentedDict()
    for k in cfg_dict['BASENAMES']:
        BASENAMES[k] = (cfg_dict['BASENAMES'][k], 'Imported Pickle')


def set_manager(manager):
    """Sets a multiprocessing manager to use for shared dicts"""

    global DL_VERIFIED, DEM_SOURCE_TABLE, DATA

    if manager:
        new_dict = manager.dict()
        new_dict.update(DL_VERIFIED)
        DL_VERIFIED = new_dict

        new_dict = manager.dict()
        new_dict.update(DEM_SOURCE_TABLE)
        DEM_SOURCE_TABLE = new_dict

        new_dict = manager.dict()
        new_dict.update(DATA)
        DATA = new_dict
    else:
        DL_VERIFIED = dict(DL_VERIFIED)
        DEM_SOURCE_TABLE = dict(DEM_SOURCE_TABLE)
        DATA = dict(DATA)


def add_to_basenames(basename, filename, docstr=''):
    """Add an entry to the list of BASENAMES.

    BASENAMES are access keys to files available at the gdir level.

    Parameters
    ----------
    basename : str
        the key (e.g. 'dem', 'model_flowlines')
    filename : str
        the associated filename (e.g. 'dem.tif', 'model_flowlines.pkl')
    docstr : str
        the associated docstring (for documentation)
    """
    global BASENAMES
    if '.' not in filename:
        raise ValueError('The filename needs a proper file suffix!')
    BASENAMES[basename] = (filename, docstr)
