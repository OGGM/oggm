"""  Configuration file and options

A number of globals are defined here to be available everywhere.
"""
import logging
import os
import shutil
import sys
import glob
import json
from collections import OrderedDict
from distutils.util import strtobool

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.signal import gaussian
from configobj import ConfigObj, ConfigObjError

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

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
        except:
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
        ResettingOrderedDict.__setitem__(self, key, os.path.expanduser(value))


# Globals
IS_INITIALIZED = False
PARAMS = ResettingOrderedDict()
PATHS = PathOrderedDict()
BASENAMES = DocumentedDict()
LRUHANDLERS = ResettingOrderedDict()

# Constants
SEC_IN_YEAR = 365*24*3600
SEC_IN_DAY = 24*3600
SEC_IN_HOUR = 3600
SEC_IN_MONTH = 2628000
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

RHO = 900.  # ice density
G = 9.81  # gravity
N = 3.  # Glen's law's exponent
A = 2.4e-24  # Glen's default creep's parameter
FS = 5.7e-20  # Default sliding parameter from Oerlemans - OUTDATED

GAUSSIAN_KERNEL = dict()
for ks in [5, 7, 9]:
    kernel = gaussian(ks, 1)
    GAUSSIAN_KERNEL[ks] = kernel / kernel.sum()

_doc = 'A geotiff file containing the DEM (reprojected into the local grid).'
BASENAMES['dem'] = ('dem.tif', _doc)

_doc = 'The glacier outlines in the local projection.'
BASENAMES['outlines'] = ('outlines.shp', _doc)

_doc = 'The glacier intersects in the local projection.'
BASENAMES['intersects'] = ('intersects.shp', _doc)

_doc = 'The flowline catchments in the local projection.'
BASENAMES['flowline_catchments'] = ('flowline_catchments.shp', _doc)

_doc = 'The catchments intersections in the local projection.'
BASENAMES['catchments_intersects'] = ('catchments_intersects.shp', _doc)

_doc = 'A ``salem.Grid`` handling the georeferencing of the local grid.'
BASENAMES['glacier_grid'] = ('glacier_grid.json', _doc)

_doc = 'A dictionary containing runtime diagnostics useful for debugging.'
BASENAMES['diagnostics'] = ('diagnostics.json', _doc)

_doc = 'A netcdf file containing several gridded data variables such as ' \
       'topography, the glacier masks and more (see the netCDF file metadata).'
BASENAMES['gridded_data'] = ('gridded_data.nc', _doc)

_doc = 'A ``dict`` containing the shapely.Polygons of a glacier. The ' \
       '"polygon_hr" entry contains the geometry transformed to the local ' \
       'grid in (i, j) coordinates, while the "polygon_pix" entry contains ' \
       'the geometries transformed into the coarse grid (the i, j elements ' \
       'are integers). The "polygon_area" entry contains the area of the ' \
       'polygon as computed by Shapely.'
BASENAMES['geometries'] = ('geometries.pkl', _doc)

_doc = 'A ``dict`` containing the downsteam line geometry as well as the bed' \
       'shape computed from a parabolic fit.'
BASENAMES['downstream_line'] = ('downstream_line.pkl', _doc)

_doc = 'A text file with the source of the topo file (GIMP, SRTM, ...).'
BASENAMES['dem_source'] = ('dem_source.txt', _doc)

_doc = 'A hypsometry file as provided by RGI (useful for diagnostics).'
BASENAMES['hypsometry'] = ('hypsometry.csv', _doc)

_doc = 'A list of :py:class:`Centerline` instances, sorted by flow order.'
BASENAMES['centerlines'] = ('centerlines.pkl', _doc)

_doc = "A list of len n_centerlines, each element conaining a numpy array " \
       "of the indices in the glacier grid which represent the centerline's" \
       " catchment area."
BASENAMES['catchment_indices'] = ('catchment_indices.pkl', _doc)

_doc = 'A "better" version of the Centerlines, now on a regular spacing ' \
       'i.e., not on the gridded (i, j) indices. The tails of the ' \
       'tributaries are cut out to make more realistic junctions. ' \
       'They are now "1.5D" i.e., with a width.'
BASENAMES['inversion_flowlines'] = ('inversion_flowlines.pkl', _doc)

_doc = 'The monthly climate timeseries stored in a netCDF file.'
BASENAMES['climate_monthly'] = ('climate_monthly.nc', _doc)

_doc = ('Some information (dictionary) about the climate data and the mass '
        'balance parameters for this glacier.')
BASENAMES['climate_info'] = ('climate_info.pkl', _doc)

_doc = 'The monthly GCM climate timeseries stored in a netCDF file.'
BASENAMES['cesm_data'] = ('cesm_data.nc', _doc)

_doc = 'A csv with three values: the local scalars mu*, t*, bias'
BASENAMES['local_mustar'] = ('local_mustar.csv', _doc)

_doc = 'List of dicts containing the data needed for the inversion.'
BASENAMES['inversion_input'] = ('inversion_input.pkl', _doc)

_doc = 'List of dicts containing the output data from the inversion.'
BASENAMES['inversion_output'] = ('inversion_output.pkl', _doc)

_doc = 'Dict of fs and fd as computed by the inversion optimisation.'
BASENAMES['inversion_params'] = ('inversion_params.pkl', _doc)

_doc = 'List of flowlines ready to be run by the model.'
BASENAMES['model_flowlines'] = ('model_flowlines.pkl', _doc)

_doc = ('When using a linear mass-balance for the inversion, this dict stores '
        'the optimal ela_h and grad.')
BASENAMES['linear_mb_params'] = ('linear_mb_params.pkl', _doc)

_doc = 'A netcdf file containing enough information to reconstruct the ' \
       'entire flowline glacier along the run (can be data expensive).'
BASENAMES['model_run'] = ('model_run.nc', _doc)

_doc = 'A netcdf file containing the model diagnostics (volume, ' \
       'mass-balance, length...).'
BASENAMES['model_diagnostics'] = ('model_diagnostics.nc', _doc)

_doc = 'Calving output'
BASENAMES['calving_output'] = ('calving_output.pkl', _doc)


def initialize(file=None):
    """Read the configuration file containing the run's parameters."""

    global IS_INITIALIZED
    global PARAMS
    global PATHS

    if file is None:
        file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'params.cfg')

    log.info('Parameter file: %s', file)

    try:
        cp = ConfigObj(file, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Param file could not be parsed (%s): %s', file, e)
        sys.exit()

    # Paths
    oggm_static_paths()
    PATHS['working_dir'] = cp['working_dir']
    PATHS['dem_file'] = cp['dem_file']
    PATHS['climate_file'] = cp['climate_file']

    # Multiprocessing pool
    PARAMS['use_multiprocessing'] = cp.as_bool('use_multiprocessing')
    PARAMS['mp_processes'] = cp.as_int('mp_processes')

    # Some non-trivial params
    PARAMS['continue_on_error'] = cp.as_bool('continue_on_error')
    PARAMS['grid_dx_method'] = cp['grid_dx_method']
    PARAMS['topo_interp'] = cp['topo_interp']
    PARAMS['use_intersects'] = cp.as_bool('use_intersects')
    PARAMS['use_compression'] = cp.as_bool('use_compression')
    PARAMS['mpi_recv_buf_size'] = cp.as_int('mpi_recv_buf_size')
    PARAMS['use_multiple_flowlines'] = cp.as_bool('use_multiple_flowlines')
    PARAMS['optimize_thick'] = cp.as_bool('optimize_thick')
    PARAMS['filter_min_slope'] = cp.as_bool('filter_min_slope')
    PARAMS['auto_skip_task'] = cp.as_bool('auto_skip_task')
    PARAMS['correct_for_neg_flux'] = cp.as_bool('correct_for_neg_flux')
    PARAMS['filter_for_neg_flux'] = cp.as_bool('filter_for_neg_flux')
    PARAMS['run_mb_calibration'] = cp.as_bool('run_mb_calibration')
    PARAMS['rgi_version'] = cp['rgi_version']
    PARAMS['use_rgi_area'] = cp.as_bool('use_rgi_area')
    PARAMS['compress_climate_netcdf'] = cp.as_bool('compress_climate_netcdf')

    # Climate
    PARAMS['baseline_climate'] = cp['baseline_climate'].strip().upper()
    PARAMS['baseline_y0'] = cp.as_int('baseline_y0')
    PARAMS['baseline_y1'] = cp.as_int('baseline_y1')
    PARAMS['hydro_month_nh'] = cp.as_int('hydro_month_nh')
    PARAMS['hydro_month_sh'] = cp.as_int('hydro_month_sh')
    PARAMS['temp_use_local_gradient'] = cp.as_bool('temp_use_local_gradient')
    k = 'temp_local_gradient_bounds'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    k = 'tstar_search_window'
    PARAMS[k] = [int(vk) for vk in cp.as_list(k)]
    PARAMS['use_bias_for_run'] = cp.as_bool('use_bias_for_run')
    PARAMS['allow_negative_mustar'] = cp.as_bool('allow_negative_mustar')

    # Inversion
    PARAMS['invert_with_sliding'] = cp.as_bool('invert_with_sliding')
    _k = 'optimize_inversion_params'
    PARAMS[_k] = cp.as_bool(_k)
    PARAMS['use_shape_factor_for_inversion'] = \
                cp['use_shape_factor_for_inversion']

    # Flowline model
    _k = 'use_optimized_inversion_params'
    PARAMS[_k] = cp.as_bool(_k)
    PARAMS['use_shape_factor_for_fluxbasedmodel'] = \
                cp['use_shape_factor_for_fluxbasedmodel']

    # Make sure we have a proper cache dir
    from oggm.utils import download_oggm_files, get_demo_file
    download_oggm_files()

    # Delete non-floats
    ltr = ['working_dir', 'dem_file', 'climate_file',
           'grid_dx_method', 'run_mb_calibration', 'compress_climate_netcdf',
           'mp_processes', 'use_multiprocessing', 'baseline_y0', 'baseline_y1',
           'temp_use_local_gradient', 'temp_local_gradient_bounds',
           'topo_interp', 'use_compression', 'bed_shape', 'continue_on_error',
           'use_optimized_inversion_params', 'invert_with_sliding',
           'optimize_inversion_params', 'use_multiple_flowlines',
           'optimize_thick', 'mpi_recv_buf_size', 'hydro_month_nh',
           'tstar_search_window', 'use_bias_for_run', 'hydro_month_sh',
           'use_intersects', 'filter_min_slope',
           'auto_skip_task', 'correct_for_neg_flux', 'filter_for_neg_flux',
           'rgi_version', 'allow_negative_mustar',
           'use_shape_factor_for_inversion', 'use_rgi_area',
           'use_shape_factor_for_fluxbasedmodel', 'baseline_climate']
    for k in ltr:
        cp.pop(k, None)

    # Other params are floats
    for k in cp:
        PARAMS[k] = cp.as_float(k)

    # Read-in the reference t* data - maybe it will be used, maybe not
    fns = ['ref_tstars_rgi5_cru4', 'ref_tstars_rgi6_cru4',
           'ref_tstars_rgi5_histalp', 'ref_tstars_rgi6_histalp']
    for fn in fns:
        PARAMS[fn] = pd.read_csv(get_demo_file('oggm_' + fn + '.csv'))
        fpath = get_demo_file('oggm_' + fn + '_calib_params.json')
        with open(fpath, 'r') as fp:
            mbpar = json.load(fp)
        PARAMS[fn+'_calib_params'] = mbpar

    # Empty defaults
    set_intersects_db()
    IS_INITIALIZED = True

    # Pre extract cru cl to avoid problems by multiproc
    from oggm.utils import get_cru_cl_file
    get_cru_cl_file()


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
        config['cru_dir'] = os.path.join(dldir, 'cru')
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
              'cru_dir', 'rgi_dir', 'test_dir', 'has_internet']:
        if k not in config:
            raise RuntimeError('The oggm config file ({}) should have an '
                               'entry for {}.'.format(CONFIG_FILE, k))

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
        config['cru_dir'] = os.path.join(edir, 'cru')
        config['rgi_dir'] = os.path.join(edir, 'rgi')

    if not config['dl_cache_dir']:
        raise RuntimeError('At the very least, the "dl_cache_dir" entry '
                           'should be provided in the oggm config file '
                           '({})'.format(CONFIG_FILE, k))

    # Fill the PATH dict
    for k, v in config.iteritems():
        if not k.endswith('_dir'):
            continue
        PATHS[k] = os.path.abspath(os.path.expanduser(v))

    # Other
    PARAMS['has_internet'] = config.as_bool('has_internet')
    PARAMS['dl_cache_readonly'] = config.as_bool('dl_cache_readonly')

    # Create cache dir if possible
    if not os.path.exists(PATHS['dl_cache_dir']):
        if not PARAMS['dl_cache_readonly']:
            os.makedirs(PATHS['dl_cache_dir'])

# Always call this one!
oggm_static_paths()


def get_lru_handler(tmpdir=None, maxsize=100, ending='.tif'):
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

    # one handler per directory and per size
    # (in practice not very useful, but a dict is easier to handle)
    k = (tmpdir, maxsize)
    if k in LRUHANDLERS:
        # was already there
        return LRUHANDLERS[k]
    else:
        # we do a new one
        from oggm.utils import LRUFileCache
        # the files already present have to be counted, too
        l0 = list(glob.glob(os.path.join(tmpdir, '*' + ending)))
        l0.sort(key=os.path.getmtime)
        lru = LRUFileCache(l0, maxsize=maxsize)
        LRUHANDLERS[k] = lru
        return lru


def set_intersects_db(path=None):
    """Read the intersects database.
    """

    if PARAMS['use_intersects'] and path is not None:
        if isinstance(path, str):
            PARAMS['intersects_gdf'] = gpd.read_file(path)
        else:
            PARAMS['intersects_gdf'] = path
    else:
        PARAMS['intersects_gdf'] = gpd.GeoDataFrame()


def reset_working_dir():
    """Deletes the working directory."""
    if PATHS['working_dir']:
        if os.path.exists(PATHS['working_dir']):
            shutil.rmtree(PATHS['working_dir'])
        os.makedirs(PATHS['working_dir'])


def pack_config():
    """Pack the entire configuration in one pickleable dict."""

    return {
        'IS_INITIALIZED': IS_INITIALIZED,
        'PARAMS': PARAMS,
        'PATHS': PATHS,
        'LRUHANDLERS': LRUHANDLERS,
        'BASENAMES': dict(BASENAMES)
    }


def unpack_config(cfg_dict):
    """Unpack and apply the config packed via pack_config."""

    global IS_INITIALIZED, PARAMS, PATHS, BASENAMES, LRUHANDLERS

    IS_INITIALIZED = cfg_dict['IS_INITIALIZED']
    PARAMS = cfg_dict['PARAMS']
    PATHS = cfg_dict['PATHS']
    LRUHANDLERS = cfg_dict['LRUHANDLERS']

    # BASENAMES is a DocumentedDict, which cannot be pickled because
    # set intentionally mismatches with get
    BASENAMES = DocumentedDict()
    for k in cfg_dict['BASENAMES']:
        BASENAMES[k] = (cfg_dict['BASENAMES'][k], 'Imported Pickle')
