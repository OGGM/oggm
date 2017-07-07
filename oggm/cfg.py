"""  Configuration file and options

A number of globals are defined here to be available everywhere.
"""
from __future__ import absolute_import, division

import logging
import os
import shutil
import sys
import glob
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
        try:
            self._set_key(key, value[0], docstr=value[1])
        except:
            raise ValueError('DocumentedDict accepts only tuple of len 2')

    def info_str(self, key):
        """Info string for the documentation."""
        return '    {}'.format(self[key]) + '\n' + '        ' + self._doc[key]

    def doc_str(self, key):
        """Info string for the documentation."""
        return '        {}'.format(self[key]) + '\n' + '            ' + \
               self._doc[key]


class PathOrderedDict(OrderedDict):
    """Quick "magic" to be sure that paths are expanded correctly."""

    def __setitem__(self, key, value):
        # Overrides the original dic to expand the path
        OrderedDict.__setitem__(self, key, os.path.expanduser(value))

# Globals
IS_INITIALIZED = False
CONTINUE_ON_ERROR = False
PARAMS = OrderedDict()
PATHS = PathOrderedDict()
BASENAMES = DocumentedDict()
RGI_REG_NAMES = False
RGI_SUBREG_NAMES = False
LRUHANDLERS = OrderedDict()

# Constants
SEC_IN_YEAR = 365*24*3600
SEC_IN_DAY = 24*3600
SEC_IN_HOUR = 3600
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SEC_IN_MONTHS = [d * SEC_IN_DAY for d in DAYS_IN_MONTH]
CUMSEC_IN_MONTHS = np.cumsum(SEC_IN_MONTHS)
BEGINSEC_IN_MONTHS = np.insert(CUMSEC_IN_MONTHS[:-1], [0], 0)

RHO = 900.  # ice density
G = 9.81  # gravity
N = 3.  # Glen's law's exponent
A = 2.4e-24  # Glen's default creep's parameter
FS = 5.7e-20  # Default sliding parameter from Oerlemans - OUTDATED
TWO_THIRDS = 2./3.
FOUR_THIRDS = 4./3.
ONE_FIFTH = 1./5.

GAUSSIAN_KERNEL = dict()
for ks in [5, 7, 9]:
    kernel = gaussian(ks, 1)
    GAUSSIAN_KERNEL[ks] = kernel / kernel.sum()

# TODO: document all files
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

_doc = 'The divides intersections in their lonlat projection.'
BASENAMES['divides_intersects'] = ('divides_intersects.shp', _doc)

_doc = 'A ``salem.Grid`` handling the georeferencing of the local grid.'
BASENAMES['glacier_grid'] = ('glacier_grid.json', _doc)

_doc = 'A netcdf file containing several gridded data variables such as ' \
       'topography, the glacier masks and more (see the netCDF file metadata).'
BASENAMES['gridded_data'] = ('gridded_data.nc', _doc)

_doc = 'A ``dict`` containing the shapely.Polygons of a divide. The ' \
       '"polygon_hr" entry contains the geometry transformed to the local ' \
       'grid in (i, j) coordinates, while the "polygon_pix" entry contains ' \
       'the geometries transformed into the coarse grid (the i, j elements ' \
       'are integers). The "polygon_area" entry contains the area of the ' \
       'polygon as computed by Shapely (it is needed because the divides ' \
       'will have their own area which is not obtained from the RGI file).'
BASENAMES['geometries'] = ('geometries.pkl', _doc)

_doc = 'A geopandas dataframe containing all downstream lines, grouped per ' \
       'intersection.'
BASENAMES['downstream_lines'] = ('downstream_lines.pkl', _doc)

_doc = 'A string with the source of the topo file (ASTER, SRTM, ...).'
BASENAMES['dem_source'] = ('dem_source.pkl', _doc)

_doc = 'A simple integer in the glacier root directory (divide 00) ' \
       'containing the ID of the "major divide", i.e. the one ' \
       'really flowing out of the glacier (the other downstream lines ' \
       'flowing into the main branch).'
BASENAMES['major_divide'] = ('major_divide.pkl', _doc)

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

_doc = 'The monthly climate timeseries for this glacier, stored in a netCDF ' \
       'file.'
BASENAMES['climate_monthly'] = ('climate_monthly.nc', _doc)

_doc = 'Some information (dictionary) about the climate data for this ' \
       'glacier, avoiding many useless accesses to the netCDF file.'
BASENAMES['climate_info'] = ('climate_info.pkl', _doc)

_doc = 'The monthly GCM climate timeseries for this glacier, stored in a netCDF ' \
       'file.'
BASENAMES['cesm_data'] = ('cesm_data.nc', _doc)

_doc = 'A Dataframe containing the bias scores as a function of the prcp ' \
       'factor. This is useful for testing mostly.'
BASENAMES['prcp_fac_optim'] = ('prcp_fac_optim.pkl', _doc)

_doc = 'A pandas.Series with the (year, mu) data.'
BASENAMES['mu_candidates'] = ('mu_candidates.pkl', _doc)

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

_doc = ''
BASENAMES['find_initial_glacier_params'] = ('find_initial_glacier_params.pkl',
                                            _doc)

_doc = ''
BASENAMES['past_model'] = ('past_model.nc', _doc)

_doc = 'Calving output'
BASENAMES['calving_output'] = ('calving_output.pkl', _doc)

_doc = 'Array of the parabolic coefficients for all centerlines, computed ' \
       'from fitting a parabola to the topography.'
BASENAMES['downstream_bed'] = ('downstream_bed.pkl', _doc)


def initialize(file=None):
    """Read the configuration file containing the run's parameters."""

    global IS_INITIALIZED
    global PARAMS
    global PATHS
    global CONTINUE_ON_ERROR
    global N
    global A
    global RHO
    global RGI_REG_NAMES
    global RGI_SUBREG_NAMES

    if file is None:
        file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'params.cfg')

    log.info('Parameter file: %s', file)

    try:
        cp = ConfigObj(file, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Param file could not be parsed (%s): %s', file, e)
        sys.exit()

    CONTINUE_ON_ERROR = cp.as_bool('continue_on_error')

    # Default
    PATHS['working_dir'] = cp['working_dir']
    if not PATHS['working_dir']:
        PATHS['working_dir'] = os.path.join(os.path.expanduser('~'),
                                            'OGGM_WORKING_DIRECTORY')

    # Paths
    oggm_static_paths()

    PATHS['dem_file'] = cp['dem_file']
    PATHS['climate_file'] = cp['climate_file']
    PATHS['wgms_rgi_links'] = cp['wgms_rgi_links']
    PATHS['glathida_rgi_links'] = cp['glathida_rgi_links']
    PATHS['leclercq_rgi_links'] = cp['leclercq_rgi_links']

    # run params
    PARAMS['run_period'] = [int(vk) for vk in cp.as_list('run_period')]

    # Multiprocessing pool
    PARAMS['use_multiprocessing'] = cp.as_bool('use_multiprocessing')
    PARAMS['mp_processes'] = cp.as_int('mp_processes')

    # Some non-trivial params
    PARAMS['grid_dx_method'] = cp['grid_dx_method']
    PARAMS['topo_interp'] = cp['topo_interp']
    PARAMS['use_divides'] = cp.as_bool('use_divides')
    PARAMS['use_intersects'] = cp.as_bool('use_intersects')
    PARAMS['use_compression'] = cp.as_bool('use_compression')
    PARAMS['mpi_recv_buf_size'] = cp.as_int('mpi_recv_buf_size')
    PARAMS['use_multiple_flowlines'] = cp.as_bool('use_multiple_flowlines')
    PARAMS['optimize_thick'] = cp.as_bool('optimize_thick')
    PARAMS['filter_min_slope'] = cp.as_bool('filter_min_slope')
    PARAMS['auto_skip_task'] = cp.as_bool('auto_skip_task')

    # Climate
    PARAMS['temp_use_local_gradient'] = cp.as_bool('temp_use_local_gradient')
    k = 'temp_local_gradient_bounds'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    k = 'tstar_search_window'
    PARAMS[k] = [int(vk) for vk in cp.as_list(k)]
    PARAMS['use_bias_for_run'] = cp.as_bool('use_bias_for_run')
    _factor = cp['prcp_scaling_factor']
    if _factor not in ['stddev', 'stddev_perglacier']:
        _factor = cp.as_float('prcp_scaling_factor')
    PARAMS['prcp_scaling_factor'] = _factor

    # Inversion
    PARAMS['invert_with_sliding'] = cp.as_bool('invert_with_sliding')
    _k = 'optimize_inversion_params'
    PARAMS[_k] = cp.as_bool(_k)

    # Flowline model
    _k = 'use_optimized_inversion_params'
    PARAMS[_k] = cp.as_bool(_k)

    # Make sure we have a proper cache dir
    from oggm.utils import download_oggm_files
    download_oggm_files()

    # Parse RGI metadata
    _d = os.path.join(CACHE_DIR, 'oggm-sample-data-master', 'rgi_meta')
    RGI_REG_NAMES = pd.read_csv(os.path.join(_d, 'rgi_regions.csv'),
                                index_col=0)
    RGI_SUBREG_NAMES = pd.read_csv(os.path.join(_d, 'rgi_subregions.csv'),
                                   index_col=0)

    # Delete non-floats
    ltr = ['working_dir', 'dem_file', 'climate_file', 'wgms_rgi_links',
           'glathida_rgi_links', 'grid_dx_method',
           'mp_processes', 'use_multiprocessing', 'use_divides',
           'temp_use_local_gradient', 'temp_local_gradient_bounds',
           'topo_interp', 'use_compression', 'bed_shape', 'continue_on_error',
           'use_optimized_inversion_params', 'invert_with_sliding',
           'optimize_inversion_params', 'use_multiple_flowlines',
           'leclercq_rgi_links', 'optimize_thick', 'mpi_recv_buf_size',
           'tstar_search_window', 'use_bias_for_run', 'run_period',
           'prcp_scaling_factor', 'use_intersects', 'filter_min_slope',
           'auto_skip_task']
    for k in ltr:
        cp.pop(k, None)

    # Other params are floats
    for k in cp:
        PARAMS[k] = cp.as_float(k)

    # Empty defaults
    from oggm.utils import get_demo_file
    set_divides_db(get_demo_file('divides_alps.shp'))
    set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    IS_INITIALIZED = True


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

    if not config['dl_cache_dir']:
        raise RuntimeError('At the very least, the "dl_cache_dir" entry '
                           'should be provided in the oggm config file '
                           '({})'.format(CONFIG_FILE, k))

    # Fill the PATH dict
    for k, v in config.iteritems():
        if not k.endswith('_dir'):
            continue
        if not v:
            # Default value?
            v = os.path.join('~', 'OGGM_' + k[:-4])
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


def set_divides_db(path=None):
    """Read the divides database.
    """

    if PARAMS['use_divides'] and path is not None:
        df = gpd.read_file(path)
        try:
            # dirty fix for RGIV5
            r5 = df.copy()
            r5.RGIID = [r.replace('RGI40', 'RGI50') for r in r5.RGIID.values]
            df = pd.concat([df, r5])
            PARAMS['divides_gdf'] = df.set_index('RGIID')
        except AttributeError:
            # dirty fix for RGIV4
            r4 = df.copy()
            r4.RGIId = [r.replace('RGI50', 'RGI40') for r in r5.RGIId.values]
            df = pd.concat([df, r4])
            PARAMS['divides_gdf'] = df.set_index('RGIId')
    else:
        PARAMS['divides_gdf'] = gpd.GeoDataFrame()


def set_intersects_db(path=None):
    """Read the intersects database.
    """

    if PARAMS['use_intersects'] and path is not None:
        PARAMS['intersects_gdf'] = gpd.read_file(path)
    else:
        PARAMS['intersects_gdf'] = gpd.GeoDataFrame()


def reset_working_dir():
    """Deletes the working directory."""
    if os.path.exists(PATHS['working_dir']):
        shutil.rmtree(PATHS['working_dir'])
    os.makedirs(PATHS['working_dir'])


def pack_config():
    """Pack the entire configuration in one pickleable dict."""

    return {
        'IS_INITIALIZED': IS_INITIALIZED,
        'CONTINUE_ON_ERROR': CONTINUE_ON_ERROR,
        'PARAMS': PARAMS,
        'PATHS': PATHS,
        'LRUHANDLERS': LRUHANDLERS,
        'BASENAMES': dict(BASENAMES)
    }


def unpack_config(cfg_dict):
    """Unpack and apply the config packed via pack_config."""

    global IS_INITIALIZED, CONTINUE_ON_ERROR, PARAMS, PATHS, \
        BASENAMES, LRUHANDLERS

    IS_INITIALIZED = cfg_dict['IS_INITIALIZED']
    CONTINUE_ON_ERROR = cfg_dict['CONTINUE_ON_ERROR']
    PARAMS = cfg_dict['PARAMS']
    PATHS = cfg_dict['PATHS']
    LRUHANDLERS = cfg_dict['LRUHANDLERS']

    # BASENAMES is a DocumentedDict, which cannot be pickled because
    # set intentionally mismatches with get
    BASENAMES = DocumentedDict()
    for k in cfg_dict['BASENAMES']:
        BASENAMES[k] = (cfg_dict['BASENAMES'][k], 'Imported Pickle')
