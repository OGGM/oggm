"""  Configuration file and options

A number of globals are defined here to be available everywhere.

Copyright: OGGM development team, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division

import logging
import os
import shutil
import sys
from collections import OrderedDict

import geopandas as gpd
from configobj import ConfigObj, ConfigObjError

# Fiona and shapely are spammers
logging.getLogger("Fiona").setLevel(logging.WARNING)
logging.getLogger("shapely").setLevel(logging.WARNING)

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Local logger
log = logging.getLogger(__name__)

# Globals definition to assist the IDE auto-completion
IS_INITIALIZED = False
PARAMS = OrderedDict()
PATHS = OrderedDict()
USE_MP = False
NPROC = -1


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
        if isinstance(value, tuple):
            self._set_key(key, value[0], docstr=value[1])
        else:
            dict.__setitem__(self, key, value)

    def info_str(self, key):
        """Info string for the documentation."""
        return '    {}'.format(self[key]) + '\n' + '        ' + self._doc[key]


BASENAMES = DocumentedDict()
# TODO: document all files
_doc = 'A geotiff file containing the DEM (reprojected into the local grid).'
BASENAMES['dem'] = ('dem.tif', _doc)
_doc = 'A ``salem.Grid`` handling the georeferencing of the local grid.'
BASENAMES['glacier_grid'] = ('glacier_grid.pkl', _doc)
_doc = 'A netcdf file containing several gridded data variables such as ' \
       'topography, the glacier masks and more.'
BASENAMES['gridded_data'] = ('gridded_data.nc', _doc)
_doc = 'The apparent mass-balance data needed for the inversion.'
BASENAMES['apparent_mb'] = ('apparent_mb.nc', _doc)
_doc = 'A ``dict`` containing the shapely.Polygons of a divide.'
BASENAMES['geometries'] = ('geometries.pkl', _doc)
_doc = 'The glacier outlines in the local projection.'
BASENAMES['outlines'] = ('outlines.shp', _doc)
_doc = ''
BASENAMES['centerlines'] = ('centerlines.pkl', _doc)
_doc = ''
BASENAMES['catchment_indices'] = ('catchment_indices.pkl', _doc)
_doc = ''
BASENAMES['inversion_flowlines'] = ('inversion_flowlines.pkl', _doc)
_doc = ''
BASENAMES['climate_monthly'] = ('climate_monthly.nc', _doc)
_doc = ''
BASENAMES['mu_candidates'] = ('mu_candidates.pkl', _doc)
_doc = ''
BASENAMES['local_mustar'] = ('local_mustar.csv', _doc)
_doc = ''
BASENAMES['inversion_input'] = ('inversion_input.pkl', _doc)
_doc = ''
BASENAMES['inversion_output'] = ('inversion_output.pkl', _doc)
_doc = ''
BASENAMES['avg_slope'] = ('avg_slope.pkl', _doc)
_doc = ''
BASENAMES['downstream_line'] = ('downstream_line.pkl', _doc)
_doc = ''
BASENAMES['major_divide'] = ('major_divide.txt', _doc)
_doc = ''
BASENAMES['model_flowlines'] = ('model_flowlines.pkl', _doc)
_doc = ''
BASENAMES['flowline_params'] = ('flowline_params.pkl', _doc)
_doc = ''
BASENAMES['past_model'] = ('past_model.pkl', _doc)


def initialize(file=None):
    """Read the parameter configfile"""

    global IS_INITIALIZED
    global PARAMS
    global PATHS
    global USE_MP
    global NPROC
    if file is None:
        file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'params.cfg')

    log.info('Parameter file: %s', file)

    try:
        cp = ConfigObj(file, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Config file could not be parsed (%s): %s', file, e)
        sys.exit()

    homedir = os.path.expanduser("~")

    if cp['working_dir'] == '~':
        cp['working_dir'] = os.path.join(homedir, 'OGGM_wd')

    PATHS['working_dir'] = cp['working_dir']

    PATHS['srtm_file'] = cp['srtm_file']
    PATHS['histalp_file'] = cp['histalp_file']
    PATHS['wgms_rgi_links'] = cp['wgms_rgi_links']
    PATHS['glathida_rgi_links'] = cp['glathida_rgi_links']

    log.info('Working directory: %s', PATHS['working_dir'])

    # Multiprocessing pool
    USE_MP = cp.as_bool('multiprocessing')
    NPROC = cp.as_int('processes')
    if NPROC == -1:
        NPROC = None
    if USE_MP:
        log.info('Multiprocessing run')
    else:
        log.info('No multiprocessing')

    # Some non-trivial params
    PARAMS['grid_dx_method'] = cp['grid_dx_method']
    PARAMS['topo_interp'] = cp['topo_interp']
    PARAMS['use_divides'] = cp.as_bool('use_divides')

    # Climate
    PARAMS['temp_use_local_gradient'] = cp.as_bool('temp_use_local_gradient')
    k = 'temp_local_gradient_bounds'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]

    # Delete non-floats
    for k in ['working_dir', 'srtm_file', 'histalp_file', 'wgms_rgi_links',
              'glathida_rgi_links', 'grid_dx_method',
              'multiprocessing', 'processes', 'use_divides',
              'temp_use_local_gradient', 'temp_local_gradient_bounds',
              'topo_interp']:
        del cp[k]

    # Other params are floats
    for k in cp:
        PARAMS[k] = cp.as_float(k)

    # Empty defaults
    set_divides_db()
    IS_INITIALIZED = True


def set_divides_db(path=None):

    # Read divides database
    if PARAMS['use_divides'] and path is not None:
        PARAMS['divides_gdf'] = gpd.GeoDataFrame.from_file(path).set_index('RGIID')
    else:
        PARAMS['divides_gdf'] = gpd.GeoDataFrame()

def reset_working_dir():
    """To use with caution"""
    if os.path.exists(PATHS['working_dir']):
        shutil.rmtree(PATHS['working_dir'])
    os.makedirs(PATHS['working_dir'])

