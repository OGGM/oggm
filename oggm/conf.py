"""  Configuration file and options

A number of globals are defined here to be available everywhere.

Copyright: OGGM development team, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division, unicode_literals
from six.moves import range

import sys
import os
from configobj import ConfigObj, ConfigObjError
import logging
from collections import OrderedDict
import geopandas as gpd
import pandas as pd
import pickle
import glob
import netCDF4
import shutil
import numpy as np

from salem import lazy_property

# Local logger
log = logging.getLogger(__name__)

# Globals definition to assist the IDE auto-completion
params = OrderedDict()
input = OrderedDict()
output = OrderedDict()
use_divides = False
use_mp = False
nproc = -1
temp_use_local_gradient = False

def initialize(file=None):
    """Read the parameter configfile"""

    global params
    global input
    global output
    global use_mp
    global use_divides
    global nproc
    global temp_use_local_gradient
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

    if cp['input_data_dir'] == '~':
        cp['input_data_dir'] = os.path.join(homedir, 'Dropbox', 'OGGM')

    if cp['working_dir'] == '~':
        cp['working_dir'] = os.path.join(homedir, 'OGGM_wd')

    input['data_dir'] = cp['input_data_dir']
    output['working_dir'] = cp['working_dir']
    params['grid_dx_method'] = cp['grid_dx_method']

    log.info('Input data directory: %s', input['data_dir'])
    log.info('Working directory: %s', output['working_dir'])

    # Divides
    use_divides = cp.as_bool('use_divides')

    # Multiprocessing pool
    use_mp = cp.as_bool('multiprocessing')
    nproc = cp.as_int('processes')
    if nproc == -1:
        nproc = None

    # Climate
    temp_use_local_gradient = cp.as_bool('temp_use_local_gradient')

    k = 'temp_local_gradient_bounds'
    params[k] = [float(vk) for vk in cp.as_list(k)]

    for k in ['input_data_dir', 'working_dir', 'grid_dx_method',
              'multiprocessing', 'processes', 'use_divides',
              'temp_use_local_gradient', 'temp_local_gradient_bounds']:
        del cp[k]

    # Other params are floats
    for k in cp:
        params[k] = cp.as_float(k)

    data_dir = input['data_dir']
    input['srtm_file'] = os.path.join(data_dir, 'GIS', 'srtm_90m_v4_alps.tif')
    input['histalp_file'] = os.path.join(data_dir, 'CLIMATE' ,
                                         'histalp_merged.nc')


def set_divides_db(path):

    # Read divides database
    if use_divides:
        input['divides_gdf'] = gpd.GeoDataFrame.from_file(path).set_index('RGIID')
    else:
        input['divides_gdf'] = gpd.GeoDataFrame()


class GlacierDir(object):
    """A glacier directory is a simple class to help access the working files.

    It handles a glacier entity directory, created in a base directory (default
    is the "per_glacier" folder in the working dir). The glacier entity has,
    a map and a dem located in the base entity directory (also called divide_00
    internally). It has one or more divide subdirectories divide_01, ... divide_XX.
    Each divide has its own outlines and masks.  In the case the glacier has
    one single divide, the outlines and masks are thus linked to the divide_00 files.

    For memory optimisations it might do a couple of things twice.
    """

    def __init__(self, entity, base_dir=None, reset=False):
        """ Instanciate.

        Parameters
        ----------
        entity: RGI entity
        base_dir: path to the directory containing the glacier folders
            default is conf.output['working_dir'] + /per_glacier/
        reset: emtpy the directory at instanciating
        """

        if base_dir is None:
            base_dir = os.path.join(output['working_dir'], 'per_glacier')

        self.rgi_id = entity.RGIID
        self.glims_id = entity.GLIMSID
        self.glacier_area = float(entity.AREA)
        self.cenlon = float(entity.CENLON)
        self.cenlat = float(entity.CENLAT)
        if hasattr(entity, 'BGNDATE'):
            bgn = pd.to_datetime(entity.BGNDATE[0:6],
                                       errors='raise',
                                       format='%Y%m')
        else:
            bgn = pd.to_datetime('199001', format='%Y%m')
        self.rgi_date = bgn

        self.dir = os.path.join(base_dir, self.rgi_id)
        if reset and os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.climate_dir = os.path.join(self.dir, 'climate')

        # Divides have to be done later by gis.define_glacier_region

        # files that will be written out
        # Entity specific
        fnames = dict()
        fnames['dem'] = 'dem.tif'
        fnames['hypso'] = 'hypso.csv'
        fnames['glacier_grid'] = 'glacier_grid.p'
        # Divides specific
        fnames['grids'] = 'grids.nc'
        fnames['apparent_mb'] = 'apparent_mb.nc'
        fnames['geometries'] = 'geometries.p'
        fnames['outlines'] = 'outlines.shp'
        fnames['centerlines'] = 'centerlines.p'
        fnames['catchment_indices'] = 'catchment_indices.p'
        fnames['inversion_flowlines'] = 'inversion_flowlines.p'
        fnames['climate_monthly'] = 'climate_monthly.nc'
        fnames['mu_candidates'] = 'mu_candidates.p'
        fnames['local_mustar'] = 'local_mustar.csv'
        fnames['inversion_input'] = 'inversion_input.p'
        fnames['inversion_output'] = 'inversion_output.p'
        fnames['avg_slope'] = 'avg_slope.p'
        fnames['downstream_line'] = 'downstream_line.p'
        fnames['major_divide'] = 'major_divide.txt'
        fnames['model_flowlines'] = 'model_flowlines.p'
        fnames['flowline_params'] = 'flowline_params.p'
        fnames['t0_glacier'] = 't0_glacier.p'
        self.fnames = fnames

    @lazy_property
    def grid(self):
        """salem.Grid singleton.

        Must be called after gis.define_glacier_region."""

        return self.read_pickle('glacier_grid')

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

        if filename not in self.fnames:
            raise ValueError(filename + ' not in the dict.')

        dir = self.divide_dirs[div_id]

        return os.path.join(dir, self.fnames[filename])

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