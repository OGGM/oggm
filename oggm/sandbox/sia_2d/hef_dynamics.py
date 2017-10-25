import os

import geopandas as gpd
import netCDF4
import numpy as np
from salem import reduce

import oggm
from oggm import cfg, tasks
from oggm import utils, workflow
from oggm.core.massbalance import (RandomMassBalance)
from oggm.sandbox.sia_2d.models import Upstream2D, filter_ice_border
from oggm.utils import get_demo_file

cfg.initialize()

# Don't use divides for now, or intersects
cfg.set_divides_db()
cfg.set_intersects_db()

cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
cfg.PARAMS['border'] = 66
cfg.PARAMS['grid_dx_method'] = 'square'
cfg.PARAMS['fixed_dx'] = 50.
cfg.PARAMS['auto_skip_task'] = True
cfg.PARAMS['use_multiprocessing'] = True

# Data directory
db_dir = '/home/mowglie/disk/Dropbox/Share/Lehre/MA_Belie/data/'

# Working directory
base_dir = os.path.join(db_dir, 'hef_run')
# utils.mkdir(base_dir, reset=True)  # reset?
entity = gpd.read_file(get_demo_file('Hintereisferner_RGI5.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir)

tasks.define_glacier_region(gdir, entity=entity)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.compute_downstream_line(gdir)
tasks.initialize_flowlines(gdir)
tasks.compute_downstream_bedshape(gdir)
tasks.catchment_area(gdir)
tasks.catchment_intersections(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)
tasks.process_cru_data(gdir)
tasks.mu_candidates(gdir)
tasks.compute_ref_t_stars([gdir])
tasks.distribute_t_stars([gdir])
tasks.apparent_mb(gdir)
tasks.prepare_for_inversion(gdir)
tasks.volume_inversion(gdir, glen_a=cfg.A, fs=0)
tasks.filter_inversion_output(gdir)
tasks.distribute_thickness(gdir, how='per_interpolation')
tasks.init_present_time_glacier(gdir)

# On single flowline run
tasks.random_glacier_evolution(gdir, nyears=800, bias=0, seed=0,
                               filesuffix='_fl_def',
                               zero_initial_glacier=True)

# For metadata
utils.glacier_characteristics([gdir], path=base_dir+'/hef_fl_out.csv')

# OK now the distributed stuffs
with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
    topo = nc.variables['topo'][:]
    ice_thick = nc.variables['thickness'][:]

# Take a subset of the topo for easier regridding
ds = gdir.grid.to_dataset()
ds['topo'] = (('y', 'x'), topo)
ds['ice_thick'] = (('y', 'x'), ice_thick)
ds = ds.salem.subset(corners=((1, 3), (250, 202)), crs=gdir.grid)

# Possible are factors 1 (50m) 2 (100m) or 5 (250m)
factor = 2
topo = reduce(ds.topo.values, factor=factor)
ice_thick = reduce(ds.ice_thick.values, factor=factor)
bed_topo = topo - ice_thick
grid = ds.salem.grid.regrid(factor=1/factor)


# Utility function to apply a specific filter
mask = grid.region_of_interest(shape=db_dir + 'AvoidIce.shp')
premove = np.where(mask)
def filter_ice_tributaries(ice_thick):
    ice_thick[premove] = 0
    # Also the borders
    ice_thick[0, :] = 0
    ice_thick[-1, :] = 0
    ice_thick[:, 0] = 0
    ice_thick[:, -1] = 0
    return ice_thick

# The model run function
def run_task(gdir, grid=None, mb_model=None, glen_a=cfg.A, outf=None,
             ice_thick_filter=None, print_stdout=None):
    dmodel = Upstream2D(bed_topo, dx=grid.dx, mb_model=mb_model,
                        y0=0., glen_a=glen_a,
                        ice_thick_filter=ice_thick_filter)
    dmodel.run_until_and_store(800, run_path=outf, grid=grid,
                               print_stdout=print_stdout)

# Multiprocessing
tasks = []

mbmod = RandomMassBalance(gdir, seed=0)
outf = db_dir + 'out_def_100m.nc'
tasks.append((run_task, {'grid': grid, 'mb_model': mbmod, 'outf': outf,
                         'ice_thick_filter':filter_ice_border,
                         'print_stdout':'Task1'}))

mbmod = RandomMassBalance(gdir, seed=0)
mbmod.temp_bias = 0.5
outf = db_dir + 'out_tbias_100m.nc'
tasks.append((run_task, {'grid': grid, 'mb_model': mbmod, 'outf': outf,
                         'ice_thick_filter':filter_ice_border,
                         'print_stdout': 'Task2'}))

mbmod = RandomMassBalance(gdir, seed=0)
outf = db_dir + 'out_filter_100m.nc'
tasks.append((run_task, {'grid': grid, 'mb_model': mbmod, 'outf': outf,
                         'ice_thick_filter':filter_ice_tributaries,
                         'print_stdout': 'Task3'}))

mbmod = RandomMassBalance(gdir, seed=0)
mbmod.temp_bias = 0.5
outf = db_dir + 'out_filter_bias_100m.nc'
tasks.append((run_task, {'grid': grid, 'mb_model': mbmod, 'outf': outf,
                         'ice_thick_filter':filter_ice_tributaries,
                         'print_stdout': 'Task4'}))

workflow.execute_parallel_tasks(gdir, tasks)
