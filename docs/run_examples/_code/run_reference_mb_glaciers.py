# Python imports
from os import path
from glob import glob

# Libs
import numpy as np
import pandas as pd
import geopandas as gpd

# Locals
import oggm
from oggm import cfg, utils, tasks, workflow
from oggm.workflow import execute_entity_task

# Module logger
import logging
log = logging.getLogger(__name__)

# RGI Version
rgi_version = '6'

# Initialize OGGM and set up the run parameters
cfg.initialize()

# Local paths (where to write the OGGM run output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp',
                        'OGGM_ref_mb_RGIV{}_OGGM{}'.format(rgi_version,
                                                           oggm.__version__))
utils.mkdir(WORKING_DIR, reset=True)
cfg.PATHS['working_dir'] = WORKING_DIR

# We are running the calibration ourselves
cfg.PARAMS['run_mb_calibration'] = True

# No need for intersects since this has an effect on the inversion only
cfg.PARAMS['use_intersects'] = False

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = False

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
_ = utils.get_cru_file(var='pre')
rgi_dir = utils.get_rgi_dir(version=rgi_version)

# Get the reference glacier ids (they are different for each RGI version)
df, _ = utils.get_wgms_files(version=rgi_version)
rids = df['RGI{}0_ID'.format(rgi_version)]

# Make a new dataframe with those (this takes a while)
log.info('Reading the RGI shapefiles...')
rgidf = []
for reg in df['RGI_REG'].unique():
    if reg == '19':
        continue  # we have no climate data in Antarctica
    fn = '*' + reg + '_rgi{}0_*.shp'.format(rgi_version)
    fs = list(sorted(glob(path.join(rgi_dir, '*', fn))))[0]
    sh = gpd.read_file(fs)
    rgidf.append(sh.loc[sh.RGIId.isin(rids)])
rgidf = pd.concat(rgidf)
rgidf.crs = sh.crs  # for geolocalisation

# We have to check which of them actually have enough mb data.
# Let OGGM do it:
gdirs = workflow.init_glacier_regions(rgidf)
# We need to know which period we have data for
log.info('Process the climate data...')
execute_entity_task(tasks.process_cru_data, gdirs, print_log=False)
gdirs = utils.get_ref_mb_glaciers(gdirs)
# Keep only these
rgidf = rgidf.loc[rgidf.RGIId.isin([g.rgi_id for g in gdirs])]

# Save
log.info('For RGIV{} we have {} reference glaciers.'.format(rgi_version,
                                                            len(rgidf)))
rgidf.to_file(path.join(WORKING_DIR, 'mb_ref_glaciers.shp'))

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf)

# Prepro tasks
task_list = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate tasks
execute_entity_task(tasks.process_cru_data, gdirs)
tasks.compute_ref_t_stars(gdirs)
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.apparent_mb, gdirs)
# Recompute after the first round - this is being picky but this is
# Because geometries may change after apparent_mb's filtering
tasks.compute_ref_t_stars(gdirs)
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.apparent_mb, gdirs)

# Model validation
tasks.quick_crossval_t_stars(gdirs)  # for later
tasks.distribute_t_stars(gdirs)  # To restore after cross-val

# Tests: for all glaciers, the mass-balance around tstar and the
# bias with observation should be approx 0
from oggm.core.massbalance import (ConstantMassBalance, PastMassBalance)
for gd in gdirs:
    heights, widths = gd.get_inversion_flowline_hw()

    mb_mod = ConstantMassBalance(gd, bias=0)  # bias=0 because of calib!
    mb = mb_mod.get_specific_mb(heights, widths)
    np.testing.assert_allclose(mb, 0, atol=10)  # numerical errors

    mb_mod = PastMassBalance(gd)  # Here we need the computed bias
    refmb = gd.get_ref_mb_data().copy()
    refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths, year=refmb.index)
    np.testing.assert_allclose(refmb.OGGM.mean(), refmb.ANNUAL_BALANCE.mean(),
                               atol=10)

# Log
log.info('Calibration is done!')
