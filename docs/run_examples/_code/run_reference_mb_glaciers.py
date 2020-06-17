# Python imports
import json
import os
from distutils.dir_util import copy_tree

# Libs
import numpy as np
import pandas as pd

# Locals
import oggm
from oggm import cfg, utils, tasks
from oggm.shop import rgitopo
from oggm.workflow import execute_entity_task, init_glacier_directories
from oggm.core.massbalance import (ConstantMassBalance, PastMassBalance,
                                   MultipleFlowlineMassBalance)

# Module logger
import logging
log = logging.getLogger(__name__)

# RGI Version
rgi_version = '62'

# Start from zero?
reset = False

# Initialize OGGM and set up the run parameters
cfg.initialize()

# Local paths (where to write the OGGM run output)
dirname = 'OGGM_ref_mb_RGIV{}_OGGM{}'.format(rgi_version, oggm.__version__)
WORKING_DIR = utils.gettempdir(dirname, home=True)
utils.mkdir(WORKING_DIR, reset=reset)

cfg.PATHS['working_dir'] = WORKING_DIR

# We are running the calibration ourselves
cfg.PARAMS['run_mb_calibration'] = True

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# Set to True for operational runs - here we want all glaciers to run
cfg.PARAMS['continue_on_error'] = False

# No need for a big map here
cfg.PARAMS['border'] = 10

# Currently verifying ECMWF data is a pain - we need to change that
cfg.PARAMS['dl_verify'] = False

# Prepare the preprocessed dirs
df, _ = utils.get_wgms_files()
rids = df['RGI{}0_ID'.format(rgi_version[0])]

per_glacier_dir = os.path.join(WORKING_DIR, 'per_glacier')
def_per_glacier_dir = os.path.join(WORKING_DIR, 'per_glacier_def')

if reset or not os.path.isdir(def_per_glacier_dir):
    gdirs = rgitopo.init_glacier_directories_from_rgitopo(rids)
else:
    utils.mkdir(per_glacier_dir, reset=True)
    copy_tree(def_per_glacier_dir, per_glacier_dir)
    gdirs = init_glacier_directories(rids)

try:
    gdirs[0].read_pickle('inversion_flowlines')
except FileNotFoundError:
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

    utils.mkdir(def_per_glacier_dir, reset=True)
    copy_tree(per_glacier_dir, def_per_glacier_dir)

for baseline in ['CRU', 'ERA5', 'ERA5L', 'CERA+ERA5', 'CERA+ERA5L']:

    # We are using which baseline data?
    cfg.PARAMS['baseline_climate'] = baseline

    # Here we will need data-specific params
    if baseline == 'CRU':
        # Default params
        pass
    elif baseline == 'HISTALP':
        # Other params: see https://oggm.org/2018/08/10/histalp-parameters/
        cfg.PARAMS['prcp_scaling_factor'] = 1.75
        cfg.PARAMS['temp_melt'] = -1.75
    elif baseline == 'ERA5':
        # Other params from mdusch
        cfg.PARAMS['prcp_scaling_factor'] = 2.0
        cfg.PARAMS['temp_melt'] = -1.25
    elif baseline == 'ERA5L':
        # Other params from mdusch
        cfg.PARAMS['prcp_scaling_factor'] = 2.0
        cfg.PARAMS['temp_melt'] = -1.0
    elif baseline == 'CERA+ERA5':
        # Other params from mdusch
        cfg.PARAMS['prcp_scaling_factor'] = 1.75
        cfg.PARAMS['temp_melt'] = -1.25
    elif baseline == 'CERA+ERA5L':
        # Other params from mdusch
        cfg.PARAMS['prcp_scaling_factor'] = 1.75
        cfg.PARAMS['temp_melt'] = -0.75

    ref_list_f = os.path.join(WORKING_DIR,
                              'mb_ref_glaciers_{}.csv'.format(baseline))
    ref_per_glacier_dir = os.path.join(WORKING_DIR, 'per_glacier_{}'.format(baseline))

    if not os.path.exists(ref_list_f) or not os.path.isdir(ref_per_glacier_dir):

        # Get the reference glacier ids
        df, _ = utils.get_wgms_files()
        rids = df['RGI{}0_ID'.format(rgi_version[0])]

        if baseline == 'HISTALP':
            # For HISTALP only RGI reg 11
            rids = [rid for rid in rids if '-11.' in rid]
        elif baseline == 'CRU':
            # For CRU we can't do Antarctica
            rids = [rid for rid in rids if not ('-19.' in rid)]

        # For histalp do a second filtering for glaciers in the Alps only
        if baseline == 'HISTALP':
            gdirs = [gdir for gdir in gdirs if gdir.rgi_subregion == '11-01']

        # We need to know which period we have data for
        log.info('Process the climate data...')
        execute_entity_task(tasks.process_climate_data, gdirs, print_log=False)

        # Let OGGM decide which of these have enough data
        gdirs = utils.get_ref_mb_glaciers(gdirs)

        # Save the list of glaciers for later
        log.info('For RGIV{} and {} we have {} reference '
                 'glaciers.'.format(rgi_version, baseline, len(gdirs)))
        rgidf = pd.Series(data=[g.rgi_id for g in gdirs])
        rgidf.to_csv(ref_list_f, header=False)

        # Copy the glacier dirs with appropriate data in it
        copy_tree(per_glacier_dir, ref_per_glacier_dir)

    else:
        # Read the rgi ids of the reference glaciers
        rids = pd.read_csv(ref_list_f, index_col=0, squeeze=True)

        # Copy the ref dir
        utils.mkdir(per_glacier_dir, reset=True)
        copy_tree(ref_per_glacier_dir, per_glacier_dir)

        gdirs = init_glacier_directories(rids)

    # Climate tasks
    execute_entity_task(tasks.historical_climate_qc, gdirs)
    tasks.compute_ref_t_stars(gdirs)
    execute_entity_task(tasks.local_t_star, gdirs)
    execute_entity_task(tasks.mu_star_calibration, gdirs)

    # We store the associated params
    mb_calib = gdirs[0].get_climate_info()['mb_calib_params']
    params_file = os.path.join(WORKING_DIR, 'mb_calib_params_'
                                            '{}.json'.format(baseline))
    with open(params_file, 'w') as fp:
        json.dump(mb_calib, fp)

    # And also some statistics
    diag_path = os.path.join(WORKING_DIR, 'glacier_statistics_'
                                          '{}.csv'.format(baseline))
    utils.compile_glacier_statistics(gdirs, path=diag_path)

    # Tests: for all glaciers, the mass-balance around tstar and the
    # bias with observation should be approx 0
    for gd in gdirs:
        mb_mod = MultipleFlowlineMassBalance(gd,
                                             mb_model_class=ConstantMassBalance,
                                             use_inversion_flowlines=True,
                                             bias=0)  # bias=0 because of calib!
        mb = mb_mod.get_specific_mb()
        np.testing.assert_allclose(mb, 0, atol=5)  # atol for numerical errors

        mb_mod = MultipleFlowlineMassBalance(gd, mb_model_class=PastMassBalance,
                                             use_inversion_flowlines=True)

        refmb = gd.get_ref_mb_data().copy()
        refmb['OGGM'] = mb_mod.get_specific_mb(year=refmb.index)
        np.testing.assert_allclose(refmb.OGGM.mean(), refmb.ANNUAL_BALANCE.mean(),
                                   atol=15)  # atol for numerical errors

    # Log
    log.info('Calibration for {} is done!'.format(baseline))
