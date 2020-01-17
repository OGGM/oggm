# Python imports
import logging
import os

# Libs
import geopandas as gpd
import pandas as pd

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow
from oggm.sandbox.numerical_benchmarks import default_run, better_run

# For timing the run
import time
start = time.time()

# Module logger
log = logging.getLogger(__name__)

# Initialize OGGM and set up the default run parameters
cfg.initialize()
rgi_version = '61'
rgi_reg = '11'  # Region Central Europe
# rgi_reg = '{:02}'.format(int(os.environ.get('RGI_REG')))

# How many grid points around the glacier?
cfg.PARAMS['border'] = 160

# Make it robust
cfg.PARAMS['continue_on_error'] = True

# Local working directory (where OGGM will write its output)
WORKING_DIR = '/home/mowglie/disk/TMP_Data/run_numerics/'
# WORKING_DIR = os.environ["WORKDIR"]
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

# RGI file
path = utils.get_rgi_region_file(rgi_reg, version=rgi_version)
rgidf = gpd.read_file(path)

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False).iloc[:2]

log.workflow('Starting OGGM run')
log.workflow('Number of glaciers: {}'.format(len(rgidf)))

# Go - get the pre-processed glacier directories
cfg.PARAMS['use_multiprocessing'] = False
gdirs = workflow.init_glacier_regions(rgidf, from_prepro_level=4)
cfg.PARAMS['use_multiprocessing'] = True

# Runs
times = pd.DataFrame()
for y0, temp_bias, exp in zip([None],
                              [-0.5],
                              ['_tsc']):

    task_names = []

    tn = exp + '_default'
    workflow.execute_entity_task(default_run, gdirs, seed=1, y0=y0,
                                 temp_bias=temp_bias,
                                 output_filesuffix=tn)
    utils.compile_run_output(gdirs, input_filesuffix=tn)
    task_names.append('default_run' + tn)

    tn = exp + '_cfl_05'
    workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=60,
                                 temp_bias=temp_bias,
                                 cfl_number=0.05,
                                 max_dt=cfg.SEC_IN_YEAR,
                                 monthly_steps=True,
                                 output_filesuffix=tn)
    utils.compile_run_output(gdirs, input_filesuffix=tn)
    task_names.append('better_run' + tn)

    tn = exp + '_cfl_03'
    workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=60,
                                 temp_bias=temp_bias,
                                 cfl_number=0.03,
                                 max_dt=cfg.SEC_IN_YEAR,
                                 monthly_steps=True,
                                 output_filesuffix=tn)
    utils.compile_run_output(gdirs, input_filesuffix=tn)
    task_names.append('better_run' + tn)

    tn = exp + '_cfl_02'
    workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=60,
                                 temp_bias=temp_bias,
                                 cfl_number=0.02,
                                 max_dt=cfg.SEC_IN_YEAR,
                                 monthly_steps=True,
                                 output_filesuffix=tn)
    utils.compile_run_output(gdirs, input_filesuffix=tn)
    task_names.append('better_run' + tn)

    tn = exp + '_cfl_01'
    workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=60,
                                 temp_bias=temp_bias,
                                 cfl_number=0.01,
                                 max_dt=cfg.SEC_IN_YEAR,
                                 monthly_steps=True,
                                 output_filesuffix=tn)
    utils.compile_run_output(gdirs, input_filesuffix=tn)
    task_names.append('better_run' + tn)

    tn = exp + '_cfl_005'
    workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=30,
                                 temp_bias=temp_bias,
                                 cfl_number=0.005,
                                 max_dt=cfg.SEC_IN_YEAR,
                                 monthly_steps=True,
                                 output_filesuffix=tn)
    utils.compile_run_output(gdirs, input_filesuffix=tn)
    task_names.append('better_run' + tn)

    utils.compile_task_log(gdirs, task_names=task_names, filesuffix=exp)
    utils.compile_task_time(gdirs, task_names=task_names, filesuffix=exp)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.workflow('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
