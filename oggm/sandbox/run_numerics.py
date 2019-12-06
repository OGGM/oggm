# Python imports
import logging
import os

# Libs
import geopandas as gpd

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
rgi_region = '11'  # Region Central Europe

# How many grid points around the glacier?
cfg.PARAMS['border'] = 160

# Make it robust
cfg.PARAMS['continue_on_error'] = True

# Local working directory (where OGGM will write its output)
# WORKING_DIR = os.environ["WORKDIR"]
WORKING_DIR = '/home/mowglie/disk/TMP_Data/run_numerics/'
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

# RGI file
path = utils.get_rgi_region_file(rgi_region, version=rgi_version)
rgidf = gpd.read_file(path)

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False).iloc[:8]

log.workflow('Starting OGGM run')
log.workflow('Number of glaciers: {}'.format(len(rgidf)))

# Go - get the pre-processed glacier directories
gdirs = workflow.init_glacier_regions(rgidf, from_prepro_level=4)

# Tstar runs
y0=None

workflow.execute_entity_task(default_run, gdirs, seed=1, y0=y0,
                             output_filesuffix='_ts_default')
utils.compile_run_output(gdirs, input_filesuffix='_ts_default')

workflow.execute_entity_task(default_run, gdirs, seed=1, y0=y0,
                             flux_limiter=True,
                             output_filesuffix='_ts_limited')
utils.compile_run_output(gdirs, input_filesuffix='_ts_limited')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             output_filesuffix='_ts_nomindt')
utils.compile_run_output(gdirs, input_filesuffix='_ts_nomindt')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             cfl_number=0.01,
                             output_filesuffix='_ts_cfl')
utils.compile_run_output(gdirs, input_filesuffix='_ts_cfl')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             cfl_number=0.01, flux_limiter=True,
                             output_filesuffix='_ts_cfl_limited')
utils.compile_run_output(gdirs, input_filesuffix='_ts_cfl_limited')

# Tstar cooling
y0=None
temp_bias=-0.2

workflow.execute_entity_task(default_run, gdirs, seed=1, y0=y0,
                             temp_bias=temp_bias,
                             output_filesuffix='_tsc_default')
utils.compile_run_output(gdirs, input_filesuffix='_tsc_default')

workflow.execute_entity_task(default_run, gdirs, seed=1, y0=y0,
                             temp_bias=temp_bias,
                             flux_limiter=True,
                             output_filesuffix='_tsc_limited')
utils.compile_run_output(gdirs, input_filesuffix='_tsc_limited')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             temp_bias=temp_bias,
                             output_filesuffix='_tsc_nomindt')
utils.compile_run_output(gdirs, input_filesuffix='_tsc_nomindt')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             temp_bias=temp_bias,
                             cfl_number=0.01,
                             output_filesuffix='_tsc_cfl')
utils.compile_run_output(gdirs, input_filesuffix='_tsc_cfl')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             temp_bias=temp_bias,
                             cfl_number=0.01, flux_limiter=True,
                             output_filesuffix='_tsc_cfl_limited')
utils.compile_run_output(gdirs, input_filesuffix='_tsc_cfl_limited')

# Commit runs
y0=2000

workflow.execute_entity_task(default_run, gdirs, seed=1, y0=y0,
                             output_filesuffix='_co_default')
utils.compile_run_output(gdirs, input_filesuffix='_co_default')

workflow.execute_entity_task(default_run, gdirs, seed=1, y0=y0,
                             flux_limiter=True,
                             output_filesuffix='_co_limited')
utils.compile_run_output(gdirs, input_filesuffix='_co_limited')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             output_filesuffix='_co_nomindt')
utils.compile_run_output(gdirs, input_filesuffix='_co_nomindt')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             cfl_number=0.01,
                             output_filesuffix='_co_cfl')
utils.compile_run_output(gdirs, input_filesuffix='_co_cfl')

workflow.execute_entity_task(better_run, gdirs, seed=1, y0=y0, min_dt=600,
                             cfl_number=0.01, flux_limiter=True,
                             output_filesuffix='_co_cfl_limited')
utils.compile_run_output(gdirs, input_filesuffix='_co_cfl_limited')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.workflow('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
