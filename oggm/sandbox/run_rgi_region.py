"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Log message format
import logging
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Module logger
log = logging.getLogger(__name__)

# Python imports
import os
from glob import glob
import shutil
# Libs
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils

# This will run OGGM on the RGI region of your choice
# After preprocessing, the glaciers are run for 1000 years in a random climate

# Regions:
# Alaska 01
# Western Canada and US 02
# Arctic Canada North 03
# Arctic Canada South 04
# Greenland 05
# Iceland 06
# Svalbard 07
# Scandinavia 08
# Russian Arctic 09
# North Asia 10
# North Asia 10
# Central Europe 11
# Caucasus and Middle East 12
# Central Asia 13
# South Asia West 14
# South Asia East 15
# Low Latitudes 16
# Southern Andes 17
# New Zealand 18
# Antarctic and Subantarctic 19
rgi_reg = '11'  # alps

# Some globals for more control on what to run
RUN_GIS_PREPRO = True  # run GIS preprocessing tasks (before climate)
RUN_CLIMATE_PREPRO = True  # run climate preprocessing tasks
RUN_INVERSION = True  # run bed inversion
RUN_DYNAMICS = True  # run dybnamics

# Initialize OGGM
cfg.initialize()

# Local paths (where to write output and where to download input)
WORKING_DIR = '/work/ubuntu/run_alps'
DATA_DIR = '/work/ubuntu/oggm-data'

cfg.PATHS['working_dir'] = WORKING_DIR
cfg.PATHS['cru_dir'] = os.path.join(DATA_DIR, 'cru')
cfg.PATHS['rgi_dir'] = os.path.join(DATA_DIR, 'rgi')

# Currently OGGM wants some directories to exist
# (maybe I'll change this but it can also catch errors in the user config)
utils.mkdir(cfg.PATHS['working_dir'])
utils.mkdir(cfg.PATHS['cru_dir'])
utils.mkdir(cfg.PATHS['rgi_dir'])

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True
cfg.CONTINUE_ON_ERROR = True

# Other params
cfg.PARAMS['border'] = 80
cfg.PARAMS['temp_use_local_gradient'] = False
cfg.PARAMS['invert_with_sliding'] = False

# Download RGI files
rgi_dir = utils.get_rgi_dir()
rgi_shp = list(glob(os.path.join(rgi_dir, "*", rgi_reg+ '_rgi50_*.shp')))
assert len(rgi_shp) == 1
rgidf = gpd.read_file(rgi_shp[0])

log.info('Number of glaciers: {}'.format(len(rgidf)))

# Download other files if needed
_ = utils.get_cru_file(var='tmp')
_ = utils.get_cru_file(var='pre')
_ = utils.get_demo_file('Hintereisferner.shp')

# Go - initialize working directories
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)

# Prepro tasks
task_list = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.compute_downstream_lines,
    tasks.catchment_area,
    tasks.initialize_flowlines,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction
]
if RUN_GIS_PREPRO:
    for task in task_list:
        execute_entity_task(task, gdirs)

if RUN_CLIMATE_PREPRO:
    # Climate related tasks
    # see if we can distribute
    workflow.execute_entity_task(tasks.process_cru_data, gdirs)
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)

if RUN_INVERSION:
    # Inversion
    execute_entity_task(tasks.prepare_for_inversion, gdirs)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)

if RUN_DYNAMICS:
    # Random dynamics
    execute_entity_task(tasks.init_present_time_glacier, gdirs)
    execute_entity_task(tasks.random_glacier_evolution, gdirs)

# Plots (if you want)
PLOTS_DIR = ''
if PLOTS_DIR == '':
    exit()
utils.mkdir(PLOTS_DIR)
for gd in gdirs:
    bname = os.path.join(PLOTS_DIR, gd.name + '_' + gd.rgi_id + '_')
    graphics.plot_googlemap(gd)
    plt.savefig(bname + 'ggl.png')
    plt.close()
    graphics.plot_domain(gd)
    plt.savefig(bname + 'dom.png')
    plt.close()
    graphics.plot_centerlines(gd)
    plt.savefig(bname + 'cls.png')
    plt.close()
    graphics.plot_catchment_width(gd, corrected=True)
    plt.savefig(bname + 'w.png')
    plt.close()
    graphics.plot_inversion(gd)
    plt.savefig(bname + 'inv.png')
    plt.close()
