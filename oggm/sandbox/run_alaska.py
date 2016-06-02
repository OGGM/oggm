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
import salem
import pandas as pd
import matplotlib.pyplot as plt
# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils

# This will run OGGM in alaska

# Initialize OGGM
cfg.initialize()

# Local paths (where to write output and where to download input)
WORKING_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/work_dir/'
DATA_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/input_data/'
RGI_FILE = '/home/beatriz/Documents/global_data_base/Alaska_tidewater_andlake/01_rgi50_Alaska.shp'

cfg.PATHS['working_dir'] = WORKING_DIR
cfg.PATHS['topo_dir'] = os.path.join(DATA_DIR, 'topo')
cfg.PATHS['cru_dir'] = os.path.join(DATA_DIR, 'cru')

# Currently OGGM wants some directories to exist
# (maybe I'll change this but it can also catch errors in the user config)
utils.mkdir(cfg.PATHS['working_dir'])
utils.mkdir(cfg.PATHS['topo_dir'])
utils.mkdir(cfg.PATHS['cru_dir'])

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = False
cfg.CONTINUE_ON_ERROR = False

# Other params
cfg.PARAMS['border'] = 80
cfg.PARAMS['temp_use_local_gradient'] = False
cfg.PARAMS['optimize_inversion_params'] = False
cfg.PARAMS['invert_with_sliding'] = False
cfg.PARAMS['bed_shape'] = 'parabolic'

# Some globals for more control on what to run
RUN_GIS_PREPRO = False  # run GIS preprocessing tasks (before climate)
RUN_CLIMATE_PREPRO = False  # run climate preprocessing tasks
RUN_INVERSION = True  # run bed inversion
RUN_DYNAMICS = False  # run dybnamics

# Read RGI file
rgidf = salem.utils.read_shapefile(RGI_FILE, cached=True)

# Select some glaciers

# Get ref glaciers (all glaciers with MB)
flink, mbdatadir = utils.get_wgms_files()
ids_with_mb = pd.read_csv(flink)['RGI_ID'].values
keep_ids = ['RGI50-01.20791', 'RGI50-01.00037', 'RGI50-01.10402']
keep_indexes = [((i in keep_ids) or (i in ids_with_mb)) for i in rgidf.RGIID]
rgidf = rgidf.iloc[keep_indexes]

# keep_ids = ['RGI50-01.20791']
# keep_indexes = [(i in keep_ids) for i in rgidf.RGIID]
# rgidf = rgidf.iloc[keep_indexes]

log.info('Number of glaciers: {}'.format(len(rgidf)))

# Download other files if needed
_ = utils.get_cru_file(var='tmp')

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
    workflow.execute_entity_task(tasks.distribute_cru_style, gdirs)
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)

if RUN_INVERSION:
    # Inversion
    execute_entity_task(tasks.prepare_for_inversion, gdirs)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)

# if RUN_DYNAMICS:
#     # Random dynamics
#     execute_entity_task(tasks.init_present_time_glacier, gdirs)
#     execute_entity_task(tasks.random_glacier_evolution, gdirs)

# Plots (if you want)
PLOTS_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/plots/'
if PLOTS_DIR == '':
    exit()
utils.mkdir(PLOTS_DIR)
for gd in gdirs:
    bname = os.path.join(PLOTS_DIR, gd.name + '_' + gd.rgi_id + '_')
    # graphics.plot_googlemap(gd)
    # plt.savefig(bname + 'ggl.png')
    # plt.close()
    # graphics.plot_centerlines(gd, add_downstream=True)
    # plt.savefig(bname + 'cls.png')
    # plt.close()
    # graphics.plot_catchment_width(gd, corrected=True)
    # plt.savefig(bname + 'w.png')
    # plt.close()
    graphics.plot_inversion(gd)
    plt.savefig(bname + 'inv.png')
    plt.close()
