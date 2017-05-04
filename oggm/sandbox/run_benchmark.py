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
import shutil
from functools import partial
# Libs
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely.geometry as shpg
import matplotlib.pyplot as plt
# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file
from oggm import tasks
from oggm.workflow import execute_entity_task, reset_multiprocessing
from oggm import graphics, utils


# Initialize OGGM
cfg.initialize()

# Local paths (where to write output and where to download input)
WORKING_DIR = os.path.expanduser('~/OGGM_wd_bench')
DATA_DIR = os.path.join(WORKING_DIR, 'datadir')

cfg.PATHS['working_dir'] = WORKING_DIR
cfg.PATHS['topo_dir'] = os.path.join(DATA_DIR, 'topo')
cfg.PATHS['cru_dir'] = os.path.join(DATA_DIR, 'cru')
cfg.PATHS['rgi_dir'] = os.path.join(DATA_DIR, 'rgi')
cfg.PATHS['tmp_dir'] = os.path.join(DATA_DIR, 'tmp')

# Currently OGGM wants some directories to exist
# (maybe I'll change this but it can also catch errors in the user config)
utils.mkdir(cfg.PATHS['working_dir'])
utils.mkdir(cfg.PATHS['topo_dir'])
utils.mkdir(cfg.PATHS['cru_dir'])
utils.mkdir(cfg.PATHS['rgi_dir'])
utils.mkdir(cfg.PATHS['tmp_dir'])

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True
cfg.CONTINUE_ON_ERROR = False

# Read in the Benchmark RGI file
rgi_pkl_path = utils.aws_file_download('rgi_benchmark.pkl')
rgidf = pd.read_pickle(rgi_pkl_path)

# Remove glaciers causing issues
rgidf = rgidf.iloc[[s not in ('RGI50-11.00291', 'RGI50-03.02479') for s in rgidf['RGIId']]]

utils.get_rgi_dir()

log.info('Number of glaciers: {}'.format(len(rgidf)))

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
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate related tasks - this will download
execute_entity_task(tasks.process_cru_data, gdirs)
tasks.compute_ref_t_stars(gdirs)
tasks.distribute_t_stars(gdirs)

# Inversion
execute_entity_task(tasks.prepare_for_inversion, gdirs)
tasks.optimize_inversion_params(gdirs)
execute_entity_task(tasks.volume_inversion, gdirs)

# Write out glacier statistics
df = utils.glacier_characteristics(gdirs)
fpath = os.path.join(cfg.PATHS['working_dir'], 'glacier_char.csv')
df.to_csv(fpath)


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
