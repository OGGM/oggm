from __future__ import division

import warnings
warnings.filterwarnings("once", category=DeprecationWarning)

import os
import shutil
import glob

import pandas as pd
import salem
import geopandas as gpd
import numpy as np
import shapely.geometry as shpg
import matplotlib.pyplot as plt

# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, rmsd, write_centerlines_to_shape
from oggm.core.models import flowline, massbalance
from oggm import graphics, utils
from oggm.sandbox import itmix
from oggm.sandbox.itmix_cfg import DATA_DIR, WORKING_DIR, PLOTS_DIR
import fiona

# Globals
LOG_DIR = ''

# Funcs
def clean_dir(d):
    shutil.rmtree(d)
    os.makedirs(d)

# Init
cfg.initialize()

# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = False
cfg.CONTINUE_ON_ERROR = False

# Working dir
cfg.PATHS['working_dir'] = WORKING_DIR

cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
cfg.PATHS['topo_dir'] = os.path.join(DATA_DIR, 'topo')

# Set up the paths and other stuffs
cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
cfg.PATHS['cru_dir'] = os.path.join(DATA_DIR, 'cru')
cfg.PATHS['rgi_dir'] = os.path.join(DATA_DIR, 'rgi')

# Update with calib data where ITMIX glaciers have been removed
cfg.PATHS['wgms_rgi_links'] = os.path.join(DATA_DIR, 'itmix',
                                           'wgms',
                                           'rgi_wgms_links_2015_RGIV5.csv')
cfg.PATHS['glathida_rgi_links'] = os.path.join(DATA_DIR, 'itmix',
                                           'glathida',
                                           'rgi_glathida_links_2014_RGIV5.csv')

# Params
cfg.PARAMS['border'] = 20


# Read in the RGI file(s)
rgidf = itmix.get_rgi_df(reset=False)

# Remove problem glaciers
# This is an ice-cap, centerlines wont work
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-07.01394'])]
# This is a huge part of an ice cap thing, not sure if we should use it
# I have the feeling that it blocks the catchment area function...
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-04.06187'])]
# This is a minimops part of Devon
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-03.02479'])]
# Remove all glaciers in Antarctica
rgidf = rgidf.loc[~ rgidf['O1Region'].isin(['19'])]
# This really is an OGGM problem with strange lines
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-02.13974'])]

# Remove bad black rapids:
# rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-01.00037'])]

# rgidf = rgidf.loc[['Columbia' in n for n in rgidf.Name]]
#
# Test problems
# rgidf = rgidf.loc[rgidf.RGIId.isin(['RGI50-01.00037'])]

print('Number of glaciers: {}'.format(len(rgidf)))

# Go
gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
# gdirs = workflow.init_glacier_regions(rgidf)

from oggm import tasks
from oggm.workflow import execute_entity_task

task_list = [
    itmix.glacier_masks_itmix,
    # tasks.compute_centerlines,
    # tasks.catchment_area,
    # tasks.initialize_flowlines,
    # tasks.catchment_width_geom,
    # tasks.catchment_width_correction
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate related tasks
# Only global tasks
# cfg.PARAMS['temp_use_local_gradient'] = False
# tasks.distribute_climate_data(gdirs)
# tasks.compute_ref_t_stars(gdirs)
# tasks.distribute_t_stars(gdirs)

# Inversion
# execute_entity_task(tasks.prepare_for_inversion, gdirs)
# cfg.PARAMS['invert_with_sliding'] = False
# itmix.optimize_thick(gdirs)
# itmix.optimize_per_glacier(gdirs)
# itmix.invert_marine(gdirs)
# execute_entity_task(tasks.volume_inversion, gdirs)

pdir = PLOTS_DIR
if not os.path.exists(pdir):
    os.makedirs(pdir)
for gd in gdirs:
    # graphics.plot_googlemap(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_ggl.png')
    # plt.close()
    graphics.plot_domain(gd)
    plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_dom.png')
    plt.close()
    # graphics.plot_centerlines(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_cls.png')
    # plt.close()
    # graphics.plot_catchment_width(gd, corrected=True)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_w.png')
    # plt.close()
    # graphics.plot_inversion(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_inv.png')
    # plt.close()
    pass
