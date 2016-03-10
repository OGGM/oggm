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
from oggm.sandbox import itmix_cfg
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
cfg.PATHS['working_dir'] = '/home/mowglie/disk/TMP/ITMIX/OGGM_ITMIX_WD_nograd'

cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
cfg.PATHS['topo_dir'] = '/home/mowglie/disk/Dropbox/Share/oggm-data/topo'

# Set up the paths and other stuffs
cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
cfg.PATHS['cru_dir'] = '/home/mowglie/disk/Data/Gridded/CRU'

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

# Test problems
# rgidf = rgidf.loc[rgidf.RGIId.isin(['RGI50-01.00037'])]

print('Number of glaciers: {}'.format(len(rgidf)))

# Go
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)
rgidf['glacier_type'] = [gd.glacier_type for gd in gdirs]
rgidf['terminus_type'] = [gd.terminus_type for gd in gdirs]

# towrite = rgidf[['Name', 'RGIId', 'Area', 'Aspect', 'CenLat', 'CenLon', 'Lmax',
#                 'Slope', 'Zmax', 'Zmed', 'Zmin', 'glacier_type',
#                 'terminus_type']]
# towrite.to_csv('/home/mowglie/disk/Data/ITMIX/final_glacier_list.csv')

from oggm import tasks
from oggm.workflow import execute_entity_task

# task_list = [
    # itmix.glacier_masks_itmix,
    # tasks.compute_centerlines,
    # tasks.catchment_area,
    # tasks.initialize_flowlines,
    # tasks.catchment_width_geom,
    # tasks.catchment_width_correction
# ]
# for task in task_list:
#     execute_entity_task(task, gdirs)

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
# execute_entity_task(tasks.volume_inversion, gdirs)

dir = '/home/mowglie/disk/TMP/ITMIX/OGGM_ITMIX_PLOTS/'
for gd in gdirs:
    # graphics.plot_googlemap(gd)
    # plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_ggl.png')
    # plt.close()
    # graphics.plot_domain(gd)
    # plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_dom.png')
    # plt.close()
    # graphics.plot_centerlines(gd)
    # plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_cls.png')
    # plt.close()
    # graphics.plot_catchment_width(gd, corrected=True)
    # plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_w.png')
    # plt.close()
    graphics.plot_inversion(gd)
    plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_inv.png')
    plt.close()
    pass
#
# try:
#     flowline.init_present_time_glacier(gdirs[0])
# except Exception:
#     reset = True
#
# if reset:
#     # First preprocessing tasks
#     workflow.gis_prepro_tasks(gdirs)
#
#     # Climate related tasks
#     workflow.climate_tasks(gdirs)
#
#     # Inversion
#     workflow.inversion_tasks(gdirs)