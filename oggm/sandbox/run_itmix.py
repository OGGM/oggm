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
import matplotlib.pyplot as plt

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, rmsd, write_centerlines_to_shape
from oggm.core.models import flowline, massbalance
from oggm import graphics

# Globals
LOG_DIR = ''
# Funcs
def clean_dir(d):
    shutil.rmtree(d)
    os.makedirs(d)

# Init
cfg.initialize()

# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True
cfg.CONTINUE_ON_ERROR = True

# Working dir
cfg.PATHS['working_dir'] = '/home/mowglie/disk/Data/ITMIX/oggm'

cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
cfg.PATHS['topo_dir'] = '/home/mowglie/disk/Dropbox/Share/oggm-data/topo'

# Set up the paths and other stuffs
cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
cfg.PATHS['histalp_file'] = get_demo_file('HISTALP_oeztal.nc')

# Get test glaciers (all glaciers with MB or Thickness data)
cfg.PATHS['wgms_rgi_links'] = get_demo_file('RGI_WGMS_oeztal.csv')
cfg.PATHS['glathida_rgi_links'] = get_demo_file('RGI_GLATHIDA_oeztal.csv')

# Read in the RGI file(s)
rgi_dir = '/home/mowglie/disk/Data/GIS/SHAPES/RGI/RGI_V5/'
df_itmix = pd.read_pickle('/home/mowglie/disk/Dropbox/Photos/itmix_rgi_plots/links/itmix_rgi_links.pkl')


df_rgi_file = '/home/mowglie/itmix_rgi_shp_t2.pkl'
if os.path.exists(df_rgi_file):
    rgidf = pd.read_pickle(df_rgi_file)
else:
    rgidf = []
    for i, row in df_itmix.iterrows():
        rgi_shp = list(glob.glob(os.path.join(rgi_dir, "*", row['rgi_reg'] + '_rgi50_*.shp')))[0]
        rgi_df = salem.utils.read_shapefile(rgi_shp, cached=True)
        rgi_parts = row.T['rgi_parts_ids']
        # if 'RGI50-09.00910' in rgi_parts:
        #     rgi_parts = ['RGI50-09.00910']
        # else:
        #     continue
        sel = rgi_df.loc[rgi_df.RGIId.isin(rgi_parts)]

        # add glacier name to the entity
        sel.loc[:, 'Name'] = [row.name] * len(sel)
        rgidf.append(sel)
    rgidf = pd.concat(rgidf)
    rgidf.to_pickle(df_rgi_file)

# import geopandas as gpd
# tosh = gpd.GeoDataFrame(rgidf)
# tosh.to_file('/home/mowglie/hell.shp')

# Params
cfg.PARAMS['border'] = 20

# Go
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)

from oggm import tasks
from oggm.workflow import execute_entity_task

# task_list = [
#     tasks.glacier_masks,
#     tasks.compute_centerlines,
#     # tasks.catchment_area,
#     # tasks.initialize_flowlines,
#     # tasks.catchment_width_geom
# ]
# for task in task_list:
#     execute_entity_task(task, gdirs)


dir = '/home/mowglie/disk/Data/ITMIX/oggm_plots/'
from oggm import graphics
for gd in gdirs:
    graphics.plot_domain(gd)
    plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_dom.png')
    plt.close()
    graphics.plot_centerlines(gd)
    plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_cls.png')
    plt.close()

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