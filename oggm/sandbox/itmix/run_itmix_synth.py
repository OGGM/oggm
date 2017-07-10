"""ITMIX 'blind run'.

:Author:
    Fabien Maussion
:Date:
    25.03.2016
 """
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
from oggm.workflow import execute_entity_task
from oggm import graphics, utils
from oggm.sandbox import itmix
from oggm.core.preprocessing.inversion import distribute_thickness

# ITMIX paths
DATA_DIR = '/home/mowglie/tmp/RUN_SYNTH/'
WORKING_DIR = '/home/mowglie/tmp/RUN_SYNTH/'
PLOTS_DIR = '/home/mowglie/tmp/RUN_SYNTH/'

# Initialize OGGM
cfg.initialize()

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['continue_on_error'] = False

# Working dir
cfg.PATHS['working_dir'] = WORKING_DIR

# Set up the paths and other stuffs
cfg.PATHS['topo_dir'] = os.path.join(DATA_DIR, 'topo')
cfg.PATHS['itmix_divs'] = '/home/mowglie/synthetic_divides.pkl'
cfg.PATHS['dem_file'] = '/home/mowglie/disk/Dropbox/Share/oggm-data/topo/alternate/ETOPO1_Ice_g_geotiff.tif'

# Read in the RGI file(s)
rgidf = pd.read_pickle('/home/mowglie/synthetic_rgi.pkl')

# Set the newly created divides (quick n dirty fix)
df = gpd.GeoDataFrame.from_file(get_demo_file('divides_workflow.shp'))
dfi = pd.read_pickle(cfg.PATHS['itmix_divs'])
dfi['RGIID'] = dfi['RGIId']
df = pd.concat([df, dfi])
cfg.PARAMS['divides_gdf'] = df.set_index('RGIID')

# Run parameters
cfg.PARAMS['force_one_flowline'] = ['Synthetic1', 'Synthetic2']

cfg.PARAMS['d1'] = 4
cfg.PARAMS['dmax'] = 100

cfg.PARAMS['border'] = 20

cfg.PARAMS['invert_with_sliding'] = False
cfg.PARAMS['min_slope'] = 2
cfg.PARAMS['max_shape_param'] = 0.006
cfg.PARAMS['max_thick_to_width_ratio'] = 0.5
cfg.PARAMS['base_binsize'] = 100.
cfg.PARAMS['temp_use_local_gradient'] = False

# Either do calibration (takes a long time) or do itmix
do_calib = True
do_itmix = True

log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)

# For calibration
if do_calib:
    # gdirs = [gd for gd in gdirs if gd.glacier_type != 'Ice cap']
    # gdirs = [gd for gd in gdirs if gd.terminus_type == 'Land-terminating']

    # Basic tasks
    task_list = [
        itmix.glacier_masks_itmix,
        tasks.compute_centerlines,
        tasks.catchment_area,
        tasks.initialize_flowlines,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction
    ]
    for task in task_list:
        execute_entity_task(task, gdirs)

    # "Climate related tasks"
    for gd in gdirs:
        itmix.synth_apparent_mb(gd)

    # Inversion
    execute_entity_task(tasks.prepare_for_inversion, gdirs)
    fac = 3.22268124479468
    use_cfg_params = {'glen_a':fac * cfg.A, 'fs':0.}
    for gd in gdirs:
        tasks.volume_inversion(gd, use_cfg_params=use_cfg_params)


if do_itmix:

    done = False

    # V1
    distrib = partial(distribute_thickness, how='per_altitude',
                      add_slope=True,
                      smooth=True)
    execute_entity_task(distrib, gdirs)
    pdir = os.path.join(PLOTS_DIR, 'out_dis') + '/'
    if not os.path.exists(pdir):
        os.mkdir(pdir)
    for gd in gdirs:
        itmix.write_itmix_ascii(gd, 1)
        graphics.plot_distributed_thickness(gd)
        plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_d1.png')
        plt.close()

    # V2
    distrib = partial(distribute_thickness, how='per_altitude',
                      add_slope=False,
                      smooth=True)
    execute_entity_task(distrib, gdirs)
    for gd in gdirs:
        itmix.write_itmix_ascii(gd, 2)
        graphics.plot_distributed_thickness(gd)
        plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_d2.png')
        plt.close()


pdir = PLOTS_DIR
if not os.path.exists(pdir):
    os.makedirs(pdir)
for gd in gdirs:
    # graphics.plot_googlemap(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_ggl.png')
    # plt.close()
    # graphics.plot_domain(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_dom.png')
    # plt.close()
    graphics.plot_centerlines(gd)
    plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_cls.png')
    plt.close()
    graphics.plot_catchment_width(gd, corrected=True)
    plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_w.png')
    plt.close()
    graphics.plot_inversion(gd)
    plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_inv.png')
    plt.close()
    # graphics.plot_distributed_thickness(gd, how='per_altitude')
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_dis1.png')
    # plt.close()
    pass
