"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Log message format
import logging

logging.getLogger("rasterio").setLevel(logging.WARNING)
logging.getLogger("shapely").setLevel(logging.WARNING)

logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# Module logger
log = logging.getLogger(__name__)

# Python imports
import os
import glob
# Libs
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import salem
# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils

# Initialize OGGM
cfg.initialize()

# Local paths (where to write output and where to download input)
_dir = '/home/mowglie/disk/TMP/OGGM_runs/test_dem/'
WORKING_DIR = _dir + 'run/'
utils.mkdir(WORKING_DIR)
PLOTS_DIR = _dir + 'plots'
cfg.PATHS['working_dir'] = WORKING_DIR
cfg.PATHS['topo_dir'] = _dir + 'data/'

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['border'] = 20
cfg.CONTINUE_ON_ERROR = False

# Read in the RGI file
rgisel = os.path.join(cfg.PATHS['topo_dir'], 'rgi_sel.shp')
if not os.path.exists(rgisel):
    rgidir = '/home/mowglie/disk/Data/GIS/SHAPES/RGI/RGI_V5/'
    regions = ['{:02d}'.format(int(p)) for p in range(1, 20)]
    files = [glob.glob(os.path.join(rgidir, '*', r + '_rgi50_*.shp'))[0] for r in regions]
    rgidf = []
    for fs in files:
        sh = salem.read_shapefile(os.path.join(rgidir, fs), cached=True)

        percs = np.asarray([0, 25, 50, 75, 100])
        idppercs = np.round(percs * 0.01 * (len(sh)-1)).astype(int)

        rgidf.append(sh.sort_values(by='Area').iloc[idppercs])
        rgidf.append(sh.sort_values(by='CenLon').iloc[idppercs])
        rgidf.append(sh.sort_values(by='CenLat').iloc[idppercs])
    rgidf = gpd.GeoDataFrame(pd.concat(rgidf))
    rgidf = rgidf.drop_duplicates('RGIId')
    rgidf.to_file(rgisel)
else:
    rgidf = salem.read_shapefile(rgisel)

# rgidf = rgidf.loc[rgidf.O1Region.isin(['2'])]
rgidf = rgidf.loc[~rgidf.RGIId.isin(['RGI50-10.00012', 'RGI50-17.00850',
                                     'RGI50-19.01497', 'RGI50-19.00990',
                                     'RGI50-19.01440'])]

log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)

# Prepro tasks
task_list = [
    tasks.glacier_masks,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Plots (if you want)
if PLOTS_DIR == '':
    exit()

utils.mkdir(PLOTS_DIR)
for gd in gdirs:

    bname = os.path.join(PLOTS_DIR, gd.rgi_id + '_')
    demsource = ' (' + gd.read_pickle('dem_source') + ')'

    # graphics.plot_googlemap(gd)
    # plt.savefig(bname + 'ggl.png')
    # plt.close()
    graphics.plot_domain(gd, title_comment=demsource)
    plt.savefig(bname + 'dom.png')
    plt.close()
