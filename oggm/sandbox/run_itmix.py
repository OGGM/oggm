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
from oggm import graphics
from oggm.sandbox import itmix
from oggm.sandbox import itmix_cfg

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
cfg.CONTINUE_ON_ERROR = False

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

reset = False
df_rgi_file = '/home/mowglie/itmix_rgi_shp.pkl'
if os.path.exists(df_rgi_file) and not reset:
    rgidf = pd.read_pickle(df_rgi_file)
else:
    rgidf = []
    for i, row in df_itmix.iterrows():
        # read the rgi region
        rgi_shp = list(glob.glob(os.path.join(rgi_dir, "*", row['rgi_reg'] + '_rgi50_*.shp')))[0]
        rgi_df = salem.utils.read_shapefile(rgi_shp, cached=True)

        rgi_parts = row.T['rgi_parts_ids']
        sel = rgi_df.loc[rgi_df.RGIId.isin(rgi_parts)].copy()

        # use the ITMIX shape where possible
        if row.name in ['Hellstugubreen', 'Freya', 'Aqqutikitsoq',
                        'Brewster', 'Kesselwandferner', 'NorthGlacier',
                        'SouthGlacier', 'Tasman', 'Unteraar',
                        'Washmawapta']:
            for shf in glob.glob(itmix_cfg.itmix_data_dir + '*/*/*_' +
                                         row.name + '*.shp'):
                pass
            shp = salem.utils.read_shapefile(shf)
            if row.name == 'Unteraar':
                shp = shp.iloc[[-1]]
            if 'LineString' == shp.iloc[0].geometry.type:
                shp.loc[shp.index[0], 'geometry'] = shpg.Polygon(shp.iloc[0].geometry)
            assert len(shp) == 1
            area_km2 = shp.iloc[0].geometry.area * 1e-6
            shp = salem.gis.transform_geopandas(shp)
            shp = shp.iloc[0].geometry
            sel = sel.iloc[[0]]
            sel.loc[sel.index[0], 'geometry'] = shp
            sel.loc[sel.index[0], 'Area'] = area_km2
            sel.loc[sel.index[0], 'RGIId'] = 'ITMIX:' + sel.loc[sel.index[0], 'RGIId']
        elif row.name == 'Urumqi':
            # ITMIX Urumqi is in fact two glaciers
            for shf in glob.glob(itmix_cfg.itmix_data_dir + '*/*/*_' +
                                         row.name + '*.shp'):
                pass
            shp2 = salem.utils.read_shapefile(shf)
            assert len(shp2) == 2
            for k in [0, 1]:
                shp = shp2.iloc[[k]].copy()
                area_km2 = shp.iloc[0].geometry.area * 1e-6
                shp = salem.gis.transform_geopandas(shp)
                shp = shp.iloc[0].geometry
                assert sel.loc[sel.index[k], 'geometry'].contains(shp.centroid)
                sel.loc[sel.index[k], 'geometry'] = shp
                sel.loc[sel.index[k], 'Area'] = area_km2
                sel.loc[sel.index[k], 'RGIId'] = 'ITMIX:' + sel.loc[sel.index[k], 'RGIId']
            assert len(sel) == 2
        else:
            pass
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

task_list = [
    itmix.glacier_masks_itmix,
    # tasks.compute_centerlines,
    # tasks.catchment_area,
    # tasks.initialize_flowlines,
    # tasks.catchment_width_geom
]
for task in task_list:
    execute_entity_task(task, gdirs)


dir = '/home/mowglie/disk/Data/ITMIX/oggm_plots/'
from oggm import graphics
for gd in gdirs:
    graphics.plot_domain(gd)
    plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_dom.png')
    plt.close()
    # graphics.plot_centerlines(gd)
    # plt.savefig(dir + gd.name + '_' + gd.rgi_id + '_cls.png')
    # plt.close()

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