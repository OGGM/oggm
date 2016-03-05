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
cfg.PATHS['working_dir'] = '/home/mowglie/OGGM_ITMIX_WD'

cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
cfg.PATHS['topo_dir'] = '/home/mowglie/disk/Dropbox/Share/oggm-data/topo'

# Set up the paths and other stuffs
cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
cfg.PATHS['cru_dir'] = '/home/mowglie/disk/Data/Gridded/CRU'

# Get test glaciers (all glaciers with MB or Thickness data)
# cfg.PATHS['glathida_rgi_links'] = get_demo_file('RGI_GLATHIDA_oeztal.csv')

# Read in the RGI file(s)

# This makes an RGI dataframe with all ITMIX + WGMS + GTD glaciers
reset = False
df_rgi_file = '/home/mowglie/itmix_rgi_shp.pkl'
if os.path.exists(df_rgi_file) and not reset:
    rgidf = pd.read_pickle(df_rgi_file)
else:

    rgi_dir = '/home/mowglie/disk/Data/GIS/SHAPES/RGI/RGI_V5/'
    df_itmix = pd.read_pickle('/home/mowglie/disk/Dropbox/Photos/itmix_rgi_plots/links/itmix_rgi_links.pkl')

    f, d = utils.get_wgms_files()
    wgms_df = pd.read_csv(f)

    f = utils.get_glathida_file()
    gtd_df = pd.read_csv(f)

    rgidf = []
    _rgi_ids = []
    for i, row in df_itmix.iterrows():
        # read the rgi region
        rgi_shp = list(glob.glob(os.path.join(rgi_dir, "*", row['rgi_reg'] + '_rgi50_*.shp')))[0]
        rgi_df = salem.utils.read_shapefile(rgi_shp, cached=True)

        rgi_parts = row.T['rgi_parts_ids']
        sel = rgi_df.loc[rgi_df.RGIId.isin(rgi_parts)].copy()
        _rgi_ids.extend(rgi_parts)

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
            assert len(sel) == 2
        else:
            pass

        # add glacier name to the entity
        name = np.array(['ITMIX:' + row.name] * len(sel))
        add_n = sel.RGIId.isin(wgms_df.RGI_ID.values)
        for z, it in enumerate(add_n.values):
            if it:
                name[z] = 'WGMS-' + name[z]
        add_n = sel.RGIId.isin(gtd_df.RGI_ID.values)
        for z, it in enumerate(add_n.values):
            if it:
                name[z] = 'GTD-' + name[z]
        sel.loc[:, 'Name'] = name
        rgidf.append(sel)

    # WGMS glaciers which are not already there
    # Actually we should remove the data of those 7 to be honest...
    f, d = utils.get_wgms_files()
    wgms_df = pd.read_csv(f)
    print('N WGMS before: {}'.format(len(wgms_df)))
    wgms_df = wgms_df.loc[~ wgms_df.RGI_ID.isin(_rgi_ids)]
    print('N WGMS after: {}'.format(len(wgms_df)))

    for i, row in wgms_df.iterrows():
        rid = row.RGI_ID
        reg = rid.split('-')[1].split('.')[0]
        # read the rgi region
        rgi_shp = list(glob.glob(os.path.join(rgi_dir, "*", reg + '_rgi50_*.shp')))[0]
        rgi_df = salem.utils.read_shapefile(rgi_shp, cached=True)

        sel = rgi_df.loc[rgi_df.RGIId.isin([rid])].copy()
        assert len(sel) == 1

        # add glacier name to the entity
        _corname = row.NAME.replace('/', 'or').replace('.', '').replace(' ', '-')
        sel.loc[:, 'Name'] = ['WGMS:' + _corname] * len(sel)
        rgidf.append(sel)

    _rgi_ids.extend(wgms_df.RGI_ID.values)

    # GTD glaciers which are not already there
    # Actually we should remove the data of those 2 to be honest...
    print('N GTD before: {}'.format(len(gtd_df)))
    gtd_df = gtd_df.loc[~ gtd_df.RGI_ID.isin(_rgi_ids)]
    print('N GTD after: {}'.format(len(gtd_df)))

    for i, row in gtd_df.iterrows():
        rid = row.RGI_ID
        reg = rid.split('-')[1].split('.')[0]
        # read the rgi region
        rgi_shp = list(glob.glob(os.path.join(rgi_dir, "*", reg + '_rgi50_*.shp')))[0]
        rgi_df = salem.utils.read_shapefile(rgi_shp, cached=True)

        sel = rgi_df.loc[rgi_df.RGIId.isin([rid])].copy()
        assert len(sel) == 1

        # add glacier name to the entity
        _corname = row.NAME.replace('/', 'or').replace('.', '').replace(' ', '-')
        sel.loc[:, 'Name'] = ['GTD:' + _corname] * len(sel)
        rgidf.append(sel)

    # Save for not computeing each time
    rgidf = pd.concat(rgidf)
    rgidf.to_pickle(df_rgi_file)


# Remove problem glaciers
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-03.04079', 'RGI50-07.01394'])]

print('Number of glaciers: {}'.format(len(rgidf)))

# Params
cfg.PARAMS['border'] = 20

# Go
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)

from oggm import tasks
from oggm.workflow import execute_entity_task

task_list = [
    # itmix.glacier_masks_itmix,
    # tasks.compute_centerlines,
    tasks.catchment_area,
    tasks.initialize_flowlines,
    tasks.catchment_width_geom
]
for task in task_list:
    execute_entity_task(task, gdirs)

#
# # Climate related tasks
# workflow.climate_tasks(gdirs)

dir = '/home/mowglie/OGGM_ITMIX_PLOTS/'
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