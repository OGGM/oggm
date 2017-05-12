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
from functools import partial
# Libs
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils
from oggm.sandbox import itmix
from oggm.core.preprocessing.inversion import distribute_thickness

# ITMIX paths
from oggm.sandbox.itmix.itmix_cfg import DATA_DIR, WORKING_DIR, PLOTS_DIR

# Initialize OGGM
cfg.initialize()

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True
cfg.CONTINUE_ON_ERROR = False

# Working dir
cfg.PATHS['working_dir'] = WORKING_DIR

# Set up the paths and other stuffs
cfg.PATHS['topo_dir'] = os.path.join(DATA_DIR, 'topo')
cfg.PATHS['cru_dir'] = os.path.join(DATA_DIR, 'cru')
cfg.PATHS['rgi_dir'] = os.path.join(DATA_DIR, 'rgi')
cfg.PATHS['itmix_divs'] = os.path.join(DATA_DIR, 'itmix', 'my_divides.pkl')

# Calibration data where ITMIX glaciers have been removed
_path = os.path.join(DATA_DIR, 'itmix', 'wgms',
                     'rgi_wgms_links_2015_RGIV5.csv')
cfg.PATHS['wgms_rgi_links'] = _path
_path = os.path.join(DATA_DIR, 'itmix',
                     'glathida',
                     'rgi_glathida_links_2014_RGIV5.csv')
cfg.PATHS['glathida_rgi_links'] = _path

# Read in the RGI file(s)
rgidf = itmix.get_rgi_df(reset=False)

# Set the newly created divides (quick n dirty fix)
df = gpd.GeoDataFrame.from_file(get_demo_file('divides_workflow.shp'))
df = pd.concat([df, pd.read_pickle(cfg.PATHS['itmix_divs'])])
cfg.PARAMS['divides_gdf'] = df.set_index('RGIID')

# Run parameters
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
do_calib = False
do_itmix = True

# Remove problem glaciers
# This is an ice-cap, centerlines wont work
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-07.01394'])]
# This is an ice-cap (Amitsuloq-Iska), it makes bullshit
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-05.00446'])]
# This is a huge part of an ice cap thing, not sure if we should use it
# I have the feeling that it blocks the catchment area function...
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-04.06187'])]
# Remove all glaciers in Antarctica
rgidf = rgidf.loc[~ rgidf['O1Region'].isin(['19'])]
# This really is an OGGM problem with strange lines
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-02.13974'])]
# Lewis glaciers (MB doesnt work)
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-16.01638'])]
# This has a very bad topography:
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-03.04079'])]
# Remove bad black rapids:
rgidf = rgidf.loc[~ rgidf.RGIId.isin(['RGI50-01.00037'])]

# This was for experiments
# Keep WGMS only
# rgidf = rgidf.loc[[('G:' in r) or ('G-' in r) for r in rgidf.Name]]
# Keep Itmix only
# rgidf = rgidf.loc[['I:' in r for r in rgidf.Name]]
# Keep only columbia
# rgidf = rgidf.loc[['Columbia' in n for n in rgidf.Name]]
# rgidf = rgidf.loc[~ np.asarray(['Wa' in n for n in rgidf.Name])]
# Keep Siachen
# rgidf = rgidf.loc[rgidf.RGIId.isin(['RGI50-14.07524'])]

log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)
# For inversion
icecaps = ['I:Devon', 'I:Academy', 'I:Austfonna', 'I:Elbrus', 'I:Mocho']
for gd in gdirs:
    if gd.name in icecaps:
        gd.glacier_type = 'Ice cap'

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

    # Climate related tasks
    execute_entity_task(tasks.process_cru_data, gdirs)
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)

    # Inversion
    execute_entity_task(tasks.prepare_for_inversion, gdirs)
    itmix.optimize_thick(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)

    # Write out glacier statistics
    df = utils.glacier_characteristics(gdirs)
    fpath = os.path.join(cfg.PATHS['working_dir'], 'glacier_char.csv')
    df.to_csv(fpath)

if do_itmix:

    done = False
    for gd in gdirs:
        if 'Urumqi' in gd.name:
            if not done:
                gd.name += '_A'
                done = True
            else:
                gd.name += '_B'

    # For final inversions - no calving
    gdirs = [gd for gd in gdirs if 'I:' in gd.name]
    col_gdir = [gd for gd in gdirs if 'Columbia' in gd.name]

    cfg.PARAMS['calving_rate'] = 0
    addt = ' no calving'
    tasks.distribute_t_stars(col_gdir)
    execute_entity_task(tasks.prepare_for_inversion, col_gdir)
    execute_entity_task(tasks.volume_inversion, gdirs)
    pdir = os.path.join(PLOTS_DIR, 'out_raw') + '/'
    if not os.path.exists(pdir):
        os.mkdir(pdir)
    for gd in gdirs:
        _addt = addt if 'Columbia' in gd.name else ''
        graphics.plot_inversion(gd, add_title_comment=_addt)
        plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_inv.png')
        plt.close()

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

    # Columbia
    del gdirs
    ver = 3
    calvings = [4, 8, 12]
    for k, ca in enumerate(calvings):
        cfg.PARAMS['calving_rate'] = ca / 0.9
        cat = '{}'.format(ca)
        addt = ' calving {}'.format(ca)
        tasks.distribute_t_stars(col_gdir)
        execute_entity_task(tasks.prepare_for_inversion, col_gdir)
        execute_entity_task(tasks.volume_inversion, col_gdir)
        pdir = os.path.join(PLOTS_DIR, 'out_raw') + '/'
        for gd in col_gdir:
            _addt = addt if 'Columbia' in gd.name else ''
            graphics.plot_inversion(gd, add_title_comment=_addt)
            plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_inv_'+cat+'.png')
            plt.close()

        # V1
        distrib = partial(distribute_thickness, how='per_altitude',
                          add_slope=True,
                          smooth=True)
        execute_entity_task(distrib, col_gdir)
        pdir = os.path.join(PLOTS_DIR, 'out_dis') + '/'
        for gd in col_gdir:
            itmix.write_itmix_ascii(gd, k*2 + ver)
            graphics.plot_distributed_thickness(gd)
            plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_d1_'+cat+'.png')
            plt.close()

        # V2
        distrib = partial(distribute_thickness, how='per_altitude',
                          add_slope=False,
                          smooth=True)
        execute_entity_task(distrib, col_gdir)
        for gd in col_gdir:
            itmix.write_itmix_ascii(gd, k*2 + 1 + ver)
            graphics.plot_distributed_thickness(gd)
            plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_d2_'+cat+'.png')
            plt.close()


# pdir = PLOTS_DIR
# if not os.path.exists(pdir):
#     os.makedirs(pdir)
# for gd in gdirs:
    # graphics.plot_googlemap(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_ggl.png')
    # plt.close()
    # graphics.plot_domain(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_dom.png')
    # plt.close()
    # graphics.plot_centerlines(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_cls.png')
    # plt.close()
    # graphics.plot_catchment_width(gd, corrected=True)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_w.png')
    # plt.close()
    # graphics.plot_inversion(gd)
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_inv.png')
    # plt.close()
    # graphics.plot_distributed_thickness(gd, how='per_altitude')
    # plt.savefig(pdir + gd.name + '_' + gd.rgi_id + '_dis1.png')
    # plt.close()
    pass
