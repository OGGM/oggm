"""Run with tidewater glaciers"""
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
import netCDF4
import matplotlib.pyplot as plt
# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils
from oggm.utils import get_demo_file
from oggm.utils import tuple2int

# This will run OGGM in different Alaska Regions

# Initialize OGGM
cfg.initialize()

# Set's where is going to run PC or AWS
RUN_inPC = True
RUN_inAWS = False

# What run we will do: without calving or with calving module
No_calving = False
With_calving = True

if RUN_inPC:
    # Local paths (where to write output and where to download input)
    WORKING_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/work_dir/'
    DATA_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/input_data/'
    RGI_FILE = '/home/beatriz/Documents/global_data_base/RGI_inventory/All_alaska_glacier_goodformat/01_rgi50_Alaska.shp'
    GLATHIDA_FILE = '/home/beatriz/Documents/OGGM_Alaska_run/input_data/rgi_glathida_links_2014_RGIV54.csv'


if RUN_inAWS:
    # TODO: this paths are going to change with the new cluster
    WORKING_DIR = '~/work_dir/'
    DATA_DIR = '~/input_data/'
    RGI_FILE = '~/Sub_region4/Sub_region4.shp'
    GLATHIDA_FILE = '~/input_data/rgi_glathida_links_2014_RGIV54.csv'

cfg.PATHS['working_dir'] = WORKING_DIR
cfg.PATHS['topo_dir'] = os.path.join(DATA_DIR, 'topo')
cfg.PATHS['cru_dir'] = os.path.join(DATA_DIR, 'cru')
cfg.PATHS['glathida_rgi_links'] = GLATHIDA_FILE

# Create directories
utils.mkdir(cfg.PATHS['working_dir'])
utils.mkdir(cfg.PATHS['topo_dir'])
utils.mkdir(cfg.PATHS['cru_dir'])

# Use multiprocessing? Change this to run in the AWS
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['continue_on_error'] = False

# Other params
cfg.PARAMS['border'] = 80
cfg.PARAMS['temp_use_local_gradient'] = False
cfg.PARAMS['optimize_inversion_params'] = True
cfg.PARAMS['invert_with_sliding'] = False
cfg.PARAMS['bed_shape'] = 'parabolic'
cfg.PARAMS['use_compression'] = False

# Some globals for more control on what to run
RUN_GIS_mask = False
RUN_GIS_PREPRO = False # run GIS pre-processing tasks (before climate)
RUN_CLIMATE_PREPRO = False # run climate pre-processing tasks
RUN_INVERSION = False  # run bed inversion
RUN_DYNAMICS = False  # run dynamics

Plotting = False
Plotting_after_calving = False

# Read RGI file
rgidf = salem.read_shapefile(RGI_FILE, cached=True)

# TODO: we must sort.() glaciers from large to small when we run in cluster
# Get ref glaciers (all glaciers with MB)
flink, mbdatadir = utils.get_wgms_files()

ids_with_mb = pd.read_csv(flink)['RGI50_ID'].values

if RUN_inPC:
    # Keep id's of glaciers in WGMS and GlathiDa V2
    keep_ids = ['RGI50-01.00037', 'RGI50-01.00570','RGI50-01.01104',
                'RGI50-01.01390', 'RGI50-01.02228', 'RGI50-01.04591',
                'RGI50-01.05211', 'RGI50-01.09162', 'RGI50-01.16316',
                'RGI50-01.22699']

    # Glaciers in the McNabb data base
    terminus_data_ids = ['RGI50-01.03377', 'RGI50-01.03622', 'RGI50-01.03890',
       'RGI50-01.04375', 'RGI50-01.09426', 'RGI50-01.09519',
       'RGI50-01.09757', 'RGI50-01.09783', 'RGI50-01.23565',
       'RGI50-01.09810', 'RGI50-01.10188', 'RGI50-01.10299',
       'RGI50-01.10325', 'RGI50-01.10355', 'RGI50-01.10402',
       'RGI50-01.10431', 'RGI50-01.10575', 'RGI50-01.10607',
       'RGI50-01.10612', 'RGI50-01.10689', 'RGI50-01.10836',
       'RGI50-01.13638', 'RGI50-01.14391', 'RGI50-01.14443',
       'RGI50-01.14683', 'RGI50-01.14876', 'RGI50-01.14878',
       'RGI50-01.17807', 'RGI50-01.17840', 'RGI50-01.17843',
       'RGI50-01.17848', 'RGI50-01.17876', 'RGI50-01.20470',
       'RGI50-01.20554', 'RGI50-01.20734', 'RGI50-01.20783',
       'RGI50-01.20791', 'RGI50-01.20830', 'RGI50-01.20841',
       'RGI50-01.20891', 'RGI50-01.20926', 'RGI50-01.23642',
       'RGI50-01.21001', 'RGI50-01.21005', 'RGI50-01.22776',
       'RGI50-01.26732', 'RGI50-01.09639']

    # problematic calving fluxes
    # terminus_data_ids = ['RGI50-01.23642', 'RGI50-01.10575', 'RGI50-01.10612',
    #                     'RGI50-01.14443', 'RGI50-01.10689']
    #, 'RGI50-01.17876', 'RGI50-01.10689']


    keep_indexes = [((i in keep_ids) or (i in ids_with_mb) or
                      (i in terminus_data_ids)) for i in rgidf.RGIID]

    # original set up
    # keep_indexes = [((i in keep_ids) or (i in ids_with_mb)) for i in rgidf.RGIID]
    # only glaciers for calibration
    # keep_indexes = [(i in keep_ids) for i in rgidf.RGIID]
    # Only errors
    # errors = ['RGI50-01.20968', 'RGI50-01.21055']

    # Other statements to exclude some glaciers
    # terminus_data_ids = ['RGI50-01.14443']
    # keep_indexes = [(i in terminus_data_ids) for i in rgidf.RGIID]
    # (i not in ids_with_mb)) for i in rgidf.RGIID]
    # keep_indexes = [(i not in keep_ids) for i in rgidf.RGIID]
    # keep_indexes = [((i in errors) or (i in keep_ids)) for i in rgidf.RGIID]

    rgidf = rgidf.iloc[keep_indexes]

if RUN_inAWS:
    # Uses all the glaciers in the shape-file, MB and GlathiDa glaciers must be
    # in side the shapefile.
    rgidf = rgidf

rgidf = rgidf.drop_duplicates('RGIID')
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Download other files if needed
_ = utils.get_cru_file(var='tmp')

# Go - initialize working directories
#gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True) #CHECK!!
gdirs = workflow.init_glacier_regions(rgidf)

# Pre-pro tasks
if RUN_GIS_mask:
    # Replacing the DEM file for Columbia with ITMIX DEM this path's can
    # change in every update!
    if RUN_inPC:
        Columbia_path = os.path.join(WORKING_DIR,
                            'per_glacier/RGI50-01/RGI50-01.10/RGI50-01.10689/')
        filename = os.path.join(Columbia_path,'dem.tif')
        Columbia_itmix = os.path.join(DATA_DIR,
                                      'RGI50-01.10689_itmixrun_new/dem.tif')

    if RUN_inAWS:
        #TODO this path will change in new cluster
        Columbia_path = os.path.join(WORKING_DIR,
            'per_glacier/RGI50-01/RGI50-01.10/RGI50-01.10689/')
        filename = os.path.join(Columbia_path,'dem.tif')
        Columbia_itmix = os.path.join(DATA_DIR,
            'RGI50-01.10689_itmixrun_new/dem.tif')

    os.remove(filename)
    shutil.copy(Columbia_itmix,Columbia_path)

    #Calculate glacier masks
    execute_entity_task(tasks.glacier_masks, gdirs)

# Then the rest
task_list = [
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

for gdir in gdirs:
        gdir.inversion_calving_rate = 0

if RUN_CLIMATE_PREPRO:
    # Climate related tasks
    workflow.execute_entity_task(tasks.process_cru_data, gdirs)
    tasks.compute_ref_t_stars(gdirs)
    for gdir in gdirs:
        gdir.inversion_calving_rate = 0
    tasks.distribute_t_stars(gdirs)

if RUN_INVERSION:
    # Inversion
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)
    # execute_entity_task(tasks.filter_inversion_output, gdirs)
    workflow.execute_entity_task(tasks.distribute_thickness, gdirs, how='per_altitude')

if No_calving:
    # Write out glacier statistics
    df = utils.glacier_characteristics(gdirs)
    #fpath = os.path.join(cfg.PATHS['working_dir'], 'glacier_char.csv')
    #df.to_csv(fpath)

###################### Plotting scripts #######################################
if Plotting:
    # Plotting things before the calving for all glaciers
    PLOTS_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/plots/'
    # PLOTS_DIR = ''
    # if PLOTS_DIR == '':
    #    exit()
    # utils.mkdir(PLOTS_DIR)

    for gd in gdirs:
        bname = os.path.join(PLOTS_DIR, gd.name + '_' + gd.rgi_id + '_')
        graphics.plot_googlemap(gd)
        plt.savefig(bname + 'ggl.png')
        plt.close()
        graphics.plot_centerlines(gd, add_downstream=False)
        plt.savefig(bname + 'cls.png')
        plt.close()
        graphics.plot_catchment_width(gd, corrected=True)
        plt.savefig(bname + 'w.png')
        plt.close()
        graphics.plot_inversion(gd)
        plt.savefig(bname + 'inv.png')
        plt.close()
        graphics.plot_distributed_thickness(gd)
        plt.savefig(bname + 'inv_cor.png')
        plt.close()

################### Making a regional thickness plot #########################
# dem = utils.get_topo_file([-149.5,-146],[60, 62])
# dem = salem.GeoTiff(dem[0])
#
# #dem.set_subset(corners=((-149,60),(-146,62)),margin=-147)
# #g = dem.grid.center_grid
# #print(g)
# #g = salem.mercator_grid(center_ll=(-147.75, 61.2), extent=(27000, 21000))
# #
# g = salem.mercator_grid(center_ll=(-148, 61), extent=(250000, 210000))
# # And a map accordingly
# sm = salem.Map(g, countries=False, nx=dem.grid.nx)
# sm.set_topography(dem.get_vardata(), crs=dem.grid)

#And a map accordingly
#sm = salem.Map(g, countries=False, nx=dem.grid.nx)
#sm.set_topography(dem.get_vardata())
#
# # Give this to the plot function
# fig, ax = plt.subplots(figsize=(20, 15))
#
# graphics.plot_region_inversion(gdirs, salemmap=sm, ax=ax)
#plt.savefig('/home/beatriz/Documents/EGU_alaska_run/plots/region_nocalving.png')
#plt.savefig('/home/beatriz/Documents/EGU_alaska_run/plots/region_thick_diff.png')
#plt.close()

###### Never use ###########################################################
# if RUN_DYNAMICS:
#     # Random dynamics
#     execute_entity_task(tasks.init_present_time_glacier, gdirs)
#     execute_entity_task(tasks.random_glacier_evolution, gdirs)
###### Never use ###########################################################

if With_calving:

    # Re-initializing climate tasks and inversion without calving to be sure
    for gdir in gdirs:
        gdir.inversion_calving_rate = 0

    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)
    #execute_entity_task(tasks.filter_inversion_output, gdirs)

    for gdir in gdirs:
        cl = gdir.read_pickle('inversion_output', div_id=1)[-1]
        if gdir.is_tidewater:
            assert cl['volume'][-1] == 0.

    # Defining a calving function
    def calving_from_depth(gdir):
        """ Finds a calving flux based on the approaches proposed by
            Huss and Hock, (2015) and Oerlemans and Nick (2005).

        We take the initial output of the model and surface elevation data
        to calculate the water depth of the calving front.
        """
        # Read width database:
        data_link = os.path.join(DATA_DIR,'Terminus_width_McNabb.csv')
        #data_link = os.path.join(DATA_DIR,
        #                         'Terminus_width_McNabb_vs_OGGMout.csv')
        dfids = pd.read_csv(data_link)['RGI_ID'].values
        dfwidth = pd.read_csv(data_link)['width'].values
        index = np.where(dfids == gdir.rgi_id)

        cl = gdir.read_pickle('inversion_output', div_id=1)[-1]

        if gdir.rgi_id not in dfids:
            # We read the model output, of the last pixel of the inversion
            width = cl['width'][-1]
            #print('width',width)

        else:
            # we read the McNabb width
            width = dfwidth[index[0][0]]
            #print('width',width)

        t_altitude = cl['hgt'][-1] # this gives us the altitude at the terminus
        if t_altitude < 0:
            t_altitude = 0

        thick = cl['volume'][-1] / cl['dx'] / width

        # We calculate the water_depth
        w_depth = thick - t_altitude
        print('t_altitude_fromfun', t_altitude)
        print('depth_fromfun', w_depth)
        print('thick_fromfun', thick)
        out = 2 * thick * w_depth * width / 1e9
        if out < 0:
            out = 0
        return out, w_depth, thick, width

    # Selecting the tidewater glaciers on the region
    for gdir in gdirs:
        if gdir.is_tidewater:
            # Starting the calving loop, with a maximum of 50 cycles
            i = 0
            data_calving = []
            w_depth = []
            thick = []
            t_width = []
            forwrite = []

            # Our first guess of calving is that of a water depth of 1m

            # We read the model output, of the last pixel of the inversion
            cl = gdir.read_pickle('inversion_output', div_id=1)[-1]

            # We assume a thickness of alt + 1
            t_altitude = cl['hgt'][-1]

            thick = t_altitude + 1
            w_depth = thick - t_altitude

            print('t_altitude',t_altitude)
            print('depth',w_depth)
            print('thick',thick)

            # Read width database:
            data_link = os.path.join(DATA_DIR,'Terminus_width_McNabb.csv')
            #data_link = os.path.join(DATA_DIR,
            #                         'Terminus_width_McNabb_vs_OGGMout.csv')
            dfids = pd.read_csv(data_link)['RGI_ID'].values
            dfwidth = pd.read_csv(data_link)['width'].values
            index = np.where(dfids == gdir.rgi_id)

            if gdir.rgi_id not in dfids:
                # We read the model output, of the last pixel of the inversion
                width = cl['width'][-1]
                #print('width',width)

            else:
                # we read the McNabb width
                width = dfwidth[index[0][0]]
                #print('width',width)

            # Lets compute the theoretical calving out of it
            pre_calving = 2 * thick * w_depth * width / 1e9
            gdir.inversion_calving_rate = pre_calving
            print('pre_calving', pre_calving)

            # Recompute all with calving
            tasks.distribute_t_stars([gdir])
            tasks.prepare_for_inversion(gdir, add_debug_var=True)
            tasks.volume_inversion(gdir)

            while i < 50:
                # First calculates a calving flux from model output

                F_calving, new_depth, new_thick, t_width = calving_from_depth(gdir)

                # Stores the data, and we see it
                data_calving += [F_calving]
                w_depth += [new_depth]
                thick += [new_thick]
                t_width = t_width
                #print('Calving rate calculated', F_calving)

                # We put the calving function output into the Model
                gdir.inversion_calving_rate = F_calving

                # Recompute mu with calving
                tasks.distribute_t_stars([gdir])

                # Inversion with calving, inv optimization is not iterative
                tasks.prepare_for_inversion(gdir, add_debug_var=True)
                tasks.volume_inversion(gdir)
                #tasks.distribute_thickness(gdir, how='per_altitude')

                i += 1
                avg_one = np.average(data_calving[-4:])
                avg_two = np.average(data_calving[-5:-1])
                difference = abs(avg_two - avg_one)
                if difference < 0.05*avg_two or data_calving[-1] == 0:
                    break

            # Making a dictionary for calving
            cal_dic = dict(calving_fluxes=data_calving, water_depth=w_depth,
                           H_ice=thick, t_width=t_width)
            forwrite.append(cal_dic)
            # We write out everything
            gdir.write_pickle(forwrite, 'calving_output', div_id=1)

        gdir.inversion_calving_rate = 0

    # Reinitialize everything
    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)
    # execute_entity_task(tasks.filter_inversion_output, gdirs)

    # Assigning to each tidewater glacier its own last_calving flux calculated
    for gdir in gdirs:
        if gdir.is_tidewater:
            calving_output = gdir.read_pickle('calving_output', div_id=1)
            for objt in calving_output:
                all_calving_data = objt['calving_fluxes']
                all_data_depth = objt['water_depth']
                all_data_H_i = objt['H_ice']
                all_data_width = objt['t_width']
            # we see the final calculated calving flux
            last_calving = all_calving_data[-1]
            last_width = all_data_width

            print('For the glacier', gdir.rgi_id)
            print('last calving value is:', last_calving)

            gdir.inversion_calving_rate = last_calving

    # Calculating everything again with a calving flux assigned
    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
    execute_entity_task(tasks.volume_inversion, gdirs)
    # execute_entity_task(tasks.filter_inversion_output, gdirs)
    #workflow.execute_entity_task(tasks.distribute_thickness, gdirs, how='per_altitude')

    # Write out glacier statistics
    df = utils.glacier_characteristics(gdirs)
    #fpath = os.path.join(cfg.PATHS['working_dir'], 'glacier_char.csv')
    #df.to_csv(fpath)

if Plotting_after_calving:
    # Plotting things after the calving for all glaciers
    PLOTS_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/plots/'
    # PLOTS_DIR = ''
    # if PLOTS_DIR == '':
    #     exit()
    # utils.mkdir(PLOTS_DIR)

    for gd in gdirs:
        bname = os.path.join(PLOTS_DIR, gd.name + '_' + gd.rgi_id + '_withcalving')
        graphics.plot_inversion(gd)
        plt.savefig(bname + 'inv.png')
        plt.close()
        graphics.plot_distributed_thickness(gd)
        plt.savefig(bname + 'inv_cor.png')
        plt.close()