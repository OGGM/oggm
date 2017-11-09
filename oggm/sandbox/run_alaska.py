# This will run OGGM preprocessing on the RGI region of your choice
from __future__ import division
import oggm

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import sys
from glob import glob
import shutil
from functools import partial
# Libs
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely.geometry as shpg
import matplotlib.pyplot as plt
import salem
# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file
from oggm import tasks
from oggm.workflow import execute_entity_task, reset_multiprocessing
from oggm import graphics, utils
# Time
import time
start = time.time()

# Regions:
# Alaska
rgi_reg = '01'

# Initialize OGGM
cfg.initialize()

# Compute a calving flux or not
No_calving = False
With_calving = True

# Set's where is going to run PC or Cluster
Cluster = False
PC = True

# Local paths (where to write output and where are the input files)
if PC:
    MAIN_PATH = '/home/beatriz/Documents/'
    WORKING_DIR = os.path.join(MAIN_PATH,'OGGM_Alaska_run/work_dir/')
    RGI_PATH = os.path.join(MAIN_PATH,
        'global_data_base/RGI_inventory/All_alaska_glacier_goodformat/')
    RGI_FILE = os.path.join(RGI_PATH,'01_rgi50_Alaska.shp')
    DATA_INPUT = os.path.join(MAIN_PATH, 'OGGM_Alaska_run/input_data/')

if Cluster:
    SLURM_WORKDIR = os.environ["WORKDIR"]
    # Local paths (where to write output and where to download input)
    WORKING_DIR = SLURM_WORKDIR
    RGI_PATH = '/home/users/bea/oggm_run_alaska/input_data/Alaska_tidewater_andlake/'
    RGI_FILE = os.path.join(RGI_PATH,'01_rgi50_Alaska.shp')
    DATA_INPUT = '/home/users/bea/oggm_run_alaska/input_data/'


# Configuration of the run
cfg.PARAMS['rgi_version'] = "5"
cfg.PATHS['working_dir'] = WORKING_DIR
# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 80
cfg.PARAMS['optimize_inversion_params'] = True

# Set to True for cluster runs
if Cluster:
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['continue_on_error'] = True
else:
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['continue_on_error'] = False

# We use intersects
# (this is slow, it could be replaced with a subset of the global file)
rgi_dir = utils.get_rgi_intersects_dir()
cfg.set_intersects_db(os.path.join(rgi_dir, '00_rgi50_AllRegs',
                                'intersects_rgi50_AllRegs.shp'))

# Pre-download other files which will be needed later
utils.get_cru_cl_file()
utils.get_cru_file(var='tmp')
utils.get_cru_file(var='pre')

# Some globals for more control on what to run
RUN_GIS_mask = False
RUN_GIS_PREPRO = False # run GIS pre-processing tasks (before climate)
RUN_CLIMATE_PREPRO = False # run climate pre-processing tasks
RUN_INVERSION = False  # run bed inversion

# Read RGI file
rgidf = salem.read_shapefile(RGI_FILE, cached=True)

# get WGMS glaciers
flink, mbdatadir = utils.get_wgms_files()
ids_with_mb = flink['RGI50_ID'].values

if PC:
    # Keep id's of glaciers in WGMS and GlathiDa V2
    keep_ids = ['RGI50-01.02228', 'RGI50-01.00037', 'RGI50-01.16316',
                'RGI50-01.00570', 'RGI50-01.22699']

    # Glaciers in the McNabb data base
    terminus_data_ids = ['RGI50-01.10689', 'RGI50-01.23642']

    keep_indexes = [((i in keep_ids) or (i in ids_with_mb) or
                      (i in terminus_data_ids)) for i in rgidf.RGIID]

    rgidf = rgidf.iloc[keep_indexes]

if Cluster:
    # exclude the columbia glacier and other errors
    errors = ['RGI50-01.10689',
              'RGI50-01.02582', 'RGI50-01.22208', 'RGI50-01.24541',
              'RGI50-01.03465', 'RGI50-01.22209', 'RGI50-01.24542',
              'RGI50-01.05242', 'RGI50-01.22572', 'RGI50-01.24551',
              'RGI50-01.05607', 'RGI50-01.23094', 'RGI50-01.24580',
              'RGI50-01.06005', 'RGI50-01.23128', 'RGI50-01.24600',
              'RGI50-01.07446', 'RGI50-01.24055', 'RGI50-01.24601',
              'RGI50-01.07577', 'RGI50-01.24066', 'RGI50-01.24602',
              'RGI50-01.07630', 'RGI50-01.24231', 'RGI50-01.24603',
              'RGI50-01.09693', 'RGI50-01.24237', 'RGI50-01.24604',
              'RGI50-01.10177', 'RGI50-01.24245', 'RGI50-01.24605',
              'RGI50-01.10453', 'RGI50-01.24246', 'RGI50-01.24607',
              'RGI50-01.10687', 'RGI50-01.24248', 'RGI50-01.26517',
              'RGI50-01.10956', 'RGI50-01.24249', 'RGI50-01.26664',
              'RGI50-01.11897', 'RGI50-01.24250', 'RGI50-01.26665',
              'RGI50-01.12077', 'RGI50-01.24252', 'RGI50-01.26684',
              'RGI50-01.13749', 'RGI50-01.24256', 'RGI50-01.26762',
              'RGI50-01.13755', 'RGI50-01.24257', 'RGI50-01.26776',
              'RGI50-01.14063', 'RGI50-01.24258', 'RGI50-01.26971',
              'RGI50-01.14232', 'RGI50-01.24259', 'RGI50-01.26978',
              'RGI50-01.14344', 'RGI50-01.24260', 'RGI50-01.26991',
              'RGI50-01.14539', 'RGI50-01.24261', 'RGI50-01.26992',
              'RGI50-01.14602', 'RGI50-01.24264', 'RGI50-01.27061',
              'RGI50-01.14631', 'RGI50-01.24287', 'RGI50-01.27106',
              'RGI50-01.14873', 'RGI50-01.24288', 'RGI50-01.27107',
              'RGI50-01.22207', 'RGI50-01.24293', 'RGI50-01.20295',
              'RGI50-01.00625', 'RGI50-01.01521', 'RGI50-01.07718',
              'RGI50-01.07806', 'RGI50-01.11922', 'RGI50-01.20294',
              'RGI50-01.22839', 'RGI50-01.00745', 'RGI50-01.02482',
              'RGI50-01.07803', 'RGI50-01.11323', 'RGI50-01.20246']

    keep_indexes = [(i not in errors) for i in rgidf.RGIID]
    rgidf = rgidf.iloc[keep_indexes]

# Sort for more efficient parallel computing = Bea's rgi has capitals
rgidf = rgidf.sort_values('AREA', ascending=False)

log.info('Starting run for RGI reg: ' + rgi_reg)
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

if RUN_GIS_mask:
    if PC:

        Columbia_path = os.path.join(WORKING_DIR,
                             'per_glacier/RGI50-01/RGI50-01.10/RGI50-01.10689/')
        filename = os.path.join(Columbia_path, 'dem.tif')
        dem_source = os.path.join(Columbia_path, 'dem_source.pkl')
        grid_json = os.path.join(Columbia_path, 'glacier_grid.json')

        # TODO : Remember to this paths on the cluster to run Columbia
        Columbia_itmix = os.path.join(DATA_INPUT,
                                      'RGI50-01.10689_itmixrun_new/')
        dem_cp = os.path.join(Columbia_itmix, 'dem.tif')
        dem_source_cp = os.path.join(Columbia_itmix, 'dem_source.pkl')
        grid_json_cp = os.path.join(Columbia_itmix, 'glacier_grid.json')

        # This is commented because we only need to replace the DEM once
        # os.remove(filename)
        # os.remove(dem_source)
        # os.remove(grid_json)
        # shutil.copy(dem_cp, filename)
        # shutil.copy(dem_source_cp,dem_source)
        # shutil.copy(grid_json_cp,grid_json)

    execute_entity_task(tasks.glacier_masks, gdirs)

# Pre-processing tasks
task_list = [
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]

if RUN_GIS_PREPRO:
    for task in task_list:
        execute_entity_task(task, gdirs)

if RUN_CLIMATE_PREPRO:
    for gdir in gdirs:
        gdir.inversion_calving_rate = 0
    cfg.PARAMS['correct_for_neg_flux'] = False
    cfg.PARAMS['filter_for_neg_flux'] = False
    execute_entity_task(tasks.process_cru_data, gdirs)
    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.apparent_mb, gdirs)

if RUN_INVERSION:
    # Inversion tasks
    execute_entity_task(tasks.prepare_for_inversion, gdirs)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs, )
    # execute_entity_task(tasks.filter_inversion_output, gdirs)

# Compile output if no calving
if No_calving:
    utils.glacier_characteristics(gdirs, filesuffix='_no_calving')

    # Log
    m, s = divmod(time.time() - start, 60)
    h, m = divmod(m, 60)
    log.info("OGGM no_calving is done! Time needed: %02d:%02d:%02d" % (h, m, s))

# Calving loop
# -----------------------------------
if With_calving:
    # Re-initializing climate tasks and inversion without calving to be sure
    for gdir in gdirs:
        gdir.inversion_calving_rate = 0

    cfg.PARAMS['correct_for_neg_flux'] = False
    cfg.PARAMS['filter_for_neg_flux'] = False
    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.apparent_mb, gdirs)
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var = True)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)

    for gdir in gdirs:
        cl = gdir.read_pickle('inversion_output')[-1]
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
        data_link = os.path.join(DATA_INPUT, 'Terminus_width_McNabb.csv')

        dfids = pd.read_csv(data_link)['RGI_ID'].values
        dfwidth = pd.read_csv(data_link)['width'].values
        #dfdepth = pd.read_csv(data_link)['depth'].values
        index = np.where(dfids == gdir.rgi_id)

        cl = gdir.read_pickle('inversion_output')[-1]

        if gdir.rgi_id not in dfids:
            # We read the model output, of the last pixel of the inversion
            width = cl['width'][-1]
            # print('width',width)

        else:
            # we read the McNabb width
            width = dfwidth[index[0][0]]
            # print('width',width)

        t_altitude = cl['hgt'][-1]  # this gives us the altitude at the terminus
        if t_altitude < 0:
            t_altitude = 0

        thick = cl['volume'][-1] / cl['dx'] / width

        # We calculate the water_depth UNCOMMENT THIS to use bathymetry
        # if gdir.rgi_id in dfids:
        #     # we read the depth from the database
        #     w_depth = dfdepth[index[0][0]]
        # else:
        #     w_depth = thick - t_altitude
        w_depth = thick - t_altitude

        #print('t_altitude_fromfun', t_altitude)
        #print('depth_fromfun', w_depth)
        #print('thick_fromfun', thick)
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
            cl = gdir.read_pickle('inversion_output')[-1]

            # We assume a thickness of alt + 1
            t_altitude = cl['hgt'][-1]

            thick = t_altitude + 1
            w_depth = thick - t_altitude

            print('t_altitude', t_altitude)
            print('depth', w_depth)
            print('thick', thick)

            # Read width database:
            data_link = os.path.join(DATA_INPUT, 'Terminus_width_McNabb.csv')
            dfids = pd.read_csv(data_link)['RGI_ID'].values
            dfwidth = pd.read_csv(data_link)['width'].values
            #dfdepth = pd.read_csv(data_link)['depth'].values
            index = np.where(dfids == gdir.rgi_id)

            if gdir.rgi_id not in dfids:
                # We read the model output, of the last pixel of the inversion
                width = cl['width'][-1]
                # print('width',width)

            else:
                # we read the McNabb width
                width = dfwidth[index[0][0]]
                # print('width',width)

            # Lets compute the theoretical calving out of it
            pre_calving = 2 * thick * w_depth * width / 1e9
            gdir.inversion_calving_rate = pre_calving
            print('pre_calving', pre_calving)

            # Recompute all with calving
            tasks.distribute_t_stars([gdir])
            tasks.apparent_mb(gdir)
            tasks.prepare_for_inversion(gdir, add_debug_var=True)
            tasks.volume_inversion(gdir)

            while i < 50:
                # First calculates a calving flux from model output

                F_calving, new_depth, new_thick, t_width = calving_from_depth(
                    gdir)

                # Stores the data, and we see it
                data_calving += [F_calving]
                w_depth += [new_depth]
                thick += [new_thick]
                t_width = t_width
                # print('Calving rate calculated', F_calving)

                # We put the calving function output into the Model
                gdir.inversion_calving_rate = F_calving

                # Recompute mu with calving
                tasks.distribute_t_stars([gdir])
                tasks.apparent_mb(gdir)

                # Inversion with calving, inv optimization is not iterative
                tasks.prepare_for_inversion(gdir, add_debug_var=True)
                tasks.volume_inversion(gdir)
                # tasks.distribute_thickness(gdir, how='per_altitude')

                i += 1
                avg_one = np.average(data_calving[-4:])
                avg_two = np.average(data_calving[-5:-1])
                difference = abs(avg_two - avg_one)
                if difference < 0.05 * avg_two or data_calving[-1] == 0:
                    break

            # Making a dictionary for calving
            cal_dic = dict(calving_fluxes=data_calving, water_depth=w_depth,
                           H_ice=thick, t_width=t_width)
            forwrite.append(cal_dic)
            # We write out everything
            gdir.write_pickle(forwrite, 'calving_output')

        gdir.inversion_calving_rate = 0

    # Reinitialize everything
    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.apparent_mb, gdirs)
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)

    # Assigning to each tidewater glacier its own last_calving flux calculated
    for gdir in gdirs:
        if gdir.is_tidewater:
            calving_output = gdir.read_pickle('calving_output')
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

    # Calculating everything again with a calving flux assigned and filtering
    # and correcting for negative flux
    cfg.PARAMS['correct_for_neg_flux'] = True
    cfg.PARAMS['filter_for_neg_flux'] = True
    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.apparent_mb, gdirs)
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
    execute_entity_task(tasks.volume_inversion, gdirs, filesuffix='_with_calving')

    # Write out glacier statistics
    utils.glacier_characteristics(gdirs, filesuffix='_with_calving')

    m, s = divmod(time.time() - start, 60)
    h, m = divmod(m, 60)
    log.info("OGGM with calving is done! Time needed: %02d:%02d:%02d" % (h, m, s))