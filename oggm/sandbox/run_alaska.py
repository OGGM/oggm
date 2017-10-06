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

# What run we will do: without calving or with calving module
With_calving = True

Cluster = False
PC = True

# Local paths (where to write output and where to download input)
if PC:
    MAIN_PATH = '/home/beatriz/Documents/'
    WORKING_DIR = os.path.join(MAIN_PATH,'OGGM_Alaska_run_new/work_dir/')
    RGI_PATH = os.path.join(MAIN_PATH,
        'global_data_base/RGI_inventory/All_alaska_glacier_goodformat/')
    RGI_FILE = os.path.join(RGI_PATH,'01_rgi50_Alaska.shp')
    DATA_INPUT = os.path.join(MAIN_PATH, 'OGGM_Alaska_run/input_data/')

    print(WORKING_DIR)
    print(RGI_FILE)

if Cluster:
    SLURM_WORKDIR = os.environ["WORKDIR"]
    # Local paths (where to write output and where to download input)
    WORKING_DIR = SLURM_WORKDIR
    RGI_PATH = '~/oggm_run_alaska/input_data/Alaska_tidewater_andlake/'
    RGI_FILE = os.path.join(RGI_PATH,'01_rgi50_Alaska.shp')
    DATA_INPUT = '~oggm_run_alaska/input_data/'

    print(WORKING_DIR)
    print(RGI_FILE)

utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 80
cfg.PARAMS['optimize_inversion_params'] = True

# Set to True for cluster runs
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['continue_on_error'] = False

# Don't use divides for now
cfg.set_divides_db()

# But we use intersects
rgi_dir = utils.get_rgi_intersects_dir()
rgi_shp = list(glob(os.path.join(rgi_dir, "*", '*intersects*' + rgi_reg + '_rgi50_*.shp')))
assert len(rgi_shp) == 1
cfg.set_intersects_db(rgi_shp[0])
print('Intersects file: ' + rgi_shp[0])

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
p = utils.get_cru_file(var='pre')
print('CRU file: ' + p)

# Copy the precalibrated tstar file
# ---------------------------------
# Note that to be exact, this procedure can only be applied if the run
# parameters don't change between the calibration and the run.
# After testing, it appears that changing the 'border' parameter won't affect
# the results much (expectedly), so that it's ok to change it. All the rest
# (e.g. smoothing, dx, prcp factor...) should imply a re-calibration
mbf = 'https://www.dropbox.com/s/23a61yxwgpprs9q/ref_tstars_no_tidewater.csv?dl=1'
mbf = utils.file_downloader(mbf)
shutil.copyfile(mbf, os.path.join(WORKING_DIR, 'ref_tstars.csv'))

# Some globals for more control on what to run
RUN_GIS_mask = False
RUN_GIS_PREPRO = False # run GIS pre-processing tasks (before climate)
RUN_CLIMATE_PREPRO = False # run climate pre-processing tasks
RUN_INVERSION = False  # run bed inversion

# Read RGI files
rgidf = salem.read_shapefile(RGI_FILE, cached=True)

# get WGMS glaciers
flink, mbdatadir = utils.get_wgms_files()

ids_with_mb = pd.read_csv(flink)['RGI50_ID'].values

if PC:
    keep_ids = ['RGI50-01.00037', 'RGI50-01.00570','RGI50-01.01104',
                'RGI50-01.01390', 'RGI50-01.02228', 'RGI50-01.04591',
                'RGI50-01.05211', 'RGI50-01.09162', 'RGI50-01.16316',
                'RGI50-01.22699']

    # TW glaciers that I want to run
    terminus_data_ids = ['RGI50-01.14443','RGI50-01.10689']

    keep_indexes = [((i in keep_ids) or (i in ids_with_mb) or
                 (i in terminus_data_ids)) for i in rgidf.RGIID]

    rgidf = rgidf.iloc[keep_indexes]

if Cluster:
    # exclude the columbia glacier and other errors
    errors = ['RGI50-01.10689', 'RGI50-01.00625', 'RGI50-01.01521',
              'RGI50-01.07718', 'RGI50-01.07806', 'RGI50-01.11922',
              'RGI50-01.20294', 'RGI50-01.22839', 'RGI50-01.00745',
              'RGI50-01.02482', 'RGI50-01.07803', 'RGI50-01.11323',
              'RGI50-01.20246', 'RGI50-01.20295']
    keep_indexes = [(i not in errors) for i in rgidf.RGIID]
    rgidf = rgidf.iloc[keep_indexes]

# Sort for more efficient parallel computing = my rgi has capitals
rgidf = rgidf.sort_values('AREA', ascending=False)

log.info('Starting run for RGI reg: ' + rgi_reg)
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

if RUN_GIS_mask:

    Columbia_path = os.path.join(WORKING_DIR,
                            'per_glacier/RGI50-01/RGI50-01.10/RGI50-01.10689/')
    filename = os.path.join(Columbia_path,'dem.tif')
    dem_source = os.path.join(Columbia_path,'dem_source.pkl')
    grid_json = os.path.join(Columbia_path,'glacier_grid.json')

    # TODO : Remember to this paths on the cluster to run Columbia
    Columbia_itmix = os.path.join(DATA_INPUT,
                                  'RGI50-01.10689_itmixrun_new/')
    dem_cp = os.path.join(Columbia_itmix,'dem.tif')
    dem_source_cp = os.path.join(Columbia_itmix,'dem_source.pkl')
    grid_json_cp = os.path.join(Columbia_itmix,'glacier_grid.json')

    os.remove(filename)
    os.remove(dem_source)
    os.remove(grid_json)
    shutil.copy(dem_cp, filename)
    shutil.copy(dem_source_cp,dem_source)
    shutil.copy(grid_json_cp,grid_json)

    #Calculate glacier masks
    execute_entity_task(tasks.glacier_masks, gdirs)

# The rest Prepro tasks
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
    # Climate tasks -- only data preparation and tstar interpolation!
    execute_entity_task(tasks.process_cru_data, gdirs)
    tasks.distribute_t_stars(gdirs)

if RUN_INVERSION:
    # Inversion tasks
    execute_entity_task(tasks.prepare_for_inversion, gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs,
                        use_cfg_params={'glen_a': cfg.A, 'fs': 0})
    #execute_entity_task(tasks.filter_inversion_output, gdirs)

# Compile output
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

    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)
    # execute_entity_task(tasks.filter_inversion_output, gdirs)

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
        data_link = os.path.join(DATA_INPUT, 'Terminus_width_McNabb.csv')

        dfids = pd.read_csv(data_link)['RGI_ID'].values
        dfwidth = pd.read_csv(data_link)['width'].values
        #dfdepth = pd.read_csv(data_link)['depth'].values
        index = np.where(dfids == gdir.rgi_id)

        cl = gdir.read_pickle('inversion_output', div_id=1)[-1]

        if gdir.rgi_id not in dfids:
            # We read the model output, of the last pixel of the inversion
            width = cl['width'][-1]
            # print('width',width)

        else:
            # we read the McNabb width
            width = dfwidth[index[0][0]]
            # print('width',width)

        t_altitude = cl['hgt'][
            -1]  # this gives us the altitude at the terminus
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

    # Write out glacier statistics
    utils.glacier_characteristics(gdirs, filesuffix='_with_calving')

    m, s = divmod(time.time() - start, 60)
    h, m = divmod(m, 60)
    log.info("OGGM with calving is done! Time needed: %02d:%02d:%02d" % (h, m, s))
