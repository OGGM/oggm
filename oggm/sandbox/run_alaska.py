"""Run with a subset of tidewater glaciers"""
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

# This will run OGGM in Alaska

# Initialize OGGM
cfg.initialize()

# Local paths (where to write output and where to download input)
WORKING_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/work_dir/'
DATA_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/input_data/'
RGI_FILE = '/home/beatriz/Documents/global_data_base/Alaska_tidewater_andlake/01_rgi50_Alaska.shp'
GLATHIDA_FILE = '/home/beatriz/Documents/OGGM_Alaska_run/input_data/rgi_glathida_links_2014_RGIV54.csv'

cfg.PATHS['working_dir'] = WORKING_DIR
cfg.PATHS['topo_dir'] = os.path.join(DATA_DIR, 'topo')
cfg.PATHS['cru_dir'] = os.path.join(DATA_DIR, 'cru')
cfg.PATHS['glathida_rgi_links'] = GLATHIDA_FILE

# Currently OGGM wants some directories to exist
# (maybe I'll change this but it can also catch errors in the user config)
utils.mkdir(cfg.PATHS['working_dir'])
utils.mkdir(cfg.PATHS['topo_dir'])
utils.mkdir(cfg.PATHS['cru_dir'])

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = False
cfg.CONTINUE_ON_ERROR = False

# Other params
cfg.PARAMS['border'] = 80
cfg.PARAMS['temp_use_local_gradient'] = False
cfg.PARAMS['optimize_inversion_params'] = True
cfg.PARAMS['invert_with_sliding'] = False
cfg.PARAMS['bed_shape'] = 'parabolic'

# Some globals for more control on what to run
RUN_GIS_PREPRO = False  # run GIS pre-processing tasks (before climate)
RUN_CLIMATE_PREPRO = True  # run climate pre-processing tasks
RUN_INVERSION = True  # run bed inversion
RUN_DYNAMICS = False  # run dynamics

# Read RGI file
rgidf = salem.utils.read_shapefile(RGI_FILE, cached=True)

# Select some glaciers
# Get ref glaciers (all glaciers with MB)
flink, mbdatadir = utils.get_wgms_files()

ids_with_mb = pd.read_csv(flink)['RGI_ID'].values
# get some tw-glaciers that we want to test inside alaska region, also that are
# inside GlathiDa
keep_ids = ['RGI50-01.10689', 'RGI50-01.20791', 'RGI50-01.00037', 'RGI50-01.10402', 'RGI50-01.22193', 'RGI50-01.22699']
keep_indexes = [((i in keep_ids) or (i in ids_with_mb)) for i in rgidf.RGIID]
rgidf = rgidf.iloc[keep_indexes]

log.info('Number of glaciers: {}'.format(len(rgidf)))

# Download other files if needed
_ = utils.get_cru_file(var='tmp')

# Go - initialize working directories
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)

# Pre-pro tasks
task_list = [
    tasks.glacier_masks,
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

if RUN_CLIMATE_PREPRO:
    # Climate related tasks
    workflow.execute_entity_task(tasks.distribute_cru_style, gdirs)
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)

if RUN_INVERSION:
    # Inversion
    execute_entity_task(tasks.prepare_for_inversion, gdirs)
    tasks.optimize_inversion_params(gdirs)
    execute_entity_task(tasks.volume_inversion, gdirs)
    workflow.execute_entity_task(tasks.distribute_thickness, gdirs, how='per_altitude')

#Plotting things before the calving for all glaciers
# PLOTS_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/plots/'
# #PLOTS_DIR = ''
# if PLOTS_DIR == '':
#    exit()
# utils.mkdir(PLOTS_DIR)
# #
# for gd in gdirs:
#     bname = os.path.join(PLOTS_DIR, gd.name + '_' + gd.rgi_id + '_')
#     graphics.plot_googlemap(gd)
#     plt.savefig(bname + 'ggl.png')
#     plt.close()
#     graphics.plot_centerlines(gd, add_downstream=True)
#     plt.savefig(bname + 'cls.png')
#     plt.close()
#     graphics.plot_catchment_width(gd, corrected=True)
#     plt.savefig(bname + 'w.png')
#     plt.close()
#     graphics.plot_inversion(gd)
#     plt.savefig(bname + 'inv.png')
#     plt.close()
#     graphics.plot_distributed_thickness(gd)
#     plt.savefig(bname + 'inv_cor.png')
#     plt.close()

# if RUN_DYNAMICS:
#     # Random dynamics
#     execute_entity_task(tasks.init_present_time_glacier, gdirs)
#     execute_entity_task(tasks.random_glacier_evolution, gdirs)

# Re-initializing climate tasks and inversion without calving just to be sure
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.prepare_for_inversion, gdirs)
tasks.optimize_inversion_params(gdirs)
execute_entity_task(tasks.volume_inversion, gdirs)

# comments ....
# CAREFUL: the invert_parabolic_bed() task is making tests on the shape of the
# bed, and if the bed shape is not realistic, it will change it to more
# realistic values. This might be undesirable for tidewater glaciers, and
# it is possible to add special conditions for tidewater glaciers in the
# function. For example: if gdir.is_tidewater: etc.
# (To fix this for now everything has been put after an if in the inversion.py)

# Selecting the glacier on which we will work
gdir = gdirs[3]

#Plot the glacier before calving for one glacier only
# PLOTS_DIR = '/home/beatriz/Documents/OGGM_Alaska_run/plots/'
#
# if PLOTS_DIR == '':
#    exit()
# utils.mkdir(PLOTS_DIR)
# bname = os.path.join(PLOTS_DIR, gdir.name + '_' + gdir.rgi_id + '_')
#
# graphics.plot_inversion(gdir)
# plt.savefig(bname + 'inv.png')
# plt.close()
# graphics.plot_distributed_thickness(gdir)
# plt.savefig(bname + 'inv_cor.png')
# plt.close()


# Test that frontal H is close to zero
cl = gdir.read_pickle('inversion_output', div_id=1)[-1]
width = cl['width'][-1]
# we take this thickness because cl['thick'] assumes a parabolic bed
# cl['volume'] its a more "correct" measurement of the geometry of the glacier
thick = cl['volume'][-1] / cl['dx'] / width
print('For the glacier', gdir.name)
print('Without calving the width is {} m and the thick is {} m'.format(width, thick))

# Defining a calving function
def calving_from_depth(gdir):
    """ Finds a calving flux based on the approaches proposed by
        Huss and Hock, (2015) and Oerlemans and Nick (2005).

    We take the initial output of the model and surface elevation data
    to calculate the water depth of the calving front.
    """
    # We read the model output, of the last pixel of the inversion
    cl = gdir.read_pickle('inversion_output', div_id=1)[-1]
    t_altitude = cl['hgt'][-1] #this gives us the altitude at the terminus
    width = cl['width'][-1]
    thick = cl['volume'][-1] / cl['dx'] / width

    # We calculate the water_depth
    w_depth = thick - t_altitude
    # or F_calving = np.amax(0, 2*H_free*w_depth) * width
    print('t_altitude', t_altitude)
    print('depth', w_depth)
    print('thick',thick)

    return np.absolute(((2*thick*w_depth)*width)/1e9), w_depth, thick

# Starting the calving loop, with a maximum of 50 cycles
i = 0
data_calving = []
w_depth = []
thick = []
forwrite = []

while i < 50:
    # First calculates a calving flux from model output with no calving
    F_calving, new_depth, new_thick = calving_from_depth(gdir)

    # Stores the data, and we see it
    data_calving += [F_calving]
    w_depth += [new_depth]
    thick += [new_thick]
    print('Calving rate calculated', F_calving)

    # We put the calving function output into the Model
    gdir.inversion_calving_rate = F_calving

    # Recompute mu with calving
    tasks.distribute_t_stars([gdir])

    # Inversion with calving, inv optimization is not iterative
    tasks.prepare_for_inversion(gdir)
    tasks.volume_inversion(gdir)
    tasks.distribute_thickness(gdir, how='per_altitude')

    i += 1
    #print('At the bottom i is', i)
    # calculating depth avg. to compare them, the loop is cut when it meets
    # the condition in the if statement
    avg_one = np.average(data_calving[-4:])
    avg_two = np.average(data_calving[-5:-1])
    difference = abs(avg_two - avg_one)
    if difference < 0.05*avg_two and i>5:
        break

# Making a dictionary for calving
cal_dic = dict(calving_fluxes=data_calving, water_depth=w_depth, H_ice=thick)
forwrite.append(cal_dic)
# We write out everything
gdir.write_pickle(forwrite, 'calving_output', div_id=1)

# We read everything
calving_output = gdir.read_pickle('calving_output', div_id=1)
for objt in calving_output:
    all_calving_data = objt['calving_fluxes']
    all_data_depth = objt['water_depth']
    all_data_H_i = objt['H_ice']


# we see the final calculated calving flux
last_calving = all_calving_data[-1]
print('last calving value', last_calving)

# Reinitialize everything
gdir.inversion_calving_rate = 0
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.prepare_for_inversion, gdirs)
tasks.optimize_inversion_params(gdirs)
execute_entity_task(tasks.volume_inversion, gdirs)

# Check if it is really doing it
# Un-comment it is not necessary to see it all the time
cl = gdir.read_pickle('inversion_output', div_id=1)[-1]
width = cl['width'][-1]
thick = cl['volume'][-1] / cl['dx'] / width
print('For the glacier', gdir.name)
print('Without calving the width is {} m and the thick is {} m'.format(width, thick))

# Do the inversion with the last calving flux
gdir.inversion_calving_rate = last_calving
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.prepare_for_inversion, gdirs)
execute_entity_task(tasks.volume_inversion, gdirs)
workflow.execute_entity_task(tasks.distribute_thickness, gdirs, how='per_altitude')

# Check how much it changes
cl = gdir.read_pickle('inversion_output', div_id=1)[-1]
width = cl['width'][-1]
thick = cl['volume'][-1] / cl['dx'] / width
w_depth = thick - cl['hgt'][-1]

print('For the glacier', gdir.name)
print('With calving the water depth is {} m and the thick is {} m'.format(w_depth, thick))

print('All calving', all_calving_data)
print('All water depth', all_data_depth)
print('All Hi', all_data_H_i)


# Uncomment the plots that you would like to see

# We plot calving flux vs water depth
# bname = os.path.join(PLOTS_DIR, gdir.name + '_' + gdir.rgi_id + '_withcalving')
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(np.arange(0, i, 1), all_calving_data, 'g-', linestyle='--', linewidth=3)
# ax1.set_ylabel('Calving Flux (Km³/yr)')
#
# ax2 = ax1.twinx()
# ax2.plot(np.arange(0, i, 1), all_data_depth, 'r-', linewidth=1)
# ax2.set_ylabel('Water depth (m)', color='r')
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')
# #plt.plot(np.arange(0, 20, 1), all_calving_data, 'r--', np.arange(0, 20, 1), all_data_H_i, 'g--')
# plt.savefig(bname + '_calvingflux_vs_wdepth.png')
# plt.show()

# Inversion
# graphics.plot_inversion(gdir)
# plt.savefig(bname + 'inv.png')
# plt.close()
# #
# # # Distribute thickness
# graphics.plot_distributed_thickness(gdir)
# plt.savefig(bname + 'inv_cor.png')
# plt.close()


# OTHER STUFF!!! that was in this script before but that I haven't touch

# if RUN_DYNAMICS:
#     # Random dynamics
#     execute_entity_task(tasks.init_present_time_glacier, gdirs)
#     execute_entity_task(tasks.random_glacier_evolution, gdirs)
#
