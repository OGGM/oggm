"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Before everythin els for the logger setting
import oggm

# Logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import salem
import zipfile
# Libs
import matplotlib.pyplot as plt
# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils

# Initialize OGGM
cfg.initialize()

# Local paths (where to write output and where to download input)
WORKING_DIR = '/home/mowglie/disk/OGGM_Runs/BENCHMARK'
PLOTS_DIR = ''

cfg.PATHS['working_dir'] = WORKING_DIR
utils.mkdir(WORKING_DIR)

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 60

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['auto_skip_task'] = True

# Don't use divides for now
cfg.set_divides_db()

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
_ = utils.get_cru_file(var='pre')

# Read in the Benchmark RGI file
rgif = 'https://dl.dropboxusercontent.com/u/20930277/rgi_benchmark.zip'
rgif = utils.file_downloader(rgif)
with zipfile.ZipFile(rgif) as zf:
    zf.extractall(WORKING_DIR)
rgif = os.path.join(WORKING_DIR, 'rgi_benchmark.shp')
rgidf = salem.read_shapefile(rgif, cached=True)

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf)  # reset=True, force=True

# Prepro tasks
task_list = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.compute_downstream_line,
    tasks.initialize_flowlines,
    tasks.compute_downstream_bedshape,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate related tasks - this will download
execute_entity_task(tasks.process_cru_data, gdirs)
# tasks.compute_ref_t_stars(gdirs)
# tasks.distribute_t_stars(gdirs)

# Inversion
execute_entity_task(tasks.prepare_for_inversion, gdirs)
execute_entity_task(tasks.volume_inversion, gdirs, glen_a=cfg.A, fs=0)
execute_entity_task(tasks.filter_inversion_output, gdirs,)

# Run
execute_entity_task(tasks.init_present_time_glacier, gdirs)

# While the above should work always, this here is no piece of fun
execute_entity_task(tasks.random_glacier_evolution, gdirs)

# Write out glacier statistics
df = utils.glacier_characteristics(gdirs)
fpath = os.path.join(cfg.PATHS['working_dir'], 'glacier_char.csv')
df.to_csv(fpath)

# Plots (if you want)
if PLOTS_DIR == '':
    exit()

utils.mkdir(PLOTS_DIR)
for gd in gdirs:
    bname = os.path.join(PLOTS_DIR, gd.name + '_' + gd.rgi_id + '_')
    graphics.plot_googlemap(gd)
    plt.savefig(bname + 'ggl.png')
    plt.close()
    graphics.plot_domain(gd)
    plt.savefig(bname + 'dom.png')
    plt.close()
    graphics.plot_centerlines(gd)
    plt.savefig(bname + 'cls.png')
    plt.close()
    graphics.plot_catchment_width(gd, corrected=True)
    plt.savefig(bname + 'w.png')
    plt.close()
    graphics.plot_inversion(gd)
    plt.savefig(bname + 'inv.png')
    plt.close()
