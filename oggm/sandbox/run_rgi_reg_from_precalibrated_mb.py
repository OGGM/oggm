"""Run OGGM for a couple of slected glaciers. The difference here is that we
use a precalibrated list of tstars for the run, i.e. won't calibrate the mass
balance anymore.
"""

# Module logger
import logging
# Python imports
import os
import shutil
from glob import glob

log = logging.getLogger(__name__)

# Libs
import matplotlib.pyplot as plt
import salem

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils
from oggm.core import flowline

# Time
import time
start = time.time()

# Alaska 01
# Western Canada and US 02
# Arctic Canada North 03
# Arctic Canada South 04
# Greenland 05
# Iceland 06
# Svalbard 07
# Scandinavia 08
# Russian Arctic 09
# North Asia 10
# North Asia 10
# Central Europe 11
# Caucasus and Middle East 12
# Central Asia 13
# South Asia West 14
# South Asia East 15
# Low Latitudes 16
# Southern Andes 17
# New Zealand 18
# Antarctic and Subantarctic 19
rgi_reg = '07'

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

cfg.initialize()

# Local paths (where to write output and where to download input)
WORKING_DIR = '/home/mowglie/disk/OGGM_Runs/TESTS'
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

PLOTS_DIR = os.path.join(WORKING_DIR, 'plots')
utils.mkdir(PLOTS_DIR)

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = False

# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 100

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = False
cfg.PARAMS['auto_skip_task'] = True

# Don't use divides for now
cfg.set_divides_db()

# But we use intersects
rgi_dir = utils.get_rgi_intersects_dir()
rgi_shp = list(glob(os.path.join(rgi_dir, "*", '*intersects*' + rgi_reg + '_rgi50_*.shp')))
assert len(rgi_shp) == 1
cfg.set_intersects_db(rgi_shp[0])

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
_ = utils.get_cru_file(var='pre')

# Copy the precalibrated tstar file
# ---------------------------------

# Note that to be exact, this procedure can only be applied if the run
# parameters don't change between the calibration and the run.
# After testing, it appears that changing the 'border' parameter won't affect
# the results much (expectedly), so that it's ok to change it. All the rest
# (e.g. smoothing, dx, prcp factor...) should imply a re-calibration

mbf = 'https://dl.dropboxusercontent.com/u/20930277/ref_tstars_no_tidewater.csv'
mbf = utils.file_downloader(mbf)
shutil.copyfile(mbf, os.path.join(WORKING_DIR, 'ref_tstars.csv'))


# Copy the RGI file
# -----------------

# Download RGI files
rgi_dir = utils.get_rgi_dir()
rgi_shp = list(glob(os.path.join(rgi_dir, "*", rgi_reg+ '_rgi50_*.shp')))
assert len(rgi_shp) == 1
rgidf = salem.read_shapefile(rgi_shp[0], cached=True)

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

rgidf = rgidf.loc[rgidf.RGIId.isin(['RGI50-07.01394'])]

log.info('Starting run for RGI reg: ' + rgi_reg)
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

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

# Climate tasks -- only data preparation and tstar interpolation!
execute_entity_task(tasks.process_cru_data, gdirs)
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.apparent_mb, gdirs)

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs)
execute_entity_task(tasks.volume_inversion, gdirs, glen_a=cfg.A, fs=0)
execute_entity_task(tasks.filter_inversion_output, gdirs)
execute_entity_task(tasks.init_present_time_glacier, gdirs)

# Compile output
utils.glacier_characteristics(gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM is done! Time needed: %d:%02d:%02d" % (h, m, s))

# Plots (if you want)
if PLOTS_DIR == '':
    exit()

utils.mkdir(PLOTS_DIR)
for gd in gdirs:

    bname = os.path.join(PLOTS_DIR, gd.rgi_id + '_')
    demsource = ' (' + gd.read_pickle('dem_source') + ')'

    fn = bname + '0_ggl.png'
    if not os.path.exists(fn):
        graphics.plot_googlemap(gd, reset=True)
        plt.savefig(fn)
        plt.close()

    fn = bname + '1_dom.png'
    if not os.path.exists(fn):
        graphics.plot_domain(gd, title_comment=demsource, reset=True)
        plt.savefig(fn)
        plt.close()
        plt.close()

    fn = bname + '2_cls.png'
    if not os.path.exists(fn):
        graphics.plot_centerlines(gd, title_comment=demsource, reset=True)
        plt.savefig(fn)
        plt.close()

    fn = bname + '3_fls.png'
    if not os.path.exists(fn):
        graphics.plot_centerlines(gd, title_comment=demsource,
                                  use_flowlines=True, add_downstream=True,
                                  reset = True)
        plt.savefig(fn)
        plt.close()

    fn = bname + '4_widths.png'
    if not os.path.exists(fn):
        graphics.plot_catchment_width(gd, corrected=True,
                                      add_intersects=True,
                                      add_touches=True,
                                      reset=True)
        plt.savefig(fn)
        plt.close()

    fn = bname + '5_thick.png'
    if not os.path.exists(fn):
        fls = gd.read_pickle('model_flowlines')
        model = flowline.FlowlineModel(fls)
        graphics.plot_modeloutput_map(gd,  model=model, reset=True)
        plt.savefig(fn)
        plt.close()
