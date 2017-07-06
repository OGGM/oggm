"""Run OGGM for a couple of slected glaciers. The difference here is that we
use a precalibrated list of tstars for the run, i.e. won't calibrate the mass
balance anymore.
"""

# Python imports
import os
import shutil
import zipfile
# Libs
import oggm
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import salem

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils
from oggm.core.models import flowline

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

cfg.initialize()

# Local paths (where to write the OGGM run output)
WORKING_DIR = '/home/mowglie/disk/OGGM_Runs/EXAMPLE_GLACIERS'
PLOTS_DIR = os.path.join(WORKING_DIR, 'plots')
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 60

# This is the default in OGGM
cfg.PARAMS['prcp_scaling_factor'] = 2.5

# Set to True for operational runs
cfg.CONTINUE_ON_ERROR = False
cfg.PARAMS['auto_skip_task'] = False

# Test
cfg.PARAMS['mixed_min_shape'] = 0.003
cfg.PARAMS['default_parabolic_bedshape'] = 0.003
cfg.PARAMS['trapezoid_lambdas'] = 0.

# Don't use divides for now
cfg.set_divides_db()
cfg.set_intersects_db()

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

mbf = 'https://dl.dropboxusercontent.com/u/20930277/ref_tstars_b60_prcpfac_25_defaults.csv'
mbf = utils.file_downloader(mbf)
shutil.copyfile(mbf, os.path.join(WORKING_DIR, 'ref_tstars.csv'))

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

# Download and read in the RGI file
rgif = 'https://dl.dropboxusercontent.com/u/20930277/RGI_example_glaciers.zip'
rgif = utils.file_downloader(rgif)
with zipfile.ZipFile(rgif) as zf:
    zf.extractall(WORKING_DIR)
rgif = os.path.join(WORKING_DIR, 'RGI_example_glaciers.shp')
rgidf = salem.read_shapefile(rgif, cached=True)

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

# rgidf = rgidf.loc[rgidf.RGIId.isin(['RGI50-01.10299'])]

print('Number of glaciers: {}'.format(len(rgidf)))


# Go - initialize working directories
# -----------------------------------

# you can use the command below to reset your run -- use with caution!
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)

utils.glacier_characteristics(gdirs)
utils.compile_run_output(gdirs, filesuffix='_fromzero')
utils.compile_run_output(gdirs, filesuffix='_fromzero_newparams')
utils.compile_run_output(gdirs, filesuffix='_fromtoday')
utils.compile_run_output(gdirs, filesuffix='_fromtoday_newparams')

exit()

# Prepro tasks
task_list = [
    # tasks.glacier_masks,
    # tasks.compute_centerlines,
    # tasks.compute_downstream_lines,
    # tasks.initialize_flowlines,
    # tasks.compute_downstream_bedshape,
    # tasks.catchment_area,
    # tasks.catchment_intersections,
    # tasks.catchment_width_geom,
    # tasks.catchment_width_correction,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate tasks -- only data preparation and tstar interpolation!
# execute_entity_task(tasks.process_cru_data, gdirs)
# tasks.distribute_t_stars(gdirs)
#
# execute_entity_task(tasks.prepare_for_inversion, gdirs)
# execute_entity_task(tasks.volume_inversion, gdirs,
#                     use_cfg_params={'glen_a': cfg.A, 'fs': 0})
# execute_entity_task(tasks.filter_inversion_output, gdirs)
#
# Tests: for all glaciers, the mass-balance around tstar and the
# bias with observation should be approx 0

from oggm.core.models.massbalance import ConstantMassBalanceModel
for gd in gdirs:
    heights, widths = gd.get_inversion_flowline_hw()

    mb_mod = ConstantMassBalanceModel(gd, bias=0)  # bias=0 because of calib!
    mb = mb_mod.get_specific_mb(heights, widths)
    np.testing.assert_allclose(mb, 0, atol=10)  # numerical errors

execute_entity_task(tasks.init_present_time_glacier, gdirs)

# Go
workflow.execute_entity_task(tasks.random_glacier_evolution, gdirs, bias=0,
                             nyears=5000, seed=0,
                             filesuffix='_fromtoday_newparams')

# Plots (if you want)
if PLOTS_DIR == '':
    exit()

utils.mkdir(PLOTS_DIR)
for gd in gdirs:

    bname = os.path.join(PLOTS_DIR, gd.rgi_id + '_')
    demsource = ' (' + gd.read_pickle('dem_source') + ')'

    fn = bname + '0_ggl.png'
    if not os.path.exists(fn):
        graphics.plot_googlemap(gd)
        plt.savefig(fn)
        plt.close()

    fn = bname + '1_dom.png'
    if not os.path.exists(fn):
        graphics.plot_domain(gd, title_comment=demsource)
        plt.savefig(fn)
        plt.close()
        plt.close()

    fn = bname + '2_cls.png'
    if not os.path.exists(fn):
        graphics.plot_centerlines(gd, title_comment=demsource)
        plt.savefig(fn)
        plt.close()

    fn = bname + '3_fls.png'
    if not os.path.exists(fn):
        graphics.plot_centerlines(gd, title_comment=demsource,
                                  use_flowlines=True, add_downstream=True)
        plt.savefig(fn)
        plt.close()

    fn = bname + '4_widths.png'
    if not os.path.exists(fn):
        graphics.plot_catchment_width(gd, corrected=True,
                                      add_intersects=True,
                                      add_touches=True)
        plt.savefig(fn)
        plt.close()

    fn = bname + '5_thick.png'
    if not os.path.exists(fn):
        fls = gd.read_pickle('model_flowlines')
        model = flowline.FlowlineModel(fls)
        graphics.plot_modeloutput_map(gd,  model=model)
        plt.savefig(fn)
        plt.close()


