"""Run OGGM for all glaciers with mass-balance data (WGMS)"""

# Python imports
import os
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

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

cfg.initialize()

# Local paths (where to write the OGGM run output)
WORKING_DIR = '/home/mowglie/disk/OGGM_Runs/WGMS_GLACIERS'
PLOTS_DIR = os.path.join(WORKING_DIR, 'plots')
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 80

# This is the default in OGGM
cfg.PARAMS['prcp_scaling_factor'] = 2.5

# Set to True for operational runs
cfg.CONTINUE_ON_ERROR = False
cfg.PARAMS['auto_skip_task'] = True

# Don't use divides for now
cfg.set_divides_db()

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
_ = utils.get_cru_file(var='pre')

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

# Download and read in the RGI file
rgif = 'https://dl.dropboxusercontent.com/u/20930277/RGI_list_WGMS_glaciers_noTidewater.zip'
# rgif = utils.file_downloader(rgif)
# with zipfile.ZipFile(rgif) as zf:
#     zf.extractall(WORKING_DIR)
rgif = os.path.join(WORKING_DIR, 'RGI_list_WGMS_glaciers_noTidewater.shp')
rgidf = salem.read_shapefile(rgif)

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

print('Number of glaciers: {}'.format(len(rgidf)))


# Go - initialize working directories
# -----------------------------------

# you can use the command below to reset your run -- use with caution!
# gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
gdirs = workflow.init_glacier_regions(rgidf)

# Prepro tasks
task_list = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.compute_downstream_lines,
    tasks.initialize_flowlines,
    tasks.compute_downstream_bedshape,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate tasks
execute_entity_task(tasks.process_cru_data, gdirs)
# tasks.quick_crossval_t_stars(gdirs)
# tasks.compute_ref_t_stars(gdirs)
# tasks.distribute_t_stars(gdirs)

# Model validation
# ----------------

# Tests: for all glaciers, the mass-balance around tstar and the
# bias with observation should be approx 0
from oggm.core.models.massbalance import (ConstantMassBalanceModel,
                                          PastMassBalanceModel)
for gd in gdirs:
    heights, widths = gd.get_inversion_flowline_hw()

    mb_mod = ConstantMassBalanceModel(gd, bias=0)  # bias=0 because of calib!
    mb = mb_mod.get_specific_mb(heights, widths)
    np.testing.assert_allclose(mb, 0, atol=5)  # numerical errors

    mb_mod = PastMassBalanceModel(gd)  # Here we need the computed bias
    refmb = gd.get_ref_mb_data().copy()
    refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths, year=refmb.index)
    np.testing.assert_allclose(refmb.OGGM.mean(), refmb.ANNUAL_BALANCE.mean(),
                               atol=5)

# Cross-validation
# What happens if we use the cross-validated mus and biases instead?
file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
cvdf = pd.read_csv(file, index_col=0)

# Reproduce Ben's Figure 3
for gd in gdirs:
    t_cvdf = cvdf.loc[gd.rgi_id]
    heights, widths = gd.get_inversion_flowline_hw()
    mb_mod = PastMassBalanceModel(gd, mu_star=t_cvdf.cv_mustar,
                                  bias=t_cvdf.cv_bias,
                                  prcp_fac=t_cvdf.cv_prcp_fac)
    refmb = gd.get_ref_mb_data().copy()
    refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths, year=refmb.index)
    cvdf.loc[gd.rgi_id, 'CV_MB_BIAS'] = refmb.OGGM.mean() - \
                                        refmb.ANNUAL_BALANCE.mean()
    mb_mod = PastMassBalanceModel(gd, mu_star=t_cvdf.interp_mustar,
                                  bias=t_cvdf.cv_bias,
                                  prcp_fac=t_cvdf.cv_prcp_fac)
    refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths, year=refmb.index)
    cvdf.loc[gd.rgi_id, 'INTERP_MB_BIAS'] = refmb.OGGM.mean() - \
                                            refmb.ANNUAL_BALANCE.mean()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
bins = np.linspace(-4000, 4000, 26)
cvdf['CV_MB_BIAS'].plot(ax=ax1, kind='hist', bins=bins, color='C3')
ax1.vlines(cvdf['CV_MB_BIAS'].quantile(), 0, 100)
ax1.vlines(cvdf['CV_MB_BIAS'].quantile([0.05, 0.95]), 0, 100, color='grey')
ax1.set_ylim(0, 90)
cvdf['INTERP_MB_BIAS'].plot(ax=ax2, kind='hist', bins=bins, color='C0')
ax2.vlines(cvdf['INTERP_MB_BIAS'].quantile(), 0, 100)
ax2.vlines(cvdf['INTERP_MB_BIAS'].quantile([0.05, 0.95]), 0, 100, color='grey')
plt.tight_layout()
plt.show()

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

    fn = bname + '2_fls.png'
    if not os.path.exists(fn):
        graphics.plot_centerlines(gd, title_comment=demsource,
                                  use_flowlines=True, add_downstream=True)
        plt.savefig(fn)
        plt.close()

    fn = bname + '3_widths.png'
    if not os.path.exists(fn):
        graphics.plot_catchment_width(gd, corrected=True,
                                      add_intersects=True,
                                      add_touches=True)
        plt.savefig(fn)
        plt.close()

    fn = bname + '4_mbref.png'
    if not os.path.exists(fn):
        mb_mod = PastMassBalanceModel(gd)
        refmb = gd.get_ref_mb_data().copy()
        heights, widths = gd.get_inversion_flowline_hw()
        refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths,
                                               year=refmb.index)
        refmb[['ANNUAL_BALANCE', 'OGGM']].plot()
        title = gd.rgi_id
        if gd.name and gd.name != '':
            title += ': ' + gd.name
        plt.title(title)
        plt.savefig(fn)
        plt.close()