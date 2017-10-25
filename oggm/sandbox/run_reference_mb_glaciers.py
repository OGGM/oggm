"""Run OGGM for all glaciers with mass-balance data (WGMS)"""

# Python imports
import os
import zipfile

import matplotlib.pyplot as plt
# Libs
import numpy as np
import pandas as pd
import salem

# Locals
import oggm.cfg as cfg
from oggm import graphics, utils
from oggm import tasks
from oggm import workflow
from oggm.workflow import execute_entity_task

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

cfg.initialize()

# Local paths (where to write the OGGM run output)
WORKING_DIR = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_REF_MB_GLACIERS')
PLOTS_DIR = os.path.join(WORKING_DIR, 'plots')
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 60

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = False
cfg.PARAMS['auto_skip_task'] = True

# Don't use divides for now, or intersects
cfg.set_divides_db()
cfg.set_intersects_db()

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
_ = utils.get_cru_file(var='pre')

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

# Download and read in the RGI file
n = 'RGI_list_WGMS_glaciers_noTidewater'
rgif = 'https://www.dropbox.com/s/ekkl99o1lglljyg/' + n + '.zip?dl=1'
rgif = utils.file_downloader(rgif)
with zipfile.ZipFile(rgif) as zf:
    zf.extractall(WORKING_DIR)
rgif = os.path.join(WORKING_DIR, n + '.shp')
rgidf = salem.read_shapefile(rgif, cached=True)

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

# Climate tasks
execute_entity_task(tasks.process_cru_data, gdirs)
tasks.quick_crossval_t_stars(gdirs)
tasks.compute_ref_t_stars(gdirs)
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.apparent_mb, gdirs)

# Model validation
# ----------------

# Tests: for all glaciers, the mass-balance around tstar and the
# bias with observation should be approx 0
from oggm.core.massbalance import (ConstantMassBalance,
                                   PastMassBalance)
for gd in gdirs:
    heights, widths = gd.get_inversion_flowline_hw()

    mb_mod = ConstantMassBalance(gd, bias=0)  # bias=0 because of calib!
    mb = mb_mod.get_specific_mb(heights, widths)
    np.testing.assert_allclose(mb, 0, atol=10)  # numerical errors

    mb_mod = PastMassBalance(gd)  # Here we need the computed bias
    refmb = gd.get_ref_mb_data().copy()
    refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths, year=refmb.index)
    np.testing.assert_allclose(refmb.OGGM.mean(), refmb.ANNUAL_BALANCE.mean(),
                               atol=10)

# Cross-validation
# What happens if we use the cross-validated mus and biases instead?
file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
cvdf = pd.read_csv(file, index_col=0)

# Reproduce Ben's Figure 3
for gd in gdirs:
    t_cvdf = cvdf.loc[gd.rgi_id]
    heights, widths = gd.get_inversion_flowline_hw()
    mb_mod = PastMassBalance(gd, mu_star=t_cvdf.cv_mustar,
                             bias=t_cvdf.cv_bias,
                             prcp_fac=t_cvdf.cv_prcp_fac)
    refmb = gd.get_ref_mb_data().copy()
    refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths,
                                           year=refmb.index)
    std_ref = refmb.ANNUAL_BALANCE.std()
    rcor = np.corrcoef(refmb.OGGM, refmb.ANNUAL_BALANCE)[0, 1]
    if std_ref == 0:
        std_ref = refmb.OGGM.std()
        rcor = 1
    cvdf.loc[gd.rgi_id, 'CV_MB_BIAS'] = refmb.OGGM.mean() - \
                                        refmb.ANNUAL_BALANCE.mean()
    cvdf.loc[gd.rgi_id, 'CV_MB_SIGMA_BIAS'] = refmb.OGGM.std() / \
                                              std_ref
    cvdf.loc[gd.rgi_id, 'CV_MB_COR'] = rcor
    mb_mod = PastMassBalance(gd, mu_star=t_cvdf.interp_mustar,
                             bias=t_cvdf.cv_bias,
                             prcp_fac=t_cvdf.cv_prcp_fac)
    refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths, year=refmb.index)
    cvdf.loc[gd.rgi_id, 'INTERP_MB_BIAS'] = refmb.OGGM.mean() - \
                                            refmb.ANNUAL_BALANCE.mean()

# Plots (if you want)
if PLOTS_DIR == '':
    exit()

utils.mkdir(PLOTS_DIR)

# Ben Figure 3
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
bins = np.arange(20) * 400 - 3800
cvdf['CV_MB_BIAS'].plot(ax=ax1, kind='hist', bins=bins, color='C3', label='')
ax1.vlines(cvdf['CV_MB_BIAS'].mean(), 0, 120, linestyles='--', label='Mean')
ax1.vlines(cvdf['CV_MB_BIAS'].quantile(), 0, 120, label='Median')
ax1.vlines(cvdf['CV_MB_BIAS'].quantile([0.05, 0.95]), 0, 120, color='grey',
                                       label='5% and 95%\npercentiles')
ax1.set_ylim(0, 120)
ax1.set_ylabel('N Glaciers')
ax1.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
ax1.legend(loc='best')
cvdf['INTERP_MB_BIAS'].plot(ax=ax2, kind='hist', bins=bins, color='C0')
ax2.vlines(cvdf['INTERP_MB_BIAS'].mean(), 0, 120, linestyles='--')
ax2.vlines(cvdf['INTERP_MB_BIAS'].quantile(), 0, 120)
ax2.vlines(cvdf['INTERP_MB_BIAS'].quantile([0.05, 0.95]), 0, 120, color='grey')
ax2.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
plt.tight_layout()
fn = os.path.join(PLOTS_DIR, '00_mb_crossval.pdf')
plt.savefig(fn)

print('Median bias', cvdf['CV_MB_BIAS'].median())
print('Mean bias', cvdf['CV_MB_BIAS'].mean())
print('RMS', np.sqrt(np.mean(cvdf['CV_MB_BIAS']**2)))
print('Sigma bias', np.mean(cvdf['CV_MB_SIGMA_BIAS']))

# Plots per glacier
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
                                  use_flowlines=True,
                                  add_downstream=True)
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
        mb_mod = PastMassBalance(gd)
        refmb = gd.get_ref_mb_data().copy()
        heights, widths = gd.get_inversion_flowline_hw()
        refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths,
                                               year=refmb.index)
        refmb = refmb.reindex(np.arange(refmb.index[0], refmb.index[-1]+1))
        refmb[['ANNUAL_BALANCE', 'OGGM']].plot()
        title = gd.rgi_id
        if gd.name and gd.name != '':
            title += ': ' + gd.name
        plt.title(title)
        plt.savefig(fn)
        plt.close()
