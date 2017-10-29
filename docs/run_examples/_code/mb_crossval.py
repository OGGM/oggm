# Python imports
from os import path

# Libs
import numpy as np
import pandas as pd
import geopandas as gpd

# Locals
import oggm
from oggm import cfg, utils, workflow
from oggm.core.massbalance import PastMassBalance
import matplotlib.pyplot as plt

# RGI Version
rgi_version = '5'

# Initialize OGGM and set up the run parameters
cfg.initialize()

# Local paths (where to write the OGGM run output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp',
                        'OGGM_ref_mb_RGIV{}'.format(rgi_version))
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

rgidf = gpd.read_file(path.join(WORKING_DIR, 'mb_ref_glaciers.shp'))

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf)

# Cross-validation
# What happens if we use the cross-validated mus and biases instead?
file = path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
cvdf = pd.read_csv(file, index_col=0)

# Reproduce Marzeion et al. (2012) Figure 3
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
    cvdf.loc[gd.rgi_id, 'CV_MB_BIAS'] = (refmb.OGGM.mean() -
                                         refmb.ANNUAL_BALANCE.mean())
    cvdf.loc[gd.rgi_id, 'CV_MB_SIGMA_BIAS'] = (refmb.OGGM.std() /
                                               std_ref)
    cvdf.loc[gd.rgi_id, 'CV_MB_COR'] = rcor
    mb_mod = PastMassBalance(gd, mu_star=t_cvdf.interp_mustar,
                             bias=t_cvdf.cv_bias,
                             prcp_fac=t_cvdf.cv_prcp_fac)
    refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths, year=refmb.index)
    cvdf.loc[gd.rgi_id, 'INTERP_MB_BIAS'] = (refmb.OGGM.mean() -
                                             refmb.ANNUAL_BALANCE.mean())

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
fn = path.join(WORKING_DIR, 'mb_crossval_rgi{}.pdf'.format(rgi_version))
plt.savefig(fn)

print('Median bias', cvdf['CV_MB_BIAS'].median())
print('Mean bias', cvdf['CV_MB_BIAS'].mean())
print('RMS', np.sqrt(np.mean(cvdf['CV_MB_BIAS']**2)))
print('Sigma bias', np.mean(cvdf['CV_MB_SIGMA_BIAS']))
