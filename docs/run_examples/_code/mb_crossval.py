# Python imports
import os

# Libs
import numpy as np
import pandas as pd
import geopandas as gpd

# Locals
import oggm
from oggm import cfg, workflow, tasks, utils
from oggm.core.massbalance import PastMassBalance, MultipleFlowlineMassBalance
import matplotlib.pyplot as plt

# RGI Version
rgi_version = '61'

# CRU or HISTALP?
baseline = 'CRU'

# Initialize OGGM and set up the run parameters
cfg.initialize()

if baseline == 'HISTALP':
    # Other params: see https://oggm.org/2018/08/10/histalp-parameters/
    cfg.PARAMS['prcp_scaling_factor'] = 1.75
    cfg.PARAMS['temp_melt'] = -1.75

# Local paths (where to find the OGGM run output)
dirname = 'OGGM_ref_mb_{}_RGIV{}_OGGM{}'.format(baseline, rgi_version,
                                                oggm.__version__)
WORKING_DIR = utils.gettempdir(dirname, home=True)
cfg.PATHS['working_dir'] = WORKING_DIR

# Read the rgi file
rgidf = gpd.read_file(os.path.join(WORKING_DIR, 'mb_ref_glaciers.shp'))

# Go - initialize glacier directories
gdirs = workflow.init_glacier_regions(rgidf)

# Cross-validation
file = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
ref_df = pd.read_csv(file, index_col=0)
for i, gdir in enumerate(gdirs):

    print('Cross-validation iteration {} of {}'.format(i + 1, len(ref_df)))

    # Now recalibrate the model blindly
    tmp_ref_df = ref_df.loc[ref_df.index != gdir.rgi_id]
    tasks.local_t_star(gdir, ref_df=tmp_ref_df)
    tasks.mu_star_calibration(gdir)

    # Mass-balance model with cross-validated parameters instead
    mb_mod = MultipleFlowlineMassBalance(gdir, mb_model_class=PastMassBalance,
                                         use_inversion_flowlines=True)

    # Mass-balance timeseries, observed and simulated
    refmb = gdir.get_ref_mb_data().copy()
    refmb['OGGM'] = mb_mod.get_specific_mb(year=refmb.index)

    # Compare their standard deviation
    std_ref = refmb.ANNUAL_BALANCE.std()
    rcor = np.corrcoef(refmb.OGGM, refmb.ANNUAL_BALANCE)[0, 1]
    if std_ref == 0:
        # I think that such a thing happens with some geodetic values
        std_ref = refmb.OGGM.std()
        rcor = 1

    # Store the scores
    ref_df.loc[gdir.rgi_id, 'CV_MB_BIAS'] = (refmb.OGGM.mean() -
                                             refmb.ANNUAL_BALANCE.mean())
    ref_df.loc[gdir.rgi_id, 'CV_MB_SIGMA_BIAS'] = (refmb.OGGM.std() / std_ref)
    ref_df.loc[gdir.rgi_id, 'CV_MB_COR'] = rcor

# Write out
ref_df.to_csv(os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv'))

# Marzeion et al Figure 3
f, ax = plt.subplots(1, 1)
bins = np.arange(20) * 400 - 3800
ref_df['CV_MB_BIAS'].plot(ax=ax, kind='hist', bins=bins, color='C3', label='')
ax.vlines(ref_df['CV_MB_BIAS'].mean(), 0, 120, linestyles='--', label='Mean')
ax.vlines(ref_df['CV_MB_BIAS'].quantile(), 0, 120, label='Median')
ax.vlines(ref_df['CV_MB_BIAS'].quantile([0.05, 0.95]), 0, 120, color='grey',
          label='5% and 95%\npercentiles')
ax.text(0.01, 0.99, 'N = {}'.format(len(gdirs)),
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

ax.set_ylim(0, 120)
ax.set_ylabel('N Glaciers')
ax.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
ax.legend(loc='best')
plt.tight_layout()
plt.show()

scores = 'Median bias: {:.2f}\n'.format(ref_df['CV_MB_BIAS'].median())
scores += 'Mean bias: {:.2f}\n'.format(ref_df['CV_MB_BIAS'].mean())
scores += 'RMS: {:.2f}\n'.format(np.sqrt(np.mean(ref_df['CV_MB_BIAS']**2)))
scores += 'Sigma bias: {:.2f}\n'.format(np.mean(ref_df['CV_MB_SIGMA_BIAS']))

# Output
print(scores)
fn = os.path.join(WORKING_DIR, 'scores.txt')
with open(fn, 'w') as f:
    f.write(scores)
