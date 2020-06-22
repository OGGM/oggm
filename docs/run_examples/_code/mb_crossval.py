# Python imports
import os
import json
import shutil

# Libs
import numpy as np
import pandas as pd

# Locals
import oggm
from oggm import cfg, workflow, tasks, utils
from oggm.core.massbalance import PastMassBalance, MultipleFlowlineMassBalance
import matplotlib.pyplot as plt

# Module logger
import logging
log = logging.getLogger(__name__)

# RGI Version
rgi_version = '62'

# Initialize OGGM and set up the run parameters
cfg.initialize(logging_level='WORKFLOW')

# Local paths (where to find the OGGM run output)
dirname = 'OGGM_ref_mb_RGIV{}_OGGM{}'.format(rgi_version, oggm.__version__)
WORKING_DIR = utils.gettempdir(dirname, home=True)
cfg.PATHS['working_dir'] = WORKING_DIR

for baseline in ['CRU', 'ERA5', 'ERA5L', 'CERA+ERA5', 'CERA+ERA5L']:

    # We are using which baseline data?
    cfg.PARAMS['baseline_climate'] = baseline

    params_file = os.path.join(WORKING_DIR, 'oggm_ref_tstars_rgi6_{}_calib_'
                                            'params.json'.format(baseline))
    with open(params_file, 'r') as fp:
        mbcalib = json.load(fp)
        for k, v in mbcalib.items():
            cfg.PARAMS[k] = v

    # Read the rgi ids of the reference glaciers
    ref_list_f = os.path.join(WORKING_DIR,
                              'mb_ref_glaciers_{}.csv'.format(baseline))
    rids = pd.read_csv(ref_list_f, index_col=0, squeeze=True)

    # Go - initialize glacier directories
    gdirs = workflow.init_glacier_directories(rids)

    # Replace current default climate file with prepared one
    for gdir in gdirs:
        fc = gdir.get_filepath('climate_historical', filesuffix=baseline)
        fo = gdir.get_filepath('climate_historical')
        shutil.copyfile(fc, fo)

    # Cross-validation
    file = os.path.join(cfg.PATHS['working_dir'],
                        os.path.join(WORKING_DIR,
                                     'oggm_ref_tstars_'
                                     'rgi6_{}.csv'.format(baseline)))
    ref_df = pd.read_csv(file, index_col=0)
    log.workflow('Cross-validation loop...')
    for i, gdir in enumerate(gdirs):
        # Now recalibrate the model blindly
        tmp_ref_df = ref_df.loc[ref_df.index != gdir.rgi_id]
        tasks.local_t_star(gdir, ref_df=tmp_ref_df)
        tasks.mu_star_calibration(gdir)

        # Mass-balance model with cross-validated parameters instead
        mb_mod = MultipleFlowlineMassBalance(gdir,
                                             mb_model_class=PastMassBalance,
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
        ref_df.loc[gdir.rgi_id, 'CV_MB_SIGMA_BIAS'] = refmb.OGGM.std() / std_ref
        ref_df.loc[gdir.rgi_id, 'CV_MB_COR'] = rcor

    # Write out
    ref_df.to_csv(os.path.join(cfg.PATHS['working_dir'],
                               'crossval_tstars_{}.csv'.format(baseline)))

    scores = 'N = {}\n'.format(len(gdirs))
    scores += 'Median bias: {:.2f}\n'.format(ref_df['CV_MB_BIAS'].median())
    scores += 'Mean bias: {:.2f}\n'.format(ref_df['CV_MB_BIAS'].mean())
    scores += 'RMS: {:.2f}\n'.format(np.sqrt(np.mean(ref_df['CV_MB_BIAS']**2)))
    scores += 'Sigma bias: {:.2f}\n'.format(np.mean(ref_df['CV_MB_SIGMA_BIAS']))

    # Marzeion et al Figure 3
    f, ax = plt.subplots(1, 1)
    bins = np.arange(20) * 400 - 3800
    ylim = 130
    ref_df['CV_MB_BIAS'].plot(ax=ax, kind='hist', bins=bins, color='C3',
                              label='')
    ax.vlines(ref_df['CV_MB_BIAS'].mean(), 0, ylim, linestyles='--',
              label='Mean')
    ax.vlines(ref_df['CV_MB_BIAS'].quantile(), 0, ylim, label='Median')
    ax.vlines(ref_df['CV_MB_BIAS'].quantile([0.05, 0.95]), 0, ylim,
              color='grey', label='5% and 95%\npercentiles')
    ax.text(0.01, 0.99, scores,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)

    ax.set_ylim(0, ylim)
    ax.set_ylabel('N Glaciers')
    ax.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
    ax.legend(loc='best')
    plt.title('Baseline: {}'.format(baseline))
    plt.tight_layout()
    plt.savefig(os.path.join(WORKING_DIR,
                             'cv_histogram_{}.png'.format(baseline)),
                dpi=150, bbox_inches='tight')
    plt.show()

    # Output
    print(scores)
    fn = os.path.join(WORKING_DIR, 'scores_{}.txt'.format(baseline))
    with open(fn, 'w') as f:
        f.write(scores)


# Smallest group of glaciers where all available
rids = []
for baseline in ['CRU', 'ERA5', 'ERA5L', 'CERA+ERA5', 'CERA+ERA5L']:
    # Read the rgi ids of the reference glaciers
    ref_list_f = os.path.join(WORKING_DIR,
                              'mb_ref_glaciers_{}.csv'.format(baseline))
    rids.append(set(pd.read_csv(ref_list_f, index_col=0, squeeze=True)))
rids = set.intersection(*rids)

# Go - initialize glacier directories
gdirs = workflow.init_glacier_directories(rids)

plot_dir = os.path.join(WORKING_DIR, 'same_glaciers_comparison')
utils.mkdir(plot_dir)

for baseline in ['CRU', 'ERA5', 'ERA5L', 'CERA+ERA5', 'CERA+ERA5L']:

    # We are using which baseline data?
    cfg.PARAMS['baseline_climate'] = baseline

    # Read the rgi ids of the reference glaciers
    params_file = os.path.join(WORKING_DIR, 'oggm_ref_tstars_rgi6_{}_calib_'
                                            'params.json'.format(baseline))
    with open(params_file, 'r') as fp:
        mbcalib = json.load(fp)
        for k, v in mbcalib.items():
            cfg.PARAMS[k] = v

    # Replace current default climate file with prepared one
    for gdir in gdirs:
        fc = gdir.get_filepath('climate_historical', filesuffix=baseline)
        fo = gdir.get_filepath('climate_historical')
        shutil.copyfile(fc, fo)

    # Cross-validation
    file = os.path.join(cfg.PATHS['working_dir'],
                        os.path.join(WORKING_DIR,
                                     'oggm_ref_tstars_'
                                     'rgi6_{}.csv'.format(baseline)))
    ref_df = pd.read_csv(file, index_col=0)
    ref_df = ref_df.loc[rids]

    log.workflow('Cross-validation loop...')
    for i, gdir in enumerate(gdirs):
        # Now recalibrate the model blindly
        tmp_ref_df = ref_df.loc[ref_df.index != gdir.rgi_id]
        tasks.local_t_star(gdir, ref_df=tmp_ref_df)
        tasks.mu_star_calibration(gdir)

        # Mass-balance model with cross-validated parameters instead
        mb_mod = MultipleFlowlineMassBalance(gdir,
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
        ref_df.loc[gdir.rgi_id, 'CV_MB_SIGMA_BIAS'] = refmb.OGGM.std() / std_ref
        ref_df.loc[gdir.rgi_id, 'CV_MB_COR'] = rcor

    # Write out
    ref_df.to_csv(os.path.join(plot_dir,
                               'crossval_tstars_{}.csv'.format(baseline)))

    scores = 'N = {}\n'.format(len(gdirs))
    scores += 'Median bias: {:.2f}\n'.format(ref_df['CV_MB_BIAS'].median())
    scores += 'Mean bias: {:.2f}\n'.format(ref_df['CV_MB_BIAS'].mean())
    scores += 'RMS: {:.2f}\n'.format(np.sqrt(np.mean(ref_df['CV_MB_BIAS']**2)))
    scores += 'Sigma bias: {:.2f}\n'.format(np.mean(ref_df['CV_MB_SIGMA_BIAS']))

    # Marzeion et al Figure 3
    f, ax = plt.subplots(1, 1)
    bins = np.arange(20) * 400 - 3800
    ylim = 130
    ref_df['CV_MB_BIAS'].plot(ax=ax, kind='hist', bins=bins, color='C3',
                              label='')
    ax.vlines(ref_df['CV_MB_BIAS'].mean(), 0, ylim, linestyles='--', label='Mean')
    ax.vlines(ref_df['CV_MB_BIAS'].quantile(), 0, ylim, label='Median')
    ax.vlines(ref_df['CV_MB_BIAS'].quantile([0.05, 0.95]), 0, ylim, color='grey',
              label='5% and 95%\npercentiles')
    ax.text(0.01, 0.99, scores,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)

    ax.set_ylim(0, ylim)
    ax.set_ylabel('N Glaciers')
    ax.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
    ax.legend(loc='best')
    plt.title('Baseline: {}'.format(baseline))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'cv_histogram_{}.png'.format(baseline)),
                dpi=150, bbox_inches='tight')
    plt.show()

    # Output
    print(scores)
    fn = os.path.join(plot_dir, 'scores_{}.txt'.format(baseline))
    with open(fn, 'w') as f:
        f.write(scores)