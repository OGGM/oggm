import os
import geopandas as gpd
import oggm
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from oggm import cfg, tasks
from oggm.utils import get_demo_file
from oggm.core.preprocessing.climate import mb_yearly_climate_on_glacier, \
    t_star_from_refmb, local_mustar_apparent_mb
from oggm.core.models.massbalance import PastMassBalanceModel

cfg.initialize()
cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
cfg.PATHS['wgms_rgi_links'] = get_demo_file('RGI_WGMS_oetztal.csv')
pcp_fac = 2.6
cfg.PARAMS['prcp_scaling_factor'] = pcp_fac

base_dir = os.path.join(os.path.expanduser('~'), 'Climate')
entity = gpd.read_file(get_demo_file('Hintereisferner.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir)

tasks.define_glacier_region(gdir, entity=entity)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)

tasks.initialize_flowlines(gdir)
tasks.catchment_area(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)
cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
tasks.process_custom_climate_data(gdir)
tasks.mu_candidates(gdir)

mbdf = gdir.get_ref_mb_data()
res = t_star_from_refmb(gdir, mbdf.ANNUAL_BALANCE)
local_mustar_apparent_mb(gdir, tstar=res['t_star'][-1], bias=res['bias'][-1],
                         prcp_fac=res['prcp_fac'])

# For plots
mu_yr_clim = gdir.read_pickle('mu_candidates')[pcp_fac]
years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir, pcp_fac, div_id=0)

# which years to look at
selind = np.searchsorted(years, mbdf.index)
temp_yr = np.mean(temp_yr[selind])
prcp_yr = np.mean(prcp_yr[selind])

# Average oberved mass-balance
ref_mb = mbdf.ANNUAL_BALANCE.mean()
mb_per_mu = prcp_yr - mu_yr_clim * temp_yr

# Diff to reference
diff = mb_per_mu - ref_mb
pdf = pd.DataFrame()
pdf[r'$\mu (t)$'] = mu_yr_clim
pdf['bias'] = diff
res = t_star_from_refmb(gdir, mbdf.ANNUAL_BALANCE)

# plot functions
def example_plot_temp_ts():
    d = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    temp = d.temp.resample(freq='12MS', dim='time', how=np.mean).to_series()
    del temp.index.name
    ax = temp.plot(figsize=(8, 4), label='Annual temp')
    temp.rolling(31, center=True, min_periods=15).mean().plot(label='31-yr avg')
    plt.legend(loc='best')
    plt.title('HISTALP annual temperature, Hintereisferner')
    plt.ylabel(r'degC')
    plt.tight_layout()
    plt.show()

def example_plot_mu_ts():
    ax = mu_yr_clim.plot(figsize=(8, 4), label=r'$\mu (t)$');
    plt.legend(loc='best'); plt.title(r'$\mu$ candidates Hintereisferner');
    plt.ylabel(r'$\mu$ (mm yr$^{-1}$ K$^{-1}$)')
    plt.tight_layout()
    plt.show()

def example_plot_bias_ts():
    ax = pdf.plot(figsize=(8, 4), secondary_y='bias')
    plt.hlines(0, 1800, 2015, linestyles='-')
    ax.set_ylabel(r'$\mu$ (mm yr$^{-1}$ K$^{-1}$)');
    ax.set_title(r'$\mu$ candidates HEF');
    plt.ylabel(r'bias (mm yr$^{-1}$)')
    yl = plt.gca().get_ylim()
    for ts in res['t_star']:
        plt.plot((ts, ts), (yl[0], 0), linestyle=':', color='grey')
    plt.ylim(yl)
    plt.tight_layout()
    plt.show()
