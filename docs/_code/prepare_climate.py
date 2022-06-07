import os
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import oggm
from oggm import cfg, tasks, graphics
from oggm.core.climate import (mb_yearly_climate_on_glacier,
                               t_star_from_refmb,
                               local_t_star, mu_star_calibration)
from oggm.core.massbalance import (ConstantMassBalance)
from oggm.utils import get_demo_file, gettempdir
from oggm.shop import histalp

cfg.initialize()
cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
histalp.set_histalp_url('https://cluster.klima.uni-bremen.de/~oggm/'
                        'test_climate/histalp/')

base_dir = gettempdir('Climate_docs')
cfg.PATHS['working_dir'] = base_dir
entity = gpd.read_file(get_demo_file('Hintereisferner_RGI5.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir, reset=True)

tasks.define_glacier_region(gdir)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.initialize_flowlines(gdir)
tasks.compute_downstream_line(gdir)
tasks.catchment_area(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)
cfg.PARAMS['baseline_climate'] = 'HISTALP'
tasks.process_histalp_data(gdir)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    mu_yr_clim = tasks.glacier_mu_candidates(gdir)

mbdf = gdir.get_ref_mb_data()
res = t_star_from_refmb(gdir, mbdf=mbdf.ANNUAL_BALANCE)
local_t_star(gdir, tstar=res['t_star'], bias=res['bias'], reset=True)
mu_star_calibration(gdir, reset=True)

# For flux plot
tasks.prepare_for_inversion(gdir)

# For plots
years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir)

# which years to look at
selind = np.searchsorted(years, mbdf.index)
temp_yr = np.mean(temp_yr[selind])
prcp_yr = np.mean(prcp_yr[selind])

# Average oberved mass balance
ref_mb = mbdf.ANNUAL_BALANCE.mean()
mb_per_mu = prcp_yr - mu_yr_clim * temp_yr

# Diff to reference
diff = mb_per_mu - ref_mb
pdf = pd.DataFrame()
pdf[r'$\mu (t)$'] = mu_yr_clim
pdf['bias'] = diff
res = t_star_from_refmb(gdir, mbdf=mbdf.ANNUAL_BALANCE)

# For the mass flux
cl = gdir.read_pickle('inversion_input')[-1]
mbmod = ConstantMassBalance(gdir)
mbx = (mbmod.get_annual_mb(cl['hgt']) * cfg.SEC_IN_YEAR *
       cfg.PARAMS['ice_density'])
fdf = pd.DataFrame(index=np.arange(len(mbx))*cl['dx'])
fdf['Flux'] = cl['flux']
fdf['Mass balance'] = mbx

# For the distributed thickness
tasks.mass_conservation_inversion(gdir, glen_a=2.4e-24 * 3, fs=0)
tasks.distribute_thickness_per_altitude(gdir)


# plot functions
def example_plot_temp_ts():
    d = xr.open_dataset(gdir.get_filepath('climate_historical'))
    temp = d.temp.resample(time='12MS').mean('time').to_series()
    temp.index = temp.index.year
    try:
        temp = temp.rename_axis(None)
    except AttributeError:
        del temp.index.name
    temp.plot(figsize=(8, 4), label='Annual temp')
    tsm = temp.rolling(31, center=True, min_periods=15).mean()
    tsm.plot(label='31-yr avg')
    plt.legend(loc='best')
    plt.title('HISTALP annual temperature, Hintereisferner')
    plt.ylabel(r'degC')
    plt.tight_layout()
    plt.show()


def example_plot_mu_ts():
    mu_yr_clim.plot(figsize=(8, 4), label=r'$\mu (t)$')
    plt.legend(loc='best')
    plt.title(r'$\mu$ candidates Hintereisferner')
    plt.ylabel(r'$\mu$ (mm yr$^{-1}$ K$^{-1}$)')
    plt.tight_layout()
    plt.show()


def example_plot_bias_ts():
    ax = pdf.plot(figsize=(8, 4), secondary_y='bias')
    plt.hlines(0, 1800, 2015, linestyles='-')
    ax.set_ylabel(r'$\mu$ (mm yr$^{-1}$ K$^{-1}$)')
    ax.set_title(r'$\mu$ candidates HEF')
    plt.ylabel(r'bias (mm yr$^{-1}$)')
    yl = plt.gca().get_ylim()
    plt.plot((res['t_star'], res['t_star']), (yl[0], 0),
             linestyle=':', color='grey')
    plt.ylim(yl)
    plt.tight_layout()
    plt.show()


def example_plot_massflux():
    fig, ax = plt.subplots(figsize=(8, 4))
    fdf.plot(ax=ax, secondary_y='Mass balance', style=['C1-', 'C0-'])
    plt.axhline(0., color='grey', linestyle=':')
    ax.set_ylabel('Flux [m$^3$ s$^{-1}$]')
    ax.right_ax.set_ylabel('MB [kg m$^{-2}$ yr$^{-1}$]')
    ax.set_xlabel('Distance along flowline (m)')
    plt.title('Mass flux and mass balance along flowline')
    plt.tight_layout()
    plt.show()
