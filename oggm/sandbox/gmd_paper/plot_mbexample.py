import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import oggm
from oggm import cfg, tasks
from oggm.core.climate import (mb_yearly_climate_on_glacier,
                               t_star_from_refmb, local_mustar, apparent_mb)
from oggm.sandbox.gmd_paper import PLOT_DIR
from oggm.utils import get_demo_file

cfg.initialize()
cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
pcp_fac = 2.5
cfg.PARAMS['prcp_scaling_factor'] = pcp_fac
cfg.PARAMS['auto_skip_task'] = True

base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'MB')
entity = gpd.read_file(get_demo_file('Hintereisferner.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir, reset=True)

tasks.define_glacier_region(gdir, entity=entity)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.initialize_flowlines(gdir)
tasks.catchment_area(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)

tasks.process_cru_data(gdir)
tasks.mu_candidates(gdir)

mbdf = gdir.get_ref_mb_data()
res = t_star_from_refmb(gdir, mbdf.ANNUAL_BALANCE)
local_mustar(gdir, tstar=res['t_star'][-1], bias=res['bias'][-1],
             prcp_fac=res['prcp_fac'], reset=True)
apparent_mb(gdir, reset=True)

# For plots
mu_yr_clim = gdir.read_pickle('mu_candidates')[pcp_fac]
years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir, pcp_fac)

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
f = 0.9
f, axs = plt.subplots(3, 1, figsize=(8*f, 10*f), sharex=True)

d = xr.open_dataset(gdir.get_filepath('climate_monthly'))
d['year'] = ('time', np.repeat(np.arange(d['time.year'][0]+1, d['time.year'][-1]+1), 12))


temp = d.temp.groupby(d.year).mean().to_series()
del temp.index.name
prcp = d.prcp.groupby(d.year).sum().to_series()
del prcp.index.name

temp.plot(ax=axs[0], label='Annual temp', color='C1')
temp.rolling(31, center=True, min_periods=15).mean().plot(ax=axs[0], linewidth=3,
                                                          color='C1',
                                                          label='31-yr avg')
axs[0].legend(loc='best')
axs[0].set_ylabel(r'°C')
axs[0].set_xlim(1901, 2015)

prcp.plot(ax=axs[1], label='Annual prcp', color='C0')
prcp.rolling(31, center=True, min_periods=15).mean().plot(ax=axs[1], linewidth=3,
                                                          color='C0',
                                                          label='31-yr avg')
axs[1].legend(loc='best')
axs[1].set_ylabel(r'mm yr$^{-1}$')

pdf[r'$\mu (t)$'].plot(ax=axs[2], linewidth=3, color='C2')
axsup = pdf['bias'].plot(ax=axs[2], secondary_y=True, linewidth=3, color='C3',
                         label='bias (right)')

plt.hlines(0, 1901, 2015, linestyles='-')
axs[2].set_ylabel(r'$\mu$ (mm yr$^{-1}$ K$^{-1}$)');
plt.ylabel(r'bias (mm w.e. yr$^{-1}$)')
yl = plt.gca().get_ylim()
for ts in res['t_star']:
    plt.plot((ts, ts), (yl[0], 0), linestyle=':', color='grey')
plt.ylim(yl)

handles, labels = [],[]
for ax in [axs[2], axsup]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
plt.legend(handles, labels, loc=6)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'mb_ex.pdf', dpi=150, bbox_inches='tight')
