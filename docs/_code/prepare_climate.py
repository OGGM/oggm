import os
import geopandas as gpd
import oggm
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from oggm import cfg, tasks, graphics
from oggm.utils import get_demo_file
from oggm.core.preprocessing.climate import mb_yearly_climate_on_glacier, \
    t_star_from_refmb

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
tasks.compute_centerlines(gdir)

tasks.initialize_flowlines(gdir)
tasks.catchment_area(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)
cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
tasks.process_custom_climate_data(gdir)
tasks.mu_candidates(gdir)

# For plots
mu_yr_clim = gdir.read_pickle('mu_candidates')[pcp_fac]
mbdf = gdir.get_ref_mb_data()
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
pdf['$\mu (t)$'] = mu_yr_clim
pdf['bias'] = diff
t_stars, bias, _ = t_star_from_refmb(gdir, mbdf.ANNUAL_BALANCE)
