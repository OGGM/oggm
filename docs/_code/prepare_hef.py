import os
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import oggm
from oggm import cfg, tasks, graphics
from oggm.core import massbalance
from oggm import workflow
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
tasks.compute_downstream_line(gdir)
tasks.compute_downstream_bedshape(gdir)
cfg.PARAMS['baseline_climate'] = 'HISTALP'
cfg.PARAMS['use_winter_prcp_fac'] = False
cfg.PARAMS['use_temp_bias_from_file'] = False
cfg.PARAMS['prcp_fac'] = 2.5
tasks.process_histalp_data(gdir)
tasks.mb_calibration_from_wgms_mb(gdir)

# For flux plot
tasks.apparent_mb_from_any_mb(gdir, mb_years=(1970, 2000))
tasks.prepare_for_inversion(gdir)
refv = 577852773  # From ITMIX
df = workflow.calibrate_inversion_from_consensus(gdir, volume_m3_reference=refv)
np.testing.assert_allclose(refv, df.vol_oggm_m3, rtol=0.01)

cl = gdir.read_pickle('inversion_input')[-1]
mbmod = massbalance.ConstantMassBalance(gdir, y0=1985)
mbx = mbmod.get_annual_mb(cl['hgt']) * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
fdf = pd.DataFrame(index=np.arange(len(mbx))*cl['dx'])
fdf['Flux'] = cl['flux']
fdf['Mass balance'] = mbx

# For thickness plot
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
