import os
import geopandas as gpd
import numpy as np
import oggm
from oggm import cfg, tasks, graphics
from oggm.utils import get_demo_file
import matplotlib.pyplot as plt
import xarray as xr
import scipy.optimize as optimization
from oggm import utils
from oggm.sandbox.gmd_paper import PLOT_DIR
from oggm.core.preprocessing.climate import (t_star_from_refmb,
                                             local_mustar_apparent_mb)
from oggm.core.preprocessing.inversion import (mass_conservation_inversion)
from oggm.core.models.flowline import (FluxBasedModel)
from oggm.core.models.massbalance import (RandomMassBalanceModel)

cfg.initialize()

# Don't use divides for now, or intersects
cfg.set_divides_db()
cfg.set_intersects_db()

cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
cfg.PARAMS['border'] = 60
cfg.PARAMS['auto_skip_task'] = True

base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'dynamics')
entity = gpd.read_file(get_demo_file('Hintereisferner_RGI5.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir)

tasks.define_glacier_region(gdir, entity=entity)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.compute_downstream_lines(gdir)
tasks.initialize_flowlines(gdir)
tasks.compute_downstream_bedshape(gdir)
tasks.catchment_area(gdir)
tasks.catchment_intersections(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)
tasks.process_cru_data(gdir)
tasks.mu_candidates(gdir)
tasks.compute_ref_t_stars([gdir])
tasks.distribute_t_stars([gdir])
tasks.prepare_for_inversion(gdir)
tasks.volume_inversion(gdir, use_cfg_params={'glen_a':cfg.A, 'fs':0})
tasks.filter_inversion_output(gdir)
tasks.init_present_time_glacier(gdir)

df = utils.glacier_characteristics([gdir], path=False)

reset = False
seed = 0
tasks.random_glacier_evolution(gdir, nyears=800, bias=0, seed=seed,
                               filesuffix='_fromzero_def', reset=reset,
                               zero_initial_glacier=True)

cfg.PARAMS['flowline_fs'] = 5.7e-20 * 0.5
tasks.random_glacier_evolution(gdir, nyears=800, bias=0, seed=seed,
                               filesuffix='_fromzero_fs', reset=reset,
                               zero_initial_glacier=True)

cfg.PARAMS['flowline_fs'] = 0
cfg.PARAMS['flowline_glen_a'] = cfg.A*2
tasks.random_glacier_evolution(gdir, nyears=800, bias=0, seed=seed,
                               filesuffix='_fromzero_2A', reset=reset,
                               zero_initial_glacier=True)

cfg.PARAMS['flowline_fs'] = 0
cfg.PARAMS['flowline_glen_a'] = cfg.A/2
tasks.random_glacier_evolution(gdir, nyears=800, bias=0, seed=seed,
                               filesuffix='_fromzero_halfA', reset=reset,
                               zero_initial_glacier=True)


f = gdir.get_filepath('model_diagnostics', filesuffix='_fromzero_def')
ds1 = xr.open_dataset(f)
f = gdir.get_filepath('model_diagnostics', filesuffix='_fromzero_fs')
ds2 = xr.open_dataset(f)
f = gdir.get_filepath('model_diagnostics', filesuffix='_fromzero_2A')
ds3 = xr.open_dataset(f)
f = gdir.get_filepath('model_diagnostics', filesuffix='_fromzero_halfA')
ds4 = xr.open_dataset(f)

letkm = dict(color='black', ha='left', va='top', fontsize=14,
             bbox=dict(facecolor='white', edgecolor='black'))
tx, ty = 0.017, .980
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax = ax1
ax.axhline(df.inv_volume_km3.values[0], 0, 800, color='k')
(ds1.volume_m3 * 1e-9).plot(ax=ax, linewidth=2, label='Default')
(ds2.volume_m3 * 1e-9).plot(ax=ax, label='Sliding')
(ds4.volume_m3 * 1e-9).plot(ax=ax, label=r'A / 2')
ax.set_xlabel('Years')
ax.set_ylabel('Volume [km$^3$]')
ax.text(tx, ty, 'a', transform=ax.transAxes, **letkm)

ax = ax2
ax.axhline(gdir.read_pickle('model_flowlines')[-1].length_m / 1000, 0, 800,
            color='k')
(ds1.length_m / 1000).plot(ax=ax, linewidth=2, label='Default')
(ds2.length_m / 1000).plot(ax=ax, label='Sliding')
(ds4.length_m / 1000).plot(ax=ax, label=r'A / 2')
ax.set_xlabel('Years')
ax.set_ylabel('Length [km]')
ax.text(tx, ty, 'b', transform=ax.transAxes, **letkm)
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_DIR + 'hef_dyns.pdf', dpi=150, bbox_inches='tight')
