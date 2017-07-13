import os
import geopandas as gpd
import numpy as np
import oggm
from oggm import cfg, tasks, graphics
from oggm.utils import get_demo_file
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from oggm.sandbox.gmd_paper import PLOT_DIR
from oggm.core.preprocessing.climate import (t_star_from_refmb,
                                             local_mustar_apparent_mb)
from oggm.core.preprocessing.inversion import (mass_conservation_inversion)
from oggm.core.models.flowline import (FluxBasedModel)
from oggm.core.models.massbalance import (RandomMassBalanceModel)

cfg.initialize()
cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
cfg.PARAMS['border'] = 25

base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'Workflow')
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
tasks.mu_candidates(gdir, div_id=0)
mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

res = t_star_from_refmb(gdir, mbdf)
local_mustar_apparent_mb(gdir, tstar=res['t_star'][-1],
                         bias=res['bias'][-1],
                         prcp_fac=res['prcp_fac'])
tasks.prepare_for_inversion(gdir)
ref_v = 0.573 * 1e9

def to_optimize(x):
    glen_a = cfg.A * x[0]
    v, _ = mass_conservation_inversion(gdir, fs=0.,
                                             glen_a=glen_a)
    return (v - ref_v)**2

out = optimization.minimize(to_optimize, [1],
                            bounds=((0.01, 10),),
                            tol=1e-4)['x']
glen_a = cfg.A * out[0]
fs = 0.
v, _ = mass_conservation_inversion(gdir, glen_a=glen_a, write=True)
d = dict(fs=fs, glen_a=glen_a)
d['factor_glen_a'] = out[0]
d['factor_fs'] = 0.
gdir.write_pickle(d, 'inversion_params')

# filter
tasks.filter_inversion_output(gdir)

# run
tasks.init_present_time_glacier(gdir)

mb = RandomMassBalanceModel(gdir, seed=1)
fls = gdir.read_pickle('model_flowlines')
model = FluxBasedModel(fls, mb_model=mb, y0=0, glen_a=glen_a)

model.run_until(150)

f = plt.figure(figsize=(10, 12))
from mpl_toolkits.axes_grid1 import ImageGrid

axs = ImageGrid(f, 111,  # as in plt.subplot(111)
                nrows_ncols=(3, 2),
                axes_pad=0.15,
                share_all=True,
                cbar_location="right",
                cbar_mode="edge",
                cbar_size="7%",
                cbar_pad=0.15,
                )

llkw = {'interval': 0}
letkm = dict(color='black', ha='left', va='top', fontsize=20,
             bbox=dict(facecolor='white', edgecolor='black'))

graphics.plot_domain(gdir, ax=axs[0], title='', add_colorbar=False,
                     lonlat_contours_kwargs=llkw)
xt, yt = 2.45, 2.45
axs[0].text(xt, yt, 'a', **letkm)

im = graphics.plot_centerlines(gdir, ax=axs[1], title='', add_colorbar=True,
                               lonlat_contours_kwargs=llkw, cbar_ax=axs[1].cax,
                               add_scalebar=False)
axs[1].text(xt, yt, 'b', **letkm)

graphics.plot_catchment_width(gdir, ax=axs[2], title='', add_colorbar=False,
                              lonlat_contours_kwargs=llkw,
                              add_scalebar=False)
axs[2].text(xt, yt, 'c', **letkm)

graphics.plot_catchment_width(gdir, ax=axs[3], title='', corrected=True,
                              add_colorbar=False,
                              lonlat_contours_kwargs=llkw,
                              add_scalebar=False)
axs[3].text(xt, yt, 'd', **letkm)

f.delaxes(axs[3].cax)

graphics.plot_inversion(gdir, ax=axs[4], title='', linewidth=2,
                        add_colorbar=False, vmax=200,
                        lonlat_contours_kwargs=llkw,
                        add_scalebar=False)
axs[4].text(xt, yt, 'e', **letkm)

graphics.plot_modeloutput_map(gdir, ax=axs[5], model=model, title='',
                              linewidth=2, subset=False, vmax=200,
                              add_colorbar=True, cbar_ax=axs[5].cax,
                              lonlat_contours_kwargs=llkw,
                              add_scalebar=False)
axs[5].text(xt, yt, 'f', **letkm)

# plt.tight_layout()
plt.savefig(PLOT_DIR + 'workflow.pdf', dpi=150, bbox_inches='tight')
