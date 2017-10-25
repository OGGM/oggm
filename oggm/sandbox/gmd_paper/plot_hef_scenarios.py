import os

import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

import oggm
from oggm import cfg, tasks, graphics
from oggm import utils
from oggm.core.flowline import (FileModel)
from oggm.sandbox.gmd_paper import PLOT_DIR
from oggm.utils import get_demo_file, mkdir

cfg.initialize()
reset = True

cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
cfg.PARAMS['border'] = 60
cfg.PARAMS['auto_skip_task'] = True

base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'scenarios')
cfg.PATHS['working_dir'] = base_dir
mkdir(base_dir, reset=reset)

entity = gpd.read_file(get_demo_file('HEF_MajDivide.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir)

tasks.define_glacier_region(gdir, entity=entity)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.initialize_flowlines(gdir)
tasks.compute_downstream_line(gdir)
tasks.compute_downstream_bedshape(gdir)
tasks.catchment_area(gdir)
tasks.catchment_intersections(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)
tasks.process_cru_data(gdir)
tasks.mu_candidates(gdir)
tasks.compute_ref_t_stars([gdir])
tasks.distribute_t_stars([gdir])
tasks.apparent_mb(gdir)
tasks.prepare_for_inversion(gdir)
tasks.volume_inversion(gdir, glen_a=cfg.A, fs=0)
tasks.filter_inversion_output(gdir)
tasks.init_present_time_glacier(gdir)

df = utils.glacier_characteristics([gdir], path=False)

reset = True
seed = 0

tasks.random_glacier_evolution(gdir, nyears=800, seed=0, y0=2000,
                               filesuffix='_2000_def', reset=reset)

tasks.random_glacier_evolution(gdir, nyears=800, seed=0, y0=1920,
                               filesuffix='_1920_def', reset=reset)


f = gdir.get_filepath('model_diagnostics', filesuffix='_2000_def')
ds1 = xr.open_dataset(f)
f = gdir.get_filepath('model_diagnostics', filesuffix='_1920_def')
ds2 = xr.open_dataset(f)


f = plt.figure(figsize=(9, 6))
from mpl_toolkits.axes_grid1 import ImageGrid
axs = ImageGrid(f, 111,  # as in plt.subplot(111)
                nrows_ncols=(2, 2),
                axes_pad=0.15,
                share_all=True,
                cbar_location="right",
                cbar_mode="edge",
                cbar_size="7%",
                cbar_pad=0.15,
                )
f.delaxes(axs[0])
f.delaxes(axs[1])
f.delaxes(axs[1].cax)

tx, ty = 0.019, .975
letkm = dict(color='black', ha='left', va='top', fontsize=14,
             bbox=dict(facecolor='white', edgecolor='black'))
llkw = {'interval': 0}

fp = gdir.get_filepath('model_run', filesuffix='_2000_def')
model = FileModel(fp)
model.run_until(800)
ax = axs[3]
graphics.plot_modeloutput_map(gdir, model=model, ax=ax, title='',
                              lonlat_contours_kwargs=llkw, cbar_ax=ax.cax,
                              linewidth=1.5,
                              add_scalebar=False, vmax=300)
ax.text(tx, ty, 'c: [1985-2015]', transform=ax.transAxes, **letkm)


fp = gdir.get_filepath('model_run', filesuffix='_1920_def')
model = FileModel(fp)
model.run_until(800)
ax = axs[2]
graphics.plot_modeloutput_map(gdir, model=model, ax=ax, title='',
                              lonlat_contours_kwargs=llkw,
                              add_colorbar=False, linewidth=1.5,
                              add_scalebar=False, vmax=300)
ax.text(tx, ty, 'b: [1905-1935]', transform=ax.transAxes, **letkm)

ax = f.add_axes([0.25, 0.57, 0.6, 0.3])
ax.axhline(df.inv_volume_km3.values[0], 0, 800, color='k')
(ds2.volume_m3 * 1e-9).plot(ax=ax, label='[1905-1935]')
(ds1.volume_m3 * 1e-9).plot(ax=ax, label='[1985-2015]')
ax.set_xlabel('Years')
ax.set_ylabel('Volume [km$^3$]')
ax.legend(loc=[0.72, 0.2])
ax.text(0.01, .97, 'a', transform=ax.transAxes, **letkm)

plt.savefig(PLOT_DIR + 'hef_scenarios.pdf', dpi=150, bbox_inches='tight')
