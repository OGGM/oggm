import os
import geopandas as gpd
import matplotlib.pyplot as plt

import oggm
from oggm import cfg, tasks, graphics
from oggm.sandbox.gmd_paper import PLOT_DIR
from oggm import utils

cfg.initialize()
cfg.PARAMS['border'] = 15
cfg.PARAMS['auto_skip_task'] = True
reset = False

base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'Workflow')
cfg.PATHS['working_dir'] = base_dir
utils.mkdir(base_dir, reset=reset)

entity = gpd.read_file(utils.get_demo_file('HEF_MajDivide.shp')).iloc[0]

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
tasks.compute_ref_t_stars([gdir])
tasks.distribute_t_stars([gdir])

glen_a = cfg.A * 1.5

tasks.prepare_for_inversion(gdir)
tasks.volume_inversion(gdir, use_cfg_params={'fs':0, 'glen_a':glen_a})
tasks.filter_inversion_output(gdir)

# run
tasks.init_present_time_glacier(gdir)
tasks.random_glacier_evolution(gdir, bias=0., glen_a=glen_a, nyears=200,
                               seed=1)

# ds = utils.compile_run_output([gdir], path=False)
# ds.volume.plot();
# plt.figure()
# ds.length.plot();
# plt.show()
# exit(0)

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
                        add_colorbar=False,
                        vmax=250,
                        lonlat_contours_kwargs=llkw,
                        add_scalebar=False)
axs[4].text(xt, yt, 'e', **letkm)

graphics.plot_modeloutput_map(gdir, ax=axs[5], modelyr=175, title='',
                              linewidth=2, add_colorbar=True,
                              vmax=250,
                              cbar_ax=axs[5].cax,
                              lonlat_contours_kwargs=llkw,
                              add_scalebar=False)
axs[5].text(xt, yt, 'f', **letkm)

# plt.tight_layout()
plt.savefig(PLOT_DIR + 'workflow.pdf', dpi=150, bbox_inches='tight')
