import os
import geopandas as gpd
import numpy as np
import oggm
import zipfile
from oggm import cfg, tasks, graphics
from oggm.utils import get_demo_file
import matplotlib.pyplot as plt
import xarray as xr
from oggm import workflow
from oggm.workflow import execute_entity_task
import salem
from oggm import utils
from oggm.sandbox.gmd_paper import PLOT_DIR

cfg.initialize()

cfg.PARAMS['border'] = 60
cfg.PARAMS['auto_skip_task'] = True

base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'iceland')
utils.mkdir(base_dir)
cfg.PATHS['working_dir'] = base_dir

# Use intersects
f = 'https://www.dropbox.com/s/om7qkdayt24frbw/rgiv5_Eyjafjallajoekull_intersects.zip?dl=1'
f = utils.file_downloader(f)
with zipfile.ZipFile(f) as zf:
    zf.extractall(base_dir)
f = os.path.join(base_dir, 'rgiv5_Eyjafjallajoekull_intersects.shp')
cfg.set_intersects_db(f)

rgif = 'https://www.dropbox.com/s/48dhohcmiiyggou/rgiv5_Eyjafjallajoekull.zip?dl=1'
rgif = utils.file_downloader(rgif)
with zipfile.ZipFile(rgif) as zf:
    zf.extractall(base_dir)
rgif = os.path.join(base_dir, 'rgiv5_Eyjafjallajoekull.shp')
rgidf = gpd.read_file(rgif)

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
_ = utils.get_cru_file(var='pre')

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

# Prepro tasks
task_list = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.compute_downstream_line,
    tasks.compute_downstream_bedshape,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate tasks
execute_entity_task(tasks.process_cru_data, gdirs)
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.apparent_mb, gdirs)

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs)
execute_entity_task(tasks.volume_inversion, gdirs, glen_a=cfg.A, fs=0)
execute_entity_task(tasks.filter_inversion_output, gdirs)
execute_entity_task(tasks.init_present_time_glacier, gdirs)

# Compile output
utils.glacier_characteristics(gdirs)

seed = 0
execute_entity_task(tasks.random_glacier_evolution, gdirs,
                    nyears=500, bias=0, seed=seed, temperature_bias=0,
                    filesuffix='_defaults')
execute_entity_task(tasks.random_glacier_evolution, gdirs,
                    nyears=500, bias=0, seed=seed, temperature_bias=-0.2,
                    filesuffix='_tbias')

utils.compile_run_output(gdirs, filesuffix='_defaults')
utils.compile_run_output(gdirs, filesuffix='_tbias')

# ds = xr.open_dataset(os.path.join(base_dir, 'run_output_defaults.nc'))
# (ds.volume.sum(dim='rgi_id') * 1e-9).plot()
# plt.show()
# exit()

# We prepare for the plot, which needs our own map to proceed.
# Lets do a local mercator grid
g = salem.mercator_grid(center_ll=(-19.61, 63.63),
                        extent=(18000, 14500))
# And a map accordingly
sm = salem.Map(g, countries=False)
sm.set_lonlat_contours(add_xtick_labels=False)
z = sm.set_topography('/home/mowglie/disk/OGGM_INPUT/tmp/ISL.tif')
sm.set_data(z)

# Figs
f = 0.9
f, axs = plt.subplots(2, 1, figsize=(7*f, 10*f))

graphics.plot_domain(gdirs, ax=axs[0], smap=sm)


sm.set_data()
sm.set_lonlat_contours()
sm.set_geometry()
sm.set_text()
graphics.plot_inversion(gdirs, ax=axs[1], smap=sm,
                        linewidth=1, add_scalebar=False,
                              title='', vmax=250)
plt.tight_layout()
plt.savefig(PLOT_DIR + 'iceland.pdf', dpi=150, bbox_inches='tight')
exit(0)

lw = 1
graphics.plot_modeloutput_map(gdirs, smap=sm, filesuffix='_tbias',
                              modelyr=0, linewidth=lw)
plt.savefig('/home/mowglie/yr000.png', dpi=150)
plt.figure()
sm.set_geometry()
graphics.plot_modeloutput_map(gdirs, smap=sm, filesuffix='_tbias',
                              modelyr=150, linewidth=lw)
plt.savefig('/home/mowglie/yr150.png', dpi=150)
plt.figure()
sm.set_geometry()
graphics.plot_modeloutput_map(gdirs, smap=sm, filesuffix='_tbias',
                              modelyr=300, linewidth=lw)
plt.savefig('/home/mowglie/yr300.png', dpi=150)
