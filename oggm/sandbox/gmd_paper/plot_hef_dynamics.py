import os
import geopandas as gpd
import numpy as np
import oggm
from oggm import cfg, tasks, graphics
from oggm.utils import get_demo_file
import matplotlib.pyplot as plt
import xarray as xr
import scipy.optimize as optimization
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
tasks.mu_candidates(gdir, div_id=0)
tasks.compute_ref_t_stars([gdir])
tasks.distribute_t_stars([gdir])
tasks.prepare_for_inversion(gdir)
tasks.volume_inversion(gdir, use_cfg_params={'glen_a':cfg.A, 'fs':0})
tasks.filter_inversion_output(gdir)
tasks.init_present_time_glacier(gdir)


reset = False
tasks.random_glacier_evolution(gdir, nyears=500, bias=0, seed=0,
                               filesuffix='seed0', reset=reset)

tasks.random_glacier_evolution(gdir, nyears=500, bias=0, seed=1,
                               filesuffix='seed1', reset=reset)

tasks.random_glacier_evolution(gdir, nyears=500, bias=0, seed=2,
                               filesuffix='seed2', reset=reset)

f = gdir.get_filepath('model_diagnostics', filesuffix='seed0')
ds1 = xr.open_dataset(f)
f = gdir.get_filepath('model_diagnostics', filesuffix='seed1')
ds2 = xr.open_dataset(f)
f = gdir.get_filepath('model_diagnostics', filesuffix='seed2')
ds3 = xr.open_dataset(f)

# ds1.volume_m3.plot()
# ds2.volume_m3.plot()
# ds3.volume_m3.plot()
# plt.show()
#
# ds1.area_m2.plot()
# ds2.area_m2.plot()
# ds3.area_m2.plot()
# plt.show()

ds1.length_m.rolling(time=5, center=True).mean().plot()
ds2.length_m.rolling(time=5, center=True).mean().plot()
ds3.length_m.rolling(time=5, center=True).mean().plot()
plt.show()

