import os
import geopandas as gpd
import numpy as np
import oggm
from oggm import cfg, tasks, graphics
from oggm.utils import get_demo_file
import matplotlib.pyplot as plt
import shapely.geometry as shpg
from oggm.sandbox.gmd_paper import PLOT_DIR
from oggm.core.preprocessing.climate import (t_star_from_refmb,
                                             local_mustar_apparent_mb)
from oggm.core.preprocessing.inversion import (mass_conservation_inversion)
from oggm.core.preprocessing.centerlines import (Centerline)
from oggm.core.models.flowline import (FluxBasedModel)
from oggm.core.models.massbalance import (LinearMassBalanceModel)

# test directory
base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'Inversions')

# Init
cfg.initialize()
cfg.set_divides_db()
entity = gpd.read_file(get_demo_file('Hintereisferner_RGI5.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir, reset=True)
tasks.define_glacier_region(gdir, entity=entity)

# Models
from oggm.tests.test_models import (dummy_constant_bed)

# Figure
f, axs = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=True)
axs = np.asarray(axs).flatten()

fls = dummy_constant_bed(map_dx=gdir.grid.dx)
mb = LinearMassBalanceModel(2600.)
model = FluxBasedModel(fls, mb_model=mb, y0=0.)
model.run_until_equilibrium()
fls = []
for fl in model.fls:
    pg = np.where(fl.thick > 0)
    line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
    flo = Centerline(line, dx=fl.dx, surface_h=fl.surface_h[pg])
    flo.widths = fl.widths[pg]
    flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
    fls.append(flo)
for did in [0, 1]:
    gdir.write_pickle(fls, 'inversion_flowlines', div_id=did)

tasks.apparent_mb_from_linear_mb(gdir)
tasks.prepare_for_inversion(gdir)
v, _ = mass_conservation_inversion(gdir)
inv = gdir.read_pickle('inversion_output', div_id=1)[-1]

tasks.apparent_mb_from_linear_mb(gdir, mb_gradient=3*5)
tasks.prepare_for_inversion(gdir)
v, _ = mass_conservation_inversion(gdir)
inv2 = gdir.read_pickle('inversion_output', div_id=1)[-1]

tasks.apparent_mb_from_linear_mb(gdir, mb_gradient=3/5)
tasks.prepare_for_inversion(gdir)
v, _ = mass_conservation_inversion(gdir)
inv3 = gdir.read_pickle('inversion_output', div_id=1)[-1]

# plot
ax = axs[0]
th1 = inv['thick']
th2 = inv2['thick']
th3 = inv3['thick']
pg = model.fls[-1].thick > 0
th = model.fls[-1].thick[pg]
ax.plot(th1, 'C0', label='Defaut grad')
ax.plot(th2, 'C1', label='5 grad')
ax.plot(th3, 'C2', label='1/5 grad')
ax.set_ylabel('Thickness [m]')
ax.legend(loc=3)


tasks.apparent_mb_from_linear_mb(gdir)
tasks.prepare_for_inversion(gdir)
v, _ = mass_conservation_inversion(gdir)
inv = gdir.read_pickle('inversion_output', div_id=1)[-1]

v, _ = mass_conservation_inversion(gdir, glen_a=cfg.A/5)
inv2 = gdir.read_pickle('inversion_output', div_id=1)[-1]

v, _ = mass_conservation_inversion(gdir, glen_a=cfg.A*5)
inv3 = gdir.read_pickle('inversion_output', div_id=1)[-1]

# plot
ax = axs[1]
th1 = inv['thick']
th2 = inv2['thick']
th3 = inv3['thick']
pg = model.fls[-1].thick > 0
th = model.fls[-1].thick[pg]
ax.plot(th1, 'C0', label='Default A')
ax.plot(th2, 'C1', label='5 A')
ax.plot(th3, 'C2', label='1/5 A')
ax.set_ylabel('Thickness [m]')
ax.legend(loc=3)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'inversions_sensi.pdf', dpi=150, bbox_inches='tight')
