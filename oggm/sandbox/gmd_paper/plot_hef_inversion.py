import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import oggm
from oggm import cfg, tasks
from oggm.core.climate import (t_star_from_refmb, local_mustar, apparent_mb)
from oggm.core.inversion import (mass_conservation_inversion)
from oggm.sandbox.gmd_paper import PLOT_DIR
from oggm.utils import get_demo_file, mkdir

cfg.initialize()
cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
cfg.PARAMS['border'] = 25
cfg.PARAMS['auto_skip_task'] = True
reset = False

base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'Invert_hef')
mkdir(base_dir, reset=reset)

entity = gpd.read_file(get_demo_file('Hintereisferner_RGI5.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir)

tasks.define_glacier_region(gdir, entity=entity)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.initialize_flowlines(gdir)
tasks.catchment_area(gdir)
tasks.catchment_intersections(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)
tasks.process_cru_data(gdir)
tasks.mu_candidates(gdir, reset=True)
res = t_star_from_refmb(gdir, gdir.get_ref_mb_data()['ANNUAL_BALANCE'])
local_mustar(gdir, tstar=res['t_star'][-1], bias=res['bias'][-1],
             prcp_fac=res['prcp_fac'], reset=True)
apparent_mb(gdir, reset=True)

tasks.prepare_for_inversion(gdir, reset=True)


f = 0.85
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8*f, 4*f), sharey=True)
cs = plt.get_cmap('Purples')(np.linspace(0.5, 1, 3)[::-1])
lw = 2
alpha = 0.8

facs = np.append((np.arange(9)+1)*0.1, (np.arange(19)+2) * 0.5)
vols1 = facs * 0.
vols2 = facs * 0.
vols3 = facs * 0.

for i, f in enumerate(facs):
    v, _ = mass_conservation_inversion(gdir, glen_a=cfg.A*f)
    vols1[i] = v * 1e-9
    v, _ = mass_conservation_inversion(gdir, glen_a=cfg.A*f,
                                       fs=cfg.FS*0.5)
    vols2[i] = v * 1e-9
    v, a = mass_conservation_inversion(gdir, glen_a=cfg.A*f,
                                       fs=cfg.FS)
    vols3[i] = v * 1e-9

v_vas = 0.034*((a*1e-6)**1.375)
v_fischer = 0.573

tx, ty = 0.02, .979
letkm = dict(color='black', ha='left', va='top', fontsize=12,
             bbox=dict(facecolor='white', edgecolor='black'))

ax1.plot(facs, vols1, label='No sliding', color=cs[0], linewidth=lw, alpha=alpha)
ax1.plot(facs, vols2, label='0.5 f$_s$', color=cs[1], linewidth=lw, alpha=alpha)
ax1.plot(facs, vols3, label='Default f$_s$', color=cs[2], linewidth=lw, alpha=alpha)
ax1.hlines(v_vas, facs[0], facs[-1], linestyles=':',
           label='VAS')
ax1.hlines(v_fischer, facs[0], facs[-1], linestyles='--',
           label='Fischer et al.')
ax1.set_xlabel("A factor")
ax1.set_ylabel("Total volume [km$^3$]")
ax1.set_ylim([0.45, 1.5])
ax1.text(tx, ty, 'a', transform=ax1.transAxes, **letkm)
ax1.legend(loc=1)

facs = np.arange(0.5, 5.01, 0.1)
vols1 = facs * 0.
vols2 = facs * 0.
vols3 = facs * 0.

for i, f in enumerate(facs):
    cfg.PARAMS['prcp_scaling_factor'] = f
    tasks.mu_candidates(gdir, reset=True)
    res = t_star_from_refmb(gdir, gdir.get_ref_mb_data()['ANNUAL_BALANCE'])
    local_mustar(gdir, tstar=res['t_star'][-1], bias=res['bias'][-1],
                 prcp_fac=res['prcp_fac'], reset=True)
    apparent_mb(gdir, reset=True)
    tasks.prepare_for_inversion(gdir, reset=True)

    v, _ = mass_conservation_inversion(gdir, glen_a=cfg.A)
    vols1[i] = v * 1e-9

    v, _ = mass_conservation_inversion(gdir, glen_a=cfg.A*2)
    vols2[i] = v * 1e-9

    v, _ = mass_conservation_inversion(gdir, glen_a=cfg.A*4)
    vols3[i] = v * 1e-9


ax2.plot(facs, vols1, label='Default A', color=cs[0], linewidth=lw, alpha=alpha)
ax2.plot(facs, vols2, label='2 A', color=cs[1], linewidth=lw, alpha=alpha)
ax2.plot(facs, vols3, label='4 A', color=cs[2], linewidth=lw, alpha=alpha)
ax2.hlines(v_vas, facs[0], facs[-1], linestyles=':')
ax2.hlines(v_fischer, facs[0], facs[-1], linestyles='--')
ax2.hlines(v_vas, facs[0], facs[-1], linestyles=':',
           label='VAS')
ax2.hlines(v_fischer, facs[0], facs[-1], linestyles='--',
           label='Fischer et al.')
ax2.set_xlabel("Precipitation factor")
ax2.text(tx, ty, 'b', transform=ax2.transAxes, **letkm)
ax2.legend(loc=1)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'hef_inv.pdf', dpi=150, bbox_inches='tight')
