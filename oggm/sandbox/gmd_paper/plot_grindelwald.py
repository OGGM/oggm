import os
import zipfile
import geopandas as gpd
import numpy as np
import salem
import netCDF4
import oggm
from oggm import cfg, tasks, graphics
from oggm.utils import get_demo_file, file_downloader, nicenumber
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from oggm.sandbox.gmd_paper import PLOT_DIR
from oggm.core.preprocessing.climate import (t_star_from_refmb,
                                             local_mustar_apparent_mb)
from oggm.core.preprocessing.inversion import (mass_conservation_inversion)
from oggm.core.models.flowline import (FluxBasedModel)
from oggm.core.models.massbalance import (RandomMassBalanceModel)

cfg.initialize()
cfg.PARAMS['border'] = 10

base_dir = os.path.join(os.path.expanduser('~/tmp'), 'OGGM_GMD', 'Grindelwald')

rgif = 'https://dl.dropboxusercontent.com/u/20930277/rgiv5_grindelwald.zip'
rgif = file_downloader(rgif)
with zipfile.ZipFile(rgif) as zf:
    zf.extractall(base_dir)
rgif = os.path.join(base_dir, 'rgiv5_grindelwald.shp')
rgidf = salem.read_shapefile(rgif, cached=True)
entity = rgidf.iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir, reset=True)

tasks.define_glacier_region(gdir, entity=entity)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.compute_downstream_lines(gdir)
tasks.initialize_flowlines(gdir)
tasks.catchment_area(gdir)
tasks.catchment_intersections(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)

with netCDF4.Dataset(gdir.get_filepath('gridded_data', div_id=0)) as nc:
    mask = nc.variables['glacier_mask'][:]
    topo = nc.variables['topo_smoothed'][:]
rhgt = topo[np.where(mask)][:]
hgt, harea = gdir.get_inversion_flowline_hw()

# Check for area distrib
bins = np.arange(nicenumber(np.min(hgt), 150, lower=True),
                 nicenumber(np.max(hgt), 150) + 1,
                 150.)
h1, b = np.histogram(hgt, weights=harea, density=True, bins=bins)
h2, b = np.histogram(rhgt, density=True, bins=bins)

h1 = h1 / np.sum(h1)
h2 = h2 / np.sum(h2)

f, axs = plt.subplots(2, 2, figsize=(9, 8))
axs = np.asarray(axs).flatten()

llkw = {'interval': 0}
letkm = dict(color='black', ha='right', va='top', fontsize=18,
             bbox=dict(facecolor='white', edgecolor='black'))
xt, yt = 109.2, 1.75

im = graphics.plot_catchment_areas(gdir, ax=axs[0], title='',
                                   lonlat_contours_kwargs=llkw,
                                   add_scalebar=True)

axs[0].text(xt, yt, 'a', **letkm)

graphics.plot_catchment_width(gdir, ax=axs[1], title='', add_colorbar=False,
                              lonlat_contours_kwargs=llkw,
                              add_scalebar=False)
axs[1].text(xt, yt, 'b', **letkm)

graphics.plot_catchment_width(gdir, ax=axs[2], title='', corrected=True,
                              add_colorbar=False,
                              lonlat_contours_kwargs=llkw,
                              add_scalebar=False, add_touches=True)
axs[2].text(xt, yt, 'c', **letkm)

width = 0.6 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
axs[3].bar(center, h2, align='center', width=width, alpha=0.5, color='C0', label='SRTM')
axs[3].bar(center, h1, align='center', width=width, alpha=0.5, color='C3', label='OGGM')
axs[3].set_xlabel('Altitude (m)')
plt.legend(loc='best')
axs[3].text(3780, 0.224, 'd', **letkm)

dy = np.abs(np.diff(axs[3].get_ylim()))
dx = np.abs(np.diff(axs[3].get_xlim()))
aspect0 = topo.shape[0] / topo.shape[1]
aspect = aspect0 / (float(dy) / dx)
axs[3].set_aspect(aspect)

# plt.tight_layout()
plt.savefig(PLOT_DIR + 'grindelwald.pdf', dpi=150, bbox_inches='tight')
