# Imports
import os
import xarray as xr
import matplotlib.pyplot as plt
from oggm.utils import get_demo_file, gettempdir

# Local working directory (where OGGM wrote its output)
WORKING_DIR = gettempdir('OGGM_Rofental')

# Read the files using xarray
ds = xr.open_dataset(os.path.join(WORKING_DIR, 'run_output_commitment.nc'))
dsp = xr.open_dataset(os.path.join(WORKING_DIR, 'run_output_bias_p.nc'))
dsm = xr.open_dataset(os.path.join(WORKING_DIR, 'run_output_bias_m.nc'))

# Compute and plot the regional sums
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
# Volume
(ds.volume.sum(dim='rgi_id') * 1e-9).plot(ax=ax1, label='[1985-2015]')
(dsp.volume.sum(dim='rgi_id') * 1e-9).plot(ax=ax1, label='+0.5°C')
(dsm.volume.sum(dim='rgi_id') * 1e-9).plot(ax=ax1, label='-0.5°C')
ax1.legend(loc='best')
# Area
(ds.area.sum(dim='rgi_id') * 1e-6).plot(ax=ax2, label='[1985-2015]')
(dsp.area.sum(dim='rgi_id') * 1e-6).plot(ax=ax2, label='+0.5°C')
(dsm.area.sum(dim='rgi_id') * 1e-6).plot(ax=ax2, label='-0.5°C')
plt.tight_layout()

# Pick a specific glacier (Hintereisferner)
rid = 'RGI60-11.00897'

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
# Volume
(ds.volume.sel(rgi_id=rid) * 1e-9).plot(ax=ax1, label='[1985-2015]')
(dsp.volume.sel(rgi_id=rid) * 1e-9).plot(ax=ax1, label='+0.5°C')
(dsm.volume.sel(rgi_id=rid) * 1e-9).plot(ax=ax1, label='-0.5°C')
ax1.legend(loc='best')
# Length
(ds.length.sel(rgi_id=rid) * 1e-3).plot(ax=ax2, label='[1985-2015]')
(dsp.length.sel(rgi_id=rid) * 1e-3).plot(ax=ax2, label='+0.5°C')
(dsm.length.sel(rgi_id=rid) * 1e-3).plot(ax=ax2, label='-0.5°C')
plt.tight_layout()
plt.show()
