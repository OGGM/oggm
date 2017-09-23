import xarray as xr
import matplotlib.pyplot as plt
from os import path

WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_precalibrated_run')
ds1 = xr.open_dataset(path.join(WORKING_DIR, 'run_output_tstar.nc'))
ds2 = xr.open_dataset(path.join(WORKING_DIR, 'run_output_commitment.nc'))

v1_km3 = ds1.volume * 1e-9
v2_km3 = ds2.volume * 1e-9
l1_km = ds1.length * 1e-3
l2_km = ds2.length * 1e-3

f, axs = plt.subplots(2, 4, figsize=(12, 4), sharex=True)

for i in range(4):
    ax = axs[0, i]
    v1_km3.isel(rgi_id=i).plot(ax=ax, label='t*')
    v2_km3.isel(rgi_id=i).plot(ax=ax, label='Commitment')
    if i == 0:
        ax.set_ylabel('Volume [km3]')
        ax.legend(loc='best')
    else:
        ax.set_ylabel('')
    ax.set_xlabel('')

    ax = axs[1, i]

    # Length can need a bit of postprocessing because of some cold years
    # Where seasonal snow is thought to be a glacier...
    for l in [l1_km, l2_km]:
        roll_yrs = 5
        sel = l.isel(rgi_id=i).to_series()
        # Take the minimum out of 5 years
        sel = sel.rolling(roll_yrs).min()
        sel.iloc[0:roll_yrs] = sel.iloc[roll_yrs]
        sel.plot(ax=ax)
    if i == 0:
        ax.set_ylabel('Length [m]')
    else:
        ax.set_ylabel('')
    ax.set_xlabel('Years')
    ax.set_title('')

plt.tight_layout()
plt.show()
