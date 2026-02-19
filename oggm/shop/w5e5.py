import logging

# External libs
import numpy as np
import xarray as xr
import pandas as pd

# Optional libs
try:
    import salem
except ImportError:
    pass

# Locals
from oggm import cfg
from oggm import utils
from oggm import entity_task
from oggm.shop import ecmwf
from oggm.exceptions import InvalidParamsError

# Module logger
log = logging.getLogger(__name__)

GSWP3_W5E5_SERVER = 'https://cluster.klima.uni-bremen.de/~oggm/climate/'

_base = 'gswp3-w5e5/flattened/2025.11.25/monthly/'

BASENAMES = {
    'GSWP3_W5E5': {
        'inv': f'{_base}gswp3-w5e5_glacier_invariant_flat_v2025.11.25.nc',
        'tmp': f'{_base}gswp3-w5e5_obsclim_tas_global_monthly_1901_2019_flat_glaciers_v2025.11.25.nc',
        'temp_std': f'{_base}gswp3-w5e5_obsclim_temp_std_global_monthly_1901_2019_flat_glaciers_v2025.11.25.nc',
        'prcp': f'{_base}gswp3-w5e5_obsclim_pr_global_monthly_1901_2019_flat_glaciers_v2025.11.25.nc'
    }
}


def get_gswp3_w5e5_file(dataset='GSWP3_W5E5', var=None):
    """Returns the path to the desired GSWP3-W5E5 baseline climate file.

    It is the observed climate dataset used for ISIMIP3a.
    For OGGM, it was preprocessed by selecting only those gridpoints
    with glaciers nearby.

    If the file is not present, downloads it.

    var : str
        'inv' for invariant
        'tmp' for temperature
        'prcp' for precipitation
        'temp_std' for mean of daily temperature standard deviation
    dataset : str
        default option in OGGM is 'GSWP3_W5E5' (could also just use 'W5E5_monthly',
        but this is the same for a shorter time period, and we only use it for testing)
    """
    # check if input makes sense
    if var not in BASENAMES[dataset].keys():
        raise InvalidParamsError('GSWP3-W5E5 variable {} not '
                                 'in {}'.format(var,
                                                BASENAMES[dataset].keys()))

    # File to look for
    return utils.file_downloader(GSWP3_W5E5_SERVER + BASENAMES[dataset][var])


@entity_task(log, writes=['climate_historical'])
def process_gswp3_w5e5_data(gdir, y0=None, y1=None, output_filesuffix=None):
    """Processes and writes the GSWP3-W5E5+W5E5 baseline climate data for a glacier.

    The data is the same as W5E5 after 79, and is GSWP3 before that.

    Data source: https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/80/
    If y0>=1979, the temperature lapse rate gradient from ERA5dr is added.

    Extracts the nearest timeseries and writes everything to a NetCDF file.

    Parameters
    ----------
    y0 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data). If y0>=1979,
        it only uses W5E5 data!
    y1 : int
        the end year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    output_filesuffix : str, optional
         None by default
    """
    dataset = 'GSWP3_W5E5'  # 'W5E5_monthly'
    tvar = 'tas'
    pvar = 'pr'

    # get the central longitude/latitudes of the glacier
    lon = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
    lat = gdir.cenlat

    path_tmp = get_gswp3_w5e5_file(dataset, 'tmp')
    path_prcp = get_gswp3_w5e5_file(dataset, 'prcp')
    path_inv = get_gswp3_w5e5_file(dataset, 'inv')

    # Use xarray to read the data
    # would go faster with only netCDF -.-, but easier with xarray
    # first temperature dataset
    with xr.open_dataset(path_tmp) as ds:
        # sanity checks, safeguarding for unwanted future side effects
        # we do it only for temp and assume no mistake on the data prep side
        info = utils.climate_file_info(ds)
        assert info['lon_bounds'][0] >= 0
        assert info['is_flat']

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1

        if y1 > 2019 or y0 < 1901:
            text = 'The climate files only go from 1901--2019'
            raise InvalidParamsError(text)
        ds = ds.sel(time=slice(f'{y0}-01-01', f'{y1}-12-01'))

        # computing all the distances and choose the nearest gridpoint
        c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
        ds = ds.isel(points=np.argmin(c.data))

        # Feth lon and lat
        ds['longitude'] = ds.longitude
        ds['latitude'] = ds.latitude

        # temperature should be in degree Celsius for the glacier climate files
        temp = ds[tvar].data - 273.15
        time = ds.time.data

        ref_lon = float(ds['longitude'])
        ref_lat = float(ds['latitude'])

        ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon

    # precipitation: similar as temperature
    with xr.open_dataset(path_prcp) as ds:

        # here we take the same y0 and y1 as given from the
        # tmp dataset
        ds = ds.sel(time=slice(f'{y0}-01-01', f'{y1}-12-01'))

        c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
        ds = ds.isel(points=np.argmin(c.data))

        # convert kg m-2 s-1 monthly mean into monthly sum!!!
        prcp = ds[pvar].data*cfg.SEC_IN_DAY*ds['time.daysinmonth']

    # w5e5 invariant file
    with xr.open_dataset(path_inv) as ds:
        ds = ds.isel(time=0)

        c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
        ds = ds.isel(points=np.argmin(c.data))

        # w5e5 inv ASurf/hgt is already in hgt coordinates
        hgt = ds['ASurf'].data

    path_temp_std = get_gswp3_w5e5_file(dataset, 'temp_std')
    with xr.open_dataset(path_temp_std) as ds:
        ds = ds.sel(time=slice(f'{y0}-01-01', f'{y1}-12-01'))
        c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
        ds = ds.isel(points=np.argmin(c.data))
        temp_std = ds['temp_std'].data

    # OK, ready to write
    gdir.write_monthly_climate_file(time, prcp, temp, hgt, ref_lon, ref_lat,
                                    filesuffix=output_filesuffix,
                                    temp_std=temp_std,
                                    source=dataset)


def process_w5e5_data(gdir, y0=None, y1=None, output_filesuffix=None):
    """Processes and writes the W5E5 baseline climate data for a glacier.

    Internally, this is actually calling process_gswp3_w5e5_data but only for
    the W5E5 part.

    data source: https://data.isimip.org/10.48364/ISIMIP.342217
    The temperature lapse rate gradient from ERA5dr is added.
    Same as process_gswp3_w5e5_data, except that y0 is set to 1979
    and can not be set to lower values.

    Extracts the nearest timeseries and writes everything to a NetCDF file.

    Parameters
    ----------
    y0 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file (1979-2019), but with this kwarg
        you can shorten it (to save space or to crop bad data).
    y1 : int
        the end year of the timeseries to write. The default is to take
        the entire time period available in the file (1979-2019), but with this kwarg
        you can shorten it (to save space or to crop bad data)
    output_filesuffix : str, optional
         None by default
    """

    y0 = 1979 if y0 is None else y0
    y1 = 2019 if y1 is None else y1

    if y0 < 1979:
        text = ('The W5E5 climate only goes from 1979-2019,'
                'if you want older climate data,'
                'use instead "process_gswp3_w5e5_data()"')
        raise InvalidParamsError(text)
    process_gswp3_w5e5_data(gdir, y0=y0, y1=y1,
                            output_filesuffix=output_filesuffix)
