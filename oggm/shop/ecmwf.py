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
from oggm.exceptions import InvalidParamsError

# Module logger
log = logging.getLogger(__name__)

ECMWF_SERVER = 'https://cluster.klima.uni-bremen.de/~oggm/climate/'

BASENAMES = {
    'ERA5': {
        'inv': 'era5/monthly/v1.1/era5_invariant.nc',
        'pre': 'era5/monthly/v1.1/era5_monthly_prcp_1979-2019.nc',
        'tmp': 'era5/monthly/v1.1/era5_monthly_t2m_1979-2019.nc'
    },
    'ERA5L': {
        'inv': 'era5-land/monthly/v1.0/era5_land_invariant_flat.nc',
        'pre': 'era5-land/monthly/v1.0/era5_land_monthly_prcp_1981-2018_flat'
               '.nc',
        'tmp': 'era5-land/monthly/v1.0/era5_land_monthly_t2m_1981-2018_flat.nc'
    },
    'ERA5L-HMA': {
        'inv': 'era5-land/monthly/vhma/era5_land_invariant_flat_HMA.nc',
        'pre': 'era5-land/monthly/vhma/era5_land_monthly_prcp_1950-2020_flat_HMA.nc',
        'tmp': 'era5-land/monthly/vhma/era5_land_monthly_t2m_1950-2020_flat_HMA.nc'
    },
    'CERA': {
        'inv': 'cera-20c/monthly/v1.0/cera-20c_invariant.nc',
        'pre': 'cera-20c/monthly/v1.0/cera-20c_pcp_1901-2010.nc',
        'tmp': 'cera-20c/monthly/v1.0/cera-20c_t2m_1901-2010.nc'
    },
    'ERA5dr': {
        'inv': 'era5/monthly/vdr/ERA5_geopotential_monthly.nc',
        'lapserates': 'era5/monthly/vdr/ERA5_lapserates_monthly.nc',
        'tmp': 'era5/monthly/vdr/ERA5_temp_monthly.nc',
        'tempstd': 'era5/monthly/vdr/ERA5_tempstd_monthly.nc',
        'pre': 'era5/monthly/vdr/ERA5_totalprecip_monthly.nc',
    }
}


def set_ecmwf_url(url):
    """If you want to use a different server for ECMWF (for testing, etc)."""
    global ECMWF_SERVER
    ECMWF_SERVER = url


@utils.locked_func
def get_ecmwf_file(dataset='ERA5', var=None):
    """Returns a path to the desired ECMWF baseline climate file.

    If the file is not present, download it.

    Parameters
    ----------
    dataset : str
        'ERA5', 'ERA5L', 'CERA', 'ERA5L-HMA', 'ERA5dr'
    var : str
        'inv' for invariant
        'tmp' for temperature
        'pre' for precipitation

    Returns
    -------
    str
        path to the file
    """

    # Be sure input makes sense
    if dataset not in BASENAMES.keys():
        raise InvalidParamsError('ECMWF dataset {} not '
                                 'in {}'.format(dataset, BASENAMES.keys()))
    if var not in BASENAMES[dataset].keys():
        raise InvalidParamsError('ECMWF variable {} not '
                                 'in {}'.format(var,
                                                BASENAMES[dataset].keys()))

    # File to look for
    return utils.file_downloader(ECMWF_SERVER + BASENAMES[dataset][var])


def _check_ds_validity(ds):
    if 'time' in ds.variables and np.any(ds['time.day'] != 1):
        # Mid-month timestamps need to be corrected
        ds['time'].data[:] = pd.to_datetime({'year': ds['time.year'],
                                             'month': ds['time.month'],
                                             'day': 1})
    assert ds.longitude.min() >= 0


@entity_task(log, writes=['climate_historical'])
def process_ecmwf_data(gdir, settings_filesuffix='', dataset=None, ensemble_member=0,
                       y0=None, y1=None, output_filesuffix=None):
    """Processes and writes the ECMWF baseline climate data for this glacier.

    Extracts the nearest timeseries and writes everything to a NetCDF file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    dataset : str
        'ERA5', 'ERA5L', 'CERA', 'ERA5L-HMA', 'ERA5dr'.
        Defaults to cfg.PARAMS['baseline_climate']
    ensemble_member : int
        for CERA, pick an ensemble member number (0-9). We might make this
        more of a clever pick later.
    y0 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    y1 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    """

    if dataset is None:
        dataset = gdir.settings['baseline_climate']

    # Use xarray to read the data
    lon = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
    lat = gdir.cenlat
    with xr.open_dataset(get_ecmwf_file(dataset, 'tmp')) as ds:
        _check_ds_validity(ds)
        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        if dataset == 'ERA5dr':
            # Last year incomplete
            assert ds['time.month'][-1] == 5
            y1 -= 1
        ds = ds.sel(time=slice(f'{y0}-01-01', f'{y1}-12-01'))
        if dataset == 'CERA':
            ds = ds.sel(number=ensemble_member)
        try:
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        except (ValueError, KeyError):
            # Flattened ERA5
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=np.argmin(c.data))
        temp = ds['t2m'].data - 273.15
        time = ds.time.data
        ref_lon = float(ds['longitude'])
        ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon
        ref_lat = float(ds['latitude'])
    with xr.open_dataset(get_ecmwf_file(dataset, 'pre')) as ds:
        _check_ds_validity(ds)
        ds = ds.sel(time=slice(f'{y0}-01-01', f'{y1}-12-01'))
        if dataset == 'CERA':
            ds = ds.sel(number=ensemble_member)
        try:
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        except (ValueError, KeyError):
            # Flattened ERA5
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=np.argmin(c.data))
        prcp = ds['tp'].data * 1000 * ds['time.daysinmonth']
    with xr.open_dataset(get_ecmwf_file(dataset, 'inv')) as ds:
        _check_ds_validity(ds)
        ds = ds.isel(time=0)
        try:
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        except (ValueError, KeyError):
            # Flattened ERA5
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=np.argmin(c.data))
        hgt = ds['z'].data / cfg.G

    temp_std = None

    if dataset == 'ERA5dr':
        with xr.open_dataset(get_ecmwf_file(dataset, 'lapserates')) as ds:
            _check_ds_validity(ds)
            ds = ds.sel(time=slice(f'{y0}-01-01', f'{y1}-12-01'))
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')

        with xr.open_dataset(get_ecmwf_file(dataset, 'tempstd')) as ds:
            _check_ds_validity(ds)
            ds = ds.sel(time=slice(f'{y0}-01-01', f'{y1}-12-01'))
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
            temp_std = ds['t2m_std'].data

    # OK, ready to write
    gdir.write_climate_file(time, prcp, temp, hgt, ref_lon, ref_lat,
                            filesuffix=output_filesuffix,
                            temp_std=temp_std,
                            source=dataset)
