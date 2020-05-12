import logging
import warnings

# External libs
import xarray as xr

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

BASENAMES = {'ERA5':{'inv':'era5/monthly/v1.0/era5_invariant.nc',
                     'pre':'era5/monthly/v1.0/era5_monthly_prcp_1979-2018.nc',
                     'tmp':'era5/monthly/v1.0/era5_monthly_t2m_1979-2018.nc'},
             'ERA5L':{'inv':'era5-land/monthly/v1.0/era5_land_invariant.nc',
                      'pre':'era5-land/monthly/v1.0/era5_land_monthly_prcp_'
                            '1981-2018.nc',
                      'tmp':'era5-land/monthly/v1.0/era5_land_monthly_t2m_'
                            '1981-2018.nc'},
             'CERA':{'inv':'cera-20c/monthly/v1.0/cera-20c_invariant.nc',
                     'pre':'cera-20c/monthly/v1.0/cera-20c_pcp_1901-2010.nc',
                     'tmp':'cera-20c/monthly/v1.0/cera-20c_t2m_1901-2010.nc'}
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
        'ERA5', 'ERA5L', 'CERA'
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
    if dataset not in ['ERA5', 'ERA5L', 'CERA']:
        raise InvalidParamsError('ECMWF dataset {} '
                                 'does not exist!'.format(dataset))
    if var not in ['inv', 'tmp', 'pre']:
        raise InvalidParamsError('ECMWF variable {} '
                                 'does not exist!'.format(var))

    # File to look for
    return utils.file_downloader(ECMWF_SERVER + BASENAMES[dataset][var])


@entity_task(log, writes=['climate_historical', 'climate_info'])
def process_ecmwf_data(gdir, dataset=None, ensemble_member=0,
                       y0=None, y1=None, output_filesuffix=None):
    """Processes and writes the ECMWF baseline climate data for this glacier.

    Extracts the nearest timeseries and writes everything to a NetCDF file.

    Parameters
    ----------
    dataset : str
        'ERA5', 'ERA5L', 'CERA'. Defaults to cfg.PARAMS['baseline_climate']
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

    if cfg.PATHS.get('climate_file', None):
        warnings.warn("You seem to have set a custom climate file for this "
                      "run, but are using the ECMWF climate file "
                      "instead.")

    if dataset is None:
        dataset = cfg.PARAMS['baseline_climate']

    if dataset not in ['ERA5', 'ERA5L', 'CERA']:
        raise InvalidParamsError("cfg.PARAMS['baseline_climate'] should be "
                                 "set to 'ERA5', 'ERA5L' or 'CERA'.")

    # Use xarray to read the data
    lon = gdir.cenlon + 180
    lat = gdir.cenlat
    with xr.open_dataset(get_ecmwf_file(dataset, 'tmp')) as ds:
        assert ds.longitude.min() > 0
        # set temporal subset for the ts data (hydro years)
        sm = cfg.PARAMS['hydro_month_nh']
        em = sm - 1 if (sm > 1) else 12
        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-01'.format(y1, em)))
        if dataset == 'CERA':
            ds = ds.sel(number=ensemble_member)
        ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        temp = ds['t2m'].data - 273.15
        time = ds.time.data
        ref_lon = float(ds['longitude'])
        ref_lat = float(ds['latitude'])
    with xr.open_dataset(get_ecmwf_file(dataset, 'pre')) as ds:
        assert ds.longitude.min() > 0
        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-01'.format(y1, em)))
        if dataset == 'CERA':
            ds = ds.sel(number=ensemble_member)
        ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        prcp = ds['tp'].data * 1000 * 24
    with xr.open_dataset(get_ecmwf_file(dataset, 'inv')) as ds:
        assert ds.longitude.min() > 0
        ds = ds.isel(time=0)
        ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        hgt = ds['z'].data / cfg.G

    # Should we compute the gradient?
    if cfg.PARAMS['temp_use_local_gradient']:
        raise NotImplementedError()

    gdir.write_monthly_climate_file(time, prcp, temp, hgt, ref_lon, ref_lat,
                                    time_unit='days since 1801-01-01 00:00:00',
                                    filesuffix=output_filesuffix)
    # metadata
    out = {'baseline_climate_source': dataset,
           'baseline_hydro_yr_0': y0 + 1,
           'baseline_hydro_yr_1': y1}
    gdir.write_json(out, 'climate_info')
