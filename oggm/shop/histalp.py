import logging
import warnings

# External libs
import numpy as np
import pandas as pd
from scipy import stats

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

HISTALP_SERVER = 'http://www.zamg.ac.at/histalp/download/grid5m/'


def set_histalp_url(url):
    """If you want to use a different server for HISTALP (for testing, etc)."""
    global HISTALP_SERVER
    HISTALP_SERVER = url


@utils.locked_func
def get_histalp_file(var=None):
    """Returns a path to the desired HISTALP baseline climate file.

    If the file is not present, download it.

    Parameters
    ----------
    var : str
        'tmp' for temperature
        'pre' for precipitation

    Returns
    -------
    str
        path to the file
    """

    # Be sure input makes sense
    if var not in ['tmp', 'pre']:
        raise InvalidParamsError('HISTALP variable {} '
                                 'does not exist!'.format(var))

    # File to look for
    if var == 'tmp':
        bname = 'HISTALP_temperature_1780-2014.nc'
    else:
        bname = 'HISTALP_precipitation_all_abs_1801-2014.nc'

    h_url = HISTALP_SERVER + bname + '.bz2'
    return utils.file_extractor(utils.file_downloader(h_url))


@entity_task(log, writes=['climate_historical'])
def process_histalp_data(gdir, y0=None, y1=None, output_filesuffix=None):
    """Processes and writes the HISTALP baseline climate data for this glacier.

    Extracts the nearest timeseries and writes everything to a NetCDF file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    y0 : int
        the starting year of the timeseries to write. The default is to take
        1850 (because the data is quite bad before that)
    y1 : int
        the ending year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    output_filesuffix : str
        this adds a suffix to the output file (useful to avoid overwriting
        previous experiments)
    """

    if cfg.PARAMS['baseline_climate'] != 'HISTALP':
        raise InvalidParamsError("cfg.PARAMS['baseline_climate'] should be "
                                 "set to HISTALP.")

    # read the time out of the pure netcdf file
    ft = get_histalp_file('tmp')
    fp = get_histalp_file('pre')
    with utils.ncDataset(ft) as nc:
        vt = nc.variables['time']
        assert vt[0] == 0
        assert vt[-1] == vt.shape[0] - 1
        t0 = vt.units.split(' since ')[1][:7]
        time_t = pd.date_range(start=t0, periods=vt.shape[0], freq='MS')
    with utils.ncDataset(fp) as nc:
        vt = nc.variables['time']
        assert vt[0] == 0.5
        assert vt[-1] == vt.shape[0] - .5
        t0 = vt.units.split(' since ')[1][:7]
        time_p = pd.date_range(start=t0, periods=vt.shape[0], freq='MS')

    # Now open with salem
    nc_ts_tmp = salem.GeoNetcdf(ft, time=time_t)
    nc_ts_pre = salem.GeoNetcdf(fp, time=time_p)

    # Some default
    if y0 is None:
        y0 = 1850

    # set temporal subset for the ts data (hydro years)
    # the reference time is given by precip, which is shorter
    yrs = nc_ts_pre.time.year
    y0 = yrs[0] if y0 is None else y0
    y1 = yrs[-1] if y1 is None else y1

    nc_ts_tmp.set_period(t0=f'{y0}-01-01', t1=f'{y1}-12-01')
    nc_ts_pre.set_period(t0=f'{y0}-01-01', t1=f'{y1}-12-01')
    time = nc_ts_pre.time
    ny, r = divmod(len(time), 12)
    assert r == 0

    # Units
    assert nc_ts_tmp._nc.variables['HSURF'].units.lower() in ['m', 'meters',
                                                              'meter',
                                                              'metres',
                                                              'metre']
    assert nc_ts_tmp._nc.variables['T_2M'].units.lower() in ['degc', 'degrees',
                                                             'degrees celcius',
                                                             'degree', 'c']
    assert nc_ts_pre._nc.variables['TOT_PREC'].units.lower() in ['kg m-2',
                                                                 'l m-2', 'mm',
                                                                 'millimeters',
                                                                 'millimeter']

    # geoloc
    lon = gdir.cenlon
    lat = gdir.cenlat
    nc_ts_tmp.set_subset(corners=((lon, lat), (lon, lat)), margin=1)
    nc_ts_pre.set_subset(corners=((lon, lat), (lon, lat)), margin=1)

    # read the data
    temp = nc_ts_tmp.get_vardata('T_2M')
    prcp = nc_ts_pre.get_vardata('TOT_PREC')
    hgt = nc_ts_tmp.get_vardata('HSURF')
    ref_lon = nc_ts_tmp.get_vardata('lon')
    ref_lat = nc_ts_tmp.get_vardata('lat')
    source = nc_ts_tmp._nc.title[:7]
    nc_ts_tmp._nc.close()
    nc_ts_pre._nc.close()

    gdir.write_monthly_climate_file(time, prcp[:, 1, 1], temp[:, 1, 1],
                                    hgt[1, 1], ref_lon[1], ref_lat[1],
                                    filesuffix=output_filesuffix,
                                    source=source)
