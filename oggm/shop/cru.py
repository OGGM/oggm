import logging
import warnings

# External libs
import numpy as np
import pandas as pd
import xarray as xr
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
from oggm.exceptions import MassBalanceCalibrationError, InvalidParamsError

# Module logger
log = logging.getLogger(__name__)

CRU_SERVER = ('https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.04/'
              'cruts.2004151855.v4.04/')

CRU_BASE = 'cru_ts4.04.1901.2019.{}.dat.nc'

CRU_CL = ('https://cluster.klima.uni-bremen.de/~oggm/climate/cru/'
          'cru_cl2.nc.zip')


def set_cru_url(url):
    """If you want to use a different server for CRU (for testing, etc)."""
    global CRU_SERVER
    CRU_SERVER = url


@utils.locked_func
def get_cru_cl_file():
    """Returns the path to the unpacked CRU CL file."""
    return utils.file_extractor(utils.file_downloader(CRU_CL))


@utils.locked_func
def get_cru_file(var=None):
    """Returns a path to the desired CRU baseline climate file.

    If the file is not present, download it.

    Parameters
    ----------
    var : str
        'tmp' for temperature
        'pre' for precipitation

    Returns
    -------
    str
        path to the CRU file
    """

    # Be sure input makes sense
    if var not in ['tmp', 'pre']:
        raise InvalidParamsError('CRU variable {} does not exist!'.format(var))

    # Download
    cru_filename = CRU_BASE.format(var)
    cru_url = CRU_SERVER + '{}/'.format(var) + cru_filename + '.gz'
    return utils.file_extractor(utils.file_downloader(cru_url))


@entity_task(log, writes=['climate_historical'])
def process_cru_data(gdir, tmp_file=None, pre_file=None, y0=None, y1=None,
                     output_filesuffix=None):
    """Processes and writes the CRU baseline climate data for this glacier.

    Interpolates the CRU TS data to the high-resolution CL2 climatologies
    (provided with OGGM) and writes everything to a NetCDF file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    tmp_file : str
        path to the CRU temperature file (defaults to the current OGGM chosen
        CRU version)
    pre_file : str
        path to the CRU precip file (defaults to the current OGGM chosen
        CRU version)
    y0 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    y1 : int
        the end year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    """

    if cfg.PARAMS['baseline_climate'] != 'CRU':
        raise InvalidParamsError("cfg.PARAMS['baseline_climate'] should be "
                                 "set to CRU")

    # read the climatology
    ncclim = salem.GeoNetcdf(get_cru_cl_file())

    # and the TS data
    if tmp_file is None:
        tmp_file = get_cru_file('tmp')
    if pre_file is None:
        pre_file = get_cru_file('pre')
    nc_ts_tmp = salem.GeoNetcdf(tmp_file, monthbegin=True)
    nc_ts_pre = salem.GeoNetcdf(pre_file, monthbegin=True)

    # set temporal subset for the ts data (hydro years)
    yrs = nc_ts_pre.time.year
    y0 = yrs[0] if y0 is None else y0
    y1 = yrs[-1] if y1 is None else y1

    nc_ts_tmp.set_period(t0=f'{y0}-01-01', t1=f'{y1}-12-01')
    nc_ts_pre.set_period(t0=f'{y0}-01-01', t1=f'{y1}-12-01')
    time = nc_ts_pre.time
    ny, r = divmod(len(time), 12)
    assert r == 0

    lon = gdir.cenlon
    lat = gdir.cenlat

    # This is guaranteed to work because I prepared the file (I hope)
    ncclim.set_subset(corners=((lon, lat), (lon, lat)), margin=1)

    # get climatology data
    loc_hgt = ncclim.get_vardata('elev')
    loc_tmp = ncclim.get_vardata('temp')
    loc_pre = ncclim.get_vardata('prcp')
    loc_lon = ncclim.get_vardata('lon')
    loc_lat = ncclim.get_vardata('lat')

    # see if the center is ok
    if not np.isfinite(loc_hgt[1, 1]):
        # take another candidate where finite
        isok = np.isfinite(loc_hgt)

        # wait: some areas are entirely NaNs, make the subset larger
        _margin = 1
        while not np.any(isok):
            _margin += 1
            ncclim.set_subset(corners=((lon, lat), (lon, lat)), margin=_margin)
            loc_hgt = ncclim.get_vardata('elev')
            isok = np.isfinite(loc_hgt)
        if _margin > 1:
            log.debug('(%s) I had to look up for far climate pixels: %s',
                      gdir.rgi_id, _margin)

        # Take the first candidate (doesn't matter which)
        lon, lat = ncclim.grid.ll_coordinates
        lon = lon[isok][0]
        lat = lat[isok][0]
        # Resubset
        ncclim.set_subset()
        ncclim.set_subset(corners=((lon, lat), (lon, lat)), margin=1)
        loc_hgt = ncclim.get_vardata('elev')
        loc_tmp = ncclim.get_vardata('temp')
        loc_pre = ncclim.get_vardata('prcp')
        loc_lon = ncclim.get_vardata('lon')
        loc_lat = ncclim.get_vardata('lat')

    assert np.isfinite(loc_hgt[1, 1])
    isok = np.isfinite(loc_hgt)
    hgt_f = loc_hgt[isok].flatten()
    assert len(hgt_f) > 0.

    # maybe this will throw out of bounds warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        nc_ts_tmp.set_subset(corners=((lon, lat), (lon, lat)), margin=1)
        nc_ts_pre.set_subset(corners=((lon, lat), (lon, lat)), margin=1)

    # compute monthly anomalies
    # of temp
    ts_tmp = nc_ts_tmp.get_vardata('tmp', as_xarray=True)
    ts_tmp_avg = ts_tmp.sel(time=slice('1961-01-01', '1990-12-01'))
    ts_tmp_avg = ts_tmp_avg.groupby('time.month').mean(dim='time')
    ts_tmp = ts_tmp.groupby('time.month') - ts_tmp_avg
    # of precip
    ts_pre = nc_ts_pre.get_vardata('pre', as_xarray=True)
    ts_pre_avg = ts_pre.sel(time=slice('1961-01-01', '1990-12-01'))
    ts_pre_avg = ts_pre_avg.groupby('time.month').mean(dim='time')
    ts_pre_ano = ts_pre.groupby('time.month') - ts_pre_avg
    # scaled anomalies is the default. Standard anomalies above
    # are used later for where ts_pre_avg == 0
    ts_pre = ts_pre.groupby('time.month') / ts_pre_avg

    # interpolate to HR grid
    if np.any(~np.isfinite(ts_tmp[:, 1, 1])):
        # Extreme case, middle pix is not valid
        # take any valid pix from the 3*3 (and hope there's one)
        found_it = False
        for idi in range(2):
            for idj in range(2):
                if np.all(np.isfinite(ts_tmp[:, idj, idi])):
                    ts_tmp[:, 1, 1] = ts_tmp[:, idj, idi]
                    ts_pre[:, 1, 1] = ts_pre[:, idj, idi]
                    ts_pre_ano[:, 1, 1] = ts_pre_ano[:, idj, idi]
                    found_it = True
        if not found_it:
            msg = '({}) there is no climate data'.format(gdir.rgi_id)
            raise MassBalanceCalibrationError(msg)
    elif np.any(~np.isfinite(ts_tmp)):
        # maybe the side is nan, but we can do nearest
        ts_tmp = ncclim.grid.map_gridded_data(ts_tmp.values, nc_ts_tmp.grid,
                                              interp='nearest')
        ts_pre = ncclim.grid.map_gridded_data(ts_pre.values, nc_ts_pre.grid,
                                              interp='nearest')
        ts_pre_ano = ncclim.grid.map_gridded_data(ts_pre_ano.values,
                                                  nc_ts_pre.grid,
                                                  interp='nearest')
    else:
        # We can do bilinear
        ts_tmp = ncclim.grid.map_gridded_data(ts_tmp.values, nc_ts_tmp.grid,
                                              interp='linear')
        ts_pre = ncclim.grid.map_gridded_data(ts_pre.values, nc_ts_pre.grid,
                                              interp='linear')
        ts_pre_ano = ncclim.grid.map_gridded_data(ts_pre_ano.values,
                                                  nc_ts_pre.grid,
                                                  interp='linear')

    # take the center pixel and add it to the CRU CL clim
    # for temp
    loc_tmp = xr.DataArray(loc_tmp[:, 1, 1], dims=['month'],
                           coords={'month': ts_tmp_avg.month})
    ts_tmp = xr.DataArray(ts_tmp[:, 1, 1], dims=['time'],
                          coords={'time': time})
    ts_tmp = ts_tmp.groupby('time.month') + loc_tmp
    # for prcp
    loc_pre = xr.DataArray(loc_pre[:, 1, 1], dims=['month'],
                           coords={'month': ts_pre_avg.month})
    ts_pre = xr.DataArray(ts_pre[:, 1, 1], dims=['time'],
                          coords={'time': time})
    ts_pre_ano = xr.DataArray(ts_pre_ano[:, 1, 1], dims=['time'],
                              coords={'time': time})
    # scaled anomalies
    ts_pre = ts_pre.groupby('time.month') * loc_pre
    # standard anomalies
    ts_pre_ano = ts_pre_ano.groupby('time.month') + loc_pre
    # Correct infinite values with standard anomalies
    ts_pre.values = np.where(np.isfinite(ts_pre.values),
                             ts_pre.values,
                             ts_pre_ano.values)
    # The last step might create negative values (unlikely). Clip them
    ts_pre.values = utils.clip_min(ts_pre.values, 0)

    # done
    loc_hgt = loc_hgt[1, 1]
    loc_lon = loc_lon[1]
    loc_lat = loc_lat[1]
    assert np.isfinite(loc_hgt)
    assert np.all(np.isfinite(ts_pre.values))
    assert np.all(np.isfinite(ts_tmp.values))

    gdir.write_monthly_climate_file(time, ts_pre.values, ts_tmp.values,
                                    loc_hgt, loc_lon, loc_lat,
                                    filesuffix=output_filesuffix,
                                    source=nc_ts_tmp._nc.title[:10])

    ncclim._nc.close()
    nc_ts_tmp._nc.close()
    nc_ts_pre._nc.close()


@entity_task(log, writes=['climate_historical'])
def process_dummy_cru_file(gdir, sigma_temp=2, sigma_prcp=0.5, seed=None,
                           y0=None, y1=None, output_filesuffix=None):
    """Create a simple baseline climate file for this glacier - for testing!

    This simply reproduces the climatology with a little randomness in it.

    TODO: extend the functionality by allowing a monthly varying sigma

    Parameters
    ----------
    gdir : GlacierDirectory
        the glacier directory
    sigma_temp : float
        the standard deviation of the random timeseries (set to 0 for constant
        ts)
    sigma_prcp : float
        the standard deviation of the random timeseries (set to 0 for constant
        ts)
    seed : int
        the RandomState seed
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

    # read the climatology
    clfile = get_cru_cl_file()
    ncclim = salem.GeoNetcdf(clfile)

    y0 = 1901 if y0 is None else y0
    y1 = 2021 if y1 is None else y1

    time = pd.date_range(start=f'{y0}-01-01', end=f'{y1}-12-01', freq='MS')
    ny, r = divmod(len(time), 12)
    assert r == 0

    lon = gdir.cenlon
    lat = gdir.cenlat

    # This is guaranteed to work because I prepared the file (I hope)
    ncclim.set_subset(corners=((lon, lat), (lon, lat)), margin=1)

    # get climatology data
    loc_hgt = ncclim.get_vardata('elev')
    loc_tmp = ncclim.get_vardata('temp')
    loc_pre = ncclim.get_vardata('prcp')
    loc_lon = ncclim.get_vardata('lon')
    loc_lat = ncclim.get_vardata('lat')

    # see if the center is ok
    if not np.isfinite(loc_hgt[1, 1]):
        # take another candidate where finite
        isok = np.isfinite(loc_hgt)

        # wait: some areas are entirely NaNs, make the subset larger
        _margin = 1
        while not np.any(isok):
            _margin += 1
            ncclim.set_subset(corners=((lon, lat), (lon, lat)), margin=_margin)
            loc_hgt = ncclim.get_vardata('elev')
            isok = np.isfinite(loc_hgt)
        if _margin > 1:
            log.debug('(%s) I had to look up for far climate pixels: %s',
                      gdir.rgi_id, _margin)

        # Take the first candidate (doesn't matter which)
        lon, lat = ncclim.grid.ll_coordinates
        lon = lon[isok][0]
        lat = lat[isok][0]
        # Resubset
        ncclim.set_subset()
        ncclim.set_subset(corners=((lon, lat), (lon, lat)), margin=1)
        loc_hgt = ncclim.get_vardata('elev')
        loc_tmp = ncclim.get_vardata('temp')
        loc_pre = ncclim.get_vardata('prcp')
        loc_lon = ncclim.get_vardata('lon')
        loc_lat = ncclim.get_vardata('lat')

    assert np.isfinite(loc_hgt[1, 1])
    isok = np.isfinite(loc_hgt)
    hgt_f = loc_hgt[isok].flatten()
    assert len(hgt_f) > 0.

    # Make DataArrays
    rng = np.random.RandomState(seed)
    loc_tmp = xr.DataArray(loc_tmp[:, 1, 1], dims=['month'],
                           coords={'month': np.arange(1, 13)})
    ts_tmp = rng.randn(len(time)) * sigma_temp
    ts_tmp = xr.DataArray(ts_tmp, dims=['time'],
                          coords={'time': time})

    loc_pre = xr.DataArray(loc_pre[:, 1, 1], dims=['month'],
                           coords={'month': np.arange(1, 13)})
    ts_pre = utils.clip_min(rng.randn(len(time)) * sigma_prcp + 1, 0)
    ts_pre = xr.DataArray(ts_pre, dims=['time'],
                          coords={'time': time})

    # Create the time series
    ts_tmp = ts_tmp.groupby('time.month') + loc_tmp
    ts_pre = ts_pre.groupby('time.month') * loc_pre

    # done
    loc_hgt = loc_hgt[1, 1]
    loc_lon = loc_lon[1]
    loc_lat = loc_lat[1]
    assert np.isfinite(loc_hgt)

    gdir.write_monthly_climate_file(time, ts_pre.values, ts_tmp.values,
                                    loc_hgt, loc_lon, loc_lat,
                                    filesuffix=output_filesuffix,
                                    source='CRU CL2 and some randomness')
    ncclim._nc.close()
