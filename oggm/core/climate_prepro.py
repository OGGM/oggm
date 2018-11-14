"""Climate data and mass-balance computations"""
# Built ins
import logging
import os
import datetime
import warnings
# External libs
import numpy as np
import netCDF4
import pandas as pd
import xarray as xr
from scipy import stats
import salem
from scipy import optimize as optimization
# Locals
from oggm import cfg
from oggm import entity_task
from oggm.core.climate import process_gcm_data
# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['gcm_data'])
def prepro_cesm_data(gdir, filesuffix='', fpath_temp=None, fpath_precc=None,
                     fpath_precl=None):
    """Processes and writes the climate data for this glacier.

    This function is made for interpolating the Community
    Earth System Model Last Millennium Ensemble (CESM-LME) climate simulations,
    from Otto-Bliesner et al. (2016), to the high-resolution CL2 climatologies
    (provided with OGGM) and writes everything to a NetCDF file.

    Parameters
    ----------
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    fpath_temp : str
        path to the temp file (default: cfg.PATHS['cesm_temp_file'])
    fpath_precc : str
        path to the precc file (default: cfg.PATHS['cesm_precc_file'])
    fpath_precl : str
        path to the precl file (default: cfg.PATHS['cesm_precl_file'])
    """

    # GCM temperature and precipitation data
    if fpath_temp is None:
        if not ('cesm_temp_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['cesm_temp_file']")
        fpath_temp = cfg.PATHS['cesm_temp_file']
    if fpath_precc is None:
        if not ('cesm_precc_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['cesm_precc_file']")
        fpath_precc = cfg.PATHS['cesm_precc_file']
    if fpath_precl is None:
        if not ('cesm_precl_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['cesm_precl_file']")
        fpath_precl = cfg.PATHS['cesm_precl_file']

    # read the files
    with warnings.catch_warnings():
        # Long time series are currently a pain pandas
        warnings.filterwarnings("ignore", message='Unable to decode time axis')
        tempds = xr.open_dataset(fpath_temp)
    precpcds = xr.open_dataset(fpath_precc, decode_times=False)
    preclpds = xr.open_dataset(fpath_precl, decode_times=False)

    # select for location
    lon = gdir.cenlon
    lat = gdir.cenlat

    # CESM files are in 0-360
    if lon <= 0:
        lon += 360

    # take the closest
    # TODO: consider GCM interpolation?
    temp = tempds.TREFHT.sel(lat=lat, lon=lon, method='nearest')
    prcp = (precpcds.PRECC.sel(lat=lat, lon=lon, method='nearest') +
            preclpds.PRECL.sel(lat=lat, lon=lon, method='nearest'))

    temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
    prcp.lon.values = prcp.lon if prcp.lon <= 180 else prcp.lon - 360

    # from normal years to hydrological years
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    em = sm - 1 if (sm > 1) else 12
    # TODO: we don't check if the files actually start in January but we should
    prcp = prcp[sm-1:sm-13].load()
    temp = temp[sm-1:sm-13].load()
    y0 = int(temp.time.values[0].strftime('%Y'))
    y1 = int(temp.time.values[-1].strftime('%Y'))
    time = pd.period_range('{}-{:02d}'.format(y0, sm),
                           '{}-{:02d}'.format(y1, em), freq='M')

    temp['time'] = time
    prcp['time'] = time
    # Workaround for https://github.com/pydata/xarray/issues/1565
    temp['month'] = ('time', time.month)
    prcp['month'] = ('time', time.month)
    temp['year'] = ('time', time.year)
    prcp['year'] = ('time', time.year)
    ny, r = divmod(len(time), 12)
    assert r == 0

    # Convert m s-1 to mm mth-1
    ndays = np.tile(np.roll(cfg.DAYS_IN_MONTH, 13-sm), y1 - y0)
    prcp = prcp * ndays * (60 * 60 * 24 * 1000)

    # load dates in right format to save
    dsindex = salem.GeoNetcdf(fpath_temp, monthbegin=True)
    time1 = dsindex.variables['time']
    time2 = time1[sm-1:sm-13] - ndays  # to hydrological years
    time2 = netCDF4.num2date(time2, time1.units, calendar='noleap')
    time_unit = time1.units

    dsindex._nc.close()
    tempds.close()
    precpcds.close()
    preclpds.close()

    process_gcm_data(gdir, filesuffix=filesuffix, prcp=prcp,
                     temp=temp, time_unit=time_unit, time2=time2)
