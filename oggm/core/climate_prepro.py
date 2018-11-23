"""Climate data pre-processing"""
# Built ins
import logging
from distutils.version import LooseVersion
# External libs
import numpy as np
import netCDF4
import xarray as xr
# Locals
from oggm import cfg
from oggm import utils
from oggm import entity_task
from oggm.core.climate import process_gcm_data

# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['gcm_data'])
def prepro_cesm_data(gdir, filesuffix='', fpath_temp=None, fpath_precc=None,
                     fpath_precl=None):
    """Processes and writes GCM climate data for this glacier.

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
    if LooseVersion(xr.__version__) < LooseVersion('0.10.9'):
        raise ImportError('This task needs xarray v0.10.9 or newer to run.')

    tempds = xr.open_dataset(fpath_temp)
    precpcds = xr.open_dataset(fpath_precc)
    preclpds = xr.open_dataset(fpath_precl)

    # Get the time right - i.e. from time bounds
    # Fix for https://github.com/pydata/xarray/issues/2565
    with utils.ncDataset(fpath_temp, mode='r') as nc:
        time_units = nc.variables['time'].units
        calendar = nc.variables['time'].calendar

    time = netCDF4.num2date(tempds.time_bnds[:, 0], time_units,
                            calendar=calendar)

    # select for location
    lon = gdir.cenlon
    lat = gdir.cenlat

    # CESM files are in 0-360
    if lon <= 0:
        lon += 360

    # take the closest
    # Should we consider GCM interpolation?
    temp = tempds.TREFHT.sel(lat=lat, lon=lon, method='nearest')
    prcp = (precpcds.PRECC.sel(lat=lat, lon=lon, method='nearest') +
            preclpds.PRECL.sel(lat=lat, lon=lon, method='nearest'))
    temp['time'] = time
    prcp['time'] = time

    temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
    prcp.lon.values = prcp.lon if prcp.lon <= 180 else prcp.lon - 360

    # Convert m s-1 to mm mth-1
    if time[0].month != 1:
        raise ValueError('We expect the files to start in January!')
    ny, r = divmod(len(time), 12)
    assert r == 0
    ndays = np.tile(cfg.DAYS_IN_MONTH, ny)
    prcp = prcp * ndays * (60 * 60 * 24 * 1000)

    tempds.close()
    precpcds.close()
    preclpds.close()

    # Here:
    # - time_unit='days since 0850-01-01 00:00:00'
    # - calendar='noleap'
    process_gcm_data(gdir, filesuffix=filesuffix, prcp=prcp, temp=temp,
                     time_unit=time_units, calendar=calendar)
