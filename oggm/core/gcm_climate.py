"""Climate data pre-processing"""
# Built ins
import logging
from distutils.version import LooseVersion
from datetime import datetime
# External libs
import numpy as np
import netCDF4
import xarray as xr
# Locals
from oggm import cfg
from oggm import utils
from oggm import entity_task

# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['gcm_data', 'climate_info'])
def process_gcm_data(gdir, filesuffix='', prcp=None, temp=None,
                     year_range=('1961', '1990'), scale_stddev=False,
                     time_unit=None, calendar=None):
    """ Applies the anomaly method to GCM climate data

    This function can be applied to any GCM data, if it is provided in a
    suitable :py:class:`xarray.DataArray`. See Parameter description for
    format details.

    For CESM-LME a specific function :py:func:`tasks.process_cesm_data` is
    available which does the preprocessing of the data and subsequently calls
    this function.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    prcp : :py:class:`xarray.DataArray`
        | monthly total precipitation [mm month-1]
        | Coordinates:
        | lat float64
        | lon float64
        | time: cftime object
    temp : :py:class:`xarray.DataArray`
        | monthly temperature [K]
        | Coordinates:
        | lat float64
        | lon float64
        | time cftime object
    year_range : tuple of str
        the year range for which you want to compute the anomalies. Default
        is `('1961', '1990')`
    scale_stddev : bool
        whether or not to scale the temperature standard deviation as well
    time_unit : str
        The unit conversion for NetCDF files. It must be adapted to the
        length of the time series. The default is to choose
        it ourselves based on the starting year.
        For example: 'days since 0850-01-01 00:00:00'
    calendar : str
        If you use an exotic calendar (e.g. 'noleap')
    """

    # Standard sanity checks
    months = temp['time.month']
    if (months[0] != 1) or (months[-1] != 12):
        raise ValueError('We expect the files to start in January and end in '
                         'December!')

    if (np.abs(temp['lon']) > 180) or (np.abs(prcp['lon']) > 180):
        raise ValueError('We expect the longitude coordinates to be within '
                         '[-180, 180].')

    # from normal years to hydrological years
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    prcp = prcp[sm-1:sm-13].load()
    temp = temp[sm-1:sm-13].load()

    # Get CRU to apply the anomaly to
    fpath = gdir.get_filepath('climate_monthly')
    ds_cru = xr.open_dataset(fpath)

    # Add CRU clim
    dscru = ds_cru.sel(time=slice(*year_range))

    # compute monthly anomalies
    # of temp
    if scale_stddev:
        # This is a bit more arithmetic
        ts_tmp_sel = temp.sel(time=slice(*year_range))
        ts_tmp_std = ts_tmp_sel.groupby('time.month').std(dim='time')
        std_fac = dscru.temp.groupby('time.month').std(dim='time') / ts_tmp_std
        std_fac = std_fac.roll(month=13-sm)
        std_fac = np.tile(std_fac.data, len(temp) // 12)
        win_size = 12 * (int(year_range[1]) - int(year_range[0]) + 1) + 1

        def roll_func(x, axis=None):
            assert axis == 1
            x = x[:, ::12]
            n = len(x[0, :]) // 2
            xm = np.nanmean(x, axis=axis)
            return xm + (x[:, n] - xm) * std_fac

        temp = temp.rolling(time=win_size, center=True,
                            min_periods=1).reduce(roll_func)

    ts_tmp_sel = temp.sel(time=slice(*year_range))
    ts_tmp_avg = ts_tmp_sel.groupby('time.month').mean(dim='time')
    ts_tmp = temp.groupby('time.month') - ts_tmp_avg
    # of precip -- scaled anomalies
    ts_pre_avg = prcp.sel(time=slice(*year_range))
    ts_pre_avg = ts_pre_avg.groupby('time.month').mean(dim='time')
    ts_pre_ano = prcp.groupby('time.month') - ts_pre_avg
    # scaled anomalies is the default. Standard anomalies above
    # are used later for where ts_pre_avg == 0
    ts_pre = prcp.groupby('time.month') / ts_pre_avg

    # for temp
    loc_tmp = dscru.temp.groupby('time.month').mean()
    ts_tmp = ts_tmp.groupby('time.month') + loc_tmp

    # for prcp
    loc_pre = dscru.prcp.groupby('time.month').mean()
    # scaled anomalies
    ts_pre = ts_pre.groupby('time.month') * loc_pre
    # standard anomalies
    ts_pre_ano = ts_pre_ano.groupby('time.month') + loc_pre
    # Correct infinite values with standard anomalies
    ts_pre.values = np.where(np.isfinite(ts_pre.values),
                             ts_pre.values,
                             ts_pre_ano.values)
    # The previous step might create negative values (unlikely). Clip them
    ts_pre.values = ts_pre.values.clip(0)

    assert np.all(np.isfinite(ts_pre.values))
    assert np.all(np.isfinite(ts_tmp.values))

    gdir.write_monthly_climate_file(temp.time.values,
                                    ts_pre.values, ts_tmp.values,
                                    float(dscru.ref_hgt),
                                    prcp.lon.values, prcp.lat.values,
                                    time_unit=time_unit,
                                    calendar=calendar,
                                    file_name='gcm_data',
                                    filesuffix=filesuffix)

    ds_cru.close()


@entity_task(log, writes=['gcm_data', 'climate_info'])
def process_cesm_data(gdir, filesuffix='', fpath_temp=None, fpath_precc=None,
                      fpath_precl=None, **kwargs):
    """Processes and writes CESM climate data for this glacier.

    This function is made for interpolating the Community
    Earth System Model Last Millennium Ensemble (CESM-LME) climate simulations,
    from Otto-Bliesner et al. (2016), to the high-resolution CL2 climatologies
    (provided with OGGM) and writes everything to a NetCDF file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    fpath_temp : str
        path to the temp file (default: cfg.PATHS['cesm_temp_file'])
    fpath_precc : str
        path to the precc file (default: cfg.PATHS['cesm_precc_file'])
    fpath_precl : str
        path to the precl file (default: cfg.PATHS['cesm_precl_file'])
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    # CESM temperature and precipitation data
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
    if LooseVersion(xr.__version__) < LooseVersion('0.11'):
        raise ImportError('This task needs xarray v0.11 or newer to run.')

    tempds = xr.open_dataset(fpath_temp)
    precpcds = xr.open_dataset(fpath_precc)
    preclpds = xr.open_dataset(fpath_precl)

    # Get the time right - i.e. from time bounds
    # Fix for https://github.com/pydata/xarray/issues/2565
    with utils.ncDataset(fpath_temp, mode='r') as nc:
        time_unit = nc.variables['time'].units
        calendar = nc.variables['time'].calendar

    try:
        # xarray v0.11
        time = netCDF4.num2date(tempds.time_bnds[:, 0], time_unit,
                                calendar=calendar)
    except TypeError:
        # xarray > v0.11
        time = tempds.time_bnds[:, 0].values

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
                     time_unit=time_unit, calendar=calendar, **kwargs)


@entity_task(log, writes=['gcm_data', 'climate_info'])
def process_cmip5_data(gdir, filesuffix='', fpath_temp=None,
                       fpath_precip=None, **kwargs):
    """Read, process and store the CMIP5 climate data data for this glacier.

    It stores the data in a format that can be used by the OGGM mass balance
    model and in the glacier directory.

    Currently, this function is built for the CMIP5 projection simulation
    (https://pcmdi.llnl.gov/mips/cmip5/) from Taylor et al. (2012).

    Parameters
    ----------
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    fpath_temp : str
        path to the temp file (default: cfg.PATHS['cmip5_temp_file'])
    fpath_precip : str
        path to the precip file (default: cfg.PATHS['cmip5_precip_file'])
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    # Get the path of GCM temperature & precipitation data
    if fpath_temp is None:
        if not ('cmip5_temp_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['cmip5_temp_file']")
        fpath_temp = cfg.PATHS['cmip5_temp_file']
    if fpath_precip is None:
        if not ('cmip5_precip_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['cmip5_precip_file']")
        fpath_precip = cfg.PATHS['cmip5_precip_file']

    # Read the GCM files
    tempds = xr.open_dataset(fpath_temp, decode_times=False)
    precipds = xr.open_dataset(fpath_precip, decode_times=False)

    with utils.ncDataset(fpath_temp, mode='r') as nc:
        time_units = nc.variables['time'].units
        calendar = nc.variables['time'].calendar
        time = netCDF4.num2date(nc.variables['time'][:], time_units)

    # Select for location
    lon = gdir.cenlon
    lat = gdir.cenlat

    # Conversion of the longitude
    if lon <= 0:
        lon += 360

    # Take the closest to the glacier
    # Should we consider GCM interpolation?
    temp = tempds.tas.sel(lat=lat, lon=lon, method='nearest')
    precip = precipds.pr.sel(lat=lat, lon=lon, method='nearest')

    # Time needs a set to start of month
    time = [datetime(t.year, t.month, 1) for t in time]
    temp['time'] = time
    precip['time'] = time

    temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
    precip.lon.values = precip.lon if precip.lon <= 180 else precip.lon - 360

    # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!
    if temp.time[0].dt.month != 1:
        raise ValueError('We expect the files to start in January!')

    ny, r = divmod(len(temp), 12)
    assert r == 0
    precip = precip * precip.time.dt.days_in_month * (60 * 60 * 24)

    tempds.close()
    precipds.close()

    # Here:
    # - time_unit='days since 1870-01-15 12:00:00'
    # - calendar='standard'
    process_gcm_data(gdir, filesuffix=filesuffix, prcp=precip, temp=temp,
                     time_unit=time_units, calendar=calendar, **kwargs)
