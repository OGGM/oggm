"""Climate data and mass-balance computations"""
from __future__ import division

# Built ins
import logging
import os
import datetime
import warnings
# External libs
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import netCDF4
import salem
from scipy import optimize as optimization
# Locals
from oggm import cfg
from oggm import utils
from oggm import entity_task, divide_task, global_task

# Module logger
log = logging.getLogger(__name__)


@global_task
def process_histalp_nonparallel(gdirs, fpath=None):
    """This is the way OGGM used to do it (deprecated).

    It requires an input file with a specific format, and uses lazy
    optimisation (computing time dependant gradients can be slow)
    """

    # Did the user specify a specific climate data file?
    if fpath is None:
        if 'climate_file' in cfg.PATHS:
            fpath = cfg.PATHS['climate_file']

    if not os.path.exists(fpath):
        raise IOError('Custom climate file not found')

    log.info('process_histalp_nonparallel')

    # read the file and data entirely (faster than many I/O)
    with netCDF4.Dataset(fpath, mode='r') as nc:
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]

        # Time
        time = nc.variables['time']
        time = netCDF4.num2date(time[:], time.units)
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        y0, y1 = time[0].year, time[-1].year

        # Units
        assert nc.variables['hgt'].units == 'm'
        assert nc.variables['temp'].units == 'degC'
        assert nc.variables['prcp'].units == 'kg m-2'

    # Gradient defaults
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    def_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    for gdir in gdirs:
        ilon = np.argmin(np.abs(lon - gdir.cenlon))
        ilat = np.argmin(np.abs(lat - gdir.cenlat))
        ref_pix_lon = lon[ilon]
        ref_pix_lat = lat[ilat]
        iprcp, itemp, igrad, ihgt = utils.joblib_read_climate(fpath, ilon,
                                                              ilat, def_grad,
                                                              g_minmax,
                                                              use_grad)
        gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                        ref_pix_lon, ref_pix_lat)
        # metadata
        out = {'climate_source': fpath,
               'hydro_yr_0': y0+1, 'hydro_yr_1': y1}
        gdir.write_pickle(out, 'climate_info')


@entity_task(log, writes=['climate_monthly'])
def process_custom_climate_data(gdir):
    """Processes and writes the climate data from a user-defined climate file.

    The input file must have a specific format (see
    oggm-sample-data/test-files/histalp_merged_hef.nc for an example).

    Uses caching for faster retrieval.

    This is the way OGGM does it for the Alps (HISTALP).
    """

    if not (('climate_file' in cfg.PATHS) and
            os.path.exists(cfg.PATHS['climate_file'])):
        raise IOError('Custom climate file not found')

    # read the file
    fpath = cfg.PATHS['climate_file']
    nc_ts = salem.GeoNetcdf(fpath)

    # set temporal subset for the ts data (hydro years)
    yrs = nc_ts.time.year
    y0, y1 = yrs[0], yrs[-1]
    nc_ts.set_period(t0='{}-10-01'.format(y0), t1='{}-09-01'.format(y1))
    time = nc_ts.time
    ny, r = divmod(len(time), 12)
    if r != 0:
        raise ValueError('Climate data should be N full years exclusively')

    # Units
    assert nc_ts._nc.variables['hgt'].units.lower() in ['m', 'meters', 'meter',
                                                        'metres', 'metre']
    assert nc_ts._nc.variables['temp'].units.lower() in ['degc', 'degrees',
                                                         'degree', 'c']
    assert nc_ts._nc.variables['prcp'].units.lower() in ['kg m-2', 'l m-2',
                                                         'mm', 'millimeters',
                                                         'millimeter']

    # geoloc
    lon = nc_ts._nc.variables['lon'][:]
    lat = nc_ts._nc.variables['lat'][:]

    # Gradient defaults
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    def_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    ilon = np.argmin(np.abs(lon - gdir.cenlon))
    ilat = np.argmin(np.abs(lat - gdir.cenlat))
    ref_pix_lon = lon[ilon]
    ref_pix_lat = lat[ilat]
    iprcp, itemp, igrad, ihgt = utils.joblib_read_climate(fpath, ilon,
                                                          ilat, def_grad,
                                                          g_minmax,
                                                          use_grad)
    gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                    ref_pix_lon, ref_pix_lat)
    # metadata
    out = {'climate_source': fpath, 'hydro_yr_0': y0+1, 'hydro_yr_1': y1}
    gdir.write_pickle(out, 'climate_info')


@entity_task(log, writes=['cesm_data'])
def process_cesm_data(gdir, filesuffix=''):
    """Processes and writes the climate data for this glacier.

    This function is made for interpolating the Community
    Earth System Model Last Millenial Ensemble (CESM-LME) climate simulations,
    from Otto-Bliesner et al. (2016), to the high-resolution CL2 climatologies
    (provided with OGGM) and writes everything to a NetCDF file.

    Parameters
    ----------
    filesuffix : str
        append a suffix to the filename (useful for model runs).
    """

    # GCM temperature and precipitation data
    if not (('gcm_temp_file' in cfg.PATHS) and
                    os.path.exists(cfg.PATHS['gcm_temp_file'])):
        raise IOError('GCM temp file not found')

    if not (('gcm_precc_file' in cfg.PATHS) and
                    os.path.exists(cfg.PATHS['gcm_precc_file'])):
        raise IOError('GCM precc file not found')

    if not (('gcm_precl_file' in cfg.PATHS) and
                    os.path.exists(cfg.PATHS['gcm_precl_file'])):
        raise IOError('GCM precl file not found')

    # read the files
    fpath_temp = cfg.PATHS['gcm_temp_file']
    fpath_precc = cfg.PATHS['gcm_precc_file']
    fpath_precl = cfg.PATHS['gcm_precl_file']

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
    precp = precpcds.PRECC.sel(lat=lat, lon=lon, method='nearest') + \
            preclpds.PRECL.sel(lat=lat, lon=lon, method='nearest')

    # from normal years to hydrological years
    # TODO: we don't check if the files actually start in January but we should
    precp = precp[9:-3]
    temp = temp[9:-3]
    y0 = int(temp.time.values[0].strftime('%Y'))
    y1 = int(temp.time.values[-1].strftime('%Y'))
    time = pd.period_range('{}-10'.format(y0), '{}-9'.format(y1), freq='M')
    temp['time'] = time
    precp['time'] = time
    # Workaround for https://github.com/pydata/xarray/issues/1565
    temp['month'] = ('time', time.month)
    precp['month'] = ('time', time.month)
    temp['year'] = ('time', time.year)
    precp['year'] = ('time', time.year)
    ny, r = divmod(len(time), 12)
    assert r == 0

    # Convert m s-1 to mm mth-1
    ndays = np.tile(cfg.DAYS_IN_MONTH_HYDRO, y1 - y0)
    precp = precp * ndays * (60 * 60 * 24 * 1000)

    # compute monthly anomalies
    # of temp
    ts_tmp_avg = temp.sel(time=(temp.year >= 1961) & (temp.year <= 1990))
    ts_tmp_avg = ts_tmp_avg.groupby(ts_tmp_avg.month).mean(dim='time')
    ts_tmp = temp.groupby(temp.month) - ts_tmp_avg
    # of precip
    ts_pre_avg = precp.isel(time=(precp.year >= 1961) & (precp.year <= 1990))
    ts_pre_avg = ts_pre_avg.groupby(ts_pre_avg.month).mean(dim='time')
    ts_pre = precp.groupby(precp.month) - ts_pre_avg

    # Get CRU to apply the anomaly to
    fpath = gdir.get_filepath('climate_monthly')
    ds_cru = xr.open_dataset(fpath)

    # Here we assume the gradient is a monthly average
    ts_grad = np.tile(ds_cru.grad[0:12], ny)

    # Add climate anomaly to CRU clim
    dscru = ds_cru.sel(time=slice('1961', '1990'))
    # for temp
    loc_tmp = dscru.temp.groupby('time.month').mean()
    ts_tmp = ts_tmp.groupby(ts_tmp.month) + loc_tmp
    # for prcp
    loc_pre = dscru.prcp.groupby('time.month').mean()
    ts_pre = ts_pre.groupby(ts_pre.month) + loc_pre

    # load dates in right format to save
    dsindex = salem.GeoNetcdf(fpath_temp, monthbegin=True)
    time1 = dsindex.variables['time']
    time2 = time1[9:-3] - ndays  # from normal years to hydrological years
    time2 = netCDF4.num2date(time2, time1.units, calendar='noleap')

    assert np.all(np.isfinite(ts_pre.values))
    assert np.all(np.isfinite(ts_tmp.values))
    assert np.all(np.isfinite(ts_grad))

    # back to -180 - 180
    loc_lon = precp.lon if precp.lon <= 180 else precp.lon - 360

    gdir.write_monthly_climate_file(time2, ts_pre.values, ts_tmp.values,
                                    ts_grad, float(dscru.ref_hgt),
                                    loc_lon, precp.lat.values,
                                    time_unit=time1.units,
                                    file_name='cesm_data',
                                    filesuffix=filesuffix)

    dsindex._nc.close()
    tempds.close()
    precpcds.close()
    preclpds.close()
    ds_cru.close()


@entity_task(log, writes=['climate_monthly'])
def process_cru_data(gdir):
    """Processes and writes the climate data for this glacier.

    Interpolates the CRU TS data to the high-resolution CL2 climatologies
    (provided with OGGM) and writes everything to a NetCDF file.
    """

    # read the climatology
    clfile = utils.get_cru_cl_file()
    ncclim = salem.GeoNetcdf(clfile)
    # and the TS data
    nc_ts_tmp = salem.GeoNetcdf(utils.get_cru_file('tmp'), monthbegin=True)
    nc_ts_pre = salem.GeoNetcdf(utils.get_cru_file('pre'), monthbegin=True)

    # set temporal subset for the ts data (hydro years)
    yrs = nc_ts_pre.time.year
    y0, y1 = yrs[0], yrs[-1]
    nc_ts_tmp.set_period(t0='{}-10-01'.format(y0), t1='{}-09-01'.format(y1))
    nc_ts_pre.set_period(t0='{}-10-01'.format(y0), t1='{}-09-01'.format(y1))
    time = nc_ts_pre.time
    ny, r = divmod(len(time), 12)
    assert r == 0

    # gradient default params
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    def_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']

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
    ts_grad = np.zeros(12) + def_grad
    if use_grad and len(hgt_f) >= 5:
        for i in range(12):
            loc_tmp_mth = loc_tmp[i, ...][isok].flatten()
            slope, _, _, p_val, _ = stats.linregress(hgt_f, loc_tmp_mth)
            ts_grad[i] = slope if (p_val < 0.01) else def_grad
    # ... but dont exaggerate too much
    ts_grad = np.clip(ts_grad, g_minmax[0], g_minmax[1])
    # convert to timeserie and hydroyears
    ts_grad = ts_grad.tolist()
    ts_grad = ts_grad[9:] + ts_grad[0:9]
    ts_grad = np.asarray(ts_grad * ny)

    # maybe this will throw out of bounds warnings
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
    ts_pre = ts_pre.groupby('time.month') - ts_pre_avg

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
                    found_it = True
        if not found_it:
            msg = '({}) there is no climate data'.format(gdir.rgi_id)
            raise RuntimeError(msg)
    elif np.any(~np.isfinite(ts_tmp)):
        # maybe the side is nan, but we can do nearest
        ts_tmp = ncclim.grid.map_gridded_data(ts_tmp.values, nc_ts_tmp.grid,
                                              interp='nearest')
        ts_pre = ncclim.grid.map_gridded_data(ts_pre.values, nc_ts_pre.grid,
                                              interp='nearest')
    else:
        # We can do bilinear
        ts_tmp = ncclim.grid.map_gridded_data(ts_tmp.values, nc_ts_tmp.grid,
                                              interp='linear')
        ts_pre = ncclim.grid.map_gridded_data(ts_pre.values, nc_ts_pre.grid,
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
    ts_pre = ts_pre.groupby('time.month') + loc_pre

    # done
    loc_hgt = loc_hgt[1, 1]
    loc_lon = loc_lon[1]
    loc_lat = loc_lat[1]
    assert np.isfinite(loc_hgt)
    assert np.all(np.isfinite(ts_pre.values))
    assert np.all(np.isfinite(ts_tmp.values))
    assert np.all(np.isfinite(ts_grad))
    gdir.write_monthly_climate_file(time, ts_pre.values, ts_tmp.values,
                                    ts_grad, loc_hgt, loc_lon, loc_lat)
    ncclim._nc.close()
    nc_ts_tmp._nc.close()
    nc_ts_pre._nc.close()
    # metadata
    out = {'climate_source': 'CRU data', 'hydro_yr_0': y0+1, 'hydro_yr_1': y1}
    gdir.write_pickle(out, 'climate_info')


def mb_climate_on_height(gdir, heights, prcp_fac,
                         time_range=None, year_range=None):
    """Mass-balance climate of the glacier at a specific height

    Reads the glacier's monthly climate data file and computes the
    temperature "energies" (temp above 0) and solid precipitation at the
    required height.

    Parameters:
    -----------
    gdir: the glacier directory
    heights: a 1D array of the heights (in meter) where you want the data
    prcp_fac: the correction factor for precipitation
    time_range (optional): default is to read all data but with this you
    can provide a [datetime, datetime] bounds (inclusive).
    year_range (optional): maybe more useful than the time bounds above.
    Provide a [y0, y1] year range to get the data for specific (hydrological)
    years only

    Returns:
    --------
    (time, tempformelt, prcpsol)::
        - time: array of shape (nt,)
        - tempformelt:  array of shape (len(heights), nt)
        - prcpsol:  array of shape (len(heights), nt)
    """

    if year_range is not None:
        t0 = datetime.datetime(year_range[0]-1, 10, 1)
        t1 = datetime.datetime(year_range[1], 9, 1)
        return mb_climate_on_height(gdir, heights, prcp_fac,
                                    time_range=[t0, t1])

    # Parameters
    temp_all_solid = cfg.PARAMS['temp_all_solid']
    temp_all_liq = cfg.PARAMS['temp_all_liq']
    temp_melt = cfg.PARAMS['temp_melt']

    # Read file
    with netCDF4.Dataset(gdir.get_filepath('climate_monthly'), mode='r') as nc:
        # time
        time = nc.variables['time']
        time = netCDF4.num2date(time[:], time.units)
        if time_range is not None:
            p0 = np.where(time == time_range[0])[0]
            try:
                p0 = p0[0]
            except IndexError:
                raise RuntimeError('time_range[0] not found in file')
            p1 = np.where(time == time_range[1])[0]
            try:
                p1 = p1[0]
            except IndexError:
                raise RuntimeError('time_range[1] not found in file')
        else:
            p0 = 0
            p1 = len(time)-1

        time = time[p0:p1+1]

        # Read timeseries
        itemp = nc.variables['temp'][p0:p1+1]
        iprcp = nc.variables['prcp'][p0:p1+1]
        igrad = nc.variables['grad'][p0:p1+1]
        ref_hgt = nc.ref_hgt

    # Correct precipitation
    iprcp *= prcp_fac

    # For each height pixel:
    # Compute temp and tempformelt (temperature above melting threshold)
    npix = len(heights)
    grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
    grad_temp *= (heights.repeat(len(time)).reshape(grad_temp.shape) - ref_hgt)
    temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
    temp2dformelt = temp2d - temp_melt
    temp2dformelt = np.clip(temp2dformelt, 0, temp2dformelt.max())
    # Compute solid precipitation from total precipitation
    prcpsol = np.atleast_2d(iprcp).repeat(npix, 0)
    fac = 1 - (temp2d - temp_all_solid) / (temp_all_liq - temp_all_solid)
    fac = np.clip(fac, 0, 1)
    prcpsol = prcpsol * fac

    return time, temp2dformelt, prcpsol


def mb_yearly_climate_on_height(gdir, heights, prcp_fac,
                                year_range=None, flatten=False):
    """Yearly mass-balance climate of the glacier at a specific height

    The precipitation time series are not corrected!

    Parameters:
    -----------
    gdir: the glacier directory
    heights: a 1D array of the heights (in meter) where you want the data
    prcp_fac: the correction factor for precipitation
    year_range (optional): a [y0, y1] year range to get the data for specific (
    hydrological) years only
    flatten: for some applications (glacier average MB) it's ok to flatten  the
    data (average over height) prior to annual summing.

    Returns:
    --------
    (years, tempformelt, prcpsol)::
        - years: array of shape (ny,)
        - tempformelt:  array of shape (len(heights), ny) (or ny if flatten
        is set)
        - prcpsol:  array of shape (len(heights), ny) (or ny if flatten
        is set)
    """

    time, temp, prcp = mb_climate_on_height(gdir, heights, prcp_fac,
                                            year_range=year_range)

    ny, r = divmod(len(time), 12)
    if r != 0:
        raise ValueError('Climate data should be N full years exclusively')
    # Last year gives the tone of the hydro year
    years = np.arange(time[-1].year-ny+1, time[-1].year+1, 1)

    if flatten:
        # Spatial average
        temp_yr = np.zeros(len(years))
        prcp_yr = np.zeros(len(years))
        temp = np.mean(temp, axis=0)
        prcp = np.mean(prcp, axis=0)
        for i, y in enumerate(years):
            temp_yr[i] = np.sum(temp[i*12:(i+1)*12])
            prcp_yr[i] = np.sum(prcp[i*12:(i+1)*12])
    else:
        # Annual prcp and temp for each point (no spatial average)
        temp_yr = np.zeros((len(heights), len(years)))
        prcp_yr = np.zeros((len(heights), len(years)))
        for i, y in enumerate(years):
            temp_yr[:, i] = np.sum(temp[:, i*12:(i+1)*12], axis=1)
            prcp_yr[:, i] = np.sum(prcp[:, i*12:(i+1)*12], axis=1)

    return years, temp_yr, prcp_yr


def mb_yearly_climate_on_glacier(gdir, prcp_fac, div_id=None, year_range=None):
    """Yearly mass-balance climate at all glacier heights,
    multiplied with the flowlines widths. (all in pix coords.)

    The precipitation time series are not corrected!

    Parameters:
    -----------
    gdir: the glacier directory
    prcp_fac: the correction factor for precipitation
    year_range (optional): a [y0, y1] year range to get the data for specific
    (hydrological) years only

    Returns:
    --------
    (years, tempformelt, prcpsol)::
        - years: array of shape (ny,)
        - tempformelt:  array of shape (len(heights), ny)
        - prcpsol:  array of shape (len(heights), ny) (not corrected!)
    """

    flowlines = []
    if div_id is None:
        raise ValueError('Must specify div_id')

    if div_id == 0:
        for i in gdir.divide_ids:
            flowlines.extend(gdir.read_pickle('inversion_flowlines',
                                              div_id=i))
    else:
        flowlines = gdir.read_pickle('inversion_flowlines', div_id=div_id)

    heights = np.array([])
    widths = np.array([])
    for fl in flowlines:
        heights = np.append(heights, fl.surface_h)
        widths = np.append(widths, fl.widths)

    years, temp, prcp = mb_yearly_climate_on_height(gdir, heights, prcp_fac,
                                                    year_range=year_range,
                                                    flatten=False)

    temp = np.average(temp, axis=0, weights=widths)
    prcp = np.average(prcp, axis=0, weights=widths)

    return years, temp, prcp


@entity_task(log, writes=['mu_candidates'])
@divide_task(log, add_0=True)
def mu_candidates(gdir, div_id=None, prcp_sf=None):
    """Computes the mu candidates.

    For each 31 year-period centered on the year of interest, mu is is the
    temperature sensitivity necessary for the glacier with its current shape
    to be in equilibrium with its climate.

    For glaciers with MB data only!

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    prcp_sf : float (optional)
        force to a certain prcp scaling factor
    """

    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])

    # Only get the years were we consider looking for tstar
    y0, y1 = cfg.PARAMS['tstar_search_window']
    ci = gdir.read_pickle('climate_info')
    y0 = y0 or ci['hydro_yr_0']
    y1 = y1 or ci['hydro_yr_1']

    years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir, 1.,
                                                           div_id=div_id,
                                                           year_range=[y0, y1])

    # Be sure we have no marine terminating glacier
    assert gdir.terminus_type == 'Land-terminating'

    # prcp scaling factor - one or more?
    if cfg.PARAMS['prcp_scaling_factor'] == 'stddev_perglacier':
        sf = np.arange(0.5, 5.01, 0.05)
    elif cfg.PARAMS['prcp_scaling_factor'] == 'stddev':
        sf = [float(prcp_sf)]
    else:
        sf = np.asarray([cfg.PARAMS['prcp_scaling_factor']])

    # Compute mu for each 31-yr climatological period and each prcp factor
    ny = len(years)
    nsf = len(sf)
    mu_yr_clim = np.zeros((ny, nsf)) * np.NaN
    for j, fac in enumerate(sf):
        for i, y in enumerate(years):
            # Ignore begin and end
            if ((i-mu_hp) < 0) or ((i+mu_hp) >= ny):
                continue
            t_avg = np.mean(temp_yr[i-mu_hp:i+mu_hp+1])
            if t_avg > 1e-3:  # if too cold no melt possible
                prcp_ts = prcp_yr[i-mu_hp:i+mu_hp+1] * fac
                mu_yr_clim[i, j] = np.mean(prcp_ts) / t_avg

    # Check that we found a least one mustar
    if np.sum(np.isfinite(mu_yr_clim)) < 1:
        raise RuntimeError('({}) no mustar candidates found.'
                           .format(gdir.rgi_id))

    # Write
    df = pd.DataFrame(data=mu_yr_clim, index=years, columns=sf)
    gdir.write_pickle(df, 'mu_candidates', div_id=div_id)


def t_star_from_refmb(gdir, mbdf):
    """Computes the t* for the glacier, given a series of MB measurements.

    Could be multiprocessed but its probably not necessary.

    Parameters
    ----------
    gdirs: the list of oggm.GlacierDirectory objects where to write the data.
    mbdf: a pd.Series containing the observed MB data indexed by year

    Returns
    -------
    A dict: {t_star:[], bias:[], bias_std:[], prcp_fac:float}
    """

    # Only divide 0, we believe the original RGI entities to be the ref...
    years, temp_yr_ts, prcp_yr_ts = mb_yearly_climate_on_glacier(gdir, 1.,
                                                                 div_id=0)

    # which years to look at
    selind = np.searchsorted(years, mbdf.index)
    temp_yr_ts = temp_yr_ts[selind]
    prcp_yr_ts = prcp_yr_ts[selind]
    temp_yr = np.mean(temp_yr_ts)
    prcp_yr = np.mean(prcp_yr_ts)

    # Average oberved mass-balance
    ref_mb = np.mean(mbdf)
    ref_mb_std = np.std(mbdf)

    # Average mass-balance per mu and fac
    mu_yr_clim_df = gdir.read_pickle('mu_candidates', div_id=0)

    odf = pd.DataFrame(index=mu_yr_clim_df.columns)
    out = dict()
    for prcp_fac in mu_yr_clim_df:

        mu_yr_clim = mu_yr_clim_df[prcp_fac]
        nmu = len(mu_yr_clim)
        mb_per_mu = prcp_yr * prcp_fac - mu_yr_clim * temp_yr
        mbts_per_mu = np.atleast_2d(prcp_yr_ts * prcp_fac).repeat(nmu, 0) - \
                      np.atleast_2d(mu_yr_clim).T * \
                      np.atleast_2d(temp_yr_ts).repeat(nmu, 0)
        std_per_mu = mb_per_mu*0 + np.std(mbts_per_mu, axis=1)

        # Diff to reference
        diff = (mb_per_mu - ref_mb).dropna()
        diff_std = (std_per_mu - ref_mb_std).dropna()
        signchange = utils.signchange(diff)

        # If sign changes save them
        # TODO: ideas to do this better:
        #  - apply a smooth (take noise into account)
        #  - take longer stable periods into account
        # these stuffs could be proven better (or not) with cross-val
        pchange = np.where(signchange == 1)[0]
        years = diff.index
        diff = diff.values
        diff_std = diff_std.values
        if len(pchange) > 0:
            t_stars = []
            bias = []
            std_bias = []
            for p in pchange:
                # Check the side with the smallest bias
                ide = p-1 if np.abs(diff[p-1]) < np.abs(diff[p]) else p
                if years[ide] not in t_stars:
                    t_stars.append(years[ide])
                    bias.append(diff[ide])
                    std_bias.append(diff_std[ide])
        else:
            amin = np.argmin(np.abs(diff))
            t_stars = [years[amin]]
            bias = [diff[amin]]
            std_bias = [diff_std[amin]]

        # (prcp_fac, t_stars, bias, std_bias)
        odf.loc[prcp_fac, 'avg_bias'] = np.mean(bias)
        odf.loc[prcp_fac, 'avg_std_bias'] = np.mean(std_bias)
        odf.loc[prcp_fac, 'n_tstar'] = len(std_bias)
        out[prcp_fac] = {'t_star': t_stars, 'bias': bias, 'std_bias': std_bias,
                         'prcp_fac': prcp_fac}

    # write
    gdir.write_pickle(odf, 'prcp_fac_optim')

    # we take the closest result and see later if it needs cleverer handling
    amin = np.argmin(np.abs(odf.avg_std_bias))  # this gives back an index!
    return out[amin]


def calving_mb(gdir):
    """Calving mass-loss in specific MB equivalent.

    This is necessary to compute mu star.

    TODO: currently this is hardcoded for Columbia, but we should come-up with
    somthing better!
    """

    if not gdir.is_tidewater:
        return 0.

    # Ok. Just take the caving rate from cfg and change its units
    # Original units: km3 a-1, to change to mm a-1 (units of specific MB)
    return gdir.inversion_calving_rate * 1e9 * cfg.RHO / gdir.rgi_area_m2


@entity_task(log, writes=['local_mustar', 'inversion_flowlines'])
def local_mustar_apparent_mb(gdir, tstar=None, bias=None, prcp_fac=None,
                             compute_apparent_mb=True):
    """Compute local mustar and apparent mb from tstar.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    tstar: int
        the year where the glacier should be equilibrium
    bias: int
        the associated reference bias
    prcp_fac: int
        the associated precipitation factor
    compute_apparent_mb: bool
        if you want to compute the apparent MB at the same time (recommended,
        unless you are cross-validating for example).
    """

    assert bias is not None
    assert prcp_fac is not None
    assert tstar is not None

    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr = [tstar-mu_hp, tstar+mu_hp]

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)

    # Ok. Looping over divides
    for div_id in [0] + list(gdir.divide_ids):
        log.info('(%s) local mu* for t*=%d, divide %d',
                 gdir.rgi_id, tstar, div_id)

        # Get the corresponding mu
        years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir, prcp_fac,
                                                               div_id=div_id,
                                                               year_range=yr)
        assert len(years) == (2 * mu_hp + 1)

        # mustar is taking calving into account (units of specific MB)
        mustar = (np.mean(prcp_yr) - cmb) / np.mean(temp_yr)
        if not np.isfinite(mustar):
            raise RuntimeError('{} has a non finite mu'.format(gdir.rgi_id))

        # Scalars in a small dataframe for later
        df = pd.DataFrame()
        df['rgi_id'] = [gdir.rgi_id]
        df['t_star'] = [tstar]
        df['mu_star'] = [mustar]
        df['prcp_fac'] = [prcp_fac]
        df['bias'] = [bias]
        df.to_csv(gdir.get_filepath('local_mustar', div_id=div_id),
                  index=False)

        if not compute_apparent_mb:
            continue

        # For each flowline compute the apparent MB
        # For div 0 it is kind of artificial but this is for validation
        fls = []
        if div_id == 0:
            for i in gdir.divide_ids:
                fls.extend(gdir.read_pickle('inversion_flowlines', div_id=i))
        else:
            fls = gdir.read_pickle('inversion_flowlines', div_id=div_id)

        # Reset flux
        for fl in fls:
            fl.flux = np.zeros(len(fl.surface_h))

        # Flowlines in order to be sure
        for fl in fls:
            y, t, p = mb_yearly_climate_on_height(gdir, fl.surface_h, prcp_fac,
                                                  year_range=yr,
                                                  flatten=False)
            fl.set_apparent_mb(np.mean(p, axis=1) - mustar*np.mean(t, axis=1))

        # Check and write
        if div_id > 0:
            aflux = fls[-1].flux[-1] * 1e-9 / cfg.RHO * gdir.grid.dx**2
            # If not marine and a bit far from zero, warning
            if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=0.01):
                log.warning('(%s) flux should be zero, but is: '
                            '%.4f km3 ice yr-1', gdir.rgi_id, aflux)
            # If not marine and quite far from zero, error
            if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=1):
                msg = ('({}) flux should be zero, but is: {:.4f} km3 ice yr-1'
                       .format(gdir.rgi_id, aflux))
                raise RuntimeError(msg)
            gdir.write_pickle(fls, 'inversion_flowlines', div_id=div_id)


@entity_task(log, writes=['inversion_flowlines', 'linear_mb_params'])
@divide_task(log, add_0=True)
def apparent_mb_from_linear_mb(gdir, div_id=None, mb_gradient=3.):
    """Compute apparent mb from a linear mass-balance assumption (for testing).

    This is for testing currently, but could be used as alternative method
    for the inversion quite easily.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)

    # Get the height and widths along the fls
    if div_id == 0:
        h, w = gdir.get_inversion_flowline_hw()
    else:
        h, w = gdir.get_inversion_flowline_hw(div_id=div_id)

    # Now find the ELA till the integrated mb is zero
    from oggm.core.models.massbalance import LinearMassBalanceModel
    def to_minimize(ela_h):
        mbmod = LinearMassBalanceModel(ela_h[0], grad=mb_gradient)
        smb = mbmod.get_specific_mb(h, w)
        return (smb - cmb)**2

    ela_h = optimization.minimize(to_minimize, [0.], bounds=((0, 10000), ))
    ela_h = ela_h['x'][0]
    mbmod = LinearMassBalanceModel(ela_h, grad=mb_gradient)

    # For each flowline compute the apparent MB
    # For div 0 it is kind of artificial but this is for validation
    fls = []
    if div_id == 0:
        for i in gdir.divide_ids:
            fls.extend(gdir.read_pickle('inversion_flowlines', div_id=i))
    else:
        fls = gdir.read_pickle('inversion_flowlines', div_id=div_id)

    # Reset flux
    for fl in fls:
        fl.flux = np.zeros(len(fl.surface_h))

    # Flowlines in order to be sure
    for fl in fls:
        mbz = mbmod.get_annual_mb(fl.surface_h) * cfg.SEC_IN_YEAR * cfg.RHO
        fl.set_apparent_mb(mbz)

    # Check and write
    if div_id > 0:
        aflux = fls[-1].flux[-1] * 1e-9 / cfg.RHO * gdir.grid.dx**2
        # If not marine and a bit far from zero, warning
        if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=0.01):
            log.warning('(%s) flux should be zero, but is: '
                        '%.4f km3 ice yr-1', gdir.rgi_id, aflux)
        # If not marine and quite far from zero, error
        if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=1):
            msg = ('({}) flux should be zero, but is: {:.4f} km3 ice yr-1'
                   .format(gdir.rgi_id, aflux))
            raise RuntimeError(msg)
        gdir.write_pickle(fls, 'inversion_flowlines', div_id=div_id)

    gdir.write_pickle({'ela_h': ela_h, 'grad': mb_gradient},
                      'linear_mb_params', div_id=div_id)


def _get_ref_glaciers(gdirs):
    """Get the list of glaciers we have valid data for."""

    flink, _ = utils.get_wgms_files()
    dfids = pd.read_csv(flink)[gdirs[0].rgi_version + '_ID'].values

    # TODO: we removed marine glaciers here. Is it ok?
    ref_gdirs = []
    for g in gdirs:
        if g.rgi_id not in dfids or g.terminus_type != 'Land-terminating':
            continue
        mbdf = g.get_ref_mb_data()
        if len(mbdf) >= 5:
            ref_gdirs.append(g)
    return ref_gdirs


def _get_optimal_scaling_factor(ref_gdirs):
    """Get the precipitation scaling factor that minimizes the std dev error.
    """

    from scipy import optimize as optimization

    def to_optimize(sf):
        abs_std = []
        for gdir in ref_gdirs:
            # all possible mus
            mu_candidates(gdir, prcp_sf=sf, reset=True)
            # list of mus compatibles with refmb
            mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
            res = t_star_from_refmb(gdir, mbdf)
            abs_std.append(np.mean(res['std_bias']))
        return np.mean(abs_std)**2

    with utils.DisableLogger():
        opti = optimization.minimize(to_optimize, [1.],
                                     bounds=((0.01, 10),),
                                     tol=1)

    fac = opti['x'][0]
    log.info('Optimal prcp factor: {:.2f}'.format(fac))
    return fac


@global_task
def compute_ref_t_stars(gdirs):
    """ Detects the best t* for the reference glaciers.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    log.info('Compute the reference t* and mu* for WGMS glaciers')

    # Reference glaciers only if in the list and period is good
    ref_gdirs = _get_ref_glaciers(gdirs)

    prcp_sf = None
    if cfg.PARAMS['prcp_scaling_factor'] == 'stddev':
        prcp_sf = _get_optimal_scaling_factor(ref_gdirs)

    # Loop
    only_one = []  # start to store the glaciers with just one t*
    per_glacier = dict()
    for gdir in ref_gdirs:
        # all possible mus
        mu_candidates(gdir, prcp_sf=prcp_sf, reset=True)
        # list of mus compatibles with refmb
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = t_star_from_refmb(gdir, mbdf)

        # if we have just one candidate this is good
        if len(res['t_star']) == 1:
            only_one.append(gdir.rgi_id)
        # this might be more than one, we'll have to select them later
        per_glacier[gdir.rgi_id] = (gdir, res['t_star'], res['bias'],
                                    res['prcp_fac'])

    # At least one of the glaciers should have a single t*, otherwise we don't
    # know how to start
    if len(only_one) == 0:
        # TODO: hardcoded stuff here, for the test workflow
        if 'RGI50-11.00897' in per_glacier:
            only_one.append('RGI50-11.00897')
            gdir, t_star, res_bias, prcp_fac = per_glacier['RGI50-11.00897']
            per_glacier['RGI50-11.00897'] = (gdir, [t_star[-1]],
                                             [res_bias[-1]], prcp_fac)
        elif 'RGI40-11.00897' in per_glacier:
            only_one.append('RGI40-11.00897')
            gdir, t_star, res_bias, prcp_fac = per_glacier['RGI40-11.00897']
            per_glacier['RGI40-11.00897'] = (gdir, [t_star[-1]],
                                             [res_bias[-1]], prcp_fac)
        else:
            raise RuntimeError('We need at least one glacier with one '
                               'tstar only.')

    log.info('%d out of %d have only one possible t*. Start from here',
             len(only_one), len(ref_gdirs))

    # Ok. now loop over the nearest glaciers until all have a unique t*
    while True:
        ids_left = [id for id in per_glacier.keys() if id not in only_one]
        if len(ids_left) == 0:
            break

        # Compute the summed distance to all glaciers with one t*
        distances = []
        for id in ids_left:
            gdir = per_glacier[id][0]
            lon, lat = gdir.cenlon, gdir.cenlat
            ldis = 0.
            for id_o in only_one:
                ogdir = per_glacier[id_o][0]
                ldis += utils.haversine(lon, lat, ogdir.cenlon, ogdir.cenlat)
            distances.append(ldis)

        # Take the shortest and choose the best t*
        pg = per_glacier[ids_left[np.argmin(distances)]]
        gdir, t_star, res_bias, prcp_fac = pg
        distances = []
        for tt in t_star:
            ldis = 0.
            for id_o in only_one:
                _, ot_star, _, _ = per_glacier[id_o]
                ldis += np.abs(tt - ot_star)
            distances.append(ldis)
        amin = np.argmin(distances)
        per_glacier[gdir.rgi_id] = (gdir, [t_star[amin]], [res_bias[amin]],
                                    prcp_fac)
        only_one.append(gdir.rgi_id)

    # Write out the data
    rgis_ids, t_stars, prcp_facs,  biases, lons, lats, n_mb = ([], [], [], [],
                                                               [], [], [])
    for id, (gdir, t_star, res_bias, prcp_fac) in per_glacier.items():
        rgis_ids.append(id)
        t_stars.append(t_star[0])
        prcp_facs.append(prcp_fac)
        biases.append(res_bias[0])
        lats.append(gdir.cenlat)
        lons.append(gdir.cenlon)
        n_mb.append(len(gdir.get_ref_mb_data()))
    df = pd.DataFrame(index=rgis_ids)
    df['lon'] = lons
    df['lat'] = lats
    df['n_mb_years'] = n_mb
    df['tstar'] = t_stars
    df['prcp_fac'] = prcp_facs
    df['bias'] = biases
    file = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
    df.sort_index().to_csv(file)


@global_task
def distribute_t_stars(gdirs, compute_apparent_mb=True, ref_df=None):
    """After the computation of the reference tstars, apply
    the interpolation to each individual glacier.

    Parameters
    ----------
    gdirs : list of oggm.GlacierDirectory objects
    compute_apparent_mb : bool (defaults to True)
    """

    log.info('Distribute t* and mu*')

    if ref_df is None:
        ref_df = pd.read_csv(os.path.join(cfg.PATHS['working_dir'],
                                          'ref_tstars.csv'))

    for gdir in gdirs:

        # Compute the distance to each glacier
        distances = utils.haversine(gdir.cenlon, gdir.cenlat,
                                    ref_df.lon, ref_df.lat)

        # Take the 10 closests
        aso = np.argsort(distances)[0:9]
        amin = ref_df.iloc[aso]
        distances = distances[aso]**2

        # If really close no need to divide, else weighted average
        if distances.iloc[0] <= 0.1:
            tstar = amin.tstar.iloc[0]
            prcp_fac = amin.prcp_fac.iloc[0]
            bias = amin.bias.iloc[0]
        else:
            tstar = int(np.average(amin.tstar, weights=1./distances))
            prcp_fac = np.average(amin.prcp_fac, weights=1./distances)
            bias = np.average(amin.bias, weights=1./distances)

        # Go
        local_mustar_apparent_mb(gdir, tstar=tstar, bias=bias,
                                 prcp_fac=prcp_fac,
                                 compute_apparent_mb=compute_apparent_mb,
                                 reset=True)


@global_task
def crossval_t_stars(gdirs):
    """Cross-validate the interpolation of tstar to each individual glacier.

    This is a naive, thorough check (redoes many, many useless calculations).

    you can use quick_crossval_t_stars for most purposes,

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    log.info('Cross-validate the t* and mu* determination')

    full_ref_df = pd.read_csv(os.path.join(cfg.PATHS['working_dir'],
                                           'ref_tstars.csv'), index_col=0)

    rgdirs = _get_ref_glaciers(gdirs)
    n = len(full_ref_df)
    for i, rid in enumerate(full_ref_df.index):

        log.info('Cross-validation iteration {} of {}'.format(i+1, n))

        # the glacier to look at
        gdir = [g for g in rgdirs if g.rgi_id == rid][0]

        # the reference glaciers
        ref_gdirs = [g for g in rgdirs if g.rgi_id != rid]

        # redo the computations
        with utils.DisableLogger():
            compute_ref_t_stars(ref_gdirs)
            distribute_t_stars([gdir], compute_apparent_mb=False)

        # store
        rdf = pd.read_csv(gdir.get_filepath('local_mustar'))
        full_ref_df.loc[rid, 'cv_tstar'] = int(rdf['t_star'].values[0])
        full_ref_df.loc[rid, 'cv_mustar'] = rdf['mu_star'].values[0]
        full_ref_df.loc[rid, 'cv_prcp_fac'] = rdf['prcp_fac'].values[0]
        full_ref_df.loc[rid, 'cv_bias'] = rdf['bias'].values[0]

    # write
    file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
    full_ref_df.to_csv(file)


@global_task
def quick_crossval_t_stars(gdirs):
    """Cross-validate the interpolation of tstar to each individual glacier.

    This version does NOT recompute the precipitation scaling factor at each
    round (this quite OK to do so)

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    log.info('Cross-validate the t* and mu* determination')

    rgdirs = _get_ref_glaciers(gdirs)

    # This might be redundant but we redo the calc here
    with utils.DisableLogger():
        compute_ref_t_stars(rgdirs)
    full_ref_df = pd.read_csv(os.path.join(cfg.PATHS['working_dir'],
                                           'ref_tstars.csv'), index_col=0)
    with utils.DisableLogger():
        distribute_t_stars(rgdirs, compute_apparent_mb=False)

    n = len(full_ref_df)
    for i, rid in enumerate(full_ref_df.index):

        # log.info('Cross-validation iteration {} of {}'.format(i+1, n))

        # the glacier to look at
        gdir = [g for g in rgdirs if g.rgi_id == rid][0]

        # the reference glaciers
        tmp_ref_df = full_ref_df.loc[full_ref_df.index != rid]

        # before the cross-val we can get the info about "real" mustar
        rdf = pd.read_csv(gdir.get_filepath('local_mustar'))
        full_ref_df.loc[rid, 'mustar'] = rdf['mu_star'].values[0]

        # redo the computations
        with utils.DisableLogger():
            distribute_t_stars([gdir], ref_df=tmp_ref_df,
                               compute_apparent_mb=False)

        # store
        rdf = pd.read_csv(gdir.get_filepath('local_mustar'))
        full_ref_df.loc[rid, 'cv_tstar'] = int(rdf['t_star'].values[0])
        full_ref_df.loc[rid, 'cv_mustar'] = rdf['mu_star'].values[0]
        full_ref_df.loc[rid, 'cv_prcp_fac'] = rdf['prcp_fac'].values[0]
        full_ref_df.loc[rid, 'cv_bias'] = rdf['bias'].values[0]

    # Reproduce Ben's figure
    for i, rid in enumerate(full_ref_df.index):
        # the glacier to look at
        gdir = full_ref_df.loc[full_ref_df.index == rid]
        # the reference glaciers
        tmp_ref_df = full_ref_df.loc[full_ref_df.index != rid]

        # Compute the distance
        distances = utils.haversine(gdir.lon.values[0], gdir.lat.values[0],
                                    tmp_ref_df.lon, tmp_ref_df.lat)

        # Take the 10 closests
        aso = np.argsort(distances)[0:9]
        amin = tmp_ref_df.iloc[aso]
        distances = distances[aso] ** 2
        interp = np.average(amin.mustar, weights=1. / distances)
        full_ref_df.loc[rid, 'interp_mustar'] = interp

    # write
    file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
    full_ref_df.to_csv(file)
