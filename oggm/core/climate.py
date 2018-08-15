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
from oggm import utils
from oggm.core import centerlines
from oggm import entity_task, global_task

# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['climate_monthly'])
def process_custom_climate_data(gdir):
    """Processes and writes the climate data from a user-defined climate file.

    The input file must have a specific format (see
    oggm-sample-data/test-files/histalp_merged_hef.nc for an example).

    This is the way OGGM used to do it for HISTALP before it got automatised.
    """

    if not (('climate_file' in cfg.PATHS) and
            os.path.exists(cfg.PATHS['climate_file'])):
        raise IOError('Custom climate file not found')

    if cfg.PARAMS['baseline_climate'] not in ['', 'CUSTOM']:
        raise ValueError("When using custom climate data please set "
                         "PARAMS['baseline_climate'] to an empty string "
                         "or `CUSTOM`. Note that you can now use the "
                         "`process_histalp_data` task for automated HISTALP "
                         "data processing.")

    # read the file
    fpath = cfg.PATHS['climate_file']
    nc_ts = salem.GeoNetcdf(fpath)

    # set temporal subset for the ts data (hydro years)
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    em = sm - 1 if (sm > 1) else 12
    yrs = nc_ts.time.year
    y0, y1 = yrs[0], yrs[-1]
    nc_ts.set_period(t0='{}-{:02d}-01'.format(y0, sm),
                     t1='{}-{:02d}-01'.format(y1, em))
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

    ilon = np.argmin(np.abs(lon - gdir.cenlon))
    ilat = np.argmin(np.abs(lat - gdir.cenlat))
    ref_pix_lon = lon[ilon]
    ref_pix_lat = lat[ilat]

    # read the data
    temp = nc_ts.get_vardata('temp')
    prcp = nc_ts.get_vardata('prcp')
    hgt = nc_ts.get_vardata('hgt')
    ttemp = temp[:, ilat-1:ilat+2, ilon-1:ilon+2]
    itemp = ttemp[:, 1, 1]
    thgt = hgt[ilat-1:ilat+2, ilon-1:ilon+2]
    ihgt = thgt[1, 1]
    thgt = thgt.flatten()
    iprcp = prcp[:, ilat, ilon]
    nc_ts.close()

    # Should we compute the gradient?
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    igrad = None
    if use_grad:
        igrad = np.zeros(len(time)) * np.NaN
        for t, loct in enumerate(ttemp):
            slope, _, _, p_val, _ = stats.linregress(thgt,
                                                     loct.flatten())
            igrad[t] = slope if (p_val < 0.01) else np.NaN

    gdir.write_monthly_climate_file(time, iprcp, itemp, ihgt,
                                    ref_pix_lon, ref_pix_lat,
                                    gradient=igrad)
    # metadata
    out = {'baseline_climate_source': fpath,
           'baseline_hydro_yr_0': y0+1,
           'baseline_hydro_yr_1': y1}
    gdir.write_pickle(out, 'climate_info')


@entity_task(log, writes=['cesm_data'])
def process_cesm_data(gdir, filesuffix='', fpath_temp=None, fpath_precc=None,
                      fpath_precl=None):
    """Processes and writes the climate data for this glacier.

    This function is made for interpolating the Community
    Earth System Model Last Millenial Ensemble (CESM-LME) climate simulations,
    from Otto-Bliesner et al. (2016), to the high-resolution CL2 climatologies
    (provided with OGGM) and writes everything to a NetCDF file.

    Parameters
    ----------
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    fpath_temp : str
        path to the temp file (default: cfg.PATHS['gcm_temp_file'])
    fpath_precc : str
        path to the precc file (default: cfg.PATHS['gcm_precc_file'])
    fpath_precl : str
        path to the precl file (default: cfg.PATHS['gcm_precl_file'])
    """

    # GCM temperature and precipitation data
    if fpath_temp is None:
        if not ('gcm_temp_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['gcm_temp_file']")
        fpath_temp = cfg.PATHS['gcm_temp_file']
    if fpath_precc is None:
        if not ('gcm_precc_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['gcm_precc_file']")
        fpath_precc = cfg.PATHS['gcm_precc_file']
    if fpath_precl is None:
        if not ('gcm_precl_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['gcm_precl_file']")
        fpath_precl = cfg.PATHS['gcm_precl_file']

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
    precp = precpcds.PRECC.sel(lat=lat, lon=lon, method='nearest') + \
            preclpds.PRECL.sel(lat=lat, lon=lon, method='nearest')

    # from normal years to hydrological years
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    em = sm - 1 if (sm > 1) else 12
    # TODO: we don't check if the files actually start in January but we should
    precp = precp[sm-1:sm-13].load()
    temp = temp[sm-1:sm-13].load()
    y0 = int(temp.time.values[0].strftime('%Y'))
    y1 = int(temp.time.values[-1].strftime('%Y'))
    time = pd.period_range('{}-{:02d}'.format(y0, sm),
                           '{}-{:02d}'.format(y1, em), freq='M')
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
    ndays = np.tile(np.roll(cfg.DAYS_IN_MONTH, 13-sm), y1 - y0)
    precp = precp * ndays * (60 * 60 * 24 * 1000)

    # compute monthly anomalies
    # of temp
    ts_tmp_avg = temp.sel(time=(temp.year >= 1961) & (temp.year <= 1990))
    ts_tmp_avg = ts_tmp_avg.groupby(ts_tmp_avg.month).mean(dim='time')
    ts_tmp = temp.groupby(temp.month) - ts_tmp_avg
    # of precip -- scaled anomalies
    ts_pre_avg = precp.isel(time=(precp.year >= 1961) & (precp.year <= 1990))
    ts_pre_avg = ts_pre_avg.groupby(ts_pre_avg.month).mean(dim='time')
    ts_pre_ano = precp.groupby(precp.month) - ts_pre_avg
    # scaled anomalies is the default. Standard anomalies above
    # are used later for where ts_pre_avg == 0
    ts_pre = precp.groupby(precp.month) / ts_pre_avg

    # Get CRU to apply the anomaly to
    fpath = gdir.get_filepath('climate_monthly')
    ds_cru = xr.open_dataset(fpath)

    # Add climate anomaly to CRU clim
    dscru = ds_cru.sel(time=slice('1961', '1990'))
    # for temp
    loc_tmp = dscru.temp.groupby('time.month').mean()
    ts_tmp = ts_tmp.groupby(ts_tmp.month) + loc_tmp
    # for prcp
    loc_pre = dscru.prcp.groupby('time.month').mean()
    # scaled anomalies
    ts_pre = ts_pre.groupby(ts_pre.month) * loc_pre
    # standard anomalies
    ts_pre_ano = ts_pre_ano.groupby(ts_pre_ano.month) + loc_pre
    # Correct infinite values with standard anomalies
    ts_pre.values = np.where(np.isfinite(ts_pre.values),
                             ts_pre.values,
                             ts_pre_ano.values)
    # The last step might create negative values (unlikely). Clip them
    ts_pre.values = ts_pre.values.clip(0)

    # load dates in right format to save
    dsindex = salem.GeoNetcdf(fpath_temp, monthbegin=True)
    time1 = dsindex.variables['time']
    time2 = time1[sm-1:sm-13] - ndays  # to hydrological years
    time2 = netCDF4.num2date(time2, time1.units, calendar='noleap')

    assert np.all(np.isfinite(ts_pre.values))
    assert np.all(np.isfinite(ts_tmp.values))

    # back to -180 - 180
    loc_lon = precp.lon if precp.lon <= 180 else precp.lon - 360

    gdir.write_monthly_climate_file(time2, ts_pre.values, ts_tmp.values,
                                    float(dscru.ref_hgt),
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

    if cfg.PARAMS['baseline_climate'] != 'CRU':
        raise ValueError("cfg.PARAMS['baseline_climate'] should be set to CRU")

    # read the climatology
    clfile = utils.get_cru_cl_file()
    ncclim = salem.GeoNetcdf(clfile)
    # and the TS data
    nc_ts_tmp = salem.GeoNetcdf(utils.get_cru_file('tmp'), monthbegin=True)
    nc_ts_pre = salem.GeoNetcdf(utils.get_cru_file('pre'), monthbegin=True)

    # set temporal subset for the ts data (hydro years)
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    em = sm - 1 if (sm > 1) else 12
    yrs = nc_ts_pre.time.year
    y0, y1 = yrs[0], yrs[-1]
    if cfg.PARAMS['baseline_y0'] != 0:
        y0 = cfg.PARAMS['baseline_y0']
    if cfg.PARAMS['baseline_y1'] != 0:
        y1 = cfg.PARAMS['baseline_y1']

    nc_ts_tmp.set_period(t0='{}-{:02d}-01'.format(y0, sm),
                         t1='{}-{:02d}-01'.format(y1, em))
    nc_ts_pre.set_period(t0='{}-{:02d}-01'.format(y0, sm),
                         t1='{}-{:02d}-01'.format(y1, em))
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

    # Should we compute the gradient?
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    ts_grad = None
    if use_grad and len(hgt_f) >= 5:
        ts_grad = np.zeros(12) * np.NaN
        for i in range(12):
            loc_tmp_mth = loc_tmp[i, ...][isok].flatten()
            slope, _, _, p_val, _ = stats.linregress(hgt_f, loc_tmp_mth)
            ts_grad[i] = slope if (p_val < 0.01) else np.NaN
        # convert to a timeseries and hydrological years
        ts_grad = ts_grad.tolist()
        ts_grad = ts_grad[em:] + ts_grad[0:em]
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
            raise RuntimeError(msg)
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
    ts_pre.values = ts_pre.values.clip(0)

    # done
    loc_hgt = loc_hgt[1, 1]
    loc_lon = loc_lon[1]
    loc_lat = loc_lat[1]
    assert np.isfinite(loc_hgt)
    assert np.all(np.isfinite(ts_pre.values))
    assert np.all(np.isfinite(ts_tmp.values))

    gdir.write_monthly_climate_file(time, ts_pre.values, ts_tmp.values,
                                    loc_hgt, loc_lon, loc_lat,
                                    gradient=ts_grad)

    source = nc_ts_tmp._nc.title[:10]
    ncclim._nc.close()
    nc_ts_tmp._nc.close()
    nc_ts_pre._nc.close()
    # metadata
    out = {'baseline_climate_source': source,
           'baseline_hydro_yr_0': y0+1,
           'baseline_hydro_yr_1': y1}
    gdir.write_pickle(out, 'climate_info')


@entity_task(log, writes=['climate_monthly'])
def process_histalp_data(gdir):
    """Processes and writes the climate data for this glacier.

    Extracts the nearest timeseries and writes everything to a NetCDF file.
    """

    if cfg.PARAMS['baseline_climate'] != 'HISTALP':
        raise ValueError("cfg.PARAMS['baseline_climate'] should be set to "
                         "HISTALP.")

    # read the time out of the pure netcdf file
    ft = utils.get_histalp_file('tmp')
    fp = utils.get_histalp_file('pre')
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

    # set temporal subset for the ts data (hydro years)
    # the reference time is given by precip, which is shorter
    sm = cfg.PARAMS['hydro_month_nh']
    em = sm - 1 if (sm > 1) else 12
    yrs = nc_ts_pre.time.year
    y0, y1 = yrs[0], yrs[-1]
    if cfg.PARAMS['baseline_y0'] != 0:
        y0 = cfg.PARAMS['baseline_y0']
    if cfg.PARAMS['baseline_y1'] != 0:
        y1 = cfg.PARAMS['baseline_y1']

    nc_ts_tmp.set_period(t0='{}-{:02d}-01'.format(y0, sm),
                         t1='{}-{:02d}-01'.format(y1, em))
    nc_ts_pre.set_period(t0='{}-{:02d}-01'.format(y0, sm),
                         t1='{}-{:02d}-01'.format(y1, em))
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

    # Should we compute the gradient?
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    igrad = None
    if use_grad:
        igrad = np.zeros(len(time)) * np.NaN
        for t, loct in enumerate(temp):
            slope, _, _, p_val, _ = stats.linregress(hgt.flatten(),
                                                     loct.flatten())
            igrad[t] = slope if (p_val < 0.01) else np.NaN

    gdir.write_monthly_climate_file(time, prcp[:, 1, 1], temp[:, 1, 1],
                                    hgt[1, 1], ref_lon[1], ref_lat[1],
                                    gradient=igrad)
    # metadata
    out = {'baseline_climate_source': source,
           'baseline_hydro_yr_0': y0 + 1,
           'baseline_hydro_yr_1': y1}
    gdir.write_pickle(out, 'climate_info')


def mb_climate_on_height(gdir, heights, *, time_range=None, year_range=None):
    """Mass-balance climate of the glacier at a specific height

    Reads the glacier's monthly climate data file and computes the
    temperature "energies" (temp above 0) and solid precipitation at the
    required height.

    Parameters
    ----------
    gdir : GlacierDirectory
        the glacier directory
    heights: ndarray
        a 1D array of the heights (in meter) where you want the data
    time_range : [datetime, datetime], optional
        default is to read all data but with this you
        can provide a [t0, t1] bounds (inclusive).
    year_range : [int, int], optional
        Provide a [y0, y1] year range to get the data for specific
        (hydrological) years only. Easier to us than the time bounds above.

    Returns
    -------
    (time, tempformelt, prcpsol)::
        - time: array of shape (nt,)
        - tempformelt:  array of shape (len(heights), nt)
        - prcpsol:  array of shape (len(heights), nt)
    """

    if year_range is not None:
        sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
        em = sm - 1 if (sm > 1) else 12
        t0 = datetime.datetime(year_range[0]-1, sm, 1)
        t1 = datetime.datetime(year_range[1], em, 1)
        return mb_climate_on_height(gdir, heights, time_range=[t0, t1])

    # Parameters
    temp_all_solid = cfg.PARAMS['temp_all_solid']
    temp_all_liq = cfg.PARAMS['temp_all_liq']
    temp_melt = cfg.PARAMS['temp_melt']
    prcp_fac = cfg.PARAMS['prcp_scaling_factor']
    default_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    # Read file
    igrad = None
    with utils.ncDataset(gdir.get_filepath('climate_monthly'), mode='r') as nc:
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
        if 'gradient' in nc.variables:
            igrad = nc.variables['gradient'][p0:p1+1]
            # Security for stuff that can happen with local gradients
            igrad = np.where(~np.isfinite(igrad), default_grad, igrad)
            igrad = np.clip(igrad, g_minmax[0], g_minmax[1])
        ref_hgt = nc.ref_hgt

    # Default gradient?
    if igrad is None:
        igrad = itemp * 0 + default_grad

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


def mb_yearly_climate_on_height(gdir, heights, *,
                                year_range=None, flatten=False):
    """Yearly mass-balance climate of the glacier at a specific height

    The precipitation time series are not corrected!

    Parameters
    ----------
    gdir : GlacierDirectory
        the glacier directory
    heights: ndarray
        a 1D array of the heights (in meter) where you want the data
    year_range : [int, int], optional
        Provide a [y0, y1] year range to get the data for specific
        (hydrological) years only.
    flatten : bool
        for some applications (glacier average MB) it's ok to flatten the
        data (average over height) prior to annual summing.

    Returns
    -------
    (years, tempformelt, prcpsol)::
        - years: array of shape (ny,)
        - tempformelt:  array of shape (len(heights), ny) (or ny if flatten
        is set)
        - prcpsol:  array of shape (len(heights), ny) (or ny if flatten
        is set)
    """

    time, temp, prcp = mb_climate_on_height(gdir, heights,
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


def mb_yearly_climate_on_glacier(gdir, *, year_range=None):
    """Yearly mass-balance climate at all glacier heights,
    multiplied with the flowlines widths. (all in pix coords.)

    The precipitation time series are not corrected!

    Parameters
    ----------
    gdir : GlacierDirectory
        the glacier directory
    year_range : [int, int], optional
        Provide a [y0, y1] year range to get the data for specific
        (hydrological) years only.

    Returns
    -------
    (years, tempformelt, prcpsol)::
        - years: array of shape (ny,)
        - tempformelt:  array of shape (len(heights), ny)
        - prcpsol:  array of shape (len(heights), ny) (not corrected!)
    """

    flowlines = gdir.read_pickle('inversion_flowlines')

    heights = np.array([])
    widths = np.array([])
    for fl in flowlines:
        heights = np.append(heights, fl.surface_h)
        widths = np.append(widths, fl.widths)

    years, temp, prcp = mb_yearly_climate_on_height(gdir, heights,
                                                    year_range=year_range,
                                                    flatten=False)

    temp = np.average(temp, axis=0, weights=widths)
    prcp = np.average(prcp, axis=0, weights=widths)

    return years, temp, prcp


@entity_task(log, writes=['climate_info'])
def mu_candidates(gdir):
    """Computes the mu candidates.

    For each 31 year-period centered on the year of interest, mu is is the
    temperature sensitivity necessary for the glacier with its current shape
    to be in equilibrium with its climate.

    For glaciers with MB data only!

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])

    # Only get the years were we consider looking for tstar
    y0, y1 = cfg.PARAMS['tstar_search_window']
    ci = gdir.read_pickle('climate_info')
    y0 = y0 or ci['baseline_hydro_yr_0']
    y1 = y1 or ci['baseline_hydro_yr_1']

    years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir,
                                                           year_range=[y0, y1])

    # Be sure we have no marine terminating glacier
    assert gdir.terminus_type == 'Land-terminating'

    # Compute mu for each 31-yr climatological period
    ny = len(years)
    mu_yr_clim = np.zeros(ny) * np.NaN
    for i, y in enumerate(years):
        # Ignore begin and end
        if ((i-mu_hp) < 0) or ((i+mu_hp) >= ny):
            continue
        t_avg = np.mean(temp_yr[i-mu_hp:i+mu_hp+1])
        if t_avg > 1e-3:  # if too cold no melt possible
            prcp_ts = prcp_yr[i-mu_hp:i+mu_hp+1]
            mu_yr_clim[i] = np.mean(prcp_ts) / t_avg

    # Check that we found a least one mustar
    if np.sum(np.isfinite(mu_yr_clim)) < 1:
        raise RuntimeError('({}) no mustar candidates found.'
                           .format(gdir.rgi_id))

    # Write
    d = gdir.read_pickle('climate_info')
    d['mu_candidates'] = pd.Series(data=mu_yr_clim, index=years)
    gdir.write_pickle(d, 'climate_info')


def t_star_from_refmb(gdir, mbdf=None):
    """Computes the t* for the glacier, given a series of MB measurements.

    Could be multiprocessed but its probably not necessary.

    Parameters
    ----------
    gdirs: the list of oggm.GlacierDirectory objects where to write the data.
    mbdf: a pd.Series containing the observed MB data indexed by year
        if None, read automatically from the reference data

    Returns
    -------
    A dict: {t_star:[], bias:[]}
    """

    # Only divide 0, we believe the original RGI entities to be the ref...
    years, temp_yr_ts, prcp_yr_ts = mb_yearly_climate_on_glacier(gdir)

    if mbdf is None:
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

    # which years to look at
    selind = np.searchsorted(years, mbdf.index)
    temp_yr_ts = temp_yr_ts[selind]
    prcp_yr_ts = prcp_yr_ts[selind]
    temp_yr = np.mean(temp_yr_ts)
    prcp_yr = np.mean(prcp_yr_ts)

    # Average oberved mass-balance
    ref_mb = np.mean(mbdf)

    # Average mass-balance per mu
    mu_yr_clim = gdir.read_pickle('climate_info')['mu_candidates']
    mb_per_mu = prcp_yr - mu_yr_clim * temp_yr

    # Diff to reference
    diff = (mb_per_mu - ref_mb).dropna()

    # Here we used to keep all possible mu* in order to later select
    # them based on some distance search algorithms.
    # (revision 81bc0923eab6301306184d26462f932b72b84117)
    #
    # As of Jul 2018, we will now stop this non-sense:
    # out of all mu*, let's just pick the one with the smallest bias.
    # It doesn't make much sense, but the same is true for other methods
    # as well -> this is how Ben used to do it, and he is clever
    amin = np.abs(diff).idxmin()
    out = {
        't_star': amin,
        'bias': diff[amin],
    }
    return out


def calving_mb(gdir):
    """Calving mass-loss in specific MB equivalent.

    This is necessary to compute mu star.
    """

    if not gdir.is_tidewater:
        return 0.

    # Ok. Just take the caving rate from cfg and change its units
    # Original units: km3 a-1, to change to mm a-1 (units of specific MB)
    return gdir.inversion_calving_rate * 1e9 * cfg.RHO / gdir.rgi_area_m2


@entity_task(log, writes=['local_mustar'])
def local_mustar(gdir, *, ref_df=None,
                 tstar=None, bias=None, minimum_mustar=0.):
    """Compute the local mustar from interpolated tstars.

    If tstar and bias are mot provided they will be interpolated from the
    reference file.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    ref_df : pd.Dataframe, optional
        replace the default calibration list with your own.
    tstar: int, optional
        the year where the glacier should be equilibrium
    bias: float, optional
        the associated reference bias
    minimum_mustar: float, optional
        if mustar goes below this threshold, clip it to that value.
        If you want this to happen with `minimum_mustar=0.` you will have
        to set `cfg.PARAMS['allow_negative_mustar']=True` first.
    """

    # Relevant mb params
    params = ['temp_default_gradient', 'temp_all_solid', 'temp_all_liq',
              'temp_melt', 'prcp_scaling_factor']

    if tstar is None or bias is None:
        # Do our own interpolation
        if ref_df is None:
            if not cfg.PARAMS['run_mb_calibration']:
                # Make some checks and use the default one
                climate_info = gdir.read_pickle('climate_info')
                source = climate_info['baseline_climate_source']
                ok_source = ['CRU TS4.01', 'CRU TS3.23', 'HISTALP']
                if not np.any(s in source.upper() for s in ok_source):
                    raise RuntimeError('If you are using a custom climate '
                                       'file you should run your own MB '
                                       'calibration.')
                v = gdir.rgi_version[0]  # major version relevant

                # Check that the params are fine
                str_s = 'cru4' if 'CRU' in source else 'histalp'
                vn = 'ref_tstars_rgi{}_{}_calib_params'.format(v, str_s)
                for k in params:
                    if cfg.PARAMS[k] != cfg.PARAMS[vn][k]:
                        raise ValueError('The reference t* you are trying '
                                         'to use was calibrated with '
                                         'difference MB parameters. You '
                                         'might have to run the calibration '
                                         'manually.')
                ref_df = cfg.PARAMS['ref_tstars_rgi{}_{}'.format(v, str_s)]
            else:
                # Use the the local calibration
                fp = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
                ref_df = pd.read_csv(fp)

        # Compute the distance to each glacier
        distances = utils.haversine(gdir.cenlon, gdir.cenlat,
                                    ref_df.lon, ref_df.lat)

        # Take the 10 closest
        aso = np.argsort(distances)[0:9]
        amin = ref_df.iloc[aso]
        distances = distances[aso]**2

        # If really close no need to divide, else weighted average
        if distances.iloc[0] <= 0.1:
            tstar = amin.tstar.iloc[0]
            bias = amin.bias.iloc[0]
        else:
            tstar = int(np.average(amin.tstar, weights=1./distances))
            bias = np.average(amin.bias, weights=1./distances)

    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr = [tstar-mu_hp, tstar+mu_hp]

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)

    log.info('(%s) local mu* for t*=%d', gdir.rgi_id, tstar)

    # Get the corresponding mu
    years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir, year_range=yr)
    assert len(years) == (2 * mu_hp + 1)

    # mustar is taking calving into account (units of specific MB)
    mustar = (np.mean(prcp_yr) - cmb) / np.mean(temp_yr)
    if not np.isfinite(mustar):
        raise RuntimeError('{} has a non finite mu'.format(gdir.rgi_id))
    if not cfg.PARAMS['allow_negative_mustar']:
        if mustar < 0:
            raise RuntimeError('{} has a negative mu'.format(gdir.rgi_id))
    # For the calving param it might be useful to clip the mu
    mustar = np.clip(mustar, minimum_mustar, np.max(mustar))

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around with out calibration
    out = gdir.read_pickle('climate_info')
    out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in params}
    gdir.write_pickle(out, 'climate_info')

    # Scalars in a small dataframe for later
    df = pd.DataFrame()
    df['rgi_id'] = [gdir.rgi_id]
    df['t_star'] = [tstar]
    df['mu_star'] = [mustar]
    df['bias'] = [bias]
    df.to_csv(gdir.get_filepath('local_mustar'), index=False)


@entity_task(log, writes=['inversion_flowlines'])
def apparent_mb(gdir):
    """Compute the apparent mb from the calibrated mustar.

    Parameters
    ----------
    """

    # Calibrated data
    df = pd.read_csv(gdir.get_filepath('local_mustar')).iloc[0]
    tstar = df['t_star']
    mu_star = df['mu_star']
    bias = df['bias']

    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr = [tstar-mu_hp, tstar+mu_hp]

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')

    # Reset flux
    for fl in fls:
        fl.flux = np.zeros(len(fl.surface_h))

    # Flowlines in order to be sure
    for fl in fls:
        y, t, p = mb_yearly_climate_on_height(gdir, fl.surface_h,
                                              year_range=yr,
                                              flatten=False)
        fl.set_apparent_mb(np.mean(p, axis=1) - mu_star*np.mean(t, axis=1))

    # Sometimes, low lying tributaries have a non-physically consistent
    # Mass-balance. We should remove these, and start all over again until
    # all tributaries are consistent
    do_filter = [fl.flux_needed_correction for fl in fls]
    if cfg.PARAMS['filter_for_neg_flux'] and np.any(do_filter):
        assert not do_filter[-1]  # This should not happen
        # Keep only the good lines
        heads = [fl.orig_head for fl in fls if not fl.flux_needed_correction]
        centerlines.compute_centerlines(gdir, heads=heads, reset=True)
        centerlines.initialize_flowlines(gdir, reset=True)
        if gdir.has_file('downstream_line'):
            centerlines.compute_downstream_line(gdir, reset=True)
            centerlines.compute_downstream_bedshape(gdir, reset=True)
        centerlines.catchment_area(gdir, reset=True)
        centerlines.catchment_intersections(gdir, reset=True)
        centerlines.catchment_width_geom(gdir, reset=True)
        centerlines.catchment_width_correction(gdir, reset=True)
        local_mustar(gdir, tstar=tstar, bias=bias, reset=True)
        # Ok, re-call ourselves
        return apparent_mb(gdir, reset=True)

    # Check and write
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
    gdir.write_pickle(fls, 'inversion_flowlines')


@entity_task(log, writes=['inversion_flowlines', 'linear_mb_params'])
def apparent_mb_from_linear_mb(gdir, mb_gradient=3.):
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
    h, w = gdir.get_inversion_flowline_hw()

    # Now find the ELA till the integrated mb is zero
    from oggm.core.massbalance import LinearMassBalance
    def to_minimize(ela_h):
        mbmod = LinearMassBalance(ela_h[0], grad=mb_gradient)
        smb = mbmod.get_specific_mb(h, w)
        return (smb - cmb)**2

    ela_h = optimization.minimize(to_minimize, [0.], bounds=((0, 10000), ))
    ela_h = ela_h['x'][0]
    mbmod = LinearMassBalance(ela_h, grad=mb_gradient)

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')

    # Reset flux
    for fl in fls:
        fl.flux = np.zeros(len(fl.surface_h))

    # Flowlines in order to be sure
    for fl in fls:
        mbz = mbmod.get_annual_mb(fl.surface_h) * cfg.SEC_IN_YEAR * cfg.RHO
        fl.set_apparent_mb(mbz)

    # Check and write
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
    gdir.write_pickle(fls, 'inversion_flowlines')
    gdir.write_pickle({'ela_h': ela_h, 'grad': mb_gradient},
                      'linear_mb_params')


@global_task
def compute_ref_t_stars(gdirs):
    """ Detects the best t* for the reference glaciers.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    if not cfg.PARAMS['run_mb_calibration']:
        raise RuntimeError('Are you sure you want to calibrate the reference '
                           't*? There is a pre-calibrated version available. '
                           'If you know what you are doing and still want to '
                           'calibrate, set the `run_mb_calibration` parameter '
                           'to `True`.')

    log.info('Compute the reference t* and mu* for WGMS glaciers')

    # Reference glaciers only if in the list and period is good
    ref_gdirs = utils.get_ref_mb_glaciers(gdirs)

    # Loop
    df = pd.DataFrame()
    for gdir in ref_gdirs:
        # all possible mus
        mu_candidates(gdir, reset=True)
        # list of mus compatibles with refmb
        res = t_star_from_refmb(gdir)
        rid = gdir.rgi_id
        df.loc[rid, 'lon'] = gdir.cenlon
        df.loc[rid, 'lat'] = gdir.cenlat
        df.loc[rid, 'n_mb_years'] = len(gdir.get_ref_mb_data())
        df.loc[rid, 'tstar'] = res['t_star']
        df.loc[rid, 'bias'] = res['bias']

    # Write out
    df['tstar'] = df['tstar'].astype(int)
    df['n_mb_years'] = df['n_mb_years'].astype(int)
    file = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
    df.sort_index().to_csv(file)


@global_task
def distribute_t_stars(gdirs, ref_df=None, minimum_mustar=0.):
    """After the computation of the reference tstars, apply
    the interpolation to each individual glacier.

    Parameters
    ----------
    gdirs : []
        list of oggm.GlacierDirectory objects
    ref_df : pd.Dataframe
        replace the default calibration list
    minimum_mustar: float
        if mustar goes below this threshold, clip it to that value.
        If you want this to happen with `minimum_mustar=0.` you will have
        to set `cfg.PARAMS['allow_negative_mustar']=True` first.
    """

    warnings.warn('The global task `distribute_t_stars` is deprecated. Use '
                  'a direct call to the entity task `local_mustar` instead.',
                  DeprecationWarning)

    log.info('Distribute t* and mu*')

    if ref_df is None:
        fp = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
        if not cfg.PARAMS['run_mb_calibration']:
            # Make some checks and use the default one
            if (('climate_file' in cfg.PATHS) and
                    os.path.exists(cfg.PATHS['climate_file'])):
                raise RuntimeError('If you are using a custom climate file '
                                   'you should run your own MB calibration.')
            v = gdirs[0].rgi_version[0]  # major version relevant
            fn = 'oggm_ref_tstars_rgi{}_cru4.csv'.format(v)
            fp = utils.get_demo_file(fn)
        ref_df = pd.read_csv(fp)

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
            bias = amin.bias.iloc[0]
        else:
            tstar = int(np.average(amin.tstar, weights=1./distances))
            bias = np.average(amin.bias, weights=1./distances)

        # Go
        local_mustar(gdir, tstar=tstar, bias=bias,
                     minimum_mustar=minimum_mustar,
                     reset=True)


@global_task
def crossval_t_stars(gdirs):
    """Cross-validate the interpolation of tstar to each individual glacier.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    log.info('Cross-validate the t* and mu* determination')

    rgdirs = utils.get_ref_mb_glaciers(gdirs)

    full_ref_df = pd.read_csv(os.path.join(cfg.PATHS['working_dir'],
                                           'ref_tstars.csv'), index_col=0)

    n = len(full_ref_df)
    for i, rid in enumerate(full_ref_df.index):

        log.info('Cross-validation iteration {} of {}'.format(i+1, n))

        # the glacier to look at
        gdir = [g for g in rgdirs if g.rgi_id == rid][0]

        # the reference glaciers
        tmp_ref_df = full_ref_df.loc[full_ref_df.index != rid]

        # before the cross-val we can get the info about "real" mustar
        rdf = pd.read_csv(gdir.get_filepath('local_mustar'))
        full_ref_df.loc[rid, 'mustar'] = rdf['mu_star'].values[0]

        # redo the computations
        with utils.DisableLogger():
            local_mustar(gdir, ref_df=tmp_ref_df)

        # store
        rdf = pd.read_csv(gdir.get_filepath('local_mustar'))
        full_ref_df.loc[rid, 'cv_tstar'] = int(rdf['t_star'].values[0])
        full_ref_df.loc[rid, 'cv_mustar'] = rdf['mu_star'].values[0]
        full_ref_df.loc[rid, 'cv_bias'] = rdf['bias'].values[0]

        # let's make sure we don't brake things
        with utils.DisableLogger():
            local_mustar(gdir, ref_df=full_ref_df)

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
