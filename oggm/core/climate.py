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
from oggm.exceptions import MassBalanceCalibrationError, InvalidParamsError

# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['climate_monthly', 'climate_info'])
def process_custom_climate_data(gdir):
    """Processes and writes the climate data from a user-defined climate file.

    The input file must have a specific format (see
    https://github.com/OGGM/oggm-sample-data test-files/histalp_merged_hef.nc
    for an example).

    This is the way OGGM used to do it for HISTALP before it got automatised.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    if not (('climate_file' in cfg.PATHS) and
            os.path.exists(cfg.PATHS['climate_file'])):
        raise InvalidParamsError('Custom climate file not found')

    if cfg.PARAMS['baseline_climate'] not in ['', 'CUSTOM']:
        raise InvalidParamsError("When using custom climate data please set "
                                 "PARAMS['baseline_climate'] to an empty "
                                 "string or `CUSTOM`. Note also that you can "
                                 "now use the `process_histalp_data` task for "
                                 "automated HISTALP data processing.")

    # read the file
    fpath = cfg.PATHS['climate_file']
    nc_ts = salem.GeoNetcdf(fpath)

    # set temporal subset for the ts data (hydro years)
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    em = sm - 1 if (sm > 1) else 12
    yrs = nc_ts.time.year
    y0, y1 = yrs[0], yrs[-1]
    if cfg.PARAMS['baseline_y0'] != 0:
        y0 = cfg.PARAMS['baseline_y0']
    if cfg.PARAMS['baseline_y1'] != 0:
        y1 = cfg.PARAMS['baseline_y1']
    nc_ts.set_period(t0='{}-{:02d}-01'.format(y0, sm),
                     t1='{}-{:02d}-01'.format(y1, em))
    time = nc_ts.time
    ny, r = divmod(len(time), 12)
    if r != 0:
        raise InvalidParamsError('Climate data should be full years')

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


@entity_task(log, writes=['climate_monthly', 'climate_info'])
def process_cru_data(gdir):
    """Processes and writes the CRU baseline climate data for this glacier.

    Interpolates the CRU TS data to the high-resolution CL2 climatologies
    (provided with OGGM) and writes everything to a NetCDF file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    if cfg.PATHS.get('climate_file', None):
        warnings.warn("You seem to have set a custom climate file for this "
                      "run, but are using the default CRU climate "
                      "file instead.")

    if cfg.PARAMS['baseline_climate'] != 'CRU':
        raise InvalidParamsError("cfg.PARAMS['baseline_climate'] should be "
                                 "set to CRU")

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


@entity_task(log, writes=['climate_monthly', 'climate_info'])
def process_dummy_cru_file(gdir, sigma_temp=2, sigma_prcp=0.5, seed=None):
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
    """

    # read the climatology
    clfile = utils.get_cru_cl_file()
    ncclim = salem.GeoNetcdf(clfile)

    # set temporal subset for the ts data (hydro years)
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    em = sm - 1 if (sm > 1) else 12

    y0, y1 = 1901, 2018
    if cfg.PARAMS['baseline_y0'] != 0:
        y0 = cfg.PARAMS['baseline_y0']
    if cfg.PARAMS['baseline_y1'] != 0:
        y1 = cfg.PARAMS['baseline_y1']

    time = pd.date_range(start='{}-{:02d}-01'.format(y0, sm),
                         end='{}-{:02d}-01'.format(y1, em),
                         freq='MS')
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

    # Make DataArrays
    rng = np.random.RandomState(seed)
    loc_tmp = xr.DataArray(loc_tmp[:, 1, 1], dims=['month'],
                           coords={'month': np.arange(1, 13)})
    ts_tmp = rng.randn(len(time)) * sigma_temp
    ts_tmp = xr.DataArray(ts_tmp, dims=['time'],
                          coords={'time': time})

    loc_pre = xr.DataArray(loc_pre[:, 1, 1], dims=['month'],
                           coords={'month': np.arange(1, 13)})
    ts_pre = (rng.randn(len(time)) * sigma_prcp + 1).clip(0)
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
                                    gradient=ts_grad)

    source = 'CRU CL2 and some randomness'
    ncclim._nc.close()
    # metadata
    out = {'baseline_climate_source': source,
           'baseline_hydro_yr_0': y0+1,
           'baseline_hydro_yr_1': y1}
    gdir.write_pickle(out, 'climate_info')


@entity_task(log, writes=['climate_monthly', 'climate_info'])
def process_histalp_data(gdir):
    """Processes and writes the HISTALP baseline climate data for this glacier.

    Extracts the nearest timeseries and writes everything to a NetCDF file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    if cfg.PATHS.get('climate_file', None):
        warnings.warn("You seem to have set a custom climate file for this "
                      "run, but are using the default HISTALP climate file "
                      "instead.")

    if cfg.PARAMS['baseline_climate'] != 'HISTALP':
        raise InvalidParamsError("cfg.PARAMS['baseline_climate'] should be "
                                 "set to HISTALP.")

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

    All MB parameters are considered here! (i.e. melt temp, precip scaling
    factor, etc.)

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
        (hydrological) years only. Easier to use than the time bounds above.

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
                raise MassBalanceCalibrationError('time_range[0] not found in '
                                                  'file')
            p1 = np.where(time == time_range[1])[0]
            try:
                p1 = p1[0]
            except IndexError:
                raise MassBalanceCalibrationError('time_range[1] not found in '
                                                  'file')
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

    See also: mb_climate_on_height

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
        raise InvalidParamsError('Climate data should be N full years '
                                 'exclusively')
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

    See also: mb_climate_on_height

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
        - years: array of shape (ny)
        - tempformelt:  array of shape (ny)
        - prcpsol:  array of shape (ny)
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
def glacier_mu_candidates(gdir):
    """Computes the mu candidates, glacier wide.

    For each 31 year-period centered on the year of interest, mu is is the
    temperature sensitivity necessary for the glacier with its current shape
    to be in equilibrium with its climate.

    This task is just for documentation and testing! It is not used in
    production anymore.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    warnings.warn('The task `glacier_mu_candidates` is deprecated. It should '
                  'only be used for testing.', DeprecationWarning)

    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])

    # Only get the years were we consider looking for tstar
    y0, y1 = cfg.PARAMS['tstar_search_window']
    ci = gdir.read_pickle('climate_info')
    y0 = y0 or ci['baseline_hydro_yr_0']
    y1 = y1 or ci['baseline_hydro_yr_1']

    years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir,
                                                           year_range=[y0, y1])

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
        raise MassBalanceCalibrationError('({}) no mustar candidates found.'
                                          .format(gdir.rgi_id))

    # Write
    ci['mu_candidates_glacierwide'] = pd.Series(data=mu_yr_clim, index=years)
    gdir.write_pickle(ci, 'climate_info')


@entity_task(log, writes=['climate_info'])
def t_star_from_refmb(gdir, mbdf=None, glacierwide=None,
                      write_diagnostics=False):
    """Computes the ref t* for the glacier, given a series of MB measurements.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    mbdf: a pd.Series containing the observed MB data indexed by year
        if None, read automatically from the reference data

    Returns
    -------
    A dict: {t_star:[], bias:[]}
    """

    from oggm.core.massbalance import MultipleFlowlineMassBalance

    if glacierwide is None:
        glacierwide = cfg.PARAMS['tstar_search_glacierwide']

    # Be sure we have no marine terminating glacier
    assert gdir.terminus_type == 'Land-terminating'

    # Reference time series
    if mbdf is None:
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

    # which years to look at
    ref_years = mbdf.index.values

    # Average oberved mass-balance
    ref_mb = np.mean(mbdf)

    # Compute one mu candidate per year and the associated statistics
    # Only get the years were we consider looking for tstar
    y0, y1 = cfg.PARAMS['tstar_search_window']
    ci = gdir.read_pickle('climate_info')
    y0 = y0 or ci['baseline_hydro_yr_0']
    y1 = y1 or ci['baseline_hydro_yr_1']
    years = np.arange(y0, y1+1)

    ny = len(years)
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    mb_per_mu = pd.Series(index=years)

    if glacierwide:
        # The old (but fast) method to find t*
        _, temp, prcp = mb_yearly_climate_on_glacier(gdir, year_range=[y0, y1])

        # which years to look at
        selind = np.searchsorted(years, mbdf.index)
        sel_temp = temp[selind]
        sel_prcp = prcp[selind]
        sel_temp = np.mean(sel_temp)
        sel_prcp = np.mean(sel_prcp)

        for i, y in enumerate(years):

            # Ignore begin and end
            if ((i - mu_hp) < 0) or ((i + mu_hp) >= ny):
                continue

            # Compute the mu candidate
            t_avg = np.mean(temp[i - mu_hp:i + mu_hp + 1])
            if t_avg < 1e-3:  # if too cold no melt possible
                continue
            mu = np.mean(prcp[i - mu_hp:i + mu_hp + 1]) / t_avg

            # Apply it
            mb_per_mu[y] = np.mean(sel_prcp - mu * sel_temp)

    else:
        # The new (but slow) method to find t*
        # Compute mu for each 31-yr climatological period
        fls = gdir.read_pickle('inversion_flowlines')
        for i, y in enumerate(years):
            # Ignore begin and end
            if ((i-mu_hp) < 0) or ((i+mu_hp) >= ny):
                continue
            # Calibrate the mu for this year
            for fl in fls:
                fl.mu_star_is_valid = False
            try:
                # TODO: this is slow and can be highly optimised
                # it reads the same data over and over again
                _recursive_mu_star_calibration(gdir, fls, y, first_call=True)
                # Compute the MB with it
                mb_mod = MultipleFlowlineMassBalance(gdir, fls, bias=0,
                                                     check_calib_params=False)
                mb_ts = mb_mod.get_specific_mb(fls=fls, year=ref_years)
                mb_per_mu[y] = np.mean(mb_ts)
            except MassBalanceCalibrationError:
                pass

    # Diff to reference
    diff = (mb_per_mu - ref_mb).dropna()

    if len(diff) == 0:
        raise MassBalanceCalibrationError('No single valid mu candidate for '
                                          'this glacier!')

    # Here we used to keep all possible mu* in order to later select
    # them based on some distance search algorithms.
    # (revision 81bc0923eab6301306184d26462f932b72b84117)
    #
    # As of Jul 2018, we will now stop this non-sense:
    # out of all mu*, let's just pick the one with the smallest bias.
    # It doesn't make much sense, but the same is true for other methods
    # as well -> this is how Ben used to do it, and he is clever
    # Another way would be to pick the closest to today or something
    amin = np.abs(diff).idxmin()

    # Write
    d = gdir.read_pickle('climate_info')
    d['t_star'] = amin
    d['bias'] = diff[amin]
    if write_diagnostics:
        d['avg_mb_per_mu'] = mb_per_mu
        d['avg_ref_mb'] = ref_mb

    gdir.write_pickle(d, 'climate_info')

    return {'t_star': amin, 'bias': diff[amin]}


def calving_mb(gdir):
    """Calving mass-loss in specific MB equivalent.

    This is necessary to compute mu star.
    """

    if not gdir.is_tidewater:
        return 0.

    # Ok. Just take the calving rate from cfg and change its units
    # Original units: km3 a-1, to change to mm a-1 (units of specific MB)
    rho = cfg.PARAMS['ice_density']
    return gdir.inversion_calving_rate * 1e9 * rho / gdir.rgi_area_m2


def _fallback_local_t_star(gdir):
    """A Fallback function if climate.local_t_star raises an Error.

    This function will still write a `local_mustar.json`, filled with NANs,
    if climate.local_t_star fails and cfg.PARAMS['continue_on_error'] = True.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    """
    # Scalars in a small dict for later
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = np.nan
    df['bias'] = np.nan
    df['mu_star_glacierwide'] = np.nan
    gdir.write_json(df, 'local_mustar')


@entity_task(log, writes=['local_mustar'], fallback=_fallback_local_t_star)
def local_t_star(gdir, *, ref_df=None, tstar=None, bias=None):
    """Compute the local t* and associated glacier-wide mu*.

    If ``tstar`` and ``bias`` are not provided, they will be interpolated from
    the reference t* list.

    Note: the glacier wide mu* is here just for indication. It might be
    different from the flowlines' mu* in some cases.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ref_df : :py:class:`pandas.DataFrame`, optional
        replace the default calibration list with your own.
    tstar: int, optional
        the year where the glacier should be equilibrium
    bias: float, optional
        the associated reference bias
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
                    msg = ('If you are using a custom climate file you should '
                           'run your own MB calibration.')
                    raise MassBalanceCalibrationError(msg)
                v = gdir.rgi_version[0]  # major version relevant

                # Check that the params are fine
                str_s = 'cru4' if 'CRU' in source else 'histalp'
                vn = 'ref_tstars_rgi{}_{}_calib_params'.format(v, str_s)
                for k in params:
                    if cfg.PARAMS[k] != cfg.PARAMS[vn][k]:
                        msg = ('The reference t* you are trying to use was '
                               'calibrated with different MB parameters. You '
                               'might have to run the calibration manually.')
                        raise MassBalanceCalibrationError(msg)
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

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    out = gdir.read_pickle('climate_info')
    out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in params}
    gdir.write_pickle(out, 'climate_info')

    # We compute the overall mu* here but this is mostly for testing
    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr = [tstar - mu_hp, tstar + mu_hp]

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)

    log.info('(%s) local mu* computation for t*=%d', gdir.rgi_id, tstar)

    # Get the corresponding mu
    years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir, year_range=yr)
    assert len(years) == (2 * mu_hp + 1)

    # mustar is taking calving into account (units of specific MB)
    mustar = (np.mean(prcp_yr) - cmb) / np.mean(temp_yr)
    if not np.isfinite(mustar):
        raise MassBalanceCalibrationError('{} has a non finite '
                                          'mu'.format(gdir.rgi_id))

    # Clip it?
    if cfg.PARAMS['clip_mu_star']:
        mustar = np.clip(mustar, 0, None)

    # If mu out of bounds, raise
    if not (cfg.PARAMS['min_mu_star'] <= mustar <= cfg.PARAMS['max_mu_star']):
        raise MassBalanceCalibrationError('mu* out of specified bounds: '
                                          '{:.2f}'.format(mustar))

    # Scalars in a small dict for later
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = int(tstar)
    df['bias'] = bias
    df['mu_star_glacierwide'] = mustar
    gdir.write_json(df, 'local_mustar')


def _mu_star_per_minimization(x, fls, cmb, temp, prcp, widths):

    # Get the corresponding mu
    mus = np.array([])
    for fl in fls:
        mu = fl.mu_star if fl.mu_star_is_valid else x
        mus = np.append(mus, np.ones(fl.nx) * mu)

    # TODO: possible optimisation here
    out = np.average(prcp - mus[:, np.newaxis] * temp, axis=0, weights=widths)
    return np.mean(out - cmb)


def _recursive_mu_star_calibration(gdir, fls, t_star, first_call=True,
                                   force_mu=None):

    # Do we have a calving glacier? This is only for the first call!
    # The calving mass-balance is distributed over the valid tributaries of the
    # main line, i.e. bad tributaries are not considered for calving
    cmb = calving_mb(gdir) if first_call else 0.

    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr_range = [t_star - mu_hp, t_star + mu_hp]

    # Get the corresponding mu
    heights = np.array([])
    widths = np.array([])
    for fl in fls:
        heights = np.append(heights, fl.surface_h)
        widths = np.append(widths, fl.widths)

    _, temp, prcp = mb_yearly_climate_on_height(gdir, heights,
                                                year_range=yr_range,
                                                flatten=False)

    if force_mu is None:
        try:
            mu_star = optimization.brentq(_mu_star_per_minimization,
                                          cfg.PARAMS['min_mu_star'],
                                          cfg.PARAMS['max_mu_star'],
                                          args=(fls, cmb, temp, prcp, widths),
                                          xtol=1e-5)
        except ValueError:
            raise MassBalanceCalibrationError('{} mu* out of specified '
                                              'bounds.'.format(gdir.rgi_id))

        if not np.isfinite(mu_star):
            raise MassBalanceCalibrationError('{} '.format(gdir.rgi_id) +
                                              'has a non finite mu.')
    else:
        mu_star = force_mu

    # Reset flux
    for fl in fls:
        fl.flux = np.zeros(len(fl.surface_h))

    # Flowlines in order to be sure - start with first guess mu*
    for fl in fls:
        y, t, p = mb_yearly_climate_on_height(gdir, fl.surface_h,
                                              year_range=yr_range,
                                              flatten=False)
        mu = fl.mu_star if fl.mu_star_is_valid else mu_star
        fl.set_apparent_mb(np.mean(p, axis=1) - mu*np.mean(t, axis=1),
                           mu_star=mu)

    # Sometimes, low lying tributaries have a non-physically consistent
    # Mass-balance. These tributaries wouldn't exist with a single
    # glacier-wide mu*, and therefore need a specific calibration.
    # All other mus may be affected
    if cfg.PARAMS['correct_for_neg_flux']:
        if np.any([fl.flux_needs_correction for fl in fls]):

            # We start with the highest Strahler number that needs correction
            not_ok = np.array([fl.flux_needs_correction for fl in fls])
            fl = np.array(fls)[not_ok][-1]

            # And we take all its tributaries
            inflows = centerlines.line_inflows(fl)

            # We find a new mu for these in a recursive call
            # TODO: this is where a flux kwarg can passed to tributaries
            _recursive_mu_star_calibration(gdir, inflows, t_star,
                                           first_call=False)

            # At this stage we should be ok
            assert np.all([~ fl.flux_needs_correction for fl in inflows])
            for fl in inflows:
                fl.mu_star_is_valid = True

            # After the above are OK we have to recalibrate all below
            _recursive_mu_star_calibration(gdir, fls, t_star,
                                           first_call=first_call)

    # At this stage we are good
    for fl in fls:
        fl.mu_star_is_valid = True


def _fallback_mu_star_calibration(gdir):
    """A Fallback function if climate.mu_star_calibration raises an Error.
￼
￼	    This function will still read, expand and write a `local_mustar.json`,
    filled with NANs, if climate.mu_star_calibration fails
    and if cfg.PARAMS['continue_on_error'] = True.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    """
    # read json
    df = gdir.read_json('local_mustar')
    # add these keys which mu_star_calibration would add
    df['mu_star_per_flowline'] = [np.nan]
    df['mu_star_flowline_avg'] = np.nan
    df['mu_star_allsame'] = np.nan
    # write
    gdir.write_json(df, 'local_mustar')


@entity_task(log, writes=['inversion_flowlines'],
             fallback=_fallback_mu_star_calibration)
def mu_star_calibration(gdir):
    """Compute the flowlines' mu* and the associated apparent mass-balance.

    If low lying tributaries have a non-physically consistent Mass-balance
    this function will either filter them out or calibrate each flowline with a
    specific mu*. The latter is default and recommended.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    # Interpolated data
    df = gdir.read_json('local_mustar')
    t_star = df['t_star']
    bias = df['bias']

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')
    # If someone calls the task a second time we need to reset this
    for fl in fls:
        fl.mu_star_is_valid = False

    force_mu = 0 if df['mu_star_glacierwide'] == 0 else None

    # Let's go
    _recursive_mu_star_calibration(gdir, fls, t_star, force_mu=force_mu)

    # If the user wants to filter the bad ones we remove them and start all
    # over again until all tributaries are physically consistent with one mu
    # This should only work if cfg.PARAMS['correct_for_neg_flux'] == False
    do_filter = [fl.flux_needs_correction for fl in fls]
    if cfg.PARAMS['filter_for_neg_flux'] and np.any(do_filter):
        assert not do_filter[-1]  # This should not happen
        # Keep only the good lines
        # TODO: this should use centerline.line_inflows for more efficiency!
        heads = [fl.orig_head for fl in fls if not fl.flux_needs_correction]
        centerlines.compute_centerlines(gdir, heads=heads, reset=True)
        centerlines.initialize_flowlines(gdir, reset=True)
        if gdir.has_file('downstream_line'):
            centerlines.compute_downstream_line(gdir, reset=True)
            centerlines.compute_downstream_bedshape(gdir, reset=True)
        centerlines.catchment_area(gdir, reset=True)
        centerlines.catchment_intersections(gdir, reset=True)
        centerlines.catchment_width_geom(gdir, reset=True)
        centerlines.catchment_width_correction(gdir, reset=True)
        local_t_star(gdir, tstar=t_star, bias=bias, reset=True)
        # Ok, re-call ourselves
        return mu_star_calibration(gdir, reset=True)

    # Check and write
    rho = cfg.PARAMS['ice_density']
    aflux = fls[-1].flux[-1] * 1e-9 / rho * gdir.grid.dx**2
    # If not marine and a bit far from zero, warning
    cmb = calving_mb(gdir)
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=0.01):
        log.warning('(%s) flux should be zero, but is: '
                    '%.4f km3 ice yr-1', gdir.rgi_id, aflux)
    # If not marine and quite far from zero, error
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=1):
        msg = ('({}) flux should be zero, but is: {:.4f} km3 ice yr-1'
               .format(gdir.rgi_id, aflux))
        raise MassBalanceCalibrationError(msg)
    gdir.write_pickle(fls, 'inversion_flowlines')

    # Store diagnostics
    mus = []
    weights = []
    for fl in fls:
        mus.append(fl.mu_star)
        weights.append(np.sum(fl.widths))
    df['mu_star_per_flowline'] = mus
    df['mu_star_flowline_avg'] = np.average(mus, weights=weights)
    all_same = np.allclose(mus, mus[0], atol=1e-3)
    df['mu_star_allsame'] = all_same
    if all_same:
        if not np.allclose(df['mu_star_flowline_avg'],
                           df['mu_star_glacierwide'],
                           atol=1e-3):
            raise MassBalanceCalibrationError('Unexpected difference between '
                                              'glacier wide mu* and the '
                                              'flowlines mu*.')
    # Write
    gdir.write_json(df, 'local_mustar')


@entity_task(log, writes=['inversion_flowlines', 'linear_mb_params'])
def apparent_mb_from_linear_mb(gdir, mb_gradient=3.):
    """Compute apparent mb from a linear mass-balance assumption (for testing).

    This is for testing currently, but could be used as alternative method
    for the inversion quite easily.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)

    # Get the height and widths along the fls
    h, w = gdir.get_inversion_flowline_hw()

    # Now find the ELA till the integrated mb is zero
    from oggm.core.massbalance import LinearMassBalance

    def to_minimize(ela_h):
        mbmod = LinearMassBalance(ela_h[0], grad=mb_gradient)
        smb = mbmod.get_specific_mb(heights=h, widths=w)
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
    rho = cfg.PARAMS['ice_density']
    for fl in fls:
        mbz = mbmod.get_annual_mb(fl.surface_h) * cfg.SEC_IN_YEAR * rho
        fl.set_apparent_mb(mbz)

    # Check and write
    aflux = fls[-1].flux[-1] * 1e-9 / rho * gdir.grid.dx**2
    # If not marine and a bit far from zero, warning
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=0.01):
        log.warning('(%s) flux should be zero, but is: '
                    '%.4f km3 ice yr-1', gdir.rgi_id, aflux)
    # If not marine and quite far from zero, error
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=1):
        msg = ('({}) flux should be zero, but is: {:.4f} km3 ice yr-1'
               .format(gdir.rgi_id, aflux))
        raise MassBalanceCalibrationError(msg)
    gdir.write_pickle(fls, 'inversion_flowlines')
    gdir.write_pickle({'ela_h': ela_h, 'grad': mb_gradient},
                      'linear_mb_params')


@global_task
def compute_ref_t_stars(gdirs):
    """ Detects the best t* for the reference glaciers and writes them to disk

    This task will be needed for mass balance calibration of custom climate
    data. For CRU and HISTALP baseline climate a precalibrated list is
    available and should be used instead.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        will be filtered for reference glaciers
    """

    if not cfg.PARAMS['run_mb_calibration']:
        raise InvalidParamsError('Are you sure you want to calibrate the '
                                 'reference t*? There is a pre-calibrated '
                                 'version available. If you know what you are '
                                 'doing and still want to calibrate, set the '
                                 '`run_mb_calibration` parameter to `True`.')

    log.info('Compute the reference t* and mu* for WGMS glaciers')

    # Reference glaciers only if in the list and period is good
    ref_gdirs = utils.get_ref_mb_glaciers(gdirs)

    # Run
    from oggm.workflow import execute_entity_task
    out = execute_entity_task(t_star_from_refmb, ref_gdirs)

    # Loop write
    df = pd.DataFrame()
    for gdir, res in zip(ref_gdirs, out):
        # list of mus compatibles with refmb
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
