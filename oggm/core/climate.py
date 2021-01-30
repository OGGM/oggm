"""Climate data and mass-balance computations"""
# Built ins
import logging
import os
import datetime
import json
import warnings

# External libs
import numpy as np
import xarray as xr
import netCDF4
import pandas as pd
from scipy import stats
from scipy import optimize

# Optional libs
try:
    import salem
except ImportError:
    pass

# Locals
from oggm import cfg
from oggm import utils
from oggm.core import centerlines
from oggm import entity_task, global_task
from oggm.exceptions import (MassBalanceCalibrationError, InvalidParamsError,
                             InvalidWorkflowError)

# Module logger
log = logging.getLogger(__name__)

# Parameters
_brentq_xtol = 2e-12

# Climate relevant params
MB_PARAMS = ['temp_default_gradient', 'temp_all_solid', 'temp_all_liq',
             'temp_melt', 'prcp_scaling_factor', 'climate_qc_months']


@entity_task(log, writes=['climate_historical'])
def process_custom_climate_data(gdir, y0=None, y1=None,
                                output_filesuffix=None):
    """Processes and writes the climate data from a user-defined climate file.

    The input file must have a specific format (see
    https://github.com/OGGM/oggm-sample-data ->test-files/histalp_merged_hef.nc
    for an example).

    This is the way OGGM used to do it for HISTALP before it got automatised.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
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
    y0 = yrs[0] if y0 is None else y0
    y1 = yrs[-1] if y1 is None else y1

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
                                    filesuffix=output_filesuffix,
                                    gradient=igrad,
                                    source=fpath)


@entity_task(log)
def process_climate_data(gdir, y0=None, y1=None, output_filesuffix=None,
                         **kwargs):
    """Adds the selected climate data to this glacier directory.

    Short wrapper deciding on which task to run based on
    `cfg.PARAMS['baseline_climate']`.

    If you want to make it explicit, simply call the relevant task
    (e.g. oggm.shop.cru.process_cru_data).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
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
    **kwargs :
        any other argument relevant to the task that will be called.
    """

    # Which climate should we use?
    baseline = cfg.PARAMS['baseline_climate']
    if baseline == 'CRU':
        from oggm.shop.cru import process_cru_data
        process_cru_data(gdir, output_filesuffix=output_filesuffix,
                         y0=y0, y1=y1, **kwargs)
    elif baseline == 'HISTALP':
        from oggm.shop.histalp import process_histalp_data
        process_histalp_data(gdir, output_filesuffix=output_filesuffix,
                             y0=y0, y1=y1, **kwargs)
    elif baseline in ['ERA5', 'ERA5L', 'CERA', 'ERA5dr']:
        from oggm.shop.ecmwf import process_ecmwf_data
        process_ecmwf_data(gdir, output_filesuffix=output_filesuffix,
                           dataset=baseline, y0=y0, y1=y1, **kwargs)
    elif '+' in baseline:
        # This bit below assumes ECMWF only datasets, but it should be
        # quite easy to extend for HISTALP+ERA5L for example
        from oggm.shop.ecmwf import process_ecmwf_data
        his, ref = baseline.split('+')
        s = 'tmp_'
        process_ecmwf_data(gdir, output_filesuffix=s+his, dataset=his,
                           y0=y0, y1=y1, **kwargs)
        process_ecmwf_data(gdir, output_filesuffix=s+ref, dataset=ref,
                           y0=y0, y1=y1, **kwargs)
        historical_delta_method(gdir,
                                ref_filesuffix=s+ref,
                                hist_filesuffix=s+his,
                                output_filesuffix=output_filesuffix)
    elif '|' in baseline:
        from oggm.shop.ecmwf import process_ecmwf_data
        his, ref = baseline.split('|')
        s = 'tmp_'
        process_ecmwf_data(gdir, output_filesuffix=s+his, dataset=his,
                           y0=y0, y1=y1, **kwargs)
        process_ecmwf_data(gdir, output_filesuffix=s+ref, dataset=ref,
                           y0=y0, y1=y1, **kwargs)
        historical_delta_method(gdir,
                                ref_filesuffix=s+ref,
                                hist_filesuffix=s+his,
                                output_filesuffix=output_filesuffix,
                                replace_with_ref_data=False)
    elif baseline == 'CUSTOM':
        process_custom_climate_data(gdir, y0=y0, y1=y1,
                                    output_filesuffix=output_filesuffix,
                                    **kwargs)
    else:
        raise ValueError("cfg.PARAMS['baseline_climate'] not understood")


@entity_task(log, writes=['climate_historical'])
def historical_delta_method(gdir, ref_filesuffix='', hist_filesuffix='',
                            output_filesuffix='', ref_year_range=None,
                            delete_input_files=True, scale_stddev=True,
                            replace_with_ref_data=True):
    """Applies the anomaly method to historical climate data.

    This function can be used to prolongate historical time series,
    for example by bias-correcting CERA-20C to ERA5 or ERA5-Land.

    The timeseries must be already available in the glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    ref_filesuffix : str
        the filesuffix of the historical climate data to take as reference
    hist_filesuffix : str
        the filesuffix of the historical climate data to apply to the
        reference
    output_filesuffix : str
        the filesuffix of the output file (usually left empty - i.e. this
        file will become the default)
    ref_year_range : tuple of str
        the year range for which you want to compute the anomalies. The
        default is to take the entire reference data period, but you could
        also choose `('1961', '1990')` for example
    delete_input_files : bool
        delete the input files after use - useful for operational runs
        where you don't want to carry too many files
    scale_stddev : bool
        whether or not to scale the temperature standard deviation as well
        (you probably want to do that)
    replace_with_ref_data : bool
        the default is to paste the bias-corrected data where no reference
        data is available, i.e. creating timeseries which are not consistent
        in time but "better" for recent times (e.g. CERA-20C until 1980,
        then ERA5). Set this to False to present this and make a consistent
        time series of CERA-20C (but bias corrected to the reference data,
        so "better" than CERA-20C out of the box).
    """

    if ref_year_range is not None:
        raise NotImplementedError()

    # Read input
    f_ref = gdir.get_filepath('climate_historical', filesuffix=ref_filesuffix)
    with xr.open_dataset(f_ref) as ds:
        ref_temp = ds['temp']
        ref_prcp = ds['prcp']
        ref_hgt = float(ds.ref_hgt)
        ref_lon = float(ds.ref_pix_lon)
        ref_lat = float(ds.ref_pix_lat)
        source = ds.attrs.get('climate_source')

    f_his = gdir.get_filepath('climate_historical', filesuffix=hist_filesuffix)
    with xr.open_dataset(f_his) as ds:
        hist_temp = ds['temp']
        hist_prcp = ds['prcp']
        # To differentiate both cases
        if replace_with_ref_data:
            source = ds.attrs.get('climate_source') + '+' + source
        else:
            source = ds.attrs.get('climate_source') + '|' + source

    # Common time period
    cmn_time = (ref_temp + hist_temp)['time']
    assert len(cmn_time) // 12 == len(cmn_time) / 12
    # We need an even number of years for this to work
    if ((len(cmn_time) // 12) % 2) == 1:
        cmn_time = cmn_time.isel(time=slice(12, len(cmn_time)))
    assert len(cmn_time) // 12 == len(cmn_time) / 12
    assert ((len(cmn_time) // 12) % 2) == 0
    cmn_time_range = cmn_time.values[[0, -1]]

    # Select ref
    sref_temp = ref_temp.sel(time=slice(*cmn_time_range))
    sref_prcp = ref_prcp.sel(time=slice(*cmn_time_range))

    # See if we need to scale the variability
    if scale_stddev:
        # This is a bit more arithmetic
        sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
        tmp_sel = hist_temp.sel(time=slice(*cmn_time_range))
        tmp_std = tmp_sel.groupby('time.month').std(dim='time')
        std_fac = sref_temp.groupby('time.month').std(dim='time') / tmp_std
        std_fac = std_fac.roll(month=13-sm, roll_coords=True)
        std_fac = np.tile(std_fac.data, len(hist_temp) // 12)
        win_size = len(cmn_time) + 1

        def roll_func(x, axis=None):
            x = x[:, ::12]
            n = len(x[0, :]) // 2
            xm = np.nanmean(x, axis=axis)
            return xm + (x[:, n] - xm) * std_fac

        hist_temp = hist_temp.rolling(time=win_size, center=True,
                                      min_periods=1).reduce(roll_func)

    # compute monthly anomalies
    # of temp
    ts_tmp_sel = hist_temp.sel(time=slice(*cmn_time_range))
    ts_tmp_avg = ts_tmp_sel.groupby('time.month').mean(dim='time')
    ts_tmp = hist_temp.groupby('time.month') - ts_tmp_avg
    # of precip -- scaled anomalies
    ts_pre_avg = hist_prcp.sel(time=slice(*cmn_time_range))
    ts_pre_avg = ts_pre_avg.groupby('time.month').mean(dim='time')
    ts_pre_ano = hist_prcp.groupby('time.month') - ts_pre_avg
    # scaled anomalies is the default. Standard anomalies above
    # are used later for where ts_pre_avg == 0
    ts_pre = hist_prcp.groupby('time.month') / ts_pre_avg

    # reference averages
    # for temp
    loc_tmp = sref_temp.groupby('time.month').mean()
    ts_tmp = ts_tmp.groupby('time.month') + loc_tmp

    # for prcp
    loc_pre = sref_prcp.groupby('time.month').mean()
    # scaled anomalies
    ts_pre = ts_pre.groupby('time.month') * loc_pre
    # standard anomalies
    ts_pre_ano = ts_pre_ano.groupby('time.month') + loc_pre
    # Correct infinite values with standard anomalies
    ts_pre.values = np.where(np.isfinite(ts_pre.values),
                             ts_pre.values,
                             ts_pre_ano.values)
    # The previous step might create negative values (unlikely). Clip them
    ts_pre.values = utils.clip_min(ts_pre.values, 0)

    assert np.all(np.isfinite(ts_pre.values))
    assert np.all(np.isfinite(ts_tmp.values))

    if not replace_with_ref_data:
        # Just write what we have
        gdir.write_monthly_climate_file(ts_tmp.time.values,
                                        ts_pre.values, ts_tmp.values,
                                        ref_hgt, ref_lon, ref_lat,
                                        filesuffix=output_filesuffix,
                                        source=source)
    else:
        # Select all hist data before the ref
        ts_tmp = ts_tmp.sel(time=slice(ts_tmp.time[0], ref_temp.time[0]))
        ts_tmp = ts_tmp.isel(time=slice(0, -1))
        ts_pre = ts_pre.sel(time=slice(ts_tmp.time[0], ref_temp.time[0]))
        ts_pre = ts_pre.isel(time=slice(0, -1))
        # Concatenate and write
        gdir.write_monthly_climate_file(np.append(ts_pre.time, ref_prcp.time),
                                        np.append(ts_pre, ref_prcp),
                                        np.append(ts_tmp, ref_temp),
                                        ref_hgt, ref_lon, ref_lat,
                                        filesuffix=output_filesuffix,
                                        source=source)

    if delete_input_files:
        # Delete all files without suffix
        if ref_filesuffix:
            os.remove(f_ref)
        if hist_filesuffix:
            os.remove(f_his)


@entity_task(log, writes=['climate_historical'])
def historical_climate_qc(gdir):
    """"Check the "quality" of climate data and correct it if needed.

    This forces the climate data to have at least one month of melt per year
    at the terminus of the glacier (i.e. simply shifting temperatures up
    when necessary), and at least one month where accumulation is possible
    at the glacier top (i.e. shifting the temperatures down).
    """

    # Parameters
    temp_s = (cfg.PARAMS['temp_all_liq'] + cfg.PARAMS['temp_all_solid']) / 2
    temp_m = cfg.PARAMS['temp_melt']
    default_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
    qc_months = cfg.PARAMS['climate_qc_months']
    if qc_months == 0:
        return

    # Read file
    fpath = gdir.get_filepath('climate_historical')
    igrad = None
    with utils.ncDataset(fpath) as nc:
        # time
        # Read timeseries
        itemp = nc.variables['temp'][:]
        if 'gradient' in nc.variables:
            igrad = nc.variables['gradient'][:]
            # Security for stuff that can happen with local gradients
            igrad = np.where(~np.isfinite(igrad), default_grad, igrad)
            igrad = utils.clip_array(igrad, g_minmax[0], g_minmax[1])
        ref_hgt = nc.ref_hgt

    # Default gradient?
    if igrad is None:
        igrad = itemp * 0 + default_grad

    ny = len(igrad) // 12
    assert ny == len(igrad) / 12

    # Geometry data
    fls = gdir.read_pickle('inversion_flowlines')
    heights = np.array([])
    for fl in fls:
        heights = np.append(heights, fl.surface_h)
    top_h = np.max(heights)
    bot_h = np.min(heights)

    # First check - there should be at least one month of melt every year
    prev_ref_hgt = ref_hgt
    while True:
        ts_bot = itemp + default_grad * (bot_h - ref_hgt)
        ts_bot = (ts_bot.reshape((ny, 12)) > temp_m).sum(axis=1)
        if np.all(ts_bot >= qc_months):
            # Ok all good
            break
        # put ref hgt a bit higher so that we warm things a bit
        ref_hgt += 10

    # If we changed this it makes no sense to lower it down again,
    # so resume here:
    if ref_hgt != prev_ref_hgt:
        with utils.ncDataset(fpath, 'a') as nc:
            nc.ref_hgt = ref_hgt
            nc.uncorrected_ref_hgt = prev_ref_hgt
        gdir.add_to_diagnostics('ref_hgt_qc_diff', int(ref_hgt - prev_ref_hgt))
        return

    # Second check - there should be at least one month of acc every year
    while True:
        ts_top = itemp + default_grad * (top_h - ref_hgt)
        ts_top = (ts_top.reshape((ny, 12)) < temp_s).sum(axis=1)
        if np.all(ts_top >= qc_months):
            # Ok all good
            break
        # put ref hgt a bit lower so that we cold things a bit
        ref_hgt -= 10

    if ref_hgt != prev_ref_hgt:
        with utils.ncDataset(fpath, 'a') as nc:
            nc.ref_hgt = ref_hgt
            nc.uncorrected_ref_hgt = prev_ref_hgt
        gdir.add_to_diagnostics('ref_hgt_qc_diff', int(ref_hgt - prev_ref_hgt))


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
    with utils.ncDataset(gdir.get_filepath('climate_historical')) as nc:
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
            igrad = utils.clip_array(igrad, g_minmax[0], g_minmax[1])
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
    temp2dformelt = utils.clip_min(temp2dformelt, 0)
    # Compute solid precipitation from total precipitation
    prcpsol = np.atleast_2d(iprcp).repeat(npix, 0)
    fac = 1 - (temp2d - temp_all_solid) / (temp_all_liq - temp_all_solid)
    fac = utils.clip_array(fac, 0, 1)
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


@entity_task(log)
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
    ci = gdir.get_climate_info()
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
    return pd.Series(data=mu_yr_clim, index=years)


@entity_task(log)
def t_star_from_refmb(gdir, mbdf=None, glacierwide=None,
                      min_mu_star=None, max_mu_star=None):
    """Computes the ref t* for the glacier, given a series of MB measurements.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    mbdf: a pd.Series containing the observed MB data indexed by year
        if None, read automatically from the reference data

    Returns
    -------
    A dict: {t_star:[], bias:[], 'avg_mb_per_mu': [], 'avg_ref_mb': []}
    """

    from oggm.core.massbalance import MultipleFlowlineMassBalance

    if glacierwide is None:
        glacierwide = cfg.PARAMS['tstar_search_glacierwide']

    # Be sure we have no marine terminating glacier
    assert not gdir.is_tidewater

    # Reference time series
    if mbdf is None:
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

    # mu* constraints
    if min_mu_star is None:
        min_mu_star = cfg.PARAMS['min_mu_star']
    if max_mu_star is None:
        max_mu_star = cfg.PARAMS['max_mu_star']

    # which years to look at
    ref_years = mbdf.index.values

    # Average oberved mass-balance
    ref_mb = np.mean(mbdf)

    # Compute one mu candidate per year and the associated statistics
    # Only get the years were we consider looking for tstar
    y0, y1 = cfg.PARAMS['tstar_search_window']
    ci = gdir.get_climate_info()
    y0 = y0 or ci['baseline_hydro_yr_0']
    y1 = y1 or ci['baseline_hydro_yr_1']
    years = np.arange(y0, y1+1)

    ny = len(years)
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    mb_per_mu = pd.Series(index=years, dtype=np.float)

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
                _recursive_mu_star_calibration(gdir, fls, y, first_call=True,
                                               min_mu_star=min_mu_star,
                                               max_mu_star=max_mu_star)
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
    d = gdir.get_climate_info()
    d['t_star'] = amin
    d['bias'] = diff[amin]
    gdir.write_json(d, 'climate_info')

    return {'t_star': amin, 'bias': diff[amin],
            'avg_mb_per_mu': mb_per_mu, 'avg_ref_mb': ref_mb}


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


@entity_task(log, writes=['local_mustar', 'climate_info'],
             fallback=_fallback_local_t_star)
def local_t_star(gdir, *, ref_df=None, tstar=None, bias=None,
                 clip_mu_star=None, min_mu_star=None, max_mu_star=None):
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
    clip_mu_star: bool, optional
        defaults to cfg.PARAMS['clip_mu_star']
    min_mu_star: bool, optional
        defaults to cfg.PARAMS['min_mu_star']
    max_mu_star: bool, optional
        defaults to cfg.PARAMS['max_mu_star']
    """

    if tstar is None or bias is None:
        # Do our own interpolation
        if ref_df is None:
            # Use the the local calibration
            fp = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
            if not os.path.exists(fp):
                raise InvalidWorkflowError('If ref_df is not given, provide '
                                           '`ref_tstars.csv` in the working '
                                           'directory')
            ref_df = pd.read_csv(fp)

            # Check that the params are fine
            fp = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars_params.json')
            if not os.path.exists(fp):
                raise InvalidWorkflowError('If ref_df is not given, provide '
                                           '`ref_tstars_params.json` in the '
                                           'working directory')
            with open(fp, 'r') as fp:
                ref_params = json.load(fp)
            for k, v in ref_params.items():
                if cfg.PARAMS[k] != v:
                    msg = ('The reference t* list you are trying to use was '
                           'calibrated with different MB parameters.')
                    raise MassBalanceCalibrationError(msg)

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
            tstar = int(np.average(amin.tstar, weights=1./distances).round())
            bias = np.average(amin.bias, weights=1./distances)

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    out = gdir.get_climate_info()
    out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in MB_PARAMS}
    gdir.write_json(out, 'climate_info')

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

    # mu* constraints
    if clip_mu_star is None:
        clip_mu_star = cfg.PARAMS['clip_mu_star']
    if min_mu_star is None:
        min_mu_star = cfg.PARAMS['min_mu_star']
    if max_mu_star is None:
        max_mu_star = cfg.PARAMS['max_mu_star']

    # Clip it?
    if clip_mu_star:
        mustar = utils.clip_min(mustar, min_mu_star)

    # If mu out of bounds, raise
    if not (min_mu_star <= mustar <= max_mu_star):
        raise MassBalanceCalibrationError('{}: mu* out of specified bounds: '
                                          '{:.2f}'.format(gdir.rgi_id, mustar))

    # Scalars in a small dict for later
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = int(tstar)
    df['bias'] = bias
    df['mu_star_glacierwide'] = mustar
    gdir.write_json(df, 'local_mustar')


def _check_terminus_mass_flux(gdir, fls, cmb):
    # Avoid code duplication

    rho = cfg.PARAMS['ice_density']

    # This variable is in "sensible" units normalized by width
    flux = fls[-1].flux[-1]
    aflux = flux * (gdir.grid.dx ** 2) / rho  # m3 ice per year

    # If not marine and a bit far from zero, warning
    if cmb == 0 and flux > 0 and not np.allclose(flux, 0, atol=0.01):
        log.info('(%s) flux should be zero, but is: '
                 '%.4f m3 ice yr-1', gdir.rgi_id, aflux)

    # If not marine and quite far from zero, error
    if cmb == 0 and flux > 0 and not np.allclose(flux, 0, atol=1):
        msg = ('({}) flux should be zero, but is: {:.4f} m3 ice yr-1'
               .format(gdir.rgi_id, aflux))
        raise MassBalanceCalibrationError(msg)


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
                                   force_mu=None, min_mu_star=None,
                                   max_mu_star=None):

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
            mu_star = optimize.brentq(_mu_star_per_minimization,
                                      min_mu_star, max_mu_star,
                                      args=(fls, cmb, temp, prcp, widths),
                                      xtol=_brentq_xtol)
        except ValueError:
            # This happens in very rare cases
            _mu_lim = _mu_star_per_minimization(min_mu_star, fls, cmb, temp,
                                                prcp, widths)
            if _mu_lim < min_mu_star and np.allclose(_mu_lim, min_mu_star):
                mu_star = min_mu_star
            else:
                raise MassBalanceCalibrationError('{} mu* out of specified '
                                                  'bounds.'.format(gdir.rgi_id)
                                                  )

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
    if cfg.PARAMS['correct_for_neg_flux'] and (len(fls) > 1):
        if np.any([fl.flux_needs_correction for fl in fls]):

            # We start with the highest Strahler number that needs correction
            not_ok = np.array([fl.flux_needs_correction for fl in fls])
            fl = np.array(fls)[not_ok][-1]

            # And we take all its tributaries
            inflows = centerlines.line_inflows(fl)

            # We find a new mu for these in a recursive call
            # TODO: this is where a flux kwarg can passed to tributaries
            _recursive_mu_star_calibration(gdir, inflows, t_star,
                                           first_call=False,
                                           min_mu_star=min_mu_star,
                                           max_mu_star=max_mu_star)

            # At this stage we should be ok
            assert np.all([~ fl.flux_needs_correction for fl in inflows])
            for fl in inflows:
                fl.mu_star_is_valid = True

            # After the above are OK we have to recalibrate all below
            _recursive_mu_star_calibration(gdir, fls, t_star,
                                           first_call=first_call,
                                           min_mu_star=min_mu_star,
                                           max_mu_star=max_mu_star)

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
def mu_star_calibration(gdir, min_mu_star=None, max_mu_star=None):
    """Compute the flowlines' mu* and the associated apparent mass-balance.

    If low lying tributaries have a non-physically consistent Mass-balance
    this function will either filter them out or calibrate each flowline with a
    specific mu*. The latter is default and recommended.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    min_mu_star: bool, optional
        defaults to cfg.PARAMS['min_mu_star']
    max_mu_star: bool, optional
        defaults to cfg.PARAMS['max_mu_star']
    """

    # Interpolated data
    df = gdir.read_json('local_mustar')
    t_star = df['t_star']
    bias = df['bias']

    # mu* constraints
    if min_mu_star is None:
        min_mu_star = cfg.PARAMS['min_mu_star']
    if max_mu_star is None:
        max_mu_star = cfg.PARAMS['max_mu_star']

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')
    # If someone calls the task a second time we need to reset this
    for fl in fls:
        fl.mu_star_is_valid = False

    force_mu = min_mu_star if df['mu_star_glacierwide'] == min_mu_star else None

    # Let's go
    _recursive_mu_star_calibration(gdir, fls, t_star, force_mu=force_mu,
                                   min_mu_star=min_mu_star,
                                   max_mu_star=max_mu_star)

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
        log.info('(%s) flux should be zero, but is: '
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
def apparent_mb_from_linear_mb(gdir, mb_gradient=3., ela_h=None):
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
        mbmod = LinearMassBalance(ela_h, grad=mb_gradient)
        smb = mbmod.get_specific_mb(heights=h, widths=w)
        return smb - cmb

    if ela_h is None:
        ela_h = optimize.brentq(to_minimize, -1e5, 1e5, xtol=_brentq_xtol)

    # For each flowline compute the apparent MB
    rho = cfg.PARAMS['ice_density']
    fls = gdir.read_pickle('inversion_flowlines')
    # Reset flux
    for fl in fls:
        fl.flux = np.zeros(len(fl.surface_h))
    # Flowlines in order to be sure
    mbmod = LinearMassBalance(ela_h, grad=mb_gradient)
    for fl in fls:
        mbz = mbmod.get_annual_mb(fl.surface_h) * cfg.SEC_IN_YEAR * rho
        fl.set_apparent_mb(mbz)

    # Check and write
    aflux = fls[-1].flux[-1] * 1e-9 / rho * gdir.grid.dx**2
    # If not marine and a bit far from zero, warning
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=0.01):
        log.info('(%s) flux should be zero, but is: '
                 '%.4f km3 ice yr-1', gdir.rgi_id, aflux)
    # If not marine and quite far from zero, error
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=1):
        msg = ('({}) flux should be zero, but is: {:.4f} km3 ice yr-1'
               .format(gdir.rgi_id, aflux))
        raise MassBalanceCalibrationError(msg)
    gdir.write_pickle(fls, 'inversion_flowlines')
    gdir.write_pickle({'ela_h': ela_h, 'grad': mb_gradient},
                      'linear_mb_params')


@entity_task(log, writes=['inversion_flowlines'])
def apparent_mb_from_any_mb(gdir, mb_model=None, mb_years=None):
    """Compute apparent mb from an arbitrary mass-balance profile.

    This searches for a mass-balance residual to add to the mass-balance
    profile so that the average specific MB is zero.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass-balance model to use
    mb_years : array
        the array of years from which you want to average the MB for (for
        mb_model only).
    """

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')

    # Unchanged SMB
    o_smb = np.mean(mb_model.get_specific_mb(fls=fls, year=mb_years))

    def to_minimize(residual_to_opt):
        return o_smb + residual_to_opt - cmb

    residual = optimize.brentq(to_minimize, -1e5, 1e5, xtol=_brentq_xtol)

    # Reset flux
    for fl in fls:
        fl.flux = np.zeros(len(fl.surface_h))

    # Flowlines in order to be sure
    rho = cfg.PARAMS['ice_density']
    for fl_id, fl in enumerate(fls):
        mbz = 0
        for yr in mb_years:
            mbz += mb_model.get_annual_mb(fl.surface_h, year=yr,
                                          fls=fls, fl_id=fl_id)
        mbz = mbz / len(mb_years)
        fl.set_apparent_mb(mbz * cfg.SEC_IN_YEAR * rho + residual)

    # Check and write
    aflux = fls[-1].flux[-1] * 1e-9 / rho * gdir.grid.dx**2
    # If not marine and a bit far from zero, warning
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=0.01):
        log.info('(%s) flux should be zero, but is: '
                 '%.4f km3 ice yr-1', gdir.rgi_id, aflux)
    # If not marine and quite far from zero, error
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=1):
        msg = ('({}) flux should be zero, but is: {:.4f} km3 ice yr-1'
               .format(gdir.rgi_id, aflux))
        raise MassBalanceCalibrationError(msg)
    gdir.add_to_diagnostics('apparent_mb_from_any_mb_residual', residual)
    gdir.write_pickle(fls, 'inversion_flowlines')


@global_task
def compute_ref_t_stars(gdirs):
    """ Detects the best t* for the reference glaciers and writes them to disk

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

    # Should be iterable
    gdirs = utils.tolist(gdirs)

    # Reference glaciers only if in the list and period is good
    ref_gdirs = utils.get_ref_mb_glaciers(gdirs)

    # Run
    from oggm.workflow import execute_entity_task
    out = execute_entity_task(t_star_from_refmb, ref_gdirs)

    # Loop write
    df = pd.DataFrame()
    for gdir, res in zip(ref_gdirs, out):
        if res is None:
            # For certain parameters there is no valid mu candidate on certain
            # glaciers. E.g. if temp is to low for melt. This will raise an
            # error in t_star_from_refmb and should only get here if
            # continue_on_error = True
            # Do not add this glacier to the ref_tstar.csv
            # Think of better solution later
            continue

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
    # We store the associated params to make sure
    # other tools cannot fool around without re-calibration

    params_file = os.path.join(cfg.PATHS['working_dir'],
                               'ref_tstars_params.json')
    with open(params_file, 'w') as fp:
        json.dump({k: cfg.PARAMS[k] for k in MB_PARAMS}, fp)
