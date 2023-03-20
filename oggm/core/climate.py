"""Climate data and mass balance computations"""
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


@entity_task(log, writes=['climate_historical'])
def process_custom_climate_data(gdir, y0=None, y1=None, output_filesuffix=None):
    """Processes and writes the climate data from a user-defined climate file.

    The input file must have a specific format
    (see https://github.com/OGGM/oggm-sample-data
    ->test-files/histalp_merged_hef.nc for an example).

    This is the way OGGM used to do it for HISTALP before it got added to the shop.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    y0 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    y1 : int
        the ending year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data). This year
        will be included, i.e. 2019 means the data will end at 2019-12-01
    output_filesuffix : str
        this adds a suffix to the output file (useful to avoid overwriting
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

    # Avoid reading all data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        nc_ts.set_subset(((gdir.cenlon, gdir.cenlat),
                          (gdir.cenlon, gdir.cenlat)),
                         margin=2)  # 2 is to be sure - also on small files

    yrs = nc_ts.time.year
    mths = nc_ts.time.month
    y0 = yrs[0] if y0 is None else y0
    if mths[0] != 1:
        # The file starts at the wrong month
        y0 += 1
    y1 = yrs[-1] if y1 is None else y1
    if mths[-1] != 12:
        # Same: wrong month
        y1 -= 1

    nc_ts.set_period(t0=f'{y0}-01-01', t1=f'{y1}-12-01')
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
    lon = nc_ts.get_vardata('lon')
    lat = nc_ts.get_vardata('lat')

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

    gdir.write_monthly_climate_file(time, iprcp, itemp, ihgt,
                                    ref_pix_lon, ref_pix_lat,
                                    filesuffix=output_filesuffix,
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
    elif baseline == 'W5E5':
        from oggm.shop.w5e5 import process_w5e5_data
        process_w5e5_data(gdir, output_filesuffix=output_filesuffix,
                          y0=y0, y1=y1, **kwargs)
    elif baseline == 'GSWP3_W5E5':
        from oggm.shop.w5e5 import process_gswp3_w5e5_data
        process_gswp3_w5e5_data(gdir, output_filesuffix=output_filesuffix,
                                y0=y0, y1=y1, **kwargs)
    elif baseline in ['ERA5', 'ERA5L', 'CERA', 'ERA5dr', 'ERA5L-HMA']:
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
        whether to scale the temperature standard deviation as well
        (you probably want to do that)
    replace_with_ref_data : bool
        the default is to paste the bias-corrected data where no reference
        data is available, i.e. creating timeseries which are not consistent
        in time but "better" for recent times (e.g. CERA-20C until 1980,
        then ERA5). Set this to False to prevent this and make a consistent
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
        tmp_sel = hist_temp.sel(time=slice(*cmn_time_range))
        tmp_std = tmp_sel.groupby('time.month').std(dim='time')
        std_fac = sref_temp.groupby('time.month').std(dim='time') / tmp_std
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
