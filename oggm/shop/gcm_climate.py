"""Climate data pre-processing"""
# Built ins
import logging
from packaging.version import Version

# External libs
import cftime
import numpy as np
import netCDF4
import xarray as xr

# Locals
from oggm import cfg
from oggm import utils
from oggm import entity_task
from oggm.exceptions import InvalidParamsError

# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['gcm_data'])
def process_gcm_data(gdir, prcp=None, temp=None,
                     year_range=('1961', '1990'), scale_stddev=True,
                     filesuffix='', output_filesuffix='',
                     time_unit=None, calendar=None, source='',
                     apply_bias_correction=True):
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
    filesuffix : str
        same as `output_filesuffix` but deprecated.
    output_filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    time_unit : str
        The unit conversion for NetCDF files. It must be adapted to the
        length of the time series. The default is to choose
        it ourselves based on the starting year.
        For example: 'days since 0850-01-01 00:00:00'
    calendar : str
        If you use an exotic calendar (e.g. 'noleap')
    source : str
        For metadata: the source of the climate data
    apply_bias_correction : boolean
        if a bias-correction should be applied. Default is True, only set it to False
        if the GCM has already been externally bias-corrected to the applied
        observational calibration dataset (true for ISIMIP 3b that is bias-corrected
        to W5E5). !!! We assume that temp is in Kelvin and convert to CELSIUS !!!
    """

    if filesuffix:
        output_filesuffix = filesuffix

    # Standard sanity checks
    months = temp['time.month']
    if months[0] != 1:
        raise ValueError('We expect the files to start in January!')
    if months[-1] != 12:
        raise ValueError('We expect the files to end in December!')

    if (np.abs(temp['lon']) > 180) or (np.abs(prcp['lon']) > 180):
        raise ValueError('We expect the longitude coordinates to be within '
                         '[-180, 180].')

    assert len(prcp) // 12 == len(prcp) / 12, "Somehow we didn't get full years"
    assert len(temp) // 12 == len(temp) / 12, "Somehow we didn't get full years"

    # Get the reference data to apply the anomaly to
    fpath = gdir.get_filepath('climate_historical')
    with xr.open_dataset(fpath) as ds_ref:

        ds_ref = ds_ref.sel(time=slice(*year_range))
        if apply_bias_correction:
            # compute monthly anomalies
            # of temp
            if scale_stddev:
                # This is a bit more arithmetic
                ts_tmp_sel = temp.sel(time=slice(*year_range))
                if len(ts_tmp_sel) // 12 != len(ts_tmp_sel) / 12:
                    raise InvalidParamsError('year_range cannot contain the first'
                                             'or last calendar year in the series')
                if ((len(ts_tmp_sel) // 12) % 2) == 1:
                    raise InvalidParamsError('We need an even number of years '
                                             'for this to work')
                ts_tmp_std = ts_tmp_sel.groupby('time.month').std(dim='time')
                std_fac = ds_ref.temp.groupby('time.month').std(dim='time') / ts_tmp_std
                std_fac = np.tile(std_fac.data, len(temp) // 12)
                # We need an even number of years for this to work
                win_size = len(ts_tmp_sel) + 1

                def roll_func(x, axis=None):
                    x = x[:, ::12]
                    n = len(x[0, :]) // 2
                    xm = np.nanmean(x, axis=axis)
                    return xm + (x[:, n] - xm) * std_fac

                temp = temp.rolling(time=win_size, center=True,
                                    min_periods=1).reduce(roll_func)

            ts_tmp_sel = temp.sel(time=slice(*year_range))
            if len(ts_tmp_sel.time) != len(ds_ref.time):
                raise InvalidParamsError('The reference climate period and the '
                                         'GCM period after window selection do '
                                         'not match.')
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
            loc_tmp = ds_ref.temp.groupby('time.month').mean()
            ts_tmp = ts_tmp.groupby('time.month') + loc_tmp

            # for prcp
            loc_pre = ds_ref.prcp.groupby('time.month').mean()
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
        else:
            # do no bias correction at all
            # (!!! only if GCM is already externally bias corrected)
            ts_tmp = temp - 273.15  # convert K to Celsius
            ts_pre = prcp  # mm month-1
            source = source + '_no_OGGM_bias_correction'

        gdir.write_monthly_climate_file(temp.time.values,
                                        ts_pre.values, ts_tmp.values,
                                        float(ds_ref.ref_hgt),
                                        prcp.lon.values, prcp.lat.values,
                                        time_unit=time_unit,
                                        calendar=calendar,
                                        file_name='gcm_data',
                                        source=source,
                                        filesuffix=output_filesuffix)


@entity_task(log, writes=['gcm_data'])
def process_monthly_isimip_data(gdir, output_filesuffix='',
                                member='mri-esm2-0_r1i1p1f1',
                                ssp='ssp126',
                                year_range=('1979', '2014'),
                                apply_bias_correction=False,
                                testing=False,
                                y0=None, y1=None,
                                **kwargs):
    """Read, process and store the isimip3b gcm data for this glacier.

    It stores the data in a format that can be used by the OGGM mass balance
    model and in the glacier directory.

    Currently, this function is built for the ISIMIP3b
    simulations that are on the OGGM servers.

    Parameters
    ----------
    output_filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
        If it is not set, we create a filesuffix with applied ensemble and ssp
    member : str
        ensemble member gcm that you want to process
    ssp : str
        ssp scenario to process (only 'ssp126', 'ssp370' or 'ssp585' are available)
    year_range : tuple of str
        the year range for which the anomalies are computed
        (passed to process_gcm_gdata). Default for ISIMIP3b `('1979', '2014')`
    apply_bias_correction : bool
        whether the bias correction is applied (default is False) or not. As
        we use already internally bias-corrected GCMs, it is default set
        to False!
    testing : boolean
        Default is False. If testing is set to True,
        the smaller test ISIMIP3b gcm files are downloaded
        instead (only useful for pytest)
    y0 : int
        start year of the ISIMIP3b data processing.
        Default is None which processes the entire timeseries.
        Set this to the beginning of your bias correction/
        projection period minus half of bc period
        to make processing faster.
    y1 : int
        end year of the CMIP data processing.
        Set this to the end of your projection period
        plus half of bc period. Default is None to process
        the entire time series, same as y0.
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    if output_filesuffix == '':
        # recognize the gcm climate file for later
        output_filesuffix = '_monthly_ISIMIP3b_{}_{}'.format(member, ssp)

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat

    if y0 is not None:
        assert y0 < 2014, 'y0 has to be below 2014'
    if y1 is not None:
        assert y1 > 2014, 'y0 has to be above 2014 at the moment'

    if testing:
        gcm_server = 'https://cluster.klima.uni-bremen.de/~oggm/test_climate/'
    else:
        gcm_server = 'https://cluster.klima.uni-bremen.de/~oggm/'

    path = f'{gcm_server}/cmip6/isimip3b/flat/2023.2/monthly/'
    add = '_global_monthly_flat_glaciers.nc'

    fpath_spec = path + '{}_w5e5_'.format(member) + '{ssp}_{var}' + add
    fpath_temp = fpath_spec.format(var='tasAdjust', ssp=ssp)
    fpath_temp_h = fpath_spec.format(var='tasAdjust', ssp='historical')
    fpath_precip = fpath_spec.format(var='prAdjust', ssp=ssp)
    fpath_precip_h = fpath_spec.format(var='prAdjust', ssp='historical')
    with utils.get_lock():
        fpath_temp = utils.file_downloader(fpath_temp)
        fpath_temp_h = utils.file_downloader(fpath_temp_h)
        fpath_precip = utils.file_downloader(fpath_precip)
        fpath_precip_h = utils.file_downloader(fpath_precip_h)

    # Read the GCM files
    with xr.open_dataset(fpath_temp_h, use_cftime=True) as tempds_hist, \
            xr.open_dataset(fpath_temp, use_cftime=True) as tempds_gcm:
        # make processing faster
        if y0 is not None:
            tempds_hist = tempds_hist.sel(time=slice(str(y0), None))
        if y1 is not None:
            tempds_gcm = tempds_gcm.sel(time=slice(None, str(y1)))
        # Check longitude conventions
        if tempds_gcm.longitude.min() >= 0 and glon <= 0:
            glon += 360
        assert tempds_gcm.attrs['experiment'] == ssp
        # Take the closest to the glacier
        # Should we consider GCM interpolation?
        # try:
        # computing all the distances and choose the nearest gridpoint
        c = ((tempds_gcm.longitude - glon) ** 2 +
             (tempds_gcm.latitude - glat) ** 2)
        # first select gridpoint, then merge, should be faster!!!
        temp_a_gcm = tempds_gcm.isel(points=np.argmin(c.data))
        temp_a_hist = tempds_hist.isel(points=np.argmin(c.data))
        # merge historical with gcm together
        # TODO: change to drop_conflicts when xarray version v0.17.0 can
        #  be used with salem
        temp_a = xr.merge([temp_a_gcm, temp_a_hist],
                          combine_attrs='override')
        temp = temp_a.tasAdjust
        temp['lon'] = temp_a.longitude
        temp['lat'] = temp_a.latitude

        temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360

    with xr.open_dataset(fpath_precip_h, use_cftime=True) as precipds_hist, \
            xr.open_dataset(fpath_precip, use_cftime=True) as precipds_gcm:
        # make processing faster
        if y0 is not None:
            precipds_hist = precipds_hist.sel(time=slice(str(y0), None))
        if y1 is not None:
            precipds_gcm = precipds_gcm.sel(time=slice(None, str(y1)))
        c = ((precipds_gcm.longitude - glon) ** 2 +
             (precipds_gcm.latitude - glat) ** 2)
        precip_a_gcm = precipds_gcm.isel(points=np.argmin(c.data))
        precip_a_hist = precipds_hist.isel(points=np.argmin(c.data))
        precip_a = xr.merge([precip_a_gcm, precip_a_hist],
                            combine_attrs='override')

        precip = precip_a.prAdjust
        precip['lon'] = precip_a.longitude
        precip['lat'] = precip_a.latitude

        # Back to [-180, 180] for OGGM
        precip.lon.values = precip.lon if precip.lon <= 180 else precip.lon - 360

        # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!
        assert 'kg m-2 s-1' in precip.units, 'Precip units not understood'
        ny, r = divmod(len(temp), 12)
        assert r == 0
        dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in temp['time.month']]
        precip = precip * dimo * (60 * 60 * 24)

    process_gcm_data(gdir, output_filesuffix=output_filesuffix,
                     prcp=precip, temp=temp,
                     year_range=year_range, source=output_filesuffix,
                     apply_bias_correction=apply_bias_correction,
                     **kwargs)


@entity_task(log, writes=['gcm_data'])
def process_cesm_data(gdir, fpath_temp=None, fpath_precc=None,
                      filesuffix='', output_filesuffix='',
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
    fpath_temp : str
        path to the temp file (default: cfg.PATHS['cesm_temp_file'])
    fpath_precc : str
        path to the precc file (default: cfg.PATHS['cesm_precc_file'])
    fpath_precl : str
        path to the precl file (default: cfg.PATHS['cesm_precl_file'])
    filesuffix : str
        same as `output_filesuffix` but deprecated.
    output_filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    if filesuffix:
        output_filesuffix = filesuffix

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
    if Version(xr.__version__) < Version('0.11'):
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
    process_gcm_data(gdir, output_filesuffix=output_filesuffix, prcp=prcp, temp=temp,
                     time_unit=time_unit, calendar=calendar, **kwargs)


@entity_task(log, writes=['gcm_data'])
def process_cmip_data(gdir, fpath_temp=None, fpath_precip=None, y0=None, y1=None,
                      filesuffix='', output_filesuffix='',
                      **kwargs):
    """Read, process and store the CMIP5 and CMIP6 climate data for this glacier.

    It stores the data in a format that can be used by the OGGM mass balance
    model and in the glacier directory.

    Currently, this function is built for the CMIP5 and CMIP6 projection
    simulations that are on the OGGM servers.

    Parameters
    ----------
    fpath_temp : str
        path to the temp file
    fpath_precip : str
        path to the precip file
    y0 : int
        start year of the CMIP data processing.
        Default is None which processes the entire timeseries.
        Set this to the beginning of your bias correction/
        projection period minus half of bc period
        to make process_cmip_data faster.
    y1 : int
        end year of the CMIP data processing.
        Set this to the end of your projection period
        plus half of bc period. Default is None to process
        the entire time series, same as y0.
    filesuffix : str
        same as `output_filesuffix` but deprecated.
    output_filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    if filesuffix:
        output_filesuffix = filesuffix

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat

    if y0 is not None:
        y0 = str(y0)
    if y1 is not None:
        y1 = str(y1)
    # Read the GCM files
    with xr.open_dataset(fpath_temp, use_cftime=True) as tempds, \
            xr.open_dataset(fpath_precip, use_cftime=True) as precipds:

        # only process and save the gcm data selected --> saves some time!
        if (y0 is not None) or (y1 is not None):
            tempds = tempds.sel(time=slice(y0, y1))
            precipds = precipds.sel(time=slice(y0, y1))
        # Check longitude conventions
        if tempds.lon.min() >= 0 and glon <= 0:
            glon += 360

        # Take the closest to the glacier
        # Should we consider GCM interpolation?
        try:
            # if gcms are not flattened, do:
            # this is the default, so try this first
            temp = tempds.tas.sel(lat=glat, lon=glon, method='nearest')
            precip = precipds.pr.sel(lat=glat, lon=glon, method='nearest')
        except:
            # are the gcms flattened? if yes,
            # compute all the distances and choose the
            # nearest gridpoint
            c_tempds = ((tempds.lon - glon) ** 2 + (tempds.lat - glat) ** 2)
            c_precipds = ((precipds.lon - glon) ** 2 + (precipds.lat - glat) ** 2)
            temp_0 = tempds.isel(points=np.argmin(c_tempds.data))
            precip_0 = precipds.isel(points=np.argmin(c_precipds.data))
            temp = temp_0.tas
            temp['lon'] = temp_0.lon
            temp['lat'] = temp_0.lat
            precip = precip_0.pr
            precip['lon'] = precip_0.lon
            precip['lat'] = precip_0.lat

        # Back to [-180, 180] for OGGM
        temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
        precip.lon.values = precip.lon if precip.lon <= 180 else precip.lon - 360

        # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!
        assert 'kg m-2 s-1' in precip.units, 'Precip units not understood'

        ny, r = divmod(len(temp), 12)
        assert r == 0
        dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in temp['time.month']]
        precip = precip * dimo * (60 * 60 * 24)

    process_gcm_data(gdir, output_filesuffix=output_filesuffix, prcp=precip, temp=temp,
                     source=output_filesuffix, **kwargs)


@entity_task(log, writes=['gcm_data'])
def process_lmr_data(gdir, fpath_temp=None, fpath_precip=None,
                     ensemble_member=None,
                     version='v2.1',
                     year_range=('1951', '1980'),
                     filesuffix='',
                     output_filesuffix='',
                     **kwargs):
    """Read, process and store the Last Millennium Reanalysis (LMR) data for this glacier.

    LMR v2.1 data: https://atmos.washington.edu/~hakim/lmr/LMRv2/
    LMR data:

    LMR data is annualised in anomaly format relative to 1951-1980. (2.1 or online)
    We create synthetic timeseries from the reference data.

    It stores the data in a format that can be used by the OGGM mass balance
    model and in the glacier directory.

    Parameters
    ----------
    fpath_temp : str
        path to the temp file (default: LMR v2.1 from server above)
    fpath_precip : str
        path to the precip file (default: LMR v2.1 from server above)
    version : str
        Version is 'v2.1' or 'online'. Default is 'v2.1'. 'online' is the
        latest version from the server.
    year_range : tuple of str
        the year range for which you want to compute the anomalies. Default
        for LMR is `('1951', '1980')`
    ensemble_member : int
        the ensemble member to use (default is the ensemble mean). An
        integer between 0 and 19.
    filesuffix : str
        Same as `output_filesuffix` but deprecated
    output_filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).

    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    if filesuffix:
        output_filesuffix = filesuffix

    # Get the path of GCM temperature & precipitation data
    if version == 'v2.1':
        base_url = 'https://atmos.washington.edu/%7Ehakim/lmr/LMRv2/'
        if fpath_temp is None:
            with utils.get_lock():
                fpath_temp = utils.file_downloader(
                    base_url + 'air_MCruns_ensemble_mean_LMRv2.1.nc')
        if fpath_precip is None:
            with utils.get_lock():
                fpath_precip = utils.file_downloader(
                    base_url + 'prate_MCruns_ensemble_mean_LMRv2.1.nc')
    elif version == 'online':
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/climate/lmr/LMR_Online/'
        if fpath_temp is None:
            with utils.get_lock():
                fpath_temp = utils.file_downloader(base_url + 'spatial_ens_mean.nc')
    else:
        raise InvalidParamsError('Version not understood: {}'.format(version))

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat

    # Read the GCM files
    if version == 'v2.1':
        with xr.open_dataset(fpath_temp, use_cftime=True) as tempds, \
                xr.open_dataset(fpath_precip, use_cftime=True) as precipds:

            # Check longitude conventions
            if tempds.lon.min() >= 0 and glon <= 0:
                glon += 360

            # Take the closest to the glacier
            # Should we consider GCM interpolation?
            temp = tempds.air.sel(lat=glat, lon=glon, method='nearest')
            precip = precipds.prate.sel(lat=glat, lon=glon, method='nearest')

            # Currently we just take the mean of the ensemble as default
            # The GCM climate will correct anyways
            if ensemble_member is not None:
                temp = temp.isel(MCrun=ensemble_member)
                precip = precip.isel(MCrun=ensemble_member)
            else:
                temp = temp.mean(dim='MCrun')
                precip = precip.mean(dim='MCrun')

    elif version == 'online':
        with xr.open_dataset(fpath_temp, use_cftime=True) as ods:
            # Check longitude conventions
            if ods.lon.min() >= 0 and glon <= 0:
                glon += 360

            temp = ods.tas_sfc.sel(lat=glat, lon=glon, method='nearest')
            precip = ods.pr_sfc.sel(lat=glat, lon=glon, method='nearest')

    # Precip unit is kg/m^2/s we convert to mm month since we apply the anomaly after
    precip = precip * 30.5 * (60 * 60 * 24)

    # Back to [-180, 180] for OGGM
    temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
    precip.lon.values = precip.lon if precip.lon <= 180 else precip.lon - 360

    # OK now we have to turn these annual timeseries in monthly data
    # We take the ref climate
    fpath = gdir.get_filepath('climate_historical')
    with xr.open_dataset(fpath) as ds_ref:
        ds_ref = ds_ref.sel(time=slice(*year_range))

        loc_tmp = ds_ref.temp.groupby('time.month').mean()
        loc_pre = ds_ref.prcp.groupby('time.month').mean()

        # Make time coord
        if version == 'v2.1':
            units = 'days since 0000-01-01 00:00:00'
        elif version == 'online':
            units = 'days since 1000-01-01 00:00:00'

        t = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * len(temp))
        t = cftime.num2date(np.append([0], t[:-1]), units, calendar='noleap')

        temp = xr.DataArray((loc_tmp.data + temp.data[:, np.newaxis]).flatten(),
                            coords={'time': t, 'lon': temp.lon, 'lat': temp.lat},
                            dims=('time',))

        # For precip the std dev is very small - lets keep it as is for now but
        # this is a bit ridiculous. We clip to zero here to be sure
        precip = utils.clip_min((loc_pre.data + precip.data[:, np.newaxis]).flatten(), 0)
        precip = xr.DataArray(precip, dims=('time',),
                              coords={'time': t, 'lon': temp.lon, 'lat': temp.lat})

    process_gcm_data(gdir, output_filesuffix=output_filesuffix,
                     prcp=precip, temp=temp,
                     year_range=year_range, calendar='noleap',
                     source='lmr', **kwargs)


@entity_task(log, writes=['gcm_data'])
def process_modera_data(gdir, fpath_temp=None, fpath_precip=None,
                        y0=None, y1=None,
                        year_range=('1901', '2000'),
                        ensemble_member=None,
                        output_filesuffix='',
                        **kwargs):
    """Read, process and store the ModE-RA climate data for this glacier.

    ModE-RA data was downloaded from WDCC and added to the cluster for
    easy download.

    Data contains anomalies with respect to the period 1901-2000.

    Parameters
    ----------
    fpath_temp : str
        path to the precip file. Defaults to the ensemble statistics
        on server or ensemble member on server.
    fpath_precip : str
        path to the precip file. Defaults to the ensemble statistics
        on server or ensemble member on server.
    y0 : int
        start year of the data processing.
        Default is None which processes the entire timeseries.
        Set this to make process_cmip_data faster if you
        can afford a shorter period.
    y1 : int
        end year of the data processing.
        Default is None which processes the entire timeseries.
        Set this to make process_cmip_data faster if you
        can afford a shorter period.
    year_range : tuple of str
        the year range for which you want to compute the anomalies.
        We default to the period 1901-2000 which is the reference period
        for the anomalies.
    ensemble_member : int
        the ensemble member to use (default is the ensemble mean). An
        integer between 0 and 19.
    output_filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat

    if y0 is not None:
        y0 = str(y0)
    if y1 is not None:
        y1 = str(y1)

    # Get the path of GCM temperature & precipitation data
    if fpath_temp is None:
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/climate/ModE-RA/'
        if ensemble_member is None:
            base_url += 'ens_stats/'
        else:
            base_url += 'ens_members/'

        if fpath_temp is None:
            file_url = 'ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc'
            if ensemble_member is not None:
                eid = int(ensemble_member) + 41
                file_url = f'ModE-RA_m0{eid}_temp2_anom_wrt_1901-2000_1421-2008_mon.nc'
            with utils.get_lock():
                fpath_temp = utils.file_downloader(base_url + file_url)
        if fpath_precip is None:
            file_url = 'ModE-RA_ensmean_totprec_anom_wrt_1901-2000_1421-2008_mon.nc'
            if ensemble_member is not None:
                eid = int(ensemble_member) + 41
                file_url = f'ModE-RA_m0{eid}_totprec_anom_wrt_1901-2000_1421-2008_mon.nc'
            with utils.get_lock():
                fpath_precip = utils.file_downloader(base_url + file_url)

    # Read the GCM files
    with xr.open_dataarray(fpath_temp, use_cftime=True) as tempds, \
            xr.open_dataarray(fpath_precip, use_cftime=True) as precipds:

        # only process and save the gcm data selected --> saves some time!
        if (y0 is not None) or (y1 is not None):
            tempds = tempds.sel(time=slice(y0, y1))
            precipds = precipds.sel(time=slice(y0, y1))

        # Take the closest to the glacier
        # Should we consider GCM interpolation?
        temp = tempds.sel(latitude=glat, longitude=glon, method='nearest')
        precip = precipds.sel(latitude=glat, longitude=glon, method='nearest')

        # Rename coords
        temp = temp.rename({'longitude': 'lon', 'latitude': 'lat'})
        precip = precip.rename({'longitude': 'lon', 'latitude': 'lat'})

        # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!
        assert 'kg m-2 s-1' in precip.units, 'Precip units not understood'

        ny, r = divmod(len(temp), 12)
        assert r == 0
        dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in temp['time.month']]
        precip = precip * dimo * (60 * 60 * 24)

        # Now add the anomalies to the reference data
        # We take the ref climate
        fpath = gdir.get_filepath('climate_historical')
        with xr.open_dataset(fpath) as ds_ref:
            ds_ref = ds_ref.sel(time=slice(*year_range))

            loc_tmp = ds_ref.temp.groupby('time.month').mean()
            loc_pre = ds_ref.prcp.groupby('time.month').mean()

        # Add this to the anomalies
        temp = np.tile(loc_tmp.data, ny) + temp
        precip = np.tile(loc_pre.data, ny) + precip

        # Anomalies are bad and precip can be negative
        precip = utils.clip_min(precip, 0)

    process_gcm_data(gdir, output_filesuffix=output_filesuffix, prcp=precip, temp=temp,
                     source=output_filesuffix, year_range=year_range, calendar='noleap',
                     **kwargs)
