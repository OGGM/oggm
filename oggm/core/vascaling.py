""" Implementation of the 'original' volume/area scaling glacier model from
Marzeion et. al. 2012, see http://www.the-cryosphere.net/6/1295/2012/.
While the mass balance model is comparable to OGGMs past mass balance model,
the 'dynamic' part does not include any ice physics but works with ares/volume
and length/volume scaling instead.

Author: Moritz Oberrauch
"""
# Built ins
import os
import logging
import datetime
from time import gmtime, strftime

# External libs
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from scipy.optimize import minimize_scalar

# import OGGM modules
import oggm
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH

from oggm import __version__

from oggm import utils, entity_task, global_task
from oggm.utils import floatyear_to_date, ncDataset
from oggm.exceptions import InvalidParamsError, MassBalanceCalibrationError

from oggm.core import climate
from oggm.core.massbalance import MassBalanceModel

# Module logger
log = logging.getLogger(__name__)


def _compute_temp_terminus(temp, temp_grad, ref_hgt,
                           terminus_hgt, temp_anomaly=0):
    """Computes the (monthly) mean temperature at the glacier terminus,
    following section 2.1.2 of Marzeion et. al., 2012. The input temperature
    is scaled by the given temperature gradient and the elevation difference
    between reference altitude and the glacier terminus elevation.

    Parameters
    ----------
    temp : netCDF4 variable
        monthly mean climatological temperature (degC)
    temp_grad : netCDF4 variable or float
        temperature lapse rate [degC per m of elevation change]
    ref_hgt : float
        reference elevation for climatological temperature [m asl.]
    terminus_hgt : float
        elevation of the glacier terminus (m asl.)
    temp_anomaly : netCDF4 variable or float, optional
        monthly mean temperature anomaly, default 0

    Returns
    -------
    netCDF4 variable
        monthly mean temperature at the glacier terminus [degC]

    """
    temp_terminus = temp + temp_grad * (terminus_hgt - ref_hgt) + temp_anomaly
    return temp_terminus


def _compute_solid_prcp(prcp, prcp_factor, ref_hgt, min_hgt, max_hgt,
                        temp_terminus, temp_all_solid, temp_grad,
                        prcp_grad=0, prcp_anomaly=0):
    """Compute the (monthly) amount of solid precipitation onto the glacier
    surface, following section 2.1.1 of Marzeion et. al., 2012. The fraction of
    solid precipitation depends mainly on the terminus temperature and the
    temperature thresholds for solid and liquid precipitation. It is possible
    to scale the precipitation amount from the reference elevation to the
    average glacier surface elevation given a gradient (zero per default).

    Parameters
    ----------
    prcp : netCDF4 variable
        monthly mean climatological precipitation [kg/m2]
    prcp_factor : float
        precipitation scaling factor []
    ref_hgt : float
        reference elevation for climatological precipitation [m asl.]
    min_hgt : float
        minimum glacier elevation [m asl.]
    max_hgt : float
        maximum glacier elevation [m asl.]
    temp_terminus : netCDF4 variable
        monthly mean temperature at the glacier terminus [degC]
    temp_all_solid : float
        temperature threshold below which all precipitation is solid [degC]
    temp_grad : netCDF4 variable or float
        temperature lapse rate [degC per m of elevation change]
    prcp_grad : netCDF4 variable or float, optional
        precipitation lapse rate [kg/m2 per m of elevation change], default = 0
    prcp_anomaly : netCDF4 variable or float, optional
        monthly mean precipitation anomaly [kg/m2], default = 0

    Returns
    -------
    netCDF4 variable
        monthly mean solid precipitation [kg/m2]

    """
    # compute fraction of solid precipitation
    if max_hgt == min_hgt:
        # prevent division by zero if max_hgt equals min_hgt
        f_solid = (temp_terminus <= temp_all_solid).astype(int)
    else:
        # use scaling defined in paper
        f_solid = (1 + (temp_terminus - temp_all_solid)
                   / (temp_grad * (max_hgt - min_hgt)))
        f_solid = np.clip(f_solid, 0, 1)

    # compute mean elevation
    mean_hgt = 0.5 * (min_hgt + max_hgt)
    # apply precipitation scaling factor
    prcp_solid = (prcp_factor * prcp + prcp_anomaly)
    # compute solid precipitation
    prcp_solid *= (1 + prcp_grad * (mean_hgt - ref_hgt)) * f_solid

    return prcp_solid


def get_min_max_elevation(gdir):
    """Reads the DEM and computes the minimal and maximal glacier surface
     elevation in meters asl, from the given (RGI) glacier outline.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`

    Returns
    -------
    [float, float]
        minimal and maximal glacier surface elevation [m asl.]

    """
    # open DEM file and mask the glacier surface area
    fpath = gdir.get_filepath('gridded_data')
    with ncDataset(fpath) as nc:
        mask = nc.variables['glacier_mask'][:]
        topo = nc.variables['topo'][:]
    # get relevant elevation information
    min_elev = np.min(topo[np.where(mask == 1)])
    max_elev = np.max(topo[np.where(mask == 1)])

    return min_elev, max_elev


def get_yearly_mb_temp_prcp(gdir, time_range=None, year_range=None):
    """Read climate file and compute mass balance relevant climate parameters.
    Those are the positive melting temperature at glacier terminus elevation
    as energy input and the amount of solid precipitation onto the glacier
    surface as mass input. Both parameters are computes as yearly sums.

    Default is to read all data, but it is possible to specify a time range by
    giving two (included) datetime bounds. Similarly, the year range limits the
    returned data to the given bounds of (hydrological) years.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    time_range : datetime tuple, optional
        [t0, t1] time bounds, default = None
    year_range : float tuple, optional
        [y0, y1] year range, default = None

    Returns
    -------
    [float array, float array, float array]
        hydrological years (index), melting temperature [degC],
        solid precipitation [kg/m2]

    """
    # convert hydrological year range into time range
    if year_range is not None:
        sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
        em = sm - 1 if (sm > 1) else 12
        t0 = datetime.datetime(year_range[0]-1, sm, 1)
        t1 = datetime.datetime(year_range[1], em, 1)
        return get_yearly_mb_temp_prcp(gdir, time_range=[t0, t1])

    # get needed parameters
    temp_all_solid = cfg.PARAMS['temp_all_solid']
    temp_melt = cfg.PARAMS['temp_melt']
    prcp_fac = cfg.PARAMS['prcp_scaling_factor']
    default_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
    # Marzeion et. al., 2012 used a precipitation lapse rate of 3%/100m.
    # But the prcp gradient is omitted for now.
    # prcp_grad = 3e-4
    prcp_grad = 0

    # read the climate file
    igrad = None
    with utils.ncDataset(gdir.get_filepath('climate_monthly'), mode='r') as nc:
        # time
        time = nc.variables['time']
        time = netCDF4.num2date(time[:], time.units)
        # limit data to given time range and
        # raise errors is bounds are outside available data
        if time_range is not None:
            p0 = np.where(time == time_range[0])[0]
            try:
                p0 = p0[0]
            except IndexError:
                raise climate.MassBalanceCalibrationError('time_range[0] '
                                                          'not found in file')
            p1 = np.where(time == time_range[1])[0]
            try:
                p1 = p1[0]
            except IndexError:
                raise climate.MassBalanceCalibrationError('time_range[1] not '
                                                          'found in file')
        else:
            p0 = 0
            p1 = len(time)-1

        time = time[p0:p1+1]

        # read time series of temperature and precipitation
        itemp = nc.variables['temp'][p0:p1+1]
        iprcp = nc.variables['prcp'][p0:p1+1]
        # read time series of temperature lapse rate
        if 'gradient' in nc.variables:
            igrad = nc.variables['gradient'][p0:p1+1]
            # Security for stuff that can happen with local gradients
            igrad = np.where(~np.isfinite(igrad), default_grad, igrad)
            igrad = np.clip(igrad, g_minmax[0], g_minmax[1])
        # read climate data reference elevation
        ref_hgt = nc.ref_hgt

    # use the default gradient if no gradient is supplied by the climate file
    if igrad is None:
        igrad = itemp * 0 + default_grad

    # Up to this point, the code is mainly copy and paste from the
    # corresponding OGGM routine, with some minor adaptions.
    # What follows is my code: So abandon all hope, you who enter here!

    # get relevant elevation information
    min_hgt, max_hgt = get_min_max_elevation(gdir)

    # get temperature at glacier terminus
    temp_terminus = _compute_temp_terminus(itemp, igrad, ref_hgt, min_hgt)
    # compute positive 'melting' temperature/energy input
    temp = np.clip(temp_terminus - temp_melt, a_min=0, a_max=None)
    # get solid precipitation
    prcp_solid = _compute_solid_prcp(iprcp, prcp_fac, ref_hgt,
                                     min_hgt, max_hgt,
                                     temp_terminus, temp_all_solid,
                                     igrad, prcp_grad)

    # check if climate data includes all 12 month of all years
    ny, r = divmod(len(time), 12)
    if r != 0:
        raise ValueError('Climate data should be N full years exclusively')
    # last year gives the tone of the hydro year
    years = np.arange(time[-1].year - ny + 1, time[-1].year + 1, 1)

    # compute sums over hydrological year
    temp_yr = np.zeros(len(years))
    prcp_yr = np.zeros(len(years))
    for i, y in enumerate(years):
        temp_yr[i] = np.sum(temp[i * 12:(i + 1) * 12])
        prcp_yr[i] = np.sum(prcp_solid[i * 12:(i + 1) * 12])

    return years, temp_yr, prcp_yr


def _fallback_local_t_star(gdir):
    """A Fallback function if vascaling.local_t_star raises an Error.

    This function will still write a `vascaling_mustar.json`, filled with NANs,
    if vascaling.local_t_star fails and cfg.PARAMS['continue_on_error'] = True.

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
    gdir.write_json(df, 'vascaling_mustar')


@entity_task(log, writes=['vascaling_mustar'], fallback=_fallback_local_t_star)
def local_t_star(gdir, ref_df=None, tstar=None, bias=None):
    """Compute the local t* and associated glacier-wide mu*.

    If `tstar` and `bias` are not provided, they will be interpolated from the
    reference t* list.
    The mass balance calibration parameters (i.e. temperature lapse rate,
    temperature thresholds for melting, solid and liquid precipitation,
    precipitation scaling factor) are written to the climate_info.pkl file.

    The results of the calibration process (i.e. t*, mu*, bias) are stored in
    the `vascaling_mustar.json` file, to be used later by other tasks.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    ref_df : :py:class:`pandas.Dataframe`, optional
        replace the default calibration list with a custom one
    tstar : int, optional
        the year when the glacier should be in equilibrium, default = None
    bias : float, optional
        the associated reference bias, default = None

    """

    # specify relevant mass balance parameters
    params = ['temp_default_gradient', 'temp_all_solid', 'temp_all_liq',
              'temp_melt', 'prcp_scaling_factor']

    if tstar is None or bias is None:
        # Do our own interpolation of t_start for given glacier
        if ref_df is None:
            if not cfg.PARAMS['run_mb_calibration']:
                # Make some checks and use the default one
                climate_info = gdir.read_json('climate_info')
                source = climate_info['baseline_climate_source']
                ok_source = ['CRU TS4.01', 'CRU TS3.23', 'HISTALP']
                if not np.any(s in source.upper() for s in ok_source):
                    msg = ('If you are using a custom climate file you should '
                           'run your own MB calibration.')
                    raise MassBalanceCalibrationError(msg)

                # major RGI version relevant
                v = gdir.rgi_version[0]
                # baseline climate
                str_s = 'cru4' if 'CRU' in source else 'histalp'
                vn = 'vas_ref_tstars_rgi{}_{}_calib_params'.format(v, str_s)
                for k in params:
                    if cfg.PARAMS[k] != cfg.PARAMS[vn][k]:
                        msg = ('The reference t* you are trying to use was '
                               'calibrated with different MB parameters. You '
                               'might have to run the calibration manually.')
                        raise MassBalanceCalibrationError(msg)

                ref_df = cfg.PARAMS['vas_ref_tstars_rgi{}_{}'.format(v, str_s)]
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
    out = gdir.read_json('climate_info')
    out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in params}
    gdir.write_json(out, 'climate_info')

    # We compute the overall mu* here but this is mostly for testing
    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr = [tstar - mu_hp, tstar + mu_hp]

    # get monthly climatological values
    # of terminus temperature and solid precipitation
    years, temp, prcp = get_yearly_mb_temp_prcp(gdir, year_range=yr)

    # solve mass balance equation for mu*
    # note: calving is not considered
    mustar = np.mean(prcp) / np.mean(temp)

    # check for a finite result
    if not np.isfinite(mustar):
        raise climate.MassBalanceCalibrationError('{} has a non finite '
                                                  'mu'.format(gdir.rgi_id))

    # Clip the mu
    if not (cfg.PARAMS['min_mu_star'] < mustar < cfg.PARAMS['max_mu_star']):
        raise climate.MassBalanceCalibrationError('mu* out of '
                                                  'specified bounds.')

    # Scalars in a small dict for later
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = int(tstar)
    df['bias'] = bias
    df['mu_star'] = mustar
    gdir.write_json(df, 'vascaling_mustar')


@entity_task(log, writes=['climate_info'])
def t_star_from_refmb(gdir, mbdf=None):
    """Computes the reference year t* for the given glacier and mass balance
    measurements.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    mbdf : :py:class:`pd.Series`
        observed MB data indexed by year. If None, read automatically from the
        reference data, default = None

    Returns
    -------
    dict
        A dictionary {'t_star': [], 'bias': []} containing t* and the
        corresponding mass balance bias


    """
    # make sure we have no marine terminating glacier
    assert gdir.terminus_type == 'Land-terminating'
    # get reference time series of mass balance measurements
    if mbdf is None:
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

    # compute average observed mass-balance
    ref_mb = np.mean(mbdf)

    # Compute one mu candidate per year and the associated statistics
    # Only get the years were we consider looking for t*
    y0, y1 = cfg.PARAMS['tstar_search_window']
    ci = gdir.read_json('climate_info')
    y0 = y0 or ci['baseline_hydro_yr_0']
    y1 = y1 or ci['baseline_hydro_yr_1']
    years = np.arange(y0, y1+1)

    ny = len(years)
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    mb_per_mu = pd.Series(index=years)

    # get mass balance relevant climate parameters
    years, temp, prcp = get_yearly_mb_temp_prcp(gdir, year_range=[y0, y1])

    # get climate parameters, but only for years with mass balance measurements
    selind = np.searchsorted(years, mbdf.index)
    sel_temp = temp[selind]
    sel_prcp = prcp[selind]
    sel_temp = np.mean(sel_temp)
    sel_prcp = np.mean(sel_prcp)

    # for each year in the climatic period around t* (ignoring the first and
    # last 15-years), compute a mu-candidate by solving the mass balance
    # equation for mu. afterwards compute the average (modeled) mass balance
    # over all years with mass balance measurements using the mu-candidate
    for i, y in enumerate(years):
        # ignore begin and end, i.e. if the
        if ((i - mu_hp) < 0) or ((i + mu_hp) >= ny):
            continue

        # compute average melting temperature
        t_avg = np.mean(temp[i - mu_hp:i + mu_hp + 1])
        # skip if if too cold, i.e. no melt occurs (division by zero)
        if t_avg < 1e-3:
            continue
        # compute the mu candidate for the current year, by solving the mass
        # balance equation for mu*
        mu = np.mean(prcp[i - mu_hp:i + mu_hp + 1]) / t_avg

        # compute mass balance using the calculated mu and the average climate
        # conditions over the years with mass balance records
        mb_per_mu[y] = np.mean(sel_prcp - mu * sel_temp)

    # compute differences between computed mass balance and reference value
    diff = (mb_per_mu - ref_mb).dropna()
    # raise error if no mu could be calculated for any year
    if len(diff) == 0:
        raise MassBalanceCalibrationError('No single valid mu candidate for '
                                          'this glacier!')

    # choose mu* as the mu candidate with the smallest absolute bias
    amin = np.abs(diff).idxmin()

    # write results to the `climate_info.pkl`
    d = gdir.read_json('climate_info')
    d['t_star'] = amin
    d['bias'] = diff[amin]
    gdir.write_json(d, 'climate_info')

    return {'t_star': amin, 'bias': diff[amin],
            'avg_mb_per_mu': mb_per_mu, 'avg_ref_mb': ref_mb}


@global_task
def compute_ref_t_stars(gdirs):
    """Detects the best t* for the reference glaciers and writes them to disk

    This task will be needed for mass balance calibration of custom climate
    data. For CRU and HISTALP baseline climate a pre-calibrated list is
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


@entity_task(log)
def find_start_area(gdir, year_start=1851):
    """This task find the start area for the given glacier, which results in
    the best results after the model integration (i.e., modeled glacier surface
    closest to measured RGI surface in 2003).

    All necessary prepro task (gis, centerline, climate) must be executed
    beforehand, as well as the local_t_star() task.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    year_start : int, optional
        year at the beginning of the model integration, default = 1851
        (best choice for working with HISTALP data)

    Returns
    -------
    :py:class:`scipy.optimize.OptimizeResult`

    """

    # instance the mass balance models
    mbmod = VAScalingMassBalance(gdir)

    # get reference area and year from RGI
    a_rgi = gdir.rgi_area_m2
    rgi_df = utils.get_rgi_glacier_entities([gdir.rgi_id])
    y_rgi = int(rgi_df.BgnDate.values[0][:4])
    # get min and max glacier surface elevation
    h_min, h_max = get_min_max_elevation(gdir)

    # set up the glacier model with the reference values (from RGI)
    model_ref = VAScalingModel(year_0=y_rgi, area_m2_0=a_rgi,
                               min_hgt=h_min, max_hgt=h_max,
                               mb_model=mbmod)

    def _to_minimize(area_m2_start, ref, year_start=year_start):
        """Initialize VAS glacier model as copy of the reference model (ref)
        and adjust the model to the given starting area (area_m2_start) and
        starting year (1851). Let the model evolve to the same year as the
        reference model. Compute and return the relative absolute area error.

        Parameters
        ----------
        area_m2_start : float
        ref : :py:class:`oggm.VAScalingModel`
        year_start : float, optional
             the default value is inherited from the surrounding task

        Returns
        -------
        float
            relative absolute area estimate error

        """
        # define model
        model_tmp = VAScalingModel(year_0=ref.year_0,
                                   area_m2_0=ref.area_m2_0,
                                   min_hgt=ref.min_hgt_0,
                                   max_hgt=ref.max_hgt,
                                   mb_model=ref.mb_model)
        # scale to desired starting size
        model_tmp.create_start_glacier(area_m2_start, year_start=year_start)
        # run and compare, return relative error
        return np.abs(model_tmp.run_and_compare(ref))

    # define bounds - between 100m2 and two times the reference size
    area_m2_bounds = [100, 2 * model_ref.area_m2_0]
    # run minimization
    minimization_res = minimize_scalar(_to_minimize, args=(model_ref),
                                       bounds=area_m2_bounds,
                                       method='bounded')

    return minimization_res


class VAScalingMassBalance(MassBalanceModel):
    """Original mass balance model, used in Marzeion et. al., 2012.
    The general concept is similar to the oggm.PastMassBalance model.
    Thereby the main difference is that the Volume/Area Scaling mass balance
    model returns only one glacier wide mass balance value per month or year.
    """

    def __init__(self, gdir, mu_star=None, bias=None,
                 filename='climate_monthly', input_filesuffix='',
                 repeat=False, ys=None, ye=None, check_calib_params=True):
        """Initialize.

        Parameters
        ----------
        gdir : :py:class:`oggm.GlacierDirectory`
        mu_star : float, optional
            set to the alternative value of mu* you want to use, while
            the default is to use the calibrated value
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data
        input_filesuffix : str, optional
            the file suffix of the input climate file, no suffix as default
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way, default=False
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated mu* by checking
            the parameters used during calibration and the ones you are
            using at run time. If they don't match, it will raise an error.
            Set to False to suppress this check.

        """
        # initalize of oggm.MassBalanceModel
        super(VAScalingMassBalance, self).__init__()

        # read mass balance parameters from file
        if mu_star is None:
            df = gdir.read_json('vascaling_mustar')
            mu_star = df['mu_star']
        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                df = gdir.read_json('vascaling_mustar')
                bias = df['bias']
            else:
                bias = 0.
        # set mass balance parameters
        self.mu_star = mu_star
        self.bias = bias

        # set mass balance calibration parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_liq = cfg.PARAMS['temp_all_liq']
        self.t_melt = cfg.PARAMS['temp_melt']
        prcp_fac = cfg.PARAMS['prcp_scaling_factor']
        default_grad = cfg.PARAMS['temp_default_gradient']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = gdir.read_json('climate_info')['mb_calib_params']
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    raise RuntimeError('You seem to use different mass-'
                                       'balance parameters than used for the '
                                       'calibration. '
                                       'Set `check_calib_params=False` '
                                       'to ignore this warning.')

        # set public attributes
        self.temp_bias = 0.
        self.prcp_bias = 1.
        self.repeat = repeat
        self.hemisphere = gdir.hemisphere

        # read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with ncDataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError('Climate data should be N full years')
            # This is where we switch to hydro float year format
            # Last year gives the tone of the hydro year
            self.years = np.repeat(np.arange(time[-1].year-ny+1,
                                             time[-1].year+1), 12)
            self.months = np.tile(np.arange(1, 13), ny)
            # Read timeseries
            self.temp = nc.variables['temp'][:]
            self.prcp = nc.variables['prcp'][:] * prcp_fac
            if 'gradient' in nc.variables:
                grad = nc.variables['gradient'][:]
                # Security for stuff that can happen with local gradients
                g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
                grad = np.where(~np.isfinite(grad), default_grad, grad)
                grad = np.clip(grad, g_minmax[0], g_minmax[1])
            else:
                grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.ys = self.years[0] if ys is None else ys
            self.ye = self.years[-1] if ye is None else ye

        # compute climatological precipitation around t*
        # needed later to estimate the volume/lenght scaling parameter
        t_star = gdir.read_json('vascaling_mustar')['t_star']
        mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
        yr = [t_star - mu_hp, t_star + mu_hp]
        _, _, prcp_clim = get_yearly_mb_temp_prcp(gdir, year_range=yr)
        # convert from [mm we. yr-1] into SI units [m we. yr-1]
        prcp_clim = prcp_clim * 1e-3
        self.prcp_clim = np.mean(prcp_clim)

    def get_monthly_climate(self, min_hgt, max_hgt, year):
        """Compute and return monthly positive terminus temperature
        and solid precipitation amount for given month.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier surface elevation [m asl.]
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        [float, float]
            (temp_for_melt) positive terminus temperature [degC] and
            (prcp_solid) solid precipitation amount [kg/m^2]

        """
        # process given time index
        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if y < self.ys or y > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(y, self.ys, self.ye))
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = self.prcp[pok] * self.prcp_bias
        igrad = self.grad[pok]

        # compute terminus temperature
        temp_terminus = _compute_temp_terminus(itemp, igrad,
                                               self.ref_hgt, min_hgt)
        # compute positive 'melting' temperature/energy input
        temp_for_melt = np.clip(temp_terminus - self.t_melt,
                                a_min=0, a_max=None)
        # compute solid precipitation
        prcp_solid = _compute_solid_prcp(iprcp, 1,
                                         self.ref_hgt, min_hgt, max_hgt,
                                         temp_terminus, self.t_solid, igrad)

        return temp_for_melt, prcp_solid

    def get_monthly_mb(self, min_hgt, max_hgt, year):
        """Compute and return the glacier wide mass balance
        for the given year/month combination.
        Possible mb bias is applied...

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier (surface) elevation [m asl.]
        year : float
            floating year and month, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        # get melting temperature and solid precipitation
        temp_for_melt, prcp_solid = self.get_monthly_climate(min_hgt,
                                                             max_hgt,
                                                             year=year)
        # compute mass balance
        mb_month = prcp_solid - self.mu_star * temp_for_melt
        # apply mass balance bias
        mb_month -= self.bias / SEC_IN_YEAR * SEC_IN_MONTH
        # convert into SI units [m_ice/s]
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_climate(self, min_hgt, max_hgt, year):
        """Compute and return monthly positive terminus temperature
        and solid precipitation amount for all months
        of the given (hydrological) year.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier (surface) elevation [m asl.]
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        [float array, float array]
            (temp_for_melt) monthly positive terminus temperature [degC] and
            (prcp_solid) monthly solid precipitation amount [kg/m2]

        """
        # process given time index
        year = np.floor(year)
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if year < self.ys or year > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(year, self.ys, self.ye))
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = self.prcp[pok] * self.prcp_bias
        igrad = self.grad[pok]

        # compute terminus temperature
        temp_terminus = _compute_temp_terminus(itemp, igrad,
                                               self.ref_hgt, min_hgt)
        # compute positive 'melting' temperature/energy input
        temp_for_melt = np.clip(temp_terminus - self.t_melt,
                                a_min=0, a_max=None)
        # compute solid precipitation
        # prcp factor is set to 1 since it the time series is already corrected
        prcp_solid = _compute_solid_prcp(iprcp, 1,
                                         self.ref_hgt, min_hgt, max_hgt,
                                         temp_terminus, self.t_solid, igrad)

        return temp_for_melt, prcp_solid

    def get_annual_mb(self, min_hgt, max_hgt, year):
        """Compute and return the annual glacier wide mass balance for the
        given year. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        # get annual mass balance climate
        temp_for_melt, prcp_solid = self.get_annual_climate(min_hgt,
                                                            max_hgt,
                                                            year)
        # compute mass balance
        mb_annual = np.sum(prcp_solid - self.mu_star * temp_for_melt)
        # apply bias and convert into SI units
        return (mb_annual - self.bias) / SEC_IN_YEAR / self.rho

    def get_specific_mb(self, min_hgt, max_hgt, year):
        """Compute and return the annual specific mass balance
        for the given year. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            float year, using the hydrological year convention

        Returns
        -------
        float
            glacier wide average mass balance, units of millimeter water
            equivalent per year [mm w.e./yr]

        """
        # get annual mass balance climate
        temp_for_melt, prcp_solid = self.get_annual_climate(min_hgt,
                                                            max_hgt,
                                                            year)
        # compute mass balance
        mb_annual = np.sum(prcp_solid - self.mu_star * temp_for_melt)
        # apply bias
        return mb_annual - self.bias

    def get_monthly_specific_mb(self, min_hgt=None, max_hgt=None, year=None):
        """Compute and return the monthly specific mass balance
        for the given month. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float, optional
            glacier terminus elevation [m asl.], default = None
        max_hgt : float, optional
            maximal glacier (surface) elevation [m asl.], default = None
        year : float, optional
            float year and month, using the hydrological year convention,
            default = None

        Returns
        -------
        float
            glacier wide average mass balance, units of millimeter water
            equivalent per months [mm w.e./yr]

        """
        # get annual mass balance climate
        temp_for_melt, prcp_solid = self.get_monthly_climate(min_hgt,
                                                             max_hgt,
                                                             year)
        # compute mass balance
        mb_monthly = np.sum(prcp_solid - self.mu_star * temp_for_melt)
        # apply bias and return
        return mb_monthly - (self.bias / SEC_IN_YEAR * SEC_IN_MONTH)

    def get_ela(self, year=None):
        """The ELA can not be calculated using this mass balance model.

        Parameters
        ----------
        year : float, optional

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError('The equilibrium line altitude can not be ' +
                                  'computed for the `VAScalingMassBalance` ' +
                                  'model.')


class RandomVASMassBalance(MassBalanceModel):
    """Random shuffle of all MB years within a given time period.

    This is useful for finding a possible past glacier state or for sensitivity
    experiments.

    Note that this is going to be sensitive to extreme years in certain
    periods, but it is by far more physically reasonable than other
    approaches based on gaussian assumptions.

    Parameters
    ----------

    """

    def __init__(self, gdir, mu_star=None, bias=None,
                 y0=None, halfsize=15, seed=None, filename='climate_monthly',
                 input_filesuffix='', all_years=False,
                 unique_samples=False):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        seed : int, optional
            Random seed used to initialize the pseudo-random number generator.
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        all_years : bool
            if True, overwrites ``y0`` and ``halfsize`` to use all available
            years.
        unique_samples: bool
            if true, chosen random mass-balance years will only be available
            once per random climate period-length
            if false, every model year will be chosen from the random climate
            period with the same probability
        """

        super(RandomVASMassBalance, self).__init__()
        # initialize the VAS equivalent of the PastMassBalance model over the
        # whole available climate period
        self.mbmod = VAScalingMassBalance(gdir, mu_star=mu_star, bias=bias,
                                          filename=filename,
                                          input_filesuffix=input_filesuffix)

        # get mb model parameters
        self.prcp_clim = self.mbmod.prcp_clim

        # define years of climate period
        if all_years:
            # use full climate period
            self.years = self.mbmod.years
        else:
            if y0 is None:
                # choose t* as center of climate period
                df = gdir.read_json('vascaling_mustar')
                self.y0 = df['t_star']
            else:
                # set y0 as attribute
                self.y0 = y0
            # use 31-year period around given year `y0`
            self.years = np.arange(self.y0-halfsize, self.y0+halfsize+1)
        # define year range and number of years
        self.yr_range = (self.years[0], self.years[-1]+1)
        self.ny = len(self.years)
        self.hemisphere = gdir.hemisphere

        # define random state
        self.rng = np.random.RandomState(seed)
        self._state_yr = dict()

        # whether or not to sample with or without replacement
        self.unique_samples = unique_samples
        if self.unique_samples:
            self.sampling_years = self.years

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.temp_bias = value

    @property
    def prcp_bias(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_bias

    @prcp_bias.setter
    def prcp_bias(self, value):
        """Precipitation factor to apply to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.prcp_bias = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.bias = value

    def get_state_yr(self, year=None):
        """For a given year, get the random year associated to it."""
        year = int(year)
        if year not in self._state_yr:
            if self.unique_samples:
                # --- Sampling without replacement ---
                if self.sampling_years.size == 0:
                    # refill sample pool when all years were picked once
                    self.sampling_years = self.years
                # choose one year which was not used in the current period
                _sample = self.rng.choice(self.sampling_years)
                # write chosen year to dictionary
                self._state_yr[year] = _sample
                # update sample pool: remove the chosen year from it
                self.sampling_years = np.delete(
                    self.sampling_years,
                    np.where(self.sampling_years == _sample))
            else:
                # --- Sampling with replacement ---
                self._state_yr[year] = self.rng.randint(*self.yr_range)
        return self._state_yr[year]

    def get_monthly_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the glacier wide mass balance
        for the given year/month combination.
        Possible mb bias is applied...

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier (surface) elevation [m asl.]
        year : float
            floating year and month, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        ryr, m = floatyear_to_date(year)
        ryr = utils.date_to_floatyear(self.get_state_yr(ryr), m)
        return self.mbmod.get_monthly_mb(min_hgt, max_hgt, year=ryr)

    def get_annual_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the annual glacier wide mass balance for the given
        year. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_annual_mb(min_hgt, max_hgt, year=ryr)

    def get_specific_mb(self, min_hgt, max_hgt, year):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the annual specific mass balance for the given year.
        Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            float year, using the hydrological year convention

        Returns
        -------
        float
            glacier wide average mass balance, units of millimeter water
            equivalent per year [mm w.e./yr]

        """
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_specific_mb(min_hgt, max_hgt, year=ryr)

    def get_ela(self, year=None):
        """The ELA can not be calculated using this mass balance model.

        Parameters
        ----------
        year : float, optional

        Raises
        -------
        NotImplementedError

        """
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_ela(year=ryr)


@entity_task(log)
def run_random_climate(gdir, nyears=1000, y0=None, halfsize=15,
                       bias=None, seed=None, temperature_bias=None,
                       climate_filename='climate_monthly',
                       climate_input_filesuffix='', output_filesuffix='',
                       init_area_m2=None, unique_samples=False):
    """Runs the random mass balance model for a given number of years.

    This initializes a :py:class:`oggm.core.vascaling.RandomVASMassBalance`,
    and runs and stores a :py:class:`oggm.core.vascaling.VAScalingModel` with
    the given mass balance model.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int, optional
        length of the simulation, default = 1000
    y0 : int, optional
        central year of the random climate period. The default is to be
        centred on t*. Default = None
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1),
        default = 15
    bias : float, optional
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero. Default = None
    seed : int
        seed for the random generator. If you ignore this, the runs will be
        different each time. Setting it to a fixed seed accross glaciers can
        be usefull if you want to have the same climate years for all of them
    temperature_bias : float, optional
        add a bias to the temperature timeseries, default = None
    climate_filename : str, optional
        name of the climate file, e.g. 'climate_monthly' (default) or
        'gcm_data'
    climate_input_filesuffix: str, optional
        filesuffix for the input climate file
    output_filesuffix : str, optional
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    init_area_m2: float, optional
        glacier area with which the model is initialized, default is RGI value
    unique_samples: bool, optional
        if true, chosen random mass-balance years will only be available once
        per random climate period-length
        if false, every model year will be chosen from the random climate
        period with the same probability (default)

    Returns
    -------
    :py:class:`oggm.core.vascaling.VAScalingModel`
    """

    # instance mass balance model
    mb_mod = RandomVASMassBalance(gdir, y0=y0, halfsize=halfsize, bias=bias,
                                  seed=seed, filename=climate_filename,
                                  input_filesuffix=climate_input_filesuffix,
                                  unique_samples=unique_samples)

    if temperature_bias is not None:
        # add given temperature bias to mass balance model
        mb_mod.temp_bias = temperature_bias

    # where to store the model output
    diag_path = gdir.get_filepath('model_diagnostics', filesuffix='vas',
                                  delete=True)

    # instance the model
    min_hgt, max_hgt = get_min_max_elevation(gdir)
    if init_area_m2 is None:
        init_area_m2 = gdir.rgi_area_m2
    model = VAScalingModel(year_0=0, area_m2_0=init_area_m2,
                           min_hgt=min_hgt, max_hgt=max_hgt,
                           mb_model=mb_mod)
    # specify path where to store model diagnostics
    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)
    # run model
    model.run_until_and_store(year_end=nyears, diag_path=diag_path)

    return model


class ConstantVASMassBalance(MassBalanceModel):
    """Constant mass-balance during a chosen period.

    This is useful for equilibrium experiments.

    """

    def __init__(self, gdir, mu_star=None, bias=None,
                 y0=None, halfsize=15, filename='climate_monthly',
                 input_filesuffix=''):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        """

        super(ConstantVASMassBalance, self).__init__()
        # initialize the VAS equivalent of the PastMassBalance model over the
        # whole available climate period
        self.mbmod = VAScalingMassBalance(gdir, mu_star=mu_star, bias=bias,
                                          filename=filename,
                                          input_filesuffix=input_filesuffix)

        # use t* as the center of the climatological period if not given
        if y0 is None:
            df = gdir.read_json('vascaling_mustar')
            y0 = df['t_star']

        # set model properties
        self.prcp_clim = self.mbmod.prcp_clim
        self.y0 = y0
        self.halfsize = halfsize
        self.years = np.arange(y0 - halfsize, y0 + halfsize + 1)
        self.hemisphere = gdir.hemisphere

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.temp_bias = value

    @property
    def prcp_bias(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_bias

    @prcp_bias.setter
    def prcp_bias(self, value):
        """Precipitation factor to apply to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.prcp_bias = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.bias = value

    def get_climate(self, min_hgt, max_hgt, year=None):
        """Average mass balance climate information for given glacier.

        Note that prcp is corrected with the precipitation factor and that
        all other biases (precipitation, temp) are applied.

        Returns
        -------
        [float, float]
            (temp_for_melt) positive terminus temperature [degC] and
            (prcp_solid) solid precipitation amount [kg/m^2]
        """
        # create monthly timeseries over whole climate period
        yrs = utils.monthly_timeseries(self.years[0], self.years[-1],
                                       include_last_year=True)
        # create empty containers
        temp = list()
        prcp = list()
        # iterate over all months
        for i, yr in enumerate(yrs):
            # get positive melting temperature and solid precipitaion
            t, p = self.mbmod.get_monthly_climate(min_hgt, max_hgt, year=yr)
            temp.append(t)
            prcp.append(p)
        # Note that we do not weight for number of days per month - bad
        return (np.mean(temp, axis=0),
                np.mean(prcp, axis=0))

    def get_monthly_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the glacier wide mass balance
        for the given year/month combination.
        Possible mb bias is applied...

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier (surface) elevation [m asl.]
        year : float
            floating year and month, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        # extract month from year
        _, m = utils.floatyear_to_date()
        # sum up the mass balance over all years in climate period
        years = [utils.date_to_floatyear(yr, m) for yr in self.years]
        mb = [self.mbmod.get_annual_mb(min_hgt, max_hgt, year=yr)
              for yr in years]
        # return average value
        return np.average(mb)

    def get_annual_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the annual glacier wide mass balance for the given
        year. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        # sum up the mass balance over all years in climate period
        mb = [self.mbmod.get_annual_mb(min_hgt, max_hgt, year=yr)
              for yr in self.years]
        # return average value
        return np.average(mb)

    def get_specific_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the annual specific mass balance for the given year.
        Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            float year, using the hydrological year convention

        Returns
        -------
        float
            glacier wide average mass balance, units of millimeter water
            equivalent per year [mm w.e./yr]

        """
        mb = [self.mbmod.get_specific_mb(min_hgt, max_hgt, year=yr)
              for yr in self.years]
        # return average value
        return np.average(mb)

    def get_ela(self, year=None):
        """The ELA can not be calculated using this mass balance model.

        Parameters
        ----------
        year : float, optional

        Raises
        -------
        NotImplementedError

        """
        return self.mbmod.get_ela(year=self.y0)


@entity_task(log)
def run_constant_climate(gdir, nyears=1000, y0=None, halfsize=15,
                         bias=None, temperature_bias=None,
                         climate_filename='climate_monthly',
                         climate_input_filesuffix='', output_filesuffix='',
                         init_area_m2=None):
    """
    Runs the constant mass balance model for a given number of years.

    This initializes a :py:class:`oggm.core.vascaling.ConstantVASMassBalance`,
    and runs and stores a :py:class:`oggm.core.vascaling.VAScalingModel` with
    the given mass balance model.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int, optional
        length of the simulation, default = 1000
    y0 : int, optional
        central year of the random climate period. The default is to be
        centred on t*. Default = None
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1),
        default = 15
    bias : float, optional
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero. Default = None
    temperature_bias : float, optional
        add a bias to the temperature timeseries, default = None
    climate_filename : str, optional
        name of the climate file, e.g. 'climate_monthly' (default) or
        'gcm_data'
    climate_input_filesuffix: str, optional
        filesuffix for the input climate file
    output_filesuffix : str, optional
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    init_area_m2: float, optional
        glacier area with which the model is initialized, default is RGI value

    Returns
    -------
    :py:class:`oggm.core.vascaling.VAScalingModel`
    """

    # instance mass balance model
    mb_mod = ConstantVASMassBalance(gdir, mu_star=None, bias=bias, y0=y0,
                                    halfsize=halfsize,
                                    filename=climate_filename,
                                    input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        # add given temperature bias to mass balance model
        mb_mod.temp_bias = temperature_bias

    # instance the model
    min_hgt, max_hgt = get_min_max_elevation(gdir)
    if init_area_m2 is None:
        init_area_m2 = gdir.rgi_area_m2
    model = VAScalingModel(year_0=0, area_m2_0=init_area_m2,
                           min_hgt=min_hgt, max_hgt=max_hgt,
                           mb_model=mb_mod)
    # specify path where to store model diagnostics
    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)
    # run model
    model.run_until_and_store(year_end=nyears, diag_path=diag_path)

    return model


class VAScalingModel(object):
    """The volume area scaling glacier model following Marzeion et. al., 2012.

    @TODO: finish DocString

    All used parameters are in SI units (even the climatological precipitation
    (attribute of the mass balance model) is given in [m. we yr-1]).

    Parameters
    ----------

    """

    def __repr__(self):
        """Object representation."""
        return "{}: {}".format(self.__class__, self.__dict__)

    def __str__(self):
        """String representation of the dynamic model, includes current
        year, area, volume, length and terminus elevation."""
        return "{}\nyear: {}\n".format(self.__class__, self.year) \
            + "area [km2]: {:.2f}\n".format(self.area_m2 / 1e6) \
            + "volume [km3]: {:.3f}\n".format(self.volume_m3 / 1e9) \
            + "length [km]: {:.2f}\n".format(self.length_m / 1e3) \
            + "min elev [m asl.]: {:.0f}\n".format(self.min_hgt) \
            + "spec mb [mm w.e. yr-1]: {:.2f}".format(self.spec_mb)

    def __init__(self, year_0, area_m2_0, min_hgt, max_hgt, mb_model):
        """Instance new glacier model.

        year_0: float
            year when the simulation starts
        area_m2_0: float
            starting area at year_0 [m2]
        min_hgt: float
            glacier terminus elevation at year_0 [m asl.]
        max_hgt: float
            maximal glacier surface elevation at year_0 [m asl.]
        mb_model: :py:class:òggm.core.vascaling.VAScalingMassBalance`
            instance of mass balance model
        """

        # get constants from cfg.PARAMS
        self.rho = cfg.PARAMS['ice_density']

        # get scaling constants
        self.cl = cfg.PARAMS['vas_c_length_m']
        self.ca = cfg.PARAMS['vas_c_area_m2']

        # get scaling exponents
        self.ql = cfg.PARAMS['vas_q_length']
        self.gamma = cfg.PARAMS['vas_gamma_area']

        # define temporal index
        self.year_0 = year_0
        self.year = year_0

        # define geometrical/spatial parameters
        self.area_m2_0 = area_m2_0
        self.area_m2 = area_m2_0
        self.min_hgt = min_hgt
        self.min_hgt_0 = min_hgt
        self.max_hgt = max_hgt

        # compute volume (m3) and length (m) from area (using scaling laws)
        self.volume_m3_0 = self.ca * self.area_m2_0 ** self.gamma
        self.volume_m3 = self.volume_m3_0
        # self.length = self.cl * area_0**self.ql
        self.length_m_0 = (self.volume_m3 / self.cl) ** (1 / self.ql)
        self.length_m = self.length_m_0

        # define mass balance model and spec mb
        self.mb_model = mb_model
        self.spec_mb = self.mb_model.get_specific_mb(self.min_hgt,
                                                     self.max_hgt,
                                                     self.year)
        # create geometry change parameters
        self.dL = 0
        self.dA = 0
        self.dV = 0

        # create time scale parameters
        self.tau_a = 1
        self.tau_l = 1

    def _get_specific_mb(self):
        """Invoke `get_specific_mb()` from mass balance model for current year
        and glacier terminus elevation."""
        self.spec_mb = self.mb_model.get_specific_mb(self.min_hgt,
                                                     self.max_hgt,
                                                     self.year)

    def _compute_time_scales(self):
        """Compute the time scales for glacier length `tau_l`
        and glacier surface area `tau_a` for current time step."""
        self.tau_l = self.volume_m3 / (self.mb_model.prcp_clim * self.area_m2)
        self.tau_a = self.tau_l * self.area_m2 / self.length_m ** 2

    def reset(self):
        """Set model attributes back to starting values."""
        self.year = self.year_0
        self.length_m = self.length_m_0
        self.area_m2 = self.area_m2_0
        self.volume_m3 = self.volume_m3_0
        self.min_hgt = self.min_hgt_0

        # define mass balance model and spec mb
        self._get_specific_mb()

        # reset geometry change parameters
        self.dL = 0
        self.dA = 0
        self.dV = 0

        # create time scale parameters
        self.tau_a = 1
        self.tau_l = 1
        pass

    def step(self):
        """Advance model glacier by one year. This includes the following:
            - computing time scales
            - computing the specific mass balance
            - computing volume change and new volume
            - computing area change and new area
            - computing length change and new length
            - computing new terminus elevation
        """
        # compute time scales
        self._compute_time_scales()

        # get specific mass balance B(t)
        self._get_specific_mb()

        # compute volume change dV(t)
        self.dV = self.area_m2 * self.spec_mb / self.rho
        # compute new volume V(t+1)
        self.volume_m3 += self.dV

        # compute area change dA(t)
        self.dA = ((self.volume_m3 / self.ca) ** (1 / self.gamma)
                   - self.area_m2) / self.tau_a
        # compute new area A(t+1)
        self.area_m2 += self.dA
        # compute length change dL(t)
        self.dL = ((self.volume_m3 / self.cl) ** (1 / self.ql)
                   - self.length_m) / self.tau_l
        # compute new length L(t+1)
        self.length_m += self.dL
        # compute new terminus elevation min_hgt(t+1)
        self.min_hgt = self.max_hgt + (self.length_m / self.length_m_0
                                       * (self.min_hgt_0 - self.max_hgt))

        # increment year
        self.year += 1

    def run_until(self, year_end, reset=False):
        """Runs the model till the specified year.
        Returns all geometric parameters (i.e. length, area, volume, terminus
        elevation and specific mass balance) at the end of the model evolution.

        Parameters
        ----------
        year_end : float
            end of modeling period
        reset : bool, optional
            If `True`, the model will start from `year_0`, otherwise from its
            current position in time (default).

        Returns
        -------
        [float, float, float, float, float, float]
            the geometric glacier parameters at the end of the model evolution:
            year, length [m], area [m2], volume [m3], terminus elevation
            [m asl.], specific mass balance [mm w.e.]

        """

        # reset parameters to starting values
        if reset:
            self.reset()

        # check validity of end year
        if year_end < self.year:
            # raise warning if model year already past given year, and don't
            # run the model - return current parameters
            raise Warning('Cannot run until {}, already at year {}'.format(
                year_end, self.year))
        else:
            # iterate over all years
            while self.year < year_end:
                # run model for one year
                self.step()

        # return metrics
        return (self.year, self.length_m, self.area_m2,
                self.volume_m3, self.min_hgt, self.spec_mb)

    def run_until_and_store(self, year_end, diag_path=None, reset=False):
        """Runs the model till the specified year. Returns all relevant
        parameters (i.e. length, area, volume, terminus elevation and specific
        mass balance) for each time step as a xarray.Dataset. If a file path is
        give the dataset is written to file.

        Parameters
        ----------
        year_end : float
            end of modeling period
        diag_path : str, optional
            path where to store glacier diagnostics, default = None
        reset : bool, optional
            If `True`, the model will start from `year_0`, otherwise from its
            current position in time (default).

        Returns
        -------
        :py:class:`xarray.Dataset`
            model parameters for each time step (year)

        """
        # reset parameters to starting values
        # TODO: is this OGGM compatible
        if reset:
            self.reset()

        # check validity of end year
        # TODO: find out how OGGM handles this
        if year_end < self.year:
            raise ValueError('Cannot run until {}, already at year {}'.format(
                year_end, self.year))

        if not self.mb_model.hemisphere:
            raise InvalidParamsError('run_until_and_store needs a '
                                     'mass-balance model with an unambiguous '
                                     'hemisphere.')

        # define different temporal indices
        yearly_time = np.arange(np.floor(self.year), np.floor(year_end) + 1)

        # TODO: include `store_monthly_step` in parameter list or remove IF:
        store_monthly_step = False
        if store_monthly_step:
            # get monthly time index
            monthly_time = utils.monthly_timeseries(self.year, year_end)
        else:
            # monthly time
            monthly_time = yearly_time.copy()
        # get years and month for hydrological year and calender year
        yrs, months = utils.floatyear_to_date(monthly_time)
        sm = cfg.PARAMS['hydro_month_' + self.mb_model.hemisphere]
        cyrs, cmonths = utils.hydrodate_to_calendardate(yrs, months,
                                                        start_month=sm)

        # get number of temporal indices
        ny = len(yearly_time)
        nm = len(monthly_time)
        # deal with one dimensional temporal indices
        if ny == 1:
            yrs = [yrs]
            cyrs = [cyrs]
            months = [months]
            cmonths = [cmonths]

        # initialize diagnostics output file
        diag_ds = xr.Dataset()

        # Global attributes
        diag_ds.attrs['description'] = 'VAS model output'
        diag_ds.attrs['oggm_version'] = __version__
        diag_ds.attrs['calendar'] = '365-day no leap'
        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                  gmtime())
        diag_ds.attrs['hemisphere'] = self.mb_model.hemisphere

        # Coordinates
        diag_ds.coords['time'] = ('time', monthly_time)
        diag_ds.coords['hydro_year'] = ('time', yrs)
        diag_ds.coords['hydro_month'] = ('time', months)
        diag_ds.coords['calendar_year'] = ('time', cyrs)
        diag_ds.coords['calendar_month'] = ('time', cmonths)
        # add description as attribute to coordinates
        diag_ds['time'].attrs['description'] = 'Floating hydrological year'
        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'
        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'

        # create empty variables and attributes
        diag_ds['volume_m3'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
        diag_ds['volume_m3'].attrs['unit'] = 'm 3'
        diag_ds['area_m2'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
        diag_ds['area_m2'].attrs['unit'] = 'm 2'
        diag_ds['length_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['length_m'].attrs['description'] = 'Glacier length'
        diag_ds['length_m'].attrs['unit'] = 'm 3'
        diag_ds['ela_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['ela_m'].attrs['description'] = ('Annual Equilibrium Line '
                                                 'Altitude  (ELA)')
        diag_ds['ela_m'].attrs['unit'] = 'm a.s.l'
        diag_ds['spec_mb'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['spec_mb'].attrs['description'] = 'Specific mass balance'
        diag_ds['spec_mb'].attrs['unit'] = 'mm w.e. yr-1'
        diag_ds['min_hgt'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['min_hgt'].attrs['description'] = 'Terminus elevation'
        diag_ds['min_hgt'].attrs['unit'] = 'm asl.'
        diag_ds['tau_l'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['tau_l'].attrs['description'] = 'Length change response time'
        diag_ds['tau_l'].attrs['unit'] = 'years'
        diag_ds['tau_a'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['tau_a'].attrs['description'] = 'Area change response time'
        diag_ds['tau_a'].attrs['unit'] = 'years'
        # TODO: handel tidewater glaciers

        # run the model
        for i, yr in enumerate(monthly_time):
            self.run_until(yr)
            # store diagnostics
            diag_ds['volume_m3'].data[i] = self.volume_m3
            diag_ds['area_m2'].data[i] = self.area_m2
            diag_ds['length_m'].data[i] = self.length_m
            diag_ds['spec_mb'].data[i] = self.spec_mb
            diag_ds['min_hgt'].data[i] = self.min_hgt
            diag_ds['tau_l'].data[i] = self.tau_l
            diag_ds['tau_a'].data[i] = self.tau_a

        if diag_path is not None:
            # write to file
            diag_ds.to_netcdf(diag_path)

        return diag_ds

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
        """ Try to run the glacier model until an equilibirum is reached.
        Works only with a constant mass balance model.

        Parameters
        ----------
        rate: float, optional
            rate of volume change for which the glacier is considered to be in
            equilibrium, whereby rate = |V0 - V1| / V0. default is 0.1 percent
        ystep: int, optional
            number of years per iteration step, default is 5
        max_ite: int, optional
            maximum number of iterations, default is 200

        """
        # TODO: isinstance is not working...
        # if not isinstance(self.mb_model, ConstantVASMassBalance):
        #     raise TypeError('The mass balance model must be of type ' +
        #                     'ConstantVASMassBalance.')
        # initialize the iteration counters and the volume change parameter
        ite = 0
        was_close_zero = 0
        t_rate = 1

        # model runs for a maximum fixed number of iterations
        # loop breaks if an equilibrium is reached (t_rate small enough)
        # or the glacier volume is below 1 for a defined number of times
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            # increment the iteration counter
            ite += 1
            #  store current volume ('before')
            v_bef = self.volume_m3
            # run for the given number of years
            self.run_until(self.year + ystep)
            # store new volume ('after')
            v_af = self.volume_m3
            #
            if np.isclose(v_bef, 0., atol=1):
                # avoid division by (values close to) zero
                t_rate = 1
                was_close_zero += 1
            else:
                # compute rate of volume change
                t_rate = np.abs(v_af - v_bef) / v_bef

        # raise RuntimeError if maximum number of iterations is reached
        if ite > max_ite:
            raise RuntimeError('Did not find equilibrium.')

    def create_start_glacier(self, area_m2_start, year_start,
                             adjust_term_elev=False):
        """Instance model with given starting glacier area, for the iterative
        process of seeking the glacier’s surface area at the beginning of the
        model integration.
        Per default, the terminus elevation is not scaled (i.e. is the same as
        for the initial glacier (probably RGI values)). This corresponds to
        the code of Marzeion et. al. (2012), but is physically not consistent.
        It is possible to scale the corresponding terminus elevation given the
        most recent (measured) outline. However, this is not recommended since
        the results may be strange. TODO: this should be fixed sometime...

        Parameters
        ----------
        area_m2_start : float
            starting surface area guess [m2]
        year_start : flaot
            corresponding starting year
        adjust_term_elev : bool, optional, default = False

        """
        # compute volume (m3) and length (m) from area (using scaling laws)
        volume_m3_start = self.ca * area_m2_start ** self.gamma
        length_m_start = (volume_m3_start / self.cl) ** (1 / self.ql)
        min_hgt_start = self.min_hgt_0
        # compute corresponding terminus elevation
        if adjust_term_elev:
            min_hgt_start = self.max_hgt + (length_m_start / self.length_m_0
                                            * (self.min_hgt_0 - self.max_hgt))

        self.__init__(year_start, area_m2_start, min_hgt_start,
                      self.max_hgt, self.mb_model)

    def run_and_compare(self, model_ref):
        """Let the model glacier evolve to the same year as the reference
        model (`model_ref`). Compute and return the relative error in area.

        Parameters
        ----------
        model_ref : :py:class:`oggm.vascaling.VAScalingModel`

        Returns
        -------
        float
            relative surface area error

        """
        # run model and store area
        year, _, area, _, _, _ = self.run_until(year_end=model_ref.year,
                                                reset=True)
        assert year == model_ref.year
        # compute relative difference to reference area
        rel_error = 1 - area/model_ref.area_m2

        return rel_error

    def start_area_minimization(self):
        """Find the start area which results in a best fitting area after
        model integration. TODO:

        """
        raise NotImplementedError
