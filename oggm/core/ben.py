"""Implementation of the 'original' model from Bens paper"""

# External libs
import numpy as np
import pandas as pd
import netCDF4
import os
import datetime


# import OGGM modules
import oggm
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH, BASENAMES

from oggm import utils
from oggm.utils import get_demo_file, tuple2int
from oggm.utils import (SuperclassMeta, lazy_property, floatyear_to_date,
                        date_to_floatyear, monthly_timeseries, ncDataset,
                        tolist)

from oggm import workflow

from oggm.core import (gis, inversion, climate, centerlines, flowline,
                       massbalance)

from oggm.core.massbalance import MassBalanceModel


def _compute_temp_terminus(temp, temp_grad, ref_hgt,
                           terminus_hgt, temp_anomaly=0):
    """ Computes the monthly mean temperature at the glacier terminus.

    :param temp: (netCDF4 variable) monthly mean climatological temperature
    :param temp_grad: (netCDF4 variable or float) temperature lapse rate
    :param ref_hgt: (float) reference elevation for climatological temperature
    :param terminus_hgt: (float) elevation of the glacier terminus
    :param temp_anomaly: (netCDF4 variable or float) monthly mean
        temperature anomaly, default 0

    :return: (netCDF4 variable) monthly mean temperature
        at the glacier terminus
    """
    temp_terminus = temp + temp_grad * (terminus_hgt - ref_hgt) + temp_anomaly
    return temp_terminus


def _compute_solid_prcp(prcp, prcp_factor, ref_hgt, min_hgt, max_hgt,
                        temp_terminus, temp_all_solid, temp_grad,
                        prcp_grad=0, prcp_anomaly=0):
    """ TODO: write description

    :param prcp: (netCDF4 variable) monthly mean climatological precipitation
    :param prcp_factor: (float) precipitation scaling factor
    :param ref_hgt: (float) reference elevation for
        climatological precipitation
    :param min_hgt: (float) minimum glacier elevation
    :param max_hgt: (float) maximum glacier elevation
    :param temp_terminus: (netCDF4 variable) monthly mean temperature
        at the glacier terminus
    :param temp_all_solid: (float) temperature threshold for
        solid precipitation
    :param temp_grad: (netCDF4 variable or float) temperature lapse rate
    :param prcp_grad: (netCDF4 variable or float) precipitation gradient
    :param prcp_anomaly: (netCDF4 variable or float) monthly mean
        precipitation anomaly, default 0

    :return: (netCDF4 variable) monthly mean solid precipitation
    """
    # compute fraction of solid precipitation
    if max_hgt == min_hgt:
        # prevent division by zero if max_hgt equals min_hgt
        f_solid = (temp_terminus <= temp_all_solid).astype(int)
    else:
        # use scaling defined in paper
        f_solid = (1
                   + (temp_terminus - temp_all_solid)
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
    """ Reads the DEM and computes the minimal and
        maximal glacier surface elevation in meters asl,
        from the given glacier outline.

    :param gdir: OGGM glacier directory
    :return: [float, float] minimal and maximal glacier surface elevation
    """
    # get relevant elevation information
    fpath = gdir.get_filepath('gridded_data')
    with ncDataset(fpath) as nc:
        mask = nc.variables['glacier_mask'][:]
        topo = nc.variables['topo'][:]
    min_elev = np.min(topo[np.where(mask == 1)])
    max_elev = np.max(topo[np.where(mask == 1)])
    return min_elev, max_elev


def get_yearly_mb_temp_prcp(gdir, time_range=None, year_range=None):
    """ @TODO:

    :param gdir:
    :param min_hgt:
    :param max_hgt:
    :param time_range:
    :param year_range:
    :return:
    """
    if year_range is not None:
        sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
        em = sm - 1 if (sm > 1) else 12
        t0 = datetime.datetime(year_range[0]-1, sm, 1)
        t1 = datetime.datetime(year_range[1], em, 1)
        return get_yearly_mb_temp_prcp(gdir, time_range=[t0, t1])

    # Parameters
    temp_all_solid = cfg.PARAMS['temp_all_solid']
    temp_melt = cfg.PARAMS['temp_melt']
    prcp_fac = cfg.PARAMS['prcp_scaling_factor']
    default_grad = cfg.PARAMS['temp_default_gradient']
    # prcp_grad = 3e-4
    prcp_grad = 0
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

    # The following is my code. So abandon all hope, you who enter here.

    # get relevant elevation information
    fpath = gdir.get_filepath('gridded_data')
    with ncDataset(fpath) as nc:
        mask = nc.variables['glacier_mask'][:]
        topo = nc.variables['topo'][:]
    min_elev = np.min(topo[np.where(mask == 1)])
    max_elev = np.max(topo[np.where(mask == 1)])

    # get temperature at glacier terminus
    temp_terminus = _compute_temp_terminus(itemp, igrad, ref_hgt, min_elev)
    # compute positive 'melting' temperature/energy input
    temp = np.clip(temp_terminus - temp_melt, a_min=0, a_max=None)
    # get solid precipitation
    prcp_solid = _compute_solid_prcp(iprcp, prcp_fac, ref_hgt,
                                     min_elev, max_elev,
                                     temp_terminus, temp_all_solid,
                                     igrad, prcp_grad)

    # Check if climate data includes all 12 month of all years
    ny, r = divmod(len(time), 12)
    if r != 0:
        raise ValueError('Climate data should be N full years exclusively')
    # Last year gives the tone of the hydro year
    years = np.arange(time[-1].year - ny + 1, time[-1].year + 1, 1)

    # compute sums over hydrological year
    temp_yr = np.zeros(len(years))
    prcp_yr = np.zeros(len(years))

    for i, y in enumerate(years):
        temp_yr[i] = np.sum(temp[i * 12:(i + 1) * 12])
        prcp_yr[i] = np.sum(prcp_solid[i * 12:(i + 1) * 12])

    return years, temp_yr, prcp_yr


def local_t_star(gdir, *, ref_df=None, tstar=None, bias=None):
    """Compute the local t* and associated glacier-wide mu*.

    If ``tstar`` and ``bias`` are not provided, they will be
    interpolated from the reference t* list.

    Note: the glacier wide mu* is here just for indication.
    It might be different from the flow lines' mu* in some cases.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    ref_df : pd.Dataframe, optional
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
        # Do our own interpolation of t_start for given glacier
        if ref_df is None:
            if not cfg.PARAMS['run_mb_calibration']:
                # Make some checks and use the default one
                climate_info = gdir.read_pickle('climate_info')
                source = climate_info['baseline_climate_source']
                ok_source = ['CRU TS4.01', 'CRU TS3.23', 'HISTALP']
                if not np.any(s in source.upper() for s in ok_source):
                    msg = ('If you are using a custom climate file you '
                           'should run your own MB calibration.')
                    raise climate.MassBalanceCalibrationError(msg)
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

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    out = gdir.read_pickle('climate_info')
    out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in params}
    gdir.write_pickle(out, 'climate_info')

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
    gdir.write_json(df, 'ben_params')


class BenMassBalance(MassBalanceModel):
    """Original mass balance model, used in Ben's paper."""

    def __init__(self, gdir, mu_star=None, bias=None,
                 filename='climate_monthly', input_filesuffix='',
                 repeat=False, ys=None, ye=None, check_calib_params=True):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
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

        super(BenMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        if mu_star is None:
            df = gdir.read_json('ben_params')
            mu_star = df['mu_star']

        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                df = gdir.read_json('ben_params')
                bias = df['bias']
            else:
                bias = 0.

        self.mu_star = mu_star
        self.bias = bias

        # Parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_liq = cfg.PARAMS['temp_all_liq']
        self.t_melt = cfg.PARAMS['temp_melt']
        prcp_fac = cfg.PARAMS['prcp_scaling_factor']
        default_grad = cfg.PARAMS['temp_default_gradient']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = gdir.read_pickle('climate_info')['mb_calib_params']
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    raise RuntimeError('You seem to use different mass-'
                                       'balance parameters than used for the '
                                       'calibration. '
                                       'Set `check_calib_params=False` '
                                       'to ignore this warning.')

        # Public attributes
        self.temp_bias = 0.
        self.prcp_bias = 1.
        self.repeat = repeat

        # Read file
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
        t_star = gdir.read_json('ben_params')['t_star']
        mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
        yr = [t_star - mu_hp, t_star + mu_hp]
        _, _, prcp_clim = get_yearly_mb_temp_prcp(gdir, year_range=yr)
        self.prcp_clim = prcp_clim

    def get_monthly_climate(self, min_hgt, max_hgt, year):
        """ Compute and return monthly positive terminus temperature
        and solid precipitation amount for given month.

        :param min_hgt: (float) glacier terminus elevation [m asl.]
        :param max_hgt: (float) maximal glacier surface elevation [m asl.]
        :param year: (float) floating year, following the
            hydrological year convention
        :return:
            temp_for_melt: (float) positive terminus temperature [°C]
            prcp_solid: (float) solid precipitation amount [kg/m^2]
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
        """ Compute and return the glacier wide mass balance
        for the given year/month combination.
        Possible mb bias is applied...

        :param min_hgt: (float) glacier terminus elevation
        :param max_hgt: (float) maximal glacier (surface) elevation
        :param year: (float) float year and month, using the
            hydrological year convention

        :return: average glacier wide mass balance [m/s]
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
        """ Compute and return monthly positive terminus temperature
        and solid precipitation amount for all months
        of the given (hydrological) year.

        :param min_hgt: (float) glacier terminus elevation [m asl.]
        :param max_hgt: (float) maximal glacier surface elevation [m asl.]
        :param year: (float) year, following the hydrological year convention
        :return:
            temp_for_melt: (float array, size 12)
                monthly positive terminus temperature [°C]
            prcp_solid: (float array, size 12)
                monthly solid precipitation amount [kg/m^2]
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
        """ Compute and return the annual glacier wide
        mass balance for the given year.
        Possible mb bias is applied...

        :param min_hgt: (float) glacier terminus elevation
        :param max_hgt: (float) maximal glacier (surface) elevation
        :param year: (float) float year, using the hydrological year convention

        :return: average glacier wide mass balance [m/s]
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
        """ Compute and return the annual specific mass balance
        for the given year.Possible mb bias is applied...

        :param min_hgt: (float) glacier terminus elevation
        :param max_hgt: (float) maximal glacier (surface) elevation
        :param year: (float) float year, using the hydrological year convention

        :return: glacier wide average mass balance, units of
            millimeter water equivalent per year [mm w.e. yr-1]
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
        """ Compute and return the monthly specific mass balance
        for the given month. Possible mb bias is applied...

        :param min_hgt: (float) glacier terminus elevation
        :param max_hgt: (float) maximal glacier (surface) elevation
        :param year: (float) float month, using the
            hydrological year convention

        :return: glacier wide average mass balance, units of
            millimeter water equivalent per months [mm w.e. yr-1]
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
        raise NotImplementedError('The equilibrium line altitude can not be ' +
                                  'computed for the `BenMassBalance` model.')


class BenModel(object):
    """ TODO: docstring """

    def __init__(self, area_0, min_hgt, max_hgt, mb_model):
        """ Instance new glacier model
            TODO: docstring
            TODO: finalise & test initialization method
        """
        # define geometrical/spatial parameters
        self.area = area_0
        self.min_hgt = min_hgt
        self.min_hgt_0 = min_hgt
        self.max_hgt_0 = max_hgt

        # compute volume and length from area (using scaling parameters)
        self.length = cfg.PARAMS['c_length'] * area_0**cfg.PARAMS['q_length']
        self.length_0 = self.length
        self.volume = (cfg.PARAMS['c_volume']
                       * area_0**cfg.PARAMS['gamma_volume'])

        # define mass balance model
        self.mb_model = mb_model

        # compute scaling parameters
        self._compute_scaling_params()

        pass

    def _compute_scaling_params(self):
        """ Compute the scaling parameters for glacier length `tau_l`
        and glacier surface area `tau_a` for current time step. """
        self.tau_l = self.volume / self.mb_model.prcp_clim
        self.tau_a = self.tau_l * self.area / self.length**2

    def step(self):
        """ TODO """
        pass

    def run_until(self):
        """ TODO """
        pass
