"""Mass balance models"""
# Built ins
import logging
# External libs
import cftime
import numpy as np
import pandas as pd
import netCDF4
from scipy.interpolate import interp1d
from scipy import optimize
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import (get_geodetic_mb_dataframe, floatyear_to_date,
                        date_to_floatyear, monthly_timeseries, ncDataset,
                        tolist, clip_min, clip_max, clip_array,
                        weighted_average_1d)
from oggm.exceptions import InvalidWorkflowError, InvalidParamsError
from oggm.core.massbalance import MassBalanceModel
from oggm import entity_task

# Module logger
log = logging.getLogger(__name__)


class MonthlyTIModel(MassBalanceModel):
    """Montly temperature index model.

    """
    def __init__(self, gdir,
                 melt_f=None,
                 temp_bias=None,
                 prcp_fac=None,
                 bias=0,
                 filename='climate_historical', input_filesuffix='',
                 repeat=False, ys=None, ye=None):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        melt_f : float, optional
            set to the alternative value of the melt factor you want to use
            (the default is to use the calibrated value).
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
        """

        super(MonthlyTIModel, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m

        if melt_f is None:
            df = gdir.read_json('mb_calib')
            melt_f = df['melt_f']

        if temp_bias is None:
            df = gdir.read_json('mb_calib')
            temp_bias = df['temp_bias']

        if prcp_fac is None:
            df = gdir.read_json('mb_calib')
            prcp_fac = df['prcp_fac']

        self.melt_f = melt_f
        self.bias = bias

        # Parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_liq = cfg.PARAMS['temp_all_liq']
        self.t_melt = cfg.PARAMS['temp_melt']

        # check if valid prcp_fac is used
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        default_grad = cfg.PARAMS['temp_default_gradient']

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat

        sm = cfg.PARAMS['hydro_month_' + self.hemisphere]
        if sm != 1:
            raise InvalidParamsError('Next generation mass balance models '
                                     'are only working on calendar years. '
                                     "Set `PARAMS['hydro_month_nh']` or `sh` "
                                     "to 1 to prevent this.")

        # Private attrs
        # to allow prcp_fac to be changed after instantiation
        # prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same for temp bias
        self._temp_bias = temp_bias

        # Read file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with ncDataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            try:
                time = netCDF4.num2date(time[:], time.units)
            except ValueError:
                # This is for longer time series
                time = cftime.num2date(time[:], time.units, calendar='noleap')
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError('Climate data should be N full years')
            # This is where we switch to hydro float year format
            # Last year gives the tone of the hydro year
            self.years = np.repeat(np.arange(time[-1].year - ny + 1,
                                             time[-1].year + 1), 12)

            pok = slice(None)  # take all is default (optim)
            if ys is not None:
                pok = self.years >= ys
            if ye is not None:
                try:
                    pok = pok & (self.years <= ye)
                except TypeError:
                    pok = self.years <= ye

            self.years = self.years[pok]
            self.months = np.tile(np.arange(1, 13), ny)[pok]
            # Read timeseries and correct it
            self.temp = nc.variables['temp'][pok].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables['prcp'][pok].astype(np.float64) * self._prcp_fac
            if 'gradient' in nc.variables:
                grad = nc.variables['gradient'][pok].astype(np.float64)
                # Security for stuff that can happen with local gradients
                g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
                grad = np.where(~np.isfinite(grad), default_grad, grad)
                grad = clip_array(grad, g_minmax[0], g_minmax[1])
            else:
                grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.ys = self.years[0]
            self.ye = self.years[-1]

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        """Precipitation factor (default: cfg.PARAMS['prcp_scaling_factor'])

        Called factor to make clear that it is a multiplicative factor in
        contrast to the additive temperature bias
        """
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        # just to check that no invalid prcp_factors are used
        if np.any(np.asarray(new_prcp_fac) <= 0):
            raise InvalidParamsError('prcp_fac has to be above zero!')

        if len(np.atleast_1d(new_prcp_fac)) == 12:
            # OK so that's monthly stuff
            new_prcp_fac = np.tile(new_prcp_fac, len(self.prcp) // 12)

        self.prcp *= new_prcp_fac / self._prcp_fac
        self._prcp_fac = new_prcp_fac

    @property
    def temp_bias(self):
        """Add a temperature bias to the time series"""
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):

        if len(np.atleast_1d(new_temp_bias)) == 12:
            # OK so that's monthly stuff
            new_temp_bias = np.tile(new_temp_bias, len(self.temp) // 12)

        self.temp += new_temp_bias - self._temp_bias
        self._temp_bias = new_temp_bias

    def get_monthly_climate(self, heights, year=None):
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """

        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if y < self.ys or y > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(y, self.ys, self.ye))
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        tempformelt = temp - self.t_melt
        clip_min(tempformelt, 0, out=tempformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.ones(npix) * iprcp
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * clip_array(fac, 0, 1)

        return temp, tempformelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if year < self.ys or year > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(year, self.ys, self.ye))
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        heights = np.asarray(heights)
        npix = len(heights)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(12).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = temp2d - self.t_melt
        clip_min(temp2dformelt, 0, out=temp2dformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * clip_array(fac, 0, 1)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None, add_climate=False, **kwargs):

        t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
        mb_month = prcpsol - self.melt_f * tmelt
        mb_month -= self.bias * SEC_IN_MONTH / SEC_IN_YEAR
        if add_climate:
            return (mb_month / SEC_IN_MONTH / self.rho, t, tmelt,
                    prcp, prcpsol)
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):

        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        mb_annual = np.sum(prcpsol - self.melt_f * tmelt, axis=1)
        mb_annual = (mb_annual - self.bias) / SEC_IN_YEAR / self.rho
        if add_climate:
            return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        return mb_annual


def _fallback_mb_calibration(gdir):
    """A Fallback function if climate.mu_star_calibration raises an Error.

    This function will still read, expand and write a `local_mustar.json`,
    filled with NANs, if climate.mu_star_calibration fails
    and if cfg.PARAMS['continue_on_error'] = True.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    """
    # read json
    try:
        df = gdir.read_json('mb_calib')
    except FileNotFoundError:
        df = dict()
        df['temp_bias'] = np.nan
        df['melt_f'] = np.nan
        df['prcp_fac'] = np.nan
    # write
    gdir.write_json(df, 'mb_calib')


def calving_mb(gdir):
    """Calving mass-loss in specific MB equivalent.

    This is necessary to calibrate the mass balance.
    """

    if not gdir.is_tidewater:
        return 0.

    # Ok. Just take the calving rate from cfg and change its units
    # Original units: km3 a-1, to change to mm a-1 (units of specific MB)
    rho = cfg.PARAMS['ice_density']
    return gdir.inversion_calving_rate * 1e9 * rho / gdir.rgi_area_m2


@entity_task(log, writes=['mb_calib'], fallback=_fallback_mb_calibration)
def mb_calibration_from_geodetic_mb(gdir,
                                    ref_mb=None,
                                    ref_period='',
                                    min_melt_f=None,
                                    max_melt_f=None):
    """Compute the flowlines' mu* from the reference geodetic MB data.

    This calibrates the mass balance parameters using the reference geodetic
    MB data instead, and this does NOT compute the apparent mass balance at
    the same time - users need to run apparent_mb_from_any_mb separately.

    Currently only works for single flowlines.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ref_mb : float
        the reference mass balance to match (units: kg m-2 yr-1)
        If None, use the default Hugonnet file.
    ref_period : str, default: PARAMS['geodetic_mb_period']
        one of '2000-01-01_2010-01-01', '2010-01-01_2020-01-01',
        '2000-01-01_2020-01-01'. If `ref_mb` is set, this should still match
        the same format but can be any date.
    min_melt_f: bool, optional
        defaults to cfg.PARAMS['min_mu_star']
    max_melt_f: bool, optional
        defaults to cfg.PARAMS['max_mu_star']
    """

    # mu* constraints
    if min_melt_f is None:
        min_melt_f = cfg.PARAMS['min_mu_star']
    if max_melt_f is None:
        max_melt_f = cfg.PARAMS['max_mu_star']

    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    if sm == 1:
        # Check that the other hemisphere is set to 1 as well to avoid surprises
        oh = 'sh' if gdir.hemisphere == 'nh' else 'nh'
        if cfg.PARAMS['hydro_month_' + oh] != 1:
            raise InvalidParamsError('Please set both hydro_month_nh and '
                                     'hydro_month_sh to 1 for geodetic '
                                     'calibration.')
    if sm != 1:
        raise InvalidParamsError('mb_calibration_from_geodetic_mb makes '
                                 'more sense when applied on calendar years '
                                 "(PARAMS['hydro_month_nh']=1 and "
                                 "`PARAMS['hydro_month_sh']=1). If you want "
                                 "to ignore this error.")

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')

    # If someone called another task before we need to reset this
    for fl in fls:
        fl.mu_star_is_valid = False

    # Let's go
    # Climate period
    if not ref_period:
        ref_period = cfg.PARAMS['geodetic_mb_period']
    y0, y1 = ref_period.split('_')
    y0 = int(y0.split('-')[0])
    y1 = int(y1.split('-')[0])
    yr_range = [y0, y1 - 1]

    # Get the reference data
    if ref_mb is None:
        ref_mb = get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        ref_mb = float(ref_mb.loc[ref_mb['period'] == ref_period]['dmdtda'])
        # dmdtda: in meters water-equivalent per year -> we convert
        ref_mb *= 1000  # kg m-2 yr-1

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)
    if cmb != 0:
        raise NotImplementedError('Calving with geodetic MB is not implemented '
                                  'yet, but it should actually work. Well keep '
                                  'you posted!')

    # Create the MB model we will calibrate upon
    mb_mod = MonthlyTIModel(gdir,
                            melt_f=100,
                            temp_bias=0,
                            prcp_fac=2.7)

    def to_minimize(x, ref_mb, model_attr):
        # Set the new attr value
        setattr(mb_mod, model_attr, x)
        out = mb_mod.get_specific_mb(fls=fls, year=np.arange(*yr_range)).mean()
        return np.mean(out - ref_mb)

    try:
        param = optimize.brentq(to_minimize,
                                -10, 10,
                                args=(ref_mb, 'temp_bias'),
                                xtol=2e-12)
    except ValueError:
        raise
    t = 1
    #     # This happens when out of bounds
    #
    #     # Funny enough, this bias correction is arbitrary.
    #     # Here I'm trying something arbitrary as well.
    #     # Let's try to find a range of corrections that would lead to an
    #     # allowed mu* and pick one
    #
    #     # Here we ignore the previous QC correction - if any -
    #     # to ensure that results are the same even after previous correction
    #     fpath = gdir.get_filepath('climate_historical')
    #     with utils.ncDataset(fpath, 'a') as nc:
    #         start = getattr(nc, 'uncorrected_ref_hgt', nc.ref_hgt)
    #         nc.uncorrected_ref_hgt = start
    #         nc.ref_hgt = start
    #
    #     # Read timeseries again after reset
    #     _, temp, prcp = mb_yearly_climate_on_height(gdir, heights,
    #                                                 year_range=yr_range,
    #                                                 flatten=False)
    #
    #     # Check in which direction we should correct the temp
    #     _lim0 = _mu_star_per_minimization(min_mu_star, fls, ref_mb, temp,
    #                                       prcp, widths)
    #     if _lim0 < 0:
    #         # The mass balances are too positive to be matched - we need to
    #         # cool down the climate data
    #         step = -step_height_for_corr
    #         end = -max_height_change_for_corr
    #     else:
    #         # The other way around
    #         step = step_height_for_corr
    #         end = max_height_change_for_corr
    #
    #     steps = np.arange(start, start + end, step, dtype=np.int64)
    #     mu_candidates = steps * np.NaN
    #     for i, h in enumerate(steps):
    #         with utils.ncDataset(fpath, 'a') as nc:
    #             nc.ref_hgt = h
    #
    #         # Read timeseries
    #         _, temp, prcp = mb_yearly_climate_on_height(gdir, heights,
    #                                                     year_range=yr_range,
    #                                                     flatten=False)
    #
    #         try:
    #             mu_star = optimize.brentq(_mu_star_per_minimization,
    #                                       min_mu_star, max_mu_star,
    #                                       args=(fls, ref_mb, temp, prcp, widths),
    #                                       xtol=_brentq_xtol)
    #         except ValueError:
    #             mu_star = np.NaN
    #
    #         # Done - store for later
    #         mu_candidates[i] = mu_star
    #         # if we find one working mu_star we can actually stop
    #         # the loop to make it faster.
    #         # We are here only interested in the candidate which
    #         # changes the ref_hgt the least!
    #         if np.isfinite(mu_star):
    #             break
    #
    #     # the workflow below works in general when having more candidates
    #     # but also works for one candidate (as we stopped the loop)
    #     sel_steps = steps[np.isfinite(mu_candidates)]
    #     sel_mus = mu_candidates[np.isfinite(mu_candidates)]
    #     if len(sel_mus) == 0:
    #         # Yeah nothing we can do here
    #         raise MassBalanceCalibrationError('We could not find a way to '
    #                                           'correct the climate data and '
    #                                           'fit within the prescribed '
    #                                           'bounds for mu*.')
    #
    #     # We have just picked the first, but to be fair it is arbitrary
    #     # We could also pick one randomly... but here we rather prefer to have
    #     # the smallest ref_hgt change as possible (hence smallest temp. bias change)
    #     mu_star = sel_mus[0]
    #     # Final correction of the data
    #     with utils.ncDataset(fpath, 'a') as nc:
    #         nc.ref_hgt = sel_steps[0]
    #     gdir.add_to_diagnostics('ref_hgt_calib_diff', float(sel_steps[0] - start))
    #
    # if not np.isfinite(mu_star):
    #     raise MassBalanceCalibrationError('{} '.format(gdir.rgi_id) +
    #                                       'has a non finite mu.')
    #
    # # Add the climate related params to the GlacierDir to make sure
    # # other tools cannot fool around without re-calibration
    # out = gdir.get_climate_info()
    # out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in MB_PARAMS}
    # gdir.write_json(out, 'climate_info')
    #
    # # Store diagnostics
    # df = gdir.read_json('local_mustar', allow_empty=True)
    # df['rgi_id'] = gdir.rgi_id
    # df['t_star'] = np.nan
    # df['bias'] = 0
    # df['mu_star_per_flowline'] = [mu_star] * len(fls)
    # df['mu_star_glacierwide'] = mu_star
    # df['mu_star_flowline_avg'] = mu_star
    # df['mu_star_allsame'] = True
    # # Write
    # gdir.write_json(df, 'local_mustar')