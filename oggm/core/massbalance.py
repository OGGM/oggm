"""Mass balance models - next generation"""

# Built ins
import logging
import os
# External libs
import cftime
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d
from scipy import optimize
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import (SuperclassMeta, get_geodetic_mb_dataframe,
                        floatyear_to_date, date_to_floatyear, get_demo_file,
                        monthly_timeseries, ncDataset, get_temp_bias_dataframe,
                        clip_min, clip_max, clip_array, clip_scalar,
                        weighted_average_1d, lazy_property, set_array_type, rmsd)
from oggm.exceptions import (InvalidWorkflowError, InvalidParamsError,
                             MassBalanceCalibrationError)
from oggm import entity_task

# Module logger
log = logging.getLogger(__name__)

# Climate relevant global params - not optimised
MB_GLOBAL_PARAMS = ['temp_default_gradient',
                    'temp_all_solid',
                    'temp_all_liq',
                    'temp_melt']


class MassBalanceModel(object, metaclass=SuperclassMeta):
    """Interface and common logic for all mass balance models used in OGGM.

    All mass balance models should implement this interface.

    Attributes
    ----------
    valid_bounds : [float, float]
        The altitudinal bounds where the MassBalanceModel is valid. This is
        necessary for automated ELA search.
    hemisphere : str, {'nh', 'sh'}
        Used for certain methods - if the hydrological year is requested.
    rho : float, default: ``cfg.PARAMS['ice_density']``
        Density of ice
    """

    def __init__(self):
        """ Initialize."""
        self.valid_bounds = None
        self.hemisphere = None
        self.rho = cfg.PARAMS['ice_density']

    def __repr__(self):
        """String Representation of the mass balance model"""
        summary = ['<oggm.MassBalanceModel>']
        summary += ['  Class: ' + self.__class__.__name__]
        summary += ['  Attributes:']
        # Add all scalar attributes
        for k, v in self.__dict__.items():
            if np.isscalar(v) and not k.startswith('_'):
                nbform = '    - {}: {}'
                summary += [nbform.format(k, v)]
        return '\n'.join(summary) + '\n'

    def reset_state(self):
        """Reset any internal state of the model.
        This might not be needed by most models, but some models have an
        internal state (e.g. a snow cover history) which can be reset
        this way.
        """
        pass

    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None):
        """Monthly mass balance at given altitude(s) for a moment in time.

        Units: [m s-1], or meters of ice per second

        Note: `year` is optional because some simpler models have no time
        component.

        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass balance will be computed
        year: float, optional
            the time (in the "floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        fls: list of flowline instances, optional
            the flowlines array, in case the MB model implementation needs
            to know details about the glacier geometry at the moment the
            MB model is called

        Returns
        -------
        the mass balance (same dim as `heights`) (units: [m s-1])
        """
        raise NotImplementedError()

    def get_annual_mb(self, heights, year=None, fl_id=None, fls=None):
        """Like `self.get_monthly_mb()`, but for annual MB.

        For some simpler mass balance models ``get_monthly_mb()` and
        `get_annual_mb()`` can be equivalent.

        Units: [m s-1], or meters of ice per second

        Note: `year` is optional because some simpler models have no time
        component.

        Parameters
        ----------
        heights: ndarray
            the altitudes at which the mass balance will be computed
        year: float, optional
            the time (in the "floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        fls: list of flowline instances, optional
            the flowlines array, in case the MB model implementation needs
            to know details about the glacier geometry at the moment the
            MB model is called

        Returns
        -------
        the mass balance (same dim as `heights`) (units: [m s-1])
        """
        raise NotImplementedError()

    def get_annual_specific_mass_balance(self, fls: list, year: float) -> float:
        """Get annual specific mass balance from multiple flowlines.

        Parameters
        ----------
        fls : list[oggm.Flowline]
            Flowline instances.
        year: float
            Time in "floating year" convention.

        Returns
        -------
        float
            Annual specific mass balance from multiple flowlines.
        """
        mbs = []
        widths = []
        for i, fl in enumerate(fls):
            _widths = fl.widths
            try:
                # For rect and parabola don't compute spec mb
                _widths = np.where(fl.thick > 0, _widths, 0)
            except AttributeError:
                pass
            widths.append(_widths)
            mbs.append(
                self.get_annual_mb(fl.surface_h, fls=fls, fl_id=i, year=year)
            )
        widths = np.concatenate(widths, axis=0)  # 2x faster than np.append
        mbs = np.concatenate(mbs, axis=0)
        mbs = weighted_average_1d(mbs, widths)

        return mbs

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None):
        """Specific mass balance for a given glacier geometry.

        Units: [mm w.e. yr-1], or millimeter water equivalent per year.

        Parameters
        ----------
        heights : ArrayLike, default None
            Altitudes at which the mass balance will be computed.
            Overridden by ``fls`` if provided.
        widths : ArrayLike, default None
            Widths of the flowline (necessary for the weighted average).
            Overridden by ``fls`` if provided.
        fls : list[oggm.Flowline], default None
            List of flowline instances. Alternative to heights and
            widths, and overrides them if provided.
        year : ArrayLike[float] or float, default None
            Year, or a range of years in "floating year" convention.

        Returns
        -------
        np.ndarray
            Specific mass balance (units: mm w.e. yr-1).
        """
        stack = []
        year = np.atleast_1d(year)
        for mb_yr in year:
            if fls is not None:
                mbs = self.get_annual_specific_mass_balance(fls=fls, year=mb_yr)
            else:
                mbs = self.get_annual_mb(heights, year=mb_yr)
                mbs = weighted_average_1d(mbs, widths)
            stack.append(mbs)

        return set_array_type(stack) * SEC_IN_YEAR * self.rho

    def get_ela(self, year=None, **kwargs):
        """Get the equilibrium line altitude for a given year.

        Parameters
        ----------
        year : ArrayLike[float] or float, default None
            Year, or a range of years in "floating year" convention.
        **kwargs
            Any other keyword argument accepted by ``self.get_annual_mb``.

        Returns
        -------
        float or np.ndarray:
            The equilibrium line altitude (ELA) in m.
        """
        stack = []
        year = np.atleast_1d(year)
        for mb_year in year:
            if self.valid_bounds is None:
                raise ValueError('attribute `valid_bounds` needs to be '
                                'set for the ELA computation.')

            # Check for invalid ELAs
            b0, b1 = self.valid_bounds
            if (np.any(~np.isfinite(
                    self.get_annual_mb([b0, b1], year=mb_year, **kwargs))) or
                    (self.get_annual_mb([b0], year=mb_year, **kwargs)[0] > 0) or
                    (self.get_annual_mb([b1], year=mb_year, **kwargs)[0] < 0)):
                stack.append(np.nan)
            else:
                def to_minimize(x):
                    return (self.get_annual_mb([x], year=mb_year, **kwargs)[0] *
                    SEC_IN_YEAR * self.rho)
                stack.append(optimize.brentq(to_minimize, *self.valid_bounds, xtol=0.1))

        return set_array_type(stack)

    def is_year_valid(self, year):
        """Checks if a given date year be simulated by this model.

        Parameters
        ----------
        year : float, optional
            the time (in the "floating year" convention)

        Returns
        -------
        True if this year can be simulated by the model
        """
        raise NotImplementedError()


class ScalarMassBalance(MassBalanceModel):
    """Constant mass balance, everywhere."""

    def __init__(self, mb=0.):
        """ Initialize.

        Parameters
        ----------
        mb : float
            Fix the mass balance to a certain value (unit: [mm w.e. yr-1])
        """
        super(ScalarMassBalance, self).__init__()
        self.hemisphere = 'nh'
        self.valid_bounds = [-2e4, 2e4]  # in m
        self._mb = mb

    def get_monthly_mb(self, heights, **kwargs):
        mb = np.asarray(heights) * 0 + self._mb
        return mb / SEC_IN_YEAR / self.rho

    def get_annual_mb(self, heights, **kwargs):
        mb = np.asarray(heights) * 0 + self._mb
        return mb / SEC_IN_YEAR / self.rho

    def is_year_valid(self, year):
        return True


class LinearMassBalance(MassBalanceModel):
    """Constant mass balance as a linear function of altitude.

    Attributes
    ----------
    ela_h: float
        the equilibrium line altitude (units: [m])
    grad: float
        the mass balance gradient (unit: [mm w.e. yr-1 m-1])
    max_mb: float
        Cap the mass balance to a certain value (unit: [mm w.e. yr-1])
    temp_bias
    """

    def __init__(self, ela_h, grad=3., max_mb=None):
        """ Initialize.

        Parameters
        ----------
        ela_h: float
            Equilibrium line altitude (units: [m])
        grad: float
            Mass balance gradient (unit: [mm w.e. yr-1 m-1])
        max_mb: float
            Cap the mass balance to a certain value (unit: [mm w.e. yr-1])
        """
        super(LinearMassBalance, self).__init__()
        self.hemisphere = 'nh'
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.orig_ela_h = ela_h
        self.ela_h = ela_h
        self.grad = grad
        self.max_mb = max_mb
        self._temp_bias = 0

    @property
    def temp_bias(self):
        """Change the ELA following a simple rule: + 1K -> ELA + 150 m

        A "temperature bias" doesn't makes much sense in the linear MB
        context, but we implemented a simple empirical rule:
        + 1K -> ELA + 150 m
        """
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        self.ela_h = self.orig_ela_h + value * 150
        self._temp_bias = value

    def get_monthly_mb(self, heights, **kwargs):
        mb = (np.asarray(heights) - self.ela_h) * self.grad
        if self.max_mb is not None:
            clip_max(mb, self.max_mb, out=mb)
        return mb / SEC_IN_YEAR / self.rho

    def get_annual_mb(self, heights, **kwargs):
        return self.get_monthly_mb(heights, **kwargs)

    def is_year_valid(self, year):
        return True


class MonthlyTIModel(MassBalanceModel):
    """Monthly temperature index model.
    """
    def __init__(self, gdir,
                 filename='climate_historical',
                 input_filesuffix='',
                 mb_params_filesuffix='',
                 fl_id=None,
                 melt_f=None,
                 temp_bias=None,
                 prcp_fac=None,
                 bias=0,
                 ys=None,
                 ye=None,
                 repeat=False,
                 check_calib_params=True,
                 ):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data. Default is 'climate_historical'
        input_filesuffix : str, optional
            append a suffix to the climate input filename (useful for GCM runs).
        mb_params_filesuffix : str, optional
            append a suffix to the mb params calibration file (useful for
            sensitivity runs).
        fl_id : int, optional
            if this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            set to the value of the melt factor you want to use,
            here the unit is kg m-2 day-1 K-1
            (the default is to use the calibrated value).
        temp_bias : float, optional
            set to the value of the temperature bias you want to use
            (the default is to use the calibrated value).
        prcp_fac : float, optional
            set to the value of the precipitation factor you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration and
            the ones you are using at run time. If they don't match, it will
            raise an error. Set to "False" to suppress this check.
        """

        super(MonthlyTIModel, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.fl_id = fl_id  # which flowline are we the model of?
        self._mb_params_filesuffix = mb_params_filesuffix  # which mb params?
        self.gdir = gdir

        if melt_f is None:
            melt_f = self.calib_params['melt_f']

        if temp_bias is None:
            temp_bias = self.calib_params['temp_bias']

        if prcp_fac is None:
            prcp_fac = self.calib_params['prcp_fac']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = self.calib_params['mb_global_params']
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    msg = ('You seem to use different mass balance parameters '
                           'than used for the calibration: '
                           f"you use cfg.PARAMS['{k}']={cfg.PARAMS[k]} while "
                           f"it was calibrated with cfg.PARAMS['{k}']={v}. "
                           'Set `check_calib_params=False` to ignore this '
                           'warning.')
                    raise InvalidWorkflowError(msg)
            src = self.calib_params['baseline_climate_source']
            src_calib = gdir.get_climate_info()['baseline_climate_source']
            if src != src_calib:
                msg = (f'You seem to have calibrated with the {src} '
                       f"climate data while this gdir was calibrated with "
                       f"{src_calib}. Set `check_calib_params=False` to "
                       f"ignore this warning.")
                raise InvalidWorkflowError(msg)

        self.melt_f = melt_f
        self.bias = bias

        # Global parameters
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

        # Private attrs
        # to allow prcp_fac to be changed after instantiation
        # prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same for temp bias
        self._temp_bias = temp_bias

        # Read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with ncDataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = cftime.num2date(time[:], time.units, calendar=time.calendar)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError('Climate data should be N full years')

            # We check for calendar years
            if (time[0].month != 1) or (time[-1].month != 12):
                raise InvalidWorkflowError('We now work exclusively with '
                                           'calendar years.')

            # Quick trick because we know the size of our array
            years = np.repeat(np.arange(time[-1].year - ny + 1,
                                        time[-1].year + 1), 12)
            pok = slice(None)  # take all is default (optim)
            if ys is not None:
                pok = years >= ys
            if ye is not None:
                try:
                    pok = pok & (years <= ye)
                except TypeError:
                    pok = years <= ye

            self.years = years[pok]
            self.months = np.tile(np.arange(1, 13), ny)[pok]

            # Read timeseries and correct it
            self.temp = nc.variables['temp'][pok].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables['prcp'][pok].astype(np.float64) * self._prcp_fac

            grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.climate_source = nc.climate_source
            self.ys = self.years[0]
            self.ye = self.years[-1]

    def __repr__(self):
        """String Representation of the mass balance model"""
        summary = ['<oggm.MassBalanceModel>']
        summary += ['  Class: ' + self.__class__.__name__]
        summary += ['  Attributes:']
        # Add all scalar attributes
        done = []
        for k in ['hemisphere', 'climate_source', 'melt_f', 'prcp_fac', 'temp_bias', 'bias']:
            done.append(k)
            v = self.__getattribute__(k)
            if k == 'climate_source':
                if v.endswith('.nc'):
                    v = os.path.basename(v)
            nofloat = ['hemisphere', 'climate_source']
            nbform = '    - {}: {}' if k in nofloat else '    - {}: {:.02f}'
            summary += [nbform.format(k, v)]
        for k, v in self.__dict__.items():
            if np.isscalar(v) and not k.startswith('_') and k not in done:
                nbform = '    - {}: {}'
                summary += [nbform.format(k, v)]
        return '\n'.join(summary) + '\n'

    @property
    def monthly_melt_f(self):
        return self.melt_f * 365 / 12

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        """Precipitation factor (default: cfg.PARAMS['prcp_fac'])

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

    @lazy_property
    def calib_params(self):
        if self.fl_id is None:
            return self.gdir.read_json('mb_calib', self._mb_params_filesuffix)
        else:
            try:
                out = self.gdir.read_json('mb_calib', filesuffix=f'_fl{self.fl_id}')
                if self._mb_params_filesuffix:
                    raise InvalidWorkflowError('mb_params_filesuffix cannot be '
                                               'used with multiple flowlines')
                return out
            except FileNotFoundError:
                return self.gdir.read_json('mb_calib', self._mb_params_filesuffix)

    def is_year_valid(self, year):
        return self.ys <= year <= self.ye

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
        if not self.is_year_valid(y):
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
        if not self.is_year_valid(year):
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
        mb_month = prcpsol - self.monthly_melt_f * tmelt
        mb_month -= self.bias * SEC_IN_MONTH / SEC_IN_YEAR
        if add_climate:
            return (mb_month / SEC_IN_MONTH / self.rho, t, tmelt,
                    prcp, prcpsol)
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):

        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        mb_annual = np.sum(prcpsol - self.monthly_melt_f * tmelt, axis=1)
        mb_annual = (mb_annual - self.bias) / SEC_IN_YEAR / self.rho
        if add_climate:
            return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        return mb_annual


class ConstantMassBalance(MassBalanceModel):
    """Constant mass balance during a chosen period.

    This is useful for equilibrium experiments.

    IMPORTANT: the "naive" implementation requires to compute the massbalance
    N times for each simulation year, where N is the number of years over the
    climate period to average. This is very expensive, and therefore we use
    interpolation. This makes it *unusable* with MB models relying on the
    computational domain being always the same.

    If your model requires constant domain size, conisder using RandomMassBalance
    instead.

    Note that it uses the "correct" way to represent the average mass balance
    over a given period. See: https://oggm.org/2021/08/05/mean-forcing/

    Attributes
    ----------
    y0 : int
        the center year of the period
    halfsize : int
        the halfsize of the period
    years : ndarray
        the years of the period
    """

    def __init__(self, gdir, mb_model_class=MonthlyTIModel,
                 y0=None, halfsize=15,
                 **kwargs):
        """Initialize

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mb_model_class : MassBalanceModel class
            the MassBalanceModel to use for the constant climate
        y0 : int, required
            the year at the center of the period of interest.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        **kwargs:
            keyword arguments to pass to the mb_model_class
        """

        super().__init__()
        self.mbmod = mb_model_class(gdir,
                                    **kwargs)

        if y0 is None:
            raise InvalidParamsError('Please set `y0` explicitly')

        # This is a quick'n dirty optimisation
        try:
            fls = gdir.read_pickle('model_flowlines')
            h = []
            for fl in fls:
                # We use bed because of overdeepenings
                h = np.append(h, fl.bed_h)
                h = np.append(h, fl.surface_h)
            zminmax = np.round([np.min(h)-50, np.max(h)+2000])
        except FileNotFoundError:
            # in case we don't have them
            with ncDataset(gdir.get_filepath('gridded_data')) as nc:
                if np.isfinite(nc.min_h_dem):
                    # a bug sometimes led to non-finite
                    zminmax = [nc.min_h_dem-250, nc.max_h_dem+1500]
                else:
                    zminmax = [nc.min_h_glacier-1250, nc.max_h_glacier+1500]
        self.hbins = np.arange(*zminmax, step=10)
        self.valid_bounds = self.hbins[[0, -1]]
        self.y0 = y0
        self.halfsize = halfsize
        self.years = np.arange(y0-halfsize, y0+halfsize+1)
        self.hemisphere = gdir.hemisphere

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        self.mbmod.bias = value

    @lazy_property
    def interp_yr(self):
        # annual MB
        mb_on_h = self.hbins * 0.
        for yr in self.years:
            mb_on_h += self.mbmod.get_annual_mb(self.hbins, year=yr)
        return interp1d(self.hbins, mb_on_h / len(self.years))

    @lazy_property
    def interp_m(self):
        # monthly MB
        months = np.arange(12)+1
        interp_m = []
        for m in months:
            mb_on_h = self.hbins*0.
            for yr in self.years:
                yr = date_to_floatyear(yr, m)
                mb_on_h += self.mbmod.get_monthly_mb(self.hbins, year=yr)
            interp_m.append(interp1d(self.hbins, mb_on_h / len(self.years)))
        return interp_m

    def is_year_valid(self, year):
        return True

    def get_monthly_climate(self, heights, year=None):
        """Average climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other biases (precipitation, temp) are applied

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        _, m = floatyear_to_date(year)
        yrs = [date_to_floatyear(y, m) for y in self.years]
        heights = np.atleast_1d(heights)
        nh = len(heights)
        shape = (len(yrs), nh)
        temp = np.zeros(shape)
        tempformelt = np.zeros(shape)
        prcp = np.zeros(shape)
        prcpsol = np.zeros(shape)
        for i, yr in enumerate(yrs):
            t, tm, p, ps = self.mbmod.get_monthly_climate(heights, year=yr)
            temp[i, :] = t
            tempformelt[i, :] = tm
            prcp[i, :] = p
            prcpsol[i, :] = ps
        return (np.mean(temp, axis=0),
                np.mean(tempformelt, axis=0),
                np.mean(prcp, axis=0),
                np.mean(prcpsol, axis=0))

    def get_annual_climate(self, heights, year=None):
        """Average climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other biases (precipitation, temp) are applied

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        yrs = monthly_timeseries(self.years[0], self.years[-1],
                                 include_last_year=True)
        heights = np.atleast_1d(heights)
        nh = len(heights)
        shape = (len(yrs), nh)
        temp = np.zeros(shape)
        tempformelt = np.zeros(shape)
        prcp = np.zeros(shape)
        prcpsol = np.zeros(shape)
        for i, yr in enumerate(yrs):
            t, tm, p, ps = self.mbmod.get_monthly_climate(heights, year=yr)
            temp[i, :] = t
            tempformelt[i, :] = tm
            prcp[i, :] = p
            prcpsol[i, :] = ps
        # Note that we do not weight for number of days per month:
        # this is consistent with OGGM's calendar
        return (np.mean(temp, axis=0),
                np.mean(tempformelt, axis=0) * 12,
                np.mean(prcp, axis=0) * 12,
                np.mean(prcpsol, axis=0) * 12)

    def get_monthly_mb(self, heights, year=None, add_climate=False, **kwargs):
        yr, m = floatyear_to_date(year)
        if add_climate:
            t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
            return self.interp_m[m-1](heights), t, tmelt, prcp, prcpsol
        return self.interp_m[m-1](heights)

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        mb = self.interp_yr(heights)
        if add_climate:
            t, tmelt, prcp, prcpsol = self.get_annual_climate(heights)
            return mb, t, tmelt, prcp, prcpsol
        return mb


class RandomMassBalance(MassBalanceModel):
    """Random shuffle of all MB years within a given time period.

    This is useful for finding a possible past glacier state or for sensitivity
    experiments.

    Note that this is going to be sensitive to extreme years in certain
    periods, but it is by far more physically reasonable than other
    approaches based on gaussian assumptions.
    """

    def __init__(self, gdir, mb_model_class=MonthlyTIModel,
                 y0=None, halfsize=15, seed=None,
                 all_years=False, unique_samples=False,
                 prescribe_years=None,
                 **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mb_model_class : MassBalanceModel class
            the MassBalanceModel to use for the random shuffle
        y0 : int, required
            the year at the center of the period of interest.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        seed : int, optional
            Random seed used to initialize the pseudo-random number generator.
        all_years : bool
            if True, overwrites ``y0`` and ``halfsize`` to use all available
            years.
        unique_samples: bool
            if true, chosen random mass balance years will only be available
            once per random climate period-length
            if false, every model year will be chosen from the random climate
            period with the same probability
        prescribe_years : pandas Series
            instead of random samples, take a series of (i, y) pairs where
            (i) is the simulation year index and (y) is the year to pick in the
            original timeseries. Overrides `y0`, `halfsize`, `all_years`,
            `unique_samples` and `seed`.
        **kwargs:
            keyword arguments to pass to the mb_model_class
        """

        super().__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.mbmod = mb_model_class(gdir, **kwargs)

        # Climate period
        self.prescribe_years = prescribe_years

        if self.prescribe_years is None:
            # Normal stuff
            self.rng = np.random.RandomState(seed)
            if all_years:
                self.years = self.mbmod.years
            else:
                if y0 is None:
                    raise InvalidParamsError('Please set `y0` explicitly')
                self.years = np.arange(y0 - halfsize, y0 + halfsize + 1)
        else:
            self.rng = None
            self.years = self.prescribe_years.index

        self.yr_range = (self.years[0], self.years[-1] + 1)
        self.ny = len(self.years)
        self.hemisphere = gdir.hemisphere

        self._state_yr = dict()

        # Sampling without replacement
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
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        self.mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.bias = value

    def is_year_valid(self, year):
        return True

    def get_state_yr(self, year=None):
        """For a given year, get the random year associated to it."""
        year = int(year)
        if year not in self._state_yr:
            if self.prescribe_years is not None:
                self._state_yr[year] = self.prescribe_years.loc[year]
            else:
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

    def get_monthly_mb(self, heights, year=None, **kwargs):
        ryr, m = floatyear_to_date(year)
        ryr = date_to_floatyear(self.get_state_yr(ryr), m)
        return self.mbmod.get_monthly_mb(heights, year=ryr, **kwargs)

    def get_annual_mb(self, heights, year=None, **kwargs):
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_annual_mb(heights, year=ryr, **kwargs)


class UncertainMassBalance(MassBalanceModel):
    """Adding uncertainty to a mass balance model.

    There are three variables for which you can add uncertainty:
    - temperature (additive bias)
    - precipitation (multiplicative factor)
    - residual (a bias in units of MB)
    """

    def __init__(self, basis_model,
                 rdn_temp_bias_seed=None, rdn_temp_bias_sigma=0.1,
                 rdn_prcp_fac_seed=None, rdn_prcp_fac_sigma=0.1,
                 rdn_bias_seed=None, rdn_bias_sigma=100):
        """Initialize.

        Parameters
        ----------
        basis_model : MassBalanceModel
            the model to which you want to add the uncertainty to
        rdn_temp_bias_seed : int
            the seed of the random number generator
        rdn_temp_bias_sigma : float
            the standard deviation of the random temperature error
        rdn_prcp_fac_seed : int
            the seed of the random number generator
        rdn_prcp_fac_sigma : float
            the standard deviation of the random precipitation error
            (to be consistent this should be renamed prcp_fac as well)
        rdn_bias_seed : int
            the seed of the random number generator
        rdn_bias_sigma : float
            the standard deviation of the random MB error
        """
        super(UncertainMassBalance, self).__init__()
        # the aim here is to change temp_bias and prcp_fac so
        self.mbmod = basis_model
        self.hemisphere = basis_model.hemisphere
        self.valid_bounds = self.mbmod.valid_bounds
        self.is_year_valid = self.mbmod.is_year_valid
        self.rng_temp = np.random.RandomState(rdn_temp_bias_seed)
        self.rng_prcp = np.random.RandomState(rdn_prcp_fac_seed)
        self.rng_bias = np.random.RandomState(rdn_bias_seed)
        self._temp_sigma = rdn_temp_bias_sigma
        self._prcp_sigma = rdn_prcp_fac_sigma
        self._bias_sigma = rdn_bias_sigma
        self._state_temp = dict()
        self._state_prcp = dict()
        self._state_bias = dict()

    def is_year_valid(self, year):
        return self.mbmod.is_year_valid(year)

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        self.mbmod.prcp_fac = value

    def _get_state_temp(self, year):
        year = int(year)
        if year not in self._state_temp:
            self._state_temp[year] = self.rng_temp.randn() * self._temp_sigma
        return self._state_temp[year]

    def _get_state_prcp(self, year):
        year = int(year)
        if year not in self._state_prcp:
            self._state_prcp[year] = self.rng_prcp.randn() * self._prcp_sigma
        return self._state_prcp[year]

    def _get_state_bias(self, year):
        year = int(year)
        if year not in self._state_bias:
            self._state_bias[year] = self.rng_bias.randn() * self._bias_sigma
        return self._state_bias[year]

    def get_monthly_mb(self, heights, year=None, **kwargs):
        raise NotImplementedError()

    def get_annual_mb(self, heights, year=None, fl_id=None, **kwargs):

        # Keep the original biases and add a random error
        _t = self.mbmod.temp_bias
        _p = self.mbmod.prcp_fac
        _b = self.mbmod.bias
        self.mbmod.temp_bias = self._get_state_temp(year) + _t
        self.mbmod.prcp_fac = self._get_state_prcp(year) + _p
        self.mbmod.bias = self._get_state_bias(year) + _b
        try:
            out = self.mbmod.get_annual_mb(heights, year=year, fl_id=fl_id)
        except BaseException:
            self.mbmod.temp_bias = _t
            self.mbmod.prcp_fac = _p
            self.mbmod.bias = _b
            raise
        # Back to normal
        self.mbmod.temp_bias = _t
        self.mbmod.prcp_fac = _p
        self.mbmod.bias = _b
        return out


class MultipleFlowlineMassBalance(MassBalanceModel):
    """Handle mass balance at the glacier level instead of flowline level.

    Convenience class doing not much more than wrapping a list of mass balance
    models, one for each flowline.

    This is useful for real-case studies, where each flowline might have
    different model parameters.

    Attributes
    ----------
    fls : list
        list of flowline objects

    """

    def __init__(self, gdir, fls=None, mb_model_class=MonthlyTIModel,
                 use_inversion_flowlines=False,
                 input_filesuffix='',
                 **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        fls : list, optional
            list of flowline objects to use (defaults to 'model_flowlines')
        mb_model_class : MassBalanceModel class
            the MassBalanceModel to use (default is MonthlyTIModel,
            alternatives are e.g. ConstantMassBalance...)
        use_inversion_flowlines: bool, optional
            use 'inversion_flowlines' instead of 'model_flowlines'
        kwargs : kwargs to pass to mb_model_class
        """

        # Read in the flowlines
        if use_inversion_flowlines:
            fls = gdir.read_pickle('inversion_flowlines')

        if fls is None:
            try:
                fls = gdir.read_pickle('model_flowlines')
            except FileNotFoundError:
                raise InvalidWorkflowError('Need a valid `model_flowlines` '
                                           'file. If you explicitly want to '
                                           'use `inversion_flowlines`, set '
                                           'use_inversion_flowlines=True.')

        self.fls = fls

        # Initialise the mb models
        self.flowline_mb_models = []
        for fl in self.fls:
            # Merged glaciers will need different climate files, use filesuffix
            if (fl.rgi_id is not None) and (fl.rgi_id != gdir.rgi_id):
                rgi_filesuffix = '_' + fl.rgi_id + input_filesuffix
            else:
                rgi_filesuffix = input_filesuffix

            self.flowline_mb_models.append(
                mb_model_class(gdir, input_filesuffix=rgi_filesuffix,
                               **kwargs))

        self.valid_bounds = self.flowline_mb_models[-1].valid_bounds
        self.hemisphere = gdir.hemisphere

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.flowline_mb_models[0].temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.flowline_mb_models[0].prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.flowline_mb_models[0].bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.bias = value

    def is_year_valid(self, year):
        return self.flowline_mb_models[0].is_year_valid(year)

    def get_monthly_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_monthly_mb(heights,
                                                             year=year,
                                                             **kwargs)

    def get_annual_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_annual_mb(heights,
                                                            year=year,
                                                            **kwargs)

    def get_annual_mb_on_flowlines(self, fls=None, year=None):
        """Get the MB on all points of the glacier at once.

        Parameters
        ----------
        fls: list, optional
            the list of flowlines to get the mass balance from. Defaults
            to self.fls
        year: float, optional
            the time (in the "floating year" convention)
        Returns
        -------
        Tuple of (heights, widths, mass_balance) 1D arrays
        """

        if fls is None:
            fls = self.fls

        heights = []
        widths = []
        mbs = []
        for i, fl in enumerate(fls):
            h = fl.surface_h
            heights = np.append(heights, h)
            widths = np.append(widths, fl.widths)
            mbs = np.append(mbs, self.get_annual_mb(h, year=year, fl_id=i))

        return heights, widths, mbs

    def get_annual_specific_mass_balance(self, fls: list, year: float) -> float:
        """Get annual specific mass balance from multiple flowlines.

        Parameters
        ----------
        fls : list[oggm.Flowline]
            Flowline instances.
        year : float
            Time in "floating year" convention.

        Returns
        -------
        float
            Annual specific mass balance from multiple flowlines.
        """
        mbs = []
        widths = []
        for i, (fl, mb_mod) in enumerate(zip(fls, self.flowline_mb_models)):
            _widths = fl.widths
            try:
                # For rect and parabola don't compute spec mb
                _widths = np.where(fl.thick > 0, _widths, 0)
            except AttributeError:
                pass
            widths.append(_widths)
            mb = mb_mod.get_annual_mb(fl.surface_h, year=year, fls=fls, fl_id=i)
            mbs.append(mb * SEC_IN_YEAR * mb_mod.rho)
        widths = np.concatenate(widths, axis=0)  # 2x faster than np.append
        mbs = np.concatenate(mbs, axis=0)
        mbs = weighted_average_1d(mbs, widths)

        return mbs

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None):
        """Specific mass balance for a given glacier geometry.

        Units: [mm w.e. yr-1], or millimeter water equivalent per year.

        Parameters
        ----------
        heights : ArrayLike, default None
            Altitudes at which the mass balance will be computed.
            Overridden by ``fls`` if provided.
        widths : ArrayLike, default None
            Widths of the flowline (necessary for the weighted average).
            Overridden by ``fls`` if provided.
        fls : list[oggm.Flowline], default None
            List of flowline instances. Alternative to heights and
            widths, and overrides them if provided.
        year : ArrayLike[float] or float, default None
            Year, or a range of years in "floating year" convention.

        Returns
        -------
        np.ndarray
            Specific mass balance (units: mm w.e. yr-1).
        """
        if heights is not None or widths is not None:
            raise ValueError(
                "`heights` and `widths` kwargs do not work with "
                "MultipleFlowlineMassBalance!"
            )

        if fls is None:
            fls = self.fls

        stack = []
        year = np.atleast_1d(year)
        for mb_yr in year:
            mbs = self.get_annual_specific_mass_balance(fls=fls, year=mb_yr)
            stack.append(mbs)

        return set_array_type(stack)

    def get_ela(self, year=None, **kwargs):
        """Get the equilibrium line altitude for a given year.

        The ELA here is not without ambiguity: it computes a mean
        weighted by area.

        Parameters
        ----------
        year : ArrayLike[float] or float, default None
            Year, or a range of years in "floating year" convention.

        Returns
        -------
        float or np.ndarray
            The equilibrium line altitude (ELA) in m.
        """
        stack = []
        year = np.atleast_1d(year)
        for mb_yr in year:
            elas = []
            areas = []
            for fl_id, (fl, mb_mod) in enumerate(
                zip(self.fls, self.flowline_mb_models)
            ):
                elas.append(
                    mb_mod.get_ela(year=mb_yr, fl_id=fl_id, fls=self.fls)
                )
                areas.append(np.sum(fl.widths))
            stack.append(weighted_average_1d(elas, areas))

        return set_array_type(stack)


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


def decide_winter_precip_factor(gdir):
    """Utility function to decide on a precip factor based on winter precip.

    The values here are hardcoded as OGGM evolves - there should be an
    easy way to change it if people need more flexibility one day.
    """

    # get non-corrected winter daily mean prcp (kg m-2 day-1)
    # it is easier to get this directly from the raw climate files
    fp = gdir.get_filepath('climate_historical')
    with xr.open_dataset(fp).prcp as ds_pr:
        # just select winter months
        if gdir.hemisphere == 'nh':
            m_winter = [10, 11, 12, 1, 2, 3, 4]
        else:
            m_winter = [4, 5, 6, 7, 8, 9, 10]

        ds_pr_winter = ds_pr.where(ds_pr['time.month'].isin(m_winter), drop=True)

        # select the correct 41 year time period
        ds_pr_winter = ds_pr_winter.sel(time=slice('1979-01-01', '2019-12-01'))

        # check if we have the full time period: 41 years * 7 months
        text = ('the climate period has to go from 1979-01 to 2019-12,',
                'use W5E5 or GSWP3_W5E5 as baseline climate and',
                'repeat the climate processing')
        assert len(ds_pr_winter.time) == 41 * 7, text
        w_prcp = float((ds_pr_winter / ds_pr_winter.time.dt.daysinmonth).mean())

    climsource = gdir.get_climate_info()['baseline_climate_source']
    if 'w5e5' in climsource.lower():
        # from OGGM calibration to winter MB, LOG
        # repeated by Lily in November 2025 with newest gdirs
        # using t_melt=-1, cte lapse rate, monthly resolution
        a, b = -1.0614, 3.9200
        prcp_fac = a * np.log(w_prcp) + b
    elif 'era5' in climsource.lower():
        # from OGGM calibration to winter MB, LINEAR
        # repeated by Lily in November 2025 with newest gdirs
        # using t_melt=-1, cte lapse rate, monthly resolution
        a, b = -0.09078476, 2.43505368
        prcp_fac = a * w_prcp + b
    else:
        msg = (f'Baseline climate {climsource} not suitable for'
               'decide_winter_precip_factor(). Set prcp_fac.')
        raise InvalidWorkflowError(msg)

    # don't allow extremely low/high prcp. factors!!!
    return clip_scalar(prcp_fac,
                       cfg.PARAMS['prcp_fac_min'],
                       cfg.PARAMS['prcp_fac_max'])


@entity_task(log, writes=['mb_calib'])
def mb_calibration_from_wgms_mb(gdir, **kwargs):
    """Calibrate for in-situ, annual MB.

    This only works for glaciers which have WGMS data!

    For now this just calls mb_calibration_from_scalar_mb internally,
    but could be cleverer than that if someone wishes to implement it.

    Parameters
    ----------
    **kwargs : any kwarg accepted by mb_calibration_from_scalar_mb
    except `ref_mb` and `ref_mb_years`
    """

    # Note that this currently does not work for hydro years (WGMS uses hydro)
    # A way to go would be to teach the mb models to use calendar years
    # internally but still output annual MB in hydro convention.
    mbdf = gdir.get_ref_mb_data()
    # Keep only valid values
    mbdf = mbdf.loc[~mbdf['ANNUAL_BALANCE'].isnull()]
    return mb_calibration_from_scalar_mb(gdir,
                                         ref_mb=mbdf['ANNUAL_BALANCE'].mean(),
                                         ref_mb_years=mbdf.index.values,
                                         **kwargs)


@entity_task(log, writes=['mb_calib'])
def mb_calibration_to_rmsd(gdir, *,
                           ref_df=None,
                           write_to_gdir=True,
                           overwrite_gdir=False,
                           use_2d_mb=False,
                           calibrate_params=('melt_f',),
                           melt_f=None,
                           melt_f_min=None,
                           melt_f_max=None,
                           prcp_fac=None,
                           prcp_fac_min=None,
                           prcp_fac_max=None,
                           temp_bias=None,
                           temp_bias_min=None,
                           temp_bias_max=None,
                           mb_model_class=MonthlyTIModel,
                           filesuffix='',):
    """Determine the MB parameters by minimising RMSD to a reference timeseries

    This calibrates the mass balance parameters using interannual
    MB data from the WGMS data over a given period. This calibration uses
    differential evolution to calibrate all given parameters to minimize
    the RMSD as much as possible.

    This function is useful to calibrate all three parameters at once,
    on glaciers where WGMS or other in-situ observations are available.
    This is achieved by minimising the RMSD between the reference MB
    timeseries and the modelled MB timeseries over the period of available
    observations. The minimisiation technique chosen here is differential
    evolution, which is a global optimization technique that does not
    require the function to be differentiable. This makes it
    suitable for our problem, where the relationship between the parameters
    and the MB timeseries can be complex and non-linear, and we are able
    to calibrate all three parameters at once.

    Note that this does not compute the apparent mass balance at
    the same time - users need to run `apparent_mb_from_any_mb after`
    calibration.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to calibrate
    ref_df : pandas dataframe, required
        the dataframe of annual mass balance values from the wgms data
        (units: kg m-2 yr-1).
        It is required here - if you want to use available observations,
    write_to_gdir : bool
        whether to write the results of the calibration to the glacier
        directory. If True (the default), this will be saved as `mb_calib.json`
        and be used by the MassBalanceModel class as parameters in subsequent
        tasks.
    overwrite_gdir : bool
        if a `mb_calib.json` exists, this task won't overwrite it per default.
        Set this to True to enforce overwriting (i.e. with consequences for the
        future workflow).
    use_2d_mb : bool
        Set to True if the mass balance calibration has to be done of the 2D mask
        of the glacier (for fully distributed runs only).
    mb_model_class : MassBalanceModel class
        the MassBalanceModel to use for the calibration. Needs to use the
        same parameters as MonthlyTIModel (the default): melt_f,
        temp_bias, prcp_fac.
    calibrate_params : tuple
        the parameter(s) that will be used in the calibration, it must be at least one of:
            'melt_f', 'temp_bias', 'prcp_fac'. Defaults to ('melt_f',)
    melt_f: float
        the default value to use as melt factor (or the starting value when
        optimizing MB). Defaults to cfg.PARAMS['melt_f'].
    melt_f_min: float
        the minimum accepted value for the melt factor during optimisation.
        Defaults to cfg.PARAMS['melt_f_min'].
    melt_f_max: float
        the maximum accepted value for the melt factor during optimisation.
        Defaults to cfg.PARAMS['melt_f_max'].
    prcp_fac: float
        the default value to use as precipitation scaling factor
        (or the starting value when optimizing MB). Defaults to the method
        chosen in `params.cfg` (winter prcp or global factor).
    prcp_fac_min: float
        the minimum accepted value for the precipitation scaling factor during
        optimisation. Defaults to cfg.PARAMS['prcp_fac_min'].
    prcp_fac_max: float
        the maximum accepted value for the precipitation scaling factor during
        optimisation. Defaults to cfg.PARAMS['prcp_fac_max'].
    temp_bias: float
        the default value to use as temperature bias (or the starting value when
        optimizing MB). Defaults to 0.
    temp_bias_min: float
        the minimum accepted value for the temperature bias during optimisation.
        Defaults to cfg.PARAMS['temp_bias_min'].
    temp_bias_max: float
        the maximum accepted value for the temperature bias during optimisation.
        Defaults to cfg.PARAMS['temp_bias_max'].
    filesuffix: str
        add a filesuffix to mb_calib.json. This could be useful for sensitivity
        analyses with MB models, if they need to fetch other sets of params for
        example.
    """

    # Param constraints
    if melt_f_min is None:
        melt_f_min = cfg.PARAMS['melt_f_min']
    if melt_f_max is None:
        melt_f_max = cfg.PARAMS['melt_f_max']
    if prcp_fac_min is None:
        prcp_fac_min = cfg.PARAMS['prcp_fac_min']
    if prcp_fac_max is None:
        prcp_fac_max = cfg.PARAMS['prcp_fac_max']
    if temp_bias_min is None:
        temp_bias_min = cfg.PARAMS['temp_bias_min']
    if temp_bias_max is None:
        temp_bias_max = cfg.PARAMS['temp_bias_max']

    if not use_2d_mb:
        fls = gdir.read_pickle('inversion_flowlines')
    else:
        # if the 2D data is used, the flowline is not needed.
        fls = None
        # get the 2D data
        fp = gdir.get_filepath('gridded_data')
        with xr.open_dataset(fp) as ds:
            # 'topo' instead of 'topo_smoothed'?
            heights = ds.topo_smoothed.data[ds.glacier_mask.data == 1]
            widths = np.ones(len(heights))

    # Climate period
    ref_mb_years = ref_df.index.values
    years = ref_mb_years

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)
    if cmb != 0:
        raise NotImplementedError('Calving with geodetic MB is not implemented '
                                  'yet, but it should actually work. Well keep '
                                  'you posted!')

    # Ok, regardless on how we want to calibrate, we start with defaults
    if melt_f is None:
        melt_f = cfg.PARAMS['melt_f']

    if prcp_fac is None:
        if cfg.PARAMS['prcp_fac'] is None:
            prcp_fac = decide_winter_precip_factor(gdir)
        else:
            prcp_fac = cfg.PARAMS['prcp_fac']

    if temp_bias is None:
        temp_bias = 0

    # Create the MB model we will calibrate
    mb_mod = mb_model_class(gdir,
                            melt_f=melt_f,
                            temp_bias=temp_bias,
                            prcp_fac=prcp_fac,
                            check_calib_params=False)

    # Check that the years are available
    for y in years:
        if not mb_mod.is_year_valid(y):
            raise ValueError(f'year {y} out of the valid time bounds: '
                             f'[{mb_mod.ys}, {mb_mod.ye}]')

    # Check that the calibrate params are valid
    for param in calibrate_params:
        if param not in ('melt_f', 'prcp_fac', 'temp_bias'):
            raise InvalidParamsError("calibrate_params must be a tuple with any of "
                                     "'melt_f', 'prcp_fac', 'temp_bias'")

    # Set the bounds for the optimization
    bounds = []
    for param in calibrate_params:
        if param == 'prcp_fac':
            bounds.append((prcp_fac_min, prcp_fac_max))
        elif param == 'melt_f':
            bounds.append((melt_f_min, melt_f_max))
        elif param == 'temp_bias':
            bounds.append((temp_bias_min, temp_bias_max))

    # Optimises all three mass balance parameters at the same time to minimize the RMSD between the simulated and reference MB timeseries
    def rmsd_cost_function(x, *model_attrs: tuple):
        for i, model_attr in enumerate(model_attrs):
            setattr(mb_mod, model_attr, x[i])

        if use_2d_mb:
            sim_out = mb_mod.get_specific_mb(heights=heights, widths=widths, year=years)
        else:
            sim_out = mb_mod.get_specific_mb(fls=fls, year=years)

        return rmsd(ref_df, sim_out)

    try:
        res = optimize.differential_evolution(rmsd_cost_function,
                                              bounds=bounds,
                                              tol=1e-8,
                                              maxiter=5000,
                                              args=(calibrate_params),
                                              )

        calib_params = res.x

        # Assign parameters
        for i, param in enumerate(calibrate_params):
            if param == 'prcp_fac':
                prcp_fac = calib_params[i]
            elif param == 'melt_f':
                melt_f = calib_params[i]
            elif param == 'temp_bias':
                temp_bias = calib_params[i]

    except ValueError:
        raise RuntimeError(f'{gdir.rgi_id}: could not minimise the rmsd. '
                           f'Try another technique.')

    # Store parameters
    df = gdir.read_json('mb_calib', allow_empty=True)
    df['rgi_id'] = gdir.rgi_id
    df['bias'] = 0
    df['melt_f'] = melt_f
    df['prcp_fac'] = prcp_fac
    df['temp_bias'] = temp_bias
    # What did we try to match?
    df['reference_mb'] = ref_df.values.mean()
    df['reference_period'] = str(ref_mb_years)
    df['rmsd'] = res.fun

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    df['mb_global_params'] = {k: cfg.PARAMS[k] for k in MB_GLOBAL_PARAMS}
    df['baseline_climate_source'] = gdir.get_climate_info()['baseline_climate_source']
    # Write
    if write_to_gdir:
        if gdir.has_file('mb_calib', filesuffix=filesuffix) and not overwrite_gdir:
            raise InvalidWorkflowError('`mb_calib.json` already exists for '
                                       'this repository. Set `overwrite_gdir` '
                                       'to True if you want to overwrite '
                                       'a previous calibration.')
        gdir.write_json(df, 'mb_calib', filesuffix=filesuffix)
    return df


@entity_task(log, writes=['mb_calib'])
def mb_calibration_from_geodetic_mb(gdir, *,
                                    ref_period=None,
                                    write_to_gdir=True,
                                    overwrite_gdir=False,
                                    use_regional_avg=False,
                                    override_missing=None,
                                    use_2d_mb=False,
                                    informed_threestep=False,
                                    calibrate_param1='melt_f',
                                    calibrate_param2=None,
                                    calibrate_param3=None,
                                    mb_model_class=MonthlyTIModel,
                                    filesuffix='',
                                    ):
    """Calibrate for geodetic MB data from Hugonnet et al., 2021.

    The data table can be obtained with utils.get_geodetic_mb_dataframe().
    It is equivalent to the original data from Hugonnet, but has some outlier
    values filtered. See this notebook* for more details.

    https://nbviewer.org/urls/cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/convert_vold1.ipynb

    This glacier-specific calibration can be replaced by a region-wide calibration
    by using regional averages (same units: mm w.e.) instead of the glacier
    specific averages.

    The problem of calibrating many unknown parameters on geodetic data is
    currently unsolved. This is OGGM's current take, based on trial and
    error and based on ideas from the literature.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to calibrate
    ref_period : str, default: PARAMS['geodetic_mb_period']
        one of '2000-01-01_2010-01-01', '2010-01-01_2020-01-01',
        '2000-01-01_2020-01-01'. If `ref_mb` is set, this should still match
        the same format but can be any date.
    write_to_gdir : bool
        whether to write the results of the calibration to the glacier
        directory. If True (the default), this will be saved as `mb_calib.json`
        and be used by the MassBalanceModel class as parameters in subsequent
        tasks.
    overwrite_gdir : bool
        if a `mb_calib.json` exists, this task won't overwrite it per default.
        Set this to True to enforce overwriting (i.e. with consequences for the
        future workflow).
    use_regional_avg : bool
        use the regional average instead of the glacier specific one.
    override_missing : scalar
        if the reference geodetic data is not available, use this value instead
        (mostly for testing with exotic datasets, but could be used to open
        the door to using other datasets).
    use_2d_mb : bool
        Set to True if the mass balance calibration has to be done of the 2D mask
        of the glacier (for fully distributed runs only).
    informed_threestep : bool
        the magic method Fabi found out one day before release.
        Overrides the calibrate_param order below.
    calibrate_param1 : str
        in the three-step calibration, the name of the first parameter
        to calibrate (one of 'melt_f', 'temp_bias', 'prcp_fac').
    calibrate_param2 : str
        in the three-step calibration, the name of the second parameter
        to calibrate (one of 'melt_f', 'temp_bias', 'prcp_fac'). If not
        set and the algorithm cannot match observations, it will raise an
        error.
    calibrate_param3 : str
        in the three-step calibration, the name of the third parameter
        to calibrate (one of 'melt_f', 'temp_bias', 'prcp_fac'). If not
        set and the algorithm cannot match observations, it will raise an
        error.
    mb_model_class : MassBalanceModel class
        the MassBalanceModel to use for the calibration. Needs to use the
        same parameters as MonthlyTIModel (the default): melt_f,
        temp_bias, prcp_fac.
    filesuffix: str
        add a filesuffix to mb_calib.json. This could be useful for sensitivity
        analyses with MB models, if they need to fetch other sets of params for
        example.

    Returns
    -------
    the calibrated parameters as dict
    """
    if not ref_period:
        ref_period = cfg.PARAMS['geodetic_mb_period']

    # Get the reference data
    ref_mb_err = np.nan
    if use_regional_avg:
        ref_mb_df_o = get_geodetic_mb_dataframe(regional=True)
        ref_mb_df = ref_mb_df_o.loc[ref_mb_df_o.period == ref_period].set_index('reg')
        if len(ref_mb_df) == 0:
            raise InvalidParamsError(f'Ref period {ref_period} not found in file: '
                                     f'{ref_mb_df_o.period.unique()}')
        # dmdtda: in meters water-equivalent per year -> we convert to kg m-2 yr-1
        ref_mb = ref_mb_df.loc[int(gdir.rgi_region), 'dmdtda'] * 1000
        ref_mb_err = ref_mb_df.loc[int(gdir.rgi_region), 'err_dmdtda'] * 1000
    else:
        try:
            ref_mb_df = get_geodetic_mb_dataframe().loc[gdir.rgi_id]
            ref_mb_df = ref_mb_df.loc[ref_mb_df['period'] == ref_period]
            # dmdtda: in meters water-equivalent per year -> we convert to kg m-2 yr-1
            ref_mb = ref_mb_df['dmdtda'].iloc[0] * 1000
            ref_mb_err = ref_mb_df['err_dmdtda'].iloc[0] * 1000
        except KeyError:
            if override_missing is None:
                raise
            ref_mb = override_missing

    temp_bias = 0
    if informed_threestep:
        climinfo = gdir.get_climate_info()
        climsource = climinfo['baseline_climate_source']
        if 'w5e5' in climsource.lower():
            bias_df = get_temp_bias_dataframe('w5e5',
                                              rgi_version=gdir.rgi_version,
                                              regional=use_regional_avg)
        elif 'era5' in climsource.lower():
            bias_df = get_temp_bias_dataframe('era5',
                                              rgi_version=gdir.rgi_version,
                                              regional=use_regional_avg)
        else:
            raise InvalidWorkflowError('Dataset not suitable for '
                                       f'informed 3-steps: {climsource}')
        ref_lon = climinfo['baseline_climate_ref_pix_lon']
        ref_lat = climinfo['baseline_climate_ref_pix_lat']
        # Take nearest
        dis = ((bias_df.lon_val - ref_lon)**2 + (bias_df.lat_val - ref_lat)**2)**0.5
        assert dis.min() < 1, 'Somethings wrong with lons'
        sel_df = bias_df.iloc[np.argmin(dis)]
        # Which bias central value to use?
        if use_regional_avg:
            centralval = 'median_temp_bias_w_area_grouped'
        else:
            centralval = 'median_temp_bias_w_err_grouped'
        temp_bias = sel_df[centralval]
        assert np.isfinite(temp_bias), 'Temp bias not finite?'

        if cfg.PARAMS['prcp_fac'] is not None:
            raise InvalidParamsError('With `informed_threestep` you cannot use '
                                     'a preset prcp_fac - we need to rely on '
                                     'decide_winter_precip_factor().')

        # Some magic heuristics - we just decide to calibrate
        # precip -> melt_f -> temp but informed by previous data.

        # Temp bias was decided anyway, we keep as previous value and
        # allow it to vary as last resort

        # We use the precip factor but allow it to vary between 0.8, 1.2 of
        # the previous value (uncertainty).
        prcp_fac = decide_winter_precip_factor(gdir)
        mi, ma = cfg.PARAMS['prcp_fac_min'], cfg.PARAMS['prcp_fac_max']
        prcp_fac_min = clip_scalar(prcp_fac * 0.8, mi, ma)
        prcp_fac_max = clip_scalar(prcp_fac * 1.2, mi, ma)

        return mb_calibration_from_scalar_mb(gdir,
                                             ref_mb=ref_mb,
                                             ref_mb_err=ref_mb_err,
                                             ref_period=ref_period,
                                             write_to_gdir=write_to_gdir,
                                             overwrite_gdir=overwrite_gdir,
                                             use_2d_mb=use_2d_mb,
                                             calibrate_param1='prcp_fac',
                                             calibrate_param2='melt_f',
                                             calibrate_param3='temp_bias',
                                             prcp_fac=prcp_fac,
                                             prcp_fac_min=prcp_fac_min,
                                             prcp_fac_max=prcp_fac_max,
                                             temp_bias=temp_bias,
                                             mb_model_class=mb_model_class,
                                             filesuffix=filesuffix,
                                             )

    else:
        return mb_calibration_from_scalar_mb(gdir,
                                             ref_mb=ref_mb,
                                             ref_mb_err=ref_mb_err,
                                             ref_period=ref_period,
                                             write_to_gdir=write_to_gdir,
                                             overwrite_gdir=overwrite_gdir,
                                             use_2d_mb=use_2d_mb,
                                             calibrate_param1=calibrate_param1,
                                             calibrate_param2=calibrate_param2,
                                             calibrate_param3=calibrate_param3,
                                             temp_bias=temp_bias,
                                             mb_model_class=mb_model_class,
                                             filesuffix=filesuffix,
                                             )


@entity_task(log, writes=['mb_calib'])
def mb_calibration_from_scalar_mb(gdir, *,
                                  ref_mb=None,
                                  ref_mb_err=None,
                                  ref_period=None,
                                  ref_mb_years=None,
                                  write_to_gdir=True,
                                  overwrite_gdir=False,
                                  use_2d_mb=False,
                                  calibrate_param1='melt_f',
                                  calibrate_param2=None,
                                  calibrate_param3=None,
                                  melt_f=None,
                                  melt_f_min=None,
                                  melt_f_max=None,
                                  prcp_fac=None,
                                  prcp_fac_min=None,
                                  prcp_fac_max=None,
                                  temp_bias=None,
                                  temp_bias_min=None,
                                  temp_bias_max=None,
                                  mb_model_class=MonthlyTIModel,
                                  filesuffix='',
                                  ):
    """Determine the mass balance parameters from a scalar mass-balance value.

    This calibrates the mass balance parameters using a reference average
    MB data over a given period (e.g. average in-situ SMB or geodetic MB).
    This flexible calibration allows to calibrate three parameters one after
    another. The first parameter is varied between two chosen values (a range)
    until the ref MB value is matched. If this fails, the second parameter
    can be changed, etc.

    This can be used for example to apply the "three-step calibration"
    introduced by Huss & Hock 2015, but you can choose any order of
    calibration.

    This task can be called by other, "higher level" tasks, for example
    :py:func:`oggm.core.massbalance.mb_calibration_from_geodetic_mb` or
    :py:func:`oggm.core.massbalance.mb_calibration_from_wgms_mb`.

    Note that this does not compute the apparent mass balance at
    the same time - users need to run `apparent_mb_from_any_mb after`
    calibration.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to calibrate
    ref_mb : float, required
        the reference mass balance to match (units: kg m-2 yr-1)
        It is required here - if you want to use available observations,
        use :py:func:`oggm.core.massbalance.mb_calibration_from_geodetic_mb`
        or :py:func:`oggm.core.massbalance.mb_calibration_from_wgms_mb`.
    ref_mb_err : float, optional
        currently only used for logging - it is not used in the calibration.
    ref_period : str, optional
        date format - for example '2000-01-01_2010-01-01'. If this is not
        set, ref_mb_years needs to be set.
    ref_mb_years : tuple of length 2 (range) or list of years.
        convenience kwarg to override ref_period. If a tuple of length 2 is
        given, all years between this range (excluding the last one) are used.
        If a list  of years is given, all these will be used (useful for
        data with gaps)
    write_to_gdir : bool
        whether to write the results of the calibration to the glacier
        directory. If True (the default), this will be saved as `mb_calib.json`
        and be used by the MassBalanceModel class as parameters in subsequent
        tasks.
    overwrite_gdir : bool
        if a `mb_calib.json` exists, this task won't overwrite it per default.
        Set this to True to enforce overwriting (i.e. with consequences for the
        future workflow).
    use_2d_mb : bool
        Set to True if the mass balance calibration has to be done of the 2D mask
        of the glacier (for fully distributed runs only).
    mb_model_class : MassBalanceModel class
        the MassBalanceModel to use for the calibration. Needs to use the
        same parameters as MonthlyTIModel (the default): melt_f,
        temp_bias, prcp_fac.
    calibrate_param1 : str
        in the three-step calibration, the name of the first parameter
        to calibrate (one of 'melt_f', 'temp_bias', 'prcp_fac').
    calibrate_param2 : str
        in the three-step calibration, the name of the second parameter
        to calibrate (one of 'melt_f', 'temp_bias', 'prcp_fac'). If not
        set and the algorithm cannot match observations, it will raise an
        error.
    calibrate_param3 : str
        in the three-step calibration, the name of the third parameter
        to calibrate (one of 'melt_f', 'temp_bias', 'prcp_fac'). If not
        set and the algorithm cannot match observations, it will raise an
        error.
    melt_f: float
        the default value to use as melt factor (or the starting value when
        optimizing MB). Defaults to cfg.PARAMS['melt_f'].
    melt_f_min: float
        the minimum accepted value for the melt factor during optimisation.
        Defaults to cfg.PARAMS['melt_f_min'].
    melt_f_max: float
        the maximum accepted value for the melt factor during optimisation.
        Defaults to cfg.PARAMS['melt_f_max'].
    prcp_fac: float
        the default value to use as precipitation scaling factor
        (or the starting value when optimizing MB). Defaults to the method
        chosen in `params.cfg` (winter prcp or global factor).
    prcp_fac_min: float
        the minimum accepted value for the precipitation scaling factor during
        optimisation. Defaults to cfg.PARAMS['prcp_fac_min'].
    prcp_fac_max: float
        the maximum accepted value for the precipitation scaling factor during
        optimisation. Defaults to cfg.PARAMS['prcp_fac_max'].
    temp_bias: float
        the default value to use as temperature bias (or the starting value when
        optimizing MB). Defaults to 0.
    temp_bias_min: float
        the minimum accepted value for the temperature bias during optimisation.
        Defaults to cfg.PARAMS['temp_bias_min'].
    temp_bias_max: float
        the maximum accepted value for the temperature bias during optimisation.
        Defaults to cfg.PARAMS['temp_bias_max'].
    filesuffix: str
        add a filesuffix to mb_calib.json. This could be useful for sensitivity
        analyses with MB models, if they need to fetch other sets of params for
        example.
    """

    # Param constraints
    if melt_f_min is None:
        melt_f_min = cfg.PARAMS['melt_f_min']
    if melt_f_max is None:
        melt_f_max = cfg.PARAMS['melt_f_max']
    if prcp_fac_min is None:
        prcp_fac_min = cfg.PARAMS['prcp_fac_min']
    if prcp_fac_max is None:
        prcp_fac_max = cfg.PARAMS['prcp_fac_max']
    if temp_bias_min is None:
        temp_bias_min = cfg.PARAMS['temp_bias_min']
    if temp_bias_max is None:
        temp_bias_max = cfg.PARAMS['temp_bias_max']
    if ref_mb_years is not None and ref_period is not None:
        raise InvalidParamsError('Cannot set `ref_mb_years` and `ref_period` '
                                 'at the same time.')

    if not use_2d_mb:
        fls = gdir.read_pickle('inversion_flowlines')
    else:
        # if the 2D data is used, the flowline is not needed.
        fls = None
        # get the 2D data
        fp = gdir.get_filepath('gridded_data')
        with xr.open_dataset(fp) as ds:
            # 'topo' instead of 'topo_smoothed'?
            heights = ds.topo_smoothed.data[ds.glacier_mask.data == 1]
            widths = np.ones(len(heights))

    # Let's go
    # Climate period
    if ref_mb_years is not None:
        if len(ref_mb_years) > 2:
            years = np.asarray(ref_mb_years)
            ref_period = 'custom'
        else:
            years = np.arange(*ref_mb_years)
            ref_period = f'{ref_mb_years[0]}-01-01_{ref_mb_years[1]}-01-01'
    elif ref_period is not None:
        y0, y1 = ref_period.split('_')
        y0 = int(y0.split('-')[0])
        y1 = int(y1.split('-')[0])
        years = np.arange(y0, y1)
    else:
        raise InvalidParamsError('One of `ref_mb_years` or `ref_period` '
                                 'is required for calibration.')

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)
    if cmb != 0:
        raise NotImplementedError('Calving with geodetic MB is not implemented '
                                  'yet, but it should actually work. Well keep '
                                  'you posted!')

    # Ok, regardless on how we want to calibrate, we start with defaults
    if melt_f is None:
        melt_f = cfg.PARAMS['melt_f']

    if prcp_fac is None:
        if cfg.PARAMS['prcp_fac'] is None:
            prcp_fac = decide_winter_precip_factor(gdir)
        else:
            prcp_fac = cfg.PARAMS['prcp_fac']

    if temp_bias is None:
        temp_bias = 0

    # Create the MB model we will calibrate
    mb_mod = mb_model_class(gdir,
                            melt_f=melt_f,
                            temp_bias=temp_bias,
                            prcp_fac=prcp_fac,
                            check_calib_params=False)

    # Check that the years are available
    for y in years:
        if not mb_mod.is_year_valid(y):
            raise ValueError(f'year {y} out of the valid time bounds: '
                             f'[{mb_mod.ys}, {mb_mod.ye}]')

    if calibrate_param1 == 'melt_f':
        min_range, max_range = melt_f_min, melt_f_max
    elif calibrate_param1 == 'prcp_fac':
        min_range, max_range = prcp_fac_min, prcp_fac_max
    elif calibrate_param1 == 'temp_bias':
        min_range, max_range = temp_bias_min, temp_bias_max
    else:
        raise InvalidParamsError("calibrate_param1 must be one of "
                                 "['melt_f', 'prcp_fac', 'temp_bias']")

    def to_minimize(x, model_attr):
        # Set the new attr value
        setattr(mb_mod, model_attr, x)
        if use_2d_mb:
            out = mb_mod.get_specific_mb(heights=heights, widths=widths, year=years).mean()
        else:
            out = mb_mod.get_specific_mb(fls=fls, year=years).mean()
        return np.mean(out - ref_mb)

    try:
        optim_param1 = optimize.brentq(to_minimize,
                                       min_range, max_range,
                                       args=(calibrate_param1,)
                                       )
    except ValueError:
        if not calibrate_param2:
            raise RuntimeError(f'{gdir.rgi_id}: ref mb not matched. '
                               f'Try to set calibrate_param2.')

        # Check which direction we need to go
        diff_1 = to_minimize(min_range, calibrate_param1)
        diff_2 = to_minimize(max_range, calibrate_param1)
        optim_param1 = min_range if abs(diff_1) < abs(diff_2) else max_range
        setattr(mb_mod, calibrate_param1, optim_param1)

        # Second step
        if calibrate_param2 == 'melt_f':
            min_range, max_range = melt_f_min, melt_f_max
        elif calibrate_param2 == 'prcp_fac':
            min_range, max_range = prcp_fac_min, prcp_fac_max
        elif calibrate_param2 == 'temp_bias':
            min_range, max_range = temp_bias_min, temp_bias_max
        else:
            raise InvalidParamsError("calibrate_param2 must be one of "
                                     "['melt_f', 'prcp_fac', 'temp_bias']")
        try:
            optim_param2 = optimize.brentq(to_minimize,
                                           min_range, max_range,
                                           args=(calibrate_param2,)
                                           )
        except ValueError:
            # Third step
            if not calibrate_param3:
                raise RuntimeError(f'{gdir.rgi_id}: ref mb not matched. '
                                   f'Try to set calibrate_param3.')

            # Check which direction we need to go
            diff_1 = to_minimize(min_range, calibrate_param2)
            diff_2 = to_minimize(max_range, calibrate_param2)
            optim_param2 = min_range if abs(diff_1) < abs(diff_2) else max_range
            setattr(mb_mod, calibrate_param2, optim_param2)

            # Third step
            if calibrate_param3 == 'melt_f':
                min_range, max_range = melt_f_min, melt_f_max
            elif calibrate_param3 == 'prcp_fac':
                min_range, max_range = prcp_fac_min, prcp_fac_max
            elif calibrate_param3 == 'temp_bias':
                min_range, max_range = temp_bias_min, temp_bias_max
            else:
                raise InvalidParamsError("calibrate_param3 must be one of "
                                         "['melt_f', 'prcp_fac', 'temp_bias']")
            try:
                optim_param3 = optimize.brentq(to_minimize,
                                               min_range, max_range,
                                               args=(calibrate_param3,)
                                               )
            except ValueError:
                raise RuntimeError(f'{gdir.rgi_id}: we tried very hard but we '
                                   f'could not find a combination of '
                                   f'parameters that works for this ref mb.')

            if calibrate_param3 == 'melt_f':
                melt_f = optim_param3
            elif calibrate_param3 == 'prcp_fac':
                prcp_fac = optim_param3
            elif calibrate_param3 == 'temp_bias':
                temp_bias = optim_param3

        if calibrate_param2 == 'melt_f':
            melt_f = optim_param2
        elif calibrate_param2 == 'prcp_fac':
            prcp_fac = optim_param2
        elif calibrate_param2 == 'temp_bias':
            temp_bias = optim_param2

    if calibrate_param1 == 'melt_f':
        melt_f = optim_param1
    elif calibrate_param1 == 'prcp_fac':
        prcp_fac = optim_param1
    elif calibrate_param1 == 'temp_bias':
        temp_bias = optim_param1

    # Store parameters
    df = gdir.read_json('mb_calib', allow_empty=True)
    df['rgi_id'] = gdir.rgi_id
    df['bias'] = 0
    df['melt_f'] = melt_f
    df['prcp_fac'] = prcp_fac
    df['temp_bias'] = temp_bias
    # What did we try to match?
    df['reference_mb'] = ref_mb
    df['reference_mb_err'] = ref_mb_err
    df['reference_period'] = ref_period

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    df['mb_global_params'] = {k: cfg.PARAMS[k] for k in MB_GLOBAL_PARAMS}
    df['baseline_climate_source'] = gdir.get_climate_info()['baseline_climate_source']
    # Write
    if write_to_gdir:
        if gdir.has_file('mb_calib', filesuffix=filesuffix) and not overwrite_gdir:
            raise InvalidWorkflowError('`mb_calib.json` already exists for '
                                       'this repository. Set `overwrite_gdir` '
                                       'to True if you want to overwrite '
                                       'a previous calibration.')
        gdir.write_json(df, 'mb_calib', filesuffix=filesuffix)
    return df


@entity_task(log, writes=['mb_calib'])
def perturbate_mb_params(gdir, perturbation=None, reset_default=False, filesuffix=''):
    """Replaces pre-calibrated MB params with perturbed ones for this glacier.

    It simply replaces the existing `mb_calib.json` file with an
    updated one with perturbed parameters. The original ones
    are stored in the file for re-use after perturbation.

    Users can change the following 4 parameters:
    - 'melt_f': unit [kg m-2 day-1 K-1], the melt factor
    - 'prcp_fac': unit [-], the precipitation factor
    - 'temp_bias': unit [K], the temperature correction applied to the timeseries
    - 'bias': unit [mm we yr-1], *substracted* from the computed MB. Rarely used.

    All parameter perturbations are additive, i.e. the value
    provided by the user is added to the *precalibrated* value.
    For example, `temp_bias=1` means that the temp_bias used by the
    model will be the precalibrated one, plus 1 Kelvin.

    The only exception is prpc_fac, which is multiplicative.
    For example prcp_fac=1 will leave the precalibrated prcp_fac unchanged,
    while 2 will double it.

    Parameters
    ----------
    perturbation : dict
        the parameters to change and the associated value (see doc above)
    reset_default : bool
        reset the parameters to their original value. This might be
        unnecessary if using the filesuffix mechanism.
    filesuffix : str
        write the modified parameters in a separate mb_calib.json file
        with the filesuffix appended. This can then be read by the
        MassBalanceModel for example instead of the default one.
        Note that it's always the default, precalibrated params
        file which is read to start with.
    """
    df = gdir.read_json('mb_calib')

    # Save original params if not there
    if 'bias_orig' not in df:
        for k in ['bias', 'melt_f', 'prcp_fac', 'temp_bias']:
            df[k + '_orig'] = df[k]

    if reset_default:
        for k in ['bias', 'melt_f', 'prcp_fac', 'temp_bias']:
            df[k] = df[k + '_orig']
        gdir.write_json(df, 'mb_calib', filesuffix=filesuffix)
        return df

    for k, v in perturbation.items():
        if k == 'prcp_fac':
            df[k] = df[k + '_orig'] * v
        elif k in ['bias', 'melt_f', 'temp_bias']:
            df[k] = df[k + '_orig'] + v
        else:
            raise InvalidParamsError(f'Perturbation not valid: {k}')

    gdir.write_json(df, 'mb_calib', filesuffix=filesuffix)
    return df


def _check_terminus_mass_flux(gdir, fls):
    # Check that we have done this correctly
    rho = cfg.PARAMS['ice_density']
    cmb = calving_mb(gdir)

    # This variable is in "sensible" units normalized by width
    flux = fls[-1].flux_out
    aflux = flux * (gdir.grid.dx ** 2) / rho * 1e-9  # km3 ice per year

    # If not marine and a bit far from zero, warning
    if cmb == 0 and not np.allclose(flux, 0, atol=0.01):
        log.info('(%s) flux should be zero, but is: '
                 '%.4f km3 ice yr-1', gdir.rgi_id, aflux)

    # If not marine and quite far from zero, error
    if cmb == 0 and not np.allclose(flux, 0, atol=1):
        msg = ('({}) flux should be zero, but is: {:.4f} km3 ice yr-1'
               .format(gdir.rgi_id, aflux))
        raise MassBalanceCalibrationError(msg)


@entity_task(log, writes=['inversion_flowlines', 'linear_mb_params'])
def apparent_mb_from_linear_mb(gdir, mb_gradient=3., ela_h=None):
    """Compute apparent mb from a linear mass balance assumption (for testing).

    This is for testing currently, but could be used as alternative method
    for the inversion quite easily.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)
    is_calving = cmb != 0.

    # Get the height and widths along the fls
    h, w = gdir.get_inversion_flowline_hw()

    # Now find the ELA till the integrated mb is zero
    from oggm.core.massbalance import LinearMassBalance

    def to_minimize(ela_h):
        mbmod = LinearMassBalance(ela_h, grad=mb_gradient)
        smb = mbmod.get_specific_mb(heights=h, widths=w)
        return smb - cmb

    if ela_h is None:
        ela_h = optimize.brentq(to_minimize, -1e5, 1e5)

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
        fl.set_apparent_mb(mbz, is_calving=is_calving)

    # Check and write
    _check_terminus_mass_flux(gdir, fls)
    gdir.write_pickle(fls, 'inversion_flowlines')
    gdir.write_pickle({'ela_h': ela_h, 'grad': mb_gradient},
                      'linear_mb_params')


@entity_task(log, writes=['inversion_flowlines'])
def apparent_mb_from_any_mb(gdir, mb_model=None,
                            mb_model_class=MonthlyTIModel,
                            mb_years=None):
    """Compute apparent mb from an arbitrary mass balance profile.

    This searches for a mass balance residual to add to the mass balance
    profile so that the average specific MB is zero.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass balance model to use - if None, will use the
        one given by mb_model_class
    mb_model_class : MassBalanceModel class
        the MassBalanceModel class to use, default is MonthlyTIModel
    mb_years : array, or tuple of length 2 (range)
        the array of years from which you want to average the MB for (for
        mb_model only). If an array of length 2 is given, all years
        between this range (excluding the last one) are used.
        Default is to pick all years from the reference
        geodetic MB period, i.e. PARAMS['geodetic_mb_period'].
        It does not matter much for the final result, but it should be a
        period long enough to have a representative MB gradient.
    """

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)
    is_calving = cmb != 0

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')

    if mb_model is None:
        mb_model = mb_model_class(gdir)

    if mb_years is None:
        mb_years = cfg.PARAMS['geodetic_mb_period']
        y0, y1 = mb_years.split('_')
        y0 = int(y0.split('-')[0])
        y1 = int(y1.split('-')[0])
        mb_years = np.arange(y0, y1, 1)

    if len(mb_years) == 2:
        # Range
        mb_years = np.arange(*mb_years, 1)

    # Unchanged SMB
    rho = cfg.PARAMS['ice_density']
    mb_on_fl = []
    spec_mb = []
    area_smb = []
    for fl_id, fl in enumerate(fls):
        widths = fl.widths
        try:
            # For rect and parabola don't compute spec mb
            widths = np.where(fl.thick > 0, widths, 0)
        except AttributeError:
            pass
        mbz = 0
        smb = 0
        for yr in mb_years:
            amb = mb_model.get_annual_mb(fl.surface_h, fls=fls, fl_id=fl_id, year=yr)
            amb *= cfg.SEC_IN_YEAR * rho
            mbz += amb
            smb += weighted_average_1d(amb, widths)
        mb_on_fl.append(mbz / len(mb_years))
        spec_mb.append(smb / len(mb_years))
        area_smb.append(np.sum(widths))

    if len(mb_on_fl) == 1:
        o_smb = spec_mb[0]
    else:
        o_smb = weighted_average_1d(spec_mb, area_smb)

    def to_minimize(residual_to_opt):
        return o_smb + residual_to_opt - cmb

    residual = optimize.brentq(to_minimize, -1e5, 1e5)

    # Reset flux
    for fl in fls:
        fl.reset_flux()

    # Flowlines in order to be sure
    for fl_id, (fl, mbz) in enumerate(zip(fls, mb_on_fl)):
        fl.set_apparent_mb(mbz + residual, is_calving=is_calving)
        if fl_id < len(fls) and fl.flux_out < -1e3:
            log.warning('({}) a tributary has a strongly negative flux. '
                        'Inversion works but is physically quite '
                        'questionable.'.format(gdir.rgi_id))

    # Check and write
    _check_terminus_mass_flux(gdir, fls)
    gdir.add_to_diagnostics('apparent_mb_from_any_mb_residual', residual)
    gdir.write_pickle(fls, 'inversion_flowlines')


@entity_task(log)
def fixed_geometry_mass_balance(gdir, ys=None, ye=None, years=None,
                                monthly_step=False,
                                use_inversion_flowlines=True,
                                climate_filename='climate_historical',
                                climate_input_filesuffix='',
                                temperature_bias=None,
                                precipitation_factor=None,
                                mb_model_class=MonthlyTIModel):
    """Computes the mass balance with climate input from e.g. CRU or a GCM.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    use_inversion_flowlines : bool
        whether to use the inversion flowlines or the model flowlines
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        default is None and means that the precipitation factor from the
        calibration is applied which is cfg.PARAMS['prcp_fac']
    mb_model_class : MassBalanceModel class
        the MassBalanceModel class to use, default is MonthlyTIModel
    """

    if monthly_step:
        raise NotImplementedError('monthly_step not implemented yet')

    mbmod = MultipleFlowlineMassBalance(gdir, mb_model_class=mb_model_class,
                                        filename=climate_filename,
                                        use_inversion_flowlines=use_inversion_flowlines,
                                        input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        mbmod.temp_bias = temperature_bias
    if precipitation_factor is not None:
        mbmod.prcp_fac = precipitation_factor

    if years is None:
        if ys is None:
            ys = mbmod.flowline_mb_models[0].ys
        if ye is None:
            ye = mbmod.flowline_mb_models[0].ye
        years = np.arange(ys, ye + 1)

    odf = pd.Series(data=mbmod.get_specific_mb(year=years),
                    index=years)
    return odf


@entity_task(log)
def compute_ela(gdir, ys=None, ye=None, years=None, climate_filename='climate_historical',
                temperature_bias=None, precipitation_factor=None, climate_input_filesuffix='',
                mb_model_class=MonthlyTIModel):

    """Computes the ELA of a glacier for a for given years and climate.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year
    ye : int
        end year
    years : array of ints
        override ys and ye with the years of your choice
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix : str
        filesuffix for the input climate file
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        default is None and means that the precipitation factor from the
        calibration is applied which is cfg.PARAMS['prcp_fac']
    mb_model_class : MassBalanceModel class
        the MassBalanceModel class to use, default is MonthlyTIModel
    """

    mbmod = mb_model_class(gdir, filename=climate_filename,
                           input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        mbmod.temp_bias = temperature_bias
    if precipitation_factor is not None:
        mbmod.prcp_fac = precipitation_factor

    mbmod.valid_bounds = [-10000, 20000]

    if years is None:
        years = np.arange(ys, ye+1)

    ela = []
    for yr in years:
        ela.append(mbmod.get_ela(year=yr))

    odf = pd.Series(data=ela, index=years)
    return odf
