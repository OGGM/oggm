"""Mass balance models - next generation"""

# Built ins
import logging
import os
# External libs
import cftime
import numpy as np
import xarray as xr
import pandas as pd
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy import optimize
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_DAY
from oggm.utils import (SuperclassMeta, get_geodetic_mb_dataframe,
                        floatyear_to_date, date_to_floatyear, get_demo_file,
                        float_years_timeseries, ncDataset, get_temp_bias_dataframe,
                        clip_min, clip_max, clip_array, clip_scalar,
                        weighted_average_1d, lazy_property, set_array_type,
                        get_days_of_year, get_seconds_of_year, get_days_of_month,
                        get_seconds_of_month, )
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
    use_leap_years : bool, default: False
        If the calendar should use leap years
    """

    def __init__(self, gdir=None, use_leap_years=False):
        """ Initialize."""
        self.valid_bounds = None
        self.hemisphere = None
        if gdir is None:
            self.rho = cfg.PARAMS['ice_density']
        else:
            self.rho = gdir.settings['ice_density']
        self.use_leap_years = use_leap_years

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

    def days_in_month(self, year):
        """Get the number of days of a month, with or without leap years,
        depending on self.use_leap_years.

        Parameters
        ----------
        year: float
            The year (in the "floating year" convention).

        Returns
        -------
        int
            The number of days of the month
        """
        return get_days_of_month(year, use_leap_years=self.use_leap_years)

    def sec_in_month(self, year):
        """Get the seconds of the month, with or without leap years, depending
        on self.use_leap_years.

        Parameters
        ----------
        year: float
            The year (in the "floating year" convention).

        Returns
        -------
        int
            The seconds of the month
        """
        return get_seconds_of_month(year, use_leap_years=self.use_leap_years)

    def days_in_year(self, year):
        """Get the number of days of a year, with or without leap years,
        depending on self.use_leap_years.

        Parameters
        ----------
        year: float
            The year (in the "floating year" convention).

        Returns
        -------
        int
            The number of days of the year
        """
        return get_days_of_year(year, use_leap_years=self.use_leap_years)

    def sec_in_year(self, year):
        """Get the seconds of the year, with or without leap years, depending
        on self.use_leap_years.

        Parameters
        ----------
        year: float
            The year (in the "floating year" convention).

        Returns
        -------
        int
            The seconds of the year
        """
        return get_seconds_of_year(year, use_leap_years=self.use_leap_years)

    def get_daily_mb(self, heights, year=None, fl_id=None, fls=None):
        """Daily mass balance at given altitude(s) for a moment in time.

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

    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None):
        """Monthly mass balance at given altitude(s) for a moment in time.

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

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None,
                        time_resolution='annual'):
        """Specific mass balance for a given glacier geometry.

        Units depends on time_resolution:
        - 'annual': [mm w.e. yr-1], or millimeter water equivalent per year.
        - 'monthly': [mm w.e. month-1], or millimeter water equivalent per month.
        - 'daily': [mm w.e. day-1], or millimeter water equivalent per day.

        Parameters
        ----------
        heights : array_like, default None
            Altitudes at which the mass balance will be computed.
            Overridden by ``fls`` if provided.
        widths : array_like, default None
            Widths of the flowline (necessary for the weighted average).
            Overridden by ``fls`` if provided.
        fls : list[oggm.Flowline], default None
            List of flowline instances. Alternative to heights and
            widths, and overrides them if provided.
        year : array_like[float] or float, default None
            Year, or a range of years in "floating year" convention.
        time_resolution : str
            The resolution of the provided "floating year". Options are
            'annual', 'monthly' or 'daily'. Default is 'annual'.

        Returns
        -------
        np.ndarray
            Specific mass balance (units: mm w.e. yr-1).
        """
        stack = []
        year = np.atleast_1d(year)

        if time_resolution == 'annual':
            mb_function = self.get_annual_mb
            # mm w.e. yr-1

            unit_conversion = self.sec_in_year
        elif time_resolution == 'monthly':
            mb_function = self.get_monthly_mb

            # mm w.e. month-1
            unit_conversion = self.sec_in_month
        elif time_resolution == 'daily':
            mb_function = self.get_daily_mb

            # mm w.e. day-1
            def unit_conversion(x):
                return SEC_IN_DAY
        else:
            raise ValueError(f"time_resolution {time_resolution} not supported. "
                             "Options are 'annual', 'monthly' or 'daily'.")

        for mb_yr in year:
            if fls is not None:
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
                        mb_function(fl.surface_h, fls=fls, fl_id=i, year=mb_yr)
                    )
                # 2x faster than np.append
                widths = np.concatenate(widths, axis=0)
                mbs = np.concatenate(mbs, axis=0)
                mbs = weighted_average_1d(mbs, widths)
            else:
                mbs = mb_function(heights, year=mb_yr)
                mbs = weighted_average_1d(mbs, widths)
            mbs *= unit_conversion(mb_yr)
            stack.append(mbs)

        return set_array_type(stack) * self.rho

    def get_ela(self, year=None, **kwargs):
        """Get the equilibrium line altitude for a given year.

        Parameters
        ----------
        year : array_like[float] or float, default None
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
                year_length = self.sec_in_year(year=mb_year)

                def to_minimize(x):
                    return (self.get_annual_mb([x], year=mb_year, **kwargs)[0] *
                            year_length * self.rho)
                stack.append(optimize.brentq(to_minimize, *self.valid_bounds,
                                             xtol=0.1))

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
    """Monthly temperature index model."""

    def __init__(
        self,
        gdir,
        filename: str = 'climate_historical',
        input_filesuffix: str = '',
        settings_filesuffix: str = '',
        fl_id: int = None,
        melt_f: float = None,
        temp_bias: float = None,
        prcp_fac: float = None,
        bias: float = 0.0,
        temp_melt: float = None,
        ys: int = None,
        ye: int = None,
        repeat: bool = False,
        check_calib_params: bool = True,
        check_climate_data: bool = True,
        use_leap_years: bool = False,
    ):
        """Monthly temperature index model.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        filename : str, default 'climate_historical'
            Set to a different BASENAME if you want to use alternative
            climate data.
        input_filesuffix : str, optional
            Append a suffix to the climate input filename (useful for
            GCM runs).
        settings_filesuffix : str, optional
            Append a suffix to the settings file (useful for sensitivity
            runs).
        fl_id : int, optional
            If this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            The value of the melt factor you want to use, here the unit
            is kg m-2 day-1 K-1. Defaults to the calibrated value.
        temp_bias : float, optional
            The value of the temperature bias. Defaults to the
            calibrated value.
        prcp_fac : float, optional
            The value of the precipitation factor. Defaults to the
            calibrated value.
        bias : float, default 0.0
            The value of the calibration bias [mm we yr-1]. Defaults to
            the calibrated value. Note that this bias is *subtracted*
            from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        temp_melt : float or None, default None
            The threshold for the air temperature above which ice melt is
            assumed to occur (-1°C the default for monthly mb models). If None
            settings['temp_melt'] is used.
        ys : int, optional
            The start of the climate period where the MB model is valid.
            Defaults to the period with available data.
        ye : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        repeat : bool, default False
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        check_calib_params : bool, default True
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration
            and the ones you are using at run time. If they don't
            match, it will raise an error. Set to ``False`` to suppress
            this check.
        check_climate_data : bool, default True
            If True the climate input data is checked if it is provided in total
            years and that the length matches.
        use_leap_years : bool, default False
            If the calendar should use leap years
        """

        self.settings_filesuffix = settings_filesuffix
        gdir.settings_filesuffix = settings_filesuffix

        super(MonthlyTIModel, self).__init__(gdir=gdir,
                                             use_leap_years=use_leap_years)

        self.valid_bounds = [-1e4, 2e4]  # in m
        self.fl_id = fl_id  # which flowline are we the model of?
        self.gdir = gdir
        self.filename = filename
        self.input_filesuffix = input_filesuffix

        if melt_f is None:  # This prevents class methods
            melt_f = self.calib_params['melt_f']

        if temp_bias is None:
            temp_bias = self.calib_params['temp_bias']

        if prcp_fac is None:
            prcp_fac = self.calib_params['prcp_fac']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = self.calib_params['mb_global_params']
            for k, v in mb_calib.items():
                if v != self.gdir.settings[k]:
                    msg = ('You seem to use different mass balance parameters '
                           'than used for the calibration: '
                           f"you use gdir.settings['{k}']={gdir.settings[k]} while "
                           f"it was calibrated with gdir.settings['{k}']={v}. "
                           'Set `check_calib_params=False` to ignore this '
                           'warning.')
                    raise InvalidWorkflowError(msg)
            src = self.calib_params['baseline_climate_source']
            src_calib = gdir.get_climate_info(
                filename=self.filename, input_filesuffix=self.input_filesuffix
            )['baseline_climate_source']
            if src != src_calib:
                msg = (f'You seem to have calibrated with the {src} '
                       f"climate data while this gdir was calibrated with "
                       f"{src_calib}. Set `check_calib_params=False` to "
                       f"ignore this warning.")
                raise InvalidWorkflowError(msg)

        self.melt_f = melt_f
        self.bias = bias

        # Global parameters
        self.temp_all_solid = gdir.settings['temp_all_solid']
        self.temp_all_liq = gdir.settings['temp_all_liq']
        if temp_melt is None:
            self.temp_melt = gdir.settings['temp_melt']
        else:
            gdir.settings['temp_melt'] = temp_melt
            self.temp_melt = temp_melt

        # check if valid prcp_fac is used
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        self.temp_default_gradient = gdir.settings['temp_default_gradient']

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
            time = nc.variables["time"]
            time = cftime.num2date(time[:], time.units, calendar=time.calendar)

            # only use defined years
            years = np.array(list(map(lambda x: x.year, time)))
            pok = slice(None)  # take all is default (optim)
            if ys is not None:
                pok = years >= ys
            if ye is not None:
                try:
                    pok = pok & (years <= ye)
                except TypeError:
                    pok = years <= ye

            self.years = years[pok]
            self.months = np.array(list(map(lambda x: x.month, time)))[pok]
            self.days = np.array(list(map(lambda x: x.day, time)))[pok]

            if check_climate_data:
                # check for full years, this is overwritten for daily
                self._check_for_full_years()

            # Read timeseries and correct it
            self.temp = nc.variables["temp"][pok].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables["prcp"][pok].astype(np.float64) * self._prcp_fac

            grad = self.prcp * 0 + self.temp_default_gradient
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

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        """Precipitation factor (default: gdir.settings['prcp_fac'])

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
            return self.gdir.settings
        else:
            fp_fl_settings = self.gdir.get_filepath('settings',
                                                    filesuffix=f'_fl{self.fl_id}')
            if os.path.exists(fp_fl_settings):
                self.gdir.settings_filesuffix = f'_fl{self.fl_id}'
                out = self.gdir.settings
                if self.settings_filesuffix:
                    raise InvalidWorkflowError('settings_filesuffix cannot be '
                                               'used with multiple flowlines')
                return out
            else:
                return self.gdir.settings

    def _check_for_full_years(self):
        # We check for full calendar years
        if self.years[0] != self.years[-1]:
            nr_of_months = (self.years[-1] - self.years[0] + 1) * 12
        else:
            nr_of_months = 12
        len_data_ok = len(self.years) == nr_of_months
        months_ok = (self.months[0] == 1) or (self.months[-1] == 12)
        if not months_ok or not len_data_ok:
            raise InvalidWorkflowError(
                "We now work exclusively with full calendar years. Check "
                "provided climate data! \n Your current selection: "
                f"{self.months[0]:02d}.{self.years[0]:04d} - "
                f"{self.months[-1]:02d}.{self.years[-1]:04d}\n"
                f"Your data has {len(self.years)} timestamps, but we expect "
                f"{nr_of_months}."
            )

    def is_year_valid(self, year: int) -> bool:
        """Check if a year is within the climate period.

        Returns
        -------
        bool
            True if the year is within the climate period.
        """
        return self.ys <= year <= self.ye

    def validate_year(self, year: int) -> int:
        """Get and validate if a year is outside the data's time range.

        Raises
        ------
        ValueError
            If the year is outside of the data's climate period.
        """
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(year):  # this is overloaded by subclasses
            raise ValueError(
                f'year {year} out of the valid time bounds: '
                f'[{self.ys}, {self.ye}]'
            )
        return year

    def _get_tempformelt(self, temp):
        tempformelt = temp - self.temp_melt
        clip_min(tempformelt, 0, out=tempformelt)
        return tempformelt

    def _get_prcpsol(self, prcp, temp):
        fac = 1 - (temp - self.temp_all_solid) / (self.temp_all_liq - self.temp_all_solid)
        return prcp * clip_array(fac, 0, 1)

    def _get_climate_for_index(self, heights: ArrayLike, pok: ArrayLike
                               ) -> tuple:
        """Returns climate information at provided heights and time indexes.

        If only one time index is provided also the climate information is
        returned in 1D, otherwise in 2D.

        Parameters
        ----------
        heights : array_like
            the heights of interest in meters
        pok : np.ndarray
            the time indexes of interest

        Returns
        -------
        tuple[np.ndarray]
            Temperature, melt temperatures, total precipitation, and
            solid precipitation for each height pixel.
        """
        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        heights = np.asarray(heights)  # sometimes heights are passed as lists
        npix = len(heights)

        if np.size(pok) == 1:
            # fast path for 1D data (one time index)
            # Compute temp and tempformelt (temperature above melting threshold)
            temp = itemp + igrad * (heights - self.ref_hgt)
            tempformelt = self._get_tempformelt(temp)

            # Compute solid precipitation from total precipitation
            prcp = np.ones(npix) * iprcp
            prcpsol = self._get_prcpsol(prcp, temp)

            return temp, tempformelt, prcp, prcpsol

        # otherwise we need to handle 2D data
        # Compute temp and tempformelt (temperature above melting threshold)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(len(pok)).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = self._get_tempformelt(temp2d)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        prcpsol = self._get_prcpsol(prcp, temp2d)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """
        y, m = floatyear_to_date(year)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok,)

        # get tmelt for the entire month
        tmelt *= self.days_in_month(year=year)

        return t, tmelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year=None):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        year = self.validate_year(year=year)
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok)

        return t, tmelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        tuple
            Mean temperature, and sums of melt temperature,
            precipitation, and solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(
            heights=heights, year=year)
        myr = date_to_floatyear(np.repeat(int(np.floor(year)), 12),
                                np.arange(1, 13))
        days_of_month = [self.days_in_month(year=yr) for yr in myr]
        # get tmelt for the entire month
        tmelt *= days_of_month

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self,
                       heights: np.ndarray,
                       year: float = None,
                       add_climate: bool = False,
                       **kwargs,
                       ) -> np.float64 or tuple:
        """Get monthly mass balance.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Monthly mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)

        # lenght of the month in days already considered in tmelt in get_monhtly_climate
        mb_month = prcpsol - self.melt_f * tmelt
        sec_in_month = self.sec_in_month(year=year)
        mb_month -= (self.bias * sec_in_month / self.sec_in_year(year=year))
        if add_climate:
            return mb_month / sec_in_month / self.rho, t, tmelt, prcp, prcpsol
        return mb_month / sec_in_month / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        """Get annual mass balance.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        myr = date_to_floatyear(np.repeat(int(np.floor(year)), 12),
                                np.arange(1, 13))
        days_of_month = [self.days_in_month(year=yr) for yr in myr]
        # get tmelt for the entire month
        tmelt *= days_of_month

        mb_annual = np.sum(prcpsol - self.melt_f * tmelt, axis=1)
        mb_annual = ((mb_annual - self.bias) / self.sec_in_year(year=year) /
                     self.rho)
        if add_climate:
            return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        return mb_annual


class DailyTIModel(MonthlyTIModel):
    """Daily temperature index model."""

    def __init__(
        self,
        gdir,
        filename: str = 'climate_historical_daily',
        input_filesuffix: str = '',
        settings_filesuffix: str = '',
        fl_id: int = None,
        melt_f: float = None,
        temp_bias: float = None,
        prcp_fac: float = None,
        bias: float = 0.0,
        temp_melt: float = 0.0,
        ys: int = None,
        ye: int = None,
        repeat: bool = False,
        check_calib_params: bool = True,
        check_climate_data: bool = True,
        use_leap_years: bool = True,
    ):
        """Inherits from MonthlyTIModel.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        filename : str, default 'climate_historical_daily'
            Set to a different BASENAME if you want to use alternative
            climate data.
        input_filesuffix : str, optional
            Append a suffix to the climate input filename (useful for
            GCM runs).
        settings_filesuffix : str, optional
            Append a suffix to the settings file (useful for sensitivity
            runs).
        fl_id : int, optional
            If this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            The value of the melt factor you want to use, here the unit
            is kg m-2 day-1 K-1. Defaults to the calibrated value.
        temp_bias : float, optional
            The value of the temperature bias. Defaults to the
            calibrated value.
        prcp_fac : float, optional
            The value of the precipitation factor. Defaults to the
            calibrated value.
        bias : float, default 0.0
            The value of the calibration bias [mm we yr-1]. Defaults to
            the calibrated value. Note that this bias is *subtracted*
            from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        temp_melt : float, default 0.0
            The threshold for the air temperature above which ice melt is
            assumed to occur (0°C the default for daily mb models).
        ys : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        ye : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        repeat : bool, default False
            Whether the climate period given by [ys, ye] should be
            repeated indefinitely in a circular way.
        check_calib_params : bool, default True
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration
            and the ones you are using at run time. If they don't
            match, it will raise an error. Set to ``False`` to suppress
            this check.
        check_climate_data : bool, default True
            If True the climate input data is checked if it is provided in total
            years and that the length matches.
        use_leap_years : bool, default False
            If the calendar should use leap years
        """

        self.settings_filesuffix = settings_filesuffix
        gdir.settings_filesuffix = settings_filesuffix

        # Do not pass kwargs to prevent subclasses passing illegal args
        # to MonthlyTIModel.
        super(DailyTIModel, self).__init__(
            gdir=gdir,
            filename=filename,
            input_filesuffix=input_filesuffix,
            settings_filesuffix=settings_filesuffix,
            fl_id=fl_id,
            melt_f=melt_f,
            temp_bias=temp_bias,
            prcp_fac=prcp_fac,
            bias=bias,
            temp_melt=temp_melt,
            ys=ys,
            ye=ye,
            repeat=repeat,
            check_calib_params=check_calib_params,
            check_climate_data=check_climate_data,
            use_leap_years=use_leap_years,
        )

    def _check_for_full_years(self):
        # We check for full calendar years
        nr_of_days = 0
        for yr in np.arange(self.years[0], self.years[-1] + 1):
            nr_of_days += self.days_in_year(yr)
        len_data_ok = len(self.years) == nr_of_days
        months_ok = (self.months[0] == 1) and (self.months[-1] == 12)
        days_ok = (self.days[0] == 1) and (self.days[-1] == 31)
        if not months_ok or not days_ok or not len_data_ok:
            raise InvalidWorkflowError(
                "We now work exclusively with full calendar years (01.01. - "
                "31.12.). Check provided climate data!\nYour current selection: "
                f"{self.days[0]:02d}.{self.months[0]:02d}.{self.years[0]:04d} - "
                f"{self.days[-1]:02d}.{self.months[-1]:02d}.{self.years[-1]:04d}"
                f"\nYour data has {len(self.years)} timestamps, but we expect "
                f"{nr_of_days}."
            )

    def get_annual_climate(self, heights, year=None):
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def _get_2d_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        y, m = floatyear_to_date(year)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) & (self.months == m))[0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok)

        return t, tmelt, prcp, prcpsol

    def get_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """

        t, tmelt, prcp, prcpsol = self._get_2d_monthly_climate(heights, year)

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_daily_climate(self,
                          heights: np.ndarray,
                          year: float = None,
                          ) -> tuple:
        """Daily climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """
        y, m, d = floatyear_to_date(year, months_only=False)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) &
                       (self.months == m) &
                       (self.days == d))[0][0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok, )

        return t, tmelt, prcp, prcpsol

    def get_daily_mb(self,
                     heights: np.ndarray,
                     year: int = None,
                     add_climate: bool = False,
                     **kwargs,
                     ) -> np.float64 or tuple:
        """Get daily mass balance.

        Accounts for leap years by default.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : int, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Daily mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self.get_daily_climate(heights, year=year)

        mb_daily = prcpsol - self.melt_f * tmelt
        mb_daily -= (self.bias * SEC_IN_DAY / self.sec_in_year(year=year))

        if add_climate:
            return (mb_daily / SEC_IN_DAY / self.rho,
                    t, tmelt, prcp, prcpsol)
        return mb_daily / SEC_IN_DAY / self.rho

    def get_monthly_mb(self,
                       heights: np.ndarray,
                       year: float = None,
                       add_climate: bool = False,
                       **kwargs,
                       ) -> np.float64 or tuple:
        """Get monthly mass balance.

        Parameters
        ----------
        heights : np.ndarray
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclass implementations of this
            method.


        Returns
        -------
        np.float64 or tuple[np.ndarray]
            Monthly mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self._get_2d_monthly_climate(heights, year)

        mb_month = np.sum(prcpsol - self.melt_f * tmelt,
                          axis=1)
        sec_in_month = self.sec_in_month(year=year)
        mb_month -= (self.bias * sec_in_month / self.sec_in_year(year=year))

        if add_climate:
            return (mb_month / sec_in_month / self.rho, t.mean(axis=1),
                    tmelt.sum(axis=1), prcp.sum(axis=1), prcpsol.sum(axis=1))

        return mb_month / sec_in_month / self.rho

    def get_annual_mb(self,
                      heights: np.ndarray,
                      year: float = None,
                      add_climate: bool = False,
                      **kwargs,
                      ) -> np.float64 or tuple:
        """Get annual mass balance.

        This is equivalent to taking the sum of ``get_daily_mb``.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to ``get_2d_temperature``.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)

        mb_annual = np.sum(prcpsol - self.melt_f * tmelt,
                           axis=1)
        mb_annual = ((mb_annual - self.bias) / self.sec_in_year(year=year) /
                     self.rho)

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
        mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
            the MassBalanceModel to use for the constant climate
        y0 : int, required
            the year at the center of the period of interest.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        **kwargs:
            keyword arguments to pass to the mb_model_class
        """

        super().__init__()
        self.mbmod = mb_model_class(gdir=gdir, **kwargs)

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
        yrs = float_years_timeseries(self.years[0], self.years[-1],
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
        mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
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
        self.mbmod = mb_model_class(gdir=gdir, **kwargs)

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

    def __init__(self, gdir, settings_filesuffix='',
                 fls=None, mb_model_class=MonthlyTIModel,
                 use_inversion_flowlines=False,
                 input_filesuffix='',
                 **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        settings_filesuffix : str, optional
            You can use a different set of settings by providing a
            filesuffix. This is useful for sensitivity experiments.
        fls : list, optional
            list of flowline objects to use (defaults to 'model_flowlines')
        mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
            the MassBalanceModel to use e.g. ``DailyTIModel``,
            ``ConstantMassBalance``.
        use_inversion_flowlines: bool, optional
            use 'inversion_flowlines' instead of 'model_flowlines'
        kwargs : kwargs to pass to mb_model_class
        """

        gdir.settings_filesuffix = settings_filesuffix

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
                mb_model_class(
                    gdir=gdir,
                    settings_filesuffix=settings_filesuffix,
                    input_filesuffix=rgi_filesuffix,
                    **kwargs,
                )
            )

        self.valid_bounds = self.flowline_mb_models[-1].valid_bounds
        self.hemisphere = gdir.hemisphere
        self.rho = self.flowline_mb_models[-1].rho
        self.use_leap_years = self.flowline_mb_models[-1].use_leap_years

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

    def get_daily_mb(self, heights, year=None, fl_id=None, **kwargs):
        if fl_id is None:
            raise ValueError("`fl_id` is required for"
                             "MultipleFlowlineMassBalance.")

        return self.flowline_mb_models[fl_id].get_daily_mb(heights,
                                                           year=year,
                                                           **kwargs)

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

    def _get_mb_on_flowlines(self, fls=None, year=None, mb_call=None):
        if fls is None:
            fls = self.fls

        heights = []
        widths = []
        mbs = []
        for i, fl in enumerate(fls):
            h = fl.surface_h
            heights = np.append(heights, h)
            widths = np.append(widths, fl.widths)
            mbs = np.append(mbs, mb_call(h, year=year, fl_id=i))

        return heights, widths, mbs

    def get_daily_mb_on_flowlines(self, fls=None, year=None):
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
        return self._get_mb_on_flowlines(fls=fls, year=year,
                                         mb_call=self.get_daily_mb)

    def get_monthly_mb_on_flowlines(self, fls=None, year=None):
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
        return self._get_mb_on_flowlines(fls=fls, year=year,
                                         mb_call=self.get_monthly_mb)

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
        return self._get_mb_on_flowlines(fls=fls, year=year,
                                         mb_call=self.get_annual_mb)

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None,
                        time_resolution='annual',):
        """Specific mass balance for a given glacier geometry.

        Units depends on time_resolution:
        - 'annual': [mm w.e. yr-1], or millimeter water equivalent per year.
        - 'monthly': [mm w.e. month-1], or millimeter water equivalent per month.
        - 'daily': [mm w.e. day-1], or millimeter water equivalent per day.

        Parameters
        ----------
        heights : array_like, default None
            Altitudes at which the mass balance will be computed.
            Overridden by ``fls`` if provided.
        widths : array_like, default None
            Widths of the flowline (necessary for the weighted average).
            Overridden by ``fls`` if provided.
        fls : list[oggm.Flowline], default None
            List of flowline instances. Alternative to heights and
            widths, and overrides them if provided.
        year : array_like[float] or float, default None
            Year, or a range of years in "floating year" convention.
        time_resolution : str
            The resolution of the provided "floating year". Options are
            'annual', 'monthly' or 'daily'. Default is 'annual'.

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

        # we can use the function from MassBalanceModel as the correct mb models
        # are selected by the fl_id in get_annual_mb, get_monthly_mb and
        # get_daily_mb as defined in MultipleFlowlineMassBalance
        return super().get_specific_mb(fls=fls, year=year,
                                       time_resolution=time_resolution,)

    def get_ela(self, year=None, **kwargs):
        """Get the equilibrium line altitude for a given year.

        The ELA here is not without ambiguity: it computes a mean
        weighted by area.

        Parameters
        ----------
        year : array_like[float] or float, default None
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
    rho = gdir.settings['ice_density']
    return gdir.inversion_calving_rate * 1e9 * rho / gdir.rgi_area_m2


def decide_winter_precip_factor(gdir, baseline_climate_suffix:str=""):
    """Utility function to decide on a precip factor based on winter precip."""

    # We have to decide on a precip factor
    if 'W5E5' not in gdir.settings['baseline_climate']:
        raise InvalidWorkflowError('prcp_fac from_winter_prcp is only '
                                   'compatible with the W5E5 climate '
                                   'dataset!')

    # get non-corrected winter daily mean prcp (kg m-2 day-1)
    # it is easier to get this directly from the raw climate files
    fp = gdir.get_filepath('climate_historical', filesuffix=baseline_climate_suffix)
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
        if not baseline_climate_suffix:
            assert len(ds_pr_winter.time) == 41 * 7, text
        w_prcp = float((ds_pr_winter / ds_pr_winter.time.dt.daysinmonth).mean())

    # from MB sandbox calibration to winter MB
    # using t_melt=-1, cte lapse rate, monthly resolution
    a, b = gdir.settings['winter_prcp_fac_ab']
    prcp_fac = a * np.log(w_prcp) + b
    # don't allow extremely low/high prcp. factors!!!
    return clip_scalar(prcp_fac,
                       gdir.settings['prcp_fac_min'],
                       gdir.settings['prcp_fac_max'])


@entity_task(log, writes=['mb_calib'])
def mb_calibration_from_wgms_mb(gdir, settings_filesuffix='',
                                observations_filesuffix='',
                                **kwargs):
    """Calibrate for in-situ, annual MB.

    This only works for glaciers which have WGMS data!

    For now this just calls mb_calibration_from_scalar_mb internally,
    but could be cleverer than that if someone wishes to implement it.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    observations_filesuffix: str
        The observations filesuffix, where the used calibration data will be
        stored. Code-wise the observations_filesuffix is set in the @entity-task
        decorater.
    **kwargs : any kwarg accepted by mb_calibration_from_scalar_mb
    except `ref_mb` and `ref_mb_years`
    """

    # Note that this currently does not work for hydro years (WGMS uses hydro)
    # A way to go would be to teach the mb models to use calendar years
    # internally but still output annual MB in hydro convention.
    mbdf = gdir.get_ref_mb_data()
    # Keep only valid values
    mbdf = mbdf.loc[~mbdf['ANNUAL_BALANCE'].isnull()]

    gdir.observations['ref_mb'] = {
        'value': mbdf['ANNUAL_BALANCE'].mean(),
        'years': mbdf.index.values,
    }

    return mb_calibration_from_scalar_mb(gdir,
                                         settings_filesuffix=settings_filesuffix,
                                         observations_filesuffix=observations_filesuffix,
                                         **kwargs)


@entity_task(log, writes=['mb_calib'])
def mb_calibration_from_hugonnet_mb(gdir, *,
                                    settings_filesuffix='',
                                    observations_filesuffix='',
                                    use_observations_file=False,
                                    ref_mb_period=None,
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
                                    **kwargs: dict,
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
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    observations_filesuffix: str
        The observations filesuffix, where the used calibration data will be
        stored. Code-wise the observations_filesuffix is set in the @entity-task
        decorater.
    use_observations_file : bool
        By default this function reads the data from Hugonnet and adds it to the
        observations file. If you want to use different observations within this
        function you can set this to True. This can be useful for sensitivity
        tests. Default is False.
    ref_mb_period : str, default: PARAMS['geodetic_mb_period']
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
    mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
        the MassBalanceModel to use for the calibration. Needs to use the
        same parameters as MonthlyTIModel (the default): melt_f,
        temp_bias, prcp_fac.
    kwargs : dict
        kwargs to pass to the mb_model_class instance

    Returns
    -------
    the calibrated parameters as dict
    """

    # instead of the given values by hugonnet use the once from the provided
    # observations file
    if use_observations_file:
        ref_mb_use = gdir.observations['ref_mb']
    else:
        if not ref_mb_period:
            ref_mb_period = gdir.settings['geodetic_mb_period']

        # Get the reference data
        ref_mb_err = np.nan
        if use_regional_avg:
            ref_mb_df = 'table_hugonnet_regions_10yr_20yr_ar6period.csv'
            ref_mb_df = pd.read_csv(get_demo_file(ref_mb_df))
            ref_mb_df = ref_mb_df.loc[ref_mb_df.period == ref_mb_period].set_index('reg')
            # dmdtda already in kg m-2 yr-1
            ref_mb = ref_mb_df.loc[int(gdir.rgi_region), 'dmdtda']
            ref_mb_err = ref_mb_df.loc[int(gdir.rgi_region), 'err_dmdtda']
        else:
            try:
                ref_mb_df = get_geodetic_mb_dataframe().loc[gdir.rgi_id]
                ref_mb_df = ref_mb_df.loc[ref_mb_df['period'] == ref_mb_period]
                # dmdtda: in meters water-equivalent per year -> we convert to kg m-2 yr-1
                ref_mb = ref_mb_df['dmdtda'].iloc[0] * 1000
                ref_mb_err = ref_mb_df['err_dmdtda'].iloc[0] * 1000
            except KeyError:
                if override_missing is None:
                    raise
                ref_mb = override_missing

        ref_mb_use = {
            'value': ref_mb,
            'period': ref_mb_period,
            'err': ref_mb_err,
        }

    gdir.observations['ref_mb'] = ref_mb_use

    temp_bias = 0
    if gdir.settings['use_temp_bias_from_file']:
        climinfo = gdir.get_climate_info()
        if 'w5e5' not in climinfo['baseline_climate_source'].lower():
            raise InvalidWorkflowError('use_temp_bias_from_file currently '
                                       'only available for W5E5 data.')
        bias_df = get_temp_bias_dataframe()
        ref_lon = climinfo['baseline_climate_ref_pix_lon']
        ref_lat = climinfo['baseline_climate_ref_pix_lat']
        # Take nearest
        dis = ((bias_df.lon_val - ref_lon)**2 + (bias_df.lat_val - ref_lat)**2)**0.5
        sel_df = bias_df.iloc[np.argmin(dis)]
        temp_bias = sel_df['median_temp_bias_w_err_grouped']
        assert np.isfinite(temp_bias), 'Temp bias not finite?'

    if informed_threestep:
        if not gdir.settings['use_temp_bias_from_file']:
            raise InvalidParamsError('With `informed_threestep` you need to '
                                     'set `use_temp_bias_from_file`.')
        if not gdir.settings['use_winter_prcp_fac']:
            raise InvalidParamsError('With `informed_threestep` you need to '
                                     'set `use_winter_prcp_fac`.')

        # Some magic heuristics - we just decide to calibrate
        # precip -> melt_f -> temp but informed by previous data.

        # Temp bias was decided anyway, we keep as previous value and
        # allow it to vary as last resort

        # We use the precip factor but allow it to vary between 0.8, 1.2 of
        # the previous value (uncertainty).
        prcp_fac = decide_winter_precip_factor(gdir)
        mi, ma = gdir.settings['prcp_fac_min'], gdir.settings['prcp_fac_max']
        prcp_fac_min = clip_scalar(prcp_fac * 0.8, mi, ma)
        prcp_fac_max = clip_scalar(prcp_fac * 1.2, mi, ma)

        return mb_calibration_from_scalar_mb(gdir,
                                             settings_filesuffix=settings_filesuffix,
                                             observations_filesuffix=observations_filesuffix,
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
                                             **kwargs
                                             )

    else:
        return mb_calibration_from_scalar_mb(gdir,
                                             settings_filesuffix=settings_filesuffix,
                                             observations_filesuffix=observations_filesuffix,
                                             write_to_gdir=write_to_gdir,
                                             overwrite_gdir=overwrite_gdir,
                                             use_2d_mb=use_2d_mb,
                                             calibrate_param1=calibrate_param1,
                                             calibrate_param2=calibrate_param2,
                                             calibrate_param3=calibrate_param3,
                                             temp_bias=temp_bias,
                                             mb_model_class=mb_model_class,
                                             **kwargs
                                             )


@entity_task(log, writes=['mb_calib'])
def mb_calibration_from_scalar_mb(gdir, *,
                                  settings_filesuffix='',
                                  observations_filesuffix='',
                                  overwrite_observations=False,
                                  ref_mb=None,
                                  ref_mb_err=None,
                                  ref_mb_period=None,
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
                                  **kwargs: dict,
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
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    observations_filesuffix: str
        You can provide a filesuffix for the mb observations to use. If you
        provide ref_mb, ref_mb_err, ref_mb_period and/or ref_mb_years, then this
        values will be stored in the observations file, if ref_mb is not already
        present. If you want to force to use the provided values and override
        the current ones, set overwrite_observations to True. Code-wise the
        observations_filesuffix is set in the @entity-task decorater.
    overwrite_observations : bool
        If you want to overwrite already existing observation values in the
        provided observations file set this to True. Default is False.
    ref_mb : float, required
        the reference mass balance to match (units: kg m-2 yr-1)
        It is required here - if you want to use available observations,
        use :py:func:`oggm.core.massbalance.mb_calibration_from_geodetic_mb`
        or :py:func:`oggm.core.massbalance.mb_calibration_from_wgms_mb`.
    ref_mb_err : float, optional
        currently only used for logging - it is not used in the calibration.
    ref_mb_period : str, optional
        date format - for example '2000-01-01_2010-01-01'. If this is not
        set, ref_mb_years needs to be set.
    ref_mb_years : tuple of length 2 (range) or list of years.
        convenience kwarg to override ref_mb_period. If a tuple of length 2 is
        given, all years between this range (excluding the last one) are used.
        If a list  of years is given, all these will be used (useful for
        data with gaps)
    write_to_gdir : bool
        whether to write the results of the calibration to the glacier
        directory. If True (the default), this will be saved as `mb_calib.json`
        and be used by the MassBalanceModel class as parameters in subsequent
        tasks.
    overwrite_gdir : bool
        if mass balance parameters exists, this task won't overwrite it per
        default. Set this to True to enforce overwriting (i.e. with consequences
        for the future workflow).
    use_2d_mb : bool
        Set to True if the mass balance calibration has to be done of the 2D mask
        of the glacier (for fully distributed runs only).
    mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
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
        optimizing MB). Defaults to gdir.settings['melt_f'].
    melt_f_min: float
        the minimum accepted value for the melt factor during optimisation.
        Defaults to gdir.settings['melt_f_min'].
    melt_f_max: float
        the maximum accepted value for the melt factor during optimisation.
        Defaults to gdir.settings['melt_f_max'].
    prcp_fac: float
        the default value to use as precipitation scaling factor
        (or the starting value when optimizing MB). Defaults to the method
        chosen in `params.cfg` (winter prcp or global factor).
    prcp_fac_min: float
        the minimum accepted value for the precipitation scaling factor during
        optimisation. Defaults to gdir.settings['prcp_fac_min'].
    prcp_fac_max: float
        the maximum accepted value for the precipitation scaling factor during
        optimisation. Defaults to gdir.settings['prcp_fac_max'].
    temp_bias: float
        the default value to use as temperature bias (or the starting value when
        optimizing MB). Defaults to 0.
    temp_bias_min: float
        the minimum accepted value for the temperature bias during optimisation.
        Defaults to gdir.settings['temp_bias_min'].
    temp_bias_max: float
        the maximum accepted value for the temperature bias during optimisation.
        Defaults to gdir.settings['temp_bias_max'].
    kwargs: dict
        kwargs to pass to the mb_model_calss instance
    """

    # Param constraints
    if melt_f_min is None:
        melt_f_min = gdir.settings['melt_f_min']
    if melt_f_max is None:
        melt_f_max = gdir.settings['melt_f_max']
    if prcp_fac_min is None:
        prcp_fac_min = gdir.settings['prcp_fac_min']
    if prcp_fac_max is None:
        prcp_fac_max = gdir.settings['prcp_fac_max']
    if temp_bias_min is None:
        temp_bias_min = gdir.settings['temp_bias_min']
    if temp_bias_max is None:
        temp_bias_max = gdir.settings['temp_bias_max']
    if ref_mb_years is not None and ref_mb_period is not None:
        raise InvalidParamsError('Cannot set `ref_mb_years` and `ref_mb_period` '
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

    # handle which ref mb to use (provided or in observations file)
    ref_mb_provided = {}
    if ref_mb is not None:
        ref_mb_provided['value'] = ref_mb
    if ref_mb_err is not None:
        ref_mb_provided['err'] = ref_mb_err
    if ref_mb_period is not None:
        ref_mb_provided['period'] = ref_mb_period
    if ref_mb_years is not None:
        ref_mb_provided['years'] = ref_mb_years

    if 'ref_mb' in gdir.observations:
        ref_mb_in_file = gdir.observations['ref_mb']
    else:
        ref_mb_in_file = None

    # if nothing is provided raise an error
    if (ref_mb_provided == {}) and (ref_mb_in_file is None):
        raise InvalidWorkflowError(
            'You have not provided an reference mass balance! Either add it to '
            'the observations file '
            f'({os.path.basename(gdir.observations.path)}), or pass it through '
            f'kwargs (ref_mb, ref_mb_err, ref_mb_period/ref_mb_years.')

    # here handle different cases of provided values for the ref mb
    if (ref_mb_in_file is None) or (ref_mb_in_file is not None and
                                    overwrite_observations):
        gdir.observations['ref_mb'] = ref_mb_provided
        ref_mb_use = ref_mb_provided
    elif ref_mb_in_file is not None and ref_mb_provided == {}:
        # only provided in file, this is ok so continue
        ref_mb_use = ref_mb_in_file
    else:
        # if the provided is the same as the one stored in the file it is fine
        if ref_mb_in_file != ref_mb_provided:
            raise InvalidWorkflowError(
                'You provided a reference mass balance, but their is already '
                'one stored in the current observation file '
                f'({os.path.basename(gdir.observations.path)}). If you want to '
                'overwrite set overwrite_observations = True.')
        else:
            ref_mb_use = ref_mb_in_file

    # now we can extract the actual values we want to use
    ref_mb = ref_mb_use['value']
    if 'err' in ref_mb_use:
        ref_mb_err = ref_mb_use['err']
    if 'period' in ref_mb_use:
        ref_mb_period = ref_mb_use['period']
    if 'years' in ref_mb_use:
        ref_mb_years = ref_mb_use['years']

    # Let's go
    # Climate period
    # TODO: doesn't support hydro years
    if ref_mb_years is not None:
        if len(ref_mb_years) > 2:
            years = np.asarray(ref_mb_years)
            ref_mb_period = 'custom'
        else:
            years = np.arange(*ref_mb_years)
            ref_mb_period = f'{ref_mb_years[0]}-01-01_{ref_mb_years[1]}-01-01'
        gdir.observations['ref_mb']['period'] = ref_mb_period
    elif ref_mb_period is not None:
        y0, y1 = ref_mb_period.split('_')
        y0 = int(y0.split('-')[0])
        y1 = int(y1.split('-')[0])
        years = np.arange(y0, y1)
    else:
        raise InvalidParamsError('One of `ref_mb_years` or `ref_mb_period` '
                                 'is required for calibration.')

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)
    if cmb != 0:
        raise NotImplementedError('Calving with geodetic MB is not implemented '
                                  'yet, but it should actually work. Well keep '
                                  'you posted!')

    # Ok, regardless on how we want to calibrate, we start with defaults
    if melt_f is None:
        melt_f = gdir.settings.defaults['melt_f']

    if prcp_fac is None:
        if gdir.settings['use_winter_prcp_fac']:
            # Some sanity check
            if gdir.settings['prcp_fac'] is not None:
                raise InvalidWorkflowError("Set PARAMS['prcp_fac'] to None "
                                           "if using PARAMS['winter_prcp_factor'].")
            prcp_fac = decide_winter_precip_factor(gdir)
        else:
            prcp_fac = gdir.settings.defaults['prcp_fac']
            if prcp_fac is None:
                raise InvalidWorkflowError(
                    'Set either PARAMS["use_winter_prcp_fac"] '
                    'or PARAMS["winter_prcp_factor"].'
                )

    if temp_bias is None:
        temp_bias = 0

    # Create the MB model we will calibrate
    mb_mod = mb_model_class(
        gdir=gdir,
        melt_f=melt_f,
        temp_bias=temp_bias,
        prcp_fac=prcp_fac,
        check_calib_params=False,
        settings_filesuffix=settings_filesuffix,
        **kwargs
    )

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
    df = {}
    df['rgi_id'] = gdir.rgi_id
    df['bias'] = 0
    df['melt_f'] = melt_f
    df['prcp_fac'] = prcp_fac
    df['temp_bias'] = temp_bias
    # What did we try to match?
    df['reference_mb'] = ref_mb
    df['reference_mb_err'] = ref_mb_err
    df['reference_period'] = ref_mb_period

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    df['mb_global_params'] = {k: gdir.settings[k] for k in MB_GLOBAL_PARAMS}
    df['baseline_climate_source'] = gdir.get_climate_info(
        filename=mb_mod.filename, input_filesuffix=mb_mod.input_filesuffix
    )['baseline_climate_source']

    # Write
    if write_to_gdir:
        if any(key in gdir.settings
               for key in ['melt_f', 'prcp_fac', 'temp_bias']) and not overwrite_gdir:
            raise InvalidWorkflowError('Their are already mass balance parameters '
                                       'stored in the settings file. Set '
                                       '`overwrite_gdir` to True if you want to '
                                       'overwrite a previous calibration.')
        for key in ['rgi_id', 'bias', 'melt_f', 'prcp_fac', 'temp_bias',
                    'reference_mb', 'reference_mb_err', 'reference_period',
                    'mb_global_params', 'baseline_climate_source']:
            gdir.settings[key] = df[key]

    return df


@entity_task(log, writes=['mb_calib'])
def perturbate_mb_params(gdir, perturbation=None, reset_default=False, filesuffix=''):
    """Replaces pre-calibrated MB params with perturbed ones for this glacier.

    It simply replaces the existing `mb_calib.json` file with an
    updated one with perturbed parameters. The original ones
    are stored in the file for re-use after perturbation.

    Users can change the following 4 parameters:
        - **melt_f': unit [kg m-2 day-1 K-1], the melt factor.
        - **prcp_fac': unit [-], the precipitation factor.
        - **temp_bias': unit [K], the temperature correction applied to the timeseries.
        - **bias': unit [mm we yr-1], *subtracted* from the computed MB. Rarely used.

    All parameter perturbations are additive, i.e. the value
    provided by the user is added to the *precalibrated* value.
    For example, `temp_bias=1` means that the temp_bias used by the
    model will be the precalibrated one, plus 1 Kelvin.

    The only exception is prpc_fac, which is multiplicative.
    For example prcp_fac=1 will leave the precalibrated prcp_fac unchanged,
    while 2 will double it.

    Parameters
    ----------
    gdir : GlacierDirectory
        The glacier directory.
    perturbation : dict
        The parameters to change and the associated value (see doc above)
    reset_default : bool, default False
        Reset the parameters to their original value. This might be
        unnecessary if using the filesuffix mechanism.
    filesuffix : str, optional
        Write the modified parameters in a separate mb_calib.json file
        with the filesuffix appended. This can then be read by the
        MassBalanceModel for example instead of the default one.
        Note that it's always the default, precalibrated params
        file which is read to start with.
    """
    df = gdir.read_yml('settings')

    # Save original params if not there
    if 'bias_orig' not in df:
        for k in ['bias', 'melt_f', 'prcp_fac', 'temp_bias']:
            df[k + '_orig'] = df[k]

    if reset_default:
        for k in ['bias', 'melt_f', 'prcp_fac', 'temp_bias']:
            df[k] = df[k + '_orig']
        gdir.write_yml(df, 'settings', filesuffix=filesuffix)
        return df

    for k, v in perturbation.items():
        if k == 'prcp_fac':
            df[k] = df[k + '_orig'] * v
        elif k in ['bias', 'melt_f', 'temp_bias']:
            df[k] = df[k + '_orig'] + v
        else:
            raise InvalidParamsError(f'Perturbation not valid: {k}')

    gdir.write_yml(df, 'settings', filesuffix=filesuffix)
    return df


def _check_terminus_mass_flux(gdir, fls):
    # Check that we have done this correctly
    rho = gdir.settings['ice_density']
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
def apparent_mb_from_linear_mb(gdir, settings_filesuffix='',
                               mb_gradient=3., ela_h=None):
    """Compute apparent mb from a linear mass balance assumption (for testing).

    This is for testing currently, but could be used as alternative method
    for the inversion quite easily.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
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
    rho = gdir.settings['ice_density']
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
def apparent_mb_from_any_mb(gdir, settings_filesuffix='',
                            input_filesuffix=None,
                            output_filesuffix=None,
                            mb_model=None,
                            mb_model_class=MonthlyTIModel,
                            mb_years=None):
    """Compute apparent mb from an arbitrary mass balance profile.

    This searches for a mass balance residual to add to the mass balance
    profile so that the average specific MB is zero.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    input_filesuffix: str
        the filesuffix of the inversion flowlines which should be used (useful
        for conducting multiple experiments in the same gdir)
    output_filesuffix: str
        the filesuffix of the final inversion flowlines which are saved back
        into the gdir (useful for conducting multiple experiments in the same
        gdir)
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass balance model to use - if None, will use the
        one given by mb_model_class.
    mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
        The MassBalanceModel class to use.
    mb_years : array_like or tuple, default None
        The array of years over which you want to average the mass
        balance. This argument has little effect on the final result,
        but it should be a period long enough to have a representative
        mass balance gradient.
        If an array of length 2 is given, the method uses all years
        between this range, excluding the last one.
        If None, the method will use all the years from the reference
        geodetic mass balance period ``gdir.settings['geodetic_mb_period']``.
    """

    if input_filesuffix is None:
        input_filesuffix = settings_filesuffix

    if output_filesuffix is None:
        output_filesuffix = settings_filesuffix

    # Do we have a calving glacier?
    cmb = calving_mb(gdir)
    is_calving = cmb != 0

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines', filesuffix=input_filesuffix)

    if mb_model is None:
        mb_model = mb_model_class(gdir, settings_filesuffix=settings_filesuffix)

    if mb_years is None:
        mb_years = gdir.settings['geodetic_mb_period']
        y0, y1 = mb_years.split('_')
        y0 = int(y0.split('-')[0])
        y1 = int(y1.split('-')[0])
        mb_years = np.arange(y0, y1, 1)

    if len(mb_years) == 2:
        # Range
        mb_years = np.arange(*mb_years, 1)

    # Unchanged SMB
    rho = gdir.settings['ice_density']
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
            amb *= mb_model.sec_in_year(year=yr) * rho
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
    gdir.settings['apparent_mb_from_any_mb_residual'] = residual

    # this is only for backwards compatibility
    if settings_filesuffix == '':
        gdir.add_to_diagnostics('apparent_mb_from_any_mb_residual', residual)

    gdir.write_pickle(fls, 'inversion_flowlines', filesuffix=output_filesuffix)


@entity_task(log)
def fixed_geometry_mass_balance(gdir, settings_filesuffix='',
                                ys=None, ye=None, years=None,
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
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int, optional
        End year of the model run (default: from the climate file).
    years : array_like[int], optional
        Override ``ys`` and ``ye`` with the years of your choice.
    monthly_step : bool, default False
        Store the diagnostic data at a monthly time resolution.
        If False, stores it at an annual resolution.
    use_inversion_flowlines : bool, default True
        Use the inversion flowlines instead of the model flowlines.
    climate_filename : str, default 'climate_historical'
        Name of the climate file, e.g. 'climate_historical', 'gcm_data'.
    climate_input_filesuffix: str, optional
        Filesuffix for the input climate file.
    temperature_bias : float, optional
        Add a bias to the temperature timeseries.
    precipitation_factor: float, optional
        Multiplicative factor applied to the precipitation time series.
        If None, uses the precipitation factor from the calibration in
        ``gdir.settings['prcp_fac']``.
    mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
        The MassBalanceModel class to use.
    """

    if monthly_step:
        raise NotImplementedError('monthly_step not implemented yet')

    mbmod = MultipleFlowlineMassBalance(gdir, mb_model_class=mb_model_class,
                                        filename=climate_filename,
                                        use_inversion_flowlines=use_inversion_flowlines,
                                        input_filesuffix=climate_input_filesuffix,
                                        settings_filesuffix=settings_filesuffix)

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

    odf = pd.Series(data=mbmod.get_specific_mb(year=years), index=years)
    return odf


@entity_task(log)
def compute_ela(gdir, settings_filesuffix='',
                ys=None, ye=None, years=None, climate_filename='climate_historical',
                temperature_bias=None, precipitation_factor=None, climate_input_filesuffix='',
                mb_model_class=MonthlyTIModel):

    """Computes the ELA of a glacier for specific years and climate.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
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

    mbmod = mb_model_class(gdir, settings_filesuffix=settings_filesuffix,
                           filename=climate_filename,
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
