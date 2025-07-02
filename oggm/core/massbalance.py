"""Mass balance models - next generation"""

# Built ins
import logging
import os
import calendar
import warnings
import copy
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
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH, SEC_IN_DAY
from oggm.utils import (SuperclassMeta, get_geodetic_mb_dataframe,
                        floatyear_to_date, date_to_floatyear, get_demo_file,
                        monthly_timeseries, ncDataset, get_temp_bias_dataframe,
                        clip_min, clip_max, clip_array, clip_scalar,
                        weighted_average_1d, weighted_average_2d,
                        lazy_property, set_array_type, get_total_days)
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
                year_length = self.get_year_length(year=mb_year)
                def to_minimize(x):
                    return (self.get_annual_mb([x], year=mb_year, **kwargs)[0] *
                    year_length * self.rho)
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

    def get_year_length(self, year: float = None) -> float:
        """Get the number of seconds in a year.

        Parameters
        ----------
        year : float or int
            Year in floating year convention.

        Returns
        -------
        float
            The number of seconds in a year.
        """
        return SEC_IN_YEAR


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
        bias : float, default 0.0
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        ys : int, optional
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int, optional
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        repeat : bool, default False
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        check_calib_params : bool, default True
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
            self.set_temporal_bounds(nc_data=nc, ys=ys, ye=ye, default_grad=default_grad)

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

    def set_temporal_bounds(
        self, nc_data: xr.DataArray, ys: int, ye: int, default_grad: float
    ) -> None:
        """Constrain data to a start and end year.

        Parameters
        ----------
        nc_data : xr.DataArray
            Climate data.
        ys : int or float
            Desired first year of data period.
        ye : int or float
            Desired last year of data period.
        default_grad : float
            Default temperature gradient.
        """
        time = nc_data.variables["time"]
        time = cftime.num2date(time[:], time.units, calendar=time.calendar)
        ny, r = divmod(len(time), 12)
        if r:
            raise ValueError("Climate data should be N full years")

        # We check for calendar years
        if (time[0].month != 1) or (time[-1].month != 12):
            raise InvalidWorkflowError(
                "We now work exclusively with " "calendar years."
            )

        # Quick trick because we know the size of our array
        years = np.repeat(
            np.arange(time[-1].year - ny + 1, time[-1].year + 1), 12
        )
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
        self.temp = (
            nc_data.variables["temp"][pok].astype(np.float64) + self._temp_bias
        )
        self.prcp = (
            nc_data.variables["prcp"][pok].astype(np.float64) * self._prcp_fac
        )

        grad = self.prcp * 0 + default_grad
        self.grad = grad
        self.ref_hgt = nc_data.ref_hgt
        self.climate_source = nc_data.climate_source
        self.ys = self.years[0]
        self.ye = self.years[-1]

    def is_year_valid(self, year: int) -> bool:
        """Check if a year is within the climate period.

        Returns
        -------
        bool
            True if the year is within the climate period.
        """
        return self.ys <= year <= self.ye

    def get_valid_year(self, year: int) -> int:
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

    def get_time_index(self, year: int, month: int) -> np.ndarray:
        """Get time index for a particular year and month.
        
        Parameters
        ----------
        year : int
            Integer year.
        month : int
            Integer month.

        Returns
        -------
        np.ndarray
            Time index (months).
        """
        time_index = np.where((self.years == year) & (self.months == month))[0][0]

        return time_index

    def get_indexed_climate_data(self, index: ArrayLike) -> tuple:
        """Get climate data at an index.

        Data are already corrected for temperature bias and
        precipitation factor.

        Parameters
        ----------
        index : array_like
            Any indexing system compatible with ``np.ndarray``. For most
            operations this will be a time index.

        Returns
        -------
        tuple[np.ndarray[np.float64]]
            Temperature, precipitation, and slope gradient for a given
            index.
        """
        return self.temp[index], self.prcp[index], self.grad[index]

    def get_precipitation(
        self, precipitation: np.ndarray, temperature: np.ndarray, npix: int
    ) -> tuple:
        """Get total and solid precipitation.

        Parameters
        ----------
        precipitation : np.ndarray[np.float64]
            Precipitation.
        temperature : np.ndarray[np.float64]
            Temperatures.
        npix : int
            Number of pixels.

        Returns
        -------
        tuple[np.ndarray]
            Total and solid precipitation.
        """
        prcp_total = np.ones(npix) * precipitation
        fac = 1 - (temperature - self.t_solid) / (self.t_liq - self.t_solid)
        prcp_solid = prcp_total * clip_array(fac, 0, 1)

        return prcp_total, prcp_solid

    def get_2d_precipitation(
        self, precipitation: np.ndarray, temperature: np.ndarray, npix: int
    ) -> tuple:
        """Get total and solid precipitation.

        Parameters
        ----------
        precipitation : np.ndarray[np.float64]
            Precipitation.
        temperature : np.ndarray[np.float64]
            Temperatures.
        npix : int
            Number of pixels.

        Returns
        -------
        tuple[np.ndarray]
            Total and solid precipitation.
        """
        prcp_total = np.atleast_2d(precipitation).repeat(npix, 0)
        fac = 1 - (temperature - self.t_solid) / (self.t_liq - self.t_solid)
        prcp_solid = prcp_total * clip_array(fac, 0, 1)

        return prcp_total, prcp_solid

    def get_2d_temperature(
        self,
        heights: np.ndarray,
        temperatures: np.ndarray,
        gradients: np.ndarray,
        npix: int,
        timesteps: int,
    ) -> np.ndarray:
        """Get temperature from 2D data.

        Parameters
        ----------
        heights : array_like
            Flowline heights [m].
        temperatures : np.ndarray[np.float64]
            Temperatures.
        gradients : np.ndarray[np.float64]
            Slope gradients.
        npix : int
            Number of pixels.
        timesteps : int
            Temporal resolution.

        Returns
        -------
        np.ndarray[np.float64]
            Temperatures.
        """
        heights = np.asarray(heights)  # sometimes passed as list
        grad_temp = np.atleast_2d(gradients).repeat(npix, 0)
        grad_temp *= (
            heights.repeat(timesteps).reshape(grad_temp.shape) - self.ref_hgt
        )
        temperature = np.atleast_2d(temperatures).repeat(npix, 0) + grad_temp

        return temperature

    def get_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Flowline heights [m].
        year : float, optional
            Hydrological year. Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """
        y, m = floatyear_to_date(year)
        y = self.get_valid_year(year=y)
        pok = self.get_time_index(year=y, month=m)

        itemp, iprcp, igrad = self.get_indexed_climate_data(index=pok)

        temp, tempformelt, prcp, prcpsol = self.get_monthly_climate_data(
            heights=heights,
            itemp=itemp,
            igrad=igrad,
            iprcp=iprcp,
            pok=pok,
        )

        return temp, tempformelt, prcp, prcpsol

    def get_melt_temperature(self, temperature: np.ndarray) -> np.ndarray:
        """Get melt temperatures.

        Parameters
        ----------
        temperature : np.ndarray
            Temperatures.

        Returns
        -------
        np.ndarray
            Melt temperatures.
        """
        melt_temperature = temperature - self.t_melt
        clip_min(melt_temperature, 0, out=melt_temperature)

        return melt_temperature

    def get_monthly_climate_data(
        self,
        heights: ArrayLike,
        itemp: np.ndarray,
        igrad: np.ndarray,
        iprcp: np.ndarray,
        **kwargs,
    ) -> tuple:
        """Get climate data for each height pixel.

        Parameters
        ----------
        heights : array_like
            Flowline heights [m].
        itemp : np.ndarray[np.float64]
            Temperatures during a given time period.
        igrad : np.ndarray[np.float64]
            Slope gradient during a given time period.
        iprcp : np.ndarray[np.float64]
            Precipitation during a given time period.
        **kwargs
            Extra arguments are passed to child implementations of this
            method.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, solid
            precipitation.
        """
        npix = len(heights)
        temperature = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        melt_temperature = self.get_melt_temperature(temperature=temperature)
        prcp, prcpsol = self.get_precipitation(
            precipitation=iprcp, temperature=temperature, npix=npix
        )

        return temperature, melt_temperature, prcp, prcpsol

    def get_annual_climate_data(
        self,
        heights: ArrayLike,
        itemp: np.ndarray,
        igrad: np.ndarray,
        iprcp: np.ndarray,
        **kwargs,
    ) -> tuple:
        """Get climate data for each height pixel.

        Parameters
        ----------
        heights : array_like
            Flowline heights [m].
        itemp : np.ndarray[np.float64]
            Temperatures during a given time period.
        igrad : np.ndarray[np.float64]
            Slope gradient during a given time period.
        iprcp : np.ndarray[np.float64]
            Precipitation during a given time period.
        **kwargs
            Extra arguments are passed to child implementations of this
            method.

        Returns
        -------
        tuple[np.ndarray]
            Temperature, melt temperatures, total precipitation, and
            solid precipitation for each height pixel.
        """
        pok: np.ndarray = kwargs.get("pok", itemp)  # get time index
        heights = np.asarray(heights)  # sometimes heights are passed as lists
        npix = len(heights)

        temperature = self.get_2d_temperature(
            heights=heights,
            temperatures=itemp,
            gradients=igrad,
            npix=npix,
            timesteps=len(pok),
        )
        melt_temperature = self.get_melt_temperature(temperature=temperature)
        prcp, prcpsol = self.get_2d_precipitation(
            precipitation=iprcp, temperature=temperature, npix=npix
        )

        return temperature, melt_temperature, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        year = self.get_valid_year(year=year)
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError(f'Year {int(year)} not in record')

        itemp, iprcp, igrad = self.get_indexed_climate_data(index=pok)
        temp2d, temp2dformelt, prcp, prcpsol = self.get_annual_climate_data(
            heights, itemp, igrad, iprcp, pok=pok
        )

        return temp2d, temp2dformelt, prcp, prcpsol

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
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(
        self,
        heights: np.ndarray,
        year: float = None,
        add_climate: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Get monthly mass balance.

        Parameters
        ----------
        heights : array_like
            Flowline heights [m].
        year : float, optional
            Hydrological year. Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Monthly mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
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


class DailyTIModel(MonthlyTIModel):
    """Daily temperature index model.

    Adapted from OGGM/massbalance-sandbox.
    """

    def __init__(
        self,
        gdir,
        filename: str = "climate_historical_daily",
        input_filesuffix : str = "",
        mb_params_filesuffix : str = "",
        fl_id : int = None,
        melt_f : float = None,
        temp_bias : float = None,
        prcp_fac : float = None,
        bias : float = 0.0,
        ys : int = None,
        ye : int = None,
        repeat : bool = False,
        check_calib_params : bool = True,
    ):
        """Inherits from MonthlyTIModel.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        filename : str, default "climate_historical_daily"
            Set to a different BASENAME if you want to use alternative
            climate data.
        input_filesuffix : str, optional
            Append a suffix to the climate input filename (useful for
            GCM runs).
        mb_params_filesuffix : str, optional
            Append a suffix to the mb params calibration file (useful
            for sensitivity runs).
        fl_id : int, optional
            If this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            Set to the value of the melt factor you want to use, here
            the unit is kg m-2 day-1 K-1 (the default is to use the
            calibrated value).
        temp_bias : float, optional
            Set to the value of the temperature bias you want to use
            (the default is to use the calibrated value).
        prcp_fac : float, optional
            Set to the value of the precipitation factor you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            Set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated
            value). Note that this bias is *subtracted* from the
            computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data).
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data).
        repeat : bool
            Whether the climate period given by [ys, ye] should be
            repeated indefinitely in a circular way.
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration
            and the ones you are using at run time. If they don't
            match, it will raise an error. Set to "False" to suppress
            this check.
        """

        # Do not pass kwargs to prevent subclasses of DailyTIModel from
        # passing illegal args to MonthlyTIModel.
        super().__init__(
            gdir=gdir,
            filename=filename,
            input_filesuffix=input_filesuffix,
            mb_params_filesuffix=mb_params_filesuffix,
            fl_id=fl_id,
            melt_f=melt_f,
            temp_bias=temp_bias,
            prcp_fac=prcp_fac,
            bias=bias,
            ys=ys,
            ye=ye,
            repeat=repeat,
            check_calib_params=check_calib_params,
        )

    def set_temporal_bounds(
        self, nc_data: xr.DataArray, ys: int, ye: int, default_grad: float
    ) -> None:
        """Constrain data to a start and end year.

        Parameters
        ----------
        nc_data : xr.DataArray
            Climate data.
        ys : int or float
            Desired first year of data period.
        ye : int or float
            Desired last year of data period.
        default_grad : float
            Default temperature gradient.
        """
        time = nc_data.variables["time"]
        time = cftime.num2date(time[:], time.units, calendar=time.calendar)
        # data are no longer in hydro years
        # time = calendardate_to_hydrodate_cftime(dates=time, start_month=int(time[0].month))

        # 1.5x faster than np.vectorize
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

        self.temp = (
            nc_data.variables["temp"][pok].astype(np.float64) + self._temp_bias
        )
        # this is prcp computed by instantiation, which changes if
        # prcp_fac is updated (see @property)
        self.prcp = (
            nc_data.variables["prcp"][pok].astype(np.float64) * self._prcp_fac
        )

        grad = self.prcp * 0 + default_grad
        self.grad = grad
        self.ref_hgt = nc_data.ref_hgt
        self.climate_source = nc_data.climate_source
        self.ys = self.years[0]
        self.ye = self.years[-1]

    def get_time_index(self, year: int, month: int) -> np.ndarray:
        """Get time index for a particular year and month.

        Parameters
        ----------
        year : int
            Integer year.
        month : int
            Integer month.

        Returns
        -------
        np.ndarray
            Time index (days).
        """
        time_index = np.where((self.years == year) & (self.months == month))[0]

        return time_index

    def get_monthly_climate_data(
        self,
        heights: ArrayLike,
        itemp: np.ndarray,
        igrad: np.ndarray,
        iprcp: np.ndarray,
        **kwargs,
    ) -> tuple:
        """Get climate data for each height pixel.

        Parameters
        ----------
        heights : array_like
            Flowline heights [m].
        itemp : np.ndarray[np.float64]
            Temperatures during a given time period.
        igrad : np.ndarray[np.float64]
            Slope gradient during a given time period.
        iprcp : np.ndarray[np.float64]
            Precipitation during a given time period.
        **kwargs
            Extra arguments passed to ``get_2d_temperature``.

        Returns
        -------
        tuple[np.ndarray]
            Temperature, melt temperatures, total precipitation, and
            solid precipitation for each height pixel."""

        return self.get_annual_climate_data(
            heights=heights, itemp=itemp, igrad=igrad, iprcp=iprcp, **kwargs
        )

    def get_daily_climate_data(
        self,
        heights: ArrayLike,
        itemp: np.ndarray,
        igrad: np.ndarray,
        iprcp: np.ndarray,
        **kwargs,
    ) -> tuple:
        """Get climate data for each height pixel.

        Parameters
        ----------
        heights : array_like
            Flowline heights [m].
        itemp : np.ndarray[np.float64]
            Temperatures during a given time period.
        igrad : np.ndarray[np.float64]
            Slope gradient during a given time period.
        iprcp : np.ndarray[np.float64]
            Precipitation during a given time period.
        **kwargs
            Extra arguments passed to ``get_2d_temperature``.

        Returns
        -------
        tuple[np.ndarray]
            Temperature, melt temperatures, total precipitation, and
            solid precipitation for each height pixel."""

        temp2d, temp2dformelt, prcp, prcpsol = self.get_annual_climate_data(
            heights=heights, itemp=itemp, igrad=igrad, iprcp=iprcp, **kwargs
        )

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_melt_temperature(
        self, temperature: np.ndarray, time_index: np.ndarray = None
    ) -> np.ndarray:
        """Get melt temperature.

        This method is kept in case it needs modifying for synthetic
        daily data (mb_pseudo_daily).

        Parameters
        ----------
        temperature : np.ndarray
            Temperatures.
        time_index : np.ndarray, default None
            Measurement times. Currently a placeholder for synthetic
            data.

        Returns
        -------
        np.ndarray
            Melt temperatures.
        """
        melt_temperature = temperature - self.t_melt
        clip_min(melt_temperature, 0, out=melt_temperature)

        return melt_temperature

    def get_monthly_mb(
        self,
        heights: ArrayLike,
        year: float = None,
        add_climate: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Get monthly mass balance.

        Parameters
        ----------
        heights : array_like
            Flowline heights [m].
        year : float, optional
            Hydrological year. Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Monthly mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        temperature, melt_temperature, prcp, prcpsol = self.get_monthly_climate(
            heights, year=year
        )
        # more accurate than using mean days per month
        days_in_month = prcpsol.shape[1]
        mb_month = np.sum(
            prcpsol - self.melt_f * days_in_month * melt_temperature, axis=1
        )
        mb_month = (
            (mb_month - self.bias) / (SEC_IN_DAY * days_in_month) / self.rho
        )
        if add_climate:
            return (
                mb_month,
                temperature.mean(axis=1),
                melt_temperature.sum(axis=1),
                prcp.sum(axis=1),
                prcpsol.sum(axis=1),
            )

        return mb_month

    def get_annual_mb(
        self,
        heights: ArrayLike,
        year: float = None,
        add_climate: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Get annual mass balance.

        This is equivalent to taking the sum of ``get_daily_mb``.

        Parameters
        ----------
        heights : ArrayLike
            Flowline heights [m].
        year : float, optional
            Hydrological year. Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        temperature, melt_temperature, prcp, prcpsol = (
            self._get_2d_annual_climate(heights, year)
        )

        year_length = self.get_year_length(year)
        upscale_factor = prcpsol.shape[1] / 365
        mb_annual = np.sum(
            prcpsol - self.melt_f * upscale_factor * melt_temperature,
            axis=1,
        )

        mb_annual = (
            (mb_annual - self.bias * upscale_factor)
            / year_length
            / self.rho
        )
        if add_climate:
            return (
                mb_annual,
                temperature.mean(axis=1),
                melt_temperature.sum(axis=1),
                prcp.sum(axis=1),
                prcpsol.sum(axis=1),
            )
        return mb_annual

    def get_daily_mb(
        self,
        heights: ArrayLike,
        year: int = None,
        add_climate: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Get daily mass balance.

        Accounts for leap years.

        Parameters
        ----------
        heights : array_like
            Flowline heights [m].
        year : int, optional
            Hydrological year. Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Daily mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        if isinstance(year, float):
            raise TypeError("Year must be an integer.")  # Why?

        temperature, melt_temperature, prcp, prcpsol = (
            self._get_2d_daily_climate(heights=heights, year=year)
        )
        upscale_factor = prcpsol.shape[1] / 365
        mb_daily = (
            prcpsol - self.melt_f * upscale_factor * melt_temperature
        )

        mb_daily = (
            (mb_daily - self.bias * upscale_factor) / SEC_IN_DAY / self.rho
        )
        if add_climate:
            return (
                mb_daily,
                temperature.mean(axis=1),
                melt_temperature.sum(axis=1),
                prcp.sum(axis=1),
                prcpsol.sum(axis=1),
            )
        return mb_daily

    def _get_2d_daily_climate(
        self, heights: ArrayLike, year: ArrayLike
    ) -> tuple:
        """Get daily climate data for specific years and heights.

        Parameters
        ----------
        heights : ArrayLike
            Flowline heights [m].
        year : ArrayLike
            Year or range of years.

        Returns
        -------
        tuple[np.ndarray]
            Temperature, melt temperatures, total precipitation, and
            solid precipitation for each height pixel.
        """
        # This works because we overload methods called by the parent's
        # ``_get_2d_annual_climate`` with this class' methods.

        return self._get_2d_annual_climate(heights=heights, year=year)

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None):
        """Specific mass balance for a given glacier geometry.

        Inheriting avoids recalculating SEC_IN_YEAR for each year. This
        changes the year length in the unit, but not how the specific MB
        is calculated.

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
        smb = super().get_specific_mb(
            heights=heights, widths=widths, fls=fls, year=year
        )
        mask = np.vectorize(calendar.isleap)(year)  # strangely faster than map
        smb = np.where(mask, smb * 366 / 365, smb)

        return smb

    def get_specific_mb_daily(self, heights=None, widths=None, fls=None, year=None):
        """ returns specific daily mass balance in kg m-2 day

        (implemented in order that Sarah Hanus can use daily input and also gets daily output)
        """
        if len(np.atleast_1d(year)) > 1:
            stack = []
            for yr in year:
                out = self.get_specific_mb_daily(heights=heights, widths=widths, year=yr)
                stack = np.append(stack, out)
            return np.asarray(stack)

        mb = self.get_daily_mb(heights, year=year)
        spec_mb = np.average(mb * self.rho * SEC_IN_DAY, weights=widths, axis=0)
        assert len(spec_mb) > 360
        return spec_mb

    def get_year_length(self, year: float = None) -> float:
        """Get the number of seconds in a year.

        Parameters
        ----------
        year : float or int
            Year in floating year convention.

        Returns
        -------
        float
            The number of seconds in a year.
        """
        if not calendar.isleap(year):
            return SEC_IN_YEAR
        else:
            return SEC_IN_DAY * 366


class DailySfcTIModel(DailyTIModel):
    """Daily temperature index model with surface tracking.

    Distinguishes surface types using a bucket system. Adapted from
    OGGM/massbalance-sandbox. Incompatible with hydro years.
    """

    def __init__(
            self,
            gdir,
            filename="climate_historical_daily",
            input_filesuffix : str = "",
            mb_params_filesuffix : str = "",
            fl_id : int = None,
            melt_f:float=None,
            temp_bias : float = None,
            prcp_fac : float = None,
            bias : float = 0.0,
            ys : int = None,
            ye : int = None,
            repeat : bool = False,
            check_calib_params : bool = True,
            resolution : str = "mb_real_daily",
            gradient_scheme : str = "var_an_cycle",
            melt_f_ratio : float = 0.5,
            melt_frequency : str = 'annual',
            melt_f_change : str = 'linear',
            spinup_years : int = 6,
            tau_e : float = 1.0,  #0.5,
            check_data_exists:bool = True,
            optim:bool = False,
            hbins: ArrayLike = None,
            buckets: ArrayLike = None,
            **kwargs,
        ):

        # doc_DailySfcTIModel =
        """
        Other terms are equal to TIModel_Parent!
        The following parameters are initialized specifically only for DailySfcTIModel

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        filename : str, optional
            Set to a different BASENAME if you want to use alternative climate
            data. Default is 'climate_historical_daily'.
        input_filesuffix : str, optional
            Append a suffix to the climate input filename (useful for
            GCM runs).
        mb_params_filesuffix : str, optional
            Append a suffix to the mb params calibration file (useful
            for sensitivity runs).
        fl_id : int, optional
            If this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            Set to the value of the melt factor you want to use, here
            the unit is kg m-2 day-1 K-1
            (the default is to use the calibrated value).
        temp_bias : float, optional
            Set to the value of the temperature bias you want to use
            (the default is to use the calibrated value).
        prcp_fac : float, optional
            Set to the value of the precipitation factor you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            Set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated
            value). Note that this bias is *subtracted* from the
            computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data).
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data).
        repeat : bool
            Whether the climate period given by [ys, ye] should be
            repeated indefinitely in a circular way.
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration
            and the ones you are using at run time. If they don't
            match, it will raise an error. Set to "False" to suppress
            this check.
        resolution : str, default 'day'
            Temporal mass balance resolution.
        gradient_scheme : str, default 'annual'
            Temperature gradient scheme:
              - *annual*: Spatial and seasonal variation, constant over
                multiple years.
        melt_f_ratio : float, default 0.5
            Ratio of snow melt factor to ice melt factor.
            Between 0 and 1, where 1 is no surface type distinction.
            Default 0.5 to match GloGEM.
        melt_frequency : str, default "year"
            Frequency to update melt factor, either "year" or "month".
            If annual, the model uses one snow and five firn buckets.
            If monthly, the snow ages over the number of spinup years.
        melt_f_change : str, default "linear"
            How the snow melt factor changes relative to the ice melt
            factor. Either:
              - **linear**: Linear with time.
              - **neg_exp**: Melt factor changes exponentially:
              .. math::
              
                  K_{{f}} = K_{{f_{{ice}}}}
                          + (K_{{f_{{snow}}}} - K_{{f_{{ice}}}})
                          * \\exp{{
                              \\frac{{-t_{{year}}}}{{\\tau_{{e}}}}
                              }}
        spinup_years : int, default 6
            Number of spinup years. Every bucket can be filled after
            the spinup period.
        tau_e : float, default 0.5
            How quickly the snow melt factor approximates the ice melt
            factor. Must be larger than zero to prevent ``melt_f``
            being set to NaN in the first bucket. Only used if
            ``melt_f_change`` is set to ``neg_exp``.
        check_data_exists : bool, default True
            Check if the year/month has already been computed. If True,
            use the pre-computed output from ``pd_mb_monthly`` or
            ``pd_mb_annual``. Set to False if using ``random_climate``
            or ``constant_climate``.
        optim : bool, default False
            Deprecated. If True, estimates a CTE bucket profile during
            spinup and reuses it to compute MB for all heights, ignoring
            changes in surface type over time.
        hbins: ArrayLike, default None
            Height bins for classifying surface types. Only needed for
            ``ConstantMBModel``. If None, defaults to ``np.nan``.
        buckets: ArrayLike, default None
            Buckets for surface tracking scheme.
        """
        
        super().__init__(
            gdir=gdir,
            filename=filename,
            melt_f=melt_f,
            input_filesuffix = input_filesuffix,
            mb_params_filesuffix = mb_params_filesuffix,
            fl_id = fl_id,
            temp_bias = temp_bias,
            prcp_fac = prcp_fac,
            bias = bias,
            ys = ys,
            ye = ye,
            repeat = repeat,
            check_calib_params = check_calib_params,
        )
        self.set_flowline_from_gdir()

        if hbins is not None:
            self.hbins = np.nan
        else:
            self.hbins = hbins
        self.optim = optim
        self.resolution = resolution
        self.gradient_scheme = gradient_scheme

        self.tau_e = tau_e
        if melt_f_change == 'neg_exp' and tau_e > 0:
            raise InvalidParamsError("tau_e has to be above zero")
        self.melt_f_change = melt_f_change
        if melt_f_change not in ['linear', 'neg_exp']:
            raise InvalidParamsError("melt_f_change has to be either 'linear' or 'neg_exp'")
        # ratio of snow melt_f to ice melt_f
        self.melt_f_ratio = melt_f_ratio
        self.melt_frequency = melt_frequency
        self.spinup_years = spinup_years
        # amount of bucket and bucket naming depends on melt_frequency:
        if self.melt_frequency == 'annual':
            self.buckets = ['snow', 'firn_yr_1', 'firn_yr_2', 'firn_yr_3',
                            'firn_yr_4', 'firn_yr_5']
        elif self.melt_frequency == 'monthly':
            # for each month over 6 years a bucket-> in total 72!
            self.buckets = np.arange(0, 12 * 6, 1).tolist()
        else:
            raise InvalidParamsError('melt_frequency can either be annual or monthly!')
        # first bucket: if annual it is 'snow', if monthly update it is 0
        self.first_snow_bucket = self.buckets[0]

        self.columns = self.buckets + ['delta_kg/m2']
        # TODO: maybe also include snow_delta_kg/m2, firn_yr_1_delta_kg/m2...
        #  (only important if I add refreezing or a densification scheme)
        # comment: I don't need an ice bucket because this is assumed to be "infinite"
        # (instead just have a 'delta_kg/m2' bucket)

        # save the inversion height to later check if the same height is applied!!!
        self.inv_heights = self.fl.surface_h
        self.check_data_exists = check_data_exists

        # container template (has to be updatable -> pd_mb_template is property/setter thing)
        # use the distance_along_flowline as index
        self._pd_mb_template = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                            columns=[]) # exectime-> directly addind columns here should be faster
        self._pd_mb_template.index.name = 'distance_along_flowline'

        # bucket template:
        # make a different template for the buckets, because we want directly the right columns inside
        self._pd_mb_template_bucket = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                            columns=self.columns)  # exectime-> directly addind columns here should be faster
        self._pd_mb_template_bucket.index.name = 'distance_along_flowline'

        # storage containers for monthly and annual mb
        # columns are the months or years respectively
        # IMPORTANT: those need other columns
        self.pd_mb_monthly = self._pd_mb_template.copy()
        self.pd_mb_annual = self._pd_mb_template.copy()
        self.pd_mb_daily = self._pd_mb_template.copy()

        # bucket containers with buckets as columns
        pd_bucket = self._pd_mb_template_bucket.copy()
        # exectime comment:  this line is quite expensive -> actually this line can be removed as
        # self._pd_mb_template was defined more clever !!!
        # pd_bucket[self.columns] = 0
        # I don't need a total_kg/m2 because I don't know it anyway
        # as we do this before inversion!



        # storage container for buckets (in kg/m2)
        # columns are the buckets
        # (6*12 or 6 buckets depending if melt_frequency is monthly or annual)
        self.pd_bucket = pd_bucket
        # self.set_melt_f(melt_f)
        # self.melt_f = melt_f


    def set_flowline_from_gdir(self):
            self.fl = self.gdir.read_pickle("inversion_flowlines")[-1]

    def set_melt_f(self, new_melt_f):
        """sets new melt_f and if DailySfcTIModel resets the pd_mb and buckets
        """
        # first update self._melt_f
        self.melt_f = new_melt_f

        self.reset_pd_mb_bucket()
        # In addition, we need to recompute here once the melt_f buckets
        # (as they depend on self._melt_f)
        # IMPORTANT: this has to be done AFTER self._melt_f got updated!!!
        self.recompute_melt_f_buckets()

    @property
    def melt_f(self):
        """ prints the _melt_f """
        return self._melt_f

    @melt_f.setter
    def melt_f(self, new_melt_f):
        """ sets new melt_f and if DailySfcTIModel resets the pd_mb and buckets
        """
        # first update self._melt_f
        self._melt_f = new_melt_f
        if isinstance(self, DailySfcTIModel) and hasattr(self, "optim"):
            self.reset_pd_mb_bucket()
            # In addition, we need to recompute here once the melt_f buckets
            # (as they depend on self._melt_f)
            # IMPORTANT: this has to be done AFTER self._melt_f got updated!!!
            self.recompute_melt_f_buckets()

    @property
    def prcp_fac(self):
        """ prints the _prcp_fac as initiated or changed"""
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        """ sets new prcp_fac, changes prcp time series
         and if DailySfcTIModel resets the pd_mb and buckets
        """
        if new_prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        # attention, prcp_fac should not be called here
        # otherwise there is recursion occurring forever...
        # use new_prcp_fac to not get maximum recusion depth error
        self.prcp *= new_prcp_fac / self._prcp_fac

        if type(self) == DailySfcTIModel:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()

        # update old prcp_fac in order that it can be updated
        # again ...
        self._prcp_fac = new_prcp_fac

    @property
    def temp_bias(self):
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):
        self.temp += new_temp_bias - self._temp_bias

        if type(self) == DailySfcTIModel:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()

        # update old temp_bias in order that it can be updated again ...
        self._temp_bias = new_temp_bias


    @property
    def melt_f_buckets(self):
        # interpolated from snow melt_f to ice melt_f by using melt_f_ratio and tau_e
        # to get the bucket melt_f. Need to do this here and not in init, because otherwise it does not get updated.
        # Or I would need to set melt_f as property / setter function that updates self.melt_f_buckets....

        # exectime too long --> need to make that more efficient:
        #  only "update" melt_f_buckets and recompute them if melt_f
        #  is changed (melt_f_ratio, tau_e and amount of buckets do not change after instantiation!)
        # Only if _melt_f_buckets does not exist, compute it out of self.melt_f ... and the other stuff!!
        try:
            return self._melt_f_buckets
        except:
            if self.melt_f_change == 'linear':
                self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                                np.linspace(self.melt_f * self.melt_f_ratio,
                                                            self.melt_f, len(self.buckets)+1)
                                                ))
            elif self.melt_f_change == 'neg_exp':
                time = np.linspace(0, 6, len(self.buckets + ['ice']))
                melt_f_snow = self.melt_f_ratio * self.melt_f
                self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                                self.melt_f + (melt_f_snow - self.melt_f) * np.exp(-time/self.tau_e)
                                                ))

            # at the moment we don't allow to set externally the melt_f_buckets but this could be added later ...
            return self._melt_f_buckets

    def recompute_melt_f_buckets(self):
        # This is called always when self.melt_f is updated  (the other self.PARAMS are not changed after instantiation)
        # IMPORTANT:
        # - we need to use it after having updated self._melt_f inside of melt_f setter  (is in DailySfcTIModel)!!!
        # - we need to use self._melt_f -> as this is what is before changed inside of self.melt_f!!!

        # interpolate from snow melt_f to ice melt_f by using melt_f_ratio and tau_e
        # to get the bucket melt_f. Need to do this here and not in init, because otherwise it does not get updated.

        # comment exectime: made that more efficient. Now it only "updates" melt_f_buckets and recomputes them if melt_f
        # is changed (melt_f_ratio_snow<_to_ice, tau_e and amount of buckets do not change after instantiation)!
        if self.melt_f_change == 'linear':
            self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                            np.linspace(self._melt_f * self.melt_f_ratio,
                                                        self._melt_f, len(self.buckets)+1)
                                            ))
        elif self.melt_f_change == 'neg_exp':
            time = np.linspace(0, 6, len(self.buckets + ['ice']))
            melt_f_snow = self.melt_f_ratio * self._melt_f
            self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                            self._melt_f +
                                            (melt_f_snow - self._melt_f) * np.exp(-time/self.tau_e)
                                            ))

    def reset_pd_mb_bucket(self,
                           init_model_fls='use_inversion_flowline'):
        """ resets pandas mass balance bucket monthly and annual dataframe as well as the bucket dataframe
        (basically removes all years/months and empties the buckets so that everything is as if freshly instantiated)
        It is called when setting new melt_f, prcp_fac, temp_bias, ...
        """
        # comment: do I need to reset sometimes only the buckets,
        #  or only mb_monthly??? -> for the moment I don't think so
        if self.optim:
            self._pd_mb_template_bucket = pd.DataFrame(0, index=self.hbins[::-1],
                                                columns=self.columns)
            self._pd_mb_template_bucket.index.name = 'hbins_height'
            self._pd_mb_template = pd.DataFrame(0, index=self.hbins[::-1],
                                                columns=[])
            self._pd_mb_template.index.name = 'distance_along_flowline'

            self.pd_mb_monthly = self._pd_mb_template.copy()
            self.pd_mb_annual = self._pd_mb_template.copy()

            pd_bucket = self._pd_mb_template_bucket.copy()
            # pd_bucket[self.columns] = 0
            self.pd_bucket = pd_bucket
        else:
            if init_model_fls != 'use_inversion_flowline':
                #fls = gdir.read_pickle('model_flowlines')
                self.fl = copy.deepcopy(init_model_fls)[-1]
                self.mod_heights = self.fl.surface_h

                # container templates
                # use the distance_along_flowline as index
                self._pd_mb_template = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                                           columns=[])
                self._pd_mb_template.index.name = 'distance_along_flowline'
                self._pd_mb_template_bucket = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                                    columns=self.columns)
                self._pd_mb_template_bucket.index.name = 'distance_along_flowline'


            self.pd_mb_monthly = self._pd_mb_template.copy()
            self.pd_mb_annual = self._pd_mb_template.copy()
            self.pd_mb_daily = self._pd_mb_template.copy()

            pd_bucket = self._pd_mb_template_bucket.copy()
            # this expensive code line below  is not anymore necessary if we define columns directly when creating
            # self._pd_mb_template during instantiation!!! (took up to 15% of
            # pd_bucket[self.columns] = 0
            self.pd_bucket = pd_bucket

    def _add_delta_mb_vary_melt_f(self, heights, year=None,
                                  climate_resol='annual',
                                  add_climate=False):
        """ get the mass balance of all buckets and add/remove the
         new mass change for each bucket

         this is probably where most computational time is needed

         Parameters
         ----
         heights: np.array()
            heights of elevation flowline, at the monent only inversion height works!!!
            Only put heights inside that fit to "distance_along_flowline.
         year: int or float
            if melt_frequency='annual', integer calendar float year,
            if melt_frequency=='monthly', year should be a calendar float year,
            so it corresponds to 1 month of a specific year, i
         climate_resol: 'annual' or 'monthly
            if melt_frequency -> climate_resol has to be always monthly
            but if annual melt_frequency can have either annual climate_resol (if get_annual_mb is used)
            or monthly climate_resol (if get_monthly_mb is used)
         add_climate : bool
            default is False. If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output
            practical for get_monthly_mb with add_climate=True, as then we do not need to compute climate twice!!!

         """

        # new: get directly at the beginning the pd_bucket and convert it to np.array, at the end it is reconverted and
        # saved again under self.pd_bucket:
        np_pd_bucket = self.pd_bucket.values  # last column is delta_kg/m2

        if self.melt_frequency == 'monthly':
            if climate_resol != 'monthly':
                raise InvalidWorkflowError('Need monthly climate_resol if melt_frequency is monthly')
            if not isinstance(year, float):
                raise InvalidParamsError('Year has to be the calendar float year '
                                         '_add_delta_mb_vary_melt_f with monthly melt_frequency,'
                                         'year needs to be a float')

        #########
        # todo: if I put heights inside that are not fitting to
        #  distance_along_flowline, it can get problematic, I can only check it by testing if
        #  length of the heights are the same as distance along
        #  flowline of pd_bucket dataframe
        #  but the commented check is not feasible as it does not work for all circumstances
        #  however, we check at other points if the height array has at least the right order!
        # if len(heights) != len(self.fl.dis_on_line):
        #    raise InvalidParamsError('length of the heights should be the same as '
        #                             'distance along flowline of pd_bucket dataframe,'
        #                             'use for heights e.g. ...fl.surface_h()')
        ##########

        # check if the first bucket is be empty if:
        condi1 = climate_resol == 'annual' and self.melt_frequency == 'annual'
        # from the last year, all potential snow should be no firn, and from this year, the
        # new snow is not yet added, so snow buckets should be empty
        # or
        condi2 = self.melt_frequency == 'monthly'
        # from the last month, all potential snow should have been update and should be now
        # in the next older month bucket
        # or 1st month with annual update and get_monthly_mb
        # !!! in case of climate_resol =='monthly' but annual update,
        # first_snow_bucket is not empty if mc != 1!!!
        _, mc = floatyear_to_date(float(year))
        condi3 = climate_resol == 'monthly' and self.melt_frequency == 'annual' and mc == 1
        if condi1 or condi2 or condi3:
            # if not np.any(self.pd_bucket[self.first_snow_bucket] == 0):
            # comment exectime: this code here below is faster than the one from above
            np_pd_values_first_snow_bucket = np_pd_bucket[:, 0]  # -> corresponds to first_snow_bucket
            # todo exectime : this still takes long to evaluate !!! is there a betterway ???
            if len(np_pd_values_first_snow_bucket[np_pd_values_first_snow_bucket != 0]) > 0:
                raise InvalidWorkflowError('the first snow buckets should be empty in this use case '
                                           'but it is not, try e.g. to do '
                                           'reset_pd_mb_buckets() before and then rerun the task')

        # Let's do the same as in get_annual_mb of TIModel but with varying melt_f:
        # first get the climate
        if climate_resol == 'annual':
            t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
            if add_climate:
                raise NotImplementedError('Not implemented. Need to check here in the code'
                                      'that the dimensions are right. ')
            # t = t.mean(axis=1)
            # temp2dformelt = temp2dformelt.sum(axis=1)
            # prcp = prcp.sum(axis=1)
            # prcpsol = prcpsol.sum(axis=1)
        elif climate_resol == 'monthly':
            if isinstance(type(year), int):
                raise InvalidWorkflowError('year should be a float with monthly climate resolution')
            # year is here float year, so it corresponds to 1 month of a specific year
            t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year)

        # put first all solid precipitation into first snow bucket  (actually corresponds to first part of the loop)
        if climate_resol == 'annual':
            # if melt_frequency annual: treat snow as the amount of solid prcp over that year
            # first add all solid prcp amount to the bucket
            # (the amount of snow that has melted in the same year is taken away in the for loop)
            np_pd_bucket[:, 0] = prcpsol.sum(axis=1)  # first snow bucket should be filled with solid prcp
            # at first, remaining temp for melt energy is all temp for melt (tfm)
            # when looping over the buckets this term will get gradually smaller until the remaining tfm corresponds
            # to the potential ice melt
            remaining_tfm = temp2dformelt.sum(axis=1)
            # delta has to be set to the solid prcp (the melting part comes in during the for loop)
            # self.pd_bucket['delta_kg/m2']
            np_pd_bucket[:, -1] = prcpsol.sum(axis=1)

        elif climate_resol == 'monthly':
            # if self.melt_frequency is  'monthly': snow is the amount
            # of solid prcp over that month, ...
            # SLOW
            # self.pd_bucket[self.first_snow_bucket] = self.pd_bucket[self.first_snow_bucket].values.copy() + prcpsol.flatten() # .sum(axis=1)
            # faster
            np_pd_bucket[:, 0] = np_pd_bucket[:, 0] + prcpsol.flatten()
            remaining_tfm = temp2dformelt.flatten()
            # the last column corresponds to self.pd_bucket['delta_kg/m2'] -> np. is faster
            np_pd_bucket[:, -1] = prcpsol.flatten()

        # need the right unit
        if self.resolution == 'mb_real_daily':
            fact = 12/365.25
        else:
            fact = 1

        # now do the melting processes for each bucket in a loop:
        # how much tempformelt (tfm) would we need to remove all snow, firn of each bucket
        # in the case of snow it corresponds to tfm to melt all solid prcp.
        # To convert from kg/m2 in the buckets to tfm [K], we use the melt_f values
        # of each bucket accordingly. No need of .copy(), because directly changed by divisor (/)
        # this can actually be done outside the for-loop!!!

        # just get the melt_f_buckets once
        melt_f_buckets = self.melt_f_buckets
        # do not include delta_kg/m2 in pd_buckets, and also do not include melt_f of 'ice'
        tfm_to_melt_b = np_pd_bucket[:, :-1] / (np.array(list(melt_f_buckets.values()))[:-1]*fact)  # in K
        # need to transpose it to have the right shape
        tfm_to_melt_b = tfm_to_melt_b.T

        ### loop:
        # todo Fabi: exectime , can I use another kind of loop -> e.g. "apply" or sth.?
        #  maybe I can convert this first into a np.array then do the loop there and reconvert it at the end
        #  again into a pd.DataFrame for better handling?

        # faster: (old pandas code commented below)
        np_delta_kg_m2 = np_pd_bucket[:, -1]
        for e, b in enumerate(self.buckets):
            # there is no ice bucket !!!

            # now recompute buckets mass:
            # kg/m2 of that bucket that is not lost (that has not melted) in that year / month (i.e. what will
            # stay in that bucket after that month/year -> and gets later on eventually updated -> older)
            # if all lost -> set it to 0
            # -> to get this need to reconvert the tfm energy unit into kg/m2 by using the right melt_factor
            # e.g. at the uppest layers there is new snow added ...
            # todo Fabi: exectime now one of slowest lines (e.g. ~8% of computational time)
            not_lost_bucket = clip_min(tfm_to_melt_b[e] - remaining_tfm, 0) * melt_f_buckets[b] * fact

            # if all of the bucket is removed (not_lost_bucket=0), how much energy (i.e. tfm) is left
            # to remove mass from other buckets?
            # remaining tfm to melt older firn layers -> for the next loop ...
            remaining_tfm = clip_min(remaining_tfm - tfm_to_melt_b[e], 0)
            # exectime: is this faster?
            # find sth. faster

            # in case of ice, the remaining_tfm is only used to update once again delta_kg/m2
            # todo : exectime time Fabi check if I can merge not_lost_bucket and remaining_tfm computation
            #  (so that we don't do the clip_min twice)?
            #  or maybe I should only compute that if not_lost_bucket == 0:

            # amount of kg/m2 lost in this bucket -> this will be added to delta_kg/m2
            # corresponds to: not yet updated total bucket - amount of not lost mass of that bucket
            # comment: the line below was very expensive (Number 1, e.g. 22% of computing time)-> rewrite to numpy?
            # .copy(), not necessary because minus
            # self.pd_bucket['delta_kg/m2'] += not_lost_bucket - self.pd_bucket[b].values
            # comment:  now faster
            # np_delta_kg_m2 = (np_delta_kg_m2 + not_lost_bucket - self.pd_bucket[b].values)
            # but we want it even faster: removed all panda stuff
            np_delta_kg_m2 = (np_delta_kg_m2 + not_lost_bucket - np_pd_bucket[:, e])

            # update pd_bucket with what has not melted from the bucket
            # how can we make this faster?
            # self.pd_bucket[b] = not_lost_bucket
            # new: removed all panda stuff
            # exectime: not so slow anymore (e.g. 3% of computational time)
            np_pd_bucket[:, e] = not_lost_bucket

            # new since autumn 2021: if all the remaining_tfm is zero, then go out of the loop,
            # to make the code faster!
            # e.g. in winter this should mean that we don't loop over all buckets (as tfm is very small)
            # if np.all(remaining_tfm == 0):
            # is this faster?
            if len(remaining_tfm[remaining_tfm != 0]) == 0:
                break

        # we assume that the ice bucket is infinite,
        # so everything that could be melted is included inside of delta_kg/m2
        # that means all the remaining tfm energy is used to melt the infinite ice bucket
        # self.pd_bucket['delta_kg/m2'] = np_delta_kg_m2 -remaining_tfm * self.melt_f_buckets['ice'] * fact
        # use the np.array and recompute delta_kg_m2
        np_pd_bucket[:, -1] = np_delta_kg_m2 -remaining_tfm * melt_f_buckets['ice'] * fact
        # create pd_bucket again at the end
        # todo Fabi: exectime -> this is also now quite time expensive (~8-13%)
        #  but on the other hand, it takes a lot of time to
        #  create a bucket system without pandas at all !
        #  if we would always stay in the self.pd_bucket.values system instead of np_pd_bucket,
        #  we would not need to recreate the pd_bucket, maybe this would be then faster?
        self.pd_bucket = pd.DataFrame(np_pd_bucket, columns=self.pd_bucket.columns.values,
                                      index=self.pd_bucket.index)
        if add_climate:
            return self.pd_bucket, t, temp2dformelt, prcp, prcpsol
        else:
            return self.pd_bucket

        # old:
        # loop_via_pd = False
        # if loop_via_pd:
        #     for e, b in enumerate(self.buckets):
        #         # there is no ice bucket !!!
        #
        #         # now recompute buckets mass:
        #         # kg/m2 of that bucket that is not lost (that has not melted) in that year / month (i.e. what will
        #         # stay in that bucket after that month/year -> and gets later on eventually updated ->older)
        #         # if all lost -> set it to 0
        #         # -> to get this need to reconvert the tfm energy unit into kg/m2 by using the right melt_factor
        #         # e.g. at the uppest layers there is new snow added ...
        #         not_lost_bucket = utils.clip_min(tfm_to_melt_b[e] - remaining_tfm, 0) * self.melt_f_buckets[b] * fact
        #
        #         # if all of the bucket is removed (not_lost_bucket=0), how much energy (i.e. tfm) is left
        #         # to remove mass from other buckets?
        #         # remaining tfm to melt older firn layers -> for the next loop ...
        #         remaining_tfm = utils.clip_min(remaining_tfm - tfm_to_melt_b[e], 0)
        #         # exectime: is this faster?
        #         # find sth. faster
        #
        #         # in case of ice, the remaining_tfm is only used to update once again delta_kg/m2
        #
        #         # amount of kg/m2 lost in this bucket -> this will be added to delta_kg/m2
        #         # corresponds to: not yet updated total bucket - amount of not lost mass of that bucket
        #         # comment: this line was very expensive (Number 1, e.g. 22% of computing time)-> rewrite to numpy?
        #         # self.pd_bucket['delta_kg/m2'] += not_lost_bucket - self.pd_bucket[b].values # .copy(), not necessary because minus
        #         self.pd_bucket['delta_kg/m2'] = (self.pd_bucket['delta_kg/m2'].values
        #                                          + not_lost_bucket - self.pd_bucket[
        #                                              b].values)  # .copy(), not necessary because minus
        #
        #         # is this faster:
        #         # self.pd_bucket['delta_kg/m2'] = not_lost_bucket - self.pd_bucket[b].values # .copy(), not necessary because minus
        #
        #         # update pd_bucket with what has not melted from the bucket
        #         # comment: exectime this line was very expensive (Number 2, e.g. 11% of computing time)->
        #         # how can we make this faster?
        #         self.pd_bucket[b] = not_lost_bucket
        #
        #         # new since autumn 2021: if all the remaining_tfm is zero, then go out of the loop,
        #         # to make the code faster!
        #         # e.g. in winter this should mean that we don't loop over all buckets (as tfm is very small)
        #         if np.all(remaining_tfm == 0):
        #             break
        #     # we assume that the ice bucket is infinite,
        #     # so everything that could be melted is included inside of delta_kg/m2
        #     # that means all the remaining tfm energy is used to melt the infinite ice bucket
        #     self.pd_bucket['delta_kg/m2'] += - remaining_tfm * self.melt_f_buckets['ice'] * fact




    # comment: should this be a setter ??? because no argument ...
    # @update_buckets.setter
    def _update(self):
        """ this is called by get_annual_mb or get_monthly_mb after one year/one month to update
        the buckets as they got older

        if it is called on monthly or annual basis depends if we set melt_frequency to monthly or to annual!


        new: I removed a loop here because I can just rename the bucket column names in order that they are
        one bucket older, then set the first_snow_bucket to 0, and the set pd_bucket[kg/m2] to np.nan
        just want to shift all buckets from one column to another one .> can do all at once via self.buckets[::-1].iloc[1:] ???
        """
        # first convert pd_bucket to np dataframe -> at the end reconvert to pd.DataFrame
        np_pd_bucket = self.pd_bucket.values

        # this here took sometimes 1.1%
        # if np.any(np.isnan(self.pd_bucket['delta_kg/m2'])):
        # now faster (~0.2%)
        if np.any(np.isnan(np_pd_bucket[:, -1])): # delta_kg/m2 is in the last column !!!
            raise InvalidWorkflowError('the buckets have been updated already, need'
                                       'to add_delta_mb first')

        # exectime: this here took sometimes 7.5% of total computing time !!!
        # if np.any(self.pd_bucket[self.buckets] < 0):
        #    raise ValueError('the buckets should only have positive values')
        # a bit faster now: but can be still ~5%
        # if self.pd_bucket[self.buckets].values.min() < 0:
        # np_pd_values = self.pd_bucket[self.buckets].values
        # buckets are in all columns except the last column!!!
        if len(np_pd_bucket[:, :-1][np_pd_bucket[:, :-1] < 0]) > 0:
            raise ValueError('the buckets should only have positive values')

        # old snow without last bucket and without delta_kg/ms
        # all buckets to update: np_pd_bucket[1:-1]
        # kg/m2 in those buckets before the update: self.bucket[0:-2]
        pd_bucket_val_old = np_pd_bucket[:, :-2]
        len_h = len(self.pd_bucket.index)
        # exectime: <0.1%
        np_updated_bucket = np.concatenate([np.zeros(len_h).reshape(len_h, 1),  # first bucket should be zero
                                            pd_bucket_val_old,
                                            np.full((len_h, 1), np.nan)],  # kg/m2 bucket should be np.nan
                                            axis=1)  # , np.nan(len(self.pd_bucket.index))])
        # as we add a zero array in the front, we don't have any snow more, what has been in the youngest bucket
        # went into the next older bucket and so on!!!

        # recreate pd_bucket:
        # todo Fabi exectime: if we want to save more time would need to restructure all -> slowest line
        #  (9-12% computational time)
        #  if we would always stay in the self.pd_bucket.values system instead of np_pd_bucket,
        #  we would not need to recreate the pd_bucket, maybe this would be then faster?
        self.pd_bucket = pd.DataFrame(np_updated_bucket, columns=self.pd_bucket.columns.values,
                                      index=self.pd_bucket.index)
        return self.pd_bucket

        # OLD stuff
        # other idea -> but did not work!
        # just rename the columns : and add afterwards the snow bucket inside
        # self.pd_bucket = self.pd_bucket.drop(columns=['delta_kg/m2', self.buckets[-1]])
        # problem here: need to have it in the right order (first snow, then firn buckets -> otherwise tests fail)
        # self.pd_bucket.columns = self.buckets[1:]

        # pd_version of above -> was still too slow!
        # exectime: this line was very expensive (sometimes even number 1 with 40% of computing time)
        # self.pd_bucket[self.buckets[1:]] = self.pd_bucket[self.buckets[:-1]].values

        # self.pd_bucket[self.first_snow_bucket] = 0
        # self.pd_bucket['delta_kg/m2'] = np.nan

        # the new version should work as the old version (e.g. this test here should check it: test_sfc_type_update)
        # the other loop version will be removed
        # else:
        #     for e, b in enumerate(self.buckets[::-1]):
        #         # start with updating oldest snow bucket (in reversed order!= ...
        #         if b != self.first_snow_bucket:
        #             # e.g. update ice by using old firn_yr_5 ...
        #             # @Fabi: do I need the copy here?
        #             self.pd_bucket[b] = self.pd_bucket[self.buckets[::-1][e + 1]].copy()
        #             # we just overwrite it so we don't need to reset it to zero
        #             # self.pd_bucket[self.buckets[::-1][e+1]] = 0 #pd_bucket['firn_yr_4']
        #         elif b == self.first_snow_bucket:
        #             # the first_snow bucket is set to 0 after the update
        #             # (if annual update there is only one snow bucket)
        #             self.pd_bucket[b] = 0
        #     # reset delta_kg/m2 to make clear that it is updated
        #     self.pd_bucket['delta_kg/m2'] = np.nan

    def get_annual_mb(self, heights, year=None, unit='m_of_ice',
                      bucket_output=False, spinup=True,
                      add_climate=False,
                      auto_spinup=True,
                      **kwargs):
        """
        computes annual mass balance in m of ice per second

        Parameters
        ----------
        heights : np.array
            at the moment works only with inversion heights!
        year: int
            integer CALENDAR year! if melt_frequency='monthly', it will loop over each month.
        unit : str
            default is 'm of ice', nothing else implemented at the moment!
            comment: include option of metre of glacier where the different densities
             are taken into account but in this case would need to add further columns in pd_buckets
             like: snow_delta_kg/m2 ... and so on (only important if I add refreezing or a densification scheme)
        bucket_output: bool (default is False)
            if True, returns as second output the pd.Dataframe with details about
            amount of kg/m2 for each bucket and height grid point,
            set it to True to visualize how the buckets change over time or for testing
            (these are the buckets before they got updated for the next year!)
        spinup : bool (default is True)
            if a spinup is applied to fill up sfc type buckets beforehand
            (default are 6 years, check under self.spinup_years)
        add_climate: bool (default is False)
            for run_with_hydro (not yet implemented!)
            todo: implement and test it
        auto_spinup: bool (default is true)
            if True, it automatically computes the spinup years (default is 6) beforehand (however in this case,
            these 6 years at the beginning, although saved in pd_mb_annual, had no spinup...)
            todo: maybe need to add a '_no_spinup' to those that had no spinup?
             or save them in a separate pd.DataFrame?
        **kwargs:
            other stuff passed to get_monthly_mb or to _add_delta_mb_vary_melt_f
            **kwargs necessary to take stuff we don't use (like fls...)

        """

        # when we set spinup_years to zero, then there should be no spinup occurring, even if spinup
        # is set to True
        if self.spinup_years == 0:
            spinup = False

        if len(self.pd_mb_annual.columns) > 0:
            if self.pd_mb_annual.columns[0] > self.pd_mb_annual.columns[-1]:
                raise InvalidWorkflowError('need to run years in ascending order! Maybe reset first!')
        # dirty check that the given heights are in the right order
        assert heights[0] > heights[-1], "heights should be in descending order!"
        # otherwise pd_buckets does not work ...
        # below would be a more robust test, but this is not compatible to all use cases:
        # np.testing.assert_allclose(heights, self.inv_heights,
        #                           err_msg='the heights should correspond to the inversion heights',
        #                           )
        # np.testing.assert_allclose(heights, self.mod_heights)

        # we just convert to integer without checking ...
        year = int(year)
        # comment: there was some reason why I commented that code again
        # if not isinstance(year, int):
        #    raise InvalidParamsError('Year has to be the full year for get_annual_mb,'
        #                             'year needs to be an integer')
        if year < 1979+self.spinup_years:
            # most climatic data starts in 1979, so if we want to
            # get the first 6 years can not use a spinup!!!
            # (or would need to think about sth. else, but not so
            # important right now!!!)
            spinup = False
        if year in self.pd_mb_annual.columns and self.check_data_exists:
            # print('takes existing annual mb')
            # if that year has already been computed, and we did not change any parameter settings
            # just get the annual_mb without redoing all the computations
            mb_annual = self.pd_mb_annual[year].values
            if bucket_output:
                raise InvalidWorkflowError('if you want to output the buckets, you need to do'
                                           'reset_pd_mb_bucket() and rerun')

            if add_climate:
                t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights,
                                                                              year)
                return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                        prcp.sum(axis=1), prcpsol.sum(axis=1))
                # raise NotImplementedError('TODO: add_climate has to be implemented!')
            else:
                return mb_annual
        else:
            # do we need to do the spinup beforehand
            # if any of the default 6 spinup years before was not computed,
            # we need to reset all and do the spinup properly -> (i.e. condi = True)
            if self.melt_frequency == 'annual':
                # so we really need to check if every year exists!!!
                condis = []
                for bef in np.arange(1, self.spinup_years, 1):
                    condis.append(int(year - bef) not in self.pd_mb_annual.columns)
                condi = np.any(np.array(condis))
            elif self.melt_frequency == 'monthly':
                try:
                    # check if first and last month of each spinup years before exists
                    condis = []
                    for bef in np.arange(1, self.spinup_years, 1):
                        condis.append(date_to_floatyear(year - bef, 1) not in self.pd_mb_monthly.columns)
                        condis.append(date_to_floatyear(year - bef, 12) not in self.pd_mb_monthly.columns)
                    condi = np.any(np.array(condis))
                except:
                    condi = True

            if condi and spinup and auto_spinup:
                # reset and do the spinup:
                # comment: I think we should always reset when
                # doing the spinup and not having computed year==2000
                # problem: the years before year should not be saved up (in get_annual_mb)!
                # (because they are not computed right!!! (they don't have a spinup)
                self.reset_pd_mb_bucket()
                for yr in np.arange(year-self.spinup_years, year):
                    self.get_annual_mb(heights, year=yr, unit=unit,
                                       bucket_output=False,
                                       spinup=False, add_climate=False,
                                       auto_spinup=False,
                                       **kwargs)

            if spinup:
                # check if the spinup years had been computed (should be inside of pd_mb_annual)
                # (it should have been computed automatically if auto_spinup=True)
                for bef in np.arange(1, 6, 1):
                    if int(year-bef) not in self.pd_mb_annual.columns.values:
                        raise InvalidWorkflowError('need to do get_annual_mb of all spinup years'
                                                   '(default is 6) beforehand')
            if self.melt_frequency == 'annual':
                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights, year=year,
                                                                climate_resol='annual')
                mb_annual = ((self.pd_bucket['delta_kg/m2'].values
                              - self.bias) / SEC_IN_YEAR / self.rho)
                if bucket_output:
                    # copy because we want to output the bucket that is not yet updated!!!
                    pd_bucket = self.pd_bucket.copy()
                # update to one year later ... (i.e. put the snow / firn into the next older bucket)
                self._update()
                # save the annual mb
                # todo Fabi exectime: --> would need to restructure code to remove pd stuff
                self.pd_mb_annual[year] = mb_annual
                # this is done already if melt_frequency is monthly (see get_monthly_mb for m == 12)
            elif self.melt_frequency == 'monthly':
                # will be summed up over each month by getting pd_mb_annual
                # that is set in december (month=12)
                for m in np.arange(1, 13, 1):
                    floatyear = date_to_floatyear(year, m)
                    out = self.get_monthly_mb(heights, year=floatyear, unit=unit,
                                              bucket_output=bucket_output, spinup=spinup,
                                              add_climate=add_climate,
                                              auto_spinup=auto_spinup,
                                              **kwargs)
                    if bucket_output and m == 12:
                        pd_bucket = out[1]
                # get mb_annual that is produced by
                mb_annual = self.pd_mb_annual[year].values
            if bucket_output:
                return mb_annual, pd_bucket
            else:
                if add_climate:
                    t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights,
                                                                                  year)
                    return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                            prcp.sum(axis=1), prcpsol.sum(axis=1))
                    #raise NotImplementedError('TODO: add_climate has to be implemented!')
                else:
                    return mb_annual

            #todo
            # if add_climate:
            #    return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
            #            prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None, unit='m_of_ice',
                       bucket_output=False, spinup=True,
                       add_climate=False,
                       auto_spinup=True,
                       **kwargs):
        """
        computes monthly mass balance in m of ice per second!

        year should be the calendar float year,
        so it corresponds to 1 month of a specific year

        Parameters
        ----------
        heights : np.array
            at the moment only works with inversion heights!
        year: int
            year has to be given as CALENDAR float year from what the (integer) year and month is taken,
            hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        unit : str
            default is 'm of ice', nothing else implemented at the moment!
            TODO: include option of metre of glacier where the different densities
             are taken into account but in this case would need to add further columns in pd_buckets
             like: snow_delta_kg/m2 ... (only important if I add refreezing or a densification scheme)
        bucket_output: bool (default is False)
            if True, returns as second output the pd.Dataframe with details about
            amount of kg/m2 for each bucket and height grid point,
            set it to True to visualize how the buckets change over time or for testing
            (these are the buckets before they got updated for the next year!)
        spinup : bool (default is True)
            if a spinup is applied to fill up sfc type buckets beforehand
            (default are 6 years, check under self.spinup_years), if month is not January,
            also checks if the preceding monts of thar year have been computed
        add_climate: bool (default is False)
            for run_with_hydro or for get_specific_winter_mb to get winter precipitation
            If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output.
            prcp and temp_for_melt as monthly sum, temp. as mean.
        auto_spinup: bool (default is true)
            if True, it automatically computes the spinup years (default is 6) beforehand and
            all months before the existing year
            (however in this case, these spinup years, although saved in pd_mb_annual, had no spinup...)
            todo: maybe need to add a '_no_spinup' to those that had no spinup?
             or save them in a separate pd.DataFrame?
        **kwargs:
            other stuff passed to get_annual_mb or to _add_delta_mb_vary_melt_f
            todo: not yet necessary, maybe later? **kwargs necessary to take stuff we don't use (like fls...)
        """

        if self.spinup_years == 0:
            spinup = False
        if year < 1979+self.spinup_years:
            # comment: most climatic data starts in 1979, so if we want to
            # get the first 6 years can not use a spinup!!!
            # (or would need to think about sth. else, but not so
            # important right now!!!)
            spinup = False

        if len(self.pd_mb_monthly.columns) > 1:
            if self.pd_mb_monthly.columns[0] > self.pd_mb_monthly.columns[-1]:
                raise InvalidWorkflowError('need to run months in ascending order! Maybe reset first!')

        assert heights[0] > heights[-1], "heights should be in descending order!"
        # below would be a more robust test, but this is not compatible to all use cases:
        # np.testing.assert_allclose(heights, self.inv_heights,
        #                           err_msg='the heights should correspond to the inversion heights',
        #                           )

        if year in self.pd_mb_monthly.columns and self.check_data_exists:
            # if that float year has already been computed
            # just get the saved monthly_mb
            # (but don't need to compute something or update pd_buckets)
            # print('takes existing monthly mb')
            mb_month = self.pd_mb_monthly[year].values
            if bucket_output:
                raise InvalidWorkflowError('if you want to output the buckets, you need to do'
                                           'reset_pd_mb_bucket() and rerun')
            if add_climate:
                if isinstance(type(year), int):
                    raise InvalidWorkflowError('year should be a float with monthly climate resolution')
                # year is here float year, so it corresponds to 1 month of a specific year
                t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year)
                if self.resolution == 'mb_pseudo_daily':
                    temp2dformelt = temp2dformelt.flatten()
                return mb_month, t, temp2dformelt, prcp, prcpsol
            else:
                return mb_month
        else:
            # only allow float years
            if not isinstance(year, float):
                raise InvalidParamsError('Year has to be the calendar float year '
                                         'for get_monthly_mb,'
                                         'year needs to be a float')
            # need to somehow check if the months before that were computed as well
            # otherwise monthly mass-balance does not make sense when using sfc type distinction ...
            y, m = floatyear_to_date(year)

            ### spinup
            # find out if the spinup has to be computed
            # this is independent of melt_frequency, always need to reset if the spinup years before
            # were not computed
            try:
                # it is sufficient to check if the first month of that year is there
                # if there is any other problem it will raise an error later anyways
                condi = date_to_floatyear(y - self.spinup_years, 1) not in self.pd_mb_monthly.columns
            except:
                condi = True
            # if annual melt_frequency and annual_mb exist from previous years
            # (by doing , get_annual_mb), and there exist not yet the mass-balance for the next year
            # then no reset & new spinup is necessary
            # comment (Feb2022): why did I have here before:
            # condi_add = int(y + 1) in self.pd_mb_annual.columns
            # NEW: don't reset if the years before already exist (by checking pd_mb_annual)
            if condi:
                try:
                    if np.any(self.pd_mb_annual.columns[-self.spinup_years:] == np.arange(y - self.spinup_years, y, 1)):
                        no_spinup_necessary = True
                    else:
                        no_spinup_necessary = False
                except:
                    no_spinup_necessary = False
                if self.melt_frequency == 'annual' and no_spinup_necessary:
                    condi = False

            if condi and spinup and auto_spinup:
                # reset and do the spinup:
                self.reset_pd_mb_bucket()
                for yr in np.arange(y-self.spinup_years, y):
                    self.get_annual_mb(heights, year=yr, unit=unit, bucket_output=False,
                                       spinup=False, add_climate=False,
                                       auto_spinup=False, **kwargs)
                # need to also run the months before our actual wanted month m
                if m > 1:
                    for mb in np.arange(1, m, 1):
                        floatyear_before = date_to_floatyear(y, mb)
                        self.get_monthly_mb(heights, year=floatyear_before,
                                            bucket_output=False,
                                            spinup=False, add_climate=False,
                                            auto_spinup=False, **kwargs)

            if spinup:
                # check if the years before that had been computed
                # (if 2000, it should have been computed above)
                for bef in np.arange(1, 6, 1):
                    if int(y-bef) not in self.pd_mb_annual.columns:
                        raise InvalidWorkflowError('need to do get_monthly_mb of all spinup years '
                                                   '(default is 6) beforehand')
                # check if the months before had been computed for that year
                for mb in np.arange(1, m, 1):
                   floatyear_before = date_to_floatyear(y, mb)
                   if floatyear_before not in self.pd_mb_monthly.columns:
                       raise InvalidWorkflowError('need to get_monthly_mb of each month '
                                                  'of that year beforehand')

            # year is here in float year!
            if add_climate:
                self.pd_bucket, t, temp2dformelt, prcp, prcpsol = self._add_delta_mb_vary_melt_f(heights,
                                                                year=year,
                                                                climate_resol='monthly',
                                                                add_climate=add_climate)
            else:
                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights,
                                                                year=year,
                                                                climate_resol='monthly',
                                                                add_climate=add_climate)

            # ?Fabi: if I want to have it in two lines, can I then  still put the .copy() away?
            #mb_month = self.pd_bucket['delta_kg/m2'].copy().values
            #mb_month -= self.bias * self.SEC_IN_MONTH / self.SEC_IN_YEAR
            mb_month = self.pd_bucket['delta_kg/m2'].values - \
                       self.bias * SEC_IN_MONTH / SEC_IN_YEAR
            # need to flatten for mb_pseudo_daily otherwise it gives the wrong shape
            mb_month = mb_month.flatten() / SEC_IN_MONTH / self.rho
            # save the mass-balance to pd_mb_monthly
            # todo Fabi exectime: -> this line is also quite expensive -> to make it faster would need
            #  to entirely remove the pd_structure!
            self.pd_mb_monthly[year] = mb_month
            self.pd_mb_monthly[year] = mb_month

            # the update happens differently fot the two cases -> so we need to differentiate here:
            if self.melt_frequency == 'annual':
                warnings.warn('get_monthly_mb with annual melt_frequency results in a different summed up annual MB'
                              'than using get_annual_mb. This is because we use bulk estimates in get_annual_mb.'
                              'Only use it when you know what you do!!! ')
                # todo: maybe better to set a NotImplementedError
                # raise NotImplementedError('get_monthly_mb works at the moment '
                #                          'only when melt_f is updated monthly')

                # if annual, only want to update at the end of the year
                if m == 12:
                    # sum up the monthly mb and update the annual pd_mb
                    # but as the mb is  m of ice per second -> use the mean !!!
                    # only works if same height ... and if
                    if bucket_output:
                        pd_bucket = self.pd_bucket.copy()
                    self._update()
                    if int(year) not in self.pd_mb_annual.columns:
                        condi = [int(c) == int(year) for c in self.pd_mb_monthly.columns]
                        # first check if we have all 12 months of the year together
                        if len(self.pd_mb_monthly.loc[:, condi].columns) != 12:
                            raise InvalidWorkflowError('Not all months were computed beforehand,'
                                                       'need to do get_monthly_mb for all months before')
                        # get all columns that correspond to that year and do the mean to get the annual estimate
                        # mean because unit is in m of ice per second
                        self.pd_mb_annual[int(year)] = self.pd_mb_monthly.loc[:, condi].mean(axis=1).values

                if bucket_output and m != 12:
                    pd_bucket = self.pd_bucket.copy()

            elif self.melt_frequency == 'monthly':

                # get the output before the update
                if bucket_output:
                    pd_bucket = self.pd_bucket.copy()
                # need to update it after each month
                self._update()
                # if December, also want to save it to the annual mb
                if m == 12:
                    # sum up the monthly mb and update the annual pd_mb
                    # run this if this year not yet inside or if we don't want to check availability
                    # e.g. for random / constant runs !!!
                    if int(year) not in self.pd_mb_annual.columns or not self.check_data_exists:
                        condi = [int(c) == int(year) for c in self.pd_mb_monthly.columns]
                        # first check if we have all 12 months of the year together
                        if len(self.pd_mb_monthly.loc[:, condi].columns) != 12:
                            raise InvalidWorkflowError('Not all months were computed beforehand,'
                                                       'need to do get_monthly_mb for all months before')
                        # get all columns that correspond to that year and do the mean to get the annual estimate
                        # mean because unit is in m of ice per second
                        self.pd_mb_annual[int(year)] = self.pd_mb_monthly.loc[:, condi].mean(axis=1).values

            if add_climate and not bucket_output:
                if self.resolution == 'mb_pseudo_daily':
                    temp2dformelt = temp2dformelt.flatten()
                return mb_month, t, temp2dformelt, prcp, prcpsol
            elif bucket_output:
                return mb_month, pd_bucket
            elif add_climate and bucket_output:
                raise InvalidWorkflowError('either set bucket_output or add_climate to True, not both. '
                                           'Otherwise you have to change sth. in the code!')
            else:
                return mb_month

    # def get_daily_mb(self, heights, year=None, unit='m_of_ice',
    #                   bucket_output=False, spinup=True,
    #                   add_climate=False,
    #                   auto_spinup=True,
    #                   **kwargs):
    #     """computes daily mass balance in m of ice per second

    #     attention this accounts as well for leap years, hence
    #     doy are not 365.25 as in get_annual_mb but the amount of days the year
    #     has in reality!!! (needed for hydro model of Sarah Hanus)

    #     year has to be given as float hydro year from what the month is taken,
    #     hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
    #     which corresponds to the real year 1999 and months October or November
    #     if hydro year starts in October

    #     (implemented in order that Sarah Hanus can use daily input and also gets daily output)
    #     """

    #     if len(self.pd_mb_annual.columns) > 0:
    #         if self.pd_mb_annual.columns[0] > self.pd_mb_annual.columns[-1]:
    #             raise InvalidWorkflowError('Run years in ascending order! Maybe reset first!')
        
    #     if heights[0] > heights[-1]:
    #         raise InvalidWorkflowError("heights should be in descending order!")
        
    #     year = int(year)

    #     if not self.spinup_years or year < 1979 + self.spinup_years:
    #         spinup = False

    #     if year in self.pd_mb_annual.columns and self.check_data_exists:
    #         mb_annual = self.pd_mb_annual[year].values
    #         if bucket_output:
    #             raise InvalidWorkflowError(
    #                 f"if you want to output the buckets, you need to do"
    #                 f"reset_pd_mb_bucket() and rerun"
    #             )

    #         if add_climate:
    #             t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(
    #                 heights, year)
    #             return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
    #                     prcp.sum(axis=1), prcpsol.sum(axis=1))
    #         else:
    #             return mb_annual
    #     else:
    #         if self.melt_frequency == 'annual':
    #             # so we really need to check if every year exists!!!
    #             condis = []
    #             for bef in np.arange(1, self.spinup_years, 1):
    #                 condis.append(int(year - bef) not in self.pd_mb_annual.columns)
    #             mask = np.any(np.array(condis))
    #         elif self.melt_frequency == 'monthly':
    #             try:
    #                 # check if first and last month of each spinup years before exists
    #                 condis = []
    #                 for bef in np.arange(1, self.spinup_years, 1):
    #                     condis.append(date_to_floatyear(year - bef, 1) not in self.pd_mb_monthly.columns)
    #                     condis.append(date_to_floatyear(year - bef, 12) not in self.pd_mb_monthly.columns)
    #                 mask = np.any(np.array(condis))
    #             except:
    #                 mask = True
    #         if mask and spinup and auto_spinup:
    #             self.reset_pd_mb_bucket()
    #             for yr in np.arange(year-self.spinup_years, year):
    #                 self.get_daily_mb(
    #                     heights,
    #                     year=yr,
    #                     unit=unit,
    #                     bucket_output=False,
    #                     spinup=False,
    #                     add_climate=False,
    #                     auto_spinup=False,
    #                     **kwargs,
    #                 )
    #         if spinup:
    #             for bef in np.arange(1, 6, 1):
    #                 if int(year-bef) not in self.pd_mb_annual.columns.values:
    #                     raise InvalidWorkflowError(
    #                         'need to do get_annual_mb() of all spinup years'
    #                         '(default is 6) beforehand')
    #         if self.melt_frequency == "annual":
    #             self.pd_bucket = self._add_delta_mb_vary_melt_f(
    #                 heights, year=year, climate_resol="annual"
    #             )
            
        
        
        
    def get_daily_mb(self, heights, year=None, add_climate=False, **kwargs):
        # return super().get_daily_mb(heights=heights, year=year, add_climate=add_climate)
        # todo: make this more user friendly
        if type(year) == float:
            raise InvalidParamsError('here year has to be the integer year')
        else:
            pass

        if self.resolution == 'mb_real_daily':
            # get 2D values, dependencies on height and time (days)
            out = self._get_2d_daily_climate(heights, year)
            t, temp2dformelt, prcp, prcpsol = out
            # days of year
            doy = len(prcpsol.T)  # 365.25
            # assert doy > 360
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            upscale_factor = prcpsol.shape[1] / 365
            melt_f_daily = self.melt_f * upscale_factor
            # melt_f_daily = self.melt_f * 12/doy
            mb_daily = prcpsol - melt_f_daily * temp2dformelt

            # mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            warnings.warn('be cautious when using get_daily_mb and test yourself if it does '
                          'what you expect')

            # residual is in mm w.e per year, so SEC_IN_MONTH .. but mb_daily
            # is per day!
            # mb_daily -= self.bias * doy
            mb_daily = (
            (mb_daily - self.bias * doy) / SEC_IN_DAY / self.rho
            )
            # this is for mb_daily otherwise it gives the wrong shape
            # mb_daily = mb_month.flatten()
            # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth
            if add_climate:
                # these are here daily values as output for the entire year
                # might need to be changed a bit to be used for run_with_hydro
                return (mb_daily,
                        t, temp2dformelt, prcp, prcpsol)
            return mb_daily
        else:
            raise InvalidParamsError('get_daily_mb works only with'
                                     'mb_real_daily as resolution!')

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None, **kwargs):
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
                mbs = self.get_annual_mb(heights=heights, year=mb_yr, **kwargs)
                mbs = weighted_average_1d(mbs, widths)
            stack.append(mbs)

        smb = set_array_type(stack) * SEC_IN_YEAR * self.rho
        mask = np.vectorize(calendar.isleap)(year)  # strangely faster than map
        smb = np.where(mask, smb * 366 / 365, smb)

        return smb

    def get_specific_mb_daily(self, heights=None, widths=None, fls=None, year=None):
        """ returns specific daily mass balance in kg m-2 day

        (implemented in order that Sarah Hanus can use daily input and also gets daily output)
        """
        if len(np.atleast_1d(year)) > 1:
            stack = []
            for yr in year:
                out = self.get_specific_mb_daily(heights=heights, widths=widths, year=yr)
                stack = np.append(stack, out)
            return np.asarray(stack)

        mb = self.get_daily_mb(heights, year=year)
        spec_mb = np.average(mb * self.rho * SEC_IN_DAY, weights=widths, axis=0)
        assert len(spec_mb) > 360
        return spec_mb

    def get_ela(self, year=None, **kwargs):
        """Compute the equilibrium line altitude for a given year.

        copied and adapted from OGGM main -> had to remove the invalid ELA check, as it won't work
        together with sfc type distinction. It still does not work, as the buckets are not made to be applied just
        that


        Parameters
        ----------
        year: float, optional
            the time (in the "hydrological floating year" convention)
        **kwargs: any other keyword argument accepted by self.get_annual_mb
        Returns
        -------
        the equilibrium line altitude (ELA, units: m)
        """
        if len(np.atleast_1d(year)) > 1:
            return np.asarray([self.get_ela(year=yr, **kwargs) for yr in year])

        if self.valid_bounds is None:
            raise ValueError('attribute `valid_bounds` needs to be '
                             'set for the ELA computation.')

        # Check for invalid ELAs
        #b0, b1 = self.valid_bounds
        #if (np.any(~np.isfinite(
        #        self.get_annual_mb([b0, b1], year=year, **kwargs))) or
        #        (self.get_annual_mb([b0], year=year, **kwargs)[0] > 0) or
        #        (self.get_annual_mb([b1], year=year, **kwargs)[0] < 0)):
        #    return np.nan

        def to_minimize(x):
            return (self.get_annual_mb([x], year=year, **kwargs)[0] *
                    SEC_IN_YEAR * self.rho)

        return optimize.brentq(to_minimize, *self.valid_bounds, xtol=0.1)

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

    def get_daily_mb(self, heights, year=None, fl_id=None, **kwargs):
        if fl_id is None:
            raise ValueError("`fl_id` is required for MultipleFlowlineMassBalance.")

        return self.flowline_mb_models[fl_id].get_daily_mb(heights, year=year, **kwargs)

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
            year_length = mb_mod.get_year_length(year)
            # Multiplying rho by weighted avg changes the results
            mbs.append(mb * year_length * mb_mod.rho)
        widths = np.concatenate(widths, axis=0)  # 2x faster than np.append
        mbs = np.concatenate(mbs, axis=0)
        mbs = weighted_average_1d(mbs, widths)

        return mbs
    
    def get_monthly_specific_mass_balance(self, fls: list, year: float) -> float:
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
            mb = mb_mod.get_monthly_mb(fl.surface_h, year=year, fls=fls, fl_id=i)
            # Multiplying rho by weighted avg changes the results
            mbs.append(mb * cfg.SEC_IN_MONTH * mb_mod.rho)
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

    def get_specific_mb_daily(
        self, heights=None, widths=None, fls=None, year=None
    ):
        """Get specific daily mass balance.

        TODO: Returns identical annual SMB to ``get_specific_mb`` when
        using ``weighted_average_1d`` (but this cannot return an output
        at daily resolution).
        Unlike ``get_specific_mb``, this returns the weighted average
        for all days.
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
            # with leap years we don't know the size of the final array
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
                mb = mb_mod.get_daily_mb(fl.surface_h, year=mb_yr, fls=fls, fl_id=i)
                mbs.append(mb)

            mbs = np.vstack(mbs)
            widths = np.hstack(widths)
            mbs = weighted_average_2d(mbs, widths) * SEC_IN_DAY * mb_mod.rho
            stack.append(mbs)
        return set_array_type(np.concatenate(stack, axis=0))
    
    def get_specific_mb_monthly(
        self, heights=None, widths=None, fls=None, year=None
    ):
        """Get specific monthly mass balance.

        TODO: Returns identical annual SMB to ``get_specific_mb`` when
        using ``weighted_average_1d`` (but this cannot return an output
        at daily resolution).
        Unlike ``get_specific_mb``, this returns the weighted average
        for all days.
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
            mbs = self.get_monthly_specific_mass_balance(fls=fls, year=mb_yr)

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


def decide_winter_precip_factor(gdir, baseline_climate_suffix:str=""):
    """Utility function to decide on a precip factor based on winter precip."""

    # We have to decide on a precip factor
    if 'W5E5' not in cfg.PARAMS['baseline_climate']:
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
    a, b = cfg.PARAMS['winter_prcp_fac_ab']
    prcp_fac = a * np.log(w_prcp) + b
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
                                    extra_model_kwargs: dict = None,
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
        ref_mb_df = 'table_hugonnet_regions_10yr_20yr_ar6period.csv'
        ref_mb_df = pd.read_csv(get_demo_file(ref_mb_df))
        ref_mb_df = ref_mb_df.loc[ref_mb_df.period == ref_period].set_index('reg')
        # dmdtda already in kg m-2 yr-1
        ref_mb = ref_mb_df.loc[int(gdir.rgi_region), 'dmdtda']
        ref_mb_err = ref_mb_df.loc[int(gdir.rgi_region), 'err_dmdtda']
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
    if cfg.PARAMS['use_temp_bias_from_file']:
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
        if not cfg.PARAMS['use_temp_bias_from_file']:
            raise InvalidParamsError('With `informed_threestep` you need to '
                                     'set `use_temp_bias_from_file`.')
        if not cfg.PARAMS['use_winter_prcp_fac']:
            raise InvalidParamsError('With `informed_threestep` you need to '
                                     'set `use_winter_prcp_fac`.')

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
                                             extra_model_kwargs=extra_model_kwargs,
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
                                             extra_model_kwargs=extra_model_kwargs,
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
                                  baseline_climate_suffix:str=None,
                                  eolis_data=None,
                                  extra_model_kwargs: dict = None,
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
    extra_model_kwargs: dict
        Extra model parameters to pass to mb_model_class.
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
    # TODO: doesn't support hydro years
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
        if cfg.PARAMS['use_winter_prcp_fac']:
            # Some sanity check
            if cfg.PARAMS['prcp_fac'] is not None:
                raise InvalidWorkflowError("Set PARAMS['prcp_fac'] to None "
                                           "if using PARAMS['winter_prcp_factor'].")
            prcp_fac = decide_winter_precip_factor(gdir, baseline_climate_suffix=baseline_climate_suffix)
        else:
            prcp_fac = cfg.PARAMS['prcp_fac']
            if prcp_fac is None:
                raise InvalidWorkflowError("Set either PARAMS['use_winter_prcp_fac'] "
                                           "or PARAMS['winter_prcp_factor'].")

    if temp_bias is None:
        temp_bias = 0

    # Create the MB model we will calibrate
    if not extra_model_kwargs:
        mb_mod = mb_model_class(
            gdir,
            melt_f=melt_f,
            temp_bias=temp_bias,
            prcp_fac=prcp_fac,
            check_calib_params=False,
            mb_params_filesuffix=filesuffix,)
    else:
        mb_mod = mb_model_class(
            gdir,
            melt_f=melt_f,
            temp_bias=temp_bias,
            prcp_fac=prcp_fac,
            check_calib_params=False,
            mb_params_filesuffix=filesuffix,
            **extra_model_kwargs)

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
