"""Mass-balance models"""
# Built ins
# External libs
import numpy as np
import pandas as pd
import netCDF4
from scipy.interpolate import interp1d
from scipy import optimize as optimization
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import (SuperclassMeta, lazy_property, floatyear_to_date,
                        date_to_floatyear, monthly_timeseries)


class MassBalanceModel(object, metaclass=SuperclassMeta):
    """Common logic for the mass balance models.

    All mass-balance models should implement this interface.

    Attributes
    ----------
    valid_bounds : [float, float]
        The altitudinal bounds where the MassBalanceModel is valid. This is
        necessary for automated ELA search.
    """

    def __init__(self):
        """ Initialize."""
        self.valid_bounds = None

    def get_monthly_mb(self, heights, year=None):
        """Monthly mass-balance at given altitude(s) for a moment in time.

        Units: [m s-1], or meters of ice per second

        Note: `year` is optional because some simpler models have no time
        component.

        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "hydrological floating year" convention)

        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        raise NotImplementedError()

    def get_annual_mb(self, heights, year=None):
        """Like `self.get_monthly_mb()`, but for annual MB.

        For some simpler mass-balance models ``get_monthly_mb()` and
        `get_annual_mb()`` can be equivalent.

        Units: [m s-1], or meters of ice per second

        Note: `year` is optional because some simpler models have no time
        component.

        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "floating year" convention)

        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        raise NotImplementedError()

    def get_specific_mb(self, heights, widths, year=None):
        """Specific mb for this year and a specific glacier geometry.

         Units: [mm w.e. yr-1], or millimeter water equivalent per year

        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass-balance will be computed
        widths: ndarray
            the widths of the flowline (necessary for the weighted average)
        year: float, optional
            the time (in the "hydrological floating year" convention)

        Returns
        -------
        the specific mass-balance (units: mm w.e. yr-1)
        """
        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_mb(heights, widths, year=yr)
                   for yr in year]
            return np.asarray(out)

        mbs = self.get_annual_mb(heights, year=year) * SEC_IN_YEAR * cfg.RHO
        return np.average(mbs, weights=widths)

    def get_ela(self, year=None):
        """Compute the equilibrium line altitude for this year

        Parameters
        ----------
        year: float, optional
            the time (in the "hydrological floating year" convention)

        Returns
        -------
        the equilibrium line altitude (ELA, units: m)
        """

        if len(np.atleast_1d(year)) > 1:
            return np.asarray([self.get_ela(year=yr) for yr in year])

        if self.valid_bounds is None:
            raise ValueError('attribute `valid_bounds` needs to be '
                             'set for the ELA computation.')

        # Check for invalid ELAs
        b0, b1 = self.valid_bounds
        if (np.any(~np.isfinite(self.get_annual_mb([b0, b1], year=year))) or
                (self.get_annual_mb([b0], year=year)[0] > 0) or
                (self.get_annual_mb([b1], year=year)[0] < 0)):
            return np.NaN

        def to_minimize(x):
            o = self.get_annual_mb([x], year=year)[0] * SEC_IN_YEAR * cfg.RHO
            return o
        return optimization.brentq(to_minimize, *self.valid_bounds, xtol=0.1)


class LinearMassBalance(MassBalanceModel):
    """Constant mass-balance as a linear function of altitude.
    """

    def __init__(self, ela_h, grad=3., max_mb=None):
        """ Initialize.

        Parameters
        ----------
        ela_h: float
            Equilibrium line altitude (units: [m])
        grad: float
            Mass-balance gradient (unit: [mm w.e. yr-1 m-1])
        max_mb: float
            Cap the mass balance to a certain value (unit: [mm w.e. yr-1])

        Attributes
        ----------
        temp_bias : float, default 0
            A "temperature bias" doesn't makes much sense in the linear MB
            context, but we implemented a simple empirical rule:
            + 1K -> ELA + 150 m
        """
        super(LinearMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.orig_ela_h = ela_h
        self.ela_h = ela_h
        self.grad = grad
        self.max_mb = max_mb
        self._temp_bias = 0

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to change the ELA."""
        self.ela_h = self.orig_ela_h + value * 150
        self._temp_bias = value

    def get_monthly_mb(self, heights, year=None):
        mb = (np.asarray(heights) - self.ela_h) * self.grad
        if self.max_mb is not None:
            mb = mb.clip(None, self.max_mb)
        return mb / SEC_IN_YEAR / cfg.RHO

    def get_annual_mb(self, heights, year=None):
        return self.get_monthly_mb(heights, year=year)


class PastMassBalance(MassBalanceModel):
    """Mass balance during the climate data period."""

    def __init__(self, gdir, mu_star=None, bias=None, prcp_fac=None,
                 filename='climate_monthly', input_filesuffix='',
                 repeat=False, ys=None, ye=None):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mustar you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        prcp_fac : float, optional
            set to the alternative value of the precipitation factor
            you want to use (the default is to use the calibrated value)
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

        Attributes
        ----------
        temp_bias : float, default 0
            Add a temperature bias to the time series
        prcp_bias : float, default 1
            Precipitation factor to the time series (called bias for
            consistency with `temp_bias`)
        """

        super(PastMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        if mu_star is None:
            df = pd.read_csv(gdir.get_filepath('local_mustar'))
            mu_star = df['mu_star'][0]
        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                df = pd.read_csv(gdir.get_filepath('local_mustar'))
                bias = df['bias'][0]
            else:
                bias = 0.
        if prcp_fac is None:
            df = pd.read_csv(gdir.get_filepath('local_mustar'))
            prcp_fac = df['prcp_fac'][0]
        self.mu_star = mu_star
        self.bias = bias

        # Parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_liq = cfg.PARAMS['temp_all_liq']
        self.t_melt = cfg.PARAMS['temp_melt']

        # Public attrs
        self.temp_bias = 0.
        self.prcp_bias = 1.
        self.repeat = repeat

        # Read file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with netCDF4.Dataset(fpath, mode='r') as nc:
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
            self.grad = nc.variables['grad'][:]
            self.ref_hgt = nc.ref_hgt
            self.ys = self.years[0] if ys is None else ys
            self.ye = self.years[-1] if ye is None else ye

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

        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = self.prcp[pok] * self.prcp_bias
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        tempformelt = temp - self.t_melt
        tempformelt[:] = np.clip(tempformelt, 0, tempformelt.max())

        # Compute solid precipitation from total precipitation
        prcp = np.ones(npix) * iprcp
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * np.clip(fac, 0, 1)

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

        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = self.prcp[pok] * self.prcp_bias
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
        temp2dformelt[:] = np.clip(temp2dformelt, 0, temp2dformelt.max())

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
        fac = np.clip(fac, 0, 1)
        prcpsol = prcp * fac

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tfmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tfmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None):

        _, tmelt, _, prcpsol = self.get_monthly_climate(heights, year=year)
        mb_month = prcpsol - self.mu_star * tmelt
        mb_month -= self.bias * SEC_IN_MONTH / SEC_IN_YEAR
        return mb_month / SEC_IN_MONTH / cfg.RHO

    def get_annual_mb(self, heights, year=None):

        _, temp2dformelt, _, prcpsol = self._get_2d_annual_climate(heights,
                                                                   year)
        mb_annual = np.sum(prcpsol - self.mu_star * temp2dformelt, axis=1)
        return (mb_annual - self.bias) / SEC_IN_YEAR / cfg.RHO


class ConstantMassBalance(MassBalanceModel):
    """Constant mass-balance during a chosen period.

    This is useful for equilibrium experiments.
    """

    def __init__(self, gdir, mu_star=None, bias=None, prcp_fac=None,
                 y0=None, halfsize=15, filename='climate_monthly', 
                 input_filesuffix=''):
        """Initialize

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mustar you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the annual bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
        prcp_fac : float, optional
            set to the alternative value of the precipitation factor
            you want to use (the default is to use the calibrated value)
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

        super(ConstantMassBalance, self).__init__()
        self.mbmod = PastMassBalance(gdir, mu_star=mu_star, bias=bias,
                                     prcp_fac=prcp_fac, filename=filename, 
                                     input_filesuffix=input_filesuffix)

        if y0 is None:
            df = pd.read_csv(gdir.get_filepath('local_mustar'))
            y0 = df['t_star'][0]

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
            with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
                zminmax = [nc.min_h_dem-250, nc.max_h_dem+1500]
        self.hbins = np.arange(*zminmax, step=10)
        self.valid_bounds = self.hbins[[0, -1]]
        self.y0 = y0
        self.halfsize = halfsize
        self.years = np.arange(y0-halfsize, y0+halfsize+1)

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

    @lazy_property
    def interp_yr(self):
        # annual MB
        mb_on_h = self.hbins*0.
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

    def get_climate(self, heights, year=None):
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
        # Note that we do not weight for number of days per month - bad
        return (np.mean(temp, axis=0),
                np.mean(tempformelt, axis=0) * 12,
                np.mean(prcp, axis=0) * 12,
                np.mean(prcpsol, axis=0) * 12)

    def get_monthly_mb(self, heights, year=None):
        yr, m = floatyear_to_date(year)
        return self.interp_m[m-1](heights)

    def get_annual_mb(self, heights, year=None):
        return self.interp_yr(heights)


class RandomMassBalance(MassBalanceModel):
    """Random shuffle of all MB years within a given time period.

    This is useful for finding a possible past glacier state or for sensitivity
    experiments.

    Note that this is going to be sensitive to extreme years in certain
    periods, but it is by far more physically reasonable than other
    approaches based on gaussian assumptions.
    """

    def __init__(self, gdir, mu_star=None, bias=None, prcp_fac=None,
                 y0=None, halfsize=15, seed=None, filename='climate_monthly',
                 input_filesuffix='', all_years=False):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mustar you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        prcp_fac : float, optional
            set to the alternative value of the precipitation factor
            you want to use (the default is to use the calibrated value)
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
        """

        super(RandomMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.mbmod = PastMassBalance(gdir, mu_star=mu_star, bias=bias,
                                     prcp_fac=prcp_fac, filename=filename,
                                     input_filesuffix=input_filesuffix)

        # Climate period
        if all_years:
            self.years = self.mbmod.years
        else:
            if y0 is None:
                df = pd.read_csv(gdir.get_filepath('local_mustar'))
                y0 = df['t_star'][0]
            self.years = np.arange(y0-halfsize, y0+halfsize+1)
        self.yr_range = (self.years[0], self.years[-1]+1)
        self.ny = len(self.years)

        # RandomState
        self.rng = np.random.RandomState(seed)
        self._state_yr = dict()

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
            self._state_yr[year] = self.rng.randint(*self.yr_range)
        return self._state_yr[year]

    def get_monthly_mb(self, heights, year=None):
        ryr, m = floatyear_to_date(year)
        ryr = date_to_floatyear(self.get_state_yr(ryr), m)
        return self.mbmod.get_monthly_mb(heights, year=ryr)

    def get_annual_mb(self, heights, year=None):
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_annual_mb(heights, year=ryr)


class UncertainMassBalance(MassBalanceModel):
    """Adding uncertainty to a mass balance model.

    There are three variables for which you can add uncertainty:
    - temperature (additive bias)
    - precipitation (multiplicative factor)
    - residual (a bias in units of MB)
    """

    def __init__(self, basis_model,
                 rdn_temp_bias_seed=None, rdn_temp_bias_sigma=0.1,
                 rdn_prcp_bias_seed=None, rdn_prcp_bias_sigma=0.1,
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
        rdn_prcp_bias_seed : int
            the seed of the random number generator
        rdn_prcp_bias_sigma : float
            the standard deviation of the random precipitation error
        rdn_bias_seed : int
            the seed of the random number generator
        rdn_bias_sigma : float
            the standard deviation of the random MB error
        """
        super(UncertainMassBalance, self).__init__()
        self.mbmod = basis_model
        self.valid_bounds = self.mbmod.valid_bounds
        self.rng_temp = np.random.RandomState(rdn_temp_bias_seed)
        self.rng_prcp = np.random.RandomState(rdn_prcp_bias_seed)
        self.rng_bias = np.random.RandomState(rdn_bias_seed)
        self._temp_sigma = rdn_temp_bias_sigma
        self._prcp_sigma = rdn_prcp_bias_sigma
        self._bias_sigma = rdn_bias_sigma
        self._state_temp = dict()
        self._state_prcp = dict()
        self._state_bias = dict()

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
        self.mbmod.prcp_bias = value

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

    def get_annual_mb(self, heights, year=None):

        # Keep the original biases and add a random error
        _t = self.mbmod.temp_bias
        _p = self.mbmod.prcp_bias
        _b = self.mbmod.bias
        self.mbmod.temp_bias = self._get_state_temp(year) + _t
        self.mbmod.prcp_bias = self._get_state_prcp(year) + _p
        self.mbmod.bias = self._get_state_bias(year) + _b
        try:
            out = self.mbmod.get_annual_mb(heights, year=year)
        except:
            self.mbmod.temp_bias = _t
            self.mbmod.prcp_bias = _p
            self.mbmod.bias = _b
            raise
        # Back to normal
        self.mbmod.temp_bias = _t
        self.mbmod.prcp_bias = _p
        self.mbmod.bias = _b
        return out
