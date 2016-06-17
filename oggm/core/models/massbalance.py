"""Mass-balance stuffs"""
from __future__ import division

# Built ins
# External libs
import numpy as np
import pandas as pd
import netCDF4
import warnings
from scipy.interpolate import interp1d
from numpy import random
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTHS, SEC_IN_DAY
from oggm.core.preprocessing import climate
from oggm import utils


class MassBalanceModel(object):
    """An interface for mass balance."""

    def __init__(self, bias=0.):
        """ Instanciate."""

        self._bias = 0.
        self.set_bias(bias=bias)
        pass

    def set_bias(self, bias=0):
        self._bias = bias

    def get_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""
        raise NotImplementedError()


class ConstantBalanceModel(MassBalanceModel):
    """Simple gradient MB model."""

    def __init__(self, ela_h, grad=3., bias=0.):
        """ Instanciate.

        Parameters
        ---------
        ela_h: float
            Equilibrium line altitude
        grad: float
            Mass-balance gradient (unit: mm m-1)
        """

        super(ConstantBalanceModel, self).__init__(bias)

        self.ela_h = ela_h
        self.grad = grad

    def get_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        mb = (heights - self.ela_h) * self.grad + self._bias
        return mb / SEC_IN_YEAR / 1000.


class TstarMassBalanceModel(MassBalanceModel):
    """Constant mass balance: equilibrium MB at period t*."""

    def __init__(self, gdir, bias=0.):
        """ Instanciate."""

        super(TstarMassBalanceModel, self).__init__(bias)

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        mu_star = df['mu_star'][0]
        t_star = df['t_star'][0]

        # Climate period
        mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
        yr = [t_star-mu_hp, t_star+mu_hp]

        fls = gdir.read_pickle('model_flowlines')
        h = np.array([])
        for fl in fls:
            h = np.append(h, fl.surface_h)
        h = np.linspace(np.min(h)-200, np.max(h)+1200, 1000)

        y, t, p = climate.mb_yearly_climate_on_height(gdir, h, year_range=yr)
        t = np.mean(t, axis=1)
        p = np.mean(p, axis=1)
        mb_on_h = p - mu_star * t

        self.interp = interp1d(h, mb_on_h)
        self.t_star = t_star

    def get_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        return (self.interp(heights) + self._bias) / SEC_IN_YEAR / cfg.RHO


class BackwardsMassBalanceModel(MassBalanceModel):
    """Constant mass balance: MB for [1983, 2003] with temperature bias.

    This is useful for finding a possible past galcier state.
    """

    def __init__(self, gdir, use_tstar=False, bias=0.):
        """ Instanciate."""

        super(BackwardsMassBalanceModel, self).__init__(bias)

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        self.mu_star = df['mu_star'][0]

        # Climate period
        if use_tstar:
            t_star = df['t_star'][0]
            mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
            yr_range = [t_star-mu_hp, t_star+mu_hp]
        else:
            # TODO temporary solution until https://github.com/OGGM/oggm/issues/88
            fpath = gdir.get_filepath('climate_monthly')
            with netCDF4.Dataset(fpath, mode='r') as nc:
                # time
                time = nc.variables['time']
                time = netCDF4.num2date(time[:], time.units)
                last_yr = time[-1].year
                yr_range = [last_yr-30, last_yr]

        # Parameters
        self.temp_all_solid = cfg.PARAMS['temp_all_solid']
        self.temp_all_liq = cfg.PARAMS['temp_all_liq']
        self.temp_melt = cfg.PARAMS['temp_melt']

        # Read file
        fpath = gdir.get_filepath('climate_monthly')
        with netCDF4.Dataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            ny, r = divmod(len(time), 12)
            yrs = np.arange(time[-1].year-ny+1, time[-1].year+1, 1).repeat(12)
            assert len(yrs) == len(time)
            p0 = np.min(np.nonzero(yrs == yr_range[0])[0])
            p1 = np.max(np.nonzero(yrs == yr_range[1])[0]) + 1

            # Read timeseries
            self.temp = nc.variables['temp'][p0:p1]
            self.prcp = nc.variables['prcp'][p0:p1]
            self.grad = nc.variables['grad'][p0:p1]
            self.ref_hgt = nc.ref_hgt

        # Ny
        ny, r = divmod(len(self.temp), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        self.ny = ny

        # For optimisation
        self._interp = dict()

        # Get default heights
        fls = gdir.read_pickle('model_flowlines')
        h = np.array([])
        for fl in fls:
            h = np.append(h, fl.surface_h)
        npix = 1000
        self.heights = np.linspace(np.min(h)-200, np.max(h)+1200, npix)
        grad_temp = np.atleast_2d(self.grad).repeat(npix, 0)
        grad_temp *= (self.heights.repeat(12*self.ny).reshape(grad_temp.shape) -
                      self.ref_hgt)
        self.temp_2d = np.atleast_2d(self.temp).repeat(npix, 0) + grad_temp
        self.prcpsol = np.atleast_2d(self.prcp).repeat(npix, 0)

    def _get_interp(self):

        if self._bias not in self._interp:

            # Bias is in megative % units of degree TODO: change this
            delta_t = - self._bias / 100.

            # For each height pixel:
            # Compute temp and tempformelt (temperature above melting threshold)
            temp2d = self.temp_2d + delta_t
            temp2dformelt = (temp2d - self.temp_melt).clip(0)

            # Compute solid precipitation from total precipitation

            fac = 1 - (temp2d - self.temp_all_solid) / (self.temp_all_liq - self.temp_all_solid)
            fac = np.clip(fac, 0, 1)
            prcpsol = self.prcpsol * fac

            mb_annual = np.sum(prcpsol - self.mu_star * temp2dformelt, axis=1) / self.ny
            self._interp[self._bias] = interp1d(self.heights, mb_annual)

        return self._interp[self._bias]


    def get_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        interp = self._get_interp()
        return interp(heights) / SEC_IN_YEAR / cfg.RHO


class BiasedMassBalanceModel(MassBalanceModel):
    """Constant mass balance: MB for a given 30yr period,
     with temperature and/or precipitation bias.

    This is useful for finding a possible past glacier state or for sensitivity
    experiments.
    """

    def __init__(self, gdir, year=None, use_tstar=False, bias=None):
        """ Instanciate.

        Parameters
        ---------
        year: int
            provide a year around which the 30 yr period will be averaged.
            default is the last 30 yrs
        use_tstar: bool
            overwrites year to be tstar
        bias: Deprecated
        """

        if (bias is not None) and (bias != 0.):
            raise ValueError('bias should be zero for BiasedMassBalanceModel')
        super(BiasedMassBalanceModel, self).__init__()

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        self.mu_star = df['mu_star'][0]

        # Climate period
        if use_tstar:
            year = df['t_star'][0]

        if year is not None:
            mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
            yr_range = [year-mu_hp, year+mu_hp]
        else:
            # temporary solution until https://github.com/OGGM/oggm/issues/88
            # is implemented
            fpath = gdir.get_filepath('climate_monthly')
            with netCDF4.Dataset(fpath, mode='r') as nc:
                # time
                time = nc.variables['time']
                time = netCDF4.num2date(time[:], time.units)
                last_yr = time[-1].year
            yr_range = [last_yr-30, last_yr]

        # Parameters
        self.temp_all_solid = cfg.PARAMS['temp_all_solid']
        self.temp_all_liq = cfg.PARAMS['temp_all_liq']
        self.temp_melt = cfg.PARAMS['temp_melt']

        # Read file
        fpath = gdir.get_filepath('climate_monthly')
        with netCDF4.Dataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            ny, r = divmod(len(time), 12)
            yrs = np.arange(time[-1].year-ny+1, time[-1].year+1, 1).repeat(12)
            assert len(yrs) == len(time)
            p0 = np.min(np.nonzero(yrs == yr_range[0])[0])
            p1 = np.max(np.nonzero(yrs == yr_range[1])[0]) + 1

            # Read timeseries
            self.temp = nc.variables['temp'][p0:p1]
            self.prcp = nc.variables['prcp'][p0:p1]
            self.grad = nc.variables['grad'][p0:p1]
            self.ref_hgt = nc.ref_hgt

        # Ny
        ny, r = divmod(len(self.temp), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        self.ny = ny

        # For optimisation
        self._temp_bias = 0.
        self._prcp_fac = 1.
        self._interp = dict()

        # Get default heights
        fls = gdir.read_pickle('model_flowlines')
        h = np.array([])
        for fl in fls:
            h = np.append(h, fl.surface_h)
        npix = 1000
        self.heights = np.linspace(np.min(h)-200, np.max(h)+1200, npix)
        grad_temp = np.atleast_2d(self.grad).repeat(npix, 0)
        grad_temp *= (self.heights.repeat(12*self.ny).reshape(grad_temp.shape) -
                      self.ref_hgt)
        self.temp_2d = np.atleast_2d(self.temp).repeat(npix, 0) + grad_temp
        self.prcpsol = np.atleast_2d(self.prcp).repeat(npix, 0)

    def set_bias(self, bias=0):
        """Override set bias"""

        if (bias is not None) and (bias != 0.):
            raise ValueError('bias should be zero for BiasedMassBalanceModel')

        self._bias = bias

    def set_temp_bias(self, bias=0):
        """Add a temperature bias (in K) to the temperature climatology"""
        self._temp_bias = bias

    def set_prcp_factor(self, factor=1):
        """Multiply precipitation climatology with a given factor"""
        self._prcp_fac = factor

    def _get_interp(self):

        key = (self._temp_bias, self._prcp_fac)
        if key not in self._interp:

            # For each height pixel:
            # Compute temp and tempformelt (temperature above melting threshold)
            temp2d = self.temp_2d + self._temp_bias
            temp2dformelt = (temp2d - self.temp_melt).clip(0)

            # Compute solid precipitation from total precipitation
            fac = 1 - (temp2d - self.temp_all_solid) / (self.temp_all_liq - self.temp_all_solid)
            fac = np.clip(fac, 0, 1)
            prcpsol = self.prcpsol * self._prcp_fac * fac

            mb_annual = np.sum(prcpsol - self.mu_star * temp2dformelt, axis=1) / self.ny
            self._interp[key] = interp1d(self.heights, mb_annual)

        return self._interp[key]

    def get_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        interp = self._get_interp()
        return interp(heights) / SEC_IN_YEAR / cfg.RHO


class RandomMassBalanceModel(MassBalanceModel):
    """This returns a mass-balance which is simply a random shuffle of all
    years within a 31 years period.

    This is useful for finding a possible past glacier state or for sensitivity
    experiments.

    Note that this is going to be sensitive to extreme years in certain
    periods, but it is by far more physically reasonable than other
    approaches based on gaussian assumptions.
    """

    def __init__(self, gdir, year=None, use_tstar=False, bias=None):
        """ Instanciate.

        Parameters
        ---------
        year: int
            provide a year around which the 30 yr period will be averaged.
            default is the last 30 yrs
        use_tstar: bool
            overwrites year to be tstar
        bias: Deprecated
        """

        if (bias is not None) and (bias != 0.):
            raise ValueError('bias should be zero for RandomMassBalanceModel')
        super(RandomMassBalanceModel, self).__init__()

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        self.mu_star = df['mu_star'][0]

        # Climate period
        if use_tstar:
            year = df['t_star'][0]

        if year is not None:
            mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
            yr_range = [year-mu_hp, year+mu_hp]
        else:
            # temporary solution until https://github.com/OGGM/oggm/issues/88
            # is implemented
            fpath = gdir.get_filepath('climate_monthly')
            with netCDF4.Dataset(fpath, mode='r') as nc:
                # time
                time = nc.variables['time']
                time = netCDF4.num2date(time[:], time.units)
                last_yr = time[-1].year
            yr_range = [last_yr-30, last_yr]

        # Parameters
        self.temp_all_solid = cfg.PARAMS['temp_all_solid']
        self.temp_all_liq = cfg.PARAMS['temp_all_liq']
        self.temp_melt = cfg.PARAMS['temp_melt']

        # Read file
        fpath = gdir.get_filepath('climate_monthly')
        with netCDF4.Dataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            ny, r = divmod(len(time), 12)
            yrs = np.arange(time[-1].year-ny+1, time[-1].year+1, 1).repeat(12)
            assert len(yrs) == len(time)
            p0 = np.min(np.nonzero(yrs == yr_range[0])[0])
            p1 = np.max(np.nonzero(yrs == yr_range[1])[0]) + 1
            ny = yr_range[1] - yr_range[0] + 1

            # Read timeseries and store the stats
            self.temp_s = nc.variables['temp'][p0:p1].reshape((ny, 12))
            self.prcp_s = nc.variables['prcp'][p0:p1].reshape((ny, 12))
            self.grad_s = nc.variables['grad'][p0:p1].reshape((ny, 12))
            self.ref_hgt = nc.ref_hgt

        # Ny
        assert ny == 31
        self.ny = ny

        # For optimisation
        self._temp_bias = 0.
        self._prcp_fac = 1.
        self._cyears = dict()

    def set_bias(self, bias=0):
        """Override set bias"""
        if (bias is not None) and (bias != 0.):
            raise ValueError('bias should be zero for RandomMassBalanceModel')

    def set_temp_bias(self, bias=0):
        """Add a temperature bias (in K) to the temperature climatology"""
        self._temp_bias = bias

    def set_prcp_factor(self, factor=1):
        """Multiply precipitation climatology with a given factor"""
        self._prcp_fac = factor

    def _get_syr(self, year):
        """One random per year"""

        if year not in self._cyears:
            # random timeseries
            r = random.randint(self.ny)
            itemp = self.temp_s[r, :]
            iprcp = self.prcp_s[r, :]
            igrad = self.grad_s[r, :]
            self._cyears[year] = (itemp, iprcp, igrad)

        return self._cyears[year]

    def get_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        y, m = utils.year_to_date(year)
        itemp, iprcp, igrad = self._get_syr(y)
        itemp, iprcp, igrad = itemp[m-1], iprcp[m-1], igrad[m-1]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        grad_temp = igrad * (heights - self.ref_hgt)
        temp = np.ones(npix) * itemp + grad_temp
        tempformelt = temp - self.temp_melt
        tempformelt = np.clip(tempformelt, 0, tempformelt.max())

        # Compute solid precipitation from total precipitation
        prcpsol = np.ones(npix) * iprcp
        fac = 1 - (temp - self.temp_all_solid) / \
                  (self.temp_all_liq - self.temp_all_solid)
        prcpsol *= np.clip(fac, 0, 1)

        mb_month = prcpsol - self.mu_star * tempformelt
        return mb_month / SEC_IN_MONTHS[m-1] / cfg.RHO

    def get_annual_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        itemp, iprcp, igrad = self._get_syr(year)

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(12).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp + self._temp_bias).repeat(npix, 0) + grad_temp
        temp2dformelt = temp2d - self.temp_melt
        temp2dformelt = np.clip(temp2dformelt, 0, temp2dformelt.max())

        # Compute solid precipitation from total precipitation
        prcpsol = np.atleast_2d(iprcp * self._prcp_fac).repeat(npix, 0)
        fac = 1 - (temp2d - self.temp_all_solid) / (self.temp_all_liq - self.temp_all_solid)
        fac = np.clip(fac, 0, 1)
        prcpsol = prcpsol * fac

        mb_annual = np.sum(prcpsol - self.mu_star * temp2dformelt, axis=1)
        return mb_annual / SEC_IN_YEAR / cfg.RHO


class TodayMassBalanceModel(MassBalanceModel):
    """Constant mass-balance: MB during the last 30 yrs."""

    def __init__(self, gdir, bias=0.):
        """ Instanciate."""

        super(TodayMassBalanceModel, self).__init__(bias)

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        mu_star = df['mu_star'][0]
        t_star = df['t_star'][0]

        # Climate period
        # TODO temporary solution until https://github.com/OGGM/oggm/issues/88
        # is implemented
        fpath = gdir.get_filepath('climate_monthly')
        with netCDF4.Dataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            last_yr = time[-1].year
        yr = [last_yr-30, last_yr]

        fls = gdir.read_pickle('model_flowlines')
        h = np.array([])
        for fl in fls:
            h = np.append(h, fl.surface_h)
        h = np.linspace(np.min(h)-100, np.max(h)+200, 1000)

        y, t, p = climate.mb_yearly_climate_on_height(gdir, h, year_range=yr)
        t = np.mean(t, axis=1)
        p = np.mean(p, axis=1)
        mb_on_h = p - mu_star * t

        self.interp = interp1d(h, mb_on_h)

    def get_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        return (self.interp(heights) + self._bias) / SEC_IN_YEAR / cfg.RHO


class HistalpMassBalanceModel(MassBalanceModel):
    """Mass balance during the HISTALP period."""

    def __init__(self, gdir):
        """ Instanciate."""

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        self.mu_star = df['mu_star'][0]

        # Parameters
        self.temp_all_solid = cfg.PARAMS['temp_all_solid']
        self.temp_all_liq = cfg.PARAMS['temp_all_liq']
        self.temp_melt = cfg.PARAMS['temp_melt']

        # Read file
        fpath = gdir.get_filepath('climate_monthly')
        with netCDF4.Dataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError('Climate data should be N full years exclusively')
            # Last year gives the tone of the hydro year
            self.years = np.repeat(np.arange(time[-1].year-ny+1,
                                             time[-1].year+1), 12)
            self.months = np.tile(np.arange(1, 13), ny)
            # Read timeseries
            self.temp = nc.variables['temp'][:]
            self.prcp = nc.variables['prcp'][:]
            self.grad = nc.variables['grad'][:]
            self.ref_hgt = nc.ref_hgt

    def get_mb(self, heights, year=None):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        y, m = utils.year_to_date(year)

        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read timeseries
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        grad_temp = igrad * (heights - self.ref_hgt)
        temp = np.ones(npix) * itemp + grad_temp
        tempformelt = temp - self.temp_melt
        tempformelt = np.clip(tempformelt, 0, tempformelt.max())

        # Compute solid precipitation from total precipitation
        prcpsol = np.ones(npix) * iprcp
        fac = 1 - (temp - self.temp_all_solid) / \
                  (self.temp_all_liq - self.temp_all_solid)
        prcpsol *= np.clip(fac, 0, 1)

        mb_month = prcpsol - self.mu_star * tempformelt
        return mb_month / SEC_IN_MONTHS[m-1] / cfg.RHO

    def get_annual_mb(self, heights, year=None):
        """Returns the annual mass-balance at given altitudes
        for a given moment in time."""

        pok = np.where(self.years == np.floor(year))[0]

        # Read timeseries
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(12).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = temp2d - self.temp_melt
        temp2dformelt = np.clip(temp2dformelt, 0, temp2dformelt.max())

        # Compute solid precipitation from total precipitation
        prcpsol = np.atleast_2d(iprcp).repeat(npix, 0)
        fac = 1 - (temp2d - self.temp_all_solid) / (self.temp_all_liq - self.temp_all_solid)
        fac = np.clip(fac, 0, 1)
        prcpsol = prcpsol * fac

        mb_annual = np.sum(prcpsol - self.mu_star * temp2dformelt, axis=1)
        return mb_annual / SEC_IN_YEAR / cfg.RHO