"""Mass-balance stuffs"""
from __future__ import division
from six.moves import zip

# Built ins
# External libs
import numpy as np
import pandas as pd
import netCDF4
from scipy.interpolate import interp1d
# Locals
import oggm.conf as cfg
from oggm.prepro import climate


sec_in_year = 365*24*3600


class MassBalanceModel(object):
    """An interface for mass balance."""

    def __init__(self, bias=0.):
        """ Instanciate."""

        self._bias = 0.
        self.set_bias(bias=bias)
        pass

    def set_bias(self, bias=0):
        self._bias = bias

    def get_mb(self, heights, year):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""
        raise NotImplementedError()


class TstarMassBalanceModel(MassBalanceModel):
    """Mass balance for the equilibrium mass-balance around t*."""

    def __init__(self, gdir, bias=0.):
        """ Instanciate."""

        super(TstarMassBalanceModel, self).__init__(bias)

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        mu_star = df['mu_star'][0]
        t_star = df['t_star'][0]

        # Climate period
        mu_hp = int(cfg.params['mu_star_halfperiod'])
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

    def get_mb(self, heights, year):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        return (self.interp(heights) + self._bias) / sec_in_year / 900


class BackwardsMassBalanceModel(MassBalanceModel):
    """Mass balance for finding the initial glacier."""

    def __init__(self, gdir, use_tstar=False, bias=0.):
        """ Instanciate."""

        super(BackwardsMassBalanceModel, self).__init__(bias)

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        self.mu_star = df['mu_star'][0]

        # Climate period
        if use_tstar:
            t_star = df['t_star'][0]
            mu_hp = int(cfg.params['mu_star_halfperiod'])
            yr_range = [t_star-mu_hp, t_star+mu_hp]
        else:
            yr_range = [1983, 2003]

        # Parameters
        self.temp_all_solid = cfg.params['temp_all_solid']
        self.temp_all_liq = cfg.params['temp_all_liq']
        self.temp_melt = cfg.params['temp_melt']

        # Read file
        nc = netCDF4.Dataset(gdir.get_filepath('climate_monthly'), mode='r')

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
        nc.close()

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


    def get_mb(self, heights, year):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        interp = self._get_interp()
        return interp(heights) / sec_in_year / 900.


class TodayMassBalanceModel(MassBalanceModel):
    """Mass-balance during the last 30 yrs."""

    def __init__(self, gdir, bias=0.):
        """ Instanciate."""

        super(TodayMassBalanceModel, self).__init__(bias)

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        mu_star = df['mu_star'][0]
        t_star = df['t_star'][0]

        # Climate period
        yr = [1983, 2003]

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

    def get_mb(self, heights, year):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        return (self.interp(heights) + self._bias) / sec_in_year / 900.


class HistalpMassBalanceModel(MassBalanceModel):
    """Mass balance Histalp period."""

    def __init__(self, gdir):
        """ Instanciate."""

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        self.mu_star = df['mu_star'][0]

        # Parameters
        self.temp_all_solid = cfg.params['temp_all_solid']
        self.temp_all_liq = cfg.params['temp_all_liq']
        self.temp_melt = cfg.params['temp_melt']

        # Read file
        nc = netCDF4.Dataset(gdir.get_filepath('climate_monthly'), mode='r')

        # time
        time = nc.variables['time']
        time = netCDF4.num2date(time[:], time.units)
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        # Last year gives the tone of the hydro year
        self.years = np.arange(time[-1].year-ny+1, time[-1].year+1, 1)
        # Read timeseries
        self.temp = nc.variables['temp'][:]
        self.prcp = nc.variables['prcp'][:]
        self.grad = nc.variables['grad'][:]
        self.ref_hgt = nc.ref_hgt
        nc.close()

    def get_mb(self, heights, year):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        pok = np.where(self.years == np.floor(year))[0][0]

        # Read timeseries
        itemp = self.temp[12*pok:12*pok+12]
        iprcp = self.prcp[12*pok:12*pok+12]
        igrad = self.grad[12*pok:12*pok+12]

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
        return mb_annual / sec_in_year / 900.