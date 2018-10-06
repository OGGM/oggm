import numpy as np
from numpy import ix_
import xarray as xr
import os

from oggm import cfg, utils
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY


def filter_ice_border(ice_thick):
    """Sets the ice thickness at the border of the domain to zero."""
    ice_thick[0, :] = 0
    ice_thick[-1, :] = 0
    ice_thick[:, 0] = 0
    ice_thick[:, -1] = 0
    return ice_thick


class Model2D(object):
    """Interface to a distributed model"""

    def __init__(self, bed_topo, init_ice_thick=None, dx=None, dy=None,
                 mb_model=None, y0=0., glen_a=None, mb_elev_feedback='annual',
                 ice_thick_filter=filter_ice_border):
        """Create a new 2D model from gridded data.

        Parameters
        ----------
        bed_topo : 2d array
            the topography
        init_ice_thick : 2d array (optional)
            the initial ice thickness (default is zero everywhere)
        dx : float
            map resolution (m)
        dy : float
            map resolution (m)
        mb_model : oggm.core.massbalance model
            the mass-balance model to use for the simulation
        y0 : int
            the starting year
        glen_a : float
            Glen's flow law parameter A
        mb_elev_feedback : str (default: 'annual')
            when to update the mass-balance model ('annual', 'monthly', or
            'always')
        ice_thick_filter : func
            function to apply to the ice thickness *after* each time step.
            See filter_ice_border for an example. Set to None for doing nothing
        """

        # Mass balance
        self.mb_elev_feedback = mb_elev_feedback
        self.mb_model = mb_model

        # Defaults
        if glen_a is None:
            glen_a = cfg.PARAMS['glen_a']
        self.glen_a = glen_a

        if dy is None:
            dy = dx

        self.dx = dx
        self.dy = dy
        self.dxdy = dx * dy

        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        self.ice_thick_filter = ice_thick_filter

        # Data
        self.bed_topo = bed_topo
        self.ice_thick = None
        self.reset_ice_thick(init_ice_thick)
        self.ny, self.nx = bed_topo.shape

    @property
    def mb_model(self):
        return self._mb_model

    @mb_model.setter
    def mb_model(self, value):
        # We need a setter because the MB func is stored as an attr too
        _mb_call = None
        if value:
            if self.mb_elev_feedback in ['always', 'monthly']:
                _mb_call = value.get_monthly_mb
            elif self.mb_elev_feedback in ['annual', 'never']:
                _mb_call = value.get_annual_mb
            else:
                raise ValueError('mb_elev_feedback not understood')
        self._mb_model = value
        self._mb_call = _mb_call
        self._mb_current_date = None
        self._mb_current_out = dict()
        self._mb_current_heights = dict()

    def reset_y0(self, y0):
        """Reset the initial model time"""
        self.y0 = y0
        self.t = 0

    def reset_ice_thick(self, ice_thick=None):
        """Reset the ice thickness"""
        if ice_thick is None:
            ice_thick = self.bed_topo * 0.
        self.ice_thick = ice_thick.copy()

    @property
    def yr(self):
        return self.y0 + self.t / SEC_IN_YEAR

    @property
    def area_m2(self):
        return np.sum(self.ice_thick > 0) * self.dxdy

    @property
    def volume_m3(self):
        return np.sum(self.ice_thick * self.dxdy)

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def surface_h(self):
        return self.bed_topo + self.ice_thick

    def get_mb(self, year=None):
        """Get the mass balance at the requested height and time.

        Optimized so that no mb model call is necessary at each step.
        """

        if year is None:
            year = self.yr

        # Do we have to optimise?
        if self.mb_elev_feedback == 'always':
            return self._mb_call(self.bed_topo + self.ice_thick, year)

        date = utils.floatyear_to_date(year)
        if self.mb_elev_feedback == 'annual':
            # ignore month changes
            date = (date[0], date[0])

        if self._mb_current_date != date or (self._mb_current_out is None):
            # We need to reset all
            self._mb_current_date = date
            _mb = self._mb_call(self.surface_h.flatten(), year)
            self._mb_current_out = _mb.reshape((self.ny, self.nx))

        return self._mb_current_out

    def step(self, dt):
        """Advance one step."""
        raise NotImplementedError

    def run_until(self, y1, stop_if_border=False):
        """Run until a selected year."""

        t = (y1 - self.y0) * SEC_IN_YEAR
        while self.t < t:
            self.step(t - self.t)
            if stop_if_border:
                if (np.any(self.ice_thick[0, :] > 10) or
                        np.any(self.ice_thick[-1, :] > 10) or
                        np.any(self.ice_thick[:, 0] > 10) or
                        np.any(self.ice_thick[:, -1] > 10)):
                    raise RuntimeError('Glacier exceeds boundaries')
            if self.ice_thick_filter is not None:
                self.ice_thick = self.ice_thick_filter(self.ice_thick)

        if np.any(~np.isfinite(self.ice_thick)):
            raise FloatingPointError('NaN in numerical solution.')

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
        """Run until an equuilibrium is reached (can take a while)."""

        ite = 0
        was_close_zero = 0
        t_rate = 1
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            ite += 1
            v_bef = self.volume_m3
            self.run_until(self.yr + ystep)
            v_af = self.volume_m3
            if np.isclose(v_bef, 0., atol=1):
                t_rate = 1
                was_close_zero += 1
            else:
                t_rate = np.abs(v_af - v_bef) / v_bef
        if ite > max_ite:
            raise RuntimeError('Did not find equilibrium.')

    def run_until_and_store(self, ye, step=2, run_path=None, grid=None,
                            print_stdout=False, stop_if_border=False):
        """Run until a selected year and store the output in a NetCDF file."""

        yrs = np.arange(np.floor(self.yr), np.floor(ye) + 1, step)
        out_thick = np.zeros((len(yrs), self.ny, self.nx))
        for i, yr in enumerate(yrs):
            if print_stdout and (yr / 10) == int(yr / 10):
                print('{}: year {} of {}, '
                      'max thick {:.1f}m\r'.format(print_stdout,
                                                   int(yr),
                                                   int(ye),
                                                   self.ice_thick.max()))
            self.run_until(yr, stop_if_border=stop_if_border)
            out_thick[i, :, :] = self.ice_thick

        run_ds = grid.to_dataset() if grid else xr.Dataset()
        run_ds['ice_thickness'] = xr.DataArray(out_thick,
                                               dims=['time', 'y', 'x'],
                                               coords={'time': yrs})

        run_ds['bed_topo'] = xr.DataArray(self.bed_topo,
                                          dims=['y', 'x'])

        # write output?
        if run_path is not None:
            if os.path.exists(run_path):
                os.remove(run_path)
            run_ds.to_netcdf(run_path)

        return run_ds


class Upstream2D(Model2D):
    """Actual model"""

    def __init__(self, bed_topo, init_ice_thick=None, dx=None,
                 mb_model=None, y0=0., glen_a=None, mb_elev_feedback='annual',
                 cfl=0.124, max_dt=31*SEC_IN_DAY,
                 ice_thick_filter=filter_ice_border):
        """Create a new 2D model from gridded data.

        Parameters
        ----------
        bed_topo : 2d array
            the topography
        init_ice_thick : 2d array (optional)
            the initial ice thickness (default is zero everywhere)
        dx : float
            map resolution (m)
        dy : float
            map resolution (m)
        mb_model : oggm.core.massbalance model
            the mass-balance model to use for the simulation
        y0 : int
            the starting year
        glen_a : float
            Glen's flow law parameter A
        mb_elev_feedback : str (default: 'annual')
            when to update the mass-balance model ('annual', 'monthly', or
            'always')
        cfl : float (default:0.124)
            forward time stepping stability criteria. Default is just beyond
            R. Hindmarsh's idea of 1/2(n+1).
        max_dt : int (default: 31 days)
            maximum allow time step (in seconds). Useful because otherwise the
            automatic time step can be quite ambitious.
        ice_thick_filter : func
            function to apply to the ice thickness *after* each time step.
            See filter_ice_border for an example. Set to None for doing nothing
        """
        super(Upstream2D, self).__init__(bed_topo,
                                         init_ice_thick=init_ice_thick,
                                         dx=dx, mb_model=mb_model, y0=y0,
                                         glen_a=glen_a,
                                         mb_elev_feedback=mb_elev_feedback,
                                         ice_thick_filter=ice_thick_filter)

        # We introduce Gamma to shorten the equations
        self.rho = cfg.PARAMS['ice_density']
        self.glen_n = cfg.PARAMS['glen_n']
        self.gamma = (2. * self.glen_a * (self.rho * G) ** self.glen_n
                      / (self.glen_n + 2))

        # forward time stepping stability criteria
        # default is just beyond R. Hindmarsh's idea of 1/2(n+1)
        self.cfl = cfl
        self.max_dt = max_dt

        # extend into 2D
        self.Lx = 0.5 * (self.nx - 1) * self.dx
        self.Ly = 0.5 * (self.ny - 1) * self.dy

        # Some indices
        self.k = np.arange(0, self.ny)
        self.kp = np.hstack([np.arange(1, self.ny), self.ny - 1])
        self.km = np.hstack([0, np.arange(0, self.ny - 1)])
        self.l = np.arange(0, self.nx)  # flake8: noqa E741
        self.lp = np.hstack([np.arange(1, self.nx), self.nx - 1])
        self.lm = np.hstack([0, np.arange(0, self.nx - 1)])
        self.H_upstream_up = np.zeros((self.ny, self.nx))
        self.H_upstream_dn = np.zeros((self.ny, self.nx))

        # Easy optimisation
        self._ixklp = ix_(self.k, self.lp)
        self._ixkl = ix_(self.k, self.l)
        self._ixklm = ix_(self.k, self.lm)
        self._ixkpl = ix_(self.kp, self.l)
        self._ixkml = ix_(self.km, self.l)
        self._ixkplp = ix_(self.kp, self.lp)
        self._ixkplm = ix_(self.kp, self.lm)
        self._ixkmlm = ix_(self.km, self.lm)
        self._ixkmlp = ix_(self.km, self.lp)

    def diffusion_upstream_2d(self):
        # Builded upon the Eq. (62) with the term in y in the diffusivity.
        # It differs from diffusion_Upstream_2D_V1 only for the definition of
        # "s_grad", l282-283 & l305-306 in V1 and l355-356 & l379-380 in V2)

        H = self.ice_thick
        S = self.surface_h
        N = self.glen_n

        # Optim
        S_ixklp = S[self._ixklp]
        S_ixkl = S[self._ixkl]
        S_ixklm = S[self._ixklm]
        S_ixkml = S[self._ixkml]
        S_ixkpl = S[self._ixkpl]
        S_ixkplp = S[self._ixkplp]
        S_ixkplm = S[self._ixkplm]
        S_ixkmlm = S[self._ixkmlm]
        S_ixkmlp = S[self._ixkmlp]

        Hl = H[self._ixkl]
        Hlp = H[self._ixklp]
        Hlm = H[self._ixklm]
        Hk = Hl
        Hkp = H[self._ixkpl]
        Hkm = H[self._ixkml]

        # --- all the l components

        # applying Eq. (61) to the scheme
        H_l_up = 0.5 * (Hlp + Hl)
        H_l_dn = 0.5 * (Hl + Hlm)

        H_l_upstream_up = self.H_upstream_up
        gt = S_ixklp > S_ixkl
        H_l_upstream_up[gt] = Hlp[gt]
        H_l_upstream_up[~gt] = Hl[~gt]

        H_l_upstream_dn = self.H_upstream_dn
        gt = S_ixkl > S_ixklm
        H_l_upstream_dn[gt] = Hl[gt]
        H_l_upstream_dn[~gt] = Hlm[~gt]

        # applying Eq. (62) to the scheme
        S_diff = S_ixkpl - S_ixkml
        S_lpdiff = S_ixklp - S_ixkl
        S_lmdiff = S_ixkl - S_ixklm
        s_l_grad_up = (((S_diff + S_ixkplp - S_ixkmlp)
                        ** 2. / (4 * self.dx) ** 2.) +
                       (S_lpdiff ** 2. / self.dy ** 2.)) ** ((N - 1.) / 2.)
        s_l_grad_dn = (((S_diff + S_ixkplm - S_ixkmlm)
                        ** 2. / (4 * self.dx) ** 2.) +
                       (S_lmdiff ** 2. / self.dy ** 2.)) ** ((N - 1.) / 2.)

        D_l_up = self.gamma * H_l_up ** (N + 1) * H_l_upstream_up * s_l_grad_up
        D_l_dn = self.gamma * H_l_dn ** (N + 1) * H_l_upstream_dn * s_l_grad_dn

        # --- all the k components

        # applying Eq. (61) to the scheme
        H_k_up = 0.5 * (Hkp + Hl)
        H_k_dn = 0.5 * (Hl + Hkm)

        H_k_upstream_up = self.H_upstream_up
        gt = S_ixkpl > S_ixkl
        H_k_upstream_up[gt] = Hkp[gt]
        H_k_upstream_up[~gt] = Hk[~gt]

        H_k_upstream_dn = self.H_upstream_dn
        gt = S_ixkl > S_ixkml
        H_k_upstream_dn[gt] = Hk[gt]
        H_k_upstream_dn[~gt] = Hkm[~gt]

        # applying Eq. (62) to the scheme
        S_diff = S_ixklp - S_ixklm
        S_kpdiff = S_ixkpl - S_ixkl
        S_kmdiff = S_ixkl - S_ixkml
        s_k_grad_up = (((S_diff + S_ixkplp - S_ixkplm)
                        ** 2. / (4 * self.dy) ** 2.) +
                       (S_kpdiff ** 2. / self.dx ** 2.)) ** ((N - 1.) / 2.)
        s_k_grad_dn = (((S_diff + S_ixkmlp - S_ixkmlm)
                        ** 2. / (4 * self.dy) ** 2.) +
                       (S_kmdiff ** 2. / self.dx ** 2.)) ** ((N - 1.) / 2.)

        D_k_up = self.gamma * H_k_up ** (N + 1) * H_k_upstream_up * s_k_grad_up
        D_k_dn = self.gamma * H_k_dn ** (N + 1) * H_k_upstream_dn * s_k_grad_dn

        # --- Check the cfl condition
        divisor = max(max(np.max(np.abs(D_k_up)), np.max(np.abs(D_k_dn))),
                      max(np.max(np.abs(D_l_up)), np.max(np.abs(D_l_dn))))
        if divisor == 0:
            dt_cfl = self.max_dt
        else:
            dt_cfl = (self.cfl * min(self.dx ** 2., self.dy ** 2.) / divisor)

        # --- Calculate Final diffusion term
        div_k = (D_k_up * S_kpdiff / self.dy -
                 D_k_dn * S_kmdiff / self.dy) / self.dy
        div_l = (D_l_up * S_lpdiff / self.dx -
                 D_l_dn * S_lmdiff / self.dx) / self.dx

        return div_l + div_k, dt_cfl

    def step(self, dt):
        """Advance one step."""

        div_q, dt_cfl = self.diffusion_upstream_2d()

        dt_use = np.clip(np.min([dt_cfl, dt]), 0, self.max_dt)

        self.ice_thick = (self.surface_h + (self.get_mb() + div_q) * dt_use -
                          self.bed_topo).clip(0)

        # Next step
        self.t += dt_use
        return dt
