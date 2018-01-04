import numpy as np
from numpy import ix_
import xarray as xr

from oggm import cfg, utils
from oggm.cfg import RHO, G, N, SEC_IN_YEAR, SEC_IN_DAY


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
        self.mb_model = mb_model
        self.mb_elev_feedback = mb_elev_feedback
        _mb_call = None
        if mb_model:
            if mb_elev_feedback == 'always':
                _mb_call = mb_model.get_monthly_mb
            elif mb_elev_feedback == 'monthly':
                _mb_call = mb_model.get_monthly_mb
            elif mb_elev_feedback == 'annual':
                _mb_call = mb_model.get_annual_mb
            else:
                raise ValueError('mb_elev_feedback not understood')
        self._mb_call = _mb_call
        self._mb_current_date = None
        self._mb_current_out = None

        # Defaults
        if glen_a is None:
            glen_a = cfg.A
        self.glen_a = glen_a

        if dy is None:
            dy = dx

        self.dx = dx
        self.dy = dy
        self.dxdy = dx*dy

        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        self.ice_thick_filter = ice_thick_filter

        # Data
        self.bed_topo = bed_topo
        self.ice_thick = None
        self.reset_ice_thick(init_ice_thick)
        self.ny, self.nx = bed_topo.shape

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

    def run_until(self, y1):
        """Run until a selected year."""

        t = (y1-self.y0) * SEC_IN_YEAR
        while self.t < t:
            self.step(t-self.t)
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
                            print_stdout=False):
        """Run until a selected year and store the output in a NetCDF file."""

        yrs = np.arange(np.floor(self.yr), np.floor(ye) + 1, step)
        out_thick = np.zeros((len(yrs), self.ny, self.nx))
        for i, yr in enumerate(yrs):
            if print_stdout and (yr / 10) == int(yr / 10):
                print('{}: year {} of {}\r'.format(print_stdout,
                                                 int(yr), int(ye)))
            self.run_until(yr)
            out_thick[i, :, :] = self.ice_thick

        run_ds = grid.to_dataset() if grid else xr.Dataset()
        run_ds['ice_thickness'] = xr.DataArray(out_thick,
                                               dims=['time', 'y', 'x'],
                                               coords={'time': yrs})

        run_ds['bed_topo'] = xr.DataArray(self.bed_topo,
                                          dims=['y', 'x'])

        # write output?
        if run_path is not None:
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
        self.gamma = 2. * self.glen_a * (RHO * G) ** N / (N + 2)

        # forward time stepping stability criteria
        # default is just beyond R. Hindmarsh's idea of 1/2(n+1)
        self.cfl = cfl
        self.max_dt = max_dt
    
        # extend into 2D
        self.Lx = 0.5 * (self.nx - 1) * self.dx
        self.Ly = 0.5 * (self.ny - 1) * self.dy

        self.k = np.arange(0, self.ny)
        self.kp = np.hstack([np.arange(1, self.ny), self.ny - 1])
        self.km = np.hstack([0, np.arange(0, self.ny - 1)])
        self.l = np.arange(0, self.nx)
        self.lp = np.hstack([np.arange(1, self.nx), self.nx - 1])
        self.lm = np.hstack([0, np.arange(0, self.nx - 1)])
        self.H_upstream_up = np.zeros((self.ny, self.nx))
        self.H_upstream_dn = np.zeros((self.ny, self.nx))

    def diffusion_upstream_2d(self):
        # Builded upon the Eq. (62) with the term in y in the diffusivity.
        # It differs from diffusion_Upstream_2D_V1 only for the definition of
        # "s_grad", l282-283 & l305-306 in V1 and l355-356 & l379-380 in V2)

        H = self.ice_thick
        S = self.surface_h

        # --- all the l components
        H_l_up = 0.5 * (H[ix_(self.k, self.lp)] + H[ix_(self.k, self.l)])
        H_l_dn = 0.5 * (H[ix_(self.k, self.l)] + H[ix_(self.k, self.lm)])

        # applying Eq. (61) to the scheme
        Hl = H[ix_(self.k, self.l)]
        Hlp = H[ix_(self.k, self.lp)]
        Hlm = H[ix_(self.k, self.lm)]

        H_l_upstream_up = self.H_upstream_up
        gt = S[ix_(self.k, self.lp)] > S[ix_(self.k, self.l)]
        H_l_upstream_up[gt] = Hlp[gt]
        H_l_upstream_up[~gt] = Hl[~gt]

        H_l_upstream_dn = self.H_upstream_dn
        gt = S[ix_(self.k, self.l)] > S[ix_(self.k, self.lm)]
        H_l_upstream_dn[gt] = Hl[gt]
        H_l_upstream_dn[~gt] = Hlm[~gt]

        # applying Eq. (62) to the scheme
        s_l_grad_up = (((S[ix_(self.kp, self.l)] - S[ix_(self.km, self.l)] +
                         S[ix_(self.kp, self.lp)] - S[ix_(self.km, self.lp)])
                        ** 2. / (4 * self.dx) ** 2.) +
                       ((S[ix_(self.k, self.lp)] - S[ix_(self.k, self.l)])
                        ** 2. / self.dy ** 2.)) ** ((N - 1.) / 2.)
        s_l_grad_dn = (((S[ix_(self.kp, self.l)] - S[ix_(self.km, self.l)] +
                         S[ix_(self.kp, self.lm)] - S[ix_(self.km, self.lm)])
                        ** 2. / (4 * self.dx) ** 2.) +
                       ((S[ix_(self.k, self.l)] - S[ix_(self.k, self.lm)])
                        ** 2. / self.dy ** 2.)) ** ((N - 1.) / 2.)

        D_l_up = self.gamma * H_l_up ** (N + 1) * H_l_upstream_up * s_l_grad_up
        D_l_dn = self.gamma * H_l_dn ** (N + 1) * H_l_upstream_dn * s_l_grad_dn

        # --- all the k components
        H_k_up = 0.5 * (H[ix_(self.kp, self.l)] + H[ix_(self.k, self.l)])
        H_k_dn = 0.5 * (H[ix_(self.k, self.l)] + H[ix_(self.km, self.l)])

        # applying Eq. (61) to the scheme
        Hk = H[ix_(self.k, self.l)]
        Hkp = H[ix_(self.kp, self.l)]
        Hkm = H[ix_(self.km, self.l)]

        H_k_upstream_up = self.H_upstream_up
        gt = S[ix_(self.kp, self.l)] > S[ix_(self.k, self.l)]
        H_k_upstream_up[gt] = Hkp[gt]
        H_k_upstream_up[~gt] = Hk[~gt]

        H_k_upstream_dn = self.H_upstream_dn
        gt = S[ix_(self.k, self.l)] > S[ix_(self.km, self.l)]
        H_k_upstream_dn[gt] = Hk[gt]
        H_k_upstream_dn[~gt] = Hkm[~gt]

        # applying Eq. (62) to the scheme
        s_k_grad_up = (((S[ix_(self.k, self.lp)] - S[ix_(self.k, self.lm)] +
                         S[ix_(self.kp, self.lp)] - S[ix_(self.kp, self.lm)])
                        ** 2. / (4 * self.dy) ** 2.) +
                       ((S[ix_(self.kp, self.l)] - S[ix_(self.k, self.l)])
                        ** 2. / self.dx ** 2.))  ** ((N - 1.) / 2.)
        s_k_grad_dn = (((S[ix_(self.k, self.lp)] - S[ix_(self.k, self.lm)] +
                         S[ix_(self.km, self.lp)] - S[ix_(self.km, self.lm)])
                        ** 2. / (4 * self.dy) ** 2.) +
                       ((S[ix_(self.k, self.l)] - S[ix_(self.km, self.l)])
                        ** 2. / self.dx ** 2.)) ** ((N - 1.) / 2.)

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
        div_k = (D_k_up * (S[ix_(self.kp, self.l)] - S[ix_(self.k, self.l)]) /
                 self.dy - D_k_dn * (S[ix_(self.k, self.l)] -
                                     S[ix_(self.km, self.l)]) /
                 self.dy) / self.dy
        div_l = (D_l_up * (S[ix_(self.k, self.lp)] - S[ix_(self.k, self.l)]) /
                 self.dx - D_l_dn * (S[ix_(self.k, self.l)] -
                                     S[ix_(self.k, self.lm)]) /
                 self.dx) / self.dx

        div_back = div_l + div_k

        return div_back, dt_cfl

    def step(self, dt):
        """Advance one step."""

        div_q, dt_cfl = self.diffusion_upstream_2d()

        dt_use = np.clip(np.min([dt_cfl, dt]), 0, self.max_dt)

        self.ice_thick = (self.surface_h + (self.get_mb() + div_q) * dt_use -
                          self.bed_topo).clip(0)

        # Next step
        self.t += dt_use
        return dt
