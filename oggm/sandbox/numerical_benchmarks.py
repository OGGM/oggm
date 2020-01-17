from oggm import utils, cfg
from oggm.core.massbalance import RandomMassBalance, MultipleFlowlineMassBalance
from oggm.core.flowline import (robust_model_run, simple_model_run,
                                FluxBasedModel, FlowlineModel)
from oggm.exceptions import InvalidParamsError
from functools import partial
import logging
import numpy as np

# Module logger
log = logging.getLogger(__name__)


class NewModel(FlowlineModel):
    """The flowline model used by OGGM in production.

    It solves for the SIA along the flowline(s) using a staggered grid. It
    computes the *ice flux* between grid points and transports the mass
    accordingly (also between flowlines).

    This model is numerically less stable than fancier schemes, but it
    is fast and works with multiple flowlines of any bed shape (rectangular,
    parabolic, trapeze, and any combination of them).

    We test that it conserves mass in most cases, but not on very stiff cliffs.
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=None,
                 min_dt=None, **kwargs):
        """Instanciate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBakanceModel
            the mass-balance model
        y0 : int
            initial year of the simulation
        glen_a : float
            Glen's creep parameter
        fs : float
            Oerlemans sliding parameter
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        fixed_dt : float
            set to a value (in seconds) to prevent adaptive time-stepping.
        cfl_number : float
            for adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx / max_u).
            To choose the "best" CFL number we would need a stability
            analysis - we used an empirical analysis (see blog post) and
            settled on 0.02.
            Defaults to cfg.PARAMS['cfl_number']
        min_dt : float
            with high velocities, time steps can become very small and your
            model might run very slowly. In production it might be useful to
            set a limit below which the model will just error.
            Defaults to cfg.PARAMS['cfl_min_dt']
        is_tidewater: bool, default: False
            use the very basic parameterization for tidewater glaciers
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        """
        super(NewModel, self).__init__(flowlines, mb_model=mb_model,
                                             y0=y0, glen_a=glen_a, fs=fs,
                                             inplace=inplace, **kwargs)

        self.fixed_dt = fixed_dt
        self.min_dt = min_dt
        self.cfl_number = cfl_number
        self.calving_m3_since_y0 = 0.  # total calving since time y0

        # Do we want to use shape factors?
        self.sf_func = None
        # Use .get to obtain default None for non-existing key
        # necessary to pass some tests
        # TODO: change to direct dictionary query after tests are adapted?
        use_sf = cfg.PARAMS.get('use_shape_factor_for_fluxbasedmodel')
        if use_sf == 'Adhikari' or use_sf == 'Nye':
            self.sf_func = utils.shape_factor_adhikari
        elif use_sf == 'Huss':
            self.sf_func = utils.shape_factor_huss

        # Optim
        self.slope_stag = []
        self.thick_stag = []
        self.section_stag = []
        self.u_stag = []
        self.shapefac_stag = []
        self.flux_stag = []
        self.trib_flux = []
        for fl, trib in zip(self.fls, self._tributary_indices):
            nx = fl.nx
            # This is not staggered
            self.trib_flux.append(np.zeros(nx))
            # We add an additional fake grid point at the end of these
            if (trib[0] is not None) or self.is_tidewater:
                nx = fl.nx + 1
            # +1 is for the staggered grid
            self.slope_stag.append(np.zeros(nx+1))
            self.thick_stag.append(np.zeros(nx+1))
            self.section_stag.append(np.zeros(nx+1))
            self.u_stag.append(np.zeros(nx+1))
            self.shapefac_stag.append(np.ones(nx+1))  # beware the ones!
            self.flux_stag.append(np.zeros(nx+1))

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # Loop over tributaries to determine the flux rate
        for fl_id, fl in enumerate(self.fls):

            # This is possibly less efficient than zip() but much clearer
            trib = self._tributary_indices[fl_id]
            slope_stag = self.slope_stag[fl_id]
            thick_stag = self.thick_stag[fl_id]
            section_stag = self.section_stag[fl_id]
            sf_stag = self.shapefac_stag[fl_id]
            flux_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            u_stag = self.u_stag[fl_id]

            # Flowline state
            surface_h = fl.surface_h
            thick = fl.thick
            section = fl.section
            dx = fl.dx_meter

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid point
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])
            elif self.is_tidewater:
                # For tidewater glacier, we trick and set the outgoing thick
                # to zero (for numerical stability and this should quite OK
                # represent what happens at the calving tongue)
                surface_h = np.append(surface_h, surface_h[-1] - thick[-1])
                thick = np.append(thick, 0)
                section = np.append(section, 0)

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Staggered thick
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            if self.sf_func is not None:
                # TODO: maybe compute new shape factors only every year?
                sf = self.sf_func(fl.widths_m, fl.thick, fl.is_rectangular)
                if is_trib or self.is_tidewater:
                    # for water termination or inflowing tributary, the sf
                    # makes no sense
                    sf = np.append(sf, 1.)
                sf_stag[1:-1] = (sf[0:-1] + sf[1:]) / 2.
                sf_stag[[0, -1]] = sf[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            # _fd = 2/(N+2) * self.glen_a
            N = self.glen_n
            rhogh = (self.rho*cfg.G*slope_stag)**N
            u_stag[:] = (thick_stag**(N+1)) * self._fd * rhogh * sf_stag**N + \
                        (thick_stag**(N-1)) * self.fs * rhogh

            # Staggered section
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # CFL condition
            if not self.fixed_dt:
                maxu = np.max(np.abs(u_stag))
                if maxu > 0.:
                    cfl_dt = self.cfl_number * dx / maxu
                else:
                    cfl_dt = dt

                # Update dt only if necessary
                if cfl_dt < dt:
                    dt = cfl_dt
                    if cfl_dt < self.min_dt:
                        raise RuntimeError(
                            'CFL error: required time step smaller '
                            'than the minimum allowed: '
                            '{:.1f}s vs {:.1f}s.'.format(cfl_dt, self.min_dt))

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt

        # A second loop for the mass exchange
        for fl_id, fl in enumerate(self.fls):

            flx_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            tr = self._tributary_indices[fl_id]

            dx = fl.dx_meter

            is_trib = tr[0] is not None
            # For these we had an additional grid point
            if is_trib or self.is_tidewater:
                flx_stag = flx_stag[:-1]

            # Mass balance
            widths = fl.widths_m
            mb = self.get_mb(fl.surface_h, self.yr, fl_id=fl_id)
            # Allow parabolic beds to grow
            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)

            # Update section with ice flow and mass balance
            new_section = (fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt/dx +
                           trib_flux*dt/dx + mb)

            # Keep positive values only and store
            fl.section = utils.clip_min(new_section, 0)

            # Add the last flux to the tributary
            # this works because the lines are sorted in order
            if is_trib:
                # tr tuple: line_index, start, stop, gaussian_kernel
                self.trib_flux[tr[0]][tr[1]:tr[2]] += \
                    utils.clip_min(flx_stag[-1], 0) * tr[3]
            elif self.is_tidewater:
                # -2 because the last flux is zero per construction
                # not sure at all if this is the way to go but mass
                # conservation is OK
                self.calving_m3_since_y0 += utils.clip_min(flx_stag[-2], 0)*dt

        # Next step
        self.t += dt
        return dt

    def get_diagnostics(self, fl_id=-1):
        """Obtain model diagnostics in a pandas DataFrame.

        Parameters
        ----------
        fl_id : int
            the index of the flowline of interest, from 0 to n_flowline-1.
            Default is to take the last (main) one

        Returns
        -------
        a pandas DataFrame, which index is distance along flowline (m). Units:
            - surface_h, bed_h, ice_tick, section_width: m
            - section_area: m2
            - slope: -
            - ice_flux, tributary_flux: m3 of *ice* per second
            - ice_velocity: m per second (depth-section integrated)
        """

        import pandas as pd

        fl = self.fls[fl_id]
        nx = fl.nx

        df = pd.DataFrame(index=fl.dx_meter * np.arange(nx))
        df.index.name = 'distance_along_flowline'
        df['surface_h'] = fl.surface_h
        df['bed_h'] = fl.bed_h
        df['ice_thick'] = fl.thick
        df['section_width'] = fl.widths_m
        df['section_area'] = fl.section

        # Staggered
        var = self.slope_stag[fl_id]
        df['slope'] = (var[1:nx+1] + var[:nx])/2
        var = self.flux_stag[fl_id]
        df['ice_flux'] = (var[1:nx+1] + var[:nx])/2
        var = self.u_stag[fl_id]
        df['ice_velocity'] = (var[1:nx+1] + var[:nx])/2
        var = self.shapefac_stag[fl_id]
        df['shape_fac'] = (var[1:nx+1] + var[:nx])/2

        # Not Staggered
        df['tributary_flux'] = self.trib_flux[fl_id]

        return df


@utils.entity_task(log)
def default_run(gdir, nyears=300, y0=None, temp_bias=None, seed=None,
                output_filesuffix=None, flux_limiter=False):
    """A simple doc"""

    bias = 0 if y0 is None else None
    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     bias=bias, y0=y0, seed=seed)
    if temp_bias is not None:
        mb.temp_bias = temp_bias
    robust_model_run(gdir, output_filesuffix=output_filesuffix,
                     mb_model=mb, ys=0, ye=nyears, flux_limiter=flux_limiter)


@utils.entity_task(log)
def better_run(gdir, nyears=300, y0=None, temp_bias=None, seed=None,
               output_filesuffix=None, min_dt=0, cfl_number=0.05):
    """A simple doc"""

    bias = 0 if y0 is None else None
    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     bias=bias, y0=y0, seed=seed)
    if temp_bias is not None:
        mb.temp_bias = temp_bias

    supermodel = partial(NewModel, monthy_steps=True, min_dt=min_dt,
                         cfl_number=cfl_number)
    supermodel.__name__ = 'Model' + output_filesuffix
    simple_model_run(gdir, output_filesuffix=output_filesuffix,
                     mb_model=mb, ys=0, ye=nyears,
                     numerical_model_class=supermodel)
