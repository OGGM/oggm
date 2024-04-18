# Builtins
import logging
import copy
from functools import partial
import warnings

# External libs
import numpy as np
import pandas as pd

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm.exceptions import InvalidParamsError
from oggm.core.inversion import find_sia_flux_from_thickness
from oggm.core.flowline import FlowlineModel, flux_gate_with_build_up

# Constants
from oggm.cfg import G

# Module logger
log = logging.getLogger(__name__)


def initialize_calving_params():
    """Initialize the parameters for the calving model.

    This should be part of params.cfg but this is still
    sandboxed...
    """

    # ocean water density in kg m-3; should be >= ice density
    # for lake-terminating glaciers this could be changed to
    # 1000 kg m-3
    if 'ocean_density' not in cfg.PARAMS:
        cfg.PARAMS['ocean_density'] = 1028

    # Stretch distance in hydrostatic pressure balance
    # calculations for terminal water-terminating cliffs
    # (in meters) - See Malles et al.
    # (the current value of 8000m might be too high)
    if 'max_calving_stretch_distance' not in cfg.PARAMS:
        cfg.PARAMS['max_calving_stretch_distance'] = 8000


class CalvingFluxBasedModelJan(FlowlineModel):
    """Jan Malles' implementation of calving as in his 2023 paper.
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=None,
                 min_dt=None, flux_gate_thickness=None,
                 flux_gate=None, flux_gate_build_up=100,
                 do_kcalving=None, calving_k=None,
                 water_level=None, **kwargs):
        """Instantiate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass balance model
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
            Defaults to cfg.PARAMS['cfl_number'].
            For adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx / max_u).
            To choose the "best" CFL number we would need a stability
            analysis - we used an empirical analysis (see blog post) and
            settled on 0.02 for the default cfg.PARAMS['cfl_number'].
        min_dt : float
            Defaults to cfg.PARAMS['cfl_min_dt'].
            At high velocities, time steps can become very small and your
            model might run very slowly. In production, it might be useful to
            set a limit below which the model will just error.
        is_tidewater: bool, default: False
            is this a tidewater glacier?
        is_lake_terminating: bool, default: False
            is this a lake terminating glacier?
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        flux_gate_thickness : float or array
            flux of ice from the left domain boundary (and tributaries).
            Units of m of ice thickness. Note that unrealistic values won't be
            met by the model, so this is really just a rough guidance.
            It's better to use `flux_gate` instead.
        flux_gate : float or function or array of floats or array of functions
            flux of ice from the left domain boundary (and tributaries)
            (unit: m3 of ice per second). If set to a high value, consider
            changing the flux_gate_buildup time. You can also provide
            a function (or an array of functions) returning the flux
            (unit: m3 of ice per second) as a function of time.
            This is overridden by `flux_gate_thickness` if provided.
        flux_gate_buildup : int
            number of years used to build up the flux gate to full value
        do_kcalving : bool
            switch on the k-calving parameterisation. Ignored if not a
            tidewater glacier. Use the option from PARAMS per default
        calving_k : float
            the calving proportionality constant (units: yr-1). Use the
            one from PARAMS per default
        water_level : float
            the water level. It should be zero m a.s.l, but:
            - sometimes the frontal elevation is unrealistically high (or low).
            - lake terminating glaciers
            - other uncertainties
            The default is 0. For lake terminating glaciers,
            it is inferred from PARAMS['free_board_lake_terminating'].
            The best way to set the water level for real glaciers is to use
            the same as used for the inversion (this is what
            `flowline_model_run` does for you)
        """
        super(CalvingFluxBasedModelJan, self).__init__(flowlines, mb_model=mb_model,
                                                       y0=y0, glen_a=glen_a, fs=fs,
                                                       inplace=inplace,
                                                       water_level=water_level,
                                                       **kwargs)

        # Initialize the parameters
        initialize_calving_params()

        self.fixed_dt = fixed_dt
        if min_dt is None:
            min_dt = cfg.PARAMS['cfl_min_dt']
        if cfl_number is None:
            cfl_number = cfg.PARAMS['cfl_number']
        self.min_dt = min_dt
        self.cfl_number = cfl_number

        # Do we want to use shape factors?
        self.sf_func = None
        use_sf = cfg.PARAMS.get('use_shape_factor_for_fluxbasedmodel')
        if use_sf == 'Adhikari' or use_sf == 'Nye':
            self.sf_func = utils.shape_factor_adhikari
        elif use_sf == 'Huss':
            self.sf_func = utils.shape_factor_huss

        # Calving params
        if do_kcalving is None:
            do_kcalving = cfg.PARAMS['use_kcalving_for_run']
        self.do_calving = do_kcalving and self.is_tidewater
        if calving_k is None:
            calving_k = cfg.PARAMS['calving_k']
        if do_kcalving:
            self.calving_k = calving_k / cfg.SEC_IN_YEAR

        self.ovars = cfg.PARAMS['store_diagnostic_variables']

        # Stretching distance (or stress coupling length) for frontal dynamics
        if self.do_calving:
            self.stretch_dist_p = cfg.PARAMS['max_calving_stretch_distance']

        # Flux gate
        self.flux_gate = utils.tolist(flux_gate, length=len(self.fls))
        self.flux_gate_m3_since_y0 = 0.
        if flux_gate_thickness is not None:
            # Compute the theoretical ice flux from the slope at the top
            flux_gate_thickness = utils.tolist(flux_gate_thickness,
                                               length=len(self.fls))
            self.flux_gate = []
            for fl, fgt in zip(self.fls, flux_gate_thickness):
                # We set the thickness to the desired value so that
                # the widths work ok
                fl = copy.deepcopy(fl)
                fl.thick = fl.thick * 0 + fgt
                slope = (fl.surface_h[0] - fl.surface_h[1]) / fl.dx_meter
                if slope == 0:
                    raise ValueError('I need a slope to compute the flux')
                flux = find_sia_flux_from_thickness(slope,
                                                    fl.widths_m[0],
                                                    fgt,
                                                    shape=fl.shape_str[0],
                                                    glen_a=self.glen_a,
                                                    fs=self.fs)
                self.flux_gate.append(flux)

        # convert the floats to function calls
        for i, fg in enumerate(self.flux_gate):
            if fg is None:
                continue
            try:
                # Do we have a function? If yes all good
                fg(self.yr)
            except TypeError:
                # If not, make one
                self.flux_gate[i] = partial(flux_gate_with_build_up,
                                            flux_value=fg,
                                            flux_gate_yr=(flux_gate_build_up +
                                                          self.y0))

        # Special output
        self._surf_vel_fac = (self.glen_n + 2) / (self.glen_n + 1)

        # Optim
        self.slope_stag = []
        self.thick_stag = []
        self.section_stag = []
        self.depth_stag = []
        self.u_drag = []
        self.u_slide = []
        self.u_stag = []
        self.shapefac_stag = []
        self.flux_stag = []
        self.trib_flux = []
        for fl, trib in zip(self.fls, self._tributary_indices):
            nx = fl.nx
            # This is not staggered
            self.trib_flux.append(np.zeros(nx))
            # We add an fake grid point at the end of tributaries
            if trib[0] is not None:
                nx = fl.nx + 1
            # +1 is for the staggered grid
            self.slope_stag.append(np.zeros(nx+1))
            self.thick_stag.append(np.zeros(nx+1))
            self.section_stag.append(np.zeros(nx+1))
            self.depth_stag.append(np.zeros(nx+1))
            self.u_stag.append(np.zeros(nx+1))
            self.u_drag.append(np.zeros(nx+1))
            self.u_slide.append(np.zeros(nx+1))
            self.shapefac_stag.append(np.ones(nx+1))  # beware the ones!
            self.flux_stag.append(np.zeros(nx+1))

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # Simple container
        mbs = []

        # Loop over tributaries to determine the flux rate
        for fl_id, fl in enumerate(self.fls):

            # This is possibly less efficient than zip() but much clearer
            trib = self._tributary_indices[fl_id]
            slope_stag = self.slope_stag[fl_id]
            thick_stag = self.thick_stag[fl_id]
            section_stag = self.section_stag[fl_id]
            depth_stag = self.depth_stag[fl_id]
            sf_stag = self.shapefac_stag[fl_id]
            flux_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            u_stag = self.u_stag[fl_id]
            u_drag = self.u_drag[fl_id]
            u_slide = self.u_slide[fl_id]
            flux_gate = self.flux_gate[fl_id]

            # Flowline state
            surface_h = fl.surface_h
            thick = fl.thick
            width = fl.widths_m
            section = fl.section
            dx = fl.dx_meter
            depth = utils.clip_min(0, self.water_level - fl.bed_h)

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid point
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])
                width = np.append(width, width[-1])
                depth = np.append(depth, depth[-1])

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            depth_stag[1:-1] = (depth[0:-1] + depth[1:]) / 2.
            depth_stag[[0, -1]] = depth[[0, -1]]

            # Resetting variables (necessary?)
            rho_ocean = cfg.PARAMS['ocean_density']
            h = []
            d = []
            no_ice = []
            last_ice = []
            last_above_wl = []
            has_ice = []
            bed_below_wl = []
            ice_above_wl = []
            ice_below_wl = []
            below_wl = []
            detached = []
            add_calving = []
            self.calving_flux = 0.
            self.discharge = 0.

            A = self.glen_a
            N = self.glen_n

            if self.sf_func is not None:
                # TODO: maybe compute new shape factors only every year?
                sf = self.sf_func(fl.widths_m, fl.thick, fl.is_rectangular)
                if is_trib:
                    # for inflowing tributary, the sf makes no sense
                    sf = np.append(sf, 1.)
                sf_stag[1:-1] = (sf[0:-1] + sf[1:]) / 2.
                sf_stag[[0, -1]] = sf[[0, -1]]

            # Determine where ice bodies are; if there is no ice below
            # water_level, we fall back to the standard ice dynamics
            ice_below_wl = (fl.bed_h < self.water_level) & (fl.thick > 0)

            # We compute more complex dynamics when we have ice below water
            if fl.has_ice() and np.any(ice_below_wl) and self.do_calving:
                ice_above_wl = ((fl.bed_h < self.water_level) &
                                (fl.thick >= (rho_ocean / self.rho) * depth))
                # Some shenanigans to make sure we are at the front and not
                # an upstream basin
                if np.any(ice_above_wl):
                    last_above_wl = np.where(ice_above_wl)[0][-1]
                else:
                    last_above_wl = np.where(ice_below_wl)[0][0]
                if fl.bed_h[last_above_wl+1] > self.water_level and \
                        np.any(ice_below_wl[last_above_wl+1:]):
                    last_above_wl = (np.where(ice_below_wl[last_above_wl+1:])[0][0] +
                                     last_above_wl+1)

                last_above_wl = int(utils.clip_max(last_above_wl,
                                                   len(fl.bed_h)-2))
                first_ice = np.where(fl.thick[0:last_above_wl+1] > 0)[0][0]
                # Check that the "last_above_wl" is not just the last in an
                # upstream basin connected to ice "above_wl" downstream
                land_interface = ((fl.bed_h[last_above_wl+1] > self.water_level)
                                  & (fl.thick[last_above_wl+1] > 0))

                # Determine water depth at the front
                h = fl.thick[last_above_wl]
                d = h - (fl.surface_h[last_above_wl] - self.water_level)
                thick_stag[last_above_wl+1] = h
                depth_stag[last_above_wl+1] = d

                # Determine height above buoancy
                z_a_b = utils.clip_min(0, thick_stag - depth_stag *
                                       (rho_ocean / self.rho))

                # Compute net hydrostatic force at the front. One could think
                # about incorporating ice mélange / sea ice here as an
                # additional backstress term. (And also in the frontal ablation
                # formulation below.)
                if not land_interface:
                    # Calculate the additional (pull) force
                    pull_last = utils.clip_min(0, 0.5 * G * (self.rho * h**2 -
                                               rho_ocean * d**2))

                    # Determine distance over which above force is distributed
                    stretch_length = (last_above_wl - first_ice) * dx
                    stretch_length = utils.clip_min(stretch_length, dx)
                    stretch_dist = utils.clip_max(stretch_length,
                                                  self.stretch_dist_p)
                    n_stretch = np.rint(stretch_dist/dx).astype(int)

                    # Define stretch factor and add to driving stress
                    stretch_factor = np.zeros(n_stretch)
                    for j in range(n_stretch):
                        stretch_factor[j] = 2*(j+1)/(n_stretch+1)
                    if dx > stretch_dist:
                        stretch_factor = stretch_dist / dx
                        n_stretch = 1

                    stretch_first = utils.clip_min(0, (last_above_wl+2) -
                                                   n_stretch).astype(int)
                    stretch_last = last_above_wl+2

                    # Take slope for stress calculation at boundary grid cell
                    # as the mean over the stretched distance (see above)
                    if stretch_first != stretch_last-1:
                        slope_stag[last_above_wl+1] = np.nanmean(slope_stag
                                                                 [stretch_first-1:
                                                                  stretch_last-1])
                stress = self.rho * G * slope_stag * thick_stag
                # Add "stretching stress" to basal shear/driving stress
                if not land_interface:
                    stress[stretch_first:stretch_last] = (stress[stretch_first:
                                                          stretch_last] +
                                                          stretch_factor *
                                                          (pull_last /
                                                           stretch_dist))
                # Compute velocities
                u_drag[:] = thick_stag * stress**N * self._fd * sf_stag**N

                # Arbitrarily manipulating u_slide for grid cells
                # approaching buoyancy to prevent it from going
                # towards infinity...
                # Not sure if sf_stag is correct here
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    u_slide[:] = (stress**N / z_a_b) * self.fs * sf_stag**N
                    u_slide[:] = np.where(z_a_b <= 0.01 * thick_stag, 4 * u_drag,
                                          u_slide)

                # Force velocity beyond grounding line to be zero in order to
                # prevent shelf dynamics. This might accumulate too much volume
                # in the last_above_wl+1 grid cell, which we deal with in the
                # calving scheme below
                if not land_interface:
                    u_slide[last_above_wl+2:] = 0
                    u_drag[last_above_wl+2:] = 0

                u_stag[:] = u_drag + u_slide

                # Staggered section
                # For the flux out of the last grid cell, the staggered section
                # is set to the cross section of the calving front, as we are
                # dealing with a terminal cliff.
                section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
                section_stag[[0, -1]] = section[[0, -1]]
                if not land_interface:
                    section_stag[last_above_wl+1] = section[last_above_wl]
                # We calculate the "baseline" calving flux here to be consistent
                # with the dynamics:
                    k = self.calving_k
                    self.calving_flux = utils.clip_min(0, k * d * h *
                                                       fl.widths_m[last_above_wl])

            # Usual ice dynamics
            else:
                rhogh = (self.rho*G*slope_stag)**N
                u_drag[:] = (thick_stag**(N+1)) * self._fd * rhogh * \
                             sf_stag**N
                u_slide[:] = (thick_stag**(N-1)) * self.fs * rhogh * \
                              sf_stag**N  # Not sure if sf is correct here
                u_stag[:] = u_drag + u_slide

                # Staggered section
                section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
                section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # Add boundary condition
            if flux_gate is not None:
                flux_stag[0] = flux_gate(self.yr)

            # CFL condition
            if not self.fixed_dt:
                maxu = np.max(np.abs(u_stag))
                if maxu > cfg.FLOAT_EPS:
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
                            '{:.1f}s vs {:.1f}s. Happening at '
                            'simulation year {:.1f}, fl_id {}, '
                            'bin_id {} and max_u {:.3f} m yr-1.'
                            ''.format(cfl_dt, self.min_dt, self.yr, fl_id,
                                      np.argmax(np.abs(u_stag)),
                                      maxu * cfg.SEC_IN_YEAR))

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

            # We compute MB in this loop, before mass-redistribution occurs,
            # so that MB models which rely on glacier geometry to decide things
            # (like PyGEM) can do wo with a clean glacier state
            mbs.append(self.get_mb(fl.surface_h, self.yr,
                                   fl_id=fl_id, fls=self.fls))

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt

        # A second loop for the mass exchange
        for fl_id, fl in enumerate(self.fls):
            # We empty the calving bucket, if there is no ice below water
            if not np.any((fl.thick > 0) & (fl.bed_h < self.water_level)):
                fl.calving_bucket_m3 = 0

            flx_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            tr = self._tributary_indices[fl_id]

            dx = fl.dx_meter

            is_trib = tr[0] is not None

            # For these we had an additional grid point
            if is_trib:
                flx_stag = flx_stag[:-1]

            # Mass balance
            widths = fl.widths_m
            mb = mbs[fl_id]
            # Allow parabolic beds to grow
            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)

            # Prevent surface melt below water level
            bed_below_wl = (fl.bed_h < self.water_level) & (fl.thick > 0)
            if self.do_calving and fl.has_ice() and np.any(bed_below_wl):
                bed_below_wl = np.where(bed_below_wl)[0]
                # We only look at the last part, a.k.a. tongue here
                diff_bwl = np.diff(bed_below_wl)
                bed_below_wl = np.split(bed_below_wl,
                                        np.where(diff_bwl > 1)[0]+1)[-1]
                if not np.any(fl.thick[:bed_below_wl[0]+1] == 0) and not \
                        np.any(fl.bed_h[bed_below_wl[-1]:] > self.water_level):
                    mb_limit_awl = ((-fl.surface_h[bed_below_wl] +
                                     self.water_level) * widths[bed_below_wl])
                    mb[bed_below_wl] = utils.clip_min(mb[bed_below_wl],
                                                      mb_limit_awl)
                    mb[fl.surface_h < self.water_level] = 0

            # Update section with ice flow and mass balance
            new_section = (fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt/dx +
                           trib_flux*dt/dx + mb)

            # Keep positive values only and store
            fl.section = utils.clip_min(new_section, 0)

            section = fl.section
            self.calving_rate_myr = 0.
            # Remove detached bodies of ice in the water, which can happen as
            # a result of dynamics + surface melt
            bed_below_wl = (fl.thick > 0) & (fl.bed_h < self.water_level)
            if fl.has_ice() and np.any(bed_below_wl) and self.do_calving:
                bed_below_wl = np.where(bed_below_wl)[0]
                no_ice = []
                first_ice = []
                no_ice = np.nonzero((fl.thick < 0.1) &
                                    (fl.bed_h < self.water_level))[0]
                if len(no_ice) > 0:
                    if no_ice[-1]+1 >= len(fl.bed_h):
                        no_ice = np.delete(no_ice, -1)
                    first_ice = np.where((fl.thick[no_ice+1] > 0) &
                                         (fl.bed_h[no_ice+1] < self.water_level))
                    first_ice = no_ice[first_ice]+1
                if len(first_ice) > 0:
                    for i in range(len(first_ice)):
                        last_ice = []
                        last_ice = np.nonzero((fl.thick[first_ice[i]:] > 0) &
                                              (fl.bed_h[first_ice[i]:] <
                                               self.water_level))[0]
                        if len(last_ice) > 0:
                            last_ice = last_ice[-1]
                            detached = np.arange(first_ice[i],
                                                 last_ice+first_ice[i]+1)
                            detached = np.intersect1d(detached, bed_below_wl)
                            add_calving = np.sum(section[detached]) * dx
                            self.calving_m3_since_y0 += add_calving
                            self.calving_rate_myr += (np.size(detached) * dx /
                                                      dt * cfg.SEC_IN_YEAR)
                            section[detached] = 0
                            fl.section = section
                            section = fl.section

            # If we use a flux-gate, store the total volume that came in
            self.flux_gate_m3_since_y0 += flx_stag[0] * dt

            # Add the last flux to the tributary
            # this works because the lines are sorted in order
            if is_trib:
                # tr tuple: line_index, start, stop, gaussian_kernel
                self.trib_flux[tr[0]][tr[1]:tr[2]] += \
                    utils.clip_min(flx_stag[-1], 0) * tr[3]

            # --- The rest is for calving only ---

            # If tributary, do calving only if we are not transferring mass
            if is_trib and flx_stag[-1] > 0:
                continue

            # No need to do calving in these cases either
            if not self.do_calving or not fl.has_ice():
                continue

            # We do calving only if the last glacier bed pixel is below water
            # (this is to avoid calving elsewhere than at the front)
            if fl.bed_h[fl.thick > 0][-1] > self.water_level:
                continue
            # We do calving only if there is some ice grounded below water
            depth = utils.clip_min(0, self.water_level - fl.bed_h)
            ice_above_wl = ((fl.bed_h < self.water_level) &
                            (fl.thick >= (rho_ocean / self.rho) * depth))
            ice_below_wl = ((fl.bed_h < self.water_level) & (fl.thick > 0))
            # If there is only ice below water, we just remove it
            if np.all(fl.surface_h[fl.thick > 0] < self.water_level):
                add_calving = np.sum(section) * dx
                self.calving_m3_since_y0 += add_calving
                self.calving_rate_myr += (np.sum(fl.surface_h[fl.thick > 0] <
                                          self.water_level) * dx / dt *
                                          cfg.SEC_IN_YEAR)
                section[:] = 0
                fl.section = section
                fl.calving_bucket_m3 = 0
                continue
            if np.any(ice_above_wl):
                last_above_wl = np.where(ice_above_wl)[0][-1]
            elif np.any(ice_below_wl):
                last_above_wl = np.where(ice_below_wl)[0][0]
            # Make sure again we are where we want to be; the end of the glacier
            if fl.bed_h[last_above_wl+1] > self.water_level and \
               np.any(ice_below_wl[last_above_wl+1:]):
                last_above_wl = (np.where(ice_below_wl[last_above_wl+1:])[0][0] +
                                 last_above_wl+1)
            last_above_wl = int(utils.clip_max(last_above_wl, len(fl.bed_h)-2))
            # As above in the dynamics, make sure we are not at a land-interface
            if (fl.bed_h[last_above_wl+1] > self.water_level and
                    fl.thick[last_above_wl+1] > 0):
                continue

            # OK, we're really calving
            # Make sure that we have a smooth front. Necessary due to inhibiting
            # ice flux beyond the grounding line in the dynamics above
            section = fl.section
            while ((fl.surface_h[last_above_wl] > fl.surface_h[last_above_wl-1])
                   and fl.thick[last_above_wl] > 0) and last_above_wl > 0:
                diff_sec = 0
                old_thick = fl.thick[last_above_wl]
                old_sec = fl.section[last_above_wl]
                new_thick = (old_thick - (fl.surface_h[last_above_wl] -
                                          fl.surface_h[last_above_wl-1]))
                fl.thick[last_above_wl] = new_thick
                diff_sec = old_sec - fl.section[last_above_wl]
                section[last_above_wl+1] += diff_sec
                section[last_above_wl] -= diff_sec
                fl.section = section
                section = fl.section
                if ((fl.bed_h[last_above_wl+1] < self.water_level) &
                    (fl.thick[last_above_wl+1] >= (rho_ocean / self.rho) *
                     depth[last_above_wl+1])):
                    last_above_wl += 1
                else:
                    break

            q_calving = self.calving_flux * dt
            # Remove ice below flotation beyond the first grid cell after the
            # grounding line. That cell is the "advance" bucket, meaning it can
            # contain ice below flotation.
            add_calving = np.sum(section[last_above_wl+2:]) * dx
            # Add to the bucket and the diagnostics. As we calculated the
            # calving flux from the glacier state at the start of the time step,
            # we do not add to the calving bucket what is already removed by the
            # flotation criterion to avoid double counting.
            self.calving_m3_since_y0 += utils.clip_min(q_calving, add_calving)
            fl.calving_bucket_m3 += utils.clip_min(0, q_calving - add_calving)
            self.calving_rate_myr += (utils.clip_min(q_calving, add_calving) /
                                      fl.section[last_above_wl] / dt *
                                      cfg.SEC_IN_YEAR)
            section[last_above_wl+2:] = 0
            # This is what remains to be removed by the calving bucket.
            to_remove = section[last_above_wl+1] * dx
            if 0 < to_remove < fl.calving_bucket_m3:
                # This is easy, we remove everything
                section[last_above_wl+1] = 0
                fl.calving_bucket_m3 -= to_remove
            elif to_remove > 0:
                # We can only remove part of it
                section[last_above_wl+1] = ((to_remove -
                                             fl.calving_bucket_m3) / dx)
                fl.calving_bucket_m3 = 0

            vol_last = section[last_above_wl] * dx
            while fl.calving_bucket_m3 >= vol_last and \
                    fl.bed_h[last_above_wl] < self.water_level:
                fl.calving_bucket_m3 -= vol_last
                section[last_above_wl] = 0

                # OK check if we need to continue (unlikely)
                last_above_wl -= 1
                if np.abs(last_above_wl) <= len(fl.bed_h):
                    vol_last = section[last_above_wl] * dx
                else:
                    fl.calving_bucket_m3 = 0
                    break
            # We update the glacier with our changes
            fl.section = section

        # Next step
        self.t += dt
        return dt

    def get_diagnostics(self, fl_id=-1):
        """Obtain model diagnostics in a pandas DataFrame.

        Velocities in OGGM's FluxBasedModel are sometimes subject to
        numerical instabilities. To deal with the issue, you can either
        set a smaller ``PARAMS['cfl_number']`` (e.g. 0.01) or smooth the
        output a bit, e.g. with ``df.rolling(5, center=True, min_periods=1).mean()``

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
            - surface_ice_velocity: m per second (corrected for surface - simplified)
        """
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
        var = self.u_drag[fl_id]
        var2 = self.u_slide[fl_id]
        _u_slide = (var2[1:nx+1] + var2[:nx])/2
        df['ice_velocity'] = (var[1:nx+1] + var[:nx])/2 + _u_slide
        df['surface_ice_velocity'] = (((var[1:nx+1] + var[:nx]) / 2  *
                                       self._surf_vel_fac) + _u_slide)
        var = self.u_drag[fl_id]
        df['deformation_velocity'] = (var[1:nx+1] + var[:nx])/2
        var = self.u_slide[fl_id]
        df['sliding_velocity'] = (var[1:nx+1] + var[:nx])/2
        var = self.shapefac_stag[fl_id]
        df['shape_fac'] = (var[1:nx+1] + var[:nx])/2

        # Not Staggered
        df['tributary_flux'] = self.trib_flux[fl_id]

        return df
