import os
import shutil
from distutils.util import strtobool

import numpy as np
import shapely.geometry as shpg
from scipy import optimize as optimization

# Local imports
import oggm
import oggm.cfg as cfg
from oggm.utils import get_demo_file, mkdir
from oggm.workflow import execute_entity_task
from oggm.utils import oggm_urlretrieve
from oggm.core import flowline
from oggm.core.flowline import FlowlineModel
from oggm.cfg import SEC_IN_DAY, SEC_IN_HOUR, G
from oggm.exceptions import InvalidParamsError
from oggm import utils


def dummy_constant_bed(hmax=3000., hmin=1000., nx=200, map_dx=100.,
                       widths=3.):

    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h
    widths = surface_h * 0. + widths
    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.RectangularBedFlowline(line, dx, map_dx, surface_h,
                                            bed_h, widths)]


def dummy_constant_bed_cliff(hmax=3000., hmin=1000., nx=200, map_dx=100.,
                             cliff_height=250.):
    """
    I introduce a cliff in the bed to test the mass conservation of the models
    Such a cliff could be real or a DEM error/artifact
    """
    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)

    surface_h[50:] = surface_h[50:] - cliff_height

    bed_h = surface_h
    widths = surface_h * 0. + 1.

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.RectangularBedFlowline(line, dx, map_dx, surface_h,
                                            bed_h, widths)]


def dummy_constant_bed_obstacle(hmax=3000., hmin=1000., nx=200):
    """
    I introduce an obstacle in the bed
    """

    map_dx = 100.
    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)

    cliff_height = 200.0
    surface_h[60:] = surface_h[60:] + cliff_height

    bed_h = surface_h
    widths = surface_h * 0. + 1.

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.RectangularBedFlowline(line, dx, map_dx, surface_h,
                                            bed_h, widths)]


def dummy_bumpy_bed():
    map_dx = 100.
    dx = 1.
    nx = 200

    coords = np.arange(0, nx - 0.5, 1)
    surface_h = np.linspace(3000, 1000, nx)
    surface_h += 170. * np.exp(-((coords - 30) / 5) ** 2)

    bed_h = surface_h
    widths = surface_h * 0. + 3.
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.RectangularBedFlowline(line, dx, map_dx, surface_h,
                                            bed_h, widths)]


def dummy_noisy_bed(map_dx=100.):
    dx = 1.
    nx = 200
    np.random.seed(42)
    coords = np.arange(0, nx - 0.5, 1)
    surface_h = np.linspace(3000, 1000, nx)
    surface_h += 100 * np.random.rand(nx) - 50.

    bed_h = surface_h
    widths = surface_h * 0. + 3.
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.RectangularBedFlowline(line, dx, map_dx, surface_h,
                                            bed_h, widths)]


def dummy_parabolic_bed(hmax=3000., hmin=1000., nx=200, map_dx=100.,
                        default_shape=5.e-3,
                        from_other_shape=None, from_other_bed=None):
    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h * 1
    shape = surface_h * 0. + default_shape
    if from_other_shape is not None:
        shape[0:len(from_other_shape)] = from_other_shape

    if from_other_bed is not None:
        bed_h[0:len(from_other_bed)] = from_other_bed

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.ParabolicBedFlowline(line, dx, map_dx, surface_h,
                                          bed_h, shape)]


def dummy_mixed_bed(deflambdas=3.5, map_dx=100., mixslice=None):
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    shape = surface_h * 0. + 3.e-03
    if mixslice:
        shape[mixslice] = np.NaN
    else:
        shape[10:20] = np.NaN
    is_trapezoid = ~np.isfinite(shape)
    lambdas = shape * 0.
    lambdas[is_trapezoid] = deflambdas

    widths_m = bed_h * 0. + 10
    section = bed_h * 0.

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

    fls = flowline.MixedBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                    surface_h=surface_h, bed_h=bed_h,
                                    section=section, bed_shape=shape,
                                    is_trapezoid=is_trapezoid,
                                    lambdas=lambdas, widths_m=widths_m)

    return [fls]


def dummy_trapezoidal_bed(hmax=3000., hmin=1000., nx=200):
    map_dx = 100.
    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 1.6

    lambdas = surface_h * 0. + 2

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

    return [flowline.TrapezoidalBedFlowline(line, dx, map_dx, surface_h,
                                            bed_h, widths, lambdas)]


def dummy_width_bed():
    """This bed has a width of 6 during the first 20 points and then 3"""

    map_dx = 100.
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 3.
    widths[0:20] = 6.

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.RectangularBedFlowline(line, dx, map_dx, surface_h,
                                            bed_h, widths)]


def dummy_width_bed_tributary(map_dx=100., n_trib=1):
    # bed with tributary glacier
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 3.
    widths[0:20] = 6 / (n_trib + 1)

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

    fl_0 = flowline.RectangularBedFlowline(line, dx, map_dx, surface_h, bed_h,
                                           widths)
    coords = np.arange(0, 19.1, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0. + 1]).T)

    out = [fl_0]
    for i in range(n_trib):
        fl_1 = flowline.RectangularBedFlowline(line, dx, map_dx,
                                               surface_h[0:20],
                                               bed_h[0:20],
                                               widths[0:20])
        fl_1.set_flows_to(fl_0)
        out.append(fl_1)

    return out[::-1]


def patch_url_retrieve_github(url, *args, **kwargs):
    """A simple patch to OGGM's download function to make sure we don't
    download elsewhere than expected."""

    assert ('github' in url or
            'cluster.klima.uni-bremen.de/~fmaussion/test_gdirs/' in url or
            'cluster.klima.uni-bremen.de/~fmaussion/demo_gdirs/' in url)
    return oggm_urlretrieve(url, *args, **kwargs)


def use_multiprocessing():
    try:
        return strtobool(os.getenv("OGGM_TEST_MULTIPROC", "True"))
    except BaseException:
        return True


def get_ident():
    ident_str = '$Id$'
    if ":" not in ident_str:
        return 'no_git_id'
    return ident_str.replace("$", "").replace("Id:", "").replace(" ", "")


def get_test_dir():

    s = get_ident()
    out = os.path.join(cfg.PATHS['test_dir'], s)
    if 'PYTEST_XDIST_WORKER' in os.environ:
        out = os.path.join(out, os.environ.get('PYTEST_XDIST_WORKER'))
    mkdir(out)

    # If new ident, remove all other dirs so spare space
    for d in os.listdir(cfg.PATHS['test_dir']):
        if d and d != s:
            shutil.rmtree(os.path.join(cfg.PATHS['test_dir'], d))
    return out


def init_hef(reset=False, border=40, logging_level='INFO'):

    from oggm.core import gis, inversion, climate, centerlines, flowline
    import geopandas as gpd

    # test directory
    testdir = os.path.join(get_test_dir(), 'tmp_border{}'.format(border))
    if not os.path.exists(testdir):
        os.makedirs(testdir)
        reset = True

    # Init
    cfg.initialize(logging_level=logging_level)
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PARAMS['baseline_climate'] = ''
    cfg.PATHS['working_dir'] = testdir
    cfg.PARAMS['border'] = border

    hef_file = get_demo_file('Hintereisferner_RGI5.shp')
    entity = gpd.read_file(hef_file).iloc[0]

    gdir = oggm.GlacierDirectory(entity, reset=reset)
    if not gdir.has_file('inversion_params'):
        reset = True
        gdir = oggm.GlacierDirectory(entity, reset=reset)

    if not reset:
        return gdir

    gis.define_glacier_region(gdir, entity=entity)
    execute_entity_task(gis.glacier_masks, [gdir])
    execute_entity_task(centerlines.compute_centerlines, [gdir])
    centerlines.initialize_flowlines(gdir)
    centerlines.compute_downstream_line(gdir)
    centerlines.compute_downstream_bedshape(gdir)
    centerlines.catchment_area(gdir)
    centerlines.catchment_intersections(gdir)
    centerlines.catchment_width_geom(gdir)
    centerlines.catchment_width_correction(gdir)
    climate.process_custom_climate_data(gdir)
    mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
    res = climate.t_star_from_refmb(gdir, mbdf=mbdf)
    climate.local_t_star(gdir, tstar=res['t_star'], bias=res['bias'])
    climate.mu_star_calibration(gdir)

    inversion.prepare_for_inversion(gdir, add_debug_var=True)

    ref_v = 0.573 * 1e9

    glen_n = cfg.PARAMS['glen_n']

    def to_optimize(x):
        # For backwards compat
        _fd = 1.9e-24 * x[0]
        glen_a = (glen_n+2) * _fd / 2.
        fs = 5.7e-20 * x[1]
        v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a)
        return (v - ref_v)**2

    out = optimization.minimize(to_optimize, [1, 1],
                                bounds=((0.01, 10), (0.01, 10)),
                                tol=1e-4)['x']
    _fd = 1.9e-24 * out[0]
    glen_a = (glen_n+2) * _fd / 2.
    fs = 5.7e-20 * out[1]
    v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                 glen_a=glen_a,
                                                 write=True)

    d = dict(fs=fs, glen_a=glen_a)
    d['factor_glen_a'] = out[0]
    d['factor_fs'] = out[1]
    gdir.write_pickle(d, 'inversion_params')

    # filter
    inversion.filter_inversion_output(gdir)

    inversion.distribute_thickness_interp(gdir, varname_suffix='_interp')
    inversion.distribute_thickness_per_altitude(gdir, varname_suffix='_alt')

    flowline.init_present_time_glacier(gdir)

    return gdir


def init_columbia(reset=False):

    from oggm.core import gis, climate, centerlines
    import geopandas as gpd

    # test directory
    testdir = os.path.join(get_test_dir(), 'tmp_columbia')
    if not os.path.exists(testdir):
        os.makedirs(testdir)
        reset = True

    # Init
    cfg.initialize()
    cfg.PATHS['working_dir'] = testdir
    cfg.PARAMS['use_intersects'] = False
    cfg.PATHS['dem_file'] = get_demo_file('dem_Columbia.tif')
    cfg.PARAMS['border'] = 10

    entity = gpd.read_file(get_demo_file('01_rgi60_Columbia.shp')).iloc[0]
    gdir = oggm.GlacierDirectory(entity, reset=reset)
    if gdir.has_file('climate_monthly'):
        return gdir

    gis.define_glacier_region(gdir, entity=entity)
    gis.glacier_masks(gdir)
    centerlines.compute_centerlines(gdir)
    centerlines.initialize_flowlines(gdir)
    centerlines.compute_downstream_line(gdir)
    centerlines.catchment_area(gdir)
    centerlines.catchment_intersections(gdir)
    centerlines.catchment_width_geom(gdir)
    centerlines.catchment_width_correction(gdir)
    climate.process_dummy_cru_file(gdir, seed=0)
    return gdir


class TempEnvironmentVariable:
    """Context manager for environment variables

    https://gist.github.com/devhero/7e015f0ce0abacab3880d33c26f07674
    """
    def __init__(self, **kwargs):
        self.envs = kwargs

    def __enter__(self):
        self.old_envs = {}
        for k, v in self.envs.items():
            self.old_envs[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *args):
        for k, v in self.old_envs.items():
            if v:
                os.environ[k] = v
            else:
                del os.environ[k]


class FluxBasedModelOld(FlowlineModel):
    """The "old" Flowline model befure the flux limiter correction
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=0.05,
                 min_dt=1*SEC_IN_HOUR, max_dt=10*SEC_IN_DAY,
                 time_stepping='user', **kwargs):
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
            Schoolbook theory says that the scheme is stable
            with CFL=1, but practice does not. There is no "best" CFL number:
            small values are more robust but also slowier...
        min_dt : float
            with high velocities, time steps can become very small and your
            model might run very slowly. In production we just take the risk
            of becoming unstable and prevent very small time steps.
        max_dt : float
            just to make sure that the adaptive time step is not going to
            choose too high values either. We could make this higher I think
        time_stepping : str
            let OGGM choose default values for the parameters above for you.
            Possible settings are: 'ambitious', 'default', 'conservative',
            'ultra-conservative'.
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
        super(FluxBasedModelOld, self).__init__(flowlines, mb_model=mb_model,
                                                y0=y0, glen_a=glen_a, fs=fs,
                                                inplace=inplace, **kwargs)

        if time_stepping == 'ambitious':
            cfl_number = 0.1
            min_dt = 1*SEC_IN_DAY
            max_dt = 15*SEC_IN_DAY
        elif time_stepping == 'default':
            cfl_number = 0.05
            min_dt = 1*SEC_IN_HOUR
            max_dt = 10*SEC_IN_DAY
        elif time_stepping == 'conservative':
            cfl_number = 0.01
            min_dt = SEC_IN_HOUR
            max_dt = 5*SEC_IN_DAY
        elif time_stepping == 'ultra-conservative':
            cfl_number = 0.01
            min_dt = SEC_IN_HOUR / 10
            max_dt = 5*SEC_IN_DAY
        else:
            if time_stepping != 'user':
                raise ValueError('time_stepping not understood.')

        self.dt_warning = False
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
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

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt

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

            # Convert to angle?
            # slope_stag = np.sin(np.arctan(slope_stag))

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
            rhogh = (self.rho*G*slope_stag)**N
            u_stag[:] = (thick_stag**(N+1)) * self._fd * rhogh * sf_stag**N + \
                        (thick_stag**(N-1)) * self.fs * rhogh

            # Staggered section
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # CFL condition
            maxu = np.max(np.abs(u_stag))
            if maxu > 0.:
                _dt = self.cfl_number * dx / maxu
            else:
                _dt = self.max_dt
            if _dt < dt:
                dt = _dt

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

        # Time step
        self.dt_warning = dt < min_dt
        dt = utils.clip_scalar(dt, min_dt, self.max_dt)

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
