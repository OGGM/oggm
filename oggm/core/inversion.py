"""Glacier thickness.


Note for later: the current code is oriented towards a consistent framework
for flowline modelling. The major direction shift happens at the
flowlines.width_correction step where the widths are computed to follow the
altitude area distribution. This must not and should not be the case when
the actual objective is to provide a glacier thickness map. For this,
the geometrical width and some other criteria such as e.g. "all altitudes
*in the current subcatchment* above the considered cross-section are
contributing to the flux" might give more interpretable results.

References:

Farinotti, D., Huss, M., Bauder, A., Funk, M. and Truffer, M.: A method to
    estimate the ice volume and ice-thickness distribution of alpine glaciers,
    J. Glaciol., 55(191), 422-430, doi:10.3189/002214309788816759, 2009.

Huss, M. and Farinotti, D.: Distributed ice thickness and volume of all
    glaciers around the globe, J. Geophys. Res. Earth Surf., 117(4), F04010,
    doi:10.1029/2012JF002523, 2012.

Bahr  Pfeffer, W. T., Kaser, G., D. B.: Glacier volume estimation as an
    ill-posed boundary value problem, Cryosph. Discuss. Cryosph. Discuss.,
    6(6), 5405-5420, doi:10.5194/tcd-6-5405-2012, 2012.

Adhikari, S., Marshall, J. S.: Parameterization of lateral drag in flowline
    models of glacier dynamics, Journal of Glaciology, 58(212), 1119-1132.
    doi:10.3189/2012JoG12J018, 2012.
"""
# Built ins
import logging
import warnings

# External libs
import numpy as np
from scipy.interpolate import griddata
from scipy import optimize

# Locals
from oggm import utils, cfg
from oggm import entity_task
from oggm.core.gis import gaussian_blur
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

# arbitrary constant
MIN_WIDTH_FOR_INV = 10


@entity_task(log, writes=['inversion_input'])
def prepare_for_inversion(gdir,
                          invert_with_rectangular=True,
                          invert_all_rectangular=False,
                          invert_with_trapezoid=True,
                          invert_all_trapezoid=False):
    """Prepares the data needed for the inversion.

    Mostly the mass flux and slope angle, the rest (width, height) was already
    computed. It is then stored in a list of dicts in order to be faster.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    # variables
    fls = gdir.read_pickle('inversion_flowlines')

    towrite = []
    for fl in fls:

        # Distance between two points
        dx = fl.dx * gdir.grid.dx

        # Widths
        widths = fl.widths * gdir.grid.dx

        # Heights
        hgt = fl.surface_h
        angle = -np.gradient(hgt, dx)  # beware the minus sign

        # Flux needs to be in [m3 s-1] (*ice* velocity * surface)
        # fl.flux is given in kg m-2 yr-1, rho in kg m-3, so this should be it:
        rho = cfg.PARAMS['ice_density']
        
        # This might error if usuer didnt compute apparent MB
        try:
            flux = fl.flux * (gdir.grid.dx**2) / cfg.SEC_IN_YEAR / rho
        except TypeError:
            raise InvalidWorkflowError('Flux through flowline unknown. '
                                       'Did you compute the apparent MB?')
        flux_out = fl.flux_out * (gdir.grid.dx**2) / cfg.SEC_IN_YEAR / rho

        # Clip flux to 0
        if np.any(flux < -0.1):
            log.info('(%s) has negative flux somewhere', gdir.rgi_id)
        utils.clip_min(flux, 0, out=flux)

        if np.sum(flux <= 0) > 1 and len(fls) == 1:
            log.warning("More than one grid point has zero or "
                        "negative flux: this should not happen.")

        if fl.flows_to is None and gdir.inversion_calving_rate == 0:
            if not np.allclose(flux_out, 0., atol=0.1):
                # TODO: this test doesn't seem meaningful here
                msg = ('({}) flux at terminus should be zero, but is: '
                       '{.4f} m3 ice s-1'.format(gdir.rgi_id, flux_out))
                raise RuntimeError(msg)

        # Shape
        is_rectangular = fl.is_rectangular
        if not invert_with_rectangular:
            is_rectangular[:] = False
        if invert_all_rectangular:
            is_rectangular[:] = True

        # Trapezoid is new - might not be available
        is_trapezoid = getattr(fl, 'is_trapezoid', None)
        if is_trapezoid is None:
            is_trapezoid = fl.is_rectangular * False
        if not invert_with_trapezoid:
            is_rectangular[:] = False
        if invert_all_trapezoid:
            is_trapezoid[:] = True

        # Optimisation: we need to compute this term of a0 only once
        flux_a0 = np.where(is_rectangular, 1, 1.5)
        flux_a0 *= flux / widths

        # Add to output
        cl_dic = dict(dx=dx, flux_a0=flux_a0, width=widths,
                      slope_angle=angle, is_rectangular=is_rectangular,
                      is_trapezoid=is_trapezoid, flux=flux,
                      is_last=fl.flows_to is None, hgt=hgt,
                      invert_with_trapezoid=invert_with_trapezoid)
        towrite.append(cl_dic)

    # Write out
    gdir.write_pickle(towrite, 'inversion_input')


def _inversion_poly(a3, a0):
    """Solve for degree 5 polynomial with coefficients a5=1, a3, a0."""
    sols = np.roots([1., 0., a3, 0., 0., a0])
    test = (np.isreal(sols)*np.greater(sols, [0]*len(sols)))
    return sols[test][0].real


def _inversion_simple(a3, a0):
    """Solve for degree 5 polynomial with coefficients a5=1, a3=0., a0."""

    return (-a0)**(1./5.)


def _compute_thick(a0s, a3, flux_a0, _inv_function):
    """Content of the original inner loop of the mass-conservation inversion.

    Put here to avoid code duplication.

    Parameters
    ----------
    a0s
    a3
    flux_a0
    _inv_function

    Returns
    -------
    the thickness
    """

    if np.any(~np.isfinite(a0s)):
        raise RuntimeError('non-finite coefficients in the polynomial.')

    # Solve the polynomials
    try:
        out_thick = np.zeros(len(a0s))
        for i, (a0, Q) in enumerate(zip(a0s, flux_a0)):
            out_thick[i] = _inv_function(a3, a0) if Q > 0 else 0
    except TypeError:
        # Scalar
        out_thick = _inv_function(a3, a0s) if flux_a0 > 0 else 0

    if np.any(~np.isfinite(out_thick)):
        raise RuntimeError('non-finite coefficients in the polynomial.')

    return out_thick


def sia_thickness_via_optim(slope, width, flux, shape='rectangular',
                            glen_a=None, fs=None, t_lambda=None):
    """Compute the thickness numerically instead of analytically.

    It's the only way that works for trapezoid shapes.

    Parameters
    ----------
    slope : -np.gradient(hgt, dx)
    width : section width in m
    flux : mass flux in m3 s-1
    shape : 'rectangular', 'trapezoid' or 'parabolic'
    glen_a : Glen A, defaults to PARAMS
    fs : sliding, defaults to PARAMS
    t_lambda: the trapezoid lambda, defaults to PARAMS

    Returns
    -------
    the ice thickness (in m)
    """

    if len(np.atleast_1d(slope)) > 1:
        shape = utils.tolist(shape, len(slope))
        t_lambda = utils.tolist(t_lambda, len(slope))
        out = []
        for sl, w, f, s, t in zip(slope, width, flux, shape, t_lambda):
            out.append(sia_thickness_via_optim(sl, w, f, shape=s,
                                               glen_a=glen_a, fs=fs,
                                               t_lambda=t))
        return np.asarray(out)

    # Sanity
    if flux <= 0:
        return 0
    if width <= MIN_WIDTH_FOR_INV:
        return 0

    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if t_lambda is None:
        t_lambda = cfg.PARAMS['trapezoid_lambdas']
    if shape not in ['parabolic', 'rectangular', 'trapezoid']:
        raise InvalidParamsError('shape must be `parabolic`, `trapezoid` '
                                 'or `rectangular`, not: {}'.format(shape))

    # Ice flow params
    n = cfg.PARAMS['glen_n']
    fd = 2 / (n+2) * glen_a
    rho = cfg.PARAMS['ice_density']
    rhogh = (rho * cfg.G * slope) ** n

    # To avoid geometrical inconsistencies
    max_h = width / t_lambda if shape == 'trapezoid' else 1e4

    def to_minimize(h):
        u = (h ** (n + 1)) * fd * rhogh + (h ** (n - 1)) * fs * rhogh
        if shape == 'parabolic':
            sect = 2./3. * width * h
        elif shape == 'trapezoid':
            w0m = width - t_lambda * h
            sect = (width + w0m) / 2 * h
        else:
            sect = width * h
        return sect * u - flux
    out_h, r = optimize.brentq(to_minimize, 0, max_h, full_output=True)
    return out_h


def sia_thickness(slope, width, flux, shape='rectangular',
                  glen_a=None, fs=None, shape_factor=None):
    """Computes the ice thickness from mass-conservation.

    This is a utility function tested against the true OGGM inversion
    function. Useful for teaching and inversion with calving.

    Parameters
    ----------
    slope : -np.gradient(hgt, dx) (we don't clip for min slope!)
    width : section width in m
    flux : mass flux in m3 s-1
    shape : 'rectangular' or 'parabolic'
    glen_a : Glen A, defaults to PARAMS
    fs : sliding, defaults to PARAMS
    shape_factor: for lateral drag

    Returns
    -------
    the ice thickness (in m)
    """

    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if shape not in ['parabolic', 'rectangular']:
        raise InvalidParamsError('shape must be `parabolic` or `rectangular`, '
                                 'not: {}'.format(shape))

    _inv_function = _inversion_simple if fs == 0 else _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.PARAMS['glen_n']+2) * glen_a
    rho = cfg.PARAMS['ice_density']

    # Convert the flux to m2 s-1 (averaged to represent the sections center)
    flux_a0 = 1 if shape == 'rectangular' else 1.5
    flux_a0 *= flux / width

    # With numerically small widths this creates very high thicknesses
    try:
        flux_a0[width < MIN_WIDTH_FOR_INV] = 0
    except TypeError:
        if width < MIN_WIDTH_FOR_INV:
            flux_a0 = 0

    # Polynomial factors (a5 = 1)
    a0 = - flux_a0 / ((rho * cfg.G * slope) ** 3 * fd)
    a3 = fs / fd

    return _compute_thick(a0, a3, flux_a0, _inv_function)


def find_sia_flux_from_thickness(slope, width, thick, glen_a=None, fs=None,
                                 shape='rectangular'):
    """Find the ice flux produced by a given thickness and slope.

    This can be done analytically but I'm lazy and use optimisation instead.
    """

    def to_minimize(x):
        h = sia_thickness(slope, width, x[0], glen_a=glen_a, fs=fs,
                          shape=shape)
        return (thick - h)**2

    out = optimize.minimize(to_minimize, [1], bounds=((0, 1e12),))
    flux = out['x'][0]

    # Sanity check
    minimum = to_minimize([flux])
    if minimum > 1:
        warnings.warn('We did not find a proper flux for this thickness',
                      RuntimeWarning)
    return flux


def _vol_below_water(surface_h, bed_h, bed_shape, thick, widths,
                     is_rectangular, is_trapezoid, fac, t_lambda,
                     dx, water_level):
    bsl = (bed_h < water_level) & (thick > 0)
    n_thick = np.copy(thick)
    n_thick[~bsl] = 0
    n_thick[bsl] = utils.clip_max(surface_h[bsl], water_level) - bed_h[bsl]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        n_w = np.sqrt(4 * n_thick / bed_shape)
    n_w[is_rectangular] = widths[is_rectangular]
    out = fac * n_thick * n_w * dx
    # Trap
    it = is_trapezoid
    out[it] = (n_w[it] + n_w[it] - t_lambda*n_thick[it]) / 2*n_thick[it]*dx
    return out


@entity_task(log, writes=['inversion_output'])
def mass_conservation_inversion(gdir, glen_a=None, fs=None, write=True,
                                filesuffix='', water_level=None,
                                t_lambda=None):
    """ Compute the glacier thickness along the flowlines

    More or less following Farinotti et al., (2009).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    glen_a : float
        glen's creep parameter A. Defaults to cfg.PARAMS.
    fs : float
        sliding parameter. Defaults to cfg.PARAMS.
    write: bool
        default behavior is to compute the thickness and write the
        results in the pickle. Set to False in order to spare time
        during calibration.
    filesuffix : str
        add a suffix to the output file
    water_level : float
        to compute volume below water level - adds an entry to the output dict
    t_lambda : float
        defining the angle of the trapezoid walls (see documentation). Defaults
        to cfg.PARAMS.
    """

    # Defaults
    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if t_lambda is None:
        t_lambda = cfg.PARAMS['trapezoid_lambdas']

    # Check input
    _inv_function = _inversion_simple if fs == 0 else _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.PARAMS['glen_n']+2) * glen_a
    a3 = fs / fd
    rho = cfg.PARAMS['ice_density']

    # Clip the slope, in rad
    min_slope = 'min_slope_ice_caps' if gdir.is_icecap else 'min_slope'
    min_slope = np.deg2rad(cfg.PARAMS[min_slope])

    out_volume = 0.

    cls = gdir.read_pickle('inversion_input')
    for cl in cls:
        # Clip slope to avoid negative and small slopes
        slope = cl['slope_angle']
        slope = utils.clip_array(slope, min_slope, np.pi/2.)

        # Glacier width
        w = cl['width']

        a0s = - cl['flux_a0'] / ((rho*cfg.G*slope)**3*fd)

        out_thick = _compute_thick(a0s, a3, cl['flux_a0'], _inv_function)

        # volume
        is_rect = cl['is_rectangular']
        fac = np.where(is_rect, 1, 2./3.)
        volume = fac * out_thick * w * cl['dx']

        # Now recompute thickness where parabola is too flat
        is_trap = cl['is_trapezoid']
        if cl['invert_with_trapezoid']:
            min_shape = cfg.PARAMS['mixed_min_shape']
            bed_shape = 4 * out_thick / w ** 2
            is_trap = ((bed_shape < min_shape) & ~ cl['is_rectangular'] &
                       (cl['flux'] > 0)) | is_trap
            for i in np.where(is_trap)[0]:
                try:
                    out_thick[i] = sia_thickness_via_optim(slope[i], w[i],
                                                           cl['flux'][i],
                                                           shape='trapezoid',
                                                           t_lambda=t_lambda,
                                                           glen_a=glen_a,
                                                           fs=fs)
                    sect = (2*w[i] - t_lambda * out_thick[i]) / 2 * out_thick[i]
                    volume[i] = sect * cl['dx']
                except ValueError:
                    # no solution error - we do with rect
                    out_thick[i] = sia_thickness_via_optim(slope[i], w[i],
                                                           cl['flux'][i],
                                                           shape='rectangular',
                                                           glen_a=glen_a,
                                                           fs=fs)
                    is_rect[i] = True
                    is_trap[i] = False
                    volume[i] = out_thick[i] * w[i] * cl['dx']

        # Sanity check
        if np.any(out_thick <= -1e-2):
            log.warning(f"{gdir.rgi_id} Found negative thickness: "
                        f"n={(out_thick <= -1e-2).sum()}, "
                        f"v={np.min(out_thick)}.")

        out_thick = utils.clip_min(out_thick, 0)

        if write:
            cl['is_trapezoid'] = is_trap
            cl['is_rectangular'] = is_rect
            cl['thick'] = out_thick
            cl['volume'] = volume

            # volume below sl
            try:
                bed_h = cl['hgt'] - out_thick
                bed_shape = 4 * out_thick / w ** 2
                if np.any(bed_h < 0):
                    cl['volume_bsl'] = _vol_below_water(cl['hgt'], bed_h,
                                                        bed_shape, out_thick,
                                                        w,
                                                        cl['is_rectangular'],
                                                        cl['is_trapezoid'],
                                                        fac, t_lambda,
                                                        cl['dx'], 0)
                if water_level is not None and np.any(bed_h < water_level):
                    cl['volume_bwl'] = _vol_below_water(cl['hgt'], bed_h,
                                                        bed_shape, out_thick,
                                                        w,
                                                        cl['is_rectangular'],
                                                        cl['is_trapezoid'],
                                                        fac, t_lambda,
                                                        cl['dx'],
                                                        water_level)
            except KeyError:
                # cl['hgt'] is not available on old prepro dirs
                pass

        out_volume += np.sum(volume)

    if write:
        gdir.write_pickle(cls, 'inversion_output', filesuffix=filesuffix)
        gdir.add_to_diagnostics('inversion_glen_a', glen_a)
        gdir.add_to_diagnostics('inversion_fs', fs)

    return out_volume


@entity_task(log, writes=['inversion_output'])
def filter_inversion_output(gdir, n_smoothing=5, min_ice_thick=1.,
                            max_depression=5.):
    """Filters the last few grid points after the physically-based inversion.

    For various reasons (but mostly: the equilibrium assumption), the last few
    grid points on a glacier flowline are often noisy and create unphysical
    depressions. Here we try to correct for that. It is not volume conserving,
    but area conserving. If a parabolic shape factor is getting smaller than
    the minimum defined one (cfg.PARAMS['mixed_min_shape']) the grid point is
    changed to a trapezoid, similar to what is done during the actual
    physically-based inversion.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    n_smoothing : int
        number of grid points which should be smoothed. Default is 5
    min_ice_thick : float
        the minimum ice thickness after the smoothing. Default is 1 m
    max_depression : float
        the limit allowed bed depression without smoothing. Default is 5 m
    """

    if gdir.is_tidewater:
        # No need for filter in tidewater case
        cls = gdir.read_pickle('inversion_output')
        init_vol = np.sum([np.sum(cl['volume']) for cl in cls])
        return init_vol

    if not gdir.has_file('downstream_line'):
        raise InvalidWorkflowError('filter_inversion_output now needs a '
                                   'previous call to the '
                                   'compute_dowstream_line and '
                                   'compute_downstream_bedshape tasks')

    dic_ds = gdir.read_pickle('downstream_line')
    cls = gdir.read_pickle('inversion_output')
    cl = cls[-1]

    # check that their are enough grid points for smoothing
    nr_grid_points = len(cl['thick'])
    if nr_grid_points <= n_smoothing:
        if nr_grid_points >= 3:
            n_smoothing = nr_grid_points - 1
        else:
            log.warning(f'({gdir.rgi_id}) filter_inversion_output: flowline '
                        f'has not enough grid points for applying the filter '
                        f'(only {nr_grid_points} grid points)!')
            # Return volume for convenience
            return np.sum([np.sum(cl['volume']) for cl in cls])

    # convert to negative number for indexing
    n_smoothing = -abs(n_smoothing)

    cl_sfc_h = cl['hgt'][n_smoothing:]
    cl_thick = cl['thick'][n_smoothing:]
    cl_width = cl['width'][n_smoothing:]
    cl_is_trap = cl['is_trapezoid'][n_smoothing:]
    cl_is_rect = cl['is_rectangular'][n_smoothing:]
    cl_bed_h = cl_sfc_h - cl_thick
    try:
        downstream_sfc_h = dic_ds['surface_h'][:5]
    except KeyError:
        raise InvalidWorkflowError('Please run compute_downstream_line and '
                                   'compute_downstream_bedshape for the '
                                   'filter.')

    # we smooth if the depression is larger than max_depression
    if downstream_sfc_h[0] - cl_bed_h[-1] > max_depression:
        # force the last grid point height to continue the downstream slope
        down_slope_avg = np.average(np.abs(np.diff(downstream_sfc_h)))
        new_last_bed_h = downstream_sfc_h[0] + down_slope_avg

        # now smoothly add the change of the bed height over all smoothing grid
        # points
        all_bed_h_changes = np.linspace(0, new_last_bed_h - cl_bed_h[-1],
                                        -n_smoothing)
        cl_bed_h = cl_bed_h + all_bed_h_changes

    # define new thick and clip, maximum value needed to avoid geometrical
    # inconsistencies with trapezoidal bed shape
    new_thick = cl_sfc_h - cl_bed_h
    # max
    max_h = np.where(cl_is_trap,
                     # -1. to get w0 > 0 at the end and not w0 = 0
                     cl_width / cfg.PARAMS['trapezoid_lambdas'] - 1.,
                     1e4)
    new_thick = np.where(new_thick > max_h, max_h, new_thick)
    # min
    new_thick = np.where(new_thick < min_ice_thick, min_ice_thick, new_thick)

    # new trap = (is trap or shape factor small) and not rectangular, we
    # conserve all bed shapes
    # if new bed shape is smaller than defined minimum it is converted to a
    # trapezoidal (as it is done during inversion)
    new_bed_shape = 4 * new_thick / cl_width ** 2
    new_is_trapezoid = ((cl_is_trap |
                         (new_bed_shape < cfg.PARAMS['mixed_min_shape'])) &
                        ~cl_is_rect)

    # and calculate new volumes depending on shape
    new_volume = np.where(
        new_is_trapezoid,
        cl_width * new_thick - cfg.PARAMS['trapezoid_lambdas'] / 2 *
        new_thick ** 2,
        np.where(cl_is_rect, 1, 2 / 3) * cl_width * new_thick) * cl['dx']

    # define new values
    cl['thick'][n_smoothing:] = new_thick
    cl['is_trapezoid'][n_smoothing:] = new_is_trapezoid
    cl['volume'][n_smoothing:] = new_volume

    gdir.write_pickle(cls, 'inversion_output')

    # Return volume for convenience
    return np.sum([np.sum(cl['volume']) for cl in cls])


@entity_task(log)
def get_inversion_volume(gdir):
    """Small utility task to get to the volume od all glaciers."""
    cls = gdir.read_pickle('inversion_output')
    return np.sum([np.sum(cl['volume']) for cl in cls])


@entity_task(log, writes=['inversion_output'])
def compute_velocities(*args, **kwargs):
    """Deprecated. Use compute_inversion_velocities instead."""
    warnings.warn("`compute_velocities` has been renamed to "
                  "`compute_inversion_velocities`. Prefer to use the new"
                  "name from now on.")
    return compute_inversion_velocities(*args, **kwargs)


@entity_task(log, writes=['inversion_output'])
def compute_inversion_velocities(gdir, glen_a=None, fs=None, filesuffix='',
                                 with_sliding=None):
    """Surface velocities along the flowlines from inverted ice thickness.

    Computed following the methods described in Cuffey and Paterson (2010)
    Eq. 8.35, pp 310:

        u_s = u_basal + (2A/n+1)* tau^n * H

    In the case of no sliding (or if with_sliding=False, which is a
    justifiable simplification given uncertainties on basal sliding):

        u_z/u_s = [n+1]/[n+2] (= 0.8 if n = 3).

    The output is written in 'inversion_output.pkl' in m yr-1

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    with_sliding : bool
        if set to True, we will compute velocities the sliding component.
        the default is to check if sliding was used for the inversion
        (fs != 0)
    glen_a : float
        Glen's deformation parameter A. Defaults to PARAMS['inversion_glen_a']
    fs : float
        sliding paramter, defaults to PARAMS['inversion_fs']
    filesuffix : str
        add a suffix to the output file (optional)
    """

    # Defaults
    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']

    rho = cfg.PARAMS['ice_density']
    glen_n = cfg.PARAMS['glen_n']

    if with_sliding is None:
        with_sliding = fs != 0

    # Getting the data for the main flowline
    cls = gdir.read_pickle('inversion_output')

    for cl in cls:
        # vol in m3 and dx in m
        section = cl['volume'] / cl['dx']

        # this flux is in m3 per second
        flux = cl['flux']
        angle = cl['slope_angle']
        thick = cl['thick']

        if fs > 0 and with_sliding:
            tau = rho * cfg.G * angle * thick

            with warnings.catch_warnings():
                # This can trigger a divide by zero Warning
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                u_basal = fs * tau ** glen_n / thick

            u_basal[~np.isfinite(u_basal)] = 0

            u_deformation = (2 * glen_a / (glen_n + 1)) * (tau**glen_n) * thick

            u_basal *= cfg.SEC_IN_YEAR
            u_deformation *= cfg.SEC_IN_YEAR
            u_surface = u_basal + u_deformation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                velocity = flux / section
            velocity *= cfg.SEC_IN_YEAR
        else:
            # velocity in cross section
            fac = (glen_n + 1) / (glen_n + 2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                velocity = flux / section
            velocity *= cfg.SEC_IN_YEAR
            u_surface = velocity / fac
            u_basal = velocity * np.NaN
            u_deformation = velocity * np.NaN

        # output
        cl['u_integrated'] = velocity
        cl['u_surface'] = u_surface
        cl['u_basal'] = u_basal
        cl['u_deformation'] = u_deformation

    gdir.write_pickle(cls, 'inversion_output', filesuffix=filesuffix)


@entity_task(log, writes=['gridded_data'])
def distribute_thickness_per_altitude(gdir, add_slope=True,
                                      topo_variable='topo_smoothed',
                                      smooth_radius=None,
                                      dis_from_border_exp=0.25,
                                      varname_suffix=''):
    """Compute a thickness map by redistributing mass along altitudinal bands.

    This is a rather cosmetic task, not relevant for OGGM but for ITMIX or
    for visualizations.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_slope : bool
        whether a corrective slope factor should be used or not
    topo_variable : str
        the topography to read from `gridded_data.nc` (could be smoothed, or
        smoothed differently).
    smooth_radius : int
        pixel size of the gaussian smoothing. Default is to use
        cfg.PARAMS['smooth_window'] (i.e. a size in meters). Set to zero to
        suppress smoothing.
    dis_from_border_exp : float
        the exponent of the distance from border mask
    varname_suffix : str
        add a suffix to the variable written in the file (for experiments)
    """

    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    # See if we have the masks, else compute them
    with utils.ncDataset(grids_file) as nc:
        has_masks = 'glacier_ext_erosion' in nc.variables
    if not has_masks:
        from oggm.core.gis import gridded_attributes
        gridded_attributes(gdir)

    with utils.ncDataset(grids_file) as nc:
        topo = nc.variables[topo_variable][:]
        glacier_mask = nc.variables['glacier_mask'][:]
        dis_from_border = nc.variables['dis_from_border'][:]
        if add_slope:
            slope_factor = nc.variables['slope_factor'][:]
        else:
            slope_factor = 1.

    # Along the lines
    cls = gdir.read_pickle('inversion_output')
    fls = gdir.read_pickle('inversion_flowlines')
    hs, ts, vs, xs, ys = [], [], [], [], []
    for cl, fl in zip(cls, fls):
        hs = np.append(hs, fl.surface_h)
        ts = np.append(ts, cl['thick'])
        vs = np.append(vs, cl['volume'])
        try:
            x, y = fl.line.xy
        except AttributeError:
            # Squeezed flowlines, dummy coords
            x = fl.surface_h * 0 - 1
            y = fl.surface_h * 0 - 1
        xs = np.append(xs, x)
        ys = np.append(ys, y)

    init_vol = np.sum(vs)

    # Assign a first order thickness to the points
    # very inefficient inverse distance stuff
    thick = glacier_mask * 0.
    yglac, xglac = np.nonzero(glacier_mask == 1)
    for y, x in zip(yglac, xglac):
        phgt = topo[y, x]
        # take the ones in a 100m range
        starth = 100.
        while True:
            starth += 10
            pok = np.nonzero(np.abs(phgt - hs) <= starth)[0]
            if len(pok) != 0:
                break
        sqr = np.sqrt((xs[pok]-x)**2 + (ys[pok]-y)**2)
        pzero = np.where(sqr == 0)
        if len(pzero[0]) == 0:
            thick[y, x] = np.average(ts[pok], weights=1 / sqr)
        elif len(pzero[0]) == 1:
            thick[y, x] = ts[pzero[0][0]]
        else:
            raise RuntimeError('We should not be there')

    # Distance from border (normalized)
    dis_from_border = dis_from_border**dis_from_border_exp
    dis_from_border /= np.mean(dis_from_border[glacier_mask == 1])
    thick *= dis_from_border

    # Slope
    thick *= slope_factor

    # Smooth
    dx = gdir.grid.dx
    if smooth_radius != 0:
        if smooth_radius is None:
            smooth_radius = np.rint(cfg.PARAMS['smooth_window'] / dx)
        thick = gaussian_blur(thick, int(smooth_radius))
        thick = np.where(glacier_mask, thick, 0.)

    # Re-mask
    utils.clip_min(thick, 0, out=thick)
    thick[glacier_mask == 0] = np.NaN
    assert np.all(np.isfinite(thick[glacier_mask == 1]))

    # Conserve volume
    tmp_vol = np.nansum(thick * dx**2)
    thick *= init_vol / tmp_vol

    # write
    with utils.ncDataset(grids_file, 'a') as nc:
        vn = 'distributed_thickness' + varname_suffix
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = '-'
        v.long_name = 'Distributed ice thickness'
        v[:] = thick

    return thick


@entity_task(log, writes=['gridded_data'])
def distribute_thickness_interp(gdir, add_slope=True, smooth_radius=None,
                                varname_suffix=''):
    """Compute a thickness map by interpolating between centerlines and border.

    IMPORTANT: this is NOT what has been used for ITMIX. We used
    distribute_thickness_per_altitude for ITMIX and global ITMIX.

    This is a rather cosmetic task, not relevant for OGGM but for ITMIX.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_slope : bool
        whether a corrective slope factor should be used or not
    smooth_radius : int
        pixel size of the gaussian smoothing. Default is to use
        cfg.PARAMS['smooth_window'] (i.e. a size in meters). Set to zero to
        suppress smoothing.
    varname_suffix : str
        add a suffix to the variable written in the file (for experiments)
    """

    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    # See if we have the masks, else compute them
    with utils.ncDataset(grids_file) as nc:
        has_masks = 'ice_divides' in nc.variables
    if not has_masks:
        from oggm.core.gis import gridded_attributes
        gridded_attributes(gdir)

    with utils.ncDataset(grids_file) as nc:
        glacier_mask = nc.variables['glacier_mask'][:]
        glacier_ext = nc.variables['glacier_ext_erosion'][:]
        ice_divides = nc.variables['ice_divides'][:]
        if add_slope:
            slope_factor = nc.variables['slope_factor'][:]
        else:
            slope_factor = 1.

    # Thickness to interpolate
    thick = glacier_ext * np.NaN
    thick[(glacier_ext-ice_divides) == 1] = 0.
    # TODO: domain border too, for convenience for a start
    thick[0, :] = 0.
    thick[-1, :] = 0.
    thick[:, 0] = 0.
    thick[:, -1] = 0.

    # Along the lines
    cls = gdir.read_pickle('inversion_output')
    fls = gdir.read_pickle('inversion_flowlines')
    vs = []
    for cl, fl in zip(cls, fls):
        vs.extend(cl['volume'])
        x, y = utils.tuple2int(fl.line.xy)
        thick[y, x] = cl['thick']
    init_vol = np.sum(vs)

    # Interpolate
    xx, yy = gdir.grid.ij_coordinates
    pnan = np.nonzero(~ np.isfinite(thick))
    pok = np.nonzero(np.isfinite(thick))
    points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
    inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
    thick[pnan] = griddata(points, np.ravel(thick[pok]), inter, method='cubic')
    utils.clip_min(thick, 0, out=thick)

    # Slope
    thick *= slope_factor

    # Smooth
    dx = gdir.grid.dx
    if smooth_radius != 0:
        if smooth_radius is None:
            smooth_radius = np.rint(cfg.PARAMS['smooth_window'] / dx)
        thick = gaussian_blur(thick, int(smooth_radius))
        thick = np.where(glacier_mask, thick, 0.)

    # Re-mask
    thick[glacier_mask == 0] = np.NaN
    assert np.all(np.isfinite(thick[glacier_mask == 1]))

    # Conserve volume
    tmp_vol = np.nansum(thick * dx**2)
    thick *= init_vol / tmp_vol

    # write
    grids_file = gdir.get_filepath('gridded_data')
    with utils.ncDataset(grids_file, 'a') as nc:
        vn = 'distributed_thickness' + varname_suffix
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = '-'
        v.long_name = 'Distributed ice thickness'
        v[:] = thick

    return thick


def calving_flux_from_depth(gdir, k=None, water_level=None, water_depth=None,
                            thick=None, fixed_water_depth=False):
    """Finds a calving flux from the calving front thickness.

    Approach based on Huss and Hock, (2015) and Oerlemans and Nick (2005).
    We take the initial output of the model and surface elevation data
    to calculate the water depth of the calving front.

    Parameters
    ----------
    gdir : GlacierDirectory
    k : float
        calving constant
    water_level : float
        in case water is not at 0 m a.s.l
    water_depth : float (mandatory)
        the default is to compute the water_depth from ice thickness
        at the terminus and altitude. Set this to force the water depth
        to a certain value
    thick :
        Set this to force the ice thickness to a certain value (for
        sensitivity experiments).
    fixed_water_depth :
        If we have water depth from Bathymetry we fix the water depth
        and forget about the free-board

    Returns
    -------
    A dictionary containing:
    - the calving flux in [km3 yr-1]
    - the frontal width in m
    - the frontal thickness in m
    - the frontal water depth in m
    - the frontal free board in m
    """

    # Defaults
    if k is None:
        k = cfg.PARAMS['inversion_calving_k']

    # Read necessary data
    fl = gdir.read_pickle('inversion_flowlines')[-1]

    # Altitude at the terminus and frontal width
    free_board = utils.clip_min(fl.surface_h[-1], 0) - water_level
    width = fl.widths[-1] * gdir.grid.dx

    # Calving formula
    if thick is None:
        cl = gdir.read_pickle('inversion_output')[-1]
        thick = cl['thick'][-1]
    if water_depth is None:
        water_depth = thick - free_board
    elif not fixed_water_depth:
        # Correct thickness with prescribed water depth
        # If fixed_water_depth=True then we forget about t_altitude
        thick = water_depth + free_board

    flux = k * thick * water_depth * width / 1e9

    if fixed_water_depth:
        # Recompute free board before returning
        free_board = thick - water_depth

    return {'flux': utils.clip_min(flux, 0),
            'width': width,
            'thick': thick,
            'inversion_calving_k': k,
            'water_depth': water_depth,
            'water_level': water_level,
            'free_board': free_board}


@entity_task(log, writes=['diagnostics'])
def find_inversion_calving_from_any_mb(gdir, mb_model=None, mb_years=None,
                                       water_level=None,
                                       glen_a=None, fs=None):
    """Optimized search for a calving flux compatible with the bed inversion.

    See Recinos et al. (2019) for details. This task is an update to
    `find_inversion_calving` but acting upon the MB residual (i.e. a shift)
    instead of the model temperature sensitivity.

    Parameters
    ----------
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass balance model to use
    mb_years : array
        the array of years from which you want to average the MB for (for
        mb_model only).
    water_level : float
        the water level. It should be zero m a.s.l, but:
        - sometimes the frontal elevation is unrealistically high (or low).
        - lake terminating glaciers
        - other uncertainties
        With this parameter, you can produce more realistic values. The default
        is to infer the water level from PARAMS['free_board_lake_terminating']
        and PARAMS['free_board_marine_terminating']
    glen_a : float, optional
    fs : float, optional
    """
    from oggm.core import massbalance

    if not gdir.is_tidewater or not cfg.PARAMS['use_kcalving_for_inversion']:
        # Do nothing
        return

    # Let's start from a fresh state
    gdir.inversion_calving_rate = 0
    with utils.DisableLogger():
        massbalance.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
                                            mb_years=mb_years)
        prepare_for_inversion(gdir)
        v_ref = mass_conservation_inversion(gdir, water_level=water_level,
                                            glen_a=glen_a, fs=fs)

    # Store for statistics
    gdir.add_to_diagnostics('volume_before_calving', v_ref)

    # Get the relevant variables
    cls = gdir.read_pickle('inversion_input')[-1]
    slope = cls['slope_angle'][-1]
    width = cls['width'][-1]

    # Stupidly enough the slope is clipped in the OGGM inversion, not
    # in inversion prepro - clip here
    min_slope = 'min_slope_ice_caps' if gdir.is_icecap else 'min_slope'
    min_slope = np.deg2rad(cfg.PARAMS[min_slope])
    slope = utils.clip_array(slope, min_slope, np.pi / 2.)

    # Check that water level is within given bounds
    if water_level is None:
        th = cls['hgt'][-1]
        if gdir.is_lake_terminating:
            water_level = th - cfg.PARAMS['free_board_lake_terminating']
        else:
            vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
            water_level = utils.clip_scalar(0, th - vmax, th - vmin)

    # The functions all have the same shape: they decrease, then increase
    # We seek the absolute minimum first
    def to_minimize(h):
        fl = calving_flux_from_depth(gdir, water_level=water_level,
                                     water_depth=h)

        flux = fl['flux'] * 1e9 / cfg.SEC_IN_YEAR
        sia_thick = sia_thickness(slope, width, flux, glen_a=glen_a, fs=fs)
        return fl['thick'] - sia_thick

    abs_min = optimize.minimize(to_minimize, [1], bounds=((1e-4, 1e4), ),
                                tol=1e-1)
    if not abs_min['success']:
        raise RuntimeError('Could not find the absolute minimum in calving '
                           'flux optimization: {}'.format(abs_min))
    if abs_min['fun'] > 0:
        # This happens, and means that this glacier simply can't calve
        # This is an indicator for physics not matching, often an unrealistic
        # slope of free-board
        out = calving_flux_from_depth(gdir, water_level=water_level)

        log.warning('({}) find_inversion_calving_from_any_mb: could not find '
                    'calving flux.'.format(gdir.rgi_id))

        odf = dict()
        odf['calving_flux'] = 0
        odf['calving_rate_myr'] = 0
        odf['calving_law_flux'] = out['flux']
        odf['calving_water_level'] = out['water_level']
        odf['calving_inversion_k'] = out['inversion_calving_k']
        odf['calving_front_slope'] = slope
        odf['calving_front_water_depth'] = out['water_depth']
        odf['calving_front_free_board'] = out['free_board']
        odf['calving_front_thick'] = out['thick']
        odf['calving_front_width'] = out['width']
        for k, v in odf.items():
            gdir.add_to_diagnostics(k, v)
        return odf

    # OK, we now find the zero between abs min and an arbitrary high front
    abs_min = abs_min['x'][0]
    opt = optimize.brentq(to_minimize, abs_min, 1e4)

    # Give the flux to the inversion and recompute
    # This is the thick guaranteeing OGGM Flux = Calving Law Flux
    out = calving_flux_from_depth(gdir, water_level=water_level,
                                  water_depth=opt)
    f_calving = out['flux']

    log.info('({}) find_inversion_calving_from_any_mb: found calving flux of '
             '{:.03f} km3 yr-1'.format(gdir.rgi_id, f_calving))
    gdir.inversion_calving_rate = f_calving

    with utils.DisableLogger():
        massbalance.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
                                            mb_years=mb_years)
        prepare_for_inversion(gdir)
        mass_conservation_inversion(gdir, water_level=water_level,
                                    glen_a=glen_a, fs=fs)

    out = calving_flux_from_depth(gdir, water_level=water_level)

    fl = gdir.read_pickle('inversion_flowlines')[-1]
    f_calving = (fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9 /
                 cfg.PARAMS['ice_density'])

    # Store results
    odf = dict()
    odf['calving_flux'] = f_calving
    odf['calving_rate_myr'] = f_calving * 1e9 / (out['thick'] * out['width'])
    odf['calving_law_flux'] = out['flux']
    odf['calving_water_level'] = out['water_level']
    odf['calving_inversion_k'] = out['inversion_calving_k']
    odf['calving_front_slope'] = slope
    odf['calving_front_water_depth'] = out['water_depth']
    odf['calving_front_free_board'] = out['free_board']
    odf['calving_front_thick'] = out['thick']
    odf['calving_front_width'] = out['width']
    for k, v in odf.items():
        gdir.add_to_diagnostics(k, v)

    return odf
