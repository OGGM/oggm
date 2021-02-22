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
def prepare_for_inversion(gdir, add_debug_var=False,
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
        flux = fl.flux * (gdir.grid.dx**2) / cfg.SEC_IN_YEAR / rho

        # Clip flux to 0
        if np.any(flux < -0.1):
            log.info('(%s) has negative flux somewhere', gdir.rgi_id)
        utils.clip_min(flux, 0, out=flux)

        if np.sum(flux <= 0) > 1 and len(fls) == 1:
            log.warning("More than one grid point has zero or "
                        "negative flux: this should not happen.")

        if fl.flows_to is None and gdir.inversion_calving_rate == 0:
            if not np.allclose(flux[-1], 0., atol=0.1):
                # TODO: this test doesn't seem meaningful here
                msg = ('({}) flux at terminus should be zero, but is: '
                       '{.4f} m3 ice s-1'.format(gdir.rgi_id, flux[-1]))
                raise RuntimeError(msg)

            # This contradicts the statement above which has been around for
            # quite some time, for the reason that it is a quality check: per
            # construction, the flux at the last grid point should be zero
            # HOWEVER, it is also meaningful to have a non-zero ice thickness
            # at the last grid point. Therefore, we add some artificial
            # flux here (an alternative would be to pmute the flux on a
            # staggered grid but I actually like the QC and its easier)
            # note that this value will be ignored if one uses the filter
            # task afterwards
            flux[-1] = flux[-2] / 3  # this is totally arbitrary

        if fl.flows_to is not None and flux[-1] <= 0:
            # Same for tributaries
            flux[-1] = flux[-2] / 3  # this is totally arbitrary

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


def _compute_thick(a0s, a3, flux_a0, shape_factor, _inv_function):
    """Content of the original inner loop of the mass-conservation inversion.

    Put here to avoid code duplication.

    Parameters
    ----------
    a0s
    a3
    flux_a0
    shape_factor
    _inv_function

    Returns
    -------
    the thickness
    """

    a0s = a0s / (shape_factor ** 3)

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

    # Inversion with shape factors?
    sf_func = None
    if shape_factor == 'Adhikari' or shape_factor == 'Nye':
        sf_func = utils.shape_factor_adhikari
    elif shape_factor == 'Huss':
        sf_func = utils.shape_factor_huss

    sf = np.ones(slope.shape)  # Default shape factor is 1
    if sf_func is not None:

        # Start iteration for shape factor with first guess of 1
        i = 0
        sf_diff = np.ones(slope.shape)

        # Some hard-coded factors here
        sf_tol = 1e-2
        max_sf_iter = 20

        while i < max_sf_iter and np.any(sf_diff > sf_tol):
            out_thick = _compute_thick(a0, a3, flux_a0, sf, _inv_function)
            is_rectangular = np.repeat(shape == 'rectangular', len(width))
            sf_diff[:] = sf[:]
            sf = sf_func(width, out_thick, is_rectangular)
            sf_diff = sf_diff - sf
            i += 1

        log.info('Shape factor {:s} used, took {:d} iterations for '
                 'convergence.'.format(shape_factor, i))

    return _compute_thick(a0, a3, flux_a0, sf, _inv_function)


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

    # Inversion with shape factors?
    sf_func = None
    use_sf = cfg.PARAMS.get('use_shape_factor_for_inversion', None)
    if use_sf == 'Adhikari' or use_sf == 'Nye':
        sf_func = utils.shape_factor_adhikari
    elif use_sf == 'Huss':
        sf_func = utils.shape_factor_huss

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

        sf = np.ones(slope.shape)  # Default shape factor is 1
        if sf_func is not None:

            # Start iteration for shape factor with first guess of 1
            i = 0
            sf_diff = np.ones(slope.shape)

            # Some hard-coded factors here
            sf_tol = 1e-2
            max_sf_iter = 20

            while i < max_sf_iter and np.any(sf_diff > sf_tol):
                out_thick = _compute_thick(a0s, a3, cl['flux_a0'], sf,
                                           _inv_function)

                sf_diff[:] = sf[:]
                sf = sf_func(w, out_thick, cl['is_rectangular'])
                sf_diff = sf_diff - sf
                i += 1

            log.info('Shape factor {:s} used, took {:d} iterations for '
                     'convergence.'.format(use_sf, i))

            # TODO: possible shape factor optimisations
            # thick update could be used as iteration end criterion instead
            # we iterate for all grid points, even if some already converged

        out_thick = _compute_thick(a0s, a3, cl['flux_a0'], sf, _inv_function)

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
        if np.any(out_thick <= 0):
            log.warning("Found zero or negative thickness: "
                        "this should not happen.")

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
def filter_inversion_output(gdir):
    """Filters the last few grid points after the physically-based inversion.

    For various reasons (but mostly: the equilibrium assumption), the last few
    grid points on a glacier flowline are often noisy and create unphysical
    depressions. Here we try to correct for that. It is not volume conserving,
    but area conserving.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
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
    bs = np.average(dic_ds['bedshapes'][:3])

    n = -5

    cls = gdir.read_pickle('inversion_output')
    cl = cls[-1]

    # First guess thickness based on width
    w = cl['width'][n:]
    old_h = cl['thick'][n:]
    s = w**3 * bs / 6
    new_h = 3/2 * s / w
    # Change only if it actually does what we want
    new_h[old_h < new_h] = old_h[old_h < new_h]

    # Smoothing things out a bit
    hts = np.append(np.append(cl['thick'][n-3:n], new_h), 0)
    h = utils.smooth1d(hts, 3)[n-1:-1]

    # Recompute bedshape based on that
    bs = utils.clip_min(4*h / w**2, cfg.PARAMS['mixed_min_shape'])

    # OK, done
    s = w**3 * bs / 6

    # Change only if it actually does what we want
    new_h = 3/2 * s / w
    if np.any(new_h > old_h):
        # No change in volume
        return np.sum([np.sum(cl['volume']) for cl in cls])

    cl['thick'][n:] = new_h
    cl['volume'][n:] = s * cl['dx']
    cl['is_trapezoid'][n:] = False
    cl['is_rectangular'][n:] = False

    gdir.write_pickle(cls, 'inversion_output')

    # Return volume for convenience
    return np.sum([np.sum(cl['volume']) for cl in cls])


@entity_task(log)
def get_inversion_volume(gdir):
    """Small utility task to get to the volume od all glaciers."""
    cls = gdir.read_pickle('inversion_output')
    return np.sum([np.sum(cl['volume']) for cl in cls])


@entity_task(log, writes=['inversion_output'])
def compute_velocities(gdir, glen_a=None, fs=None, filesuffix=''):
    """Surface velocities along the flowlines from inverted ice thickness.

    Computed following the methods described in
    Cuffey and Paterson (2010) Eq. 8.35, pp 310:

        u_s = u_basal + (2A/n+1)* tau^n * H

    In the case of no sliding:

        u_z/u_s = [n+1]/[n+2] = 0.8 if n = 3.

    The output is written in 'inversion_output.pkl' in m yr-1

    You'll need to call prepare_for_inversion with the `add_debug_var=True`
    kwarg for this to work!

    Parameters
    ----------
    gdir : Glacier directory
    with_sliding : bool
        default is True, if set to False will not add the sliding component.
    filesuffix : str
        add a suffix to the output file
    """

    # Defaults
    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']

    rho = cfg.PARAMS['ice_density']
    glen_n = cfg.PARAMS['glen_n']

    # Getting the data for the main flowline
    cls = gdir.read_pickle('inversion_output')

    for cl in cls:
        # vol in m3 and dx in m
        section = cl['volume'] / cl['dx']

        # this flux is in m3 per second
        flux = cl['flux']
        angle = cl['slope_angle']
        thick = cl['thick']

        if fs > 0:
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
            u_basal = velocity * 0
            u_deformation = velocity * 0

        # output
        cl['u_integrated'] = velocity
        cl['u_surface'] = u_surface
        cl['u_basal'] = u_basal
        cl['u_deformation'] = u_deformation

    gdir.write_pickle(cls, 'inversion_output', filesuffix=filesuffix)


@entity_task(log, writes=['gridded_data'])
def distribute_thickness_per_altitude(gdir, add_slope=True,
                                      smooth_radius=None,
                                      dis_from_border_exp=0.25,
                                      varname_suffix=''):
    """Compute a thickness map by redistributing mass along altitudinal bands.

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
        topo_smoothed = nc.variables['topo_smoothed'][:]
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
    thick = glacier_mask * np.NaN
    for y in range(thick.shape[0]):
        for x in range(thick.shape[1]):
            phgt = topo_smoothed[y, x]
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
                thick[y, x] = ts[pzero]
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
def find_inversion_calving(gdir, water_level=None, fixed_water_depth=None,
                           glen_a=None, fs=None, min_mu_star_frac=None):
    """Optimized search for a calving flux compatible with the bed inversion.

    See Recinos et al 2019 for details.

    Parameters
    ----------
    water_level : float
        the water level. It should be zero m a.s.l, but:
        - sometimes the frontal elevation is unrealistically high (or low).
        - lake terminating glaciers
        - other uncertainties
        With this parameter, you can produce more realistic values. The default
        is to infer the water level from PARAMS['free_board_lake_terminating']
        and PARAMS['free_board_marine_terminating']
    fixed_water_depth : float
        fix the water depth to an observed value and let the free board vary
        instead.
    glen_a : float, optional
    fs : float, optional
    min_mu_star_frac : float, optional
        fraction of the original (non-calving) mu* you are ready to allow
        for. Defaults cfg.PARAMS['calving_min_mu_star_frac'].
    """
    from oggm.core import climate
    from oggm.exceptions import MassBalanceCalibrationError

    if not gdir.is_tidewater or not cfg.PARAMS['use_kcalving_for_inversion']:
        # Do nothing
        return

    if min_mu_star_frac is None:
        min_mu_star_frac = cfg.PARAMS['calving_min_mu_star_frac']

    # Let's start from a fresh state
    gdir.inversion_calving_rate = 0

    with utils.DisableLogger():
        climate.local_t_star(gdir)
        climate.mu_star_calibration(gdir)
        prepare_for_inversion(gdir)
        v_ref = mass_conservation_inversion(gdir, water_level=water_level,
                                            glen_a=glen_a, fs=fs)

    # We have a stop condition on mu*
    prev_params = gdir.read_json('local_mustar')
    mu_star_orig = np.min(prev_params['mu_star_per_flowline'])

    # Store for statistics
    gdir.add_to_diagnostics('volume_before_calving', v_ref)
    gdir.add_to_diagnostics('mu_star_before_calving', mu_star_orig)

    # Get the relevant variables
    cls = gdir.read_pickle('inversion_input')[-1]
    slope = cls['slope_angle'][-1]
    width = cls['width'][-1]

    # Stupidly enough the slope is clipped in the OGGM inversion, not
    # in prepro - clip here
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
        if fixed_water_depth is not None:
            fl = calving_flux_from_depth(gdir, thick=h,
                                         water_level=water_level,
                                         water_depth=fixed_water_depth,
                                         fixed_water_depth=True)
        else:
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
        # This is an indicator for physics not matching, often a unrealistic
        # slope of free-board
        df = gdir.read_json('local_mustar')
        out = calving_flux_from_depth(gdir, water_level=water_level)

        log.warning('({}) find_inversion_calving: could not find '
                    'calving flux.'.format(gdir.rgi_id))

        odf = dict()
        odf['calving_flux'] = 0
        odf['calving_rate_myr'] = 0
        odf['calving_mu_star'] = df['mu_star_glacierwide']
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
        return

    # OK, we now find the zero between abs min and an arbitrary high front
    abs_min = abs_min['x'][0]
    opt = optimize.brentq(to_minimize, abs_min, 1e4)

    # This is the thick guaranteeing OGGM Flux = Calving Law Flux
    # Let's see if it results in a meaningful mu_star

    # Give the flux to the inversion and recompute
    if fixed_water_depth is not None:
        out = calving_flux_from_depth(gdir, water_level=water_level, thick=opt,
                                      water_depth=fixed_water_depth,
                                      fixed_water_depth=True)
        f_calving = out['flux']
    else:
        out = calving_flux_from_depth(gdir, water_level=water_level,
                                      water_depth=opt)
        f_calving = out['flux']

    gdir.inversion_calving_rate = f_calving

    with utils.DisableLogger():
        # We accept values down to zero before stopping
        min_mu_star = mu_star_orig * min_mu_star_frac

        # At this step we might raise a MassBalanceCalibrationError
        try:
            climate.local_t_star(gdir, clip_mu_star=False,
                                 min_mu_star=min_mu_star,
                                 continue_on_error=False,
                                 add_to_log_file=False)
            df = gdir.read_json('local_mustar')
        except MassBalanceCalibrationError as e:
            assert 'mu* out of specified bounds' in str(e)
            # When this happens we clip mu*
            climate.local_t_star(gdir, clip_mu_star=True,
                                 min_mu_star=min_mu_star)
            df = gdir.read_json('local_mustar')

        climate.mu_star_calibration(gdir, min_mu_star=min_mu_star)
        prepare_for_inversion(gdir)
        mass_conservation_inversion(gdir, water_level=water_level,
                                    glen_a=glen_a, fs=fs)

    if fixed_water_depth is not None:
        out = calving_flux_from_depth(gdir, water_level=water_level,
                                      water_depth=fixed_water_depth,
                                      fixed_water_depth=True)
    else:
        out = calving_flux_from_depth(gdir, water_level=water_level)

    fl = gdir.read_pickle('inversion_flowlines')[-1]
    f_calving = (fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9 /
                 cfg.PARAMS['ice_density'])

    log.info('({}) find_inversion_calving_from_any_mb: found calving flux of '
             '{:.03f} km3 yr-1'.format(gdir.rgi_id, f_calving))

    # Store results
    odf = dict()
    odf['calving_flux'] = f_calving
    odf['calving_rate_myr'] = f_calving * 1e9 / (out['thick'] * out['width'])
    odf['calving_mu_star'] = df['mu_star_glacierwide']
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


@entity_task(log, writes=['diagnostics'])
def find_inversion_calving_from_any_mb(gdir, mb_model=None, mb_years=None,
                                       water_level=None,
                                       glen_a=None, fs=None):
    """Optimized search for a calving flux compatible with the bed inversion.

    See Recinos et al 2019 for details. This task is an update to
    `find_inversion_calving` but acting upon a MB residual (i.e. a shift)
    instead of the model temperature sensitivity.

    Parameters
    ----------
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass-balance model to use
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
    from oggm.core import climate

    if not gdir.is_tidewater or not cfg.PARAMS['use_kcalving_for_inversion']:
        # Do nothing
        return

    # Let's start from a fresh state
    gdir.inversion_calving_rate = 0
    with utils.DisableLogger():
        climate.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
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
        # This is an indicator for physics not matching, often a unrealistic
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
        return

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
        climate.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
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
