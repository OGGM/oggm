"""Glacier thickness.


Note for later: the current code is oriented towards a consistent framework
for flowline modelling. The major direction shift happens at the
flowlines.width_correction step where the widths are computed to follow the
altitude area distribution. This must not and should not be the case when
the actual objective is to provide a glacier thickness map. For this,
the geometrical width and some other criterias such as e.g. "all altitudes
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
import pandas as pd
from scipy.interpolate import griddata
from scipy import optimize

# Locals
from oggm import utils, cfg
from oggm import entity_task
from oggm.core.gis import gaussian_blur
from oggm.exceptions import InvalidParamsError

# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['inversion_input'])
def prepare_for_inversion(gdir, add_debug_var=False,
                          invert_with_rectangular=True,
                          invert_all_rectangular=False):
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
            log.warning('(%s) has negative flux somewhere', gdir.rgi_id)
        utils.clip_min(flux, 0, out=flux)

        if fl.flows_to is None and gdir.inversion_calving_rate == 0:
            if not np.allclose(flux[-1], 0., atol=0.1):
                # TODO: this test doesn't seem meaningful here
                msg = ('({}) flux at terminus should be zero, but is: '
                       '{.4f} m3 ice s-1'.format(gdir.rgi_id, flux[-1]))
                raise RuntimeError(msg)
            flux[-1] = 0.

        # Shape
        is_rectangular = fl.is_rectangular
        if not invert_with_rectangular:
            is_rectangular[:] = False
        if invert_all_rectangular:
            is_rectangular[:] = True

        # Optimisation: we need to compute this term of a0 only once
        flux_a0 = np.where(is_rectangular, 1, 1.5)
        flux_a0 *= flux / widths

        # Add to output
        cl_dic = dict(dx=dx, flux_a0=flux_a0, width=widths,
                      slope_angle=angle, is_rectangular=is_rectangular,
                      is_last=fl.flows_to is None)
        if add_debug_var:
            cl_dic['flux'] = flux
            cl_dic['hgt'] = hgt
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


def sia_thickness(slope, width, flux, shape='rectangular',
                  glen_a=None, fs=None, shape_factor=None):
    """Computes the ice thickness from mass-conservation.

    This is a utility function tested against the true OGGM inversion
    function. Useful for teaching and inversion with calving.

    Parameters
    ----------
    slope : -np.gradient(hgt, dx)
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
        raise InvalidParamsError('shape must be `parabolic` or `rectangular`,'
                                 'not: {}'.format(shape))

    _inv_function = _inversion_simple if fs == 0 else _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.PARAMS['glen_n']+2) * glen_a
    rho = cfg.PARAMS['ice_density']

    # Clip the slope, in degrees
    clip_angle = cfg.PARAMS['min_slope']

    # Clip slope to avoid negative and small slopes
    slope = utils.clip_array(slope, np.deg2rad(clip_angle), np.pi / 2.)

    # Convert the flux to m2 s-1 (averaged to represent the sections center)
    flux_a0 = 1 if shape == 'rectangular' else 1.5
    flux_a0 *= flux / width

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


@entity_task(log, writes=['inversion_output'])
def mass_conservation_inversion(gdir, glen_a=None, fs=None, write=True,
                                filesuffix=''):
    """ Compute the glacier thickness along the flowlines

    More or less following Farinotti et al., (2009).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    glen_a : float
        glen's creep parameter A
    fs : float
        sliding parameter
    write: bool
        default behavior is to compute the thickness and write the
        results in the pickle. Set to False in order to spare time
        during calibration.
    filesuffix : str
        add a suffix to the output file
    """

    # Defaults
    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']

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

    # Clip the slope, in degrees
    clip_angle = cfg.PARAMS['min_slope']

    out_volume = 0.

    cls = gdir.read_pickle('inversion_input')
    for cl in cls:
        # Clip slope to avoid negative and small slopes
        slope = cl['slope_angle']
        slope = utils.clip_array(slope, np.deg2rad(clip_angle), np.pi/2.)

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
        fac = np.where(cl['is_rectangular'], 1, 2./3.)
        volume = fac * out_thick * w * cl['dx']
        if write:
            cl['thick'] = out_thick
            cl['volume'] = volume
        out_volume += np.sum(volume)

    if write:
        gdir.write_pickle(cls, 'inversion_output', filesuffix=filesuffix)

    return out_volume, gdir.rgi_area_km2 * 1e6


@entity_task(log, writes=['inversion_output'])
def volume_inversion(gdir, glen_a=None, fs=None, filesuffix=''):
    """Computes the inversion the glacier.

    If glen_a and fs are not given, it will use the optimized params.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    glen_a : float, optional
        the ice creep parameter (defaults to cfg.PARAMS['inversion_glen_a'])
    fs : float, optional
        the sliding parameter (defaults to cfg.PARAMS['inversion_fs'])
    fs : float, optional
        the sliding parameter (defaults to cfg.PARAMS['inversion_fs'])
    filesuffix : str
        add a suffix to the output file
    """

    warnings.warn('The task `volume_inversion` is deprecated. Use '
                  'a direct call to `mass_conservation_inversion` instead.',
                  DeprecationWarning)

    if fs is not None and glen_a is None:
        raise ValueError('Cannot set fs without glen_a.')

    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']

    if fs is None:
        fs = cfg.PARAMS['inversion_fs']

    # go
    return mass_conservation_inversion(gdir, glen_a=glen_a, fs=fs, write=True,
                                       filesuffix=filesuffix)


@entity_task(log, writes=['inversion_output'])
def filter_inversion_output(gdir):
    """Filters the last few grid point whilst conserving total volume.

    The last few grid points sometimes are noisy or can have a negative slope.
    This function filters them while conserving the total volume.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    if gdir.is_tidewater:
        # No need for filter in tidewater case
        return

    cls = gdir.read_pickle('inversion_output')
    for cl in cls:

        init_vol = np.sum(cl['volume'])
        if init_vol == 0 or not cl['is_last']:
            continue

        w = cl['width']
        out_thick = cl['thick']
        fac = np.where(cl['is_rectangular'], 1, 2./3.)

        # Last thicknesses can be noisy sometimes: interpolate
        out_thick[-4:] = np.NaN
        out_thick = utils.interp_nans(np.append(out_thick, 0))[:-1]
        assert len(out_thick) == len(fac)

        # final volume
        volume = fac * out_thick * w * cl['dx']

        # conserve it
        new_vol = np.nansum(volume)
        if new_vol == 0:
            # Very small glaciers
            return
        volume = init_vol / new_vol * volume
        np.testing.assert_allclose(np.nansum(volume), init_vol)

        # recompute thickness on that base
        out_thick = volume / (fac * w * cl['dx'])

        # output
        cl['thick'] = out_thick
        cl['volume'] = volume

    gdir.write_pickle(cls, 'inversion_output')


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
        x, y = fl.line.xy
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
        thick = gaussian_blur(thick, np.int(smooth_radius))
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
        has_masks = 'glacier_ext_erosion' in nc.variables
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
        thick = gaussian_blur(thick, np.int(smooth_radius))
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


def calving_flux_from_depth(gdir, k=None, water_depth=None, thick=None,
                            fixed_water_depth=False):
    """Finds a calving flux from the calving front thickness.

    Approach based on Huss and Hock, (2015) and Oerlemans and Nick (2005).
    We take the initial output of the model and surface elevation data
    to calculate the water depth of the calving front.

    Parameters
    ----------
    gdir : GlacierDirectory
    k : float
        calving constant
    water_depth :
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
        k = cfg.PARAMS['k_calving']

    # Read inversion output
    cl = gdir.read_pickle('inversion_output')[-1]
    fl = gdir.read_pickle('inversion_flowlines')[-1]

    # Altitude at the terminus and frontal width
    t_altitude = utils.clip_min(fl.surface_h[-1], 0)
    width = fl.widths[-1] * gdir.grid.dx

    # Calving formula
    if thick is None:
        thick = cl['thick'][-1]
    if water_depth is None:
        water_depth = thick - t_altitude
    elif not fixed_water_depth:
        # Correct thickness with prescribed water depth
        # If fixed_water_depth=True then we forget about t_altitude
        thick = water_depth + t_altitude

    flux = k * thick * water_depth * width / 1e9

    if fixed_water_depth:
        # Recompute free board before returning
        t_altitude = thick - water_depth

    return {'flux': utils.clip_min(flux, 0),
            'width': width,
            'thick': thick,
            'water_depth': water_depth,
            'free_board': t_altitude}


def _calving_fallback():
    """Restore defaults in case we exit with error"""

    # Bounds on mu*
    cfg.PARAMS['min_mu_star'] = 1.
    # Whether to clip mu to a min of zero (only recommended for calving exps)
    cfg.PARAMS['clip_mu_star'] = False


@entity_task(log, writes=['calving_loop'], fallback=_calving_fallback)
def find_inversion_calving_loop(gdir, initial_water_depth=None, max_ite=30,
                                stop_after_convergence=True,
                                fixed_water_depth=False):
    """Iterative search for a calving flux compatible with the bed inversion.

    See Recinos et al 2019 for details.

    Parameters
    ----------
    initial_water_depth : float
        the initial water depth starting the loop (for sensitivity experiments
        or to fix it to an observed value). The default is to use 1/3 of the
        terminus elevation if > 10 m, and 10 m otherwise
    max_ite : int
        the maximal number of iterations allowed before raising an error
    stop_after_convergence : bool
        continue to loop after convergence is reached
        (for sensitivity experiments)
    fixed_water_depth : bool
        fix the water depth and let the frontal altitude vary instead
    """

    # Shortcuts
    from oggm.core import climate, inversion
    from oggm.exceptions import MassBalanceCalibrationError

    # Input
    if initial_water_depth is None:
        fl = gdir.read_pickle('inversion_flowlines')[-1]
        initial_water_depth = utils.clip_min(fl.surface_h[-1] / 3, 10)

    rho = cfg.PARAMS['ice_density']

    # We accept values down to zero before stopping
    cfg.PARAMS['min_mu_star'] = 0

    # Start iteration
    i = 0
    cfg.PARAMS['clip_mu_star'] = False
    odf = pd.DataFrame()
    mu_is_zero = False
    while i < max_ite:

        # Calculates a calving flux from model output
        if i == 0:
            # First call we set to zero (it's just to be sure we start
            # from a non-calving glacier)
            f_calving = 0
        elif i == 1:
            # Second call, we set a small positive calving to start with

            # Default is to get the thickness from free board and
            # initial water depth
            thick = None
            if fixed_water_depth:
                # This leaves the free board open for change
                thick = initial_water_depth + 1
            out = calving_flux_from_depth(gdir,
                                          water_depth=initial_water_depth,
                                          thick=thick,
                                          fixed_water_depth=fixed_water_depth)
            f_calving = out['flux']
        elif cfg.PARAMS['clip_mu_star']:
            # If we had to clip mu, the inversion calving becomes the real
            # flux, i.e. not compatible with calving law but with the
            # inversion
            fl = gdir.read_pickle('inversion_flowlines')[-1]
            f_calving = fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9 / rho
            mu_is_zero = True
        else:
            # Otherwise it is parameterized by the calving law
            if fixed_water_depth:
                out = calving_flux_from_depth(gdir,
                                              water_depth=initial_water_depth,
                                              fixed_water_depth=True)
                f_calving = out['flux']
            else:
                f_calving = calving_flux_from_depth(gdir)['flux']

        # Give it back to the inversion and recompute
        gdir.inversion_calving_rate = f_calving

        # At this step we might raise a MassBalanceCalibrationError
        try:
            climate.local_t_star(gdir)
            df = gdir.read_json('local_mustar')
        except MassBalanceCalibrationError as e:
            assert 'mu* out of specified bounds' in str(e)
            # When this happens we clip mu* to zero and store the
            # bad value (just for plotting)
            cfg.PARAMS['clip_mu_star'] = True
            df = gdir.read_json('local_mustar')
            df['mu_star_glacierwide'] = float(str(e).split(':')[-1])
            climate.local_t_star(gdir)

        climate.mu_star_calibration(gdir)

        inversion.prepare_for_inversion(gdir, add_debug_var=True)
        v_inv, _ = inversion.mass_conservation_inversion(gdir)
        if fixed_water_depth:
            out = calving_flux_from_depth(gdir,
                                          water_depth=initial_water_depth,
                                          fixed_water_depth=True)
        else:
            out = calving_flux_from_depth(gdir)

        # Store the data
        odf.loc[i, 'calving_flux'] = f_calving
        odf.loc[i, 'mu_star'] = df['mu_star_glacierwide']
        odf.loc[i, 'calving_law_flux'] = out['flux']
        odf.loc[i, 'width'] = out['width']
        odf.loc[i, 'thick'] = out['thick']
        odf.loc[i, 'water_depth'] = out['water_depth']
        odf.loc[i, 'free_board'] = out['free_board']

        # Do we have to do another_loop? Start testing at 5th iteration
        calving_flux = odf.calving_flux.values
        if stop_after_convergence and i > 4:
            # We want to make sure that we don't converge by chance
            # so we test on last two iterations
            conv = (np.allclose(calving_flux[[-1, -2]],
                                [out['flux'], out['flux']],
                                rtol=0.01))
            if mu_is_zero or conv:
                break
        i += 1

    # Write output
    odf.index.name = 'iterations'
    odf.to_csv(gdir.get_filepath('calving_loop'))

    # Restore defaults
    cfg.PARAMS['min_mu_star'] = 1.
    cfg.PARAMS['clip_mu_star'] = False

    return odf


@entity_task(log, writes=['diagnostics'], fallback=_calving_fallback)
def find_inversion_calving(gdir, fixed_water_depth=None):
    """Optimized search for a calving flux compatible with the bed inversion.

    See Recinos et al 2019 for details.

    Parameters
    ----------
    fixed_water_depth : float
        fix the water depth to an observed value and let the free board vary
        instead.
    """
    from oggm.core import climate, inversion
    from oggm.exceptions import MassBalanceCalibrationError

    # Let's start from a fresh state
    gdir.inversion_calving_rate = 0
    climate.local_t_star(gdir)
    climate.mu_star_calibration(gdir)
    inversion.prepare_for_inversion(gdir, add_debug_var=True)
    inversion.mass_conservation_inversion(gdir)

    # Get the relevant variables
    cls = gdir.read_pickle('inversion_input')[-1]
    slope = cls['slope_angle'][-1]
    width = cls['width'][-1]

    # The functions all have the same shape: they decrease, then increase
    # We seek the absolute minimum first
    def to_minimize(h):
        if fixed_water_depth is not None:
            fl = calving_flux_from_depth(gdir, thick=h,
                                         water_depth=fixed_water_depth,
                                         fixed_water_depth=True)
        else:
            fl = calving_flux_from_depth(gdir, water_depth=h)

        flux = fl['flux'] * 1e9 / cfg.SEC_IN_YEAR
        sia_thick = sia_thickness(slope, width, flux)
        return fl['thick'] - sia_thick

    abs_min = optimize.minimize(to_minimize, [1], bounds=((1e-4, 1e4), ),
                                tol=1e-1)
    if not abs_min['success']:
        raise RuntimeError('Could not find the absolute minimum in calving '
                           'flux optimization: {}'.format(abs_min))
    if abs_min['fun'] > 0:
        # This happens, and means that this glacier simply can't calve
        # See e.g. RGI60-01.23642
        df = gdir.read_json('local_mustar')
        out = calving_flux_from_depth(gdir)

        odf = dict()
        odf['calving_flux'] = 0
        odf['calving_mu_star'] = df['mu_star_glacierwide']
        odf['calving_law_flux'] = out['flux']
        odf['calving_slope'] = slope
        odf['calving_thick'] = out['thick']
        odf['calving_water_depth'] = out['water_depth']
        odf['calving_free_board'] = out['free_board']
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
        out = calving_flux_from_depth(gdir, thick=opt,
                                      water_depth=fixed_water_depth,
                                      fixed_water_depth=True)
        f_calving = out['flux']
    else:
        out = calving_flux_from_depth(gdir, water_depth=opt)
        f_calving = out['flux']

    gdir.inversion_calving_rate = f_calving

    # We accept values down to zero before stopping
    cfg.PARAMS['min_mu_star'] = 0
    cfg.PARAMS['clip_mu_star'] = False

    # At this step we might raise a MassBalanceCalibrationError
    try:
        climate.local_t_star(gdir)
        df = gdir.read_json('local_mustar')
    except MassBalanceCalibrationError as e:
        assert 'mu* out of specified bounds' in str(e)
        # When this happens we clip mu* to zero
        cfg.PARAMS['clip_mu_star'] = True
        climate.local_t_star(gdir)
        df = gdir.read_json('local_mustar')

    climate.mu_star_calibration(gdir)
    inversion.prepare_for_inversion(gdir, add_debug_var=True)
    inversion.mass_conservation_inversion(gdir)

    if fixed_water_depth is not None:
        out = calving_flux_from_depth(gdir,
                                      water_depth=fixed_water_depth,
                                      fixed_water_depth=True)
    else:
        out = calving_flux_from_depth(gdir)

    fl = gdir.read_pickle('inversion_flowlines')[-1]
    f_calving = (fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9 /
                 cfg.PARAMS['ice_density'])

    # Store results
    odf = dict()
    odf['calving_flux'] = f_calving
    odf['calving_mu_star'] = df['mu_star_glacierwide']
    odf['calving_law_flux'] = out['flux']
    odf['calving_slope'] = slope
    odf['calving_thick'] = out['thick']
    odf['calving_water_depth'] = out['water_depth']
    odf['calving_free_board'] = out['free_board']
    odf['calving_front_width'] = out['width']
    for k, v in odf.items():
        gdir.add_to_diagnostics(k, v)

    # Restore defaults
    cfg.PARAMS['min_mu_star'] = 1.
    cfg.PARAMS['clip_mu_star'] = False

    return odf
