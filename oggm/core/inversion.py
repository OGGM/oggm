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
from scipy.interpolate import griddata
# Locals
from oggm import utils, cfg
from oggm import entity_task
from oggm.core.gis import gaussian_blur

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
        flux = flux.clip(0)

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
    """Solve for degree 5 polynom with coefs a5=1, a3, a0."""
    sols = np.roots([1., 0., a3, 0., 0., a0])
    test = (np.isreal(sols)*np.greater(sols, [0]*len(sols)))
    return sols[test][0].real


def _inversion_simple(a3, a0):
    """Solve for degree 5 polynom with coefs a5=1, a3=0., a0."""

    return (-a0)**(1./5.)


def _compute_thick(gdir, a0s, a3, flux_a0, shape_factor, _inv_function):
    """
    TODO: Documentation
    Content of the original inner loop of the mass-conservation inversion.
    Extracted to avoid code duplication
    Parameters
    ----------
    gdir
    a0s
    a3
    flux_a0
    shape_factor
    _inv_function

    Returns
    -------

    """

    a0s = a0s / (shape_factor ** 3)
    if np.any(~np.isfinite(a0s)):
        raise RuntimeError('({}) something went wrong with the '
                           'inversion'.format(gdir.rgi_id))

    # GO
    out_thick = np.zeros(len(a0s))
    for i, (a0, Q) in enumerate(zip(a0s, flux_a0)):
        if Q > 0.:
            out_thick[i] = _inv_function(a3, a0)
        else:
            out_thick[i] = 0.
    assert np.all(np.isfinite(out_thick))
    return out_thick


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
    if fs == 0.:
        _inv_function = _inversion_simple
    else:
        _inv_function = _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.PARAMS['glen_n']+2) * glen_a
    a3 = fs / fd
    rho = cfg.PARAMS['ice_density']

    # Shape factor params
    sf_func = None
    # Use .get to obatin default None for non-existing key
    # necessary to pass some tests
    # TODO: remove after tests are adapted
    use_sf = cfg.PARAMS.get('use_shape_factor_for_inversion')
    if use_sf == 'Adhikari' or use_sf == 'Nye':
        sf_func = utils.shape_factor_adhikari
    elif use_sf == 'Huss':
        sf_func = utils.shape_factor_huss
    sf_tol = 1e-2  # TODO: better as params in cfg?
    max_sf_iter = 20

    # Clip the slope, in degrees
    clip_angle = cfg.PARAMS['min_slope']

    out_volume = 0.

    cls = gdir.read_pickle('inversion_input')
    for cl in cls:
        # Clip slope to avoid negative and small slopes
        slope = cl['slope_angle']
        slope = np.clip(slope, np.deg2rad(clip_angle), np.pi/2.)

        # Parabolic bed rock
        w = cl['width']

        a0s = - cl['flux_a0'] / ((rho*cfg.G*slope)**3*fd)

        sf = np.ones(slope.shape)  # Default shape factor is 1
        # TODO: maybe take height update as criterion for iteration end instead
        # of sf_diff?
        if sf_func is not None:

            # Start iteration for shape factor with guess of 1
            i = 0
            sf_diff = np.ones(slope.shape)

            while i < max_sf_iter and np.any(sf_diff > sf_tol):
                out_thick = _compute_thick(gdir, a0s, a3, cl['flux_a0'],
                                           sf, _inv_function)

                sf_diff[:] = sf[:]
                sf = sf_func(w, out_thick, cl['is_rectangular'])
                sf_diff = sf_diff - sf
                i += 1
            # TODO: Iteration at the moment for all grid points,
            # even if some already converged. Change?

            log.info('Shape factor {:s} used, took {:d} iterations for '
                     'convergence.'.format(use_sf, i))

        out_thick = _compute_thick(gdir, a0s, a3, cl['flux_a0'],
                                   sf, _inv_function)

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
        from oggm.core.gis import interpolation_masks
        interpolation_masks(gdir)

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
    thick = thick.clip(0)
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
        from oggm.core.gis import interpolation_masks
        interpolation_masks(gdir)

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
    thick = thick.clip(0)

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
