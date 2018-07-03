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
import os
# External libs
import numpy as np
import pandas as pd
from scipy import optimize as optimization
from scipy.interpolate import griddata
# Locals
from oggm import utils, cfg
from oggm import entity_task, global_task
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
    gdir : oggm.GlacierDirectory
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
        flux = fl.flux * (gdir.grid.dx**2) / cfg.SEC_IN_YEAR / cfg.RHO

        # Clip flux to 0
        if np.any(flux < -0.1):
            log.warning('(%s) has negative flux somewhere', gdir.rgi_id)
        flux = flux.clip(0)

        if fl.flows_to is None and gdir.inversion_calving_rate == 0:
            if not np.allclose(flux[-1], 0., atol=0.1):
                msg = '({}) flux at terminus should be zero, but is: ' \
                      '%.4f km3 ice yr-1'.format(gdir.rgi_id, flux[-1])
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


def mass_conservation_inversion(gdir, glen_a=cfg.A, fs=0., write=True,
                                filesuffix=''):
    """ Compute the glacier thickness along the flowlines

    More or less following Farinotti et al., (2009).
    
    Parameters
    ----------
    gdir : oggm.GlacierDirectory
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

    Returns
    -------
    (vol, thick) in [m3, m]
    """

    # Check input
    if fs == 0.:
        _inv_function = _inversion_simple
    else:
        _inv_function = _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.N+2) * glen_a
    a3 = fs / fd

    # Shape factor params
    sf_func = None
    use_sf = None
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

        a0s = - cl['flux_a0'] / ((cfg.RHO*cfg.G*slope)**3*fd)

        sf = np.ones(slope.shape)  # Default shape factor is 1
        # TODO: maybe take height update as criterion for iteration end instead
        # of sf_diff?
        if sf_func is not None:

            # Start iteration for shape factor with guess of 1
            i = 0
            sf_diff = np.ones(slope.shape)

            while i < max_sf_iter and \
                    np.any(sf_diff > sf_tol):
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


@global_task
def optimize_inversion_params(gdirs):
    """Optimizes fs and fd based on GlaThiDa thicknesses.

    We use the glacier averaged thicknesses provided by GlaThiDa and correct
    them for differences in area with RGI, using a glacier specific volume-area
    scaling formula.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    # Do we even need to do this?
    if not cfg.PARAMS['optimize_inversion_params']:
        raise RuntimeError('User did not want to optimize the '
                           'inversion params')

    # Get test glaciers (all glaciers with thickness data)
    fpath = utils.get_glathida_file()
    col_name = 'RGI{}_ID'.format(gdirs[0].rgi_version)
    try:
        gtd_df = pd.read_csv(fpath).sort_values(by=[col_name])
    except AttributeError:
        gtd_df = pd.read_csv(fpath).sort(columns=[col_name])

    dfids = gtd_df[col_name].values

    ref_gdirs = [gdir for gdir in gdirs if gdir.rgi_id in dfids]
    if len(ref_gdirs) == 0:
        raise RuntimeError('No reference GlaThiDa glaciers. Maybe something '
                           'went wrong with the link list?')
    ref_rgiids = [gdir.rgi_id for gdir in ref_gdirs]
    gtd_df = gtd_df.set_index(col_name).loc[ref_rgiids]

    # Account for area differences between glathida and rgi
    gtd_df['RGI_AREA'] = [gdir.rgi_area_km2 for gdir in ref_gdirs]
    ref_area_km2 = gtd_df.RGI_AREA.values
    ref_area_m2 = ref_area_km2 * 1e6
    gtd_df['VOLUME'] = gtd_df.MEAN_THICKNESS * gtd_df.GTD_AREA * 1e-3
    ref_cs = gtd_df.VOLUME.values / (gtd_df.GTD_AREA.values**1.375)
    ref_volume_km3 = ref_cs * ref_area_km2**1.375
    ref_thickness_m = ref_volume_km3 / ref_area_km2 * 1000.

    # Minimize volume or thick RMSD?
    optim_t = cfg.PARAMS['optimize_thick']
    if optim_t:
        ref_data = ref_thickness_m
        tol = 0.1
    else:
        ref_data = ref_volume_km3
        tol = 1.e-4

    if cfg.PARAMS['invert_with_sliding']:
        # Optimize with both params
        log.info('Compute the inversion parameters.')

        def to_optimize(x):
            tmp_ref = np.zeros(len(ref_gdirs))
            glen_a = cfg.A * x[0]
            fs = cfg.FS * x[1]
            for i, gdir in enumerate(ref_gdirs):
                v, a = mass_conservation_inversion(gdir, glen_a=glen_a,
                                                   fs=fs, write=False)
                if optim_t:
                    tmp_ref[i] = v / a
                else:
                    tmp_ref[i] = v * 1e-9
            return utils.rmsd(tmp_ref, ref_data)

        opti = optimization.minimize(to_optimize, [1., 1.],
                                     bounds=((0.01, 10), (0.01, 10)),
                                     tol=tol)
        # Check results and save.
        glen_a = cfg.A * opti['x'][0]
        fs = cfg.FS * opti['x'][1]
    else:
        # Optimize without sliding
        log.info('Compute the inversion parameter.')

        def to_optimize(x):
            tmp_ref = np.zeros(len(ref_gdirs))
            glen_a = cfg.A * x[0]
            for i, gdir in enumerate(ref_gdirs):
                v, a = mass_conservation_inversion(gdir, glen_a=glen_a,
                                                   fs=0., write=False)
                if optim_t:
                    tmp_ref[i] = v / a
                else:
                    tmp_ref[i] = v * 1e-9
            return utils.rmsd(tmp_ref, ref_data)
        opti = optimization.minimize(to_optimize, [1.],
                                     bounds=((0.01, 10),),
                                     tol=tol)
        # Check results and save.
        glen_a = cfg.A * opti['x'][0]
        fs = 0.

    # This is for the stats
    oggm_volume_m3 = np.zeros(len(ref_gdirs))
    rgi_area_m2 = np.zeros(len(ref_gdirs))
    for i, gdir in enumerate(ref_gdirs):
        v, a = mass_conservation_inversion(gdir, glen_a=glen_a, fs=fs,
                                           write=False)
        oggm_volume_m3[i] = v
        rgi_area_m2[i] = a
    assert np.allclose(rgi_area_m2 * 1e-6, ref_area_km2)

    # This is for each glacier
    out = dict()
    out['glen_a'] = glen_a
    out['fs'] = fs
    out['factor_glen_a'] = opti['x'][0]
    try:
        out['factor_fs'] = opti['x'][1]
    except IndexError:
        out['factor_fs'] = 0.
    for gdir in gdirs:
        gdir.write_pickle(out, 'inversion_params')

    # This is for the working dir
    # Simple stats
    out['vol_rmsd'] = utils.rmsd(oggm_volume_m3 * 1e-9, ref_volume_km3)
    out['thick_rmsd'] = utils.rmsd(oggm_volume_m3 / ref_area_m2,
                                   ref_thickness_m)

    log.info('Optimized glen_a and fs with a factor {factor_glen_a:.2f} and '
             '{factor_fs:.2f} for a thick RMSD of '
             '{thick_rmsd:.1f} m and a volume RMSD of '
             '{vol_rmsd:.3f} km3'.format(**out))

    df = pd.DataFrame(out, index=[0])
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_params.csv')
    df.to_csv(fpath)

    # All results
    df = dict()
    df['ref_area_km2'] = ref_area_km2
    df['ref_volume_km3'] = ref_volume_km3
    df['oggm_volume_km3'] = oggm_volume_m3 * 1e-9
    df['vas_volume_km3'] = 0.034*(df['ref_area_km2']**1.375)

    rgi_id = [gdir.rgi_id for gdir in ref_gdirs]
    df = pd.DataFrame(df, index=rgi_id)
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_results.csv')
    df.to_csv(fpath)

    # return value for tests
    return out


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

    if fs is not None and glen_a is None:
        raise ValueError('Cannot set fs without glen_a.')

    if glen_a is None and cfg.PARAMS['optimize_inversion_params']:
        # use the optimized ones
        d = gdir.read_pickle('inversion_params')
        fs = d['fs']
        glen_a = d['glen_a']

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
        assert new_vol != 0
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
    gdir : oggm.GlacierDirectory
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


@entity_task(log, writes=['gridded_data'])
def distribute_thickness_interp(gdir, add_slope=True, smooth_radius=None,
                                varname_suffix=''):
    """Compute a thickness map by interpolating between centerlines and border.

    This is a rather cosmetic task, not relevant for OGGM but for ITMIX.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
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
