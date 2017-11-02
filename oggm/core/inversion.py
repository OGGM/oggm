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
"""
# Built ins
import logging
import os
# External libs
import numpy as np
import pandas as pd
import netCDF4
from scipy import optimize as optimization
from scipy.ndimage.morphology import distance_transform_edt
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

    # for testing only
    if 'invert_with_rectangular' in cfg.PARAMS:
        invert_with_rectangular = cfg.PARAMS['invert_with_rectangular']
    if 'invert_all_rectangular' in cfg.PARAMS:
        invert_all_rectangular = cfg.PARAMS['invert_all_rectangular']

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

        # Optimisation: we need to compute this term of a0 only once
        flux_a0 = np.where(fl.is_rectangular, 1, 1.5)
        flux_a0 *= flux / widths

        # Shape
        is_rectangular = fl.is_rectangular
        if not invert_with_rectangular:
            is_rectangular[:] = False
        if invert_all_rectangular:
            is_rectangular[:] = True

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

    return (-a0)**cfg.ONE_FIFTH


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

        if np.any(~np.isfinite(a0s)):
            raise RuntimeError('({}) something went wrong with the '
                               'inversion'.format(gdir.rgi_id))

        # GO
        out_thick = np.zeros(len(slope))
        for i, (a0, Q) in enumerate(zip(a0s, cl['flux_a0'])):
            if Q > 0.:
                out_thick[i] = _inv_function(a3, a0)
            else:
                out_thick[i] = 0.
        assert np.all(np.isfinite(out_thick))

        # volume
        fac = np.where(cl['is_rectangular'], 1, cfg.TWO_THIRDS)
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
    try:
        gtd_df = pd.read_csv(fpath).sort_values(by=['RGI_ID'])
    except AttributeError:
        gtd_df = pd.read_csv(fpath).sort(columns=['RGI_ID'])

    dfids = gtd_df['RGI_ID'].values

    ref_gdirs = [gdir for gdir in gdirs if gdir.rgi_id in dfids]
    if len(ref_gdirs) == 0:
        raise RuntimeError('No reference GlaThiDa glaciers. Maybe something '
                           'went wrong with the link list?')
    ref_rgiids = [gdir.rgi_id for gdir in ref_gdirs]
    gtd_df = gtd_df.set_index('RGI_ID').loc[ref_rgiids]

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

    cls = gdir.read_pickle('inversion_output')
    for cl in cls:

        init_vol = np.sum(cl['volume'])
        if init_vol == 0 or gdir.is_tidewater or not cl['is_last']:
            continue

        w = cl['width']
        out_thick = cl['thick']
        fac = np.where(cl['is_rectangular'], 1, cfg.TWO_THIRDS)

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


def _distribute_thickness_per_altitude(glacier_mask, topo, cls, fls, grid,
                                       add_slope=True,
                                       smooth=True):
    """Where the job is actually done."""

    # Along the lines
    dx = grid.dx
    hs, ts, vs, xs, ys = [], [], [], [] ,[]
    for cl, fl in zip(cls, fls):
        # TODO: here one should see if parabola is always the best choice
        hs = np.append(hs, fl.surface_h)
        ts = np.append(ts, cl['thick'])
        vs = np.append(vs, cl['volume'])
        x, y = fl.line.xy
        xs = np.append(xs, x)
        ys = np.append(ys, y)
    vol = np.sum(vs)

    # very inefficient inverse distance stuff
    to_compute = np.nonzero(glacier_mask)
    thick = topo * np.NaN
    for (y, x) in np.asarray(to_compute).T:
        assert glacier_mask[y, x] == 1
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
            thick[y, x] = ts[pzero]
        else:
            raise RuntimeError('We should not be there')

    # Smooth
    if smooth:
        thick = np.where(np.isfinite(thick), thick, 0.)
        gsize = np.rint(cfg.PARAMS['smooth_window'] / dx)
        thick = gaussian_blur(thick, np.int(gsize))
    thick = np.where(glacier_mask, thick, 0.)

    # Distance
    dis = distance_transform_edt(glacier_mask)
    dis = np.where(glacier_mask, dis, np.NaN)**0.5

    # Slope
    slope = 1.
    if add_slope:
        sy, sx = np.gradient(topo, dx, dx)
        slope = np.arctan(np.sqrt(sy**2 + sx**2))
        slope = np.clip(slope, np.deg2rad(6.), np.pi/2.)
        slope = 1 / slope**(cfg.N / (cfg.N+2))

    # Conserve volume
    tmp_vol = np.nansum(thick * dis * slope * dx**2)
    final_t = thick * dis * slope * vol / tmp_vol

    # Done
    final_t = np.where(np.isfinite(final_t), final_t, 0.)
    assert np.allclose(np.sum(final_t * dx**2), vol)
    return final_t


def _distribute_thickness_per_interp(glacier_mask, topo, cls, fls, grid,
                                     smooth=True, add_slope=True):
    """Where the job is actually done."""

    # Thick to interpolate
    dx = grid.dx
    thick = np.where(glacier_mask, np.NaN, 0)

    # Along the lines
    vs = []
    for cl, fl in zip(cls, fls):
        # TODO: here one should see if parabola is always the best choice
        vs.extend(cl['volume'])
        x, y = utils.tuple2int(fl.line.xy)
        thick[y, x] = cl['thick']
    vol = np.sum(vs)

    # Interpolate
    xx, yy = grid.ij_coordinates
    pnan = np.nonzero(~ np.isfinite(thick))
    pok = np.nonzero(np.isfinite(thick))
    points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
    inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
    thick[pnan] = griddata(points, np.ravel(thick[pok]), inter, method='cubic')

    # Smooth
    if smooth:
        gsize = np.rint(cfg.PARAMS['smooth_window'] / dx)
        thick = gaussian_blur(thick, np.int(gsize))
        thick = np.where(glacier_mask, thick, 0.)

    # Slope
    slope = 1.
    if add_slope:
        sy, sx = np.gradient(topo, dx, dx)
        slope = np.arctan(np.sqrt(sy**2 + sx**2))
        slope = np.clip(slope, np.deg2rad(6.), np.pi/2.)
        slope = 1 / slope**(cfg.N / (cfg.N+2))

    # Conserve volume
    tmp_vol = np.nansum(thick * slope * dx**2)
    final_t = thick * slope * vol / tmp_vol

    # Add to grids
    final_t = np.where(np.isfinite(final_t), final_t, 0.)
    assert np.allclose(np.sum(final_t * dx**2), vol)
    return final_t


@entity_task(log, writes=['gridded_data'])
def distribute_thickness(gdir, how='', add_slope=True, smooth=True,
                         add_nc_name=False):
    """Compute a thickness map of the glacier using the nearest centerlines.

    This is a rather cosmetic task, not relevant for OGGM but for ITMIX.
    Here we take the nearest neighbors in a certain altitude range.

    Parameters
    ----------
    """

    if how == 'per_altitude':
        inv_g = _distribute_thickness_per_altitude
    elif how == 'per_interpolation':
        inv_g = _distribute_thickness_per_interp
    else:
        raise ValueError('interpolation method not understood')


    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    with netCDF4.Dataset(grids_file) as nc:
        glacier_mask = nc.variables['glacier_mask'][:]
        topo = nc.variables['topo_smoothed'][:]
    cls = gdir.read_pickle('inversion_output')
    fls = gdir.read_pickle('inversion_flowlines')
    thick = inv_g(glacier_mask, topo, cls, fls, gdir.grid,
                  add_slope=add_slope, smooth=smooth)

    # write
    grids_file = gdir.get_filepath('gridded_data')
    with netCDF4.Dataset(grids_file, 'a') as nc:
        vn = 'thickness'
        # TODO: this is for testing -- remove later
        if add_nc_name:
            vn += '_' + how
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = '-'
        v.long_name = 'Distributed ice thickness'
        v[:] = thick
