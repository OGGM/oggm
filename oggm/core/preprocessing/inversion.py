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
from __future__ import division
from six.moves import zip

# Built ins
import logging
import os
# External libs
import numpy as np
import pandas as pd
import scipy.optimize as optimization
from scipy.ndimage.filters import gaussian_filter1d
# Locals
from oggm import utils, cfg
from oggm import entity_task, divide_task

# Module logger
log = logging.getLogger(__name__)

@entity_task(log, writes=['inversion_input'])
@divide_task(log, add_0=False)
def prepare_for_inversion(gdir, div_id=None):
    """Prepares the data needed for the inversion.

    Mostly the mass flux and slope angle, the rest (width, height) was already
    computed. It is then stored in a list of dicts in order to be faster.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    # variables
    fls = gdir.read_pickle('inversion_flowlines', div_id=div_id)

    towrite = []
    slopes = np.array([])
    for fl in fls:

        # Distance between two points
        dx = fl.dx * gdir.grid.dx

        # Widths
        widths = fl.widths * gdir.grid.dx

        # Heigth and slope(s)
        hgt = fl.surface_h
        angle = np.arctan(-np.gradient(hgt, dx))  # beware the minus sign
        slopes = np.append(slopes, angle)

        # Flux needs to be in [m3 s-1] (*ice* velocity * surface)
        # fl.flux is given in kg m-2 yr-1, rho in kg m-3, so this should be it:
        flux = fl.flux * (gdir.grid.dx**2) / cfg.SEC_IN_YEAR / cfg.RHO

        # Clip flux to 0
        if np.any(flux < -0.1):
            log.warning('%s: has negative flux somewhere', gdir.rgi_id)

        # add to output
        cl_dic = dict(dx=dx, flux=flux, width=widths, hgt=hgt,
                      slope_angle=angle, is_last=fl.flows_to is None)
        towrite.append(cl_dic)

    # Write out
    gdir.write_pickle(towrite, 'inversion_input', div_id=div_id)


def _inversion_poly(a3, a0):
    """Solve for degree 5 polynom with coefs a5=1, a3, a0."""
    sols = np.roots([1., 0., a3, 0., 0., a0])
    test = (np.isreal(sols)*np.greater(sols,[0]*len(sols)))
    return sols[test][0].real


def _inversion_simple(a3, a0):
    """Solve for degree 5 polynom with coefs a5=1, a3=0., a0."""

    return (-a0)**cfg.ONE_FIFTH


def inversion_parabolic_point_slope(gdir,
                                    fs=5.7e-20,
                                    fd=1.9e-24,
                                    write=True):
    """ Compute the glacier thickness along the flowlines

    More or less following Farinotti et al., (2009).

    The thickness estimation is computed under the assumption of a parabolic
    bed section.

    It returns (vol, thick) in (m3, m).

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    fs : float
        sliding parameter
    fd : float
        deformation parameter
    write: bool
        default behavior is to compute the thickness and write the
        results in the pickle. Set to False in order to spare time
        during calibration.
    """

    # Check input
    if fs == 0.:
        _inv = _inversion_simple
    else:
        _inv = _inversion_poly

    a3 = fs / fd

    # sometimes the width is small and the flux is big. crop this too
    max_ratio = cfg.PARAMS['max_thick_to_width_ratio']
    max_shape = cfg.PARAMS['max_shape_param']
    # sigma of the smoothing window after inversion
    sec_smooth = cfg.PARAMS['section_smoothing']
    # Clip the slope, in degrees
    clip_angle = cfg.PARAMS['min_slope']

    out_volume = 0.
    for div in gdir.divide_ids:
        cls = gdir.read_pickle('inversion_input', div_id=div)
        for cl in cls:
            # Clip slope to avoid negative and small slopes
            slope = cl['slope_angle']
            slope = np.clip(slope, np.deg2rad(clip_angle), np.pi/2.)

            # Parabolic bed rock
            w = cl['width']
            a0s = -(3*cl['flux'])/(2*w*((cfg.RHO*cfg.G*slope)**3)*fd)
            assert np.all(np.isfinite(a0s))

            # GO
            out_thick = np.zeros(len(slope))
            for i, (a0, Q) in enumerate(zip(a0s, cl['flux'])):
                if Q > 0.:
                    out_thick[i] = _inv(a3, a0)
                else:
                    out_thick[i] = 0.

            # Check for thick to width ratio (should ne be too large)
            ratio = out_thick / w  # there's no 0 width so we're good
            pno = np.where(ratio > max_ratio)
            if len(pno[0]) > 0:
                ratio[pno] = np.NaN
                ratio = utils.interp_nans(ratio, default=max_ratio)
                out_thick = w * ratio

            # Very last heights can be rough sometimes: interpolate
            if cl['is_last']:
                out_thick[-4:-1] = np.NaN
                out_thick = utils.interp_nans(out_thick)

            # Check for the shape parameter (should not be too large)
            out_shape = (4*out_thick)/(w**2)
            pno = np.where(out_shape > max_shape)
            if len(pno[0]) > 0:
                out_shape[pno] = np.NaN
                out_shape = utils.interp_nans(out_shape, default=max_shape)
                out_thick = (out_shape * w**2) / 4

            # smooth section
            section = cfg.TWO_THIRDS * w * out_thick
            section = gaussian_filter1d(section, sec_smooth)
            out_thick = section / (w * cfg.TWO_THIRDS)

            # volume
            volume = cfg.TWO_THIRDS * out_thick * w * cl['dx']

            if write:
                cl['thick'] = out_thick
                cl['volume'] = volume
            out_volume += np.nansum(volume)
        if write:
            gdir.write_pickle(cls, 'inversion_output', div_id=div)

    return out_volume, gdir.rgi_area_km2 * 1e6


def optimize_inversion_params(gdirs):
    """Optimizes fs and fd based on GlaThiDa thicknesses.

    We use the glacier averaged thicknesses provided by GlaThiDa and correct
    them for differences in area with RGI, using a glacier specific volume-area
    scaling formula.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    log.info('Compute the reference fs and fd parameters.')

    # Get test glaciers (all glaciers with thickness data)
    dfids = cfg.PATHS['glathida_rgi_links']
    try:
        gtd_df = pd.read_csv(dfids).sort_values(by=['RGI_ID'])
    except AttributeError:
        gtd_df = pd.read_csv(dfids).sort(columns=['RGI_ID'])
    dfids = gtd_df['RGI_ID'].values

    ref_gdirs = [gdir for gdir in gdirs if gdir.rgi_id in dfids]

    # Account for area differences between glathida and rgi
    ref_area_km2 = gtd_df.RGI_AREA.values
    ref_cs = gtd_df.VOLUME.values / (gtd_df.GTD_AREA.values**1.375)
    ref_volume_km3 = ref_cs * ref_area_km2**1.375
    ref_thickness_m = ref_volume_km3 / ref_area_km2 * 1000.

    # Optimize
    def to_optimize(x):
        tmp_vols = np.zeros(len(ref_gdirs))
        fd = 1.9e-24 * x[0]
        fs = 5.7e-20 * x[1]
        for i, gdir in enumerate(ref_gdirs):
            v, _ = inversion_parabolic_point_slope(gdir, fs=fs, fd=fd,
                                                   write=False)
            tmp_vols[i] = v * 1e-9
        return utils.rmsd(tmp_vols, ref_volume_km3)

    out = optimization.minimize(to_optimize, [1., 1.],
                                bounds=((0.01, 1), (0.01, 1)),
                                tol=1.e-3)

    # Check results and save.
    fd = 1.9e-24 * out['x'][0]
    fs = 5.7e-20 * out['x'][1]

    oggm_volume_m3 = np.zeros(len(ref_gdirs))
    rgi_area_m2 = np.zeros(len(ref_gdirs))
    for i, gdir in enumerate(ref_gdirs):
        v, a = inversion_parabolic_point_slope(gdir, fs=fs, fd=fd,
                                               write=False)
        oggm_volume_m3[i] = v
        rgi_area_m2[i] = a
    assert np.allclose(rgi_area_m2 * 1e-6, ref_area_km2)

    # This is for each glacier
    d = dict()
    d['fs'] = fs
    d['fd'] = fd
    for gdir in gdirs:
        gdir.write_pickle(d, 'inversion_params')

    # This is for the working dir
    # Simple stats
    d = dict()
    d['fs'] = fs
    d['fd'] = fd
    d['factor_fs'] = out['x'][1]
    d['factor_fd'] = out['x'][0]
    d['vol_rmsd'] = utils.rmsd(oggm_volume_m3 * 1e-9, ref_volume_km3)
    d['thick_rmsd'] = utils.rmsd(oggm_volume_m3 * 1e-9 / ref_area_km2 / 1000.,
                                 ref_thickness_m)
    log.info('Optimized fs and fd with a factor {factor_fs} and {factor_fd} '
             'for a volume RMSD of {vol_rmsd:.3f} '.format(**d))

    df = pd.DataFrame(d, index=[0])
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_params.csv')
    df.to_csv(fpath)

    # All results
    d = dict()
    d['ref_area_km2'] = ref_area_km2
    d['ref_volume_km3'] = ref_volume_km3
    d['oggm_volume_km3'] = oggm_volume_m3 * 1e-9
    d['vas_volume_km3'] = 0.034*(d['ref_area_km2']**1.375)

    rgi_id = [gdir.rgi_id for gdir in ref_gdirs]
    df = pd.DataFrame(d, index=rgi_id)
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_results.csv')
    df.to_csv(fpath)

    return fs, fd


@entity_task(log, writes=['inversion_output'])
def volume_inversion(gdir, use_cfg_params=False):
    """Computes the inversion for all the glaciers.

    If fs and fd are not given, it will use the optimized params.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    use_cfg_params : bool, default=False
        if True, use the user provided fs and fd. Else, use the results of
        the optimisation.
    """

    if use_cfg_params:
        raise NotImplementedError('use_cfg_params=True')
    else:
        # use the optimized ones
        d = gdir.read_pickle('inversion_params')
        fs = d['fs']
        fd = d['fd']

    # go
    inversion_parabolic_point_slope(gdir, fs=fs, fd=fd, write=True)