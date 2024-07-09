import os
import shutil
from distutils.util import strtobool
import hashlib

import numpy as np
import xarray as xr
import shapely.geometry as shpg
from scipy import optimize as optimization

# Local imports
import oggm
import oggm.cfg as cfg
from oggm.utils import (get_demo_file, mkdir, get_git_ident, get_sys_info,
                        get_env_info)
from oggm.workflow import execute_entity_task
from oggm.core import flowline, massbalance
from oggm import tasks
from oggm.core.flowline import RectangularBedFlowline

_TEST_DIR = None


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
        shape[mixslice] = np.nan
    else:
        shape[10:20] = np.nan
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


def dummy_mixed_trap_rect_bed(deflambdas=2., map_dx=100., mixslice=None):
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    shape = np.ones(nx) * np.nan
    is_trapezoid = ~np.isfinite(shape)
    lambdas = shape * 0.
    lambdas[is_trapezoid] = deflambdas
    # use a second value for lambda in between
    lambdas[15:20] = deflambdas / 2
    widths_m = bed_h * 0. + 10
    if mixslice:
        lambdas[mixslice] = 0
        widths_m[mixslice] = 25
    else:
        lambdas[0:10] = 0
        widths_m[0:10] = 25
    section = bed_h * 0.

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

    fls = flowline.MixedBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                    surface_h=surface_h, bed_h=bed_h,
                                    section=section, bed_shape=shape,
                                    is_trapezoid=is_trapezoid,
                                    lambdas=lambdas, widths_m=widths_m)

    return [fls]


def dummy_trapezoidal_bed(hmax=3000., hmin=1000., nx=200, map_dx=100.,
                          def_lambdas=2):

    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 1.6

    lambdas = surface_h * 0. + def_lambdas

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


def dummy_bed_tributary_tail_to_head(map_dx=100., n_trib=1, small_cliff=False):
    # bed with tributary glacier(s) flowing directly into their top
    # (for split flowline experiments)
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 3.

    pix_id = np.linspace(20, 180, n_trib).round().astype(int)

    fls = [flowline.RectangularBedFlowline(dx=dx, map_dx=map_dx,
                                           surface_h=surface_h[:pix_id[0]],
                                           bed_h=bed_h[:pix_id[0]],
                                           widths=widths[:pix_id[0]])]

    for i, pid in enumerate(pix_id):
        if i == (len(pix_id) - 1):
            eid = nx + 1
        else:
            eid = pix_id[i + 1]
        dh = -100 if small_cliff else 0

        fl = flowline.RectangularBedFlowline(dx=dx, map_dx=map_dx,
                                             surface_h=surface_h[pid:eid] + dh,
                                             bed_h=bed_h[pid:eid] + dh,
                                             widths=widths[pid:eid])
        fls[-1].set_flows_to(fl, to_head=True, check_tail=False)
        fls.append(fl)

    return fls


def bu_tidewater_bed(gridsize=200, gridlength=6e4, widths_m=600,
                     b_0=260, alpha=0.017, b_1=350, x_0=4e4, sigma=1e4,
                     water_level=0, split_flowline_before_water=None):

    # Bassis & Ultee bed profile
    dx_meter = gridlength / gridsize
    x = np.arange(gridsize+1) * dx_meter
    bed_h = b_0 - alpha * x + b_1 * np.exp(-((x - x_0) / sigma)**2)
    bed_h += water_level
    surface_h = bed_h
    widths = surface_h * 0. + widths_m / dx_meter

    if split_flowline_before_water is not None:
        bs = np.min(np.nonzero(bed_h < 0)[0]) - split_flowline_before_water
        fls = [RectangularBedFlowline(dx=1, map_dx=dx_meter,
                                      surface_h=surface_h[:bs],
                                      bed_h=bed_h[:bs],
                                      widths=widths[:bs]),
               RectangularBedFlowline(dx=1, map_dx=dx_meter,
                                      surface_h=surface_h[bs:],
                                      bed_h=bed_h[bs:],
                                      widths=widths[bs:]),
               ]
        fls[0].set_flows_to(fls[1], check_tail=False, to_head=True)
        return fls
    else:
        return [
            RectangularBedFlowline(dx=1, map_dx=dx_meter, surface_h=surface_h,
                                   bed_h=bed_h, widths=widths)]


def patch_minimal_download_oggm_files(*args, **kwargs):
    """A simple patch to make sure we don't download."""

    raise RuntimeError('We should not be there in minimal mode')


def use_multiprocessing():
    try:
        return strtobool(os.getenv("OGGM_TEST_MULTIPROC", "False"))
    except BaseException:
        return False


def get_test_dir():

    global _TEST_DIR

    if _TEST_DIR is None:
        s = get_git_ident()
        s += ''.join([str(k) + str(v) for k, v in get_sys_info()])
        s += ''.join([str(k) + str(v) for k, v in get_env_info()])
        s = hashlib.md5(s.encode()).hexdigest()
        out = os.path.join(cfg.PATHS['test_dir'], s)
        if 'PYTEST_XDIST_WORKER' in os.environ:
            out = os.path.join(out, os.environ.get('PYTEST_XDIST_WORKER'))
        mkdir(out)
        _TEST_DIR = out

        # If new ident, remove all other dirs so spare space
        for d in os.listdir(cfg.PATHS['test_dir']):
            if d and d != s:
                try:
                    shutil.rmtree(os.path.join(cfg.PATHS['test_dir'], d))
                except NotADirectoryError:
                    pass

    return _TEST_DIR


def init_hef(reset=False, border=40, logging_level='INFO', rgi_id=None,
             flowline_type='centerlines'):
    """

    Parameters
    ----------
    reset
    border
    logging_level
    rgi_id
    flowline_type : str
        Select which flowline type should be used.
        Options: 'centerlines' (default), 'elevation_bands'

    Returns
    -------

    """

    from oggm.core import gis, inversion, climate, centerlines, flowline
    import geopandas as gpd

    # test directory
    testdir = os.path.join(get_test_dir(), 'tmp_border{}'.format(border))
    if rgi_id is not None:
        testdir = os.path.join(get_test_dir(), f'tmp_border{border}_{rgi_id}')

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
    cfg.PARAMS['trapezoid_lambdas'] = 1
    cfg.PARAMS['border'] = border
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['evolution_model'] = 'FluxBased'
    cfg.PARAMS['downstream_line_shape'] = 'parabola'
    cfg.PARAMS['prcp_fac'] = 2.5
    cfg.PARAMS['temp_bias_min'] = -10
    cfg.PARAMS['temp_bias_max'] = 10
    hef_file = get_demo_file('Hintereisferner_RGI5.shp')
    entity = gpd.read_file(hef_file).iloc[0]

    if rgi_id is not None:
        entity['RGIId'] = rgi_id

    gdir = oggm.GlacierDirectory(entity, reset=reset)

    if 'inversion_glen_a' not in gdir.get_diagnostics():
        reset = True
        gdir = oggm.GlacierDirectory(entity, reset=reset)

    if not reset:
        return gdir

    gis.define_glacier_region(gdir)
    if flowline_type == 'centerlines':
        execute_entity_task(gis.glacier_masks, [gdir])
        execute_entity_task(centerlines.compute_centerlines, [gdir])
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
    elif flowline_type == 'elevation_bands':
        execute_entity_task(tasks.simple_glacier_masks, [gdir])
        execute_entity_task(tasks.elevation_band_flowline, [gdir])
        execute_entity_task(tasks.fixed_dx_elevation_band_flowline, [gdir])
    else:
        raise NotImplementedError(f'flowline_type {flowline_type} not'
                                  f'implemented!')
    centerlines.compute_downstream_line(gdir)
    centerlines.compute_downstream_bedshape(gdir)
    climate.process_custom_climate_data(gdir)
    if rgi_id is not None:
        gdir.rgi_id = 'RGI50-11.00897'
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        gdir.rgi_id = rgi_id
    else:
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

    ref_mb = mbdf.mean()
    ref_period = f'{mbdf.index[0]}-01-01_{mbdf.index[-1] + 1}-01-01'
    massbalance.mb_calibration_from_scalar_mb(gdir,
                                              ref_mb=ref_mb,
                                              ref_period=ref_period)
    massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))
    inversion.prepare_for_inversion(gdir)

    ref_v = 0.573 * 1e9

    glen_n = cfg.PARAMS['glen_n']

    def to_optimize(x):
        # For backwards compat
        _fd = 1.9e-24 * x[0]
        glen_a = (glen_n+2) * _fd / 2.
        fs = 5.7e-20 * x[1]
        v = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a)
        return (v - ref_v)**2

    out = optimization.minimize(to_optimize, [1, 1],
                                bounds=((0.01, 10), (0.01, 10)),
                                tol=1e-4)['x']
    _fd = 1.9e-24 * out[0]
    glen_a = (glen_n+2) * _fd / 2.
    fs = 5.7e-20 * out[1]
    v = inversion.mass_conservation_inversion(gdir, fs=fs,
                                              glen_a=glen_a,
                                              write=True)
    inversion.filter_inversion_output(gdir)

    if flowline_type == 'centerlines':
        inversion.distribute_thickness_interp(gdir, varname_suffix='_interp')
    inversion.distribute_thickness_per_altitude(gdir, varname_suffix='_alt')

    flowline.init_present_time_glacier(gdir)

    return gdir


def init_columbia(reset=False):

    from oggm.core import gis, centerlines
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
    cfg.PARAMS['use_kcalving_for_inversion'] = True
    cfg.PARAMS['use_kcalving_for_run'] = True
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['prcp_fac'] = 2.5
    cfg.PARAMS['baseline_climate'] = 'CRU'
    cfg.PARAMS['evolution_model'] = 'FluxBased'

    entity = gpd.read_file(get_demo_file('01_rgi60_Columbia.shp')).iloc[0]
    gdir = oggm.GlacierDirectory(entity, reset=reset)
    if gdir.has_file('climate_historical'):
        return gdir

    gis.define_glacier_region(gdir)
    gis.glacier_masks(gdir)
    centerlines.compute_centerlines(gdir)
    centerlines.initialize_flowlines(gdir)
    centerlines.compute_downstream_line(gdir)
    centerlines.catchment_area(gdir)
    centerlines.catchment_intersections(gdir)
    centerlines.catchment_width_geom(gdir)
    centerlines.catchment_width_correction(gdir)
    tasks.process_dummy_cru_file(gdir, seed=0)
    return gdir


def init_columbia_eb(dir_name, reset=False):

    from oggm.core import gis, centerlines
    import geopandas as gpd

    # test directory
    testdir = os.path.join(get_test_dir(), dir_name)
    mkdir(testdir, reset=reset)

    # Init
    cfg.initialize()
    cfg.PATHS['working_dir'] = testdir
    cfg.PARAMS['use_intersects'] = False
    cfg.PATHS['dem_file'] = get_demo_file('dem_Columbia.tif')
    cfg.PARAMS['border'] = 10
    cfg.PARAMS['use_kcalving_for_inversion'] = True
    cfg.PARAMS['use_kcalving_for_run'] = True
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['prcp_fac'] = 2.5
    cfg.PARAMS['baseline_climate'] = 'CRU'
    cfg.PARAMS['evolution_model'] = 'FluxBased'

    entity = gpd.read_file(get_demo_file('01_rgi60_Columbia.shp')).iloc[0]
    gdir = oggm.GlacierDirectory(entity)
    if gdir.has_file('climate_historical'):
        return gdir

    gis.define_glacier_region(gdir)
    gis.simple_glacier_masks(gdir)
    centerlines.elevation_band_flowline(gdir)
    centerlines.fixed_dx_elevation_band_flowline(gdir)
    centerlines.compute_downstream_line(gdir)
    tasks.process_dummy_cru_file(gdir, seed=0)
    tasks.mb_calibration_from_geodetic_mb(gdir)
    tasks.apparent_mb_from_any_mb(gdir)
    tasks.find_inversion_calving_from_any_mb(gdir)
    return gdir


def characs_apply_func(gdir, d):

    # add some new stats to the mix
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        glc_ext = ds['glacier_ext'].values
        glc_mask = ds['glacier_mask'].values
        d['glc_ext_num_perc'] = np.sum(glc_ext) / np.sum(glc_mask)


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
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]

    def __exit__(self, *args):
        for k, v in self.old_envs.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]
