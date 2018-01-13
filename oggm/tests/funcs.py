import os
import shutil
from distutils.util import strtobool

import geopandas as gpd
import numpy as np
import shapely.geometry as shpg
from scipy import optimize as optimization

# Local imports
import oggm
import oggm.cfg as cfg
from oggm.core import gis, inversion, climate, centerlines, flowline
from oggm.utils import get_demo_file, mkdir
from oggm.workflow import execute_entity_task
from oggm.utils import _urlretrieve


def dummy_constant_bed(hmax=3000., hmin=1000., nx=200, map_dx=100.,
                       widths=3.):

    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h
    widths = surface_h * 0. + widths
    coords = np.arange(0, nx- 0.5, 1)
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


def dummy_width_bed_tributary(map_dx=100.):
    # bed with tributary glacier
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 3.
    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

    fl_0 = flowline.RectangularBedFlowline(line, dx, map_dx, surface_h, bed_h,
                                           widths)
    coords = np.arange(0, 19.1, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0. + 1]).T)
    fl_1 = flowline.RectangularBedFlowline(line, dx, map_dx, surface_h[0:20],
                                         bed_h[0:20], widths[0:20])
    fl_1.set_flows_to(fl_0)
    return [fl_1, fl_0]


def patch_url_retrieve(url, *args, **kwargs):
    """A simple patch to OGGM's download function to make sure we don't
    download elsewhere than expected."""

    assert 'github' in url
    return _urlretrieve(url, *args, **kwargs)


def use_multiprocessing():
    try:
        return strtobool(os.getenv("OGGM_TEST_MULTIPROC", "True"))
    except:
        return True


def get_ident():
    ident_str = '$Id$'
    if ":" not in ident_str:
        return 'no_git_id'
    return ident_str.replace("$", "").replace("Id:", "").replace(" ", "")


def get_test_dir():

    s = get_ident()
    out = os.path.join(cfg.PATHS['test_dir'], s)
    mkdir(out)

    # If new ident, remove all other dirs so spare space
    for d in os.listdir(cfg.PATHS['test_dir']):
        if d and d != s:
            shutil.rmtree(os.path.join(cfg.PATHS['test_dir'], d))
    return out


def init_hef(reset=False, border=40, invert_with_sliding=True,
             invert_with_rectangular=True):

    # test directory
    testdir = os.path.join(get_test_dir(), 'tmp_border{}'.format(border))
    if not invert_with_sliding:
        testdir += '_withoutslide'
    if not invert_with_rectangular:
        testdir += '_withoutrectangular'
    if not os.path.exists(testdir):
        os.makedirs(testdir)
        reset = True

    # Init
    cfg.initialize()
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PARAMS['border'] = border
    cfg.PARAMS['use_optimized_inversion_params'] = True

    hef_file = get_demo_file('Hintereisferner_RGI5.shp')
    entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

    gdir = oggm.GlacierDirectory(entity, base_dir=testdir, reset=reset)
    if not gdir.has_file('inversion_params'):
        reset = True
        gdir = oggm.GlacierDirectory(entity, base_dir=testdir, reset=reset)

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
    climate.process_histalp_nonparallel([gdir])
    climate.mu_candidates(gdir)
    mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
    res = climate.t_star_from_refmb(gdir, mbdf)
    climate.local_mustar(gdir, tstar=res['t_star'][-1], bias=res['bias'][-1],
                         prcp_fac=res['prcp_fac'])
    climate.apparent_mb(gdir)

    inversion.prepare_for_inversion(gdir, add_debug_var=True,
                                    invert_with_rectangular=invert_with_rectangular)
    ref_v = 0.573 * 1e9

    if invert_with_sliding:
        def to_optimize(x):
            # For backwards compat
            _fd = 1.9e-24 * x[0]
            glen_a = (cfg.N+2) * _fd / 2.
            fs = 5.7e-20 * x[1]
            v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                         glen_a=glen_a)
            return (v - ref_v)**2

        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-4)['x']
        _fd = 1.9e-24 * out[0]
        glen_a = (cfg.N+2) * _fd / 2.
        fs = 5.7e-20 * out[1]
        v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a,
                                                     write=True)
    else:
        def to_optimize(x):
            glen_a = cfg.A * x[0]
            v, _ = inversion.mass_conservation_inversion(gdir, fs=0.,
                                                         glen_a=glen_a)
            return (v - ref_v)**2

        out = optimization.minimize(to_optimize, [1],
                                    bounds=((0.01, 10),),
                                    tol=1e-4)['x']
        glen_a = cfg.A * out[0]
        fs = 0.
        v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a,
                                                     write=True)
    d = dict(fs=fs, glen_a=glen_a)
    d['factor_glen_a'] = out[0]
    try:
        d['factor_fs'] = out[1]
    except IndexError:
        d['factor_fs'] = 0.
    gdir.write_pickle(d, 'inversion_params')

    # filter
    inversion.filter_inversion_output(gdir)

    inversion.distribute_thickness(gdir, how='per_altitude',
                                   add_nc_name=True)
    inversion.distribute_thickness(gdir, how='per_interpolation',
                                   add_slope=False, smooth=False,
                                   add_nc_name=True)

    return gdir
