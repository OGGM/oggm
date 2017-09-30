import shapely.geometry as shpg
import numpy as np

# Local imports
from oggm.core.models import flowline


def dummy_constant_bed(hmax=3000., hmin=1000., nx=200, map_dx=100.,
                       widths=3.):

    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h
    widths = surface_h * 0. + widths
    coords = np.arange(0, nx- 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
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
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
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
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
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
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
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
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
                                          bed_h, widths)]


def dummy_parabolic_bed(hmax=3000., hmin=1000., nx=200, map_dx=100.,
                        default_shape=5.e-03,
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
    return [flowline.ParabolicFlowline(line, dx, map_dx, surface_h,
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

    fls = flowline.MixedFlowline(line=line, dx=dx, map_dx=map_dx,
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

    return [flowline.TrapezoidalFlowline(line, dx, map_dx, surface_h,
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
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
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

    fl_0 = flowline.VerticalWallFlowline(line, dx, map_dx, surface_h, bed_h,
                                         widths)
    coords = np.arange(0, 19.1, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0. + 1]).T)
    fl_1 = flowline.VerticalWallFlowline(line, dx, map_dx, surface_h[0:20],
                                         bed_h[0:20], widths[0:20])
    fl_1.set_flows_to(fl_0)
    return [fl_1, fl_0]


def get_ident():
    ident_str="$Id$"
    if ":" not in ident_str:
        return "default"
    return ident_str.replace("$", "").replace("Id:", "").replace(" ", "")
