from __future__ import division

import warnings

import oggm.utils

warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r'.*guessing baseline image.*')

import nose

import os
import pandas as pd
import geopandas as gpd

# Local imports
from oggm.core.preprocessing import gis, geometry, climate, inversion
from oggm.core.preprocessing import centerlines
import oggm.cfg as cfg
from oggm.utils import get_demo_file
from oggm import graphics
from oggm.core.models import flowline
from oggm.tests import requires_mpltest, requires_internet

# this should be no problem since caught in __init__
try:
    from matplotlib.testing.decorators import image_comparison
except ImportError:
    pass

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDIR_BASE = os.path.join(CURRENT_DIR, 'tmp')
SUFFIX = '_1.5+'


# ----------------------------------------------------------
# Lets go


def init_hef(reset=False, border=40, invert_with_sliding=True):

    # test directory
    testdir = TESTDIR_BASE + '_border{}'.format(border)
    if not invert_with_sliding:
        testdir += '_withoutslide'
    if not os.path.exists(testdir):
        os.makedirs(testdir)
        reset = True
    if not os.path.exists(os.path.join(testdir, 'RGI40-11.00897')):
        reset = True
    if not os.path.exists(os.path.join(testdir, 'RGI40-11.00897',
                                       'inversion_params.pkl')):
        reset = True

    # Init
    cfg.initialize()
    cfg.set_divides_db(get_demo_file('divides_workflow.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PARAMS['border'] = border

    # loop because for some reason indexing wont work
    hef_file = get_demo_file('Hintereisferner.shp')
    rgidf = gpd.GeoDataFrame.from_file(hef_file)
    for index, entity in rgidf.iterrows():
        gdir = oggm.GlacierDirectory(entity, base_dir=testdir, reset=reset)

    if not reset:
        return gdir

    gis.define_glacier_region(gdir, entity=entity)
    gis.glacier_masks(gdir)
    centerlines.compute_centerlines(gdir)
    centerlines.compute_downstream_lines(gdir)
    geometry.initialize_flowlines(gdir)
    geometry.catchment_area(gdir)
    geometry.catchment_width_geom(gdir)
    geometry.catchment_width_correction(gdir)
    climate.distribute_climate_data([gdir])
    climate.mu_candidates(gdir, div_id=0)
    hef_file = get_demo_file('mbdata_RGI40-11.00897.csv')
    mbdf = pd.read_csv(hef_file).set_index('YEAR')
    t_star, bias = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
    climate.local_mustar_apparent_mb(gdir, tstar=t_star[-1], bias=bias[-1])

    inversion.prepare_for_inversion(gdir)
    ref_v = 0.573 * 1e9

    if invert_with_sliding:
        def to_optimize(x):
            # For backwards compat
            _fd = 1.9e-24 * x[0]
            glen_a = (cfg.N+2) * _fd / 2.
            fs = 5.7e-20 * x[1]
            v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
                                                  glen_a=glen_a)
            return (v - ref_v)**2

        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-4)['x']
        _fd = 1.9e-24 * out[0]
        glen_a = (cfg.N+2) * _fd / 2.
        fs = 5.7e-20 * out[1]
        v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
                                              glen_a=glen_a,
                                              write=True)
    else:
        def to_optimize(x):
            glen_a = cfg.A * x[0]
            v, _ = inversion.invert_parabolic_bed(gdir, fs=0.,
                                                  glen_a=glen_a)
            return (v - ref_v)**2

        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1],
                                    bounds=((0.01, 10),),
                                    tol=1e-4)['x']
        glen_a = cfg.A * out[0]
        fs = 0.
        v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
                                              glen_a=glen_a,
                                              write=True)
    d = dict(fs=fs, glen_a=glen_a)
    d['factor_glen_a'] = out[0]
    try:
        d['factor_fs'] = out[1]
    except IndexError:
        d['factor_fs'] = 0.
    gdir.write_pickle(d, 'inversion_params')

    inversion.distribute_thickness(gdir, how='per_altitude',
                                   add_nc_name=True)
    inversion.distribute_thickness(gdir, how='per_interpolation',
                                   add_slope=False, smooth=False,
                                   add_nc_name=True)

    return gdir


@image_comparison(baseline_images=['test_googlestatic' + SUFFIX],
                  extensions=['png'], tol=15)
@requires_internet
@requires_mpltest
def test_googlemap():

    gdir = init_hef()
    graphics.plot_googlemap(gdir)


@image_comparison(baseline_images=['test_centerlines' + SUFFIX,
                                   'test_flowlines' + SUFFIX,
                                   'test_downstream' + SUFFIX,
                                   'test_downstream_cls' + SUFFIX,
                                   ],
                  extensions=['png'])
@requires_mpltest
def test_centerlines():

    gdir = init_hef()
    graphics.plot_centerlines(gdir)
    graphics.plot_centerlines(gdir, use_flowlines=True)
    graphics.plot_centerlines(gdir, add_downstream=True, use_flowlines=True)
    graphics.plot_centerlines(gdir, add_downstream=True)


@image_comparison(baseline_images=['test_width' + SUFFIX,
                                   'test_width_corrected' + SUFFIX
                                   ],
                  extensions=['png'])
@requires_mpltest
def test_width():

    gdir = init_hef()
    graphics.plot_catchment_width(gdir)
    graphics.plot_catchment_width(gdir, corrected=True)


@image_comparison(baseline_images=['test_inversion' + SUFFIX],
                  extensions=['png'])
@requires_mpltest
def test_inversion():

    gdir = init_hef()
    graphics.plot_inversion(gdir)


@image_comparison(baseline_images=['test_nodivide' + SUFFIX],
                  extensions=['png'])
@requires_mpltest
def test_nodivide():

    # test directory
    testdir = TESTDIR_BASE + '_nodiv'
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    # Init
    cfg.initialize()
    cfg.set_divides_db()
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PARAMS['border'] = 40

    # loop because for some reason indexing wont work
    hef_file = get_demo_file('Hintereisferner.shp')
    rgidf = gpd.GeoDataFrame.from_file(hef_file)
    for index, entity in rgidf.iterrows():
        gdir = oggm.GlacierDirectory(entity, base_dir=testdir, reset=True)

    gis.define_glacier_region(gdir, entity=entity)
    gis.glacier_masks(gdir)
    centerlines.compute_centerlines(gdir)

    graphics.plot_centerlines(gdir)


@image_comparison(baseline_images=['test_modelsection' + SUFFIX,
                                   'test_modelmap' + SUFFIX,
                                   ],
                  extensions=['png'])
@requires_mpltest
def test_plot_model():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    fls = flowline.convert_to_mixed_flowline(gdir.read_pickle('model_flowlines'))
    model = flowline.FlowlineModel(fls)
    graphics.plot_modeloutput_section(gdir, model=model)
    graphics.plot_modeloutput_map(gdir, model=model)


@image_comparison(baseline_images=['test_thick_alt' + SUFFIX,
                                   'test_thick_interp' + SUFFIX,
                                   ],
                  extensions=['png'])
@requires_mpltest
def test_plot_distrib():

    gdir = init_hef()
    graphics.plot_distributed_thickness(gdir, how='per_altitude')
    graphics.plot_distributed_thickness(gdir, how='per_interpolation')

if __name__ == '__main__':  # pragma: no cover
    nose.runmodule(argv=['-s', '-v', '--with-doctest'], exit=False)
