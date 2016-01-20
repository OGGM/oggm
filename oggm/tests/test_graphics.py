from __future__ import division

import warnings

import oggm.utils

warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r'.*guessing baseline image.*')

import unittest
import nose

import os
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError
import pandas as pd
import geopandas as gpd

# Local imports
from oggm.core.preprocessing import gis, geometry, climate, inversion
from oggm.core.preprocessing import centerlines
import oggm.cfg as cfg
from oggm.utils import get_demo_file
from oggm import graphics
from oggm.core.models import flowline
from oggm.tests import HAS_NEW_GDAL, requires_fabiens_laptop, is_slow

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDIR_BASE = os.path.join(CURRENT_DIR, 'tmp')

# Because mpl was broken on conda
# https://github.com/matplotlib/matplotlib/issues/5487
try:
    from matplotlib.testing.decorators import image_comparison
    HAS_MPL_TEST = True
except ImportError:
    HAS_MPL_TEST = False


def requires_mpltest(test):
    # Decorator
    msg = 'requires matplotlib.testing.decorators'
    return test if HAS_MPL_TEST else unittest.skip(msg)(test)


import matplotlib as mpl
suffix = '_' + mpl.__version__
if mpl.__version__ >= '1.5':
    suffix = '_1.5+'

if HAS_NEW_GDAL:
    suffix += '_conda'

def internet_on():
    # Not so recommended it seems
    try:
        _ = urlopen('http://www.google.com', timeout=1)
        return True
    except URLError:
        pass
    return False

HAS_INTERNET = internet_on()

def requires_internet(test):
    # Decorator
    msg = 'requires internet'
    return test if HAS_INTERNET else unittest.skip(msg)(test)


def requires_mpl15(test):
    # Decorator
    msg = 'requires mpl V 1.5+'
    return test if mpl.__version__ >= '1.5' else unittest.skip(msg)(test)

# ----------------------------------------------------------
# Lets go


def init_hef(reset=False, border=40):

    # test directory
    testdir = TESTDIR_BASE + '_border{}'.format(border)
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
    cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
    cfg.PATHS['srtm_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['histalp_file'] = get_demo_file('histalp_merged_hef.nc')
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

    def to_optimize(x):
        fd = 1.9e-24 * x[0]
        fs = 5.7e-20 * x[1]
        v, _ = inversion.inversion_parabolic_point_slope(gdir,
                                                         fs=fs,
                                                         fd=fd)
        return (v - ref_v)**2

    import scipy.optimize as optimization
    out = optimization.minimize(to_optimize, [1,1],
                                bounds=((0.01, 1), (0.01, 1)),
                                tol=1e-3)['x']
    fd = 1.9e-24 * out[0]
    fs = 5.7e-20 * out[1]
    v, _ = inversion.inversion_parabolic_point_slope(gdir,
                                                     fs=fs,
                                                     fd=fd,
                                                     write=True)
    d = dict(fs=fs, fd=fd)
    gdir.write_pickle(d, 'inversion_params')

    return gdir



@image_comparison(baseline_images=['test_googlestatic' + suffix],
                  extensions=['png'])
@requires_internet
@requires_mpltest
def test_googlemap():

    gdir = init_hef()
    graphics.plot_googlemap(gdir)


@image_comparison(baseline_images=['test_centerlines' + suffix,
                                   'test_flowlines' + suffix,
                                   'test_downstream' + suffix,
                                   'test_downstream_cls' + suffix,
                                   ],
                  extensions=['png'])
@requires_mpltest
def test_centerlines():

    gdir = init_hef()
    graphics.plot_centerlines(gdir)
    graphics.plot_centerlines(gdir, use_flowlines=True)
    graphics.plot_centerlines(gdir, add_downstream=True, use_flowlines=True)
    graphics.plot_centerlines(gdir, add_downstream=True)


@image_comparison(baseline_images=['test_width' + suffix,
                                   'test_width_corrected' + suffix
                                   ],
                  extensions=['png'])
@requires_mpltest
def test_width():

    gdir = init_hef()
    graphics.plot_catchment_width(gdir)
    graphics.plot_catchment_width(gdir, corrected=True)


@image_comparison(baseline_images=['test_inversion' + suffix],
                  extensions=['png'])
@requires_mpltest
def test_inversion():

    gdir = init_hef()
    graphics.plot_inversion(gdir)


@image_comparison(baseline_images=['test_initial_glacier' + suffix],
                  extensions=['png'])
@is_slow
@requires_mpltest
@requires_fabiens_laptop
def test_initial_glacier():  # pragma: no cover

    gdir = init_hef(border=80)
    flowline.init_present_time_glacier(gdir)
    flowline.find_inital_glacier(gdir, y0=1847, init_bias=150)
    past_model = gdir.read_pickle('past_model')
    graphics.plot_modeloutput_map(gdir, past_model)


@image_comparison(baseline_images=['test_nodivide' + suffix],
                  extensions=['png'])
@requires_mpltest
@requires_mpl15
def test_nodivide():

    # test directory
    testdir = TESTDIR_BASE + '_nodiv'
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    # Init
    cfg.initialize()
    cfg.set_divides_db()
    cfg.PATHS['srtm_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['histalp_file'] = get_demo_file('histalp_merged_hef.nc')
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


@image_comparison(baseline_images=['test_modelsection' + suffix],
                  extensions=['png'])
@requires_mpltest
@requires_mpl15
def test_plot_model():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    model = flowline.FlowlineModel(gdir.read_pickle('model_flowlines'))
    graphics.plot_modeloutput_section(model)


if __name__ == '__main__':  # pragma: no cover
    nose.runmodule(argv=['-s', '-v', '--with-doctest'], exit=False)
