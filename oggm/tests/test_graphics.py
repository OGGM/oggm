from __future__ import division

import unittest
import warnings
import oggm.utils

warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r'.*guessing baseline image.*')

import nose

import os
import geopandas as gpd

# Local imports
from oggm.tests import requires_mpltest, requires_internet, RUN_GRAPHIC_TESTS
from oggm.tests import init_hef
from oggm import graphics
from oggm.core.preprocessing import gis, centerlines
import oggm.cfg as cfg
from oggm.utils import get_demo_file
from oggm.core.models import flowline

# this should be no problem since caught in __init__
try:
    from matplotlib.testing.decorators import image_comparison
except ImportError:
    pass

# do we event want to run the tests?
if not RUN_GRAPHIC_TESTS:
    raise unittest.SkipTest('Skipping all graphic tests.')

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDIR_BASE = os.path.join(CURRENT_DIR, 'tmp')
SUFFIX = '_1.5+'


# ----------------------------------------------------------
# Lets go

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

    hef_file = get_demo_file('Hintereisferner.shp')
    entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
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
