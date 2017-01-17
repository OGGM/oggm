from __future__ import division

import unittest
import warnings

warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r'.*guessing baseline image.*')

import pytest

import os
import geopandas as gpd
import matplotlib.pyplot as plt

# Local imports
import oggm.utils
from oggm.tests import requires_mpltest, requires_internet, RUN_GRAPHIC_TESTS
from oggm.tests import init_hef, BASELINE_DIR
from oggm import graphics
from oggm.core.preprocessing import gis, centerlines
import oggm.cfg as cfg
from oggm.utils import get_demo_file
from oggm.core.models import flowline


# do we event want to run the tests?
if not RUN_GRAPHIC_TESTS:
    raise unittest.SkipTest('Skipping all graphic tests.')

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDIR_BASE = os.path.join(CURRENT_DIR, 'tmp')

# ----------------------------------------------------------
# Lets go


@requires_internet
@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tol=15)
def test_googlemap():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_googlemap(gdir, ax=ax)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_centerlines():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_flowlines():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax, use_flowlines=True)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_downstream():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax, add_downstream=True,
                              use_flowlines=True)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_downstream_cls():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax, add_downstream=True)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_width():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_width(gdir, ax=ax)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_width_corrected():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_width(gdir, ax=ax, corrected=True)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_inversion():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_inversion(gdir, ax=ax)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
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

    fig, ax = plt.subplots()
    graphics.plot_centerlines(gdir, ax=ax)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_modelsection():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    fls = gdir.read_pickle('model_flowlines')
    fls = flowline.convert_to_mixed_flowline(fls)
    model = flowline.FlowlineModel(fls)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
    graphics.plot_modeloutput_section(gdir, ax=ax, model=model)
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_modelmap():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    fls = gdir.read_pickle('model_flowlines')
    fls = flowline.convert_to_mixed_flowline(fls)
    model = flowline.FlowlineModel(fls)

    fig, ax = plt.subplots()
    graphics.plot_modeloutput_map(gdir, ax=ax, model=model)
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_thick_alt():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_distributed_thickness(gdir, ax=ax, how='per_altitude')
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_thick_interp():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_distributed_thickness(gdir, ax=ax, how='per_interpolation')
    fig.tight_layout()
    return fig


@requires_mpltest
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_catch_areas():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_areas(gdir, ax=ax)
    fig.tight_layout()
    return fig
