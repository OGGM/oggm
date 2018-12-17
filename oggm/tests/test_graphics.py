import warnings

import pytest
import shutil

import os
import geopandas as gpd
import matplotlib.pyplot as plt

# Local imports
import oggm.utils
from oggm.tests import mpl_image_compare
from oggm.tests.funcs import init_hef, get_test_dir, patch_url_retrieve_github
from oggm import graphics
from oggm.core import (gis, inversion, climate, centerlines, flowline,
                       massbalance)
import oggm.cfg as cfg
from oggm.utils import get_demo_file
from oggm import utils, workflow

# Warnings
warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r'.*guessing baseline image.*')

# Globals
pytestmark = pytest.mark.test_env("graphics")
_url_retrieve = None


def setup_module(module):
    module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github


def teardown_module(module):
    oggm.utils._downloads.oggm_urlretrieve = module._url_retrieve

# ----------------------------------------------------------
# Lets go


@pytest.mark.internet
@pytest.mark.graphic
@mpl_image_compare()
def test_googlemap():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_googlemap(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.internet
@pytest.mark.graphic
@mpl_image_compare()
def test_domain():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_domain(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_centerlines():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_flowlines():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax, use_flowlines=True)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_downstream():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax, add_downstream=True,
                              use_flowlines=True)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_width():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_width(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_width_corrected():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_width(gdir, ax=ax, corrected=True,
                                  add_intersects=True,
                                  add_touches=True)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_inversion():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_inversion(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_multiple_inversion():

    # test directory
    testdir = os.path.join(get_test_dir(), 'tmp_mdir')
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    # Init
    cfg.initialize()
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PARAMS['border'] = 40
    cfg.PARAMS['run_mb_calibration'] = True
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'
    cfg.PATHS['working_dir'] = testdir

    # Get the RGI ID
    hef_rgi = gpd.read_file(get_demo_file('divides_hef.shp'))
    hef_rgi.loc[0, 'RGIId'] = 'RGI50-11.00897'

    gdirs = workflow.init_glacier_regions(hef_rgi)
    workflow.gis_prepro_tasks(gdirs)
    workflow.climate_tasks(gdirs)
    workflow.inversion_tasks(gdirs)

    fig, ax = plt.subplots()
    graphics.plot_inversion(gdirs, ax=ax)
    fig.tight_layout()
    shutil.rmtree(testdir)
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_modelsection():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FlowlineModel(fls)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
    graphics.plot_modeloutput_section(ax=ax, model=model)
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_modelsection_withtrib():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FlowlineModel(fls)

    fig = plt.figure(figsize=(14, 10))
    graphics.plot_modeloutput_section_withtrib(fig=fig, model=model)
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_modeloutput_map():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FlowlineModel(fls)

    fig, ax = plt.subplots()
    graphics.plot_modeloutput_map(gdir, ax=ax, model=model)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_multiple_models():

    # test directory
    testdir = os.path.join(get_test_dir(), 'tmp_mdir')
    utils.mkdir(testdir, reset=True)

    # Init
    cfg.initialize()
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PATHS['working_dir'] = testdir
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'
    cfg.PARAMS['run_mb_calibration'] = True
    cfg.PARAMS['border'] = 40

    # Get the RGI ID
    hef_rgi = gpd.read_file(get_demo_file('divides_hef.shp'))
    hef_rgi.loc[0, 'RGIId'] = 'RGI50-11.00897'

    gdirs = workflow.init_glacier_regions(hef_rgi)
    workflow.gis_prepro_tasks(gdirs)
    workflow.climate_tasks(gdirs)
    workflow.inversion_tasks(gdirs)

    models = []
    for gdir in gdirs:
        flowline.init_present_time_glacier(gdir)
        fls = gdir.read_pickle('model_flowlines')
        models.append(flowline.FlowlineModel(fls))

    fig, ax = plt.subplots()
    graphics.plot_modeloutput_map(gdirs, ax=ax, model=models)
    fig.tight_layout()

    shutil.rmtree(testdir)
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_thick_alt():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_distributed_thickness(gdir, ax=ax,
                                        varname_suffix='_alt')
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_thick_interp():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_distributed_thickness(gdir, ax=ax,
                                        varname_suffix='_interp')
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_catch_areas():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_areas(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_chhota_shigri():

    testdir = os.path.join(get_test_dir(), 'tmp_chhota')
    utils.mkdir(testdir, reset=True)

    # Init
    cfg.initialize()
    cfg.PATHS['dem_file'] = get_demo_file('dem_chhota_shigri.tif')
    cfg.PARAMS['border'] = 80
    cfg.PARAMS['use_intersects'] = False
    cfg.PATHS['working_dir'] = testdir

    hef_file = get_demo_file('divides_RGI50-14.15990.shp')
    df = gpd.read_file(hef_file)
    df['Area'] = df.Area * 1e-6  # cause it was in m2
    df['RGIId'] = ['RGI50-14.15990' + d for d in ['_d01', '_d02']]

    gdirs = workflow.init_glacier_regions(df)
    workflow.gis_prepro_tasks(gdirs)
    for gdir in gdirs:
        climate.apparent_mb_from_linear_mb(gdir)
    workflow.execute_entity_task(inversion.prepare_for_inversion, gdirs)
    workflow.execute_entity_task(inversion.mass_conservation_inversion, gdirs)
    workflow.execute_entity_task(inversion.filter_inversion_output, gdirs)
    workflow.execute_entity_task(flowline.init_present_time_glacier, gdirs)

    models = []
    for gdir in gdirs:
        flowline.init_present_time_glacier(gdir)
        fls = gdir.read_pickle('model_flowlines')
        models.append(flowline.FlowlineModel(fls))

    fig, ax = plt.subplots()
    graphics.plot_modeloutput_map(gdirs, ax=ax, model=models)
    fig.tight_layout()
    shutil.rmtree(testdir)
    return fig


@pytest.mark.graphic
@mpl_image_compare()
def test_ice_cap():

    testdir = os.path.join(get_test_dir(), 'tmp_icecap')
    utils.mkdir(testdir, reset=True)

    cfg.initialize()
    cfg.PARAMS['use_intersects'] = False
    cfg.PATHS['dem_file'] = get_demo_file('dem_RGI50-05.08389.tif')
    cfg.PARAMS['border'] = 60
    cfg.PATHS['working_dir'] = testdir

    df = gpd.read_file(get_demo_file('divides_RGI50-05.08389.shp'))
    df['Area'] = df.Area * 1e-6  # cause it was in m2
    df['RGIId'] = ['RGI50-05.08389_d{:02d}'.format(d+1) for d in df.index]

    gdirs = workflow.init_glacier_regions(df)
    workflow.gis_prepro_tasks(gdirs)

    from salem import mercator_grid, Map
    smap = mercator_grid((gdirs[0].cenlon, gdirs[0].cenlat),
                         extent=[20000, 23000])
    smap = Map(smap)

    fig, ax = plt.subplots()
    graphics.plot_catchment_width(gdirs, ax=ax, add_intersects=True,
                                  add_touches=True, smap=smap)
    fig.tight_layout()
    shutil.rmtree(testdir)
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_coxe():

    testdir = os.path.join(get_test_dir(), 'tmp_coxe')
    utils.mkdir(testdir, reset=True)

    # Init
    cfg.initialize()
    cfg.PARAMS['use_intersects'] = False
    cfg.PATHS['dem_file'] = get_demo_file('dem_RGI50-01.10299.tif')
    cfg.PARAMS['border'] = 40
    cfg.PARAMS['clip_tidewater_border'] = False
    cfg.PARAMS['use_multiple_flowlines'] = False

    hef_file = get_demo_file('rgi_RGI50-01.10299.shp')
    entity = gpd.read_file(hef_file).iloc[0]

    gdir = oggm.GlacierDirectory(entity, base_dir=testdir, reset=True)
    gis.define_glacier_region(gdir, entity=entity)
    gis.glacier_masks(gdir)
    centerlines.compute_centerlines(gdir)
    centerlines.initialize_flowlines(gdir)
    centerlines.compute_downstream_line(gdir)
    centerlines.compute_downstream_bedshape(gdir)
    centerlines.catchment_area(gdir)
    centerlines.catchment_intersections(gdir)
    centerlines.catchment_width_geom(gdir)
    centerlines.catchment_width_correction(gdir)
    climate.apparent_mb_from_linear_mb(gdir)
    inversion.prepare_for_inversion(gdir)
    inversion.mass_conservation_inversion(gdir)
    inversion.filter_inversion_output(gdir)

    flowline.init_present_time_glacier(gdir)

    fls = gdir.read_pickle('model_flowlines')

    p = gdir.read_pickle('linear_mb_params')
    mb_mod = massbalance.LinearMassBalance(ela_h=p['ela_h'],
                                           grad=p['grad'])
    mb_mod.temp_bias = -0.3
    model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0,
                                    inplace=True,
                                    is_tidewater=True)

    # run
    model.run_until(200)
    assert model.calving_m3_since_y0 > 0

    fig, ax = plt.subplots()
    graphics.plot_modeloutput_map(gdir, ax=ax, model=model)
    fig.tight_layout()
    shutil.rmtree(testdir)
    return fig
