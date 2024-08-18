import warnings

import pytest
import shutil

import os
import matplotlib.pyplot as plt
import numpy as np

salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')
pytest.importorskip('pytest_mpl')

# Local imports
import oggm.utils
from oggm.tests import mpl_image_compare
from oggm.tests.funcs import init_columbia_eb, init_hef, get_test_dir
from oggm import graphics
from oggm.core import (gis, inversion, climate, centerlines, flowline,
                       massbalance)
import oggm.cfg as cfg
from oggm.utils import get_demo_file
from oggm import utils, workflow

# Warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r'.*guessing baseline image.*')

# Globals
pytestmark = pytest.mark.test_env("graphics")


def setup_module():
    graphics.set_oggm_cmaps()


def teardown_module():
    graphics.set_oggm_cmaps()

# ----------------------------------------------------------
# Lets go


def test_surf_to_nan():

    surf = np.array([1., 0, 0, 1])
    thick = np.array([1, 0, 0, 1])
    sh = graphics.surf_to_nan(surf, thick)
    np.testing.assert_allclose(sh, [1, 0, 0, 1])

    surf = np.array([1., 0, 0, 0, 1])
    thick = np.array([1, 0, 0, 0, 1])
    sh = graphics.surf_to_nan(surf, thick)
    np.testing.assert_allclose(sh, [1, 0, np.nan, 0, 1])

    surf = np.array([1., 0, 0, 0, 0, 1])
    thick = np.array([1, 0, 0, 0, 0, 1])
    sh = graphics.surf_to_nan(surf, thick)
    np.testing.assert_allclose(sh, [1, 0, np.nan, np.nan, 0, 1])

    surf = np.array([1., 0, 1, 0, 1])
    thick = np.array([1, 0, 1, 0, 1])
    sh = graphics.surf_to_nan(surf, thick)
    np.testing.assert_allclose(sh, [1, 0, 1, 0, 1])


@pytest.mark.static_map
@pytest.mark.internet
@pytest.mark.graphic
@mpl_image_compare(tolerance=26)
def test_googlemap():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_googlemap(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.internet
@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_domain():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_domain(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_centerlines():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_raster():
    fig, ax = plt.subplots()
    gdir = init_hef()
    gis.gridded_attributes(gdir)
    graphics.plot_raster(gdir, var_name='aspect', cmap='twilight', ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_flowlines():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax, use_flowlines=True)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_downstream():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_centerlines(gdir, ax=ax, add_downstream=True,
                              use_flowlines=True)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_width():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_width(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_width_corrected():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_width(gdir, ax=ax, corrected=True,
                                  add_intersects=True,
                                  add_touches=True)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_inversion():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_inversion(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.slow
@pytest.mark.graphic
@mpl_image_compare(multi=True)
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
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'
    cfg.PARAMS['trapezoid_lambdas'] = 1
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['prcp_fac'] = 2.5
    cfg.PATHS['working_dir'] = testdir

    # Get the RGI ID
    hef_rgi = gpd.read_file(get_demo_file('divides_hef.shp'))
    hef_rgi.loc[0, 'RGIId'] = 'RGI50-11.00897'

    gdirs = workflow.init_glacier_directories(hef_rgi)
    workflow.gis_prepro_tasks(gdirs)
    workflow.execute_entity_task(climate.process_climate_data, gdirs)
    workflow.execute_entity_task(massbalance.mb_calibration_from_scalar_mb,
                                 gdirs, ref_mb_years=(1980, 2000), ref_mb=0)
    workflow.execute_entity_task(massbalance.apparent_mb_from_any_mb,
                                 gdirs, mb_years=(1980, 2000))
    workflow.inversion_tasks(gdirs)

    fig, ax = plt.subplots()
    graphics.plot_inversion(gdirs, ax=ax)
    fig.tight_layout()
    shutil.rmtree(testdir)
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
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
@mpl_image_compare(multi=True)
def test_modelsection_withtrib():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FlowlineModel(fls)

    fig = plt.figure(figsize=(14, 10))
    graphics.plot_modeloutput_section_withtrib(fig=fig, model=model)
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_modeloutput_map():

    gdir = init_hef()
    flowline.init_present_time_glacier(gdir)
    fls = gdir.read_pickle('model_flowlines')
    model = flowline.FlowlineModel(fls)

    fig, ax = plt.subplots()
    graphics.plot_modeloutput_map(gdir, ax=ax, model=model)
    fig.tight_layout()
    return fig


@pytest.mark.slow
@pytest.mark.graphic
@mpl_image_compare(multi=True)
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
    cfg.PARAMS['trapezoid_lambdas'] = 1
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['prcp_fac'] = 2.5
    cfg.PARAMS['border'] = 40

    # Get the RGI ID
    hef_rgi = gpd.read_file(get_demo_file('divides_hef.shp'))
    hef_rgi.loc[0, 'RGIId'] = 'RGI50-11.00897'

    gdirs = workflow.init_glacier_directories(hef_rgi)
    workflow.gis_prepro_tasks(gdirs)
    workflow.execute_entity_task(climate.process_climate_data, gdirs)
    workflow.execute_entity_task(massbalance.mb_calibration_from_scalar_mb,
                                 gdirs, ref_mb_years=(1980, 2000), ref_mb=0)
    workflow.execute_entity_task(massbalance.apparent_mb_from_any_mb,
                                 gdirs, mb_years=(1980, 2000))
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
@mpl_image_compare(multi=True)
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
@mpl_image_compare(multi=True)
def test_thick_elev_bands():
    fig, ax = plt.subplots()
    gdir = init_columbia_eb(dir_name='test_thick_eb')
    workflow.inversion_tasks(utils.tolist(gdir))
    inversion.distribute_thickness_per_altitude(gdir)
    graphics.plot_distributed_thickness(gdir, ax=ax)
    fig.tight_layout()
    return fig

@pytest.mark.graphic
@mpl_image_compare(multi=True)
@pytest.mark.xfail
def test_model_section_calving():
    # I have no clue why this test fails on gh sometimes
    gdir = init_columbia_eb(dir_name='test_thick_eb')
    workflow.inversion_tasks(utils.tolist(gdir))
    flowline.init_present_time_glacier(gdir)

    fls = gdir.read_pickle('model_flowlines')
    mb_mod = massbalance.LinearMassBalance(1600)
    model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0,
                                    inplace=True,
                                    is_tidewater=True)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
    graphics.plot_modeloutput_section(model=model, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_catch_areas():
    fig, ax = plt.subplots()
    gdir = init_hef()
    graphics.plot_catchment_areas(gdir, ax=ax)
    fig.tight_layout()
    return fig


@pytest.mark.slow
@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_chhota_shigri():

    testdir = os.path.join(get_test_dir(), 'tmp_chhota')
    utils.mkdir(testdir, reset=True)

    # Init
    cfg.initialize()
    cfg.PATHS['dem_file'] = get_demo_file('dem_chhota_shigri.tif')
    cfg.PARAMS['border'] = 80
    cfg.PARAMS['use_intersects'] = False
    cfg.PATHS['working_dir'] = testdir
    cfg.PARAMS['trapezoid_lambdas'] = 1
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['prcp_fac'] = 2.5

    hef_file = get_demo_file('divides_RGI50-14.15990.shp')
    df = gpd.read_file(hef_file)
    df['Area'] = df.Area * 1e-6  # cause it was in m2
    df['RGIId'] = ['RGI50-14.15990' + d for d in ['_d01', '_d02']]

    gdirs = workflow.init_glacier_directories(df)
    workflow.gis_prepro_tasks(gdirs)
    for gdir in gdirs:
        massbalance.apparent_mb_from_linear_mb(gdir)
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


@pytest.mark.slow
@pytest.mark.graphic
@mpl_image_compare(multi=True)
def test_ice_cap():

    testdir = os.path.join(get_test_dir(), 'tmp_icecap')
    utils.mkdir(testdir, reset=True)

    cfg.initialize()
    cfg.PARAMS['use_intersects'] = False
    cfg.PATHS['dem_file'] = get_demo_file('dem_RGI50-05.08389.tif')
    cfg.PARAMS['border'] = 60
    cfg.PATHS['working_dir'] = testdir
    cfg.PARAMS['trapezoid_lambdas'] = 1
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['prcp_fac'] = 2.5

    df = gpd.read_file(get_demo_file('divides_RGI50-05.08389.shp'))
    df['Area'] = df.Area * 1e-6  # cause it was in m2
    df['RGIId'] = ['RGI50-05.08389_d{:02d}'.format(d+1) for d in df.index]
    df['GlacType'] = '1099'  # Make an ice cap

    gdirs = workflow.init_glacier_directories(df)
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


@pytest.mark.slow
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
    cfg.PARAMS['use_kcalving_for_inversion'] = True
    cfg.PARAMS['use_kcalving_for_run'] = True
    cfg.PARAMS['trapezoid_lambdas'] = 1
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['prcp_fac'] = 2.5

    hef_file = get_demo_file('rgi_RGI50-01.10299.shp')
    entity = gpd.read_file(hef_file).iloc[0]

    gdir = oggm.GlacierDirectory(entity, base_dir=testdir, reset=True)
    gis.define_glacier_region(gdir)
    gis.glacier_masks(gdir)
    centerlines.compute_centerlines(gdir)
    centerlines.initialize_flowlines(gdir)
    centerlines.compute_downstream_line(gdir)
    centerlines.compute_downstream_bedshape(gdir)
    centerlines.catchment_area(gdir)
    centerlines.catchment_intersections(gdir)
    centerlines.catchment_width_geom(gdir)
    centerlines.catchment_width_correction(gdir)
    massbalance.apparent_mb_from_linear_mb(gdir)
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
