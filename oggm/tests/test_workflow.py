import os
import shutil
import unittest
import pickle
import pytest
import json
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from oggm import graphics

salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')

# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, write_centerlines_to_shape, ModelSettings
from oggm.tests import mpl_image_compare
from oggm.tests.funcs import get_test_dir, use_multiprocessing, characs_apply_func
from oggm.shop import cru
from oggm.core import flowline, gis, inversion, centerlines, massbalance
from oggm import tasks
from oggm import utils
from oggm.core.massbalance import (MonthlyTIModel, MultipleFlowlineMassBalance,
                                   apparent_mb_from_any_mb)
from oggm.core.flowline import (run_from_climate_data, run_random_climate,
                                init_present_time_glacier, run_constant_climate, MixedBedFlowline)
from oggm.core.inversion import get_inversion_volume
from oggm.exceptions import InvalidWorkflowError

# Globals
pytestmark = pytest.mark.test_env("workflow")
_TEST_DIR = os.path.join(get_test_dir(), 'tmp_workflow')
CLI_LOGF = os.path.join(_TEST_DIR, 'clilog.pkl')


def clean_dir(testdir):
    shutil.rmtree(testdir)
    os.makedirs(testdir)


def up_to_climate(reset=False, use_mp=None, params_file=None):
    """Run the tasks you want."""

    # test directory
    if not os.path.exists(_TEST_DIR):
        os.makedirs(_TEST_DIR)
    if reset:
        clean_dir(_TEST_DIR)

    if not os.path.exists(CLI_LOGF):
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('none', f)

    # Init
    cfg.initialize(file=params_file)

    # Use multiprocessing
    use_mp = False
    if use_mp is None:
        cfg.PARAMS['use_multiprocessing'] = use_multiprocessing()
    else:
        cfg.PARAMS['use_multiprocessing'] = use_mp

    # Working dir
    cfg.PATHS['working_dir'] = _TEST_DIR
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))

    # Read in the RGI file
    rgi_file = get_demo_file('rgi_oetztal.shp')
    rgidf = gpd.read_file(rgi_file)

    # Make a fake marine and lake terminating glacier
    cfg.PARAMS['tidewater_type'] = 4  # make lake also calve
    rgidf.loc[0, 'GlacType'] = '0199'
    rgidf.loc[1, 'GlacType'] = '0299'

    # Use RGI6
    new_ids = []
    count = 0
    for s in rgidf.RGIId:
        s = s.replace('RGI50', 'RGI60')
        if '_d0' in s:
            # We dont do this anymore
            s = 'RGI60-11.{:05d}'.format(99999 - count)
            count += 1
        new_ids.append(s)
    rgidf['RGIId'] = new_ids

    # Here as well - we don't do the custom RGI IDs anymore
    rgidf = rgidf.loc[['_d0' not in d for d in rgidf.RGIId]].copy()

    # Be sure data is downloaded
    cru.get_cru_cl_file()

    # Params
    cfg.PARAMS['border'] = 70
    cfg.PARAMS['prcp_fac'] = 1.75
    cfg.PARAMS['temp_melt'] = -1.75
    cfg.PARAMS['use_kcalving_for_inversion'] = True
    cfg.PARAMS['geodetic_mb_period'] = '2000-01-01_2010-01-01'
    cfg.PARAMS['use_kcalving_for_run'] = True
    cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['baseline_climate'] = 'CRU'
    cfg.PARAMS['evolution_model'] = 'FluxBased'
    cfg.PARAMS['downstream_line_shape'] = 'parabola'

    # Go
    gdirs = workflow.init_glacier_directories(rgidf)

    try:
        tasks.catchment_width_correction(gdirs[0])
    except Exception:
        reset = True

    if reset:
        # First preprocessing tasks
        workflow.gis_prepro_tasks(gdirs)

    return gdirs


def up_to_inversion(reset=False, params_file=None):
    """Run the tasks you want."""

    gdirs = up_to_climate(reset=reset, params_file=params_file)

    with open(CLI_LOGF, 'rb') as f:
        clilog = pickle.load(f)

    if clilog != 'histalp':
        reset = True
    else:
        try:
            tasks.prepare_for_inversion(gdirs[0])
        except Exception:
            reset = True

    if reset:
        # Use histalp for the actual inversion test
        cfg.PARAMS['baseline_climate'] = 'HISTALP'
        workflow.climate_tasks(gdirs, overwrite_gdir=True,
                               override_missing=-500)
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('histalp', f)

        # Inversion
        workflow.inversion_tasks(gdirs)

    return gdirs


def random_for_plot():

    # Fake Reset (all these tests are horribly coded)
    try:
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('none', f)
    except FileNotFoundError:
        pass
    gdirs = up_to_inversion(reset=True)

    workflow.execute_entity_task(flowline.init_present_time_glacier, gdirs)
    workflow.execute_entity_task(flowline.run_random_climate, gdirs, y0=1985,
                                 nyears=10, seed=0, output_filesuffix='_plot')
    return gdirs


class TestFullRun(unittest.TestCase):

    @pytest.mark.slow
    def test_some_characs(self):

        gdirs = up_to_inversion(reset=True)

        # Test the glacier charac
        dfc = utils.compile_glacier_statistics(gdirs,
                                               apply_func=characs_apply_func)

        assert 'glc_ext_num_perc' in dfc.columns
        assert np.all(np.isfinite(dfc.glc_ext_num_perc.values))

        self.assertFalse(np.all(dfc.terminus_type == 'Land-terminating'))
        assert np.all(dfc.iloc[:2].calving_rate_myr.values > 100)
        assert np.all(dfc.inv_volume_km3 > 0)
        assert np.all(dfc.bias == 0)
        assert np.all(dfc.temp_bias == 0)
        assert np.all(dfc.melt_f > cfg.PARAMS['melt_f_min'])
        assert np.all(dfc.melt_f < cfg.PARAMS['melt_f_max'])
        dfc = utils.compile_climate_statistics(gdirs)
        sel = ['flowline_mean_elev', '1980-2010_avg_temp_mean_elev']
        cc = dfc[sel].corr().values[0, 1]
        assert cc < -0.8
        assert np.all(0 < dfc['1980-2010_aar'])
        assert np.all(0.6 > dfc['1980-2010_aar'])

    @pytest.mark.slow
    def test_calibrate_inversion_from_consensus(self):

        # test mini params file, define path relative to file location
        fp_mini_params = os.path.join(os.path.dirname(__file__),
                                      'mini_params_for_test.cfg')
        gdirs = up_to_inversion(params_file=fp_mini_params)

        # check if mini params file is used as expected
        assert cfg.PARAMS['lru_maxsize'] == 123

        df_orig = workflow.calibrate_inversion_from_consensus(
            gdirs, overwrite_observations=True, ignore_missing=True)
        df = df_orig.dropna()
        np.testing.assert_allclose(df.vol_itmix_m3.sum(),
                                   df.vol_oggm_m3.sum(),
                                   rtol=0.001)
        np.testing.assert_allclose(df.vol_itmix_m3, df.vol_oggm_m3, rtol=0.41)

        # check that observations values are added to the file
        assert 'ref_volume_m3' in gdirs[0].observations
        gdir_0_ref_vol = gdirs[0].observations['ref_volume_m3']

        # test that using observations from file is working
        provided_vol_095 = 0
        for gdir in gdirs:
            ref_vol = gdir.observations['ref_volume_m3']
            ref_vol['value'] *= 0.95
            provided_vol_095 += ref_vol['value']
            gdir.observations['ref_volume_m3'] = ref_vol
        df_095 = workflow.calibrate_inversion_from_volume(
            gdirs,
            overwrite_observations=True,
            apply_fs_on_mismatch=True,
        )
        assert df_095.vol_oggm_m3.sum() < df_orig.vol_oggm_m3.sum()
        assert (gdirs[0].observations['ref_volume_m3']['value'] <
                gdir_0_ref_vol['value'])
        np.testing.assert_allclose(
            provided_vol_095, df_095['vol_oggm_m3'].sum(), rtol=1e-2)

        # test user provided volume is working
        delta_volume_m3 = 100000000
        user_provided_volume_m3 = df.vol_itmix_m3.sum() - delta_volume_m3
        # we only use those which have an itmix volume
        rgi_ids_in_ref_volume = list(df.dropna().index)
        # check that error is raised if already an observations stored in file
        with pytest.raises(InvalidWorkflowError) as exc_info:
            df = workflow.calibrate_inversion_from_volume(
                gdirs,
                overwrite_observations=False,
                ref_volume_m3=user_provided_volume_m3,
                rgi_ids_in_ref_volume=rgi_ids_in_ref_volume,
            )
        assert 'You have provided an reference volume, ' in str(exc_info.value)

        df = workflow.calibrate_inversion_from_volume(
            gdirs,
            overwrite_observations=True,
            ref_volume_m3=user_provided_volume_m3,
            rgi_ids_in_ref_volume=rgi_ids_in_ref_volume,
        )

        np.testing.assert_allclose(
            user_provided_volume_m3,
            df.loc[rgi_ids_in_ref_volume].vol_oggm_m3.sum(),
            rtol=0.001)
        gdir_0_ref_vol_user_provided = gdirs[0].observations['ref_volume_m3']
        assert (gdir_0_ref_vol['value'] !=
                gdir_0_ref_vol_user_provided['value'])
        assert (gdir_0_ref_vol['vol_itmix_m3'] ==
                gdir_0_ref_vol_user_provided['vol_itmix_m3'])


    @pytest.mark.slow
    def test_shapefile_output(self):

        gdirs = up_to_climate(use_mp=True)

        fpath = os.path.join(_TEST_DIR, 'centerlines.shp')
        write_centerlines_to_shape(gdirs, path=fpath)

        import salem
        shp = salem.read_shapefile(fpath)
        self.assertTrue(shp is not None)
        shp = shp.loc[shp.RGIID == 'RGI60-11.00897']
        self.assertEqual(len(shp), 3)
        self.assertEqual(shp.loc[shp.LE_SEGMENT.idxmax()].MAIN, 1)

        fpath = os.path.join(_TEST_DIR, 'centerlines_ext.shp')
        write_centerlines_to_shape(gdirs, path=fpath,
                                   ensure_exterior_match=True)
        shp_ext = salem.read_shapefile(fpath)
        # We check the length of the segment for a change
        shp_ext = shp_ext.to_crs('EPSG:32632')
        assert_allclose(shp_ext.geometry.length, shp_ext['LE_SEGMENT'],
                        rtol=1e-3)

        fpath = os.path.join(_TEST_DIR, 'centerlines_ext_smooth.shp')
        write_centerlines_to_shape(gdirs, path=fpath,
                                   ensure_exterior_match=True,
                                   simplify_line_before=0.2,
                                   corner_cutting=3,
                                   simplify_line_after=0.05)
        shp_ext_smooth = salem.read_shapefile(fpath)
        # This is a bit different of course
        assert_allclose(shp_ext['LE_SEGMENT'], shp_ext_smooth['LE_SEGMENT'],
                        rtol=2)

        fpath = os.path.join(_TEST_DIR, 'flowlines.shp')
        write_centerlines_to_shape(gdirs, path=fpath, flowlines_output=True)
        shp_f = salem.read_shapefile(fpath)
        self.assertTrue(shp_f is not None)
        shp_f = shp_f.loc[shp_f.RGIID == 'RGI60-11.00897']
        self.assertEqual(len(shp_f), 3)
        self.assertEqual(shp_f.loc[shp_f.LE_SEGMENT.idxmax()].MAIN, 1)
        # The flowline is cut so shorter
        assert shp_f.LE_SEGMENT.max() < shp.LE_SEGMENT.max() * 0.8

        fpath = os.path.join(_TEST_DIR, 'widths_geom.shp')
        write_centerlines_to_shape(gdirs, path=fpath, geometrical_widths_output=True)
        # Salem can't read it
        shp_w = gpd.read_file(fpath)
        self.assertTrue(shp_w is not None)
        shp_w = shp_w.loc[shp_w.RGIID == 'RGI60-11.00897']
        self.assertEqual(len(shp_w), 90)

        fpath = os.path.join(_TEST_DIR, 'widths_corr.shp')
        write_centerlines_to_shape(gdirs, path=fpath, corrected_widths_output=True)
        # Salem can't read it
        shp_w = gpd.read_file(fpath)
        self.assertTrue(shp_w is not None)
        shp_w = shp_w.loc[shp_w.RGIID == 'RGI60-11.00897']
        self.assertEqual(len(shp_w), 90)

        # Test that one wrong glacier still works
        base_dir = os.path.join(cfg.PATHS['working_dir'], 'dummy_pergla')
        utils.mkdir(base_dir, reset=True)
        gdirs = workflow.execute_entity_task(utils.copy_to_basedir, gdirs,
                                             base_dir=base_dir, setup='all')
        os.remove(gdirs[0].get_filepath('centerlines'))
        cfg.PARAMS['continue_on_error'] = True
        write_centerlines_to_shape(gdirs)

    @pytest.mark.slow
    def test_random(self):

        # Fake Reset (all these tests are horribly coded)
        if not os.path.exists(_TEST_DIR):
            os.makedirs(_TEST_DIR)
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('none', f)
        gdirs = up_to_inversion(reset=False)

        # First tests
        df = utils.compile_glacier_statistics(gdirs)
        df['volume_before_calving_km3'] = df['volume_before_calving'] * 1e-9
        assert np.sum(~ df.volume_before_calving.isnull()) == 2
        dfs = df.iloc[:2]
        assert np.all(dfs['volume_before_calving_km3'] < dfs['inv_volume_km3'])
        assert_allclose(df['inv_flowline_glacier_area']*1e-6,
                        df['rgi_area_km2'])

        workflow.execute_entity_task(flowline.init_present_time_glacier, gdirs)
        # Check init_present_time_glacier not messing around too much
        for gd in gdirs:
            from oggm.core.massbalance import LinearMassBalance
            from oggm.core.flowline import FluxBasedModel
            mb_mod = LinearMassBalance(ela_h=2500)
            fls = gd.read_pickle('model_flowlines')
            model = FluxBasedModel(fls, mb_model=mb_mod)
            df.loc[gd.rgi_id, 'start_area_km2'] = model.area_km2
            df.loc[gd.rgi_id, 'start_volume_km3'] = model.volume_km3
            df.loc[gd.rgi_id, 'start_length'] = model.length_m
        assert_allclose(df['rgi_area_km2'], df['start_area_km2'], rtol=0.02)
        assert_allclose(df['rgi_area_km2'].sum(), df['start_area_km2'].sum(),
                        rtol=0.005)
        assert_allclose(df['inv_volume_km3'], df['start_volume_km3'])
        assert_allclose(df['inv_volume_km3'].sum(),
                        df['start_volume_km3'].sum())
        assert_allclose(df['main_flowline_length'], df['start_length'])

        workflow.execute_entity_task(flowline.run_random_climate, gdirs,
                                     nyears=100, seed=0, y0=1985,
                                     store_monthly_step=True,
                                     mb_elev_feedback='monthly',
                                     output_filesuffix='_test')

        for gd in gdirs:
            path = gd.get_filepath('model_geometry', filesuffix='_test')
            # See that we are running ok
            model = flowline.FileModel(path)
            vol = model.volume_km3_ts()
            area = model.area_km2_ts()

            self.assertTrue(np.all(np.isfinite(vol) & vol != 0.))
            self.assertTrue(np.all(np.isfinite(area) & area != 0.))

            ds_diag = gd.get_filepath('model_diagnostics', filesuffix='_test')
            ds_diag = xr.open_dataset(ds_diag)
            df = vol.to_frame('RUN')
            df['DIAG'] = ds_diag.volume_m3.to_series() * 1e-9
            assert_allclose(df.RUN, df.DIAG)
            df = area.to_frame('RUN')
            df['DIAG'] = ds_diag.area_m2.to_series() * 1e-6
            assert_allclose(df.RUN, df.DIAG)

        # Test output
        ds = utils.compile_run_output(gdirs, input_filesuffix='_test')
        assert_allclose(ds_diag.volume_m3, ds.volume.sel(rgi_id=gd.rgi_id))
        assert_allclose(ds_diag.area_m2, ds.area.sel(rgi_id=gd.rgi_id))
        assert_allclose(ds_diag.length_m, ds.length.sel(rgi_id=gd.rgi_id))
        df = ds.volume.sel(rgi_id=gd.rgi_id).to_series().to_frame('OUT')
        df['RUN'] = ds_diag.volume_m3.to_series()
        assert_allclose(df.RUN, df.OUT)

        # Compare to statistics
        df = utils.compile_glacier_statistics(gdirs)
        df['y0_vol'] = ds.volume.sel(rgi_id=df.index, time=0) * 1e-9
        df['y0_area'] = ds.area.sel(rgi_id=df.index, time=0) * 1e-6
        df['y0_len'] = ds.length.sel(rgi_id=df.index, time=0)
        assert_allclose(df['rgi_area_km2'], df['y0_area'], 0.06)
        assert_allclose(df['inv_volume_km3'], df['y0_vol'], 0.04)
        assert_allclose(df['inv_volume_bsl_km3'], 0)
        assert_allclose(df['main_flowline_length'], df['y0_len'])

        # Calving stuff
        assert ds.isel(rgi_id=0).calving[-1] > 0
        assert ds.isel(rgi_id=0).calving_rate[1] > 0
        assert ds.isel(rgi_id=0).volume_bsl[0] == 0
        assert ds.isel(rgi_id=0).volume_bwl[0] > 0
        assert ds.isel(rgi_id=1).calving[-1] > 0
        assert ds.isel(rgi_id=1).calving_rate[1] > 0
        assert ds.isel(rgi_id=1).volume_bsl[-1] == 0


@pytest.mark.slow
def test_merge_gridded_data():
    gdirs = up_to_inversion()
    workflow.execute_entity_task(tasks.distribute_thickness_per_altitude,
                                 gdirs
                                 )
    ds_merged = workflow.merge_gridded_data(
        gdirs,
        included_variables='distributed_thickness',
        reset=True)

    df = utils.compile_glacier_statistics(gdirs)

    # check if distributed volume is the same as inversion volume for each gdir
    for gdir in gdirs:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            ds = ds.load()

        inv_volume = df[df.index == gdir.rgi_id]['inv_volume_km3'].values
        inv_volume_gridded = (ds.distributed_thickness.sum() *
                              ds.salem.grid.dx**2) * 1e-9
        assert_allclose(inv_volume, inv_volume_gridded, atol=1.1e-7)

    # check if merged distributed volume is the same as total inversion volume
    inv_volume_gridded_merged = (ds_merged.distributed_thickness.sum() *
                                 ds_merged.salem.grid.dx**2) * 1e-9
    assert_allclose(df['inv_volume_km3'].sum(), inv_volume_gridded_merged,
                    rtol=1e-6)

    # test providing a custom grid
    default_grid = utils.combine_grids(gdirs)
    custom_grid_dict = default_grid.to_dict()
    custom_grid_dict['nxny'] = (custom_grid_dict['nxny'][0] - 2,
                                custom_grid_dict['nxny'][1] - 2)
    custom_grid = salem.gis.Grid.from_dict(custom_grid_dict)

    ds_merged_custom = workflow.merge_gridded_data(
        gdirs,
        output_grid=custom_grid,
        included_variables='distributed_thickness',
        reset=True)

    assert ds_merged_custom.salem.grid.nx < ds_merged.salem.grid.nx
    assert ds_merged_custom.salem.grid.ny < ds_merged.salem.grid.ny

    # total volume should be the same with a custom grid
    inv_volume_gridded_merged_custom = (ds_merged_custom.distributed_thickness.sum() *
                                        ds_merged_custom.salem.grid.dx ** 2) * 1e-9
    assert_allclose(inv_volume_gridded_merged_custom, inv_volume_gridded_merged,
                    rtol=1e-6)


@pytest.mark.slow
@pytest.mark.graphic
@mpl_image_compare(remove_text=True, multi=True)
def test_plot_region_inversion():

    pytest.importorskip('pytest_mpl')

    gdirs = up_to_inversion(reset=True)

    # We prepare for the plot, which needs our own map to proceed.
    # Lets do a local mercator grid
    g = salem.mercator_grid(center_ll=(10.86, 46.85),
                            extent=(27000, 21000))
    # And a map accordingly
    sm = salem.Map(g, countries=False)
    sm.set_topography(get_demo_file('srtm_oetztal.tif'))

    # Give this to the plot function
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    graphics.plot_inversion(gdirs, smap=sm, ax=ax1, linewidth=1.5, vmax=250)

    # test automatic definition of larger plotting grid with extend_plot_limit
    graphics.plot_inversion(gdirs, ax=ax2, linewidth=1.5, vmax=250,
                            extend_plot_limit=True)

    fig.tight_layout()
    return fig


@pytest.mark.slow
@pytest.mark.graphic
@mpl_image_compare(remove_text=True, multi=True)
def test_plot_region_model():

    pytest.importorskip('pytest_mpl')

    gdirs = random_for_plot()

    dfc = utils.compile_task_log(gdirs,
                                 task_names=['run_random_climate_plot'])
    assert np.all(dfc['run_random_climate_plot'] == 'SUCCESS')

    # We prepare for the plot, which needs our own map to proceed.
    # Lets do a local mercator grid
    g = salem.mercator_grid(center_ll=(10.86, 46.85),
                            extent=(27000, 21000))
    # And a map accordingly
    sm = salem.Map(g, countries=False)
    sm.set_topography(get_demo_file('srtm_oetztal.tif'))

    # Give this to the plot function
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    graphics.plot_modeloutput_map(gdirs, smap=sm, ax=ax1,
                                  filesuffix='_plot', vmax=250,
                                  modelyr=10, linewidth=1.5)
    # test automatic definition of larger plotting grid with extend_plot_limit
    graphics.plot_modeloutput_map(gdirs, ax=ax2,
                                  filesuffix='_plot', vmax=250,
                                  modelyr=10, linewidth=1.5,
                                  extend_plot_limit=True)

    fig.tight_layout()
    return fig


def test_rgi7_glacier_dirs():
    # create test dir
    if not os.path.exists(_TEST_DIR):
        os.makedirs(_TEST_DIR)
    # initialize
    cfg.initialize()
    # Set intersects
    cfg.set_intersects_db(gpd.read_file(get_demo_file('rgi7g_hef_intersects.shp')))
    cfg.PATHS['working_dir'] = _TEST_DIR
    # load and read test data
    hef_rgi7_df = gpd.read_file(get_demo_file('rgi7g_hef.shp'))
    # create GDIR
    gdir = workflow.init_glacier_directories(hef_rgi7_df)[0]
    assert gdir
    assert gdir.rgi_region == '11'
    assert gdir.rgi_area_km2 > 8
    assert gdir.name == 'Hintereisferner'
    assert len(gdir.intersects_ids) == 2
    assert gdir.rgi_region == '11'
    assert gdir.rgi_subregion == '11-01'
    assert gdir.rgi_region_name == '11: Central Europe'
    assert gdir.rgi_subregion_name == '11-01: Alps'
    assert gdir.rgi_version == '70G'
    assert gdir.rgi_dem_source == 'COPDEM30'
    assert gdir.utm_zone == 32


def test_rgi7_complex_glacier_dirs():
    # create test dir
    if not os.path.exists(_TEST_DIR):
        os.makedirs(_TEST_DIR)
    # initialize
    cfg.initialize()
    cfg.PATHS['working_dir'] = _TEST_DIR
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
    cfg.PARAMS['border'] = 5

    # load and read test data
    hef_rgi7_df = gpd.read_file(get_demo_file('rgi7c_hef.shp'))
    # create GDIR
    gdir = workflow.init_glacier_directories(hef_rgi7_df)[0]
    assert gdir
    assert gdir.rgi_region == '11'
    assert gdir.rgi_area_km2 > 20
    assert gdir.name == ''
    assert gdir.rgi_region == '11'
    assert gdir.rgi_subregion == '11-01'
    assert gdir.rgi_region_name == '11: Central Europe'
    assert gdir.rgi_subregion_name == '11-01: Alps'
    assert gdir.rgi_version == '70C'
    assert gdir.rgi_dem_source is None
    assert gdir.utm_zone == 32

    # Also test rgi7g_to_complex
    tasks.define_glacier_region(gdir)
    tasks.glacier_masks(gdir)
    rgi7g_file = gpd.read_file(get_demo_file('rgi7g_hef_complex.shp'))
    with open(get_demo_file('hef-CtoG_links.json'), 'r') as f:
        rgi7c_to_g_links = json.load(f)
    tasks.rgi7g_to_complex(gdir,
                           rgi7g_file=rgi7g_file,
                           rgi7c_to_g_links=rgi7c_to_g_links)

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        assert ds.sub_entities.max().item() == (len(rgi7c_to_g_links[gdir.rgi_id]) - 1)

@pytest.fixture(scope='class')
def with_class_wd(request, test_dir, hef_gdir):
    # dependency on hef_gdir to ensure proper initialization order
    prev_wd = cfg.PATHS['working_dir']
    cfg.PATHS['working_dir'] = os.path.join(
        test_dir, request.cls.__name__ + '_wd')
    utils.mkdir(cfg.PATHS['working_dir'], reset=True)
    yield
    # teardown
    cfg.PATHS['working_dir'] = prev_wd


@pytest.mark.usefixtures('with_class_wd')
class TestGdirSettings:

    @pytest.mark.slow
    def test_settings_massbalance(self):
        rgi_ids = ['RGI60-11.00897']
        gdirs = workflow.init_glacier_directories(
            rgi_ids, from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')
        gdir = gdirs[0]
        mb_calib_cluster = gdirs[0].read_json('mb_calib')

        # checking for backwards compatibility
        mb_model_cluster = MonthlyTIModel(gdir)
        assert mb_model_cluster.melt_f == mb_calib_cluster['melt_f']
        assert mb_model_cluster.prcp_fac == mb_calib_cluster['prcp_fac']
        assert mb_model_cluster.temp_bias == mb_calib_cluster['temp_bias']

        # test usage of individual settings file for the informed threestep,
        # should result in the same parameters as on the cluster
        settings_informed_threestep = ModelSettings(gdir,
                                                    filesuffix='_informed_threestep')
        settings_informed_threestep['use_temp_bias_from_file'] = True
        settings_informed_threestep['use_winter_prcp_fac'] = True
        settings_informed_threestep['baseline_climate'] = 'W5E5'

        workflow.execute_entity_task(tasks.mb_calibration_from_hugonnet_mb,
                                     gdirs,
                                     overwrite_gdir=True,
                                     informed_threestep=True,
                                     settings_filesuffix='_informed_threestep',
                                     )
        mb_calib_threestep = gdirs[0].read_yml('settings',
                                               '_informed_threestep')
        assert mb_calib_cluster['melt_f'] == mb_calib_threestep['melt_f']
        # now MonthlyTIModel takes length of months into account which explains
        # this 'large' difference in prcp_fac
        assert_allclose(mb_calib_cluster['prcp_fac'],
                        mb_calib_threestep['prcp_fac'],
                        atol=5e-2)
        assert_allclose(mb_calib_cluster['temp_bias'],
                        mb_calib_threestep['temp_bias'],
                        atol=1e-3)

        # recalibration without overwrite_gdir should raise an error
        with pytest.raises(InvalidWorkflowError):
            workflow.execute_entity_task(tasks.mb_calibration_from_hugonnet_mb,
                                         gdirs,
                                         overwrite_gdir=False,
                                         informed_threestep=True,
                                         settings_filesuffix='_informed_threestep',
                                         )

        # test MassBalance settings during dynamic run
        custom_settings = ModelSettings(gdir,
                                        filesuffix='_adapted_melt_f',
                                        parent_filesuffix='_informed_threestep'
                                        )
        custom_settings['melt_f'] *= 1.05

        with pytest.raises(FileNotFoundError):
            ModelSettings(gdir,
                          filesuffix='_no_parent',
                          parent_filesuffix='_not_existing',
                          )
        workflow.execute_entity_task(init_present_time_glacier, gdirs)
        model_orig = run_from_climate_data(gdir, ye=2020)
        mb_model_custom = MultipleFlowlineMassBalance(
            gdir,
            mb_model_class=MonthlyTIModel,
            settings_filesuffix='_adapted_melt_f',
        )

        model_custom = run_from_climate_data(gdir, ye=2020,
                                             mb_model=mb_model_custom,
                                             output_filesuffix='_custom')
        # original volume should be larger due to smaller original melt_f
        assert model_orig.volume_m3 > model_custom.volume_m3

        # should also work with just providing the run_settings to run-task
        model_custom_2 = run_from_climate_data(gdir, ye=2020,
                                               settings_filesuffix='_adapted_melt_f',
                                               output_filesuffix='_custom_2')
        assert model_custom.volume_m3 == model_custom_2.volume_m3

        # test run_random_climate
        random_orig = run_random_climate(gdir, nyears=100, y0=2000, seed=0)
        random_custom = run_random_climate(gdir, nyears=100, y0=2000, seed=0,
                                           settings_filesuffix='_adapted_melt_f',)
        assert random_custom.volume_m3 < random_orig.volume_m3

        # test run_constant_climate
        const_orig = run_constant_climate(gdir, nyears=100, y0=2000)
        const_custom = run_constant_climate(gdir, nyears=100, y0=2000,
                                            settings_filesuffix='_adapted_melt_f',)
        assert const_custom.volume_m3 < const_orig.volume_m3

        # try climate_tasks with different baseline_climate
        # first test for current setting
        assert gdir.get_climate_info()['baseline_climate_source'] == 'GSWP3_W5E5'
        # now change baseline climate in run_settings
        custom_settings = ModelSettings(gdir, filesuffix='_adapted_baseline')
        custom_settings['baseline_climate'] = 'ERA5'
        custom_settings['geodetic_mb_period'] = '2000-01-01_2010-01-01'  # ERA5 only unitl 2018
        # ERA5 only available until 2018
        #custom_settings['geodetic_mb_period'] = '2000-01-01_2018-01-01'
        workflow.climate_tasks(gdirs, settings_filesuffix='_adapted_baseline',
                               input_filesuffix='',
                               override_missing='-1000')
        # Now it should be W5E5
        assert gdir.get_climate_info()['baseline_climate_source'] == 'ERA5'

    @pytest.mark.slow
    def test_settings_dynamics(self):
        rgi_ids = ['RGI60-11.00897']
        gdirs = workflow.init_glacier_directories(
            rgi_ids, from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')
        gdir = gdirs[0]

        # create a test settings file with a larger glen a parameter
        custom_settings = ModelSettings(gdir,
                                        filesuffix='_large_glen_a',)
        custom_settings['inversion_glen_a'] = gdir.get_diagnostics()['inversion_glen_a'] * 10

        # test run_random_climate
        random_orig = run_random_climate(gdir, nyears=100, y0=2000, seed=0)
        random_big = run_random_climate(gdir, settings_filesuffix='_large_glen_a',
                                        nyears=100, y0=2000, seed=0)
        assert random_big.volume_m3 < random_orig.volume_m3

        # test run_constant_climate
        const_orig = run_constant_climate(gdir, nyears=100, y0=2000)
        const_big = run_constant_climate(gdir,
                                         settings_filesuffix='_large_glen_a',
                                         nyears=100, y0=2000)
        assert const_big.volume_m3 < const_orig.volume_m3

        # test run_from_climate_data
        clim_orig = run_from_climate_data(gdir, ye=2020)
        clim_big = run_from_climate_data(gdir,
                                         settings_filesuffix='_large_glen_a',
                                         ye=2020)
        assert clim_big.volume_m3 < clim_orig.volume_m3

        # test run_with_hydro
        default_settings = ModelSettings(gdir)
        default_settings['store_model_geometry'] = True
        tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data, ye=2020,
                             output_filesuffix='_hydro_orig')
        custom_settings['store_model_geometry'] = True
        tasks.run_with_hydro(gdir, settings_filesuffix='_large_glen_a',
                             run_task=tasks.run_from_climate_data, ye=2020,
                             output_filesuffix='_hydro_big')
        ds_hydro_orig = utils.compile_run_output(gdir,
                                                 input_filesuffix='_hydro_orig')
        ds_hydro_big = utils.compile_run_output(gdir,
                                                input_filesuffix='_hydro_big')
        assert ds_hydro_big.volume.values[-1] < ds_hydro_orig.volume.values[-1]

        # test for cfl error with small cfl number in run_settings
        custom_settings['cfl_number'] = 1e-6
        with pytest.raises(RuntimeError):
            run_from_climate_data(gdir, settings_filesuffix='_large_glen_a',
                                  ys=2000, ye=2020)
        # SemiImplicit model does not use the cfl_number as it is fixed
        custom_settings['evolution_model'] = 'SemiImplicit'
        run_from_climate_data(gdir, settings_filesuffix='_large_glen_a',
                              ys=2000, ye=2020,
                              output_filesuffix='_rand_orig_thick')

        # Initialize a new flowline an check if settings are used for
        custom_settings['min_ice_thick_for_length'] = 50
        run_from_climate_data(gdir, settings_filesuffix='_large_glen_a',
                              ys=2000, ye=2020,
                              output_filesuffix='_rand_larger_thick')
        ds_orig_thick = utils.compile_run_output(
            gdir, input_filesuffix='_rand_orig_thick')
        ds_larger_thick = utils.compile_run_output(
            gdir, input_filesuffix='_rand_larger_thick')

        assert np.all(ds_orig_thick.length >= ds_larger_thick.length)

    @pytest.mark.slow
    def test_settings_inversion(self):
        rgi_ids = ['RGI60-11.00897']
        gdirs = workflow.init_glacier_directories(
            rgi_ids, from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')
        gdir = gdirs[0]

        # for reference, perform an inversion with default paramters
        workflow.inversion_tasks(gdir)
        default_volume = get_inversion_volume(gdir)

        # test with a larger glen a parameter in the settings file
        custom_settings = ModelSettings(gdir,
                                        filesuffix='_large_glen_a', )
        custom_settings['inversion_glen_a'] = gdir.settings['inversion_glen_a'] * 10
        workflow.inversion_tasks(gdir, settings_filesuffix='_large_glen_a',
                                 input_filesuffix='')
        assert default_volume > get_inversion_volume(gdir,
                                                     input_filesuffix='_large_glen_a')

        # test with different lambda value
        custom_settings = ModelSettings(gdir,
                                        filesuffix='_large_lambda',
                                        )
        custom_settings['trapezoid_lambdas'] = gdir.settings['trapezoid_lambdas'] * 1.5
        workflow.inversion_tasks(gdir, settings_filesuffix='_large_lambda',
                                 input_filesuffix='')
        inv_out_default = gdir.read_pickle('inversion_output')
        inv_out_lambda = gdir.read_pickle('inversion_output',
                                          filesuffix='_large_lambda')
        # do not look at last 5 grid points because of filter_inversion_output
        assert np.all(inv_out_default[0]['thick'][:-5] <
                      inv_out_lambda[0]['thick'][:-5])

        # test calibrate from consensus with different lambda,
        glen_a_before = custom_settings['inversion_glen_a']
        workflow.calibrate_inversion_from_consensus(
            gdirs, settings_filesuffix='_large_lambda',
            input_filesuffix='',
            apply_fs_on_mismatch=True)
        glen_a_after = custom_settings['inversion_glen_a']
        assert glen_a_before < glen_a_after

        # test calibration from consensus with different melt_f, including
        # apparent mass balance
        custom_settings = ModelSettings(gdir,
                                        filesuffix='_large_melt_f',
                                        )
        custom_settings['melt_f'] = gdir.settings['melt_f'] * 1.1
        massbalance.apparent_mb_from_any_mb(gdir,
                                            settings_filesuffix='_large_melt_f',
                                            input_filesuffix='',
                                            output_filesuffix='_large_melt_f',)
        workflow.calibrate_inversion_from_consensus(
            gdirs, settings_filesuffix='_large_melt_f',
            input_filesuffix='_large_melt_f', apply_fs_on_mismatch=True)
        glen_a_after = custom_settings['inversion_glen_a']
        inv_out_melt_f = gdir.read_pickle('inversion_output',
                                          filesuffix='_large_melt_f')
        # with larger melt_f the residual of the apparent mass balance must be
        # larger, to compensate for the more negative mb
        assert (gdir.get_diagnostics()['apparent_mb_from_any_mb_residual'] <
                custom_settings['apparent_mb_from_any_mb_residual'])
        # hence also the resulting flux must be more negative everywhere
        assert np.all(inv_out_default[0]['flux'] < inv_out_melt_f[0]['flux'])

        # test inversion calving with settings
        coxe_file = get_demo_file('rgi_RGI50-01.10299.shp')
        cfg.PATHS['dem_file'] = get_demo_file('dem_RGI50-01.10299.tif')
        entity = gpd.read_file(coxe_file).iloc[0]
        entity.RGIId = 'RGI60-01.10299'
        cfg.PARAMS['use_kcalving_for_inversion'] = True
        cfg.PARAMS['use_kcalving_for_run'] = True

        gdir = oggm.GlacierDirectory(entity)
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
        tasks.process_dummy_cru_file(gdir, seed=0)
        massbalance.mb_calibration_from_hugonnet_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir)

        inversion.prepare_for_inversion(gdir)
        inversion.mass_conservation_inversion(gdir)
        cls1 = gdir.read_pickle('inversion_output')
        # Increase calving for this one using settings
        custom_settings = ModelSettings(gdir,
                                        filesuffix='_large_k',
                                        parent_filesuffix=''
                                        )
        custom_settings['inversion_calving_k'] = 1

        res_bef = gdir.settings['apparent_mb_from_any_mb_residual']
        out = inversion.find_inversion_calving_from_any_mb(
            gdir, settings_filesuffix='_large_k', output_filesuffix='_large_k',
        )
        cls2 = gdir.read_pickle('inversion_output', filesuffix='_large_k')

        # Calving increases the volume and adds a residual
        v_ref = np.sum([np.sum(fl['volume']) for fl in cls1])
        v_new = np.sum([np.sum(fl['volume']) for fl in cls2])
        assert v_ref < v_new
        res_aft = custom_settings['apparent_mb_from_any_mb_residual']
        assert res_aft > res_bef


@pytest.mark.usefixtures('with_class_wd')
class TestGdirObservations:

    @pytest.mark.slow
    def test_observations_ref_mb(self):
        rgi_ids = ['RGI60-11.00897']
        gdirs = workflow.init_glacier_directories(
            rgi_ids, from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')
        gdir = gdirs[0]
        mb_calib_cluster = gdirs[0].read_json('mb_calib')

        settings_informed_threestep = ModelSettings(gdir,
                                                    filesuffix='_informed_threestep')
        settings_informed_threestep['use_temp_bias_from_file'] = True
        settings_informed_threestep['use_winter_prcp_fac'] = True
        settings_informed_threestep['baseline_climate'] = 'W5E5'
        assert 'ref_mb' not in gdir.observations
        workflow.execute_entity_task(tasks.mb_calibration_from_hugonnet_mb,
                                     gdirs,
                                     overwrite_gdir=True,
                                     informed_threestep=True,
                                     settings_filesuffix='_informed_threestep',
                                     )
        # check that observation was added to the observation files and that the
        # result is the same as the previous implementation
        assert 'ref_mb' in gdir.observations
        for key in ['value', 'unit', 'err', 'period']:
            assert key in gdir.observations['ref_mb']
        mb_calib_threestep = gdirs[0].read_yml('settings',
                                               '_informed_threestep')
        assert mb_calib_cluster['melt_f'] == mb_calib_threestep['melt_f']
        # now MonthlyTIModel takes length of months into account which explains
        # this 'large' difference in prcp_fac
        assert_allclose(mb_calib_cluster['prcp_fac'],
                        mb_calib_threestep['prcp_fac'],
                        atol=5e-2)
        assert_allclose(mb_calib_cluster['temp_bias'],
                        mb_calib_threestep['temp_bias'],
                        atol=1e-3)
        # use slightly different observation from hugonnet for sensitivity test
        hugonnet_adapted = gdir.observations['ref_mb']
        hugonnet_adapted['value'] = (hugonnet_adapted['value'] +
                                     hugonnet_adapted['err'] / 2)
        gdir.observations_filesuffix = '_hugonnet_adapted'
        gdir.observations['ref_mb'] = hugonnet_adapted
        workflow.execute_entity_task(tasks.mb_calibration_from_hugonnet_mb,
                                     gdirs,
                                     overwrite_gdir=True,
                                     informed_threestep=True,
                                     settings_filesuffix='_informed_threestep',
                                     observations_filesuffix='_hugonnet_adapted',
                                     use_observations_file=True,
                                     )
        mb_calib_hugonnet_adapted = gdirs[0].read_yml('settings',
                                                      '_informed_threestep')
        # with a less negative geodetic mass balance we need more mass input,
        # with the same melt_f
        assert mb_calib_threestep['melt_f'] == mb_calib_hugonnet_adapted['melt_f']
        assert mb_calib_threestep['prcp_fac'] < mb_calib_hugonnet_adapted['prcp_fac']

        # set back to default for the following tests
        gdir.observations_filesuffix = ''

        # different combinations of providing observation in file and as kwarg
        ref_mb = gdir.observations['ref_mb']['value']
        ref_mb_unit = gdir.observations['ref_mb']['unit']
        ref_mb_err = gdir.observations['ref_mb']['err']
        ref_mb_period = gdir.observations['ref_mb']['period']
        # provided and file observations the same without overwriting should work
        tasks.mb_calibration_from_scalar_mb(
            gdir, observations_filesuffix='', overwrite_observations=False,
            ref_mb=ref_mb, ref_mb_unit=ref_mb_unit, ref_mb_err=ref_mb_err,
            ref_mb_period=ref_mb_period, ref_mb_years=None,
            write_to_gdir=False,
        )
        # providing different in file and kwargs should raise
        with pytest.raises(InvalidWorkflowError) as exc_info:
            tasks.mb_calibration_from_scalar_mb(
                gdir, observations_filesuffix='', overwrite_observations=False,
                ref_mb=ref_mb-100, ref_mb_err=ref_mb_err,
                ref_mb_period=ref_mb_period, ref_mb_years=None,
                write_to_gdir=False,
            )
        assert 'You provided a reference mass balance, but' in str(exc_info.value)
        # but is allowed when overwrite is True
        tasks.mb_calibration_from_scalar_mb(
            gdir, observations_filesuffix='', overwrite_observations=True,
            ref_mb=ref_mb-100, ref_mb_err=ref_mb_err,
            ref_mb_period=ref_mb_period, ref_mb_years=None,
            write_to_gdir=False,
        )
        assert ref_mb - gdir.observations['ref_mb']['value'] == 100
        # use an empty observations file and provide through kwargs
        tasks.mb_calibration_from_scalar_mb(
            gdir, observations_filesuffix='_empty', overwrite_observations=False,
            ref_mb=ref_mb, ref_mb_err=ref_mb_err,
            ref_mb_period=ref_mb_period, ref_mb_years=None,
            write_to_gdir=False,
        )
        assert 'ref_mb' in gdir.observations
        # provide nothing should raise
        with pytest.raises(InvalidWorkflowError) as exc_info:
            tasks.mb_calibration_from_scalar_mb(
                gdir, observations_filesuffix='empty2',
                overwrite_observations=False,
                ref_mb=None, ref_mb_err=None,
                ref_mb_period=None, ref_mb_years=None,
                write_to_gdir=False,
            )
        assert 'You have not provided an reference' in str(exc_info.value)
