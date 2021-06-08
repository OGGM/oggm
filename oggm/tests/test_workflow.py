import os
import shutil
import unittest
import pickle
import pytest
import warnings
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from oggm import graphics

salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, write_centerlines_to_shape
from oggm.tests import mpl_image_compare
from oggm.tests.funcs import get_test_dir, use_multiprocessing, characs_apply_func
from oggm.shop import cru
from oggm.core import flowline
from oggm import tasks
from oggm import utils

# Globals
pytestmark = pytest.mark.test_env("workflow")
_TEST_DIR = os.path.join(get_test_dir(), 'tmp_workflow')
CLI_LOGF = os.path.join(_TEST_DIR, 'clilog.pkl')


def clean_dir(testdir):
    shutil.rmtree(testdir)
    os.makedirs(testdir)


def up_to_climate(reset=False):
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
    cfg.initialize()

    # Use multiprocessing
    cfg.PARAMS['use_multiprocessing'] = use_multiprocessing()
    cfg.PARAMS['use_multiprocessing'] = True

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
    rgidf['RGIId'] = [s.replace('RGI50', 'RGI60') for s in rgidf.RGIId]

    # Be sure data is downloaded
    cru.get_cru_cl_file()

    # Params
    cfg.PARAMS['border'] = 70
    cfg.PARAMS['tstar_search_window'] = [1902, 0]
    cfg.PARAMS['prcp_scaling_factor'] = 1.75
    cfg.PARAMS['temp_melt'] = -1.75
    cfg.PARAMS['use_kcalving_for_inversion'] = True
    cfg.PARAMS['use_kcalving_for_run'] = True

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


def up_to_inversion(reset=False):
    """Run the tasks you want."""

    gdirs = up_to_climate(reset=reset)

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
        cfg.PARAMS['temp_use_local_gradient'] = True
        cfg.PARAMS['baseline_climate'] = 'HISTALP'
        utils.apply_test_ref_tstars('histalp')
        workflow.climate_tasks(gdirs)
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('histalp', f)

        # Inversion
        workflow.inversion_tasks(gdirs)

    return gdirs


def up_to_distrib(reset=False):
    # for cross val basically

    gdirs = up_to_climate(reset=reset)

    with open(CLI_LOGF, 'rb') as f:
        clilog = pickle.load(f)

    if clilog != 'cru':
        reset = True
    else:
        try:
            tasks.compute_ref_t_stars(gdirs)
        except Exception:
            reset = True

    if reset:
        # Use CRU
        cfg.PARAMS['prcp_scaling_factor'] = 2.5
        cfg.PARAMS['temp_use_local_gradient'] = False
        cfg.PARAMS['baseline_climate'] = 'CRU'
        with warnings.catch_warnings():
            # There is a warning from salem
            warnings.simplefilter("ignore")
            workflow.execute_entity_task(tasks.process_cru_data, gdirs)
        tasks.compute_ref_t_stars(gdirs)
        workflow.execute_entity_task(tasks.local_t_star, gdirs)
        workflow.execute_entity_task(tasks.mu_star_calibration, gdirs)
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('cru', f)

    return gdirs


def random_for_plot():

    # Fake Reset (all these tests are horribly coded)
    try:
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('none', f)
    except FileNotFoundError:
        pass
    gdirs = up_to_inversion()

    workflow.execute_entity_task(flowline.init_present_time_glacier, gdirs)
    workflow.execute_entity_task(flowline.run_random_climate, gdirs,
                                 nyears=10, seed=0, output_filesuffix='_plot')
    return gdirs


class TestFullRun(unittest.TestCase):

    @pytest.mark.slow
    def test_some_characs(self):

        gdirs = up_to_inversion()

        # Test the glacier charac
        dfc = utils.compile_glacier_statistics(gdirs,
                                               apply_func=characs_apply_func)

        assert 'glc_ext_num_perc' in dfc.columns
        assert np.all(np.isfinite(dfc.glc_ext_num_perc.values))

        self.assertFalse(np.all(dfc.terminus_type == 'Land-terminating'))
        assert np.all(dfc.t_star > 1850)
        dfc = utils.compile_climate_statistics(gdirs)
        cc = dfc[['flowline_mean_elev',
                  'tstar_avg_temp_mean_elev']].corr().values[0, 1]
        assert cc < -0.8
        assert np.all(dfc.tstar_aar.mean() > 0.5)

    @pytest.mark.slow
    def test_calibrate_inversion_from_consensus(self):

        gdirs = up_to_inversion()
        df = workflow.calibrate_inversion_from_consensus(gdirs,
                                                         ignore_missing=True)
        df = df.dropna()
        np.testing.assert_allclose(df.vol_itmix_m3.sum(),
                                   df.vol_oggm_m3.sum(),
                                   rtol=0.01)
        np.testing.assert_allclose(df.vol_itmix_m3, df.vol_oggm_m3, rtol=0.36)

    @pytest.mark.slow
    def test_shapefile_output(self):

        gdirs = up_to_climate()
        fpath = os.path.join(_TEST_DIR, 'centerlines.shp')
        write_centerlines_to_shape(gdirs, path=fpath)

        import salem
        shp = salem.read_shapefile(fpath)
        self.assertTrue(shp is not None)
        shp = shp.loc[shp.RGIID == 'RGI60-11.00897']
        self.assertEqual(len(shp), 3)
        self.assertEqual(shp.loc[shp.LE_SEGMENT.idxmax()].MAIN, 1)

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
        assert_allclose(df['rgi_area_km2'], df['start_area_km2'], rtol=0.01)
        assert_allclose(df['rgi_area_km2'].sum(), df['start_area_km2'].sum(),
                        rtol=0.005)
        assert_allclose(df['inv_volume_km3'], df['start_volume_km3'])
        assert_allclose(df['inv_volume_km3'].sum(),
                        df['start_volume_km3'].sum())
        assert_allclose(df['main_flowline_length'], df['start_length'])

        workflow.execute_entity_task(flowline.run_random_climate, gdirs,
                                     nyears=100, seed=0,
                                     store_monthly_step=True,
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
        assert ds.isel(rgi_id=0).calving_rate[-1] > 0
        assert ds.isel(rgi_id=0).volume_bsl[-1] == 0
        assert ds.isel(rgi_id=0).volume_bwl[-1] > 0
        assert ds.isel(rgi_id=1).calving[-1] > 0
        assert ds.isel(rgi_id=1).calving_rate[-1] > 0
        assert ds.isel(rgi_id=1).volume_bsl[-1] == 0


@pytest.mark.slow
@pytest.mark.graphic
@mpl_image_compare(remove_text=True)
def test_plot_region_inversion():

    gdirs = up_to_inversion()

    # We prepare for the plot, which needs our own map to proceed.
    # Lets do a local mercator grid
    g = salem.mercator_grid(center_ll=(10.86, 46.85),
                            extent=(27000, 21000))
    # And a map accordingly
    sm = salem.Map(g, countries=False)
    sm.set_topography(get_demo_file('srtm_oetztal.tif'))

    # Give this to the plot function
    fig, ax = plt.subplots()
    graphics.plot_inversion(gdirs, smap=sm, ax=ax, linewidth=1.5, vmax=250)

    fig.tight_layout()
    return fig


@pytest.mark.slow
@pytest.mark.graphic
@mpl_image_compare(remove_text=True)
def test_plot_region_model():

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
    fig, ax = plt.subplots()
    graphics.plot_modeloutput_map(gdirs, smap=sm, ax=ax,
                                  filesuffix='_plot', vmax=250,
                                  modelyr=10, linewidth=1.5)

    fig.tight_layout()
    return fig
