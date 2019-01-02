import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import os
import shutil
import unittest
import pickle
import pytest
import geopandas as gpd
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import salem
from oggm import graphics

# Locals
import oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, write_centerlines_to_shape
from oggm.tests import mpl_image_compare
from oggm.tests.funcs import (get_test_dir, use_multiprocessing,
                              patch_url_retrieve_github)
from oggm.core import flowline
from oggm import tasks
from oggm import utils

# Globals
pytestmark = pytest.mark.test_env("workflow")
TEST_DIR = os.path.join(get_test_dir(), 'tmp_workflow')
CLI_LOGF = os.path.join(TEST_DIR, 'clilog.pkl')

_url_retrieve = None


def setup_module(module):
    module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github


def teardown_module(module):
    oggm.utils._downloads.oggm_urlretrieve = module._url_retrieve


def clean_dir(testdir):
    shutil.rmtree(testdir)
    os.makedirs(testdir)


def up_to_climate(reset=False):
    """Run the tasks you want."""

    # test directory
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    if reset:
        clean_dir(TEST_DIR)

    if not os.path.exists(CLI_LOGF):
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('none', f)

    # Init
    cfg.initialize()

    # Use multiprocessing
    cfg.PARAMS['use_multiprocessing'] = use_multiprocessing()

    # Working dir
    cfg.PATHS['working_dir'] = TEST_DIR
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))

    # Read in the RGI file
    rgi_file = get_demo_file('rgi_oetztal.shp')
    rgidf = gpd.read_file(rgi_file)

    # Be sure data is downloaded
    utils.get_cru_cl_file()

    # Params
    cfg.PARAMS['border'] = 70
    cfg.PARAMS['tstar_search_window'] = [1902, 0]
    cfg.PARAMS['run_mb_calibration'] = True

    # Go
    gdirs = workflow.init_glacier_regions(rgidf)

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
        cru_dir = get_demo_file('HISTALP_precipitation_all_abs_1801-2014.nc')
        cfg.PATHS['cru_dir'] = os.path.dirname(cru_dir)
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
        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cfg.PATHS['cru_dir'] = os.path.dirname(cru_dir)
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
        dfc = utils.compile_glacier_statistics(gdirs)
        self.assertTrue(np.all(dfc.terminus_type == 'Land-terminating'))
        assert np.all(dfc.t_star > 1900)
        dfc = utils.compile_climate_statistics(gdirs)
        cc = dfc[['flowline_mean_elev',
                  'tstar_avg_temp_mean_elev']].corr().values[0, 1]
        assert cc < -0.8
        assert np.all(dfc.tstar_aar.mean() > 0.5)

    @pytest.mark.slow
    def test_shapefile_output(self):

        # Just to increase coveralls, hehe
        gdirs = up_to_climate()
        fpath = os.path.join(TEST_DIR, 'centerlines.shp')
        write_centerlines_to_shape(gdirs, path=fpath)

        import salem
        shp = salem.read_shapefile(fpath)
        self.assertTrue(shp is not None)
        shp = shp.loc[shp.RGIID == 'RGI50-11.00897']
        self.assertEqual(len(shp), 3)
        self.assertEqual(shp.loc[shp.LE_SEGMENT.idxmax()].MAIN, 1)

    @pytest.mark.slow
    def test_random(self):

        # Fake Reset (all these tests are horribly coded)
        if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('none', f)
        gdirs = up_to_inversion()

        workflow.execute_entity_task(flowline.init_present_time_glacier, gdirs)
        workflow.execute_entity_task(flowline.run_random_climate, gdirs,
                                     nyears=200, seed=0,
                                     store_monthly_step=True,
                                     output_filesuffix='_test')

        for gd in gdirs:

            path = gd.get_filepath('model_run', filesuffix='_test')
            # See that we are running ok
            with flowline.FileModel(path) as model:
                vol = model.volume_km3_ts()
                area = model.area_km2_ts()
                length = model.length_m_ts()

                self.assertTrue(np.all(np.isfinite(vol) & vol != 0.))
                self.assertTrue(np.all(np.isfinite(area) & area != 0.))
                self.assertTrue(np.all(np.isfinite(length) & length != 0.))

            ds_diag = gd.get_filepath('model_diagnostics', filesuffix='_test')
            ds_diag = xr.open_dataset(ds_diag)
            df = vol.to_frame('RUN')
            df['DIAG'] = ds_diag.volume_m3.to_series() * 1e-9
            assert_allclose(df.RUN, df.DIAG)
            df = area.to_frame('RUN')
            df['DIAG'] = ds_diag.area_m2.to_series() * 1e-6
            assert_allclose(df.RUN, df.DIAG)
            df = length.to_frame('RUN')
            df['DIAG'] = ds_diag.length_m.to_series()
            assert_allclose(df.RUN, df.DIAG)

        # Test output
        ds = utils.compile_run_output(gdirs, filesuffix='_test')
        assert_allclose(ds_diag.volume_m3, ds.volume.sel(rgi_id=gd.rgi_id))
        assert_allclose(ds_diag.area_m2, ds.area.sel(rgi_id=gd.rgi_id))
        assert_allclose(ds_diag.length_m, ds.length.sel(rgi_id=gd.rgi_id))
        # Test output
        ds = utils.compile_run_output(gdirs, filesuffix='_test')
        df = ds.volume.sel(rgi_id=gd.rgi_id).to_series().to_frame('OUT')
        df['RUN'] = ds_diag.volume_m3.to_series()
        assert_allclose(df.RUN, df.OUT)


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
