from __future__ import division

import warnings
warnings.filterwarnings("once", category=DeprecationWarning)

import os
import shutil
import unittest
import pickle
from functools import partial

import pytest

import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, rmsd, write_centerlines_to_shape
from oggm.tests import is_slow, RUN_WORKFLOW_TESTS
from oggm.tests import is_graphic_test, BASELINE_DIR
from oggm.tests.funcs import get_test_dir, use_multiprocessing
from oggm.core.models import flowline, massbalance
from oggm import tasks
from oggm import utils

# do we event want to run the tests?
if not RUN_WORKFLOW_TESTS:
    raise unittest.SkipTest('Skipping all workflow tests.')

# Globals
TEST_DIR = os.path.join(get_test_dir(), 'tmp_workflow')
CLI_LOGF = os.path.join(TEST_DIR, 'clilog.pkl')


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

    # Read in the RGI file
    rgi_file = get_demo_file('rgi_oetztal.shp')
    rgidf = gpd.GeoDataFrame.from_file(rgi_file)

    # Be sure data is downloaded
    cl = utils.get_cru_cl_file()

    # Params
    cfg.PARAMS['border'] = 70
    cfg.PARAMS['use_optimized_inversion_params'] = True
    cfg.PARAMS['tstar_search_window'] = [1902, 0]
    cfg.PARAMS['invert_with_rectangular'] = False

    # Reset MP
    workflow.reset_multiprocessing()

    # Go
    gdirs = workflow.init_glacier_regions(rgidf)

    assert gdirs[14].name == 'Hintereisferner'

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
        cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
        cfg.PATHS['cru_dir'] = ''
        workflow.reset_multiprocessing()
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
        cfg.PATHS['climate_file'] = ''
        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cfg.PATHS['cru_dir'] = os.path.dirname(cru_dir)
        workflow.reset_multiprocessing()
        with warnings.catch_warnings():
            # There is a warning from salem
            warnings.simplefilter("ignore")
            workflow.execute_entity_task(tasks.process_cru_data, gdirs)
        tasks.compute_ref_t_stars(gdirs)
        tasks.distribute_t_stars(gdirs)
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('cru', f)

    return gdirs


def random_for_plot():

    # Fake Reset (all these tests are horribly coded)
    with open(CLI_LOGF, 'wb') as f:
        pickle.dump('none', f)
    gdirs = up_to_inversion()

    workflow.execute_entity_task(flowline.init_present_time_glacier, gdirs)
    workflow.execute_entity_task(flowline.random_glacier_evolution, gdirs,
                                 nyears=10, seed=0, filesuffix='_plot')
    return gdirs


class TestWorkflow(unittest.TestCase):

    @is_slow
    def test_init_present_time_glacier(self):

        gdirs = up_to_inversion()

        # Inversion Results
        cfg.PARAMS['invert_with_sliding'] = True
        cfg.PARAMS['optimize_thick'] = True
        workflow.inversion_tasks(gdirs)

        fpath = os.path.join(cfg.PATHS['working_dir'],
                             'inversion_optim_results.csv')
        df = pd.read_csv(fpath, index_col=0)
        r1 = rmsd(df['ref_volume_km3'], df['oggm_volume_km3'])
        r2 = rmsd(df['ref_volume_km3'], df['vas_volume_km3'])
        self.assertTrue(r1 < r2)

        cfg.PARAMS['invert_with_sliding'] = False
        cfg.PARAMS['optimize_thick'] = False
        workflow.inversion_tasks(gdirs)

        fpath = os.path.join(cfg.PATHS['working_dir'],
                             'inversion_optim_results.csv')
        df = pd.read_csv(fpath, index_col=0)
        r1 = rmsd(df['ref_volume_km3'], df['oggm_volume_km3'])
        r2 = rmsd(df['ref_volume_km3'], df['vas_volume_km3'])
        self.assertTrue(r1 < r2)

        # Init glacier
        d = gdirs[0].read_pickle('inversion_params')
        fs = d['fs']
        glen_a = d['glen_a']
        for gdir in gdirs:
            flowline.init_present_time_glacier(gdir)
            mb_mod = massbalance.ConstantMassBalanceModel(gdir)
            fls = gdir.read_pickle('model_flowlines')
            model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                            fs=fs, glen_a=glen_a)
            _vol = model.volume_km3
            _area = model.area_km2
            if gdir.rgi_id in df.index:
                gldf = df.loc[gdir.rgi_id]
                assert_allclose(gldf['oggm_volume_km3'], _vol, rtol=0.05)
                assert_allclose(gldf['ref_area_km2'], _area, rtol=0.05)
                maxo = max([fl.order for fl in model.fls])
                for fl in model.fls:
                    if len(model.fls) > 1:
                        if fl.order == (maxo-1):
                            self.assertTrue(fl.flows_to is fls[-1])

        # Test the glacier charac
        dfc = utils.glacier_characteristics(gdirs)
        self.assertTrue(np.all(dfc.terminus_type == 'Land-terminating'))
        cc = dfc[['dem_mean_elev', 'clim_temp_avgh']].corr().values[0, 1]
        self.assertTrue(cc > 0.4)

    @is_slow
    def test_crossval(self):

        gdirs = up_to_distrib()

        # in case we ran crossval we need to rerun
        tasks.compute_ref_t_stars(gdirs)
        tasks.distribute_t_stars(gdirs)

        # before crossval
        refmustars = []
        for gdir in gdirs:
            tdf = pd.read_csv(gdir.get_filepath('local_mustar'))
            refmustars.append(tdf['mu_star'].values[0])

        tasks.crossval_t_stars(gdirs)
        file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
        df = pd.read_csv(file, index_col=0)

        # after crossval we need to rerun
        tasks.compute_ref_t_stars(gdirs)
        tasks.distribute_t_stars(gdirs)

        # Test if quicker crossval is also OK
        tasks.quick_crossval_t_stars(gdirs)
        file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
        dfq = pd.read_csv(file, index_col=0)

        # after crossval we need to rerun
        tasks.compute_ref_t_stars(gdirs)
        tasks.distribute_t_stars(gdirs)

        np.testing.assert_allclose(np.abs(df.cv_bias), np.abs(dfq.cv_bias),
                                   rtol=0.05)
        np.testing.assert_allclose(df.cv_prcp_fac, dfq.cv_prcp_fac)

        print(df)

        # see if the process didn't brake anything
        mustars = []
        for gdir in gdirs:
            tdf = pd.read_csv(gdir.get_filepath('local_mustar'))
            mustars.append(tdf['mu_star'].values[0])
        np.testing.assert_allclose(refmustars, mustars)

        # make some mb tests
        from oggm.core.models.massbalance import PastMassBalanceModel
        for rid in df.index:
            gdir = [g for g in gdirs if g.rgi_id == rid][0]
            h, w = gdir.get_inversion_flowline_hw()
            cfg.PARAMS['use_bias_for_run'] = False
            mbmod = PastMassBalanceModel(gdir)
            mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE'].to_frame(name='ref')
            for yr in mbdf.index:
                mbdf.loc[yr, 'mine'] = mbmod.get_specific_mb(h, w, year=yr)
            mm = mbdf.mean()
            np.testing.assert_allclose(df.loc[rid].bias,
                                       mm['mine'] - mm['ref'], atol=1e-3)
            cfg.PARAMS['use_bias_for_run'] = True
            mbmod = PastMassBalanceModel(gdir)
            mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE'].to_frame(name='ref')
            for yr in mbdf.index:
                mbdf.loc[yr, 'mine'] = mbmod.get_specific_mb(h, w, year=yr)
            mm = mbdf.mean()
            np.testing.assert_allclose(mm['mine'], mm['ref'], atol=1e-3)

    @is_slow
    def test_shapefile_output(self):

        # Just to increase coveralls, hehe
        gdirs = up_to_climate()
        fpath = os.path.join(TEST_DIR, 'centerlines.shp')
        write_centerlines_to_shape(gdirs, fpath)

        import salem
        shp = salem.read_shapefile(fpath)
        self.assertTrue(shp is not None)
        shp = shp.loc[shp.RGIID == 'RGI50-11.00897']
        self.assertEqual(len(shp), 4)
        self.assertEqual(shp.MAIN.sum(), 3)
        self.assertEqual(shp.loc[shp.LE_SEGMENT.argmax()].MAIN, 1)

    @is_slow
    def test_random(self):

        # Fake Reset (all these tests are horribly coded)
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('none', f)
        gdirs = up_to_inversion()

        workflow.execute_entity_task(flowline.init_present_time_glacier, gdirs)
        workflow.execute_entity_task(flowline.random_glacier_evolution, gdirs,
                                     nyears=200, seed=0, filesuffix='_test')

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
        ds = utils.compile_run_output(gdirs, filesuffix='_test', monthly=True)
        assert_allclose(ds_diag.volume_m3, ds.volume.sel(rgi_id=gd.rgi_id))
        assert_allclose(ds_diag.area_m2, ds.area.sel(rgi_id=gd.rgi_id))
        assert_allclose(ds_diag.length_m, ds.length.sel(rgi_id=gd.rgi_id))
        # Test output
        ds = utils.compile_run_output(gdirs, filesuffix='_test')
        df = ds.volume.sel(rgi_id=gd.rgi_id).to_series().to_frame('OUT')
        df['RUN'] = ds_diag.volume_m3.to_series()
        assert_allclose(df.RUN, df.OUT)


    @is_slow
    def test_random_mb_seed(self):
        gdirs = up_to_inversion()
        seed = None
        years = np.arange(1800, 2201)
        odf = pd.DataFrame(index=years)
        for gd in gdirs[:6]:
            mb = massbalance.RandomMassBalanceModel(gd, y0=1970, seed=seed)
            h, w = gd.get_inversion_flowline_hw()
            odf[gd.rgi_id] = mb.get_specific_mb(h, w, year=years)
        self.assertLessEqual(odf.corr().mean().mean(), 0.5)
        seed = 1
        for gd in gdirs[:6]:
            mb = massbalance.RandomMassBalanceModel(gd, y0=1970, seed=seed)
            h, w = gd.get_inversion_flowline_hw()
            odf[gd.rgi_id] = mb.get_specific_mb(h, w, year=years)
        self.assertGreaterEqual(odf.corr().mean().mean(), 0.9)


@is_slow
@is_graphic_test
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=20)
def test_plot_region_inversion():

    import matplotlib.pyplot as plt
    import salem
    from oggm import graphics

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
    graphics.plot_region_inversion(gdirs, salemmap=sm, ax=ax)

    fig.tight_layout()
    return fig


@is_slow
@is_graphic_test
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=20)
def test_plot_region_model():

    import matplotlib.pyplot as plt
    import salem
    from oggm import graphics

    gdirs = random_for_plot()

    # We prepare for the plot, which needs our own map to proceed.
    # Lets do a local mercator grid
    g = salem.mercator_grid(center_ll=(10.86, 46.85),
                            extent=(27000, 21000))
    # And a map accordingly
    sm = salem.Map(g, countries=False)
    sm.set_topography(get_demo_file('srtm_oetztal.tif'))

    # Give this to the plot function
    fig, ax = plt.subplots()
    graphics.plot_region_model_output(gdirs, salemmap=sm, ax=ax,
                                      filesuffix='_plot',
                                      modelyr=10)

    fig.tight_layout()
    return fig
