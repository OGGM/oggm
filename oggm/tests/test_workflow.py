from __future__ import division

import warnings
warnings.filterwarnings("once", category=DeprecationWarning)

import os
import shutil
import unittest
import pickle
from functools import partial

import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.testing import assert_allclose

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, rmsd, write_centerlines_to_shape
from oggm.tests import is_slow, ON_TRAVIS, RUN_WORKFLOW_TESTS
from oggm.core.models import flowline, massbalance
from oggm import tasks
from oggm import utils

# do we event want to run the tests?
if not RUN_WORKFLOW_TESTS:
    raise unittest.SkipTest('Skipping all workflow tests.')

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_workflow')
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
    cfg.PARAMS['use_multiprocessing'] = not ON_TRAVIS

    # Working dir
    cfg.PATHS['working_dir'] = TEST_DIR

    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')

    # Read in the RGI file
    rgi_file = get_demo_file('rgi_oetztal.shp')
    rgidf = gpd.GeoDataFrame.from_file(rgi_file)

    # Be sure data is downloaded because lock doesn't work
    cl = utils.get_cru_cl_file()

    # Params
    cfg.PARAMS['border'] = 70
    cfg.PARAMS['use_optimized_inversion_params'] = True
    cfg.PARAMS['tstar_search_window'] = [1902, 0]

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
        cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
        cfg.PATHS['cru_dir'] = '~'
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
        cfg.PATHS['climate_file'] = '~'
        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cfg.PATHS['cru_dir'] = os.path.dirname(cru_dir)
        with warnings.catch_warnings():
            # There is a warning from salem
            warnings.simplefilter("ignore")
            workflow.execute_entity_task(tasks.process_cru_data, gdirs)
        tasks.compute_ref_t_stars(gdirs)
        tasks.distribute_t_stars(gdirs)
        with open(CLI_LOGF, 'wb') as f:
            pickle.dump('cru', f)

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
        maxs = cfg.PARAMS['max_shape_param']
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
                assert_allclose(gldf['oggm_volume_km3'], _vol, rtol=0.03)
                assert_allclose(gldf['ref_area_km2'], _area, rtol=0.03)
                maxo = max([fl.order for fl in model.fls])
                for fl in model.fls:
                    self.assertTrue(np.all(fl.bed_shape > 0))
                    self.assertTrue(np.all(fl.bed_shape <= maxs))
                    if len(model.fls) > 1:
                        if fl.order == (maxo-1):
                            self.assertTrue(fl.flows_to is fls[-1])

    @is_slow
    def test_crossval(self):

        gdirs = up_to_distrib()

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
            h, w = gdir.get_flowline_hw()
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

        gdirs = up_to_inversion()

        workflow.execute_entity_task(flowline.init_present_time_glacier, gdirs)
        rand_glac = partial(flowline.random_glacier_evolution, nyears=200)
        workflow.execute_entity_task(rand_glac, gdirs)

        for gd in gdirs:

            path = gd.get_filepath('past_model')

            # See that we are running ok
            with flowline.FileModel(path) as model:
                vol = model.volume_km3_ts()
                area = model.area_km2_ts()
                len = model.length_m_ts()

                self.assertTrue(np.all(np.isfinite(vol) & vol != 0.))
                self.assertTrue(np.all(np.isfinite(area) & area != 0.))
                self.assertTrue(np.all(np.isfinite(len) & len != 0.))
