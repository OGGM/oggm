import unittest
import os
import shutil
import time
import hashlib
import tarfile
import pytest
import itertools
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_array_equal, assert_allclose

salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')

import oggm
from oggm import utils, workflow, tasks
from oggm.utils import _downloads
from oggm import cfg
from oggm.tests.funcs import (get_test_dir, init_hef, TempEnvironmentVariable,
                              characs_apply_func)
from oggm.utils import shape_factor_adhikari
from oggm.exceptions import (InvalidParamsError, InvalidDEMError,
                             DownloadVerificationFailedException)


pytestmark = pytest.mark.test_env("utils")

TEST_GDIR_URL = ('https://cluster.klima.uni-bremen.de/~oggm/'
                 'test_gdirs/oggm_v1.1/')


def clean_dir(testdir):
    shutil.rmtree(testdir)
    os.makedirs(testdir)


class TestFuncs(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_show_versions(self):
        # just see that it runs
        out = utils.show_versions()
        assert len(out) > 30

    def test_signchange(self):
        ts = pd.Series([-2., -1., 1., 2., 3], index=np.arange(5))
        sc = utils.signchange(ts)
        assert_array_equal(sc, [0, 0, 1, 0, 0])
        ts = pd.Series([-2., -1., 1., 2., 3][::-1], index=np.arange(5))
        sc = utils.signchange(ts)
        assert_array_equal(sc, [0, 0, 0, 1, 0])

    def test_smooth(self):
        a = np.array([1., 4, 7, 7, 4, 1])
        b = utils.smooth1d(a, 3, kernel='mean')
        assert_allclose(b, [3, 4, 6, 6, 4, 3])
        kernel = [0.60653066,  1., 0.60653066]
        b = utils.smooth1d(a, 3, kernel=kernel)
        c = utils.smooth1d(a, 3)
        assert_allclose(b, c)

    def test_filter_rgi_name(self):

        name = 'Tustumena Glacier                              \x9c'
        expected = 'Tustumena Glacier'
        self.assertTrue(utils.filter_rgi_name(name), expected)

        name = 'Hintereisferner                                À'
        expected = 'Hintereisferner'
        self.assertTrue(utils.filter_rgi_name(name), expected)

        name = 'SPECIAL GLACIER                                3'
        expected = 'Special Glacier'
        self.assertTrue(utils.filter_rgi_name(name), expected)

    def test_floatyear_to_date(self):

        r = utils.floatyear_to_date(0)
        self.assertEqual(r, (0, 1))

        y, m = utils.floatyear_to_date([0, 1])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(m, [1, 1])

        y, m = utils.floatyear_to_date([0.00001, 1.00001])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(m, [1, 1])

        y, m = utils.floatyear_to_date([0.99999, 1.99999])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(m, [12, 12])

        yr = 1998 + 2 / 12
        r = utils.floatyear_to_date(yr)
        self.assertEqual(r, (1998, 3))

        yr = 1998 + 1 / 12
        r = utils.floatyear_to_date(yr)
        self.assertEqual(r, (1998, 2))

    def test_date_to_floatyear(self):

        r = utils.date_to_floatyear(0, 1)
        self.assertEqual(r, 0)

        r = utils.date_to_floatyear(1, 1)
        self.assertEqual(r, 1)

        r = utils.date_to_floatyear([0, 1], [1, 1])
        np.testing.assert_array_equal(r, [0, 1])

        yr = utils.date_to_floatyear([1998, 1998], [6, 7])
        y, m = utils.floatyear_to_date(yr)
        np.testing.assert_array_equal(y, [1998, 1998])
        np.testing.assert_array_equal(m, [6, 7])

        yr = utils.date_to_floatyear([1998, 1998], [2, 3])
        y, m = utils.floatyear_to_date(yr)
        np.testing.assert_array_equal(y, [1998, 1998])
        np.testing.assert_array_equal(m, [2, 3])

        time = pd.date_range('1/1/1800', periods=300*12-11, freq='MS')
        yr = utils.date_to_floatyear(time.year, time.month)
        y, m = utils.floatyear_to_date(yr)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        myr = utils.monthly_timeseries(1800, 2099)
        y, m = utils.floatyear_to_date(myr)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        myr = utils.monthly_timeseries(1800, ny=300)
        y, m = utils.floatyear_to_date(myr)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        time = pd.period_range('0001-01', '6000-1', freq='M')
        myr = utils.monthly_timeseries(1, 6000)
        y, m = utils.floatyear_to_date(myr)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        time = pd.period_range('0001-01', '6000-12', freq='M')
        myr = utils.monthly_timeseries(1, 6000, include_last_year=True)
        y, m = utils.floatyear_to_date(myr)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        with self.assertRaises(ValueError):
            utils.monthly_timeseries(1)

    def test_hydro_convertion(self):

        # October
        y, m = utils.hydrodate_to_calendardate(1, 1, start_month=10)
        assert (y, m) == (0, 10)
        y, m = utils.hydrodate_to_calendardate(1, 4, start_month=10)
        assert (y, m) == (1, 1)
        y, m = utils.hydrodate_to_calendardate(1, 12, start_month=10)
        assert (y, m) == (1, 9)

        y, m = utils.hydrodate_to_calendardate([1, 1, 1], [1, 4, 12],
                                               start_month=10)
        np.testing.assert_array_equal(y, [0, 1, 1])
        np.testing.assert_array_equal(m, [10, 1, 9])

        y, m = utils.calendardate_to_hydrodate(1, 1, start_month=10)
        assert (y, m) == (1, 4)
        y, m = utils.calendardate_to_hydrodate(1, 9, start_month=10)
        assert (y, m) == (1, 12)
        y, m = utils.calendardate_to_hydrodate(1, 10, start_month=10)
        assert (y, m) == (2, 1)

        y, m = utils.calendardate_to_hydrodate([1, 1, 1], [1, 9, 10],
                                               start_month=10)
        np.testing.assert_array_equal(y, [1, 1, 2])
        np.testing.assert_array_equal(m, [4, 12, 1])

        # Roundtrip
        time = pd.period_range('0001-01', '1000-12', freq='M')
        y, m = utils.calendardate_to_hydrodate(time.year, time.month,
                                               start_month=10)
        y, m = utils.hydrodate_to_calendardate(y, m, start_month=10)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        # April
        y, m = utils.hydrodate_to_calendardate(1, 1, start_month=4)
        assert (y, m) == (0, 4)
        y, m = utils.hydrodate_to_calendardate(1, 4, start_month=4)
        assert (y, m) == (0, 7)
        y, m = utils.hydrodate_to_calendardate(1, 9, start_month=4)
        assert (y, m) == (0, 12)
        y, m = utils.hydrodate_to_calendardate(1, 10, start_month=4)
        assert (y, m) == (1, 1)
        y, m = utils.hydrodate_to_calendardate(1, 12, start_month=4)
        assert (y, m) == (1, 3)

        y, m = utils.hydrodate_to_calendardate([1, 1, 1], [1, 4, 12],
                                               start_month=4)
        np.testing.assert_array_equal(y, [0, 0, 1])
        np.testing.assert_array_equal(m, [4, 7, 3])

        # Roundtrip
        time = pd.period_range('0001-01', '1000-12', freq='M')
        y, m = utils.calendardate_to_hydrodate(time.year, time.month,
                                               start_month=4)
        y, m = utils.hydrodate_to_calendardate(y, m, start_month=4)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        # January
        # hydro year/month corresponds to calendar year/month
        y, m = utils.hydrodate_to_calendardate(1, 1, start_month=1)
        assert (y, m) == (1, 1)
        y, m = utils.hydrodate_to_calendardate(1, 4, start_month=1)
        assert (y, m) == (1, 4)
        y, m = utils.hydrodate_to_calendardate(1, 12, start_month=1)
        assert (y, m) == (1, 12)

        y, m = utils.hydrodate_to_calendardate([1, 1, 1], [1, 4, 12],
                                               start_month=1)
        np.testing.assert_array_equal(y, [1, 1, 1])
        np.testing.assert_array_equal(m, [1, 4, 12])

        y, m = utils.calendardate_to_hydrodate(1, 1, start_month=1)
        assert (y, m) == (1, 1)
        y, m = utils.calendardate_to_hydrodate(1, 9, start_month=1)
        assert (y, m) == (1, 9)
        y, m = utils.calendardate_to_hydrodate(1, 10, start_month=1)
        assert (y, m) == (1, 10)

        y, m = utils.calendardate_to_hydrodate([1, 1, 1], [1, 9, 10],
                                               start_month=1)
        np.testing.assert_array_equal(y, [1, 1, 1])
        np.testing.assert_array_equal(m, [1, 9, 10])

        # Roundtrip
        time = pd.period_range('0001-01', '1000-12', freq='M')
        y, m = utils.calendardate_to_hydrodate(time.year, time.month,
                                               start_month=1)
        y, m = utils.hydrodate_to_calendardate(y, m, start_month=1)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)


    def test_rgi_meta(self):
        cfg.initialize()
        reg_names, subreg_names = utils.parse_rgi_meta(version='6')
        assert len(reg_names) == 20
        assert reg_names.loc[3].values[0] == 'Arctic Canada North'

    def test_adhikari_shape_factors(self):
        factors_rectangular = np.array([0.2, 0.313, 0.558, 0.790,
                                        0.884, 0.929, 0.954, 0.990, 1.])
        factors_parabolic = np.array([0.2, 0.251, 0.448, 0.653,
                                      0.748, 0.803, 0.839, 0.917, 1.])
        w = np.array([0.1, 0.5, 1, 2, 3, 4, 5, 10, 50])
        h = 0.5 * np.ones(w.shape)
        is_rect = h.copy()
        np.testing.assert_equal(shape_factor_adhikari(w, h, is_rect),
                                factors_rectangular)
        np.testing.assert_equal(shape_factor_adhikari(w, h, is_rect*0.),
                                factors_parabolic)

    def test_clips(self):

        a = np.arange(50) - 5
        assert_array_equal(np.clip(a, None, 10), utils.clip_max(a, 10))
        assert_array_equal(np.clip(a, 0, None), utils.clip_min(a, 0))
        assert_array_equal(np.clip(a, 0, 10), utils.clip_array(a, 0, 10))

        for a in [-1, 2, 12]:
            assert_array_equal(np.clip(a, 0, 10), utils.clip_scalar(a, 0, 10))
            assert_array_equal(np.clip(a, None, 10), utils.clip_max(a, 10))
            assert_array_equal(np.clip(a, 0, None), utils.clip_min(a, 0))

    def test_clips_performance(self):
        import timeit

        n = int(1e5)

        # Scalar
        s1 = 'import numpy as np'
        s2 = 'from oggm import utils'

        t1 = timeit.timeit('np.clip(1, 0, 10)', number=n, setup=s1)
        t2 = timeit.timeit('utils.clip_scalar(1, 0, 10)', number=n, setup=s2)
        assert t2 < t1

        t1 = timeit.timeit('np.clip(12, None, 10)', number=n, setup=s1)
        t2 = timeit.timeit('utils.clip_max(12, 10)', number=n, setup=s2)
        assert t2 < t1

        t1 = timeit.timeit('np.clip(12, 15, None)', number=n, setup=s1)
        t2 = timeit.timeit('utils.clip_min(12, 15)', number=n, setup=s2)
        assert t2 < t1

        # Array
        s1 = 'import numpy as np; a = np.arange(50) - 5'
        s2 = 'from oggm import utils; import numpy as np; a = np.arange(50)-5'

        # t1 = timeit.timeit('np.clip(a, 0, 10)', number=n, setup=s1)
        # t2 = timeit.timeit('utils.clip_array(a, 0, 10)', number=n, setup=s2)
        # This usually fails as advertised by numpy
        # (although with np 1.17 not)
        # assert t2 < t1

        t1 = timeit.timeit('np.clip(a, None, 10)', number=n, setup=s1)
        t2 = timeit.timeit('utils.clip_max(a, 10)', number=n, setup=s2)
        assert t2 < t1

        t1 = timeit.timeit('np.clip(a, 15, None)', number=n, setup=s1)
        t2 = timeit.timeit('utils.clip_min(a, 15)', number=n, setup=s2)
        assert t2 < t1

    def test_cook_rgidf(self):
        from oggm import workflow
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        working_dir = utils.get_temp_dir(dirname='test_cook_rgidf')
        cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=True)
        path = utils.get_demo_file('cgi2.shp')
        cgidf = gpd.read_file(path)
        rgidf = utils.cook_rgidf(cgidf, o1_region='13',
                                 assign_column_values={'Glc_Long': 'CenLon',
                                                       'Glc_Lati': 'CenLat'})
        rgidf['Area'] = cgidf.Glc_Area * 1e-6
        gdirs = workflow.init_glacier_directories(rgidf)
        df = utils.compile_glacier_statistics(gdirs)
        assert np.all(df.glacier_type == 'Glacier')
        assert np.all(df.rgi_region == '13')
        assert_allclose(df.cenlon, cgidf['Glc_Long'])
        assert_allclose(df.rgi_area_km2, cgidf['Glc_Area'] * 1e-6, rtol=1e-3)


class TestInitialize(unittest.TestCase):

    def setUp(self):
        cfg.initialize()
        self.homedir = os.path.expanduser('~')

    def test_env_var(self):
        with TempEnvironmentVariable(OGGM_USE_MULTIPROCESSING='1',
                                     OGGM_USE_MP_SPAWN=None):
            cfg.initialize()
            assert cfg.PARAMS['use_multiprocessing']
            assert cfg.PARAMS['mp_processes'] >= 1
            assert not cfg.PARAMS['use_mp_spawn']
        with TempEnvironmentVariable(OGGM_USE_MULTIPROCESSING='1',
                                     OGGM_USE_MP_SPAWN='1',
                                     SLURM_JOB_CPUS_PER_NODE='13'):
            cfg.initialize()
            assert cfg.PARAMS['use_multiprocessing']
            assert cfg.PARAMS['mp_processes'] == 13
            assert cfg.PARAMS['use_mp_spawn']
        with TempEnvironmentVariable(OGGM_USE_MULTIPROCESSING='0'):
            cfg.initialize()
            assert not cfg.PARAMS['use_multiprocessing']
            assert cfg.PARAMS['mp_processes'] >= 1

    def test_defaults(self):
        self.assertFalse(cfg.PATHS['working_dir'])

    def test_pathsetter(self):
        cfg.PATHS['working_dir'] = os.path.join('~', 'my_OGGM_wd')
        expected = os.path.join(self.homedir, 'my_OGGM_wd')
        self.assertEqual(cfg.PATHS['working_dir'], expected)


class TestWorkflowTools(unittest.TestCase):

    def setUp(self):
        cfg.initialize()
        self.testdir = os.path.join(get_test_dir(), 'tmp_gir')
        self.reset_dir()

    def tearDown(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    def reset_dir(self):
        utils.mkdir(self.testdir, reset=True)

    def test_leclercq_data(self):

        hef_file = utils.get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        cfg.PARAMS['use_intersects'] = False
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

        df = gdir.get_ref_length_data()
        assert df.name == 'Hintereis'
        assert len(df) == 105

    def test_glacier_characs(self):

        gdir = init_hef()

        df = utils.compile_glacier_statistics([gdir],
                                              apply_func=characs_apply_func,
                                              path=False)
        assert len(df) == 1

        assert 'glc_ext_num_perc' in df.columns
        assert np.all(np.isfinite(df.glc_ext_num_perc.values))

        df = df.iloc[0]
        np.testing.assert_allclose(df['dem_mean_elev'],
                                   df['flowline_mean_elev'], atol=5)



        df = utils.compile_climate_statistics([gdir], path=False,
                                              add_climate_period=1985)
        np.testing.assert_allclose(df['tstar_avg_prcp'],
                                   2853, atol=5)
        np.testing.assert_allclose(df['tstar_avg_prcpsol_max_elev'],
                                   2811, atol=5)
        np.testing.assert_allclose(df['1970-2000_avg_prcpsol_max_elev'],
                                   2811, atol=200)

    def test_demo_glacier_id(self):

        cfg.initialize()
        assert utils.demo_glacier_id('hef') == 'RGI60-11.00897'
        assert utils.demo_glacier_id('HeF') == 'RGI60-11.00897'
        assert utils.demo_glacier_id('HintereisFerner') == 'RGI60-11.00897'
        assert utils.demo_glacier_id('Mer de Glace') == 'RGI60-11.03643'
        assert utils.demo_glacier_id('RGI60-11.03643') == 'RGI60-11.03643'
        assert utils.demo_glacier_id('Mer Glace') is None

    def test_download_ref_tstars(self):

        cfg.initialize()
        cfg.PATHS['working_dir'] = self.testdir
        url = ('https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/'
               'oggm_v1.4/RGIV62/CRU/centerlines/qc3/pcp2.5/')
        workflow.download_ref_tstars(url)
        assert os.path.exists(os.path.join(cfg.PATHS['working_dir'],
                                           'ref_tstars.csv'))
        assert os.path.exists(os.path.join(cfg.PATHS['working_dir'],
                                           'ref_tstars_params.json'))


class TestStartFromTar(unittest.TestCase):

    def setUp(self):
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_tar_tools')

        # Init
        cfg.initialize()
        cfg.set_intersects_db(utils.get_demo_file('rgi_intersect_oetztal.shp'))

        # Read in the RGI file
        rgi_file = utils.get_demo_file('rgi_oetztal.shp')
        self.rgidf = gpd.read_file(rgi_file).sample(4)
        cfg.PATHS['dem_file'] = utils.get_demo_file('srtm_oetztal.tif')
        cfg.PATHS['working_dir'] = self.testdir
        self.clean_dir()

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        utils.mkdir(self.testdir, reset=True)

    @pytest.mark.slow
    def test_to_and_from_tar(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(self.rgidf)
        workflow.execute_entity_task(tasks.define_glacier_region, gdirs)

        # End - compress all
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs)

        # Test - reopen form tar
        gdirs = workflow.init_glacier_directories(self.rgidf, from_tar=True,
                                                  delete_tar=True)
        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert not os.path.exists(gdir.dir + '.tar.gz')
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

        workflow.execute_entity_task(utils.gdir_to_tar, gdirs)

        gdirs = workflow.init_glacier_directories(self.rgidf, from_tar=True)
        for gdir in gdirs:
            assert gdir.has_file('gridded_data')
            assert os.path.exists(gdir.dir + '.tar.gz')

    @pytest.mark.slow
    def test_to_and_from_basedir_tar(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(self.rgidf)
        workflow.execute_entity_task(tasks.define_glacier_region, gdirs)

        # End - compress all
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs)
        utils.base_dir_to_tar()

        # Test - reopen form tar
        gdirs = workflow.init_glacier_directories(self.rgidf, from_tar=True)

        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert not os.path.exists(gdir.dir + '.tar.gz')
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

        workflow.execute_entity_task(utils.gdir_to_tar, gdirs)
        utils.base_dir_to_tar()

        tar_dir = os.path.join(self.testdir, 'new_dir')
        shutil.copytree(os.path.join(cfg.PATHS['working_dir'],
                                     'per_glacier'), tar_dir)

        gdirs = workflow.init_glacier_directories(self.rgidf, from_tar=tar_dir)
        for gdir in gdirs:
            assert gdir.has_file('gridded_data')

    @pytest.mark.slow
    def test_to_and_from_tar_new_dir(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(self.rgidf)
        workflow.execute_entity_task(tasks.define_glacier_region, gdirs)

        # End - compress all
        base_dir = os.path.join(self.testdir, 'new_base_dir')
        paths = workflow.execute_entity_task(utils.gdir_to_tar, gdirs,
                                             base_dir=base_dir)

        # Test - reopen form tar after copy
        for p, gdir in zip(paths, gdirs):
            assert base_dir in p
            shutil.copyfile(p, os.path.normpath(gdir.dir) + '.tar.gz')

        gdirs = workflow.init_glacier_directories(self.rgidf,
                                                  from_tar=True,
                                                  delete_tar=True)
        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert not os.path.exists(gdir.dir + '.tar.gz')
            assert gdir.rgi_area_km2 > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    def test_to_and_from_tar_string(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(self.rgidf)
        workflow.execute_entity_task(tasks.define_glacier_region, gdirs)

        # End - compress all
        base_dir = os.path.join(self.testdir, 'new_base_dir')
        utils.mkdir(base_dir, reset=True)
        paths = workflow.execute_entity_task(utils.gdir_to_tar, gdirs,
                                             base_dir=base_dir, delete=False)

        # Test - reopen form tar after copy
        new_base_dir = os.path.join(self.testdir, 'newer_base_dir')
        utils.mkdir(new_base_dir, reset=True)
        for p, gdir in zip(paths, gdirs):
            assert base_dir in p
            new_gdir = utils.GlacierDirectory(gdir.rgi_id,
                                              base_dir=new_base_dir,
                                              from_tar=p)
            assert new_gdir.rgi_area_km2 == gdir.rgi_area_km2
            assert new_base_dir in new_gdir.base_dir


class TestStartFromOnlinePrepro(unittest.TestCase):

    def setUp(self):
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_prepro_tools')
        self.dldir = os.path.join(get_test_dir(), 'dl_cache')

        # Init
        cfg.initialize()
        cfg.PATHS['dl_cache_dir'] = self.dldir

        # Read in the RGI file
        rgi_file = utils.get_demo_file('rgi_oetztal.shp')
        self.rgidf = gpd.read_file(rgi_file)
        self.rgidf['RGIId'] = [rid.replace('RGI50', 'RGI60')
                               for rid in self.rgidf.RGIId]
        cfg.PATHS['working_dir'] = self.testdir
        self.clean_dir()

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)
        shutil.rmtree(self.dldir)

    def clean_dir(self):
        utils.mkdir(self.testdir, reset=True)
        utils.mkdir(self.dldir, reset=True)

    @mock.patch('oggm.utils._downloads.GDIR_L1L2_URL', TEST_GDIR_URL)
    @mock.patch('oggm.utils._downloads.GDIR_L3L5_URL', TEST_GDIR_URL)
    def test_start_from_level_1(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(self.rgidf.iloc[:2],
                                                  from_prepro_level=1,
                                                  prepro_rgi_version='61',
                                                  prepro_border=20)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    @mock.patch('oggm.utils._downloads.GDIR_L1L2_URL', TEST_GDIR_URL)
    @mock.patch('oggm.utils._downloads.GDIR_L3L5_URL', TEST_GDIR_URL)
    def test_start_from_level_1_str(self):

        # Go - initialize working directories
        entitites = self.rgidf.iloc[:2].RGIId
        cfg.PARAMS['border'] = 20
        gdirs = workflow.init_glacier_directories(entitites,
                                                  prepro_rgi_version='61',
                                                  from_prepro_level=1)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

        # One string
        cfg.PARAMS['border'] = 20
        gdirs = workflow.init_glacier_directories('RGI60-11.00897',
                                                  prepro_rgi_version='61',
                                                  from_prepro_level=1)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    @mock.patch('oggm.utils._downloads.GDIR_L1L2_URL', TEST_GDIR_URL)
    @mock.patch('oggm.utils._downloads.GDIR_L3L5_URL', TEST_GDIR_URL)
    def test_start_from_level_2(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(self.rgidf.iloc[:2],
                                                  from_prepro_level=2,
                                                  prepro_rgi_version='61',
                                                  prepro_border=20)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert gdir.has_file('climate_historical')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    @mock.patch('oggm.utils._downloads.GDIR_L1L2_URL', TEST_GDIR_URL)
    @mock.patch('oggm.utils._downloads.GDIR_L3L5_URL', TEST_GDIR_URL)
    def test_start_from_level_3(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(self.rgidf.iloc[:2],
                                                  from_prepro_level=3,
                                                  prepro_rgi_version='61',
                                                  prepro_border=20)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert gdir.has_file('gridded_data')
            assert gdir.has_file('climate_historical')
            assert gdir.has_file('climate_info')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0

        assert gdir.get_climate_info()
        assert gdir.read_pickle('climate_info')
        fls = gdir.read_pickle('inversion_flowlines')
        with pytest.raises(AttributeError):
            # This is going to raise as long as we keep the "old" prepro gdirs
            # remove this test after update.
            # At least we can read the old pickles
            fls[0].widths_m

        df = utils.compile_glacier_statistics(gdirs)
        assert 'dem_med_elev' in df

        df = utils.compile_climate_statistics(gdirs, add_climate_period=[1920,
                                                                         1960,
                                                                         2000])
        assert 'tstar_avg_temp_mean_elev' in df
        assert '1905-1935_avg_temp_mean_elev' in df

        workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    @mock.patch('oggm.utils._downloads.GDIR_L1L2_URL', TEST_GDIR_URL)
    @mock.patch('oggm.utils._downloads.GDIR_L3L5_URL', TEST_GDIR_URL)
    def test_start_from_level_4(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(self.rgidf.iloc[:2],
                                                  from_prepro_level=4,
                                                  prepro_rgi_version='61',
                                                  prepro_border=20)
        workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                                     nyears=10)

    @mock.patch('oggm.utils._downloads.GDIR_L1L2_URL', TEST_GDIR_URL)
    @mock.patch('oggm.utils._downloads.GDIR_L3L5_URL', TEST_GDIR_URL)
    def test_corrupted_file(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(['RGI60-11.00787'],
                                                  from_prepro_level=4,
                                                  prepro_rgi_version='61',
                                                  prepro_border=20)

        cfile = utils.get_prepro_gdir('61', 'RGI60-11.00787', 20, 4,
                                      base_url=TEST_GDIR_URL)
        assert 'cluster.klima.uni-bremen.de/~oggm/' in cfile

        # Replace with a dummy file
        os.remove(cfile)
        with open(cfile, 'w') as f:
            f.write('ups')

        # Since we already verified this will error
        with pytest.raises(tarfile.ReadError):
            gdirs = workflow.init_glacier_directories(['RGI60-11.00787'],
                                                      from_prepro_level=4,
                                                      prepro_rgi_version='61',
                                                      prepro_border=20)

        # This should retrigger a download and just work
        cfg.DL_VERIFIED.clear()
        gdirs = workflow.init_glacier_directories(['RGI60-11.00787'],
                                                  from_prepro_level=4,
                                                  prepro_rgi_version='61',
                                                  prepro_border=20)
        workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                                     nyears=10)


class TestPreproCLI(unittest.TestCase):

    def setUp(self):
        from _pytest.monkeypatch import MonkeyPatch
        self.monkeypatch = MonkeyPatch()
        self.testdir = os.path.join(get_test_dir(), 'tmp_prepro_levs')
        self.reset_dir()

    def tearDown(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    def reset_dir(self):
        utils.mkdir(self.testdir, reset=True)

    def _read_shp(self):
        # Read in the RGI file
        inter = gpd.read_file(utils.get_demo_file('rgi_intersect_oetztal.shp'))
        rgidf = gpd.read_file(utils.get_demo_file('rgi_oetztal.shp'))

        rgidf['RGIId'] = [rid.replace('RGI50', 'RGI60') for rid in rgidf.RGIId]
        inter['RGIId_1'] = [rid.replace('RGI50', 'RGI60')
                            for rid in inter.RGIId_1]
        inter['RGIId_2'] = [rid.replace('RGI50', 'RGI60')
                            for rid in inter.RGIId_2]
        return inter, rgidf

    def test_parse_args(self):

        from oggm.cli import prepro_levels

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
                                           '--map-border', '160'])

        assert 'working_dir' in kwargs
        assert 'output_folder' in kwargs
        assert kwargs['rgi_version'] is None
        assert kwargs['params_file'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
                                           '--map-border', '160',
                                           '--start-level', '2',
                                           '--start-base-url', 'http://foo',
                                           '--ref-tstars-base-url', 'http://bar',
                                           ])

        assert 'working_dir' in kwargs
        assert 'output_folder' in kwargs
        assert kwargs['rgi_version'] is None
        assert kwargs['params_file'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160
        assert kwargs['start_level'] == 2
        assert kwargs['start_base_url'] == 'http://foo'
        assert kwargs['ref_tstars_base_url'] == 'http://bar'

        with pytest.raises(InvalidParamsError):
            prepro_levels.parse_args([])

        with pytest.raises(InvalidParamsError):
            prepro_levels.parse_args(['--rgi-reg', '1'])

        with pytest.raises(InvalidParamsError):
            prepro_levels.parse_args(['--map-border', '160'])

        with TempEnvironmentVariable(OGGM_RGI_REG='1', OGGM_MAP_BORDER='160'):

            kwargs = prepro_levels.parse_args([])

            assert 'working_dir' in kwargs
            assert 'output_folder' in kwargs
            assert kwargs['rgi_version'] is None
            assert kwargs['rgi_reg'] == '01'
            assert kwargs['border'] == 160
            assert not kwargs['is_test']

        with pytest.raises(InvalidParamsError):
            prepro_levels.parse_args([])

        kwargs = prepro_levels.parse_args(['--demo',
                                           '--map-border', '160',
                                           '--output', 'local/out',
                                           '--working-dir', 'local/work',
                                           '--params-file', 'dir/params.cfg',
                                           ])

        assert 'working_dir' in kwargs
        assert 'output_folder' in kwargs
        assert 'local' in kwargs['working_dir']
        assert 'local' in kwargs['output_folder']
        assert 'dir/params.cfg' in kwargs['params_file']
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '00'
        assert kwargs['dem_source'] == ''
        assert kwargs['border'] == 160
        assert not kwargs['is_test']
        assert kwargs['demo']
        assert not kwargs['disable_mp']
        assert kwargs['max_level'] == 5

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
                                           '--map-border', '160',
                                           '--output', 'local/out',
                                           '--working-dir', 'local/work',
                                           '--dem-source', 'ALL',
                                           ])

        assert 'working_dir' in kwargs
        assert 'output_folder' in kwargs
        assert 'local' in kwargs['working_dir']
        assert 'local' in kwargs['output_folder']
        assert kwargs['dem_source'] == 'ALL'
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160
        assert not kwargs['is_test']
        assert not kwargs['elev_bands']
        assert not kwargs['match_geodetic_mb']
        assert not kwargs['centerlines_only']

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
                                           '--map-border', '160',
                                           '--output', 'local/out',
                                           '--working-dir', 'local/work',
                                           '--test',
                                           '--test-ids', 'RGI60-19.00134',
                                                         'RGI60-19.00156',
                                           ])

        assert 'local' in kwargs['working_dir']
        assert 'local' in kwargs['output_folder']
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160
        assert kwargs['is_test']
        assert kwargs['test_ids']
        assert len(kwargs['test_ids']) == 2
        assert kwargs['test_ids'][0] == 'RGI60-19.00134'

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
                                           '--map-border', '160',
                                           '--output', '/local/out',
                                           '--elev-bands',
                                           '--centerlines-only',
                                           '--match-geodetic-mb', 'zemp',
                                           '--working-dir', '/local/work',
                                           ])

        assert 'tests' not in kwargs['working_dir']
        assert 'tests' not in kwargs['output_folder']
        assert 'local' in kwargs['working_dir']
        assert 'local' in kwargs['output_folder']
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160
        assert kwargs['elev_bands']
        assert kwargs['match_geodetic_mb'] == 'zemp'
        assert kwargs['centerlines_only']

        with TempEnvironmentVariable(OGGM_RGI_REG='12',
                                     OGGM_MAP_BORDER='120',
                                     OGGM_OUTDIR='local/out',
                                     OGGM_WORKDIR='local/work',
                                     ):

            kwargs = prepro_levels.parse_args([])

            assert 'local' in kwargs['working_dir']
            assert 'local' in kwargs['output_folder']
            assert kwargs['rgi_version'] is None
            assert kwargs['rgi_reg'] == '12'
            assert kwargs['border'] == 120

        with TempEnvironmentVariable(OGGM_RGI_REG='12',
                                     OGGM_MAP_BORDER='120',
                                     OGGM_OUTDIR='/local/out',
                                     OGGM_WORKDIR='/local/work',
                                     ):

            kwargs = prepro_levels.parse_args([])

            assert 'local' in kwargs['working_dir']
            assert 'local' in kwargs['output_folder']
            assert kwargs['rgi_version'] is None
            assert kwargs['rgi_reg'] == '12'
            assert kwargs['border'] == 120

    @pytest.mark.slow
    def test_full_run(self):

        from oggm.cli.prepro_levels import run_prepro_levels

        inter, rgidf = self._read_shp()

        wdir = os.path.join(self.testdir, 'wd')
        utils.mkdir(wdir)
        odir = os.path.join(self.testdir, 'my_levs')
        topof = utils.get_demo_file('srtm_oetztal.tif')
        np.random.seed(0)
        run_prepro_levels(rgi_version='61', rgi_reg='11', border=20,
                          output_folder=odir, working_dir=wdir, is_test=True,
                          test_rgidf=rgidf, test_intersects_file=inter,
                          test_topofile=topof, match_geodetic_mb='hugonnet')

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L3', 'summary',
                                      'climate_statistics_11.csv'))
        assert '1980-2010_avg_prcp' in df
        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L0', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'glacier_type' in df

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L1', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'dem_source' in df

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L2', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'main_flowline_length' in df
        assert os.path.isfile(os.path.join(odir, 'RGI61', 'b_020', 'L2',
                                           'summary', 'centerlines_11.tar.gz'))

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L3', 'summary',
                                      'glacier_statistics_11.csv'), index_col=0)
        assert 'inv_volume_km3' in df
        assert 'mb_bias_before_geodetic_corr' in df

        dfm = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L3', 'summary',
                                       'fixed_geometry_mass_balance_11.csv'),
                          index_col=0)
        dfm = dfm.dropna(axis=0, how='all').dropna(axis=1, how='all')

        odf = pd.DataFrame(dfm.loc[2000:2019].mean(), columns=['SMB'])
        odf['AREA'] = df.rgi_area_km2
        smb_oggm = np.average(odf['SMB'], weights=odf['AREA'])

        dfh = 'table_hugonnet_regions_10yr_20yr_ar6period.csv'
        dfh = pd.read_csv(utils.get_demo_file(dfh))
        dfh = dfh.loc[dfh.period == '2000-01-01_2020-01-01'].set_index('reg')
        smb_ref = dfh.loc[11, 'dmdtda']
        np.testing.assert_allclose(smb_oggm, smb_ref)

        assert os.path.isfile(os.path.join(odir, 'RGI61', 'b_020',
                                           'package_versions.txt'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L1'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L2'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L3'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L4'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L5'))

        # See if we can start from all levs
        from oggm import tasks
        from oggm.core.flowline import FlowlineModel, FileModel
        cfg.PARAMS['continue_on_error'] = False
        rid = df.index[0]
        entity = rgidf.loc[rgidf.RGIId == rid].iloc[0]

        # L1
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L1',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        tasks.glacier_masks(gdir)
        with pytest.raises(FileNotFoundError):
            tasks.init_present_time_glacier(gdir)

        # L2
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L2',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        assert gdir.has_file('inversion_flowlines')
        with pytest.raises(FileNotFoundError):
            tasks.init_present_time_glacier(gdir)

        # L3
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L3',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        model = tasks.run_random_climate(gdir, nyears=10)
        assert isinstance(model, FlowlineModel)

        # L4
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L4',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        model = tasks.run_random_climate(gdir, nyears=10)
        assert isinstance(model, FlowlineModel)
        with xr.open_dataset(gdir.get_filepath('model_diagnostics')) as ds:
            # cannot be the same after tuning
            assert ds.glen_a != cfg.PARAMS['glen_a']
        with pytest.raises(FileNotFoundError):
            tasks.init_present_time_glacier(gdir)

        # L5
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L5',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        model = FileModel(gdir.get_filepath('model_geometry',
                                            filesuffix='_historical'))
        assert model.y0 == 2004
        assert model.last_yr == 2015
        with pytest.raises(FileNotFoundError):
            # We can't create this because the glacier dir is mini
            tasks.init_present_time_glacier(gdir)

        # Extended file
        fp = os.path.join(odir, 'RGI61', 'b_020', 'L5', 'summary',
                          'historical_run_output_extended_11.nc')
        with xr.open_dataset(fp) as ods:

            ref = ods.volume
            new = ods.volume_fixed_geom
            np.testing.assert_allclose(new.isel(time=-1),
                                       ref.isel(time=-1),
                                       rtol=0.02)

            vn = 'volume'
            np.testing.assert_allclose(ods[vn].sel(time=1990),
                                       ods[vn].sel(time=2015),
                                       rtol=0.3)

            for vn in ['calving', 'volume_bsl', 'volume_bwl']:
                np.testing.assert_allclose(ods[vn].sel(time=1990), 0)

    @pytest.mark.slow
    def test_elev_bands_run(self):

        from oggm.cli.prepro_levels import run_prepro_levels

        # Read in the RGI file
        inter, rgidf = self._read_shp()

        wdir = os.path.join(self.testdir, 'wd')
        utils.mkdir(wdir)
        odir = os.path.join(self.testdir, 'my_levs')
        topof = utils.get_demo_file('srtm_oetztal.tif')
        np.random.seed(0)
        run_prepro_levels(rgi_version='61', rgi_reg='11', border=20,
                          output_folder=odir, working_dir=wdir, is_test=True,
                          test_rgidf=rgidf, test_intersects_file=inter,
                          test_topofile=topof, elev_bands=True)

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L0', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'glacier_type' in df

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L1', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'dem_source' in df

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L2', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'main_flowline_length' in df

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L3', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'inv_volume_km3' in df
        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L3', 'summary',
                                      'climate_statistics_11.csv'))
        assert '1980-2010_avg_prcp' in df

        assert os.path.isfile(os.path.join(odir, 'RGI61', 'b_020',
                                           'package_versions.txt'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L1'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L2'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L3'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L4'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L5'))

        # See if we can start from L3 and L4
        from oggm import tasks
        from oggm.core.flowline import FlowlineModel, FileModel
        cfg.PARAMS['continue_on_error'] = False
        rid = df.rgi_id.iloc[0]
        entity = rgidf.loc[rgidf.RGIId == rid].iloc[0]

        # L3
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L3',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        model = tasks.run_random_climate(gdir, nyears=10)
        assert isinstance(model, FlowlineModel)

        # L4
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L4',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        model = tasks.run_random_climate(gdir, nyears=10)
        assert isinstance(model, FlowlineModel)
        with pytest.raises(FileNotFoundError):
            # We can't create this because the glacier dir is mini
            tasks.init_present_time_glacier(gdir)

        # L5
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L5',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        model = FileModel(gdir.get_filepath('model_geometry',
                                            filesuffix='_historical'))
        assert model.y0 == 2004
        assert model.last_yr == 2015
        with pytest.raises(FileNotFoundError):
            # We can't create this because the glacier dir is mini
            tasks.init_present_time_glacier(gdir)

    def test_start_from_prepro(self):

        from oggm.cli.prepro_levels import run_prepro_levels

        base_url = ('https://cluster.klima.uni-bremen.de/~oggm/test_gdirs/'
                    'oggm_v1.1/')

        # Read in the RGI file
        inter, rgidf = self._read_shp()
        wdir = os.path.join(self.testdir, 'wd')
        utils.mkdir(wdir)
        odir = os.path.join(self.testdir, 'my_levs')
        np.random.seed(0)
        run_prepro_levels(rgi_version='61', rgi_reg='11', border=20,
                          output_folder=odir, working_dir=wdir, is_test=True,
                          test_rgidf=rgidf, test_intersects_file=inter,
                          start_level=1, start_base_url=base_url,
                          max_level=5)

        assert not os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L1'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L2'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L3'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L4'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L5'))

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L2', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'main_flowline_length' in df

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L3', 'summary',
                                      'glacier_statistics_11.csv'), index_col=0)
        assert 'inv_volume_km3' in df

        with pytest.raises(InvalidParamsError):
            run_prepro_levels(rgi_version=None, rgi_reg='11', border=20,
                              output_folder=odir, working_dir=wdir, is_test=True,
                              test_rgidf=rgidf, test_intersects_file=inter,
                              start_level=2, max_level=4)

    def test_source_run(self):

        self.monkeypatch.setattr(oggm.utils, 'DEM_SOURCES', ['USER'])

        from oggm.cli.prepro_levels import run_prepro_levels

        # Read in the RGI file
        inter, rgidf = self._read_shp()
        rgidf = rgidf.iloc[:4]

        wdir = os.path.join(self.testdir, 'wd')
        utils.mkdir(wdir)
        odir = os.path.join(self.testdir, 'my_levs')
        topof = utils.get_demo_file('srtm_oetztal.tif')
        np.random.seed(0)
        run_prepro_levels(rgi_version='61', rgi_reg='11', border=20,
                          output_folder=odir, working_dir=wdir, is_test=True,
                          test_rgidf=rgidf, test_intersects_file=inter,
                          test_topofile=topof, dem_source='ALL')

        rid = rgidf.iloc[0].RGIId
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L1',
                            rid[:8], rid[:8] + '.00.tar')
        assert os.path.isfile(tarf)

        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L1',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)

        entity = rgidf.iloc[0]
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        assert os.path.isfile(os.path.join(gdir.dir, 'USER', 'dem.tif'))


class TestBenchmarkCLI(unittest.TestCase):

    def setUp(self):
        self.testdir = os.path.join(get_test_dir(), 'tmp_benchmarks')
        self.reset_dir()

    def tearDown(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    def reset_dir(self):
        utils.mkdir(self.testdir, reset=True)

    def test_parse_args(self):

        from oggm.cli import benchmark

        kwargs = benchmark.parse_args(['--rgi-reg', '1',
                                       '--map-border', '160'])

        assert 'working_dir' in kwargs
        assert 'output_folder' in kwargs
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160

        with pytest.raises(InvalidParamsError):
            benchmark.parse_args([])

        with pytest.raises(InvalidParamsError):
            benchmark.parse_args(['--rgi-reg', '1'])

        with pytest.raises(InvalidParamsError):
            benchmark.parse_args(['--map-border', '160'])

        with TempEnvironmentVariable(OGGM_RGI_REG='1', OGGM_MAP_BORDER='160'):

            kwargs = benchmark.parse_args([])

            assert 'working_dir' in kwargs
            assert 'output_folder' in kwargs
            assert kwargs['rgi_version'] is None
            assert kwargs['rgi_reg'] == '01'
            assert kwargs['border'] == 160
            assert not kwargs['is_test']

        with pytest.raises(InvalidParamsError):
            benchmark.parse_args([])

        kwargs = benchmark.parse_args(['--rgi-reg', '1',
                                       '--map-border', '160',
                                       '--output', 'local/out',
                                       '--working-dir', 'local/work',
                                       ])

        assert 'working_dir' in kwargs
        assert 'output_folder' in kwargs
        assert 'local' in kwargs['working_dir']
        assert 'local' in kwargs['output_folder']
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160
        assert not kwargs['is_test']

        kwargs = benchmark.parse_args(['--rgi-reg', '1',
                                       '--map-border', '160',
                                       '--output', 'local/out',
                                       '--working-dir', 'local/work',
                                       '--test',
                                       ])

        assert 'local' in kwargs['working_dir']
        assert 'local' in kwargs['output_folder']
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160
        assert kwargs['is_test']

        kwargs = benchmark.parse_args(['--rgi-reg', '1',
                                       '--map-border', '160',
                                       '--output', '/local/out',
                                       '--working-dir', '/local/work',
                                       ])

        assert 'tests' not in kwargs['working_dir']
        assert 'tests' not in kwargs['output_folder']
        assert 'local' in kwargs['working_dir']
        assert 'local' in kwargs['output_folder']
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160

        with TempEnvironmentVariable(OGGM_RGI_REG='12',
                                     OGGM_MAP_BORDER='120',
                                     OGGM_OUTDIR='local/out',
                                     OGGM_WORKDIR='local/work',
                                     ):

            kwargs = benchmark.parse_args([])

            assert 'local' in kwargs['working_dir']
            assert 'local' in kwargs['output_folder']
            assert kwargs['rgi_version'] is None
            assert kwargs['rgi_reg'] == '12'
            assert kwargs['border'] == 120

        with TempEnvironmentVariable(OGGM_RGI_REG='12',
                                     OGGM_MAP_BORDER='120',
                                     OGGM_OUTDIR='/local/out',
                                     OGGM_WORKDIR='/local/work',
                                     ):

            kwargs = benchmark.parse_args([])

            assert 'local' in kwargs['working_dir']
            assert 'local' in kwargs['output_folder']
            assert kwargs['rgi_version'] is None
            assert kwargs['rgi_reg'] == '12'
            assert kwargs['border'] == 120

    @pytest.mark.slow
    def test_full_run(self):

        from oggm.cli.benchmark import run_benchmark

        # Read in the RGI file
        inter = gpd.read_file(utils.get_demo_file('rgi_intersect_oetztal.shp'))
        rgidf = gpd.read_file(utils.get_demo_file('rgi_oetztal.shp'))

        rgidf['RGIId'] = [rid.replace('RGI50', 'RGI60') for rid in rgidf.RGIId]
        inter['RGIId_1'] = [rid.replace('RGI50', 'RGI60')
                            for rid in inter.RGIId_1]
        inter['RGIId_2'] = [rid.replace('RGI50', 'RGI60')
                            for rid in inter.RGIId_2]

        cru_file = utils.get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')

        wdir = os.path.join(self.testdir, 'wd')
        utils.mkdir(wdir)
        odir = os.path.join(self.testdir, 'my_levs')
        topof = utils.get_demo_file('srtm_oetztal.tif')
        np.random.seed(0)
        run_benchmark(rgi_version=None, rgi_reg='11', border=80,
                      output_folder=odir, working_dir=wdir, is_test=True,
                      test_rgidf=rgidf, test_intersects_file=inter,
                      test_topofile=topof)

        df = pd.read_csv(os.path.join(odir, 'benchmarks_b080.csv'),
                         index_col=0)
        assert len(df) > 15


def touch(path):
    """Equivalent to linux's touch"""
    with open(path, 'a'):
        os.utime(path, None)
    return path


def make_fake_zipdir(dir_path, fakefile=None, base_dir=None,
                     archiv='zip', extension='.zip'):
    """Creates a directory with a file in it if asked to, then compresses it"""
    utils.mkdir(dir_path)
    if fakefile:
        touch(os.path.join(dir_path, fakefile))
    shutil.make_archive(dir_path, archiv, dir_path, base_dir)
    return dir_path + extension


class FakeDownloadManager():
    """We mess around with oggm internals, so the last we can do is to try
    to keep things clean after the tests."""

    def __init__(self, func_name, new_func):
        self.func_name = func_name
        self.new_func = new_func
        self._store = getattr(_downloads, func_name)

    def __enter__(self):
        self._store = getattr(_downloads, self.func_name)
        setattr(_downloads, self.func_name, self.new_func)

    def __exit__(self, *args):
        setattr(_downloads, self.func_name, self._store)


class TestFakeDownloads(unittest.TestCase):

    def setUp(self):
        self.dldir = os.path.join(get_test_dir(), 'tmp_download')
        utils.mkdir(self.dldir)

        # Get the path to the file before we mess around
        self.dem3_testfile = utils.get_demo_file('T10.zip')

        cfg.initialize()
        cfg.PATHS['dl_cache_dir'] = os.path.join(self.dldir, 'dl_cache')
        cfg.PATHS['working_dir'] = os.path.join(self.dldir, 'wd')
        cfg.PATHS['tmp_dir'] = os.path.join(self.dldir, 'extract')
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_test')
        self.reset_dir()

    def tearDown(self):
        if os.path.exists(self.dldir):
            shutil.rmtree(self.dldir)

    def reset_dir(self):
        if os.path.exists(self.dldir):
            shutil.rmtree(self.dldir)
        utils.mkdir(self.dldir)
        utils.mkdir(cfg.PATHS['dl_cache_dir'])
        utils.mkdir(cfg.PATHS['working_dir'])
        utils.mkdir(cfg.PATHS['tmp_dir'])
        utils.mkdir(cfg.PATHS['rgi_dir'])

    def prepare_verify_test(self, valid_size=True, valid_crc32=True,
                            reset_dl_dict=True):
        self.reset_dir()
        cfg.PARAMS['dl_verify'] = True

        if reset_dl_dict:
            cfg.DL_VERIFIED.clear()

        tgt_path = os.path.join(cfg.PATHS['dl_cache_dir'], 'test.com',
                                'test.txt')

        file_size = 1024
        file_data = os.urandom(file_size)
        file_sha256 = hashlib.sha256()
        file_sha256.update(file_data)

        utils.mkdir(os.path.dirname(tgt_path))
        with open(tgt_path, 'wb') as f:
            f.write(file_data)

        if not valid_size:
            file_size += 1
        if not valid_crc32:
            file_sha256.update(b'1234ABCD')

        file_sha256 = file_sha256.digest()

        data = utils.get_dl_verify_data('cluster.klima.uni-bremen.de')
        s = pd.Series({'size': file_size, 'sha256': file_sha256},
                      name='test.txt')
        data = data.append(s)
        cfg.DATA['dl_verify_data_test.com'] = data

        return 'https://test.com/test.txt'

    def test_dl_verify(self):
        def fake_down(dl_func, cache_path):
            assert False

        with FakeDownloadManager('_call_dl_func', fake_down):
            url = self.prepare_verify_test(True, True)
            utils.oggm_urlretrieve(url)

            url = self.prepare_verify_test(False, True)
            with self.assertRaises(DownloadVerificationFailedException):
                utils.oggm_urlretrieve(url)

            url = self.prepare_verify_test(True, False)
            with self.assertRaises(DownloadVerificationFailedException):
                utils.oggm_urlretrieve(url)

            url = self.prepare_verify_test(False, False)
            with self.assertRaises(DownloadVerificationFailedException):
                utils.oggm_urlretrieve(url)

    def test_github_no_internet(self):
        self.reset_dir()
        cache_dir = cfg.CACHE_DIR
        try:
            cfg.CACHE_DIR = os.path.join(self.dldir, 'cache')

            def fake_down(dl_func, cache_path):
                # This should never be called, if it still is assert
                assert False
            with FakeDownloadManager('_call_dl_func', fake_down):
                with self.assertRaises(utils.NoInternetException):
                    tmp = cfg.PARAMS['has_internet']
                    cfg.PARAMS['has_internet'] = False
                    try:
                        utils.download_oggm_files()
                    finally:
                        cfg.PARAMS['has_internet'] = tmp
        finally:
            cfg.CACHE_DIR = cache_dir

    def test_rgi(self):

        # Make a fake RGI file
        rgi_dir = os.path.join(self.dldir, 'rgi50')
        utils.mkdir(rgi_dir)
        make_fake_zipdir(os.path.join(rgi_dir, '01_rgi50_Region'),
                         fakefile='test.txt')
        rgi_f = make_fake_zipdir(rgi_dir, fakefile='000_rgi50_manifest.txt')

        def down_check(url, *args, **kwargs):
            expected = 'http://www.glims.org/RGI/rgi50_files/rgi50.zip'
            self.assertEqual(url, expected)
            return rgi_f

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            rgi = utils.get_rgi_dir(version='5')

        assert os.path.isdir(rgi)
        assert os.path.exists(os.path.join(rgi, '000_rgi50_manifest.txt'))
        assert os.path.exists(os.path.join(rgi, '01_rgi50_Region', 'test.txt'))

        # Make a fake RGI file
        rgi_dir = os.path.join(self.dldir, 'rgi60')
        utils.mkdir(rgi_dir)
        make_fake_zipdir(os.path.join(rgi_dir, '01_rgi60_Region'),
                         fakefile='01_rgi60_Region.shp')
        rgi_f = make_fake_zipdir(rgi_dir, fakefile='000_rgi60_manifest.txt')

        def down_check(url, *args, **kwargs):
            expected = 'http://www.glims.org/RGI/rgi60_files/00_rgi60.zip'
            self.assertEqual(url, expected)
            return rgi_f

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            rgi = utils.get_rgi_dir(version='6')

        assert os.path.isdir(rgi)
        assert os.path.exists(os.path.join(rgi, '000_rgi60_manifest.txt'))
        assert os.path.exists(os.path.join(rgi, '01_rgi60_Region',
                                           '01_rgi60_Region.shp'))

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            rgi_f = utils.get_rgi_region_file('01', version='6')

        assert os.path.exists(rgi_f)
        assert '01_rgi60_Region.shp' in rgi_f

    def test_rgi_intersects(self):

        # Make a fake RGI file
        rgi_dir = os.path.join(self.dldir, 'RGI_V50_Intersects')
        utils.mkdir(rgi_dir)
        make_fake_zipdir(os.path.join(rgi_dir),
                         fakefile='Intersects_OGGM_Manifest.txt')
        make_fake_zipdir(os.path.join(rgi_dir, '11_rgi50_CentralEurope'),
                         fakefile='intersects_11_rgi50_CentralEurope.shp')
        make_fake_zipdir(os.path.join(rgi_dir, '00_rgi50_AllRegs'),
                         fakefile='intersects_rgi50_AllRegs.shp')
        rgi_f = make_fake_zipdir(rgi_dir)

        def down_check(url, *args, **kwargs):
            expected = ('https://cluster.klima.uni-bremen.de/data/rgi/' +
                        'RGI_V50_Intersects.zip')
            self.assertEqual(url, expected)
            return rgi_f

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            rgi = utils.get_rgi_intersects_dir(version='5')
            utils.get_rgi_intersects_region_file('11', version='5')
            utils.get_rgi_intersects_region_file('00', version='5')

        assert os.path.isdir(rgi)
        assert os.path.exists(os.path.join(rgi,
                                           'Intersects_OGGM_Manifest.txt'))

        # Make a fake RGI file
        rgi_dir = os.path.join(self.dldir, 'RGI_V60_Intersects')
        utils.mkdir(rgi_dir)
        make_fake_zipdir(os.path.join(rgi_dir),
                         fakefile='Intersects_OGGM_Manifest.txt')
        rgi_f = make_fake_zipdir(rgi_dir)

        def down_check(url, *args, **kwargs):
            expected = ('https://cluster.klima.uni-bremen.de/data/rgi/' +
                        'RGI_V60_Intersects.zip')
            self.assertEqual(url, expected)
            return rgi_f

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            rgi = utils.get_rgi_intersects_dir(version='6')

        assert os.path.isdir(rgi)
        assert os.path.exists(os.path.join(rgi,
                                           'Intersects_OGGM_Manifest.txt'))

    def test_srtm(self):

        # Make a fake topo file
        tf = make_fake_zipdir(os.path.join(self.dldir, 'srtm_39_03'),
                              fakefile='srtm_39_03.tif')

        def down_check(url, *args, **kwargs):
            expected = ('http://srtm.csi.cgiar.org/wp-content/uploads/files/'
                        'srtm_5x5/TIFF/srtm_39_03.zip')
            self.assertEqual(url, expected)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([11.3, 11.3], [47.1, 47.1],
                                             source='SRTM')

        assert os.path.exists(of[0])
        assert source == 'SRTM'

    def test_nasadem(self):

        # Make a fake topo file
        tf = make_fake_zipdir(os.path.join(self.dldir, 'NASADEM_HGT_n47e011'),
                              fakefile='n47e011.hgt')

        def down_check(url, *args, **kwargs):
            expected = ('https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/'
                        '2000.02.11/NASADEM_HGT_n47e011.zip')
            self.assertEqual(url, expected)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([11.3, 11.3], [47.1, 47.1],
                                             source='NASADEM')

        assert os.path.exists(of[0])
        assert source == 'NASADEM'

    def test_dem3(self):

        def down_check(url, *args, **kwargs):
            expected = 'http://viewfinderpanoramas.org/dem3/T10.zip'
            self.assertEqual(url, expected)
            return self.dem3_testfile

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [76.8, 76.8],
                                             source='DEM3')

        assert os.path.exists(of[0])
        assert source == 'DEM3'

        def down_check(url, *args, **kwargs):
            expected = ('https://cluster.klima.uni-bremen.de/~oggm/dem/'
                        'DEM3_MERGED/ISL.tif')
            self.assertEqual(url, expected)
            return self.dem3_testfile

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file(-22.26, 66.16, source='DEM3')

        assert os.path.exists(of[0])
        assert source == 'DEM3'

    def test_mapzen(self):

        # Make a fake topo file
        tf = touch(os.path.join(self.dldir, 'file.tif'))

        def down_check(url, *args, **kwargs):
            expected = ('http://s3.amazonaws.com/elevation-tiles-prod/'
                        'geotiff/10/170/160.tif')
            self.assertEqual(url, expected)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [76.8, 76.8],
                                             source='MAPZEN', dx_meter=100)

        assert os.path.exists(of[0])
        assert source == 'MAPZEN'

    def test_aw3d30(self):

        # Make a fake topo file
        deep_path = os.path.join(self.dldir, 'N049W006', 'N049W006')
        utils.mkdir(deep_path)
        upper_path = os.path.dirname(deep_path)
        fakefile = os.path.join(deep_path, 'N049W006_AVE_DSM.tif')
        tf = make_fake_zipdir(upper_path, fakefile=fakefile,
                              archiv='gztar', extension='.tar.gz')

        def down_check(url, *args, **kwargs):
            expected = ('ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/' +
                        'release_v1804/N045W010/N049W006.tar.gz')
            self.assertEqual(expected, url)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([-5.3, -5.2], [49.5, 49.6],
                                             source='AW3D30')

        assert os.path.exists(of[0])
        assert source == 'AW3D30'

    def test_copdem(self):

        # Make a fake topo file
        deep_path = os.path.join(self.dldir,
                                 'DEM1_SAR_DGE_90_20110517T170701_20140817T170'
                                 '857_ADS_000000_4723.DEM',
                                 'Copernicus_DSM_30_N46_00_E010_00', 'DEM')
        utils.mkdir(deep_path)
        upper_path = os.path.dirname(os.path.dirname(deep_path))
        fakefile = os.path.join(deep_path,
                                'Copernicus_DSM_30_N46_00_E010_00_DEM.tif')
        tf = make_fake_zipdir(upper_path, fakefile=fakefile,
                              base_dir='Copernicus_DSM_30_N46_00_E010_00',
                              archiv='tar', extension='.tar')

        def down_check(url, *args, **kwargs):
            expected = ('ftps://cdsdata.copernicus.eu:990/datasets/'
                        'COP-DEM_GLO-90-DGED/2019_1/'
                        'DEM1_SAR_DGE_90_20110517T170701_20140817T170857_ADS_'
                        '000000_4723.DEM.tar')
            self.assertEqual(expected, url)
            return tf

        with FakeDownloadManager('download_with_authentication', down_check):
            of, source = utils.get_topo_file([10.5, 10.8], [46.6, 46.8],
                                             source='COPDEM')

        assert os.path.exists(of[0])
        assert source == 'COPDEM'

    def test_ramp(self):

        def down_check(url, *args, **kwargs):
            expected = 'AntarcticDEM_wgs84.tif'
            self.assertEqual(url, expected)
            return 'yo'

        with FakeDownloadManager('_download_topo_file_from_cluster',
                                 down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [-88, -88],
                                             source='RAMP')

        assert of[0] == 'yo'
        assert source == 'RAMP'

    def test_rema(self):

        # Make a fake topo file
        tf = touch(os.path.join(self.dldir, 'file.tif'))

        def down_check(url, *args, **kwargs):
            expected = ('https://cluster.klima.uni-bremen.de/~oggm/dem/'
                        'REMA_100m_v1.1/'
                        '40_10_100m_v1.1/40_10_100m_v1.1_reg_dem.tif')
            self.assertEqual(expected, url)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file(-65., -69., source='REMA')

        assert os.path.exists(of[0])
        assert source == 'REMA'

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([-65.1, -65.],
                                             [-69.1, -69.],
                                             source='REMA')

        assert os.path.exists(of[0])
        assert source == 'REMA'

    def test_alaska(self):

        # Make a fake topo file
        tf = touch(os.path.join(self.dldir, 'file.tif'))

        def down_check(url, *args, **kwargs):
            expected = ('https://cluster.klima.uni-bremen.de/~oggm/dem/'
                        'Alaska_albers_V3/008_004_Alaska_albers_V3/'
                        '008_004_Alaska_albers_V3.tif')
            self.assertEqual(expected, url)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([-140.428, -140.428],
                                             [60.177, 60.177],
                                             source='ALASKA')

        assert os.path.exists(of[0])
        assert source == 'ALASKA'

    def test_arcticdem(self):

        # Make a fake topo file
        tf = touch(os.path.join(self.dldir, 'file.tif'))

        def down_check(url, *args, **kwargs):
            expected = ('https://cluster.klima.uni-bremen.de/~oggm/dem/'
                        'ArcticDEM_100m_v3.0/'
                        '14_52_100m_v3.0/14_52_100m_v3.0_reg_dem.tif')
            self.assertEqual(expected, url)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file(-21.93, 64.13, source='ARCTICDEM')

        assert os.path.exists(of[0])
        assert source == 'ARCTICDEM'

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([-21.93, -21.92],
                                             [64.13, 64.14],
                                             source='ARCTICDEM')

        assert os.path.exists(of[0])
        assert source == 'ARCTICDEM'

    def test_gimp(self):

        def down_check(url, *args, **kwargs):
            expected = 'gimpdem_90m_v01.1.tif'
            self.assertEqual(url, expected)
            return 'yo'

        with FakeDownloadManager('_download_topo_file_from_cluster',
                                 down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [-88, -88],
                                             source='GIMP')

        assert of[0] == 'yo'
        assert source == 'GIMP'

    def test_aster(self):

        # Make a fake topo file
        tf = make_fake_zipdir(os.path.join(self.dldir, 'ASTGTMV003_S88W121'),
                              fakefile='ASTGTMV003_S89W121_dem.tif')

        def down_check(url, *args, **kwargs):
            expect = ('https://e4ftl01.cr.usgs.gov/ASTER_B/ASTT/ASTGTM.003/' +
                      '2000.03.01/ASTGTMV003_S89W121.zip')
            self.assertEqual(url, expect)
            return tf

        with FakeDownloadManager('download_with_authentication', down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [-88.1, -88.9],
                                             source='ASTER')

        assert os.path.exists(of[0])
        assert source == 'ASTER'

    def test_tandem(self):
        # Make a fake topo file
        tf = make_fake_zipdir(os.path.join(self.dldir, 'TDM1_DEM__30_N60W146'),
                              fakefile='TDM1_DEM__30_N60W146.tif')

        def down_check(url, *args, **kwargs):
            expect = ('https://download.geoservice.dlr.de/TDM90/files/N60/' +
                      'W140/' + 'TDM1_DEM__30_N60W146.zip')
            self.assertEqual(url, expect)
            return tf

        with FakeDownloadManager('download_with_authentication', down_check):
            of, source = utils.get_topo_file([-145.2, -145.3], [60.1, 60.9],
                                             source='TANDEM')

        assert os.path.exists(of[0])
        assert source == 'TANDEM'

    def test_tandem_invalid(self):
        # If the TanDEM-X tile does not exist, a invalid file is created.
        # See https://github.com/OGGM/oggm/issues/893 for more details

        # Make a fake topo file
        tf = {'TDM1_DEM__30_N60W144.zip':
              make_fake_zipdir(os.path.join(self.dldir,
                                            'TDM1_DEM__30_N60W144'),
                               fakefile='TDM1_DEM__30_N60W146.tif'),
              'TDM1_DEM__30_N60W145.zip':
              touch(os.path.join(self.dldir, 'TDM1_DEM__30_N60W145.zip')),
              'TDM1_DEM__30_N60W146.zip':
              make_fake_zipdir(os.path.join(self.dldir,
                                            'TDM1_DEM__30_N60W146'),
                               fakefile='TDM1_DEM__30_N60W146.tif')
              }

        def down_check(url, *args, **kwargs):
            file = tf.get(url.split('/')[-1])
            self.assertIsNotNone(file)

            expect = 'https://download.geoservice.dlr.de/TDM90/files/N60/W140/'
            expect += file.split('/')[-1].replace('tif', '.zip')
            self.assertEqual(url, expect)

            return file

        with FakeDownloadManager('download_with_authentication', down_check):
            of, source = utils.get_topo_file([-144.1, -143.9], [60.5, 60.6],
                                             source='TANDEM')

        self.assertTrue(len(of) == 2)
        self.assertTrue(os.path.exists(of[0]))
        self.assertTrue(os.path.exists(of[1]))
        self.assertTrue(source == 'TANDEM')

        # check files
        files = [os.path.basename(f) for f in of]
        self.assertTrue('TDM1_DEM__30_N60W144_DEM.tif' in files)
        self.assertTrue('TDM1_DEM__30_N60W146_DEM.tif' in files)
        self.assertFalse('TDM1_DEM__30_N60W145_DEM.tif' in files)


class TestDataFiles(unittest.TestCase):

    def setUp(self):
        self.dldir = os.path.join(get_test_dir(), 'tmp_download')
        utils.mkdir(self.dldir)
        cfg.initialize()
        cfg.PATHS['dl_cache_dir'] = os.path.join(self.dldir, 'dl_cache')
        cfg.PATHS['working_dir'] = os.path.join(self.dldir, 'wd')
        cfg.PATHS['tmp_dir'] = os.path.join(self.dldir, 'extract')
        self.reset_dir()

    def tearDown(self):
        if os.path.exists(self.dldir):
            shutil.rmtree(self.dldir)

    def reset_dir(self):
        if os.path.exists(self.dldir):
            shutil.rmtree(self.dldir)
        utils.mkdir(cfg.PATHS['dl_cache_dir'])
        utils.mkdir(cfg.PATHS['working_dir'])
        utils.mkdir(cfg.PATHS['tmp_dir'])

    def test_download_demo_files(self):

        f = utils.get_demo_file('Hintereisferner_RGI5.shp')
        self.assertTrue(os.path.exists(f))

        sh = salem.read_shapefile(f)
        self.assertTrue(hasattr(sh, 'geometry'))

        # Data files
        cfg.initialize()

        lf, df = utils.get_wgms_files()
        self.assertTrue(os.path.exists(df))

        lf = utils.get_glathida_file()
        self.assertTrue(os.path.exists(lf))

    def test_file_extractor(self):

        tmp_file = os.path.join(cfg.PATHS['working_dir'], 'f.nc')
        touch(tmp_file)

        # writing file to a zipfile
        tmp_comp = os.path.join(cfg.PATHS['working_dir'], 'f.nc.zip')
        import zipfile
        with zipfile.ZipFile(tmp_comp, 'w') as zip:
            zip.write(tmp_file, arcname='f.nc')
        of = utils.file_extractor(tmp_comp)
        assert 'f.nc' in of
        assert os.path.isfile(of)
        os.remove(of)

        # writing file to a gzip
        tmp_comp = os.path.join(cfg.PATHS['working_dir'], 'f.nc.gz')
        import gzip
        with gzip.GzipFile(tmp_comp, 'w') as zip:
            with open(tmp_file, 'rb') as f_in:
                shutil.copyfileobj(f_in, zip)
        of = utils.file_extractor(tmp_comp)
        assert 'f.nc' in of
        assert os.path.isfile(of)
        os.remove(of)

        # writing file to a tar.gz
        tmp_comp = os.path.join(cfg.PATHS['working_dir'], 'f.nc.tar.gz')
        import tarfile
        with tarfile.open(tmp_comp, 'w:gz') as zip:
            zip.add(tmp_file, arcname='f.nc')
        of = utils.file_extractor(tmp_comp)
        assert 'f.nc' in of
        assert os.path.isfile(of)
        os.remove(of)

        # Raise error if nothing to extract
        with pytest.raises(InvalidParamsError):
            utils.file_extractor(tmp_file)

    def test_srtmzone(self):

        z = utils.srtm_zone(lon_ex=[-112, -112], lat_ex=[57, 57])
        self.assertTrue(len(z) == 1)
        self.assertEqual('14_01', z[0])

        z = utils.srtm_zone(lon_ex=[-72, -73], lat_ex=[-52, -53])
        self.assertTrue(len(z) == 1)
        self.assertEqual('22_23', z[0])

        # Alps
        ref = sorted(['39_04', '38_03', '38_04', '39_03'])
        z = utils.srtm_zone(lon_ex=[6, 14], lat_ex=[41, 48])
        self.assertTrue(len(z) == 4)
        self.assertEqual(ref, z)

    def test_tandemzone(self):

        z = utils.tandem_zone(lon_ex=[-112.1, -112.1], lat_ex=[-57.1, -57.1])
        self.assertTrue(len(z) == 1)
        self.assertEqual('S58/W110/TDM1_DEM__30_S58W113', z[0])

        z = utils.tandem_zone(lon_ex=[71, 71], lat_ex=[52, 52])
        self.assertTrue(len(z) == 1)
        self.assertEqual('N52/E070/TDM1_DEM__30_N52E071', z[0])

        z = utils.tandem_zone(lon_ex=[71, 71], lat_ex=[62, 62])
        self.assertTrue(len(z) == 1)
        self.assertEqual('N62/E070/TDM1_DEM__30_N62E070', z[0])

        z = utils.tandem_zone(lon_ex=[71, 71], lat_ex=[-82.1, -82.1])
        self.assertTrue(len(z) == 1)
        self.assertEqual('S83/E060/TDM1_DEM__30_S83E068', z[0])

        ref = sorted(['N00/E000/TDM1_DEM__30_N00E000',
                      'N00/E000/TDM1_DEM__30_N00E001',
                      'N00/W000/TDM1_DEM__30_N00W001',
                      'N00/W000/TDM1_DEM__30_N00W002',
                      'N01/E000/TDM1_DEM__30_N01E000',
                      'N01/E000/TDM1_DEM__30_N01E001',
                      'N01/W000/TDM1_DEM__30_N01W001',
                      'N01/W000/TDM1_DEM__30_N01W002',
                      'S01/E000/TDM1_DEM__30_S01E000',
                      'S01/E000/TDM1_DEM__30_S01E001',
                      'S01/W000/TDM1_DEM__30_S01W001',
                      'S01/W000/TDM1_DEM__30_S01W002',
                      'S02/E000/TDM1_DEM__30_S02E000',
                      'S02/E000/TDM1_DEM__30_S02E001',
                      'S02/W000/TDM1_DEM__30_S02W001',
                      'S02/W000/TDM1_DEM__30_S02W002'
                      ])
        z = utils.tandem_zone(lon_ex=[-1.3, 1.4], lat_ex=[-1.3, 1.4])
        self.assertTrue(len(z) == len(ref))
        self.assertEqual(ref, z)

        z = utils.tandem_zone(lon_ex=[-144.1, -143.9], lat_ex=[60.5, 60.6])
        self.assertTrue(len(z) == 3)
        self.assertEqual('N60/W140/TDM1_DEM__30_N60W145', z[1])

    def test_asterzone(self):

        z = utils.aster_zone(lon_ex=[137.5, 137.5],
                             lat_ex=[-72.5, -72.5])
        self.assertTrue(len(z) == 1)
        self.assertEqual('ASTGTMV003_S73E137', z[0])

        z = utils.aster_zone(lon_ex=[-95.5, -95.5],
                             lat_ex=[30.5, 30.5])
        self.assertTrue(len(z) == 1)
        self.assertEqual('ASTGTMV003_N30W096', z[0])

        z = utils.aster_zone(lon_ex=[-96.5, -94.5],
                             lat_ex=[30.5, 30.5])
        self.assertTrue(len(z) == 3)
        self.assertEqual('ASTGTMV003_N30W095', z[0])
        self.assertEqual('ASTGTMV003_N30W096', z[1])
        self.assertEqual('ASTGTMV003_N30W097', z[2])

    def test_nasazone(self):

        z = utils.nasadem_zone(lon_ex=[137.5, 137.5],
                               lat_ex=[-72.5, -72.5])
        self.assertTrue(len(z) == 1)
        self.assertEqual('s73e137', z[0])

        z = utils.nasadem_zone(lon_ex=[-95.5, -95.5],
                               lat_ex=[30.5, 30.5])
        self.assertTrue(len(z) == 1)
        self.assertEqual('n30w096', z[0])

        z = utils.nasadem_zone(lon_ex=[-96.5, -94.5],
                               lat_ex=[30.5, 30.5])
        self.assertTrue(len(z) == 3)
        self.assertEqual('n30w095', z[0])
        self.assertEqual('n30w096', z[1])
        self.assertEqual('n30w097', z[2])

    def test_mapzen_zone(self):

        z = utils.mapzen_zone(lon_ex=[137.5, 137.5], lat_ex=[45, 45],
                              zoom=10)
        self.assertTrue(len(z) == 1)
        self.assertEqual('10/903/368.tif', z[0])

        z = utils.mapzen_zone(lon_ex=[137.5, 137.5], lat_ex=[45, 45],
                              dx_meter=110)
        self.assertTrue(len(z) == 1)
        self.assertEqual('10/903/368.tif', z[0])

        z = utils.mapzen_zone(lon_ex=[137.5, 137.5], lat_ex=[-45, -45],
                              zoom=10)
        self.assertTrue(len(z) == 1)
        self.assertEqual('10/903/655.tif', z[0])

        z = utils.mapzen_zone(lon_ex=[137.5, 137.5], lat_ex=[-45, -45],
                              dx_meter=110)
        self.assertTrue(len(z) == 1)
        self.assertEqual('10/903/655.tif', z[0])

        # Check the minimum zoom level
        dx = 200
        for lat in np.arange(10) * 10:
            z = utils.mapzen_zone(lon_ex=[137.5, 137.5], lat_ex=[lat, lat],
                                  dx_meter=dx)
            assert int(z[0].split('/')[0]) > 9
            z = utils.mapzen_zone(lon_ex=[181, 181], lat_ex=[-lat, -lat],
                                  dx_meter=dx)
            assert int(z[0].split('/')[0]) > 9

    def test_dem3_viewpano_zone(self):

        lon_ex = -22.26
        lat_ex = 66.16
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['ISL']

        lon_ex = 17
        lat_ex = 80
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['SVALBARD']

        lon_ex = -8
        lat_ex = 71
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['JANMAYEN']

        lon_ex = 60
        lat_ex = 80
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['FJ']

        lon_ex = -7
        lat_ex = 62
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['FAR']

        lon_ex = 19
        lat_ex = 74.5
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['BEAR']

        lon_ex = -1
        lat_ex = 60.5
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['SHL']

        lon_ex = -30.733021
        lat_ex = 82.930238
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['GL-North']

        lon_ex = -52
        lat_ex = 70
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['GL-West']

        lon_ex = -43
        lat_ex = 60
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['GL-South']

        lon_ex = -24.7
        lat_ex = 69.8
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['GL-East']

        lon_ex = -22.26
        lat_ex = 66.16
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['ISL']

        lon_ex = -127.592802
        lat_ex = -74.479523
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['01-15']

        lat_ex = -72.383226
        lon_ex = -60.648126
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['16-30']

        lat_ex = -67.993527
        lon_ex = 65.482151
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['31-45']

        lat_ex = -71.670896
        lon_ex = 166.878916
        assert utils.dem3_viewpano_zone(lon_ex, lat_ex) == ['46-60']

        # normal tile
        z = utils.dem3_viewpano_zone([-179., -178.], [65., 65.])
        self.assertTrue(len(z) == 1)
        self.assertEqual('Q01', z[0])

        # normal tile
        z = utils.dem3_viewpano_zone([107, 107], [69, 69])
        self.assertTrue(len(z) == 1)
        self.assertEqual('R48', z[0])

        # Alps
        ref = sorted(['K31', 'K32', 'K33', 'L31', 'L32',
                      'L33', 'M31', 'M32', 'M33'])
        z = utils.dem3_viewpano_zone([6, 14], [41, 48])
        self.assertTrue(len(z) == 9)
        self.assertEqual(ref, z)

    def test_copdemzone(self):
        z = utils.copdem_zone(lat_ex=[-9.8, -9.5], lon_ex=[-77.6, -77.3])
        self.assertTrue(len(z) == 1)
        self.assertEqual(('DEM1_SAR_DGE_90_20110427T104941_20140819T105300_'
                          'ADS_000000_2733.DEM.tar'), z[0][0])
        self.assertEqual('Copernicus_DSM_30_S10_00_W078_00', z[0][1])

        z = utils.copdem_zone(lat_ex=[46.37, 46.59], lon_ex=[7.89, 8.12])
        self.assertTrue(len(z) == 2)
        self.assertTrue('Copernicus_DSM_30_N46_00_E008_00' in
                        [z[0][1], z[1][1]])
        self.assertTrue('Copernicus_DSM_30_N46_00_E007_00' in
                        [z[0][1], z[1][1]])

        # we want an error if copdem does not find all or any
        self.assertRaises(InvalidDEMError, utils.copdem_zone,
                          lat_ex=[0, 1], lon_ex=[0, 1])

    def test_is_dem_source_available(self):
        assert utils.is_dem_source_available('SRTM', [11, 11], [47, 47])
        assert utils.is_dem_source_available('GIMP', [-25, -25], [71, 71])
        assert not utils.is_dem_source_available('GIMP', [11, 11], [47, 47])
        assert utils.is_dem_source_available('ARCTICDEM', [-25, -25], [71, 71])
        assert utils.is_dem_source_available('RAMP', [-25, -25], [-71, -71])
        assert utils.is_dem_source_available('REMA', [-25, -25], [-71, -71])
        assert not utils.is_dem_source_available('AW3D30', [5, 5], [-60, -60])

        for s in ['TANDEM', 'AW3D30', 'MAPZEN', 'DEM3', 'ASTER', 'AW3D30']:
            assert utils.is_dem_source_available(s, [11, 11], [47, 47])

    def test_find_dem_zone(self):

        assert utils.default_dem_source('RGI60-11.00897') == 'NASADEM'
        assert utils.default_dem_source('RGI60-11.00897_merged') == 'NASADEM'
        assert utils.default_dem_source('RGI60-19.01251') == 'COPDEM'
        assert utils.default_dem_source('RGI60-19.00970') == 'REMA'
        assert utils.default_dem_source('RGI60-05.10315') == 'GIMP'
        assert utils.default_dem_source('RGI60-19.01820') == 'MAPZEN'

    def test_lrufilecache(self):

        f1 = os.path.join(self.dldir, 'f1.txt')
        f2 = os.path.join(self.dldir, 'f2.txt')
        f3 = os.path.join(self.dldir, 'f3.txt')
        open(f1, 'a').close()
        open(f2, 'a').close()
        open(f3, 'a').close()

        assert os.path.exists(f1)
        lru = utils.LRUFileCache(maxsize=2)
        lru.append(f1)
        assert os.path.exists(f1)
        lru.append(f2)
        assert os.path.exists(f1)
        lru.append(f3)
        assert not os.path.exists(f1)
        assert os.path.exists(f2)
        lru.append(f2)
        assert os.path.exists(f2)

        open(f1, 'a').close()
        lru = utils.LRUFileCache(l0=[f2, f3], maxsize=2)
        assert os.path.exists(f1)
        assert os.path.exists(f2)
        assert os.path.exists(f3)
        lru.append(f1)
        assert os.path.exists(f1)
        assert not os.path.exists(f2)
        assert os.path.exists(f3)

    def test_lruhandler(self):

        # initiate some files in empty directory
        self.reset_dir()
        foo = []
        bar = []
        for i, e in itertools.product(range(1, 4), ['foo', 'bar']):
            f = os.path.join(self.dldir, 'f{}.{}'.format(i, e))
            open(f, 'a').close()
            (foo.append(f) if e == 'foo' else bar.append(f))
            time.sleep(0.1)

        # init handler for dldir and ending 'foo'
        lru_foo = cfg.get_lru_handler(self.dldir, ending='.foo')
        # no maxsize specified: use default
        self.assertTrue(lru_foo.maxsize == cfg.PARAMS['lru_maxsize'])
        # all files should exist
        self.assertTrue(all([os.path.exists(f) for f in foo]))

        # init handler for dldir and ending bar
        cfg.get_lru_handler(self.dldir, maxsize=3, ending='.bar')
        # all files should exist
        self.assertTrue(all([os.path.exists(b) for b in bar]))

        # update foo handler, decresing maxsize should delete first file
        lru_foo2 = cfg.get_lru_handler(self.dldir, maxsize=2, ending='.foo')
        self.assertFalse(os.path.exists(foo[0]))
        self.assertTrue(os.path.exists(foo[1]))
        self.assertTrue(os.path.exists(foo[2]))
        # but this should be the same handler instance as above:
        self.assertTrue(lru_foo is lru_foo2)
        self.assertTrue(lru_foo.maxsize == 2)

        # And bar files should not be affected by decreased cache size of foo
        self.assertTrue(all([os.path.exists(b) for b in bar]))

    @pytest.mark.download
    def test_srtmdownload(self):
        from oggm.utils._downloads import _download_srtm_file

        # this zone does exist and file should be small enough for download
        zone = '68_11'
        fp = _download_srtm_file(zone)
        self.assertTrue(os.path.exists(fp))
        fp = _download_srtm_file(zone)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    def test_srtmdownloadfails(self):
        from oggm.utils._downloads import _download_srtm_file

        # this zone does not exist
        zone = '41_20'
        self.assertTrue(_download_srtm_file(zone) is None)

    @pytest.mark.download
    def test_nasadownload(self):
        from oggm.utils._downloads import _download_nasadem_file

        # this zone does exist and file should be small enough for download
        zone = 'n01e119'
        fp = _download_nasadem_file(zone)
        self.assertTrue(os.path.exists(fp))
        fp = _download_nasadem_file(zone)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    def test_nasademdownloadfails(self):
        from oggm.utils._downloads import _download_nasadem_file

        # this zone does not exist
        zone = 'n80w100'
        self.assertTrue(_download_nasadem_file(zone) is None)

    @pytest.mark.download
    @pytest.mark.creds
    def test_tandemdownload(self):
        from oggm.utils._downloads import _download_tandem_file

        # this zone does exist and file should be small enough for download
        zone = 'N47/E010/TDM1_DEM__30_N47E011'
        fp = _download_tandem_file(zone)
        self.assertTrue(os.path.exists(fp))
        fp = _download_tandem_file(zone)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    @pytest.mark.creds
    def test_tandemdownloadfails(self):
        from oggm.utils._downloads import _download_tandem_file

        # this zone does not exist, so it should return None
        zone = 'N47/E010/TDM1_DEM__30_NxxExxx'
        fp = _download_tandem_file(zone)
        self.assertIsNone(fp)

    @pytest.mark.download
    @pytest.mark.creds
    def test_copdemdownload(self):
        from oggm.utils._downloads import _download_copdem_file

        # this zone does exist and file should be small enough for download
        cppfile = ('DEM1_SAR_DGE_90_20110109T094723_20140326T001716_ADS_'
                   '000000_5845.DEM.tar')
        tilename = 'Copernicus_DSM_30_N78_00_E102_00'
        fp = _download_copdem_file(cppfile, tilename)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.creds
    def test_asterdownload(self):
        from oggm.utils._downloads import _download_aster_file

        # this zone does exist and file should be small enough for download
        zone = 'ASTGTMV003_S73E137'
        fp = _download_aster_file(zone)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    def test_mapzen_download(self):
        from oggm.utils._downloads import _download_mapzen_file

        # this zone does exist and file should be small enough for download
        zone = '10/903/368.tif'
        fp = _download_mapzen_file(zone)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    def test_gimp(self):
        fp, z = utils.get_topo_file([], [], source='GIMP')
        self.assertTrue(os.path.exists(fp[0]))
        self.assertEqual(z, 'GIMP')

    @pytest.mark.download
    def test_iceland(self):
        fp, z = utils.get_topo_file([-20, -20], [65, 65], source='DEM3')
        self.assertTrue(os.path.exists(fp[0]))

    @pytest.mark.creds
    def test_asterdownloadfails(self):
        from oggm.utils._downloads import _download_aster_file

        # this zone does not exist
        zone = 'ASTGTMV003_S99E010'
        self.assertTrue(_download_aster_file(zone) is None)

    @pytest.mark.download
    def test_download_histalp(self):

        from oggm.shop.histalp import get_histalp_file
        of = get_histalp_file('tmp')
        self.assertTrue(os.path.exists(of))
        of = get_histalp_file('pre')
        self.assertTrue(os.path.exists(of))

    @pytest.mark.download
    def test_download_rgi5(self):

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_extract')

        of = utils.get_rgi_dir(version='5')
        of = os.path.join(of, '01_rgi50_Alaska', '01_rgi50_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @pytest.mark.download
    def test_download_rgi6(self):

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_extract')

        of = utils.get_rgi_dir(version='6')
        of = os.path.join(of, '01_rgi60_Alaska', '01_rgi60_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @pytest.mark.download
    def test_download_rgi61(self):

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_extract')

        of = utils.get_rgi_dir(version='61')
        of = os.path.join(of, '01_rgi61_Alaska', '01_rgi61_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @pytest.mark.download
    def test_download_rgi60_intersects(self):

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_extract')

        of = utils.get_rgi_intersects_dir(version='6')
        of = os.path.join(of, '01_rgi60_Alaska',
                          'intersects_01_rgi60_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @pytest.mark.download
    def test_download_rgi61_intersects(self):

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_extract')

        of = utils.get_rgi_intersects_dir(version='61')
        of = os.path.join(of, '01_rgi61_Alaska',
                          'intersects_01_rgi61_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @pytest.mark.download
    def test_download_dem3_viewpano(self):
        from oggm.utils._downloads import _download_dem3_viewpano

        # this zone does exist and file should be small enough for download
        zone = 'L32'
        fp = _download_dem3_viewpano(zone)
        self.assertTrue(os.path.exists(fp))
        zone = 'U44'
        fp = _download_dem3_viewpano(zone)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    def test_download_dem3_viewpano_fails(self):
        from oggm.utils._downloads import _download_dem3_viewpano

        # this zone does not exist
        zone = 'dummy'
        fp = _download_dem3_viewpano(zone)
        self.assertTrue(fp is None)

    @pytest.mark.download
    def test_download_aw3d30(self):
        from oggm.utils._downloads import _download_aw3d30_file

        # this zone does exist and file should be small enough for download
        zone = 'N000E105/N002E107'
        fp = _download_aw3d30_file(zone)
        self.assertTrue(os.path.exists(fp))
        zone = 'S085W050/S081W048'
        fp = _download_aw3d30_file(zone)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    def test_download_aw3d30_fails(self):
        from oggm.utils._downloads import _download_aw3d30_file
        from urllib.error import URLError

        # this zone does not exist
        zone = 'N000E005/N000E005'
        self.assertRaises(URLError, _download_aw3d30_file, zone)

    @pytest.mark.download
    def test_from_prepro(self):

        # Read in the RGI file
        rgi_file = utils.get_demo_file('rgi_oetztal.shp')
        rgidf = gpd.read_file(rgi_file)
        rgidf['RGIId'] = [rid.replace('RGI50', 'RGI60')
                          for rid in rgidf.RGIId]

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf.iloc[:2],
                                                  from_prepro_level=1,
                                                  prepro_rgi_version='61',
                                                  prepro_border=10)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)


class TestSkyIsFalling(unittest.TestCase):

    def test_projplot(self):

        # this caused many problems on Linux Mint distributions
        # this is just to be sure that on your system, everything is fine
        import pyproj
        import matplotlib.pyplot as plt

        pyproj.Proj(proj='latlong', datum='WGS84')
        plt.figure()
        plt.close()

        srs = ('+units=m +proj=lcc +lat_1=29.0 +lat_2=29.0 '
               '+lat_0=29.0 +lon_0=89.8')

        proj_out = pyproj.Proj("EPSG:4326", preserve_units=True)
        proj_in = pyproj.Proj(srs, preserve_units=True)

        from salem.gis import transform_proj
        lon, lat = transform_proj(proj_in, proj_out, -2235000, -2235000)
        np.testing.assert_allclose(lon, 70.75731, atol=1e-5)
