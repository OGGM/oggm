import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import unittest
import os
import shutil
import time
import gzip
import bz2
import zlib
import pytest
from unittest import mock

import salem
import numpy as np
import pandas as pd
import geopandas as gpd
from numpy.testing import assert_array_equal, assert_allclose

import oggm
from oggm import utils, workflow, tasks
from oggm.utils import _downloads
from oggm import cfg
from oggm.tests.funcs import (get_test_dir, patch_url_retrieve_github,
                              init_hef, TempEnvironmentVariable)
from oggm.utils import shape_factor_adhikari
from oggm.exceptions import (InvalidParamsError,
                             DownloadVerificationFailedException)


pytestmark = pytest.mark.test_env("utils")
_url_retrieve = None

TEST_GDIR_URL = ('https://cluster.klima.uni-bremen.de/~fmaussion/'
                 'test_gdirs/oggm_v1.1/')


def setup_module(module):
    module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github


def teardown_module(module):
    oggm.utils._downloads.oggm_urlretrieve = module._url_retrieve


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
        y, m = utils.hydrodate_to_calendardate(1, 1)
        assert (y, m) == (0, 10)
        y, m = utils.hydrodate_to_calendardate(1, 4)
        assert (y, m) == (1, 1)
        y, m = utils.hydrodate_to_calendardate(1, 12)
        assert (y, m) == (1, 9)

        y, m = utils.hydrodate_to_calendardate([1, 1, 1], [1, 4, 12])
        np.testing.assert_array_equal(y, [0, 1, 1])
        np.testing.assert_array_equal(m, [10, 1, 9])

        y, m = utils.calendardate_to_hydrodate(1, 1)
        assert (y, m) == (1, 4)
        y, m = utils.calendardate_to_hydrodate(1, 9)
        assert (y, m) == (1, 12)
        y, m = utils.calendardate_to_hydrodate(1, 10)
        assert (y, m) == (2, 1)

        y, m = utils.calendardate_to_hydrodate([1, 1, 1], [1, 9, 10])
        np.testing.assert_array_equal(y, [1, 1, 2])
        np.testing.assert_array_equal(m, [4, 12, 1])

        # Roundtrip
        time = pd.period_range('0001-01', '1000-12', freq='M')
        y, m = utils.calendardate_to_hydrodate(time.year, time.month)
        y, m = utils.hydrodate_to_calendardate(y, m)
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


class TestInitialize(unittest.TestCase):

    def setUp(self):
        cfg.initialize()
        self.homedir = os.path.expanduser('~')

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
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

        df = gdir.get_ref_length_data()
        assert df.name == 'Hintereis'
        assert len(df) == 105

    def test_glacier_characs(self):

        gdir = init_hef()

        df = utils.compile_glacier_statistics([gdir], path=False)
        assert len(df) == 1
        assert np.all(~df.isnull())
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


class TestStartFromTar(unittest.TestCase):

    def setUp(self):
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_tar_tools')

        # Init
        cfg.initialize()
        cfg.set_intersects_db(utils.get_demo_file('rgi_intersect_oetztal.shp'))

        # Read in the RGI file
        rgi_file = utils.get_demo_file('rgi_oetztal.shp')
        self.rgidf = gpd.read_file(rgi_file)
        cfg.PARAMS['use_multiprocessing'] = False
        cfg.PATHS['dem_file'] = utils.get_demo_file('srtm_oetztal.tif')
        cfg.PATHS['working_dir'] = self.testdir
        self.clean_dir()

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        utils.mkdir(self.testdir, reset=True)

    def test_to_and_from_tar(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(self.rgidf)

        # End - compress all
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs)

        # Test - reopen form tar
        gdirs = workflow.init_glacier_regions(self.rgidf, from_tar=True,
                                              delete_tar=True)
        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert not os.path.exists(gdir.dir + '.tar.gz')
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

        workflow.execute_entity_task(utils.gdir_to_tar, gdirs)

        gdirs = workflow.init_glacier_regions(self.rgidf, from_tar=True)
        for gdir in gdirs:
            assert gdir.has_file('gridded_data')
            assert os.path.exists(gdir.dir + '.tar.gz')

    def test_to_and_from_tar_new_dir(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(self.rgidf)

        # End - compress all
        base_dir = os.path.join(self.testdir, 'new_base_dir')
        paths = workflow.execute_entity_task(utils.gdir_to_tar, gdirs,
                                             base_dir=base_dir)

        # Test - reopen form tar after copy
        for p, gdir in zip(paths, gdirs):
            assert base_dir in p
            shutil.copyfile(p, os.path.normpath(gdir.dir) + '.tar.gz')

        gdirs = workflow.init_glacier_regions(self.rgidf, from_tar=True,
                                              delete_tar=True)
        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert not os.path.exists(gdir.dir + '.tar.gz')
            assert gdir.rgi_area_km2 > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    def test_to_and_from_tar_string(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(self.rgidf)

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
        cfg.PARAMS['use_multiprocessing'] = False
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

    @mock.patch('oggm.utils._downloads.GDIR_URL', TEST_GDIR_URL)
    def test_start_from_level_1(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(self.rgidf.iloc[:2],
                                              from_prepro_level=1,
                                              prepro_rgi_version='61',
                                              prepro_border=20,
                                              use_demo_glaciers=False)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    @mock.patch('oggm.utils._downloads.GDIR_URL', TEST_GDIR_URL)
    def test_start_from_level_1_str(self):

        # Go - initialize working directories
        entitites = self.rgidf.iloc[:2].RGIId
        cfg.PARAMS['border'] = 20
        gdirs = workflow.init_glacier_regions(entitites,
                                              from_prepro_level=1,
                                              use_demo_glaciers=False)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

        # One string
        cfg.PARAMS['border'] = 20
        gdirs = workflow.init_glacier_regions('RGI60-11.00897',
                                              from_prepro_level=1,
                                              use_demo_glaciers=False)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    @mock.patch('oggm.utils._downloads.GDIR_URL', TEST_GDIR_URL)
    def test_start_from_level_2(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(self.rgidf.iloc[:2],
                                              from_prepro_level=2,
                                              prepro_rgi_version='61',
                                              prepro_border=20,
                                              use_demo_glaciers=False)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert gdir.has_file('climate_monthly')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    @mock.patch('oggm.utils._downloads.GDIR_URL', TEST_GDIR_URL)
    def test_start_from_level_3(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(self.rgidf.iloc[:2],
                                              from_prepro_level=3,
                                              prepro_rgi_version='61',
                                              prepro_border=20,
                                              use_demo_glaciers=False)
        n_intersects = 0
        for gdir in gdirs:
            assert gdir.has_file('dem')
            assert gdir.has_file('gridded_data')
            assert gdir.has_file('climate_monthly')
            n_intersects += gdir.has_file('intersects')
        assert n_intersects > 0

        df = utils.compile_glacier_statistics(gdirs)
        assert 'dem_med_elev' in df

        df = utils.compile_climate_statistics(gdirs, add_climate_period=[1920,
                                                                         1960,
                                                                         2000])
        assert 'tstar_avg_temp_mean_elev' in df
        assert '1905-1935_avg_temp_mean_elev' in df

        workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    @mock.patch('oggm.utils._downloads.GDIR_URL', TEST_GDIR_URL)
    def test_start_from_level_4(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(self.rgidf.iloc[:2],
                                              from_prepro_level=4,
                                              prepro_rgi_version='61',
                                              prepro_border=20,
                                              use_demo_glaciers=False)
        workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                                     nyears=10)

    def test_start_from_demo(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(['kwf', 'hef'],
                                              from_prepro_level=4,
                                              prepro_rgi_version='61',
                                              prepro_border=10)
        workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                                     nyears=10)

    def test_corrupted_file(self):

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(['hef'],
                                              from_prepro_level=4,
                                              prepro_rgi_version='61',
                                              prepro_border=10)

        cfile = utils.get_prepro_gdir('61', 'RGI60-11.00787', 10, 4,
                                      demo_url=True)
        assert 'cluster.klima.uni-bremen.de/~fmaussion/' in cfile

        # Replace with a dummy file
        os.remove(cfile)
        with open(cfile, 'w') as f:
            f.write('ups')

        # This should retrigger a download and just work
        gdirs = workflow.init_glacier_regions(['hef'],
                                              from_prepro_level=4,
                                              prepro_rgi_version='61',
                                              prepro_border=10)
        workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                                     nyears=10)


class TestPreproCLI(unittest.TestCase):

    def setUp(self):
        self.testdir = os.path.join(get_test_dir(), 'tmp_prepro_levs')
        self.reset_dir()

    def tearDown(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    def reset_dir(self):
        utils.mkdir(self.testdir, reset=True)

    def test_parse_args(self):

        from oggm.cli import prepro_levels

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
                                           '--map-border', '160'])

        assert 'working_dir' in kwargs
        assert 'output_folder' in kwargs
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '01'
        assert kwargs['border'] == 160

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
                                           ])

        assert 'working_dir' in kwargs
        assert 'output_folder' in kwargs
        assert 'local' in kwargs['working_dir']
        assert 'local' in kwargs['output_folder']
        assert kwargs['rgi_version'] is None
        assert kwargs['rgi_reg'] == '00'
        assert kwargs['border'] == 160
        assert not kwargs['is_test']
        assert kwargs['demo']

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
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

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
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

        kwargs = prepro_levels.parse_args(['--rgi-reg', '1',
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

    def test_full_run(self):

        from oggm.cli.prepro_levels import run_prepro_levels

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
        run_prepro_levels(rgi_version=None, rgi_reg='11', border=20,
                          output_folder=odir, working_dir=wdir, is_test=True,
                          test_rgidf=rgidf, test_intersects_file=inter,
                          test_topofile=topof,
                          test_crudir=os.path.dirname(cru_file))

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L1', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'dem_source' in df

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L3', 'summary',
                                      'glacier_statistics_11.csv'))
        assert 'inv_volume_km3' in df

        df = pd.read_csv(os.path.join(odir, 'RGI61', 'b_020', 'L3', 'summary',
                                      'climate_statistics_11.csv'))
        assert '1945-1975_avg_prcp' in df

        assert os.path.isfile(os.path.join(odir, 'RGI61', 'b_020',
                                           'package_versions.txt'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L1'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L2'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L3'))
        assert os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L4'))
        assert not os.path.isdir(os.path.join(odir, 'RGI61', 'b_020', 'L5'))

        # See if we can start from all levs
        from oggm import tasks
        from oggm.core.flowline import FlowlineModel
        cfg.PARAMS['continue_on_error'] = False
        rid = df.rgi_id.iloc[0]
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
        tasks.glacier_masks(gdir)
        with pytest.raises(FileNotFoundError):
            tasks.init_present_time_glacier(gdir)

        # L3
        tarf = os.path.join(odir, 'RGI61', 'b_020', 'L3',
                            rid[:8], rid[:11], rid + '.tar.gz')
        assert not os.path.isfile(tarf)
        gdir = oggm.GlacierDirectory(entity, from_tar=tarf)
        tasks.init_present_time_glacier(gdir)
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
            tasks.init_present_time_glacier(gdir)


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
        run_benchmark(rgi_version=None, rgi_reg='11', border=80,
                      output_folder=odir, working_dir=wdir, is_test=True,
                      test_rgidf=rgidf, test_intersects_file=inter,
                      test_topofile=topof,
                      test_crudir=os.path.dirname(cru_file))

        df = pd.read_csv(os.path.join(odir, 'benchmarks_b080.csv'),
                         index_col=0)
        assert len(df) > 15


def touch(path):
    """Equivalent to linux's touch"""
    with open(path, 'a'):
        os.utime(path, None)
    return path


def make_fake_zipdir(dir_path, fakefile=None):
    """Creates a directory with a file in it if asked to, then compresses it"""
    utils.mkdir(dir_path)
    if fakefile:
        touch(os.path.join(dir_path, fakefile))
    shutil.make_archive(dir_path, 'zip', dir_path)
    return dir_path + '.zip'


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
        cfg.PATHS['cru_dir'] = os.path.join(self.dldir, 'cru_test')
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
        utils.mkdir(cfg.PATHS['cru_dir'])

    def prepare_verify_test(self, valid_size=True, valid_crc32=True):
        self.reset_dir()
        cfg.PARAMS['dl_verify'] = True

        tgt_path = os.path.join(cfg.PATHS['dl_cache_dir'], 'test.com',
                                'test.txt')

        file_size = 1024
        file_data = os.urandom(file_size)
        file_crc32 = zlib.crc32(file_data)

        utils.mkdir(os.path.dirname(tgt_path))
        with open(tgt_path, 'wb') as f:
            f.write(file_data)

        if not valid_size:
            file_size += 1
        if not valid_crc32:
            file_crc32 += 1

        data = utils.get_dl_verify_data()
        data['test.com/test.txt'] = (file_size, file_crc32)

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

        def down_check(url, cache_name=None, reset=False):
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

        def down_check(url, cache_name=None, reset=False):
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

        def down_check(url, cache_name=None, reset=False):
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

        def down_check(url, cache_name=None, reset=False):
            expected = ('https://cluster.klima.uni-bremen.de/data/rgi/' +
                        'RGI_V60_Intersects.zip')
            self.assertEqual(url, expected)
            return rgi_f

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            rgi = utils.get_rgi_intersects_dir(version='6')

        assert os.path.isdir(rgi)
        assert os.path.exists(os.path.join(rgi,
                                           'Intersects_OGGM_Manifest.txt'))

    def test_cru(self):

        # Create fake cru file
        cf = os.path.join(self.dldir, 'cru_ts4.01.1901.2016.tmp.dat.nc.gz')
        with gzip.open(cf, 'wb') as gz:
            gz.write(b'dummy')

        def down_check(url, cache_name=None, reset=False):
            expected = ('https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.01/'
                        'cruts.1709081022.v4.01/tmp/'
                        'cru_ts4.01.1901.2016.tmp.dat.nc.gz')
            self.assertEqual(url, expected)
            return cf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            tf = utils.get_cru_file('tmp')

        assert os.path.exists(tf)

    def test_histalp(self):

        # Create fake histalp file
        cf = os.path.join(self.dldir, 'HISTALP_temperature_1780-2014.nc.bz2')
        with bz2.open(cf, 'wb') as gz:
            gz.write(b'dummy')

        def down_check(url, cache_name=None, reset=False):
            expected = ('http://www.zamg.ac.at/histalp/download/grid5m/'
                        'HISTALP_temperature_1780-2014.nc.bz2')
            self.assertEqual(url, expected)
            return cf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            tf = utils.get_histalp_file('tmp')

        assert os.path.exists(tf)

    def test_srtm(self):

        # Make a fake topo file
        tf = make_fake_zipdir(os.path.join(self.dldir, 'srtm_39_03'),
                              fakefile='srtm_39_03.tif')

        def down_check(url, cache_name=None, reset=False):
            expected = ('http://srtm.csi.cgiar.org/wp-content/uploads/files/'
                        'srtm_5x5/TIFF/srtm_39_03.zip')
            self.assertEqual(url, expected)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([11.3, 11.3], [47.1, 47.1])

        assert os.path.exists(of[0])
        assert source == 'SRTM'

    def test_dem3(self):

        def down_check(url, cache_name=None, reset=False):
            expected = 'http://viewfinderpanoramas.org/dem3/T10.zip'
            self.assertEqual(url, expected)
            return self.dem3_testfile

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [76.8, 76.8])

        assert os.path.exists(of[0])
        assert source == 'DEM3'

    def test_ramp(self):

        def down_check(url):
            expected = 'AntarcticDEM_wgs84.tif'
            self.assertEqual(url, expected)
            return 'yo'

        with FakeDownloadManager('_download_topo_file_from_cluster',
                                 down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [-88, -88],
                                             rgi_region='19',
                                             rgi_subregion='19-11')

        assert of[0] == 'yo'
        assert source == 'RAMP'

    def test_gimp(self):

        def down_check(url):
            expected = 'gimpdem_90m_v01.1.tif'
            self.assertEqual(url, expected)
            return 'yo'

        with FakeDownloadManager('_download_topo_file_from_cluster',
                                 down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [-88, -88],
                                             rgi_region=5)

        assert of[0] == 'yo'
        assert source == 'GIMP'

    def test_aster(self):

        # Make a fake topo file
        tf = make_fake_zipdir(os.path.join(self.dldir, 'ASTGTM2_S88W121'),
                              fakefile='ASTGTM2_S88W121_dem.tif')

        def down_check(url):
            expected = 'ASTGTM_V2/UNIT_S90W125/ASTGTM2_S88W121.zip'
            self.assertEqual(url, expected)
            return tf

        with FakeDownloadManager('_aws_file_download_unlocked', down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [-88, -88],
                                             source='ASTER')

        assert os.path.exists(of[0])
        assert source == 'ASTER'

    def test_cmip5(self):

        fn = 'pr_mon_NorESM1-M_historicalNat_r1i1p1_g025.nc'

        def down_check(url, reset=False):
            expected = ('https://cluster.klima.uni-bremen.de/~nicolas/cmip5-ng'
                        '/pr/pr_mon_NorESM1-M_historicalNat_r1i1p1_g025.nc')
            self.assertEqual(url, expected)
            return True

        with FakeDownloadManager('file_downloader', down_check):
            assert utils.get_cmip5_file(fn)

        fn = 'tas_mon_CCSM4_historicalNat_r1i1p1_g025.nc'

        def down_check(url, reset=False):
            expected = ('https://cluster.klima.uni-bremen.de/~nicolas/cmip5-ng'
                        '/tas/tas_mon_CCSM4_historicalNat_r1i1p1_g025.nc')
            self.assertEqual(url, expected)
            return True

        with FakeDownloadManager('file_downloader', down_check):
            assert utils.get_cmip5_file(fn)


class TestDataFiles(unittest.TestCase):

    def setUp(self):
        self.dldir = os.path.join(get_test_dir(), 'tmp_download')
        utils.mkdir(self.dldir)
        cfg.initialize()
        cfg.PATHS['dl_cache_dir'] = os.path.join(self.dldir, 'dl_cache')
        cfg.PATHS['working_dir'] = os.path.join(self.dldir, 'wd')
        cfg.PATHS['tmp_dir'] = os.path.join(self.dldir, 'extract')
        self.reset_dir()
        oggm.utils._downloads.oggm_urlretrieve = _url_retrieve

    def tearDown(self):
        if os.path.exists(self.dldir):
            shutil.rmtree(self.dldir)
        utils.oggm_urlretrieve = patch_url_retrieve_github

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

    def test_asterzone(self):

        z, u = utils.aster_zone(lon_ex=[137.5, 137.5],
                                lat_ex=[-72.5, -72.5])
        self.assertTrue(len(z) == 1)
        self.assertTrue(len(u) == 1)
        self.assertEqual('S73E137', z[0])
        self.assertEqual('S75E135', u[0])

        z, u = utils.aster_zone(lon_ex=[-95.5, -95.5],
                                lat_ex=[30.5, 30.5])
        self.assertTrue(len(z) == 1)
        self.assertTrue(len(u) == 1)
        self.assertEqual('N30W096', z[0])
        self.assertEqual('N30W100', u[0])

        z, u = utils.aster_zone(lon_ex=[-96.5, -95.5],
                                lat_ex=[30.5, 30.5])
        self.assertTrue(len(z) == 2)
        self.assertTrue(len(u) == 2)
        self.assertEqual('N30W096', z[1])
        self.assertEqual('N30W100', u[1])
        self.assertEqual('N30W097', z[0])
        self.assertEqual('N30W100', u[0])

    def test_dem3_viewpano_zone(self):

        test_loc = {'ISL': [-25., -12., 63., 67.],  # Iceland
                    'SVALBARD': [10., 34., 76., 81.],
                    'JANMAYEN': [-10., -7., 70., 72.],
                    'FJ': [36., 66., 79., 82.],  # Franz Josef Land
                    'FAR': [-8., -6., 61., 63.],  # Faroer
                    'BEAR': [18., 20., 74., 75.],  # Bear Island
                    'SHL': [-3., 0., 60., 61.],  # Shetland
                    # Antarctica tiles as UTM zones, FILES ARE LARGE!!!!!
                    # '01-15': [-180., -91., -90, -60.],
                    # '16-30': [-91., -1., -90., -60.],
                    # '31-45': [-1., 89., -90., -60.],
                    # '46-60': [89., 189., -90., -60.],
                    # Greenland tiles
                    # 'GL-North': [-78., -11., 75., 84.],
                    # 'GL-West': [-68., -42., 64., 76.],
                    # 'GL-South': [-52., -40., 59., 64.],
                    # 'GL-East': [-42., -17., 64., 76.]
                    }
        # special names
        for key in test_loc:
            z = utils.dem3_viewpano_zone([test_loc[key][0], test_loc[key][1]],
                                         [test_loc[key][2], test_loc[key][3]])
            self.assertTrue(len(z) == 1)

            self.assertEqual(key, z[0])

        # weird Antarctica tile
        # z = utils.dem3_viewpano_zone([-91., -90.], [-72., -68.])
        # self.assertTrue(len(z) == 1)
        # self.assertEqual('SR15', z[0])

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

        self.reset_dir()
        f1 = os.path.join(self.dldir, 'f1.txt')
        f2 = os.path.join(self.dldir, 'f2.txt')
        f3 = os.path.join(self.dldir, 'f3.txt')
        open(f1, 'a').close()
        time.sleep(0.1)
        open(f2, 'a').close()
        time.sleep(0.1)
        open(f3, 'a').close()

        cfg.get_lru_handler(self.dldir, maxsize=3, ending='.txt')
        assert os.path.exists(f1)
        assert os.path.exists(f2)
        assert os.path.exists(f3)

        cfg.get_lru_handler(self.dldir, maxsize=2, ending='.txt')
        assert not os.path.exists(f1)
        assert os.path.exists(f2)
        assert os.path.exists(f3)

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
    @pytest.mark.creds
    def test_tandemdownload(self):
        from oggm.utils._downloads import _download_tandem_file

        # this zone does exist and file should be small enough for download
        zone = 'N47/E010/TDM1_DEM__30_N47E011'
        fp = _download_tandem_file(zone)
        self.assertTrue(os.path.exists(fp))
        fp = _download_tandem_file(zone)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.creds
    def test_asterdownload(self):
        from oggm.utils._downloads import _download_aster_file

        # this zone does exist and file should be small enough for download
        zone = 'S73E137'
        unit = 'S75E135'
        fp = _download_aster_file(zone, unit)
        self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    def test_gimp(self):
        fp, z = utils.get_topo_file([], [], rgi_region=5)
        self.assertTrue(os.path.exists(fp[0]))
        self.assertEqual(z, 'GIMP')

    @pytest.mark.download
    def test_iceland(self):
        fp, z = utils.get_topo_file([-20, -20], [65, 65])
        self.assertTrue(os.path.exists(fp[0]))

    @pytest.mark.creds
    def test_asterdownloadfails(self):
        from oggm.utils._downloads import _download_aster_file

        # this zone does not exist
        zone = 'bli'
        unit = 'S75E135'
        self.assertTrue(_download_aster_file(zone, unit) is None)

    @pytest.mark.download
    def test_download_cru(self):

        tmp = cfg.PATHS['cru_dir']
        cfg.PATHS['cru_dir'] = os.path.join(self.dldir, 'cru_extract')

        of = utils.get_cru_file('tmp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['cru_dir'] = tmp

    @pytest.mark.download
    def test_download_histalp(self):

        tmp = cfg.PATHS['cru_dir']
        cfg.PATHS['cru_dir'] = os.path.join(self.dldir, 'cru_extract')

        of = utils.get_histalp_file('tmp')
        self.assertTrue(os.path.exists(of))
        of = utils.get_histalp_file('pre')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['cru_dir'] = tmp

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
    def test_download_cmip5(self):
        fn = 'pr_mon_NorESM1-M_historicalNat_r1i1p1_g025.nc'
        fp = utils.get_cmip5_file(fn)
        self.assertTrue(os.path.isfile(fp))

    @pytest.mark.download
    def test_auto_topo(self):
        # Test for combine
        fdem, src = utils.get_topo_file([6, 14], [41, 41])
        self.assertEqual(src, 'SRTM')
        self.assertEqual(len(fdem), 2)
        for fp in fdem:
            self.assertTrue(os.path.exists(fp))

        fdem, src = utils.get_topo_file([-143, -131], [61, 61])
        self.assertEqual(src, 'DEM3')
        self.assertEqual(len(fdem), 3)
        for fp in fdem:
            self.assertTrue(os.path.exists(fp))

    @pytest.mark.download
    def test_from_prepro(self):

        # Read in the RGI file
        rgi_file = utils.get_demo_file('rgi_oetztal.shp')
        rgidf = gpd.read_file(rgi_file)
        rgidf['RGIId'] = [rid.replace('RGI50', 'RGI60')
                          for rid in rgidf.RGIId]

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(rgidf.iloc[:2],
                                              from_prepro_level=1,
                                              prepro_rgi_version='61',
                                              prepro_border=10,
                                              use_demo_glaciers=False)
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

        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        proj_in = pyproj.Proj(srs, preserve_units=True)

        lon, lat = pyproj.transform(proj_in, proj_out, -2235000, -2235000)
        np.testing.assert_allclose(lon, 70.75731, atol=1e-5)
