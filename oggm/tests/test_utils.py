import warnings
warnings.filterwarnings("once", category=DeprecationWarning)
import unittest
import os
import shutil
import time
import gzip

import salem
import numpy as np
import pandas as pd
import geopandas as gpd
from numpy.testing import assert_array_equal, assert_allclose

import oggm
from oggm import utils
from oggm import cfg
from oggm.tests import is_download
from oggm.tests.funcs import get_test_dir, patch_url_retrieve, init_hef

_url_retrieve = None


def setup_module(module):
    module._url_retrieve = utils._urlretrieve
    utils._urlretrieve = patch_url_retrieve


def teardown_module(module):
    utils._urlretrieve = module._url_retrieve


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
        utils.show_versions()

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
        assert len(reg_names) == 19
        assert reg_names.loc[3].values[0] == 'Arctic Canada North'


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

        df = utils.glacier_characteristics([gdir], path=False)
        assert len(df) == 1
        assert np.all(~df.isnull())
        assert len(df.columns) >= 36
        df = df.iloc[0]
        np.testing.assert_allclose(df['dem_mean_elev'],
                                   df['flowline_mean_elev'], atol=5)
        np.testing.assert_allclose(df['tstar_avg_prcp'],
                                   2811, atol=5)
        np.testing.assert_allclose(df['tstar_avg_prcpsol_max_elev'],
                                   2811, atol=50)


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
        self._store = getattr(utils, func_name)

    def __enter__(self):
        self._store = getattr(utils, self.func_name)
        setattr(utils, self.func_name, self.new_func)

    def __exit__(self, *args):
        setattr(utils, self.func_name, self._store)


class TestFakeDownloads(unittest.TestCase):

    def setUp(self):
        self.dldir = os.path.join(get_test_dir(), 'tmp_download')
        utils.mkdir(self.dldir)
        cfg.initialize()
        cfg.PATHS['dl_cache_dir'] = os.path.join(self.dldir, 'dl_cache')
        cfg.PATHS['working_dir'] = os.path.join(self.dldir, 'wd')
        cfg.PATHS['tmp_dir'] = os.path.join(self.dldir, 'extract')
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_test')
        cfg.PATHS['cru_dir'] = os.path.join(self.dldir, 'cru_test')
        cfg.CACHE_DIR = os.path.join(self.dldir, 'cache_dir')
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
        utils.mkdir(cfg.CACHE_DIR)

    def test_github_no_internet(self):
        self.reset_dir()
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
            rgi = utils.get_rgi_dir()

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
        rgi_dir = os.path.join(self.dldir, 'rgi50')
        utils.mkdir(rgi_dir)
        make_fake_zipdir(os.path.join(rgi_dir, 'RGI_V5_Intersects'),
                         fakefile='Intersects_OGGM_Manifest.txt')
        rgi_f = make_fake_zipdir(rgi_dir)

        def down_check(url, cache_name=None, reset=False):
            expected = ('https://www.dropbox.com/s/y73sdxygdiq7whv/' +
                        'RGI_V5_Intersects.zip?dl=1')
            self.assertEqual(url, expected)
            return rgi_f

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            rgi = utils.get_rgi_intersects_dir()

        assert os.path.isdir(rgi)
        assert os.path.exists(os.path.join(rgi,
                                           'Intersects_OGGM_Manifest.txt'))

        # Make a fake RGI file
        rgi_dir = os.path.join(self.dldir, 'rgi60')
        utils.mkdir(rgi_dir)
        make_fake_zipdir(os.path.join(rgi_dir, 'RGI_V6_Intersects'),
                         fakefile='Intersects_OGGM_Manifest.txt')
        rgi_f = make_fake_zipdir(rgi_dir)

        def down_check(url, cache_name=None, reset=False):
            expected = ('https://www.dropbox.com/s/vawryxl8lkzxowu/' +
                        'RGI_V6_Intersects.zip?dl=1')
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

    def test_srtm(self):

        # Make a fake topo file
        tf = make_fake_zipdir(os.path.join(self.dldir, 'srtm_39_03'),
                              fakefile='srtm_39_03.tif')

        def down_check(url, cache_name=None, reset=False):
            expected = 'http://droppr.org/srtm/v4.1/6_5x5_TIFs/srtm_39_03.zip'
            self.assertEqual(url, expected)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([11.3, 11.3], [47.1, 47.1])

        assert os.path.exists(of[0])
        assert source == 'SRTM'

    def test_dem3(self):

        # GEt the path to the file before we mess around

        tf = utils.get_demo_file('T10.zip')
        def down_check(url, cache_name=None, reset=False):
            expected = 'http://viewfinderpanoramas.org/dem3/T10.zip'
            self.assertEqual(url, expected)
            return tf

        with FakeDownloadManager('_progress_urlretrieve', down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [76.8, 76.8])

        assert os.path.exists(of[0])
        assert source == 'DEM3'

    def test_ramp(self):

        def down_check(url):
            expected = 'AntarcticDEM_wgs84.tif'
            self.assertEqual(url, expected)
            return 'yo'

        with FakeDownloadManager('_download_alternate_topo_file', down_check):
            of, source = utils.get_topo_file([-120.2, -120.2], [-88, -88],
                                             rgi_region=19)

        assert of[0] == 'yo'
        assert source == 'RAMP'

    def test_gimp(self):

        def down_check(url):
            expected = 'gimpdem_90m.tif'
            self.assertEqual(url, expected)
            return 'yo'

        with FakeDownloadManager('_download_alternate_topo_file', down_check):
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

        f = utils.get_demo_file('Hintereisferner.shp')
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

    def test_asterzone(self):

        z, u = utils.aster_zone(lon_ex=[137.5, 137.5],
                                lat_ex=[-72.5, -72.5])
        self.assertTrue(len(z) == 1)
        self.assertTrue(len(u) == 1)
        self.assertEqual('S73E137', z[0])
        self.assertEqual('S75E135', u[0])

        z, u= utils.aster_zone(lon_ex=[-95.5, -95.5],
                               lat_ex=[30.5, 30.5])
        self.assertTrue(len(z) == 1)
        self.assertTrue(len(u) == 1)
        self.assertEqual('N30W096', z[0])
        self.assertEqual('N30W100', u[0])

        z, u= utils.aster_zone(lon_ex=[-96.5, -95.5],
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

        l = cfg.get_lru_handler(self.dldir, maxsize=3, ending='.txt')
        assert os.path.exists(f1)
        assert os.path.exists(f2)
        assert os.path.exists(f3)

        l = cfg.get_lru_handler(self.dldir, maxsize=2, ending='.txt')
        assert not os.path.exists(f1)
        assert os.path.exists(f2)
        assert os.path.exists(f3)

    @is_download
    def test_srtmdownload(self):

        # this zone does exist and file should be small enough for download
        zone = '68_11'
        fp = utils._download_srtm_file(zone)
        self.assertTrue(os.path.exists(fp))
        fp = utils._download_srtm_file(zone)
        self.assertTrue(os.path.exists(fp))

    @is_download
    def test_srtmdownloadfails(self):

        # this zone does not exist
        zone = '41_20'
        self.assertTrue(utils._download_srtm_file(zone) is None)

    @is_download
    def test_asterdownload(self):

        # this zone does exist and file should be small enough for download
        zone = 'S73E137'
        unit = 'S75E135'
        fp = utils._download_aster_file(zone, unit)
        self.assertTrue(os.path.exists(fp))

    @is_download
    def test_gimp(self):
        fp, z = utils.get_topo_file([], [], rgi_region=5)
        self.assertTrue(os.path.exists(fp[0]))
        self.assertEqual(z, 'GIMP')

    @is_download
    def test_iceland(self):
        fp, z = utils.get_topo_file([-20, -20], [65, 65])
        self.assertTrue(os.path.exists(fp[0]))

    @is_download
    def test_asterdownloadfails(self):

        # this zone does not exist
        zone = 'bli'
        unit = 'S75E135'
        self.assertTrue(utils._download_aster_file(zone, unit) is None)

    @is_download
    def test_alternatedownload(self):

        # this is a simple file
        fp = utils._download_alternate_topo_file('iceland.tif')
        self.assertTrue(os.path.exists(fp))

    @is_download
    def test_download_cru(self):

        tmp = cfg.PATHS['cru_dir']
        cfg.PATHS['cru_dir'] = os.path.join(self.dldir, 'cru_extract')

        of = utils.get_cru_file('tmp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['cru_dir'] = tmp

    @is_download
    def test_download_rgi(self):

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_extract')

        of = utils.get_rgi_dir(version='5')
        of = os.path.join(of, '01_rgi50_Alaska', '01_rgi50_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @is_download
    def test_download_rgi6(self):

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_extract')

        of = utils.get_rgi_dir(version='6')
        of = os.path.join(of, '01_rgi60_Alaska', '01_rgi60_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @is_download
    def test_download_rgi_intersects(self):

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = os.path.join(self.dldir, 'rgi_extract')

        of = utils.get_rgi_intersects_dir()
        of = os.path.join(of, '01_rgi50_Alaska',
                          'intersects_01_rgi50_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @is_download
    def test_download_dem3_viewpano(self):

        # this zone does exist and file should be small enough for download
        zone = 'L32'
        fp = utils._download_dem3_viewpano(zone)
        self.assertTrue(os.path.exists(fp))
        zone = 'U44'
        fp = utils._download_dem3_viewpano(zone)
        self.assertTrue(os.path.exists(fp))

    @is_download
    def test_download_dem3_viewpano_fails(self):

        # this zone does not exist
        zone = 'dummy'
        fp = utils._download_dem3_viewpano(zone)
        self.assertTrue(fp is None)

    @is_download
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
