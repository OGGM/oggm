from __future__ import division
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)
import unittest
import os
import shutil

import salem
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from oggm.tests import is_download
from oggm import utils
from oggm import cfg

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_download')
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)


class TestFuncs(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_signchange(self):
        ts = pd.Series([-2., -1., 1., 2., 3], index=np.arange(5))
        sc = utils.signchange(ts)
        assert_array_equal(sc, [0, 0, 1, 0, 0])
        ts = pd.Series([-2., -1., 1., 2., 3][::-1], index=np.arange(5))
        sc = utils.signchange(ts)
        assert_array_equal(sc, [0, 0, 0, 1, 0])

    def test_year_to_date(self):

        r = utils.year_to_date(0)
        self.assertEqual(r, (0, 1))

        y, m = utils.year_to_date([0, 1])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(m, [1, 1])

        y, m = utils.year_to_date([0.00001, 1.00001])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(m, [1, 1])

        y, m = utils.year_to_date([0.99999, 1.99999])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(m, [12, 12])

        yr = 1998 + cfg.CUMSEC_IN_MONTHS[2] / cfg.SEC_IN_YEAR
        r = utils.year_to_date(yr)
        self.assertEqual(r, (1998, 4))

        yr = 1998 + (cfg.CUMSEC_IN_MONTHS[2] - 1) / cfg.SEC_IN_YEAR
        r = utils.year_to_date(yr)
        self.assertEqual(r, (1998, 3))

    def test_date_to_year(self):

        r = utils.date_to_year(0, 1)
        self.assertEqual(r, 0)

        r = utils.date_to_year(1, 1)
        self.assertEqual(r, 1)

        r = utils.date_to_year([0, 1], [1, 1])
        np.testing.assert_array_equal(r, [0, 1])

        yr = utils.date_to_year([1998, 1998], [6, 7])
        y, m = utils.year_to_date(yr)
        np.testing.assert_array_equal(y, [1998, 1998])
        np.testing.assert_array_equal(m, [6, 7])

        yr = utils.date_to_year([1998, 1998], [2, 3])
        y, m = utils.year_to_date(yr)
        np.testing.assert_array_equal(y, [1998, 1998])
        np.testing.assert_array_equal(m, [2, 3])

        time = pd.date_range('1/1/1800', periods=300*12, freq='MS')
        yr = utils.date_to_year(time.year, time.month)
        y, m = utils.year_to_date(yr)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        myr = utils.monthly_timeseries(1800, 2099)
        y, m = utils.year_to_date(myr)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        myr = utils.monthly_timeseries(1800, ny=300)
        y, m = utils.year_to_date(myr)
        np.testing.assert_array_equal(y, time.year)
        np.testing.assert_array_equal(m, time.month)

        with self.assertRaises(ValueError):
            utils.monthly_timeseries(1)


class TestDataFiles(unittest.TestCase):

    def setUp(self):
        cfg.PATHS['topo_dir'] = TEST_DIR

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
            os.makedirs(TEST_DIR)

    def tearDown(self):
        del cfg.PATHS['topo_dir']

    def test_download_demo_files(self):

        f = utils.get_demo_file('Hintereisferner.shp')
        self.assertTrue(os.path.exists(f))

        sh = salem.read_shapefile(f)
        self.assertTrue(hasattr(sh, 'geometry'))

        # Data files
        cfg.initialize()

        lf, df = utils.get_leclercq_files()
        self.assertTrue(os.path.exists(lf))

        lf, df = utils.get_wgms_files()
        self.assertTrue(os.path.exists(lf))

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

        test_loc = {'ISL': [-25., -12., 63., 67.],      # Iceland
                        'SVALBARD': [10., 34., 76., 81.],
                        'JANMAYEN': [-10., -7., 70., 72.],
                        'FJ': [36., 66., 79., 82.],         # Franz Josef Land
                        'FAR': [-8., -6., 61., 63.],        # Faroer
                        'BEAR': [18., 20., 74., 75.],       # Bear Island
                        'SHL': [-3., 0., 60., 61.],         # Shetland
                        '01-15': [-180., -91., -90, -60.],  # Antarctica tiles as UTM zones, FILES ARE LARGE!!!!!
                        '16-30': [-91., -1., -90., -60.],
                        '31-45': [-1., 89., -90., -60.],
                        '46-60': [89., 189., -90., -60.],
                        'GL-North': [-78., -11., 75., 84.],  # Greenland tiles
                        'GL-West': [-68., -42., 64., 76.],
                        'GL-South': [-52., -40., 59., 64.],
                        'GL-East': [-42., -17., 64., 76.]}
        # special names
        for key in test_loc:
            z = utils.dem3_viewpano_zone([test_loc[key][0], test_loc[key][1]], [test_loc[key][2], test_loc[key][3]],
                                       test_loc)
            self.assertTrue(len(z) == 1)

            self.assertEqual(key, z[0])

        # weired Antarctica tile
        z = utils.dem3_viewpano_zone([-91., -90.], [-72., -68.], test_loc)
        self.assertTrue(len(z) == 1)
        self.assertEqual('SR15', z[0])

        # normal tile
        z = utils.dem3_viewpano_zone([-179., -178.], [65., 65.], test_loc)
        self.assertTrue(len(z) == 1)
        self.assertEqual('Q01', z[0])

        # Alps
        ref = sorted(['K31', 'K32', 'K33', 'L31', 'L32', 'L33', 'M31', 'M32', 'M33'])
        z = utils.dem3_viewpano_zone([6, 14], [41, 48], test_loc)
        self.assertTrue(len(z) == 9)
        self.assertEqual(ref, z)

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

    def test_download_dem3_viewpano(self):

        # this zone does exist and file should be small enough for download
        zone = 'L32'
        fp = utils._download_dem3_viewpano(zone, {})
        self.assertTrue(os.path.exists(fp))
        zone = 'U44'
        fp = utils._download_dem3_viewpano(zone, {})
        self.assertTrue(os.path.exists(fp))


    def test_download_dem3_viewpano_fails(self):

        # this zone does not exist
        zone = 'SZ20'
        self.assertTrue(utils._download_dem3_viewpano(zone, {}) is None)

    @is_download
    def test_asterdownload(self):

        # this zone does exist and file should be small enough for download
        zone = 'S73E137'
        unit = 'S75E135'
        fp = utils._download_aster_file(zone, unit)
        self.assertTrue(os.path.exists(fp))

    @is_download
    def test_gimp(self):
        fp, z = utils.get_topo_file([], [], region=5)
        self.assertTrue(os.path.exists(fp))

    @is_download
    def test_iceland(self):
        fp, z = utils.get_topo_file([-20, -20], [65, 65])
        self.assertTrue(os.path.exists(fp))

    @is_download
    def test_asterdownloadfails(self):

        # this zone does exist and file should be small enough for download
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

        cfg.initialize()

        tmp = cfg.PATHS['cru_dir']
        cfg.PATHS['cru_dir'] = TEST_DIR

        of = utils.get_cru_file('tmp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['cru_dir'] = tmp

    @is_download
    def test_download_rgi(self):

        cfg.initialize()

        tmp = cfg.PATHS['rgi_dir']
        cfg.PATHS['rgi_dir'] = TEST_DIR

        of = utils.get_rgi_dir()
        of = os.path.join(of, '01_rgi50_Alaska', '01_rgi50_Alaska.shp')
        self.assertTrue(os.path.exists(of))

        cfg.PATHS['rgi_dir'] = tmp

    @is_download
    def test_download_dem3_viewpano(self):

        # this zone does exist and file should be small enough for download
        zone = 'L32'
        fp = utils._download_dem3_viewpano(zone, {})
        self.assertTrue(os.path.exists(fp))
        zone = 'U44'
        fp = utils._download_dem3_viewpano(zone, {})
        self.assertTrue(os.path.exists(fp))

    @is_download
    def test_download_dem3_viewpano_fails(self):

        # this zone does not exist
        zone = 'SZ20'
        self.assertTrue(utils._download_dem3_viewpano(zone, {}) is None)
