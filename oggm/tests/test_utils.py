from __future__ import division
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)
import unittest
import os
import shutil

import salem

from oggm.tests import is_download
from oggm import utils
from oggm import cfg

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_download')
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)


class TestDataFiles(unittest.TestCase):

    def setUp(self):
        cfg.PATHS['topo_dir'] = TEST_DIR

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
            os.makedirs(TEST_DIR)

    def tearDown(self):
        del cfg.PATHS['topo_dir']

    def test_download(self):

        f = utils.get_demo_file('Hintereisferner.shp')
        self.assertTrue(os.path.exists(f))

        sh = salem.utils.read_shapefile(f)
        self.assertTrue(hasattr(sh, 'geometry'))

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
    def test_asterdownloadfails(self):

        # this zone does exist and file should be small enough for download
        zone = 'bli'
        unit = 'S75E135'
        self.assertTrue(utils._download_aster_file(zone, unit) is None)

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
