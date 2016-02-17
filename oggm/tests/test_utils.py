from __future__ import division
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)
import unittest
import os

import salem

from oggm import utils
from oggm import cfg

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_topo')
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

class TestDataFiles(unittest.TestCase):

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