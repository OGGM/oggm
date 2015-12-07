from __future__ import division
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)
import unittest
import os

import salem

from oggm import utils


class TestDataFiles(unittest.TestCase):

    def test_download(self):

        f = utils.get_demo_file('Hintereisferner.shp')
        self.assertTrue(os.path.exists(f))

        sh = salem.utils.read_shapefile(f)
        self.assertTrue(hasattr(sh, 'geometry'))