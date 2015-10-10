from __future__ import absolute_import, division, unicode_literals

import unittest
import os
import pickle

import shapely.geometry as shpg
import numpy as np
import shutil
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4
import multiprocessing as mp

# Local imports
from oggm.prepro import gis
import oggm.conf as cfg
from oggm.utils import get_demo_file
import logging
from xml.dom import minidom

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))

do_plot = False

def read_svgcoords(svg_file):
    """Get the vertices coordinates out of a SVG file"""
    doc = minidom.parse(svg_file)
    coords = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    doc.unlink()
    _, _, coords = coords[0].partition('C')
    x = []
    y = []
    for c in coords.split(' '):
        if c == '': continue
        c = c.split(',')
        x.append(np.float(c[0]))
        y.append(np.float(c[1]))
    x.append(x[0])
    y.append(y[0])

    return np.rint(np.asarray((x, y)).T).astype(np.int64)


class TestGIS(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(current_dir, 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
        cfg.input['srtm_file'] = get_demo_file('hef_srtm.tif')

        logging.getLogger("Fiona").setLevel(logging.WARNING)
        logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_define_region(self):
        """Very basic test to see if the transform went well"""

        hef_file = get_demo_file('Hintereisferner.shp')
        rgidf = gpd.GeoDataFrame.from_file(hef_file)

        # loop because for some reason indexing wont work
        for index, entity in rgidf.iterrows():
            gdir = cfg.GlacierDir(entity, base_dir=self.testdir)
            gis.define_glacier_region(gdir, entity)

        tdf = gpd.GeoDataFrame.from_file(gdir.get_filepath('outlines'))
        myarea = tdf.geometry.area * 10**-6
        np.testing.assert_allclose(myarea, np.float(tdf['AREA']), rtol=1e-2)

    def test_glacier_masks(self):
        """Again, easy test.

        The GIS was double checked externally with IDL.
        """

        hef_file = get_demo_file('Hintereisferner.shp')
        rgidf = gpd.GeoDataFrame.from_file(hef_file)

        # loop because for some reason indexing wont work
        for index, entity in rgidf.iterrows():
            gdir = cfg.GlacierDir(entity, base_dir=self.testdir)
            gis.define_glacier_region(gdir, entity)
            gis.glacier_masks(gdir)

        nc = netCDF4.Dataset(gdir.get_filepath('grids'))
        area = np.sum(nc.variables['glacier_mask'][:] * gdir.grid.dx**2) * 10**-6
        np.testing.assert_allclose(area,gdir.glacier_area, rtol=1e-1)
