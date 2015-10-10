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
from oggm.prepro import gis, centerlines
import oggm.conf as cfg
from oggm.utils import get_demo_file
import logging
from xml.dom import minidom
import salem

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))


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


class TestCenterlines(unittest.TestCase):

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
        cfg.params['border'] = 10

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

    def test_filter_heads(self):

        f = get_demo_file('glacier.svg')

        coords = read_svgcoords(f)
        polygon = shpg.Polygon(coords)

        hidx = np.array([3, 9, 80, 92, 108, 116, 170, len(coords)-12])
        heads = [shpg.Point(*c) for c in coords[hidx]]
        heads_height = np.array([200, 210, 1000., 900, 1200, 1400, 1300, 250])
        radius = 25

        _heads, _ = centerlines._filter_heads(heads, heads_height, radius,
                                            polygon)
        _headsi, _ = centerlines._filter_heads(heads[::-1], heads_height[
                                                          ::-1], radius, polygon)

        self.assertEqual(_heads, _headsi[::-1])
        self.assertEqual(_heads, [heads[h] for h in [2,5,6,7]])

    def test_centerlines(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        rgidf = gpd.GeoDataFrame.from_file(hef_file)

        # loop because for some reason indexing wont work
        for index, entity in rgidf.iterrows():
            gdir = cfg.GlacierDir(entity, base_dir=self.testdir)
            gis.define_glacier_region(gdir, entity)
            gis.glacier_masks(gdir)
            centerlines.compute_centerlines(gdir)

        for div_id in gdir.divide_ids:
            cls = gdir.read_pickle('centerlines', div_id=div_id)
            for cl in cls:
                for j, ip, ob in zip(cl.inflow_indices, cl.inflow_points, cl.inflows):
                    self.assertTrue(cl.line.coords[j] == ip.coords[0])
                    self.assertTrue(ob.flows_to_point.coords[0] == ip.coords[0])
                    self.assertTrue(cl.line.coords[ob.flows_to_indice] == ip.coords[0])

        lens = [len(gdir.read_pickle('centerlines', div_id=i)) for i in [1,2,3]]
        self.assertTrue(sorted(lens) == [1, 1, 3])

    def test_downstream(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        rgidf = gpd.GeoDataFrame.from_file(hef_file)

        # loop because for some reason indexing wont work
        for index, entity in rgidf.iterrows():
            gdir = cfg.GlacierDir(entity, base_dir=self.testdir)
            gis.define_glacier_region(gdir, entity)
            gis.glacier_masks(gdir)
            centerlines.compute_centerlines(gdir)
            centerlines.compute_downstream_lines(gdir)

    def test_baltoro_centerlines(self):

        cfg.params['border'] = 2
        cfg.input['srtm_file'] =  get_demo_file('baltoro_srtm_clip.tif')

        hef_file = get_demo_file('baltoro_wgs84.shp')
        rgidf = gpd.GeoDataFrame.from_file(hef_file)

        kienholz_file = get_demo_file('centerlines_baltoro_wgs84.shp')
        kdf = gpd.read_file(kienholz_file)

        # loop because for some reason indexing wont work
        for index, entity in rgidf.iterrows():
            gdir = cfg.GlacierDir(entity, base_dir=self.testdir)
            gis.define_glacier_region(gdir, entity)
            gis.glacier_masks(gdir)
            centerlines.compute_centerlines(gdir)

        my_mask = np.zeros((gdir.grid.ny, gdir.grid.nx), dtype=np.uint8)
        cls = gdir.read_pickle('centerlines', div_id=1)
        for cl in cls:
            ext_yx = tuple(reversed(cl.line.xy))
            my_mask[ext_yx] = 1

        # Transform
        kien_mask = np.zeros((gdir.grid.ny, gdir.grid.nx), dtype=np.uint8)
        from shapely.ops import transform
        for index, entity in kdf.iterrows():
            def proj(lon, lat):
                return salem.transform_proj(salem.wgs84, gdir.grid.proj,
                                            lon, lat)
            kgm = transform(proj, entity.geometry)

            # Interpolate shape to a regular path
            e_line = []
            for distance in np.arange(0.0, kgm.length, gdir.grid.dx):
                e_line.append(*kgm.interpolate(distance).coords)
            kgm = shpg.LineString(e_line)

            # Transform geometry into grid coordinates
            def proj(x, y):
                return gdir.grid.transform(x, y, crs=gdir.grid.proj)
            kgm = transform(proj, kgm)

            # Rounded nearest pix
            project = lambda x, y: (np.rint(x).astype(np.int64),
                            np.rint(y).astype(np.int64))

            kgm = transform(project, kgm)

            ext_yx = tuple(reversed(kgm.xy))
            kien_mask[ext_yx] = 1

        # We test the Heidke Skill score of our predictions
        rest = kien_mask + 2 * my_mask
        # gr.plot_array(rest)
        na = len(np.where(rest == 3)[0])
        nb = len(np.where(rest == 2)[0])
        nc = len(np.where(rest == 1)[0])
        nd = len(np.where(rest == 0)[0])
        denom = np.float64((na+nc)*(nd+nc)+(na+nb)*(nd+nb))
        hss = np.float64(2.) * ((na*nd)-(nb*nc)) / denom
        self.assertTrue(hss > 0.53)