from os import path
import warnings

import oggm
import oggm.utils

warnings.filterwarnings("once", category=DeprecationWarning)

import unittest
import os
import shutil

import shapely.geometry as shpg
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4
import salem
import xarray as xr

# Local imports
from oggm.core import (gis, inversion, climate, centerlines, flowline,
                       massbalance)
import oggm.cfg as cfg
from oggm import utils
from oggm.utils import get_demo_file, tuple2int
from oggm.tests import is_slow, RUN_PREPRO_TESTS
from oggm.tests.funcs import get_test_dir, patch_url_retrieve_github
from oggm import workflow

# do we event want to run the tests?
if not RUN_PREPRO_TESTS:
    raise unittest.SkipTest('Skipping all prepro tests.')

_url_retrieve = None


def setup_module(module):
    module._url_retrieve = utils._urlretrieve
    utils._urlretrieve = patch_url_retrieve_github


def teardown_module(module):
    utils._urlretrieve = module._url_retrieve


def read_svgcoords(svg_file):
    """Get the vertices coordinates out of a SVG file"""

    from xml.dom import minidom
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
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['working_dir'] = self.testdir

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_define_region(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        extent = gdir.extent_ll

        tdf = gpd.GeoDataFrame.from_file(gdir.get_filepath('outlines'))
        myarea = tdf.geometry.area * 10**-6
        np.testing.assert_allclose(myarea, np.float(tdf['AREA']), rtol=1e-2)
        self.assertFalse(os.path.exists(gdir.get_filepath('intersects')))

        # From string
        gdir = oggm.GlacierDirectory(gdir.rgi_id, base_dir=self.testdir)
        # This is not guaranteed to be equal because of projection issues
        np.testing.assert_allclose(extent, gdir.extent_ll, atol=1e-5)

    def test_divides_as_glaciers(self):

        hef_rgi = gpd.GeoDataFrame.from_file(get_demo_file('divides_alps.shp'))
        hef_rgi = hef_rgi.loc[hef_rgi.RGIId == 'RGI50-11.00897']

        # Rename the RGI ID
        hef_rgi['RGIId'] = ['RGI50-11.00897' + d for d in
                            ['_d01', '_d02', '_d03']]

        # Just check that things are working
        gdirs = workflow.init_glacier_regions(hef_rgi)
        workflow.gis_prepro_tasks(gdirs)

        assert gdirs[0].rgi_id == 'RGI50-11.00897_d01'
        assert gdirs[-1].rgi_id == 'RGI50-11.00897_d03'

    def test_dx_methods(self):
        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

        # Test fixed method
        cfg.PARAMS['grid_dx_method'] = 'fixed'
        cfg.PARAMS['fixed_dx'] = 50
        gis.define_glacier_region(gdir, entity=entity)
        mygrid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
        np.testing.assert_allclose(np.abs(mygrid.dx), 50.)

        # Test linear method
        cfg.PARAMS['grid_dx_method'] = 'linear'
        cfg.PARAMS['d1'] = 5.
        cfg.PARAMS['d2'] = 10.
        cfg.PARAMS['dmax'] = 100.
        gis.define_glacier_region(gdir, entity=entity)
        targetdx = np.rint(5. * gdir.rgi_area_km2 + 10.)
        targetdx = np.clip(targetdx, 10., 100.)
        mygrid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
        np.testing.assert_allclose(mygrid.dx, targetdx)

        # Test square method
        cfg.PARAMS['grid_dx_method'] = 'square'
        cfg.PARAMS['d1'] = 5.
        cfg.PARAMS['d2'] = 10.
        cfg.PARAMS['dmax'] = 100.
        gis.define_glacier_region(gdir, entity=entity)
        targetdx = np.rint(5. * np.sqrt(gdir.rgi_area_km2) + 10.)
        targetdx = np.clip(targetdx, 10., 100.)
        mygrid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
        np.testing.assert_allclose(mygrid.dx, targetdx)

    def test_repr(self):
        from textwrap import dedent

        expected = dedent("""\
        <oggm.GlacierDirectory>
          RGI id: RGI40-11.00897
          Region: 11: Central Europe
          Subregion: 11-01: Alps
          Glacier type: Not assigned
          Terminus type: Land-terminating
          Area: 8.036 km2
          Lon, Lat: (10.7584, 46.8003)
          Grid (nx, ny): (159, 114)
          Grid (dx, dy): (50.0, -50.0)
        """)

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        self.assertEqual(gdir.__repr__(), expected)

    def test_glacierdir(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)

        # this should simply run
        mygdir = oggm.GlacierDirectory(entity.RGIID, base_dir=self.testdir)

    def test_glacier_masks(self):

        # The GIS was double checked externally with IDL.
        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
            area = np.sum(nc.variables['glacier_mask'][:] * gdir.grid.dx**2)
            np.testing.assert_allclose(area*10**-6, gdir.rgi_area_km2,
                                       rtol=1e-1)

    def test_intersects(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        self.assertTrue(os.path.exists(gdir.get_filepath('intersects')))


class TestCenterlines(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PARAMS['border'] = 10

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

    def test_mask_to_polygon(self):
        from oggm.core.centerlines import _mask_to_polygon

        mask = np.zeros((5, 5))
        mask[1, 1] = 1
        p1, p2 = _mask_to_polygon(mask)
        assert p1 == p2

        mask = np.zeros((5, 5))
        mask[1:-1, 1:-1] = 1
        p1, p2 = _mask_to_polygon(mask)
        assert p1 == p2

        mask = np.zeros((5, 5))
        mask[1:-1, 1:-1] = 1
        mask[2, 2] = 0
        p1, _ = _mask_to_polygon(mask)
        assert len(p1.interiors) == 1
        assert p1.exterior == p2.exterior
        for i_line in p1.interiors:
            assert p2.contains(i_line)

        n = 30
        for i in range(n):
            mask = np.zeros((n, n))
            mask[1:-1, 1:-1] = 1
            _, p2 = _mask_to_polygon(mask)
            for i in range(n*2):
                mask[np.random.randint(2, n-2), np.random.randint(2, n-2)] = 0
            p1, _ = _mask_to_polygon(mask)
            assert len(p1.interiors) > 1
            assert p1.exterior == p2.exterior
            for i_line in p1.interiors:
                assert p2.contains(i_line)

    def test_centerlines(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)

        cls = gdir.read_pickle('centerlines')
        for cl in cls:
            for j, ip, ob in zip(cl.inflow_indices, cl.inflow_points,
                                 cl.inflows):
                self.assertEqual(cl.line.coords[j], ip.coords[0])
                self.assertEqual(ob.flows_to_point.coords[0],
                                 ip.coords[0])
                self.assertEqual(cl.line.coords[ob.flows_to_indice],
                                 ip.coords[0])

        self.assertEqual(len(cls), 3)

    def test_downstream(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)

        d = gdir.read_pickle('downstream_line')
        cl = gdir.read_pickle('inversion_flowlines')[-1]
        self.assertEqual(
            len(d['full_line'].coords) - len(d['downstream_line'].coords),
            cl.nx)

    def test_downstream_bedshape(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        default_b = cfg.PARAMS['border']
        cfg.PARAMS['border'] = 80

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)

        out = gdir.read_pickle('downstream_line')
        for o, h in zip(out['bedshapes'], out['surface_h']):
            assert np.all(np.isfinite(o))
            assert np.all(np.isfinite(h))

        tpl = gdir.read_pickle('inversion_flowlines')[-1]
        c = gdir.read_pickle('downstream_line')['downstream_line']
        c = centerlines.Centerline(c, dx=tpl.dx)

        # Independant reproduction for a few points
        o = out['bedshapes']
        i0s = [0, 5, 10, 15, 20]
        for i0 in i0s:
            wi = 11
            i0 = int(i0)
            cur = c.line.coords[i0]
            n1, n2 = c.normals[i0]
            l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                 shpg.Point(cur + wi / 2. * n2)])
            from oggm.core.centerlines import line_interpol
            from scipy.interpolate import RegularGridInterpolator
            points = line_interpol(l, 0.5)
            with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
                topo = nc.variables['topo_smoothed'][:]
                x = nc.variables['x'][:]
                y = nc.variables['y'][:]
            xy = (np.arange(0, len(y) - 0.1, 1), np.arange(0, len(x) - 0.1, 1))
            interpolator = RegularGridInterpolator(xy, topo)

            zref = [interpolator((p.xy[1][0], p.xy[0][0])) for p in points]

            myx = np.arange(len(points))
            myx = (myx - np.argmin(zref)) / 2 * gdir.grid.dx
            myz = o[i0] * myx**2 + np.min(zref)

            # In this case the fit is simply very good (plot it if you want!)
            assert utils.rmsd(zref, myz) < 20

        cfg.PARAMS['border'] = default_b

    @is_slow
    def test_baltoro_centerlines(self):

        cfg.PARAMS['border'] = 2
        cfg.PARAMS['dmax'] = 100
        cfg.PATHS['dem_file'] = get_demo_file('baltoro_srtm_clip.tif')

        b_file = get_demo_file('baltoro_wgs84.shp')
        entity = gpd.GeoDataFrame.from_file(b_file).iloc[0]

        kienholz_file = get_demo_file('centerlines_baltoro_wgs84.shp')
        kdf = gpd.read_file(kienholz_file)

        # add fake attribs
        entity.RGIID = 'RGI40-00.00000'
        entity.O1REGION = 0
        entity.BGNDATE = 0
        entity.NAME = 'Baltoro'
        entity.GLACTYPE = '0000'
        entity.O1REGION = '01'
        entity.O2REGION = '01'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)

        my_mask = np.zeros((gdir.grid.ny, gdir.grid.nx), dtype=np.uint8)
        cls = gdir.read_pickle('centerlines')
        for cl in cls:
            x, y = tuple2int(cl.line.xy)
            my_mask[y, x] = 1

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

            x, y = tuple2int(kgm.xy)
            kien_mask[y, x] = 1

        # We test the Heidke Skill score of our predictions
        rest = kien_mask + 2 * my_mask
        # gr.plot_array(rest)
        na = len(np.where(rest == 3)[0])
        nb = len(np.where(rest == 2)[0])
        nc = len(np.where(rest == 1)[0])
        nd = len(np.where(rest == 0)[0])
        denom = np.float((na+nc)*(nd+nc)+(na+nb)*(nd+nb))
        hss = np.float(2.) * ((na*nd)-(nb*nc)) / denom
        if cfg.PARAMS['grid_dx_method'] == 'linear':
            self.assertTrue(hss > 0.53)
        if cfg.PARAMS['grid_dx_method'] == 'fixed':  # quick fix
            self.assertTrue(hss > 0.41)


class TestGeometry(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PARAMS['border'] = 10

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_catchment_area(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.catchment_area(gdir)

        cis = gdir.read_pickle('catchment_indices')

        # The catchment area must be as big as expected
        with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
            mask = nc.variables['glacier_mask'][:]

        mymask_a = mask * 0
        mymask_b = mask * 0
        for i, ci in enumerate(cis):
            mymask_a[tuple(ci.T)] += 1
            mymask_b[tuple(ci.T)] = i+1
        self.assertTrue(np.max(mymask_a) == 1)
        np.testing.assert_allclose(mask, mymask_a)

    def test_flowlines(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)

        cls = gdir.read_pickle('inversion_flowlines')
        for cl in cls:
            for j, ip, ob in zip(cl.inflow_indices, cl.inflow_points,
                                 cl.inflows):
                self.assertEqual(cl.line.coords[j], ip.coords[0])
                self.assertEqual(ob.flows_to_point.coords[0], ip.coords[0])
                self.assertEqual(cl.line.coords[ob.flows_to_indice],
                                 ip.coords[0])

        self.assertEqual(len(cls), 3)

        x, y = map(np.array, cls[0].line.xy)
        dis = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        np.testing.assert_allclose(dis * 0 + cfg.PARAMS['flowline_dx'], dis,
                                   rtol=0.01)

    def test_geom_width(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)

    def test_width(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        area = 0.
        otherarea = 0.
        hgt = []
        harea = []

        cls = gdir.read_pickle('inversion_flowlines')
        for cl in cls:
            harea.extend(list(cl.widths * cl.dx))
            hgt.extend(list(cl.surface_h))
            area += np.sum(cl.widths * cl.dx)
        with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
            otherarea += np.sum(nc.variables['glacier_mask'][:])

        with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo_smoothed'][:]
        rhgt = topo[np.where(mask)][:]

        tdf = gpd.GeoDataFrame.from_file(gdir.get_filepath('outlines'))
        np.testing.assert_allclose(area, otherarea, rtol=0.1)
        area *= (gdir.grid.dx) ** 2
        otherarea *= (gdir.grid.dx) ** 2
        np.testing.assert_allclose(area * 10**-6, np.float(tdf['AREA']),
                                   rtol=1e-4)

        # Check for area distrib
        bins = np.arange(utils.nicenumber(np.min(hgt), 50, lower=True),
                         utils.nicenumber(np.max(hgt), 50)+1,
                         50.)
        h1, b = np.histogram(hgt, weights=harea, density=True, bins=bins)
        h2, b = np.histogram(rhgt, density=True, bins=bins)
        self.assertTrue(utils.rmsd(h1*100*50, h2*100*50) < 1)

        # Check that utility function is doing what is expected
        hh, ww = gdir.get_inversion_flowline_hw()
        new_area = np.sum(ww * cl.dx * gdir.grid.dx)
        np.testing.assert_allclose(new_area * 10**-6, np.float(tdf['AREA']))

    def test_nodivides_correct_slope(self):

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['border'] = 40

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)

        fls = gdir.read_pickle('inversion_flowlines')
        min_slope = np.deg2rad(cfg.PARAMS['min_slope'])
        for fl in fls:
            dx = fl.dx * gdir.grid.dx
            slope = np.arctan(-np.gradient(fl.surface_h, dx))
            self.assertTrue(np.all(slope >= min_slope))


class TestClimate(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_prepro')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.testdir_cru = os.path.join(get_test_dir(), 'tmp_prepro_cru')
        if not os.path.exists(self.testdir_cru):
            os.makedirs(self.testdir_cru)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['border'] = 10
        cfg.PARAMS['run_mb_calibration'] = True

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)
        shutil.rmtree(self.testdir_cru)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)
        shutil.rmtree(self.testdir_cru)
        os.makedirs(self.testdir_cru)

    def test_distribute_climate(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)

        ci = gdir.read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1802)
        self.assertEqual(ci['hydro_yr_1'], 2003)

        with netCDF4.Dataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        with netCDF4.Dataset(os.path.join(gdir.dir, 'climate_monthly.nc')) as nc_r:
            self.assertTrue(ref_h == nc_r.ref_hgt)
            np.testing.assert_allclose(ref_t, nc_r.variables['temp'][:])
            np.testing.assert_allclose(ref_p, nc_r.variables['prcp'][:])

    def test_distribute_climate_grad(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        cfg.PARAMS['temp_use_local_gradient'] = True
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)

        ci = gdir.read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1802)
        self.assertEqual(ci['hydro_yr_1'], 2003)

        with netCDF4.Dataset(gdir.get_filepath('climate_monthly')) as nc_r:
            grad = nc_r.variables['grad'][:]
            assert np.std(grad) > 0.0001

        cfg.PARAMS['temp_use_local_gradient'] = False

    def test_distribute_climate_parallel(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)
        climate.process_custom_climate_data(gdir)

        ci = gdir.read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1802)
        self.assertEqual(ci['hydro_yr_1'], 2003)

        with netCDF4.Dataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        with netCDF4.Dataset(os.path.join(gdir.dir, 'climate_monthly.nc')) as nc_r:
            self.assertTrue(ref_h == nc_r.ref_hgt)
            np.testing.assert_allclose(ref_t, nc_r.variables['temp'][:])
            np.testing.assert_allclose(ref_p, nc_r.variables['prcp'][:])

    def test_distribute_climate_cru(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir_cru)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)

        climate.process_histalp_nonparallel([gdirs[0]])
        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cru_dir = os.path.dirname(cru_dir)
        cfg.PATHS['climate_file'] = ''
        cfg.PATHS['cru_dir'] = cru_dir
        climate.process_cru_data(gdirs[1])
        cfg.PATHS['cru_dir'] = ''
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        ci = gdir.read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1902)
        self.assertEqual(ci['hydro_yr_1'], 2014)

        gdh = gdirs[0]
        gdc = gdirs[1]
        with xr.open_dataset(os.path.join(gdh.dir, 'climate_monthly.nc')) as nc_h:
            with xr.open_dataset(os.path.join(gdc.dir, 'climate_monthly.nc')) as nc_c:
                # put on the same altitude
                # (using default gradient because better)
                temp_cor = nc_c.temp -0.0065 * (nc_h.ref_hgt - nc_c.ref_hgt)
                totest = temp_cor - nc_h.temp
                self.assertTrue(totest.mean() < 0.5)
                # precip
                totest = nc_c.prcp - nc_h.prcp
                self.assertTrue(totest.mean() < 100)

    def test_sh(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        # We have to make a non cropped custom file
        fpath = cfg.PATHS['climate_file']
        ds = xr.open_dataset(fpath)
        ds = ds.sel(time=slice('1802-01-01', '2002-12-01'))
        nf = os.path.join(self.testdir, 'testdata.nc')
        ds.to_netcdf(nf)
        cfg.PATHS['climate_file'] = nf
        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # Trick
        assert gdir.hemisphere == 'nh'
        gdir.hemisphere = 'sh'

        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir_cru)
        assert gdir.hemisphere == 'nh'
        gdir.hemisphere = 'sh'
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)

        climate.process_custom_climate_data(gdirs[0])
        ci = gdirs[0].read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1803)
        self.assertEqual(ci['hydro_yr_1'], 2002)

        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cru_dir = os.path.dirname(cru_dir)
        cfg.PATHS['climate_file'] = ''
        cfg.PATHS['cru_dir'] = cru_dir
        climate.process_cru_data(gdirs[1])
        cfg.PATHS['cru_dir'] = ''
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        ci = gdir.read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1902)
        self.assertEqual(ci['hydro_yr_1'], 2014)

        gdh = gdirs[0]
        gdc = gdirs[1]
        with xr.open_dataset(
                os.path.join(gdh.dir, 'climate_monthly.nc')) as nc_h:

            assert nc_h['time.month'][0] == 4
            assert nc_h['time.year'][0] == 1802
            assert nc_h['time.month'][-1] == 3
            assert nc_h['time.year'][-1] == 2002

            with xr.open_dataset(
                    os.path.join(gdc.dir, 'climate_monthly.nc')) as nc_c:

                assert nc_c['time.month'][0] == 4
                assert nc_c['time.year'][0] == 1901
                assert nc_c['time.month'][-1] == 3
                assert nc_c['time.year'][-1] == 2014

                # put on the same altitude
                # (using default gradient because better)
                temp_cor = nc_c.temp - 0.0065 * (nc_h.ref_hgt - nc_c.ref_hgt)
                totest = temp_cor - nc_h.temp
                self.assertTrue(totest.mean() < 0.5)
                # precip
                totest = nc_c.prcp - nc_h.prcp
                self.assertTrue(totest.mean() < 100)

    def test_mb_climate(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)

        with netCDF4.Dataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]
            ref_t = np.where(ref_t < cfg.PARAMS['temp_melt'], 0,
                             ref_t - cfg.PARAMS['temp_melt'])

        hgts = np.array([ref_h, ref_h, -8000, 8000])
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts, 1.)

        ref_nt = 202*12
        self.assertTrue(len(time) == ref_nt)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(temp[0, :], ref_t)
        np.testing.assert_allclose(temp[0, :], temp[1, :])
        np.testing.assert_allclose(prcp[0, :], prcp[1, :])
        np.testing.assert_allclose(prcp[3, :], ref_p)
        np.testing.assert_allclose(prcp[2, :], ref_p*0)
        np.testing.assert_allclose(temp[3, :], ref_p*0)

        yr = [1802, 1802]
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts, 1.,
                                                        year_range=yr)
        ref_nt = 1*12
        self.assertTrue(len(time) == ref_nt)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(temp[0, :], ref_t[0:12])
        np.testing.assert_allclose(temp[0, :], temp[1, :])
        np.testing.assert_allclose(prcp[0, :], prcp[1, :])
        np.testing.assert_allclose(prcp[3, :], ref_p[0:12])
        np.testing.assert_allclose(prcp[2, :], ref_p[0:12]*0)
        np.testing.assert_allclose(temp[3, :], ref_p[0:12]*0)

        yr = [1803, 1804]
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts, 1.,
                                                        year_range=yr)
        ref_nt = 2*12
        self.assertTrue(len(time) == ref_nt)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(temp[0, :], ref_t[12:36])
        np.testing.assert_allclose(temp[0, :], temp[1, :])
        np.testing.assert_allclose(prcp[0, :], prcp[1, :])
        np.testing.assert_allclose(prcp[3, :], ref_p[12:36])
        np.testing.assert_allclose(prcp[2, :], ref_p[12:36]*0)
        np.testing.assert_allclose(temp[3, :], ref_p[12:36]*0)

    def test_yearly_mb_climate(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)

        with netCDF4.Dataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]
            ref_t = np.where(ref_t <= cfg.PARAMS['temp_melt'], 0,
                             ref_t - cfg.PARAMS['temp_melt'])

        # NORMAL --------------------------------------------------------------
        hgts = np.array([ref_h, ref_h, -8000, 8000])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts, 1)

        ref_nt = 202
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))

        yr = [1802, 1802]
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts, 1,
                                                                year_range=yr)
        ref_nt = 1
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(years == 1802)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(temp[0, :], np.sum(ref_t[0:12]))
        np.testing.assert_allclose(temp[0, :], temp[1, :])
        np.testing.assert_allclose(prcp[0, :], prcp[1, :])
        np.testing.assert_allclose(prcp[3, :], np.sum(ref_p[0:12]))
        np.testing.assert_allclose(prcp[2, :], np.sum(ref_p[0:12])*0)
        np.testing.assert_allclose(temp[3, :], np.sum(ref_p[0:12])*0)

        yr = [1803, 1804]
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts, 1,
                                                                year_range=yr)
        ref_nt = 2
        self.assertTrue(len(years) == ref_nt)
        np.testing.assert_allclose(years, yr)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(prcp[2, :], [0, 0])
        np.testing.assert_allclose(temp[3, :], [0, 0])

        # FLATTEN -------------------------------------------------------------
        hgts = np.array([ref_h, ref_h, -8000, 8000])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts, 1,
                                                                flatten=True)

        ref_nt = 202
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(temp.shape == (ref_nt,))
        self.assertTrue(prcp.shape == (ref_nt,))

        yr = [1802, 1802]
        hgts = np.array([ref_h])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts, 1,
                                                                year_range=yr,
                                                                flatten=True)
        ref_nt = 1
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(years == 1802)
        self.assertTrue(temp.shape == (ref_nt,))
        self.assertTrue(prcp.shape == (ref_nt,))
        np.testing.assert_allclose(temp[:], np.sum(ref_t[0:12]))

        yr = [1802, 1802]
        hgts = np.array([8000])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts, 1,
                                                                year_range=yr,
                                                                flatten=True)
        np.testing.assert_allclose(prcp[:], np.sum(ref_p[0:12]))

    def test_mu_candidates(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)
        climate.mu_candidates(gdir)

        se = gdir.read_pickle('mu_candidates')[2.5]
        self.assertTrue(se.index[0] == 1802)
        self.assertTrue(se.index[-1] == 2003)

        df = pd.DataFrame()
        df['mu'] = se

        # Check that the moovin average of temp is negatively correlated
        # with the mus
        with netCDF4.Dataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_t = nc_r.variables['temp'][:, 1, 1]
        ref_t = np.mean(ref_t.reshape((len(df), 12)), 1)
        ma = np.convolve(ref_t, np.ones(31) / float(31), 'same')
        df['temp'] = ma
        df = df.dropna()
        self.assertTrue(np.corrcoef(df['mu'], df['temp'])[0, 1] < -0.75)

    def test_find_tstars(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)
        climate.mu_candidates(gdir)

        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        self.assertEqual(prcp_fac, 2.5)
        y, t, p = climate.mb_yearly_climate_on_glacier(gdir, prcp_fac)

        # which years to look at
        selind = np.searchsorted(y, mbdf.index)
        t = t[selind]
        p = p[selind]

        mu_yr_clim = gdir.read_pickle('mu_candidates')[prcp_fac]
        for t_s, rmd in zip(t_stars, bias):
            mb_per_mu = p - mu_yr_clim.loc[t_s] * t
            md = utils.md(mbdf, mb_per_mu)
            np.testing.assert_allclose(md, rmd, rtol=1e-5)
            self.assertTrue(np.abs(md/np.mean(mbdf)) < 0.1)
            r = utils.corrcoef(mbdf, mb_per_mu)
            self.assertTrue(r > 0.8)

        _t_s = t_s
        _rmd = rmd

        # test crop years
        cfg.PARAMS['tstar_search_window'] = [1902, 0]
        climate.mu_candidates(gdir)
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        self.assertEqual(prcp_fac, 2.5)
        for t_s, rmd in zip(t_stars, bias):
            mb_per_mu = p - mu_yr_clim.loc[t_s] * t
            md = utils.md(mbdf, mb_per_mu)
            np.testing.assert_allclose(md, rmd, rtol=1e-5)
            self.assertTrue(np.abs(md/np.mean(mbdf)) < 0.1)
            r = utils.corrcoef(mbdf, mb_per_mu)
            self.assertTrue(r > 0.8)
            self.assertTrue(t_s >= 1902)
            self.assertEqual(t_s, _t_s)
            self.assertEqual(rmd, _rmd)

        # test distribute
        climate.compute_ref_t_stars(gdirs)
        climate.distribute_t_stars(gdirs)
        cfg.PARAMS['tstar_search_window'] = [0, 0]

        df = pd.read_csv(gdir.get_filepath('local_mustar'))
        np.testing.assert_allclose(df['t_star'], _t_s)
        np.testing.assert_allclose(df['bias'], _rmd)
        np.testing.assert_allclose(df['prcp_fac'], 2.5)

    def test_find_tstars_stddev_perglacier_prcp_fac(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        cfg.PARAMS['prcp_scaling_factor'] = 'stddev_perglacier'

        gdirs = []
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)
        climate.mu_candidates(gdir)

        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']

        y, t, p = climate.mb_yearly_climate_on_glacier(gdir, prcp_fac)

        # which years to look at
        selind = np.searchsorted(y, mbdf.index)
        t = t[selind]
        p = p[selind]

        dffac = gdir.read_pickle('prcp_fac_optim').loc[prcp_fac]
        np.testing.assert_allclose(dffac['avg_bias'], np.mean(bias))
        mu_yr_clim = gdir.read_pickle('mu_candidates')[prcp_fac]
        std_bias = []
        for t_s, rmd in zip(t_stars, bias):
            mb_per_mu = p - mu_yr_clim.loc[t_s] * t
            md = utils.md(mbdf, mb_per_mu)
            np.testing.assert_allclose(md, rmd, rtol=1e-4)
            self.assertTrue(np.abs(md / np.mean(mbdf)) < 0.1)
            r = utils.corrcoef(mbdf, mb_per_mu)
            self.assertTrue(r > 0.8)
            std_bias.append(np.std(mb_per_mu) - np.std(mbdf))

        np.testing.assert_allclose(dffac['avg_std_bias'], np.mean(std_bias),
                                   rtol=1e-4)

        # test crop years
        cfg.PARAMS['tstar_search_window'] = [1902, 0]
        climate.mu_candidates(gdir)
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        mu_yr_clim = gdir.read_pickle('mu_candidates')[prcp_fac]
        y, t, p = climate.mb_yearly_climate_on_glacier(gdir, prcp_fac)
        selind = np.searchsorted(y, mbdf.index)
        t = t[selind]
        p = p[selind]
        for t_s, rmd in zip(t_stars, bias):
            mb_per_mu = p - mu_yr_clim.loc[t_s] * t
            md = utils.md(mbdf, mb_per_mu)
            np.testing.assert_allclose(md, rmd, rtol=1e-4)
            self.assertTrue(np.abs(md / np.mean(mbdf)) < 0.1)
            r = utils.corrcoef(mbdf, mb_per_mu)
            self.assertTrue(r > 0.8)
            self.assertTrue(t_s >= 1902)

        # test distribute
        climate.compute_ref_t_stars(gdirs)
        climate.distribute_t_stars(gdirs)
        cfg.PARAMS['tstar_search_window'] = [0, 0]

        df = pd.read_csv(gdir.get_filepath('local_mustar'))
        np.testing.assert_allclose(df['t_star'], t_s)
        np.testing.assert_allclose(df['bias'], rmd)
        np.testing.assert_allclose(df['prcp_fac'], prcp_fac)

        cfg.PARAMS['prcp_scaling_factor'] = 2.5

    def test_find_tstars_stddev_prcp_fac(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        cfg.PARAMS['prcp_scaling_factor'] = 'stddev'

        gdirs = []
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)

        # test distribute
        climate.compute_ref_t_stars(gdirs)
        climate.distribute_t_stars(gdirs)
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        bias_std, prcp_fac = res['std_bias'], res['prcp_fac']

        # check that other prcp_factors are less good
        cfg.PARAMS['prcp_scaling_factor'] = prcp_fac + 0.3
        climate.compute_ref_t_stars(gdirs)
        climate.distribute_t_stars(gdirs)
        res = climate.t_star_from_refmb(gdir, mbdf)
        bias_std_after, prcp_fac_after = res['std_bias'], res['prcp_fac']
        self.assertLessEqual(np.abs(np.mean(bias_std)), np.abs(np.mean(bias_std_after)))
        self.assertEqual(prcp_fac + 0.3, prcp_fac_after)

        cfg.PARAMS['prcp_scaling_factor'] = prcp_fac - 0.3
        climate.compute_ref_t_stars(gdirs)
        climate.distribute_t_stars(gdirs)
        res = climate.t_star_from_refmb(gdir, mbdf)
        bias_std_after, prcp_fac_after = res['std_bias'], res['prcp_fac']
        self.assertLessEqual(np.abs(np.mean(bias_std)), np.abs(np.mean(bias_std_after)))
        self.assertEqual(prcp_fac - 0.3, prcp_fac_after)

        cfg.PARAMS['prcp_scaling_factor'] = 2.5

    def test_local_mustar(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        cfg.PARAMS['prcp_scaling_factor'] = 2.9

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        t_star, bias, prcp_fac = res['t_star'], res['bias'],  res['prcp_fac']
        self.assertEqual(prcp_fac, 2.9)

        t_star = t_star[-1]
        bias = bias[-1]

        climate.local_mustar(gdir, tstar=t_star, bias=bias, prcp_fac=prcp_fac)
        climate.apparent_mb(gdir)

        df = pd.read_csv(gdir.get_filepath('local_mustar'))
        mu_ref = gdir.read_pickle('mu_candidates')
        mu_ref = mu_ref[prcp_fac].loc[t_star]
        np.testing.assert_allclose(mu_ref, df['mu_star'][0], atol=1e-3)

        # Check for apparent mb to be zeros
        fls = gdir.read_pickle('inversion_flowlines')
        tmb = 0.
        for fl in fls:
            self.assertTrue(fl.apparent_mb.shape == fl.widths.shape)
            tmb += np.sum(fl.apparent_mb * fl.widths)
            assert not fl.flux_needed_correction
        np.testing.assert_allclose(tmb, 0., atol=0.01)
        np.testing.assert_allclose(fls[-1].flux[-1], 0., atol=0.01)

        # ------ Look for gradient
        # which years to look at
        fls = gdir.read_pickle('inversion_flowlines')
        mb_on_h = np.array([])
        h = np.array([])
        for fl in fls:
            y, t, p = climate.mb_yearly_climate_on_height(gdir, fl.surface_h,
                                                          prcp_fac)
            selind = np.searchsorted(y, mbdf.index)
            t = np.mean(t[:, selind], axis=1)
            p = np.mean(p[:, selind], axis=1)
            mb_on_h = np.append(mb_on_h, p - mu_ref * t)
            h = np.append(h, fl.surface_h)
        dfg = pd.read_csv(get_demo_file('mbgrads_RGI40-11.00897.csv'),
                          index_col='ALTITUDE').mean(axis=1)
        # Take the altitudes below 3100 and fit a line
        dfg = dfg[dfg.index < 3100]
        pok = np.where(h < 3100)
        from scipy.stats import linregress
        slope_obs, _, _, _, _ = linregress(dfg.index, dfg.values)
        slope_our, _, _, _, _ = linregress(h[pok], mb_on_h[pok])
        np.testing.assert_allclose(slope_obs, slope_our, rtol=0.1)

        cfg.PARAMS['prcp_scaling_factor'] = 2.5

    def test_automated_workflow(self):

        cfg.PARAMS['run_mb_calibration'] = False
        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cru_dir = os.path.dirname(cru_dir)
        cfg.PATHS['climate_file'] = ''
        cfg.PATHS['cru_dir'] = cru_dir

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        assert gdir.rgi_version == '5'
        gis.define_glacier_region(gdir, entity=entity)
        workflow.gis_prepro_tasks([gdir])
        workflow.climate_tasks([gdir])

        hef_file = get_demo_file('Hintereisferner_RGI6.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        assert gdir.rgi_version == '6'
        workflow.gis_prepro_tasks([gdir])
        workflow.climate_tasks([gdir])
        cfg.PARAMS['run_mb_calibration'] = True


class TestFilterNegFlux(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
        cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
        cfg.PARAMS['border'] = 10

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_filter(self):

        entity = gpd.read_file(get_demo_file('rgi_oetztal.shp'))
        entity = entity.loc[entity.RGIId == 'RGI50-11.00666'].iloc[0]

        cfg.PARAMS['filter_for_neg_flux'] = False

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        climate.local_mustar(gdir, tstar=1931, bias=0, prcp_fac=2.5)
        climate.apparent_mb(gdir)

        fls1 = gdir.read_pickle('inversion_flowlines')
        assert np.any([fl.flux_needed_correction for fl in fls1])

        cfg.PARAMS['filter_for_neg_flux'] = True
        climate.apparent_mb(gdir)

        fls = gdir.read_pickle('inversion_flowlines')
        assert len(fls) < len(fls1)
        assert not np.any([fl.flux_needed_correction for fl in fls])


class TestInversion(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['border'] = 10

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_invert_hef(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        t_star, bias, prcp_fac = res['t_star'],  res['bias'],  res['prcp_fac']
        t_star = t_star[-1]
        climate.local_mustar(gdir, tstar=t_star, bias=bias, prcp_fac=prcp_fac)
        climate.apparent_mb(gdir)

        # OK. Values from Fischer and Kuhn 2013
        # Area: 8.55
        # meanH = 67+-7
        # Volume = 0.573+-0.063
        # maxH = 242+-13
        inversion.prepare_for_inversion(gdir, add_debug_var=True)
        # Check how many clips:
        cls = gdir.read_pickle('inversion_input')
        nabove = 0
        maxs = 0.
        npoints = 0.
        for cl in cls:
            # Clip slope to avoid negative and small slopes
            slope = cl['slope_angle']
            nm = np.where(slope <  np.deg2rad(2.))
            nabove += len(nm[0])
            npoints += len(slope)
            _max = np.max(slope)
            if _max > maxs:
                maxs = _max
            if cl['is_last']:
                self.assertEqual(cl['flux'][-1], 0.)

        self.assertTrue(nabove == 0)
        self.assertTrue(np.rad2deg(maxs) < 40.)

        ref_v = 0.573 * 1e9

        def to_optimize(x):
            glen_a = cfg.A * x[0]
            fs = cfg.FS * x[1]
            v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                         glen_a=glen_a)
            return (v - ref_v)**2

        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-4)['x']
        self.assertTrue(out[0] > 0.1)
        self.assertTrue(out[1] > 0.1)
        self.assertTrue(out[0] < 1.1)
        self.assertTrue(out[1] < 1.1)
        glen_a = cfg.A * out[0]
        fs = cfg.FS * out[1]
        v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a,
                                                     write=True)
        np.testing.assert_allclose(ref_v, v)

        cls = gdir.read_pickle('inversion_output')
        fls = gdir.read_pickle('inversion_flowlines')
        maxs = 0.
        for cl, fl in zip(cls, fls):
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
            if fl.flows_to is None:
                self.assertEqual(cl['volume'][-1], 0)
                self.assertEqual(cl['thick'][-1], 0)

        np.testing.assert_allclose(242, maxs, atol=40)

        # Filter
        inversion.filter_inversion_output(gdir)
        maxs = 0.
        v = 0.
        cls = gdir.read_pickle('inversion_output')
        for cl in cls:
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
            v += np.nansum(cl['volume'])
        np.testing.assert_allclose(242, maxs, atol=10)
        np.testing.assert_allclose(ref_v, v)

    def test_invert_hef_from_linear_mb(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.apparent_mb_from_linear_mb(gdir)

        # OK. Values from Fischer and Kuhn 2013
        # Area: 8.55
        # meanH = 67+-7
        # Volume = 0.573+-0.063
        # maxH = 242+-13
        inversion.prepare_for_inversion(gdir, add_debug_var=True)

        # Check how many clips:
        cls = gdir.read_pickle('inversion_input')
        nabove = 0
        maxs = 0.
        npoints = 0.
        for cl in cls:
            # Clip slope to avoid negative and small slopes
            slope = cl['slope_angle']
            nm = np.where(slope <  np.deg2rad(2.))
            nabove += len(nm[0])
            npoints += len(slope)
            _max = np.max(slope)
            if _max > maxs:
                maxs = _max
            if cl['is_last']:
                self.assertEqual(cl['flux'][-1], 0.)

        self.assertTrue(nabove == 0)
        self.assertTrue(np.rad2deg(maxs) < 40.)

        ref_v = 0.573 * 1e9

        def to_optimize(x):
            glen_a = cfg.A * x[0]
            fs = cfg.FS * x[1]
            v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                         glen_a=glen_a)
            return (v - ref_v)**2

        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-4)['x']
        self.assertTrue(out[0] > 0.1)
        self.assertTrue(out[1] > 0.1)
        self.assertTrue(out[0] < 1.1)
        self.assertTrue(out[1] < 1.1)
        glen_a = cfg.A * out[0]
        fs = cfg.FS * out[1]
        v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a,
                                                     write=True)
        np.testing.assert_allclose(ref_v, v)

        cls = gdir.read_pickle('inversion_output')
        fls = gdir.read_pickle('inversion_flowlines')
        maxs = 0.
        for cl, fl in zip(cls, fls):
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
            if fl.flows_to is None:
                self.assertEqual(cl['volume'][-1], 0)
                self.assertEqual(cl['thick'][-1], 0)

        np.testing.assert_allclose(242, maxs, atol=10)

        # Filter
        inversion.filter_inversion_output(gdir)
        maxs = 0.
        v = 0.
        cls = gdir.read_pickle('inversion_output')
        for cl in cls:
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
            v += np.nansum(cl['volume'])
        np.testing.assert_allclose(242, maxs, atol=10)
        np.testing.assert_allclose(ref_v, v)

    @is_slow
    def test_distribute(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        t_star, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        t_star = t_star[-1]
        bias = bias[-1]
        climate.local_mustar(gdir, tstar=t_star, bias=bias, prcp_fac=prcp_fac)
        climate.apparent_mb(gdir)

        # OK. Values from Fischer and Kuhn 2013
        # Area: 8.55
        # meanH = 67+-7
        # Volume = 0.573+-0.063
        # maxH = 242+-13
        inversion.prepare_for_inversion(gdir)

        ref_v = 0.573 * 1e9
        def to_optimize(x):
            glen_a = cfg.A * x[0]
            fs = cfg.FS * x[1]
            v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                         glen_a=glen_a)
            return (v - ref_v)**2
        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-1)['x']
        glen_a = cfg.A * out[0]
        fs = cfg.FS * out[1]
        v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a,
                                                     write=True)
        np.testing.assert_allclose(ref_v, v)

        inversion.distribute_thickness(gdir, how='per_altitude',
                                       add_nc_name=True)
        inversion.distribute_thickness(gdir, how='per_interpolation',
                                       add_slope=False,
                                       add_nc_name=True)

        grids_file = gdir.get_filepath('gridded_data')
        with netCDF4.Dataset(grids_file) as nc:
            t1 = nc.variables['thickness_per_altitude'][:]
            t2 = nc.variables['thickness_per_interpolation'][:]

        np.testing.assert_allclose(np.sum(t1), np.sum(t2))
        np.testing.assert_allclose(np.max(t1), np.max(t2), atol=30)

    @is_slow
    def test_invert_hef_nofs(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        t_star, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        t_star = t_star[-1]
        bias = bias[-1]
        climate.local_mustar(gdir, tstar=t_star, bias=bias, prcp_fac=prcp_fac)
        climate.apparent_mb(gdir)

        # OK. Values from Fischer and Kuhn 2013
        # Area: 8.55
        # meanH = 67+-7
        # Volume = 0.573+-0.063
        # maxH = 242+-13

        inversion.prepare_for_inversion(gdir)

        ref_v = 0.573 * 1e9

        def to_optimize(x):
            glen_a = cfg.A * x[0]
            fs = 0.
            v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                         glen_a=glen_a)
            return (v - ref_v)**2

        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1],
                                    bounds=((0.00001, 100000),),
                                    tol=1e-4)['x']

        self.assertTrue(out[0] > 0.1)
        self.assertTrue(out[0] < 10)

        glen_a = cfg.A * out[0]
        fs = 0.
        v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a,
                                                     write=True)
        np.testing.assert_allclose(ref_v, v)

        cls = gdir.read_pickle('inversion_output')
        fls = gdir.read_pickle('inversion_flowlines')
        maxs = 0.
        for cl, fl in zip(cls, fls):
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
        np.testing.assert_allclose(242, maxs, atol=45)

        # check that its not tooo sensitive to the dx
        cfg.PARAMS['flowline_dx'] = 1.
        cfg.PARAMS['filter_for_neg_flux'] = False
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_star, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        t_star = t_star[-1]
        bias = bias[-1]
        climate.local_mustar(gdir, tstar=t_star, bias=bias, prcp_fac=prcp_fac)
        climate.apparent_mb(gdir)
        inversion.prepare_for_inversion(gdir)
        v, _ = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a,
                                                     write=True)

        np.testing.assert_allclose(v, ref_v, rtol=0.06)
        cls = gdir.read_pickle('inversion_output')
        maxs = 0.
        for cl in cls:
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max

        np.testing.assert_allclose(242, maxs, atol=31)
        cfg.PARAMS['filter_for_neg_flux'] = True

    def test_continue_on_error(self):

        cfg.PARAMS['continue_on_error'] = True
        cfg.PATHS['working_dir'] = self.testdir

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        miniglac = shpg.Point(entity.CENLON, entity.CENLAT).buffer(0.0001)
        entity.geometry = miniglac
        entity.RGIID = 'RGI50-11.fake'

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        climate.mu_candidates(gdir)
        climate.local_mustar(gdir, tstar=1970, bias=0, prcp_fac=2.)
        climate.apparent_mb(gdir)
        inversion.prepare_for_inversion(gdir)
        inversion.volume_inversion(gdir)

        rdir = os.path.join(self.testdir, 'RGI50-11', 'RGI50-11.fa',
                            'RGI50-11.fake')
        self.assertTrue(os.path.exists(rdir))

        rdir = os.path.join(rdir, 'log.txt')
        self.assertTrue(os.path.exists(rdir))

        cfg.PARAMS['continue_on_error'] = False

        # Test the glacier charac
        dfc = utils.glacier_characteristics([gdir], path=False)
        self.assertEqual(dfc.terminus_type.values[0], 'Land-terminating')
        self.assertFalse('tstar_avg_temp_mean_elev' in dfc)


class TestGrindelInvert(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_grindel')
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PARAMS['use_multiple_flowlines'] = False
        # not crop
        cfg.PARAMS['max_thick_to_width_ratio'] = 10
        cfg.PARAMS['max_shape_param'] = 10
        cfg.PARAMS['section_smoothing'] = 0.

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    def clean_dir(self):
        self.rm_dir()
        tfile = get_demo_file('glacier_grid.json')
        gpath = os.path.dirname(tfile)
        self.rgin = os.path.basename(gpath)
        gpath = os.path.dirname(gpath)
        assert self.rgin == 'RGI50-11.01270'
        shutil.copytree(gpath, os.path.join(self.testdir, 'RGI50-11',
                                            'RGI50-11.01'))

    def _parabolic_bed(self):

        map_dx = 100.
        dx = 1.
        nx = 200

        surface_h = np.linspace(3000, 1000, nx)
        bed_h = surface_h
        shape = surface_h * 0. + 3.e-03

        coords = np.arange(0, nx-0.5, 1)
        line = shpg.LineString(np.vstack([coords, coords*0.]).T)
        return [flowline.ParabolicBedFlowline(line, dx, map_dx, surface_h,
                                              bed_h, shape)]

    def test_ideal_glacier(self):

        # we are making a
        glen_a = cfg.A * 1
        from oggm.core import flowline

        gdir = utils.GlacierDirectory(self.rgin, base_dir=self.testdir)

        fls = self._parabolic_bed()
        mbmod = massbalance.LinearMassBalance(2800.)
        model = flowline.FluxBasedModel(fls, mb_model=mbmod, glen_a=glen_a)
        model.run_until_equilibrium()

        # from dummy bed
        map_dx = 100.
        towrite = []
        for fl in model.fls:
            # Distance between two points
            dx = fl.dx * map_dx
            # Widths
            widths = fl.widths * map_dx
            # Heights
            hgt = fl.surface_h
            # Flux
            mb = mbmod.get_annual_mb(hgt) * cfg.SEC_IN_YEAR * cfg.RHO
            fl.flux = np.zeros(len(fl.surface_h))
            fl.set_apparent_mb(mb)
            flux = fl.flux * (map_dx**2) / cfg.SEC_IN_YEAR / cfg.RHO
            pok = np.nonzero(widths > 10.)
            widths = widths[pok]
            hgt = hgt[pok]
            flux = flux[pok]
            flux_a0 = 1.5 * flux / widths
            angle = np.arctan(-np.gradient(hgt, dx))  # beware the minus sign
            # Clip flux to 0
            assert not np.any(flux < -0.1)
            # add to output
            cl_dic = dict(dx=dx, flux=flux, flux_a0=flux_a0, width=widths,
                          hgt=hgt, slope_angle=angle, is_last=True,
                          is_rectangular=np.zeros(len(flux), dtype=bool))
            towrite.append(cl_dic)

        # Write out
        gdir.write_pickle(towrite, 'inversion_input')
        v, a = inversion.mass_conservation_inversion(gdir, glen_a=glen_a)
        v_km3 = v * 1e-9
        a_km2 = np.sum(widths * dx) * 1e-6
        v_vas = 0.034*(a_km2**1.375)

        np.testing.assert_allclose(v, model.volume_m3, rtol=0.01)

        cl = gdir.read_pickle('inversion_output')[0]
        rmsd = utils.rmsd(cl['thick'], model.fls[0].thick[:len(cl['thick'])])
        assert rmsd < 10.

    def test_invert_and_run(self):

        from oggm.core import flowline, massbalance

        glen_a = cfg.A*2

        gdir = utils.GlacierDirectory(self.rgin, base_dir=self.testdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.local_mustar(gdir, tstar=1975, bias=0., prcp_fac=1)
        climate.apparent_mb(gdir)
        inversion.prepare_for_inversion(gdir)
        v, a = inversion.mass_conservation_inversion(gdir, glen_a=glen_a)
        inversion.filter_inversion_output(gdir)
        flowline.init_present_time_glacier(gdir)
        mb_mod = massbalance.ConstantMassBalance(gdir)
        fls = gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                        fs=0, glen_a=glen_a)

        ref_vol = model.volume_m3
        np.testing.assert_allclose(v, ref_vol, rtol=0.01)
        model.run_until(10)
        after_vol = model.volume_m3
        np.testing.assert_allclose(ref_vol, after_vol, rtol=0.1)

    def test_intersections(self):
        cfg.PARAMS['use_multiple_flowlines'] = True
        gdir = utils.GlacierDirectory(self.rgin, base_dir=self.testdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        # see that we have as many catchments as flowlines
        fls = gdir.read_pickle('inversion_flowlines')
        gdfc = gpd.read_file(gdir.get_filepath('flowline_catchments'))
        self.assertEqual(len(fls), len(gdfc))
        # and at least as many intersects
        gdfc = gpd.read_file(gdir.get_filepath('catchments_intersects'))
        self.assertGreaterEqual(len(gdfc), len(fls)-1)

        # check touch borders qualitatively
        self.assertGreaterEqual(np.sum(fls[-1].is_rectangular),  10)


class TestGCMClimate(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_prepro')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cru_dir = os.path.dirname(cru_dir)
        cfg.PATHS['climate_file'] = ''
        cfg.PATHS['cru_dir'] = cru_dir
        cfg.PARAMS['border'] = 10

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_process_cesm(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)
        climate.process_cru_data(gdir)

        ci = gdir.read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1902)
        self.assertEqual(ci['hydro_yr_1'], 2014)

        f = get_demo_file('cesm.TREFHT.160001-200512.selection.nc')
        cfg.PATHS['gcm_temp_file'] = f
        f = get_demo_file('cesm.PRECC.160001-200512.selection.nc')
        cfg.PATHS['gcm_precc_file'] = f
        f = get_demo_file('cesm.PRECL.160001-200512.selection.nc')
        cfg.PATHS['gcm_precl_file'] = f
        climate.process_cesm_data(gdir)
        with warnings.catch_warnings():
            # Long time series are currently a pain pandas
            warnings.filterwarnings("ignore",
                                    message='Unable to decode time axis')
            fh = gdir.get_filepath('climate_monthly')
            fcesm = gdir.get_filepath('cesm_data')
            with xr.open_dataset(fh) as cru, xr.open_dataset(fcesm) as cesm:

                tv = cesm.time.values
                time = pd.period_range(tv[0].strftime('%Y-%m-%d'),
                                       tv[-1].strftime('%Y-%m-%d'),
                                       freq='M')
                cesm['time'] = time
                cesm.coords['year'] = ('time', time.year)
                cesm.coords['month'] = ('time', time.month)

                # Let's do some basic checks
                scru = cru.sel(time=slice('1961', '1990'))
                scesm = cesm.load().isel(time=((cesm.year >= 1961) &
                                               (cesm.year <= 1990)))
                # Climate during the chosen period should be the same
                np.testing.assert_allclose(scru.temp.mean(),
                                           scesm.temp.mean(),
                                           rtol=1e-3)
                np.testing.assert_allclose(scru.prcp.mean(),
                                           scesm.prcp.mean(),
                                           rtol=1e-3)
                np.testing.assert_allclose(scru.grad.mean(),
                                           scesm.grad.mean())
                # And also the anual cycle
                scru = scru.groupby('time.month').mean()
                scesm = scesm.groupby(scesm.month).mean()
                np.testing.assert_allclose(scru.temp, scesm.temp, rtol=1e-3)
                np.testing.assert_allclose(scru.prcp, scesm.prcp, rtol=1e-3)
                np.testing.assert_allclose(scru.grad, scesm.grad)

                # How did the annua cycle change with time?
                scesm1 = cesm.isel(time=((cesm.year >= 1961) &
                                         (cesm.year <= 1990)))
                scesm2 = cesm.isel(time=((cesm.year >= 1661) &
                                         (cesm.year <= 1690)))
                scesm1 = scesm1.groupby(scesm1.month).mean()
                scesm2 = scesm2.groupby(scesm2.month).mean()
                # No more than one degree? (silly test)
                np.testing.assert_allclose(scesm1.temp, scesm2.temp, atol=1)
                # N more than 30%? (silly test)
                np.testing.assert_allclose(scesm1.prcp, scesm2.prcp, rtol=0.3)

    def test_compile_climate_input(self):

        filename = 'cesm_data'
        filesuffix = '_cesm'

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)

        climate.process_cru_data(gdir)
        utils.compile_climate_input([gdir])

        f = get_demo_file('cesm.TREFHT.160001-200512.selection.nc')
        cfg.PATHS['gcm_temp_file'] = f
        f = get_demo_file('cesm.PRECC.160001-200512.selection.nc')
        cfg.PATHS['gcm_precc_file'] = f
        f = get_demo_file('cesm.PRECL.160001-200512.selection.nc')
        cfg.PATHS['gcm_precl_file'] = f
        climate.process_cesm_data(gdir, filesuffix=filesuffix)
        utils.compile_climate_input([gdir], filename=filename,
                                    filesuffix=filesuffix)

        with warnings.catch_warnings():
            # Long time series are currently a pain pandas
            warnings.filterwarnings("ignore", message='Unable to decode')

            # CRU
            f1 = path.join(cfg.PATHS['working_dir'], 'climate_input.nc')
            f2 = gdir.get_filepath(filename='climate_monthly')
            with xr.open_dataset(f1) as clim_cru1, xr.open_dataset(f2) as clim_cru2:
                np.testing.assert_allclose(np.squeeze(clim_cru1.prcp),
                                           clim_cru2.prcp)
                np.testing.assert_allclose(np.squeeze(clim_cru1.temp),
                                           clim_cru2.temp)
                np.testing.assert_allclose(np.squeeze(clim_cru1.grad),
                                           clim_cru2.grad)
                np.testing.assert_allclose(np.squeeze(clim_cru1.ref_hgt),
                                           clim_cru2.ref_hgt)
                np.testing.assert_allclose(np.squeeze(clim_cru1.ref_pix_lat),
                                           clim_cru2.ref_pix_lat)
                np.testing.assert_allclose(np.squeeze(clim_cru1.ref_pix_lon),
                                           clim_cru2.ref_pix_lon)
                np.testing.assert_allclose(clim_cru1.calendar_month,
                                           clim_cru2['time.month'])
                np.testing.assert_allclose(clim_cru1.calendar_year,
                                           clim_cru2['time.year'])
                np.testing.assert_allclose(clim_cru1.hydro_month[[0, -1]],
                                           [1, 12])

            # CESM
            f1 = path.join(cfg.PATHS['working_dir'], 'climate_input_cesm.nc')
            f2 = gdir.get_filepath(filename=filename, filesuffix=filesuffix)
            with xr.open_dataset(f1) as clim_cesm1, xr.open_dataset(f2) as clim_cesm2:
                np.testing.assert_allclose(np.squeeze(clim_cesm1.prcp),
                                           clim_cesm2.prcp)
                np.testing.assert_allclose(np.squeeze(clim_cesm1.temp),
                                           clim_cesm2.temp)
                np.testing.assert_allclose(np.squeeze(clim_cesm1.grad),
                                           clim_cesm2.grad)
                np.testing.assert_allclose(np.squeeze(clim_cesm1.ref_hgt),
                                           clim_cesm2.ref_hgt)
                np.testing.assert_allclose(np.squeeze(clim_cesm1.ref_pix_lat),
                                           clim_cesm2.ref_pix_lat)
                np.testing.assert_allclose(np.squeeze(clim_cesm1.ref_pix_lon),
                                           clim_cesm2.ref_pix_lon)


class TestIdealizedGdir(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PARAMS['use_intersects'] = False
        cfg.PARAMS['use_multiple_flowlines'] = False

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_invert(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.apparent_mb_from_linear_mb(gdir)
        inversion.prepare_for_inversion(gdir, invert_all_rectangular=True)
        v1, _ = inversion.mass_conservation_inversion(gdir, fs=0, glen_a=cfg.A)
        tt1 = gdir.read_pickle('inversion_input')[0]
        gdir1 = gdir

        fl = gdir.read_pickle('inversion_flowlines')[0]
        map_dx = gdir.grid.dx
        gdir = utils.idealized_gdir(fl.surface_h,
                                    fl.widths * map_dx,
                                    map_dx,
                                    flowline_dx=fl.dx,
                                    base_dir=self.testdir)
        climate.apparent_mb_from_linear_mb(gdir)
        inversion.prepare_for_inversion(gdir, invert_all_rectangular=True)
        v2, _ = inversion.mass_conservation_inversion(gdir, fs=0, glen_a=cfg.A)

        tt2 = gdir.read_pickle('inversion_input')[0]
        np.testing.assert_allclose(tt1['width'], tt2['width'])
        np.testing.assert_allclose(tt1['slope_angle'], tt2['slope_angle'])
        np.testing.assert_allclose(tt1['dx'], tt2['dx'])
        np.testing.assert_allclose(tt1['flux_a0'], tt2['flux_a0'])
        np.testing.assert_allclose(v1, v2)
        np.testing.assert_allclose(gdir1.rgi_area_km2, gdir.rgi_area_km2)


class TestCatching(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_errors')


        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PARAMS['use_multiprocessing'] = False
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['working_dir'] = self.testdir
        self.log_dir = os.path.join(self.testdir, 'log')
        self.clean_dir()

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        utils.mkdir(self.testdir, reset=True)
        utils.mkdir(self.log_dir, reset=True)

    def test_pipe_log(self):

        self.clean_dir()

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        cfg.PARAMS['continue_on_error'] = True

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # This will "run" but log an error
        from oggm.tasks import run_random_climate
        workflow.execute_entity_task(run_random_climate,
                                     [(gdir, {'filesuffix':'_testme'})])

        tfile = os.path.join(self.log_dir, 'RGI40-11.00897.ERROR')
        assert os.path.exists(tfile)
        with open(tfile, 'r') as f:
            first_line = f.readline()

        spl = first_line.split(';')
        assert len(spl) == 4
        assert spl[1].strip() == 'run_random_climate_testme'

    def test_task_status(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        cfg.PARAMS['continue_on_error'] = True

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        self.assertEqual(gdir.get_task_status(gis.glacier_masks.__name__),
                         'SUCCESS')
        self.assertIsNone(gdir.get_task_status(
            centerlines.compute_centerlines.__name__))

        centerlines.compute_downstream_bedshape(gdir)

        s = gdir.get_task_status(centerlines.compute_downstream_bedshape.__name__)
        assert 'FileNotFoundError' in s

        # Try overwrite
        cfg.PARAMS['auto_skip_task'] = True
        gis.glacier_masks(gdir)

        with open(gdir.logfile) as logfile:
            lines = logfile.readlines()
        isrun = ['glacier_masks' in l for l in lines]
        assert np.sum(isrun) == 1

        cfg.PARAMS['auto_skip_task'] = False
        gis.glacier_masks(gdir)
        with open(gdir.logfile) as logfile:
            lines = logfile.readlines()
        isrun = ['glacier_masks' in l for l in lines]
        assert np.sum(isrun) == 2

        df = utils.compile_task_log([gdir], path=False)
        assert len(df) == 1
        assert len(df.columns) == 0

        tn = ['glacier_masks', 'compute_downstream_bedshape', 'not_a_task']
        df = utils.compile_task_log([gdir], task_names=tn)
        assert len(df) == 1
        assert len(df.columns) == 3
        df = df.iloc[0]
        assert df['glacier_masks'] == 'SUCCESS'
        assert df['compute_downstream_bedshape'] != 'SUCCESS'
        assert df['not_a_task'] == ''

        # Append
        centerlines.compute_centerlines(gdir)
        tn = ['compute_centerlines']
        df = utils.compile_task_log([gdir], task_names=tn)
        assert len(df) == 1
        assert len(df.columns) == 4
        df = df.iloc[0]
        assert df['glacier_masks'] == 'SUCCESS'
        assert df['compute_centerlines'] == 'SUCCESS'
        assert df['compute_downstream_bedshape'] != 'SUCCESS'
        assert not np.isfinite(df['not_a_task'])

