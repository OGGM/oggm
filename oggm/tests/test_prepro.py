from __future__ import absolute_import, division

import warnings

import oggm.utils

warnings.filterwarnings("once", category=DeprecationWarning)

import unittest
import os
import glob
import shutil

import shapely.geometry as shpg
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4
import salem
import xarray as xr

# Local imports
from oggm.core.preprocessing import gis, geometry, climate, inversion
from oggm.core.preprocessing import centerlines
import oggm.cfg as cfg
from oggm import utils
from oggm.utils import get_demo_file, tuple2int
from oggm.tests import is_slow, HAS_NEW_GDAL, requires_py3, RUN_PREPRO_TESTS

# do we event want to run the tests?
if not RUN_PREPRO_TESTS:
    raise unittest.SkipTest('Skipping all prepro tests.')

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))


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
        self.testdir = os.path.join(current_dir, 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')

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

        tdf = gpd.GeoDataFrame.from_file(gdir.get_filepath('outlines'))
        myarea = tdf.geometry.area * 10**-6
        np.testing.assert_allclose(myarea, np.float(tdf['AREA']), rtol=1e-2)
        self.assertFalse(os.path.exists(gdir.get_filepath('intersects')))

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
          Area: 8.036 mk2
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
            area = np.sum(nc.variables['glacier_mask'][:] * gdir.grid.dx**2) * 10**-6
            np.testing.assert_allclose(area, gdir.rgi_area_km2, rtol=1e-1)

    def test_nodivides(self):

        cfg.PARAMS['use_divides'] = False
        cfg.set_divides_db()
        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
            area = np.sum(nc.variables['glacier_mask'][:] * gdir.grid.dx**2) * 10**-6
            np.testing.assert_allclose(area, gdir.rgi_area_km2, rtol=1e-1)

        fp = gdir.get_filepath('outlines', div_id=1)
        self.assertTrue(os.path.exists(fp.replace('.shp', '.cpg')))
        cfg.PARAMS['use_divides'] = True
        cfg.set_divides_db()

    def test_intersects(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        self.assertTrue(os.path.exists(gdir.get_filepath('intersects')))


class TestCenterlines(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(current_dir, 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
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

    def test_centerlines(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
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
        self.assertTrue(sorted(lens) == [1, 1, 2])

    def test_downstream(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.compute_downstream_lines(gdir)

        fls = gdir.read_pickle('centerlines', div_id='major')
        self.assertIs(fls[0].flows_to, fls[-1])

    @is_slow
    def test_baltoro_centerlines(self):

        cfg.PARAMS['border'] = 2
        cfg.PATHS['dem_file'] = get_demo_file('baltoro_srtm_clip.tif')

        b_file = get_demo_file('baltoro_wgs84.shp')
        entity = gpd.GeoDataFrame.from_file(b_file).iloc[0]

        kienholz_file = get_demo_file('centerlines_baltoro_wgs84.shp')
        kdf = gpd.read_file(kienholz_file)

        # add fake attribs
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
        cls = gdir.read_pickle('centerlines', div_id=1)
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
        denom = np.float64((na+nc)*(nd+nc)+(na+nb)*(nd+nb))
        hss = np.float64(2.) * ((na*nd)-(nb*nc)) / denom
        if cfg.PARAMS['grid_dx_method'] == 'linear':
            self.assertTrue(hss > 0.53)
        if cfg.PARAMS['grid_dx_method'] == 'fixed':  # quick fix
            self.assertTrue(hss > 0.41)


class TestGeometry(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(current_dir, 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
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
        geometry.catchment_area(gdir)

        for div_id in gdir.divide_ids:

            cis = gdir.read_pickle('catchment_indices', div_id=div_id)

            # The catchment area must be as big as expected
            with netCDF4.Dataset(gdir.get_filepath('gridded_data', div_id=div_id)) as nc:
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
        geometry.initialize_flowlines(gdir)

        for div_id in gdir.divide_ids:
            cls = gdir.read_pickle('inversion_flowlines', div_id=div_id)
            for cl in cls:
                for j, ip, ob in zip(cl.inflow_indices, cl.inflow_points, cl.inflows):
                    self.assertTrue(cl.line.coords[j] == ip.coords[0])
                    self.assertTrue(ob.flows_to_point.coords[0] == ip.coords[0])
                    self.assertTrue(cl.line.coords[ob.flows_to_indice] == ip.coords[0])

        lens = [len(gdir.read_pickle('centerlines', div_id=i)) for i in [1,2,3]]
        self.assertTrue(sorted(lens) == [1, 1, 2])

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
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_intersections(gdir)
        geometry.catchment_width_geom(gdir)

    def test_width(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)

        area = 0.
        otherarea = 0.
        hgt = []
        harea = []
        for i in gdir.divide_ids:
            cls = gdir.read_pickle('inversion_flowlines', div_id=i)
            for cl in cls:
                harea.extend(list(cl.widths * cl.dx))
                hgt.extend(list(cl.surface_h))
                area += np.sum(cl.widths * cl.dx)
            with netCDF4.Dataset(gdir.get_filepath('gridded_data', div_id=i)) as nc:
                otherarea += np.sum(nc.variables['glacier_mask'][:])

        with netCDF4.Dataset(gdir.get_filepath('gridded_data', div_id=0)) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo_smoothed'][:]
        rhgt = topo[np.where(mask)][:]

        tdf = gpd.GeoDataFrame.from_file(gdir.get_filepath('outlines'))
        np.testing.assert_allclose(area, otherarea, rtol=0.1)
        area *= (gdir.grid.dx) ** 2
        otherarea *= (gdir.grid.dx) ** 2
        np.testing.assert_allclose(area * 10**-6, np.float(tdf['AREA']), rtol=1e-4)

        # Check for area distrib
        bins = np.arange(utils.nicenumber(np.min(hgt), 50, lower=True),
                         utils.nicenumber(np.max(hgt), 50)+1,
                         50.)
        h1, b = np.histogram(hgt, weights=harea, density=True, bins=bins)
        h2, b = np.histogram(rhgt, density=True, bins=bins)
        self.assertTrue(utils.rmsd(h1*100*50, h2*100*50) < 1)


class TestClimate(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(current_dir, 'tmp_prepro')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.testdir_cru = os.path.join(current_dir, 'tmp_prepro_cru')
        if not os.path.exists(self.testdir_cru):
            os.makedirs(self.testdir_cru)
        self.clean_dir()

        # Init
        cfg.initialize()
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
        cfg.PATHS['climate_file'] = '~'
        cfg.PATHS['cru_dir'] = cru_dir
        climate.process_cru_data(gdirs[1])
        cfg.PATHS['cru_dir'] = '~'
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
            ref_t = np.where(ref_t < 0, 0, ref_t)

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
            ref_t = np.where(ref_t < 0, 0, ref_t)

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
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)
        climate.mu_candidates(gdir, div_id=0)

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
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)
        climate.mu_candidates(gdir, div_id=0)

        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        self.assertEqual(prcp_fac, 2.5)
        y, t, p = climate.mb_yearly_climate_on_glacier(gdir, prcp_fac, div_id=0)

        # which years to look at
        selind = np.searchsorted(y, mbdf.index)
        t = t[selind]
        p = p[selind]

        mu_yr_clim = gdir.read_pickle('mu_candidates', div_id=0)[prcp_fac]
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
        climate.mu_candidates(gdir, div_id=0)
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

    def test_find_tstars__stddev_perglacier_prcp_fac(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        cfg.PARAMS['prcp_scaling_factor'] = 'stddev_perglacier'

        gdirs = []
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)
        climate.mu_candidates(gdir, div_id=0)

        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']

        y, t, p = climate.mb_yearly_climate_on_glacier(gdir, prcp_fac,
                                                       div_id=0)

        # which years to look at
        selind = np.searchsorted(y, mbdf.index)
        t = t[selind]
        p = p[selind]

        dffac = gdir.read_pickle('prcp_fac_optim').loc[prcp_fac]
        np.testing.assert_allclose(dffac['avg_bias'], np.mean(bias))
        mu_yr_clim = gdir.read_pickle('mu_candidates', div_id=0)[prcp_fac]
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
        climate.mu_candidates(gdir, div_id=0)
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        mu_yr_clim = gdir.read_pickle('mu_candidates', div_id=0)[prcp_fac]
        y, t, p = climate.mb_yearly_climate_on_glacier(gdir, prcp_fac,
                                                       div_id=0)
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

    def test_find_tstars_stddev_perglacier_prcp_fac(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        cfg.PARAMS['prcp_scaling_factor'] = 'stddev_perglacier'

        gdirs = []
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)
        climate.mu_candidates(gdir, div_id=0)

        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']

        y, t, p = climate.mb_yearly_climate_on_glacier(gdir, prcp_fac,
                                                       div_id=0)

        # which years to look at
        selind = np.searchsorted(y, mbdf.index)
        t = t[selind]
        p = p[selind]

        dffac = gdir.read_pickle('prcp_fac_optim').loc[prcp_fac]
        np.testing.assert_allclose(dffac['avg_bias'], np.mean(bias))
        mu_yr_clim = gdir.read_pickle('mu_candidates', div_id=0)[prcp_fac]
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
        climate.mu_candidates(gdir, div_id=0)
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_stars, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        mu_yr_clim = gdir.read_pickle('mu_candidates', div_id=0)[prcp_fac]
        y, t, p = climate.mb_yearly_climate_on_glacier(gdir, prcp_fac,
                                                       div_id=0)
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
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        gdirs.append(gdir)
        climate.process_histalp_nonparallel(gdirs)

        # test distribute
        climate.compute_ref_t_stars(gdirs)
        climate.distribute_t_stars(gdirs)
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        bias_std, prcp_fac = res['std_bias'], res['prcp_fac']

        # check that other prcp_factors are less good
        cfg.PARAMS['prcp_scaling_factor'] = prcp_fac + 0.1
        climate.compute_ref_t_stars(gdirs)
        climate.distribute_t_stars(gdirs)
        res = climate.t_star_from_refmb(gdir, mbdf)
        bias_std_after, prcp_fac_after = res['std_bias'], res['prcp_fac']
        self.assertLessEqual(np.abs(np.mean(bias_std)), np.abs(np.mean(bias_std_after)))
        self.assertEqual(prcp_fac + 0.1, prcp_fac_after)

        cfg.PARAMS['prcp_scaling_factor'] = prcp_fac - 0.1
        climate.compute_ref_t_stars(gdirs)
        climate.distribute_t_stars(gdirs)
        res = climate.t_star_from_refmb(gdir, mbdf)
        bias_std_after, prcp_fac_after = res['std_bias'], res['prcp_fac']
        self.assertLessEqual(np.abs(np.mean(bias_std)), np.abs(np.mean(bias_std_after)))
        self.assertEqual(prcp_fac - 0.1, prcp_fac_after)

        cfg.PARAMS['prcp_scaling_factor'] = 2.5

    def test_local_mustar(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir, div_id=0)

        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        t_star, bias, prcp_fac = res['t_star'], res['bias'],  res['prcp_fac']
        self.assertEqual(prcp_fac, 2.5)

        t_star = t_star[-1]
        bias = bias[-1]

        climate.local_mustar_apparent_mb(gdir, tstar=t_star, bias=bias,
                                         prcp_fac=prcp_fac)

        df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
        mu_ref = gdir.read_pickle('mu_candidates', div_id=0)[prcp_fac].loc[t_star]
        np.testing.assert_allclose(mu_ref, df['mu_star'][0], atol=1e-3)

        # Check for apparent mb to be zeros
        for i in [0] + list(gdir.divide_ids):
             fls = gdir.read_pickle('inversion_flowlines', div_id=i)
             tmb = 0.
             for fl in fls:
                 self.assertTrue(fl.apparent_mb.shape == fl.widths.shape)
                 tmb += np.sum(fl.apparent_mb * fl.widths)
             np.testing.assert_allclose(tmb, 0., atol=0.01)
             if i == 0: continue
             np.testing.assert_allclose(fls[-1].flux[-1], 0., atol=0.01)

        # ------ Look for gradient
        # which years to look at
        fls = gdir.read_pickle('inversion_flowlines', div_id=0)
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


class TestInversion(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(current_dir, 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
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

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir, div_id=0)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        t_star, bias, prcp_fac = res['t_star'],  res['bias'],  res['prcp_fac']
        t_star = t_star[-1]
        bias = bias[-1]
        climate.local_mustar_apparent_mb(gdir, tstar=t_star, bias=bias,
                                         prcp_fac=prcp_fac)

        # OK. Values from Fischer and Kuhn 2013
        # Area: 8.55
        # meanH = 67+-7
        # Volume = 0.573+-0.063
        # maxH = 242+-13
        inversion.prepare_for_inversion(gdir)

        lens = [len(gdir.read_pickle('centerlines', div_id=i)) for i in [1,2,3]]
        pid = np.argmax(lens) + 1

        # Check how many clips:
        cls = gdir.read_pickle('inversion_input', div_id=pid)
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
            v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
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
        v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
                                              glen_a=glen_a,
                                              write=True)
        np.testing.assert_allclose(ref_v, v)

        lens = [len(gdir.read_pickle('centerlines', div_id=i)) for i in [1,2,3]]
        pid = np.argmax(lens) + 1
        cls = gdir.read_pickle('inversion_output', div_id=pid)
        fls = gdir.read_pickle('inversion_flowlines', div_id=pid)
        maxs = 0.
        for cl, fl in zip(cls, fls):
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
            # This doesn't pass because of smoothing
            # if fl.flows_to is None:
            #     self.assertEqual(cl['volume'][-1], 0)
            #     self.assertEqual(cl['thick'][-1], 0)

        np.testing.assert_allclose(242, maxs, atol=40)

    @is_slow
    def test_distribute(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir, div_id=0)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        t_star, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        t_star = t_star[-1]
        bias = bias[-1]
        climate.local_mustar_apparent_mb(gdir, tstar=t_star, bias=bias,
                                         prcp_fac=prcp_fac)

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
            v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
                                                  glen_a=glen_a)
            return (v - ref_v)**2
        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-1)['x']
        glen_a = cfg.A * out[0]
        fs = cfg.FS * out[1]
        v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
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
        if not HAS_NEW_GDAL:
            np.testing.assert_allclose(np.max(t1), np.max(t2), atol=30)

    @is_slow
    def test_invert_hef_nofs(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir, div_id=0)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        t_star, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        t_star = t_star[-1]
        bias = bias[-1]
        climate.local_mustar_apparent_mb(gdir, tstar=t_star, bias=bias,
                                         prcp_fac=prcp_fac)

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
            v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
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
        v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
                                              glen_a=glen_a,
                                              write=True)
        np.testing.assert_allclose(ref_v, v)

        lens = [len(gdir.read_pickle('centerlines', div_id=i)) for i in [1,2,3]]
        pid = np.argmax(lens) + 1
        cls = gdir.read_pickle('inversion_output', div_id=pid)
        fls = gdir.read_pickle('inversion_flowlines', div_id=pid)
        maxs = 0.
        for cl, fl in zip(cls, fls):
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
        np.testing.assert_allclose(242, maxs, atol=25)

        # check that its not tooo sensitive to the dx
        cfg.PARAMS['flowline_dx'] = 1.
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        climate.process_histalp_nonparallel([gdir])
        climate.mu_candidates(gdir, div_id=0)
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        t_star, bias, prcp_fac = res['t_star'], res['bias'], res['prcp_fac']
        t_star = t_star[-1]
        bias = bias[-1]
        climate.local_mustar_apparent_mb(gdir, tstar=t_star, bias=bias,
                                         prcp_fac=prcp_fac)
        inversion.prepare_for_inversion(gdir)
        v, _ = inversion.invert_parabolic_bed(gdir, fs=fs,
                                              glen_a=glen_a,
                                              write=True)

        np.testing.assert_allclose(ref_v, v, rtol=0.02)
        cls = gdir.read_pickle('inversion_output', div_id=pid)
        maxs = 0.
        for cl in cls:
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max

        np.testing.assert_allclose(242, maxs, atol=25)

    def test_continue_on_error(self):

        cfg.CONTINUE_ON_ERROR = True

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]
        miniglac = shpg.Point(entity.CENLON, entity.CENLAT).buffer(0.0001)
        entity.geometry = miniglac
        entity.RGIID = 'RGI50-11.fake'

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        climate.mu_candidates(gdir, div_id=0)
        climate.local_mustar_apparent_mb(gdir, tstar=1970, bias=0,
                                         prcp_fac=2.)
        inversion.prepare_for_inversion(gdir)
        inversion.volume_inversion(gdir, use_cfg_params={'fd':12, 'fs':0})

        rdir = os.path.join(self.testdir, 'RGI50-11', 'RGI50-11.fake')
        self.assertTrue(os.path.exists(rdir))

        rdir = os.path.join(rdir, 'log')
        self.assertTrue(os.path.exists(rdir))

        self.assertEqual(len(glob.glob(os.path.join(rdir, '*.SUCCESS'))), 2)
        self.assertTrue(len(glob.glob(os.path.join(rdir, '*.ERROR'))) >= 10)

        cfg.CONTINUE_ON_ERROR = False

        # Test the glacier charac
        dfc = utils.glacier_characteristics([gdir])
        self.assertEqual(dfc.terminus_type.values[0], 'Land-terminating')
        self.assertFalse(np.isfinite(dfc.clim_temp_avgh.values[0]))


class TestGrindelInvert(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(current_dir, 'tmp_grindel')
        self.clean_dir()

        # Init
        cfg.initialize()
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
        shutil.copytree(gpath, os.path.join(self.testdir, 'RGI50-11'))

    def _parabolic_bed(self):

        from oggm.core.models import flowline

        map_dx = 100.
        dx = 1.
        nx = 200

        surface_h = np.linspace(3000, 1000, nx)
        bed_h = surface_h
        shape = surface_h * 0. + 3.e-03

        coords = np.arange(0, nx-0.5, 1)
        line = shpg.LineString(np.vstack([coords, coords*0.]).T)
        return [flowline.ParabolicFlowline(line, dx, map_dx, surface_h,
                                           bed_h, shape)]

    @requires_py3
    def test_ideal_glacier(self):

        # we are making a
        glen_a = cfg.A * 1
        from oggm.core.models import flowline, massbalance

        gdir = utils.GlacierDirectory(self.rgin, base_dir=self.testdir)

        fls = self._parabolic_bed()
        mbmod = massbalance.LinearMassBalanceModel(2800.)
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
            mb = mbmod.get_mb(hgt) * cfg.SEC_IN_YEAR * cfg.RHO
            fl.flux = np.zeros(len(fl.surface_h))
            fl.set_apparent_mb(mb)
            flux = fl.flux * (map_dx**2) / cfg.SEC_IN_YEAR / cfg.RHO
            pok = np.nonzero(widths > 10.)
            widths = widths[pok]
            hgt = hgt[pok]
            flux = flux[pok]
            angle = np.arctan(-np.gradient(hgt, dx))  # beware the minus sign
            # Clip flux to 0
            assert not np.any(flux < -0.1)
            # add to output
            cl_dic = dict(dx=dx, flux=flux, width=widths, hgt=hgt,
                          slope_angle=angle, is_last=True)
            towrite.append(cl_dic)

        # Write out
        gdir.write_pickle(towrite, 'inversion_input', div_id=1)
        v, a = inversion.invert_parabolic_bed(gdir, glen_a=glen_a)
        v_km3 = v * 1e-9
        a_km2 = np.sum(widths * dx) * 1e-6
        v_vas = 0.034*(a_km2**1.375)

        np.testing.assert_allclose(v, model.volume_m3, rtol=0.01)

        cl = gdir.read_pickle('inversion_output', div_id=1)[0]
        assert utils.rmsd(cl['thick'], model.fls[0].thick[:len(cl['thick'])]) < 10.

    @requires_py3
    def test_invert_and_run(self):

        from oggm.core.models import flowline, massbalance

        glen_a = cfg.A*2

        gdir = utils.GlacierDirectory(self.rgin, base_dir=self.testdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.compute_downstream_lines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)
        climate.local_mustar_apparent_mb(gdir, tstar=1975, bias=0., prcp_fac=1)
        inversion.prepare_for_inversion(gdir)
        v, a = inversion.invert_parabolic_bed(gdir, glen_a=glen_a)
        cfg.PARAMS['bed_shape'] = 'parabolic'
        flowline.init_present_time_glacier(gdir)
        mb_mod = massbalance.ConstantMassBalanceModel(gdir)
        fls = gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                        fs=0, glen_a=glen_a)
        ref_vol = model.volume_m3
        model.run_until_equilibrium()
        after_vol = model.volume_m3
        np.testing.assert_allclose(ref_vol, after_vol, rtol=0.1)

    @requires_py3
    def test_intersections(self):
        cfg.PARAMS['use_multiple_flowlines'] = True
        gdir = utils.GlacierDirectory(self.rgin, base_dir=self.testdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        geometry.initialize_flowlines(gdir)
        geometry.catchment_area(gdir)
        geometry.catchment_intersections(gdir)
        geometry.catchment_width_geom(gdir)
        geometry.catchment_width_correction(gdir)

        # see that we have as many catchments as flowlines
        fls = gdir.read_pickle('inversion_flowlines', div_id=1)
        gdfc = gpd.read_file(gdir.get_filepath('flowline_catchments',
                                               div_id=1))
        self.assertEqual(len(fls), len(gdfc))
        # and at least as many intersects
        gdfc = gpd.read_file(gdir.get_filepath('catchments_intersects',
                                               div_id=1))
        self.assertGreaterEqual(len(gdfc), len(fls)-1)

        # check touch borders qualitatively
        self.assertGreaterEqual(np.sum(fls[-1].touches_border),  10)


class TestCatching(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(current_dir, 'tmp_errors')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['working_dir'] = current_dir

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_log(self):

        hef_file = get_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        # Make it large to raise an error
        cfg.PARAMS['border'] = 250
        cfg.CONTINUE_ON_ERROR = True

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)