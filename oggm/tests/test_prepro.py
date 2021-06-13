import unittest
import os
import shutil
from distutils.version import LooseVersion
import pytest
import warnings

import shapely.geometry as shpg
import numpy as np
import pandas as pd
import xarray as xr

salem = pytest.importorskip('salem')
rasterio = pytest.importorskip('rasterio')
gpd = pytest.importorskip('geopandas')

# Local imports
import oggm
from oggm.core import (gis, inversion, climate, centerlines,
                       flowline, massbalance)
from oggm.shop import gcm_climate
import oggm.cfg as cfg
from oggm import utils, tasks
from oggm.utils import get_demo_file, tuple2int
from oggm.tests.funcs import (get_test_dir, init_columbia, init_columbia_eb,
                              apply_test_ref_tstars)
from oggm import workflow
from oggm.exceptions import InvalidWorkflowError, MassBalanceCalibrationError

pytestmark = pytest.mark.test_env("prepro")


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
        if c == '':
            continue
        c = c.split(',')
        x.append(float(c[0]))
        y.append(float(c[1]))
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
        cfg.PARAMS['border'] = 20

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_init_gdir(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        assert gdir.has_file('outlines')
        assert gdir.has_file('intersects')
        assert not gdir.has_file('glacier_grid')
        with pytest.warns(FutureWarning):
            gdir.get_filepath('model_run')

    def test_define_region(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        extent = gdir.extent_ll

        tdf = gdir.read_shapefile('outlines')
        myarea = tdf.geometry.area * 10**-6
        np.testing.assert_allclose(myarea, float(tdf['Area']), rtol=1e-2)
        self.assertTrue(gdir.has_file('intersects'))
        np.testing.assert_array_equal(gdir.intersects_ids,
                                      ['RGI50-11.00846', 'RGI50-11.00950'])

        # From string
        gdir = oggm.GlacierDirectory(gdir.rgi_id, base_dir=self.testdir)
        # This is not guaranteed to be equal because of projection issues
        np.testing.assert_allclose(extent, gdir.extent_ll, atol=1e-5)
        assert gdir.grid == gdir.grid_from_params()

        # Change area
        prev_area = gdir.rgi_area_km2
        prev_lon = gdir.cenlon
        prev_lat = gdir.cenlat
        cfg.PARAMS['use_rgi_area'] = False
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir,
                                     reset=True)
        gis.define_glacier_region(gdir)
        # Close but not same
        assert gdir.rgi_area_km2 != prev_area
        assert gdir.cenlon != prev_lon
        assert gdir.cenlat != prev_lat
        np.testing.assert_allclose(gdir.rgi_area_km2, prev_area, atol=0.01)
        np.testing.assert_allclose(gdir.cenlon, prev_lon, atol=1e-2)
        np.testing.assert_allclose(gdir.cenlat, prev_lat, atol=1e-2)

        assert gdir.status == 'Glacier or ice cap'

    def test_reproject(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)

        fn = 'resampled_dem'
        cfg.BASENAMES[fn] = ('res_dem.tif', 'for testing')
        gis.rasterio_to_gdir(gdir, get_demo_file('hef_srtm.tif'), fn)
        with rasterio.open(gdir.get_filepath(fn), 'r',
                           driver='GTiff') as ds:
            totest = ds.read(1).astype(rasterio.float32)
        np.testing.assert_allclose(gis.read_geotiff_dem(gdir), totest)

        # With other resampling less exact
        fn = 'resampled_dem_n'
        cfg.BASENAMES[fn] = ('res_dem.tif', 'for testing')
        gis.rasterio_to_gdir(gdir, get_demo_file('hef_srtm.tif'), fn,
                             resampling='bilinear')
        with rasterio.open(gdir.get_filepath(fn), 'r',
                           driver='GTiff') as ds:
            totest = ds.read(1).astype(rasterio.float32)
        np.testing.assert_allclose(gis.read_geotiff_dem(gdir), totest,
                                   rtol=0.01)

    def test_init_glacier_regions(self):

        hef_rgi = gpd.read_file(get_demo_file('Hintereisferner_RGI5.shp'))
        gdir = workflow.init_glacier_regions(hef_rgi)[0]
        nx, ny = gdir.grid.nx, gdir.grid.ny

        # Change something and note that no change occurs because dem is there
        cfg.PARAMS['border'] = 12
        gdir = workflow.init_glacier_regions(hef_rgi)[0]
        assert nx == gdir.grid.nx
        assert ny == gdir.grid.ny

    def test_divides_as_glaciers(self):

        hef_rgi = gpd.read_file(get_demo_file('divides_alps.shp'))
        hef_rgi = hef_rgi.loc[hef_rgi.RGIId == 'RGI50-11.00897']

        # Rename the RGI ID
        hef_rgi['RGIId'] = ['RGI50-11.00897' + d for d in
                            ['_d01', '_d02', '_d03']]

        # Just check that things are working
        gdirs = workflow.init_glacier_directories(hef_rgi)
        workflow.gis_prepro_tasks(gdirs)

        assert gdirs[0].rgi_id == 'RGI50-11.00897_d01'
        assert gdirs[-1].rgi_id == 'RGI50-11.00897_d03'

    def test_raise_on_duplicate(self):

        hef_rgi = gpd.read_file(get_demo_file('divides_alps.shp'))
        hef_rgi = hef_rgi.loc[hef_rgi.RGIId == 'RGI50-11.00897']

        # Rename the RGI ID
        rids = ['RGI60-11.00897', 'RGI60-11.00897_d01', 'RGI60-11.00897']
        hef_rgi['RGIId'] = rids

        # Just check that things are raised
        with pytest.raises(InvalidWorkflowError):
            workflow.init_glacier_directories(hef_rgi)
        with pytest.raises(InvalidWorkflowError):
            workflow.init_glacier_directories(rids)

    def test_dx_methods(self):
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

        # Test fixed method
        cfg.PARAMS['grid_dx_method'] = 'fixed'
        cfg.PARAMS['fixed_dx'] = 50
        gis.define_glacier_region(gdir)
        mygrid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
        np.testing.assert_allclose(np.abs(mygrid.dx), 50.)

        # Test linear method
        cfg.PARAMS['grid_dx_method'] = 'linear'
        cfg.PARAMS['d1'] = 5.
        cfg.PARAMS['d2'] = 10.
        cfg.PARAMS['dmax'] = 100.
        gis.define_glacier_region(gdir)
        targetdx = np.rint(5. * gdir.rgi_area_km2 + 10.)
        targetdx = np.clip(targetdx, 10., 100.)
        mygrid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
        np.testing.assert_allclose(mygrid.dx, targetdx)

        # Test square method
        cfg.PARAMS['grid_dx_method'] = 'square'
        cfg.PARAMS['d1'] = 5.
        cfg.PARAMS['d2'] = 10.
        cfg.PARAMS['dmax'] = 100.
        gis.define_glacier_region(gdir)
        targetdx = np.rint(5. * np.sqrt(gdir.rgi_area_km2) + 10.)
        targetdx = np.clip(targetdx, 10., 100.)
        mygrid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
        np.testing.assert_allclose(mygrid.dx, targetdx)

    def test_repr(self):
        from textwrap import dedent

        expected = dedent("""\
        <oggm.GlacierDirectory>
          RGI id: RGI50-11.00897
          Region: 11: Central Europe
          Subregion: 11-01: Alps
          Name: Hintereisferner
          Glacier type: Glacier
          Terminus type: Land-terminating
          Status: Glacier or ice cap
          Area: 8.036 km2
          Lon, Lat: (10.7584, 46.8003)
          Grid (nx, ny): (159, 114)
          Grid (dx, dy): (50.0, -50.0)
        """)

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        self.assertEqual(gdir.__repr__(), expected)

    def test_glacierdir(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)

        # this should simply run
        oggm.GlacierDirectory(entity.RGIId, base_dir=self.testdir)

    def test_glacier_masks(self):

        # The GIS was double checked externally with IDL.
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.process_dem(gdir)
        gis.glacier_masks(gdir)
        gis.gridded_attributes(gdir)

        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            glacier_mask = nc.variables['glacier_mask'][:]
            glacier_ext = nc.variables['glacier_ext'][:]
            glacier_ext_erosion = nc.variables['glacier_ext_erosion'][:]
            ice_divides = nc.variables['ice_divides'][:]

        area = np.sum(glacier_mask * gdir.grid.dx**2)
        np.testing.assert_allclose(area*10**-6, gdir.rgi_area_km2,
                                   rtol=1e-1)
        assert np.all(glacier_mask[glacier_ext == 1])
        assert np.all(glacier_mask[glacier_ext_erosion == 1])
        assert np.all(glacier_ext[ice_divides == 1])
        assert np.all(glacier_ext_erosion[ice_divides == 1])
        np.testing.assert_allclose(np.std(glacier_ext_erosion - glacier_ext),
                                   0, atol=0.1)

        entity['RGIFlag'] = '2909'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir, reset=True)
        with pytest.raises(RuntimeError):
            gis.glacier_masks(gdir)

    @pytest.mark.skipif((LooseVersion(rasterio.__version__) <
                         LooseVersion('1.0')),
                        reason='requires rasterio >= 1.0')
    def test_simple_glacier_masks(self):

        # The GIS was double checked externally with IDL.
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.simple_glacier_masks(gdir, write_hypsometry=True)

        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            area = np.sum(nc.variables['glacier_mask'][:] * gdir.grid.dx**2)
            np.testing.assert_allclose(area*10**-6, gdir.rgi_area_km2,
                                       rtol=1e-1)

            # Check that HEF doesn't "badly" need a divide
            mask = nc.variables['glacier_mask'][:]
            ext = nc.variables['glacier_ext'][:]
            dem = nc.variables['topo'][:]
            np.testing.assert_allclose(np.max(dem[mask.astype(bool)]),
                                       np.max(dem[ext.astype(bool)]),
                                       atol=10)

        df = utils.compile_glacier_statistics([gdir], path=False)
        np.testing.assert_allclose(df['dem_max_elev_on_ext'],
                                   df['dem_max_elev'],
                                   atol=10)

        assert np.all(df['dem_max_elev'] > df['dem_max_elev_on_ext'])

        dfh = pd.read_csv(gdir.get_filepath('hypsometry'))

        np.testing.assert_allclose(dfh['Slope'], entity.Slope, atol=0.5)
        np.testing.assert_allclose(dfh['Aspect'], entity.Aspect, atol=5)
        np.testing.assert_allclose(dfh['Zmed'], entity.Zmed, atol=20)
        np.testing.assert_allclose(dfh['Zmax'], entity.Zmax, atol=20)
        np.testing.assert_allclose(dfh['Zmin'], entity.Zmin, atol=20)

        bins = []
        for c in dfh.columns:
            try:
                int(c)
                bins.append(c)
            except ValueError:
                pass
        dfr = pd.read_csv(get_demo_file('Hintereisferner_V5_hypso.csv'))

        dfh.index = ['oggm']
        dft = dfh[bins].T
        dft['ref'] = dfr[bins].T
        assert dft.sum()[0] == 1000
        assert utils.rmsd(dft['ref'], dft['oggm']) < 5

    @pytest.mark.skipif((LooseVersion(rasterio.__version__) <
                         LooseVersion('1.0')),
                        reason='requires rasterio >= 1.0')
    def test_glacier_masks_other_glacier(self):

        # This glacier geometry is simplified by OGGM
        # https://github.com/OGGM/oggm/issues/451
        entity = gpd.read_file(get_demo_file('RGI60-14.03439.shp')).iloc[0]

        cfg.PATHS['dem_file'] = get_demo_file('RGI60-14.03439.tif')
        cfg.PARAMS['border'] = 1

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        # The test below does NOT pass on OGGM
        shutil.copyfile(gdir.get_filepath('gridded_data'),
                        os.path.join(self.testdir, 'default_masks.nc'))
        gis.simple_glacier_masks(gdir, write_hypsometry=True)
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            area = np.sum(nc.variables['glacier_mask'][:] * gdir.grid.dx**2)
            np.testing.assert_allclose(area*10**-6, gdir.rgi_area_km2,
                                       rtol=1e-1)
        shutil.copyfile(gdir.get_filepath('gridded_data'),
                        os.path.join(self.testdir, 'simple_masks.nc'))

        dfh = pd.read_csv(gdir.get_filepath('hypsometry'))
        np.testing.assert_allclose(dfh['Slope'], entity.Slope, atol=1)
        np.testing.assert_allclose(dfh['Aspect'], entity.Aspect, atol=10)
        np.testing.assert_allclose(dfh['Zmed'], entity.Zmed, atol=20)
        np.testing.assert_allclose(dfh['Zmax'], entity.Zmax, atol=20)
        np.testing.assert_allclose(dfh['Zmin'], entity.Zmin, atol=20)

    @pytest.mark.skipif((LooseVersion(rasterio.__version__) <
                         LooseVersion('1.0')),
                        reason='requires rasterio >= 1.0')
    def test_rasterio_glacier_masks(self):

        # The GIS was double checked externally with IDL.
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)

        # specifying a source will look for a DEN in a respective folder
        self.assertRaises(ValueError, gis.rasterio_glacier_mask,
                          gdir, source='SRTM')

        # this should work
        gis.rasterio_glacier_mask(gdir, source=None)

        # read dem mask
        with rasterio.open(gdir.get_filepath('glacier_mask'),
                           'r', driver='GTiff') as ds:
            profile = ds.profile
            data = ds.read(1).astype(profile['dtype'])

        # compare projections
        self.assertEqual(ds.width, gdir.grid.nx)
        self.assertEqual(ds.height, gdir.grid.ny)
        self.assertEqual(ds.transform[0], gdir.grid.dx)
        self.assertEqual(ds.transform[4], gdir.grid.dy)
        # orgin is center for gdir grid but corner for dem_mask, so shift
        self.assertAlmostEqual(ds.transform[2], gdir.grid.x0 - gdir.grid.dx/2)
        self.assertAlmostEqual(ds.transform[5], gdir.grid.y0 - gdir.grid.dy/2)

        # compare dem_mask size with RGI area
        mask_area_km2 = data.sum() * gdir.grid.dx**2 * 1e-6
        self.assertAlmostEqual(mask_area_km2, gdir.rgi_area_km2, 1)

        # how the mask is derived from the outlines it should always be larger
        self.assertTrue(mask_area_km2 > gdir.rgi_area_km2)

        # not sure if we want such a hard coded test, but this will fail if the
        # sample data changes but could also indicate changes in rasterio
        self.assertTrue(data.sum() == 3218)

    def test_intersects(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        self.assertTrue(gdir.has_file('intersects'))

    def test_dem_source_text(self):

        for s in ['TANDEM', 'AW3D30', 'MAPZEN', 'DEM3', 'ASTER', 'SRTM',
                  'RAMP', 'GIMP', 'ARCTICDEM', 'DEM3', 'REMA', 'COPDEM',
                  'NASADEM', 'ALASKA']:
            assert s in gis.DEM_SOURCE_INFO.keys()

    def test_dem_daterange_dateinfo(self):
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)

        # dem_info should return a string
        self.assertIsInstance(gdir.dem_info, str)

        # there is no daterange for demo/custom data
        self.assertIsNone(gdir.dem_daterange)

        # but we can make some
        with open(os.path.join(gdir.dir, 'dem_source.txt'), 'a') as f:
            f.write('Date range: 2000-2000')
        # delete lazy properties
        delattr(gdir, '_lazy_dem_daterange')

        # now call again and check return type
        self.assertIsInstance(gdir.dem_daterange, tuple)
        self.assertTrue(all(isinstance(year, int)
                            for year in gdir.dem_daterange))

    def test_custom_basename(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)

        cfg.add_to_basenames('mybn', 'testfb.pkl', docstr='Some docs')

        out = {'foo': 1.5}
        gdir.write_pickle(out, 'mybn')
        assert gdir.read_pickle('mybn') == out

    def test_gridded_data_var_to_geotiff(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        target_var = 'topo'
        gis.gridded_data_var_to_geotiff(gdir, varname=target_var)
        gtiff_path = os.path.join(gdir.dir, target_var+'.tif')
        assert os.path.exists(gtiff_path)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            gridded_topo = ds[target_var]
            gtiff_ds = salem.open_xr_dataset(gtiff_path)
            assert ds.salem.grid == gtiff_ds.salem.grid
            assert np.allclose(gridded_topo.data, gtiff_ds.data)


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
        _headsi, _ = centerlines._filter_heads(heads[::-1],
                                               heads_height[::-1],
                                               radius, polygon)

        self.assertEqual(_heads, _headsi[::-1])
        self.assertEqual(_heads, [heads[h] for h in [2, 5, 6, 7]])

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

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
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

        self.assertEqual(set(cls), set(centerlines.line_inflows(cls[-1])))

        df = utils.glacier_statistics(gdir)
        # From google map checks
        np.testing.assert_allclose(df['terminus_lon'], 10.80, atol=0.01)
        np.testing.assert_allclose(df['terminus_lat'], 46.81, atol=0.01)

    def test_downstream(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)

        d = gdir.read_pickle('downstream_line')
        cl = gdir.read_pickle('inversion_flowlines')[-1]
        self.assertEqual(
            len(d['full_line'].coords) - len(d['downstream_line'].coords),
            cl.nx)
        np.testing.assert_allclose(d['downstream_line'].length, 12, atol=0.5)

    def test_downstream_bedshape(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        default_b = cfg.PARAMS['border']
        cfg.PARAMS['border'] = 80

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
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
            line = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                    shpg.Point(cur + wi / 2. * n2)])
            from oggm.core.centerlines import line_interpol
            from scipy.interpolate import RegularGridInterpolator
            points = line_interpol(line, 0.5)
            with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
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

    @pytest.mark.slow
    def test_baltoro_centerlines(self):

        cfg.PARAMS['border'] = 2
        cfg.PARAMS['dmax'] = 100
        cfg.PATHS['dem_file'] = get_demo_file('baltoro_srtm_clip.tif')

        b_file = get_demo_file('baltoro_wgs84.shp')
        gdf = gpd.read_file(b_file)

        kienholz_file = get_demo_file('centerlines_baltoro_wgs84.shp')
        kdf = gpd.read_file(kienholz_file)

        # add fake attribs
        area = gdf['AREA']
        del gdf['RGIID']
        del gdf['AREA']
        gdf['RGIId'] = 'RGI50-00.00000'
        gdf['GLIMSId'] = gdf['GLIMSID']
        gdf['Area'] = area
        gdf['CenLat'] = gdf['CENLAT']
        gdf['CenLon'] = gdf['CENLON']
        gdf['BgnDate'] = '-999'
        gdf['Name'] = 'Baltoro'
        gdf['GlacType'] = '0000'
        gdf['Status'] = '0'
        gdf['O1Region'] = '01'
        gdf['O2Region'] = '01'
        entity = gdf.iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)

        my_mask = np.zeros((gdir.grid.ny, gdir.grid.nx), dtype=np.uint8)
        cls = gdir.read_pickle('centerlines')

        assert gdir.rgi_date == 2009

        sub = centerlines.line_inflows(cls[-1])
        self.assertEqual(set(cls), set(sub))
        assert sub[-1] is cls[-1]

        sub = centerlines.line_inflows(cls[-2])
        assert set(sub).issubset(set(cls))
        np.testing.assert_equal(np.unique(sorted([cl.order for cl in sub])),
                                np.arange(cls[-2].order+1))
        assert sub[-1] is cls[-2]

        # Mask
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
            def project(x, y):
                return (np.rint(x).astype(np.int64),
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
        denom = float((na+nc)*(nd+nc)+(na+nb)*(nd+nb))
        hss = float(2.) * ((na*nd)-(nb*nc)) / denom
        if cfg.PARAMS['grid_dx_method'] == 'linear':
            self.assertTrue(hss > 0.53)
        if cfg.PARAMS['grid_dx_method'] == 'fixed':  # quick fix
            self.assertTrue(hss > 0.41)


class TestElevationBandFlowlines(unittest.TestCase):

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
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['baseline_climate'] = ''

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_irregular_grid(self):
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.simple_glacier_masks(gdir)
        centerlines.elevation_band_flowline(gdir)

        df = pd.read_csv(gdir.get_filepath('elevation_band_flowline'), index_col=0)

        # Almost same because of grid VS shape
        np.testing.assert_allclose(df.area.sum(), gdir.rgi_area_m2, rtol=0.01)

        # Length is very different but that's how it is
        np.testing.assert_allclose(df.dx.sum(), entity['Lmax'], rtol=0.2)

        # Slope is similar enough
        avg_slope = np.average(np.rad2deg(df.slope), weights=df.area)
        np.testing.assert_allclose(avg_slope, entity['Slope'], rtol=0.12)

    def test_to_inversion_flowline(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.simple_glacier_masks(gdir)
        centerlines.elevation_band_flowline(gdir)
        centerlines.fixed_dx_elevation_band_flowline(gdir)

        # The tests below are overkill but copied from another test
        # they check everything, which is OK
        area = 0.
        otherarea = 0.
        evenotherarea = 0
        hgt = []
        harea = []

        cls = gdir.read_pickle('inversion_flowlines')
        for cl in cls:
            harea.extend(list(cl.widths * cl.dx))
            hgt.extend(list(cl.surface_h))
            area += np.sum(cl.widths * cl.dx)
            evenotherarea += np.sum(cl.widths_m * cl.dx_meter)
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            otherarea += np.sum(nc.variables['glacier_mask'][:])
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo_smoothed'][:]
        rhgt = topo[np.where(mask)][:]

        tdf = gdir.read_shapefile('outlines')
        np.testing.assert_allclose(area, otherarea, rtol=0.1)
        np.testing.assert_allclose(evenotherarea, gdir.rgi_area_m2)
        area *= gdir.grid.dx ** 2
        otherarea *= gdir.grid.dx ** 2
        np.testing.assert_allclose(area * 10 ** -6, float(tdf['Area']),
                                   rtol=1e-4)

        # Check for area distrib
        bins = np.arange(utils.nicenumber(np.min(hgt), 50, lower=True),
                         utils.nicenumber(np.max(hgt), 50) + 1,
                         50.)
        h1, b = np.histogram(hgt, weights=harea, density=True, bins=bins)
        h2, b = np.histogram(rhgt, density=True, bins=bins)
        assert utils.rmsd(h1 * 100 * 50, h2 * 100 * 50) < 1.5

        # Check that utility function is doing what is expected
        hh, ww = gdir.get_inversion_flowline_hw()
        new_area = np.sum(ww * cl.dx * gdir.grid.dx)
        np.testing.assert_allclose(new_area * 10 ** -6, float(tdf['Area']))

    def test_inversion(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.simple_glacier_masks(gdir)
        centerlines.elevation_band_flowline(gdir)
        centerlines.fixed_dx_elevation_band_flowline(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        v1 = inversion.mass_conservation_inversion(gdir)
        inversion.distribute_thickness_per_altitude(gdir)
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            ds1 = ds.load()

        # Repeat normal workflow
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir, reset=True)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        v2 = inversion.mass_conservation_inversion(gdir)
        inversion.distribute_thickness_per_altitude(gdir)
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            ds2 = ds.load()

        # Total volume is different at only 15%
        np.testing.assert_allclose(v1, v2, rtol=0.15)

        # And the distributed diff is not too large either
        rms = utils.rmsd(ds1.distributed_thickness, ds2.distributed_thickness)
        assert rms < 20

    def test_run(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.simple_glacier_masks(gdir)
        centerlines.elevation_band_flowline(gdir)
        centerlines.fixed_dx_elevation_band_flowline(gdir)
        centerlines.compute_downstream_line(gdir)

        dl = gdir.read_pickle('downstream_line')
        np.testing.assert_allclose(dl['downstream_line'].length, 12, atol=0.5)

        centerlines.compute_downstream_bedshape(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        inversion.mass_conservation_inversion(gdir)
        inversion.filter_inversion_output(gdir)
        flowline.init_present_time_glacier(gdir)
        model = flowline.run_random_climate(gdir, nyears=50, y0=1985)

        fl = model.fls[-1]
        assert np.all(fl.is_trapezoid[:30])

        with xr.open_dataset(gdir.get_filepath('model_diagnostics')) as ds:
            # it's running and it is retreating
            assert ds.volume_m3[-1] < ds.volume_m3[0]
            assert ds.length_m[-1] < ds.length_m[0]


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

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.catchment_area(gdir)

        cis = gdir.read_pickle('geometries')['catchment_indices']

        # The catchment area must be as big as expected
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            mask = nc.variables['glacier_mask'][:]

        mymask_a = mask * 0
        mymask_b = mask * 0
        for i, ci in enumerate(cis):
            mymask_a[tuple(ci.T)] += 1
            mymask_b[tuple(ci.T)] = i+1
        self.assertTrue(np.max(mymask_a) == 1)
        np.testing.assert_allclose(mask, mymask_a)

    def test_flowlines(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
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

        d = gdir.get_diagnostics()
        assert d['perc_invalid_flowline'] > 0.1

        df = utils.compile_glacier_statistics([gdir], path=False)
        assert np.all(df['dem_source'] == 'USER')
        assert np.all(df['perc_invalid_flowline'] > 0.1)
        assert np.all(df['dem_perc_area_above_max_elev_on_ext'] < 0.1)

    def test_geom_width(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)

    def test_width(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        area = 0.
        otherarea = 0.
        evenotherarea = 0
        hgt = []
        harea = []

        cls = gdir.read_pickle('inversion_flowlines')
        for cl in cls:
            harea.extend(list(cl.widths * cl.dx))
            hgt.extend(list(cl.surface_h))
            area += np.sum(cl.widths * cl.dx)
            evenotherarea += np.sum(cl.widths_m * cl.dx_meter)
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            otherarea += np.sum(nc.variables['glacier_mask'][:])

        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo_smoothed'][:]
        rhgt = topo[np.where(mask)][:]

        tdf = gdir.read_shapefile('outlines')
        np.testing.assert_allclose(area, otherarea, rtol=0.1)
        np.testing.assert_allclose(evenotherarea, gdir.rgi_area_m2)
        area *= (gdir.grid.dx) ** 2
        otherarea *= (gdir.grid.dx) ** 2
        np.testing.assert_allclose(area * 10**-6, float(tdf['Area']),
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
        np.testing.assert_allclose(new_area * 10**-6, float(tdf['Area']))

    def test_nodivides_correct_slope(self):

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['border'] = 40

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

        gis.define_glacier_region(gdir)
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
        cfg.PARAMS['baseline_climate'] = ''

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

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        climate.process_custom_climate_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1802)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2003)

        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        f = os.path.join(gdir.dir, 'climate_historical.nc')
        with utils.ncDataset(f) as nc_r:
            self.assertTrue(ref_h == nc_r.ref_hgt)
            np.testing.assert_allclose(ref_t, nc_r.variables['temp'][:])
            np.testing.assert_allclose(ref_p, nc_r.variables['prcp'][:])

    @pytest.mark.slow
    def test_distribute_climate_grad(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        cfg.PARAMS['temp_use_local_gradient'] = True
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        climate.process_custom_climate_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1802)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2003)

        with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
            grad = ds['gradient'].data
            try:
                assert np.std(grad) > 0.0001
            except TypeError:
                pass
        cfg.PARAMS['temp_use_local_gradient'] = False

    def test_distribute_climate_parallel(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        climate.process_custom_climate_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1802)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2003)

        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        f = os.path.join(gdir.dir, 'climate_historical.nc')
        with utils.ncDataset(f) as nc_r:
            self.assertTrue(ref_h == nc_r.ref_hgt)
            np.testing.assert_allclose(ref_t, nc_r.variables['temp'][:])
            np.testing.assert_allclose(ref_p, nc_r.variables['prcp'][:])

    def test_distribute_climate_cru(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gdirs.append(gdir)
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir_cru)
        gis.define_glacier_region(gdir)
        gdirs.append(gdir)

        climate.process_custom_climate_data(gdirs[0])
        cfg.PATHS['climate_file'] = ''
        cfg.PARAMS['baseline_climate'] = 'CRU'
        tasks.process_cru_data(gdirs[1])
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1902)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2014)

        gdh = gdirs[0]
        gdc = gdirs[1]
        f1 = os.path.join(gdh.dir, 'climate_historical.nc')
        f2 = os.path.join(gdc.dir, 'climate_historical.nc')
        with xr.open_dataset(f1) as nc_h:
            with xr.open_dataset(f2) as nc_c:
                # put on the same altitude
                # (using default gradient because better)
                temp_cor = nc_c.temp - 0.0065 * (nc_h.ref_hgt - nc_c.ref_hgt)
                totest = temp_cor - nc_h.temp
                self.assertTrue(totest.mean() < 0.5)
                # precip
                totest = nc_c.prcp - nc_h.prcp
                self.assertTrue(totest.mean() < 100)

    def test_distribute_climate_dummy(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gdirs.append(gdir)
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir_cru)
        gis.define_glacier_region(gdir)
        gdirs.append(gdir)

        tasks.process_dummy_cru_file(gdirs[0], seed=0)
        cfg.PATHS['climate_file'] = ''
        cfg.PARAMS['baseline_climate'] = 'CRU'
        tasks.process_cru_data(gdirs[1])
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1902)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2014)

        gdh = gdirs[0]
        gdc = gdirs[1]
        f1 = os.path.join(gdh.dir, 'climate_historical.nc')
        f2 = os.path.join(gdc.dir, 'climate_historical.nc')
        with xr.open_dataset(f1) as nc_d:
            with xr.open_dataset(f2) as nc_c:
                # same altitude
                assert nc_d.ref_hgt == nc_c.ref_hgt
                np.testing.assert_allclose(nc_d.temp.mean(), nc_c.temp.mean(),
                                           atol=0.2)
                np.testing.assert_allclose(nc_d.temp.mean(), nc_c.temp.mean(),
                                           rtol=0.1)

                an1 = nc_d.temp.groupby('time.month').mean()
                an2 = nc_c.temp.groupby('time.month').mean()
                np.testing.assert_allclose(an1, an2, atol=1)

                an1 = nc_d.prcp.groupby('time.month').mean()
                an2 = nc_c.prcp.groupby('time.month').mean()
                np.testing.assert_allclose(an1, an2, rtol=0.2)

    @pytest.mark.slow
    def test_distribute_climate_historical_alps_new(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdirs = []

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gdirs.append(gdir)
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir_cru)
        gis.define_glacier_region(gdir)
        gdirs.append(gdir)

        climate.process_custom_climate_data(gdirs[0])
        cfg.PATHS['climate_file'] = ''
        cfg.PARAMS['baseline_climate'] = 'HISTALP'
        tasks.process_histalp_data(gdirs[1], y0=1850, y1=2003)
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1851)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2003)

        gdh = gdirs[0]
        gdc = gdirs[1]
        f1 = os.path.join(gdh.dir, 'climate_historical.nc')
        f2 = os.path.join(gdc.dir, 'climate_historical.nc')
        with xr.open_dataset(f1) as nc_h:
            with xr.open_dataset(f2) as nc_c:
                nc_hi = nc_h.isel(time=slice(49*12, 2424))
                np.testing.assert_allclose(nc_hi['temp'], nc_c['temp'])
                # for precip the data changed in between versions, we
                # can't test for absolute equality
                np.testing.assert_allclose(nc_hi['prcp'].mean(),
                                           nc_c['prcp'].mean(),
                                           atol=1)
                np.testing.assert_allclose(nc_hi.ref_pix_dis,
                                           nc_c.ref_pix_dis)

    def test_sh(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

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

        gis.define_glacier_region(gdir)
        gdirs.append(gdir)
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir_cru)
        assert gdir.hemisphere == 'nh'
        gdir.hemisphere = 'sh'
        gis.define_glacier_region(gdir)
        gdirs.append(gdir)
        climate.process_custom_climate_data(gdirs[0])
        ci = gdirs[0].get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1803)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2002)

        cfg.PATHS['climate_file'] = ''
        cfg.PARAMS['baseline_climate'] = 'CRU'
        tasks.process_cru_data(gdirs[1])
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1902)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2014)

        gdh = gdirs[0]
        gdc = gdirs[1]
        with xr.open_dataset(
                os.path.join(gdh.dir, 'climate_historical.nc')) as nc_h:

            assert nc_h['time.month'][0] == 4
            assert nc_h['time.year'][0] == 1802
            assert nc_h['time.month'][-1] == 3
            assert nc_h['time.year'][-1] == 2002

            with xr.open_dataset(
                    os.path.join(gdc.dir, 'climate_historical.nc')) as nc_c:

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

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        climate.process_custom_climate_data(gdir)

        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]
            ref_t = np.where(ref_t < cfg.PARAMS['temp_melt'], 0,
                             ref_t - cfg.PARAMS['temp_melt'])

        hgts = np.array([ref_h, ref_h, -8000, 8000])
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts)
        prcp /= cfg.PARAMS['prcp_scaling_factor']

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
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts,
                                                        year_range=yr)
        prcp /= cfg.PARAMS['prcp_scaling_factor']
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
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts,
                                                        year_range=yr)
        prcp /= cfg.PARAMS['prcp_scaling_factor']
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

        cfg.PARAMS['prcp_scaling_factor'] = 1

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        climate.process_custom_climate_data(gdir)

        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]
            ref_t = np.where(ref_t <= cfg.PARAMS['temp_melt'], 0,
                             ref_t - cfg.PARAMS['temp_melt'])

        # NORMAL --------------------------------------------------------------
        hgts = np.array([ref_h, ref_h, -8000, 8000])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts)

        ref_nt = 202
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))

        yr = [1802, 1802]
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
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
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
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
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
                                                                flatten=True)

        ref_nt = 202
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(temp.shape == (ref_nt,))
        self.assertTrue(prcp.shape == (ref_nt,))

        yr = [1802, 1802]
        hgts = np.array([ref_h])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
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
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
                                                                year_range=yr,
                                                                flatten=True)
        np.testing.assert_allclose(prcp[:], np.sum(ref_p[0:12]))

    def test_mu_candidates(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        with pytest.warns(FutureWarning):
            se = climate.glacier_mu_candidates(gdir)

        self.assertTrue(se.index[0] == 1802)
        self.assertTrue(se.index[-1] == 2003)

        df = pd.DataFrame()
        df['mu'] = se

        # Check that the moovin average of temp is negatively correlated
        # with the mus
        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_t = nc_r.variables['temp'][:, 1, 1]
        ref_t = np.mean(ref_t.reshape((len(df), 12)), 1)
        ma = np.convolve(ref_t, np.ones(31) / float(31), 'same')
        df['temp'] = ma
        df = df.dropna()
        self.assertTrue(np.corrcoef(df['mu'], df['temp'])[0, 1] < -0.75)

    def test_find_tstars(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        with pytest.warns(FutureWarning):
            mu_yr_clim = climate.glacier_mu_candidates(gdir)

        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

        res = climate.t_star_from_refmb(gdir, mbdf=mbdf, glacierwide=True)

        t_star, bias = res['t_star'], res['bias']
        y, t, p = climate.mb_yearly_climate_on_glacier(gdir)

        # which years to look at
        selind = np.searchsorted(y, mbdf.index)
        t = t[selind]
        p = p[selind]

        mb_per_mu = p - mu_yr_clim.loc[t_star] * t
        md = utils.md(mbdf, mb_per_mu)
        self.assertTrue(np.abs(md/np.mean(mbdf)) < 0.1)
        r = utils.corrcoef(mbdf, mb_per_mu)
        self.assertTrue(r > 0.8)

        # test crop years
        cfg.PARAMS['tstar_search_window'] = [1902, 0]
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf)

        t_star, bias = res['t_star'], res['bias']
        mb_per_mu = p - mu_yr_clim.loc[t_star] * t
        md = utils.md(mbdf, mb_per_mu)
        self.assertTrue(np.abs(md/np.mean(mbdf)) < 0.1)
        r = utils.corrcoef(mbdf, mb_per_mu)
        self.assertTrue(r > 0.8)
        self.assertTrue(t_star >= 1902)

        # test distribute
        cfg.PARAMS['run_mb_calibration'] = True
        climate.compute_ref_t_stars([gdir])
        climate.local_t_star(gdir)
        cfg.PARAMS['tstar_search_window'] = [0, 0]

        df = gdir.read_json('local_mustar')
        np.testing.assert_allclose(df['t_star'], t_star)
        np.testing.assert_allclose(df['bias'], bias)

    def test_climate_qc(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)

        # Raise ref hgt a lot
        fc = gdir.get_filepath('climate_historical')
        with utils.ncDataset(fc, 'a') as nc:
            nc.ref_hgt = 10000
        climate.historical_climate_qc(gdir)

        with utils.ncDataset(fc, 'r') as nc:
            assert (nc.ref_hgt - nc.uncorrected_ref_hgt) < -4000

        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'],
                                        glacierwide=True)

        cfg.PARAMS['min_mu_star'] = 10
        with pytest.raises(MassBalanceCalibrationError):
            climate.local_t_star(gdir, tstar=res['t_star'], bias=res['bias'])

        cfg.PARAMS['min_mu_star'] = 5
        climate.local_t_star(gdir, tstar=res['t_star'], bias=res['bias'])
        climate.mu_star_calibration(gdir)

        from oggm.core.massbalance import MultipleFlowlineMassBalance

        mb = MultipleFlowlineMassBalance(gdir, use_inversion_flowlines=True)
        mbdf['CALIB_1'] = mb.get_specific_mb(year=mbdf.index.values)

        # Lower ref hgt a lot
        fc = gdir.get_filepath('climate_historical')
        with utils.ncDataset(fc, 'a') as nc:
            nc.ref_hgt = 0
        climate.historical_climate_qc(gdir)

        with utils.ncDataset(fc, 'r') as nc:
            assert (nc.ref_hgt - nc.uncorrected_ref_hgt) > 2500

        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'],
                                        glacierwide=True)
        climate.local_t_star(gdir, tstar=res['t_star'], bias=res['bias'])
        climate.mu_star_calibration(gdir)

        mb = MultipleFlowlineMassBalance(gdir, use_inversion_flowlines=True)
        mbdf['CALIB_2'] = mb.get_specific_mb(year=mbdf.index.values)

        mm = mbdf[['ANNUAL_BALANCE', 'CALIB_1', 'CALIB_2']].mean()
        np.testing.assert_allclose(mm['ANNUAL_BALANCE'], mm['CALIB_1'],
                                   rtol=1e-5)
        np.testing.assert_allclose(mm['ANNUAL_BALANCE'], mm['CALIB_2'],
                                   rtol=1e-5)

        cor = mbdf[['ANNUAL_BALANCE', 'CALIB_1', 'CALIB_2']].corr()
        assert cor.min().min() > 0.35

    @pytest.mark.slow
    def test_find_tstars_multiple_mus(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir, y0=1940, y1=2000)

        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

        # Normal flowlines, i.e should be equivalent
        res_new = climate.t_star_from_refmb(gdir, mbdf=mbdf, glacierwide=False)
        mb_new = res_new['avg_mb_per_mu']
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf, glacierwide=True)
        mb = res['avg_mb_per_mu']

        np.testing.assert_allclose(res['t_star'], res_new['t_star'])
        np.testing.assert_allclose(res['bias'], res_new['bias'], atol=1e-3)
        np.testing.assert_allclose(mb, mb_new, atol=1e-3)

        # Artificially make some arms even lower to have multiple branches
        # This is not equivalent any more
        fls = gdir.read_pickle('inversion_flowlines')
        assert fls[0].flows_to is fls[-1]
        assert fls[1].flows_to is fls[-1]
        fls[0].surface_h -= 700
        fls[1].surface_h -= 700
        gdir.write_pickle(fls, 'inversion_flowlines')

        res_new = climate.t_star_from_refmb(gdir, mbdf=mbdf, glacierwide=False)
        mb_new = res['avg_mb_per_mu']
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf, glacierwide=True)
        mb = res['avg_mb_per_mu']

        np.testing.assert_allclose(res['bias'], res_new['bias'], atol=20)
        np.testing.assert_allclose(mb, mb_new, rtol=2e-1, atol=20)

    def test_local_t_star(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        cfg.PARAMS['prcp_scaling_factor'] = 2.9

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        with pytest.warns(FutureWarning):
            mu_ref = climate.glacier_mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']

        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)

        mu_ref = mu_ref.loc[t_star]

        # Check for apparent mb to be zeros
        fls = gdir.read_pickle('inversion_flowlines')
        tmb = 0.
        for fl in fls:
            self.assertTrue(fl.apparent_mb.shape == fl.widths.shape)
            np.testing.assert_allclose(mu_ref, fl.mu_star, atol=1e-3)
            tmb += np.sum(fl.apparent_mb * fl.widths)
            assert not fl.flux_needs_correction
        np.testing.assert_allclose(tmb, 0., atol=0.01)
        np.testing.assert_allclose(fls[-1].flux[-1], 0., atol=0.01)

        df = gdir.read_json('local_mustar')
        assert df['mu_star_allsame']
        np.testing.assert_allclose(mu_ref, df['mu_star_flowline_avg'],
                                   atol=1e-3)
        np.testing.assert_allclose(mu_ref, df['mu_star_glacierwide'],
                                   atol=1e-3)

        # ------ Look for gradient
        # which years to look at
        fls = gdir.read_pickle('inversion_flowlines')
        mb_on_h = np.array([])
        h = np.array([])
        for fl in fls:
            y, t, p = climate.mb_yearly_climate_on_height(gdir, fl.surface_h)
            selind = np.searchsorted(y, mbdf.index)
            t = np.mean(t[:, selind], axis=1)
            p = np.mean(p[:, selind], axis=1)
            mb_on_h = np.append(mb_on_h, p - mu_ref * t)
            h = np.append(h, fl.surface_h)

        dfg = gdir.get_ref_mb_profile().mean()

        # Take the altitudes below 3100 and fit a line
        dfg = dfg[dfg.index < 3100]
        pok = np.where(h < 3100)
        from scipy.stats import linregress
        slope_obs, _, _, _, _ = linregress(dfg.index, dfg.values)
        slope_our, _, _, _, _ = linregress(h[pok], mb_on_h[pok])
        np.testing.assert_allclose(slope_obs, slope_our, rtol=0.1)

        cfg.PARAMS['prcp_scaling_factor'] = 2.5

    @pytest.mark.slow
    def test_local_t_star_fallback(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        _prcp_sf = cfg.PARAMS['prcp_scaling_factor']
        # small scaling factor will force small mu* to compensate lack of PRCP
        cfg.PARAMS['prcp_scaling_factor'] = 1e-3
        cfg.PARAMS['continue_on_error'] = True

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        with pytest.warns(FutureWarning):
            climate.glacier_mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']

        # here, an error should occur as mu* < cfg.PARAMS['min_mu_star']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        # check if file has been written
        assert os.path.isfile(gdir.get_filepath('local_mustar'))

        climate.mu_star_calibration(gdir)

        df = gdir.read_json('local_mustar')
        assert np.isnan(df['bias'])
        assert np.isnan(df['t_star'])
        assert np.isnan(df['mu_star_glacierwide'])
        assert np.isnan(df['mu_star_flowline_avg'])
        assert np.isnan(df['mu_star_allsame'])
        assert np.isnan(df['mu_star_per_flowline']).all()
        assert df['rgi_id'] == gdir.rgi_id

        cfg.PARAMS['prcp_scaling_factor'] = _prcp_sf
        cfg.PARAMS['continue_on_error'] = False

    def test_ref_mb_glaciers(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

        rids = utils.get_ref_mb_glaciers_candidates()
        assert len(rids) > 200

        rids = utils.get_ref_mb_glaciers_candidates(gdir.rgi_version)
        assert len(rids) > 200

        assert len(cfg.DATA) >= 2

        with pytest.raises(InvalidWorkflowError):
            utils.get_ref_mb_glaciers([gdir])

        climate.process_custom_climate_data(gdir)
        ref_gd = utils.get_ref_mb_glaciers([gdir])
        assert len(ref_gd) == 1

    def test_fake_ref_mb_glacier(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir1 = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        climate.process_custom_climate_data(gdir1)
        entity['RGIId'] = 'RGI50-11.99999'
        gdir2 = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        climate.process_custom_climate_data(gdir2)

        ref_gd = utils.get_ref_mb_glaciers([gdir2])
        assert len(ref_gd) == 0

        gdir2.set_ref_mb_data(gdir1.get_ref_mb_data())

        ref_gd = utils.get_ref_mb_glaciers([gdir2])
        assert len(ref_gd) == 0

        cfg.DATA['RGI50_ref_ids'].append('RGI50-11.99999')

        ref_gd = utils.get_ref_mb_glaciers([gdir2])
        assert len(ref_gd) == 1

    @pytest.mark.slow
    def test_automated_workflow(self):

        cfg.PATHS['climate_file'] = ''
        cfg.PARAMS['baseline_climate'] = 'CRU'

        # Bck change
        with pytest.raises(ValueError):
            cfg.PARAMS['baseline_y0'] = 1

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        entity['RGIId'] = 'RGI60-11.00897'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        assert gdir.rgi_version == '60'
        gis.define_glacier_region(gdir)
        workflow.gis_prepro_tasks([gdir])

        with pytest.raises(InvalidWorkflowError):
            workflow.climate_tasks([gdir])

        # We copy the files
        apply_test_ref_tstars()
        workflow.climate_tasks([gdir])

        # Test match geod - method 1
        workflow.match_regional_geodetic_mb([gdir], rgi_reg='11',
                                            period='2000-01-01_2010-01-01')
        df = 'table_hugonnet_regions_10yr_20yr_ar6period.csv'
        df = pd.read_csv(utils.get_demo_file(df))
        df = df.loc[df.period == '2000-01-01_2010-01-01'].set_index('reg')
        smb_ref = df.loc[int('11'), 'dmdtda']
        df = utils.compile_fixed_geometry_mass_balance([gdir])
        mb = df.loc[2000:2009].mean()
        np.testing.assert_allclose(mb, smb_ref)

        workflow.match_regional_geodetic_mb(gdir, rgi_reg='11',
                                            dataset='zemp')
        df = 'zemp_ref_2006_2016.csv'
        df = pd.read_csv(utils.get_demo_file(df), index_col=0)
        smb_ref = df.loc[int('11'), 'SMB'] * 1000
        df = utils.compile_fixed_geometry_mass_balance([gdir])
        mb = df.loc[2006:2015].mean()
        np.testing.assert_allclose(mb, smb_ref)

        # Test match geod - method 2
        workflow.match_geodetic_mb_for_selection([gdir],
                                                 period='2000-01-01_2010-01-01')

        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/'
        file_name = 'hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide_filled.hdf'
        df = pd.read_hdf(utils.file_downloader(base_url + file_name))
        df = df.loc[df['period'] == '2000-01-01_2010-01-01']
        rdf = df.loc[['RGI60-11.00897']]

        mbdf = utils.compile_fixed_geometry_mass_balance([gdir])
        mb = mbdf.loc[2000:2009].mean()

        wgms = gdir.get_ref_mb_data().loc[2000:2009]

        np.testing.assert_allclose(mb, rdf.dmdtda * 1000)
        np.testing.assert_allclose(mb, wgms['ANNUAL_BALANCE'].mean(), atol=30)

        # Check error management
        # We trick with a glaciers whith no valid data
        rdf['is_cor'] = True
        fpath = os.path.join(self.testdir, 'test_hugo.hdf')
        rdf.to_hdf(fpath, key='df')

        # This raises
        with pytest.raises(InvalidWorkflowError):
            workflow.match_geodetic_mb_for_selection([gdir], file_path=fpath,
                                                     period='2000-01-01_2010-01-01')

        # This doesnt
        workflow.match_geodetic_mb_for_selection([gdir], file_path=fpath,
                                                 fail_safe=True,
                                                 period='2000-01-01_2010-01-01')

        mbdf = utils.compile_fixed_geometry_mass_balance([gdir])
        mb = mbdf.loc[2000:2009].mean()
        np.testing.assert_allclose(mb, rdf.dmdtda * 1000)

        # This raises as well, for different reasons
        cfg.PARAMS['prcp_scaling_factor'] = 1.8
        with pytest.raises(MassBalanceCalibrationError):
            workflow.climate_tasks([gdir])


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
        cfg.PARAMS['baseline_climate'] = ''
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

        cfg.PARAMS['correct_for_neg_flux'] = False

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        climate.local_t_star(gdir, tstar=1931, bias=0)
        climate.mu_star_calibration(gdir)

        fls1 = gdir.read_pickle('inversion_flowlines')
        assert np.any([fl.flux_needs_correction for fl in fls1])

        cfg.PARAMS['filter_for_neg_flux'] = True
        climate.mu_star_calibration(gdir)

        fls = gdir.read_pickle('inversion_flowlines')
        assert len(fls) < len(fls1)
        assert not np.any([fl.flux_needs_correction for fl in fls])

    @pytest.mark.slow
    def test_correct(self):

        entity = gpd.read_file(get_demo_file('rgi_oetztal.shp'))
        entity = entity.loc[entity.RGIId == 'RGI50-11.00666'].iloc[0]

        cfg.PARAMS['correct_for_neg_flux'] = False

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)

        # Artificially make some arms even lower to have multiple branches
        fls = gdir.read_pickle('inversion_flowlines')
        assert fls[2].flows_to is fls[3]
        assert fls[1].flows_to is fls[-1]
        fls[1].surface_h -= 500
        fls[2].surface_h -= 500
        fls[3].surface_h -= 500
        gdir.write_pickle(fls, 'inversion_flowlines')

        climate.local_t_star(gdir, tstar=1931, bias=0)
        climate.mu_star_calibration(gdir)

        fls1 = gdir.read_pickle('inversion_flowlines')
        assert np.sum([fl.flux_needs_correction for fl in fls1]) == 3

        cfg.PARAMS['correct_for_neg_flux'] = True
        climate.mu_star_calibration(gdir)

        fls = gdir.read_pickle('inversion_flowlines')
        assert len(fls) == len(fls1)
        assert not np.any([fl.flux_needs_correction for fl in fls])
        assert np.all([fl.mu_star_is_valid for fl in fls])
        mus = np.array([fl.mu_star for fl in fls])
        assert np.max(mus[[1, 2, 3]]) < (np.max(mus[[0, -1]]) / 2)

        df = gdir.read_json('local_mustar')
        mu_star_gw = df['mu_star_glacierwide']

        assert np.max(mus[[1, 2, 3]]) < mu_star_gw
        assert np.min(mus[[0, -1]]) > mu_star_gw

        bias = df['bias']
        np.testing.assert_allclose(bias, 0)

        from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                           ConstantMassBalance)
        mb_mod = MultipleFlowlineMassBalance(gdir, fls=fls, bias=0,
                                             mb_model_class=ConstantMassBalance
                                             )

        for mb, fl in zip(mb_mod.flowline_mb_models[1:4], fls[1:4]):
            mbs = mb.get_specific_mb(fl.surface_h, fl.widths)
            np.testing.assert_allclose(mbs, 0, atol=1e-1)

        np.testing.assert_allclose(mb_mod.get_specific_mb(), 0, atol=1e-1)

    @pytest.mark.slow
    def test_and_compare_two_methods(self):

        entity = gpd.read_file(get_demo_file('rgi_oetztal.shp'))
        entity = entity.loc[entity.RGIId == 'RGI50-11.00666'].iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)

        # Artificially make some arms even lower to have multiple branches
        fls = gdir.read_pickle('inversion_flowlines')
        assert fls[2].flows_to is fls[3]
        assert fls[1].flows_to is fls[-1]
        fls[1].surface_h -= 500
        fls[2].surface_h -= 500
        fls[3].surface_h -= 500
        gdir.write_pickle(fls, 'inversion_flowlines')

        climate.local_t_star(gdir, tstar=1931, bias=0)
        climate.mu_star_calibration(gdir)

        fls = gdir.read_pickle('inversion_flowlines')

        # These are the params:
        # rgi_id                RGI50-11.00666
        # t_star                          1931
        # bias                               0
        # mu_star_glacierwide          133.235
        # mustar_flowline_001          165.673
        # mustar_flowline_002           46.728
        # mustar_flowline_003           63.759
        # mustar_flowline_004          66.3795
        # mustar_flowline_005          165.673
        # mu_star_flowline_avg         146.924
        # mu_star_allsame                False

        from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                           PastMassBalance)

        mb_mod_1 = PastMassBalance(gdir, check_calib_params=False)
        mb_mod_2 = MultipleFlowlineMassBalance(gdir, fls=fls)

        years = np.arange(1951, 2000)
        mbs1 = mb_mod_1.get_specific_mb(fls=fls, year=years)
        mbs2 = mb_mod_2.get_specific_mb(fls=fls, year=years)

        # The two are NOT equivalent because of non-linear effects,
        # but they are close:
        assert utils.rmsd(mbs1, mbs2) < 50


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
        cfg.PARAMS['baseline_climate'] = ''
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
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)

        # OK. Values from Fischer and Kuhn 2013
        # Area: 8.55
        # meanH = 67+-7
        # Volume = 0.573+-0.063
        # maxH = 242+-13
        inversion.prepare_for_inversion(gdir)
        # Check how many clips:
        cls = gdir.read_pickle('inversion_input')
        nabove = 0
        maxs = 0.
        npoints = 0.
        for cl in cls:
            # Clip slope to avoid negative and small slopes
            slope = cl['slope_angle']
            nm = np.where(slope < np.deg2rad(2.))
            nabove += len(nm[0])
            npoints += len(slope)
            _max = np.max(slope)
            if _max > maxs:
                maxs = _max

        self.assertTrue(nabove == 0)
        self.assertTrue(np.rad2deg(maxs) < 40.)

        ref_v = 0.573 * 1e9

        glen_a = 2.4e-24
        fs = 5.7e-20

        def to_optimize(x):
            v = inversion.mass_conservation_inversion(gdir, fs=fs * x[1],
                                                      glen_a=glen_a * x[0])
            return (v - ref_v)**2

        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-4)['x']
        self.assertTrue(out[0] > 0.1)
        self.assertTrue(out[1] > 0.1)
        self.assertTrue(out[0] < 1.1)
        self.assertTrue(out[1] < 1.1)
        glen_a = glen_a * out[0]
        fs = fs * out[1]
        v = inversion.mass_conservation_inversion(gdir, fs=fs,
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

        np.testing.assert_allclose(242, maxs, atol=40)

        maxs = 0.
        v = 0.
        cls = gdir.read_pickle('inversion_output')
        for cl in cls:
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
            v += np.nansum(cl['volume'])
        np.testing.assert_allclose(242, maxs, atol=40)
        np.testing.assert_allclose(ref_v, v)
        np.testing.assert_allclose(ref_v, inversion.get_inversion_volume(gdir))

        # Sanity check - velocities
        inv = gdir.read_pickle('inversion_output')[-1]

        # vol in m3 and dx in m -> section in m2
        section = inv['volume'] / inv['dx']

        # Flux in m3 s-1 -> convert to velocity m s-1
        velocity = inv['flux'] / section

        # Then in m yr-1
        velocity *= cfg.SEC_IN_YEAR

        # Some reference value I just computed - see if other computers agree
        np.testing.assert_allclose(np.mean(velocity[:-1]), 37, atol=5)

        inversion.compute_velocities(gdir, fs=fs, glen_a=glen_a)

        inv = gdir.read_pickle('inversion_output')[-1]

        np.testing.assert_allclose(velocity, inv['u_integrated'])

    @pytest.mark.slow
    def test_invert_hef_from_consensus(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        entity['RGIId'] = 'RGI60-11.00897'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        df = workflow.calibrate_inversion_from_consensus(gdir)
        np.testing.assert_allclose(df.vol_itmix_m3, df.vol_oggm_m3, rtol=0.01)
        # Make it fail
        with pytest.raises(ValueError):
            a = (0.1, 3)
            workflow.calibrate_inversion_from_consensus(gdir,
                                                        a_bounds=a)

        a = (0.1, 5)
        df = workflow.calibrate_inversion_from_consensus(gdir,
                                                         a_bounds=a,
                                                         error_on_mismatch=False)
        np.testing.assert_allclose(df.vol_itmix_m3, df.vol_oggm_m3, rtol=0.07)

        # With fs it can work
        a = (0.1, 3)
        df = workflow.calibrate_inversion_from_consensus(gdir,
                                                         a_bounds=a,
                                                         apply_fs_on_mismatch=True)
        np.testing.assert_allclose(df.vol_itmix_m3, df.vol_oggm_m3, rtol=0.01)

    @pytest.mark.slow
    def test_invert_hef_shapes(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        entity['RGIId'] = 'RGI60-11.00897'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)

        cfg.PARAMS['inversion_fs'] = 5.7e-20
        cfg.PARAMS['inversion_glen_a'] = 2.4e-24

        inversion.prepare_for_inversion(gdir,
                                        invert_with_rectangular=False,
                                        invert_with_trapezoid=False)
        vp = inversion.mass_conservation_inversion(gdir)

        inversion.prepare_for_inversion(gdir, invert_all_trapezoid=True)
        vt1 = inversion.mass_conservation_inversion(gdir)

        cfg.PARAMS['trapezoid_lambdas'] = 1
        inversion.prepare_for_inversion(gdir, invert_all_trapezoid=True)
        vt2 = inversion.mass_conservation_inversion(gdir)

        inversion.prepare_for_inversion(gdir, invert_all_rectangular=True)
        vr = inversion.mass_conservation_inversion(gdir)

        np.testing.assert_allclose(vp/vr, 0.75, atol=0.02)
        np.testing.assert_allclose(vt1/vr, 0.86, atol=0.02)
        np.testing.assert_allclose(vt2/vr, 0.93, atol=0.02)

    @pytest.mark.slow
    def test_invert_hef_water_level(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        entity['RGIId'] = 'RGI60-11.00897'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        v = inversion.mass_conservation_inversion(gdir, water_level=10000)

        cls = gdir.read_pickle('inversion_output')
        v_bwl = np.nansum([np.nansum(fl.get('volume_bwl', 0)) for fl in cls])
        n_trap = np.sum([np.sum(fl['is_trapezoid']) for fl in cls])
        np.testing.assert_allclose(v, v_bwl)
        assert n_trap > 10

    @pytest.mark.slow
    def test_invert_hef_from_linear_mb(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
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
        inversion.prepare_for_inversion(gdir)

        # Check how many clips:
        cls = gdir.read_pickle('inversion_input')
        nabove = 0
        maxs = 0.
        npoints = 0.
        for cl in cls:
            # Clip slope to avoid negative and small slopes
            slope = cl['slope_angle']
            nm = np.where(slope < np.deg2rad(2.))
            nabove += len(nm[0])
            npoints += len(slope)
            _max = np.max(slope)
            if _max > maxs:
                maxs = _max

        self.assertTrue(nabove == 0)
        self.assertTrue(np.rad2deg(maxs) < 40.)

        ref_v = 0.573 * 1e9

        glen_a = 2.4e-24
        fs = 5.7e-20

        def to_optimize(x):
            v = inversion.mass_conservation_inversion(gdir, fs=fs * x[1],
                                                         glen_a=glen_a * x[0])
            return (v - ref_v)**2

        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-4)['x']
        self.assertTrue(out[0] > 0.1)
        self.assertTrue(out[1] > 0.1)
        self.assertTrue(out[0] < 1.1)
        self.assertTrue(out[1] < 1.1)
        glen_a = glen_a * out[0]
        fs = fs * out[1]
        v = inversion.mass_conservation_inversion(gdir, fs=fs,
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

        maxs = 0.
        v = 0.
        cls = gdir.read_pickle('inversion_output')
        for cl in cls:
            thick = cl['thick']
            _max = np.max(thick)
            if _max > maxs:
                maxs = _max
            v += np.nansum(cl['volume'])
        np.testing.assert_allclose(242, maxs, atol=50)
        np.testing.assert_allclose(ref_v, v)

    def test_invert_hef_from_any_mb(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        # Reference
        climate.apparent_mb_from_linear_mb(gdir)
        inversion.prepare_for_inversion(gdir)
        cls1 = gdir.read_pickle('inversion_input')
        v1 = inversion.mass_conservation_inversion(gdir)
        # New should be equivalent
        mb_model = massbalance.LinearMassBalance(ela_h=1800, grad=3)
        climate.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
                                        mb_years=np.arange(30))
        inversion.prepare_for_inversion(gdir)
        v2 = inversion.mass_conservation_inversion(gdir)
        cls2 = gdir.read_pickle('inversion_input')

        # Now the tests
        for cl1, cl2 in zip(cls1, cls2):
            np.testing.assert_allclose(cl1['flux_a0'], cl2['flux_a0'])
        np.testing.assert_allclose(v1, v2)

    def test_distribute(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)

        # OK. Values from Fischer and Kuhn 2013
        # Area: 8.55
        # meanH = 67+-7
        # Volume = 0.573+-0.063
        # maxH = 242+-13
        inversion.prepare_for_inversion(gdir)

        ref_v = 0.573 * 1e9

        def to_optimize(x):
            glen_a = cfg.PARAMS['inversion_glen_a'] * x[0]
            fs = cfg.PARAMS['inversion_fs'] * x[1]
            v = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                         glen_a=glen_a)
            return (v - ref_v)**2
        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1, 1],
                                    bounds=((0.01, 10), (0.01, 10)),
                                    tol=1e-1)['x']
        glen_a = cfg.PARAMS['inversion_glen_a'] * out[0]
        fs = cfg.PARAMS['inversion_fs'] * out[1]
        v = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                     glen_a=glen_a,
                                                     write=True)
        np.testing.assert_allclose(ref_v, v)

        inversion.distribute_thickness_interp(gdir, varname_suffix='_interp')
        inversion.distribute_thickness_per_altitude(gdir,
                                                    varname_suffix='_alt')

        grids_file = gdir.get_filepath('gridded_data')
        with utils.ncDataset(grids_file) as nc:
            with warnings.catch_warnings():
                # https://github.com/Unidata/netcdf4-python/issues/766
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                t1 = nc.variables['distributed_thickness_interp'][:]
                t2 = nc.variables['distributed_thickness_alt'][:]

        np.testing.assert_allclose(np.nansum(t1), np.nansum(t2))

    @pytest.mark.slow
    def test_invert_hef_nofs(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)

        # OK. Values from Fischer and Kuhn 2013
        # Area: 8.55
        # meanH = 67+-7
        # Volume = 0.573+-0.063
        # maxH = 242+-13

        inversion.prepare_for_inversion(gdir)

        ref_v = 0.573 * 1e9

        def to_optimize(x):
            glen_a = cfg.PARAMS['inversion_glen_a'] * x[0]
            fs = 0.
            v = inversion.mass_conservation_inversion(gdir, fs=fs,
                                                         glen_a=glen_a)
            return (v - ref_v)**2

        import scipy.optimize as optimization
        out = optimization.minimize(to_optimize, [1],
                                    bounds=((0.00001, 100000),),
                                    tol=1e-4)['x']

        self.assertTrue(out[0] > 0.1)
        self.assertTrue(out[0] < 10)

        glen_a = cfg.PARAMS['inversion_glen_a'] * out[0]
        fs = 0.
        v = inversion.mass_conservation_inversion(gdir, fs=fs,
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

        # check that its not tooo sensitive to the dx
        cfg.PARAMS['flowline_dx'] = 1.
        cfg.PARAMS['filter_for_neg_flux'] = False
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf)
        t_star, bias = res['t_star'], res['bias']
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        v = inversion.mass_conservation_inversion(gdir, fs=fs,
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

        cfg.PARAMS['filter_for_neg_flux'] = True

        inversion.compute_velocities(gdir, fs=0, glen_a=glen_a)

        inv = gdir.read_pickle('inversion_output')[-1]

        # In the middle section the velocities look OK and should be close
        # to the no sliding assumption
        np.testing.assert_allclose(inv['u_surface'][20:60],
                                   inv['u_integrated'][20:60] / 0.8)

    def test_continue_on_error(self):

        cfg.PARAMS['continue_on_error'] = True
        cfg.PATHS['working_dir'] = self.testdir

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        miniglac = shpg.Point(entity.CenLon, entity.CenLat).buffer(0.0001)
        entity.geometry = miniglac
        entity.RGIId = 'RGI50-11.fake'

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        climate.local_t_star(gdir, tstar=1970, bias=0, prcp_fac=2.)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        inversion.mass_conservation_inversion(gdir)

        rdir = os.path.join(self.testdir, 'RGI50-11', 'RGI50-11.fa',
                            'RGI50-11.fake')
        self.assertTrue(os.path.exists(rdir))

        rdir = os.path.join(rdir, 'log.txt')
        self.assertTrue(os.path.exists(rdir))

        cfg.PARAMS['continue_on_error'] = False

        # Test the glacier charac
        dfc = utils.compile_glacier_statistics([gdir], path=False)
        self.assertEqual(dfc.terminus_type.values[0], 'Land-terminating')
        self.assertFalse('tstar_avg_temp_mean_elev' in dfc)


class TestCoxeCalving(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_calving')

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('dem_RGI50-01.10299.tif')
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PARAMS['border'] = 40

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    @pytest.mark.slow
    def test_inversion_with_calving(self):

        coxe_file = get_demo_file('rgi_RGI50-01.10299.shp')
        entity = gpd.read_file(coxe_file).iloc[0]

        cfg.PARAMS['use_kcalving_for_inversion'] = True
        cfg.PARAMS['use_kcalving_for_run'] = True

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        tasks.process_dummy_cru_file(gdir, seed=0)
        apply_test_ref_tstars()
        climate.local_t_star(gdir)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        inversion.mass_conservation_inversion(gdir)
        fls1 = gdir.read_pickle('inversion_flowlines')
        cls1 = gdir.read_pickle('inversion_output')
        # Increase calving for this one
        cfg.PARAMS['inversion_calving_k'] = 1
        out = inversion.find_inversion_calving(gdir)
        fls2 = gdir.read_pickle('inversion_flowlines')
        cls2 = gdir.read_pickle('inversion_output')

        # Calving increases the volume and reduces the mu
        v_ref = np.sum([np.sum(fl['volume']) for fl in cls1])
        v_new = np.sum([np.sum(fl['volume']) for fl in cls2])
        assert v_ref < v_new
        for fl1, fl2 in zip(fls1, fls2):
            assert round(fl2.mu_star, 5) <= round(fl1.mu_star, 5)

        # Redundancy test
        v_new_bsl = np.sum([np.sum(fl.get('volume_bsl', 0)) for fl in cls2])
        v_new_bwl = np.sum([np.sum(fl.get('volume_bwl', 0)) for fl in cls2])
        flowline.init_present_time_glacier(gdir)
        flsg = gdir.read_pickle('model_flowlines')
        for fl in flsg:
            fl.water_level = out['calving_water_level']
        v_new_bsl_g = np.sum([np.sum(fl.volume_bsl_m3) for fl in flsg])
        v_new_bwl_g = np.sum([np.sum(fl.volume_bwl_m3) for fl in flsg])
        assert v_new_bsl < v_new_bwl
        np.testing.assert_allclose(v_new_bsl, v_new_bsl_g)
        np.testing.assert_allclose(v_new_bwl, v_new_bwl_g)

    @pytest.mark.slow
    def test_inversion_and_run_with_calving(self):

        coxe_file = get_demo_file('rgi_RGI50-01.10299.shp')
        entity = gpd.read_file(coxe_file).iloc[0]

        cfg.PARAMS['use_kcalving_for_inversion'] = True
        cfg.PARAMS['use_kcalving_for_run'] = True
        cfg.PARAMS['inversion_calving_k'] = 1
        cfg.PARAMS['run_calving_k'] = 1

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        tasks.process_dummy_cru_file(gdir, seed=0)
        apply_test_ref_tstars()
        inversion.find_inversion_calving(gdir)

        # Test make a run
        flowline.init_present_time_glacier(gdir)
        flowline.run_constant_climate(gdir, bias=0, nyears=100)
        with xr.open_dataset(gdir.get_filepath('model_diagnostics')) as ds:
            assert ds.calving_m3[-1] > 10
            assert ds.volume_bwl_m3[-1] > 0
            assert ds.volume_bsl_m3[-1] < ds.volume_bwl_m3[-1]


class TestColumbiaCalving(unittest.TestCase):

    @pytest.mark.slow
    def test_find_calving_full_fl(self):

        gdir = init_columbia(reset=True)

        # For these tests we allow mu to 0
        cfg.PARAMS['calving_min_mu_star_frac'] = 0

        # Test default k (it overshoots)
        df = inversion.find_inversion_calving(gdir)

        assert df['calving_flux'] > 2
        assert df['calving_rate_myr'] > 500
        assert df['calving_mu_star'] == 0

        # Test that new MB equal flux
        mbmod = massbalance.MultipleFlowlineMassBalance
        mb = mbmod(gdir, use_inversion_flowlines=True,
                   mb_model_class=massbalance.ConstantMassBalance,
                   bias=0)

        rho = cfg.PARAMS['ice_density']
        flux_mb = (mb.get_specific_mb() * gdir.rgi_area_m2) * 1e-9 / rho
        np.testing.assert_allclose(flux_mb, df['calving_flux'],
                                   atol=0.001)

        # Test that accumulation equal flux (for Bea)
        # We use a simple MB model
        mbmod = massbalance.ConstantMassBalance(gdir)
        heights, widths = gdir.get_inversion_flowline_hw()  # width is in m
        temp, tempformelt, prcp, prcpsol = mbmod.get_annual_climate(heights)
        # prcpsol is in units mm w.e per year - let's convert
        # compute the area of each section
        fls = gdir.read_pickle('inversion_flowlines')
        area_sec = widths * fls[0].dx * gdir.grid.dx
        # Sum integral over the glacier
        prcpsol = np.sum(prcpsol * area_sec)
        # Convert to ice and km3
        accu_ice = prcpsol * 1e-9 / rho
        # Finally, chech that this is equal to our calving flux
        # units: mk3 ice yr-1
        np.testing.assert_allclose(accu_ice, df['calving_flux'],
                                   atol=0.001)

        # Test with smaller k (it doesn't overshoot)
        cfg.PARAMS['inversion_calving_k'] = 0.2
        df = inversion.find_inversion_calving(gdir)

        assert df['calving_flux'] > 0.2
        assert df['calving_flux'] < 1
        assert df['calving_rate_myr'] < 200
        assert df['calving_mu_star'] > 0
        np.testing.assert_allclose(df['calving_flux'], df['calving_law_flux'])

        # Test with fixed water depth and high k
        water_depth = 275.282
        cfg.PARAMS['inversion_calving_k'] = 2.4

        # Test with fixed water depth (it still overshoot)
        df = inversion.find_inversion_calving(gdir,
                                              fixed_water_depth=water_depth)

        assert df['calving_flux'] > 1
        assert df['calving_mu_star'] == 0
        assert df['calving_front_water_depth'] == water_depth
        assert df['calving_front_width'] > 100  # just to check its here

        # Test with smaller k (it doesn't overshoot)
        cfg.PARAMS['inversion_calving_k'] = 0.2
        df = inversion.find_inversion_calving(gdir,
                                              fixed_water_depth=water_depth)

        assert df['calving_flux'] > 0.1
        assert df['calving_flux'] < 1
        assert df['calving_mu_star'] > 0
        assert df['calving_front_water_depth'] == water_depth
        np.testing.assert_allclose(df['calving_flux'], df['calving_law_flux'])

        # Test glacier stats
        odf = utils.compile_glacier_statistics([gdir],
                                               inversion_only=True).iloc[0]
        np.testing.assert_allclose(odf.calving_flux, df['calving_flux'])
        np.testing.assert_allclose(odf.calving_front_water_depth, water_depth)

        # Check stats
        df = utils.compile_glacier_statistics([gdir])
        assert df.loc[gdir.rgi_id, 'error_task'] is None

    def test_find_calving_eb(self):

        gdir = init_columbia_eb('test_find_calving_eb')

        # Test default k (it overshoots)
        df = inversion.find_inversion_calving(gdir)

        mu_bef = gdir.get_diagnostics()['mu_star_before_calving']
        frac = cfg.PARAMS['calving_min_mu_star_frac']
        assert df['calving_mu_star'] == mu_bef * frac
        assert df['calving_flux'] > 0.5
        assert df['calving_rate_myr'] > 400

        # Test that new MB equal flux
        mbmod = massbalance.MultipleFlowlineMassBalance
        mb = mbmod(gdir, use_inversion_flowlines=True,
                   mb_model_class=massbalance.ConstantMassBalance,
                   bias=0)

        rho = cfg.PARAMS['ice_density']
        flux_mb = (mb.get_specific_mb() * gdir.rgi_area_m2) * 1e-9 / rho
        np.testing.assert_allclose(flux_mb, df['calving_flux'],
                                   atol=0.001)

        # Test glacier stats
        odf = utils.compile_glacier_statistics([gdir]).iloc[0]
        np.testing.assert_allclose(odf.calving_flux, df['calving_flux'])
        assert odf.calving_front_water_depth > 500

        # Test with smaller k (no overshoot)
        cfg.PARAMS['inversion_calving_k'] = 0.5
        df = inversion.find_inversion_calving(gdir)

        assert df['calving_flux'] > 0.5
        assert df['calving_mu_star'] > mu_bef * frac
        np.testing.assert_allclose(df['calving_flux'], df['calving_law_flux'])

        # Check stats
        df = utils.compile_glacier_statistics([gdir])
        assert df.loc[gdir.rgi_id, 'error_task'] is None

    @pytest.mark.slow
    def test_find_calving_workflow(self):

        gdir = init_columbia_eb('test_find_calving_workflow')

        # Check that all this also works with
        cfg.PARAMS['continue_on_error'] = True

        # Just a standard run
        workflow.calibrate_inversion_from_consensus([gdir])
        diag = gdir.get_diagnostics()
        assert diag['calving_law_flux'] > 0
        assert diag['calving_mu_star'] < diag['mu_star_before_calving']
        np.testing.assert_allclose(diag['calving_flux'], diag['calving_law_flux'])

        # Where we also match MB - method 1
        workflow.match_regional_geodetic_mb(gdir, rgi_reg='01')

        # Check OGGM part
        df = utils.compile_fixed_geometry_mass_balance([gdir])
        mb = df.loc[2000:2019].mean()
        rho = cfg.PARAMS['ice_density']
        cal = diag['calving_flux'] * 1e9 * rho / gdir.rgi_area_m2

        # Ref part
        df = 'table_hugonnet_regions_10yr_20yr_ar6period.csv'
        df = pd.read_csv(utils.get_demo_file(df))
        df = df.loc[df.period == '2000-01-01_2020-01-01'].set_index('reg')
        smb_ref = df.loc[int('01'), 'dmdtda']
        np.testing.assert_allclose(mb - cal, smb_ref)

        # Test match geod - method 2
        workflow.match_geodetic_mb_for_selection([gdir])

        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/'
        file_name = 'hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide_filled.hdf'
        df = pd.read_hdf(utils.file_downloader(base_url + file_name))
        df = df.loc[df['period'] == '2000-01-01_2020-01-01']
        rdf = df.loc[[gdir.rgi_id]]

        # Check OGGM part
        df = utils.compile_fixed_geometry_mass_balance([gdir])
        mb = df.loc[2000:2019].mean()
        rho = cfg.PARAMS['ice_density']
        cal = diag['calving_flux'] * 1e9 * rho / gdir.rgi_area_m2

        np.testing.assert_allclose(mb - cal, rdf.dmdtda * 1000)

        # OK - run
        tasks.init_present_time_glacier(gdir)
        tasks.run_from_climate_data(gdir, min_ys=1980, ye=2019,
                                    output_filesuffix='_hist')

        past_run_file = os.path.join(cfg.PATHS['working_dir'], 'compiled.nc')
        mb_file = os.path.join(cfg.PATHS['working_dir'], 'fixed_mb.csv')
        stats_file = os.path.join(cfg.PATHS['working_dir'], 'stats.csv')
        out_path = os.path.join(cfg.PATHS['working_dir'], 'extended.nc')

        # Check stats
        df = utils.compile_glacier_statistics([gdir], path=stats_file)
        assert df.loc[gdir.rgi_id, 'error_task'] is None
        assert df.loc[gdir.rgi_id, 'is_tidewater']

        # Compile stuff
        utils.compile_fixed_geometry_mass_balance([gdir], path=mb_file)
        utils.compile_run_output([gdir], path=past_run_file,
                                 input_filesuffix='_hist')

        # Extend
        utils.extend_past_climate_run(past_run_file=past_run_file,
                                      fixed_geometry_mb_file=mb_file,
                                      glacier_statistics_file=stats_file,
                                      path=out_path)

        with xr.open_dataset(out_path) as ods, \
                xr.open_dataset(past_run_file) as ds:

            ref = ds.volume
            new = ods.volume
            for y in [2010, 2012, 2019]:
                assert new.sel(time=y).data == ref.sel(time=y).data

            new = ods.volume_fixed_geom
            np.testing.assert_allclose(new.sel(time=2019), ref.sel(time=2019),
                                       rtol=0.01)

            del ods['volume_fixed_geom']
            assert sorted(list(ds.data_vars)) == sorted(list(ods.data_vars))

            for vn in ['area', 'length', 'calving_rate']:
                ref = ds[vn]
                new = ods[vn]
                for y in [2010, 2012, 2019]:
                    if y == 2010 and vn == 'calving_rate':
                        assert ref.sel(time=y).data == 0
                        assert new.sel(time=y).data == new.sel(time=y-1).data
                    else:
                        assert new.sel(time=y).data == ref.sel(time=y).data
                assert new.sel(time=1950).data == new.sel(time=2000).data

            # just some common sense
            assert ods['calving_rate'].sel(time=1950).data < 1000
            assert ods['calving_rate'].sel(time=1950).data > 10

            # We pick symmetry around rgi date so show that somehow it works
            for vn in ['volume', 'calving', 'volume_bsl', 'volume_bwl']:
                rtol = 0.3
                if 'bsl' in vn or 'bwl' in vn:
                    rtol = 0.6
                np.testing.assert_allclose(ods[vn].sel(time=2010) -
                                           ods[vn].sel(time=2002),
                                           ods[vn].sel(time=2018) -
                                           ods[vn].sel(time=2010),
                                           rtol=rtol)

    def test_find_calving_any_mb(self):

        gdir = init_columbia_eb('test_find_calving_any_mb')

        # Test default k
        mb = massbalance.LinearMassBalance(ela_h=2000)
        df = inversion.find_inversion_calving_from_any_mb(gdir, mb_model=mb,
                                                          mb_years=[2000])

        diag = gdir.get_diagnostics()
        assert diag['calving_flux'] > 0.9
        assert df['calving_rate_myr'] > 500

        # Test that new MB equal flux
        rho = cfg.PARAMS['ice_density']
        fls = gdir.read_pickle('inversion_flowlines')
        mb_ref = mb.get_specific_mb(fls=fls)
        mb_shift = diag['apparent_mb_from_any_mb_residual']
        flux_mb = ((mb_ref + mb_shift) * gdir.rgi_area_m2) * 1e-9 / rho
        np.testing.assert_allclose(flux_mb, df['calving_flux'],
                                   atol=0.001)

        # Test glacier stats
        odf = utils.compile_glacier_statistics([gdir]).iloc[0]
        np.testing.assert_allclose(odf.calving_flux, df['calving_flux'])
        assert odf.calving_front_water_depth > 500

        # Test with larger k
        cfg.PARAMS['inversion_calving_k'] = 1
        df_ = inversion.find_inversion_calving_from_any_mb(gdir, mb_model=mb,
                                                           mb_years=[2000])

        assert df_['calving_flux'] > df['calving_flux']
        np.testing.assert_allclose(df_['calving_flux'],
                                   df_['calving_law_flux'])


@pytest.mark.slow
class TestGrindelInvert(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_grindel')
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PARAMS['use_multiple_flowlines'] = False
        cfg.PARAMS['use_tar_shapefiles'] = False
        # not crop
        cfg.PARAMS['max_thick_to_width_ratio'] = 10
        cfg.PARAMS['max_shape_param'] = 10
        cfg.PARAMS['section_smoothing'] = 0.
        cfg.PARAMS['prcp_scaling_factor'] = 1

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
        glen_a = cfg.PARAMS['inversion_glen_a'] * 1
        from oggm.core import flowline

        gdir = utils.GlacierDirectory(self.rgin, base_dir=self.testdir)

        fls = self._parabolic_bed()
        mbmod = massbalance.LinearMassBalance(2800.)
        model = flowline.FluxBasedModel(fls, mb_model=mbmod, glen_a=glen_a,
                                        inplace=True)
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
            rho = cfg.PARAMS['ice_density']
            mb = mbmod.get_annual_mb(hgt) * cfg.SEC_IN_YEAR * rho
            fl.flux = np.zeros(len(fl.surface_h))
            fl.set_apparent_mb(mb)
            flux = fl.flux * (map_dx**2) / cfg.SEC_IN_YEAR / rho
            pok = np.nonzero(widths > 10.)
            widths = widths[pok]
            hgt = hgt[pok]
            flux = flux[pok]
            flux_a0 = 1.5 * flux / widths
            angle = -np.gradient(hgt, dx)  # beware the minus sign
            # Clip flux to 0
            assert not np.any(flux < -0.1)
            # add to output
            cl_dic = dict(dx=dx, flux=flux, flux_a0=flux_a0, width=widths,
                          hgt=hgt, slope_angle=angle, is_last=True,
                          is_rectangular=np.zeros(len(flux), dtype=bool),
                          is_trapezoid=np.zeros(len(flux), dtype=bool),
                          invert_with_trapezoid=False,
                          )
            towrite.append(cl_dic)

        # Write out
        gdir.write_pickle(towrite, 'inversion_input')
        v = inversion.mass_conservation_inversion(gdir, glen_a=glen_a)
        np.testing.assert_allclose(v, model.volume_m3, rtol=0.01)

        cl = gdir.read_pickle('inversion_output')[0]
        rmsd = utils.rmsd(cl['thick'], model.fls[0].thick[:len(cl['thick'])])
        assert rmsd < 10.

    def test_invert_and_run(self):

        from oggm.core import flowline, massbalance

        glen_a = cfg.PARAMS['inversion_glen_a'] * 2

        gdir = utils.GlacierDirectory(self.rgin, base_dir=self.testdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        # Trick
        climate.local_t_star(gdir, tstar=1975, bias=0.)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        v = inversion.mass_conservation_inversion(gdir, glen_a=glen_a)
        flowline.init_present_time_glacier(gdir)
        mb_mod = massbalance.ConstantMassBalance(gdir)
        fls = gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                        inplace=True,
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
        gdfc = gdir.read_shapefile('flowline_catchments')
        self.assertEqual(len(fls), len(gdfc))
        # and at least as many intersects
        gdfc = gdir.read_shapefile('catchments_intersects')
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
        cfg.PATHS['climate_file'] = ''
        cfg.PARAMS['border'] = 10

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_process_cesm(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        tasks.process_cru_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1902)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2014)

        f = get_demo_file('cesm.TREFHT.160001-200512.selection.nc')
        cfg.PATHS['cesm_temp_file'] = f
        f = get_demo_file('cesm.PRECC.160001-200512.selection.nc')
        cfg.PATHS['cesm_precc_file'] = f
        f = get_demo_file('cesm.PRECL.160001-200512.selection.nc')
        cfg.PATHS['cesm_precl_file'] = f
        gcm_climate.process_cesm_data(gdir)

        fh = gdir.get_filepath('climate_historical')
        fcesm = gdir.get_filepath('gcm_data')
        with xr.open_dataset(fh) as cru, xr.open_dataset(fcesm) as cesm:

            # Let's do some basic checks
            scru = cru.sel(time=slice('1961', '1990'))
            scesm = cesm.load().isel(time=((cesm['time.year'] >= 1961) &
                                           (cesm['time.year'] <= 1990)))
            # Climate during the chosen period should be the same
            np.testing.assert_allclose(scru.temp.mean(),
                                       scesm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(scru.prcp.mean(),
                                       scesm.prcp.mean(),
                                       rtol=1e-3)

            # And also the annual cycle
            scru = scru.groupby('time.month').mean(dim='time')
            scesm = scesm.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(scru.temp, scesm.temp, rtol=1e-3)
            np.testing.assert_allclose(scru.prcp, scesm.prcp, rtol=1e-3)

            # How did the annua cycle change with time?
            scesm1 = cesm.isel(time=((cesm['time.year'] >= 1961) &
                                     (cesm['time.year'] <= 1990)))
            scesm2 = cesm.isel(time=((cesm['time.year'] >= 1661) &
                                     (cesm['time.year'] <= 1690)))
            scesm1 = scesm1.groupby('time.month').mean(dim='time')
            scesm2 = scesm2.groupby('time.month').mean(dim='time')
            # No more than one degree? (silly test)
            np.testing.assert_allclose(scesm1.temp, scesm2.temp, atol=1)
            # N more than 30%? (silly test)
            np.testing.assert_allclose(scesm1.prcp, scesm2.prcp, rtol=0.3)

    def test_process_cmip5(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        tasks.process_cru_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1902)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2014)

        f = get_demo_file('tas_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        cfg.PATHS['cmip5_temp_file'] = f
        f = get_demo_file('pr_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        cfg.PATHS['cmip5_precip_file'] = f
        gcm_climate.process_cmip_data(gdir, filesuffix='_CCSM4')

        fh = gdir.get_filepath('climate_historical')
        fcmip = gdir.get_filepath('gcm_data', filesuffix='_CCSM4')
        with xr.open_dataset(fh) as cru, xr.open_dataset(fcmip) as cmip:

            # Let's do some basic checks
            scru = cru.sel(time=slice('1961', '1990'))
            scesm = cmip.load().isel(time=((cmip['time.year'] >= 1961) &
                                           (cmip['time.year'] <= 1990)))
            # Climate during the chosen period should be the same
            np.testing.assert_allclose(scru.temp.mean(),
                                       scesm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(scru.prcp.mean(),
                                       scesm.prcp.mean(),
                                       rtol=1e-3)

            # Here also std dev! But its not perfect because std_dev
            # is preserved over 31 years
            _scru = scru.groupby('time.month').std(dim='time')
            _scesm = scesm.groupby('time.month').std(dim='time')
            assert np.allclose(_scru.temp, _scesm.temp, rtol=1e-2)

            # And also the annual cycle
            scru = scru.groupby('time.month').mean(dim='time')
            scesm = scesm.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(scru.temp, scesm.temp, rtol=1e-3)
            np.testing.assert_allclose(scru.prcp, scesm.prcp, rtol=1e-3)

            # How did the annual cycle change with time?
            scmip1 = cmip.isel(time=((cmip['time.year'] >= 1961) &
                                     (cmip['time.year'] <= 1990)))
            scmip2 = cmip.isel(time=((cmip['time.year'] >= 2061) &
                                     (cmip['time.year'] <= 2090)))
            scmip1 = scmip1.groupby('time.month').mean(dim='time')
            scmip2 = scmip2.groupby('time.month').mean(dim='time')
            # It has warmed
            assert scmip1.temp.mean() < (scmip2.temp.mean() - 1)
            # N more than 30%? (silly test)
            np.testing.assert_allclose(scmip1.prcp, scmip2.prcp, rtol=0.3)

    def test_process_lmr(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        tasks.process_cru_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1902)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2014)

        fpath_temp = get_demo_file('air_MCruns_ensemble_mean_LMRv2.1.nc')
        fpath_precip = get_demo_file('prate_MCruns_ensemble_mean_LMRv2.1.nc')
        gcm_climate.process_lmr_data(gdir, fpath_temp=fpath_temp,
                                     fpath_precip=fpath_precip)

        fh = gdir.get_filepath('climate_historical')
        fcmip = gdir.get_filepath('gcm_data')
        with xr.open_dataset(fh) as cru, xr.open_dataset(fcmip) as cmip:

            # Let's do some basic checks
            scru = cru.sel(time=slice('1951', '1980'))
            scesm = cmip.sel(time=slice('1951', '1980'))
            # Climate during the chosen period should be the same
            np.testing.assert_allclose(scru.temp.mean(),
                                       scesm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(scru.prcp.mean(),
                                       scesm.prcp.mean(),
                                       rtol=1e-3)

            # Here also std dev! But its not perfect because std_dev
            # is preserved over 31 years
            _scru = scru.groupby('time.month').std(dim='time')
            _scesm = scesm.groupby('time.month').std(dim='time')
            np.testing.assert_allclose(_scru.temp, _scesm.temp, rtol=0.15)

            # And also the annual cycle
            scru = scru.groupby('time.month').mean(dim='time')
            scesm = scesm.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(scru.temp, scesm.temp, rtol=1e-3)
            np.testing.assert_allclose(scru.prcp, scesm.prcp, rtol=1e-3)

            # How did the annual cycle change with time?
            scmip1 = cmip.sel(time=slice('1970', '1999'))
            scmip2 = cmip.sel(time=slice('1800', '1829'))
            scmip1 = scmip1.groupby('time.month').mean(dim='time')
            scmip2 = scmip2.groupby('time.month').mean(dim='time')
            # It has warmed
            assert scmip2.temp.mean() < scmip1.temp.mean()

    def test_process_cmip5_scale(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        tasks.process_cru_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_hydro_yr_0'], 1902)
        self.assertEqual(ci['baseline_hydro_yr_1'], 2014)

        f = get_demo_file('tas_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        cfg.PATHS['cmip5_temp_file'] = f
        f = get_demo_file('pr_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        cfg.PATHS['cmip5_precip_file'] = f
        gcm_climate.process_cmip_data(gdir, filesuffix='_CCSM4_ns',
                                      scale_stddev=False)
        gcm_climate.process_cmip_data(gdir, filesuffix='_CCSM4')

        fh = gdir.get_filepath('climate_historical')
        fcmip = gdir.get_filepath('gcm_data', filesuffix='_CCSM4')
        with xr.open_dataset(fh) as cru, xr.open_dataset(fcmip) as cmip:

            # Let's do some basic checks
            scru = cru.sel(time=slice('1961', '1990'))
            scesm = cmip.load().isel(time=((cmip['time.year'] >= 1961) &
                                           (cmip['time.year'] <= 1990)))
            # Climate during the chosen period should be the same
            np.testing.assert_allclose(scru.temp.mean(),
                                       scesm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(scru.prcp.mean(),
                                       scesm.prcp.mean(),
                                       rtol=1e-3)

            # And also the annual cycle
            _scru = scru.groupby('time.month').mean(dim='time')
            _scesm = scesm.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(_scru.temp, _scesm.temp, rtol=1e-3)
            np.testing.assert_allclose(_scru.prcp, _scesm.prcp, rtol=1e-3)

            # Here also std dev!
            _scru = scru.groupby('time.month').std(dim='time')
            _scesm = scesm.groupby('time.month').std(dim='time')
            np.testing.assert_allclose(_scru.temp, _scesm.temp, rtol=1e-2)

            # How did the annual cycle change with time?
            scmip1 = cmip.isel(time=((cmip['time.year'] >= 1961) &
                                     (cmip['time.year'] <= 1990)))
            scmip2 = cmip.isel(time=((cmip['time.year'] >= 2061) &
                                     (cmip['time.year'] <= 2090)))
            scmip1 = scmip1.groupby('time.month').mean(dim='time')
            scmip2 = scmip2.groupby('time.month').mean(dim='time')
            # It has warmed
            assert scmip1.temp.mean() < (scmip2.temp.mean() - 1)
            # N more than 30%? (silly test)
            np.testing.assert_allclose(scmip1.prcp, scmip2.prcp, rtol=0.3)

        # Check that the two variabilies still correlate a lot
        f1 = gdir.get_filepath('gcm_data', filesuffix='_CCSM4_ns')
        f2 = gdir.get_filepath('gcm_data', filesuffix='_CCSM4')
        with xr.open_dataset(f1) as ds1, xr.open_dataset(f2) as ds2:
            n = 30*12+1
            ss1 = ds1.temp.rolling(time=n, min_periods=1, center=True).std()
            ss2 = ds2.temp.rolling(time=n, min_periods=1, center=True).std()
            assert utils.corrcoef(ss1, ss2) > 0.9

    def test_compile_climate_input(self):

        filename = 'gcm_data'
        filesuffix = '_cesm'

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)

        tasks.process_cru_data(gdir)
        utils.compile_climate_input([gdir])

        f = get_demo_file('cesm.TREFHT.160001-200512.selection.nc')
        cfg.PATHS['cesm_temp_file'] = f
        f = get_demo_file('cesm.PRECC.160001-200512.selection.nc')
        cfg.PATHS['cesm_precc_file'] = f
        f = get_demo_file('cesm.PRECL.160001-200512.selection.nc')
        cfg.PATHS['cesm_precl_file'] = f
        gcm_climate.process_cesm_data(gdir, filesuffix=filesuffix)
        utils.compile_climate_input([gdir], filename=filename,
                                    input_filesuffix=filesuffix)

        # CRU
        f1 = os.path.join(cfg.PATHS['working_dir'], 'climate_input.nc')
        f2 = gdir.get_filepath(filename='climate_historical')
        with xr.open_dataset(f1) as clim_cru1, \
                xr.open_dataset(f2) as clim_cru2:
            np.testing.assert_allclose(np.squeeze(clim_cru1.prcp),
                                       clim_cru2.prcp)
            np.testing.assert_allclose(np.squeeze(clim_cru1.temp),
                                       clim_cru2.temp)
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
        f1 = os.path.join(cfg.PATHS['working_dir'],
                          'climate_input_cesm.nc')
        f2 = gdir.get_filepath(filename=filename, filesuffix=filesuffix)
        with xr.open_dataset(f1) as clim_cesm1, \
                xr.open_dataset(f2) as clim_cesm2:
            np.testing.assert_allclose(np.squeeze(clim_cesm1.prcp),
                                       clim_cesm2.prcp)
            np.testing.assert_allclose(np.squeeze(clim_cesm1.temp),
                                       clim_cesm2.temp)
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
        cfg.PARAMS['use_tar_shapefiles'] = False
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
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.apparent_mb_from_linear_mb(gdir)
        inversion.prepare_for_inversion(gdir, invert_all_rectangular=True)
        v1 = inversion.mass_conservation_inversion(gdir)
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
        v2 = inversion.mass_conservation_inversion(gdir)

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

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        cfg.PARAMS['continue_on_error'] = True

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)

        # This will "run" but log an error
        from oggm.tasks import run_random_climate
        workflow.execute_entity_task(run_random_climate,
                                     [(gdir, {'filesuffix': '_testme'})])

        tfile = os.path.join(self.log_dir, 'RGI50-11.00897.ERROR')
        assert os.path.exists(tfile)
        with open(tfile, 'r') as f:
            first_line = f.readline()

        spl = first_line.split(';')
        assert len(spl) == 4
        assert spl[1].strip() == 'run_random_climate_testme'

    def test_task_status(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        cfg.PARAMS['continue_on_error'] = True

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)

        self.assertEqual(gdir.get_task_status(gis.glacier_masks.__name__),
                         'SUCCESS')
        assert gdir.get_task_time(gis.glacier_masks.__name__) > 0
        self.assertIsNone(gdir.get_task_status(
            centerlines.compute_centerlines.__name__))
        self.assertIsNone(gdir.get_task_time(
            centerlines.compute_centerlines.__name__))
        self.assertIsNone(gdir.get_error_log())

        centerlines.compute_downstream_bedshape(gdir)

        s = gdir.get_task_status(
            centerlines.compute_downstream_bedshape.__name__)
        assert 'FileNotFoundError' in s
        assert 'FileNotFoundError' in gdir.get_error_log()
        dft = utils.compile_task_time(
            [gdir], task_names=['compute_downstream_bedshape'])
        assert dft['compute_downstream_bedshape'].iloc[0] is None

        # Try overwrite
        cfg.PARAMS['auto_skip_task'] = True
        gis.glacier_masks(gdir)

        with open(gdir.logfile) as logfile:
            lines = logfile.readlines()
        isrun = ['glacier_masks' in l for l in lines]
        assert np.sum(isrun) == 1
        assert 'FileNotFoundError' in gdir.get_error_log()

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
        dft = utils.compile_task_time([gdir], task_names=tn)
        assert len(df) == 1
        assert len(df.columns) == 4
        assert len(dft.columns) == 2
        df = df.iloc[0]
        assert df['glacier_masks'] == 'SUCCESS'
        assert df['compute_centerlines'] == 'SUCCESS'
        assert df['compute_downstream_bedshape'] != 'SUCCESS'
        assert not np.isfinite(df['not_a_task'])
        assert dft['compute_centerlines'].iloc[0] > 0

        # Glacier stats
        df = utils.compile_glacier_statistics([gdir])
        assert 'error_task' in df.columns


class TestPyGEM_compat(unittest.TestCase):

    def setUp(self):
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PARAMS['use_intersects'] = False

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_read_gmip_data(self):
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        entity['RGIId'] = 'RGI60-11.00897'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)

        from oggm.sandbox import pygem_compat
        area_path = get_demo_file('gmip_area_centraleurope_10_sel.dat')
        thick_path = get_demo_file('gmip_thickness_centraleurope_10m_sel.dat')
        width_path = get_demo_file('gmip_width_centraleurope_10_sel.dat')
        data = pygem_compat.read_gmip_data(gdir,
                                           area_path=area_path,
                                           thick_path=thick_path,
                                           width_path=width_path)
        np.testing.assert_allclose(data['area'].sum(), gdir.rgi_area_m2,
                                   rtol=0.01)

    def test_flowlines_from_gmip_data(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        entity['RGIId'] = 'RGI60-11.00897'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)

        from oggm.sandbox import pygem_compat
        area_path = get_demo_file('gmip_area_centraleurope_10_sel.dat')
        thick_path = get_demo_file('gmip_thickness_centraleurope_10m_sel.dat')
        width_path = get_demo_file('gmip_width_centraleurope_10_sel.dat')
        data = pygem_compat.read_gmip_data(gdir,
                                           area_path=area_path,
                                           thick_path=thick_path,
                                           width_path=width_path)

        pygem_compat.present_time_glacier_from_bins(gdir, data=data)
        fls = gdir.read_pickle('model_flowlines')
        data = data.loc[::-1]
        area = np.asarray(data['area'])
        width = np.asarray(data['width'])
        thick = np.asarray(data['thick'])
        elevation = np.asarray(data.index).astype(float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            dx_meter = area / width
        dx_meter = np.where(np.isfinite(dx_meter), dx_meter, 0)

        np.testing.assert_allclose(fls[0].dx_meter, dx_meter)
        # Careful! The thickness changed
        np.testing.assert_allclose(fls[0].thick, 3/2 * thick)
        np.testing.assert_allclose(fls[0].widths_m, width)
        np.testing.assert_allclose(fls[0].surface_h, elevation)
        np.testing.assert_allclose(fls[0].section, width*thick)
        np.testing.assert_allclose(fls[0].area_m2, gdir.rgi_area_m2,
                                   rtol=0.01)
        np.testing.assert_allclose(fls[0].volume_m3,
                                   np.sum(width*thick*dx_meter),
                                   rtol=0.01)
