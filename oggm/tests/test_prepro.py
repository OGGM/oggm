import unittest
import os
import shutil
from packaging.version import Version
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
from oggm.tests.funcs import get_test_dir
from oggm import workflow
from oggm.exceptions import InvalidWorkflowError

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

    def test_define_region(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        extent = gdir.extent_ll

        tdf = gdir.read_shapefile('outlines').iloc[0]
        myarea = tdf.geometry.area * 10**-6
        np.testing.assert_allclose(myarea, float(tdf['Area']), rtol=1e-2)
        self.assertTrue(gdir.has_file('intersects'))
        np.testing.assert_array_equal(gdir.intersects_ids,
                                      ['RGI50-11.00846', 'RGI50-11.00950'])

        # From string
        gdir = oggm.GlacierDirectory(gdir.rgi_id, base_dir=self.testdir)
        # This is not guaranteed to be equal because of projection issues
        np.testing.assert_allclose(extent, gdir.extent_ll, atol=1e-5)
        warnings.filterwarnings("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # Warning in salem
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

        # Test binned method
        cfg.PARAMS['grid_dx_method'] = 'by_bin'
        gis.define_glacier_region(gdir)
        mygrid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
        np.testing.assert_allclose(np.abs(mygrid.dx), 50.)

        cfg.PARAMS['grid_dx_method'] = 'by_bin'
        cfg.PARAMS['by_bin_dx'] = [25, 75, 100, 200]
        gis.define_glacier_region(gdir)
        mygrid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
        np.testing.assert_allclose(np.abs(mygrid.dx), 75.)

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

    @pytest.mark.skipif((Version(rasterio.__version__) <
                         Version('1.0')),
                        reason='requires rasterio >= 1.0')
    def test_simple_glacier_masks(self):

        # The GIS was double checked externally with IDL.
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.simple_glacier_masks(gdir)

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

    @pytest.mark.skipif((Version(rasterio.__version__) <
                         Version('1.0')),
                        reason='requires rasterio >= 1.0')
    def test_compute_hypsometry_attributes(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.rasterio_glacier_mask(gdir)
        gis.rasterio_glacier_mask(gdir, no_nunataks=True)
        gis.rasterio_glacier_exterior_mask(gdir)
        gis.compute_hypsometry_attributes(gdir)

        dfh = pd.read_csv(gdir.get_filepath('hypsometry'))

        np.testing.assert_allclose(dfh['slope_deg'], entity.Slope, atol=0.5)
        np.testing.assert_allclose(dfh['aspect_deg'], entity.Aspect, atol=5)
        np.testing.assert_allclose(dfh['zmed_m'], entity.Zmed, atol=20)
        np.testing.assert_allclose(dfh['zmax_m'], entity.Zmax, atol=20)
        np.testing.assert_allclose(dfh['zmin_m'], entity.Zmin, atol=20)
        np.testing.assert_allclose(dfh['zmax_m'], entity.Zmax, atol=20)
        np.testing.assert_allclose(dfh['zmin_m'], entity.Zmin, atol=20)
        # From google map checks
        np.testing.assert_allclose(dfh['terminus_lon'], 10.80, atol=0.01)
        np.testing.assert_allclose(dfh['terminus_lat'], 46.81, atol=0.01)

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

    @pytest.mark.skipif((Version(rasterio.__version__) <
                         Version('1.0')),
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
        gis.simple_glacier_masks(gdir)
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            area = np.sum(nc.variables['glacier_mask'][:] * gdir.grid.dx**2)
            np.testing.assert_allclose(area*10**-6, gdir.rgi_area_km2,
                                       rtol=1e-1)
        shutil.copyfile(gdir.get_filepath('gridded_data'),
                        os.path.join(self.testdir, 'simple_masks.nc'))

    @pytest.mark.skipif((Version(rasterio.__version__) <
                         Version('1.0')),
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
        # origin is center for gdir grid but corner for dem_mask, so shift
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
                  'RAMP', 'GIMP', 'ARCTICDEM', 'DEM3', 'REMA', 'COPDEM30',
                  'COPDEM90', 'NASADEM', 'ALASKA']:
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

        # Independent reproduction for a few points
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
            interpolator = RegularGridInterpolator(xy, topo.astype(np.float64))

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
        cfg.PARAMS['use_winter_prcp_fac'] = False
        cfg.PARAMS['use_temp_bias_from_file'] = False
        cfg.PARAMS['prcp_fac'] = 2.5

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

        tdf = gdir.read_shapefile('outlines').iloc[0]
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
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))

        inversion.prepare_for_inversion(gdir)
        v1 = inversion.mass_conservation_inversion(gdir)
        inversion.distribute_thickness_per_altitude(gdir)
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            ds1 = ds.load()

        # Repeat multiple flowlines workflow
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir, reset=True)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))
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
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))
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

        tdf = gdir.read_shapefile('outlines').iloc[0]
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
        assert utils.rmsd(h1*100*50, h2*100*50) < 1

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
        self.testdir = os.path.join(get_test_dir(), 'tmp_prepro_climate')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.testdir_cru = os.path.join(get_test_dir(), 'tmp_prepro_climate_cru')
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
        # Ranges a bis smaller for simplicity
        cfg.PARAMS['temp_bias_min'] = -10
        cfg.PARAMS['temp_bias_max'] = 10
        cfg.PARAMS['prcp_fac_min'] = 0.1
        cfg.PARAMS['prcp_fac_max'] = 5
        cfg.PARAMS['use_winter_prcp_fac'] = False
        cfg.PARAMS['prcp_fac'] = 2.5

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
        assert ci['baseline_yr_0'] == 1802
        assert ci['baseline_yr_1'] == 2002

        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            # this file is in freakin hydro time
            ref_p = nc_r.variables['prcp'][3:-9, 1, 1]
            ref_t = nc_r.variables['temp'][3:-9, 1, 1]

        f = os.path.join(gdir.dir, 'climate_historical.nc')
        with utils.ncDataset(f) as nc_r:
            self.assertTrue(ref_h == nc_r.ref_hgt)
            np.testing.assert_allclose(ref_t, nc_r.variables['temp'][:])
            np.testing.assert_allclose(ref_p, nc_r.variables['prcp'][:])

    def test_distribute_climate_parallel(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        climate.process_custom_climate_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_yr_0'], 1802)
        self.assertEqual(ci['baseline_yr_1'], 2002)

        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            # this file is in freakin hydro time
            ref_p = nc_r.variables['prcp'][3:-9, 1, 1]
            ref_t = nc_r.variables['temp'][3:-9, 1, 1]

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
        self.assertEqual(ci['baseline_yr_0'], 1901)
        self.assertEqual(ci['baseline_yr_1'], 2014)

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
        self.assertEqual(ci['baseline_yr_0'], 1901)
        self.assertEqual(ci['baseline_yr_1'], 2014)

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
        tasks.process_histalp_data(gdirs[1], y0=1850, y1=2002)
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_yr_0'], 1850)
        self.assertEqual(ci['baseline_yr_1'], 2002)

        gdh = gdirs[0]
        gdc = gdirs[1]
        f1 = os.path.join(gdh.dir, 'climate_historical.nc')
        f2 = os.path.join(gdc.dir, 'climate_historical.nc')
        with xr.open_dataset(f1) as nc_h:
            with xr.open_dataset(f2) as nc_c:
                nc_h = nc_h.sel(time=nc_c.time)
                np.testing.assert_allclose(nc_h['temp'], nc_c['temp'])
                # for precip the data changed in between versions, we
                # can't test for absolute equality
                np.testing.assert_allclose(nc_h['prcp'].mean(),
                                           nc_c['prcp'].mean(),
                                           atol=1)
                np.testing.assert_allclose(nc_h.ref_pix_dis,
                                           nc_c.ref_pix_dis)

    def test_mb_calibration_from_scalar_mb(self):

        from oggm.core.massbalance import mb_calibration_from_scalar_mb
        from functools import partial
        mb_calibration_from_scalar_mb = partial(mb_calibration_from_scalar_mb,
                                                overwrite_gdir=True)

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.simple_glacier_masks(gdir)
        centerlines.elevation_band_flowline(gdir)
        centerlines.fixed_dx_elevation_band_flowline(gdir)
        climate.process_custom_climate_data(gdir)

        mbdf = gdir.get_ref_mb_data()
        mbdf['ref_mb'] = mbdf['ANNUAL_BALANCE']
        ref_mb = mbdf.ANNUAL_BALANCE.mean()
        ref_period = f'{mbdf.index[0]}-01-01_{mbdf.index[-1] + 1}-01-01'

        # Default is to calibrate melt_f
        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period)

        h, w = gdir.get_inversion_flowline_hw()
        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['melt_mb'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['melt_mb'].mean())
        # Yeah, it correlates but also not too crazy
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'melt_mb']].corr(),
                                   atol=0.35)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] == 0
        assert pdf['melt_f'] != cfg.PARAMS['melt_f']
        assert pdf['prcp_fac'] == cfg.PARAMS['prcp_fac']

        # Let's calibrate on temp_bias
        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period,
                                      calibrate_param1='temp_bias')

        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['temp_mb'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['temp_mb'].mean())
        # Yeah, it correlates but also not too crazy
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'temp_mb']].corr(),
                                   atol=0.35)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] != 0
        assert pdf['melt_f'] == cfg.PARAMS['melt_f']
        assert pdf['prcp_fac'] == cfg.PARAMS['prcp_fac']

        # Let's calibrate on precip
        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period,
                                      calibrate_param1='prcp_fac')

        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['prcp_mb'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['prcp_mb'].mean())
        # Yeah, it correlates but also not too crazy
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'prcp_mb']].corr(),
                                   atol=0.35)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] == 0
        assert pdf['melt_f'] == cfg.PARAMS['melt_f']
        assert pdf['prcp_fac'] != cfg.PARAMS['prcp_fac']

        # mbdf[['ref_mb', 'melt_mb', 'temp_mb', 'prcp_mb']].plot()
        # plt.show()

        # OK now check what happens with unrealistic climate input
        # Very positive
        ref_mb = 2000
        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period)
        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period,
                                      calibrate_param2='temp_bias')

        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['melt_mb2'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['melt_mb2'].mean())
        # It should correlate even less
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'melt_mb2']].corr(),
                                   atol=0.55)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] < 0
        assert pdf['melt_f'] != cfg.PARAMS['melt_f']
        assert pdf['melt_f'] == cfg.PARAMS['melt_f_min']
        assert pdf['prcp_fac'] == cfg.PARAMS['prcp_fac']

        # Very negative
        ref_mb = -10000
        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period)
        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period,
                                      calibrate_param2='temp_bias')

        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['melt_mb2'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['melt_mb2'].mean())
        # It should correlate even less (maybe not)
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'melt_mb2']].corr(),
                                   atol=0.5)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] > 0
        assert pdf['melt_f'] != cfg.PARAMS['melt_f']
        assert pdf['melt_f'] == cfg.PARAMS['melt_f_max']
        assert pdf['prcp_fac'] == cfg.PARAMS['prcp_fac']

        # Try with prcp_fac as variable 1
        # Very positive
        ref_mb = 3000
        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period,
                                          calibrate_param1='prcp_fac')
        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period,
                                      calibrate_param1='prcp_fac',
                                      calibrate_param2='temp_bias')

        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['melt_mb2'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['melt_mb2'].mean())
        # It should correlate even less (maybe not)
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'melt_mb2']].corr(),
                                   atol=0.45)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] < 0
        assert pdf['melt_f'] == cfg.PARAMS['melt_f']
        assert pdf['prcp_fac'] > cfg.PARAMS['prcp_fac']

        # Very negative
        ref_mb = -10000
        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period,
                                          calibrate_param1='prcp_fac')
        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period,
                                      calibrate_param1='prcp_fac',
                                      calibrate_param2='temp_bias')

        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['melt_mb2'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['melt_mb2'].mean())
        # It should correlate even less (maybe not)
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'melt_mb2']].corr(),
                                   atol=0.5)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] > 0
        assert pdf['melt_f'] == cfg.PARAMS['melt_f']
        assert pdf['prcp_fac'] < cfg.PARAMS['prcp_fac']

        # Extremely negative
        ref_mb = -20000
        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period,
                                          calibrate_param1='prcp_fac')

        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period,
                                          calibrate_param1='prcp_fac',
                                          calibrate_param2='temp_bias')

        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period,
                                      calibrate_param1='prcp_fac',
                                      calibrate_param2='temp_bias',
                                      calibrate_param3='melt_f')

        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['melt_mb3'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['melt_mb3'].mean())
        # It should correlate even less (maybe not)
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'melt_mb3']].corr(),
                                   atol=0.5)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] == cfg.PARAMS['temp_bias_max']
        assert pdf['melt_f'] > cfg.PARAMS['melt_f']
        assert pdf['prcp_fac'] == cfg.PARAMS['prcp_fac_min']

        # Unmatchable positive
        ref_mb = 10000
        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period,
                                          calibrate_param1='prcp_fac',
                                          calibrate_param2='temp_bias',
                                          calibrate_param3='melt_f')

        # Matchable positive with less range
        ref_mb = 1000
        cfg.PARAMS['temp_bias_min'] = -0.5
        cfg.PARAMS['temp_bias_max'] = 0.5
        cfg.PARAMS['prcp_fac_min'] = 2
        cfg.PARAMS['prcp_fac_max'] = 3
        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period,
                                          calibrate_param1='prcp_fac')

        with pytest.raises(RuntimeError):
            mb_calibration_from_scalar_mb(gdir,
                                          ref_mb=ref_mb,
                                          ref_period=ref_period,
                                          calibrate_param1='prcp_fac',
                                          calibrate_param2='temp_bias')

        mb_calibration_from_scalar_mb(gdir,
                                      ref_mb=ref_mb,
                                      ref_period=ref_period,
                                      calibrate_param1='prcp_fac',
                                      calibrate_param2='temp_bias',
                                      calibrate_param3='melt_f')

        mb_new = massbalance.MonthlyTIModel(gdir)
        mbdf['melt_mb3'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['melt_mb3'].mean())
        # It should correlate even less (maybe not)
        np.testing.assert_allclose(1, mbdf[['ref_mb', 'melt_mb3']].corr(),
                                   atol=0.5)

        pdf = gdir.read_json('mb_calib')
        assert pdf['temp_bias'] == cfg.PARAMS['temp_bias_min']
        assert pdf['melt_f'] < cfg.PARAMS['melt_f']
        assert pdf['prcp_fac'] == cfg.PARAMS['prcp_fac_max']

    @pytest.mark.slow
    def test_mb_calibration_from_scalar_mb_multiple_fl(self):

        from oggm.core.massbalance import mb_calibration_from_scalar_mb
        from functools import partial
        mb_calibration_from_scalar_mb = partial(mb_calibration_from_scalar_mb,
                                                overwrite_gdir=True)

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
        mbdf['ref_mb'] = mbdf['ANNUAL_BALANCE']
        ref_mb = mbdf.ref_mb.mean()
        ref_period = f'{mbdf.index[0]}-01-01_{mbdf.index[-1] + 1}-01-01'
        mb_calibration_from_scalar_mb(gdir, ref_mb=ref_mb, ref_period=ref_period)
        mb_new = massbalance.MonthlyTIModel(gdir)

        h, w = gdir.get_inversion_flowline_hw()
        mbdf['new_mb'] = mb_new.get_specific_mb(h, w, year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['new_mb'].mean())
        # not perfect
        np.testing.assert_allclose(1, mbdf[['new_mb', 'ref_mb']].corr(),
                                   atol=0.35)

        # Check that inversion works
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=[1953, 2002])

        # Artificially make some arms even lower to have multiple branches
        fls = gdir.read_pickle('inversion_flowlines')
        assert fls[0].flows_to is fls[-1]
        assert fls[1].flows_to is fls[-1]
        fls[0].surface_h -= 700
        fls[1].surface_h -= 700
        gdir.write_pickle(fls, 'inversion_flowlines')

        mb_calibration_from_scalar_mb(gdir, ref_mb=ref_mb,
                                        ref_period=ref_period)
        mb_new = massbalance.MultipleFlowlineMassBalance(gdir,
                                                         use_inversion_flowlines=True)

        mbdf['new_mb2'] = mb_new.get_specific_mb(year=mbdf.index)

        # Check that results are all the same
        np.testing.assert_allclose(ref_mb, mbdf['new_mb2'].mean())

        # Check that model parameters
        np.testing.assert_allclose(mb_new.bias, 0)

        # Check that inversion works but it logs a warning
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=[1953, 2003])

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
        cfg.PARAMS['use_winter_prcp_fac'] = False
        cfg.PARAMS['use_temp_bias_from_file'] = False
        cfg.PARAMS['prcp_fac'] = 2.5

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
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))

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
        np.testing.assert_allclose(np.mean(velocity[:-1]), 42, atol=5)
        inversion.compute_inversion_velocities(gdir, fs=fs, glen_a=glen_a)
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
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))
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
        np.testing.assert_allclose(df.vol_itmix_m3, df.vol_oggm_m3, rtol=0.08)

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
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))

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
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))
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
        massbalance.apparent_mb_from_linear_mb(gdir)

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
        massbalance.apparent_mb_from_linear_mb(gdir)
        inversion.prepare_for_inversion(gdir)
        cls1 = gdir.read_pickle('inversion_input')
        v1 = inversion.mass_conservation_inversion(gdir)
        # New should be equivalent
        mb_model = massbalance.LinearMassBalance(ela_h=1800, grad=3)
        massbalance.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
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
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))

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
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))

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

        # check that its not too sensitive to the dx
        cfg.PARAMS['flowline_dx'] = 1.
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        massbalance.mb_calibration_from_wgms_mb(gdir, overwrite_gdir=True)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))
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

        inversion.compute_inversion_velocities(gdir, fs=0, glen_a=glen_a)

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
        entity.RGIId = 'RGI50-11.faked'

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        massbalance.mb_calibration_from_wgms_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1953, 2002))

        rdir = os.path.join(self.testdir, 'RGI50-11', 'RGI50-11.fa',
                            'RGI50-11.faked')
        self.assertTrue(os.path.exists(rdir))

        rdir = os.path.join(rdir, 'log.txt')
        self.assertTrue(os.path.exists(rdir))

        cfg.PARAMS['continue_on_error'] = False

        # Test the glacier charac
        dfc = utils.compile_glacier_statistics([gdir], path=False)
        self.assertEqual(dfc.terminus_type.values[0], 'Land-terminating')


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
        cfg.PARAMS['use_winter_prcp_fac'] = False
        cfg.PARAMS['use_temp_bias_from_file'] = False
        cfg.PARAMS['prcp_fac'] = 2.5

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    @pytest.mark.slow
    def test_inversion_with_calving(self):

        coxe_file = get_demo_file('rgi_RGI50-01.10299.shp')
        entity = gpd.read_file(coxe_file).iloc[0]
        entity.RGIId = 'RGI60-01.10299'
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
        massbalance.mb_calibration_from_geodetic_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir)

        inversion.prepare_for_inversion(gdir)
        inversion.mass_conservation_inversion(gdir)
        cls1 = gdir.read_pickle('inversion_output')
        # Increase calving for this one
        cfg.PARAMS['inversion_calving_k'] = 1

        res_bef = gdir.get_diagnostics()['apparent_mb_from_any_mb_residual']
        out = inversion.find_inversion_calving_from_any_mb(gdir)
        cls2 = gdir.read_pickle('inversion_output')

        # Calving increases the volume and adds a residual
        v_ref = np.sum([np.sum(fl['volume']) for fl in cls1])
        v_new = np.sum([np.sum(fl['volume']) for fl in cls2])
        assert v_ref < v_new
        res_aft = gdir.get_diagnostics()['apparent_mb_from_any_mb_residual']
        assert res_aft > res_bef

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
        entity.RGIId = 'RGI60-01.10299'

        cfg.PARAMS['use_kcalving_for_inversion'] = True
        cfg.PARAMS['use_kcalving_for_run'] = True
        cfg.PARAMS['inversion_calving_k'] = 1
        cfg.PARAMS['calving_k'] = 1
        cfg.PARAMS['evolution_model'] = 'FluxBased'
        cfg.PARAMS['calving_k'] = 1

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
        massbalance.mb_calibration_from_geodetic_mb(gdir)
        massbalance.apparent_mb_from_any_mb(gdir)
        inversion.find_inversion_calving_from_any_mb(gdir)

        # Test make a run
        flowline.init_present_time_glacier(gdir)
        flowline.run_constant_climate(gdir, y0=1980, nyears=100,
                                      temperature_bias=-2)
        with xr.open_dataset(gdir.get_filepath('model_diagnostics')) as ds:
            assert ds.calving_m3[-1] > 10
            assert ds.volume_bwl_m3[-1] > 0
            assert ds.volume_bsl_m3[-1] < ds.volume_bwl_m3[-1]


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
        cfg.PARAMS['use_winter_prcp_fac'] = False
        cfg.PARAMS['use_temp_bias_from_file'] = False
        cfg.PARAMS['prcp_fac'] = 2.5

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
            fl.set_apparent_mb(mb, is_calving=False)
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
        self.assertGreaterEqual(np.sum(fls[-1].is_rectangular), 10)


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
        cfg.PARAMS['use_winter_prcp_fac'] = False
        cfg.PARAMS['use_temp_bias_from_file'] = False
        cfg.PARAMS['prcp_fac'] = 2.5
        cfg.PARAMS['baseline_climate'] = 'CRU'

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_process_monthly_isimip_data(self):
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        ssp = 'ssp126'
        member = 'mri-esm2-0_r1i1p1f1'

        tasks.process_w5e5_data(gdir)
        # testing = True to only download the HEF gridpoint and 3 other gridppoints around
        tasks.process_monthly_isimip_data(gdir, member=member, ssp=ssp,
                                          output_filesuffix='_no_OGGM_bc',
                                          apply_bias_correction=False,
                                          testing=True)
        tasks.process_monthly_isimip_data(gdir, member=member, ssp=ssp,
                                          output_filesuffix='_with_OGGM_bc_y0_y1',
                                          apply_bias_correction=True,
                                          testing=True, y0=1979-18, y1=2095)
        tasks.process_monthly_isimip_data(gdir, member=member, ssp=ssp,
                                          apply_bias_correction=True,
                                          output_filesuffix='_with_OGGM_bc',
                                          testing=True)
        fh = gdir.get_filepath('climate_historical')
        # with additional bias correction of OGGM
        fgcm = gdir.get_filepath('gcm_data',
                                 filesuffix='_with_OGGM_bc')
        # without any bias correction of OGGM (as ISIMIP3b is already
        # externally bias-corrected to W5E5)
        fgcm_nc = gdir.get_filepath('gcm_data',
                                    filesuffix='_no_OGGM_bc')
        # with additional bias correction and with selected time series
        fgcm_y0_y1 = gdir.get_filepath('gcm_data',
                                       filesuffix='_with_OGGM_bc_y0_y1')

        with xr.open_dataset(fh) as clim, xr.open_dataset(fgcm) as gcm, \
                xr.open_dataset(fgcm_nc) as gcm_nc, \
                xr.open_dataset(fgcm_y0_y1) as gcm_y0_y1:
            # Let's do some basic checks
            sclim = clim.sel(time=slice('1979', '2014'))
            sgcm = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                         (gcm['time.year'] <= 2014)))

            sgcm_nc = gcm_nc.load().isel(time=((gcm_nc['time.year'] >= 1979) &
                                               (gcm_nc['time.year'] <= 2014)))

            # # just check if the right time period is chosen and that it is equal to
            # not choosing a time period over common period
            sgcm_y0_y1 = gcm_y0_y1.load()
            assert sgcm_y0_y1['time.year'].min() == 1979-18
            assert sgcm_y0_y1['time.year'].max() == 2095
            sgcm_sel = gcm.load().sel(time=slice('1961', '2095'))
            assert np.all(sgcm_sel.time.values == sgcm_y0_y1.time.values)
            assert np.all(sgcm_sel.ref_hgt == sgcm_y0_y1.ref_hgt)
            assert np.all(sgcm_sel.ref_pix_lon == sgcm_y0_y1.ref_pix_lon)
            assert np.all(sgcm_sel.ref_pix_lat == sgcm_y0_y1.ref_pix_lat)
            np.testing.assert_allclose(sgcm_sel.prcp, sgcm_y0_y1.prcp)
            # because of standard deviation rolling function, the temperature
            # timeseries are slightly different at the starts/ends of the timeseries
            # -> only similar for +/- half of rolling window years
            # (i.e., here +/- 36/2 years)
            np.testing.assert_allclose(sgcm_sel.temp.sel(time=slice('1979', '2077')),
                                       sgcm_y0_y1.temp.sel(time=slice('1979', '2077')))
            # the entire timeseries should still be similar
            np.testing.assert_allclose(sgcm_sel.temp, sgcm_y0_y1.temp, atol=0.2)

            # first check if the same grid point was chosen and the same ref_hgt:
            np.testing.assert_allclose(sgcm.ref_hgt, sgcm_nc.ref_hgt)
            np.testing.assert_allclose(sgcm_nc.ref_hgt, sclim.ref_hgt)

            np.testing.assert_allclose(sgcm.ref_pix_lon, sgcm_nc.ref_pix_lon)
            np.testing.assert_allclose(sgcm_nc.ref_pix_lon, sclim.ref_pix_lon)

            np.testing.assert_allclose(sgcm.ref_pix_lat, sgcm_nc.ref_pix_lat)
            np.testing.assert_allclose(sgcm_nc.ref_pix_lat, sclim.ref_pix_lat)

            # Climate during the chosen period should be the same when corrected
            np.testing.assert_allclose(sclim.temp.mean(),
                                       sgcm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(sclim.prcp.mean(),
                                       sgcm.prcp.mean(),
                                       rtol=1e-3)
            # even if not corrected the climate should be quite similar because
            # ISIMIP3b was internally bias corrected to match W5E5
            np.testing.assert_allclose(sclim.temp.mean(),
                                       sgcm_nc.temp.mean(),
                                       rtol=2e-2)
            np.testing.assert_allclose(sclim.prcp.mean(),
                                       sgcm_nc.prcp.mean(),
                                       rtol=2e-2)

            # annual lapse rate cycle is applied anyway
            _sclim = sclim.groupby('time.month').std(dim='time')
            _sgcm = sgcm.groupby('time.month').std(dim='time')
            _sgcm_nc = sgcm_nc.groupby('time.month').std(dim='time')
            # need higher tolerance here:
            np.testing.assert_allclose(_sclim.temp, _sgcm.temp, rtol=0.08)  # 1e-3
            # even higher for non-OGGM bias ccrrection
            np.testing.assert_allclose(_sclim.temp, _sgcm_nc.temp, rtol=0.3)  # 1e-3
            # not done for precipitation!

            # And also the annual cycle
            sclim = sclim.groupby('time.month').mean(dim='time')
            sgcm = sgcm.groupby('time.month').mean(dim='time')
            sgcm_nc = sgcm_nc.groupby('time.month').mean(dim='time')

            np.testing.assert_allclose(sclim.temp, sgcm.temp, rtol=1e-3)
            np.testing.assert_allclose(sclim.prcp, sgcm.prcp, rtol=1e-3)

            # same for non corrected stuff
            np.testing.assert_allclose(sclim.temp, sgcm_nc.temp, rtol=8e-2)
            np.testing.assert_allclose(sclim.prcp, sgcm_nc.prcp, rtol=2e-1)

            # How did the annual cycle change with time?
            sgcm1 = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                          (gcm['time.year'] <= 2019)))
            sgcm2 = gcm.isel(time=((gcm['time.year'] >= 2060) &
                                   (gcm['time.year'] <= 2100)))
            sgcm1_nc = gcm_nc.load().isel(time=((gcm_nc['time.year'] >= 1979) &
                                                (gcm_nc['time.year'] <= 2019)))
            sgcm2_nc = gcm_nc.isel(time=((gcm_nc['time.year'] >= 2060) &
                                         (gcm_nc['time.year'] <= 2100)))

            _sgcm1_std = sgcm1.groupby('time.month').mean(dim='time').std()
            _sgcm2_std = sgcm2.groupby('time.month').mean(dim='time').std()

            _sgcm1_nc_std = sgcm1_nc.groupby('time.month').mean(dim='time').std()
            _sgcm2_nc_std = sgcm2_nc.groupby('time.month').mean(dim='time').std()
            # the mean standard deviation over the year between the months
            # should be different for the time periods
            assert not np.allclose(_sgcm1_std.temp, _sgcm2_std.temp, rtol=1e-2)
            assert not np.allclose(_sgcm1_nc_std.temp, _sgcm2_nc_std.temp, rtol=1e-2)
            # but should be similar between corrected and not corrected
            np.testing.assert_allclose(_sgcm1_std.temp, _sgcm1_nc_std.temp, rtol=1e-2)
            np.testing.assert_allclose(_sgcm2_std.temp, _sgcm2_nc_std.temp, rtol=1e-2)

            sgcm1 = sgcm1.groupby('time.month').mean(dim='time')
            sgcm2 = sgcm2.groupby('time.month').mean(dim='time')
            sgcm1_nc = sgcm1_nc.groupby('time.month').mean(dim='time')
            sgcm2_nc = sgcm2_nc.groupby('time.month').mean(dim='time')
            # It has warmed at least 1 degree for each scenario???
            assert sgcm1.temp.mean() < (sgcm2.temp.mean() - 1)
            assert sgcm1_nc.temp.mean() < (sgcm2_nc.temp.mean() - 1)
            # No prcp change more than 30%? (silly test)
            np.testing.assert_allclose(sgcm1.prcp, sgcm2.prcp, rtol=0.3)
            np.testing.assert_allclose(sgcm1_nc.prcp, sgcm2_nc.prcp, rtol=0.3)

            # mean temperature similar between OGGM and ISIMIP corrected
            np.testing.assert_allclose(sgcm1.temp.mean(),
                                       sgcm1_nc.temp.mean(), rtol=0.05)
            np.testing.assert_allclose(sgcm2.temp.mean(),
                                       sgcm2_nc.temp.mean(), rtol=0.05)

            # mean prcp similar between OGGM and ISIMIP corrected
            np.testing.assert_allclose(sgcm1.prcp.mean(),
                                       sgcm1_nc.prcp.mean(), rtol=0.05)
            np.testing.assert_allclose(sgcm2.prcp.mean(),
                                       sgcm2_nc.prcp.mean(), rtol=0.05)

            # Check that temp still correlate a lot between non corrected
            # and corrected gcm:
            n = 36 * 12 + 1
            ss1 = gcm.temp.rolling(time=n, min_periods=1, center=True).std()
            ss2 = gcm_nc.temp.rolling(time=n, min_periods=1, center=True).std()
            assert utils.corrcoef(ss1, ss2) > 0.99

    def test_process_cesm(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        tasks.process_cru_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_yr_0'], 1901)
        self.assertEqual(ci['baseline_yr_1'], 2014)

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
        self.assertEqual(ci['baseline_yr_0'], 1901)
        self.assertEqual(ci['baseline_yr_1'], 2014)

        fpath_temp = get_demo_file('tas_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        fpath_precip = get_demo_file('pr_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        gcm_climate.process_cmip_data(gdir,
                                      fpath_temp=fpath_temp,
                                      fpath_precip=fpath_precip,
                                      filesuffix='_CCSM4')
        gcm_climate.process_cmip_data(gdir,
                                      fpath_temp=fpath_temp,
                                      fpath_precip=fpath_precip,
                                      filesuffix='_CCSM4_y0_y1', y0=1960-15,
                                      y1=2080+15)

        fh = gdir.get_filepath('climate_historical')
        fcmip = gdir.get_filepath('gcm_data', filesuffix='_CCSM4')
        fcmip_y0_y1 = gdir.get_filepath('gcm_data', filesuffix='_CCSM4_y0_y1')
        with xr.open_dataset(fh) as cru, \
              xr.open_dataset(fcmip) as cmip, \
              xr.open_dataset(fcmip_y0_y1) as cmip_y0_y1:

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
            # just check if the right time period is chosen, and
            # that it is equal over common period with not choosing
            # y0 and y1
            scesm_y0_y1 = cmip_y0_y1.load()
            assert scesm_y0_y1['time.year'].min() == 1960-15
            assert scesm_y0_y1['time.year'].max() == 2080+15
            scmip_sel = cmip.load().sel(time=slice('1945', '2095'))
            assert np.all(scmip_sel.time.values == scesm_y0_y1.time.values)
            assert np.all(scmip_sel.ref_hgt == scesm_y0_y1.ref_hgt)
            assert np.all(scmip_sel.ref_pix_lon == scesm_y0_y1.ref_pix_lon)
            assert np.all(scmip_sel.ref_pix_lat == scesm_y0_y1.ref_pix_lat)
            np.testing.assert_allclose(scmip_sel.prcp, scesm_y0_y1.prcp)
            # because of standard deviation rolling function, the temperature
            # timeseries are slightly different at the starts/ends of the timeseries
            # -> only similar for +/- half of rolling window years
            # (i.e., here +/- 30/2 years)
            np.testing.assert_allclose(scmip_sel.temp.sel(time=slice('1960', '2080')),
                                       scesm_y0_y1.temp.sel(time=slice('1960', '2080')))
            # the entire timeseries should still be similar
            np.testing.assert_allclose(scmip_sel.temp, scesm_y0_y1.temp, atol=0.2)

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

    @pytest.mark.slow
    def test_process_cmip_no_hydromonths(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        tasks.process_cru_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_yr_0'], 1901)
        self.assertEqual(ci['baseline_yr_1'], 2014)

        ft = get_demo_file('tas_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        fp = get_demo_file('pr_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        gcm_climate.process_cmip_data(gdir, fpath_temp=ft,
                                      fpath_precip=fp,
                                      filesuffix='_CCSM4')

        fh = gdir.get_filepath('climate_historical')
        fcmip = gdir.get_filepath('gcm_data', filesuffix='_CCSM4')
        with xr.open_dataset(fh) as cru, xr.open_dataset(fcmip) as cmip:

            # Let's do some basic checks
            assert cru['time.year'][0] == 1901
            assert cru['time.year'][-1] == 2014
            assert cru['time.month'][0] == 1
            assert cru['time.month'][-1] == 12
            assert cmip['time.year'][0] == 1870
            assert cmip['time.year'][-1] == 2100
            assert cmip['time.month'][0] == 1
            assert cmip['time.month'][-1] == 12

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

    def test_process_lmr(self):

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        tasks.process_cru_data(gdir)

        ci = gdir.get_climate_info()
        self.assertEqual(ci['baseline_yr_0'], 1901)
        self.assertEqual(ci['baseline_yr_1'], 2014)

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
        self.assertEqual(ci['baseline_yr_0'], 1901)
        self.assertEqual(ci['baseline_yr_1'], 2014)

        fpath_temp = get_demo_file('tas_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        fpath_precip = get_demo_file('pr_mon_CCSM4_rcp26_r1i1p1_g025.nc')
        gcm_climate.process_cmip_data(gdir,
                                      fpath_temp=fpath_temp,
                                      fpath_precip=fpath_precip,
                                      filesuffix='_CCSM4_ns',
                                      scale_stddev=False)
        gcm_climate.process_cmip_data(gdir,
                                      fpath_temp=fpath_temp,
                                      fpath_precip=fpath_precip,
                                      filesuffix='_CCSM4')

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
                                       [4, 3])
            np.testing.assert_allclose(clim_cru1.hydro_year[[0, -1]],
                                       clim_cru1.calendar_year[[0, -1]]+[0, 1])

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
        massbalance.apparent_mb_from_linear_mb(gdir)
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
        massbalance.apparent_mb_from_linear_mb(gdir)
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
