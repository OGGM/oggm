# Python imports
import unittest
import numpy as np
import os
import shutil
import xarray as xr
import pytest
import oggm
from scipy import optimize as optimization

salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task
from oggm.tests.funcs import get_test_dir
from oggm.utils import get_demo_file
from oggm.core import gis, centerlines
from oggm.core.massbalance import ConstantMassBalance

pytestmark = pytest.mark.test_env("benchmark")
do_plot = False


class TestSouthGlacier(unittest.TestCase):

    # Test case obtained from ITMIX
    # Data available at:
    # oggm-sample-data/tree/master/benchmarks/south_glacier
    #
    # Citation:
    #
    # Flowers, G.E., N. Roux, S. Pimentel, and C.G. Schoof (2011). Present
    # dynamics and future prognosis of a slowly surging glacier.
    # The Cryosphere, 5, 299-313. DOI: 10.5194/tc-5-299-2011, 2011.

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['dem_file'] = get_demo_file('dem_SouthGlacier.tif')
        cfg.PARAMS['border'] = 10
        cfg.PARAMS['prcp_fac'] = 2.5
        cfg.PARAMS['baseline_climate'] = 'CRU'

        self.tf = get_demo_file('cru_ts4.01.1901.2016.SouthGlacier.tmp.dat.nc')
        self.pf = get_demo_file('cru_ts4.01.1901.2016.SouthGlacier.pre.dat.nc')

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def get_ref_data(self, gdir):

        # Reference data
        df = salem.read_shapefile(get_demo_file('IceThick_SouthGlacier.shp'))
        coords = np.array([p.xy for p in df.geometry]).squeeze()
        df['lon'] = coords[:, 0]
        df['lat'] = coords[:, 1]
        df = df[['lon', 'lat', 'thick']]
        ii, jj = gdir.grid.transform(df['lon'], df['lat'], crs=salem.wgs84,
                                     nearest=True)
        df['i'] = ii
        df['j'] = jj
        df['ij'] = ['{:04d}_{:04d}'.format(i, j) for i, j in zip(ii, jj)]
        df = df.groupby('ij').mean()
        # Averaging converts to floats
        df['i'] = df['i'].astype(np.int64)
        df['j'] = df['j'].astype(np.int64)
        return df

    def test_mb(self):

        # This is a function to produce the MB function needed by Anna

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)

        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_period='2000-01-01_2010-01-01')

        mbref = salem.GeoTiff(get_demo_file('mb_SouthGlacier.tif'))
        demref = salem.GeoTiff(get_demo_file('dem_SouthGlacier.tif'))

        mbref = mbref.get_vardata()
        mbref[mbref == -9999] = np.nan
        demref = demref.get_vardata()[np.isfinite(mbref)]
        mbref = mbref[np.isfinite(mbref)] * 1000

        # compute the bias to make it 0 SMB on the 2D DEM
        rho = cfg.PARAMS['ice_density']
        mbmod = ConstantMassBalance(gdirs[0], bias=0, y0=1995)
        mymb = mbmod.get_annual_mb(demref) * cfg.SEC_IN_YEAR * rho
        mbmod = ConstantMassBalance(gdirs[0], y0=1995, bias=np.average(mymb))
        mymb = mbmod.get_annual_mb(demref) * cfg.SEC_IN_YEAR * rho
        np.testing.assert_allclose(np.average(mymb), 0., atol=1e-3)

        # Same for ref
        mbref = mbref - np.average(mbref)
        np.testing.assert_allclose(np.average(mbref), 0., atol=1e-3)

        # Fit poly
        p = np.polyfit(demref, mbref, deg=2)
        poly = np.poly1d(p)
        myfit = poly(demref)
        np.testing.assert_allclose(np.average(myfit), 0., atol=1e-3)

        if do_plot:
            import matplotlib.pyplot as plt
            plt.scatter(mbref, demref, s=5,
                        label='Obs (2007-2012), shifted to Avg(SMB) = 0')
            plt.scatter(mymb, demref, s=5, label='OGGM MB')
            plt.scatter(myfit, demref, s=5, label='Polyfit', c='C3')
            plt.xlabel('MB (mm w.e yr-1)')
            plt.ylabel('Altidude (m)')
            plt.legend()
            plt.show()

    def test_inversion_attributes(self):

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_period='2000-01-01_2010-01-01')

        # Tested tasks
        task_list = [
            tasks.gridded_attributes,
            tasks.gridded_mb_attributes,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # Check certain things
        gdir = gdirs[0]
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:

            # The max catchment area should be area of glacier
            assert (ds['catchment_area'].max() ==
                    ds['glacier_mask'].sum() * gdir.grid.dx**2)
            assert (ds['catchment_area_on_catch'].max() ==
                    ds['glacier_mask'].sum() * gdir.grid.dx**2)

            # In the lowest parts of the glaciers the data should be equivalent
            ds_low = ds.isel(y=ds.y < 6741500)
            np.testing.assert_allclose(ds_low['lin_mb_above_z'],
                                       ds_low['lin_mb_above_z_on_catch'])
            np.testing.assert_allclose(ds_low['oggm_mb_above_z'],
                                       ds_low['oggm_mb_above_z_on_catch'])

        # Build some loose tests based on correlation
        df = self.get_ref_data(gdir)
        vns = ['topo',
               'slope',
               'aspect',
               'slope_factor',
               'dis_from_border',
               'catchment_area',
               'catchment_area_on_catch',
               'lin_mb_above_z',
               'lin_mb_above_z_on_catch',
               'oggm_mb_above_z',
               'oggm_mb_above_z_on_catch',
               ]

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            for vn in vns:
                df[vn] = ds[vn].isel(x=('z', df['i']), y=('z', df['j']))

        # Loose tests based on correlations
        cf = df.corr()
        assert cf.loc['slope', 'slope_factor'] < -0.85
        assert cf.loc['slope', 'thick'] < -0.4
        assert cf.loc['dis_from_border', 'thick'] > 0.2
        assert cf.loc['oggm_mb_above_z', 'thick'] > 0.5
        assert cf.loc['lin_mb_above_z', 'thick'] > 0.5
        assert cf.loc['lin_mb_above_z', 'oggm_mb_above_z'] > 0.95

    def test_inversion(self):

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_period='2000-01-01_2010-01-01')
        execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs,
                            mb_years=[2000, 2009])

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # We use the default parameters for this run
        execute_entity_task(tasks.mass_conservation_inversion, gdirs)
        execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                            smooth_radius=None,
                            varname_suffix='_alt')
        execute_entity_task(tasks.distribute_thickness_interp, gdirs,
                            varname_suffix='_int')

        # Reference data
        gdir = gdirs[0]
        df = self.get_ref_data(gdir)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:

            v = ds.distributed_thickness_alt
            df['oggm_alt'] = v.isel(x=('z', df['i']), y=('z', df['j']))
            v = ds.distributed_thickness_int
            df['oggm_int'] = v.isel(x=('z', df['i']), y=('z', df['j']))

            ds['ref'] = xr.zeros_like(ds.distributed_thickness_int) * np.nan
            ds['ref'].data[df['j'], df['i']] = df['thick']

        rmsd_int = ((df.oggm_int - df.thick) ** 2).mean() ** .5
        rmsd_alt = ((df.oggm_int - df.thick) ** 2).mean() ** .5
        assert rmsd_int < 85
        assert rmsd_alt < 85

        dfm = df.mean()
        np.testing.assert_allclose(dfm.thick, dfm.oggm_int, 50)
        np.testing.assert_allclose(dfm.thick, dfm.oggm_alt, 50)

        if do_plot:
            import matplotlib.pyplot as plt
            df.plot(kind='scatter', x='oggm_int', y='thick')
            plt.axis('equal')
            df.plot(kind='scatter', x='oggm_alt', y='thick')
            plt.axis('equal')
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
            ds.ref.plot(ax=ax1)
            ds.distributed_thickness_int.plot(ax=ax2)
            ds.distributed_thickness_alt.plot(ax=ax3)
            plt.tight_layout()
            plt.show()

    @pytest.mark.slow
    def test_optimize_inversion(self):

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_period='2000-01-01_2010-01-01')
        execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs,
                            mb_years=[2000, 2009])

        # Reference data
        gdir = gdirs[0]
        df = self.get_ref_data(gdir)

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)

        glen_a = cfg.PARAMS['inversion_glen_a']
        fs = cfg.PARAMS['inversion_fs']

        def to_optimize(x):
            tasks.mass_conservation_inversion(gdir,
                                              glen_a=glen_a * x[0],
                                              fs=fs * x[1])
            tasks.distribute_thickness_per_altitude(gdir, smooth_radius=None)
            with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
                thick = ds.distributed_thickness.isel(x=('z', df['i']),
                                                      y=('z', df['j']))
                out = (np.abs(thick - df.thick)).mean()
            return out

        opti = optimization.minimize(to_optimize, [1., 1.],
                                     bounds=((0.01, 10), (0.01, 10)),
                                     tol=0.1)
        # Check results and save.
        execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                            glen_a=glen_a*opti['x'][0],
                            fs=0)
        execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                            smooth_radius=None)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            df['oggm'] = ds.distributed_thickness.isel(x=('z', df['i']),
                                                       y=('z', df['j']))
            ds['ref'] = xr.zeros_like(ds.distributed_thickness) * np.nan
            ds['ref'].data[df['j'], df['i']] = df['thick']

        rmsd = ((df.oggm - df.thick) ** 2).mean() ** .5
        assert rmsd < 30

        dfm = df.mean()
        np.testing.assert_allclose(dfm.thick, dfm.oggm, 10)
        if do_plot:
            import matplotlib.pyplot as plt
            df.plot(kind='scatter', x='oggm', y='thick')
            plt.axis('equal')
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            ds.ref.plot(ax=ax1)
            ds.distributed_thickness.plot(ax=ax2)
            plt.tight_layout()
            plt.show()

    def test_workflow(self):

        # This is a check that the inversion workflow works fine

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
            tasks.compute_downstream_line,
            tasks.compute_downstream_bedshape,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_period='2000-01-01_2010-01-01')
        execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs,
                            mb_years=[2000, 2009])

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # We use the default parameters for this run
        execute_entity_task(tasks.mass_conservation_inversion, gdirs)
        execute_entity_task(tasks.filter_inversion_output, gdirs)

        df = utils.compile_glacier_statistics(gdirs)
        df['inv_thickness_m'] = df['inv_volume_km3'] / df['rgi_area_km2'] * 1e3
        assert df.inv_thickness_m.iloc[0] < 100

        df = utils.compile_fixed_geometry_mass_balance(gdirs)
        assert len(df) > 100
        df.columns = ['glacier']

        # recalibrate with regio values
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            use_regional_avg=True,
                            overwrite_gdir=True,
                            ref_period='2000-01-01_2005-01-01')
        df['region'] = utils.compile_fixed_geometry_mass_balance(gdirs)['RGI60-01.16195']
        assert 0.99 < df.corr().iloc[0, 1] < 1
        assert np.all(df.std() > 450)

        if do_plot:
            import matplotlib.pyplot as plt
            from oggm.graphics import plot_inversion
            plot_inversion(gdirs)
            plt.show()


@pytest.mark.slow
class TestCoxeGlacier(unittest.TestCase):

    # Test case for a tidewater glacier

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        self.rgi_file = get_demo_file('rgi_RGI50-01.10299.shp')

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('dem_RGI50-01.10299.tif')
        cfg.PARAMS['border'] = 40
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PARAMS['use_kcalving_for_inversion'] = True
        cfg.PARAMS['use_kcalving_for_run'] = True
        cfg.PARAMS['prcp_fac'] = 2.5
        cfg.PARAMS['baseline_climate'] = 'CRU'
        cfg.PARAMS['evolution_model'] = 'FluxBased'

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_set_width(self):
        entity = gpd.read_file(self.rgi_file).iloc[0]

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

        # Test that area and area-altitude elev is fine
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo_smoothed'][:]
        rhgt = topo[np.where(mask)][:]

        fls = gdir.read_pickle('inversion_flowlines')
        hgt, widths = gdir.get_inversion_flowline_hw()

        bs = 100
        bins = np.arange(utils.nicenumber(np.min(hgt), bs, lower=True),
                         utils.nicenumber(np.max(hgt), bs) + 1,
                         bs)
        h1, b = np.histogram(hgt, weights=widths, density=True, bins=bins)
        h2, b = np.histogram(rhgt, density=True, bins=bins)
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)
        assert utils.rmsd(h1, h2) < 0.02  # less than 2% error
        new_area = np.sum(widths * fls[-1].dx * gdir.grid.dx)
        np.testing.assert_allclose(new_area, gdir.rgi_area_m2)

        centerlines.terminus_width_correction(gdir, new_width=714)

        fls = gdir.read_pickle('inversion_flowlines')
        hgt, widths = gdir.get_inversion_flowline_hw()

        # Check that the width is ok
        np.testing.assert_allclose(fls[-1].widths[-1] * gdir.grid.dx, 714)

        # Check for area distrib
        bins = np.arange(utils.nicenumber(np.min(hgt), bs, lower=True),
                         utils.nicenumber(np.max(hgt), bs) + 1,
                         bs)
        h1, b = np.histogram(hgt, weights=widths, density=True, bins=bins)
        h2, b = np.histogram(rhgt, density=True, bins=bins)
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)
        assert utils.rmsd(h1, h2) < 0.02  # less than 2% error
        new_area = np.sum(widths * fls[-1].dx * gdir.grid.dx)
        np.testing.assert_allclose(new_area, gdir.rgi_area_m2)

    def test_run(self):

        entity = gpd.read_file(self.rgi_file).iloc[0]
        entity.RGIId = 'RGI60-01.10299'
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

        # Climate tasks
        tasks.process_dummy_cru_file(gdir, seed=0)
        tasks.mb_calibration_from_geodetic_mb(gdir)
        tasks.apparent_mb_from_any_mb(gdir)

        # Inversion tasks
        tasks.find_inversion_calving_from_any_mb(gdir)

        # Final preparation for the run
        tasks.init_present_time_glacier(gdir)

        # check that calving happens in the real context as well
        tasks.run_constant_climate(gdir, bias=0, y0=1985, nyears=200,
                                   temperature_bias=-0.5)
        with xr.open_dataset(gdir.get_filepath('model_diagnostics')) as ds:
            assert ds.calving_m3[-1] > 10
