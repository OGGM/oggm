# Python imports
import unittest
import geopandas as gpd
import numpy as np
import os
import shutil
import salem
import xarray as xr
import pytest
import oggm
from scipy import optimize as optimization

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task
from oggm.tests.funcs import get_test_dir, patch_url_retrieve_github
from oggm.utils import get_demo_file
from oggm.core import gis, centerlines
from oggm.core.massbalance import ConstantMassBalance

pytestmark = pytest.mark.test_env("benchmark")
do_plot = False
_url_retrieve = None


def setup_module(module):
    module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github


def teardown_module(module):
    oggm.utils._downloads.oggm_urlretrieve = module._url_retrieve


class TestSouthGlacier(unittest.TestCase):

    # Test case optained from ITMIX
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
        cfg.PATHS['cru_dir'] = os.path.dirname(cfg.PATHS['dem_file'])
        cfg.PARAMS['border'] = 10

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
        return df.groupby('ij').mean()

    def test_mb(self):

        # This is a function to produce the MB function needed by Anna

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
            tasks.process_cru_data,
            tasks.local_t_star,
            tasks.mu_star_calibration,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        mbref = salem.GeoTiff(get_demo_file('mb_SouthGlacier.tif'))
        demref = salem.GeoTiff(get_demo_file('dem_SouthGlacier.tif'))

        mbref = mbref.get_vardata()
        mbref[mbref == -9999] = np.NaN
        demref = demref.get_vardata()[np.isfinite(mbref)]
        mbref = mbref[np.isfinite(mbref)] * 1000

        # compute the bias to make it 0 SMB on the 2D DEM
        rho = cfg.PARAMS['ice_density']
        mbmod = ConstantMassBalance(gdirs[0], bias=0)
        mymb = mbmod.get_annual_mb(demref) * cfg.SEC_IN_YEAR * rho
        mbmod = ConstantMassBalance(gdirs[0], bias=np.average(mymb))
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
            plt.scatter(mymb, demref, s=5, label='OGGM MB at t*')
            plt.scatter(myfit, demref, s=5, label='Polyfit', c='C3')
            plt.xlabel('MB (mm w.e yr-1)')
            plt.ylabel('Altidude (m)')
            plt.legend()
            plt.show()

    def test_inversion(self):

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
            tasks.process_cru_data,
            tasks.local_t_star,
            tasks.mu_star_calibration,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # We use the default parameters for this run
        execute_entity_task(tasks.mass_conservation_inversion, gdirs)
        execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
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

            ds['ref'] = xr.zeros_like(ds.distributed_thickness_int) * np.NaN
            ds['ref'].data[df['j'], df['i']] = df['thick']

        rmsd_int = ((df.oggm_int - df.thick) ** 2).mean() ** .5
        rmsd_alt = ((df.oggm_int - df.thick) ** 2).mean() ** .5
        assert rmsd_int < 80
        assert rmsd_alt < 80

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

    def test_optimize_inversion(self):

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_regions(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
            tasks.process_cru_data,
            tasks.local_t_star,
            tasks.mu_star_calibration,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

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
            tasks.distribute_thickness_per_altitude(gdir)
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
        execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            df['oggm'] = ds.distributed_thickness.isel(x=('z', df['i']),
                                                       y=('z', df['j']))
            ds['ref'] = xr.zeros_like(ds.distributed_thickness) * np.NaN
            ds['ref'].data[df['j'], df['i']] = df['thick']

        rmsd = ((df.oggm - df.thick) ** 2).mean() ** .5
        assert rmsd < 60

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
        gdirs = workflow.init_glacier_regions(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
            tasks.process_cru_data,
            tasks.local_t_star,
            tasks.mu_star_calibration,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # We use the default parameters for this run
        execute_entity_task(tasks.mass_conservation_inversion, gdirs)
        execute_entity_task(tasks.filter_inversion_output, gdirs)

        df = utils.compile_glacier_statistics(gdirs)
        assert df.inv_thickness_m[0] < 100

        if do_plot:
            import matplotlib.pyplot as plt
            from oggm.graphics import plot_inversion
            plot_inversion(gdirs)
            plt.show()


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
        gis.define_glacier_region(gdir, entity=entity)
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
