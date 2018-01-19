# Python imports
import unittest
import netCDF4
import geopandas as gpd
import numpy as np
import os
import shutil
import salem
import oggm

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task
from oggm.tests.funcs import get_test_dir
from oggm.tests import RUN_BENCHMARK_TESTS
from oggm.utils import get_demo_file
from oggm.core import gis, centerlines
from oggm.core.massbalance import ConstantMassBalance

# do we event want to run the tests?
if not RUN_BENCHMARK_TESTS:
    raise unittest.SkipTest('Skipping all benchmark tests.')

do_plot = False


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
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # Climate tasks -- only data IO and tstar interpolation!
        execute_entity_task(tasks.process_cru_data, gdirs)
        tasks.distribute_t_stars(gdirs)
        execute_entity_task(tasks.apparent_mb, gdirs)

        mbref = salem.GeoTiff(get_demo_file('mb_SouthGlacier.tif'))
        demref = salem.GeoTiff(get_demo_file('dem_SouthGlacier.tif'))

        mbref = mbref.get_vardata()
        mbref[mbref == -9999] = np.NaN
        demref = demref.get_vardata()[np.isfinite(mbref)]
        mbref = mbref[np.isfinite(mbref)] * 1000

        # compute the bias to make it 0 SMB on the 2D DEM
        mbmod = ConstantMassBalance(gdirs[0], bias=0)
        mymb = mbmod.get_annual_mb(demref) * cfg.SEC_IN_YEAR * cfg.RHO
        mbmod = ConstantMassBalance(gdirs[0], bias=np.average(mymb))
        mymb = mbmod.get_annual_mb(demref) * cfg.SEC_IN_YEAR * cfg.RHO
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
            plt.scatter(mbref, demref, s=5, label='Obs (2007-2012), shifted to '
                                                   'Avg(SMB) = 0')
            plt.scatter(mymb, demref, s=5, label='OGGM MB at t*')
            plt.scatter(myfit, demref, s=5, label='Polyfit', c='C3')
            plt.xlabel('MB (mm w.e yr-1)')
            plt.ylabel('Altidude (m)')
            plt.legend()
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
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # Climate tasks -- only data IO and tstar interpolation!
        execute_entity_task(tasks.process_cru_data, gdirs)
        tasks.distribute_t_stars(gdirs)
        execute_entity_task(tasks.apparent_mb, gdirs)

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # We use the default parameters for this run
        execute_entity_task(tasks.volume_inversion, gdirs, glen_a=cfg.A, fs=0)
        execute_entity_task(tasks.filter_inversion_output, gdirs)

        df = utils.glacier_characteristics(gdirs)
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
        entity = gpd.GeoDataFrame.from_file(self.rgi_file).iloc[0]

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
        with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
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
