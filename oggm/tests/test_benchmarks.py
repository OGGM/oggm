# Python imports
import unittest
import geopandas as gpd
import numpy as np
import os
import shutil
import oggm

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task
from oggm.tests.funcs import get_test_dir
from oggm.tests import RUN_BENCHMARK_TESTS
from oggm.utils import get_demo_file

# do we event want to run the tests?
if not RUN_BENCHMARK_TESTS:
    raise unittest.SkipTest('Skipping all benchmark tests.')


class TestSouthGlacier(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
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
