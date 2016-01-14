from __future__ import division

import warnings
warnings.filterwarnings("once", category=DeprecationWarning)

import os
import shutil
import unittest

import pandas as pd
import geopandas as gpd
import numpy as np

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, rmsd
from oggm.tests import is_slow
from oggm.core.models import flowline, massbalance

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_workflow')


def clean_dir(testdir):
    shutil.rmtree(testdir)
    os.makedirs(testdir)


def up_to_inversion():
    """Run the tasks you want."""

    # test directory
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    clean_dir(TEST_DIR)

    # Init
    cfg.initialize()

    # Use multiprocessing
    cfg.PARAMS['use_multiprocessing'] = False

    # Working dir
    cfg.PATHS['working_dir'] = TEST_DIR

    cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
    cfg.PATHS['srtm_file'] = get_demo_file('srtm_oeztal.tif')

    # Set up the paths and other stuffs
    cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
    cfg.PATHS['histalp_file'] = get_demo_file('HISTALP_oeztal.nc')

    # Get test glaciers (all glaciers with MB or Thickness data)
    cfg.PATHS['wgms_rgi_links'] = get_demo_file('RGI_WGMS_oeztal.csv')
    cfg.PATHS['glathida_rgi_links'] = get_demo_file('RGI_GLATHIDA_oeztal.csv')

    # Read in the RGI file
    rgi_file = get_demo_file('rgi_oeztal.shp')
    rgidf = gpd.GeoDataFrame.from_file(rgi_file)

    # Params
    cfg.PARAMS['border'] = 50

    # Go
    gdirs = workflow.init_glacier_regions(rgidf)

    # First preprocessing tasks
    workflow.gis_prepro_tasks(gdirs)

    # Climate related tasks
    workflow.climate_tasks(gdirs)

    # Inversion related tasks
    workflow.inversion_tasks(gdirs)

    # Models!
    workflow.model_tasks(gdirs)

    return gdirs


class TestBasic(unittest.TestCase):

    @is_slow
    def test_up_to_inversion(self):

        gdirs = up_to_inversion()

        # Inversion Results
        fpath = os.path.join(cfg.PATHS['working_dir'],
                             'inversion_optim_results.csv')
        df = pd.read_csv(fpath, index_col=0)
        r1 = rmsd(df['ref_volume_km3'], df['oggm_volume_km3'])
        r2 = rmsd(df['ref_volume_km3'], df['vas_volume_km3'])
        self.assertTrue(r1 < r2)

        # Init glacier
        d = gdirs[0].read_pickle('inversion_params')
        fs = d['fs']
        fd = d['fd']
        for gdir in gdirs:
            mb_mod = massbalance.TstarMassBalanceModel(gdir)
            fls = gdir.read_pickle('model_flowlines')
            model = flowline.FluxBasedModel(fls, mb_mod, 0., fs, fd)
            _vol = model.volume_km3
            _area = model.area_km2
            gldf = df.loc[gdir.rgi_id]
            np.testing.assert_allclose(gldf['oggm_volume_km3'], _vol)
            np.testing.assert_allclose(gldf['ref_area_km2'], _area)

        # shutil.rmtree(TEST_DIR)