from __future__ import division

import warnings
warnings.filterwarnings("once", category=DeprecationWarning)

import os
import shutil
import unittest

import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm.utils import get_demo_file, rmsd, write_centerlines_to_shape
from oggm.tests import is_slow, ON_TRAVIS
from oggm.core.models import flowline, massbalance
from oggm import graphics

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_workflow')


def clean_dir(testdir):
    shutil.rmtree(testdir)
    os.makedirs(testdir)


def up_to_inversion(reset=False):
    """Run the tasks you want."""

    # test directory
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    if reset:
        clean_dir(TEST_DIR)

    # Init
    cfg.initialize()

    # Use multiprocessing
    cfg.PARAMS['use_multiprocessing'] = not ON_TRAVIS

    # Working dir
    cfg.PATHS['working_dir'] = TEST_DIR

    cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oeztal.tif')

    # Set up the paths and other stuffs
    cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
    cfg.PATHS['wgms_rgi_links'] = get_demo_file('RGI_WGMS_oeztal.csv')
    cfg.PATHS['glathida_rgi_links'] = get_demo_file('RGI_GLATHIDA_oeztal.csv')

    # Read in the RGI file
    rgi_file = get_demo_file('rgi_oeztal.shp')
    rgidf = gpd.GeoDataFrame.from_file(rgi_file)

    # Params
    cfg.PARAMS['border'] = 70
    cfg.PARAMS['use_inversion_params'] = True

    # Go
    gdirs = workflow.init_glacier_regions(rgidf)

    try:
        flowline.init_present_time_glacier(gdirs[0])
    except Exception:
        reset = True

    if reset:
        # First preprocessing tasks
        workflow.gis_prepro_tasks(gdirs)

        # Climate related tasks
        # See if CRU is running
        cfg.PARAMS['temp_use_local_gradient'] = False
        cfg.PATHS['climate_file'] = '~'
        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cfg.PATHS['cru_dir'] = os.path.dirname(cru_dir)
        workflow.climate_tasks(gdirs)

        # Use histalp for the actual test
        cfg.PARAMS['temp_use_local_gradient'] = True
        cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oeztal.nc')
        cfg.PATHS['cru_dir'] = '~'
        workflow.climate_tasks(gdirs)

        # Inversion
        workflow.inversion_tasks(gdirs)

    return gdirs


class TestWorkflow(unittest.TestCase):

    @is_slow
    def test_grow(self):

        gdirs = up_to_inversion()

        for gd in gdirs[0:2]:

            if gd.rgi_id in ['RGI40-11.00719']:
                # Bad bad glacier
                continue

            flowline.init_present_time_glacier(gd)
            flowline.find_inital_glacier(gd)


    @is_slow
    def test_shapefile_output(self):

        # Just to increase coveralls, hehe
        gdirs = up_to_inversion()
        fpath = os.path.join(TEST_DIR, 'centerlines.shp')
        write_centerlines_to_shape(gdirs, fpath)

        import salem
        shp = salem.utils.read_shapefile(fpath)
        self.assertTrue(shp is not None)

    @is_slow
    def test_init_present_time_glacier(self):

        gdirs = up_to_inversion()

        # Inversion Results
        cfg.PARAMS['invert_with_sliding'] = True
        workflow.inversion_tasks(gdirs)

        fpath = os.path.join(cfg.PATHS['working_dir'],
                             'inversion_optim_results.csv')
        df = pd.read_csv(fpath, index_col=0)
        r1 = rmsd(df['ref_volume_km3'], df['oggm_volume_km3'])
        r2 = rmsd(df['ref_volume_km3'], df['vas_volume_km3'])
        self.assertTrue(r1 < r2)

        cfg.PARAMS['invert_with_sliding'] = False
        workflow.inversion_tasks(gdirs)

        fpath = os.path.join(cfg.PATHS['working_dir'],
                             'inversion_optim_results.csv')
        df = pd.read_csv(fpath, index_col=0)
        r1 = rmsd(df['ref_volume_km3'], df['oggm_volume_km3'])
        r2 = rmsd(df['ref_volume_km3'], df['vas_volume_km3'])
        self.assertTrue(r1 < r2)

        # Init glacier
        d = gdirs[0].read_pickle('inversion_params')
        fs = d['fs']
        glen_a = d['glen_a']
        maxs = cfg.PARAMS['max_shape_param']
        for gdir in gdirs:
            flowline.init_present_time_glacier(gdir)
            mb_mod = massbalance.TstarMassBalanceModel(gdir)
            fls = gdir.read_pickle('model_flowlines')
            model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                            fs=fs, glen_a=glen_a)
            _vol = model.volume_km3
            _area = model.area_km2
            gldf = df.loc[gdir.rgi_id]
            assert_allclose(gldf['oggm_volume_km3'], _vol, rtol=0.01)
            assert_allclose(gldf['ref_area_km2'], _area, rtol=0.03)
            maxo = max([fl.order for fl in model.fls])
            for fl in model.fls:
                self.assertTrue(np.all(fl.bed_shape > 0))
                self.assertTrue(np.all(fl.bed_shape <= maxs))
                if len(model.fls) > 1:
                    if fl.order == (maxo-1):
                        self.assertTrue(fl.flows_to is fls[-1])
