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
from oggm.utils import get_demo_file, rmsd, interp_nans
from oggm.tests import is_slow, ON_TRAVIS, ON_FABIENS_LAPTOP, requires_fabiens_laptop
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
    cfg.PARAMS['border'] = 70

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
        workflow.climate_tasks(gdirs)

        # Inversion related tasks
        workflow.inversion_tasks(gdirs)

    return gdirs


class TestWorkflow(unittest.TestCase):

    # def test_find_past_glacier(self):
    #
    #     gdirs = up_to_inversion()
    #     for gd in gdirs:
    #
    #         # if gd.rgi_id not in ['RGI40-11.00897']:
    #         #     continue
    #         flowline.init_present_time_glacier(gd)
    #         flowline.find_inital_glacier(gd, do_plot=True, init_bias=100)

    @is_slow
    @requires_fabiens_laptop
    def test_grow(self):

        gdirs = up_to_inversion()

        d = gdirs[0].read_pickle('inversion_params')
        fs = d['fs']
        fd = d['fd']

        for gd in gdirs:

            if gd.rgi_id in ['RGI40-11.00719']:
                # Bad bad glacier
                continue

            flowline.init_present_time_glacier(gd)
            mb_mod = massbalance.TstarMassBalanceModel(gd, bias=0)

            fls = gd.read_pickle('model_flowlines')
            model = flowline.FluxBasedModel(fls, mb_mod, 0., fs, fd)

            if ON_FABIENS_LAPTOP:
                graphics.plot_modeloutput_section_withtrib(model)
                plt.savefig('/home/mowglie/' + gd.rgi_id + '_as0.png')
                graphics.plot_modeloutput_section(model, title=gd.rgi_id)
                plt.savefig('/home/mowglie/' + gd.rgi_id + '_s0.png')
                graphics.plot_modeloutput_map(gd, model)
                plt.savefig('/home/mowglie/' + gd.rgi_id + '_m0.png')

            model.run_until_equilibrium(rate=0.001)

            if ON_FABIENS_LAPTOP:
                print(gd.rgi_id + ' equi found in: {}'.format(model.yr))
                graphics.plot_modeloutput_section_withtrib(model)
                plt.savefig('/home/mowglie/' + gd.rgi_id + '_as1.png')
                graphics.plot_modeloutput_section(model, title=gd.rgi_id)
                plt.savefig('/home/mowglie/' + gd.rgi_id + '_s1.png')
                graphics.plot_modeloutput_map(gd, model)
                plt.savefig('/home/mowglie/' + gd.rgi_id + '_m1.png')
                plt.close('all')

    @is_slow
    def test_init_present_time_glacier(self):

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
        maxs = cfg.PARAMS['max_shape_param']
        for gdir in gdirs:
            flowline.init_present_time_glacier(gdir)
            mb_mod = massbalance.TstarMassBalanceModel(gdir)
            fls = gdir.read_pickle('model_flowlines')
            model = flowline.FluxBasedModel(fls, mb_mod, 0., fs, fd)
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
