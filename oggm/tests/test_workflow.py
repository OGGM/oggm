from __future__ import division
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)
import os
import logging
import shutil
import unittest

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from nose.plugins.attrib import attr

# Locals
import oggm.conf as cfg
from oggm import workflow
from oggm.prepro import inversion
from oggm.utils import get_demo_file
from oggm.utils import rmsd
from oggm.tests import is_slow

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_dir(testdir):
    shutil.rmtree(testdir)
    os.makedirs(testdir)


def up_to_inversion():
    """Run the tasks you want."""

    # test directory
    testdir = os.path.join(CURRENT_DIR, 'tmp_workflow')
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    clean_dir(testdir)

    # Init
    cfg.initialize()

    # Prevent multiprocessing
    cfg.USE_MP = False

    # Working dir
    cfg.PATHS['working_dir'] = testdir

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

    # Go
    gdirs = workflow.init_glacier_regions(rgidf)

    # First preprocessing tasks
    workflow.gis_prepro_tasks(gdirs)

    # Climate related tasks
    workflow.climate_tasks(gdirs)

    # Merge climate and catchments
    workflow.execute_task(inversion.prepare_for_inversion, gdirs)
    fs, fd = inversion.optimize_inversion_params(gdirs)

    # Tests
    dfids = cfg.PATHS['glathida_rgi_links']
    try:
        gtd_df = pd.read_csv(dfids).sort_values(by=['RGI_ID'])
    except AttributeError:
        gtd_df = pd.read_csv(dfids).sort(columns=['RGI_ID'])
    dfids = gtd_df['RGI_ID'].values
    ref_gdirs = [gdir for gdir in gdirs if gdir.rgi_id in dfids]

    # Account for area differences between glathida and rgi
    ref_area_km2 = gtd_df.RGI_AREA.values
    ref_cs = gtd_df.VOLUME.values / (gtd_df.GTD_AREA.values**1.375)
    ref_volume_km3 = ref_cs * ref_area_km2**1.375

    vol = []
    area = []
    rgi = []
    for gdir in ref_gdirs:
        v, a = inversion.inversion_parabolic_point_slope(gdir, fs=fs, fd=fd,
                                                         write=True)
        vol.append(v)
        area.append(a)
        rgi.append(gdir.rgi_id)

    df = pd.DataFrame()
    df['rgi'] = rgi
    df['area'] = area
    df['ref_vol'] = ref_volume_km3
    df['oggm_vol'] = np.array(vol) * 1e-9
    df['vas_vol'] = 0.034*(ref_area_km2**1.375)
    df = df.set_index('rgi')

    shutil.rmtree(testdir)

    return df

class TestBasic(unittest.TestCase):

    @is_slow
    def test_first_shot(self):

        df = up_to_inversion()
        self.assertTrue(rmsd(df['ref_vol'], df['oggm_vol']) < rmsd(df['ref_vol'], df['vas_vol']))