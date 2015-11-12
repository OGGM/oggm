from __future__ import division

from nose.tools import assert_true

import os
import sys
import copy
import shutil
import warnings
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# Local imports
from oggm.prepro import gis, centerlines, geometry, climate, inversion
import oggm.conf as cfg
from oggm.utils import get_demo_file
from oggm import graphics
import pandas as pd
import geopandas as gpd

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(current_dir, 'tmp')

# For windows
suffix = ''
if 'win' in sys.platform:
    suffix = '_win'


def init_hef(reset=False):

    # test directory
    if not os.path.exists(testdir):
        os.makedirs(testdir)
        reset = True
    if not os.path.exists(os.path.join(testdir, 'RGI40-11.00897')):
        reset = True
    if not os.path.exists(os.path.join(testdir, 'RGI40-11.00897',
                                       'flowline_params.p')):
        reset = True

    # Init
    cfg.initialize()
    cfg.set_divides_db(get_demo_file('HEF_divided.shp'))
    cfg.paths['srtm_file'] = get_demo_file('hef_srtm.tif')
    cfg.paths['histalp_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.params['border'] = 40

    # loop because for some reason indexing wont work
    hef_file = get_demo_file('Hintereisferner.shp')
    rgidf = gpd.GeoDataFrame.from_file(hef_file)
    for index, entity in rgidf.iterrows():
        gdir = cfg.GlacierDir(entity, base_dir=testdir, reset=reset)

    if not reset:
        return gdir

    gis.define_glacier_region(gdir, entity)
    gis.glacier_masks(gdir)
    centerlines.compute_centerlines(gdir)
    centerlines.compute_downstream_lines(gdir)
    geometry.initialize_flowlines(gdir)
    geometry.catchment_area(gdir)
    geometry.catchment_width_geom(gdir)
    geometry.catchment_width_correction(gdir)
    climate.distribute_climate_data([gdir])
    climate.mu_candidates(gdir, div_id=0)
    hef_file = get_demo_file('mbdata_RGI40-11.00897.csv')
    mbdf = pd.read_csv(hef_file).set_index('YEAR')
    t_star, bias = climate.t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
    climate.local_mustar_apparent_mb(gdir, t_star[-1], bias[-1])

    inversion.prepare_for_inversion(gdir)
    ref_v = 0.573 * 1e9

    def to_optimize(x):
        fd = 1.9e-24 * x[0]
        fs = 5.7e-20 * x[1]
        v, _ = inversion.inversion_parabolic_point_slope(gdir,
                                                         fs=fs,
                                                         fd=fd)
        return (v - ref_v)**2

    import scipy.optimize as optimization
    out = optimization.minimize(to_optimize, [1,1],
                                bounds=((0.01, 1), (0.01, 1)),
                                tol=1e-3)['x']
    fd = 1.9e-24 * out[0]
    fs = 5.7e-20 * out[1]
    v, _ = inversion.inversion_parabolic_point_slope(gdir,
                                                     fs=fs,
                                                     fd=fd,
                                                     write=True)
    d = dict(fs=fs, fd=fd)
    gdir.write_pickle(d, 'flowline_params')
    return gdir


def internet_on():
    """Not so recommended it seems"""
    try:
        _ = urlopen('http://www.google.com', timeout=1)
        return True
    except URLError:
        pass
    return False


@image_comparison(baseline_images=['test_googlestatic' + suffix],
                  extensions=['png'])
def test_googlemap():
    if not internet_on():
        print('Internet is off, couldnt test for googlemaps')
        return
    gdir = init_hef()
    graphics.plot_googlemap(gdir)


@image_comparison(baseline_images=['test_centerlines' + suffix,
                                   'test_flowlines' + suffix,
                                   'test_downstream' + suffix,
                                   'test_downstream_cls' + suffix,
                                   ],
                  extensions=['png'])
def test_centerlines():

    gdir = init_hef()
    graphics.plot_centerlines(gdir)
    graphics.plot_centerlines(gdir, use_flowlines=True)
    graphics.plot_centerlines(gdir, add_downstream=True, use_flowlines=True)
    graphics.plot_centerlines(gdir, add_downstream=True)


@image_comparison(baseline_images=['test_width' + suffix,
                                   'test_width_corrected' + suffix
                                   ],
                  extensions=['png'])
def test_width():

    gdir = init_hef()
    graphics.plot_catchment_width(gdir)
    graphics.plot_catchment_width(gdir, corrected=True)




@image_comparison(baseline_images=['test_inversion' + suffix
                                   ],
                  extensions=['png'])
def test_inversion():
    gdir = init_hef()
    graphics.plot_inversion(gdir)

if __name__ == '__main__':
    import nose
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=r'.*guessing baseline image.*')
        nose.runmodule(argv=['-s', '-v', '--with-doctest'], exit=False)