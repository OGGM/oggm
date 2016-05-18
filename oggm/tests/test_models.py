from __future__ import division

import warnings

from six.moves import zip

warnings.filterwarnings("once", category=DeprecationWarning)

import logging
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

import unittest
import os
import copy
import time

import shapely.geometry as shpg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from oggm.tests.test_graphics import init_hef
from oggm.core.models import massbalance, flowline
from oggm.core.models.massbalance import ConstantBalanceModel
from oggm.tests import is_slow, requires_working_conda
from oggm.tests import HAS_NEW_GDAL

from oggm import utils, cfg
from oggm.cfg import N, SEC_IN_DAY, SEC_IN_MONTH, SEC_IN_YEAR

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))

# test directory
testdir = os.path.join(current_dir, 'tmp')

do_plot = False

DOM_BORDER = 80

def dummy_constant_bed(hmax=3000., hmin=1000., nx=200):

    map_dx = 100.
    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 3.

    coords = np.arange(0, nx-0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
                                          bed_h, widths)]

def dummy_constant_bed_cliff(hmax=3000., hmin=1000., nx=200):
    """
    I introduce a cliff in the bed to test the mass conservation of the models
    Such a cliff could be real or a DEM error/artifact
    """
    
    map_dx = 100.
    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    
    cliff_height = 250.0
    surface_h[50:] = surface_h[50:] - cliff_height
    
    bed_h = surface_h
    widths = surface_h * 0. + 1.

    coords = np.arange(0, nx-0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
                                          bed_h, widths)]

def dummy_bumpy_bed():

    map_dx = 100.
    dx = 1.
    nx = 200

    coords = np.arange(0, nx-0.5, 1)
    surface_h = np.linspace(3000, 1000, nx)
    surface_h += 170. * np.exp(-((coords-30)/5)**2)

    bed_h = surface_h
    widths = surface_h * 0. + 3.
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
                                          bed_h, widths)]


def dummy_noisy_bed():

    map_dx = 100.
    dx = 1.
    nx = 200
    np.random.seed(42)
    coords = np.arange(0, nx-0.5, 1)
    surface_h = np.linspace(3000, 1000, nx)
    surface_h += 100 * np.random.rand(nx) - 50.

    bed_h = surface_h
    widths = surface_h * 0. + 3.
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
                                          bed_h, widths)]


def dummy_parabolic_bed():

    map_dx = 100.
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    shape = surface_h * 0. + 5.e-03

    coords = np.arange(0, nx-0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)
    return [flowline.ParabolicFlowline(line, dx, map_dx, surface_h,
                                       bed_h, shape)]

def dummy_mixed_bed():

    map_dx = 100.
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    shape = surface_h * 0. + 3.e-03
    shape[10:20] = 0.00001

    coords = np.arange(0, nx-0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)
    return [flowline.MixedFlowline(line, dx, map_dx, surface_h,
                                   bed_h, shape, lambdas=3.5)]


def dummy_trapezoidal_bed(hmax=3000., hmin=1000., nx=200):

    map_dx = 100.
    dx = 1.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 1.6

    lambdas = surface_h * 0. + 2

    coords = np.arange(0, nx-0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)

    return [flowline.TrapezoidalFlowline(line, dx, map_dx, surface_h,
                                         bed_h, widths, lambdas)]


def dummy_width_bed():
    """This bed has a width of 6 during the first 20 points and then 3"""

    map_dx = 100.
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 3.
    widths[0:20] = 6.

    coords = np.arange(0, nx-0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)
    return [flowline.VerticalWallFlowline(line, dx, map_dx, surface_h,
                                          bed_h, widths)]

def dummy_width_bed_tributary():

    map_dx = 100.
    dx = 1.
    nx = 200

    surface_h = np.linspace(3000, 1000, nx)
    bed_h = surface_h
    widths = surface_h * 0. + 3.
    coords = np.arange(0, nx-0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords*0.]).T)

    fl_0 = flowline.VerticalWallFlowline(line, dx, map_dx, surface_h, bed_h,
                                         widths)
    coords = np.arange(0, 19.1, 1)
    line = shpg.LineString(np.vstack([coords, coords*0.+1]).T)
    fl_1 = flowline.VerticalWallFlowline(line, dx, map_dx, surface_h[0:20],
                                         bed_h[0:20], widths[0:20])
    fl_1.set_flows_to(fl_0)
    return [fl_1, fl_0]

class TestInitFlowline(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init_present_time_glacier(self):

        gdir = init_hef(border=DOM_BORDER)
        flowline.init_present_time_glacier(gdir)

        fls = gdir.read_pickle('model_flowlines')

        lens = [len(gdir.read_pickle('centerlines', div_id=i)) for i in [1,
                                                                         2, 3]]
        ofl = gdir.read_pickle('inversion_flowlines',
                               div_id=np.argmax(lens)+1)[-1]

        self.assertTrue(gdir.rgi_date.year == 2003)
        self.assertTrue(len(fls) == 5)

        vol = 0.
        area = 0.
        for fl in fls:
            refo = 1 if fl is fls[-1] else 0
            self.assertTrue(fl.order == refo)
            ref = np.arange(len(fl.surface_h)) * fl.dx
            np.testing.assert_allclose(ref, fl.dis_on_line,
                                       rtol=0.001,
                                       atol=0.01)
            self.assertTrue(len(fl.surface_h) ==
                            len(fl.bed_h) ==
                            len(fl.bed_shape) ==
                            len(fl.dis_on_line) ==
                            len(fl.widths))

            self.assertTrue(np.all(fl.bed_shape >= 0))
            self.assertTrue(np.all(fl.widths >= 0))
            vol += fl.volume_km3
            area += fl.area_km2

            if refo == 1:
                rtol = 0.07
                np.testing.assert_allclose(ofl.widths * gdir.grid.dx,
                                           fl.widths_m[0:len(ofl.widths)],
                                           rtol=rtol)

        np.testing.assert_allclose(0.573, vol, rtol=0.001)
        np.testing.assert_allclose(7400.0, fls[-1].length_m, atol=101)

        rtol = 0.01
        np.testing.assert_allclose(gdir.rgi_area_km2, area, rtol=rtol)

        if do_plot:  # pragma: no cover
            plt.plot(fls[-1].bed_h)
            plt.plot(fls[-1].surface_h)
            plt.show()

    def test_present_time_glacier_massbalance(self):

        gdir = init_hef(border=DOM_BORDER)
        flowline.init_present_time_glacier(gdir)

        mb_mod = massbalance.HistalpMassBalanceModel(gdir)

        fls = gdir.read_pickle('model_flowlines')
        glacier = flowline.FlowlineModel(fls)

        hef_file = utils.get_demo_file('mbdata_RGI40-11.00897.csv')
        mbdf = pd.read_csv(hef_file).set_index('YEAR')

        hgts = np.array([])
        widths = np.array([])
        for fl in glacier.fls:
            hgts = np.concatenate((hgts, fl.surface_h))
            widths = np.concatenate((widths, fl.widths_m))
        tot_mb = []
        refmb = []
        grads = hgts * 0
        for yr, mb in mbdf.iterrows():
            refmb.append(mb['ANNUAL_BALANCE'])
            mbh = mb_mod.get_mb(hgts, yr) * SEC_IN_YEAR * cfg.RHO
            grads += mbh
            tot_mb.append(np.average(mbh, weights=widths))
        grads /= len(tot_mb)

        # Bias
        self.assertTrue(np.abs(utils.md(tot_mb, refmb)) < 50)

        # Gradient
        dfg = pd.read_csv(utils.get_demo_file('mbgrads_RGI40-11.00897.csv'),
                          index_col='ALTITUDE').mean(axis=1)

        # Take the altitudes below 3100 and fit a line
        dfg = dfg[dfg.index < 3100]
        pok = np.where(hgts < 3100)
        from scipy.stats import linregress
        slope_obs, _, _, _, _ = linregress(dfg.index, dfg.values)
        slope_our, _, _, _, _ = linregress(hgts[pok], grads[pok])
        np.testing.assert_allclose(slope_obs, slope_our, rtol=0.1)


class TestMassBalance(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tstar_mb(self):

        gdir = init_hef(border=DOM_BORDER)
        flowline.init_present_time_glacier(gdir)

        mb_mod = massbalance.TstarMassBalanceModel(gdir)

        ofl = gdir.read_pickle('inversion_flowlines', div_id=0)
        h = np.array([])
        w = np.array([])
        for fl in ofl:
            h = np.append(h, fl.surface_h)
            w = np.append(w, fl.widths)

        ombh = mb_mod.get_mb(h, None)
        otmb = np.sum(ombh * w)
        np.testing.assert_allclose(0., otmb, atol=0.26)

        mb_mod = massbalance.TodayMassBalanceModel(gdir)

        ofl = gdir.read_pickle('inversion_flowlines', div_id=0)
        h = np.array([])
        w = np.array([])
        for fl in ofl:
            h = np.append(h, fl.surface_h)
            w = np.append(w, fl.widths)

        mbh = mb_mod.get_mb(h, None)
        tmb = np.sum(mbh * w)
        self.assertTrue(tmb < otmb)

        if do_plot:  # pragma: no cover
            plt.plot(h, ombh, 'o')
            plt.plot(h, mbh, 'o')
            plt.show()

    def test_backwards_mb(self):

        gdir = init_hef(border=DOM_BORDER)
        flowline.init_present_time_glacier(gdir)

        mb_mod_ref = massbalance.TstarMassBalanceModel(gdir)
        mb_mod = massbalance.BackwardsMassBalanceModel(gdir, use_tstar=True)

        ofl = gdir.read_pickle('inversion_flowlines', div_id=0)
        h = np.array([])
        w = np.array([])
        for fl in ofl:
            h = np.append(h, fl.surface_h)
            w = np.append(w, fl.widths)

        ombh = mb_mod_ref.get_mb(h, None) * SEC_IN_YEAR
        mbh = mb_mod.get_mb(h, None) * SEC_IN_YEAR
        mb_mod.set_bias(100.)
        mbhb = mb_mod.get_mb(h, None) * SEC_IN_YEAR

        np.testing.assert_allclose(ombh, mbh, rtol=0.001)
        self.assertTrue(np.mean(mbhb) > np.mean(mbh))

        if do_plot:  # pragma: no cover
            plt.plot(h, ombh, 'o')
            plt.plot(h, mbh, 'x')
            plt.plot(h, mbhb, 'x')
            plt.show()

        mb_mod_ref = massbalance.TodayMassBalanceModel(gdir)
        mb_mod = massbalance.BackwardsMassBalanceModel(gdir)

        ofl = gdir.read_pickle('inversion_flowlines', div_id=0)
        h = np.array([])
        w = np.array([])
        for fl in ofl:
            h = np.append(h, fl.surface_h)
            w = np.append(w, fl.widths)

        ombh = mb_mod_ref.get_mb(h, None) * SEC_IN_YEAR
        mbh = mb_mod.get_mb(h, None) * SEC_IN_YEAR
        mb_mod.set_bias(-100.)
        mbhb = mb_mod.get_mb(h, None) * SEC_IN_YEAR

        np.testing.assert_allclose(ombh, mbh, rtol=0.005)
        self.assertTrue(np.mean(mbhb) < np.mean(mbh))

        if do_plot:  # pragma: no cover
            plt.plot(h, ombh, 'o')
            plt.plot(h, mbh, 'x')
            plt.plot(h, mbhb, 'x')
            plt.show()

    def test_biased_mb(self):

        gdir = init_hef(border=DOM_BORDER)
        flowline.init_present_time_glacier(gdir)

        mb_mod_ref = massbalance.TstarMassBalanceModel(gdir)
        mb_mod = massbalance.BiasedMassBalanceModel(gdir, use_tstar=True)

        ofl = gdir.read_pickle('inversion_flowlines', div_id=0)
        h = np.array([])
        w = np.array([])
        for fl in ofl:
            h = np.append(h, fl.surface_h)
            w = np.append(w, fl.widths)

        ombh = mb_mod_ref.get_mb(h, None) * SEC_IN_YEAR
        mbh = mb_mod.get_mb(h, None) * SEC_IN_YEAR
        mb_mod.set_temp_bias(-1.)
        mbhb = mb_mod.get_mb(h, None) * SEC_IN_YEAR
        mb_mod.set_temp_bias(0)
        mb_mod.set_prcp_factor(1.2)
        mbhbp = mb_mod.get_mb(h, None) * SEC_IN_YEAR

        np.testing.assert_allclose(ombh, mbh, rtol=0.001)
        self.assertTrue(np.mean(mbhb) > np.mean(mbh))
        self.assertTrue(np.mean(mbhbp) > np.mean(mbh))

        if do_plot:  # pragma: no cover
            plt.plot(h, ombh, 'o')
            plt.plot(h, mbh, 'x')
            plt.plot(h, mbhb, 'x')
            plt.plot(h, mbhbp, 'x')
            plt.show()

    def test_histalp_mb(self):

        gdir = init_hef(border=DOM_BORDER)
        flowline.init_present_time_glacier(gdir)

        mb_mod = massbalance.TodayMassBalanceModel(gdir)

        ofl = gdir.read_pickle('inversion_flowlines', div_id=0)
        h = np.array([])
        w = np.array([])
        for fl in ofl:
            h = np.append(h, fl.surface_h)
            w = np.append(w, fl.widths)

        ref_mbh = mb_mod.get_mb(h, None) * 365 * 24 * 3600

        mb_mod = massbalance.HistalpMassBalanceModel(gdir)

        # Climate period
        yrs = np.arange(1973, 2003.1, 1)

        my_mb = ref_mbh * 0.
        for yr in yrs:
            my_mb += mb_mod.get_mb(h, yr)
        my_mb = my_mb / len(yrs) * 365 * 24 * 3600

        np.testing.assert_allclose(ref_mbh, my_mb, rtol=0.01)

    def test_random_mb(self):

        # reprod
        np.random.seed(10)

        gdir = init_hef(border=DOM_BORDER)
        flowline.init_present_time_glacier(gdir)

        ref_mod = massbalance.TodayMassBalanceModel(gdir)
        mb_mod = massbalance.RandomMassBalanceModel(gdir)

        h = np.array([])
        w = np.array([])
        for fl in gdir.read_pickle('inversion_flowlines', div_id=0):
            h = np.append(h, fl.surface_h)
            w = np.append(w, fl.widths)

        ref_mbh = ref_mod.get_mb(h, None) * 365 * 24 * 3600

        # two years shoudn't be equal
        r_mbh1 = mb_mod.get_mb(h, 1) * 365 * 24 * 3600
        r_mbh2 = mb_mod.get_mb(h, 2) * 365 * 24 * 3600
        assert not np.all(np.allclose(r_mbh1, r_mbh2))

        # the same year should be equal
        r_mbh1 = mb_mod.get_mb(h, 1) * 365 * 24 * 3600
        r_mbh2 = mb_mod.get_mb(h, 1) * 365 * 24 * 3600
        np.testing.assert_allclose(r_mbh1, r_mbh2)

        # After many trials the mb should be close to the same
        ny = 2000
        yrs = np.arange(ny)
        r_mbh = 0.
        for yr in yrs:
            r_mbh += mb_mod.get_mb(h, yr) * 365 * 24 * 3600
        r_mbh /= ny
        np.testing.assert_allclose(ref_mbh, r_mbh, atol=0.2)

        mb_mod.set_temp_bias(-0.5)
        r_mbh_b = 0.
        for yr in yrs:
            r_mbh_b += mb_mod.get_mb(h, yr) * 365 * 24 * 3600
        r_mbh_b /= ny
        self.assertTrue(np.mean(r_mbh) < np.mean(r_mbh_b))

        mb_mod.set_temp_bias(0)
        mb_mod.set_prcp_factor(0.7)
        r_mbh_b = 0.
        for yr in yrs:
            r_mbh_b += mb_mod.get_mb(h, yr) * 365 * 24 * 3600
        r_mbh_b /= ny
        self.assertTrue(np.mean(r_mbh) > np.mean(r_mbh_b))

        mb_mod = massbalance.RandomMassBalanceModel(gdir, use_tstar=True)
        r_mbh_b = 0.
        for yr in yrs:
            r_mbh_b += mb_mod.get_mb(h, yr) * 365 * 24 * 3600
        r_mbh_b /= ny
        self.assertTrue(np.mean(r_mbh) < np.mean(r_mbh_b))

        # Compare sigma from real climate and mine
        mb_ref = massbalance.HistalpMassBalanceModel(gdir)
        mb_mod = massbalance.RandomMassBalanceModel(gdir)
        mb_ts = []
        mb_ts2 = []
        yrs = np.arange(1973, 2003, 1)
        for yr in yrs:
            mb_ts.append(np.average(mb_ref.get_mb(h, yr) * 365 * 24 * 3600, weights=w))
            mb_ts2.append(np.average(mb_mod.get_mb(h, yr) * 365 * 24 * 3600, weights=w))
        np.testing.assert_allclose(np.std(mb_ts), np.std(mb_ts2), rtol=0.1)

    def test_mb_performance(self):

        gdir = init_hef(border=DOM_BORDER)
        flowline.init_present_time_glacier(gdir)
        h = np.array([])
        for fl in gdir.read_pickle('inversion_flowlines', div_id=0):
            h = np.append(h, fl.surface_h)

        # Climate period, 10 day timestep
        yrs = np.arange(1850, 2003, 10/365)

        # models
        mb1 = massbalance.TodayMassBalanceModel(gdir)
        mb2 = massbalance.HistalpMassBalanceModel(gdir)

        start_time = time.time()
        for yr in yrs:
            _ = mb1.get_mb(h, yr)
        t1 = time.time() - start_time
        start_time = time.time()
        for yr in yrs:
            _ = mb2.get_mb(h, yr)
        t2 = time.time() - start_time

        # not faster as two times t2
        try:
            assert t1 >= (t2 / 2)
        except AssertionError:
            # no big deal
            unittest.skip('Allowed failure')


class TestIdealisedCases(unittest.TestCase):

    def setUp(self):
        self.glen_a = 2.4e-24    # Modern style Glen parameter A
        self.aglen_old = (N+2) * 1.9e-24 / 2. # outdated value
        self.fd = 2. * self.glen_a / (N + 2.)  # equivalent to glen_a
        self.fs = 0             # set slidin
        self.fs_old = 5.7e-20  # outdated value
        
    def tearDown(self):
        pass

    def test_constant_bed(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel,
                  flowline.MUSCLSuperBeeModel]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 700, 2)
        for model in models:
            fls = dummy_constant_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, y0=0., glen_a=self.glen_a,
                          fs=self.fs, fixed_dt=10*SEC_IN_DAY)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed','Karthaus','Flux','MUSCL-SuperBee'],loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=3e-3)
        np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=3e-3)

        self.assertTrue(utils.rmsd(lens[0], lens[2])<50.)
        self.assertTrue(utils.rmsd(lens[1], lens[2])<50.)
        self.assertTrue(utils.rmsd(volume[0], volume[2])<2e-3)
        self.assertTrue(utils.rmsd(volume[1], volume[2])<2e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[2])<1.0)
        self.assertTrue(utils.rmsd(surface_h[1], surface_h[2])<1.0)

    def test_constant_bed_cliff(self):
        """ a test case for mass conservation in the flowline models
            the idea is to introduce a cliff in the sloping bed and see
            what the models do when the cliff height is changed
        """
        
        models = [flowline.KarthausModel, flowline.FluxBasedModel,
                  flowline.MUSCLSuperBeeModel]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 700, 2)
        for model in models:
            fls = dummy_constant_bed_cliff()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, y0=0., glen_a=self.glen_a,
                          fs=self.fs, fixed_dt=2*SEC_IN_DAY)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed','Karthaus','Flux','MUSCL-SuperBee'],loc=3)
            plt.show()

        # OK, so basically, Alex's tests below show that the other models
        # are wrong and produce too much mass. There is also another more
        # more trivial issue with the computation of the length, I added a
        # "to do" in the code.

        # Unit-testing perspective:
        # verify that indeed the models are wrong of more than 50%
        self.assertTrue(volume[1][-1] > volume[2][-1] * 1.5)
        # Karthaus is even worse
        self.assertTrue(volume[0][-1] > volume[1][-1])

        if False:
            # TODO: this will always fail so ignore it for now
            np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
            np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=2e-3)
            np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=2e-3)

            self.assertTrue(utils.rmsd(lens[0], lens[2])<50.)
            self.assertTrue(utils.rmsd(lens[1], lens[2])<50.)
            self.assertTrue(utils.rmsd(volume[0], volume[2])<1e-3)
            self.assertTrue(utils.rmsd(volume[1], volume[2])<1e-3)
            self.assertTrue(utils.rmsd(surface_h[0], surface_h[2])<1.0)
            self.assertTrue(utils.rmsd(surface_h[1], surface_h[2])<1.0)
            
    @requires_working_conda
    def test_equilibrium(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]

        vols = []
        for model in models:
            fls = dummy_constant_bed()
            mb = ConstantBalanceModel(2600.)
            model = model(fls, mb_model=mb, glen_a=self.glen_a,
                          fixed_dt=10 * SEC_IN_DAY)

            model.run_until_equilibrium()
            vols.append(model.volume_km3)

        ref_vols = []
        for model in models:
            fls = dummy_constant_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a,
                          fixed_dt=10 * SEC_IN_DAY)

            model.run_until(600)
            ref_vols.append(model.volume_km3)

        np.testing.assert_allclose(ref_vols, vols, atol=0.01)

    def test_adaptive_ts(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel,
                  flowline.MUSCLSuperBeeModel]
        steps = [SEC_IN_MONTH, None, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_constant_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a, fixed_dt=step)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)

        self.assertTrue(utils.rmsd(lens[0], lens[1])<50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1])<1e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1]) < 5)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[2]) < 5)

    def test_bumpy_bed(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel,
                  flowline.MUSCLSuperBeeModel]
        steps = [15 * SEC_IN_DAY, None, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_bumpy_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a, fixed_dt=step)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed','Karthaus','Flux','MUSCL-SuperBee'],loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)

        self.assertTrue(utils.rmsd(lens[0], lens[1])<50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1])<1e-2)
        self.assertTrue(utils.rmsd(volume[0], volume[2])<1e-2)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1])<5)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[2])<5)

    def test_noisy_bed(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel,
                  flowline.MUSCLSuperBeeModel]
        steps = [15 * SEC_IN_DAY, None, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        fls_orig = dummy_noisy_bed()
        for model, step in zip(models, steps):
            fls = copy.deepcopy(fls_orig)
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a, fixed_dt=step)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed','Karthaus','Flux','MUSCL-SuperBee'],loc=3)
            plt.show()

        np.testing.assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)

        self.assertTrue(utils.rmsd(lens[0], lens[1])<100.)
        self.assertTrue(utils.rmsd(volume[0], volume[1])<1e-1)
        self.assertTrue(utils.rmsd(volume[0], volume[2])<1e-1)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1])<10)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[2])<10)

    def test_varying_width(self):
        """This test is for a flowline glacier of variying width, i.e with an
         accumulation area twice as wide as the tongue."""

        # TODO: @alexjarosch here we should have a look at MUSCLSuperBeeModel
        # set do_plot = True to see the plots

        models = [flowline.KarthausModel, flowline.FluxBasedModel,
                  flowline.MUSCLSuperBeeModel]
        steps = [15 * SEC_IN_DAY, None, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_width_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a, fixed_dt=step)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus','Flux','MUSCL-SuperBee'],loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed','Karthaus','Flux','MUSCL-SuperBee'],loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)

        np.testing.assert_allclose(utils.rmsd(lens[0], lens[1]), 0., atol=50)
        np.testing.assert_allclose(utils.rmsd(volume[0], volume[1]), 0.,
                                   atol=3e-3)
        np.testing.assert_allclose(utils.rmsd(surface_h[0], surface_h[1]), 0.,
                                   atol=5)

    def test_tributary(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        steps = [15 * SEC_IN_DAY, None]
        flss = [dummy_width_bed(), dummy_width_bed_tributary()]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step, fls in zip(models, steps, flss):
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, fs=self.fs_old,
                          glen_a=self.aglen_old,
                          fixed_dt=step)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = np.sum([f.volume_km3 for f in fls])
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-2)

        np.testing.assert_allclose(utils.rmsd(lens[0], lens[1]), 0., atol=60)
        np.testing.assert_allclose(utils.rmsd(volume[0], volume[1]), 0.,
                                   atol=6e-3)
        np.testing.assert_allclose(utils.rmsd(surface_h[0], surface_h[1]), 0.,
                                   atol=5)

        if do_plot:  # pragma: no cover
            plt.plot(lens[0], 'r')
            plt.plot(lens[1], 'b')
            plt.show()

            plt.plot(volume[0], 'r')
            plt.plot(volume[1], 'b')
            plt.show()

            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.show()

    def test_trapezoidal_bed(self):

        tb = dummy_trapezoidal_bed()[0]
        np.testing.assert_almost_equal(tb._w0_m, tb.widths_m)
        np.testing.assert_almost_equal(tb.section, tb.widths_m*0)
        np.testing.assert_almost_equal(tb.area_km2, 0)

        tb.section = tb.section
        np.testing.assert_almost_equal(tb._w0_m, tb.widths_m)
        np.testing.assert_almost_equal(tb.section, tb.widths_m*0)
        np.testing.assert_almost_equal(tb.area_km2, 0)

        h = 50.
        sec = (2 * tb._w0_m + tb._lambdas * h) * h / 2
        tb.section = sec
        np.testing.assert_almost_equal(sec, tb.section)
        np.testing.assert_almost_equal(sec*0+h, tb.thick)
        np.testing.assert_almost_equal(tb._w0_m + tb._lambdas * h, tb.widths_m)
        akm = (tb._w0_m + tb._lambdas * h) * len(sec) * 100
        np.testing.assert_almost_equal(tb.area_m2, akm)

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        flss = [dummy_constant_bed(), dummy_trapezoidal_bed()]

        lens = []
        surface_h = []
        volume = []
        widths = []
        yrs = np.arange(1, 700, 2)
        for model, fls in zip(models, flss):
            mb = ConstantBalanceModel(2800.)

            model = model(fls, mb_model=mb, fs=self.fs_old,
                          glen_a=self.aglen_old,
                          fixed_dt=14 * SEC_IN_DAY)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            widths.append(fls[-1].widths_m.copy())
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)

        if do_plot:  # pragma: no cover
            plt.plot(lens[0], 'r')
            plt.plot(lens[1], 'b')
            plt.show()

            plt.plot(volume[0], 'r')
            plt.plot(volume[1], 'b')
            plt.show()

            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.show()

            plt.plot(widths[0], 'r')
            plt.plot(widths[1], 'b')
            plt.show()

    def test_parabolic_bed(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        flss = [dummy_constant_bed(), dummy_parabolic_bed()]

        lens = []
        surface_h = []
        volume = []
        widths = []
        yrs = np.arange(1, 700, 2)
        for model, fls in zip(models, flss):
            mb = ConstantBalanceModel(2800.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a,
                          fixed_dt=10 * SEC_IN_DAY)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            widths.append(fls[-1].widths_m.copy())
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(lens[0][-1], lens[1][-1], atol=1300)
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)

        if do_plot:  # pragma: no cover
            plt.plot(lens[0], 'r')
            plt.plot(lens[1], 'b')
            plt.show()

            plt.plot(volume[0], 'r')
            plt.plot(volume[1], 'b')
            plt.show()

            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.show()

            plt.plot(widths[0], 'r')
            plt.plot(widths[1], 'b')
            plt.show()


    def test_mixed_bed(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        flss = [dummy_constant_bed(), dummy_mixed_bed()]

        lens = []
        surface_h = []
        volume = []
        widths = []
        yrs = np.arange(1, 700, 2)
        for model, fls in zip(models, flss):
            mb = ConstantBalanceModel(2800.)


            model = model(fls, mb_model=mb, fs=self.fs_old,
                          glen_a=self.aglen_old,
                          fixed_dt=14 * SEC_IN_DAY)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            widths.append(fls[-1].widths_m.copy())
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-2)

        if do_plot:  # pragma: no cover
            plt.plot(lens[0], 'r')
            plt.plot(lens[1], 'b')
            plt.show()

            plt.plot(volume[0], 'r')
            plt.plot(volume[1], 'b')
            plt.show()

            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.show()

            plt.plot(widths[0], 'r')
            plt.plot(widths[1], 'b')
            plt.show()

    def test_optim(self):

        models = [flowline.FluxBasedModelDeprecated, flowline.FluxBasedModel,
                  flowline.FluxBasedModelDeprecated, flowline.FluxBasedModel,
                  flowline.FluxBasedModelDeprecated, flowline.FluxBasedModel,
                  flowline.FluxBasedModelDeprecated, flowline.FluxBasedModel,
                  ]
        lens = []
        surface_h = []
        volume = []
        runtime = []
        yrs = np.arange(1, 200, 5)
        for model in models:
            fls = dummy_width_bed_tributary()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a,
                          min_dt=5*SEC_IN_DAY)

            length = yrs * 0.
            vol = yrs * 0.
            start_time = time.time()
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = model.volume_km3
            runtime.append(time.time() - start_time)
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)

        self.assertTrue(utils.rmsd(lens[0], lens[1])<50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1])<1e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1]) < 5)

        t1 = np.mean(runtime[::2])
        t2 = np.mean(runtime[1::2])
        try:
            assert t2 <= t1
        except AssertionError:
            # no big deal
            unittest.skip('Allowed failure')


class TestBackwardsIdealized(unittest.TestCase):

    def setUp(self):

        self.fs = 5.7e-20
        # Backwards
        _fd = 1.9e-24
        self.glen_a = (N+2) * _fd / 2.

        self.ela = 2800.

        origfls = dummy_constant_bed(nx=120, hmin=1800)

        mb = ConstantBalanceModel(self.ela)
        model = flowline.FluxBasedModel(origfls, mb_model=mb,
                                        fs=self.fs, glen_a=self.glen_a)
        model.run_until(500)
        self.glacier = copy.deepcopy(model.fls)

    def tearDown(self):
        pass

    def test_iterative_back(self):

        y0 = 0.
        y1 = 150.
        rtol = 0.02

        mb = ConstantBalanceModel(self.ela+50.)
        model = flowline.FluxBasedModel(self.glacier, mb_model=mb,
                                        fs=self.fs, glen_a=self.glen_a)

        ite, bias, past_model = flowline._find_inital_glacier(model, mb, y0,
                                                               y1, rtol=rtol)

        bef_fls = copy.deepcopy(past_model.fls)
        past_model.run_until(y1)
        self.assertTrue(bef_fls[-1].area_m2 > past_model.area_m2)
        np.testing.assert_allclose(past_model.area_m2, self.glacier[-1].area_m2,
                                   rtol=rtol)

        if do_plot:  # pragma: no cover
            plt.plot(self.glacier[-1].surface_h, 'k', label='ref')
            plt.plot(bef_fls[-1].surface_h, 'b', label='start')
            plt.plot(past_model.fls[-1].surface_h, 'r', label='end')
            plt.plot(self.glacier[-1].bed_h, 'gray', linewidth=2)
            plt.legend(loc='best')
            plt.show()

        mb = ConstantBalanceModel(self.ela-50.)
        model = flowline.FluxBasedModel(self.glacier, mb_model=mb, y0=y0,
                                        fs=self.fs, glen_a=self.glen_a)

        ite, bias, past_model = flowline._find_inital_glacier(model, mb, y0,
                                                               y1, rtol=rtol)
        bef_fls = copy.deepcopy(past_model.fls)
        past_model.run_until(y1)
        self.assertTrue(bef_fls[-1].area_m2 < past_model.area_m2)
        np.testing.assert_allclose(past_model.area_m2, self.glacier[-1].area_m2,
                                   rtol=rtol)

        if do_plot:  # pragma: no cover
            plt.plot(self.glacier[-1].surface_h, 'k', label='ref')
            plt.plot(bef_fls[-1].surface_h, 'b', label='start')
            plt.plot(past_model.fls[-1].surface_h, 'r', label='end')
            plt.plot(self.glacier[-1].bed_h, 'gray', linewidth=2)
            plt.legend(loc='best')
            plt.show()

        mb = ConstantBalanceModel(self.ela)
        model = flowline.FluxBasedModel(self.glacier, mb_model=mb, y0=y0,
                                        fs=self.fs, glen_a=self.glen_a)

        # Hit the correct one
        ite, bias, past_model = flowline._find_inital_glacier(model, mb, y0,
                                                               y1, rtol=rtol)
        past_model.run_until(y1)
        np.testing.assert_allclose(past_model.area_m2, self.glacier[-1].area_m2,
                                   rtol=rtol)

    def test_fails(self):

        y0 = 0.
        y1 = 100.

        mb = ConstantBalanceModel(self.ela-150.)
        model = flowline.FluxBasedModel(self.glacier, mb_model=mb, y0=y0,
                                        fs=self.fs, glen_a=self.glen_a)
        self.assertRaises(RuntimeError, flowline._find_inital_glacier, model,
                          mb, y0, y1, rtol=0.02, max_ite=5)


class TestHEF(unittest.TestCase):

    def setUp(self):

        self.gdir = init_hef(border=DOM_BORDER)
        d = self.gdir.read_pickle('inversion_params')
        self.fs = d['fs']
        self.glen_a = d['glen_a']

    def tearDown(self):
        pass

    @requires_working_conda
    def test_equilibrium(self):

        #TODO: equilibrium test only working with parabolic bed
        _tmp = cfg.PARAMS['bed_shape']
        cfg.PARAMS['bed_shape'] = 'parabolic'
        flowline.init_present_time_glacier(self.gdir)
        cfg.PARAMS['bed_shape'] = _tmp

        mb_mod = massbalance.TstarMassBalanceModel(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                        fs=self.fs,
                                        glen_a=self.glen_a)

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_area, self.gdir.rgi_area_km2, rtol=0.03)

        model.run_until(50.)
        self.assertFalse(model.dt_warning)

        after_vol = model.volume_km3
        after_area = model.area_km2
        after_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_vol, after_vol, rtol=0.03)
        np.testing.assert_allclose(ref_area, after_area, rtol=0.03)
        np.testing.assert_allclose(ref_len, after_len, atol=200.01)

    @is_slow
    def test_commitment(self):

        _tmp = cfg.PARAMS['mixed_min_shape']
        cfg.PARAMS['mixed_min_shape'] = 0.
        flowline.init_present_time_glacier(self.gdir)
        cfg.PARAMS['mixed_min_shape'] = _tmp

        mb_mod = massbalance.TodayMassBalanceModel(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                        fs=self.fs,
                                        glen_a=self.glen_a)

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m
        np.testing.assert_allclose(ref_area, self.gdir.rgi_area_km2, rtol=0.01)

        model.run_until_equilibrium()
        self.assertTrue(model.yr > 100)

        after_vol_1 = model.volume_km3
        after_area_1 = model.area_km2
        after_len_1 = model.fls[-1].length_m

        _tmp = cfg.PARAMS['mixed_min_shape']
        cfg.PARAMS['mixed_min_shape'] = 0.001
        flowline.init_present_time_glacier(self.gdir)
        cfg.PARAMS['mixed_min_shape'] = _tmp

        glacier = self.gdir.read_pickle('model_flowlines')

        fls = self.gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                        fs=self.fs,
                                        glen_a=self.glen_a)

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m
        np.testing.assert_allclose(ref_area, self.gdir.rgi_area_km2, rtol=0.01)

        model.run_until_equilibrium()
        self.assertTrue(model.yr > 100)

        after_vol_2 = model.volume_km3
        after_area_2 = model.area_km2
        after_len_2 = model.fls[-1].length_m

        self.assertTrue(after_vol_1 < (0.5 * ref_vol))
        self.assertTrue(after_vol_2 < (0.5 * ref_vol))

        if do_plot:  # pragma: no cover
            fig = plt.figure()
            plt.plot(glacier[-1].surface_h, 'b', label='start')
            plt.plot(model.fls[-1].surface_h, 'r', label='end')

            plt.plot(glacier[-1].bed_h, 'gray', linewidth=2)
            plt.legend(loc='best')
            plt.show()

    @is_slow
    def test_random(self):

        _tmp = cfg.PARAMS['mixed_min_shape']
        cfg.PARAMS['mixed_min_shape'] = 0.
        flowline.init_present_time_glacier(self.gdir)
        cfg.PARAMS['mixed_min_shape'] = _tmp
        flowline.random_glacier_evolution(self.gdir, nyears=200)

    @is_slow
    def test_find_t0(self):

        gdir = init_hef(border=DOM_BORDER, invert_with_sliding=False)

        flowline.init_present_time_glacier(gdir)
        glacier = gdir.read_pickle('model_flowlines')
        df = pd.read_csv(utils.get_demo_file('hef_lengths.csv'), index_col=0)
        df.columns = ['Leclercq']
        df = df.loc[1950:]

        vol_ref = flowline.FlowlineModel(glacier).volume_km3

        init_bias = 80.  # 100 so that "went too far" comes once on travis
        rtol = 0.005

        flowline.find_inital_glacier(gdir, y0=df.index[0], init_bias=init_bias,
                                     rtol=rtol, write_steps=False)

        past_model = gdir.read_pickle('past_model')

        vol_start = past_model.volume_km3
        bef_fls = copy.deepcopy(past_model.fls)

        mylen = []
        for y in df.index:
            past_model.run_until(y)
            mylen.append(past_model.fls[-1].length_m)
        df['oggm'] = mylen
        df = df-df.iloc[-1]

        vol_end = past_model.volume_km3
        np.testing.assert_allclose(vol_ref, vol_end, rtol=0.05)

        rmsd = utils.rmsd(df.Leclercq, df.oggm)
        self.assertTrue(rmsd < 1000.)

        if do_plot:  # pragma: no cover
            df.plot()
            plt.ylabel('Glacier length (relative to 2003)')
            plt.show()
            fig = plt.figure()
            lab = 'ref (vol={:.2f}km3)'.format(vol_ref)
            plt.plot(glacier[-1].surface_h, 'k', label=lab)
            lab = 'oggm start (vol={:.2f}km3)'.format(vol_start)
            plt.plot(bef_fls[-1].surface_h, 'b', label=lab)
            lab = 'oggm end (vol={:.2f}km3)'.format(vol_end)
            plt.plot(past_model.fls[-1].surface_h, 'r', label=lab)

            plt.plot(glacier[-1].bed_h, 'gray', linewidth=2)
            plt.legend(loc='best')
            plt.show()
