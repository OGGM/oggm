from __future__ import division
from six.moves import zip

import warnings
warnings.filterwarnings("once", category=DeprecationWarning)

import unittest
import os
import copy
import pickle
import logging

logging.getLogger("Fiona").setLevel(logging.WARNING)
logging.getLogger("shapely").setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.DEBUG)

import shapely.geometry as shpg
import numpy as np
import shutil
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4
import multiprocessing as mp

# Local imports
from oggm.tests.test_graphics import init_hef
from oggm.models import flowline
from oggm.models import massbalance
from oggm import graphics as gr
from oggm import utils

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))

# test directory
testdir = os.path.join(current_dir, 'tmp')

# TODO: temporary: for conda installs
import osgeo.gdal
skip_conda = False
if osgeo.gdal.__version__ >= '1.11':
    skip_conda = True

do_plot = False

sec_in_year = 365*24*3600
sec_in_month = 31*24*3600
sec_in_day = 24*3600
sec_in_hour = 3600

dom_border = 80


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
    shape = surface_h * 0. + 3.e-03

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


class ConstantBalanceModel(massbalance.MassBalanceModel):
    """Simple gradient MB model."""

    def __init__(self, ela_h, grad=3., bias=0.):
        """ Instanciate."""

        super(ConstantBalanceModel, self).__init__(bias)

        self.ela_h = ela_h
        self.grad = grad

    def get_mb(self, heights, year):
        """Returns the mass-balance at given altitudes
        for a given moment in time."""

        mb = (heights - self.ela_h) * self.grad + self._bias
        return mb / sec_in_year / 900


class TestInitFlowline(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init_present_time_glacier(self):

        gdir = init_hef(border=dom_border)
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
                np.testing.assert_allclose(ofl.widths * gdir.grid.dx,
                                           fl.widths_m[0:len(ofl.widths)])

        np.testing.assert_allclose(0.573, vol)
        np.testing.assert_allclose(7350.0, fls[-1].length_m)
        np.testing.assert_allclose(gdir.glacier_area, area)

        if do_plot:  # pragma: no cover
            plt.plot(fls[-1].bed_h)
            plt.plot(fls[-1].surface_h)
            plt.show()


class TestMassBalance(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tstar_mb(self):

        gdir = init_hef(border=dom_border)
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

        gdir = init_hef(border=dom_border)
        flowline.init_present_time_glacier(gdir)

        mb_mod_ref = massbalance.TstarMassBalanceModel(gdir)
        mb_mod = massbalance.BackwardsMassBalanceModel(gdir, use_tstar=True)

        ofl = gdir.read_pickle('inversion_flowlines', div_id=0)
        h = np.array([])
        w = np.array([])
        for fl in ofl:
            h = np.append(h, fl.surface_h)
            w = np.append(w, fl.widths)

        ombh = mb_mod_ref.get_mb(h, None) * sec_in_year
        mbh = mb_mod.get_mb(h, None) * sec_in_year
        mb_mod.set_bias(100.)
        mbhb = mb_mod.get_mb(h, None) * sec_in_year

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

        ombh = mb_mod_ref.get_mb(h, None) * sec_in_year
        mbh = mb_mod.get_mb(h, None) * sec_in_year
        mb_mod.set_bias(-100.)
        mbhb = mb_mod.get_mb(h, None) * sec_in_year

        np.testing.assert_allclose(ombh, mbh, rtol=0.005)
        self.assertTrue(np.mean(mbhb) < np.mean(mbh))

        if do_plot:  # pragma: no cover
            plt.plot(h, ombh, 'o')
            plt.plot(h, mbh, 'x')
            plt.plot(h, mbhb, 'x')
            plt.show()


    def test_histalp_mb(self):

        gdir = init_hef(border=dom_border)
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
        yrs = np.arange(1983, 2003.1, 1)

        my_mb = ref_mbh * 0.
        for yr in yrs:
            my_mb += mb_mod.get_mb(h, yr)
        my_mb = my_mb / len(yrs) * 365 * 24 * 3600

        np.testing.assert_allclose(ref_mbh, my_mb, rtol=0.01)


class TestIdealisedCases(unittest.TestCase):

    def setUp(self):
        self.fs = 5.7e-20
        self.fd = 1.9e-24

    def tearDown(self):
        pass

    def test_constant_bed(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model in models:
            fls = dummy_constant_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=14*sec_in_day)

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
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-3)

        self.assertTrue(utils.rmsd(lens[0], lens[1])<50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1])<1e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1])<0.7)

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

    def test_equilibrium(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]

        vols = []
        for model in models:
            fls = dummy_constant_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=14*sec_in_day)

            model.run_until_equilibrium()
            vols.append(model.volume_km3)

        ref_vols = []
        for model in models:
            fls = dummy_constant_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=14*sec_in_day)

            model.run_until(600)
            ref_vols.append(model.volume_km3)

        np.testing.assert_allclose(ref_vols, vols, atol=0.005)


    def test_adaptive_ts(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        steps = [sec_in_month, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_constant_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=step)

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
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-3)

        self.assertTrue(utils.rmsd(lens[0], lens[1])<50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1])<1e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1])<0.7)

    def test_bumpy_bed(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        steps = [15*sec_in_day, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_bumpy_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=step)

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

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-3)

        self.assertTrue(utils.rmsd(lens[0], lens[1])<50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1])<1e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1])<0.7)

    def test_noisy_bed(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        steps = [15*sec_in_day, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        fls_orig = dummy_noisy_bed()
        for model, step in zip(models, steps):
            fls = copy.deepcopy(fls_orig)
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=step)

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

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)

    def test_varying_width(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        steps = [15*sec_in_day, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_width_bed()
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=step)

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

        np.testing.assert_allclose(utils.rmsd(lens[0], lens[1]), 0., atol=40)
        np.testing.assert_allclose(utils.rmsd(volume[0], volume[1]), 0.,
                                   atol=3e-3)
        np.testing.assert_allclose(utils.rmsd(surface_h[0], surface_h[1]), 0.,
                                   atol=2)

    def test_tributary(self):

        models = [flowline.KarthausModel, flowline.FluxBasedModel]
        steps = [15*sec_in_day, None]
        flss = [dummy_width_bed(), dummy_width_bed_tributary()]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step, fls in zip(models, steps, flss):
            mb = ConstantBalanceModel(2600.)

            model = model(fls, mb, 0., self.fs, self.fd,
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

        np.testing.assert_allclose(utils.rmsd(lens[0], lens[1]), 0., atol=56)
        np.testing.assert_allclose(utils.rmsd(volume[0], volume[1]), 0.,
                                   atol=6e-3)
        np.testing.assert_allclose(utils.rmsd(surface_h[0], surface_h[1]), 0.,
                                   atol=2)

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

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=14*sec_in_day)

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

        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-3)

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

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=14*sec_in_day)

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

        np.testing.assert_allclose(lens[0][-1], lens[1][-1], atol=700)
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-3)

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

            model = model(fls, mb, 0., self.fs, self.fd,
                          fixed_dt=14*sec_in_day)

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

        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-3)
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


class TestBackwardsIdealized(unittest.TestCase):

    def setUp(self):

        self.fs = 5.7e-20
        self.fd = 1.9e-24

        self.ela = 2800.

        origfls = dummy_constant_bed(nx=120, hmin=1800)

        mb = ConstantBalanceModel(self.ela)
        model = flowline.FluxBasedModel(origfls, mb, 0., self.fs, self.fd)
        model.run_until(500)
        self.glacier = copy.deepcopy(model.fls)

    def tearDown(self):
        pass

    def test_iterative_back(self):

        y0 = 0.
        y1 = 150.
        rtol = 0.001

        mb = ConstantBalanceModel(self.ela+50.)
        model = flowline.FluxBasedModel(self.glacier, mb, y0,
                                        self.fs, self.fd)

        past_model = flowline._find_inital_glacier(model, mb, y0, y1, rtol=rtol)
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
        model = flowline.FluxBasedModel(self.glacier, mb, y0,
                                        self.fs, self.fd)

        past_model = flowline._find_inital_glacier(model, mb, y0, y1, rtol=rtol)
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

    def test_fails(self):

        y0 = 0.
        y1 = 100.

        mb = ConstantBalanceModel(self.ela-150.)
        model = flowline.FluxBasedModel(self.glacier, mb, y0,
                                        self.fs, self.fd)
        self.assertRaises(RuntimeError, flowline._find_inital_glacier, model,
                          mb, y0, y1, rtol=0.02, max_ite=5)


class TestHEF(unittest.TestCase):

    def setUp(self):

        self.gdir = init_hef(border=dom_border)

        d = self.gdir.read_pickle('flowline_params')
        self.fs = d['fs']
        self.fd = d['fd']

    def tearDown(self):
        pass

    def test_equilibrium(self):

        #TODO: equilibrium test only working with parabolic bed
        flowline.init_present_time_glacier(self.gdir, min_shape=0)

        mb_mod = massbalance.TstarMassBalanceModel(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_mod, 0.,
                                        self.fs,
                                        self.fd)

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_area, self.gdir.glacier_area)

        model.run_until(50.)
        self.assertFalse(model.dt_warning)

        after_vol = model.volume_km3
        after_area = model.area_km2
        after_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_vol, after_vol, rtol=0.03)
        np.testing.assert_allclose(ref_area, after_area, rtol=0.03)
        np.testing.assert_allclose(ref_len, after_len, atol=200.01)


    def test_commitment(self):

        flowline.init_present_time_glacier(self.gdir, min_shape=0.,
                                           lambdas=0.2)

        mb_mod = massbalance.BackwardsMassBalanceModel(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_mod, 0.,
                                        self.fs,
                                        self.fd)

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m
        np.testing.assert_allclose(ref_area, self.gdir.glacier_area)

        model.run_until_equilibrium()
        self.assertTrue(model.yr > 100)

        after_vol_1 = model.volume_km3
        after_area_1 = model.area_km2
        after_len_1 = model.fls[-1].length_m

        flowline.init_present_time_glacier(self.gdir, min_shape=0.001,
                                           lambdas=0.2)
        glacier = self.gdir.read_pickle('model_flowlines')

        mb_mod = massbalance.BackwardsMassBalanceModel(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        model = flowline.FluxBasedModel(fls, mb_mod, 0.,
                                        self.fs,
                                        self.fd)

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m
        np.testing.assert_allclose(ref_area, self.gdir.glacier_area)

        model.run_until_equilibrium()
        self.assertTrue(model.yr > 100)

        after_vol_2 = model.volume_km3
        after_area_2 = model.area_km2
        after_len_2 = model.fls[-1].length_m

        self.assertTrue(after_vol_1 < (0.3 * ref_vol))
        self.assertTrue(after_vol_2 < (0.3 * ref_vol))

        if do_plot:  # pragma: no cover
            fig = plt.figure()
            plt.plot(glacier[-1].surface_h, 'b', label='start')
            plt.plot(model.fls[-1].surface_h, 'r', label='end')

            plt.plot(glacier[-1].bed_h, 'gray', linewidth=2)
            plt.legend(loc='best')
            plt.show()

    def test_find_t0(self):

        if skip_conda:
            raise unittest.SkipTest('On conda this wont work')

        # init bias 130, min_shape=0.0015
        # init bias 160, min_shape=0.001
        # init bias 150, min_shape=0.0012

        flowline.init_present_time_glacier(self.gdir)
        glacier = self.gdir.read_pickle('model_flowlines')
        df = pd.read_csv(utils.get_demo_file('hef_lengths.csv'), index_col=0)
        df.columns = ['Leclercq']

        vol_ref = flowline.FlowlineModel(glacier, None, None,
                                         None, None).volume_km3

        # flowline.find_inital_glacier(self.gdir)
        past_model = self.gdir.read_pickle('past_model')

        # from oggm import graphics as gr
        # import matplotlib.pyplot as plt
        # gr.plot_modeloutput(self.gdir, past_model)
        # plt.show()
        # return

        vol_start = past_model.volume_km3
        bef_fls = copy.deepcopy(past_model.fls)

        mylen = []
        for y in df.index:
            past_model.run_until(y)
            mylen.append(past_model.fls[-1].length_m)
        df['oggm'] = mylen
        df = df-df.iloc[-1]

        vol_end = past_model.volume_km3

        # np.testing.assert_allclose(df.iloc[0].oggm, df.iloc[0].dl, atol=250)
        np.testing.assert_allclose(vol_ref, vol_end, rtol=0.05)
        self.assertTrue(utils.rmsd(df.Leclercq, df.oggm) < 500.)

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
