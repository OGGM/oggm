import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import unittest
from functools import partial
import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

# Local imports
import oggm
from oggm.core.massbalance import LinearMassBalance
from oggm import utils, cfg
from oggm.cfg import SEC_IN_DAY
from oggm.core.sia2d import Upstream2D

# Tests
from oggm.tests.funcs import (dummy_bumpy_bed, dummy_constant_bed,
                              dummy_constant_bed_cliff,
                              dummy_mixed_bed, dummy_constant_bed_obstacle,
                              dummy_noisy_bed, dummy_parabolic_bed,
                              dummy_trapezoidal_bed, dummy_width_bed,
                              dummy_width_bed_tributary,
                              patch_url_retrieve_github)

# after oggm.test
import matplotlib.pyplot as plt

from oggm.core.flowline import (KarthausModel, FluxBasedModel,
                                MUSCLSuperBeeModel, MassConservationChecker)

FluxBasedModel = partial(FluxBasedModel, inplace=True)
KarthausModel = partial(KarthausModel, inplace=True)
MUSCLSuperBeeModel = partial(MUSCLSuperBeeModel, inplace=True)

pytestmark = pytest.mark.test_env("numerics")
do_plot = False
_url_retrieve = None


def setup_module(module):
    module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github


def teardown_module(module):
    oggm.utils._downloads.oggm_urlretrieve = module._url_retrieve


class TestIdealisedCases(unittest.TestCase):

    def setUp(self):
        N = 3
        cfg.initialize()
        self.glen_a = 2.4e-24    # Modern style Glen parameter A
        self.aglen_old = (N + 2) * 1.9e-24 / 2.  # outdated value
        self.fd = 2. * self.glen_a / (N + 2.)  # equivalent to glen_a
        self.fs = 0             # set slidin
        self.fs_old = 5.7e-20  # outdated value

    def tearDown(self):
        pass

    @pytest.mark.slow
    def test_constant_bed(self):

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 700, 2)
        for model in models:
            fls = dummy_constant_bed()
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, y0=0., glen_a=self.glen_a,
                          fs=self.fs, fixed_dt=10 * SEC_IN_DAY)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        if do_plot:
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=3e-3)
        np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=3e-3)

        self.assertTrue(utils.rmsd(lens[0], lens[2]) < 50.)
        self.assertTrue(utils.rmsd(lens[1], lens[2]) < 50.)
        self.assertTrue(utils.rmsd(volume[0], volume[2]) < 2e-3)
        self.assertTrue(utils.rmsd(volume[1], volume[2]) < 2e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[2]) < 1.0)
        self.assertTrue(utils.rmsd(surface_h[1], surface_h[2]) < 1.0)

    @pytest.mark.slow
    def test_mass_conservation(self):

        mb = LinearMassBalance(2600.)

        fls = dummy_constant_bed()
        model = MassConservationChecker(fls, mb_model=mb, y0=0.,
                                        glen_a=self.glen_a)
        model.run_until(200)
        assert_allclose(model.total_mass, model.volume_m3, rtol=1e-3)

        fls = dummy_noisy_bed()
        model = MassConservationChecker(fls, mb_model=mb, y0=0.,
                                        glen_a=self.glen_a)
        model.run_until(200)
        assert_allclose(model.total_mass, model.volume_m3, rtol=1e-3)

        fls = dummy_width_bed_tributary()
        model = MassConservationChecker(fls, mb_model=mb, y0=0.,
                                        glen_a=self.glen_a)
        model.run_until(200)
        assert_allclose(model.total_mass, model.volume_m3, rtol=1e-3)

        # Calving!
        fls = dummy_constant_bed(hmax=1000., hmin=0., nx=100)
        mb = LinearMassBalance(450.)
        model = MassConservationChecker(fls, mb_model=mb, y0=0.,
                                        glen_a=self.glen_a,
                                        is_tidewater=True)
        model.run_until(500)
        tot_vol = model.volume_m3 + model.calving_m3_since_y0
        assert_allclose(model.total_mass, tot_vol, rtol=2e-2)

    @pytest.mark.slow
    def test_min_slope(self):
        """ Check what is the min slope a flowline model can produce
        """

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel]
        kwargs = [{'fixed_dt': 3*SEC_IN_DAY}, {}, {}]
        lens = []
        surface_h = []
        volume = []
        min_slope = []
        yrs = np.arange(1, 700, 2)
        for model, kw in zip(models, kwargs):
            fls = dummy_constant_bed_obstacle()
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, y0=0., glen_a=self.glen_a,
                          **kw)

            length = yrs * 0.
            vol = yrs * 0.
            slope = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                fl = fls[-1]
                length[i] = fl.length_m
                vol[i] = fl.volume_km3

                hgt = np.where(fl.thick > 0, fl.surface_h, np.NaN)
                sl = np.arctan(-np.gradient(hgt, fl.dx_meter))
                slope[i] = np.rad2deg(np.nanmin(sl))

            lens.append(length)
            volume.append(vol)
            min_slope.append(slope)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=2e-3)
        np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=5e-3)

        self.assertTrue(utils.rmsd(volume[0], volume[2]) < 1e-2)
        self.assertTrue(utils.rmsd(volume[1], volume[2]) < 1e-2)

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(yrs, min_slope[0], 'r')
            plt.plot(yrs, min_slope[1], 'b')
            plt.plot(yrs, min_slope[2], 'g')
            plt.title('Compare min slope')
            plt.xlabel('years')
            plt.ylabel('[degrees]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=3)
            plt.show()

    @pytest.mark.slow
    def test_cliff(self):
        """ a test case for mass conservation in the flowline models
            the idea is to introduce a cliff in the sloping bed and see
            what the models do when the cliff height is changed
        """

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model in models:
            fls = dummy_constant_bed_cliff()
            mb = LinearMassBalance(2600.)

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

        if False:  # pragma: no cover
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=3)
            plt.show()

        # OK, so basically, Alex's tests below show that the other models
        # are wrong and produce too much mass. There is also another more
        # more trivial issue with the computation of the length, I added a
        # "to do" in the code.

        # Unit-testing perspective:
        # "verify" that indeed the models are wrong of more than 50%
        self.assertTrue(volume[1][-1] > volume[2][-1] * 1.5)
        # Karthaus is even worse
        self.assertTrue(volume[0][-1] > volume[1][-1])

        if False:
            # TODO: this will always fail so ignore it for now
            np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
            np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=2e-3)
            np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=2e-3)

            self.assertTrue(utils.rmsd(lens[0], lens[2]) < 50.)
            self.assertTrue(utils.rmsd(lens[1], lens[2]) < 50.)
            self.assertTrue(utils.rmsd(volume[0], volume[2]) < 1e-3)
            self.assertTrue(utils.rmsd(volume[1], volume[2]) < 1e-3)
            self.assertTrue(utils.rmsd(surface_h[0], surface_h[2]) < 1.0)
            self.assertTrue(utils.rmsd(surface_h[1], surface_h[2]) < 1.0)

    @pytest.mark.slow
    def test_equilibrium(self):

        models = [KarthausModel, FluxBasedModel]

        vols = []
        for model in models:
            fls = dummy_constant_bed()
            mb = LinearMassBalance(2600.)
            model = model(fls, mb_model=mb, glen_a=self.glen_a,
                          fixed_dt=10 * SEC_IN_DAY)

            model.run_until_equilibrium()
            vols.append(model.volume_km3)

        ref_vols = []
        for model in models:
            fls = dummy_constant_bed()
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a,
                          fixed_dt=10 * SEC_IN_DAY)

            model.run_until(600)
            ref_vols.append(model.volume_km3)

        np.testing.assert_allclose(ref_vols, vols, atol=0.01)

    def test_adaptive_ts(self):

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel]
        steps = [31 * SEC_IN_DAY, None, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_constant_bed()
            mb = LinearMassBalance(2600.)

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

        self.assertTrue(utils.rmsd(lens[0], lens[1]) < 50.)
        self.assertTrue(utils.rmsd(volume[2], volume[1]) < 1e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1]) < 5)
        self.assertTrue(utils.rmsd(surface_h[1], surface_h[2]) < 5)

    @pytest.mark.slow
    def test_timestepping(self):

        steps = ['ambitious',
                 'default',
                 'conservative',
                 'ultra-conservative'][::-1]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 400, 2)
        for step in steps:
            fls = dummy_constant_bed()
            mb = LinearMassBalance(2600.)

            model = FluxBasedModel(fls, mb_model=mb,
                                   glen_a=self.glen_a,
                                   time_stepping=step)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[3][-1], atol=1e-2)

    @pytest.mark.slow
    def test_bumpy_bed(self):

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel]
        steps = [15 * SEC_IN_DAY, None, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_bumpy_bed()
            mb = LinearMassBalance(2600.)

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
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)

        self.assertTrue(utils.rmsd(lens[0], lens[1]) < 50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1]) < 1e-2)
        self.assertTrue(utils.rmsd(volume[0], volume[2]) < 1e-2)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1]) < 5)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[2]) < 5)

    @pytest.mark.slow
    def test_noisy_bed(self):

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel]
        steps = [15 * SEC_IN_DAY, None, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        fls_orig = dummy_noisy_bed()
        for model, step in zip(models, steps):
            fls = copy.deepcopy(fls_orig)
            mb = LinearMassBalance(2600.)

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
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=3)
            plt.show()

        np.testing.assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)

        self.assertTrue(utils.rmsd(lens[0], lens[1]) < 100.)
        self.assertTrue(utils.rmsd(volume[0], volume[1]) < 1e-1)
        self.assertTrue(utils.rmsd(volume[0], volume[2]) < 1e-1)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1]) < 10)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[2]) < 10)

    @pytest.mark.slow
    def test_varying_width(self):
        """This test is for a flowline glacier of variying width, i.e with an
         accumulation area twice as wide as the tongue."""

        # TODO: @alexjarosch here we should have a look at MUSCLSuperBeeModel
        # set do_plot = True to see the plots

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel]
        steps = [15 * SEC_IN_DAY, None, None]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step in zip(models, steps):
            fls = dummy_width_bed()
            mb = LinearMassBalance(2600.)

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
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee'], loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-2)

        np.testing.assert_allclose(utils.rmsd(lens[0], lens[1]), 0., atol=70)
        np.testing.assert_allclose(utils.rmsd(volume[0], volume[1]), 0.,
                                   atol=1e-2)
        np.testing.assert_allclose(utils.rmsd(surface_h[0], surface_h[1]), 0.,
                                   atol=5)

    @pytest.mark.slow
    def test_tributary(self):

        models = [KarthausModel, FluxBasedModel]
        steps = [15 * SEC_IN_DAY, None]
        flss = [dummy_width_bed(), dummy_width_bed_tributary()]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model, step, fls in zip(models, steps, flss):
            mb = LinearMassBalance(2600.)

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

        np.testing.assert_allclose(utils.rmsd(lens[0], lens[1]), 0., atol=70)
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

    @pytest.mark.slow
    def test_trapezoidal_bed(self):

        tb = dummy_trapezoidal_bed()[0]
        np.testing.assert_almost_equal(tb._w0_m, tb.widths_m)
        np.testing.assert_almost_equal(tb.section, tb. widths_m * 0)
        np.testing.assert_almost_equal(tb.area_km2, 0)

        tb.section = tb.section
        np.testing.assert_almost_equal(tb._w0_m, tb.widths_m)
        np.testing.assert_almost_equal(tb.section, tb. widths_m * 0)
        np.testing.assert_almost_equal(tb.area_km2, 0)

        h = 50.
        sec = (2 * tb._w0_m + tb._lambdas * h) * h / 2
        tb.section = sec
        np.testing.assert_almost_equal(sec, tb.section)
        np.testing.assert_almost_equal(sec * 0 + h, tb.thick)
        np.testing.assert_almost_equal(tb._w0_m + tb._lambdas * h, tb.widths_m)
        akm = (tb._w0_m + tb._lambdas * h) * len(sec) * 100
        np.testing.assert_almost_equal(tb.area_m2, akm)

        models = [KarthausModel, FluxBasedModel]
        flss = [dummy_constant_bed(), dummy_trapezoidal_bed()]

        lens = []
        surface_h = []
        volume = []
        widths = []
        yrs = np.arange(1, 700, 2)
        for model, fls in zip(models, flss):
            mb = LinearMassBalance(2800.)

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

    @pytest.mark.slow
    def test_parabolic_bed(self):

        models = [KarthausModel, FluxBasedModel]
        flss = [dummy_constant_bed(), dummy_parabolic_bed()]

        lens = []
        surface_h = []
        volume = []
        widths = []
        yrs = np.arange(1, 700, 2)
        for model, fls in zip(models, flss):
            mb = LinearMassBalance(2800.)

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

    @pytest.mark.slow
    def test_mixed_bed(self):

        models = [KarthausModel, FluxBasedModel]
        flss = [dummy_constant_bed(), dummy_mixed_bed()]

        lens = []
        surface_h = []
        volume = []
        widths = []
        yrs = np.arange(1, 700, 2)
        # yrs = np.arange(1, 100, 2)
        for model, fls in zip(models, flss):
            mb = LinearMassBalance(2800.)

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
            plt.plot(lens[0], 'r', label='normal')
            plt.plot(lens[1], 'b', label='mixed')
            plt.legend()
            plt.show()

            plt.plot(volume[0], 'r', label='normal')
            plt.plot(volume[1], 'b', label='mixed')
            plt.legend()
            plt.show()

            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r', label='normal')
            plt.plot(surface_h[1], 'b', label='mixed')
            plt.legend()
            plt.show()

            plt.plot(widths[0], 'r', label='normal')
            plt.plot(widths[1], 'b', label='mixed')
            plt.legend()
            plt.show()

    @pytest.mark.slow
    def test_boundaries(self):

        fls = dummy_constant_bed()
        mb = LinearMassBalance(2000.)
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a,
                               fs=self.fs)
        with pytest.raises(RuntimeError) as excinfo:
            model.run_until(300)
        assert 'exceeds domain boundaries' in str(excinfo.value)


class TestSia2d(unittest.TestCase):

    def setUp(self):
        cfg.initialize()

    def tearDown(self):
        pass

    @pytest.mark.slow
    def test_constant_bed(self):

        map_dx = 100.
        yrs = np.arange(1, 400, 5)
        lens = []
        volume = []
        areas = []
        surface_h = []

        # Flowline case
        fls = dummy_constant_bed(hmax=3000., hmin=1000., nx=200, map_dx=map_dx,
                                 widths=1.)
        mb = LinearMassBalance(2600.)

        flmodel = FluxBasedModel(fls, mb_model=mb, y0=0.)

        length = yrs * 0.
        vol = yrs * 0.
        area = yrs * 0
        for i, y in enumerate(yrs):
            flmodel.run_until(y)
            length[i] = fls[-1].length_m
            vol[i] = fls[-1].volume_km3
            area[i] = fls[-1].area_km2

        lens.append(length)
        volume.append(vol)
        areas.append(area)
        surface_h.append(fls[-1].surface_h.copy())

        # Make a 2D bed out of the 1D
        bed_2d = np.repeat(fls[-1].bed_h, 3).reshape((fls[-1].nx, 3))

        sdmodel = Upstream2D(bed_2d, dx=map_dx, mb_model=mb, y0=0.,
                             ice_thick_filter=None)

        length = yrs * 0.
        vol = yrs * 0.
        area = yrs * 0
        for i, y in enumerate(yrs):
            sdmodel.run_until(y)
            surf_1d = sdmodel.ice_thick[:, 1]
            length[i] = np.sum(surf_1d > 0) * sdmodel.dx
            vol[i] = sdmodel.volume_km3 / 3
            area[i] = sdmodel.area_km2 / 3

        lens.append(length)
        volume.append(vol)
        areas.append(area)
        surface_h.append(sdmodel.surface_h[:, 1])

        if do_plot:
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Flowline', '2D'], loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Flowline', '2D'], loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Flowline', '2D'], loc=2)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=3e-3)

        self.assertTrue(utils.rmsd(lens[0], lens[1]) < 50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1]) < 2e-3)
        self.assertTrue(utils.rmsd(areas[0], areas[1]) < 2e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1]) < 1.0)

        # Equilibrium
        sdmodel.run_until_equilibrium()
        flmodel.run_until_equilibrium()
        assert_allclose(sdmodel.volume_km3 / 3, flmodel.volume_km3, atol=2e-3)
        assert_allclose(sdmodel.area_km2 / 3, flmodel.area_km2, atol=2e-3)

        # Store
        run_ds = sdmodel.run_until_and_store(sdmodel.yr+50)
        ts = run_ds['ice_thickness'].mean(dim=['y', 'x'])
        assert_allclose(ts, ts.values[0], atol=1)

        # Other direction
        bed_2d = np.repeat(fls[-1].bed_h, 3).reshape((fls[-1].nx, 3)).T

        sdmodel = Upstream2D(bed_2d, dx=map_dx, mb_model=mb, y0=0.,
                             ice_thick_filter=None)

        length = yrs * 0.
        vol = yrs * 0.
        area = yrs * 0
        for i, y in enumerate(yrs):
            sdmodel.run_until(y)
            surf_1d = sdmodel.ice_thick[1, :]
            length[i] = np.sum(surf_1d > 0) * sdmodel.dx
            vol[i] = sdmodel.volume_km3 / 3
            area[i] = sdmodel.area_km2 / 3

        lens.append(length)
        volume.append(vol)
        areas.append(area)
        surface_h.append(sdmodel.surface_h[:, 1])

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=3e-3)

        self.assertTrue(utils.rmsd(lens[0], lens[1]) < 50.)
        self.assertTrue(utils.rmsd(volume[0], volume[1]) < 2e-3)
        self.assertTrue(utils.rmsd(areas[0], areas[1]) < 2e-3)
        self.assertTrue(utils.rmsd(surface_h[0], surface_h[1]) < 1.0)

        # Equilibrium
        sdmodel.run_until_equilibrium()
        assert_allclose(sdmodel.volume_km3 / 3, flmodel.volume_km3, atol=2e-3)
        assert_allclose(sdmodel.area_km2 / 3, flmodel.area_km2, atol=2e-3)

    def test_bueler(self):
        # TODO: add formal test like Alex's
        # https://github.com/alexjarosch/sia-fluxlim
        pass
