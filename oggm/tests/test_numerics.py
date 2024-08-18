import unittest
from functools import partial
import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

# Local imports
import oggm
from oggm.core.massbalance import LinearMassBalance, ScalarMassBalance
from oggm.core.inversion import find_sia_flux_from_thickness
from oggm import utils, cfg
from oggm.cfg import SEC_IN_DAY
from oggm.core.sia2d import Upstream2D
from oggm.exceptions import InvalidParamsError

# Tests
from oggm.tests.funcs import (dummy_bumpy_bed, dummy_constant_bed,
                              dummy_constant_bed_cliff,
                              dummy_mixed_bed, dummy_constant_bed_obstacle,
                              dummy_noisy_bed, dummy_parabolic_bed,
                              dummy_trapezoidal_bed, dummy_width_bed,
                              dummy_width_bed_tributary, bu_tidewater_bed,
                              dummy_bed_tributary_tail_to_head,
                              dummy_mixed_trap_rect_bed)

# after oggm.test
import matplotlib.pyplot as plt

from oggm.core.flowline import (KarthausModel, FluxBasedModel,
                                MassRedistributionCurveModel,
                                MassConservationChecker, SemiImplicitModel)
from oggm.tests.ext.sia_fluxlim import MUSCLSuperBeeModel

FluxBasedModel = partial(FluxBasedModel, inplace=True)
KarthausModel = partial(KarthausModel, inplace=True)
MUSCLSuperBeeModel = partial(MUSCLSuperBeeModel, inplace=True)
SemiImplicitModel = partial(SemiImplicitModel, inplace=True)

pytestmark = pytest.mark.test_env("numerics")
do_plot = False

pytest.importorskip('geopandas')
pytest.importorskip('rasterio')
pytest.importorskip('salem')


class TestIdealisedCases(unittest.TestCase):

    def setUp(self):
        N = 3
        cfg.initialize()
        self.glen_a = 2.4e-24  # Modern style Glen parameter A
        self.aglen_old = (N + 2) * 1.9e-24 / 2.  # outdated value
        self.fd = 2. * self.glen_a / (N + 2.)  # equivalent to glen_a
        self.fs = 0  # set sliding
        self.fs_old = 5.7e-20  # outdated value

    def tearDown(self):
        pass

    @pytest.mark.slow
    def test_constant_bed(self):

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel,
                  SemiImplicitModel]

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
                assert model.yr == y
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

            # We are almost at equilibrium. Spec MB should be close to 0
            assert_allclose(mb.get_specific_mb(fls=fls), 0, atol=10)

        if do_plot:
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.plot(yrs, lens[3], 'm')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.plot(yrs, volume[3], 'm')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.plot(surface_h[3], 'm')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee',
                        'Implicit'], loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_almost_equal(lens[3][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=3e-3)
        np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=3e-3)
        np.testing.assert_allclose(volume[3][-1], volume[2][-1], atol=3e-3)

        assert utils.rmsd(lens[0], lens[2]) < 50.
        assert utils.rmsd(lens[1], lens[2]) < 50.
        assert utils.rmsd(lens[3], lens[2]) < 50.
        assert utils.rmsd(volume[0], volume[2]) < 2e-3
        assert utils.rmsd(volume[1], volume[2]) < 2e-3
        assert utils.rmsd(volume[3], volume[2]) < 2e-3
        assert utils.rmsd(surface_h[0], surface_h[2]) < 1.0
        assert utils.rmsd(surface_h[1], surface_h[2]) < 1.0
        assert utils.rmsd(surface_h[3], surface_h[2]) < 1.0

    def test_length(self):

        mb = LinearMassBalance(2600.)
        yrs = np.arange(1, 300, 2)

        model = FluxBasedModel(dummy_constant_bed(), mb_model=mb, y0=0.)
        length_1 = yrs * 0.
        for i, y in enumerate(yrs):
            model.run_until(y)
            length_1[i] = model.length_m

        cfg.PARAMS['glacier_length_method'] = 'consecutive'
        model = FluxBasedModel(dummy_constant_bed(), mb_model=mb, y0=0.)
        length_2 = yrs * 0.
        for i, y in enumerate(yrs):
            model.run_until(y)
            length_2[i] = model.length_m

        np.testing.assert_allclose(length_1, length_2)

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
        fls = dummy_constant_bed(hmax=900., hmin=-100., nx=100)
        mb = LinearMassBalance(450.)
        model = MassConservationChecker(fls, mb_model=mb, y0=0.,
                                        glen_a=self.glen_a,
                                        do_kcalving=True,
                                        is_tidewater=True)
        model.run_until(500)
        tot_vol = model.volume_m3 + model.calving_m3_since_y0
        assert_allclose(model.total_mass, tot_vol, rtol=0.02)

    @pytest.mark.slow
    def test_staggered_diagnostics(self):

        mb = LinearMassBalance(2600.)
        fls = dummy_constant_bed()
        model_flux = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model_flux.run_until(700)
        assert_allclose(mb.get_specific_mb(fls=fls), 0, atol=10)

        model_impl = SemiImplicitModel(fls, mb_model=mb, y0=0.)
        model_impl.run_until(700)

        for model in [model_flux, model_impl]:
            # Check the flux just for fun
            fl = model.flux_stag[0]
            assert fl[0] == 0

            # Now check the diags
            df = model.get_diagnostics()
            fl = model.fls[0]
            df['my_flux'] = np.cumsum(mb.get_annual_mb(fl.surface_h) *
                                      fl.widths_m * fl.dx_meter *
                                      cfg.SEC_IN_YEAR).clip(0)

            df = df.loc[df['ice_thick'] > 0]

            # Also convert ours
            df['ice_flux'] *= cfg.SEC_IN_YEAR
            df['ice_velocity'] *= cfg.SEC_IN_YEAR

            assert_allclose(np.abs(df['ice_flux'] - df['my_flux']), 0, atol=35e3)
            assert df['ice_velocity'].max() > 25

            if isinstance(model, oggm.core.flowline.FluxBasedModel):
                df['tributary_flux'] *= cfg.SEC_IN_YEAR
                assert df['tributary_flux'].max() == 0

        fls = dummy_width_bed_tributary()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until(500)

        df = model.get_diagnostics()
        df['ice_velocity'] *= cfg.SEC_IN_YEAR
        df['tributary_flux'] *= cfg.SEC_IN_YEAR
        df = df.loc[df['ice_thick'] > 0]
        assert df['ice_velocity'].max() > 50
        assert df['tributary_flux'].max() > 30e4

        df = model.get_diagnostics(fl_id=0)
        df = df.loc[df['ice_thick'] > 0]
        df['ice_velocity'] *= cfg.SEC_IN_YEAR
        df['tributary_flux'] *= cfg.SEC_IN_YEAR
        assert df['ice_velocity'].max() > 10
        assert df['tributary_flux'].max() == 0

    @pytest.mark.slow
    def test_min_slope(self):
        """ Check what is the min slope a flowline model can produce
        """

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel,
                  SemiImplicitModel]
        kwargs = [{'fixed_dt': 3 * SEC_IN_DAY}, {}, {}, {}]
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
                assert model.yr == y
                fl = fls[-1]
                length[i] = fl.length_m
                vol[i] = fl.volume_km3

                hgt = np.where(fl.thick > 0, fl.surface_h, np.nan)
                sl = np.arctan(-np.gradient(hgt, fl.dx_meter))
                slope[i] = np.rad2deg(np.nanmin(sl))

            lens.append(length)
            volume.append(vol)
            min_slope.append(slope)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        np.testing.assert_allclose(lens[3][-1], lens[1][-1], atol=101)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=2e-3)
        np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=5e-3)
        np.testing.assert_allclose(volume[3][-1], volume[2][-1], atol=5e-3)

        assert utils.rmsd(volume[0], volume[2]) < 1e-2
        assert utils.rmsd(volume[1], volume[2]) < 1e-2
        assert utils.rmsd(volume[3], volume[2]) < 1e-2

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(yrs, lens[0], 'r')
            plt.plot(yrs, lens[1], 'b')
            plt.plot(yrs, lens[2], 'g')
            plt.plot(yrs, lens[3], 'm')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.plot(yrs, volume[3], 'm')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(yrs, min_slope[0], 'r')
            plt.plot(yrs, min_slope[1], 'b')
            plt.plot(yrs, min_slope[2], 'g')
            plt.plot(yrs, min_slope[3], 'm')
            plt.title('Compare min slope')
            plt.xlabel('years')
            plt.ylabel('[degrees]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.plot(surface_h[3], 'm')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee',
                        'Implicit'], loc=3)
            plt.show()

    @pytest.mark.slow
    def test_cliff(self):
        """ a test case for mass conservation in the flowline models
            the idea is to introduce a cliff in the sloping bed and see
            what the models do when the cliff height is changed
        """

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel,
                  SemiImplicitModel]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model in models:
            fls = dummy_constant_bed_cliff()
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, y0=0., glen_a=self.glen_a,
                          fs=self.fs, fixed_dt=2 * SEC_IN_DAY)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                assert model.yr == y
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
            plt.plot(yrs, lens[3], 'm')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.plot(yrs, volume[3], 'm')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.plot(surface_h[3], 'm')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee',
                        'Implicit'], loc=3)
            plt.show()

        # OK, so basically, Alex's tests below show that the other models
        # are wrong and produce too much mass. There is also another more
        # more trivial issue with the computation of the length, I added a
        # "to do" in the code.

        # Unit-testing perspective:
        # "verify" that indeed the models are wrong of more than 50%
        assert volume[1][-1] > volume[2][-1] * 1.5
        # SemiImplicit less wrong compared to FuxBased
        assert volume[3][-1] > volume[2][-1] * 1.25
        # Karthaus is even worse
        assert volume[0][-1] > volume[1][-1]

        if False:  # pragma: no cover
            # TODO: this will always fail so ignore it for now
            np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
            np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=2e-3)
            np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=2e-3)

            assert utils.rmsd(lens[0], lens[2]) < 50.
            assert utils.rmsd(lens[1], lens[2]) < 50.
            assert utils.rmsd(volume[0], volume[2]) < 1e-3
            assert utils.rmsd(volume[1], volume[2]) < 1e-3
            assert utils.rmsd(surface_h[0], surface_h[2]) < 1.0
            assert utils.rmsd(surface_h[1], surface_h[2]) < 1.0

    @pytest.mark.slow
    def test_equilibrium(self):

        models = [KarthausModel, FluxBasedModel, SemiImplicitModel]

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

    def test_run_until(self):

        # Just check that exotic times are guaranteed to be met
        yrs = np.array([10.2, 10.2, 10.200001, 10.3, 99.999, 150.])

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel,
                  SemiImplicitModel]
        steps = [31 * SEC_IN_DAY, None, None, None]

        # Annual update
        lens = []
        surface_h = []
        volume = []
        for model, step in zip(models, steps):
            fls = dummy_constant_bed()
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a, fixed_dt=step)

            # Codecov
            with pytest.raises(InvalidParamsError):
                model.step(0.)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                assert model.yr == y
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_almost_equal(lens[3][-1], lens[1][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)
        np.testing.assert_allclose(volume[3][-1], volume[2][-1], atol=1e-2)

        assert utils.rmsd(lens[0], lens[1]) < 50.
        assert utils.rmsd(lens[0], lens[3]) < 50.
        assert utils.rmsd(volume[2], volume[1]) < 1e-3
        assert utils.rmsd(volume[2], volume[3]) < 1e-3
        assert utils.rmsd(surface_h[0], surface_h[1]) < 5
        assert utils.rmsd(surface_h[1], surface_h[2]) < 5
        assert utils.rmsd(surface_h[3], surface_h[2]) < 5

        # Always update
        lens = []
        surface_h = []
        volume = []
        for model, step in zip(models, steps):
            fls = dummy_constant_bed()
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, glen_a=self.glen_a, fixed_dt=step,
                          mb_elev_feedback='always')

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                assert model.yr == y
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_almost_equal(lens[0][-1], lens[3][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[3][-1], atol=1e-2)

        assert utils.rmsd(lens[0], lens[1]) < 50.
        assert utils.rmsd(lens[0], lens[3]) < 50.
        assert utils.rmsd(volume[2], volume[1]) < 1e-3
        assert utils.rmsd(volume[2], volume[3]) < 1e-3
        assert utils.rmsd(surface_h[0], surface_h[1]) < 5
        assert utils.rmsd(surface_h[1], surface_h[2]) < 5
        assert utils.rmsd(surface_h[3], surface_h[2]) < 5

    @pytest.mark.slow
    def test_adaptive_ts(self):

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel,
                  SemiImplicitModel]
        steps = [31 * SEC_IN_DAY, None, None, None]
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
                assert model.yr == y
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_almost_equal(lens[0][-1], lens[3][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[3][-1], atol=1e-2)

        assert utils.rmsd(lens[0], lens[1]) < 50.
        assert utils.rmsd(lens[0], lens[3]) < 50.
        assert utils.rmsd(volume[2], volume[1]) < 1e-3
        assert utils.rmsd(volume[2], volume[3]) < 1e-3
        assert utils.rmsd(surface_h[0], surface_h[1]) < 5
        assert utils.rmsd(surface_h[1], surface_h[2]) < 5
        assert utils.rmsd(surface_h[3], surface_h[2]) < 5

    @pytest.mark.slow
    def test_bumpy_bed(self):

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel,
                  SemiImplicitModel]
        steps = [15 * SEC_IN_DAY, None, None, None]
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
                assert model.yr == y
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
            plt.plot(yrs, lens[3], 'm')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.plot(yrs, volume[2], 'm')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.plot(surface_h[2], 'm')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee',
                        'Implicit'], loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_almost_equal(lens[0][-1], lens[3][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[3][-1], atol=1e-2)

        assert utils.rmsd(lens[0], lens[1]) < 50.
        assert utils.rmsd(lens[0], lens[3]) < 50.
        assert utils.rmsd(volume[0], volume[1]) < 1e-2
        assert utils.rmsd(volume[0], volume[2]) < 1e-2
        assert utils.rmsd(volume[0], volume[3]) < 1e-2
        assert utils.rmsd(surface_h[0], surface_h[1]) < 5
        assert utils.rmsd(surface_h[0], surface_h[2]) < 5
        assert utils.rmsd(surface_h[0], surface_h[3]) < 5

    @pytest.mark.slow
    def test_noisy_bed(self):

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel,
                  SemiImplicitModel]
        steps = [15 * SEC_IN_DAY, None, None, None]
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
                assert model.yr == y
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
            plt.plot(yrs, lens[3], 'm')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.plot(yrs, volume[2], 'm')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.plot(surface_h[2], 'm')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee',
                        'Implicit'], loc=3)
            plt.show()

        np.testing.assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        np.testing.assert_allclose(lens[0][-1], lens[3][-1], atol=101)
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[3][-1], atol=1e-2)

        assert utils.rmsd(lens[0], lens[1]) < 100.
        assert utils.rmsd(lens[0], lens[3]) < 100.
        assert utils.rmsd(volume[0], volume[1]) < 1e-1
        assert utils.rmsd(volume[0], volume[2]) < 1e-1
        assert utils.rmsd(volume[0], volume[3]) < 1e-1
        assert utils.rmsd(surface_h[0], surface_h[1]) < 10
        assert utils.rmsd(surface_h[0], surface_h[2]) < 10
        assert utils.rmsd(surface_h[0], surface_h[3]) < 10

    @pytest.mark.slow
    def test_varying_width(self):
        """This test is for a flowline glacier of varying width, i.e with an
         accumulation area twice as wide as the tongue."""
        # set do_plot = True to see the plots

        models = [KarthausModel, FluxBasedModel, MUSCLSuperBeeModel,
                  SemiImplicitModel]
        steps = [15 * SEC_IN_DAY, None, None, None]
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
                assert model.yr == y
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
            plt.plot(yrs, lens[3], 'm')
            plt.title('Compare Length')
            plt.xlabel('years')
            plt.ylabel('[m]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(yrs, volume[0], 'r')
            plt.plot(yrs, volume[1], 'b')
            plt.plot(yrs, volume[2], 'g')
            plt.plot(yrs, volume[3], 'm')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km^3]')
            plt.legend(['Karthaus', 'Flux', 'MUSCL-SuperBee', 'Implicit'],
                       loc=2)

            plt.figure()
            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'g')
            plt.plot(surface_h[3], 'm')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.legend(['Bed', 'Karthaus', 'Flux', 'MUSCL-SuperBee',
                        'Implicit'], loc=3)
            plt.show()

        np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
        np.testing.assert_almost_equal(lens[0][-1], lens[3][-1])
        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-2)
        np.testing.assert_allclose(volume[1][-1], volume[3][-1], atol=1e-2)

        np.testing.assert_allclose(utils.rmsd(lens[0], lens[1]), 0., atol=70)
        np.testing.assert_allclose(utils.rmsd(lens[3], lens[1]), 0., atol=50)
        np.testing.assert_allclose(utils.rmsd(volume[0], volume[1]), 0.,
                                   atol=1e-2)
        np.testing.assert_allclose(utils.rmsd(volume[3], volume[1]), 0.,
                                   atol=4e-3)
        np.testing.assert_allclose(utils.rmsd(surface_h[0], surface_h[1]), 0.,
                                   atol=5)
        np.testing.assert_allclose(utils.rmsd(surface_h[3], surface_h[1]), 0.,
                                   atol=3)

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
                assert model.yr == y
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
    def test_multiple_tributary(self):

        models = [FluxBasedModel, FluxBasedModel]
        flss = [dummy_width_bed(),
                dummy_width_bed_tributary(n_trib=5)]
        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 300, 2)
        for model, fls in zip(models, flss):
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, fs=self.fs_old,
                          glen_a=self.aglen_old)

            length = yrs * 0.
            vol = yrs * 0.
            for i, y in enumerate(yrs):
                model.run_until(y)
                assert model.yr == y
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
        np.testing.assert_almost_equal(tb.section, tb.widths_m * 0)
        np.testing.assert_almost_equal(tb.area_km2, 0)

        tb.section = tb.section
        np.testing.assert_almost_equal(tb._w0_m, tb.widths_m)
        np.testing.assert_almost_equal(tb.section, tb.widths_m * 0)
        np.testing.assert_almost_equal(tb.area_km2, 0)

        h = 50.
        sec = (2 * tb._w0_m + tb._lambdas * h) * h / 2
        tb.section = sec
        np.testing.assert_almost_equal(sec, tb.section)
        np.testing.assert_almost_equal(sec * 0 + h, tb.thick)
        np.testing.assert_almost_equal(tb._w0_m + tb._lambdas * h, tb.widths_m)
        akm = (tb._w0_m + tb._lambdas * h) * len(sec) * 100
        np.testing.assert_almost_equal(tb.area_m2, akm)

        models = [KarthausModel, FluxBasedModel, SemiImplicitModel]
        flss = [dummy_constant_bed(), dummy_trapezoidal_bed(),
                dummy_trapezoidal_bed()]

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
                assert model.yr == y
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            widths.append(fls[-1].widths_m.copy())
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-2)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=1e-2)

        np.testing.assert_allclose(lens[2][-1], lens[1][-1])
        np.testing.assert_allclose(volume[2][-1], volume[1][-1], atol=2e-5)

        np.testing.assert_allclose(utils.rmsd(lens[2], lens[1]), 0., atol=6)
        np.testing.assert_allclose(utils.rmsd(volume[2], volume[1]), 0.,
                                   atol=5e-5)
        np.testing.assert_allclose(utils.rmsd(surface_h[2], surface_h[1]), 0.,
                                   atol=2e-2)

        if do_plot:  # pragma: no cover
            plt.plot(lens[0], 'r')
            plt.plot(lens[1], 'b')
            plt.plot(lens[2], 'm')
            plt.title('Length')
            plt.show()

            plt.plot(volume[0], 'r')
            plt.plot(volume[1], 'b')
            plt.plot(volume[2], 'm')
            plt.title('Volume')
            plt.show()

            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r')
            plt.plot(surface_h[1], 'b')
            plt.plot(surface_h[2], 'm')
            plt.title('Surface_h')
            plt.show()

            plt.plot(widths[0], 'r')
            plt.plot(widths[1], 'b')
            plt.plot(widths[2], 'm')
            plt.title('Widths')
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
                assert model.yr == y
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
                assert model.yr == y
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
    def test_mixed_trap_rect_bed(self):

        models = [KarthausModel, FluxBasedModel, SemiImplicitModel]
        flss = [dummy_constant_bed(), dummy_mixed_trap_rect_bed(),
                dummy_mixed_trap_rect_bed()]

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
                assert model.yr == y
                length[i] = fls[-1].length_m
                vol[i] = fls[-1].volume_km3
            lens.append(length)
            volume.append(vol)
            widths.append(fls[-1].widths_m.copy())
            surface_h.append(fls[-1].surface_h.copy())

        np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=2e-1)
        np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=2e-1)

        # maximum allow difference in length is one grid point
        np.testing.assert_allclose(lens[2][-1], lens[1][-1], atol=1e2)

        np.testing.assert_allclose(volume[2][-1], volume[1][-1], atol=2e-5)
        np.testing.assert_allclose(utils.rmsd(lens[2], lens[1]), 0., atol=23)
        np.testing.assert_allclose(utils.rmsd(volume[2], volume[1]), 0.,
                                   atol=3e-5)
        np.testing.assert_allclose(utils.rmsd(surface_h[2], surface_h[1]), 0.,
                                   atol=6e-2)

        if do_plot:  # pragma: no cover
            plt.plot(lens[0], 'r', label='normal')
            plt.plot(lens[1], 'b', label='mixed flux')
            plt.plot(lens[2], 'm', label='mixed impl')
            plt.title('Length')
            plt.legend()
            plt.show()

            plt.plot(volume[0], 'r', label='normal')
            plt.plot(volume[1], 'b', label='mixed flux')
            plt.plot(volume[2], 'm', label='mixed impl')
            plt.title('Volume')
            plt.legend()
            plt.show()

            plt.plot(fls[-1].bed_h, 'k')
            plt.plot(surface_h[0], 'r', label='normal')
            plt.plot(surface_h[1], 'b', label='mixed flux')
            plt.plot(surface_h[2], 'm', label='mixed impl')
            plt.title('Surface_h')
            plt.legend()
            plt.show()

            plt.plot(widths[0], 'r', label='normal')
            plt.plot(widths[1], 'b', label='mixed flux')
            plt.plot(widths[2], 'm', label='mixed impl')
            plt.title('Widths')
            plt.legend()
            plt.show()

    @pytest.mark.slow
    def test_raise_on_boundary(self):

        fls = dummy_constant_bed()
        mb = LinearMassBalance(2000.)
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a,
                               fs=self.fs)
        with pytest.raises(RuntimeError) as excinfo:
            model.run_until(300)
        assert 'exceeds domain boundaries' in str(excinfo.value)

    def test_raise_cfl(self):

        fls = dummy_constant_bed()
        mb = LinearMassBalance(2000.)
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               cfl_number=1e-7,
                               min_dt=cfg.SEC_IN_YEAR,
                               glen_a=self.glen_a,
                               fs=self.fs)
        with pytest.raises(RuntimeError) as excinfo:
            model.run_until(300)
        assert 'required time step smaller than' in str(excinfo.value)

    def test_mass_redistribution_retreat(self):

        fls = dummy_constant_bed()
        mb = LinearMassBalance(2600.)

        from oggm.core.flowline import FluxBasedModel, MassRedistributionCurveModel

        fl_model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        fl_model.run_until(500)

        mb = LinearMassBalance(2800.)
        fl_model = FluxBasedModel(fl_model.fls, mb_model=mb, y0=0.)
        dh_model_1 = MassRedistributionCurveModel(fl_model.fls, mb_model=mb,
                                                  y0=0, advance_method=1)
        dh_model_2 = MassRedistributionCurveModel(fl_model.fls, mb_model=mb,
                                                  y0=0, advance_method=2)

        # The test fails if the simulation is too long
        yrs = np.arange(1, 500)

        lens = []
        vols = []
        areas = []
        shs = []

        for model in [fl_model, dh_model_1, dh_model_2]:

            length = yrs * 0.
            vol = yrs * 0.
            area = yrs * 0.
            surface_h = []
            for i, y in enumerate(yrs):
                model.run_until(y)
                assert model.yr == y
                length[i] = model.fls[-1].length_m
                vol[i] = model.fls[-1].volume_km3
                area[i] = model.fls[-1].area_m2

                # Just for the plot
                _sh = model.fls[-1].surface_h.copy()
                pp = np.nonzero(model.fls[-1].thick)[0][-1] + 2
                _sh[pp:] = np.nan
                surface_h.append(_sh)

            # We are almost at equilibrium. Spec MB should be close to 0
            assert_allclose(mb.get_specific_mb(fls=model.fls), 0, atol=10)

            lens.append(length)
            vols.append(vol)
            areas.append(area)
            shs.append(surface_h)

        assert_allclose(vols[0], vols[1], rtol=0.1)
        assert_allclose(vols[1], vols[2])

        assert_allclose(lens[0], lens[1], rtol=0.1)
        assert_allclose(lens[1], lens[2])

        if do_plot:
            f, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = np.array(axs).flatten()

            ax = axs[0]
            ax.plot(fls[-1].bed_h, 'k')
            ax.plot(shs[0][0], 'C3')
            ax.plot(shs[1][0], 'C1')
            ax.plot(shs[2][0], 'C0')
            ax.set_title('Start Shape')
            ax.set_xlabel('[m]')
            ax.set_ylabel('Elevation [m]')
            ax.legend(['Bed', 'SIA', 'MassRedis'])

            ax = axs[1]
            ax.plot(fls[-1].bed_h, 'k')
            ax.plot(shs[0][-1], 'C3')
            ax.plot(shs[1][-1], 'C1')
            ax.plot(shs[2][-1], 'C0')
            ax.set_title('End Shape')
            ax.set_xlabel('[m]')
            ax.set_ylabel('Elevation [m]')

            ax = axs[2]
            ax.plot(yrs, vols[0], 'C3')
            ax.plot(yrs, vols[1], 'C1')
            ax.plot(yrs, vols[1], 'C0')
            ax.set_title('Volume')
            ax.set_xlabel('years')
            ax.set_ylabel('[m3]')
            ax.legend(['SIA', 'MassRedis 1', 'MassRedis 2'])

            ax = axs[3]
            ax.plot(yrs, lens[0], 'C3')
            ax.plot(yrs, lens[1], 'C1')
            ax.plot(yrs, lens[1], 'C0')
            ax.set_title('Length')
            ax.set_xlabel('years')
            ax.set_ylabel('[m]')
            ax.legend(['SIA', 'MassRedis 1', 'MassRedis 2'])

            plt.tight_layout()
            plt.show()

    @pytest.mark.slow
    def test_mass_redistribution_grow(self):

        fls = dummy_constant_bed()
        mb = LinearMassBalance(2700.)

        from oggm.core.flowline import FluxBasedModel, MassRedistributionCurveModel

        fl_model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        fl_model.run_until(100)

        fl_model = FluxBasedModel(fl_model.fls, mb_model=mb, y0=0.)
        dh_model_1 = MassRedistributionCurveModel(fl_model.fls, mb_model=mb,
                                                  y0=0, advance_method=1)
        dh_model_2 = MassRedistributionCurveModel(fl_model.fls, mb_model=mb,
                                                  y0=0, advance_method=2)

        # The test fails if the simulation is too long
        yrs = np.arange(1, 800)

        lens = []
        vols = []
        areas = []
        shs = []

        for model in [fl_model, dh_model_1, dh_model_2]:

            length = yrs * 0.
            vol = yrs * 0.
            area = yrs * 0.
            surface_h = []

            for i, y in enumerate(yrs):
                model.run_until(y)
                assert model.yr == y
                length[i] = model.fls[-1].length_m
                vol[i] = model.fls[-1].volume_km3
                area[i] = model.fls[-1].area_m2

                # Just for the plot
                _sh = model.fls[-1].surface_h.copy()
                pp = np.nonzero(model.fls[-1].thick)[0][-1] + 2
                _sh[pp:] = np.nan
                surface_h.append(_sh)

            # We are almost at equilibrium. Spec MB should be close to 0
            # assert_allclose(mb.get_specific_mb(fls=model.fls), 0, atol=10)

            lens.append(length)
            vols.append(vol)
            areas.append(area)
            shs.append(surface_h)

        assert_allclose(vols[0][-1], vols[1][-1], rtol=0.1)
        assert_allclose(vols[1], vols[2])

        assert_allclose(lens[0][-1], lens[1][-1], rtol=0.1)
        assert_allclose(lens[1], lens[2])

        if do_plot:
            f, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = np.array(axs).flatten()

            ax = axs[0]
            ax.plot(fls[-1].bed_h, 'k')
            ax.plot(shs[0][0], 'C3')
            ax.plot(shs[1][0], 'C1')
            ax.plot(shs[2][0], 'C0')
            ax.set_title('Start Shape')
            ax.set_xlabel('[m]')
            ax.set_ylabel('Elevation [m]')
            ax.legend(['Bed', 'SIA', 'MassRedis'])

            ax = axs[1]
            ax.plot(fls[-1].bed_h, 'k')
            ax.plot(shs[0][-1], 'C3')
            ax.plot(shs[1][-1], 'C1')
            ax.plot(shs[2][-1], 'C0')
            ax.set_title('End Shape')
            ax.set_xlabel('[m]')
            ax.set_ylabel('Elevation [m]')

            ax = axs[2]
            ax.plot(yrs, vols[0], 'C3')
            ax.plot(yrs, vols[1], 'C1')
            ax.plot(yrs, vols[2], 'C0')
            ax.set_title('Volume')
            ax.set_xlabel('years')
            ax.set_ylabel('[m3]')
            ax.legend(['SIA', 'MassRedis 1', 'MassRedis 2'])

            ax = axs[3]
            ax.plot(yrs, lens[0], 'C3')
            ax.plot(yrs, lens[1], 'C1')
            ax.plot(yrs, lens[2], 'C0')
            ax.set_title('Length')
            ax.set_xlabel('years')
            ax.set_ylabel('[m]')
            ax.legend(['SIA', 'MassRedis 1', 'MassRedis 2'])

            plt.tight_layout()
            plt.show()

    @pytest.mark.slow
    def test_stop_criterions(self):

        fls = dummy_constant_bed()
        mb = LinearMassBalance(2600.)

        from oggm.core.flowline import FluxBasedModel, MassRedistributionCurveModel
        spinup = FluxBasedModel(fls, mb_model=mb, y0=0.)
        spinup.run_until(500)

        mb = LinearMassBalance(3400.)
        fl_model = FluxBasedModel(spinup.fls, mb_model=mb, y0=0.)
        dh_model = MassRedistributionCurveModel(spinup.fls, mb_model=mb, y0=0)

        from oggm.core.flowline import zero_glacier_stop_criterion
        fl_ds = fl_model.run_until_and_store(1000, stop_criterion=zero_glacier_stop_criterion)
        dh_ds = dh_model.run_until_and_store(1000, stop_criterion=zero_glacier_stop_criterion)

        assert fl_ds.volume_m3.isnull().sum() == 0
        assert dh_ds.volume_m3.isnull().sum() == 0

        fl_ds = fl_ds.volume_m3
        dh_ds = dh_ds.volume_m3

        assert fl_ds.isel(time=-5) == 0
        assert dh_ds.isel(time=-5) == 0
        assert fl_ds.isel(time=-6) > 0
        assert dh_ds.isel(time=-6) > 0

        if do_plot:
            fl_ds.plot(label='Flowline')
            dh_ds.plot(label='MassRedis')
            plt.legend()
            plt.show()

        mb = LinearMassBalance(2800.)
        fl_model = FluxBasedModel(spinup.fls, mb_model=mb, y0=0.)
        dh_model = MassRedistributionCurveModel(spinup.fls, mb_model=mb, y0=0)

        from oggm.core.flowline import spec_mb_stop_criterion
        fl_ds = fl_model.run_until_and_store(1000, stop_criterion=spec_mb_stop_criterion)
        dh_ds = dh_model.run_until_and_store(1000, stop_criterion=spec_mb_stop_criterion)

        fl_ds_ns = fl_model.run_until_and_store(800)
        dh_ds_ns = dh_model.run_until_and_store(800)

        assert fl_ds.volume_m3.isnull().sum() == 0
        assert dh_ds.volume_m3.isnull().sum() == 0

        fl_ds = fl_ds.volume_m3
        dh_ds = dh_ds.volume_m3

        fl_ds_ns = fl_ds_ns.volume_m3
        dh_ds_ns = dh_ds_ns.volume_m3

        assert_allclose(fl_ds.isel(time=-1), fl_ds_ns.isel(time=-1), rtol=0.07)
        assert_allclose(dh_ds.isel(time=-1), dh_ds_ns.isel(time=-1), rtol=0.12)

        if do_plot:
            fl_ds.plot(label='Flowline', color='C0', linewidth=5)
            fl_ds_ns.plot(label='Flowline - cont', color='C0')
            dh_ds.plot(label='MassRedis', color='C1', linewidth=5)
            dh_ds_ns.plot(label='MassRedis - cont', color='C1')
            plt.legend()
            plt.show()


class TestFluxGate(unittest.TestCase):

    def setUp(self):
        cfg.initialize()

    def test_find_flux_from_thickness(self):

        mb = LinearMassBalance(2600.)
        model_flux = FluxBasedModel(dummy_constant_bed(), mb_model=mb)
        model_flux.run_until(700)

        model_impl = SemiImplicitModel(dummy_constant_bed(), mb_model=mb)
        model_impl.run_until(700)

        for model in [model_flux, model_impl]:
            # Pick a flux and slope somewhere in the glacier
            for i in [1, 10, 20, 50]:
                flux = model.flux_stag[0][i]
                slope = model.slope_stag[0][i]
                thick = model.thick_stag[0][i]
                width = model.fls[0].widths_m[i]

                out = find_sia_flux_from_thickness(slope, width, thick)
                assert_allclose(out, flux, atol=1e-7)

    def test_simple_flux_gate(self):

        mb = ScalarMassBalance()
        model = FluxBasedModel(dummy_constant_bed(), mb_model=mb,
                               flux_gate_thickness=150, flux_gate_build_up=50)
        model.run_until(1000)
        assert_allclose(model.volume_m3, model.flux_gate_m3_since_y0)

        model = FluxBasedModel(dummy_mixed_bed(), mb_model=mb,
                               flux_gate_thickness=150, flux_gate_build_up=50)
        model.run_until(1000)
        assert_allclose(model.volume_m3, model.flux_gate_m3_since_y0)
        # Make sure that we cover the types of beds
        beds = np.unique(model.fls[0].shape_str[model.fls[0].thick > 0])
        assert len(beds) == 2

        if do_plot:  # pragma: no cover
            plt.plot(model.fls[-1].bed_h, 'k')
            plt.plot(model.fls[-1].surface_h, 'b')
            plt.show()

    @pytest.mark.slow
    def test_flux_gate_with_trib(self):

        mb = ScalarMassBalance()
        model = FluxBasedModel(dummy_width_bed_tributary(), mb_model=mb,
                               flux_gate_thickness=150, flux_gate_build_up=50)
        model.run_until(1000)
        assert_allclose(model.volume_m3, model.flux_gate_m3_since_y0)

        if do_plot:  # pragma: no cover
            plt.plot(model.fls[-1].bed_h, 'k')
            plt.plot(model.fls[-1].surface_h, 'b')
            plt.show()

    @pytest.mark.slow
    def test_flux_gate_with_trib_tail_to_head(self):

        mb = ScalarMassBalance()
        fls = dummy_bed_tributary_tail_to_head(n_trib=1, small_cliff=True)
        model = FluxBasedModel(fls, mb_model=mb,
                               flux_gate_thickness=[200, 0],
                               flux_gate_build_up=50,
                               smooth_trib_influx=False)
        model.run_until(500)
        assert_allclose(model.volume_m3, model.flux_gate_m3_since_y0)

        if do_plot:  # pragma: no cover
            plt.plot(np.append(model.fls[0].bed_h, model.fls[1].bed_h), 'k')
            plt.plot(np.append(model.fls[0].surface_h,
                               model.fls[1].surface_h), 'b')
            plt.show()

    @pytest.mark.slow
    def test_flux_gate_with_calving(self):

        mb = ScalarMassBalance()
        model = FluxBasedModel(dummy_constant_bed(), mb_model=mb,
                               flux_gate_thickness=150, flux_gate_build_up=50,
                               water_level=2000, do_kcalving=True,
                               is_tidewater=True,
                               )
        model.run_until(2000)
        assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)

        if do_plot:  # pragma: no cover
            plt.plot(model.fls[-1].bed_h, 'k')
            plt.plot(model.fls[-1].surface_h, 'b')
            plt.show()


@pytest.fixture(scope='class')
def default_calving():
    cfg.initialize()
    model = FluxBasedModel(bu_tidewater_bed(),
                           mb_model=ScalarMassBalance(),
                           is_tidewater=True, calving_use_limiter=True,
                           flux_gate=0.06, do_kcalving=True,
                           calving_k=0.2)
    ds = model.run_until_and_store(3000)
    df_diag = model.get_diagnostics()
    assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                    model.flux_gate_m3_since_y0)
    assert_allclose(ds.calving_m3[-1], model.calving_m3_since_y0)
    return model, ds, df_diag


@pytest.mark.usefixtures('default_calving')
class TestKCalving():

    @pytest.mark.slow
    def test_limiter(self, default_calving):

        _, ds1, df_diag1 = default_calving

        model = FluxBasedModel(bu_tidewater_bed(),
                               mb_model=ScalarMassBalance(),
                               is_tidewater=True, calving_use_limiter=False,
                               flux_gate=0.06, do_kcalving=True,
                               calving_k=0.2)
        ds2 = model.run_until_and_store(3000)
        df_diag2 = model.get_diagnostics()
        assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)
        assert_allclose(ds2.calving_m3[-1], model.calving_m3_since_y0)
        assert_allclose(ds2.volume_bsl_m3[-1], model.volume_bsl_km3 * 1e9)
        assert_allclose(ds2.volume_bwl_m3[-1], model.volume_bwl_km3 * 1e9)

        # Not exact same of course
        assert_allclose(ds1.volume_m3[-1], ds2.volume_m3[-1], rtol=0.06)
        assert_allclose(ds1.calving_m3[-1], ds2.calving_m3[-1], rtol=0.15)
        assert_allclose(ds1.volume_bsl_m3[-1], ds2.volume_bsl_m3[-1], rtol=0.3)
        assert_allclose(ds2.volume_bsl_m3, ds2.volume_bwl_m3)

        if do_plot:
            f, ax = plt.subplots(1, 1, figsize=(12, 5))
            df_diag1[['surface_h']].plot(ax=ax, color=['C3'])
            df_diag2[['surface_h', 'bed_h']].plot(ax=ax, color=['C1', 'k'])
            plt.hlines(0, 0, 60000, color='C0', linestyles=':')
            plt.ylim(-350, 800)
            plt.ylabel('Altitude [m]')
            plt.show()

    @pytest.mark.slow
    def test_tributary(self, default_calving):

        _, ds1, df_diag1 = default_calving

        model = FluxBasedModel(bu_tidewater_bed(split_flowline_before_water=5),
                               mb_model=ScalarMassBalance(),
                               is_tidewater=True, calving_use_limiter=True,
                               smooth_trib_influx=False,
                               flux_gate=[0.06, 0], do_kcalving=True,
                               calving_k=0.2)
        ds2 = model.run_until_and_store(3000)
        df_diag2_a = model.get_diagnostics(fl_id=0)
        df_diag2_b = model.get_diagnostics(fl_id=1)

        assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)
        assert_allclose(ds2.calving_m3[-1], model.calving_m3_since_y0)
        assert_allclose(ds2.volume_bsl_m3[-1], model.volume_bsl_km3 * 1e9)
        assert_allclose(ds2.volume_bwl_m3[-1], model.volume_bwl_km3 * 1e9)

        # should be veeery close
        rtol = 5e-4
        assert_allclose(ds1.volume_m3[-1], ds2.volume_m3[-1], rtol=rtol)
        assert_allclose(ds1.calving_m3[-1], ds2.calving_m3[-1], rtol=rtol)
        assert_allclose(ds1.volume_bsl_m3[-1], ds2.volume_bsl_m3[-1],
                        rtol=rtol)
        assert_allclose(ds2.volume_bsl_m3, ds2.volume_bwl_m3, rtol=rtol)

        df_diag1['surface_h_trib'] = np.append(df_diag2_a['surface_h'],
                                               df_diag2_b['surface_h'])

        if do_plot:
            f, ax = plt.subplots(1, 1, figsize=(12, 5))
            df_diag1[['surface_h', 'surface_h_trib',
                      'bed_h']].plot(ax=ax, color=['C3', 'C1', 'k'])
            plt.hlines(0, 0, 60000, color='C0', linestyles=':')
            plt.ylim(-350, 800)
            plt.ylabel('Altitude [m]')
            plt.show()

    @pytest.mark.slow
    def test_water_level(self, default_calving):

        _, ds_1, _ = default_calving

        model = FluxBasedModel(bu_tidewater_bed(water_level=1000),
                               mb_model=ScalarMassBalance(),
                               is_tidewater=True, calving_use_limiter=True,
                               flux_gate=0.06, do_kcalving=True,
                               water_level=1000,
                               calving_k=0.2)
        ds_2 = model.run_until_and_store(3000)
        assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)

        assert_allclose(ds_1.calving_m3, ds_2.calving_m3)

        if do_plot:
            df_diag = model.get_diagnostics()
            f, ax = plt.subplots(1, 1, figsize=(12, 5))
            df_diag[['surface_h', 'bed_h']].plot(ax=ax, color=['C3', 'k'])
            plt.hlines(1000, 0, 60000, color='C0', linestyles=':')
            plt.ylim(1000 - 350, 1000 + 800)
            plt.ylabel('Altitude [m]')
            plt.show()

    @pytest.mark.slow
    def test_other_calving_law(self, default_calving):

        _, ds_1, _ = default_calving

        # We just multiply by 2 inside and divide by 2 outside, should be same
        def my_calving_law(model, flowline, last_above_wl):
            h = flowline.thick[last_above_wl]
            d = h - (flowline.surface_h[last_above_wl] - model.water_level)
            k = model.calving_k
            q_calving = k * d * h * flowline.widths_m[last_above_wl] * 2
            return q_calving

        model = FluxBasedModel(bu_tidewater_bed(),
                               mb_model=ScalarMassBalance(),
                               is_tidewater=True, calving_use_limiter=True,
                               flux_gate=0.06, do_kcalving=True,
                               calving_law=my_calving_law,
                               calving_k=0.2 / 2)
        ds_2 = model.run_until_and_store(3000)
        assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)
        assert_allclose(ds_1.calving_m3, ds_2.calving_m3)


class TestSia2d(unittest.TestCase):

    def setUp(self):
        cfg.initialize()

    def tearDown(self):
        pass

    @pytest.mark.slow
    def test_flat_2d_bed(self):

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
            assert flmodel.yr == y
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
            assert sdmodel.yr == y
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

        assert utils.rmsd(lens[0], lens[1]) < 50.
        assert utils.rmsd(volume[0], volume[1]) < 2e-3
        assert utils.rmsd(areas[0], areas[1]) < 2e-3
        assert utils.rmsd(surface_h[0], surface_h[1]) < 1.0

        # Equilibrium
        sdmodel.run_until_equilibrium()
        flmodel.run_until_equilibrium()
        assert_allclose(sdmodel.volume_km3 / 3, flmodel.volume_km3, atol=2e-3)
        assert_allclose(sdmodel.area_km2 / 3, flmodel.area_km2, atol=2e-3)

        # Store
        run_ds = sdmodel.run_until_and_store(sdmodel.yr + 50)
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
            assert sdmodel.yr == y
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

        assert utils.rmsd(lens[0], lens[1]) < 50.
        assert utils.rmsd(volume[0], volume[1]) < 2e-3
        assert utils.rmsd(areas[0], areas[1]) < 2e-3
        assert utils.rmsd(surface_h[0], surface_h[1]) < 1.0

        # Equilibrium
        sdmodel.run_until_equilibrium()
        assert_allclose(sdmodel.volume_km3 / 3, flmodel.volume_km3, atol=2e-3)
        assert_allclose(sdmodel.area_km2 / 3, flmodel.area_km2, atol=2e-3)

    def test_bueler(self):
        # TODO: add formal test like Alex's
        # https://github.com/alexjarosch/sia-fluxlim
        pass
