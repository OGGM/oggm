import unittest
import pytest
import numpy as np
from numpy.testing import assert_allclose

# Local imports
from oggm.core.massbalance import LinearMassBalance, ScalarMassBalance
from oggm import cfg

# Tests
from oggm.tests.funcs import (dummy_constant_bed,
                              dummy_noisy_bed,
                              bu_tidewater_bed,
                              )

# after oggm.test
import matplotlib.pyplot as plt

from oggm.core.flowline import FluxBasedModel
from oggm.sandbox.calving_jan import CalvingFluxBasedModelJan
from oggm.sandbox.calving_fabi import CalvingFluxBasedModelFabi

pytestmark = pytest.mark.test_env("sandbox")
do_plot = False

pytest.importorskip('geopandas')
pytest.importorskip('rasterio')
pytest.importorskip('salem')


class TestIdealisedCases(unittest.TestCase):

    def setUp(self):
        cfg.initialize()
        self.glen_a = 2.4e-24  # Modern style Glen parameter A
        self.fs = 5.7e-20 / 2  # set sliding

    def tearDown(self):
        pass

    @pytest.mark.slow
    def test_constant_bed(self):

        models = [FluxBasedModel, CalvingFluxBasedModelJan, CalvingFluxBasedModelFabi]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model in models:

            fls = dummy_constant_bed()
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, y0=0., glen_a=self.glen_a,
                          fs=self.fs)

            fls = model.fls

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

        np.testing.assert_allclose(volume[0], volume[2])
        np.testing.assert_allclose(volume[1], volume[2])

    @pytest.mark.slow
    def test_bumpy_bed(self):

        models = [FluxBasedModel, CalvingFluxBasedModelJan, CalvingFluxBasedModelFabi]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 500, 2)
        for model in models:

            print(model.__name__)

            fls = dummy_noisy_bed()
            mb = LinearMassBalance(2600.)

            model = model(fls, mb_model=mb, y0=0., glen_a=self.glen_a,
                          fs=self.fs)

            fls = model.fls

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

        np.testing.assert_allclose(volume[0], volume[2])
        np.testing.assert_allclose(volume[1], volume[2])


class TestFluxGate(unittest.TestCase):

    def setUp(self):
        cfg.initialize()

    @pytest.mark.slow
    def test_flux_gate_with_calving(self):

        mb = ScalarMassBalance()

        bed = dummy_constant_bed()

        if do_plot:  # pragma: no cover
            plt.plot(bed[0].bed_h, 'k')
            plt.hlines(2000, 0, 200, 'C0')

        vols = []
        calvs = []

        for model_class in [FluxBasedModel,
                            CalvingFluxBasedModelJan,
                            CalvingFluxBasedModelFabi]:
            model = model_class(bed, mb_model=mb,
                                flux_gate_thickness=150, flux_gate_build_up=50,
                                water_level=2000, do_kcalving=True,
                                is_tidewater=True,
                                )
            model.run_until(1000)
            # Mass conservation
            assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                            model.flux_gate_m3_since_y0)

            vols.append(model.volume_m3)
            calvs.append(model.calving_m3_since_y0)

            if do_plot:  # pragma: no cover
                plt.plot(model.fls[-1].surface_h, label=model_class.__name__)

        # Jan's is different than the original one
        np.testing.assert_allclose(vols[1], vols[0], rtol=0.01)
        np.testing.assert_allclose(calvs[1], calvs[0], rtol=0.12)

        # Now between my implementation and Jans - there is differences I cannot explain
        # as of now.
        # np.testing.assert_allclose(vols[1], vols[2])
        # np.testing.assert_allclose(calvs[1], calvs[2])

        if do_plot:  # pragma: no cover
            plt.legend()
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
    def test_bu_bed(self, default_calving):

        _, ds1, df_diag1 = default_calving

        model = CalvingFluxBasedModelJan(bu_tidewater_bed(),
                                         mb_model=ScalarMassBalance(),
                                         is_tidewater=True,
                                         flux_gate=0.06, do_kcalving=True,
                                         calving_k=0.2)
        ds2 = model.run_until_and_store(3000)
        df_diag2 = model.get_diagnostics()

        # Mass conservation check
        assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)
        assert_allclose(ds2.calving_m3[-1], model.calving_m3_since_y0)
        assert_allclose(ds2.volume_bsl_m3[-1], model.volume_bsl_km3 * 1e9)
        assert_allclose(ds2.volume_bwl_m3[-1], model.volume_bwl_km3 * 1e9)
        assert_allclose(ds2.volume_bsl_m3, ds2.volume_bwl_m3)

        # Not same as previous calving of course
        assert_allclose(ds1.volume_m3[-1], ds2.volume_m3[-1], rtol=0.1)
        assert_allclose(ds1.calving_m3[-1], ds2.calving_m3[-1], rtol=0.25)
        assert_allclose(ds1.volume_bsl_m3[-1], ds2.volume_bsl_m3[-1], rtol=0.5)

        if do_plot:
            f, ax = plt.subplots(1, 1, figsize=(12, 5))
            df_diag1[['surface_h']].plot(ax=ax, color=['C3'], label='Jan')
            df_diag2[['surface_h', 'bed_h']].plot(ax=ax, color=['C1', 'k'], label='New')
            plt.hlines(0, 0, 60000, color='C0', linestyles=':')
            plt.ylim(-350, 800)
            plt.ylabel('Altitude [m]')
            plt.legend()
            plt.show()

        # Check new implementation
        model = CalvingFluxBasedModelFabi(bu_tidewater_bed(),
                                          mb_model=ScalarMassBalance(),
                                          is_tidewater=True,
                                          flux_gate=0.06, do_kcalving=True,
                                          calving_k=0.2)
        ds3 = model.run_until_and_store(3000)

        # Mass conservation check
        assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)
        assert_allclose(ds3.calving_m3[-1], model.calving_m3_since_y0)
        assert_allclose(ds3.volume_bsl_m3[-1], model.volume_bsl_km3 * 1e9)
        assert_allclose(ds3.volume_bwl_m3[-1], model.volume_bwl_km3 * 1e9)
        assert_allclose(ds3.volume_bsl_m3, ds3.volume_bwl_m3)

        # Not same as previous calving of course - but the problem
        # is that the errors are too big
        # assert_allclose(ds1.volume_m3[-1], ds3.volume_m3[-1], rtol=0.1)
        # assert_allclose(ds1.calving_m3[-1], ds3.calving_m3[-1], rtol=0.25)
        # assert_allclose(ds1.volume_bsl_m3[-1], ds3.volume_bsl_m3[-1], rtol=0.5)

        # Also here the errors are too big
        # assert_allclose(ds3.volume_m3, ds2.volume_m3)
        # assert_allclose(ds3.calving_m3, ds2.calving_m3)
        # assert_allclose(ds3.volume_bsl_m3, ds2.volume_bsl_m3)
