import pytest
pytestmark = pytest.mark.test_env("minimal")

import unittest
from functools import partial

import numpy as np
from numpy.testing import assert_allclose

# Local imports
from oggm.core.massbalance import LinearMassBalance
from oggm import cfg, utils
from oggm.utils import rmsd
from oggm.cfg import SEC_IN_DAY
from oggm.core.sia2d import Upstream2D

# Tests
from oggm.tests.funcs import (dummy_constant_bed,
                              patch_minimal_download_oggm_files)
from oggm.core.flowline import KarthausModel, FluxBasedModel
from oggm.tests.ext.sia_fluxlim import MUSCLSuperBeeModel

FluxBasedModel = partial(FluxBasedModel, inplace=True)
KarthausModel = partial(KarthausModel, inplace=True)
MUSCLSuperBeeModel = partial(MUSCLSuperBeeModel, inplace=True)

_patched_download_oggm_files = None


def setup_module(module):
    module._patched_download_oggm_files = utils.download_oggm_files
    utils._downloads.download_oggm_files = patch_minimal_download_oggm_files


def teardown_module(module):
    utils._downloads.download_oggm_files = module._patched_download_oggm_files


class TestIdealisedCases(unittest.TestCase):

    def setUp(self):
        N = 3
        cfg.initialize_minimal()
        self.glen_a = 2.4e-24    # Modern style Glen parameter A
        self.aglen_old = (N + 2) * 1.9e-24 / 2.  # outdated value
        self.fd = 2. * self.glen_a / (N + 2.)  # equivalent to glen_a
        self.fs = 0             # set slidin
        self.fs_old = 5.7e-20  # outdated value

    def tearDown(self):
        pass

    def test_constant_bed(self):

        models = [FluxBasedModel, MUSCLSuperBeeModel]

        lens = []
        surface_h = []
        volume = []
        yrs = np.arange(1, 300, 2)
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

        assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        assert_allclose(volume[0][-1], volume[1][-1], atol=3e-3)

        assert rmsd(lens[0], lens[1]) < 50.
        assert rmsd(volume[0], volume[1]) < 2e-3
        assert rmsd(surface_h[0], surface_h[1]) < 1.2

    def test_run_until_and_store(self):

        fls = dummy_constant_bed()
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb)
        ds_d = model.run_until_and_store(150)
        assert ds_d['length_m'][-1] > 1e3

        df = model.get_diagnostics()
        assert (df['ice_velocity'].max() * cfg.SEC_IN_YEAR) > 10


class TestSia2d(unittest.TestCase):

    def setUp(self):
        cfg.initialize_minimal()

    def tearDown(self):
        pass

    def test_constant_bed(self):

        map_dx = 100.
        yrs = np.arange(1, 200, 5)
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

        assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        assert_allclose(volume[0][-1], volume[1][-1], atol=3e-3)

        assert rmsd(lens[0], lens[1]) < 50.
        assert rmsd(volume[0], volume[1]) < 2e-3
        assert rmsd(areas[0], areas[1]) < 2e-3
        assert rmsd(surface_h[0], surface_h[1]) < 1.2

        # Store
        run_ds = sdmodel.run_until_and_store(sdmodel.yr+50)
        assert 'ice_thickness' in run_ds

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

        assert_allclose(lens[0][-1], lens[1][-1], atol=101)
        assert_allclose(volume[0][-1], volume[1][-1], atol=3e-3)

        assert rmsd(lens[0], lens[1]) < 50.
        assert rmsd(volume[0], volume[1]) < 2e-3
        assert rmsd(areas[0], areas[1]) < 2e-3
        assert rmsd(surface_h[0], surface_h[1]) < 1.2
