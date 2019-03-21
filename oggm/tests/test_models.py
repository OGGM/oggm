import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import os
from functools import partial
import shutil
import unittest
import copy
import time
import numpy as np
import pandas as pd
import shapely.geometry as shpg
from numpy.testing import assert_allclose
import pytest

# Local imports
import oggm
from oggm.core import massbalance
from oggm.core.massbalance import LinearMassBalance
import xarray as xr
from oggm import utils, workflow, tasks, cfg
from oggm.core import gcm_climate, climate, inversion, centerlines
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import get_demo_file

from oggm.tests.funcs import init_hef, get_test_dir, patch_url_retrieve_github
from oggm.tests.funcs import (dummy_bumpy_bed, dummy_constant_bed,
                              dummy_constant_bed_cliff,
                              dummy_mixed_bed,
                              dummy_noisy_bed, dummy_parabolic_bed,
                              dummy_trapezoidal_bed, dummy_width_bed,
                              dummy_width_bed_tributary)

import matplotlib.pyplot as plt
from oggm.core.flowline import (FluxBasedModel, FlowlineModel,
                                init_present_time_glacier, glacier_from_netcdf,
                                RectangularBedFlowline, TrapezoidalBedFlowline,
                                ParabolicBedFlowline, MixedBedFlowline,
                                flowline_from_dataset, FileModel,
                                run_constant_climate, run_random_climate,
                                run_from_climate_data)

FluxBasedModel = partial(FluxBasedModel, inplace=True)
FlowlineModel = partial(FlowlineModel, inplace=True)
_url_retrieve = None


def setup_module(module):
    module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github


def teardown_module(module):
    oggm.utils._downloads.oggm_urlretrieve = module._url_retrieve


pytestmark = pytest.mark.test_env("models")
do_plot = False

DOM_BORDER = 80


class TestInitFlowline(unittest.TestCase):

    def setUp(self):
        gdir = init_hef(border=DOM_BORDER)
        self.testdir = os.path.join(get_test_dir(), type(self).__name__)
        utils.mkdir(self.testdir, reset=True)
        self.gdir = tasks.copy_to_basedir(gdir, base_dir=self.testdir,
                                          setup='all')

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    def test_init_present_time_glacier(self):

        gdir = self.gdir
        init_present_time_glacier(gdir)

        fls = gdir.read_pickle('model_flowlines')

        ofl = gdir.read_pickle('inversion_flowlines')[-1]

        self.assertTrue(gdir.rgi_date == 2003)
        self.assertTrue(len(fls) == 3)

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

            self.assertTrue(np.all(fl.widths >= 0))
            vol += fl.volume_km3
            area += fl.area_km2

            if refo == 1:
                rmsd = utils.rmsd(ofl.widths[:-5] * gdir.grid.dx,
                                  fl.widths_m[0:len(ofl.widths)-5])
                self.assertTrue(rmsd < 5.)

        rtol = 0.02
        np.testing.assert_allclose(0.573, vol, rtol=rtol)
        np.testing.assert_allclose(6900.0, fls[-1].length_m, atol=101)
        np.testing.assert_allclose(gdir.rgi_area_km2, area, rtol=rtol)

        if do_plot:
            plt.plot(fls[-1].bed_h)
            plt.plot(fls[-1].surface_h)
            plt.show()

    def test_present_time_glacier_massbalance(self):

        gdir = self.gdir
        init_present_time_glacier(gdir)

        mb_mod = massbalance.PastMassBalance(gdir)

        fls = gdir.read_pickle('model_flowlines')
        glacier = FlowlineModel(fls)

        mbdf = gdir.get_ref_mb_data()

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
            mbh = (mb_mod.get_annual_mb(hgts, yr) * SEC_IN_YEAR *
                   cfg.PARAMS['ice_density'])
            grads += mbh
            tot_mb.append(np.average(mbh, weights=widths))
        grads /= len(tot_mb)

        # Bias
        self.assertTrue(np.abs(utils.md(tot_mb, refmb)) < 50)

        # Gradient
        dfg = gdir.get_ref_mb_profile().mean()

        # Take the altitudes below 3100 and fit a line
        dfg = dfg[dfg.index < 3100]
        pok = np.where(hgts < 3100)
        from scipy.stats import linregress
        slope_obs, _, _, _, _ = linregress(dfg.index, dfg.values)
        slope_our, _, _, _, _ = linregress(hgts[pok], grads[pok])
        np.testing.assert_allclose(slope_obs, slope_our, rtol=0.15)


class TestOtherGlacier(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_div')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        # self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_define_divides(self):

        from oggm.core import centerlines
        from oggm.core import climate
        from oggm.core import inversion
        from oggm.core import gis
        from oggm import GlacierDirectory
        import geopandas as gpd

        hef_file = utils.get_demo_file('rgi_oetztal.shp')
        rgidf = gpd.read_file(hef_file)

        # This is another glacier with divides
        entity = rgidf.loc[rgidf.RGIId == 'RGI50-11.00719_d01'].iloc[0]
        gdir = GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        cfg.PARAMS['baseline_climate'] = ''
        climate.process_custom_climate_data(gdir)
        climate.local_t_star(gdir, tstar=1930, bias=0)
        climate.mu_star_calibration(gdir)
        inversion.prepare_for_inversion(gdir)
        v, ainv = inversion.mass_conservation_inversion(gdir)
        init_present_time_glacier(gdir)

        myarea = 0.
        cls = gdir.read_pickle('inversion_flowlines')
        for cl in cls:
            myarea += np.sum(cl.widths * cl.dx * gdir.grid.dx**2)

        np.testing.assert_allclose(ainv, gdir.rgi_area_m2, rtol=1e-2)
        np.testing.assert_allclose(myarea, gdir.rgi_area_m2, rtol=1e-2)

        myarea = 0.
        cls = gdir.read_pickle('inversion_flowlines')
        for cl in cls:
            myarea += np.sum(cl.widths * cl.dx * gdir.grid.dx**2)

        np.testing.assert_allclose(myarea, gdir.rgi_area_m2, rtol=1e-2)

        fls = gdir.read_pickle('model_flowlines')
        if cfg.PARAMS['grid_dx_method'] == 'square':
            self.assertEqual(len(fls), 3)
        vol = 0.
        area = 0.
        for fl in fls:
            ref = np.arange(len(fl.surface_h)) * fl.dx
            np.testing.assert_allclose(ref, fl.dis_on_line,
                                       rtol=0.001,
                                       atol=0.01)
            self.assertTrue(len(fl.surface_h) ==
                            len(fl.bed_h) ==
                            len(fl.bed_shape) ==
                            len(fl.dis_on_line) ==
                            len(fl.widths))

            self.assertTrue(np.all(fl.widths >= 0))
            vol += fl.volume_km3
            area += fl.area_km2

        rtol = 0.08
        np.testing.assert_allclose(gdir.rgi_area_km2, area, rtol=rtol)
        np.testing.assert_allclose(v*1e-9, vol, rtol=rtol)


class TestMassBalance(unittest.TestCase):

    def setUp(self):
        gdir = init_hef(border=DOM_BORDER)
        self.testdir = os.path.join(get_test_dir(), type(self).__name__)
        utils.mkdir(self.testdir, reset=True)
        self.gdir = tasks.copy_to_basedir(gdir, base_dir=self.testdir,
                                          setup='all')

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    def test_past_mb_model(self):

        rho = cfg.PARAMS['ice_density']

        F = SEC_IN_YEAR * rho

        gdir = self.gdir
        init_present_time_glacier(gdir)

        df = gdir.read_json('local_mustar')
        mu_star = df['mu_star_glacierwide']
        bias = df['bias']

        # Climate period
        yrp = [1851, 2000]

        # Flowlines height
        h, w = gdir.get_inversion_flowline_hw()
        _, t, p = climate.mb_yearly_climate_on_height(gdir, h,
                                                      year_range=yrp)

        mb_mod = massbalance.PastMassBalance(gdir, bias=0)
        for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):
            ref_mb_on_h = p[:, i] - mu_star * t[:, i]
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * F
            np.testing.assert_allclose(ref_mb_on_h, my_mb_on_h,
                                       atol=1e-2)
            ela_z = mb_mod.get_ela(year=yr)
            totest = mb_mod.get_annual_mb([ela_z], year=yr) * F
            assert_allclose(totest[0], 0, atol=1)

        mb_mod = massbalance.PastMassBalance(gdir)
        for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):
            ref_mb_on_h = p[:, i] - mu_star * t[:, i]
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * F
            np.testing.assert_allclose(ref_mb_on_h, my_mb_on_h + bias,
                                       atol=1e-2)
            ela_z = mb_mod.get_ela(year=yr)
            totest = mb_mod.get_annual_mb([ela_z], year=yr) * F
            assert_allclose(totest[0], 0, atol=1)

        for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):

            ref_mb_on_h = p[:, i] - mu_star * t[:, i]
            my_mb_on_h = ref_mb_on_h*0.
            for m in np.arange(12):
                yrm = utils.date_to_floatyear(yr, m + 1)
                tmp = mb_mod.get_monthly_mb(h, yrm) * SEC_IN_MONTH * rho
                my_mb_on_h += tmp

            np.testing.assert_allclose(ref_mb_on_h,
                                       my_mb_on_h + bias,
                                       atol=1e-2)

        # real data
        h, w = gdir.get_inversion_flowline_hw()
        mbdf = gdir.get_ref_mb_data()
        mbdf.loc[yr, 'MY_MB'] = np.NaN
        mb_mod = massbalance.PastMassBalance(gdir)
        for yr in mbdf.index.values:
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
            mbdf.loc[yr, 'MY_MB'] = np.average(my_mb_on_h, weights=w)

        np.testing.assert_allclose(mbdf['ANNUAL_BALANCE'].mean(),
                                   mbdf['MY_MB'].mean(),
                                   atol=1e-2)
        mbdf['MY_ELA'] = mb_mod.get_ela(year=mbdf.index.values)
        assert mbdf[['MY_ELA', 'MY_MB']].corr().values[0, 1] < -0.9
        assert mbdf[['MY_ELA', 'ANNUAL_BALANCE']].corr().values[0, 1] < -0.7

        mb_mod = massbalance.PastMassBalance(gdir, bias=0)
        for yr in mbdf.index.values:
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
            mbdf.loc[yr, 'MY_MB'] = np.average(my_mb_on_h, weights=w)

        np.testing.assert_allclose(mbdf['ANNUAL_BALANCE'].mean() + bias,
                                   mbdf['MY_MB'].mean(),
                                   atol=1e-2)

        mb_mod = massbalance.PastMassBalance(gdir)
        for yr in mbdf.index.values:
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
            mbdf.loc[yr, 'MY_MB'] = np.average(my_mb_on_h, weights=w)
            mb_mod.temp_bias = 1
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
            mbdf.loc[yr, 'BIASED_MB'] = np.average(my_mb_on_h, weights=w)
            mb_mod.temp_bias = 0

        np.testing.assert_allclose(mbdf['ANNUAL_BALANCE'].mean(),
                                   mbdf['MY_MB'].mean(),
                                   atol=1e-2)
        self.assertTrue(mbdf.ANNUAL_BALANCE.mean() > mbdf.BIASED_MB.mean())

        # Repeat
        mb_mod = massbalance.PastMassBalance(gdir, repeat=True,
                                             ys=1901, ye=1950)
        yrs = np.arange(100) + 1901
        mb = mb_mod.get_specific_mb(h, w, year=yrs)
        assert_allclose(mb[50], mb[-50])

        # Go for glacier wide now
        fls = gdir.read_pickle('inversion_flowlines')
        mb_gw_mod = massbalance.MultipleFlowlineMassBalance(gdir, fls=fls,
                                                            repeat=True,
                                                            ys=1901, ye=1950)
        mb_gw = mb_gw_mod.get_specific_mb(year=yrs)
        assert_allclose(mb, mb_gw)

    def test_glacierwide_mb_model(self):

        gdir = self.gdir
        init_present_time_glacier(gdir)

        fls = gdir.read_pickle('model_flowlines')
        h = np.array([])
        w = np.array([])
        for fl in fls:
            w = np.append(w, fl.widths)
            h = np.append(h, fl.surface_h)

        yrs = np.arange(100) + 1901

        classes = [massbalance.PastMassBalance,
                   massbalance.ConstantMassBalance,
                   massbalance.RandomMassBalance]

        for cl in classes:

            if cl is massbalance.RandomMassBalance:
                kwargs = {'seed': 0}
            else:
                kwargs = {}

            mb = cl(gdir, **kwargs)
            mb_gw = massbalance.MultipleFlowlineMassBalance(gdir, fls=fls,
                                                            mb_model_class=cl,
                                                            **kwargs)

            assert_allclose(mb.get_specific_mb(h, w, year=yrs),
                            mb_gw.get_specific_mb(year=yrs))

            assert_allclose(mb.get_ela(year=yrs),
                            mb_gw.get_ela(year=yrs))

            _h, _w, mbs_gw = mb_gw.get_annual_mb_on_flowlines(year=1950)
            mbs_h = mb.get_annual_mb(_h, year=1950)

            assert_allclose(mbs_h, mbs_gw)

            mb.bias = 100
            mb_gw.bias = 100

            assert_allclose(mb.get_specific_mb(h, w, year=yrs[:10]),
                            mb_gw.get_specific_mb(year=yrs[:10]))

            assert_allclose(mb.get_ela(year=yrs[:10]),
                            mb_gw.get_ela(year=yrs[:10]))

            mb.temp_bias = 100
            mb_gw.temp_bias = 100

            assert mb.temp_bias == mb_gw.temp_bias

            assert_allclose(mb.get_specific_mb(h, w, year=yrs[:10]),
                            mb_gw.get_specific_mb(year=yrs[:10]))

            assert_allclose(mb.get_ela(year=yrs[:10]),
                            mb_gw.get_ela(year=yrs[:10]))

            mb.prcp_bias = 100
            mb_gw.prcp_bias = 100

            assert mb.prcp_bias == mb_gw.prcp_bias

            assert_allclose(mb.get_specific_mb(h, w, year=yrs[:10]),
                            mb_gw.get_specific_mb(year=yrs[:10]))

            assert_allclose(mb.get_ela(year=yrs[:10]),
                            mb_gw.get_ela(year=yrs[:10]))

        cl = massbalance.PastMassBalance
        mb = cl(gdir)
        mb_gw = massbalance.MultipleFlowlineMassBalance(gdir,
                                                        mb_model_class=cl)
        mb = massbalance.UncertainMassBalance(mb, rdn_bias_seed=1,
                                              rdn_prcp_bias_seed=2,
                                              rdn_temp_bias_seed=3)
        mb_gw = massbalance.UncertainMassBalance(mb_gw, rdn_bias_seed=1,
                                                 rdn_prcp_bias_seed=2,
                                                 rdn_temp_bias_seed=3)

        assert_allclose(mb.get_specific_mb(h, w, year=yrs[:30]),
                        mb_gw.get_specific_mb(fls=fls, year=yrs[:30]))

        # ELA won't pass because of API incompatibility
        # assert_allclose(mb.get_ela(year=yrs[:30]),
        #                 mb_gw.get_ela(year=yrs[:30]))

    def test_constant_mb_model(self):

        rho = cfg.PARAMS['ice_density']

        gdir = self.gdir
        init_present_time_glacier(gdir)

        df = gdir.read_json('local_mustar')
        bias = df['bias']

        h, w = gdir.get_inversion_flowline_hw()

        cmb_mod = massbalance.ConstantMassBalance(gdir, bias=0)
        ombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        otmb = np.average(ombh, weights=w)
        np.testing.assert_allclose(0., otmb, atol=0.2)

        cmb_mod = massbalance.ConstantMassBalance(gdir)
        ombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        otmb = np.average(ombh, weights=w)
        np.testing.assert_allclose(0, otmb + bias, atol=0.2)

        mb_mod = massbalance.ConstantMassBalance(gdir, y0=2003 - 15)
        nmbh = mb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        ntmb = np.average(nmbh, weights=w)

        self.assertTrue(ntmb < otmb)

        if do_plot:  # pragma: no cover
            plt.plot(h, ombh, 'o', label='tstar')
            plt.plot(h, nmbh, 'o', label='today')
            plt.legend()
            plt.show()

        cmb_mod.temp_bias = 1
        biasombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        biasotmb = np.average(biasombh, weights=w)
        self.assertTrue(biasotmb < (otmb - 500))

        cmb_mod.temp_bias = 0
        nobiasombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        nobiasotmb = np.average(nobiasombh, weights=w)
        np.testing.assert_allclose(0, nobiasotmb + bias, atol=0.2)

        months = np.arange(12)
        monthly_1 = months * 0.
        monthly_2 = months * 0.
        for m in months:
            yr = utils.date_to_floatyear(0, m + 1)
            cmb_mod.temp_bias = 0
            tmp = cmb_mod.get_monthly_mb(h, yr) * SEC_IN_MONTH * rho
            monthly_1[m] = np.average(tmp, weights=w)
            cmb_mod.temp_bias = 1
            tmp = cmb_mod.get_monthly_mb(h, yr) * SEC_IN_MONTH * rho
            monthly_2[m] = np.average(tmp, weights=w)

        # check that the winter months are close but summer months no
        np.testing.assert_allclose(monthly_1[1: 5], monthly_2[1: 5], atol=1)
        self.assertTrue(np.mean(monthly_1[5:]) >
                        (np.mean(monthly_2[5:]) + 100))

        if do_plot:  # pragma: no cover
            plt.plot(monthly_1, '-', label='Normal')
            plt.plot(monthly_2, '-', label='Temp bias')
            plt.legend()
            plt.show()

        # Climate info
        h = np.sort(h)
        cmb_mod = massbalance.ConstantMassBalance(gdir, bias=0)
        t, tm, p, ps = cmb_mod.get_climate(h)

        # Simple sanity checks
        assert np.all(np.diff(t) <= 0)
        assert np.all(np.diff(tm) <= 0)
        assert np.all(np.diff(p) == 0)
        assert np.all(np.diff(ps) >= 0)

        if do_plot:  # pragma: no cover
            f, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs = axs.flatten()
            axs[0].plot(h, t, label='Temp')
            axs[0].legend()
            axs[1].plot(h, tm, label='TempMelt')
            axs[1].legend()
            axs[2].plot(h, p, label='Prcp')
            axs[2].plot(h, ps, label='SolidPrcp')
            axs[2].legend()
            plt.tight_layout()
            plt.show()

        # ELA
        elah = cmb_mod.get_ela()
        t, tm, p, ps = cmb_mod.get_climate([elah])
        mb = ps - cmb_mod.mbmod.mu_star * tm
        # not perfect because of time/months/zinterp issues
        np.testing.assert_allclose(mb, 0, atol=0.12)

    def test_random_mb(self):

        gdir = self.gdir
        init_present_time_glacier(gdir)

        ref_mod = massbalance.ConstantMassBalance(gdir)
        mb_mod = massbalance.RandomMassBalance(gdir, seed=10)

        h, w = gdir.get_inversion_flowline_hw()

        ref_mbh = ref_mod.get_annual_mb(h, None) * SEC_IN_YEAR

        # two years shoudn't be equal
        r_mbh1 = mb_mod.get_annual_mb(h, 1) * SEC_IN_YEAR
        r_mbh2 = mb_mod.get_annual_mb(h, 2) * SEC_IN_YEAR
        assert not np.all(np.allclose(r_mbh1, r_mbh2))

        # the same year should be equal
        r_mbh1 = mb_mod.get_annual_mb(h, 1) * SEC_IN_YEAR
        r_mbh2 = mb_mod.get_annual_mb(h, 1) * SEC_IN_YEAR
        np.testing.assert_allclose(r_mbh1, r_mbh2)

        # After many trials the mb should be close to the same
        ny = 2000
        yrs = np.arange(ny)
        r_mbh = 0.
        mbts = yrs * 0.
        for i, yr in enumerate(yrs):
            mbts[i] = mb_mod.get_specific_mb(h, w, year=yr)
            r_mbh += mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR
        r_mbh /= ny
        np.testing.assert_allclose(ref_mbh, r_mbh, atol=0.2)
        elats = mb_mod.get_ela(yrs[:200])
        assert np.corrcoef(mbts[:200], elats)[0, 1] < -0.95

        mb_mod.temp_bias = -0.5
        r_mbh_b = 0.
        for yr in yrs:
            r_mbh_b += mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR
        r_mbh_b /= ny
        self.assertTrue(np.mean(r_mbh) < np.mean(r_mbh_b))

        # Compare sigma from real climate and mine
        mb_ref = massbalance.PastMassBalance(gdir)
        mb_mod = massbalance.RandomMassBalance(gdir, y0=2003 - 15,
                                               seed=10)
        mb_ts = []
        mb_ts2 = []
        yrs = np.arange(1973, 2004, 1)
        for yr in yrs:
            mb_ts.append(np.average(mb_ref.get_annual_mb(h, yr) * SEC_IN_YEAR,
                                    weights=w))
            mb_ts2.append(np.average(mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR,
                                     weights=w))
        np.testing.assert_allclose(np.std(mb_ts), np.std(mb_ts2), rtol=0.1)

        # Monthly
        time = pd.date_range('1/1/1973', periods=31*12, freq='MS')
        yrs = utils.date_to_floatyear(time.year, time.month)

        ref_mb = np.zeros(12)
        my_mb = np.zeros(12)
        for yr, m in zip(yrs, time.month):
            ref_mb[m-1] += np.average(mb_ref.get_monthly_mb(h, yr) *
                                      SEC_IN_MONTH, weights=w)
            my_mb[m-1] += np.average(mb_mod.get_monthly_mb(h, yr) *
                                     SEC_IN_MONTH, weights=w)
        my_mb = my_mb / 31
        ref_mb = ref_mb / 31
        self.assertTrue(utils.rmsd(ref_mb, my_mb) < 0.1)

    def test_random_mb_unique(self):

        gdir = self.gdir
        init_present_time_glacier(gdir)

        ref_mod = massbalance.ConstantMassBalance(gdir,
                                                  halfsize=15)
        mb_mod = massbalance.RandomMassBalance(gdir, seed=10,
                                               unique_samples=True,
                                               halfsize=15)
        mb_mod2 = massbalance.RandomMassBalance(gdir, seed=20,
                                                unique_samples=True,
                                                halfsize=15)
        mb_mod3 = massbalance.RandomMassBalance(gdir, seed=20,
                                                unique_samples=True,
                                                halfsize=15)

        h, w = gdir.get_inversion_flowline_hw()

        ref_mbh = ref_mod.get_annual_mb(h, None) * SEC_IN_YEAR

        # the same year should be equal
        r_mbh1 = mb_mod.get_annual_mb(h, 1) * SEC_IN_YEAR
        r_mbh2 = mb_mod.get_annual_mb(h, 1) * SEC_IN_YEAR
        np.testing.assert_allclose(r_mbh1, r_mbh2)

        # test 31 years (2*halfsize +1)
        ny = 31
        yrs = np.arange(ny)
        mbts = yrs * 0.
        r_mbh = 0.
        r_mbh2 = 0.
        r_mbh3 = 0.
        mb_mod3.temp_bias = -0.5
        annual_previous = -999.
        for i, yr in enumerate(yrs):
            # specific mass balance
            mbts[i] = mb_mod.get_specific_mb(h, w, year=yr)

            # annual mass balance
            annual = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR

            # annual mass balance must be different than the previous one
            assert not np.all(np.allclose(annual, annual_previous))

            # sum over all years should be equal to ref_mbh
            r_mbh += annual
            r_mbh2 += mb_mod2.get_annual_mb(h, yr) * SEC_IN_YEAR

            # mass balance with temperature bias
            r_mbh3 += mb_mod3.get_annual_mb(h, yr) * SEC_IN_YEAR

            annual_previous = annual

        r_mbh /= ny
        r_mbh2 /= ny
        r_mbh3 /= ny

        # test sums
        np.testing.assert_allclose(ref_mbh, r_mbh, atol=0.02)
        np.testing.assert_allclose(r_mbh, r_mbh2, atol=0.02)

        # test uniqueness
        # size
        self.assertTrue(len(list(mb_mod._state_yr.values())) ==
                        np.unique(list(mb_mod._state_yr.values())).size)
        # size2
        self.assertTrue(len(list(mb_mod2._state_yr.values())) ==
                        np.unique(list(mb_mod2._state_yr.values())).size)
        # state years 1 vs 2
        self.assertTrue(np.all(np.unique(list(mb_mod._state_yr.values())) ==
                               np.unique(list(mb_mod2._state_yr.values()))))
        # state years 1 vs reference model
        self.assertTrue(np.all(np.unique(list(mb_mod._state_yr.values())) ==
                               ref_mod.years))

        # test ela vs specific mb
        elats = mb_mod.get_ela(yrs[:200])
        assert np.corrcoef(mbts[:200], elats)[0, 1] < -0.95

        # test mass balance with temperature bias
        self.assertTrue(np.mean(r_mbh) < np.mean(r_mbh3))

    def test_uncertain_mb(self):

        gdir = self.gdir

        ref_mod = massbalance.ConstantMassBalance(gdir, bias=0)
        mb_mod = massbalance.UncertainMassBalance(ref_mod)

        yrs = np.arange(100)
        h, w = gdir.get_inversion_flowline_hw()
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc2_mb = mb_mod.get_specific_mb(h, w, year=yrs)

        assert_allclose(ref_mb, check_mb)
        assert_allclose(unc_mb, unc2_mb)
        assert np.std(unc_mb) > 50

        mb_mod = massbalance.UncertainMassBalance(ref_mod,
                                                  rdn_temp_bias_sigma=0.1,
                                                  rdn_prcp_bias_sigma=0,
                                                  rdn_bias_sigma=0)
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        assert_allclose(ref_mb, check_mb)
        assert np.std(unc_mb) > 50

        mb_mod = massbalance.UncertainMassBalance(ref_mod,
                                                  rdn_temp_bias_sigma=0,
                                                  rdn_prcp_bias_sigma=0.1,
                                                  rdn_bias_sigma=0)
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        assert_allclose(ref_mb, check_mb)
        assert np.std(unc_mb) > 50

        mb_mod = massbalance.UncertainMassBalance(ref_mod,
                                                  rdn_temp_bias_sigma=0,
                                                  rdn_prcp_bias_sigma=0,
                                                  rdn_bias_sigma=100)
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        assert_allclose(ref_mb, check_mb)
        assert np.std(unc_mb) > 50

        # Other MBs
        ref_mod = massbalance.PastMassBalance(gdir)
        mb_mod = massbalance.UncertainMassBalance(ref_mod)

        yrs = np.arange(100) + 1901
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc2_mb = mb_mod.get_specific_mb(h, w, year=yrs)

        assert_allclose(ref_mb, check_mb)
        assert_allclose(unc_mb, unc2_mb)
        assert np.std(unc_mb - ref_mb) > 50
        assert np.corrcoef(ref_mb, unc_mb)[0, 1] > 0.5

        # Other MBs
        ref_mod = massbalance.RandomMassBalance(gdir)
        mb_mod = massbalance.UncertainMassBalance(ref_mod)

        yrs = np.arange(100) + 1901
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc2_mb = mb_mod.get_specific_mb(h, w, year=yrs)

        assert_allclose(ref_mb, check_mb)
        assert_allclose(unc_mb, unc2_mb)
        assert np.std(unc_mb - ref_mb) > 50
        assert np.corrcoef(ref_mb, unc_mb)[0, 1] > 0.5

    def test_mb_performance(self):

        gdir = self.gdir
        init_present_time_glacier(gdir)

        h, w = gdir.get_inversion_flowline_hw()

        # Climate period, 10 day timestep
        yrs = np.arange(1850, 2003, 10/365)

        # models
        start_time = time.time()
        mb1 = massbalance.ConstantMassBalance(gdir)
        for yr in yrs:
            mb1.get_monthly_mb(h, yr)
        t1 = time.time() - start_time
        start_time = time.time()
        mb2 = massbalance.PastMassBalance(gdir)
        for yr in yrs:
            mb2.get_monthly_mb(h, yr)
        t2 = time.time() - start_time

        # not faster as two times t2
        try:
            assert t1 >= (t2 / 2)
        except AssertionError:
            # no big deal
            unittest.skip('Allowed failure')


class TestModelFlowlines(unittest.TestCase):

    def test_rectangular(self):
        map_dx = 100.
        dx = 1.
        nx = 200
        coords = np.arange(0, nx - 0.5, 1)
        line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

        bed_h = np.linspace(3000, 1000, nx)
        surface_h = bed_h + 100
        surface_h[:20] += 50
        surface_h[-20:] -= 100
        widths = bed_h * 0. + 20
        widths[:30] = 40
        widths[-30:] = 10

        rec = RectangularBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                     surface_h=surface_h, bed_h=bed_h,
                                     widths=widths)
        thick = surface_h - bed_h
        widths_m = widths * map_dx
        section = thick * widths_m
        vol_m3 = thick * map_dx * widths_m
        area_m2 = map_dx * widths_m
        area_m2[thick == 0] = 0

        assert_allclose(rec.thick, thick)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, vol_m3.sum())

        # We set something and everything stays same
        rec.thick = thick
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.surface_h, surface_h)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, vol_m3.sum())
        rec.section = section
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.surface_h, surface_h)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, vol_m3.sum())
        rec.surface_h = surface_h
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.surface_h, surface_h)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, vol_m3.sum())

        # More adventurous
        rec.section = section / 2
        assert_allclose(rec.thick, thick/2)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section/2)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, (vol_m3/2).sum())

    def test_trapeze_mixed_rec(self):

        # Special case of lambda = 0

        map_dx = 100.
        dx = 1.
        nx = 200
        coords = np.arange(0, nx - 0.5, 1)
        line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

        bed_h = np.linspace(3000, 1000, nx)
        surface_h = bed_h + 100
        surface_h[:20] += 50
        surface_h[-20:] -= 80
        widths = bed_h * 0. + 20
        widths[:30] = 40
        widths[-30:] = 10

        lambdas = bed_h*0.
        is_trap = np.ones(len(lambdas), dtype=np.bool)

        # tests
        thick = surface_h - bed_h
        widths_m = widths * map_dx
        section = thick * widths_m
        vol_m3 = thick * map_dx * widths_m
        area_m2 = map_dx * widths_m
        area_m2[thick == 0] = 0

        rec1 = TrapezoidalBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                      surface_h=surface_h,
                                      bed_h=bed_h, widths=widths,
                                      lambdas=lambdas)

        rec2 = MixedBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                surface_h=surface_h, bed_h=bed_h,
                                section=section, bed_shape=lambdas,
                                is_trapezoid=is_trap, lambdas=lambdas)

        recs = [rec1, rec2]
        for rec in recs:
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())

            # We set something and everything stays same
            rec.thick = thick
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.surface_h, surface_h)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())
            rec.section = section
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.surface_h, surface_h)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())
            rec.surface_h = surface_h
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.surface_h, surface_h)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())

            # More adventurous
            rec.section = section / 2
            assert_allclose(rec.thick, thick/2)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section/2)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, (vol_m3/2).sum())

    def test_trapeze_mixed_lambda1(self):

        # Real lambdas

        map_dx = 100.
        dx = 1.
        nx = 200
        coords = np.arange(0, nx - 0.5, 1)
        line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

        bed_h = np.linspace(3000, 1000, nx)
        surface_h = bed_h + 100
        surface_h[:20] += 50
        surface_h[-20:] -= 80
        widths_0 = bed_h * 0. + 20
        widths_0[:30] = 40
        widths_0[-30:] = 10

        lambdas = bed_h*0. + 1

        # tests
        thick = surface_h - bed_h
        widths_m = widths_0 * map_dx + lambdas * thick
        widths = widths_m / map_dx
        section = thick * (widths_0 * map_dx + widths_m) / 2
        vol_m3 = section * map_dx
        area_m2 = map_dx * widths_m
        area_m2[thick == 0] = 0

        is_trap = np.ones(len(lambdas), dtype=np.bool)

        rec1 = TrapezoidalBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                      surface_h=surface_h,
                                      bed_h=bed_h, widths=widths,
                                      lambdas=lambdas)

        rec2 = MixedBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                surface_h=surface_h, bed_h=bed_h,
                                section=section, bed_shape=lambdas,
                                is_trapezoid=is_trap, lambdas=lambdas)

        recs = [rec1, rec2]
        for rec in recs:
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())

            # We set something and everything stays same
            rec.thick = thick
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.surface_h, surface_h)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())
            rec.section = section
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.surface_h, surface_h)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())
            rec.surface_h = surface_h
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.surface_h, surface_h)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())

    def test_parab_mixed(self):

        # Real parabolas

        map_dx = 100.
        dx = 1.
        nx = 200
        coords = np.arange(0, nx - 0.5, 1)
        line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

        bed_h = np.linspace(3000, 1000, nx)
        surface_h = bed_h + 100
        surface_h[:20] += 50
        surface_h[-20:] -= 80

        shapes = bed_h*0. + 0.003
        shapes[:30] = 0.002
        shapes[-30:] = 0.004

        # tests
        thick = surface_h - bed_h
        widths_m = np.sqrt(4 * thick / shapes)
        widths = widths_m / map_dx
        section = 2 / 3 * widths_m * thick
        vol_m3 = section * map_dx
        area_m2 = map_dx * widths_m
        area_m2[thick == 0] = 0

        is_trap = np.zeros(len(shapes), dtype=np.bool)

        rec1 = ParabolicBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                    surface_h=surface_h, bed_h=bed_h,
                                    bed_shape=shapes)

        rec2 = MixedBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                surface_h=surface_h, bed_h=bed_h,
                                section=section, bed_shape=shapes,
                                is_trapezoid=is_trap, lambdas=shapes)

        recs = [rec1, rec2]
        for rec in recs:
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())

            # We set something and everything stays same
            rec.thick = thick
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())
            rec.section = section
            assert_allclose(rec.thick, thick)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, vol_m3.sum())
            assert_allclose(rec.surface_h, surface_h)

    def test_mixed(self):

        # Set a section and see if it all matches

        map_dx = 100.
        dx = 1.
        nx = 200
        coords = np.arange(0, nx - 0.5, 1)
        line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

        bed_h = np.linspace(3000, 1000, nx)
        surface_h = bed_h + 100
        surface_h[:20] += 50
        surface_h[-20:] -= 80
        widths_0 = bed_h * 0. + 20
        widths_0[:30] = 40
        widths_0[-30:] = 10

        lambdas = bed_h*0. + 1
        lambdas[0:50] = 0

        thick = surface_h - bed_h
        widths_m = widths_0 * map_dx + lambdas * thick
        widths = widths_m / map_dx
        section_trap = thick * (widths_0 * map_dx + widths_m) / 2

        rec1 = TrapezoidalBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                      surface_h=surface_h,
                                      bed_h=bed_h, widths=widths,
                                      lambdas=lambdas)

        shapes = bed_h*0. + 0.003
        shapes[-30:] = 0.004

        # tests
        thick = surface_h - bed_h
        widths_m = np.sqrt(4 * thick / shapes)
        section_para = 2 / 3 * widths_m * thick

        rec2 = ParabolicBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                    surface_h=surface_h, bed_h=bed_h,
                                    bed_shape=shapes)

        is_trap = np.ones(len(shapes), dtype=np.bool)
        is_trap[100:] = False

        section = section_trap.copy()
        section[~is_trap] = section_para[~is_trap]

        rec = MixedBedFlowline(line=line, dx=dx, map_dx=map_dx,
                               surface_h=surface_h, bed_h=bed_h,
                               section=section, bed_shape=shapes,
                               is_trapezoid=is_trap, lambdas=lambdas)

        thick = rec1.thick
        thick[~is_trap] = rec2.thick[~is_trap]
        assert_allclose(rec.thick, thick)

        widths = rec1.widths
        widths[~is_trap] = rec2.widths[~is_trap]
        assert_allclose(rec.widths, widths)

        widths_m = rec1.widths_m
        widths_m[~is_trap] = rec2.widths_m[~is_trap]
        assert_allclose(rec.widths_m, widths_m)

        section = rec1.section
        section[~is_trap] = rec2.section[~is_trap]
        assert_allclose(rec.section, section)

        # We set something and everything stays same
        area_m2 = rec.area_m2
        volume_m3 = rec.volume_m3
        rec.thick = rec.thick
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2)
        assert_allclose(rec.volume_m3, volume_m3)
        rec.section = rec.section
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2)
        assert_allclose(rec.volume_m3, volume_m3)
        rec.surface_h = rec.surface_h
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.surface_h, surface_h)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2)
        assert_allclose(rec.volume_m3, volume_m3)
        rec.surface_h = rec.surface_h - 10
        assert_allclose(rec.thick, thick - 10)
        assert_allclose(rec.surface_h, surface_h - 10)


class TestIO(unittest.TestCase):

    def setUp(self):
        gdir = init_hef(border=DOM_BORDER)
        self.test_dir = os.path.join(get_test_dir(), type(self).__name__)
        utils.mkdir(self.test_dir, reset=True)
        self.gdir = tasks.copy_to_basedir(gdir, base_dir=self.test_dir,
                                          setup='all')

        init_present_time_glacier(self.gdir)
        self.glen_a = 2.4e-24    # Modern style Glen parameter A

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_flowline_to_dataset(self):

        beds = [dummy_constant_bed, dummy_width_bed, dummy_noisy_bed,
                dummy_bumpy_bed, dummy_parabolic_bed, dummy_trapezoidal_bed,
                dummy_mixed_bed]

        for bed in beds:
            fl = bed()[0]
            ds = fl.to_dataset()
            fl_ = flowline_from_dataset(ds)
            ds_ = fl_.to_dataset()
            self.assertTrue(ds_.equals(ds))

    def test_model_to_file(self):

        p = os.path.join(self.test_dir, 'grp.nc')
        if os.path.isfile(p):
            os.remove(p)

        fls = dummy_width_bed_tributary()
        model = FluxBasedModel(fls)
        model.to_netcdf(p)
        fls_ = glacier_from_netcdf(p)

        for fl, fl_ in zip(fls, fls_):
            ds = fl.to_dataset()
            ds_ = fl_.to_dataset()
            self.assertTrue(ds_.equals(ds))

        self.assertTrue(fls_[0].flows_to is fls_[1])
        self.assertEqual(fls[0].flows_to_indice, fls_[0].flows_to_indice)

        # They should be sorted
        to_test = [fl.order for fl in fls_]
        assert np.array_equal(np.sort(to_test), to_test)

        # They should be able to start a run
        mb = LinearMassBalance(2600.)
        model = FluxBasedModel(fls_, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        model.run_until(100)

    @pytest.mark.slow
    def test_run(self):
        mb = LinearMassBalance(2600.)

        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        ds, ds_diag = model.run_until_and_store(500, store_monthly_step=True)
        ds = ds[0]

        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)

        years = utils.monthly_timeseries(0, 500)
        vol_ref = []
        a_ref = []
        l_ref = []
        vol_diag = []
        a_diag = []
        l_diag = []
        ela_diag = []
        for yr in years:
            model.run_until(yr)
            vol_diag.append(model.volume_m3)
            a_diag.append(model.area_m2)
            l_diag.append(model.length_m)
            ela_diag.append(model.mb_model.get_ela(year=yr))
            if int(yr) == yr:
                vol_ref.append(model.volume_m3)
                a_ref.append(model.area_m2)
                l_ref.append(model.length_m)
                if int(yr) == 500:
                    secfortest = model.fls[0].section

        np.testing.assert_allclose(ds.ts_section.isel(time=-1),
                                   secfortest)

        np.testing.assert_allclose(ds_diag.volume_m3, vol_diag)
        np.testing.assert_allclose(ds_diag.area_m2, a_diag)
        np.testing.assert_allclose(ds_diag.length_m, l_diag)
        np.testing.assert_allclose(ds_diag.ela_m, ela_diag)

        fls = dummy_constant_bed()
        run_path = os.path.join(self.test_dir, 'ts_ideal.nc')
        diag_path = os.path.join(self.test_dir, 'ts_diag.nc')
        if os.path.exists(run_path):
            os.remove(run_path)
        if os.path.exists(diag_path):
            os.remove(diag_path)
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        model.run_until_and_store(500, run_path=run_path,
                                  diag_path=diag_path,
                                  store_monthly_step=True)

        with xr.open_dataset(diag_path) as ds_:
            # the identical (i.e. attrs + names) doesn't work because of date
            del ds_diag.attrs['creation_date']
            del ds_.attrs['creation_date']
            xr.testing.assert_identical(ds_diag, ds_)

        with FileModel(run_path) as fmodel:
            assert fmodel.last_yr == 500
            fls = dummy_constant_bed()
            model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                                   glen_a=self.glen_a)
            for yr in years:
                model.run_until(yr)
                if yr in [100, 300, 500]:
                    # this is sloooooow so we test a little bit only
                    fmodel.run_until(yr)
                    np.testing.assert_allclose(model.fls[0].section,
                                               fmodel.fls[0].section)
                    np.testing.assert_allclose(model.fls[0].widths_m,
                                               fmodel.fls[0].widths_m)

            np.testing.assert_allclose(fmodel.volume_m3_ts(), vol_ref)
            np.testing.assert_allclose(fmodel.area_m2_ts(), a_ref)
            np.testing.assert_allclose(fmodel.length_m_ts(), l_ref)

            # Can we start a run from the middle?
            fmodel.run_until(300)
            model = FluxBasedModel(fmodel.fls, mb_model=mb, y0=300,
                                   glen_a=self.glen_a)
            model.run_until(500)
            fmodel.run_until(500)
            np.testing.assert_allclose(model.fls[0].section,
                                       fmodel.fls[0].section)

    @pytest.mark.slow
    def test_run_annual_step(self):
        mb = LinearMassBalance(2600.)

        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        ds, ds_diag = model.run_until_and_store(500)
        ds = ds[0]

        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)

        years = np.arange(0, 501)
        vol_ref = []
        a_ref = []
        l_ref = []
        vol_diag = []
        a_diag = []
        l_diag = []
        ela_diag = []
        for yr in years:
            model.run_until(yr)
            vol_diag.append(model.volume_m3)
            a_diag.append(model.area_m2)
            l_diag.append(model.length_m)
            ela_diag.append(model.mb_model.get_ela(year=yr))
            vol_ref.append(model.volume_m3)
            a_ref.append(model.area_m2)
            l_ref.append(model.length_m)
            if int(yr) == 500:
                secfortest = model.fls[0].section

        np.testing.assert_allclose(ds.ts_section.isel(time=-1),
                                   secfortest)

        np.testing.assert_allclose(ds_diag.volume_m3, vol_diag)
        np.testing.assert_allclose(ds_diag.area_m2, a_diag)
        np.testing.assert_allclose(ds_diag.length_m, l_diag)
        np.testing.assert_allclose(ds_diag.ela_m, ela_diag)

        fls = dummy_constant_bed()
        run_path = os.path.join(self.test_dir, 'ts_ideal.nc')
        diag_path = os.path.join(self.test_dir, 'ts_diag.nc')
        if os.path.exists(run_path):
            os.remove(run_path)
        if os.path.exists(diag_path):
            os.remove(diag_path)
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        model.run_until_and_store(500, run_path=run_path,
                                  diag_path=diag_path)

        with xr.open_dataset(diag_path) as ds_:
            # the identical (i.e. attrs + names) doesn't work because of date
            del ds_diag.attrs['creation_date']
            del ds_.attrs['creation_date']
            xr.testing.assert_identical(ds_diag, ds_)

        with FileModel(run_path) as fmodel:
            assert fmodel.last_yr == 500
            fls = dummy_constant_bed()
            model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                                   glen_a=self.glen_a)
            for yr in years:
                model.run_until(yr)
                if yr in [100, 300, 500]:
                    # this is sloooooow so we test a little bit only
                    fmodel.run_until(yr)
                    np.testing.assert_allclose(model.fls[0].section,
                                               fmodel.fls[0].section)
                    np.testing.assert_allclose(model.fls[0].widths_m,
                                               fmodel.fls[0].widths_m)

            np.testing.assert_allclose(fmodel.volume_m3_ts(), vol_ref)
            np.testing.assert_allclose(fmodel.area_m2_ts(), a_ref)
            np.testing.assert_allclose(fmodel.length_m_ts(), l_ref)

            # Can we start a run from the middle?
            fmodel.run_until(300)
            model = FluxBasedModel(fmodel.fls, mb_model=mb, y0=300,
                                   glen_a=self.glen_a)
            model.run_until(500)
            fmodel.run_until(500)
            np.testing.assert_allclose(model.fls[0].section,
                                       fmodel.fls[0].section)

    def test_gdir_copy(self):

        new_dir = os.path.join(get_test_dir(), 'tmp_testcopy')
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        new_gdir = tasks.copy_to_basedir(self.gdir, base_dir=new_dir,
                                         setup='all')
        init_present_time_glacier(new_gdir)
        shutil.rmtree(new_dir)

        new_gdir = tasks.copy_to_basedir(self.gdir, base_dir=new_dir,
                                         setup='run')
        run_random_climate(new_gdir, nyears=10)
        shutil.rmtree(new_dir)

        new_gdir = tasks.copy_to_basedir(self.gdir, base_dir=new_dir,
                                         setup='inversion')
        inversion.prepare_for_inversion(new_gdir, invert_all_rectangular=True)
        inversion.mass_conservation_inversion(new_gdir)
        inversion.filter_inversion_output(new_gdir)
        init_present_time_glacier(new_gdir)
        run_constant_climate(new_gdir, nyears=10, bias=0)
        shutil.rmtree(new_dir)

    def test_hef(self):

        p = os.path.join(self.test_dir, 'grp_hef.nc')
        if os.path.isfile(p):
            os.remove(p)

        init_present_time_glacier(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls)

        model.to_netcdf(p)
        fls_ = glacier_from_netcdf(p)

        for fl, fl_ in zip(fls, fls_):
            ds = fl.to_dataset()
            ds_ = fl_.to_dataset()
            for v in ds.variables.keys():
                np.testing.assert_allclose(ds_[v], ds[v], equal_nan=True)

        for fl, fl_ in zip(fls[:-1], fls_[:-1]):
            self.assertEqual(fl.flows_to_indice, fl_.flows_to_indice)

        # mixed flowline
        fls = self.gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls)

        p = os.path.join(self.test_dir, 'grp_hef_mix.nc')
        if os.path.isfile(p):
            os.remove(p)
        model.to_netcdf(p)
        fls_ = glacier_from_netcdf(p)

        np.testing.assert_allclose(fls[0].section, fls_[0].section)
        np.testing.assert_allclose(fls[0]._ptrap, fls_[0]._ptrap)
        np.testing.assert_allclose(fls[0].bed_h, fls_[0].bed_h)

        for fl, fl_ in zip(fls, fls_):
            ds = fl.to_dataset()
            ds_ = fl_.to_dataset()
            np.testing.assert_allclose(fl.section, fl_.section)
            np.testing.assert_allclose(fl._ptrap, fl_._ptrap)
            np.testing.assert_allclose(fl.bed_h, fl_.bed_h)
            xr.testing.assert_allclose(ds, ds_)

        for fl, fl_ in zip(fls[:-1], fls_[:-1]):
            self.assertEqual(fl.flows_to_indice, fl_.flows_to_indice)


class TestBackwardsIdealized(unittest.TestCase):

    def setUp(self):

        self.fs = 5.7e-20
        # Backwards
        N = 3
        _fd = 1.9e-24
        self.glen_a = (N+2) * _fd / 2.

        self.ela = 2800.

        origfls = dummy_constant_bed(nx=120, hmin=1800)

        mb = LinearMassBalance(self.ela)
        model = FluxBasedModel(origfls, mb_model=mb,
                               fs=self.fs, glen_a=self.glen_a)
        model.run_until(500)
        self.glacier = copy.deepcopy(model.fls)

    def tearDown(self):
        pass

    @pytest.mark.slow
    def test_iterative_back(self):

        # This test could be deleted
        from oggm.sandbox.ideas import _find_inital_glacier

        y0 = 0.
        y1 = 150.
        rtol = 0.02

        mb = LinearMassBalance(self.ela + 50.)
        model = FluxBasedModel(self.glacier, mb_model=mb,
                               fs=self.fs, glen_a=self.glen_a,
                               time_stepping='ambitious')

        ite, bias, past_model = _find_inital_glacier(model, mb, y0,
                                                     y1, rtol=rtol)

        bef_fls = copy.deepcopy(past_model.fls)
        past_model.run_until(y1)
        self.assertTrue(bef_fls[-1].area_m2 > past_model.area_m2)
        np.testing.assert_allclose(past_model.area_m2,
                                   self.glacier[-1].area_m2,
                                   rtol=rtol)

        if do_plot:  # pragma: no cover
            plt.plot(self.glacier[-1].surface_h, 'k', label='ref')
            plt.plot(bef_fls[-1].surface_h, 'b', label='start')
            plt.plot(past_model.fls[-1].surface_h, 'r', label='end')
            plt.plot(self.glacier[-1].bed_h, 'gray', linewidth=2)
            plt.legend(loc='best')
            plt.show()

        mb = LinearMassBalance(self.ela - 50.)
        model = FluxBasedModel(self.glacier, mb_model=mb, y0=y0,
                               fs=self.fs, glen_a=self.glen_a,
                               time_stepping='ambitious')

        ite, bias, past_model = _find_inital_glacier(model, mb, y0,
                                                     y1, rtol=rtol)
        bef_fls = copy.deepcopy(past_model.fls)
        past_model.run_until(y1)
        self.assertTrue(bef_fls[-1].area_m2 < past_model.area_m2)
        np.testing.assert_allclose(past_model.area_m2,
                                   self.glacier[-1].area_m2,
                                   rtol=rtol)

        if do_plot:  # pragma: no cover
            plt.plot(self.glacier[-1].surface_h, 'k', label='ref')
            plt.plot(bef_fls[-1].surface_h, 'b', label='start')
            plt.plot(past_model.fls[-1].surface_h, 'r', label='end')
            plt.plot(self.glacier[-1].bed_h, 'gray', linewidth=2)
            plt.legend(loc='best')
            plt.show()

        mb = LinearMassBalance(self.ela)
        model = FluxBasedModel(self.glacier, mb_model=mb, y0=y0,
                               fs=self.fs, glen_a=self.glen_a)

        # Hit the correct one
        ite, bias, past_model = _find_inital_glacier(model, mb, y0,
                                                     y1, rtol=rtol)
        past_model.run_until(y1)
        np.testing.assert_allclose(past_model.area_m2,
                                   self.glacier[-1].area_m2,
                                   rtol=rtol)

    @pytest.mark.slow
    def test_fails(self):

        # This test could be deleted
        from oggm.sandbox.ideas import _find_inital_glacier

        y0 = 0.
        y1 = 100.

        mb = LinearMassBalance(self.ela - 150.)
        model = FluxBasedModel(self.glacier, mb_model=mb, y0=y0,
                               fs=self.fs, glen_a=self.glen_a)
        self.assertRaises(RuntimeError, _find_inital_glacier, model,
                          mb, y0, y1, rtol=0.02, max_ite=5)


class TestIdealisedInversion(unittest.TestCase):

    def setUp(self):
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_ideal_inversion')

        from oggm import GlacierDirectory
        from oggm.tasks import define_glacier_region
        import geopandas as gpd

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        self.gdir = GlacierDirectory(entity, base_dir=self.testdir, reset=True)
        define_glacier_region(self.gdir, entity=entity)

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    def simple_plot(self, model):  # pragma: no cover
        ocls = self.gdir.read_pickle('inversion_output')
        ithick = ocls[-1]['thick']
        pg = model.fls[-1].thick > 0
        plt.figure()
        bh = model.fls[-1].bed_h[pg]
        sh = model.fls[-1].surface_h[pg]
        plt.plot(sh, 'k')
        plt.plot(bh, 'C0', label='Real bed')
        plt.plot(sh - ithick, 'C3', label='Computed bed')
        plt.title('Compare Shape')
        plt.xlabel('[dx]')
        plt.ylabel('Elevation [m]')
        plt.legend(loc=3)
        plt.show()

    def double_plot(self, model):  # pragma: no cover
        ocls = self.gdir.read_pickle('inversion_output')
        f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        for i, ax in enumerate(axs):
            ithick = ocls[i]['thick']
            pg = model.fls[i].thick > 0
            bh = model.fls[i].bed_h[pg]
            sh = model.fls[i].surface_h[pg]
            ax.plot(sh, 'k')
            ax.plot(bh, 'C0', label='Real bed')
            ax.plot(sh - ithick, 'C3', label='Computed bed')
            ax.set_title('Compare Shape')
            ax.set_xlabel('[dx]')
            ax.legend(loc=3)
        plt.show()

    def test_inversion_vertical(self):

        fls = dummy_constant_bed(map_dx=self.gdir.grid.dx, widths=10)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=fl.surface_h[pg])
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.01)
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

    def test_inversion_parabolic(self):

        fls = dummy_parabolic_bed(map_dx=self.gdir.grid.dx)
        mb = LinearMassBalance(2500.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=fl.surface_h[pg])
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.zeros(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)
        assert_allclose(v, model.volume_m3, rtol=0.01)

        inv = self.gdir.read_pickle('inversion_output')[-1]
        bed_shape_gl = 4 * inv['thick'] / (flo.widths * self.gdir.grid.dx) ** 2
        bed_shape_ref = (4 * fl.thick[pg] /
                         (flo.widths * self.gdir.grid.dx) ** 2)

        # assert utils.rmsd(fl.bed_shape[pg], bed_shape_gl) < 0.001
        if do_plot:  # pragma: no cover
            plt.plot(bed_shape_ref[:-3])
            plt.plot(bed_shape_gl[:-3])
            plt.show()

    def test_inversion_parabolic_sf_adhikari(self):
        old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
        old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Adhikari'
        cfg.PARAMS['use_shape_factor_for_inversion'] = 'Adhikari'

        fls = dummy_parabolic_bed(map_dx=self.gdir.grid.dx)
        for fl in fls:
            fl.is_rectangular = np.zeros(fl.nx).astype(np.bool)

        mb = LinearMassBalance(2500.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=fl.surface_h[pg])
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.zeros(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)
        assert_allclose(v, model.volume_m3, rtol=0.02)

        inv = self.gdir.read_pickle('inversion_output')[-1]
        bed_shape_gl = 4 * inv['thick'] / (flo.widths * self.gdir.grid.dx) ** 2
        bed_shape_ref = (4 * fl.thick[pg] /
                         (flo.widths * self.gdir.grid.dx) ** 2)

        # assert utils.rmsd(fl.bed_shape[pg], bed_shape_gl) < 0.001
        if do_plot:  # pragma: no cover
            plt.plot(bed_shape_ref[:-3])
            plt.plot(bed_shape_gl[:-3])
            plt.show()

        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
        cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    def test_inversion_parabolic_sf_huss(self):
        old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
        old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Huss'
        cfg.PARAMS['use_shape_factor_for_inversion'] = 'Huss'

        fls = dummy_parabolic_bed(map_dx=self.gdir.grid.dx)
        for fl in fls:
            fl.is_rectangular = np.zeros(fl.nx).astype(np.bool)

        mb = LinearMassBalance(2500.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=fl.surface_h[pg])
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.zeros(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)
        assert_allclose(v, model.volume_m3, rtol=0.01)

        inv = self.gdir.read_pickle('inversion_output')[-1]
        bed_shape_gl = 4 * inv['thick'] / (flo.widths * self.gdir.grid.dx) ** 2
        bed_shape_ref = (4 * fl.thick[pg] /
                         (flo.widths * self.gdir.grid.dx) ** 2)

        # assert utils.rmsd(fl.bed_shape[pg], bed_shape_gl) < 0.001
        if do_plot:  # pragma: no cover
            plt.plot(bed_shape_ref[:-3])
            plt.plot(bed_shape_gl[:-3])
            plt.show()

        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
        cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    @pytest.mark.slow
    def test_inversion_mixed(self):

        fls = dummy_mixed_bed(deflambdas=0, map_dx=self.gdir.grid.dx,
                              mixslice=slice(10, 30))
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        # This reduces the test's accuracy but makes it much faster.
        model.run_until_equilibrium(rate=0.01)

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = fl.is_trapezoid[pg]
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

    @pytest.mark.slow
    def test_inversion_cliff(self):

        fls = dummy_constant_bed_cliff(map_dx=self.gdir.grid.dx,
                                       cliff_height=100)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

    @pytest.mark.slow
    def test_inversion_cliff_sf_adhikari(self):
        old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
        old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Adhikari'
        cfg.PARAMS['use_shape_factor_for_inversion'] = 'Adhikari'

        fls = dummy_constant_bed_cliff(map_dx=self.gdir.grid.dx,
                                       cliff_height=100)
        for fl in fls:
            fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
        cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    @pytest.mark.slow
    def test_inversion_cliff_sf_huss(self):
        old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
        old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Huss'
        cfg.PARAMS['use_shape_factor_for_inversion'] = 'Huss'

        fls = dummy_constant_bed_cliff(map_dx=self.gdir.grid.dx,
                                       cliff_height=100)
        for fl in fls:
            fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
        cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    def test_inversion_noisy(self):

        fls = dummy_noisy_bed(map_dx=self.gdir.grid.dx)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

    def test_inversion_noisy_sf_adhikari(self):
        old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
        old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Adhikari'
        cfg.PARAMS['use_shape_factor_for_inversion'] = 'Adhikari'

        fls = dummy_noisy_bed(map_dx=self.gdir.grid.dx)
        for fl in fls:
            fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
        cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    @pytest.mark.slow
    def test_inversion_noisy_sf_huss(self):
        old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
        old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Huss'
        cfg.PARAMS['use_shape_factor_for_inversion'] = 'Huss'

        fls = dummy_noisy_bed(map_dx=self.gdir.grid.dx)
        for fl in fls:
            fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
        cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    def test_inversion_tributary(self):

        fls = dummy_width_bed_tributary(map_dx=self.gdir.grid.dx)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)

        fls[0].set_flows_to(fls[1])

        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.02)
        if do_plot:  # pragma: no cover
            self.double_plot(model)

    def test_inversion_tributary_sf_adhikari(self):
        old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
        old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Adhikari'
        cfg.PARAMS['use_shape_factor_for_inversion'] = 'Adhikari'

        fls = dummy_width_bed_tributary(map_dx=self.gdir.grid.dx)
        for fl in fls:
            fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)

        fls[0].set_flows_to(fls[1])

        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.02)
        if do_plot:  # pragma: no cover
            self.double_plot(model)

        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
        cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    @pytest.mark.slow
    def test_inversion_tributary_sf_huss(self):
        old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
        old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Huss'
        cfg.PARAMS['use_shape_factor_for_inversion'] = 'Huss'

        fls = dummy_width_bed_tributary(map_dx=self.gdir.grid.dx)
        for fl in fls:
            fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               time_stepping='conservative')
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)

        fls[0].set_flows_to(fls[1])

        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.02)
        if do_plot:  # pragma: no cover
            self.double_plot(model)

        cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
        cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    def test_inversion_non_equilibrium(self):

        fls = dummy_constant_bed(map_dx=self.gdir.grid.dx)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()

        mb = LinearMassBalance(2800.)
        model = FluxBasedModel(fls, mb_model=mb, y0=0)
        model.run_until(50)

        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        # expected errors
        assert v > model.volume_m3
        ocls = self.gdir.read_pickle('inversion_output')
        ithick = ocls[0]['thick']
        assert np.mean(ithick) > np.mean(model.fls[0].thick)*1.1
        if do_plot:  # pragma: no cover
            self.simple_plot(model)

    def test_inversion_and_run(self):

        fls = dummy_parabolic_bed(map_dx=self.gdir.grid.dx)
        mb = LinearMassBalance(2500.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where(fl.thick > 0)
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.zeros(flo.nx).astype(np.bool)
            fls.append(flo)
        self.gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(self.gdir)
        inversion.prepare_for_inversion(self.gdir)
        v, _ = inversion.mass_conservation_inversion(self.gdir)

        assert_allclose(v, model.volume_m3, rtol=0.01)

        inv = self.gdir.read_pickle('inversion_output')[-1]
        bed_shape_gl = 4 * inv['thick'] / (flo.widths * self.gdir.grid.dx) ** 2

        ithick = inv['thick']
        fls = dummy_parabolic_bed(map_dx=self.gdir.grid.dx,
                                  from_other_shape=bed_shape_gl[:-2],
                                  from_other_bed=sh-ithick)
        model2 = FluxBasedModel(fls, mb_model=mb, y0=0.,
                                time_stepping='conservative')
        model2.run_until_equilibrium()
        assert_allclose(model2.volume_m3, model.volume_m3, rtol=0.01)

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(model.fls[-1].bed_h, 'C0')
            plt.plot(model2.fls[-1].bed_h, 'C3')
            plt.plot(model.fls[-1].surface_h, 'C0')
            plt.plot(model2.fls[-1].surface_h, 'C3')
            plt.title('Compare Shape')
            plt.xlabel('[m]')
            plt.ylabel('Elevation [m]')
            plt.show()


class TestHEF(unittest.TestCase):

    def setUp(self):
        gdir = init_hef(border=DOM_BORDER)
        self.testdir = os.path.join(get_test_dir(), type(self).__name__)
        utils.mkdir(self.testdir, reset=True)
        self.gdir = tasks.copy_to_basedir(gdir, base_dir=self.testdir,
                                          setup='all')

        d = self.gdir.read_pickle('inversion_params')
        self.fs = d['fs']
        self.glen_a = d['glen_a']

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)

    @pytest.mark.slow
    def test_equilibrium(self):

        init_present_time_glacier(self.gdir)

        mb_mod = massbalance.ConstantMassBalance(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                               fs=self.fs,
                               glen_a=self.glen_a,
                               min_dt=SEC_IN_DAY/2.,
                               mb_elev_feedback='never')

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_area, self.gdir.rgi_area_km2, rtol=0.03)

        model.run_until_equilibrium(rate=1e-4)
        self.assertFalse(model.dt_warning)
        assert model.yr > 50
        after_vol = model.volume_km3
        after_area = model.area_km2
        after_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_vol, after_vol, rtol=0.1)
        np.testing.assert_allclose(ref_area, after_area, rtol=0.03)
        np.testing.assert_allclose(ref_len, after_len, atol=500.01)

    @pytest.mark.slow
    def test_equilibrium_glacier_wide(self):

        init_present_time_glacier(self.gdir)

        cl = massbalance.ConstantMassBalance
        mb_mod = massbalance.MultipleFlowlineMassBalance(self.gdir,
                                                         mb_model_class=cl)

        fls = self.gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                               fs=self.fs,
                               glen_a=self.glen_a,
                               min_dt=SEC_IN_DAY/2.,
                               mb_elev_feedback='never')

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_area, self.gdir.rgi_area_km2, rtol=0.03)

        model.run_until_equilibrium(rate=1e-4)
        self.assertFalse(model.dt_warning)
        assert model.yr > 50
        after_vol = model.volume_km3
        after_area = model.area_km2
        after_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_vol, after_vol, rtol=0.1)
        np.testing.assert_allclose(ref_area, after_area, rtol=0.03)
        np.testing.assert_allclose(ref_len, after_len, atol=500.01)

    @pytest.mark.slow
    def test_commitment(self):

        init_present_time_glacier(self.gdir)

        mb_mod = massbalance.ConstantMassBalance(self.gdir, y0=2003 - 15)

        fls = self.gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                               fs=self.fs,
                               glen_a=self.glen_a)

        ref_area = model.area_km2
        np.testing.assert_allclose(ref_area, self.gdir.rgi_area_km2, rtol=0.02)

        model.run_until_equilibrium()
        self.assertTrue(model.yr > 100)

        after_vol_1 = model.volume_km3

        _tmp = cfg.PARAMS['mixed_min_shape']
        cfg.PARAMS['mixed_min_shape'] = 0.001
        init_present_time_glacier(self.gdir)
        cfg.PARAMS['mixed_min_shape'] = _tmp

        glacier = self.gdir.read_pickle('model_flowlines')

        fls = self.gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                               fs=self.fs,
                               glen_a=self.glen_a)

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        np.testing.assert_allclose(ref_area, self.gdir.rgi_area_km2, rtol=0.02)

        model.run_until_equilibrium()
        self.assertTrue(model.yr > 100)

        after_vol_2 = model.volume_km3

        self.assertTrue(after_vol_1 < (0.5 * ref_vol))
        self.assertTrue(after_vol_2 < (0.5 * ref_vol))

        if do_plot:  # pragma: no cover
            plt.figure()
            plt.plot(glacier[-1].surface_h, 'b', label='start')
            plt.plot(model.fls[-1].surface_h, 'r', label='end')

            plt.plot(glacier[-1].bed_h, 'gray', linewidth=2)
            plt.legend(loc='best')
            plt.show()

    @pytest.mark.slow
    def test_random(self):

        init_present_time_glacier(self.gdir)
        run_random_climate(self.gdir, nyears=100, seed=6,
                           fs=self.fs, glen_a=self.glen_a,
                           bias=0, output_filesuffix='_rdn')
        run_constant_climate(self.gdir, nyears=100,
                             fs=self.fs, glen_a=self.glen_a,
                             bias=0, output_filesuffix='_ct')

        paths = [self.gdir.get_filepath('model_run', filesuffix='_rdn'),
                 self.gdir.get_filepath('model_run', filesuffix='_ct'),
                 ]

        for path in paths:
            with FileModel(path) as model:
                vol = model.volume_km3_ts()
                len = model.length_m_ts()
                area = model.area_km2_ts()
                np.testing.assert_allclose(vol.iloc[0], np.mean(vol),
                                           rtol=0.1)
                np.testing.assert_allclose(area.iloc[0], np.mean(area),
                                           rtol=0.1)
                if do_plot:
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
                    vol.plot(ax=ax1)
                    ax1.set_title('Volume')
                    area.plot(ax=ax2)
                    ax2.set_title('Area')
                    len.plot(ax=ax3)
                    ax3.set_title('Length')
                    plt.tight_layout()
                    plt.show()

    @pytest.mark.slow
    def test_random_sh(self):

        init_present_time_glacier(self.gdir)

        self.gdir.hemisphere = 'sh'
        cfg.PATHS['climate_file'] = ''
        cfg.PARAMS['baseline_climate'] = 'CRU'
        cfg.PARAMS['run_mb_calibration'] = True
        cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
        cfg.PATHS['cru_dir'] = os.path.dirname(cru_dir)
        climate.process_cru_data(self.gdir)
        climate.compute_ref_t_stars([self.gdir])
        climate.local_t_star(self.gdir)

        run_random_climate(self.gdir, nyears=20, seed=4,
                           bias=0, output_filesuffix='_rdn')
        run_constant_climate(self.gdir, nyears=20,
                             bias=0, output_filesuffix='_ct')

        paths = [self.gdir.get_filepath('model_run', filesuffix='_rdn'),
                 self.gdir.get_filepath('model_run', filesuffix='_ct'),
                 ]
        for path in paths:
            with FileModel(path) as model:
                vol = model.volume_km3_ts()
                len = model.length_m_ts()
                area = model.area_km2_ts()
                np.testing.assert_allclose(vol.iloc[0], np.mean(vol),
                                           rtol=0.1)
                np.testing.assert_allclose(area.iloc[0], np.mean(area),
                                           rtol=0.1)
                if do_plot:
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
                    vol.plot(ax=ax1)
                    ax1.set_title('Volume')
                    area.plot(ax=ax2)
                    ax2.set_title('Area')
                    len.plot(ax=ax3)
                    ax3.set_title('Length')
                    plt.tight_layout()
                    plt.show()

        self.gdir.hemisphere = 'nh'

    def test_start_from_spinup(self):

        init_present_time_glacier(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        vol = 0
        area = 0
        for fl in fls:
            vol += fl.volume_km3
            area += fl.area_km2
        assert self.gdir.rgi_date == 2003

        # Make a dummy run for 0 years
        run_from_climate_data(self.gdir, ye=2003, output_filesuffix='_1')

        fp = self.gdir.get_filepath('model_run', filesuffix='_1')
        with FileModel(fp) as fmod:
            fmod.run_until(fmod.last_yr)
            np.testing.assert_allclose(fmod.area_km2, area)
            np.testing.assert_allclose(fmod.volume_km3, vol)

        # Again
        run_from_climate_data(self.gdir, ye=2003, init_model_filesuffix='_1',
                              output_filesuffix='_2')
        fp = self.gdir.get_filepath('model_run', filesuffix='_2')
        with FileModel(fp) as fmod:
            fmod.run_until(fmod.last_yr)
            np.testing.assert_allclose(fmod.area_km2, area)
            np.testing.assert_allclose(fmod.volume_km3, vol)

    def test_start_from_spinup_min_ys(self):

        init_present_time_glacier(self.gdir)

        fls = self.gdir.read_pickle('model_flowlines')
        vol = 0
        area = 0
        for fl in fls:
            vol += fl.volume_km3
            area += fl.area_km2
        assert self.gdir.rgi_date == 2003

        # Make a dummy run for 0 years
        run_from_climate_data(self.gdir, ye=2002, min_ys=2002,
                              output_filesuffix='_1')

        fp = self.gdir.get_filepath('model_run', filesuffix='_1')
        with FileModel(fp) as fmod:
            fmod.run_until(fmod.last_yr)
            np.testing.assert_allclose(fmod.area_km2, area)
            np.testing.assert_allclose(fmod.volume_km3, vol)

        # Again
        run_from_climate_data(self.gdir, ys=2002, ye=2003,
                              init_model_filesuffix='_1',
                              output_filesuffix='_2')
        fp = self.gdir.get_filepath('model_run', filesuffix='_2')
        with FileModel(fp) as fmod:
            fmod.run_until(fmod.last_yr)
            np.testing.assert_allclose(fmod.area_km2, area, rtol=0.05)
            np.testing.assert_allclose(fmod.volume_km3, vol, rtol=0.05)

    @pytest.mark.slow
    def test_cesm(self):

        gdir = self.gdir

        # init
        f = get_demo_file('cesm.TREFHT.160001-200512.selection.nc')
        cfg.PATHS['cesm_temp_file'] = f
        f = get_demo_file('cesm.PRECC.160001-200512.selection.nc')
        cfg.PATHS['cesm_precc_file'] = f
        f = get_demo_file('cesm.PRECL.160001-200512.selection.nc')
        cfg.PATHS['cesm_precl_file'] = f
        gcm_climate.process_cesm_data(self.gdir)

        # Climate data
        fh = gdir.get_filepath('climate_monthly')
        fcesm = gdir.get_filepath('gcm_data')
        with xr.open_dataset(fh) as hist, xr.open_dataset(fcesm) as cesm:

            # Let's do some basic checks
            shist = hist.sel(time=slice('1961', '1990'))
            scesm = cesm.sel(time=slice('1961', '1990'))
            # Climate during the chosen period should be the same
            np.testing.assert_allclose(shist.temp.mean(),
                                       scesm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(shist.prcp.mean(),
                                       scesm.prcp.mean(),
                                       rtol=1e-3)
            # And also the anual cycle
            scru = shist.groupby('time.month').mean(dim='time')
            scesm = scesm.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(scru.temp, scesm.temp, rtol=5e-3)
            np.testing.assert_allclose(scru.prcp, scesm.prcp, rtol=1e-3)

        # Mass balance models
        mb_cru = massbalance.PastMassBalance(self.gdir)
        mb_cesm = massbalance.PastMassBalance(self.gdir,
                                              filename='gcm_data')

        # Average over 1961-1990
        h, w = self.gdir.get_inversion_flowline_hw()
        yrs = np.arange(1961, 1991)
        ts1 = mb_cru.get_specific_mb(h, w, year=yrs)
        ts2 = mb_cesm.get_specific_mb(h, w, year=yrs)
        # due to non linear effects the MBs are not equivalent! See if they
        # aren't too far:
        assert np.abs(np.mean(ts1) - np.mean(ts2)) < 100

        # For my own interest, some statistics
        yrs = np.arange(1851, 2004)
        ts1 = mb_cru.get_specific_mb(h, w, year=yrs)
        ts2 = mb_cesm.get_specific_mb(h, w, year=yrs)
        if do_plot:
            df = pd.DataFrame(index=yrs)
            k1 = 'Histalp (mean={:.1f}, stddev={:.1f})'.format(np.mean(ts1),
                                                               np.std(ts1))
            k2 = 'CESM (mean={:.1f}, stddev={:.1f})'.format(np.mean(ts2),
                                                            np.std(ts2))
            df[k1] = ts1
            df[k2] = ts2

            df.plot()
            plt.plot(yrs,
                     df[k1].rolling(31, center=True, min_periods=15).mean(),
                     color='C0', linewidth=3)
            plt.plot(yrs,
                     df[k2].rolling(31, center=True, min_periods=15).mean(),
                     color='C1', linewidth=3)
            plt.title('SMB Hintereisferner Histalp VS CESM')
            plt.show()

        # See what that means for a run
        init_present_time_glacier(gdir)
        run_from_climate_data(gdir, ys=1961, ye=1990,
                              output_filesuffix='_hist')
        run_from_climate_data(gdir, ys=1961, ye=1990,
                              climate_filename='gcm_data',
                              output_filesuffix='_cesm')

        ds1 = utils.compile_run_output([gdir], filesuffix='_hist')
        ds2 = utils.compile_run_output([gdir], filesuffix='_cesm')

        assert_allclose(ds1.volume.isel(rgi_id=0, time=-1),
                        ds2.volume.isel(rgi_id=0, time=-1),
                        rtol=0.1)
        # ELA should be close
        assert_allclose(ds1.ela.mean(), ds2.ela.mean(), atol=50)

        # Do a spinup run
        run_constant_climate(gdir, nyears=100, temperature_bias=-0.5,
                             output_filesuffix='_spinup')
        run_from_climate_data(gdir, ys=1961, ye=1990,
                              init_model_filesuffix='_spinup',
                              output_filesuffix='_afterspinup')
        ds3 = utils.compile_run_output([gdir], path=False,
                                       filesuffix='_afterspinup')
        assert (ds1.volume.isel(rgi_id=0, time=-1) <
                0.7*ds3.volume.isel(rgi_id=0, time=-1))
        ds3.close()

    @pytest.mark.slow
    def test_elevation_feedback(self):

        init_present_time_glacier(self.gdir)

        feedbacks = ['annual', 'monthly', 'always', 'never']
        # Mutliproc
        tasks = []
        for feedback in feedbacks:
            tasks.append((run_random_climate,
                          dict(nyears=200, seed=5, mb_elev_feedback=feedback,
                               output_filesuffix=feedback,
                               store_monthly_step=True)))
        workflow.execute_parallel_tasks(self.gdir, tasks)

        out = []
        for feedback in feedbacks:
            out.append(utils.compile_run_output([self.gdir], path=False,
                                                filesuffix=feedback))

        # Check that volume isn't so different
        assert_allclose(out[0].volume, out[1].volume, rtol=0.05)
        assert_allclose(out[0].volume, out[2].volume, rtol=0.05)
        assert_allclose(out[1].volume, out[2].volume, rtol=0.05)
        # Except for "never", where things are different
        assert out[3].volume.mean() < out[2].volume.mean()

        if do_plot:
            plt.figure()
            for ds, lab in zip(out, feedbacks):
                (ds.volume*1e-9).plot(label=lab)
            plt.xlabel('Vol (km3)')
            plt.legend()
            plt.show()


class TestMergedHEF(unittest.TestCase):

    def setUp(self):
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_merged')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['correct_for_neg_flux'] = True
        cfg.PARAMS['baseline_climate'] = 'CUSTOM'
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PARAMS['border'] = 100
        cfg.PARAMS['prcp_scaling_factor'] = 1.75
        cfg.PARAMS['temp_melt'] = -1.75
        cfg.PARAMS['use_multiprocessing'] = False

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    @pytest.mark.slow
    def test_merged_simulation(self):
        import geopandas as gpd

        hef_file = utils.get_demo_file('rgi_oetztal.shp')
        rgidf = gpd.read_file(hef_file)

        # Get HEF Kesselwand and Gepatschferner
        glcdf = rgidf.loc[(rgidf.RGIId == 'RGI50-11.00897') |
                          (rgidf.RGIId == 'RGI50-11.00787') |
                          (rgidf.RGIId == 'RGI50-11.00746')].copy()
        gdirs = workflow.init_glacier_regions(glcdf)
        workflow.gis_prepro_tasks(gdirs)
        workflow.climate_tasks(gdirs)
        workflow.inversion_tasks(gdirs)
        workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

        # run parameters
        years = 200  # arbitrary
        tbias = -1.0  # arbitrary

        # run HEF and Kesselwandferner as entities
        gdirs_entity = [gd for gd in gdirs if gd.rgi_id != 'RGI50-11.00746']
        workflow.execute_entity_task(tasks.run_constant_climate,
                                     gdirs_entity,
                                     nyears=years,
                                     output_filesuffix='_entity',
                                     temperature_bias=tbias)

        ds_entity = utils.compile_run_output(gdirs_entity,
                                             path=False, filesuffix='_entity')

        # merge HEF and KWF, include Gepatschferner but should not be merged
        gdir_merged = workflow.merge_glacier_tasks(gdirs, ['RGI50-11.00897'],
                                                   glcdf=glcdf)

        # and run the merged glacier
        workflow.execute_entity_task(tasks.run_constant_climate,
                                     gdir_merged, output_filesuffix='_merged',
                                     nyears=years,
                                     temperature_bias=tbias)

        ds_merged = utils.compile_run_output(gdir_merged,
                                             path=False, filesuffix='_merged')

        # with this setting, both runs should still be quite close after 50yrs
        assert_allclose(ds_entity.volume.isel(time=50).sum(),
                        ds_merged.volume.isel(time=50),
                        rtol=1e-2)

        # After 100yrs a difference will be present but should still be small
        assert_allclose(ds_entity.volume.isel(time=100).sum(),
                        ds_merged.volume.isel(time=100),
                        rtol=1e-1)

        # After 200yrs the merged glacier should have a larger volume
        assert (ds_entity.volume.isel(time=200).sum() <
                ds_merged.volume.isel(time=200))
