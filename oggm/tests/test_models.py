import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import os
from functools import partial
import shutil
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

pytest.importorskip('geopandas')
pytest.importorskip('rasterio')
pytest.importorskip('salem')


@pytest.fixture(autouse=True, scope='module')
def init_url_retrieve(request):
    request.module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github
    yield
    oggm.utils._downloads.oggm_urlretrieve = request.module._url_retrieve


pytestmark = pytest.mark.test_env("models")
do_plot = False

DOM_BORDER = 80


@pytest.fixture(scope='module')
def io_model_factory():
    glen_a = 2.4e-24
    return lambda fls, **kwargs: FluxBasedModel(fls, glen_a=glen_a, **kwargs)


@pytest.fixture(scope='module')
def backwards_idealized_fls(backwards_model_factory):
    cfg.initialize()
    origfls = dummy_constant_bed(nx=120, hmin=1800)

    model = backwards_model_factory(origfls)
    model.run_until(500)
    return copy.deepcopy(model.fls)


@pytest.fixture(scope='module')
def backwards_model_factory():
    fs = 5.7e-20
    # Backwards
    N = 3
    fd = 1.9e-24
    glen_a = (N + 2) * fd / 2.

    ela = 2800.

    def _model_factory(*args, ela_delta=0., **kwargs):
        mb = LinearMassBalance(ela + ela_delta)
        return FluxBasedModel(*args, mb_model=mb, fs=fs,
                              glen_a=glen_a, **kwargs)

    return _model_factory


@pytest.fixture(scope='module')
def inversion_gdir(test_dir):
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

    gdir = GlacierDirectory(entity, base_dir=test_dir, reset=True)
    define_glacier_region(gdir, entity=entity)
    return gdir


@pytest.fixture(scope='module')
def gdir_sh(request, test_dir, hef_gdir_base):
    dir_sh = os.path.join(test_dir, request.module.__name__ + '_sh')
    utils.mkdir(dir_sh, reset=True)
    gdir_sh = tasks.copy_to_basedir(hef_gdir_base, base_dir=dir_sh,
                                    setup='all')
    gdir_sh.hemisphere = 'sh'
    yield gdir_sh
    # teardown
    if os.path.exists(dir_sh):
        shutil.rmtree(dir_sh)


@pytest.fixture
def with_class_wd(request, test_dir, hef_gdir_base):
    # dependency on hef_gdir_base to ensure proper initialization order
    prev_wd = cfg.PATHS['working_dir']
    cfg.PATHS['working_dir'] = os.path.join(
        test_dir, request.function.__name__ + '_wd')
    utils.mkdir(cfg.PATHS['working_dir'], reset=True)
    yield
    # teardown
    cfg.PATHS['working_dir'] = prev_wd


@pytest.fixture
def inversion_params(hef_gdir):
    return hef_gdir.read_pickle('inversion_params')


def test_init_present_time_glacier(hef_gdir):
    gdir = hef_gdir
    init_present_time_glacier(gdir)

    fls = gdir.read_pickle('model_flowlines')

    ofl = gdir.read_pickle('inversion_flowlines')[-1]

    assert gdir.rgi_date == 2003
    assert len(fls) == 3

    vol = 0.
    area = 0.
    for fl in fls:
        refo = 1 if fl is fls[-1] else 0
        assert fl.order == refo
        ref = np.arange(len(fl.surface_h)) * fl.dx
        np.testing.assert_allclose(ref, fl.dis_on_line,
                                   rtol=0.001,
                                   atol=0.01)
        assert(len(fl.surface_h) ==
               len(fl.bed_h) ==
               len(fl.bed_shape) ==
               len(fl.dis_on_line) ==
               len(fl.widths))

        assert np.all(fl.widths >= 0)
        vol += fl.volume_km3
        area += fl.area_km2

        if refo == 1:
            rmsd = utils.rmsd(ofl.widths[:-5] * gdir.grid.dx,
                              fl.widths_m[0:len(ofl.widths) - 5])
            assert rmsd < 5.

    rtol = 0.02
    np.testing.assert_allclose(0.573, vol, rtol=rtol)
    np.testing.assert_allclose(6900.0, fls[-1].length_m, atol=101)
    np.testing.assert_allclose(gdir.rgi_area_km2, area, rtol=rtol)

    if do_plot:
        plt.plot(fls[-1].bed_h)
        plt.plot(fls[-1].surface_h)
        plt.show()


def test_present_time_glacier_massbalance(hef_gdir):

    gdir = hef_gdir
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
    assert np.abs(utils.md(tot_mb, refmb)) < 50

    # Gradient
    dfg = gdir.get_ref_mb_profile().mean()

    # Take the altitudes below 3100 and fit a line
    dfg = dfg[dfg.index < 3100]
    pok = np.where(hgts < 3100)
    from scipy.stats import linregress
    slope_obs, _, _, _, _ = linregress(dfg.index, dfg.values)
    slope_our, _, _, _, _ = linregress(hgts[pok], grads[pok])
    np.testing.assert_allclose(slope_obs, slope_our, rtol=0.15)


def test_define_divides(case_dir):

    from oggm.core import centerlines
    from oggm.core import climate
    from oggm.core import inversion
    from oggm.core import gis
    from oggm import GlacierDirectory
    import geopandas as gpd

    cfg.initialize()
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

    hef_file = utils.get_demo_file('rgi_oetztal.shp')
    rgidf = gpd.read_file(hef_file)

    # This is another glacier with divides
    entity = rgidf.loc[rgidf.RGIId == 'RGI50-11.00719_d01'].iloc[0]
    gdir = GlacierDirectory(entity, base_dir=case_dir)
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
        assert len(fls) == 3
    vol = 0.
    area = 0.
    for fl in fls:
        ref = np.arange(len(fl.surface_h)) * fl.dx
        np.testing.assert_allclose(ref, fl.dis_on_line,
                                   rtol=0.001,
                                   atol=0.01)
        assert(len(fl.surface_h) ==
               len(fl.bed_h) ==
               len(fl.bed_shape) ==
               len(fl.dis_on_line) ==
               len(fl.widths))

        assert np.all(fl.widths >= 0)
        vol += fl.volume_km3
        area += fl.area_km2

    rtol = 0.08
    np.testing.assert_allclose(gdir.rgi_area_km2, area, rtol=rtol)
    np.testing.assert_allclose(v * 1e-9, vol, rtol=rtol)


def test_past_mb_model(hef_gdir):

    rho = cfg.PARAMS['ice_density']

    F = SEC_IN_YEAR * rho

    gdir = hef_gdir
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
    for i, yr in enumerate(np.arange(yrp[0], yrp[1] + 1)):
        ref_mb_on_h = p[:, i] - mu_star * t[:, i]
        my_mb_on_h = mb_mod.get_annual_mb(h, yr) * F
        np.testing.assert_allclose(ref_mb_on_h, my_mb_on_h,
                                   atol=1e-2)
        ela_z = mb_mod.get_ela(year=yr)
        totest = mb_mod.get_annual_mb([ela_z], year=yr) * F
        assert_allclose(totest[0], 0, atol=1)

    mb_mod = massbalance.PastMassBalance(gdir)
    for i, yr in enumerate(np.arange(yrp[0], yrp[1] + 1)):
        ref_mb_on_h = p[:, i] - mu_star * t[:, i]
        my_mb_on_h = mb_mod.get_annual_mb(h, yr) * F
        np.testing.assert_allclose(ref_mb_on_h, my_mb_on_h + bias,
                                   atol=1e-2)
        ela_z = mb_mod.get_ela(year=yr)
        totest = mb_mod.get_annual_mb([ela_z], year=yr) * F
        assert_allclose(totest[0], 0, atol=1)

    for i, yr in enumerate(np.arange(yrp[0], yrp[1] + 1)):

        ref_mb_on_h = p[:, i] - mu_star * t[:, i]
        my_mb_on_h = ref_mb_on_h * 0.
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
    assert mbdf.ANNUAL_BALANCE.mean() > mbdf.BIASED_MB.mean()

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


@pytest.mark.parametrize("cl", [massbalance.PastMassBalance,
                                massbalance.ConstantMassBalance,
                                massbalance.RandomMassBalance])
def test_glacierwide_mb_model(hef_gdir, cl):
    gdir = hef_gdir
    init_present_time_glacier(gdir)

    fls = gdir.read_pickle('model_flowlines')
    h = np.array([])
    w = np.array([])
    for fl in fls:
        w = np.append(w, fl.widths)
        h = np.append(h, fl.surface_h)

    yrs = np.arange(100) + 1901

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

    if cl is massbalance.PastMassBalance:
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


def test_constant_mb_model(hef_gdir):

    rho = cfg.PARAMS['ice_density']

    gdir = hef_gdir
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

    assert ntmb < otmb

    if do_plot:  # pragma: no cover
        plt.plot(h, ombh, 'o', label='tstar')
        plt.plot(h, nmbh, 'o', label='today')
        plt.legend()
        plt.show()

    cmb_mod.temp_bias = 1
    biasombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
    biasotmb = np.average(biasombh, weights=w)
    assert biasotmb < (otmb - 500)

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
    assert np.mean(monthly_1[5:]) > (np.mean(monthly_2[5:]) + 100)

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


def test_random_mb(hef_gdir):

    gdir = hef_gdir
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
    assert np.mean(r_mbh) < np.mean(r_mbh_b)

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
    time = pd.date_range('1/1/1973', periods=31 * 12, freq='MS')
    yrs = utils.date_to_floatyear(time.year, time.month)

    ref_mb = np.zeros(12)
    my_mb = np.zeros(12)
    for yr, m in zip(yrs, time.month):
        ref_mb[m - 1] += np.average(mb_ref.get_monthly_mb(h, yr) *
                                    SEC_IN_MONTH, weights=w)
        my_mb[m - 1] += np.average(mb_mod.get_monthly_mb(h, yr) *
                                   SEC_IN_MONTH, weights=w)
    my_mb = my_mb / 31
    ref_mb = ref_mb / 31
    assert utils.rmsd(ref_mb, my_mb) < 0.1


def test_random_mb_unique(hef_gdir):

    gdir = hef_gdir
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
    assert(len(list(mb_mod._state_yr.values())) ==
           np.unique(list(mb_mod._state_yr.values())).size)
    # size2
    assert(len(list(mb_mod2._state_yr.values())) ==
           np.unique(list(mb_mod2._state_yr.values())).size)
    # state years 1 vs 2
    assert(np.all(np.unique(list(mb_mod._state_yr.values())) ==
                  np.unique(list(mb_mod2._state_yr.values()))))
    # state years 1 vs reference model
    assert(np.all(np.unique(list(mb_mod._state_yr.values())) ==
                  ref_mod.years))

    # test ela vs specific mb
    elats = mb_mod.get_ela(yrs[:200])
    assert np.corrcoef(mbts[:200], elats)[0, 1] < -0.95

    # test mass balance with temperature bias
    assert np.mean(r_mbh) < np.mean(r_mbh3)


def test_uncertain_mb(hef_gdir):

    gdir = hef_gdir

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


def test_mb_performance(hef_gdir):

    gdir = hef_gdir
    init_present_time_glacier(gdir)

    h, w = gdir.get_inversion_flowline_hw()

    # Climate period, 10 day timestep
    yrs = np.arange(1850, 2003, 10 / 365)

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
        pytest.skip('Allowed failure')


class MixedParabolicBedFlowline(MixedBedFlowline):
    """ subclass to determine expected kwargs for mixed cases in
        fl_shape_factory
    """

    def __init__(self, *args, **kwargs):
        super(MixedParabolicBedFlowline, self).__init__(*args, **kwargs)


MODEL_LEN = 200


@pytest.fixture(scope='module')
def fl_shape_factory():
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

    thick = surface_h - bed_h

    shapes = bed_h * 0. + 0.003
    shapes[:30] = 0.002
    shapes[-30:] = 0.004

    def _factory(fl_model, expects_factory, lambdas=None,
                 is_trap=None, section=None):
        kwargs = {}

        parabolic = (fl_model is ParabolicBedFlowline or
                     fl_model is MixedParabolicBedFlowline)
        mixed = (fl_model is MixedBedFlowline or
                 fl_model is MixedParabolicBedFlowline)

        (widths_m, section_, vol_m3, area_m2, w_adj) = expects_factory(
            thick=thick, widths=widths, map_dx=map_dx, lambdas=lambdas,
            shapes=shapes)

        widths_ = widths if w_adj is None else w_adj

        lambdas = lambdas
        section = section if section is not None else section_

        if parabolic:
            kwargs['bed_shape'] = shapes
        if mixed:
            kwargs['section'] = section
            if parabolic and lambdas is None:
                lambdas = shapes
            if not parabolic:
                kwargs['bed_shape'] = lambdas
            kwargs['is_trapezoid'] = is_trap
        if not parabolic and not mixed:
            kwargs['widths'] = widths_
        if lambdas is not None:
            kwargs['lambdas'] = lambdas

        return (fl_model(line=line, dx=dx, map_dx=map_dx,
                         surface_h=surface_h, bed_h=bed_h,
                         **kwargs),
                (thick, surface_h, widths_, widths_m,
                 section, vol_m3, area_m2))

    return _factory


def rectangular_expects(thick, widths, map_dx, **kwargs):
    widths_m = widths * map_dx
    section = thick * widths_m
    vol_m3 = thick * map_dx * widths_m
    area_m2 = map_dx * widths_m
    area_m2[thick == 0] = 0
    return (widths_m, section, vol_m3, area_m2, None)


def trapeze_expects(thick, widths, map_dx, lambdas, **kwargs):
    widths_m = widths * map_dx + lambdas * thick
    w_adj = widths_m / map_dx
    section = thick * (widths * map_dx + widths_m) / 2
    vol_m3 = section * map_dx
    area_m2 = map_dx * widths_m
    area_m2[thick == 0] = 0
    return (widths_m, section, vol_m3, area_m2, w_adj)


def parabola_expects(thick, shapes, map_dx, **kwargs):
    widths_m = np.sqrt(4 * thick / shapes)
    w_adj = widths_m / map_dx
    section = 2 / 3 * widths_m * thick
    vol_m3 = section * map_dx
    area_m2 = map_dx * widths_m
    area_m2[thick == 0] = 0
    return (widths_m, section, vol_m3, area_m2, w_adj)


@pytest.mark.parametrize(
    "fl_model, expects_factory, lambdas, is_trap",
    [
        pytest.param(RectangularBedFlowline, rectangular_expects,
                     None, None, id="rectangular"),
        pytest.param(TrapezoidalBedFlowline, rectangular_expects, np.zeros(
            MODEL_LEN, dtype=np.float), None, id="trapeze_rec"),
        pytest.param(MixedBedFlowline, rectangular_expects,
                     np.zeros(MODEL_LEN, dtype=np.float),
                     np.ones(MODEL_LEN, dtype=np.bool),
                     id="trapeze_rec_mixed"),
        pytest.param(TrapezoidalBedFlowline, trapeze_expects, np.ones(
            MODEL_LEN, dtype=np.float), None, id="trapeze_lambda1"),
        pytest.param(MixedBedFlowline, trapeze_expects,
                     np.ones(MODEL_LEN, dtype=np.float),
                     np.ones(MODEL_LEN, dtype=np.bool),
                     id="trapeze_lambda1_mixed"),
        pytest.param(ParabolicBedFlowline, parabola_expects,
                     None, None, id="parabola"),
        pytest.param(MixedParabolicBedFlowline, parabola_expects, None,
                     np.zeros(MODEL_LEN, dtype=np.bool), id="parabola_mixed")
    ]
)
def test_model_flowlines(
        fl_shape_factory, expects_factory, fl_model, lambdas, is_trap):
    (fl, expects) = fl_shape_factory(fl_model, expects_factory,
                                     lambdas, is_trap)

    (thick, surface_h, widths, widths_m, section, vol_m3, area_m2) = expects
    print("VALUE OF SECTION IS ", section)

    assert_allclose(fl.thick, thick)
    assert_allclose(fl.widths, widths)
    assert_allclose(fl.widths_m, widths_m)
    assert_allclose(fl.section, section)
    assert_allclose(fl.area_m2, area_m2.sum())
    assert_allclose(fl.volume_m3, vol_m3.sum())

    # We set something and everything stays same
    fl.thick = thick
    assert_allclose(fl.thick, thick)
    assert_allclose(fl.surface_h, surface_h)
    assert_allclose(fl.widths, widths)
    assert_allclose(fl.widths_m, widths_m)
    assert_allclose(fl.section, section)
    assert_allclose(fl.area_m2, area_m2.sum())
    assert_allclose(fl.volume_m3, vol_m3.sum())
    fl.section = section
    assert_allclose(fl.thick, thick)
    assert_allclose(fl.surface_h, surface_h)
    assert_allclose(fl.widths, widths)
    assert_allclose(fl.widths_m, widths_m)
    assert_allclose(fl.section, section)
    assert_allclose(fl.area_m2, area_m2.sum())
    assert_allclose(fl.volume_m3, vol_m3.sum())
    fl.surface_h = surface_h
    assert_allclose(fl.thick, thick)
    assert_allclose(fl.surface_h, surface_h)
    assert_allclose(fl.widths, widths)
    assert_allclose(fl.widths_m, widths_m)
    assert_allclose(fl.section, section)
    assert_allclose(fl.area_m2, area_m2.sum())
    assert_allclose(fl.volume_m3, vol_m3.sum())

    if fl_model is RectangularBedFlowline:
        # More adventurous
        fl.section = section / 2
        assert_allclose(fl.thick, thick / 2)
        assert_allclose(fl.widths, widths)
        assert_allclose(fl.widths_m, widths_m)
        assert_allclose(fl.section, section / 2)
        assert_allclose(fl.area_m2, area_m2.sum())
        assert_allclose(fl.volume_m3, (vol_m3 / 2).sum())


def test_mixed(fl_shape_factory):

    # Set a section and see if it all matches

    lambdas = np.ones(MODEL_LEN, dtype=np.float)
    lambdas[0:50] = 0

    (rec1, expects) = fl_shape_factory(
        TrapezoidalBedFlowline, trapeze_expects, lambdas)
    (thick, surface_h, widths, widths_m, section_trap, *_) = expects

    (rec2, expects) = fl_shape_factory(ParabolicBedFlowline, parabola_expects)
    (thick, surface_h, widths, widths_m, section_para, *_) = expects

    is_trap = np.ones(MODEL_LEN, dtype=np.bool)
    is_trap[100:] = False

    section = section_trap.copy()
    section[~is_trap] = section_para[~is_trap]

    (rec, expects) = fl_shape_factory(MixedParabolicBedFlowline,
                                      trapeze_expects, lambdas, is_trap,
                                      section)

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


@pytest.mark.parametrize("bed", [dummy_constant_bed, dummy_width_bed,
                                 dummy_noisy_bed, dummy_bumpy_bed,
                                 dummy_parabolic_bed, dummy_trapezoidal_bed,
                                 dummy_mixed_bed])
def test_flowline_to_dataset(bed):
    fl = bed()[0]
    ds = fl.to_dataset()
    fl_ = flowline_from_dataset(ds)
    ds_ = fl_.to_dataset()
    assert ds_.equals(ds)


def test_model_to_file(case_dir, hef_gdir, io_model_factory):
    init_present_time_glacier(hef_gdir)

    p = os.path.join(case_dir, 'grp.nc')
    if os.path.isfile(p):
        os.remove(p)

    fls = dummy_width_bed_tributary()
    model = FluxBasedModel(fls)
    model.to_netcdf(p)
    fls_ = glacier_from_netcdf(p)

    for fl, fl_ in zip(fls, fls_):
        ds = fl.to_dataset()
        ds_ = fl_.to_dataset()
        assert ds_.equals(ds)

    assert fls_[0].flows_to is fls_[1]
    assert fls[0].flows_to_indice == fls_[0].flows_to_indice

    # They should be sorted
    to_test = [fl.order for fl in fls_]
    assert np.array_equal(np.sort(to_test), to_test)

    # They should be able to start a run
    mb = LinearMassBalance(2600.)
    model = io_model_factory(fls_, mb_model=mb)
    model.run_until(100)


@pytest.mark.slow
def test_run(case_dir, hef_gdir, io_model_factory):
    init_present_time_glacier(hef_gdir)

    mb = LinearMassBalance(2600.)

    fls = dummy_constant_bed()
    model = io_model_factory(fls, mb_model=mb)
    ds, ds_diag = model.run_until_and_store(500, store_monthly_step=True)
    ds = ds[0]

    fls = dummy_constant_bed()
    model = io_model_factory(fls, mb_model=mb)

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
    run_path = os.path.join(case_dir, 'ts_ideal.nc')
    diag_path = os.path.join(case_dir, 'ts_diag.nc')
    if os.path.exists(run_path):
        os.remove(run_path)
    if os.path.exists(diag_path):
        os.remove(diag_path)
    model = io_model_factory(fls, mb_model=mb)
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
        model = io_model_factory(fls, mb_model=mb)
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
        model = io_model_factory(fmodel.fls, mb_model=mb, y0=300)
        model.run_until(500)
        fmodel.run_until(500)
        np.testing.assert_allclose(model.fls[0].section,
                                   fmodel.fls[0].section)


@pytest.mark.slow
def test_run_annual_step(case_dir, hef_gdir, io_model_factory):
    init_present_time_glacier(hef_gdir)

    mb = LinearMassBalance(2600.)

    fls = dummy_constant_bed()
    model = io_model_factory(fls, mb_model=mb)
    ds, ds_diag = model.run_until_and_store(500)
    ds = ds[0]

    fls = dummy_constant_bed()
    model = io_model_factory(fls, mb_model=mb)

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
    run_path = os.path.join(case_dir, 'ts_ideal.nc')
    diag_path = os.path.join(case_dir, 'ts_diag.nc')
    if os.path.exists(run_path):
        os.remove(run_path)
    if os.path.exists(diag_path):
        os.remove(diag_path)
    model = io_model_factory(fls, mb_model=mb)
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
        model = io_model_factory(fls, mb_model=mb)
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
        model = io_model_factory(fmodel.fls, mb_model=mb, y0=300)
        model.run_until(500)
        fmodel.run_until(500)
        np.testing.assert_allclose(model.fls[0].section,
                                   fmodel.fls[0].section)


def test_gdir_copy(case_dir, hef_gdir):
    new_dir = os.path.join(case_dir, 'tmp_testcopy')
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    new_gdir = tasks.copy_to_basedir(hef_gdir, base_dir=new_dir,
                                     setup='all')
    init_present_time_glacier(new_gdir)
    shutil.rmtree(new_dir)

    new_gdir = tasks.copy_to_basedir(hef_gdir, base_dir=new_dir,
                                     setup='run')
    run_random_climate(new_gdir, nyears=10)
    shutil.rmtree(new_dir)

    new_gdir = tasks.copy_to_basedir(hef_gdir, base_dir=new_dir,
                                     setup='inversion')
    inversion.prepare_for_inversion(new_gdir, invert_all_rectangular=True)
    inversion.mass_conservation_inversion(new_gdir)
    inversion.filter_inversion_output(new_gdir)
    init_present_time_glacier(new_gdir)
    run_constant_climate(new_gdir, nyears=10, bias=0)
    shutil.rmtree(new_dir)


def test_hef(case_dir, hef_gdir):
    p = os.path.join(case_dir, 'grp_hef.nc')
    if os.path.isfile(p):
        os.remove(p)

    init_present_time_glacier(hef_gdir)

    fls = hef_gdir.read_pickle('model_flowlines')
    model = FluxBasedModel(fls)

    model.to_netcdf(p)
    fls_ = glacier_from_netcdf(p)

    for fl, fl_ in zip(fls, fls_):
        ds = fl.to_dataset()
        ds_ = fl_.to_dataset()
        for v in ds.variables.keys():
            np.testing.assert_allclose(ds_[v], ds[v], equal_nan=True)

    for fl, fl_ in zip(fls[:-1], fls_[:-1]):
        assert fl.flows_to_indice == fl_.flows_to_indice

    # mixed flowline
    fls = hef_gdir.read_pickle('model_flowlines')
    model = FluxBasedModel(fls)

    p = os.path.join(case_dir, 'grp_hef_mix.nc')
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
        assert fl.flows_to_indice == fl_.flows_to_indice


@pytest.mark.slow
def test_iterative_back(backwards_idealized_fls, backwards_model_factory):

    # This test could be deleted
    from oggm.sandbox.ideas import _find_inital_glacier

    y0 = 0.
    y1 = 150.
    rtol = 0.02

    model = backwards_model_factory(
        backwards_idealized_fls, ela_delta=50., time_stepping='ambitious')

    ite, bias, past_model = _find_inital_glacier(model, model.mb_model, y0,
                                                 y1, rtol=rtol)

    bef_fls = copy.deepcopy(past_model.fls)
    past_model.run_until(y1)
    assert bef_fls[-1].area_m2 > past_model.area_m2
    np.testing.assert_allclose(past_model.area_m2,
                               backwards_idealized_fls[-1].area_m2,
                               rtol=rtol)

    if do_plot:  # pragma: no cover
        plt.plot(backwards_idealized_fls[-1].surface_h, 'k', label='ref')
        plt.plot(bef_fls[-1].surface_h, 'b', label='start')
        plt.plot(past_model.fls[-1].surface_h, 'r', label='end')
        plt.plot(backwards_idealized_fls[-1].bed_h, 'gray', linewidth=2)
        plt.legend(loc='best')
        plt.show()

    model = backwards_model_factory(backwards_idealized_fls, ela_delta=-50.,
                                    y0=y0, time_stepping='ambitious')

    ite, bias, past_model = _find_inital_glacier(model, model.mb_model, y0,
                                                 y1, rtol=rtol)
    bef_fls = copy.deepcopy(past_model.fls)
    past_model.run_until(y1)
    assert bef_fls[-1].area_m2 < past_model.area_m2
    np.testing.assert_allclose(past_model.area_m2,
                               backwards_idealized_fls[-1].area_m2,
                               rtol=rtol)

    if do_plot:  # pragma: no cover
        plt.plot(backwards_idealized_fls[-1].surface_h, 'k', label='ref')
        plt.plot(bef_fls[-1].surface_h, 'b', label='start')
        plt.plot(past_model.fls[-1].surface_h, 'r', label='end')
        plt.plot(backwards_idealized_fls[-1].bed_h, 'gray', linewidth=2)
        plt.legend(loc='best')
        plt.show()

    model = backwards_model_factory(backwards_idealized_fls, y0=y0)

    # Hit the correct one
    ite, bias, past_model = _find_inital_glacier(model, model.mb_model, y0,
                                                 y1, rtol=rtol)
    past_model.run_until(y1)
    np.testing.assert_allclose(past_model.area_m2,
                               backwards_idealized_fls[-1].area_m2,
                               rtol=rtol)


@pytest.mark.slow
def test_fails(backwards_idealized_fls, backwards_model_factory):
    # This test could be deleted
    from oggm.sandbox.ideas import _find_inital_glacier

    y0 = 0.
    y1 = 100.

    model = backwards_model_factory(
        backwards_idealized_fls, ela_delta=-150., y0=y0)
    with pytest.raises(RuntimeError):
        _find_inital_glacier(model, model.mb_model, y0,
                             y1, rtol=0.02, max_ite=5)


def simple_plot(model, gdir):  # pragma: no cover
    ocls = gdir.read_pickle('inversion_output')
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


def double_plot(model, gdir):  # pragma: no cover
    ocls = gdir.read_pickle('inversion_output')
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


def test_inversion_rectangular(inversion_gdir):

    fls = dummy_constant_bed(map_dx=inversion_gdir.grid.dx, widths=10)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir, add_debug_var=True)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.01)

    # Equations
    mb_on_z = mb.get_annual_mb(fl.surface_h[pg])
    flux = np.cumsum(fl.widths_m[pg] * fl.dx_meter * mb_on_z)

    inv_out = inversion_gdir.read_pickle('inversion_input')
    inv_flux = inv_out[0]['flux']

    slope = - np.gradient(fl.surface_h[pg], fl.dx_meter)

    est_h = inversion.sia_thickness(slope, fl.widths_m[pg], flux)
    mod_h = fl.thick[pg]

    # Test in the middle where slope is not too important
    assert_allclose(est_h[25:75], mod_h[25:75], rtol=0.01)

    # OGGM internal flux
    est_h_ofl = inversion.sia_thickness(slope, fl.widths_m[pg], inv_flux)

    # Test in the middle where slope is not too important
    assert_allclose(est_h[25:75], mod_h[25:75], rtol=0.01)

    # OK so what's happening here is following: the flux computed in
    # OGGM intern is less good than the real one with the real MB,
    # because of the zero-flux assumption at the last grid-point
    # so this RMSD is smaller:
    assert (utils.rmsd(est_h[25:95], mod_h[25:95]) <
            utils.rmsd(est_h_ofl[25:95], mod_h[25:95]))

    # And with our current inversion?
    inv_out = inversion_gdir.read_pickle('inversion_output')
    our_h = inv_out[0]['thick']
    assert_allclose(est_h[25:75], our_h[25:75], rtol=0.01)

    # Check with scalars
    assert inversion.sia_thickness(slope[-5], fl.widths_m[pg][-5],
                                   inv_flux[-5]) > 1

    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)


def test_inversion_parabolic(inversion_gdir):

    fls = dummy_parabolic_bed(map_dx=inversion_gdir.grid.dx)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir, add_debug_var=True)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)
    assert_allclose(v, model.volume_m3, rtol=0.01)

    inv = inversion_gdir.read_pickle('inversion_output')[-1]
    bed_shape_gl = 4 * inv['thick'] / \
        (flo.widths * inversion_gdir.grid.dx) ** 2
    bed_shape_ref = (4 * fl.thick[pg] /
                     (flo.widths * inversion_gdir.grid.dx) ** 2)

    # Equations
    mb_on_z = mb.get_annual_mb(fl.surface_h[pg])
    flux = np.cumsum(fl.widths_m[pg] * fl.dx_meter * mb_on_z)

    inv_out = inversion_gdir.read_pickle('inversion_input')
    inv_flux = inv_out[0]['flux']

    slope = - np.gradient(fl.surface_h[pg], fl.dx_meter)

    est_h = inversion.sia_thickness(slope, fl.widths_m[pg], flux,
                                    shape='parabolic')
    mod_h = fl.thick[pg]

    # Test in the middle where slope is not too important
    assert_allclose(est_h[25:75], mod_h[25:75], rtol=0.01)

    # OGGM internal flux
    est_h_ofl = inversion.sia_thickness(slope, fl.widths_m[pg], inv_flux)

    # Test in the middle where slope is not too important
    assert_allclose(est_h[25:75], mod_h[25:75], rtol=0.01)

    # OK so what's happening here is following: the flux computed in
    # OGGM intern is less good than the real one with the real MB,
    # because of the zero-flux assumption at the last grid-point
    # so this RMSD is smaller:
    assert (utils.rmsd(est_h[25:95], mod_h[25:95]) <
            utils.rmsd(est_h_ofl[25:95], mod_h[25:95]))

    # And with our current inversion?
    inv_out = inversion_gdir.read_pickle('inversion_output')
    our_h = inv_out[0]['thick']
    assert_allclose(est_h[25:75], our_h[25:75], rtol=0.01)

    # assert utils.rmsd(fl.bed_shape[pg], bed_shape_gl) < 0.001
    if do_plot:  # pragma: no cover
        plt.plot(bed_shape_ref[:-3])
        plt.plot(bed_shape_gl[:-3])
        plt.show()


def test_inversion_parabolic_sf_adhikari(inversion_gdir):
    old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
    old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Adhikari'
    cfg.PARAMS['use_shape_factor_for_inversion'] = 'Adhikari'

    fls = dummy_parabolic_bed(map_dx=inversion_gdir.grid.dx)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)
    assert_allclose(v, model.volume_m3, rtol=0.02)

    inv = inversion_gdir.read_pickle('inversion_output')[-1]
    bed_shape_gl = 4 * inv['thick'] / \
        (flo.widths * inversion_gdir.grid.dx) ** 2
    bed_shape_ref = (4 * fl.thick[pg] /
                     (flo.widths * inversion_gdir.grid.dx) ** 2)

    # assert utils.rmsd(fl.bed_shape[pg], bed_shape_gl) < 0.001
    if do_plot:  # pragma: no cover
        plt.plot(bed_shape_ref[:-3])
        plt.plot(bed_shape_gl[:-3])
        plt.show()

    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
    cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    # Equations
    mb_on_z = mb.get_annual_mb(fl.surface_h[pg])
    flux = np.cumsum(fl.widths_m[pg] * fl.dx_meter * mb_on_z)

    slope = - np.gradient(fl.surface_h[pg], fl.dx_meter)

    est_h = inversion.sia_thickness(slope, fl.widths_m[pg], flux,
                                    shape='parabolic',
                                    shape_factor='Adhikari')
    mod_h = fl.thick[pg]

    # Test in the middle where slope is not too important
    assert_allclose(est_h[25:75], mod_h[25:75], rtol=0.01)


def test_inversion_parabolic_sf_huss(inversion_gdir):
    old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
    old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Huss'
    cfg.PARAMS['use_shape_factor_for_inversion'] = 'Huss'

    fls = dummy_parabolic_bed(map_dx=inversion_gdir.grid.dx)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)
    assert_allclose(v, model.volume_m3, rtol=0.01)

    inv = inversion_gdir.read_pickle('inversion_output')[-1]
    bed_shape_gl = 4 * inv['thick'] / \
        (flo.widths * inversion_gdir.grid.dx) ** 2
    bed_shape_ref = (4 * fl.thick[pg] /
                     (flo.widths * inversion_gdir.grid.dx) ** 2)

    # assert utils.rmsd(fl.bed_shape[pg], bed_shape_gl) < 0.001
    if do_plot:  # pragma: no cover
        plt.plot(bed_shape_ref[:-3])
        plt.plot(bed_shape_gl[:-3])
        plt.show()

    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
    cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf

    # Equations
    mb_on_z = mb.get_annual_mb(fl.surface_h[pg])
    flux = np.cumsum(fl.widths_m[pg] * fl.dx_meter * mb_on_z)

    slope = - np.gradient(fl.surface_h[pg], fl.dx_meter)

    est_h = inversion.sia_thickness(slope, fl.widths_m[pg], flux,
                                    shape='parabolic',
                                    shape_factor='Huss')
    mod_h = fl.thick[pg]

    # Test in the middle where slope is not too important
    assert_allclose(est_h[25:75], mod_h[25:75], rtol=0.01)


@pytest.mark.slow
def test_inversion_mixed(inversion_gdir):

    fls = dummy_mixed_bed(deflambdas=0, map_dx=inversion_gdir.grid.dx,
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.05)
    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)


@pytest.mark.slow
def test_inversion_cliff(inversion_gdir):

    fls = dummy_constant_bed_cliff(map_dx=inversion_gdir.grid.dx,
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.05)
    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)


@pytest.mark.slow
def test_inversion_cliff_sf_adhikari(inversion_gdir):
    old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
    old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Adhikari'
    cfg.PARAMS['use_shape_factor_for_inversion'] = 'Adhikari'

    fls = dummy_constant_bed_cliff(map_dx=inversion_gdir.grid.dx,
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.05)
    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)

    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
    cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf


@pytest.mark.slow
def test_inversion_cliff_sf_huss(inversion_gdir):
    old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
    old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Huss'
    cfg.PARAMS['use_shape_factor_for_inversion'] = 'Huss'

    fls = dummy_constant_bed_cliff(map_dx=inversion_gdir.grid.dx,
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.05)
    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)

    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
    cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf


def test_inversion_noisy(inversion_gdir):

    fls = dummy_noisy_bed(map_dx=inversion_gdir.grid.dx)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.05)
    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)


def test_inversion_noisy_sf_adhikari(inversion_gdir):
    old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
    old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Adhikari'
    cfg.PARAMS['use_shape_factor_for_inversion'] = 'Adhikari'

    fls = dummy_noisy_bed(map_dx=inversion_gdir.grid.dx)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.05)
    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)

    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
    cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf


@pytest.mark.slow
def test_inversion_noisy_sf_huss(inversion_gdir):
    old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
    old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Huss'
    cfg.PARAMS['use_shape_factor_for_inversion'] = 'Huss'

    fls = dummy_noisy_bed(map_dx=inversion_gdir.grid.dx)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.05)
    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)

    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
    cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf


def test_inversion_tributary(inversion_gdir):

    fls = dummy_width_bed_tributary(map_dx=inversion_gdir.grid.dx)
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

    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.02)
    if do_plot:  # pragma: no cover
        double_plot(model, inversion_gdir)


def test_inversion_tributary_sf_adhikari(inversion_gdir):
    old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
    old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Adhikari'
    cfg.PARAMS['use_shape_factor_for_inversion'] = 'Adhikari'

    fls = dummy_width_bed_tributary(map_dx=inversion_gdir.grid.dx)
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

    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.02)
    if do_plot:  # pragma: no cover
        double_plot(model, inversion_gdir)

    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
    cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf


@pytest.mark.slow
def test_inversion_tributary_sf_huss(inversion_gdir):
    old_model_sf = cfg.PARAMS['use_shape_factor_for_fluxbasedmodel']
    old_inversion_sf = cfg.PARAMS['use_shape_factor_for_inversion']
    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = 'Huss'
    cfg.PARAMS['use_shape_factor_for_inversion'] = 'Huss'

    fls = dummy_width_bed_tributary(map_dx=inversion_gdir.grid.dx)
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

    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.02)
    if do_plot:  # pragma: no cover
        double_plot(model, inversion_gdir)

    cfg.PARAMS['use_shape_factor_for_fluxbasedmodel'] = old_model_sf
    cfg.PARAMS['use_shape_factor_for_inversion'] = old_inversion_sf


def test_inversion_non_equilibrium(inversion_gdir):

    fls = dummy_constant_bed(map_dx=inversion_gdir.grid.dx)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    # expected errors
    assert v > model.volume_m3
    ocls = inversion_gdir.read_pickle('inversion_output')
    ithick = ocls[0]['thick']
    assert np.mean(ithick) > np.mean(model.fls[0].thick) * 1.1
    if do_plot:  # pragma: no cover
        simple_plot(model, inversion_gdir)


def test_inversion_and_run(inversion_gdir):

    fls = dummy_parabolic_bed(map_dx=inversion_gdir.grid.dx)
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
    inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

    climate.apparent_mb_from_linear_mb(inversion_gdir)
    inversion.prepare_for_inversion(inversion_gdir)
    v, _ = inversion.mass_conservation_inversion(inversion_gdir)

    assert_allclose(v, model.volume_m3, rtol=0.01)

    inv = inversion_gdir.read_pickle('inversion_output')[-1]
    bed_shape_gl = 4 * inv['thick'] / \
        (flo.widths * inversion_gdir.grid.dx) ** 2

    ithick = inv['thick']
    fls = dummy_parabolic_bed(map_dx=inversion_gdir.grid.dx,
                              from_other_shape=bed_shape_gl[:-2],
                              from_other_bed=sh - ithick)
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


@pytest.mark.slow
def test_equilibrium(hef_gdir, inversion_params):

    init_present_time_glacier(hef_gdir)

    mb_mod = massbalance.ConstantMassBalance(hef_gdir)

    fls = hef_gdir.read_pickle('model_flowlines')
    model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                           fs=inversion_params['fs'],
                           glen_a=inversion_params['glen_a'],
                           min_dt=SEC_IN_DAY / 2.,
                           mb_elev_feedback='never')

    ref_vol = model.volume_km3
    ref_area = model.area_km2
    ref_len = model.fls[-1].length_m

    np.testing.assert_allclose(ref_area, hef_gdir.rgi_area_km2, rtol=0.03)

    model.run_until_equilibrium(rate=1e-4)
    assert not model.dt_warning
    assert model.yr > 50
    after_vol = model.volume_km3
    after_area = model.area_km2
    after_len = model.fls[-1].length_m

    np.testing.assert_allclose(ref_vol, after_vol, rtol=0.1)
    np.testing.assert_allclose(ref_area, after_area, rtol=0.03)
    np.testing.assert_allclose(ref_len, after_len, atol=500.01)


@pytest.mark.slow
def test_equilibrium_glacier_wide(hef_gdir, inversion_params):

    init_present_time_glacier(hef_gdir)

    cl = massbalance.ConstantMassBalance
    mb_mod = massbalance.MultipleFlowlineMassBalance(hef_gdir,
                                                     mb_model_class=cl)

    fls = hef_gdir.read_pickle('model_flowlines')
    model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                           fs=inversion_params['fs'],
                           glen_a=inversion_params['glen_a'],
                           min_dt=SEC_IN_DAY / 2.,
                           mb_elev_feedback='never')

    ref_vol = model.volume_km3
    ref_area = model.area_km2
    ref_len = model.fls[-1].length_m

    np.testing.assert_allclose(ref_area, hef_gdir.rgi_area_km2, rtol=0.03)

    model.run_until_equilibrium(rate=1e-4)
    assert not model.dt_warning
    assert model.yr > 50
    after_vol = model.volume_km3
    after_area = model.area_km2
    after_len = model.fls[-1].length_m

    np.testing.assert_allclose(ref_vol, after_vol, rtol=0.1)
    np.testing.assert_allclose(ref_area, after_area, rtol=0.03)
    np.testing.assert_allclose(ref_len, after_len, atol=500.01)


@pytest.mark.slow
def test_commitment(hef_gdir, inversion_params):

    init_present_time_glacier(hef_gdir)

    mb_mod = massbalance.ConstantMassBalance(hef_gdir, y0=2003 - 15)

    fls = hef_gdir.read_pickle('model_flowlines')
    model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                           fs=inversion_params['fs'],
                           glen_a=inversion_params['glen_a'])

    ref_area = model.area_km2
    np.testing.assert_allclose(ref_area, hef_gdir.rgi_area_km2, rtol=0.02)

    model.run_until_equilibrium()
    assert model.yr > 100

    after_vol_1 = model.volume_km3

    _tmp = cfg.PARAMS['mixed_min_shape']
    cfg.PARAMS['mixed_min_shape'] = 0.001
    init_present_time_glacier(hef_gdir)
    cfg.PARAMS['mixed_min_shape'] = _tmp

    glacier = hef_gdir.read_pickle('model_flowlines')

    fls = hef_gdir.read_pickle('model_flowlines')
    model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                           fs=inversion_params['fs'],
                           glen_a=inversion_params['glen_a'])

    ref_vol = model.volume_km3
    ref_area = model.area_km2
    np.testing.assert_allclose(ref_area, hef_gdir.rgi_area_km2, rtol=0.02)

    model.run_until_equilibrium()
    assert model.yr > 100

    after_vol_2 = model.volume_km3

    assert after_vol_1 < (0.5 * ref_vol)
    assert after_vol_2 < (0.5 * ref_vol)

    if do_plot:  # pragma: no cover
        plt.figure()
        plt.plot(glacier[-1].surface_h, 'b', label='start')
        plt.plot(model.fls[-1].surface_h, 'r', label='end')

        plt.plot(glacier[-1].bed_h, 'gray', linewidth=2)
        plt.legend(loc='best')
        plt.show()


@pytest.mark.slow
def test_random(hef_gdir, inversion_params):

    init_present_time_glacier(hef_gdir)
    run_random_climate(hef_gdir, nyears=100, seed=6,
                       fs=inversion_params['fs'],
                       glen_a=inversion_params['glen_a'],
                       bias=0, output_filesuffix='_rdn')
    run_constant_climate(hef_gdir, nyears=100,
                         fs=inversion_params['fs'],
                         glen_a=inversion_params['glen_a'],
                         bias=0, output_filesuffix='_ct')

    paths = [hef_gdir.get_filepath('model_run', filesuffix='_rdn'),
             hef_gdir.get_filepath('model_run', filesuffix='_ct'),
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


@pytest.mark.usefixtures('with_class_wd')
@pytest.mark.slow
def test_random_sh(gdir_sh, hef_gdir):

    gdir = hef_gdir
    init_present_time_glacier(gdir_sh)

    cfg.PATHS['climate_file'] = ''
    cfg.PARAMS['baseline_climate'] = 'CRU'
    cfg.PARAMS['run_mb_calibration'] = True
    cru_dir = get_demo_file('cru_ts3.23.1901.2014.tmp.dat.nc')
    cfg.PATHS['cru_dir'] = os.path.dirname(cru_dir)
    climate.process_cru_data(gdir_sh)
    climate.compute_ref_t_stars([gdir_sh])
    climate.local_t_star(gdir_sh)

    run_random_climate(gdir_sh, nyears=20, seed=4,
                       bias=0, output_filesuffix='_rdn')
    run_constant_climate(gdir_sh, nyears=20,
                         bias=0, output_filesuffix='_ct')

    paths = [gdir_sh.get_filepath('model_diagnostics', filesuffix='_rdn'),
             gdir_sh.get_filepath('model_diagnostics', filesuffix='_ct'),
             ]
    for path in paths:
        with xr.open_dataset(path) as ds:
            assert ds.calendar_month[0] == 4

    paths = [gdir_sh.get_filepath('model_run', filesuffix='_rdn'),
             gdir_sh.get_filepath('model_run', filesuffix='_ct'),
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

    # Test a SH/NH mix
    init_present_time_glacier(gdir)
    run_constant_climate(gdir, nyears=20,
                         bias=0, output_filesuffix='_ct')

    utils.compile_climate_input([gdir_sh, gdir])
    utils.compile_run_output([gdir_sh, gdir],
                             input_filesuffix='_ct')

    f = os.path.join(cfg.PATHS['working_dir'], 'run_output_ct_sh.nc')
    with xr.open_dataset(f) as ds:
        assert ds.calendar_month[0] == 4
    f = os.path.join(cfg.PATHS['working_dir'], 'run_output_ct_nh.nc')
    with xr.open_dataset(f) as ds:
        assert ds.calendar_month[0] == 10
    f = os.path.join(cfg.PATHS['working_dir'], 'climate_input_sh.nc')
    with xr.open_dataset(f) as ds:
        assert ds.calendar_month[0] == 4
    f = os.path.join(cfg.PATHS['working_dir'], 'climate_input_nh.nc')
    with xr.open_dataset(f) as ds:
        assert ds.calendar_month[0] == 10


def test_start_from_spinup(hef_gdir):

    init_present_time_glacier(hef_gdir)

    fls = hef_gdir.read_pickle('model_flowlines')
    vol = 0
    area = 0
    for fl in fls:
        vol += fl.volume_km3
        area += fl.area_km2
    assert hef_gdir.rgi_date == 2003

    # Make a dummy run for 0 years
    run_from_climate_data(hef_gdir, ye=2003, output_filesuffix='_1')

    fp = hef_gdir.get_filepath('model_run', filesuffix='_1')
    with FileModel(fp) as fmod:
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area)
        np.testing.assert_allclose(fmod.volume_km3, vol)

    # Again
    run_from_climate_data(hef_gdir, ye=2003, init_model_filesuffix='_1',
                          output_filesuffix='_2')
    fp = hef_gdir.get_filepath('model_run', filesuffix='_2')
    with FileModel(fp) as fmod:
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area)
        np.testing.assert_allclose(fmod.volume_km3, vol)


def test_start_from_spinup_min_ys(hef_gdir):

    init_present_time_glacier(hef_gdir)

    fls = hef_gdir.read_pickle('model_flowlines')
    vol = 0
    area = 0
    for fl in fls:
        vol += fl.volume_km3
        area += fl.area_km2
    assert hef_gdir.rgi_date == 2003

    # Make a dummy run for 0 years
    run_from_climate_data(hef_gdir, ye=2002, min_ys=2002,
                          output_filesuffix='_1')

    fp = hef_gdir.get_filepath('model_run', filesuffix='_1')
    with FileModel(fp) as fmod:
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area)
        np.testing.assert_allclose(fmod.volume_km3, vol)

    # Again
    run_from_climate_data(hef_gdir, ys=2002, ye=2003,
                          init_model_filesuffix='_1',
                          output_filesuffix='_2')
    fp = hef_gdir.get_filepath('model_run', filesuffix='_2')
    with FileModel(fp) as fmod:
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area, rtol=0.05)
        np.testing.assert_allclose(fmod.volume_km3, vol, rtol=0.05)


@pytest.mark.usefixtures('with_class_wd')
@pytest.mark.slow
def test_cesm(hef_gdir):

    gdir = hef_gdir

    # init
    f = get_demo_file('cesm.TREFHT.160001-200512.selection.nc')
    cfg.PATHS['cesm_temp_file'] = f
    f = get_demo_file('cesm.PRECC.160001-200512.selection.nc')
    cfg.PATHS['cesm_precc_file'] = f
    f = get_demo_file('cesm.PRECL.160001-200512.selection.nc')
    cfg.PATHS['cesm_precl_file'] = f
    gcm_climate.process_cesm_data(gdir)

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
    mb_cru = massbalance.PastMassBalance(gdir)
    mb_cesm = massbalance.PastMassBalance(gdir,
                                          filename='gcm_data')

    # Average over 1961-1990
    h, w = gdir.get_inversion_flowline_hw()
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

    ds1 = utils.compile_run_output([gdir], input_filesuffix='_hist')
    ds2 = utils.compile_run_output([gdir], input_filesuffix='_cesm')

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
                                   input_filesuffix='_afterspinup')
    assert (ds1.volume.isel(rgi_id=0, time=-1) <
            0.7 * ds3.volume.isel(rgi_id=0, time=-1))
    ds3.close()

    # Try the compile optimisation
    out = utils.compile_run_output([gdir, gdir, gdir],
                                   tmp_file_size=2,
                                   input_filesuffix='_hist',
                                   output_filesuffix='_rehist')
    assert out is None
    path = os.path.join(cfg.PATHS['working_dir'], 'run_output_rehist.nc')
    with xr.open_dataset(path) as ds:
        assert len(ds.rgi_id) == 3


@pytest.mark.slow
def test_elevation_feedback(hef_gdir):

    init_present_time_glacier(hef_gdir)

    feedbacks = ['annual', 'monthly', 'always', 'never']
    # Mutliproc
    tasks = []
    for feedback in feedbacks:
        tasks.append((run_random_climate,
                      dict(nyears=200, seed=5, mb_elev_feedback=feedback,
                           output_filesuffix=feedback,
                           store_monthly_step=True)))
    workflow.execute_parallel_tasks(hef_gdir, tasks)

    out = []
    for feedback in feedbacks:
        out.append(utils.compile_run_output([hef_gdir], path=False,
                                            input_filesuffix=feedback))

    # Check that volume isn't so different
    assert_allclose(out[0].volume, out[1].volume, rtol=0.05)
    assert_allclose(out[0].volume, out[2].volume, rtol=0.05)
    assert_allclose(out[1].volume, out[2].volume, rtol=0.05)
    # Except for "never", where things are different
    assert out[3].volume.mean() < out[2].volume.mean()

    if do_plot:
        plt.figure()
        for ds, lab in zip(out, feedbacks):
            (ds.volume * 1e-9).plot(label=lab)
        plt.xlabel('Vol (km3)')
        plt.legend()
        plt.show()


@pytest.mark.slow
def test_merged_simulation(case_dir):
    import geopandas as gpd

    # setup logging
    import logging
    log = logging.getLogger(__name__)

    # Init
    cfg.initialize()
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PARAMS['correct_for_neg_flux'] = True
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'
    # should we be resetting working_dir at teardown?
    cfg.PATHS['working_dir'] = case_dir
    cfg.PARAMS['border'] = 100
    cfg.PARAMS['prcp_scaling_factor'] = 1.75
    cfg.PARAMS['temp_melt'] = -1.75
    cfg.PARAMS['use_multiprocessing'] = False

    hef_file = utils.get_demo_file('rgi_oetztal.shp')
    rgidf = gpd.read_file(hef_file)

    # Get HEF, Vernagt1/2 and Gepatschferner
    glcdf = rgidf.loc[(rgidf.RGIId == 'RGI50-11.00897') |
                      (rgidf.RGIId == 'RGI50-11.00719_d01') |
                      (rgidf.RGIId == 'RGI50-11.00779') |
                      (rgidf.RGIId == 'RGI50-11.00746')].copy()
    gdirs = workflow.init_glacier_regions(glcdf)
    workflow.gis_prepro_tasks(gdirs)
    workflow.climate_tasks(gdirs)
    workflow.inversion_tasks(gdirs)
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    # store HEF
    hef = [gd for gd in gdirs if gd.rgi_id == 'RGI50-11.00897']

    # merge, but with 0 buffer, should not do anything
    merge0 = workflow.merge_glacier_tasks(gdirs, 'RGI50-11.00897',
                                          glcdf=glcdf, buffer=0)
    assert 'RGI50-11.00897' == np.unique([fl.rgi_id for fl in
                                          merge0.read_pickle(
                                              'model_flowlines')])[0]
    gdirs += hef

    # merge, but with 50 buffer. overlapping glaciers should be excluded
    merge1 = workflow.merge_glacier_tasks(gdirs, 'RGI50-11.00897',
                                          glcdf=glcdf, buffer=50)
    assert 'RGI50-11.00719_d01' in [fl.rgi_id for fl in
                                    merge1.read_pickle('model_flowlines')]
    assert 'RGI50-11.00779' not in [fl.rgi_id for fl in
                                    merge1.read_pickle('model_flowlines')]

    gdirs += hef

    # merge HEF and Vernagt, include Gepatsch but it should not be merged
    gdir_merged = workflow.merge_glacier_tasks(gdirs, 'RGI50-11.00897',
                                               glcdf=glcdf)

    # test flowlines
    fls = gdir_merged.read_pickle('model_flowlines')

    # check for gepatsch, should not be there
    assert 'RGI50-11.00746' not in [fl.rgi_id for fl in fls]

    # ascending order
    assert np.all(np.diff([fl.order for fl in fls]) >= 0)
    # last flowline has max order
    assert np.max([fl.order for fl in fls]) == fls[-1].order
    # first flowline hast 0 order
    assert fls[0].order == 0

    # test flows to order
    fls1 = [fl for fl in fls if fl.rgi_id == 'RGI50-11.00779']
    assert fls1[0].flows_to == fls1[-1]
    assert fls1[-1].flows_to.rgi_id == 'RGI50-11.00719_d01'
    assert fls1[-1].flows_to.flows_to.rgi_id == 'RGI50-11.00897'

    gdirs += hef

    # run parameters
    years = 200  # arbitrary
    tbias = -1.0  # arbitrary

    # run HEF and the two Vernagts as entities
    gdirs_entity = [gd for gd in gdirs if gd.rgi_id != 'RGI50-11.00746']
    workflow.execute_entity_task(tasks.run_constant_climate,
                                 gdirs_entity,
                                 nyears=years,
                                 output_filesuffix='_entity',
                                 temperature_bias=tbias)

    ds_entity = utils.compile_run_output(gdirs_entity,
                                         path=False,
                                         input_filesuffix='_entity')

    # and run the merged glacier
    workflow.execute_entity_task(tasks.run_constant_climate,
                                 gdir_merged, output_filesuffix='_merged',
                                 nyears=years,
                                 temperature_bias=tbias)

    ds_merged = utils.compile_run_output(gdir_merged,
                                         path=False,
                                         input_filesuffix='_merged')

    # areas should be quite similar after 10yrs
    assert_allclose(ds_entity.area.isel(time=10).sum(),
                    ds_merged.area.isel(time=10),
                    rtol=1e-4)

    # After 100yrs, merged one should be smaller as Vernagt1 is slightly
    # flowing into Vernagt2
    assert (ds_entity.area.isel(time=100).sum() >
            ds_merged.area.isel(time=100))

    # Merged glacier should have a larger area after 200yrs from advancing
    assert (ds_entity.area.isel(time=200).sum() <
            ds_merged.area.isel(time=200))
