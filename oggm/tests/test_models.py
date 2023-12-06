import os
from functools import partial
import shutil
import warnings
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
from oggm.core import climate, inversion, centerlines
from oggm.shop import gcm_climate, bedtopo
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import get_demo_file
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

from oggm.tests.funcs import get_test_dir
from oggm.tests.funcs import (dummy_bumpy_bed, dummy_constant_bed,
                              dummy_constant_bed_cliff,
                              dummy_mixed_bed, bu_tidewater_bed,
                              dummy_noisy_bed, dummy_parabolic_bed,
                              dummy_trapezoidal_bed, dummy_width_bed,
                              dummy_width_bed_tributary)

import matplotlib.pyplot as plt
from oggm.core.flowline import (FluxBasedModel, FlowlineModel, MassRedistributionCurveModel,
                                init_present_time_glacier, glacier_from_netcdf,
                                RectangularBedFlowline, TrapezoidalBedFlowline,
                                ParabolicBedFlowline, MixedBedFlowline,
                                flowline_from_dataset, FileModel,
                                run_constant_climate, run_random_climate,
                                run_from_climate_data, equilibrium_stop_criterion,
                                run_with_hydro, SemiImplicitModel)
from oggm.core.dynamic_spinup import (
    run_dynamic_spinup, run_dynamic_melt_f_calibration,
    dynamic_melt_f_run_with_dynamic_spinup,
    dynamic_melt_f_run_with_dynamic_spinup_fallback,
    dynamic_melt_f_run,
    dynamic_melt_f_run_fallback)

FluxBasedModel = partial(FluxBasedModel, inplace=True)
FlowlineModel = partial(FlowlineModel, inplace=True)

pytest.importorskip('geopandas')
pytest.importorskip('rasterio')
pytest.importorskip('salem')

pytestmark = pytest.mark.test_env("models")
do_plot = False

DOM_BORDER = 80

ALL_DIAGS = ['volume', 'volume_bsl', 'volume_bwl', 'area', 'length',
             'calving', 'calving_rate', 'off_area', 'on_area', 'melt_off_glacier',
             'melt_on_glacier', 'liq_prcp_off_glacier', 'liq_prcp_on_glacier',
             'snowfall_off_glacier', 'snowfall_on_glacier', 'model_mb',
             'residual_mb', 'snow_bucket']

has_shapely2 = False
try:
    import shapely.io
    has_shapely2 = True
except ImportError:
    pass


class TestInitPresentDayFlowline:

    @pytest.mark.parametrize('downstream_line_shape', ['parabola', 'trapezoidal'])
    def test_init_present_time_glacier(self, hef_gdir, downstream_line_shape):

        gdir = hef_gdir
        cfg.PARAMS['downstream_line_shape'] = downstream_line_shape
        init_present_time_glacier(gdir)

        fls = gdir.read_pickle('model_flowlines')
        inv = gdir.read_pickle('inversion_output')

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
            assert (len(fl.surface_h) ==
                    len(fl.bed_h) ==
                    len(fl.bed_shape) ==
                    len(fl.dis_on_line) ==
                    len(fl.widths) ==
                    len(fl.bin_area_m2)
                    )

            assert np.all(fl.widths >= 0)
            vol += fl.volume_m3
            area += fl.area_km2

        # New loc stuff - checked with google maps
        lons = fl.point_lons
        lats = fl.point_lats
        assert_allclose([lons[0], lats[0]], [10.7470, 46.8048], atol=1e-4)
        assert_allclose([lons[-1], lats[-1]], [10.8551, 46.8376], atol=1e-4)

        # Diags
        ref_vol = 0.
        ref_area = 0.
        for cl in inv:
            ref_vol += np.sum(cl['volume'])
            ref_area += np.sum(cl['width'] * fl.dx_meter)

        np.testing.assert_allclose(ref_vol, vol)
        np.testing.assert_allclose(6900.0, fls[-1].length_m, atol=101)
        np.testing.assert_allclose(gdir.rgi_area_km2, ref_area * 1e-6)
        np.testing.assert_allclose(gdir.rgi_area_km2, area)

        # test that final downstream line has the desired shape
        fl = fls[-1]
        ice_mask = fls[-1].thick > 0.
        if downstream_line_shape == 'parabola':
            assert np.all(np.isfinite(fl.bed_shape[~ice_mask]))
            assert np.all(~np.isfinite(fl._w0_m[~ice_mask]))
        if downstream_line_shape == 'trapezoidal':
            assert np.all(np.isfinite(fl._w0_m[~ice_mask]))
            assert np.all(~np.isfinite(fl.bed_shape[~ice_mask]))
            # check that bottom width of downstream line is larger than minimum
            assert np.all(fl._w0_m[~ice_mask] >
                          cfg.PARAMS['trapezoid_min_bottom_width'])

        if do_plot:
            plt.plot(fls[-1].bed_h, color='k')
            plt.plot(fls[-1].surface_h)
            plt.figure()
            plt.plot(fls[-1].surface_h - fls[-1].bed_h)
            plt.show()

        # test if providing a filesuffix is working
        init_present_time_glacier(gdir, filesuffix='_test')
        assert os.path.isfile(os.path.join(gdir.dir, 'model_flowlines_test.pkl'))

        cfg.PARAMS['downstream_line_shape'] = 'free_shape'
        with pytest.raises(InvalidParamsError):
            init_present_time_glacier(gdir)

    def test_init_present_time_glacier_obs_thick(self, hef_elev_gdir,
                                                 monkeypatch):

        gdir = hef_elev_gdir

        # need to change rgi_id, which is needed to be different in other tests
        # when comparing to centerlines
        gdir.rgi_id = 'RGI60-11.00897'

        # add some thickness data
        ft = utils.get_demo_file('RGI60-11.00897_thickness.tif')
        monkeypatch.setattr(utils, 'file_downloader', lambda x: ft)
        bedtopo.add_consensus_thickness(gdir)
        vn = 'consensus_ice_thickness'
        centerlines.elevation_band_flowline(gdir, bin_variables=[vn])
        centerlines.fixed_dx_elevation_band_flowline(gdir,
                                                     bin_variables=[vn])

        tasks.init_present_time_glacier(gdir, filesuffix='_consensus',
                                        use_binned_thickness_data=vn)
        fl_consensus = gdir.read_pickle('model_flowlines',
                                        filesuffix='_consensus')[0]

        # check that resulting flowline has the same volume as observation
        cdf = pd.read_hdf(utils.get_demo_file('rgi62_itmix_df.h5'))
        ref_vol = cdf.loc[gdir.rgi_id].vol_itmix_m3
        np.testing.assert_allclose(fl_consensus.volume_m3, ref_vol)

        # should be trapezoid where ice
        assert np.all(fl_consensus.is_trapezoid[fl_consensus.thick > 0])

        # test that we can use fl in an dynamic model run without an error
        mb = LinearMassBalance(3000.)
        model_ref = FluxBasedModel(gdir.read_pickle('model_flowlines'),
                                   mb_model=mb)
        model_ref.run_until(100)
        model_consensus = FluxBasedModel([fl_consensus], mb_model=mb)
        model_consensus.run_until(100)
        np.testing.assert_allclose(model_ref.volume_km3,
                                   model_consensus.volume_km3,
                                   atol=0.02)

        # test that if w0<0 it is converted to rectangular
        # set some thickness to very large values to force it
        df_fixed_dx = pd.read_csv(gdir.get_filepath('elevation_band_flowline',
                                                    filesuffix='_fixed_dx'))
        new_thick = df_fixed_dx['consensus_ice_thickness']
        new_thick[-10:] = new_thick[-10:] + 1000
        df_fixed_dx['consensus_ice_thickness'] = new_thick
        ref_vol_rect = np.sum(df_fixed_dx['area_m2'] * new_thick)
        df_fixed_dx.to_csv(gdir.get_filepath('elevation_band_flowline',
                                             filesuffix='_fixed_dx'))

        tasks.init_present_time_glacier(gdir, filesuffix='_consensus_rect',
                                        use_binned_thickness_data=vn)
        fl_consensus_rect = gdir.read_pickle('model_flowlines',
                                             filesuffix='_consensus_rect')[0]

        np.testing.assert_allclose(fl_consensus_rect.volume_m3, ref_vol_rect)
        assert np.sum(fl_consensus_rect.is_rectangular) == 10

    def test_present_time_glacier_massbalance(self, hef_gdir):

        gdir = hef_gdir
        init_present_time_glacier(gdir)

        mb_mod = massbalance.MonthlyTIModel(gdir)

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

        mb_profile_constant_dh = gdir.get_ref_mb_profile(constant_dh=True)
        step_dh = mb_profile_constant_dh.columns[1:] - mb_profile_constant_dh.columns[:-1]
        assert np.all(step_dh == 50)
        mb_profile_raw = gdir.get_ref_mb_profile()

        mb_profile_constant_dh_filtered_0_6 = gdir.get_ref_mb_profile(constant_dh=True,
                                                                      obs_ratio_needed=0.6)
        mb_profile_constant_dh_filtered_1 = gdir.get_ref_mb_profile(constant_dh=True,
                                                                    obs_ratio_needed=1)
        n = len(mb_profile_constant_dh.index)
        n_obs_h_0_6 = mb_profile_constant_dh_filtered_0_6.describe().loc['count']
        n_obs_h_1 = mb_profile_constant_dh_filtered_1.describe().loc['count']
        assert np.all(n_obs_h_0_6 / n >= 0.6)
        assert np.all(n_obs_h_1 / n >= 1)

        # fake "filter" that is not really filtering should give
        # the same estimate as default (filter = 0)
        mb_profile_constant_dh_filtered_0 = gdir.get_ref_mb_profile(constant_dh=True,
                                                                    obs_ratio_needed=0.00001)
        np.testing.assert_allclose(mb_profile_constant_dh_filtered_0, mb_profile_constant_dh)

        # Gradient
        dfg = mb_profile_raw.mean()
        dfg_constant_dh = mb_profile_constant_dh.mean()
        dfg_constant_dh_0_6 = mb_profile_constant_dh_filtered_0_6.mean()

        # Take the altitudes below 3100 and fit a line
        pok = np.where(hgts < 3100)
        dfg = dfg[dfg.index < 3100]
        dfg_constant_dh = dfg_constant_dh[dfg_constant_dh.index < 3100]
        dfg_constant_dh_0_6 = dfg_constant_dh_0_6[dfg_constant_dh_0_6.index < 3100]

        from scipy.stats import linregress
        slope_obs, _, _, _, _ = linregress(dfg.index, dfg.values)
        slope_obs_constant_dh, _, _, _, _ = linregress(dfg_constant_dh.index,
                                                       dfg_constant_dh.values)
        slope_obs_constant_dh_0_6, _, _, _, _ = linregress(dfg_constant_dh_0_6.index,
                                                           dfg_constant_dh_0_6.values)
        slope_our, _, _, _, _ = linregress(hgts[pok], grads[pok])

        np.testing.assert_allclose(slope_obs, slope_our, rtol=0.15)
        # the observed MB gradient and the interpolated observed one (with
        # constant dh) should be very similar!
        np.testing.assert_allclose(slope_obs, slope_obs_constant_dh, rtol=0.01)
        # the filtered slope with obs_ratio_0_6 should be at least a bit similar to
        # the one where all elevation band measurements are taken into account
        np.testing.assert_allclose(slope_obs_constant_dh, slope_obs_constant_dh_0_6, rtol=0.2)


@pytest.fixture(scope='class')
def other_glacier_cfg():
    cfg.initialize()
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False
    cfg.PARAMS['prcp_fac'] = 2.5
    cfg.PARAMS['baseline_climate'] = 'CRU'


@pytest.mark.usefixtures('other_glacier_cfg')
class TestInitFlowlineOtherGlacier:
    def test_define_divides(self, class_case_dir):

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
        gdir = GlacierDirectory(entity, base_dir=class_case_dir)
        gis.define_glacier_region(gdir)
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
        ref_period = '1980-01-01_2000-01-01'
        ref_mb = -500
        massbalance.mb_calibration_from_scalar_mb(gdir,
                                                  ref_mb=ref_mb,
                                                  ref_period=ref_period)
        massbalance.apparent_mb_from_any_mb(gdir, mb_years=(1980, 2000))
        inversion.prepare_for_inversion(gdir)
        v = inversion.mass_conservation_inversion(gdir)
        init_present_time_glacier(gdir)

        myarea = 0.
        cls = gdir.read_pickle('inversion_flowlines')
        for cl in cls:
            myarea += np.sum(cl.widths * cl.dx * gdir.grid.dx ** 2)

        np.testing.assert_allclose(myarea, gdir.rgi_area_m2, rtol=1e-2)

        myarea = 0.
        cls = gdir.read_pickle('inversion_flowlines')
        for cl in cls:
            myarea += np.sum(cl.widths * cl.dx * gdir.grid.dx ** 2)

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
            assert (len(fl.surface_h) ==
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


class TestMassBalanceModels:
    def test_past_mb_model(self, hef_gdir):

        rho = cfg.PARAMS['ice_density']

        F = SEC_IN_YEAR * rho

        gdir = hef_gdir
        init_present_time_glacier(gdir)

        df = gdir.read_json('mb_calib')

        # Climate period
        yrp = [1851, 2000]

        # Flowlines height
        h, w = gdir.get_inversion_flowline_hw()

        mb_mod = massbalance.MonthlyTIModel(gdir, bias=0)
        for i, yr in enumerate(np.arange(yrp[0], yrp[1] + 1)):
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * F
            ela_z = mb_mod.get_ela(year=yr)
            totest = mb_mod.get_annual_mb([ela_z], year=yr) * F
            assert_allclose(totest[0], 0, atol=1)

        mb_mod = massbalance.MonthlyTIModel(gdir)
        for i, yr in enumerate(np.arange(yrp[0], yrp[1] + 1)):
            ela_z = mb_mod.get_ela(year=yr)
            totest = mb_mod.get_annual_mb([ela_z], year=yr) * F
            assert_allclose(totest[0], 0, atol=1)

        # real data
        h, w = gdir.get_inversion_flowline_hw()
        mbdf = gdir.get_ref_mb_data()
        mbdf.loc[yr, 'MY_MB'] = np.NaN
        mb_mod = massbalance.MonthlyTIModel(gdir)
        for yr in mbdf.index.values:
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
            mbdf.loc[yr, 'MY_MB'] = np.average(my_mb_on_h, weights=w)

        np.testing.assert_allclose(mbdf['ANNUAL_BALANCE'].mean(),
                                   mbdf['MY_MB'].mean(),
                                   atol=1e-2)
        mbdf['MY_ELA'] = mb_mod.get_ela(year=mbdf.index.values)
        assert mbdf[['MY_ELA', 'MY_MB']].corr().values[0, 1] < -0.9
        assert mbdf[['MY_ELA', 'ANNUAL_BALANCE']].corr().values[0, 1] < -0.6

        mb_mod = massbalance.MonthlyTIModel(gdir, bias=0)
        for yr in mbdf.index.values:
            my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
            mbdf.loc[yr, 'MY_MB'] = np.average(my_mb_on_h, weights=w)

        np.testing.assert_allclose(mbdf['ANNUAL_BALANCE'].mean(),
                                   mbdf['MY_MB'].mean(),
                                   atol=1e-2)

        mb_mod = massbalance.MonthlyTIModel(gdir)
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
        mb_mod = massbalance.MonthlyTIModel(gdir, repeat=True,
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

        # Test massbalance task
        s = massbalance.fixed_geometry_mass_balance(gdir)
        assert s.index[0] == 1802
        assert s.index[-1] == 2002

        s = massbalance.fixed_geometry_mass_balance(gdir, ys=1990, ye=2000)
        assert s.index[0] == 1990
        assert s.index[-1] == 2000

        s = massbalance.fixed_geometry_mass_balance(gdir,
                                                    years=mbdf.index.values)
        assert_allclose(s, mbdf['MY_MB'])

    def test_repr(self, hef_gdir):
        from textwrap import dedent

        expected = dedent("""\
        <oggm.MassBalanceModel>
          Class: MonthlyTIModel
          Attributes:
            - hemisphere: nh
            - climate_source: histalp_merged_hef.nc
            - melt_f: 6.59
            - prcp_fac: 2.50
            - temp_bias: 0.00
            - bias: 0.00
            - rho: 900.0
            - t_solid: 0.0
            - t_liq: 2.0
            - t_melt: -1.0
            - repeat: False
            - ref_hgt: 3160.0
            - ys: 1802
            - ye: 2002
        """)
        mb_mod = massbalance.MonthlyTIModel(hef_gdir, bias=0)
        assert mb_mod.__repr__() == expected

    def test_prcp_fac_temp_bias_update(self, hef_gdir):

        gdir = hef_gdir
        init_present_time_glacier(gdir)

        mb_mod = massbalance.MonthlyTIModel(gdir, bias=0)
        # save old precipitation/temperature time series
        prcp_old = mb_mod.prcp.copy()
        temp_old = mb_mod.temp.copy()
        prcp_fac_old = cfg.PARAMS['prcp_fac']
        temp_bias_old = 0
        # basic checks
        assert mb_mod.prcp_fac == prcp_fac_old
        assert mb_mod._prcp_fac == prcp_fac_old
        assert mb_mod.temp_bias == temp_bias_old

        # Now monthly stuff
        mb_mod.temp_bias = [0] * 12
        np.testing.assert_allclose(mb_mod.temp_bias, temp_bias_old)
        mb_mod.prcp_fac = [prcp_fac_old] * 12
        np.testing.assert_allclose(mb_mod.prcp_fac, prcp_fac_old)

        # increase prcp by factor of 10 and add a temperature bias of 1
        factor = 10
        mb_mod.prcp_fac = factor
        temp_bias = 1
        mb_mod.temp_bias = temp_bias
        assert mb_mod.prcp_fac == factor
        assert mb_mod._prcp_fac == factor
        assert mb_mod.temp_bias == temp_bias
        assert mb_mod._temp_bias == temp_bias
        prcp_new = mb_mod.prcp
        temp_new = mb_mod.temp
        assert_allclose(prcp_new, prcp_old * factor / prcp_fac_old)
        assert_allclose(temp_new, temp_old + temp_bias - temp_bias_old)

        # check if it gets back to the old prcp/temp time series
        mb_mod.prcp_fac = prcp_fac_old
        assert mb_mod.prcp_fac == prcp_fac_old
        assert mb_mod._prcp_fac == prcp_fac_old
        assert_allclose(mb_mod.prcp, prcp_old)

        mb_mod.temp_bias = temp_bias_old
        assert mb_mod.temp_bias == temp_bias_old
        assert mb_mod._temp_bias == temp_bias_old
        assert_allclose(mb_mod.temp, temp_old)

        # check if error occurs for invalid prcp_fac
        with pytest.raises(InvalidParamsError):
            mb_mod.prcp_fac = -100

    @pytest.mark.parametrize("cl", [massbalance.MonthlyTIModel,
                                    massbalance.ConstantMassBalance,
                                    massbalance.RandomMassBalance])
    def test_glacierwide_mb_model(self, hef_gdir, cl):
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
            kwargs = {'seed': 0, 'y0': 1985}
        elif cl is massbalance.ConstantMassBalance:
            kwargs = {'y0': 1985}
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

        mb.prcp_fac = 100
        mb_gw.prcp_fac = 100

        assert mb.prcp_fac == mb_gw.prcp_fac

        assert_allclose(mb.get_specific_mb(h, w, year=yrs[:10]),
                        mb_gw.get_specific_mb(year=yrs[:10]))

        assert_allclose(mb.get_ela(year=yrs[:10]),
                        mb_gw.get_ela(year=yrs[:10]))

        if cl is massbalance.MonthlyTIModel:
            mb = cl(gdir)
            mb_gw = massbalance.MultipleFlowlineMassBalance(gdir,
                                                            mb_model_class=cl)
            mb = massbalance.UncertainMassBalance(mb, rdn_bias_seed=1,
                                                  rdn_prcp_fac_seed=2,
                                                  rdn_temp_bias_seed=3)
            mb_gw = massbalance.UncertainMassBalance(mb_gw, rdn_bias_seed=1,
                                                     rdn_prcp_fac_seed=2,
                                                     rdn_temp_bias_seed=3)
        assert_allclose(mb.get_specific_mb(h, w, year=yrs[:30]),
                        mb_gw.get_specific_mb(fls=fls, year=yrs[:30]))

        # ELA won't pass because of API incompatibility
        # assert_allclose(mb.get_ela(year=yrs[:30]),
        #                 mb_gw.get_ela(year=yrs[:30]))

    def test_constant_mb_model(self, hef_gdir):

        rho = cfg.PARAMS['ice_density']

        gdir = hef_gdir
        init_present_time_glacier(gdir)

        h, w = gdir.get_inversion_flowline_hw()

        # We calibrate to zero
        df = massbalance.mb_calibration_from_scalar_mb(gdir,
                                                       calibrate_param1='temp_bias',
                                                       ref_mb=0,
                                                       ref_mb_years=(1970, 2001),
                                                       write_to_gdir=False)

        cmb_mod = massbalance.ConstantMassBalance(gdir,
                                                  melt_f=df['melt_f'],
                                                  temp_bias=df['temp_bias'],
                                                  prcp_fac=df['prcp_fac'],
                                                  y0=1985)
        ombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        otmb = np.average(ombh, weights=w)
        np.testing.assert_allclose(0., otmb, atol=0.2)

        mb_mod = massbalance.ConstantMassBalance(gdir, y0=2002 - 15)
        nmbh = mb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        ntmb = np.average(nmbh, weights=w)

        assert ntmb < otmb

        if do_plot:  # pragma: no cover
            plt.plot(h, ombh, 'o', label='zero')
            plt.plot(h, nmbh, 'o', label='today')
            plt.legend()
            plt.show()

        orig_bias = cmb_mod.temp_bias
        cmb_mod.temp_bias = orig_bias + 1
        biasombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        biasotmb = np.average(biasombh, weights=w)
        assert biasotmb < (otmb - 500)

        cmb_mod.temp_bias = orig_bias
        nobiasombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        nobiasotmb = np.average(nobiasombh, weights=w)
        np.testing.assert_allclose(0, nobiasotmb, atol=0.2)

        months = np.arange(12)
        monthly_1 = months * 0.
        monthly_2 = months * 0.
        monthly_3 = months * 0.
        for m in months:
            yr = utils.date_to_floatyear(0, m + 1)
            cmb_mod.temp_bias = orig_bias
            tmp = cmb_mod.get_monthly_mb(h, yr) * SEC_IN_MONTH * rho
            monthly_1[m] = np.average(tmp, weights=w)
            cmb_mod.temp_bias = orig_bias + 1
            tmp = cmb_mod.get_monthly_mb(h, yr) * SEC_IN_MONTH * rho
            monthly_2[m] = np.average(tmp, weights=w)
            cmb_mod.temp_bias = [orig_bias] * 6 + [orig_bias + 1] + [orig_bias] * 5
            cmb_mod.prcp_fac = [10] * 3 + [2.5] * 9  # This adds solid precip in win
            tmp = cmb_mod.get_monthly_mb(h, yr) * SEC_IN_MONTH * rho
            monthly_3[m] = np.average(tmp, weights=w)
            cmb_mod.prcp_fac = 2.5

        if do_plot:  # pragma: no cover
            plt.plot(monthly_1, '-', label='Normal')
            plt.plot(monthly_2, '-', label='Temp bias')
            plt.plot(monthly_3, '-', label='Temp bias monthly')
            plt.legend()
            plt.show()

        # check that the winter months are close but summer months no
        np.testing.assert_allclose(monthly_1[:4], monthly_2[:4], atol=1)
        assert np.mean(monthly_3[:4]) > (np.mean(monthly_1[:4]) + 100)
        assert monthly_3[6] == monthly_2[6]
        assert monthly_3[7] != monthly_2[7]

        # Climate info
        h = np.sort(h)
        cmb_mod = massbalance.ConstantMassBalance(gdir,
                                                  melt_f=df['melt_f'],
                                                  temp_bias=df['temp_bias'],
                                                  prcp_fac=df['prcp_fac'],
                                                  y0=1985)
        t, tm, p, ps = cmb_mod.get_annual_climate(h)

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
        t, tm, p, ps = cmb_mod.get_annual_climate([elah])
        mb = ps - cmb_mod.mbmod.monthly_melt_f * tm
        # not perfect because of time/months/zinterp issues
        np.testing.assert_allclose(mb, 0, atol=0.2)

    def test_random_mb(self, hef_gdir):

        gdir = hef_gdir
        init_present_time_glacier(gdir)

        ref_mod = massbalance.ConstantMassBalance(gdir, y0=1985)
        mb_mod = massbalance.RandomMassBalance(gdir, seed=10, y0=1985)

        h, w = gdir.get_inversion_flowline_hw()

        ref_mbh = ref_mod.get_annual_mb(h, None) * SEC_IN_YEAR

        # two years shouldn't be equal
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
        mb_ref = massbalance.MonthlyTIModel(gdir)
        mb_mod = massbalance.RandomMassBalance(gdir, y0=2002 - 15,
                                               seed=10)
        mb_ts = []
        mb_ts2 = []
        yrs = np.arange(1972, 2003, 1)
        for yr in yrs:
            mb_ts.append(np.average(mb_ref.get_annual_mb(h, yr) * SEC_IN_YEAR,
                                    weights=w))
            mb_ts2.append(np.average(mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR,
                                     weights=w))
        np.testing.assert_allclose(np.std(mb_ts), np.std(mb_ts2), rtol=0.15)

        # Monthly
        time = pd.date_range('1/1/1972', periods=31 * 12, freq='MS')
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

        # Prescribe MB
        pdf = pd.Series(index=mb_mod._state_yr.keys(), data=mb_mod._state_yr.values())
        p_mod = massbalance.RandomMassBalance(gdir, prescribe_years=pdf)

        mb_ts = []
        mb_ts2 = []
        yrs = np.arange(1972, 2003, 1)
        for yr in yrs:
            mb_ts.append(np.average(mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR,
                                    weights=w))
            mb_ts2.append(np.average(p_mod.get_annual_mb(h, yr) * SEC_IN_YEAR,
                                     weights=w))
        np.testing.assert_allclose(mb_ts, mb_ts2)

    def test_random_mb_unique(self, hef_gdir):

        gdir = hef_gdir
        init_present_time_glacier(gdir)

        ref_mod = massbalance.ConstantMassBalance(gdir,
                                                  y0=2002 - 15,
                                                  halfsize=15)
        mb_mod = massbalance.RandomMassBalance(gdir, seed=10,
                                               y0=2002 - 15,
                                               unique_samples=True,
                                               halfsize=15)
        mb_mod2 = massbalance.RandomMassBalance(gdir, seed=20,
                                                y0=2002 - 15,
                                                unique_samples=True,
                                                halfsize=15)
        mb_mod3 = massbalance.RandomMassBalance(gdir, seed=20,
                                                y0=2002 - 15,
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
        assert (len(list(mb_mod._state_yr.values())) ==
                np.unique(list(mb_mod._state_yr.values())).size)
        # size2
        assert (len(list(mb_mod2._state_yr.values())) ==
                np.unique(list(mb_mod2._state_yr.values())).size)
        # state years 1 vs 2
        assert (np.all(np.unique(list(mb_mod._state_yr.values())) ==
                       np.unique(list(mb_mod2._state_yr.values()))))
        # state years 1 vs reference model
        assert (np.all(np.unique(list(mb_mod._state_yr.values())) ==
                       ref_mod.years))

        # test ela vs specific mb
        elats = mb_mod.get_ela(yrs[:200])
        assert np.corrcoef(mbts[:200], elats)[0, 1] < -0.95

        # test mass balance with temperature bias
        assert np.mean(r_mbh) < np.mean(r_mbh3)

    def test_uncertain_mb(self, hef_gdir):

        gdir = hef_gdir

        ref_mod = massbalance.ConstantMassBalance(gdir, y0=2002-15)

        # only change bias: this works as before
        mb_mod = massbalance.UncertainMassBalance(ref_mod,
                                                  rdn_temp_bias_sigma=0,
                                                  rdn_prcp_fac_sigma=0,
                                                  rdn_bias_sigma=100)
        yrs = np.arange(100)
        h, w = gdir.get_inversion_flowline_hw()
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        assert_allclose(ref_mb, check_mb)
        assert np.std(unc_mb) > 50

        mb_mod = massbalance.UncertainMassBalance(ref_mod)
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        temp_1 = ref_mod.mbmod.temp.copy()
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        temp_2 = ref_mod.mbmod.temp.copy()
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        temp_3 = ref_mod.mbmod.temp.copy()
        unc2_mb = mb_mod.get_specific_mb(h, w, year=yrs)

        assert_allclose(temp_1, temp_2)
        assert_allclose(temp_1, temp_3)
        assert_allclose(ref_mb, check_mb)
        assert_allclose(unc_mb, unc2_mb)
        assert np.std(unc_mb) > 50

        mb_mod = massbalance.UncertainMassBalance(ref_mod,
                                                  rdn_temp_bias_sigma=0.1,
                                                  rdn_prcp_fac_sigma=0,
                                                  rdn_bias_sigma=0)
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        assert_allclose(ref_mb, check_mb)
        assert np.std(unc_mb) > 50

        mb_mod = massbalance.UncertainMassBalance(ref_mod,
                                                  rdn_temp_bias_sigma=0,
                                                  rdn_prcp_fac_sigma=0.1,
                                                  rdn_bias_sigma=0)
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        assert_allclose(ref_mb, check_mb)
        assert np.std(unc_mb) > 50

        # Other MBs
        ref_mod = massbalance.MonthlyTIModel(gdir)
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
        ref_mod = massbalance.RandomMassBalance(gdir, y0=2002-15)
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

        # do the same but with larger _sigma:
        mb_mod = massbalance.UncertainMassBalance(ref_mod,
                                                  rdn_temp_bias_sigma=1)
        ref_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        temp_1 = ref_mod.mbmod.temp.copy()
        unc_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        temp_2 = ref_mod.mbmod.temp.copy()
        check_mb = ref_mod.get_specific_mb(h, w, year=yrs)
        temp_3 = ref_mod.mbmod.temp.copy()
        unc2_mb = mb_mod.get_specific_mb(h, w, year=yrs)
        assert_allclose(temp_1, temp_2)
        assert_allclose(temp_1, temp_3)
        assert_allclose(ref_mb, check_mb)
        assert_allclose(unc_mb, unc2_mb)
        assert np.std(unc_mb) > 50

    def test_mb_performance(self, hef_gdir):

        gdir = hef_gdir
        init_present_time_glacier(gdir)

        h, w = gdir.get_inversion_flowline_hw()

        # Climate period, 10 day timestep
        yrs = np.arange(1850, 2002, 10 / 365)

        # models
        start_time = time.time()
        mb1 = massbalance.ConstantMassBalance(gdir, y0=2002-15)
        for yr in yrs:
            mb1.get_monthly_mb(h, yr)
        t1 = time.time() - start_time
        start_time = time.time()
        mb2 = massbalance.MonthlyTIModel(gdir)
        for yr in yrs:
            mb2.get_monthly_mb(h, yr)
        t2 = time.time() - start_time

        # not faster as two times t2
        try:
            assert t1 >= (t2 / 2)
        except AssertionError:
            # no big deal
            pytest.skip('Allowed failure')


class TestModelFlowlines():

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
        ref_l = 18000

        rec = RectangularBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                     surface_h=surface_h, bed_h=bed_h,
                                     widths=widths)

        assert np.all([s == 'rectangular' for s in rec.shape_str])

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
        assert_allclose(rec.length_m, ref_l)

        # We set something and everything stays same
        rec.thick = thick
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.surface_h, surface_h)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, vol_m3.sum())
        assert_allclose(rec.length_m, ref_l)
        rec.section = section
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.surface_h, surface_h)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, vol_m3.sum())
        assert_allclose(rec.length_m, ref_l)
        rec.surface_h = surface_h
        assert_allclose(rec.thick, thick)
        assert_allclose(rec.surface_h, surface_h)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, vol_m3.sum())
        assert_allclose(rec.length_m, ref_l)

        # More adventurous
        rec.section = section / 2
        assert_allclose(rec.thick, thick / 2)
        assert_allclose(rec.widths, widths)
        assert_allclose(rec.widths_m, widths_m)
        assert_allclose(rec.section, section / 2)
        assert_allclose(rec.area_m2, area_m2.sum())
        assert_allclose(rec.volume_m3, (vol_m3 / 2).sum())

        # Water level
        rec = RectangularBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                     surface_h=surface_h, bed_h=bed_h,
                                     widths=widths, water_level=0)
        assert rec.volume_bsl_km3 == 0
        assert rec.volume_bwl_km3 == 0

        rec = RectangularBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                     surface_h=surface_h, bed_h=bed_h,
                                     widths=widths, water_level=5000)
        assert rec.volume_bsl_km3 == 0
        assert rec.volume_bwl_km3 == rec.volume_km3

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

        lambdas = bed_h * 0.
        is_trap = np.ones(len(lambdas), dtype=bool)

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

        assert np.all([s == 'trapezoid' for s in rec1.shape_str])

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
            assert_allclose(rec.thick, thick / 2)
            assert_allclose(rec.widths, widths)
            assert_allclose(rec.widths_m, widths_m)
            assert_allclose(rec.section, section / 2)
            assert_allclose(rec.area_m2, area_m2.sum())
            assert_allclose(rec.volume_m3, (vol_m3 / 2).sum())

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

        lambdas = bed_h * 0. + 1

        # tests
        thick = surface_h - bed_h
        widths_m = widths_0 * map_dx + lambdas * thick
        widths = widths_m / map_dx
        section = thick * (widths_0 * map_dx + widths_m) / 2
        vol_m3 = section * map_dx
        area_m2 = map_dx * widths_m
        area_m2[thick == 0] = 0

        is_trap = np.ones(len(lambdas), dtype=bool)

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

    def test_parab_conserves_shape(self):

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

        shapes = bed_h * 0. + 0.003
        shapes[:30] = 0.002
        shapes[-30:] = 0.004

        # tests
        thick = surface_h - bed_h
        widths_m = np.sqrt(4 * thick / shapes)
        area_m2 = map_dx * widths_m
        area_m2[thick == 0] = 0

        fl = ParabolicBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                  surface_h=surface_h, bed_h=bed_h,
                                  bed_shape=shapes)

        fl2 = MixedBedFlowline(line=line, dx=dx, map_dx=map_dx,
                               surface_h=surface_h, bed_h=bed_h,
                               section=fl.section, bed_shape=shapes,
                               lambdas=np.zeros(len(shapes)),
                               is_trapezoid=np.zeros(len(shapes)).astype(bool))

        assert np.all([s == 'parabolic' for s in fl.shape_str])

        thick_bef = fl.thick
        widths_bef = fl.widths_m
        section_bef = fl.section
        assert_allclose(fl2.thick, thick_bef)
        assert_allclose(fl2.widths_m, widths_bef)

        fl.section = 0
        fl2.section = 0
        assert_allclose(fl.thick, 0)
        assert_allclose(fl.widths_m, 0)
        assert_allclose(fl2.thick, 0)
        assert_allclose(fl2.widths_m, 0)

        fl.section = section_bef
        fl2.section = section_bef
        assert_allclose(fl.thick, thick_bef)
        assert_allclose(fl.widths_m, widths_bef)
        assert_allclose(fl2.thick, thick_bef)
        assert_allclose(fl2.widths_m, widths_bef)

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

        shapes = bed_h * 0. + 0.003
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

        is_trap = np.zeros(len(shapes), dtype=bool)

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

        lambdas = bed_h * 0. + 1
        lambdas[0:50] = 0

        thick = surface_h - bed_h
        widths_m = widths_0 * map_dx + lambdas * thick
        widths = widths_m / map_dx
        section_trap = thick * (widths_0 * map_dx + widths_m) / 2

        rec1 = TrapezoidalBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                      surface_h=surface_h,
                                      bed_h=bed_h, widths=widths,
                                      lambdas=lambdas)

        shapes = bed_h * 0. + 0.003
        shapes[-30:] = 0.004

        # tests
        thick = surface_h - bed_h
        widths_m = np.sqrt(4 * thick / shapes)
        section_para = 2 / 3 * widths_m * thick

        rec2 = ParabolicBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                    surface_h=surface_h, bed_h=bed_h,
                                    bed_shape=shapes)

        is_trap = np.ones(len(shapes), dtype=bool)
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

        # Water level
        rec.water_level = 0
        assert rec.volume_bsl_km3 == 0

        rec.water_level = 5000
        assert rec.volume_bwl_km3 == rec.volume_km3
        assert rec.volume_bsl_km3 == 0

        rec.water_level = 2500
        assert 0 < rec.volume_bwl_km3 < rec.volume_km3
        assert rec.volume_bsl_km3 == 0

    def test_length_methods(self):

        cfg.initialize()

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
        ref_l = 18000
        full_l = 20000

        rec = RectangularBedFlowline(line=line, dx=dx, map_dx=map_dx,
                                     surface_h=surface_h, bed_h=bed_h,
                                     widths=widths)

        assert rec.length_m == ref_l
        rec.thick = rec.thick * 0 + 100
        assert rec.length_m == full_l
        assert rec.terminus_index == nx - 1

        cfg.PARAMS['glacier_length_method'] = 'consecutive'
        assert rec.length_m == full_l
        assert rec.terminus_index == nx - 1

        cfg.PARAMS['min_ice_thick_for_length'] = 1
        rec.thick = rec.thick * 0 + 0.5
        assert rec.length_m == 0
        assert rec.terminus_index == -1

        cfg.PARAMS['glacier_length_method'] = 'naive'
        assert rec.length_m == 0
        assert rec.terminus_index == -1

        t = rec.thick * 0 + 20
        t[10] = 0.5
        rec.thick = t
        assert rec.length_m == full_l - map_dx
        assert rec.terminus_index == nx - 1

        cfg.PARAMS['glacier_length_method'] = 'consecutive'
        assert rec.length_m == 1000
        assert rec.terminus_index == 9


@pytest.fixture(scope='class')
def io_init_gdir(hef_gdir):
    init_present_time_glacier(hef_gdir)


@pytest.mark.usefixtures('io_init_gdir')
class TestIO():
    glen_a = 2.4e-24

    @pytest.mark.parametrize("bed", [dummy_constant_bed, dummy_width_bed,
                                     dummy_noisy_bed, dummy_bumpy_bed,
                                     dummy_parabolic_bed,
                                     dummy_trapezoidal_bed, dummy_mixed_bed])
    def test_flowline_to_geometry_dataset(self, bed):
        fl = bed()[0]
        ds = fl.to_geometry_dataset()
        fl_ = flowline_from_dataset(ds)
        ds_ = fl_.to_geometry_dataset()
        assert ds_.equals(ds)

    def test_model_to_file(self, class_case_dir):

        p = os.path.join(class_case_dir, 'grp.nc')
        if os.path.isfile(p):
            os.remove(p)

        fls = dummy_width_bed_tributary()
        model = FluxBasedModel(fls)
        model.to_geometry_netcdf(p)
        fls_ = glacier_from_netcdf(p)

        for fl, fl_ in zip(fls, fls_):
            ds = fl.to_geometry_dataset()
            ds_ = fl_.to_geometry_dataset()
            assert ds_.equals(ds)

        assert fls_[0].flows_to is fls_[1]
        assert fls[0].flows_to_indice == fls_[0].flows_to_indice

        # They should be sorted
        to_test = [fl.order for fl in fls_]
        assert np.array_equal(np.sort(to_test), to_test)

        # They should be able to start a run
        mb = LinearMassBalance(2600.)
        model = FluxBasedModel(fls_, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        model.run_until(100)

    @pytest.mark.slow
    def test_run(self, class_case_dir):
        mb = LinearMassBalance(2600.)

        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        ds_diag, ds_fl, ds = model.run_until_and_store(500,
                                                       store_monthly_step=True,
                                                       fl_diag_path=None,
                                                       geom_path=None)
        ds = ds[0]
        ds_fl = ds_fl[0]

        # Check attrs
        assert ds.attrs['mb_model_class'] == 'LinearMassBalance'
        assert ds.attrs['mb_model_rho'] == cfg.PARAMS['ice_density']
        assert ds_diag.attrs['mb_model_class'] == 'LinearMassBalance'
        assert ds_diag.attrs['mb_model_ela_h'] == 2600

        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)

        years = utils.monthly_timeseries(0, 500)
        vol_ref = []
        a_ref = []
        l_ref = []
        h_previous_timestep = model.fls[0].thick
        dhdt_ref = []
        surface_h_previous_timestep = model.fls[0].surface_h
        climatic_mb_ref = []
        flux_divergence_ref = []
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
                if yr > 0:
                    dhdt_ref.append(model.fls[0].thick -
                                    h_previous_timestep)
                    h_previous_timestep = model.fls[0].thick
                    # for mb use previous surface height and previous year, only
                    # save climatic mb where dhdt is non zero
                    climatic_mb_ref.append(
                        np.where(np.isclose(dhdt_ref[-1], 0.),
                                 0.,
                                 model.get_mb(surface_h_previous_timestep,
                                              model.yr - 1,
                                              fl_id=0) *
                                 SEC_IN_YEAR)  # converted to m yr-1
                    )
                    surface_h_previous_timestep = model.fls[0].surface_h
                    # smooth flux divergence where glacier is getting ice free
                    has_become_ice_free = np.logical_and(
                                np.isclose(model.fls[0].thick, 0.),
                                dhdt_ref[-1] < 0.
                            )
                    flux_divergence_ref.append(
                        (dhdt_ref[-1] - climatic_mb_ref[-1]) *
                        np.where(has_become_ice_free, 0.1, 1.))
                if int(yr) == 500:
                    secfortest = model.fls[0].section
                    hfortest = model.fls[0].thick

        np.testing.assert_allclose(ds.ts_section.isel(time=-1),
                                   secfortest)
        np.testing.assert_allclose(ds_fl.thickness_m.isel(time=-1),
                                   hfortest)

        np.testing.assert_allclose(ds_diag.volume_m3, vol_diag)
        np.testing.assert_allclose(ds_diag.area_m2, a_diag)
        np.testing.assert_allclose(ds_diag.length_m, l_diag)

        np.testing.assert_allclose(ds_fl.volume_m3.sum(dim='dis_along_flowline'),
                                   vol_ref)
        np.testing.assert_allclose(ds_fl.volume_bwl_m3.sum(dim='dis_along_flowline'),
                                   0)
        np.testing.assert_allclose(ds_fl.volume_bsl_m3.sum(dim='dis_along_flowline'),
                                   0)
        np.testing.assert_allclose(ds_fl.area_m2.sum(dim='dis_along_flowline'),
                                   a_ref)

        np.testing.assert_allclose(dhdt_ref,
                                   ds_fl.dhdt_myr[1:])  # first step is nan
        np.testing.assert_allclose(climatic_mb_ref,
                                   ds_fl.climatic_mb_myr[1:])
        np.testing.assert_allclose(flux_divergence_ref,
                                   ds_fl.flux_divergence_myr[1:])

        vel = ds_fl.ice_velocity_myr.isel(time=-1)
        assert 20 < vel.max() < 40

        fls = dummy_constant_bed()
        geom_path = os.path.join(class_case_dir, 'ts_ideal.nc')
        diag_path = os.path.join(class_case_dir, 'ts_diag.nc')
        fl_diag_path = os.path.join(class_case_dir, 'ts_fl_diag.nc')
        if os.path.exists(geom_path):
            os.remove(geom_path)
        if os.path.exists(diag_path):
            os.remove(diag_path)
        if os.path.exists(fl_diag_path):
            os.remove(fl_diag_path)
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)

        # We add this because this discovered a bug
        from oggm.core.flowline import zero_glacier_stop_criterion
        model.run_until_and_store(500, geom_path=geom_path,
                                  diag_path=diag_path,
                                  fl_diag_path=fl_diag_path,
                                  stop_criterion=zero_glacier_stop_criterion,
                                  store_monthly_step=True)

        with xr.open_dataset(diag_path) as ds_:
            # the identical (i.e. attrs + names) doesn't work because of date
            del ds_diag.attrs['creation_date']
            del ds_.attrs['creation_date']
            xr.testing.assert_identical(ds_diag, ds_)

        # Test new fl diags
        with xr.open_dataset(fl_diag_path, group='fl_0') as ds_fl:
            assert_allclose(ds_fl.volume_m3.sum(dim='dis_along_flowline'),
                            vol_ref)
            assert_allclose(ds_fl.volume_bwl_m3.sum(dim='dis_along_flowline'),
                            0)
            assert_allclose(ds_fl.volume_bsl_m3.sum(dim='dis_along_flowline'),
                            0)
            assert_allclose(ds_fl.area_m2.sum(dim='dis_along_flowline'),
                            a_ref)

        # Test restart files
        with pytest.warns(FutureWarning):
            with FileModel(geom_path):
                pass

        fmodel = FileModel(geom_path)
        assert fmodel.last_yr == 500

        with pytest.raises(IndexError):
            fmodel.run_until(500.1)

        with pytest.raises(IndexError):
            fmodel.run_until(500, 12)

        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        for yr in years:
            model.run_until(yr)
            if yr in [100, 300, 500]:
                # this is sloooooow so we test a little bit only
                fmodel.run_until(yr, 1)
                np.testing.assert_allclose(model.fls[0].section,
                                           fmodel.fls[0].section)
                np.testing.assert_allclose(model.fls[0].widths_m,
                                           fmodel.fls[0].widths_m)

        np.testing.assert_allclose(fmodel.volume_m3_ts(), vol_ref)
        np.testing.assert_allclose(fmodel.area_m2_ts(), a_ref)
        with pytest.raises(NotImplementedError):
            fmodel.length_m_ts()

        # Can we start a run from the middle?
        fmodel.run_until(300)
        model = FluxBasedModel(fmodel.fls, mb_model=mb, y0=300,
                               glen_a=self.glen_a)
        model.run_until(500)
        fmodel.run_until(500)
        np.testing.assert_allclose(model.fls[0].section,
                                   fmodel.fls[0].section)
        np.testing.assert_allclose(model.fls[0].thick,
                                   fmodel.fls[0].thick)

    @pytest.mark.slow
    def test_fixed_geom_spinup(self, class_case_dir):
        mb = LinearMassBalance(2600.)
        model = FluxBasedModel(dummy_constant_bed(), mb_model=mb, y0=0., glen_a=self.glen_a)
        model.run_until(200)

        mb = LinearMassBalance(2800.)
        model = FluxBasedModel(model.fls, mb_model=mb, y0=20, glen_a=self.glen_a)

        ds_diag, ds_fl, ds = model.run_until_and_store(40,
                                                       fl_diag_path=None,
                                                       geom_path=None,
                                                       fixed_geometry_spinup_yr=0)
        assert ds_diag.time[0] == 0
        assert ds_diag.time[-1] == 40

        area = ds_diag.area_m2.to_series()
        vol = ds_diag.volume_m3.to_series()
        is_spin = ds_diag.is_fixed_geometry_spinup.to_series()
        assert_allclose(area.loc[:19], area.loc[0])
        assert_allclose(is_spin.loc[:19], True)
        assert_allclose(is_spin.loc[20:], False)
        dv = vol.iloc[1:].values - vol.iloc[:-1]
        assert_allclose(dv.loc[:19], dv.loc[0])
        # This doesn't work though
        assert np.all(dv.loc[:19] != dv.loc[20])

    @pytest.mark.slow
    def test_calving_filemodel(self, class_case_dir):
        y1 = 1200
        geom_path = os.path.join(class_case_dir, 'ts_ideal.nc')
        diag_path = os.path.join(class_case_dir, 'ts_diag.nc')
        if os.path.exists(geom_path):
            os.remove(geom_path)
        if os.path.exists(diag_path):
            os.remove(diag_path)
        model = FluxBasedModel(bu_tidewater_bed(),
                               mb_model=massbalance.ScalarMassBalance(),
                               is_tidewater=True,
                               flux_gate=0.12, do_kcalving=True,
                               calving_k=0.2)
        diag, fl_diag, _ = model.run_until_and_store(y1,
                                                     fl_diag_path=None,
                                                     diag_path=diag_path,
                                                     geom_path=geom_path)
        assert model.calving_m3_since_y0 > 0

        assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)
        assert_allclose(diag.volume_m3.max() + diag.calving_m3.max(),
                        model.flux_gate_m3_since_y0)

        fmodel = FileModel(geom_path)
        assert fmodel.last_yr == y1
        assert fmodel.do_calving

        assert_allclose(fmodel.volume_m3_ts(), diag.volume_m3)
        assert_allclose(fmodel.area_m2_ts(), diag.area_m2)

        fl_diag = fl_diag[0]
        assert_allclose(fl_diag.volume_m3.sum(dim='dis_along_flowline') -
                        fl_diag.calving_bucket_m3,
                        diag.volume_m3)
        assert_allclose(fl_diag.area_m2.sum(dim='dis_along_flowline'),
                        diag.area_m2)

        fmodel.run_until(y1)
        assert_allclose(fmodel.volume_m3 + fmodel.calving_m3_since_y0,
                        model.flux_gate_m3_since_y0)

    @pytest.mark.slow
    def test_run_annual_step(self, class_case_dir):
        mb = LinearMassBalance(2600.)

        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        ds_diag, ds = model.run_until_and_store(500, geom_path=None)
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

        fls = dummy_constant_bed()
        geom_path = os.path.join(class_case_dir, 'ts_ideal.nc')
        diag_path = os.path.join(class_case_dir, 'ts_diag.nc')
        if os.path.exists(geom_path):
            os.remove(geom_path)
        if os.path.exists(diag_path):
            os.remove(diag_path)
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)
        model.run_until_and_store(500, geom_path=geom_path,
                                  diag_path=diag_path)

        with xr.open_dataset(diag_path) as ds_:
            # the identical (i.e. attrs + names) doesn't work because of date
            del ds_diag.attrs['creation_date']
            del ds_.attrs['creation_date']
            xr.testing.assert_identical(ds_diag, ds_)

        fmodel = FileModel(geom_path)
        assert fmodel.last_yr == 500
        fls = dummy_constant_bed()
        model = FluxBasedModel(fls, mb_model=mb, y0=0.,
                               glen_a=self.glen_a)

        for yr in years:
            model.run_until(yr)
            if yr in [100, 300, 500]:
                # this used to be sloooooow so we test a little bit only
                fmodel.run_until(yr)
                np.testing.assert_allclose(model.fls[0].section,
                                           fmodel.fls[0].section)
                np.testing.assert_allclose(model.fls[0].widths_m,
                                           fmodel.fls[0].widths_m)

        np.testing.assert_allclose(fmodel.volume_m3_ts(), vol_ref)
        np.testing.assert_allclose(fmodel.area_m2_ts(), a_ref)

        # Can we start a run from the middle?
        fmodel.run_until(300)
        model = FluxBasedModel(fmodel.fls, mb_model=mb, y0=300,
                               glen_a=self.glen_a)
        model.run_until(500)
        fmodel.run_until(500)
        np.testing.assert_allclose(model.fls[0].section,
                                   fmodel.fls[0].section)

    def test_gdir_copy(self, hef_gdir):

        new_dir = os.path.join(get_test_dir(), 'tmp_testcopy')
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        new_gdir = tasks.copy_to_basedir(hef_gdir, base_dir=new_dir,
                                         setup='all')
        init_present_time_glacier(new_gdir)
        shutil.rmtree(new_dir)

        new_gdir = tasks.copy_to_basedir(hef_gdir, base_dir=new_dir,
                                         setup='run')
        task_status = new_gdir.get_task_status('init_present_time_glacier')
        assert task_status == 'SUCCESS'
        assert new_gdir.grid
        run_random_climate(new_gdir, nyears=10, y0=1985)
        shutil.rmtree(new_dir)

        new_gdir = tasks.copy_to_basedir(hef_gdir, base_dir=new_dir,
                                         setup='inversion')
        inversion.prepare_for_inversion(new_gdir, invert_all_rectangular=True)
        inversion.mass_conservation_inversion(new_gdir)
        init_present_time_glacier(new_gdir)
        run_constant_climate(new_gdir, nyears=10, bias=0, y0=1985)
        shutil.rmtree(new_dir)

    def test_hef(self, class_case_dir, hef_gdir):

        p = os.path.join(class_case_dir, 'grp_hef.nc')
        if os.path.isfile(p):
            os.remove(p)

        init_present_time_glacier(hef_gdir)

        fls = hef_gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls)

        model.to_geometry_netcdf(p)
        fls_ = glacier_from_netcdf(p)

        for fl, fl_ in zip(fls, fls_):
            ds = fl.to_geometry_dataset()
            ds_ = fl_.to_geometry_dataset()
            for v in ds.variables.keys():
                np.testing.assert_allclose(ds_[v], ds[v], equal_nan=True)

        for fl, fl_ in zip(fls[:-1], fls_[:-1]):
            assert fl.flows_to_indice == fl_.flows_to_indice

        # mixed flowline
        fls = hef_gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls)

        p = os.path.join(class_case_dir, 'grp_hef_mix.nc')
        if os.path.isfile(p):
            os.remove(p)
        model.to_geometry_netcdf(p)
        fls_ = glacier_from_netcdf(p)

        np.testing.assert_allclose(fls[0].section, fls_[0].section)
        np.testing.assert_allclose(fls[0]._ptrap, fls_[0]._ptrap)
        np.testing.assert_allclose(fls[0].bed_h, fls_[0].bed_h)

        for fl, fl_ in zip(fls, fls_):
            ds = fl.to_geometry_dataset()
            ds_ = fl_.to_geometry_dataset()
            np.testing.assert_allclose(fl.section, fl_.section)
            np.testing.assert_allclose(fl._ptrap, fl_._ptrap)
            np.testing.assert_allclose(fl.bed_h, fl_.bed_h)
            xr.testing.assert_allclose(ds, ds_)

        for fl, fl_ in zip(fls[:-1], fls_[:-1]):
            assert fl.flows_to_indice == fl_.flows_to_indice


@pytest.fixture(scope='class')
def inversion_gdir(class_case_dir):
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

    gdir = GlacierDirectory(entity, base_dir=class_case_dir, reset=True)
    define_glacier_region(gdir)
    return gdir


class TestIdealisedInversion():
    def simple_plot(self, model, gdir):  # pragma: no cover
        ocls = gdir.read_pickle('inversion_output')
        ithick = ocls[-1]['thick']
        pg = np.where((model.fls[-1].thick > 0) & (model.fls[-1].widths_m > 1))
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

    def double_plot(self, model, gdir):  # pragma: no cover
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

    def test_inversion_rectangular(self, inversion_gdir):

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
            flo.is_rectangular = np.ones(flo.nx).astype(bool)
            fls.append(flo)
        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)
        inversion.prepare_for_inversion(inversion_gdir)
        v = inversion.mass_conservation_inversion(inversion_gdir)

        assert_allclose(v, model.volume_m3, rtol=0.01)

        # Equations
        mb_on_z = mb.get_annual_mb(fl.surface_h[pg])
        flux = np.cumsum(fl.widths_m[pg] * fl.dx_meter * mb_on_z)

        inv_out = inversion_gdir.read_pickle('inversion_input')
        inv_flux = inv_out[0]['flux']

        slope = - np.gradient(fl.surface_h[pg], fl.dx_meter)

        est_h = inversion.sia_thickness(slope, fl.widths_m[pg], flux)
        est_ho = inversion.sia_thickness_via_optim(slope, fl.widths_m[pg],
                                                   flux)
        mod_h = fl.thick[pg]

        # Test in the middle where slope is not too important
        assert_allclose(est_h[25:75], mod_h[25:75], rtol=0.01)
        assert_allclose(est_ho[25:75], mod_h[25:75], rtol=0.01)
        assert_allclose(est_h, est_ho, rtol=0.01)

        # OGGM internal flux
        est_h_ofl = inversion.sia_thickness(slope, fl.widths_m[pg], inv_flux)
        est_ho = inversion.sia_thickness_via_optim(slope, fl.widths_m[pg],
                                                   inv_flux)

        # Test in the middle where slope is not too important
        assert_allclose(est_h_ofl[25:75], mod_h[25:75], rtol=0.02)
        assert_allclose(est_ho[25:75], mod_h[25:75], rtol=0.02)
        assert_allclose(est_h_ofl, est_ho, rtol=0.01)

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
            self.simple_plot(model, inversion_gdir)

    def test_inversion_trapeze(self, inversion_gdir):

        fls = dummy_trapezoidal_bed(map_dx=inversion_gdir.grid.dx,
                                    def_lambdas=cfg.PARAMS['trapezoid_lambdas'])
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
            flo.is_rectangular = np.ones(flo.nx).astype(bool)
            fls.append(flo)
        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)

        # Equations
        mb_on_z = mb.get_annual_mb(fl.surface_h[pg])
        flux = np.cumsum(fl.widths_m[pg] * fl.dx_meter * mb_on_z)

        slope = - np.gradient(fl.surface_h[pg], fl.dx_meter)
        est_ho = inversion.sia_thickness_via_optim(slope, fl.widths_m[pg],
                                                   flux, shape='trapezoid')
        mod_h = fl.thick[pg]

        # Test in the middle where slope is not too important
        assert_allclose(est_ho[25:75], mod_h[25:75], rtol=0.01)

        if do_plot:  # pragma: no cover
            plt.plot(mod_h)
            plt.plot(est_ho)
            plt.show()

    def test_inversion_parabolic(self, inversion_gdir):

        fls = dummy_parabolic_bed(map_dx=inversion_gdir.grid.dx)
        mb = LinearMassBalance(2500.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where((fl.thick > 0) & (fl.widths_m > 1))
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=fl.surface_h[pg])
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.zeros(flo.nx).astype(bool)
            fls.append(flo)
        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)
        inversion.prepare_for_inversion(inversion_gdir)
        v = inversion.mass_conservation_inversion(inversion_gdir)
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
        est_ho = inversion.sia_thickness_via_optim(slope, fl.widths_m[pg],
                                                   flux, shape='parabolic')
        mod_h = fl.thick[pg]

        # Test in the middle where slope is not too important
        assert_allclose(est_h[25:75], mod_h[25:75], rtol=0.01)
        assert_allclose(est_ho[25:75], mod_h[25:75], rtol=0.01)
        assert_allclose(est_h, est_ho, rtol=0.01)

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

    @pytest.mark.slow
    def test_inversion_mixed(self, inversion_gdir):

        fls = dummy_mixed_bed(deflambdas=0, map_dx=inversion_gdir.grid.dx,
                              mixslice=slice(10, 30))
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        # This reduces the test's accuracy but makes it much faster.
        model.run_until_equilibrium(rate=0.01)

        fls = []
        for fl in model.fls:
            pg = np.where((fl.thick > 0) & (fl.widths_m > 1))
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = fl.is_trapezoid[pg]
            fls.append(flo)
        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)
        inversion.prepare_for_inversion(inversion_gdir)
        v = inversion.mass_conservation_inversion(inversion_gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model, inversion_gdir)

    @pytest.mark.slow
    def test_inversion_cliff(self, inversion_gdir):

        fls = dummy_constant_bed_cliff(map_dx=inversion_gdir.grid.dx,
                                       cliff_height=100)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where((fl.thick > 0) & (fl.widths_m > 1))
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(bool)
            fls.append(flo)
        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)
        inversion.prepare_for_inversion(inversion_gdir)
        v = inversion.mass_conservation_inversion(inversion_gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model, inversion_gdir)

    def test_inversion_noisy(self, inversion_gdir):

        fls = dummy_noisy_bed(map_dx=inversion_gdir.grid.dx)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where((fl.thick > 0) & (fl.widths_m > 1))
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(bool)
            fls.append(flo)
        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)
        inversion.prepare_for_inversion(inversion_gdir)
        v = inversion.mass_conservation_inversion(inversion_gdir)

        assert_allclose(v, model.volume_m3, rtol=0.05)
        if do_plot:  # pragma: no cover
            self.simple_plot(model, inversion_gdir)

    def test_inversion_tributary(self, inversion_gdir):

        fls = dummy_width_bed_tributary(map_dx=inversion_gdir.grid.dx)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()

        fls = []
        for fl in model.fls:
            pg = np.where((fl.thick > 0) & (fl.widths_m > 1))
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(bool)
            fls.append(flo)

        fls[0].set_flows_to(fls[1])

        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)
        inversion.prepare_for_inversion(inversion_gdir)
        v = inversion.mass_conservation_inversion(inversion_gdir)

        assert_allclose(v, model.volume_m3, rtol=0.02)
        if do_plot:  # pragma: no cover
            self.double_plot(model, inversion_gdir)

    def test_inversion_non_equilibrium(self, inversion_gdir):

        fls = dummy_constant_bed(map_dx=inversion_gdir.grid.dx)
        mb = LinearMassBalance(2600.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()

        mb = LinearMassBalance(2800.)
        model = FluxBasedModel(fls, mb_model=mb, y0=0)
        model.run_until(50)

        fls = []
        for fl in model.fls:
            pg = np.where((fl.thick > 0) & (fl.widths_m > 1))
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.ones(flo.nx).astype(bool)
            fls.append(flo)
        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)
        inversion.prepare_for_inversion(inversion_gdir)
        v = inversion.mass_conservation_inversion(inversion_gdir)

        # expected errors
        assert v > model.volume_m3
        ocls = inversion_gdir.read_pickle('inversion_output')
        ithick = ocls[0]['thick']
        assert np.mean(ithick) > np.mean(model.fls[0].thick) * 1.1
        if do_plot:  # pragma: no cover
            self.simple_plot(model, inversion_gdir)

    def test_inversion_and_run(self, inversion_gdir):

        fls = dummy_parabolic_bed(map_dx=inversion_gdir.grid.dx)
        mb = LinearMassBalance(2500.)

        model = FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until_equilibrium()
        fls = []
        for fl in model.fls:
            pg = np.where((fl.thick > 0) & (fl.widths_m > 1))
            line = shpg.LineString([fl.line.coords[int(p)] for p in pg[0]])
            sh = fl.surface_h[pg]
            flo = centerlines.Centerline(line, dx=fl.dx,
                                         surface_h=sh)
            flo.widths = fl.widths[pg]
            flo.is_rectangular = np.zeros(flo.nx).astype(bool)
            fls.append(flo)
        inversion_gdir.write_pickle(copy.deepcopy(fls), 'inversion_flowlines')

        massbalance.apparent_mb_from_linear_mb(inversion_gdir)
        inversion.prepare_for_inversion(inversion_gdir)
        v = inversion.mass_conservation_inversion(inversion_gdir)

        assert_allclose(v, model.volume_m3, rtol=0.01)

        inv = inversion_gdir.read_pickle('inversion_output')[-1]
        bed_shape_gl = 4 * inv['thick'] / (flo.widths * inversion_gdir.grid.dx) ** 2

        ithick = inv['thick']
        fls = dummy_parabolic_bed(map_dx=inversion_gdir.grid.dx,
                                  from_other_shape=bed_shape_gl[:-2],
                                  from_other_bed=(sh - ithick)[:-2])
        model2 = FluxBasedModel(fls, mb_model=mb, y0=0.)
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


@pytest.fixture(scope='class')
def gdir_sh(request, test_dir, hef_gdir):
    dir_sh = os.path.join(test_dir, request.cls.__name__ + '_sh')
    utils.mkdir(dir_sh, reset=True)
    gdir_sh = tasks.copy_to_basedir(hef_gdir, base_dir=dir_sh,
                                    setup='all')
    gdir_sh.hemisphere = 'sh'
    yield gdir_sh
    # teardown
    if os.path.exists(dir_sh):
        shutil.rmtree(dir_sh)


@pytest.fixture(scope='class')
def gdir_calving(request, test_dir, hef_gdir):
    dir_sh = os.path.join(test_dir, request.cls.__name__ + '_calving')
    utils.mkdir(dir_sh, reset=True)
    gdir_sh = tasks.copy_to_basedir(hef_gdir, base_dir=dir_sh,
                                    setup='all')
    gdir_sh.is_tidewater = True
    yield gdir_sh
    # teardown
    if os.path.exists(dir_sh):
        shutil.rmtree(dir_sh)


@pytest.fixture(scope='class')
def with_class_wd(request, test_dir, hef_gdir):
    # dependency on hef_gdir to ensure proper initialization order
    prev_wd = cfg.PATHS['working_dir']
    cfg.PATHS['working_dir'] = os.path.join(
        test_dir, request.cls.__name__ + '_wd')
    utils.mkdir(cfg.PATHS['working_dir'], reset=True)
    yield
    # teardown
    cfg.PATHS['working_dir'] = prev_wd


@pytest.fixture(scope='class')
def inversion_params(hef_gdir):
    diag = hef_gdir.get_diagnostics()
    return {k: diag[k] for k in ('inversion_glen_a', 'inversion_fs')}


@pytest.mark.usefixtures('with_class_wd')
class TestHEFNonPolluted:
    """The tests are so convoluted that this does not work when all class
    tests are run"""

    @pytest.mark.slow
    def test_flux_gate_on_hef(self, hef_gdir, inversion_params):

        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        init_present_time_glacier(hef_gdir)

        mb_mod = massbalance.ScalarMassBalance()
        fls = hef_gdir.read_pickle('model_flowlines')
        for fl in fls:
            fl.thick = fl.thick * 0
        model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                               flux_gate=0.03, flux_gate_build_up=50)
        model.run_until(500)
        assert_allclose(model.volume_m3, model.flux_gate_m3_since_y0)
        beds = np.unique(model.fls[-1].shape_str[model.fls[-1].thick > 0])
        assert len(beds) == 3
        if do_plot:
            from oggm import graphics
            graphics.plot_modeloutput_section_withtrib(model)
            plt.show()

    @pytest.mark.slow
    def test_output_management(self, hef_gdir, inversion_params):

        gdir = hef_gdir
        gdir.rgi_date = 1990
        cfg.PARAMS['store_model_geometry'] = True
        cfg.PARAMS['store_fl_diagnostics'] = True

        # Try minimal output and see if it works
        cfg.PARAMS['store_diagnostic_variables'] = ['volume', 'area', 'length',
                                                    'terminus_thick_0',
                                                    'terminus_thick_1',
                                                    'terminus_thick_2',
                                                    ]
        cfg.PARAMS['store_fl_diagnostic_variables'] = ['area', 'volume']
        # using relative large min ice thick due to overdeepening of inversion
        # -> sometimes small thicknesses after overdeepening (important for
        # terminus thickness check)
        cfg.PARAMS['min_ice_thick_for_length'] = 0.1

        init_present_time_glacier(gdir)
        tasks.run_from_climate_data(gdir, min_ys=1980,
                                    output_filesuffix='_hist')
        tasks.run_from_climate_data(gdir, fixed_geometry_spinup_yr=1980,
                                    output_filesuffix='_spin')

        # Check fl diagnostics
        fl_diag_path = gdir.get_filepath('fl_diagnostics', filesuffix='_hist')
        with xr.open_dataset(fl_diag_path, group='fl_0') as ds_fl:
            assert 'area_m2' in ds_fl
            assert 'volume_m3' in ds_fl
            assert 'volume_bsl_m3' not in ds_fl

        past_run_file = os.path.join(cfg.PATHS['working_dir'], 'compiled.nc')
        mb_file = os.path.join(cfg.PATHS['working_dir'], 'fixed_mb.csv')
        stats_file = os.path.join(cfg.PATHS['working_dir'], 'stats.csv')
        out_path = os.path.join(cfg.PATHS['working_dir'], 'extended.nc')

        # Check stats
        df = utils.compile_glacier_statistics([gdir], path=stats_file)
        assert df.loc[gdir.rgi_id, 'error_task'] is None
        assert not df.loc[gdir.rgi_id, 'is_tidewater']

        # Compile stuff
        utils.compile_fixed_geometry_mass_balance([gdir], path=mb_file)
        utils.compile_run_output([gdir], path=past_run_file,
                                 input_filesuffix='_hist')

        # Extend
        utils.extend_past_climate_run(past_run_file=past_run_file,
                                      fixed_geometry_mb_file=mb_file,
                                      glacier_statistics_file=stats_file,
                                      path=out_path)

        with xr.open_dataset(out_path) as ods, \
                xr.open_dataset(past_run_file) as ds:

            ref = ds.volume
            new = ods.volume
            for y in [1992, 2000, 2003]:
                assert new.sel(time=y).data == ref.sel(time=y).data

            new = ods.volume_fixed_geom
            np.testing.assert_allclose(new.sel(time=2000), ref.sel(time=2000),
                                       rtol=0.01)

            del ods['volume_fixed_geom']
            all_vars = list(ds.data_vars)
            no_term = [vn for vn in all_vars if 'terminus_thick_' not in vn]
            assert sorted(no_term) == sorted(list(ods.data_vars))

            assert np.all(ds.terminus_thick_0 > 0.1)
            assert np.all(ds.terminus_thick_1 >= ds.terminus_thick_0)
            # exclude first two time steps because of bed geometry
            assert np.all(ds.terminus_thick_2[2:] > ds.terminus_thick_1[2:])

            for vn in ['area']:
                ref = ds[vn]
                new = ods[vn]
                for y in [1992, 2000, 2003]:
                    assert new.sel(time=y).data == ref.sel(time=y).data
                assert new.sel(time=1950).data == new.sel(time=1980).data

            # We pick symmetry around rgi date so show that somehow it works
            for vn in ['volume']:
                rtol = 0.5
                np.testing.assert_allclose(ods[vn].sel(time=2000) -
                                           ods[vn].sel(time=1990),
                                           ods[vn].sel(time=1990) -
                                           ods[vn].sel(time=1980),
                                           rtol=rtol)


@pytest.mark.usefixtures('with_class_wd')
class TestHEF:

    @pytest.mark.slow
    def test_stop_criterion(self, hef_gdir, inversion_params):
        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        init_present_time_glacier(hef_gdir)
        cfg.PARAMS['min_ice_thick_for_length'] = 1

        # Check more output
        cfg.PARAMS['store_fl_diagnostics'] = True
        cfg.PARAMS['store_model_geometry'] = True

        run_random_climate(hef_gdir, y0=1985, nyears=200,
                           stop_criterion=equilibrium_stop_criterion,
                           output_filesuffix='_stop', seed=1)
        run_random_climate(hef_gdir, y0=1985, nyears=200,
                           output_filesuffix='_nostop', seed=1)

        ft = 'fl_diagnostics'
        fp = hef_gdir.get_filepath(ft, filesuffix='_stop')
        with xr.open_dataset(fp, group='fl_2') as ds:
            ds_stop = ds.load()
        fp = hef_gdir.get_filepath(ft, filesuffix='_nostop')
        with xr.open_dataset(fp, group='fl_2') as ds:
            ds_nostop = ds.load()

        ds_stop = ds_stop.volume_m3.sum(dim='dis_along_flowline')
        ds_nostop = ds_nostop.volume_m3.sum(dim='dis_along_flowline')

        assert ds_stop.isnull().sum() == 0
        assert ds_nostop.isnull().sum() == 0

        assert_allclose(ds_stop.isel(time=-1), ds_nostop.isel(time=-1), rtol=0.2)
        assert ds_stop.time[-1] < 150

        ft = 'model_diagnostics'
        fp = hef_gdir.get_filepath(ft, filesuffix='_stop')
        with xr.open_dataset(fp) as ds:
            ds_stop = ds.load()
        fp = hef_gdir.get_filepath(ft, filesuffix='_nostop')
        with xr.open_dataset(fp) as ds:
            ds_nostop = ds.load()

        assert ds_stop.volume_m3.isnull().sum() == 0
        assert ds_nostop.volume_m3.isnull().sum() == 0

        ds_stop = ds_stop.volume_m3
        ds_nostop = ds_nostop.volume_m3
        assert_allclose(ds_stop.isel(time=-1), ds_nostop.isel(time=-1), rtol=0.2)
        assert ds_stop.time[-1] < 150

        if do_plot:
            ds_nostop.plot(label='NoStop')
            ds_stop.plot(label='Stop')
            plt.legend()
            plt.show()

    @pytest.mark.slow
    def test_compile_time_workflow(self, hef_gdir, hef_copy_gdir, inversion_params):
        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        init_present_time_glacier(hef_gdir)
        init_present_time_glacier(hef_copy_gdir)
        cfg.PARAMS['min_ice_thick_for_length'] = 1
        cfg.PARAMS['store_model_geometry'] = True

        run_with_hydro(hef_gdir, run_task=run_from_climate_data,
                       ys=1985, ye=1995, store_monthly_hydro=True)
        run_with_hydro(hef_copy_gdir, run_task=run_from_climate_data,
                       ys=1985, ye=1995, store_monthly_hydro=True)

        # Roundtrip
        ds1 = utils.compile_run_output([hef_gdir, hef_copy_gdir])
        ds2 = utils.compile_run_output([hef_copy_gdir, hef_gdir])
        xr.testing.assert_allclose(ds1.sum(dim='rgi_id'), ds2.sum(dim='rgi_id'))

        # If we were using xarray (which we should), we get:
        fps = [gd.get_filepath('model_diagnostics') for gd in [hef_gdir, hef_copy_gdir]]
        ds = xr.open_mfdataset(fps, combine='nested', concat_dim='rgi_id')
        assert_allclose(ds.volume_m3.T, ds1.volume)
        assert_allclose(ds.area_m2.T, ds1.area)
        assert_allclose(ds.calving_m3.T, ds1.calving)

        fps = [gd.get_filepath('model_diagnostics') for gd in [hef_copy_gdir, hef_gdir]]
        ds = xr.open_mfdataset(fps, combine='nested', concat_dim='rgi_id')
        assert_allclose(ds.volume_m3.T, ds2.volume)
        assert_allclose(ds.area_m2.T, ds2.area)
        assert_allclose(ds.calving_m3.T, ds2.calving)

    @pytest.mark.slow
    def test_equilibrium_glacier_wide(self, hef_gdir, inversion_params):

        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        cfg.PARAMS['min_ice_thick_for_length'] = 1

        # We calibrate to zero
        df = massbalance.mb_calibration_from_scalar_mb(hef_gdir,
                                                       calibrate_param1='temp_bias',
                                                       ref_mb=0,
                                                       ref_mb_years=(1970, 2001),
                                                       write_to_gdir=False)

        cl = massbalance.ConstantMassBalance
        mb_mod = massbalance.MultipleFlowlineMassBalance(hef_gdir,
                                                         melt_f=df['melt_f'],
                                                         temp_bias=df['temp_bias'],
                                                         prcp_fac=df['prcp_fac'],
                                                         y0=1985,
                                                         mb_model_class=cl)

        # We invert again
        massbalance.apparent_mb_from_any_mb(hef_gdir, mb_model=mb_mod,
                                            mb_years=(1970, 2001))

        inversion.mass_conservation_inversion(hef_gdir,
                                              fs=inversion_params['inversion_fs'],
                                              glen_a=inversion_params['inversion_glen_a'])
        init_present_time_glacier(hef_gdir)


        fls = hef_gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                               fs=inversion_params['inversion_fs'],
                               glen_a=inversion_params['inversion_glen_a'],
                               mb_elev_feedback='never')

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_area, hef_gdir.rgi_area_km2)

        model.run_until_equilibrium(rate=1e-4)

        assert model.yr >= 30
        after_vol = model.volume_km3
        after_area = model.area_km2
        after_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_vol, after_vol, rtol=0.1)
        np.testing.assert_allclose(ref_area, after_area, rtol=0.01)
        np.testing.assert_allclose(ref_len, after_len, atol=200.01)

    @pytest.mark.slow
    def test_commitment(self, hef_gdir, inversion_params):

        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        init_present_time_glacier(hef_gdir)

        mb_mod = massbalance.ConstantMassBalance(hef_gdir, y0=2002 - 15)

        fls = hef_gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                               fs=inversion_params['inversion_fs'],
                               glen_a=inversion_params['inversion_glen_a'])

        ref_area = model.area_km2
        np.testing.assert_allclose(ref_area, hef_gdir.rgi_area_km2)

        model.run_until_equilibrium()
        assert model.yr > 100

        after_vol_1 = model.volume_km3

        init_present_time_glacier(hef_gdir)

        glacier = hef_gdir.read_pickle('model_flowlines')

        fls = hef_gdir.read_pickle('model_flowlines')
        model = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                               fs=inversion_params['inversion_fs'],
                               glen_a=inversion_params['inversion_glen_a'])

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        np.testing.assert_allclose(ref_area, hef_gdir.rgi_area_km2)

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
    def test_random(self, hef_gdir, inversion_params):

        cfg.PARAMS['store_model_geometry'] = True

        init_present_time_glacier(hef_gdir)

        # Try something else here - find out the bias needed for 0 mb
        dfo = hef_gdir.read_json('mb_calib')
        df = massbalance.mb_calibration_from_scalar_mb(hef_gdir,
                                                       calibrate_param1='temp_bias',
                                                       melt_f=dfo['melt_f'],
                                                       ref_mb=0,
                                                       ref_mb_years=(1970, 2001),
                                                       write_to_gdir=False)

        assert dfo['temp_bias'] == 0
        assert dfo['melt_f'] == df['melt_f']
        assert df['temp_bias'] < 0.5

        run_random_climate(hef_gdir, nyears=100, seed=6, y0=2002 - 15,
                           fs=inversion_params['inversion_fs'],
                           glen_a=inversion_params['inversion_glen_a'],
                           bias=0, output_filesuffix='_rdn',
                           temperature_bias=df['temp_bias'])
        run_constant_climate(hef_gdir, nyears=100, y0=2002 - 15,
                             fs=inversion_params['inversion_fs'],
                             glen_a=inversion_params['inversion_glen_a'],
                             bias=0, output_filesuffix='_ct',
                             temperature_bias=df['temp_bias'])

        paths = [hef_gdir.get_filepath('model_geometry', filesuffix='_rdn'),
                 hef_gdir.get_filepath('model_geometry', filesuffix='_ct'),
                 ]

        for path in paths:
            model = FileModel(path)
            vol = model.volume_km3_ts()
            area = model.area_km2_ts()
            np.testing.assert_allclose(vol.iloc[0], np.mean(vol),
                                       rtol=0.12)
            np.testing.assert_allclose(area.iloc[0], np.mean(area),
                                       rtol=0.1)

    @pytest.mark.slow
    def test_start_from_date(self, hef_gdir, inversion_params):

        cfg.PARAMS['store_model_geometry'] = True

        init_present_time_glacier(hef_gdir)
        run_constant_climate(hef_gdir, nyears=20, y0=1985,
                             fs=inversion_params['inversion_fs'],
                             glen_a=inversion_params['inversion_glen_a'],
                             bias=0, output_filesuffix='_ct')

        run_constant_climate(hef_gdir, nyears=10, y0=1985,
                             fs=inversion_params['inversion_fs'],
                             glen_a=inversion_params['inversion_glen_a'],
                             bias=0, output_filesuffix='_ct_1')
        run_constant_climate(hef_gdir, nyears=10, y0=1985,
                             fs=inversion_params['inversion_fs'],
                             glen_a=inversion_params['inversion_glen_a'],
                             init_model_filesuffix='_ct_1',
                             bias=0, output_filesuffix='_ct_2')

        ds = utils.compile_run_output([hef_gdir], input_filesuffix='_ct')
        ds1 = utils.compile_run_output([hef_gdir], input_filesuffix='_ct_1')
        ds2 = utils.compile_run_output([hef_gdir], input_filesuffix='_ct_2')

        ds_ = xr.concat([ds1.isel(time=slice(0, -1)), ds2], dim='time')
        np.testing.assert_allclose(ds.volume, ds_.volume, rtol=1e-5)

    @pytest.mark.slow
    def test_compile_calving(self, hef_gdir, gdir_calving):

        # This works because no calving output
        cfg.PARAMS['use_kcalving_for_run'] = False
        init_present_time_glacier(hef_gdir)
        init_present_time_glacier(gdir_calving)
        run_constant_climate(hef_gdir, nyears=10, y0=1985,
                             bias=0, output_filesuffix='_def')
        run_constant_climate(gdir_calving, nyears=10, y0=1985,
                             bias=0, output_filesuffix='_def')
        utils.compile_run_output([gdir_calving, hef_gdir],
                                 input_filesuffix='_def',
                                 tmp_file_size=1)

        # This should work although one calves the other not
        cfg.PARAMS['use_kcalving_for_run'] = True
        init_present_time_glacier(hef_gdir)
        init_present_time_glacier(gdir_calving)
        run_constant_climate(hef_gdir, nyears=10, y0=1985,
                             bias=0, output_filesuffix='_def')
        run_constant_climate(gdir_calving, nyears=10, water_level=0,
                             bias=0, y0=1985, output_filesuffix='_def')
        utils.compile_run_output([gdir_calving, hef_gdir],
                                 input_filesuffix='_def',
                                 tmp_file_size=1)

    def test_start_from_spinup(self, hef_gdir):

        init_present_time_glacier(hef_gdir)
        cfg.PARAMS['store_model_geometry'] = True

        fls = hef_gdir.read_pickle('model_flowlines')
        vol = 0
        area = 0
        for fl in fls:
            vol += fl.volume_km3
            area += fl.area_km2
        assert hef_gdir.rgi_date == 2003

        # Make a dummy run for 0 years
        run_from_climate_data(hef_gdir, ye=2004, output_filesuffix='_1')

        fp = hef_gdir.get_filepath('model_geometry', filesuffix='_1')
        fmod = FileModel(fp)
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area)
        np.testing.assert_allclose(fmod.volume_km3, vol)

        # Again
        run_from_climate_data(hef_gdir, ye=2004, init_model_filesuffix='_1',
                              output_filesuffix='_2')
        fp = hef_gdir.get_filepath('model_geometry', filesuffix='_2')
        fmod = FileModel(fp)
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area)
        np.testing.assert_allclose(fmod.volume_km3, vol)

    def test_start_from_spinup_minmax_ys(self, hef_gdir):

        init_present_time_glacier(hef_gdir)
        cfg.PARAMS['store_model_geometry'] = True

        fls = hef_gdir.read_pickle('model_flowlines')
        vol = 0
        area = 0
        for fl in fls:
            vol += fl.volume_km3
            area += fl.area_km2
        assert hef_gdir.rgi_date == 2003

        # Make a dummy run for 0 years
        run_from_climate_data(hef_gdir, ye=2002, max_ys=2002,
                              output_filesuffix='_1')

        fp = hef_gdir.get_filepath('model_geometry', filesuffix='_1')
        fmod = FileModel(fp)
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area)
        np.testing.assert_allclose(fmod.volume_km3, vol)

        # Again
        run_from_climate_data(hef_gdir, ye=2005, min_ys=2005,
                              output_filesuffix='_2')
        fp = hef_gdir.get_filepath('model_geometry', filesuffix='_2')
        fmod = FileModel(fp)
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area)
        np.testing.assert_allclose(fmod.volume_km3, vol)

        # Again
        run_from_climate_data(hef_gdir, ys=2002, ye=2003,
                              init_model_filesuffix='_1',
                              output_filesuffix='_3')
        fp = hef_gdir.get_filepath('model_geometry', filesuffix='_3')
        fmod = FileModel(fp)
        fmod.run_until(fmod.last_yr)
        np.testing.assert_allclose(fmod.area_km2, area, rtol=0.05)
        np.testing.assert_allclose(fmod.volume_km3, vol, rtol=0.05)

        # Again to check that time is correct
        run_from_climate_data(hef_gdir, ys=None, ye=None,
                              init_model_filesuffix='_1',
                              output_filesuffix='_4')
        fp = hef_gdir.get_filepath('model_geometry', filesuffix='_4')
        fmod = FileModel(fp)
        assert fmod.y0 == 2002
        assert fmod.last_yr == 2003

    @pytest.mark.slow
    def test_cesm(self, hef_gdir):

        cfg.PARAMS['store_model_geometry'] = True

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
        fh = gdir.get_filepath('climate_historical')
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
            # And also the annual cycle
            scru = shist.groupby('time.month').mean(dim='time')
            scesm = scesm.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(scru.temp, scesm.temp, rtol=5e-3)
            np.testing.assert_allclose(scru.prcp, scesm.prcp, rtol=1e-3)

        # Mass balance models
        mb_cru = massbalance.MonthlyTIModel(gdir)
        mb_cesm = massbalance.MonthlyTIModel(gdir, filename='gcm_data')

        # Average over 1961-1990
        h, w = gdir.get_inversion_flowline_hw()
        yrs = np.arange(1961, 1991)
        ts1 = mb_cru.get_specific_mb(h, w, year=yrs)
        ts2 = mb_cesm.get_specific_mb(h, w, year=yrs)
        # due to nonlinear effects the MBs are not equivalent! See if they
        # aren't too far:
        assert np.abs(np.mean(ts1) - np.mean(ts2)) < 100

        # For my own interest, some statistics
        yrs = np.arange(1851, 2003)
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

        # Do a spinup run
        run_constant_climate(gdir, nyears=100, y0=1985,
                             temperature_bias=-1,
                             output_filesuffix='_spinup')
        run_from_climate_data(gdir, ys=1961, ye=1990,
                              init_model_filesuffix='_spinup',
                              output_filesuffix='_afterspinup')
        ds3 = utils.compile_run_output([gdir], path=False,
                                       input_filesuffix='_afterspinup')
        assert (ds1.volume.isel(rgi_id=0, time=-1).data <
                0.85 * ds3.volume.isel(rgi_id=0, time=-1).data)
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
    def test_elevation_feedback(self, hef_gdir):

        init_present_time_glacier(hef_gdir)

        feedbacks = ['annual', 'monthly', 'always', 'never']
        # Mutliproc
        tasks = []
        for feedback in feedbacks:
            tasks.append((run_random_climate,
                          dict(nyears=200, seed=5,
                               y0=1985, temperature_bias=-0.5,
                               mb_elev_feedback=feedback,
                               output_filesuffix=feedback,
                               store_monthly_step=True)))
        with warnings.catch_warnings():
            # Warning about MB model update
            warnings.filterwarnings("ignore", category=UserWarning)
            workflow.execute_parallel_tasks(hef_gdir, tasks)

        out = []
        for feedback in feedbacks:
            out.append(utils.compile_run_output([hef_gdir], path=False,
                                                input_filesuffix=feedback))

        # Check that volume isn't so different
        assert_allclose(out[0].volume, out[1].volume, rtol=0.05)
        assert_allclose(out[0].volume, out[2].volume, rtol=0.05)
        assert_allclose(out[1].volume, out[2].volume, rtol=0.05)
        # Except for "never", where things are different and less variable
        assert out[3].volume.min() > out[2].volume.min()
        assert out[3].volume.max() < out[2].volume.max()

        if do_plot:
            plt.figure()
            for ds, lab in zip(out, feedbacks):
                (ds.volume * 1e-9).plot(label=lab)
            plt.xlabel('Vol (km3)')
            plt.legend()
            plt.show()


@pytest.mark.usefixtures('with_class_wd')
class TestDynamicSpinup:

    @pytest.mark.parametrize('minimise_for', ['area', 'volume'])
    @pytest.mark.slow
    @pytest.mark.skipif(not has_shapely2, reason="requires shapely2")
    def test_run_dynamic_spinup(self, hef_gdir, minimise_for):

        # value we want to match after dynamic spinup
        fls = hef_gdir.read_pickle('model_flowlines')
        ref_value = 0
        if minimise_for == 'area':
            unit = 'km2'
        elif minimise_for == 'volume':
            unit = 'km3'
        else:
            raise ValueError('Unknown variable to minimise for!')
        var_name = f'{minimise_for}_{unit}'
        for fl in fls:
            ref_value += getattr(fl, var_name)

        precision_percent = 10
        # this value is chosen in a way that it effects the result in the 'area'
        # run but not in the 'volume' run
        precision_absolute = 0.1
        min_ice_thickness = 10
        assert hef_gdir.rgi_date == 2003
        # is needed because the test climate dataset has ye = 2003
        yr_rgi = 2002
        # test version were the whole model evolution is saved and when it is
        # not saved
        for store_model_evolution in [True, False]:
            model_dynamic_spinup = run_dynamic_spinup(
                hef_gdir,
                target_yr=yr_rgi,
                minimise_for=minimise_for,
                precision_percent=precision_percent,
                precision_absolute=precision_absolute,
                min_ice_thickness=min_ice_thickness,
                output_filesuffix='_dynamic_spinup',
                store_model_evolution=store_model_evolution)

            # check if resulting model match wanted value with prescribed precision
            if var_name == 'area_km2':
                model_value = np.sum(
                    [np.sum(fl.bin_area_m2[fl.thick > min_ice_thickness])
                     for fl in model_dynamic_spinup.fls]) * 1e-6
            elif var_name == 'volume_km3':
                model_value = np.sum(
                    [np.sum((fl.section * fl.dx_meter)[fl.thick > min_ice_thickness])
                     for fl in model_dynamic_spinup.fls]) * 1e-9
            else:
                raise NotImplementedError(f'{var_name}')
            assert np.isclose(model_value, ref_value,
                              rtol=precision_percent / 100, atol=0)
            assert np.isclose(model_value, ref_value,
                              rtol=0, atol=precision_absolute)
            assert model_dynamic_spinup.yr == yr_rgi
            assert len(model_dynamic_spinup.fls) == len(fls)
            # but surface_h should not be the same
            # (also checks all individual flowlines has same number of grid points)
            assert not np.allclose(model_dynamic_spinup.fls[0].surface_h,
                                   fls[0].surface_h)
            assert not np.allclose(model_dynamic_spinup.fls[1].surface_h,
                                   fls[1].surface_h)
            assert not np.allclose(model_dynamic_spinup.fls[2].surface_h,
                                   fls[2].surface_h)

            # check if stuff is saved in model diagnostics
            gdir_diagnostics = hef_gdir.get_diagnostics()
            assert 'temp_bias_dynamic_spinup' in gdir_diagnostics.keys()
            assert 'dynamic_spinup_period' in gdir_diagnostics.keys()
            assert 'dynamic_spinup_forward_model_iterations' in gdir_diagnostics.keys()
            mismatch_key = f'{minimise_for}_mismatch_dynamic_spinup_{unit}_percent'
            assert mismatch_key in gdir_diagnostics.keys()
            assert 'dynamic_spinup_other_variable_reference' in \
                   gdir_diagnostics.keys()
            assert 'dynamic_spinup_mismatch_other_variable_percent' in \
                   gdir_diagnostics.keys()

            # check if model geometry is correctly saved in gdir with
            fp = hef_gdir.get_filepath('model_geometry',
                                       filesuffix='_dynamic_spinup')
            fmod = FileModel(fp)
            if store_model_evolution:
                assert len(fmod.years) > 1
            else:
                assert len(fmod.years) == 1
            fmod.run_until(fmod.last_yr)
            assert np.isclose(getattr(model_dynamic_spinup, var_name),
                              getattr(fmod, var_name))
            assert fmod.last_yr == yr_rgi
            assert len(model_dynamic_spinup.fls) == len(fmod.fls)

        # test user provided target year and value
        target_yr = 2000
        if minimise_for == 'area':
            ref_value = 8.5
        elif minimise_for == 'volume':
            ref_value = 0.6
        model_dynamic_spinup_target_yr = run_dynamic_spinup(
            hef_gdir,
            target_yr=target_yr,
            target_value=ref_value,
            minimise_for=minimise_for,
            precision_percent=precision_percent,
            precision_absolute=precision_absolute,
            min_ice_thickness=min_ice_thickness,
            output_filesuffix='_dynamic_spinup',
            store_model_evolution=store_model_evolution)

        # check if resulting model match wanted value with prescribed precision
        if var_name == 'area_km2':
            model_value = np.sum(
                [np.sum(fl.bin_area_m2[fl.thick > min_ice_thickness])
                 for fl in model_dynamic_spinup_target_yr.fls]) * 1e-6
        elif var_name == 'volume_km3':
            model_value = np.sum(
                [np.sum((fl.section * fl.dx_meter)[fl.thick > min_ice_thickness])
                 for fl in model_dynamic_spinup_target_yr.fls]) * 1e-9
        else:
            raise NotImplementedError(f'{var_name}')
        assert np.isclose(model_value, ref_value,
                          rtol=precision_percent / 100, atol=0)
        assert np.isclose(model_value, ref_value,
                          rtol=0, atol=precision_absolute)
        assert model_dynamic_spinup_target_yr.yr == target_yr
        assert len(model_dynamic_spinup_target_yr.fls) == len(fls)

        # test if spinup_start_yr is handled correctly and overrides the spinup_period
        spinup_start_yr = yr_rgi - 20
        model_dynamic_spinup_ys = run_dynamic_spinup(
            hef_gdir,
            spinup_period=40,
            spinup_start_yr=spinup_start_yr,
            target_yr=yr_rgi,
            minimise_for=minimise_for,
            precision_percent=precision_percent,
            precision_absolute=precision_absolute,
            min_ice_thickness=min_ice_thickness,
            output_filesuffix='_dynamic_spinup_ys', )
        # check that is the same if we provide spinup_start_yr instead of spinup_period
        assert np.isclose(model_dynamic_spinup_ys.volume_km3,
                          model_dynamic_spinup.volume_km3)
        assert np.isclose(model_dynamic_spinup_ys.area_km2,
                          model_dynamic_spinup.area_km2)
        assert np.isclose(model_dynamic_spinup_ys.yr,
                          model_dynamic_spinup.yr)

        # Here start with test if errors are handled correctly by the dynamic
        # spinup function and if 'ignore_errors' works

        # create a flowline with zero ice
        fls_zero_ice = hef_gdir.read_pickle('model_flowlines')
        for i in range(len(fls_zero_ice)):
            fls_zero_ice[i].section = np.zeros(len(fls_zero_ice[i].section))

        # first we artificially produce some errors in hef_gdir using extreme kwargs
        error_settings = {
            'Not able to conduct one error free run. Error is "ice_free"':
                {'first_guess_t_bias': 100},
            'The difference between the rgi_date and the start year of the '
            'climate data is too small to run a dynamic spinup!':
                {'min_spinup_period': 300},
            'The given reference value is Zero, no dynamic spinup possible!':
                {'init_model_fls': fls_zero_ice},
            'Not able to conduct one error free run. Error is "out_of_domain"':
                {'first_guess_t_bias': -100},
            'Could not find mismatch smaller 0.1%':
                {'precision_percent': 0.1}
        }

        for err_msg, kwarg_dyn_spn in error_settings.items():
            # test that error is thrown
            ignore_errors=False
            with pytest.raises(RuntimeError,
                               match=err_msg):
                run_dynamic_spinup(
                    hef_gdir,
                    minimise_for=minimise_for,
                    target_yr=2002,
                    ye=2002,
                    ignore_errors=ignore_errors,
                    spinup_period=10,
                    maxiter=2,
                    output_filesuffix='_dynamic_spinup',
                    **kwarg_dyn_spn)
            for filename in ['model_geometry', 'fl_diagnostics',
                             'model_diagnostics']:
                assert not os.path.exists(
                    hef_gdir.get_filepath(filename,
                                          filesuffix='_dynamic_spinup', ))

            # check that it passes with ignore_errors=True
            ignore_errors=True

            model = run_dynamic_spinup(
                hef_gdir,
                minimise_for=minimise_for,
                target_yr=2002,
                ye=2002,
                ignore_errors=ignore_errors,
                maxiter=2,
                output_filesuffix='_dynamic_spinup',
                **kwarg_dyn_spn)

            fmod = FileModel(fp)
            fmod.run_until(fmod.last_yr)
            assert np.isclose(getattr(model, var_name),
                              getattr(fmod, var_name))
            yr_min = hef_gdir.get_climate_info()['baseline_yr_0']
            yr_rgi = 2002
            assert fmod.last_yr == np.clip(yr_rgi, yr_min, None)
            assert len(model.fls) == len(fmod.fls)

        yr_rgi = 2000
        yr_min = hef_gdir.get_climate_info()['baseline_yr_0']
        ye = hef_gdir.get_climate_info()['baseline_yr_1'] + 1
        precision_percent = 1
        precision_absolute = 1
        model_dynamic_spinup_ye, t_bias = run_dynamic_spinup(
            hef_gdir,
            spinup_period=40,
            spinup_start_yr=spinup_start_yr,
            target_yr=yr_rgi,
            ye=ye,
            return_t_bias_best=True,
            minimise_for=minimise_for,
            precision_percent=precision_percent,
            precision_absolute=precision_absolute,
            output_filesuffix='_dynamic_spinup_historical', )

        assert isinstance(t_bias, float)
        assert model_dynamic_spinup_ye.yr == ye
        ds = utils.compile_run_output(
            hef_gdir, input_filesuffix='_dynamic_spinup_historical', path=False)
        if minimise_for == 'volume':
            assert np.isclose(ds.loc[{'time': yr_rgi}].volume.values,
                              model_dynamic_spinup.volume_m3,
                              rtol=precision_percent,
                              atol=precision_absolute)
        elif minimise_for == 'area':
            assert np.isclose(ds.loc[{'time': yr_rgi}].area.values,
                              model_dynamic_spinup.area_m2,
                              rtol=precision_percent,
                              atol=precision_absolute)
        else:
            raise ValueError(f'Unknown parameter for minimise for '
                             f'"{minimise_for}"!')

        # test parameter and spinup_start_yr_max, should override spinup_period
        # to start at spinup_start_yr_max
        model_dynamic_spinup_max_start_yr = run_dynamic_spinup(
            hef_gdir,
            spinup_period=5,
            spinup_start_yr=None,
            spinup_start_yr_max=1990,
            target_yr=yr_rgi,
            minimise_for=minimise_for,
            precision_percent=precision_percent,
            precision_absolute=precision_absolute,
            min_ice_thickness=min_ice_thickness,
            output_filesuffix='_dynamic_spinup_max_start', )
        ds = utils.compile_run_output(
            hef_gdir, input_filesuffix='_dynamic_spinup_max_start', path=False)
        assert ds.time.min().values == 1990

        # test that provided start_yr_max is inside climate data
        with pytest.raises(RuntimeError,
                           match='The provided maximum start year *'):
            run_dynamic_spinup(
                hef_gdir,
                minimise_for=minimise_for,
                spinup_start_yr_max=yr_min - 1)

        # test that provided start year is smaller than start_yr_max
        with pytest.raises(RuntimeError,
                           match='The provided start year *'):
            run_dynamic_spinup(
                hef_gdir,
                minimise_for=minimise_for,
                spinup_start_yr_max=yr_rgi - 10,
                spinup_start_yr=yr_rgi - 5)

        # test that provided ye is larger than target_yr
        with pytest.raises(RuntimeError,
                           match='The provided end year *'):
            run_dynamic_spinup(
                hef_gdir,
                minimise_for=minimise_for,
                target_yr=yr_rgi,
                ye=yr_rgi - 1)

        # test if provided model geometry works and some other principle
        # parameter tests (use_inversion_params_for_run and
        # store_model_geometry)
        cfg.PARAMS['use_inversion_params_for_run'] = False
        cfg.PARAMS['store_model_geometry'] = True
        workflow.execute_entity_task(tasks.run_from_climate_data, [hef_gdir],
                                     ys=yr_rgi - 1, ye=yr_rgi,
                                     output_filesuffix='_one_yr')
        run_dynamic_spinup(
            hef_gdir,
            minimise_for=minimise_for,
            init_model_filesuffix='_one_yr',
            init_model_yr=yr_rgi - 1,
            target_yr=yr_rgi,
            store_model_geometry=False)

        # test that error is raised if mb_elev_feedback not annual
        with pytest.raises(InvalidParamsError,
                           match='Only use annual mb_elev_feedback with the '
                                 'dynamic spinup function!'):
            run_dynamic_spinup(
                hef_gdir,
                minimise_for=minimise_for,
                mb_elev_feedback='monthly')

        # test that error is raised if used together with calving
        cfg.PARAMS['use_kcalving_for_run'] = True
        with pytest.raises(InvalidParamsError,
                           match='Dynamic spinup not tested with *'):
            run_dynamic_spinup(
                hef_gdir,
                minimise_for=minimise_for)
        cfg.PARAMS['use_kcalving_for_run'] = False

        # test that fixed_geometry_spinup is added correctly if spinup period
        # is shorten due to too large precision
        if minimise_for == 'area':
            run_dynamic_spinup(
                hef_gdir,
                spinup_start_yr=1979,
                precision_percent=0.00012,
                minimise_for=minimise_for,
                output_filesuffix='_without_fixed_spinup',
                target_yr=yr_rgi,
                add_fixed_geometry_spinup=False)
            run_without_fixed_spinup = utils.compile_run_output(
                hef_gdir, input_filesuffix='_without_fixed_spinup', path=False)

            # now add fixed spinup to the whole period
            run_dynamic_spinup(
                hef_gdir,
                spinup_start_yr=1979,
                precision_percent=0.00012,
                minimise_for=minimise_for,
                output_filesuffix='_with_fixed_spinup',
                target_yr=yr_rgi,
                add_fixed_geometry_spinup=True)
            run_with_fixed_spinup = utils.compile_run_output(
                hef_gdir, input_filesuffix='_with_fixed_spinup', path=False)
            assert (run_without_fixed_spinup.time.values[0] >=
                    run_with_fixed_spinup.time.values[0])
            assert run_with_fixed_spinup.time.values[0] == 1979

    @pytest.mark.parametrize('minimise_for', ['area', 'volume'])
    @pytest.mark.slow
    @pytest.mark.skip
    @pytest.mark.skipif(not has_shapely2, reason="requires shapely2")
    def test_run_dynamic_spinup_special_cases(self, hef_gdir, minimise_for):

        if minimise_for == 'area':
            unit = 'km2'
        elif minimise_for == 'volume':
            unit = 'km3'
        else:
            raise ValueError('Unknown variable to minimise for!')
        var_name = f'{minimise_for}_{unit}'

        # for some errors we need to use other glaciers
        rgi_ids = {
            'RGI60-04.03249': 'Not able to minimise without ice '
                              'free glacier',
            'RGI60-04.03109': 'Not able to minimise! Problem is unknown, '
                              'need to check by hand!',
            'RGI60-04.02180': 'Not able to minimise without '
                              'exceeding the domain!',
        }
        gdirs = workflow.init_glacier_directories(
            rgi_ids.keys(), from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')

        for gdir in gdirs:
            # Test that the correct error is raised
            ignore_errors = False
            with pytest.raises(RuntimeError,
                               match=rgi_ids[gdir.rgi_id]):
                run_dynamic_spinup(gdir,
                                   minimise_for=minimise_for,
                                   output_filesuffix='_dynamic_spinup',
                                   maxiter=10,
                                   ignore_errors=ignore_errors,
                                   )
            # check that all _dynamic_spinup files are deleted if error occurred
            for filename in ['model_geometry', 'fl_diagnostics',
                             'model_diagnostics']:
                assert not os.path.exists(
                    gdir.get_filepath(filename,
                                      filesuffix='_dynamic_spinup', ))

            # check that ignore_error is working correctly
            ignore_errors = True

            model_dynamic_spinup_error, t_bias_best = run_dynamic_spinup(
                gdir,
                minimise_for=minimise_for,
                output_filesuffix='_dynamic_spinup',
                maxiter=10,
                ignore_errors=ignore_errors,
                return_t_bias_best=True)

            # check if model geometry is correctly saved in gdir
            fp = gdir.get_filepath('model_geometry',
                                   filesuffix='_dynamic_spinup')
            fmod = FileModel(fp)
            fmod.run_until(fmod.last_yr)
            assert np.isclose(getattr(model_dynamic_spinup_error, var_name),
                              getattr(fmod, var_name))
            yr_min = gdir.get_climate_info()['baseline_yr_0']
            yr_rgi = gdir.rgi_date + 1  # convert to hydro year
            assert fmod.last_yr == np.clip(yr_rgi, yr_min, None)
            assert len(model_dynamic_spinup_error.fls) == len(fmod.fls)
            assert np.isnan(t_bias_best)

    @pytest.mark.parametrize('do_inversion', [True, False])
    @pytest.mark.parametrize('minimise_for', ['area', 'volume'])
    @pytest.mark.slow
    @pytest.mark.skipif(not has_shapely2, reason="requires shapely2")
    def test_run_dynamic_melt_f_calibration_with_dynamic_spinup(self,
                                                                minimise_for,
                                                                do_inversion):

        # use a prepro dir as the hef_gdir climate data only goes to 2003 and
        # for the geodetic data we need climate data up to 2020
        gdir = workflow.init_glacier_directories(
            ['RGI60-11.00897'],  # Hintereisferner
            from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')[0]

        # save original melt_f to be able to reset back to default for testing
        melt_f_orig = gdir.read_json('mb_calib')['melt_f']

        def reset_melt_f():
            mb_calib = gdir.read_json('mb_calib')
            mb_calib['melt_f'] = melt_f_orig
            gdir.write_json(mb_calib, 'mb_calib')

        # value we want to match after dynamic melt_f calibration with dynamic
        # spinup
        fls = gdir.read_pickle('model_flowlines')
        ref_value_dynamic_spinup = 0
        if minimise_for == 'area':
            unit = 'km2'
        elif minimise_for == 'volume':
            unit = 'km3'
        else:
            raise ValueError('Unknown variable to minimise for!')
        var_name = f'{minimise_for}_{unit}'
        for fl in fls:
            ref_value_dynamic_spinup += getattr(fl, var_name)

        ref_period = cfg.PARAMS['geodetic_mb_period']

        yr0_ref_dmdtda, yr1_ref_dmdtda = ref_period.split('_')
        yr0_ref_dmdtda = int(yr0_ref_dmdtda.split('-')[0])
        yr1_ref_dmdtda = int(yr1_ref_dmdtda.split('-')[0])

        df_ref_dmdtda = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        sel = df_ref_dmdtda.loc[df_ref_dmdtda['period'] == ref_period].iloc[0]
        ref_dmdtda = float(sel['dmdtda'])
        ref_dmdtda *= 1000  # kg m-2 yr-1
        err_ref_dmdtda = float(sel['err_dmdtda'])
        err_ref_dmdtda *= 1000  # kg m-2 yr-1

        if do_inversion:
            # before the run, check that the dyn model flowlines does not exist
            # only important if inversion is included, so original
            # model_flowlines are unchagned (to be able to conduct more dynamic
            # calibration runs in the same gdir)
            assert not os.path.isfile(
                os.path.join(gdir.dir, 'model_flowlines_dyn_melt_f_calib.pkl'))

        # conduct a run including a dynamic spinup and inversion
        melt_f_max = 1000 * 12 / 365
        precision_percent = 10
        precision_absolute = 0.1
        ye = gdir.get_climate_info()['baseline_yr_1'] + 1
        yr_rgi = gdir.rgi_date
        run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run_with_dynamic_spinup,
            kwargs_run_function={'minimise_for': minimise_for,
                                 'precision_percent': precision_percent,
                                 'precision_absolute': precision_absolute,
                                 'do_inversion': do_inversion},
            fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
            kwargs_fallback_function={'minimise_for': minimise_for,
                                      'precision_percent': precision_percent,
                                      'precision_absolute': precision_absolute,
                                      'do_inversion': do_inversion},
            output_filesuffix='_dyn_melt_f_calib_spinup_inversion',
            ys=1979, ye=ye)

        # check that we are matching all desired ref values
        ds = utils.compile_run_output(
            gdir, input_filesuffix='_dyn_melt_f_calib_spinup_inversion',
            path=False)
        if minimise_for == 'volume':
            assert np.isclose(ds.loc[{'time': yr_rgi}].volume.values * 1e-9,
                              ref_value_dynamic_spinup,
                              rtol=precision_percent / 100,
                              atol=precision_absolute)
        elif minimise_for == 'area':
            assert np.isclose(ds.loc[{'time': yr_rgi}].area.values * 1e-6,
                              ref_value_dynamic_spinup,
                              rtol=precision_percent / 100,
                              atol=precision_absolute)
        dmdtda_mdl = ((ds.volume.loc[yr1_ref_dmdtda].values -
                       ds.volume.loc[yr0_ref_dmdtda].values) /
                      gdir.rgi_area_m2 /
                      (yr1_ref_dmdtda - yr0_ref_dmdtda) *
                      cfg.PARAMS['ice_density'])
        assert np.isclose(dmdtda_mdl, ref_dmdtda,
                          rtol=np.abs(err_ref_dmdtda / ref_dmdtda))
        assert gdir.get_diagnostics()['used_spinup_option'] == \
               'dynamic melt_f calibration (full success)'

        if do_inversion:
            # after the run, check that the dyn model flowlines exists and that
            # the original model flowlines are unchanged
            assert os.path.isfile(
                os.path.join(gdir.dir, 'model_flowlines_dyn_melt_f_calib.pkl'))
            assert np.all([np.all(getattr(fl_prev, 'surface_h') ==
                                  getattr(fl_now, 'surface_h')) and
                           np.all(getattr(fl_prev, 'bed_h') ==
                                  getattr(fl_now, 'bed_h'))
                           for fl_prev, fl_now in
                           zip(fls, gdir.read_pickle('model_flowlines'))])

        # test that error is raised if ignore_error=False
        reset_melt_f()

        with pytest.raises(RuntimeError,
                           match='Dynamic melt_f calibration not successful.*'):
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                run_function=dynamic_melt_f_run_with_dynamic_spinup,
                kwargs_run_function={'minimise_for': minimise_for,
                                     'do_inversion': do_inversion,
                                     'precision_percent': 6,
                                     'maxiter': 2},
                fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
                kwargs_fallback_function={'minimise_for': minimise_for,
                                          'do_inversion': do_inversion,
                                          'precision_percent': 6,
                                          'maxiter': 2},
                output_filesuffix='_dyn_melt_f_calib_spinup_inversion_error',
                ignore_errors=False,
                ref_dmdtda=ref_dmdtda, err_ref_dmdtda=0.000001,
                maxiter=2)

        # test that error is raised if no dict is provided for local_variables
        # in dynamic_melt_f_run_with_dynamic_spinup
        for local_variables in [None, []]:
            with pytest.raises(ValueError,
                               match='You must provide a dict for '
                                     'local_variables.*'):
                dynamic_melt_f_run_with_dynamic_spinup(
                    gdir,
                    melt_f=gdir.read_json('mb_calib')['melt_f'],
                    yr0_ref_mb=yr0_ref_dmdtda,
                    yr1_ref_mb=yr1_ref_dmdtda,
                    fls_init=fls, ys=1980, ye=2020, do_inversion=do_inversion,
                    local_variables=local_variables)

        # test that error is raised if user provided dmdtda is given without an
        # error and vice versa
        for use_ref_dmdtda, use_err_ref_dmdtda in zip([ref_dmdtda, None],
                                                      [None, err_ref_dmdtda]):
            with pytest.raises(RuntimeError,
                               match='If you provide a reference geodetic '
                                     'mass balance .*'):
                run_dynamic_melt_f_calibration(
                    gdir, melt_f_max=melt_f_max,
                    ref_dmdtda=use_ref_dmdtda,
                    err_ref_dmdtda=use_err_ref_dmdtda)

        # test that error is raised if user provided dmdtda error is 0 or
        # negative
        for use_err_ref_dmdtda in [0., -0.1]:
            with pytest.raises(RuntimeError,
                               match='The provided error for the geodetic '
                                     'mass-balance.*'):
                run_dynamic_melt_f_calibration(
                    gdir, melt_f_max=melt_f_max,
                    ref_dmdtda=ref_dmdtda,
                    err_ref_dmdtda=use_err_ref_dmdtda)

        # test if fallback raise error if no local variable provided
        with pytest.raises(RuntimeError,
                           match='Need the volume to do *'):
            dynamic_melt_f_run_with_dynamic_spinup_fallback(
                gdir,
                melt_f=gdir.read_json('mb_calib')['melt_f'],
                fls_init=gdir.read_pickle('model_flowlines'),
                ys=gdir.get_climate_info()['baseline_yr_0'],
                ye=gdir.get_climate_info()['baseline_yr_1'] + 1,
                local_variables=None,
                minimise_for=minimise_for
            )

    @pytest.mark.parametrize('do_inversion', [True, False])
    @pytest.mark.parametrize('minimise_for', ['area', 'volume'])
    @pytest.mark.slow
    @pytest.mark.skip
    @pytest.mark.skipif(not has_shapely2, reason="requires shapely2")
    def test_run_dynamic_melt_f_calibration_with_dynamic_spinup_special_cases(
            self, minimise_for, do_inversion):

        # use a prepro dir as the hef_gdir climate data only goes to 2003 and
        # for the geodetic data we need climate data up to 2020
        gdir = workflow.init_glacier_directories(
            ['RGI60-11.00897'],  # Hintereisferner
            from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')[0]

        # save original melt_f to be able to reset back to default for testing
        melt_f_orig = gdir.read_json('mb_calib')['melt_f']

        def reset_melt_f():
            mb_calib = gdir.read_json('mb_calib')
            mb_calib['melt_f'] = melt_f_orig
            gdir.write_json(mb_calib, 'mb_calib')

        # value we want to match after dynamic melt_f calibration with dynamic
        # spinup
        fls = gdir.read_pickle('model_flowlines')
        ref_value_dynamic_spinup = 0
        if minimise_for == 'area':
            unit = 'km2'
        elif minimise_for == 'volume':
            unit = 'km3'
        else:
            raise ValueError('Unknown variable to minimise for!')
        var_name = f'{minimise_for}_{unit}'
        for fl in fls:
            ref_value_dynamic_spinup += getattr(fl, var_name)

        ref_period = cfg.PARAMS['geodetic_mb_period']

        yr0_ref_dmdtda, yr1_ref_dmdtda = ref_period.split('_')
        yr0_ref_dmdtda = int(yr0_ref_dmdtda.split('-')[0])
        yr1_ref_dmdtda = int(yr1_ref_dmdtda.split('-')[0])

        df_ref_dmdtda = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        sel = df_ref_dmdtda.loc[df_ref_dmdtda['period'] == ref_period]
        ref_dmdtda = float(sel['dmdtda'])
        ref_dmdtda *= 1000  # kg m-2 yr-1
        err_ref_dmdtda = float(sel['err_dmdtda'])
        err_ref_dmdtda *= 1000  # kg m-2 yr-1

        melt_f_max = 1000 * 12 / 365
        precision_percent = 10
        precision_absolute = 0.1
        ye = gdir.get_climate_info()['baseline_yr_1'] + 1
        yr_rgi = gdir.rgi_date

        # successful run to compare to
        run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run_with_dynamic_spinup,
            kwargs_run_function={'minimise_for': minimise_for,
                                 'precision_percent': precision_percent,
                                 'precision_absolute': precision_absolute,
                                 'do_inversion': do_inversion},
            fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
            kwargs_fallback_function={'minimise_for': minimise_for,
                                      'precision_percent': precision_percent,
                                      'precision_absolute': precision_absolute,
                                      'do_inversion': do_inversion},
            output_filesuffix='_dyn_melt_f_calib_spinup_inversion',
            ys=1979, ye=ye)

        # check that we are matching all desired ref values
        ds = utils.compile_run_output(
            gdir, input_filesuffix='_dyn_melt_f_calib_spinup_inversion',
            path=False)
        dmdtda_mdl = ((ds.volume.loc[yr1_ref_dmdtda].values -
                       ds.volume.loc[yr0_ref_dmdtda].values) /
                      gdir.rgi_area_m2 /
                      (yr1_ref_dmdtda - yr0_ref_dmdtda) *
                      cfg.PARAMS['ice_density'])

        # test err_dmdtda_scaling_factor (not working for volume with inversion)
        if not (do_inversion and minimise_for == 'volume'):
            err_dmdtda_scaling_factor = 0.2
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                err_dmdtda_scaling_factor=err_dmdtda_scaling_factor,
                run_function=dynamic_melt_f_run_with_dynamic_spinup,
                kwargs_run_function={'minimise_for': minimise_for,
                                     'precision_percent': precision_percent,
                                     'precision_absolute': precision_absolute,
                                     'do_inversion': do_inversion},
                fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
                kwargs_fallback_function={'minimise_for': minimise_for,
                                          'precision_percent': precision_percent,
                                          'precision_absolute': precision_absolute,
                                          'do_inversion': do_inversion},
                output_filesuffix='_dyn_melt_f_calib_err_scaling',
                ys=1979, ye=ye)

            # check that we are matching all desired ref values
            ds = utils.compile_run_output(
                gdir, input_filesuffix='_dyn_melt_f_calib_err_scaling',
                path=False)
            if minimise_for == 'volume':
                assert np.isclose(ds.loc[{'time': yr_rgi}].volume.values * 1e-9,
                                  ref_value_dynamic_spinup,
                                  rtol=precision_percent / 100,
                                  atol=precision_absolute)
            elif minimise_for == 'area':
                assert np.isclose(ds.loc[{'time': yr_rgi}].area.values * 1e-6,
                                  ref_value_dynamic_spinup,
                                  rtol=precision_percent / 100,
                                  atol=precision_absolute)
            dmdtda_mdl_scale = ((ds.volume.loc[yr1_ref_dmdtda].values -
                                 ds.volume.loc[yr0_ref_dmdtda].values) /
                                gdir.rgi_area_m2 /
                                (yr1_ref_dmdtda - yr0_ref_dmdtda) *
                                cfg.PARAMS['ice_density'])
            assert np.isclose(dmdtda_mdl_scale, ref_dmdtda,
                              rtol=np.abs(err_ref_dmdtda *
                                          err_dmdtda_scaling_factor / ref_dmdtda))
            # check that calibration without scaling factor is outside adapted
            # uncertainty (use previous result without err scaling factor), this
            # tests if the scaling factor has any effect
            assert not np.isclose(dmdtda_mdl, ref_dmdtda,
                                  rtol=np.abs(err_ref_dmdtda *
                                              err_dmdtda_scaling_factor /
                                              ref_dmdtda))
            assert gdir.get_diagnostics()['used_spinup_option'] == \
                   'dynamic melt_f calibration (full success)'
            err_scaling_key = 'dmdtda_dynamic_calibration_error_scaling_factor'
            assert gdir.get_diagnostics()[err_scaling_key] == \
                   err_dmdtda_scaling_factor

        # test that error is raised if user provides flowlines but want to
        # include inversion during dynamic melt_f calibration
        if do_inversion:
            # artificial change of flowlines to force error
            fls[0].thick = np.zeros(fls[0].nx)
            with pytest.raises(InvalidWorkflowError,
                               match='If you want to perform a dynamic '
                                     'melt_f calibration including an '
                                     'inversion*'):
                run_dynamic_melt_f_calibration(
                    gdir, melt_f_max=melt_f_max,
                    run_function=dynamic_melt_f_run_with_dynamic_spinup,
                    kwargs_run_function={'minimise_for': minimise_for,
                                         'precision_percent': precision_percent,
                                         'precision_absolute': precision_absolute,
                                         'do_inversion': do_inversion},
                    fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
                    kwargs_fallback_function={'minimise_for': minimise_for,
                                              'precision_percent': precision_percent,
                                              'precision_absolute': precision_absolute,
                                              'do_inversion': do_inversion},
                    output_filesuffix='_dyn_melt_f_calib_spinup_inversion',
                    ys=1979, ye=ye, init_model_fls=fls)

        # test that fallback function works as expected if ignore_error=True and
        # if the first guess can improve (but not enough)
        model_fallback = run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run_with_dynamic_spinup,
            kwargs_run_function={'minimise_for': minimise_for,
                                 'precision_percent': precision_percent,
                                 'precision_absolute': precision_absolute,
                                 'do_inversion': do_inversion},
            fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
            kwargs_fallback_function={'minimise_for': minimise_for,
                                      'precision_percent': precision_percent,
                                      'precision_absolute': precision_absolute,
                                      'do_inversion': do_inversion},
            output_filesuffix='_dyn_melt_f_calib_spinup_inversion_error',
            ignore_errors=True,
            ref_dmdtda=ref_dmdtda, err_ref_dmdtda=0.000001,
            maxiter=2)
        assert isinstance(model_fallback, oggm.core.flowline.FluxBasedModel)
        assert gdir.get_diagnostics()['used_spinup_option'] == \
               'dynamic melt_f calibration (part success)'
        ds = utils.compile_run_output(
            gdir, input_filesuffix='_dyn_melt_f_calib_spinup_inversion_error',
            path=False)
        if minimise_for == 'volume':
            assert np.isclose(ds.loc[{'time': yr_rgi}].volume.values * 1e-9,
                              ref_value_dynamic_spinup,
                              rtol=precision_percent / 100,
                              atol=precision_absolute)
        elif minimise_for == 'area':
            assert np.isclose(ds.loc[{'time': yr_rgi}].area.values * 1e-6,
                              ref_value_dynamic_spinup,
                              rtol=precision_percent / 100,
                              atol=precision_absolute)

        # test that fallback function works as expected if ignore_error=True and
        # if no successful run can be conducted
        model_fallback = run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run_with_dynamic_spinup,
            kwargs_run_function={'minimise_for': minimise_for,
                                 'precision_percent': 0.1,
                                 'precision_absolute': 0.0001,
                                 'maxiter': 2,
                                 'do_inversion': do_inversion},
            fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
            kwargs_fallback_function={'minimise_for': minimise_for,
                                      'precision_percent': 0.1,
                                      'precision_absolute': 0.0001,
                                      'maxiter': 2,
                                      'do_inversion': do_inversion},
            output_filesuffix='_dyn_melt_f_calib_spinup_inversion_error',
            ignore_errors=True,
            ref_dmdtda=ref_dmdtda, err_ref_dmdtda=0.000001,
            maxiter=2)
        assert isinstance(model_fallback, oggm.core.flowline.FluxBasedModel)
        assert gdir.get_diagnostics()['used_spinup_option'] == \
               'fixed geometry spinup'
        if do_inversion:
            # check that the dyn model flowlines are deleted if no success
            assert not os.path.isfile(
                os.path.join(gdir.dir, 'model_flowlines_dyn_melt_f_calib.pkl'))

        # test if fallback function is resetting melt_f correctly
        mb_calib_params = gdir.read_json('mb_calib')
        original_melt_f = mb_calib_params['melt_f']
        mb_calib_params['melt_f'] = original_melt_f - 10 * 12 / 365
        gdir.write_json(mb_calib_params, 'mb_calib')
        assert original_melt_f != gdir.read_json('mb_calib')['melt_f']
        dynamic_melt_f_run_with_dynamic_spinup_fallback(
            gdir,
            melt_f=original_melt_f,
            fls_init=gdir.read_pickle('model_flowlines'),
            ys=gdir.get_climate_info()['baseline_yr_0'],
            ye=gdir.get_climate_info()['baseline_yr_1'] + 1,
            local_variables={'vol_m3_ref':
                                 gdir.read_pickle('model_flowlines')[0].volume_m3},
            minimise_for=minimise_for
        )
        assert original_melt_f == gdir.read_json('mb_calib')['melt_f']

        # only testing for area, because for volume we need to search for other
        # parameters for the test
        if minimise_for == 'area':
            # tests for user provided dmdtda
            reset_melt_f()

            delta_ref_dmdtda = 100
            delta_err_ref_dmdtda = -50
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                ref_dmdtda=ref_dmdtda + delta_ref_dmdtda,
                err_ref_dmdtda=err_ref_dmdtda + delta_err_ref_dmdtda,
                run_function=dynamic_melt_f_run_with_dynamic_spinup,
                kwargs_run_function={'minimise_for': minimise_for,
                                     'precision_percent': precision_percent,
                                     'precision_absolute': precision_absolute,
                                     'do_inversion': do_inversion},
                fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
                kwargs_fallback_function={'minimise_for': minimise_for,
                                          'precision_percent': precision_percent,
                                          'precision_absolute': precision_absolute,
                                          'do_inversion': do_inversion},
                output_filesuffix='_dyn_melt_f_calib_spinup_inversion_user_dmdtda',
                ys=1979, ye=ye)
            ds = utils.compile_run_output(
                gdir, input_filesuffix='_dyn_melt_f_calib_spinup_inversion_user_dmdtda',
                path=False)
            dmdtda_mdl = ((ds.volume.loc[yr1_ref_dmdtda].values -
                           ds.volume.loc[yr0_ref_dmdtda].values) /
                          gdir.rgi_area_m2 /
                          (yr1_ref_dmdtda - yr0_ref_dmdtda) *
                          cfg.PARAMS['ice_density'])
            assert np.isclose(dmdtda_mdl, ref_dmdtda + delta_ref_dmdtda,
                              rtol=np.abs((err_ref_dmdtda + delta_err_ref_dmdtda) /
                                          (ref_dmdtda + delta_ref_dmdtda)))

            # check if spinup_start_year_max works as expected, for this use a
            # glacier where the period is shorten
            gdir = workflow.init_glacier_directories(
                ['RGI60-11.00033'],
                from_prepro_level=3, prepro_border=160,
                prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                                'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')[0]

            df_ref_dmdtda = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
            sel = df_ref_dmdtda.loc[df_ref_dmdtda['period'] == ref_period].iloc[0]
            ref_dmdtda = float(sel['dmdtda'])
            ref_dmdtda *= 1000  # kg m-2 yr-1
            err_ref_dmdtda = float(sel['err_dmdtda'])
            err_ref_dmdtda *= 1000  # kg m-2 yr-1
            delta_ref_dmdtda = 100
            delta_err_ref_dmdtda = -50
            precision_percent = 1
            precision_absolute = 0.1
            ye = gdir.get_climate_info()['baseline_yr_1'] + 1

            # run without max spinup_start_yr_max
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                ref_dmdtda=ref_dmdtda + delta_ref_dmdtda,
                err_ref_dmdtda=err_ref_dmdtda + delta_err_ref_dmdtda,
                run_function=dynamic_melt_f_run_with_dynamic_spinup,
                kwargs_run_function={'minimise_for': minimise_for,
                                     'precision_percent': precision_percent,
                                     'precision_absolute': precision_absolute,
                                     'do_inversion': do_inversion,
                                     'add_fixed_geometry_spinup': False},
                fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
                kwargs_fallback_function={'minimise_for': minimise_for,
                                          'precision_percent': precision_percent,
                                          'precision_absolute': precision_absolute,
                                          'do_inversion': do_inversion,
                                          'add_fixed_geometry_spinup': False},
                output_filesuffix='_dyn_melt_f_calib_spinup_reduce_period_no_limit',
                ys=1979, ye=ye)
            # run with max limit
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                ref_dmdtda=ref_dmdtda + delta_ref_dmdtda,
                err_ref_dmdtda=err_ref_dmdtda + delta_err_ref_dmdtda,
                ignore_errors=True,
                run_function=dynamic_melt_f_run_with_dynamic_spinup,
                kwargs_run_function={'minimise_for': minimise_for,
                                     'spinup_start_yr_max': 1979,
                                     'add_fixed_geometry_spinup': False,
                                     'precision_percent': precision_percent,
                                     'precision_absolute': precision_absolute,
                                     'do_inversion': do_inversion},
                fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
                kwargs_fallback_function={'minimise_for': minimise_for,
                                          'spinup_start_yr_max': 1979,
                                          'add_fixed_geometry_spinup': False,
                                          'precision_percent': precision_percent,
                                          'precision_absolute': precision_absolute,
                                          'do_inversion': do_inversion},
                output_filesuffix='_dyn_melt_f_calib_spinup_reduce_period',
                ys=1979, ye=ye)

            with xr.open_dataset(
                    gdir.get_filepath(
                        'model_diagnostics',
                        filesuffix='_dyn_melt_f_calib_spinup_reduce_period_'
                                   'no_limit')) as ds:
                run_no_limit = ds.load()
            with xr.open_dataset(
                    gdir.get_filepath(
                        'model_diagnostics',
                        filesuffix='_dyn_melt_f_calib_spinup_reduce_period')) as ds:
                run_with_limit = ds.load()

            assert run_no_limit.time.values[0] > run_with_limit.time.values[0]
            assert run_with_limit.time.values[0] == 1979

            # test if add_fixed_geomtry_spinup of dynamic spinup works here as well
            # run with add_fixed_geometry_spinup
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                ref_dmdtda=ref_dmdtda + delta_ref_dmdtda,
                err_ref_dmdtda=err_ref_dmdtda + delta_err_ref_dmdtda,
                ignore_errors=True,
                run_function=dynamic_melt_f_run_with_dynamic_spinup,
                kwargs_run_function={'minimise_for': minimise_for,
                                     'add_fixed_geometry_spinup': True,
                                     'precision_percent': precision_percent,
                                     'precision_absolute': precision_absolute,
                                     'do_inversion': do_inversion},
                fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
                kwargs_fallback_function={'minimise_for': minimise_for,
                                          'add_fixed_geometry_spinup': True,
                                          'precision_percent': precision_percent,
                                          'precision_absolute': precision_absolute,
                                          'do_inversion': do_inversion},
                output_filesuffix='_dyn_melt_f_calib_add_fixed_spinup',
                ys=1979, ye=ye)
            with xr.open_dataset(
                    gdir.get_filepath(
                        'model_diagnostics',
                        filesuffix='_dyn_melt_f_calib_add_fixed_spinup')) as ds:
                run_with_fixed_spinup = ds.load()

            assert (run_no_limit.time.values[0] >
                    run_with_fixed_spinup.time.values[0])
            assert run_with_fixed_spinup.time.values[0] == 1979
            # also compare the difference of spinup_start_yr_max and
            # add_fixed_geometry_spinup
            assert (run_with_limit.time.values[0] ==
                    run_with_fixed_spinup.time.values[0])
            assert (run_with_fixed_spinup.is_fixed_geometry_spinup.sum() <
                    run_with_limit.is_fixed_geometry_spinup.sum())

    @pytest.mark.slow
    @pytest.mark.skipif(not has_shapely2, reason="requires shapely2")
    def test_run_dynamic_melt_f_calibration_without_dynamic_spinup(self):

        # use a prepro dir as the hef_gdir climate data only goes to 2003 and
        # for the geodetic data we need climate data up to 2020
        gdir = workflow.init_glacier_directories(
            ['RGI60-11.00897'],  # Hintereisferner
            from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')[0]

        # save original melt_f to be able to reset back to default for testing
        melt_f_orig = gdir.read_json('mb_calib')['melt_f']

        def reset_melt_f():
            mb_calib = gdir.read_json('mb_calib')
            mb_calib['melt_f'] = melt_f_orig
            gdir.write_json(mb_calib, 'mb_calib')

        # value we want to match after dynamic melt_f calibration
        ref_period = cfg.PARAMS['geodetic_mb_period']

        yr0_ref_dmdtda, yr1_ref_dmdtda = ref_period.split('_')
        yr0_ref_dmdtda = int(yr0_ref_dmdtda.split('-')[0])
        yr1_ref_dmdtda = int(yr1_ref_dmdtda.split('-')[0])

        df_ref_dmdtda = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        sel = df_ref_dmdtda.loc[df_ref_dmdtda['period'] == ref_period].iloc[0]
        ref_dmdtda = float(sel['dmdtda'])
        ref_dmdtda *= 1000  # kg m-2 yr-1
        err_ref_dmdtda = float(sel['err_dmdtda'])
        err_ref_dmdtda *= 1000  # kg m-2 yr-1

        # conduct the run
        melt_f_max = 1000 * 12 / 365
        ye = gdir.get_climate_info()['baseline_yr_1'] + 1
        yr_rgi = gdir.rgi_date
        run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run,
            fallback_function=dynamic_melt_f_run_fallback,
            output_filesuffix='_dyn_melt_f_calib',
            ys=1979, ye=ye)

        # check that we are matching all desired ref values
        ds = utils.compile_run_output(
            gdir, input_filesuffix='_dyn_melt_f_calib',
            path=False)
        dmdtda_mdl = ((ds.volume.loc[yr1_ref_dmdtda].values -
                       ds.volume.loc[yr0_ref_dmdtda].values) /
                      gdir.rgi_area_m2 /
                      (yr1_ref_dmdtda - yr0_ref_dmdtda) *
                      cfg.PARAMS['ice_density'])
        assert np.isclose(dmdtda_mdl, ref_dmdtda,
                          rtol=np.abs(err_ref_dmdtda / ref_dmdtda))
        assert gdir.get_diagnostics()['used_spinup_option'] == \
               'dynamic melt_f calibration (full success)'

        # test err_dmdtda_scaling_factor
        err_dmdtda_scaling_factor = 0.01
        run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            err_dmdtda_scaling_factor=err_dmdtda_scaling_factor,
            run_function=dynamic_melt_f_run,
            fallback_function=dynamic_melt_f_run_fallback,
            output_filesuffix='_dyn_melt_f_calib_err_scaling',
            ys=1979, ye=ye)
        # check that we are matching all desired ref values
        ds = utils.compile_run_output(
            gdir, input_filesuffix='_dyn_melt_f_calib_err_scaling',
            path=False)

        dmdtda_mdl_scale = ((ds.volume.loc[yr1_ref_dmdtda].values -
                             ds.volume.loc[yr0_ref_dmdtda].values) /
                            gdir.rgi_area_m2 /
                            (yr1_ref_dmdtda - yr0_ref_dmdtda) *
                            cfg.PARAMS['ice_density'])
        assert np.isclose(dmdtda_mdl_scale, ref_dmdtda,
                          rtol=np.abs(err_ref_dmdtda *
                                      err_dmdtda_scaling_factor / ref_dmdtda))
        # check that calibration without scaling factor is outside adapted
        # uncertainty (use previous result without err scaling factor), this
        # tests if the scaling factor has any effect
        assert not np.isclose(dmdtda_mdl, ref_dmdtda,
                              rtol=np.abs(err_ref_dmdtda *
                                          err_dmdtda_scaling_factor /
                                          ref_dmdtda))
        assert gdir.get_diagnostics()['used_spinup_option'] == \
               'dynamic melt_f calibration (full success)'

        # tests for user provided dmdtda
        reset_melt_f()

        delta_ref_dmdtda = 100
        delta_err_ref_dmdtda = -50
        run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            ref_dmdtda=ref_dmdtda + delta_ref_dmdtda,
            err_ref_dmdtda=err_ref_dmdtda + delta_err_ref_dmdtda,
            run_function=dynamic_melt_f_run,
            fallback_function=dynamic_melt_f_run_fallback,
            output_filesuffix='_dyn_melt_f_calib_user_dmdtda',
            ys=1979, ye=ye)
        ds = utils.compile_run_output(
            gdir, input_filesuffix='_dyn_melt_f_calib_user_dmdtda',
            path=False)
        dmdtda_mdl = ((ds.volume.loc[yr1_ref_dmdtda].values -
                       ds.volume.loc[yr0_ref_dmdtda].values) /
                      gdir.rgi_area_m2 /
                      (yr1_ref_dmdtda - yr0_ref_dmdtda) *
                      cfg.PARAMS['ice_density'])
        assert np.isclose(dmdtda_mdl, ref_dmdtda + delta_ref_dmdtda,
                          rtol=np.abs((err_ref_dmdtda + delta_err_ref_dmdtda) /
                                      (ref_dmdtda + delta_ref_dmdtda)))

        # test that error is raised if ignore_error=False
        reset_melt_f()

        with pytest.raises(RuntimeError,
                           match='Dynamic melt_f calibration not successful.*'):
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                run_function=dynamic_melt_f_run,
                fallback_function=dynamic_melt_f_run,
                output_filesuffix='_dyn_melt_f_calib_error',
                ignore_errors=False,
                ref_dmdtda=ref_dmdtda, err_ref_dmdtda=0.000001,
                maxiter=2)
        # test that fallback function works as expected if ignore_error=True and
        # if the first guess can improve (but not enough)
        model_fallback = run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run,
            fallback_function=dynamic_melt_f_run_fallback,
            output_filesuffix='_dyn_melt_f_calib_spinup_inversion_error',
            ignore_errors=True,
            ref_dmdtda=ref_dmdtda, err_ref_dmdtda=0.000001,
            maxiter=2)
        assert isinstance(model_fallback, oggm.core.flowline.FluxBasedModel)
        assert gdir.get_diagnostics()['used_spinup_option'] == \
               'dynamic melt_f calibration (part success)'

        # test that fallback function works as expected if ignore_error=True and
        # if no successful run can be conducted
        model_fallback = run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run,
            kwargs_run_function={'cfl_number': 1e-8},  # force an error
            fallback_function=dynamic_melt_f_run_fallback,
            output_filesuffix='_dyn_melt_f_calib_error',
            ignore_errors=True,
            ref_dmdtda=ref_dmdtda, err_ref_dmdtda=0.000001,
            maxiter=2)
        assert isinstance(model_fallback, oggm.core.flowline.FluxBasedModel)
        assert gdir.get_diagnostics()['used_spinup_option'] == 'no spinup'

        # test that error is raised if user provided dmdtda is given without an
        # error and vice versa
        for use_ref_dmdtda, use_err_ref_dmdtda in zip([ref_dmdtda, None],
                                                      [None, err_ref_dmdtda]):
            with pytest.raises(RuntimeError,
                               match='If you provide a reference geodetic '
                                     'mass balance .*'):
                run_dynamic_melt_f_calibration(
                    gdir, melt_f_max=melt_f_max,
                    run_function=dynamic_melt_f_run,
                    fallback_function=dynamic_melt_f_run_fallback,
                    ref_dmdtda=use_ref_dmdtda,
                    err_ref_dmdtda=use_err_ref_dmdtda)

        # test error is raised if given years outside of geodetic period
        with pytest.raises(RuntimeError,
                           match='The provided ye is smaller than the end year'
                                 ' of the given *'):
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                run_function=dynamic_melt_f_run,
                fallback_function=dynamic_melt_f_run_fallback,
                ye=yr1_ref_dmdtda - 1)

        with pytest.raises(RuntimeError,
                           match='The provided ys is larger than the start year'
                                 ' of the given *'):
            run_dynamic_melt_f_calibration(
                gdir, melt_f_max=melt_f_max,
                run_function=dynamic_melt_f_run,
                fallback_function=dynamic_melt_f_run_fallback,
                ys=yr0_ref_dmdtda + 1)

        # test initialisation from an previous glacier geometry
        cfg.PARAMS['store_model_geometry'] = True
        workflow.execute_entity_task(tasks.run_from_climate_data, [gdir],
                                     ys=yr_rgi, ye=yr_rgi + 1,
                                     output_filesuffix='_one_yr')
        run_dynamic_melt_f_calibration(
            gdir, melt_f_max=melt_f_max,
            init_model_filesuffix='_one_yr',
            run_function=dynamic_melt_f_run,
            fallback_function=dynamic_melt_f_run_fallback)


@pytest.mark.usefixtures('with_class_wd')
class TestHydro:

    @pytest.mark.slow
    @pytest.mark.parametrize('store_monthly_hydro', [False, True], ids=['annual', 'monthly'])
    def test_hydro_out_from_no_glacier(self, hef_gdir, inversion_params, store_monthly_hydro):

        gdir = hef_gdir

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        init_present_time_glacier(gdir)
        tasks.run_with_hydro(gdir, run_task=tasks.run_constant_climate,
                             store_monthly_hydro=store_monthly_hydro,
                             y0=1985, nyears=50, zero_initial_glacier=True,
                             output_filesuffix='_const')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_const')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Sanity checks
        # Tot prcp here is constant (constant climate)
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])
        # assert_allclose(odf['tot_prcp'], odf['tot_prcp'].iloc[0])
        # this test fails because snowfall_on_glacier changes, as the
        # formerly negative melt_on_glacier was added to snowfall_on_glacier
        # let's check instead if the remainning years are close:
        assert_allclose(odf['tot_prcp'].iloc[1:], odf['tot_prcp'].iloc[1])

        # So is domain area
        odf['dom_area'] = odf['on_area'] + odf['off_area']
        assert_allclose(odf['dom_area'], odf['dom_area'].iloc[0])

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        odf['reconstructed_vol'] = odf['model_mb'].cumsum() / cfg.PARAMS['ice_density']
        assert_allclose(odf['volume_m3'].iloc[1:], odf['reconstructed_vol'].iloc[:-1])

        # Mass-conservation
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in - mass_out - mass_in_snow - mass_in_glacier,
                        0, atol=1e-2)  # 0.01 kg is OK as numerical error

        # At the very first timesep there is no glacier so the
        # melt_on_glacier var is negative - this is a numerical artifact
        # from the residual
        # assert_allclose(odf['melt_on_glacier'].iloc[0],
        #                - odf['residual_mb'].iloc[0])
        # we changed that: if negative, the absolute value should go to snowfall_on_glacier
        # let's check that (this happens also at other times than the first step,
        # but only in the first step it always happens)
        assert_allclose(odf['snowfall_on_glacier'].iloc[0],
                        odf['residual_mb'].iloc[0])
        # check if melt on glacier is always above or equal zero
        assert np.all(odf['melt_on_glacier'] >= 0)
        # Now with zero ref area
        tasks.run_with_hydro(gdir, run_task=tasks.run_constant_climate,
                             store_monthly_hydro=store_monthly_hydro,
                             y0=1985, nyears=50, zero_initial_glacier=True,
                             ref_area_from_y0=True, output_filesuffix='_const_y0')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_const_y0')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Sanity checks
        # Tot prcp here is not constant but grows always since glacier area grows
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])
        assert np.all(odf['tot_prcp'].iloc[1:].values -
                      odf['tot_prcp'].iloc[:-1].values > 0)

        # So is domain area
        odf['dom_area'] = odf['on_area'] + odf['off_area']
        assert np.all(odf['dom_area'].iloc[1:].values -
                      odf['dom_area'].iloc[:-1].values > 0)

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        odf['reconstructed_vol'] = odf['model_mb'].cumsum() / cfg.PARAMS['ice_density']
        assert_allclose(odf['volume_m3'].iloc[1:], odf['reconstructed_vol'].iloc[:-1])

        # Mass-conservation
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in - mass_out - mass_in_snow - mass_in_glacier,
                        0, atol=1e-2)  # 0.01 kg is OK as numerical error

        # At the very first timesep there is no glacier so the
        # melt_on_glacier var is negative - this is a numerical artifact
        # from the residual
        # assert_allclose(odf['melt_on_glacier'].iloc[0],
        #                - odf['residual_mb'].iloc[0])
        # we changed, that, if negative, the absolute value should go to snowfall_on_glacier
        # let's check that (this happens also at other times than the first step,
        # but only in the first step it always happens)
        assert_allclose(odf['snowfall_on_glacier'].iloc[0],
                        odf['residual_mb'].iloc[0])

        # check if melt on glacier is always above or equal zero
        assert np.all(odf['melt_on_glacier'] >= 0)

    @pytest.mark.slow
    @pytest.mark.parametrize('store_monthly_hydro', [False, True], ids=['annual', 'monthly'])
    def test_hydro_out_commitment(self, hef_gdir, inversion_params, store_monthly_hydro):

        gdir = hef_gdir

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        init_present_time_glacier(gdir)
        tasks.run_with_hydro(gdir, run_task=tasks.run_constant_climate,
                             store_monthly_hydro=store_monthly_hydro,
                             nyears=100, y0=2002 - 5, halfsize=5,
                             output_filesuffix='_const')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_const')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Sanity checks
        # Tot prcp here is constant (constant climate)
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])
        # this test failed in test above, maybe just a coincidence
        # that it works here
        assert_allclose(odf['tot_prcp'], odf['tot_prcp'].iloc[0])


        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                             odf['volume_m3'].iloc[0])
        assert_allclose(odf['volume_m3'].iloc[1:], reconstructed_vol.iloc[:-1])

        # Mass-conservation
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error

        # Qualitative assessments
        assert odf['melt_on_glacier'].iloc[-1] < odf['melt_on_glacier'].iloc[0] * 0.7
        assert odf['liq_prcp_off_glacier'].iloc[-1] > odf['liq_prcp_on_glacier'].iloc[-1]
        assert odf['liq_prcp_off_glacier'].iloc[0] < odf['liq_prcp_on_glacier'].iloc[0]

        # Residual MB should not be crazy large
        frac = odf['residual_mb'] / odf['melt_on_glacier']
        assert_allclose(frac, 0, atol=0.025)

        # check if melt on glacier is always above or equal zero
        assert np.all(odf['melt_on_glacier'] >= 0)

    @pytest.mark.slow
    @pytest.mark.parametrize('store_monthly_hydro', [True, False], ids=['monthly', 'annual'])
    def test_hydro_out_past_climate(self, hef_gdir, inversion_params, store_monthly_hydro):

        gdir = hef_gdir
        gdir.rgi_date = 1990

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        init_present_time_glacier(gdir)
        tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data,
                             store_monthly_hydro=store_monthly_hydro,
                             min_ys=1980, output_filesuffix='_hist')

        tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data,
                             store_monthly_hydro=store_monthly_hydro,
                             fixed_geometry_spinup_yr=1980,
                             output_filesuffix='_spin')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_hist')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_spin')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf_spin = ds[sel_vars].to_dataframe().iloc[:-1]

        # Sanity checks
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                             odf['volume_m3'].iloc[0])
        assert_allclose(odf['volume_m3'].iloc[1:], reconstructed_vol.iloc[:-1])

        # Mass-conservation
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error

        # Mass-conservation spinup
        odf_spin['tot_prcp'] = (odf_spin['liq_prcp_off_glacier'] +
                                odf_spin['liq_prcp_on_glacier'] +
                                odf_spin['snowfall_off_glacier'] +
                                odf_spin['snowfall_on_glacier'])

        odf_spin['runoff'] = (odf_spin['melt_on_glacier'] +
                              odf_spin['melt_off_glacier'] +
                              odf_spin['liq_prcp_on_glacier'] +
                              odf_spin['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf_spin['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf_spin['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf_spin['snow_bucket'].iloc[-1]
        mass_in = odf_spin['tot_prcp'].iloc[:-1].sum()
        mass_out = odf_spin['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error
        # Other checks
        assert_allclose(odf_spin['is_fixed_geometry_spinup'].loc[:1990], 1)

        # Residual MB should not be crazy large
        frac = odf['residual_mb'] / odf['melt_on_glacier']
        assert_allclose(frac, 0, atol=0.05)
        # In the spinup run the residual is zero for the spinup part
        assert_allclose(odf_spin['residual_mb'].loc[:1990], 0)

        # Also check output stuff
        nds = utils.compile_run_output([gdir], input_filesuffix='_hist')
        assert nds.residual_mb.attrs['unit'] == 'kg yr-1'
        assert_allclose(nds['snowfall_on_glacier'].squeeze()[:-1],
                        odf['snowfall_on_glacier'])
        if 'month_2d' in nds:
            # 3d vars
            sel_vars = [v for v in nds.variables if 'month_2d' in nds[v].dims]
            odf_ma = nds[sel_vars].mean(dim=('time', 'rgi_id')).to_dataframe()
            odf_ma.columns = [c.replace('_monthly', '') for c in odf_ma.columns]
            # Runoff peak should follow a temperature curve
            assert_allclose(odf_ma['melt_on_glacier'].idxmax(), 8)

        # check if melt on glacier is always above or equal zero
        assert np.all(odf['melt_on_glacier'] >= 0)

    @pytest.mark.slow
    def test_hydro_ref_area(self, hef_gdir, inversion_params):

        gdir = hef_gdir
        gdir.rgi_date = 1990

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        init_present_time_glacier(gdir)
        tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data,
                             store_monthly_hydro=False,
                             fixed_geometry_spinup_yr=1980,
                             ref_area_from_y0=True,
                             ye=1999,
                             output_filesuffix='_hist')
        tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data,
                             store_monthly_hydro=False,
                             init_model_filesuffix='_hist',
                             ref_geometry_filesuffix='_hist',
                             ref_area_from_y0=True,
                             output_filesuffix='_run')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_hist')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf_hist = ds[sel_vars].to_dataframe().iloc[:-1]

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_run')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf_run = ds[sel_vars].to_dataframe().iloc[:-1]

        assert odf_hist.index[0] == 1980
        assert odf_hist.index[-1] == 1998  # this matches because last year is removed above
        assert odf_run.index[0] == 1999

        odf = pd.concat([odf_hist, odf_run])
        # Domain area is constant and equal to the first year
        # Except at the beginning of the simulation where the glacier
        # advances a little
        odf['dom_area'] = odf['on_area'] + odf['off_area']
        assert_allclose(odf['dom_area'], odf['dom_area'].iloc[0], rtol=3e-3)
        assert_allclose(odf['dom_area'].iloc[0], odf['dom_area'].iloc[-1])

    @pytest.mark.slow
    def test_hydro_dynamical_spinup(self, hef_gdir, inversion_params):

        gdir = hef_gdir
        gdir.rgi_date = 1990

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        # need to add area min h if I want to merge two runs for compatibility
        ovars = cfg.PARAMS['store_diagnostic_variables']
        ovars += ['area_min_h']

        init_present_time_glacier(gdir)
        tasks.run_with_hydro(gdir, run_task=tasks.run_dynamic_spinup,
                             store_monthly_hydro=True,
                             ref_area_from_y0=True,
                             output_filesuffix='_spinup')
        tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data,
                             store_monthly_hydro=True,
                             init_model_filesuffix='_spinup',
                             ref_geometry_filesuffix='_spinup',
                             ref_area_from_y0=True,
                             output_filesuffix='_run')

        utils.merge_consecutive_run_outputs(gdir,
                                            input_filesuffix_1='_spinup',
                                            input_filesuffix_2='_run',
                                            output_filesuffix='_merged')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_merged')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Domain area is constant and equal to the first year
        odf['dom_area'] = odf['on_area'] + odf['off_area']

        # Sanity checks
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                             odf['volume_m3'].iloc[0])
        assert_allclose(odf['volume_m3'].iloc[1:], reconstructed_vol.iloc[:-1])

        # Ensure mass-conservation even at junction?
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error

        # Residual MB should not be crazy large
        frac = odf['residual_mb'] / odf['melt_on_glacier']
        assert_allclose(frac, 0, atol=0.05)

        # check if melt on glacier is always above or equal zero
        assert np.all(odf['melt_on_glacier'] >= 0)

    @pytest.mark.parametrize('do_inversion', [True, False])
    @pytest.mark.slow
    @pytest.mark.skipif(not has_shapely2, reason="requires shapely2")
    def test_hydro_dynamic_melt_f_with_dynamic_spinup(self, inversion_params,
                                                      do_inversion):

        gdir = workflow.init_glacier_directories(
            ['RGI60-11.00897'],  # Hintereisferner
            from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')[0]

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        melt_f_max = 1000 * 12 / 365
        tasks.run_with_hydro(
            gdir, run_task=tasks.run_dynamic_melt_f_calibration,
            store_monthly_hydro=True, ref_area_from_y0=True,
            output_filesuffix='_dyn_melt_f_calib', melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run_with_dynamic_spinup,
            kwargs_run_function={'do_inversion': do_inversion},
            fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
            kwargs_fallback_function={'do_inversion': do_inversion},)

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_dyn_melt_f_calib')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Domain area is constant and equal to the first year
        odf['dom_area'] = odf['on_area'] + odf['off_area']

        # Sanity checks
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                             odf['volume_m3'].iloc[0])
        assert_allclose(odf['volume_m3'].iloc[1:], reconstructed_vol.iloc[:-1])

        # Ensure mass-conservation even at junction?
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error

        # Residual MB should not be crazy large
        frac = odf['residual_mb'] / odf['melt_on_glacier']
        assert_allclose(frac, 0, atol=0.05)

    @pytest.mark.slow
    @pytest.mark.skipif(not has_shapely2, reason="requires shapely2")
    def test_hydro_dynamic_melt_f_without_dynamic_spinup(self, inversion_params):

        gdir = workflow.init_glacier_directories(
            ['RGI60-11.00897'],  # Hintereisferner
            from_prepro_level=3, prepro_border=160,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')[0]

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        melt_f_max = 1000 * 12 / 365
        tasks.run_with_hydro(
            gdir, run_task=tasks.run_dynamic_melt_f_calibration,
            store_monthly_hydro=True, ref_area_from_y0=True,
            output_filesuffix='_dyn_melt_f_calib', melt_f_max=melt_f_max,
            run_function=dynamic_melt_f_run,
            fallback_function=dynamic_melt_f_run_fallback)

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_dyn_melt_f_calib')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Domain area is constant and equal to the first year
        odf['dom_area'] = odf['on_area'] + odf['off_area']

        # Sanity checks
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                             odf['volume_m3'].iloc[0])
        assert_allclose(odf['volume_m3'].iloc[1:], reconstructed_vol.iloc[:-1])

        # Ensure mass-conservation even at junction?
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error

        # Residual MB should not be crazy large
        frac = odf['residual_mb'] / odf['melt_on_glacier']
        assert_allclose(frac, 0, atol=0.05)

    @pytest.mark.slow
    @pytest.mark.parametrize('store_monthly_hydro', [False, True], ids=['annual', 'monthly'])
    def test_hydro_out_random(self, hef_gdir, inversion_params, store_monthly_hydro):

        gdir = hef_gdir

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        init_present_time_glacier(gdir)
        tasks.run_with_hydro(gdir, run_task=tasks.run_random_climate,
                             store_monthly_hydro=store_monthly_hydro,
                             seed=0, nyears=100, y0=2002 - 5, halfsize=5,
                             output_filesuffix='_rand')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_rand')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Sanity checks
        # Tot prcp here is constant (constant climate)
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                             odf['volume_m3'].iloc[0])
        assert_allclose(odf['volume_m3'].iloc[1:], reconstructed_vol.iloc[:-1])

        # Mass-conservation
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error

        # Qualitative assessments
        assert odf['melt_on_glacier'].iloc[-1] < odf['melt_on_glacier'].iloc[0] * 0.7
        assert odf['liq_prcp_off_glacier'].iloc[0] < odf['liq_prcp_on_glacier'].iloc[0]

        # Residual MB should not be crazy large
        frac = odf['residual_mb'] / odf['melt_on_glacier']
        assert_allclose(frac, 0, atol=0.044)  # annual can be large (prob)

        # check if melt on glacier is always above or equal zero
        assert np.all(odf['melt_on_glacier'] >= 0)

    #@pytest.mark.slow
    @pytest.mark.parametrize('mb_type', ['random', 'const', 'hist'])
    @pytest.mark.parametrize('mb_bias', [500, -500, 0])
    def test_hydro_monhly_vs_annual(self, hef_gdir, inversion_params,
                                    mb_type, mb_bias):

        gdir = hef_gdir
        gdir.rgi_date = 1990

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        # Needed for this to run
        cfg.PARAMS['store_model_geometry'] = True

        init_present_time_glacier(gdir)

        if mb_type == 'random':
            tasks.run_with_hydro(gdir, run_task=tasks.run_random_climate,
                                 bias=mb_bias,
                                 store_monthly_hydro=False,
                                 seed=0, nyears=20, y0=2002 - 5, halfsize=5,
                                 output_filesuffix='_annual')
            tasks.run_with_hydro(gdir, run_task=tasks.run_random_climate,
                                 bias=mb_bias,
                                 store_monthly_hydro=True,
                                 seed=0, nyears=20, y0=2002 - 5, halfsize=5,
                                 output_filesuffix='_monthly')
        elif mb_type == 'const':
            tasks.run_with_hydro(gdir, run_task=tasks.run_constant_climate,
                                 bias=mb_bias,
                                 store_monthly_hydro=False,
                                 nyears=20, y0=2002 - 5, halfsize=5,
                                 output_filesuffix='_annual')
            tasks.run_with_hydro(gdir, run_task=tasks.run_constant_climate,
                                 bias=mb_bias,
                                 store_monthly_hydro=True,
                                 nyears=20, y0=2002 - 5, halfsize=5,
                                 output_filesuffix='_monthly')
        elif mb_type == 'hist':
            tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data,
                                 bias=mb_bias,
                                 store_monthly_hydro=False,
                                 min_ys=1980, output_filesuffix='_annual')
            tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data,
                                 bias=mb_bias,
                                 store_monthly_hydro=True,
                                 min_ys=1980, output_filesuffix='_monthly')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_annual')) as ds:
            assert_allclose(ds.calendar_month.data, 1)
            assert_allclose(ds.hydro_month.data, 4)
            odf_a = ds.to_dataframe()

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_monthly')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf_m = ds[sel_vars].to_dataframe()
            sel_vars = [v for v in ds.variables if 'month_2d' in ds[v].dims]
            odf_ma = ds[sel_vars].mean(dim='time').to_dataframe()
            odf_ma.columns = [c.replace('_monthly', '') for c in odf_ma.columns]

        # Check that yearly equals monthly
        np.testing.assert_array_equal(odf_a.columns, odf_m.columns)
        for c in odf_a.columns:
            if c == 'melt_off_glacier':
                # This one is quite different - reason is snow cover buildup
                # and melt. Monthly is better than annual
                assert_allclose(odf_a[c].iloc[-5:].mean(),
                                odf_m[c].iloc[-5:].mean(),
                                rtol=0.15)
                continue
            if c in ['snow_bucket']:
                continue
            assert_allclose(odf_a[c], odf_m[c], rtol=1e-5)

        # Check monthly stuff
        odf_ma['tot_prcp'] = (odf_ma['liq_prcp_off_glacier'] +
                              odf_ma['liq_prcp_on_glacier'] +
                              odf_ma['snowfall_off_glacier'] +
                              odf_ma['snowfall_on_glacier'])

        odf_ma['runoff'] = (odf_ma['melt_on_glacier'] +
                            odf_ma['melt_off_glacier'] +
                            odf_ma['liq_prcp_on_glacier'] +
                            odf_ma['liq_prcp_off_glacier'])

        # Regardless of MB bias the melt in winter months should be close to zero
        assert_allclose(odf_ma['melt_on_glacier'].loc[[12, 1, 2]], 0, atol=1e4)

        # Residual MB should not be crazy large
        frac = odf_ma['residual_mb'] / odf_ma['melt_on_glacier']
        frac[odf_ma['melt_on_glacier'] < 1e-4] = 0
        assert_allclose(frac.loc[~frac.isnull()], 0, atol=0.01)

        # Runoff peak should follow a temperature curve
        assert_allclose(odf_ma['runoff'].idxmax(), 8)


class TestMassRedis:

    def test_hef_retreat(self, class_case_dir):

        import geopandas as gpd

        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['border'] = 40
        cfg.PARAMS['baseline_climate'] = ''
        cfg.PARAMS['use_multiprocessing'] = False
        cfg.PARAMS['min_ice_thick_for_length'] = 5
        cfg.PARAMS['use_winter_prcp_fac'] = False
        cfg.PARAMS['use_temp_bias_from_file'] = False
        cfg.PARAMS['prcp_fac'] = 2.5

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=class_case_dir)
        tasks.define_glacier_region(gdir)
        tasks.simple_glacier_masks(gdir)
        tasks.elevation_band_flowline(gdir)
        tasks.fixed_dx_elevation_band_flowline(gdir)
        tasks.compute_downstream_line(gdir)
        tasks.compute_downstream_bedshape(gdir)
        tasks.process_custom_climate_data(gdir)

        mbdf = gdir.get_ref_mb_data()
        cfg.PARAMS['melt_f_max'] = 600 * 12 / 365
        ref_mb = mbdf.ANNUAL_BALANCE.mean()
        tasks.mb_calibration_from_scalar_mb(gdir, ref_mb=ref_mb,
                                            ref_period='1953-01-01_2003-01-01')
        tasks.apparent_mb_from_any_mb(gdir, mb_years=[1953, 2003])
        workflow.calibrate_inversion_from_consensus([gdir])
        tasks.init_present_time_glacier(gdir)

        seed = 0

        odf_l = pd.DataFrame()
        odf_v = pd.DataFrame()
        biases = [-0.6, -0.3, 0]
        for bias in biases:
            tasks.run_random_climate(gdir, nyears=500, y0=1990, halfsize=10,
                                     temperature_bias=bias,
                                     seed=seed,
                                     output_filesuffix='_fl')
            with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                                   filesuffix='_fl')) as ds:
                df_fl = ds.to_dataframe()

            odf_v[f'fl_t{bias}'] = df_fl['volume_m3']
            odf_l[f'fl_t{bias}'] = df_fl['length_m']

            for advance_method in [0, 1, 2]:
                MethodCurveModel = partial(MassRedistributionCurveModel,
                                           advance_method=advance_method)
                tasks.run_random_climate(gdir, nyears=500, y0=1990, halfsize=10,
                                         temperature_bias=bias,
                                         seed=seed,
                                         evolution_model=MethodCurveModel,
                                         output_filesuffix='_mr')
                with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                                       filesuffix='_mr')) as ds:
                    df_mr = ds.to_dataframe()

                odf_v[f'mr_m{advance_method}_t{bias}'] = df_mr['volume_m3']
                odf_l[f'mr_m{advance_method}_t{bias}'] = df_mr['length_m']

        # Test that during the retreat phase all is very close
        # (incredibly close actually)
        for bias in biases:
            cc = [c for c in odf_v if f'_t{bias}' in c]
            sdf = odf_v[cc].loc[:100]
            for c in sdf.columns[1:]:
                assert_allclose(sdf[sdf.columns[0]], sdf[c], rtol=0.07)

        if do_plot:
            for advance_method in [0, 1, 2]:
                cc = [c for c in odf_v if f'_m{advance_method}' in c or 'fl' in c]
                v = odf_v[cc]
                l = odf_l[cc]

                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                v.plot(ax=ax1,
                       color=['C0'] * 2 + ['C1'] * 2 + ['C3'] * 2,
                       style=['-', ':'] * 3)
                ax1.set_title(f'Volume, advance_method={advance_method}')
                l.plot(ax=ax2,
                       color=['C0'] * 2 + ['C1'] * 2 + ['C3'] * 2,
                       style=['-', ':'] * 3)
                ax2.set_title(f'Length, advance_method={advance_method}')
                plt.show()


@pytest.fixture(scope='class')
def merged_hef_cfg(class_case_dir):

    # Init
    cfg.initialize()
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'
    # should this be resetting working_dir at teardown?
    cfg.PATHS['working_dir'] = class_case_dir
    cfg.PARAMS['border'] = 100
    cfg.PARAMS['prcp_fac'] = 1.75
    cfg.PARAMS['temp_melt'] = -1.75
    cfg.PARAMS['use_winter_prcp_fac'] = False
    cfg.PARAMS['use_temp_bias_from_file'] = False


@pytest.mark.usefixtures('merged_hef_cfg')
class TestMergedHEF:

    @pytest.mark.slow
    @pytest.mark.skip
    def test_merged_simulation(self):
        import geopandas as gpd

        hef_file = utils.get_demo_file('rgi_oetztal.shp')
        rgidf = gpd.read_file(hef_file)

        # Get HEF, Vernagt1/2 and Gepatschferner
        glcdf = rgidf.loc[(rgidf.RGIId == 'RGI50-11.00897') |
                          (rgidf.RGIId == 'RGI50-11.00719_d01') |
                          (rgidf.RGIId == 'RGI50-11.00779') |
                          (rgidf.RGIId == 'RGI50-11.00746')].copy()

        cfg.PARAMS['use_multiprocessing'] = True

        gdirs = workflow.init_glacier_directories(glcdf)
        workflow.gis_prepro_tasks(gdirs)
        # # Process climate data
        # execute_entity_task(tasks.process_climate_data, gdirs)
        # # mass balance and the apparent mass balance
        # execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
        #                     override_missing=override_missing,
        #                     overwrite_gdir=overwrite_gdir)
        # execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs)
        # workflow.climate_tasks(gdirs, override_missing=-200,
        #                        ref_mb_years=(1980, 2000))
        workflow.inversion_tasks(gdirs)
        workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

        # merge, but with 0 buffer, should not do anything
        merge0 = workflow.merge_glacier_tasks(gdirs,
                                              main_rgi_id='RGI50-11.00897',
                                              glcdf=glcdf, buffer=0)
        assert 'RGI50-11.00897' == np.unique([fl.rgi_id for fl in
                                              merge0.read_pickle(
                                                  'model_flowlines')])[0]

        # merge, but with 50 buffer. overlapping glaciers should be excluded
        merge1 = workflow.merge_glacier_tasks(gdirs,
                                              main_rgi_id='RGI50-11.00897',
                                              glcdf=glcdf, buffer=50)
        assert 'RGI50-11.00719_d01' in [fl.rgi_id for fl in
                                        merge1.read_pickle('model_flowlines')]
        assert 'RGI50-11.00779' not in [fl.rgi_id for fl in
                                        merge1.read_pickle('model_flowlines')]

        # merge HEF and Vernagt, include Gepatsch but it should not be merged
        gdir_merged = workflow.merge_glacier_tasks(gdirs,
                                                   main_rgi_id='RGI50-11.00897',
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

        # run parameters
        years = 200  # arbitrary
        tbias = -1.0  # arbitrary

        # run HEF and the two Vernagts as entities
        gdirs_entity = [gd for gd in gdirs if gd.rgi_id != 'RGI50-11.00746']
        workflow.execute_entity_task(tasks.run_constant_climate,
                                     gdirs_entity,
                                     y0=1985,
                                     nyears=years,
                                     output_filesuffix='_entity',
                                     temperature_bias=tbias)

        ds_entity = utils.compile_run_output(gdirs_entity,
                                             path=False,
                                             input_filesuffix='_entity')

        # and run the merged glacier
        workflow.execute_entity_task(tasks.run_constant_climate,
                                     gdir_merged, output_filesuffix='_merged',
                                     y0=1985,
                                     nyears=years,
                                     temperature_bias=tbias)

        ds_merged = utils.compile_run_output(gdir_merged,
                                             path=False,
                                             input_filesuffix='_merged')

        # areas should be quite similar after 10yrs
        assert_allclose(ds_entity.area.isel(time=10).sum(),
                        ds_merged.area.isel(time=10),
                        rtol=1e-3)

        # After 100yrs, merged one should be smaller as Vernagt1 is slightly
        # flowing into Vernagt2
        assert (ds_entity.area.isel(time=100).sum() >
                ds_merged.area.isel(time=100))

        # Merged glacier should have a larger area after 200yrs from advancing
        assert (ds_entity.area.isel(time=200).sum() <
                ds_merged.area.isel(time=200))


class TestSemiImplicitModel:

    @pytest.mark.slow
    def test_equilibrium(self, hef_elev_gdir, inversion_params):
        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        cfg.PARAMS['downstream_line_shape'] = 'trapezoidal'
        init_present_time_glacier(hef_elev_gdir)
        cfg.PARAMS['min_ice_thick_for_length'] = 1

        # year 1930 is used with equilibrium climate period in mind (old t*)
        mb_mod = massbalance.ConstantMassBalance(hef_elev_gdir, y0=1930)

        fls = hef_elev_gdir.read_pickle('model_flowlines')
        model = SemiImplicitModel(fls, mb_model=mb_mod, y0=0,
                                  fs=inversion_params['inversion_fs'],
                                  glen_a=inversion_params['inversion_glen_a'],
                                  mb_elev_feedback='never')

        ref_vol = model.volume_km3
        ref_area = model.area_km2
        ref_len = model.fls[-1].length_m

        np.testing.assert_allclose(ref_area, hef_elev_gdir.rgi_area_km2)

        model.run_until_equilibrium(rate=1e-5)
        assert model.yr >= 50
        after_vol = model.volume_km3
        after_area = model.area_km2
        after_len = model.fls[-1].length_m

        # this tests assume we are currently close to equilibrium with mb_mod
        np.testing.assert_allclose(ref_vol, after_vol, rtol=0.4)
        np.testing.assert_allclose(ref_area, after_area, rtol=0.03)
        np.testing.assert_allclose(ref_len, after_len, atol=300.01)

        # compare to FluxBasedModel
        model_flux = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                    fs=inversion_params['inversion_fs'],
                                    glen_a=inversion_params['inversion_glen_a'],
                                    mb_elev_feedback='never')
        model_flux.run_until_equilibrium(rate=1e-5)

        np.testing.assert_allclose(model_flux.volume_km3, after_vol, rtol=7e-4)
        np.testing.assert_allclose(model_flux.area_km2, after_area, rtol=6e-5)
        np.testing.assert_allclose(model_flux.fls[-1].length_m, after_len)

        # now glacier wide with MultipleFlowlineMassBalance
        cl = massbalance.ConstantMassBalance
        mb_mod = massbalance.MultipleFlowlineMassBalance(hef_elev_gdir,
                                                         mb_model_class=cl,
                                                         y0=1930)

        model = SemiImplicitModel(fls, mb_model=mb_mod, y0=0,
                                  fs=inversion_params['inversion_fs'],
                                  glen_a=inversion_params['inversion_glen_a'],
                                  mb_elev_feedback='never')
        model.run_until_equilibrium(rate=1e-5)

        model_flux = FluxBasedModel(fls, mb_model=mb_mod, y0=0.,
                                    fs=inversion_params['inversion_fs'],
                                    glen_a=inversion_params['inversion_glen_a'],
                                    mb_elev_feedback='never')
        model_flux.run_until_equilibrium(rate=1e-5)

        assert model.yr >= 50

        after_vol = model.volume_km3
        after_area = model.area_km2
        after_len = model.fls[-1].length_m

        np.testing.assert_allclose(model_flux.volume_km3, after_vol, rtol=8e-4)
        np.testing.assert_allclose(model_flux.area_km2, after_area, rtol=2e-5)
        np.testing.assert_allclose(model_flux.fls[-1].length_m, after_len,
                                   atol=100.1)

    @pytest.mark.slow
    def test_random(self, hef_elev_gdir, inversion_params):
        cfg.PARAMS['store_model_geometry'] = True
        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        cfg.PARAMS['downstream_line_shape'] = 'trapezoidal'

        init_present_time_glacier(hef_elev_gdir)
        run_random_climate(hef_elev_gdir, nyears=100, seed=6, y0=1930,
                           fs=inversion_params['inversion_fs'],
                           glen_a=inversion_params['inversion_glen_a'],
                           bias=0, output_filesuffix='_rdn',
                           evolution_model=SemiImplicitModel)
        run_constant_climate(hef_elev_gdir, nyears=100, y0=1930,
                             fs=inversion_params['inversion_fs'],
                             glen_a=inversion_params['inversion_glen_a'],
                             bias=0, output_filesuffix='_ct',
                             evolution_model=SemiImplicitModel)

        paths = [hef_elev_gdir.get_filepath('model_geometry', filesuffix='_rdn'),
                 hef_elev_gdir.get_filepath('model_geometry', filesuffix='_ct'),
                 ]

        for path in paths:
            model = FileModel(path)
            vol = model.volume_km3_ts()
            area = model.area_km2_ts()
            np.testing.assert_allclose(vol.iloc[0], np.mean(vol),
                                       rtol=0.26)
            np.testing.assert_allclose(area.iloc[0], np.mean(area),
                                       rtol=0.1)

    @pytest.mark.slow
    def test_sliding_and_compare_to_fluxbased(self, hef_elev_gdir,
                                              inversion_params):
        cfg.PARAMS['store_model_geometry'] = True
        cfg.PARAMS['store_fl_diagnostics'] = True
        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        cfg.PARAMS['downstream_line_shape'] = 'trapezoidal'
        init_present_time_glacier(hef_elev_gdir)
        cfg.PARAMS['min_ice_thick_for_length'] = 1

        start_time_impl = time.time()
        run_random_climate(hef_elev_gdir, nyears=1000, seed=6, y0=1930,
                           fs=5.7e-20,
                           glen_a=inversion_params['inversion_glen_a'],
                           bias=0, output_filesuffix='_implicit_run',
                           evolution_model=SemiImplicitModel)
        impl_time_needed = time.time() - start_time_impl

        start_time_expl = time.time()
        run_random_climate(hef_elev_gdir, nyears=1000, seed=6, y0=1930,
                           fs=5.7e-20,
                           glen_a=inversion_params['inversion_glen_a'],
                           bias=0, output_filesuffix='_fluxbased_run',
                           evolution_model=FluxBasedModel)
        flux_time_needed = time.time() - start_time_expl

        fmod_impl = FileModel(
            hef_elev_gdir.get_filepath('model_geometry',
                                       filesuffix='_implicit_run'))
        fmod_flux = FileModel(
            hef_elev_gdir.get_filepath('model_geometry',
                                       filesuffix='_fluxbased_run'))

        # check that the two runs are close the whole time
        np.testing.assert_allclose(fmod_impl.volume_km3_ts(),
                                   fmod_flux.volume_km3_ts(),
                                   rtol=0.01)
        np.testing.assert_allclose(fmod_impl.area_km2_ts(),
                                   fmod_flux.area_km2_ts(),
                                   rtol=0.02)

        years = np.arange(0, 1001)
        if do_plot:
            plt.figure()
            plt.plot(years, fmod_impl.volume_km3_ts(), 'r')
            plt.plot(years, fmod_flux.volume_km3_ts(), 'b')
            plt.title('Compare Volume')
            plt.xlabel('years')
            plt.ylabel('[km3]')
            plt.legend(['Implicit', 'Flux'], loc=2)

            plt.figure()
            plt.plot(years, fmod_impl.area_km2_ts(), 'r')
            plt.plot(years, fmod_flux.area_km2_ts(), 'b')
            plt.title('Compare Area')
            plt.xlabel('years')
            plt.ylabel('[km2]')
            plt.legend(['Implicit', 'Flux'], loc=2)

            plt.show()

        for year in years:
            fmod_impl.run_until(year)
            fmod_flux.run_until(year)

            np.testing.assert_allclose(fmod_flux.fls[-1].length_m,
                                       fmod_impl.fls[-1].length_m,
                                       atol=100.1)
            assert utils.rmsd(fmod_impl.fls[-1].thick,
                              fmod_flux.fls[-1].thick) < 2.5

        # compare velocities
        f = hef_elev_gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_implicit_run')
        with xr.open_dataset(f, group='fl_0') as ds_impl:
            ds_impl = ds_impl.load()

        f = hef_elev_gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_fluxbased_run')
        with xr.open_dataset(f, group='fl_0') as ds_flux:
            ds_flux = ds_flux.load()

        # only used to plot the velocity with the largest difference
        max_velocity_rmsd = 0
        max_velocity_year = 0

        for year in np.arange(1, 1001):
            velocity_impl = ds_impl.ice_velocity_myr.loc[{'time': year}].values
            velocity_flux = ds_flux.ice_velocity_myr.loc[{'time': year}].values

            velocity_rmsd = utils.rmsd(velocity_impl, velocity_flux)
            if velocity_rmsd > max_velocity_rmsd:
                max_velocity_rmsd = velocity_rmsd
                max_velocity_year = year

            assert velocity_rmsd < 7

        if do_plot:
            plt.figure()
            plt.plot(ds_impl.ice_velocity_myr.loc[{'time': max_velocity_year}].values,
                     'r')
            plt.plot(ds_flux.ice_velocity_myr.loc[{'time': max_velocity_year}].values,
                     'b')
            plt.title(f'Compare Velocity at year {max_velocity_year}')
            plt.xlabel('gridpoints along flowline')
            plt.ylabel('[m yr-1]')
            plt.legend(['Implicit', 'Flux'], loc=2)

            plt.show()

        # Testing the run times should be the last test, as it is not
        # independent of the computing environment and by putting it as the
        # last test all other tests will still be executed
        if impl_time_needed > flux_time_needed / 2:
            pytest.xfail(f'SemiImplicitModel ({impl_time_needed:.2f} s) is '
                         f'not twice as fast as FluxBasedModel ('
                         f'{flux_time_needed:.2f} s).')

    @pytest.mark.slow
    def test_fixed_dt(self, hef_elev_gdir, inversion_params):
        cfg.PARAMS['store_model_geometry'] = True
        cfg.PARAMS['store_fl_diagnostics'] = True
        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        cfg.PARAMS['downstream_line_shape'] = 'trapezoidal'
        init_present_time_glacier(hef_elev_gdir)

        # test if a large fixed_dt results in an instability
        run_constant_climate(hef_elev_gdir, nyears=100, y0=1985,
                             temperature_bias=-1,
                             fs=inversion_params['inversion_fs'],
                             glen_a=inversion_params['inversion_glen_a'],
                             output_filesuffix='_cfl_criterion',
                             evolution_model=SemiImplicitModel)
        f = hef_elev_gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_cfl_criterion')
        with xr.open_dataset(f, group='fl_0') as ds_cfl:
            ds_cfl = ds_cfl.load()

        run_constant_climate(hef_elev_gdir, nyears=100, y0=1985,
                             fixed_dt=SEC_IN_MONTH,
                             fs=inversion_params['inversion_fs'],
                             glen_a=inversion_params['inversion_glen_a'],
                             bias=0, output_filesuffix='_fixed_dt',
                             evolution_model=SemiImplicitModel)
        f = hef_elev_gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_fixed_dt')
        with xr.open_dataset(f, group='fl_0') as ds_fixed_dt:
            ds_fixed_dt = ds_fixed_dt.load()

        # check there are instabilities when using fixed_dt
        max_velocity_rmsd = 0
        max_velocity_year = 0

        for year in np.arange(1, 101):
            velocity_cfl = ds_cfl.ice_velocity_myr.loc[{'time': year}].values
            velocity_fixed_dt = ds_fixed_dt.ice_velocity_myr.loc[{'time': year}].values

            velocity_rmsd = utils.rmsd(velocity_cfl, velocity_fixed_dt)
            if velocity_rmsd > max_velocity_rmsd:
                max_velocity_rmsd = velocity_rmsd
                max_velocity_year = year

        assert max_velocity_rmsd > 150

        if do_plot:
            plt.figure()
            plt.plot(ds_cfl.ice_velocity_myr.loc[{'time': max_velocity_year}].values,
                     'r')
            plt.plot(ds_fixed_dt.ice_velocity_myr.loc[{'time': max_velocity_year}].values,
                     'b')
            plt.title(f'Compare Velocity at year {max_velocity_year}')
            plt.xlabel('gridpoints along flowline')
            plt.ylabel('[m yr-1]')
            plt.legend(['CFL', 'Fixed dt'], loc=2)

            plt.show()


class TestDistribute2D:

    @pytest.mark.slow
    def test_distribute(self, hef_elev_gdir, inversion_params):
        # As long as hef_gdir uses 1, we need to use 1 here as well
        cfg.PARAMS['trapezoid_lambdas'] = 1
        cfg.PARAMS['downstream_line_shape'] = 'trapezoidal'
        init_present_time_glacier(hef_elev_gdir)
        cfg.PARAMS['min_ice_thick_for_length'] = 1

        # This can be done without any run
        from oggm.sandbox import distribute_2d
        distribute_2d.add_smoothed_glacier_topo(hef_elev_gdir)
        tasks.distribute_thickness_per_altitude(hef_elev_gdir);
        distribute_2d.assign_points_to_band(hef_elev_gdir)

        mb_mod = massbalance.RandomMassBalance(hef_elev_gdir, y0=1980,
                                               seed=4)
        mb_mod.temp_bias = 0.5

        fls = hef_elev_gdir.read_pickle('model_flowlines')
        model = SemiImplicitModel(fls, mb_model=mb_mod, y0=2000,
                                  fs=inversion_params['inversion_fs'],
                                  glen_a=inversion_params['inversion_glen_a'])

        fl_diag_path = hef_elev_gdir.get_filepath('fl_diagnostics',
                                                  filesuffix='_commit',
                                                  delete=True)
        ds_diag, fl_diag = model.run_until_and_store(2100, fl_diag_path=fl_diag_path)
        fl_diag = fl_diag[0]

        distribute_2d.distribute_thickness_from_simulation(hef_elev_gdir,
                                                           input_filesuffix='_commit')
        fp = hef_elev_gdir.get_filepath('gridded_simulation', filesuffix='_commit')
        with xr.open_dataset(fp) as ds:
            thick = ds.distributed_thickness.load()
            
        fp = hef_elev_gdir.get_filepath('gridded_data')
        with xr.open_dataset(fp) as ds:
            ds = ds.load()
        dx2 = hef_elev_gdir.grid.dx ** 2

        area_dis = (thick > 0).sum(dim=('x', 'y')) * dx2
        vol_dis = thick.sum(dim=('x', 'y')) * dx2

        # We have a very close volume and area conservation
        assert_allclose(area_dis, ds_diag.area_m2, rtol=0.01)
        assert_allclose(vol_dis, ds_diag.volume_m3, rtol=0.01)

        # The flowline views should be quite good as well
        fl_diag['area_m2_dis'] = fl_diag['area_m2'] * 0
        fl_diag['volume_m3_dis'] = fl_diag['volume_m3'] * 0

        band_ids = np.unique(np.sort(ds.band_index.data[ds.glacier_mask == 1])).astype(int)
        fl_diag = fl_diag.isel(dis_along_flowline=slice(0, band_ids.max()+1))
        for bid in band_ids:
            thick_band = thick.where(ds.band_index == bid)
            fl_diag['volume_m3_dis'].data[:, bid] = thick_band.sum(dim=['x', 'y']) * dx2
            fl_diag['area_m2_dis'].data[:, bid] = (thick_band > 1).sum(dim=['x', 'y']) * dx2

        for yr in [2003]:
            # All the other years have larger errors but they somehow still look
            # OK - just harder to test
            sel = fl_diag.sel(time=yr)
            assert_allclose(sel['volume_m3_dis'], sel['volume_m3'], rtol=0.01, atol=2e6)
            assert_allclose(sel['area_m2_dis'], sel['area_m2'], rtol=0.01, atol=1e4)

        if do_plot:
            yr = 2030
            plt.figure()
            f, ax = plt.subplots()
            fl_diag.sel(time=yr)['volume_m3'].plot(ax=ax);
            fl_diag.sel(time=yr)['volume_m3_dis'].plot(ax=ax);
            plt.show(); plt.figure();
            f, ax = plt.subplots()
            fl_diag.sel(time=yr)['area_m2'].plot(ax=ax);
            fl_diag.sel(time=yr)['area_m2_dis'].plot(ax=ax);
            plt.show(); plt.figure();
            f, ax = plt.subplots()
            ds_diag.area_m2.plot(ax=ax);
            area_dis.plot(ax=ax);
            plt.show(); plt.figure();
            f, ax = plt.subplots()
            ds_diag.volume_m3.plot(ax=ax);
            vol_dis.plot(ax=ax);
            plt.show();

        if False:
            from matplotlib import animation
            import matplotlib; matplotlib.use("TkAgg");
            # Get a handle on the figure and the axes
            fig, ax = plt.subplots()
            # Plot the initial frame.
            cax = thick.isel(time=0).plot(add_colorbar=True, cmap='viridis',
                vmin=0, vmax=350, cbar_kwargs={'extend': 'neither'})
            def animate(frame):
                cax.set_array(thick.values[frame, :].flatten())
            animation.FuncAnimation(fig, animate, frames=len(thick.time),
                                    interval=200)
            plt.show()
