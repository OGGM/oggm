import os
import warnings

import matplotlib.pyplot as plt
import pytest
salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')

import oggm
import xarray as xr
import rioxarray as rioxr
import numpy as np
import pandas as pd
from oggm import utils
from oggm.utils import get_demo_file
from oggm.shop import its_live, rgitopo, bedtopo, millan22, hugonnet_maps, glathida
from oggm.core import gis, centerlines, massbalance
from oggm import cfg, tasks, workflow

pytestmark = pytest.mark.test_env("utils")

DO_PLOT = False


class Test_its_live:

    @pytest.mark.slow
    def test_repro_to_glacier(self, class_case_dir, monkeypatch):

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('dem_Columbia.tif')
        cfg.PARAMS['border'] = 10

        entity = gpd.read_file(get_demo_file('RGI60-01.10689.shp')).iloc[0]
        gdir = oggm.GlacierDirectory(entity)
        tasks.define_glacier_region(gdir)
        tasks.glacier_masks(gdir)

        # use our files
        region_files = {'ALA': {
            'vx': get_demo_file('crop_ALA_G0120_0000_vx.tif'),
            'vy': get_demo_file('crop_ALA_G0120_0000_vy.tif')}}
        monkeypatch.setattr(its_live, 'region_files', region_files)
        monkeypatch.setattr(utils, 'file_downloader', lambda x: x)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            its_live.velocity_to_gdir(gdir)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            mask = ds.glacier_mask.data.astype(bool)
            vx = ds.itslive_vx.where(mask).data
            vy = ds.itslive_vy.where(mask).data

        vel = ds.itslive_v.where(mask).data
        assert np.nanmax(vel) > 2900
        assert np.nanmin(vel) < 2

        # We reproject with rasterio and check no big diff
        cfg.BASENAMES['itslive_vx'] = ('itslive_vx.tif', '')
        cfg.BASENAMES['itslive_vy'] = ('itslive_vy.tif', '')
        gis.rasterio_to_gdir(gdir, region_files['ALA']['vx'], 'itslive_vx',
                             resampling='bilinear')
        gis.rasterio_to_gdir(gdir, region_files['ALA']['vy'], 'itslive_vy',
                             resampling='bilinear')

        with rioxr.open_rasterio(gdir.get_filepath('itslive_vx')) as da:
            _vx = da.where(mask).data.squeeze()
        with rioxr.open_rasterio(gdir.get_filepath('itslive_vy')) as da:
            _vy = da.where(mask).data.squeeze()

        _vel = np.sqrt(_vx**2 + _vy**2)
        np.testing.assert_allclose(utils.rmsd(vel[mask], _vel[mask]), 0,
                                   atol=40)
        np.testing.assert_allclose(utils.md(vel[mask], _vel[mask]), 0,
                                   atol=8)

        df = its_live.compile_itslive_statistics([gdir]).iloc[0]
        assert df['itslive_avg_vel'] > 180
        assert df['itslive_max_vel'] > 2000
        assert df['itslive_perc_cov'] > 0.95

        if DO_PLOT:

            smap = salem.Map(gdir.grid.center_grid, countries=False)
            smap.set_shapefile(gdir.read_shapefile('outlines'))

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                smap.set_topography(gdir.get_filepath('dem'))

            smap.set_data(vel)
            smap.set_plot_params(cmap='Blues', vmin=None, vmax=None)

            xx, yy = gdir.grid.center_grid.xy_coordinates
            xx, yy = smap.grid.transform(xx, yy, crs=gdir.grid.proj)

            yy = yy[2::5, 2::5]
            xx = xx[2::5, 2::5]
            vx = vx[2::5, 2::5]
            vy = vy[2::5, 2::5]

            f, ax = plt.subplots()
            smap.visualize(ax=ax, title='ITS_LIVE velocity',
                           cbar_title='m yr-1')
            ax.quiver(xx, yy, vx, vy)
            plt.show()


class Test_millan22:

    @pytest.mark.slow
    def test_thickvel_to_glacier(self, class_case_dir, monkeypatch):

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('dem_Columbia.tif')
        cfg.PARAMS['border'] = 10

        entity = gpd.read_file(get_demo_file('RGI60-01.10689.shp')).iloc[0]
        gdir = oggm.GlacierDirectory(entity)
        tasks.define_glacier_region(gdir)
        tasks.glacier_masks(gdir)

        # use our files
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/test_files/millan22/'
        monkeypatch.setattr(millan22, 'default_base_url', base_url)

        millan22.thickness_to_gdir(gdir)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            mask = ds.glacier_mask.data.astype(bool)
            thick = ds.millan_ice_thickness.where(mask).data

        # Simply some coverage and sanity checks
        assert np.isfinite(thick).sum() / mask.sum() > 0.98
        assert np.nanmax(thick) > 800
        assert np.nansum(thick) * gdir.grid.dx**2 * 1e-9 > 174

        # Velocity
        millan22.velocity_to_gdir(gdir)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            mask = ds.glacier_mask.data.astype(bool)
            v = ds.millan_v.where(mask).data
            vx = ds.millan_vx.where(mask).data
            vy = ds.millan_vy.where(mask).data

        # Simply some coverage and sanity checks
        assert np.isfinite(v).sum() / mask.sum() > 0.98
        assert np.nanmax(v) > 2000
        assert np.nanmax(vx) > 500
        assert np.nanmax(vy) > 400

        # Stats
        df = millan22.compile_millan_statistics([gdir]).iloc[0]
        assert df['millan_avg_vel'] > 180
        assert df['millan_max_vel'] > 2000
        assert df['millan_perc_cov'] > 0.95
        assert df['millan_vel_perc_cov'] > 0.97
        assert df['millan_vol_km3'] > 174


class Test_HugonnetMaps:

    @pytest.mark.slow
    def test_dhdt_to_glacier(self, class_case_dir, monkeypatch):

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('dem_Columbia.tif')
        cfg.PARAMS['border'] = 10

        entity = gpd.read_file(get_demo_file('RGI60-01.10689.shp')).iloc[0]
        gdir = oggm.GlacierDirectory(entity)
        tasks.define_glacier_region(gdir)
        tasks.glacier_masks(gdir)

        # use our files
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/test_files/geodetic_ref_mb_maps/'
        monkeypatch.setattr(hugonnet_maps, 'default_base_url', base_url)

        hugonnet_maps.hugonnet_to_gdir(gdir)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            mask = ds.glacier_mask.data.astype(bool)
            dhdt = ds.hugonnet_dhdt.where(mask)

        # Simply some coverage and sanity checks
        assert np.isfinite(dhdt).sum() / mask.sum() > 0.97
        assert np.nanmax(dhdt) > 0
        assert np.nanmean(dhdt) < -3

        df = hugonnet_maps.compile_hugonnet_statistics([gdir]).iloc[0]
        assert df['hugonnet_perc_cov'] > 0.97
        assert df['hugonnet_avg_dhdt'] < -3


class Test_rgitopo:

    def test_from_dem(self, class_case_dir, monkeypatch):

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PARAMS['border'] = 10

        monkeypatch.setattr(rgitopo, 'DEMS_URL', 'https://cluster.klima.uni-br'
                                                 'emen.de/~oggm/test_gdirs/dem'
                                                 's_v1/default/')

        gd = rgitopo.init_glacier_directories_from_rgitopo(['RGI60-09.01004'],
                                                           dem_source='COPDEM')
        gd = gd[0]

        assert gd.has_file('dem')
        assert gd.has_file('dem_source')
        assert gd.has_file('outlines')
        assert gd.has_file('intersects')

        # we can work from here
        tasks.glacier_masks(gd)

    def test_qc(self, class_case_dir, monkeypatch):

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PARAMS['border'] = 10

        monkeypatch.setattr(rgitopo, 'DEMS_URL', 'https://cluster.klima.uni-br'
                                                 'emen.de/~oggm/test_gdirs/dem'
                                                 's_v1/default/')

        gd = rgitopo.init_glacier_directories_from_rgitopo(['RGI60-09.01004'],
                                                           dem_source='COPDEM',
                                                           keep_dem_folders=True)
        out = rgitopo.dem_quality_check(gd[0])
        assert len(out) > 5
        assert pd.Series(out).iloc[1:].sum() > 5


class Test_w5e5:

    def test_get_gswp3_w5e5_file(self):
        from oggm.shop import w5e5
        d = 'GSWP3_W5E5'
        for vars, _ in w5e5.BASENAMES[d].items():
            assert os.path.isfile(w5e5.get_gswp3_w5e5_file(d, vars))
        with pytest.raises(ValueError):
            w5e5.get_gswp3_w5e5_file(d, 'zoup')

    def test_glacier_gridpoint_selection(self):

        from oggm.shop import w5e5
        d = 'GSWP3_W5E5'
        # test is only done for the `inv` file, as the other files are only
        # downloaded for the HEF gridpoints as they would be too large otherwise.
        # However, the same test and other tests are done for all files
        # (also ISIMIP3b) and all glaciers in this notebook:
        # https://nbviewer.org/urls/cluster.klima.uni-bremen.de/
        # ~lschuster/example_ipynb/flatten_glacier_gridpoint_tests.ipynb
        with xr.open_dataset(w5e5.get_gswp3_w5e5_file(d, 'inv')) as dinv:
            dinv = dinv.load()

        # select three glaciers where two failed in the
        # previous gswp3_w5e5 version
        for coord in [(10.7584, 46.8003),  # HEF
                      (-70.8931 + 360, -72.4474),  # RGI60-19.00124
                      (51.495, 30.9010),  # RGI60-12.01691
                      (0, 0)  # random gridpoint not near to a glacier
                      ]:
            lon, lat = coord
            # get the distances to the glacier coordinate
            c = (dinv.longitude - lon) ** 2 + (dinv.latitude - lat) ** 2
            c = c.to_dataframe('distance').sort_values('distance')
            # select the nearest climate point from the flattened glacier gridpoint
            lat_near, lon_near, dist = c.iloc[0]
            # for a randomly chosen gridpoint, the next climate gridpoint is far away
            if coord == (0, 0):
                with pytest.raises(AssertionError):
                    assert np.abs(lat_near - lat) <= 0.25
                    assert np.abs(lon_near - lon) <= 0.25
                    assert dist <= (0.25 ** 2 + 0.25 ** 2) ** 0.5
            # for glaciers the next gridpoint should be the nearest
            # (GSWP3-W5E5 resolution is 0.5°)
            else:
                assert np.abs(lat_near - lat) <= 0.25
                assert np.abs(lon_near - lon) <= 0.25
                assert dist <= (0.25 ** 2 + 0.25 ** 2) ** 0.5

        # this only contains data for two glaciers, let's still check some basics
        # both glaciers are not at latitude or longitude 0
        with xr.open_dataset(w5e5.get_gswp3_w5e5_file(d, 'temp_std')) as dtemp_std:
            assert np.all(dtemp_std.latitude != 0)
            assert np.all(dtemp_std.longitude != 0)
            assert dtemp_std.isel(time=0).temp_std.std() > 0
            assert dtemp_std.longitude.std() > 0
            assert dtemp_std.latitude.std() > 0

    def test_process_w5e5_data(self, class_case_dir):

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        cfg.PARAMS['hydro_month_nh'] = 1
        gdir = workflow.init_glacier_directories(gpd.read_file(hef_file))[0]
        from oggm.shop import w5e5
        w5e5.process_w5e5_data(gdir)
        path_clim = gdir.get_filepath('climate_historical')

        ds_clim = xr.open_dataset(path_clim)
        assert ds_clim.time[0]['time.year'] == 1979
        assert ds_clim.time[-1]['time.year'] == 2019
        assert ds_clim.time[0]['time.month'] == 1
        assert ds_clim.time[-1]['time.month'] == 12
        assert (ds_clim.ref_hgt > 2000) and (ds_clim.ref_hgt < 3000)
        assert (ds_clim.temp.mean() < 5) and (ds_clim.temp.mean() > -5)
        assert ds_clim.temp.min() > -30  # °C
        assert ds_clim.temp.max() < 30
        # temp_std
        assert np.all(ds_clim.temp_std > 0)
        assert np.all(ds_clim.temp_std <= 10)

        # prcp
        assert ds_clim.prcp.min() > 0  # kg/m2/month
        annual_prcp_sum = ds_clim.prcp.groupby('time.year').sum().values
        assert ds_clim.prcp.max() < annual_prcp_sum.mean()  # kg/m2/month
        assert np.all(annual_prcp_sum > 500)  # kg /m2/year
        assert np.all(annual_prcp_sum < 1500)

    def test_process_gswp3_w5e5_data(self, class_case_dir):

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        cfg.PARAMS['hydro_month_nh'] = 1
        gdir = workflow.init_glacier_directories(gpd.read_file(hef_file))[0]
        from oggm.shop import w5e5
        w5e5.process_gswp3_w5e5_data(gdir)
        path_clim = gdir.get_filepath('climate_historical')

        ds_clim = xr.open_dataset(path_clim)
        assert ds_clim.time[0]['time.year'] == 1901
        assert ds_clim.time[-1]['time.year'] == 2019
        assert ds_clim.time[0]['time.month'] == 1
        assert ds_clim.time[-1]['time.month'] == 12
        assert (ds_clim.ref_hgt > 2000) and (ds_clim.ref_hgt < 3000)
        assert (ds_clim.temp.mean() < 5) and (ds_clim.temp.mean() > -5)
        assert ds_clim.temp.min() > -30  # °C
        assert ds_clim.temp.max() < 30
        # temp_std
        assert np.all(ds_clim.temp_std > 0)
        assert np.all(ds_clim.temp_std <= 10)

        # prcp
        try:
            assert ds_clim.prcp.min() > 0  # kg/m2/month
        except AssertionError:
            # ok, we allow a small amount of months with just 0 prcp,
            # (here it is just one month)
            assert ds_clim.prcp.quantile(0.001) >= 0
            assert ds_clim.prcp.min() >= 0
        annual_prcp_sum = ds_clim.prcp.groupby('time.year').sum().values
        assert ds_clim.prcp.max() < annual_prcp_sum.mean()  # kg/m2/month
        assert np.all(annual_prcp_sum > 400)  # kg /m2/year
        assert np.all(annual_prcp_sum < 1500)
        # no gradient for GSWP3-W5E5!

        # test climate statistics with winter_daily_mean_prcp
        # they should be computed even if cfg.PARAMS['use_winter_prcp_fac'] is False!
        df = utils.compile_climate_statistics([gdir], path=False,
                                              add_climate_period=[1999, 2010],
                                              add_raw_climate_statistics=True,
                                              halfsize=20)
        fs = '1979-2019'
        assert np.all(df[f'{fs}_uncorrected_winter_daily_mean_prcp'] > 1.5)
        assert np.all(df[f'{fs}_uncorrected_winter_daily_mean_prcp'] < 1.8)

        # we don't have climate data for that time period
        with pytest.raises(KeyError):
            df['1990-2030_uncorrected_winter_daily_mean_prcp']


class Test_ecmwf:

    def test_get_ecmwf_file(self):
        from oggm.shop import ecmwf
        for d, vars in ecmwf.BASENAMES.items():
            for v, _ in vars.items():
                assert os.path.isfile(ecmwf.get_ecmwf_file(d, v))

        with pytest.raises(ValueError):
            ecmwf.get_ecmwf_file('ERA5', 'zoup')
        with pytest.raises(ValueError):
            ecmwf.get_ecmwf_file('zoup', 'tmp')

    def test_ecmwf_historical_delta_method(self, class_case_dir):

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PARAMS['hydro_month_nh'] = 10
        cfg.PARAMS['hydro_month_sh'] = 4

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')

        gdir = workflow.init_glacier_directories(gpd.read_file(hef_file))[0]
        tasks.process_ecmwf_data(gdir, dataset='ERA5',
                                 output_filesuffix='ERA5')
        tasks.process_ecmwf_data(gdir, dataset='CERA',
                                 output_filesuffix='CERA')

        # Original BC
        tasks.historical_delta_method(gdir,
                                      replace_with_ref_data=False,
                                      delete_input_files=False,
                                      ref_filesuffix='ERA5',
                                      hist_filesuffix='CERA',
                                      output_filesuffix='CERA_alone')

        f_ref = gdir.get_filepath('climate_historical', filesuffix='ERA5')
        f_h = gdir.get_filepath('climate_historical', filesuffix='CERA_alone')
        with xr.open_dataset(f_ref) as ref, xr.open_dataset(f_h) as his:

            # Let's do some basic checks
            assert ref.attrs['ref_hgt'] == his.attrs['ref_hgt']
            ci = gdir.get_climate_info('CERA_alone')
            assert ci['baseline_climate_source'] == 'CERA|ERA5'
            assert ci['baseline_yr_0'] == 1901
            assert ci['baseline_yr_1'] == 2010

            # Climate on common period
            # (minus one year because of the automated stuff in code
            sref = ref.sel(time=slice(ref.time[0], his.time[-1]))
            shis = his.sel(time=slice(ref.time[0], his.time[-1]))

            # Climate during the chosen period should be the same
            np.testing.assert_allclose(sref.temp.mean(),
                                       shis.temp.mean(),
                                       atol=1e-3)
            np.testing.assert_allclose(sref.prcp.mean(),
                                       shis.prcp.mean(),
                                       rtol=1e-3)

            # And also the annual cycle
            srefm = sref.groupby('time.month').mean(dim='time')
            shism = shis.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(srefm.temp, shism.temp, atol=1e-3)
            np.testing.assert_allclose(srefm.prcp, shism.prcp, rtol=1e-3)

            # And its std dev - but less strict
            srefm = sref.groupby('time.month').std(dim='time')
            shism = shis.groupby('time.month').std(dim='time')
            np.testing.assert_allclose(srefm.temp, shism.temp, rtol=5e-2)
            with pytest.raises(AssertionError):
                # This clearly is not scaled
                np.testing.assert_allclose(srefm.prcp, shism.prcp, rtol=0.5)

        # Replaced
        tasks.historical_delta_method(gdir,
                                      replace_with_ref_data=True,
                                      delete_input_files=False,
                                      ref_filesuffix='ERA5',
                                      hist_filesuffix='CERA',
                                      output_filesuffix='CERA_repl')

        f_ref = gdir.get_filepath('climate_historical', filesuffix='ERA5')
        f_h = gdir.get_filepath('climate_historical', filesuffix='CERA_repl')
        f_hr = gdir.get_filepath('climate_historical', filesuffix='CERA')
        with xr.open_dataset(f_ref) as ref, xr.open_dataset(f_h) as his, \
                xr.open_dataset(f_hr) as his_ref:

            # Let's do some basic checks
            assert ref.attrs['ref_hgt'] == his.attrs['ref_hgt']
            ci = gdir.get_climate_info('CERA_repl')
            assert ci['baseline_climate_source'] == 'CERA+ERA5'
            assert ci['baseline_yr_0'] == 1901
            assert ci['baseline_yr_1'] == 2018

            # Climate on common period
            sref = ref.sel(time=slice(ref.time[0], his.time[-1]))
            shis = his.sel(time=slice(ref.time[0], his.time[-1]))

            # Climate during the chosen period should be the same
            np.testing.assert_allclose(sref.temp.mean(),
                                       shis.temp.mean())
            np.testing.assert_allclose(sref.prcp.mean(),
                                       shis.prcp.mean())

            # And also the annual cycle
            srefm = sref.groupby('time.month').mean(dim='time')
            shism = shis.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(srefm.temp, shism.temp)
            np.testing.assert_allclose(srefm.prcp, shism.prcp)

            # And its std dev - should be same
            srefm = sref.groupby('time.month').std(dim='time')
            shism = shis.groupby('time.month').std(dim='time')
            np.testing.assert_allclose(srefm.temp, shism.temp)
            np.testing.assert_allclose(srefm.prcp, shism.prcp)

            # In the past the two CERA datasets are different
            his_ref = his_ref.sel(time=slice('1910', '1940'))
            his = his.sel(time=slice('1910', '1940'))
            assert np.abs(his.temp.mean() - his_ref.temp.mean()) > 1
            assert np.abs(his.temp.std() - his_ref.temp.std()) > 0.3

        # Delete files
        tasks.historical_delta_method(gdir,
                                      ref_filesuffix='ERA5',
                                      hist_filesuffix='CERA')
        assert not os.path.exists(gdir.get_filepath('climate_historical',
                                                    filesuffix='ERA5'))
        assert not os.path.exists(gdir.get_filepath('climate_historical',
                                                    filesuffix='CERA'))

    def test_ecmwf_workflow(self, class_case_dir):

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        gdir = workflow.init_glacier_directories(gpd.read_file(hef_file))[0]

        cfg.PARAMS['baseline_climate'] = 'CERA+ERA5L'
        tasks.process_climate_data(gdir)
        assert gdir.get_filepath('climate_historical')
        # Let's do some basic checks
        ci = gdir.get_climate_info()
        assert ci['baseline_climate_source'] == 'CERA+ERA5L'
        assert ci['baseline_yr_0'] == 1901
        assert ci['baseline_yr_1'] == 2018

        cfg.PARAMS['baseline_climate'] = 'CERA|ERA5'
        tasks.process_climate_data(gdir)
        assert gdir.get_filepath('climate_historical')
        # Let's do some basic checks
        ci = gdir.get_climate_info()
        assert ci['baseline_climate_source'] == 'CERA|ERA5'
        assert ci['baseline_yr_0'] == 1901
        assert ci['baseline_yr_1'] == 2010


class Test_climate_datasets:

    def test_all_at_once(self, class_case_dir):

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')

        gdir = workflow.init_glacier_directories(gpd.read_file(hef_file))[0]

        exps = ['W5E5', 'GSWP3_W5E5', 'CRU', 'HISTALP', 'ERA5', 'ERA5L', 'CERA']
        ref_hgts = []
        dft = []
        dfp = []
        for base in exps:
            cfg.PARAMS['baseline_climate'] = base
            tasks.process_climate_data(gdir, output_filesuffix=base)
            f = gdir.get_filepath('climate_historical', filesuffix=base)
            with xr.open_dataset(f) as ds:
                ref_hgts.append(ds.ref_hgt)
                assert ds.ref_pix_dis < 30000
                dft.append(ds.temp.to_series())
                dfp.append(ds.prcp.to_series())
        dft = pd.concat(dft, axis=1, keys=exps)
        dfp = pd.concat(dfp, axis=1, keys=exps)

        # Common period
        dfy = dft.resample('AS').mean().dropna().iloc[1:]
        dfm = dft.groupby(dft.index.month).mean()
        assert dfy.corr().min().min() > 0.44  # ERA5L and CERA do no correlate
        assert dfm.corr().min().min() > 0.97
        dfavg = dfy.describe()

        # Correct for hgt
        ref_h = ref_hgts[0]
        for h, d in zip(ref_hgts, exps):
            dfy[d] = dfy[d] - 0.0065 * (ref_h - h)
            dfm[d] = dfm[d] - 0.0065 * (ref_h - h)
        dfavg_cor = dfy.describe()

        # After correction less spread
        assert dfavg_cor.loc['mean'].std() < 0.8 * dfavg.loc['mean'].std()
        assert dfavg_cor.loc['mean'].std() < 2.1

        # PRECIP
        # Common period
        dfy = dfp.resample('AS').mean().dropna().iloc[1:] * 12
        dfm = dfp.groupby(dfp.index.month).mean()
        assert dfy.corr().min().min() > 0.5
        assert dfm.corr().min().min() > 0.8
        dfavg = dfy.describe()
        assert dfavg.loc['mean'].std() / dfavg.loc['mean'].mean() < 0.25  # %

    def test_vdr(self, class_case_dir):

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')

        gdir = workflow.init_glacier_directories(gpd.read_file(hef_file))[0]

        exps = ['ERA5', 'ERA5dr']
        files = []
        ref_hgts = []
        for base in exps:
            cfg.PARAMS['baseline_climate'] = base
            tasks.process_climate_data(gdir, output_filesuffix=base)
            files.append(gdir.get_filepath('climate_historical',
                                           filesuffix=base))
            with xr.open_dataset(files[-1]) as ds:
                ref_hgts.append(ds.ref_hgt)
                assert ds.ref_pix_dis < 10000

        with xr.open_dataset(files[0]) as d1, xr.open_dataset(files[1]) as d2:
            np.testing.assert_allclose(d1.temp, d2.temp)
            np.testing.assert_allclose(d1.prcp, d2.prcp)
            # Fake tests, the plots look plausible
            np.testing.assert_allclose(d2.temp_std.mean(), 3.35, atol=0.1)


class Test_bedtopo:

    def test_add_consensus(self, class_case_dir, monkeypatch):

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')

        entity = gpd.read_file(get_demo_file('Hintereisferner_RGI5.shp'))
        entity['RGIId'] = 'RGI60-11.00897'
        gdir = workflow.init_glacier_directories(entity)[0]
        tasks.define_glacier_region(gdir)
        tasks.glacier_masks(gdir)

        ft = utils.get_demo_file('RGI60-11.00897_thickness.tif')
        monkeypatch.setattr(utils, 'file_downloader', lambda x: ft)
        bedtopo.add_consensus_thickness(gdir)

        # Check with rasterio
        cfg.add_to_basenames('consensus', 'consensus.tif')
        gis.rasterio_to_gdir(gdir, ft, 'consensus', resampling='bilinear')

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            mine = ds.consensus_ice_thickness

        with rioxr.open_rasterio(gdir.get_filepath('consensus')) as ds:
            ref = ds.isel(band=0)

        # Check area
        my_area = np.sum(np.isfinite(mine.data)) * gdir.grid.dx**2
        np.testing.assert_allclose(my_area, gdir.rgi_area_m2, rtol=0.07)

        rio_area = np.sum(ref.data > 0) * gdir.grid.dx**2
        np.testing.assert_allclose(rio_area, gdir.rgi_area_m2, rtol=0.15)
        np.testing.assert_allclose(my_area, rio_area, rtol=0.15)

        # They are not same:
        # - interpolation not 1to1 same especially at borders
        # - we preserve total volume
        np.testing.assert_allclose(mine.sum(), ref.sum(), rtol=0.01)
        assert utils.rmsd(ref, mine) < 2

        # Check vol
        cdf = pd.read_hdf(utils.get_demo_file('rgi62_itmix_df.h5'))
        ref_vol = cdf.loc[gdir.rgi_id].vol_itmix_m3
        my_vol = mine.sum() * gdir.grid.dx**2
        np.testing.assert_allclose(my_vol, ref_vol)

        # Now check the rest of the workflow
        # Check that no error when var not there
        vn = 'consensus_ice_thickness'
        centerlines.elevation_band_flowline(gdir, bin_variables=[vn, 'foo'])

        # Check vol
        df = pd.read_csv(gdir.get_filepath('elevation_band_flowline'),
                         index_col=0)
        my_vol = (df[vn] * df['area']).sum()
        np.testing.assert_allclose(my_vol, ref_vol)

        centerlines.fixed_dx_elevation_band_flowline(gdir,
                                                     bin_variables=[vn, 'foo'])
        fdf = pd.read_csv(gdir.get_filepath('elevation_band_flowline',
                                            filesuffix='_fixed_dx'),
                          index_col=0)

        # Check vol
        my_vol = (fdf[vn] * fdf['area_m2']).sum()
        np.testing.assert_allclose(my_vol, ref_vol)

        df = bedtopo.compile_consensus_statistics([gdir]).iloc[0]
        np.testing.assert_allclose(my_vol*1e-9, df['consensus_vol_km3'], rtol=1e-2)
        np.testing.assert_allclose(1, df['consensus_perc_cov'], rtol=0.05)


class Test_Glathida:

    @pytest.mark.slow
    def test_to_glacier(self, class_case_dir, monkeypatch):

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('dem_Columbia.tif')
        cfg.PARAMS['border'] = 10

        entity = gpd.read_file(get_demo_file('RGI60-01.10689.shp')).iloc[0]
        gdir = oggm.GlacierDirectory(entity)
        tasks.define_glacier_region(gdir)
        tasks.glacier_masks(gdir)

        # use our files
        base_url = ('https://cluster.klima.uni-bremen.de/~oggm/test_files/'
                    'glathida/glathida_2023-11-16_rgi_{}.h5')
        monkeypatch.setattr(glathida, 'GTD_BASE_URL', base_url)

        df = glathida.glathida_to_gdir(gdir)
        assert 11000 < len(df) < 12000

        assert 'i_grid' in df
        assert 'x_proj' in df

        dsf = df[['ij_grid', 'thickness']].groupby('ij_grid').mean()
        assert 1600 < len(dsf) < 1700

        sdf = glathida.compile_glathida_statistics([gdir]).iloc[0]

        assert sdf['n_valid_gridded_points'] < sdf['n_valid_thick_points']
        assert sdf['date_mode'] < sdf['date_max']
        assert sdf['avg_thick'] < sdf['max_thick']
