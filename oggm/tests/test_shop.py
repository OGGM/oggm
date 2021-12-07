import os
import warnings

import pytest
salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')

import oggm
import xarray as xr
import numpy as np
import pandas as pd
from oggm import utils
from oggm.utils import get_demo_file
from oggm.shop import its_live, rgitopo, bedtopo
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
        region_files = {'ALA':
                            {'vx': get_demo_file('crop_ALA_G0120_0000_vx.tif'),
                             'vy': get_demo_file('crop_ALA_G0120_0000_vy.tif')}
                        }
        monkeypatch.setattr(its_live, 'region_files', region_files)
        monkeypatch.setattr(utils, 'file_downloader', lambda x: x)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            its_live.velocity_to_gdir(gdir)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            mask = ds.glacier_mask.data.astype(bool)
            vx = ds.obs_icevel_x.where(mask).data
            vy = ds.obs_icevel_y.where(mask).data

        vel = np.sqrt(vx**2 + vy**2)
        assert np.nanmax(vel) > 2900
        assert np.nanmin(vel) < 2

        # We reproject with rasterio and check no big diff
        cfg.BASENAMES['its_live_vx'] = ('its_live_vx.tif', '')
        cfg.BASENAMES['its_live_vy'] = ('its_live_vy.tif', '')
        gis.rasterio_to_gdir(gdir, region_files['ALA']['vx'], 'its_live_vx',
                             resampling='bilinear')
        gis.rasterio_to_gdir(gdir, region_files['ALA']['vy'], 'its_live_vy',
                             resampling='bilinear')

        with xr.open_rasterio(gdir.get_filepath('its_live_vx')) as da:
            _vx = da.where(mask).data.squeeze()
        with xr.open_rasterio(gdir.get_filepath('its_live_vy')) as da:
            _vy = da.where(mask).data.squeeze()

        _vel = np.sqrt(_vx**2 + _vy**2)
        np.testing.assert_allclose(utils.rmsd(vel[mask], _vel[mask]), 0,
                                   atol=40)
        np.testing.assert_allclose(utils.md(vel[mask], _vel[mask]), 0,
                                   atol=8)

        if DO_PLOT:
            import matplotlib.pyplot as plt

            smap = salem.Map(gdir.grid.center_grid, countries=False)
            smap.set_shapefile(gdir.read_shapefile('outlines'))

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                smap.set_topography(gdir.get_filepath('dem'))

            vel = np.sqrt(vx ** 2 + vy ** 2)
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


class Test_rgitopo:

    def test_from_dem(self, class_case_dir, monkeypatch):

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PARAMS['border'] = 10

        monkeypatch.setattr(rgitopo, 'DEMS_URL', 'https://cluster.klima.uni-br'
                                                 'emen.de/~oggm/test_gdirs/dem'
                                                 's_v1/default/')

        gd = rgitopo.init_glacier_directories_from_rgitopo(['RGI60-09.01004'])
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
                                                           keep_dem_folders=True)
        out = rgitopo.dem_quality_check(gd[0])
        assert len(out) > 5
        assert np.sum(list(out.values())) > 5


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
            assert ci['baseline_hydro_yr_0'] == 1902
            assert ci['baseline_hydro_yr_1'] == 2010

            # Climate on common period
            # (minus one year because of the automated stuff in code
            sref = ref.sel(time=slice(ref.time[12], his.time[-1]))
            shis = his.sel(time=slice(ref.time[12], his.time[-1]))

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
            assert ci['baseline_hydro_yr_0'] == 1902
            assert ci['baseline_hydro_yr_1'] == 2018

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
        f_ref = gdir.get_filepath('climate_historical')
        with xr.open_dataset(f_ref) as his:
            # Let's do some basic checks
            ci = gdir.get_climate_info()
            assert ci['baseline_climate_source'] == 'CERA+ERA5L'
            assert ci['baseline_hydro_yr_0'] == 1902
            assert ci['baseline_hydro_yr_1'] == 2018

        cfg.PARAMS['baseline_climate'] = 'CERA|ERA5'
        tasks.process_climate_data(gdir)
        f_ref = gdir.get_filepath('climate_historical')
        with xr.open_dataset(f_ref) as his:
            # Let's do some basic checks
            ci = gdir.get_climate_info()
            assert ci['baseline_climate_source'] == 'CERA|ERA5'
            assert ci['baseline_hydro_yr_0'] == 1902
            assert ci['baseline_hydro_yr_1'] == 2010


class Test_climate_datasets:

    def test_all_at_once(self, class_case_dir):

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = class_case_dir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')

        gdir = workflow.init_glacier_directories(gpd.read_file(hef_file))[0]

        exps = ['CRU', 'HISTALP', 'ERA5', 'ERA5L', 'CERA']
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
            np.testing.assert_allclose(d2.gradient.mean(), -0.0058, atol=.001)
            np.testing.assert_allclose(d2.temp_std.mean(), 3.35, atol=0.1)

    @pytest.mark.slow
    def test_hydro_month_changes(self, hef_gdir):
        # test for HEF if applying different hydro_months does the right thing
        # check if mb of neighbouring hydro_months correlate
        # do this for different climate scenarios

        # maybe there is already somewhere an overview or a better way to get
        # these dates, but I did not find it
        base_data_time = {'CRU': {'start_year': 1901, 'end_year': 2014},
                          'ERA5': {'start_year': 1979, 'end_year': 2018},
                          'ERA5dr': {'start_year': 1979, 'end_year': 2019},
                          'HISTALP': {'start_year': 1850, 'end_year': 2014},
                          'CERA': {'start_year': 1901, 'end_year': 2010},
                          'ERA5L': {'start_year': 1981, 'end_year': 2018}}

        gdir = hef_gdir
        oggm.core.flowline.init_present_time_glacier(gdir)
        mb_mod = oggm.core.massbalance.PastMassBalance(gdir)
        h, w = gdir.get_inversion_flowline_hw()

        exps = ['ERA5dr', 'CRU', 'HISTALP', 'ERA5', 'ERA5L', 'CERA']
        for base in exps:
            # this does not need to be the best one,
            # just for comparison between different hydro months
            mu_opt = 213.54

            files = []
            ref_hgts = []
            dft = []
            dfp = []
            tot_mbs = []
            cfg.PARAMS['baseline_climate'] = base

            for m in np.arange(1, 13):
                cfg.PARAMS['hydro_month_nh'] = m
                fsuff = '_{}_{}'.format(base, m)
                tasks.process_climate_data(gdir, output_filesuffix=fsuff)
                files.append(gdir.get_filepath('climate_historical',
                                               filesuffix=fsuff))

                with xr.open_dataset(files[-1]) as ds:
                    ref_hgts.append(ds.ref_hgt)
                    dft.append(ds.temp.to_series())
                    dfp.append(ds.prcp.to_series())

                    ci = gdir.get_climate_info(input_filesuffix=fsuff)

                    # check if the right climate source is used
                    assert base in ci['baseline_climate_source']
                    mm = str(m) if m > 9 else str(0)+str(m)
                    mm_e = str(m-1) if (m-1) > 9 else str(0)+str(m-1)
                    b_s_y = base_data_time[base]['start_year']
                    b_e_y = base_data_time[base]['end_year']

                    stime = '{}-{}-01'.format(b_s_y, mm)
                    assert ds.time[0] == np.datetime64(stime)
                    if m == 1:
                        assert ci['baseline_hydro_yr_0'] == b_s_y
                        if base == 'ERA5dr':
                            # do not have full 2019
                            assert ci['baseline_hydro_yr_1'] == b_e_y - 1
                        else:
                            assert ci['baseline_hydro_yr_1'] == b_e_y

                    elif m < 7 and base == 'ERA5dr':
                        # have data till 2019-05 for ERA5dr
                        stime = '{}-{}-01'.format(b_e_y, mm_e)
                        assert ds.time[-1] == np.datetime64(stime)
                        assert ci['baseline_hydro_yr_0'] == b_s_y + 1
                        assert ci['baseline_hydro_yr_1'] == b_e_y

                    else:
                        assert ci['baseline_hydro_yr_0'] == b_s_y + 1
                        if base == 'ERA5dr':
                            # do not have full 2019
                            stime = '{}-{}-01'.format(b_e_y-1, mm_e)
                            assert ds.time[-1] == np.datetime64(stime)
                            assert ci['baseline_hydro_yr_1'] == b_e_y - 1
                        else:
                            assert ci['baseline_hydro_yr_1'] == b_e_y
                            stime = '{}-{}-01'.format(b_e_y, mm_e)
                            assert ds.time[-1] == np.datetime64(stime)

                    mb_mod = massbalance.PastMassBalance(gdir,
                                                         mu_star=mu_opt,
                                                         input_filesuffix=fsuff,
                                                         bias=0,
                                                         check_calib_params=False)

                    years = np.arange(ds.hydro_yr_0, ds.hydro_yr_1 + 1)
                    mb_ts = mb_mod.get_specific_mb(heights=h, widths=w,
                                                   year=years)
                    tot_mbs.append(pd.Series(mb_ts))

            # check if all ref_hgts are equal
            # means that we likely compare same glacier and climate dataset
            assert len(np.unique(ref_hgts)) == 1
            # concatenate temperature and prcp from different hydromonths
            dft = pd.concat(dft, axis=1, keys=np.arange(1, 13))
            dfp = pd.concat(dfp, axis=1, keys=np.arange(1, 13))
            # Common period
            dft_na = dft.dropna().iloc[1:]
            dfp_na = dfp.dropna().iloc[1:]

            # check if the common period of temperature prcp
            # series is equal for all starting hydromonth dates
            assert np.all(dft_na.eq(dft_na.iloc[:, 0], axis=0).all(1))
            assert np.all(dfp_na.eq(dfp_na.iloc[:, 0], axis=0).all(1))

            # mass balance of different years
            pd_tot_mbs = pd.concat(tot_mbs, axis=1, keys=np.arange(1, 13))
            pd_tot_mbs = pd_tot_mbs.dropna()

            # compute correlations
            corrs = []
            for m in np.arange(1, 12):
                # check if correlation between time series of hydro_month =1,
                # is high to hydro_month = 2 and so on
                corrs.append(pd_tot_mbs.corr().loc[m, m+1])
                # would be better if for hydro_month=12,
                # correlation is tested to next year
            assert np.mean(corrs) > 0.9


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

        with xr.open_rasterio(gdir.get_filepath('consensus')) as ds:
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
