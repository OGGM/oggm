import os
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import pytest
pytest.importorskip('geopandas')
pytest.importorskip('rasterio')
pytest.importorskip('salem')


salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')

import oggm
import xarray as xr
import numpy as np
from oggm import utils
from oggm.utils import get_demo_file
from oggm.shop import its_live, rgitopo
from oggm.core import gis
from oggm import cfg, tasks, workflow

pytestmark = pytest.mark.test_env("utils")

DO_PLOT = False


class Test_its_live:

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
        cfg.PARAMS['use_multiprocessing'] = False
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
                                      out_filesuffix='CERA_alone')

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
            np.testing.assert_allclose(srefm.temp, shism.temp, rtol=2e-2)
            with pytest.raises(AssertionError):
                # This clearly is not scaled
                np.testing.assert_allclose(srefm.prcp, shism.prcp, rtol=0.5)

        # Replaced
        tasks.historical_delta_method(gdir,
                                      replace_with_ref_data=True,
                                      delete_input_files=False,
                                      ref_filesuffix='ERA5',
                                      hist_filesuffix='CERA',
                                      out_filesuffix='CERA_repl')

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
        files = []
        ref_hgts = []
        for base in exps:
            cfg.PARAMS['baseline_climate'] = base
            tasks.process_climate_data(gdir, output_filesuffix=base)
            files.append(gdir.get_filepath('climate_historical',
                                           filesuffix=base))
            with xr.open_dataset(files[-1]) as ds:
                ref_hgts.append(ds.ref_hgt)

        # TEMP
        with xr.open_mfdataset(files, concat_dim=exps) as ds:
            dft = ds.temp.to_dataframe().unstack().T
            dft.index = dft.index.levels[1]

        # Common period
        dfy = dft.resample('AS').mean().dropna().iloc[1:]
        dfm = dft.groupby(dft.index.month).mean()
        assert dfy.corr().min().min() > 0.6
        assert dfm.corr().min().min() > 0.97
        dfavg = dfy.describe()

        # Correct for hgt
        ref_h = ref_hgts[0]
        for h, d in zip(ref_hgts, exps):
            dfy[d] = dfy[d] - 0.0065 * (ref_h - h)
            dfm[d] = dfm[d] - 0.0065 * (ref_h - h)
        dfavg_cor = dfy.describe()

        # After correction less spread
        assert dfavg_cor.loc['mean'].std() < 0.6 * dfavg.loc['mean'].std()
        assert dfavg_cor.loc['mean'].std() < 2

        # PRECIP
        with xr.open_mfdataset(files, concat_dim=exps) as ds:
            dft = ds.prcp.to_dataframe().unstack().T
            dft.index = dft.index.levels[1]

        # Common period
        dfy = dft.resample('AS').mean().dropna().iloc[1:] * 12
        dfm = dft.groupby(dft.index.month).mean()
        assert dfy.corr().min().min() > 0.5
        assert dfm.corr().min().min() > 0.85
        dfavg = dfy.describe()
        assert dfavg.loc['mean'].std() / dfavg.loc['mean'].mean() < 0.15  # %
