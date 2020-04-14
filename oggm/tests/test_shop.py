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
from oggm import cfg, tasks
from oggm.tests.funcs import patch_url_retrieve_github


pytestmark = pytest.mark.test_env("utils")
_url_retrieve = None

DO_PLOT = False


@pytest.fixture(autouse=True, scope='module')
def init_url_retrieve(request):
    # Called at module set-up and shut-down
    # We patch OGGM's download functions to make sure that we don't
    # access data sources we shouldnt touch from within the tests
    request.module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github
    # This below is a shut-down
    yield
    oggm.utils._downloads.oggm_urlretrieve = request.module._url_retrieve


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
