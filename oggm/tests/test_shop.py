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
from oggm.shop import its_live
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
        its_live.velocity_to_gdir(gdir)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            mask = ds.glacier_mask
            vx = ds.obs_icevel_x.where(mask).data
            vy = ds.obs_icevel_y.where(mask).data

        assert np.nanmax(np.sqrt(vx**2 + vy**2)) > 2900
        assert np.nanmin(np.sqrt(vx**2 + vy**2)) < 2

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
