"""Pytest fixtures to be used in other test modules"""
import copy
import os
import shutil
import logging
import getpass
from functools import wraps

import numpy as np
import pytest
import shapely.geometry as shpg
import matplotlib.pyplot as plt

from oggm.shop import cru, histalp, ecmwf, w5e5
from oggm import cfg, tasks
from oggm.core import flowline
from oggm.tests.funcs import init_hef, get_test_dir
from oggm import utils
from oggm.utils import mkdir, _downloads
from oggm.utils import oggm_urlretrieve
from oggm.tests import HAS_MPL_FOR_TESTS, HAS_INTERNET
from oggm.workflow import reset_multiprocessing

logger = logging.getLogger(__name__)


def pytest_configure(config):
    for marker in ["slow", "download", "creds", "internet", "test_env",
                   "graphic", "static_map"]:
        config.addinivalue_line("markers", marker)
    if config.pluginmanager.hasplugin('xdist'):
        try:
            from ilock import ILock
            utils.lock = ILock("oggm_xdist_download_lock_" + getpass.getuser())
            logger.info("ilock locking setup successfully for xdist tests")
        except BaseException:
            logger.warning("could not setup ilock locking for distributed "
                           "tests")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Run slow tests")
    parser.addoption("--run-download", action="store_true", default=False,
                     help="Run download tests")
    parser.addoption("--run-creds", action="store_true", default=False,
                     help="Run download tests requiring credentials")
    parser.addoption("--run-test-env", metavar="ENVNAME", default="",
                     help="Run only specified test env")
    parser.addoption("--no-run-internet", action="store_true", default=False,
                     help="Don't run any tests accessing the internet")


def pytest_collection_modifyitems(config, items):
    use_internet = HAS_INTERNET and not config.getoption("--no-run-internet")
    skip_slow = not config.getoption("--run-slow")
    skip_download = not use_internet or not config.getoption("--run-download")
    skip_cred = skip_download or not config.getoption("--run-creds")
    run_test_env = config.getoption("--run-test-env")
    skip_static = (("STATIC_MAP_API_KEY" not in os.environ) or
                   (not os.environ.get("STATIC_MAP_API_KEY")))

    slow_marker = pytest.mark.skip(reason="need --run-slow option to run")
    download_marker = pytest.mark.skip(reason="need --run-download option to "
                                              "run, internet access is "
                                              "required")
    cred_marker = pytest.mark.skip(reason="need --run-creds option to run, "
                                          "internet access is required")
    internet_marker = pytest.mark.skip(reason="internet access is required")
    static_marker = pytest.mark.skip(reason="requires STATIC_MAP_API_KEY env variable")
    test_env_marker = pytest.mark.skip(reason="only test_env=%s tests are run"
                                              % run_test_env)
    graphic_marker = pytest.mark.skip(reason="requires mpl V1.5+ and "
                                             "pytest-mpl")

    for item in items:
        if skip_slow and "slow" in item.keywords:
            item.add_marker(slow_marker)
        if skip_download and "download" in item.keywords:
            item.add_marker(download_marker)
        if skip_cred and "creds" in item.keywords:
            item.add_marker(cred_marker)
        if skip_static and "static_map" in item.keywords:
            item.add_marker(static_marker)
        if not use_internet and "internet" in item.keywords:
            item.add_marker(internet_marker)

        if run_test_env:
            test_env = item.get_closest_marker("test_env")
            if not test_env or test_env.args[0] != run_test_env:
                item.add_marker(test_env_marker)

        if "graphic" in item.keywords:
            def wrap_graphic_test(test):
                @wraps(test)
                def test_wrapper(*args, **kwargs):
                    try:
                        return test(*args, **kwargs)
                    finally:
                        plt.close()
                return test_wrapper
            item.obj = wrap_graphic_test(item.obj)

            if not HAS_MPL_FOR_TESTS:
                item.add_marker(graphic_marker)


@pytest.fixture(scope='function', autouse=True)
def restore_oggm_cfg():
    """Ensures a test cannot pollute cfg for other tests running after it"""
    old_cfg = cfg.pack_config()
    reset_multiprocessing()
    yield
    cfg.unpack_config(old_cfg)


@pytest.fixture(scope='class', autouse=True)
def restore_oggm_class_cfg():
    """Ensures a test-class cannot pollute cfg for other tests running after it"""
    old_cfg = cfg.pack_config()
    yield
    cfg.unpack_config(old_cfg)


@pytest.fixture(autouse=True)
def patch_data_urls(monkeypatch):
    """This makes sure we never download the big files with our tests"""
    url = 'https://cluster.klima.uni-bremen.de/~oggm/test_climate/'
    monkeypatch.setattr(w5e5, 'GSWP3_W5E5_SERVER', url)
    monkeypatch.setattr(cru, 'CRU_SERVER', url + 'cru/')
    monkeypatch.setattr(cru, 'CRU_BASE', 'cru_ts3.23.1901.2014.{}.dat.nc')
    monkeypatch.setattr(histalp, 'HISTALP_SERVER', url + 'histalp/')
    monkeypatch.setattr(ecmwf, 'ECMWF_SERVER', url)

    basenames = {
        'ERA5': {
            'inv': 'era5/monthly/v1.2/flattened/era5_glacier_invariant_flat_v2026.01.14.nc',
            'pre': 'era5/monthly/v1.2/flattened/era5_tp_global_monthly_1940_2025_flat_glaciers_v2026.01.14.nc',
            'tmp': 'era5/monthly/v1.2/flattened/era5_t2m_global_monthly_1940_2025_flat_glaciers_v2026.01.14.nc'
        },
        'ERA5L': {
            'inv': 'era5-land/monthly/v1.0/era5_land_invariant_flat.nc',
            'pre': 'era5-land/monthly/v1.0/era5_land_monthly_prcp_1981-2018_flat'
                   '.nc',
            'tmp': 'era5-land/monthly/v1.0/era5_land_monthly_t2m_1981-2018_flat.nc'
        },
        'CERA': {
            'inv': 'cera-20c/monthly/v1.0/cera-20c_invariant.nc',
            'pre': 'cera-20c/monthly/v1.0/cera-20c_pcp_1901-2010.nc',
            'tmp': 'cera-20c/monthly/v1.0/cera-20c_t2m_1901-2010.nc'
        },
        'ERA5dr': {
            'inv': 'era5/monthly/vdr/ERA5_geopotential_monthly.nc',
            'lapserates': 'era5/monthly/vdr/ERA5_lapserates_monthly.nc',
            'tmp': 'era5/monthly/vdr/ERA5_temp_monthly.nc',
            'tempstd': 'era5/monthly/vdr/ERA5_tempstd_monthly.nc',
            'pre': 'era5/monthly/vdr/ERA5_totalprecip_monthly.nc',
        }
    }
    monkeypatch.setattr(ecmwf, 'BASENAMES', basenames)


def secure_url_retrieve(url, *args, **kwargs):
    """A simple patch to OGGM's download function to make sure we don't
    download elsewhere than expected."""

    # We added a few extra glaciers recently - this really needs to be
    # handled better
    base_extra_l3 = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                     'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/RGI62/'
                     'b_160/L3/')

    base_extra_v14 = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                      'oggm_v1.4/L1-L2_files/elev_bands/RGI62/b_040/{}/'
                      'RGI60-15/RGI60-15.13.tar')

    base_extra_v14l3 = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                        'oggm_v1.4/L3-L5_files/CRU/elev_bands/qc3/pcp2.5/'
                        'no_match/RGI62/b_040/{}/RGI60-15/RGI60-15.13.tar')

    assert ('github' in url or
            'cluster.klima.uni-bremen.de/~oggm/ref_mb_params/' in url or
            'cluster.klima.uni-bremen.de/~oggm/test_gdirs/' in url or
            'cluster.klima.uni-bremen.de/~oggm/demo_gdirs/' in url or
            'cluster.klima.uni-bremen.de/~oggm/test_climate/' in url or
            'cluster.klima.uni-bremen.de/~oggm/test_files/' in url or
            'klima.uni-bremen.de/~oggm/climate/cru/cru_cl2.nc.zip' in url or
            'klima.uni-bremen.de/~oggm/geodetic_ref_mb' in url or
            'RGI2000-v7.0-regions.zip' in url or
            base_extra_v14.format('L1') in url or
            base_extra_v14.format('L2') in url or
            base_extra_v14l3.format('L3') in url or
            base_extra_l3 in url
            )
    return oggm_urlretrieve(url, *args, **kwargs)


@pytest.fixture(autouse=True)
def patch_url_retrieve(monkeypatch):
    monkeypatch.setattr(_downloads, 'oggm_urlretrieve', secure_url_retrieve)


@pytest.fixture()
def dummy_constant_bed():
    dx = 1.

    hmax = 3000.
    hmin = 1000.
    nx = 200
    map_dx = 100.
    widths = 3.

    surface_h = np.linspace(hmax, hmin, nx)
    bed_h = surface_h
    widths = surface_h * 0. + widths
    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    return [flowline.RectangularBedFlowline(line, dx, map_dx, surface_h,
                                            bed_h, widths)]


@pytest.fixture(scope='session')
def test_dir():
    """ Provides a reference to the test directory for the entire test session.

        Named after the current git revision.
        As a session-scoped fixture, this will only be created once and
        then injected to each test that depends on it.
    """
    return get_test_dir()


def _setup_case_dir(call, test_dir):
    casedir = os.path.join(test_dir, call.__name__)
    mkdir(casedir, reset=True)
    return casedir


def _teardown_case_dir(casedir):
    if os.path.exists(casedir):
        shutil.rmtree(casedir)


@pytest.fixture(scope='function')
def case_dir(request, test_dir):
    """ Provides a unique directory for the current test function, a child of
        the session test directory (test_dir > case_dir). Named after the
        current test function.
        As a function-scoped fixture, a new directory is created for
        each function that uses this and then cleaned up when the case
        completes.
    """
    cd = _setup_case_dir(request.function, test_dir)
    yield cd
    _teardown_case_dir(cd)


@pytest.fixture(scope='class')
def class_case_dir(request, test_dir):
    """ Provides a unique directory for the current test class, a child of
        the session test directory (test_dir > class_case_dir). Named after
        the current test class.
        As a class-scoped fixture, a class directory is created once for
        the current class and used by each test inside it. It is cleaned
        up when the all the cases in the class complete.
    """
    cd = _setup_case_dir(request.cls, test_dir)
    yield cd
    _teardown_case_dir(cd)


@pytest.fixture(scope='module')
def hef_gdir_base(request, test_dir):
    """ Provides an initialized Hintereisferner glacier directory.
        As a module fixture, the initialization is run only once per test
        module that uses it.
        IMPORTANT: To preserve a constant starting condition, hef_gdir_base
        should almost never be directly injected into a test case. Test cases
        should use the below hef_gdir fixture to provide a directory that has
        been copied into an ephemeral case directory.
    """
    try:
        module = request.module
        border = module.DOM_BORDER if module.DOM_BORDER is not None else 40
        return init_hef(border=border)
    except AttributeError:
        return init_hef()


@pytest.fixture(scope='module')
def hef_copy_gdir_base(request, test_dir):
    """same as hef_gdir but with another RGI ID (for workflow testing)
    """
    try:
        module = request.module
        border = module.DOM_BORDER if module.DOM_BORDER is not None else 40
        return init_hef(border=border, rgi_id='RGI50-11.99999')
    except AttributeError:
        return init_hef(rgi_id='RGI50-11.99999')


@pytest.fixture(scope='module')
def hef_elev_gdir_base(request, test_dir):
    """ Same as hef_gdir, but with elevation bands instead of centerlines.
        Further, also a different RGI ID so that a unique directory is created
        (as the RGI ID defines the final folder name) and to have no
        collisions with hef_gdir
    """
    try:
        module = request.module
        border = module.DOM_BORDER if module.DOM_BORDER is not None else 40
        return init_hef(border=border, flowline_type='elevation_bands',
                        rgi_id='RGI50-11.99998')
    except AttributeError:
        return init_hef(flowline_type='elevation_bands',
                        rgi_id='RGI50-11.99998')


@pytest.fixture(scope='class')
def hef_gdir(hef_gdir_base, class_case_dir):
    """ Provides a copy of the base Hintereisenferner glacier directory in
        a case directory specific to the current test class. All cases in
        the test class will use the same copy of this glacier directory.
    """
    return tasks.copy_to_basedir(hef_gdir_base, base_dir=class_case_dir,
                                 setup='all')


@pytest.fixture(scope='class')
def hef_copy_gdir(hef_copy_gdir_base, class_case_dir):
    """Same as hef_gdir but with another RGI ID (for testing)
    """
    return tasks.copy_to_basedir(hef_copy_gdir_base, base_dir=class_case_dir,
                                 setup='all')


@pytest.fixture(scope='class')
def hef_elev_gdir(hef_elev_gdir_base, class_case_dir):
    """ Same as hef_gdir but with elevation bands instead of centerlines.
    """
    return tasks.copy_to_basedir(hef_elev_gdir_base, base_dir=class_case_dir,
                                 setup='all')
