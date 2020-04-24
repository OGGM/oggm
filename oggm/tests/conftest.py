"""Pytest fixtures to be used in other test modules"""

import os
import shutil

import numpy as np
import pytest
import shapely.geometry as shpg

from oggm.shop import cru, histalp
from oggm import cfg, tasks
from oggm.core import flowline
from oggm.tests.funcs import get_ident, init_hef
from oggm.utils import mkdir, _downloads
from oggm.utils import oggm_urlretrieve


@pytest.fixture(autouse=True)
def patch_data_urls(monkeypatch):
    """This makes sure we never download the big files with our tests"""
    url = 'https://cluster.klima.uni-bremen.de/~oggm/test_climate/'
    monkeypatch.setattr(cru, 'CRU_SERVER', url + 'cru/')
    monkeypatch.setattr(cru, 'CRU_BASE', 'cru_ts3.23.1901.2014.{}.dat.nc')
    monkeypatch.setattr(histalp, 'HISTALP_SERVER', url + 'histalp/')


def secure_url_retrieve(url, *args, **kwargs):
    """A simple patch to OGGM's download function to make sure we don't
    download elsewhere than expected."""

    assert ('github' in url or
            'cluster.klima.uni-bremen.de/~oggm/test_gdirs/' in url or
            'cluster.klima.uni-bremen.de/~oggm/demo_gdirs/' in url or
            'cluster.klima.uni-bremen.de/~oggm/test_climate/' in url or
            'klima.uni-bremen.de/~oggm/climate/cru/cru_cl2.nc.zip' in url
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
    s = get_ident()
    out = os.path.join(cfg.PATHS['test_dir'], s)
    if 'PYTEST_XDIST_WORKER' in os.environ:
        out = os.path.join(out, os.environ.get('PYTEST_XDIST_WORKER'))
    mkdir(out)

    # If new ident, remove all other dirs so spare space
    for d in os.listdir(cfg.PATHS['test_dir']):
        if d and d != s:
            shutil.rmtree(os.path.join(cfg.PATHS['test_dir'], d))
    return out


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


@pytest.fixture(scope='class')
def hef_gdir(hef_gdir_base, class_case_dir):
    """ Provides a copy of the base Hintereisenferner glacier directory in
        a case directory specific to the current test class. All cases in
        the test class will use the same copy of this glacier directory.
    """
    return tasks.copy_to_basedir(hef_gdir_base, base_dir=class_case_dir,
                                 setup='all')
