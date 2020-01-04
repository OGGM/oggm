import os
import shutil

import numpy as np
import pytest
import shapely.geometry as shpg


from oggm import cfg
from oggm.core import flowline
from oggm.tests.funcs import get_ident, init_hef, init_columbia
from oggm.utils import mkdir


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


@pytest.fixture(scope='class')
def class_test_dir(request, test_dir):
    clsdir = os.path.join(test_dir, request.cls.__name__)
    mkdir(clsdir, reset=True)
    yield clsdir
    # teardown
    if os.path.exists(clsdir):
        shutil.rmtree(clsdir)


@pytest.fixture(scope='class')
def hef_gdir(request, test_dir):
    border = request.module.DOM_BORDER if request.module.DOM_BORDER else 40
    return init_hef(border=border)
