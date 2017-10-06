import logging
import os
import socket
import sys
import unittest
import functools
from distutils.version import LooseVersion

import matplotlib.ft2font
import matplotlib.pyplot as plt
import osgeo.gdal
import six
from six.moves.urllib.request import urlopen

from oggm import cfg

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# Some logic to see which environment we are running on

# GDAL version changes the way interpolation is made (sigh...)
HAS_NEW_GDAL = False
if osgeo.gdal.__version__ >= '1.11':
    HAS_NEW_GDAL = True

# Matplotlib version changes plots, too
HAS_MPL_FOR_TESTS = False
if LooseVersion(matplotlib.__version__) >= LooseVersion('2'):
    HAS_MPL_FOR_TESTS = True
    BASELINE_DIR = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master',
                                'baseline_images')
    ftver = LooseVersion(matplotlib.ft2font.__freetype_version__)
    if ftver >= LooseVersion('2.8.0'):
        BASELINE_DIR = os.path.join(BASELINE_DIR, 'alt')
    else:
        BASELINE_DIR = os.path.join(BASELINE_DIR, '2.0.x')


# Some control on which tests to run (useful to avoid too long tests)
# defaults everywhere else than travis
ON_AWS = False
ON_TRAVIS = False
RUN_SLOW_TESTS = False
RUN_DOWNLOAD_TESTS = False
RUN_PREPRO_TESTS = True
RUN_NUMERIC_TESTS = True
RUN_MODEL_TESTS = True
RUN_WORKFLOW_TESTS = True
RUN_GRAPHIC_TESTS = True
RUN_PERFORMANCE_TESTS = False
if os.environ.get('TRAVIS') is not None:
    # specific to travis to reduce global test time
    ON_TRAVIS = True
    RUN_DOWNLOAD_TESTS = False
    matplotlib.use('Agg')

    if sys.version_info < (3, 5):
        # Minimal tests
        RUN_SLOW_TESTS = False
        RUN_PREPRO_TESTS = True
        RUN_NUMERIC_TESTS = True
        RUN_MODEL_TESTS = True
        RUN_WORKFLOW_TESTS = True
        RUN_GRAPHIC_TESTS = True
    else:
        # distribute the tests
        RUN_SLOW_TESTS = True
        env = os.environ.get('OGGM_TEST_ENV')
        if env == 'prepro':
            RUN_PREPRO_TESTS = True
            RUN_NUMERIC_TESTS = False
            RUN_MODEL_TESTS = False
            RUN_WORKFLOW_TESTS = False
            RUN_GRAPHIC_TESTS = False
        if env == 'numerics':
            RUN_PREPRO_TESTS = False
            RUN_NUMERIC_TESTS = True
            RUN_MODEL_TESTS = False
            RUN_WORKFLOW_TESTS = False
            RUN_GRAPHIC_TESTS = False
        if env == 'models':
            RUN_PREPRO_TESTS = False
            RUN_NUMERIC_TESTS = False
            RUN_MODEL_TESTS = True
            RUN_WORKFLOW_TESTS = False
            RUN_GRAPHIC_TESTS = False
        if env == 'workflow':
            RUN_PREPRO_TESTS = False
            RUN_NUMERIC_TESTS = False
            RUN_MODEL_TESTS = False
            RUN_WORKFLOW_TESTS = True
            RUN_GRAPHIC_TESTS = False
        if env == 'graphics':
            RUN_PREPRO_TESTS = False
            RUN_NUMERIC_TESTS = False
            RUN_MODEL_TESTS = False
            RUN_WORKFLOW_TESTS = False
            RUN_GRAPHIC_TESTS = True
elif 'ip-' in socket.gethostname():
    # we are on AWS (hacky way)
    ON_AWS = True
    RUN_SLOW_TESTS = True
    matplotlib.use('Agg')

# give user some control
if os.environ.get('OGGM_SLOW_TESTS') is not None:
    RUN_SLOW_TESTS = True
if os.environ.get('OGGM_DOWNLOAD_TESTS') is not None:
    RUN_DOWNLOAD_TESTS = True

# quick n dirty method to see if internet is on
try:
    _ = urlopen('http://www.google.com', timeout=1)
    HAS_INTERNET = True
except:
    HAS_INTERNET = False


def requires_internet(test):
    # Test decorator
    msg = 'requires internet'
    return test if HAS_INTERNET else unittest.skip(msg)(test)


def requires_py3(test):
    # Test decorator
    msg = "requires python3"
    return unittest.skip(msg)(test) if six.PY2 else test


def is_graphic_test(test):
    # Decorator

    @functools.wraps(test)
    def new_test(*args, **kwargs):
        try:
            return test(*args, **kwargs)
        finally:
            plt.close()

    msg = 'requires mpl V1.5+ and matplotlib.testing.decorators'
    return new_test if HAS_MPL_FOR_TESTS else unittest.skip(msg)(test)


def is_slow(test):
    # Test decorator
    msg = "requires explicit environment for slow tests"
    return test if RUN_SLOW_TESTS else unittest.skip(msg)(test)


def is_download(test):
    # Test decorator
    msg = "requires explicit environment for download tests"
    return test if RUN_DOWNLOAD_TESTS else unittest.skip(msg)(test)


def is_performance_test(test):
    # Test decorator
    msg = "requires explicit environment for performance tests"
    return test if RUN_PERFORMANCE_TESTS else unittest.skip(msg)(test)
