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
from urllib.request import urlopen

from oggm import cfg
from oggm.utils import SAMPLE_DATA_COMMIT

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# Some logic to see which environment we are running on

# Matplotlib version changes plots, too
HAS_MPL_FOR_TESTS = False
if LooseVersion(matplotlib.__version__) >= LooseVersion('2'):
    HAS_MPL_FOR_TESTS = True
    BASELINE_DIR = os.path.join(cfg.CACHE_DIR,
                                'oggm-sample-data-%s' % SAMPLE_DATA_COMMIT,
                                'baseline_images')
    ftver = LooseVersion(matplotlib.ft2font.__freetype_version__)
    if ftver >= LooseVersion('2.8.0'):
        BASELINE_DIR = os.path.join(BASELINE_DIR, 'freetype_28')
    else:
        BASELINE_DIR = os.path.join(BASELINE_DIR, 'freetype_old')


# Some control on which tests to run (useful to avoid too long tests)
# defaults everywhere else than travis
RUN_PREPRO_TESTS = True
RUN_NUMERIC_TESTS = True
RUN_MODEL_TESTS = True
RUN_WORKFLOW_TESTS = True
RUN_GRAPHIC_TESTS = True
RUN_BENCHMARK_TESTS = True
if os.environ.get('TRAVIS') is not None:
    env = os.environ.get('OGGM_TEST_ENV')
    if env == 'prepro':
        RUN_PREPRO_TESTS = True
        RUN_NUMERIC_TESTS = False
        RUN_MODEL_TESTS = False
        RUN_WORKFLOW_TESTS = False
        RUN_GRAPHIC_TESTS = False
        RUN_BENCHMARK_TESTS = False
    if env == 'numerics':
        RUN_PREPRO_TESTS = False
        RUN_NUMERIC_TESTS = True
        RUN_MODEL_TESTS = False
        RUN_WORKFLOW_TESTS = False
        RUN_GRAPHIC_TESTS = False
        RUN_BENCHMARK_TESTS = False
    if env == 'models':
        RUN_PREPRO_TESTS = False
        RUN_NUMERIC_TESTS = False
        RUN_MODEL_TESTS = True
        RUN_WORKFLOW_TESTS = False
        RUN_GRAPHIC_TESTS = False
        RUN_BENCHMARK_TESTS = False
    if env == 'workflow':
        RUN_PREPRO_TESTS = False
        RUN_NUMERIC_TESTS = False
        RUN_MODEL_TESTS = False
        RUN_WORKFLOW_TESTS = True
        RUN_GRAPHIC_TESTS = False
        RUN_BENCHMARK_TESTS = False
    if env == 'graphics':
        RUN_PREPRO_TESTS = False
        RUN_NUMERIC_TESTS = False
        RUN_MODEL_TESTS = False
        RUN_WORKFLOW_TESTS = False
        RUN_GRAPHIC_TESTS = True
        RUN_BENCHMARK_TESTS = False
    if env == 'benchmark':
        RUN_PREPRO_TESTS = False
        RUN_NUMERIC_TESTS = False
        RUN_MODEL_TESTS = False
        RUN_WORKFLOW_TESTS = False
        RUN_GRAPHIC_TESTS = False
        RUN_BENCHMARK_TESTS = True

# quick n dirty method to see if internet is on
try:
    _ = urlopen('http://www.google.com', timeout=1)
    HAS_INTERNET = True
except:
    HAS_INTERNET = False

