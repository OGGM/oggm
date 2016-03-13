import osgeo.gdal
import os
import platform
import unittest
import logging
import sys

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# TODO: latest gdal seems to modify the results
HAS_NEW_GDAL = False
if osgeo.gdal.__version__ >= '1.11':
    HAS_NEW_GDAL = True

ON_TRAVIS = False
RUN_SLOW_TESTS = False
RUN_DOWNLOAD_TESTS = False
if os.environ.get('TRAVIS') is not None:
    RUN_SLOW_TESTS = True
    ON_TRAVIS = True
    RUN_DOWNLOAD_TESTS = False  # security
if os.environ.get('OGGM_SLOW_TESTS') is not None:
    RUN_SLOW_TESTS = True
if os.environ.get('OGGM_DOWNLOAD_TESTS') is not None:
    RUN_DOWNLOAD_TESTS = True

# TODO: conda builds on python 3.4 have EVEN MORE issues
# https://ci.appveyor.com/project/fmaussion/oggm/build/job/k9qvsxp4k3h3l2y9
ON_WINDOWS_PY3_CONDA = False
if (platform.system() == 'Windows') and (sys.version_info >= (3, 0)):
    ON_WINDOWS_PY3_CONDA = True


def requires_working_conda(test):
    # Test decorator
    msg = "requires a conda build which works like the others"
    return unittest.skip(msg)(test) if ON_WINDOWS_PY3_CONDA else test


def is_slow(test):
    # Test decorator
    msg = "requires explicit environment for slow tests"
    return test if RUN_SLOW_TESTS else unittest.skip(msg)(test)


def is_download(test):
    # Test decorator
    msg = "requires explicit environment for download tests"
    return test if RUN_DOWNLOAD_TESTS else unittest.skip(msg)(test)