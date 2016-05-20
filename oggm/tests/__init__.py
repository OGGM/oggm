import six
import osgeo.gdal
import os
import platform
import unittest
import logging
import sys
import numpy as np

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


def requires_py3(test):
    # Test decorator
    msg = "requires python3"
    return unittest.skip(msg)(test) if six.PY2 else test


def is_slow(test):
    # Test decorator
    msg = "requires explicit environment for slow tests"
    return test if RUN_SLOW_TESTS else unittest.skip(msg)(test)


def is_download(test):
    # Test decorator
    msg = "requires explicit environment for download tests"
    return test if RUN_DOWNLOAD_TESTS else unittest.skip(msg)(test)


# the code below is copy/pasted from xarray
# TODO: go back to xarray when https://github.com/pydata/xarray/issues/754
def assertEqual(a1, a2):
    assert a1 == a2 or (a1 != a1 and a2 != a2)


def decode_string_data(data):
    if data.dtype.kind == 'S':
        return np.core.defchararray.decode(data, 'utf-8', 'replace')


def data_allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08):
    from xarray.core import ops

    if any(arr.dtype.kind == 'S' for arr in [arr1, arr2]):
        arr1 = decode_string_data(arr1)
        arr2 = decode_string_data(arr2)
    exact_dtypes = ['M', 'm', 'O', 'U']
    if any(arr.dtype.kind in exact_dtypes for arr in [arr1, arr2]):
        return ops.array_equiv(arr1, arr2)
    else:
        return ops.allclose_or_equiv(arr1, arr2, rtol=rtol, atol=atol)


def assertVariableAllClose(v1, v2, rtol=1e-05, atol=1e-08):
    assertEqual(v1.dims, v2.dims)
    allclose = data_allclose_or_equiv(
        v1.values, v2.values, rtol=rtol, atol=atol)
    assert allclose, (v1.values, v2.values)


def assertDatasetAllClose(d1, d2, rtol=1e-05, atol=1e-08):
    assertEqual(sorted(d1, key=str), sorted(d2, key=str))
    for k in d1:
        v1 = d1.variables[k]
        v2 = d2.variables[k]
        assertVariableAllClose(v1, v2, rtol=rtol, atol=atol)
