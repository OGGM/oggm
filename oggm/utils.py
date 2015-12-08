"""Some useful functions

Copyright: OGGM developers, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division
from six.moves.urllib.request import urlretrieve

# Builtins
import os
import shutil
import zipfile
import sys
from functools import partial

# External libs
import numpy as np
import netCDF4
import scipy.stats as stats
from joblib import Memory

# Locals
from oggm import cache_dir

# Globals
gh_zip = 'https://github.com/OGGM/oggm-sample-data/archive/master.zip'

# Joblib
memory = Memory(cachedir=cache_dir, verbose=0)

gaussian_kernel = dict()
gaussian_kernel[9] = np.array([1.33830625e-04, 4.43186162e-03,
                               5.39911274e-02, 2.41971446e-01,
                               3.98943469e-01, 2.41971446e-01,
                               5.39911274e-02, 4.43186162e-03,
                               1.33830625e-04])
gaussian_kernel[7] = np.array([1.78435052e-04, 1.51942011e-02,
                               2.18673667e-01, 5.31907394e-01,
                               2.18673667e-01, 1.51942011e-02,
                               1.78435052e-04])
gaussian_kernel[5] = np.array([2.63865083e-04, 1.06450772e-01,
                               7.86570726e-01, 1.06450772e-01,
                               2.63865083e-04])

tuple2int = partial(np.array, dtype=np.int64)


def empty_cache():  # pragma: no cover
    """Empty oggm's cache directory."""

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)


def _download_demo_files():
    """Checks if the demo data is already on the cache and downloads it.

    TODO: Currently there's no check to see of the server file has changed
    this is bad. In the mean time, with empty_cache() you can ensure that the
    files are up-to-date.
    """

    ofile = os.path.join(cache_dir, 'oggm-sample-data.zip')
    odir = os.path.join(cache_dir)
    if not os.path.exists(ofile):  # pragma: no cover
        urlretrieve(gh_zip, ofile)
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(odir)

    out = dict()
    sdir = os.path.join(cache_dir, 'oggm-sample-data-master')
    for root, directories, filenames in os.walk(sdir):
        for filename in filenames:
            out[filename] = os.path.join(root, filename)
    return out


def get_demo_file(fname):
    """Returns the path to the desired demo file."""

    d = _download_demo_files()
    if fname in d:
        return d[fname]
    else:
        return None


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Credits: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between one point
    on the earth and an array of points (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius of earth in meters
    return c * r


def interp_nans(array):
    """Interpolate NaNs using np.interp.

    np.interp is reasonable in that it does not extrapolate, it replaces
    NaNs at the bounds with the closest valid value.
    """

    _tmp = array.copy()
    nans, x = np.isnan(array), lambda z: z.nonzero()[0]
    _tmp[nans] = np.interp(x(nans), x(~nans), array[~nans])

    return _tmp


def md(ref, data, axis=None):
    """Mean Deviation."""
    return np.mean(data-ref, axis=axis)


def mad(ref, data, axis=None):
    """Mean Absolute Deviation."""
    return np.mean(np.abs(data-ref), axis=axis)


def rmsd(ref, data, axis=None):
    """Root Mean Square Deviation."""
    return np.sqrt(np.mean((ref-data)**2, axis=axis))


def rel_err(ref, data):
    """Relative error. Ref should be non-zero"""
    return (data - ref) / ref


def corrcoef(ref, data):
    """Peason correlation coefficient."""
    return np.corrcoef(ref, data)[0, 1]


def nicenumber(number, binsize, lower=False):
    """Returns the next higher or lower "nice number", given by binsize.

    Examples:
    ---------
    >>> nicenumber(12, 10)
    20
    >>> nicenumber(19, 50)
    50
    >>> nicenumber(51, 50)
    100
    >>> nicenumber(51, 50, lower=True)
    50
    """

    e, _ = divmod(number, binsize)
    if lower:
        return e * binsize
    else:
        return (e+1) * binsize

@memory.cache
def joblib_read_climate(ncpath, ilon, ilat, default_grad, minmax_grad,
                        prcp_scaling_factor):
    """Prevent to re-compute a timeserie if it was done before.

    TODO: dirty solution, should be replaced by proper input.
    """

    # read the file and data
    nc = netCDF4.Dataset(ncpath, mode='r')
    temp = nc.variables['temp']
    prcp = nc.variables['prcp']
    hgt = nc.variables['hgt']

    igrad = np.zeros(len(nc.dimensions['time'])) + default_grad

    ttemp = temp[:, ilat-1:ilat+2, ilon-1:ilon+2]
    itemp = ttemp[:, 1, 1]
    thgt = hgt[ilat-1:ilat+2, ilon-1:ilon+2]
    ihgt = hgt[1, 1]
    thgt = thgt.flatten()
    iprcp = prcp[:, ilat, ilon] * prcp_scaling_factor

    # Now the gradient
    for t, loct in enumerate(ttemp):
        slope, _, _, p_val, _ = stats.linregress(thgt,
                                                 loct.flatten())
        igrad[t] = slope if (p_val < 0.01) else default_grad

    # dont exagerate too much
    igrad = np.clip(igrad, minmax_grad[0], minmax_grad[1])

    return iprcp, itemp, igrad, ihgt