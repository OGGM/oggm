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

# External libs

# Locals
from oggm import cache_dir

gh_zip = 'https://github.com/OGGM/oggm-sample-data/archive/master.zip'


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
    sdir = os.path.join(cache_dir, 'oggm-sample-data-master', 'test-files')
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