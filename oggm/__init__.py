""" OGGM package.

Copyright: OGGM developers, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division

from os import path
from os import makedirs

# Path to the cache directory
CACHE_DIR = path.join(path.expanduser('~'), '.oggm')
if not path.exists(CACHE_DIR):
    makedirs(CACHE_DIR)  # pragma: no cover
