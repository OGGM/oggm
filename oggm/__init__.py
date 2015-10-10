"""
OGGM
====

Copyright: OGGM developers, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division, unicode_literals

from os import path
from os import makedirs

# Path to the cache directory
cache_dir = path.join(path.expanduser('~'), '.oggm')
if not path.exists(cache_dir):
    makedirs(cache_dir)  # pragma: no cover