""" OGGM package.

Copyright: OGGM developers, 2014-2017

License: GPLv3+
"""
from __future__ import absolute_import, division
import logging

try:
    from .version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('oggm is not properly installed. If you are running '
                      'from the source directory, please instead create a '
                      'new virtual environment (using conda or virtualenv) '
                      'and  then install it in-place by running: '
                      'pip install -e .')

# Fiona, rasterio and shapely are spammers
logging.getLogger("Fiona").setLevel(logging.WARNING)
logging.getLogger("shapely").setLevel(logging.WARNING)
logging.getLogger("rasterio").setLevel(logging.WARNING)

# Basic config
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

try:
    from oggm.mpi import _init_oggm_mpi
    _init_oggm_mpi()
except ImportError:
    pass

# API
from oggm.utils import GlacierDirectory, entity_task, divide_task, global_task
from oggm.core.preprocessing.centerlines import Centerline
