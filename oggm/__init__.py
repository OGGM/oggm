""" OGGM package.

Copyright: OGGM developers, 2014-2017

License: GPLv3+
"""
import logging

try:
    from .version import version as __version__
except ImportError:
    raise ImportError('oggm is not properly installed. If you are running '
                      'from the source directory, please instead create a '
                      'new virtual environment (using conda or virtualenv) '
                      'and  then install it in-place by running: '
                      'pip install -e .')

# Spammers
logging.getLogger("Fiona").setLevel(logging.CRITICAL)
logging.getLogger("shapely").setLevel(logging.CRITICAL)
logging.getLogger("rasterio").setLevel(logging.CRITICAL)

# Basic config
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

try:
    from oggm.mpi import _init_oggm_mpi
    _init_oggm_mpi()
except ImportError:
    pass

# API
from oggm.utils import GlacierDirectory, entity_task, global_task
from oggm.core.centerlines import Centerline
from oggm.core.flowline import Flowline
