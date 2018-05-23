""" OGGM package.

Copyright: OGGM developers, 2014-2017

License: GPLv3+
"""
import logging

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

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
from oggm.utils import (GlacierDirectory, entity_task, global_task,
                        gettempdir, get_demo_file)
from oggm.core.centerlines import Centerline
from oggm.core.flowline import Flowline

# Make sure we have the sample data at import
get_demo_file('')
