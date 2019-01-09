""" OGGM package.

Copyright: OGGM developers, 2014-2018

License: GPLv3+
"""
# flake8: noqa
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


try:
    from oggm.mpi import _init_oggm_mpi
    _init_oggm_mpi()
except ImportError:
    pass

# API
# TODO: why are some funcs here? maybe reconsider what API actually is
from oggm.utils import entity_task, global_task, GlacierDirectory
from oggm.core.centerlines import Centerline
from oggm.core.flowline import Flowline
