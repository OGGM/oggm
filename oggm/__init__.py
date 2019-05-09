""" OGGM package.

Copyright: OGGM developers, 2014-2018

License: GPLv3+
"""
# flake8: noqa
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
finally:
    del get_distribution, DistributionNotFound


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
