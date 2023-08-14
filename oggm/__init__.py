""" OGGM package.

Copyright: OGGM e.V. and OGGM Contributors

License: BSD-3-Clause
"""
# flake8: noqa
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version(__name__)
    except PackageNotFoundError:
        # package is not installed
        pass
    finally:
        del version, PackageNotFoundError
except ModuleNotFoundError:
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

# TODO: remove this when geopandas will behave a bit better
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module=r'.*geopandas')

# API
# Some decorators used by many
from oggm.utils import entity_task, global_task, DEFAULT_BASE_URL

# Classes
from oggm.utils import GlacierDirectory
from oggm.core.centerlines import Centerline
from oggm.core.flowline import Flowline
from oggm.core.flowline import FlowlineModel
from oggm.core.flowline import FileModel
from oggm.core.massbalance import MassBalanceModel
