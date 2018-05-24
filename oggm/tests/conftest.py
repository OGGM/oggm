import pytest
import logging
import getpass
from oggm import cfg, utils

logger = logging.getLogger(__name__)

def pytest_configure(config):
    if config.pluginmanager.hasplugin('xdist'):
        try:
            from ilock import ILock
            utils.lock = ILock("oggm_xdist_download_lock_" + getpass.getuser())
            logger.info("ilock locking setup successfully for xdist tests")
        except:
            logger.warning("could not setup ilock locking for distributed tests")

