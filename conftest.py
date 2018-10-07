import pytest
import logging
import getpass
import matplotlib.pyplot as plt
from functools import wraps
from oggm import utils
from oggm.tests import HAS_MPL_FOR_TESTS, HAS_INTERNET


logger = logging.getLogger(__name__)


def pytest_configure(config):
    if config.pluginmanager.hasplugin('xdist'):
        try:
            from ilock import ILock
            utils.lock = ILock("oggm_xdist_download_lock_" + getpass.getuser())
            logger.info("ilock locking setup successfully for xdist tests")
        except BaseException:
            logger.warning("could not setup ilock locking for distributed "
                           "tests")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Run slow tests")
    parser.addoption("--run-download", action="store_true", default=False,
                     help="Run download tests")
    parser.addoption("--run-creds", action="store_true", default=False,
                     help="Run download tests requiring credentians")
    parser.addoption("--run-test-env", metavar="ENVNAME", default="",
                     help="Run only specified test env")
    parser.addoption("--no-run-internet", action="store_true", default=False,
                     help="Don't run any tests accessing the internet")


def pytest_collection_modifyitems(config, items):
    use_internet = HAS_INTERNET and not config.getoption("--no-run-internet")
    skip_slow = not config.getoption("--run-slow")
    skip_download = not use_internet or not config.getoption("--run-download")
    skip_cred = skip_download or not config.getoption("--run-creds")
    run_test_env = config.getoption("--run-test-env")

    slow_marker = pytest.mark.skip(reason="need --run-slow option to run")
    download_marker = pytest.mark.skip(reason="need --run-download option to "
                                              "run, internet access is "
                                              "required")
    cred_marker = pytest.mark.skip(reason="need --run-creds option to run, "
                                          "internet access is required")
    internet_marker = pytest.mark.skip(reason="internet access is required")
    test_env_marker = pytest.mark.skip(reason="only test_env=%s tests are run"
                                              % run_test_env)
    graphic_marker = pytest.mark.skip(reason="requires mpl V1.5+ and "
                                             "pytest-mpl")

    for item in items:
        if skip_slow and "slow" in item.keywords:
            item.add_marker(slow_marker)
        if skip_download and "download" in item.keywords:
            item.add_marker(download_marker)
        if skip_cred and "creds" in item.keywords:
            item.add_marker(cred_marker)
        if not use_internet and "internet" in item.keywords:
            item.add_marker(internet_marker)

        if run_test_env:
            test_env = item.get_closest_marker("test_env")
            if not test_env or test_env.args[0] != run_test_env:
                item.add_marker(test_env_marker)

        if "graphic" in item.keywords:
            def wrap_graphic_test(test):
                @wraps(test)
                def test_wrapper(*args, **kwargs):
                    try:
                        return test(*args, **kwargs)
                    finally:
                        plt.close()
                return test_wrapper
            item.obj = wrap_graphic_test(item.obj)

            if not HAS_MPL_FOR_TESTS:
                item.add_marker(graphic_marker)
