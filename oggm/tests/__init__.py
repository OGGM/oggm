import os
import socket
from urllib.request import URLError, urlopen

import matplotlib.ft2font
import pytest
from packaging.version import Version

from oggm import cfg
from oggm.utils import SAMPLE_DATA_COMMIT

# Some logic to see which environment we are running on

# Matplotlib version changes plots, too
HAS_MPL_FOR_TESTS = False
if Version(matplotlib.__version__) >= Version("2"):
    HAS_MPL_FOR_TESTS = True
    BASELINE_DIR = os.path.join(cfg.CACHE_DIR,
                                'oggm-sample-data-%s' % SAMPLE_DATA_COMMIT,
                                'baseline_images', 'freetype_28')

def check_internet_access(
    hostname: str = "8.8.8.8", port: int = 53, timeout: int = 1
):
    """Check if Internet is available.
    
    hostname : str, default "8.8.8.8"
        Web address. Can be a public DNS or an HTTP link.
    port : int, default 53
        An open and unfiltered port number. This should be 53 for
        the domain, or 443 for https.
    timeout : int, default 1
        Time in seconds before the connection times out.

    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((hostname, port))
        return True
    except socket.error as e:
        return False

HAS_INTERNET = check_internet_access()


def mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=1, **kwargs):
    return pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                         tolerance=tolerance,
                                         **kwargs)
