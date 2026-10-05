import os
import socket
from urllib.request import URLError, urlopen

import matplotlib.ft2font
import pytest
from packaging.version import Version

from oggm import cfg
from oggm.utils import SAMPLE_DATA_COMMIT

# The temperature bias prior file to use in the tests: this is the file the
# OGGM v1.6 preprocessed directories were calibrated with (W5E5, RGI6). There
# is no default in OGGM, it always has to be given explicitly.
TEMP_BIAS_FILE_W5E5_RGI6 = ('https://cluster.klima.uni-bremen.de/~oggm/'
                            'ref_mb_params/oggm_v1.6/'
                            'w5e5_rgi6_perglacier_temp_bias_v2025.6.2.csv')

# Regional averages of the geodetic observations. OGGM calibrates on the
# per-glacier values only, but this is a useful reference in the tests.
GEODETIC_MB_REGIONAL_AVG = ('https://cluster.klima.uni-bremen.de/~oggm/'
                            'geodetic_ref_mb/hugonnet_2021_regional_avg.csv')

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
