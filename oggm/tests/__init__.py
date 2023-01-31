import os
from packaging.version import Version

import pytest
import matplotlib.ft2font
from urllib.request import urlopen, URLError

from oggm import cfg
from oggm.utils import SAMPLE_DATA_COMMIT

# Some logic to see which environment we are running on

# Matplotlib version changes plots, too
HAS_MPL_FOR_TESTS = False
if Version(matplotlib.__version__) >= Version('2'):
    HAS_MPL_FOR_TESTS = True
    BASELINE_DIR = os.path.join(cfg.CACHE_DIR,
                                'oggm-sample-data-%s' % SAMPLE_DATA_COMMIT,
                                'baseline_images', 'freetype_28')

# quick n dirty method to see if internet is on
try:
    _ = urlopen('http://www.google.com', timeout=1)
    HAS_INTERNET = True
except URLError:
    HAS_INTERNET = False


def mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=1, **kwargs):
    return pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                         tolerance=tolerance,
                                         **kwargs)
