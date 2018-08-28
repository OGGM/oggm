import logging
import os
from distutils.version import LooseVersion

import matplotlib.ft2font
from urllib.request import urlopen, URLError

from oggm import cfg
from oggm.utils import SAMPLE_DATA_COMMIT

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# Some logic to see which environment we are running on

# Matplotlib version changes plots, too
HAS_MPL_FOR_TESTS = False
if LooseVersion(matplotlib.__version__) >= LooseVersion('2'):
    HAS_MPL_FOR_TESTS = True
    BASELINE_DIR = os.path.join(cfg.CACHE_DIR,
                                'oggm-sample-data-%s' % SAMPLE_DATA_COMMIT,
                                'baseline_images')
    ftver = LooseVersion(matplotlib.ft2font.__freetype_version__)
    if ftver >= LooseVersion('2.8.0'):
        BASELINE_DIR = os.path.join(BASELINE_DIR, 'freetype_28')
    else:
        BASELINE_DIR = os.path.join(BASELINE_DIR, 'freetype_old')

# quick n dirty method to see if internet is on
try:
    _ = urlopen('http://www.google.com', timeout=1)
    HAS_INTERNET = True
except URLError:
    HAS_INTERNET = False
