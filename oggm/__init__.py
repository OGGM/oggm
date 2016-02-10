""" OGGM package.

Copyright: OGGM developers, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division
import logging

# Basic config
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Fiona and shapely are spammers
logging.getLogger("Fiona").setLevel(logging.WARNING)
logging.getLogger("shapely").setLevel(logging.WARNING)

from oggm.utils import GlacierDirectory, entity_task, divide_task
