import logging

import os
import shutil

try:
    import salem
except ImportError:
    pass

from oggm import utils, cfg
from oggm.core import gis
from oggm.exceptions import InvalidParamsError

# Module logger
log = logging.getLogger(__name__)


@utils.entity_task(log, writes=['dem'])
def select_dem(gdir, dem_source=None):
    """Select DEM"""

    if dem_source is None:
        raise InvalidParamsError('Currently not choosing anything - pick one')

    shutil.copyfile(os.path.join(gdir.dir, dem_source, 'dem.tif'),
                    gdir.get_filepath('dem'))
    shutil.copyfile(os.path.join(gdir.dir, dem_source, 'dem_source.txt'),
                    gdir.get_filepath('dem_source'))
