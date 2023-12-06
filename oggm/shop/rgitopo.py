import logging

import os
import shutil
import warnings

try:
    import salem
except ImportError:
    pass
try:
    import rasterio
except ImportError:
    pass
import numpy as np

from oggm import utils, workflow
from oggm.exceptions import InvalidParamsError

# Module logger
log = logging.getLogger(__name__)

DEMS_URL = 'https://cluster.klima.uni-bremen.de/data/gdirs/dems_v2/default'
DEMS_HR_URL = 'https://cluster.klima.uni-bremen.de/data/gdirs/dems_v1/highres/'


def init_glacier_directories_from_rgitopo(rgidf=None, dem_source=None,
                                          resolution='default',
                                          keep_dem_folders=False,
                                          reset=False,
                                          force=True):
    """Initialize a glacier directory from an RGI-TOPO DEM.

    Wrapper around :func:`workflow.init_glacier_directories`, which selects
    a default source for you (if not set manually with `dem_source`).

    The default source is NASADEM for all latitudes between 60N and 56S,
    and COPDEM90 elsewhere.

    Parameters
    ----------
    rgidf : GeoDataFrame or list of ids, optional for pre-computed runs
        the RGI glacier outlines. If unavailable, OGGM will parse the
        information from the glacier directories found in the working
        directory. It is required for new runs.
    reset : bool
        delete the existing glacier directories if found.
    force : bool
        setting `reset=True` will trigger a yes/no question to the user. Set
        `force=True` to avoid this.
    dem_source : str
        the source to pick from (default: NASADEM and COPDEM90)
    keep_dem_folders : bool
        the default is to delete the other DEM directories to save space.
        Set this to True to prevent that (e.g. for sensitivity tests)
    Returns
    -------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the initialised glacier directories
    """

    if resolution == 'default':
        base_url = DEMS_URL
    elif resolution == 'lr':
        base_url = DEMS_URL
    elif resolution == 'hr':
        log.warning('High-res DEMs not available yet in version 2 with COPDEM30')
        base_url = DEMS_HR_URL
    else:
        raise InvalidParamsError('`resolution` should be of `lr` or `hr`')

    gdirs = workflow.init_glacier_directories(rgidf, reset=reset, force=force,
                                              prepro_base_url=base_url,
                                              from_prepro_level=1,
                                              prepro_rgi_version='62')

    workflow.execute_entity_task(select_dem_from_dir, gdirs,
                                 dem_source=dem_source,
                                 keep_dem_folders=keep_dem_folders)

    return gdirs


@utils.entity_task(log, writes=['dem'])
def select_dem_from_dir(gdir, dem_source=None, keep_dem_folders=False):
    """Select a DEM from the ones available in an RGI-TOPO glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    dem_source : str
        the source to pick from. If 'RGI', we assume that there is a
        `dem_source` attribute in the RGI file. If 'BY_RES', we use
        COPDEM30 for all gdirs with resolution smaller than 60m
    keep_dem_folders : bool
        the default is to delete the other DEM directories to save space.
        Set this to True to prevent that (e.g. for sensitivity tests)
    """

    if dem_source == 'RGI':
        dem_source = gdir.rgi_dem_source
    if dem_source == 'BY_RES':
        dem_source = 'COPDEM30' if gdir.grid.dx < 60 else 'COPDEM90'

    sources = [f.name for f in os.scandir(gdir.dir) if f.is_dir()
               and not f.name.startswith('.')]
    if dem_source not in sources:
        raise RuntimeError('source {} not in folder'.format(dem_source))

    gdir.add_to_diagnostics('dem_source', dem_source)
    shutil.copyfile(os.path.join(gdir.dir, dem_source, 'dem.tif'),
                    gdir.get_filepath('dem'))
    shutil.copyfile(os.path.join(gdir.dir, dem_source, 'dem_source.txt'),
                    gdir.get_filepath('dem_source'))

    if not keep_dem_folders:
        for source in sources:
            shutil.rmtree(os.path.join(gdir.dir, source))


def _fallback_dem_quality_check(gdir):
    return dict(rgi_id=gdir.rgi_id)


@utils.entity_task(log, writes=[], fallback=_fallback_dem_quality_check)
def dem_quality_check(gdir):
    """Run a simple quality check on the rgitopo DEMs

    Parameters
    ----------
    gdir : GlacierDirectory
        the glacier directory

    Returns
    -------
    a dict of DEMSOURCE:frac pairs, where frac is the percentage of
    valid DEM grid points on the glacier.
    """

    with rasterio.open(gdir.get_filepath('glacier_mask'), 'r',
                       driver='GTiff') as ds:
        mask = ds.read(1) == 1
    area = mask.sum()

    sources = [f.name for f in os.scandir(gdir.dir) if f.is_dir()
               and not f.name.startswith('.')]

    out = dict(rgi_id=gdir.rgi_id)
    for s in sources:
        try:
            with rasterio.open(os.path.join(gdir.dir, s, 'dem.tif'), 'r',
                               driver='GTiff') as ds:
                topo = ds.read(1).astype(rasterio.float32)
                topo[topo <= -999.] = np.NaN
                topo[ds.read_masks(1) == 0] = np.NaN
            valid_mask = np.isfinite(topo) & mask
            if np.all(~valid_mask):
                continue
            if np.nanmax(topo[valid_mask]) == 0:
                continue
            out[s] = valid_mask.sum() / area
        except BaseException:
            pass
    return out
