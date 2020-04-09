import logging

import numpy as np

try:
    import salem
except ImportError:
    pass

from oggm import utils, cfg
from oggm.core import gis
from oggm.exceptions import InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

base_url = ('http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/'
            'velocity_mosaic/landsat/v00.0/static/cog/')
regions = ['HMA', 'ANT', 'PAT', 'ALA', 'CAN', 'GRE', 'ICE', 'SRA']

region_files = {}
for reg in regions:
    d = {}
    for var in ['vx', 'vy', 'vy_err', 'vx_err']:
        d[var] = '{}_G0120_0000_{}.tif'.format(reg, var)
    region_files[reg] = d

region_grids = {}

rgi_region_links = {'01': 'ALA', '02': 'ALA',
                    '03': 'CAN', '04': 'CAN',
                    '05': 'GRE',
                    '06': 'ICE',
                    '07': 'SRA', '09': 'SRA',
                    '13': 'HMA', '14': 'HMA', '15': 'HMA',
                    '17': 'PAT',
                    '19': 'ANT',
                    }


cfg.BASENAMES['its_live_vx'] = ('its_live_vx.tif', 'ITS_LIVE velocity files')
cfg.BASENAMES['its_live_vy'] = ('its_live_vy.tif', 'ITS_LIVE velocity files')


def region_grid(reg):

    global region_grids

    if reg not in region_grids:
        with utils.get_download_lock():
            fp = utils.file_downloader(base_url + region_files[reg]['vx'])
            ds = salem.GeoTiff(fp)
            region_grids[reg] = ds.grid

    return region_grids[reg]


def _in_grid(grid, lon, lat):

    i, j = grid.transform([lon], [lat], maskout=True)
    return np.all(~ (i.mask | j.mask))


def find_region(gdir):

    reg = rgi_region_links.get(gdir.rgi_region, None)

    if reg is None:
        return None

    grid = region_grid(reg)

    if _in_grid(grid, gdir.cenlon, gdir.cenlat):
        return reg
    else:
        return None


@utils.entity_task(log, writes=['its_live_vx', 'its_live_vy'])
def vel_to_gdir(gdir):
    """Reproject the its_live files to the given glacier direcory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data

    """

    reg = find_region(gdir)
    if reg is None:
        raise InvalidWorkflowError('There does not seem to be its_live data '
                                   'available for this glacier')

    with utils.get_download_lock():
        fx = utils.file_downloader(base_url + region_files[reg]['vx'])
        fy = utils.file_downloader(base_url + region_files[reg]['vy'])

    gis.rasterio_to_gdir(gdir, fx, 'its_live_vx')
    gis.rasterio_to_gdir(gdir, fy, 'its_live_vy')
