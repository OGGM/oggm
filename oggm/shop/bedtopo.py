import logging

import numpy as np

try:
    import salem
except ImportError:
    pass

from oggm import utils

# Module logger
log = logging.getLogger(__name__)

default_base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/icevol/composite/'


@utils.entity_task(log, writes=['gridded_data'])
def add_consensus_thickness(gdir, base_url=None):
    """Add the consensus thickness estimate to the gridded_data file.

    varname: consensus_ice_thickness

    Parameters
    ----------
    gdir ::py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    base_url : str
        where to find the thickness data. Default is
        https://cluster.klima.uni-bremen.de/~fmaussion/icevol/composite
    """

    if base_url is None:
        base_url = default_base_url
    if not base_url.endswith('/'):
        base_url += '/'

    rgi_str = gdir.rgi_id
    rgi_reg_str = rgi_str[:8]

    url = base_url + rgi_reg_str + '/' + rgi_str + '_thickness.tif'
    input_file = utils.file_downloader(url)

    dsb = salem.GeoTiff(input_file)
    thick = utils.clip_min(dsb.get_vardata(), 0)
    in_volume = thick.sum() * dsb.grid.dx ** 2
    thick = gdir.grid.map_gridded_data(thick, dsb.grid, interp='linear')

    # Correct for volume
    thick = utils.clip_min(thick.filled(0), 0)
    out_volume = thick.sum() * gdir.grid.dx ** 2
    if out_volume > 0:
        thick *= in_volume / out_volume

    # We mask zero ice as nodata
    thick = np.where(thick == 0, np.NaN, thick)

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'consensus_ice_thickness'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm'
        ln = 'Ice thickness from the consensus estimate'
        v.long_name = ln
        v.base_url = base_url
        v[:] = thick
