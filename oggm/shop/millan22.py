import logging

import numpy as np
import shapely.geometry as shpg

try:
    import salem
except ImportError:
    pass

try:
    import geopandas as gpd
except ImportError:
    pass

from oggm import utils
from oggm.exceptions import InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

default_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/velocities/millan22/'

_lookup_thickness = None

def _get_lookup_thickness():
    global _lookup_thickness
    if _lookup_thickness is None:
        fname = default_base_url + 'millan22_thickness_lookup_shp.zip'
        _lookup_thickness = gpd.read_file('zip://' + utils.file_downloader(fname))
    return _lookup_thickness


@utils.entity_task(log, writes=['gridded_data'])
def add_millan_thickness(gdir):
    """Add the Millan 22 thickness data to this glacier directory.

    Parameters
    ----------
    gdir ::py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    # Find out which file(s) we need
    gdf = _get_lookup_thickness()
    cp = shpg.Point(gdir.cenlon, gdir.cenlat)
    sel = gdf.loc[gdf.contains(cp)]
    if len(sel) > 1:
        raise NotImplementedError('Multifile Millan not implemented yet')
    if len(sel) == 0:
        raise InvalidWorkflowError(f'There seems to be no Millan file for this '
                                   f'glacier: {gdir.rgi_id}')

    # Fetch it
    url = default_base_url + sel['thickness'].iloc[0]
    input_file = utils.file_downloader(url)

    # Subset to avoid mega files
    dsb = salem.GeoTiff(input_file)
    x0, x1, y0, y1 = gdir.grid.extent
    dsb.set_subset(corners=((x0, y0), (x1, y1)), crs=gdir.grid.proj, margin=5)

    # Read the data and prevent bad surprises
    thick = dsb.get_vardata().astype(np.float64)
    # Nans with 0
    thick[~ np.isfinite(thick)] = 0
    nmax = np.nanmax(thick)
    if nmax == np.inf:
        # Replace inf with 0
        thick[thick == nmax] = 0
    # Replace negative values with 0
    thick[thick < 0] = 0

    # Reproject now
    thick = gdir.grid.map_gridded_data(thick, dsb.grid, interp='linear').filled(np.nan)

    # We mask zero ice as nodata
    thick = np.where(thick == 0, np.NaN, thick)

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'millan_ice_thickness'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm'
        ln = 'Ice thickness from Millan et al. 2022'
        v.long_name = ln
        v[:] = thick
