import logging

# External libs
import pandas as pd

# Optional libs
try:
    import salem
except ImportError:
    pass

# Locals
from oggm import utils

# Module logger
log = logging.getLogger(__name__)

GTD_BASE_URL = ('https://cluster.klima.uni-bremen.de/~oggm/glathida/glathida-main/'
                'data/glathida_2023-11-16_rgi_{}.h5')


@utils.entity_task(log, writes=['glathida_data'])
def glathida_to_gdir(gdir):
    """Add GlaThiDa data to this glacier directory.

    The data is from the GlaThiDa main repository as of 2023-11-16. It's the raw data,
    i.e. no quality control was applied. It may contain zeros or unrealisic values.
    It's the users task to make sure the data suits their use case.

    The dataframe contains a few additional columns:
    - x_proj, y_proj: the point coordinates in the local map projection
    - i_grid, j_grid: the point coordinates in the local *nearest* grid point coordinates
                      (e.g. to access the data from a grid)
    - ij_grid: a unique identifier per grid point. Allows to group the data by gridpoint
               (see tutorials)

    https://gitlab.com/wgms/glathida

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    Returns
    -------
    the dataframe, or None if there is no data for this glacier.
    """

    rgi_version = gdir.rgi_version
    if rgi_version == '60':
        rgi_version = '62'

    base_url = GTD_BASE_URL.format(rgi_version)
    gtd_file = utils.file_downloader(base_url)

    try:
        df = pd.read_hdf(gtd_file, key=gdir.rgi_id)
    except KeyError:
        return None

    # OK - transform for later
    xx, yy = salem.transform_proj(salem.wgs84, gdir.grid.proj, df['longitude'], df['latitude'])
    df['x_proj'] = xx
    df['y_proj'] = yy

    ii, jj = gdir.grid.transform(df['longitude'], df['latitude'], crs=salem.wgs84, nearest=True)
    df['i_grid'] = ii
    df['j_grid'] = jj

    # We trick by creating an index of similar i's and j's
    df['ij_grid'] = ['{:04d}_{:04d}'.format(i, j) for i, j in zip(ii, jj)]

    df.to_csv(gdir.get_filepath('glathida_data'))
    return df
