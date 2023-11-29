import logging
import os

# External libs
import pandas as pd
import numpy as np

# Optional libs
try:
    import salem
except ImportError:
    pass

# Locals
from oggm import utils, cfg

# Module logger
log = logging.getLogger(__name__)

GTD_BASE_URL = ('https://cluster.klima.uni-bremen.de/~oggm/glathida/glathida-main/'
                'data/glathida_2023-11-16_rgi_{}_per_id.h5')


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
        log.debug(f'({gdir.rgi_id}): no GlaThiDa data for this glacier')
        return None

    # OK - transform for later
    xx, yy = salem.transform_proj(salem.wgs84, gdir.grid.proj, df['longitude'], df['latitude'])
    df['x_proj'] = xx
    df['y_proj'] = yy

    log.debug(f'({gdir.rgi_id}): {len(df)} GlaThiDa points for this glacier')

    ii, jj = gdir.grid.transform(df['longitude'], df['latitude'], crs=salem.wgs84, nearest=True)
    df['i_grid'] = ii
    df['j_grid'] = jj

    # We trick by creating an index of similar i's and j's
    df['ij_grid'] = ['{:04d}_{:04d}'.format(i, j) for i, j in zip(ii, jj)]

    df.to_csv(gdir.get_filepath('glathida_data'), index=None)
    return df


@utils.entity_task(log)
def glathida_statistics(gdir):
    """Gather statistics about the Glathida data interpolated to this glacier.
    """

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['n_points'] = 0
    d['n_valid_thick_points'] = 0
    d['n_valid_elev_points'] = 0
    d['n_valid_gridded_points'] = 0
    d['avg_thick'] = np.NaN
    d['max_thick'] = np.NaN
    d['date_mode'] = None
    d['date_min'] = None
    d['date_max'] = None

    try:
        dfo = pd.read_csv(gdir.get_filepath('glathida_data'))
        d['n_points'] = len(dfo)
        df = dfo.loc[dfo.thickness > 0]
        dfg = df.groupby(by='ij_grid')['thickness'].mean()
        d['n_valid_thick_points'] = len(df)
        d['n_valid_elev_points'] = (dfo.elevation > 0).sum()
        d['n_valid_gridded_points'] = len(dfg)
        d['avg_thick'] = df.thickness.mean()
        d['max_thick'] = df.thickness.max()
        d['date_mode'] = df.date.mode().iloc[0]
        d['date_min'] = df.date.min()
        d['date_max'] = df.date.max()
    except:
        pass

    return d


@utils.global_task(log)
def compile_glathida_statistics(gdirs, filesuffix='', path=True):
    """Gather as much statistics as possible about a list of glaciers.

    It can be used to do result diagnostics and other stuffs.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(glathida_statistics, gdirs)
    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('glathida_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out
