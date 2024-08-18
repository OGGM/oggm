import logging
import warnings
import os

import numpy as np
import pandas as pd
import xarray as xr

try:
    import salem
except ImportError:
    pass

from oggm import utils, cfg

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
    with warnings.catch_warnings():
        # This can trigger an out of bounds warning
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message='.*out of bounds.*')
        thick = gdir.grid.map_gridded_data(thick, dsb.grid, interp='linear')

    # Correct for volume
    thick = utils.clip_min(thick.filled(0), 0)
    out_volume = thick.sum() * gdir.grid.dx ** 2
    if out_volume > 0:
        thick *= in_volume / out_volume

    # We mask zero ice as nodata
    thick = np.where(thick == 0, np.nan, thick)

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


@utils.entity_task(log)
def consensus_statistics(gdir):
    """Gather statistics about the consensus data interpolated to this glacier.
    """

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['consensus_vol_km3'] = 0
    d['consensus_area_km2'] = 0
    d['consensus_perc_cov'] = 0

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            thick = ds['consensus_ice_thickness'].where(ds['glacier_mask'], np.nan).load()
            d['consensus_vol_km3'] = float(thick.sum() * gdir.grid.dx ** 2 * 1e-9)
            d['consensus_area_km2'] = float((~thick.isnull()).sum() * gdir.grid.dx ** 2 * 1e-6)
            d['consensus_perc_cov'] = float(d['consensus_area_km2'] / gdir.rgi_area_km2)
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@utils.global_task(log)
def compile_consensus_statistics(gdirs, filesuffix='', path=True):
    """Gather as much statistics as possible about a list of glaciers.

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

    out_df = execute_entity_task(consensus_statistics, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('consensus_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out
