import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
import shapely.geometry as shpg

try:
    import salem
except ImportError:
    pass

try:
    import geopandas as gpd
except ImportError:
    pass

from oggm import utils, cfg
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
    if len(sel) == 0:
        raise InvalidWorkflowError(f'There seems to be no Millan file for this '
                                   f'glacier: {gdir.rgi_id}')

    # We may have more than one file
    total_thick = 0
    for i, s in sel.iterrows():
        # Fetch it
        url = default_base_url + s['thickness']
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

        if np.nansum(thick) == 0:
            # No need to continue
            continue

        # Reproject now
        r_thick = gdir.grid.map_gridded_data(thick, dsb.grid, interp='linear')
        total_thick += r_thick.filled(0)

    # We mask zero ice as nodata
    total_thick = np.where(total_thick == 0, np.NaN, total_thick)

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
        v[:] = total_thick

@utils.entity_task(log)
def millan_statistics(gdir):
    """Gather statistics about the Millan data interpolated to this glacier.
    """

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['millan_vol_km3'] = 0
    d['millan_area_km2'] = 0
    d['millan_perc_cov'] = 0

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            thick = ds['millan_ice_thickness'].where(ds['glacier_mask'], np.NaN).load()
            d['millan_vol_km3'] = float(thick.sum() * gdir.grid.dx ** 2 * 1e-9)
            d['millan_area_km2'] = float((~thick.isnull()).sum() * gdir.grid.dx ** 2 * 1e-6)
            d['millan_perc_cov'] = float(d['millan_area_km2'] / gdir.rgi_area_km2)
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@utils.global_task(log)
def compile_millan_statistics(gdirs, filesuffix='', path=True):
    """Gather as much statistics as possible about a list of glaciers.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

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

    out_df = execute_entity_task(millan_statistics, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('millan_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out
