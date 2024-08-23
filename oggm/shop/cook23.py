import logging
import os
import warnings

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

default_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/ice_thickness/cook23/'

_lookup_thickness = None


def _get_lookup_thickness():
    global _lookup_thickness
    if _lookup_thickness is None:
        fname = default_base_url + 'cook23_thickness_lookup_shp_20240815.zip'
        _lookup_thickness = gpd.read_file('zip://' + utils.file_downloader(fname))
    return _lookup_thickness


@utils.entity_task(log, writes=['gridded_data'])
def cook23_to_gdir(gdir, vars=['thk', 'divflux']):
    """Add the Cook 23 thickness data (and others if wanted)
    to this glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    vars : list of str
        the list of variables to add
        to gdir. Must be available in the netcdf files.

    """

    # Very easy case: no cook file for this glacier
    if gdir.rgi_region != '11' and gdir.rgi_subregion != '01':
        raise InvalidWorkflowError(f'Cook23 data is only for the Alps: '
                                   f'{gdir.rgi_id}')

    # Find out which file(s) we need
    gdf = _get_lookup_thickness()
    cp = shpg.Point(gdir.cenlon, gdir.cenlat)
    sel = gdf.loc[gdf.contains(cp)]
    if len(sel) == 0:
        raise InvalidWorkflowError(f'There seems to be no Cook23 file for this '
                                   f'glacier: {gdir.rgi_id}')

    if len(sel) > 1:
        # We have multiple files for this glacier
        other = gdir.read_shapefile('outlines').to_crs(gdf.crs)
        sel = gdf.loc[gdf.contains_properly(other)]
        if len(sel) != 1:
            raise InvalidWorkflowError(f'Weird double file situation for '
                                       f'glacier: {gdir.rgi_id}')

    sel = sel.iloc[0]
    url = default_base_url + sel['thickness']
    input_file = utils.file_downloader(url)

    # Read and reproject
    with xr.open_dataset(input_file) as ds:
        for var in vars:
            if var not in ds:
                raise InvalidWorkflowError(f'Variable {var} not found in the '
                                           f'Cook file for glacier {gdir.rgi_id}')
            ds[var].attrs['pyproj_srs'] = 'EPSG:32632'
        ds.attrs['pyproj_srs'] = 'EPSG:32632'
        ds = ds[vars].load()

    # Reproject and write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        for var in vars:

            # Reproject
            data = gdir.grid.map_gridded_data(ds[var].data,
                                              ds.salem.grid,
                                              interp='linear')

            # Write
            vn = 'cook23_' + var
            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)

            if var == 'thk':
                v.units = 'm'
                v.long_name = 'Ice thickness from Cook et al. 2023'
            elif var == 'divflux':
                v.units = 'm yr-1'
                v.long_name = 'Divergence of ice flux from Cook et al. 2023'

            v.data_source = url
            v[:] = data


@utils.entity_task(log)
def cook23_statistics(gdir):
    """Gather statistics about the Cook23 data interpolated to this glacier.
    """

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['cook23_vol_km3'] = 0
    d['cook23_area_km2'] = 0
    d['cook23_perc_cov'] = 0

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            thick = ds['cook23_thk'].where(ds['glacier_mask'], np.nan).load()
            with warnings.catch_warnings():
                # For operational runs we ignore the warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                d['cook23_vol_km3'] = float(thick.sum() * gdir.grid.dx ** 2 * 1e-9)
                d['cook23_area_km2'] = float((~thick.isnull()).sum() * gdir.grid.dx ** 2 * 1e-6)
                d['cook23_perc_cov'] = float(d['cook23_area_km2'] / gdir.rgi_area_km2)

    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@utils.global_task(log)
def compile_cook23_statistics(gdirs, filesuffix='', path=True):
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

    out_df = execute_entity_task(cook23_statistics, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('cook23_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out
