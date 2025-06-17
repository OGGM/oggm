import logging
import warnings
from packaging.version import Version

import numpy as np
import pandas as pd
import xarray as xr
import os

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio import MemoryFile
    try:
        # rasterio V > 1.0
        from rasterio.merge import merge as merge_tool
    except ImportError:
        from rasterio.tools.merge import merge as merge_tool
except ImportError:
    pass


from oggm import utils, cfg

# Module logger
log = logging.getLogger(__name__)

aa_base_url = ('https://n5eil01u.ecs.nsidc.org/MEASURES/'
               'NSIDC-0756.003/1970.01.01/BedMachineAntarctica-v3.nc')
grl_base_url = ('https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IDBMG4.005/'
                '1993.01.01/BedMachineGreenland-v5.nc')


@utils.entity_task(log, writes=['gridded_data'])
def bedmachine_to_gdir(gdir):
    """Add the Bedmachine ice thickness maps to this glacier directory.

    For Antarctica: BedMachineAntarctica-v3.nc
    For Greenland: BedMachineGreenland-v5.nc

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    if gdir.rgi_region == '19':
        file_url = aa_base_url
    elif gdir.rgi_region == '05':
        file_url = grl_base_url
    else:
        raise NotImplementedError('Bedmachine data not available '
                                  f'for this region: {gdir.rgi_region}')

    file_local = utils.download_with_authentication(file_url,
                                                    'urs.earthdata.nasa.gov')

    with xr.open_dataset(file_local) as ds:
        ds.attrs['pyproj_srs'] = ds.proj4

        x0, x1, y0, y1 = gdir.grid.extent_in_crs(ds.proj4)
        dsroi = ds.salem.subset(corners=((x0, y0), (x1, y1)), crs=ds.proj4, margin=10)
        thick = gdir.grid.map_gridded_data(dsroi['thickness'].data,
                                           grid=dsroi.salem.grid,
                                           interp='linear')
        thick[thick <= 0] = np.NaN

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'bedmachine_ice_thickness'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True, fill_value=np.nan)
        v.units = 'm'
        ln = 'Ice thickness from BedMachine'
        v.long_name = ln
        v.data_source = file_url
        v[:] = thick.data.astype(np.float32)


@utils.entity_task(log)
def bedmachine_statistics(gdir):
    """Gather statistics about the Bedmachine data interpolated to this glacier.
    """

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['bedmachine_area_km2'] = 0
    d['bedmachine_perc_cov'] = 0
    d['bedmachine_vol_km3'] = np.nan

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            thick = ds['bedmachine_ice_thickness'].where(ds['glacier_mask'], np.nan).load()
            gridded_area = ds['glacier_mask'].sum() * gdir.grid.dx ** 2 * 1e-6
            d['bedmachine_area_km2'] = float((~thick.isnull()).sum() * gdir.grid.dx ** 2 * 1e-6)
            d['bedmachine_perc_cov'] = float(d['bedmachine_area_km2'] / gridded_area)
            d['bedmachine_vol_km3'] = float(thick.sum() * gdir.grid.dx ** 2 * 1e-9)
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@utils.global_task(log)
def compile_bedmachine_statistics(gdirs, filesuffix='', path=True):
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

    out_df = execute_entity_task(bedmachine_statistics, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('bedmachine_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out
