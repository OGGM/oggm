import logging
from packaging.version import Version

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio import MemoryFile
    try:
        # rasterio V > 1.0
        from rasterio.merge import merge as merge_tool
    except ImportError:
        from rasterio.tools.merge import merge as merge_tool
except ImportError:
    pass

try:
    import salem
except ImportError:
    pass

try:
    import geopandas as gpd
except ImportError:
    pass

from oggm import utils, cfg

# Module logger
log = logging.getLogger(__name__)

default_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb_maps/'

_lookup_csv = None


def _get_lookup_csv():
    global _lookup_csv
    if _lookup_csv is None:
        fname = default_base_url + 'hugonnet_dhdt_lookup_csv_20230129.csv'
        _lookup_csv = pd.read_csv(utils.file_downloader(fname), index_col=0)
    return _lookup_csv


@utils.entity_task(log, writes=['gridded_data'])
def hugonnet_to_gdir(gdir, add_error=False):
    """Add the Hugonnet 21 dhdt maps to this glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_error : bool
        add the error data or not
    """

    if add_error:
        raise NotImplementedError('Not yet')

    # Find out which file(s) we need
    df = _get_lookup_csv()
    lon_ex, lat_ex = gdir.extent_ll

    # adding small buffer for unlikely case where one lon/lat_ex == xx.0
    lons = np.arange(np.floor(lon_ex[0] - 1e-9), np.ceil(lon_ex[1] + 1e-9))
    lats = np.arange(np.floor(lat_ex[0] - 1e-9), np.ceil(lat_ex[1] + 1e-9))

    flist = []
    for lat in lats:
        # north or south?
        ns = 'S' if lat < 0 else 'N'
        for lon in lons:
            # east or west?
            ew = 'W' if lon < 0 else 'E'
            ll_str = f'{ns}{abs(lat):02.0f}{ew}{abs(lon):03.0f}'
            try:
                filename = df.loc[(df['file_id'] == ll_str)]['dhdt'].iloc[0]
            except IndexError:
                # We can maybe be on the edge (unlikely but hey
                pass
            file_local = utils.file_downloader(default_base_url + filename)
            if file_local is not None:
                flist.append(file_local)

    # A glacier area can cover more than one tile:
    if len(flist) == 1:
        dem_dss = [rasterio.open(flist[0])]  # if one tile, just open it
        dem_data = rasterio.band(dem_dss[0], 1)
        if Version(rasterio.__version__) >= Version('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
        nodata = dem_dss[0].meta.get('nodata', None)
    else:
        dem_dss = [rasterio.open(s) for s in flist]  # list of rasters
        nodata = dem_dss[0].meta.get('nodata', None)
        dem_data, src_transform = merge_tool(dem_dss, nodata=nodata)  # merge

    # Set up profile for writing output
    with rasterio.open(gdir.get_filepath('dem')) as dem_ds:
        dst_array = dem_ds.read().astype(np.float32)
        dst_array[:] = np.NaN
        profile = dem_ds.profile
        transform = dem_ds.transform
        dst_crs = dem_ds.crs

    # Set up profile for writing output
    profile.update({
        'nodata': np.NaN,
    })

    resampling = Resampling.bilinear

    with MemoryFile() as dest:
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=dem_dss[0].crs,
            src_transform=src_transform,
            src_nodata=nodata,
            # Destination parameters
            destination=dst_array,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.NaN,
            # Configuration
            resampling=resampling)
        dest.write(dst_array)

    for dem_ds in dem_dss:
        dem_ds.close()

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'hugonnet_dhdt'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm'
        ln = 'dhdt (2000-2020) from Hugonnet et al. 2021'
        v.long_name = ln
        data_str = ' '.join(flist) if len(flist) > 1 else flist[0]
        v.data_source = data_str
        v[:] = np.squeeze(dst_array)
