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
        file_crs = dem_dss[0].crs
        dem_data = rasterio.band(dem_dss[0], 1)
        if Version(rasterio.__version__) >= Version('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
        nodata = dem_dss[0].meta.get('nodata', None)
    else:
        dem_dss = [rasterio.open(s) for s in flist]  # list of rasters

        # make sure all files have the same crs and reproject if needed;
        # defining the target crs to the one most commonly used, minimizing
        # the number of files for reprojection
        crs_list = np.array([dem_ds.crs.to_string() for dem_ds in dem_dss])
        unique_crs, crs_counts = np.unique(crs_list, return_counts=True)
        file_crs = rasterio.crs.CRS.from_string(
            unique_crs[np.argmax(crs_counts)])

        if len(unique_crs) != 1:
            # more than one crs, we need to do reprojection
            memory_files = []
            for i, src in enumerate(dem_dss):
                if file_crs != src.crs:
                    transform, width, height = calculate_default_transform(
                        src.crs, file_crs, src.width, src.height, *src.bounds)
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': file_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })

                    reprojected_array = np.empty(shape=(src.count, height, width),
                                                 dtype=src.dtypes[0])
                    # just for completeness; even the data only has one band
                    for band in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, band),
                                  destination=reprojected_array[band - 1],
                                  src_transform=src.transform,
                                  src_crs=src.crs,
                                  dst_transform=transform,
                                  dst_crs=file_crs,
                                  resampling=Resampling.nearest)

                    memfile = MemoryFile()
                    with memfile.open(**kwargs) as mem_dst:
                        mem_dst.write(reprojected_array)
                    memory_files.append(memfile)
                else:
                    memfile = MemoryFile()
                    with memfile.open(**src.meta) as mem_src:
                        mem_src.write(src.read())
                    memory_files.append(memfile)

            with rasterio.Env():
                datasets_to_merge = [memfile.open() for memfile in memory_files]
                nodata = datasets_to_merge[0].meta.get('nodata', None)
                dem_data, src_transform = merge_tool(datasets_to_merge,
                                                     nodata=nodata)
        else:
            # only one single crs occurring, no reprojection needed
            nodata = dem_dss[0].meta.get('nodata', None)
            dem_data, src_transform = merge_tool(dem_dss, nodata=nodata)

    # Set up profile for writing output
    with rasterio.open(gdir.get_filepath('dem')) as dem_ds:
        dst_array = dem_ds.read().astype(np.float32)
        dst_array[:] = np.nan
        profile = dem_ds.profile
        transform = dem_ds.transform
        dst_crs = dem_ds.crs

    # Set up profile for writing output
    profile.update({
        'nodata': np.nan,
    })

    resampling = Resampling.bilinear

    with MemoryFile() as dest:
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=file_crs,
            src_transform=src_transform,
            src_nodata=nodata,
            # Destination parameters
            destination=dst_array,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
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


@utils.entity_task(log)
def hugonnet_statistics(gdir):
    """Gather statistics about the Hugonnet data interpolated to this glacier.
    """

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['hugonnet_area_km2'] = 0
    d['hugonnet_perc_cov'] = 0
    d['hugonnet_avg_dhdt'] = np.nan

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            dhdt = ds['hugonnet_dhdt'].where(ds['glacier_mask'], np.nan).load()
            d['hugonnet_area_km2'] = float((~dhdt.isnull()).sum() * gdir.grid.dx ** 2 * 1e-6)
            d['hugonnet_perc_cov'] = float(d['hugonnet_area_km2'] / gdir.rgi_area_km2)
            with warnings.catch_warnings():
                # This can trigger an empty mean warning
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                d['hugonnet_avg_dhdt'] = np.nanmean(dhdt.data)
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@utils.global_task(log)
def compile_hugonnet_statistics(gdirs, filesuffix='', path=True):
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

    out_df = execute_entity_task(hugonnet_statistics, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('hugonnet_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out
