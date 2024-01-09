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
try:
    import rasterio
except ImportError:
    pass

from oggm import utils, cfg
from oggm.exceptions import InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

base_url = ('http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/'
            'velocity_mosaic/landsat/v00.0/static/cog/')
regions = ['HMA', 'ANT', 'PAT', 'ALA', 'CAN', 'GRE', 'ICE', 'SRA']

region_files = {}
for reg in regions:
    d = {}
    for var in ['vx', 'vy', 'vy_err', 'vx_err']:
        d[var] = base_url + '{}_G0120_0000_{}.tif'.format(reg, var)
    region_files[reg] = d

region_grids = {}

rgi_region_links = {'01': 'ALA', '02': 'ALA',
                    '03': 'CAN', '04': 'CAN',
                    '05': 'GRE',
                    '06': 'ICE',
                    '07': 'SRA', '09': 'SRA',
                    '13': 'HMA', '14': 'HMA', '15': 'HMA',
                    '17': 'PAT',
                    '19': 'ANT',
                    }


def region_grid(reg):

    global region_grids

    if reg not in region_grids:
        with utils.get_lock():
            fp = utils.file_downloader(region_files[reg]['vx'])
            ds = salem.GeoTiff(fp)
            region_grids[reg] = ds.grid

    return region_grids[reg]


def _in_grid(grid, lon, lat):

    i, j = grid.transform([lon], [lat], maskout=True)
    return np.all(~ (i.mask | j.mask))


def find_region(gdir):

    reg = rgi_region_links.get(gdir.rgi_region, None)

    if reg is None:
        return None

    grid = region_grid(reg)

    if _in_grid(grid, gdir.cenlon, gdir.cenlat):
        return reg
    else:
        return None


def _reproject_and_scale(gdir, do_error=False):
    """Reproject and scale itslive data, avoid code duplication for error"""

    reg = find_region(gdir)
    if reg is None:
        raise InvalidWorkflowError('There does not seem to be its_live data '
                                   'available for this glacier')

    vnx = 'vx'
    vny = 'vy'
    if do_error:
        vnx += '_err'
        vny += '_err'

    with utils.get_lock():
        fx = utils.file_downloader(region_files[reg][vnx])
        fy = utils.file_downloader(region_files[reg][vny])

    # Open the files
    dsx = salem.GeoTiff(fx)
    dsy = salem.GeoTiff(fy)
    # subset them to our map
    grid_gla = gdir.grid.center_grid
    proj_vel = dsx.grid.proj
    x0, x1, y0, y1 = grid_gla.extent_in_crs(proj_vel)
    with warnings.catch_warnings():
        # This can trigger an out-of-bounds warning
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message='.*out of bounds.*')
        dsx.set_subset(corners=((x0, y0), (x1, y1)), crs=proj_vel, margin=4)
        dsy.set_subset(corners=((x0, y0), (x1, y1)), crs=proj_vel, margin=4)
    grid_vel = dsx.grid.center_grid

    # TODO: this should be taken care of by salem
    # https://github.com/fmaussion/salem/issues/171
    with rasterio.Env():
        with rasterio.open(fx) as src:
            nodata = getattr(src, 'nodata', -32767.0)

    # Error files are wrong
    if nodata == 0:
        nodata = -32767.0

    # Get the coords at t0
    xx0, yy0 = grid_vel.center_grid.xy_coordinates

    # Compute coords at t1
    xx1 = dsx.get_vardata()
    yy1 = dsy.get_vardata()
    non_valid = (xx1 == nodata) | (yy1 == nodata)
    xx1[non_valid] = np.NaN
    yy1[non_valid] = np.NaN
    orig_vel = np.sqrt(xx1**2 + yy1**2)
    xx1 += xx0
    yy1 += yy0

    # Transform both to glacier proj
    xx0, yy0 = salem.transform_proj(proj_vel, grid_gla.proj, xx0, yy0)
    xx1, yy1 = salem.transform_proj(proj_vel, grid_gla.proj, xx1, yy1)

    # Correct no data after proj as well (inf)
    xx1[non_valid] = np.NaN
    yy1[non_valid] = np.NaN

    # Compute velocities from there
    vx = xx1 - xx0
    vy = yy1 - yy0

    # Scale back velocities - https://github.com/OGGM/oggm/issues/1014
    new_vel = np.sqrt(vx**2 + vy**2)
    p_ok = new_vel > 1  # avoid div by zero
    vx[p_ok] = vx[p_ok] * orig_vel[p_ok] / new_vel[p_ok]
    vy[p_ok] = vy[p_ok] * orig_vel[p_ok] / new_vel[p_ok]

    # And transform to local map
    vx = grid_gla.map_gridded_data(vx, grid=grid_vel, interp='linear')
    vy = grid_gla.map_gridded_data(vy, grid=grid_vel, interp='linear')

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        vn = 'itslive_vx'
        if do_error:
            vn += '_err'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm yr-1'
        ln = 'ITS LIVE velocity data in x map direction'
        if do_error:
            ln = 'Uncertainty of ' + ln
        v.long_name = ln
        v[:] = vx.filled(np.nan)

        vn = 'itslive_vy'
        if do_error:
            vn += '_err'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm yr-1'
        ln = 'ITS LIVE velocity data in y map direction'
        if do_error:
            ln = 'Uncertainty of ' + ln
        v.long_name = ln
        v[:] = vy.filled(np.nan)

        if not do_error:
            vel = np.sqrt(vx ** 2 + vy ** 2)
            vn = 'itslive_v'
            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
            v.units = 'm yr-1'
            ln = 'ITS LIVE velocity data'
            v.long_name = ln
            v[:] = vel.filled(np.nan)


@utils.entity_task(log, writes=['gridded_data'])
def velocity_to_gdir(gdir, add_error=False):
    """Reproject the its_live files to the given glacier directory.

    The data source used is https://its-live.jpl.nasa.gov/#data
    Currently the only data downloaded is the 120m composite for both
    (u, v) and their uncertainty. The composite is computed from the
    1985 to 2018 average.

    Variables are added to the gridded_data nc file.

    Reprojecting velocities from one map proj to another is done
    reprojecting the vector distances. In this process, absolute velocities
    might change as well because map projections do not always preserve
    distances -> we scale them back to the original velocities as per the
    ITS_LIVE documentation that states that velocities are given in
    ground units, i.e. absolute velocities.

    We use bilinear interpolation to reproject the velocities to the local
    glacier map.

    If you want more velocity products, feel free to open a new topic
    on OGGM's issue tracker!

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    add_error : bool
        also reproject and scale the error data
    """

    if not gdir.has_file('gridded_data'):
        raise InvalidWorkflowError('Please run `glacier_masks` before running '
                                   'this task')

    _reproject_and_scale(gdir, do_error=False)
    if add_error:
        _reproject_and_scale(gdir, do_error=True)


@utils.entity_task(log)
def itslive_statistics(gdir):
    """Gather statistics about the itslive data interpolated to this glacier.
    """

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            v = ds['itslive_v'].where(ds['glacier_mask'], np.NaN).load()
            with warnings.catch_warnings():
                # For operational runs we ignore the warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                d['itslive_avg_vel'] = np.nanmean(v)
                d['itslive_max_vel'] = np.nanmax(v)
                d['itslive_perc_cov'] = (float((~v.isnull()).sum() * gdir.grid.dx ** 2 * 1e-6) /
                                         gdir.rgi_area_km2)
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@utils.global_task(log)
def compile_itslive_statistics(gdirs, filesuffix='', path=True):
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

    out_df = execute_entity_task(itslive_statistics, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('its_live_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out
