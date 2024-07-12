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

default_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/velocities/millan22/'

_lookup_thickness = None
_lookup_velocity = None


def _get_lookup_thickness():
    global _lookup_thickness
    if _lookup_thickness is None:
        fname = default_base_url + 'millan22_thickness_lookup_shp_20220902.zip'
        _lookup_thickness = gpd.read_file('zip://' + utils.file_downloader(fname))
    return _lookup_thickness


def _get_lookup_velocity():
    global _lookup_velocity
    if _lookup_velocity is None:
        fname = default_base_url + 'millan22_velocity_lookup_shp_20231127.zip'
        _lookup_velocity = gpd.read_file('zip://' + utils.file_downloader(fname))
    return _lookup_velocity


def _filter(ds):
    # Read the data and prevent bad surprises
    data = ds.get_vardata().astype(np.float64)
    # Nans with 0
    data[~ np.isfinite(data)] = 0
    nmax = np.nanmax(data)
    if nmax == np.inf:
        # Replace inf with 0
        data[data == nmax] = 0

    return data


def _filter_and_reproj(gdir, var, gdf, allow_neg=True):
    """ Same code for thickness and error

    Parameters
    ----------
    var : 'thickness' or 'err'
    gdf : the lookup
    """

    # We may have more than one file
    total_data = 0
    grids_used = []
    files_used = []
    for i, s in gdf.iterrows():
        # Fetch it
        url = default_base_url + s[var]
        input_file = utils.file_downloader(url)

        # Subset to avoid mega files
        dsb = salem.GeoTiff(input_file)
        x0, x1, y0, y1 = gdir.grid.extent_in_crs(dsb.grid.proj)
        with warnings.catch_warnings():
            # This can trigger an out of bounds warning
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message='.*out of bounds.*')
            dsb.set_subset(corners=((x0, y0), (x1, y1)),
                           crs=dsb.grid.proj,
                           margin=5)

        data = _filter(dsb)
        if not allow_neg:
            # Replace negative values with 0
            data[data < 0] = 0

        if np.nansum(data) == 0:
            # No need to continue
            continue

        # Reproject now
        with warnings.catch_warnings():
            # This can trigger an out of bounds warning
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message='.*out of bounds.*')
            r_data = gdir.grid.map_gridded_data(data, dsb.grid, interp='linear')
        total_data += r_data.filled(0)
        grids_used.append(dsb)
        files_used.append(s.file_id)

    return total_data, files_used, grids_used


@utils.entity_task(log, writes=['gridded_data'])
def thickness_to_gdir(gdir, add_error=False):
    """Add the Millan 22 thickness data to this glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_error : bool
        add the error data or not
    """

    # Find out which file(s) we need
    gdf = _get_lookup_thickness()
    cp = shpg.Point(gdir.cenlon, gdir.cenlat)
    sel = gdf.loc[gdf.contains(cp)]
    if len(sel) == 0:
        raise InvalidWorkflowError(f'There seems to be no Millan file for this '
                                   f'glacier: {gdir.rgi_id}')

    total_thick, files_used, _ = _filter_and_reproj(gdir, 'thickness', sel,
                                                    allow_neg=False)

    # We mask zero ice as nodata
    total_thick = np.where(total_thick == 0, np.nan, total_thick)

    if add_error:
        total_err, _, _ = _filter_and_reproj(gdir, 'err', sel, allow_neg=False)
        total_err[~ np.isfinite(total_thick)] = np.nan
        # Error cannot be larger than ice thickness itself
        total_err = utils.clip_max(total_err, total_thick)

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
        data_str = ' '.join(files_used) if len(files_used) > 1 else files_used[0]
        v.data_source = data_str
        v[:] = total_thick

        if add_error:
            vn = 'millan_ice_thickness_err'
            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
            v.units = 'm'
            ln = 'Ice thickness error from Millan et al. 2022'
            v.long_name = ln
            v.data_source = data_str
            v[:] = total_err


@utils.entity_task(log, writes=['gridded_data'])
def velocity_to_gdir(gdir, add_error=False):
    """Add the Millan 22 velocity data to this glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_error : bool
        add the error data or not
    """

    if gdir.rgi_region in ['05']:
        raise NotImplementedError('Millan 22 does not provide velocity '
                                  'products for Greenland - we would need to '
                                  'implement MEASURES in the shop for this.')

    # Find out which file(s) we need
    gdf = _get_lookup_velocity()
    cp = shpg.Point(gdir.cenlon, gdir.cenlat)
    sel = gdf.loc[gdf.contains(cp)]
    if len(sel) == 0:
        raise InvalidWorkflowError(f'There seems to be no Millan file for this '
                                   f'glacier: {gdir.rgi_id}')

    vel, files, grids = _filter_and_reproj(gdir, 'v', sel, allow_neg=False)
    if len(grids) == 0:
        raise RuntimeError('There is no velocity data for this glacier')
    if len(grids) > 1:
        raise RuntimeError('Multiple velocity grids - dont know what to do.')

    sel = sel.loc[sel.file_id == files[0]]
    vx, _, gridsx = _filter_and_reproj(gdir, 'vx', sel)
    vy, _, gridsy = _filter_and_reproj(gdir, 'vy', sel)

    dsx = gridsx[0]
    dsy = gridsy[0]
    grid_vel = dsx.grid
    proj_vel = grid_vel.proj
    grid_gla = gdir.grid

    # Get the coords at t0
    xx0, yy0 = grid_vel.center_grid.xy_coordinates

    # Compute coords at t1
    xx1 = _filter(dsx)
    yy1 = _filter(dsy)
    xx1 += xx0
    yy1 += yy0

    # Transform both to glacier proj
    xx0, yy0 = salem.transform_proj(proj_vel, grid_gla.proj, xx0, yy0)
    xx1, yy1 = salem.transform_proj(proj_vel, grid_gla.proj, xx1, yy1)

    # Compute velocities from there
    vx = xx1 - xx0
    vy = yy1 - yy0

    # And transform to local map
    vx = grid_gla.map_gridded_data(vx, grid=grid_vel, interp='linear')
    vy = grid_gla.map_gridded_data(vy, grid=grid_vel, interp='linear')

    # Scale back to match velocity
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        new_vel = np.sqrt(vx**2 + vy**2)
        p_ok = np.isfinite(new_vel) & (new_vel > 1)  # avoid div by zero
        scaler = vel[p_ok] / new_vel[p_ok]
        vx[p_ok] = vx[p_ok] * scaler
        vy[p_ok] = vy[p_ok] * scaler

    vx = vx.filled(np.nan)
    vy = vy.filled(np.nan)

    if add_error:
        err_vx, _, _ = _filter_and_reproj(gdir, 'err_vx', sel, allow_neg=False)
        err_vy, _, _ = _filter_and_reproj(gdir, 'err_vy', sel, allow_neg=False)
        err_vx[p_ok] = err_vx[p_ok] * scaler
        err_vy[p_ok] = err_vy[p_ok] * scaler

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'millan_v'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm'
        ln = 'Ice velocity from Millan et al. 2022'
        v.long_name = ln
        v.data_source = files[0]
        v[:] = vel

        vn = 'millan_vx'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm'
        ln = 'Ice velocity in map x direction from Millan et al. 2022'
        v.long_name = ln
        v.data_source = files[0]
        v[:] = vx

        vn = 'millan_vy'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm'
        ln = 'Ice velocity in map y direction from Millan et al. 2022'
        v.long_name = ln
        v.data_source = files[0]
        v[:] = vy

        if add_error:
            vn = 'millan_err_vx'
            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
            v.units = 'm'
            ln = 'Ice velocity error in map x direction from Millan et al. 2022'
            v.long_name = ln
            v.data_source = files[0]
            v[:] = err_vx

            vn = 'millan_err_vy'
            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
            v.units = 'm'
            ln = 'Ice velocity error in map y direction from Millan et al. 2022'
            v.long_name = ln
            v.data_source = files[0]
            v[:] = err_vy


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
            thick = ds['millan_ice_thickness'].where(ds['glacier_mask'], np.nan).load()
            with warnings.catch_warnings():
                # For operational runs we ignore the warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                d['millan_vol_km3'] = float(thick.sum() * gdir.grid.dx ** 2 * 1e-9)
                d['millan_area_km2'] = float((~thick.isnull()).sum() * gdir.grid.dx ** 2 * 1e-6)
                d['millan_perc_cov'] = float(d['millan_area_km2'] / gdir.rgi_area_km2)

                if 'millan_ice_thickness_err' in ds:
                    err = ds['millan_ice_thickness_err'].where(ds['glacier_mask'], np.nan).load()
                    d['millan_vol_err_km3'] = float(err.sum() * gdir.grid.dx ** 2 * 1e-9)
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            v = ds['millan_v'].where(ds['glacier_mask'], np.nan).load()
            with warnings.catch_warnings():
                # For operational runs we ignore the warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                d['millan_avg_vel'] = np.nanmean(v)
                d['millan_max_vel'] = np.nanmax(v)
                d['millan_vel_perc_cov'] = (float((~v.isnull()).sum() * gdir.grid.dx ** 2 * 1e-6) /
                                            gdir.rgi_area_km2)

                if 'millan_err_vx' in ds:
                    err_vx = ds['millan_err_vx'].where(ds['glacier_mask'], np.nan).load()
                    err_vy = ds['millan_err_vy'].where(ds['glacier_mask'], np.nan).load()
                    err = (err_vx**2 + err_vy**2)**0.5
                    d['millan_avg_err_vel'] = np.nanmean(err)

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
