import logging

import numpy as np

try:
    import salem
except ImportError:
    pass
try:
    import rasterio
except ImportError:
    pass

from oggm import utils
from oggm.exceptions import InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

default_base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/icevol/composite/'


@utils.entity_task(log)
def reproject_thickness(gdir, base_url=None):
    """Dummy docs

    Parameters
    ----------
    gdir
    dem_source
    """

    if base_url is None:
        base_url = default_base_url
    if not base_url.endswith('/'):
        base_url += '/'

    rgi_str = gdir.rgi_id
    rgi_reg_str = rgi_str[:8]

    url = base_url + rgi_reg_str  + '/' + rgi_str + '_thickness.tif'
    input_file = utils.file_downloader(url)

    base_dir = os.path.join(output_dir, 'thickness', rgi_reg_str)
    mkdir(base_dir)
    output_file = os.path.join(base_dir, rgi_str + '_thickness.tif')
    if os.path.exists(output_file):
        os.remove(output_file)

    with rasterio.open(input_file) as src:

        kwargs = src.meta.copy()
        data = src.read(1)
        in_volume = np.sum(data * src.transform.a**2)

        with rasterio.open(gdir.get_filepath('dem')) as tpl:

            kwargs.update({
                'crs': tpl.crs,
                'transform': tpl.transform,
                'width': tpl.width,
                'height': tpl.height
            })

            with rasterio.open(output_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):

                    dest = np.zeros(shape=(tpl.height, tpl.width), dtype=data.dtype)

                    reproject(
                        source=rasterio.band(src, i),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tpl.transform,
                        dst_crs=tpl.crs,
                        resampling=Resampling.bilinear)

                    # Correct for volume
                    dest = clip_min(dest, 0)
                    out_volume = np.sum(dest * dst.transform.a**2)
                    if in_volume > 0:
                        dest = dest * out_volume / in_volume

                    dst.write(dest, indexes=i)


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
    p_ok = new_vel > 1e-5  # avoid div by zero
    vx[p_ok] = vx[p_ok] * orig_vel[p_ok] / new_vel[p_ok]
    vy[p_ok] = vy[p_ok] * orig_vel[p_ok] / new_vel[p_ok]

    # And transform to local map
    vx = grid_gla.map_gridded_data(vx, grid=grid_vel, interp='linear')
    vy = grid_gla.map_gridded_data(vy, grid=grid_vel, interp='linear')

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        vn = 'obs_icevel_x'
        if do_error:
            vn = vn.replace('obs', 'err')
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

        vn = 'obs_icevel_y'
        if do_error:
            vn = vn.replace('obs', 'err')
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


@utils.entity_task(log, writes=['gridded_data'])
def velocity_to_gdir(gdir, add_error=False):
    """Reproject the its_live files to the given glacier directory.

    Variables are added to the gridded_data nc file.

    Reprojecting velocities from one map proj to another is done
    reprojecting the vector distances. In this process, absolute velocities
    might change as well because map projections do not always preserve
    distances -> we scale them back to the original velocities as per the
    ITS_LIVE documentation that states that velocities are given in
    ground units, i.e. absolute velocities.

    We use bilinear interpolation to reproject the velocities to the local
    glacier map.

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
