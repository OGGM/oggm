""" Handling of the local glacier map and masks. Defines the first tasks
to be realized by any OGGM pre-processing workflow.

"""
# Built ins
import os
import logging
import json
import warnings
from functools import partial
from distutils.version import LooseVersion
# External libs
import salem
import pyproj
import numpy as np
import shapely.ops
import pandas as pd
import geopandas as gpd
import skimage.draw as skdraw
import shapely.geometry as shpg
import scipy.signal
from scipy.ndimage.measurements import label
from scipy.ndimage import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import griddata
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask as riomask
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool
# Locals
from oggm import entity_task
import oggm.cfg as cfg
from oggm.exceptions import (InvalidParamsError, InvalidGeometryError,
                             InvalidDEMError)
from oggm.utils import (tuple2int, get_topo_file, get_demo_file,
                        nicenumber, ncDataset)


# Module logger
log = logging.getLogger(__name__)

# Needed later
label_struct = np.ones((3, 3))

with open(get_demo_file('dem_sources.json'), 'r') as fr:
    DEM_SOURCE_INFO = json.loads(fr.read())


def gaussian_blur(in_array, size):
    """Applies a Gaussian filter to a 2d array.

    Parameters
    ----------
    in_array : numpy.array
        The array to smooth.
    size : int
        The half size of the smoothing window.

    Returns
    -------
    a smoothed numpy.array
    """

    # expand in_array to fit edge of kernel
    padded_array = np.pad(in_array, size, 'symmetric')

    # build kernel
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)

    # do the Gaussian blur
    return scipy.signal.fftconvolve(padded_array, g, mode='valid')


def multi_to_poly(geometry, gdir=None):
    """Sometimes an RGI geometry is a multipolygon: this should not happen.

    Parameters
    ----------
    geometry : shpg.Polygon or shpg.MultiPolygon
        the geometry to check
    gdir : GlacierDirectory, optional
        for logging

    Returns
    -------
    the corrected geometry
    """

    # Log
    rid = gdir.rgi_id + ': ' if gdir is not None else ''

    if 'Multi' in geometry.type:
        parts = np.array(geometry)
        for p in parts:
            assert p.type == 'Polygon'
        areas = np.array([p.area for p in parts])
        parts = parts[np.argsort(areas)][::-1]
        areas = areas[np.argsort(areas)][::-1]

        # First case (was RGIV4):
        # let's assume that one poly is exterior and that
        # the other polygons are in fact interiors
        exterior = parts[0].exterior
        interiors = []
        was_interior = 0
        for p in parts[1:]:
            if parts[0].contains(p):
                interiors.append(p.exterior)
                was_interior += 1
        if was_interior > 0:
            # We are done here, good
            geometry = shpg.Polygon(exterior, interiors)
        else:
            # This happens for bad geometries. We keep the largest
            geometry = parts[0]
            if np.any(areas[1:] > (areas[0] / 4)):
                log.warning('Geometry {} lost quite a chunk.'.format(rid))

    if geometry.type != 'Polygon':
        raise InvalidGeometryError('Geometry {} is not a Polygon.'.format(rid))
    return geometry


def _interp_polygon(polygon, dx):
    """Interpolates an irregular polygon to a regular step dx.

    Interior geometries are also interpolated if they are longer then 3*dx,
    otherwise they are ignored.

    Parameters
    ----------
    polygon: The shapely.geometry.Polygon instance to interpolate
    dx : the step (float)

    Returns
    -------
    an interpolated shapely.geometry.Polygon class instance.
    """

    # remove last (duplex) point to build a LineString from the LinearRing
    line = shpg.LineString(np.asarray(polygon.exterior.xy).T)

    e_line = []
    for distance in np.arange(0.0, line.length, dx):
        e_line.append(*line.interpolate(distance).coords)
    e_line = shpg.LinearRing(e_line)

    i_lines = []
    for ipoly in polygon.interiors:
        line = shpg.LineString(np.asarray(ipoly.xy).T)
        if line.length < 3*dx:
            continue
        i_points = []
        for distance in np.arange(0.0, line.length, dx):
            i_points.append(*line.interpolate(distance).coords)
        i_lines.append(shpg.LinearRing(i_points))

    return shpg.Polygon(e_line, i_lines)


def _polygon_to_pix(polygon):
    """Transforms polygon coordinates to integer pixel coordinates. It makes
    the geometry easier to handle and reduces the number of points.

    Parameters
    ----------
    polygon: the shapely.geometry.Polygon instance to transform.

    Returns
    -------
    a shapely.geometry.Polygon class instance.
    """

    def project(x, y):
        return np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)

    poly_pix = shapely.ops.transform(project, polygon)

    # simple trick to correct invalid polys:
    tmp = poly_pix.buffer(0)

    # sometimes the glacier gets cut out in parts
    if tmp.type == 'MultiPolygon':
        # If only small arms are cut out, remove them
        area = np.array([_tmp.area for _tmp in tmp])
        _tokeep = np.argmax(area).item()
        tmp = tmp[_tokeep]

        # check that the other parts really are small,
        # otherwise replace tmp with something better
        area = area / area[_tokeep]
        for _a in area:
            if _a != 1 and _a > 0.05:
                # these are extremely thin glaciers
                # eg. RGI40-11.01381 RGI40-11.01697 params.d1 = 5. and d2 = 8.
                # make them bigger until its ok
                for b in np.arange(0., 1., 0.01):
                    tmp = shapely.ops.transform(project, polygon.buffer(b))
                    tmp = tmp.buffer(0)
                    if tmp.type == 'MultiPolygon':
                        continue
                    if tmp.is_valid:
                        break
                if b == 0.99:
                    raise InvalidGeometryError('This glacier geometry is not '
                                               'valid.')

    if not tmp.is_valid:
        raise InvalidGeometryError('This glacier geometry is not valid.')

    return tmp


@entity_task(log, writes=['glacier_grid', 'dem', 'outlines'])
def define_glacier_region(gdir, entity=None):
    """Very first task: define the glacier's local grid.

    Defines the local projection (Transverse Mercator), centered on the
    glacier. There is some options to set the resolution of the local grid.
    It can be adapted depending on the size of the glacier with::

        dx (m) = d1 * AREA (km) + d2 ; clipped to dmax

    or be set to a fixed value. See ``params.cfg`` for setting these options.
    Default values of the adapted mode lead to a resolution of 50 m for
    Hintereisferner, which is approx. 8 km2 large.
    After defining the grid, the topography and the outlines of the glacier
    are transformed into the local projection. The default interpolation for
    the topography is `cubic`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    entity : geopandas.GeoSeries
        the glacier geometry to process
    """

    # Make a local glacier map
    proj_params = dict(name='tmerc', lat_0=0., lon_0=gdir.cenlon,
                       k=0.9996, x_0=0, y_0=0, datum='WGS84')
    proj4_str = "+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} " \
                "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**proj_params)
    proj_in = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
    proj_out = pyproj.Proj(proj4_str, preserve_units=True)
    project = partial(pyproj.transform, proj_in, proj_out)
    # transform geometry to map
    geometry = shapely.ops.transform(project, entity['geometry'])
    geometry = multi_to_poly(geometry, gdir=gdir)
    xx, yy = geometry.exterior.xy

    # Save transformed geometry to disk
    entity = entity.copy()
    entity['geometry'] = geometry
    # Avoid fiona bug: https://github.com/Toblerity/Fiona/issues/365
    for k, s in entity.iteritems():
        if type(s) in [np.int32, np.int64]:
            entity[k] = int(s)
    towrite = gpd.GeoDataFrame(entity).T
    towrite.crs = proj4_str
    # Delete the source before writing
    if 'DEM_SOURCE' in towrite:
        del towrite['DEM_SOURCE']

    # Define glacier area to use
    area = entity['Area']

    # Do we want to use the RGI area or ours?
    if not cfg.PARAMS['use_rgi_area']:
        area = geometry.area * 1e-6
        entity['Area'] = area
        towrite['Area'] = area

    # Write shapefile
    gdir.write_shapefile(towrite, 'outlines')

    # Also transform the intersects if necessary
    gdf = cfg.PARAMS['intersects_gdf']
    if len(gdf) > 0:
        gdf = gdf.loc[((gdf.RGIId_1 == gdir.rgi_id) |
                       (gdf.RGIId_2 == gdir.rgi_id))]
        if len(gdf) > 0:
            gdf = salem.transform_geopandas(gdf, to_crs=proj_out)
            if hasattr(gdf.crs, 'srs'):
                # salem uses pyproj
                gdf.crs = gdf.crs.srs
            gdir.write_shapefile(gdf, 'intersects')
    else:
        # Sanity check
        if cfg.PARAMS['use_intersects']:
            raise InvalidParamsError('You seem to have forgotten to set the '
                                     'intersects file for this run. OGGM '
                                     'works better with such a file. If you '
                                     'know what your are doing, set '
                                     "cfg.PARAMS['use_intersects'] = False to "
                                     "suppress this error.")

    # 6. choose a spatial resolution with respect to the glacier area
    dxmethod = cfg.PARAMS['grid_dx_method']
    if dxmethod == 'linear':
        dx = np.rint(cfg.PARAMS['d1'] * area + cfg.PARAMS['d2'])
    elif dxmethod == 'square':
        dx = np.rint(cfg.PARAMS['d1'] * np.sqrt(area) + cfg.PARAMS['d2'])
    elif dxmethod == 'fixed':
        dx = np.rint(cfg.PARAMS['fixed_dx'])
    else:
        raise InvalidParamsError('grid_dx_method not supported: {}'
                                 .format(dxmethod))
    # Additional trick for varying dx
    if dxmethod in ['linear', 'square']:
        dx = np.clip(dx, cfg.PARAMS['d2'], cfg.PARAMS['dmax'])

    log.debug('(%s) area %.2f km, dx=%.1f', gdir.rgi_id, area, dx)

    # Safety check
    border = cfg.PARAMS['border']
    if border > 1000:
        raise InvalidParamsError("You have set a cfg.PARAMS['border'] value "
                                 "of {}. ".format(cfg.PARAMS['border']) +
                                 'This a very large value, which is '
                                 'currently not supported in OGGM.')

    # For tidewater glaciers we force border to 10
    if gdir.is_tidewater and cfg.PARAMS['clip_tidewater_border']:
        border = 10

    # Corners, incl. a buffer of N pix
    ulx = np.min(xx) - border * dx
    lrx = np.max(xx) + border * dx
    uly = np.max(yy) + border * dx
    lry = np.min(yy) - border * dx
    # n pixels
    nx = np.int((lrx - ulx) / dx)
    ny = np.int((uly - lry) / dx)

    # Back to lon, lat for DEM download/preparation
    tmp_grid = salem.Grid(proj=proj_out, nxny=(nx, ny), x0y0=(ulx, uly),
                          dxdy=(dx, -dx), pixel_ref='corner')
    minlon, maxlon, minlat, maxlat = tmp_grid.extent_in_crs(crs=salem.wgs84)

    # Open DEM
    source = entity.DEM_SOURCE if hasattr(entity, 'DEM_SOURCE') else None
    dem_list, dem_source = get_topo_file((minlon, maxlon), (minlat, maxlat),
                                         rgi_region=gdir.rgi_region,
                                         rgi_subregion=gdir.rgi_subregion,
                                         source=source)
    log.debug('(%s) DEM source: %s', gdir.rgi_id, dem_source)
    log.debug('(%s) N DEM Files: %s', gdir.rgi_id, len(dem_list))

    # A glacier area can cover more than one tile:
    if len(dem_list) == 1:
        dem_dss = [rasterio.open(dem_list[0])]  # if one tile, just open it
        dem_data = rasterio.band(dem_dss[0], 1)
        if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
    else:
        dem_dss = [rasterio.open(s) for s in dem_list]  # list of rasters
        dem_data, src_transform = merge_tool(dem_dss)  # merged rasters

    # Use Grid properties to create a transform (see rasterio cookbook)
    dst_transform = rasterio.transform.from_origin(
        ulx, uly, dx, dx  # sign change (2nd dx) is done by rasterio.transform
    )

    # Set up profile for writing output
    profile = dem_dss[0].profile
    profile.update({
        'crs': proj4_str,
        'transform': dst_transform,
        'width': nx,
        'height': ny
    })

    # Could be extended so that the cfg file takes all Resampling.* methods
    if cfg.PARAMS['topo_interp'] == 'bilinear':
        resampling = Resampling.bilinear
    elif cfg.PARAMS['topo_interp'] == 'cubic':
        resampling = Resampling.cubic
    else:
        raise InvalidParamsError('{} interpolation not understood'
                                 .format(cfg.PARAMS['topo_interp']))

    dem_reproj = gdir.get_filepath('dem')
    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)
    profile.pop('compress', None)
    nodata = dem_dss[0].meta.get('nodata', None)
    if source == 'TANDEM' and nodata is None:
        # badly tagged geotiffs, let's do it ourselves
        nodata = -32767
    with rasterio.open(dem_reproj, 'w', **profile) as dest:
        dst_array = np.empty((ny, nx), dtype=dem_dss[0].dtypes[0])
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=dem_dss[0].crs,
            src_transform=src_transform,
            src_nodata=nodata,
            # Destination parameters
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=proj4_str,
            dst_nodata=nodata,
            # Configuration
            resampling=resampling)

        dest.write(dst_array, 1)

    for dem_ds in dem_dss:
        dem_ds.close()

    # Glacier grid
    x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
    glacier_grid = salem.Grid(proj=proj_out, nxny=(nx, ny),  dxdy=(dx, -dx),
                              x0y0=x0y0)
    glacier_grid.to_json(gdir.get_filepath('glacier_grid'))

    # Write DEM source info
    gdir.add_to_diagnostics('dem_source', dem_source)
    source_txt = DEM_SOURCE_INFO.get(dem_source, dem_source)
    with open(gdir.get_filepath('dem_source'), 'w') as fw:
        fw.write(source_txt)
        fw.write('\n\n')
        fw.write('# Data files\n\n')
        for fname in dem_list:
            fw.write('{}\n'.format(os.path.basename(fname)))


@entity_task(log, writes=['gridded_data', 'geometries'])
def glacier_masks(gdir):
    """Makes a gridded mask of the glacier outlines and topography.

    This function fills holes in the source DEM and produces smoothed gridded
    topography and glacier outline arrays. These are the ones which will later
    be used to determine bed and surface height.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # open srtm tif-file:
    dem_dr = rasterio.open(gdir.get_filepath('dem'), 'r', driver='GTiff')
    dem = dem_dr.read(1).astype(rasterio.float32)

    # Grid
    nx = dem_dr.width
    ny = dem_dr.height
    assert nx == gdir.grid.nx
    assert ny == gdir.grid.ny

    # Correct the DEM
    # Currently we just do a linear interp -- filling is totally shit anyway
    min_z = -999.
    dem[dem <= min_z] = np.NaN
    isfinite = np.isfinite(dem)
    if np.all(~isfinite):
        raise InvalidDEMError('Not a single valid grid point in DEM')
    if np.any(~isfinite):
        xx, yy = gdir.grid.ij_coordinates
        pnan = np.nonzero(~isfinite)
        pok = np.nonzero(isfinite)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        try:
            dem[pnan] = griddata(points, np.ravel(dem[pok]), inter,
                                 method='linear')
        except ValueError:
            raise InvalidDEMError('DEM interpolation not possible.')
        log.warning(gdir.rgi_id + ': DEM needed interpolation.')
        gdir.add_to_diagnostics('dem_needed_interpolation', True)
        gdir.add_to_diagnostics('dem_invalid_perc', len(pnan[0]) / (nx*ny))

    isfinite = np.isfinite(dem)
    if np.any(~isfinite):
        # this happens when extrapolation is needed
        # see how many percent of the dem
        if np.sum(~isfinite) > (0.5 * nx * ny):
            log.warning('({}) many NaNs in DEM'.format(gdir.rgi_id))
        xx, yy = gdir.grid.ij_coordinates
        pnan = np.nonzero(~isfinite)
        pok = np.nonzero(isfinite)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        try:
            dem[pnan] = griddata(points, np.ravel(dem[pok]), inter,
                                 method='nearest')
        except ValueError:
            raise InvalidDEMError('DEM extrapolation not possible.')
        log.warning(gdir.rgi_id + ': DEM needed extrapolation.')
        gdir.add_to_diagnostics('dem_needed_extrapolation', True)
        gdir.add_to_diagnostics('dem_extrapol_perc', len(pnan[0]) / (nx*ny))

    if np.min(dem) == np.max(dem):
        raise InvalidDEMError('({}) min equal max in the DEM.'
                              .format(gdir.rgi_id))

    # Projection
    if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
        transf = dem_dr.transform
    else:
        transf = dem_dr.affine
    x0 = transf[2]  # UL corner
    y0 = transf[5]  # UL corner
    dx = transf[0]
    dy = transf[4]  # Negative

    if not (np.allclose(dx, -dy) or np.allclose(dx, gdir.grid.dx) or
            np.allclose(y0, gdir.grid.corner_grid.y0, atol=1e-2) or
            np.allclose(x0, gdir.grid.corner_grid.x0, atol=1e-2)):
        raise InvalidDEMError('DEM file and Salem Grid do not match!')
    dem_dr.close()

    # Clip topography to 0 m a.s.l.
    dem = dem.clip(0)

    # Smooth DEM?
    if cfg.PARAMS['smooth_window'] > 0.:
        gsize = np.rint(cfg.PARAMS['smooth_window'] / dx)
        smoothed_dem = gaussian_blur(dem, np.int(gsize))
    else:
        smoothed_dem = dem.copy()

    if not np.all(np.isfinite(smoothed_dem)):
        raise InvalidDEMError('({}) NaN in smoothed DEM'.format(gdir.rgi_id))

    # Geometries
    geometry = gdir.read_shapefile('outlines').geometry[0]

    # Interpolate shape to a regular path
    glacier_poly_hr = _interp_polygon(geometry, gdir.grid.dx)

    # Transform geometry into grid coordinates
    # It has to be in pix center coordinates because of how skimage works
    def proj(x, y):
        grid = gdir.grid.center_grid
        return grid.transform(x, y, crs=grid.proj)
    glacier_poly_hr = shapely.ops.transform(proj, glacier_poly_hr)

    # simple trick to correct invalid polys:
    # http://stackoverflow.com/questions/20833344/
    # fix-invalid-polygon-python-shapely
    glacier_poly_hr = glacier_poly_hr.buffer(0)
    if not glacier_poly_hr.is_valid:
        raise InvalidGeometryError('This glacier geometry is not valid.')

    # Rounded nearest pix
    glacier_poly_pix = _polygon_to_pix(glacier_poly_hr)
    if glacier_poly_pix.exterior is None:
        raise InvalidGeometryError('Problem in converting glacier geometry '
                                   'to grid resolution.')

    # Compute the glacier mask (currently: center pixels + touched)
    nx, ny = gdir.grid.nx, gdir.grid.ny
    glacier_mask = np.zeros((ny, nx), dtype=np.uint8)
    glacier_ext = np.zeros((ny, nx), dtype=np.uint8)
    (x, y) = glacier_poly_pix.exterior.xy
    glacier_mask[skdraw.polygon(np.array(y), np.array(x))] = 1
    for gint in glacier_poly_pix.interiors:
        x, y = tuple2int(gint.xy)
        glacier_mask[skdraw.polygon(y, x)] = 0
        glacier_mask[y, x] = 0  # on the nunataks, no
    x, y = tuple2int(glacier_poly_pix.exterior.xy)
    glacier_mask[y, x] = 1
    glacier_ext[y, x] = 1

    # Because of the 0 values at nunataks boundaries, some "Ice Islands"
    # can happen within nunataks (e.g.: RGI40-11.00062)
    # See if we can filter them out easily
    regions, nregions = label(glacier_mask, structure=label_struct)
    if nregions > 1:
        log.debug('(%s) we had to cut an island in the mask', gdir.rgi_id)
        # Check the size of those
        region_sizes = [np.sum(regions == r) for r in np.arange(1, nregions+1)]
        am = np.argmax(region_sizes)
        # Check not a strange glacier
        sr = region_sizes.pop(am)
        for ss in region_sizes:
            assert (ss / sr) < 0.1
        glacier_mask[:] = 0
        glacier_mask[np.where(regions == (am+1))] = 1

    # Last sanity check based on the masked dem
    tmp_max = np.max(dem[np.where(glacier_mask == 1)])
    tmp_min = np.min(dem[np.where(glacier_mask == 1)])
    if tmp_max < (tmp_min + 1):
        raise InvalidDEMError('({}) min equal max in the masked DEM.'
                              .format(gdir.rgi_id))

    # write out the grids in the netcdf file
    nc = gdir.create_gridded_ncdf_file('gridded_data')

    v = nc.createVariable('topo', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = 'DEM topography'
    v[:] = dem

    v = nc.createVariable('topo_smoothed', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = ('DEM topography smoothed '
                   'with radius: {:.1} m'.format(cfg.PARAMS['smooth_window']))
    v[:] = smoothed_dem

    v = nc.createVariable('glacier_mask', 'i1', ('y', 'x', ), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier mask'
    v[:] = glacier_mask

    v = nc.createVariable('glacier_ext', 'i1', ('y', 'x', ), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier external boundaries'
    v[:] = glacier_ext

    # add some meta stats and close
    nc.max_h_dem = np.max(dem)
    nc.min_h_dem = np.min(dem)
    dem_on_g = dem[np.where(glacier_mask)]
    nc.max_h_glacier = np.max(dem_on_g)
    nc.min_h_glacier = np.min(dem_on_g)
    nc.close()

    geometries = dict()
    geometries['polygon_hr'] = glacier_poly_hr
    geometries['polygon_pix'] = glacier_poly_pix
    geometries['polygon_area'] = geometry.area
    gdir.write_pickle(geometries, 'geometries')


@entity_task(log, writes=['gridded_data', 'hypsometry'])
def simple_glacier_masks(gdir):
    """Compute glacier masks based on much simpler rules than OGGM's default.

    This is therefore more robust: we use this function to compute glacier
    hypsometries.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # open srtm tif-file:
    dem_dr = rasterio.open(gdir.get_filepath('dem'), 'r', driver='GTiff')
    dem = dem_dr.read(1).astype(rasterio.float32)

    # Grid
    nx = dem_dr.width
    ny = dem_dr.height
    assert nx == gdir.grid.nx
    assert ny == gdir.grid.ny

    # Correct the DEM
    # Currently we just do a linear interp -- filling is totally shit anyway
    min_z = -999.
    dem[dem <= min_z] = np.NaN
    isfinite = np.isfinite(dem)
    if np.all(~isfinite):
        raise InvalidDEMError('Not a single valid grid point in DEM')
    if np.any(~isfinite):
        xx, yy = gdir.grid.ij_coordinates
        pnan = np.nonzero(~isfinite)
        pok = np.nonzero(isfinite)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        try:
            dem[pnan] = griddata(points, np.ravel(dem[pok]), inter,
                                 method='linear')
        except ValueError:
            raise InvalidDEMError('DEM interpolation not possible.')
        log.warning(gdir.rgi_id + ': DEM needed interpolation.')
        gdir.add_to_diagnostics('dem_needed_interpolation', True)
        gdir.add_to_diagnostics('dem_invalid_perc', len(pnan[0]) / (nx*ny))

    isfinite = np.isfinite(dem)
    if np.any(~isfinite):
        # this happens when extrapolation is needed
        # see how many percent of the dem
        if np.sum(~isfinite) > (0.5 * nx * ny):
            log.warning('({}) many NaNs in DEM'.format(gdir.rgi_id))
        xx, yy = gdir.grid.ij_coordinates
        pnan = np.nonzero(~isfinite)
        pok = np.nonzero(isfinite)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        try:
            dem[pnan] = griddata(points, np.ravel(dem[pok]), inter,
                                 method='nearest')
        except ValueError:
            raise InvalidDEMError('DEM extrapolation not possible.')
        log.warning(gdir.rgi_id + ': DEM needed extrapolation.')
        gdir.add_to_diagnostics('dem_needed_extrapolation', True)
        gdir.add_to_diagnostics('dem_extrapol_perc', len(pnan[0]) / (nx*ny))

    if np.min(dem) == np.max(dem):
        raise InvalidDEMError('({}) min equal max in the DEM.'
                              .format(gdir.rgi_id))

    # Proj
    if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
        transf = dem_dr.transform
    else:
        raise ImportError('This task needs rasterio >= 1.0 to work properly')
    x0 = transf[2]  # UL corner
    y0 = transf[5]  # UL corner
    dx = transf[0]
    dy = transf[4]  # Negative
    assert dx == -dy
    assert dx == gdir.grid.dx
    assert y0 == gdir.grid.corner_grid.y0
    assert x0 == gdir.grid.corner_grid.x0

    profile = dem_dr.profile
    dem_dr.close()

    # Clip topography to 0 m a.s.l.
    dem = dem.clip(0)

    # Smooth DEM?
    if cfg.PARAMS['smooth_window'] > 0.:
        gsize = np.rint(cfg.PARAMS['smooth_window'] / dx)
        smoothed_dem = gaussian_blur(dem, np.int(gsize))
    else:
        smoothed_dem = dem.copy()

    if not np.all(np.isfinite(smoothed_dem)):
        raise InvalidDEMError('({}) NaN in smoothed DEM'.format(gdir.rgi_id))

    # Geometries
    geometry = gdir.read_shapefile('outlines').geometry[0]

    # simple trick to correct invalid polys:
    # http://stackoverflow.com/questions/20833344/
    # fix-invalid-polygon-python-shapely
    geometry = geometry.buffer(0)
    if not geometry.is_valid:
        raise InvalidDEMError('This glacier geometry is not valid.')

    # Compute the glacier mask using rasterio
    # Small detour as mask only accepts DataReader objects
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(dem.astype(np.int16)[np.newaxis, ...])
        dem_data = rasterio.open(memfile.name)
        masked_dem, _ = riomask(dem_data, [shpg.mapping(geometry)],
                                filled=False)
    glacier_mask = ~masked_dem[0, ...].mask

    # Smame without nunataks
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(dem.astype(np.int16)[np.newaxis, ...])
        dem_data = rasterio.open(memfile.name)
        poly = shpg.mapping(shpg.Polygon(geometry.exterior))
        masked_dem, _ = riomask(dem_data, [poly],
                                filled=False)
    glacier_mask_nonuna = ~masked_dem[0, ...].mask

    # Glacier exterior excluding nunataks
    erode = binary_erosion(glacier_mask_nonuna)
    glacier_ext = glacier_mask_nonuna ^ erode
    glacier_ext = np.where(glacier_mask_nonuna, glacier_ext, 0)

    # Last sanity check based on the masked dem
    tmp_max = np.max(dem[glacier_mask])
    tmp_min = np.min(dem[glacier_mask])
    if tmp_max < (tmp_min + 1):
        raise InvalidDEMError('({}) min equal max in the masked DEM.'
                              .format(gdir.rgi_id))

    # hypsometry
    bsize = 50.
    dem_on_ice = dem[glacier_mask]
    bins = np.arange(nicenumber(dem_on_ice.min(), bsize, lower=True),
                     nicenumber(dem_on_ice.max(), bsize) + 0.01, bsize)

    h, _ = np.histogram(dem_on_ice, bins)
    h = h / np.sum(h) * 1000  # in permil

    # We want to convert the bins to ints but preserve their sum to 1000
    # Start with everything rounded down, then round up the numbers with the
    # highest fractional parts until the desired sum is reached.
    hi = np.floor(h).astype(np.int)
    hup = np.ceil(h).astype(np.int)
    aso = np.argsort(hup - h)
    for i in aso:
        hi[i] = hup[i]
        if np.sum(hi) == 1000:
            break

    # slope
    sy, sx = np.gradient(dem, dx)
    aspect = np.arctan2(np.mean(-sx[glacier_mask]), np.mean(sy[glacier_mask]))
    aspect = np.rad2deg(aspect)
    if aspect < 0:
        aspect += 360
    slope = np.arctan(np.sqrt(sx ** 2 + sy ** 2))
    avg_slope = np.rad2deg(np.mean(slope[glacier_mask]))

    # write
    df = pd.DataFrame()
    df['RGIId'] = [gdir.rgi_id]
    df['GLIMSId'] = [gdir.glims_id]
    df['Zmin'] = [dem_on_ice.min()]
    df['Zmax'] = [dem_on_ice.max()]
    df['Zmed'] = [np.median(dem_on_ice)]
    df['Area'] = [gdir.rgi_area_km2]
    df['Slope'] = [avg_slope]
    df['Aspect'] = [aspect]
    for b, bs in zip(hi, (bins[1:] + bins[:-1])/2):
        df['{}'.format(np.round(bs).astype(int))] = [b]
    df.to_csv(gdir.get_filepath('hypsometry'), index=False)

    # write out the grids in the netcdf file
    nc = gdir.create_gridded_ncdf_file('gridded_data')

    v = nc.createVariable('topo', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = 'DEM topography'
    v[:] = dem

    v = nc.createVariable('topo_smoothed', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = ('DEM topography smoothed '
                   'with radius: {:.1} m'.format(cfg.PARAMS['smooth_window']))
    v[:] = smoothed_dem

    v = nc.createVariable('glacier_mask', 'i1', ('y', 'x', ), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier mask'
    v[:] = glacier_mask

    v = nc.createVariable('glacier_ext', 'i1', ('y', 'x', ), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier external boundaries'
    v[:] = glacier_ext

    # add some meta stats and close
    nc.max_h_dem = np.max(dem)
    nc.min_h_dem = np.min(dem)
    dem_on_g = dem[np.where(glacier_mask)]
    nc.max_h_glacier = np.max(dem_on_g)
    nc.min_h_glacier = np.min(dem_on_g)
    nc.close()


@entity_task(log, writes=['gridded_data'])
def interpolation_masks(gdir):
    """Computes the glacier exterior masks taking ice divides into account.

    This is useful for distributed ice thickness. The masks are added to the
    gridded data file. For convenience we also add a slope mask.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    with ncDataset(grids_file) as nc:
        topo_smoothed = nc.variables['topo_smoothed'][:]
        glacier_mask = nc.variables['glacier_mask'][:]

    # Glacier exterior including nunataks
    erode = binary_erosion(glacier_mask)
    glacier_ext = glacier_mask ^ erode
    glacier_ext = np.where(glacier_mask == 1, glacier_ext, 0)

    # Intersects between glaciers
    gdfi = gpd.GeoDataFrame(columns=['geometry'])
    if gdir.has_file('intersects'):
        # read and transform to grid
        gdf = gdir.read_shapefile('intersects')
        salem.transform_geopandas(gdf, gdir.grid, inplace=True)
        gdfi = pd.concat([gdfi, gdf[['geometry']]])

    # Ice divide mask
    # Probably not the fastest way to do this, but it works
    dist = np.array([])
    jj, ii = np.where(glacier_ext)
    for j, i in zip(jj, ii):
        dist = np.append(dist, np.min(gdfi.distance(shpg.Point(i, j))))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pok = np.where(dist <= 1)
    glacier_ext_intersect = glacier_ext * 0
    glacier_ext_intersect[jj[pok], ii[pok]] = 1

    # Distance from border mask - Scipy does the job
    dx = gdir.grid.dx
    dis_from_border = 1 + glacier_ext_intersect - glacier_ext
    dis_from_border = distance_transform_edt(dis_from_border) * dx

    # Slope
    glen_n = cfg.PARAMS['glen_n']
    sy, sx = np.gradient(topo_smoothed, dx, dx)
    slope = np.arctan(np.sqrt(sy**2 + sx**2))
    slope = np.clip(slope, np.deg2rad(cfg.PARAMS['min_slope']*4), np.pi/2.)
    slope = 1 / slope**(glen_n / (glen_n+2))

    with ncDataset(grids_file, 'a') as nc:

        vn = 'glacier_ext_erosion'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'i1', ('y', 'x', ))
        v.units = '-'
        v.long_name = 'Glacier exterior with binary erosion method'
        v[:] = glacier_ext

        vn = 'ice_divides'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'i1', ('y', 'x', ))
        v.units = '-'
        v.long_name = 'Glacier ice divides'
        v[:] = glacier_ext_intersect

        vn = 'slope_factor'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ))
        v.units = '-'
        v.long_name = 'Slope factor as defined in Farinotti et al 2009'
        v[:] = slope

        vn = 'dis_from_border'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ))
        v.units = 'm'
        v.long_name = 'Distance from border'
        v[:] = dis_from_border


def merged_glacier_masks(gdir, geometry):
    """Makes a gridded mask of a merged glacier outlines.

    This is a simplified version of glacier_masks. We don't need fancy
    corrections or smoothing here: The flowlines for the actual model run are
    based on a proper call of glacier_masks.

    This task is only to get outlines etc. for visualization!

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    geometry: shapely.geometry.multipolygon.MultiPolygon
        united outlines of the merged glaciers
    """

    # open srtm tif-file:
    dem_dr = rasterio.open(gdir.get_filepath('dem'), 'r', driver='GTiff')
    dem = dem_dr.read(1).astype(rasterio.float32)

    # Grid
    nx = dem_dr.width
    ny = dem_dr.height
    assert nx == gdir.grid.nx
    assert ny == gdir.grid.ny

    if np.min(dem) == np.max(dem):
        raise RuntimeError('({}) min equal max in the DEM.'
                           .format(gdir.rgi_id))

    # Projection
    if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
        transf = dem_dr.transform
    else:
        transf = dem_dr.affine
    x0 = transf[2]  # UL corner
    y0 = transf[5]  # UL corner
    dx = transf[0]
    dy = transf[4]  # Negative

    if not (np.allclose(dx, -dy) or np.allclose(dx, gdir.grid.dx) or
            np.allclose(y0, gdir.grid.corner_grid.y0, atol=1e-2) or
            np.allclose(x0, gdir.grid.corner_grid.x0, atol=1e-2)):
        raise RuntimeError('DEM file and Salem Grid do not match!')
    dem_dr.close()

    # Clip topography to 0 m a.s.l.
    dem = dem.clip(0)

    # Interpolate shape to a regular path
    glacier_poly_hr = list(geometry)
    for nr, poly in enumerate(glacier_poly_hr):
        # transform geometry to map
        _geometry = salem.transform_geometry(poly, to_crs=gdir.grid.proj)
        glacier_poly_hr[nr] = _interp_polygon(_geometry, gdir.grid.dx)

    glacier_poly_hr = shpg.MultiPolygon(glacier_poly_hr)

    # Transform geometry into grid coordinates
    # It has to be in pix center coordinates because of how skimage works
    def proj(x, y):
        grid = gdir.grid.center_grid
        return grid.transform(x, y, crs=grid.proj)

    glacier_poly_hr = shapely.ops.transform(proj, glacier_poly_hr)

    # simple trick to correct invalid polys:
    # http://stackoverflow.com/questions/20833344/
    # fix-invalid-polygon-python-shapely
    glacier_poly_hr = glacier_poly_hr.buffer(0)
    if not glacier_poly_hr.is_valid:
        raise RuntimeError('This glacier geometry is not valid.')

    # Rounded geometry to nearest nearest pix
    # I can not use _polyg
    # glacier_poly_pix = _polygon_to_pix(glacier_poly_hr)
    def project(x, y):
        return np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)

    glacier_poly_pix = shapely.ops.transform(project, glacier_poly_hr)

    # Compute the glacier mask (currently: center pixels + touched)
    nx, ny = gdir.grid.nx, gdir.grid.ny
    glacier_mask = np.zeros((ny, nx), dtype=np.uint8)
    glacier_ext = np.zeros((ny, nx), dtype=np.uint8)

    for poly in glacier_poly_pix:
        (x, y) = poly.exterior.xy
        glacier_mask[skdraw.polygon(np.array(y), np.array(x))] = 1
        for gint in poly.interiors:
            x, y = tuple2int(gint.xy)
            glacier_mask[skdraw.polygon(y, x)] = 0
            glacier_mask[y, x] = 0  # on the nunataks, no
        x, y = tuple2int(poly.exterior.xy)
        glacier_mask[y, x] = 1
        glacier_ext[y, x] = 1

    # Last sanity check based on the masked dem
    tmp_max = np.max(dem[np.where(glacier_mask == 1)])
    tmp_min = np.min(dem[np.where(glacier_mask == 1)])
    if tmp_max < (tmp_min + 1):
        raise RuntimeError('({}) min equal max in the masked DEM.'
                           .format(gdir.rgi_id))

    # write out the grids in the netcdf file
    nc = gdir.create_gridded_ncdf_file('gridded_data')

    v = nc.createVariable('topo', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = 'DEM topography'
    v[:] = dem

    v = nc.createVariable('glacier_mask', 'i1', ('y', 'x', ), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier mask'
    v[:] = glacier_mask

    v = nc.createVariable('glacier_ext', 'i1', ('y', 'x', ), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier external boundaries'
    v[:] = glacier_ext

    # add some meta stats and close
    nc.max_h_dem = np.max(dem)
    nc.min_h_dem = np.min(dem)
    dem_on_g = dem[np.where(glacier_mask)]
    nc.max_h_glacier = np.max(dem_on_g)
    nc.min_h_glacier = np.min(dem_on_g)
    nc.close()

    geometries = dict()
    geometries['polygon_hr'] = glacier_poly_hr
    geometries['polygon_pix'] = glacier_poly_pix
    geometries['polygon_area'] = geometry.area
    gdir.write_pickle(geometries, 'geometries')
