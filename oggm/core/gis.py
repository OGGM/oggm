""" Handling of the local glacier map and masks. Defines the first tasks
to be realized by any OGGM pre-processing workflow.

References::

    Kienholz, C., Rich, J. L., Arendt, a. a., and Hock, R. (2014).
        A new method for deriving glacier centerlines applied to glaciers in
        Alaska and northwest Canada. The Cryosphere, 8(2), 503-519.
        doi:10.5194/tc-8-503-2014

    Pfeffer, W. T., Arendt, A. a., Bliss, A., Bolch, T., Cogley, J. G.,
        Gardner, A. S., ... Sharp, M. J. (2014). The Randolph Glacier Inventory:
        a globally complete inventory of glaciers. Journal of Glaciology,
        60(221), 537-552. http://doi.org/10.3189/2014JoG13J176
"""
# Built ins
import os
import logging
from shutil import copyfile
from functools import partial
from distutils.version import LooseVersion
# External libs
import salem
import pyproj
import numpy as np
import shapely.ops
import geopandas as gpd
import skimage.draw as skdraw
import shapely.geometry as shpg
import scipy.signal
from scipy.ndimage.measurements import label
from scipy.interpolate import griddata
import rasterio
from rasterio.warp import reproject, Resampling
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool
# Locals
from oggm import entity_task
import oggm.cfg as cfg
from oggm.utils import tuple2int, get_topo_file, polygon_intersections

# Module logger
log = logging.getLogger(__name__)

# Needed later
label_struct = np.ones((3, 3))


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


def _check_geometry(geometry, gdir=None):
    """RGI polygons are not always clean: try to make these better.

    In particular, MultiPolygons should be converted to Polygons
    """

    if 'Multi' in geometry.type:
        parts = list(geometry)
        for p in parts:
            assert p.type == 'Polygon'
        exterior = parts[0].exterior
        # let's assume that all other polygons are in fact interiors
        interiors = []
        for p in parts[1:]:
            if parts[0].contains(p):
                interiors.append(p.exterior)
            else:
                # This should not happen. Check that we have a small geom here
                rid = gdir.rgi_id + ': ' if gdir is not None else ''
                msg = ('{}problem while correcting geometry. Area '
                       'was: {} but it should be smaller.'.format(rid, p.area))
                if p.area > 1e-4:
                    log.warning(msg)
        geometry = shpg.Polygon(exterior, interiors)

    assert 'Polygon' in geometry.type
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

    project = lambda x, y: (np.rint(x).astype(np.int64),
                            np.rint(y).astype(np.int64))

    poly_pix = shapely.ops.transform(project, polygon)

    # simple trick to correct invalid polys:
    tmp = poly_pix.buffer(0)

    # sometimes the glacier gets cut out in parts
    if tmp.type == 'MultiPolygon':
        # If only small arms are cut out, remove them
        area = np.array([_tmp.area for _tmp in tmp])
        _tokeep = np.asscalar(np.argmax(area))
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
                    raise RuntimeError('This glacier geometry is crazy.')

    if not tmp.is_valid:
        raise RuntimeError('This glacier geometry is crazy.')

    return tmp


@entity_task(log, writes=['glacier_grid', 'dem', 'outlines'])
def define_glacier_region(gdir, entity=None):
    """
    Very first task: define the glacier's local grid.

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
    entity : geopandas GeoSeries
        the glacier geometry to process
    """

    # choose a spatial resolution with respect to the glacier area
    dxmethod = cfg.PARAMS['grid_dx_method']
    area = gdir.rgi_area_km2
    if dxmethod == 'linear':
        dx = np.rint(cfg.PARAMS['d1'] * area + cfg.PARAMS['d2'])
    elif dxmethod == 'square':
        dx = np.rint(cfg.PARAMS['d1'] * np.sqrt(area) + cfg.PARAMS['d2'])
    elif dxmethod == 'fixed':
        dx = np.rint(cfg.PARAMS['fixed_dx'])
    else:
        raise ValueError('grid_dx_method not supported: {}'.format(dxmethod))
    # Additional trick for varying dx
    if dxmethod in ['linear', 'square']:
        dx = np.clip(dx, cfg.PARAMS['d2'], cfg.PARAMS['dmax'])

    log.debug('(%s) area %.2f km, dx=%.1f', gdir.rgi_id, area, dx)

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
    geometry = _check_geometry(geometry, gdir=gdir)
    xx, yy = geometry.exterior.xy

    # Corners, incl. a buffer of N pix
    ulx = np.min(xx) - cfg.PARAMS['border'] * dx
    lrx = np.max(xx) + cfg.PARAMS['border'] * dx
    uly = np.max(yy) + cfg.PARAMS['border'] * dx
    lry = np.min(yy) - cfg.PARAMS['border'] * dx
    # n pixels
    nx = np.int((lrx - ulx) / dx)
    ny = np.int((uly - lry) / dx)

    # Back to lon, lat for DEM download/preparation
    tmp_grid = salem.Grid(proj=proj_out, nxny=(nx, ny), x0y0=(ulx, uly),
                          dxdy=(dx, -dx), pixel_ref='corner')
    minlon, maxlon, minlat, maxlat = tmp_grid.extent_in_crs(crs=salem.wgs84)

    # save transformed geometry to disk
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
    towrite.to_file(gdir.get_filepath('outlines'))

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
            gdf.to_file(gdir.get_filepath('intersects'))

    # Open DEM
    source = entity.DEM_SOURCE if hasattr(entity, 'DEM_SOURCE') else None
    dem_list, dem_source = get_topo_file((minlon, maxlon), (minlat, maxlat),
                                         rgi_region=gdir.rgi_region,
                                         source=source)
    log.debug('(%s) DEM source: %s', gdir.rgi_id, dem_source)

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
        raise ValueError('{} interpolation not understood'
                         .format(cfg.PARAMS['topo_interp']))

    dem_reproj = gdir.get_filepath('dem')
    with rasterio.open(dem_reproj, 'w', **profile) as dest:
        dst_array = np.empty((ny, nx), dtype=dem_dss[0].dtypes[0])
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=dem_dss[0].crs,
            src_transform=src_transform,
            # Destination parameters
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=proj4_str,
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
    gdir.write_pickle(dem_source, 'dem_source')


@entity_task(log, writes=['gridded_data', 'geometries'])
def glacier_masks(gdir):
    """Makes a gridded mask of the glacier outlines.

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

    # Correct the DEM (ASTER...)
    # Currently we just do a linear interp -- ASTER is totally shit anyway
    min_z = -999.
    isfinite = np.isfinite(dem)
    if (np.min(dem) <= min_z) or np.any(~isfinite):
        xx, yy = gdir.grid.ij_coordinates
        pnan = np.nonzero((dem <= min_z) | (~isfinite))
        pok = np.nonzero((dem > min_z) | isfinite)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        dem[pnan] = griddata(points, np.ravel(dem[pok]), inter)
        log.warning(gdir.rgi_id + ': DEM needed interpolation.')

    isfinite = np.isfinite(dem)
    if not np.all(isfinite):
        # see how many percent of the dem
        if np.sum(~isfinite) > (0.2 * nx * ny):
            raise RuntimeError('({}) too many NaNs in DEM'.format(gdir.rgi_id))
        log.warning('({}) DEM needed zeros somewhere.'.format(gdir.rgi_id))
        dem[isfinite] = 0

    if np.min(dem) == np.max(dem):
        raise RuntimeError('({}) min equal max in the DEM.'
                           .format(gdir.rgi_id))

    # Proj
    if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
        transf = dem_dr.transform
    else:
        transf = dem_dr.affine
    x0 = transf[2]  # UL corner
    y0 = transf[5]  # UL corner
    dx = transf[0]
    dy = transf[4]  # Negative
    assert dx == -dy
    assert dx == gdir.grid.dx
    assert y0 == gdir.grid.corner_grid.y0
    assert x0 == gdir.grid.corner_grid.x0
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
        raise RuntimeError('({}) NaN in smoothed DEM'.format(gdir.rgi_id))

    # Geometries
    outlines_file = gdir.get_filepath('outlines')
    geometry = gpd.GeoDataFrame.from_file(outlines_file).geometry[0]

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
        raise RuntimeError('This glacier geometry is crazy.')

    # Rounded nearest pix
    glacier_poly_pix = _polygon_to_pix(glacier_poly_hr)

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
        raise RuntimeError('({}) min equal max in the masked DEM.'
                           .format(gdir.rgi_id))

    # write out the grids in the netcdf file
    nc = gdir.create_gridded_ncdf_file('gridded_data')

    v = nc.createVariable('topo', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = 'DEM topography'
    v[:] = dem

    v = nc.createVariable('topo_smoothed', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = ('DEM topography smoothed' 
                   ' with radius: {:.1} m'.format(cfg.PARAMS['smooth_window']))
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
