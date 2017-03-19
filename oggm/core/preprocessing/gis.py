""" Handling of the local glacier map and masks. Defines the first two tasks
to be realized by any OGGM pre-processing workflow:

    - define_glacier_region: prepares a local directory, map and DEM
    for each glacier.
    - glacier_masks: computes the glacier grids and masks for each divide

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
from __future__ import absolute_import, division

# Built ins
import os
import logging
from shutil import copyfile
from functools import partial
# External libs
from osgeo import osr
import salem
from osgeo import gdal
import pyproj
import numpy as np
import shapely.ops
import geopandas as gpd
import skimage.draw as skdraw
import shapely.geometry as shpg
import scipy.signal
from scipy.ndimage.measurements import label
from scipy.interpolate import griddata
# Locals
from oggm import entity_task
import oggm.cfg as cfg
from oggm.utils import tuple2int, get_topo_file

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


def _check_geometry(geometry):
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
            assert parts[0].contains(p)
            interiors.append(p.exterior)
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


def _mask_per_divide(gdir, div_id, dem, smoothed_dem):
    """Compute mask and geometries for each glacier divide.

    Is called by glacier masks.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    div_id: int
        id of the divide to process
    dem: 2D array
        topography
    smoothed_dem: 2D array
        smoothed topography
    """

    outlines_file = gdir.get_filepath('outlines', div_id=div_id)
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
        log.debug('%s: we had to cut an island in the mask', gdir.rgi_id)
        # Check the size of those
        region_sizes = [np.sum(regions == r) for r in np.arange(1, nregions+1)]
        am = np.argmax(region_sizes)
        # Check not a strange glacier
        sr = region_sizes.pop(am)
        for ss in region_sizes:
            assert (ss / sr) < 0.1
        glacier_mask[:] = 0
        glacier_mask[np.where(regions == (am+1))] = 1

    # write out the grids in the netcdf file
    nc = gdir.create_gridded_ncdf_file('gridded_data', div_id=div_id)

    v = nc.createVariable('topo', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = 'DEM topography'
    v[:] = dem

    v = nc.createVariable('topo_smoothed', 'f4', ('y', 'x', ), zlib=True)
    v.units = 'm'
    v.long_name = 'DEM topography smoothed' \
                  ' with radius: {:.1} m'.format(cfg.PARAMS['smooth_window'])
    v[:] = smoothed_dem

    v = nc.createVariable('glacier_mask', 'i1', ('y', 'x', ), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier mask'
    v[:] = glacier_mask

    v = nc.createVariable('glacier_ext', 'i1', ('y', 'x', ), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier external boundaries'
    v[:] = glacier_ext

    nc.close()

    geometries = dict()
    geometries['polygon_hr'] = glacier_poly_hr
    geometries['polygon_pix'] = glacier_poly_pix
    geometries['polygon_area'] = geometry.area
    gdir.write_pickle(geometries, 'geometries', div_id=div_id)


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

    log.debug('%s: area %.2f km, dx=%.1f', gdir.rgi_id, area, dx)

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
    geometry = _check_geometry(geometry)
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
    towrite.to_file(gdir.get_filepath('outlines'))

    # Also transform the intersects if necessary
    gdf = cfg.PARAMS['intersects_gdf']
    gdf = gdf.loc[(gdf.RGIId_1 == gdir.rgi_id) | (gdf.RGIId_2 == gdir.rgi_id)]
    if len(gdf) > 0:
        gdf = salem.transform_geopandas(gdf, to_crs=proj_out)
        if hasattr(gdf.crs, 'srs'):
            # salem uses pyproj
            gdf.crs = gdf.crs.srs
        gdf.to_file(gdir.get_filepath('intersects'))

    # Open DEM
    source = entity.DEM_SOURCE if hasattr(entity, 'DEM_SOURCE') else None
    dem_file, dem_source = get_topo_file((minlon, maxlon), (minlat, maxlat),
                                         rgi_region=gdir.rgi_region,
                                         source=source)
    log.debug('%s: DEM source: %s', gdir.rgi_id, dem_source)
    dem = gdal.Open(dem_file)
    geo_t = dem.GetGeoTransform()

    # Input proj
    dem_proj = dem.GetProjection()
    if dem_proj == '':
        # Assume defaults
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        dem_proj = wgs84.ExportToWkt()


    # Dest proj
    gproj = osr.SpatialReference()
    gproj.SetProjCS('Local transverse Mercator')
    gproj.SetWellKnownGeogCS("WGS84")
    gproj.SetTM(np.float(0), gdir.cenlon,
                np.float(0.9996), np.float(0), np.float(0))

    # Create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')

    # GDALDataset Create (char *pszName, int nXSize, int nYSize,
    #                     int nBands, GDALDataType eType)
    dest = mem_drv.Create('', nx, ny, 1, gdal.GDT_Float32)

    # Calculate the new geotransform
    new_geo = (ulx, dx, geo_t[2], uly, geo_t[4], -dx)

    # Set the geotransform
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(gproj.ExportToWkt())

    # Perform the projection/resampling
    if cfg.PARAMS['topo_interp'] == 'bilinear':
        interp = gdal.GRA_Bilinear
    elif cfg.PARAMS['topo_interp'] == 'cubic':
        interp = gdal.GRA_Cubic
    else:
        raise ValueError('{} interpolation not understood'
                         .format(cfg.PARAMS['topo_interp']))

    res = gdal.ReprojectImage(dem, dest, dem_proj,
                              gproj.ExportToWkt(), interp)

    # Let's save it as a GeoTIFF.
    driver = gdal.GetDriverByName("GTiff")
    tmp = driver.CreateCopy(gdir.get_filepath('dem'), dest, 0)
    tmp = None # without this, GDAL is getting crazy in python3
    dest = None # the memfree above is necessary, this one is to be sure...

    # Glacier grid
    x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
    glacier_grid = salem.Grid(proj=proj_out, nxny=(nx, ny),  dxdy=(dx, -dx),
                              x0y0=x0y0)
    glacier_grid.to_json(gdir.get_filepath('glacier_grid'))
    gdir.write_pickle(dem_source, 'dem_source')

    # Looks in the database if the glacier has divides.
    gdf = cfg.PARAMS['divides_gdf']
    if gdir.rgi_id in gdf.index.values:
        divdf = [g for g in gdf.loc[gdir.rgi_id].geometry]

        # Reproject the shape
        def proj(lon, lat):
            return salem.transform_proj(salem.wgs84, gdir.grid.proj,
                                        lon, lat)
        divdf = [shapely.ops.transform(proj, g) for g in divdf]

        # Keep only the ones large enough
        log.debug('%s: divide candidates: %d', gdir.rgi_id, len(divdf))
        divdf = [g for g in divdf if (g.area >= (25*dx**2))]
        log.debug('%s: number of divides: %d', gdir.rgi_id, len(divdf))

        # Write the directories and the files
        for i, g in enumerate(divdf):
            _dir = os.path.join(gdir.dir, 'divide_{0:0=2d}'.format(i + 1))
            if not os.path.exists(_dir):
                os.makedirs(_dir)
            # File
            entity['geometry'] = g
            towrite = gpd.GeoDataFrame(entity).T
            towrite.crs = proj4_str
            towrite.to_file(os.path.join(_dir, cfg.BASENAMES['outlines']))
    else:
        # Make a single directory and link the files
        log.debug('%s: number of divides: %d', gdir.rgi_id, 1)
        _dir = os.path.join(gdir.dir, 'divide_01')
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        linkname = os.path.join(_dir, cfg.BASENAMES['outlines'])
        sourcename = gdir.get_filepath('outlines')
        for ending in ['.cpg', '.dbf', '.shp', '.shx', '.prj']:
            _s = sourcename.replace('.shp', ending)
            _l = linkname.replace('.shp', ending)
            if os.path.exists(_s):
                try:
                    # we are on UNIX
                    os.link(_s, _l)
                except AttributeError:
                    # we are on windows
                    copyfile(_s, _l)


@entity_task(log, writes=['gridded_data', 'geometries'])
def glacier_masks(gdir):
    """Makes a gridded mask of the glacier outlines.

    Additionally, it prepares the divide directories and further gridded data
    needed for subsequent tasks.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # open srtm tif-file:
    dem_ds = gdal.Open(gdir.get_filepath('dem'))
    dem = dem_ds.ReadAsArray().astype(float)

    # Correct the DEM (ASTER...)
    # Currently we just do a linear interp -- ASTER is totally shit anyway
    min_z = -999.
    if np.min(dem) <= min_z:
        xx, yy = gdir.grid.ij_coordinates
        pnan = np.nonzero(dem <= min_z)
        pok = np.nonzero(dem > min_z)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        dem[pnan] = griddata(points, np.ravel(dem[pok]), inter)
        log.warning(gdir.rgi_id + ': DEM needed interpolation.')
    if np.min(dem) == np.max(dem):
        raise RuntimeError(gdir.rgi_id + ': min equal max in the DEM.')

    # Grid
    nx = dem_ds.RasterXSize
    ny = dem_ds.RasterYSize
    assert nx == gdir.grid.nx
    assert ny == gdir.grid.ny

    # Proj
    geot = dem_ds.GetGeoTransform()
    x0 = geot[0]  # UL corner
    y0 = geot[3]  # UL corner
    dx = geot[1]
    dy = geot[5]  # Negative
    assert dx == -dy
    assert dx == gdir.grid.dx
    assert y0 == gdir.grid.corner_grid.y0
    assert x0 == gdir.grid.corner_grid.x0
    dem_ds = None  # to be sure...

    # Clip topography to 0 m a.s.l.
    dem = dem.clip(0)

    # Smooth DEM?
    if cfg.PARAMS['smooth_window'] > 0.:
        gsize = np.rint(cfg.PARAMS['smooth_window'] / dx)
        smoothed_dem = gaussian_blur(dem, np.int(gsize))
    else:
        smoothed_dem = dem.copy()

    # Make entity masks
    log.debug('%s: glacier mask, divide %d', gdir.rgi_id, 0)
    _mask_per_divide(gdir, 0, dem, smoothed_dem)

    # Glacier divides
    nd = gdir.n_divides
    if nd == 1:
        # Optim: just make links
        linkname = gdir.get_filepath('gridded_data', div_id=1)
        sourcename = gdir.get_filepath('gridded_data')
        # overwrite as default
        if os.path.exists(linkname):
            os.remove(linkname)
        # TODO: temporary suboptimal solution
        try:
            # we are on UNIX
            os.link(sourcename, linkname)
        except AttributeError:
            # we are on windows
            copyfile(sourcename, linkname)
        linkname = gdir.get_filepath('geometries', div_id=1)
        sourcename = gdir.get_filepath('geometries')
        # overwrite as default
        if os.path.exists(linkname):
            os.remove(linkname)
        # TODO: temporary suboptimal solution
        try:
            # we are on UNIX
            os.link(sourcename, linkname)
        except AttributeError:
            # we are on windows
            copyfile(sourcename, linkname)
    else:
        # Loop over divides
        for i in gdir.divide_ids:
            log.debug('%s: glacier mask, divide %d', gdir.rgi_id, i)
            _mask_per_divide(gdir, i, dem, smoothed_dem)
