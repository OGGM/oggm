""" Handling of the local glacier map and masks. Defines the first tasks
to be realized by any OGGM pre-processing workflow.

"""
# Built ins
import os
import logging
import warnings
import json
from packaging.version import Version
from functools import partial

# External libs
import numpy as np
import shapely.ops
import pandas as pd
import xarray as xr
import shapely.geometry as shpg
import scipy.signal
from scipy.ndimage import label, distance_transform_edt, binary_erosion
from scipy.interpolate import griddata
from scipy import optimize as optimization

# Optional libs
try:
    import salem
    from salem.gis import transform_proj
except ImportError:
    pass
try:
    import pyproj
except ImportError:
    pass
try:
    import geopandas as gpd
except ImportError:
    pass
try:
    import skimage.draw as skdraw
except ImportError:
    pass
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.mask import mask as riomask
    from rasterio.merge import merge as merge_tool
except ImportError:
    pass

# Locals
from oggm import entity_task, utils
import oggm.cfg as cfg
from oggm.exceptions import (InvalidParamsError, InvalidGeometryError,
                             InvalidDEMError, GeometryError,
                             InvalidWorkflowError)
from oggm.utils import (tuple2int, get_topo_file, is_dem_source_available,
                        nicenumber, ncDataset, tolist)


# Module logger
log = logging.getLogger(__name__)

# Needed later
label_struct = np.ones((3, 3))


def _parse_source_text():
    fp = os.path.join(os.path.abspath(os.path.dirname(cfg.__file__)),
                      'data', 'dem_sources.txt')

    out = dict()
    cur_key = None
    with open(fp, 'r', encoding='utf-8') as fr:
        this_text = []
        for l in fr.readlines():
            l = l.strip()
            if l and (l[0] == '[' and l[-1] == ']'):
                if cur_key:
                    out[cur_key] = '\n'.join(this_text)
                this_text = []
                cur_key = l.strip('[]')
                continue
            this_text.append(l)
    out[cur_key] = '\n'.join(this_text)
    return out


DEM_SOURCE_INFO = _parse_source_text()


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

    def project_coarse(x, y, c=2):
        return ((np.rint(x/c)*c).astype(np.int64),
                (np.rint(y/c)*c).astype(np.int64))

    poly_pix = shapely.ops.transform(project, polygon)

    # simple trick to correct invalid polys:
    tmp = poly_pix.buffer(0)

    # try to deal with a bug in buffer where the corrected poly would be null
    c = 2
    while tmp.length == 0 and c < 7:
        project = partial(project_coarse, c=c)
        poly_pix = shapely.ops.transform(project_coarse, polygon)
        tmp = poly_pix.buffer(0)
        c += 1

    # We tried all we could
    if tmp.length == 0:
        raise InvalidGeometryError('This glacier geometry is not valid for '
                                   'OGGM.')

    # sometimes the glacier gets cut out in parts
    if tmp.geom_type == 'MultiPolygon':
        # If only small arms are cut out, remove them
        area = np.array([_tmp.area for _tmp in tmp.geoms])
        _tokeep = np.argmax(area).item()
        tmp = tmp.geoms[_tokeep]

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
                    if tmp.geom_type == 'MultiPolygon':
                        continue
                    if tmp.is_valid:
                        break
                if b == 0.99:
                    raise InvalidGeometryError('This glacier geometry is not '
                                               'valid for OGGM.')

    if not tmp.is_valid:
        raise InvalidGeometryError('This glacier geometry is not valid '
                                   'for OGGM.')

    return tmp


def glacier_grid_params(gdir):
    """Define the glacier grid map based on the user params."""

    # Get the local map proj params and glacier extent
    gdf = gdir.read_shapefile('outlines')

    # Get the map proj
    utm_proj = salem.check_crs(gdf.crs)

    # Get glacier extent
    try:
        xx, yy = gdf.iloc[0]['geometry'].exterior.xy
    # special treatment for Multipolygons
    except AttributeError:
        if not cfg.PARAMS['keep_multipolygon_outlines']:
            raise
        parts = []
        for p in gdf.iloc[0]['geometry'].geoms:
            parts.append(p)
        parts = np.array(parts)

        xx = []
        yy = []
        for part in parts:
            xx_tmp, yy_tmp = part.exterior.xy
            xx = np.append(xx, xx_tmp)
            yy = np.append(yy, yy_tmp)

    # Define glacier area to use
    area = gdir.rgi_area_km2

    # Choose a spatial resolution with respect to the glacier area
    dxmethod = cfg.PARAMS['grid_dx_method']
    if dxmethod == 'linear':
        dx = np.rint(cfg.PARAMS['d1'] * area + cfg.PARAMS['d2'])
    elif dxmethod == 'square':
        dx = np.rint(cfg.PARAMS['d1'] * np.sqrt(area) + cfg.PARAMS['d2'])
    elif dxmethod == 'fixed':
        dx = np.rint(cfg.PARAMS['fixed_dx'])
    elif dxmethod == 'by_bin':
        bins = cfg.PARAMS['by_bin_bins']
        bin_dx = cfg.PARAMS['by_bin_dx']
        for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):
            if b1 < area <= b2:
                dx = np.rint(bin_dx[i])
                break
    else:
        raise InvalidParamsError('grid_dx_method not supported: {}'
                                 .format(dxmethod))
    # Additional trick for varying dx
    if dxmethod in ['linear', 'square']:
        dx = utils.clip_scalar(dx, cfg.PARAMS['d2'], cfg.PARAMS['dmax'])

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
    nx = int((lrx - ulx) / dx)
    ny = int((uly - lry) / dx)

    return utm_proj, nx, ny, ulx, uly, dx


def check_dem_source(source, extent_ll, rgi_id=None):
    """
    This function can check for multiple DEM sources and is in charge of the
    error handling if a requested source is not available for the given
    glacier/extent

    Parameters
    ----------
    source : str or list of str
        If you want to force the use of a certain DEM source. If list is
        provided they are checked in order. For a list of available options
        check docstring of oggm.core.gis.define_glacier_region.
    extent_ll : list
        The longitude and latitude extend which should be checked for. Should
        be provided as extent_ll = [[minlon, maxlon], [minlat, maxlat]].
    rgi_id : str
        The RGI-ID of the glacier, only used for a descriptive error message in
        case.

    Returns
    -------
    String of the working DEM source.
    """

    # when multiple sources are provided, try them sequentially
    if isinstance(source, list):
        for src in source:
            source_exists = is_dem_source_available(src, *extent_ll)
            if source_exists:
                source = src  # pick the first source which exists
                break
    else:
        source_exists = is_dem_source_available(source, *extent_ll)

    if not source_exists:
        if rgi_id is None:
            extent_string = (f"the grid extent of longitudes {extent_ll[0]} "
                             f"and latitudes {extent_ll[1]}")
        else:
            extent_string = (f"the glacier {rgi_id} with border "
                             f"{cfg.PARAMS['border']}")
        raise InvalidWorkflowError(f"Source: {source} is not available for "
                                   f"{extent_string}")
    return source


def reproject_dem(dem_list, dem_source, dst_grid_prop, output_path):
    """
    This function reprojects a provided DEM to the destination grid and saves
    the result to disk.

    Parameters
    ----------
    dem_list : list of str
        list with path(s) to the DEM file(s)
    dem_source : str
        DEM source string
    dst_grid_prop : dict
        Holds necessary grid properties. Must contain 'utm_proj', 'dx', 'ulx',
        'uly', 'nx' and 'ny.
    output_path : str
        Filepath where to store the reprojected DEM.
    """

    # Decide how to tag nodata
    def _get_nodata(rio_ds):
        nodata = rio_ds[0].meta.get('nodata', None)
        if nodata is None:
            # badly tagged geotiffs, let's do it ourselves
            nodata = -32767 if dem_source == 'TANDEM' else -9999
        return nodata

    # A glacier area can cover more than one tile:
    if len(dem_list) == 1:
        dem_dss = [rasterio.open(dem_list[0])]  # if one tile, just open it
        dem_data = rasterio.band(dem_dss[0], 1)
        if Version(rasterio.__version__) >= Version('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
        nodata = _get_nodata(dem_dss)
    else:
        dem_dss = [rasterio.open(s) for s in dem_list]  # list of rasters
        nodata = _get_nodata(dem_dss)
        dem_data, src_transform = merge_tool(dem_dss, nodata=nodata)  # merge

    # Use Grid properties to create a transform (see rasterio cookbook)
    dst_transform = rasterio.transform.from_origin(
        dst_grid_prop['ulx'], dst_grid_prop['uly'], dst_grid_prop['dx'],
        dst_grid_prop['dx']
        # sign change (2nd dx) is done by rasterio.transform
    )

    # Set up profile for writing output
    profile = dem_dss[0].profile
    profile.update({
        'crs': dst_grid_prop['utm_proj'].srs,
        'transform': dst_transform,
        'nodata': nodata,
        'width': dst_grid_prop['nx'],
        'height': dst_grid_prop['ny'],
        'driver': 'GTiff'
    })
    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)
    profile.pop('compress', None)

    # Could be extended so that the cfg file takes all Resampling.* methods
    if cfg.PARAMS['topo_interp'] == 'bilinear':
        resampling = Resampling.bilinear
    elif cfg.PARAMS['topo_interp'] == 'cubic':
        resampling = Resampling.cubic
    else:
        raise InvalidParamsError('{} interpolation not understood'
                                 .format(cfg.PARAMS['topo_interp']))

    with rasterio.open(output_path, 'w', **profile) as dest:
        dst_array = np.empty((dst_grid_prop['ny'], dst_grid_prop['nx']),
                             dtype=dem_dss[0].dtypes[0])
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=dem_dss[0].crs,
            src_transform=src_transform,
            src_nodata=nodata,
            # Destination parameters
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=dst_grid_prop['utm_proj'].srs,
            dst_nodata=nodata,
            # Configuration
            resampling=resampling)
        dest.write(dst_array, 1)

    for dem_ds in dem_dss:
        dem_ds.close()


def get_dem_for_grid(grid, fpath, source=None, gdir=None):
    """
    Fetch a DEM from source, reproject it to the extent defined by grid and
    saves it to disk.

    Parameters
    ----------
    grid : :py:class:`salem.gis.Grid`
        Grid which defines the extent and projection of the final DEM
    fpath : str
        The output filepath for the final DEM.
    source : str or list of str
        If you want to force the use of a certain DEM source. If list is
        provided they are checked in order. For a list of available options
        check docstring of oggm.core.gis.define_glacier_region.
    gdir: py:class:`oggm.GlacierDirectory`
        If source is None, gdir is used to decide on the source

    Returns
    -------
    tuple: (list with path(s) to the DEM file(s), data source str)
    """
    minlon, maxlon, minlat, maxlat = grid.extent_in_crs(crs=salem.wgs84)
    extent_ll = [[minlon, maxlon], [minlat, maxlat]]
    if gdir is not None:
        rgi_id = gdir.rgi_id
    else:
        rgi_id = None

    grid_prop = {
        'utm_proj': grid.proj,
        'dx': grid.dx,
        'ulx': grid.x0,
        'uly': grid.y0,
        'nx': grid.nx,
        'ny': grid.ny
    }

    source = check_dem_source(source, extent_ll, rgi_id=rgi_id)

    dem_list, dem_source = get_topo_file((minlon, maxlon), (minlat, maxlat),
                                         gdir=gdir,
                                         dx_meter=grid_prop['dx'],
                                         source=source)

    if rgi_id is not None:
        log.debug('(%s) DEM source: %s', rgi_id, dem_source)
        log.debug('(%s) N DEM Files: %s', rgi_id, len(dem_list))

    # further checks if given fpath exists?
    if fpath[-4:] != '.tif':
        # we add the default filename
        fpath = os.path.join(fpath, 'dem.tif')

    reproject_dem(dem_list=dem_list, dem_source=dem_source,
                  dst_grid_prop=grid_prop,
                  output_path=fpath)

    return dem_list, dem_source


@entity_task(log, writes=['glacier_grid', 'dem', 'outlines'])
def define_glacier_region(gdir, entity=None, source=None):
    """Very first task after initialization: define the glacier's local grid.

    Defines the local projection (Transverse Mercator or UTM depending on
    user choice), centered on the glacier.
    There is some options to set the resolution of the local grid.
    It can be adapted depending on the size of the glacier.
    See ``params.cfg`` for setting these options.

    Default values of the adapted mode lead to a resolution of 50 m for
    Hintereisferner, which is approx. 8 km2 large.

    After defining the grid, the topography and the outlines of the glacier
    are transformed into the local projection.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    entity : geopandas.GeoSeries
        the glacier geometry to process - DEPRECATED. It is now ignored
    source : str or list of str, optional
        If you want to force the use of a certain DEM source. Available are:
          - 'USER' : file set in cfg.PATHS['dem_file']
          - 'SRTM' : http://srtm.csi.cgiar.org/
          - 'GIMP' : https://bpcrc.osu.edu/gdg/data/gimpdem
          - 'RAMP' : https://nsidc.org/data/nsidc-0082/versions/2/documentation
          - 'REMA' : https://www.pgc.umn.edu/data/rema/
          - 'DEM3' : http://viewfinderpanoramas.org/
          - 'ASTER' : https://asterweb.jpl.nasa.gov/gdem.asp
          - 'TANDEM' : https://geoservice.dlr.de/web/dataguide/tdm90/
          - 'ARCTICDEM' : https://www.pgc.umn.edu/data/arcticdem/
          - 'AW3D30' : https://www.eorc.jaxa.jp
          - 'MAPZEN' : https://registry.opendata.aws/terrain-tiles/
          - 'ALASKA' : https://www.the-cryosphere.net/8/503/2014/
          - 'COPDEM30' : Copernicus DEM GLO30 https://spacedata.copernicus.eu/web/cscda/cop-dem-faq
          - 'COPDEM90' : Copernicus DEM GLO90 https://spacedata.copernicus.eu/web/cscda/cop-dem-faq
          - 'NASADEM':  https://doi.org/10.5069/G93T9FD9
    """

    utm_proj, nx, ny, ulx, uly, dx = glacier_grid_params(gdir)

    # Back to lon, lat for DEM download/preparation
    tmp_grid = salem.Grid(proj=utm_proj, nxny=(nx, ny), x0y0=(ulx, uly),
                          dxdy=(dx, -dx), pixel_ref='corner')

    dem_list, dem_source = get_dem_for_grid(grid=tmp_grid,
                                            fpath=gdir.get_filepath('dem'),
                                            source=source, gdir=gdir)

    # Glacier grid
    x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
    glacier_grid = salem.Grid(proj=utm_proj, nxny=(nx, ny), dxdy=(dx, -dx),
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


def rasterio_to_gdir(gdir, input_file, output_file_name,
                     resampling='cubic'):
    """Reprojects a file that rasterio can read into the glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    input_file : str
        path to the file to reproject
    output_file_name : str
        name of the output file (must be in cfg.BASENAMES)
    resampling : str
        nearest', 'bilinear', 'cubic', 'cubic_spline', or one of
        https://rasterio.readthedocs.io/en/latest/topics/resampling.html
    """

    output_file = gdir.get_filepath(output_file_name)
    assert '.tif' in output_file, 'output_file should end with .tif'

    if not gdir.has_file('dem'):
        raise InvalidWorkflowError('Need a dem.tif file to reproject to')

    with rasterio.open(input_file) as src:

        kwargs = src.meta.copy()
        data = src.read(1)

        with rasterio.open(gdir.get_filepath('dem')) as tpl:

            kwargs.update({
                'crs': tpl.crs,
                'transform': tpl.transform,
                'width': tpl.width,
                'height': tpl.height
            })

            with rasterio.open(output_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):

                    dest = np.zeros(shape=(tpl.height, tpl.width),
                                    dtype=data.dtype)

                    reproject(
                        source=rasterio.band(src, i),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tpl.transform,
                        dst_crs=tpl.crs,
                        resampling=getattr(Resampling, resampling)
                    )

                    dst.write(dest, indexes=i)


def read_geotiff_dem(gdir=None, fpath=None):
    """Reads (and masks out) the DEM out of the gdir's geotiff file or a
    geotiff file given by the fpath variable.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    fpath : 'str'
        path to a gdir

    Returns
    -------
    2D np.float32 array
    """
    if gdir is not None:
        dem_path = gdir.get_filepath('dem')
    else:
        if fpath is not None:
            if fpath[-4:] != '.tif':
                # we add the default filename
                fpath = os.path.join(fpath, 'dem.tif')
            dem_path = fpath
        else:
            raise InvalidParamsError('If you do not provide a gdir you must'
                                     f'define a fpath! Given fpath={fpath}.')

    with rasterio.open(dem_path, 'r', driver='GTiff') as ds:
        topo = ds.read(1).astype(rasterio.float32)
        topo[ds.read_masks(1) == 0] = np.nan
        # This is for bad tiffs where the above doesn't work
        topo[topo <= -999.] = np.nan

    if gdir is not None and gdir.get_diagnostics()['dem_source'] in ['COPDEM30', 'COPDEM90']:
        # Latest COP DEM versions have nodata for ocean pixels
        # I'm not sure nodata is *always* ocean, but hey
        topo[np.isnan(topo)] = 0

    return topo


class GriddedNcdfFile(object):
    """Creates or opens a gridded netcdf file template.

    The other variables have to be created and filled by the calling
    routine.
    """
    def __init__(self, gdir=None, grid=None, fpath=None,
                 basename=None, reset=False):
        """
        Parameters
        ----------
        gdir : :py:class:`oggm.GlacierDirectory`
            The glacier directory. If provided it defines the filepath and the
            grid used for the netcdf file. It overrules grid and fpath if all
            are provided. (It is assumed to be the first kwarg at some places
            in the code.)
        grid : :py:class:`salem.gis.Grid`
            Grid which defines the extent of the netcdf file. Needed if gdir is
            not provided.
        fpath : str
            The output filepath for the netcdf file. Needed if gdir is not
            provided.
        basename : str
            The filename of the resulting netcdf file
        reset : bool
            If True, a potentially existing file will be deleted.
        """

        if basename is None:
            basename = 'gridded_data'

        if gdir is not None:
            self.fpath = gdir.get_filepath(basename)
            self.grid = gdir.grid
        else:
            if grid is None or fpath is None:
                raise InvalidParamsError('If you do not provide a gdir you must'
                                         'define grid and fpath! Given grid='
                                         f'{grid} and fpath={fpath}.')
            self.fpath = os.path.join(fpath, basename + '.nc')
            self.grid = grid

        if reset and os.path.exists(self.fpath):
            os.remove(self.fpath)

    def __enter__(self):

        if os.path.exists(self.fpath):
            # Already there - just append
            self.nc = ncDataset(self.fpath, 'a', format='NETCDF4')
            return self.nc

        # Create and fill
        nc = ncDataset(self.fpath, 'w', format='NETCDF4')

        nc.createDimension('x', self.grid.nx)
        nc.createDimension('y', self.grid.ny)

        nc.author = 'OGGM'
        nc.author_info = 'Open Global Glacier Model'
        nc.pyproj_srs = self.grid.proj.srs

        x = self.grid.x0 + np.arange(self.grid.nx) * self.grid.dx
        y = self.grid.y0 + np.arange(self.grid.ny) * self.grid.dy

        v = nc.createVariable('x', 'f4', ('x',), zlib=True)
        v.units = 'm'
        v.long_name = 'x coordinate of projection'
        v.standard_name = 'projection_x_coordinate'
        v[:] = x

        v = nc.createVariable('y', 'f4', ('y',), zlib=True)
        v.units = 'm'
        v.long_name = 'y coordinate of projection'
        v.standard_name = 'projection_y_coordinate'
        v[:] = y

        self.nc = nc
        return nc

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.nc.close()


@entity_task(log, writes=['gridded_data'])
def process_dem(gdir=None, grid=None, fpath=None, output_filename=None):
    """Reads the DEM from the tiff, attempts to fill voids and apply smooth.

    The data is then written to `gridded_data.nc`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        Where to write the data. If set it overrules grid and fpath.
    grid : :py:class:`salem.gis.Grid`
        Grid of the DEM file. Needed if gdir is not provided.
    fpath : str
        The filepath of the DEM file. Needed if gdir is not provided.
    output_filename : str
        The filename of the nc file to add the DEM to. Defaults to gridded_data
    """
    if gdir is not None:
        # open srtm tif-file:
        dem = read_geotiff_dem(gdir)
        # Grid
        dem_grid = gdir.grid
        grid_name = gdir.rgi_id
    else:
        if grid is None or fpath is None:
            raise InvalidParamsError('If you do not provide a gdir you must'
                                     'define grid and fpath! Given grid='
                                     f'{grid} and fpath={fpath}.')
        dem = read_geotiff_dem(fpath=fpath)
        grid_name = 'custom_grid'
        dem_grid = grid
        diagnostics_dict = dict()
    # Grid parameters
    nx = dem_grid.nx
    ny = dem_grid.ny
    dx = dem_grid.dx
    xx, yy = dem_grid.ij_coordinates

    # Correct the DEM
    valid_mask = np.isfinite(dem)
    if np.all(~valid_mask):
        raise InvalidDEMError('Not a single valid grid point in DEM')

    if np.any(~valid_mask):
        # We interpolate
        if np.sum(~valid_mask) > (0.25 * nx * ny) and gdir is not None:
            log.info('({}) more than 25% nans in DEM'.format(gdir.rgi_id))
        pnan = np.nonzero(~valid_mask)
        pok = np.nonzero(valid_mask)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        try:
            dem[pnan] = griddata(points, np.ravel(dem[pok]), inter,
                                 method='linear')
        except ValueError:
            raise InvalidDEMError('DEM interpolation not possible.')
        if gdir is not None:
            log.info(gdir.rgi_id + ': DEM needed interpolation.')
            gdir.add_to_diagnostics('dem_needed_interpolation', True)
            gdir.add_to_diagnostics('dem_invalid_perc',
                                    len(pnan[0]) / (nx * ny))
        else:
            diagnostics_dict['dem_needed_interpolation'] = True
            diagnostics_dict['dem_invalid_perc'] = len(pnan[0]) / (nx * ny)

    isfinite = np.isfinite(dem)
    if np.any(~isfinite):
        # interpolation will still leave nans in DEM:
        # extrapolate with NN if needed (e.g. coastal areas)
        pnan = np.nonzero(~isfinite)
        pok = np.nonzero(isfinite)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        try:
            dem[pnan] = griddata(points, np.ravel(dem[pok]), inter,
                                 method='nearest')
        except ValueError:
            raise InvalidDEMError('DEM extrapolation not possible.')
        if gdir is not None:
            log.info(gdir.rgi_id + ': DEM needed extrapolation.')
            gdir.add_to_diagnostics('dem_needed_extrapolation', True)
            gdir.add_to_diagnostics('dem_extrapol_perc',
                                    len(pnan[0]) / (nx * ny))
        else:
            diagnostics_dict['dem_needed_extrapolation'] = True
            diagnostics_dict['dem_extrapol_perc'] = len(pnan[0]) / (nx * ny)

    if np.min(dem) == np.max(dem):
        raise InvalidDEMError('({}) min equal max in the DEM.'
                              .format(grid_name))

    # Clip topography to 0 m a.s.l.
    if cfg.PARAMS['clip_dem_to_zero']:
        utils.clip_min(dem, 0, out=dem)

    # Smooth DEM?
    if cfg.PARAMS['smooth_window'] > 0.:
        gsize = np.rint(cfg.PARAMS['smooth_window'] / dx)
        smoothed_dem = gaussian_blur(dem, int(gsize))
    else:
        smoothed_dem = dem.copy()

    # Clip topography to 0 m a.s.l.
    if cfg.PARAMS['clip_dem_to_zero']:
        utils.clip_min(smoothed_dem, 0, out=smoothed_dem)

    if gdir is None:
        with open(os.path.join(fpath, 'dem_diagnostics.json'), 'w') as f:
            json.dump(diagnostics_dict, f)

    # Write to file
    with GriddedNcdfFile(gdir=gdir, grid=dem_grid, fpath=fpath,
                         basename=output_filename, reset=True) as nc:

        v = nc.createVariable('topo', 'f4', ('y', 'x',), zlib=True)
        v.units = 'm'
        v.long_name = 'DEM topography'
        v[:] = dem

        v = nc.createVariable('topo_smoothed', 'f4', ('y', 'x',), zlib=True)
        v.units = 'm'
        v.long_name = ('DEM topography smoothed with radius: '
                       '{:.1} m'.format(cfg.PARAMS['smooth_window']))
        v[:] = smoothed_dem

        # If there was some invalid data store this as well
        v = nc.createVariable('topo_valid_mask', 'i1', ('y', 'x',), zlib=True)
        v.units = '-'
        v.long_name = 'DEM validity mask according to geotiff input (1-0)'
        v[:] = valid_mask.astype(int)

        # add some meta stats and close
        nc.max_h_dem = np.nanmax(dem)
        nc.min_h_dem = np.nanmin(dem)


@entity_task(log, writes=['gridded_data', 'geometries'])
def glacier_masks(gdir):
    """Makes a gridded mask of the glacier outlines that can be used by OGGM.

    For a more robust solution (not OGGM compatible) see simple_glacier_masks.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # In case nominal, just raise
    if gdir.is_nominal:
        raise GeometryError('{} is a nominal glacier.'.format(gdir.rgi_id))

    if not os.path.exists(gdir.get_filepath('gridded_data')):
        # In a possible future, we might actually want to raise a
        # deprecation warning here
        process_dem(gdir)

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

    # Write geometries
    geometries = dict()
    geometries['polygon_hr'] = glacier_poly_hr
    geometries['polygon_pix'] = glacier_poly_pix
    geometries['polygon_area'] = geometry.area
    gdir.write_pickle(geometries, 'geometries')

    # write out the grids in the netcdf file
    with GriddedNcdfFile(gdir) as nc:

        if 'glacier_mask' not in nc.variables:
            v = nc.createVariable('glacier_mask', 'i1', ('y', 'x', ),
                                  zlib=True)
            v.units = '-'
            v.long_name = 'Glacier mask'
        else:
            v = nc.variables['glacier_mask']
        v[:] = glacier_mask

        if 'glacier_ext' not in nc.variables:
            v = nc.createVariable('glacier_ext', 'i1', ('y', 'x', ),
                                  zlib=True)
            v.units = '-'
            v.long_name = 'Glacier external boundaries'
        else:
            v = nc.variables['glacier_ext']
        v[:] = glacier_ext

        dem = nc.variables['topo'][:]

        if 'topo_valid_mask' not in nc.variables:
            msg = ('You seem to be running from old preprocessed directories. '
                   'See https://github.com/OGGM/oggm/issues/1095 for a fix.')
            raise InvalidWorkflowError(msg)
        valid_mask = nc.variables['topo_valid_mask'][:]

        # Last sanity check based on the masked dem
        tmp_max = np.max(dem[np.where(glacier_mask == 1)])
        tmp_min = np.min(dem[np.where(glacier_mask == 1)])
        if tmp_max < (tmp_min + 0.1):
            raise InvalidDEMError('({}) min equal max in the masked DEM.'
                                  .format(gdir.rgi_id))

        # Log DEM that needed processing within the glacier mask
        if gdir.get_diagnostics().get('dem_needed_interpolation', False):
            pnan = (valid_mask == 0) & glacier_mask
            gdir.add_to_diagnostics('dem_invalid_perc_in_mask',
                                    np.sum(pnan) / np.sum(glacier_mask))

        # add some meta stats and close
        dem_on_g = dem[np.where(glacier_mask)]
        nc.max_h_glacier = np.nanmax(dem_on_g)
        nc.min_h_glacier = np.nanmin(dem_on_g)


@entity_task(log, writes=['gridded_data'])
def simple_glacier_masks(gdir):
    """Compute glacier masks based on much simpler rules than OGGM's default.

    This is therefore more robust: we use this task in a elev_bands workflow.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # In case nominal, just raise
    if gdir.is_nominal:
        raise GeometryError('{} is a nominal glacier.'.format(gdir.rgi_id))

    if not os.path.exists(gdir.get_filepath('gridded_data')):
        # In a possible future, we might actually want to raise a
        # deprecation warning here
        process_dem(gdir)

    # Geometries
    geometry = gdir.read_shapefile('outlines').geometry[0]

    # rio metadata
    with rasterio.open(gdir.get_filepath('dem'), 'r', driver='GTiff') as ds:
        data = ds.read(1).astype(rasterio.float32)
        profile = ds.profile

    # simple trick to correct invalid polys:
    # http://stackoverflow.com/questions/20833344/
    # fix-invalid-polygon-python-shapely
    geometry = geometry.buffer(0)
    if not geometry.is_valid:
        raise InvalidDEMError('This glacier geometry is not valid.')

    # Compute the glacier mask using rasterio
    # Small detour as mask only accepts DataReader objects
    profile['dtype'] = 'int16'
    profile.pop('nodata', None)
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data.astype(np.int16)[np.newaxis, ...])
        dem_data = rasterio.open(memfile.name)
        masked_dem, _ = riomask(dem_data, [shpg.mapping(geometry)],
                                filled=False)
    glacier_mask = ~masked_dem[0, ...].mask

    # Same without nunataks
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data.astype(np.int16)[np.newaxis, ...])
        dem_data = rasterio.open(memfile.name)
        try:
            poly = [shpg.mapping(shpg.Polygon(geometry.exterior))]
        except AttributeError:
            if not cfg.PARAMS['keep_multipolygon_outlines']:
                raise
            # special treatment for MultiPolygons
            parts = []
            for p in geometry.geoms:
                parts.append(p)
            parts = np.array(parts)

            poly = []
            for part in parts:
                poly.append(shpg.mapping(shpg.Polygon(part.exterior)))

        masked_dem, _ = riomask(dem_data, poly,
                                filled=False)
    glacier_mask_nonuna = ~masked_dem[0, ...].mask

    # Glacier exterior excluding nunataks
    erode = binary_erosion(glacier_mask_nonuna)
    glacier_ext = glacier_mask_nonuna ^ erode
    glacier_ext = np.where(glacier_mask_nonuna, glacier_ext, 0)

    dem = read_geotiff_dem(gdir)

    # Last sanity check based on the masked dem
    tmp_max = np.nanmax(dem[glacier_mask])
    tmp_min = np.nanmin(dem[glacier_mask])
    if tmp_max < (tmp_min + 1):
        raise InvalidDEMError('({}) min equal max in the masked DEM.'
                              .format(gdir.rgi_id))

    # write out the grids in the netcdf file
    with GriddedNcdfFile(gdir) as nc:

        if 'glacier_mask' not in nc.variables:
            v = nc.createVariable('glacier_mask', 'i1', ('y', 'x', ),
                                  zlib=True)
            v.units = '-'
            v.long_name = 'Glacier mask'
        else:
            v = nc.variables['glacier_mask']
        v[:] = glacier_mask

        if 'glacier_ext' not in nc.variables:
            v = nc.createVariable('glacier_ext', 'i1', ('y', 'x', ),
                                  zlib=True)
            v.units = '-'
            v.long_name = 'Glacier external boundaries'
        else:
            v = nc.variables['glacier_ext']
        v[:] = glacier_ext

        # Log DEM that needed processing within the glacier mask
        if 'topo_valid_mask' not in nc.variables:
            msg = ('You seem to be running from old preprocessed directories. '
                   'See https://github.com/OGGM/oggm/issues/1095 for a fix.')
            raise InvalidWorkflowError(msg)
        valid_mask = nc.variables['topo_valid_mask'][:]
        if gdir.get_diagnostics().get('dem_needed_interpolation', False):
            pnan = (valid_mask == 0) & glacier_mask
            gdir.add_to_diagnostics('dem_invalid_perc_in_mask',
                                    np.sum(pnan) / np.sum(glacier_mask))

        # add some meta stats and close
        nc.max_h_dem = np.nanmax(dem)
        nc.min_h_dem = np.nanmin(dem)
        dem_on_g = dem[np.where(glacier_mask)]
        nc.max_h_glacier = np.nanmax(dem_on_g)
        nc.min_h_glacier = np.nanmin(dem_on_g)

        # Last sanity check
        if nc.max_h_glacier < (nc.min_h_glacier + 1):
            raise InvalidDEMError('({}) min equal max in the masked DEM.'
                                  .format(gdir.rgi_id))


@entity_task(log, writes=['hypsometry'])
def compute_hypsometry_attributes(gdir, min_perc=0.2):
    """Adds some attributes to the glacier directory.

    Mostly useful for RGI stuff.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # First things first - delete hypsometry file
    hypso_path = gdir.get_filepath('hypsometry')
    if os.path.exists(hypso_path):
        os.remove(hypso_path)

    dem = read_geotiff_dem(gdir)

    # This is the very robust way
    fp = gdir.get_filepath('glacier_mask')
    with rasterio.open(fp, 'r', driver='GTiff') as ds:
        glacier_mask = ds.read(1).astype(rasterio.int16) == 1

    fp = gdir.get_filepath('glacier_mask', filesuffix='_exterior')
    with rasterio.open(fp, 'r', driver='GTiff') as ds:
        glacier_exterior_mask = ds.read(1).astype(rasterio.int16) == 1

    valid_mask = glacier_mask & np.isfinite(dem)
    # we cant proceed if we have very little info
    avail_perc = valid_mask.sum() / glacier_mask.sum()
    if avail_perc < min_perc:
        raise InvalidDEMError(f"Cant proceed with {avail_perc*100:.1f}%"
                              f"available.")

    bsize = 50.
    dem_on_ice = dem[valid_mask]
    if dem_on_ice.min() < -99:
        raise InvalidDEMError(f"Cant proceed with {dem_on_ice.min()}m "
                              f"minimum DEM elevation.")
    bins = np.arange(nicenumber(dem_on_ice.min(), bsize, lower=True),
                     nicenumber(dem_on_ice.max(), bsize) + 0.01, bsize)

    h, _ = np.histogram(dem_on_ice, bins)
    h = h / np.sum(h) * 1000  # in permil

    # We want to convert the bins to ints but preserve their sum to 1000
    # Start with everything rounded down, then round up the numbers with the
    # highest fractional parts until the desired sum is reached.
    hi = np.floor(h).astype(int)
    hup = np.ceil(h).astype(int)
    aso = np.argsort(hup - h)
    for i in aso:
        hi[i] = hup[i]
        if np.sum(hi) == 1000:
            break

    # slope
    sy, sx = np.gradient(dem, gdir.grid.dx)
    aspect = np.arctan2(np.nanmean(-sx[glacier_mask]), np.nanmean(sy[glacier_mask]))
    aspect = np.rad2deg(aspect)
    if aspect < 0:
        aspect += 360
    slope = np.arctan(np.sqrt(sx ** 2 + sy ** 2))
    avg_slope = np.rad2deg(np.nanmean(slope[glacier_mask]))

    sec_bins = -22.5 + 45 * np.arange(9)
    aspect_for_bin = aspect
    if aspect_for_bin >= sec_bins[-1]:
        aspect_for_bin -= 360
    aspect_sec = np.digitize(aspect_for_bin, sec_bins)
    dx2 = gdir.grid.dx**2 * 1e-6

    # Terminus loc
    min_ext = np.nanmin(dem[glacier_exterior_mask])
    if np.isfinite(min_ext):
        # Find it on exterior
        j, i = np.nonzero((min_ext == dem) & glacier_exterior_mask)
    else:
        # In some bad cases this might be nan - find it inside
        j, i = np.nonzero((dem[valid_mask].min() == dem) & valid_mask)

    if len(j) > 2:
        # We have a situation - take the closest to the euclidian center
        mi, mj = np.mean(i), np.mean(j)
        c = np.argmin((mi - i)**2 + (mj - j)**2)
        j, i = j[[c]], i[[c]]

    lon, lat = gdir.grid.ij_to_crs(i[0], j[0], crs=salem.wgs84)

    # write
    df = pd.DataFrame()
    df['rgi_id'] = [gdir.rgi_id]
    df['area_km2'] = [gdir.rgi_area_km2]
    df['area_grid_km2'] = [glacier_mask.sum() * dx2]
    df['valid_dem_perc'] = [avail_perc]
    df['grid_dx'] = [gdir.grid.dx]
    df['zmin_m'] = [np.nanmin(dem_on_ice)]
    df['zmax_m'] = [np.nanmax(dem_on_ice)]
    df['zmed_m'] = [np.nanmedian(dem_on_ice)]
    df['zmean_m'] = [np.nanmean(dem_on_ice)]
    df['terminus_lon'] = lon
    df['terminus_lat'] = lat
    df['slope_deg'] = [avg_slope]
    df['aspect_deg'] = [aspect]
    df['aspect_sec'] = [aspect_sec]
    df['dem_source'] = [gdir.get_diagnostics()['dem_source']]
    for b, bs in zip(hi, (bins[1:] + bins[:-1])/2):
        df['{}'.format(np.round(bs).astype(int))] = [b]
    df.to_csv(hypso_path, index=False)


@entity_task(log, writes=['glacier_mask'])
def rasterio_glacier_mask(gdir, source=None, no_nunataks=False, overwrite=True):
    """Writes a 1-0 glacier mask GeoTiff with the same dimensions as dem.tif

    If no_nunataks, does the same but without nunataks. Writes a file
    with the suffix "_no_nunataks" appended.


    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier in question
    source : str

        - None (default): the task reads `dem.tif` from the GDir root
        - 'ALL': try to open any folder from `utils.DEM_SOURCE` and use first
        - any of `utils.DEM_SOURCE`: try only that one
    overwrite : bool
        compute even if the file is already there
    """
    # No need to if already there
    filesuffix = '_no_nunataks' if no_nunataks else None
    if not overwrite and gdir.has_file('glacier_mask', filesuffix=filesuffix):
        return

    if source is None:
        dempath = gdir.get_filepath('dem')
    elif source in utils.DEM_SOURCES:
        dempath = os.path.join(gdir.dir, source, 'dem.tif')
    else:
        for src in utils.DEM_SOURCES:
            dempath = os.path.join(gdir.dir, src, 'dem.tif')
            if os.path.isfile(dempath):
                break

    if not os.path.isfile(dempath):
        raise ValueError('The specified source does not give a valid DEM file')

    # read dem profile
    with rasterio.open(dempath, 'r', driver='GTiff') as ds:
        profile = ds.profile
        # don't even bother reading the actual DEM, just mimic it
        data = np.zeros((ds.height, ds.width))

    # Read RGI outlines
    geometry = gdir.read_shapefile('outlines').geometry[0]

    # simple trick to correct invalid polys:
    # http://stackoverflow.com/questions/20833344/
    # fix-invalid-polygon-python-shapely
    geometry = geometry.buffer(0)
    if not geometry.is_valid:
        raise InvalidDEMError('This glacier geometry is not valid.')

    if no_nunataks:
        mapping = shpg.mapping(shpg.Polygon(geometry.exterior))
    else:
        mapping = shpg.mapping(geometry)

    # Compute the glacier mask using rasterio
    # Small detour as mask only accepts DataReader objects
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data.astype(profile['dtype'])[np.newaxis, ...])
        dem_data = rasterio.open(memfile.name)
        masked_dem, _ = riomask(dem_data, [mapping], filled=False)
    glacier_mask = ~masked_dem[0, ...].mask

    # parameters to for the new tif
    nodata = -32767
    dtype = rasterio.int16

    # let's use integer
    out = glacier_mask.astype(dtype)

    # and check for sanity
    if not np.all(np.unique(out) == np.array([0, 1])):
        raise InvalidDEMError('({}) masked DEM does not consist of 0/1 only.'
                              .format(gdir.rgi_id))

    # Update existing profile for output
    profile.update({
        'dtype': dtype,
        'nodata': nodata,
    })

    if no_nunataks:
        fp = gdir.get_filepath('glacier_mask', filesuffix='_no_nunataks')
    else:
        fp = gdir.get_filepath('glacier_mask')

    with rasterio.open(fp, 'w', **profile) as r:
        r.write(out.astype(dtype), 1)


@entity_task(log, writes=['glacier_mask'])
def rasterio_glacier_exterior_mask(gdir, overwrite=True):
    """Writes a 1-0 glacier exterior mask GeoTiff with the same dimensions as dem.tif

    This is the "one" grid point on the glacier exterior (ignoring nunataks).
    This is useful to know where the terminus might be, for example.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier in question
    overwrite : bool
        compute even if the file is already there
    """

    # No need to if already there
    if not overwrite and gdir.has_file('glacier_mask', filesuffix='_exterior'):
        return

    fp = gdir.get_filepath('glacier_mask', filesuffix='_no_nunataks')
    with rasterio.open(fp, 'r', driver='GTiff') as ds:
        glacier_mask_nonuna = ds.read(1).astype(rasterio.int16) == 1
        profile = ds.profile

    # Glacier exterior excluding nunataks
    erode = binary_erosion(glacier_mask_nonuna)
    glacier_ext = glacier_mask_nonuna ^ erode
    glacier_ext = np.where(glacier_mask_nonuna, glacier_ext, 0)

    # parameters to for the new tif
    fp = gdir.get_filepath('glacier_mask', filesuffix='_exterior')
    with rasterio.open(fp, 'w', **profile) as r:
        r.write(glacier_ext.astype(rasterio.int16), 1)


@entity_task(log, writes=['gridded_data'])
def gridded_attributes(gdir):
    """Adds attributes to the gridded file, useful for thickness interpolation.

    This could be useful for distributed ice thickness models.
    The raster data are added to the gridded_data file.

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
        salem.transform_geopandas(gdf, to_crs=gdir.grid, inplace=True)
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
    min_slope = np.deg2rad(cfg.PARAMS['distributed_inversion_min_slope'])
    slope_factor = utils.clip_array(slope, min_slope, np.pi/2)
    slope_factor = 1 / slope_factor**(glen_n / (glen_n+2))

    aspect = np.arctan2(-sx, sy)
    aspect[aspect < 0] += 2 * np.pi

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

        vn = 'slope'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True, fill_value=np.nan)
        v.units = 'rad'
        v.long_name = 'Local slope based on smoothed topography'
        v[:] = slope.astype(np.float32)

        vn = 'aspect'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True, fill_value=np.nan)
        v.units = 'rad'
        v.long_name = 'Local aspect based on smoothed topography'
        v[:] = aspect.astype(np.float32)

        vn = 'slope_factor'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True, fill_value=np.nan)
        v.units = '-'
        v.long_name = 'Slope factor as defined in Farinotti et al 2009'
        v[:] = slope_factor.astype(np.float32)

        vn = 'dis_from_border'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True, fill_value=np.nan)
        v.units = 'm'
        v.long_name = 'Distance from glacier boundaries'
        v[:] = dis_from_border.astype(np.float32)


def _all_inflows(cls, cl):
    """Find all centerlines flowing into the centerline examined.

    Parameters
    ----------
    cls : list
        all centerlines of the examined glacier
    cline : Centerline
        centerline to control

    Returns
    -------
    list of strings of centerlines
    """

    ixs = [str(cls.index(cl.inflows[i])) for i in range(len(cl.inflows))]
    for cl in cl.inflows:
        ixs.extend(_all_inflows(cls, cl))
    return ixs


@entity_task(log)
def gridded_mb_attributes(gdir):
    """Adds mass balance related attributes to the gridded data file.

    This could be useful for distributed ice thickness models.
    The raster data are added to the gridded_data file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    from oggm.core.massbalance import LinearMassBalance, ConstantMassBalance
    from oggm.core.centerlines import line_inflows

    # Get the input data
    with ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo_2d = nc.variables['topo_smoothed'][:]
        glacier_mask_2d = nc.variables['glacier_mask'][:]
        glacier_mask_2d = glacier_mask_2d == 1
        catchment_mask_2d = glacier_mask_2d * np.nan

    topo = topo_2d[glacier_mask_2d]

    # Prepare the distributed mass balance data
    rho = cfg.PARAMS['ice_density']
    dx2 = gdir.grid.dx ** 2

    # Linear
    def to_minimize(ela_h):
        mbmod = LinearMassBalance(ela_h[0])
        smb = mbmod.get_annual_mb(heights=topo)
        return np.sum(smb)**2
    ela_h = optimization.minimize(to_minimize, [0.], method='Powell')
    mbmod = LinearMassBalance(float(ela_h['x'][0]))
    lin_mb_on_z = mbmod.get_annual_mb(heights=topo) * cfg.SEC_IN_YEAR * rho
    if not np.isclose(np.sum(lin_mb_on_z), 0, atol=10):
        raise RuntimeError('Spec mass balance should be zero but is: {}'
                           .format(np.sum(lin_mb_on_z)))

    # Normal OGGM (a bit tweaked)
    def to_minimize(temp_bias):
        mbmod = ConstantMassBalance(gdir, temp_bias=temp_bias, y0=1995,
                                    check_calib_params=False)
        smb = mbmod.get_annual_mb(heights=topo)
        return np.sum(smb)**2
    opt = optimization.minimize(to_minimize, [0.], method='Powell')
    mbmod = ConstantMassBalance(gdir, temp_bias=float(opt['x'][0]), y0=1995,
                                check_calib_params=False)
    oggm_mb_on_z = mbmod.get_annual_mb(heights=topo) * cfg.SEC_IN_YEAR * rho
    if not np.isclose(np.sum(oggm_mb_on_z), 0, atol=10):
        raise RuntimeError('Spec mass balance should be zero but is: {}'
                           .format(np.sum(oggm_mb_on_z)))

    # Altitude based mass balance
    catch_area_above_z = topo * np.nan
    lin_mb_above_z = topo * np.nan
    oggm_mb_above_z = topo * np.nan
    for i, h in enumerate(topo):
        catch_area_above_z[i] = np.sum(topo >= h) * dx2
        lin_mb_above_z[i] = np.sum(lin_mb_on_z[topo >= h]) * dx2
        oggm_mb_above_z[i] = np.sum(oggm_mb_on_z[topo >= h]) * dx2

    # Make 2D again
    def _fill_2d_like(data):
        out = topo_2d * np.nan
        out[glacier_mask_2d] = data
        return out

    catch_area_above_z = _fill_2d_like(catch_area_above_z)
    lin_mb_above_z = _fill_2d_like(lin_mb_above_z)
    oggm_mb_above_z = _fill_2d_like(oggm_mb_above_z)

    # Save to file
    with ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'catchment_area'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True, fill_value=np.nan)
        v.units = 'm^2'
        v.long_name = 'Catchment area above point'
        v.description = ('This is a very crude method: just the area above '
                         'the points elevation on glacier.')
        v[:] = catch_area_above_z.astype(np.float32)

        vn = 'lin_mb_above_z'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True, fill_value=np.nan)
        v.units = 'kg/year'
        v.long_name = 'MB above point from linear MB model, without catchments'
        v.description = ('Mass balance cumulated above the altitude of the'
                         'point, hence in unit of flux. Note that it is '
                         'a coarse approximation of the real flux. '
                         'The mass balance model is a simple linear function'
                         'of altitude.')
        v[:] = lin_mb_above_z.astype(np.float32)

        vn = 'oggm_mb_above_z'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True, fill_value=np.nan)
        v.units = 'kg/year'
        v.long_name = 'MB above point from OGGM MB model, without catchments'
        v.description = ('Mass balance cumulated above the altitude of the'
                         'point, hence in unit of flux. Note that it is '
                         'a coarse approximation of the real flux. '
                         'The mass balance model is a calibrated temperature '
                         'index model like OGGM.')
        v[:] = oggm_mb_above_z.astype(np.float32)

    # Hardest part - MB per catchment
    try:
        cls = gdir.read_pickle('centerlines')
    except FileNotFoundError:
        return

    # Make everything we need flat
    # Catchment areas
    cis = gdir.read_pickle('geometries')['catchment_indices']
    for j, ci in enumerate(cis):
        catchment_mask_2d[tuple(ci.T)] = j

    catchment_mask = catchment_mask_2d[glacier_mask_2d].astype(int)

    catchment_area = topo * np.nan
    lin_mb_above_z_on_catch = topo * np.nan
    oggm_mb_above_z_on_catch = topo * np.nan

    # First, find all inflows indices and min altitude per catchment
    inflows = []
    lowest_h = []
    for i, cl in enumerate(cls):
        lowest_h.append(np.min(topo[catchment_mask == i]))
        inflows.append([cls.index(l) for l in line_inflows(cl, keep=False)])

    for i, (catch_id, h) in enumerate(zip(catchment_mask, topo)):

        # Find the catchment area of the point itself by eliminating points
        # below the point altitude. We assume we keep all of them first,
        # then remove those we don't want
        sel_catchs = inflows[catch_id].copy()
        for catch in inflows[catch_id]:
            if h >= lowest_h[catch]:
                for cc in np.append(inflows[catch], catch):
                    try:
                        sel_catchs.remove(cc)
                    except ValueError:
                        pass

        # At the very least we need or own catchment
        sel_catchs.append(catch_id)

        # Then select all the catchment points
        sel_points = np.isin(catchment_mask, sel_catchs)

        # And keep the ones above our altitude
        sel_points = sel_points & (topo >= h)

        # Compute
        lin_mb_above_z_on_catch[i] = np.sum(lin_mb_on_z[sel_points]) * dx2
        oggm_mb_above_z_on_catch[i] = np.sum(oggm_mb_on_z[sel_points]) * dx2
        catchment_area[i] = np.sum(sel_points) * dx2

    catchment_area = _fill_2d_like(catchment_area)
    lin_mb_above_z_on_catch = _fill_2d_like(lin_mb_above_z_on_catch)
    oggm_mb_above_z_on_catch = _fill_2d_like(oggm_mb_above_z_on_catch)

    # Save to file
    with ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        vn = 'catchment_area_on_catch'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True, fill_value=np.nan)
        v.units = 'm^2'
        v.long_name = 'Catchment area above point on flowline catchments'
        v.description = ('Uses the catchments masks of the flowlines to '
                         'compute the area above the altitude of the given '
                         'point.')
        v[:] = catchment_area.astype(np.float32)

        vn = 'lin_mb_above_z_on_catch'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True, fill_value=np.nan)
        v.units = 'kg/year'
        v.long_name = 'MB above point from linear MB model, with catchments'
        v.description = ('Mass balance cumulated above the altitude of the'
                         'point in a flowline catchment, hence in unit of '
                         'flux. Note that it is a coarse approximation of the '
                         'real flux. The mass balance model is a simple '
                         'linear function of altitude.')
        v[:] = lin_mb_above_z_on_catch.astype(np.float32)

        vn = 'oggm_mb_above_z_on_catch'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True, fill_value=np.nan)
        v.units = 'kg/year'
        v.long_name = 'MB above point from OGGM MB model, with catchments'
        v.description = ('Mass balance cumulated above the altitude of the'
                         'point in a flowline catchment, hence in unit of '
                         'flux. Note that it is a coarse approximation of the '
                         'real flux. The mass balance model is a calibrated '
                         'temperature index model like OGGM.')
        v[:] = oggm_mb_above_z_on_catch.astype(np.float32)


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
    dem = read_geotiff_dem(gdir)

    if np.min(dem) == np.max(dem):
        raise RuntimeError('({}) min equal max in the DEM.'
                           .format(gdir.rgi_id))

    # Clip topography to 0 m a.s.l.
    utils.clip_min(dem, 0, out=dem)

    # Interpolate shape to a regular path
    glacier_poly_hr = tolist(geometry)

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

    # Rounded geometry to nearest pix
    # I can not use _polyg
    # glacier_poly_pix = _polygon_to_pix(glacier_poly_hr)
    def project(x, y):
        return np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)

    glacier_poly_pix = shapely.ops.transform(project, glacier_poly_hr)
    glacier_poly_pix_iter = tolist(glacier_poly_pix)

    # Compute the glacier mask (currently: center pixels + touched)
    nx, ny = gdir.grid.nx, gdir.grid.ny
    glacier_mask = np.zeros((ny, nx), dtype=np.uint8)
    glacier_ext = np.zeros((ny, nx), dtype=np.uint8)

    for poly in glacier_poly_pix_iter:
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
    with GriddedNcdfFile(gdir, reset=True) as nc:

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
        nc.max_h_dem = np.nanmax(dem)
        nc.min_h_dem = np.nanmin(dem)
        dem_on_g = dem[np.where(glacier_mask)]
        nc.max_h_glacier = np.max(dem_on_g)
        nc.min_h_glacier = np.min(dem_on_g)

    geometries = dict()
    geometries['polygon_hr'] = glacier_poly_hr
    geometries['polygon_pix'] = glacier_poly_pix
    geometries['polygon_area'] = geometry.area
    gdir.write_pickle(geometries, 'geometries')


@entity_task(log)
def gridded_data_var_to_geotiff(gdir, varname, fname=None, output_folder=None):
    """Writes a NetCDF variable to a georeferenced geotiff file.

    The geotiff file will be written in the gdir directory or a specified folder.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    varname : str
        variable name in gridded_data.nc
    fname : str
        output file name (should end with `tif`), default is `varname.tif`
    output_folder : str
        optional path to write the geotiff file. If None, writes to gdir.dir.
        If provided, files will be organized into subfolders based on RGI ID
        (e.g., RGI60-11/RGI60-11.00/RGI60-11.00897_varname.tif)
    """

    # Assign the output path
    if fname is None:
        fname = f'{gdir.rgi_id}_{varname}.tif'

    if output_folder is not None:
        # Create subfolder structure based on RGI ID
        # RGI6: RGI60-11.00897 -> RGI60-11 (8 chars) / RGI60-11.00 (11 chars)
        # RGI7: RGI2000-v7.0-G-01-00001 -> RGI2000-v7.0-G-01 (17 chars) / RGI2000-v7.0-G-01-00 (20 chars)
        rid = gdir.rgi_id
        # Determine folder structure based on RGI ID length
        if len(rid) <= 14:
            # RGI6 format
            base_dir = os.path.join(output_folder, rid[:8], rid[:11])
        else:
            # RGI7 format
            base_dir = os.path.join(output_folder, rid[:17], rid[:20])
        utils.mkdir(base_dir)
    else:
        base_dir = gdir.dir

    outpath = os.path.join(base_dir, fname)

    # Locate gridded_data.nc file and read it
    nc_path = gdir.get_filepath('gridded_data')
    with xr.open_dataset(nc_path) as ds:

        # Prepare the profile dict
        crs = ds.pyproj_srs
        var = ds[varname]
        grid = ds.salem.grid.corner_grid
        data = var.data
        data_type = data.dtype.name
        height, width = var.data.shape
        dx, dy = grid.dx, grid.dy
        x0, y0 = grid.x0, grid.y0

        profile = {'driver': 'GTiff', 'dtype': data_type, 'nodata': None,
                   'width': width, 'height': height, 'count': 1,
                   'crs': crs,
                   'transform': rasterio.Affine(dx, 0.0, x0,
                                                0.0, dy, y0),
                   'tiled': True,
                   'interleave': 'band'}

        # Write GeoTiff file
        with rasterio.open(outpath, 'w', **profile) as dst:
            dst.write(data, 1)


@entity_task(log)
def reproject_gridded_data_variable_to_grid(gdir,
                                            variable,
                                            target_grid,
                                            filename='gridded_data',
                                            filesuffix='',
                                            use_glacier_mask=True,
                                            interp='nearest',
                                            preserve_totals=True,
                                            smooth_radius=None,
                                            slice_of_variable=None):
    """
    Function for reprojecting a gridded data variable to a different grid.
    Useful when combining gridded data from different gdirs (see
    workflow.merge_gridded_data).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        Defines from where we should open the gridded data file.
    variable : str
        The variable for reprojection.
    target_grid : salem.gis.Grid
        The target grid for reprojection.
    filename : str
        The filename of the gridded data file. Default is gridded_data.
    filesuffix : str
        The filesuffix of filename. Default is ''.
    use_glacier_mask : bool
        If True only the data cropped by the glacier mask is included in the
        merged file. You must make sure that the variable 'glacier_mask' exists
        in the input file (which is the oggm default). Default is True.
    interp : str
        The interpolation method used by salem.Grid.map_gridded_data. Currently
        available 'nearest' (default), 'linear', or 'spline'.
    preserve_totals : bool
        If True we preserve the total value of variable if variable is a float.
        The total value is defined as the sum of all grid cell values times the
        area of the grid cell (e.g. preserving ice volume).
        Default is True.
    smooth_radius : int
        pixel size of the gaussian smoothing, only used if preserve_totals is
        True. Default is to use cfg.PARAMS['smooth_window'] (i.e. a size in
        meters). Set to zero to suppress smoothing.
    slice_of_variable : None | dict
        Can provide dimensions with values as a dictionary for extracting only
        a slice of the data before reprojecting. This can be useful for large
        datasets or if only part of the data is of interest. Default is None.

    Returns
    -------
    r_data : np.ndarray
        The reprojected data as a np.array with spatial-dimensions defined by
        target_grid.
    """
    with xr.open_dataset(gdir.get_filepath(filename,
                                           filesuffix=filesuffix)) as ds:
        if use_glacier_mask:
            # transpose is used to keep dimension order for 3d data, important
            # for map_gridded_data which expects the last two dimensions are y
            # and x
            data = xr.where(ds['glacier_mask'], ds[variable],
                            0).transpose(*ds[variable].dims)
        else:
            data = ds[variable]

        if slice_of_variable is not None:
            data = data.sel(**slice_of_variable)

        r_data = target_grid.map_gridded_data(data,
                                              grid=gdir.grid,
                                              interp=interp,).filled(0)

    if preserve_totals:
        # only preserve for float variables
        if not np.issubdtype(data, np.integer):
            # do we want to do smoothing?

            if len(data.dims) == 2:
                sum_axis = None
            elif len(data.dims) == 3:
                # for 3D data we want to preserve value for each time step
                sum_axis = (1, 2)
            else:
                raise NotImplementedError(f'len(data.dims) = {len(data.dims)}')

            total_before = (np.nansum(data.values, axis=sum_axis) *
                            ds.salem.grid.dx ** 2)

            total_before_is_zero = np.isclose(total_before, 0, atol=1e-6)

            if smooth_radius != 0:
                if smooth_radius is None:
                    smooth_radius = np.rint(cfg.PARAMS['smooth_window'] /
                                            target_grid.dx)
                # use data_mask to not expand the extent of the data
                data_mask = ~np.isclose(r_data, 0, atol=1e-6)
                if r_data.ndim == 3:
                    r_data = np.array(
                        [gaussian_blur(r_data[i, :, :], int(smooth_radius))
                         for i in range(r_data.shape[0])])
                else:
                    r_data = gaussian_blur(r_data, int(smooth_radius))
                r_data[~data_mask] = 0

            total_after = (np.nansum(r_data, axis=sum_axis) *
                           target_grid.dx ** 2)
            total_after_is_zero = np.isclose(total_after, 0, atol=1e-6)

            # if a relatively small grid is reprojected into a larger grid, it
            # could happen that no data is assigned at all
            no_data_after_but_before = np.logical_and(total_after_is_zero,
                                                      ~total_before_is_zero)
            if np.any(no_data_after_but_before):
                # we just assign the maximum value to one grid point and use the
                # factor for conserving the total value
                def _assign_max_value(data_provided, data_target):
                    j_max, i_max = np.unravel_index(
                        np.nanargmax(data_provided.values), data_provided.shape)
                    oi_max, oj_max = target_grid.center_grid.transform(
                        i_max, j_max, crs=gdir.grid.center_grid, nearest=True)
                    data_target[oj_max, oi_max] = data_provided[j_max, i_max]
                    return data_target

                if r_data.ndim == 3:
                    for i in range(r_data.shape[0]):
                        if no_data_after_but_before[i]:
                            r_data[i, :, :] = _assign_max_value(data[i, :, :],
                                                                r_data[i, :, :])
                else:
                    r_data = _assign_max_value(data, r_data)

                # and recalculate the total after again
                total_after = (np.nansum(r_data, axis=sum_axis) *
                               target_grid.dx ** 2)

            # only preserve total if there is some data before
            with warnings.catch_warnings():
                # Divide by zero is fine
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                factor = np.where(total_before_is_zero,
                                  0., total_before / total_after)

            if len(data.dims) == 3:
                # need to add two axis for broadcasting
                factor = factor[:, np.newaxis, np.newaxis]

            r_data *= factor

    return r_data


@entity_task(log, writes=['gridded_data', 'complex_sub_entities'])
def rgi7g_to_complex(gdir, rgi7g_file=None, rgi7c_to_g_links=None):
    """Adds the individual glacier outlines to this glacier complex gdir.

    Also adds a mask to gridded_data.nc with number indicating the
    respective subentity index in the shapefile (-1 means no glacier).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory` object
        the glacier directory to process
    rgi7g_file : gpd.GeoDataFrame
        the RGI7G file to extract the outlines from (we can read it
        from disk but if you give it, this may faster for large number
        of glaciers)
    rgi7c_to_g_links : gpd.GeoDataFrame
        the RGI7G file to extract the outlines from (we can read it
        from disk but if you give it, this may faster for large number
        of glaciers)
    """

    if not gdir.rgi_version == '70C':
        raise InvalidWorkflowError('Needs to be run on a glacier complex!')

    if rgi7g_file is None:
        from oggm.utils import get_rgi_region_file
        rgi7g_file = get_rgi_region_file(gdir.rgi_region, version='70G')
        rgi7g_file = gpd.read_file(rgi7g_file)
    if rgi7c_to_g_links is None:
        from oggm.utils import get_rgi7c_to_g_links
        rgi7c_to_g_links = get_rgi7c_to_g_links(gdir.rgi_region)

    subset = rgi7g_file.loc[rgi7g_file.rgi_id.isin(rgi7c_to_g_links[gdir.rgi_id])]
    subset = subset.reset_index(drop=True)

    # Reproject and write
    subset = subset.to_crs(gdir.grid.proj.srs)
    gdir.write_shapefile(subset, 'complex_sub_entities')

    # OK all good, now the mask
    # Load the DEM
    # TODO: this is really really unnecessary, we should have a better way
    with rasterio.open(gdir.get_filepath('dem'), 'r', driver='GTiff') as ds:
        data = ds.read(1).astype(rasterio.float32)
        profile = ds.profile

    # Initialize the mask with -1s, the same shape as the DEM
    mask = np.full(data.shape, -1, dtype=np.int16)

    # Compute the glacier mask using rasterio
    # Small detour as mask only accepts DataReader objects
    profile['dtype'] = 'int16'
    profile.pop('nodata', None)
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data.astype(np.int16)[np.newaxis, ...])
        dem_data = rasterio.open(memfile.name)

        # Iterate over each polygon, rasterizing it onto the mask
        for index, geometry in enumerate(subset.geometry):
            # Correct invalid polygons
            geometry = geometry.buffer(0)
            if not geometry.is_valid:
                raise Exception('Invalid geometry.')
            masked_dem, _ = riomask(dem_data, [shpg.mapping(geometry)],
                                    filled=False)
            glacier_mask = ~masked_dem[0, ...].mask

            # Update the mask: only change -1 to the new index
            mask = np.where((mask == -1) & glacier_mask, index, mask)

    grids_file = gdir.get_filepath('gridded_data')
    with ncDataset(grids_file, 'a') as nc:

        vn = 'sub_entities'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'i4', ('y', 'x', ))
        v.units = '-'
        v.long_name = 'Sub-entities glacier mask (number is index)'
        v[:] = mask
