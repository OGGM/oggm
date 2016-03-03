from __future__ import absolute_import, division

# Built ins
import os
import logging
from shutil import copyfile
from functools import partial
import glob
# External libs
from osgeo import osr
import salem
from salem.datasets import EsriITMIX
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
from oggm.core.preprocessing.gis import _gaussian_blur, _mask_per_divide
from oggm.sandbox import itmix_cfg

# Module logger
log = logging.getLogger(__name__)

# Needed later
label_struct = np.ones((3, 3))


@entity_task(log, writes=['gridded_data', 'geometries'])
def glacier_masks_itmix(gdir):
    """Converts the glacier vector geometries to grids.

    Uses where possible the ITMIX DEM

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
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

    # Replace DEM values with ITMIX ones where possible
    # Open DEM
    dem_f = None
    for dem_f in glob.glob(itmix_cfg.itmix_data_dir + '/*/02_surface_' +
                                  gdir.name + '*.asc'):
        pass

    if dem_f is not None:
        log.debug('%s: ITMIX DEM file: %s', gdir.rgi_id, dem_f)
        it_dem_ds = EsriITMIX(dem_f)
        it_dem = it_dem_ds.get_vardata()
        it_dem = np.where(it_dem < -999., np.NaN, it_dem)

        # for some glaciers, trick
        if gdir.name in ['Academy', 'Devon']:
            it_dem = np.where(it_dem <= 0, np.NaN, it_dem)
            it_dem = np.where(np.isfinite(it_dem), it_dem, np.nanmin(it_dem))
        if gdir.name in ['Brewster', 'Austfonna']:
            it_dem = np.where(it_dem <= 0, np.NaN, it_dem)

        # Transform to local grid
        it_dem = gdir.grid.map_gridded_data(it_dem, it_dem_ds.grid,
                                            interp='linear')
        # And update values where possible
        dem = np.where(~ it_dem.mask, it_dem, dem)

    # Disallow negative
    dem = dem.clip(0)

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

    # Smooth SRTM?
    if cfg.PARAMS['smooth_window'] > 0.:
        gsize = np.rint(cfg.PARAMS['smooth_window'] / dx)
        smoothed_dem = _gaussian_blur(dem, np.int(gsize))
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
