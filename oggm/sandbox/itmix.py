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
import matplotlib.pyplot as plt
# Locals
from oggm import entity_task
import oggm.cfg as cfg
from oggm.core.preprocessing.gis import _gaussian_blur, _mask_per_divide
from oggm.sandbox import itmix_cfg
from oggm import utils

import fiona

# Module logger
log = logging.getLogger(__name__)

# Needed later
label_struct = np.ones((3, 3))

# Globals
import glob
import pandas as pd
RGI_DIR = '/home/mowglie/disk/Dropbox/Share/oggm-data/rgi/v5'
DATA_DIR = '/home/mowglie/disk/Dropbox/Share/oggm-data'


def get_rgi_df(reset=False):
    # This makes an RGI dataframe with all ITMIX + WGMS + GTD glaciers

    df_rgi_file = os.path.expanduser('~/itmix_rgi_shp.pkl')
    if os.path.exists(df_rgi_file) and not reset:
        rgidf = pd.read_pickle(df_rgi_file)
    else:
        linkf = os.path.join(DATA_DIR, 'itmix', 'itmix_rgi_links.pkl')
        df_itmix = pd.read_pickle(linkf)

        f, d = utils.get_wgms_files()
        wgms_df = pd.read_csv(f)

        f = utils.get_glathida_file()
        gtd_df = pd.read_csv(f)

        rgidf = []
        _rgi_ids = []
        for i, row in df_itmix.iterrows():
            # read the rgi region
            rgi_shp = os.path.join(RGI_DIR, "*",
                                   row['rgi_reg'] + '_rgi50_*.shp')
            rgi_shp = list(glob.glob(rgi_shp))[0]
            rgi_df = salem.utils.read_shapefile(rgi_shp, cached=True)

            rgi_parts = row.T['rgi_parts_ids']
            sel = rgi_df.loc[rgi_df.RGIId.isin(rgi_parts)].copy()
            _rgi_ids.extend(rgi_parts)

            # use the ITMIX shape where possible
            if row.name in ['Hellstugubreen', 'Freya', 'Aqqutikitsoq',
                            'Brewster', 'Kesselwandferner', 'NorthGlacier',
                            'SouthGlacier', 'Tasman', 'Unteraar',
                            'Washmawapta']:
                for shf in glob.glob(itmix_cfg.itmix_data_dir + '*/*/*_' +
                                             row.name + '*.shp'):
                    pass
                shp = salem.utils.read_shapefile(shf)
                if row.name == 'Unteraar':
                    shp = shp.iloc[[-1]]
                if 'LineString' == shp.iloc[0].geometry.type:
                    shp.loc[shp.index[0], 'geometry'] = shpg.Polygon(shp.iloc[0].geometry)
                assert len(shp) == 1
                area_km2 = shp.iloc[0].geometry.area * 1e-6
                shp = salem.gis.transform_geopandas(shp)
                shp = shp.iloc[0].geometry
                sel = sel.iloc[[0]]
                sel.loc[sel.index[0], 'geometry'] = shp
                sel.loc[sel.index[0], 'Area'] = area_km2
            elif row.name == 'Urumqi':
                # ITMIX Urumqi is in fact two glaciers
                for shf in glob.glob(itmix_cfg.itmix_data_dir + '*/*/*_' +
                                             row.name + '*.shp'):
                    pass
                shp2 = salem.utils.read_shapefile(shf)
                assert len(shp2) == 2
                for k in [0, 1]:
                    shp = shp2.iloc[[k]].copy()
                    area_km2 = shp.iloc[0].geometry.area * 1e-6
                    shp = salem.gis.transform_geopandas(shp)
                    shp = shp.iloc[0].geometry
                    assert sel.loc[sel.index[k], 'geometry'].contains(shp.centroid)
                    sel.loc[sel.index[k], 'geometry'] = shp
                    sel.loc[sel.index[k], 'Area'] = area_km2
                assert len(sel) == 2
            else:
                pass

            # add glacier name to the entity
            name = ['I:' + row.name] * len(sel)
            add_n = sel.RGIId.isin(wgms_df.RGI_ID.values)
            for z, it in enumerate(add_n.values):
                if it:
                    name[z] = 'W-' + name[z]
            add_n = sel.RGIId.isin(gtd_df.RGI_ID.values)
            for z, it in enumerate(add_n.values):
                if it:
                    name[z] = 'G-' + name[z]
            sel.loc[:, 'Name'] = name
            rgidf.append(sel)

        # WGMS glaciers which are not already there
        # Actually we should remove the data of those 7 to be honest...
        f, d = utils.get_wgms_files()
        wgms_df = pd.read_csv(f)
        print('N WGMS before: {}'.format(len(wgms_df)))
        wgms_df = wgms_df.loc[~ wgms_df.RGI_ID.isin(_rgi_ids)]
        print('N WGMS after: {}'.format(len(wgms_df)))

        for i, row in wgms_df.iterrows():
            rid = row.RGI_ID
            reg = rid.split('-')[1].split('.')[0]
            # read the rgi region
            rgi_shp = os.path.join(RGI_DIR, "*",
                                   reg + '_rgi50_*.shp')
            rgi_shp = list(glob.glob(rgi_shp))[0]
            rgi_df = salem.utils.read_shapefile(rgi_shp, cached=True)

            sel = rgi_df.loc[rgi_df.RGIId.isin([rid])].copy()
            assert len(sel) == 1

            # add glacier name to the entity
            _cor = row.NAME.replace('/', 'or').replace('.', '').replace(' ', '-')
            name = ['W:' + _cor] * len(sel)
            add_n = sel.RGIId.isin(gtd_df.RGI_ID.values)
            for z, it in enumerate(add_n.values):
                if it:
                    name[z] = 'G-' + name[z]
            for n in name:
                if len(n) > 48:
                    raise
            sel.loc[:, 'Name'] = name
            rgidf.append(sel)

        _rgi_ids.extend(wgms_df.RGI_ID.values)

        # GTD glaciers which are not already there
        # Actually we should remove the data of those 2 to be honest...
        print('N GTD before: {}'.format(len(gtd_df)))
        gtd_df = gtd_df.loc[~ gtd_df.RGI_ID.isin(_rgi_ids)]
        print('N GTD after: {}'.format(len(gtd_df)))

        for i, row in gtd_df.iterrows():
            rid = row.RGI_ID
            reg = rid.split('-')[1].split('.')[0]
            # read the rgi region
            rgi_shp = os.path.join(RGI_DIR, "*",
                                   reg + '_rgi50_*.shp')
            rgi_shp = list(glob.glob(rgi_shp))[0]
            rgi_df = salem.utils.read_shapefile(rgi_shp, cached=True)

            sel = rgi_df.loc[rgi_df.RGIId.isin([rid])].copy()
            assert len(sel) == 1

            # add glacier name to the entity
            _corname = row.NAME.replace('/', 'or').replace('.', '').replace(' ', '-')
            name = ['G:' + _corname] * len(sel)
            for n in name:
                if len(n) > 48:
                    raise
            sel.loc[:, 'Name'] = name
            rgidf.append(sel)

        # Save for not computing each time
        rgidf = pd.concat(rgidf)
        rgidf.to_pickle(df_rgi_file)

    return rgidf


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
        msg = gdir.rgi_id + ': DEM needed interpolation'
        msg += '({:.1f}% missing).'.format(len(pnan[0])/len(dem.flatten())*100)
        log.warning(msg)
    if np.min(dem) == np.max(dem):
        raise RuntimeError(gdir.rgi_id + ': min equal max in the DEM.')

    # Replace DEM values with ITMIX ones where possible
    # Open DEM
    dem_f = None
    n_g = gdir.name.split(':')[-1]
    for dem_f in glob.glob(itmix_cfg.itmix_data_dir + '/*/02_surface_' +
                           n_g + '*.asc'):
        pass

    if dem_f is not None:
        log.debug('%s: ITMIX DEM file: %s', gdir.rgi_id, dem_f)
        it_dem_ds = EsriITMIX(dem_f)
        it_dem = it_dem_ds.get_vardata()
        it_dem = np.where(it_dem < -999., np.NaN, it_dem)

        # for some glaciers, trick
        if n_g in ['Academy', 'Devon']:
            it_dem = np.where(it_dem <= 0, np.NaN, it_dem)
            it_dem = np.where(np.isfinite(it_dem), it_dem, np.nanmin(it_dem))
        if n_g in ['Brewster', 'Austfonna']:
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


def correct_dem(gdir, glacier_mask, dem, smoothed_dem):
    """Compare with Huss and stuff."""

    dem_glac = dem[np.nonzero(glacier_mask)]

    # Read RGI hypso for compa
    tosearch = '{:02d}'.format(np.int(gdir.rgi_region))
    tosearch = os.path.join(RGI_DIR, '*', tosearch + '*_hypso.csv')
    for fh in glob.glob(tosearch):
        pass
    df = pd.read_csv(fh)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[df.RGIId.isin([gdir.rgi_id])]
    df = df[df.columns[3:]].T
    df.columns = ['RGI (Huss)']
    hs = np.asarray(df.index.values, np.int)
    bins = utils.nicenumber(hs, 50, lower=True)
    bins = np.append(bins, bins[-1] + 50)
    myhist, _ = np.histogram(dem_glac, bins=bins)
    myhist = myhist / np.sum(myhist) * 1000
    df['OGGM'] = myhist
    df = df / 10
    df.index.rename('Alt (m)', inplace=True)
    df.plot()
    plt.ylabel('Freq (%)')
    plt.savefig('/home/mowglie/hypso_' + gdir.rgi_id + '.png')

    minz = None
    if gdir.rgi_id == 'RGI50-06.00424': minz = 800
    if gdir.rgi_id == 'RGI50-06.00443': minz = 600

    return dem, smoothed_dem
