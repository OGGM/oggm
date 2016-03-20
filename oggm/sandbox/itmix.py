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
import glob
import pandas as pd
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
from oggm.sandbox.itmix_cfg import DATA_DIR
from oggm import utils

import fiona

# Module logger
log = logging.getLogger(__name__)

# Needed later
label_struct = np.ones((3, 3))

# Globals
SEARCHD = os.path.join(DATA_DIR, 'itmix', 'glaciers_sorted')

def get_rgi_df(reset=False):

    # This makes an RGI dataframe with all ITMIX + WGMS + GTD glaciers
    RGI_DIR = utils.get_rgi_dir()

    df_rgi_file = os.path.join(DATA_DIR, 'itmix', 'itmix_rgi_shp.pkl')
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
                            'Washmawapta', 'Columbia']:
                for shf in glob.glob(SEARCHD + '/*/*/*_' +
                                             row.name + '*.shp'):
                    pass
                shp = salem.utils.read_shapefile(shf)
                if row.name == 'Unteraar':
                    shp = shp.iloc[[-1]]
                if 'LineString' == shp.iloc[0].geometry.type:
                    shp.loc[shp.index[0], 'geometry'] = shpg.Polygon(shp.iloc[0].geometry)
                if shp.iloc[0].geometry.type == 'MultiLineString':
                    # Columbia
                    geometry = shp.iloc[0].geometry
                    parts = list(geometry)
                    for p in parts:
                        assert p.type == 'LineString'
                    exterior = shpg.Polygon(parts[0])
                    # let's assume that all other polygons are in fact interiors
                    interiors = []
                    for p in parts[1:]:
                        assert exterior.contains(p)
                        interiors.append(p)
                    geometry = shpg.Polygon(parts[0], interiors)
                    assert 'Polygon' in geometry.type
                    shp.loc[shp.index[0], 'geometry'] = geometry

                assert len(shp) == 1
                area_km2 = shp.iloc[0].geometry.area * 1e-6
                shp = salem.gis.transform_geopandas(shp)
                shp = shp.iloc[0].geometry
                sel = sel.iloc[[0]]
                sel.loc[sel.index[0], 'geometry'] = shp
                sel.loc[sel.index[0], 'Area'] = area_km2
            elif row.name == 'Urumqi':
                # ITMIX Urumqi is in fact two glaciers
                for shf in glob.glob(SEARCHD + '/*/*/*_' +
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
    searchf = os.path.join(DATA_DIR, 'itmix', 'glaciers_sorted', '*')
    searchf = os.path.join(searchf, '02_surface_' + n_g + '*.asc')
    for dem_f in glob.glob(searchf):
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


# def correct_dem(gdir, glacier_mask, dem, smoothed_dem):
#     """Compare with Huss and stuff."""
#
#     dem_glac = dem[np.nonzero(glacier_mask)]
#     RGI_DIR = utils.get_rgi_dir()
#
#     # Read RGI hypso for compa
#     tosearch = '{:02d}'.format(np.int(gdir.rgi_region))
#     tosearch = os.path.join(RGI_DIR, '*', tosearch + '*_hypso.csv')
#     for fh in glob.glob(tosearch):
#         pass
#     df = pd.read_csv(fh)
#     df.columns = [c.strip() for c in df.columns]
#     df = df.loc[df.RGIId.isin([gdir.rgi_id])]
#     df = df[df.columns[3:]].T
#     df.columns = ['RGI (Huss)']
#     hs = np.asarray(df.index.values, np.int)
#     bins = utils.nicenumber(hs, 50, lower=True)
#     bins = np.append(bins, bins[-1] + 50)
#     myhist, _ = np.histogram(dem_glac, bins=bins)
#     myhist = myhist / np.sum(myhist) * 1000
#     df['OGGM'] = myhist
#     df = df / 10
#     df.index.rename('Alt (m)', inplace=True)
#     df.plot()
#     plt.ylabel('Freq (%)')
#     plt.close()
#
#     minz = None
#     if gdir.rgi_id == 'RGI50-06.00424': minz = 800
#     if gdir.rgi_id == 'RGI50-06.00443': minz = 600
#
#     return dem, smoothed_dem

from oggm.core.preprocessing.inversion import invert_parabolic_bed
from scipy import optimize as optimization


def _prepare_inv(gdirs):

    # Get test glaciers (all glaciers with thickness data)
    fpath = utils.get_glathida_file()

    try:
        gtd_df = pd.read_csv(fpath).sort_values(by=['RGI_ID'])
    except AttributeError:
        gtd_df = pd.read_csv(fpath).sort(columns=['RGI_ID'])
    dfids = gtd_df['RGI_ID'].values

    print('GTD Glac before', len(dfids))
    ref_gdirs = []
    for gdir in gdirs:
        if gdir.rgi_id not in dfids:
            continue
        if gdir.glacier_type == 'Ice cap':
            continue
        if gdir.terminus_type in ['Marine-terminating', 'Lake-terminating',
                                  'Dry calving', 'Regenerated',
                                  'Shelf-terminating']:
            continue
        ref_gdirs.append(gdir)

    print('GTD Glac after', len(ref_gdirs))

    ref_rgiids = [gdir.rgi_id for gdir in ref_gdirs]
    gtd_df = gtd_df.set_index('RGI_ID').loc[ref_rgiids]

    # Account for area differences between glathida and rgi
    ref_area_km2 = np.asarray([gdir.rgi_area_km2 for gdir in ref_gdirs])
    gtd_df.VOLUME = gtd_df.MEAN_THICKNESS * gtd_df.GTD_AREA * 1e-3
    ref_cs = gtd_df.VOLUME.values / (gtd_df.GTD_AREA.values**1.375)
    ref_volume_km3 = ref_cs * ref_area_km2**1.375
    ref_thickness_m = ref_volume_km3 / ref_area_km2 * 1000.

    gtd_df['ref_area_km2'] = ref_area_km2
    gtd_df['ref_volume_km3'] = ref_volume_km3
    gtd_df['ref_thickness_m'] = ref_thickness_m
    gtd_df['ref_gdirs'] = ref_gdirs

    return gtd_df

def _prepare_marine(gdirs):

    # Get test glaciers (all glaciers with thickness data)
    fpath = utils.get_glathida_file()

    try:
        gtd_df = pd.read_csv(fpath).sort_values(by=['RGI_ID'])
    except AttributeError:
        gtd_df = pd.read_csv(fpath).sort(columns=['RGI_ID'])
    dfids = gtd_df['RGI_ID'].values

    print('GTD Glac before marine', len(dfids))
    ref_gdirs = []
    for gdir in gdirs:
        if gdir.rgi_id not in dfids:
            continue
        if gdir.glacier_type == 'Ice cap':
            continue
        if gdir.terminus_type in ['Marine-terminating', 'Lake-terminating',
                                  'Dry calving', 'Regenerated',
                                  'Shelf-terminating']:
            ref_gdirs.append(gdir)


    print('GTD Glac after marine', len(ref_gdirs))

    ref_rgiids = [gdir.rgi_id for gdir in ref_gdirs]
    gtd_df = gtd_df.set_index('RGI_ID').loc[ref_rgiids]

    # Account for area differences between glathida and rgi
    ref_area_km2 = np.asarray([gdir.rgi_area_km2 for gdir in ref_gdirs])
    gtd_df.VOLUME = gtd_df.MEAN_THICKNESS * gtd_df.GTD_AREA * 1e-3
    ref_cs = gtd_df.VOLUME.values / (gtd_df.GTD_AREA.values**1.375)
    ref_volume_km3 = ref_cs * ref_area_km2**1.375
    ref_thickness_m = ref_volume_km3 / ref_area_km2 * 1000.

    gtd_df['ref_area_km2'] = ref_area_km2
    gtd_df['ref_volume_km3'] = ref_volume_km3
    gtd_df['ref_thickness_m'] = ref_thickness_m
    gtd_df['ref_gdirs'] = ref_gdirs

    return gtd_df


def optimize_volume(gdirs):
    """Optimizesfd based on GlaThiDa thicknesses.

    We use the glacier averaged thicknesses provided by GlaThiDa and correct
    them for differences in area with RGI, using a glacier specific volume-area
    scaling formula.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    gtd_df = _prepare_inv(gdirs)
    ref_gdirs = gtd_df['ref_gdirs']
    ref_volume_km3 = gtd_df['ref_volume_km3']
    ref_area_km2 = gtd_df['ref_area_km2']
    ref_thickness_m = gtd_df['ref_thickness_m']

    # Optimize without sliding
    log.info('Compute the inversion parameter.')

    def to_optimize(x):
        tmp_vols = np.zeros(len(ref_gdirs))
        glen_a = cfg.A * x[0]
        for i, gdir in enumerate(ref_gdirs):
            v, _ = invert_parabolic_bed(gdir, glen_a=glen_a,
                                        fs=0., write=False)
            tmp_vols[i] = v * 1e-9
        return utils.rmsd(tmp_vols, ref_volume_km3)
    opti = optimization.minimize(to_optimize, [1.],
                                bounds=((0.01, 10), ),
                                tol=1.e-4)
    # Check results and save.
    glen_a = cfg.A * opti['x'][0]
    fs = 0.

    # This is for the stats
    oggm_volume_m3 = np.zeros(len(ref_gdirs))
    rgi_area_m2 = np.zeros(len(ref_gdirs))
    for i, gdir in enumerate(ref_gdirs):
        v, a = invert_parabolic_bed(gdir, glen_a=glen_a, fs=fs,
                                    write=False)
        oggm_volume_m3[i] = v
        rgi_area_m2[i] = a
    assert np.allclose(rgi_area_m2 * 1e-6, ref_area_km2)

    # This is for each glacier
    out = dict()
    out['glen_a'] = glen_a
    out['fs'] = fs
    out['factor_glen_a'] = opti['x'][0]
    try:
        out['factor_fs'] = opti['x'][1]
    except IndexError:
        out['factor_fs'] = 0.
    for gdir in gdirs:
        gdir.write_pickle(out, 'inversion_params')

    # This is for the working dir
    # Simple stats
    out['vol_rmsd'] = utils.rmsd(oggm_volume_m3 * 1e-9, ref_volume_km3)
    out['thick_rmsd'] = utils.rmsd(oggm_volume_m3 * 1e-9 / ref_area_km2 / 1000.,
                                 ref_thickness_m)
    log.info('Optimized glen_a and fs with a factor {factor_glen_a:.2f} and '
             '{factor_fs:.2f} for a volume RMSD of {vol_rmsd:.3f}'.format(**out))

    df = pd.DataFrame(out, index=[0])
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_params.csv')
    df.to_csv(fpath)

    # All results
    df = dict()
    df['ref_area_km2'] = ref_area_km2
    df['ref_volume_km3'] = ref_volume_km3
    df['ref_thickness_m'] = ref_thickness_m
    df['oggm_volume_km3'] = oggm_volume_m3 * 1e-9
    df['oggm_thickness_m'] = oggm_volume_m3 / (ref_area_km2 * 1e6)
    df['vas_volume_km3'] = 0.034*(df['ref_area_km2']**1.375)
    df['vas_thickness_m'] = df['vas_volume_km3'] / ref_area_km2 * 1000

    rgi_id = [gdir.rgi_id for gdir in ref_gdirs]
    df = pd.DataFrame(df, index=rgi_id)
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_results.csv')
    df.to_csv(fpath)

    # return value for tests
    return out


def optimize_thick(gdirs):
    """Optimizesfd based on GlaThiDa thicknesses.

    We use the glacier averaged thicknesses provided by GlaThiDa and correct
    them for differences in area with RGI, using a glacier specific volume-area
    scaling formula.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    gtd_df = _prepare_inv(gdirs)
    ref_gdirs = gtd_df['ref_gdirs']
    ref_volume_km3 = gtd_df['ref_volume_km3']
    ref_area_km2 = gtd_df['ref_area_km2']
    ref_thickness_m = gtd_df['ref_thickness_m']

    # Optimize without sliding
    log.info('Compute the inversion parameter.')

    def to_optimize(x):
        tmp_ = np.zeros(len(ref_gdirs))
        glen_a = cfg.A * x[0]
        for i, gdir in enumerate(ref_gdirs):
            v, a = invert_parabolic_bed(gdir, glen_a=glen_a,
                                        fs=0., write=False)
            tmp_[i] = v / a
        return utils.rmsd(tmp_, ref_thickness_m)
    opti = optimization.minimize(to_optimize, [1.],
                                bounds=((0.01, 10), ),
                                tol=1.e-4)
    # Check results and save.
    glen_a = cfg.A * opti['x'][0]
    fs = 0.

    # This is for the stats
    oggm_volume_m3 = np.zeros(len(ref_gdirs))
    rgi_area_m2 = np.zeros(len(ref_gdirs))
    for i, gdir in enumerate(ref_gdirs):
        v, a = invert_parabolic_bed(gdir, glen_a=glen_a, fs=fs,
                                    write=False)
        oggm_volume_m3[i] = v
        rgi_area_m2[i] = a
    assert np.allclose(rgi_area_m2 * 1e-6, ref_area_km2)

    # This is for each glacier
    out = dict()
    out['glen_a'] = glen_a
    out['fs'] = fs
    out['factor_glen_a'] = opti['x'][0]
    try:
        out['factor_fs'] = opti['x'][1]
    except IndexError:
        out['factor_fs'] = 0.
    for gdir in gdirs:
        gdir.write_pickle(out, 'inversion_params')

    # This is for the working dir
    # Simple stats
    out['vol_rmsd'] = utils.rmsd(oggm_volume_m3 * 1e-9, ref_volume_km3)
    out['thick_rmsd'] = utils.rmsd(oggm_volume_m3 / (ref_area_km2 * 1e6),
                                   ref_thickness_m)
    log.info('Optimized glen_a and fs with a factor {factor_glen_a:.2f} and '
             '{factor_fs:.2f} for a thick RMSD of {thick_rmsd:.3f}'.format(
        **out))

    df = pd.DataFrame(out, index=[0])
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_params.csv')
    df.to_csv(fpath)

    # All results
    df = dict()
    df['ref_area_km2'] = ref_area_km2
    df['ref_volume_km3'] = ref_volume_km3
    df['ref_thickness_m'] = ref_thickness_m
    df['oggm_volume_km3'] = oggm_volume_m3 * 1e-9
    df['oggm_thickness_m'] = oggm_volume_m3 / (ref_area_km2 * 1e6)
    df['vas_volume_km3'] = 0.034*(df['ref_area_km2']**1.375)
    df['vas_thickness_m'] = df['vas_volume_km3'] / ref_area_km2 * 1000

    rgi_id = [gdir.rgi_id for gdir in ref_gdirs]
    df = pd.DataFrame(df, index=rgi_id)
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_results.csv')
    df.to_csv(fpath)

    # return value for tests
    return out


def optimize_per_glacier(gdirs):

    gtd_df = _prepare_inv(gdirs)
    ref_gdirs = gtd_df['ref_gdirs']
    ref_volume_km3 = gtd_df['ref_volume_km3']
    ref_area_km2 = gtd_df['ref_area_km2']
    ref_thickness_m = gtd_df['ref_thickness_m']

    # Optimize without sliding
    log.info('Compute the inversion parameter.')

    fac = []
    for gdir in ref_gdirs:
        def to_optimize(x):
            glen_a = cfg.A * x[0]
            v, a = invert_parabolic_bed(gdir, glen_a=glen_a, fs=0.,
                                        write=False)
            return utils.rmsd(v / a, gtd_df['ref_thickness_m'].loc[gdir.rgi_id])
        opti = optimization.minimize(to_optimize, [1.],
                                    bounds=((0.01, 10), ),
                                    tol=0.1)
        # Check results and save.
        fac.append(opti['x'][0])

    # All results
    df = utils.glacier_characteristics(ref_gdirs)

    df['ref_area_km2'] = ref_area_km2
    df['ref_volume_km3'] = ref_volume_km3
    df['ref_thickness_m'] = ref_thickness_m
    df['vas_volume_km3'] = 0.034*(df['ref_area_km2']**1.375)
    df['vas_thickness_m'] = df['vas_volume_km3'] / ref_area_km2 * 1000
    df['fac'] = fac
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_pergla.csv')
    df.to_csv(fpath)


def invert_marine_terminating(gdirs):

    gtd_df = _prepare_inv(gdirs)
    ref_gdirs = gtd_df['ref_gdirs']
    ref_volume_km3 = gtd_df['ref_volume_km3']
    ref_area_km2 = gtd_df['ref_area_km2']
    ref_thickness_m = gtd_df['ref_thickness_m']

    # Optimize without sliding
    log.info('Compute the inversion parameter.')

    fac = []
    for gdir in ref_gdirs:
        def to_optimize(x):
            glen_a = cfg.A * x[0]
            v, a = invert_parabolic_bed(gdir, glen_a=glen_a, fs=0.,
                                        write=False)
            return utils.rmsd(v / a, gtd_df['ref_thickness_m'].loc[gdir.rgi_id])
        opti = optimization.minimize(to_optimize, [1.],
                                    bounds=((0.01, 10), ),
                                    tol=1.e-4)
        # Check results and save.
        fac.append(opti['x'][0])

    # All results
    df = utils.glacier_characteristics(ref_gdirs)

    df['ref_area_km2'] = ref_area_km2
    df['ref_volume_km3'] = ref_volume_km3
    df['ref_thickness_m'] = ref_thickness_m
    df['vas_volume_km3'] = 0.034*(df['ref_area_km2']**1.375)
    df['vas_thickness_m'] = df['vas_volume_km3'] / ref_area_km2 * 1000
    df['fac'] = fac
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_optim_pergla.csv')
    df.to_csv(fpath)


def invert_marine(gdirs, fac=3.02634401808867):

    gtd_df = _prepare_marine(gdirs)

    ref_gdirs = gtd_df['ref_gdirs']
    ref_volume_km3 = gtd_df['ref_volume_km3']
    ref_area_km2 = gtd_df['ref_area_km2']
    ref_thickness_m = gtd_df['ref_thickness_m']

    glen_a = cfg.A * fac

    oggm_volume_m3 = np.zeros(len(ref_gdirs))
    rgi_area_m2 = np.zeros(len(ref_gdirs))
    for i, gdir in enumerate(ref_gdirs):
        v, a = invert_parabolic_bed(gdir, glen_a=glen_a, fs=0,
                                    write=False)
        oggm_volume_m3[i] = v
        rgi_area_m2[i] = a
    assert np.allclose(rgi_area_m2 * 1e-6, ref_area_km2)

    df = utils.glacier_characteristics(ref_gdirs)

    df['ref_area_km2'] = ref_area_km2
    df['ref_volume_km3'] = ref_volume_km3
    df['ref_thickness_m'] = ref_thickness_m
    df['oggm_volume_km3'] = oggm_volume_m3 * 1e-9
    df['oggm_thickness_m'] = oggm_volume_m3 / (ref_area_km2 * 1e6)
    df['vas_volume_km3'] = 0.034*(df['ref_area_km2']**1.375)
    df['vas_thickness_m'] = df['vas_volume_km3'] / ref_area_km2 * 1000
    fpath = os.path.join(cfg.PATHS['working_dir'],
                         'inversion_marine.csv')
    df.to_csv(fpath)