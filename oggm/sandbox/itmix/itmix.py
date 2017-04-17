from __future__ import absolute_import, division

import glob
import logging
# Built ins
import os
from shutil import copyfile

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import rasterio
# External libs
import salem
import shapely.geometry as shpg
from osgeo import gdal
from salem.datasets import EsriITMIX
from scipy import optimize as optimization
from scipy.interpolate import griddata

import oggm.cfg as cfg
# Locals
from oggm import entity_task
from oggm import utils
from oggm.core.preprocessing.gis import gaussian_blur, _mask_per_divide
from oggm.core.preprocessing.inversion import mass_conservation_inversion
from oggm.sandbox.itmix.itmix_cfg import DATA_DIR, ITMIX_ODIR

# Module logger
log = logging.getLogger(__name__)

# Globals
SEARCHD = os.path.join(DATA_DIR, 'itmix', 'glaciers_sorted')
LABEL_STRUCT = np.ones((3, 3))


def find_path(start_dir, pattern, allow_more=False):
    """Find a file in a dir and subdir"""

    files = []
    for dir, _, _ in os.walk(start_dir):
        files.extend(glob.glob(os.path.join(dir,pattern)))

    if allow_more:
        assert len(files) > 0
        return files
    else:
        assert len(files) == 1
        return files[0]


def get_rgi_df(reset=False):
    """This function prepares a kind of `fake` RGI file, with the updated
    geometries for ITMIX.
    """

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

        divides = []
        rgidf = []
        _rgi_ids_for_overwrite = []
        for i, row in df_itmix.iterrows():

            log.info('Prepare RGI df for ' + row.name)

            # read the rgi region
            rgi_shp = find_path(RGI_DIR, row['rgi_reg'] + '_rgi50_*.shp')
            rgi_df = salem.read_shapefile(rgi_shp, cached=True)

            rgi_parts = row.T['rgi_parts_ids']
            sel = rgi_df.loc[rgi_df.RGIId.isin(rgi_parts)].copy()

            # use the ITMIX shape where possible
            if row.name in ['Hellstugubreen', 'Freya', 'Aqqutikitsoq',
                            'Brewster', 'Kesselwandferner', 'NorthGlacier',
                            'SouthGlacier', 'Tasman', 'Unteraar',
                            'Washmawapta', 'Columbia']:
                shf = find_path(SEARCHD, '*_' + row.name + '*.shp')
                shp = salem.read_shapefile(shf)
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
                shf = find_path(SEARCHD, '*_' + row.name + '*.shp')
                shp2 = salem.read_shapefile(shf)
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
            elif len(rgi_parts) > 1:
                # Ice-caps. Make divides
                # First we gather all the parts:
                sel = rgi_df.loc[rgi_df.RGIId.isin(rgi_parts)].copy()
                # Make the multipolygon for the record
                multi = shpg.MultiPolygon([g for g in sel.geometry])
                # update the RGI attributes. We take a dummy rgi ID
                new_area = np.sum(sel.Area)
                found = False
                for i in range(len(sel)):
                    tsel = sel.iloc[[i]].copy()
                    if 'Multi' in tsel.loc[tsel.index[0], 'geometry'].type:
                        continue
                    else:
                        found = True
                        sel = tsel
                        break
                if not found:
                    raise RuntimeError()

                inif = 0.
                add = 1e-5
                if row.name == 'Devon':
                    inif = 0.001
                    add = 1e-4
                while True:
                    buff = multi.buffer(inif)
                    if 'Multi' in buff.type:
                        inif += add
                    else:
                        break
                x, y = multi.centroid.xy
                if 'Multi' in buff.type:
                    raise RuntimeError
                sel.loc[sel.index[0], 'geometry'] = buff
                sel.loc[sel.index[0], 'Area'] = new_area
                sel.loc[sel.index[0], 'CenLon'] = np.asarray(x)[0]
                sel.loc[sel.index[0], 'CenLat'] = np.asarray(y)[0]

                # Divides db
                div_sel = dict()
                for k, v in sel.iloc[0].iteritems():
                    if k == 'geometry':
                        div_sel[k] = multi
                    elif k == 'RGIId':
                        div_sel['RGIID'] = v
                    else:
                        div_sel[k] = v
                divides.append(div_sel)
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

            # Add divides to the original one
            adf = pd.DataFrame(divides)
            adf.to_pickle(cfg.PATHS['itmix_divs'])

        log.info('N glaciers ITMIX: {}'.format(len(rgidf)))

        # WGMS glaciers which are not already there
        # Actually we should remove the data of those 7 to be honest...
        f, d = utils.get_wgms_files()
        wgms_df = pd.read_csv(f)
        wgms_df = wgms_df.loc[~ wgms_df.RGI_ID.isin(_rgi_ids_for_overwrite)]

        log.info('N glaciers WGMS: {}'.format(len(wgms_df)))
        for i, row in wgms_df.iterrows():
            rid = row.RGI_ID
            reg = rid.split('-')[1].split('.')[0]
            # read the rgi region
            rgi_shp = find_path(RGI_DIR, reg + '_rgi50_*.shp')
            rgi_df = salem.read_shapefile(rgi_shp, cached=True)

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

        _rgi_ids_for_overwrite.extend(wgms_df.RGI_ID.values)

        # GTD glaciers which are not already there
        # Actually we should remove the data of those 2 to be honest...
        gtd_df = gtd_df.loc[~ gtd_df.RGI_ID.isin(_rgi_ids_for_overwrite)]
        log.info('N glaciers GTD: {}'.format(len(gtd_df)))

        for i, row in gtd_df.iterrows():
            rid = row.RGI_ID
            reg = rid.split('-')[1].split('.')[0]
            # read the rgi region
            rgi_shp = find_path(RGI_DIR, reg + '_rgi50_*.shp')
            rgi_df = salem.read_shapefile(rgi_shp, cached=True)

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
        if len(pok[0]) > 0:
            points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
            inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
            dem[pnan] = griddata(points, np.ravel(dem[pok]), inter)
            msg = gdir.rgi_id + ': DEM needed interpolation'
            msg += '({:.1f}% missing).'.format(len(pnan[0])/len(dem.flatten())*100)
            log.warning(msg)
        else:
            dem = dem*np.NaN

    # Replace DEM values with ITMIX ones where possible
    # Open DEM
    dem_f = None
    n_g = gdir.name.split(':')[-1]
    searchf = os.path.join(DATA_DIR, 'itmix', 'glaciers_sorted', '*')
    searchf = os.path.join(searchf, '02_surface_' + n_g + '_*.asc')
    for dem_f in glob.glob(searchf):
        pass

    if dem_f is None:
        # try synth
        n_g = gdir.rgi_id
        searchf = os.path.join(DATA_DIR, 'itmix', 'glaciers_synth', '*')
        searchf = os.path.join(searchf, '02_surface_' + n_g + '*.asc')
        for dem_f in glob.glob(searchf):
            pass

    if dem_f is not None:
        log.info('%s: ITMIX DEM file: %s', gdir.rgi_id, dem_f)
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
        if  n_g in ['Synthetic2', 'Synthetic1']:
            dem = np.where(~ it_dem.mask, it_dem, np.nanmin(it_dem))
        else:
            dem = np.where(~ it_dem.mask, it_dem, dem)
    else:
        if 'Devon' in n_g:
            raise RuntimeError('Should have found DEM for Devon')
    
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


def optimize_thick(gdirs):
    """Optimizes fd based on GlaThiDa thicknesses.

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
            v, a = mass_conservation_inversion(gdir, glen_a=glen_a,
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
        v, a = mass_conservation_inversion(gdir, glen_a=glen_a, fs=fs,
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
    df = utils.glacier_characteristics(ref_gdirs)
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


def synth_apparent_mb(gdir, tstar=None, bias=None):
    """Compute local mustar and apparent mb from tstar.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    tstar: int
        the year where the glacier should be equilibrium
    bias: int
        the associated reference bias
    """

    # Ok. Looping over divides
    for div_id in list(gdir.divide_ids):
        log.info('%s: apparent mb synth')

        # For each flowline compute the apparent MB
        fls = gdir.read_pickle('inversion_flowlines', div_id=div_id)

        # Reset flux
        for fl in fls:
            fl.flux = np.zeros(len(fl.surface_h))


        n_g = gdir.rgi_id

        searchf = os.path.join(DATA_DIR, 'itmix', 'glaciers_synth', '*')
        searchf = os.path.join(searchf, '04_mb_' + n_g + '*.asc')
        for dem_f in glob.glob(searchf):
            pass
        ds_mb = salem.EsriITMIX(dem_f)
        mb = ds_mb.get_vardata() * 1000.
        mb = np.where(mb < -9998, np.NaN, mb)

        f = os.path.join(DATA_DIR, 'itmix', 'glaciers_synth',
                         '01_margin_'+ n_g +'_0000_UTM00.shp')
        ds_mb.set_roi(f)
        mb = np.where(ds_mb.roi, mb, np.NaN)

        searchf = os.path.join(DATA_DIR, 'itmix', 'glaciers_synth', '*')
        searchf = os.path.join(searchf, '02_*_' + n_g + '*.asc')
        for dem_f in glob.glob(searchf):
            pass
        ds_dem = salem.EsriITMIX(dem_f)
        dem = ds_dem.get_vardata()


        from scipy import stats
        pok = np.where(np.isfinite(mb) & (mb > 0))
        slope_p, _, _, _, _ = stats.linregress(dem[pok], mb[pok])
        pok = np.where(np.isfinite(mb) & (mb < 0))
        slope_m, _, _, _, _ = stats.linregress(dem[pok], mb[pok])

        def ela_mb_grad(ela_h, h):
            return np.where(h < ela_h, slope_m * (h - ela_h),
                            slope_p * (h - ela_h))

        # Get all my hs
        hs = []
        ws = []
        for fl in fls:
            hs = np.append(hs, fl.surface_h)
            ws = np.append(ws, fl.widths)

        # find ela for zero mb
        def to_optim(x):
            tot_mb = np.average(ela_mb_grad(x[0], hs), weights=ws)
            return tot_mb**2

        opti = optimization.minimize(to_optim, [1000.],
                                     bounds=((0., 10000),),
                                     tol=1e-6)

        # Check results and save.
        final_elah = opti['x'][0]
        # print(final_elah)
        # pok = np.where(np.isfinite(mb))
        # plt.plot(dem[pok], mb[pok], 'o')
        # plt.plot(hs, ela_mb_grad(final_elah, hs), 'o')
        # plt.show()

        # Flowlines in order to be sure
        # TODO: here it would be possible to test for a minimum mb gradient
        # and change prcp factor if judged useful
        for fl in fls:
            mb_on_h = ela_mb_grad(final_elah, fl.surface_h)
            fl.set_apparent_mb(mb_on_h)

        # Check
        if div_id >= 1:
            if not np.allclose(fls[-1].flux[-1], 0., atol=0.01):
                log.warning('%s: flux should be zero, but is: %.2f',
                            gdir.rgi_id,
                            fls[-1].flux[-1])

        # Overwrite
        gdir.write_pickle(fls, 'inversion_flowlines', div_id=div_id)


def write_itmix_ascii(gdir, version):
    """Write the results"""

    gname = gdir.name.replace('I:', '')
    real = gname
    gname = gname.replace('_A', '')
    gname = gname.replace('_B', '')

    log.info('Write ITMIX ' + real)

    # Get the data
    grids_file = gdir.get_filepath('gridded_data', div_id=0)
    with netCDF4.Dataset(grids_file) as nc:
        thick = nc.variables['thickness'][:]
    vol = np.nansum(thick * gdir.grid.dx**2)

    # Transform to output grid
    try:
        ifile = find_path(os.path.join(DATA_DIR, 'itmix', 'glaciers_sorted'),
                          '02_surface_' + gname + '*.asc')
    except AssertionError:
        gname = gdir.rgi_id
        searchf = os.path.join(DATA_DIR, 'itmix', 'glaciers_synth')
        ifile = find_path(searchf,  '02_surface_' + gname + '*.asc')
    itmix = salem.EsriITMIX(ifile)

    thick = itmix.grid.map_gridded_data(thick, gdir.grid, interp='linear')

    # Mask out
    itmix.set_roi(shape=gdir.get_filepath('outlines'))
    omask = itmix.roi
    thick[np.nonzero(omask==0)] = np.nan

    # Output path
    bname = os.path.basename(ifile).split('.')[0]
    pok = bname.find('UTM')
    zone = bname[pok:]
    ofile = os.path.join(ITMIX_ODIR, gname)
    if not os.path.exists(ofile):
        os.mkdir(ofile)
    fname = 'Maussion_' + real + '_bedrock_v{}_'.format(version) + zone +'.asc'
    ofile = os.path.join(ofile, fname)

    # Write out
    with rasterio.drivers():
        with rasterio.open(ifile) as src:
            topo = src.read(1).astype(np.float)
            topo = np.where(topo < -999., np.NaN, topo)

            # Also for ours
            thick = np.where(topo < -999., np.NaN, thick)

            # Be sure volume is conserved
            thick *= vol / np.nansum(thick*itmix.grid.dx**2)
            assert np.isclose(np.nansum(thick*itmix.grid.dx**2), vol)

            # convert
            topo -= thick
            with rasterio.open(ofile, 'w',
                               driver=src.driver,
                               width=src.width,
                               height=src.height,
                               transform=src.transform,
                               count=1,
                               dtype=np.float,
                               nodata=np.NaN) as dst:
                dst.write_band(1, topo)

    # Check
    with rasterio.open(ifile) as src:
        topo = src.read(1).astype(np.float)
        topo = np.where(topo < -9999., np.NaN, topo)

    with rasterio.open(ofile) as src:
        mtopo = src.read(1).astype(np.float)
        mtopo = np.where(mtopo < -9999., np.NaN, mtopo)

    if not np.allclose(np.nanmax(topo - mtopo), np.nanmax(thick), atol=5):
        print(np.nanmax(topo - mtopo), np.nanmax(thick))
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        ax1.imshow(topo - mtopo)
        ax2.imshow(thick)
        ax3.imshow(topo - mtopo - thick)
        ax4.imshow(~np.isclose(topo - mtopo, thick, atol=1, equal_nan=True))
        plt.show()