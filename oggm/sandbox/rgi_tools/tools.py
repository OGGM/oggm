"""This script reads all the RGI files and computes intersects out of them."""
# flake8: noqa
import os
import time
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as shpg
from salem import wgs84
from shapely.ops import linemerge

from oggm.core.gis import multi_to_poly
from oggm.utils import haversine, mkdir, get_wgms_files

INDIR_DIVIDES = '/home/mowglie/disk/Data/OGGM_DATA/results_global_partitioning/altitude_filter/'

OUTDIR_INTERSECTS = '/home/mowglie/tmp/RGI_V61_Intersects/'
OUTDIR_DIVIDES = '/home/mowglie/disk/Data/OGGM_DATA/RGI_V5_Modified/'


def compute_intersects(rgi_shp):
    """Processes the rgi file and writes the intersects to OUTDIR"""

    out_path = os.path.basename(rgi_shp)
    odir = os.path.basename(os.path.dirname(rgi_shp))
    odir = os.path.join(OUTDIR_INTERSECTS, odir)
    mkdir(odir)
    out_path = os.path.join(odir, 'intersects_' + out_path)

    print('Start ' + os.path.basename(rgi_shp) + ' ...')
    start_time = time.time()

    gdf = gpd.read_file(rgi_shp)

    # clean geometries like OGGM does
    ngeos = []
    keep = []
    for g in gdf.geometry:
        try:
            g = multi_to_poly(g)
            ngeos.append(g)
            keep.append(True)
        except:
            keep.append(False)
    gdf = gdf.loc[keep]
    gdf['geometry'] = ngeos

    out_cols = ['RGIId_1', 'RGIId_2', 'geometry']
    out = gpd.GeoDataFrame(columns=out_cols)

    for i, major in gdf.iterrows():

        # Exterior only
        major_poly = major.geometry.exterior

        # sort by distance to the current glacier
        gdf['dis'] = haversine(major.CenLon, major.CenLat,
                               gdf.CenLon, gdf.CenLat)
        gdfs = gdf.sort_values(by='dis').iloc[1:]

        # Keep glaciers in which intersect
        gdfs = gdfs.loc[gdfs.dis < 200000]
        try:
            gdfs = gdfs.loc[gdfs.intersects(major_poly)]
        except:
            gdfs = gdfs.loc[gdfs.intersects(major_poly.buffer(0))]

        for i, neighbor in gdfs.iterrows():

            if neighbor.RGIId in out.RGIId_1 or neighbor.RGIId in out.RGIId_2:
                continue

            # Exterior only
            # Buffer is needed for numerical reasons
            neighbor_poly = neighbor.geometry.exterior.buffer(0.0001)

            # Go
            try:
                mult_intersect = major_poly.intersection(neighbor_poly)
            except:
                continue

            if isinstance(mult_intersect, shpg.Point):
                continue
            if isinstance(mult_intersect, shpg.linestring.LineString):
                mult_intersect = [mult_intersect]
            if len(mult_intersect) == 0:
                continue
            mult_intersect = [m for m in mult_intersect if
                              not isinstance(m, shpg.Point)]
            if len(mult_intersect) == 0:
                continue
            mult_intersect = linemerge(mult_intersect)
            if isinstance(mult_intersect, shpg.linestring.LineString):
                mult_intersect = [mult_intersect]
            for line in mult_intersect:
                assert isinstance(line, shpg.linestring.LineString)
                line = gpd.GeoDataFrame([[major.RGIId, neighbor.RGIId, line]],
                                        columns=out_cols)
                out = out.append(line)

    out.crs = wgs84.srs
    out.to_file(out_path)

    print(os.path.basename(rgi_shp) +
          ' took {0:.2f} seconds'.format(time.time() - start_time))
    return


def prepare_divides(rgi_f):
    """Processes the rgi file and writes the intersects to OUTDIR"""

    rgi_reg = os.path.basename(rgi_f).split('_')[0]

    print('Start RGI reg ' + rgi_reg + ' ...')
    start_time = time.time()

    wgms, _ = get_wgms_files()
    f = glob(INDIR_DIVIDES + '*/*-' + rgi_reg + '.shp')[0]

    df = gpd.read_file(f)
    rdf = gpd.read_file(rgi_f)

    # Read glacier attrs
    key2 = {'0': 'Land-terminating',
            '1': 'Marine-terminating',
            '2': 'Lake-terminating',
            '3': 'Dry calving',
            '4': 'Regenerated',
            '5': 'Shelf-terminating',
            '9': 'Not assigned',
            }
    TerminusType = [key2[gtype[1]] for gtype in df.GlacType]
    IsTidewater = np.array([ttype in ['Marine-terminating', 'Lake-terminating']
                            for ttype in TerminusType])

    # Plots
    # dfref = df.loc[df.RGIId.isin(wgms.RGI50_ID)]
    # for gid in np.unique(dfref.GLIMSId):
    #     dfs = dfref.loc[dfref.GLIMSId == gid]
    #     dfs.plot(cmap='Set3', linestyle='-', linewidth=5);

    # Filter
    df = df.loc[~IsTidewater]
    df = df.loc[~df.RGIId.isin(wgms.RGI50_ID)]

    df['CenLon'] = pd.to_numeric(df['CenLon'])
    df['CenLat'] = pd.to_numeric(df['CenLat'])
    df['Area'] = pd.to_numeric(df['Area'])

    # Correct areas and stuffs
    n_gl_before = len(df)
    divided_ids = []
    for rid in np.unique(df.RGIId):
        sdf = df.loc[df.RGIId == rid].copy()
        srdf = rdf.loc[rdf.RGIId == rid]

        # Correct Area
        sdf.Area = np.array([float(a) for a in sdf.Area])

        geo_is_ok = []
        new_geo = []
        for g, a in zip(sdf.geometry, sdf.Area):
            if a < 0.01*1e6:
                geo_is_ok.append(False)
                continue
            try:
                new_geo.append(multi_to_poly(g))
                geo_is_ok.append(True)
            except:
                geo_is_ok.append(False)

        sdf = sdf.loc[geo_is_ok]
        if len(sdf) < 2:
            # print(rid + ' is too small or has no valid divide...')
            df = df[df.RGIId != rid]
            continue

        area_km = sdf.Area * 1e-6

        cor_factor = srdf.Area.values / np.sum(area_km)
        if cor_factor > 1.2 or cor_factor < 0.8:
            # print(rid + ' is not OK...')
            df = df[df.RGIId != rid]
            continue
        area_km = cor_factor * area_km

        # Correct Centroid
        cenlon = [g.centroid.xy[0][0] for g in sdf.geometry]
        cenlat = [g.centroid.xy[1][0] for g in sdf.geometry]

        # ID
        new_id = [rid + '_d{:02}'.format(i + 1) for i in range(len(sdf))]

        # Write
        df.loc[sdf.index, 'Area'] = area_km
        df.loc[sdf.index, 'CenLon'] = cenlon
        df.loc[sdf.index, 'CenLat'] = cenlat
        df.loc[sdf.index, 'RGIId'] = new_id
        df.loc[sdf.index, 'geometry'] = new_geo

        divided_ids.append(rid)

    n_gl_after = len(df)

    # We make three data dirs: divides only, divides into rgi, divides + RGI
    bn = os.path.basename(rgi_f)
    bd = os.path.basename(os.path.dirname(rgi_f))
    base_dir_1 = OUTDIR_DIVIDES + '/RGIV5_DividesOnly/' + bd
    base_dir_2 = OUTDIR_DIVIDES + '/RGIV5_Corrected/' + bd
    base_dir_3 = OUTDIR_DIVIDES + '/RGIV5_OrigAndDivides/' + bd
    mkdir(base_dir_1, reset=True)
    mkdir(base_dir_2, reset=True)
    mkdir(base_dir_3, reset=True)

    df.to_file(os.path.join(base_dir_1, bn))

    dfa = pd.concat([df, rdf]).sort_values('RGIId')
    dfa.to_file(os.path.join(base_dir_3, bn))

    dfa = dfa.loc[~dfa.RGIId.isin(divided_ids)]
    dfa.to_file(os.path.join(base_dir_2, bn))

    print('RGI reg ' + rgi_reg +
          ' took {:.2f} seconds. We had to remove '
          '{} divides'.format(time.time() - start_time,
                              n_gl_before - n_gl_after))
    return
