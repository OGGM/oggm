"""This script reads all the RGI files and computes intersects out of them."""
import os
import geopandas as gpd
from shapely.ops import linemerge
import shapely.geometry as shpg
from oggm.utils import haversine, mkdir
from oggm.core.preprocessing.gis import _check_geometry
from salem import wgs84
import time

OUTDIR = '/home/mowglie/disk/Data/GIS/SHAPES/RGI/RGI_V5_Intersects'


def compute_intersects(rgi_shp):
    """Processes the rgi file and writes the interscects to out_path"""

    out_path = os.path.basename(rgi_shp)
    odir = os.path.basename(os.path.dirname(rgi_shp))
    odir = os.path.join(OUTDIR, odir)
    mkdir(odir)
    out_path = os.path.join(odir, 'intersects_' + out_path)

    print('Start ' + os.path.basename(rgi_shp) + ' ...')
    start_time = time.time()

    gdf = gpd.read_file(rgi_shp)

    # clean geometries like OGGM does
    gdf['geometry'] = [_check_geometry(g) for g in gdf.geometry]

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
        gdfs = gdfs.loc[gdfs.intersects(major_poly)]

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
