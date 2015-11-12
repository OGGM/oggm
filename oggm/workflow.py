"""Wrappers for the single tasks, multi processor handling."""
from __future__ import division
from six.moves import zip

# Built ins
import logging
import os
# External libs
import pandas as pd
import geopandas as gpd
import multiprocessing as mp

# Locals
import oggm.conf as cfg
from oggm.prepro import gis
from oggm.prepro import centerlines
from oggm.prepro import geometry
from oggm.prepro import inversion
from oggm.prepro import climate
from oggm import utils

# Module logger
log = logging.getLogger(__name__)

# Multiprocessing pool
mppool = None

def _init_pool():
    """Necessary because at import time, cfg might be unitialized"""

    global  mppool
    if cfg.use_mp:
        mppool = mp.Pool(cfg.nproc)


def execute_task(task, gdirs):
    """Execute any taks on gdirs. If you asked for multiprocessing,
    it will do it."""

    if cfg.use_mp:
        if mppool is None:
            _init_pool()
        poolargs = gdirs
        mppool.map(task, poolargs, chunksize=1)
    else:
        for gdir in gdirs:
            task(gdir)


def init_glacier_regions(rgidf, reset=False, force=False):
    """Very first task to do (allways). Set reset=False
    in order not to delete the content"""

    if reset and not force:
        reset = utils.query_yes_no('Delete all glacier directories?')

    gdirs = []
    for _, entity in rgidf.iterrows():
        gdir = cfg.GlacierDir(entity, reset=reset)
        if not os.path.exists(gdir.get_filepath('dem')):
            gis.define_glacier_region(gdir, entity)
        gdirs.append(gdir)

    return gdirs


def write_centerlines_to_shape(gdirs, filename):
    """Write centerlines in a shapefile"""

    olist = []
    for gdir in gdirs:
        olist.extend(gis.get_centerline_lonlat(gdir))

    odf = gpd.GeoDataFrame(olist)

    from collections import OrderedDict
    shema = dict()
    props = OrderedDict()
    props['RGIID'] = 'str:14'
    props['DIVIDE'] = 'int:9'
    props['LE_SEGMENT'] = 'int:9'
    props['MAIN'] = 'int:9'
    shema['geometry'] = 'LineString'
    shema['properties'] = props

    crs = {'init': 'epsg:4326'}

    #some writing function from geopandas rep
    from six import iteritems
    from shapely.geometry import mapping
    import fiona
    def feature(i, row):
        return {
            'id': str(i),
            'type': 'Feature',
            'properties':
                dict((k, v) for k, v in iteritems(row) if k != 'geometry'),
            'geometry': mapping(row['geometry'])}
    with fiona.open(filename, 'w', driver='ESRI Shapefile',
                    crs=crs, schema=shema) as c:
        for i, row in odf.iterrows():
            c.write(feature(i, row))

def gis_prepro_tasks(gdirs):
    """Prepare the flowlines"""
    tasks = [
             gis.glacier_masks,
             centerlines.compute_centerlines,
             centerlines.compute_downstream_lines,
             geometry.catchment_area,
             geometry.initialize_flowlines,
             geometry.catchment_width_geom,
             geometry.catchment_width_correction
             ]
    for task in tasks:
        execute_task(task, gdirs)

def climate_tasks(gdirs):
    """Prepare the climate data"""

    climate.distribute_climate_data(gdirs)

    # Get ref glaciers (all glaciers with MB)
    dfids = cfg.paths['wgms_rgi_links']
    dfids = pd.read_csv(dfids)['RGI_ID'].values

    ref_gdirs = [g for g in gdirs if g.rgi_id in dfids]
    execute_task(climate.mu_candidates, ref_gdirs)
    climate.compute_ref_t_stars(ref_gdirs)
    climate.distribute_t_stars(gdirs)