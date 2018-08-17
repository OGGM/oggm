"""This script reads all the RGI files and computes intersects out of them."""
# flake8: noqa
import oggm
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob

from oggm import cfg
from oggm.utils import get_rgi_dir, mkdir
from oggm.sandbox.rgi_tools.tools import (compute_intersects, prepare_divides,
                                          OUTDIR_INTERSECTS)

rgi_version = '61'


def intersects_script():
    # Download RGI files
    cfg.initialize()
    rgi_dir = get_rgi_dir(version=rgi_version)
    fp = '*_rgi' + rgi_version + '_*.shp'
    rgi_shps = list(glob(os.path.join(rgi_dir, "*", fp)))
    rgi_shps = [r for r in rgi_shps if 'Regions' not in r]
    with mp.Pool() as p:
        p.map(compute_intersects, rgi_shps, chunksize=1)


def merge_intersects():
    # Download RGI files
    fp = 'intersects_*_rgi' + rgi_version + '_*.shp'
    shps = list(glob(os.path.join(OUTDIR_INTERSECTS, "*", fp)))
    assert len(shps) == 19
    out = []
    for sh in sorted(shps):
        sh = gpd.read_file(sh)
        out.append(sh)
    # Make a new dataframe of those
    inter = pd.concat(out)
    inter.csr = sh.crs  # for geolocalisation

    odir = '00_rgi' + rgi_version + '0_AllRegs'
    odir = os.path.join(OUTDIR_INTERSECTS, odir)
    mkdir(odir)
    ofile = os.path.join(odir, 'intersects_rgi' +
                         rgi_version + '0_AllRegs.shp')
    inter.to_file(ofile)


def intersects_ondivides_script():
    # Download RGI files
    cfg.initialize()
    rgi_dir = ''  # get_rgi_corrected_dir()
    rgi_shps = list(glob(os.path.join(rgi_dir, "*", '*_rgi50_*.shp')))
    rgi_shps = sorted([r for r in rgi_shps if 'Regions' not in r])
    rgi_shps = np.array(rgi_shps)[[3, 16]]
    with mp.Pool() as p:
        p.map(compute_intersects, rgi_shps, chunksize=1)


def divides_script():
    # Download RGI files
    cfg.initialize()
    rgi_dir = get_rgi_dir()
    rgi_shps = list(glob(os.path.join(rgi_dir, "*", '*_rgi50_*.shp')))
    rgi_shps = sorted([r for r in rgi_shps if 'Regions' not in r])

    for s in rgi_shps:
        prepare_divides(s)

    # with mp.Pool() as p:
    #     p.map(prepare_divides, rgi_shps, chunksize=1)


if __name__ == "__main__":
    merge_intersects()
