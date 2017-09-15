"""This script reads all the RGI files and computes intersects out of them."""
import multiprocessing as mp
import os
import oggm
from glob import glob

from oggm import cfg
from oggm.utils import get_rgi_dir
from oggm.sandbox.rgi_tools.tools import compute_intersects, prepare_divides


def intersects_script():
    # Download RGI files
    cfg.initialize()
    rgi_dir = get_rgi_dir()
    rgi_shps = list(glob(os.path.join(rgi_dir, "*", '*_rgi50_*.shp')))
    rgi_shps = [r for r in rgi_shps if 'Regions' not in r]
    with mp.Pool() as p:
        p.map(compute_intersects, rgi_shps, chunksize=1)


INDIR_INTERSECTS = '/home/mowglie/disk/Data/OGGM_DATA/RGI_V5_Modified/RGIV5_Corrected'


def intersects_ondivides_script():
    rgi_shps = list(glob(os.path.join(INDIR_INTERSECTS, "*", '*_rgi50_*.shp')))
    rgi_shps = sorted([r for r in rgi_shps if 'Regions' not in r])

    for s in rgi_shps:
        compute_intersects(s)

    # with mp.Pool() as p:
    #     p.map(compute_intersects, rgi_shps, chunksize=1)


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
    intersects_ondivides_script()
