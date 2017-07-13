"""This script reads all the RGI files and computes intersects out of them."""
import multiprocessing as mp
import os
from glob import glob

from oggm import cfg
from oggm.utils import get_rgi_dir

# Download RGI files
cfg.initialize()
rgi_dir = get_rgi_dir()
rgi_shps = list(glob(os.path.join(rgi_dir, "*", '*_rgi50_*.shp')))
rgi_shps = [r for r in rgi_shps if 'Regions' not in r]

from oggm.sandbox.rgi_tools.tools import compute_intersects

with mp.Pool() as p:
    p.map(compute_intersects, rgi_shps, chunksize=1)
