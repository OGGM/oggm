"""Some useful functions that did not fit into the other modules.
"""
from __future__ import absolute_import, division

import six.moves.cPickle as pickle
from six import string_types
from six.moves.urllib.request import urlretrieve, urlopen
from six.moves.urllib.error import HTTPError, URLError, ContentTooShortError
from six.moves.urllib.parse import urlparse

# Builtins
import glob
import os
import gzip
import shutil
import zipfile
import sys
import math
import datetime
import logging
from collections import OrderedDict
from functools import partial, wraps
import json
import time
import fnmatch
import platform
import struct
import importlib

# External libs
import geopandas as gpd
import pandas as pd
import salem
from salem import lazy_property, read_shapefile
import numpy as np
import netCDF4
from scipy import stats
from joblib import Memory
from shapely.ops import transform as shp_trafo
from salem import wgs84
import xarray as xr
import rasterio
from scipy.ndimage import filters
from scipy.signal import gaussian
import shapely.geometry as shpg
from shapely.ops import linemerge
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool
import multiprocessing as mp

# Locals
import oggm.cfg as cfg
from oggm.cfg import CUMSEC_IN_MONTHS, SEC_IN_YEAR, BEGINSEC_IN_MONTHS

# Module logger
logger = logging.getLogger(__name__)

SAMPLE_DATA_GH_REPO = 'OGGM/oggm-sample-data'
CRU_SERVER = 'https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_3.24.01/cruts' \
             '.1701201703.v3.24.01/'

DEM3REG = {
    'ISL': [-25., -12., 63., 67.],  # Iceland
    'SVALBARD': [10., 34., 76., 81.],
    'JANMAYEN': [-10., -7., 70., 72.],
    'FJ': [36., 66., 79., 82.],  # Franz Josef Land
    'FAR': [-8., -6., 61., 63.],  # Faroer
    'BEAR': [18., 20., 74., 75.],  # Bear Island
    'SHL': [-3., 0., 60., 61.],  # Shetland
    # Antarctica tiles as UTM zones, large files
    # '01-15': [-180., -91., -90, -60.],
    # '16-30': [-91., -1., -90., -60.],
    # '31-45': [-1., 89., -90., -60.],
    # '46-60': [89., 189., -90., -60.],
    # Greenland tiles
    # 'GL-North': [-78., -11., 75., 84.],
    # 'GL-West': [-68., -42., 64., 76.],
    # 'GL-South': [-52., -40., 59., 64.],
    # 'GL-East': [-42., -17., 64., 76.]
}

# Joblib
MEMORY = Memory(cachedir=cfg.CACHE_DIR, verbose=0)

# Function
tuple2int = partial(np.array, dtype=np.int64)

# Global Lock
lock = mp.Lock()


def _get_download_lock():
    return lock


class NoInternetException(Exception):
    pass


def _call_dl_func(dl_func, cache_path):
    """Helper so the actual call to downloads cann be overridden
    """
    return dl_func(cache_path)


def _cached_download_helper(cache_obj_name, dl_func, reset=False):
    """Helper function for downloads.

    Takes care of checking if the file is already cached.
    Only calls the actuall download function when no cached version exists.
    """
    cache_dir = cfg.PATHS['dl_cache_dir']
    cache_ro = cfg.PARAMS['dl_cache_readonly']
    fb_cache_dir = os.path.join(cfg.PATHS['working_dir'], 'cache')

    if not cache_dir:
        cache_dir = fb_cache_dir
        cache_ro = False

    fb_path = os.path.join(fb_cache_dir, cache_obj_name)
    if not reset and os.path.isfile(fb_path):
        return fb_path

    cache_path = os.path.join(cache_dir, cache_obj_name)
    if not reset and os.path.isfile(cache_path):
        return cache_path

    if cache_ro:
        cache_path = fb_path

    if not cfg.PARAMS['has_internet']:
        raise NoInternetException("Download required, but has_internet is False.")

    mkdir(os.path.dirname(cache_path))

    try:
        cache_path = _call_dl_func(dl_func, cache_path)
    except:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        raise

    return cache_path


def _urlretrieve(url, cache_obj_name=None, reset=False, *args, **kwargs):
    """Wrapper around urlretrieve, to implement our caching logic.

    Instead of accepting a destination path, it decided where to store the file
    and returns the local path.
    """

    if cache_obj_name is None:
        cache_obj_name = urlparse(url)
        cache_obj_name = cache_obj_name.netloc + cache_obj_name.path

    def _dlf(cache_path):
        logger.info("Downloading %s to %s..." % (url, cache_path))
        urlretrieve(url, cache_path, *args, **kwargs)
        return cache_path

    return _cached_download_helper(cache_obj_name, _dlf, reset)


def _progress_urlretrieve(url, cache_name=None, reset=False):
    """Downloads a file, returns its local path, and shows a progressbar."""

    try:
        from progressbar import DataTransferBar, UnknownLength
        pbar = [None]
        def _upd(count, size, total):
            if pbar[0] is None:
                pbar[0] = DataTransferBar()
            if pbar[0].max_value is None:
                if total > 0:
                    pbar[0].start(total)
                else:
                    pbar[0].start(UnknownLength)
            pbar[0].update(min(count * size, total))
            sys.stdout.flush()
        res = _urlretrieve(url, cache_obj_name=cache_name, reset=reset,
                           reporthook=_upd)
        try:
            pbar[0].finish()
        except:
            pass
        return res
    except ImportError:
        return _urlretrieve(url, cache_obj_name=cache_name, reset=reset)


def aws_file_download(aws_path, cache_name=None, reset=False):
    with _get_download_lock():
        return _aws_file_download_unlocked(aws_path, cache_name, reset)


def _aws_file_download_unlocked(aws_path, cache_name=None, reset=False):
    """Download a file from the AWS drive s3://astgtmv2/

    **Note:** you need AWS credentials for this to work.

    Parameters
    ----------
    aws_path: path relative to s3://astgtmv2/
    """

    while aws_path.startswith('/'):
        aws_path = aws_path[1:]

    if cache_name is not None:
        cache_obj_name = cache_name
    else:
        cache_obj_name = 'astgtmv2/' + aws_path

    def _dlf(cache_path):
        import boto3
        import botocore
        client = boto3.client('s3')
        logger.info("Downloading %s from s3 to %s..." % (aws_path, cache_path))
        try:
            client.download_file('astgtmv2', aws_path, cache_path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return None
            else:
                raise
        return cache_path

    return _cached_download_helper(cache_obj_name, _dlf, reset)


def file_downloader(www_path, retry_max=5, cache_name=None, reset=False):
    """A slightly better downloader: it tries more than once."""

    local_path = None
    retry_counter = 0
    while retry_counter <= retry_max:
        # Try to download
        try:
            retry_counter += 1
            local_path = _progress_urlretrieve(www_path, cache_name=cache_name,
                                               reset=reset)
            # if no error, exit
            break
        except HTTPError as err:
            # This works well for py3
            if err.code == 404:
                # Ok so this *should* be an ocean tile
                return None
            elif err.code >= 500 and err.code < 600:
                logger.info("Downloading %s failed with HTTP error %s, "
                            "retrying in 10 seconds... %s/%s" %
                            (www_path, err.code, retry_counter, retry_max))
                time.sleep(10)
                continue
            else:
                raise
        except ContentTooShortError:
            logger.info("Downloading %s failed with ContentTooShortError"
                        " error %s, retrying in 10 seconds... %s/%s" %
                        (www_path, err.code, retry_counter, retry_max))
            time.sleep(10)
            continue

    # See if we managed (fail is allowed)
    if not local_path or not os.path.exists(local_path):
        logger.warning('Downloading %s failed.' % www_path)

    return local_path


def empty_cache():
    """Empty oggm's cache directory."""

    if os.path.exists(cfg.CACHE_DIR):
        shutil.rmtree(cfg.CACHE_DIR)
    os.makedirs(cfg.CACHE_DIR)


def expand_path(p):
    """Helper function for os.path.expanduser and os.path.expandvars"""

    return os.path.expandvars(os.path.expanduser(p))


def del_empty_dirs(s_dir):
    """Delete empty directories."""
    b_empty = True
    for s_target in os.listdir(s_dir):
        s_path = os.path.join(s_dir, s_target)
        if os.path.isdir(s_path):
            if not del_empty_dirs(s_path):
                b_empty = False
        else:
            b_empty = False
    if b_empty:
        os.rmdir(s_dir)
    return b_empty


def get_sys_info():
    """Returns system information as a dict"""

    blob = []
    try:
        (sysname, nodename, release,
         version, machine, processor) = platform.uname()
        blob.extend([
            ("python", "%d.%d.%d.%s.%s" % sys.version_info[:]),
            ("python-bits", struct.calcsize("P") * 8),
            ("OS", "%s" % (sysname)),
            ("OS-release", "%s" % (release)),
            ("machine", "%s" % (machine)),
            ("processor", "%s" % (processor)),
        ])
    except:
        pass

    return blob


def show_versions(logger=None):
    """Prints the OGGM version and other system information.

    Parameters
    ----------
    logger : optional
        the logger you want to send the printouts to. If None, will use stdout
    """

    _print = print if logger is None else logger.info

    sys_info = get_sys_info()

    deps = [
        # (MODULE_NAME, f(mod) -> mod version)
        ("oggm", lambda mod: mod.__version__),
        ("numpy", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("geopandas", lambda mod: mod.__version__),
        ("netCDF4", lambda mod: mod.__version__),
        ("matplotlib", lambda mod: mod.__version__),
        ("rasterio", lambda mod: mod.__version__),
        ("fiona", lambda mod: mod.__version__),
        ("osgeo.gdal", lambda mod: mod.__version__),
        ("pyproj", lambda mod: mod.__version__),
    ]

    deps_blob = list()
    for (modname, ver_f) in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = ver_f(mod)
            deps_blob.append((modname, ver))
        except:
            deps_blob.append((modname, None))

    _print("  System info:")
    for k, stat in sys_info:
        _print("%s: %s" % (k, stat))
    _print("  Packages info:")
    for k, stat in deps_blob:
        _print("%s: %s" % (k, stat))


class SuperclassMeta(type):
    """Metaclass for abstract base classes.

    http://stackoverflow.com/questions/40508492/python-sphinx-inherit-
    method-documentation-from-superclass
    """
    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)
        for name, member in cls_dict.items():
            if not getattr(member, '__doc__'):
                try:
                    member.__doc__ = getattr(bases[-1], name).__doc__
                except AttributeError:
                    pass
        return cls


class LRUFileCache():
    """A least recently used cache for temporary files.

    The files which are no longer used are deleted from the disk.
    """
    def __init__(self, l0=None, maxsize=100):
        """Instanciate.

        Parameters
        ----------
        l0 : list
            a list of file paths
        maxsize : int
            the max number of files to keep
        """
        self.files = [] if l0 is None else l0
        self.maxsize = maxsize
        self.purge()

    def purge(self):
        """Remove expired entries."""
        if len(self.files) > self.maxsize:
            fpath = self.files.pop(0)
            if os.path.exists(fpath):
                os.remove(fpath)

    def append(self, fpath):
        """Append a file to the list."""
        if fpath not in self.files:
            self.files.append(fpath)
        self.purge()


def download_oggm_files():
    with _get_download_lock():
        return _download_oggm_files_unlocked()


def _download_oggm_files_unlocked():
    """Checks if the demo data is already on the cache and downloads it."""

    master_sha_url = 'https://api.github.com/repos/%s/commits/master' % \
                     SAMPLE_DATA_GH_REPO
    master_zip_url = 'https://github.com/%s/archive/master.zip' % \
                     SAMPLE_DATA_GH_REPO
    rename_output = False
    shafile = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-commit.txt')
    odir = os.path.join(cfg.CACHE_DIR)
    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master')

    # a file containing the online's file's hash and the time of last check
    if os.path.exists(shafile):
        with open(shafile, 'r') as sfile:
            local_sha = sfile.read().strip()
        last_mod = os.path.getmtime(shafile)
    else:
        # very first download
        local_sha = '0000'
        last_mod = 0

    # test only every hour
    if (time.time() - last_mod) > 3600 and cfg.PARAMS['has_internet']:
        write_sha = True
        try:
            # this might fail with HTTP 403 when server overload
            resp = urlopen(master_sha_url)

            # following try/finally is just for py2/3 compatibility
            # https://mail.python.org/pipermail/python-list/2016-March/704073.html
            try:
                json_str = resp.read().decode('utf-8')
            finally:
                resp.close()
            json_obj = json.loads(json_str)
            master_sha = json_obj['sha']
            # if not same, delete entire dir
            if local_sha != master_sha:
                empty_cache()
            # use sha based download url to avoid cache issues
            master_zip_url = 'https://github.com/%s/archive/%s.zip' % \
                (SAMPLE_DATA_GH_REPO, master_sha)
            rename_output = "oggm-sample-data-%s" % master_sha
        except (HTTPError, URLError):
            master_sha = 'error'
            write_sha = False
    else:
        write_sha = False

    # download only if necessary
    if not os.path.exists(sdir):
        ofile = file_downloader(master_zip_url)
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(odir)
        # rename dir in case of download from different url
        if rename_output:
            fdir = os.path.join(cfg.CACHE_DIR, rename_output)
            shutil.move(fdir, sdir)

    # sha did change, replace
    if write_sha:
        with open(shafile, 'w') as sfile:
            sfile.write(master_sha)

    # list of files for output
    out = dict()
    for root, directories, filenames in os.walk(sdir):
        for filename in filenames:
            if filename in out:
                # This was a stupid thing, and should not happen
                # TODO: duplicates in sample data...
                k = os.path.join(os.path.basename(root), filename)
                assert k not in out
                out[k] = os.path.join(root, filename)
            else:
                out[filename] = os.path.join(root, filename)

    return out


def _download_srtm_file(zone):
    with _get_download_lock():
        return _download_srtm_file_unlocked(zone)


def _download_srtm_file_unlocked(zone):
    """Checks if the srtm data is in the directory and if not, download it.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    outpath = os.path.join(tmpdir, 'srtm_' + zone + '.tif')

    # check if extracted file exists already
    if os.path.exists(outpath):
        return outpath

    # Did we download it yet?
    wwwfile = 'http://droppr.org/srtm/v4.1/6_5x5_TIFs/srtm_' + zone + '.zip'
    dest_file = file_downloader(wwwfile)

    # None means we tried hard but we couldn't find it
    if not dest_file:
        return None

    # ok we have to extract it
    if not os.path.exists(outpath):
        with zipfile.ZipFile(dest_file) as zf:
            zf.extractall(tmpdir)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(outpath)
    cfg.get_lru_handler(tmpdir).append(outpath)
    return outpath


def _download_dem3_viewpano(zone):
    with _get_download_lock():
        return _download_dem3_viewpano_unlocked(zone)


def _download_dem3_viewpano_unlocked(zone):
    """Checks if the DEM3 data is in the directory and if not, download it.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    outpath = os.path.join(tmpdir, zone + '.tif')

    # check if extracted file exists already
    if os.path.exists(outpath):
        return outpath

    # OK, so see if downloaded already
    # some files have a newer version 'v2'
    if zone in ['R33', 'R34', 'R35', 'R36', 'R37', 'R38', 'Q32', 'Q33', 'Q34',
                'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'P31', 'P32', 'P33',
                'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40']:
        ifile = 'http://viewfinderpanoramas.org/dem3/' + zone + 'v2.zip'
    elif zone in ['01-15', '16-30', '31-45', '46-60']:
        ifile = 'http://viewfinderpanoramas.org/ANTDEM3/' + zone + '.zip'
    else:
        ifile = 'http://viewfinderpanoramas.org/dem3/' + zone + '.zip'

    dfile = file_downloader(ifile)

    # None means we tried hard but we couldn't find it
    if not dfile:
        return None

    # ok we have to extract it
    with zipfile.ZipFile(dfile) as zf:
        zf.extractall(tmpdir)

    # Serious issue: sometimes, if a southern hemisphere URL is queried for
    # download and there is none, a NH zip file is downloaded.
    # Example: http://viewfinderpanoramas.org/dem3/SN29.zip yields N29!
    # BUT: There are southern hemisphere files that download properly. However,
    # the unzipped folder has the file name of
    # the northern hemisphere file. Some checks if correct file exists:
    if len(zone)==4 and zone.startswith('S'):
        zonedir = os.path.join(tmpdir, zone[1:])
    else:
        zonedir = os.path.join(tmpdir, zone)
    globlist = glob.glob(os.path.join(zonedir, '*.hgt'))

    # take care of the special file naming cases
    if zone in DEM3REG.keys():
        globlist = glob.glob(os.path.join(tmpdir, '*', '*.hgt'))

    if not globlist:
        raise RuntimeError("We should have some files here, but we don't")

    # merge the single HGT files (can be a bit ineffective, because not every
    # single file might be exactly within extent...)
    rfiles = [rasterio.open(s) for s in globlist]
    dest, output_transform = merge_tool(rfiles)
    profile = rfiles[0].profile
    if 'affine' in profile:
        profile.pop('affine')
    profile['transform'] = output_transform
    profile['height'] = dest.shape[1]
    profile['width'] = dest.shape[2]
    profile['driver'] = 'GTiff'
    with rasterio.open(outpath, 'w', **profile) as dst:
        dst.write(dest)
    for rf in rfiles:
        rf.close()

    # delete original files to spare disk space
    for s in globlist:
        os.remove(s)
    del_empty_dirs(tmpdir)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(outpath)
    cfg.get_lru_handler(tmpdir).append(outpath)
    return outpath


def _download_aster_file(zone, unit):
    with _get_download_lock():
        return _download_aster_file_unlocked(zone, unit)


def _download_aster_file_unlocked(zone, unit):
    """Checks if the aster data is in the directory and if not, download it.

    You need AWS cli and AWS credentials for this. Quoting Timo:

    $ aws configure

    Key ID und Secret you should have
    Region is eu-west-1 and Output Format is json.
    """

    fbname = 'ASTGTM2_' + zone + '.zip'
    dirbname = 'UNIT_' + unit
    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    obname = 'ASTGTM2_' + zone + '_dem.tif'
    outpath = os.path.join(tmpdir, obname)

    aws_path = 'ASTGTM_V2/' + dirbname + '/' + fbname
    dfile = _aws_file_download_unlocked(aws_path)

    if dfile is None:
        # Ok so this *should* be an ocean tile
        return None

    if not os.path.exists(outpath):
        # Extract
        with zipfile.ZipFile(dfile) as zf:
            zf.extract(obname, tmpdir)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(outpath)
    cfg.get_lru_handler(tmpdir).append(outpath)
    return outpath


def _download_alternate_topo_file(fname):
    with _get_download_lock():
        return _download_alternate_topo_file_unlocked(fname)


def _download_alternate_topo_file_unlocked(fname):
    """Checks if the special topo data is in the directory and if not,
    download it from AWS.

    You need AWS cli and AWS credentials for this. Quoting Timo:

        $ aws configure

        Key ID und Secret you should have
        Region is eu-west-1 and Output Format is json.

    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    outpath = os.path.join(tmpdir, fname)

    aws_path = 'topo/' + fname + '.zip'
    dfile = _aws_file_download_unlocked(aws_path)

    if not os.path.exists(outpath):
        logger.info('Extracting ' + fname + '.zip to ' + outpath + '...')
        with zipfile.ZipFile(dfile) as zf:
            zf.extractall(tmpdir)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(outpath)
    cfg.get_lru_handler(tmpdir).append(outpath)
    return outpath


def _get_centerline_lonlat(gdir):
    """Quick n dirty solution to write the centerlines as a shapefile"""

    olist = []
    for i in gdir.divide_ids:
        cls = gdir.read_pickle('centerlines', div_id=i)
        for j, cl in enumerate(cls[::-1]):
            mm = 1 if j==0 else 0
            gs = gpd.GeoSeries()
            gs['RGIID'] = gdir.rgi_id
            gs['DIVIDE'] = i
            gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
            gs['MAIN'] = mm
            tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
            gs['geometry'] = shp_trafo(tra_func, cl.line)
            olist.append(gs)

    return olist


def mkdir(path, reset=False):
    """Checks if directory exists and if not, create one.

    Parameters
    ----------
    reset: erase the content of the directory if exists
    """

    if reset and os.path.exists(path):
        shutil.rmtree(path)

    if not os.path.exists(path):
        os.makedirs(path)


def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().

    https://stackoverflow.com/questions/35155382/copying-specific-files-to-a-
    new-folder-while-maintaining-the-original-subdirect
    """

    def _ignore_patterns(path, names):
        # This is our cuisine
        bname = os.path.basename(path)
        if 'divide' in bname or 'log' in bname:
            keep = []
        else:
            keep = set(name for pattern in patterns
                       for name in fnmatch.filter(names, pattern))
        ignore = set(name for name in names
                     if name not in keep and not
                     os.path.isdir(os.path.join(path, name)))
        return ignore

    return _ignore_patterns


def query_yes_no(question, default="yes"):  # pragma: no cover
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Credits: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def haversine(lon1, lat1, lon2, lat2):
    """Great circle distance between two (or more) points on Earth

    Parameters
    ----------
    lon1 : float
       scalar or array of point(s) longitude
    lat1 : float
       scalar or array of point(s) longitude
    lon2 : float
       scalar or array of point(s) longitude
    lat2 : float
       scalar or array of point(s) longitude

    Returns
    -------
    the distances

    Examples:
    ---------
    >>> haversine(34, 42, 35, 42)
    82633.464752871543
    >>> haversine(34, 42, [35, 36], [42, 42])
    array([  82633.46475287,  165264.11172113])
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius of earth in meters
    return c * r


def interp_nans(array, default=None):
    """Interpolate NaNs using np.interp.

    np.interp is reasonable in that it does not extrapolate, it replaces
    NaNs at the bounds with the closest valid value.
    """

    _tmp = array.copy()
    nans, x = np.isnan(array), lambda z: z.nonzero()[0]
    if np.all(nans):
        # No valid values
        if default is None:
            raise ValueError('No points available to interpolate: '
                             'please set default.')
        _tmp[:] = default
    else:
        _tmp[nans] = np.interp(x(nans), x(~nans), array[~nans])

    return _tmp


def smooth1d(array, window_size=None, kernel='gaussian'):
    """Apply a centered window smoothing to a 1D array.

    Parameters
    ----------
    array : ndarray
        the array to apply the smoothing to
    window_size : int
        the size of the smoothing window
    kernel : str
        the type of smoothing (`gaussian`, `mean`)

    Returns
    -------
    the smoothed array (same dim as input)
    """

    # some defaults
    if window_size is None:
        if len(array) >= 9:
            window_size = 9
        elif len(array) >= 7:
            window_size = 7
        elif len(array) >= 5:
            window_size = 5
        elif len(array) >= 3:
            window_size = 3

    if window_size % 2 == 0:
        raise ValueError('Window should be an odd number.')

    if isinstance(kernel, str):
        if kernel == 'gaussian':
            kernel = gaussian(window_size, 1)
        elif kernel == 'mean':
            kernel = np.ones(window_size)
        else:
            raise NotImplementedError('Kernel: ' + kernel)
    kernel = kernel / np.asarray(kernel).sum()
    return filters.convolve1d(array, kernel, mode='mirror')


def line_interpol(line, dx):
    """Interpolates a shapely LineString to a regularly spaced one.
    
    Shapely's interpolate function does not guaranty equally
    spaced points in space. This is what this function is for.

    We construct new points on the line but at constant distance from the
    preceding one.

    Parameters
    ----------
    line: a shapely.geometry.LineString instance
    dx: the spacing

    Returns
    -------
    a list of equally distanced points
    """

    # First point is easy
    points = [line.interpolate(dx/2.)]

    # Continue as long as line is not finished
    while True:
        pref = points[-1]
        pbs = pref.buffer(dx).boundary.intersection(line)
        if pbs.type == 'Point':
            pbs = [pbs]
        elif pbs.type == 'LineString':
            # This is rare
            pbs = [shpg.Point(c) for c in pbs.coords]
            assert len(pbs) == 2
        elif pbs.type == 'GeometryCollection':
            # This is rare
            opbs = []
            for p in pbs:
                if p.type == 'Point':
                    opbs.append(p)
                elif p.type == 'LineString':
                    opbs.extend([shpg.Point(c) for c in p.coords])
            pbs = opbs
        else:
            assert pbs.type == 'MultiPoint'

        # Out of the point(s) that we get, take the one farthest from the top
        refdis = line.project(pref)
        tdis = np.array([line.project(pb) for pb in pbs])
        p = np.where(tdis > refdis)[0]
        if len(p) == 0:
            break
        points.append(pbs[int(p[0])])

    return points

def md(ref, data, axis=None):
    """Mean Deviation."""
    return np.mean(np.asarray(data)-ref, axis=axis)


def mad(ref, data, axis=None):
    """Mean Absolute Deviation."""
    return np.mean(np.abs(np.asarray(data)-ref), axis=axis)


def rmsd(ref, data, axis=None):
    """Root Mean Square Deviation."""
    return np.sqrt(np.mean((np.asarray(ref)-data)**2, axis=axis))


def rel_err(ref, data):
    """Relative error. Ref should be non-zero"""
    return (np.asarray(data) - ref) / ref


def corrcoef(ref, data):
    """Peason correlation coefficient."""
    return np.corrcoef(ref, data)[0, 1]


def nicenumber(number, binsize, lower=False):
    """Returns the next higher or lower "nice number", given by binsize.

    Examples:
    ---------
    >>> nicenumber(12, 10)
    20
    >>> nicenumber(19, 50)
    50
    >>> nicenumber(51, 50)
    100
    >>> nicenumber(51, 50, lower=True)
    50
    """

    e, _ = divmod(number, binsize)
    if lower:
        return e * binsize
    else:
        return (e+1) * binsize


def signchange(ts):
    """Detect sign changes in a time series.

    http://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-
    for-elements-in-a-numpy-array

    Returns
    -------
    An array with 0s everywhere and 1's when the sign changes
    """
    asign = np.sign(ts)
    sz = asign == 0
    while sz.any():
        asign[sz] = np.roll(asign, 1)[sz]
        sz = asign == 0
    out = ((np.roll(asign, 1) - asign) != 0).astype(int)
    if asign.iloc[0] == asign.iloc[1]:
        out.iloc[0] = 0
    return out


def polygon_intersections(gdf):
    """Computes the intersections between all polygons in a GeoDataFrame.

    Parameters
    ----------
    gdf : Geopandas.GeoDataFrame

    Returns
    -------
    a Geodataframe containing the intersections
    """

    out_cols = ['id_1', 'id_2', 'geometry']
    out = gpd.GeoDataFrame(columns=out_cols)

    gdf = gdf.reset_index()

    for i, major in gdf.iterrows():

        # Exterior only
        major_poly = major.geometry.exterior

        # Remove the major from the list
        agdf = gdf.loc[gdf.index != i]

        # Keep catchments which intersect
        gdfs = agdf.loc[agdf.intersects(major_poly)]

        for j, neighbor in gdfs.iterrows():

            # No need to check if we already found the intersect
            if j in out.id_1 or j in out.id_2:
                continue

            # Exterior only
            neighbor_poly = neighbor.geometry.exterior

            # Ok, the actual intersection
            mult_intersect = major_poly.intersection(neighbor_poly)

            # All what follows is to catch all possibilities
            # Should not happen in our simple geometries but ya never know
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
                line = gpd.GeoDataFrame([[i, j, line]],
                                        columns=out_cols)
                out = out.append(line)

    return out


def year_to_date(yr):
    """Converts a float year to an actual (year, month) tuple.

    Note that this doesn't account for leap years.
    """

    try:
        sec, out_y = math.modf(yr)
        out_y = int(out_y)
        sec = round(sec * SEC_IN_YEAR)
        out_m = np.nonzero(sec < CUMSEC_IN_MONTHS)[0][0] + 1
    except TypeError:
        # TODO: inefficient but no time right now
        out_y = np.zeros(len(yr), np.int64)
        out_m = np.zeros(len(yr), np.int64)
        for i, y in enumerate(yr):
            y, m = year_to_date(y)
            out_y[i] = y
            out_m[i] = m
    return out_y, out_m


def date_to_year(y, m):
    """Converts an integer (year, month) to a float year.

    Note that this doesn't account for leap years.
    """
    ids = np.asarray(m, dtype=np.int) - 1
    return y + BEGINSEC_IN_MONTHS[ids] / SEC_IN_YEAR


def monthly_timeseries(y0, y1=None, ny=None, include_last_year=False):
    """Creates a monthly timeseries in units of floating years.
    """

    if y1 is not None:
        years = np.arange(np.floor(y0), np.floor(y1)+1)
    elif ny is not None:
        years = np.arange(np.floor(y0), np.floor(y0)+ny)
    else:
        raise ValueError("Need at least two positional arguments.")
    months = np.tile(np.arange(12)+1, len(years))
    years = years.repeat(12)
    out = date_to_year(years, months)
    if not include_last_year:
        out = out[:-11]
    return out


@MEMORY.cache
def joblib_read_climate(ncpath, ilon, ilat, default_grad, minmax_grad,
                        use_grad):
    """Prevent to re-compute a timeserie if it was done before.

    TODO: dirty solution, should be replaced by proper input.
    """

    # read the file and data
    with netCDF4.Dataset(ncpath, mode='r') as nc:
        temp = nc.variables['temp']
        prcp = nc.variables['prcp']
        hgt = nc.variables['hgt']
        igrad = np.zeros(len(nc.dimensions['time'])) + default_grad
        ttemp = temp[:, ilat-1:ilat+2, ilon-1:ilon+2]
        itemp = ttemp[:, 1, 1]
        thgt = hgt[ilat-1:ilat+2, ilon-1:ilon+2]
        ihgt = thgt[1, 1]
        thgt = thgt.flatten()
        iprcp = prcp[:, ilat, ilon]

    # Now the gradient
    if use_grad:
        for t, loct in enumerate(ttemp):
            slope, _, _, p_val, _ = stats.linregress(thgt,
                                                     loct.flatten())
            igrad[t] = slope if (p_val < 0.01) else default_grad
        # dont exagerate too much
        igrad = np.clip(igrad, minmax_grad[0], minmax_grad[1])

    return iprcp, itemp, igrad, ihgt


def pipe_log(gdir, task_func_name, err=None):
    """Log the error in a specific directory."""

    time_str = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
    mkdir(fpath)

    fpath = os.path.join(fpath, gdir.rgi_id)

    sep = '; '

    if err is not None:
        fpath += '.ERROR'
    else:
        return  # for now
        fpath += '.SUCCESS'

    with open(fpath, 'a') as f:
        f.write(time_str + sep + task_func_name + sep)
        if err is not None:
            f.write(err.__class__.__name__ + sep + '{}\n'.format(err))
        else:
            f.write(sep + '\n')


def write_centerlines_to_shape(gdirs, filename):
    """Write centerlines in a shapefile"""

    olist = []
    for gdir in gdirs:
        olist.extend(_get_centerline_lonlat(gdir))

    odf = gpd.GeoDataFrame(olist)

    shema = dict()
    props = OrderedDict()
    props['RGIID'] = 'str:14'
    props['DIVIDE'] = 'int:9'
    props['LE_SEGMENT'] = 'int:9'
    props['MAIN'] = 'int:9'
    shema['geometry'] = 'LineString'
    shema['properties'] = props

    crs = {'init': 'epsg:4326'}

    # some writing function from geopandas rep
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


def srtm_zone(lon_ex, lat_ex):
    """Returns a list of SRTM zones covering the desired extent.
    """

    # SRTM are sorted in tiles of 5 degrees
    srtm_x0 = -180.
    srtm_y0 = 60.
    srtm_dx = 5.
    srtm_dy = -5.

    # quick n dirty solution to be sure that we will cover the whole range
    mi, ma = np.min(lon_ex), np.max(lon_ex)
    # int() to avoid Deprec warning:
    lon_ex = np.linspace(mi, ma, int(np.ceil((ma - mi) + 3)))
    mi, ma = np.min(lat_ex), np.max(lat_ex)
    # int() to avoid Deprec warning
    lat_ex = np.linspace(mi, ma, int(np.ceil((ma - mi) + 3)))

    zones = []
    for lon in lon_ex:
        for lat in lat_ex:
            dx = lon - srtm_x0
            dy = lat - srtm_y0
            assert dy < 0
            zx = np.ceil(dx / srtm_dx)
            zy = np.ceil(dy / srtm_dy)
            zones.append('{:02.0f}_{:02.0f}'.format(zx, zy))
    return list(sorted(set(zones)))


def dem3_viewpano_zone(lon_ex, lat_ex):
    """Returns a list of DEM3 zones covering the desired extent.

    http://viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm
    """

    for _f in DEM3REG.keys():

        if (np.min(lon_ex) >= DEM3REG[_f][0]) and \
           (np.max(lon_ex) <= DEM3REG[_f][1]) and \
           (np.min(lat_ex) >= DEM3REG[_f][2]) and \
           (np.max(lat_ex) <= DEM3REG[_f][3]):

            # test some weird inset files in Antarctica
            if (np.min(lon_ex) >= -91.) and (np.max(lon_ex) <= -90.) and \
               (np.min(lat_ex) >= -72.) and (np.max(lat_ex) <= -68.):
                return ['SR15']

            elif (np.min(lon_ex) >= -47.) and (np.max(lon_ex) <= -43.) and \
                 (np.min(lat_ex) >= -61.) and (np.max(lat_ex) <= -60.):
                return ['SP23']

            elif (np.min(lon_ex) >= 162.) and (np.max(lon_ex) <= 165.) and \
                 (np.min(lat_ex) >= -68.) and (np.max(lat_ex) <= -66.):
                return ['SQ58']

            # test some Greenland tiles as GL-North is not rectangular
            elif (np.min(lon_ex) >= -66.) and (np.max(lon_ex) <= -60.) and \
                 (np.min(lat_ex) >= 80.) and (np.max(lat_ex) <= 83.):
                return ['U20']

            elif (np.min(lon_ex) >= -60.) and (np.max(lon_ex) <= -54.) and \
                 (np.min(lat_ex) >= 80.) and (np.max(lat_ex) <= 83.):
                return ['U21']

            elif (np.min(lon_ex) >= -54.) and (np.max(lon_ex) <= -48.) and \
                 (np.min(lat_ex) >= 80.) and (np.max(lat_ex) <= 83.):
                return ['U22']

            else:
                return [_f]

    # if the tile doesn't have a special name, its name can be found like this:
    # corrected SRTMs are sorted in tiles of 6 deg longitude and 4 deg latitude
    srtm_x0 = -180.
    srtm_y0 = 0.
    srtm_dx = 6.
    srtm_dy = 4.

    # quick n dirty solution to be sure that we will cover the whole range
    mi, ma = np.min(lon_ex), np.max(lon_ex)
    # TODO: Fabien, find out what Johannes wanted with this +3
    # +3 is just for the number to become still a bit larger
    # int() to avoid Deprec warning
    lon_ex = np.linspace(mi, ma, int(np.ceil((ma - mi)/srtm_dy)+3))
    mi, ma = np.min(lat_ex), np.max(lat_ex)
    # int() to avoid Deprec warning
    lat_ex = np.linspace(mi, ma, int(np.ceil((ma - mi)/srtm_dx)+3))

    zones = []
    for lon in lon_ex:
        for lat in lat_ex:
            dx = lon - srtm_x0
            dy = lat - srtm_y0
            zx = np.ceil(dx / srtm_dx)
            # convert number to letter
            zy = chr(int(abs(dy / srtm_dy)) + ord('A'))
            if lat >= 0:
                zones.append('%s%02.0f' % (zy, zx))
            else:
                zones.append('S%s%02.0f' % (zy, zx))
    return list(sorted(set(zones)))


def aster_zone(lon_ex, lat_ex):
    """Returns a list of ASTER V2 zones and units covering the desired extent.
    """

    # ASTER is a bit more work. The units are directories of 5 by 5,
    # tiles are 1 by 1. The letter in the filename depends on the sign
    units_dx = 5.

    # quick n dirty solution to be sure that we will cover the whole range
    mi, ma = np.min(lon_ex), np.max(lon_ex)
    # int() to avoid Deprec warning:
    lon_ex = np.linspace(mi, ma, int(np.ceil((ma - mi) + 3)))
    mi, ma = np.min(lat_ex), np.max(lat_ex)
    # int() to avoid Deprec warning:
    lat_ex = np.linspace(mi, ma, int(np.ceil((ma - mi) + 3)))

    zones = []
    units = []
    for lon in lon_ex:
        for lat in lat_ex:
            dx = np.floor(lon)
            zx = np.floor(lon / units_dx) * units_dx
            if math.copysign(1, dx) == -1:
                dx = -dx
                zx = -zx
                lon_let = 'W'
            else:
                lon_let = 'E'

            dy = np.floor(lat)
            zy = np.floor(lat / units_dx) * units_dx
            if math.copysign(1, dy) == -1:
                dy = -dy
                zy = -zy
                lat_let = 'S'
            else:
                lat_let = 'N'

            z = '{}{:02.0f}{}{:03.0f}'.format(lat_let, dy, lon_let, dx)
            u = '{}{:02.0f}{}{:03.0f}'.format(lat_let, zy, lon_let, zx)
            if z not in zones:
                zones.append(z)
                units.append(u)

    return zones, units


def get_demo_file(fname):
    """Returns the path to the desired OGGM file."""

    d = download_oggm_files()
    if fname in d:
        return d[fname]
    else:
        return None


def get_cru_cl_file():
    """Returns the path to the unpacked CRU CL file (is in sample data)."""

    download_oggm_files()

    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master', 'cru')
    fpath = os.path.join(sdir, 'cru_cl2.nc')
    if os.path.exists(fpath):
        return fpath
    else:
        with zipfile.ZipFile(fpath + '.zip') as zf:
            zf.extractall(sdir)
        assert os.path.exists(fpath)
        return fpath


def get_wgms_files():
    """Get the path to the default WGMS-RGI link file and the data dir.

    Returns
    -------
    (file, dir): paths to the files
    """

    if cfg.PATHS['wgms_rgi_links']:
        if not os.path.exists(cfg.PATHS['wgms_rgi_links']):
            raise ValueError('Wrong wgms_rgi_links path provided.')
        # User provided data
        outf = cfg.PATHS['wgms_rgi_links']
        datadir = os.path.join(os.path.dirname(outf), 'mbdata')
        if not os.path.exists(datadir):
            raise ValueError('The WGMS data directory is missing')
        return outf, datadir

    # Roll our own
    download_oggm_files()
    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master', 'wgms')
    outf = os.path.join(sdir, 'rgi_wgms_links_20170217_RGIV5.csv')
    assert os.path.exists(outf)
    datadir = os.path.join(sdir, 'mbdata')
    assert os.path.exists(datadir)
    return outf, datadir


def get_glathida_file():
    """Get the path to the default GlaThiDa-RGI link file.

    Returns
    -------
    file: paths to the file
    """

    if cfg.PATHS['glathida_rgi_links']:
        if not os.path.exists(cfg.PATHS['glathida_rgi_links']):
            raise ValueError('Wrong glathida_rgi_links path provided.')
        # User provided data
        return cfg.PATHS['glathida_rgi_links']

    # Roll our own
    download_oggm_files()
    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master', 'glathida')
    outf = os.path.join(sdir, 'rgi_glathida_links_2014_RGIV5.csv')
    assert os.path.exists(outf)
    return outf


def get_rgi_dir():
    """Returns a path to the RGI directory.

    If the RGI files are not present, download them.

    Returns
    -------
    path to the RGI directory
    """

    with _get_download_lock():
        return _get_rgi_dir_unlocked()


def _get_rgi_dir_unlocked():

    rgi_dir = cfg.PATHS['rgi_dir']

    # Be sure the user gave a sensible path to the RGI dir
    if not rgi_dir:
        raise ValueError('The RGI data directory has to be'
                         'specified explicitly.')
    rgi_dir = os.path.abspath(os.path.expanduser(rgi_dir))
    rgi_dir = os.path.join(rgi_dir, 'RGIV5')
    mkdir(rgi_dir)

    dfile = 'http://www.glims.org/RGI/rgi50_files/rgi50.zip'
    test_file = os.path.join(rgi_dir, '000_rgi50_manifest.txt')

    if not os.path.exists(test_file):
        # if not there download it
        ofile = file_downloader(dfile)
        # Extract root
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(rgi_dir)
        # Extract subdirs
        pattern = '*_rgi50_*.zip'
        for root, dirs, files in os.walk(cfg.PATHS['rgi_dir']):
            for filename in fnmatch.filter(files, pattern):
                zfile = os.path.join(root, filename)
                with zipfile.ZipFile(zfile) as zf:
                    ex_root = zfile.replace('.zip', '')
                    mkdir(ex_root)
                    zf.extractall(ex_root)
                # delete the zipfile after success
                os.remove(zfile)
    return rgi_dir


def get_rgi_intersects_dir(reset=False):
    """Returns a path to the RGI directory containing the intersects.

    If the files are not present, download them.

    Returns
    -------
    path to the directory
    """

    with _get_download_lock():
        return _get_rgi_intersects_dir_unlocked(reset=reset)


def _get_rgi_intersects_dir_unlocked(reset=False):

    rgi_dir = cfg.PATHS['rgi_dir']

    # Be sure the user gave a sensible path to the RGI dir
    if not rgi_dir:
        raise ValueError('The RGI data directory has to be'
                         'specified explicitly.')

    rgi_dir = os.path.abspath(os.path.expanduser(rgi_dir))
    mkdir(rgi_dir, reset=reset)

    dfile = ('https://www.dropbox.com/s/y73sdxygdiq7whv/' +
             'RGI_V5_Intersects.zip?dl=1')
    test_file = os.path.join(rgi_dir, 'RGI_V5_Intersects',
                             'Intersects_OGGM_Manifest.txt')
    if not os.path.exists(test_file):
        # if not there download it
        ofile = file_downloader(dfile, reset=reset)
        # Extract root
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(rgi_dir)

    return os.path.join(rgi_dir, 'RGI_V5_Intersects')


def get_rgi_corrected_dir(reset=False):
    """Returns a path to the RGI directory containing the new divided files.

    If the files are not present, download them.

    Returns
    -------
    path to the directory
    """

    with _get_download_lock():
        return _get_rgi_corrected_dir_unlocked(reset=reset)


def _get_rgi_corrected_dir_unlocked(reset=False):

    rgi_dir = cfg.PATHS['rgi_dir']

    # Be sure the user gave a sensible path to the RGI dir
    if not rgi_dir:
        raise ValueError('The RGI data directory has to be'
                         'specified explicitly.')

    rgi_dir = os.path.abspath(os.path.expanduser(rgi_dir))
    mkdir(rgi_dir, reset=reset)

    dfile = ('https://www.dropbox.com/s/85xdglv0zredue9/' +
             'RGIV5_Corrected.zip?dl=1')
    test_file = os.path.join(rgi_dir, 'RGIV5_Corrected',
                             'RGIV5_Corrected_OGGM_Manifest.txt')
    if not os.path.exists(test_file):
        # if not there download it
        ofile = file_downloader(dfile, reset=reset)
        # Extract root
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(rgi_dir)

    return os.path.join(rgi_dir, 'RGIV5_Corrected')


def get_cru_file(var=None):
    """Returns a path to the desired CRU TS file.

    If the file is not present, download it.

    Parameters
    ----------
    var: 'tmp' or 'pre'

    Returns
    -------
    path to the CRU file
    """
    with _get_download_lock():
        return _get_cru_file_unlocked(var)


def _get_cru_file_unlocked(var=None):

    cru_dir = cfg.PATHS['cru_dir']

    # Be sure the user gave a sensible path to the climate dir
    if not cru_dir:
        raise ValueError('The CRU data directory has to be'
                         'specified explicitly.')
    cru_dir = os.path.abspath(os.path.expanduser(cru_dir))
    mkdir(cru_dir)

    # Be sure input makes sense
    if var not in ['tmp', 'pre']:
        raise ValueError('CRU variable {} does not exist!'.format(var))

    # The user files may have different dates, so search for patterns
    bname = 'cru_ts*.{}.dat.nc'.format(var)
    search = glob.glob(os.path.join(cru_dir, bname))
    if len(search) == 1:
        ofile = search[0]
    elif len(search) > 1:
        raise ValueError('The CRU filename should match "{}".'.format(bname))
    else:
        # if not there download it
        cru_filename = 'cru_ts3.24.01.1901.2015.{}.dat.nc'.format(var)
        cru_url = CRU_SERVER + '{}/'.format(var) + cru_filename + '.gz'
        dlfile = file_downloader(cru_url)
        ofile = os.path.join(cru_dir, cru_filename)
        with gzip.GzipFile(dlfile) as zf:
            with open(ofile, 'wb') as outfile:
                for line in zf:
                    outfile.write(line)
    return ofile


def get_topo_file(lon_ex, lat_ex, rgi_region=None, source=None):
    """
    Returns a list with path(s) to the DEM file(s) covering the desired extent.

    If the needed files for covering the extent are not present, download them.

    By default it will be referred to SRTM for [-60S; 60N], GIMP for Greenland,
    RAMP for Antarctica, and a corrected DEM3 (viewfinderpanoramas.org)
    elsewhere.

    A user-specified data source can be given with the ``source`` keyword.

    Parameters
    ----------
    lon_ex : tuple, required
        a (min_lon, max_lon) tuple delimiting the requested area longitudes
    lat_ex : tuple, required
        a (min_lat, max_lat) tuple delimiting the requested area latitudes
    rgi_region : int, optional
        the RGI region number (required for the GIMP DEM)
    source : str or list of str, optional
        If you want to force the use of a certain DEM source. Available are:
          - 'USER' : file set in cfg.PATHS['dem_file']
          - 'SRTM' : SRTM v4.1
          - 'GIMP' : https://bpcrc.osu.edu/gdg/data/gimpdem
          - 'RAMP' : http://nsidc.org/data/docs/daac/nsidc0082_ramp_dem.gd.html
          - 'DEM3' : http://viewfinderpanoramas.org/
          - 'ASTER' : ASTER data

    Returns
    -------
    tuple: (list with path(s) to the DEM file, data source)
    """

    if source is not None and not isinstance(source, string_types):
        # check all user options
        for s in source:
            demf, source_str = get_topo_file(lon_ex, lat_ex,
                                             rgi_region=rgi_region,
                                             source=s)
            if demf[0]:
                return demf, source_str

    # Did the user specify a specific DEM file?
    if 'dem_file' in cfg.PATHS and os.path.isfile(cfg.PATHS['dem_file']):
        source = 'USER' if source is None else source
        if source == 'USER':
            return [cfg.PATHS['dem_file']], source

    # GIMP is in polar stereographic, not easy to test if glacier is on the map
    # It would be possible with a salem grid but this is a bit more expensive
    # Instead, we are just asking RGI for the region
    if source == 'GIMP' or (rgi_region is not None and int(rgi_region) == 5):
        source = 'GIMP' if source is None else source
        if source == 'GIMP':
            _file = _download_alternate_topo_file('gimpdem_90m.tif')
            return [_file], source

    # Same for Antarctica
    if source == 'RAMP' or (rgi_region is not None and int(rgi_region) == 19):
        if np.max(lat_ex) > -60:
            # special case for some distant islands
            source = 'DEM3' if source is None else source
        else:
            source = 'RAMP' if source is None else source
        if source == 'RAMP':
            _file = _download_alternate_topo_file('AntarcticDEM_wgs84.tif')
            return [_file], source

    # Anywhere else on Earth we check for DEM3, ASTER, or SRTM
    if (np.min(lat_ex) < -60.) or (np.max(lat_ex) > 60.) or \
            (source == 'DEM3') or (source == 'ASTER'):
        # default is DEM3
        source = 'DEM3' if source is None else source
        if source == 'DEM3':
            # use corrected viewpanoramas.org DEM
            zones = dem3_viewpano_zone(lon_ex, lat_ex)
            sources = []
            for z in zones:
                sources.append(_download_dem3_viewpano(z))
            source_str = source
        if source == 'ASTER':
            # use ASTER
            zones, units = aster_zone(lon_ex, lat_ex)
            sources = []
            for z, u in zip(zones, units):
                sf = _download_aster_file(z, u)
                if sf is not None:
                    sources.append(sf)
            source_str = source
    else:
        source = 'SRTM' if source is None else source
        if source == 'SRTM':
            zones = srtm_zone(lon_ex, lat_ex)
            sources = []
            for z in zones:
                sources.append(_download_srtm_file(z))
            source_str = source

    # filter for None (e.g. oceans)
    sources = [s for s in sources if s]
    if sources:
        return sources, source_str
    else:
        raise RuntimeError('No topography file available for extent lat:{0},'
                           'lon:{1}!'.format(lat_ex, lon_ex))


def compile_run_output(gdirs, path=True, monthly=False, filesuffix=''):
    """Merge the runs output of the glacier directories into one file.


    Parameters
    ----------
    gdirs: the list of GlacierDir to process.
    path: where to store (default is on the working dir).
    monthly: wether to store monthly values (default is yearly)
    filesuffix: the filesuffix of the run
    """

    # Get the dimensions of all this
    rgi_ids = [gd.rgi_id for gd in gdirs]

    # The first gdir might have blown up, try some others
    i = 0
    while True:
        if i >= len(gdirs):
            raise RuntimeError('Found no valid glaciers!')
        try:
            ppath = gdirs[i].get_filepath('model_diagnostics',
                                          filesuffix=filesuffix)
            with xr.open_dataset(ppath) as ds_diag:
                time = ds_diag.time.values
                year = ds_diag.year.values
                month = ds_diag.month.values
            break
        except:
            i += 1

    # Monthly or not
    if monthly:
        pkeep = np.ones(len(time), dtype=np.bool)
    else:
        pkeep = np.where(month == 1)

    time = time[pkeep]
    year = year[pkeep]
    month = month[pkeep]
    ds = xr.Dataset(coords={'time': ('time', time),
                            'year': ('time', year),
                            'month': ('time', month),
                            'rgi_id': ('rgi_id', rgi_ids)
                            })
    shape = (len(time), len(rgi_ids))
    vol = np.zeros(shape)
    area = np.zeros(shape)
    length = np.zeros(shape)
    ela = np.zeros(shape)
    for i, gdir in enumerate(gdirs):
        try:
            ppath = gdir.get_filepath('model_diagnostics',
                                      filesuffix=filesuffix)
            with xr.open_dataset(ppath) as ds_diag:
                vol[:, i] = ds_diag.volume_m3.values[pkeep]
                area[:, i] = ds_diag.area_m2.values[pkeep]
                length[:, i] = ds_diag.length_m.values[pkeep]
                ela[:, i] = ds_diag.ela_m.values[pkeep]
        except:
            vol[:, i] = np.NaN
            area[:, i] = np.NaN
            length[:, i] = np.NaN
            ela[:, i] = np.NaN

    ds['volume'] = (('time', 'rgi_id'), vol)
    ds['volume'].attrs['units'] = 'm3'
    ds['volume'].attrs['description'] = 'Total glacier volume'
    ds['area'] = (('time', 'rgi_id'), area)
    ds['area'].attrs['units'] = 'm2'
    ds['area'].attrs['description'] = 'Total glacier area'
    ds['length'] = (('time', 'rgi_id'), length)
    ds['length'].attrs['units'] = 'm'
    ds['length'].attrs['description'] = 'Glacier length'
    ds['ela'] = (('time', 'rgi_id'), ela)
    ds['ela'].attrs['units'] = 'm'
    ds['ela'].attrs['description'] = 'Glacier Equilibrium Line Altitude (ELA)'

    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                'run_output' + filesuffix + '.nc')
        ds.to_netcdf(path)
    return ds


def glacier_characteristics(gdirs, filesuffix='', path=True):
    """Gathers as many statistics as possible about a list of glacier
    directories.
    
    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.
    
    Parameters
    ----------
    gdirs: the list of GlacierDir to process.
    filesuffix : str
        add suffix to output file
    path:
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """

    out_df = []
    for gdir in gdirs:

        d = OrderedDict()

        # Easy stats - this should always be possible
        d['rgi_id'] = gdir.rgi_id
        d['name'] = gdir.name
        d['cenlon'] = gdir.cenlon
        d['cenlat'] = gdir.cenlat
        d['rgi_area_km2'] = gdir.rgi_area_km2
        d['glacier_type'] = gdir.glacier_type
        d['terminus_type'] = gdir.terminus_type

        # The rest is less certain. We put this in a try block and see
        try:
            # Masks related stuff
            if gdir.has_file('gridded_data', div_id=0):
                fpath = gdir.get_filepath('gridded_data', div_id=0)
                with netCDF4.Dataset(fpath) as nc:
                    mask = nc.variables['glacier_mask'][:]
                    topo = nc.variables['topo'][:]
                d['dem_mean_elev'] = np.mean(topo[np.where(mask == 1)])
                d['dem_max_elev'] = np.max(topo[np.where(mask == 1)])
                d['dem_min_elev'] = np.min(topo[np.where(mask == 1)])

            # Divides
            d['n_divides'] = len(list(gdir.divide_ids))

            # Centerlines
            if gdir.has_file('centerlines', div_id=1):
                cls = []
                for i in gdir.divide_ids:
                    cls.extend(gdir.read_pickle('centerlines', div_id=i))
                longuest = 0.
                for cl in cls:
                    longuest = np.max([longuest, cl.dis_on_line[-1]])
                d['n_centerlines'] = len(cls)
                d['longuest_centerline_km'] = longuest * gdir.grid.dx / 1000.

            # MB and flowline related stuff
            if gdir.has_file('inversion_flowlines', div_id=1):
                amb = np.array([])
                h = np.array([])
                widths = np.array([])
                slope = np.array([])

                for div_id in gdir.divide_ids:
                    fls = gdir.read_pickle('inversion_flowlines',
                                           div_id=div_id)
                    dx = fls[0].dx * gdir.grid.dx
                    for fl in fls:
                        amb = np.append(amb, fl.apparent_mb)
                        hgt = fl.surface_h
                        h = np.append(h, hgt)
                        widths = np.append(widths, fl.widths * dx)
                        slope = np.append(slope,
                                          np.arctan(-np.gradient(hgt, dx)))

                pacc = np.where(amb >= 0)
                pab = np.where(amb < 0)
                d['aar'] = np.sum(widths[pacc]) / np.sum(widths[pab])
                try:
                    # Try to get the slope
                    mb_slope, _, _, _, _ = stats.linregress(h[pab], amb[pab])
                    d['mb_grad'] = mb_slope
                except:
                    # we don't mind if something goes wrong
                    d['mb_grad'] = np.NaN
                d['avg_width'] = np.mean(widths)
                d['avg_slope'] = np.mean(slope)

            # Climate
            if gdir.has_file('climate_monthly', div_id=0):
                cf = gdir.get_filepath('climate_monthly')
                with xr.open_dataset(cf) as cds:
                    d['clim_alt'] = cds.ref_hgt
                    t = cds.temp.mean(dim='time').values
                    if 'dem_mean_elev' in d:
                        t = (t - (d['dem_mean_elev'] - d['clim_alt']) *
                             cfg.PARAMS['temp_default_gradient'])
                    else:
                        t = np.NaN
                    d['clim_temp_avgh'] = t
                    d['clim_prcp'] = cds.prcp.mean(dim='time').values * 12

            # Inversion
            if gdir.has_file('inversion_output', div_id=1):
                vol = []
                for i in gdir.divide_ids:
                    cl = gdir.read_pickle('inversion_output', div_id=i)
                    for c in cl:
                        vol.extend(c['volume'])
                d['inv_volume_km3'] = np.nansum(vol) * 1e-9
                area = gdir.rgi_area_km2
                d['inv_thickness_m'] = d['inv_volume_km3'] / area * 1000
                d['vas_volume_km3'] = 0.034*(area**1.375)
                d['vas_thickness_m'] = d['vas_volume_km3'] / area * 1000

            # Calving
            if gdir.has_file('calving_output', div_id=1):
                all_calving_data = []
                all_width = []
                for i in gdir.divide_ids:
                    cl = gdir.read_pickle('calving_output', div_id=i)
                    for c in cl:
                        all_calving_data = c['calving_fluxes'][-1]
                        all_width = c['t_width']
                d['calving_flux'] = all_calving_data
                d['calving_front_width'] = all_width
            else:
                d['calving_flux'] = np.NaN
                d['calving_front_width'] = np.NaN
        except:
            # We're good with any error - we store the dict anyway below
            pass

        out_df.append(d)

    out = pd.DataFrame(out_df).set_index('rgi_id')
    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                       'glacier_characteristics'+filesuffix+'.csv'))
        else:
            out.to_csv(path)
    return out


class DisableLogger():
    """Context manager to temporarily disable all loggers."""
    def __enter__(self):
        logging.disable(logging.ERROR)
    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


class entity_task(object):
    """Decorator for common job-controlling logic.

    All tasks share common operations. This decorator is here to handle them:
    exceptions, logging, and (some day) database for job-controlling.
    """

    def __init__(self, log, writes=[]):
        """Decorator syntax: ``@oggm_task(writes=['dem', 'outlines'])``

        Parameters
        ----------
        writes: list
            list of files that the task will write down to disk (must be
            available in ``cfg.BASENAMES``)
        """
        self.log = log
        self.writes = writes

        cnt =  ['    Returns']
        cnt += ['    -------']
        cnt += ['    Files writen to the glacier directory:']

        for k in sorted(writes):
            cnt += [cfg.BASENAMES.doc_str(k)]
        self.iodoc = '\n'.join(cnt)

    def __call__(self, task_func):
        """Decorate."""

        # Add to the original docstring
        if task_func.__doc__ is None:
            raise RuntimeError('Entity tasks should have a docstring!')

        task_func.__doc__ = '\n'.join((task_func.__doc__, self.iodoc))

        @wraps(task_func)
        def _entity_task(gdir, reset=None, **kwargs):

            if reset is None:
                reset = not cfg.PARAMS['auto_skip_task']

            # Do we need to run this task?
            s = gdir.get_task_status(task_func)
            if not reset and s and ('SUCCESS' in s):
                return

            # Log what we are doing
            if not task_func.__dict__.get('divide_task', False):
                self.log.info('(%s) %s', gdir.rgi_id, task_func.__name__)

            # Run the task
            try:
                out = task_func(gdir, **kwargs)
                gdir.log(task_func)
            except Exception as err:
                # Something happened
                out = None
                gdir.log(task_func, err=err)
                task_func_name = task_func.__name__
                # Filesuffix are typically used to differenciate tasks
                fsuffix = kwargs.get('filesuffix', False)
                if fsuffix:
                    task_func_name += fsuffix
                pipe_log(gdir, task_func_name, err=err)
                self.log.error('%s occurred during task %s on %s!',
                        type(err).__name__, task_func.__name__, gdir.rgi_id)
                if not cfg.PARAMS['continue_on_error']:
                    raise
            return out

        _entity_task.__dict__['is_entity_task'] = True
        return _entity_task


class divide_task(object):
    """Decorator for common logic on divides.

    Simply calls the decorated task once for each divide.
    """

    def __init__(self, log, add_0=False):
        """Decorator

        Parameters
        ----------
        add_0: bool, default=False
            If the task also needs to be run on divide 0
        """
        self.log = log
        self.add_0 = add_0
        self._cdoc = """"
            div_id : int
                the ID of the divide to process. Should be left to the default
                ``None`` unless you know what you do.
        """

    def __call__(self, task_func):
        """Decorate."""

        @wraps(task_func)
        def _divide_task(gdir, div_id=None, **kwargs):
            if div_id is None:
                ids = gdir.divide_ids
                if self.add_0:
                    ids = list(ids) + [0]
                for i in ids:
                    self.log.info('(%s) %s, divide %d', gdir.rgi_id,
                                  task_func.__name__, i)
                    task_func(gdir, div_id=i, **kwargs)
            else:
                # For testing only
                task_func(gdir, div_id=div_id, **kwargs)

        # For the logger later on
        _divide_task.__dict__['divide_task'] = True
        return _divide_task


def global_task(task_func):
    """
    Decorator for common job-controlling logic.

    Indicates that this task expects a list of all GlacierDirs as parameter
    instead of being called once per dir.
    """

    task_func.__dict__['global_task'] = True
    return task_func


def filter_rgi_name(name):
    """Remove spurious characters and trailing blanks from RGI glacier name.
    """

    if name is None or len(name) == 0:
        return ''

    if name[-1] in ['À', 'È', 'è', '\x9c', '3', 'Ð', '°', '¾',
                    '\r', '\x93', '¤', '0', '`', '/', 'C', '@',
                    'Å', '\x06', '\x10', '^', 'å', ';']:
        return filter_rgi_name(name[:-1])

    return name.strip().title()


class GlacierDirectory(object):
    """Organizes read and write access to the glacier's files.

    It handles a glacier directory created in a base directory (default
    is the "per_glacier" folder in the working directory). The role of a
    GlacierDirectory is to give access to file paths and to I/O operations.
    The user should not care about *where* the files are
    located, but should know their name (see :ref:`basenames`).

    If the directory does not exist, it will be created.

    A glacier entity has one or more divides. See :ref:`glacierdir`
    for more information.

    Attributes
    ----------
    dir : str
        path to the directory
    rgi_id : str
        The glacier's RGI identifier
    glims_id : str
        The glacier's GLIMS identifier (when available)
    rgi_area_km2 : float
        The glacier's RGI area (km2)
    cenlon, cenlat : float
        The glacier centerpoint's lon/lat
    rgi_date : datetime
        The RGI's BGNDATE attribute if available. Otherwise, defaults to
        2003-01-01
    rgi_region : str
        The RGI region name
    name : str
        The RGI glacier name (if Available)
    glacier_type : str
        The RGI glacier type ('Glacier', 'Ice cap', 'Perennial snowfield',
        'Seasonal snowfield')
    terminus_type : str
        The RGI terminus type ('Land-terminating', 'Marine-terminating',
        'Lake-terminating', 'Dry calving', 'Regenerated', 'Shelf-terminating')
    is_tidewater : bool
        Is the glacier a caving glacier?
    inversion_calving_rate : float
        Calving rate used for the inversion
    """

    def __init__(self, rgi_entity, base_dir=None, reset=False):
        """Creates a new directory or opens an existing one.

        Parameters
        ----------
        rgi_entity : a `GeoSeries <http://geopandas.org/data_structures.html#geoseries>`_ or str
            glacier entity read from the shapefile (or a valid RGI ID if the
            directory exists)
        base_dir : str
            path to the directory where to open the directory.
            Defaults to `cfg.PATHS['working_dir'] + /per_glacier/`
        reset : bool, default=False
            empties the directory at construction (careful!)

        """

        if base_dir is None:
            base_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')

        # RGI IDs are also valid entries
        if isinstance(rgi_entity, string_types):
            _shp = os.path.join(base_dir, rgi_entity[:8], rgi_entity[:11],
                                rgi_entity, 'outlines.shp')
            rgi_entity = read_shapefile(_shp).iloc[0]

        try:
            # Assume RGI V4
            self.rgi_id = rgi_entity.RGIID
            self.glims_id = rgi_entity.GLIMSID
            self.rgi_area_km2 = float(rgi_entity.AREA)
            self.cenlon = float(rgi_entity.CENLON)
            self.cenlat = float(rgi_entity.CENLAT)
            self.rgi_region = '{:02d}'.format(int(rgi_entity.O1REGION))
            self.rgi_subregion = self.rgi_region + '-' + \
                                 '{:02d}'.format(int(rgi_entity.O2REGION))
            name = rgi_entity.NAME
            rgi_datestr = rgi_entity.BGNDATE
            gtype = rgi_entity.GLACTYPE
        except AttributeError:
            # Should be V5
            self.rgi_id = rgi_entity.RGIId
            self.glims_id = rgi_entity.GLIMSId
            self.rgi_area_km2 = float(rgi_entity.Area)
            self.cenlon = float(rgi_entity.CenLon)
            self.cenlat = float(rgi_entity.CenLat)
            self.rgi_region = '{:02d}'.format(int(rgi_entity.O1Region))
            self.rgi_subregion = (self.rgi_region + '-' +
                                  '{:02d}'.format(int(rgi_entity.O2Region)))
            name = rgi_entity.Name
            rgi_datestr = rgi_entity.BgnDate
            gtype = rgi_entity.GlacType

        # remove spurious characters and trailing blanks
        self.name = filter_rgi_name(name)

        # region
        n = cfg.RGI_REG_NAMES.loc[int(self.rgi_region)].values[0]
        self.rgi_region_name = self.rgi_region + ': ' + n
        n = cfg.RGI_SUBREG_NAMES.loc[self.rgi_subregion].values[0]
        self.rgi_subregion_name = self.rgi_subregion + ': ' + n

        # Read glacier attrs
        keys = {'0': 'Glacier',
                '1': 'Ice cap',
                '2': 'Perennial snowfield',
                '3': 'Seasonal snowfield',
                '9': 'Not assigned',
                }
        self.glacier_type = keys[gtype[0]]
        keys = {'0': 'Land-terminating',
                '1': 'Marine-terminating',
                '2': 'Lake-terminating',
                '3': 'Dry calving',
                '4': 'Regenerated',
                '5': 'Shelf-terminating',
                '9': 'Not assigned',
                }
        self.terminus_type = keys[gtype[1]]
        self.is_tidewater = self.terminus_type in ['Marine-terminating',
                                                   'Lake-terminating']
        self.inversion_calving_rate = 0.
        self.is_icecap = self.glacier_type == 'Ice cap'

        # convert the date
        try:
            rgi_date = pd.to_datetime(rgi_datestr[0:4],
                                      errors='raise', format='%Y')
        except:
            rgi_date = None
        self.rgi_date = rgi_date

        # rgi version can be useful, too
        self.rgi_version = self.rgi_id.split('-')[0]

        # The divides dirs are created by gis.define_glacier_region, but we
        # make the root dir
        self.dir = os.path.join(base_dir, self.rgi_id[:8], self.rgi_id[:11],
                                self.rgi_id)
        if reset and os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        mkdir(self.dir)

        # logging file
        self.logfile = os.path.join(self.dir, 'log.txt')

        # Optimization
        self._mbdf = None

    def __repr__(self):

        summary = ['<oggm.GlacierDirectory>']
        summary += ['  RGI id: ' + self.rgi_id]
        summary += ['  Region: ' + self.rgi_region_name]
        summary += ['  Subregion: ' + self.rgi_subregion_name]
        if self.name :
            summary += ['  Name: ' + self.name]
        summary += ['  Glacier type: ' + str(self.glacier_type)]
        summary += ['  Terminus type: ' + str(self.terminus_type)]
        summary += ['  Area: ' + str(self.rgi_area_km2) + ' mk2']
        summary += ['  Lon, Lat: (' + str(self.cenlon) + ', ' +
                    str(self.cenlat) + ')']
        if os.path.isfile(self.get_filepath('glacier_grid')):
            summary += ['  Grid (nx, ny): (' + str(self.grid.nx) + ', ' +
                        str(self.grid.ny) + ')']
            summary += ['  Grid (dx, dy): (' + str(self.grid.dx) + ', ' +
                        str(self.grid.dy) + ')']
        return '\n'.join(summary) + '\n'

    @lazy_property
    def grid(self):
        """A ``salem.Grid`` handling the georeferencing of the local grid"""
        return salem.Grid.from_json(self.get_filepath('glacier_grid'))

    @lazy_property
    def rgi_area_m2(self):
        """The glacier's RGI area (m2)."""
        return self.rgi_area_km2 * 10**6

    @property
    def divide_dirs(self):
        """List of the glacier divides directories"""
        dirs = [self.dir]
        dirs += sorted(glob.glob(os.path.join(self.dir, 'divide_*')))
        return dirs

    @property
    def n_divides(self):
        """Number of glacier divides"""
        return len(self.divide_dirs)-1

    @property
    def divide_ids(self):
        """Iterator over the glacier divides ids"""
        return range(1, self.n_divides+1)

    def copy_to_basedir(self, base_dir, setup='run'):
        """Copies the glacier directory and its content to a new location.

        This utility function allows to select certain files only, thus
        saving time at copy.

        Parameters
        ----------
        basedir : str
            path to the new base directory (should end with "per_glacier" most
            of the times)
        setup : str
            set up you want the copied directory to be useful for. Currently
            supported are 'all' (copy the entire directory) and 'run' (copy)
            the necessary files for a dynamical run).
        """

        base_dir = os.path.abspath(base_dir)
        new_dir = os.path.join(base_dir, self.rgi_id[:8], self.rgi_id[:11],
                               self.rgi_id)
        if setup == 'run':
            paths = ['model_flowlines', 'inversion_params',
                     'local_mustar', 'climate_monthly']
            paths = ('*' + p + '*' for p in paths)
            shutil.copytree(self.dir, new_dir,
                            ignore=include_patterns(*paths))
        elif setup == 'all':
            shutil.copytree(self.dir, new_dir)
        else:
            raise ValueError('setup not understood: {}'.format(setup))

    def get_filepath(self, filename, div_id=0, delete=False, filesuffix=''):
        """Absolute path to a specific file.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        div_id : int or str
            the divide for which you want to get the file path (set to
            'major' to get the major divide according to
            compute_downstream_lines)
        delete : bool
            delete the file if exists
        filesuffix : str
            append a suffix to the filename (useful for model runs). Note
            that the BASENAME remains same.

        Returns
        -------
        The absolute path to the desired file
        """

        if filename not in cfg.BASENAMES:
            raise ValueError(filename + ' not in cfg.BASENAMES.')

        if div_id == 'major':
            div_id = self.read_pickle('major_divide', div_id=0)

        dir = self.divide_dirs[div_id]
        fname = cfg.BASENAMES[filename]
        if filesuffix:
            fname = fname.split('.')
            assert len(fname) == 2
            fname = fname[0] + filesuffix + '.' + fname[1]
        out = os.path.join(dir, fname)
        if delete and os.path.isfile(out):
            os.remove(out)
        return out

    def has_file(self, filename, div_id=0):
        """Checks if a file exists.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        div_id : int
            the divide for which you want to get the file path
        """

        return os.path.exists(self.get_filepath(filename, div_id=div_id))

    def read_pickle(self, filename, div_id=0, use_compression=None):
        """Reads a pickle located in the directory.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        div_id : int
            the divide for which you want to get the file path
        use_compression : bool
            whether or not the file ws compressed. Default is to use
            cfg.PARAMS['use_compression'] for this (recommended)
        Returns
        -------
        An object read from the pickle
        """
        use_comp = (use_compression if use_compression is not None
                    else cfg.PARAMS['use_compression'])
        _open = gzip.open if use_comp else open
        with _open(self.get_filepath(filename, div_id), 'rb') as f:
            out = pickle.load(f)

        return out

    def write_pickle(self, var, filename, div_id=0, use_compression=None):
        """ Writes a variable to a pickle on disk.

        Parameters
        ----------
        var : object
            the variable to write to disk
        filename : str
            file name (must be listed in cfg.BASENAME)
        div_id : int
            the divide for which you want to get the file path
        use_compression : bool
            whether or not the file ws compressed. Default is to use
            cfg.PARAMS['use_compression'] for this (recommended)
        """
        use_comp = (use_compression if use_compression is not None
                    else cfg.PARAMS['use_compression'])
        _open = gzip.open if use_comp else open
        with _open(self.get_filepath(filename, div_id), 'wb') as f:
            pickle.dump(var, f, protocol=-1)

    def create_gridded_ncdf_file(self, fname, div_id=0):
        """Makes a gridded netcdf file template.

        The other variables have to be created and filled by the calling
        routine.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        div_id : int
            the divide for which you want to get the file path

        Returns
        -------
        a ``netCDF4.Dataset`` object.
        """

        # overwrite as default
        fpath = self.get_filepath(fname, div_id)
        if os.path.exists(fpath):
            os.remove(fpath)

        nc = netCDF4.Dataset(fpath, 'w', format='NETCDF4')

        xd = nc.createDimension('x', self.grid.nx)
        yd = nc.createDimension('y', self.grid.ny)

        nc.author = 'OGGM'
        nc.author_info = 'Open Global Glacier Model'
        nc.proj_srs = self.grid.proj.srs

        lon, lat = self.grid.ll_coordinates
        x = self.grid.x0 + np.arange(self.grid.nx) * self.grid.dx
        y = self.grid.y0 + np.arange(self.grid.ny) * self.grid.dy

        v = nc.createVariable('x', 'f4', ('x',), zlib=True)
        v.units = 'm'
        v.long_name = 'x coordinate of projection'
        v.standard_name = 'projection_x_coordinate'
        v[:] = x

        v = nc.createVariable('y', 'f4', ('y',), zlib=True)
        v.units = 'm'
        v.long_name = 'y coordinate of projection'
        v.standard_name = 'projection_y_coordinate'
        v[:] = y

        v = nc.createVariable('longitude', 'f4', ('y', 'x'), zlib=True)
        v.units = 'degrees_east'
        v.long_name = 'longitude coordinate'
        v.standard_name = 'longitude'
        v[:] = lon

        v = nc.createVariable('latitude', 'f4', ('y', 'x'), zlib=True)
        v.units = 'degrees_north'
        v.long_name = 'latitude coordinate'
        v.standard_name = 'latitude'
        v[:] = lat

        return nc

    def write_monthly_climate_file(self, time, prcp, temp, grad, ref_pix_hgt,
                                   ref_pix_lon, ref_pix_lat,
                                   time_unit='days since 1801-01-01 00:00:00',
                                   file_name='climate_monthly',
                                   filesuffix=''):
        """Creates a netCDF4 file with climate data.

        See :py:func:`~oggm.tasks.process_cru_data`.
        """

        # overwrite as default
        fpath = self.get_filepath(file_name, filesuffix=filesuffix)
        if os.path.exists(fpath):
            os.remove(fpath)

        with netCDF4.Dataset(fpath, 'w', format='NETCDF4') as nc:
            nc.ref_hgt = ref_pix_hgt
            nc.ref_pix_lon = ref_pix_lon
            nc.ref_pix_lat = ref_pix_lat
            nc.ref_pix_dis = haversine(self.cenlon, self.cenlat,
                                       ref_pix_lon, ref_pix_lat)

            dtime = nc.createDimension('time', None)

            nc.author = 'OGGM'
            nc.author_info = 'Open Global Glacier Model'

            timev = nc.createVariable('time','i4',('time',))
            timev.setncatts({'units':time_unit})
            timev[:] = netCDF4.date2num([t for t in time],
                                        time_unit)

            v = nc.createVariable('prcp', 'f4', ('time',), zlib=True)
            v.units = 'kg m-2'
            v.long_name = 'total monthly precipitation amount'
            v[:] = prcp

            v = nc.createVariable('temp', 'f4', ('time',), zlib=True)
            v.units = 'degC'
            v.long_name = '2m temperature at height ref_hgt'
            v[:] = temp

            v = nc.createVariable('grad', 'f4', ('time',), zlib=True)
            v.units = 'degC m-1'
            v.long_name = 'temperature gradient'
            v[:] = grad

    def get_inversion_flowline_hw(self, div_id=None):
        """ Shortcut function to read the heights and widths of the glacier.

        Parameters
        ----------
        div_id : int
            the divide you want the data for. Default to use all divides.

        Returns
        -------
        (height, widths) in units of m
        """

        h = np.array([])
        w = np.array([])

        if div_id is None:
            div_id = self.divide_ids

        for div_id in np.atleast_1d(div_id):
            fls = self.read_pickle('inversion_flowlines', div_id=div_id)
            for fl in fls:
                w = np.append(w, fl.widths)
                h = np.append(h, fl.surface_h)
        return h, w * fl.dx * self.grid.dx

    def get_ref_mb_data(self):
        """Get the reference mb data from WGMS (for some glaciers only!)."""

        if self._mbdf is None:
            flink, mbdatadir = get_wgms_files()
            flink = pd.read_csv(flink)
            wid = flink.loc[flink[self.rgi_version +'_ID'] == self.rgi_id]
            if len(wid) == 0:
                raise RuntimeError('Not a reference glacier!')
            wid = wid.WGMS_ID.values[0]

            # file
            reff = os.path.join(mbdatadir,
                                'mbdata_WGMS-{:05d}.csv'.format(wid))
            # list of years
            self._mbdf = pd.read_csv(reff).set_index('YEAR')

        # logic for period
        y0, y1 = cfg.PARAMS['run_period']
        ci = self.read_pickle('climate_info')
        y0 = y0 or ci['hydro_yr_0']
        y1 = y1 or ci['hydro_yr_1']
        out = self._mbdf.loc[y0:y1]
        return out.dropna(subset=['ANNUAL_BALANCE'])

    def log(self, func, err=None):
        """Logs a message to the glacier directory.

        It is usually called by the :py:class:`entity_task` decorator, normally
        you shouldn't take care about that.

        Parameters
        ----------
        func : a function
            the function which wants to log
        err : Exception
            the exception which has been raised by func (if no exception was
            raised, a success is logged)
        """

        # a line per function call
        nowsrt = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        line = nowsrt + ';' + func.__name__ + ';'
        if err is None:
            line += 'SUCCESS'
        else:
            line += err.__class__.__name__ + ': {}'.format(err)
        with open(self.logfile, 'a') as logfile:
            logfile.write(line + '\n')

    def get_task_status(self, func):
        """Opens this directory's log file to see if a task was already run.

        It is usually called by the :py:class:`entity_task` decorator, normally
        you shouldn't take care about that.

        Parameters
        ----------
        func : a function
            the tasks which wants to know

        Returns
        -------
        The last message for this task (SUCCESS if was successful),
        None if the task was not run yet
        """

        if not os.path.isfile(self.logfile):
            return None

        with open(self.logfile) as logfile:
            lines = logfile.readlines()

        lines = [l.replace('\n', '') for l in lines if func.__name__ in l]
        if lines:
            # keep only the last log
            return lines[-1].split(';')[-1]
        else:
            return None
