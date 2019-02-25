"""Automated data download and IO."""

# Builtins
import glob
import os
import gzip
import lzma
import bz2
import zlib
import shutil
import zipfile
import sys
import math
import logging
from functools import partial
import time
import fnmatch
from urllib.parse import urlparse

# External libs
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import transform as shp_trafo
from salem import wgs84
import rasterio
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool
import multiprocessing as mp
import requests

try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError

# Locals
import oggm.cfg as cfg
from oggm.exceptions import (InvalidParamsError, NoInternetException,
                             DownloadVerificationFailedException,
                             HttpDownloadError, HttpContentTooShortError)

# Module logger
logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))

# Github repository and commit hash/branch name/tag name on that repository
# The given commit will be downloaded from github and used as source for
# all sample data
SAMPLE_DATA_GH_REPO = 'OGGM/oggm-sample-data'
SAMPLE_DATA_COMMIT = 'cd3dc3df73d16dc734a50e72363115760153b644'

CRU_SERVER = ('https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.01/cruts'
              '.1709081022.v4.01/')

HISTALP_SERVER = 'http://www.zamg.ac.at/histalp/download/grid5m/'

GDIR_URL = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/oggm_v1.1/'
DEMO_GDIR_URL = 'https://cluster.klima.uni-bremen.de/~fmaussion/demo_gdirs/'

CMIP5_URL = 'https://cluster.klima.uni-bremen.de/~nicolas/cmip5-ng/'

_RGI_METADATA = dict()

DEM3REG = {
    'ISL': [-25., -12., 63., 67.],  # Iceland
    'SVALBARD': [9., 35.99, 75., 84.],
    'JANMAYEN': [-10., -7., 70., 72.],
    'FJ': [36., 68., 79., 90.],  # Franz Josef Land
    'FAR': [-8., -6., 61., 63.],  # Faroer
    'BEAR': [18., 20., 74., 75.],  # Bear Island
    'SHL': [-3., 0., 60., 61.],  # Shetland
    # Antarctica tiles as UTM zones, large files
    '01-15': [-180., -91., -90, -60.],
    '16-30': [-91., -1., -90., -60.],
    '31-45': [-1., 89., -90., -60.],
    '46-60': [89., 189., -90., -60.],
    # Greenland tiles
    # These are creating problems!
    # 'GL-North': [-78., -11., 75., 84.],
    # 'GL-West': [-68., -42., 64., 76.],
    # 'GL-South': [-52., -40., 59., 64.],
    # 'GL-East': [-42., -17., 64., 76.]
}

# Function
tuple2int = partial(np.array, dtype=np.int64)

# Global Lock
lock = mp.Lock()

# Download verification dictionary
_dl_verify_data = None


def mkdir(path, reset=False):
    """Checks if directory exists and if not, create one.

    Parameters
    ----------
    reset: erase the content of the directory if exists

    Returns
    -------
    the path
    """

    if reset and os.path.exists(path):
        shutil.rmtree(path)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path


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


def _get_download_lock():
    return lock


def get_dl_verify_data():
    """Returns a dictionary with all known download object hashes.

    The returned dictionary resolves str: cache_obj_name
    to a tuple (int: size, int: crc32).
    """
    global _dl_verify_data

    if _dl_verify_data is None:
        data = dict()
        path = os.path.join(os.path.dirname(__file__), '_downloads.crc32.xz')
        with lzma.open(path, 'rb') as f:
            for line in f:
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                elems = line.split(' ', 2)
                assert len(elems[0]) == 8
                data[elems[2]] = (int(elems[1]), int(elems[0], 16))
        _dl_verify_data = data

    return _dl_verify_data


def _call_dl_func(dl_func, cache_path):
    """Helper so the actual call to downloads can be overridden
    """
    return dl_func(cache_path)


def _cached_download_helper(cache_obj_name, dl_func, reset=False):
    """Helper function for downloads.

    Takes care of checking if the file is already cached.
    Only calls the actual download function when no cached version exists.
    """
    cache_dir = cfg.PATHS['dl_cache_dir']
    cache_ro = cfg.PARAMS['dl_cache_readonly']
    try:
        # this is for real runs
        fb_cache_dir = os.path.join(cfg.PATHS['working_dir'], 'cache')
        check_fb_dir = False
    except KeyError:
        # Nothing have been set up yet, this is bad - use tmp
        # This should happen on RO cluster only but still
        fb_cache_dir = os.path.join(cfg.CACHE_DIR, 'cache')
        check_fb_dir = True

    if not cache_dir:
        # Defaults to working directory: it must be set!
        if not cfg.PATHS['working_dir']:
            raise InvalidParamsError("Need a valid PATHS['working_dir']!")
        cache_dir = fb_cache_dir
        cache_ro = False

    fb_path = os.path.join(fb_cache_dir, cache_obj_name)
    if not reset and os.path.isfile(fb_path):
        return fb_path

    cache_path = os.path.join(cache_dir, cache_obj_name)
    if not reset and os.path.isfile(cache_path):
        return cache_path

    if cache_ro:
        if check_fb_dir:
            # Add a manual check that we are caching sample data download
            if 'oggm-sample-data' not in fb_path:
                raise InvalidParamsError('Attempting to download something '
                                         'with invalid global settings.')
        cache_path = fb_path

    if not cfg.PARAMS['has_internet']:
        raise NoInternetException("Download required, but "
                                  "`has_internet` is False.")

    mkdir(os.path.dirname(cache_path))

    try:
        cache_path = _call_dl_func(dl_func, cache_path)
    except BaseException:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        raise

    return cache_path


def _verified_download_helper(cache_obj_name, dl_func, reset=False):
    """Helper function for downloads.

    Verifies the size and hash of the downloaded file against the included
    list of known static files.
    Uses _cached_download_helper to perform the actual download.
    """
    path = _cached_download_helper(cache_obj_name, dl_func, reset)

    try:
        dl_verify = cfg.PARAMS['dl_verify']
    except KeyError:
        dl_verify = True

    if dl_verify:
        data = get_dl_verify_data()
        crc32 = 0
        with open(path, 'rb') as f:
            while True:
                b = f.read(0xFFFF)
                if not b:
                    break
                crc32 = zlib.crc32(b, crc32)
        size = os.path.getsize(path)

        if cache_obj_name not in data:
            logger.warning('No known hash for %s: %s %s' %
                           (path, size, crc32.to_bytes(4, byteorder='big').hex()))
        else:
            data = data[cache_obj_name]
            if data[0] != size or data[1] != crc32:
                err = '%s failed to verify!\nis: %s %s\nexpected: %s %s' % (
                    path,
                    size, crc32.to_bytes(4, byteorder='big').hex(),
                    data[0], data[1].to_bytes(4, byteorder='big').hex())
                raise DownloadVerificationFailedException(msg=err, path=path)
            logger.info('%s verified successfully.' % path)

    return path


def _requests_urlretrieve(url, path, reporthook):
    """Implements the required features of urlretrieve on top of requests
    """

    chunk_size = 128 * 1024
    chunk_count = 0

    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            raise HttpDownloadError(r.status_code)
        r.raise_for_status()

        size = r.headers.get('content-length') or 0
        size = int(size)

        if reporthook:
            reporthook(chunk_count, chunk_size, size)

        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                chunk_count += 1
                if reporthook:
                    reporthook(chunk_count, chunk_size, size)

        if chunk_count * chunk_size < size:
            raise HttpContentTooShortError()


def oggm_urlretrieve(url, cache_obj_name=None, reset=False, reporthook=None):
    """Wrapper around urlretrieve, to implement our caching logic.

    Instead of accepting a destination path, it decided where to store the file
    and returns the local path.
    """

    if cache_obj_name is None:
        cache_obj_name = urlparse(url)
        cache_obj_name = cache_obj_name.netloc + cache_obj_name.path

    def _dlf(cache_path):
        logger.info("Downloading %s to %s..." % (url, cache_path))
        _requests_urlretrieve(url, cache_path, reporthook)
        return cache_path

    return _verified_download_helper(cache_obj_name, _dlf, reset)


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
        res = oggm_urlretrieve(url, cache_obj_name=cache_name, reset=reset,
                               reporthook=_upd)
        try:
            pbar[0].finish()
        except BaseException:
            pass
        return res
    except (ImportError, ModuleNotFoundError):
        return oggm_urlretrieve(url, cache_obj_name=cache_name, reset=reset)


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

    return _verified_download_helper(cache_obj_name, _dlf, reset)


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
        except HttpDownloadError as err:
            # This works well for py3
            if err.code == 404 or err.code == 300:
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
        except HttpContentTooShortError as err:
            logger.info("Downloading %s failed with ContentTooShortError"
                        " error %s, retrying in 10 seconds... %s/%s" %
                        (www_path, err.code, retry_counter, retry_max))
            time.sleep(10)
            continue
        except DownloadVerificationFailedException as err:
            try:
                os.remove(err.path)
            except FileNotFoundError:
                pass
            logger.info("Downloading %s failed with "
                        "DownloadVerificationFailedException\n %s\n"
                        "The file might have changed or is corrupted. "
                        "File deleted. Re-downloading... %s/%s" %
                        (www_path, err.msg, retry_counter, retry_max))
            continue

    # See if we managed (fail is allowed)
    if not local_path or not os.path.exists(local_path):
        logger.warning('Downloading %s failed.' % www_path)

    return local_path


def download_oggm_files():
    with _get_download_lock():
        return _download_oggm_files_unlocked()


def _download_oggm_files_unlocked():
    """Checks if the demo data is already on the cache and downloads it."""

    zip_url = 'https://github.com/%s/archive/%s.zip' % \
              (SAMPLE_DATA_GH_REPO, SAMPLE_DATA_COMMIT)
    odir = os.path.join(cfg.CACHE_DIR)
    sdir = os.path.join(cfg.CACHE_DIR,
                        'oggm-sample-data-%s' % SAMPLE_DATA_COMMIT)

    # download only if necessary
    if not os.path.exists(sdir):
        ofile = file_downloader(zip_url)
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(odir)
        assert os.path.isdir(sdir)

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
    wwwfile = ('http://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/'
               'TIFF/srtm_' + zone + '.zip')
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


def _download_tandem_file(zone):
    with _get_download_lock():
        return _download_tandem_file_unlocked(zone)


def _download_tandem_file_unlocked(zone):
    """Checks if the tandem data is in the directory and if not, download it.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    bname = zone.split('/')[-1] + '_DEM.tif'
    outpath = os.path.join(tmpdir, bname)

    # check if extracted file exists already
    if os.path.exists(outpath):
        return outpath

    # Did we download it yet?
    scpfile = ('/home/data/download/tandemx-90m.dlr.de/90mdem/DEM/'
               '{}.zip'.format(zone))
    cache_dir = cfg.PATHS['dl_cache_dir']
    dest_file = os.path.join(cache_dir, 'scp_tandem', zone + '.zip')
    if not os.path.exists(dest_file):
        mkdir(os.path.dirname(dest_file))
        cmd = 'scp bremen:%s %s' % (scpfile, dest_file)
        os.system(cmd)

    # That means we tried hard but we couldn't find it
    if not os.path.exists(dest_file):
        return None

    # ok we have to extract it
    if not os.path.exists(outpath):
        with zipfile.ZipFile(dest_file) as zf:
            for fn in zf.namelist():
                if 'DEM/' + bname in fn:
                    break
            with open(outpath, 'wb') as fo:
                fo.write(zf.read(fn))

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
    if len(zone) == 4 and zone.startswith('S'):
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


def _download_topo_file_from_cluster(fname):
    with _get_download_lock():
        return _download_topo_file_from_cluster_unlocked(fname)


def _download_topo_file_from_cluster_unlocked(fname):
    """Checks if the special topo data is in the directory and if not,
    download it from the cluster.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    outpath = os.path.join(tmpdir, fname)

    url = 'https://cluster.klima.uni-bremen.de/data/dems/'
    url += fname + '.zip'
    dfile = file_downloader(url)

    if not os.path.exists(outpath):
        logger.info('Extracting ' + fname + '.zip to ' + outpath + '...')
        with zipfile.ZipFile(dfile) as zf:
            zf.extractall(tmpdir)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(outpath)
    cfg.get_lru_handler(tmpdir).append(outpath)
    return outpath


def _download_arcticdem_from_cluster():
    with _get_download_lock():
        return _download_arcticdem_from_cluster_unlocked()


def _download_arcticdem_from_cluster_unlocked():
    """Checks if the special topo data is in the directory and if not,
    download it from the cluster.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    # mkdir(tmpdir)
    fname = 'arcticdem_mosaic_100m_v3.0.tif'
    outpath = os.path.join(tmpdir, fname)

    url = 'https://cluster.klima.uni-bremen.de/data/dems/'
    url += fname
    dfile = file_downloader(url)

    if not os.path.exists(outpath):
        shutil.copyfile(dfile, outpath)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(outpath)
    cfg.get_lru_handler(tmpdir).append(outpath)
    return outpath


def _get_centerline_lonlat(gdir):
    """Quick n dirty solution to write the centerlines as a shapefile"""

    cls = gdir.read_pickle('centerlines')
    olist = []
    for j, cl in enumerate(cls[::-1]):
        mm = 1 if j == 0 else 0
        gs = gpd.GeoSeries()
        gs['RGIID'] = gdir.rgi_id
        gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
        gs['MAIN'] = mm
        tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
        gs['geometry'] = shp_trafo(tra_func, cl.line)
        olist.append(gs)

    return olist


def get_prepro_gdir(rgi_version, rgi_id, border, prepro_level, demo_url=False):
    with _get_download_lock():
        return _get_prepro_gdir_unlocked(rgi_version, rgi_id, border,
                                         prepro_level, demo_url)


def _get_prepro_gdir_unlocked(rgi_version, rgi_id, border, prepro_level,
                              demo_url=False):
    # Prepro URL
    url = DEMO_GDIR_URL if demo_url else GDIR_URL
    url += 'RGI{}/'.format(rgi_version)
    url += 'b_{:03d}/'.format(border)
    url += 'L{:d}/'.format(prepro_level)
    url += '{}/{}.tar' .format(rgi_id[:8], rgi_id[:11])

    tar_base = file_downloader(url)
    if tar_base is None:
        raise RuntimeError('Could not find file at ' + url)

    return tar_base


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


def _tandem_path(lon_tile, lat_tile):

    # OK we have a proper tile now

    # First folder level is sorted from S to N
    level_0 = 'S' if lat_tile < 0 else 'N'
    level_0 += '{:02d}'.format(abs(lat_tile))

    # Second folder level is sorted from W to E, but in 10 steps
    level_1 = 'W' if lon_tile < 0 else 'E'
    level_1 += '{:03d}'.format(divmod(abs(lon_tile), 10)[0] * 10)

    # Level 2 is formating, but depends on lat
    level_2 = 'W' if lon_tile < 0 else 'E'
    if abs(lat_tile) <= 60:
        level_2 += '{:03d}'.format(abs(lon_tile))
    elif abs(lat_tile) <= 80:
        level_2 += '{:03d}'.format(divmod(abs(lon_tile), 2)[0] * 2)
    else:
        level_2 += '{:03d}'.format(divmod(abs(lon_tile), 4)[0] * 4)

    # Final path
    out = (level_0 + '/' + level_1 + '/' +
           'TDM1_DEM__30_{}{}'.format(level_0, level_2))
    return out


def tandem_zone(lon_ex, lat_ex):
    """Returns a list of TanDEM-X zones covering the desired extent.
    """

    # Files are one by one tiles, so lets loop over them
    # For higher lats they are stored in steps of 2 and 4. My code below
    # is probably giving more files than needed but better safe than sorry
    lat_tiles = np.arange(np.floor(lat_ex[0]), np.ceil(lat_ex[1]+1e-9),
                          dtype=np.int)
    zones = []
    for lat in lat_tiles:
        if abs(lat) < 60:
            l0 = np.floor(lon_ex[0])
            l1 = np.floor(lon_ex[1])
        elif abs(lat) < 80:
            l0 = divmod(lon_ex[0], 2)[0] * 2
            l1 = divmod(lon_ex[1], 2)[0] * 2
        elif abs(lat) < 90:
            l0 = divmod(lon_ex[0], 4)[0] * 4
            l1 = divmod(lon_ex[1], 4)[0] * 4
        lon_tiles = np.arange(l0, l1+1, dtype=np.int)
        for lon in lon_tiles:
            zones.append(_tandem_path(lon, lat))
    return list(sorted(set(zones)))


def arcticdem_zone(lon_ex, lat_ex):
    """Returns a list of Arctic-DEM zones covering the desired extent.
    """

    # Files are one by one tiles, so lets loop over them

    lon_tiles = np.arange(np.floor(lon_ex[0]), np.ceil(lon_ex[1]+1e-9),
                          dtype=np.int)
    lat_tiles = np.arange(np.floor(lat_ex[0]), np.ceil(lat_ex[1]+1e-9),
                          dtype=np.int)
    zones = []
    for lon in lon_tiles:
        for lat in lat_tiles:
            zones.append(_tandem_path(lon, lat))
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
    lon_ex = np.linspace(mi, ma, int(np.ceil((ma - mi) / srtm_dy) + 3))
    mi, ma = np.min(lat_ex), np.max(lat_ex)
    # int() to avoid Deprec warning
    lat_ex = np.linspace(mi, ma, int(np.ceil((ma - mi) / srtm_dx) + 3))

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
    """Returns the path to the desired OGGM-sample-file.

    If Sample data is not cached it will be downloaded from
    https://github.com/OGGM/oggm-sample-data

    Parameters
    ----------
    fname : str
        Filename of the desired OGGM-sample-file

    Returns
    -------
    str
        Absolute path to the desired file.
    """

    d = download_oggm_files()
    if fname in d:
        return d[fname]
    else:
        return None


def get_cru_cl_file():
    """Returns the path to the unpacked CRU CL file (is in sample data)."""

    download_oggm_files()

    sdir = os.path.join(cfg.CACHE_DIR,
                        'oggm-sample-data-%s' % SAMPLE_DATA_COMMIT,
                        'cru')
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
    (file, dir) : paths to the files
    """

    download_oggm_files()
    sdir = os.path.join(cfg.CACHE_DIR,
                        'oggm-sample-data-%s' % SAMPLE_DATA_COMMIT,
                        'wgms')
    datadir = os.path.join(sdir, 'mbdata')
    assert os.path.exists(datadir)

    outf = os.path.join(sdir, 'rgi_wgms_links_20171101.csv')
    outf = pd.read_csv(outf, dtype={'RGI_REG': object})

    return outf, datadir


def get_glathida_file():
    """Get the path to the default GlaThiDa-RGI link file.

    Returns
    -------
    file : paths to the file
    """

    # Roll our own
    download_oggm_files()
    sdir = os.path.join(cfg.CACHE_DIR,
                        'oggm-sample-data-%s' % SAMPLE_DATA_COMMIT,
                        'glathida')
    outf = os.path.join(sdir, 'rgi_glathida_links.csv')
    assert os.path.exists(outf)
    return outf


def get_rgi_dir(version=None, reset=False):
    """Path to the RGI directory.

    If the RGI files are not present, download them.

    Parameters
    ----------
    version : str
        '5', '6', defaults to None (linking to the one specified in cfg.PARAMS)
    reset : bool
        If True, deletes the RGI directory first and downloads the data

    Returns
    -------
    str
        path to the RGI directory
    """

    with _get_download_lock():
        return _get_rgi_dir_unlocked(version=version, reset=reset)


def _get_rgi_dir_unlocked(version=None, reset=False):

    rgi_dir = cfg.PATHS['rgi_dir']
    if version is None:
        version = cfg.PARAMS['rgi_version']

    if len(version) == 1:
        version += '0'

    # Be sure the user gave a sensible path to the RGI dir
    if not rgi_dir:
        raise InvalidParamsError('The RGI data directory has to be'
                                 'specified explicitly.')
    rgi_dir = os.path.abspath(os.path.expanduser(rgi_dir))
    rgi_dir = os.path.join(rgi_dir, 'RGIV' + version)
    mkdir(rgi_dir, reset=reset)

    if version == '50':
        dfile = 'http://www.glims.org/RGI/rgi50_files/rgi50.zip'
    elif version == '60':
        dfile = 'http://www.glims.org/RGI/rgi60_files/00_rgi60.zip'
    elif version == '61':
        dfile = 'https://cluster.klima.uni-bremen.de/data/rgi/rgi_61.zip'

    test_file = os.path.join(rgi_dir,
                             '*_rgi*{}_manifest.txt'.format(version))

    if len(glob.glob(test_file)) == 0:
        # if not there download it
        ofile = file_downloader(dfile, reset=reset)
        # Extract root
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(rgi_dir)
        # Extract subdirs
        pattern = '*_rgi{}_*.zip'.format(version)
        for root, dirs, files in os.walk(cfg.PATHS['rgi_dir']):
            for filename in fnmatch.filter(files, pattern):
                zfile = os.path.join(root, filename)
                with zipfile.ZipFile(zfile) as zf:
                    ex_root = zfile.replace('.zip', '')
                    mkdir(ex_root)
                    zf.extractall(ex_root)
                # delete the zipfile after success
                os.remove(zfile)
        if len(glob.glob(test_file)) == 0:
            raise RuntimeError('Could not find a manifest file in the RGI '
                               'directory: ' + rgi_dir)
    return rgi_dir


def get_rgi_region_file(region, version=None, reset=False):
    """Path to the RGI region file.

    If the RGI files are not present, download them.

    Parameters
    ----------
    region : str
        from '01' to '19'
    version : str
        '5', '6', defaults to None (linking to the one specified in cfg.PARAMS)
    reset : bool
        If True, deletes the RGI directory first and downloads the data

    Returns
    -------
    str
        path to the RGI shapefile
    """

    rgi_dir = get_rgi_dir(version=version, reset=reset)
    f = list(glob.glob(rgi_dir + "/*/{}_*.shp".format(region)))
    assert len(f) == 1
    return f[0]


def get_rgi_glacier_entities(rgi_ids, version=None):
    """Get a list of glacier outlines selected from their RGI IDs.

    Will download RGI data if not present.

    Parameters
    ----------
    rgi_ids : list of str
        the glaciers you want the outlines for
    version : str
        the rgi version

    Returns
    -------
    geopandas.GeoDataFrame
        containing the desired RGI glacier outlines
    """

    regions = [s.split('-')[1].split('.')[0] for s in rgi_ids]
    if version is None:
        version = rgi_ids[0].split('-')[0][-2:]
    selection = []
    for reg in sorted(np.unique(regions)):
        sh = gpd.read_file(get_rgi_region_file(reg, version=version))
        selection.append(sh.loc[sh.RGIId.isin(rgi_ids)])

    # Make a new dataframe of those
    selection = pd.concat(selection)
    selection.crs = sh.crs  # for geolocalisation
    if len(selection) != len(rgi_ids):
        raise RuntimeError('Could not find all RGI ids')

    return selection


def get_rgi_intersects_dir(version=None, reset=False):
    """Path to the RGI directory containing the intersect files.

    If the files are not present, download them.

    Parameters
    ----------
    version : str
        '5', '6', defaults to None (linking to the one specified in cfg.PARAMS)
    reset : bool
        If True, deletes the intersects before redownloading them

    Returns
    -------
    str
        path to the directory
    """

    with _get_download_lock():
        return _get_rgi_intersects_dir_unlocked(version=version, reset=reset)


def _get_rgi_intersects_dir_unlocked(version=None, reset=False):

    rgi_dir = cfg.PATHS['rgi_dir']
    if version is None:
        version = cfg.PARAMS['rgi_version']

    if len(version) == 1:
        version += '0'

    # Be sure the user gave a sensible path to the RGI dir
    if not rgi_dir:
        raise InvalidParamsError('The RGI data directory has to be'
                                 'specified explicitly.')

    rgi_dir = os.path.abspath(os.path.expanduser(rgi_dir))
    mkdir(rgi_dir)

    dfile = 'https://cluster.klima.uni-bremen.de/data/rgi/'
    dfile += 'RGI_V{}_Intersects.zip'.format(version)

    odir = os.path.join(rgi_dir, 'RGI_V' + version + '_Intersects')
    if reset and os.path.exists(odir):
        shutil.rmtree(odir)

    # A lot of code for backwards compat (sigh...)
    if version in ['50', '60']:
        test_file = os.path.join(odir, 'Intersects_OGGM_Manifest.txt')
        if not os.path.exists(test_file):
            # if not there download it
            ofile = file_downloader(dfile, reset=reset)
            # Extract root
            with zipfile.ZipFile(ofile) as zf:
                zf.extractall(odir)
            if not os.path.exists(test_file):
                raise RuntimeError('Could not find a manifest file in the RGI '
                                   'directory: ' + odir)
    else:
        test_file = os.path.join(odir,
                                 '*ntersect*anifest.txt'.format(version))
        if len(glob.glob(test_file)) == 0:
            # if not there download it
            ofile = file_downloader(dfile, reset=reset)
            # Extract root
            with zipfile.ZipFile(ofile) as zf:
                zf.extractall(odir)
            # Extract subdirs
            pattern = '*_rgi{}_*.zip'.format(version)
            for root, dirs, files in os.walk(cfg.PATHS['rgi_dir']):
                for filename in fnmatch.filter(files, pattern):
                    zfile = os.path.join(root, filename)
                    with zipfile.ZipFile(zfile) as zf:
                        ex_root = zfile.replace('.zip', '')
                        mkdir(ex_root)
                        zf.extractall(ex_root)
                    # delete the zipfile after success
                    os.remove(zfile)
            if len(glob.glob(test_file)) == 0:
                raise RuntimeError('Could not find a manifest file in the RGI '
                                   'directory: ' + odir)

    return odir


def get_rgi_intersects_region_file(region=None, version=None, reset=False):
    """Path to the RGI regional intersect file.

    If the RGI files are not present, download them.

    Parameters
    ----------
    region : str
        from '00' to '19', with '00' being the global file (deprecated).
        From RGI version '61' onwards, please use `get_rgi_intersects_entities`
        with a list of glaciers instead of relying to the global file.
    version : str
        '5', '6', '61'... defaults the one specified in cfg.PARAMS
    reset : bool
        If True, deletes the intersect file before redownloading it

    Returns
    -------
    str
        path to the RGI intersects shapefile
    """

    if version is None:
        version = cfg.PARAMS['rgi_version']
    if len(version) == 1:
        version += '0'

    rgi_dir = get_rgi_intersects_dir(version=version, reset=reset)

    if region == '00':
        if version in ['50', '60']:
            version = 'AllRegs'
            region = '*'
        else:
            raise InvalidParamsError("From RGI version 61 onwards, please use "
                                     "get_rgi_intersects_entities() instead.")
    f = list(glob.glob(os.path.join(rgi_dir, "*", '*intersects*' + region +
                                    '_rgi*' + version + '*.shp')))
    assert len(f) == 1
    return f[0]


def get_rgi_intersects_entities(rgi_ids, version=None):
    """Get a list of glacier intersects selected from their RGI IDs.

    Parameters
    ----------
    rgi_ids: list of str
        list of rgi_ids you want to look for intersections for
    version: str
        '5', '6', '61'... defaults the one specified in cfg.PARAMS

    Returns
    -------
    geopandas.GeoDataFrame
        with the selected intersects
    """

    if version is None:
        version = cfg.PARAMS['rgi_version']
    if len(version) == 1:
        version += '0'

    regions = [s.split('-')[1].split('.')[0] for s in rgi_ids]
    selection = []
    for reg in sorted(np.unique(regions)):
        sh = gpd.read_file(get_rgi_intersects_region_file(reg,
                                                          version=version))
        selection.append(sh.loc[sh.RGIId_1.isin(rgi_ids) |
                                sh.RGIId_2.isin(rgi_ids)])

    # Make a new dataframe of those
    selection = pd.concat(selection)
    selection.crs = sh.crs  # for geolocalisation

    return selection


def get_cru_file(var=None):
    """Returns a path to the desired CRU baseline climate file.

    If the file is not present, download it.

    Parameters
    ----------
    var : str
        'tmp' for temperature
        'pre' for precipitation

    Returns
    -------
    str
        path to the CRU file
    """
    with _get_download_lock():
        return _get_cru_file_unlocked(var)


def _get_cru_file_unlocked(var=None):

    cru_dir = cfg.PATHS['cru_dir']

    # Be sure the user gave a sensible path to the climate dir
    if not cru_dir:
        raise InvalidParamsError('The CRU data directory has to be'
                                 'specified explicitly.')
    cru_dir = os.path.abspath(os.path.expanduser(cru_dir))
    mkdir(cru_dir)

    # Be sure input makes sense
    if var not in ['tmp', 'pre']:
        raise InvalidParamsError('CRU variable {} does not exist!'.format(var))

    # The user files may have different dates, so search for patterns
    bname = 'cru_ts*.{}.dat.nc'.format(var)
    search = glob.glob(os.path.join(cru_dir, bname))
    if len(search) == 1:
        ofile = search[0]
    elif len(search) > 1:
        raise RuntimeError('You seem to have more than one file in your CRU '
                           'directory: {}. Help me by deleting the one'
                           'you dont want to use anymore.'.format(cru_dir))
    else:
        # if not there download it
        cru_filename = 'cru_ts4.01.1901.2016.{}.dat.nc'.format(var)
        cru_url = CRU_SERVER + '{}/'.format(var) + cru_filename + '.gz'
        dlfile = file_downloader(cru_url)
        ofile = os.path.join(cru_dir, cru_filename)
        with gzip.GzipFile(dlfile) as zf:
            with open(ofile, 'wb') as outfile:
                for line in zf:
                    outfile.write(line)
    return ofile


def get_histalp_file(var=None):
    """Returns a path to the desired HISTALP baseline climate file.

    If the file is not present, download it.

    Parameters
    ----------
    var : str
        'tmp' for temperature
        'pre' for precipitation

    Returns
    -------
    str
        path to the CRU file
    """
    with _get_download_lock():
        return _get_histalp_file_unlocked(var)


def _get_histalp_file_unlocked(var=None):

    cru_dir = cfg.PATHS['cru_dir']

    # Be sure the user gave a sensible path to the climate dir
    if not cru_dir:
        raise InvalidParamsError('The CRU data directory has to be'
                                 'specified explicitly.')
    cru_dir = os.path.abspath(os.path.expanduser(cru_dir))
    mkdir(cru_dir)

    # Be sure input makes sense
    if var not in ['tmp', 'pre']:
        raise InvalidParamsError('HISTALP variable {} '
                                 'does not exist!'.format(var))

    # File to look for
    if var == 'tmp':
        bname = 'HISTALP_temperature_1780-2014.nc'
    else:
        bname = 'HISTALP_precipitation_all_abs_1801-2014.nc'

    search = glob.glob(os.path.join(cru_dir, bname))
    if len(search) == 1:
        ofile = search[0]
    elif len(search) > 1:
        raise RuntimeError('You seem to have more than one matching file in '
                           'your CRU directory: {}. Help me by deleting the '
                           'one you dont want to use anymore.'.format(cru_dir))
    else:
        # if not there download it
        h_url = HISTALP_SERVER + bname + '.bz2'
        dlfile = file_downloader(h_url)
        ofile = os.path.join(cru_dir, bname)
        with bz2.BZ2File(dlfile) as zf:
            with open(ofile, 'wb') as outfile:
                for line in zf:
                    outfile.write(line)
    return ofile


def get_topo_file(lon_ex, lat_ex, rgi_region=None, rgi_subregion=None,
                  source=None):
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
    rgi_region : str, optional
        the RGI region number (required for the GIMP DEM)
    rgi_subregion : str, optional
        the RGI subregion str (useful for RGI Reg 19)
    source : str or list of str, optional
        If you want to force the use of a certain DEM source. Available are:
          - 'USER' : file set in cfg.PATHS['dem_file']
          - 'SRTM' : SRTM v4.1
          - 'GIMP' : https://bpcrc.osu.edu/gdg/data/gimpdem
          - 'RAMP' : http://nsidc.org/data/docs/daac/nsidc0082_ramp_dem.gd.html
          - 'DEM3' : http://viewfinderpanoramas.org/
          - 'ASTER' : ASTER data
          - 'TANDEM' : https://geoservice.dlr.de/web/dataguide/tdm90/
          - 'ARCTICDEM' : https://www.pgc.umn.edu/data/arcticdem/

    Returns
    -------
    tuple: (list with path(s) to the DEM file, data source)
    """

    if source is not None and not isinstance(source, str):
        # check all user options
        for s in source:
            demf, source_str = get_topo_file(lon_ex, lat_ex,
                                             rgi_region=rgi_region,
                                             rgi_subregion=rgi_subregion,
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
            _file = _download_topo_file_from_cluster('gimpdem_90m_v01.1.tif')
            return [_file], source

    # ArcticDEM also has a polar projection, but here we have a shape
    if source == 'ARCTICDEM':
        source = 'ARCTICDEM' if source is None else source
        # If we have to automatise this one day, we should use the shapefile
        # of the tiles, and then check for RGI region:
        # use_without_check = ['03', '05', '06', '07', '09']
        # to_test_on_shape = ['01', '02', '04', '08']
        if source == 'ARCTICDEM':
            _file = _download_arcticdem_from_cluster()
            return [_file], source

    # Same for Antarctica
    if source == 'RAMP' or (rgi_region is not None and int(rgi_region) == 19):
        if rgi_subregion is None:
            raise InvalidParamsError('Must specify subregion for Antarctica')
        else:
            dem3_regs = ['19-01', '19-02', '19-03', '19-04', '19-05']
            should_dem3 = rgi_subregion in dem3_regs
        if should_dem3:
            # special case for some distant islands
            source = 'DEM3' if source is None else source
        else:
            source = 'RAMP' if source is None else source
        if source == 'RAMP':
            _file = _download_topo_file_from_cluster('AntarcticDEM_wgs84.tif')
            return [_file], source

    # TANDEM
    if source == 'TANDEM':
        zones = tandem_zone(lon_ex, lat_ex)
        sources = []
        for z in zones:
            sources.append(_download_tandem_file(z))
        source_str = source

    # Anywhere else on Earth we check for DEM3, ASTER, or SRTM
    # exceptional test for eastern russia:
    east_max = np.min(lat_ex) > 59 and np.min(lon_ex) > 170
    if (np.min(lat_ex) < -60.) or (np.max(lat_ex) > 60.) or \
            (source == 'DEM3') or (source == 'ASTER') or east_max:
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


def get_cmip5_file(filename, reset=False):
    """Download a global CMIP5 file.

    List of files: https://cluster.klima.uni-bremen.de/~nicolas/cmip5-ng/

    Parameters
    ----------
    filename : str
        the file to download, e.g 'pr_ann_ACCESS1-3_rcp85_r1i1p1_g025.nc'
        or 'tas_ann_ACCESS1-3_rcp45_r1i1p1_g025.nc'
    reset : bool
        force re-download of an existing file

    Returns
    -------
    the path to the netCDF file
    """

    prefix = filename.split('_')[0]
    dfile = CMIP5_URL + prefix + '/' + filename
    return file_downloader(dfile, reset=reset)


def get_ref_mb_glaciers(gdirs):
    """Get the list of glaciers we have valid mass balance measurements for.

    To be valid glaciers must have more than 5 years of measurements and
    be land terminating.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        list of glaciers to check for valid reference mass balance data

    Returns
    -------
    ref_gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        list of those glaciers with valid reference mass balance data
    """

    # Get the links
    flink, _ = get_wgms_files()
    dfids = flink['RGI{}0_ID'.format(gdirs[0].rgi_version[0])].values

    # We remove tidewater glaciers and glaciers with < 5 years
    ref_gdirs = []
    for g in gdirs:
        if g.rgi_id not in dfids or g.is_tidewater:
            continue
        try:
            mbdf = g.get_ref_mb_data()
            if len(mbdf) >= 5:
                ref_gdirs.append(g)
        except RuntimeError:
            pass
    return ref_gdirs
