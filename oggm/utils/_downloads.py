"""Automated data download and IO."""

# Builtins
import glob
import os
import gzip
import bz2
import hashlib
import shutil
import zipfile
import sys
import math
import logging
from functools import partial, wraps
import time
import fnmatch
import urllib.request
import urllib.error
from urllib.parse import urlparse
import socket
import multiprocessing
from netrc import netrc
import ftplib
import ssl
import tarfile

# External libs
import pandas as pd
import numpy as np
import shapely.geometry as shpg
import requests

# Optional libs
try:
    import geopandas as gpd
except ImportError:
    pass
try:
    import salem
    from salem import wgs84
except ImportError:
    pass
try:
    import rasterio
    from rasterio.merge import merge as merge_tool
except ImportError:
    pass

# Locals
import oggm.cfg as cfg
from oggm.exceptions import (InvalidParamsError, NoInternetException,
                             DownloadVerificationFailedException,
                             DownloadCredentialsMissingException,
                             HttpDownloadError, HttpContentTooShortError,
                             InvalidDEMError, FTPSDownloadError)

# Module logger
logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))

# Github repository and commit hash/branch name/tag name on that repository
# The given commit will be downloaded from github and used as source for
# all sample data
SAMPLE_DATA_GH_REPO = 'OGGM/oggm-sample-data'
SAMPLE_DATA_COMMIT = '56aa8f23e3b450e6f56a8d4e8a5cdb58e03c9cc1'

CHECKSUM_URL = 'https://cluster.klima.uni-bremen.de/data/downloads.sha256.hdf'
CHECKSUM_VALIDATION_URL = CHECKSUM_URL + '.sha256'
CHECKSUM_LIFETIME = 24 * 60 * 60

# Recommended url for runs
DEFAULT_BASE_URL = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/'
                    'L3-L5_files/2023.3/elev_bands/W5E5_spinup')

# Web mercator proj constants
WEB_N_PIX = 256
WEB_EARTH_RADUIS = 6378137.

DEM_SOURCES = ['GIMP', 'ARCTICDEM', 'RAMP', 'TANDEM', 'AW3D30', 'MAPZEN', 'DEM3',
               'ASTER', 'SRTM', 'REMA', 'ALASKA', 'COPDEM30', 'COPDEM90', 'NASADEM']
DEM_SOURCES_PER_GLACIER = None

_RGI_METADATA = dict()

DEM3REG = {
    'ISL': [-25., -13., 63., 67.],  # Iceland
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
    'GL-North': [-72., -11., 76., 84.],
    'GL-West': [-62., -42., 64., 76.],
    'GL-South': [-52., -40., 59., 64.],
    'GL-East': [-42., -17., 64., 76.]
}

# Function
tuple2int = partial(np.array, dtype=np.int64)

lock = None


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
    os.makedirs(path, exist_ok=True)
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


def findfiles(root_dir, endswith):
    """Finds all files with a specific ending in a directory

    Parameters
    ----------
    root_dir : str
       The directory to search fo
    endswith : str
       The file ending (e.g. '.hgt'

    Returns
    -------
    the list of files
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in [f for f in filenames if f.endswith(endswith)]:
            out.append(os.path.join(dirpath, filename))
    return out


def get_lock():
    """Get multiprocessing lock."""
    global lock
    if lock is None:
        # Global Lock
        if cfg.PARAMS.get('use_mp_spawn', False):
            lock = multiprocessing.get_context('spawn').Lock()
        else:
            lock = multiprocessing.Lock()
    return lock


def get_dl_verify_data(section):
    """Returns a pandas DataFrame with all known download object hashes.

    The returned dictionary resolves str: cache_obj_name (without section)
    to a tuple int(size) and bytes(sha256)
    """

    verify_key = 'dl_verify_data_' + section
    if verify_key in cfg.DATA:
        return cfg.DATA[verify_key]

    verify_file_path = os.path.join(cfg.CACHE_DIR, 'downloads.sha256.hdf')

    def verify_file(force=False):
        """Check the hash file's own hash"""
        if not cfg.PARAMS['has_internet']:
            return

        if not force and os.path.isfile(verify_file_path) and \
           os.path.getmtime(verify_file_path) + CHECKSUM_LIFETIME > time.time():
            return

        logger.info('Checking the download verification file checksum...')
        try:
            with requests.get(CHECKSUM_VALIDATION_URL) as req:
                req.raise_for_status()
                verify_file_sha256 = req.text.split(maxsplit=1)[0]
                verify_file_sha256 = bytearray.fromhex(verify_file_sha256)
        except Exception as e:
            verify_file_sha256 = None
            logger.warning('Failed getting verification checksum: ' + repr(e))

        if os.path.isfile(verify_file_path) and verify_file_sha256:
            sha256 = hashlib.sha256()
            with open(verify_file_path, 'rb') as f:
                for b in iter(lambda: f.read(0xFFFF), b''):
                    sha256.update(b)
            if sha256.digest() != verify_file_sha256:
                logger.warning('%s changed or invalid, deleting.'
                               % (verify_file_path))
                os.remove(verify_file_path)
            else:
                os.utime(verify_file_path)

    if not np.any(['dl_verify_data_' in k for k in cfg.DATA.keys()]):
        # We check the hash file only once per session
        # no need to do it at each call
        verify_file()

    if not os.path.isfile(verify_file_path):
        if not cfg.PARAMS['has_internet']:
            return pd.DataFrame()

        logger.info('Downloading %s to %s...'
                    % (CHECKSUM_URL, verify_file_path))

        with requests.get(CHECKSUM_URL, stream=True) as req:
            if req.status_code == 200:
                mkdir(os.path.dirname(verify_file_path))
                with open(verify_file_path, 'wb') as f:
                    for b in req.iter_content(chunk_size=0xFFFF):
                        if b:
                            f.write(b)

        logger.info('Done downloading.')

        verify_file(force=True)

    if not os.path.isfile(verify_file_path):
        logger.warning('Downloading and verifying checksums failed.')
        return pd.DataFrame()

    try:
        data = pd.read_hdf(verify_file_path, key=section)
    except KeyError:
        data = pd.DataFrame()

    cfg.DATA[verify_key] = data

    return data


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

    # A lot of logic below could be simplified but it's also not too important
    wd = cfg.PATHS.get('working_dir')
    if wd:
        # this is for real runs
        fb_cache_dir = os.path.join(wd, 'cache')
        check_fb_dir = False
    else:
        # Nothing have been set up yet, this is bad - find a place to write
        # This should happen on read-only cluster only but still
        wd = os.environ.get('OGGM_WORKDIR')
        if wd is not None and os.path.isdir(wd):
            fb_cache_dir = os.path.join(wd, 'cache')
        else:
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

    dl_verify = cfg.PARAMS.get('dl_verify', False)

    if dl_verify and path and cache_obj_name not in cfg.DL_VERIFIED:
        cache_section, cache_path = cache_obj_name.split('/', 1)
        data = get_dl_verify_data(cache_section)
        if cache_path not in data.index:
            logger.info('No known hash for %s' % cache_obj_name)
            cfg.DL_VERIFIED[cache_obj_name] = True
        else:
            # compute the hash
            sha256 = hashlib.sha256()
            with open(path, 'rb') as f:
                for b in iter(lambda: f.read(0xFFFF), b''):
                    sha256.update(b)
            sha256 = sha256.digest()
            size = os.path.getsize(path)

            # check
            data = data.loc[cache_path]
            if data['size'] != size or bytes(data['sha256']) != sha256:
                err = '%s failed to verify!\nis: %s %s\nexpected: %s %s' % (
                    path, size, sha256.hex(), data[0], data[1].hex())
                raise DownloadVerificationFailedException(msg=err, path=path)
            logger.info('%s verified successfully.' % path)
            cfg.DL_VERIFIED[cache_obj_name] = True

    return path


def _requests_urlretrieve(url, path, reporthook, auth=None, timeout=None):
    """Implements the required features of urlretrieve on top of requests
    """

    chunk_size = 128 * 1024
    chunk_count = 0

    with requests.get(url, stream=True, auth=auth, timeout=timeout) as r:
        if r.status_code != 200:
            raise HttpDownloadError(r.status_code, url)
        r.raise_for_status()

        size = r.headers.get('content-length') or -1
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


def _classic_urlretrieve(url, path, reporthook, auth=None, timeout=None):
    """Thin wrapper around pythons urllib urlretrieve
    """

    ourl = url
    if auth:
        u = urlparse(url)
        if '@' not in u.netloc:
            netloc = auth[0] + ':' + auth[1] + '@' + u.netloc
            url = u._replace(netloc=netloc).geturl()

    old_def_timeout = socket.getdefaulttimeout()
    if timeout is not None:
        socket.setdefaulttimeout(timeout)

    try:
        urllib.request.urlretrieve(url, path, reporthook)
    except urllib.error.HTTPError as e:
        raise HttpDownloadError(e.code, ourl)
    except urllib.error.ContentTooShortError:
        raise HttpContentTooShortError()
    finally:
        socket.setdefaulttimeout(old_def_timeout)


class ImplicitFTPTLS(ftplib.FTP_TLS):
    """ FTP_TLS subclass that automatically wraps sockets in SSL to support
        implicit FTPS.

        Taken from https://stackoverflow.com/a/36049814
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sock = None

    @property
    def sock(self):
        """Return the socket."""
        return self._sock

    @sock.setter
    def sock(self, value):
        """When modifying the socket, ensure that it is ssl wrapped."""
        if value is not None and not isinstance(value, ssl.SSLSocket):
            value = self.context.wrap_socket(value)
        self._sock = value


def url_exists(url):
    """Checks if a given a URL exists or not."""
    request = requests.get(url)
    return request.status_code < 400


def _ftps_retrieve(url, path, reporthook, auth=None, timeout=None):
    """ Wrapper around ftplib to download from FTPS server
    """

    if not auth:
        raise DownloadCredentialsMissingException('No authentication '
                                                  'credentials given!')

    upar = urlparse(url)

    # Decide if Implicit or Explicit FTPS is used based on the port in url
    if upar.port == 990:
        ftps = ImplicitFTPTLS()
    elif upar.port == 21:
        ftps = ftplib.FTP_TLS()

    try:
        # establish ssl connection
        ftps.connect(host=upar.hostname, port=upar.port, timeout=timeout)
        ftps.login(user=auth[0], passwd=auth[1])
        ftps.prot_p()

        logger.info('Established connection %s' % upar.hostname)

        # meta for progress bar size
        count = 0
        total = ftps.size(upar.path)
        bs = 12*1024

        def _ftps_progress(data):
            outfile.write(data)
            nonlocal count
            count += 1
            reporthook(count, count*bs, total)

        with open(path, 'wb') as outfile:
            ftps.retrbinary('RETR ' + upar.path, _ftps_progress, blocksize=bs)

    except (ftplib.error_perm, socket.timeout, socket.gaierror) as err:
        raise FTPSDownloadError(err)
    finally:
        ftps.close()


def _get_url_cache_name(url):
    """Returns the cache name for any given url.
    """

    res = urlparse(url)
    return res.netloc.split(':', 1)[0] + res.path


def oggm_urlretrieve(url, cache_obj_name=None, reset=False,
                     reporthook=None, auth=None, timeout=None):
    """Wrapper around urlretrieve, to implement our caching logic.

    Instead of accepting a destination path, it decided where to store the file
    and returns the local path.

    auth is expected to be either a tuple of ('username', 'password') or None.
    """

    if cache_obj_name is None:
        cache_obj_name = _get_url_cache_name(url)

    def _dlf(cache_path):
        logger.info("Downloading %s to %s..." % (url, cache_path))
        try:
            _requests_urlretrieve(url, cache_path, reporthook, auth, timeout)
        except requests.exceptions.InvalidSchema:
            if 'ftps://' in url:
                _ftps_retrieve(url, cache_path, reporthook, auth, timeout)
            else:
                _classic_urlretrieve(url, cache_path, reporthook, auth,
                                     timeout)
        return cache_path

    return _verified_download_helper(cache_obj_name, _dlf, reset)


def _progress_urlretrieve(url, cache_name=None, reset=False,
                          auth=None, timeout=None):
    """Downloads a file, returns its local path, and shows a progressbar."""

    try:
        from progressbar import DataTransferBar, UnknownLength
        pbar = None

        def _upd(count, size, total):
            nonlocal pbar
            if pbar is None:
                pbar = DataTransferBar()
                if not pbar.is_terminal:
                    pbar.min_poll_interval = 15
            if pbar.max_value is None:
                if total > 0:
                    pbar.start(total)
                else:
                    pbar.start(UnknownLength)
            pbar.update(min(count * size, total))
            sys.stdout.flush()
        res = oggm_urlretrieve(url, cache_obj_name=cache_name, reset=reset,
                               reporthook=_upd, auth=auth, timeout=timeout)
        try:
            pbar.finish()
        except BaseException:
            pass
        return res
    except ImportError:
        return oggm_urlretrieve(url, cache_obj_name=cache_name,
                                reset=reset, auth=auth, timeout=timeout)


def aws_file_download(aws_path, cache_name=None, reset=False):
    with get_lock():
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
        raise NotImplementedError("Downloads from AWS are no longer supported")

    return _verified_download_helper(cache_obj_name, _dlf, reset)


def file_downloader(www_path, retry_max=3, sleep_on_retry=5,
                    cache_name=None, reset=False, auth=None,
                    timeout=None):
    """A slightly better downloader: it tries more than once."""

    local_path = None
    retry_counter = 0
    while retry_counter < retry_max:
        # Try to download
        try:
            retry_counter += 1
            local_path = _progress_urlretrieve(www_path, cache_name=cache_name,
                                               reset=reset, auth=auth,
                                               timeout=timeout)
            # if no error, exit
            break
        except HttpDownloadError as err:
            # This works well for py3
            if err.code == 404 or err.code == 300:
                # Ok so this *should* be an ocean tile
                return None
            elif err.code >= 500 and err.code < 600:
                logger.info(f"Downloading {www_path} failed with "
                            f"HTTP error {err.code}, "
                            f"retrying in {sleep_on_retry} seconds... "
                            f"{retry_counter}/{retry_max}")
                time.sleep(sleep_on_retry)
                continue
            else:
                raise
        except HttpContentTooShortError as err:
            logger.info("Downloading %s failed with ContentTooShortError"
                        " error %s, retrying in %s seconds... %s/%s" %
                        (www_path, err.code, sleep_on_retry, retry_counter, retry_max))
            time.sleep(sleep_on_retry)
            continue
        except DownloadVerificationFailedException as err:
            if (cfg.PATHS['dl_cache_dir'] and
                  err.path.startswith(cfg.PATHS['dl_cache_dir']) and
                  cfg.PARAMS['dl_cache_readonly']):
                if not cache_name:
                    cache_name = _get_url_cache_name(www_path)
                cache_name = "GLOBAL_CACHE_INVALID/" + cache_name
                retry_counter -= 1
                logger.info("Global cache for %s is invalid!")
            else:
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
        except requests.ConnectionError as err:
            if err.args[0].__class__.__name__ == 'MaxRetryError':
                # if request tried often enough we don't have to do this
                # this error does happen for not existing ASTERv3 files
                return None
            else:
                # in other cases: try again
                logger.info("Downloading %s failed with ConnectionError, "
                            "retrying in %s seconds... %s/%s" %
                            (www_path, sleep_on_retry, retry_counter, retry_max))
                time.sleep(sleep_on_retry)
                continue
        except FTPSDownloadError as err:
            logger.info("Downloading %s failed with FTPSDownloadError"
                        " error: '%s', retrying in %s seconds... %s/%s" %
                        (www_path, err.orgerr, sleep_on_retry, retry_counter, retry_max))
            time.sleep(sleep_on_retry)
            continue

    # See if we managed (fail is allowed)
    if not local_path or not os.path.exists(local_path):
        logger.warning('Downloading %s failed.' % www_path)

    return local_path


def locked_func(func):
    """To decorate a function that needs to be locked for multiprocessing"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with get_lock():
            return func(*args, **kwargs)
    return wrapper


def file_extractor(file_path):
    """For archives with only one file inside extract the file to tmpdir."""

    filename, file_extension = os.path.splitext(file_path)
    # Second one for tar.gz files
    f2, ex2 = os.path.splitext(filename)
    if ex2 == '.tar':
        filename, file_extension = f2, '.tar.gz'
    bname = os.path.basename(file_path)

    # This is to give a unique name to the tmp file
    hid = hashlib.md5(file_path.encode()).hexdigest()[:7] + '_'

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)

    # Check output extension
    def _check_ext(f):
        _, of_ext = os.path.splitext(f)
        if of_ext not in ['.nc', '.tif']:
            raise InvalidParamsError('Extracted file extension not recognized'
                                     ': {}'.format(of_ext))
        return of_ext

    if file_extension == '.zip':
        with zipfile.ZipFile(file_path) as zf:
            members = zf.namelist()
            if len(members) != 1:
                raise RuntimeError('Cannot extract multiple files')
            o_name = hid + members[0]
            o_path = os.path.join(tmpdir, o_name)
            of_ext = _check_ext(o_path)
            if not os.path.exists(o_path):
                logger.info('Extracting {} to {}...'.format(bname, o_path))
                with open(o_path, 'wb') as f:
                    f.write(zf.read(members[0]))
    elif file_extension == '.gz':
        # Gzip files cannot be inspected. It's always only one file
        # Decide on its name
        o_name = hid + os.path.basename(filename)
        o_path = os.path.join(tmpdir, o_name)
        of_ext = _check_ext(o_path)
        if not os.path.exists(o_path):
            logger.info('Extracting {} to {}...'.format(bname, o_path))
            with gzip.GzipFile(file_path) as zf:
                with open(o_path, 'wb') as outfile:
                    for line in zf:
                        outfile.write(line)
    elif file_extension == '.bz2':
        # bzip2 files cannot be inspected. It's always only one file
        # Decide on its name
        o_name = hid + os.path.basename(filename)
        o_path = os.path.join(tmpdir, o_name)
        of_ext = _check_ext(o_path)
        if not os.path.exists(o_path):
            logger.info('Extracting {} to {}...'.format(bname, o_path))
            with bz2.open(file_path) as zf:
                with open(o_path, 'wb') as outfile:
                    for line in zf:
                        outfile.write(line)
    elif file_extension in ['.tar.gz', '.tar']:
        with tarfile.open(file_path) as zf:
            members = zf.getmembers()
            if len(members) != 1:
                raise RuntimeError('Cannot extract multiple files')
            o_name = hid + members[0].name
            o_path = os.path.join(tmpdir, o_name)
            of_ext = _check_ext(o_path)
            if not os.path.exists(o_path):
                logger.info('Extracting {} to {}...'.format(bname, o_path))
                with open(o_path, 'wb') as f:
                    f.write(zf.extractfile(members[0]).read())
    else:
        raise InvalidParamsError('Extension not recognized: '
                                 '{}'.format(file_extension))

    # Be sure we don't overfill the folder
    cfg.get_lru_handler(tmpdir, ending=of_ext).append(o_path)

    return o_path


def download_with_authentication(wwwfile, key):
    """ Uses credentials from a local .netrc file to download files

    This is function is currently used for TanDEM-X and ASTER

    Parameters
    ----------
    wwwfile : str
        path to the file to download
    key : str
        the machine to to look at in the .netrc file

    Returns
    -------

    """
    # Check the cache first. Use dummy download function to assure nothing is
    # tried to be downloaded without credentials:

    def _always_none(foo):
        return None

    cache_obj_name = _get_url_cache_name(wwwfile)
    dest_file = _verified_download_helper(cache_obj_name, _always_none)

    # Grab auth parameters
    if not dest_file:
        authfile = os.path.expanduser('~/.netrc')

        if not os.path.isfile(authfile):
            raise DownloadCredentialsMissingException(
                (authfile, ' does not exist. Add necessary credentials for ',
                 key, ' with `oggm_netrc_credentials. You may have to ',
                 'register at the respective service first.'))

        try:
            netrc(authfile).authenticators(key)[0]
        except TypeError:
            raise DownloadCredentialsMissingException(
                ('Credentials for ', key, ' are not in ', authfile, '. Add ',
                 'credentials for with `oggm_netrc_credentials`.'))

        dest_file = file_downloader(
            wwwfile, auth=(netrc(authfile).authenticators(key)[0],
                           netrc(authfile).authenticators(key)[2]))

    return dest_file


def download_oggm_files():
    with get_lock():
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
    with get_lock():
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


def _download_nasadem_file(zone):
    with get_lock():
        return _download_nasadem_file_unlocked(zone)


def _download_nasadem_file_unlocked(zone):
    """Checks if the NASADEM data is in the directory and if not, download it.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    wwwfile = ('https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/'
               '2000.02.11/NASADEM_HGT_{}.zip'.format(zone))
    demfile = '{}.hgt'.format(zone)
    outpath = os.path.join(tmpdir, demfile)

    # check if extracted file exists already
    if os.path.exists(outpath):
        return outpath

    # Did we download it yet?
    dest_file = file_downloader(wwwfile)

    # None means we tried hard but we couldn't find it
    if not dest_file:
        return None

    # ok we have to extract it
    if not os.path.exists(outpath):
        with zipfile.ZipFile(dest_file) as zf:
            zf.extract(demfile, path=tmpdir)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(outpath)
    cfg.get_lru_handler(tmpdir).append(outpath)
    return outpath


def _download_tandem_file(zone):
    with get_lock():
        return _download_tandem_file_unlocked(zone)


def _download_tandem_file_unlocked(zone):
    """Checks if the tandem data is in the directory and if not, download it.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    bname = zone.split('/')[-1] + '_DEM.tif'
    wwwfile = ('https://download.geoservice.dlr.de/TDM90/files/DEM/'
               '{}.zip'.format(zone))
    outpath = os.path.join(tmpdir, bname)

    # check if extracted file exists already
    if os.path.exists(outpath):
        return outpath

    dest_file = download_with_authentication(wwwfile, 'geoservice.dlr.de')

    # That means we tried hard but we couldn't find it
    if not dest_file:
        return None
    elif not zipfile.is_zipfile(dest_file):
        # If the TanDEM-X tile does not exist, a invalid file is created.
        # See https://github.com/OGGM/oggm/issues/893 for more details
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
    with get_lock():
        return _download_dem3_viewpano_unlocked(zone)


def _download_dem3_viewpano_unlocked(zone):
    """Checks if the DEM3 data is in the directory and if not, download it.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    outpath = os.path.join(tmpdir, zone + '.tif')
    extract_dir = os.path.join(tmpdir, 'tmp_' + zone)
    mkdir(extract_dir, reset=True)

    # check if extracted file exists already
    if os.path.exists(outpath):
        return outpath

    # OK, so see if downloaded already
    # some files have a newer version 'v2'
    if zone in ['R33', 'R34', 'R35', 'R36', 'R37', 'R38', 'Q32', 'Q33', 'Q34',
                'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'P31', 'P32', 'P33',
                'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40']:
        ifile = 'http://viewfinderpanoramas.org/dem3/' + zone + 'v2.zip'
    elif zone in DEM3REG.keys():
        # We prepared these files as tif already
        ifile = ('https://cluster.klima.uni-bremen.de/~oggm/dem/'
                 'DEM3_MERGED/{}.tif'.format(zone))
        return file_downloader(ifile)
    else:
        ifile = 'http://viewfinderpanoramas.org/dem3/' + zone + '.zip'

    dfile = file_downloader(ifile)

    # None means we tried hard but we couldn't find it
    if not dfile:
        return None

    # ok we have to extract it
    with zipfile.ZipFile(dfile) as zf:
        zf.extractall(extract_dir)

    # Serious issue: sometimes, if a southern hemisphere URL is queried for
    # download and there is none, a NH zip file is downloaded.
    # Example: http://viewfinderpanoramas.org/dem3/SN29.zip yields N29!
    # BUT: There are southern hemisphere files that download properly. However,
    # the unzipped folder has the file name of
    # the northern hemisphere file. Some checks if correct file exists:
    if len(zone) == 4 and zone.startswith('S'):
        zonedir = os.path.join(extract_dir, zone[1:])
    else:
        zonedir = os.path.join(extract_dir, zone)
    globlist = glob.glob(os.path.join(zonedir, '*.hgt'))

    # take care of the special file naming cases
    if zone in DEM3REG.keys():
        globlist = glob.glob(os.path.join(extract_dir, '*', '*.hgt'))

    if not globlist:
        # Final resort
        globlist = (findfiles(extract_dir, '.hgt') or
                    findfiles(extract_dir, '.HGT'))
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


def _download_aster_file(zone):
    with get_lock():
        return _download_aster_file_unlocked(zone)


def _download_aster_file_unlocked(zone):
    """Checks if the ASTER data is in the directory and if not, download it.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)
    wwwfile = ('https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/'
               '2000.03.01/{}.zip'.format(zone))
    outpath = os.path.join(tmpdir, zone + '_dem.tif')

    # check if extracted file exists already
    if os.path.exists(outpath):
        return outpath

    # download from NASA Earthdata with credentials
    dest_file = download_with_authentication(wwwfile, 'urs.earthdata.nasa.gov')

    # That means we tried hard but we couldn't find it
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


def _download_topo_file_from_cluster(fname):
    with get_lock():
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


def _download_copdem_file(cppfile, tilename, source):
    with get_lock():
        return _download_copdem_file_unlocked(cppfile, tilename, source)


def _download_copdem_file_unlocked(cppfile, tilename, source):
    """Checks if Copernicus DEM file is in the directory, if not download it.

    cppfile : name of the tarfile to download
    tilename : name of folder and tif file within the cppfile
    source : either 'COPDEM90' or 'COPDEM30'

    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)

    # tarfiles are extracted in directories per each tile
    fpath = '{0}_DEM.tif'.format(tilename)
    demfile = os.path.join(tmpdir, fpath)

    # check if extracted file exists already
    if os.path.exists(demfile):
        return demfile

    # Did we download it yet?
    ftpfile = ('ftps://cdsdata.copernicus.eu:990/' +
               'datasets/COP-DEM_GLO-{}-DGED/2022_1/'.format(source[-2:]) +
               cppfile)

    dest_file = download_with_authentication(ftpfile,
                                             'spacedata.copernicus.eu')

    # None means we tried hard but we couldn't find it
    if not dest_file:
        return None

    # ok we have to extract it
    if not os.path.exists(demfile):
        tiffile = os.path.join(tilename, 'DEM', fpath)
        with tarfile.open(dest_file) as tf:
            tmember = tf.getmember(tiffile)
            # do not extract the full path of the file
            tmember.name = os.path.basename(tf.getmember(tiffile).name)
            tf.extract(tmember, tmpdir)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(demfile)
    cfg.get_lru_handler(tmpdir).append(demfile)

    return demfile


def _download_aw3d30_file(zone):
    with get_lock():
        return _download_aw3d30_file_unlocked(zone)


def _download_aw3d30_file_unlocked(fullzone):
    """Checks if the AW3D30 data is in the directory and if not, download it.
    """

    # extract directory
    tmpdir = cfg.PATHS['tmp_dir']
    mkdir(tmpdir)

    # tarfiles are extracted in directories per each tile
    tile = fullzone.split('/')[1]
    demfile = os.path.join(tmpdir, tile, tile + '_AVE_DSM.tif')

    # check if extracted file exists already
    if os.path.exists(demfile):
        return demfile

    # Did we download it yet?
    ftpfile = ('ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_v1804/'
               + fullzone + '.tar.gz')
    try:
        dest_file = file_downloader(ftpfile, timeout=180)
    except urllib.error.URLError:
        # This error is raised if file is not available, could be water
        return None

    # None means we tried hard but we couldn't find it
    if not dest_file:
        return None

    # ok we have to extract it
    if not os.path.exists(demfile):
        from oggm.utils import robust_tar_extract
        dempath = os.path.dirname(demfile)
        robust_tar_extract(dest_file, dempath)

    # See if we're good, don't overfill the tmp directory
    assert os.path.exists(demfile)
    # this tarfile contains several files
    for file in os.listdir(dempath):
        cfg.get_lru_handler(tmpdir).append(os.path.join(dempath, file))
    return demfile


def _download_mapzen_file(zone):
    with get_lock():
        return _download_mapzen_file_unlocked(zone)


def _download_mapzen_file_unlocked(zone):
    """Checks if the mapzen data is in the directory and if not, download it.
    """
    bucket = 'elevation-tiles-prod'
    prefix = 'geotiff'
    url = 'http://s3.amazonaws.com/%s/%s/%s' % (bucket, prefix, zone)

    # That's all
    return file_downloader(url, timeout=180)


def get_prepro_gdir(rgi_version, rgi_id, border, prepro_level, base_url=None):
    with get_lock():
        return _get_prepro_gdir_unlocked(rgi_version, rgi_id, border,
                                         prepro_level, base_url=base_url)


def get_prepro_base_url(base_url=None, rgi_version=None, border=None,
                        prepro_level=None):
    """Extended base url where to find the desired gdirs."""

    if base_url is None:
        raise InvalidParamsError('Starting with v1.6, users now have to '
                                 'explicitly indicate the url they want '
                                 'to start from.')

    if not base_url.endswith('/'):
        base_url += '/'

    if rgi_version is None:
        rgi_version = cfg.PARAMS['rgi_version']

    if border is None:
        border = cfg.PARAMS['border']

    url = base_url
    url += 'RGI{}/'.format(rgi_version)
    url += 'b_{:03d}/'.format(int(border))
    url += 'L{:d}/'.format(prepro_level)
    return url


def _get_prepro_gdir_unlocked(rgi_version, rgi_id, border, prepro_level,
                              base_url=None):

    url = get_prepro_base_url(rgi_version=rgi_version, border=border,
                              prepro_level=prepro_level, base_url=base_url)
    if len(rgi_id) == 23:
        # RGI7
        url += '{}/{}.tar'.format(rgi_id[:17], rgi_id[:20])
    else:
        url += '{}/{}.tar'.format(rgi_id[:8], rgi_id[:11])
    tar_base = file_downloader(url)
    if tar_base is None:
        raise RuntimeError('Could not find file at ' + url)

    return tar_base


def get_geodetic_mb_dataframe(file_path=None):
    """Fetches the reference geodetic dataframe for calibration.

    Currently that's the data from Hughonnet et al 2021, corrected for
    outliers and with void filled. The data preparation script is
    available at
    https://nbviewer.jupyter.org/urls/cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/convert.ipynb

    Parameters
    ----------
    file_path : str
        in case you have your own file to parse (check the format first!)

    Returns
    -------
    a DataFrame with the data.
    """

    # fetch the file online or read custom file
    if file_path is None:
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/'
        file_name = 'hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide_filled.hdf'
        file_path = file_downloader(base_url + file_name)

    # Did we open it yet?
    if file_path in cfg.DATA:
        return cfg.DATA[file_path]

    # If not let's go
    extension = os.path.splitext(file_path)[1]
    if extension == '.csv':
        df = pd.read_csv(file_path, index_col=0)
    elif extension == '.hdf':
        df = pd.read_hdf(file_path)

    # Check for missing data (old files)
    if len(df.loc[df['dmdtda'].isnull()]) > 0:
        raise InvalidParamsError('The reference file you are using has missing '
                                 'data and is probably outdated (sorry for '
                                 'that). Delete the file at '
                                 f'{file_path} and start again.')
    cfg.DATA[file_path] = df
    return df


def get_temp_bias_dataframe(dataset='w5e5'):
    """Fetches the reference geodetic dataframe for calibration.

    Currently, that's the data from Hughonnet et al. (2021), corrected for
    outliers and with void filled. The data preparation script is
    available at
    https://nbviewer.jupyter.org/urls/cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/convert.ipynb

    Parameters
    ----------
    file_path : str
        in case you have your own file to parse (check the format first!)

    Returns
    -------
    a DataFrame with the data.
    """

    if dataset != 'w5e5':
        raise NotImplementedError(f'No such dataset available yet: {dataset}')

    # fetch the file online
    base_url = ('https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.6/'
                'w5e5_temp_bias_v2023.4.csv')

    file_path = file_downloader(base_url)

    # Did we open it yet?
    if file_path in cfg.DATA:
        return cfg.DATA[file_path]

    # If not let's go
    extension = os.path.splitext(file_path)[1]
    if extension == '.csv':
        df = pd.read_csv(file_path, index_col=0)
    elif extension == '.hdf':
        df = pd.read_hdf(file_path)

    cfg.DATA[file_path] = df
    return df


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
            if dy > 0:
                continue
            zx = np.ceil(dx / srtm_dx)
            zy = np.ceil(dy / srtm_dy)
            zones.append('{:02.0f}_{:02.0f}'.format(zx, zy))
    return list(sorted(set(zones)))


def _tandem_path(lon_tile, lat_tile):

    # OK we have a proper tile now
    # This changed in December 2022

    # First folder level is sorted from S to N
    level_0 = 'S' if lat_tile < 0 else 'N'
    level_0 += '{:02d}'.format(abs(lat_tile))

    # Second folder level is sorted from W to E, but in 10 steps
    level_1 = 'W' if lon_tile < 0 else 'E'
    level_1 += '{:03d}'.format(divmod(abs(lon_tile), 10)[0] * 10)

    # Level 2 is formatting, but depends on lat
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
                          dtype=int)
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
        lon_tiles = np.arange(l0, l1+1, dtype=int)
        for lon in lon_tiles:
            zones.append(_tandem_path(lon, lat))
    return list(sorted(set(zones)))


def _aw3d30_path(lon_tile, lat_tile):

    # OK we have a proper tile now

    # Folders are sorted with N E S W in 5 degree steps
    # But in N and E the lower boundary is indicated
    # e.g. N060 contains N060 - N064
    # e.g. E000 contains E000 - E004
    # but S and W indicate the upper boundary:
    # e.g. S010 contains S006 - S010
    # e.g. W095 contains W091 - W095

    # get letters
    ns = 'S' if lat_tile < 0 else 'N'
    ew = 'W' if lon_tile < 0 else 'E'

    # get lat/lon
    lon = abs(5 * np.floor(lon_tile/5))
    lat = abs(5 * np.floor(lat_tile/5))

    folder = '%s%.3d%s%.3d' % (ns, lat, ew, lon)
    filename = '%s%.3d%s%.3d' % (ns, abs(lat_tile), ew, abs(lon_tile))

    # Final path
    out = folder + '/' + filename
    return out


def aw3d30_zone(lon_ex, lat_ex):
    """Returns a list of AW3D30 zones covering the desired extent.
    """

    # Files are one by one tiles, so lets loop over them
    lon_tiles = np.arange(np.floor(lon_ex[0]), np.ceil(lon_ex[1]+1e-9),
                          dtype=int)
    lat_tiles = np.arange(np.floor(lat_ex[0]), np.ceil(lat_ex[1]+1e-9),
                          dtype=int)
    zones = []
    for lon in lon_tiles:
        for lat in lat_tiles:
            zones.append(_aw3d30_path(lon, lat))
    return list(sorted(set(zones)))


def _extent_to_polygon(lon_ex, lat_ex, to_crs=None):

    if lon_ex[0] == lon_ex[1] and lat_ex[0] == lat_ex[1]:
        out = shpg.Point(lon_ex[0], lat_ex[0])
    else:
        x = [lon_ex[0], lon_ex[1], lon_ex[1], lon_ex[0], lon_ex[0]]
        y = [lat_ex[0], lat_ex[0], lat_ex[1], lat_ex[1], lat_ex[0]]
        out = shpg.Polygon(np.array((x, y)).T)
    if to_crs is not None:
        out = salem.transform_geometry(out, to_crs=to_crs)
    return out


def arcticdem_zone(lon_ex, lat_ex):
    """Returns a list of Arctic-DEM zones covering the desired extent.
    """

    gdf = gpd.read_file(get_demo_file('ArcticDEM_Tile_Index_Rel7_by_tile.shp'))
    p = _extent_to_polygon(lon_ex, lat_ex, to_crs=gdf.crs)
    gdf = gdf.loc[gdf.intersects(p)]
    return gdf.tile.values if len(gdf) > 0 else []


def rema_zone(lon_ex, lat_ex):
    """Returns a list of REMA-DEM zones covering the desired extent.
    """

    gdf = gpd.read_file(get_demo_file('REMA_Tile_Index_Rel1.1.shp'))
    p = _extent_to_polygon(lon_ex, lat_ex, to_crs=gdf.crs)
    gdf = gdf.loc[gdf.intersects(p)]
    return gdf.tile.values if len(gdf) > 0 else []


def alaska_dem_zone(lon_ex, lat_ex):
    """Returns a list of Alaska-DEM zones covering the desired extent.
    """

    gdf = gpd.read_file(get_demo_file('Alaska_albers_V3_tiles.shp'))
    p = _extent_to_polygon(lon_ex, lat_ex, to_crs=gdf.crs)
    gdf = gdf.loc[gdf.intersects(p)]
    return gdf.tile.values if len(gdf) > 0 else []


def copdem_zone(lon_ex, lat_ex, source):
    """Returns a list of Copernicus DEM tarfile and tilename tuples
    """

    # because we use both meters and arc secs in our filenames...
    if source[-2:] == '90':
        asec = '30'
    elif source[-2:] == '30':
        asec = '10'
    else:
        raise InvalidDEMError('COPDEM Version not valid.')

    # either reuse or load lookup table
    if source in cfg.DATA:
        df = cfg.DATA[source]
    else:
        df = pd.read_csv(get_demo_file('{}_2022_1.csv'.format(source.lower())))
        cfg.DATA[source] = df

    # adding small buffer for unlikely case where one lon/lat_ex == xx.0
    lons = np.arange(np.floor(lon_ex[0]-1e-9), np.ceil(lon_ex[1]+1e-9))
    lats = np.arange(np.floor(lat_ex[0]-1e-9), np.ceil(lat_ex[1]+1e-9))

    flist = []
    for lat in lats:
        # north or south?
        ns = 'S' if lat < 0 else 'N'
        for lon in lons:
            # east or west?
            ew = 'W' if lon < 0 else 'E'
            lat_str = '{}{:02.0f}'.format(ns, abs(lat))
            lon_str = '{}{:03.0f}'.format(ew, abs(lon))
            try:
                filename = df.loc[(df['Long'] == lon_str) &
                                  (df['Lat'] == lat_str)]['CPP filename'].iloc[0]
                flist.append((filename,
                              'Copernicus_DSM_{}_{}_00_{}_00'.format(asec, lat_str, lon_str)))
            except IndexError:
                # COPDEM is global, if we miss tiles it is probably in the ocean
                pass
    return flist


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

            # test some rogue Greenland tiles as well
            elif (np.min(lon_ex) >= -72.) and (np.max(lon_ex) <= -66.) and \
                 (np.min(lat_ex) >= 76.) and (np.max(lat_ex) <= 80.):
                return ['T19']

            elif (np.min(lon_ex) >= -72.) and (np.max(lon_ex) <= -66.) and \
                 (np.min(lat_ex) >= 80.) and (np.max(lat_ex) <= 83.):
                return ['U19']

            elif (np.min(lon_ex) >= -66.) and (np.max(lon_ex) <= -60.) and \
                 (np.min(lat_ex) >= 80.) and (np.max(lat_ex) <= 83.):
                return ['U20']

            elif (np.min(lon_ex) >= -60.) and (np.max(lon_ex) <= -54.) and \
                 (np.min(lat_ex) >= 80.) and (np.max(lat_ex) <= 83.):
                return ['U21']

            elif (np.min(lon_ex) >= -54.) and (np.max(lon_ex) <= -48.) and \
                 (np.min(lat_ex) >= 80.) and (np.max(lat_ex) <= 83.):
                return ['U22']

            elif (np.min(lon_ex) >= -25.) and (np.max(lon_ex) <= -13.) and \
                 (np.min(lat_ex) >= 63.) and (np.max(lat_ex) <= 67.):
                return ['ISL']

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
    """Returns a list of ASTGTMV3 zones covering the desired extent.

    ASTER v3 tiles are 1 degree x 1 degree
    N50 contains 50 to 50.9
    E10 contains 10 to 10.9
    S70 contains -69.99 to -69.0
    W20 contains -19.99 to -19.0
    """

    # adding small buffer for unlikely case where one lon/lat_ex == xx.0
    lons = np.arange(np.floor(lon_ex[0]-1e-9), np.ceil(lon_ex[1]+1e-9))
    lats = np.arange(np.floor(lat_ex[0]-1e-9), np.ceil(lat_ex[1]+1e-9))

    zones = []
    for lat in lats:
        # north or south?
        ns = 'S' if lat < 0 else 'N'
        for lon in lons:
            # east or west?
            ew = 'W' if lon < 0 else 'E'
            filename = 'ASTGTMV003_{}{:02.0f}{}{:03.0f}'.format(ns, abs(lat),
                                                                ew, abs(lon))
            zones.append(filename)
    return list(sorted(set(zones)))


def nasadem_zone(lon_ex, lat_ex):
    """Returns a list of NASADEM zones covering the desired extent.

    NASADEM tiles are 1 degree x 1 degree
    N50 contains 50 to 50.9
    E10 contains 10 to 10.9
    S70 contains -69.99 to -69.0
    W20 contains -19.99 to -19.0
    """

    # adding small buffer for unlikely case where one lon/lat_ex == xx.0
    lons = np.arange(np.floor(lon_ex[0]-1e-9), np.ceil(lon_ex[1]+1e-9))
    lats = np.arange(np.floor(lat_ex[0]-1e-9), np.ceil(lat_ex[1]+1e-9))

    zones = []
    for lat in lats:
        # north or south?
        ns = 's' if lat < 0 else 'n'
        for lon in lons:
            # east or west?
            ew = 'w' if lon < 0 else 'e'
            filename = '{}{:02.0f}{}{:03.0f}'.format(ns, abs(lat), ew,
                                                     abs(lon))
            zones.append(filename)
    return list(sorted(set(zones)))


def mapzen_zone(lon_ex, lat_ex, dx_meter=None, zoom=None):
    """Returns a list of AWS mapzen zones covering the desired extent.

    For mapzen one has to specify the level of detail (zoom) one wants. The
    best way in OGGM is to specify dx_meter of the underlying map and OGGM
    will decide which zoom level works best.
    """

    if dx_meter is None and zoom is None:
        raise InvalidParamsError('Need either zoom level or dx_meter.')

    bottom, top = lat_ex
    left, right = lon_ex
    ybound = 85.0511
    if bottom <= -ybound:
        bottom = -ybound
    if top <= -ybound:
        top = -ybound
    if bottom > ybound:
        bottom = ybound
    if top > ybound:
        top = ybound
    if right >= 180:
        right = 179.999
    if left >= 180:
        left = 179.999

    if dx_meter:
        # Find out the zoom so that we are close to the desired accuracy
        lat = np.max(np.abs([bottom, top]))
        zoom = int(np.ceil(math.log2((math.cos(lat * math.pi / 180) *
                                      2 * math.pi * WEB_EARTH_RADUIS) /
                                     (WEB_N_PIX * dx_meter))))

        # According to this we should just always stay above 10 (sorry)
        # https://github.com/tilezen/joerd/blob/master/docs/data-sources.md
        zoom = 10 if zoom < 10 else zoom

    # Code from planetutils
    size = 2 ** zoom
    xt = lambda x: int((x + 180.0) / 360.0 * size)
    yt = lambda y: int((1.0 - math.log(math.tan(math.radians(y)) +
                                       (1 / math.cos(math.radians(y))))
                        / math.pi) / 2.0 * size)
    tiles = []
    for x in range(xt(left), xt(right) + 1):
        for y in range(yt(top), yt(bottom) + 1):
            tiles.append('/'.join(map(str, [zoom, x, str(y) + '.tif'])))
    return tiles


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

    outf = os.path.join(sdir, 'rgi_wgms_links_20220112.csv')
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
        '5', '6', '62', '70G', '70C'
        defaults to None (linking to the one specified in cfg.PARAMS)
    reset : bool
        If True, deletes the RGI directory first and downloads the data

    Returns
    -------
    str
        path to the RGI directory
    """

    with get_lock():
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

    pattern = '*_rgi{}_*.zip'.format(version)
    test_file = os.path.join(rgi_dir, f'*_rgi*{version}_manifest.txt')

    if version == '50':
        dfile = 'http://www.glims.org/RGI/rgi50_files/rgi50.zip'
    elif version == '60':
        dfile = 'http://www.glims.org/RGI/rgi60_files/00_rgi60.zip'
    elif version == '61':
        dfile = 'https://cluster.klima.uni-bremen.de/data/rgi/rgi_61.zip'
    elif version == '62':
        dfile = 'https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62.zip'
    elif version == '70G':
        pattern = 'RGI2000-*.zip'
        test_file = os.path.join(rgi_dir, 'RGI2000-v7.0-G-01_alaska', 'README.md')
        dfile = 'https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/'
        dfile += 'global_files/RGI2000-v7.0-G-global.zip'
    elif version == '70C':
        pattern = 'RGI2000-*.zip'
        test_file = os.path.join(rgi_dir, 'RGI2000-v7.0-C-01_alaska', 'README.md')
        dfile = 'https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/'
        dfile += 'global_files/RGI2000-v7.0-C-global.zip'

    if len(glob.glob(test_file)) == 0:
        # if not there download it
        ofile = file_downloader(dfile, reset=reset)
        if ofile is None:
            raise RuntimeError(f'Could not download RGI file: {dfile}')
        # Extract root
        try:
            with zipfile.ZipFile(ofile) as zf:
                zf.extractall(rgi_dir)
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(f'RGI file BadZipFile error: {ofile}')
        # Extract subdirs
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
            raise RuntimeError('Could not find a readme file in the RGI '
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
        '62', '70G', '70C', defaults to None (linking to the one specified in cfg.PARAMS)
    reset : bool
        If True, deletes the RGI directory first and downloads the data

    Returns
    -------
    str
        path to the RGI shapefile
    """

    rgi_dir = get_rgi_dir(version=version, reset=reset)
    if version in ['70G', '70C']:
        f = list(glob.glob(rgi_dir + f"/*/*-{region}_*.shp"))
    else:
        f = list(glob.glob(rgi_dir + "/*/*{}_*.shp".format(region)))
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
        the rgi version ('62', '70G', '70C')

    Returns
    -------
    geopandas.GeoDataFrame
        containing the desired RGI glacier outlines
    """

    if version is None:
        if len(rgi_ids[0]) == 14:
            # RGI6
            version = rgi_ids[0].split('-')[0][-2:]
        else:
            # RGI7 RGI2000-v7.0-G-02-00003
            assert rgi_ids[0].split('-')[1] == 'v7.0'
            version = '70' + rgi_ids[0].split('-')[2]

    if version in ['70G', '70C']:
        regions = [s.split('-')[-2] for s in rgi_ids]
    else:
        regions = [s.split('-')[1].split('.')[0] for s in rgi_ids]

    selection = []
    for reg in sorted(np.unique(regions)):
        sh = gpd.read_file(get_rgi_region_file(reg, version=version))
        try:
            selection.append(sh.loc[sh.RGIId.isin(rgi_ids)])
        except AttributeError:
            selection.append(sh.loc[sh.rgi_id.isin(rgi_ids)])

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
        '5', '6', '70G', defaults to None (linking to the one specified in cfg.PARAMS)
    reset : bool
        If True, deletes the intersects before redownloading them

    Returns
    -------
    str
        path to the directory
    """

    with get_lock():
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
    if version == '62':
        dfile = ('https://cluster.klima.uni-bremen.de/~oggm/rgi/'
                 'rgi62_Intersects.zip')
    if version == '70G':
        dfile = 'https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/'
        dfile += 'global_files/RGI2000-v7.0-I-global.zip'

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
    elif version == '62':
        test_file = os.path.join(odir, '*ntersect*anifest.txt')
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
    elif version == '70G':
        test_file = os.path.join(odir, 'README.md')
        if len(glob.glob(test_file)) == 0:
            # if not there download it
            ofile = file_downloader(dfile, reset=reset)
            # Extract root
            with zipfile.ZipFile(ofile) as zf:
                zf.extractall(odir)
            # Extract subdirs
            pattern = 'RGI2000-*.zip'
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
                raise RuntimeError('Could not find a README file in the RGI intersects'
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
        '5', '6', '61', '70G'. Defaults the one specified in cfg.PARAMS
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

    if version == '70G':
        f = list(glob.glob(os.path.join(rgi_dir, "*", f'*-{region}_*.shp')))
    else:
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
        '5', '6', '61', '70G'. Defaults the one specified in cfg.PARAMS

    Returns
    -------
    geopandas.GeoDataFrame
        with the selected intersects
    """

    if version is None:
        version = cfg.PARAMS['rgi_version']

    if len(version) == 1:
        version += '0'

    # RGI V6 or 7
    if version == '70G':
        regions = [s.split('-')[-2] for s in rgi_ids]
    else:
        try:
            regions = [s.split('-')[3] for s in rgi_ids]
        except IndexError:
            regions = [s.split('-')[1].split('.')[0] for s in rgi_ids]

    selection = []
    for reg in sorted(np.unique(regions)):
        sh = gpd.read_file(get_rgi_intersects_region_file(reg, version=version))
        if version == '70G':
            selection.append(sh.loc[sh.rgi_g_id_1.isin(rgi_ids) |
                                    sh.rgi_g_id_2.isin(rgi_ids)])
        else:
            selection.append(sh.loc[sh.RGIId_1.isin(rgi_ids) |
                                    sh.RGIId_2.isin(rgi_ids)])

    # Make a new dataframe of those
    selection = pd.concat(selection)
    selection.crs = sh.crs  # for geolocalisation

    return selection


def is_dem_source_available(source, lon_ex, lat_ex):
    """Checks if a DEM source is available for your purpose.

    This is only a very rough check! It doesn't mean that the data really is
    available, but at least it's worth a try.

    Parameters
    ----------
    source : str, required
        the source you want to check for
    lon_ex : tuple or int, required
        a (min_lon, max_lon) tuple delimiting the requested area longitudes
    lat_ex : tuple or int, required
        a (min_lat, max_lat) tuple delimiting the requested area latitudes

    Returns
    -------
    True or False
    """
    from oggm.utils import tolist
    lon_ex = tolist(lon_ex, length=2)
    lat_ex = tolist(lat_ex, length=2)

    def _in_grid(grid_json, lon, lat):
        i, j = cfg.DATA['dem_grids'][grid_json].transform(lon, lat,
                                                          maskout=True)
        return np.all(~ (i.mask | j.mask))

    if source == 'GIMP':
        return _in_grid('gimpdem_90m_v01.1.json', lon_ex, lat_ex)
    elif source == 'ARCTICDEM':
        return _in_grid('arcticdem_mosaic_100m_v3.0.json', lon_ex, lat_ex)
    elif source == 'RAMP':
        return _in_grid('AntarcticDEM_wgs84.json', lon_ex, lat_ex)
    elif source == 'REMA':
        return _in_grid('REMA_100m_dem.json', lon_ex, lat_ex)
    elif source == 'ALASKA':
        return _in_grid('Alaska_albers_V3.json', lon_ex, lat_ex)
    elif source == 'TANDEM':
        return True
    elif source == 'AW3D30':
        return np.min(lat_ex) > -60
    elif source == 'MAPZEN':
        return True
    elif source == 'DEM3':
        return True
    elif source == 'ASTER':
        return True
    elif source == 'SRTM':
        return np.max(np.abs(lat_ex)) < 60
    elif source in ['COPDEM30', 'COPDEM90']:
        return True
    elif source == 'NASADEM':
        return (np.min(lat_ex) > -56) and (np.max(lat_ex) < 60)
    elif source == 'USER':
        return True
    elif source is None:
        return True


def default_dem_source(rgi_id):
    """Current default DEM source at a given location.

    Parameters
    ----------
    rgi_id : str
        the RGI id

    Returns
    -------
    the chosen DEM source
    """
    rgi_reg = 'RGI{}'.format(rgi_id[6:8])
    rgi_id = rgi_id[:14]
    if cfg.DEM_SOURCE_TABLE.get(rgi_reg) is None:
        fp = get_demo_file('rgi62_dem_frac.h5')
        cfg.DEM_SOURCE_TABLE[rgi_reg] = pd.read_hdf(fp)

    sel = cfg.DEM_SOURCE_TABLE[rgi_reg].loc[rgi_id]
    for s in ['NASADEM', 'COPDEM90', 'COPDEM30', 'GIMP', 'REMA',
              'RAMP', 'TANDEM', 'MAPZEN']:
        if sel.loc[s] > 0.75:
            return s
    # If nothing works, try COPDEM again
    return 'COPDEM90'


def get_topo_file(lon_ex=None, lat_ex=None, gdir=None, *,
                  dx_meter=None, zoom=None, source=None):
    """Path(s) to the DEM file(s) covering the desired extent.

    If the needed files for covering the extent are not present, download them.

    The default behavior is to try a list of DEM sources in order, and
    stop once the downloaded data is covering a large enough part of the
    glacier. The DEM sources are tested in the following order:

    'NASADEM' -> 'COPDEM' -> 'GIMP' -> 'REMA' -> 'TANDEM' -> 'MAPZEN'

    To force usage of a certain data source, use the ``source`` kwarg argument.

    Parameters
    ----------
    lon_ex : tuple or int, required
        a (min_lon, max_lon) tuple delimiting the requested area longitudes
    lat_ex : tuple or int, required
        a (min_lat, max_lat) tuple delimiting the requested area latitudes
    gdir : GlacierDirectory, required if source=None
        the glacier id, used to decide on the DEM source
    rgi_region : str, optional
        the RGI region number (required for the GIMP DEM)
    rgi_subregion : str, optional
        the RGI subregion str (useful for RGI Reg 19)
    dx_meter : float, required for source='MAPZEN'
        the resolution of the glacier map (to decide the zoom level of mapzen)
    zoom : int, optional
        if you know the zoom already (for MAPZEN only)
    source : str or list of str, optional
        Name of specific DEM source. see utils.DEM_SOURCES for a list

    Returns
    -------
    tuple: (list with path(s) to the DEM file(s), data source str)
    """
    from oggm.utils import tolist
    lon_ex = tolist(lon_ex, length=2)
    lat_ex = tolist(lat_ex, length=2)

    if source is not None and not isinstance(source, str):
        # check all user options
        for s in source:
            demf, source_str = get_topo_file(lon_ex=lon_ex, lat_ex=lat_ex,
                                             gdir=gdir, source=s)
            if demf[0]:
                return demf, source_str

    # Did the user specify a specific DEM file?
    if 'dem_file' in cfg.PATHS and os.path.isfile(cfg.PATHS['dem_file']):
        source = 'USER' if source is None else source
        if source == 'USER':
            return [cfg.PATHS['dem_file']], source

    # Some logic to decide which source to take if unspecified
    if source is None:
        if gdir is None:
            raise InvalidParamsError('gdir is needed if source=None')
        source = getattr(gdir, 'rgi_dem_source')
        if source is None:
            source = default_dem_source(gdir.rgi_id)

    if source not in DEM_SOURCES:
        raise InvalidParamsError('`source` must be one of '
                                 '{}'.format(DEM_SOURCES))

    # OK go
    files = []
    if source == 'GIMP':
        _file = _download_topo_file_from_cluster('gimpdem_90m_v01.1.tif')
        files.append(_file)

    if source == 'ARCTICDEM':
        zones = arcticdem_zone(lon_ex, lat_ex)
        for z in zones:
            with get_lock():
                url = 'https://cluster.klima.uni-bremen.de/~oggm/'
                url += 'dem/ArcticDEM_100m_v3.0/'
                url += '{}_100m_v3.0/{}_100m_v3.0_reg_dem.tif'.format(z, z)
                files.append(file_downloader(url))

    if source == 'RAMP':
        _file = _download_topo_file_from_cluster('AntarcticDEM_wgs84.tif')
        files.append(_file)

    if source == 'ALASKA':
        zones = alaska_dem_zone(lon_ex, lat_ex)
        for z in zones:
            with get_lock():
                url = 'https://cluster.klima.uni-bremen.de/~oggm/'
                url += 'dem/Alaska_albers_V3/'
                url += '{}_Alaska_albers_V3/'.format(z)
                url += '{}_Alaska_albers_V3.tif'.format(z)
                files.append(file_downloader(url))

    if source == 'REMA':
        zones = rema_zone(lon_ex, lat_ex)
        for z in zones:
            with get_lock():
                url = 'https://cluster.klima.uni-bremen.de/~oggm/'
                url += 'dem/REMA_100m_v1.1/'
                url += '{}_100m_v1.1/{}_100m_v1.1_reg_dem.tif'.format(z, z)
                files.append(file_downloader(url))

    if source == 'TANDEM':
        zones = tandem_zone(lon_ex, lat_ex)
        for z in zones:
            files.append(_download_tandem_file(z))

    if source == 'AW3D30':
        zones = aw3d30_zone(lon_ex, lat_ex)
        for z in zones:
            files.append(_download_aw3d30_file(z))

    if source == 'MAPZEN':
        zones = mapzen_zone(lon_ex, lat_ex, dx_meter=dx_meter, zoom=zoom)
        for z in zones:
            files.append(_download_mapzen_file(z))

    if source == 'ASTER':
        zones = aster_zone(lon_ex, lat_ex)
        for z in zones:
            files.append(_download_aster_file(z))

    if source == 'DEM3':
        zones = dem3_viewpano_zone(lon_ex, lat_ex)
        for z in zones:
            files.append(_download_dem3_viewpano(z))

    if source == 'SRTM':
        zones = srtm_zone(lon_ex, lat_ex)
        for z in zones:
            files.append(_download_srtm_file(z))

    if source in ['COPDEM30', 'COPDEM90']:
        filetuple = copdem_zone(lon_ex, lat_ex, source)
        for cpp, eop in filetuple:
            files.append(_download_copdem_file(cpp, eop, source))

    if source == 'NASADEM':
        zones = nasadem_zone(lon_ex, lat_ex)
        for z in zones:
            files.append(_download_nasadem_file(z))

    # filter for None (e.g. oceans)
    files = [s for s in files if s]
    if files:
        return files, source
    else:
        raise InvalidDEMError('Source: {2} no topography file available for '
                              'extent lat:{0}, lon:{1}!'.
                              format(lat_ex, lon_ex, source))
