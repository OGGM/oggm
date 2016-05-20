"""Some useful functions that did not fit into the other modules.

Copyright: OGGM developers, 2014-2015

License: GPLv3+
"""
from __future__ import absolute_import, division

import six.moves.cPickle as pickle
from six import string_types
from six.moves.urllib.request import urlretrieve, urlopen
from six.moves.urllib.error import HTTPError, URLError

# Builtins
import glob
import os
import gzip
import shutil
import zipfile
import sys
import math
from shutil import copyfile
from collections import OrderedDict
from functools import partial, wraps
import json
import time
import fnmatch
import subprocess

# External libs
import geopandas as gpd
import pandas as pd
from salem import lazy_property
from salem.utils import read_shapefile
import numpy as np
import netCDF4
from scipy import stats
from joblib import Memory
from shapely.ops import transform as shp_trafo
from salem import wgs84
import xarray as xr
import rasterio
from rasterio.tools.merge import merge as merge_tool
import multiprocessing as mp

# Locals
import oggm.cfg as cfg

SAMPLE_DATA_GH_REPO = 'OGGM/oggm-sample-data'
CRU_SERVER = 'https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_3.23/cruts' \
             '.1506241137.v3.23/'
# Joblib
MEMORY = Memory(cachedir=cfg.CACHE_DIR, verbose=0)

# Function
tuple2int = partial(np.array, dtype=np.int64)

# File Download lock
download_lock = mp.Lock()


def _urlretrieve(url, ofile, *args, **kwargs):
    try:
        return urlretrieve(url, ofile, *args, **kwargs)
    except:
        if os.path.exists(ofile):
            os.remove(ofile)
        raise


def progress_urlretrieve(url, ofile):
    print("Downloading %s ..." % url)
    try:
        from progressbar import DataTransferBar, UnknownLength
        pbar = DataTransferBar()
        def _upd(count, size, total):
            if pbar.max_value is None:
                if total > 0:
                    pbar.start(total)
                else:
                    pbar.start(UnknownLength)
            pbar.update(min(count * size, total))
        res = _urlretrieve(url, ofile, reporthook=_upd)
        try:
            pbar.finish()
        except:
            pass
        return res
    except ImportError:
        return _urlretrieve(url, ofile)


def empty_cache():  # pragma: no cover
    """Empty oggm's cache directory."""

    if os.path.exists(cfg.CACHE_DIR):
        shutil.rmtree(cfg.CACHE_DIR)
    os.makedirs(cfg.CACHE_DIR)


def _download_oggm_files():
    with download_lock:
        return _download_oggm_files_unlocked()

def _download_oggm_files_unlocked():
    """Checks if the demo data is already on the cache and downloads it."""

    master_sha_url = 'https://api.github.com/repos/%s/commits/master' % \
                     SAMPLE_DATA_GH_REPO
    master_zip_url = 'https://github.com/%s/archive/master.zip' % \
                     SAMPLE_DATA_GH_REPO
    ofile = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data.zip')
    shafile = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-commit.txt')
    odir = os.path.join(cfg.CACHE_DIR)

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
    if time.time() - last_mod > 3600:
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
        except (HTTPError, URLError):
            master_sha = 'error'
    else:
        write_sha = False

    # download only if necessary
    if not os.path.exists(ofile):
        progress_urlretrieve(master_zip_url, ofile)
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(odir)

    # sha did change, replace
    if write_sha:
        with open(shafile, 'w') as sfile:
            sfile.write(master_sha)

    # list of files for output
    out = dict()
    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master')
    for root, directories, filenames in os.walk(sdir):
        for filename in filenames:
            out[filename] = os.path.join(root, filename)

    return out


def _download_srtm_file(zone):
    with download_lock:
        return _download_srtm_file_unlocked(zone)


def _download_srtm_file_unlocked(zone):
    """Checks if the srtm data is in the directory and if not, download it.
    """

    odir = os.path.join(cfg.PATHS['topo_dir'], 'srtm')
    if not os.path.exists(odir):
        os.makedirs(odir)
    ofile = os.path.join(odir, 'srtm_' + zone + '.zip')
#    ifile = 'http://srtm.csi.cgiar.org/SRT-ZIP/SRTM_V41/SRTM_Data_GeoTiff' \
    ifile = 'http://droppr.org/srtm/v4.1/6_5x5_TIFs' \
            '/srtm_' + zone + '.zip'
    if not os.path.exists(ofile):
        retry_counter = 0
        retry_max = 5
        while True:
            # Try to download
            try:
                retry_counter += 1
                progress_urlretrieve(ifile, ofile)
                with zipfile.ZipFile(ofile) as zf:
                    zf.extractall(odir)
                break
            except HTTPError as err:
                # This works well for py3
                if err.code == 404:
                    # Ok so this *should* be an ocean tile
                    return None
                elif err.code >= 500 and err.code < 600 and retry_counter <= retry_max:
                    print("Downloading srtm data failed with HTTP error %s, retrying in 10 seconds... %s/%s" %
                          (err.code, retry_counter, retry_max))
                    time.sleep(10)
                    continue
                else:
                    raise
            except zipfile.BadZipfile:
                # This is for py2
                # Ok so this *should* be an ocean tile
                return None

    out = os.path.join(odir, 'srtm_' + zone + '.tif')
    assert os.path.exists(out)
    return out


def _download_dem3_viewpano(zone, specialzones):
    with download_lock:
        return _download_dem3_viewpano_unlocked(zone, specialzones)


def _download_dem3_viewpano_unlocked(zone, specialzones):
    """Checks if the srtm data is in the directory and if not, download it.
    """

    odir = os.path.join(cfg.PATHS['topo_dir'], 'dem3')

    if not os.path.exists(odir):
        os.makedirs(odir)
    ofile = os.path.join(odir, 'dem3_' + zone + '.zip')
    outpath = os.path.join(odir, zone+'.tif')

    # check if TIFF file exists already
    if os.path.exists(outpath):
        return outpath

    # some files have a newer version 'v2'
    if zone in ['R33', 'R34', 'R35', 'R36', 'R37', 'R38', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40',
                'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40']:
        ifile = 'http://viewfinderpanoramas.org/dem3/' + zone + 'v2.zip'
    elif zone in ['01-15', '16-30', '31-45', '46-60']:
        ifile = 'http://viewfinderpanoramas.org/ANTDEM3/' + zone + '.zip'
    else:
        ifile = 'http://viewfinderpanoramas.org/dem3/' + zone + '.zip'

    if not os.path.exists(ofile):
        retry_counter = 0
        retry_max = 5
        while True:
            # Try to download
            try:
                retry_counter += 1
                urlretrieve(ifile, ofile)
                with zipfile.ZipFile(ofile) as zf:
                    zf.extractall(odir)
                break
            except HTTPError as err:
                # This works well for py3
                if err.code == 404:
                    # Ok so this *should* be an ocean tile
                    return None
                elif err.code >= 500 and err.code < 600 and retry_counter <= retry_max:
                    print("Downloading srtm data failed with HTTP error %s, retrying in 10 seconds... %s/%s" %
                          (err.code, retry_counter, retry_max))
                    time.sleep(10)
                    continue
                else:
                    raise
            except zipfile.BadZipfile:
                # This is for py2
                # Ok so this *should* be an ocean tile
                return None

    # Serious issue: sometimes, if a southern hemisphere URL is queried for download and there is none
    # a NH zip file os downloaded. Example: http://viewfinderpanoramas.org/dem3/SN29.zip yields N29!
    # BUT: There are southern hemisphere files that download properly. However, the unzipped folder has the file name of
    # the northern hemisphere file. Some checks if correct file exists:

    if len(zone)==4 and zone.startswith('S'):
        zonedir = os.path.join(odir, zone[1:])
        globlist = glob.glob(zonedir+'\\S*.hgt') #'S' is important: cloud be confused with NH files otherwise
    else:
        zonedir = os.path.join(odir, zone)
        globlist = glob.glob(zonedir+'\\*.hgt')

    # take care of the special file naming cases (very ineffective...)
    if zone in specialzones.keys():
        globlist = glob.glob(odir+'\\*\\*.hgt')

    if not globlist:
        return None

    # merge the single HGT files (can be a bit ineffective, bcz not every single file might be exactly within extent...)
    rfiles = [rasterio.open(s) for s in globlist]
    dest, output_transform = merge_tool(rfiles)
    profile = rfiles[0].profile
    profile.pop('affine')
    profile['transform'] = output_transform
    profile['height'] = dest.shape[1]
    profile['width'] = dest.shape[2]
    profile['driver'] = 'GTiff'
    with rasterio.open(outpath, 'w', **profile) as dst:
        dst.write(dest)

    assert os.path.exists(outpath)

    return outpath


def _download_aster_file(zone, unit):
    with download_lock:
        return _download_aster_file_unlocked(zone, unit)


def _download_aster_file_unlocked(zone, unit):
    """Checks if the aster data is in the directory and if not, download it.

    You need AWS cli and AWS credentials for this. Quoting Timo:

    $ aws configure

    Key ID und Secret you should have
    Region is eu-west-1 and Output Format is json.
    """

    odir = os.path.join(cfg.PATHS['topo_dir'], 'aster')
    if not os.path.exists(odir):
        os.makedirs(odir)
    fbname = 'ASTGTM2_' + zone + '.zip'
    dirbname = 'UNIT_' + unit
    ofile = os.path.join(odir, fbname)

    cmd = 'aws --region eu-west-1 s3 cp s3://astgtmv2/ASTGTM_V2/'
    cmd = cmd + dirbname + '/' + fbname + ' ' + ofile
    if not os.path.exists(ofile):
        subprocess.call(cmd, shell=True)
        if os.path.exists(ofile):
            # Ok so the tile is a valid one
            with zipfile.ZipFile(ofile) as zf:
                zf.extractall(odir)
        else:
            # Ok so this *should* be an ocean tile
            return None

    out = os.path.join(odir, 'ASTGTM2_' + zone + '_dem.tif')
    assert os.path.exists(out)
    return out


def _download_alternate_topo_file(fname):
    with download_lock:
        return _download_alternate_topo_file_unlocked(fname)


def _download_alternate_topo_file_unlocked(fname):
    """Checks if the special topo data is in the directory and if not,
    download it from AWS.

    You need AWS cli and AWS credentials for this. Quoting Timo:

    $ aws configure

    Key ID und Secret you should have
    Region is eu-west-1 and Output Format is json.
    """

    fzipname = fname + '.zip'
    # Here we had a file exists check

    odir = os.path.join(cfg.PATHS['topo_dir'], 'alternate')
    if not os.path.exists(odir):
        os.makedirs(odir)
    ofile = os.path.join(odir, fzipname)

    cmd = 'aws --region eu-west-1 s3 cp s3://astgtmv2/topo/'
    cmd = cmd + fzipname + ' ' + ofile
    if not os.path.exists(ofile):
        subprocess.call(cmd, shell=True)
        if os.path.exists(ofile):
            # Ok so the tile is a valid one
            with zipfile.ZipFile(ofile) as zf:
                zf.extractall(odir)
        else:
            # Ok so this *should* be an ocean tile
            return None

    out = os.path.join(odir, fname)
    assert os.path.exists(out)
    return out


def _get_centerline_lonlat(gdir):
    """Quick n dirty solution to write the centerlines as a shapefile"""

    olist = []
    for i in gdir.divide_ids:
        cls = gdir.read_pickle('centerlines', div_id=i)
        for i, cl in enumerate(cls):
            mm = 1 if i==0 else 0
            gs = gpd.GeoSeries()
            gs['RGIID'] = gdir.rgi_id
            gs['DIVIDE'] = i
            gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
            gs['MAIN'] = mm
            tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
            gs['geometry'] = shp_trafo(tra_func, cl.line)
            olist.append(gs)

    return olist


def aws_file_download(aws_path, local_path, reset=False):
    with download_lock:
        return _aws_file_download_unlocked(aws_path, local_path, reset)


def _aws_file_download_unlocked(aws_path, local_path, reset=False):
    """Download a file from the AWS drive s3://astgtmv2/

    **Note:** you need AWS credentials for this to work.

    Parameters
    ----------
    aws_path: path relative to  s3://astgtmv2/
    local_path: where to copy the file
    reset: overwrite the local file
    """

    if reset and os.path.exists(local_path):
        os.remove(local_path)

    cmd = 'aws --region eu-west-1 s3 cp s3://astgtmv2/'
    cmd = cmd + aws_path + ' ' + local_path
    if not os.path.exists(local_path):
        subprocess.call(cmd, shell=True)
    if not os.path.exists(local_path):
        raise RuntimeError('Something went wrong with the download')


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

def query_yes_no(question, default="yes"):
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
    """
    Calculate the great circle distance between one point
    on the earth and an array of points (specified in decimal degrees)
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


@MEMORY.cache
def joblib_read_climate(ncpath, ilon, ilat, default_grad, minmax_grad,
                        prcp_scaling_factor, use_grad):
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
        iprcp = prcp[:, ilat, ilon] * prcp_scaling_factor

    # Now the gradient
    if use_grad:
        for t, loct in enumerate(ttemp):
            slope, _, _, p_val, _ = stats.linregress(thgt,
                                                     loct.flatten())
            igrad[t] = slope if (p_val < 0.01) else default_grad
        # dont exagerate too much
        igrad = np.clip(igrad, minmax_grad[0], minmax_grad[1])

    return iprcp, itemp, igrad, ihgt


def pipe_log(gdir, task_func, err=None):
    """Log the error in a specific directory."""

    fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    fpath = os.path.join(fpath, gdir.rgi_id)

    if err is not None:
        fpath += '.ERROR'
    else:
        return  # for now
        fpath += '.SUCCESS'

    with open(fpath, 'a') as f:
        f.write(task_func.__name__ + ': ')
        if err is not None:
            f.write(err.__class__.__name__ + ': {}'.format(err))


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
    lon_ex = np.linspace(mi, ma, np.ceil((ma - mi) + 3))
    mi, ma = np.min(lat_ex), np.max(lat_ex)
    lat_ex = np.linspace(mi, ma, np.ceil((ma - mi) + 3))

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


def dem3_viewpano_zone(lon_ex, lat_ex, specialfiles):
    """Returns a list of corrected SRTM zones from
    http://viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm covering the desired extent.
    """

    for _f in specialfiles.keys():  # Python version independent iteration through dictionary

        if (np.min(lon_ex) >= specialfiles[_f][0]) and (np.max(lon_ex) <= specialfiles[_f][1]) and \
           (np.min(lat_ex) >= specialfiles[_f][2]) and (np.max(lat_ex) <= specialfiles[_f][3]):

            # test some weired inset files in Antarctica
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

    # if the tile does *not* have a special name, its name can be found like this:
    # corrected SRTMs are sorted in tiles of 6 deg longitude and 4 deg latitude
    srtm_x0 = -180.
    srtm_y0 = 0.
    srtm_dx = 6.
    srtm_dy = 4.


    # quick n dirty solution to be sure that we will cover the whole range
    mi, ma = np.min(lon_ex), np.max(lon_ex)
    lon_ex = np.linspace(mi, ma, np.ceil((ma - mi)/srtm_dy)+3) # +3 is just for the number to become still a bit larger
    mi, ma = np.min(lat_ex), np.max(lat_ex)
    lat_ex = np.linspace(mi, ma, np.ceil((ma - mi)/srtm_dx)+3)

    zones = []
    for lon in lon_ex:
        for lat in lat_ex:
            dx = lon - srtm_x0
            dy = lat - srtm_y0

            zx = np.ceil(dx / srtm_dx)
            zy = chr(int(abs(dy / srtm_dy)) + ord('A'))  # convert number to letter

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
    lon_ex = np.linspace(mi, ma, np.ceil((ma - mi) + 3))
    mi, ma = np.min(lat_ex), np.max(lat_ex)
    lat_ex = np.linspace(mi, ma, np.ceil((ma - mi) + 3))

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

    d = _download_oggm_files()
    if fname in d:
        return d[fname]
    else:
        return None


def get_cru_cl_file():
    """Returns the path to the unpacked CRU CL file (is in sample data)."""

    _download_oggm_files()

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

    if cfg.PATHS['wgms_rgi_links'] != '~':
        if not os.path.exists(cfg.PATHS['wgms_rgi_links']):
            raise ValueError('wrong wgms_rgi_links path provided.')
        # User provided data
        outf = cfg.PATHS['wgms_rgi_links']
        datadir = os.path.join(os.path.dirname(outf), 'mbdata')
        if not os.path.exists(datadir):
            raise ValueError('The WGMS data directory is missing')
        return outf, datadir

    # Roll our own
    _download_oggm_files()
    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master', 'wgms')
    outf = os.path.join(sdir, 'rgi_wgms_links_2015_RGIV5.csv')
    assert os.path.exists(outf)
    datadir = os.path.join(sdir, 'mbdata')
    assert os.path.exists(datadir)
    return outf, datadir


def get_leclercq_files():
    """Get the path to the default Leclercq-RGI link file and the data dir.

    Returns
    -------
    (file, dir): paths to the files
    """

    if cfg.PATHS['leclercq_rgi_links'] != '~':
        if not os.path.exists(cfg.PATHS['leclercq_rgi_links']):
            raise ValueError('wrong leclercq_rgi_links path provided.')
        # User provided data
        outf = cfg.PATHS['leclercq_rgi_links']
        # TODO: This doesnt exist yet
        datadir = os.path.join(os.path.dirname(outf), 'lendata')
        # if not os.path.exists(datadir):
        #     raise ValueError('The Leclercq data directory is missing')
        return outf, datadir

    # Roll our own
    _download_oggm_files()
    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master', 'leclercq')
    outf = os.path.join(sdir, 'rgi_leclercq_links_2012_RGIV5.csv')
    assert os.path.exists(outf)
    # TODO: This doesnt exist yet
    datadir = os.path.join(sdir, 'lendata')
    # assert os.path.exists(datadir)
    return outf, datadir


def get_glathida_file():
    """Get the path to the default WGMS-RGI link file and the data dir.

    Returns
    -------
    (file, dir): paths to the files
    """

    if cfg.PATHS['glathida_rgi_links'] != '~':
        if not os.path.exists(cfg.PATHS['glathida_rgi_links']):
            raise ValueError('wrong glathida_rgi_links path provided.')
        # User provided data
        return cfg.PATHS['glathida_rgi_links']

    # Roll our own
    _download_oggm_files()
    sdir = os.path.join(cfg.CACHE_DIR, 'oggm-sample-data-master', 'glathida')
    outf = os.path.join(sdir, 'rgi_glathida_links_2014_RGIV5.csv')
    assert os.path.exists(outf)
    return outf


def get_rgi_dir():
    with download_lock:
        return _get_rgi_dir_unlocked()


def _get_rgi_dir_unlocked():
    """
    Returns a path to the RGI directory.

    If the files are not present, download them.

    Returns
    -------
    path to the RGI directory
    """

    # Be sure the user gave a sensible path to the rgi dir
    rgi_dir = cfg.PATHS['rgi_dir']
    if not os.path.exists(rgi_dir):
        raise ValueError('The RGI data directory does not exist!')

    bname = 'rgi50.zip'
    ofile = os.path.join(rgi_dir, bname)

    # if not there download it
    if not os.path.exists(ofile):  # pragma: no cover
        tf = 'http://www.glims.org/RGI/rgi50_files/' + bname
        progress_urlretrieve(tf, ofile)

        # Extract root
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(rgi_dir)

        # Extract subdirs
        pattern = '*_rgi50_*.zip'
        for root, dirs, files in os.walk(cfg.PATHS['rgi_dir']):
            for filename in fnmatch.filter(files, pattern):
                ofile = os.path.join(root, filename)
                with zipfile.ZipFile(ofile) as zf:
                    ex_root = ofile.replace('.zip', '')
                    if not os.path.exists(ex_root):
                        os.makedirs(ex_root)
                    zf.extractall(ex_root)

    return rgi_dir


def get_cru_file(var=None):
    with download_lock:
        return _get_cru_file_unlocked(var)


def _get_cru_file_unlocked(var=None):
    """
    Returns a path to the desired CRU TS file.

    If the file is not present, download it.

    Parameters
    ----------
    var: 'tmp' or 'pre'

    Returns
    -------
    path to the CRU file
    """

    # Be sure the user gave a sensible path to the climate dir
    cru_dir = cfg.PATHS['cru_dir']
    if not os.path.exists(cru_dir):
        raise ValueError('The CRU data directory({}) does not exist!'.format(cru_dir))

    # Be sure input makes sense
    if var not in ['tmp', 'pre']:
        raise ValueError('CRU variable {} does not exist!'.format(var))

    # cru_ts3.23.1901.2014.tmp.dat.nc
    bname = 'cru_ts3.23.1901.2014.{}.dat.nc'.format(var)
    ofile = os.path.join(cru_dir, bname)

    # if not there download it
    if not os.path.exists(ofile):  # pragma: no cover
        tf = CRU_SERVER + '{}/cru_ts3.23.1901.2014.{}.dat.nc.gz'.format(var,
                                                                        var)
        progress_urlretrieve(tf, ofile + '.gz')
        with gzip.GzipFile(ofile + '.gz') as zf:
            with open(ofile, 'wb') as outfile:
                for line in zf:
                    outfile.write(line)

    return ofile


def get_topo_file(lon_ex, lat_ex, region=None):
    """
    Returns a path to the DEM file covering the desired extent.

    If the file is not present, download it. If the extent covers two or
    more files, merge them.

    Returns a downloaded SRTM file for [-60S;60N], and
    a corrected DEM3 from viewfinderpanoramas.org else

    Parameters
    ----------
    lon_ex: (min_lon, max_lon)
    lat_ex: (min_lat, max_lat)

    Returns
    -------
    tuple: (path to the dem file, data source)
    """

    # Did the user specify a specific SRTM file?
    if ('dem_file' in cfg.PATHS) and os.path.isfile(cfg.PATHS['dem_file']):
        return cfg.PATHS['dem_file'], 'USER'

    # If not, do the job ourselves: download and merge stuffs
    topodir = cfg.PATHS['topo_dir']

    # TODO: GIMP is in polar stereographic, not easy to test
    # would be possible with a salem grid but this is a bit more expensive
    # than just asking RGI for the region
    if region is not None and int(region) == 5:
        gimp_file = _download_alternate_topo_file('gimpdem_90m.tif')
        return gimp_file, 'GIMP'

    # Some regional files I could gather
    # Iceland http://viewfinderpanoramas.org/dem3/ISL.zip
    # Svalbard http://viewfinderpanoramas.org/dem3/SVALBARD.zip
    # NorthCanada (could be larger - need tiles download)
    _exs = (
        [-25., -12., 63., 67.],
        [10., 34., 76., 81.],
        [-96., -60., 76., 84.]
    )
    _files = (
        'iceland.tif',
        'svalbard.tif',
        'northcanada.tif',
    )
    for _ex, _f in zip(_exs, _files):

        if (np.min(lon_ex) >= _ex[0]) and (np.max(lon_ex) <= _ex[1]) and \
           (np.min(lat_ex) >= _ex[2]) and (np.max(lat_ex) <= _ex[3]):
            r_file = _download_alternate_topo_file(_f)
            return r_file, 'REGIO'

    if (np.min(lat_ex) < -60.) or (np.max(lat_ex) > 60.):
        # TODO: https://github.com/OGGM/oggm/pull/79 does not work properly yet
        # # Use corrected viewpanoramas.org SRTM!
        # # some filenames deviate from the scheme (some do not contain glaciers, but for uniformity...)
        # specialfiles = {'ISL': [-25., -12., 63., 67.],       # Iceland
        #                 'SVALBARD': [10., 34., 76., 81.],
        #                 'JANMAYEN': [-10., -7., 70., 72.],
        #                 'FJ': [36., 66., 79., 82.],          # Franz Josef Land
        #                 'FAR': [-8., -6., 61., 63.],         # Faroer
        #                 'BEAR': [18., 20., 74., 75.],        # Bear Island
        #                 'SHL': [-3., 0., 60., 61.],          # Shetland
        #                 '01-15': [-180., -91., -90, -60.],   # Antarctica tiles as UTM zones, FILES ARE LARGE!!!!!
        #                 '16-30': [-91., -1., -90., -60.],
        #                 '31-45': [-1., 89., -90., -60.],
        #                 '46-60': [89., 189., -90., -60.],
        #                 'GL-North': [-78., -11., 75., 84.],  # Greenland tiles
        #                 'GL-West': [-68., -42., 64., 76.],
        #                 'GL-South': [-52., -40., 59., 64.],
        #                 'GL-East': [-42., -17., 64., 76.]}
        # zones = dem3_viewpano_zone(lon_ex, lat_ex, specialfiles)
        # sources = []
        # for z in zones:
        #     sources.append(_download_dem3_viewpano(z, specialfiles))
        # source_str = 'DEM3'
        #
        # # if download failed for some reason, use ASTER
        # if len(sources) < 1:
        zones, units = aster_zone(lon_ex, lat_ex)
        sources = []
        for z, u in zip(zones, units):
            sf = _download_aster_file(z, u)
            if sf is not None:
                sources.append(sf)
        source_str = 'ASTER'
    else:
        zones = srtm_zone(lon_ex, lat_ex)
        sources = []
        for z in zones:
            sources.append(_download_srtm_file(z))
        source_str = 'SRTM'

    if len(sources) < 1:
        raise RuntimeError('No topography file available!')
        # for the very last cases a very coarse dataset ?
        t_file = os.path.join(topodir, 'ETOPO1_Ice_g_geotiff.tif')
        assert os.path.exists(t_file)
        return t_file, 'ETOPO1'

    if len(sources) == 1:
        return sources[0], source_str
    else:
        # merge
        zone_str = '+'.join(zones)
        bname = source_str.lower() + '_merged_' + zone_str + '.tif'
        merged_file = os.path.join(topodir, source_str.lower(),
                                   bname)
        if not os.path.exists(merged_file):

            # check case where wrong zip file is downloaded from
            if all(x is None for x in sources):
                raise ValueError('Chosen lat/lon values are not available')

            # write it
            rfiles = [rasterio.open(s) for s in sources]
            dest, output_transform = merge_tool(rfiles)
            profile = rfiles[0].profile
            profile.pop('affine')
            profile['transform'] = output_transform
            profile['height'] = dest.shape[1]
            profile['width'] = dest.shape[2]
            profile['driver'] = 'GTiff'
            with rasterio.open(merged_file, 'w', **profile) as dst:
                dst.write(dest)
        return merged_file, source_str + '_MERGED'


def glacier_characteristics(gdirs):
    """Gathers as many statistics as possible about a list of glacier
    directories.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    gdirs: the list of GlacierDir to process.
    """

    out_df = []
    for gdir in gdirs:

        d = OrderedDict()

        # Easy stats
        d['rgi_id'] = gdir.rgi_id
        d['name'] = gdir.name
        d['cenlon'] = gdir.cenlon
        d['cenlat'] = gdir.cenlat
        d['rgi_area_km2'] = gdir.rgi_area_km2
        d['glacier_type'] = gdir.glacier_type
        d['terminus_type'] = gdir.terminus_type

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
        if gdir.has_file('inversion_flowlines', div_id=0):
            amb = np.array([])
            h = np.array([])
            widths = np.array([])
            slope = np.array([])
            fls = gdir.read_pickle('inversion_flowlines', div_id=0)
            dx = fls[0].dx * gdir.grid.dx
            for fl in fls:
                amb = np.append(amb, fl.apparent_mb)
                hgt = fl.surface_h
                h = np.append(h, hgt)
                widths = np.append(widths, fl.widths * dx)
                slope = np.append(slope, np.arctan(-np.gradient(hgt, dx)))

            pacc = np.where(amb >= 0)
            pab = np.where(amb < 0)
            d['aar'] = np.sum(widths[pacc]) / np.sum(widths[pab])
            mb_slope, _, _, _, _ = stats.linregress(h[pab], amb[pab])
            d['mb_grad'] = mb_slope
            d['avg_width'] = np.mean(widths)
            d['avg_slope'] = np.mean(slope)

        # Climate
        if gdir.has_file('climate_monthly', div_id=0):
            cds = xr.open_dataset(gdir.get_filepath('climate_monthly'))
            d['clim_alt'] = cds.ref_hgt
            t = cds.temp.mean(dim='time').values
            t = t - (d['dem_mean_elev'] - d['clim_alt']) * \
                cfg.PARAMS['temp_default_gradient']
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

        out_df.append(d)

    cols = list(out_df[0].keys())
    return pd.DataFrame(out_df, columns=cols).set_index('rgi_id')


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
        task_func.__doc__ = '\n'.join((task_func.__doc__, self.iodoc))

        @wraps(task_func)
        def _entity_task(gdir, **kwargs):
            # Log only if needed:
            if not task_func.__dict__.get('divide_task', False):
                self.log.info('%s: %s', gdir.rgi_id, task_func.__name__)

            # Run the task
            if cfg.CONTINUE_ON_ERROR:
                try:
                    out = task_func(gdir, **kwargs)
                    gdir.log(task_func)
                except Exception as err:
                    # Something happened
                    out = None
                    gdir.log(task_func, err=err)
                    pipe_log(gdir, task_func, err=err)
            else:
                out = task_func(gdir, **kwargs)
            return out
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
                the ID of the divide to process. Should be left to  the default
                ``None`` unless you know what you do.
        """

    def __call__(self, task_func):
        """Decorate."""

        @wraps(task_func)
        def _divide_task(gdir, div_id=None, **kwargs):
            if div_id is None:
                ids = gdir.divide_ids
                if self.add_0:
                    ids = [0] + list(ids)
                for i in ids:
                    self.log.info('%s: %s, divide %d', gdir.rgi_id,
                                  task_func.__name__, i)
                    task_func(gdir, div_id=i, **kwargs)
            else:
                # For testing only
                task_func(gdir, div_id=div_id, **kwargs)

        # For the logger later on
        _divide_task.__dict__['divide_task'] = True
        return _divide_task


class GlacierDirectory(object):
    """Organizes read and write access to the glacier's files.

    It handles a glacier directory created in a base directory (default
    is the "per_glacier" folder in the working directory). The role of a
    GlacierDirectory is to give access to file paths and to I/O operations
    in a transparent way. The user should not care about *where* the files are
    located, but should know their name (see :ref:`basenames`).

    A glacier entity has one or more divides. See :ref:`glacierdir`
    for more information.
    """

    def __init__(self, rgi_entity, base_dir=None, reset=False):
        """Creates a new directory or opens an existing one.

        Parameters
        ----------
        rgi_entity: glacier entity read from the shapefile  or a valid RGI ID
        base_dir: path to the directory where to open the directory
            defaults to "conf.PATHPATHS['working_dir'] + /per_glacier/"
        reset: emtpy the directory at construction (careful!)

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
        cenlon : float
            The glacier's RGI central longitude
        rgi_date : datetime
            The RGI's BGNDATE attribute if available. Otherwise, defaults to
            2003-01-01
        rgi_region : str
            The RGI region name
        name : str
            The RGI glacier name (if Available)
        """

        if base_dir is None:
            base_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')

        # RGI IDs are also valid entries
        if isinstance(rgi_entity, string_types):
            _shp = os.path.join(base_dir, rgi_entity, 'outlines.shp')
            rgi_entity = read_shapefile(_shp).iloc[0]

        try:
            # Assume RGI V4
            self.rgi_id = rgi_entity.RGIID
            self.glims_id = rgi_entity.GLIMSID
            self.rgi_area_km2 = float(rgi_entity.AREA)
            self.cenlon = float(rgi_entity.CENLON)
            self.cenlat = float(rgi_entity.CENLAT)
            self.rgi_region = rgi_entity.O1REGION
            self.name = rgi_entity.NAME
            rgi_datestr = rgi_entity.BGNDATE
            # this is for the tests which are still in V4
            self.glacier_type = 'Glacier'
            self.terminus_type = 'Land-terminating'
        except AttributeError:
            # Should be V5
            self.rgi_id = rgi_entity.RGIId
            self.glims_id = rgi_entity.GLIMSId
            self.rgi_area_km2 = float(rgi_entity.Area)
            self.cenlon = float(rgi_entity.CenLon)
            self.cenlat = float(rgi_entity.CenLat)
            self.rgi_region = rgi_entity.O1Region
            self.name = rgi_entity.Name
            rgi_datestr = rgi_entity.BgnDate

            # Read glacier attrs
            gt = rgi_entity.GlacType
            keys = {'0': 'Glacier',
                    '1': 'Ice cap',
                    '2': 'Perennial snowfield',
                    '3': 'Seasonal snowfield',
                    '9': 'Not assigned',
                    }

            self.glacier_type = keys[gt[0]]
            keys = {'0': 'Land-terminating',
                    '1': 'Marine-terminating',
                    '2': 'Lake-terminating',
                    '3': 'Dry calving',
                    '4': 'Regenerated',
                    '5': 'Shelf-terminating',
                    '9': 'Not assigned',
                    }
            self.terminus_type = keys[gt[1]]

        # convert the date
        try:
            rgi_date = pd.to_datetime(rgi_datestr[0:6],
                                      errors='raise', format='%Y%m')
        except:
            rgi_date = pd.to_datetime('200301', format='%Y%m')
        self.rgi_date = rgi_date

        self.dir = os.path.join(base_dir, self.rgi_id)
        if reset and os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # The divides dirs are created by gis.define_glacier_region

    @lazy_property
    def grid(self):
        """A ``salem.Grid`` handling the georeferencing of the local grid."""
        return self.read_pickle('glacier_grid')

    @lazy_property
    def rgi_area_m2(self):
        """The glacier's RGI area (m2)."""
        return self.rgi_area_km2 * 10**6

    @property
    def divide_dirs(self):
        """list of the glacier divides directories."""
        dirs = [self.dir] + list(glob.glob(os.path.join(self.dir, 'divide_*')))
        return dirs

    @property
    def n_divides(self):
        """Number of glacier divides."""
        return len(self.divide_dirs)-1

    @property
    def divide_ids(self):
        """Iterator over the glacier divides ids."""
        return range(1, self.n_divides+1)

    def get_filepath(self, filename, div_id=0):
        """Absolute path to a specific file.

        Parameters
        ----------
        filename: str
            file name (must be listed in cfg.BASENAME)
        div_id: int
            the divide for which you want to get the file path

        Returns
        -------
        The absolute path to the desired file
        """

        if filename not in cfg.BASENAMES:
            raise ValueError(filename + ' not in cfg.BASENAMES.')

        dir = self.divide_dirs[div_id]

        return os.path.join(dir, cfg.BASENAMES[filename])

    def has_file(self, filename, div_id=0):

        return os.path.exists(self.get_filepath(filename, div_id=div_id))

    def read_pickle(self, filename, div_id=0):
        """ Reads a pickle located in the directory.

        Parameters
        ----------
        filename: str
            file name (must be listed in cfg.BASENAME)
        div_id: int
            the divide for which you want to get the file path

        Returns
        -------
        An object read from the pickle
        """

        _open = gzip.open if cfg.PARAMS['use_compression'] else open
        with _open(self.get_filepath(filename, div_id), 'rb') as f:
            out = pickle.load(f)

        return out

    def write_pickle(self, var, filename, div_id=0):
        """ Writes a variable to a pickle on disk.

        Parameters
        ----------
        var: object
            the variable to write to disk
        filename: str
            file name (must be listed in cfg.BASENAME)
        div_id: int
            the divide for which you want to get the file path
        """

        _open = gzip.open if cfg.PARAMS['use_compression'] else open
        with _open(self.get_filepath(filename, div_id), 'wb') as f:
            pickle.dump(var, f, protocol=-1)

    def create_gridded_ncdf_file(self, fname, div_id=0):
        """Makes a gridded netcdf file template.

        The other variables have to be created and filled by the calling
        routine.

        Parameters
        ----------
        filename: str
            file name (must be listed in cfg.BASENAME)
        div_id: int
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

    def write_monthly_climate_file(self, time, prcp, temp, grad, hgt):
        """Creates a netCDF4 file with climate data.

        See :py:func:`oggm.tasks.distribute_climate_data`.
        """

        # overwrite as default
        fpath = self.get_filepath('climate_monthly')
        if os.path.exists(fpath):
            os.remove(fpath)

        with netCDF4.Dataset(fpath, 'w', format='NETCDF4') as nc:
            nc.ref_hgt = hgt

            dtime = nc.createDimension('time', None)

            nc.author = 'OGGM'
            nc.author_info = 'Open Global Glacier Model'

            timev = nc.createVariable('time','i4',('time',))
            timev.setncatts({'units':'days since 1801-01-01 00:00:00'})
            timev[:] = netCDF4.date2num([t for t in time],
                                 'days since 1801-01-01 00:00:00')

            v = nc.createVariable('prcp', 'f4', ('time',), zlib=True)
            v.units = 'kg m-2'
            v.long_name = 'total precipitation amount'
            v[:] = prcp

            v = nc.createVariable('temp', 'f4', ('time',), zlib=True)
            v.units = 'degC'
            v.long_name = '2m temperature at height ref_hgt'
            v[:] = temp

            v = nc.createVariable('grad', 'f4', ('time',), zlib=True)
            v.units = 'degC m-1'
            v.long_name = 'temperature gradient'
            v[:] = grad

    def log(self, func, err=None):

        fpath = os.path.join(self.dir, 'log')
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fpath = os.path.join(fpath, func.__name__)

        if err is not None:
            fpath += '.ERROR'
        else:
            fpath += '.SUCCESS'

        with open(fpath, 'w') as f:
            f.write(func.__name__ + '\n')
            if err is not None:
                f.write(err.__class__.__name__ + ': {}'.format(err))

