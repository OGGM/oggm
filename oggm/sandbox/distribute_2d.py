import logging
import oggm.cfg as cfg
from oggm import tasks, graphics, utils, workflow
from oggm.core import flowline
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import scipy
from scipy import ndimage
from scipy.stats import mstats
import os
import matplotlib.pyplot as plt
import salem
# This is needed to display graphics calculated outside of jupyter notebook
from IPython.display import HTML, display
from oggm.core.gis import gaussian_blur
from oggm import global_tasks
from oggm.core import massbalance
from oggm import graphics
import warnings


import shapely.geometry as shpg
from scipy.ndimage import binary_erosion,distance_transform_edt
from oggm.utils import ncDataset, entity_task

# Module logger
log = logging.getLogger(__name__)


def filter_nan_gaussian_conserving(arr, sigma=1):
    """Apply a gaussian filter to an array with nans.

    Source: https://stackoverflow.com/a/61481246/4057931

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution. All nans in arr, stay nans in output.

    One comment on StackOverflow indicates that it may not work as well
    for other values of sigma. To investigate
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    return gauss + loss * arr


@entity_task(log, writes=['gridded_data'])
def smooth_glacier_topo(gdir, outline_offset=-40):
    """Smooth the glacier topography while ignoring surrounding slopes.

    It is different from the smoothing that occurs in the 'process_dem'
    function, that generates 'topo_smoothed' in the gridded_data.nc file.

    This is of importance when extrapolation to the 2D grid, because the sides
    of a glacier tongue can otherwise be higher than the tongue middle,
    resulting in an odd shape once the glacier retreats (e.g. with the
    middle of the tongue retreating faster that the edges).

    Write the `glacier_topo_smoothed` variable in `gridded_data.nc`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    outline_offset : float, optional
        add an offset to the topography on the glacier outline mask. This
        allows to obtain better looking results because topography on the
        outline is often affected by surrounding slopes, and artificially
        reducing its elevation before smoothing makes the border pixel melt
        earlier. -40 seems good, but it may depend on the glacier.
    """

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        raw_topo = xr.where(ds.glacier_mask == 1, ds.topo, np.nan)
        if outline_offset is not None:
            raw_topo += ds.glacier_ext * outline_offset
        raw_topo = raw_topo.data

    smooth_glacier = filter_nan_gaussian_conserving(raw_topo, 1)
    with ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        vn = 'glacier_topo_smoothed'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',))
        v.units = 'm'
        v.long_name = 'Glacier topo smoothed'
        v.description = ("DEM smoothed just on the glacier. The DEM outside "
                         "the glacier doesn't impact the smoothing.")
        v[:] = smooth_glacier


@entity_task(log, writes=['gridded_data'])
def assign_points_to_band(gdir, fl_diagnostics_filesuffix='',
                          topo_variable='glacier_topo_smoothed'):
    """Assigns the points on the grid to the different elevation bands.


    """

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        topo_data = ds[topo_variable].data.copy()
        glacier_mask = ds.glacier_mask.data == 1
        topo_data_flat = topo_data[glacier_mask]
        band_index = topo_data * np.NaN  # container
        per_band_rank = topo_data * np.NaN  # container
        distrib_thick = ds.distributed_thickness.data

    fp = gdir.get_filepath('fl_diagnostics', filesuffix=fl_diagnostics_filesuffix)
    with xr.open_dataset(fp) as dg:
        assert len(dg.flowlines.data) == 1, 'Only works with one flowline.'
    with xr.open_dataset(fp, group=f'fl_0') as dg:
        dg = dg.isel(time=0).load()

    # number of pixels in section along flowline
    npix_per_band = dg.area_m2 / (gdir.grid.dx ** 2)
    nnpix_per_band_cumsum = np.around(npix_per_band.values[::-1].cumsum()[::-1])

    if not np.allclose(nnpix_per_band_cumsum.max(), len(topo_data_flat), rtol=0.1):
        raise RuntimeError('More than 10% difference in area - this wont work well')

    rank_elev = mstats.rankdata(topo_data_flat)
    bins = nnpix_per_band_cumsum[nnpix_per_band_cumsum > 0].copy()
    bins[0] = len(topo_data_flat) + 1
    band_index[glacier_mask] = np.digitize(rank_elev, bins, right=False) - 1

    assert np.nanmax(band_index) == len(bins) - 1
    assert np.nanmin(band_index) == 0
    assert np.all(np.isfinite(band_index[glacier_mask]))

    # Ok now assign within band using thickness weighted by elevation
    min_alt = np.nanmin(topo_data)
    weighted_thick = ((topo_data - min_alt + 1) * 1.003) * distrib_thick
    for band_id in np.unique(np.sort(band_index[glacier_mask])):
        is_band = band_index == band_id
        per_band_rank[is_band] = mstats.rankdata(weighted_thick[is_band])

    with ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        vn = 'band_index'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',))
        v.units = '-'
        v.long_name = 'Points grouped by band along the flowline'
        v.description = ('Points grouped by section along the flowline, '
                         'ordered from top to bottom.')
        v[:] = band_index

        vn = 'rank_per_band'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',))
        v.units = '-'
        v.long_name = 'Points ranked by thickness and elevation within band'
        v.description = ('Points ranked by thickness and elevation within each '
                         'band.')
        v[:] = per_band_rank


@entity_task(log, writes=['gridded_data'])
def distribute_thickness_from_simulation(gdir, ys=None, ye=None,
                                         fl_diagnostics_filesuffix='',
                                         smooth_radius=None):
    """This function redistributes the simulated glacier area and volume along
    the elevation band flowline back onto
    a 2D grid. In order for this to work, the glacier can not advance beyond
    its initial state, the glacier diagnostics
    during the simulation have to be stored, the gridded_data file needs to
    contain both the points_by_section and
    points_ranked_within_section variables.

    ys:  the first year to be redistributed
    ye:  the last year to be redistributed """

    fp = gdir.get_filepath('fl_diagnostics', filesuffix=fl_diagnostics_filesuffix)
    with xr.open_dataset(fp) as dg:
        assert len(dg.flowlines.data) == 1, 'Only works with one flowline.'
    with xr.open_dataset(fp, group=f'fl_0') as dg:
        if ys or ye:
            dg = dg.sel(time=slice(ye, ye))
        dg = dg.load()

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        band_index_mask = ds.band_index.data
        rank_per_band = ds.rank_per_band.data
        glacier_mask = ds.glacier_mask.data == 1
        orig_distrib_thick = ds.distributed_thickness.data

    band_ids, counts = np.unique(np.sort(band_index_mask[glacier_mask]), return_counts=True)

    dx2 = gdir.grid.dx**2
    out_thick = np.empty((len(dg.time), *glacier_mask.shape))
    for i, yr in enumerate(dg.time):

        dgy = dg.sel(time=yr)

        residual_pix = 0
        area_cov = 0
        new_thick = out_thick[i, :]
        for band_id, npix in zip(band_ids.astype(int), counts):
            band_area = dgy.area_m2.values[band_id]
            band_volume = dgy.volume_m3.values[band_id]
            if band_area != 0:
                # We have some ice left
                pix_cov = (band_area / dx2) + residual_pix
                mask = (band_index_mask == band_id) & (rank_per_band >= (npix - pix_cov))
                residual_pix = pix_cov - mask.sum()
                vol_orig = np.where(mask, orig_distrib_thick, 0).sum() * dx2
                area_dis = mask.sum() * dx2
                thick_cor = (vol_orig - band_volume) / area_dis
                area_cov += area_dis
                new_thick[mask] = orig_distrib_thick[mask] - thick_cor
                # Make sure all glacier covered cells have the minimum thickness
                new_thick[mask] = utils.clip_min(new_thick[mask], 1)

        this_glacier_mask = new_thick > 0
        this_vol = dgy.volume_m3.values.sum()

        # Smooth
        dx = gdir.grid.dx
        if smooth_radius != 0:
            if smooth_radius is None:
                smooth_radius = np.rint(cfg.PARAMS['smooth_window'] / dx)
            new_thick = gaussian_blur(new_thick, int(smooth_radius))
            new_thick[~this_glacier_mask] = np.NaN

        # Conserve volume
        tmp_vol = np.nansum(new_thick) * dx2
        new_thick *= this_vol / tmp_vol
        out_thick[i, :] = new_thick

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()

    ds['time'] = dg['time']
    vn = "simulation_distributed_thickness"
    ds[vn] = (('time', 'y', 'x',), out_thick)
    ds.to_netcdf(gdir.get_filepath('gridded_data'))
