import logging
import warnings

import oggm.cfg as cfg
from oggm import utils
import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.stats import mstats
from oggm.core.gis import gaussian_blur
from oggm.utils import ncDataset, entity_task
import matplotlib.pyplot as plt

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
def add_smoothed_glacier_topo(gdir, outline_offset=-40,
                              smooth_radius=1):
    """Smooth the glacier topography while ignoring surrounding slopes.

    It is different from the smoothing that occurs in the 'process_dem'
    function, that generates 'topo_smoothed' in the gridded_data.nc file.

    This is of importance when redistributing thicknesses to the 2D grid,
    because the sides of a glacier tongue can otherwise be higher than the
    tongue middle, resulting in an odd shape once the glacier retreats
    (e.g. with the middle of the tongue retreating faster that the edges).

    Write the `glacier_topo_smoothed` variable in `gridded_data.nc`.

    Source: https://stackoverflow.com/a/61481246/4057931

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
    smooth_radius : int, optional
        the gaussian radius. One comment on StackOverflow indicates that it
        may not work well for other values than 1. To investigate.
    """

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        raw_topo = xr.where(ds.glacier_mask == 1, ds.topo, np.nan)
        if outline_offset is not None:
            raw_topo += ds.glacier_ext * outline_offset
        raw_topo = raw_topo.data

    smooth_glacier = filter_nan_gaussian_conserving(raw_topo, smooth_radius)
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
def assign_points_to_band(gdir, topo_variable='glacier_topo_smoothed',
                          elevation_weight=1.003):
    """Assigns glacier grid points to flowline elevation bands and ranks them.

    Creates two variables in gridded_data.nc:

    `band_index`, which assigns one number per grid point (the index to
    which band this grid point belongs). This ordering is done to preserve
    area by elevation, i.e. point elevation does matter, but not strictly.
    What is more important is the "rank" by elevation, i.e. each flowline band
    has the correct gridded area.

    `rank_per_band`, which assigns another index per grid point: the
    "rank" withing a band, from thinner to thicker and from bottom to top.
    This rank indicates which grid point will melt faster within a band.
    There is one aibtrary parameter here, which is by how much to weight the
    elevation factor (see Parameters)

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    topo_variable : str
        the topography to read from `gridded_data.nc` (could be smoothed, or
        smoothed differently).
    elevation_weight : float
        how much weight to give to the elevation of the grid point versus the
        thickness. Arbitrary number, might be tuned differently.
    """
    # We need quite a few data from the gridded dataset
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        topo_data = ds[topo_variable].data.copy()
        glacier_mask = ds.glacier_mask.data == 1
        topo_data_flat = topo_data[glacier_mask]
        band_index = topo_data * np.NaN  # container
        per_band_rank = topo_data * np.NaN  # container
        distrib_thick = ds.distributed_thickness.data

    # For the flowline we need the model flowlines only
    fls = gdir.read_pickle('model_flowlines')
    assert len(fls) == 1, 'Only works with one flowline.'
    fl = fls[0]

    # number of pixels per band along flowline
    npix_per_band = fl.bin_area_m2 / (gdir.grid.dx ** 2)
    nnpix_per_band_cumsum = np.around(npix_per_band[::-1].cumsum()[::-1])

    rank_elev = mstats.rankdata(topo_data_flat)
    bins = nnpix_per_band_cumsum[nnpix_per_band_cumsum > 0].copy()
    bins[0] = len(topo_data_flat) + 1
    band_index[glacier_mask] = np.digitize(rank_elev, bins, right=False) - 1

    # Some sanity checks for now
    # Area gridded and area flowline should be similar
    assert np.allclose(nnpix_per_band_cumsum.max(), len(topo_data_flat), rtol=0.1)
    # All bands should have pixels in them
    # Below not allways working - to investigate
    # rgi_ids = ['RGI60-11.03887']  # This is Marmlolada
    # base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/
    # L3-L5_files/2023.1/elev_bands/W5E5'
    # assert np.nanmax(band_index) == len(bins) - 1
    # assert np.nanmin(band_index) == 0
    assert np.all(np.isfinite(band_index[glacier_mask]))

    # Ok now assign within band using ice thickness weighted by elevation
    # We rank the pixels within one band by elevation, but also add
    # a penaltly is added to higher elevation grid points
    min_alt = np.nanmin(topo_data)
    weighted_thick = ((topo_data - min_alt + 1) * 1.003) * distrib_thick
    for band_id in np.unique(np.sort(band_index[glacier_mask])):
        # We work per band here
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
        v.description = ('Points grouped by band along the flowline, '
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
def distribute_thickness_from_simulation(gdir, input_filesuffix='',
                                         ys=None, ye=None,
                                         smooth_radius=None,
                                         add_monthly=False,
                                         fl_thickness_threshold=0,
                                         rolling_mean_smoothing=False,
                                         show_area_plot=False):
    """Redistributes the simulated flowline area and volume back onto the 2D grid.

    For this to work, the glacier cannot advance beyond its initial area!

    We assume that add_smoothed_glacier_topo and assign_points_to_band have
    been run before, and that the user stored the data from their simulation
    in a flowline diagnostics file (turned off per default).

    The algorithm simply melts each flowline band onto the
    2D grid points, but adds some heuristics (see :py:func:`assign_points_to_band`)
    as to which grid points melts faster. Currently it does not take elevation
    into account for the melt *within* one band, a downside which is somehow
    mitigated with smoothing (the default is quite some smoothing).

    Writes a new variable to gridded_data.nc (simulation_distributed_thickness)
    together with a new time dimension. If a variable already exists we
    will try to concatenate.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    input_filesuffix : str
        the filesuffix of the flowline diagnostics file.
    ys : int
        pick another year to start the series (default: the first year
        of the diagnostic file)
    ye : int
        pick another year to end the series (default: the first year
        of the diagnostic file)
    smooth_radius : int
        pixel size of the gaussian smoothing. Default is to use
        cfg.PARAMS['smooth_window'] (i.e. a size in meters). Set to zero to
        suppress smoothing.
    add_monthly : bool
        If True the yearly flowline diagnostics will be linearly interpolated
        to a monthly resolution. Default is False
    fl_thickness_threshold : float
        A minimum threshold (all values below the threshold are set to 0) is
        applied to the area and volume of the flowline diagnostics before the
        distribution process. Default is 0 (using no threshold).
    rolling_mean_smoothing : bool or int
        If int, the area and volume of the flowline diagnostics will be
        smoothed using a rolling mean. The window size is defined with this
        number. If True a default window size of 3 is used. If False no
        smoothing is applied.  Default is False.
    show_area_plot : bool
        If True, only a plot of the 'original-total-area- evolution' and the
        'smoothed-total-area- evolution' (after using fl_thickness_threshold
        and rolling_mean_smoothing) is returned. This is useful for finding the
        best smoothing parameters for the visualisation of your glacier.
    """

    fp = gdir.get_filepath('fl_diagnostics', filesuffix=input_filesuffix)
    with xr.open_dataset(fp) as dg:
        assert len(dg.flowlines.data) == 1, 'Only works with one flowline.'
    with xr.open_dataset(fp, group=f'fl_0') as dg:
        if ys or ye:
            dg = dg.sel(time=slice(ye, ye))
        dg = dg.load()

    # save the original area evolution for the area plot
    if show_area_plot:
        area_evolution_orig = dg['area_m2'].sum(dim='dis_along_flowline')

    # applying the thickness threshold
    dg = xr.where(dg['thickness_m'] < fl_thickness_threshold, 0, dg)

    # applying the rolling mean smoothing
    if rolling_mean_smoothing:
        if isinstance(rolling_mean_smoothing, bool):
            rolling_mean_smoothing = 3

        dg[['area_m2', 'volume_m3']] = dg[['area_m2', 'volume_m3']].rolling(
            min_periods=1, time=rolling_mean_smoothing, center=True).mean()

    # monthly interpolation for higher temporal resolution
    if add_monthly:
        # create new monthly time coordinate, last year only with month 1
        years = np.append(np.repeat(dg.time[:-1], 12),
                          dg.time[-1])
        months = np.append(np.tile(np.arange(1, 13), len(dg.time[:-1])),
                           1)
        time_monthly = utils.date_to_floatyear(years, months)

        # interpolate and add years and months as new coords
        dg = dg[['area_m2', 'volume_m3']].interp(time=time_monthly,
                                                 method='linear')
        dg.coords['calender_year'] = ('time', years)
        dg.coords['calender_month'] = ('time', months)

    if show_area_plot:
        area_evolution_orig.plot(label='original')
        dg['area_m2'].sum(dim='dis_along_flowline').plot(label='smoothed')
        plt.legend()
        plt.show()
        return None

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        band_index_mask = ds.band_index.data
        rank_per_band = ds.rank_per_band.data
        glacier_mask = ds.glacier_mask.data == 1
        orig_distrib_thick = ds.distributed_thickness.data

    band_ids, counts = np.unique(np.sort(band_index_mask[glacier_mask]),
                                 return_counts=True)

    dx2 = gdir.grid.dx**2
    out_thick = np.zeros((len(dg.time), *glacier_mask.shape))
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
                mask = (band_index_mask == band_id) & \
                       (rank_per_band >= (npix - pix_cov))
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

    # the distinction between time_monthly and time is needed to have data in
    # yearly AND monthly resolution in the same gridded data, maybe at one
    # point we decide for one option
    if add_monthly:
        time_var = 'time_monthly'
    else:
        time_var = 'time'
    ds.coords[time_var] = dg['time'].data
    vn = "simulation_distributed_thickness" + input_filesuffix
    if vn in ds:
        warnings.warn(f'Overwriting existing variable {vn}')
    ds[vn] = ((time_var, 'y', 'x',), out_thick)
    if add_monthly:
        ds.coords['calender_year_monthly'] = (time_var, dg.calender_year.data)
        ds.coords['calender_month_monthly'] = (time_var, dg.calender_month.data)
    ds.to_netcdf(gdir.get_filepath('gridded_data'))
