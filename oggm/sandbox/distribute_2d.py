import logging
import warnings

import os
import oggm
import oggm.cfg as cfg
from oggm import utils, workflow
from oggm.exceptions import InvalidWorkflowError
import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.stats import mstats
from oggm.core.gis import gaussian_blur, get_dem_for_grid, GriddedNcdfFile, process_dem
from oggm.utils import ncDataset, entity_task, global_task
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
        band_index = topo_data * np.nan  # container
        per_band_rank = topo_data * np.nan  # container
        distrib_thick = ds.distributed_thickness.data

    # For the flowline we need the model flowlines only
    fls = gdir.read_pickle('model_flowlines')
    assert len(fls) == 1, 'Only works with one flowline.'
    fl = fls[0]

    # number of pixels per band along flowline
    npix_per_band = fl.bin_area_m2 / (gdir.grid.dx ** 2)
    nnpix_per_band_cumsum = np.around(npix_per_band[::-1].cumsum()[::-1])

    # check if their is a differenze between the flowline area and gridded area
    pix_diff = int(nnpix_per_band_cumsum[0] - len(topo_data_flat))
    if pix_diff > 0:
        # we have more fl pixels, for this we can adapt a little
        # search for the largest differences between elevation bands and reduce
        # area for one pixel
        sorted_indices = np.argsort(np.abs(np.diff(nnpix_per_band_cumsum)))
        npix_per_band[sorted_indices[-pix_diff:]] = \
            npix_per_band[sorted_indices[-pix_diff:]] - 1
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
    # a penalty is added to higher elevation grid points
    min_alt = np.nanmin(topo_data)
    weighted_thick = ((topo_data - min_alt + 1) * elevation_weight) * distrib_thick
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


@entity_task(log, writes=['gridded_simulation'])
def distribute_thickness_from_simulation(gdir,
                                         input_filesuffix='',
                                         concat_input_filesuffix=None,
                                         output_filesuffix='',
                                         fl_diag=None,
                                         ys=None, ye=None,
                                         smooth_radius=None,
                                         add_monthly=False,
                                         fl_thickness_threshold=0,
                                         rolling_mean_smoothing=0,
                                         only_allow_retreating=False,
                                         debug_area_timeseries=False,
                                         concat_ds=None):
    """Redistributes the simulated flowline area and volume back onto the 2D grid.

    For this to work, the glacier cannot advance beyond its initial area! It
    wont fail, but it will redistribute mass at the bottom of the glacier.

    We assume that add_smoothed_glacier_topo and assign_points_to_band have
    been run before, and that the user stored the data from their simulation
    in a flowline diagnostics file (turned off per default).

    The algorithm simply melts each flowline band onto the
    2D grid points, but adds some heuristics (see :py:func:`assign_points_to_band`)
    as to which grid points melt faster. Currently it does not take elevation
    into account for the melt *within* one band, a downside which is somehow
    mitigated with smoothing (the default is quite some smoothing).

    Writes a new file called gridded_simulation.nc together with a new time
    dimension.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    input_filesuffix : str
        the filesuffix of the flowline diagnostics file.
    output_filesuffix : str
        the filesuffix of the gridded_simulation.nc file to write. If empty,
        it will be set to input_filesuffix.
    concat_input_filesuffix : str
        the filesuffix of the flowline diagnostics file to concat with the main
        one, provided with input_filesuffix. `concat_input_filesuffix` is
        assumed to be prior in time to the main one, i.e. often you will be
        calling `concat_input_filesuffix='_spinup_historical'`.
    fl_diag : xarray.core.dataset.Dataset
        directly provide a flowline diagnostics file instead of reading it
        from disk. This could be useful, for example, to merge two files
        before sending it to the algorithm, and if you want to smooth the time
        series. If provided, 'input_filesuffix' is only used to name
        the distributed data file to save.
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
        distribution process. Default is 0 (using no threshold). We recommend
        to set it to 1 if you are having artefacts in your visualizations.
    rolling_mean_smoothing : int
        If > 0, the area and volume of the flowline diagnostics will be
        smoothed using a rolling mean over time. The window size is defined
        with this number. We recommend 3, 5, or more (in extreme cases).
    only_allow_retreating : bool
        If True, the algorithm will adapt the flowline diagnostics data in the way
        that each bin can only shrink over time. If the simulated flowline's bin would
        actually gain mass/volume in a timestep, it stays unchanged when
        "only_allow_retreating" is set to True.
        This can prevent flickering in distributed animations.
    debug_area_timeseries : bool
        If True, the algorithm will return a dataframe additionally to the
        gridded dataset. The dataframe contains two columns: the original area
        timeseries, and the post-processed one (i.e. after applying the thickness
        threshold filter and the rolling mean). This is useful for finding the
        best smoothing parameters for the visualisation of your glacier.
    concat_ds = None
        set this to a dataset you want to concatenate to the newly computed
        one. The dataset should be prior in time to the one to compute, and
        should end when the new one starts. Note that this is only a convenience
        function, and won't work well if you use smoothing on the timeseries
        (for smoothing, use the `concat_input_filesuffix` or `fl_diag`
        keyword above).
    """

    if fl_diag is not None:
        dg = fl_diag
    else:
        fp = gdir.get_filepath('fl_diagnostics', filesuffix=input_filesuffix)
        with xr.open_dataset(fp) as dg:
            assert len(dg.flowlines.data) == 1, 'Only works with one flowline.'
        with xr.open_dataset(fp, group=f'fl_0') as dg:
            if concat_input_filesuffix is not None:
                fp0 = gdir.get_filepath('fl_diagnostics',
                                        filesuffix=concat_input_filesuffix)
                with xr.open_dataset(fp0, group=f'fl_0') as dg0:
                    dg0 = dg0.load()
                    if dg0.time[-1] != dg.time[0]:
                        raise InvalidWorkflowError(f'The two dataset times dont match: '
                                                   f'{float(dg0.time[-1])} vs '
                                                   f'{float(dg.time[0])}.')
                    # Only include thickness, area, and volume to avoid errors
                    # with variables not present in both.
                    vars_of_interest = ['thickness_m', 'area_m2', 'volume_m3']
                    dg = xr.concat([dg0[vars_of_interest],
                                    dg[vars_of_interest].isel(time=slice(1, None))],
                                   dim='time')
            if ys is not None or ye is not None:
                dg = dg.sel(time=slice(ys, ye))
            dg = dg.load()

    if not output_filesuffix:
        output_filesuffix = input_filesuffix

    # save the original area evolution for the area plot
    if debug_area_timeseries:
        out_df = dg['area_m2'].sum(dim='dis_along_flowline').to_dataframe(name='initial_area')

    # applying the thickness threshold
    dg = xr.where(dg['thickness_m'] < fl_thickness_threshold, 0, dg)

    # applying the only retreating algorithm
    if only_allow_retreating:
        # stay in the loop as long as there is a glacier growing
        for _ in range(len(dg['time'])):
            # get time_steps where thickness is increasing
            mask = dg.thickness_m.diff(dim='time') > 0
            # add False for the first time step to keep the same dimensions
            mask = xr.concat([xr.full_like(dg.thickness_m.isel(time=0),
                                           False),
                              mask],
                             dim='time')
            if mask.any():
                # for each increasing time-step use the past time-step
                dg = xr.where(mask, dg.shift(time=1), dg)
            else:
                # if nothing is increasing we are done
                break

    # applying the rolling mean smoothing
    if rolling_mean_smoothing:
        dg[['area_m2', 'volume_m3']] = dg[['area_m2', 'volume_m3']].rolling(
            min_periods=1, time=rolling_mean_smoothing, center=True).mean()

    # monthly interpolation for higher temporal resolution
    if add_monthly:
        # create new monthly time coordinate, last year only with month 1
        monthly_time = utils.monthly_timeseries(dg.time[0], dg.time[-1])
        yrs, months = utils.floatyear_to_date(monthly_time)
        # interpolate and add years and months as new coords
        dg = dg[['area_m2', 'volume_m3']].interp(time=monthly_time,
                                                 method='linear')
    else:
        yrs, months = utils.floatyear_to_date(dg.time)

    if debug_area_timeseries:
        out_df['smoothed_area'] = dg['area_m2'].sum(dim='dis_along_flowline').to_series()

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

        area_cov = 0
        new_thick = out_thick[i, :]
        for band_id, npix in zip(band_ids.astype(int), counts):
            band_area = dgy.area_m2.values[band_id]
            band_volume = dgy.volume_m3.values[band_id]
            if ~np.isclose(band_area, 0):
                # We have some ice left
                pix_cov = band_area / dx2
                mask = (band_index_mask == band_id) & \
                       (rank_per_band >= (npix - pix_cov))
                vol_orig = np.where(mask, orig_distrib_thick, 0).sum() * dx2
                area_dis = mask.sum() * dx2
                thick_cor = (vol_orig - band_volume) / area_dis
                area_cov += area_dis
                new_thick[mask] = orig_distrib_thick[mask] - thick_cor
                # Make sure all glacier covered cells have the minimum thickness
                new_thick[mask] = utils.clip_min(new_thick[mask], 1)

        this_glacier_mask = new_thick > 0
        this_vol = dgy.volume_m3.values.sum()

        if np.isclose(this_vol, 0, atol=1e-6):
            # No ice left
            new_thick = np.nan
        else:
            # Smooth
            dx = gdir.grid.dx
            if smooth_radius != 0:
                if smooth_radius is None:
                    smooth_radius = np.rint(cfg.PARAMS['smooth_window'] / dx)
                new_thick = gaussian_blur(new_thick, int(smooth_radius))

            new_thick[~this_glacier_mask] = np.nan

            # Conserve volume
            tmp_vol = np.nansum(new_thick) * dx2
            new_thick *= this_vol / tmp_vol

        out_thick[i, :] = new_thick

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds['bedrock'] = ds['topo'] - ds['distributed_thickness'].fillna(0)
        ds = ds[['glacier_mask', 'topo', 'bedrock']].load()

    ds.coords['time'] = dg['time']

    vn = "simulated_thickness"
    ds[vn] = (('time', 'y', 'x',), out_thick)
    ds.coords['calendar_year'] = ('time', yrs)
    ds.coords['calendar_month'] = ('time', months)

    if concat_ds is not None:
        if concat_ds.time[-1] != ds.time[0]:
            raise InvalidWorkflowError(f'The two dataset times dont match: '
                                       f'{concat_ds.time[-1]} vs {ds.time[0]}.')
        ds = xr.concat([concat_ds, ds.isel(time=(1, None))], dim='time')

    ds.to_netcdf(gdir.get_filepath('gridded_simulation',
                                   filesuffix=output_filesuffix))

    if debug_area_timeseries:
        return ds, out_df

    return ds


@global_task(log)
def merge_simulated_thickness(gdirs,
                              output_folder=None,
                              output_filename=None,
                              simulation_filesuffix='',
                              years_to_merge=None,
                              keep_dem_file=False,
                              interp='nearest',
                              preserve_totals=True,
                              smooth_radius=None,
                              use_glacier_mask=True,
                              add_topography=True,
                              use_multiprocessing=False,
                              save_as_multiple_files=True,
                              reset=False):
    """
    This function is a wrapper for workflow.merge_gridded_data when one wants
    to merge the distributed thickness after a simulation. It adds all
    variables needed for a '3D-visualisation'. The bedrock topography is
    recalculated for the combined grid. For more information about the
    parameters see the docstring of workflow.merge_gridded_data.

    Parameters
    ----------
    gdirs
    output_folder
    output_filename : str
        Default is gridded_simulation_merged{simulation_filesuffix}.
    simulation_filesuffix : str
        the filesuffix of the gridded_simulation file
    years_to_merge : list | xr.DataArray | None
        If not None, and list of integers only the years at month 1 are
        merged. You can also provide the timesteps as xarray.DataArray
        containing calendar_year and calendar_month as it is the standard
        output format for gridded nc files of OGGM.
        Default is None.
    keep_dem_file
    interp
    add_topography
    use_glacier_mask
    preserve_totals
    smooth_radius
    use_multiprocessing : bool
        If we use multiprocessing the merging is done in parallel, but requires
        more memory. Default is True.
    save_as_multiple_files : bool
        If True, the merged simulated-thickness data is saved as multiple
        files, one for each year of simulation to avoid memory overflow
        problems. In this case the other data like glacier mask, glacier
        extent, bedrock topography, etc. are saved in a separate file (with
        suffix _topo_data). If False, all data is saved in one file.
        Default is True.
    reset
    """
    if output_filename is None:
        output_filename = f'gridded_simulation_merged{simulation_filesuffix}'

    if output_folder is None:
        output_folder = cfg.PATHS['working_dir']

    # function for calculating the bedrock topography
    def _calc_bedrock_topo(fp):
        with xr.open_dataset(fp) as ds:
            ds = ds.load()
        ds['bedrock'] = (ds['topo'] - ds['distributed_thickness'].fillna(0))
        ds.to_netcdf(fp)

    if save_as_multiple_files:
        # first the _topo_data file
        workflow.merge_gridded_data(
            gdirs,
            output_folder=output_folder,
            output_filename=f"{output_filename}_topo_data",
            input_file='gridded_data',
            input_filesuffix='',
            included_variables=['glacier_ext',
                                'glacier_mask',
                                'distributed_thickness',
                                ],
            preserve_totals=preserve_totals,
            smooth_radius=smooth_radius,
            use_glacier_mask=use_glacier_mask,
            add_topography=add_topography,
            keep_dem_file=keep_dem_file,
            interp=interp,
            use_multiprocessing=use_multiprocessing,
            return_dataset=False,
            reset=reset)

        # recalculate bed topography after reprojection, if topo was added
        if add_topography:
            fp = os.path.join(output_folder,
                              f"{output_filename}_topo_data.nc")
            _calc_bedrock_topo(fp)

        # then the simulated thickness files
        if years_to_merge is None:
            # open first file to get all available timesteps
            with xr.open_dataset(
                    gdirs[0].get_filepath('gridded_simulation',
                                          filesuffix=simulation_filesuffix)
            ) as ds:
                years_to_merge = ds.time

        for timestep in years_to_merge:
            if isinstance(timestep, int) or isinstance(timestep, np.int64):
                year = timestep
                month = 1
            elif isinstance(timestep, xr.DataArray):
                year = int(timestep.calendar_year)
                month = int(timestep.calendar_month)
            else:
                raise NotImplementedError('Wrong type for years_to_merge! '
                                          'Should be list of int or '
                                          'xarray.DataArray for monthly '
                                          'timesteps.')

            workflow.merge_gridded_data(
                gdirs,
                output_folder=output_folder,
                output_filename=f"{output_filename}_{year}_{month:02d}",
                input_file='gridded_simulation',
                input_filesuffix=simulation_filesuffix,
                included_variables=[('simulated_thickness',
                                     {'time': [timestep]})],
                preserve_totals=preserve_totals,
                smooth_radius=smooth_radius,
                use_glacier_mask=use_glacier_mask,
                add_topography=False,
                keep_dem_file=False,
                interp=interp,
                use_multiprocessing=use_multiprocessing,
                return_dataset=False,
                reset=reset)

    else:
        # here we save everything in one file
        if years_to_merge is None:
            selected_time = None
        else:
            selected_time = {'time': years_to_merge}

        workflow.merge_gridded_data(
            gdirs,
            output_folder=output_folder,
            output_filename=output_filename,
            input_file=['gridded_data', 'gridded_simulation'],
            input_filesuffix=['', simulation_filesuffix],
            included_variables=[['glacier_ext',
                                 'glacier_mask',
                                 'distributed_thickness',
                                 ],
                                [('simulated_thickness', selected_time)]],
            preserve_totals=preserve_totals,
            smooth_radius=smooth_radius,
            use_glacier_mask=use_glacier_mask,
            add_topography=add_topography,
            keep_dem_file=keep_dem_file,
            interp=interp,
            use_multiprocessing=use_multiprocessing,
            return_dataset=False,
            reset=reset)

        # recalculate bed topography after reprojection, if topo was added
        if add_topography:
            fp = os.path.join(output_folder, f'{output_filename}.nc')
            _calc_bedrock_topo(fp)
