"""Climate data and mass-balance computations"""
from __future__ import division

# Built ins
import logging
import os
import datetime
# External libs
import numpy as np
import pandas as pd
import netCDF4
# Locals
from oggm import cfg
from oggm import utils
from oggm import entity_task, divide_task

# Module logger
log = logging.getLogger(__name__)


def distribute_climate_data(gdirs):
    """Reads the Histalp climate data and distributes to each glacier.

    Generates a NetCDF file in the root glacier directory (climate_monthly.nc)
    It contains the timeseries of temperature, temperature gradient, and
    precipitation at the nearest grid point. The climate data reference height
    is provided as global attribute.

    Not to be multi-processed.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    log.info('distribute_climate_data')

    # read the file and data entirely (faster than many I/O)
    ncpath = cfg.PATHS['histalp_file']
    nc = netCDF4.Dataset(ncpath, mode='r')
    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]

    # Time
    time = nc.variables['time']
    time = netCDF4.num2date(time[:], time.units)
    ny, r = divmod(len(time), 12)
    if r != 0:
        raise ValueError('Climate data should be N full years exclusively')

    # Units
    assert nc.variables['hgt'].units == 'm'
    assert nc.variables['temp'].units == 'degC'
    assert nc.variables['prcp'].units == 'kg m-2'

    # Gradient defaults
    def_grad = -0.0065
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
    sf = cfg.PARAMS['prcp_scaling_factor']

    for gdir in gdirs:
        ilon = np.argmin(np.abs(lon - gdir.cenlon))
        ilat = np.argmin(np.abs(lat - gdir.cenlat))
        iprcp, itemp, igrad, ihgt = utils.joblib_read_climate(ncpath, ilon,
                                                              ilat, def_grad,
                                                              g_minmax,
                                                              sf)
        gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt)
    nc.close()


def mb_climate_on_height(gdir, heights, time_range=None, year_range=None):
    """Mass-balance climate of the glacier at a specific height

    Reads the glacier's monthly climate data file and computes the
    temperature "energies" (temp above 0) and solid precipitation at the
    required height.

    Parameters:
    -----------
    gdir: the glacier directory
    heights: a 1D array of the heights (in meter) where you want the data
    time_range (optional): default is to read all data but with this you
    can provide a [datetime, datetime] bounds (inclusive).
    year_range (optional): maybe more useful than the time bounds above.
    Provide a [y0, y1] year range to get the data for specific (hydrological)
    years only

    Returns:
    --------
    (time, tempformelt, prcpsol)::
        - time: array of shape (nt,)
        - tempformelt:  array of shape (len(heights), nt)
        - prcpsol:  array of shape (len(heights), nt)
    """

    if year_range is not None:
        t0 = datetime.datetime(year_range[0]-1, 10, 1)
        t1 = datetime.datetime(year_range[1], 9, 1)
        return mb_climate_on_height(gdir, heights, time_range=[t0, t1])

    # Parameters
    temp_all_solid = cfg.PARAMS['temp_all_solid']
    temp_all_liq = cfg.PARAMS['temp_all_liq']
    temp_melt = cfg.PARAMS['temp_melt']

    # Read file
    nc = netCDF4.Dataset(gdir.get_filepath('climate_monthly'), mode='r')

    # time
    time = nc.variables['time']
    time = netCDF4.num2date(time[:], time.units)
    if time_range is not None:
        p0 = np.where(time == time_range[0])[0]
        try:
            p0 = p0[0]
        except IndexError:
            raise RuntimeError('time_range[0] not found in file')
        p1 = np.where(time == time_range[1])[0]
        try:
            p1 = p1[0]
        except IndexError:
            raise RuntimeError('time_range[1] not found in file')
    else:
        p0 = 0
        p1 = len(time)-1

    time = time[p0:p1+1]

    # Read timeseries
    itemp = nc.variables['temp'][p0:p1+1]
    iprcp = nc.variables['prcp'][p0:p1+1]
    igrad = nc.variables['grad'][p0:p1+1]
    ref_hgt = nc.ref_hgt
    nc.close()

    # For each height pixel:
    # Compute temp and tempformelt (temperature above melting threshold)
    npix = len(heights)
    grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
    grad_temp *= (heights.repeat(len(time)).reshape(grad_temp.shape) - ref_hgt)
    temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
    temp2dformelt = temp2d - temp_melt
    temp2dformelt = np.clip(temp2dformelt, 0, temp2dformelt.max())
    # Compute solid precipitation from total precipitation
    prcpsol = np.atleast_2d(iprcp).repeat(npix, 0)
    fac = 1 - (temp2d - temp_all_solid) / (temp_all_liq - temp_all_solid)
    fac = np.clip(fac, 0, 1)
    prcpsol = prcpsol * fac

    return time, temp2dformelt, prcpsol


def mb_yearly_climate_on_height(gdir, heights, year_range=None, flatten=False):
    """Yearly mass-balance climate of the glacier at a specific height

    Parameters:
    -----------
    gdir: the glacier directory
    heights: a 1D array of the heights (in meter) where you want the data
    year_range (optional): a [y0, y1] year range to get the data for specific (
    hydrological) years only
    flatten: for some applications (glacier average MB) it's ok to flatten  the
    data (average over height) prior to annual summing.

    Returns:
    --------
    (years, tempformelt, prcpsol)::
        - years: array of shape (ny,)
        - tempformelt:  array of shape (len(heights), ny) (or ny if flatten
        is set)
        - prcpsol:  array of shape (len(heights), ny) (or ny if flatten
        is set)
    """

    time, temp, prcp = mb_climate_on_height(gdir, heights,
                                            year_range=year_range)

    ny, r = divmod(len(time), 12)
    if r != 0:
        raise ValueError('Climate data should be N full years exclusively')
    # Last year gives the tone of the hydro year
    years = np.arange(time[-1].year-ny+1, time[-1].year+1, 1)

    if flatten:
        # Spatial average
        temp_yr = np.zeros(len(years))
        prcp_yr = np.zeros(len(years))
        temp = np.mean(temp, axis=0)
        prcp = np.mean(prcp, axis=0)
        for i, y in enumerate(years):
            temp_yr[i] = np.sum(temp[i*12:(i+1)*12])
            prcp_yr[i] = np.sum(prcp[i*12:(i+1)*12])
    else:
        # Annual prcp and temp for each point (no spatial average)
        temp_yr = np.zeros((len(heights), len(years)))
        prcp_yr = np.zeros((len(heights), len(years)))
        for i, y in enumerate(years):
            temp_yr[:, i] = np.sum(temp[:, i*12:(i+1)*12], axis=1)
            prcp_yr[:, i] = np.sum(prcp[:, i*12:(i+1)*12], axis=1)

    return years, temp_yr, prcp_yr


def mb_yearly_climate_on_glacier(gdir, div_id=None, year_range=None):
    """Yearly mass-balance climate at all glacier heights,
    multiplied with the flowlines widths. (all in pix coords.)

    Parameters:
    -----------
    gdir: the glacier directory
    year_range (optional): a [y0, y1] year range to get the data for specific
    (hydrological) years only

    Returns:
    --------
    (years, tempformelt, prcpsol)::
        - years: array of shape (ny,)
        - tempformelt:  array of shape (len(heights), ny)
        - prcpsol:  array of shape (len(heights), ny)
    """

    flowlines = []
    if div_id is None:
        raise ValueError('Must specify div_id')

    if div_id == 0:
        for i in gdir.divide_ids:
             flowlines.extend(gdir.read_pickle('inversion_flowlines', div_id=i))
    else:
        flowlines = gdir.read_pickle('inversion_flowlines', div_id=div_id)

    heights = np.array([])
    widths = np.array([])
    for fl in flowlines:
        heights = np.append(heights, fl.surface_h)
        widths = np.append(widths, fl.widths)

    years, temp, prcp = mb_yearly_climate_on_height(gdir, heights,
                                                    year_range=year_range,
                                                    flatten=False)

    temp = np.average(temp, axis=0, weights=widths)
    prcp = np.average(prcp, axis=0, weights=widths)

    return years, temp, prcp


@entity_task(log, writes=['mu_candidates'])
@divide_task(log, add_0=True)
def mu_candidates(gdir, div_id=None):
    """Computes the mu candidates.

    For each 31 year-period centered on the year of interest, mu is is the
    temperature sensitivity necessary for the glacier with its current shape
    to be in equilibrium with its climate.

    For glaciers with MB data only!

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])

    years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir,
                                                           div_id=div_id)

    # Compute mu for each 31-yr climatological period
    ny = len(years)
    mu_yr_clim = np.zeros(ny) * np.NaN
    for i, y in enumerate(years):
        # Ignore begin and end
        if ((i-mu_hp) < 0) or ((i+mu_hp) >= ny):
            continue
        t_avg = np.mean(temp_yr[i-mu_hp:i+mu_hp+1])
        if t_avg > 1e-3 :
            mu_yr_clim[i] = np.mean(prcp_yr[i-mu_hp:i+mu_hp+1]) / t_avg

    # Check mu's
    if np.sum(np.isfinite(mu_yr_clim)) < (len(years) / 2.):
        raise RuntimeError('{}: has no normal climate'.format(gdir.rgi_id))

    # Write
    df = pd.Series(data=mu_yr_clim, index=years)
    gdir.write_pickle(df, 'mu_candidates', div_id=div_id)


def t_star_from_refmb(gdir, mbdf):
    """Computes the t* for the glacier, given a series of MB measurements.

    Could be multiprocessed but its probably not necessary.

    Parameters
    ----------
    gdirs: the list of oggm.GlacierDirectory objects where to write the data.
    mbdf: a pd.Series containing the observed MB data indexed by year

    Returns:
    --------
    (mu_star, bias, rmsd): lists of mu*, their associated bias and rmsd
    """

    # Only divide 0, we believe the original RGI entities to be the ref...
    years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir, div_id=0)

    # which years to look at
    selind = np.searchsorted(years, mbdf.index)
    temp_yr = np.mean(temp_yr[selind])
    prcp_yr = np.mean(prcp_yr[selind])

    # Average oberved mass-balance
    ref_mb = np.mean(mbdf)

    # Average mass-balance per mu
    mu_yr_clim = gdir.read_pickle('mu_candidates', div_id=0)
    mb_per_mu = prcp_yr - mu_yr_clim * temp_yr

    # Diff to reference
    diff = mb_per_mu - ref_mb
    diff = diff.dropna()

    # Solution to detect sign changes
    # http://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-
    # for-elements-in-a-numpy-array
    asign = np.sign(diff)
    sz = asign == 0
    while sz.any():
        asign[sz] = np.roll(asign, 1)[sz]
        sz = asign == 0
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    # If sign changes save them
    # TODO: ideas to do this better: apply a smooth (take noise into account)
    # TODO: ideas to do this better: take longer stable periods into account
    # TODO: these stuffs could be proven better (or not) with cross-val
    pchange = np.where(signchange == 1)[0]
    years = diff.index
    diff = diff.values
    if len(pchange) > 0:
        t_stars = []
        bias = []
        for p in pchange:
            # Check the side with the smallest bias
            if np.abs(diff[p-1]) < np.abs(diff[p]):
                t_stars.append(years[p-1])
                bias.append(diff[p-1])
            else:
                t_stars.append(years[p])
                bias.append(diff[p])
    else:
        amin = np.argmin(np.abs(diff))
        t_stars = [years[amin]]
        bias = [diff[amin]]

    return t_stars, bias


@entity_task(log, writes=['local_mustar', 'inversion_flowlines'])
def local_mustar_apparent_mb(gdir, tstar=None, bias=None):
    """Compute local mustar and apparent mb from tstar.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    tstar: int
        the year where the glacier should be equilibrium
    bias: int
        the associated reference bias
    """

    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr = [tstar-mu_hp, tstar+mu_hp]
    # Ok. Looping over divides
    for div_id in [0] + list(gdir.divide_ids):
        log.info('%s: local mu*, divide %d', gdir.rgi_id, div_id)

        # Get the corresponding mu
        years, temp_yr, prcp_yr = mb_yearly_climate_on_glacier(gdir,
                                                               div_id=div_id,
                                                               year_range=yr)
        assert len(years) == (2 * mu_hp + 1)
        mustar = np.mean(prcp_yr) / np.mean(temp_yr)
        if not np.isfinite(mustar):
            raise RuntimeError('{} has an infinite mu'.format(gdir.rgi_id))

        # Scalars in a small dataframe for later
        df = pd.DataFrame()
        df['rgi_id'] = [gdir.rgi_id]
        df['t_star'] = [tstar]
        df['mu_star'] = [mustar]
        df['bias'] = [bias]
        df.to_csv(gdir.get_filepath('local_mustar', div_id=div_id),
                  index=False)

        # For each flowline compute the apparent MB
        # For div 0 it is kind of artificial but this is for validation
        fls = []
        if div_id == 0:
            for i in gdir.divide_ids:
                 fls.extend(gdir.read_pickle('inversion_flowlines', div_id=i))
        else:
            fls = gdir.read_pickle('inversion_flowlines', div_id=div_id)

        # Reset flux
        for fl in fls:
            fl.flux = np.zeros(len(fl.surface_h))

        # Flowlines in order to be sure
        for fl in fls:
            y, t, p = mb_yearly_climate_on_height(gdir, fl.surface_h,
                                                  year_range=yr,
                                                  flatten=False)
            fl.set_apparent_mb(np.mean(p, axis=1) - mustar*np.mean(t, axis=1))

        # Overwrite
        if div_id >= 1:
            assert np.allclose(fls[-1].flux[-1], 0., atol=0.01)
        gdir.write_pickle(fls, 'inversion_flowlines', div_id=div_id)


def compute_ref_t_stars(gdirs):
    """ Detects the best t* for the reference glaciers.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
       for glaciers with MB data only!
    """

    log.info('Compute the reference t* and mu*')

    # Loop
    mbdatadir = os.path.join(os.path.dirname(cfg.PATHS['wgms_rgi_links']),
                             'WGMS')
    only_one = []  # start to store the glaciers with just one t*
    per_glacier = dict()
    for gdir in gdirs:
        reff = os.path.join(mbdatadir, 'mbdata_' + gdir.rgi_id + '.csv')
        mbdf = pd.read_csv(reff).set_index('YEAR')
        t_star, res_bias = t_star_from_refmb(gdir, mbdf['ANNUAL_BALANCE'])
        if len(t_star) == 1:
            only_one.append(gdir.rgi_id)
        per_glacier[gdir.rgi_id] = (gdir, t_star, res_bias)

    if len(only_one) == 0:
        # TODO: hardcoded shit here
        only_one.append('RGI40-11.00887')
        gdir, t_star, res_bias = per_glacier['RGI40-11.00887']
        per_glacier['RGI40-11.00887'] = (gdir, [t_star[-1]], [res_bias[-1]])
        # raise RuntimeError('Didnt expect to be here.')

    # Ok. now loop over the glaciers until all have a unique t*
    while True:
        ids_left = [id for id in per_glacier.keys() if id not in only_one]
        if len(ids_left) == 0:
            break

        # Compute the summed distance to all glaciers with one t*
        distances = []
        for id in ids_left:
            gdir, t_star, res_bias = per_glacier[id]
            lon, lat = gdir.cenlon, gdir.cenlat
            ldis = 0.
            for id_o in only_one:
                ogdir, _, _ = per_glacier[id_o]
                ldis += utils.haversine(lon, lat, ogdir.cenlon, ogdir.cenlat)
            distances.append(ldis)

        # Take the shortest and choose the best t*
        gdir, t_star, res_bias = per_glacier[ids_left[np.argmin(distances)]]
        distances = []
        for tt in t_star:
            ldis = 0.
            for id_o in only_one:
                _, ot_star, _ = per_glacier[id_o]
                ldis += np.abs(tt - ot_star)
            distances.append(ldis)
        amin = np.argmin(distances)
        per_glacier[gdir.rgi_id] = (gdir, [t_star[amin]], [res_bias[amin]])
        only_one.append(gdir.rgi_id)

    # Write out the data
    rgis_ids = []
    t_stars = []
    biases = []
    lons = []
    lats = []
    for id, (gdir, t_star, res_bias) in per_glacier.items():
        rgis_ids.append(id)
        t_stars.append(t_star[0])
        biases.append(res_bias[0])
        lats.append(gdir.cenlat)
        lons.append(gdir.cenlon)
    df = pd.DataFrame(index=rgis_ids)
    df['tstar'] = t_stars
    df['bias'] = biases
    df['lon'] = lons
    df['lat'] = lats
    file = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
    df.sort_index().to_csv(file)


def distribute_t_stars(gdirs):
    """After the computation of the reference tstars, apply
    the interpolation to each individual glacier.

    Parameters
    ----------
    gdirs: list of oggm.GlacierDirectory objects
    """

    log.info('Distribute t* and mu*')

    ref_df = pd.read_csv(os.path.join(cfg.PATHS['working_dir'],
                                      'ref_tstars.csv'))

    for gdir in gdirs:

        # Compute the distance to each glacier
        distances = utils.haversine(gdir.cenlon, gdir.cenlat,
                                    ref_df.lon, ref_df.lat)

        # Take the 10 closests
        aso = np.argsort(distances)[0:9]
        amin = ref_df.iloc[aso]
        distances = distances[aso]**2

        # If really close no need to divide, else weighted average
        if distances.iloc[0] <= 0.1:
            tstar = amin.tstar.iloc[0]
            bias = amin.bias.iloc[0]
        else:
            tstar = int(np.average(amin.tstar, weights=1./distances))
            bias = np.average(amin.bias, weights=1./distances)

        # Go
        local_mustar_apparent_mb(gdir, tstar=tstar, bias=bias)