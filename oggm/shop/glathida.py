import logging
import os

# External libs
import pandas as pd
import numpy as np
import xarray as xr

# Optional libs
try:
    import salem
except ImportError:
    pass

# Locals
from oggm import utils, cfg

# Module logger
log = logging.getLogger(__name__)

GTD_BASE_URL = ('https://cluster.klima.uni-bremen.de/~oggm/glathida/glathida-main/'
                'data/glathida_2023-11-16_rgi_{}_per_id.h5')


@utils.entity_task(log, writes=['glathida_data'])
def glathida_to_gdir(gdir):
    """Add GlaThiDa data to this glacier directory.

    The data is from the GlaThiDa main repository as of 2023-11-16. It's the raw data,
    i.e. no quality control was applied. It may contain zeros or unrealisic values.
    It's the users task to make sure the data suits their use case.

    The dataframe contains a few additional columns:
    - x_proj, y_proj: the point coordinates in the local map projection
    - i_grid, j_grid: the point coordinates in the local *nearest* grid point coordinates
                      (e.g. to access the data from a grid)
    - ij_grid: a unique identifier per grid point. Allows to group the data by gridpoint
               (see tutorials)

    https://gitlab.com/wgms/glathida

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    Returns
    -------
    the dataframe, or None if there is no data for this glacier.
    """

    rgi_version = gdir.rgi_version
    if rgi_version == '60':
        rgi_version = '62'

    base_url = GTD_BASE_URL.format(rgi_version)
    gtd_file = utils.file_downloader(base_url)

    try:
        df = pd.read_hdf(gtd_file, key=gdir.rgi_id)
    except KeyError:
        log.debug(f'({gdir.rgi_id}): no GlaThiDa data for this glacier')
        return None

    # OK - transform for later
    xx, yy = salem.transform_proj(salem.wgs84, gdir.grid.proj, df['longitude'], df['latitude'])
    df['x_proj'] = xx
    df['y_proj'] = yy

    log.debug(f'({gdir.rgi_id}): {len(df)} GlaThiDa points for this glacier')

    ii, jj = gdir.grid.transform(df['longitude'], df['latitude'], crs=salem.wgs84, nearest=True)
    df['i_grid'] = ii
    df['j_grid'] = jj

    # We trick by creating an index of similar i's and j's
    df['ij_grid'] = ['{:04d}_{:04d}'.format(i, j) for i, j in zip(ii, jj)]

    df.to_csv(gdir.get_filepath('glathida_data'), index=None)
    return df


@utils.entity_task(log)
def glathida_statistics(gdir):
    """Gather statistics about the Glathida data interpolated to this glacier.
    """

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['n_points'] = 0
    d['n_valid_thick_points'] = 0
    d['n_valid_elev_points'] = 0
    d['n_valid_gridded_points'] = 0
    d['avg_thick'] = np.nan
    d['max_thick'] = np.nan
    d['date_mode'] = None
    d['date_min'] = None
    d['date_max'] = None

    try:
        dfo = pd.read_csv(gdir.get_filepath('glathida_data'))
        d['n_points'] = len(dfo)
        df = dfo.loc[dfo.thickness > 0]
        dfg = df.groupby(by='ij_grid')['thickness'].mean()
        d['n_valid_thick_points'] = len(df)
        d['n_valid_elev_points'] = (dfo.elevation > 0).sum()
        d['n_valid_gridded_points'] = len(dfg)
        d['avg_thick'] = df.thickness.mean()
        d['max_thick'] = df.thickness.max()
        d['date_mode'] = df.date.mode().iloc[0]
        d['date_min'] = df.date.min()
        d['date_max'] = df.date.max()
    except:
        pass

    return d


@utils.global_task(log)
def compile_glathida_statistics(gdirs, filesuffix='', path=True):
    """Gather as much statistics as possible about a list of glaciers.

    It can be used to do result diagnostics and other stuffs.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(glathida_statistics, gdirs)
    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('glathida_statistics' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out


@utils.entity_task(log)
def glathida_on_grid(gdir):
    """Gather a dataframe of predictors for this glacier.
    """

    # Read the data
    try:
        df = pd.read_csv(gdir.get_filepath('glathida_data'))
    except FileNotFoundError:
        return None

    # Agregate on grid
    drop_vars = ['date', 'elevation_date', 'flag', 'rgi_id', 'survey_id', 'elevation']
    df_agg = df.drop(drop_vars, axis=1).groupby('ij_grid').mean()

    # Conversion does not preserve ints
    df_agg['i_grid'] = df_agg['i_grid'].astype(int)
    df_agg['j_grid'] = df_agg['j_grid'].astype(int)

    # Some other variables do not make much sense as mean
    def mode(x):
        return x.mode().iloc[0]

    df_agg['survey_id'] = df[['ij_grid', 'survey_id']].groupby('ij_grid').agg(mode)['survey_id']
    df_agg['date'] = df[['ij_grid', 'date']].groupby('ij_grid').agg(mode)['date']
    df_agg['rgi_id'] = df['rgi_id'].iloc[0]

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            ds = ds.load()
    except:
        return None

    # Select variables
    vns = ['topo',
           'topo_smoothed',
           'slope',
           'slope_factor',
           'aspect',
           'dis_from_border',
           'catchment_area',
           'lin_mb_above_z',
           'oggm_mb_above_z',
           'consensus_ice_thickness',
           'millan_ice_thickness',
           'bedmachine_ice_thickness',
           'millan_v',
           'itslive_v',
           'hugonnet_dhdt',
           ]
    for vn in vns:
        try:
            df_agg[vn] = ds[vn].isel(x=('z', df_agg['i_grid']), y=('z', df_agg['j_grid']))
        except KeyError:
            pass

    return df_agg.reset_index()


@utils.global_task(log)
def compile_glathida_on_grid(gdirs, filesuffix='', path=True,
                             climate_statistics_file='',
                             glacier_statistics_file='',
                             ):
    """Gather as many predictors as possible for the big ML table.

    The assumption is that the tasks:
    - tasks.gridded_attributes,
    - tasks.gridded_mb_attributes,
    have been run prior to this.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(glathida_on_grid, gdirs)
    out_df = [df for df in out_df if df is not None]
    out = pd.concat(out_df).reset_index()
    out = out.drop(['index'], axis=1)

    if glacier_statistics_file:
        vns = {'cenlon': 'glacier_cenlon',
               'cenlat': 'glacier_cenlat',
               'rgi_area_km2': 'glacier_area_km2',
               'rgi_year': 'glacier_outline_year',
               'is_tidewater': 'glacier_is_tidewater',
               'dem_med_elev': 'glacier_median_elev',
               'dem_min_elev': 'glacier_min_elev',
               'dem_max_elev': 'glacier_max_elev',
               'main_flowline_length': 'glacier_length',
               'inv_volume_km3': 'glacier_oggm_volume',
               'reference_mb': 'glacier_reference_mb',
               }
        dfc = pd.read_csv(glacier_statistics_file, index_col=0)

        # Select only needed columns
        dfc_sel = dfc[list(vns.keys())]

        # Merge: left join keeps all rows from `out`
        out = out.merge(dfc_sel, left_on='rgi_id', right_index=True, how='left')

        # Rename columns to final output names
        out = out.rename(columns=vns)

    if climate_statistics_file:
        vns = {'1980-2010_aar': 'glacier_aar',
               '1980-2010_avg_temp_mean_elev': 'glacier_avg_temp',
               '1980-2010_avg_prcpsol_mean_elev': 'glacier_avg_prcpsol',
               '1980-2010_avg_prcp': 'glacier_avg_prcp',
               '1980-2010_avg_tempmelt_mean_elev': 'glacier_avg_meltingtemp',
               }
        dfc = pd.read_csv(climate_statistics_file, index_col=0)

        # Select only needed columns
        dfc_sel = dfc[list(vns.keys())]

        # Merge: left join keeps all rows from `out`
        out = out.merge(dfc_sel, left_on='rgi_id', right_index=True, how='left')

        # Rename columns to final output names
        out = out.rename(columns=vns)

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('glathida_on_grid' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out
