"""Command line arguments to the oggm_prepro command

Type `$ oggm_prepro -h` for help

"""

# External modules
import os
import sys
import shutil
import argparse
import time
import logging
import json
import pandas as pd
import numpy as np
import geopandas as gpd

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow, tasks, GlacierDirectory
from oggm.core import gis
from oggm.exceptions import InvalidParamsError, InvalidDEMError

# Module logger
from oggm.utils import get_prepro_base_url, file_downloader

log = logging.getLogger(__name__)


@utils.entity_task(log)
def _rename_dem_folder(gdir, source=''):
    """Put the DEM files in a subfolder of the gdir.

    Parameters
    ----------
    gdir : GlacierDirectory
    source : str
        the DEM source
    """

    # open tif-file to check if it's worth it
    dem_f = gdir.get_filepath('dem')
    try:
        dem = gis.read_geotiff_dem(gdir)
    except IOError:
        # Error reading file, no problem - still, delete the file if needed
        if os.path.exists(dem_f):
            os.remove(dem_f)
        gdir.log('{},DEM SOURCE,{}'.format(gdir.rgi_id, source),
                 err=InvalidDEMError('File does not exist'))
        return

    # Check the DEM
    isfinite = np.isfinite(dem)
    if np.all(~isfinite) or (np.min(dem) == np.max(dem)):
        # Remove the file and return
        if os.path.exists(dem_f):
            os.remove(dem_f)
        gdir.log('{},DEM SOURCE,{}'.format(gdir.rgi_id, source),
                 err=InvalidDEMError('DEM does not contain more than one '
                                     'valid values.'))
        return

    # Create a source dir and move the files
    out = os.path.join(gdir.dir, source)
    utils.mkdir(out)
    for fname in ['dem', 'dem_source']:
        f = gdir.get_filepath(fname)
        os.rename(f, os.path.join(out, os.path.basename(f)))

    # log SUCCESS for this DEM source
    gdir.log('{},DEM SOURCE,{}'.format(gdir.rgi_id, source))


def run_prepro_levels(rgi_version=None, rgi_reg=None, border=None,
                      output_folder='', working_dir='', dem_source='',
                      is_test=False, test_ids=None, demo=False, test_rgidf=None,
                      test_intersects_file=None, test_topofile=None,
                      disable_mp=False, params_file=None,
                      elev_bands=False, centerlines=False,
                      override_params=None,
                      mb_calibration_strategy='informed_threestep',
                      select_source_from_dir=None, keep_dem_folders=False,
                      add_consensus_thickness=False, add_itslive_velocity=False,
                      add_millan_thickness=False, add_millan_velocity=False,
                      add_hugonnet_dhdt=False, add_glathida=False,
                      start_level=None, start_base_url=None, max_level=5,
                      logging_level='WORKFLOW',
                      dynamic_spinup=False, err_dmdtda_scaling_factor=0.2,
                      dynamic_spinup_start_year=1979,
                      continue_on_error=True, store_fl_diagnostics=False):
    """Generate the preprocessed OGGM glacier directories for this OGGM version

    Parameters
    ----------
    rgi_version : str
        the RGI version to use (defaults to cfg.PARAMS)
    rgi_reg : str
        the RGI region to process
    border : int
        the number of pixels at the maps border
    output_folder : str
        path to the output folder (where to put the preprocessed tar files)
    dem_source : str
        which DEM source to use: default, SOURCE_NAME, STANDARD or ALL
        "standard" is COPDEM + NASADEM
    working_dir : str
        path to the OGGM working directory
    params_file : str
        path to the OGGM parameter file (to override defaults)
    is_test : bool
        to test on a couple of glaciers only!
    test_ids : list
        if is_test: list of ids to process
    demo : bool
        to run the prepro for the list of demo glaciers
    test_rgidf : shapefile
        for testing purposes only
    test_intersects_file : shapefile
        for testing purposes only
    test_topofile : str
        for testing purposes only
    test_crudir : str
        for testing purposes only
    disable_mp : bool
        disable multiprocessing
    elev_bands : bool
        compute all flowlines based on the Huss & Farinotti 2012 method.
    centerlines : bool
        compute all flowlines based on the OGGM centerline(s) method.
    mb_calibration_strategy : str
        how to calibrate the massbalance. Currently one of:
        - 'informed_threestep' (default)
        - 'melt_temp'
        - 'temp_melt'
    select_source_from_dir : str
        if starting from a level 1 "ALL" or "STANDARD" DEM sources directory,
        select the chosen DEM source here. If you set it to "BY_RES" here,
        COPDEM will be used and its resolution chosen based on the gdir's
        map resolution (COPDEM30 for dx < 60 m, COPDEM90 elsewhere).
    keep_dem_folders : bool
        if `select_source_from_dir` is used, wether to keep the original
        DEM folders in or not.
    add_consensus_thickness : bool
        adds (reprojects) the consensus estimates thickness to the glacier
        directories. With elev_bands=True, the data will also be binned.
    add_itslive_velocity : bool
        adds (reprojects) the ITS_LIVE velocity to the glacier
        directories. With elev_bands=True, the data will also be binned.
    add_millan_thickness : bool
        adds (reprojects) the millan thickness to the glacier
        directories. With elev_bands=True, the data will also be binned.
    add_millan_velocity : bool
        adds (reprojects) the millan velocity to the glacier
        directories. With elev_bands=True, the data will also be binned.
    add_hugonnet_dhdt : bool
        adds (reprojects) the hugonnet dhdt maps to the glacier
        directories. With elev_bands=True, the data will also be binned.
    add_glathida : bool
        adds (reprojects) the glathida thickness data to the glacier
        directories. Data points are stored as csv files.
    start_level : int
        the pre-processed level to start from (default is to start from
        scratch). If set, you'll need to indicate start_base_url as well.
    start_base_url : str
        the pre-processed base-url to fetch the data from.
    max_level : int
        the maximum pre-processing level before stopping
    logging_level : str
        the logging level to use (DEBUG, INFO, WARNING, WORKFLOW)
    override_params : dict
        a dict of parameters to override.
    dynamic_spinup : str
        include a dynamic spinup matching 'area/dmdtda' OR 'volume/dmdtda' at
        the RGI-date
    err_dmdtda_scaling_factor : float
        scaling factor to reduce individual geodetic mass balance uncertainty
    dynamic_spinup_start_year : int
        if dynamic_spinup is set, define the starting year for the simulation.
        The default is 1979, unless the climate data starts later.
    continue_on_error : bool
        if True the workflow continues if a task raises an error. For operational
        runs it should be set to True (the default).
    store_fl_diagnostics : bool
        if True, also compute and store flowline diagnostics during preprocessing.
        This can increase data usage quite a bit.
    """

    # Input check
    if max_level not in [1, 2, 3, 4, 5]:
        raise InvalidParamsError('max_level should be one of [1, 2, 3, 4, 5]')

    if start_level is not None:
        if start_level not in [0, 1, 2, 3, 4]:
            raise InvalidParamsError('start_level should be one of [0, 1, 2, 3, 4]')
        if start_level > 0 and start_base_url is None:
            raise InvalidParamsError('With start_level, please also indicate '
                                     'start_base_url')
    else:
        start_level = 0

    if dynamic_spinup:
        if dynamic_spinup not in ['area/dmdtda', 'volume/dmdtda']:
            raise InvalidParamsError(f"Dynamic spinup option '{dynamic_spinup}' "
                                     "not supported")

    # Time
    start = time.time()

    def _time_log():
        # Log util
        m, s = divmod(time.time() - start, 60)
        h, m = divmod(m, 60)
        log.workflow('OGGM prepro_levels is done! Time needed: '
                     '{:02d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

    # Local paths
    if override_params is None:
        override_params = {}

    # Use multiprocessing?
    override_params['use_multiprocessing'] = not disable_mp

    # How many grid points around the glacier?
    # Make it large if you expect your glaciers to grow large
    override_params['border'] = border

    # Some arbitrary heuristics on the length of tidewater extension
    extension = int(utils.clip_min(border / 2, 30))
    override_params['calving_line_extension'] = extension

    # Set to True for operational runs
    override_params['continue_on_error'] = continue_on_error

    # Do not use bias file if user wants melt_temp only
    if mb_calibration_strategy in ['melt_temp', 'temp_melt']:
        override_params['use_temp_bias_from_file'] = False

    # For centerlines we have to change the default evolution model and bed
    if centerlines:
        override_params['downstream_line_shape'] = 'parabola'
        override_params['evolution_model'] = 'FluxBased'

    # Other things that make sense
    override_params['store_model_geometry'] = True
    override_params['store_fl_diagnostics'] = store_fl_diagnostics

    utils.mkdir(working_dir)
    override_params['working_dir'] = working_dir

    # Initialize OGGM and set up the run parameters
    cfg.initialize(file=params_file, params=override_params,
                   logging_level=logging_level)

    # Prepare the download of climate file to be shared across processes
    # TODO

    # Log the parameters
    msg = '# OGGM Run parameters:'
    for k, v in cfg.PARAMS.items():
        if type(v) in [pd.DataFrame, dict]:
            continue
        msg += '\n    {}: {}'.format(k, v)
    log.workflow(msg)

    if rgi_version is None:
        rgi_version = cfg.PARAMS['rgi_version']
    output_base_dir = os.path.join(output_folder,
                                   'RGI{}'.format(rgi_version),
                                   'b_{:03d}'.format(border))

    # Add a package version file
    utils.mkdir(output_base_dir)
    opath = os.path.join(output_base_dir, 'package_versions.txt')
    with open(opath, 'w') as vfile:
        vfile.write(utils.show_versions(logger=log))

    if demo:
        rgidf = utils.get_rgi_glacier_entities(cfg.DATA['demo_glaciers'].index)
    elif test_rgidf is None:

        # Get the RGI file
        rgidf = gpd.read_file(utils.get_rgi_region_file(rgi_reg,
                                                        version=rgi_version))
        # We use intersects
        if rgi_version != '70C':
            rgif = utils.get_rgi_intersects_region_file(rgi_reg,
                                                        version=rgi_version)
            cfg.set_intersects_db(rgif)

        if rgi_version == '62':
            # Some RGI input quality checks - this is based on visual checks
            # of large glaciers in the RGI
            ids_to_ice_cap = [
                'RGI60-05.10315',  # huge Greenland ice cap
                'RGI60-03.01466',  # strange thing next to Devon
                'RGI60-09.00918',  # Academy of sciences Ice cap
                'RGI60-09.00969',
                'RGI60-09.00958',
                'RGI60-09.00957',
            ]
            rgidf.loc[rgidf.RGIId.isin(ids_to_ice_cap), 'Form'] = '1'

            # In AA almost all large ice bodies are actually ice caps
            if rgi_reg == '19':
                rgidf.loc[rgidf.Area > 100, 'Form'] = '1'

            # For greenland we omit connectivity level 2
            if rgi_reg == '05':
                rgidf = rgidf.loc[rgidf['Connect'] != 2]
    else:
        rgidf = test_rgidf
        cfg.set_intersects_db(test_intersects_file)

    if is_test:
        if test_ids is not None:
            try:
                rgidf = rgidf.loc[rgidf.RGIId.isin(test_ids)]
            except AttributeError:
                # RGI7
                rgidf = rgidf.loc[rgidf.rgi_id.isin(test_ids)]
        else:
            rgidf = rgidf.sample(4)

    log.workflow('Starting prepro run for RGI reg: {} '
                 'and border: {}'.format(rgi_reg, border))
    log.workflow('Number of glaciers: {}'.format(len(rgidf)))

    # L0 - go
    if start_level == 0:
        gdirs = workflow.init_glacier_directories(rgidf, reset=True, force=True)

        # Glacier stats
        sum_dir = os.path.join(output_base_dir, 'L0', 'summary')
        utils.mkdir(sum_dir)
        opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
        utils.compile_glacier_statistics(gdirs, path=opath)

        # L0 OK - compress all in output directory
        log.workflow('L0 done. Writing to tar...')
        level_base_dir = os.path.join(output_base_dir, 'L0')
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                     base_dir=level_base_dir)
        utils.base_dir_to_tar(level_base_dir)
        if max_level == 0:
            _time_log()
            return
    else:
        gdirs = workflow.init_glacier_directories(rgidf, reset=True, force=True,
                                                  from_prepro_level=start_level,
                                                  prepro_border=border,
                                                  prepro_rgi_version=rgi_version,
                                                  prepro_base_url=start_base_url
                                                  )

    # L1 - Add dem files
    if start_level == 0:
        if test_topofile:
            cfg.PATHS['dem_file'] = test_topofile

        # Which DEM source?
        if dem_source.upper() in ['ALL', 'STANDARD']:
            # This is the complex one, just do the job and leave

            if dem_source.upper() == 'ALL':
                sources = utils.DEM_SOURCES
            if dem_source.upper() == 'STANDARD':
                sources = ['COPDEM30', 'COPDEM90', 'NASADEM']

            log.workflow('Running prepro on several sources')
            for i, s in enumerate(sources):
                rs = i == 0
                log.workflow('Running prepro on sources: {}'.format(s))
                gdirs = workflow.init_glacier_directories(rgidf, reset=rs,
                                                          force=rs)
                workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                             source=s)
                workflow.execute_entity_task(_rename_dem_folder, gdirs, source=s)

            # make a GeoTiff mask of the glacier, choose any source
            workflow.execute_entity_task(gis.rasterio_glacier_mask,
                                         gdirs, source='ALL')

            # Glacier stats
            sum_dir = os.path.join(output_base_dir, 'L1', 'summary')
            utils.mkdir(sum_dir)
            opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
            utils.compile_glacier_statistics(gdirs, path=opath)

            # L1 OK - compress all in output directory
            log.workflow('L1 done. Writing to tar...')
            level_base_dir = os.path.join(output_base_dir, 'L1')
            workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                         base_dir=level_base_dir)
            utils.base_dir_to_tar(level_base_dir)

            _time_log()
            return

        # Force a given source
        source = dem_source.upper() if dem_source else None

        # L1 - go
        workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                     source=source)

        # Glacier stats
        sum_dir = os.path.join(output_base_dir, 'L1', 'summary')
        utils.mkdir(sum_dir)
        opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
        utils.compile_glacier_statistics(gdirs, path=opath)

        # L1 OK - compress all in output directory
        log.workflow('L1 done. Writing to tar...')
        level_base_dir = os.path.join(output_base_dir, 'L1')
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                     base_dir=level_base_dir)
        utils.base_dir_to_tar(level_base_dir)
        if max_level == 1:
            _time_log()
            return

    # L2 - Tasks
    if start_level <= 1:
        # Check which glaciers will be processed as what
        if elev_bands:
            gdirs_band = gdirs
            gdirs_cent = []
        elif centerlines:
            gdirs_band = []
            gdirs_cent = gdirs
        else:
            raise InvalidParamsError('Need to specify if `elev_bands` or '
                                     '`centerlines` type.')

        log.workflow('Start flowline processing with: '
                     'N centerline type: {}, '
                     'N elev bands type: {}.'
                     ''.format(len(gdirs_cent), len(gdirs_band)))

        # If we are coming from a multi-dem setup, let's select it from there
        if select_source_from_dir is not None:
            from oggm.shop.rgitopo import select_dem_from_dir
            workflow.execute_entity_task(select_dem_from_dir, gdirs_band,
                                         dem_source=select_source_from_dir,
                                         keep_dem_folders=keep_dem_folders)
            workflow.execute_entity_task(select_dem_from_dir, gdirs_cent,
                                         dem_source=select_source_from_dir,
                                         keep_dem_folders=keep_dem_folders)

        # HH2015 method
        workflow.execute_entity_task(tasks.simple_glacier_masks, gdirs_band)

        # Centerlines OGGM
        workflow.execute_entity_task(tasks.glacier_masks, gdirs_cent)

        bin_variables = []
        if add_consensus_thickness:
            from oggm.shop.bedtopo import add_consensus_thickness
            workflow.execute_entity_task(add_consensus_thickness, gdirs)
            bin_variables.append('consensus_ice_thickness')
        if add_itslive_velocity:
            from oggm.shop.its_live import velocity_to_gdir
            workflow.execute_entity_task(velocity_to_gdir, gdirs)
            bin_variables.append('itslive_v')
        if add_millan_thickness:
            from oggm.shop.millan22 import thickness_to_gdir
            workflow.execute_entity_task(thickness_to_gdir, gdirs)
            bin_variables.append('millan_ice_thickness')
        if add_millan_velocity:
            from oggm.shop.millan22 import velocity_to_gdir
            workflow.execute_entity_task(velocity_to_gdir, gdirs)
            bin_variables.append('millan_v')
        if add_hugonnet_dhdt:
            from oggm.shop.hugonnet_maps import hugonnet_to_gdir
            workflow.execute_entity_task(hugonnet_to_gdir, gdirs)
            bin_variables.append('hugonnet_dhdt')
        if add_glathida:
            from oggm.shop.glathida import glathida_to_gdir
            workflow.execute_entity_task(glathida_to_gdir, gdirs)

        if bin_variables and gdirs_band:
            workflow.execute_entity_task(tasks.elevation_band_flowline,
                                         gdirs_band,
                                         bin_variables=bin_variables)
            workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline,
                                         gdirs_band,
                                         bin_variables=bin_variables)
        else:
            # HH2015 method without it
            task_list = [
                tasks.elevation_band_flowline,
                tasks.fixed_dx_elevation_band_flowline,
            ]
            for task in task_list:
                workflow.execute_entity_task(task, gdirs_band)

        # Centerlines OGGM
        task_list = [
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            workflow.execute_entity_task(task, gdirs_cent)

        # Same for all glaciers
        if border >= 20:
            task_list = [
                tasks.compute_downstream_line,
                tasks.compute_downstream_bedshape,
            ]
            for task in task_list:
                workflow.execute_entity_task(task, gdirs)
        else:
            log.workflow('L2: for map border values < 20, wont compute '
                         'downstream lines.')

        # Glacier stats
        sum_dir = os.path.join(output_base_dir, 'L2', 'summary')
        utils.mkdir(sum_dir)
        opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
        utils.compile_glacier_statistics(gdirs, path=opath)

        if add_itslive_velocity:
            from oggm.shop.its_live import compile_itslive_statistics
            opath = os.path.join(sum_dir, 'itslive_statistics_{}.csv'.format(rgi_reg))
            compile_itslive_statistics(gdirs, path=opath)
        if add_millan_thickness or add_millan_velocity:
            from oggm.shop.millan22 import compile_millan_statistics
            opath = os.path.join(sum_dir, 'millan_statistics_{}.csv'.format(rgi_reg))
            compile_millan_statistics(gdirs, path=opath)
        if add_consensus_thickness:
            from oggm.shop.bedtopo import compile_consensus_statistics
            opath = os.path.join(sum_dir, 'consensus_statistics_{}.csv'.format(rgi_reg))
            compile_consensus_statistics(gdirs, path=opath)
        if add_hugonnet_dhdt:
            from oggm.shop.hugonnet_maps import compile_hugonnet_statistics
            opath = os.path.join(sum_dir, 'hugonnet_statistics_{}.csv'.format(rgi_reg))
            compile_hugonnet_statistics(gdirs, path=opath)
        if add_glathida:
            from oggm.shop.glathida import compile_glathida_statistics
            opath = os.path.join(sum_dir, 'glathida_statistics_{}.csv'.format(rgi_reg))
            compile_glathida_statistics(gdirs, path=opath)

        # And for level 2: shapes
        if len(gdirs_cent) > 0:
            opath = os.path.join(sum_dir, f'centerlines_{rgi_reg}.shp')
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             path=opath)
            opath = os.path.join(sum_dir, f'centerlines_smoothed_{rgi_reg}.shp')
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             ensure_exterior_match=True,
                                             simplify_line_before=0.75,
                                             corner_cutting=3,
                                             path=opath)
            opath = os.path.join(sum_dir, f'flowlines_{rgi_reg}.shp')
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             flowlines_output=True,
                                             path=opath)
            opath = os.path.join(sum_dir, f'geom_widths_{rgi_reg}.shp')
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             geometrical_widths_output=True,
                                             path=opath)
            opath = os.path.join(sum_dir, f'widths_{rgi_reg}.shp')
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             corrected_widths_output=True,
                                             path=opath)

        # L2 OK - compress all in output directory
        log.workflow('L2 done. Writing to tar...')
        level_base_dir = os.path.join(output_base_dir, 'L2')
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                     base_dir=level_base_dir)
        utils.base_dir_to_tar(level_base_dir)
        if max_level == 2:
            _time_log()
            return

    # L3 - Tasks
    if start_level <= 2:
        sum_dir = os.path.join(output_base_dir, 'L3', 'summary')
        utils.mkdir(sum_dir)

        # Climate
        workflow.execute_entity_task(tasks.process_climate_data, gdirs)

        # Small optim to avoid concurrency
        utils.get_geodetic_mb_dataframe()
        utils.get_temp_bias_dataframe()
        if mb_calibration_strategy == 'informed_threestep':
            workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                         gdirs, informed_threestep=True)
        elif mb_calibration_strategy == 'melt_temp':
            workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                         gdirs,
                                         calibrate_param1='melt_f',
                                         calibrate_param2='temp_bias')
        elif mb_calibration_strategy == 'temp_melt':
            workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                         gdirs,
                                         calibrate_param1='temp_bias',
                                         calibrate_param2='melt_f')
        else:
            raise InvalidParamsError('mb_calibration_strategy not understood: '
                                     f'{mb_calibration_strategy}')

        workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs)

        # Inversion: we match the consensus
        filter = border >= 20
        workflow.calibrate_inversion_from_consensus(gdirs,
                                                    apply_fs_on_mismatch=True,
                                                    error_on_mismatch=False,
                                                    filter_inversion_output=filter)

        # We get ready for modelling
        if border >= 20:
            workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)
        else:
            log.workflow('L3: for map border values < 20, wont initialize glaciers '
                         'for the run.')
        # Glacier stats
        opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
        utils.compile_glacier_statistics(gdirs, path=opath)
        opath = os.path.join(sum_dir, 'climate_statistics_{}.csv'.format(rgi_reg))
        utils.compile_climate_statistics(gdirs, path=opath)
        opath = os.path.join(sum_dir, 'fixed_geometry_mass_balance_{}.csv'.format(rgi_reg))
        utils.compile_fixed_geometry_mass_balance(gdirs, path=opath)

        # L3 OK - compress all in output directory
        log.workflow('L3 done. Writing to tar...')
        level_base_dir = os.path.join(output_base_dir, 'L3')
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                     base_dir=level_base_dir)
        utils.base_dir_to_tar(level_base_dir)
        if max_level == 3:
            _time_log()
            return
        if border < 20:
            log.workflow('L3: for map border values < 20, wont compute L4 and L5.')
            _time_log()
            return

        # is needed to copy some files for L4 and L5
        sum_dir_L3 = sum_dir

    # L4 - Tasks (add historical runs (old default) and dynamic spinup runs)
    if start_level <= 3:
        sum_dir = os.path.join(output_base_dir, 'L4', 'summary')
        utils.mkdir(sum_dir)

        # Copy L3 files for consistency
        for bn in ['glacier_statistics', 'climate_statistics',
                   'fixed_geometry_mass_balance']:
            if start_level <= 2:
                ipath = os.path.join(sum_dir_L3, bn + '_{}.csv'.format(rgi_reg))
            else:
                ipath = file_downloader(os.path.join(
                    get_prepro_base_url(base_url=start_base_url,
                                        rgi_version=rgi_version, border=border,
                                        prepro_level=start_level), 'summary',
                    bn + '_{}.csv'.format(rgi_reg)))

            opath = os.path.join(sum_dir, bn + '_{}.csv'.format(rgi_reg))
            shutil.copyfile(ipath, opath)

        # Get end date. The first gdir might have blown up, try some others
        i = 0
        while True:
            if i >= len(gdirs):
                raise RuntimeError('Found no valid glaciers!')
            try:
                y0 = gdirs[i].get_climate_info()['baseline_yr_0']
                # One adds 1 because the run ends at the end of the year
                ye = gdirs[i].get_climate_info()['baseline_yr_1'] + 1
                break
            except BaseException:
                i += 1

        # conduct historical run before dynamic melt_f calibration
        # (for comparison to old default behavior)
        workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
                                     min_ys=y0, ye=ye,
                                     output_filesuffix='_historical')
        # Now compile the output
        opath = os.path.join(sum_dir, f'historical_run_output_{rgi_reg}.nc')
        utils.compile_run_output(gdirs, path=opath, input_filesuffix='_historical')

        # conduct dynamic spinup if wanted
        if dynamic_spinup:
            if y0 > dynamic_spinup_start_year:
                dynamic_spinup_start_year = y0

            minimise_for = dynamic_spinup.split('/')[0]

            melt_f_max = cfg.PARAMS['melt_f_max']
            workflow.execute_entity_task(
                tasks.run_dynamic_melt_f_calibration, gdirs,
                err_dmdtda_scaling_factor=err_dmdtda_scaling_factor,
                ys=dynamic_spinup_start_year, ye=ye,
                melt_f_max=melt_f_max,
                kwargs_run_function={'minimise_for': minimise_for},
                ignore_errors=True,
                kwargs_fallback_function={'minimise_for': minimise_for},
                output_filesuffix='_spinup_historical',)
            # Now compile the output
            opath = os.path.join(sum_dir, f'spinup_historical_run_output_{rgi_reg}.nc')
            utils.compile_run_output(gdirs, path=opath,
                                     input_filesuffix='_spinup_historical')

        # Glacier statistics we recompute here for error analysis
        opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
        utils.compile_glacier_statistics(gdirs, path=opath)

        # Add the extended files
        pf = os.path.join(sum_dir, 'historical_run_output_{}.nc'.format(rgi_reg))
        # We have copied the files above
        mf = os.path.join(sum_dir, 'fixed_geometry_mass_balance_{}.csv'.format(rgi_reg))
        sf = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
        opath = os.path.join(sum_dir, 'historical_run_output_extended_{}.nc'.format(rgi_reg))
        utils.extend_past_climate_run(past_run_file=pf,
                                      fixed_geometry_mb_file=mf,
                                      glacier_statistics_file=sf,
                                      path=opath)

        # L4 OK - compress all in output directory
        log.workflow('L4 done. Writing to tar...')
        level_base_dir = os.path.join(output_base_dir, 'L4')
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                     base_dir=level_base_dir)
        utils.base_dir_to_tar(level_base_dir)

        sum_dir_L4 = sum_dir

        if max_level == 4:
            _time_log()
            return

    # L5 - No tasks: make the dirs small
    sum_dir = os.path.join(output_base_dir, 'L5', 'summary')
    utils.mkdir(sum_dir)

    # Copy L4 files for consistency
    files_to_copy = ['glacier_statistics', 'climate_statistics',
                     'fixed_geometry_mass_balance', 'historical_run_output',
                     'historical_run_output_extended']
    files_suffixes = ['csv', 'csv', 'csv', 'nc', 'nc']
    if dynamic_spinup:
        files_to_copy.append('spinup_historical_run_output')
        files_suffixes.append('nc')
    for bn, suffix in zip(files_to_copy, files_suffixes):
        if start_level <= 3:
            ipath = os.path.join(sum_dir_L4, bn + f'_{rgi_reg}.{suffix}')
        else:
            ipath = file_downloader(os.path.join(
                get_prepro_base_url(base_url=start_base_url,
                                    rgi_version=rgi_version, border=border,
                                    prepro_level=start_level), 'summary',
                bn + f'_{rgi_reg}.{suffix}'))
        opath = os.path.join(sum_dir, bn + f'_{rgi_reg}.{suffix}')
        shutil.copyfile(ipath, opath)

    # Copy mini data to new dir
    mini_base_dir = os.path.join(working_dir, 'mini_perglacier',
                                 'RGI{}'.format(rgi_version),
                                 'b_{:03d}'.format(border))
    mini_gdirs = workflow.execute_entity_task(tasks.copy_to_basedir, gdirs,
                                              base_dir=mini_base_dir,
                                              setup='run/spinup')

    # L5 OK - compress all in output directory
    log.workflow('L5 done. Writing to tar...')
    level_base_dir = os.path.join(output_base_dir, 'L5')
    workflow.execute_entity_task(utils.gdir_to_tar, mini_gdirs, delete=False,
                                 base_dir=level_base_dir)
    utils.base_dir_to_tar(level_base_dir)

    _time_log()


def parse_args(args):
    """Check input arguments and env variables"""

    # CLI args
    description = ('Generate the preprocessed OGGM glacier directories for '
                   'this OGGM version.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--map-border', type=int,
                        help='the size of the map border. Is required if '
                             '$OGGM_MAP_BORDER is not set.')
    parser.add_argument('--rgi-reg', type=str,
                        help='the rgi region to process. Is required if '
                             '$OGGM_RGI_REG is not set.')
    parser.add_argument('--rgi-version', type=str,
                        help='the RGI version to use. Defaults to the OGGM '
                             'default.')
    parser.add_argument('--start-level', type=int, default=0,
                        help='the pre-processed level to start from (default '
                             'is to start from 0). If set, you will need to '
                             'indicate --start-base-url as well.')
    parser.add_argument('--start-base-url', type=str,
                        help='the pre-processed base-url to fetch the data '
                             'from when starting from level > 0.')
    parser.add_argument('--max-level', type=int, default=5,
                        help='the maximum level you want to run the '
                             'pre-processing for (1, 2, 3, 4 or 5).')
    parser.add_argument('--working-dir', type=str,
                        help='path to the directory where to write the '
                             'output. Defaults to current directory or '
                             '$OGGM_WORKDIR.')
    parser.add_argument('--params-file', type=str,
                        help='path to the OGGM parameter file to use in place '
                             'of the default one.')
    parser.add_argument('--output', type=str,
                        help='path to the directory where to write the '
                             'output. Defaults to current directory or '
                             '$OGGM_OUTDIR.')
    parser.add_argument('--logging-level', type=str, default='WORKFLOW',
                        help='the logging level to use (DEBUG, INFO, WARNING, '
                             'WORKFLOW).')
    parser.add_argument('--elev-bands', nargs='?', const=True, default=False,
                        help='compute the flowlines based on the Huss & Farinotti '
                             '2012 method.')
    parser.add_argument('--centerlines', nargs='?', const=True, default=False,
                        help='compute the flowlines based on the OGGM '
                             'centerline(s) method.')
    parser.add_argument('--mb-calibration-strategy', type=str,
                        default='informed_threestep',
                        help='how to calibrate the massbalance. Currently one of '
                             'informed_threestep (default) , melt_temp'
                             'or temp_melt.')
    parser.add_argument('--dem-source', type=str, default='',
                        help='which DEM source to use. Possible options are '
                             'the name of a specific DEM (e.g. RAMP, SRTM...) '
                             'or ALL, in which case all available DEMs will '
                             'be processed and adjoined with a suffix at the '
                             'end of the file name. The ALL option is only '
                             'compatible with level 1 folders, after which '
                             'the processing will stop. The default is to use '
                             'the default OGGM DEM.')
    parser.add_argument('--select-source-from-dir', type=str,
                        default=None,
                        help='if starting from a level 1 "ALL" or "STANDARD" DEM '
                        'sources directory, select the chosen DEM source here. '
                        'If you set it to "BY_RES" here, COPDEM will be used and '
                        'its resolution chosen based on the gdirs map resolution '
                        '(COPDEM30 for dx < 60 m, COPDEM90 elsewhere).')
    parser.add_argument('--keep-dem-folders', nargs='?', const=True, default=False,
                        help='if `select_source_from_dir` is used, wether to keep '
                        'the original DEM folders in or not.')
    parser.add_argument('--add-consensus-thickness', nargs='?', const=True, default=False,
                        help='adds (reprojects) the consensus thickness '
                             'estimates to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-itslive-velocity', nargs='?', const=True, default=False,
                        help='adds (reprojects) the ITS_LIVE velocity '
                             'estimates to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-millan-thickness', nargs='?', const=True, default=False,
                        help='adds (reprojects) the millan thickness '
                             'estimates to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-millan-velocity', nargs='?', const=True, default=False,
                        help='adds (reprojects) the millan velocity '
                             'estimates to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-hugonnet-dhdt', nargs='?', const=True, default=False,
                        help='adds (reprojects) the millan dhdt '
                             'maps to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-glathida', nargs='?', const=True, default=False,
                        help='adds (reprojects) the glathida point thickness '
                             'observations to the glacier directories. '
                             'The data points are stored as csv.')
    parser.add_argument('--demo', nargs='?', const=True, default=False,
                        help='if you want to run the prepro for the '
                             'list of demo glaciers.')
    parser.add_argument('--test', nargs='?', const=True, default=False,
                        help='if you want to do a test on a couple of '
                             'glaciers first.')
    parser.add_argument('--test-ids', nargs='+',
                        help='if --test, specify the RGI ids to run separated '
                             'by a space (default: 4 randomly selected).')
    parser.add_argument('--disable-mp', nargs='?', const=True, default=False,
                        help='if you want to disable multiprocessing.')
    parser.add_argument('--dynamic-spinup', type=str, default='',
                        help="include a dynamic spinup for matching glacier area "
                             "('area/dmdtda') OR volume ('volume/dmdtda') at "
                             "the RGI-date, AND mass-change from Hugonnet "
                             "in the period 2000-2020 (dynamic mu* "
                             "calibration).")
    parser.add_argument('--err-dmdtda-scaling-factor', type=float, default=0.2,
                        help="scaling factor to account for correlated "
                             "uncertainties of geodetic mass balance "
                             "observations when looking at regional scale. "
                             "Should be smaller or equal to 1.")
    parser.add_argument('--dynamic-spinup-start-year', type=int, default=1979,
                        help="if --dynamic-spinup is set, define the starting"
                             "year for the simulation. The default is 1979, "
                             "unless the climate data starts later.")
    parser.add_argument('--store-fl-diagnostics', nargs='?', const=True, default=False,
                        help="Also compute and store flowline diagnostics during "
                             "preprocessing. This can increase data usage quite "
                             "a bit.")
    parser.add_argument('--override-params', type=json.loads, default=None)

    args = parser.parse_args(args)

    # Check input
    rgi_reg = args.rgi_reg
    if args.demo:
        rgi_reg = 0
    if not rgi_reg and not args.demo:
        rgi_reg = os.environ.get('OGGM_RGI_REG', None)
        if rgi_reg is None:
            raise InvalidParamsError('--rgi-reg is required!')
    rgi_reg = '{:02}'.format(int(rgi_reg))
    ok_regs = ['{:02}'.format(int(r)) for r in range(1, 20)]
    if not args.demo and rgi_reg not in ok_regs:
        raise InvalidParamsError('--rgi-reg should range from 01 to 19!')

    rgi_version = args.rgi_version

    border = args.map_border
    if not border:
        border = os.environ.get('OGGM_MAP_BORDER', None)
        if border is None:
            raise InvalidParamsError('--map-border is required!')

    working_dir = args.working_dir
    if not working_dir:
        working_dir = os.environ.get('OGGM_WORKDIR', '')

    output_folder = args.output
    if not output_folder:
        output_folder = os.environ.get('OGGM_OUTDIR', '')

    border = int(border)
    output_folder = os.path.abspath(output_folder)
    working_dir = os.path.abspath(working_dir)

    dynamic_spinup = False if args.dynamic_spinup == '' else args.dynamic_spinup

    # All good
    return dict(rgi_version=rgi_version, rgi_reg=rgi_reg,
                border=border, output_folder=output_folder,
                working_dir=working_dir, params_file=args.params_file,
                is_test=args.test, test_ids=args.test_ids,
                demo=args.demo, dem_source=args.dem_source,
                start_level=args.start_level, start_base_url=args.start_base_url,
                max_level=args.max_level, disable_mp=args.disable_mp,
                logging_level=args.logging_level,
                elev_bands=args.elev_bands,
                centerlines=args.centerlines,
                select_source_from_dir=args.select_source_from_dir,
                keep_dem_folders=args.keep_dem_folders,
                add_consensus_thickness=args.add_consensus_thickness,
                add_millan_thickness=args.add_millan_thickness,
                add_itslive_velocity=args.add_itslive_velocity,
                add_millan_velocity=args.add_millan_velocity,
                add_hugonnet_dhdt=args.add_hugonnet_dhdt,
                add_glathida=args.add_glathida,
                dynamic_spinup=dynamic_spinup,
                err_dmdtda_scaling_factor=args.err_dmdtda_scaling_factor,
                dynamic_spinup_start_year=args.dynamic_spinup_start_year,
                mb_calibration_strategy=args.mb_calibration_strategy,
                store_fl_diagnostics=args.store_fl_diagnostics,
                override_params=args.override_params,
                )


def main():
    """Script entry point"""

    run_prepro_levels(**parse_args(sys.argv[1:]))
