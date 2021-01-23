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
import pandas as pd
import numpy as np
import geopandas as gpd

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow, tasks, GlacierDirectory
from oggm.core import gis
from oggm.exceptions import InvalidParamsError, InvalidDEMError

# Module logger
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
                      disable_mp=False, params_file=None, elev_bands=False,
                      match_geodetic_mb=False, centerlines_only=False,
                      add_consensus=False, max_level=5,
                      logging_level='WORKFLOW', disable_dl_verify=False):
    """Does the actual job.

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
        which DEM source to use: default, SOURCE_NAME or ALL
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
        compute all flowlines based on the Huss&Hock 2015 method instead
        of the OGGM default, which is a mix of elev_bands and centerlines.
    centerlines_only : bool
        compute all flowlines based on the OGGM centerline(s) method instead
        of the OGGM default, which is a mix of elev_bands and centerlines.
    match_geodetic_mb : bool
        match the regional mass-balance estimates at the regional level
        (currently Hugonnet et al., 2020).
    add_consensus : bool
        adds (reprojects) the consensus estimates thickness to the glacier
        directories. With elev_bands=True, the data will also be binned.
    max_level : int
        the maximum pre-processing level before stopping
    logging_level : str
        the logging level to use (DEBUG, INFO, WARNING, WORKFLOW)
    disable_dl_verify : bool
        disable the hash verification of OGGM downloads
    """

    # TODO: temporarily silence Fiona and other deprecation warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Input check
    if max_level not in [1, 2, 3, 4, 5]:
        raise InvalidParamsError('max_level should be one of [1, 2, 3, 4, 5]')

    # Time
    start = time.time()

    def _time_log():
        # Log util
        m, s = divmod(time.time() - start, 60)
        h, m = divmod(m, 60)
        log.workflow('OGGM prepro_levels is done! Time needed: '
                     '{:02d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

    # Config Override Params
    params = {}

    # Local paths
    utils.mkdir(working_dir)
    params['working_dir'] = working_dir

    # Initialize OGGM and set up the run parameters
    cfg.initialize(file=params_file, params=params,
                   logging_level=logging_level,
                   future=True)

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = not disable_mp

    # How many grid points around the glacier?
    # Make it large if you expect your glaciers to grow large
    cfg.PARAMS['border'] = border

    # Set to True for operational runs
    cfg.PARAMS['continue_on_error'] = True

    # Check for the integrity of the files OGGM downloads at run time
    # For large files (e.g. using a 1 tif DEM like ALASKA) calculating the hash
    # takes a long time, so deactivating this can make sense
    cfg.PARAMS['dl_verify'] = not disable_dl_verify

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
        rgif = utils.get_rgi_intersects_region_file(rgi_reg,
                                                    version=rgi_version)
        cfg.set_intersects_db(rgif)

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
            rgidf = rgidf.loc[rgidf.RGIId.isin(test_ids)]
        else:
            rgidf = rgidf.sample(4)

    log.workflow('Starting prepro run for RGI reg: {} '
                 'and border: {}'.format(rgi_reg, border))
    log.workflow('Number of glaciers: {}'.format(len(rgidf)))

    # L0 - go
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

    # L1 - Add dem files
    if test_topofile:
        cfg.PATHS['dem_file'] = test_topofile

    # Which DEM source?
    if dem_source.upper() == 'ALL':
        # This is the complex one, just do the job and leave
        log.workflow('Running prepro on ALL sources')
        for i, s in enumerate(utils.DEM_SOURCES):
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

        # Compress all in output directory
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
    # Check which glaciers will be processed as what
    if elev_bands:
        gdirs_band = gdirs
        gdirs_cent = []
    elif centerlines_only:
        gdirs_band = []
        gdirs_cent = gdirs
    else:
        # Default is to mix
        # Curated list of large (> 50 km2) glaciers that don't run
        # (CFL error) mostly because the centerlines are crap
        # This is a really temporary fix until we have some better
        # solution here
        ids_to_bands = [
            'RGI60-01.13696', 'RGI60-03.01710', 'RGI60-01.13635',
            'RGI60-01.14443', 'RGI60-03.01678', 'RGI60-03.03274',
            'RGI60-01.17566', 'RGI60-03.02849', 'RGI60-01.16201',
            'RGI60-01.14683', 'RGI60-07.01506', 'RGI60-07.01559',
            'RGI60-03.02687', 'RGI60-17.00172', 'RGI60-01.23649',
            'RGI60-09.00077', 'RGI60-03.00994', 'RGI60-01.26738',
            'RGI60-03.00283', 'RGI60-01.16121', 'RGI60-01.27108',
            'RGI60-09.00132', 'RGI60-13.43483', 'RGI60-09.00069',
            'RGI60-14.04404', 'RGI60-17.01218', 'RGI60-17.15877',
            'RGI60-13.30888', 'RGI60-17.13796', 'RGI60-17.15825',
            'RGI60-01.09783']
        if rgi_reg == '19':
            gdirs_band = gdirs
            gdirs_cent = []
        else:
            gdirs_band = []
            gdirs_cent = []
            for gdir in gdirs:
                if gdir.is_icecap or gdir.rgi_id in ids_to_bands:
                    gdirs_band.append(gdir)
                else:
                    gdirs_cent.append(gdir)

    log.workflow('Start flowline processing with: '
                 'N centerline type: {}, '
                 'N elev bands type: {}.'
                 ''.format(len(gdirs_cent), len(gdirs_band)))

    # HH2015 method
    workflow.execute_entity_task(tasks.simple_glacier_masks, gdirs_band)

    # Centerlines OGGM
    workflow.execute_entity_task(tasks.glacier_masks, gdirs_cent)

    if add_consensus:
        from oggm.shop.bedtopo import add_consensus_thickness
        workflow.execute_entity_task(add_consensus_thickness, gdirs_band)
        workflow.execute_entity_task(add_consensus_thickness, gdirs_cent)

        # Elev bands with var data
        vn = 'consensus_ice_thickness'
        workflow.execute_entity_task(tasks.elevation_band_flowline,
                                     gdirs_band, bin_variables=vn)
        workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline,
                                     gdirs_band, bin_variables=vn)
    else:
        # HH2015 method without it
        task_list = [
            tasks.elevation_band_flowline,
            tasks.fixed_dx_elevation_band_flowline,
        ]
        for task in task_list:
            workflow.execute_entity_task(task, gdirs_band)

    # HH2015 method
    task_list = [
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs_band)

    # Centerlines OGGM
    task_list = [
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs_cent)

    # Glacier stats
    sum_dir = os.path.join(output_base_dir, 'L2', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

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
    # Climate
    workflow.execute_entity_task(tasks.process_climate_data, gdirs)
    if cfg.PARAMS['climate_qc_months'] > 0:
        workflow.execute_entity_task(tasks.historical_climate_qc, gdirs)
    workflow.execute_entity_task(tasks.local_t_star, gdirs)
    workflow.execute_entity_task(tasks.mu_star_calibration, gdirs)

    # Inversion: we match the consensus
    workflow.calibrate_inversion_from_consensus(gdirs,
                                                apply_fs_on_mismatch=True,
                                                error_on_mismatch=False)

    # Do we want to match geodetic estimates?
    # This affects only the bias so we can actually do this *after*
    # the inversion, but we really want to take calving into account here
    if match_geodetic_mb:
        workflow.match_regional_geodetic_mb(gdirs, rgi_reg)

    # We get ready for modelling
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    # Glacier stats
    sum_dir = os.path.join(output_base_dir, 'L3', 'summary')
    utils.mkdir(sum_dir)
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

    # L4 - No tasks: add some stats for consistency and make the dirs small
    sum_dir_L3 = sum_dir
    sum_dir = os.path.join(output_base_dir, 'L4', 'summary')
    utils.mkdir(sum_dir)

    # Copy L3 files for consistency
    for bn in ['glacier_statistics', 'climate_statistics',
               'fixed_geometry_mass_balance']:
        ipath = os.path.join(sum_dir_L3, bn + '_{}.csv'.format(rgi_reg))
        opath = os.path.join(sum_dir, bn + '_{}.csv'.format(rgi_reg))
        shutil.copyfile(ipath, opath)

    # Copy mini data to new dir
    mini_base_dir = os.path.join(working_dir, 'mini_perglacier')
    mini_gdirs = workflow.execute_entity_task(tasks.copy_to_basedir, gdirs,
                                              base_dir=mini_base_dir)

    # L4 OK - compress all in output directory
    log.workflow('L4 done. Writing to tar...')
    level_base_dir = os.path.join(output_base_dir, 'L4')
    workflow.execute_entity_task(utils.gdir_to_tar, mini_gdirs, delete=False,
                                 base_dir=level_base_dir)
    utils.base_dir_to_tar(level_base_dir)
    if max_level == 4:
        _time_log()
        return

    # L5 - spinup run in mini gdirs
    gdirs = mini_gdirs

    # Get end date. The first gdir might have blown up, try some others
    i = 0
    while True:
        if i >= len(gdirs):
            raise RuntimeError('Found no valid glaciers!')
        try:
            y0 = gdirs[i].get_climate_info()['baseline_hydro_yr_0']
            # One adds 1 because the run ends at the end of the year
            ye = gdirs[i].get_climate_info()['baseline_hydro_yr_1'] + 1
            break
        except BaseException:
            i += 1

    # OK - run
    workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
                                 min_ys=y0, ye=ye,
                                 output_filesuffix='_historical')

    # Now compile the output
    sum_dir = os.path.join(output_base_dir, 'L5', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'historical_run_output_{}.nc'.format(rgi_reg))
    utils.compile_run_output(gdirs, path=opath, input_filesuffix='_historical')

    # Glacier statistics we recompute here for error analysis
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # Other stats for consistency
    for bn in ['climate_statistics', 'fixed_geometry_mass_balance']:
        ipath = os.path.join(sum_dir_L3, bn + '_{}.csv'.format(rgi_reg))
        opath = os.path.join(sum_dir, bn + '_{}.csv'.format(rgi_reg))
        shutil.copyfile(ipath, opath)

    # Add the extended files
    pf = os.path.join(sum_dir, 'historical_run_output_{}.nc'.format(rgi_reg))
    mf = os.path.join(sum_dir, 'fixed_geometry_mass_balance_{}.csv'.format(rgi_reg))
    # This is crucial - extending calving only possible with L3 data!!!
    sf = os.path.join(sum_dir_L3, 'glacier_statistics_{}.csv'.format(rgi_reg))
    opath = os.path.join(sum_dir, 'historical_run_output_extended_{}.nc'.format(rgi_reg))
    utils.extend_past_climate_run(past_run_file=pf,
                                  fixed_geometry_mb_file=mf,
                                  glacier_statistics_file=sf,
                                  path=opath)

    # L5 OK - compress all in output directory
    log.workflow('L5 done. Writing to tar...')
    level_base_dir = os.path.join(output_base_dir, 'L5')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
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
                        help='compute the flowlines based on the Huss&Hock '
                             '2015 ethod instead of the OGGM default, which is '
                             'a mix of elev_bands and centerlines.')
    parser.add_argument('--centerlines-only', nargs='?', const=True, default=False,
                        help='compute the flowlines based on the OGGM '
                             'centerline(s) method instead of the OGGM '
                             'default, which is a mix of elev_bands and '
                             'centerlines.')
    parser.add_argument('--match-geodetic-mb', nargs='?', const=True, default=False,
                        help='match regional SMB values to geodetic estimates '
                             '(currently Hugonnet et al., 2020) '
                             'by shifting the SMB residual.')
    parser.add_argument('--dem-source', type=str, default='',
                        help='which DEM source to use. Possible options are '
                             'the name of a specific DEM (e.g. RAMP, SRTM...) '
                             'or ALL, in which case all available DEMs will '
                             'be processed and adjoined with a suffix at the '
                             'end of the file name. The ALL option is only '
                             'compatible with level 1 folders, after which '
                             'the processing will stop. The default is to use '
                             'the default OGGM DEM.')
    parser.add_argument('--add-consensus', nargs='?', const=True, default=False,
                        help='adds (reprojects) the consensus estimates '
                             'thickness to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--demo', nargs='?', const=True, default=False,
                        help='if you want to run the prepro for the '
                             'list of demo glaciers.')
    parser.add_argument('--test', nargs='?', const=True, default=False,
                        help='if you want to do a test on a couple of '
                             'glaciers first.')
    parser.add_argument('--test-ids', nargs='+',
                        help='if --test, specify the RGI ids to run separated '
                             'by a space (default: 4 randomly selected).')
    parser.add_argument('--disable-dl-verify', nargs='?', const=True,
                        default=False,
                        help='if used OGGM downloads will not be verified '
                             'against a hash sum.')
    parser.add_argument('--disable-mp', nargs='?', const=True, default=False,
                        help='if you want to disable multiprocessing.')
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

    # All good
    return dict(rgi_version=rgi_version, rgi_reg=rgi_reg,
                border=border, output_folder=output_folder,
                working_dir=working_dir, params_file=args.params_file,
                is_test=args.test, test_ids=args.test_ids,
                demo=args.demo, dem_source=args.dem_source,
                max_level=args.max_level, disable_mp=args.disable_mp,
                logging_level=args.logging_level, elev_bands=args.elev_bands,
                centerlines_only=args.centerlines_only,
                match_geodetic_mb=args.match_geodetic_mb,
                add_consensus=args.add_consensus,
                disable_dl_verify=args.disable_dl_verify
                )


def main():
    """Script entry point"""

    run_prepro_levels(**parse_args(sys.argv[1:]))
