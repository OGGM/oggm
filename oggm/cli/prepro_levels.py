"""Command line arguments to the oggm_prepro command

Type `$ oggm_prepro -h` for help

"""

# External modules
import os
import sys
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
                      match_zemp=False, centerlines_only=False, max_level=4,
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
    if max_level not in [1, 2, 3, 4]:
        raise InvalidParamsError('max_level should be one of [1, 2, 3, 4]')

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
                   logging_level=logging_level)

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
    rgi_dir_name = 'RGI{}'.format(rgi_version)
    border_dir_name = 'b_{:03d}'.format(border)
    base_dir = os.path.join(output_folder, rgi_dir_name, border_dir_name)

    # Add a package version file
    utils.mkdir(base_dir)
    opath = os.path.join(base_dir, 'package_versions.txt')
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

        # Not that in matters much but in AA almost all large ice bodies
        # are actually ice caps
        if rgi_reg == '19':
            rgidf.loc[rgidf.Area > 100, 'Form'] = '1'

    else:
        rgidf = test_rgidf
        cfg.set_intersects_db(test_intersects_file)

    if is_test:
        if test_ids is not None:
            rgidf = rgidf.loc[rgidf.RGIId.isin(test_ids)]
        else:
            rgidf = rgidf.sample(4)

    # Sort for more efficient parallel computing
    rgidf = rgidf.sort_values('Area', ascending=False)

    log.workflow('Starting prepro run for RGI reg: {} '
                 'and border: {}'.format(rgi_reg, border))
    log.workflow('Number of glaciers: {}'.format(len(rgidf)))

    # L0 - go
    gdirs = workflow.init_glacier_directories(rgidf, reset=True, force=True)

    # Glacier stats
    sum_dir = os.path.join(base_dir, 'L0', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L0 OK - compress all in output directory
    log.workflow('L0 done. Writing to tar...')
    l_base_dir = os.path.join(base_dir, 'L0')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=l_base_dir)
    utils.base_dir_to_tar(l_base_dir)
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
        l_base_dir = os.path.join(base_dir, 'L1')
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                     base_dir=l_base_dir)
        utils.base_dir_to_tar(l_base_dir)

        _time_log()
        return

    # Force a given source
    source = dem_source.upper() if dem_source else None

    # L1 - go
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                 source=source)

    # Glacier stats
    sum_dir = os.path.join(base_dir, 'L1', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L1 OK - compress all in output directory
    log.workflow('L1 done. Writing to tar...')
    l_base_dir = os.path.join(base_dir, 'L1')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=l_base_dir)
    utils.base_dir_to_tar(l_base_dir)
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
        if rgi_reg == '19':
            gdirs_band = gdirs
            gdirs_cent = []
        else:
            gdirs_band = []
            gdirs_cent = []
            for gdir in gdirs:
                if gdir.is_icecap:
                    gdirs_band.append(gdir)
                else:
                    gdirs_cent.append(gdir)

    log.workflow('Start flowline processing with: '
                 'N centerline type: {}, '
                 'N elev bands type: {}.'
                 ''.format(len(gdirs_cent), len(gdirs_band)))

    # HH2015 method
    task_list = [
        tasks.simple_glacier_masks,
        tasks.elevation_band_flowline,
        tasks.fixed_dx_elevation_band_flowline,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs_band)

    # Centerlines OGGM
    task_list = [
        tasks.glacier_masks,
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
    sum_dir = os.path.join(base_dir, 'L2', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L2 OK - compress all in output directory
    log.workflow('L2 done. Writing to tar...')
    l_base_dir = os.path.join(base_dir, 'L2')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=l_base_dir)
    utils.base_dir_to_tar(l_base_dir)
    if max_level == 2:
        _time_log()
        return

    # L3 - Tasks
    task_list = [
        tasks.process_climate_data,
        tasks.historical_climate_qc,
        tasks.local_t_star,
        tasks.mu_star_calibration,
        tasks.prepare_for_inversion,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs)

    # Do we want to match Zemp et al?
    if match_zemp:
        df = utils.compile_fixed_geometry_mass_balance(gdirs, path=False)
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        dfs = utils.compile_glacier_statistics(gdirs, path=False)
        odf = pd.DataFrame(df.loc[2006:2016].mean(), columns=['SMB'])
        odf['AREA'] = dfs.rgi_area_km2

        area_oggm = odf['AREA'].sum()
        smb_oggm = np.average(odf['SMB'], weights=odf['AREA'])

        df = pd.read_csv(utils.get_demo_file('zemp_ref_2006_2016.csv'),
                         index_col=0)
        area_zemp = df.loc[int(rgi_reg), 'Area']
        smb_zemp = df.loc[int(rgi_reg), 'SMB'] * 1000
        if not np.allclose(area_oggm, area_zemp, rtol=0.05):
            log.warning('OGGM regional area and Zemp regional area differ '
                        'by more than 5%.')
        residual = smb_zemp - smb_oggm
        log.workflow('Shifting regional bias by {}'.format(residual))
        for gdir in gdirs:
            try:
                df = gdir.read_json('local_mustar')
                df['bias'] = df['bias'] - residual
                gdir.write_json(df, 'local_mustar')
            except FileNotFoundError:
                pass

    # Inversion: we match the consensus
    workflow.execute_entity_task(tasks.prepare_for_inversion, gdirs)
    workflow.calibrate_inversion_from_consensus(gdirs,
                                                apply_fs_on_mismatch=True,
                                                error_on_mismatch=False)

    # We get ready for modelling
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    # Glacier stats
    sum_dir = os.path.join(base_dir, 'L3', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)
    opath = os.path.join(sum_dir, 'climate_statistics_{}.csv'.format(rgi_reg))
    utils.compile_climate_statistics(gdirs, path=opath)
    opath = os.path.join(sum_dir, 'fixed_geometry_mass_balance_{}.csv'.format(rgi_reg))
    utils.compile_fixed_geometry_mass_balance(gdirs, path=opath)

    # L3 OK - compress all in output directory
    log.workflow('L3 done. Writing to tar...')
    l_base_dir = os.path.join(base_dir, 'L3')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=l_base_dir)
    utils.base_dir_to_tar(l_base_dir)
    if max_level == 3:
        _time_log()
        return

    # L4 - No tasks: add some stats for consistency and make the dirs small
    sum_dir = os.path.join(base_dir, 'L4', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # Copy mini data to new dir
    log.workflow('L4 done. Writing to tar...')
    base_dir = os.path.join(base_dir, 'L4')
    mini_gdirs = workflow.execute_entity_task(tasks.copy_to_basedir, gdirs,
                                              base_dir=base_dir)

    # L4 OK - compress all in output directory
    workflow.execute_entity_task(utils.gdir_to_tar, mini_gdirs, delete=True)
    utils.base_dir_to_tar(base_dir)

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
    parser.add_argument('--max-level', type=int, default=4,
                        help='the maximum level you want to run the '
                             'pre-processing for (1, 2, 3 or 3).')
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
    parser.add_argument('--match-zemp', nargs='?', const=True, default=False,
                        help='match regional SMB values to Zemp et al., 2019 '
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
                match_zemp=args.match_zemp,
                disable_dl_verify=args.disable_dl_verify
                )


def main():
    """Script entry point"""

    run_prepro_levels(**parse_args(sys.argv[1:]))
