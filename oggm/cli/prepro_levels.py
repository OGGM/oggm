"""Command line arguments to the oggm_prepro command

Type `$ oggm_prepro -h` for help

"""

# External modules
import os
import sys
import argparse
import time
import logging
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
                      disable_mp=False, timeout=0, params_file=None,
                      max_level=4, logging_level='WORKFLOW',
                      map_dmax=None, map_d1=None, disable_dl_verify=False):
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
    max_level : int
        the maximum pre-processing level before stopping
    logging_level : str
        the logging level to use (DEBUG, INFO, WARNING, WORKFLOW)
    map_dmax : float
        maximum resolution [m] of spatial grid resolution
    map_d1 : float
        equation parameter which is used to calculate the grid resolution
    disable_dl_verify : bool
        disable the hash verification of OGGM downloads
    """

    # TODO: temporarily silence Fiona deprecation warnings
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

    # Size of the spatial map
    cfg.PARAMS['dmax'] = map_dmax if map_dmax else cfg.PARAMS['dmax']
    cfg.PARAMS['d1'] = map_d1 if map_d1 else cfg.PARAMS['d1']

    # Set to True for operational runs
    cfg.PARAMS['continue_on_error'] = True

    # Timeout
    cfg.PARAMS['task_timeout'] = timeout

    # Check for the integrity of the files OGGM downloads at run time
    # For large files (e.g. using a 1 tif DEM like ALASKA) calculating the hash
    # takes a long time, so deactivating this can make sense
    cfg.PARAMS['dl_verify'] = not disable_dl_verify

    # For statistics
    climate_periods = [1920, 1960, 2000]

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
    l_base_dir = os.path.join(base_dir, 'L1')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=l_base_dir)
    utils.base_dir_to_tar(l_base_dir)
    if max_level == 1:
        _time_log()
        return

    # L2 - Tasks
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
        workflow.execute_entity_task(task, gdirs)

    # Glacier stats
    sum_dir = os.path.join(base_dir, 'L2', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L2 OK - compress all in output directory
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
        tasks.local_t_star,
        tasks.mu_star_calibration,
        tasks.prepare_for_inversion,
        tasks.mass_conservation_inversion,
        tasks.filter_inversion_output,
        tasks.init_present_time_glacier,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs)

    # Glacier stats
    sum_dir = os.path.join(base_dir, 'L3', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)
    opath = os.path.join(sum_dir, 'climate_statistics_{}.csv'.format(rgi_reg))
    utils.compile_climate_statistics(gdirs, add_climate_period=climate_periods,
                                     path=opath)

    # L3 OK - compress all in output directory
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
    parser.add_argument('--dem-source', type=str, default='',
                        help='which DEM source to use. Possible options are '
                             'the name of a specific DEM (e.g. RAMP, SRTM...) '
                             'or ALL, in which case all available DEMs will '
                             'be processed and adjoined with a suffix at the '
                             'end of the file name. The ALL option is only '
                             'compatible with level 1 folders, after which '
                             'the processing will stop. The default is to use '
                             'the default OGGM DEM.')
    parser.add_argument('--disable-mp', nargs='?', const=True, default=False,
                        help='if you want to disable multiprocessing.')
    parser.add_argument('--timeout', type=int, default=0,
                        help='apply a timeout to the entity tasks '
                             '(in seconds).')
    parser.add_argument('--demo', nargs='?', const=True, default=False,
                        help='if you want to run the prepro for the '
                             'list of demo glaciers.')
    parser.add_argument('--test', nargs='?', const=True, default=False,
                        help='if you want to do a test on a couple of '
                             'glaciers first.')
    parser.add_argument('--test-ids', nargs='+',
                        help='if --test, specify the RGI ids to run separated '
                             'by a space (default: 4 randomly selected).')
    parser.add_argument('--logging-level', type=str, default='WORKFLOW',
                        help='the logging level to use (DEBUG, INFO, WARNING, '
                             'WORKFLOW).')
    parser.add_argument('--map-dmax', type=float,
                        help='maximal resolution of the spatial grid. Defaults'
                             ' to value from params.cfg.')
    parser.add_argument('--map-d1', type=float,
                        help='d1 parameter to calculate the resolution of the '
                             'spatial grid. Defaults to value from '
                             'params.cfg.')
    parser.add_argument('--disable-dl-verify', nargs='?', const=True,
                        default=False,
                        help='if used OGGM downloads will not be verified '
                             'against a hash sum.')
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
                max_level=args.max_level, timeout=args.timeout,
                disable_mp=args.disable_mp, logging_level=args.logging_level,
                map_dmax=args.map_dmax, map_d1=args.map_d1,
                disable_dl_verify=args.disable_dl_verify
                )


def main():
    """Script entry point"""

    run_prepro_levels(**parse_args(sys.argv[1:]))
