"""Command line arguments to the oggm_prepro command

Type `$ oggm_prepro -h` for help

"""

# External modules
import os
import sys
import argparse
import time
import logging
import geopandas as gpd

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow, tasks
from oggm.exceptions import InvalidParamsError


def run_prepro_levels(rgi_version=None, rgi_reg=None, border=None,
                      output_folder='', working_dir='', is_test=False,
                      test_rgidf=None, test_intersects_file=None,
                      test_topofile=None, test_crudir=None):
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
    working_dir : str
        path to the OGGM working directory
    is_test : bool
        to test on a couple of glaciers only!
    test_rgidf : shapefile
        for testing purposes only
    test_intersects_file : shapefile
        for testing purposes only
    test_topofile : str
        for testing purposes only
    test_crudir : str
        for testing purposes only
    """

    # TODO: temporarily silence Fiona deprecation warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Module logger
    log = logging.getLogger(__name__)

    # Time
    start = time.time()

    # Initialize OGGM and set up the run parameters
    cfg.initialize(logging_level='WORKFLOW')

    # Local paths
    utils.mkdir(working_dir)
    cfg.PATHS['working_dir'] = working_dir

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # How many grid points around the glacier?
    # Make it large if you expect your glaciers to grow large
    cfg.PARAMS['border'] = border

    # Set to True for operational runs
    cfg.PARAMS['continue_on_error'] = True

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

    if test_rgidf is None:
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
        # Just for fun
        rgidf = rgidf.sample(4)

    # Sort for more efficient parallel computing
    rgidf = rgidf.sort_values('Area', ascending=False)

    log.workflow('Starting prepro run for RGI reg: {} '
                 'and border: {}'.format(rgi_reg, border))
    log.workflow('Number of glaciers: {}'.format(len(rgidf)))

    # Input
    if test_topofile:
        cfg.PATHS['dem_file'] = test_topofile

    # L1 - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)

    # Glacier stats
    sum_dir = os.path.join(base_dir, 'L1', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L1 OK - compress all in output directory
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=os.path.join(base_dir, 'L1'))

    # L2 - Tasks
    # Pre-download other files just in case
    if test_crudir is None:
        _ = utils.get_cru_file(var='tmp')
        _ = utils.get_cru_file(var='pre')
    else:
        cfg.PATHS['cru_dir'] = test_crudir

    workflow.execute_entity_task(tasks.process_cru_data, gdirs)

    # Glacier stats
    sum_dir = os.path.join(base_dir, 'L2', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L2 OK - compress all in output directory
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=os.path.join(base_dir, 'L2'))

    # L3 - Tasks
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)
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
        tasks.local_t_star,
        tasks.mu_star_calibration,
        tasks.prepare_for_inversion,
        tasks.mass_conservation_inversion,
        tasks.filter_inversion_output,
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
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=os.path.join(base_dir, 'L3'))

    # L4 - Tasks
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    # Glacier stats
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

    # Log
    m, s = divmod(time.time() - start, 60)
    h, m = divmod(m, 60)
    log.workflow('OGGM prepro_levels is done! Time needed: '
                 '{:02d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))


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
    parser.add_argument('--working-dir', type=str,
                        help='path to the directory where to write the '
                             'output. Defaults to current directory or '
                             '$OGGM_WORKDIR.')
    parser.add_argument('--output', type=str,
                        help='path to the directory where to write the '
                             'output. Defaults to current directory or'
                             '$OGGM_OUTDIR.')
    parser.add_argument('--test', nargs='?', const=True, default=False,
                        help='if you want to do a test on a couple of '
                             'glaciers first.')
    args = parser.parse_args(args)

    # Check input
    rgi_reg = args.rgi_reg
    if not rgi_reg:
        rgi_reg = os.environ.get('OGGM_RGI_REG', None)
        if rgi_reg is None:
            raise InvalidParamsError('--rgi-reg is required!')
    rgi_reg = '{:02}'.format(int(rgi_reg))

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
                working_dir=working_dir, is_test=args.test)


def main():
    """Script entry point"""

    run_prepro_levels(**parse_args(sys.argv[1:]))
