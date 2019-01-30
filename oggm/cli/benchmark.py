"""Command line arguments to the oggm_benchmark command

Type `$ oggm_benchmark -h` for help

"""

# External modules
import os
import sys
import argparse
import time
import logging
import pandas as pd
import geopandas as gpd

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow, tasks
from oggm.exceptions import InvalidParamsError


def _add_time_to_df(df, index, t):
    df.loc[index, 't'] = t
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    df.loc[index, 'H'] = h
    df.loc[index, 'M'] = m
    df.loc[index, 'S'] = s


def run_benchmark(rgi_version=None, rgi_reg=None, border=None,
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
    odf = pd.DataFrame()

    if rgi_version is None:
        rgi_version = cfg.PARAMS['rgi_version']
    base_dir = os.path.join(output_folder)

    # Add a package version file
    utils.mkdir(base_dir)
    opath = os.path.join(base_dir, 'package_versions.txt')
    with open(opath, 'w') as vfile:
        vfile.write(utils.show_versions(logger=log))

    # Read RGI
    start = time.time()
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
        rgidf = rgidf.sample(2)
    _add_time_to_df(odf, 'Read RGI', time.time()-start)

    # Sort for more efficient parallel computing
    rgidf = rgidf.sort_values('Area', ascending=False)

    log.workflow('Starting prepro run for RGI reg: {} '
                 'and border: {}'.format(rgi_reg, border))
    log.workflow('Number of glaciers: {}'.format(len(rgidf)))

    # Input
    if test_topofile:
        cfg.PATHS['dem_file'] = test_topofile

    # Initialize working directories
    start = time.time()
    gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)
    _add_time_to_df(odf, 'init_glacier_regions', time.time()-start)

    # Pre-download other files just in case
    if test_crudir is None:
        _ = utils.get_cru_file(var='tmp')
        _ = utils.get_cru_file(var='pre')
    else:
        cfg.PATHS['cru_dir'] = test_crudir

    # Tasks
    task_list = [
        tasks.process_cru_data,
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
        tasks.init_present_time_glacier,
    ]
    for task in task_list:
        start = time.time()
        workflow.execute_entity_task(task, gdirs)
        _add_time_to_df(odf, task.__name__, time.time()-start)

    # Runs
    start = time.time()
    workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                                 nyears=250, bias=0, seed=0,
                                 output_filesuffix='_tstar')
    _add_time_to_df(odf, 'run_random_climate_tstar_250', time.time()-start)

    workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                                 nyears=250, y0=1995, seed=0,
                                 output_filesuffix='_commit')
    _add_time_to_df(odf, 'run_random_climate_commit_250', time.time()-start)

    # Compile results
    start = time.time()
    utils.compile_glacier_statistics(gdirs)
    _add_time_to_df(odf, 'compile_glacier_statistics', time.time()-start)

    start = time.time()
    utils.compile_climate_statistics(gdirs,
                                     add_climate_period=[1920, 1960, 2000])
    _add_time_to_df(odf, 'compile_climate_statistics', time.time()-start)

    start = time.time()
    utils.compile_run_output(gdirs, filesuffix='_tstar')
    _add_time_to_df(odf, 'compile_run_output_tstar', time.time()-start)

    start = time.time()
    utils.compile_run_output(gdirs, filesuffix='_commit')
    _add_time_to_df(odf, 'compile_run_output_commit', time.time()-start)

    # Log
    opath = os.path.join(base_dir, 'benchmarks_b{:03d}.csv'.format(border))
    odf.index.name = 'Task'
    odf.to_csv(opath)
    log.workflow('OGGM benchmarks is done!')


def parse_args(args):
    """Check input arguments and env variables"""

    # CLI args
    description = ('Run an OGGM benchmark on a selected RGI Region. '
                   'This writes a benchmark_{border}.txt file where '
                   'the results are summarized')
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

    run_benchmark(**parse_args(sys.argv[1:]))
