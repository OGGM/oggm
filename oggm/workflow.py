"""Wrappers for the single tasks, multi processor handling."""
# Built ins
import logging
import os
from shutil import rmtree
import collections
from functools import partial
# External libs
import multiprocessing as mp

# Locals
import oggm
from oggm import cfg, tasks, utils

# MPI
try:
    import oggm.mpi as ogmpi
    _have_ogmpi = True
except ImportError:
    _have_ogmpi = False

# Module logger
log = logging.getLogger(__name__)

# Multiprocessing Pool
_mp_pool = None


def _init_pool_globals(_cfg_contents, global_lock):
    cfg.unpack_config(_cfg_contents)
    utils.lock = global_lock


def init_mp_pool(reset=False):
    """Necessary because at import time, cfg might be uninitialized"""
    global _mp_pool
    if _mp_pool and not reset:
        return _mp_pool
    cfg.CONFIG_MODIFIED = False
    if _mp_pool and reset:
        _mp_pool.terminate()
        _mp_pool = None
    cfg_contents = cfg.pack_config()
    global_lock = mp.Manager().Lock()
    mpp = cfg.PARAMS['mp_processes']
    if mpp == -1:
        try:
            mpp = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            log.info('Multiprocessing: using slurm allocated '
                     'processors (N={})'.format(mpp))
        except KeyError:
            mpp = mp.cpu_count()
            log.info('Multiprocessing: using all available '
                     'processors (N={})'.format(mpp))
    else:
        log.info('Multiprocessing: using the requested number of '
                 'processors (N={})'.format(mpp))
    _mp_pool = mp.Pool(mpp, initializer=_init_pool_globals,
                       initargs=(cfg_contents, global_lock))
    return _mp_pool


def _merge_dicts(*dicts):
    r = {}
    for d in dicts:
        r.update(d)
    return r


class _pickle_copier(object):
    """Pickleable alternative to functools.partial,
    Which is not pickleable in python2 and thus doesn't work
    with Multiprocessing."""

    def __init__(self, func, kwargs):
        self.call_func = func
        self.out_kwargs = kwargs

    def __call__(self, arg):
        if self.call_func:
            gdir = arg
            call_func = self.call_func
        else:
            call_func, gdir = arg
        try:
            if isinstance(gdir, collections.Sequence):
                gdir, gdir_kwargs = gdir
                gdir_kwargs = _merge_dicts(self.out_kwargs, gdir_kwargs)
                return call_func(gdir, **gdir_kwargs)
            else:
                return call_func(gdir, **self.out_kwargs)
        except Exception as e:
            try:
                err_msg = '({0}) exception occured while processing task ' \
                          '{1}: {2}'.format(gdir.rgi_id, call_func.__name__, str(e))
                raise RuntimeError(err_msg) from e
            except AttributeError:
                pass
            raise


def reset_multiprocessing():
    """Reset multiprocessing state

    Call this if you changed configuration parameters mid-run and need them to
    be re-propagated to child processes.
    """
    global _mp_pool
    if _mp_pool:
        _mp_pool.terminate()
        _mp_pool = None
    cfg.CONFIG_MODIFIED = False


def execute_entity_task(task, gdirs, **kwargs):
    """Execute a task on gdirs.

    If you asked for multiprocessing, it will do it.

    Parameters
    ----------
    task : function
         the entity task to apply
    gdirs : list
         the list of oggm.GlacierDirectory to process.
    """

    # If not iterable it's ok
    try:
        len(gdirs)
    except TypeError:
        gdirs = [gdirs]

    if task.__dict__.get('global_task', False):
        return task(gdirs, **kwargs)

    pc = _pickle_copier(task, kwargs)

    if _have_ogmpi:
        if ogmpi.OGGM_MPI_COMM is not None:
            ogmpi.mpi_master_spin_tasks(pc, gdirs)
            return

    if cfg.PARAMS['use_multiprocessing']:
        mppool = init_mp_pool(cfg.CONFIG_MODIFIED)
        mppool.map(pc, gdirs, chunksize=1)
    else:
        for gdir in gdirs:
            pc(gdir)


def execute_parallel_tasks(gdir, tasks):
    """Execute a list of task on a single gdir (experimental!).

    This is useful when running a non-sequential list of task on a gdir,
    mostly for e.g. different experiments with different output files.

    Parameters
    ----------
    gdirs : oggm.GlacierDirectory
         the directory to process.
    tasks : list
         the the list of entity tasks to apply.
         Optionally, each list element can be a tuple, with the first element
         being the task, and the second element a dict that
         will be passed to the task function as ``**kwargs``.
    """

    pc = _pickle_copier(None, {})

    _tasks = []
    for task in tasks:
        kwargs = {}
        if isinstance(task, collections.Sequence):
            task, kwargs = task
        _tasks.append((task, (gdir, kwargs)))

    if _have_ogmpi:
        if ogmpi.OGGM_MPI_COMM is not None:
            ogmpi.mpi_master_spin_tasks(pc, _tasks)
            return

    if cfg.PARAMS['use_multiprocessing']:
        mppool = init_mp_pool(cfg.CONFIG_MODIFIED)
        mppool.map(pc, _tasks, chunksize=1)
    else:
        for task in _tasks:
            task()


def init_glacier_regions(rgidf=None, reset=False, force=False):
    """Very first task to do (always).

    Set reset=True in order to delete the content of the directories.
    """

    if reset and not force:
        reset = utils.query_yes_no('Delete all glacier directories?')

    # if reset delete also the log directory
    if reset:
        fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
        if os.path.exists(fpath):
            rmtree(fpath)

    gdirs = []
    new_gdirs = []
    if rgidf is None:
        if reset:
            raise ValueError('Cannot use reset without a rgi file')
        # The dirs should be there already
        gl_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')
        for root, _, files in os.walk(gl_dir):
            if files and ('dem.tif' in files):
                gdirs.append(oggm.GlacierDirectory(os.path.basename(root)))
    else:
        for _, entity in rgidf.iterrows():
            gdir = oggm.GlacierDirectory(entity, reset=reset)
            if not os.path.exists(gdir.get_filepath('dem')):
                new_gdirs.append((gdir, dict(entity=entity)))
            gdirs.append(gdir)

    # If not initialized, run the task in parallel
    execute_entity_task(tasks.define_glacier_region, new_gdirs)

    return gdirs


def gis_prepro_tasks(gdirs):
    """Helper function: run all flowlines tasks."""

    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction
    ]
    for task in task_list:
        execute_entity_task(task, gdirs)


def climate_tasks(gdirs):
    """Helper function: run all climate tasks."""

    # If not iterable it's ok
    try:
        len(gdirs)
    except TypeError:
        gdirs = [gdirs]

    # Which climate should we use?
    if cfg.PARAMS['baseline_climate'] == 'CRU':
        _process_task = tasks.process_cru_data
    elif cfg.PARAMS['baseline_climate'] == 'CUSTOM':
        _process_task = tasks.process_custom_climate_data
    elif cfg.PARAMS['baseline_climate'] == 'HISTALP':
        _process_task = tasks.process_histalp_data
    else:
        raise ValueError('baseline_climate parameter not understood')

    execute_entity_task(_process_task, gdirs)

    # Then, calibration?
    if cfg.PARAMS['run_mb_calibration']:
        tasks.compute_ref_t_stars(gdirs)

    # Mustar and the apparent mass-balance
    execute_entity_task(tasks.local_mustar, gdirs)
    execute_entity_task(tasks.apparent_mb, gdirs)


def inversion_tasks(gdirs):
    """Helper function: run all bed inversion tasks."""

    # Init
    execute_entity_task(tasks.prepare_for_inversion, gdirs)

    # Global task
    if cfg.PARAMS['optimize_inversion_params']:
        tasks.optimize_inversion_params(gdirs)

    # Inversion for all glaciers
    execute_entity_task(tasks.volume_inversion, gdirs)

    # Filter
    execute_entity_task(tasks.filter_inversion_output, gdirs)
