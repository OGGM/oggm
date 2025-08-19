"""Wrappers for the single tasks, multi processor handling."""
# Built ins
import logging
import os
import shutil
from collections.abc import Sequence
# External libs
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize as optimization

# Locals
import oggm
from oggm import cfg, tasks, utils
from oggm.core import centerlines, flowline, climate, gis
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.utils import global_task

# MPI
try:
    import oggm.mpi as ogmpi
    _have_ogmpi = True
except ImportError:
    _have_ogmpi = False

# Module logger
log = logging.getLogger(__name__)

# Multiprocessing Pool
_mp_manager = None
_mp_pool = None


def _init_pool_globals(_cfg_contents, global_lock):
    cfg.unpack_config(_cfg_contents)
    utils.lock = global_lock


def init_mp_pool(reset=False):
    """Necessary because at import time, cfg might be uninitialized"""
    global _mp_manager, _mp_pool
    if _mp_pool and _mp_manager and not reset:
        return _mp_pool

    cfg.CONFIG_MODIFIED = False
    if _mp_pool:
        _mp_pool.terminate()
        _mp_pool = None
    if _mp_manager:
        cfg.set_manager(None)
        _mp_manager.shutdown()
        _mp_manager = None

    if cfg.PARAMS['use_mp_spawn']:
        mp = multiprocessing.get_context('spawn')
    else:
        mp = multiprocessing

    _mp_manager = mp.Manager()

    cfg.set_manager(_mp_manager)
    cfg_contents = cfg.pack_config()

    global_lock = _mp_manager.Lock()

    mpp = cfg.PARAMS['mp_processes']
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

    def _call_internal(self, call_func, gdir, kwargs):
        # If the function is None, assume gdir is tuple with task function
        if not call_func:
            call_func, gdir = gdir

        # Merge main kwargs with per-task kwargs
        kwargs = _merge_dicts(self.out_kwargs, kwargs)

        # If gdir is a sequence, again assume it's a tuple with per-gdir kwargs.
        if isinstance(gdir, Sequence) and not isinstance(gdir, str):
            gdir, gdir_kwargs = gdir
            kwargs.update(gdir_kwargs)

        return call_func(gdir, **kwargs)

    def __call__(self, arg):
        res = None
        for func in self.call_func:
            func, kwargs = func
            res = self._call_internal(func, arg, kwargs)
        return res


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

    If ``task`` has more arguments than `gdir` they have to be keyword
    arguments.

    Parameters
    ----------
    task : function or sequence of functions
         The entity task(s) to apply.
         Can be None, in which case each gdir is expected to be a tuple of (task, gdir).
         When passing a sequence, each item can also optionally be a tuple of (task, dictionary).
         In this case the dictionary items will be passed to the task as kwargs.
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        The glacier directories to process.
        Each individual gdir can optionally be a tuple of (gdir, dictionary).
        In this case, the values in the dictionary will be passed to the task as
        keyword arguments for that specific gdir.

    Returns
    -------
    List of results from task. Last task if a list of tasks was given.
    """

    # Normalize task into list of tuples for simplicity
    if not isinstance(task, Sequence):
        task = [task]
    tasks = []
    for t in task:
        if isinstance(t, tuple):
            tasks.append(t)
        else:
            tasks.append((t, {}))

    # Reject global tasks
    for t in tasks:
        if t[0].__dict__.get('is_global_task', False):
            raise InvalidWorkflowError('execute_entity_task cannot be used on '
                                       'global tasks.')

    # Should be iterable
    gdirs = utils.tolist(gdirs)
    ng = len(gdirs)
    if ng == 0:
        log.workflow('Called execute_entity_task on 0 glaciers. Returning...')
        return

    log.workflow('Execute entity tasks [%s] on %d glaciers',
                 ', '.join([t[0].__name__ for t in tasks]), ng)

    pc = _pickle_copier(tasks, kwargs)

    if _have_ogmpi:
        if ogmpi.OGGM_MPI_COMM is not None:
            return ogmpi.mpi_master_spin_tasks(pc, gdirs)

    if cfg.PARAMS['use_multiprocessing'] and ng > 1:
        mppool = init_mp_pool(cfg.CONFIG_MODIFIED)
        out = mppool.map(pc, gdirs, chunksize=1)
    else:
        if ng > 3:
            log.workflow('WARNING: you are trying to run an entity task on '
                         '%d glaciers with multiprocessing turned off. OGGM '
                         'will run faster with multiprocessing turned on.', ng)
        out = [pc(gdir) for gdir in gdirs]

    return out


def execute_parallel_tasks(gdir, tasks):
    """Execute a list of task on a single gdir (experimental!).

    This is useful when running a non-sequential list of task on a gdir,
    mostly for e.g. different experiments with different output files.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
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
        if isinstance(task, Sequence):
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
        for task, (gd, kw) in _tasks:
            task(gd, **kw)


def gdir_from_prepro(entity, from_prepro_level=None,
                     prepro_border=None, prepro_rgi_version=None,
                     base_url=None):

    if prepro_border is None:
        prepro_border = int(cfg.PARAMS['border'])
    if prepro_rgi_version is None:
        prepro_rgi_version = cfg.PARAMS['rgi_version']

    if isinstance(entity, pd.Series):
        try:
            rid = entity.RGIId
        except AttributeError:
            rid = entity.rgi_id
    else:
        rid = entity

    tar_base = utils.get_prepro_gdir(prepro_rgi_version, rid, prepro_border,
                                     from_prepro_level, base_url=base_url)
    from_tar = os.path.join(tar_base.replace('.tar', ''), rid + '.tar.gz')
    return oggm.GlacierDirectory(entity, from_tar=from_tar)


def gdir_from_tar(entity, from_tar):

    try:
        rgi_id = entity.RGIId
    except AttributeError:
        rgi_id = entity

    from_tar = os.path.join(from_tar, '{}'.format(rgi_id[:8]),
                            '{}.tar' .format(rgi_id[:11]))
    assert os.path.exists(from_tar), 'tarfile does not exist'
    from_tar = os.path.join(from_tar.replace('.tar', ''), rgi_id + '.tar.gz')
    return oggm.GlacierDirectory(entity, from_tar=from_tar)


def _check_rgi_input(rgidf=None, err_on_lvl2=False):
    """Complain if the input has duplicates."""

    if rgidf is None:
        return

    msg = ('You have glaciers with connectivity level 2 in your list. '
           'OGGM does not provide pre-processed directories for these.')

    # Check if dataframe or list of strs
    is_dataframe = isinstance(rgidf, pd.DataFrame)
    if is_dataframe:
        try:
            rgi_ids = rgidf.RGIId
            # if dataframe we can also check for connectivity
            if 'Connect' in rgidf and np.any(rgidf['Connect'] == 2):
                if err_on_lvl2:
                    raise RuntimeError(msg)
        except AttributeError:
            # RGI7
            rgi_ids = rgidf.rgi_id
    else:
        rgi_ids = utils.tolist(rgidf)
        # Check for Connectivity level 2 here as well
        not_good_ids = pd.read_csv(utils.get_demo_file('rgi6_ids_conn_lvl2.csv'),
                                   index_col=0)
        try:
            if err_on_lvl2 and len(not_good_ids.loc[rgi_ids]) > 0:
                raise RuntimeError(msg)
        except KeyError:
            # Were good
            pass

    u, c = np.unique(rgi_ids, return_counts=True)
    if len(u) < len(rgi_ids):
        raise InvalidWorkflowError('Found duplicates in the list of '
                                   'RGI IDs: {}'.format(u[c > 1]))


def _isdir(path):
    """os.path.isdir, returning False instead of an error on non-string/path-like objects
    """
    try:
        return os.path.isdir(path)
    except TypeError:
        return False


def init_glacier_directories(rgidf=None, *, reset=False, force=False,
                             from_prepro_level=None, prepro_border=None,
                             prepro_rgi_version=None, prepro_base_url=None,
                             from_tar=False, delete_tar=False):
    """Initializes the list of Glacier Directories for this run.

    This is the very first task to do (always). If the directories are already
    available in the working directory, use them. If not, create new ones.

    **Careful**: when starting from a pre-processed directory with
    `from_prepro_level` or `from_tar`, the existing directories will be overwritten!

    Parameters
    ----------
    rgidf : GeoDataFrame or list of ids, optional for pre-computed runs
        the RGI glacier outlines. If unavailable, OGGM will parse the
        information from the glacier directories found in the working
        directory. It is required for new runs.
    reset : bool
        delete the existing glacier directories if found.
    force : bool
        setting `reset=True` will trigger a yes/no question to the user. Set
        `force=True` to avoid this.
    from_prepro_level : int
        get the gdir data from the official pre-processed pool. If this
        argument is set, the existing directories will be overwritten!
    prepro_border : int
        for `from_prepro_level` only: if you want to override the default
        behavior which is to use `cfg.PARAMS['border']`
    prepro_rgi_version : str
        for `from_prepro_level` only: if you want to override the default
        behavior which is to use `cfg.PARAMS['rgi_version']`
    prepro_base_url : str
        for `from_prepro_level` only: the preprocessed directory url from
        which to download the directories (became mandatory in OGGM v1.6)
    from_tar : bool or str, default=False
        extract the gdir data from a tar file. If set to `True`,
        will check for a tar file at the expected location in `base_dir`.
        delete the original tar file after extraction. If this
        argument is set, the existing directories will be overwritten!

    Returns
    -------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the initialised glacier directories
    """

    _check_rgi_input(rgidf, err_on_lvl2=from_prepro_level)

    if reset and not force:
        reset = utils.query_yes_no('Delete all glacier directories?')

    if from_prepro_level:
        url = utils.get_prepro_base_url(base_url=prepro_base_url,
                                        border=prepro_border,
                                        prepro_level=from_prepro_level,
                                        rgi_version=prepro_rgi_version)
        if cfg.PARAMS['has_internet'] and not utils.url_exists(url):
            raise InvalidParamsError("base url seems unreachable with these "
                                     "parameters: {}".format(url))
        if ('oggm_v1.4' in url and
                from_prepro_level >= 3 and
                not cfg.PARAMS['prcp_fac']):
            log.warning('You seem to be using v1.4 directories with a more '
                        'recent version of OGGM. While this is possible, be '
                        'aware that some defaults parameters have changed. '
                        'See the documentation for details: '
                        'http://docs.oggm.org/en/stable/whats-new.html')

    # if reset delete also the log directory
    if reset:
        fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
        if os.path.exists(fpath):
            shutil.rmtree(fpath)

    if rgidf is None:
        # Infer the glacier directories from folders available in working dir
        if reset:
            raise ValueError('Cannot use reset without setting rgidf')
        log.workflow('init_glacier_directories by parsing all available '
                     'folders (this takes time: if possible, provide rgidf '
                     'instead).')
        # The dirs should be there already
        gl_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')
        gdirs = []
        for root, _, files in os.walk(gl_dir):
            if files and ('outlines.shp' in files or
                          'outlines.tar.gz' in files):
                gdirs.append(oggm.GlacierDirectory(os.path.basename(root)))
    else:
        # Create glacier directories from input
        # Check if dataframe or list of str
        try:
            entities = []
            for _, entity in rgidf.iterrows():
                entities.append(entity)
        except AttributeError:
            entities = utils.tolist(rgidf)

        if from_prepro_level is not None:
            log.workflow('init_glacier_directories from prepro level {} on '
                         '{} glaciers.'.format(from_prepro_level,
                                               len(entities)))
            # Read the hash dictionary before we use multiproc
            if cfg.PARAMS['dl_verify']:
                utils.get_dl_verify_data('cluster.klima.uni-bremen.de')
            gdirs = execute_entity_task(gdir_from_prepro, entities,
                                        from_prepro_level=from_prepro_level,
                                        prepro_border=prepro_border,
                                        prepro_rgi_version=prepro_rgi_version,
                                        base_url=prepro_base_url)
        else:
            # We can set the intersects file automatically here
            if (cfg.PARAMS['use_intersects'] and
                    len(cfg.PARAMS['intersects_gdf']) == 0 and
                    not from_tar):
                try:
                    rgi_ids = np.unique(np.sort([entity.rgi_id for entity in
                                                 entities]))
                    if len(rgi_ids[0]) == 23:
                        # RGI7
                        assert rgi_ids[0].split('-')[1] == 'v7.0'
                        if rgi_ids[0].split('-')[2] == 'C':
                            # No need for interstects
                            fp = []
                            rgi_version = '70C'
                        else:
                            rgi_version = '70G'
                            fp = utils.get_rgi_intersects_entities(rgi_ids,
                                                                   version=rgi_version)

                    else:
                        rgi_version = rgi_ids[0].split('-')[0][-2:]
                        if rgi_version == '60':
                            rgi_version = '62'
                        fp = utils.get_rgi_intersects_entities(rgi_ids,
                                                               version=rgi_version)
                    cfg.set_intersects_db(fp)
                except AttributeError:
                    # RGI V6
                    try:
                        rgi_ids = np.unique(np.sort([entity.RGIId for entity in
                                                     entities]))
                        rgi_version = rgi_ids[0].split('-')[0][-2:]
                        if rgi_version == '60':
                            rgi_version = '62'
                        fp = utils.get_rgi_intersects_entities(rgi_ids,
                                                               version=rgi_version)
                        cfg.set_intersects_db(fp)
                    except AttributeError:
                        # List of str
                        pass

            if _isdir(from_tar):
                gdirs = execute_entity_task(gdir_from_tar, entities,
                                            from_tar=from_tar)
            else:
                gdirs = execute_entity_task(utils.GlacierDirectory, entities,
                                            reset=reset,
                                            from_tar=from_tar,
                                            delete_tar=delete_tar)

    return gdirs


@global_task(log)
def gis_prepro_tasks(gdirs):
    """Run all flowline preprocessing tasks on a list of glaciers.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    """

    task_list = [
        tasks.define_glacier_region,
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


@global_task(log)
def climate_tasks(gdirs, settings_filesuffix='', input_filesuffix=None,
                  overwrite_gdir=False, override_missing=None):
    """Run all climate related entity tasks on a list of glaciers.
    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    input_filesuffix: str
        the filesuffix of the input inversion flowlines which should be used
        (useful for conducting multiple experiments in the same gdir)
    """

    # Process climate data
    execute_entity_task(tasks.process_climate_data, gdirs,
                        settings_filesuffix=settings_filesuffix)
    # mass balance and the apparent mass balance
    execute_entity_task(tasks.mb_calibration_from_hugonnet_mb, gdirs,
                        settings_filesuffix=settings_filesuffix,
                        override_missing=override_missing,
                        overwrite_gdir=overwrite_gdir)
    execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs,
                        settings_filesuffix=settings_filesuffix,
                        input_filesuffix=input_filesuffix,)


@global_task(log)
def inversion_tasks(gdirs, settings_filesuffix='', input_filesuffix=None,
                    output_filesuffix=None,
                    glen_a=None, fs=None, filter_inversion_output=True,
                    add_to_log_file=True):
    """Run all ice thickness inversion tasks on a list of glaciers.

    Quite useful to deal with calving glaciers as well.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments.
    input_filesuffix: str
        The filesuffix of the input inversion flowlines. If None the
        settings_filesuffix will be used.
    output_filesuffix: str
        The filesuffix used for saving resulting inversion files to the gdir. If
        None the settings_filesuffix will be used.
    add_to_log_file : bool
        if the called entity tasks should write into log of gdir. Default True
    """

    if input_filesuffix is None:
        input_filesuffix = settings_filesuffix

    if output_filesuffix is None:
        output_filesuffix = settings_filesuffix

    # We use the settings of the first gdir for defining general parameters
    gdirs[0].settings_filesuffix = settings_filesuffix

    if gdirs[0].settings['use_kcalving_for_inversion']:
        # Differentiate between calving and non-calving glaciers
        gdirs_nc = []
        gdirs_c = []
        for gd in gdirs:
            if gd.is_tidewater:
                gdirs_c.append(gd)
            else:
                gdirs_nc.append(gd)

        log.workflow('Starting inversion tasks for {} tidewater and {} '
                     'non-tidewater glaciers.'.format(len(gdirs_c),
                                                      len(gdirs_nc)))

        if gdirs_nc:
            execute_entity_task(tasks.prepare_for_inversion, gdirs_nc,
                                settings_filesuffix=settings_filesuffix,
                                # only use input_filesuffix for first task as
                                # subsequent task use the results of previous
                                # tasks
                                input_filesuffix=input_filesuffix,
                                output_filesuffix=output_filesuffix,
                                add_to_log_file=add_to_log_file)
            execute_entity_task(tasks.mass_conservation_inversion, gdirs_nc,
                                settings_filesuffix=settings_filesuffix,
                                input_filesuffix=output_filesuffix,
                                output_filesuffix=output_filesuffix,
                                glen_a=glen_a, fs=fs,
                                add_to_log_file=add_to_log_file)
            if filter_inversion_output:
                execute_entity_task(tasks.filter_inversion_output, gdirs_nc,
                                    settings_filesuffix=settings_filesuffix,
                                    input_filesuffix=output_filesuffix,
                                    output_filesuffix=output_filesuffix,
                                    add_to_log_file=add_to_log_file)

        if gdirs_c:
            execute_entity_task(tasks.find_inversion_calving_from_any_mb,
                                gdirs_c,
                                settings_filesuffix=settings_filesuffix,
                                input_filesuffix=output_filesuffix,
                                output_filesuffix=output_filesuffix,
                                glen_a=glen_a, fs=fs,
                                add_to_log_file=add_to_log_file)
    else:
        execute_entity_task(tasks.prepare_for_inversion, gdirs,
                            settings_filesuffix=settings_filesuffix,
                            # only use input_filesuffix for first task as
                            # subsequent task use the results of previous
                            # tasks
                            input_filesuffix=input_filesuffix,
                            output_filesuffix=output_filesuffix,
                            add_to_log_file=add_to_log_file)
        execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                            settings_filesuffix=settings_filesuffix,
                            input_filesuffix=output_filesuffix,
                            output_filesuffix=output_filesuffix,
                            glen_a=glen_a, fs=fs,
                            add_to_log_file=add_to_log_file)
        if filter_inversion_output:
            execute_entity_task(tasks.filter_inversion_output, gdirs,
                                settings_filesuffix=settings_filesuffix,
                                input_filesuffix=output_filesuffix,
                                output_filesuffix=output_filesuffix,
                                add_to_log_file=add_to_log_file)


@global_task(log)
def calibrate_inversion_from_volume(gdirs, settings_filesuffix='',
                                    observations_filesuffix='',
                                    overwrite_observations=False,
                                    ref_volume_m3=None,
                                    ref_volume_year=None,
                                    rgi_ids_in_ref_volume=None,
                                    input_filesuffix=None,
                                    output_filesuffix=None,
                                    fs=0, a_bounds=(0.1, 10),
                                    apply_fs_on_mismatch=False,
                                    error_on_mismatch=True,
                                    filter_inversion_output=True,
                                    add_to_log_file=True):
    """Fit the total volume of the glaciers to the provided estimate.

    This method finds the "best Glen A" to match all glaciers in gdirs with
    a valid inverted volume.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments.
    observations_filesuffix: str
        You can provide a filesuffix for the reference volume to use. If you
        provide ref_volume_m3, then this values will be stored in the
        observations file, if ref_volume_m3 is not already present. If you want
        to force to use the provided values and override the current ones, set
        overwrite_observations to True.
    overwrite_observations : bool
        If you want to overwrite already existing observation values in the
        provided observations file set this to True. Default is False.
    ref_volume_m3 : float
        Option to give an own total glacier volume to match to
    ref_volume_year : int or None
        The year when the reference volume is valid. If None the RGI date is
        used.
    rgi_ids_in_ref_volume : list or None
        If the reference volume is only valid for part of the provided gdirs,
        because some glaciers do not have a volume estimate available. But in
        the end the inversion will be performed on all gdirs. If None all gdirs
        are used. Default is None.
    input_filesuffix: str
        The filesuffix of the input inversion flowlines. If None the
        settings_filesuffix will be used.
    output_filesuffix: str
        The filesuffix used for saving resulting inversion files to the gdir. If
        None the settings_filesuffix will be used.
    fs : float
        invert with sliding (default: no)
    a_bounds: tuple
        factor to apply to default A
    apply_fs_on_mismatch: false
        on mismatch, try to apply an arbitrary value of fs (fs = 5.7e-20 from
        Oerlemans) and try to optimize A again.
    error_on_mismatch: bool
        sometimes the given bounds do not allow to find a zero mismatch:
        this will normally raise an error, but you can switch this off,
        use the closest value instead and move on.
    filter_inversion_output : bool
        whether or not to apply terminus thickness filtering on the inversion
        output (needs the downstream lines to work).
    add_to_log_file : bool
        if the called entity tasks should write into log of gdir. Default True

    Returns
    -------
    a dataframe with the individual glacier volumes
    """

    if input_filesuffix is None:
        input_filesuffix = settings_filesuffix

    if output_filesuffix is None:
        output_filesuffix = settings_filesuffix

    gdirs = utils.tolist(gdirs)

    for gdir in gdirs:
        gdir.observations_filesuffix = observations_filesuffix

    # check if reference volume referes to all gdirs
    if rgi_ids_in_ref_volume is not None:
        gdirs_use = [gdir for gdir in gdirs
                     if gdir.rgi_id in rgi_ids_in_ref_volume]
    else:
        gdirs_use = gdirs

    if any('ref_volume_m3' in gdir.observations for gdir in gdirs):
        ref_volume_m3_file = sum([gdir.observations['ref_volume_m3']['value']
                                  for gdir in gdirs_use])
        if ref_volume_m3 is None:
            # if no reference volume is porvided use the one from the obs-file
            ref_volume_m3 = ref_volume_m3_file
        elif np.isclose(ref_volume_m3_file, ref_volume_m3, rtol=1e-2):
            # ok the provided ref volume is the same as stored in the obs-file
            pass
        elif not overwrite_observations:
            raise InvalidWorkflowError(
                'You have provided an reference volume, but their is already '
                'one stored in the current observations file (filesuffix = '
                f'{observations_filesuffix})! If you want to overwrite set '
                f'overwrite_observations = True.')
        else:
            for gdir in gdirs:
                if 'ref_volume_m3' in gdir.observations:
                    gdir.observations['ref_volume_m3']['value'] = None

    # Optimize the diff to ref, using the settings of the first gdir
    gdirs_use[0].settings_filesuffix = settings_filesuffix
    def_a = gdirs_use[0].settings['inversion_glen_a']

    # create a container to store the individual modelled glacier volumes
    rids = [gdir.rgi_id for gdir in gdirs_use]
    df = pd.DataFrame(index=rids)

    def compute_vol(x):
        inversion_tasks(gdirs_use, settings_filesuffix=settings_filesuffix,
                        input_filesuffix=input_filesuffix,
                        output_filesuffix=output_filesuffix,
                        glen_a=x*def_a, fs=fs,
                        filter_inversion_output=filter_inversion_output,
                        add_to_log_file=add_to_log_file)
        odf = df.copy()
        odf['oggm'] = execute_entity_task(tasks.get_inversion_volume, gdirs_use,
                                          input_filesuffix=output_filesuffix,
                                          add_to_log_file=add_to_log_file)
        return odf

    def to_minimize(x):
        log.workflow('Volume estimate optimisation with '
                     'A factor: {} and fs: {}'.format(x, fs))
        odf = compute_vol(x)
        return ref_volume_m3 - odf.oggm.sum()

    try:
        out_fac, r = optimization.brentq(to_minimize, *a_bounds, rtol=1e-2,
                                         full_output=True)
        if r.converged:
            log.workflow('calibrate_inversion_from_volume '
                         'converged after {} iterations and fs={}. The '
                         'resulting Glen A factor is {}.'
                         ''.format(r.iterations, fs, out_fac))
        else:
            raise ValueError('Unexpected error in optimization.brentq')
    except ValueError:
        # Ok can't find an A. Log for debug:
        odf1 = compute_vol(a_bounds[0]).sum() * 1e-9
        odf2 = compute_vol(a_bounds[1]).sum() * 1e-9
        ref_vol_1 = ref_volume_m3 * 1e-9
        ref_vol_2 = ref_volume_m3 * 1e-9
        msg = ('calibration from volume estimate CAN\'T converge with fs={}.\n'
               'Bound values (km3):\nRef={:.3f} OGGM={:.3f} for A factor {}\n'
               'Ref={:.3f} OGGM={:.3f} for A factor {}'
               ''.format(fs,
                         ref_vol_1, odf1.oggm, a_bounds[0],
                         ref_vol_2, odf2.oggm, a_bounds[1]))
        if apply_fs_on_mismatch and fs == 0 and odf2.oggm > ref_vol_2:
            do_filter = filter_inversion_output
            return calibrate_inversion_from_volume(gdirs,
                                                   settings_filesuffix=settings_filesuffix,
                                                   observations_filesuffix=observations_filesuffix,
                                                   overwrite_observations=overwrite_observations,
                                                   ref_volume_m3=ref_volume_m3,
                                                   rgi_ids_in_ref_volume=rgi_ids_in_ref_volume,
                                                   input_filesuffix=input_filesuffix,
                                                   output_filesuffix=output_filesuffix,
                                                   fs=5.7e-20, a_bounds=a_bounds,
                                                   apply_fs_on_mismatch=False,
                                                   error_on_mismatch=error_on_mismatch,
                                                   filter_inversion_output=do_filter,
                                                   add_to_log_file=add_to_log_file)
        if error_on_mismatch:
            raise ValueError(msg)

        out_fac = a_bounds[int(abs(ref_vol_1 - odf1.oggm) >
                               abs(ref_vol_2 - odf2.oggm))]
        log.workflow(msg)
        log.workflow('We use A factor = {} and fs = {} and move on.'
                     ''.format(out_fac, fs))

    # Compute the final volume with the correct A for all gdirs
    rids = [gdir.rgi_id for gdir in gdirs]
    df = pd.DataFrame(index=rids)
    inversion_tasks(gdirs, settings_filesuffix=settings_filesuffix,
                    input_filesuffix=input_filesuffix,
                    output_filesuffix=output_filesuffix,
                    glen_a=out_fac*def_a, fs=fs,
                    filter_inversion_output=filter_inversion_output,
                    add_to_log_file=add_to_log_file)
    df['vol_oggm_m3'] = execute_entity_task(tasks.get_inversion_volume, gdirs,
                                            input_filesuffix=output_filesuffix,
                                            add_to_log_file=add_to_log_file)
    # add the actually derived volume to the observations file
    for gdir in gdirs:
        vol_single = df.vol_oggm_m3.loc[gdir.rgi_id]
        if ref_volume_year is None:
            year_single = gdir.rgi_date + 1
        else:
            year_single = ref_volume_year
        if 'ref_volume_m3' in gdir.observations:
            current_vol = gdir.observations['ref_volume_m3']
            current_vol['value'] = vol_single
        else:
            current_vol = {'value': vol_single}
        current_vol['year'] = year_single
        gdir.observations['ref_volume_m3'] = current_vol

    return df


@global_task(log)
def calibrate_inversion_from_consensus(gdirs, settings_filesuffix='',
                                       observations_filesuffix='',
                                       overwrite_observations=False,
                                       input_filesuffix=None,
                                       output_filesuffix=None,
                                       ignore_missing=True,
                                       fs=0, a_bounds=(0.1, 10),
                                       apply_fs_on_mismatch=False,
                                       error_on_mismatch=True,
                                       filter_inversion_output=True,
                                       add_to_log_file=True):
    """Fit the total volume of the glaciers to the 2019 consensus estimate.

    This method finds the "best Glen A" to match all glaciers in gdirs with
    a valid inverted volume.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments.
    observations_filesuffix: str
        You can provide a filesuffix where the reference volume used will be
        stored.
    overwrite_observations : bool
        If you want to overwrite already existing observation values in the
        provided observations file set this to True. Default is False.
    input_filesuffix: str
        The filesuffix of the input inversion flowlines. If None the
        settings_filesuffix will be used.
    output_filesuffix: str
        The filesuffix used for saving resulting inversion files to the gdir. If
        None the settings_filesuffix will be used.
    ignore_missing : bool
        set this to true to silence the error if some glaciers could not be
        found in the consensus estimate.
    fs : float
        invert with sliding (default: no)
    a_bounds: tuple
        factor to apply to default A
    apply_fs_on_mismatch: false
        on mismatch, try to apply an arbitrary value of fs (fs = 5.7e-20 from
        Oerlemans) and try to optimize A again.
    error_on_mismatch: bool
        sometimes the given bounds do not allow to find a zero mismatch:
        this will normally raise an error, but you can switch this off,
        use the closest value instead and move on.
    filter_inversion_output : bool
        whether or not to apply terminus thickness filtering on the inversion
        output (needs the downstream lines to work).
    volume_m3_reference : float
        Option to give an own total glacier volume to match to
    add_to_log_file : bool
        if the called entity tasks should write into log of gdir. Default True

    Returns
    -------
    a dataframe with the individual glacier volumes
    """

    if input_filesuffix is None:
        input_filesuffix = settings_filesuffix

    if output_filesuffix is None:
        output_filesuffix = settings_filesuffix

    gdirs = utils.tolist(gdirs)

    # Get the ref data for the glaciers we have
    df = pd.read_hdf(utils.get_demo_file('rgi62_itmix_df.h5'))
    rids = [gdir.rgi_id for gdir in gdirs]

    found_ids = df.index.intersection(rids)
    if not ignore_missing and (len(found_ids) != len(rids)):
        raise InvalidWorkflowError('Could not find matching indices in the '
                                   'consensus estimate for all provided '
                                   'glaciers. Set ignore_missing=True to '
                                   'ignore this error.')

    df = df.reindex(rids)

    # only use those glaciers which have an itmix estimate
    odf = df.copy().dropna()
    ref_volume_m3 = odf.vol_itmix_m3.sum()
    rgi_ids_in_ref_volume = list(odf.index)

    # do the calibration
    df_oggm = calibrate_inversion_from_volume(
        gdirs, settings_filesuffix=settings_filesuffix,
        observations_filesuffix=observations_filesuffix,
        overwrite_observations=overwrite_observations,
        ref_volume_m3=ref_volume_m3,
        rgi_ids_in_ref_volume=rgi_ids_in_ref_volume,
        input_filesuffix=input_filesuffix,
        output_filesuffix=output_filesuffix,
        fs=fs, a_bounds=a_bounds,
        apply_fs_on_mismatch=apply_fs_on_mismatch,
        error_on_mismatch=error_on_mismatch,
        filter_inversion_output=filter_inversion_output,
        add_to_log_file=add_to_log_file)

    # add the oggm volume to the dataframe which will be returned
    df['vol_oggm_m3'] = df_oggm['vol_oggm_m3']

    # add the original estimates to the observations file
    for gdir in gdirs:
        gdir.observations_filesuffix = observations_filesuffix
        ref_volume_sinlge = gdir.observations['ref_volume_m3']
        for col in df.columns:
            if col != 'vol_oggm_m3':
                ref_volume_sinlge[col] = df[col].loc[gdir.rgi_id]
        gdir.observations['ref_volume_m3'] = ref_volume_sinlge
    return df


@global_task(log)
def merge_glacier_tasks(gdirs, settings_filesuffix='',
                        main_rgi_id=None, return_all=False, buffer=None,
                        **kwargs):
    """Shortcut function: run all tasks to merge tributaries to a main glacier

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory`
        all glaciers, main and tributary. Preprocessed and initialised
    main_rgi_id: str
        RGI ID of the main glacier of interest. If None is provided merging
        will start based upon the largest glacier
    return_all : bool
        if main_rgi_id is given and return_all = False: only the main glacier
        is returned
        if main_rgi_is given and return_all = True, the main glacier and every
        remaining glacier from the initial gdirs list is returned, possible
        merged as well.
    buffer : float
        buffer around a flowline to first better find an overlap with another
        flowline. And second assure some distance between the lines at a
        junction. Will default to `cfg.PARAMS['kbuffer']`.
    kwargs: keyword argument for the recursive merging

    Returns
    -------
    merged_gdirs: list of all merged :py:class:`oggm.GlacierDirectory`
    """

    if len(gdirs) > 100:
        raise InvalidParamsError('this could take time! I should include an '
                                 'optional parameter to ignore this.')

    # sort all glaciers descending by area
    gdirs.sort(key=lambda x: x.rgi_area_m2, reverse=True)

    # if main glacier is asked, put it in first position
    if main_rgi_id is not None:
        gdir_main = [gd for gd in gdirs if gd.rgi_id == main_rgi_id][0]
        gdirs.remove(gdir_main)
        gdirs = [gdir_main] + gdirs

    merged_gdirs = []
    while len(gdirs) > 1:
        # main glacier is always the first: either given or the largest one
        gdir_main = gdirs.pop(0)
        gdir_merged, gdirs = _recursive_merging(gdirs, gdir_main, **kwargs)
        merged_gdirs.append(gdir_merged)

    # now we have gdirs which contain all the necessary flowlines,
    # time to clean them up
    for gdir in merged_gdirs:
        flowline.clean_merged_flowlines(
            gdir, settings_filesuffix=settings_filesuffix, buffer=buffer)

    if main_rgi_id is not None and return_all is False:
        return [gd for gd in merged_gdirs if main_rgi_id in gd.rgi_id][0]

    # add the remaining glacier to the final list
    merged_gdirs = merged_gdirs + gdirs

    return merged_gdirs


def _recursive_merging(gdirs, gdir_main, glcdf=None, dem_source=None,
                       filename='climate_historical', input_filesuffix=''):
    """ Recursive function to merge all tributary glaciers.

    This function should start with the largest glacier and then be called
    upon all smaller glaciers.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory`
        all glaciers, main and tributary. Preprocessed and initialised
    gdir_main: :py:class:`oggm.GlacierDirectory`
        the current main glacier where the others are merge to
    glcdf: geopandas.GeoDataFrame
        which contains the main glaciers, will be downloaded if None
    filename: str
        Baseline climate file
    dem_source: str
        the DEM source to use
    input_filesuffix: str
        Filesuffix to the climate file

    Returns
    -------
    merged_gdir: :py:class:`oggm.GlacierDirectory`
        the mergeed current main glacier
    gdirs : list of :py:class:`oggm.GlacierDirectory`
        updated list of glaciers, removed the already merged ones
    """
    # find glaciers which intersect with the main
    tributaries = centerlines.intersect_downstream_lines(gdir_main,
                                                         candidates=gdirs)
    if len(tributaries) == 0:
        # if no tributaries: nothing to do
        return gdir_main, gdirs

    # separate those glaciers which are not already found to be a tributary
    gdirs = [gd for gd in gdirs if gd not in tributaries]

    gdirs_to_merge = []

    for trib in tributaries:
        # for each tributary: check if we can merge additional glaciers to it
        merged, gdirs = _recursive_merging(gdirs, trib, glcdf=glcdf,
                                           filename=filename,
                                           input_filesuffix=input_filesuffix,
                                           dem_source=dem_source)
        gdirs_to_merge.append(merged)

    # create merged glacier directory
    gdir_merged = utils.initialize_merged_gdir(
        gdir_main, tribs=gdirs_to_merge, glcdf=glcdf, filename=filename,
        input_filesuffix=input_filesuffix, dem_source=dem_source)

    flowline.merge_to_one_glacier(gdir_merged, gdirs_to_merge,
                                  filename=filename,
                                  input_filesuffix=input_filesuffix)

    return gdir_merged, gdirs


@global_task(log)
def merge_gridded_data(gdirs, output_folder=None,
                       output_filename='gridded_data_merged',
                       output_grid=None,
                       input_file='gridded_data',
                       input_filesuffix='',
                       included_variables='all',
                       preserve_totals=True,
                       smooth_radius=None,
                       use_glacier_mask=True,
                       add_topography=False,
                       keep_dem_file=False,
                       interp='nearest',
                       use_multiprocessing=True,
                       return_dataset=True,
                       reset=False):
    """ This function takes a list of glacier directories and combines their
    gridded_data into a new NetCDF file and saves it into the output_folder. It
    also could merge data from different source files if you provide a list
    of input_file(s) (together with a list of input_filesuffix and a list of
    included_variables).

    Attention: You always should check the first gdir from gdirs as this
    defines the projection of the resulting dataset and the data which is
    merged, if included_variables is set to 'all'.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        The glacier directories which should be combined. If an additonal
        dimension than x or y is given (e.g. time) we assume it has the same
        length for all gdirs (we currently do not check). The first gdir in the
        list serves as the template for the merged gridded_data (it defines the
        used projection, if you want to merge all variables they are taken from
        the input data of the first gdir).
    output_folder : str
        Folder where the intermediate files and the final combined gridded data
        should be stored. Default is cfg.PATHS['working_dir']
    output_filename : str
        The name for the resulting file. Default is 'gridded_data_merged'.
    output_grid : salem.gis.Grid
        You can provide a custom grid on which the gridded data should be
        merged on. If None, a combined grid of all gdirs will be constructed.
        Default is None.
    input_file : str or list
        The file(s) which should be merged. If a list is provided the data of
        all files is merged into the same dataset. Default is 'gridded_data'.
    input_filesuffix : str or list
        Potential filesuffix for the input file(s). If input_file is a list,
        input_filesuffix should also be a list of the same length.
        Default is ''.
    included_variables : str or list or list of lists
        The variable(s) which should be merged from the input_file(s). For one
        variable it can be provided as str, otherwise as a list. If set to
        'all' we merge everything. If input_file is a list, include_variables
        should be a list of lists with the same length, where the lists define
        the variables for the individual input_files. Furthermore, if you only
        want to merge a subset of the variables you can define the variable as
        a tuple with the first element being the variable name and the second
        element being the selected coordinates as a dictionary (e.g.
        ('variable', {'time': [0, 1, 2]})). Default is 'all'.
    preserve_totals : bool
        If True we preserve the total value of all float-variables of the
        original file. The total value is defined as the sum of all grid cell
        values times the area of the grid cell (e.g. preserving ice volume).
        Default is True.
    smooth_radius : int
        pixel size of the gaussian smoothing, only used if preserve_totals is
        True. Default is to use cfg.PARAMS['smooth_window'] (i.e. a size in
        meters). Set to zero to suppress smoothing.
    use_glacier_mask : bool
        If True only the data cropped by the glacier mask is included in the
        merged file. You must make sure that the variable 'glacier_mask' exists
        in the input_file(s) (which is the oggm default). Default is True.
    add_topography : bool or str
        If True we try to add the default DEM source of the first glacier
        directory of gdirs. Alternatively you could define a DEM source
        directly as string. Default is False.
    keep_dem_file : bool
        If we add a topography to the merged gridded_data we save the DEM as
        a tiff in the output_folder as an intermediate step. If keep_dem_file
        is True we will keep this file, otherwise we delete it at the end.
        Default is False.
    interp : str
        The interpolation method used by salem.Grid.map_gridded_data. Currently
        available 'nearest' (default), 'linear', or 'spline'.
    use_multiprocessing : bool
        If True the merging is done in parallel using multiprocessing. This
        could require a lot of memory. Default is True.
    return_dataset : bool
        If True the merged dataset is returned. Default is True.
    reset : bool
        If the file defined in output_filename already exists and reset is
        False an error is raised. If reset is True and the file exists it is
        deleted before merging. Default is False.
    """

    # check if output_folder exists, otherwise creates it
    if output_folder is None:
        output_folder = cfg.PATHS['working_dir']
    utils.mkdir(output_folder)

    # for some data we want to set zero values outside of outline to nan
    # (e.g. for visualization purposes)
    vars_setting_zero_to_nan = ['distributed_thickness', 'simulated_thickness',
                                'consensus_ice_thickness',
                                'millan_ice_thickness']

    # check if file already exists
    fpath = os.path.join(output_folder, f'{output_filename}.nc')
    if os.path.exists(fpath):
        if reset:
            os.remove(fpath)
        else:
            raise InvalidWorkflowError(f'The file {output_filename}.nc already'
                                       f' exists in the output folder. If you '
                                       f'want to replace it set reset=True!')

    if not isinstance(input_file, list):
        input_file = [input_file]
    if not isinstance(input_filesuffix, list):
        input_filesuffix = [input_filesuffix]
    if not isinstance(included_variables, list):
        # special case if only one variable should be merged
        included_variables = [included_variables]
    if len(input_file) == 1:
        # in the case of one input file we still convert included_variables
        # into a list of lists
        included_variables = [included_variables]

    if output_grid is None:
        # create a combined salem.Grid object, which serves as canvas/boundaries of
        # the combined glacier region
        output_grid = utils.combine_grids(gdirs)

    if add_topography:
        # ok, lets get a DEM and add it to the final file
        if isinstance(add_topography, str):
            dem_source = add_topography
            dem_gdir = None
        else:
            dem_source = None
            dem_gdir = gdirs[0]
        gis.get_dem_for_grid(output_grid, output_folder,
                             source=dem_source, gdir=dem_gdir)
        # unwrapped is needed to execute process_dem without the entity_task
        # overhead (this would need a valid gdir)
        gis.process_dem.unwrapped(gdir=None, grid=output_grid,
                                  fpath=output_folder,
                                  output_filename=output_filename)
        if not keep_dem_file:
            fpath = os.path.join(output_folder, 'dem.tif')
            if os.path.exists(fpath):
                os.remove(fpath)

            # also delete diagnostics
            fpath = os.path.join(output_folder, 'dem_diagnostics.json')
            if os.path.exists(fpath):
                os.remove(fpath)

    with gis.GriddedNcdfFile(grid=output_grid, fpath=output_folder,
                             basename=output_filename) as nc:

        # adding the data of one file after another to the merged dataset
        for in_file, in_filesuffix, included_var in zip(input_file,
                                                        input_filesuffix,
                                                        included_variables):

            # if want to save all variables, take them from the first gdir
            if 'all' in included_var:
                with xr.open_dataset(
                        gdirs[0].get_filepath(in_file,
                                              filesuffix=in_filesuffix)) as ds:
                    included_var = list(ds.data_vars)

            # add one variable after another
            for var in included_var:
                # check if we only want to merge a subset of the variable
                if isinstance(var, tuple):
                    var, slice_of_var = var
                else:
                    slice_of_var = None

                # do not merge topo variables, for this we have add_topography
                if var in ['topo', 'topo_smoothed', 'topo_valid_mask']:
                    continue

                # check dimensions, if other than y or x it is added to file
                with xr.open_dataset(
                        gdirs[0].get_filepath(in_file,
                                              filesuffix=in_filesuffix)) as ds:
                    ds_templ = ds
                dims = ds_templ[var].dims

                dim_lengths = []
                for dim in dims:
                    if dim == 'y':
                        dim_lengths.append(output_grid.ny)
                    elif dim == 'x':
                        dim_lengths.append(output_grid.nx)
                    else:
                        if slice_of_var is not None:
                            # only keep selected part of the variable
                            if dim in slice_of_var:
                                dim_var = ds_templ[var][dim].sel(
                                    {dim: slice_of_var[dim]})
                            else:
                                dim_var = ds_templ[var][dim]
                        else:
                            dim_var = ds_templ[var][dim]
                        if dim not in nc.dimensions:
                            nc.createDimension(dim, len(dim_var))
                            v = nc.createVariable(dim, 'f4', (dim,), zlib=True)
                            # add attributes
                            for attr in dim_var.attrs:
                                setattr(v, attr, dim_var.attrs[attr])
                            if slice_of_var is not None:
                                if dim in slice_of_var:
                                    v[:] = slice_of_var[dim]
                                else:
                                    v[:] = dim_var.values
                            else:
                                v[:] = dim_var.values
                            # also add potential coords (e.g. calender_year)
                            for coord in dim_var.coords:
                                if coord != dim:
                                    if slice_of_var is not None:
                                        if dim in slice_of_var:
                                            coord_val = ds_templ[coord].sel(
                                                {dim: slice_of_var[dim]}).values
                                        else:
                                            coord_val = ds_templ[coord].values
                                    else:
                                        coord_val = ds_templ[coord].values
                                    tmp_coord = nc.createVariable(
                                        coord, 'f4', (dim,))
                                    tmp_coord[:] = coord_val
                                    for attr in ds_templ[coord].attrs:
                                        setattr(tmp_coord, attr,
                                                ds_templ[coord].attrs[attr])
                        dim_lengths.append(len(dim_var))

                # before merging add variable attributes to final file
                v = nc.createVariable(var, 'f4', dims, zlib=True)
                for attr in ds_templ[var].attrs:
                    setattr(v, attr, ds_templ[var].attrs[attr])

                kwargs_reproject = dict(
                    variable=var,
                    target_grid=output_grid,
                    filename=in_file,
                    filesuffix=in_filesuffix,
                    use_glacier_mask=use_glacier_mask,
                    interp=interp,
                    preserve_totals=preserve_totals,
                    smooth_radius=smooth_radius,
                    slice_of_variable=slice_of_var,
                )

                if use_multiprocessing:
                    r_data = execute_entity_task(
                        gis.reproject_gridded_data_variable_to_grid,
                        gdirs,
                        **kwargs_reproject
                    )

                    # if we continue_on_error and their was a file or a variable
                    # missing some entries could be None, here we filter them
                    r_data = list(filter(lambda e: e is not None, r_data))

                    r_data = np.sum(r_data, axis=0)
                    if var in vars_setting_zero_to_nan:
                        r_data = np.where(r_data == 0, np.nan, r_data)

                    v[:] = r_data
                else:
                    # if we do not use multiprocessing we have to loop over the
                    # gdirs and add the data one after another
                    r_data = np.zeros(dim_lengths)
                    for gdir in gdirs:
                        tmp_data = gis.reproject_gridded_data_variable_to_grid(
                            gdir, **kwargs_reproject)
                        if tmp_data is not None:
                            r_data += tmp_data

                    if var in vars_setting_zero_to_nan:
                        r_data = np.where(r_data == 0, np.nan, r_data)

                    v[:] = r_data

        # and some metadata to the merged dataset
        nc.nr_of_merged_glaciers = len(gdirs)
        nc.rgi_ids = [gdir.rgi_id for gdir in gdirs]

    # finally we set potential additional time coordinates correctly again
    fp = os.path.join(output_folder, output_filename + '.nc')
    ds_was_adapted = False

    with xr.open_dataset(fp) as ds:
        for time_var in ['calendar_year', 'calendar_month',
                         'hydro_year', 'hydro_month']:
            if time_var in ds.data_vars:
                ds = ds.set_coords(time_var)
                ds_was_adapted = True
        ds_adapted = ds.load()

    if ds_was_adapted:
        ds_adapted.to_netcdf(fp)

    if return_dataset:
        return ds_adapted
