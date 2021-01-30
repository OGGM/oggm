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
from scipy import optimize as optimization

# Locals
import oggm
from oggm import cfg, tasks, utils
from oggm.core import centerlines, flowline
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

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

    def __call__(self, arg):
        if self.call_func:
            gdir = arg
            call_func = self.call_func
        else:
            call_func, gdir = arg
        if isinstance(gdir, Sequence) and not isinstance(gdir, str):
            gdir, gdir_kwargs = gdir
            gdir_kwargs = _merge_dicts(self.out_kwargs, gdir_kwargs)
            return call_func(gdir, **gdir_kwargs)
        else:
            return call_func(gdir, **self.out_kwargs)


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
    task : function
         the entity task to apply
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    """

    # Should be iterable
    gdirs = utils.tolist(gdirs)

    if len(gdirs) == 0:
        return

    log.workflow('Execute entity task %s on %d glaciers',
                 task.__name__, len(gdirs))

    if task.__dict__.get('global_task', False):
        return task(gdirs, **kwargs)

    pc = _pickle_copier(task, kwargs)

    if _have_ogmpi:
        if ogmpi.OGGM_MPI_COMM is not None:
            return ogmpi.mpi_master_spin_tasks(pc, gdirs)

    if cfg.PARAMS['use_multiprocessing']:
        mppool = init_mp_pool(cfg.CONFIG_MODIFIED)
        out = mppool.map(pc, gdirs, chunksize=1)
    else:
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
    try:
        rid = entity.RGIId
    except AttributeError:
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


def _check_duplicates(rgidf=None):
    """Complain if the input has duplicates."""

    if rgidf is None:
        return
    # Check if dataframe or list of strs
    try:
        rgidf = rgidf.RGIId
    except AttributeError:
        rgidf = utils.tolist(rgidf)
    u, c = np.unique(rgidf, return_counts=True)
    if len(u) < len(rgidf):
        raise InvalidWorkflowError('Found duplicates in the list of '
                                   'RGI IDs: {}'.format(u[c > 1]))


def init_glacier_regions(rgidf=None, *, reset=False, force=False,
                         from_prepro_level=None, prepro_border=None,
                         prepro_rgi_version=None, prepro_base_url=None,
                         from_tar=False, delete_tar=False):
    """DEPRECATED: Initializes the list of Glacier Directories for this run.

    This is the very first task to do (always). If the directories are already
    available in the working directory, use them. If not, create new ones.

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
        get the gdir data from the official pre-processed pool. See the
        documentation for more information
    prepro_border : int
        for `from_prepro_level` only: if you want to override the default
        behavior which is to use `cfg.PARAMS['border']`
    prepro_rgi_version : str
        for `from_prepro_level` only: if you want to override the default
        behavior which is to use `cfg.PARAMS['rgi_version']`
    prepro_base_url : str
        for `from_prepro_level` only: if you want to override the default
        URL from which to download the gdirs. Default currently is
        https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.1/
    from_tar : bool, default=False
        extract the gdir data from a tar file. If set to `True`,
        will check for a tar file at the expected location in `base_dir`.
    delete_tar : bool, default=False
        delete the original tar file after extraction.
    delete_tar : bool, default=False
        delete the original tar file after extraction.

    Returns
    -------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the initialised glacier directories

    Notes
    -----
    This task is deprecated in favor of the more explicit
    init_glacier_directories. Indeed, init_glacier_directories is very
    similar to init_glacier_regions, but it does not process the DEMs:
    a glacier directory is valid also without DEM.
    """

    _check_duplicates(rgidf)

    if reset and not force:
        reset = utils.query_yes_no('Delete all glacier directories?')

    if prepro_border is None:
        prepro_border = int(cfg.PARAMS['border'])

    # if reset delete also the log directory
    if reset:
        fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
        if os.path.exists(fpath):
            shutil.rmtree(fpath)

    gdirs = []
    new_gdirs = []
    if rgidf is None:
        if reset:
            raise ValueError('Cannot use reset without setting rgidf')
        log.workflow('init_glacier_regions by parsing available folders '
                     '(can be slow).')
        # The dirs should be there already
        gl_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')
        for root, _, files in os.walk(gl_dir):
            if files and ('dem.tif' in files):
                gdirs.append(oggm.GlacierDirectory(os.path.basename(root)))
    else:

        # Check if dataframe or list of strs
        try:
            entities = []
            for _, entity in rgidf.iterrows():
                entities.append(entity)
        except AttributeError:
            entities = utils.tolist(rgidf)

        if from_prepro_level is not None:
            log.workflow('init_glacier_regions from prepro level {} on '
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
                    len(cfg.PARAMS['intersects_gdf']) == 0):
                rgi_ids = np.unique(np.sort([entity.RGIId for entity in
                                             entities]))
                rgi_version = rgi_ids[0].split('-')[0][-2:]
                fp = utils.get_rgi_intersects_entities(rgi_ids,
                                                       version=rgi_version)
                cfg.set_intersects_db(fp)

            gdirs = execute_entity_task(utils.GlacierDirectory, entities,
                                        reset=reset,
                                        from_tar=from_tar,
                                        delete_tar=delete_tar)

            for gdir in gdirs:
                if not os.path.exists(gdir.get_filepath('dem')):
                    new_gdirs.append(gdir)

    if len(new_gdirs) > 0:
        # If not initialized, run the task in parallel
        execute_entity_task(tasks.define_glacier_region, new_gdirs)

    return gdirs


def init_glacier_directories(rgidf=None, *, reset=False, force=False,
                             from_prepro_level=None, prepro_border=None,
                             prepro_rgi_version=None, prepro_base_url=None,
                             from_tar=False, delete_tar=False):
    """Initializes the list of Glacier Directories for this run.

    This is the very first task to do (always). If the directories are already
    available in the working directory, use them. If not, create new ones.

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
        get the gdir data from the official pre-processed pool. See the
        documentation for more information
    prepro_border : int
        for `from_prepro_level` only: if you want to override the default
        behavior which is to use `cfg.PARAMS['border']`
    prepro_rgi_version : str
        for `from_prepro_level` only: if you want to override the default
        behavior which is to use `cfg.PARAMS['rgi_version']`
    prepro_base_url : str
        for `from_prepro_level` only: if you want to override the default
        URL from which to download the gdirs. Default currently is
        https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/oggm_v1.1/
    from_tar : bool or str, default=False
        extract the gdir data from a tar file. If set to `True`,
        will check for a tar file at the expected location in `base_dir`.
        delete the original tar file after extraction.

    Returns
    -------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the initialised glacier directories

    Notes
    -----
    This task is very similar to init_glacier_regions, with one main
    difference: it does not process the DEMs for this glacier.
    Eventually, init_glacier_regions will be deprecated and removed from the
    codebase.
    """

    _check_duplicates(rgidf)

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
                    rgi_ids = np.unique(np.sort([entity.RGIId for entity in
                                                 entities]))
                    rgi_version = rgi_ids[0].split('-')[0][-2:]
                    fp = utils.get_rgi_intersects_entities(rgi_ids,
                                                           version=rgi_version)
                    cfg.set_intersects_db(fp)
                except AttributeError:
                    # List of str
                    pass

            if os.path.isdir(from_tar):
                gdirs = execute_entity_task(gdir_from_tar, entities,
                                            from_tar=from_tar)
            else:
                gdirs = execute_entity_task(utils.GlacierDirectory, entities,
                                            reset=reset,
                                            from_tar=from_tar,
                                            delete_tar=delete_tar)

    return gdirs


def gis_prepro_tasks(gdirs):
    """Shortcut function: run all flowline preprocessing tasks.

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


def download_ref_tstars(base_url=None):
    """Downloads and copies the reference list of t* to the working directory.

    Example url:
    https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/RGIV62/CRU/centerlines/qc3/pcp2.5

    Parameters
    ----------
    base_url : str
        url of the params file.
    """
    shutil.copyfile(utils.file_downloader(base_url + '/ref_tstars.csv'),
                    os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv'))
    shutil.copyfile(utils.file_downloader(base_url + '/ref_tstars_params.json'),
                    os.path.join(cfg.PATHS['working_dir'], 'ref_tstars_params.json'))


def climate_tasks(gdirs, base_url=None):
    """Shortcut function: run all climate related tasks.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    base_url : str, optional
        url of the params file.
    """

    # Process climate data
    execute_entity_task(tasks.process_climate_data, gdirs)

    # Then, calibration?
    if cfg.PARAMS['run_mb_calibration']:
        tasks.compute_ref_t_stars(gdirs)
    elif base_url:
        download_ref_tstars(base_url=base_url)

    # Mustar and the apparent mass-balance
    execute_entity_task(tasks.local_t_star, gdirs)
    execute_entity_task(tasks.mu_star_calibration, gdirs)


def inversion_tasks(gdirs, glen_a=None, fs=None, filter_inversion_output=True):
    """Shortcut function: run all ice thickness inversion tasks.

    Quite useful to deal with calving glaciers as well.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    """

    if cfg.PARAMS['use_kcalving_for_inversion']:
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
            execute_entity_task(tasks.prepare_for_inversion, gdirs_nc)
            execute_entity_task(tasks.mass_conservation_inversion, gdirs_nc,
                                glen_a=glen_a, fs=fs)
            if filter_inversion_output:
                execute_entity_task(tasks.filter_inversion_output, gdirs_nc)

        if gdirs_c:
            execute_entity_task(tasks.find_inversion_calving, gdirs_c,
                                glen_a=glen_a, fs=fs)
    else:
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                            glen_a=glen_a, fs=fs)
        if filter_inversion_output:
            execute_entity_task(tasks.filter_inversion_output, gdirs)


def calibrate_inversion_from_consensus(gdirs, ignore_missing=True,
                                       fs=0, a_bounds=(0.1, 10),
                                       apply_fs_on_mismatch=False,
                                       error_on_mismatch=True,
                                       filter_inversion_output=True):
    """Fit the total volume of the glaciers to the 2019 consensus estimate.

    This method finds the "best Glen A" to match all glaciers in gdirs with
    a valid inverted volume.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    ignore_missing : bool
        set this to true to silence the error if some glaciers could not be
        found in the consensus estimate.
    fs : float
        invert with sliding (default: no)
    a_bounds: tuple
        factor to apply to default A
    apply_fs_on_mismatch: false
        on mismatch, try to apply an arbitrary value of fs (fs = 5.7e-20 from
        Oerlemans) and try to otpimize A again.
    error_on_mismatch: bool
        sometimes the given bounds do not allow to find a zero mismatch:
        this will normally raise an error, but you can switch this off,
        use the closest value instead and move on.
    filter_inversion_output : bool
        wether or not to apply terminus thickness filtering on the inversion
        output (needs the downstream lines to work).

    Returns
    -------
    a dataframe with the individual glacier volumes
    """

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

    # Optimize the diff to ref
    def_a = cfg.PARAMS['inversion_glen_a']

    def compute_vol(x):
        inversion_tasks(gdirs, glen_a=x*def_a, fs=fs,
                        filter_inversion_output=filter_inversion_output)
        odf = df.copy()
        odf['oggm'] = execute_entity_task(tasks.get_inversion_volume, gdirs)
        return odf.dropna()

    def to_minimize(x):
        log.workflow('Consensus estimate optimisation with '
                     'A factor: {} and fs: {}'.format(x, fs))
        odf = compute_vol(x)
        return odf.vol_itmix_m3.sum() - odf.oggm.sum()

    try:
        out_fac, r = optimization.brentq(to_minimize, *a_bounds, rtol=1e-2,
                                         full_output=True)
        if r.converged:
            log.workflow('calibrate_inversion_from_consensus '
                         'converged after {} iterations and fs={}. The '
                         'resulting Glen A factor is {}.'
                         ''.format(r.iterations, fs, out_fac))
        else:
            raise ValueError('Unexpected error in optimization.brentq')
    except ValueError:
        # Ok can't find an A. Log for debug:
        odf1 = compute_vol(a_bounds[0]).sum() * 1e-9
        odf2 = compute_vol(a_bounds[1]).sum() * 1e-9
        msg = ('calibration fom consensus estimate CANT converge with fs={}.\n'
               'Bound values (km3):\nRef={:.3f} OGGM={:.3f} for A factor {}\n'
               'Ref={:.3f} OGGM={:.3f} for A factor {}'
               ''.format(fs,
                         odf1.vol_itmix_m3, odf1.oggm, a_bounds[0],
                         odf2.vol_itmix_m3, odf2.oggm, a_bounds[1]))
        if apply_fs_on_mismatch and fs == 0 and odf2.oggm > odf2.vol_itmix_m3:
            return calibrate_inversion_from_consensus(gdirs,
                                                      ignore_missing=ignore_missing,
                                                      fs=5.7e-20, a_bounds=a_bounds,
                                                      apply_fs_on_mismatch=False,
                                                      error_on_mismatch=error_on_mismatch)
        if error_on_mismatch:
            raise ValueError(msg)

        out_fac = a_bounds[int(abs(odf1.vol_itmix_m3 - odf1.oggm) >
                               abs(odf2.vol_itmix_m3 - odf2.oggm))]
        log.workflow(msg)
        log.workflow('We use A factor = {} and fs = {} and move on.'
                     ''.format(out_fac, fs))

    # Compute the final volume with the correct A
    inversion_tasks(gdirs, glen_a=out_fac*def_a, fs=fs,
                    filter_inversion_output=filter_inversion_output)
    df['vol_oggm_m3'] = execute_entity_task(tasks.get_inversion_volume, gdirs)
    return df


def match_regional_geodetic_mb(gdirs, rgi_reg, dataset='hugonnet',
                               period='2000-01-01_2020-01-01'):
    """Regional shift of the mass-balance residual to match observations.

    This is useful for operational runs, but also quite hacky.
    Let's hope we won't need this for too long.

    Parameters
    ----------
    gdirs : the list of gdirs (ideally the entire region)
    rgi_reg : str
       the rgi region to match
    dataset : str
       'hugonnet', or 'zemp'
    period : str
       for 'hugonnet' only. One of
       '2000-01-01_2010-01-01',
       '2010-01-01_2020-01-01',
       '2006-01-01_2019-01-01',
       '2000-01-01_2020-01-01'.
       For 'zemp', the period is always 2006-2016.
    """

    # Get the mass-balance OGGM would give out of the box
    df = utils.compile_fixed_geometry_mass_balance(gdirs, path=False)
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # And also the Area and calving fluxes
    dfs = utils.compile_glacier_statistics(gdirs, path=False)

    if dataset == 'hugonnet':
        y0 = int(period.split('_')[0].split('-')[0])
        y1 = int(period.split('_')[1].split('-')[0]) - 1
    elif dataset == 'zemp':
        y0, y1 = 2006, 2015

    odf = pd.DataFrame(df.loc[y0:y1].mean(), columns=['SMB'])

    odf['AREA'] = dfs.rgi_area_km2 * 1e6
    # Just take the calving rate and change its units
    # Original units: km3 a-1, to change to mm a-1 (units of specific MB)
    rho = cfg.PARAMS['ice_density']
    if 'calving_flux' in dfs:
        odf['CALVING'] = dfs['calving_flux'].fillna(0) * 1e9 * rho / odf['AREA']
    else:
        odf['CALVING'] = 0

    # We have to drop nans here, which occur when calving glaciers fail to run
    odf = odf.dropna()

    # Compare area with total RGI area
    rdf = 'rgi62_areas.csv'
    rdf = pd.read_csv(utils.get_demo_file(rdf), dtype={'O1Region': str})
    ref_area = rdf.loc[rdf['O1Region'] == rgi_reg].iloc[0]['AreaNoC2NoNominal']
    diff = (1 - odf['AREA'].sum() * 1e-6 / ref_area) * 100
    msg = 'Applying geodetic MB correction on RGI reg {}. Diff area: {:.2f}%'
    log.workflow(msg.format(rgi_reg, diff))

    # Total MB OGGM
    out_smb = np.average(odf['SMB'], weights=odf['AREA'])  # for logging
    out_cal = np.average(odf['CALVING'], weights=odf['AREA'])  # for logging
    smb_oggm = np.average(odf['SMB'] - odf['CALVING'], weights=odf['AREA'])

    # Total MB Reference
    if dataset == 'hugonnet':
        df = 'table_hugonnet_regions_10yr_20yr_ar6period.csv'
        df = pd.read_csv(utils.get_demo_file(df))
        df = df.loc[df.period == period].set_index('reg')
        smb_ref = df.loc[int(rgi_reg), 'dmdtda']
    elif dataset == 'zemp':
        df = 'zemp_ref_2006_2016.csv'
        df = pd.read_csv(utils.get_demo_file(df), index_col=0)
        smb_ref = df.loc[int(rgi_reg), 'SMB'] * 1000

    # Diff between the two
    residual = smb_ref - smb_oggm

    # Let's just shift
    log.workflow('Shifting regional MB bias by {}'.format(residual))
    log.workflow('Observations give {}'.format(smb_ref))
    log.workflow('OGGM SMB gives {}'.format(out_smb))
    log.workflow('OGGM frontal ablation gives {}'.format(out_cal))
    for gdir in gdirs:
        try:
            df = gdir.read_json('local_mustar')
            gdir.add_to_diagnostics('mb_bias_before_geodetic_corr', df['bias'])
            df['bias'] = df['bias'] - residual
            gdir.write_json(df, 'local_mustar')
        except FileNotFoundError:
            pass


def merge_glacier_tasks(gdirs, main_rgi_id=None, return_all=False, buffer=None,
                        **kwargs):
    """Shortcut function: run all tasks to merge tributaries to a main glacier

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory`
        all glaciers, main and tributary. Preprocessed and initialised
    main_rgi_id: str
        RGI ID of the main glacier of interest. If None is provided merging
        will start based uppon the largest glacier
    return_all : bool
        if main_rgi_id is given and return_all = False: only the main glaicer
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
        flowline.clean_merged_flowlines(gdir, buffer=buffer)

    if main_rgi_id is not None and return_all is False:
        return [gd for gd in merged_gdirs if main_rgi_id in gd.rgi_id][0]

    # add the remaining glacier to the final list
    merged_gdirs = merged_gdirs + gdirs

    return merged_gdirs


def _recursive_merging(gdirs, gdir_main, glcdf=None,
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

    # seperate those glaciers which are not already found to be a tributary
    gdirs = [gd for gd in gdirs if gd not in tributaries]

    gdirs_to_merge = []

    for trib in tributaries:
        # for each tributary: check if we can merge additional glaciers to it
        merged, gdirs = _recursive_merging(gdirs, trib, glcdf=glcdf,
                                           filename=filename,
                                           input_filesuffix=input_filesuffix)
        gdirs_to_merge.append(merged)

    # create merged glacier directory
    gdir_merged = utils.initialize_merged_gdir(
        gdir_main, tribs=gdirs_to_merge, glcdf=glcdf, filename=filename,
        input_filesuffix=input_filesuffix)

    flowline.merge_to_one_glacier(gdir_merged, gdirs_to_merge,
                                  filename=filename,
                                  input_filesuffix=input_filesuffix)

    return gdir_merged, gdirs
