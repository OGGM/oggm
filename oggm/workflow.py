"""Wrappers for the single tasks, multi processor handling."""
from __future__ import division

# Built ins
import logging
import os
from shutil import rmtree
# External libs
import pandas as pd
import multiprocessing as mp

# Locals
import oggm
from oggm import cfg, tasks, utils
from oggm.utils import download_lock


# Module logger
log = logging.getLogger(__name__)


def _init_pool_globals(_dl_lock, _cfg_contents):
    global download_lock
    download_lock = _dl_lock
    cfg.unpack_config(_cfg_contents)


def _init_pool():
    """Necessary because at import time, cfg might be unitialized"""
    cfg_contents = cfg.pack_config()
    return mp.Pool(cfg.PARAMS['mp_processes'], initializer=_init_pool_globals, initargs=(download_lock, cfg_contents))


class _pickle_copier(object):
    """Pickleable alternative to functools.partial,
    Which is not pickleable in python2 and thus doesn't work
    with Multiprocessing."""

    def __init__(self, func, **kwargs):
        self.call_func = func
        self.out_kwargs = kwargs

    def __call__(self, gdir):
        return self.call_func(gdir, **self.out_kwargs)


def execute_entity_task(task, gdirs, **kwargs):
    """Execute a task on gdirs.

    If you asked for multiprocessing, it will do it.

    Parameters
    ----------
    task: function
        the entity task to apply
    gdirs: list
        the list of oggm.GlacierDirectory to process
    """

    if cfg.PARAMS['use_multiprocessing']:
        mppool = _init_pool()
        mppool.map(_pickle_copier(task, **kwargs), gdirs, chunksize=1)
    else:
        for gdir in gdirs:
            task(gdir, **kwargs)


def init_glacier_regions(rgidf, reset=False, force=False):
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
    for _, entity in rgidf.iterrows():
        gdir = oggm.GlacierDirectory(entity, reset=reset)
        if not os.path.exists(gdir.get_filepath('dem')):
            tasks.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)

    return gdirs


def gis_prepro_tasks(gdirs):
    """Prepare the flowlines."""

    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.compute_downstream_lines,
        tasks.catchment_area,
        tasks.initialize_flowlines,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction
    ]
    for task in task_list:
        execute_entity_task(task, gdirs)


def climate_tasks(gdirs):
    """Prepare the climate data."""

    # Only global tasks
    tasks.distribute_climate_data(gdirs)
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)


def inversion_tasks(gdirs):
    """Invert the bed topography."""

    # Init
    execute_entity_task(tasks.prepare_for_inversion, gdirs)

    # Global task
    tasks.optimize_inversion_params(gdirs)

    # Inversion for all glaciers
    execute_entity_task(tasks.volume_inversion, gdirs)
