"""Command line arguments to the oggm_prepro command

Type `$ oggm_prepro -h` for help

"""

# Standard libraries
import os
import sys
import shutil
import argparse
import time
import logging
import json
import importlib
from pathlib import Path

# External modules
import pandas as pd
import numpy as np
import geopandas as gpd

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow, tasks, GlacierDirectory
from oggm.core import gis
from oggm.core.massbalance import MonthlyTIModel, SfcTypeTIModel
from oggm.exceptions import InvalidParamsError, InvalidDEMError, InvalidWorkflowError

# Module logger
from oggm.utils import get_prepro_base_url, file_downloader

log = logging.getLogger(__name__)

# The preprocessing levels. Besides the usual 0 to 5, two "half levels" allow
# to split L3 and L4 where the work stops being per-glacier and starts needing
# the whole RGI region at once. This is what makes it possible to run the
# expensive part in many small chunk jobs on a cluster:
#
#   2  -> 3a   climate, mass balance calibration, apparent mb   (chunkable)
#   3a -> 3    Glen A calibration, inversion, L3 summaries      (whole region)
#   3  -> 4a   the historical and dynamic spinup runs           (chunkable)
#   4a -> 5    L4 summaries, then the L5 directories            (whole region)
#
# `3a` holds different glacier directory content than `3`, so it is written to
# its own `L3a` folder (a scratch one, it can be deleted afterwards). `4a` on
# the other hand holds exactly the L4 directories - only the summary files are
# missing - so it writes straight into `L4`. The rule is: the folder is named
# after the glacier directory content.
PREPRO_LEVELS = ['0', '1', '2', '3a', '3', '4a', '4', '5']

# The half levels, as (integer level, name of the flag they set)
_HALF_LEVELS = {'3a': 3, '4a': 4}


def _parse_max_level(max_level):
    """Split `max_level` into an int and the two "stop early" flags."""

    max_level = str(max_level)
    if max_level not in PREPRO_LEVELS[1:]:
        raise InvalidParamsError('max_level should be one of {}'
                                 ''.format(PREPRO_LEVELS[1:]))
    # '3a' stops before the inversion, '4a' before the L4 summaries
    if max_level in _HALF_LEVELS:
        return _HALF_LEVELS[max_level], max_level == '3a', max_level == '4a'
    return int(max_level), False, False


def _parse_start_level(start_level):
    """Split `start_level` into an int and the two "resume here" flags.

    The integer is the level the *previous* full level block ends at, so that
    all the existing `start_level <= n` logic keeps working untouched: a run
    resuming at `3a` still has to enter the L3 block, hence start_level 2.
    """

    if start_level is None:
        return 0, False, False

    start_level = str(start_level)
    if start_level not in PREPRO_LEVELS[:-1]:
        raise InvalidParamsError('start_level should be one of {}'
                                 ''.format(PREPRO_LEVELS[:-1]))
    if start_level in _HALF_LEVELS:
        return _HALF_LEVELS[start_level] - 1, start_level == '3a', start_level == '4a'
    return int(start_level), False, False


def _level_dir_name(level):
    """The folder name holding the glacier directories of a level.

    Almost always `L<level>`, but `4a` holds exactly the L4 directories - only
    the summary files are missing - so it shares the `L4` folder instead of
    making a copy of the largest directories of the workflow. `3a` does hold
    something else than `L3` and gets its own folder.
    """
    return 'L4' if str(level) == '4a' else f'L{level}'


def _level_dir(root, rgi_version, border, level):
    """The folder holding the glacier directories of a level.

    Mirrors the layout that :py:func:`oggm.utils.get_prepro_base_url` builds
    for the remote ones.
    """
    return (Path(root) / f'RGI{rgi_version}' / f'b_{border:03d}' /
            _level_dir_name(level))


def _forward_summary_path(basename, level, start_from_dir, start_base_url,
                          rgi_version, border):
    """Where to read a summary file which is only carried forward.

    A run starting from glacier directories it did not make itself still has
    to copy the summary files of the level it started from along. Those can
    sit next to the directories on disk, or on the base url they were
    downloaded from. A chunked run often has both at once: the directories
    are local, because the previous stage wrote them, while the summary files
    are still remote, because no local run ever made them.
    """

    if start_from_dir is not None:
        ipath = (_level_dir(start_from_dir, rgi_version, border, level) /
                 'summary' / basename)
        if ipath.exists():
            return ipath
        if start_base_url is None:
            raise InvalidWorkflowError(
                f'Could not find {ipath}. This run has to carry the L{level} '
                'summary files forward, but they are not next to the glacier '
                'directories it started from - which is normal if a previous '
                'stage made those directories without writing any summary. '
                'Also point start_base_url at the url they came from.')

    return file_downloader(os.path.join(
        get_prepro_base_url(base_url=start_base_url, rgi_version=rgi_version,
                            border=border, prepro_level=int(level)),
        'summary', basename))


def apply_rgi_fixes(rgidf, rgi_version, rgi_reg):
    """The RGI input quality fixes the preprocessing applies before running.

    These are based on visual checks of large glaciers in the RGI. Note that
    for Greenland this also *removes* glaciers, which is why the chunk
    definition has to be computed after this has been applied - see
    :py:func:`oggm.workflow.get_rgi_chunk`.

    Parameters
    ----------
    rgidf : geopandas.GeoDataFrame
        the RGI region file
    rgi_version : str
        the RGI version it comes from
    rgi_reg : str
        the RGI region, zero padded to two digits

    Returns
    -------
    the fixed dataframe
    """

    if rgi_version != '62':
        return rgidf

    ids_to_ice_cap = [
        'RGI60-05.10315',  # huge Greenland ice cap
        'RGI60-03.01466',  # strange thing next to Devon
        'RGI60-09.00918',  # Academy of sciences Ice cap
        'RGI60-09.00969',
        'RGI60-09.00958',
        'RGI60-09.00957',
    ]
    rgidf.loc[rgidf.RGIId.isin(ids_to_ice_cap), 'Form'] = 1

    # In AA almost all large ice bodies are actually ice caps
    if rgi_reg == '19':
        rgidf.loc[rgidf.Area > 100, 'Form'] = 1

    # For greenland we omit connectivity level 2
    if rgi_reg == '05':
        rgidf = rgidf.loc[rgidf['Connect'] != 2]

    return rgidf


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


@utils.entity_task(log)
def _move_hypsometry_to_dem_folder(gdir, source=''):
    """Move the hypsometry file to the DEM source folder if it exists.

    Parameters
    ----------
    gdir : GlacierDirectory
    source : str
        the DEM source
    """

    hypso_f = gdir.get_filepath('hypsometry')
    if not os.path.exists(hypso_f):
        return

    out = os.path.join(gdir.dir, source)
    if not os.path.exists(out):
        raise InvalidWorkflowError('We should not be there')
    os.rename(hypso_f, os.path.join(out, os.path.basename(hypso_f)))


def run_prepro_levels(rgi_version=None, rgi_reg=None, border=None,
                      output_folder='', working_dir='', dem_source='',
                      is_test=False, test_ids=None, rgi_file=None,
                      intersects_file=None, test_topofile=None,
                      disable_mp=False, params_file=None,
                      elev_bands=False, centerlines=False,
                      override_params=None, skip_inversion=False,
                      inversion_volume_dataset='iceboost',
                      mb_model_class='MonthlyTIModel',
                      mb_calibration_strategy='informed_threestep',
                      geodetic_mb_file_path=None,
                      temp_bias_file_path=None,
                      select_source_from_dir=None, keep_dem_folders=False,
                      add_consensus_thickness=False, add_itslive_velocity=False,
                      add_millan_thickness=False, add_millan_velocity=False,
                      add_hugonnet_dhdt=False, add_bedmachine=False,
                      add_glathida=False, add_distributed_thickness=False,
                      add_export_thickness_geotiff=False, compute_hypsometry=False,
                      custom_climate_task=None,
                      custom_climate_task_kwargs=None,
                      start_level=None, start_base_url=None,
                      start_from_dir=None, max_level=5,
                      chunk_idx=None, chunk_size=1000,
                      glen_a_factor=None, inversion_fs=0,
                      logging_level='WORKFLOW',
                      dynamic_spinup=False, ref_mb_err_scaling_factor=0.2,
                      dynamic_spinup_start_year=1979,
                      dynamic_spinup_extra_years_to_try=None,
                      dynamic_spinup_allow_shorter=True,
                      continue_on_error=True, store_fl_diagnostics=False,
                      store_hydro_output=False, store_monthly_hydro=False,
                      ref_area_yr=None, temp_bias_run=False):
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
        ALL is to generate RGITOPO
        "STANDARD" is doina small RGITOPO using COPDEM + NASADEM
        default is the current default lookup tables found at
        https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/rgitopo/2025.4/
    working_dir : str
        path to the OGGM working directory
    params_file : str
        path to the OGGM parameter file (to override defaults)
    is_test : bool
        to test on a couple of glaciers only! Picks 4 glaciers, always the
        same ones (see `test_ids` to choose them): a chunked run needs every
        job to select the same glaciers.
    test_ids : list
        if is_test: list of ids to process
    rgi_file : str or geopandas.GeoDataFrame, optional
        path to an RGI shapefile or a GeoDataFrame to use instead of
        the default RGI region file. Useful to override the default RGI
        files for custom runs as well as for testing.
    intersects_file : str or geopandas.GeoDataFrame, optional
        path to an intersects shapefile or a GeoDataFrame to use instead of
        the default RGI intersects file. Can also be None to skip setting
        the intersects database.
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
    mb_model_class : str
        The mb_model_class to use. Options are 'MonthlyTIModel' (default) and
        'SfcTypeTIModel'.
    mb_calibration_strategy : str
        how to calibrate the massbalance. Currently one of:
        - 'informed_threestep' (default)
        - 'melt_temp'
        - 'temp_melt'
    geodetic_mb_file_path : str
        optional path or URL to a custom geodetic MB file, passed to
        utils.get_geodetic_mb_dataframe and
        tasks.mb_calibration_from_geodetic_mb.
    temp_bias_file_path : str
        path or URL to the temperature-bias prior file, passed to
        tasks.mb_calibration_from_geodetic_mb. Required by the
        'informed_threestep' calibration strategy (and unused otherwise):
        there is no default, the file has to match the setup it is used with
        (climate dataset, RGI version, ...). It is created with a
        `temp_bias_run` and the `oggm_temp_bias` command (see
        utils.get_temp_bias_dataframe).
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
    add_bedmachine : bool
        adds (reprojects) the bedmachine ice thickness maps to the glacier
        directories. With elev_bands=True, the data will also be binned.
    add_glathida : bool
        adds (reprojects) the glathida thickness data to the glacier
        directories. Data points are stored as csv files.
    add_distributed_thickness : bool
        adds a thickness field to gridded_data using
        distribute_thickness_per_altitude.
    add_export_thickness_geotiff : bool
        exports the distributed thickness field to GeoTIFF files in a
        subfolder of the L3 summary directory.
    compute_hypsometry : bool
        Compute the hypsometry tables for all glaciers,
        added to the glacier directory and compiled in
        the summary folder.
    custom_climate_task : str
        optional import path to a custom climate task in the form
        "module_path:function_name". If provided, it will be called instead of
        the default process_climate_data.
    custom_climate_task_kwargs : dict
        optional kwargs passed to the custom climate task when it is executed.
    start_level : str or int
        the pre-processed level to start from (default is to start from
        scratch). If set, you'll need to indicate start_base_url or
        start_from_dir as well. One of 0, 1, 2, '3a', 3, '4a', 4 - see
        `max_level` for what the half levels are.
    start_base_url : str
        the pre-processed base-url to fetch the data from.
    start_from_dir : str
        like `start_base_url`, but for glacier directory tar files which are
        already on disk (a url can only be fetched over http). This is what
        chains the stages of a chunked run together: it points at the folder
        which contains the `RGI{version}/b_{border}/L{level}/` tree, and the
        directories are read from there instead of being downloaded.
        It can be combined with `start_base_url`, and often has to be: the
        directories then come from disk, while the summary files which are
        only carried forward are still fetched from the url. That is the
        normal case when the previous stage made the directories without
        writing any summary file (`max_level` '3a' or '4a').
    max_level : str or int
        the maximum pre-processing level before stopping. Besides 1 to 5,
        two half levels split L3 and L4 where the work stops being
        per-glacier and starts needing the whole RGI region at once:
        - '3a': L3 up to and including the apparent mass balance, i.e. no
          inversion and no summary files. This is the chunkable part of L3.
        - '4a': the L4 runs, without the summary files. This is the chunkable
          part of L4.
        A chunked cluster run is then 2 -> '3a' (chunks), '3a' -> 3 (whole
        region), 3 -> '4a' (chunks), '4a' -> 5 (whole region).
    chunk_idx : int
        process only the glaciers of this chunk (see `chunk_size`). Chunks are
        blocks of the RGI id space, so that several chunk jobs writing into
        the same output folder produce disjoint, complete tar files. Use
        :py:func:`oggm.workflow.count_rgi_chunks` to know how many chunks a
        region has. Default is to process all glaciers.
    chunk_size : int
        the number of glaciers per chunk: 100 or 1000 (default). These are the
        only two allowed, because they are the bundle sizes the glacier
        directory tars are written and read with.
    glen_a_factor : float
        skip the Glen A calibration and invert with this factor instead (and
        with `inversion_fs`). The values of a previous calibration are written
        to `L3/summary/inversion_glen_a_{rgi_reg}.json` by the run which did
        it, so they can be given back here.
    inversion_fs : float
        the sliding parameter to use together with `glen_a_factor`. Ignored if
        `glen_a_factor` is not set.
    skip_inversion : bool
         do not run the inversion (level 3 files). This is a temporary
         workaround for workflows that wont run that far into level 3.
    inversion_volume_dataset : str
        which reference volume dataset to calibrate the ice thickness
        inversion (Glen A) against. One of:
        - 'iceboost' (default): the IceBoost v2 product, auto-selected by RGI
          version. Supported for RGI62, RGI70G and RGI70C.
        - 'consensus': the Farinotti et al. (2019) consensus (ITMIX) estimate.
          Only supported for RGI62.
    logging_level : str
        the logging level to use (DEBUG, INFO, WARNING, WORKFLOW)
    override_params : dict
        a dict of parameters to override.
    dynamic_spinup : str
        include a dynamic spinup matching 'area/dmdtda' OR 'volume/dmdtda' at
        the RGI-date
    ref_mb_err_scaling_factor : float
        scaling factor to reduce individual geodetic mass balance uncertainty
    dynamic_spinup_start_year : int
        if dynamic_spinup is set, define the starting year for the simulation.
        The default is 1979, unless the climate data starts later.
    dynamic_spinup_extra_years_to_try : list or None
        As a last resort, if all other spinup periods failed, you can provide
        here a list of years to try to start the spinup *before*
        dynamic_spinup_start_year (e.g. [10, 20] means the start years
        'dynamic_spinup_start_year - 10' and 'dynamic_spinup_start_year - 20'
        are tried, in this order, so the longest spinup is tried last). Start
        years before the start of the climate data are clipped to it.
        Default is None
    dynamic_spinup_allow_shorter : bool
        If True, and the spinup starting at dynamic_spinup_start_year was not
        successful, shorter spinup periods are tried first (down to the start
        year of the geodetic mass balance period), before the
        dynamic_spinup_extra_years_to_try. If False, the dynamic spinup never
        starts after dynamic_spinup_start_year.
        Default is True
    continue_on_error : bool
        if True the workflow continues if a task raises an error. For operational
        runs it should be set to True (the default).
    store_fl_diagnostics : bool
        if True, also compute and store flowline diagnostics during preprocessing.
        This can increase data usage quite a bit.
    store_hydro_output : bool
        if True, also store the hydrological model output.
    store_monthly_hydro : bool
        if True and store_hydro_output is True, the hydrological model output
        is also stored at monthly resolution (see flowline.run_with_hydro).
        This increases data usage quite a bit, hence the False default.
    ref_area_yr : int
        the hydrological output is computed over a reference area, which
        per default is the largest area covered by the glacier in the simulation
        period. Use this kwarg to force a specific area to the state of the
        glacier at the provided simulation year.
    temp_bias_run : bool
        set to True to run the preprocessing needed to create the temperature
        bias prior file used by the `informed_threestep` calibration. This is
        a preset which forces `max_level=3` and `skip_inversion=True`, and
        skips everything which is of no use for this purpose: the glacier
        directory tar files, the climate statistics and the fixed geometry
        mass balance. `mb_calibration_strategy` has to be set explicitly to
        `temp_melt`, an error is raised otherwise.
        The only output is the L3 `glacier_statistics` file, which is then
        turned into the temperature bias file with the `oggm_temp_bias`
        command (the grouping of climate grid points crosses RGI region
        borders, so this has to be done over all the regions at once).
    """

    # The temp bias preset overrides a couple of options. We log about it
    # further down, once cfg.initialize() has set the logging up.
    if temp_bias_run:
        if mb_calibration_strategy != 'temp_melt':
            raise InvalidParamsError(
                'With `temp_bias_run`, the mass balance calibration strategy '
                'has to be set explicitly to `temp_melt`, not '
                f'`{mb_calibration_strategy}`.')
        max_level = 3
        skip_inversion = True

    # Input check. The levels are strings so that they can carry the two half
    # levels '3a' and '4a', but everything below works on the integer part
    # plus a flag - see `_parse_max_level` / `_parse_start_level`.
    max_level_name = str(max_level)
    max_level, stop_before_inversion, skip_summary = _parse_max_level(max_level)

    start_level_name = '0' if start_level is None else str(start_level)
    start_level, resume_at_inversion, summary_only = \
        _parse_start_level(start_level)

    if mb_calibration_strategy not in ['informed_threestep', 'melt_temp',
                                       'temp_melt']:
        raise InvalidParamsError('mb_calibration_strategy not understood: '
                                 f'{mb_calibration_strategy}')

    if start_level_name != '0':
        if start_base_url is None and start_from_dir is None:
            raise InvalidParamsError('With start_level, please also indicate '
                                     'start_base_url or start_from_dir')
    # We log about this further down, once cfg.initialize() has set the
    # logging up - `log.workflow` does not exist before that
    ignoring_intersects = start_level_name != '0' and intersects_file is not None

    if start_level_name == '0' and start_from_dir is not None:
        raise InvalidParamsError('start_from_dir needs a start_level: with '
                                 'level 0 the glacier directories are built '
                                 'from the RGI file, not read from tars.')

    if stop_before_inversion and skip_inversion:
        # `3a` exists to hand the directories over to the whole-region job
        # which does the inversion, and that job needs the apparent mass
        # balance - which skip_inversion is what skips
        raise InvalidParamsError('max_level `3a` and skip_inversion cannot be '
                                 'combined: `3a` stops right before the '
                                 'inversion so that another run can do it, '
                                 'while skip_inversion drops it altogether.')

    if PREPRO_LEVELS.index(max_level_name) <= \
            PREPRO_LEVELS.index(start_level_name):
        raise InvalidParamsError(
            f'max_level ({max_level_name}) should be above start_level '
            f'({start_level_name}).')

    if chunk_idx is not None:
        # Fail early on a bad chunk_size, rather than after the RGI file has
        # been read (this validates it and returns 0 on the empty list)
        workflow.count_rgi_chunks([], chunk_size=chunk_size)
        if chunk_idx < 0:
            raise InvalidParamsError('chunk_idx should be positive, got '
                                     f'{chunk_idx}')

    # The mass balance is calibrated in L3 only - and not by a run which
    # resumes at the inversion, since that part is already done by then
    if (start_level <= 2 and max_level >= 3 and not resume_at_inversion and
            mb_calibration_strategy == 'informed_threestep' and
            temp_bias_file_path is None):
        raise InvalidParamsError(
            'The `informed_threestep` calibration strategy needs a temperature '
            'bias prior file: set `temp_bias_file_path` to the file matching '
            'your setup. Such a file is created with a `temp_bias_run` and '
            'the `oggm_temp_bias` command.')

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

    # For centerlines we have to change the default evolution model and bed
    if centerlines:
        override_params['downstream_line_shape'] = 'parabola'
        override_params['evolution_model'] = 'FluxBased'

    # define the default melt_f depending on the the used mb_model_class
    if mb_model_class == 'MonthlyTIModel':
        mb_model_class = MonthlyTIModel
        store_mb_diagnostics = False
    elif mb_model_class == 'SfcTypeTIModel':
        mb_model_class = SfcTypeTIModel
        store_mb_diagnostics = True
    else:
        raise NotImplementedError(f"Unknown mb_model: {mb_model_class}")

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

    if ignoring_intersects:
        log.workflow('`intersects_file` is ignored with start_level > 0: '
                     'the intersects are written to the glacier '
                     'directories at L0 and are already in the prepro '
                     'files we start from.')

    if temp_bias_run:
        log.workflow('`temp_bias_run` is set: forcing max_level=3 and '
                     'skip_inversion=True. The only output will be the L3 '
                     'glacier statistics file: no glacier directory tar '
                     'files, no climate statistics, no fixed geometry mass '
                     'balance.')

    # Log the parameters
    msg = '# OGGM Run parameters:'
    for k, v in cfg.PARAMS.items():
        if type(v) in [pd.DataFrame, dict]:
            continue
        msg += '\n    {}: {}'.format(k, v)
    log.workflow(msg)

    if rgi_version is None:
        rgi_version = cfg.PARAMS['rgi_version']
    output_base_dir = Path(output_folder) / f'RGI{rgi_version}' / f'b_{border:03d}'

    # Add a package version file
    utils.mkdir(output_base_dir)
    opath = output_base_dir / 'package_versions.txt'
    with open(opath, 'w') as vfile:
        vfile.write(utils.show_versions(logger=log))

    if rgi_file is None:

        # Get the RGI file
        rgidf = gpd.read_file(utils.get_rgi_region_file(rgi_reg,
                                                        version=rgi_version))
        # We use intersects. They are only needed to build the glacier
        # directories from the RGI (L0): from L1 on, each directory has its
        # own intersects.shp and the region wide file is never read again
        if rgi_version != '70C' and start_level == 0:
            if intersects_file is None:
                rgif = utils.get_rgi_intersects_region_file(rgi_reg,
                                                            version=rgi_version)
            else:
                rgif = intersects_file
            cfg.set_intersects_db(rgif)

        rgidf = apply_rgi_fixes(rgidf, rgi_version, rgi_reg)
    else:
        if isinstance(rgi_file, str):
            rgidf = gpd.read_file(rgi_file)
        else:
            rgidf = rgi_file
        if start_level == 0:
            cfg.set_intersects_db(intersects_file)

    if is_test:
        if test_ids is not None:
            try:
                rgidf = rgidf.loc[rgidf.RGIId.isin(test_ids)]
            except AttributeError:
                # RGI7
                rgidf = rgidf.loc[rgidf.rgi_id.isin(test_ids)]
        else:
            # Seeded for chucked runs
            rgidf = rgidf.sample(4, random_state=0)

    if len(rgidf) == 0:
        raise InvalidParamsError('Zero glaciers selected!')

    # Select our chunk of glaciers, if any. This has to happen after all the
    # filtering above (region 05 connectivity, the RGI62 fixes, ...) so that
    # a given chunk index always means the same glaciers. What comes after
    # (the 70C dates, the DEM source lookup) are per-glacier maps and works
    # just as well on a subset.
    if chunk_idx is not None:
        n_chunks = workflow.count_rgi_chunks(rgidf, chunk_size=chunk_size)
        rgidf = workflow.get_rgi_chunk(rgidf, chunk_idx,
                                       chunk_size=chunk_size)
        log.workflow('Selected chunk {} of {} (chunk size {}): {} glaciers.'
                     ''.format(chunk_idx, n_chunks, chunk_size, len(rgidf)))
        if len(rgidf) == 0:
            # Chunks are blocks of the RGI id space, and the ids have gaps,
            # so an empty chunk is a normal thing. Stop here, but without an
            # error: on a cluster this is one task of an array job, and a
            # failure would take the dependent jobs down with it.
            log.workflow('This chunk is empty, nothing to do.')
            _time_log()
            return

    log.workflow('Starting prepro run for RGI reg: {} '
                 'and border: {}'.format(rgi_reg, border))
    log.workflow('Number of glaciers: {}'.format(len(rgidf)))

    # Try to avoid concurrency
    if rgi_version == '70C':
        from oggm.utils._downloads import get_lock
        with get_lock():
            fp = file_downloader('https://cluster.klima.uni-bremen.de/~oggm/'
                                'ref_mb_params/oggm_v1.6/inv_rgi7/'
                                'rgi7c_rgi_year_2025.1.csv')
            rgi_year_by_id = pd.read_csv(fp, index_col=0)['rgi_year'].astype(int).astype(str)
            rgidf['src_date'] = rgidf['rgi_id'].map(rgi_year_by_id) + '-01-01 00:00:00'

    # Add a new default source
    if not dem_source:
        fs_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/rgitopo/2025.4/'
        if rgi_version == '62':
            with get_lock():
                fs = utils.file_downloader(fs_url + 'chosen_dem_RGI62_20251029.csv')
                dfs = pd.read_csv(fs, index_col=0)
                rgidf['dem_source'] = dfs.loc[rgidf['RGIId'], 'dem_source'].values
        if rgi_version == '70G':
            with get_lock():
                fs = utils.file_downloader(fs_url + 'chosen_dem_RGI70G_20251029.csv')
                dfs = pd.read_csv(fs, index_col=0)
                rgidf['dem_source'] = dfs.loc[rgidf['rgi_id'], 'dem_source'].values
        if rgi_version == '70C':
            with get_lock():
                fs = utils.file_downloader(fs_url + 'chosen_dem_RGI70C_20251029.csv')
                dfs = pd.read_csv(fs, index_col=0)
                rgidf['dem_sourc`e'] = dfs.loc[rgidf['rgi_id'], 'dem_source'].values

    # L0 - go
    if start_level == 0:
        gdirs = workflow.init_glacier_directories(rgidf, reset=True, force=True)

        # Glacier stats
        sum_dir = Path(output_base_dir) / 'L0' / 'summary'
        utils.mkdir(sum_dir)
        opath = sum_dir / f'glacier_statistics_{rgi_reg}.csv'
        utils.compile_glacier_statistics(gdirs, path=opath)

        # L0 OK - compress all in output directory
        if not temp_bias_run:
            log.workflow('L0 done. Writing to tar...')
            level_base_dir = Path(output_base_dir) / 'L0'
            workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                         base_dir=level_base_dir)
            utils.base_dir_to_tar(level_base_dir)
        if max_level == 0:
            _time_log()
            return
    elif start_from_dir is not None:
        # The tar files are already on disk (this is how the stages of a
        # chunked run are chained together)
        from_tar = _level_dir(start_from_dir, rgi_version, border,
                              start_level_name)
        if not from_tar.is_dir():
            raise InvalidParamsError('Could not find the glacier directories '
                                     f'to start from in {from_tar}')
        log.workflow(f'Reading the L{start_level_name} glacier directories '
                     f'from {from_tar}')
        gdirs = workflow.init_glacier_directories(rgidf, reset=True,
                                                  force=True,
                                                  from_tar=str(from_tar))
    else:
        # The level to fetch is the one the directories are *stored* under,
        # which is not the integer we use for the level logic: resuming at
        # '4a' means reading the L4 directories, and start_level is 3 there.
        prepro_level = _level_dir_name(start_level_name)[1:]
        gdirs = workflow.init_glacier_directories(rgidf, reset=True, force=True,
                                                  from_prepro_level=prepro_level,
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
            sum_dir = Path(output_base_dir) / 'L1' / 'summary'
            utils.mkdir(sum_dir)
            opath = sum_dir / f'glacier_statistics_{rgi_reg}.csv'
            utils.compile_glacier_statistics(gdirs, path=opath)

            # Add hypsometry files
            if compute_hypsometry:
                for dem_source in utils.DEM_SOURCES:
                    from oggm.shop.rgitopo import select_dem_from_dir
                    workflow.execute_entity_task(select_dem_from_dir, gdirs,
                                                 dem_source=dem_source,
                                                 keep_dem_folders=True)
                    workflow.execute_entity_task(tasks.rasterio_glacier_mask, gdirs,
                                                 no_nunataks=True,
                                                 overwrite=False)
                    workflow.execute_entity_task(tasks.rasterio_glacier_exterior_mask,
                                                 gdirs,
                                                 overwrite=False)
                    workflow.execute_entity_task(tasks.compute_hypsometry_attributes, gdirs)
                    opath = sum_dir / f'hypsometry_{rgi_reg}_{dem_source}.csv'
                    utils.compile_glacier_hypsometry(gdirs, path=opath,
                                                     add_column=('dem_source', dem_source))
                    workflow.execute_entity_task(_move_hypsometry_to_dem_folder,
                                                 gdirs, source=dem_source)

            # L1 OK - compress all in output directory
            if not temp_bias_run:
                log.workflow('L1 done. Writing to tar...')
                level_base_dir = Path(output_base_dir) / 'L1'
                workflow.execute_entity_task(utils.gdir_to_tar, gdirs,
                                             delete=False,
                                             base_dir=level_base_dir)
                utils.base_dir_to_tar(level_base_dir)

            _time_log()
            return

        # Force a given source
        source = dem_source.upper() if dem_source else None

        # L1 - go
        workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                     source=source)

        # Summaries
        sum_dir = Path(output_base_dir) / 'L1' / 'summary'
        utils.mkdir(sum_dir)

        # Add hypsometry files
        if compute_hypsometry:
            workflow.execute_entity_task(tasks.rasterio_glacier_mask, gdirs)
            workflow.execute_entity_task(tasks.rasterio_glacier_mask, gdirs,
                                         no_nunataks=True)
            workflow.execute_entity_task(tasks.rasterio_glacier_exterior_mask, gdirs)
            workflow.execute_entity_task(tasks.compute_hypsometry_attributes, gdirs)
            opath = sum_dir / f'hypsometry_{rgi_reg}.csv'
            utils.compile_glacier_hypsometry(gdirs, path=opath)

        # Glacier stats
        opath = sum_dir / f'glacier_statistics_{rgi_reg}.csv'
        utils.compile_glacier_statistics(gdirs, path=opath)

        # L1 OK - compress all in output directory
        if not temp_bias_run:
            log.workflow('L1 done. Writing to tar...')
            level_base_dir = Path(output_base_dir) / 'L1'
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
            from oggm.shop.its_live import itslive_velocity_to_gdir
            workflow.execute_entity_task(itslive_velocity_to_gdir, gdirs)
            bin_variables.append('itslive_v')
        if add_millan_thickness:
            from oggm.shop.millan22 import millan_thickness_to_gdir
            workflow.execute_entity_task(millan_thickness_to_gdir, gdirs)
            bin_variables.append('millan_ice_thickness')
        if add_millan_velocity:
            from oggm.shop.millan22 import millan_velocity_to_gdir
            workflow.execute_entity_task(millan_velocity_to_gdir, gdirs)
            bin_variables.append('millan_v')
        if add_hugonnet_dhdt:
            from oggm.shop.hugonnet_maps import hugonnet_to_gdir
            workflow.execute_entity_task(hugonnet_to_gdir, gdirs)
            bin_variables.append('hugonnet_dhdt')
        if add_bedmachine:
            from oggm.shop.bedmachine import bedmachine_to_gdir
            workflow.execute_entity_task(bedmachine_to_gdir, gdirs)
            bin_variables.append('bedmachine_ice_thickness')
        if add_glathida:
            from oggm.shop.glathida import glathida_to_gdir
            workflow.execute_entity_task(glathida_to_gdir, gdirs)
        if rgi_version == '70C':
            # Some additional data for the 70C glaciers
            workflow.execute_entity_task(tasks.rgi7g_to_complex, gdirs)

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
        sum_dir = Path(output_base_dir) / 'L2' / 'summary'
        utils.mkdir(sum_dir)
        opath = sum_dir / f'glacier_statistics_{rgi_reg}.csv'
        utils.compile_glacier_statistics(gdirs, path=opath)

        if add_itslive_velocity:
            from oggm.shop.its_live import compile_itslive_statistics
            opath = sum_dir / f'itslive_statistics_{rgi_reg}.csv'
            compile_itslive_statistics(gdirs, path=opath)
        if add_millan_thickness or add_millan_velocity:
            from oggm.shop.millan22 import compile_millan_statistics
            opath = sum_dir / f'millan_statistics_{rgi_reg}.csv'
            compile_millan_statistics(gdirs, path=opath)
        if add_consensus_thickness:
            from oggm.shop.bedtopo import compile_consensus_statistics
            opath = sum_dir / f'consensus_statistics_{rgi_reg}.csv'
            compile_consensus_statistics(gdirs, path=opath)
        if add_hugonnet_dhdt:
            from oggm.shop.hugonnet_maps import compile_hugonnet_statistics
            opath = sum_dir / f'hugonnet_statistics_{rgi_reg}.csv'
            compile_hugonnet_statistics(gdirs, path=opath)
        if add_bedmachine:
            from oggm.shop.bedmachine import compile_bedmachine_statistics
            opath = sum_dir / f'bedmachine_statistics_{rgi_reg}.csv'
            compile_bedmachine_statistics(gdirs, path=opath)
        if add_glathida:
            from oggm.shop.glathida import compile_glathida_statistics
            opath = sum_dir / f'glathida_statistics_{rgi_reg}.csv'
            compile_glathida_statistics(gdirs, path=opath)

        # And for level 2: shapes
        if len(gdirs_cent) > 0:
            opath = sum_dir / f'centerlines_{rgi_reg}.shp'
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             path=opath)
            opath = sum_dir / f'centerlines_smoothed_{rgi_reg}.shp'
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             ensure_exterior_match=True,
                                             simplify_line_before=0.75,
                                             corner_cutting=3,
                                             path=opath)
            opath = sum_dir / f'flowlines_{rgi_reg}.shp'
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             flowlines_output=True,
                                             path=opath)
            opath = sum_dir / f'geom_widths_{rgi_reg}.shp'
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             geometrical_widths_output=True,
                                             path=opath)
            opath = sum_dir / f'widths_{rgi_reg}.shp'
            utils.write_centerlines_to_shape(gdirs_cent, to_tar=True,
                                             corrected_widths_output=True,
                                             path=opath)

        # L2 OK - compress all in output directory
        if not temp_bias_run:
            log.workflow('L2 done. Writing to tar...')
            level_base_dir = Path(output_base_dir) / 'L2'
            workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                         base_dir=level_base_dir)
            utils.base_dir_to_tar(level_base_dir)
        if max_level == 2:
            _time_log()
            return

    # L3 - Tasks
    if start_level <= 2:
        sum_dir = Path(output_base_dir) / 'L3' / 'summary'

        # Everything down to the apparent mass balance is per-glacier, so it
        # is the part of L3 which can be run in chunks. A run resuming at
        # `3a` has it done already and goes straight to the inversion.
        if resume_at_inversion:
            log.workflow('Resuming L3 at the inversion: the climate and the '
                         'mass balance calibration are read from the glacier '
                         'directories we started from.')
        else:
            # Climate
            climate_kwargs = custom_climate_task_kwargs or {}
            if custom_climate_task:
                try:
                    mod_path, func_name = custom_climate_task.rsplit(':', 1)
                except ValueError:
                    raise InvalidParamsError('custom_climate_task must be of the form "module:function"')
                try:
                    mod = importlib.import_module(mod_path)
                except ModuleNotFoundError as err:
                    raise InvalidParamsError(f'Cannot import module {mod_path}') from err
                try:
                    custom_task_func = getattr(mod, func_name)
                except AttributeError as err:
                    raise InvalidParamsError(f'Module {mod_path} has no attribute {func_name}') from err
                workflow.execute_entity_task(custom_task_func, gdirs, **climate_kwargs)
            else:
                workflow.execute_entity_task(tasks.process_climate_data, gdirs)

            if mb_calibration_strategy == 'informed_threestep':
                workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                             gdirs,
                                             informed_threestep=True,
                                             mb_model_class=mb_model_class,
                                             file_path=geodetic_mb_file_path,
                                             temp_bias_file_path=temp_bias_file_path)
            elif mb_calibration_strategy == 'melt_temp':
                workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                             gdirs,
                                             calibrate_param1='melt_f',
                                             calibrate_param2='temp_bias',
                                             mb_model_class=mb_model_class,
                                             file_path=geodetic_mb_file_path)
            elif mb_calibration_strategy == 'temp_melt':
                workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                             gdirs,
                                             calibrate_param1='temp_bias',
                                             calibrate_param2='melt_f',
                                             mb_model_class=mb_model_class,
                                             file_path=geodetic_mb_file_path)
            else:
                raise InvalidParamsError('mb_calibration_strategy not understood: '
                                         f'{mb_calibration_strategy}')

            if not skip_inversion:
                workflow.execute_entity_task(tasks.apparent_mb_from_any_mb,
                                             gdirs,
                                             mb_model_class=mb_model_class,)

        if stop_before_inversion:
            # End of the chunkable part of L3. Write the directories out so
            # that a whole-region job can calibrate Glen A and finish the
            # level: that calibration needs all the glaciers at once.
            log.workflow('L3a done (no inversion, no summary). '
                         'Writing to tar...')
            level_base_dir = Path(output_base_dir) / 'L3a'
            workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                         base_dir=level_base_dir)
            utils.base_dir_to_tar(level_base_dir)
            _time_log()
            return

        utils.mkdir(sum_dir)

        if not skip_inversion:

            filter = border >= 20

            # Inversion: calibrate Glen A to a reference volume dataset.
            # Be explicit about which dataset is used for which RGI version.
            if inversion_volume_dataset not in ('iceboost', 'consensus'):
                raise InvalidParamsError(
                    "inversion_volume_dataset must be 'iceboost' or "
                    f"'consensus', not '{inversion_volume_dataset}'.")

            if rgi_version in ('70G', '70C') and \
                    inversion_volume_dataset != 'iceboost':
                raise InvalidParamsError(
                    f"For {rgi_version} only inversion_volume_dataset='iceboost' "
                    f"is supported, not '{inversion_volume_dataset}' (the "
                    "consensus estimate is only available for RGI62).")

            # 'iceboost'/'consensus' map directly to ref_table presets
            inv_df = workflow.calibrate_inversion_from_ref_table(
                gdirs,
                ref_table=inversion_volume_dataset,
                glen_a_factor=glen_a_factor,
                fs=inversion_fs,
                apply_fs_on_mismatch=True,
                error_on_mismatch=False,
                filter_inversion_output=filter)

            # Write down which Glen A this region ended up with, so that a
            # later run can reproduce it with `glen_a_factor` instead of
            # calibrating again (the calibration needs the whole region)
            opath = sum_dir / f'inversion_glen_a_{rgi_reg}.json'
            with open(opath, 'w') as f:
                json.dump({'rgi_version': rgi_version,
                           'rgi_reg': rgi_reg,
                           'border': border,
                           'ref_table': inversion_volume_dataset,
                           'n_glaciers': len(gdirs),
                           'glen_a_factor': float(inv_df.attrs['glen_a_factor']),
                           'glen_a': float(inv_df.attrs['glen_a']),
                           'fs': float(inv_df.attrs['fs']),
                           }, f, indent=2)

            # Distribute thickness per altitude for gridded data
            if add_distributed_thickness:
                workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs)

            # We get ready for modelling
            if border >= 20:
                workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)
            else:
                log.workflow('L3: for map border values < 20, wont initialize glaciers '
                             'for the run.')
        # Glacier stats
        opath = sum_dir / f'glacier_statistics_{rgi_reg}.csv'
        utils.compile_glacier_statistics(gdirs, path=opath)

        if temp_bias_run:
            # The glacier statistics is all we need: the temperature bias file
            # itself is made by the `oggm_temp_bias` command, out of the
            # statistics of all the RGI regions at once.
            log.workflow('`temp_bias_run` is done. Now run the '
                         '`oggm_temp_bias` command on the L3 summary folder '
                         'of all the regions to create the temperature bias '
                         'file.')
            _time_log()
            return

        # Export thickness to GeoTIFF if requested
        if add_export_thickness_geotiff and add_distributed_thickness:
            thickness_dir = sum_dir / 'distributed_thickness'
            utils.mkdir(thickness_dir)
            workflow.execute_entity_task(tasks.gridded_data_var_to_geotiff, gdirs,
                                         varname='distributed_thickness',
                                         output_folder=thickness_dir)
        opath = sum_dir / f'climate_statistics_{rgi_reg}.csv'
        utils.compile_climate_statistics(gdirs, path=opath)
        opath = sum_dir / f'fixed_geometry_mass_balance_{rgi_reg}.csv'
        utils.compile_fixed_geometry_mass_balance(gdirs, path=opath,
                                                  mb_model_class=mb_model_class)

        # L3 OK - compress all in output directory
        log.workflow('L3 done. Writing to tar...')
        level_base_dir = Path(output_base_dir) / 'L3'
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
        sum_dir = Path(output_base_dir) / 'L4' / 'summary'

        # The summary files are written by the whole-region part of L4, so a
        # chunked run stopping at `4a` has nothing to do with them
        if not skip_summary:
            utils.mkdir(sum_dir)

            # Copy L3 files for consistency
            for bn in ['glacier_statistics', 'climate_statistics',
                       'fixed_geometry_mass_balance']:
                if start_level <= 2:
                    ipath = sum_dir_L3 / f'{bn}_{rgi_reg}.csv'
                else:
                    ipath = _forward_summary_path(
                        f'{bn}_{rgi_reg}.csv', start_level,
                        start_from_dir, start_base_url, rgi_version, border)

                opath = sum_dir / f'{bn}_{rgi_reg}.csv'
                shutil.copyfile(ipath, opath)

        # The runs are per-glacier: this is the part of L4 which can be run
        # in chunks. A run resuming at `4a` has them done already and only
        # needs to compile the summary files, which needs the whole region.
        if summary_only:
            log.workflow('Resuming L4 at the summary files: the model runs '
                         'are read from the glacier directories we started '
                         'from.')
        else:
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

            # here we define the actual start date of the model outputs
            if y0 > dynamic_spinup_start_year:
                dynamic_spinup_start_year = y0

            # conduct historical run before dynamic melt_f calibration
            # (for comparison to old default behavior)
            kwargs_run_from_climate_data = {
                'min_ys': y0, 'ye': ye, 'mb_model_class': mb_model_class,
                'save_mb_diagnostics_filesuffix': '_historical' if store_mb_diagnostics else None,
                'output_filesuffix': '_historical',
                'fixed_geometry_spinup_yr': dynamic_spinup_start_year,
            }
            if not store_hydro_output:
                workflow.execute_entity_task(
                    tasks.run_from_climate_data, gdirs,
                    **kwargs_run_from_climate_data
                )
            else:
                workflow.execute_entity_task(
                    tasks.run_with_hydro, gdirs,
                    run_task=tasks.run_from_climate_data,
                    store_monthly_hydro=store_monthly_hydro,
                    ref_area_yr=ref_area_yr,
                    **kwargs_run_from_climate_data
                )

        if not skip_summary:
            # Now compile the output
            opath = Path(sum_dir) / f'historical_run_output_{rgi_reg}.nc'
            utils.compile_run_output(gdirs, path=opath,
                                     input_filesuffix='_historical')

        # conduct dynamic spinup if wanted
        if dynamic_spinup:

            if not summary_only:
                minimise_for = dynamic_spinup.split('/')[0]

                melt_f_max = cfg.PARAMS['melt_f_max']
                kwargs_run_dynamic_melt_f_calibration = {
                    'ref_mb_err_scaling_factor': ref_mb_err_scaling_factor,
                    'ys': dynamic_spinup_start_year, 'ye': ye,
                    'melt_f_max': melt_f_max,
                    'mb_model_class': mb_model_class,
                    'kwargs_run_function': {
                        'minimise_for': minimise_for,
                        'spinup_extra_years_to_try':
                            dynamic_spinup_extra_years_to_try,
                        'allow_shorter_spinup': dynamic_spinup_allow_shorter,
                    },
                    'ignore_errors': True,
                    'kwargs_fallback_function': {
                        'minimise_for': minimise_for,
                        'spinup_extra_years_to_try':
                            dynamic_spinup_extra_years_to_try,
                        'allow_shorter_spinup': dynamic_spinup_allow_shorter,
                    },
                    'save_mb_diagnostics_filesuffix': ('_spinup_historical'
                                                       if store_mb_diagnostics else None),
                    'output_filesuffix': '_spinup_historical',
                }

                if not store_hydro_output:
                    workflow.execute_entity_task(
                        tasks.run_dynamic_melt_f_calibration, gdirs,
                        **kwargs_run_dynamic_melt_f_calibration
                        )
                else:
                    workflow.execute_entity_task(
                        tasks.run_with_hydro, gdirs,
                        run_task=tasks.run_dynamic_melt_f_calibration,
                        store_monthly_hydro=store_monthly_hydro,
                        ref_area_yr=ref_area_yr,
                        **kwargs_run_dynamic_melt_f_calibration
                    )

            if not skip_summary:
                # Now compile the output
                opath = sum_dir / f'spinup_historical_run_output_{rgi_reg}.nc'
                utils.compile_run_output(gdirs, path=opath,
                                         input_filesuffix='_spinup_historical')

        if not skip_summary:
            # Glacier statistics we recompute here for error analysis
            opath = sum_dir / f'glacier_statistics_{rgi_reg}.csv'
            utils.compile_glacier_statistics(gdirs, path=opath)

            # Add the extended files
            pf = sum_dir / f'historical_run_output_{rgi_reg}.nc'
            # We have copied the files above
            mf = sum_dir / f'fixed_geometry_mass_balance_{rgi_reg}.csv'
            sf = sum_dir / f'glacier_statistics_{rgi_reg}.csv'
            opath = sum_dir / f'historical_run_output_extended_{rgi_reg}.nc'
            utils.extend_past_climate_run(past_run_file=pf,
                                          fixed_geometry_mb_file=mf,
                                          glacier_statistics_file=sf,
                                          path=opath)

        # L4 OK - compress all in output directory. A run which only added
        # the summary files did not touch the directories, and the `4a` stage
        # has already written them to the very same folder.
        if not summary_only:
            log.workflow('L4 done. Writing to tar...')
            level_base_dir = Path(output_base_dir) / 'L4'
            workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                         base_dir=level_base_dir)
            utils.base_dir_to_tar(level_base_dir)

        sum_dir_L4 = sum_dir

        if max_level == 4:
            _time_log()
            return

    # L5 - No tasks: make the dirs small
    sum_dir = Path(output_base_dir) / 'L5' / 'summary'
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
            ipath = sum_dir_L4 / f'{bn}_{rgi_reg}.{suffix}'
        else:
            ipath = _forward_summary_path(
                f'{bn}_{rgi_reg}.{suffix}', start_level,
                start_from_dir, start_base_url, rgi_version, border)
        opath = sum_dir / f'{bn}_{rgi_reg}.{suffix}'
        shutil.copyfile(ipath, opath)

    # Copy mini data to new dir
    mini_base_dir = (
        Path(working_dir)
        / 'mini_perglacier'
        / f'RGI{rgi_version}'
        / f'b_{border:03d}'
    )
    mini_gdirs = workflow.execute_entity_task(tasks.copy_to_basedir, gdirs,
                                              base_dir=mini_base_dir,
                                              setup='run/spinup')

    # L5 OK - compress all in output directory
    log.workflow('L5 done. Writing to tar...')
    level_base_dir = Path(output_base_dir) / 'L5'
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
    parser.add_argument('--start-level', type=str, default='0',
                        choices=PREPRO_LEVELS[:-1],
                        help='the pre-processed level to start from (default '
                             'is to start from 0). If set, you will need to '
                             'indicate --start-base-url or --start-from-dir '
                             'as well. See --max-level for the half levels.')
    parser.add_argument('--start-base-url', type=str,
                        help='the pre-processed base-url to fetch the data '
                             'from when starting from level > 0.')
    parser.add_argument('--start-from-dir', type=str,
                        help='like --start-base-url, but for glacier '
                             'directory tar files which are already on disk. '
                             'This is what chains the stages of a chunked '
                             'run together. Point it at the folder holding '
                             'the RGI{version}/b_{border}/L{level}/ tree. Can '
                             'be combined with --start-base-url, which is '
                             'then used for the summary files which are only '
                             'carried forward.')
    parser.add_argument('--max-level', type=str, default='5',
                        choices=PREPRO_LEVELS[1:],
                        help='the maximum level you want to run the '
                             'pre-processing for. Besides 1 to 5, the two '
                             'half levels 3a and 4a stop where the work stops '
                             'being per-glacier and starts needing the whole '
                             'RGI region: 3a is L3 without the inversion and '
                             'the summary files, 4a is L4 without the summary '
                             'files. A chunked cluster run is 2 -> 3a '
                             '(chunks), 3a -> 3 (region), 3 -> 4a (chunks), '
                             '4a -> 5 (region).')
    parser.add_argument('--chunk-idx', type=int, default=None,
                        help='process only the glaciers of this chunk. Chunks '
                             'are blocks of the RGI id space, so that several '
                             'chunk jobs writing into the same output folder '
                             'produce disjoint, complete tar files. Meant to '
                             'be set to $SLURM_ARRAY_TASK_ID. Use the '
                             'oggm_prepro_chunks command to know how many '
                             'chunks a region has.')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        choices=[100, 1000],
                        help='the number of glaciers per chunk (default '
                             '1000). Only 100 and 1000 are allowed: they are '
                             'the bundle sizes the glacier directory tars are '
                             'written and read with.')
    parser.add_argument('--inversion-glen-a-factor', type=float, default=None,
                        help='skip the Glen A calibration and invert with '
                             'this factor instead. The value of a previous '
                             'calibration is written to the L3 summary folder '
                             'as inversion_glen_a_{rgi_reg}.json.')
    parser.add_argument('--inversion-fs', type=float, default=0,
                        help='the sliding parameter for the inversion. Mostly '
                             'useful together with --inversion-glen-a-factor, '
                             'to reproduce a calibration which needed it.')
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
    parser.add_argument('--elev-bands', action='store_true',
                        help='compute the flowlines based on the Huss & Farinotti '
                             '2012 method.')
    parser.add_argument('--centerlines', action='store_true',
                        help='compute the flowlines based on the OGGM '
                             'centerline(s) method.')
    parser.add_argument('--skip-inversion', action='store_true',
                        help='do not run the inversion (level 3 files). '
                             'this is a temporary workaround for workflows '
                             'that wont run that far into level 3.')
    parser.add_argument('--mb-model-class', type=str, default='MonthlyTIModel',
                        help='the mass balance model class to use. Options are '
                             'MonthlyTIModel (default) or SfcTypeTIModel.')
    parser.add_argument('--inversion-volume-dataset', type=str,
                        default='iceboost',
                        choices=['iceboost', 'consensus'],
                        help="reference volume dataset to calibrate the ice "
                             "thickness inversion against. 'iceboost' (default, "
                             "IceBoost v2, RGI62/RGI70G/RGI70C) or 'consensus' "
                             "(Farinotti et al. 2019, RGI62 only).")
    parser.add_argument('--mb-calibration-strategy', type=str,
                        default='informed_threestep',
                        choices=['informed_threestep', 'melt_temp',
                                 'temp_melt'],
                        help='how to calibrate the massbalance. Currently one '
                             'of informed_threestep (default), melt_temp '
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
    parser.add_argument('--keep-dem-folders', action='store_true',
                        help='if `select_source_from_dir` is used, wether to keep '
                        'the original DEM folders in or not.')
    parser.add_argument('--add-consensus-thickness', action='store_true',
                        help='adds (reprojects) the consensus thickness '
                             'estimates to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-itslive-velocity', action='store_true',
                        help='adds (reprojects) the ITS_LIVE velocity '
                             'estimates to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-millan-thickness', action='store_true',
                        help='adds (reprojects) the millan thickness '
                             'estimates to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-millan-velocity', action='store_true',
                        help='adds (reprojects) the millan velocity '
                             'estimates to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-hugonnet-dhdt', action='store_true',
                        help='adds (reprojects) the hugonnet dhdt '
                             'maps to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-bedmachine', action='store_true',
                        help='adds (reprojects) the Bedmachine ice thickness '
                             'maps to the glacier directories. '
                             'With --elev-bands, the data will also be '
                             'binned.')
    parser.add_argument('--add-glathida', action='store_true',
                        help='adds (reprojects) the glathida point thickness '
                             'observations to the glacier directories. '
                             'The data points are stored as csv.')
    parser.add_argument('--custom-climate-task', type=str, default=None,
                        help='Custom climate task import path in the form module:function. '
                            'If provided, it replaces the default process_climate_data.')
    parser.add_argument('--custom-climate-task-kwargs', type=json.loads, default=None,
                        help='JSON dict of kwargs passed to the custom climate task.')
    parser.add_argument('--add-distributed-thickness', action='store_true',
                        help='adds a thickness field to gridded_data using '
                             'distribute_thickness_per_altitude.')
    parser.add_argument('--add-export-thickness-geotiff', action='store_true',
                        help='exports the distributed thickness field to '
                             'GeoTIFF files in a subfolder of the L3 summary '
                             'directory. Requires --add-distributed-thickness.')
    parser.add_argument('--compute-hypsometry', action='store_true',
                        help='Compute the hypsometry tables for all glaciers, '
                             'added to the glacier directory and compiled in '
                             'the summary folder')
    parser.add_argument('--test', action='store_true',
                        help='if you want to do a test on a couple of '
                             'glaciers first.')
    parser.add_argument('--test-ids', nargs='+',
                        help='if --test, specify the RGI ids to run separated '
                             'by a space (default: 4 randomly selected).')
    parser.add_argument('--rgi-file', type=str, default=None,
                        help='path to an RGI shapefile to use instead of '
                             'the default RGI region file.')
    parser.add_argument('--intersects-file', type=str, default=None,
                        help='path to an intersects shapefile to use instead '
                             'of the default RGI intersects file.')
    parser.add_argument('--disable-mp', action='store_true',
                        help='if you want to disable multiprocessing.')
    parser.add_argument('--dynamic-spinup', type=str, default='',
                        help="include a dynamic spinup for matching glacier area "
                             "('area/dmdtda') OR volume ('volume/dmdtda') at "
                             "the RGI-date, AND mass-change from Hugonnet "
                             "in the period 2000-2020 (dynamic melt_f "
                             "calibration).")
    parser.add_argument('--ref-mb-err-scaling-factor', type=float, default=0.2,
                        help="scaling factor to account for correlated "
                             "uncertainties of geodetic mass balance "
                             "observations when looking at regional scale. "
                             "Should be smaller or equal to 1.")
    parser.add_argument('--dynamic-spinup-start-year', type=int, default=1979,
                        help="if --dynamic-spinup is set, define the starting"
                             "year for the simulation. The default is 1979, "
                             "unless the climate data starts later.")
    parser.add_argument('--dynamic-spinup-extra-years-to-try', nargs='*',
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help="if --dynamic-spinup is set, define additional "
                             "years to start the spinup BEFORE "
                             "--dynamic-spinup-start-year, tried as a last "
                             "resort if all other spinup periods failed (e.g. "
                             "'10 20' first tries to start 10 years before "
                             "--dynamic-spinup-start-year, and then 20 years "
                             "before, so the longest spinup is tried last). "
                             "Start years before the start of the climate data "
                             "are clipped to it. If you do not want to use it "
                             "set '--dynamic-spinup-extra-years-to-try none' "
                             "in the terminal.")
    parser.add_argument('--dynamic-spinup-no-shorter-periods',
                        action='store_true',
                        help="if --dynamic-spinup is set, prevent the dynamic "
                             "spinup from starting AFTER "
                             "--dynamic-spinup-start-year. Per default, if the "
                             "spinup at --dynamic-spinup-start-year failed, "
                             "shorter spinup periods are tried first (down to "
                             "the start year of the geodetic mass balance "
                             "period).")
    parser.add_argument('--geodetic-mb-file-path', type=str, default=None,
                        help='optional path or URL to a custom geodetic MB '
                             'file passed to MB calibration.')
    parser.add_argument('--temp-bias-file-path', type=str, default=None,
                        help='path or URL to the temperature-bias prior file '
                             'passed to MB calibration. Required by the '
                             'informed_threestep strategy (and unused '
                             'otherwise): there is no default, the file has to '
                             'match the setup it is used with. It is created '
                             'with --temp-bias-run and the `oggm_temp_bias` '
                             'command.')
    parser.add_argument('--temp-bias-run', action='store_true',
                        help='run the preprocessing needed to create the '
                             'temperature bias prior file. This forces '
                             '--max-level 3 and --skip-inversion, and writes '
                             'nothing but the L3 glacier statistics file. '
                             'Requires --mb-calibration-strategy temp_melt. '
                             'Feed the result to the `oggm_temp_bias` command '
                             '(together with the other regions) to create the '
                             'file.')
    parser.add_argument('--store-fl-diagnostics', action='store_true',
                        help="Also compute and store flowline diagnostics during "
                             "preprocessing. This can increase data usage quite "
                             "a bit.")
    parser.add_argument('--store-hydro-output', action='store_true',
                        help='Add optional hydrological model output')
    parser.add_argument('--store-monthly-hydro', action='store_true',
                        help='Requires --store-hydro-output. Also store the '
                             'hydrological model output at monthly resolution. '
                             'This increases data usage quite a bit.')
    parser.add_argument('--ref-area-yr', type=int, default=None,
                        help='Force the reference area used for the hydrological '
                             'output to the glacier state of the given simulation '
                             'year, instead of the largest area during the '
                             'simulation period.')
    parser.add_argument('--override-params', type=json.loads, default=None)

    args = parser.parse_args(args)

    # Check input
    rgi_reg = args.rgi_reg
    if not rgi_reg:
        rgi_reg = os.environ.get('OGGM_RGI_REG', None)
        if rgi_reg is None:
            raise InvalidParamsError('--rgi-reg is required!')
    rgi_reg = '{:02}'.format(int(rgi_reg))
    ok_regs = ['{:02}'.format(int(r)) for r in range(1, 20)]
    if rgi_reg not in ok_regs:
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

    extra_years_to_try = args.dynamic_spinup_extra_years_to_try
    if extra_years_to_try in [['none'], []]:
        extra_years_to_try = None
    else:
        # argparse gives us strings if the user provided them in the terminal
        try:
            extra_years_to_try = [int(yr) for yr in extra_years_to_try]
        except (TypeError, ValueError):
            raise InvalidParamsError(
                '--dynamic-spinup-extra-years-to-try takes years to start the '
                'spinup before --dynamic-spinup-start-year, or the single '
                f'value "none", but got {extra_years_to_try}!')
        if any(yr <= 0 for yr in extra_years_to_try):
            raise InvalidParamsError(
                '--dynamic-spinup-extra-years-to-try must be positive (they '
                'are counted backwards from --dynamic-spinup-start-year)!')

    # All good
    return dict(rgi_version=rgi_version, rgi_reg=rgi_reg,
                border=border, output_folder=output_folder,
                working_dir=working_dir, params_file=args.params_file,
                is_test=args.test, test_ids=args.test_ids,
                rgi_file=args.rgi_file,
                intersects_file=args.intersects_file,
                dem_source=args.dem_source,
                start_level=args.start_level, start_base_url=args.start_base_url,
                start_from_dir=args.start_from_dir,
                max_level=args.max_level, disable_mp=args.disable_mp,
                chunk_idx=args.chunk_idx, chunk_size=args.chunk_size,
                glen_a_factor=args.inversion_glen_a_factor,
                inversion_fs=args.inversion_fs,
                logging_level=args.logging_level,
                elev_bands=args.elev_bands,
                skip_inversion=args.skip_inversion,
                inversion_volume_dataset=args.inversion_volume_dataset,
                centerlines=args.centerlines,
                select_source_from_dir=args.select_source_from_dir,
                keep_dem_folders=args.keep_dem_folders,
                add_consensus_thickness=args.add_consensus_thickness,
                add_millan_thickness=args.add_millan_thickness,
                add_itslive_velocity=args.add_itslive_velocity,
                add_millan_velocity=args.add_millan_velocity,
                add_hugonnet_dhdt=args.add_hugonnet_dhdt,
                add_bedmachine=args.add_bedmachine,
                add_glathida=args.add_glathida,
                add_distributed_thickness=args.add_distributed_thickness,
                add_export_thickness_geotiff=args.add_export_thickness_geotiff,
                compute_hypsometry=args.compute_hypsometry,
                custom_climate_task=args.custom_climate_task,
                custom_climate_task_kwargs=args.custom_climate_task_kwargs,
                dynamic_spinup=dynamic_spinup,
                ref_mb_err_scaling_factor=args.ref_mb_err_scaling_factor,
                dynamic_spinup_start_year=args.dynamic_spinup_start_year,
                dynamic_spinup_extra_years_to_try=extra_years_to_try,
                dynamic_spinup_allow_shorter=(
                    not args.dynamic_spinup_no_shorter_periods),
                mb_model_class=args.mb_model_class,
                mb_calibration_strategy=args.mb_calibration_strategy,
                geodetic_mb_file_path=args.geodetic_mb_file_path,
                temp_bias_file_path=args.temp_bias_file_path,
                temp_bias_run=args.temp_bias_run,
                store_fl_diagnostics=args.store_fl_diagnostics,
                store_hydro_output=args.store_hydro_output,
                store_monthly_hydro=args.store_monthly_hydro,
                ref_area_yr=args.ref_area_yr,
                override_params=args.override_params,
                )


def main():
    """Script entry point"""

    run_prepro_levels(**parse_args(sys.argv[1:]))
