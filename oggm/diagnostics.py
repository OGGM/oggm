"""Diagnostics of an OGGM preprocessing run.

This module reads the *summary* files written by ``oggm_prepro`` (the compiled
per region csv and netcdf files, not the glacier directories) and turns them
into a report: how much of the run actually completed, how well the dynamic
spinup did its job, how the modelled mass loss compares to the geodetic
observations of Hugonnet et al. (2021), how the modelled area compares to the
RGI inventory, and what the calibrated mass balance parameters look like.

The command line interface to all this is ``oggm_prepro_diag`` (see
:py:mod:`oggm.cli.prepro_diag`), but every table is also available on its own
so that the numbers can be used in a notebook.

A note on areas: everywhere in here, "area" means ``area_min_h``, i.e. the area
of the flowline gridpoints thicker than ``min_ice_thick_for_area``. This is the
area the dynamic spinup calibrates against, and therefore the only one which
can be compared to the RGI inventory area. The plain ``area`` variable is
reported in the time series table as well, but never plotted or used in a
metric.
"""

# Standard libraries
import glob
import logging
import datetime
import warnings
from collections import OrderedDict
from pathlib import Path

# External libs
import numpy as np
import pandas as pd
import xarray as xr

# Locals
import oggm.cfg as cfg
from oggm import __version__, utils
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.utils._workflow import _read_glacier_statistics_files

log = logging.getLogger(__name__)

# The two runs `oggm_prepro` writes at L4 and L5. The keys are used in the
# tables and file names, the file names are what prepro_levels writes.
PREPRO_RUNS = OrderedDict([
    ('spinup', {'fname': 'spinup_historical_run_output',
                'label': 'dynamic spinup'}),
    ('fixed_geom', {'fname': 'historical_run_output',
                    'label': 'fixed geometry spinup'}),
])

# The name of the first order RGI regions. Identical in RGI6 and RGI7, and
# hardcoded on purpose: this is used for plot titles only, and we do not want
# a download for that.
RGI_REGION_NAMES = {
    '01': 'Alaska',
    '02': 'Western Canada and USA',
    '03': 'Arctic Canada North',
    '04': 'Arctic Canada South',
    '05': 'Greenland Periphery',
    '06': 'Iceland',
    '07': 'Svalbard and Jan Mayen',
    '08': 'Scandinavia',
    '09': 'Russian Arctic',
    '10': 'North Asia',
    '11': 'Central Europe',
    '12': 'Caucasus and Middle East',
    '13': 'Central Asia',
    '14': 'South Asia West',
    '15': 'South Asia East',
    '16': 'Low Latitudes',
    '17': 'Southern Andes',
    '18': 'New Zealand',
    '19': 'Subantarctic and Antarctic Islands',
}

# The variables of the compiled run output the diagnostics need. Note that
# `area_min_h` and not `area` is the area used everywhere (see the module
# docstring).
RUN_VARIABLES = ['volume', 'area', 'area_min_h', 'mass_kg',
                 'error_during_run', 'is_partial_output',
                 'is_fixed_geometry_spinup']


def _slug(name):
    """A column-name friendly version of a free text option."""
    for c in ' ()':
        name = name.replace(c, '_')
    return name.replace('__', '_').strip('_').lower()


# The initialisation options `run_dynamic_melt_f_calibration` can end up with,
# as they are written in the `used_spinup_option` column, plus the glaciers
# for which nothing was written down at all. In plotting order.
SPINUP_OPTIONS = ['dynamic melt_f calibration (full success)',
                  'dynamic melt_f calibration (part success)',
                  'dynamic spinup only',
                  'fixed geometry spinup']
SPINUP_CATEGORIES = [_slug(_o) for _o in SPINUP_OPTIONS] + ['not_recorded']

# The index of the aggregated row in all the tables
GLOBAL = 'global'


def _add_area_percentages(df, total='rgi_area_km2'):
    """Adds a `perc_<name>` column for each `<name>_km2` area column."""
    for c in [c for c in df.columns if c.endswith('_km2') and c != total]:
        df['perc_' + c[:-len('_km2')]] = df[c] / df[total] * 100
    return df


def _global_row(df, how='sum'):
    """Adds the aggregated row at the end of a per region table."""
    if how == 'sum':
        df.loc[GLOBAL] = df.sum(numeric_only=True)
    else:
        raise NotImplementedError(how)
    return df


def _region_of(rgi_ids):
    """The RGI region ('01', '02', ...) out of RGI ids, RGI6 or RGI7."""
    rgi_ids = pd.Series(pd.Index(rgi_ids).astype(str))
    # RGI60-11.00001 and RGI2000-v7.0-G-11-00001 both have the region right
    # after the first separator which follows the version
    out = rgi_ids.str.extract(r'RGI\d+-(\d\d)\.', expand=False)
    out = out.fillna(rgi_ids.str.extract(r'-[GC]-(\d\d)-', expand=False))
    return out.values


def _level_of(path):
    """The preprocessing level of a path, out of its `L?` directory."""
    for part in Path(path).parts:
        if len(part) == 2 and part[0] == 'L' and part[1].isdigit():
            return int(part[1])
    return -1


def _rgi_version_from_ids(rgi_ids):
    """Guesses the RGI version out of the glacier ids."""
    rid = str(pd.Index(rgi_ids)[0])
    if rid.startswith('RGI60-') or rid.startswith('RGI61-'):
        return '62'
    if '-v7.0-G-' in rid:
        return '70G'
    if '-v7.0-C-' in rid:
        return '70C'
    return None


class PreproRun(object):
    """The summary output of one `oggm_prepro` run, ready to be diagnosed.

    Use :py:func:`read_prepro_run` to create one.

    Attributes
    ----------
    summary_dir : Path
        the `summary` directory the data was read from
    name : str
        a human readable name for the run (used in the report and plot titles)
    level : int or None
        the preprocessing level (3, 4 or 5)
    rgi_version : str or None
        '62', '70G' or '70C'
    border : int or None
        the map border of the run
    regions : list of str
        the RGI regions found, e.g. ['01', '11']
    stats : pandas.DataFrame
        the concatenated glacier statistics, indexed by rgi_id
    runs : list of str
        the keys of `PREPRO_RUNS` for which output files were found
    """

    def __init__(self, summary_dir, stats, regions, runs, level=None,
                 rgi_version=None, border=None, name=None):
        self.summary_dir = Path(summary_dir)
        self.stats = stats
        self.regions = regions
        self.runs = runs
        self.level = level
        self.rgi_version = rgi_version
        self.border = border
        self.name = name or self.summary_dir.parent.name
        self._ds_cache = {}
        self._pop_cache = {}

    def __repr__(self):
        return ('<PreproRun {}: L{}, RGI{}, border {}, {} regions, {} '
                'glaciers>'.format(self.name, self.level, self.rgi_version,
                                   self.border, len(self.regions),
                                   len(self.stats)))

    def has_run(self, run):
        """Is the output of this run (e.g. 'spinup') available?"""
        return run in self.runs

    def open_run(self, run, region):
        """The compiled run output of one region, as an xarray Dataset.

        Only the variables the diagnostics need are read: the files also hold
        a dozen others, and decompressing them all is by far the slowest part
        of the whole thing (5 s instead of 0.1 s for a large region).
        """
        key = (run, region)
        if key not in self._ds_cache:
            fname = f'{PREPRO_RUNS[run]["fname"]}_{region}.nc'
            fpath = self.summary_dir / fname
            if not fpath.exists():
                return None
            with xr.open_dataset(fpath) as ds:
                keep = [v for v in RUN_VARIABLES if v in ds]
                self._ds_cache[key] = ds[keep].load()
        return self._ds_cache[key]

    def region_stats(self, region):
        """The glacier statistics of one region."""
        return self.stats.loc[self.stats['_reg'] == region]

    def population(self, region):
        """The glaciers which are valid in *all* the available runs.

        All the aggregated numbers (time series, geodetic mass balance) are
        computed on this population, so that the runs are always compared on
        the same glaciers.

        Returns
        -------
        (index of the valid glaciers, dict of diagnostics)
        """
        if region in self._pop_cache:
            return self._pop_cache[region]

        sdf = self.region_stats(region)
        valid = sdf.index
        if not self.runs and 'error_task' in sdf:
            # Without any run output (a level 3 directory) the statistics file
            # is all we have to go by. With runs we deliberately do *not*
            # filter on `error_task`: a glacier can have a task error on
            # record and still have a complete, usable run - the dynamic
            # melt_f calibration logs an error and then falls back, and it
            # does so on 20% of RGI19 in the example runs. Dropping those
            # would silently take a fifth of the largest region out of every
            # regional sum. The two accountings are reported side by side in
            # `compute_completion` instead.
            valid = sdf.index[sdf['error_task'].isnull()]
        diag = {'n_stats': len(sdf)}
        for run in self.runs:
            ds = self.open_run(run, region)
            ok, _ = _valid_glaciers(ds)
            diag[f'n_ok_{run}'] = int(ok.sum())
            valid = valid.intersection(ds.rgi_id.values[ok])
        diag['n_population'] = len(valid)
        out = (valid, diag)
        self._pop_cache[region] = out
        return out


def _select_glaciers(ds, rgi_ids):
    """Selects glaciers in a compiled run output, by id.

    This is `ds.sel(rgi_id=ds.rgi_id.isin(rgi_ids))`, but `xarray.isin` falls
    back on `numpy.in1d`, which is quadratic on the string arrays we have here
    (two minutes for a large region). `pandas.Index.isin` hashes instead.
    """
    mask = pd.Index(ds['rgi_id'].values).isin(pd.Index(rgi_ids))
    return ds.isel(rgi_id=np.nonzero(mask)[0])


def _valid_glaciers(ds):
    """Which glaciers of a compiled run output can be used, and from when.

    A glacier is valid if it ran to the end of the simulation (this excludes
    the glaciers which errored, and the partial outputs). Because of the
    variable spinup periods, valid glaciers do not necessarily start in the
    same year - hence the second return value.

    Returns
    -------
    (boolean array of the valid glaciers, array of their first year)
    """
    vol = ds['volume'].values  # (time, rgi_id)
    finite = np.isfinite(vol)
    ok = finite[-1, :]
    if 'error_during_run' in ds:
        ok = ok & (ds['error_during_run'].values == '')
    if 'is_partial_output' in ds:
        with warnings.catch_warnings():
            # NaNs in there are fine
            warnings.simplefilter('ignore', category=RuntimeWarning)
            ok = ok & (np.nan_to_num(ds['is_partial_output'].values) == 0)

    # The first year with data, per glacier (nonsense where not ok)
    first = ds['time'].values[np.argmax(finite, axis=0)]
    return ok, first


def read_prepro_run(input_dir, rgi_region=None, name=None):
    """Reads the summary output of an `oggm_prepro` run.

    Parameters
    ----------
    input_dir : str or Path
        the directory to read. This can be the `summary` directory itself,
        the level directory containing it (e.g. `.../L5`), or the directory
        containing the levels (e.g. `.../b_160`), in which case the highest
        level available is used.
    rgi_region : str or list, optional
        read only these RGI regions (e.g. '11').
    name : str, optional
        a human readable name for this run (defaults to the name of the
        directory above the levels).

    Returns
    -------
    a :py:class:`PreproRun`
    """

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise InvalidParamsError(f'No such directory: {input_dir}')

    # Find the summary dir: here, or somewhere below. If there are several
    # (i.e. we were given the directory above the levels), the highest level
    # wins - it is the one with the most output files.
    if glob.glob(str(input_dir / 'glacier_statistics*.csv')):
        summary_dir = input_dir
    else:
        found = sorted({Path(f).parent for f in glob.glob(
            str(input_dir / '**' / 'glacier_statistics_*.csv'),
            recursive=True)})
        if len(found) == 0:
            raise InvalidParamsError(
                'Could not find a `summary` directory with glacier statistics '
                f'in {input_dir}. Point `--input` at the summary directory of '
                'a preprocessing level, e.g. `.../RGI62/b_160/L5/summary`.')
        levels = [_level_of(f) for f in found]
        keep = [f for f, lev in zip(found, levels) if lev == max(levels)]
        if len(keep) > 1:
            raise InvalidParamsError(
                f'Found several preprocessing runs below {input_dir}:\n' +
                '\n'.join(f'    {f}' for f in keep) +
                '\nPoint `--input` at one of them.')
        summary_dir = keep[0]

    # What we can learn from the path itself
    level, rgi_version, border = _level_of(summary_dir), None, None
    level = level if level >= 0 else None
    for part in summary_dir.parts:
        if part.startswith('RGI') and len(part) > 3:
            rgi_version = part[3:]
        elif part.startswith('b_') and part[2:].isdigit():
            border = int(part[2:])

    # The regions we have
    files = sorted(glob.glob(str(summary_dir / 'glacier_statistics_*.csv')))
    regions = [Path(f).stem.split('_')[-1] for f in files]
    if rgi_region is not None:
        sel = ['{:02d}'.format(int(r)) for r in utils.tolist(rgi_region)]
        files = [f for f, r in zip(files, regions) if r in sel]
        regions = [r for r in regions if r in sel]
        if not files:
            raise InvalidParamsError(
                f'No glacier statistics file found for region(s) {sel} in '
                f'{summary_dir}.')
    if not files:
        raise InvalidParamsError('No `glacier_statistics_*.csv` file found '
                                 f'in {summary_dir}.')

    stats, _ = _read_glacier_statistics_files(files)
    stats['_reg'] = _region_of(stats.index)
    if pd.isnull(stats['_reg']).any():
        # Should not happen, but we do not want to silently drop glaciers
        raise InvalidWorkflowError('Could not parse the RGI region out of the '
                                   'glacier ids of the statistics file(s).')

    # Cross-check the RGI version with the ids - the path can lie (or be
    # missing), the ids cannot
    rgi_version_ids = _rgi_version_from_ids(stats.index)
    if rgi_version_ids is None:
        pass
    elif rgi_version is None:
        rgi_version = rgi_version_ids
    elif rgi_version != rgi_version_ids:
        log.warning(f'The directory says RGI{rgi_version} but the glacier ids '
                    f'say RGI{rgi_version_ids} - using the latter.')
        rgi_version = rgi_version_ids

    # Which runs do we have?
    runs = []
    for run, meta in PREPRO_RUNS.items():
        if all((summary_dir / f'{meta["fname"]}_{r}.nc').exists()
               for r in regions):
            runs.append(run)
        elif any((summary_dir / f'{meta["fname"]}_{r}.nc').exists()
                 for r in regions):
            log.warning(f'The output of the `{run}` run is available for some '
                        'regions only - ignoring it.')

    if name is None:
        # .../<name>/RGI62/b_160/L5/summary -> <name>
        parts = summary_dir.absolute().parts
        rgi_at = [i for i, p in enumerate(parts) if p.startswith('RGI')]
        name = parts[rgi_at[0] - 1] if rgi_at else summary_dir.parent.name

    prun = PreproRun(summary_dir, stats, regions, runs, level=level,
                     rgi_version=rgi_version, border=border, name=name)
    log.workflow(f'Read {prun}')
    return prun


def compute_completion(prun):
    """How much of the run completed, per region.

    Two independent accountings, which do *not* always agree: the glacier
    statistics file (`error_task`) knows about the glaciers which failed
    during the preprocessing tasks, while the compiled run output knows about
    the glaciers which failed (or stopped early) during the runs themselves.

    Returns
    -------
    a DataFrame indexed by region, with a `global` row.
    """

    rows = OrderedDict()
    for reg in prun.regions:
        sdf = prun.region_stats(reg)
        area = sdf['rgi_area_km2']
        d = {'n_glaciers': len(sdf), 'rgi_area_km2': area.sum()}

        ok = sdf['error_task'].isnull() if 'error_task' in sdf else \
            pd.Series(True, index=sdf.index)
        d['n_ok_stats'] = int(ok.sum())
        d['area_ok_stats_km2'] = area[ok].sum()

        for run in prun.runs:
            ds = prun.open_run(run, reg)
            valid, _ = _valid_glaciers(ds)
            ids = pd.Index(ds.rgi_id.values[valid]).intersection(sdf.index)
            d[f'n_ok_{run}'] = len(ids)
            d[f'area_ok_{run}_km2'] = area.reindex(ids).sum()

        pop, _ = prun.population(reg)
        d['n_population'] = len(pop)
        d['area_population_km2'] = area.reindex(pop).sum()
        rows[reg] = d

    df = pd.DataFrame.from_dict(rows, orient='index')
    df = _global_row(df)

    # The percentages are computed after the sum, not summed
    return _add_area_percentages(df)


def compute_errors(prun, max_msg_length=80):
    """Which tasks the failed glaciers failed on, by number and by area.

    Errors are split by what they cost us, in the `status` column:

    - `fatal`: the glacier has no usable output at the end, so it is out of
      every aggregated number of the report.
    - `recovered`: the task errored, but the glacier still has a complete run
      and is used like any other. This is what the dynamic melt_f calibration
      does when it does not converge: it runs with `ignore_errors=True`, logs
      the error and falls back to a simpler initialisation.

    The split is made on the outcome (does this glacier have complete output
    in every run?), not on the task itself. It has to be: the glacier
    directory keeps only the *last* error of each glacier, so a task which
    errored and was recovered is overwritten in the record by any later
    failure. In other words `recovered` means "this glacier ended up usable
    despite this error", not "this particular task was retried successfully".
    The per-glacier log files of the run hold the full trace, but they are not
    part of the summary output this tool reads.

    Returns
    -------
    a DataFrame with one row per (source, region, error, status), sorted by
    area.
    """

    rows = []
    for reg in prun.regions:
        sdf = prun.region_stats(reg)
        pop, _ = prun.population(reg)
        if 'error_task' in sdf:
            failed = sdf.loc[sdf['error_task'].notnull()]
            recovered = failed.index.isin(pop)
            for status, group in [('recovered', failed.loc[recovered]),
                                  ('fatal', failed.loc[~recovered])]:
                for task, sel in group.groupby('error_task'):
                    msg = ''
                    if 'error_msg' in sel:
                        msg = str(sel['error_msg'].iloc[0])[:max_msg_length]
                    rows.append({'source': 'statistics', 'region': reg,
                                 'status': status,
                                 'error': task, 'n': len(sel),
                                 'area_km2': sel['rgi_area_km2'].sum(),
                                 'example_msg': msg})

        for run in prun.runs:
            ds = prun.open_run(run, reg)
            valid, _ = _valid_glaciers(ds)
            ids = pd.Index(ds.rgi_id.values[~valid])
            if len(ids) == 0:
                continue
            errs = pd.Series('', index=pd.Index(ds.rgi_id.values))
            if 'error_during_run' in ds:
                errs = pd.Series(ds['error_during_run'].values,
                                 index=pd.Index(ds.rgi_id.values))
            errs = errs.reindex(ids).replace('', 'no output (did not run to '
                                                 'the end)')
            for err, sel in errs.groupby(errs):
                sel_ids = sel.index.intersection(sdf.index)
                # These are fatal by definition: no complete output
                rows.append({'source': f'run_{run}', 'region': reg,
                             'status': 'fatal',
                             'error': str(err)[:max_msg_length],
                             'n': len(sel),
                             'area_km2': sdf['rgi_area_km2']
                             .reindex(sel_ids).sum(),
                             'example_msg': ''})

    df = pd.DataFrame(rows, columns=['source', 'region', 'status', 'error',
                                     'n', 'area_km2', 'example_msg'])
    if len(df) == 0:
        return df

    # The same, globally
    glob_df = df.groupby(['source', 'status', 'error'], as_index=False).agg(
        {'n': 'sum', 'area_km2': 'sum', 'example_msg': 'first'})
    glob_df['region'] = GLOBAL
    df = pd.concat([glob_df[df.columns], df], ignore_index=True)
    return df.sort_values(['region', 'status', 'area_km2'],
                          ascending=[True, True, False]).reset_index(drop=True)


def _fixed_geometry_spinup_stats(ds, ok, area, ref_yr0,
                                 labelled_success=None,
                                 labelled_fallback=None):
    """How much of the spinup run is a fixed geometry spinup, not a dynamic one.

    `is_fixed_geometry_spinup` is True for the years of a glacier's series
    which come from a fixed geometry reconstruction, i.e. before the dynamic
    model starts (`flowline.py`, `run_until_and_store`). A glacier whose
    dynamic spinup had to be shortened has its first years filled that way,
    even though its melt_f calibration is reported as a success.

    Parameters
    ----------
    ds : xarray.Dataset
        the compiled output of the spinup run
    ok : ndarray of bool
        the glaciers of `ds` which ran to the end
    area : pandas.Series
        the RGI area of the glaciers, indexed by id
    ref_yr0 : int
        the first year of the geodetic reference period
    labelled_success : pandas.Index, optional
        the glaciers which `used_spinup_option` reports as a success (any of
        the three options which involve a dynamic spinup). Those of them which
        nevertheless start with a fixed geometry are the shortened spinups -
        the ones the statistics alone cannot show.
    labelled_fallback : pandas.Index, optional
        the glaciers which `used_spinup_option` calls a fixed geometry spinup.
        Needed because the flag alone misses some of them: when the RGI date
        is *before* the start year of the run there is nothing to pad, so the
        series carries no fixed geometry year at all even though the glacier
        was never spun up (its initial state is simply the inventory
        geometry). The union of the two is what "not dynamically spun up"
        means.

    Returns
    -------
    a dict of diagnostics for one region.
    """

    fg = ds['is_fixed_geometry_spinup'].values  # (time, rgi_id)
    time = ds['time'].values
    ids = pd.Index(ds['rgi_id'].values)
    a = area.reindex(ids).values
    fg = np.nan_to_num(fg) > 0

    # "At the start" means at each glacier's *own* first year, not at the
    # first year of the file: with variable spinup periods the series do not
    # all start together, and the early rows of a ragged file are NaN for
    # most glaciers - which would read as "starts dynamic" for all of them.
    finite = np.isfinite(ds['volume'].values)
    first_idx = np.argmax(finite, axis=0)
    fg_at_own_start = fg[first_idx, np.arange(fg.shape[1])]

    d = {}
    at_start = fg_at_own_start & ok
    dynamic = (~fg_at_own_start) & ok
    d['n_fixed_geom_at_start'] = int(at_start.sum())
    d['area_fixed_geom_at_start_km2'] = np.nansum(a[at_start])
    d['n_dynamic_from_start'] = int(dynamic.sum())
    d['area_dynamic_from_start_km2'] = np.nansum(a[dynamic])

    if labelled_success is not None:
        hidden = at_start & np.asarray(ids.isin(labelled_success))
        d['n_fixed_geom_at_start_not_labelled'] = int(hidden.sum())
        d['area_fixed_geom_at_start_not_labelled_km2'] = np.nansum(a[hidden])

    if labelled_fallback is not None:
        fallback = np.asarray(ids.isin(labelled_fallback)) & ok
        # No dynamic spinup at all, and nothing padded either: these are the
        # glaciers whose RGI date precedes the start year of the run
        no_pad = fallback & ~at_start
        d['n_no_spinup_no_padding'] = int(no_pad.sum())
        d['area_no_spinup_no_padding_km2'] = np.nansum(a[no_pad])
        # The honest total: everything which was not dynamically spun up
        not_spun_up = at_start | fallback
        d['n_not_spun_up'] = int(not_spun_up.sum())
        d['area_not_spun_up_km2'] = np.nansum(a[not_spun_up])
        d['n_spun_up'] = int((ok & ~not_spun_up).sum())
        d['area_spun_up_km2'] = np.nansum(a[ok & ~not_spun_up])

    # How long the fixed geometry part lasts, in years
    n_years = fg.sum(axis=0)[ok]
    if len(n_years):
        d['median_fixed_geom_years'] = float(np.median(n_years))
        d['max_fixed_geom_years'] = int(np.max(n_years))

    # Does it still reach into the period the mass balance is evaluated over?
    # If it does, that glacier's dmdtda is a fixed geometry mass balance.
    if ref_yr0 in time:
        i = int(np.nonzero(time == ref_yr0)[0][0])
        at_ref = fg[i, :] & ok
        d['n_fixed_geom_at_ref_yr'] = int(at_ref.sum())
        d['area_fixed_geom_at_ref_yr_km2'] = np.nansum(a[at_ref])
    return d


def _recovered_outcome(prun):
    """What the glaciers which errored but kept a usable run ended up with.

    An error on record does not say what happened next. The fallback of
    `run_dynamic_melt_f_calibration` puts melt_f back to its pre-calibration
    value and *retries a dynamic spinup*; only if that fails too does the
    glacier get a fixed geometry spinup. So "recovered" is not a synonym for
    "fixed geometry was used instead" - this tells which it was.

    Returns
    -------
    a DataFrame indexed by outcome, with the number and area of the glaciers.
    """

    rows = []
    for reg in prun.regions:
        sdf = prun.region_stats(reg)
        if 'error_task' not in sdf:
            continue
        pop, _ = prun.population(reg)
        ids = sdf.index[sdf['error_task'].notnull()].intersection(pop)
        if len(ids) == 0:
            continue
        opt = sdf['used_spinup_option'].reindex(ids) \
            if 'used_spinup_option' in sdf else pd.Series('?', index=ids)
        opt = opt.fillna('not recorded in the statistics')
        # The run output knows even when the statistics do not
        starts_fg = pd.Series(False, index=ids)
        if prun.has_run('spinup'):
            ds = prun.open_run('spinup', reg)
            if 'is_fixed_geometry_spinup' in ds:
                fg = pd.Series(
                    np.nan_to_num(ds['is_fixed_geometry_spinup']
                                  .values[0, :]) > 0,
                    index=pd.Index(ds['rgi_id'].values))
                starts_fg = fg.reindex(ids).fillna(False)
        for out, sel in opt.groupby(opt):
            rows.append({'outcome': out, 'n': len(sel),
                         'area_km2': sdf['rgi_area_km2'].reindex(sel.index).sum(),
                         'n_starts_fixed_geom': int(starts_fg[sel.index].sum())})
    if not rows:
        return None
    df = pd.DataFrame(rows).groupby('outcome').sum()
    return df.sort_values('area_km2', ascending=False)


def compute_spinup(prun, reference_period=None):
    """How the dynamic spinup and the dynamic melt_f calibration went.

    Two independent sources, and they do not say the same thing:

    - `used_spinup_option`, from the glacier statistics, is the *outcome of
      the melt_f calibration*: whether it converged, and what the fallback
      chain had to settle for. See `SPINUP_OPTIONS`.
    - `is_fixed_geometry_spinup`, from the run output itself, is what the
      series actually contains, year by year: it is True wherever the glacier
      geometry is held fixed because the dynamic model had not started yet.

    The second is the honest measure of "did the spinup do what was asked",
    and it is worse than the first suggests: when a dynamic spinup does not
    converge over the requested period, it is retried over shorter ones, and
    the years before the shortened period are filled with a fixed geometry
    spinup. Such a glacier is still labelled a full success, because the
    melt_f calibration did succeed - but its early years are not dynamic.
    `is_fixed_geometry_spinup` also survives a glacier whose statistics were
    not written, so it covers glaciers `used_spinup_option` knows nothing
    about.

    Parameters
    ----------
    prun : PreproRun
        the run to diagnose
    reference_period : str, optional
        the geodetic period, e.g. '2000-01-01_2020-01-01'. Used to report
        whether the fixed geometry part of the spinup still reaches into the
        period the mass balance is evaluated over. Defaults to
        `cfg.PARAMS['geodetic_mb_period']`.

    Returns
    -------
    a DataFrame indexed by region, with a `global` row.
    """

    if reference_period is None:
        reference_period = cfg.PARAMS['geodetic_mb_period']
    ref_yr0 = int(reference_period.split('_')[0].split('-')[0])

    if 'used_spinup_option' not in prun.stats:
        return None

    options = SPINUP_OPTIONS

    rows = OrderedDict()
    for reg in prun.regions:
        sdf = prun.region_stats(reg)
        area = sdf['rgi_area_km2']
        d = {'n_glaciers': len(sdf), 'rgi_area_km2': area.sum()}

        opt = sdf['used_spinup_option']
        for o in options:
            sel = opt == o
            d['n_' + _slug(o)] = int(sel.sum())
            d['area_' + _slug(o) + '_km2'] = area[sel].sum()
        # Not all glaciers have their initialisation on record: when a task
        # errors, the statistics are written without the diagnostics of that
        # task. Most of these do have a complete (fallback) run - saying they
        # had "no spinup" would overstate the failure by a lot, so we count
        # them apart and say how many of them ran.
        missing = opt.isnull()
        d['n_not_recorded'] = int(missing.sum())
        d['area_not_recorded_km2'] = area[missing].sum()
        if prun.has_run('spinup'):
            ds = prun.open_run('spinup', reg)
            ok, _ = _valid_glaciers(ds)
            ran = sdf.index[missing].intersection(ds['rgi_id'].values[ok])
            d['n_not_recorded_but_ran'] = len(ran)
            d['area_not_recorded_but_ran_km2'] = area.reindex(ran).sum()

            # What the series actually contains, from the run output. This
            # does not depend on the statistics having been written, and it
            # sees the shortened spinups which `used_spinup_option` reports
            # as a success.
            if 'is_fixed_geometry_spinup' in ds:
                success = [o for o in SPINUP_OPTIONS
                           if o != 'fixed geometry spinup']
                d.update(_fixed_geometry_spinup_stats(
                    ds, ok, area, ref_yr0,
                    labelled_success=sdf.index[opt.isin(success)],
                    labelled_fallback=sdf.index[opt == 'fixed geometry '
                                                       'spinup']))

        # The area match of the dynamic spinup. The calibration stops when it
        # is below 1%, so this is the fraction which reached its target.
        if 'area_mismatch_dynamic_spinup_km2_percent' in sdf:
            mm = sdf['area_mismatch_dynamic_spinup_km2_percent'].abs()
            valid = mm.notnull()
            d['n_area_match'] = int((mm[valid] < 1).sum())
            d['area_area_match_km2'] = area[valid & (mm < 1)].sum()
            d['median_area_mismatch_percent'] = mm[valid].median()

        # The dmdtda match of the dynamic melt_f calibration: did it converge
        # inside the (scaled) uncertainty of the observation?
        cols = ['dmdtda_mismatch_dynamic_calibration',
                'dmdtda_dynamic_calibration_given_error',
                'dmdtda_dynamic_calibration_error_scaling_factor']
        if all(c in sdf for c in cols):
            mm = sdf[cols[0]].abs()
            tol = sdf[cols[1]].abs() * sdf[cols[2]]
            valid = mm.notnull() & tol.notnull()
            d['n_dmdtda_match'] = int((mm[valid] <= tol[valid]).sum())
            d['area_dmdtda_match_km2'] = area[valid & (mm <= tol)].sum()
            d['median_dmdtda_mismatch'] = mm[valid].median()
            d['n_dmdtda_calibrated'] = int(valid.sum())

        # How long a constant climate spinup was needed. This is the period
        # the calibration ended up using, which is not necessarily the one
        # which was asked for (it shortens it when it does not converge).
        if 'dynamic_spinup_period' in sdf:
            sp = sdf['dynamic_spinup_period']
            d['median_spinup_period'] = sp.median()
            d['min_spinup_period'] = sp.min()
            d['max_spinup_period'] = sp.max()

        # When the model output of the spinup run actually starts. This is
        # `target_yr - dynamic_spinup_period`, clipped at the start year of
        # the run - and it is what makes the regional sums ragged, so we take
        # it from the run output itself rather than from the statistics.
        if prun.has_run('spinup'):
            ds = prun.open_run('spinup', reg)
            ok, first = _valid_glaciers(ds)
            if ok.sum() > 0:
                d['first_output_yr_min'] = int(np.min(first[ok]))
                d['first_output_yr_median'] = float(np.median(first[ok]))
                d['first_output_yr_max'] = int(np.max(first[ok]))
                d['n_before_common_start'] = int((first[ok] <
                                                  np.max(first[ok])).sum())

        rows[reg] = d

    df = pd.DataFrame.from_dict(rows, orient='index')

    # Sum what can be summed, then recompute the rest globally: a sum makes
    # no sense for a median or an extreme value
    per_reg = df.copy()
    df = _global_row(df)
    w = per_reg['rgi_area_km2']
    for c in per_reg.columns:
        vals = per_reg[c]
        ok = vals.notnull()
        if not ok.any():
            continue
        if c.startswith('median_') or c.startswith('first_output_yr_median'):
            df.loc[GLOBAL, c] = np.average(vals[ok], weights=w[ok])
        elif c.startswith('min_') or c.endswith('_min'):
            df.loc[GLOBAL, c] = vals[ok].min()
        elif c.startswith('max_') or c.endswith('_max'):
            df.loc[GLOBAL, c] = vals[ok].max()

    return _add_area_percentages(df)


def compute_mb_params(prun):
    """Summary statistics of the calibrated mass balance parameters.

    Returns
    -------
    a DataFrame indexed by region, with a `global` row.
    """

    params = [p for p in ['melt_f', 'prcp_fac', 'temp_bias', 'bias']
              if p in prun.stats]
    if not params:
        return None

    melt_f_max = cfg.PARAMS.get('melt_f_max', None)
    melt_f_min = cfg.PARAMS.get('melt_f_min', None)

    rows = OrderedDict()
    for reg in list(prun.regions) + [GLOBAL]:
        sdf = prun.stats if reg == GLOBAL else prun.region_stats(reg)
        area = sdf['rgi_area_km2']
        d = {'n_glaciers': len(sdf), 'rgi_area_km2': area.sum()}
        for p in params:
            s = sdf[p]
            ok = s.notnull()
            d[f'{p}_n'] = int(ok.sum())
            if ok.sum() == 0:
                continue
            d[f'{p}_mean'] = s[ok].mean()
            d[f'{p}_wmean'] = np.average(s[ok], weights=area[ok])
            for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
                d[f'{p}_q{int(q * 100):02d}'] = s[ok].quantile(q)
            d[f'{p}_std'] = s[ok].std()

        if 'melt_f' in params and melt_f_max is not None:
            at_max = np.isclose(sdf['melt_f'], melt_f_max)
            at_min = np.isclose(sdf['melt_f'], melt_f_min)
            d['melt_f_perc_at_max'] = at_max.mean() * 100
            d['melt_f_perc_at_min'] = at_min.mean() * 100
            d['melt_f_perc_area_at_max'] = (area[at_max].sum() /
                                            area.sum() * 100)

        # What the dynamic calibration did to melt_f
        if all(c in sdf for c in ['melt_f_dynamic_calibration',
                                  'melt_f_before_dynamic_calibration']):
            dmf = (sdf['melt_f_dynamic_calibration'] -
                   sdf['melt_f_before_dynamic_calibration'])
            ok = dmf.notnull()
            moved = ok & (np.abs(dmf) > 1e-4)
            if ok.sum() > 0:
                d['melt_f_dyn_change_n'] = int(ok.sum())
                d['melt_f_dyn_change_perc_moved'] = moved.sum() / ok.sum() * 100
            # The statistics are those of the glaciers it actually moved:
            # over all of them the median is 0 by construction
            if moved.sum() > 0:
                d['melt_f_dyn_change_median'] = dmf[moved].median()
                d['melt_f_dyn_change_q25'] = dmf[moved].quantile(0.25)
                d['melt_f_dyn_change_q75'] = dmf[moved].quantile(0.75)
        rows[reg] = d

    return pd.DataFrame.from_dict(rows, orient='index')


def compute_rgi_reference(prun):
    """The RGI inventory area of each region, and when it was surveyed.

    The RGI date varies from glacier to glacier, so the reference year of a
    region is the median of the glacier dates, weighted by their area.

    Returns
    -------
    a DataFrame indexed by region, with a `global` row.
    """

    rows = OrderedDict()
    for reg in list(prun.regions) + [GLOBAL]:
        sdf = prun.stats if reg == GLOBAL else prun.region_stats(reg)
        area = sdf['rgi_area_km2']
        d = {'rgi_area_km2': area.sum(), 'n_glaciers': len(sdf)}
        if 'rgi_year' in sdf:
            ok = sdf['rgi_year'].notnull() & (area > 0)
            if ok.sum() > 0:
                d['rgi_year_wmedian'] = utils.weighted_quantile_1d(
                    sdf['rgi_year'][ok].values, area[ok].values, 0.5)
                d['rgi_year_min'] = sdf['rgi_year'][ok].min()
                d['rgi_year_max'] = sdf['rgi_year'][ok].max()
        rows[reg] = d
    return pd.DataFrame.from_dict(rows, orient='index')


def compute_timeseries(prun):
    """The regional volume and area evolution of each run.

    Only the glaciers which are valid in all the runs are summed (see
    :py:meth:`PreproRun.population`), so that the runs can be compared.

    Because the spinup period can differ from glacier to glacier, valid
    glaciers do not all start in the same year. The `n_glaciers` column tells
    how many glaciers each year is made of, and `is_common_period` flags the
    years for which *all* of them have data - the only ones which should be
    plotted or interpreted.

    Returns
    -------
    a DataFrame in long format (one row per year, region and run).
    """

    if not prun.runs:
        return None

    variables = {'volume': ('volume_km3', 1e-9),
                 'area_min_h': ('area_min_h_km2', 1e-6),
                 'area': ('area_km2', 1e-6),
                 'mass_kg': ('mass_Gt', 1e-12)}

    frames = []
    for reg in prun.regions:
        pop, _ = prun.population(reg)
        for run in prun.runs:
            ds = _select_glaciers(prun.open_run(run, reg), pop)
            df = pd.DataFrame(index=pd.Index(ds['time'].values, name='year'))
            n_valid = np.isfinite(ds['volume'].values).sum(axis=1)
            df['n_glaciers'] = n_valid
            df['is_common_period'] = n_valid == len(ds.rgi_id)
            for v, (name, fac) in variables.items():
                if v not in ds:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    df[name] = np.nansum(ds[v].values, axis=1) * fac
            df['region'] = reg
            df['run'] = run
            frames.append(df.reset_index())

    df = pd.concat(frames, ignore_index=True)

    # The global aggregate: only the years all regions have in common
    gdf = df.groupby(['run', 'year'], as_index=False).agg(
        {c: 'sum' for c in df.columns
         if c not in ['run', 'year', 'region', 'is_common_period']})
    common = df.groupby(['run', 'year'])['is_common_period'].all()
    gdf['is_common_period'] = common.reindex(
        pd.MultiIndex.from_frame(gdf[['run', 'year']])).values
    gdf['region'] = GLOBAL
    df = pd.concat([df, gdf[df.columns]], ignore_index=True)

    return df.sort_values(['region', 'run', 'year']).reset_index(drop=True)


def common_period_start(ts, region, run):
    """The first year of the common period of a region and run."""
    sel = ts.loc[(ts['region'] == region) & (ts['run'] == run)]
    sel = sel.loc[sel['is_common_period']]
    return int(sel['year'].min()) if len(sel) else None


def compute_area_match(prun, ts=None, rgi_ref=None):
    """Modelled area at the RGI date, compared to the RGI inventory area.

    The modelled area is `area_min_h`, the only one comparable to the
    inventory (see the module docstring), interpolated at the area weighted
    median RGI year of the region.

    Returns
    -------
    a DataFrame indexed by region, with a `global` row.
    """

    if ts is None:
        ts = compute_timeseries(prun)
    if rgi_ref is None:
        rgi_ref = compute_rgi_reference(prun)
    if ts is None:
        return None

    out = rgi_ref.copy()

    # The inventory area of the glaciers which actually ran. Comparing to the
    # full regional area mixes two very different problems (a model which
    # shrinks too much, and glaciers which are missing altogether).
    pop_area = {}
    for reg in prun.regions:
        pop, _ = prun.population(reg)
        pop_area[reg] = prun.stats['rgi_area_km2'].reindex(pop).sum()
    pop_area[GLOBAL] = sum(pop_area.values())
    out['rgi_area_population_km2'] = pd.Series(pop_area)

    for run in prun.runs:
        vals, diffs = {}, {}
        for reg in out.index:
            yr = out.loc[reg].get('rgi_year_wmedian', np.nan)
            sel = ts.loc[(ts['region'] == reg) & (ts['run'] == run)]
            sel = sel.loc[sel['is_common_period']]
            if not np.isfinite(yr) or len(sel) == 0:
                continue
            yr = float(np.clip(yr, sel['year'].min(), sel['year'].max()))
            v = np.interp(yr, sel['year'].values,
                          sel['area_min_h_km2'].values)
            vals[reg] = v
            diffs[reg] = (v - out.loc[reg, 'rgi_area_km2'])
        out[f'area_min_h_at_rgi_yr_{run}_km2'] = pd.Series(vals)
        out[f'area_mismatch_{run}_km2'] = pd.Series(diffs)
        out[f'area_mismatch_{run}_percent'] = (pd.Series(diffs) /
                                               out['rgi_area_km2'] * 100)
        # The same, against the glaciers which ran only - this is the number
        # which says something about the model rather than about the errors
        out[f'area_mismatch_pop_{run}_percent'] = (
            (pd.Series(vals) - out['rgi_area_population_km2']) /
            out['rgi_area_population_km2'] * 100)
    return out


def compute_geodetic(prun, reference_period=None):
    """Compares the modelled mass change to Hugonnet et al. (2021).

    The model value is computed exactly the way OGGM's own calibration does
    it (see `run_dynamic_melt_f_calibration`): the mass change of the glacier
    over the reference period, divided by its *RGI* area and by the length of
    the period. The regional value is the area weighted mean of these, i.e.
    the total mass change of the region divided by its total RGI area - the
    same aggregation Hugonnet et al. use.

    Three references are provided:

    - `hug_pergla_*`: the per glacier observations, aggregated over exactly
      the glaciers OGGM modelled. This is the apples-to-apples comparison:
      same glaciers, same area weights, no extrapolation.
    - `hug_reg_*`: the regional averages published by Hugonnet et al., which
      include an extrapolation to the unmeasured glaciers, and an error
      estimate. Note that `hug_reg_dmdtda_published` is relative to the
      measured area only: `hug_reg_dmdtda` (derived from `dmdt` and the full
      regional area) is the one to compare to. The published regional file is
      RGI6 based.
    - the per glacier calibration residuals of the run itself, in
      :py:func:`compute_spinup`.

    Parameters
    ----------
    prun : PreproRun
        the run to diagnose
    reference_period : str, optional
        the geodetic period to compare over, e.g. '2000-01-01_2020-01-01'.
        Defaults to `cfg.PARAMS['geodetic_mb_period']`.

    Returns
    -------
    a DataFrame indexed by region, with a `global` row.
    """

    if not prun.runs:
        return None

    if reference_period is None:
        reference_period = cfg.PARAMS['geodetic_mb_period']
    y0, y1 = reference_period.split('_')
    y0, y1 = int(y0.split('-')[0]), int(y1.split('-')[0])
    dt = y1 - y0

    # The observations
    hug_pergla, hug_reg = None, None
    try:
        hug_pergla = utils.get_geodetic_mb_dataframe(
            rgi_version=prun.rgi_version)
        hug_pergla = hug_pergla.loc[hug_pergla['period'] == reference_period]
    except Exception as err:
        log.warning('Could not fetch the per glacier geodetic observations '
                    f'({err}) - the comparison will be incomplete.')
    try:
        hug_reg = utils.get_geodetic_mb_dataframe(regional=True)
        hug_reg = hug_reg.loc[hug_reg['period'] == reference_period]
    except Exception as err:
        log.warning('Could not fetch the regional geodetic observations '
                    f'({err}) - the comparison will be incomplete.')

    rows = OrderedDict()
    for reg in prun.regions:
        pop, _ = prun.population(reg)
        sdf = prun.stats.reindex(pop)
        area_m2 = sdf['rgi_area_km2'] * 1e6
        d = {'n_glaciers': len(pop), 'rgi_area_km2': sdf['rgi_area_km2'].sum()}

        for run in prun.runs:
            ds = _select_glaciers(prun.open_run(run, reg), pop)
            if y0 < ds['time'].values[0] or y1 > ds['time'].values[-1]:
                log.warning(f'Region {reg}: the `{run}` run does not cover '
                            f'the reference period {reference_period}.')
                continue
            mass = ds['mass_kg'].sel(time=[y0, y1]).to_pandas().T
            dm = (mass[y1] - mass[y0]) / dt  # kg yr-1
            ok = dm.notnull()
            a = area_m2.reindex(dm.index)[ok]
            d[f'{run}_dmdt_Gt'] = dm[ok].sum() * 1e-12
            d[f'{run}_dmdtda'] = dm[ok].sum() / a.sum() / 1000
            d[f'{run}_n'] = int(ok.sum())
            d[f'{run}_area_km2'] = a.sum() * 1e-6

        if hug_pergla is not None:
            sel = hug_pergla.reindex(pop).dropna(subset=['dmdtda'])
            if len(sel) > 0:
                # The observations are aggregated with the *RGI* areas, i.e.
                # the very weights the model values use: this way the two
                # differ by the observation only, not by the area they are
                # relative to. This matters where the two disagree, e.g. in
                # RGI region 12, for which Hugonnet et al. corrected the (very
                # wrong) RGI6 areas of some glaciers.
                # dmdtda is in m w.e. yr-1, area in m2: 1 m w.e. over 1 m2 is
                # 1000 kg, hence the 1e3 (and 1e-12 for kg -> Gt).
                w = area_m2.reindex(sel.index)
                d['hug_pergla_dmdtda'] = np.average(sel['dmdtda'], weights=w)
                d['hug_pergla_dmdt_Gt'] = (sel['dmdtda'] * w).sum() * 1e3 * 1e-12
                d['hug_pergla_n'] = len(sel)
                d['hug_pergla_area_km2'] = w.sum() * 1e-6
                # Hugonnet's own areas for the same glaciers - if this is far
                # from the RGI one, the region is worth a closer look
                d['hug_pergla_area_in_obs_km2'] = sel['area'].sum() * 1e-6

        if hug_reg is not None and int(reg) in hug_reg.index:
            s = hug_reg.loc[int(reg)]
            d['hug_reg_dmdt_Gt'] = s['dmdt']
            d['hug_reg_err_dmdt_Gt'] = s['err_dmdt']
            d['hug_reg_dmdtda'] = s['dmdtda_full_area']
            d['hug_reg_err_dmdtda'] = s['err_dmdtda_full_area']
            d['hug_reg_dmdtda_published'] = s['dmdtda']
            d['hug_reg_area_km2'] = s['area'] * 1e-6
            d['hug_reg_measured_area_km2'] = s['tarea'] * 1e-6

        rows[reg] = d

    df = pd.DataFrame.from_dict(rows, orient='index')
    if len(df) == 0:
        return df

    # The global row. Everything which is a total can be summed - the errors
    # too: they are correlated, so a quadratic sum would be optimistic, and
    # summing them is the conservative end of the range. The specific rates
    # (m w.e. yr-1) are *not* summable and are recomputed from the totals.
    specific = [c for c in df.columns if 'dmdtda' in c]
    g = {c: df[c].sum() for c in df.columns if c not in specific}
    for pref in list(prun.runs) + ['hug_pergla', 'hug_reg']:
        acol, dcol = f'{pref}_area_km2', f'{pref}_dmdt_Gt'
        if acol in g and dcol in g and g[acol] > 0:
            # Gt yr-1 over km2 -> m w.e. yr-1
            g[f'{pref}_dmdtda'] = g[dcol] * 1e3 / g[acol]
        ecol = f'{pref}_err_dmdt_Gt'
        if ecol in g and g.get(acol, 0) > 0:
            g[f'{pref}_err_dmdtda'] = g[ecol] * 1e3 / g[acol]
    if 'hug_reg_dmdtda_published' in df:
        # This one is relative to the measured area only (see docstring)
        w = df['hug_reg_measured_area_km2']
        v = df['hug_reg_dmdtda_published']
        ok = v.notnull() & w.notnull() & (w > 0)
        if ok.any():
            g['hug_reg_dmdtda_published'] = np.average(v[ok], weights=w[ok])
    df.loc[GLOBAL] = pd.Series(g)

    # The bias, which is what we are after
    for run in prun.runs:
        for ref in ['hug_pergla', 'hug_reg']:
            if f'{ref}_dmdtda' not in df or f'{run}_dmdtda' not in df:
                continue
            df[f'bias_{run}_vs_{ref}_dmdtda'] = (df[f'{run}_dmdtda'] -
                                                 df[f'{ref}_dmdtda'])
            df[f'bias_{run}_vs_{ref}_dmdt_Gt'] = (df[f'{run}_dmdt_Gt'] -
                                                  df[f'{ref}_dmdt_Gt'])
        if 'hug_reg_err_dmdtda' in df and f'{run}_dmdtda' in df:
            df[f'within_err_{run}'] = (
                np.abs(df[f'{run}_dmdtda'] - df['hug_reg_dmdtda']) <=
                df['hug_reg_err_dmdtda'])
    return df


def _table_str(df, columns=None, indent=4, float_format='{:.2f}'):
    """A per region table, as text."""
    if df is None or len(df) == 0:
        return ' ' * indent + '(no data)'
    if columns is not None:
        columns = [c for c in columns if c in df]
        df = df[columns]
    txt = df.to_string(float_format=float_format.format, na_rep='-')
    return '\n'.join(' ' * indent + ln for ln in txt.split('\n'))


def _run_label(run):
    return PREPRO_RUNS[run]['label']


def _run_name(run):
    """The name of the run as it appears on disk, e.g. `spinup_historical`.

    Used where `_run_label` ('dynamic spinup') would claim more than the
    number does: the `spinup_historical` files also hold the glaciers whose
    dynamic spinup failed and fell back to something else.
    """
    return PREPRO_RUNS[run]['fname'].replace('_run_output', '')


def prepro_diag_report(prun, tables, reference_period=None):
    """The diagnostic report of a preprocessing run, as text.

    Parameters
    ----------
    prun : PreproRun
        the run which was diagnosed
    tables : dict
        the output of :py:func:`compile_prepro_diagnostics`
    reference_period : str, optional
        the geodetic period the comparison was made over

    Returns
    -------
    the report, as a string.
    """

    lines = []
    add = lines.append

    def title(t):
        add('')
        add(t)
        add('-' * len(t))

    head = f'OGGM preprocessing diagnostics - {prun.name}'
    add(head)
    add('=' * len(head))
    add('Created on {} with OGGM {}'
        ''.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  __version__))
    add(f'Input directory : {prun.summary_dir}')
    add('Preprocessing   : level {}, RGI{}, map border {}'
        ''.format(prun.level, prun.rgi_version, prun.border))
    add('Regions         : {} ({} glaciers, {:.0f} km2)'
        ''.format(len(prun.regions), len(prun.stats),
                  prun.stats['rgi_area_km2'].sum()))
    add('                  {}'.format(', '.join(prun.regions)))
    if prun.runs:
        add('Runs found      : {}'.format(
            ', '.join(f'{_run_label(r)} ({PREPRO_RUNS[r]["fname"]})'
                      for r in prun.runs)))
    else:
        add('Runs found      : none (this is a level 3 directory - the '
            'sections about the runs are skipped)')

    # What the model output itself says about its provenance
    for run in prun.runs:
        ds = prun.open_run(run, prun.regions[0])
        add('    {:<22s}: OGGM {}, written on {}, years {} to {}'
            ''.format(_run_label(run), ds.attrs.get('oggm_version', '?'),
                      ds.attrs.get('creation_date', '?'),
                      int(ds['time'].values[0]), int(ds['time'].values[-1])))

    for col, label in [('baseline_climate_source', 'Climate'),
                       ('reference_period', 'MB calib period'),
                       ('dem_source', 'DEM')]:
        if col in prun.stats:
            vals = prun.stats[col].value_counts().head(3)
            add('{:<16s}: {}'.format(
                label, ', '.join(f'{k} ({v})' for k, v in vals.items())))
    if all(c in prun.stats for c in ['baseline_yr_0', 'baseline_yr_1']):
        add('{:<16s}: {:.0f} to {:.0f}'.format(
            'Climate years', prun.stats['baseline_yr_0'].min(),
            prun.stats['baseline_yr_1'].max()))
    if reference_period is not None:
        add('{:<16s}: {}'.format('Geodetic period', reference_period))
    add('')
    add('Note: "area" is always `area_min_h` below, i.e. the area of the '
        'gridpoints thicker')
    add('than `min_ice_thick_for_area`. This is what the dynamic spinup '
        'calibrates against,')
    add('and the only modelled area which can be compared to the RGI '
        'inventory area.')
    add('The full tables (and more columns than shown here) are in the '
        '`tables` directory.')

    # -- Completion
    comp = tables.get('completion')
    if comp is not None:
        title('Completion')
        g = comp.loc[GLOBAL]
        add('    Of the {:.0f} glaciers ({:.0f} km2) in the statistics '
            'file(s), most to least'.format(g['n_glaciers'],
                                            g['rgi_area_km2']))
        add('    important:')
        add('        {:.0f} are usable ({:.3f}% of the area): they have '
            'complete output in'.format(g['n_population'],
                                        g['perc_area_population']))
        add('            {} - this is the population all the aggregated '
            'numbers below'.format('every run' if len(prun.runs) > 1
                                   else 'the run'))
        add('            are computed on.')
        for run in prun.runs:
            add('        {:.0f} have complete output in the `{}` run ({:.3f}% '
                'of the area)'.format(g[f'n_ok_{run}'], _run_name(run),
                                      g[f'perc_area_ok_{run}']))
        add('        {:.0f} have no error at all on record ({:.3f}% of the '
            'area)'.format(g['n_ok_stats'], g['perc_area_ok_stats']))
        add('')
        add('    Complete output is NOT the same as a successful dynamic '
            'spinup: the glaciers')
        add('    whose spinup or calibration failed and fell back to '
            'something else are in')
        add('    there too, with a perfectly complete run. How the glaciers '
            'were actually')
        add('    initialised is the next section - that is the number to look '
            'at for the')
        add('    success of the spinup itself.')
        add('')
        add('    All the percentages are of the total RGI area of the region. '
            'The two columns')
        add('    do not measure the same thing, and `ok_stats` can be the '
            'lower one: it counts')
        add('    the glaciers whose error log is empty, while `ok_<run>` '
            'counts those whose run')
        add('    output is complete. A glacier which errored in a task that '
            'then fell back to')
        add('    something usable (the dynamic melt_f calibration runs with '
            '`ignore_errors`)')
        add('    keeps the error on record but does have a complete run - '
            'hence the population')
        add('    follows the runs, not the error log. Note also that the '
            'error log only keeps')
        add('    the *last* error of each glacier.')
        cols = (['n_glaciers', 'rgi_area_km2', 'n_ok_stats',
                 'perc_area_ok_stats'] +
                [f'perc_area_ok_{r}' for r in prun.runs] +
                ['n_population', 'perc_area_population'])
        add('')
        add(_table_str(comp, cols, float_format='{:.3f}'))

    errs = tables.get('errors')
    if errs is not None and len(errs) > 0:
        title('Errors (all regions together, the 10 largest by area of each)')
        glob = errs.loc[errs['region'] == GLOBAL]
        for status, what in [
                ('fatal', 'FATAL: these glaciers have no usable output and '
                          'are out of every'),
                ('recovered', 'RECOVERED: the task errored, but the glacier '
                              'still has a complete')]:
            sel = glob.loc[glob['status'] == status]
            add('')
            add('    ' + what)
            if status == 'fatal':
                add('    number of this report.')
            else:
                add('    run and is used like any other. The dynamic melt_f '
                    'calibration runs with')
                add('    `ignore_errors`: its fallback puts melt_f back to '
                    'the pre-calibration')
                add('    value and *retries a dynamic spinup*, and only if '
                    'that fails too does the')
                add('    glacier end up with a fixed geometry spinup. So this '
                    'is not the same as')
                add('    "fixed geometry was used instead" - what they '
                    'actually got is below.')
            if len(sel) == 0:
                add('        (none)')
                continue
            if status == 'fatal' and comp is not None:
                # Summing the rows would count a glacier once per source: the
                # same glacier is missing from the statistics *and* from each
                # run. The completion table has the unique count.
                cg = comp.loc[GLOBAL]
                add('        {:.0f} glaciers, {:.1f} km2 in total (unique '
                    'glaciers; the rows below'
                    ''.format(cg['n_glaciers'] - cg['n_population'],
                              cg['rgi_area_km2'] - cg['area_population_km2']))
                add('        count the same glacier once per source it is '
                    'missing from)')
            else:
                add('        {:.0f} glaciers, {:.1f} km2 in total'
                    ''.format(sel['n'].sum(), sel['area_km2'].sum()))
            add(_table_str(sel.head(10).set_index('source')[
                ['error', 'n', 'area_km2', 'example_msg']],
                float_format='{:.1f}', indent=8))
        rec = _recovered_outcome(prun)
        if rec is not None and len(rec):
            add('')
            add('    What the recovered glaciers ended up with '
                '(`used_spinup_option`, and')
            add('    `n_starts_fixed_geom` from the run output, which is set '
                'even when the')
            add('    statistics were not written):')
            add(_table_str(rec, float_format='{:.1f}', indent=8))

        add('')
        add('    The split is on the outcome, not on the task: the glacier '
            'directory keeps')
        add('    only the *last* error of each glacier, so `recovered` means '
            '"this glacier')
        add('    ended up usable despite this error", not "this task was '
            'retried successfully".')
        add('    The per-glacier log files of the run hold the full trace.')

    # -- Spinup
    spin = tables.get('spinup')
    if spin is not None:
        title('Dynamic spinup and dynamic melt_f calibration')
        ref_yr0 = int((reference_period or
                       cfg.PARAMS['geodetic_mb_period']).split('_')[0]
                      .split('-')[0])
        g = spin.loc[GLOBAL]
        add('    How the {:.0f} glaciers ended up being initialised (in % of '
            'the area):'.format(g['n_glaciers']))
        for cat in SPINUP_CATEGORIES:
            c = f'perc_area_{cat}'
            if c not in spin:
                continue
            add('        {:<45s}: {:>8.3f}%'.format(cat.replace('_', ' '),
                                                    g[c]))
        if 'n_not_recorded_but_ran' in spin:
            add('    `not recorded` is not the same as "no spinup happened": '
                'when a task errors,')
            add('    the statistics are written without its diagnostics, so '
                'we cannot tell how')
            add('    those glaciers were initialised. {:.0f} of these {:.0f} '
                'do have a complete run'.format(g['n_not_recorded_but_ran'],
                                                g['n_not_recorded']))
            add('    and are used below ({:.2f}% of the area, out of the '
                '{:.2f}%); the rest never ran'
                ''.format(g['perc_area_not_recorded_but_ran'],
                          g['perc_area_not_recorded']))
            add('    at all.')
        add('')
        add('    What these mean - they are the outcome of the melt_f '
            'calibration, not a')
        add('    statement about the geometry (see '
            '`run_dynamic_melt_f_calibration`):')
        add('      full success        : melt_f was tuned until the modelled '
            'dmdtda matched the')
        add('                            observation within its tolerance, '
            'with a dynamic spinup.')
        add('      part success        : the calibration ran but never '
            'reached the tolerance;')
        add('                            the best melt_f it found is kept.')
        add('      dynamic spinup only : the calibration failed. The fallback '
            'put melt_f back to')
        add('                            its pre-calibration value and did '
            'get a dynamic spinup,')
        add('                            so the geometry is dynamic but '
            'melt_f is NOT')
        add('                            dynamically calibrated.')
        add('      fixed geometry      : the fallback could not do a dynamic '
            'spinup either. This')
        add('                            glacier has no dynamic spinup at '
            'all: the years before')
        add('                            the RGI date are a fixed geometry '
            'mass balance.')

        # What the series actually contains - the number `used_spinup_option`
        # cannot give, because a shortened spinup is still a success to it
        if 'perc_area_not_spun_up' in spin:
            add('')
            add('    Was the geometry actually spun up? This combines the '
                'labels above with')
            add('    `is_fixed_geometry_spinup` from the run output, and '
                'neither of the two is')
            add('    enough on its own (see below). Of the glaciers which '
                'ran:')
            add('        dynamically spun up     : {:>7.2f}% of the area'
                ''.format(g['perc_area_spun_up']))
            add('        NOT dynamically spun up : {:>7.2f}% of the area, '
                'made of'.format(g['perc_area_not_spun_up']))
            add('            [a] the series starts with fixed geometry years '
                ': {:>7.2f}% ({:.0f} glaciers)'
                ''.format(g['perc_area_fixed_geom_at_start'],
                          g['n_fixed_geom_at_start']))
            if 'perc_area_fixed_geom_at_start_not_labelled' in spin:
                add('                of which reported as a SUCCESS above  '
                    ': {:>7.2f}% ({:.0f} glaciers)'
                    ''.format(g['perc_area_fixed_geom_at_start_not_labelled'],
                              g['n_fixed_geom_at_start_not_labelled']))
                add('                Their dynamic spinup did not converge '
                    'over the requested period,')
                add('                so it was retried over a shorter one and '
                    'the years before it were')
                add('                filled with a fixed geometry. The melt_f '
                    'calibration still')
                add('                succeeded, which is why the labels above '
                    'call them a success -')
                add('                but their early years are not dynamic. '
                    'This is the part the')
                add('                statistics alone cannot show.')
            if 'perc_area_no_spinup_no_padding' in spin:
                add('            [b] no dynamic spinup, and nothing padded    '
                    ': {:>7.2f}% ({:.0f} glaciers)'
                    ''.format(g['perc_area_no_spinup_no_padding'],
                              g['n_no_spinup_no_padding']))
                add('                Their RGI date is *before* the start '
                    'year of the run, so there')
                add('                is nothing to pad: the series is dynamic '
                    'from its first year,')
                add('                but the initial state is simply the '
                    'inventory geometry - it was')
                add('                never spun up. `is_fixed_geometry_spinup`'
                    ' is empty for them, so')
                add('                the flag alone would call them fine.')
            if 'median_fixed_geom_years' in spin:
                add('        length of the fixed geometry part   : median '
                    '{:.0f} years, max {:.0f}'
                    ''.format(g['median_fixed_geom_years'],
                              g['max_fixed_geom_years']))
            if 'perc_area_fixed_geom_at_ref_yr' in spin:
                add('        still fixed geometry at {}        : {:>7.2f}% of '
                    'the area - for those'.format(
                        ref_yr0, g['perc_area_fixed_geom_at_ref_yr']))
                add('            glaciers the mass balance over the geodetic '
                    'period is a fixed')
                add('            geometry one, not a dynamic one.')

        add('')
        if 'perc_area_area_match' in spin:
            add('    Glaciers whose dynamic spinup matched its area target '
                '(< 1%):')
            add('        {:.1f}% of the glaciers, {:.2f}% of the area, median '
                'mismatch {:.2f}%'
                ''.format(g['n_area_match'] / g['n_glaciers'] * 100,
                          g['perc_area_area_match'],
                          g['median_area_mismatch_percent']))
        if 'perc_area_dmdtda_match' in spin:
            add('    Glaciers whose dynamic melt_f calibration reached the '
                'geodetic observation')
            add('    within its (scaled) uncertainty:')
            add('        {:.1f}% of the calibrated glaciers, {:.2f}% of the '
                'area, median absolute'.format(g['n_dmdtda_match'] /
                                               g['n_dmdtda_calibrated'] * 100,
                                               g['perc_area_dmdtda_match']))
            add('        mismatch {:.1f} kg m-2 yr-1'
                ''.format(g['median_dmdtda_mismatch']))
        if 'first_output_yr_min' in spin:
            add('    First year of the spinup run output: {:.0f} at the '
                'earliest, {:.0f} at the'.format(g['first_output_yr_min'],
                                                 g['first_output_yr_max']))
            add('        latest. {:.0f} glaciers start before their regional '
                'common period, which'.format(g['n_before_common_start']))
            add('        is the one the regional sums below are computed on.')
        add('')
        cols = ([f'perc_area_{c}' for c in SPINUP_CATEGORIES] +
                ['perc_area_spun_up', 'perc_area_not_spun_up',
                 'perc_area_no_spinup_no_padding',
                 'perc_area_fixed_geom_at_start',
                 'perc_area_fixed_geom_at_start_not_labelled',
                 'perc_area_fixed_geom_at_ref_yr',
                 'median_fixed_geom_years',
                 'perc_area_not_recorded_but_ran', 'perc_area_area_match',
                 'perc_area_dmdtda_match',
                 'median_area_mismatch_percent', 'median_dmdtda_mismatch',
                 'median_spinup_period', 'first_output_yr_min',
                 'first_output_yr_max'])
        add(_table_str(spin, cols, float_format='{:.2f}'))

    # -- Geodetic
    geo = tables.get('geodetic')
    if geo is not None and len(geo) > 0:
        title('Geodetic mass balance vs Hugonnet et al. (2021)')
        add('    Period: {}. The model value is the mass change of the '
            'glaciers over'.format(reference_period))
        add('    that period, divided by their RGI area - exactly what the '
            'dynamic melt_f')
        add('    calibration matches. Two references:')
        add('      `hug_pergla`: the per glacier observations of the very '
            'glaciers which')
        add('                    were modelled, aggregated with the same RGI '
            'areas. This is')
        add('                    the apples-to-apples comparison.')
        add('      `hug_reg`   : the published regional averages, which are '
            'extrapolated to')
        add('                    the unmeasured glaciers and come with an '
            'error estimate.')
        if prun.rgi_version not in ['62', None]:
            add('                    CAREFUL: this file is RGI6 based, and '
                'this is a RGI{} run.'.format(prun.rgi_version))
        add('    In the `global` row the totals are summed over the regions '
            'present, and so')
        add('    are the observation errors: they are correlated, so this is '
            'the conservative')
        add('    end of the range (Hugonnet et al. give a smaller global '
            'error than this sum).')
        add('')
        g = geo.loc[GLOBAL]
        for run in prun.runs:
            add('    {} ({}):'.format(_run_label(run), run))
            add('        dmdt   : {:>9.2f} Gt yr-1  (obs {:>9.2f} +/- {:.2f}, '
                'bias {:>+8.2f} = {:>+6.1f}%)'
                ''.format(g[f'{run}_dmdt_Gt'], g.get('hug_reg_dmdt_Gt',
                                                     np.nan),
                          g.get('hug_reg_err_dmdt_Gt', np.nan),
                          g.get(f'bias_{run}_vs_hug_reg_dmdt_Gt', np.nan),
                          g.get(f'bias_{run}_vs_hug_reg_dmdt_Gt', np.nan) /
                          abs(g.get('hug_reg_dmdt_Gt', np.nan)) * 100))
            add('        dmdtda : {:>9.3f} m w.e. yr-1  (obs {:>6.3f} +/- '
                '{:.3f}, bias {:>+7.3f})'
                ''.format(g[f'{run}_dmdtda'], g.get('hug_reg_dmdtda', np.nan),
                          g.get('hug_reg_err_dmdtda', np.nan),
                          g.get(f'bias_{run}_vs_hug_reg_dmdtda', np.nan)))
            bcol = f'bias_{run}_vs_hug_pergla_dmdtda'
            if bcol in geo:
                per_reg = geo.drop(index=GLOBAL)[bcol]
                w = geo.drop(index=GLOBAL)['rgi_area_km2']
                ok = per_reg.notnull() & (w > 0)
                if ok.any():
                    add('        bias vs the per glacier observations: '
                        '{:>+7.3f} m w.e. yr-1 globally,'
                        ''.format(g.get(bcol, np.nan)))
                    add('            area weighted mean absolute regional '
                        'bias {:.3f} m w.e. yr-1'
                        ''.format(np.average(np.abs(per_reg[ok]),
                                             weights=w[ok])))
            wcol = f'within_err_{run}'
            if wcol in geo:
                sel = geo.drop(index=GLOBAL)[wcol]
                bad = list(sel[~sel.astype(bool)].index)
                add('        regions outside the observed error bar: {} of {}'
                    '{}'.format(len(bad), len(sel),
                                (' (' + ', '.join(bad) + ')') if bad else ''))
        add('')
        cols = ([f'{r}_dmdtda' for r in prun.runs] +
                ['hug_pergla_dmdtda', 'hug_reg_dmdtda', 'hug_reg_err_dmdtda',
                 'hug_reg_dmdtda_published'] +
                [f'bias_{r}_vs_hug_pergla_dmdtda' for r in prun.runs] +
                [f'within_err_{r}' for r in prun.runs])
        add(_table_str(geo, cols, float_format='{:.3f}'))
        add('')
        add('    (`hug_reg_dmdtda_published` is relative to the *measured* '
            'area only and is')
        add('     shown for reference: `hug_reg_dmdtda` is the one to compare '
            'to.)')

    # -- Area vs RGI
    am = tables.get('area_match')
    if am is not None and any(f'area_mismatch_{r}_percent' in am
                              for r in prun.runs):
        title('Modelled area at the RGI date vs the RGI inventory')
        add('    The modelled `area_min_h` is taken at the area weighted '
            'median RGI year of')
        add('    the region (the RGI date varies from glacier to glacier). '
            'Two mismatches are')
        add('    given: `area_mismatch_<run>_percent` is against the '
            'inventory area of *all*')
        add('    the glaciers of the region, `area_mismatch_pop_<run>_percent`'
            ' against the ones')
        add('    which actually ran. The second is the one which says '
            'something about the')
        add('    model - the difference between the two is the glaciers which '
            'are missing.')
        add('')
        cols = (['rgi_area_km2', 'rgi_area_population_km2', 'rgi_year_wmedian',
                 'rgi_year_min', 'rgi_year_max'] +
                [f'area_min_h_at_rgi_yr_{r}_km2' for r in prun.runs] +
                [f'area_mismatch_{r}_percent' for r in prun.runs] +
                [f'area_mismatch_pop_{r}_percent' for r in prun.runs])
        add(_table_str(am, cols, float_format='{:.2f}'))

    # -- Time series
    ts = tables.get('timeseries')
    if ts is not None:
        title('Volume and area evolution')
        rows = OrderedDict()
        for reg in list(prun.regions) + [GLOBAL]:
            d = {}
            for run in prun.runs:
                sel = ts.loc[(ts['region'] == reg) & (ts['run'] == run) &
                             ts['is_common_period']]
                if len(sel) == 0:
                    continue
                sel = sel.set_index('year')
                y0, y1 = sel.index[0], sel.index[-1]
                d[f'{run}_start_yr'] = y0
                d[f'{run}_vol_start_km3'] = sel.loc[y0, 'volume_km3']
                d[f'{run}_vol_end_km3'] = sel.loc[y1, 'volume_km3']
                d[f'{run}_vol_change_perc'] = (
                    (sel.loc[y1, 'volume_km3'] / sel.loc[y0, 'volume_km3'] - 1)
                    * 100)
                if 2000 in sel.index:
                    d[f'{run}_vol_change_2000_perc'] = (
                        (sel.loc[y1, 'volume_km3'] /
                         sel.loc[2000, 'volume_km3'] - 1) * 100)
                    d[f'{run}_area_change_2000_perc'] = (
                        (sel.loc[y1, 'area_min_h_km2'] /
                         sel.loc[2000, 'area_min_h_km2'] - 1) * 100)
            rows[reg] = d
        cdf = pd.DataFrame.from_dict(rows, orient='index')
        add('    Over the common period of each region (all the glaciers of '
            'the population')
        add('    have data), and since 2000. End year: {}.'
            ''.format(int(ts['year'].max())))
        add('')
        add(_table_str(cdf, float_format='{:.2f}'))

    # -- MB params
    mb = tables.get('mb_params')
    if mb is not None:
        title('Calibrated mass balance parameters')
        g = mb.loc[GLOBAL]
        for p in ['melt_f', 'prcp_fac', 'temp_bias', 'bias']:
            if f'{p}_mean' not in mb:
                continue
            add('    {:<10s}: mean {:>8.3f} (area weighted {:>8.3f}), median '
                '{:>8.3f}, 5-95% {:>8.3f} to {:>8.3f}'
                ''.format(p, g[f'{p}_mean'], g[f'{p}_wmean'], g[f'{p}_q50'],
                          g[f'{p}_q05'], g[f'{p}_q95']))
        if 'melt_f_perc_at_max' in mb:
            add('    melt_f at the bounds: {:.2f}% of the glaciers at '
                'melt_f_max ({:.2f}% of the'
                ''.format(g['melt_f_perc_at_max'], g['melt_f_perc_area_at_max']))
            add('        area), {:.2f}% at melt_f_min. A large number here '
                'means the calibration'.format(g['melt_f_perc_at_min']))
            add('        could not do its job for those glaciers.')
        if 'melt_f_dyn_change_median' in mb:
            add('    The dynamic melt_f calibration moved melt_f for {:.1f}% '
                'of the glaciers it'.format(g['melt_f_dyn_change_perc_moved']))
            add('        ran on, by {:+.3f} (median, IQR {:+.3f} to {:+.3f}).'
                ''.format(g['melt_f_dyn_change_median'],
                          g['melt_f_dyn_change_q25'],
                          g['melt_f_dyn_change_q75']))
        add('')
        cols = []
        for p in ['melt_f', 'prcp_fac', 'temp_bias', 'bias']:
            cols += [f'{p}_wmean', f'{p}_q50']
        cols += ['melt_f_perc_at_max', 'melt_f_dyn_change_median']
        add(_table_str(mb, cols, float_format='{:.3f}'))

    add('')
    return '\n'.join(lines)


# The colors of the runs in the plots
RUN_COLORS = {'spinup': 'C0', 'fixed_geom': 'C3'}
RUN_STYLES = {'spinup': '-', 'fixed_geom': '--'}


def _reg_label(reg):
    if reg == GLOBAL:
        return 'Global'
    return 'RGI{}: {}'.format(reg, RGI_REGION_NAMES.get(reg, ''))


def _add_rgi_marker(ax, s, ms=11):
    """The RGI inventory area at the median RGI year, on an area plot.

    Two markers: the area of all the glaciers of the region, and the area of
    the ones which actually ran (identical when nothing failed). The model
    curve can only be compared to the second one.
    """
    yr = s.get('rgi_year_wmedian', np.nan)
    if not np.isfinite(yr):
        return
    ax.plot(yr, s['rgi_area_km2'], marker='*', ls='none', mfc='none',
            mec='k', ms=ms, zorder=5, label='RGI area, all the glaciers')
    if 'rgi_area_population_km2' in s:
        ax.plot(yr, s['rgi_area_population_km2'], 'k*', ms=ms, zorder=6,
                label='RGI area, the glaciers which ran')


def _grid_axes(regions, ncols=4, panel_size=(3.6, 2.6)):
    """A grid of subplots, one per region plus one for the global aggregate."""
    import matplotlib.pyplot as plt
    n = len(regions)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, sharex=False,
                            figsize=(panel_size[0] * ncols,
                                     panel_size[1] * nrows))
    axs = np.atleast_1d(axs).flatten()
    for ax in axs[n:]:
        ax.set_visible(False)
    return fig, axs[:n]


def _plot_timeseries_grid(prun, ts, area_match, path, var='volume_km3',
                          ylabel='Volume (km$^3$)', since=None, title=''):
    """One panel per region: the evolution of `var` in each run."""
    import matplotlib.pyplot as plt

    regions = list(prun.regions) + [GLOBAL]
    fig, axs = _grid_axes(regions)
    for ax, reg in zip(axs, regions):
        for run in prun.runs:
            sel = ts.loc[(ts['region'] == reg) & (ts['run'] == run) &
                         ts['is_common_period']]
            if len(sel) == 0:
                continue
            if since is not None:
                sel = sel.loc[sel['year'] >= since]
            ax.plot(sel['year'], sel[var], color=RUN_COLORS.get(run, 'k'),
                    ls=RUN_STYLES.get(run, '-'), lw=1.2,
                    label=_run_label(run))
        if var == 'area_min_h_km2' and area_match is not None and \
                reg in area_match.index:
            _add_rgi_marker(ax, area_match.loc[reg])
        ax.set_title(_reg_label(reg), fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel(ylabel, fontsize=9)
    handles, labels = axs[0].get_legend_handles_labels()
    if len(handles) == 0 and len(axs) > 1:
        handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
               frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_timeseries_normalized(prun, ts, path, var='volume_km3',
                                ylabel='Volume', title=''):
    """All regions on two axes: normalized to the start year, and to 2000."""
    import matplotlib.pyplot as plt

    regions = list(prun.regions) + [GLOBAL]
    cmap = plt.get_cmap('tab20')
    run = prun.runs[0]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for i, (ax, ref) in enumerate(zip(axs, ['start', 2000])):
        for j, reg in enumerate(regions):
            sel = ts.loc[(ts['region'] == reg) & (ts['run'] == run) &
                         ts['is_common_period']].set_index('year')
            if len(sel) == 0:
                continue
            if ref == 'start':
                y0 = sel.index[0]
            else:
                if ref not in sel.index:
                    continue
                y0 = ref
                sel = sel.loc[sel.index >= ref]
            v = sel[var] / sel.loc[y0, var] * 100
            color = 'k' if reg == GLOBAL else cmap(j % 20)
            ax.plot(sel.index, v, color=color, lw=2.5 if reg == GLOBAL else 1.2,
                    label=_reg_label(reg).split(':')[0])
        ax.axhline(100, color='grey', lw=0.8, zorder=0)
        ax.grid(alpha=0.3)
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{ylabel} (% of the {"start year" if ref == "start" else 2000})')
        ax.set_title('% of the common start year of each region'
                     if ref == 'start' else '% of the year 2000', fontsize=10)
    axs[1].legend(ncol=2, fontsize=8, frameon=False, loc='best')
    fig.suptitle(f'{title} ({_run_label(run)})', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_geodetic_bars(prun, geo, path, specific=True, title=''):
    """The comparison to Hugonnet et al., per region.

    `specific=True` gives the specific rate (m w.e. yr-1, the more diagnostic
    of the two), `specific=False` the mass loss rate (Gt yr-1, log scale).
    """
    import matplotlib.pyplot as plt

    suf = 'dmdtda' if specific else 'dmdt_Gt'
    regions = [r for r in list(prun.regions) + [GLOBAL] if r in geo.index]
    x = np.arange(len(regions))
    # Sign: displayed as a positive loss, which is the convention of the
    # published figures. This is a plotting choice only.
    sign = -1
    width = 0.8 / max(len(prun.runs), 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, run in enumerate(prun.runs):
        if f'{run}_{suf}' not in geo:
            continue
        vals = sign * geo.loc[regions, f'{run}_{suf}'].values
        ax.bar(x + (i - (len(prun.runs) - 1) / 2) * width, vals, width * 0.92,
               color=RUN_COLORS.get(run, 'C0'), alpha=0.85, zorder=2,
               label=f'OGGM, {_run_label(run)}')
    if f'hug_pergla_{suf}' in geo:
        ax.plot(x, sign * geo.loc[regions, f'hug_pergla_{suf}'].values, 's',
                color='0.45', ms=6, ls='none', zorder=3,
                label='Hugonnet et al. 2021, per glacier, same glaciers')
    if f'hug_reg_{suf}' in geo:
        err = geo.loc[regions, f'hug_reg_err_{suf}'].values \
            if f'hug_reg_err_{suf}' in geo else None
        ax.errorbar(x, sign * geo.loc[regions, f'hug_reg_{suf}'].values,
                    yerr=err, fmt='o', color='k', ms=5, lw=1.2, capsize=3,
                    zorder=4, label='Hugonnet et al. 2021, regional')
    if not specific:
        ax.set_yscale('log')
        ax.set_ylabel('Mass loss rate (Gt yr$^{-1}$)')
    else:
        ax.axhline(0, color='k', lw=0.8, zorder=1)
        ax.set_ylabel('Specific mass loss rate (m w.e. yr$^{-1}$)')
    ax.set_xticks(x)
    ax.set_xticklabels([r if r != GLOBAL else 'Global' for r in regions],
                       rotation=45, ha='right')
    ax.axvline(len(regions) - 1.5, color='grey', lw=0.8, ls=':')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_completion(prun, comp, spin, path, title=''):
    """Stacked bars of what completed and how the glaciers were initialised."""
    import matplotlib.pyplot as plt

    regions = list(prun.regions) + [GLOBAL]
    x = np.arange(len(regions))
    has_fg = spin is not None and 'perc_area_fixed_geom_at_start' in spin
    nrows = 1 + (spin is not None) + has_fg
    fig, axs = plt.subplots(nrows, 1, figsize=(13, 4 * nrows), squeeze=False)
    axs = axs.flatten()

    ax = axs[0]
    tot = comp.loc[regions, 'rgi_area_km2'].values
    ok = comp.loc[regions, 'area_population_km2'].values if \
        'area_population_km2' in comp else comp.loc[regions,
                                                    'area_ok_stats_km2'].values
    ax.bar(x, ok / tot * 100, color='C2', label='complete', zorder=2)
    ax.bar(x, 100 - ok / tot * 100, bottom=ok / tot * 100, color='C3',
           label='missing (error or incomplete run)', zorder=2)
    ax.set_ylabel('% of the RGI area')
    # Zoom in on what is usually a nearly full bar, but never hide a real
    # problem: the axis only crops when there is nothing below it
    ymin = min(90, np.floor(np.nanmin(ok / tot * 100) / 10) * 10)
    ax.set_ylim(ymin, 100.5)
    ax.set_title('Completion (note the cropped y axis: it starts at '
                 f'{ymin:.0f}%)', fontsize=10)

    if spin is not None:
        ax = axs[1]
        bottom = np.zeros(len(regions))
        cols = [f'perc_area_{c}' for c in SPINUP_CATEGORIES
                if f'perc_area_{c}' in spin]
        for i, c in enumerate(cols):
            v = spin.loc[regions, c].fillna(0).values
            ax.bar(x, v, bottom=bottom, zorder=2, color=plt.get_cmap('tab10')(i),
                   label=c.replace('perc_area_', '').replace('_', ' '))
            bottom += v
        ax.set_ylabel('% of the RGI area')
        ax.set_title('How the glaciers were initialised (outcome of the '
                     'melt_f calibration)', fontsize=10)

    if has_fg:
        # What the series really contains: a shortened dynamic spinup is a
        # success above, but its first years are not dynamic
        ax = axs[2]
        dyn = spin.loc[regions, 'perc_area_spun_up'].values
        hidden = spin.reindex(regions).get(
            'perc_area_fixed_geom_at_start_not_labelled',
            pd.Series(0., index=regions)).fillna(0).values
        rest = (spin.loc[regions, 'perc_area_not_spun_up'].values - hidden)
        ax.bar(x, dyn, color='C0', zorder=2, label='dynamically spun up')
        ax.bar(x, hidden, bottom=dyn, color='C1', zorder=2,
               label='NOT spun up, but reported a success above '
                     '(shortened spinup)')
        ax.bar(x, rest, bottom=dyn + hidden, color='C3', zorder=2,
               label='NOT spun up (no dynamic spinup at all)')
        if 'perc_area_fixed_geom_at_ref_yr' in spin:
            ax.plot(x, 100 - spin.loc[regions,
                                      'perc_area_fixed_geom_at_ref_yr'].values,
                    'k_', ms=14, mew=2, zorder=4,
                    label='still dynamic at the start of the geodetic period')
        ax.set_ylabel('% of the RGI area')
        ax.set_title('Was the geometry actually spun up? (labels + '
                     '`is_fixed_geometry_spinup`)', fontsize=10)

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels([r if r != GLOBAL else 'Global' for r in regions],
                           rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_mb_params(prun, path_hist, path_reg, title=''):
    """The distribution of the calibrated mass balance parameters."""
    import matplotlib.pyplot as plt

    stats = prun.stats
    params = [p for p in ['melt_f', 'prcp_fac', 'temp_bias', 'bias']
              if p in stats and stats[p].notnull().any()]
    if not params:
        return

    fig, axs = plt.subplots(1, len(params), figsize=(4 * len(params), 3.4))
    axs = np.atleast_1d(axs)
    for ax, p in zip(axs, params):
        s = stats[p].dropna()
        ax.hist(s, bins=50, color='C0', alpha=0.85)
        ax.axvline(s.median(), color='k', lw=1,
                   label='median {:.2f}'.format(s.median()))
        if p == 'melt_f':
            for b, n in [('melt_f_min', 'min'), ('melt_f_max', 'max')]:
                if cfg.PARAMS.get(b, None) is not None:
                    ax.axvline(cfg.PARAMS[b], color='C3', ls='--', lw=1,
                               label=f'{n} ({cfg.PARAMS[b]})')
        ax.set_xlabel(p)
        ax.set_ylabel('N glaciers')
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path_hist, dpi=110)
    plt.close(fig)

    # Per region: a boxplot per parameter
    regions = list(prun.regions)
    fig, axs = plt.subplots(len(params), 1, figsize=(13, 2.8 * len(params)),
                            squeeze=False)
    for ax, p in zip(axs.flatten(), params):
        data = [stats.loc[stats['_reg'] == r, p].dropna().values
                for r in regions]
        ax.boxplot(data, tick_labels=regions, showfliers=False,
                   medianprops={'color': 'C3'})
        ax.set_ylabel(p)
        ax.grid(axis='y', alpha=0.3)
    axs.flatten()[-1].set_xlabel('RGI region')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path_reg, dpi=110)
    plt.close(fig)


def _plot_dmdtda_mismatch(prun, path, title=''):
    """The per glacier residuals of the dynamic melt_f calibration."""
    import matplotlib.pyplot as plt

    stats = prun.stats
    cols = ['dmdtda_mismatch_dynamic_calibration',
            'dmdtda_dynamic_calibration_given_error',
            'dmdtda_dynamic_calibration_error_scaling_factor',
            'dmdtda_mismatch_with_initial_melt_f']
    if cols[0] not in stats:
        return

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    for c, label, color in [
            (cols[3], 'before the dynamic calibration', '0.6'),
            (cols[0], 'after the dynamic calibration', 'C0')]:
        if c not in stats:
            continue
        s = stats[c].dropna()
        if len(s):
            ax.hist(s, bins=np.linspace(-2000, 2000, 81), color=color,
                    alpha=0.7, label=f'{label} (median |x| {s.abs().median():.0f})')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('dmdtda mismatch to the observation (kg m$^{-2}$ yr$^{-1}$)')
    ax.set_ylabel('N glaciers')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    ax = axs[1]
    if all(c in stats for c in cols[:3]):
        tol = stats[cols[1]].abs() * stats[cols[2]]
        rel = (stats[cols[0]].abs() / tol).replace([np.inf, -np.inf], np.nan)
        rel = rel.dropna()
        if len(rel):
            ax.hist(np.clip(rel, 0, 5), bins=np.linspace(0, 5, 51),
                    color='C0', alpha=0.85)
            ax.axvline(1, color='C3', lw=1.2,
                       label='tolerance ({:.1f}% inside)'
                             ''.format((rel <= 1).mean() * 100))
            ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel('|mismatch| / tolerance (clipped at 5)')
    ax.set_ylabel('N glaciers')
    ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_one_region(prun, ts, area_match, geo, reg, path):
    """The detailed figure of one region."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, var, ylabel in [(axs[0], 'volume_km3', 'Volume (km$^3$)'),
                            (axs[1], 'area_min_h_km2',
                             'area_min_h (km$^2$)')]:
        for run in prun.runs:
            sel = ts.loc[(ts['region'] == reg) & (ts['run'] == run) &
                         ts['is_common_period']]
            if len(sel) == 0:
                continue
            ax.plot(sel['year'], sel[var], color=RUN_COLORS.get(run, 'k'),
                    ls=RUN_STYLES.get(run, '-'), lw=1.4, label=_run_label(run))
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Year')
        ax.grid(alpha=0.3)
    if area_match is not None and reg in area_match.index:
        _add_rgi_marker(axs[1], area_match.loc[reg], ms=13)
    axs[0].legend(fontsize=8, frameon=False)
    axs[1].legend(fontsize=8, frameon=False)

    ax = axs[2]
    if geo is not None and reg in geo.index:
        s = geo.loc[reg]
        labels, vals, colors = [], [], []
        for run in prun.runs:
            if f'{run}_dmdtda' in s:
                labels.append(f'OGGM\n{run}')
                vals.append(-s[f'{run}_dmdtda'])
                colors.append(RUN_COLORS.get(run, 'C0'))
        for ref, lab, col in [('hug_pergla', 'Hugonnet\nper glacier', '0.45'),
                              ('hug_reg', 'Hugonnet\nregional', 'k')]:
            if f'{ref}_dmdtda' in s and np.isfinite(s[f'{ref}_dmdtda']):
                labels.append(lab)
                vals.append(-s[f'{ref}_dmdtda'])
                colors.append(col)
        err = [0] * len(vals)
        if 'hug_reg_err_dmdtda' in s and labels and \
                labels[-1].startswith('Hugonnet\nreg'):
            err[-1] = s['hug_reg_err_dmdtda']
        ax.bar(np.arange(len(vals)), vals, yerr=err, color=colors, alpha=0.85,
               capsize=4)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Specific mass loss rate (m w.e. yr$^{-1}$)')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('{} - {}'.format(prun.name, _reg_label(reg)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compile_prepro_diagnostics(input_dir, output_dir, rgi_region=None,
                               reference_period=None, make_plots=True,
                               per_region_plots=True, name=None):
    """Diagnoses a preprocessing run and writes a report, tables and plots.

    This is what the ``oggm_prepro_diag`` command does. It reads the summary
    files of an `oggm_prepro` run (see :py:func:`read_prepro_run`) and writes,
    in `output_dir`:

    - `report.txt`, the diagnostic report. Read this one first.
    - `tables/*.csv`, all the numbers of the report (and more).
    - `plots/*.png` and `plots/per_region/*.png`.

    Parameters
    ----------
    input_dir : str or Path
        the directory of the run to diagnose (see :py:func:`read_prepro_run`).
    output_dir : str or Path
        where to write the report, the tables and the plots. Created if
        needed.
    rgi_region : str or list, optional
        diagnose these RGI regions only (e.g. '11').
    reference_period : str, optional
        the geodetic period to compare the mass balance over, e.g.
        '2000-01-01_2020-01-01'. Defaults to
        `cfg.PARAMS['geodetic_mb_period']`.
    make_plots : bool
        write the plots (the report and the tables are always written).
    per_region_plots : bool
        also write one detailed figure per region (in `plots/per_region`).
    name : str, optional
        a human readable name for the run, used in the report and the plot
        titles.

    Returns
    -------
    a dict of DataFrames - the tables of the report, so that the numbers can
    be used directly.
    """

    output_dir = Path(output_dir).absolute()
    utils.mkdir(output_dir)
    table_dir = output_dir / 'tables'
    utils.mkdir(table_dir)

    if reference_period is None:
        reference_period = cfg.PARAMS['geodetic_mb_period']

    prun = read_prepro_run(input_dir, rgi_region=rgi_region, name=name)

    log.workflow('Computing the diagnostics...')
    tables = OrderedDict()
    tables['completion'] = compute_completion(prun)
    tables['errors'] = compute_errors(prun)
    tables['spinup'] = compute_spinup(prun,
                                      reference_period=reference_period)
    tables['mb_params'] = compute_mb_params(prun)
    tables['rgi_reference'] = compute_rgi_reference(prun)
    tables['timeseries'] = compute_timeseries(prun)
    tables['area_match'] = compute_area_match(prun, ts=tables['timeseries'],
                                              rgi_ref=tables['rgi_reference'])
    tables['geodetic'] = compute_geodetic(prun,
                                          reference_period=reference_period)
    tables = OrderedDict((k, v) for k, v in tables.items() if v is not None)

    for name_, df in tables.items():
        df.to_csv(table_dir / f'{name_}.csv')

    report = prepro_diag_report(prun, tables, reference_period=reference_period)
    with open(output_dir / 'report.txt', 'w') as f:
        f.write(report)
    log.workflow('oggm_prepro_diag report:\n' + report)

    if make_plots:
        _make_all_plots(prun, tables, output_dir,
                        per_region_plots=per_region_plots)

    log.workflow(f'oggm_prepro_diag is done! Output written to: {output_dir} '
                 f'(the report is in {output_dir / "report.txt"})')
    return tables


def _make_all_plots(prun, tables, output_dir, per_region_plots=True):
    """All the plots of the report. Errors here are logged, not raised."""

    try:
        import matplotlib  # noqa
    except ImportError:
        log.warning('matplotlib is not available - no plots will be written.')
        return

    plot_dir = Path(output_dir) / 'plots'
    utils.mkdir(plot_dir)

    ts = tables.get('timeseries')
    geo = tables.get('geodetic')
    comp = tables.get('completion')
    spin = tables.get('spinup')
    am = tables.get('area_match')
    tt = prun.name

    log.workflow('Plotting...')

    if comp is not None:
        _plot_completion(prun, comp, spin, plot_dir / 'completion_by_region.png',
                         title=f'{tt} - completion and initialisation')
    _plot_mb_params(prun, plot_dir / 'mb_params_hist.png',
                    plot_dir / 'mb_params_by_region.png',
                    title=f'{tt} - calibrated mass balance parameters')
    _plot_dmdtda_mismatch(prun, plot_dir / 'dmdtda_mismatch_hist.png',
                          title=f'{tt} - dynamic melt_f calibration residuals')

    if geo is not None and len(geo) > 0:
        _plot_geodetic_bars(prun, geo, plot_dir / 'dmdtda_by_region.png',
                            specific=True,
                            title=f'{tt} - specific mass loss rate vs '
                                  'Hugonnet et al. (2021)')
        _plot_geodetic_bars(prun, geo, plot_dir / 'mass_loss_by_region.png',
                            specific=False,
                            title=f'{tt} - mass loss rate vs '
                                  'Hugonnet et al. (2021)')

    if ts is not None:
        for var, ylabel, stem in [
                ('volume_km3', 'Volume (km$^3$)', 'volume'),
                ('area_min_h_km2', 'area_min_h (km$^2$)', 'area_min_h')]:
            _plot_timeseries_grid(
                prun, ts, am, plot_dir / f'{stem}_by_region.png', var=var,
                ylabel=ylabel,
                title=f'{tt} - {stem} since the common start year')
            _plot_timeseries_grid(
                prun, ts, am, plot_dir / f'{stem}_by_region_from2000.png',
                var=var, ylabel=ylabel, since=2000,
                title=f'{tt} - {stem} since 2000')
            _plot_timeseries_normalized(
                prun, ts, plot_dir / f'{stem}_norm_all_regions.png', var=var,
                ylabel=stem, title=f'{tt} - normalized {stem}')

        if per_region_plots:
            reg_dir = plot_dir / 'per_region'
            utils.mkdir(reg_dir)
            for reg in prun.regions:
                _plot_one_region(prun, ts, am, geo, reg,
                                 reg_dir / f'RGI{reg}.png')
