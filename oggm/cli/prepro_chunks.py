"""Command line arguments to the oggm_prepro_chunks command

Type `$ oggm_prepro_chunks -h` for help

This tells you how many chunks the glaciers of an RGI region fall into, which
is what you need to size the SLURM array of a chunked preprocessing run::

    $ oggm_prepro_chunks --rgi-reg 13
    15

    $ sbatch --array=0-14 run_prepro.sh

Chunks are blocks of the RGI id space (with `--chunk-size 1000`, chunk 3 holds
the glaciers whose id ends in 03000 to 03999), so their number is a property
of the region rather than something you choose. See
:py:func:`oggm.workflow.get_rgi_chunk` for why it works that way, and note
that some chunks can be small or even empty - an empty chunk job simply logs
that there is nothing to do and exits.

The counts are shipped with OGGM in `oggm/data/rgi_chunks.csv`. They follow
from the RGI ids, so they only ever change when a new RGI version is added.
Rebuild the table with `--recompute`, which reads the RGI files themselves::

    $ python -c "from oggm import cfg; cfg.initialize_minimal(); \
        from oggm.cli.prepro_chunks import compute_rgi_chunks_table as c; \
        c().to_csv('oggm/data/rgi_chunks.csv', index=False)"

or, for one version and chunk size, `oggm_prepro_chunks --recompute --csv`.

This is not covered by the test suite: it needs the full RGI region files,
which are far too large to download in CI (and are blocked by the download
allowlist the tests run with). Rerun it by hand when the RGI files change.
"""

# Standard libraries
import os
import sys
import argparse
import logging

# External modules
import pandas as pd

# Locals
from oggm import cfg, utils, workflow
from oggm.exceptions import InvalidParamsError

log = logging.getLogger(__name__)

# The RGI versions we ship chunk counts for
RGI_VERSIONS = ['62', '70G', '70C']

# The two allowed chunk sizes (see oggm.workflow.get_rgi_chunk)
CHUNK_SIZES = [100, 1000]


def _table_path():
    return os.path.join(os.path.abspath(os.path.dirname(cfg.__file__)),
                        'data', 'rgi_chunks.csv')


def get_rgi_chunks_table():
    """The table of chunk counts shipped with OGGM.

    Returns
    -------
    a dataframe with the rgi_version, rgi_reg, chunk_size and n_chunks columns
    """
    return pd.read_csv(_table_path(),
                       dtype={'rgi_version': str, 'rgi_reg': str})


def compute_rgi_chunks_table(rgi_versions=None, chunk_sizes=None):
    """Work the chunk counts out of the RGI files.

    This is how `oggm/data/rgi_chunks.csv` is made. It needs the RGI region
    files, so it is slow the first time (they are cached afterwards).

    Parameters
    ----------
    rgi_versions : list of str
        which RGI versions to do (defaults to all of them)
    chunk_sizes : list of int
        which chunk sizes to do (defaults to 100 and 1000)

    Returns
    -------
    a dataframe with the rgi_version, rgi_reg, chunk_size and n_chunks columns
    """

    import geopandas as gpd
    from oggm.cli.prepro_levels import apply_rgi_fixes

    if rgi_versions is None:
        rgi_versions = RGI_VERSIONS
    if chunk_sizes is None:
        chunk_sizes = CHUNK_SIZES

    out = []
    for rgi_version in rgi_versions:
        for reg in range(1, 20):
            rgi_reg = '{:02d}'.format(reg)
            rgidf = gpd.read_file(
                utils.get_rgi_region_file(rgi_reg, version=rgi_version))
            # The counts have to match what the preprocessing actually
            # processes, and these fixes drop glaciers in Greenland
            rgidf = apply_rgi_fixes(rgidf, rgi_version, rgi_reg)
            for chunk_size in chunk_sizes:
                out.append({
                    'rgi_version': rgi_version,
                    'rgi_reg': rgi_reg,
                    'chunk_size': chunk_size,
                    'n_chunks': workflow.count_rgi_chunks(
                        rgidf, chunk_size=chunk_size),
                    'n_glaciers': len(rgidf),
                })

    return pd.DataFrame(out)


def run_prepro_chunks(rgi_reg=None, rgi_version='62', chunk_size=1000,
                      recompute=False, verbose=False, to_csv=False):
    """How many chunks does this RGI region fall into?

    Parameters
    ----------
    rgi_reg : str
        the RGI region, or None for all of them
    rgi_version : str
        the RGI version to use
    chunk_size : int
        100 or 1000 (default)
    recompute : bool
        read the RGI files and work the counts out again, instead of using
        the table shipped with OGGM
    verbose : bool
        print the whole table instead of only the number of chunks
    to_csv : bool
        print the table as csv, so that it can be redirected into
        `oggm/data/rgi_chunks.csv` to rebuild it

    Returns
    -------
    the number of chunks (int), or the dataframe if rgi_reg is None
    """

    if chunk_size not in CHUNK_SIZES:
        raise InvalidParamsError('chunk_size must be one of {}, got {}'
                                 ''.format(CHUNK_SIZES, chunk_size))

    if recompute:
        regs = None if rgi_reg is None else [rgi_reg]
        df = compute_rgi_chunks_table(rgi_versions=[rgi_version],
                                      chunk_sizes=[chunk_size])
        if regs is not None:
            df = df.loc[df.rgi_reg.isin(regs)]
    else:
        df = get_rgi_chunks_table()
        df = df.loc[(df.rgi_version == rgi_version) &
                    (df.chunk_size == chunk_size)]

    if rgi_reg is not None:
        df = df.loc[df.rgi_reg == rgi_reg]
        if len(df) == 0:
            raise InvalidParamsError(
                f'No chunk count for RGI version {rgi_version}, region '
                f'{rgi_reg}, chunk size {chunk_size}. Try --recompute.')

    if to_csv:
        print(df.to_csv(index=False), end='')
        return df

    if verbose or rgi_reg is None:
        print(df.to_string(index=False))
        return df

    return int(df.n_chunks.iloc[0])


def parse_args(args):
    """Check input arguments and env variables"""

    parser = argparse.ArgumentParser(
        description='How many chunks does an RGI region fall into? This is '
                    'what you need to size the SLURM array of a chunked '
                    'preprocessing run.')
    parser.add_argument('--rgi-reg', type=str,
                        help='the RGI region to look up. Defaults to '
                             '$OGGM_RGI_REG, or to printing all of them.')
    parser.add_argument('--rgi-version', type=str, default='62',
                        choices=RGI_VERSIONS,
                        help='the RGI version to use (default: 62).')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        choices=CHUNK_SIZES,
                        help='the number of glaciers per chunk '
                             '(default: 1000).')
    parser.add_argument('--recompute', action='store_true',
                        help='read the RGI files and work the counts out '
                             'again, instead of using the table shipped with '
                             'OGGM. Slow, and needs the RGI files.')
    parser.add_argument('--verbose', action='store_true',
                        help='print the whole table instead of only the '
                             'number of chunks.')
    parser.add_argument('--csv', action='store_true',
                        help='print the table as csv, to rebuild '
                             'oggm/data/rgi_chunks.csv.')

    args = parser.parse_args(args)

    rgi_reg = args.rgi_reg
    if not rgi_reg:
        rgi_reg = os.environ.get('OGGM_RGI_REG', None)
    if rgi_reg:
        rgi_reg = '{:02d}'.format(int(rgi_reg))
        if rgi_reg not in ['{:02d}'.format(r) for r in range(1, 20)]:
            raise InvalidParamsError('--rgi-reg should range from 01 to 19!')

    return dict(rgi_reg=rgi_reg, rgi_version=args.rgi_version,
                chunk_size=args.chunk_size, recompute=args.recompute,
                verbose=args.verbose, to_csv=args.csv)


def main():
    """Script entry point"""

    kwargs = parse_args(sys.argv[1:])
    out = run_prepro_chunks(**kwargs)
    if isinstance(out, int):
        print(out)
