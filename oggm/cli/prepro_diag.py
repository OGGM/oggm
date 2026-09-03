"""Command line arguments to the oggm_prepro_diag command

Type `$ oggm_prepro_diag -h` for help

This diagnoses a finished preprocessing run: it reads the summary files
written by `oggm_prepro` and writes a report, the tables it is made of, and a
set of plots::

    $ oggm_prepro_diag --input /path/to/run/RGI62/b_160/L5 \\
                       --output-dir diag_my_run

The report (`diag_my_run/report.txt`) answers, per RGI region and globally:
how much of the run completed, how the dynamic spinup and the dynamic melt_f
calibration went, how the modelled mass loss compares to Hugonnet et al.
(2021), how the modelled area compares to the RGI inventory, and what the
calibrated mass balance parameters look like.

Run it on two runs made with different options and compare the reports to
decide which setup is the better one.

"""

# Standard libraries
import os
import sys
import argparse

# Locals
import oggm.cfg as cfg
from oggm import diagnostics
from oggm.exceptions import InvalidParamsError


def run_prepro_diag(input_dir=None, output_dir=None, rgi_region=None,
                    reference_period=None, make_plots=True,
                    per_region_plots=True, name=None,
                    logging_level='WORKFLOW'):
    """Diagnoses an `oggm_prepro` run and writes a report, tables and plots.

    This is a thin wrapper around
    :py:func:`diagnostics.compile_prepro_diagnostics`, which does all the work
    (and where all the parameters are documented).

    Parameters
    ----------
    input_dir : str
        path to the run to diagnose. This can be the `summary` directory
        itself, the level directory containing it (e.g. `.../L5`), or the
        directory containing the levels (e.g. `.../b_160`), in which case the
        highest level available is used.
    output_dir : str
        where to write the report, the tables and the plots.
    """

    cfg.initialize(logging_level=logging_level)

    diagnostics.compile_prepro_diagnostics(
        input_dir, output_dir,
        rgi_region=rgi_region,
        reference_period=reference_period,
        make_plots=make_plots,
        per_region_plots=per_region_plots,
        name=name,
    )


def parse_args(args):
    """Check input arguments and env variables"""

    description = ('Diagnose an OGGM preprocessing run: read the summary '
                   'files written by `oggm_prepro` and write a report (with '
                   'tables and plots) about what completed, how the dynamic '
                   'spinup went, and how the run compares to the geodetic '
                   'observations and to the RGI inventory.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', type=str,
                        help='path to the run to diagnose: the `summary` '
                             'directory of a preprocessing level (e.g. '
                             '`.../RGI62/b_160/L5/summary`), the level '
                             'directory itself, or the directory containing '
                             'the levels, in which case the highest level '
                             'available is used.')
    parser.add_argument('--output-dir', type=str, default='prepro_diag',
                        help='where to write the report, the tables and the '
                             'plots (default: `prepro_diag` in the current '
                             'directory).')
    parser.add_argument('--rgi-region', type=str, nargs='+', default=None,
                        help='diagnose only these RGI regions (e.g. 11).')
    parser.add_argument('--reference-period', type=str, default=None,
                        help='the geodetic period to compare the mass balance '
                             'over (default: the `geodetic_mb_period` '
                             'parameter, i.e. 2000-01-01_2020-01-01).')
    parser.add_argument('--name', type=str, default=None,
                        help='a name for this run, used in the report and in '
                             'the plot titles (default: the name of the '
                             'directory above `RGI*`).')
    parser.add_argument('--no-plots', action='store_true',
                        help='do not write the plots. The report and the '
                             'tables are always written.')
    parser.add_argument('--no-per-region-plots', action='store_true',
                        help='do not write the detailed figure of each '
                             'region (one file per region).')
    parser.add_argument('--logging-level', type=str, default='WORKFLOW',
                        help='the logging level to use (DEBUG, INFO, WARNING, '
                             'WORKFLOW).')

    args = parser.parse_args(args)

    input_dir = args.input
    if not input_dir:
        input_dir = os.environ.get('OGGM_PREPRO_DIAG_INPUT', None)
        if input_dir is None:
            raise InvalidParamsError('--input is required!')

    return dict(input_dir=input_dir,
                output_dir=args.output_dir,
                rgi_region=args.rgi_region,
                reference_period=args.reference_period,
                make_plots=not args.no_plots,
                per_region_plots=not args.no_per_region_plots,
                name=args.name,
                logging_level=args.logging_level,
                )


def main():
    """Script entry point"""

    run_prepro_diag(**parse_args(sys.argv[1:]))
