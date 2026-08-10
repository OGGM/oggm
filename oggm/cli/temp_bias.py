"""Command line arguments to the oggm_temp_bias command

Type `$ oggm_temp_bias -h` for help

This creates the temperature bias prior file used by the `informed_threestep`
mass balance calibration. The full workflow to make one for a new setup is:

1. run the preprocessing with the same options as your target setup, but with
   the `temp_melt` calibration strategy::

     $ oggm_prepro --temp-bias-run --mb-calibration-strategy temp_melt \\
                   <all your other options> --rgi-reg 01
     $ ...  # one job per RGI region

   (use `temp_melt_regional` instead to build the regional flavor of the file)

2. summarize the per-glacier biases of all the regions per climate grid point.
   The grouping of grid points crosses RGI region borders, so this is always a
   separate step, run once over all the regions::

     $ oggm_temp_bias --input RGI62/b_080/L3/summary \\
                      --output-file temp_bias_v2026.1.csv

   Besides the csv, this writes `temp_bias_v2026.1_summary.txt` (what was used,
   what is missing, how much had to be grouped, and the statistics of the final
   values - always read it) and two diagnostic plots.

3. put the resulting csv somewhere your runs can reach it, and use it for the
   real preprocessing::

     $ oggm_prepro --temp-bias-file-path <path or url> <your options>

"""

# Standard libraries
import os
import sys
import argparse
import logging
from pathlib import Path

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm.exceptions import InvalidParamsError

log = logging.getLogger(__name__)


def run_temp_bias(input_files=None, output_file=None, make_plots=True,
                  min_glaciers=12, max_radius=10, err_fill_quantile=0.9,
                  rgi_region=None, rgi_subregion=None,
                  logging_level='WORKFLOW'):
    """Creates the temperature bias prior file out of a `temp_melt` run.

    This is a thin wrapper around
    :py:func:`utils.compute_temp_bias_dataframe`, which does all the work
    (and where all the parameters are documented).

    Parameters
    ----------
    input_files : str or list of str
        path(s) to the `glacier_statistics` files of the `temp_melt` run.
        Directories and glob patterns are accepted as well.
    output_file : str
        path of the csv file to write.
    make_plots : bool
        write the diagnostic plots next to the output file. The diagnostic
        summary (`<output_file stem>_summary.txt`) is always written.
    """

    cfg.initialize(logging_level=logging_level)

    output_file = Path(output_file).absolute()
    utils.mkdir(output_file.parent)

    plot_path = None
    if make_plots:
        plot_path = output_file.parent / output_file.stem

    summary_path = output_file.parent / (output_file.stem + '_summary.txt')

    utils.compute_temp_bias_dataframe(input_files,
                                      min_glaciers=min_glaciers,
                                      max_radius=max_radius,
                                      err_fill_quantile=err_fill_quantile,
                                      rgi_region=rgi_region,
                                      rgi_subregion=rgi_subregion,
                                      path=output_file,
                                      plot_path=plot_path,
                                      summary_path=summary_path)

    log.workflow(f'oggm_temp_bias is done! Output written to: {output_file} '
                 f'(with the diagnostic summary in {summary_path})')


def parse_args(args):
    """Check input arguments and env variables"""

    description = ('Create the OGGM temperature bias prior file out of the '
                   'glacier statistics of a preprocessing run made with the '
                   '`temp_melt` mass balance calibration strategy. The '
                   'resulting file can be fed back into `oggm_prepro` with '
                   '`--temp-bias-file-path`.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', type=str, nargs='+',
                        help='path(s) to the `glacier_statistics_*.csv` files '
                             'of the `temp_melt` run. Can be files, '
                             'directories or glob patterns. Since the grouping '
                             'of grid points crosses RGI region borders, all '
                             'the regions should be given at once.')
    parser.add_argument('--output-file', type=str, default='temp_bias.csv',
                        help='path of the csv file to write (default: '
                             '`temp_bias.csv` in the current directory).')
    parser.add_argument('--no-plots', nargs='?', const=True, default=False,
                        help='do not write the diagnostic plots. The '
                             '`_summary.txt` diagnostic file is always '
                             'written.')
    parser.add_argument('--min-glaciers', type=int, default=12,
                        help='minimum number of glaciers per grid point. Grid '
                             'points with fewer glaciers are grouped with '
                             'their neighbours (default: 12).')
    parser.add_argument('--max-radius', type=int, default=10,
                        help='the maximum search radius (in grid points) used '
                             'for the grouping (default: 10).')
    parser.add_argument('--err-fill-quantile', type=float, default=0.9,
                        help='glaciers with a missing reference mass balance '
                             'error are attributed this quantile of the error '
                             'distribution (default: 0.9).')
    parser.add_argument('--rgi-region', type=str, nargs='+', default=None,
                        help='select only these RGI regions (e.g. 11).')
    parser.add_argument('--rgi-subregion', type=str, nargs='+', default=None,
                        help='select only these RGI subregions (e.g. 11-01).')
    parser.add_argument('--logging-level', type=str, default='WORKFLOW',
                        help='the logging level to use (DEBUG, INFO, WARNING, '
                             'WORKFLOW).')

    args = parser.parse_args(args)

    input_files = args.input
    if not input_files:
        input_files = os.environ.get('OGGM_TEMP_BIAS_INPUT', None)
        if input_files is None:
            raise InvalidParamsError('--input is required!')
        input_files = input_files.split()

    return dict(input_files=input_files,
                output_file=args.output_file,
                make_plots=not args.no_plots,
                min_glaciers=args.min_glaciers,
                max_radius=args.max_radius,
                err_fill_quantile=args.err_fill_quantile,
                rgi_region=args.rgi_region,
                rgi_subregion=args.rgi_subregion,
                logging_level=args.logging_level,
                )


def main():
    """Script entry point"""

    run_temp_bias(**parse_args(sys.argv[1:]))
