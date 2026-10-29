"""Projections for one chunk of an RGI region, from preprocessed directories.

Usage::

    python run_projections.py count <rgi_reg>   # prints the number of chunks
    python run_projections.py run <rgi_reg>     # runs chunk $SLURM_ARRAY_TASK_ID

Each chunk writes its own compiled output files, named after the chunk. Put
them together afterwards, for example with::

    ds = xr.open_mfdataset('RGI11/run_output_*_chunk*.nc',
                           combine='nested', concat_dim='rgi_id')

Note that all side effects are inside `main()`: OGGM's multiprocessing
workers import this file, and must not run it again.
"""
import os
import sys
import logging

import geopandas as gpd

from oggm import cfg, utils, workflow, tasks
from oggm.cli.prepro_levels import apply_rgi_fixes
from oggm.shop.gcm_climate import process_monthly_isimip_data

log = logging.getLogger(__name__)

RGI_VERSION = '62'

# Unlike for the preprocessing, any chunk size works here: we read
# preprocessed directories but write no tar files. Choose it so that one
# chunk comfortably fits in your job time limit.
CHUNK_SIZE = 250

# The preprocessed directories to start from (L5 are the smallest ones)
PREPRO_BASE_URL = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/'
                   'L3-L5_files/2025.6/elev_bands/W5E5/per_glacier_spinup/')
PREPRO_LEVEL = 5
BORDER = 160

# The climate projections to run: here, the ISIMIP3b data which OGGM
# downloads for you
MEMBERS = ['mri-esm2-0_r1i1p1f1', 'gfdl-esm4_r1i1p1f1']
SSPS = ['ssp126', 'ssp585']
END_YEAR = 2101


def region_glaciers(rgi_reg):
    """All glaciers of the region, in the same order for every job.

    apply_rgi_fixes is what oggm_prepro applies before running: for RGI62
    it drops the Greenland connectivity level 2 glaciers, which therefore
    have no preprocessed directory. The chunks must be computed after it.
    """
    rgidf = gpd.read_file(utils.get_rgi_region_file(rgi_reg,
                                                    version=RGI_VERSION))
    return apply_rgi_fixes(rgidf, RGI_VERSION, rgi_reg)


def main():

    action, rgi_reg = sys.argv[1], '{:02d}'.format(int(sys.argv[2]))

    cfg.initialize(logging_level='WORKFLOW')

    if action == 'count':
        # Same function, same glaciers as the jobs: this can not disagree
        # with the chunk selection below
        print(workflow.count_rgi_chunks(region_glaciers(rgi_reg),
                                        chunk_size=CHUNK_SIZE))
        return

    # Array task ids are chunk indices: the array has to start at 0
    chunk_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    rgidf = workflow.get_rgi_chunk(region_glaciers(rgi_reg), chunk_idx,
                                   chunk_size=CHUNK_SIZE)
    if len(rgidf) == 0:
        # The RGI ids have gaps, so a chunk can be empty. Exit without error:
        # a failed array task could cancel the jobs depending on it.
        log.workflow(f'Chunk {chunk_idx} is empty, nothing to do.')
        return

    cfg.PATHS['working_dir'] = utils.mkdir(os.environ['OGGM_WORKDIR'])
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['border'] = BORDER

    gdirs = workflow.init_glacier_directories(
        rgidf, from_prepro_level=PREPRO_LEVEL, prepro_border=BORDER,
        prepro_base_url=PREPRO_BASE_URL, prepro_rgi_version=RGI_VERSION)

    out_dir = utils.mkdir(os.path.join(os.environ['OGGM_OUTDIR'],
                                       f'RGI{rgi_reg}'))

    for member in MEMBERS:
        for ssp in SSPS:
            rid = f'_{member}_{ssp}'
            workflow.execute_entity_task(process_monthly_isimip_data, gdirs,
                                         member=member, ssp=ssp,
                                         output_filesuffix=rid)
            # Start from the end of the historical run of the preprocessing
            workflow.execute_entity_task(
                tasks.run_from_climate_data, gdirs,
                climate_filename='gcm_data', climate_input_filesuffix=rid,
                init_model_filesuffix='_spinup_historical',
                ye=END_YEAR, output_filesuffix=rid)
            # One output file per chunk
            utils.compile_run_output(
                gdirs, input_filesuffix=rid,
                path=os.path.join(out_dir, f'run_output{rid}_'
                                           f'chunk{chunk_idx:03d}.nc'))


if __name__ == '__main__':
    main()
