#!/bin/bash
# Submit the projections of run_projections.py, one array per RGI region.
#
#   ./submit_projections.sh          # all RGI regions
#   ./submit_projections.sh 1 13     # only these
#
# The number of chunks of a region is computed by run_projections.py itself,
# with the same function which selects the chunks, so that the two can not
# disagree. This needs an OGGM environment where you submit from. If you do
# not have one, run `python run_projections.py count <reg>` once elsewhere
# and hardcode the numbers here.
set -e

REGIONS="$@"
if [ -z "$REGIONS" ]; then REGIONS=$(seq 1 19); fi

mkdir -p logs
for R in $REGIONS; do
    REG=$(printf "%02d" "$((10#$R))")
    N=$(python run_projections.py count $REG | tail -n 1)
    # Chunk indices start at 0: so does the array
    sbatch --job-name=proj_$REG --array=0-$(( N - 1 )) \
           run_projections.slurm $REG
done
