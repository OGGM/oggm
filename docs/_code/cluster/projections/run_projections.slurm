#!/bin/bash
#
#SBATCH --job-name=projections
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#
# One chunk of one RGI region. Submitted as an array by submit_projections.sh:
#
#   sbatch --array=0-<n_chunks - 1> run_projections.slurm <rgi_reg>
#
# The array task id is the chunk index, read by run_projections.py.

set -e

# A fast, node-local directory, unique to this job. Adapt to your cluster.
JOB_DIR="/tmp/$USER/$SLURM_JOB_ID"
export OGGM_WORKDIR="$JOB_DIR/wd"
export OGGM_OUTDIR="$JOB_DIR/out"
mkdir -p "$OGGM_WORKDIR" "$OGGM_OUTDIR"

# Where all the chunks gather their output (on shared storage)
FINAL_DIR="$SLURM_SUBMIT_DIR/output"

CONTAINER=/path/to/oggm_container.sif
OGGM_COMMIT=master       # better: a commit hash or a tag

srun -n 1 -c "${SLURM_JOB_CPUS_PER_NODE}" singularity exec "$CONTAINER" bash -s <<EOF
  set -e
  export HOME="$OGGM_WORKDIR/fake_home"
  mkdir -p "\$HOME"
  python3 -m venv --system-site-packages "$OGGM_WORKDIR/oggm_env"
  source "$OGGM_WORKDIR/oggm_env/bin/activate"
  pip install --no-deps "git+https://github.com/OGGM/oggm.git@$OGGM_COMMIT"
  ulimit -n 65000
  python "$SLURM_SUBMIT_DIR/run_projections.py" run $1
EOF

# Each chunk writes differently named files, so they can all be copied into
# the same folder
mkdir -p "$FINAL_DIR"
rsync -a "$OGGM_OUTDIR/" "$FINAL_DIR/"
rm -rf "$JOB_DIR"

echo "SLURM DONE"
