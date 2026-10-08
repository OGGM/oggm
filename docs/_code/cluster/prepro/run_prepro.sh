#!/bin/bash
#
#SBATCH --ntasks=1
#
# Run one stage of a chunked preprocessing run, for one RGI region (and one
# chunk if the stage is chunked). Do not run this directly: it is submitted by
# submit_prepro.sh, which sets the job name, the resources and the array.
# Everything about WHAT is run comes from config.sh.
#
#   $1   level to start from   (2, 3a, 3, 4a)
#   $2   level to stop at      (3a, 3, 4a, 4, 5)
#   $3   RGI region
#   $4   chunk size
#   $5   "test" for a smoke test: a few glaciers, written into test_run/

set -e

# The directory we were submitted from must be on storage shared by all
# nodes: this is where the stages hand the glacier directories over to each
# other, and where config.sh lives.
SUBMIT_DIR=$SLURM_SUBMIT_DIR
source "$SUBMIT_DIR/config.sh"

FROM=$1
TO=$2
# 10# forces base 10: printf "%02d" 08 is an "invalid octal number"
RGI_REG=$(printf "%02d" "$((10#$3))")
CHUNK_SIZE=$4
TEST=$5
# The chunk index is the array task id. Arrays have to start at 0!
CHUNK_IDX=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TEST" = "test" ]; then
    SHARED="$SUBMIT_DIR/test_run"
    mkdir -p "$SHARED"
    TEST_ARGS="--test"
else
    SHARED="$SUBMIT_DIR"
    TEST_ARGS=""
fi

echo "RGI$RGI_VERSION region $RGI_REG: L$FROM -> L$TO, chunk $CHUNK_IDX${TEST:+ (TEST)}"

# A fast, node-local directory, unique to this job (every job needs its own
# working directory). Adapt this to your cluster.
JOB_DIR="/tmp/$USER/$SLURM_JOB_ID"
OGGM_WORKDIR="$JOB_DIR/wd"
OGGM_OUTDIR="$JOB_DIR/out"
mkdir -p "$OGGM_WORKDIR" "$OGGM_OUTDIR"

# The first stage reads from the url. The next ones read the directories the
# previous stage left in the shared folder, and keep the url as a fallback
# for the summary files which are only carried forward.
if [ "$FROM" = "$START_LEVEL" ]; then
    SRC_ARGS="--start-base-url $START_URL"
else
    SRC_ARGS="--start-from-dir $SHARED --start-base-url $START_URL"
fi

# A stage ending at a half level (3a, 4a) is per glacier: it runs one chunk.
# A stage ending at a full level needs the whole region.
case $TO in
    *a) CHUNK_ARGS="--chunk-idx $CHUNK_IDX --chunk-size $CHUNK_SIZE" ;;
    *)  CHUNK_ARGS="" ;;
esac

# Everything in the EOF block runs inside the container
srun -n 1 -c "${SLURM_JOB_CPUS_PER_NODE}" singularity exec "$CONTAINER" bash -s <<EOF
  set -e
  export HOME="$OGGM_WORKDIR/fake_home"
  mkdir -p "\$HOME"
  # A venv on top of the container, to install the pinned OGGM version
  python3 -m venv --system-site-packages "$OGGM_WORKDIR/oggm_env"
  source "$OGGM_WORKDIR/oggm_env/bin/activate"
  pip install --no-deps "git+https://github.com/OGGM/oggm.git@$OGGM_COMMIT"
  ulimit -n 65000
  oggm_prepro \
    --working-dir "$OGGM_WORKDIR" --output "$OGGM_OUTDIR" \
    --rgi-reg $RGI_REG --rgi-version $RGI_VERSION --map-border $BORDER \
    --start-level $FROM --max-level $TO \
    $SRC_ARGS $CHUNK_ARGS $TEST_ARGS \
    $PREPRO_OPTS
EOF

# Copy the output to the shared folder. The chunk jobs of a stage all write
# into the same tree at the same time: this is safe, because each chunk writes
# its own tar files and no summary file.
rsync -a "$OGGM_OUTDIR/" "$SHARED/"
rm -rf "$JOB_DIR"

echo "SLURM DONE"
