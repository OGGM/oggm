# ============================================================================
#  Configuration of a chunked preprocessing run - the only file you edit.
#
#  Both submit_prepro.sh and run_prepro.sh read it, so that the options can
#  not drift apart between the stages of a run. See the OGGM documentation
#  (practicalities -> chunked runs) for what the stages are.
# ============================================================================

# ------------------------------- the science -------------------------------

RGI_VERSION=70G          # 62, 70G or 70C
BORDER=160               # map border in grid points

# The levels this run covers. The stages in between are worked out by
# submit_prepro.sh from the chain  2 -> 3a -> 3 -> 4a -> 4 -> 5
START_LEVEL=2            # the level read from START_URL
END_LEVEL=5              # the last level to produce

# Where to start from: a tree holding RGI{version}/b_{border}/L{START_LEVEL}/
START_URL=https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/2025.6/elev_bands/

# The temperature bias prior needed by the informed_threestep calibration.
# It must match your setup (RGI version, climate, mass balance model).
TEMP_BIAS_FILE=https://your.server/path/to/temp_bias_rgi70g.csv

# Passed to oggm_prepro in EVERY stage. Mind the trailing backslashes.
PREPRO_OPTS="--elev-bands \
             --mb-calibration-strategy informed_threestep \
             --temp-bias-file-path $TEMP_BIAS_FILE \
             --inversion-volume-dataset iceboost \
             --dynamic-spinup area/dmdtda \
             --store-hydro-output"

# ------------------------------ how it runs --------------------------------

CHUNK_SIZE=1000          # 100 or 1000 glaciers per chunk

# Resources of the chunk jobs (many small ones) and of the whole-region jobs
CPUS_CHUNK=32  ; TIME_CHUNK=08:00:00
CPUS_REGION=64 ; TIME_REGION=08:00:00

# The software. Pin the OGGM version, so that all stages run the same code.
CONTAINER=/path/to/oggm_container.sif
OGGM_COMMIT=master       # better: a commit hash or a tag

JOB_TAG=prepro           # prefix of the slurm job names
LOGDIR=logs              # one slurm log per job, in $LOGDIR/<region>/
