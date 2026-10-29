#!/bin/bash
# Submit the chunked preprocessing run described in config.sh.
#
# The levels are covered in stages, alternating between many small chunk jobs
# (a SLURM array) and one job for the whole region:
#
#   2  -> 3a   chunked   climate, mass balance calibration, apparent mb
#   3a -> 3    region    Glen A calibration, inversion, L3 summaries
#   3  -> 4a   chunked   the historical and dynamic spinup runs
#   4a -> 4/5  region    L4 summaries (and the L5 directories)
#
# Each stage waits for the previous one to succeed. The regions are
# independent, so the small ones finish and free their nodes while the big
# ones are still running.
#
#   ./submit_prepro.sh               # all RGI regions
#   ./submit_prepro.sh 1 13          # only these
#   ./submit_prepro.sh --test        # smoke test of config.sh, see below
#   ./submit_prepro.sh --test 11     # ... on a region of your choice
#
# --test runs the whole chain on 4 glaciers, with chunks of 100 so that the
# chunking is exercised, and writes into test_run/. Everything else is what
# config.sh says: it tests the configuration you are about to run.
#
# Recovering from failed chunks: if one task of an array fails, the rest of
# that region's chain is cancelled (--kill-on-invalid-dep). Find the failed
# tasks, fix the problem, and resubmit only what is needed:
#
#   sacct -j <jobid> -X --format=JobID,State,ExitCode | grep -v COMPLETED
#   STAGE=1 CHUNKS=3,7,12 ./submit_prepro.sh 13   # redo these chunks only
#   STAGE=2 ./submit_prepro.sh 13                 # then carry on, stage by stage
#
# STAGE=n       run only stage n (1-based), without dependency
# CHUNKS=...    an sbatch array spec (3,7,12 or 0-9) instead of all chunks
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"

TEST=""
if [ "$1" = "--test" ]; then
    shift
    TEST=test
    CHUNK_SIZE=100
    CPUS_CHUNK=4  ; TIME_CHUNK=02:00:00
    CPUS_REGION=8 ; TIME_REGION=02:00:00
    LOGDIR=${LOGDIR}_test
    set -- ${@:-6}             # region 06 is the smallest one
fi

REGIONS="$@"
if [ -z "$REGIONS" ]; then REGIONS=$(seq 1 19); fi

pad () { printf "%02d" "$((10#$1))"; }

# The number of chunks of a region follows from the RGI ids, it is not
# something you choose. oggm_prepro_chunks reads it from a table shipped with
# OGGM, so it needs OGGM on the machine you submit from. If that is not the
# case, run `oggm_prepro_chunks --rgi-version <v> --chunk-size <n>` once
# elsewhere and hardcode the numbers here.
n_chunks () {
    oggm_prepro_chunks --rgi-version "$RGI_VERSION" --chunk-size "$CHUNK_SIZE" \
                       --rgi-reg "$1"
}

# ------------------------------------------------- work out the stage chain
# A stage ending at a half level is chunked, all others need the whole region.
# L5 is made by the same whole-region job as the L4 summaries.
if [ "$END_LEVEL" = "5" ]; then CHAIN="2 3a 3 4a 5"; else CHAIN="2 3a 3 4a 4"; fi

STAGES=""
prev=""
for lev in $CHAIN; do
    if [ -n "$prev" ]; then STAGES="$STAGES $prev:$lev"; fi
    if [ "$lev" = "$END_LEVEL" ]; then break; fi
    if [ -n "$prev" ] || [ "$lev" = "$START_LEVEL" ]; then prev=$lev; fi
done
case " $STAGES " in
    *" $START_LEVEL:"*":$END_LEVEL "*|*" $START_LEVEL:$END_LEVEL "*) ;;
    *) echo "config.sh: cannot go from START_LEVEL=$START_LEVEL to" \
            "END_LEVEL=$END_LEVEL along the chain $CHAIN" >&2; exit 1 ;;
esac

# -------------------------------------------------------------- show & check
SRC="${START_URL%/}/RGI$RGI_VERSION/$(printf "b_%03d" "$BORDER")/L$START_LEVEL/"
echo "------------------------------------------------------------------"
[ -n "$TEST" ] && echo "  *** TEST RUN: 4 glaciers, output in test_run/ ***"
echo "  RGI$RGI_VERSION, border $BORDER, chunks of $CHUNK_SIZE, OGGM $OGGM_COMMIT"
echo "  source   $SRC"
echo "  regions  $(echo $REGIONS | tr '\n' ' ')"
i=0
for s in $STAGES; do
    i=$(( i + 1 ))
    case ${s##*:} in *a) kind="chunked" ;; *) kind="region" ;; esac
    echo "  stage $i  ${s%%:*} -> ${s##*:}  $kind"
done
echo "------------------------------------------------------------------"

# Right options, wrong tree: better to find out now than once the jobs run
if ! curl -sfI --max-time 30 "$SRC" >/dev/null; then
    echo "ERROR: $SRC does not exist. Check config.sh." >&2
    exit 1
fi

# -------------------------------------------------------------------- submit
for R in $REGIONS; do
    REG=$(pad $R)
    N=$(n_chunks $REG)
    ARRAY=${CHUNKS:-0-$(( N - 1 ))}
    mkdir -p "$LOGDIR/$REG"

    dep=""
    i=0
    line="region $REG ($N chunks):"
    for s in $STAGES; do
        i=$(( i + 1 ))
        from=${s%%:*}; to=${s##*:}
        if [ -n "$STAGE" ] && [ "$STAGE" != "$i" ]; then continue; fi

        opts="--job-name=${JOB_TAG}_${REG}_s${i} $dep --kill-on-invalid-dep=yes"
        case $to in
            *a) opts="$opts --array=$ARRAY --cpus-per-task=$CPUS_CHUNK"
                opts="$opts --time=$TIME_CHUNK"
                opts="$opts --output=$LOGDIR/$REG/s${i}_${from}-${to}_%A_%a.out" ;;
            *)  opts="$opts --cpus-per-task=$CPUS_REGION --time=$TIME_REGION"
                opts="$opts --output=$LOGDIR/$REG/s${i}_${from}-${to}_%j.out" ;;
        esac
        jid=$(sbatch --parsable $opts "$HERE/run_prepro.sh" \
                     $from $to $REG $CHUNK_SIZE $TEST)
        # afterok on an array waits for (and requires) all of its tasks
        dep="--dependency=afterok:$jid"
        line="$line  s$i=$jid"
    done
    echo "$line"
done
