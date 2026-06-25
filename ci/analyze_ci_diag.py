#!/usr/bin/env python3
"""TEMPORARY CI diagnostic (issue #1906). Analyse sampler + timeline artifacts.

Given an artifact dir containing test_timeline.csv and resource_samples.csv:
  1. compute per-test wall-duration from the timeline,
  2. surface the tests far above a baseline (>FACTOR x --baseline, or top N),
  3. for each, print the resource-sample window spanning that test.

Stdlib only. Usage:
    python3 ci/analyze_ci_diag.py DIAG_DIR [--baseline SEC] [--factor X] [--top N]

DIAG_DIR may be a single artifact dir or a parent holding several ci-diag-* dirs
(each is analysed separately). Revert: delete this file (PR #1906).
"""
import argparse
import csv
import datetime as dt
import glob
import os
import sys


def _parse_ts(s):
    return dt.datetime.fromisoformat(s)


def load_timeline(path):
    """Return list of (nodeid, start_ts, finish_ts, duration_s)."""
    starts = {}
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            ts = _parse_ts(r["ts"])
            nodeid = r["nodeid"]
            if r["phase"] == "start":
                starts[nodeid] = (ts, r)
            elif r["phase"] == "finish" and nodeid in starts:
                start_ts, _ = starts.pop(nodeid)
                rows.append((nodeid, start_ts, ts, (ts - start_ts).total_seconds()))
    return rows


def load_samples(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_window(samples, start_ts, finish_ts, pad=10.0):
    """Print sample rows inside [start-pad, finish+pad]."""
    cols = ["ts", "mem_available_kb", "swap_free_kb", "cpu_steal_pct",
            "cpu_idle_pct", "psi_mem_some_avg10", "psi_mem_full_avg10",
            "psi_cpu_some_avg10", "psi_io_some_avg10", "cg_mem_current",
            "cg_mem_high_evt", "cg_mem_oom_evt"]
    lo = start_ts - dt.timedelta(seconds=pad)
    hi = finish_ts + dt.timedelta(seconds=pad)
    win = [s for s in samples if s.get("ts") and lo <= _parse_ts(s["ts"]) <= hi]
    if not win:
        print("    (no resource samples in window)")
        return
    print("    " + "  ".join(c for c in cols))
    for s in win:
        print("    " + "  ".join(str(s.get(c, "")) for c in cols))
    # tiny summary of the discriminating signals
    def peak(col):
        vals = [_f(s.get(col)) for s in win]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    def lo_(col):
        vals = [_f(s.get(col)) for s in win]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None
    print(f"    >> peak psi_mem_full_avg10={peak('psi_mem_full_avg10')} "
          f"peak cpu_steal_pct={peak('cpu_steal_pct')} "
          f"min mem_available_kb={lo_('mem_available_kb')} "
          f"min swap_free_kb={lo_('swap_free_kb')}")


def analyse_dir(d, baseline, factor, top):
    tl_path = os.path.join(d, "test_timeline.csv")
    sm_path = os.path.join(d, "resource_samples.csv")
    if not os.path.exists(tl_path):
        print(f"[skip] no test_timeline.csv in {d}")
        return
    rows = load_timeline(tl_path)
    samples = load_samples(sm_path) if os.path.exists(sm_path) else []
    rows.sort(key=lambda r: r[3], reverse=True)

    print(f"\n=== {d} ===")
    print(f"{len(rows)} tests, {len(samples)} resource samples")
    if baseline:
        flagged = [r for r in rows if r[3] > factor * baseline]
        print(f"tests > {factor}x baseline ({baseline}s): {len(flagged)}")
    else:
        flagged = rows[:top]
        print(f"top {top} slowest tests:")

    for nodeid, start_ts, finish_ts, dur in flagged:
        print(f"\n  {dur:8.1f}s  {nodeid}")
        print(f"    {start_ts.isoformat()} -> {finish_ts.isoformat()}")
        if samples:
            _fmt_window(samples, start_ts, finish_ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diag_dir")
    ap.add_argument("--baseline", type=float, default=None,
                    help="local reference duration (s); flag tests > factor x this")
    ap.add_argument("--factor", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    subdirs = sorted(glob.glob(os.path.join(args.diag_dir, "ci-diag-*")))
    targets = subdirs if subdirs else [args.diag_dir]
    for d in targets:
        analyse_dir(d, args.baseline, args.factor, args.top)


if __name__ == "__main__":
    sys.exit(main())
