#!/usr/bin/env python3
"""TEMPORARY CI diagnostic (issue #1906). Stdlib-only resource sampler.

Appends one CSV row every SAMPLER_INTERVAL seconds with the memory / swap /
CPU-steal / PSI signals needed to tell *memory pressure* apart from *CPU steal*
on the GitHub Actions runners. No third-party deps so it runs in any container.

Revert: delete this file + the workflow steps that launch it (see PR #1906).

cgroup support: prefers v2 (unified hierarchy under /sys/fs/cgroup/*). If v2 is
not present it falls back to the classic v1 layout. PSI is read from
/proc/pressure (kernel-wide) with a cgroup-scoped fallback.
"""
import csv
import os
import time
import datetime

INTERVAL = float(os.environ.get("SAMPLER_INTERVAL", "5"))
OUT = os.environ.get("SAMPLER_OUT", "ci_resource_samples.csv")

# cgroup v2 if the unified controllers file exists, else assume v1.
CGROUP_V2 = os.path.exists("/sys/fs/cgroup/cgroup.controllers")


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _first(path):
    txt = _read(path)
    return txt.splitlines()[0] if txt else ""


def meminfo():
    d = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, rest = line.partition(":")
        d[k.strip()] = rest.strip()

    def kb(k):
        return int(d[k].split()[0]) if d.get(k) else None

    return kb("MemAvailable"), kb("SwapFree"), kb("SwapTotal")


def cpu():  # (idle+iowait, steal, total) jiffies
    p = _first("/proc/stat").split()
    if not p or p[0] != "cpu":
        return None
    v = list(map(int, p[1:]))
    return v[3] + (v[4] if len(v) > 4 else 0), (v[7] if len(v) > 7 else 0), sum(v)


def psi(*paths):
    for p in paths:
        txt = _read(p)
        if not txt:
            continue
        out = {}
        for line in txt.splitlines():
            tag = line.split(" ", 1)[0]
            for kv in line.split()[1:]:
                k, _, val = kv.partition("=")
                out[f"{tag}_{k}"] = val
        return out
    return {}


def cg(*paths):
    """First readable value among candidate paths (v2 first, then v1)."""
    for p in paths:
        v = _first(p)
        if v:
            return v
    return None


def cg_events():
    """cgroup v2 memory.events (high / max / oom counters).

    v1 has no equivalent unified file; we fall back to memory.failcnt mapped to
    'high' so a rising count is still visible in the same column.
    """
    d = {}
    txt = _read("/sys/fs/cgroup/memory.events")
    if txt:
        for line in txt.splitlines():
            k, _, v = line.partition(" ")
            d[k] = v.strip()
        return d
    failcnt = _first("/sys/fs/cgroup/memory/memory.failcnt")
    if failcnt:
        d["high"] = failcnt
    return d


FIELDS = ["ts", "mem_available_kb", "swap_free_kb", "swap_total_kb",
          "cpu_steal_pct", "cpu_idle_pct", "loadavg_1m",
          "psi_mem_some_avg10", "psi_mem_full_avg10", "psi_cpu_some_avg10", "psi_io_some_avg10",
          "cg_mem_current", "cg_mem_max", "cg_swap_current",
          "cg_mem_high_evt", "cg_mem_max_evt", "cg_mem_oom_evt"]

prev = cpu()
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    while True:
        time.sleep(INTERVAL)
        cur = cpu()
        steal_pct = idle_pct = None
        if prev and cur:
            di, ds, dt = cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]
            if dt > 0:
                steal_pct = round(100 * ds / dt, 2)
                idle_pct = round(100 * di / dt, 2)
        prev = cur
        ma, sf, st = meminfo()
        pm = psi("/proc/pressure/memory", "/sys/fs/cgroup/memory.pressure")
        pc = psi("/proc/pressure/cpu", "/sys/fs/cgroup/cpu.pressure")
        pi = psi("/proc/pressure/io", "/sys/fs/cgroup/io.pressure")
        ev = cg_events()
        la = _first("/proc/loadavg").split()
        w.writerow({
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mem_available_kb": ma, "swap_free_kb": sf, "swap_total_kb": st,
            "cpu_steal_pct": steal_pct, "cpu_idle_pct": idle_pct,
            "loadavg_1m": la[0] if la else None,
            "psi_mem_some_avg10": pm.get("some_avg10"), "psi_mem_full_avg10": pm.get("full_avg10"),
            "psi_cpu_some_avg10": pc.get("some_avg10"), "psi_io_some_avg10": pi.get("some_avg10"),
            "cg_mem_current": cg("/sys/fs/cgroup/memory.current",
                                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            "cg_mem_max": cg("/sys/fs/cgroup/memory.max",
                             "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            "cg_swap_current": cg("/sys/fs/cgroup/memory.swap.current",
                                  "/sys/fs/cgroup/memory/memory.memsw.usage_in_bytes"),
            "cg_mem_high_evt": ev.get("high"), "cg_mem_max_evt": ev.get("max"),
            "cg_mem_oom_evt": ev.get("oom"),
        })
        fh.flush()
