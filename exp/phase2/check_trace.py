#!/usr/bin/env python
"""Phase 2 acceptance: do the runs inside rattan follow the trace, and do all
runs see the same trace bandwidth at the same t - t0?

Per frame, the transfer rate is (bytes - 1200) * 8 / (responseEnd -
responseStart): the first packet carries the headers, the rest of the frame
arrives at the bottleneck rate unless QUIC holds it back. Only frames of at
least 3 kB (three packets; profiles 0 to 2) are used. The trace bandwidth for a frame is taken at the middle
of its transfer, minus the downlink delay (rattan's downlink applies the
bandwidth, then the delay).

For every run it prints the trace start relative to t0, the median ratio of
transfer rate to trace bandwidth, their correlation (log scale), and the
shift of the trace (-2..2 s) that gives the best correlation, which should be
about 0. It draws one figure per scale.

  python exp/phase2/check_trace.py RUN_DIR ... --fig-dir DIR
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_traces import bw_at, segments  # noqa: E402

MIN_BYTES = 3000


def frame_rates(run_dir):
    run = json.load(open(os.path.join(run_dir, "run.json")))
    r = run["rattan"]
    frames = [json.loads(l) for l in open(os.path.join(run_dir, "frames.ndjson"))]
    pts = []
    for f in frames:
        if "error" in f or f["bytes"] < MIN_BYTES or f["resp_end"] <= f["resp_start"]:
            continue
        mid = (f["resp_start"] + f["resp_end"]) / 2
        tau = (mid - r["trace_start_ms"] - r["rtt_ms"] / 2) / 1e3
        rate = (f["bytes"] - 1200) * 8 / (f["resp_end"] - f["resp_start"]) / 1e3  # Mbit/s
        pts.append((tau, rate, f["profile"]))
    return run, frames, np.array(pts)


def corr_at(segs, pts, shift):
    bw = np.array([bw_at(segs, t + shift) or np.nan for t in pts[:, 0]]) / 1e6
    ok = np.isfinite(bw)
    return np.corrcoef(np.log(pts[ok, 1]), np.log(bw[ok]))[0, 1], bw


def check(run_dir):
    run, frames, pts = frame_rates(run_dir)
    segs = segments(run["rattan"]["trace"])
    c0, bw = corr_at(segs, pts, 0.0)
    shifts = np.arange(-2.0, 2.0001, 0.05)
    cs = [corr_at(segs, pts, s)[0] for s in shifts]
    ratio = pts[:, 1] / bw
    return {
        "run": run_dir, "abr": run["args"]["abr"], "trace": os.path.basename(run["rattan"]["trace"]),
        "frames": len(frames), "used": len(pts),
        "duration_s": (run["end_ms"] - run["start_ms"]) / 1e3,
        "profiles": [run["profiles"].get(str(p), 0) for p in range(4)],
        "trace_start_ms": run["rattan"]["trace_start_ms"],
        "first_packet_from": run["rattan"]["first_packet_from"],
        "busy_drops": run["rattan"]["busy_drops"],
        "ratio_median": float(np.nanmedian(ratio)),
        "ratio_p90": float(np.nanpercentile(ratio, 90)),
        "corr": float(c0), "best_shift_s": float(shifts[int(np.nanargmax(cs))]),
        "pts": pts, "segs": segs, "start_s": (run["start_ms"] - run["rattan"]["trace_start_ms"]) / 1e3,
        "end_s": (run["end_ms"] - run["rattan"]["trace_start_ms"]) / 1e3,
    }


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 2.4 * len(results)), sharex=True, squeeze=False)
    colors = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]
    for ax, r in zip(axes[:, 0], results):
        t = np.array([s[0] for s in r["segs"]] + [r["segs"][-1][1]])
        bw = np.array([s[2] for s in r["segs"]] + [r["segs"][-1][2]]) / 1e6
        keep = t <= r["end_s"] + 1
        ax.step(t[keep], bw[keep], where="post", color="black", lw=0.8)
        for p in range(4):
            m = r["pts"][:, 2] == p
            ax.scatter(r["pts"][m, 0], r["pts"][m, 1], s=2, color=colors[p])
        ax.set_ylabel(f"{r['abr']}\nMbit/s")
        ax.set_xlim(r["start_s"] - 1, r["end_s"] + 1)
    axes[-1, 0].set_xlabel("trace time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fig-dir")
    a = ap.parse_args()
    res = [check(r) for r in a.runs]
    print("run                 trace       time s  frames  p0/p1/p2/p3            start-t0 ms  "
          "rate/trace med p90  corr  best shift s  busy")
    for r in res:
        name = "/".join(r["run"].rstrip("/").split("/")[-2:])
        print(f"{name:18}  {r['trace']:10}  {r['duration_s']:6.1f}  {r['frames']:6}  "
              f"{'/'.join(map(str, r['profiles'])):21}  {r['trace_start_ms']:+10.1f}  "
              f"{r['ratio_median']:8.2f} {r['ratio_p90']:5.2f}  {r['corr']:4.2f}  {r['best_shift_s']:+11.2f}  "
              f"{r['busy_drops']:4}  ({r['used']} frames used, first packet {r['first_packet_from']})")
    starts = [r["trace_start_ms"] for r in res]
    print(f"trace start - t0 over all runs: {min(starts):+.1f} .. {max(starts):+.1f} ms")
    if a.fig_dir:
        by_trace = {}
        for r in res:
            by_trace.setdefault(r["trace"], []).append(r)
        for trace, rs in by_trace.items():
            path = os.path.join(a.fig_dir, f"follow_{os.path.splitext(trace)[0]}.png")
            figure(rs, path)
            print(f"figure: {path}")


if __name__ == "__main__":
    main()
