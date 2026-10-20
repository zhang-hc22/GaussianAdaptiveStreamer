#!/usr/bin/env python
"""Draws the quality-latency tradeoff figures (compare_tms.fig_tradeoff,
against the mean, 90th and 99th pct. latency) from the runs.csv a
comparison wrote, without its raw results:

  tms   exp/compare_tms.py         QUIC Latency, L2A, LoL+ and WebRTC, own frames
  push  exp/compare_push.py        QUIC and WebRTC push
  dips  exp/compare_dips.py        the five push groups under deep dips
  drop  ~/tigas-data/drop_compare.py  old, base, mae, afd, drop

  python exp/tradeoff.py tms|push|dips|drop RUNS_CSV [--root RESULTS] [--out DIR]

--root: recompute the 90th and 99th pct. latency of each run from the
frames.ndjson under RESULTS (a run's results directory is looked up there by
its part after "results/"), the way its comparison measures latency, and
write them into RUNS_CSV. Only run.json and frames.ndjson are needed.
--out defaults to the directory of RUNS_CSV; the figures are tradeoff.pdf,
tradeoff_p90.pdf and tradeoff_p99.pdf (and PNGs).
"""
import argparse
import os

import numpy as np
import pandas as pd


def setups():
    import compare_dips as d
    import compare_push as p
    import compare_tms as t
    drop = ["old", "base", "mae", "afd", "drop"]
    return {
        "tms": dict(marker={"webrtc-tms": "D"}),
        "push": dict(traces=p.TRACES, groups=p.SCHEMES, label=p.LABEL, color=p.COLOR, marker={"webrtc-push": "D"},
                     quality=("psnr", "ssim", "vmaf"), names=p.METRICS),
        "dips": dict(traces=d.TRACES, groups=d.GROUPS, key="group", label=d.LABEL, color=d.COLOR, marker=d.MARKER,
                     quality=("psnr", "ssim", "vmaf"), names=d.METRICS),
        # colours of drop_compare.py
        "drop": dict(traces={"bus_x8": "bus ×8", "bus_x1": "bus ×1"}, groups=drop, key="group",
                     label={g: g for g in drop},
                     color={"old": "0.6", "base": "k", "mae": "tab:blue", "afd": "tab:orange", "drop": "tab:red"},
                     names=t.METRICS),
    }


def latencies(kind, d):
    """Latency of the viewpoints with a frame, as the comparison has it:
    tms and drop from the due time (compare_tms.qoe), push and dips from
    setting the viewpoint (compare_push.measures)."""
    import json
    run = json.load(open(os.path.join(d, "run.json")))
    if kind in ("tms", "drop"):
        from compare_tms import viewpoints
        vp = viewpoints(d, run)
        own = vp[vp["own"]]
        return own["drawn"] - own["due"]
    from compare_push import ndjson
    vp = ndjson(os.path.join(d, "frames.ndjson"))
    own = vp[vp["drawn"].notna() & (vp["error"].isna() if "error" in vp else True)]
    return own["drawn"] - own["send"]


def tails(runs, kind, root):
    """Adds latency_p90_ms and latency_p99_ms; checks latency_p95_ms."""
    from compare_tms import run_dir
    out = []
    for r in runs.itertuples():
        res = os.path.join(root, r.results.split("/results/", 1)[1])
        lat = latencies(kind, run_dir(res, r))
        out.append((lat.quantile(0.90), lat.quantile(0.95), lat.quantile(0.99)))
    out = np.array(out)
    err = np.nanmax(np.abs(out[:, 1] - runs["latency_p95_ms"].to_numpy()))
    assert err < 0.01, f"95th pct. off by {err:.3f} ms: not the comparison's latency"
    return runs.assign(latency_p90_ms=out[:, 0], latency_p99_ms=out[:, 2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["tms", "push", "dips", "drop"])
    ap.add_argument("runs", help="runs.csv of the comparison")
    ap.add_argument("--root", help="results directory to recompute the tails from")
    ap.add_argument("--out")
    a = ap.parse_args()
    from compare_tms import TRADEOFFS, fig_tradeoff
    runs = pd.read_csv(a.runs)
    if a.root:
        runs = tails(runs, a.kind, os.path.abspath(a.root))
        runs.to_csv(a.runs, index=False)
    out = a.out or os.path.dirname(os.path.abspath(a.runs))
    for lat, suffix in TRADEOFFS.items():
        if lat in runs:
            fig_tradeoff(runs, os.path.join(out, f"tradeoff{suffix}.pdf"), lat=lat, **setups()[a.kind])
            print(f"tradeoff: {os.path.join(out, f'tradeoff{suffix}.pdf')}")


if __name__ == "__main__":
    main()
