#!/usr/bin/env python
"""Draws the quality-latency tradeoff figure (compare_tms.fig_tradeoff) from
the runs.csv a comparison wrote, without its raw results:

  tms   exp/compare_tms.py         QUIC Latency, L2A, LoL+ and WebRTC, own frames
  push  exp/compare_push.py        QUIC and WebRTC push
  dips  exp/compare_dips.py        the five push groups under deep dips
  drop  ~/tigas-data/drop_compare.py  old, base, mae, afd, drop

  python exp/tradeoff.py tms|push|dips|drop RUNS_CSV [--out PATH]

--out defaults to tradeoff.pdf next to RUNS_CSV (and a PNG of it).
"""
import argparse
import os

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
                     color={"old": "0.6", "base": "k", "mae": "tab:blue", "afd": "tab:orange", "drop": "tab:red"}),
    }, t.fig_tradeoff


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["tms", "push", "dips", "drop"])
    ap.add_argument("runs", help="runs.csv of the comparison")
    ap.add_argument("--out")
    a = ap.parse_args()
    s, fig_tradeoff = setups()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.runs)), "tradeoff.pdf")
    fig_tradeoff(pd.read_csv(a.runs), out, **s[a.kind])
    print(f"tradeoff: {out}")


if __name__ == "__main__":
    main()
