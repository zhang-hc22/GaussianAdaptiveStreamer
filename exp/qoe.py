#!/usr/bin/env python
"""The video-call QoE of Mortise (NSDI version, appendix C) for the runs of
exp/compare_push.py and exp/compare_dips.py, one score per run:

    S = W_VIDEO S_video + W_DELAY S_delay + W_DROP S_drop
    S_video = mean VMAF of the viewpoints with their own frame (vmaf)
    S_delay = max(100 - mean latency in ms (latency_ms) / 3, 0)
    S_drop  = 100 (1 - M), M the share of viewpoints without their own
              frame drawn, 1 - shown

All three are 0 to 100. The paper's frame delay is capture to display; here
it is viewpoint set to its first frame drawn.

  python exp/qoe.py RUNS_CSV [--out CSV]

adds qoe, qoe_video, qoe_delay and qoe_drop to the table (in place without
--out) and prints the mean score per group and trace.
"""
import argparse

import numpy as np
import pandas as pd

W_VIDEO, W_DELAY, W_DROP = 0.5, 0.3, 0.2
DELAY_MS_PER_POINT = 3.0
QOE = {"qoe": "QoE", "qoe_video": "S_video (VMAF)", "qoe_delay": "S_delay", "qoe_drop": "S_drop"}


def add_qoe(runs):
    """runs with the four QoE columns (nan where vmaf, latency_ms or shown
    is missing)."""
    runs = runs.copy()
    runs["qoe_video"] = runs["vmaf"]
    runs["qoe_delay"] = np.maximum(100 - runs["latency_ms"] / DELAY_MS_PER_POINT, 0).where(runs["latency_ms"].notna())
    runs["qoe_drop"] = 100 * runs["shown"]
    runs["qoe"] = W_VIDEO * runs["qoe_video"] + W_DELAY * runs["qoe_delay"] + W_DROP * runs["qoe_drop"]
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", help="runs.csv with vmaf, latency_ms and shown")
    ap.add_argument("--out", help="write here instead of in place")
    a = ap.parse_args()
    runs = add_qoe(pd.read_csv(a.runs))
    runs.to_csv(a.out or a.runs, index=False)
    key = "group" if "group" in runs else "abr"
    print(runs.groupby(["trace", key])[list(QOE)].mean().round(1).to_string())


if __name__ == "__main__":
    main()
