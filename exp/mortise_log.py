#!/usr/bin/env python
"""Mortise's state in a WebRTC run (tigas_sender --enable_mortise, sparkrtc
modules/remote_bitrate_estimator/tradeoff_control.cc): the TigasMortise
lines of sender.log, one per period of about 1 s and one per change point,

    TigasMortise utc_ms=.. at_ms=.. branch=.. u_before=.. u=.. u_last=..
      beta=.. alpha=.. ai=.. lambda=.. since_decrease_ms=.. last_bitrate=..
      [bitrate=.. tput=.. rtt=.. q=.. q_last=.. u_min=..]

branch: rule_up, rule_down, rule_hold (the rule on Q), cp_enter (no decrease
for 3 s: u reset), cp_decay (u / 1.81 while still no decrease), cp_exit (a
decrease: back to the rule), skip_no_target, skip_no_sample. The bracketed
fields are the period's means (bit/s, ms) and are missing on cp_enter and
skips. u holds from its line to the next.

Per run (summary), over the playback: the time mean of u, alpha, beta and
ai, the share of time with u <= 0.5 and with u >= 0.9, change points and
changes of u per minute, the share of periods per branch kind and the
median R. Before the first line GCC has its defaults and u is nan.

  python exp/mortise_log.py RUN_DIR ...

prints the summary of each run.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

LINE = re.compile(r"TigasMortise (utc_ms=.*branch=.*)$")
FIELD = re.compile(r"(\w+)=(\S+)")


def read_log(path):
    """The TigasMortise lines of sender.log, in order."""
    rows = []
    for line in open(path, errors="replace"):
        m = LINE.search(line.rstrip("\n"))
        if not m:
            continue
        r = dict(FIELD.findall(m.group(1)))
        rows.append({k: v if k == "branch" else float(v) for k, v in r.items()})
    return pd.DataFrame(rows)


def series(d, run):
    """read_log with t: s of playback time (as compare_dips.target_series);
    empty without sender.log."""
    path = os.path.join(d, "sender.log")
    log = read_log(path) if os.path.exists(path) else pd.DataFrame()
    if len(log):
        log["t"] = (log["utc_ms"] - run["t0"] * 1e3 - run["start_ms"]) / 1e3
    return log


def step_mean(t, v, lo, hi):
    """Time mean over [lo, hi] of the step function v from t on, and the
    covered time (nan before the first t)."""
    if not len(t):
        return np.nan, 0.0
    edges = np.clip(np.append(t, hi), lo, hi)
    w = np.diff(edges)
    ok = np.isfinite(v) & (w > 0)
    return (np.sum(v[ok] * w[ok]) / w[ok].sum() if w[ok].sum() > 0 else np.nan), w[ok].sum()


def summary(d, run):
    """The per-run measures (module docstring); {} without TigasMortise lines."""
    log = series(d, run)
    if not len(log):
        return {}
    hi = (run["end_ms"] - run["start_ms"]) / 1e3
    t, u = log["t"].to_numpy(), log["u"].to_numpy()
    x = {}
    for k in ("u", "alpha", "beta", "ai"):
        x[f"mortise_{k}"] = step_mean(t, log[k].to_numpy(), 0, hi)[0]
    _, cover = step_mean(t, u, 0, hi)
    for k, m in (("u_le05", u <= 0.5), ("u_ge09", u >= 0.9)):
        x[f"mortise_{k}"] = step_mean(t, m.astype(float), 0, hi)[0]
    play = log[(log["t"] >= 0) & (log["t"] < hi)]
    minutes = hi / 60
    x["mortise_cp_min"] = (play["branch"] == "cp_enter").sum() / minutes
    x["mortise_changes_min"] = (play["u"] != play["u_before"]).sum() / minutes
    periods = play[play["branch"] != "cp_enter"]
    kind = periods["branch"].str.split("_").str[0]
    for k in ("rule", "cp", "skip"):
        x[f"mortise_{k}_share"] = (kind == k).mean() if len(periods) else np.nan
    x["mortise_rtt_ms"] = periods["rtt"].median() if "rtt" in periods else np.nan
    x["mortise_covered"] = cover / hi
    return x


def main():
    for d in sys.argv[1:]:
        run = json.load(open(os.path.join(d, "run.json")))
        print(d)
        for k, v in summary(d, run).items():
            print(f"  {k} {v:.3f}")


if __name__ == "__main__":
    main()
