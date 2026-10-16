#!/usr/bin/env python
"""Compares the WebTransport push mode (/player-wt, exp/wt.toml) with the
request mode (/player) under the same networks: bus x8, bus x1 and no
bandwidth limit, RTT 20 ms, 1280x720, the same trajectory.

/player sends one viewpoint per request and the next one after the frame is
drawn, so every viewpoint gets exactly one frame. /player-wt sets a new
viewpoint every 40 ms while the server pushes a frame of the latest one 60
times a second; a viewpoint can get several frames or none.

Per run:
- latency: from setting a viewpoint to drawing its (first) frame; per
  viewpoint, viewpoints without a frame left out;
- frames/s: frames drawn per second of playback;
- shown: share of viewpoints with a frame drawn;
- lost (push only): share of the frames the server sent that never
  completed: some chunks missing (the page drops a frame 100 ms after its
  first chunk) or no chunk at all (image ids skipped);
- profile: mean and shares over all frames drawn; switches per second
  between consecutive frames drawn;
- bandwidth used: bytes received per second of playback;
- quality: exp/metrics.py, the first frame of every 12th viewpoint.

  python exp/compare_wt.py --player exp/results/r720 --wt exp/results/wt [--out DIR]

Writes tables.md and the figures (PDF and PNG) to --out (default
<wt>/figs/compare).
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from make_traces import segments  # noqa: E402
from summarize import COLOR, LABEL, PROFILE_COLOR  # noqa: E402

RTT = 20
TRACES = {"bus_x8": "bus ×8", "bus_x1": "bus ×1", "none": "no limit"}
ABRS = ["simple", "l2a", "lol+", "throughput"]
LABEL = dict(LABEL, throughput="Throughput")
METRICS = {"latency_ms": "mean latency (ms)", "latency_p95_ms": "95th pct. latency (ms)",
           "fps": "frames drawn / s", "shown": "viewpoints shown", "lost": "frames lost",
           "profile_mean": "mean profile", "switches_s": "switches / s",
           "bandwidth_mbps": "bandwidth used (Mbit/s)", "psnr": "PSNR (dB)", "ssim": "SSIM",
           "vmaf": "VMAF", "lpips": "LPIPS", "duration_s": "playback (s)"}
TABLE = ["latency_ms", "latency_p95_ms", "fps", "shown", "lost", "profile_mean", "switches_s",
         "bandwidth_mbps", "psnr", "ssim", "vmaf", "lpips", "duration_s"]
COLUMNS = ["latency_ms", "latency_p95_ms", "fps", "profile_mean", "psnr", "bandwidth_mbps"]
JITTER = np.random.default_rng(0)

# fonts, lines and markers 1.5 times matplotlib's defaults
SCALE = 1.5
plt.rcParams.update({k: plt.rcParams[k] * SCALE for k in (
    "font.size", "lines.linewidth", "lines.markersize", "axes.linewidth", "patch.linewidth",
    "xtick.major.size", "xtick.major.width", "xtick.minor.size", "xtick.minor.width",
    "ytick.major.size", "ytick.major.width", "ytick.minor.size", "ytick.minor.width")})
DOT = 12 * SCALE ** 2


def run_dir(results, r):
    return os.path.join(results, "runs", r.scene, r.trajectory, r.trace, f"rtt{r.rtt_ms:g}", r.abr, f"rep{r.rep}")


def load(results):
    runs = pd.read_csv(os.path.join(results, "summary", "runs.csv"))
    runs = runs[(runs["rtt_ms"] == RTT) & runs["trace"].isin(TRACES)].copy()
    runs["results"] = results
    return runs


def drawn_frames(d, run):
    """(time drawn in ms since t0, profile) of every frame drawn in the
    playback, in order."""
    if "wt" in run:
        fr = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "wt_frames.ndjson"))])
        fr = fr[fr["drawn"].notna() & (fr["drawn"] <= run["end_ms"])]
    else:
        fr = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "frames.ndjson"))])
        if "error" in fr:
            fr = fr[fr["error"].isna()]
    return fr.sort_values("drawn")[["drawn", "profile"]]


def extend(runs):
    """Adds shown, lost, profile over all frames drawn, switches per second;
    returns the per-viewpoint latencies of all runs."""
    rows, lat = [], []
    for r in runs.itertuples():
        d = run_dir(r.results, r)
        run = json.load(open(os.path.join(d, "run.json")))
        fr = drawn_frames(d, run)
        x = {"shown": 1 - run["failed_frames"] / run["viewpoints"], "lost": np.nan,
             "profile_mean": fr["profile"].mean(),
             "switches_s": (fr["profile"].diff().fillna(0) != 0).sum() / r.duration_s}
        for p in range(4):
            x[f"share_p{p}"] = (fr["profile"] == p).mean()
        if "wt" in run:
            w = run["wt"]
            x["lost"] = (w["frames_incomplete"] + w["frames_missing"]) / (w["frames_seen"] + w["frames_missing"])
            x["frame_latency_ms"] = frame_latency(d, run)
        rows.append(x)
        vp = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "frames.ndjson"))])
        if "error" in vp:
            vp = vp[vp["error"].isna()]
        lat.append(pd.DataFrame({"trace": r.trace, "abr": r.abr, "rep": r.rep,
                                 "latency_ms": vp["drawn"] - vp["send"]}))
    ext = pd.DataFrame(rows, index=runs.index)
    runs = runs.drop(columns=[c for c in ext.columns if c in runs.columns]).join(ext)
    return runs, pd.concat(lat, ignore_index=True)


def frame_latency(d, run):
    """Mean time from the pose message a frame was rendered from to drawing it."""
    fr = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "wt_frames.ndjson"))])
    fr = fr[fr["drawn"].notna() & (fr["drawn"] <= run["end_ms"])]
    return (fr["drawn"] - fr["ts"]).mean()


# ---- tables

def fmt(v, m=None):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        if m in ("shown", "lost"):
            return f"{100 * v:.1f} %"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def tables(runs):
    out = []
    for trace, name in TRACES.items():
        g = runs[runs["trace"] == trace]
        out.append(f"### {name}\n")
        cols = ["ABR (mode)", "runs"] + [METRICS[m] for m in TABLE]
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "---|" * len(cols))
        for abr in ABRS:
            a = g[g["abr"] == abr]
            if not len(a):
                continue
            mode = "push" if abr == "throughput" else "request"
            vals = [fmt(a[m].mean(), m) for m in TABLE]
            out.append(f"| {LABEL[abr]} ({mode}) | {len(a)} | " + " | ".join(vals) + " |")
        out.append("")
    out.append("### profile shares over all frames drawn\n")
    cols = ["trace", "ABR (mode)", "p0", "p1", "p2", "p3"]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "---|" * len(cols))
    for trace, name in TRACES.items():
        for abr in ABRS:
            a = runs[(runs["trace"] == trace) & (runs["abr"] == abr)]
            if len(a):
                out.append(f"| {name} | {LABEL[abr]} | " +
                           " | ".join(f"{100 * a[f'share_p{p}'].mean():.1f} %" for p in range(4)) + " |")
    wt = runs[runs["abr"] == "throughput"]
    if len(wt):
        out.append("\n### push mode: time from the pose message to the frame drawn (mean of runs, ms)\n")
        out.append("| trace | ms |\n|---|---|")
        for trace, name in TRACES.items():
            out.append(f"| {name} | {fmt(wt[wt['trace'] == trace]['frame_latency_ms'].mean())} |")
    return "\n".join(out) + "\n"


# ---- figures

def log_axis(axis, ticks=(20, 30, 50, 100, 200, 500, 1000)):
    """Log scale with plain numbers at the ticks inside the data range."""
    lo, hi = axis.get_view_interval()
    axis.set_ticks([t for t in ticks if lo <= t <= hi])
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def save(fig, path):
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=150)
    plt.close(fig)


def legend(fig, y=1.0):
    h = [Line2D([], [], color=COLOR[a], label=f"{LABEL[a]} ({'push' if a == 'throughput' else 'request'})")
         for a in ABRS]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, y))


def fig_cdf(lat, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (trace, name) in zip(axes, TRACES.items()):
        for abr in ABRS:
            v = np.sort(lat[(lat["trace"] == trace) & (lat["abr"] == abr)]["latency_ms"].dropna().to_numpy())
            if len(v):
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLOR[abr],
                        ls="--" if abr == "throughput" else "-")
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 300, 1000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"{name}: latency (ms)")
    axes[0].set_ylabel("CDF")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, path)


def fig_columns(runs, path, metrics=COLUMNS):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    A = len(ABRS)
    for ax, m in zip(axes.flat, metrics):
        for i, trace in enumerate(TRACES):
            for j, abr in enumerate(ABRS):
                x = i * (A + 1) + j
                v = runs[(runs["trace"] == trace) & (runs["abr"] == abr)][m].dropna()
                jit = (JITTER.random(len(v)) - 0.5) * 0.3
                ax.scatter(x + jit, v, s=DOT, color=COLOR[abr], zorder=3,
                           marker="D" if abr == "throughput" else "o")
                if len(v):
                    ax.hlines(v.mean(), x - 0.4, x + 0.4, color="black", lw=1.2 * SCALE, zorder=4)
        ax.set_xticks([i * (A + 1) + (A - 1) / 2 for i in range(len(TRACES))], list(TRACES.values()))
        ax.set_ylabel(METRICS[m])
        if m in ("latency_ms", "latency_p95_ms"):
            ax.set_yscale("log")
            log_axis(ax.yaxis)
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, path)


def fig_timeseries(runs, path, rep=1):
    """Push mode, one run each of bus x8 and x1: trace and receive rate,
    profile of the frames drawn, latency of the frames drawn."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 8), sharex="col")
    for col, trace in enumerate(["bus_x8", "bus_x1"]):
        r = runs[(runs["trace"] == trace) & (runs["abr"] == "throughput") & (runs["rep"] == rep)]
        if not len(r):
            continue
        r = next(r.itertuples())
        d = run_dir(r.results, r)
        run = json.load(open(os.path.join(d, "run.json")))
        t0 = run["start_ms"]
        fr = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "wt_frames.ndjson"))])
        fr = fr[fr["first"] <= run["end_ms"]]
        dr = fr[fr["drawn"].notna()]
        ax = axes[0, col]
        path_tr = run["rattan"].get("trace")
        if not path_tr or not os.path.exists(path_tr):
            path_tr = os.path.join(r.results, "traces", trace + ".json")
        segs = segments(path_tr)
        ts = np.array([s[0] for s in segs]) + run["rattan"]["trace_start_ms"] / 1e3 - t0 / 1e3
        bw = np.array([s[2] for s in segs]) / 1e6
        keep = (ts <= run["end_ms"] / 1e3 - t0 / 1e3 + 1)
        ax.step(ts[keep], bw[keep], where="post", color="black", lw=0.8 * SCALE)
        # bytes of the frames by the second their first chunk arrived
        sec = ((fr["first"] - t0) // 1000).astype(int)
        rate = fr.groupby(sec)["bytes"].sum() * 8 / 1e6
        ax.plot(rate.index + 0.5, rate.values, color=COLOR["throughput"], marker="o", ms=3 * SCALE, lw=0)
        t_end = np.append(ts[1:], np.inf)
        seen = bw[(t_end > 0.5) & keep]
        ax.set_ylim(0, 1.1 * max(list(seen) + list(rate.values)))
        ax.set_ylabel("Mbit/s")
        ax.set_title(TRACES[trace])
        ax = axes[1, col]
        ax.step((dr["drawn"] - t0) / 1e3, dr["profile"], where="post", color="black", lw=0.5 * SCALE)
        ax.set_yticks(range(4))
        ax.set_ylim(3.3, -0.3)
        ax.set_ylabel("profile")
        ax = axes[2, col]
        ax.scatter((dr["drawn"] - t0) / 1e3, dr["drawn"] - dr["ts"], s=1.5, color="black")
        ax.set_yscale("log")
        log_axis(ax.yaxis)
        ax.set_ylabel("message to frame (ms)")
        ax.set_xlabel("time (s)")
        ax.set_xlim(-0.5, (run["end_ms"] - t0) / 1e3 + 0.5)
    fig.tight_layout()
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--player", required=True, help="results of /player (r720)")
    ap.add_argument("--wt", required=True, help="results of /player-wt (exp/wt.toml)")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.wt, "figs", "compare")
    os.makedirs(out, exist_ok=True)
    runs = pd.concat([load(os.path.abspath(a.player)), load(os.path.abspath(a.wt))], ignore_index=True)
    runs, lat = extend(runs)
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    open(os.path.join(out, "tables.md"), "w").write(tables(runs))
    fig_cdf(lat, os.path.join(out, "latency_cdf.pdf"))
    fig_columns(runs, os.path.join(out, "columns.pdf"))
    fig_timeseries(runs, os.path.join(out, "wt_timeseries.pdf"))
    print(open(os.path.join(out, "tables.md")).read())
    print(f"compare_wt: {len(runs)} runs, output in {out}")


if __name__ == "__main__":
    main()
