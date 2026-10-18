#!/usr/bin/env python
"""Compares /player over QUIC (exp/quic_tms.toml) with WebRTC
(exp/webrtc.toml), both with the viewpoints sent at their trajectory times
(pacing tms), under the same networks: bus x8, bus x1 and no bandwidth
limit, RTT 20 ms, 1280x720, the same trajectory.

Viewpoint i is due at d_i = t0 + W + tMs_i in both. /player asks for one
viewpoint at a time with the ABR's profile and skips the viewpoints that
came due while a request was out; WebRTC sends every viewpoint, the sender
renders each one at full size and x264 encodes it at GCC's rate.

The same measures for both, from frames.ndjson (QUIC: drawn = frame drawn;
WebRTC: drawn = frame read from the track):
- latency: drawn_i - d_i, viewpoints with their own frame;
- catch-up latency: for every viewpoint i, the first time a frame of a
  viewpoint >= i is drawn, minus d_i; a skipped viewpoint waits for the next
  frame;
- own frames: share of viewpoints with their own frame; frames/s: their
  number over the trajectory's length T;
- stalls: gaps between consecutive frames drawn longer than
  max(3 dt, dt + 150 ms), dt the mean viewpoint interval (WebRTC's freeze
  rule with the trajectory's rate, 178 ms here); stall share = their total
  length / T, and stalls per minute;
- quality: exp/metrics.py on every 12th viewpoint: "own" over the viewpoints
  with their own frame, "all" with the stand-ins too (the frame on screen
  while the viewpoint was due);
- bandwidth used, trace bandwidth: summary/runs.csv.

  python exp/compare_tms.py --quic exp/results/quic-tms --webrtc exp/results/webrtc [--out DIR]

Writes runs.csv, tables.md and the figures (PDF and PNG) to --out (default
<webrtc>/figs/compare-tms).
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
from summarize import COLOR  # noqa: E402

RTT = 20
TRACES = {"bus_x8": "bus ×8", "bus_x1": "bus ×1", "none": "no limit"}
SCHEMES = ["webrtc-tms", "simple", "l2a", "lol+"]
LABEL = {"webrtc-tms": "WebRTC", "simple": "QUIC Latency", "l2a": "QUIC L2A", "lol+": "QUIC LoL+"}
QUALITY = ["psnr", "ssim", "vmaf", "lpips"]
METRICS = {"psnr_own": "PSNR, own frames (dB)", "psnr_all": "PSNR, all (dB)", "ssim_own": "SSIM, own",
           "vmaf_own": "VMAF, own", "lpips_own": "LPIPS, own", "vmaf_all": "VMAF, all",
           "latency_ms": "mean latency (ms)", "latency_median_ms": "median latency (ms)",
           "latency_p95_ms": "95th pct. latency (ms)", "catchup_p95_ms": "95th pct. catch-up (ms)",
           "own_share": "own frames", "fps": "frames / s", "stall_share": "stall share",
           "stalls_min": "stalls / min", "bandwidth_mbps": "bandwidth used (Mbit/s)",
           "trace_mbps": "trace (Mbit/s)", "profile_mean": "mean profile", "qp_mean": "mean QP"}
TABLES = {"quality": ["psnr_own", "psnr_all", "ssim_own", "vmaf_own", "vmaf_all", "lpips_own"],
          "latency and smoothness": ["latency_ms", "latency_median_ms", "latency_p95_ms", "catchup_p95_ms",
                                     "own_share", "fps", "stall_share", "stalls_min"],
          "rate": ["bandwidth_mbps", "trace_mbps", "profile_mean", "qp_mean"]}
PERCENT = {"own_share", "stall_share"}
COLUMNS = ["psnr_own", "vmaf_own", "lpips_own", "latency_ms", "latency_p95_ms", "catchup_p95_ms",
           "stall_share", "fps", "bandwidth_mbps"]
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
    runs = runs[(runs["rtt_ms"] == RTT) & runs["trace"].isin(TRACES) & runs["abr"].isin(SCHEMES)].copy()
    runs["results"] = results
    return runs


def viewpoints(d, run):
    """frames.ndjson with due (ms since t0) and own (has its own frame)."""
    vp = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "frames.ndjson"))])
    vp["due"] = run["args"]["warmup"] * 1e3 + vp["t_ms"]
    err = vp["error"].notna() if "error" in vp else pd.Series(False, index=vp.index)
    vp["own"] = ~err & vp["drawn"].notna()
    return vp.sort_values("i").reset_index(drop=True)


def qoe(vp, q):
    """The measures of the module doc of one run."""
    n = len(vp)
    span = vp["t_ms"].iloc[-1] - vp["t_ms"].iloc[0]
    dt = span / (n - 1)
    own = vp[vp["own"]]
    lat = own["drawn"] - own["due"]
    # earliest frame drawn of a viewpoint >= i
    drawn = np.where(vp["own"], vp["drawn"], np.inf)
    first = np.minimum.accumulate(drawn[::-1])[::-1]
    catchup = np.where(np.isfinite(first), first - vp["due"], np.nan)
    gaps = np.diff(np.sort(own["drawn"].to_numpy()))
    stall = gaps[gaps > max(3 * dt, dt + 150)]
    x = {"latency_ms": lat.mean(), "latency_median_ms": lat.median(), "latency_p95_ms": lat.quantile(0.95),
         "catchup_ms": np.nanmean(catchup), "catchup_p95_ms": np.nanpercentile(catchup, 95),
         "own_share": len(own) / n, "fps": len(own) / (span / 1e3),
         "stall_share": stall.sum() / span, "stalls_min": len(stall) / (span / 6e4),
         "stall_max_ms": stall.max() if len(stall) else 0.0, "stall_threshold_ms": max(3 * dt, dt + 150)}
    for m in QUALITY:
        x[m + "_all"] = q[m].mean() if q is not None and len(q) else np.nan
        x[m + "_own"] = q.loc[q["standin"] == 0, m].mean() if q is not None and len(q) else np.nan
    x["quality_frames"] = len(q) if q is not None else 0
    x["standin_frames"] = int(q["standin"].sum()) if q is not None and len(q) else 0
    return x, lat.to_numpy(), catchup


def extend(runs):
    """Adds the measures of qoe(); returns the per-viewpoint latencies of all runs."""
    rows, lat = [], []
    for r in runs.itertuples():
        d = run_dir(r.results, r)
        run = json.load(open(os.path.join(d, "run.json")))
        vp = viewpoints(d, run)
        rel = os.path.relpath(d, os.path.join(r.results, "runs"))
        path = os.path.join(r.results, "metrics", rel + ".csv")
        q = pd.read_csv(path) if os.path.exists(path) else None
        if q is not None and "standin" not in q:
            q["standin"] = 0
        x, l, c = qoe(vp, q)
        rows.append(x)
        lat.append(pd.DataFrame({"trace": r.trace, "abr": r.abr, "rep": r.rep, "kind": "latency", "ms": l}))
        lat.append(pd.DataFrame({"trace": r.trace, "abr": r.abr, "rep": r.rep, "kind": "catchup", "ms": c}))
    ext = pd.DataFrame(rows, index=runs.index)
    runs = runs.drop(columns=[c for c in ext.columns if c in runs.columns]).join(ext)
    webrtc = runs["abr"] == "webrtc-tms"
    runs.loc[webrtc, "profile_mean"] = np.nan   # WebRTC has no profiles, QUIC no QP
    if "qp_mean" in runs:
        runs.loc[~webrtc, "qp_mean"] = np.nan
    return runs, pd.concat(lat, ignore_index=True)


# ---- tables

def fmt(v, m=None):
    if isinstance(v, (float, np.floating, int, np.integer)):
        if np.isnan(v):
            return "–"
        if m in PERCENT:
            return f"{100 * v:.1f} %"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def tables(runs):
    out = []
    for title, cols in TABLES.items():
        out.append(f"### {title}\n")
        head = ["trace", "scheme", "runs"] + [METRICS[m] for m in cols]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for trace, name in TRACES.items():
            for s in SCHEMES:
                a = runs[(runs["trace"] == trace) & (runs["abr"] == s)]
                if len(a):
                    out.append(f"| {name} | {LABEL[s]} | {len(a)} | "
                               + " | ".join(fmt(a[m].mean(), m) if m in a else "–" for m in cols) + " |")
        out.append("")
    out.append("### standard deviation over the repetitions\n")
    cols = ["psnr_own", "latency_ms", "latency_p95_ms", "stall_share", "bandwidth_mbps"]
    head = ["trace", "scheme"] + [METRICS[m] for m in cols]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for trace, name in TRACES.items():
        for s in SCHEMES:
            a = runs[(runs["trace"] == trace) & (runs["abr"] == s)]
            if len(a) > 1:
                out.append(f"| {name} | {LABEL[s]} | " + " | ".join(fmt(a[m].std(), m) for m in cols) + " |")
    return "\n".join(out) + "\n"


# ---- figures

def log_axis(axis, ticks=(20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000)):
    """Log scale with plain numbers at the ticks inside the data range."""
    lo, hi = axis.get_view_interval()
    axis.set_ticks([t for t in ticks if lo <= t <= hi])
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def save(fig, path):
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=150)
    plt.close(fig)


def style(s):
    return {"color": COLOR[s], "ls": "-" if s == "webrtc-tms" else "--"}


def fig_cdf(lat, path):
    """Top row latency of the viewpoints with their own frame, bottom row
    catch-up latency of all viewpoints; one column per trace."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for row, kind in enumerate(["latency", "catchup"]):
        for ax, (trace, name) in zip(axes[row], TRACES.items()):
            for s in SCHEMES:
                v = lat[(lat["trace"] == trace) & (lat["abr"] == s) & (lat["kind"] == kind)]["ms"]
                v = np.sort(v.dropna().to_numpy())
                if len(v):
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), **style(s))
            ax.set_xscale("log")
            ax.set_xlim(left=20)
            log_axis(ax.xaxis)
            ax.set_xlabel(f"{name}: {'latency' if kind == 'latency' else 'catch-up latency'} (ms)")
        axes[row, 0].set_ylabel("CDF")
    fig.tight_layout()   # no legend on CDFs: the colours are those of columns.pdf
    save(fig, path)


def fig_columns(runs, path, metrics=COLUMNS):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    A = len(SCHEMES)
    for ax, m in zip(axes.flat, metrics):
        for i, trace in enumerate(TRACES):
            for j, s in enumerate(SCHEMES):
                x = i * (A + 1) + j
                v = runs[(runs["trace"] == trace) & (runs["abr"] == s)][m].dropna()
                jit = (JITTER.random(len(v)) - 0.5) * 0.3
                ax.scatter(x + jit, v, s=DOT, color=COLOR[s], zorder=3, marker="D" if s == "webrtc-tms" else "o")
                if len(v):
                    ax.hlines(v.mean(), x - 0.4, x + 0.4, color="black", lw=1.2 * SCALE, zorder=4)
        ax.set_xticks([i * (A + 1) + (A - 1) / 2 for i in range(len(TRACES))], list(TRACES.values()))
        ax.set_ylabel(METRICS[m])
        if m in ("latency_ms", "latency_p95_ms", "catchup_p95_ms", "stall_share", "fps", "bandwidth_mbps"):
            ax.set_ylim(bottom=0)
        if m in PERCENT:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    h = [Line2D([], [], color=COLOR[s], marker="D" if s == "webrtc-tms" else "o", lw=0, label=LABEL[s])
         for s in SCHEMES]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def trace_steps(results, run, trace, w0):
    """(start in s of playback time, Mbit/s) of the run's trace; w0 = t0 + W
    in ms since t0."""
    path = run["rattan"].get("trace")
    if not path or not os.path.exists(path):
        path = os.path.join(results, "traces", (run.get("plan") or {}).get("trace_file", trace) + ".json")
    segs = segments(path)
    ts = np.array([s[0] for s in segs]) + (run["rattan"]["trace_start_ms"] - w0) / 1e3
    return ts, np.array([s[2] for s in segs]) / 1e6


def sent_rate(d, run, w0, bins):
    """Mbit/s per bin of playback time: QUIC frame bytes by the time they were
    drawn, WebRTC encoded bytes by the time they were encoded."""
    if run["args"].get("mode") == "webrtc":
        enc = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "sender_encoded.ndjson"))])
        t, b = (enc["enc"] - w0) / 1e3, enc["bytes"]
    else:
        vp = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "frames.ndjson"))])
        vp = vp[vp["bytes"].notna()] if "bytes" in vp else vp.iloc[:0]
        t, b = (vp["drawn"] - w0) / 1e3, vp["bytes"]
    h, _ = np.histogram(t, bins=bins, weights=b)
    return h * 8 / np.diff(bins) / 1e6


def fig_timeseries(runs, path, rep=1, width=1.0):
    """One repetition each of bus x8 and x1: trace and rate used (1 s bins),
    then the latency of each scheme's own frames over playback time."""
    fig, axes = plt.subplots(1 + len(SCHEMES), 2, figsize=(15, 2.3 * (1 + len(SCHEMES))), sharex="col")
    for col, trace in enumerate(["bus_x8", "bus_x1"]):
        sel = runs[(runs["trace"] == trace) & (runs["rep"] == rep)]
        top, end, worst = [], 62, 100
        for k, s in enumerate(SCHEMES):
            r = sel[sel["abr"] == s]
            if not len(r):
                continue
            r = next(r.itertuples())
            d = run_dir(r.results, r)
            run = json.load(open(os.path.join(d, "run.json")))
            w0 = run["args"]["warmup"] * 1e3
            vp = viewpoints(d, run)
            if k == 0:
                ts, bw = trace_steps(r.results, run, trace, w0)
                keep = ts <= end
                axes[0, col].step(np.append(ts[keep], end), np.append(bw[keep], bw[keep][-1:]), where="post",
                                  color="black", lw=0.8 * SCALE)
                t_end = np.append(ts[1:], np.inf)
                top += list(bw[(t_end > 0.5) & keep])
            bins = np.arange(0, end + width, width)
            rate = sent_rate(d, run, w0, bins)
            axes[0, col].step(bins[:-1], rate, where="post", lw=0.8 * SCALE, **style(s))
            top.append(rate.max())
            own = vp[vp["own"]]
            ax = axes[1 + k, col]
            ax.scatter((own["due"] - w0) / 1e3, own["drawn"] - own["due"], s=1.5, color=COLOR[s])
            ax.set_yscale("log")
            ax.set_ylabel(f"{LABEL[s]}\nlatency (ms)")
            worst = max(worst, (own["drawn"] - own["due"]).max())
        for ax in axes[1:, col]:   # one latency range per column
            ax.set_ylim(20, 1.3 * worst)
            log_axis(ax.yaxis, (20, 50, 100, 200, 500, 1000, 2000, 5000))
        axes[0, col].set_ylim(0, 1.1 * max(top) if top else 1)
        axes[0, col].set_ylabel("Mbit/s")
        axes[0, col].set_title(TRACES[trace])
        axes[-1, col].set_xlabel("time (s)")
        axes[-1, col].set_xlim(-0.5, end)
    fig.tight_layout()
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quic", required=True, help="results of exp/quic_tms.toml")
    ap.add_argument("--webrtc", required=True, help="results of exp/webrtc.toml (tms tables)")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.webrtc, "figs", "compare-tms")
    os.makedirs(out, exist_ok=True)
    runs = pd.concat([load(os.path.abspath(a.quic)), load(os.path.abspath(a.webrtc))], ignore_index=True)
    runs, lat = extend(runs)
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    open(os.path.join(out, "tables.md"), "w").write(tables(runs))
    fig_cdf(lat, os.path.join(out, "latency_cdf.pdf"))
    fig_columns(runs, os.path.join(out, "columns.pdf"))
    fig_timeseries(runs, os.path.join(out, "timeseries.pdf"))
    print(open(os.path.join(out, "tables.md")).read())
    print(f"compare_tms: {len(runs)} runs, output in {out}")


if __name__ == "__main__":
    main()
