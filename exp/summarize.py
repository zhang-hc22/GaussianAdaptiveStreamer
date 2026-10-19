#!/usr/bin/env python
"""Tables and figures of a results directory.

Per frame (frames.ndjson; times in ms since t0):
- latency: drawn - send, from asking for the viewpoint to drawing its frame;
- network time: responseEnd - requestStart of the /render request;
- render time: what the server reported;
- transfer rate: (bytes - 1200) * 8 / (responseEnd - responseStart), frames
  of at least 3 kB only (the first packet carries the headers, the rest
  arrives at the bottleneck rate).
- wait: responseStart - send, until the first byte of the frame arrived;
  download: responseEnd - responseStart.
Runs of the WebTransport push mode (client.py --mode wt) have one line per
viewpoint with the first frame drawn for it: latency = drawn - send, from
setting the viewpoint to drawing its first frame; network time = drawn - msg,
from the pose message the frame was rendered from; wait = first chunk - send;
download = first to last chunk. Viewpoints never shown are left out like
failed frames; fps counts every frame drawn and bandwidth every byte received
(run.json "wt").
Per run: bandwidth used = all frame bytes / playback time; switches = frames
whose profile differs from the previous frame; quality = mean over the
frames in metrics/ (ours) and in metrics_paper/ (the paper's method,
columns paper_<metric>); trace_mbps = mean bandwidth of the trace during
playback; ghent_pos_s = second of the Ghent trace at the start of playback.
Runs shaped by tc on lo (run_one.py --shaper tc-lo) use the times the rates
actually took effect.

Tables:
  summary/runs.csv        one row per run
  summary/conditions.csv  mean and standard deviation over the repetitions
Figures (a condition is scene, trajectory, trace and RTT):
  figs/quality_<condition>.pdf   six quality metrics per ABR, frames of all
                                 repetitions (paper figure 5)
  figs/quality_paper_<condition>.pdf   the same with the paper's method
  figs/system_<condition>.pdf    latency, render time, profiles, bandwidth,
                                 switches per ABR (paper figure 7)
  figs/rtt_<scene>_<trajectory>.pdf   runs without a trace: latency per frame
                                 at the lowest RTT (first repetition), and mean
                                 latency, mean profile and switches against
                                 RTT (paper figure 8)
  figs/timeseries/<run>.pdf      one run over time: trace bandwidth and
                                 transfer rate, profile, latency

  python exp/summarize.py RESULTS_DIR
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import REPO, find_runs  # noqa: E402
from make_traces import segments  # noqa: E402

ABRS = ["simple", "l2a", "lol+"]   # /player; abrs_in() puts others after them
LABEL = {"simple": "Latency", "l2a": "L2A", "lol+": "LoL+",   # names in the paper
         "throughput": "Throughput (WT)"}
COLOR = {"simple": "#1b9e77", "l2a": "#d95f02", "lol+": "#7570b3", "throughput": "#e7298a"}
PROFILE_COLOR = ["#253494", "#2c7fb8", "#41b6c4", "#a1dab4"]
METRICS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "vmaf": "VMAF", "vif": "VIF",
           "lpips": "LPIPS", "dists": "DISTS"}
MIN_BYTES = 3000
KEYS = ["scene", "trajectory", "trace", "rtt_ms"]


def shaping(run):
    """RTT (ms), and when the trace and its Ghent part started (ms since t0)."""
    if "tc" in run:
        tc = run["tc"]
        return tc["rtt_ms"], tc["trace_start_ms"], tc["ghent_start_ms"]
    ra = run["rattan"]
    ts = ra["trace_start_ms"]
    return ra["rtt_ms"], ts, None if ts is None else ts + run["args"]["warmup"] * 1e3


def bw_steps(results, r, run):
    """(start of each rate in s since t0, rate in bps) of the run's trace, or
    None without one."""
    if "tc" in run:
        st = np.array(run["tc"]["steps"]).reshape(-1, 2)
        return st[:, 0] / 1e3, st[:, 1] * 1e3
    if r["trace"] == "none" or run["rattan"]["trace_start_ms"] is None:
        return None
    path = run["rattan"].get("trace")
    if not path or not os.path.exists(path):
        path = os.path.join(results, "traces", r["trace"] + ".json")
    segs = segments(path)
    return (np.array([x[0] for x in segs]) + run["rattan"]["trace_start_ms"] / 1e3,
            np.array([x[2] for x in segs], dtype=float))


def mean_bw(steps, a, b):
    """Mean of the step function steps over [a, b] (s since t0)."""
    t, bw = steps
    edges = np.append(t, np.inf)
    span = np.clip(edges[1:], a, b) - np.clip(edges[:-1], a, b)
    return float(np.sum(bw * span) / (b - a))


def ghent_len(trace):
    """Seconds in traces/<name>.csv of a trace label (bus_x8_phi -> bus)."""
    path = os.path.join(REPO, "traces", trace.split("_x")[0] + ".csv")
    return sum(1 for l in open(path) if l.strip()) if os.path.exists(path) else None


def frame_table(r, run):
    fr = pd.DataFrame([json.loads(l) for l in open(os.path.join(r["dir"], "frames.ndjson"))])
    if "error" in fr:
        fr = fr[fr["error"].isna()].copy()
    fr["latency_ms"] = fr["drawn"] - fr["send"]
    if run["args"].get("mode") == "wt":
        first, last, sent = fr["first"], fr["drawn"], fr["msg"]
    else:
        first, last, sent = fr["resp_start"], fr["resp_end"], fr["req_start"]
    fr["wait_ms"] = first - fr["send"]
    fr["download_ms"] = last - first
    fr["net_ms"] = last - sent
    dt = last - first
    big = (fr["bytes"] >= MIN_BYTES) & (dt > 0)
    fr["rate_mbps"] = np.where(big, (fr["bytes"] - 1200) * 8 / dt.where(dt > 0) / 1e3, np.nan)
    fr["t_s"] = (fr["send"] - run["start_ms"]) / 1e3
    # middle of the transfer at the bottleneck, as playback time
    fr["rate_t_s"] = ((first + last) / 2 - shaping(run)[0] / 2 - run["start_ms"]) / 1e3
    return fr


def run_row(r, run, fr, q, qp, steps):
    dur = (run["end_ms"] - run["start_ms"]) / 1e3
    row = {k: r[k] for k in KEYS + ["abr", "rep"]}
    row.update({
        "resolution": f"{run['camera']['width']}x{run['camera']['height']}", "seed": run["args"]["seed"], "viewpoints": run["viewpoints"], "failed": run["failed_frames"],
        "duration_s": dur, "fps": len(fr) / dur,
        "latency_ms": fr["latency_ms"].mean(), "latency_median_ms": fr["latency_ms"].median(),
        "latency_p95_ms": fr["latency_ms"].quantile(0.95), "wait_ms": fr["wait_ms"].mean(),
        "download_ms": fr["download_ms"].mean(), "net_ms": fr["net_ms"].mean(),
        "render_ms": fr["render_ms"].mean(), "bandwidth_mbps": fr["bytes"].sum() * 8 / dur / 1e6,
        "shown": len(fr) / run["viewpoints"],
        "profile_mean": fr["profile"].mean(), "profile_median": fr["profile"].median(),
        "switches": int((fr["profile"].diff().fillna(0) != 0).sum()),
    })
    if "wt" in run:
        row["fps"] = run["wt"]["drawn_in_playback"] / dur
        row["bandwidth_mbps"] = run["wt"]["bytes"] * 8 / dur / 1e6
    for p in range(4):
        row[f"share_p{p}"] = (fr["profile"] == p).mean()
    for m in METRICS:
        row[m] = q[m].mean() if q is not None else np.nan
    row["quality_frames"] = len(q) if q is not None else 0
    for m in METRICS:
        row["paper_" + m] = qp[m].mean() if qp is not None else np.nan
    row["paper_other_profile"] = (qp["profile"] != qp["sent_profile"]).mean() if qp is not None else np.nan
    _, row["trace_start_ms"], ghent_start = shaping(run)
    row["trace_mbps"] = (mean_bw(steps, run["start_ms"] / 1e3, run["end_ms"] / 1e3) / 1e6
                         if steps is not None else np.nan)
    plan, n = run.get("plan") or {}, ghent_len(r["trace"])
    row["offset_s"] = plan.get("offset_s", 0)
    pos = ((run["start_ms"] - ghent_start) / 1e3 + row["offset_s"]
           if steps is not None and ghent_start is not None and n else np.nan)
    row["ghent_pos_s"] = pos % n if pos > 0 else pos   # slightly negative: before the Ghent part
    row["session_index"] = plan.get("session_index") or np.nan
    row["tc_lag_ms"] = run["tc"]["lag_ms"] if "tc" in run else np.nan
    row["tigas_commit"] = run["tigas_commit"]
    return row


def cond_name(c):
    scene, traj, trace, rtt = c
    return f"{scene}_{traj.replace('/', '-')}_{trace}_rtt{rtt:g}"


def abrs_in(df):
    return [a for a in ABRS if a in set(df["abr"])] + sorted(set(df["abr"]) - set(ABRS))


def boxes(ax, data, abrs, ylabel):
    bp = ax.boxplot([data[a] for a in abrs], showfliers=False, showmeans=True, widths=0.6,
                    patch_artist=True, meanprops={"marker": "o", "markerfacecolor": "white",
                                                  "markeredgecolor": "black", "markersize": 4})
    for patch, a in zip(bp["boxes"], abrs):
        patch.set_facecolor(COLOR.get(a, "grey"))
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(abrs) + 1), [LABEL.get(a, a) for a in abrs])
    ax.set_ylabel(ylabel)


def per_run(ax, runs, abrs, col, ylabel):
    for k, a in enumerate(abrs, 1):
        v = runs[runs["abr"] == a][col]
        ax.bar(k, v.mean(), width=0.6, color=COLOR.get(a, "grey"), alpha=0.6)
        ax.scatter(np.full(len(v), k), v, s=10, color="black", zorder=3)
    ax.set_xticks(range(1, len(abrs) + 1), [LABEL.get(a, a) for a in abrs])
    ax.set_ylabel(ylabel)


def fig_quality(quality, path):
    abrs = abrs_in(quality)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, (m, label) in zip(axes.flat, METRICS.items()):
        boxes(ax, {a: quality[quality["abr"] == a][m].dropna() for a in abrs}, abrs, label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_system(frames, runs, path):
    abrs = abrs_in(runs)
    w, h = map(int, runs["resolution"].iloc[0].split("x"))
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    boxes(axes[0], {a: frames[frames["abr"] == a]["latency_ms"] for a in abrs}, abrs, "latency (ms)")
    boxes(axes[1], {a: frames[frames["abr"] == a]["render_ms"] for a in abrs}, abrs, "render time (ms)")
    ax = axes[2]
    bottom = np.zeros(len(abrs))
    for p in range(4):
        share = np.array([runs[runs["abr"] == a][f"share_p{p}"].mean() for a in abrs])
        ax.bar(range(1, len(abrs) + 1), share, bottom=bottom, width=0.6, color=PROFILE_COLOR[p],
               label=f"{w >> p}x{h >> p}")
        bottom += share
    ax.set_xticks(range(1, len(abrs) + 1), [LABEL.get(a, a) for a in abrs])
    ax.set_ylabel("share of frames")
    ax.set_ylim(0, 1)
    per_run(axes[3], runs, abrs, "bandwidth_mbps", "bandwidth used (Mbit/s)")
    per_run(axes[4], runs, abrs, "switches", "profile switches")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def fig_rtt(frames, runs, path):
    abrs = abrs_in(runs)
    rtts = sorted(set(runs["rtt_ms"]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax = axes[0, 0]
    base = frames[(frames["rtt_ms"] == rtts[0]) & (frames["rep"] == frames["rep"].min())]
    for a in abrs:
        f = base[base["abr"] == a]
        ax.plot(f["t_s"], f["latency_ms"], lw=0.6, color=COLOR.get(a, "grey"), label=LABEL.get(a, a))
    ax.set_xlabel(f"time (s), RTT {rtts[0]:g} ms")
    ax.set_ylabel("latency (ms)")
    ax.legend(frameon=False)
    for ax, (col, label) in zip(axes.flat[1:], [("latency_ms", "mean latency (ms)"),
                                                 ("profile_mean", "mean profile"),
                                                 ("switches", "profile switches")]):
        for a in abrs:
            g = runs[runs["abr"] == a].groupby("rtt_ms")[col]
            ax.errorbar(g.mean().index, g.mean(), yerr=g.std().fillna(0), marker="o", ms=4, capsize=3,
                        color=COLOR.get(a, "grey"))
        ax.set_xlabel("RTT (ms)")
        ax.set_ylabel(label)
        ax.set_xticks(rtts)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_timeseries(steps, run, fr, path):
    trace = steps is not None
    rows = 3 if trace else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.2 * rows), sharex=True)
    k = 0
    if trace:
        ax = axes[0]
        end = fr["t_s"].max() + 1
        t = steps[0] - run["start_ms"] / 1e3   # playback time
        bw = steps[1] / 1e6
        keep = t <= end
        ax.step(t[keep], bw[keep], where="post", color="black", lw=0.8)
        for p in range(4):
            m = (fr["profile"] == p) & fr["rate_mbps"].notna()
            ax.scatter(fr["rate_t_s"][m], fr["rate_mbps"][m], s=2, color=PROFILE_COLOR[p])
        # y range of the playback part; the warm-up segment, much higher, ends
        # a few ms after playback starts
        t_end = np.append(t[1:], np.inf)
        seen = list(bw[(t_end > 0.5) & (t < end)])
        rates = fr["rate_mbps"].dropna()
        ax.set_ylim(0, 1.1 * max(seen + ([rates.quantile(0.99)] if len(rates) else [])))
        ax.set_ylabel("Mbit/s")
        k = 1
    axes[k].step(fr["t_s"], fr["profile"], where="post", color="black", lw=0.6)
    axes[k].set_yticks(range(4))
    axes[k].set_ylim(3.3, -0.3)
    axes[k].set_ylabel("profile")
    axes[k + 1].scatter(fr["t_s"], fr["latency_ms"], s=2, color="black")
    axes[k + 1].set_ylabel("latency (ms)")
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(-0.5, fr["t_s"].max() + 0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    a = ap.parse_args()
    results = os.path.abspath(a.results)
    rows, frames, quality, quality_paper = [], [], [], []
    found = find_runs(results)
    for r in found:
        run = json.load(open(os.path.join(r["dir"], "run.json")))
        if run["status"] != "ok" or ("rattan" not in run and "tc" not in run):
            continue
        fr = frame_table(r, run)
        q, qp = (pd.read_csv(f) if os.path.exists(f) else None
                 for f in (os.path.join(results, d, r["rel"] + ".csv") for d in ("metrics", "metrics_paper")))
        steps = bw_steps(results, r, run)
        rows.append(run_row(r, run, fr, q, qp, steps))
        tag = {k: r[k] for k in KEYS + ["abr", "rep"]}
        frames.append(fr.assign(**tag))
        if q is not None:
            quality.append(q.assign(**tag))
        if qp is not None:
            quality_paper.append(qp.assign(**tag))
        r["run"], r["fr"], r["steps"] = run, fr, steps
    if not rows:
        raise SystemExit("summarize: no finished runs")
    runs = pd.DataFrame(rows)
    frames = pd.concat(frames, ignore_index=True)
    quality = pd.concat(quality, ignore_index=True) if quality else None
    quality_paper = pd.concat(quality_paper, ignore_index=True) if quality_paper else None

    os.makedirs(os.path.join(results, "summary"), exist_ok=True)
    runs.to_csv(os.path.join(results, "summary", "runs.csv"), index=False)
    num = [c for c in runs.columns if c not in KEYS + ["abr", "rep", "resolution", "seed", "tigas_commit"]]
    g = runs.groupby(KEYS + ["abr"], sort=False)[num]
    cond = g.mean().join(g.std(), rsuffix="_std").join(g.size().rename("runs")).reset_index()
    cond.to_csv(os.path.join(results, "summary", "conditions.csv"), index=False)

    figs = os.path.join(results, "figs")
    os.makedirs(os.path.join(figs, "timeseries"), exist_ok=True)
    written = 0
    for c, rs in runs.groupby(KEYS, sort=False):
        sel = (frames[KEYS] == pd.Series(c, index=KEYS)).all(axis=1)
        fig_system(frames[sel], rs, os.path.join(figs, f"system_{cond_name(c)}.pdf"))
        written += 1
        for qa, prefix in ((quality, "quality"), (quality_paper, "quality_paper")):
            if qa is None:
                continue
            qs = qa[(qa[KEYS] == pd.Series(c, index=KEYS)).all(axis=1)]
            if len(qs):
                fig_quality(qs, os.path.join(figs, f"{prefix}_{cond_name(c)}.pdf"))
                written += 1
    nt = runs[runs["trace"] == "none"]
    for (scene, traj), rs in nt.groupby(["scene", "trajectory"], sort=False):
        sel = (frames["trace"] == "none") & (frames["scene"] == scene) & (frames["trajectory"] == traj)
        fig_rtt(frames[sel], rs,
                os.path.join(figs, f"rtt_{scene}_{traj.replace('/', '-')}.pdf"))
        written += 1
    for r in found:
        if "fr" not in r:
            continue
        name = f"{cond_name((r['scene'], r['trajectory'], r['trace'], r['rtt_ms']))}_{r['abr']}_rep{r['rep']}"
        fig_timeseries(r["steps"], r["run"], r["fr"], os.path.join(figs, "timeseries", name + ".pdf"))
        written += 1

    show = ["trace", "rtt_ms", "abr", "runs", "duration_s", "latency_ms", "bandwidth_mbps",
            "profile_mean", "switches", *METRICS]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(cond[show].to_string(index=False))
        print("\nthe paper's quality method (paper_other_profile: share of frames re-rendered at "
              "another profile than requested)")
        print(cond[["trace", "rtt_ms", "abr", "paper_other_profile", *("paper_" + m for m in METRICS)]]
              .to_string(index=False))
    print(f"summarize: {len(runs)} runs, {written} figures in {figs}")


if __name__ == "__main__":
    main()
