#!/usr/bin/env python
"""Compares the two push modes under the same networks: /player-wt over
WebTransport (QUIC, exp/wt.toml) and WebRTC with pacing push
(exp/webrtc_push.toml). Bus x8, bus x1 and no bandwidth limit, RTT 20 ms,
1280x720, the same trajectory.

In both, viewpoint k is set at start + 40 k ms, the page sends the current
viewpoint every 10 ms, and the sender renders the latest one 60 times a
second, so a viewpoint can get several frames or none. QUIC pushes each
frame as JPEG datagrams at the profile of the page's ThroughputABR; WebRTC
encodes it with x264 at GCC's rate, always at full size.

Per run, the same measures for both (drawn: QUIC frame drawn, WebRTC frame
read from the track):
- latency: from setting a viewpoint to drawing its first frame, viewpoints
  without a frame left out; the shares above 150 and 200 ms;
- shown: share of viewpoints with a frame drawn;
- frames/s: frames drawn per second of playback; repeat share: frames drawn
  whose viewpoint is that of the frame drawn before, by count and by bytes;
- screen age: for every ms of playback, the time since the viewpoint of the
  frame on screen was set; the shares of playback time above 150, 200 and
  300 ms;
- stalls: gaps between frames drawn longer than max(3 dt, dt + 150 ms) with
  dt = 1/60 s (167 ms, WebRTC's freeze rule at the push rate);
- bandwidth used: bytes per second of playback (QUIC: bytes of the frames
  whose first chunk arrived in playback; WebRTC: bytes encoded in playback);
- quality: exp/metrics.py, the first frame of every 12th viewpoint.
Latency parts: set to the message the frame was rendered from (both), then
QUIC message to first chunk and first chunk to drawn; WebRTC message to
the sender (uplink), wait for the tick, render, encode and send, network
and read (the last two only for frames requestVideoFrameCallback saw).

  python exp/compare_push.py --wt exp/results/wt --webrtc exp/results/webrtc-push [--out DIR]

Writes runs.csv, tables.md and the figures (PDF and PNG) to --out (default
<webrtc>/figs/compare-push), laid out as those of exp/compare_tms.py; screen
age takes the place of catch-up latency, which is the latency itself when
nearly every viewpoint is drawn.
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

from compare_tms import DOT, SCALE, log_axis, save, trace_steps  # noqa: E402
from summarize import COLOR  # noqa: E402

RTT = 20
TRACES = {"bus_x8": "bus ×8", "bus_x1": "bus ×1", "none": "no limit"}
SCHEMES = ["throughput", "webrtc-push"]
LABEL = {"throughput": "QUIC push", "webrtc-push": "WebRTC push"}
STALL_MS = max(3 * 1000 / 60, 1000 / 60 + 150)
METRICS = {"latency_ms": "mean latency (ms)", "latency_median_ms": "median latency (ms)",
           "latency_p95_ms": "95th pct. latency (ms)", "latency_max_ms": "max latency (ms)",
           "lat_gt150": "latency > 150 ms", "lat_gt200": "latency > 200 ms",
           "shown": "viewpoints shown", "fps": "frames drawn / s",
           "repeat_share": "repeat frames", "repeat_bytes": "repeat bytes",
           "age_gt150": "screen age > 150 ms", "age_gt200": "screen age > 200 ms",
           "age_gt300": "screen age > 300 ms", "age_max_ms": "max screen age (ms)",
           "stalls_min": "stalls / min", "stall_share": "stall share",
           "bandwidth_mbps": "bandwidth used (Mbit/s)", "trace_mbps": "trace (Mbit/s)",
           "frame_kb": "frame (KB)", "new_kb": "new-viewpoint frame (KB)", "repeat_kb": "repeat frame (KB)",
           "psnr": "PSNR (dB)", "ssim": "SSIM", "vmaf": "VMAF", "lpips": "LPIPS",
           "profile_mean": "mean profile", "qp_new": "QP, new viewpoint", "qp_repeat": "QP, repeat",
           "target_mbps": "GCC target (Mbit/s)", "duration_s": "playback (s)"}
TABLES = {"latency": ["latency_ms", "latency_median_ms", "latency_p95_ms", "latency_max_ms", "lat_gt150",
                      "lat_gt200", "shown"],
          "smoothness": ["fps", "age_gt150", "age_gt200", "age_gt300", "age_max_ms", "stalls_min", "stall_share"],
          "rate and quality": ["bandwidth_mbps", "trace_mbps", "target_mbps", "psnr", "ssim", "vmaf", "lpips",
                               "profile_mean"],
          "frames": ["repeat_share", "repeat_bytes", "frame_kb", "new_kb", "repeat_kb", "qp_new", "qp_repeat"]}
PERCENT = {"lat_gt150", "lat_gt200", "shown", "repeat_share", "repeat_bytes", "age_gt150", "age_gt200",
           "age_gt300", "stall_share"}
COLUMNS = ["psnr", "vmaf", "lpips", "latency_ms", "latency_p95_ms", "age_gt200", "stall_share", "fps",
           "bandwidth_mbps"]
PARTS_WT = {"set → message": ("send", "msg"), "message → first chunk": ("msg", "first"),
            "first chunk → drawn": ("first", "drawn")}
PARTS_WEBRTC = {"set → message": ("send", "msg"), "uplink": ("msg", "arrive"),
                "tick wait": ("arrive", "render_start"), "render": ("render_start", "render_end"),
                "encode, send": ("render_end", "pkt"), "network": ("pkt", "recv"), "read": ("recv", "dec")}
JITTER = np.random.default_rng(0)


def run_dir(results, r):
    return os.path.join(results, "runs", r.scene, r.trajectory, r.trace, f"rtt{r.rtt_ms:g}", r.abr, f"rep{r.rep}")


def ndjson(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])


def load(results):
    runs = pd.read_csv(os.path.join(results, "summary", "runs.csv"))
    runs = runs[(runs["rtt_ms"] == RTT) & runs["trace"].isin(TRACES) & runs["abr"].isin(SCHEMES)].copy()
    runs["results"] = results
    return runs


def frames_drawn(d, run):
    """Every frame drawn in playback, in order: drawn (ms since t0), i, bytes."""
    lo, hi = run["start_ms"], run["end_ms"]
    if "wt" in run:
        fr = ndjson(os.path.join(d, "wt_frames.ndjson"))
        fr = fr[fr["drawn"].notna()][["drawn", "i", "bytes"]]
    else:
        fr = ndjson(os.path.join(d, "webrtc_frames.ndjson")).rename(columns={"dec": "drawn"})
        enc = ndjson(os.path.join(d, "sender_encoded.ndjson"))[["rtp_ts", "bytes", "qp"]]
        fr = fr[fr["i"].notna()].merge(enc, left_on="rtp", right_on="rtp_ts", how="left")[
            ["drawn", "i", "bytes", "qp"]]
    fr = fr[(fr["drawn"] >= lo) & (fr["drawn"] <= hi)].sort_values("drawn").reset_index(drop=True)
    fr["repeat"] = fr["i"].eq(fr["i"].shift())
    return fr


def screen_age(vp, fr, lo, hi):
    """Time since the viewpoint of the frame on screen was set, every ms:
    its shares and maximum, and the samples."""
    t = np.arange(lo, hi, 1.0)
    k = np.searchsorted(fr["drawn"].to_numpy(), t, side="right") - 1
    set_at = vp.set_index("i")["send"]
    age = t[k >= 0] - set_at.reindex(fr["i"].to_numpy()[k[k >= 0]].astype(int)).to_numpy()
    age = age[np.isfinite(age)]   # a warm-up frame still on screen
    return {"age_gt150": (age > 150).mean(), "age_gt200": (age > 200).mean(), "age_gt300": (age > 300).mean(),
            "age_max_ms": age.max()}, age


def measures(d, run, q):
    vp = ndjson(os.path.join(d, "frames.ndjson"))
    if "drawn" not in vp:
        vp["drawn"] = np.nan
    own = vp[vp["drawn"].notna() & (vp["error"].isna() if "error" in vp else True)]
    lat = own["drawn"] - own["send"]
    fr = frames_drawn(d, run)
    lo, hi = run["start_ms"], run["end_ms"]
    span = (hi - lo) / 1e3
    gaps = np.diff(fr["drawn"].to_numpy())
    stall = gaps[gaps > STALL_MS]
    rep = fr["repeat"]
    shares, age = screen_age(vp, fr, lo, hi)
    x = {"latency_ms": lat.mean(), "latency_median_ms": lat.median(), "latency_p95_ms": lat.quantile(0.95),
         "latency_max_ms": lat.max(), "lat_gt150": (lat > 150).mean(), "lat_gt200": (lat > 200).mean(),
         "shown": len(own) / len(vp), "fps": len(fr) / span,
         "repeat_share": rep.mean(), "repeat_bytes": fr.loc[rep, "bytes"].sum() / fr["bytes"].sum(),
         "frame_kb": fr["bytes"].mean() / 1e3, "new_kb": fr.loc[~rep, "bytes"].mean() / 1e3,
         "repeat_kb": fr.loc[rep, "bytes"].mean() / 1e3,
         "qp_new": fr.loc[~rep, "qp"].mean() if "qp" in fr else np.nan,
         "qp_repeat": fr.loc[rep, "qp"].mean() if "qp" in fr else np.nan,
         "stalls_min": len(stall) / (span / 60), "stall_share": stall.sum() / 1e3 / span, **shares}
    parts = PARTS_WT if "wt" in run else PARTS_WEBRTC
    for k, (a, b) in parts.items():
        x["part " + k] = (own[b] - own[a]).mean() if a in own and b in own else np.nan
    x["read_frames"] = own["recv"].notna().mean() if "recv" in own else np.nan
    for m in ("psnr", "ssim", "vmaf", "lpips"):
        qq = q[q["standin"] == 0] if q is not None and "standin" in q else q
        x[m] = qq[m].mean() if qq is not None and len(qq) else np.nan
    return x, lat.to_numpy(), age


def extend(runs):
    rows, lat = [], []
    for r in runs.itertuples():
        d = run_dir(r.results, r)
        run = json.load(open(os.path.join(d, "run.json")))
        rel = os.path.relpath(d, os.path.join(r.results, "runs"))
        path = os.path.join(r.results, "metrics", rel + ".csv")
        q = pd.read_csv(path) if os.path.exists(path) else None
        x, l, age = measures(d, run, q)
        rows.append(x)
        for kind, v in (("latency", l), ("age", age)):
            lat.append(pd.DataFrame({"trace": r.trace, "abr": r.abr, "rep": r.rep, "kind": kind, "ms": v}))
    ext = pd.DataFrame(rows, index=runs.index)
    runs = runs.drop(columns=[c for c in ext.columns if c in runs.columns]).join(ext)
    return runs, pd.concat(lat, ignore_index=True)


# ---- tables

def fmt(v, m=None):
    if isinstance(v, (float, np.floating, int, np.integer)):
        if np.isnan(v):
            return "–"
        if m in PERCENT:
            return f"{100 * v:.2f} %"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def table(runs, cols, names, agg="mean"):
    out = ["| trace | scheme | runs | " + " | ".join(names) + " |", "|" + "---|" * (len(cols) + 3)]
    for trace, name in TRACES.items():
        for s in SCHEMES:
            a = runs[(runs["trace"] == trace) & (runs["abr"] == s)]
            if len(a):
                vals = [fmt(a[m].max() if m.endswith("max_ms") else getattr(a[m], agg)(), m) if m in a else "–"
                        for m in cols]
                out.append(f"| {name} | {LABEL[s]} | {len(a)} | " + " | ".join(vals) + " |")
    return out + [""]


def tables(runs):
    out = []
    for title, cols in TABLES.items():
        out += [f"### {title}", ""] + table(runs, cols, [METRICS[m] for m in cols])
    out += ["### latency parts (ms, mean of runs)", ""]
    for s, parts in (("throughput", PARTS_WT), ("webrtc-push", PARTS_WEBRTC)):
        cols = ["part " + k for k in parts] + (["read_frames"] if s == "webrtc-push" else [])
        sub = runs[runs["abr"] == s]
        out += table(sub, cols, list(parts) + (["frames with network and read"] if s == "webrtc-push" else []))
    cols = ["latency_ms", "latency_p95_ms", "age_gt200", "psnr", "bandwidth_mbps"]
    out += ["### standard deviation over the repetitions", ""] + table(runs, cols, [METRICS[m] for m in cols], "std")
    return "\n".join(out) + "\n"


# ---- figures

def style(s):
    return {"color": COLOR[s], "ls": "--" if s == "throughput" else "-"}


def fig_cdf(lat, path):
    """Top row latency of the viewpoints shown, bottom row screen age over
    playback time; one column per trace. No title or legend (QUIC dashed,
    WebRTC solid)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for row, kind in enumerate(["latency", "age"]):
        for ax, (trace, name) in zip(axes[row], TRACES.items()):
            for s in SCHEMES:
                v = lat[(lat["trace"] == trace) & (lat["abr"] == s) & (lat["kind"] == kind)]["ms"]
                v = np.sort(v.dropna().to_numpy())
                if len(v):
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), **style(s))
            ax.set_xscale("log")
            ax.set_xlim(left=20)
            log_axis(ax.xaxis, (20, 30, 50, 100, 200, 500, 1000))   # sparse: the QUIC stalls reach 1.7 s
            ax.set_xlabel(f"{name}: {'latency' if kind == 'latency' else 'screen age'} (ms)")
        axes[row, 0].set_ylabel("CDF")
    fig.tight_layout()
    save(fig, path)


def fig_columns(runs, path):
    """One dot per run, black bar = mean."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    A = len(SCHEMES)
    for ax, m in zip(axes.flat, COLUMNS):
        top = 0
        for i, trace in enumerate(TRACES):
            for j, s in enumerate(SCHEMES):
                x = i * (A + 1) + j
                v = runs[(runs["trace"] == trace) & (runs["abr"] == s)][m].dropna()
                ax.scatter(x + (JITTER.random(len(v)) - 0.5) * 0.3, v, s=DOT, color=COLOR[s], zorder=3,
                           marker="D" if s == "webrtc-push" else "o")
                if len(v):
                    ax.hlines(v.mean(), x - 0.4, x + 0.4, color="black", lw=1.2 * SCALE, zorder=4)
                    top = max(top, v.max())
        ax.set_xticks([i * (A + 1) + (A - 1) / 2 for i in range(len(TRACES))], list(TRACES.values()))
        ax.set_ylabel(METRICS[m])
        if m in ("latency_ms", "latency_p95_ms", "age_gt200", "stall_share", "fps", "bandwidth_mbps"):
            ax.set_ylim(0, 1.08 * top or 1)   # room above the 60 frames/s
        if m in PERCENT:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    h = [Line2D([], [], color=COLOR[s], marker="D" if s == "webrtc-push" else "o", lw=0, label=LABEL[s])
         for s in SCHEMES]
    fig.legend(handles=h, loc="upper center", ncol=len(SCHEMES), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def sent_rate(d, run, w0, bins):
    """Mbit/s per bin of playback time: QUIC frame bytes by their first
    chunk, WebRTC encoded bytes by their encode time."""
    if "wt" in run:
        fr = ndjson(os.path.join(d, "wt_frames.ndjson"))
        t, b = (fr["first"] - w0) / 1e3, fr["bytes"]
    else:
        enc = ndjson(os.path.join(d, "sender_encoded.ndjson"))
        t, b = (enc["enc"] - w0) / 1e3, enc["bytes"]
    h, _ = np.histogram(t, bins=bins, weights=b)
    return h * 8 / np.diff(bins) / 1e6


def fig_timeseries(runs, path, rep=1):
    """One repetition each of bus x8 and x1: trace (black) and rate per
    scheme (1 s bins), then the latency of each scheme's viewpoints shown."""
    fig, axes = plt.subplots(1 + len(SCHEMES), 2, figsize=(15, 2.3 * (1 + len(SCHEMES))), sharex="col")
    for col, trace in enumerate(["bus_x8", "bus_x1"]):
        sel = runs[(runs["trace"] == trace) & (runs["rep"] == rep)]
        end, worst, top = 88, 100, []
        for k, s in enumerate(SCHEMES):
            r = sel[sel["abr"] == s]
            if not len(r):
                continue
            r = next(r.itertuples())
            d = run_dir(r.results, r)
            run = json.load(open(os.path.join(d, "run.json")))
            w0 = run["start_ms"]
            if k == 0:
                ts, bw = trace_steps(r.results, run, trace, w0)
                keep = ts <= end
                axes[0, col].step(np.append(ts[keep], end), np.append(bw[keep], bw[keep][-1:]), where="post",
                                  color="black", lw=0.8 * SCALE)
                top += list(bw[(np.append(ts[1:], np.inf) > 0.5) & keep])
            bins = np.arange(0, end + 1, 1.0)
            rate = sent_rate(d, run, w0, bins)
            axes[0, col].step(bins[:-1], rate, where="post", lw=0.8 * SCALE, **style(s))
            top.append(rate.max())
            vp = ndjson(os.path.join(d, "frames.ndjson"))
            own = vp[vp["drawn"].notna()] if "drawn" in vp else vp.iloc[:0]
            ax = axes[1 + k, col]
            ax.scatter((own["send"] - w0) / 1e3, own["drawn"] - own["send"], s=1.5, color=COLOR[s])
            ax.set_yscale("log")
            ax.set_ylabel(f"{LABEL[s]}\nlatency (ms)")
            worst = max(worst, (own["drawn"] - own["send"]).max())
        for ax in axes[1:, col]:   # one latency range per column
            ax.set_ylim(20, 1.3 * worst)
            log_axis(ax.yaxis, (20, 50, 100, 200, 500, 1000, 2000))
        axes[0, col].set_ylim(0, 1.1 * max(top) if top else 1)
        axes[0, col].set_ylabel("Mbit/s")
        axes[0, col].set_title(TRACES[trace])
        axes[-1, col].set_xlabel("time (s)")
        axes[-1, col].set_xlim(-0.5, end)
    fig.tight_layout()
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wt", required=True, help="results of exp/wt.toml")
    ap.add_argument("--webrtc", required=True, help="results of exp/webrtc_push.toml")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.webrtc, "figs", "compare-push")
    os.makedirs(out, exist_ok=True)
    runs = pd.concat([load(os.path.abspath(a.wt)), load(os.path.abspath(a.webrtc))], ignore_index=True)
    runs.loc[runs["abr"] == "webrtc-push", "profile_mean"] = np.nan   # always full size
    runs, lat = extend(runs)
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    open(os.path.join(out, "tables.md"), "w").write(tables(runs))
    fig_cdf(lat, os.path.join(out, "latency_cdf.pdf"))
    fig_columns(runs, os.path.join(out, "columns.pdf"))
    fig_timeseries(runs, os.path.join(out, "timeseries.pdf"))
    print(open(os.path.join(out, "tables.md")).read())
    print(f"compare_push: {len(runs)} runs, output in {out}")


if __name__ == "__main__":
    main()
