#!/usr/bin/env python
"""Compares the push modes under deep dips of the bandwidth
(exp/push_dips.toml): QUIC (/player-wt) and WebRTC push, with and without
MAE, with GCC as upstream and with its increase raised (--gcc_ai=4
--gcc_mi=1.16). Five traces, RTT 20 ms, 1280x720, 3 repetitions.

Per run, the measures of exp/compare_push.py, and with the seconds of
playback split by their trace bandwidth: below 0.35 Mbit/s (the WebRTC
floor: x264 at QP 51 and the rest of the downlink), 0.35 to 1.45 Mbit/s
(the QUIC floor: p3 at 60 frames/s) and above. In each class:
- sent / trace: the rate sent (as compare_push.sent_rate) over the trace
  bandwidth, both summed over the seconds of the class;
- screen age > 200 ms: share of the class's time;
- new viewpoints / s: frames the sender rendered from a viewpoint it had
  not rendered before, per second. The viewpoint messages go up over SCTP
  (WebRTC) or QUIC, whose acknowledgements come down the shaped link, so
  this falls when the downlink is blocked;
- WebRTC: share of frames encoded at QP 51 and encoded bytes over GCC's
  target (availableOutgoingBitrate); QUIC: share of frames at p3.
For GCC, target / trace over the playback (sums over the seconds), and the
seconds whose mean target is below 0.35 Mbit/s with the share of their
frames at QP 51 (x264's floor reached because of the target, whatever the
trace). After each dip (a run of seconds below 0.35 Mbit/s): the time until
GCC's target is at 80 % of the second's trace bandwidth again, and until
the screen age is below 200 ms again (nan if that did not happen before the
next dip or the end of playback); target / trace in the 10 s after it.

  python exp/compare_dips.py exp/results [--prefix dips-] [--out DIR]

Reads <results>/<prefix><group> (_ in a group's name as -); writes runs.csv, dips.csv, tables.md and the
figures (PDF and PNG) to --out (default <results>/dips-compare).
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

from compare_push import METRICS, PERCENT, TABLES, frames_drawn, measures, ndjson, run_dir, sent_rate  # noqa: E402
from compare_tms import DOT, SCALE, log_axis, save, trace_steps  # noqa: E402

RTT = 20
TRACES = {"bus_x8": "bus ×8", "train_x8_s147": "train ×8", "car_x8_s198": "car ×8",
          "nyc-3109898-bus_x1_s370": "NYC 3109898", "nyc-3114405-bus_x1_s82": "NYC 3114405"}
GROUPS = ["quic", "webrtc", "mae", "gcc", "gcc_mae"]
LABEL = {"quic": "QUIC", "webrtc": "WebRTC", "mae": "WebRTC + MAE", "gcc": "WebRTC, GCC raised",
         "gcc_mae": "WebRTC + MAE, GCC raised"}
COLOR = {"quic": "#e7298a", "webrtc": "#1b9e77", "mae": "#7570b3", "gcc": "#d95f02", "gcc_mae": "#e6ab02"}
MARKER = {"quic": "o", "webrtc": "D", "mae": "s", "gcc": "^", "gcc_mae": "v"}
FLOOR_WEBRTC, FLOOR_QUIC = 0.35, 1.45   # Mbit/s
CLASSES = {"low": f"< {FLOOR_WEBRTC} Mbit/s", "mid": f"{FLOOR_WEBRTC}–{FLOOR_QUIC} Mbit/s",
           "high": f"> {FLOOR_QUIC} Mbit/s"}
DIP_METRICS = {"secs": "seconds", "sent_ratio": "sent / trace", "age_gt200": "screen age > 200 ms",
               "new_vp_s": "new viewpoints / s", "qp51": "QP 51 frames", "enc_target": "encoded / target",
               "p3": "p3 frames"}
METRICS = {**METRICS, "target_ratio": "target / trace", "low_target_secs": "s with target < 0.35",
           "low_target_qp51": "their QP 51 frames", "low_target_qp": "their mean QP", "dips": "dips",
           "target_after10": "target / trace, 10 s after", "target_rec_s": "target back to 0.8 trace (s)",
           "target_rec_max_s": "max (s)", "target_unrec": "not back", "age_rec_s": "screen age back < 200 ms (s)",
           "age_rec_max_s": "max (s)", "age_unrec": "not back"}
PERCENT = PERCENT | {"age_gt200", "qp51", "p3", "low_target_qp51", "target_unrec", "age_unrec"}
COLUMNS = ["psnr", "vmaf", "lpips", "latency_ms", "latency_p95_ms", "shown", "age_gt200", "stall_share",
           "bandwidth_mbps"]
JITTER = np.random.default_rng(0)


def load(root, prefix):
    out = []
    for g in GROUPS:
        res = os.path.join(root, prefix + g.replace("_", "-"))
        path = os.path.join(res, "summary", "runs.csv")
        if not os.path.exists(path):
            print(f"compare_dips: no {path}, group {g} left out")
            continue
        runs = pd.read_csv(path)
        runs = runs[(runs["rtt_ms"] == RTT) & runs["trace"].isin(TRACES)].copy()
        runs["results"], runs["group"] = res, g
        out.append(runs)
    return pd.concat(out, ignore_index=True)


def bin_mean(ts, v, bins):
    """Mean of the step function v (from ts on) over each bin."""
    area = np.concatenate([[0], np.cumsum(v[:-1] * np.diff(ts))])
    return np.diff(np.interp(bins, ts, area)) / np.diff(bins)


def target_series(d, run, w0):
    """GCC's target: s of playback time, Mbit/s (every stats_interval_ms)."""
    tt, tg = [], []
    for l in open(os.path.join(d, "sender_stats.ndjson")):
        r = json.loads(l)
        cp = [x["availableOutgoingBitrate"] for x in r["stats"]
              if x["type"] == "candidate-pair" and x.get("availableOutgoingBitrate")]
        if cp:
            tt.append((r["t_wall"] - run["t0"] * 1e3 - w0) / 1e3)
            tg.append(max(cp) / 1e6)
    return np.array(tt), np.array(tg)


def age_series(vp, fr, lo, hi):
    """Screen age (compare_push.screen_age) every ms: s of playback, ms."""
    t = np.arange(lo, hi, 1.0)
    k = np.searchsorted(fr["drawn"].to_numpy(), t, side="right") - 1
    set_at = vp.set_index("i")["send"]
    age = np.full(len(t), np.nan)
    age[k >= 0] = t[k >= 0] - set_at.reindex(fr["i"].to_numpy()[k[k >= 0]].astype(int)).to_numpy()
    return (t - lo) / 1e3, age


def episodes(mask):
    """[(start, length)] of the runs of True."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - start))
            start = None
    return out


def first_after(t, ok, a, b):
    """Time from a to the first t in [a, b) with ok, or nan."""
    k = np.flatnonzero(ok & (t >= a) & (t < b))
    return t[k[0]] - a if len(k) else np.nan


def new_viewpoints(d, run, w0):
    """s of playback time of each frame the sender rendered from a viewpoint
    it had not rendered before."""
    if "wt" in run:
        fr = ndjson(os.path.join(d, "wt_frames.ndjson"))
        t = fr["ts"]
    else:
        fr = ndjson(os.path.join(d, "sender_frames.ndjson"))
        t = fr["t_render_start"] - run["t0"] * 1e3
    new = fr["i"] > fr["i"].cummax().shift(fill_value=-1)
    return ((t[new] - w0) / 1e3).to_numpy()


def dips(r, d, run):
    """The class statistics of the run, and one row per dip."""
    lo, hi = run["start_ms"], run["end_ms"]
    w0, webrtc = lo, "wt" not in run
    bins = np.arange(0, np.floor((hi - lo) / 1e3) + 1)
    ts, bw = trace_steps(r.results, run, r.trace, w0)
    bw_s = bin_mean(ts, bw, bins)
    cls = np.where(bw_s < FLOOR_WEBRTC, "low", np.where(bw_s < FLOOR_QUIC, "mid", "high"))
    sent = sent_rate(d, run, w0, bins)
    vp = ndjson(os.path.join(d, "frames.ndjson"))
    ta, age = age_series(vp, frames_drawn(d, run), lo, hi)
    age_cls = cls[np.clip(ta.astype(int), 0, len(cls) - 1)]
    if webrtc:
        enc = ndjson(os.path.join(d, "sender_encoded.ndjson"))
        te = (enc["enc"].to_numpy() - w0) / 1e3
        tt, tg = target_series(d, run, w0)
        tg_s = np.array([tg[(tt >= a) & (tt < a + 1)].mean() if ((tt >= a) & (tt < a + 1)).any() else np.nan
                         for a in bins[:-1]])
    else:
        wf = ndjson(os.path.join(d, "wt_frames.ndjson"))
        te = (wf["first"].to_numpy() - w0) / 1e3
    keep = (te >= 0) & (te < bins[-1])
    frame_cls = cls[te[keep].astype(int)]
    tv = new_viewpoints(d, run, w0)
    nv = np.histogram(tv, bins=bins)[0]
    x = {}
    for c in CLASSES:
        m = cls == c
        x[f"{c} secs"] = int(m.sum())
        x[f"{c} sent_ratio"] = sent[m].sum() / bw_s[m].sum() if m.any() and bw_s[m].sum() > 0 else np.nan
        a = age[age_cls == c]
        a = a[np.isfinite(a)]
        x[f"{c} age_gt200"] = (a > 200).mean() if len(a) else np.nan
        x[f"{c} new_vp_s"] = nv[m].mean() if m.any() else np.nan
        f = frame_cls == c
        if webrtc:
            x[f"{c} qp51"] = (enc["qp"].to_numpy()[keep][f] == 51).mean() if f.any() else np.nan
            t = tg_s[m]
            x[f"{c} enc_target"] = sent[m].sum() / np.nansum(t) if m.any() and np.nansum(t) > 0 else np.nan
        else:
            x[f"{c} p3"] = (wf["profile"].to_numpy()[keep][f] == 3).mean() if f.any() else np.nan
    if webrtc:
        x["target_ratio"] = np.nansum(tg_s) / bw_s[np.isfinite(tg_s)].sum()
        low = tg_s < FLOOR_WEBRTC
        x["low_target_secs"] = int(low.sum())
        f = low[te[keep].astype(int)]
        qp = enc["qp"].to_numpy()[keep][f]
        x["low_target_qp51"] = (qp == 51).mean() if len(qp) else np.nan
        x["low_target_qp"] = qp.mean() if len(qp) else np.nan
    rows = []
    eps = episodes(cls == "low")
    for k, (a, n) in enumerate(eps):
        end = a + n
        nxt = eps[k + 1][0] if k + 1 < len(eps) else bins[-1]
        row = {"group": r.group, "trace": r.trace, "rep": r.rep, "dip_start_s": a, "dip_s": n,
               "until_next_s": nxt - end,
               "age_rec_s": first_after(ta, np.nan_to_num(age, nan=np.inf) < 200, end, nxt)}
        if webrtc:
            back = tg >= 0.8 * bw_s[np.clip(tt.astype(int), 0, len(bw_s) - 1)]
            row["target_rec_s"] = first_after(tt, back, end, nxt)
            k = np.arange(end, min(end + 10, nxt, len(bw_s))).astype(int)
            row["target_after10"] = np.nansum(tg_s[k]) / bw_s[k].sum() if len(k) and bw_s[k].sum() > 0 else np.nan
        rows.append(row)
    return x, rows


def extend(runs):
    rows, lat, dip_rows = [], [], []
    for r in runs.itertuples():
        d = run_dir(r.results, r)
        run = json.load(open(os.path.join(d, "run.json")))
        rel = os.path.relpath(d, os.path.join(r.results, "runs"))
        path = os.path.join(r.results, "metrics", rel + ".csv")
        q = pd.read_csv(path) if os.path.exists(path) else None
        x, l, age = measures(d, run, q)
        y, dr = dips(r, d, run)
        rows.append({**x, **y})
        dip_rows += dr
        for kind, v in (("latency", l), ("age", age)):
            lat.append(pd.DataFrame({"trace": r.trace, "group": r.group, "rep": r.rep, "kind": kind, "ms": v}))
    ext = pd.DataFrame(rows, index=runs.index)
    runs = runs.drop(columns=[c for c in ext.columns if c in runs.columns]).join(ext)
    dip = pd.DataFrame(dip_rows)
    if len(dip):   # per run: the dips and their recoveries
        for k in ("target_rec_s", "age_rec_s", "target_after10"):
            if k not in dip:
                dip[k] = np.nan
        g = dip.groupby(["group", "trace", "rep"])
        per = pd.DataFrame({"dips": g.size(), "target_after10": g["target_after10"].mean(),
                            "target_rec_s": g["target_rec_s"].mean(),
                            "target_rec_max_s": g["target_rec_s"].max(),
                            "target_unrec": g["target_rec_s"].apply(lambda v: v.isna().mean()),
                            "age_rec_s": g["age_rec_s"].mean(), "age_rec_max_s": g["age_rec_s"].max(),
                            "age_unrec": g["age_rec_s"].apply(lambda v: v.isna().mean())}).reset_index()
        runs = runs.merge(per, on=["group", "trace", "rep"], how="left")
        quic = runs["group"] == "quic"
        runs.loc[quic, ["target_after10", "target_rec_s", "target_rec_max_s", "target_unrec"]] = np.nan
    return runs, pd.concat(lat, ignore_index=True), dip


# ---- tables

def fmt(v, m=None):
    if isinstance(v, (float, np.floating, int, np.integer)):
        if np.isnan(v):
            return "–"
        if m in PERCENT or (m and m.split()[-1] in PERCENT):
            return f"{100 * v:.1f} %"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def table(runs, cols, names, agg="mean"):
    out = ["| trace | group | runs | " + " | ".join(names) + " |", "|" + "---|" * (len(cols) + 3)]
    for trace, name in TRACES.items():
        for g in GROUPS:
            a = runs[(runs["trace"] == trace) & (runs["group"] == g)]
            if len(a):
                vals = [fmt(a[m].max() if m.endswith(("max_ms", "max_s")) else getattr(a[m], agg)(), m)
                        if m in a else "–" for m in cols]
                out.append(f"| {name} | {LABEL[g]} | {len(a)} | " + " | ".join(vals) + " |")
    return out + [""]


def tables(runs):
    out = []
    for title, cols in TABLES.items():
        out += [f"### {title}", ""] + table(runs, cols, [METRICS[m] for m in cols])
    for c, name in CLASSES.items():
        cols = [f"{c} {m}" for m in DIP_METRICS]
        out += [f"### seconds with the trace {name}", ""] + table(runs, cols, list(DIP_METRICS.values()))
    cols = ["target_ratio", "low_target_secs", "low_target_qp51", "low_target_qp", "dips", "target_after10",
            "target_rec_s", "target_rec_max_s", "target_unrec", "age_rec_s", "age_rec_max_s", "age_unrec"]
    out += ["### GCC and the recovery after a dip", ""] + table(runs, cols, [METRICS[m] for m in cols])
    cols = ["latency_ms", "latency_p95_ms", "age_gt200", "psnr", "bandwidth_mbps"]
    out += ["### standard deviation over the repetitions", ""] + table(runs, cols, [METRICS[m] for m in cols], "std")
    return "\n".join(out) + "\n"


# ---- figures

def style(g):
    return {"color": COLOR[g], "ls": "--" if g == "quic" else "-"}


def fig_cdf(lat, path):
    """Top row latency of the viewpoints shown, bottom row screen age over
    playback time; one column per trace. No title or legend (colors as
    COLOR, QUIC dashed)."""
    fig, axes = plt.subplots(2, len(TRACES), figsize=(4.3 * len(TRACES), 7), sharey=True)
    for row, kind in enumerate(["latency", "age"]):
        for ax, (trace, name) in zip(axes[row], TRACES.items()):
            for g in GROUPS:
                v = lat[(lat["trace"] == trace) & (lat["group"] == g) & (lat["kind"] == kind)]["ms"]
                v = np.sort(v.dropna().to_numpy())
                if len(v):
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), **style(g))
            ax.set_xscale("log")
            ax.set_xlim(left=20)
            log_axis(ax.xaxis, (20, 50, 100, 200, 500, 1000, 5000))
            ax.set_xlabel(f"{name}: {'latency' if kind == 'latency' else 'screen age'} (ms)")
        axes[row, 0].set_ylabel("CDF")
    fig.tight_layout()
    save(fig, path)


def fig_columns(runs, path):
    """One dot per run, black bar = mean."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    A = len(GROUPS)
    for ax, m in zip(axes.flat, COLUMNS):
        top = 0
        for i, trace in enumerate(TRACES):
            for j, g in enumerate(GROUPS):
                x = i * (A + 1) + j
                v = runs[(runs["trace"] == trace) & (runs["group"] == g)][m].dropna()
                ax.scatter(x + (JITTER.random(len(v)) - 0.5) * 0.3, v, s=DOT, color=COLOR[g], zorder=3,
                           marker=MARKER[g])
                if len(v):
                    ax.hlines(v.mean(), x - 0.4, x + 0.4, color="black", lw=1.2 * SCALE, zorder=4)
                    top = max(top, v.max())
        ax.set_xticks([i * (A + 1) + (A - 1) / 2 for i in range(len(TRACES))], list(TRACES.values()),
                      fontsize="small")
        ax.set_ylabel(METRICS[m])
        if m in ("latency_ms", "latency_p95_ms", "age_gt200", "stall_share", "bandwidth_mbps", "shown"):
            ax.set_ylim(0, 1.08 * top or 1)
        if m in PERCENT:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    h = [Line2D([], [], color=COLOR[g], marker=MARKER[g], lw=0, label=LABEL[g]) for g in GROUPS]
    fig.legend(handles=h, loc="upper center", ncol=len(GROUPS), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def fig_timeseries(runs, trace, path, rep=1):
    """One repetition of one trace: trace bandwidth per second (black) and
    the rate each group sent; screen age; GCC's target (WebRTC groups); QP of
    the encoded frames, mean per second."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    end = 88
    bins = np.arange(0, end + 1, 1.0)
    drawn_trace = False
    for g in GROUPS:
        r = runs[(runs["trace"] == trace) & (runs["group"] == g) & (runs["rep"] == rep)]
        if not len(r):
            continue
        r = next(r.itertuples())
        d = run_dir(r.results, r)
        run = json.load(open(os.path.join(d, "run.json")))
        w0 = run["start_ms"]
        if not drawn_trace:
            ts, bw = trace_steps(r.results, run, trace, w0)
            bw_s = bin_mean(ts, bw, bins)
            for ax in axes[:3:2]:
                ax.stairs(bw_s, bins, baseline=None, color="black", lw=1.0 * SCALE)
            drawn_trace = True
        axes[0].stairs(sent_rate(d, run, w0, bins), bins, baseline=None, lw=0.8 * SCALE, **style(g))
        ta, age = age_series(ndjson(os.path.join(d, "frames.ndjson")), frames_drawn(d, run), w0, run["end_ms"])
        axes[1].plot(ta[::10], age[::10], lw=0.8 * SCALE, **style(g))
        if g != "quic":
            tt, tg = target_series(d, run, w0)
            axes[2].plot(tt, tg, lw=0.8 * SCALE, **style(g))
            enc = ndjson(os.path.join(d, "sender_encoded.ndjson"))
            te = (enc["enc"] - w0) / 1e3
            qp = enc.groupby(np.floor(te))["qp"].mean()
            qp = qp[(qp.index >= 0) & (qp.index < end)]
            axes[3].plot(qp.index + 0.5, qp.to_numpy(), lw=0.8 * SCALE, **style(g))
    for ax in axes[:3:2]:
        ax.set_yscale("log")
        ax.set_ylim(0.05, None)
        ax.axhline(FLOOR_WEBRTC, color="0.5", lw=0.8, ls=":")
        ax.axhline(FLOOR_QUIC, color="0.5", lw=0.8, ls="-.")
    axes[0].set_ylabel("sent (Mbit/s)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(20, None)
    log_axis(axes[1].yaxis, (20, 50, 100, 200, 500, 1000, 5000, 20000))
    axes[1].set_ylabel("screen age (ms)")
    axes[2].set_ylabel("GCC target (Mbit/s)")
    axes[3].set_ylabel("QP")
    axes[3].set_ylim(0, 52)
    axes[3].set_xlabel("time (s)")
    axes[3].set_xlim(0, end)
    h = [Line2D([], [], label=LABEL[g], **style(g)) for g in GROUPS] + [Line2D([], [], color="black",
                                                                              label="trace")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", help="the directory with <prefix><group>")
    ap.add_argument("--prefix", default="dips-")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "dips-compare")
    os.makedirs(out, exist_ok=True)
    runs = load(os.path.abspath(a.results), a.prefix)
    runs.loc[runs["group"] != "quic", "profile_mean"] = np.nan   # always full size
    runs, lat, dip = extend(runs)
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    dip.to_csv(os.path.join(out, "dips.csv"), index=False)
    open(os.path.join(out, "tables.md"), "w").write(tables(runs))
    fig_cdf(lat, os.path.join(out, "latency_cdf.pdf"))
    fig_columns(runs, os.path.join(out, "columns.pdf"))
    for trace in TRACES:
        if (runs["trace"] == trace).any():
            fig_timeseries(runs, trace, os.path.join(out, f"timeseries_{trace}.pdf"))
    print(open(os.path.join(out, "tables.md")).read())
    print(f"compare_dips: {len(runs)} runs, {len(dip)} dips, output in {out}")


if __name__ == "__main__":
    main()
