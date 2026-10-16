#!/usr/bin/env python
"""Figures and tables of the ablations of exp/ablate.toml, from the
summaries of exp/summarize.py.

Each ablation puts one of four parts of the experiment back to the way of
the original code ("origin") and keeps the other three as we changed them.
"revised" is our version: bus x8 at RTT 20 ms in --base (r720 [ghent], 9
runs per ABR). The origin runs in --ablate are told apart by the trace
label:

  bus_x8_phi    trace alignment: playback starts phi s into the trace
  bus_x8_S1..S3 restart: 27 runs in one page, the trace not restarted
  bus_x8_tclo   shaper: tc on lo instead of rattan
  bus_x8        revised again with the new code, as a check

The bandwidth unit compares bus_x1 (origin) and bus_x8 (revised) of --base
at every RTT.

For each ablation, ABR and metric: the difference of the means (revised
minus origin) and the ratio of the standard deviations (origin over
revised), each with a bootstrap 95 % interval (10000 resamples of the runs
of both). For trace alignment and restart also P_flip: over all pairs of
one run of ABR A and one of ABR B, the share whose order differs from the
order of the two revised means.

  python exp/ablation.py --base exp/results/r720 --ablate exp/results/ablate

Writes <ablate>/figs/ablation/: the figures (PDF and PNG) and tables.md.
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

from common import REPO  # noqa: E402
from summarize import ABRS, COLOR, LABEL  # noqa: E402

METRICS = {"latency_ms": "mean latency (ms)", "profile_mean": "mean profile", "switches": "switches",
           "psnr": "PSNR (dB)", "ssim": "SSIM", "vmaf": "VMAF", "lpips": "LPIPS",
           "wait_ms": "wait for first byte (ms)", "download_ms": "download (ms)",
           "duration_s": "playback (s)", "trace_mbps": "trace bandwidth (Mbit/s)"}
TABLE = ["latency_ms", "profile_mean", "switches", "psnr", "ssim", "vmaf", "lpips"]
MAIN = ["latency_ms", "profile_mean", "psnr"]
PAIRS = [("simple", "l2a"), ("simple", "lol+"), ("l2a", "lol+")]
SESSIONS = ["S1", "S2", "S3"]
ORIGIN, REVISED = "origin", "revised"
B = 10000
JITTER = np.random.default_rng(0)

# fonts, lines and markers 1.5 times matplotlib's defaults
SCALE = 1.5
plt.rcParams.update({k: plt.rcParams[k] * SCALE for k in (
    "font.size", "lines.linewidth", "lines.markersize", "axes.linewidth", "patch.linewidth",
    "xtick.major.size", "xtick.major.width", "xtick.minor.size", "xtick.minor.width",
    "ytick.major.size", "ytick.major.width", "ytick.minor.size", "ytick.minor.width")})
DOT = 12 * SCALE ** 2   # scatter marker area


# ---- statistics

def boot(x, y, stat):
    """Bootstrap 95 % interval of stat(x, y), resampling x and y."""
    rng = np.random.default_rng(0)
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs = x[rng.integers(0, len(x), (B, len(x)))]
    ys = y[rng.integers(0, len(y), (B, len(y)))]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = stat(xs, ys)
    v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)


def mean_diff(x, y):
    return x.mean(axis=-1) - y.mean(axis=-1)


def std_ratio(x, y):
    return x.std(axis=-1, ddof=1) / y.std(axis=-1, ddof=1)


def compare(revised, origin, metrics=TABLE):
    """One row per ABR and metric: means, stds, difference, ratio, intervals."""
    rows = []
    for abr in ABRS:
        for m in metrics:
            o = origin[origin["abr"] == abr][m].dropna().to_numpy()
            r = revised[revised["abr"] == abr][m].dropna().to_numpy()
            if len(o) < 2 or len(r) < 2:
                continue
            so, sr = o.std(ddof=1), r.std(ddof=1)
            rows.append({"ABR": LABEL[abr], "metric": m, "n origin": len(o), "n revised": len(r),
                         "origin": o.mean(), "revised": r.mean(), "revised - origin": r.mean() - o.mean(),
                         "diff 95%": tuple(boot(r, o, mean_diff)), "std origin": so, "std revised": sr,
                         "std origin / revised": so / sr if sr > 0 else np.nan,
                         "ratio 95%": tuple(boot(o, r, std_ratio))})
    return pd.DataFrame(rows)


def p_flip(a, b, ref):
    """Share of pairs (run of A, run of B) ordered unlike sign ref."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.mean(np.sign(np.subtract.outer(np.asarray(a), np.asarray(b))) != ref))


def flips(revised, groups, metrics=MAIN):
    rows = []
    for m in metrics:
        for a, b in PAIRS:
            ref = np.sign(revised[revised["abr"] == a][m].mean() - revised[revised["abr"] == b][m].mean())
            row = {"metric": m, "pair": f"{LABEL[a]} vs {LABEL[b]}"}
            for name, g in [(REVISED, revised)] + list(groups.items()):
                row[name] = p_flip(g[g["abr"] == a][m].dropna(), g[g["abr"] == b][m].dropna(), ref)
            rows.append(row)
    return pd.DataFrame(rows)


def ranking(df, m):
    """ABRs from the lowest to the highest mean of m."""
    means = df.groupby("abr")[m].mean().sort_values()
    return " < ".join(LABEL[a] for a in means.index)


def order(g):
    """The ABR order of a session, e.g. Latency → L2A → LoL+."""
    return " → ".join(LABEL[a] for a in g.groupby("abr")["session_index"].min().sort_values().index)


# ---- tables as Markdown

def fmt(v):
    if isinstance(v, tuple):
        return f"[{fmt(v[0])}, {fmt(v[1])}]"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def md(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# ---- data

def load(results):
    runs = pd.read_csv(os.path.join(results, "summary", "runs.csv"))
    runs["results"] = results
    return runs


def frames_of(runs):
    """Per-frame latency, wait and download of the given runs."""
    out = []
    for r in runs.itertuples():
        d = os.path.join(r.results, "runs", r.scene, r.trajectory, r.trace, f"rtt{r.rtt_ms:g}", r.abr, f"rep{r.rep}")
        fr = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "frames.ndjson"))])
        if "error" in fr:
            fr = fr[fr["error"].isna()]
        out.append(pd.DataFrame({"abr": r.abr, "trace": r.trace, "latency_ms": fr["drawn"] - fr["send"],
                                 "wait_ms": fr["resp_start"] - fr["send"],
                                 "download_ms": fr["resp_end"] - fr["resp_start"]}))
    return pd.concat(out, ignore_index=True)


def tc_counters(runs):
    """What run.json says about the tc runs: rate loop and tc -s counters."""
    rows = []
    for r in runs.itertuples():
        d = os.path.join(r.results, "runs", r.scene, r.trajectory, r.trace, f"rtt{r.rtt_ms:g}", r.abr, f"rep{r.rep}")
        tc = json.load(open(os.path.join(d, "run.json")))["tc"]
        h, n = tc.get("htb_class", {}), tc.get("netem", {})
        rows.append({"ABR": LABEL[r.abr], "rep": r.rep, "step (ms)": tc["step_ms_mean"],
                     "lag at end (s)": tc["lag_ms"] / 1e3, "Ghent start - W (ms)":
                     tc["ghent_start_ms"] - 1e3 * tc["warmup_s"] if tc["ghent_start_ms"] else np.nan,
                     "packets": h.get("packets", np.nan),
                     "bytes/packet": h["bytes"] / h["packets"] if h.get("packets") else np.nan,
                     "netem drops": n.get("dropped", np.nan), "htb drops": h.get("dropped", np.nan),
                     "tc errors": tc["errors"], "removed": tc["removed"]})
    return pd.DataFrame(rows)


# ---- figures

def save(fig, path):
    """PDF at path and a PNG next to it."""
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=150)
    plt.close(fig)


def kinds(origin, revised):
    """Legend entries: origin hollow, revised filled."""
    return [Line2D([], [], ls="", marker="o", mfc="none", mec="black", label=origin),
            Line2D([], [], ls="", marker="o", color="black", label=revised)]


def abr_ticks(ax, names, G):
    """Per ABR one tick per group (origin left), the ABR name above."""
    ticks = [i * (G + 1) + j for i in range(len(ABRS)) for j in range(G)]
    ax.set_xticks(ticks, names * len(ABRS), rotation=45, ha="right", rotation_mode="anchor")
    for i, abr in enumerate(ABRS):
        ax.annotate(LABEL[abr], (i * (G + 1) + (G - 1) / 2, 1.01), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", color=COLOR[abr], fontweight="bold")


def columns(ax, groups, m):
    """Per ABR one column of points per group, the mean as a black line.
    Origin points hollow, revised filled."""
    G = len(groups)
    for i, abr in enumerate(ABRS):
        for j, (name, g) in enumerate(groups.items()):
            x = i * (G + 1) + j
            v = g[g["abr"] == abr][m].dropna()
            jit = (JITTER.random(len(v)) - 0.5) * 0.3
            filled = name == REVISED
            ax.scatter(x + jit, v, s=DOT, facecolors=COLOR[abr] if filled else "none", edgecolors=COLOR[abr],
                       linewidths=SCALE, zorder=3)
            if len(v):
                ax.hlines(v.mean(), x - 0.35, x + 0.35, color="black", lw=1.2 * SCALE, zorder=4)
    abr_ticks(ax, list(groups), G)
    ax.set_ylabel(METRICS[m])


def fig_columns(groups, path, metrics=MAIN):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        columns(ax, groups, m)
    fig.tight_layout()
    save(fig, path)


def fig_bw_scatter(revised, origin, path, metrics=MAIN):
    """Each run against the trace bandwidth during its playback."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, m in zip(axes, metrics):
        for abr in ABRS:
            for g, filled in ((origin, False), (revised, True)):
                v = g[g["abr"] == abr]
                ax.scatter(v["trace_mbps"], v[m], s=DOT, facecolors=COLOR[abr] if filled else "none",
                           edgecolors=COLOR[abr], linewidths=SCALE)
        ax.set_xlabel("trace bandwidth (Mbit/s)")
        ax.set_ylabel(METRICS[m])
    h = [Line2D([], [], ls="", marker="s", color=COLOR[a], label=LABEL[a]) for a in ABRS]
    fig.legend(handles=h + kinds(ORIGIN, REVISED), loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, path)


def fig_session_order(sessions, revised, path):
    """Mean latency of each run in the order of the session. The revised runs
    of each ABR sit at the places of the origin runs of that ABR."""
    fig, axes = plt.subplots(len(sessions), 1, figsize=(9, 2.6 * len(sessions) + 0.6), sharex=True, sharey=True)
    tws = []
    for ax, g in zip(np.atleast_1d(axes), sessions.values()):
        g = g.sort_values("session_index")
        tw = ax.twinx()
        tws.append(tw)
        tw.plot(g["session_index"], g["trace_mbps"], color="grey")
        for abr in ABRS:
            v = g[g["abr"] == abr]
            r = revised[revised["abr"] == abr].sort_values("rep").head(len(v))
            x = v["session_index"].to_numpy()[:len(r)]
            ax.scatter(v["session_index"], v["latency_ms"], s=DOT, facecolors="none", edgecolors=COLOR[abr],
                       linewidths=SCALE, zorder=3)
            ax.scatter(x, r["latency_ms"], s=DOT, color=COLOR[abr], zorder=3)
            tw.plot(x, r["trace_mbps"], color="grey", ls="--")
            ax.annotate(LABEL[abr], (v["session_index"].mean(), 1.02), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", color=COLOR[abr], fontweight="bold")
        tw.set_ylabel("trace (Mbit/s)", color="grey")
        ax.set_ylabel("latency (ms)")
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)
    for tw in tws[1:]:
        tw.sharey(tws[0])
    np.atleast_1d(axes)[-1].set_xlabel("run in the session")
    h = kinds(ORIGIN, REVISED) + [Line2D([], [], color="grey", label="trace, origin"),
                                  Line2D([], [], color="grey", ls="--", label="trace, revised")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    save(fig, path)


def fig_cdf(fo, fr, path):
    """Per ABR: CDF of the per-frame latency, origin (tc + lo) red dashed, revised (rattan) black."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True)
    for ax, abr in zip(axes, ABRS):
        for f, color, ls in ((fr, "black", "-"), (fo, "#d62728", "--")):
            v = np.sort(f[f["abr"] == abr]["latency_ms"].dropna().to_numpy())
            if len(v):
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, ls=ls)
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 300, 1000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"{LABEL[abr]}: latency (ms)")
    axes[0].set_ylabel("CDF")
    h = [Line2D([], [], color="#d62728", ls="--", label="origin: tc + lo"),
         Line2D([], [], color="black", label="revised: rattan")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, path)


def fig_split(fo, fr, path):
    """Mean wait, download and the rest of the per-frame latency, stacked."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    parts = [("wait_ms", "wait for first byte", "#4c72b0"), ("download_ms", "download", "#dd8452"),
             ("rest_ms", "decode and draw", "#bbbbbb")]
    for i, abr in enumerate(ABRS):
        for j, f in enumerate((fo, fr)):
            f = f[f["abr"] == abr]
            vals = {"wait_ms": f["wait_ms"].mean(), "download_ms": f["download_ms"].mean()}
            vals["rest_ms"] = f["latency_ms"].mean() - vals["wait_ms"] - vals["download_ms"]
            x, bottom = i * 3 + j, 0.0
            for k, label, color in parts:
                ax.bar(x, vals[k], bottom=bottom, width=0.8, color=color, label=label if x == 0 else None)
                bottom += vals[k]
    abr_ticks(ax, [ORIGIN, REVISED], 2)
    ax.set_ylabel("mean per frame (ms)")
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize="small")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, path)


def fig_trace_units(path, name="bus", seconds=200):
    v = np.array([float(l) for l in open(os.path.join(REPO, "traces", f"{name}.csv")) if l.strip()][:seconds])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    t = np.arange(len(v) + 1)
    for scale, ls, label in ((8, "-", f"{REVISED} ×8"), (1, "--", f"{ORIGIN} ×1")):
        y = np.append(v, v[-1]) * scale / 1000
        ax.step(t, y, where="post", ls=ls, color="black")
        ax.annotate(label, (seconds, y[-1]), xytext=(4, 0), textcoords="offset points", va="center")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Mbit/s")
    ax.set_xlim(0, seconds)
    fig.tight_layout()
    save(fig, path)


def fig_units_rtt(runs, path, metrics=("latency_ms", "profile_mean", "psnr", "switches")):
    """Each metric against the RTT, origin (x1) dashed, revised (x8) solid."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.6))
    for ax, m in zip(axes, metrics):
        for abr in ABRS:
            for trace, ls in (("bus_x1", "--"), ("bus_x8", "-")):
                g = runs[(runs["abr"] == abr) & (runs["trace"] == trace)].groupby("rtt_ms")[m]
                if not len(g.mean()):
                    continue
                ax.errorbar(g.mean().index, g.mean(), yerr=g.std().fillna(0), ls=ls, marker="o", ms=3 * SCALE,
                            capsize=2 * SCALE, color=COLOR[abr])
        ax.set_xscale("log")
        rtts = sorted(set(runs["rtt_ms"]))
        ax.set_xticks(rtts, [f"{r:g}" for r in rtts])
        ax.minorticks_off()
        ax.set_xlabel("RTT (ms)")
        ax.set_ylabel(METRICS[m])
    h = [Line2D([], [], color=COLOR[a], label=LABEL[a]) for a in ABRS]
    h += [Line2D([], [], color="black", ls="--", label=f"{ORIGIN} ×1"),
          Line2D([], [], color="black", label=f"{REVISED} ×8")]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, path)


# ---- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="results with the revised runs and the bandwidth unit (r720)")
    ap.add_argument("--ablate", required=True, help="results of exp/ablate.toml")
    ap.add_argument("--rtt", type=float, default=20)
    a = ap.parse_args()
    r720, abl = load(os.path.abspath(a.base)), load(os.path.abspath(a.ablate))
    out = os.path.join(os.path.abspath(a.ablate), "figs", "ablation")
    os.makedirs(out, exist_ok=True)
    rev = r720[(r720["trace"] == "bus_x8") & (r720["rtt_ms"] == a.rtt)]
    abl = abl[abl["rtt_ms"] == a.rtt]
    grp = {t: abl[abl["trace"] == t] for t in sorted(set(abl["trace"]))}
    text = [f"# Ablation tables\n\norigin: one part as in the original code, the other three as revised. "
            f"revised: {a.base}, bus_x8, RTT {a.rtt:g} ms, {rev.groupby('abr').size().to_dict()} runs. "
            f"Differences are revised minus origin, ratios of the standard deviations origin over revised.\n"]

    # revised again with the new code: its means within the range of the revised runs?
    if "bus_x8" in grp:
        rows = []
        for abr in ABRS:
            for m in TABLE + ["wait_ms", "download_ms"]:
                y, x = rev[rev["abr"] == abr][m], grp["bus_x8"][grp["bus_x8"]["abr"] == abr][m]
                if len(x) and len(y):
                    rows.append({"ABR": LABEL[abr], "metric": m, "revised min": y.min(), "revised max": y.max(),
                                 "new code mean": x.mean(), "within": bool(y.min() <= x.mean() <= y.max())})
        text += ["## Revised again with the new code\n", md(pd.DataFrame(rows))]

    # trace alignment
    if "bus_x8_phi" in grp:
        o = grp["bus_x8_phi"]
        fig_columns({ORIGIN: o, REVISED: rev}, os.path.join(out, "align_columns.pdf"))
        fig_bw_scatter(rev, o, os.path.join(out, "align_bandwidth.pdf"))
        text += ["## Trace alignment (origin: playback starts phi s into the trace)\n", md(compare(rev, o)),
                 "\nP_flip\n\n", md(flips(rev, {ORIGIN: o}))]

    # restart for every run
    sess = {s: grp[f"bus_x8_{s}"] for s in SESSIONS if f"bus_x8_{s}" in grp}
    if sess:
        allg = pd.concat(sess.values())
        fig_columns({ORIGIN: allg, REVISED: rev}, os.path.join(out, "restart_columns.pdf"))
        fig_session_order(sess, rev, os.path.join(out, "restart_order.pdf"))
        text.append("## Restart for every run (origin: 27 runs in one page, the trace not restarted)\n")
        for s, g in sess.items():
            text += [f"\n### {s}: {order(g)}\n\n", md(compare(rev, g))]
        text += ["\n### S1 to S3 together\n\n", md(compare(rev, allg)), "\nP_flip\n\n",
                 md(flips(rev, dict({f"{ORIGIN} {s}": g for s, g in sess.items()}, **{f"{ORIGIN} S1-S3": allg})))]
        rank = [{"metric": m, REVISED: ranking(rev, m), **{f"{ORIGIN} {s}": ranking(g, m) for s, g in sess.items()}}
                for m in MAIN]
        text += ["\nOrder of the ABR means (lowest first)\n\n", md(pd.DataFrame(rank))]
        bw = pd.concat([g.assign(session=s) for s, g in sess.items()]).groupby(["session", "abr"])["trace_mbps"]
        text += ["\nTrace bandwidth during the runs (Mbit/s)\n\n",
                 md(bw.agg(["mean", "std", "min", "max"]).reset_index())]

    # shaper
    if "bus_x8_tclo" in grp:
        o = grp["bus_x8_tclo"]
        fr, fo = frames_of(rev), frames_of(o)
        fig_cdf(fo, fr, os.path.join(out, "shaper_cdf.pdf"))
        fig_columns({ORIGIN: o, REVISED: rev}, os.path.join(out, "shaper_columns.pdf"))
        fig_split(fo, fr, os.path.join(out, "shaper_split.pdf"))
        pct = []
        for name, f in ((f"{ORIGIN} (tc + lo)", fo), (f"{REVISED} (rattan)", fr)):
            for abr in ABRS:
                v = f[f["abr"] == abr]["latency_ms"]
                pct.append({"setting": name, "ABR": LABEL[abr], "p50": v.quantile(0.5), "p90": v.quantile(0.9),
                            "p99": v.quantile(0.99), "max": v.max()})
        text += ["## Shaper (origin: tc on lo)\n", md(compare(rev, o, TABLE + ["wait_ms", "download_ms"])),
                 "\nPer-frame latency (ms)\n\n", md(pd.DataFrame(pct)),
                 "\nRate loop and tc counters of the origin runs\n\n", md(tc_counters(o))]

    # bandwidth unit
    fig_trace_units(os.path.join(out, "units_trace.pdf"))
    units = r720[r720["trace"].isin(["bus_x8", "bus_x1"])]
    if len(units):
        fig_units_rtt(units, os.path.join(out, "units_rtt.pdf"))
        rows = []
        for rtt in sorted(set(units["rtt_ms"])):
            for m in ("latency_ms", "profile_mean", "psnr", "switches"):
                u = units[units["rtt_ms"] == rtt]
                r1, r8 = ranking(u[u["trace"] == "bus_x1"], m), ranking(u[u["trace"] == "bus_x8"], m)
                rows.append({"RTT (ms)": rtt, "metric": m, "runs": int(u.groupby(["trace", "abr"]).size().min()),
                             f"{ORIGIN} x1": r1, f"{REVISED} x8": r8, "same": r1 == r8})
        text += ["## Bandwidth unit (origin: x1)\n", md(pd.DataFrame(rows))]

    open(os.path.join(out, "tables.md"), "w").write("\n".join(text))
    print(f"ablation: figures and tables.md in {out}")


if __name__ == "__main__":
    main()
