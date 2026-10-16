#!/usr/bin/env python
"""Figures and tables of the ablations of exp/ablate.toml, from the
summaries of exp/summarize.py.

The baseline is bus x8 at RTT 20 ms in --base (r720 [ghent], 9 runs per
ABR). The groups in --ablate are told apart by the trace label: bus_x8 (the
baseline again with the new code), bus_x8_phi (A1), bus_x8_S1 to S3 (A2),
bus_x8_tclo (A3). A4 compares bus_x8 and bus_x1 of --base at every RTT.

For each group, ABR and metric: the difference of the means to the baseline
and the ratio of the standard deviations, each with a bootstrap 95 %
interval (10000 resamples of the runs of both groups). For A1 and A2 also
P_flip: over all pairs of one run of ABR A and one of ABR B, the share whose
order differs from the order of the two baseline means.

  python exp/ablation.py --base exp/results/r720 --ablate exp/results/ablate

Writes <ablate>/figs/ablation/: the figures (PDF) and tables.md.
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
B = 10000
RNG = np.random.default_rng(0)


# ---- statistics

def boot(x, y, stat):
    """Bootstrap 95 % interval of stat(x, y), resampling x and y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs = x[RNG.integers(0, len(x), (B, len(x)))]
    ys = y[RNG.integers(0, len(y), (B, len(y)))]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = stat(xs, ys)
    v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)


def mean_diff(x, y):
    return x.mean(axis=-1) - y.mean(axis=-1)


def std_ratio(x, y):
    return x.std(axis=-1, ddof=1) / y.std(axis=-1, ddof=1)


def compare(base, grp, metrics=TABLE):
    """One row per ABR and metric: means, stds, difference, ratio, intervals."""
    rows = []
    for abr in ABRS:
        for m in metrics:
            x = grp[grp["abr"] == abr][m].dropna().to_numpy()
            y = base[base["abr"] == abr][m].dropna().to_numpy()
            if len(x) < 2 or len(y) < 2:
                continue
            sx, sy = x.std(ddof=1), y.std(ddof=1)
            rows.append({"ABR": LABEL[abr], "metric": m, "n": len(x), "baseline": y.mean(), "ablation": x.mean(),
                         "diff": x.mean() - y.mean(), "diff 95%": tuple(boot(x, y, mean_diff)),
                         "std base": sy, "std abl": sx, "ratio": sx / sy if sy > 0 else np.nan,
                         "ratio 95%": tuple(boot(x, y, std_ratio))})
    return pd.DataFrame(rows)


def p_flip(a, b, ref):
    """Share of pairs (run of A, run of B) ordered unlike sign ref."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.mean(np.sign(np.subtract.outer(np.asarray(a), np.asarray(b))) != ref))


def flips(base, groups, metrics=MAIN):
    rows = []
    for m in metrics:
        for a, b in PAIRS:
            ref = np.sign(base[base["abr"] == a][m].mean() - base[base["abr"] == b][m].mean())
            row = {"metric": m, "pair": f"{LABEL[a]} vs {LABEL[b]}"}
            for name, g in [("baseline", base)] + list(groups.items()):
                row[name] = p_flip(g[g["abr"] == a][m].dropna(), g[g["abr"] == b][m].dropna(), ref)
            rows.append(row)
    return pd.DataFrame(rows)


def ranking(df, m):
    """ABRs from the lowest to the highest mean of m."""
    means = df.groupby("abr")[m].mean().sort_values()
    return " < ".join(LABEL[a] for a in means.index)


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

def columns(ax, groups, m, names):
    """Per ABR one column of points per group, the mean as a bar."""
    G = len(groups)
    ticks, labels = [], []
    for i, abr in enumerate(ABRS):
        for j, (name, g) in enumerate(zip(names, groups)):
            x = i * (G + 1) + j
            v = g[g["abr"] == abr][m].dropna()
            jit = (RNG.random(len(v)) - 0.5) * 0.3
            ax.scatter(x + jit, v, s=12, color=COLOR[abr], alpha=1.0 if j else 0.5,
                       marker="o" if j else "s", zorder=3)
            if len(v):
                ax.hlines(v.mean(), x - 0.35, x + 0.35, color="black", lw=1.2, zorder=4)
            ticks.append(x)
            labels.append(name)
        ax.annotate(LABEL[abr], ((i * (G + 1) + (G - 1) / 2), -0.16), xycoords=("data", "axes fraction"),
                    ha="center", va="top")
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_ylabel(METRICS[m])


def fig_columns(groups, names, path, metrics=MAIN):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.2))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        columns(ax, groups, m, names)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_bw_scatter(base, a1, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, m in zip(axes, ["profile_mean", "psnr"]):
        for abr in ABRS:
            for g, mk, label in ((base, "x", "baseline"), (a1, "o", "A1")):
                v = g[g["abr"] == abr]
                ax.scatter(v["trace_mbps"], v[m], s=16, marker=mk, color=COLOR[abr],
                           label=f"{LABEL[abr]}, {label}")
        ax.set_xlabel("trace bandwidth during playback (Mbit/s)")
        ax.set_ylabel(METRICS[m])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def fig_session_order(groups, path):
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 2.3 * len(groups)), sharex=True)
    for ax, (name, g) in zip(np.atleast_1d(axes), groups.items()):
        g = g.sort_values("session_index")
        tw = ax.twinx()
        tw.plot(g["session_index"], g["trace_mbps"], color="grey", lw=1)
        tw.set_ylabel("trace (Mbit/s)", color="grey")
        for abr in ABRS:
            v = g[g["abr"] == abr]
            ax.scatter(v["session_index"], v["profile_mean"], s=14, color=COLOR[abr], zorder=3)
            if len(v):
                ax.annotate(LABEL[abr], (v["session_index"].mean(), 1.02), xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=8, color=COLOR[abr])
        ax.set_ylabel(f"{name}: mean profile")
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("run in the session")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cdf(fb, ft, path):
    """Per ABR: CDF of the per-frame latency, baseline black, tc + lo red."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
    for ax, abr in zip(axes, ABRS):
        for f, color in ((fb, "black"), (ft, "#d62728")):
            v = np.sort(f[f["abr"] == abr]["latency_ms"].dropna().to_numpy())
            if len(v):
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=1)
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 200, 500, 1000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"{LABEL[abr]}: latency per frame (ms)")
    axes[0].set_ylabel("CDF")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_bars(groups, names, path, metrics=("latency_ms", "profile_mean", "psnr", "switches")):
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3))
    G = len(groups)
    for ax, m in zip(axes, metrics):
        ticks, labels = [], []
        for i, abr in enumerate(ABRS):
            for j, (name, g) in enumerate(zip(names, groups)):
                x = i * (G + 1) + j
                v = g[g["abr"] == abr][m].dropna()
                ax.bar(x, v.mean(), yerr=v.std(ddof=1), width=0.8, color=COLOR[abr], alpha=0.45 if j == 0 else 0.9,
                       capsize=3)
                ticks.append(x)
                labels.append(name)
            ax.annotate(LABEL[abr], (i * (G + 1) + (G - 1) / 2, -0.16), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8)
        ax.set_xticks(ticks, labels, fontsize=7)
        ax.set_ylabel(METRICS[m])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_split(fb, ft, path):
    """Mean wait, download and the rest of the per-frame latency, stacked."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ticks, labels = [], []
    parts = [("wait_ms", "wait for first byte", "#4c72b0"), ("download_ms", "download", "#dd8452"),
             ("rest_ms", "decode and draw", "#bbbbbb")]
    for i, abr in enumerate(ABRS):
        for j, (name, f) in enumerate((("base", fb), ("tc", ft))):
            f = f[f["abr"] == abr]
            vals = {"wait_ms": f["wait_ms"].mean(), "download_ms": f["download_ms"].mean()}
            vals["rest_ms"] = f["latency_ms"].mean() - vals["wait_ms"] - vals["download_ms"]
            x, bottom = i * 3 + j, 0.0
            for k, label, color in parts:
                ax.bar(x, vals[k], bottom=bottom, width=0.8, color=color, label=label if x == 0 else None)
                bottom += vals[k]
            ticks.append(x)
            labels.append(name)
        ax.annotate(LABEL[abr], (i * 3 + 0.5, -0.16), xycoords=("data", "axes fraction"), ha="center", va="top")
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.set_ylabel("mean per frame (ms)")
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def fig_trace_units(path, name="bus", seconds=200):
    v = np.array([float(l) for l in open(os.path.join(REPO, "traces", f"{name}.csv")) if l.strip()][:seconds])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    t = np.arange(len(v) + 1)
    for scale, ls in ((8, "-"), (1, "--")):
        y = np.append(v, v[-1]) * scale / 1000
        ax.step(t, y, where="post", ls=ls, color="black", lw=0.9)
        ax.annotate(f"×{scale}", (seconds, y[-1]), xytext=(4, 0), textcoords="offset points", va="center")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Mbit/s")
    ax.set_xlim(0, seconds)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_units_rtt(runs, path, metrics=("latency_ms", "profile_mean", "psnr", "switches")):
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3))
    for ax, m in zip(axes, metrics):
        for abr in ABRS:
            for trace, ls in (("bus_x8", "-"), ("bus_x1", "--")):
                g = runs[(runs["abr"] == abr) & (runs["trace"] == trace)].groupby("rtt_ms")[m]
                if not len(g.mean()):
                    continue
                ax.errorbar(g.mean().index, g.mean(), yerr=g.std().fillna(0), ls=ls, marker="o", ms=3,
                            capsize=2, color=COLOR[abr], label=f"{LABEL[abr]} ×{trace[-1]}")
        ax.set_xscale("log")
        rtts = sorted(set(runs["rtt_ms"]))
        ax.set_xticks(rtts, [f"{r:g}" for r in rtts])
        ax.minorticks_off()
        ax.set_xlabel("RTT (ms)")
        ax.set_ylabel(METRICS[m])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path)
    plt.close(fig)


# ---- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="results with the baseline and A4 (r720)")
    ap.add_argument("--ablate", required=True, help="results of exp/ablate.toml")
    ap.add_argument("--rtt", type=float, default=20)
    a = ap.parse_args()
    r720, abl = load(os.path.abspath(a.base)), load(os.path.abspath(a.ablate))
    out = os.path.join(os.path.abspath(a.ablate), "figs", "ablation")
    os.makedirs(out, exist_ok=True)
    base = r720[(r720["trace"] == "bus_x8") & (r720["rtt_ms"] == a.rtt)]
    abl = abl[abl["rtt_ms"] == a.rtt]
    grp = {t: abl[abl["trace"] == t] for t in sorted(set(abl["trace"]))}
    text = [f"# Ablation tables\n\nBaseline: {a.base}, bus_x8, RTT {a.rtt:g} ms, "
            f"{base.groupby('abr').size().to_dict()} runs.\n"]

    # baseline again with the new code: its means within the range of the baseline runs?
    if "bus_x8" in grp:
        rows = []
        for abr in ABRS:
            for m in TABLE + ["wait_ms", "download_ms"]:
                y, x = base[base["abr"] == abr][m], grp["bus_x8"][grp["bus_x8"]["abr"] == abr][m]
                if len(x) and len(y):
                    rows.append({"ABR": LABEL[abr], "metric": m, "baseline min": y.min(), "baseline max": y.max(),
                                 "new mean": x.mean(), "within": bool(y.min() <= x.mean() <= y.max())})
        text += ["## Baseline again\n", md(pd.DataFrame(rows))]

    # A1
    if "bus_x8_phi" in grp:
        a1 = grp["bus_x8_phi"]
        fig_columns([base, a1], ["base", "A1"], os.path.join(out, "a1_1_columns.pdf"))
        fig_bw_scatter(base, a1, os.path.join(out, "a1_2_bandwidth.pdf"))
        text += ["## A1 (playback starts phi s into the trace)\n", md(compare(base, a1)),
                 "\nP_flip\n\n", md(flips(base, {"A1": a1}))]

    # A2
    sess = {s: grp[f"bus_x8_{s}"] for s in SESSIONS if f"bus_x8_{s}" in grp}
    if sess:
        fig_columns([base] + list(sess.values()), ["base"] + list(sess), os.path.join(out, "a2_1_columns.pdf"))
        fig_session_order(sess, os.path.join(out, "a2_2_order.pdf"))
        text.append("## A2 (sessions)\n")
        for s, g in sess.items():
            text += [f"\n### {s}\n\n", md(compare(base, g))]
        allg = pd.concat(sess.values())
        text += ["\n### S1 to S3 together\n\n", md(compare(base, allg)), "\nP_flip\n\n",
                 md(flips(base, dict(sess, **{"S1-S3": allg})))]
        rank = [{"metric": m, "baseline": ranking(base, m), **{s: ranking(g, m) for s, g in sess.items()}}
                for m in MAIN]
        text += ["\nOrder of the ABR means (lowest first)\n\n", md(pd.DataFrame(rank))]
        bw = pd.concat([g.assign(session=s) for s, g in sess.items()]).groupby(["session", "abr"])["trace_mbps"]
        text += ["\nTrace bandwidth during the runs (Mbit/s)\n\n",
                 md(bw.agg(["mean", "std", "min", "max"]).reset_index())]

    # A3
    if "bus_x8_tclo" in grp:
        a3 = grp["bus_x8_tclo"]
        fb, ft = frames_of(base), frames_of(a3)
        fig_cdf(fb, ft, os.path.join(out, "a3_1_latency_cdf.pdf"))
        fig_bars([base, a3], ["base", "tc"], os.path.join(out, "a3_2_bars.pdf"))
        fig_split(fb, ft, os.path.join(out, "a3_3_latency_split.pdf"))
        pct = []
        for name, f in (("baseline", fb), ("tc + lo", ft)):
            for abr in ABRS:
                v = f[f["abr"] == abr]["latency_ms"]
                pct.append({"setting": name, "ABR": LABEL[abr], "p50": v.quantile(0.5), "p90": v.quantile(0.9),
                            "p99": v.quantile(0.99), "max": v.max()})
        text += ["## A3 (tc + lo)\n", md(compare(base, a3, TABLE + ["wait_ms", "download_ms"])),
                 "\nPer-frame latency (ms)\n\n", md(pd.DataFrame(pct)),
                 "\nRate loop and tc counters\n\n", md(tc_counters(a3))]

    # A4
    fig_trace_units(os.path.join(out, "a4_1_trace.pdf"))
    units = r720[r720["trace"].isin(["bus_x8", "bus_x1"])]
    if len(units):
        fig_units_rtt(units, os.path.join(out, "a4_2_rtt.pdf"))
        rows = []
        for rtt in sorted(set(units["rtt_ms"])):
            for m in ("latency_ms", "profile_mean", "psnr", "switches"):
                u = units[units["rtt_ms"] == rtt]
                r8, r1 = ranking(u[u["trace"] == "bus_x8"], m), ranking(u[u["trace"] == "bus_x1"], m)
                rows.append({"RTT (ms)": rtt, "metric": m, "runs": int(u.groupby(["trace", "abr"]).size().min()),
                             "x8": r8, "x1": r1, "same": r8 == r1})
        text += ["## A4 (x8 against x1)\n", md(pd.DataFrame(rows))]

    open(os.path.join(out, "tables.md"), "w").write("\n".join(text))
    print(f"ablation: figures and tables.md in {out}")


if __name__ == "__main__":
    main()
