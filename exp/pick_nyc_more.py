"""Reproducibly select five unused NYC traces and write the five-group config.

Run from any directory. No selection depends on experiment outcomes.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO.parent / "mortise/traces/cellular-nyc"
OUT = REPO.parent / "work/figs/tigas-push-nyc-more"
USED = {"3109898-bus", "3114405-bus", "2925703-home1", "3203600-timessquare",
        "3219061-home", "3453943-timessquare", "3552192-bus"}
SEED = 20261021
GROUPS = {"quic": [], "webrtc": [], "mae": ["--enable_mae=true"],
          "mortise": ["--enable_mortise=true"],
          "mortise_mae": ["--enable_mortise=true", "--enable_mae=true"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = sorted(p.name[6:-6] for p in SOURCE.glob("trace-*.trace") if p.name[6:-6] not in USED)
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(candidates, 5, replace=False))
    rows = []
    for name in chosen:
        src = SOURCE / f"trace-{name}.trace"
        ts = np.loadtxt(src, dtype=np.int64)
        bw = np.bincount(ts // 1000) * 1500 * 8 / 1e6
        start = int(rng.integers(0, len(bw) - 100 + 1))
        w = bw[start:start + 87]
        trace = "nyc-" + name
        shutil.copyfile(src, REPO / "traces" / f"{trace}.trace")
        rows.append(dict(name=name, trace=trace, key=f"{trace}_x1_s{start}", start_s=start,
                         length_s=len(bw), mean_mbps=float(w.mean()), min_mbps=float(w.min()),
                         low_seconds=int((w < 0.35).sum()), midlow_seconds=int((w < 1.45).sum()),
                         sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    manifest = dict(seed=SEED, excluded=sorted(USED), candidates=candidates, groups=list(GROUPS), traces=rows)
    (OUT / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    text = '# Five unused NYC traces; seed 20261021. See work/figs/tigas-push-nyc-more/selection.json.\n'
    text += '[server]\nresolution = "1280x720"\nwarmup_s = 10\ntimeout_s = 900\nmetric_stride = 12\npaper_metrics = false\n'
    for group, args in GROUPS.items():
        text += f'\n[{group}]\n'
        if group == "quic":
            text += 'mode = "wt"\nabrs = ["throughput"]\n'
        else:
            text += 'transport = "webrtc"\npacing = "push"\npush_fps = 60\nwarmup_mode = "reverse"\nwarmup_rate = "first"\n'
            if args:
                text += 'sender_args = ' + json.dumps(args) + '\n'
        text += 'event_ms = 40\nscenes = ["train"]\ntrajectories = ["NTHU/train/user3_train.json"]\n'
        text += 'traces = ' + json.dumps([r["trace"] for r in rows]) + '\nscales = [1]\n'
        text += 'trace_start_s = {' + ', '.join(f'{r["trace"]} = {r["start_s"]}' for r in rows) + '}\n'
        text += 'rtt_ms = [20]\nreps = 3\n'
    (REPO / "exp/push_nyc_more.toml").write_text(text)
    print(json.dumps(manifest, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, rows):
        ts = np.loadtxt(SOURCE / f'trace-{r["name"]}.trace', dtype=np.int64)
        bw = np.bincount(ts // 1000) * 1500 * 8 / 1e6
        w = bw[r["start_s"]:r["start_s"] + 87]
        ax.stairs(w, np.arange(88), baseline=None, color="black", lw=1)
        for i in np.flatnonzero(w < 0.35):
            ax.axvspan(i, i + 1, color="0.85", lw=0)
        ax.axhline(0.35, color="tab:brown", lw=1)
        ax.axhline(1.45, color="tab:pink", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(0.07, 80)
        ax.text(.01, .95, f'NYC {r["name"]}, from {r["start_s"]} s', transform=ax.transAxes, va="top", fontsize=10)
    axes[2, 1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mbit/s")
    for ax in (axes[2, 0], axes[1, 1]):
        ax.set_xlabel("time (s)")
        ax.tick_params(labelbottom=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"traces.{ext}", dpi=150)


if __name__ == "__main__":
    main()
