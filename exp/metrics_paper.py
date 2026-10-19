#!/usr/bin/env python
"""Image quality computed the way the paper did, next to exp/metrics.py.

The original page logs after every request the ABR's profile (abr.profile,
our abr_profile) and the camera (POST /movement). After the runs,
/materialize_sampled_frames re-renders every 12th logged viewpoint at that
profile as a JPEG of the streaming quality (original_jpeg) and at profile 0
as PNG (gt_png), and scripts/compute_metrics.py compares the two folders.
Here, per run:

- re-render the viewpoints i % stride == 0 at abr_profile with render.py and
  encoding.py, as routes.render_original_and_gt_for_movement does; gt_png
  links to the images of render_gt.py, which are the same profile-0 PNGs;
- run scripts/compute_metrics.py unchanged: PSNR and SSIM, OpenCV INTER_AREA
  to enlarge the low profiles, VMAF of the 30 fps sequence with libvmaf;
- VIF, LPIPS and DISTS, which have no code in the repository, with piq on
  the same INTER_AREA-enlarged images.

abr_profile is not always the profile the frame was requested with: an ABR
that changes its profile in endRequest (the Latency ABR does) logs the next
frame's profile. Where the two agree, the re-rendered JPEG should be the
received one byte for byte; same_bytes in the CSV records it.

  python exp/metrics_paper.py RESULTS_DIR [--stride 12] [--ffmpeg PATH] [--jobs N]

Writes <RESULTS_DIR>/metrics_paper/<run path under runs/>.csv, one row per
frame, and keeps the two folders of each run in <RESULTS_DIR>/paper_frames/.
Runs whose CSV is newer than their run.json are skipped.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from common import CAMERA_KEYS, REPO, find_runs, jpeg_quality, load_events
from metrics import FFMPEG, PiqMetrics
from render_gt import INTRINSICS, gt_dir

sys.path.insert(0, os.path.join(REPO, "scripts"))
import compute_metrics as paper  # noqa: E402

FIELDS = ["i", "profile", "sent_profile", "same_bytes", "psnr", "ssim", "vmaf", "vif", "lpips", "dists"]


def rerender(results, r, run, stride, model, render):
    """Writes original_jpeg/ and gt_png/ of one run; rows without metrics."""
    render_image_raw, encode_jpeg = render
    frames = [json.loads(l) for l in open(os.path.join(r["dir"], "frames.ndjson"))]
    frames = [f for f in frames if f["i"] % stride == 0 and "error" not in f]
    cam = {k: run["camera"][k] for k in INTRINSICS}
    events = load_events(run["args"]["trajectory"])
    gt = gt_dir(results, run["args"]["scene"], run["args"]["trajectory"], run["camera"])
    d = os.path.join(results, "paper_frames", r["rel"])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "original_jpeg"))
    os.makedirs(os.path.join(d, "gt_png"))
    rows = []
    for f in frames:
        i, p = f["i"], int(f["abr_profile"])
        ev = dict(events[i], **cam)
        img, _ = render_image_raw(*[ev[k] for k in CAMERA_KEYS], p, model)
        jpg = encode_jpeg(img, quality=jpeg_quality(p))
        open(os.path.join(d, "original_jpeg", f"frame-{i:06d}.jpg"), "wb").write(jpg)
        os.symlink(os.path.join(gt, f"{i:06d}.png"), os.path.join(d, "gt_png", f"frame-{i:06d}.png"))
        same = ""
        if p == f["profile"]:
            sent = open(os.path.join(r["dir"], "frames", f"{i:06d}_p{p}.jpg"), "rb").read()
            same = int(sent == jpg)
        rows.append({"i": i, "profile": p, "sent_profile": f["profile"], "same_bytes": same})
    return d, rows


def compute_metrics_py(d, ffmpeg):
    """{i: (psnr, ssim, vmaf)} from scripts/compute_metrics.py on one run."""
    out = os.path.join(d, "compute_metrics.csv")
    p = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "compute_metrics.py"),
                        "--test", os.path.join(d, "original_jpeg"), "--ref", os.path.join(d, "gt_png"),
                        "--ffmpeg", ffmpeg, "--out_csv", out], capture_output=True, text=True)
    if p.returncode != 0 or "[warn]" in p.stderr:   # it writes nan and goes on when VMAF fails
        raise RuntimeError(f"compute_metrics.py on {d}: {p.stderr.strip()[-500:]}")
    return {int(re.search(r"\d+", row["filename"]).group()): (float(row["psnr"]), float(row["ssim"]),
                                                             float(row["vmaf"]))
            for row in csv.DictReader(open(out))}


def piq_inputs(d, rows):
    """Test and reference images as compute_metrics.py prepares them, in RGB."""
    imgs, refs = [], []
    for row in rows:
        ref = paper.load_bgr(os.path.join(d, "gt_png", f"frame-{row['i']:06d}.png"))
        img = paper.ensure_same_size(ref, paper.load_bgr(os.path.join(d, "original_jpeg",
                                                                      f"frame-{row['i']:06d}.jpg")))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        refs.append(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    return imgs, refs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--ffmpeg", default=FFMPEG)
    ap.add_argument("--jobs", type=int, default=max(1, min(16, os.cpu_count() // 2)),
                    help="compute_metrics.py processes at a time")
    a = ap.parse_args()
    results = os.path.abspath(a.results)

    todo, skipped = [], 0
    for r in find_runs(results):
        run_json = os.path.join(r["dir"], "run.json")
        run = json.load(open(run_json))
        if run["status"] != "ok":
            continue
        out = os.path.join(results, "metrics_paper", r["rel"] + ".csv")
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(run_json):
            skipped += 1
            continue
        todo.append((r, run, out))
    if not todo:
        print(f"metrics_paper: 0 runs computed, {skipped} already there")
        return

    os.chdir(REPO)   # models.py finds the scenes relative to the repo
    sys.path.insert(0, REPO)
    from encoding import encode_jpeg
    from models import get_model
    from render import render_image_raw

    t = time.time()
    models, rendered = {}, []
    for r, run, out in todo:
        scene = run["args"]["scene"]
        models[scene] = models.get(scene) or get_model(scene)
        rendered.append(rerender(results, r, run, a.stride, models[scene], (render_image_raw, encode_jpeg)))
    print(f"metrics_paper: re-rendered {sum(len(x[1]) for x in rendered)} frames of {len(todo)} runs "
          f"({time.time() - t:.0f} s)", flush=True)

    piq_metrics = PiqMetrics()
    with ThreadPoolExecutor(a.jobs) as ex:
        futures = [ex.submit(compute_metrics_py, d, a.ffmpeg) for d, _ in rendered]
        for (r, _, out), (d, rows), fut in zip(todo, rendered, futures):
            t = time.time()
            for name, vals in piq_metrics(*piq_inputs(d, rows)).items():
                for row, v in zip(rows, vals):
                    row[name] = v
            scores = fut.result()
            for row in rows:
                row["psnr"], row["ssim"], row["vmaf"] = scores[row["i"]]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + ".tmp", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            os.replace(out + ".tmp", out)
            same = [row["same_bytes"] for row in rows if row["same_bytes"] != ""]
            means = "  ".join(f"{m} {np.nanmean([row[m] for row in rows]):.3f}" for m in FIELDS[4:])
            print(f"metrics_paper: {r['rel']}: {len(rows)} frames, {len(rows) - len(same)} at another "
                  f"profile, {sum(same)}/{len(same)} byte-identical ({time.time() - t:.0f} s)  {means}",
                  flush=True)
    print(f"metrics_paper: {len(todo)} runs computed, {skipped} already there")


if __name__ == "__main__":
    main()
