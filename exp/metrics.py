#!/usr/bin/env python
"""Image quality of the frames each run received, against the reference
images of render_gt.py, on every --stride-th viewpoint.

- A frame of profile p is first enlarged 2^p times with Lanczos, as
  scripts/upscale.py does, but in memory: upscale.py saves the result as a
  JPEG again, which would add a second compression.
- PSNR and SSIM as in scripts/compute_metrics.py (scikit-image, RGB, PSNR
  capped at 100 dB), in parallel over the CPU cores.
- VMAF as in scripts/compute_metrics.py: the sampled frames of a run form a
  30 fps sequence for ffmpeg's libvmaf, compared with the reference sequence.
- VIF (piq.vif_p), LPIPS and DISTS with piq, on the GPU when there is one.

Frames that failed are left out. Runs whose CSV is newer than their
run.json are skipped.

  python exp/metrics.py RESULTS_DIR [--stride 12] [--ffmpeg PATH]

Writes <RESULTS_DIR>/metrics/<run path under runs/>.csv, one row per frame.
"""
import argparse
import csv
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common import find_runs
from render_gt import gt_dir

METRICS = ("psnr", "ssim", "vmaf", "vif", "lpips", "dists")
FFMPEG = os.environ.get("FFMPEG", os.path.expanduser("~/opt/ffmpeg/bin/ffmpeg"))


def received(path, profile):
    im = Image.open(path).convert("RGB")
    f = 1 << profile
    if f > 1:
        im = im.resize((im.width * f, im.height * f), Image.Resampling.LANCZOS)
    return np.asarray(im)


def psnr_ssim(ref, img):
    p = float(peak_signal_noise_ratio(ref, img, data_range=255))
    p = 100.0 if math.isinf(p) or p > 100 else p
    s = float(structural_similarity(ref, img, channel_axis=2, data_range=255))
    return p, s


def frame_cpu(job):
    """Decoding, upscaling, PSNR and SSIM of one frame (in a worker process)."""
    i, img_path, profile, ref_path = job
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    img = received(img_path, profile)
    if img.shape != ref.shape:
        raise RuntimeError(f"frame {i}: {img.shape} after upscaling, reference {ref.shape}")
    return (img, ref, *psnr_ssim(ref, img))


def vmaf(imgs, ref_paths, ffmpeg):
    """Per-frame VMAF of the sequence imgs against the reference PNGs."""
    h, w = imgs[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "dist"))
        os.makedirs(os.path.join(tmp, "ref"))
        for k, (img, ref) in enumerate(zip(imgs, ref_paths)):
            Image.fromarray(img).save(os.path.join(tmp, "dist", f"{k:06d}.png"), compress_level=1)
            os.symlink(ref, os.path.join(tmp, "ref", f"{k:06d}.png"))
        log = os.path.join(tmp, "vmaf.json")
        lavfi = (f"[0:v]scale={w}:{h}:flags=bicubic[d];[1:v]scale={w}:{h}:flags=bicubic[r];"
                 f"[d][r]libvmaf=log_path={log}:log_fmt=json:n_threads={os.cpu_count()}")
        p = subprocess.run([ffmpeg, "-v", "error",
                            "-r", "30", "-i", os.path.join(tmp, "dist", "%06d.png"),
                            "-r", "30", "-i", os.path.join(tmp, "ref", "%06d.png"),
                            "-lavfi", lavfi, "-f", "null", "-"], capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(log):
            raise RuntimeError(f"ffmpeg/libvmaf failed: {p.stderr[-500:]}")
        out = [float("nan")] * len(imgs)
        for fr in json.load(open(log))["frames"]:
            out[fr["frameNum"]] = float(fr["metrics"]["vmaf"])
        return out


class PiqMetrics:
    def __init__(self):
        import piq
        import torch
        self.torch, self.piq = torch, piq
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips = piq.LPIPS(reduction="none").to(self.dev)
        self.dists = piq.DISTS(reduction="none").to(self.dev)

    def __call__(self, imgs, refs, batch=4):
        torch, out = self.torch, {"vif": [], "lpips": [], "dists": []}

        def tensor(a):
            return torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float().div(255).to(self.dev)
        with torch.no_grad():
            for k in range(0, len(imgs), batch):
                x, y = tensor(imgs[k:k + batch]), tensor(refs[k:k + batch])
                out["vif"] += self.piq.vif_p(x, y, data_range=1.0, reduction="none").tolist()
                out["lpips"] += self.lpips(x, y).flatten().tolist()
                out["dists"] += self.dists(x, y).flatten().tolist()
        return out


def run_metrics(results, r, stride, ffmpeg, piq_metrics, pool):
    run = json.load(open(os.path.join(r["dir"], "run.json")))
    frames = [json.loads(l) for l in open(os.path.join(r["dir"], "frames.ndjson"))]
    frames = [f for f in frames if f["i"] % stride == 0 and "error" not in f]
    gt = gt_dir(results, run["args"]["scene"], run["args"]["trajectory"], run["camera"])
    ref_paths = [os.path.join(gt, f"{f['i']:06d}.png") for f in frames]
    jobs = [(f["i"], os.path.join(r["dir"], "frames", f"{f['i']:06d}_p{f['profile']}.jpg"), f["profile"], ref)
            for f, ref in zip(frames, ref_paths)]
    rows, imgs, refs = [], [], []
    for f, (img, ref, p, s) in zip(frames, pool.map(frame_cpu, jobs, chunksize=4)):
        rows.append({"i": f["i"], "profile": f["profile"], "bytes": f["bytes"], "psnr": p, "ssim": s})
        imgs.append(img)
        refs.append(ref)
    for k, v in enumerate(vmaf(imgs, ref_paths, ffmpeg)):
        rows[k]["vmaf"] = v
    for name, vals in piq_metrics(imgs, refs).items():
        for k, v in enumerate(vals):
            rows[k][name] = v
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--ffmpeg", default=FFMPEG)
    a = ap.parse_args()
    results = os.path.abspath(a.results)
    piq_metrics, done, skipped = None, 0, 0
    # workers from a fork server, so they do not inherit the CUDA state of this process
    pool = ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("forkserver"))
    for r in find_runs(results):
        run_json = os.path.join(r["dir"], "run.json")
        if json.load(open(run_json))["status"] != "ok":
            continue
        out = os.path.join(results, "metrics", r["rel"] + ".csv")
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(run_json):
            skipped += 1
            continue
        t = time.time()
        piq_metrics = piq_metrics or PiqMetrics()
        rows = run_metrics(results, r, a.stride, a.ffmpeg, piq_metrics, pool)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["i", "profile", "bytes", *METRICS])
            w.writeheader()
            w.writerows(rows)
        os.replace(out + ".tmp", out)
        done += 1
        means = "  ".join(f"{m} {np.nanmean([x[m] for x in rows]):.3f}" for m in METRICS)
        print(f"metrics: {r['rel']}: {len(rows)} frames ({time.time() - t:.0f} s)  {means}", flush=True)
    print(f"metrics: {done} runs computed, {skipped} already there")


if __name__ == "__main__":
    main()
