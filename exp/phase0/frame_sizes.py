#!/usr/bin/env python
"""JPEG size of every profile along a viewpoint trajectory.

Renders and encodes in-process, the same way routes.render_handler does
(render_image_raw, then JPEG quality max(50, 70 - 5 * 2^p)), without the
network. Run from the repo root with the server stopped, so the GPU is free:

  python exp/phase0/frame_sizes.py --out sizes.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from encoding import encode_jpeg  # noqa: E402
from models import get_model  # noqa: E402
from render import render_image_raw  # noqa: E402

KEYS = ("angle", "elevation", "x", "y", "z", "fx", "fy", "cx", "cy", "width", "height")


def quality(profile):  # routes.py render_handler
    return max(50, 70 - (1 << profile) * 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="train")
    ap.add_argument("--trajectory", default="TestMovements/NTHU/train/user3_train.json")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    events = json.load(open(a.trajectory))
    model = get_model(a.scene)
    for _ in range(3):  # load the model, warm up the kernels
        render_image_raw(*[events[0][k] for k in KEYS], 0, model)

    rows = []
    for i in range(0, len(events), a.stride):
        for p in range(4):
            img, render_ms = render_image_raw(*[events[i][k] for k in KEYS], p, model)
            t = time.perf_counter()
            jpeg = encode_jpeg(img, quality=quality(p))
            rows.append({"frame": i, "profile": p, "width": img.shape[1], "height": img.shape[0],
                         "bytes": len(jpeg), "render_ms": round(render_ms, 2),
                         "encode_ms": round((time.perf_counter() - t) * 1e3, 2)})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"{len(events)} viewpoints, every {a.stride}th: {len(rows) // 4} frames per profile")
    print("profile  size      quality  kB mean  p50   p90   max   render ms  encode ms")
    for p in range(4):
        r = [x for x in rows if x["profile"] == p]
        kb = np.array([x["bytes"] for x in r]) / 1e3
        print(f"{p:7}  {r[0]['width']}x{r[0]['height']:<4}  {quality(p):7}  {kb.mean():7.1f}"
              f"  {np.percentile(kb, 50):4.1f}  {np.percentile(kb, 90):4.1f}  {kb.max():4.1f}"
              f"  {np.mean([x['render_ms'] for x in r]):9.1f}  {np.mean([x['encode_ms'] for x in r]):9.1f}")


if __name__ == "__main__":
    main()
