#!/usr/bin/env python
"""Checks the output of one client.py run:

- run.json says ok, and frames.ndjson has one complete line per viewpoint;
- every frame was saved, with the size the page reported;
- a few saved frames are byte-identical to the same viewpoint and profile
  rendered and encoded in-process (render_image_raw + encode_jpeg), i.e. the
  frames on disk are exactly what the server sent.

Run from the repo root; the byte check uses the GPU.

  python exp/check_run.py RUN_DIR [RUN_DIR ...] [--samples 8]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())
from common import CAMERA_KEYS, REPO, jpeg_quality  # noqa: E402

FIELDS = ("i", "send", "drawn", "profile", "bytes", "render_ms", "req_start",
          "resp_start", "resp_end", "proto", "abr_thr", "abr_profile", "t_ms")


def check(run_dir, samples, model_cache):
    problems = []
    run = json.load(open(os.path.join(run_dir, "run.json")))
    frames = [json.loads(l) for l in open(os.path.join(run_dir, "frames.ndjson"))]
    traj = os.path.join(REPO, run["trajectory_path"])
    events = json.load(open(traj))
    events = events if isinstance(events, list) else events["events"]
    events = events[:run["viewpoints"]]

    if run["status"] != "ok":
        problems.append(f"status {run['status']}")
    if len(frames) != len(events):
        problems.append(f"{len(frames)} frame lines for {len(events)} viewpoints")
    if [f["i"] for f in frames] != list(range(len(frames))):
        problems.append("frame indices not 0..n-1")
    missing = [(f["i"], k) for f in frames for k in FIELDS if f.get(k) is None]
    if missing:
        problems.append(f"{len(missing)} missing fields, first {missing[:3]}")
    if any(f.get("proto") != "h3" for f in frames):
        problems.append("some /render requests not over h3")

    files = set(os.listdir(os.path.join(run_dir, "frames")))
    sizes_ok = 0
    for f in frames:
        name = f"{f['i']:06d}_p{f.get('profile')}.jpg"
        if name not in files:
            continue
        if os.path.getsize(os.path.join(run_dir, "frames", name)) == f["bytes"]:
            sizes_ok += 1
    if len(files) != len(frames) or sizes_ok != len(frames):
        problems.append(f"{len(files)} files, {sizes_ok} with the reported size, {len(frames)} frames")

    # byte check: one frame of every profile used, the rest random
    from encoding import encode_jpeg
    from models import get_model
    from render import render_image_raw
    model = model_cache.setdefault(run["args"]["scene"], get_model(run["args"]["scene"]))
    rng = random.Random(0)
    good = [f for f in frames if "error" not in f]
    by_profile = {}
    for f in good:
        by_profile.setdefault(f["profile"], []).append(f)
    pick = [rng.choice(v) for v in by_profile.values()]
    pick += rng.sample(good, max(0, min(samples - len(pick), len(good))))
    cam = run["camera"]
    same = 0
    for f in pick:
        ev = dict(events[f["i"]])
        ev.update({k: cam[k] for k in ("fx", "fy", "cx", "cy", "width", "height")})
        img, _ = render_image_raw(*[ev[k] for k in CAMERA_KEYS], f["profile"], model)
        mine = encode_jpeg(img, quality=jpeg_quality(f["profile"]))
        got = open(os.path.join(run_dir, "frames", f"{f['i']:06d}_p{f['profile']}.jpg"), "rb").read()
        same += mine == got
    if same != len(pick):
        problems.append(f"byte check: {same} of {len(pick)} frames identical")

    prof = [run["profiles"].get(str(p), 0) for p in range(4)]
    print(f"{'PASS' if not problems else 'FAIL'}  {run_dir}: {run['args']['abr']}, "
          f"{len(frames)} frames, profiles {prof}, "
          f"byte check {same}/{len(pick)}" + "".join(f"\n      {p}" for p in problems))
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()
    cache = {}
    ok = [check(r, a.samples, cache) for r in a.runs]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
