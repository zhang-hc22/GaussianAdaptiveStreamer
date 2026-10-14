#!/usr/bin/env python
"""Renders the reference images for the quality metrics: profile 0 (full
resolution) as lossless PNG, every --stride-th viewpoint, in-process with
render.py (no network). They depend only on the scene, the viewpoint and
the camera, so all runs of a trajectory share them.

For every (scene, trajectory) with runs that finished, it uses the camera
those runs recorded (they must all agree, and match --resolution) and
renders the viewpoints up to the longest run. Images already there are kept.

  python exp/render_gt.py RESULTS_DIR [--stride 12] [--resolution 800x600]

Writes <RESULTS_DIR>/gt/<scene>/<trajectory>/<width>x<height>/<index>.png.
"""
import argparse
import json
import os
import sys
import time

from common import CAMERA_KEYS, REPO, find_runs, load_events, traj_name

INTRINSICS = ("fx", "fy", "cx", "cy", "width", "height")


def gt_dir(results, scene, trajectory, camera):
    return os.path.join(results, "gt", scene, traj_name(trajectory),
                        f"{camera['width']}x{camera['height']}")


def groups(results):
    """{(scene, trajectory): (camera, viewpoints)} over the finished runs."""
    out = {}
    for r in find_runs(results):
        run = json.load(open(os.path.join(r["dir"], "run.json")))
        if run["status"] != "ok":
            continue
        key = (run["args"]["scene"], run["args"]["trajectory"])
        cam = {k: run["camera"][k] for k in INTRINSICS}
        if key in out and out[key][0] != cam:
            raise SystemExit(f"{r['rel']}: camera {cam} differs from other runs {out[key][0]}")
        out[key] = (cam, max(run["viewpoints"], out.get(key, (None, 0))[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--resolution", help="WxH the runs must have used")
    a = ap.parse_args()
    results = os.path.abspath(a.results)
    todo = groups(results)

    os.chdir(REPO)   # models.py finds the scenes relative to the repo
    sys.path.insert(0, REPO)
    model = None
    for (scene, trajectory), (cam, n) in sorted(todo.items()):
        if a.resolution and f"{cam['width']}x{cam['height']}" != a.resolution:
            raise SystemExit(f"{scene} {trajectory}: runs used {cam['width']}x{cam['height']}, "
                             f"config says {a.resolution}")
        d = gt_dir(results, scene, trajectory, cam)
        os.makedirs(d, exist_ok=True)
        events = load_events(trajectory)
        idx = [i for i in range(0, n, a.stride) if not os.path.exists(os.path.join(d, f"{i:06d}.png"))]
        t = time.time()
        if idx:
            from encoding import encode_png
            from models import get_model
            from render import render_image_raw
            model = get_model(scene)
        for i in idx:
            ev = dict(events[i], **cam)
            img, _ = render_image_raw(*[ev[k] for k in CAMERA_KEYS], 0, model)
            path = os.path.join(d, f"{i:06d}.png")
            open(path + ".tmp", "wb").write(encode_png(img))
            os.replace(path + ".tmp", path)
        print(f"render_gt: {scene} {trajectory} {cam['width']}x{cam['height']}: "
              f"{len(range(0, n, a.stride))} images, {len(idx)} rendered now ({time.time() - t:.1f} s)")


if __name__ == "__main__":
    main()
