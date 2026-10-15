#!/usr/bin/env python
"""Headless experiment client: plays one viewpoint trajectory through
/player with one ABR and saves every frame it receives.

Run it inside `rattan link` (the server address comes from $RATTAN_BASE) or
directly with --server 127.0.0.1. Steps:

1. start headless Chromium with the flags HTTP/3 needs;
2. note t0, open /player?modelId=<scene> and wait for the page's own first
   frame (this also loads the model on the server);
3. pick --resolution in the page's Resolution menu and apply it, as the
   paper's experiment start does (applyConfig), and wait for that frame;
4. at t0 + W start the playback loop of exp/runner.js with a fresh ABR;
5. write run.json, frames.ndjson (one line per viewpoint) and
   frames/<index>_p<profile>.jpg.

Times in frames.ndjson are ms since t0.
"""
import argparse
import base64
import json
import os
import platform
import socket
import sys
import time

from playwright.sync_api import sync_playwright

from common import REPO, chromium_args, git_commit

FIRST_FRAME_JS = "() => document.getElementById('stage_placeholder')?.style.display === 'none'"
# The Apply button: the page reads its settings again (cx, cy = half the size)
# and requests the current view at the new size.
APPLY_RESOLUTION_JS = """res => {
    const sel = document.getElementById('resolution');
    sel.value = res;
    if (sel.value !== res) throw new Error(`the Resolution menu has no ${res}`);
    window.TIGAS.applyConfig();
}"""
TIMES = ("send", "req_start", "resp_start", "resp_end", "drawn")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default=os.environ.get("RATTAN_BASE", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=4433)
    ap.add_argument("--scene", default="train")
    ap.add_argument("--trajectory", required=True, help="path, or relative to TestMovements/")
    ap.add_argument("--abr", required=True, choices=["simple", "l2a", "lol+"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=10.0, help="W: playback starts at t0 + W s")
    ap.add_argument("--resolution", default="800x600",
                    help="WxH from the page's Resolution menu (800x600 is its default)")
    ap.add_argument("--max-frames", type=int, help="play only the first N viewpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose-console", action="store_true",
                    help="keep the ABRs' console.log (off by default)")
    a = ap.parse_args()

    traj = a.trajectory
    if not os.path.exists(traj):
        traj = os.path.join(REPO, "TestMovements", traj)
    events = json.load(open(traj))
    events = events if isinstance(events, list) else events["events"]
    events = events[:a.max_frames] if a.max_frames else events

    frame_dir = os.path.join(a.out, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    origin = f"{a.server}:{a.port}"
    run = {
        "status": "running",
        "args": vars(a),
        "trajectory_path": os.path.relpath(os.path.abspath(traj), REPO),
        "viewpoints": len(events),
        "tigas_commit": git_commit(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "rattan_env": {k: v for k, v in os.environ.items() if k.startswith("RATTAN_")},
    }

    def save_run():
        json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1)

    def save_frame(_source, i, profile, b64):
        with open(os.path.join(frame_dir, f"{i:06d}_p{profile}.jpg"), "wb") as fh:
            fh.write(base64.b64decode(b64))

    errors = []
    try:
        with sync_playwright() as p:
            args = chromium_args(origin)
            browser = p.chromium.launch(headless=True, args=args)
            run.update(chromium=browser.version, chromium_args=args)
            page = browser.new_context(ignore_https_errors=True).new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.expose_binding("tigasSaveFrame", save_frame)

            t0 = time.time()
            run["t0"] = t0
            page.goto(f"https://{origin}/player?modelId={a.scene}", timeout=30000)
            run["loaded_ms"] = (time.time() - t0) * 1e3
            page.wait_for_function(FIRST_FRAME_JS, timeout=60000, polling=10)
            page.evaluate(APPLY_RESOLUTION_JS, a.resolution)
            page.wait_for_function("() => !window.TIGAS.busy()", timeout=60000, polling=10)
            run["first_frame_ms"] = (time.time() - t0) * 1e3
            if run["first_frame_ms"] > a.warmup * 1e3:
                raise RuntimeError(f"first frame took {run['first_frame_ms']:.0f} ms, "
                                   f"longer than the warm-up W = {a.warmup} s")

            page.add_script_tag(path=os.path.join(REPO, "exp", "runner.js"))
            res = page.evaluate("cfg => window.tigasRun(cfg)", {
                "events": events, "abr": a.abr, "seed": a.seed,
                "startAtMs": (t0 + a.warmup) * 1e3, "quiet": not a.verbose_console})
            browser.close()
    except Exception as e:
        run.update(status="error", error=f"{type(e).__name__}: {e}", page_errors=errors[-20:])
        save_run()
        print(f"client: {run['error']}", file=sys.stderr)
        sys.exit(1)

    t0_ms = t0 * 1e3
    with open(os.path.join(a.out, "frames.ndjson"), "w") as fh:
        for f, ev in zip(res["frames"], events):
            for k in TIMES:
                if k in f:
                    f[k] = round(f[k] - t0_ms, 3)
            f["t_ms"] = ev.get("tMs")
            fh.write(json.dumps(f) + "\n")
    frames = res["frames"]
    failed = sum("error" in f for f in frames)
    run.update(
        status="ok" if failed == 0 else "error",
        start_ms=res["start"] - t0_ms, end_ms=res["end"] - t0_ms,
        camera=res["camera"], frames=len(frames), failed_frames=failed,
        profiles={p: sum(f.get("profile") == p for f in frames) for p in range(4)},
        page_errors=errors[-20:])
    save_run()
    print(f"client: {a.abr} {len(frames)} frames in {(res['end'] - res['start']) / 1e3:.1f} s, "
          f"profiles {list(run['profiles'].values())}, failed {failed}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
