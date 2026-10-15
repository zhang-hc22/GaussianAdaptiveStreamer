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

With --session the page plays several runs one after another, as the
paper's runExperimentSequentialABRs (templates/player.html:1013) does: the
same page and QUIC connection, a new ABR object per run and --gap-ms between
runs. Only the first run waits for t0 + W. Each run is written to its own
directory as above, all with the same t0; --out gets the session's run.json.
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
    ap.add_argument("--abr", choices=["simple", "l2a", "lol+"], help="required without --session")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--session", help="JSON list of runs [{abr, seed, out, meta}] played one after "
                    "another in the same page")
    ap.add_argument("--gap-ms", type=float, default=150, help="pause between the runs of a session")
    ap.add_argument("--t0", type=float, help="epoch s that times count from, instead of the moment "
                    "the page is opened (run_one.py --shaper tc-lo starts the rate loop at t0)")
    ap.add_argument("--meta", help="JSON object stored in run.json as plan")
    ap.add_argument("--warmup", type=float, default=10.0, help="W: playback starts at t0 + W s")
    ap.add_argument("--resolution", default="800x600",
                    help="WxH from the page's Resolution menu (800x600 is its default)")
    ap.add_argument("--max-frames", type=int, help="play only the first N viewpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose-console", action="store_true",
                    help="keep the ABRs' console.log (off by default)")
    a = ap.parse_args()
    if not a.abr and not a.session:
        ap.error("--abr or --session is required")

    traj = a.trajectory
    if not os.path.exists(traj):
        traj = os.path.join(REPO, "TestMovements", traj)
    events = json.load(open(traj))
    events = events if isinstance(events, list) else events["events"]
    events = events[:a.max_frames] if a.max_frames else events

    if a.session:
        plan = json.load(open(a.session))
    else:
        plan = [{"abr": a.abr, "seed": a.seed, "out": a.out, "meta": json.loads(a.meta) if a.meta else None}]
    for p in plan:
        os.makedirs(os.path.join(p["out"], "frames"), exist_ok=True)
    origin = f"{a.server}:{a.port}"
    common = {
        "trajectory_path": os.path.relpath(os.path.abspath(traj), REPO),
        "viewpoints": len(events),
        "tigas_commit": git_commit(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "rattan_env": {k: v for k, v in os.environ.items() if k.startswith("RATTAN_")},
    }
    runs = [dict(status="running", args=dict(vars(a), abr=p["abr"], seed=p["seed"], out=p["out"]), **common)
            for p in plan]
    for run, p in zip(runs, plan):
        if p.get("meta"):
            run["plan"] = p["meta"]
    session = {"status": "running", "args": vars(a), "runs": [p["out"] for p in plan], **common} \
        if a.session else None

    def save(run, out):
        json.dump(run, open(os.path.join(out, "run.json"), "w"), indent=1)

    def save_frame(_source, k, i, profile, b64):
        with open(os.path.join(plan[k]["out"], "frames", f"{i:06d}_p{profile}.jpg"), "wb") as fh:
            fh.write(base64.b64decode(b64))

    def finish(k, res):
        """frames.ndjson and run.json of run k."""
        run, out = runs[k], plan[k]["out"]
        with open(os.path.join(out, "frames.ndjson"), "w") as fh:
            for f, ev in zip(res["frames"], events):
                for key in TIMES:
                    if key in f:
                        f[key] = round(f[key] - t0_ms, 3)
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
        if session:
            run["session"] = {"dir": a.out, "index": k + 1, "runs": len(plan),
                              "gap_ms": run["start_ms"] - runs[k - 1]["end_ms"] if k else None}
        save(run, out)
        print(f"client: {run['args']['abr']} {len(frames)} frames in {(res['end'] - res['start']) / 1e3:.1f} s, "
              f"profiles {list(run['profiles'].values())}, failed {failed}", flush=True)
        return failed == 0

    errors, done = [], 0
    try:
        with sync_playwright() as pw:
            args = chromium_args(origin)
            browser = pw.chromium.launch(headless=True, args=args)
            page = browser.new_context(ignore_https_errors=True).new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.expose_binding("tigasSaveFrame", save_frame)

            t0 = a.t0 or time.time()
            t0_ms = t0 * 1e3
            page.goto(f"https://{origin}/player?modelId={a.scene}", timeout=30000)
            loaded_ms = (time.time() - t0) * 1e3
            page.wait_for_function(FIRST_FRAME_JS, timeout=60000, polling=10)
            page.evaluate(APPLY_RESOLUTION_JS, a.resolution)
            page.wait_for_function("() => !window.TIGAS.busy()", timeout=60000, polling=10)
            first_ms = (time.time() - t0) * 1e3
            for x in runs + [session] if session else runs:
                x.update(chromium=browser.version, chromium_args=args, t0=t0, loaded_ms=loaded_ms,
                         first_frame_ms=first_ms)
            if first_ms > a.warmup * 1e3:
                raise RuntimeError(f"first frame took {first_ms:.0f} ms, "
                                   f"longer than the warm-up W = {a.warmup} s")

            page.add_script_tag(path=os.path.join(REPO, "exp", "runner.js"))
            ok = True
            for k, p in enumerate(plan):
                res = page.evaluate("cfg => window.tigasRun(cfg)", {
                    "events": events, "abr": p["abr"], "seed": p["seed"], "run": k,
                    "startAtMs": (t0 + a.warmup) * 1e3, "delayMs": a.gap_ms if k else 0,
                    "quiet": not a.verbose_console})
                ok = finish(k, res) and ok
                done += 1
            browser.close()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        for x in runs[done:] + [session] if session else runs[done:]:
            x.update(status="error", error=err, page_errors=errors[-20:])
        for run, p in zip(runs[done:], plan[done:]):
            save(run, p["out"])
        if session:
            save(session, a.out)
        print(f"client: {err}", file=sys.stderr)
        sys.exit(1)

    if session:
        session["status"] = "ok" if ok else "error"
        session["end_ms"] = runs[-1]["end_ms"]
        save(session, a.out)
        print(f"client: session of {len(runs)} runs in {(runs[-1]['end_ms'] - runs[0]['start_ms']) / 1e3:.1f} s, "
              f"{sum(r['status'] == 'ok' for r in runs)} ok")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
