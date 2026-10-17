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

With --mode wt it plays through /player-wt instead, the page that streams
over WebTransport: the server renders the latest viewpoint 60 times a second
and pushes every frame as datagrams, whose profile the page's ThroughputABR
picks (--abr throughput). exp/runner_wt.js sets viewpoint k at
start + k * --event-ms and records every pose message and every frame.
frames.ndjson then has one line per viewpoint with the first frame drawn for
it (send: viewpoint set, msg: the pose message the frame was rendered from,
first/drawn: its first and last chunk arrived; "error": "not shown" if no
frame of it was drawn), wt_frames.ndjson one line per frame of the playback
and wt_sends.json the pose messages [time, viewpoint, profile]. At the end
the page sends the server the message it sends on beforeunload, so the server
stops rendering for it.

With --mode webrtc it opens a blank page and runs exp/runner_webrtc.js: the
page signs in to peerconnection_server at --server:--signaling-port, takes
the offer of tigas_sender (run_one.py starts both), warms up from the moment
the DataChannel opens to t0 + W (--warmup-mode) and sends the viewpoints with
--pacing. Each viewpoint is rendered at profile 0 and --resolution with the
camera /player uses after Apply (fx 1300, fy 800, cx and cy half the size).
frames.ndjson has one line per viewpoint: send, dec (its first frame read
from the track; drawn is the same time, for the scripts that read drawn),
shown (that frame's presentationTime from requestVideoFrameCallback, if it
was presented), recv (the receiveTime requestVideoFrameCallback gives for
that frame; track frames carry none, so viewpoints never presented have no
recv), rtp_ts, render_ms, t_push (frame
handed to the encoder, sender clock = page clock on one host); "error":
"no frame" or "timeout" if none was read. webrtc_frames.ndjson has every
frame read, webrtc_warmup.ndjson the warm-up messages, webrtc_shown.ndjson
the presented frames, receiver_stats.ndjson getStats every 100 ms, and
frames/<i>.yuv the I420 of every 12th viewpoint. Such a viewpoint that got
no frame has frames/<i>_standin.yuv instead, the frame on screen while its
frame was due (exp/runner_webrtc.js), and standin_i in frames.ndjson: the
viewpoint that frame belongs to (negative: a warm-up frame).
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

from common import REPO, chromium_args, git_commit, webrtc_chromium_args

FIRST_FRAME_JS = "() => document.getElementById('stage_placeholder')?.style.display === 'none'"
FIRST_FRAME_WT_JS = "() => window.TIGAS && window.TIGAS.last !== null"
# a frame of the resolution just applied was drawn
RESOLUTION_WT_JS = "res => { const l = window.TIGAS.last; return !!l && `${l.width}x${l.height}` === res; }"
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
    ap.add_argument("--mode", choices=["player", "wt", "webrtc"], default="player",
                    help="player: /player, one request per viewpoint; wt: /player-wt, WebTransport push; "
                    "webrtc: tigas_sender over WebRTC")
    ap.add_argument("--signaling-port", type=int, default=8888, help="--mode webrtc: peerconnection_server")
    ap.add_argument("--pacing", choices=["tms", "closed"], default="tms",
                    help="--mode webrtc: tms sends viewpoint i at t0 + W + tMs_i; closed sends the next "
                    "one when the frame of the previous one has been read")
    ap.add_argument("--warmup-mode", choices=["reverse", "pose0"], default="reverse",
                    help="--mode webrtc: what the page sends before t0 + W (exp/runner_webrtc.js)")
    ap.add_argument("--closed-timeout-ms", type=float, default=2000,
                    help="--pacing closed: send the next viewpoint after this long without a frame")
    ap.add_argument("--save-every", type=int, default=12, help="--mode webrtc: save every Nth viewpoint's frame")
    ap.add_argument("--no-display", action="store_true",
                    help="--mode webrtc: no <video> element, so no requestVideoFrameCallback times")
    ap.add_argument("--abr", choices=["simple", "l2a", "lol+", "throughput"],
                    help="required without --session; throughput (ThroughputABR) is the one of --mode wt")
    ap.add_argument("--event-ms", type=float, default=40, help="--mode wt: one viewpoint every N ms")
    ap.add_argument("--tail-ms", type=float, default=1000,
                    help="--mode wt: keep recording frames N ms after the last viewpoint")
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
    if a.mode == "webrtc":
        if a.session:
            ap.error("--mode webrtc plays one run")
        a.abr = a.abr or f"webrtc-{a.pacing}"
    if not a.abr and not a.session:
        ap.error("--abr or --session is required")
    if a.mode == "wt" and (a.session or a.abr != "throughput"):
        ap.error("--mode wt plays one run with --abr throughput")
    if a.mode == "player" and a.abr == "throughput":
        ap.error("--abr throughput needs --mode wt")

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

    def finish_wt(k, res):
        """frames.ndjson, wt_frames.ndjson, wt_sends.json and run.json of a
        --mode wt run."""
        run, out = runs[k], plan[k]["out"]

        def rel(t):
            return None if t is None else round(t - t0_ms, 3)
        frames = sorted(res["frames"], key=lambda f: f["id"])
        by_id = {f["id"]: f for f in frames}
        drawn = [f for f in frames if "drawn" in f]
        per_vp = {}
        for f in drawn:
            per_vp[f["i"]] = per_vp.get(f["i"], 0) + 1
        with open(os.path.join(out, "frames.ndjson"), "w") as fh:
            for i, (ev, t_set, fid) in enumerate(zip(events, res["applied"], res["shown"])):
                rec = {"i": i, "send": rel(t_set), "t_ms": ev.get("tMs")}
                if fid is None:
                    rec["error"] = "not shown"
                else:
                    f = by_id[fid]
                    rec.update(id=fid, msg=rel(f["ts"]), first=rel(f["first"]), drawn=rel(f["drawn"]),
                               profile=f["profile"], abr_profile=f["profile"], bytes=f["bytes"],
                               render_ms=f["render_ms"], drawn_frames=per_vp[i])
                fh.write(json.dumps(rec) + "\n")
        with open(os.path.join(out, "wt_frames.ndjson"), "w") as fh:
            for f in frames:
                f = dict(f, **{key: rel(f.get(key)) for key in ("ts", "first", "last", "drawn")})
                fh.write(json.dumps(f) + "\n")
        json.dump([[rel(t), i, p] for t, i, p in res["sends"]], open(os.path.join(out, "wt_sends.json"), "w"))
        ids = [f["id"] for f in frames]
        start, end = res["start"], res["end"]
        shown = sum(x is not None for x in res["shown"])
        run.update(
            status="ok" if drawn else "error",
            start_ms=start - t0_ms, end_ms=end - t0_ms, camera=res["camera"],
            frames=len(drawn), failed_frames=len(events) - shown,
            profiles={p: sum(f["profile"] == p for f in drawn) for p in range(4)},
            wt={"event_ms": a.event_ms, "tail_ms": a.tail_ms, "sends": len(res["sends"]),
                "frames_seen": len(frames), "frames_drawn": len(drawn),
                "frames_incomplete": len(frames) - len(drawn),
                # image ids are consecutive per session: ids in between that never
                # arrived lost every chunk
                "frames_missing": (max(ids) - min(ids) + 1 - len(ids)) if ids else 0,
                "viewpoints_shown": shown,
                "bytes": sum(f["bytes"] for f in frames if start <= f["first"] <= end),
                "drawn_in_playback": sum(start <= f["drawn"] <= end for f in drawn)},
            page_errors=errors[-20:])
        save(run, out)
        print(f"client: wt {len(drawn)} frames drawn in {(end - start) / 1e3:.1f} s, "
              f"{shown}/{len(events)} viewpoints shown, {len(frames) - len(drawn)} incomplete, "
              f"profiles {list(run['profiles'].values())}", flush=True)
        return bool(drawn)

    def save_yuv(_source, i, w, h, b64, kind="frame"):
        name = f"{i:06d}.yuv" if kind == "frame" else f"{i:06d}_standin.yuv"
        with open(os.path.join(a.out, "frames", name), "wb") as fh:
            fh.write(base64.b64decode(b64))

    def finish_webrtc(res):
        """frames.ndjson, the webrtc_*.ndjson files, receiver_stats.ndjson and
        run.json of a --mode webrtc run."""
        run, out = runs[0], a.out

        def rel(t):
            return None if t is None else round(t - t0_ms, 3)

        def write(name, recs, keys):
            with open(os.path.join(out, name), "w") as fh:
                for r in recs:
                    fh.write(json.dumps(dict(r, **{k: rel(r.get(k)) for k in keys if k in r})) + "\n")

        rtp_to_i = {m["rtp_ts"]: m["i"] for m in res["metas"]}
        first_shown = {}
        for s in res["shown"]:
            first_shown.setdefault(s["rtp"], s)
        poses = res["poses"]
        standin = {x["i"]: x["from_rtp"] for x in res["standins"]}
        for p, ev in zip(poses, events):
            p["t_ms"] = ev.get("tMs")
            p["drawn"] = p.get("dec")
            s = first_shown.get(p.get("rtp_ts"))
            if s:
                p["shown"] = s["presented"]
                if p.get("recv") is None:
                    p["recv"] = s["recv"]
            if p["i"] in standin:
                p["standin_i"] = rtp_to_i.get(standin[p["i"]])
        write("frames.ndjson", poses, ("send", "dec", "drawn", "shown", "recv", "t_push", "t_meta"))
        for f in res["decoded"]:
            f.setdefault("i", rtp_to_i.get(f["rtp"]))
        write("webrtc_frames.ndjson", res["decoded"], ("dec", "recv", "cap"))
        write("webrtc_warmup.ndjson", res["warm"], ("send", "dec"))
        for s in res["shown"]:
            s["i"] = rtp_to_i.get(s["rtp"])
        write("webrtc_shown.ndjson", res["shown"], ("presented", "expected", "recv"))
        write("receiver_stats.ndjson", res["rstats"], ("t",))

        got = [p for p in poses if "dec" in p]
        decoded = res["decoded"]
        run.update(
            status="ok" if got else "error",
            start_ms=rel(res["start"]), end_ms=rel(res["end"]), camera=res["camera"],
            frames=len(got), failed_frames=len(poses) - len(got), profiles={0: len(got)},
            webrtc={"pacing": a.pacing, "warmup_mode": a.warmup_mode,
                    "times": {k: rel(v) for k, v in res["times"].items()},
                    "warmup_sent": len(res["warm"]),
                    "warmup_decoded": sum("dec" in w for w in res["warm"]),
                    "frames_read": len(decoded),
                    "frames_without_message": sum(f["i"] is None for f in decoded),
                    "frames_before_message": res["pendingAtRead"],
                    "standins": len(standin),
                    "formats": sorted({f["fmt"] for f in decoded if f.get("fmt")}),
                    "messages": len(res["metas"]), "presented": len(res["shown"]),
                    "viewpoints_presented": sum("shown" in p for p in poses),
                    "timeouts": sum(p.get("error") == "timeout" for p in poses)},
            page_errors=errors[-20:])
        save(run, out)
        print(f"client: webrtc {a.pacing} {len(got)}/{len(poses)} viewpoints decoded in "
              f"{(res['end'] - res['start']) / 1e3:.1f} s, {len(decoded)} frames read, "
              f"{run['webrtc']['warmup_decoded']}/{len(res['warm'])} warm-up", flush=True)
        return len(got) == len(poses)

    errors, done = [], 0
    try:
        with sync_playwright() as pw:
            args = webrtc_chromium_args() if a.mode == "webrtc" else chromium_args(origin)
            browser = pw.chromium.launch(headless=True, args=args)
            page = browser.new_context(ignore_https_errors=True).new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.expose_binding("tigasSaveFrame", save_frame)

            t0 = a.t0 or time.time()
            t0_ms = t0 * 1e3
            if a.mode == "webrtc":
                page.expose_binding("tigasSaveYuv", save_yuv)
                page.goto("about:blank")
                page.add_script_tag(path=os.path.join(REPO, "exp", "runner_webrtc.js"))
                # The first RTCPeerConnection of a page takes ~150 ms on our test host;
                # a throwaway one here keeps that before t0.
                page.evaluate("ev => { window.tigasEvents = ev; new RTCPeerConnection().close(); }", events)
                # The page's first packet, its sign-in to the signaling server,
                # starts the rattan trace. t0 is taken once the script and the
                # events are in, so that packet comes a few ms after t0, as
                # /player's first request does; before, page setup put it
                # ~130 ms after.
                loaded_ms = (time.time() - t0) * 1e3
                t0 = a.t0 or time.time()
                t0_ms = t0 * 1e3
                w, h = map(int, a.resolution.split("x"))
                camera = {"fx": 1300.0, "fy": 800.0, "cx": w / 2, "cy": h / 2, "width": w, "height": h}
                runs[0].update(chromium=browser.version, chromium_args=args, t0=t0, loaded_ms=loaded_ms)
                res = page.evaluate("cfg => window.tigasRunWebRTC(cfg)", {
                    "signaling": f"http://{a.server}:{a.signaling_port}",
                    "modelId": a.scene, "camera": camera, "startAtMs": (t0 + a.warmup) * 1e3,
                    "pacing": a.pacing, "warmup": a.warmup_mode, "closedTimeoutMs": a.closed_timeout_ms,
                    "tailMs": a.tail_ms, "saveEvery": a.save_every, "display": not a.no_display,
                    "connectTimeoutMs": a.warmup * 1e3, "quiet": not a.verbose_console})
                ok = finish_webrtc(res)
                done = 1
                browser.close()
                sys.exit(0 if ok else 1)
            wt = a.mode == "wt"
            page.goto(f"https://{origin}/{'player-wt' if wt else 'player'}?modelId={a.scene}", timeout=30000)
            loaded_ms = (time.time() - t0) * 1e3
            page.wait_for_function(FIRST_FRAME_WT_JS if wt else FIRST_FRAME_JS, timeout=60000, polling=10)
            page.evaluate(APPLY_RESOLUTION_JS, a.resolution)
            if wt:
                page.wait_for_function(RESOLUTION_WT_JS, arg=a.resolution, timeout=60000, polling=10)
            else:
                page.wait_for_function("() => !window.TIGAS.busy()", timeout=60000, polling=10)
            first_ms = (time.time() - t0) * 1e3
            for x in runs + [session] if session else runs:
                x.update(chromium=browser.version, chromium_args=args, t0=t0, loaded_ms=loaded_ms,
                         first_frame_ms=first_ms)
            if first_ms > a.warmup * 1e3:
                raise RuntimeError(f"first frame took {first_ms:.0f} ms, "
                                   f"longer than the warm-up W = {a.warmup} s")

            if wt:
                page.add_script_tag(path=os.path.join(REPO, "exp", "runner_wt.js"))
                res = page.evaluate("cfg => window.tigasRunWT(cfg)", {
                    "events": events, "eventMs": a.event_ms, "tailMs": a.tail_ms, "run": 0,
                    "startAtMs": (t0 + a.warmup) * 1e3, "quiet": not a.verbose_console})
                ok = finish_wt(0, res)
                done = 1
                page.evaluate("() => window.TIGAS.close()")
                time.sleep(0.5)
                browser.close()
                sys.exit(0 if ok else 1)
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
