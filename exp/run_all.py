#!/usr/bin/env python
"""Runs every experiment of a config file (exp/config.toml) and turns the
results into tables and figures:

1. check the machine (exp/setup/check_env.sh, model and trajectory files);
2. write the rattan traces (make_traces.py);
3. start the server and load every scene once with a short local run;
4. run every combination of the config, one at a time, through run_one.py.
   The outer loop is the repetition and the ABRs take turns inside it, so a
   slow drift of the machine reaches all ABRs alike. A failed run is tried
   once more. Runs whose run.json says ok are skipped, so after an
   interruption the same command carries on;
5. render the reference images (render_gt.py), after the runs so they do
   not share the GPU with an experiment;
6. compute the quality metrics, ours (metrics.py) and the paper's
   (metrics_paper.py);
7. write the summary tables and figures (summarize.py).

Steps 1 and 3 are skipped when no run is left. Steps 5 to 7 keep what
earlier invocations produced and only add what is missing; their output goes
to render_gt.log, metrics.log, metrics_paper.log and summarize.log.

Tables can also restore one of the methods of the paper (the ablations of
exp/ablate.toml); the run directories then get a suffix after the trace:
- offsets_s = [...]: repetition k starts the Ghent trace at its second
  offsets_s[k-1] (make_traces.py --offset); suffix _phi;
- sessions = {S1 = [abr order], ...}: each session plays reps runs of each
  ABR in that order in one page, under one rattan (client.py --session);
  suffix _<session>. A session is run, retried and kept as a whole;
- shaper = "tc-lo": tc on lo instead of rattan (run_one.py --shaper tc-lo);
  suffix _tclo. The qdisc on lo is removed before and after the matrix.
A table with mode = "wt" plays through /player-wt, the WebTransport push
mode (client.py --mode wt, abrs = ["throughput"], event_ms = 40). The
server keeps rendering for a WebTransport session that was not closed
(gs_app.TimedSend), so it is restarted before every such run.

  python exp/run_all.py [--config exp/config.toml] [--out exp/results/main]

Output in --out; run_all.log there keeps the progress lines of
every invocation. SIGTERM or Ctrl-C stops the current run cleanly.
"""
import argparse
import fcntl
import filecmp
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib

from common import REPO, find_runs, load_events, run_rel, trace_name, traj_name
from make_traces import build, tc_rates

EXP = os.path.join(REPO, "exp")
PORT = 4433
LOG = None


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


# ---- the run list

def plan_runs(cfg, only):
    """Every run of the config, in the order they are run. Runs of a session
    share r["unit"]; any other run is a unit of its own."""
    runs, seen, lengths = [], set(), {}
    for name, t in cfg.items():
        if name == "server" or (only and name not in only):
            continue
        conds = [(scene, traj, trace, scale, rtt)
                 for scene in t["scenes"] for traj in t["trajectories"]
                 for trace, scale in ([(tr, s) for tr in t["traces"] for s in t["scales"]]
                                      or [(None, None)])
                 for rtt in t["rtt_ms"]]
        offsets, shaper = t.get("offsets_s"), t.get("shaper", "rattan")
        reps = len(offsets) if offsets else t["reps"]
        if t.get("sessions"):
            units = [[(c, abr, rep, sname) for abr in order for rep in range(1, reps + 1)]
                     for sname, order in t["sessions"].items() for c in conds]
        else:
            units = [[(c, abr, rep, None)] for rep in range(1, reps + 1) for c in conds for abr in t["abrs"]]
        for unit in units:
            for k, ((scene, traj, trace, scale, rtt), abr, rep, sname) in enumerate(unit, 1):
                n = lengths.setdefault(traj, len(load_events(traj)))
                off = offsets[rep - 1] if offsets else 0
                label = (trace_name(trace, scale) + ("_phi" if offsets else "")
                         + (f"_{sname}" if sname else "") + ("_tclo" if shaper == "tc-lo" else ""))
                r = {"exp": name, "mode": t.get("mode", "player"), "event_ms": t.get("event_ms", 40),
                     "scene": scene, "trajectory": traj, "trace": trace,
                     "scale": scale, "trace_name": trace_name(trace, scale) + (f"_phi{off}" if off else ""),
                     "label": label, "offset_s": off, "shaper": shaper, "rtt_ms": rtt,
                     "abr": abr, "rep": rep, "seed": rep,
                     "session": sname, "session_index": k if sname else None,
                     "viewpoints": min(n, t.get("max_viewpoints") or n),
                     "max_viewpoints": t.get("max_viewpoints")}
                r["rel"] = run_rel(scene, traj, label, rtt, abr, rep)
                r["unit"] = (os.path.join("sessions", scene, traj_name(traj), label, f"rtt{rtt:g}")
                             if sname else r["rel"])
                if r["rel"] not in seen:
                    seen.add(r["rel"])
                    runs.append(r)
    return runs


def units_of(runs):
    """runs grouped by r["unit"], in order."""
    out = {}
    for r in runs:
        out.setdefault(r["unit"], []).append(r)
    return list(out.values())


def is_done(run_dir, viewpoints):
    try:
        run = json.load(open(os.path.join(run_dir, "run.json")))
    except (OSError, ValueError):
        return False
    return (run.get("status") == "ok" and ("rattan" in run or "tc" in run)
            and run.get("viewpoints") == viewpoints)


# ---- steps 1 and 2

def check_env(cfg, runs):
    log("check: exp/setup/check_env.sh")
    scenes = sorted({r["scene"] for r in runs})
    p = subprocess.run(["bash", "exp/setup/check_env.sh"], cwd=REPO, capture_output=True, text=True,
                       env=dict(os.environ, SCENE=scenes[0]))
    for line in p.stdout.splitlines():
        if line.startswith(("FAIL", "WARN")):
            log("  " + line)
    problems = [] if p.returncode == 0 else ["check_env.sh failed (lines above)"]
    for scene in scenes:
        if not glob.glob(os.path.join(REPO, "static", "models", scene, "*.ply")):
            problems.append(f"no model for scene {scene} (static/models/{scene}/*.ply)")
    if udp_port_used(PORT):
        problems.append(f"UDP port {PORT} in use; is another server running?")
    if problems:
        raise SystemExit("check failed: " + "; ".join(problems))
    log("check: passed")


def write_traces(cfg, out, runs):
    """traces/<trace>_x<scale>[_phi<offset>].json for rattan, .rates for
    tc_lo_loop.sh; refuses to change a trace that finished runs used."""
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    s, paths = cfg["server"], {}
    for r in runs:
        key = (r["trace_name"], r["shaper"])
        if not r["trace"] or key in paths:
            continue
        if r["shaper"] == "tc-lo":
            path = os.path.join(out, "traces", r["trace_name"] + ".rates")
            text = tc_rates(r["trace"], r["scale"], r["offset_s"])
        else:
            path = os.path.join(out, "traces", r["trace_name"] + ".json")
            text = json.dumps(build(r["trace"], r["scale"], s["warmup_s"], total=s.get("trace_total_s", 3600),
                                    offset=r["offset_s"]), separators=(",", ":"))
        if os.path.exists(path) and open(path).read() != text:
            used = [x["rel"] for x in runs if (x["trace_name"], x["shaper"]) == key
                    and is_done(os.path.join(out, "runs", x["rel"]), x["viewpoints"])]
            if used:
                raise SystemExit(f"{path} would change, but {len(used)} finished runs used it; "
                                 "use a new --out")
        if not os.path.exists(path) or open(path).read() != text:
            open(path, "w").write(text)
            log(f"trace: wrote {os.path.relpath(path, out)}")
        paths[key] = path
    return paths


# ---- processes

def udp_port_used(port):
    for f in ("/proc/net/udp", "/proc/net/udp6"):
        for line in open(f).read().splitlines()[1:]:
            if int(line.split()[1].split(":")[1], 16) == port:
                return True
    return False


def experiment_procs():
    """PIDs of this user's rattan, client.py and Playwright processes."""
    uid, me, out = os.getuid(), os.getpid(), set()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != uid:
                continue
            cmd = open(f"/proc/{d}/cmdline", "rb").read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if (os.path.basename(cmd[0]) == "rattan" or "ms-playwright" in cmd[0]
                or "playwright/driver" in cmd[0]
                or any(a.endswith(("exp/client.py", "exp/tc_lo_loop.sh")) for a in cmd)):
            out.add(int(d))
    return out


def kill_new(before):
    """Kills experiment processes that were not there before the run."""
    left = experiment_procs() - before
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in left:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(1)
        left = {p for p in left if os.path.exists(f"/proc/{p}")}
        if not left:
            break
    return left


def lo_shaped():
    """Whether lo has a qdisc of ours (tc via sudo -n, see run_one.py)."""
    p = subprocess.run(["sudo", "-n", "tc", "qdisc", "show", "dev", "lo"], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"sudo -n tc failed ({p.stderr.strip()}); allow tc in sudoers without a password")
    return "htb" in p.stdout or "netem" in p.stdout


def clear_lo(why):
    if lo_shaped():
        subprocess.run(["sudo", "-n", "tc", "qdisc", "del", "dev", "lo", "root"], capture_output=True)
        log(f"tc: removed the qdisc on lo ({why})" + (", FAILED" if lo_shaped() else ""))


def netns_count():
    return len(os.listdir("/run/netns")) if os.path.isdir("/run/netns") else 0


class Server:
    def __init__(self, out):
        self.out, self.proc = out, None

    def start(self):
        log("server: starting")
        logf = open(os.path.join(self.out, "server.log"), "a")
        self.proc = subprocess.Popen(
            [sys.executable, "http3_server.py", "-c", "certificates/ssl_cert.pem",
             "-k", "certificates/ssl_key.pem", "--host", "0.0.0.0", "--port", str(PORT)],
            cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            if udp_port_used(PORT):
                return
            if self.proc.poll() is not None:
                break
            time.sleep(0.2)
        raise SystemExit("server did not start; see server.log")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if not self.alive():
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        log("server: stopped")


CHILD = None   # the subprocess currently running, stopped on an interruption


def call(cmd, stop_group=True):
    """Runs cmd in its own session; returns (exit code, output)."""
    global CHILD
    CHILD = (subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, start_new_session=True), stop_group)
    out = CHILD[0].communicate()[0]
    rc, CHILD = CHILD[0].returncode, None
    return rc, out


def stop_child():
    """run_one.py ends its run on SIGTERM and lets rattan remove its network
    namespaces, so it gets SIGTERM alone; other steps get it as a group."""
    if CHILD is None or CHILD[0].poll() is not None:
        return
    proc, group = CHILD
    if group:
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def warm_up(out, runs, resolution, warmup_s=30):
    """One short local run per scene: the server loads the model and
    compiles its kernels before the first measured run."""
    for scene in sorted({r["scene"] for r in runs}):
        traj = next(r["trajectory"] for r in runs if r["scene"] == scene)
        d = os.path.join(out, "warmup", scene)
        shutil.rmtree(d, ignore_errors=True)
        rc, text = call([sys.executable, "exp/client.py", "--server", "127.0.0.1", "--scene", scene,
                         "--trajectory", traj, "--abr", "simple", "--max-frames", "12",
                         "--resolution", resolution, "--warmup", str(warmup_s), "--out", d])
        if rc != 0:
            raise SystemExit(f"warm-up of {scene} failed: {text.strip().splitlines()[-1:]}")
        first = json.load(open(os.path.join(d, "run.json")))["first_frame_ms"]
        log(f"warm-up: {scene} first frame after {first / 1e3:.1f} s")


def meta(r):
    return {k: r[k] for k in ("exp", "mode", "label", "trace_name", "offset_s", "shaper", "session",
                              "session_index")}


def shaper_args(r, trace_path):
    if r["shaper"] == "tc-lo":
        return ["--shaper", "tc-lo", "--tc-rates", trace_path]
    return ["--trace", trace_path] if trace_path else []


def client_args(s, r):
    out = ["--scene", r["scene"], "--trajectory", r["trajectory"],
           "--warmup", f"{s['warmup_s']:g}", "--resolution", s["resolution"]]
    if r["mode"] == "wt":
        out += ["--mode", "wt", "--event-ms", f"{r['event_ms']:g}"]
    return out + (["--max-frames", str(r["max_viewpoints"])] if r["max_viewpoints"] else [])


def run_one(cfg, out, r, trace_path):
    s = cfg["server"]
    cmd = [sys.executable, "exp/run_one.py", "--rtt", f"{r['rtt_ms']:g}", *shaper_args(r, trace_path),
           "--timeout", str(s["timeout_s"]), "--out", os.path.join(out, "runs", r["rel"]),
           "--", *client_args(s, r), "--abr", r["abr"], "--seed", str(r["seed"]),
           "--meta", json.dumps(meta(r))]
    return last_line(*call(cmd, stop_group=False))


def run_session(cfg, out, unit, trace_path):
    """All runs of a session in one client.py --session (one page, one rattan)."""
    s, r = cfg["server"], unit[0]
    sdir = os.path.join(out, r["unit"])
    os.makedirs(sdir, exist_ok=True)
    plan = [{"abr": x["abr"], "seed": x["seed"], "out": os.path.join(out, "runs", x["rel"]), "meta": meta(x)}
            for x in unit]
    json.dump(plan, open(os.path.join(sdir, "plan.json"), "w"), indent=1)
    cmd = [sys.executable, "exp/run_one.py", "--rtt", f"{r['rtt_ms']:g}", *shaper_args(r, trace_path),
           "--timeout", str(s.get("session_timeout_s", 6000)), "--out", sdir,
           "--", *client_args(s, r), "--session", os.path.join(sdir, "plan.json"),
           "--gap-ms", f"{s.get('session_gap_ms', 150):g}"]
    return last_line(*call(cmd, stop_group=False))


def last_line(rc, text):
    lines = [l for l in text.splitlines() if l.startswith("run_one:")]
    return lines[-1][len("run_one: "):] if lines else f"exit {rc}: {text.strip()[-300:]}"


# ---- main

def save_config(path, out):
    dst = os.path.join(out, "config.toml")
    if os.path.exists(dst) and not filecmp.cmp(path, dst, shallow=False):
        old = os.path.join(out, time.strftime("config-%Y%m%d-%H%M%S.toml",
                                              time.localtime(os.path.getmtime(dst))))
        os.rename(dst, old)
        log(f"config changed; the previous one is {os.path.basename(old)}")
    shutil.copyfile(path, dst)


def run_matrix(cfg, out, runs, pending, args):
    if not args.skip_check:
        check_env(cfg, pending)
    traces = write_traces(cfg, out, runs)
    tc_lo = any(r["shaper"] == "tc-lo" for r in pending)
    if tc_lo:
        clear_lo("before the runs")
    units = units_of(pending)
    server = Server(out)
    failed, used = [], False
    try:
        server.start()
        warm_up(out, pending, cfg["server"]["resolution"])
        for k, unit in enumerate(units, 1):
            r = unit[0]
            name = (f"session {k}/{len(units)} {r['unit']} ({len(unit)} runs)" if r["session"]
                    else f"run {k}/{len(units)} {r['rel']}")
            dirs = [os.path.join(out, "runs", x["rel"]) for x in unit] + ([os.path.join(out, r["unit"])]
                                                                          if r["session"] else [])
            for attempt in (1, 2):
                if r["mode"] == "wt" and used:
                    server.stop()
                    server.start()
                    warm_up(out, pending, cfg["server"]["resolution"])
                elif not server.alive():
                    log("server: died; restarting")
                    server.start()
                    warm_up(out, pending, cfg["server"]["resolution"])
                used = True
                for d in dirs:
                    shutil.rmtree(d, ignore_errors=True)
                before, ns = experiment_procs(), netns_count()
                t = time.time()
                path = traces.get((r["trace_name"], r["shaper"]))
                status = run_session(cfg, out, unit, path) if r["session"] else run_one(cfg, out, r, path)
                left = kill_new(before)
                done = [is_done(os.path.join(out, "runs", x["rel"]), x["viewpoints"]) for x in unit]
                ok = all(done)
                log(f"{name} {'ok' if ok else 'FAILED'} ({time.time() - t:.0f} s): {status}")
                if r["session"]:
                    for x, d in zip(unit, done):
                        log(f"  {x['rel']}: {'ok' if d else 'FAILED'} {run_brief(out, x)}")
                if left:
                    log(f"  killed {len(left)} leftover processes after the run")
                if netns_count() > ns:
                    log(f"  {netns_count() - ns} network namespaces left behind "
                        "(remove with sudo ip netns delete)")
                if r["shaper"] == "tc-lo":
                    clear_lo("left after the run")
                if ok:
                    break
                stamp = time.strftime("__%Y%m%d-%H%M%S")
                for d in dirs:
                    if os.path.exists(d):
                        dst = os.path.join(out, "failed", os.path.relpath(d, out).replace(os.sep, "__") + stamp)
                        shutil.move(d, dst)
                        log(f"  moved the attempt to {os.path.relpath(dst, out)}")
            else:
                failed += [x["rel"] for x in unit]
    finally:
        stop_child()
        server.stop()
        if tc_lo:
            clear_lo("after the runs")
    return failed


def run_brief(out, r):
    try:
        run = json.load(open(os.path.join(out, "runs", r["rel"], "run.json")))
        return (f"{run['frames']} frames in {(run['end_ms'] - run['start_ms']) / 1e3:.1f} s, "
                f"profiles {list(run['profiles'].values())}")
    except (OSError, ValueError, KeyError):
        return "(no complete run.json)"


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=os.path.join(EXP, "config.toml"))
    ap.add_argument("--out", default=os.path.join(EXP, "results", "main"))
    ap.add_argument("--only", nargs="+", metavar="EXP", help="run only these tables of the config")
    ap.add_argument("--skip-check", action="store_true", help="skip step 1")
    a = ap.parse_args()

    cfg = tomllib.load(open(a.config, "rb"))
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    lock = open(os.path.join(out, ".run_all.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run_all.py is working on {out}")
    LOG = os.path.join(out, "run_all.log")
    signal.signal(signal.SIGTERM, interrupt)
    log(f"run_all: config {os.path.relpath(a.config, REPO)}, out {out}, pid {os.getpid()}")
    save_config(a.config, out)

    runs = plan_runs(cfg, a.only)
    pending = [r for r in runs if not is_done(os.path.join(out, "runs", r["rel"]), r["viewpoints"])]
    # a session with any run left is run again as a whole
    left = {r["unit"] for r in pending}
    pending = [r for r in runs if r["unit"] in left]
    log(f"runs: {len(runs)} in the config, {len(runs) - len(pending)} already done, "
        f"{len(pending)} to run")
    failed = []
    try:
        if pending:
            failed = run_matrix(cfg, out, runs, pending, a)
        else:
            write_traces(cfg, out, runs)
        s = cfg["server"]
        steps = [("render_gt", ["exp/render_gt.py", out, "--stride", str(s["metric_stride"]),
                                "--resolution", s["resolution"]]),
                 ("metrics", ["exp/metrics.py", out, "--stride", str(s["metric_stride"])]),
                 ("metrics_paper", ["exp/metrics_paper.py", out, "--stride", str(s["metric_stride"])]),
                 ("summarize", ["exp/summarize.py", out])]
        if not s.get("paper_metrics", True):
            steps = [x for x in steps if x[0] != "metrics_paper"]
        for step, cmd in steps:
            t = time.time()
            rc, text = call([sys.executable, *cmd])
            open(os.path.join(out, step + ".log"), "w").write(text)
            lines = text.strip().splitlines()
            if rc == 0:   # the step's own lines, without library warnings
                lines = [l for l in lines if l.startswith(step + ":")]
            for line in lines[-2:]:
                log(f"  {line}")
            log(f"{step}: {'done' if rc == 0 else 'FAILED'} ({time.time() - t:.0f} s, output in {step}.log)")
            if rc != 0:
                sys.exit(1)
    except KeyboardInterrupt:
        stop_child()
        log("interrupted; run the same command again to carry on")
        sys.exit(130)
    if failed:
        log(f"finished with {len(failed)} failed runs: {', '.join(failed)}")
        sys.exit(1)
    log(f"finished: {len(runs)} runs; figures in {os.path.join(out, 'figs')}")


if __name__ == "__main__":
    main()
