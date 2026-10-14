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
6. compute the quality metrics (metrics.py);
7. write the summary tables and figures (summarize.py).

Steps 1 and 3 are skipped when no run is left. Steps 5 to 7 keep what
earlier invocations produced and only add what is missing; their output goes
to render_gt.log, metrics.log and summarize.log.

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

from common import REPO, find_runs, load_events, run_rel, trace_name
from make_traces import build

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
    """Every run of the config, in the order they are run."""
    runs, seen, lengths = [], set(), {}
    for name, t in cfg.items():
        if name == "server" or (only and name not in only):
            continue
        conds = [(scene, traj, trace, scale, rtt)
                 for scene in t["scenes"] for traj in t["trajectories"]
                 for trace, scale in ([(tr, s) for tr in t["traces"] for s in t["scales"]]
                                      or [(None, None)])
                 for rtt in t["rtt_ms"]]
        for rep in range(1, t["reps"] + 1):
            for scene, traj, trace, scale, rtt in conds:
                for abr in t["abrs"]:
                    n = lengths.setdefault(traj, len(load_events(traj)))
                    r = {"exp": name, "scene": scene, "trajectory": traj, "trace": trace,
                         "scale": scale, "trace_name": trace_name(trace, scale), "rtt_ms": rtt,
                         "abr": abr, "rep": rep,
                         "viewpoints": min(n, t.get("max_viewpoints") or n),
                         "max_viewpoints": t.get("max_viewpoints")}
                    r["rel"] = run_rel(scene, traj, r["trace_name"], rtt, abr, rep)
                    if r["rel"] not in seen:
                        seen.add(r["rel"])
                        runs.append(r)
    return runs


def is_done(run_dir, viewpoints):
    try:
        run = json.load(open(os.path.join(run_dir, "run.json")))
    except (OSError, ValueError):
        return False
    return run.get("status") == "ok" and "rattan" in run and run.get("viewpoints") == viewpoints


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
    """traces/<trace>_x<scale>.json; refuses to change a trace that finished
    runs used."""
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    paths = {}
    for r in runs:
        if not r["trace"] or r["trace_name"] in paths:
            continue
        path = os.path.join(out, "traces", r["trace_name"] + ".json")
        text = json.dumps(build(r["trace"], r["scale"], cfg["server"]["warmup_s"]), separators=(",", ":"))
        if os.path.exists(path) and open(path).read() != text:
            used = [x["rel"] for x in runs if x["trace_name"] == r["trace_name"]
                    and is_done(os.path.join(out, "runs", x["rel"]), x["viewpoints"])]
            if used:
                raise SystemExit(f"{path} would change, but {len(used)} finished runs used it; "
                                 "use a new --out")
        if not os.path.exists(path) or open(path).read() != text:
            open(path, "w").write(text)
            log(f"trace: wrote {os.path.relpath(path, out)}")
        paths[r["trace_name"]] = path
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
                or "playwright/driver" in cmd[0] or any(a.endswith("exp/client.py") for a in cmd)):
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


def warm_up(out, runs, warmup_s=30):
    """One short local run per scene: the server loads the model and
    compiles its kernels before the first measured run."""
    for scene in sorted({r["scene"] for r in runs}):
        traj = next(r["trajectory"] for r in runs if r["scene"] == scene)
        d = os.path.join(out, "warmup", scene)
        shutil.rmtree(d, ignore_errors=True)
        rc, text = call([sys.executable, "exp/client.py", "--server", "127.0.0.1", "--scene", scene,
                         "--trajectory", traj, "--abr", "simple", "--max-frames", "12",
                         "--warmup", str(warmup_s), "--out", d])
        if rc != 0:
            raise SystemExit(f"warm-up of {scene} failed: {text.strip().splitlines()[-1:]}")
        first = json.load(open(os.path.join(d, "run.json")))["first_frame_ms"]
        log(f"warm-up: {scene} first frame after {first / 1e3:.1f} s")


def run_one(cfg, out, r, trace_path):
    s = cfg["server"]
    cmd = [sys.executable, "exp/run_one.py", "--rtt", f"{r['rtt_ms']:g}",
           "--timeout", str(s["timeout_s"]), "--out", os.path.join(out, "runs", r["rel"])]
    if trace_path:
        cmd += ["--trace", trace_path]
    cmd += ["--", "--scene", r["scene"], "--trajectory", r["trajectory"], "--abr", r["abr"],
            "--seed", str(r["rep"]), "--warmup", f"{s['warmup_s']:g}"]
    if r["max_viewpoints"]:
        cmd += ["--max-frames", str(r["max_viewpoints"])]
    rc, text = call(cmd, stop_group=False)
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
    server = Server(out)
    failed = []
    try:
        server.start()
        warm_up(out, pending)
        for k, r in enumerate(pending, 1):
            run_dir = os.path.join(out, "runs", r["rel"])
            name = f"run {k}/{len(pending)} {r['rel']}"
            for attempt in (1, 2):
                if not server.alive():
                    log("server: died; restarting")
                    server.start()
                    warm_up(out, pending)
                shutil.rmtree(run_dir, ignore_errors=True)
                before, ns = experiment_procs(), netns_count()
                t = time.time()
                status = run_one(cfg, out, r, traces.get(r["trace_name"]))
                left = kill_new(before)
                ok = is_done(run_dir, r["viewpoints"])
                log(f"{name} {'ok' if ok else 'FAILED'} ({time.time() - t:.0f} s): {status}")
                if left:
                    log(f"  killed {len(left)} leftover processes after the run")
                if netns_count() > ns:
                    log(f"  {netns_count() - ns} network namespaces left behind "
                        "(remove with sudo ip netns delete)")
                if ok:
                    break
                dst = os.path.join(out, "failed", r["rel"].replace(os.sep, "__")
                                   + time.strftime("__%Y%m%d-%H%M%S"))
                if os.path.exists(run_dir):
                    shutil.move(run_dir, dst)
                    log(f"  moved the attempt to {os.path.relpath(dst, out)}")
            else:
                failed.append(r["rel"])
    finally:
        stop_child()
        server.stop()
    return failed


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
    log(f"runs: {len(runs)} in the config, {len(runs) - len(pending)} already done, "
        f"{len(pending)} to run")
    failed = []
    try:
        if pending:
            failed = run_matrix(cfg, out, runs, pending, a)
        else:
            write_traces(cfg, out, runs)
        s = cfg["server"]
        for step, cmd in (
                ("render_gt", ["exp/render_gt.py", out, "--stride", str(s["metric_stride"]),
                               "--resolution", s["resolution"]]),
                ("metrics", ["exp/metrics.py", out, "--stride", str(s["metric_stride"])]),
                ("summarize", ["exp/summarize.py", out])):
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
