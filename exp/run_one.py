#!/usr/bin/env python
"""One client.py run inside `rattan link`.

The downlink (server to client) replays --trace through a droptail queue of
1000 packets, as the tc scripts did; the uplink is not limited. Each
direction gets half of --rtt as delay. Without --trace only the delay is
applied (the RTT experiment).

rattan's own log goes to <out>/rattan.log. From it this script adds to
run.json when the trace started (rattan logs "first-payload is disabled" when
it forwards the first packet) relative to t0, which side sent that packet,
and how many packets rattan dropped because it was busy. It also records
the GPU.

On SIGTERM or SIGINT the run is stopped the same way as on a timeout, so
rattan can remove its network namespaces (run_all.py relies on this).

  python exp/run_one.py --trace T.json --rtt 20 --out DIR -- \\
      --trajectory NTHU/train/user3_train.json --abr l2a --seed 1
(arguments after -- go to client.py)
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone

from common import REPO

RATTAN = os.environ.get("RATTAN", "/usr/local/bin/rattan")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log_time(line):
    ts = ANSI.sub("", line).split()[0]
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()


def descendants(pid):
    children = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(d))
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def stop(proc):
    """Ends a run that timed out. Killing rattan itself can leave its network
    namespaces behind, so first end the command it runs and let rattan exit
    on its own."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in descendants(proc.pid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


class Interrupted(Exception):
    pass


def interrupt(signum, _frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise Interrupted(signal.Signals(signum).name)


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return out or None


def rattan_cmd(trace, rtt_ms, queue_packets=1000):
    half = f"{rtt_ms / 2:g}ms"
    cmd = [RATTAN, "link", "--uplink-delay", half, "--downlink-delay", half]
    if trace:
        cmd += ["--downlink-trace", trace, "--downlink-queue", "droptail",
                "--downlink-queue-args", json.dumps({"packet_limit": queue_packets})]
    return cmd


def annotate(out, trace, rtt_ms):
    """Adds what rattan.log says to run.json."""
    path = os.path.join(out, "run.json")
    if not os.path.exists(path):
        return None
    run = json.load(open(path))
    lines = open(os.path.join(out, "rattan.log"), errors="replace").read().splitlines()
    start = [log_time(l) for l in lines if "first-payload is disabled" in l]
    first = [ANSI.sub("", l) for l in lines if "Sent FirstPacket notify" in l]
    version = subprocess.run([RATTAN, "--version"], capture_output=True, text=True).stdout.split("\n")
    run["rattan"] = {
        "version": " ".join(v.strip() for v in version if v.strip()),
        "trace": os.path.abspath(trace) if trace else None,
        "rtt_ms": rtt_ms,
        "trace_start_ms": (start[0] - run["t0"]) * 1e3 if start and "t0" in run else None,
        "first_packet_from": re.search(r'rx_id="(\w+)"', first[0]).group(1) if first else None,
        "busy_drops": sum("system is busy" in l for l in lines),
    }
    run["gpu"] = gpu()
    json.dump(run, open(path, "w"), indent=1)
    return run


def main():
    argv = sys.argv[1:]
    client_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="rattan trace JSON (make_traces.py); none: delay only")
    ap.add_argument("--rtt", type=float, default=20, help="ms")
    ap.add_argument("--timeout", type=float, default=900, help="s")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    cmd = rattan_cmd(a.trace, a.rtt) + [
        "--", sys.executable, os.path.join(REPO, "exp", "client.py"), "--out", os.path.abspath(a.out),
        *client_args]
    env = dict(os.environ, RUST_LOG="info,rattan_core::core=debug")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with open(os.path.join(a.out, "rattan.log"), "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            stop(proc)
            rc = "timeout"
        except Interrupted as e:
            stop(proc)
            rc = f"interrupted by {e}"
    run = annotate(a.out, a.trace, a.rtt)
    status = run["status"] if run else "no run.json"
    if rc == "timeout":
        status = f"timeout after {a.timeout:g} s"
    elif isinstance(rc, str):
        status = rc
    elif rc != 0 and run and run["status"] == "ok":
        status = f"rattan exit {rc}"
    lines = [l for l in open(os.path.join(a.out, "rattan.log")) if l.startswith("client:")]
    print(f"run_one: {status}; " + (lines[-1].strip() if lines else f"rattan exit {rc}"))
    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
