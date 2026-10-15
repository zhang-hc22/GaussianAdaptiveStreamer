#!/usr/bin/env python
"""One client.py run (or session, client.py --session) inside `rattan link`.

The downlink (server to client) replays --trace through a droptail queue of
1000 packets, as the tc scripts did; the uplink is not limited. Each
direction gets half of --rtt as delay. Without --trace only the delay is
applied (the RTT experiment).

rattan's own log goes to <out>/rattan.log. From it this script adds to
run.json when the trace started (rattan logs "first-payload is disabled" when
it forwards the first packet) relative to t0, which side sent that packet,
and how many packets rattan dropped because it was busy. It also records
the GPU.

With --shaper tc-lo it uses tc on lo instead of rattan, as
scripts/bandwidth_fluctuations.sh did (the ablation A3): one htb
class with netem (RTT/2 delay each way) shapes both directions together.
At t0 it starts exp/tc_lo_loop.sh with --tc-rates, and client.py talks to
127.0.0.1 with that t0. run.json gets the time each rate took effect and
the counters of `tc -s`. The qdisc is removed at the end, also after a
timeout or an error. tc runs through `sudo -n` (no password prompt).

On SIGTERM or SIGINT the run is stopped the same way as on a timeout, so
rattan can remove its network namespaces (run_all.py relies on this).

  python exp/run_one.py --trace T.json --rtt 20 --out DIR -- \\
      --trajectory NTHU/train/user3_train.json --abr l2a --seed 1
  python exp/run_one.py --shaper tc-lo --tc-rates R.txt --rtt 20 --out DIR -- ...
(arguments after -- go to client.py)
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from common import REPO

RATTAN = os.environ.get("RATTAN", "/usr/local/bin/rattan")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TC = ["sudo", "-n", "tc"]
WARMUP_KBIT = 50000   # the warm-up rate of make_traces.build()


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


def update_runs(out, key, value):
    """Sets run[key] in run.json of out and, for a session, of all its runs;
    returns out's run.json."""
    path = os.path.join(out, "run.json")
    if not os.path.exists(path):
        return None
    run = json.load(open(path))
    for p in [out] + (run["runs"] if "runs" in run else []):
        if p != out and not os.path.exists(os.path.join(p, "run.json")):
            continue
        x = run if p == out else json.load(open(os.path.join(p, "run.json")))
        x[key] = value
        x["gpu"] = run["gpu"] = run.get("gpu") or gpu()
        json.dump(x, open(os.path.join(p, "run.json"), "w"), indent=1)
    return run


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
    return update_runs(out, "rattan", {
        "version": " ".join(v.strip() for v in version if v.strip()),
        "trace": os.path.abspath(trace) if trace else None,
        "rtt_ms": rtt_ms,
        "trace_start_ms": (start[0] - run["t0"]) * 1e3 if start and "t0" in run else None,
        "first_packet_from": re.search(r'rx_id="(\w+)"', first[0]).group(1) if first else None,
        "busy_drops": sum("system is busy" in l for l in lines),
    })


# ---- tc on lo (scripts/bandwidth_fluctuations.sh)

def tc(*args, check=True):
    p = subprocess.run(TC + list(args), capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"tc {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def tc_lo_setup(rtt_ms):
    """The qdisc of bandwidth_fluctuations.sh:13-17 on lo, loss 0 %."""
    tc("qdisc", "del", "dev", "lo", "root", check=False)
    for cmd in ("qdisc add dev lo root handle 1: htb default 1 r2q 20",
                "class add dev lo parent 1: classid 1:1 htb rate 20mbit",
                "filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dst 0.0.0.0/0 flowid 1:1",
                f"qdisc add dev lo parent 1:1 handle 10: netem delay {rtt_ms / 2:g}ms loss 0%"):
        tc(*cmd.split())


SENT = re.compile(r"Sent (\d+) bytes (\d+) pkt \(dropped (\d+), overlimits (\d+)")


def tc_lo_teardown(out):
    """Saves the counters of tc -s, removes the qdisc; returns the counters."""
    stats = {}
    try:
        text = tc("-s", "qdisc", "show", "dev", "lo") + tc("-s", "class", "show", "dev", "lo")
        open(os.path.join(out, "tc_stats.txt"), "w").write(text)
        blocks = re.split(r"\n(?=\S)", text)
        for name, head in (("htb_class", "class htb 1:1"), ("netem", "qdisc netem 10:")):
            m = [SENT.search(b) for b in blocks if b.startswith(head)]
            if m and m[0]:
                stats[name] = dict(zip(("bytes", "packets", "dropped", "overlimits"), map(int, m[0].groups())))
    except RuntimeError as e:
        stats["error"] = str(e)
    tc("qdisc", "del", "dev", "lo", "root", check=False)
    stats["removed"] = "htb" not in tc("qdisc", "show", "dev", "lo", check=False)
    return stats


def tc_lo_annotate(out, t0, rtt_ms, rates, warmup_s, stats):
    """Adds the rate changes (tc_loop.log) and the tc counters to run.json."""
    steps = []
    for line in open(os.path.join(out, "tc_loop.log")):
        p = line.split()
        if len(p) == 2 and t0:
            try:
                steps.append([round((float(p[0]) - t0) * 1e3, 3), float(p[1])])
            except ValueError:
                pass
    gaps = [b[0] - a[0] for a, b in zip(steps, steps[1:])]
    return update_runs(out, "tc", {
        "rtt_ms": rtt_ms, "rates": os.path.abspath(rates), "warmup_s": warmup_s,
        "steps": steps,     # [ms since t0 when the rate took effect, kbit/s]
        "trace_start_ms": steps[0][0] if steps else None,
        "ghent_start_ms": steps[int(warmup_s)][0] if len(steps) > warmup_s else None,
        "step_ms_mean": sum(gaps) / len(gaps) if gaps else None,
        "lag_ms": steps[-1][0] - steps[0][0] - 1e3 * (len(steps) - 1) if steps else None,
        # htb warns "quantum of class 10001 is big" above 32 Mbit/s (r2q 20); with one
        # class the quantum does not matter
        "warnings": sum(1 for l in open(os.path.join(out, "tc_loop.log")) if l.startswith("Warning:")),
        "errors": sum(1 for l in open(os.path.join(out, "tc_loop.log"))
                      if len(l.split()) != 2 and not l.startswith("Warning:")),
        **stats,
    })


def kill_tree(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in [proc.pid] + descendants(proc.pid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def main():
    argv = sys.argv[1:]
    client_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--shaper", choices=["rattan", "tc-lo"], default="rattan")
    ap.add_argument("--trace", help="rattan trace JSON (make_traces.py); none: delay only")
    ap.add_argument("--tc-rates", help="--shaper tc-lo: rates of make_traces.py --tc-rates")
    ap.add_argument("--rtt", type=float, default=20, help="ms")
    ap.add_argument("--timeout", type=float, default=900, help="s")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.shaper == "tc-lo":
        return main_tc_lo(a, client_args)

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
    report(a, rc, run, "rattan.log", "rattan")


def report(a, rc, run, logname, what):
    status = run["status"] if run else "no run.json"
    if rc == "timeout":
        status = f"timeout after {a.timeout:g} s"
    elif isinstance(rc, str):
        status = rc
    elif rc != 0 and run and run["status"] == "ok":
        status = f"{what} exit {rc}"
    lines = [l for l in open(os.path.join(a.out, logname)) if l.startswith("client:")]
    print(f"run_one: {status}; " + (lines[-1].strip() if lines else f"{what} exit {rc}"))
    sys.exit(0 if status == "ok" else 1)


def main_tc_lo(a, client_args):
    if not a.tc_rates:
        raise SystemExit("--shaper tc-lo needs --tc-rates")
    os.makedirs(a.out, exist_ok=True)
    warmup = argparse.ArgumentParser(add_help=False)
    warmup.add_argument("--warmup", type=float, default=10.0)
    warmup_s = round(warmup.parse_known_args(client_args)[0].warmup)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    loop = proc = t0 = None
    rc, stats = None, {}
    open(os.path.join(a.out, "tc_loop.log"), "w").close()
    open(os.path.join(a.out, "client.log"), "w").close()
    try:
        tc_lo_setup(a.rtt)
        t0 = time.time()
        loop = subprocess.Popen(["bash", os.path.join(REPO, "exp", "tc_lo_loop.sh"), os.path.abspath(a.tc_rates),
                                 str(warmup_s), str(WARMUP_KBIT)],
                                stdout=open(os.path.join(a.out, "tc_loop.log"), "w"), stderr=subprocess.STDOUT)
        cmd = [sys.executable, os.path.join(REPO, "exp", "client.py"), "--server", "127.0.0.1",
               "--t0", repr(t0), "--out", os.path.abspath(a.out), *client_args]
        with open(os.path.join(a.out, "client.log"), "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            rc = proc.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        stop(proc)
        rc = "timeout"
    except Interrupted as e:
        if proc:
            stop(proc)
        rc = f"interrupted by {e}"
    except RuntimeError as e:
        rc = str(e)
    finally:
        # a signal now must not stop the teardown, or lo stays shaped
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if loop:
            kill_tree(loop)
        stats = tc_lo_teardown(a.out)
    run = tc_lo_annotate(a.out, t0, a.rtt, a.tc_rates, warmup_s, stats)
    if not stats.get("removed"):
        print("run_one: WARNING the qdisc on lo could not be removed", file=sys.stderr)
    report(a, rc, run, "client.log", "client")


if __name__ == "__main__":
    main()
