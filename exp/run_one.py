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

With --transport webrtc (client.py --mode webrtc)
it first starts, on the host, peerconnection_server and tigas_sender from
--webrtc-bin, waits for the sender to sign in, then runs the client as
above. The render worker (webrtc/render_worker.py) must already listen on
--render-socket; run_all.py keeps one running. After the client the sender
exits on the page's bye (or is stopped), and this script adds to the run
directory:
- sender.log, signaling.log, and the sender's sender_info.json,
  sender_frames.ndjson (one line per rendered viewpoint) and
  sender_stats.ndjson (getStats every 100 ms);
- sender_encoded.ndjson: one line per encoded frame from sender.log (bytes,
  qp, key frame, encode done and first packet sent, ms since t0);
- render_worker.ndjson: the worker's lines of this run, if --render-log;
- in frames.ndjson, per viewpoint: arrive, render_start, render_end,
  queue_ahead (sender_frames), bytes, qp, key, enc, pkt (sender_encoded);
- run.json "sender": the switches, field trials, build, exit code, counts.
--shaper none runs the client without shaping, at 127.0.0.1 (a local check).

On SIGTERM or SIGINT the run is stopped the same way as on a timeout, so
rattan can remove its network namespaces (run_all.py relies on this).

  python exp/run_one.py --trace T.json --rtt 20 --out DIR -- \\
      --trajectory NTHU/train/user3_train.json --abr l2a --seed 1
  python exp/run_one.py --shaper tc-lo --tc-rates R.txt --rtt 20 --out DIR -- ...
  python exp/run_one.py --transport webrtc --trace T.json --rtt 20 --out DIR -- \\
      --mode webrtc --trajectory NTHU/train/user3_train.json --resolution 1280x720
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
WEBRTC_BIN = os.path.join(REPO, "webrtc", "bin")
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
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:   # no driver (a local test with the fake worker)
        return None
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
    ap.add_argument("--shaper", choices=["rattan", "tc-lo", "none"], default="rattan")
    ap.add_argument("--transport", choices=["quic", "webrtc"], default="quic")
    ap.add_argument("--webrtc-bin", default=WEBRTC_BIN, help="tigas_sender and peerconnection_server")
    ap.add_argument("--signaling-port", type=int, default=8888)
    ap.add_argument("--render-socket", default="/tmp/tigas_render.sock")
    ap.add_argument("--render-log", help="the worker's --log, to copy this run's lines from")
    ap.add_argument("--candidate-prefix", default="10.2.",
                    help="tigas_sender sends only local candidates starting with this "
                    "(the host side of rattan); --shaper none sends all")
    ap.add_argument("--sender-arg", action="append", default=[], help="extra tigas_sender flag")
    ap.add_argument("--trace", help="rattan trace JSON (make_traces.py); none: delay only")
    ap.add_argument("--tc-rates", help="--shaper tc-lo: rates of make_traces.py --tc-rates")
    ap.add_argument("--rtt", type=float, default=20, help="ms")
    ap.add_argument("--timeout", type=float, default=900, help="s")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.transport == "webrtc":
        return main_webrtc(a, client_args)
    if a.shaper == "tc-lo":
        return main_tc_lo(a, client_args)
    if a.shaper == "none":
        raise SystemExit("--shaper none is for --transport webrtc")

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


# ---- WebRTC (tigas_sender)

def tcp_listening(port):
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in open(f).read().splitlines()[1:]:
            p = line.split()
            if int(p[1].split(":")[1], 16) == port and p[3] == "0A":
                return True
    return False


def wait_for(cond, timeout, proc=None):
    t = time.time() + timeout
    while time.time() < t:
        if cond():
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.05)
    return cond()


def end_proc(proc, wait_s):
    """Waits wait_s for proc to exit, then SIGTERM, then SIGKILL; returns its
    exit code (negative: the signal that ended it)."""
    for sig in (None, signal.SIGTERM, signal.SIGKILL):
        if sig is not None and proc.poll() is None:
            proc.send_signal(sig)
        try:
            return proc.wait(timeout=wait_s if sig is None else 5)
        except subprocess.TimeoutExpired:
            pass
    return proc.wait()


ENCODED = re.compile(r"Send Frame Size: (\d+) current time: (\d+) rtp_ts: (\d+) qp: (-?\d+) key: (\d)")
FIRST_PACKET = re.compile(r"Time Stamp Start:\d+:(\d+)")


def read_ndjson(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def annotate_sender(out, t0, rc, bin_dir, render_log, t_start, t_end):
    """sender_encoded.ndjson, render_worker.ndjson, the sender's columns of
    frames.ndjson and run.json "sender" (see the docstring)."""
    rendered = read_ndjson(os.path.join(out, "sender_frames.ndjson"))
    # rtc::TimeMillis of the log is the monotonic clock of t_push_mono
    offs = sorted(f["t_push"] - f["t_push_mono"] for f in rendered)
    off = offs[len(offs) // 2] if offs else None
    t0_ms = t0 * 1e3 if t0 else None

    def rel_mono(t):
        return None if off is None or t0_ms is None else round(t + off - t0_ms, 3)

    def rel(t):
        return None if t is None or t0_ms is None else round(t - t0_ms, 3)
    enc, starts = [], []
    log = os.path.join(out, "sender.log")
    for line in open(log, errors="replace") if os.path.exists(log) else []:
        m = ENCODED.search(line)
        if m:
            enc.append({"rtp_ts": int(m[3]), "bytes": int(m[1]), "qp": int(m[4]), "key": int(m[5]),
                        "t_mono": int(m[2])})
            continue
        m = FIRST_PACKET.search(line)
        if m:
            starts.append(int(m[1]))
    # first packet of each frame: the first one sent at or after its encode
    k = 0
    for e in enc:
        while k < len(starts) and starts[k] < e["t_mono"]:
            k += 1
        e["pkt"] = rel_mono(starts[k]) if k < len(starts) else None
        k += k < len(starts)
        e["enc"] = rel_mono(e.pop("t_mono"))
    with open(os.path.join(out, "sender_encoded.ndjson"), "w") as fh:
        for e in enc:
            fh.write(json.dumps(e) + "\n")
    if render_log and os.path.exists(render_log):
        with open(os.path.join(out, "render_worker.ndjson"), "w") as fh:
            for line in open(render_log):
                try:
                    if t_start * 1e3 <= json.loads(line)["t_recv"] <= t_end * 1e3:
                        fh.write(line)
                except (ValueError, KeyError):
                    pass

    by_rtp = {e["rtp_ts"]: e for e in enc}
    by_i = {f["i"]: f for f in rendered}
    path = os.path.join(out, "frames.ndjson")
    if os.path.exists(path):
        lines = []
        for f in read_ndjson(path):
            s = by_i.get(f["i"])
            if s:
                f.update(arrive=rel(s["t_arrive"]), render_start=rel(s["t_render_start"]),
                         render_end=rel(s["t_render_end"]), queue_ahead=s["queue_ahead"],
                         sender_rtp_ts=s["rtp_ts"])
                e = by_rtp.get(s["rtp_ts"])
                if e:
                    f.update(bytes=e["bytes"], qp=e["qp"], key=e["key"], enc=e["enc"], pkt=e["pkt"])
            lines.append(json.dumps(f))
        open(path, "w").write("\n".join(lines) + ("\n" if lines else ""))

    info_path = os.path.join(out, "sender_info.json")
    build_path = os.path.join(bin_dir, "BUILD_INFO")
    playback = [f for f in rendered if f["i"] >= 0]
    return update_runs(out, "sender", {
        "rc": rc,
        "info": json.load(open(info_path)) if os.path.exists(info_path) else None,
        "build": open(build_path).read().strip() if os.path.exists(build_path) else None,
        "rendered": len(rendered), "rendered_playback": len(playback),
        "meta_not_sent": sum(not f.get("meta_sent", True) for f in rendered),
        "encoded": len(enc), "key_frames": sum(e["key"] for e in enc),
        "encoded_without_render": sum(e["rtp_ts"] not in {f["rtp_ts"] for f in rendered} for e in enc),
        "mono_to_wall_ms": off,
    })


def main_webrtc(a, client_args):
    os.makedirs(a.out, exist_ok=True)
    out = os.path.abspath(a.out)
    for name in ("tigas_sender", "peerconnection_server"):
        if not os.access(os.path.join(a.webrtc_bin, name), os.X_OK):
            raise SystemExit(f"run_one: no {name} in {a.webrtc_bin}")
    if tcp_listening(a.signaling_port):
        raise SystemExit(f"run_one: TCP port {a.signaling_port} in use; another signaling server?")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    sig = sender = proc = None
    rc, sender_rc, t_start = None, None, time.time()
    shaped = a.shaper == "rattan"
    logname = "rattan.log" if shaped else "client.log"
    open(os.path.join(out, logname), "w").close()
    try:
        sig = subprocess.Popen([os.path.join(a.webrtc_bin, "peerconnection_server"), f"--port={a.signaling_port}"],
                               stdout=open(os.path.join(out, "signaling.log"), "w"), stderr=subprocess.STDOUT)
        if not wait_for(lambda: tcp_listening(a.signaling_port), 5, sig):
            raise RuntimeError("peerconnection_server did not listen; see signaling.log")
        sender_log = os.path.join(out, "sender.log")
        sender = subprocess.Popen(
            [os.path.join(a.webrtc_bin, "tigas_sender"), "--server=127.0.0.1", f"--port={a.signaling_port}",
             f"--render_socket={a.render_socket}", "--render_connect_timeout_s=30", f"--out_dir={out}",
             f"--candidate_prefix={a.candidate_prefix if shaped else ''}",
             f"--max_duration_s={int(a.timeout) + 60}", *a.sender_arg],
            stdout=open(sender_log, "w"), stderr=subprocess.STDOUT)
        if not wait_for(lambda: "Signed in to the signaling server" in open(sender_log, errors="replace").read(),
                        40, sender):
            raise RuntimeError("tigas_sender did not sign in (render worker running?); see sender.log")
        client = [sys.executable, os.path.join(REPO, "exp", "client.py"), "--out", out,
                  "--signaling-port", str(a.signaling_port), *client_args]
        if shaped:
            cmd = rattan_cmd(a.trace, a.rtt) + ["--", *client]
            env = dict(os.environ, RUST_LOG="info,rattan_core::core=debug")
        else:
            cmd, env = client + ["--server", "127.0.0.1"], dict(os.environ)
        with open(os.path.join(out, logname), "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
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
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if sender:
            # it exits on the page's bye; give it a moment
            sender_rc = end_proc(sender, 15 if isinstance(rc, int) else 0)
        if sig:
            end_proc(sig, 0)
    t_end = time.time()
    try:
        t0 = json.load(open(os.path.join(out, "run.json"))).get("t0")
    except (OSError, ValueError):
        t0 = None
    run = annotate(out, a.trace, a.rtt) if shaped else None
    run = annotate_sender(out, t0, sender_rc, a.webrtc_bin, a.render_log, t_start, t_end) or run
    if isinstance(rc, int) and rc == 0 and sender_rc != 0 and run and run["status"] == "ok":
        rc = f"tigas_sender exit {sender_rc}"
    report(a, rc, run, logname, "rattan" if shaped else "client")


if __name__ == "__main__":
    main()
