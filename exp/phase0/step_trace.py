#!/usr/bin/env python
"""Step trace for phase 0: writes the trace rattan replays, and checks a
`probe.py step` run against it.

The trace: W s at HI (page load), then LO and HI alternating every PERIOD s
for N segments, then TAIL s at LO for the uplink test. rattan starts the trace
at the first packet it forwards, and logs "first-payload is disabled" at that
moment (with the first-packet feature, first-payload off).

Downlink cells are bandwidth then delay, so a bandwidth change at trace time b
reaches the client at trace start + b + downlink delay.

  python step_trace.py write trace.json
  python step_trace.py check --probe step.json --rattan-log rattan.log
"""
import argparse
import bisect
import json
import re
from datetime import datetime, timezone

W, PERIOD, N, TAIL = 10, 2, 10, 20
LO, HI = 2, 50  # Mbit/s
DOWN_DELAY = 0.010


def segments():
    segs = [(W, HI)] + [(PERIOD, LO if i % 2 == 0 else HI) for i in range(N)]
    return segs + [(TAIL, LO)]


def write(path):
    pattern = [[f"{d}s", [f"{bw}Mbps"]] for d, bw in segments()]
    cfg = {"RepeatedBwPatternConfig": {
        "pattern": [{"TraceBwConfig": {"pattern": pattern}}], "count": 0}}
    json.dump(cfg, open(path, "w"))


def edges():
    """[(trace time, 'rise' or 'fall')] of the alternating part."""
    out, t, prev = [], 0, None
    for d, bw in segments():
        if prev is not None and bw != prev:
            out.append((t, "rise" if bw > prev else "fall"))
        prev, t = bw, t + d
    return out[:-1] if out[-1][0] >= W + PERIOD * N else out


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log_time(line):
    ts = ANSI.sub("", line).split()[0]
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()


def check(probe_path, log_path):
    p = json.load(open(probe_path))
    t0 = p["t0"]
    lines = open(log_path, errors="replace").read().splitlines()
    start = [log_time(l) for l in lines if "first-payload is disabled" in l]
    first = [ANSI.sub("", l) for l in lines if "FirstPacket" in l]
    busy = sum("system is busy" in l for l in lines)
    ts = start[0]

    chunks = sorted(c for r in p["requests"] for c in r["chunks"])
    times = [t for t, _ in chunks]
    cum = [0]
    for _, n in chunks:
        cum.append(cum[-1] + n)

    def bytes_between(a, b):  # chunks with a < t <= b
        return cum[bisect.bisect_right(times, b)] - cum[bisect.bisect_right(times, a)]

    def rate(a, b):
        return bytes_between(a, b) * 8 / (b - a) / 1e6

    # A chunk is fast when the 5 ms ending at it carried >= 6 kB (~10 Mbit/s);
    # at 2 Mbit/s 5 ms carry 1.25 kB.
    def fast(i):
        return bytes_between(times[i] - 0.005, times[i]) >= 6000

    res = {"t0": t0, "trace_start": ts, "start_minus_t0_ms": (ts - t0) * 1e3,
           "first_packet_log": first[:2], "busy_drops": busy,
           "first_frame_s": p["t_first_frame"] - t0, "edges": [], "phases": []}
    for b, kind in edges():
        expect = ts + b + DOWN_DELAY
        lo = bisect.bisect_left(times, expect - 0.5)
        hi = bisect.bisect_right(times, expect + 0.5)
        idx = [i for i in range(lo, hi) if fast(i)]
        if kind == "rise":
            # first fast chunk after a slow stretch
            got = next((times[i] for i in idx if bytes_between(times[i] - 0.3, times[i] - 0.005) < 0.3 * 3 / 8 * 1e6), None)
        else:
            # last fast chunk before a slow stretch
            got = next((times[i] for i in reversed(idx) if bytes_between(times[i], times[i] + 0.3) < 0.3 * 3 / 8 * 1e6), None)
        res["edges"].append({
            "trace_s": b, "kind": kind,
            "vs_trace_start_ms": None if got is None else (got - expect) * 1e3,
            "vs_t0_ms": None if got is None else (got - (t0 + b + DOWN_DELAY)) * 1e3})
    t = 0
    for d, bw in segments():
        if W <= t < W + PERIOD * N:
            a, b = ts + t + DOWN_DELAY + 0.3, ts + t + d + DOWN_DELAY - 0.1
            res["phases"].append({"trace_s": t, "limit_mbps": bw, "got_mbps": rate(a, b)})
        t += d

    up = p["upload"]
    res["upload"] = {"small_s": up[0]["hdr"] - up[0]["send"],
                     "large_s": up[1]["hdr"] - up[1]["send"],
                     "large_bytes": up[1]["bytes"]}
    res["requests"] = len(p["requests"])
    res["frame_bytes"] = cum[-1] / max(1, len(p["requests"]))
    return res


def report(r):
    print(f"t0 -> trace start (rattan log): {r['start_minus_t0_ms']:+.1f} ms; "
          f"first packet: {r['first_packet_log'] or 'no debug line'}")
    print(f"first frame drawn {r['first_frame_s']:.2f} s after t0; "
          f"{r['requests']} requests, {r['frame_bytes'] / 1e3:.1f} kB each; "
          f"rattan busy drops: {r['busy_drops']}")
    print("edge   trace_s  vs_trace_start_ms  vs_t0_ms")
    for e in r["edges"]:
        f = lambda v: "   n/a" if v is None else f"{v:+7.1f}"
        print(f"{e['kind']:5}  {e['trace_s']:7}  {f(e['vs_trace_start_ms']):>17}  {f(e['vs_t0_ms']):>8}")
    print("phase  trace_s  limit  got (Mbit/s, app bytes)")
    for ph in r["phases"]:
        print(f"       {ph['trace_s']:7}  {ph['limit_mbps']:5}  {ph['got_mbps']:6.2f}")
    u = r["upload"]
    print(f"uplink: small request {u['small_s'] * 1e3:.0f} ms to headers, "
          f"{u['large_bytes'] / 1e6:.1f} MB request {u['large_s'] * 1e3:.0f} ms "
          f"(at {LO} Mbit/s it would take {u['large_bytes'] * 8 / LO / 1e6:.1f} s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("out")
    c = sub.add_parser("check")
    c.add_argument("--probe", required=True)
    c.add_argument("--rattan-log", required=True)
    c.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.out)
    else:
        r = check(a.probe, a.rattan_log)
        report(r)
        if a.out:
            json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
