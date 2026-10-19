#!/usr/bin/env python
"""Converts a Ghent trace (traces/<name>.csv, one value per second, kbit/s)
or a mahimahi trace (traces/<name>.trace) into the bandwidth trace rattan
replays (netem-trace JSON), and reads such a trace back for the analysis.

The trace starts with W s of a fixed high bandwidth, used to start the
browser, load the page and warm up the model. The Ghent trace follows at
trace time W, one segment per second, and repeats from its start when it
ends, up to --total s. Values are multiplied by --scale: 8 gives the measured
throughput, 1 the value stored in the repository file.
--offset phi starts the Ghent part at its second phi instead of 0 (the
ablation A1: the trace had played phi s when playback began).

A mahimahi trace (the NYC cellular traces, traces/nyc-*.trace) has one line
per 1500 B delivery opportunity, the millisecond it falls in. It is replayed
per millisecond: a millisecond with n lines gets n * 12 Mbit/s * --scale, one
with none 0 bps (rattan-premium dev replays 0 bps; v0.1.1 panics on it).
Runs of equal milliseconds become one segment. --offset phi starts it at its
second phi, and only MAHIMAHI_S s of it follow the warm-up, against about
1 MB of JSON per 100 s.

--tc-rates writes the same rates for exp/tc_lo_loop.sh instead, one kbit/s
value per line and without the warm-up (the loop adds it); Ghent traces only.

rattan starts the trace at the first packet it forwards; client.py takes t0
just before that (3.4 ms before, measured with a step trace), so trace
time is t - t0.

  python exp/make_traces.py bus --scale 8 --out bus_x8.json
  python exp/make_traces.py bus --scale 8 --offset 101 --out bus_x8_phi101.json
  python exp/make_traces.py nyc-3109898-bus --scale 1 --offset 370 --out nyc.json
"""
import argparse
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ghent values below this become this. car and train contain 0 kbit/s
# seconds; rattan should never get 0 bps, and a very low rate keeps a packet
# in transmission for a long time (a 1400 B packet takes 0.11 s at 100 kbit/s).
FLOOR_BPS = 100_000
# Seconds of a mahimahi trace after the warm-up; a run is over after about
# 110 s of trace time. Then the trace repeats from the warm-up.
MAHIMAHI_S = 300
MAHIMAHI_BPS = 1500 * 8 * 1000   # one line per millisecond


def is_mahimahi(name):
    return os.path.exists(os.path.join(REPO, "traces", f"{name}.trace"))


def ghent_bps(name, scale, offset=0):
    vals = [float(l) for l in open(os.path.join(REPO, "traces", f"{name}.csv")) if l.strip()]
    bps = [max(FLOOR_BPS, round(v * 1000 * scale)) for v in vals]
    offset %= len(bps)
    return bps[offset:] + bps[:offset]


def mahimahi_ms(name, scale, offset=0, seconds=MAHIMAHI_S):
    """bps of each millisecond of seconds s of traces/<name>.trace from its
    second offset on, from its start again when it ends."""
    count = {}
    for l in open(os.path.join(REPO, "traces", f"{name}.trace")):
        if l.strip():
            ms = int(l)
            count[ms] = count.get(ms, 0) + 1
    n = max(count) + 1   # mahimahi repeats the trace after its last line
    start = offset * 1000 % n
    return [round(count.get((start + k) % n, 0) * MAHIMAHI_BPS * scale) for k in range(seconds * 1000)]


def first_bps(name, scale, offset=0):
    """The mean rate of the trace's first second (from offset)."""
    if is_mahimahi(name):
        return round(sum(mahimahi_ms(name, scale, offset, 1)) / 1000)
    return ghent_bps(name, scale, offset)[0]


def build(name, scale, warmup=10, warmup_bps=50_000_000, total=3600, offset=0):
    if is_mahimahi(name):
        runs = []
        for b in mahimahi_ms(name, scale, offset):
            if runs and runs[-1][1] == b:
                runs[-1][0] += 1
            else:
                runs.append([1, b])
        pattern = [[f"{warmup}s", [f"{warmup_bps}bps"]]] + [[f"{d}ms", [f"{b}bps"]] for d, b in runs]
        return {"RepeatedBwPatternConfig": {
            "pattern": [{"TraceBwConfig": {"pattern": pattern}}], "count": 0}}
    bps = ghent_bps(name, scale, offset)
    copies = math.ceil(max(0, total - warmup) / len(bps))
    pattern = [[f"{warmup}s", [f"{warmup_bps}bps"]],
               ["1s", [f"{b}bps" for b in bps * copies]]]
    return {"RepeatedBwPatternConfig": {
        "pattern": [{"TraceBwConfig": {"pattern": pattern}}], "count": 0}}


def tc_rates(name, scale, offset=0):
    """The Ghent part of build() as tc_lo_loop.sh reads it: kbit/s per line."""
    if is_mahimahi(name):
        raise ValueError(f"{name}: tc_lo_loop.sh takes one rate per second, no mahimahi trace")
    return "".join(f"{b / 1000:g}\n" for b in ghent_bps(name, scale, offset))


def _seconds(s):
    for unit, mul in (("ms", 1e-3), ("s", 1.0)):
        if s.endswith(unit):
            return float(s[:-len(unit)]) * mul
    raise ValueError(f"unknown duration {s}")


def segments(path):
    """[(start s, end s, bps)] of a trace written by build() (one pass)."""
    cfg = json.load(open(path))["RepeatedBwPatternConfig"]["pattern"][0]["TraceBwConfig"]
    out, t = [], 0.0
    for dur, bws in cfg["pattern"]:
        d = _seconds(dur)
        for bw in bws:
            out.append((t, t + d, int(bw[:-3])))
            t += d
    return out


def bw_at(segs, t):
    """Trace bandwidth (bps) at trace time t, or None outside the trace."""
    lo, hi = 0, len(segs)
    while lo < hi:
        mid = (lo + hi) // 2
        if segs[mid][1] <= t:
            lo = mid + 1
        else:
            hi = mid
    return segs[lo][2] if lo < len(segs) and segs[lo][0] <= t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="traces/<name>.csv or traces/<name>.trace")
    ap.add_argument("--scale", type=float, default=8)
    ap.add_argument("--warmup", type=int, default=10, help="W, s")
    ap.add_argument("--warmup-mbps", type=float, default=50)
    ap.add_argument("--total", type=int, default=3600, help="trace length, s")
    ap.add_argument("--offset", type=int, default=0, help="phi: first second of the Ghent trace")
    ap.add_argument("--tc-rates", action="store_true", help="write the rates for tc_lo_loop.sh")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.tc_rates:
        open(a.out, "w").write(tc_rates(a.name, a.scale, a.offset))
    else:
        cfg = build(a.name, a.scale, a.warmup, round(a.warmup_mbps * 1e6), a.total, a.offset)
        json.dump(cfg, open(a.out, "w"), separators=(",", ":"))
    if is_mahimahi(a.name):
        ms = mahimahi_ms(a.name, a.scale, a.offset)
        sec = [sum(ms[k:k + 1000]) / 1000 for k in range(0, len(ms), 1000)]
        print(f"{a.out}: {a.name} x{a.scale:g} from second {a.offset}, {len(sec)} s per millisecond, "
              f"mean {sum(sec) / len(sec) / 1e6:.2f} Mbit/s, seconds below 0.35 Mbit/s "
              f"{sum(b < 350_000 for b in sec)}, after {a.warmup} s at {a.warmup_mbps:g} Mbit/s")
        return
    bps = ghent_bps(a.name, a.scale)
    print(f"{a.out}: {a.name} x{a.scale:g} from second {a.offset}, {len(bps)} s, "
          f"mean {sum(bps) / len(bps) / 1e6:.2f} Mbit/s, min {min(bps) / 1e6:.2f}, "
          f"after {a.warmup} s at {a.warmup_mbps:g} Mbit/s")


if __name__ == "__main__":
    main()
