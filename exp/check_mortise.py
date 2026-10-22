#!/usr/bin/env python
"""Replays Mortise's rules (sparkrtc tradeoff_control.cc, from AlphaRTC-mortise
8b09a04) on the TigasMortise lines of a run's sender.log
(exp/mortise_log.py) and checks that the logged values follow:

- beta, alpha and ai are 0.93 - 0.1 u, 1.01 + 0.3 (1 - u), 0.7 + 20 (1 - u);
- lambda = (B' + 0.15) / 150, B' the last period's bitrate in Mbit/s
  (1 Mbit/s before the first period); q = B - lambda R and q_last = B' -
  lambda R' (R' 200 ms before the first period);
- u_min = 0.15 min(R, 250) / 50 + 0.15, or 0.7, 0.8, 0.9 above 1.7, 2.2,
  2.5 Mbit/s;
- the branch and the new u: after a change point, cp_exit with
  u = max(u, u_min) if the last decrease is less than 3 s ago, else
  cp_decay with u = max(u / 1.81, u_min); otherwise rule_up (u + 0.2) if
  u went up and Q rose or u went down and Q fell, rule_down
  (max(u / 1.78, u_min)) for the opposite, else rule_hold; u at most 1;
  cp_enter only in the rule state with no decrease for 3 s, u =
  min(0.2 + B' / 5, 0.9) below 1 Mbit/s, else min(0.3 + B' / 5, 0.9);
- each TigasAimdDecrease line: decreased = beta * throughput - 5 kbit/s
  (or beta * link capacity when that was above the current target), with
  beta of the last TigasMortise line before it (0.85 before the first).

  python exp/check_mortise.py RUN_DIR ...

Prints the mismatches and a count per run; exits 1 if there is any.
"""
import os
import re
import sys

from mortise_log import FIELD, LINE

TOL = 2e-3   # the log prints 6 significant digits
DECREASE = re.compile(r"TigasAimdDecrease throughput (\d+) current (\d+) decreased (\d+) link_capacity (-?\d+)")


def kbps(bps):
    return int(bps / 1000 + 0.5)   # DataRate::kbps() rounds


def params(u):
    return 0.93 - 0.1 * u, 1.01 + 0.3 * (1 - u), 0.7 + 20 * (1 - u)


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(b))


def check(path):
    bad, n_periods, n_dec = [], 0, 0
    state, u, u_last = "probing", 0.8, 0.8
    b_last_bps, r_last, beta = 1e6, 200.0, 0.85

    def expect(i, what, got, want):
        if not close(got, want):
            bad.append(f"line {i}: {what} {got:.6g}, expected {want:.6g}")

    for i, line in enumerate(open(path, errors="replace"), 1):
        m = DECREASE.search(line)
        if m:
            tput, cur, dec, cap = map(int, m.groups())
            want = tput * beta
            want = want - 5000 if want > 5000 else want
            if want > cur and cap > 0:
                want = beta * cap
            if not close(dec, want, 1e-4):
                bad.append(f"line {i}: decreased {dec}, expected {want:.0f} (beta {beta:.4g})")
            n_dec += 1
            continue
        m = LINE.search(line.rstrip("\n"))
        if not m:
            continue
        r = {k: v if k == "branch" else float(v) for k, v in FIELD.findall(m.group(1))}
        br = r["branch"]
        expect(i, "u_before", r["u_before"], u)
        expect(i, "lambda", r["lambda"], (kbps(b_last_bps) / 1000 + 0.15) / 150)
        since = r["since_decrease_ms"]
        recent = 0 <= since < 3000
        if br == "cp_enter":
            if state != "probing" or recent:
                bad.append(f"line {i}: cp_enter in state {state}, since_decrease_ms {since:.0f}")
            k = kbps(b_last_bps)
            new, new_last, state = min((0.2 if k < 1000 else 0.3) + k / 5000, 0.9), u, "cp"
        elif br.startswith("skip"):
            new, new_last = u, u_last
        else:
            n_periods += 1
            b, rtt = r["bitrate"], r["rtt"]
            lam = r["lambda"]
            expect(i, "q", r["q"], kbps(b) / 1000 - lam * rtt)
            expect(i, "q_last", r["q_last"], kbps(b_last_bps) / 1000 - lam * r_last)
            u_min = 0.15 * min(rtt, 250) / 50 + 0.15
            for lim, v in ((1700, 0.7), (2200, 0.8), (2500, 0.9)):
                if kbps(b) > lim:
                    u_min = v
            expect(i, "u_min", r["u_min"], u_min)
            q, ql = r["q"], r["q_last"]
            if state == "cp":
                want_br = "cp_exit" if recent else "cp_decay"
                new = min(max(u, u_min), 1.0) if recent else min(max(u / 1.81, u_min), 1.0)
                new_last = u
                if recent:
                    state = "probing"
            elif (u > u_last and q > ql) or (u < u_last and q < ql):
                want_br, new, new_last = "rule_up", min(u + 0.2, 1.0), u
            elif (u < u_last and q > ql) or (u > u_last and q < ql):
                want_br, new, new_last = "rule_down", min(max(u / 1.78, u_min), 1.0), u
            else:
                want_br, new, new_last = "rule_hold", u, u_last
            if br != want_br:
                bad.append(f"line {i}: branch {br}, expected {want_br}")
            b_last_bps, r_last = b, rtt
        expect(i, "u", r["u"], new)
        expect(i, "u_last", r["u_last"], new_last)
        for k, v in zip(("beta", "alpha", "ai"), params(r["u"])):
            expect(i, k, r[k], v)
        expect(i, "last_bitrate", r["last_bitrate"], b_last_bps)
        u, u_last, beta = r["u"], r["u_last"], r["beta"]
    return bad, n_periods, n_dec


def main():
    failed = False
    for d in sys.argv[1:]:
        bad, n, n_dec = check(os.path.join(d, "sender.log"))
        for b in bad[:20]:
            print(f"  {b}")
        print(f"{d}: {n} periods, {n_dec} decreases, {len(bad)} mismatches")
        failed |= bool(bad) or n == 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
