#!/usr/bin/env python
"""Phase-0 probe: a headless Chromium that talks to the TIGAS server over
HTTP/3, the way the experiment client will.

  load  open /player and wait until the page has drawn its first frame.
  step  open /player, then keep --inflight /render requests in flight at a
        fixed viewpoint and profile until --until seconds after t0, recording
        when every chunk of every response arrives. Afterwards send one small
        and one large (--upload-mb) request to time the uplink.

t0 is the wall-clock time just before page.goto(), the moment the experiment
client will treat as the start of the trace. The server address is --server,
else $RATTAN_BASE (inside `rattan link`), else 127.0.0.1. All times in the
output are Unix epoch seconds.
"""
import argparse
import base64
import hashlib
import json
import os
import time

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from playwright.sync_api import sync_playwright


def spki_hash(cert_path):
    cert = x509.load_pem_x509_certificate(open(cert_path, "rb").read())
    der = cert.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(hashlib.sha256(der).digest()).decode()


# Runs in the page. Times are performance.now(); the caller adds timeOrigin.
STEP_JS = """
async ({untilMs, inflight, body}) => {
    const recs = [];
    async function one(b) {
        const r = {send: performance.now(), chunks: []};
        const resp = await fetch('/render', {method: 'POST', cache: 'no-store',
            headers: {'Content-Type': 'application/json'}, body: b});
        r.hdr = performance.now();
        r.status = resp.status;
        r.renderMs = parseFloat(resp.headers.get('X-Render-Time-Ms'));
        const reader = resp.body.getReader();
        for (;;) {
            const {done, value} = await reader.read();
            if (done) break;
            r.chunks.push([performance.now(), value.length]);
        }
        r.end = performance.now();
        return r;
    }
    async function loop() {
        while (performance.now() < untilMs) recs.push(await one(body));
    }
    await Promise.all(Array.from({length: inflight}, loop));
    return {timeOrigin: performance.timeOrigin, recs};
}
"""

UPLOAD_JS = """
async ({body}) => {
    const t = performance.now();
    const resp = await fetch('/render', {method: 'POST', cache: 'no-store',
        headers: {'Content-Type': 'application/json'}, body});
    const hdr = performance.now();
    await resp.arrayBuffer();
    return {bytes: body.length, send: t, hdr, end: performance.now(),
            renderMs: parseFloat(resp.headers.get('X-Render-Time-Ms'))};
}
"""

FIRST_FRAME_JS = """
() => document.getElementById('stage_placeholder')?.style.display === 'none'
"""

PROTOCOLS_JS = """
() => performance.getEntries()
    .filter(e => e.nextHopProtocol !== undefined)
    .map(e => [e.name.replace(location.origin, ''), e.nextHopProtocol])
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["load", "step"])
    ap.add_argument("--server", default=os.environ.get("RATTAN_BASE", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=4433)
    ap.add_argument("--cert", default="certificates/ssl_cert.pem")
    ap.add_argument("--scene", default="train")
    ap.add_argument("--trajectory", default="TestMovements/NTHU/train/user3_train.json")
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--inflight", type=int, default=4)
    ap.add_argument("--until", type=float, default=30.0, help="step: stop at t0 + this many s")
    ap.add_argument("--upload-mb", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    origin = f"{a.server}:{a.port}"
    args = [
        f"--origin-to-force-quic-on={origin}",
        f"--ignore-certificate-errors-spki-list={spki_hash(a.cert)}",
        "--enable-experimental-web-platform-features",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
    ]
    out = {"mode": a.mode, "server": a.server, "chromium_args": args}
    console = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=args)
        out["chromium"] = browser.version
        page = browser.new_context(ignore_https_errors=True).new_page()
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))

        out["t0"] = time.time()
        page.goto(f"https://{origin}/player?modelId={a.scene}", timeout=30000)
        out["t_loaded"] = time.time()
        page.wait_for_function(FIRST_FRAME_JS, timeout=30000, polling=10)
        out["t_first_frame"] = time.time()
        out["protocols"] = page.evaluate(PROTOCOLS_JS)

        if a.mode == "step":
            ev = json.load(open(a.trajectory))[0]
            req = {k: ev[k] for k in ("angle", "elevation", "x", "y", "z",
                                      "fx", "fy", "cx", "cy", "width", "height")}
            req.update(modelId=a.scene, profile=a.profile)
            # page time of t0 + until
            origin_s = page.evaluate("performance.timeOrigin") / 1000
            until_ms = (out["t0"] + a.until - origin_s) * 1000
            res = page.evaluate(STEP_JS, {"untilMs": until_ms, "inflight": a.inflight,
                                          "body": json.dumps(req)})
            to = res["timeOrigin"] / 1000
            out["requests"] = [{
                "send": to + r["send"] / 1000, "hdr": to + r["hdr"] / 1000,
                "end": to + r["end"] / 1000, "status": r["status"],
                "render_ms": r["renderMs"],
                "chunks": [[round(to + t / 1000, 6), n] for t, n in r["chunks"]],
            } for r in res["recs"]]

            pad = "x" * int(a.upload_mb * 1e6)
            out["upload"] = []
            for body in (json.dumps(req), json.dumps(dict(req, pad=pad))):
                u = page.evaluate(UPLOAD_JS, {"body": body})
                out["upload"].append({"bytes": u["bytes"], "render_ms": u["renderMs"],
                                      "send": to + u["send"] / 1000,
                                      "hdr": to + u["hdr"] / 1000,
                                      "end": to + u["end"] / 1000})
        browser.close()
    out["console"] = console[-20:]
    json.dump(out, open(a.out, "w"))
    print(f"{a.mode}: first frame {out['t_first_frame'] - out['t0']:.3f} s after t0, "
          f"protocols {sorted({p for _, p in out['protocols']})}")


if __name__ == "__main__":
    main()
