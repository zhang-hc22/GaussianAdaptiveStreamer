"""Helpers shared by the experiment scripts."""
import base64
import hashlib
import json
import os
import subprocess

from cryptography import x509
from cryptography.hazmat.primitives import serialization

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CERT = os.path.join(REPO, "certificates", "ssl_cert.pem")

# Camera fields of a trajectory event, in the order render_image_raw takes them
CAMERA_KEYS = ("angle", "elevation", "x", "y", "z", "fx", "fy", "cx", "cy", "width", "height")


def jpeg_quality(profile):
    """JPEG quality the server uses for a profile (routes.render_handler)."""
    return max(50, 70 - (1 << max(0, min(3, int(profile)))) * 5)


def spki_hash(cert_path=CERT):
    cert = x509.load_pem_x509_certificate(open(cert_path, "rb").read())
    der = cert.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(hashlib.sha256(der).digest()).decode()


def chromium_args(origin, cert_path=CERT):
    """Flags Chrome needs to reach the server over HTTP/3,
    plus two that keep headless timers from being throttled."""
    return [
        f"--origin-to-force-quic-on={origin}",
        f"--ignore-certificate-errors-spki-list={spki_hash(cert_path)}",
        "--enable-experimental-web-platform-features",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
    ]


def git_commit(path=REPO):
    def git(*a):
        return subprocess.run(["git", "-C", path, *a], capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD")
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def traj_name(trajectory):
    """NTHU/train/user3_train.json -> NTHU/train/user3_train"""
    return trajectory[:-5] if trajectory.endswith(".json") else trajectory


def trace_name(trace, scale):
    """bus, 8 -> bus_x8; no trace (delay only) -> none"""
    return f"{trace}_x{scale:g}" if trace else "none"


def run_rel(scene, trajectory, trace, rtt_ms, abr, rep):
    """Path of a run under <results>/runs; trace as trace_name()."""
    return os.path.join(scene, traj_name(trajectory), trace, f"rtt{rtt_ms:g}", abr, f"rep{rep}")


def find_runs(results):
    """Every run directory (one with run.json) under <results>/runs, with the
    fields of its path, sorted by path."""
    base = os.path.join(results, "runs")
    out = []
    for d, dirs, files in os.walk(base, followlinks=True):
        dirs[:] = [x for x in dirs if x != "frames"]
        if "run.json" not in files:
            continue
        rel = os.path.relpath(d, base)
        p = rel.split(os.sep)
        out.append({"dir": d, "rel": rel, "scene": p[0], "trajectory": "/".join(p[1:-4]),
                    "trace": p[-4], "rtt_ms": float(p[-3][3:]), "abr": p[-2], "rep": int(p[-1][3:])})
    return sorted(out, key=lambda r: r["rel"])


def load_events(trajectory):
    """Viewpoint events of a trajectory (a path, or relative to TestMovements/)."""
    path = trajectory if os.path.exists(trajectory) else os.path.join(REPO, "TestMovements", trajectory)
    events = json.load(open(path))
    return events if isinstance(events, list) else events["events"]
