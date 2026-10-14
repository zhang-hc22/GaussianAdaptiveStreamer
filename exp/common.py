"""Helpers shared by the experiment scripts."""
import base64
import hashlib
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
