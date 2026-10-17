#!/usr/bin/env bash
# Checks that the experiment machine has everything the headless experiments
# need. Run from the repo root without sudo:
#
#   bash exp/setup/check_env.sh
#
# Every line starts with PASS, WARN or FAIL. The first gsplat check compiles
# the CUDA kernels and takes a few minutes; later runs use the cache.
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-tigas}
FFMPEG=${FFMPEG:-$HOME/opt/ffmpeg/bin/ffmpeg}
RATTAN=${RATTAN:-/usr/local/bin/rattan}
RATTAN_COMMIT=${RATTAN_COMMIT:-a63a99c4}
SCENE=${SCENE:-train}

nfail=0
pass() { echo "PASS  $1: $2"; }
warn() { echo "WARN  $1: $2"; }
fail() { echo "FAIL  $1: $2"; nfail=$((nfail + 1)); }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ---- hardware
if out=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1); then
    pass gpu "$out"
else
    fail gpu "nvidia-smi: $out"
fi
gov=$(sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | tr '\n' ' ')
if [[ $gov == "performance " ]]; then
    pass cpu "$(nproc) cores, governor $gov"
else
    warn cpu "$(nproc) cores, governor $gov(root_setup.sh sets performance)"
fi
pass disk "$(df -h --output=avail . | tail -1 | tr -d ' ') free under $(pwd)"

# ---- python environment
if [[ -f $CONDA_DIR/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    # The compiler activation scripts read unset variables, so set -u has to
    # be off while activating, or bash exits here without a message.
    set +u
    conda activate "$CONDA_ENV" 2>"$TMP/conda.err"
    rc=$?
    set -u
    if [[ $rc -eq 0 ]]; then
        pass conda "env $CONDA_ENV, $(python --version 2>&1), nvcc $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | cut -d' ' -f2), CC $($CC -dumpversion 2>/dev/null)"
    else
        fail conda "cannot activate env $CONDA_ENV: $(tail -1 "$TMP/conda.err")"
    fi
else
    fail conda "no conda under $CONDA_DIR"
fi

py() {  # name, python code; the code prints the detail line
    local name=$1 code=$2 out
    if out=$(python -c "$code" 2>"$TMP/$name.err"); then
        pass "$name" "$(tail -1 <<<"$out")"
    else
        fail "$name" "$(tail -1 "$TMP/$name.err")"
    fi
}

py torch '
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'

py gsplat '
import torch, gsplat
from gsplat import rasterization
n, dev = 2000, "cuda"
torch.manual_seed(0)
means = torch.randn(n, 3, device=dev) * 0.5 + torch.tensor([0, 0, 3.0], device=dev)
quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
scales = torch.full((n, 3), 0.02, device=dev)
opac = torch.ones(n, device=dev)
colors = torch.rand(n, 1, 3, device=dev)
K = torch.tensor([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]], device=dev)[None]
img, _, _ = rasterization(means, quats, scales, opac, colors, torch.eye(4, device=dev)[None], K, 128, 128, sh_degree=0)
assert img.abs().sum() > 0, "empty render"
print(gsplat.__version__, "rasterization OK", tuple(img.shape))'

py server-deps '
import aioquic, starlette, plyfile, scipy, PIL
print("aioquic", aioquic.__version__, "starlette", starlette.__version__)'

py metric-deps '
import piq, torch
x = torch.rand(1, 3, 64, 64)
print("piq", piq.__version__,
      "lpips", round(piq.LPIPS()(x, x * 0.9).item(), 4),
      "dists", round(piq.DISTS()(x, x * 0.9).item(), 4))'

py chromium '
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    print("playwright chromium", b.version)
    b.close()'

# ---- ffmpeg with libvmaf
if [[ -x $FFMPEG ]] && "$FFMPEG" -hide_banner -filters 2>/dev/null | grep ' libvmaf ' >/dev/null; then
    score=$("$FFMPEG" -hide_banner -nostats \
        -f lavfi -i testsrc2=size=320x240:rate=10:duration=1 \
        -f lavfi -i testsrc2=size=320x240:rate=10:duration=1 \
        -lavfi libvmaf -f null - 2>&1 | grep -oE 'VMAF score: [0-9.]+')
    if [[ -n $score ]]; then
        pass ffmpeg "$("$FFMPEG" -version | head -1 | cut -d' ' -f1-3), libvmaf $score"
    else
        fail ffmpeg "libvmaf did not produce a score"
    fi
else
    fail ffmpeg "$FFMPEG missing or built without libvmaf"
fi

# ---- rattan
if [[ -x $RATTAN ]]; then
    ver=$("$RATTAN" --version | tr '\n' ' ')
    caps=$(getcap "$RATTAN")
    if [[ $ver == *"$RATTAN_COMMIT"* ]]; then
        pass rattan-version "$ver"
    else
        fail rattan-version "want commit $RATTAN_COMMIT, got: $ver"
    fi
    if [[ $caps == *cap_net_admin* && $caps == *cap_sys_admin* ]]; then
        pass rattan-caps "$caps"
    else
        fail rattan-caps "missing capabilities (run root_setup.sh): $caps"
    fi
else
    fail rattan-version "$RATTAN not found (run root_setup.sh)"
fi

# Downlink limited to 10 Mbit/s, 10 ms each way. iperf3 -R sends from the
# host (server side) to ns-left (client side), i.e. along the downlink. The
# client of -R is the receiver and reports no RTT, so ping measures the RTT
# first, on the idle link. rattan logs to stdout, so the commands inside it
# write their results to files.
if [[ -x $RATTAN ]] && ! command -v iperf3 >/dev/null; then
    fail rattan-link "iperf3 not installed"
elif [[ -x $RATTAN ]]; then
    port=5299
    iperf3 -s -1 -p $port >/dev/null 2>&1 &
    srv=$!
    sleep 0.5
    timeout 30 "$RATTAN" link --downlink-bandwidth 10Mbps \
        --downlink-delay 10ms --uplink-delay 10ms -- \
        bash -c "ping -c 5 -i 0.2 -q \$RATTAN_BASE >'$TMP/ping.txt' 2>&1;
                 iperf3 -c \$RATTAN_BASE -p $port -R -t 4 -J --connect-timeout 3000 >'$TMP/iperf.json'" \
        >"$TMP/rattan.log" 2>&1
    kill $srv 2>/dev/null   # still waiting if the client never connected
    wait $srv 2>/dev/null
    # iperf3 -J puts its errors into the JSON, not on stderr
    rate=$(python -c '
import json, sys
d = json.loads(sys.stdin.read())
print("iperf3 error: " + d["error"] if "error" in d else "%.2f" % (d["end"]["sum_received"]["bits_per_second"] / 1e6))' <"$TMP/iperf.json" 2>/dev/null)
    rtt=$(grep -oE '= [0-9.]+/[0-9.]+' "$TMP/ping.txt" 2>/dev/null | cut -d/ -f2)
    msg="downlink ${rate:-no iperf3 output} Mbit/s (limit 10), ping RTT ${rtt:-none} ms (expect 20)"
    if [[ $rate =~ ^[0-9.]+$ && -n $rtt ]] &&
        python -c "import sys; sys.exit(not (8.5 < $rate < 10.5 and 19 < $rtt < 25))"; then
        pass rattan-link "$msg"
    else
        fail rattan-link "$msg; rattan: $(grep -v first-payload "$TMP/rattan.log" | tail -2 | tr '\n' ' ')(ufw? see root_setup.sh)"
    fi

    # Chromium inside the rattan namespace, loading a page from the host.
    hport=5298
    echo ok >"$TMP/index.html"
    python -m http.server $hport --bind 0.0.0.0 --directory "$TMP" >/dev/null 2>&1 &
    hsrv=$!
    sleep 0.5
    timeout 60 "$RATTAN" link --downlink-delay 10ms --uplink-delay 10ms -- \
        python -c "
import os
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page()
    pg.goto('http://%s:$hport/index.html' % os.environ['RATTAN_BASE'], timeout=10000)
    open('$TMP/chrome-ns.out', 'w').write(pg.inner_text('body').strip())
    b.close()" >"$TMP/chrome-ns.log" 2>&1
    kill $hsrv 2>/dev/null
    wait $hsrv 2>/dev/null
    if [[ $(cat "$TMP/chrome-ns.out" 2>/dev/null) == ok ]]; then
        pass chromium-in-rattan "page from host loaded inside ns-left"
    else
        fail chromium-in-rattan "$(grep -v first-payload "$TMP/chrome-ns.log" | tail -2 | tr '\n' ' ')"
    fi
fi

# ---- WebRTC binaries (run_all.py sets CHECK_WEBRTC=1 for transport = "webrtc")
WEBRTC_BIN=${WEBRTC_BIN:-webrtc/bin}
if [[ ${CHECK_WEBRTC:-1} == 1 ]]; then
    for b in tigas_sender peerconnection_server; do
        if [[ ! -x $WEBRTC_BIN/$b ]]; then
            fail "webrtc-$b" "$WEBRTC_BIN/$b missing (build it from sparkrtc)"
        elif missing=$(ldd "$WEBRTC_BIN/$b" 2>&1 | grep 'not found' | awk '{print $1}' | tr '\n' ' ') && [[ -n $missing ]]; then
            fail "webrtc-$b" "missing libraries: $missing"
        fi
    done
    # the output goes to a variable first: with pipefail, grep -q closing the
    # pipe early would fail the check
    if [[ -x $WEBRTC_BIN/tigas_sender ]]; then
        out=$("$WEBRTC_BIN/tigas_sender" --helpon=main 2>&1)
        if [[ $out == *render_socket* ]]; then
            pass webrtc-sender "$WEBRTC_BIN/tigas_sender runs; $(head -1 "$WEBRTC_BIN/BUILD_INFO" 2>/dev/null || echo 'no BUILD_INFO')"
        else
            fail webrtc-sender "$WEBRTC_BIN/tigas_sender --helpon=main: $(tail -1 <<<"$out")"
        fi
    fi
    if [[ -x $WEBRTC_BIN/peerconnection_server ]]; then
        out=$("$WEBRTC_BIN/peerconnection_server" --help 2>&1)
        if [[ $out == *port* ]]; then
            pass webrtc-signaling "$WEBRTC_BIN/peerconnection_server runs"
        else
            fail webrtc-signaling "$WEBRTC_BIN/peerconnection_server --help: $(tail -1 <<<"$out")"
        fi
    fi
    if [[ $(ss -tln) == *":8888 "* ]]; then
        fail webrtc-port "TCP 8888 (signaling) already in use"
    else
        pass webrtc-port "TCP 8888 free"
    fi
fi

# ---- scene model and port
ply=$(ls static/models/"$SCENE"/*.ply 2>/dev/null | head -1)
if [[ -n $ply ]]; then
    pass model "$ply ($(du -hL "$ply" | cut -f1))"
else
    fail model "no static/models/$SCENE/*.ply"
fi
if ss -uln | grep -q ':4433 '; then
    warn port "UDP 4433 already in use"
else
    pass port "UDP 4433 free"
fi

echo
if [[ $nfail -eq 0 ]]; then
    echo "all checks passed"
else
    echo "$nfail check(s) failed"
fi
exit $((nfail > 0))
