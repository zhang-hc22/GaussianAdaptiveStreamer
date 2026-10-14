#!/usr/bin/env bash
# Phase 0 checks on the experiment machine, about 4 minutes. Run from the
# repo root after exp/setup/check_env.sh passes:
#
#   bash exp/phase0/phase0.sh [out dir]
#
# 1. starts the TIGAS server on 0.0.0.0:4433;
# 2. load: headless Chromium opens /player and draws a frame, first directly
#    (this also loads the model), then inside `rattan link` (20 ms RTT);
# 3. step: inside `rattan link` with a 2/50 Mbit/s step trace, a probe keeps
#    4 requests in flight and records when every chunk arrives; step_trace.py
#    compares the bandwidth changes seen by the client with t0 and with the
#    trace start rattan logs. Repeated STEP_REPS times;
# 4. stops the server and measures the JPEG size of every profile along the
#    trajectory (frame_sizes.py).
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root
ROOT=$(pwd)

CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-tigas}
RATTAN=${RATTAN:-/usr/local/bin/rattan}
STEP_REPS=${STEP_REPS:-3}
OUT=$(realpath -m "${1:-exp/results/phase0/$(date +%Y%m%d-%H%M%S)}")
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
set +u   # the compiler activation scripts read unset variables
conda activate "$CONDA_ENV"
set -u

LINK=(--downlink-delay 10ms --uplink-delay 10ms)
PROBE=(python "$ROOT/exp/phase0/probe.py" --cert "$ROOT/certificates/ssl_cert.pem"
       --trajectory "$ROOT/TestMovements/NTHU/train/user3_train.json")

echo "== server"
python http3_server.py -c certificates/ssl_cert.pem -k certificates/ssl_key.pem \
    --host 0.0.0.0 --port 4433 >"$OUT/server.log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null' EXIT
for _ in $(seq 50); do
    ss -uln | grep ':4433 ' >/dev/null && break
    sleep 0.2
done
ss -uln | grep ':4433 ' >/dev/null || { echo "server did not start:"; tail "$OUT/server.log"; exit 1; }

echo "== load, direct"
"${PROBE[@]}" load --server 127.0.0.1 --out "$OUT/load-direct.json"

echo "== load, in rattan"
timeout 120 "$RATTAN" link "${LINK[@]}" -- "${PROBE[@]}" load --out "$OUT/load-rattan.json" \
    >"$OUT/load-rattan.log" 2>&1
grep -E '^load:|Error|error' "$OUT/load-rattan.log" | grep -v first-payload

echo "== step trace"
python exp/phase0/step_trace.py write "$OUT/step-trace.json"
for i in $(seq "$STEP_REPS"); do
    RUST_LOG=info,rattan_core::core=debug timeout 180 "$RATTAN" link "${LINK[@]}" \
        --downlink-trace "$OUT/step-trace.json" \
        --downlink-queue droptail --downlink-queue-args '{"packet_limit":1000}' \
        -- "${PROBE[@]}" step --until 30 --out "$OUT/step$i-probe.json" \
        >"$OUT/step$i-rattan.log" 2>&1
    echo "-- rep $i"
    python exp/phase0/step_trace.py check --probe "$OUT/step$i-probe.json" \
        --rattan-log "$OUT/step$i-rattan.log" --out "$OUT/step$i-check.json"
done

echo "== frame sizes"
kill $srv
wait $srv 2>/dev/null
python exp/phase0/frame_sizes.py --out "$OUT/sizes.csv"

echo "done: $OUT"
