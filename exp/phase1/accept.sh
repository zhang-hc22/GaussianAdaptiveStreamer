#!/usr/bin/env bash
# Phase 1 acceptance: plays the whole trajectory once per ABR straight to the
# local server (no rattan) and checks every run with check_run.py. Run from
# the repo root:
#
#   bash exp/phase1/accept.sh [out dir]
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-tigas}
TRAJ=${TRAJ:-NTHU/train/user3_train.json}
OUT=$(realpath -m "${1:-exp/results/phase1/$(date +%Y%m%d-%H%M%S)}")
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
set +u   # the compiler activation scripts read unset variables
conda activate "$CONDA_ENV"
set -u

echo "== server"
python http3_server.py -c certificates/ssl_cert.pem -k certificates/ssl_key.pem \
    --host 0.0.0.0 --port 4433 >"$OUT/server.log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null' EXIT
for _ in $(seq 50); do
    ss -uln | grep ':4433 ' >/dev/null && break
    sleep 0.2
done

echo "== runs"
for abr in simple l2a lol+; do
    python exp/client.py --server 127.0.0.1 --trajectory "$TRAJ" --abr "$abr" \
        --warmup 5 --out "$OUT/$abr" 2>&1 | tee "$OUT/$abr.log" | grep -E '^client:'
done
kill $srv
wait $srv 2>/dev/null

echo "== checks"
python exp/check_run.py "$OUT"/simple "$OUT"/l2a "$OUT"/lol+ 2>&1 | grep -vE 'UserWarning|warnings.warn|INFO:gs'
echo "done: $OUT"
