#!/usr/bin/env bash
# Phase 2 acceptance: the client inside `rattan link` with the bus trace at
# x8 and x1 (20 ms RTT), once per ABR, then check_run.py on every run and
# check_trace.py for how well the runs follow the trace. About 15 minutes.
# Run from the repo root:
#
#   bash exp/phase2/accept.sh [out dir]
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-tigas}
TRACE=${TRACE:-bus}
TRAJ=${TRAJ:-NTHU/train/user3_train.json}
OUT=$(realpath -m "${1:-exp/results/phase2/$(date +%Y%m%d-%H%M%S)}")
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
set +u   # the compiler activation scripts read unset variables
conda activate "$CONDA_ENV"
set -u

echo "== traces"
for scale in 8 1; do
    python exp/make_traces.py "$TRACE" --scale $scale --out "$OUT/traces/${TRACE}_x$scale.json"
done

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
for scale in 8 1; do
    for abr in simple l2a lol+; do
        echo -n "x$scale $abr: "
        python exp/run_one.py --trace "$OUT/traces/${TRACE}_x$scale.json" --rtt 20 \
            --out "$OUT/x$scale/$abr" -- --trajectory "$TRAJ" --abr "$abr" --seed 1
    done
done
kill $srv
wait $srv 2>/dev/null

echo "== checks"
python exp/check_run.py "$OUT"/x*/* 2>&1 | grep -vE 'UserWarning|warnings.warn|TORCH_CUDA_ARCH_LIST|INFO:gs'
python exp/phase2/check_trace.py "$OUT"/x8/* "$OUT"/x1/* --fig-dir "$OUT"
echo "done: $OUT"
