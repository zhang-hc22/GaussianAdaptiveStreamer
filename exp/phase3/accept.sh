#!/usr/bin/env bash
# Phase 3 acceptance: exp/run_all.py from the start to the figures with a
# short matrix (exp/phase3/config.toml: the first 300 viewpoints, 2
# repetitions, 24 runs), interrupted once on the way:
#
# 1. start run_all.py; a few seconds after $STOP_AFTER runs have finished
#    (inside the next run), stop it with SIGTERM;
# 2. check that no rattan, client or Chromium process is left and that no
#    network namespace was added;
# 3. start the same command again: it must skip the finished runs (their
#    run.json unchanged) and finish everything;
# 4. check_run.py on every run; print the summary table.
#
# About 25 minutes. Run from the repo root:
#
#   bash exp/phase3/accept.sh [out dir]
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-tigas}
CONFIG=${CONFIG:-exp/phase3/config.toml}
STOP_AFTER=${STOP_AFTER:-4}
OUT=$(realpath -m "${1:-exp/results/phase3/$(date +%Y%m%d-%H%M%S)}")
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
set +u   # the compiler activation scripts read unset variables
conda activate "$CONDA_ENV"
set -u

leftovers() {   # experiment processes of this user
    pgrep -u "$USER" -x rattan
    pgrep -u "$USER" -f 'ms-playwright|exp/client[.]py'
}
finished() {    # run.json of the runs that finished (run_one.py added the rattan fields)
    find "$OUT/runs" -name run.json -exec grep -l '"busy_drops"' {} + 2>/dev/null | sort
}

echo "== 1. run_all.py, stopped after $STOP_AFTER runs"
ns0=$(ip netns list | wc -l)
python exp/run_all.py --config "$CONFIG" --out "$OUT" >"$OUT/first.out" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
    n=$(grep -c ' ok (' "$OUT/run_all.log" 2>/dev/null)
    [[ ${n:-0} -ge $STOP_AFTER ]] && break
    sleep 2
done
sleep 12   # into the playback of the next run
kill -TERM $pid
wait $pid
echo "run_all.py exit $? (130 expected)"
grep -E ' ok \(|FAILED|interrupted|killed|left behind' "$OUT/first.out"

echo "== 2. after the interruption"
sleep 2
echo "leftover processes: $(leftovers | wc -l) (0 expected)"
echo "network namespaces: $ns0 before, $(ip netns list | wc -l) after"
finished | xargs stat -c '%Y %n' >"$OUT/finished_before.txt"
echo "finished runs: $(wc -l <"$OUT/finished_before.txt")"

echo "== 3. same command again"
python exp/run_all.py --config "$CONFIG" --out "$OUT" >"$OUT/second.out" 2>&1
echo "run_all.py exit $? (0 expected)"
grep -vE ' ok \(' "$OUT/second.out"
changed=0
while read -r t f; do
    [[ $(stat -c %Y "$f") == "$t" ]] || { echo "run again: $f"; changed=$((changed + 1)); }
done <"$OUT/finished_before.txt"
echo "finished runs run again: $changed (0 expected)"
echo "leftover processes: $(leftovers | wc -l), network namespaces: $(ip netns list | wc -l)"

echo "== 4. checks"
mapfile -t runs < <(finished | xargs -n1 dirname)
python exp/check_run.py "${runs[@]}" 2>&1 | grep -E '^(PASS|FAIL)|^      ' | sed "s|$OUT/runs/||"
cat "$OUT/summarize.log"
echo "figures: $(find "$OUT/figs" -name '*.pdf' | wc -l)"
ls "$OUT/figs" "$OUT/summary"
echo "done: $OUT"
