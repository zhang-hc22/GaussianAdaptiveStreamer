#!/usr/bin/env bash
# Rate loop of scripts/bandwidth_fluctuations.sh, for the tc + lo ablation
# (A3). run_one.py --shaper tc-lo sets up the htb class 1:1 on lo the same
# way the script does, starts this loop at t0 and stops it at the end.
#
# Like the script, each step is one `tc class change` and a `sleep 1`, so a
# step takes 1 s plus the time of sudo and tc, and the trace falls behind.
# Changes, so that only the shared queue is restored:
# - first WARMUP_S steps at WARMUP_KBIT (the warm-up W of the rattan traces);
# - then RATES (make_traces.py --tc-rates, one kbit/s value per line) over
#   and over, instead of 150 lines of traces/lte_cascading.csv;
# - each step prints the time it took effect (epoch s) and the rate, instead
#   of the rate alone.
#
#   bash exp/tc_lo_loop.sh RATES WARMUP_S WARMUP_KBIT
set -u
rates=$1
warmup_s=$2
warmup_kbit=$3

step() {
    sudo -n tc class change dev lo parent 1: classid 1:1 htb rate "$1"kbit
    echo "$EPOCHREALTIME $1"
    sleep 1
}

for ((k = 0; k < warmup_s; k++)); do
    step "$warmup_kbit"
done
while true
do
    while IFS=, read -r tput
    do
        step "$tput"
    done < "$rates"
done
