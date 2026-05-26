#!/usr/bin/env bash
# Launch cross_year_train_3_models.py detached via nohup so the SSH session
# can be closed. Pins one GPU, timestamps the log, saves PID.
#
# Usage:
#   ./run_cross_year_bg.sh                       # both datasets, 50 trials, GPU 1
#   ./run_cross_year_bg.sh --trials 30           # override trials
#   GPU=0 ./run_cross_year_bg.sh                 # override GPU
#   ./run_cross_year_bg.sh --models tft xgb      # forward any flag to the train script

set -euo pipefail

GPU="${GPU:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAIN="$SCRIPT_DIR/cross_year_train_3_models.py"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO_ROOT/training_${TS}.log"
PIDFILE="$REPO_ROOT/training.pid"

DEFAULT_ARGS=(
    --data-dirs pipeline_workspace/clean_datasets_210526
                pipeline_workspace/clean_dataset_backup
    --trials 50
)
ARGS=("$@")
if [ "${#ARGS[@]}" -eq 0 ]; then
    ARGS=("${DEFAULT_ARGS[@]}")
fi

export CUDA_VISIBLE_DEVICES="$GPU"

cd "$REPO_ROOT"
# setsid detaches from the controlling tty AND creates a new session, so the
# process survives SSH logout even when systemd-logind has KillUserProcesses=yes.
# Stronger than plain nohup which only ignores SIGHUP.
setsid nohup python3 "$TRAIN" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"
disown

cat <<EOF
=== started ===
PID         : $PID                 (saved to $PIDFILE)
GPU         : $GPU                 (CUDA_VISIBLE_DEVICES)
Script      : $TRAIN
Args        : ${ARGS[@]}
Log         : $LOG
Tail        : tail -f $LOG
Status      : ps -p $PID -o pid,etime,cmd
Kill        : kill \$(cat $PIDFILE)   |   pkill -f cross_year_train_3_models
EOF
