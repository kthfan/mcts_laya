#!/usr/bin/env bash
# Run experiment configs one after another: scripts/run_queue.sh LOG_DIR CONFIG...
# Each run's output goes to LOG_DIR/<config name>.log; START/END lines go to LOG_DIR/queue.log.
# Resumable: a config already logged as "END <config> exit=0" is skipped, so the same command can
# simply be re-issued after an interruption (an interrupted run restarts from the beginning).
set -u
cd "$(dirname "$0")/.."
log_dir=$1; shift
mkdir -p "$log_dir"
for cfg in "$@"; do
  if grep -qF "END $cfg exit=0" "$log_dir/queue.log" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) SKIP $cfg (already finished)" >> "$log_dir/queue.log"
    continue
  fi
  name=$(basename "$cfg" .yaml)
  echo "$(date -u +%FT%TZ) START $cfg" >> "$log_dir/queue.log"
  .venv/bin/python -m mcts_laya -v run "$cfg" > "$log_dir/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) END $cfg exit=$?" >> "$log_dir/queue.log"
done
