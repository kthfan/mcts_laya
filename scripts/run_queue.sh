#!/usr/bin/env bash
# Run experiment configs one after another: scripts/run_queue.sh LOG_DIR CONFIG...
# Each run's output goes to LOG_DIR/<config name>.log; exit codes are appended to LOG_DIR/queue.log.
set -u
cd "$(dirname "$0")/.."
log_dir=$1; shift
mkdir -p "$log_dir"
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "$(date -u +%FT%TZ) START $cfg" >> "$log_dir/queue.log"
  .venv/bin/python -m mcts_laya -v run "$cfg" > "$log_dir/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) END $cfg exit=$?" >> "$log_dir/queue.log"
done
