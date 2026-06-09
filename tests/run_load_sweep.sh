#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 SERVER_IP [PORT]" >&2
  exit 2
fi

server_ip="$1"
port="${2:-7878}"
rollouts="2,4,8,16"
build_workers="${BUILD_WORKERS:-8}"
timeout_seconds="${TIMEOUT_SECONDS:-60}"
run_id="$(date +%Y%m%d-%H%M%S)"

cd "$(dirname "$0")"

ulimit -n 65535 2>/dev/null || true

echo "Checking server ${server_ip}:${port}"
if command -v nc >/dev/null 2>&1; then
  nc -vz "$server_ip" "$port"
fi

for num_tasks in 32 64 128; do
  echo
  echo "Running load test: num_tasks=${num_tasks} num_rollouts=${rollouts}"
  uv run load_test.py \
    --host "$server_ip" \
    --port "$port" \
    --num-tasks "$num_tasks" \
    --num-rollouts "$rollouts" \
    --build-workers "$build_workers" \
    --timeout "$timeout_seconds" \
    --task-prefix "load-${run_id}-tasks-${num_tasks}"
done
