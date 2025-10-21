#!/usr/bin/env bash
set -euo pipefail

# If the host mounted a requirements.txt into /workspace, install it now.
if [ -f /workspace/requirements.txt ]; then
  echo "Found /workspace/requirements.txt — installing requirements"
  pip install -r /workspace/requirements.txt
else
  echo "No /workspace/requirements.txt found — skipping pip install"
fi

# Exec the container command
exec "$@"
