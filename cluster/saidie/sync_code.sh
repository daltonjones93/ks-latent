#!/bin/bash
# (Run on the Mac.) Copy the code needed for CM1 training to Saidie: code only, no data or artifacts.
# Usage: bash cluster/saidie/sync_code.sh [saidie]
set -euo pipefail
REMOTE=${1:-saidie}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' \
  --include 'ks_latent/***' --include 'scripts/' --include 'scripts/cm1/***' \
  --include 'cluster/' --include 'cluster/saidie/***' --include 'tests/' --include 'tests/unit/' \
  --include 'tests/unit/test_cm1.py' --include 'pyproject.toml' --exclude '*' \
  "$REPO/" "$REMOTE:~/latent_DA/"
echo "code synced to $REMOTE:~/latent_DA"
