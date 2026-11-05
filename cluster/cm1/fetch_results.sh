#!/bin/bash
# (Run on the Mac.) Copy an ensemble's results back from Cashew into cm1data/.
# Copies only outputs, inputs and logs; skips the symlinked lookup tables and cm1.exe.
# Usage: bash cluster/cm1/fetch_results.sh USER@CASHEW_HOST ~/cm1_bundle/ens_A [local_dest]
set -euo pipefail
REMOTE=$1; RDIR=$2
KIT=$(cd "$(dirname "$0")" && pwd)
DEST=${3:-$KIT/../../cm1data/$(basename "$RDIR")}
mkdir -p "$DEST"
rsync -avz --partial --progress \
  --include='*/' --include='manifest.*' --include='cm1out*' --include='namelist.input' \
  --include='input_sounding' --include='cm1.log' --include='cm1.done' --include='cm1.failed' \
  --exclude='*' "$REMOTE:$RDIR/" "$DEST/"
echo "copied to $DEST"; du -sh "$DEST"
