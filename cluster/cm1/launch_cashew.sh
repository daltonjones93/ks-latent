#!/bin/bash
# (Run on the Mac.) The single command: bundle -> copy to Cashew -> run kit/go.sh there.
# Usage: bash cluster/cm1/launch_cashew.sh USER@CASHEW_HOST
#   settings pass through, e.g.:  N=30 EMAIL=you@colostate.edu bash cluster/cm1/launch_cashew.sh USER@HOST
# Re-running is safe: it re-copies the kit and resubmits only unfinished work.
set -euo pipefail
REMOTE=${1:?usage: launch_cashew.sh USER@CASHEW_HOST}
KIT=$(cd "$(dirname "$0")" && pwd)
B=$KIT/../../cm1data/cm1_bundle.tar.gz
bash "$KIT/make_bundle.sh" "$KIT/../../cm1data/cm1r21.1" "$B"
echo "copying bundle to $REMOTE (327 MB)..."
scp "$B" "$REMOTE:~/cm1_bundle.tar.gz"
VARS=""
for v in N SEED ENS CPUS TIMAX TAPFRQ EMAIL EXTRA_ARGS; do
  [ -n "${!v:-}" ] && VARS+="$v=$(printf %q "${!v}") "
done
ssh "$REMOTE" "tar xzf ~/cm1_bundle.tar.gz -C ~ && cd ~/cm1_bundle && $VARS bash kit/go.sh"
