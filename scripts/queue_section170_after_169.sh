#!/bin/zsh
# Polls for Section 169's own train_stage1_patched.py/train_stage2_patched.py
# processes (tag section169_realpropagator) to exit, then launches Section
# 170 (energy-floor fix) -- avoids MPS contention. Order reversed from the
# original queue_section169_after_170.sh: a race during 170's mid-fix
# restart let 169 launch prematurely while 170 was briefly down for a NaN
# fix, giving 169 a genuine head start -- so 170 now queues behind 169
# instead.
set -e
cd /Users/daltonjones/Documents/latent_DA

echo "=== waiting for Section 169's MPS training to finish ==="
while pgrep -f "train_stage.*section169_realpropagator" > /dev/null; do
  sleep 15
done
echo "=== Section 169 training finished -- launching Section 170 (energy floor) ==="
zsh scripts/section170_energyfloor.sh
