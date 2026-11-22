#!/bin/bash
# 10d driver (runs on the Mac: Cashew and Saidie cannot trigger each other). For each new-storm wave: wait for its verified cache job on
# Cashew, screen it for domain-wide initiation (scripts/cm1/screen_storms.py -> ~/flagged_E.txt on Cashew), pull the clean caches to
# Saidie (Saidie rsyncs from Cashew with agent forwarding), then after the last wave submit cluster/saidie/train_new_storms.sh on Saidie.
#   nohup bash cluster/drive_new_storms.sh > logs/drive_new_storms.log 2>&1 &
set -u
WAVES=(2:789576 3:789594 4:789597 5:789600)
S="ssh -o ConnectTimeout=30 -o BatchMode=yes -o IdentitiesOnly=yes"
log() { echo "$(date '+%F %T') $*"; }
retry() { for i in $(seq 1 20); do "$@" && return 0; sleep 120; done; return 1; }
for wj in "${WAVES[@]}"; do
  w=${wj%%:*}; j=${wj##*:}
  log "wave $w: waiting for cache job $j"
  while true; do
    st=$($S cashew "sacct -n -X -P -j $j -o State" 2>/dev/null | head -1)
    case "$st" in COMPLETED) break;; FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*) log "wave $w: cache job $j $st -- stopping"; exit 1;; esac
    sleep 600
  done
  log "wave $w: screening"
  retry $S cashew "cd ~/latent_DA && source ~/venvs/cm1/bin/activate && PYTHONPATH=. nice python scripts/cm1/screen_storms.py ~/cm1_cache_E$w --out ~/flagged_E.txt | tail -1" || { log "screen failed"; exit 1; }
  retry bash -c "$S cashew 'cat ~/flagged_E.txt' > /tmp/flagged_E_drive.txt" || exit 1
  retry scp -q -o BatchMode=yes -o IdentitiesOnly=yes /tmp/flagged_E_drive.txt saidie:flagged_E.txt || exit 1
  retry $S saidie 'sed "s/^\(ens_E[^_]*\)_\(mem\)/\1__\2/; s/^ens_E_pilot4_mem/ens_E_pilot4__mem/; s/$/.npz/" ~/flagged_E.txt > ~/flagged_E_files.txt' || exit 1
  log "wave $w: pulling to Saidie"
  retry ssh -A -o ControlPath=none -o ConnectTimeout=30 -o BatchMode=yes -o IdentitiesOnly=yes saidie "bash ~/pull_e.sh E$w" || { log "pull failed"; exit 1; }
  log "wave $w: done ($(wc -l < /tmp/flagged_E_drive.txt) storms flagged so far)"
done
log "all waves on Saidie; submitting the larger training"
retry $S saidie 'bash -l -c "cd ~/latent_DA && bash cluster/saidie/train_new_storms.sh"' && log "submitted"
