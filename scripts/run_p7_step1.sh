#!/bin/bash
# P7 step 1 (2026-10-03): is 2D KS's error native or transfer?
#  (a) the two deferred normalized-IC transfer seeds at 1.5x (66^2), Section 230 model;
#  (b) a direct true-PDE D_KY at the native 44^2 box (150 directions; until now only an
#      area-scaled estimate, 85.6, existed). Sequential; tasks skip if done.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
AE=$A/stage1_ae_2dks_full_section230_ks2d_localfield_localmlp2d_window3.pt
PR=$A/stage2_prop_2dks_full_section230_ks2d_localfield_localmlp2d_window3_warmstart_k12_300ep.pt
log () { echo "[p7s1 $(date '+%m-%d %H:%M:%S')] $*"; }
for s in 1 2; do
  t=ks2d_x1.5_normalized_seed$s
  if [ -f $E/icnorm/$t.json ]; then log "skip $t"; continue; fi
  log "running $t"
  OMP_NUM_THREADS=2 python scripts/exp_ic_normalization_check.py --system ks2d --ae-ckpt $AE --prop-ckpt $PR \
    --dataset $A/datasets/ks2d_trajectories_default.h5 --x-scale 1.5 --y-scale 1.5 --ic-mode normalized \
    --seed $s --tag $t >> $A/logs/icnorm_$t.log 2>&1 && log "done $t" || log "FAILED $t"
done
if ls $E/g1_2dks_groundtruth_transfer_native44_n150*.json > /dev/null 2>&1; then log "skip native truth"; else
  log "running native 44^2 truth (150 directions)"
  OMP_NUM_THREADS=1 python scripts/exp_g1_2dks_groundtruth_transfer.py --scale 1.0 --n-directions 150 \
    --tag native44_n150 >> $A/logs/p7_native44_truth.log 2>&1 && log "done native truth" || log "FAILED native truth"
fi
log "ALL DONE"
