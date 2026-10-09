#!/bin/bash
# Queued after the C0->C4 chain: extra IC seeds for the 2D KS normalized-IC transfer.
while pgrep -f run_c0_then_c4.sh > /dev/null; do sleep 30; done
A=artifacts
for s in 1 2; do for mode in normalized raw; do
  tag=ks2d_x1.5_${mode}_seed$s
  [ -f $A/experiments/icnorm/$tag.json ] && continue
  echo "[$(date +%T)] start $tag"
  OMP_NUM_THREADS=1 python scripts/exp_ic_normalization_check.py --tag $tag --system ks2d \
    --ae-ckpt $A/stage1_ae_2dks_full_section230_ks2d_localfield_localmlp2d_window3.pt \
    --prop-ckpt $A/stage2_prop_2dks_full_section230_ks2d_localfield_localmlp2d_window3_warmstart_k12_300ep.pt \
    --dataset $A/datasets/ks2d_trajectories_default.h5 --x-scale 1.5 --y-scale 1.5 --ic-mode $mode --seed $s \
    > $A/logs/icnorm_$tag.log 2>&1 && echo "[$(date +%T)] done $tag" || echo "[$(date +%T)] FAILED $tag"
done; done
echo "[$(date +%T)] ALL DONE"
