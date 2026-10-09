#!/bin/bash
# C19 (2026-10-04, user: "queue it next, before P8"): encoder/decoder reach ablation on the
# P2 recipe (alpha=0.05, 16x-wide data, 48-site x-crops, window-3 propagator, halo 6).
# Only the autoencoder's site-mixing reach changes, so the AE edge mask shrinks from 12 to
# enc+dec sites. Seed-0 evals at 1/2/3x (user: one seed; background QoS for heat).
# Then the remainder of the P8 queue (replaces run_alpha_sweep_queue.sh's tail).
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
log () { echo "[c19p8 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
while pgrep -f run_p7_eval_seed0.sh > /dev/null; do sleep 60; done

D05=$A/datasets/kolmogorov_wide16_alpha0.05.h5
T3=$E/crop/truth_a005_x3_seed0.json
for arm in \
  "c19_e6d0:--dec-site-mix-radius 0 --dec-n-site-mix-layers 0" \
  "c19_e2d0:--site-mix-radius 1 --n-site-mix-layers 2 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0" \
  "c19_e0d0:--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0" \
  "c19_e0d0_c6:--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --local-channels 6"
do
  tag=${arm%%:*}; extra=${arm#*:}
  run $A/stage2_prop_kolmogorov_crop_$tag.pt "$tag train" taskpolicy -b \
    python scripts/train_kolmogorov_crop.py --dataset $D05 --tag $tag $extra >> $A/logs/crop_train_$tag.log 2>&1
  for sc in 1 2 3; do
    ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
    run $E/crop/${tag}_x${sc}_seed0.json "$tag eval x$sc" taskpolicy -b \
      python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt \
        --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt --scale $sc --seed 0 --tag ${tag}_x${sc}_seed0 $ta \
        >> $A/logs/crop_eval_c19.log 2>&1
  done
done
log "C19 done; P8 next"

D025=$A/datasets/kolmogorov_wide16_alpha0.025.h5
while pgrep -f "gen_kolmogorov_wide_dataset.py --alpha 0.025" > /dev/null; do sleep 60; done
run $D025 "alpha=0.025 wide data" taskpolicy -b python scripts/gen_kolmogorov_wide_dataset.py --alpha 0.025 \
  --out $D025 >> $A/logs/p0_wide_a0.025.log 2>&1
run $A/stage2_prop_kolmogorov_crop_p8_a0025.pt "P8 (alpha=0.025) train" taskpolicy -b \
  python scripts/train_kolmogorov_crop.py --dataset $D025 --tag p8_a0025 >> $A/logs/crop_train_p8_a0025.log 2>&1
for sc in 1 2 3; do for s in 0 1 2; do
  run $E/crop/p8_a0025_x${sc}_seed$s.json "P8 eval x$sc seed$s" taskpolicy -b \
    python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_p8_a0025.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_p8_a0025.pt --scale $sc --seed $s --tag p8_a0025_x${sc}_seed$s \
      --truth-json $E/crop/truth_a0p025_x${sc}_seed0.json >> $A/logs/crop_eval_p8.log 2>&1
done; done
log "ALL DONE"
