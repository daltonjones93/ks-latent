#!/bin/bash
# User-directed redirect (2026-10-05): the spectral-band-share regularizer
# (ks_latent.training.crop.spectral_band_shares, sec:propagator_lessons item 3) needs a
# system with real headroom to show improvement. p7b_ks2d (2D KS, canonical recipe, no
# regularizer) already landed within 1% of the extensive prediction at all 3 sizes, so
# p7c_ks2d (same + regularizer) was cancelled as uninformative. k_a0025 (Kolmogorov
# alpha=0.025, canonical recipe, no regularizer) is the better target instead: P8's old
# recipe had a large, well-characterized native-size collapse there (n_positive 14 -> 4-9).
# k_c0025 = same canonical recipe + the regularizer (Kolmogorov-default bands 0,2,3,5).
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; D=$A/datasets/kolmogorov_wide6x6_alpha0.025.h5
log () { echo "[kc0025 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }

while pgrep -f "train_kolmogorov_crop.py" > /dev/null; do sleep 60; done
log "train lane clear; starting k_c0025"

CANON="--crop-sites 40 --crop-sites-y 40 --n-offsets 16 --local-channels 6 --site-mix-radius 0 --n-site-mix-layers 0 \
--dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 --y-align 32"
run $A/stage2_prop_kolmogorov_crop_k_c0025.pt "train k_c0025" sh -c "python scripts/train_kolmogorov_crop.py \
  --dataset $D --tag k_c0025 $CANON --w-spectral 1.0 >> $A/logs/crop_train_k_c0025.log 2>&1"

for sc in 1 2 3; do for s in 0 1; do
  ta=""; [ $sc != 1 ] && ta="--truth-json $E/crop/truth_a0p025_x${sc}_seed0.json"
  run $E/crop/k_c0025_x${sc}_seed$s.json "k_c0025 eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_k_c0025.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_k_c0025.pt \
    --scale $sc --seed $s --tag k_c0025_x${sc}_seed$s $ta >> $A/logs/crop_eval_k_c0025.log 2>&1"
done; done
log "ALL DONE"
