#!/bin/bash
# User-directed redirect (2026-10-05): the Phase 1 Kolmogorov retrains (k_a005/k_a01/
# k_a0025/k_b005/k_c0025) all used 2D-patch training, which P9 (sec:p9_results, FAILURE)
# already showed does not suit Kolmogorov's anisotropic/forced dynamics. k_a005 confirmed
# this again tonight: it was LESS accurate than c19_e0d0_c6 (C19 autoencoder, x-strip,
# short rollout) at every size (1x 12.9% vs 8.7%, 2x 13.6% vs 8.3%, 3x 15.1% vs 2.9%), and
# P9b (old autoencoder, 2D-patch, long rollout) landed in the same degraded range
# (10.9/15.0/14.1%) -- two different autoencoders, same 2D-patch penalty. All of
# k_a01/k_b005/k_a0025/k_c0025 were stopped (none had produced a trained checkpoint).
#
# This replaces them with the ONE combination not yet tried: x-strip training (matching
# c19_e0d0_c6 exactly: same dataset shape, no --crop-sites-y/--y-align) + the C19 per-patch
# c=6 autoencoder + P9b's long free rollouts (k_max=32) -- isolating whether long rollouts
# help AT ALL for Kolmogorov, independent of the already-known-bad 2D-patch training cost.
# alpha=0.025 (user-specified): the hardest case, where P8's original recipe collapsed
# worst (sec:p8_model_first, native n_positive 4-9 against truth 14).
#   k_xs0025  = x-strip + long rollout, no regularizer
#   k_xsr0025 = x-strip + long rollout + spectral regularizer (Kolmogorov-default bands)
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; D=$A/datasets/kolmogorov_wide16_alpha0.025.h5
log () { echo "[kxs0025 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
ev () { tag=$1; sc=$2; s=$3
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s --truth-json $E/crop/truth_a0p025_x${sc}_seed0.json \
    >> $A/logs/crop_eval_$tag.log 2>&1"; }

# Exactly c19_e0d0_c6's own flags (x-strip, no crop-sites-y/y-align), alpha swapped, plus the
# long-rollout fix (k_max 32, k_mid 16, noise-end 0.02, prop-steps 9000 -- P9b's settings).
BASE="--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
--local-channels 6 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000"

run $A/stage2_prop_kolmogorov_crop_k_xs0025.pt "train k_xs0025 (x-strip, long rollout, no reg)" sh -c \
  "python scripts/train_kolmogorov_crop.py --dataset $D --tag k_xs0025 $BASE >> $A/logs/crop_train_k_xs0025.log 2>&1"
for sc in 1 2 3; do for s in 0 1 2; do ev k_xs0025 $sc $s; done; done

run $A/stage2_prop_kolmogorov_crop_k_xsr0025.pt "train k_xsr0025 (x-strip, long rollout, + spectral reg)" sh -c \
  "python scripts/train_kolmogorov_crop.py --dataset $D --tag k_xsr0025 $BASE --w-spectral 1.0 \
   >> $A/logs/crop_train_k_xsr0025.log 2>&1"
for sc in 1 2 3; do for s in 0 1 2; do ev k_xsr0025 $sc $s; done; done

log "ALL DONE"
