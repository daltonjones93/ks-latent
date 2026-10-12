#!/bin/bash
# R4K (user 2026-10-12): the R4 small-source test for Kolmogorov flow. Train on the 1x box (2pi x 2pi, 8 x 8 sites;
# kolmogorov_trajectories_default.h5, alpha = 0.05, 24 + 6 trajectories x 40 time units) and evaluate at 1x, 2x, 3x
# (channel widened in x; the height, 8 sites, is periodic at every width, so only x must not wrap). Crops: 8 sites in
# x, full height; propagator reach 3 (1 layer x window 3), halo 3 in x. Batches scaled to the canonical strips' sites
# per step (48 x 8 = 384 vs 64 sites per crop: x6 -> 192 / 384).
#   KC6:  c = 6, clean data (Paper I recipe)        -> D_KY at 1x/2x/3x (IC seeds 0-1; 3x seed 0)
#   KD12: c = 12, data noise 0.1 (Paper II recipe)  -> D_KY, then the tiled filter (x-only tilings) at 1x and 3x,
#         obs error matched to the k_xs005_c12d10 runs in physical units.
# Waits for the R4 training lane (one MPS training at a time). Logs: artifacts/logs/r4k.log.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; E=$A/experiments; D=$A/datasets; C=$E/crop
log() { echo "$(date '+%F %T') [R4K] $*" >> $L/r4k.log; }
BASE="--dataset $D/kolmogorov_trajectories_default.h5 --n-offsets 16 --site-mix-radius 0 --n-site-mix-layers 0 \
--dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 \
--crop-sites 8 --window 3 --prop-layers 1 --ae-batch 192 --prop-batch 384"
truth() { sc=$1; s=$2  # per-IC truth files (Paper I Table III runs)
  case $sc in 1) echo $C/truth_a0p05_x1_seed${s}_T120.json ;;
              2) [ $s = 0 ] && echo $C/truth_a0p05_x2_seed0_T120.json || echo $C/truth_a005_x2_seed$s.json ;;
              3) echo $C/truth_a005_x3_seed0.json ;; esac; }
evals() { tag=$1
  for s in 0 1; do for sc in 1 2 3; do [ $sc = 3 ] && [ $s = 1 ] && continue
    out=$C/${tag}_x${sc}_seed$s.json; [ -f $out ] && continue
    $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s \
      --truth-json $(truth $sc $s) >> $L/crop_eval_$tag.log 2>&1
    rc=$?; log "eval $tag x$sc seed$s rc=$rc $($PY scripts/r4_summary.py dky $out)"
  done; done; log "evals $tag done"; }
train() { tag=$1; shift; [ -f $A/stage2_prop_kolmogorov_crop_$tag.pt ] && return
  log "train $tag start"
  $PY scripts/train_kolmogorov_crop.py --tag $tag $BASE "$@" > $L/crop_train_$tag.log 2>&1
  log "train $tag rc=$? $(grep -E 'step 9000/9000' $L/crop_train_$tag.log | tail -1)"; }
until grep -q "TRAIN LANE DONE" $L/r4.log 2>/dev/null; do sleep 120; done
log "start"
train r4k_c6_a005_s1x_r3 --local-channels 6; evals r4k_c6_a005_s1x_r3 &
train r4k_c12d10_a005_s1x_r3 --local-channels 12 --data-noise 0.1; evals r4k_c12d10_a005_s1x_r3 &
T=r4k_c12d10_a005_s1x_r3
o=$($PY -c "import torch; ck=torch.load('$A/stage1_ae_kolmogorov_crop_$T.pt', map_location='cpu', weights_only=False); ref=torch.load('$A/stage1_ae_kolmogorov_crop_k_xs005_c12d10.pt', map_location='cpu', weights_only=False); print(round(0.1 * ref['normalization_std'] / ck['normalization_std'], 5))")
for sc in 1 3; do out=R4K_${T}_x$sc; [ -f $E/kda/k_$out.json ] && continue
  $PY scripts/exp_k_tiled_da.py --tag $T --scale $sc --n-tilings 4 --tiling-axes x --obs-noise-std $o \
    --floor-per-tiling --out $out > $L/kda_$out.log 2>&1
  log "DA $out rc=$? obs=$o $($PY -c "import json;s=json.load(open('$E/kda/k_$out.json'))['steady'];print(' '.join(k+' '+str(round(s[k],3)) for k in ('blend','tiling0','interp','floorK','free') if k in s))" 2>/dev/null)"
done
wait; log "R4K DONE"
