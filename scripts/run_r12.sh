#!/bin/bash
# R12 (user 2026-10-12): can the small-box paradigm work for Kolmogorov flow?  The alpha = 0.05 small-box model
# (r4k_c6_a005_s1x_r3) gives D_KY 22.3 at 1x (-0.9%) but 25.4 at 2x (truth 46.6). Hypothesis: weak drag lets energy
# reach the largest scale the domain allows, so a 2x domain has active scales (wavelength 4pi) the 2pi box never had.
#  R12.1 diagnosis: free-rollout statistics of the model vs the true PDE at 1x and 2x (energy beyond the native box,
#        large-scale share, field std, and whether the 2x rollout locks into a 2pi-periodic state).
#  R12.2 the same recipe at alpha = 0.1 (stronger drag, shorter largest active scale): train, rollout stats at
#        1x/2x/3x, then D_KY at 2x (IC seed 1, multi-run truth T120), 1x, 3x.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; CR=$A/experiments/crop; RS=$A/experiments/rollout_stats
log() { echo "$(date '+%F %T') [R12] $*" >> $L/r4.log; }
rs() { tag=$1; sc=$2; $PY scripts/exp_rollout_stats_kolmogorov.py --tag $tag --scales $sc --n-steps 1000 --with-truth > $L/r12_rollout_$tag.log 2>&1
  log "rollout stats $tag rc=$? $(grep '^\[rollout\]' $L/r12_rollout_$tag.log | tr '\n' ' ')"; }
rs r4k_c6_a005_s1x_r3 1,2 &
T=r12_c6_a01_s1x_r3
if [ ! -f $A/stage2_prop_kolmogorov_crop_$T.pt ]; then log "train $T start"
  $PY scripts/train_kolmogorov_crop.py --tag $T --dataset $A/datasets/kolmogorov_trajectories_alpha0.1.h5 --n-offsets 16 \
    --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 \
    --noise-end 0.02 --prop-steps 9000 --crop-sites 8 --window 3 --prop-layers 1 --local-channels 6 > $L/crop_train_$T.log 2>&1
  log "train $T rc=$? $(grep -E 'step 9000/9000' $L/crop_train_$T.log | tail -1)"; fi
rs $T 1,2,3 &
ev() { sc=$1; s=$2; tj=$3; out=$CR/${T}_x${sc}_seed$s.json; [ -f $out ] && return
  $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$T.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$T.pt \
    --scale $sc --seed $s --tag ${T}_x${sc}_seed$s $tj >> $L/crop_eval_$T.log 2>&1
  log "eval $T x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; }
ev 2 1 "--truth-json $CR/truth_a0p1_x2_seed1_T120.json" &
ev 1 0 "" &
wait
ev 3 1 "--truth-json $CR/truth_a0p1_x3_seed1_T120.json"
log "R12 DONE"
