#!/bin/bash
# R4 (docs/split_paper_tasks.md, user 2026-10-12): does a 2D KS model trained on a SMALL periodic domain transfer to
# larger ones, as the 1D model does? c19-style per-patch autoencoder (reach 0, c = 6) and the Paper I recipe
# (k_max 32, k_mid 16, 9000 propagator steps, end noise 0.02), with the propagator's reach cut so that a training
# window never wraps onto itself on the small periodic box (window = 2 x reach + interior <= box sites):
#   C  44^2 data (8 sites):  reach 3 (1 layer x window 3), crop 8, halo 3, 2x2 interior
#   A  66^2 data (12 sites): reach 4 (2 layers x window 2), crop 12, halo 4, 4x4 interior
#   B  66^2 data (12 sites): reach 5 (1 layer x window 5), crop 12, halo 5, 2x2 interior
# Equal work per step (2026-10-12): small crops see far fewer patches (Stage A) and free interior sites (Stage B)
# per step than the canonical 40x40-site crops, so the batches are scaled to the canonical sites per step
# (Stage A 32 x 1600, Stage B 64 x 1600 sites): 44^2 crops (64 sites) 800 / 1600; 66^2 crops (144 sites) 356 / 711.
# Tags end in _eq; the first r4c run (batch 32 / 64, decoder floor 0.66 vs 0.25) is kept for the record.
# Trainings run one at a time on MPS (C, A, B); each model's D_KY evaluations (44^2, 66^2, 88^2; IC seeds 0-1)
# run on CPU while the next model trains. Truth: 44^2 and 66^2 tables; 88^2 by the extensive prediction (~333).
# Log: artifacts/logs/r4.log.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; E=$A/experiments; D=$A/datasets
log() { echo "$(date '+%F %T') [R4] $*" >> $L/r4.log; }
BASE="--n-offsets 16 --local-channels 6 --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 \
--dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000"
RUNS=(
  "r4c_ks2d_s44_r3_eq|$D/ks2d_trajectories_default.h5|--crop-sites 8 --crop-sites-y 8 --window 3 --prop-layers 1 --ae-batch 800 --prop-batch 1600"
  "r4a_ks2d_s66_r4_eq|$D/ks2d_residual_L1_transfer66.h5|--crop-sites 12 --crop-sites-y 12 --window 2 --prop-layers 2 --ae-batch 356 --prop-batch 711"
  "r4b_ks2d_s66_r5_eq|$D/ks2d_residual_L1_transfer66.h5|--crop-sites 12 --crop-sites-y 12 --window 5 --prop-layers 1 --ae-batch 356 --prop-batch 711"
)
evals() { tag=$1
  for s in 0 1; do for sc in 1.0 1.5 2.0; do
    out=$E/crop/${tag}_x${sc}_seed$s.json; [ -f $out ] && continue
    nd=""; [ $sc = 2.0 ] && nd="--n-directions 512"
    $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $nd \
      >> $L/crop_eval_$tag.log 2>&1
    rc=$?; log "eval $tag x$sc seed$s rc=$rc $($PY scripts/r4_summary.py dky $out)"
  done; done; log "evals $tag done"; }
log "start"
pids=()
for r in "${RUNS[@]}"; do IFS='|' read -r tag data extra <<< "$r"
  if [ ! -f $A/stage2_prop_kolmogorov_crop_$tag.pt ]; then
    log "train $tag start"
    $PY scripts/train_kolmogorov_crop.py --dataset $data --tag $tag $BASE $extra > $L/crop_train_$tag.log 2>&1
    log "train $tag rc=$? $(grep -E 'step 9000/9000' $L/crop_train_$tag.log | tail -1)"
  fi
  [ -f $A/stage2_prop_kolmogorov_crop_$tag.pt ] && { evals $tag & pids+=($!); }
done
wait "${pids[@]}"
log "R4 DONE"
