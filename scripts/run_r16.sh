#!/bin/bash
# R16 / Z10a (user 2026-10-13, Tier 1): KC2 recipe + m = 2 propagator-written memory channels (option (i): memory
# reads its own memory + neighbours' encoder channels; ks_latent/models/memory_prop.py), warm-up 8. Stage A REUSED from
# r14_KC2_a<alpha>_seed0 (copied), so R16 vs KC2 differs only in the propagator. alpha 0.05 first, then 0.1.
# Read-out: rollout statistics 1/2/3x (2x |k|<2 share: truth 0.146, KC2 0.441, window model 0.173), then D_KY at 2x IC 1
# and 3x (IC 0 at alpha 0.05, IC 1 at 0.1) regardless. Normal priority, 3 threads, one training at a time.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; CR=$A/experiments/crop; D=$A/datasets
log() { echo "$(date '+%F %T') [R16] $*" >> $L/r4.log; }
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
truth() { a=$1; sc=$2; s=$3; if [ $a = 0.05 ]; then [ $sc = 3 ] && echo $CR/truth_a005_x3_seed0.json || echo $CR/truth_a0p05_x${sc}_seed${s}_T120.json
  else echo $CR/truth_a0p1_x${sc}_seed${s}_T120.json; fi; }
ev() { tag=$1; a=$2; sc=$3; s=$4; out=$CR/${tag}_x${sc}_seed$s.json; [ -f $out ] && return
  $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s --truth-json $(truth $a $sc $s) >> $L/crop_eval_$tag.log 2>&1
  log "eval $tag x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; }
log "R16 start"
for al in 0.05 0.1; do
  tag=r16_KM_a${al}_seed0
  if [ ! -f $A/stage1_ae_kolmogorov_crop_$tag.pt ]; then
    cp $A/stage1_ae_kolmogorov_crop_r14_KC2_a${al}_seed0.pt $A/stage1_ae_kolmogorov_crop_$tag.pt
    cp $A/stage1_ae_kolmogorov_crop_r14_KC2_a${al}_seed0.pt.meta.json $A/stage1_ae_kolmogorov_crop_$tag.pt.meta.json 2>/dev/null
  fi
  if [ ! -f $A/stage2_prop_kolmogorov_crop_$tag.pt ]; then log "train $tag start"
    $PY scripts/train_kolmogorov_crop.py --tag $tag --dataset $D/kolmogorov_box1x_large_a$al.h5 --n-offsets 16 \
      --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 48 --k-mid 24 \
      --noise-end 0.02 --prop-steps 12000 --crop-sites 8 --window 3 --prop-layers 1 --local-channels 6 \
      --memory-channels 2 --memory-warmup 8 > $L/crop_train_$tag.log 2>&1
    log "train $tag rc=$? $(grep -E 'step 12000/12000' $L/crop_train_$tag.log | tail -1)"; fi
  ( $PY scripts/exp_rollout_stats_kolmogorov.py --tag $tag --scales 1,2,3 --n-steps 1000 --with-truth > $L/r16_rollout_$tag.log 2>&1
    log "rollout $tag rc=$? $(grep '^\[rollout\]' $L/r16_rollout_$tag.log | sed -E 's/\{.*std_late\": ([0-9.]+).*share_k_lt_2_late\": ([0-9.]+).*\| truth.*std_late\": ([0-9.]+).*share_k_lt_2_late\": ([0-9.]+).*/std \1 (truth \3) k<2 \2 (truth \4)/' | tr '\n' ' ')" ) &
  ev $tag $al 2 1 &
  s3=1; [ $al = 0.05 ] && s3=0
  ev $tag $al 3 $s3 &
done
wait; log "R16 DONE"
