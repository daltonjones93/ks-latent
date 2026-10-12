#!/bin/bash
# R4 queue, reordered (user 2026-10-12): after run C (equal-work batches), the 12-channel noisy-data version of C
# (run D, the Paper II recipe on the 44^2 data) goes first; then the p7b control at 440^2; then A and B (66^2 source).
# Training lane (MPS, one at a time): [C_eq, already running] -> D -> A -> B, each followed by its D_KY evals (CPU).
# Filter lane (CPU, one 440^2 run at a time): C_eq 44/44 T1/440 -> D 44/440 -> p7b 440 -> A 44/440 -> B 44/440.
# Settings and the matched physical observation error as in scripts/run_r4_da.sh. Logs: r4.log, r4_da.log.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; E=$A/experiments; D=$A/datasets
log() { echo "$(date '+%F %T') [R4] $*" >> $L/r4.log; }
dlog() { echo "$(date '+%F %T') [R4-DA] $*" >> $L/r4_da.log; }
BASE="--n-offsets 16 --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
--k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000"
C=r4c_ks2d_s44_r3_eq; DT=r4d_ks2d_s44_r3_c12d10; AT=r4a_ks2d_s66_r4_eq; BT=r4b_ks2d_s66_r5_eq
args() { case $1 in  # bash 3.2 (macOS): no associative arrays
  $DT) echo "--dataset $D/ks2d_trajectories_default.h5 --local-channels 12 --data-noise 0.1 --crop-sites 8 --crop-sites-y 8 --window 3 --prop-layers 1 --ae-batch 800 --prop-batch 1600" ;;
  $AT) echo "--dataset $D/ks2d_residual_L1_transfer66.h5 --local-channels 6 --crop-sites 12 --crop-sites-y 12 --window 2 --prop-layers 2 --ae-batch 356 --prop-batch 711" ;;
  $BT) echo "--dataset $D/ks2d_residual_L1_transfer66.h5 --local-channels 6 --crop-sites 12 --crop-sites-y 12 --window 5 --prop-layers 1 --ae-batch 356 --prop-batch 711" ;;
esac; }
ckpt() { echo $A/stage2_prop_kolmogorov_crop_$1.pt; }
evals() { tag=$1
  for s in 0 1; do for sc in 1.0 1.5 2.0; do out=$E/crop/${tag}_x${sc}_seed$s.json; [ -f $out ] && continue
    nd=""; [ $sc = 2.0 ] && nd="--n-directions 512"
    $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $(ckpt $tag) \
      --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $nd >> $L/crop_eval_$tag.log 2>&1
    rc=$?; log "eval $tag x$sc seed$s rc=$rc $($PY scripts/r4_summary.py dky $out)"
  done; done; log "evals $tag done"; }
train() { tag=$1; [ -f $(ckpt $tag) ] && return
  log "train $tag start"
  $PY scripts/train_kolmogorov_crop.py --tag $tag $BASE $(args $tag) > $L/crop_train_$tag.log 2>&1
  log "train $tag rc=$? $(grep -E 'step 9000/9000' $L/crop_train_$tag.log | tail -1)"; }
da() { tag=$1; Ls=$2; T=$3; out=R4_${tag}_L${Ls}_T$T; [ -f $E/da2d/h9_$out.json ] && return
  until [ -f $(ckpt $tag) ] && ! pgrep -f "train_kolmogorov_crop.py --tag $tag " > /dev/null; do sleep 60; done
  o=$($PY -c "import torch; ck=torch.load('$A/stage1_ae_kolmogorov_crop_$tag.pt', map_location='cpu', weights_only=False); print(round(0.1 * 6.172126667325263 / ck['normalization_std'], 5))")
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $tag --L $Ls --n-tilings $T --rtps 0.7 --obs-stride 8 \
    --obs-noise-std $o --out $out > $L/h9_$out.log 2>&1
  rc=$?; dlog "$out rc=$rc obs=$o $($PY scripts/r4_summary.py da $E/da2d/h9_$out.json)"; }
train_lane() {
  while pgrep -f "train_kolmogorov_crop.py --dataset .* --tag $C " > /dev/null; do sleep 60; done
  log "train $C done $(grep -E 'step 9000/9000' $L/crop_train_$C.log | tail -1)"
  evals $C &
  train $DT; evals $DT &
  train $AT; evals $AT &
  train $BT; evals $BT &
  wait; log "TRAIN LANE DONE"; }
da_lane() {
  da $C 44 4; da $C 44 1; da $C 440 4
  da $DT 44 4; da $DT 440 4
  da p7b_ks2d 440 4
  da $AT 44 4; da $AT 440 4; da $BT 44 4; da $BT 440 4
  dlog "R4 DA DONE"; }
log "queue2 start (order: C_eq, D, A, B)"
train_lane & da_lane & wait
log "R4 DONE"
