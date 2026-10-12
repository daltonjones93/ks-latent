#!/bin/bash
# R4 queue 3 (user 2026-10-12): after equal-work C, check whether equal-work batches matter. If C_eq's D_KY at 44^2
# and 66^2 (IC seed 0) is within 2% of the first C (83.1, 187.1), train D, A, B and the Kolmogorov models with the
# default batches (32/64); otherwise keep equal-work batches. Filter lane as in run_r4_queue2.sh.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; E=$A/experiments; D=$A/datasets; CR=$E/crop
log() { echo "$(date '+%F %T') [R4] $*" >> $L/r4.log; }
dlog() { echo "$(date '+%F %T') [R4-DA] $*" >> $L/r4_da.log; }
C=r4c_ks2d_s44_r3_eq
ckpt() { echo $A/stage2_prop_kolmogorov_crop_$1.pt; }
ev2d() { tag=$1; sc=$2; s=$3; out=$CR/${tag}_x${sc}_seed$s.json; [ -f $out ] && return
  nd=""; [ $sc = 2.0 ] && nd="--n-directions 512"
  $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $(ckpt $tag) \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $nd >> $L/crop_eval_$tag.log 2>&1
  log "eval $tag x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; }
evals2d() { for s in 0 1; do for sc in 1.0 1.5 2.0; do ev2d $1 $sc $s; done; done; log "evals $1 done"; }
ktruth() { case $1 in 1) echo $CR/truth_a0p05_x1_seed${2}_T120.json ;;
  2) [ $2 = 0 ] && echo $CR/truth_a0p05_x2_seed0_T120.json || echo $CR/truth_a005_x2_seed$2.json ;;
  3) echo $CR/truth_a005_x3_seed0.json ;; esac; }
evalsk() { tag=$1; for s in 0 1; do for sc in 1 2 3; do [ $sc = 3 ] && [ $s = 1 ] && continue
  out=$CR/${tag}_x${sc}_seed$s.json; [ -f $out ] && continue
  $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $(ckpt $tag) --scale $sc \
    --seed $s --tag ${tag}_x${sc}_seed$s --truth-json $(ktruth $sc $s) >> $L/crop_eval_$tag.log 2>&1
  log "eval $tag x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; done; done; log "evals $tag done"; }
train() { tag=$1; shift; [ -f $(ckpt $tag) ] && return; log "train $tag start ($*)"
  $PY scripts/train_kolmogorov_crop.py --tag $tag --n-offsets 16 --site-mix-radius 0 --n-site-mix-layers 0 \
    --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 "$@" \
    > $L/crop_train_$tag.log 2>&1
  log "train $tag rc=$? $(grep -E 'step 9000/9000' $L/crop_train_$tag.log | tail -1)"; }
da() { tag=$1; Ls=$2; T=$3; out=R4_${tag}_L${Ls}_T$T; [ -f $E/da2d/h9_$out.json ] && return
  until [ -f $(ckpt $tag) ] && ! pgrep -f "train_kolmogorov_crop.py --tag $tag " > /dev/null; do sleep 60; done
  o=$($PY -c "import torch; ck=torch.load('$A/stage1_ae_kolmogorov_crop_$tag.pt', map_location='cpu', weights_only=False); print(round(0.1 * 6.172126667325263 / ck['normalization_std'], 5))")
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $tag --L $Ls --n-tilings $T --rtps 0.7 --obs-stride 8 \
    --obs-noise-std $o --out $out > $L/h9_$out.log 2>&1
  dlog "$out rc=$? obs=$o $($PY scripts/r4_summary.py da $E/da2d/h9_$out.json)"; }
log "queue3 start"
until [ -f $(ckpt $C) ] && ! pgrep -f "train_kolmogorov_crop.py --dataset .* --tag $C " > /dev/null; do sleep 60; done
log "train $C done $(grep -E 'step 9000/9000' $L/crop_train_$C.log | tail -1)"
ev2d $C 1.0 0; ev2d $C 1.5 0
SAME=$($PY -c "
import json; a=json.load(open('$CR/${C}_x1.0_seed0.json'))['model_D_KY']; b=json.load(open('$CR/${C}_x1.5_seed0.json'))['model_D_KY']
print(int(abs(a/83.1-1)<0.02 and abs(b/187.1-1)<0.02))")
if [ "$SAME" = 1 ]; then SB=""; SA=""; SK=""; log "C_eq matches the first C within 2%: default batches from here"
else SB="--ae-batch 800 --prop-batch 1600"; SA="--ae-batch 356 --prop-batch 711"; SK="--ae-batch 192 --prop-batch 384"
  log "C_eq differs from the first C by more than 2%: keeping equal-work batches"; fi
DT=r4d_ks2d_s44_r3_c12d10; AT=r4a_ks2d_s66_r4; BT=r4b_ks2d_s66_r5; K6=r4k_c6_a005_s1x_r3; K12=r4k_c12d10_a005_s1x_r3
[ -n "$SB" ] && { AT=${AT}_eq; BT=${BT}_eq; }
K44="--dataset $D/ks2d_trajectories_default.h5 --crop-sites 8 --crop-sites-y 8 --window 3 --prop-layers 1"
K66="--dataset $D/ks2d_residual_L1_transfer66.h5 --crop-sites 12 --crop-sites-y 12"
KK="--dataset $D/kolmogorov_trajectories_default.h5 --crop-sites 8 --window 3 --prop-layers 1"
train_lane() {
  { evals2d $C; } &
  train $DT $K44 --local-channels 12 --data-noise 0.1 $SB; { evals2d $DT; } &
  train $AT $K66 --local-channels 6 --window 2 --prop-layers 2 $SA; { evals2d $AT; } &
  train $BT $K66 --local-channels 6 --window 5 --prop-layers 1 $SA; { evals2d $BT; } &
  train $K6 $KK --local-channels 6 $SK; { evalsk $K6; } &
  train $K12 $KK --local-channels 12 --data-noise 0.1 $SK; { evalsk $K12; } &
  wait; log "TRAIN LANE DONE"; }
da_lane() {
  da $C 44 4; da $C 44 1; da $DT 44 4; da $C 440 4; da $DT 440 4; da p7b_ks2d 440 4
  da $AT 44 4; da $BT 44 4; da $AT 440 4; da $BT 440 4
  until [ -f $(ckpt $K12) ] && ! pgrep -f "train_kolmogorov_crop.py --tag $K12 " > /dev/null; do sleep 60; done
  o=$($PY -c "import torch; ck=torch.load('$A/stage1_ae_kolmogorov_crop_$K12.pt', map_location='cpu', weights_only=False); ref=torch.load('$A/stage1_ae_kolmogorov_crop_k_xs005_c12d10.pt', map_location='cpu', weights_only=False); print(round(0.1 * ref['normalization_std'] / ck['normalization_std'], 5))")
  for sc in 1 3; do out=R4K_${K12}_x$sc; [ -f $E/kda/k_$out.json ] && continue
    $PY scripts/exp_k_tiled_da.py --tag $K12 --scale $sc --n-tilings 4 --tiling-axes x --obs-noise-std $o \
      --floor-per-tiling --out $out > $L/kda_$out.log 2>&1
    dlog "$out rc=$? obs=$o $($PY scripts/r4_summary.py da $E/kda/k_$out.json)"; done
  dlog "R4 DA DONE"; }
train_lane & da_lane & wait
log "R4 DONE"
