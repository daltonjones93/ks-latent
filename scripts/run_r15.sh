#!/bin/bash
# R15 (user 2026-10-13, NOT on the critical path): two attempts to make the small-box (1x) Kolmogorov model transfer
# at alpha = 0.05, where the C2/C3 recipes fail (R14: large-scale energy builds up at 2x/3x, vorticity variance and
# forcing-band energy fall; Block Z9). Both use the C2 recipe (4x data, k_max 48, 12000 steps, forced-halo windows
# of the 8-site box) and change one thing:
#   KV  (option 1) --input-fields wuv: the encoder also sees the velocity (u, v), computed exactly from the
#       whole-domain vorticity, so the advecting velocity is local state carried by the latent.
#   KS  (option 2) --w-stats 1: free rollouts of a 2x domain (mosaics of 1x states) penalized for drifting from the
#       1x data's vorticity variance and forcing-band share (no 2x data; the choice of statistics used truth-side
#       2x measurements -- disclose). Stage A is REUSED from r14_KC2_a0.05_seed0 (copied), so KS vs KC2 differs
#       only in the propagator loss.
#   KVS both (queued last; drop it if the machine is needed).
# Read-out, same as R14: rollout statistics 1x/2x/3x with truth, D_KY at 2x IC 1 (T120 truth) and 3x IC 0.
# Low priority: one job at a time, efficiency cores (taskpolicy -b), 2 threads. Log: artifacts/logs/r4.log [R15].
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; CR=$A/experiments/crop; D=$A/datasets
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
log() { echo "$(date '+%F %T') [R15] $*" >> $L/r4.log; }
run() { taskpolicy -b "$PY" "$@"; }
DS=$D/kolmogorov_box1x_large_a0.05.h5
truth() { [ $1 = 3 ] && echo $CR/truth_a005_x3_seed0.json || echo $CR/truth_a0p05_x${1}_seed${2}_T120.json; }
ev() { tag=$1; sc=$2; s=$3; out=$CR/${tag}_x${sc}_seed$s.json; [ -f $out ] && return
  run scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s --truth-json $(truth $sc $s) >> $L/crop_eval_$tag.log 2>&1
  log "eval $tag x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; }
log "R15 start"
for v in KV KS KVS; do
  tag=r15_${v}_a0.05_seed0; extra=""
  case $v in KV) extra="--input-fields wuv";; KS) extra="--w-stats 1.0";; KVS) extra="--input-fields wuv --w-stats 1.0";; esac
  if [ $v = KS ] && [ ! -f $A/stage1_ae_kolmogorov_crop_$tag.pt ]; then
    cp $A/stage1_ae_kolmogorov_crop_r14_KC2_a0.05_seed0.pt $A/stage1_ae_kolmogorov_crop_$tag.pt
    cp $A/stage1_ae_kolmogorov_crop_r14_KC2_a0.05_seed0.pt.meta.json $A/stage1_ae_kolmogorov_crop_$tag.pt.meta.json 2>/dev/null
    log "$tag: Stage A reused from r14_KC2_a0.05_seed0"; fi
  if [ ! -f $A/stage2_prop_kolmogorov_crop_$tag.pt ]; then log "train $tag start ($extra)"
    run scripts/train_kolmogorov_crop.py --tag $tag --dataset $DS --n-offsets 16 \
      --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 48 --k-mid 24 \
      --noise-end 0.02 --prop-steps 12000 --crop-sites 8 --window 3 --prop-layers 1 --local-channels 6 $extra \
      > $L/crop_train_$tag.log 2>&1
    log "train $tag rc=$? $(grep -E 'step 12000/12000' $L/crop_train_$tag.log | tail -1)"; fi
  run scripts/exp_rollout_stats_kolmogorov.py --tag $tag --scales 1,2,3 --n-steps 1000 --with-truth > $L/r15_rollout_$tag.log 2>&1
  log "rollout $tag rc=$? $(grep '^\[rollout\]' $L/r15_rollout_$tag.log | sed -E 's/\{.*std_late\": ([0-9.]+).*share_k_lt_2_late\": ([0-9.]+).*\| truth.*std_late\": ([0-9.]+).*share_k_lt_2_late\": ([0-9.]+).*/std \1 (truth \3) k<2 \2 (truth \4)/' | tr '\n' ' ')"
  ev $tag 2 1
  ev $tag 3 0
done
log "R15 DONE"
