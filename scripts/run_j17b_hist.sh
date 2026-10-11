#!/bin/bash
# J17B memory test (2026-10-10, user-directed: "queue the two step test, except use 3 steps of memory").
# Same eps=0.2 data, autoencoder and Stage-B recipe as p7c12d10_memB0.2, but the propagator sees the
# current latent state and the 3 previous ones (mode="history", n_history=4, 0.3 time units of past).
# Question: is the latent filter's degradation on the hidden-field truth (0.628 vs 0.596 physics) due to
# the missing memory? Latent filter: seeds 0-2 with four tilings, seed 0 with one, as for memB0.2.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
EPS=0.2; MEM="$EPS,2,-1"; LOG=artifacts/logs/j17b.log; E=artifacts/experiments/da2d
DS=artifacts/datasets/ks2d_memB${EPS}_wide6.h5; BASE=p7c12d10_memB$EPS; TAG=${BASE}_h4; P=J17B_e${EPS}_h4
log() { echo "$(date '+%F %T') [$P] $*" >> $LOG; }
# same (frozen) autoencoder as the markovian model, so only the propagator differs
for f in "" .meta.json; do
  [ -f artifacts/stage1_ae_kolmogorov_crop_$TAG.pt$f ] || cp artifacts/stage1_ae_kolmogorov_crop_$BASE.pt$f artifacts/stage1_ae_kolmogorov_crop_$TAG.pt$f
done
if [ ! -f artifacts/stage2_prop_kolmogorov_crop_$TAG.pt ]; then
  while pgrep -f "train_kolmogorov_crop.py" > /dev/null; do sleep 120; done
  log "training started (n_history 4)"
  $PY scripts/train_kolmogorov_crop.py --dataset $DS --tag $TAG --crop-sites 40 --crop-sites-y 40 --n-offsets 16 \
    --local-channels 12 --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
    --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 --data-noise 0.1 --n-history 4 \
    > artifacts/logs/crop_train_$TAG.log 2>&1
  log "training done rc=$?"
fi
lat() { out=$1; shift
  [ -f $E/h9_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --obs-stride 8 --rtps 0.7 --truth-memory $MEM --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200 | tr '\n' ' ')"; }
lat ${P}_lat_T4_s0 --n-tilings 4 --seed 0 & lat ${P}_lat_T4_s1 --n-tilings 4 --seed 1 & lat ${P}_lat_T4_s2 --n-tilings 4 --seed 2 & wait
lat ${P}_lat_T1_s0 --n-tilings 1 --seed 0
log "DONE"
