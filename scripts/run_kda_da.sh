#!/bin/bash
# SK5 (2026-10-11): Kolmogorov-flow DA with the noisy-trained 12-channel model k_xs005_c12d10 (alpha = 0.05).
# Waits for scripts/run_kda_train.sh (training + D_KY at 1x/2x/3x), then runs the latent filter with the 2D KS
# settings (64 members, GC 1.5 sites, RTPS 0.7, model noise 0.02, 5 model steps = 1.0 time unit = 0.08
# Lyapunov times per cycle, observation stride 8), tilings shifted in x only (SK4 smoke: y-shifted tilings
# put the encoder at forcing phases outside training and their analyses fail):
#   1x: 4 tilings, seeds 0-2; 1 tiling, seed 0;  3x: 4 tilings and 1 tiling, seed 0;  100 cycles;
#   sensitivity at 1x, seed 0, 4 tilings: RTPS 0.9; model noise 0.04 (the smoke run was under-dispersed).
# Log: artifacts/logs/kda_da.log; results artifacts/experiments/kda/k_KDA_*.json.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
TAG=k_xs005_c12d10; LOG=artifacts/logs/kda_da.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q "KDA TRAIN DONE" artifacts/logs/kda.log 2>/dev/null; do sleep 300; done
log "start; D_KY: $(grep 'eval x' artifacts/logs/kda.log | tr '\n' ' ')"
run() { out=$1; shift
  [ -f artifacts/experiments/kda/k_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_k_tiled_da.py --tag $TAG --tiling-axes x --floor-per-tiling --out $out "$@" \
    > artifacts/logs/kda_$out.log 2>&1
  log "done $out rc=$? $(tail -1 artifacts/logs/kda_$out.log | cut -c1-300)"; }
( run KDA_x1_T4_s0 --scale 1 --n-tilings 4 --seed 0; run KDA_x1_T1_s0 --scale 1 --n-tilings 1 --seed 0 ) &
( run KDA_x1_T4_s1 --scale 1 --n-tilings 4 --seed 1; run KDA_x1_T4_s2 --scale 1 --n-tilings 4 --seed 2 ) &
( run KDA_x1_T4_rtps0.9_s0 --scale 1 --n-tilings 4 --rtps 0.9 --seed 0
  run KDA_x1_T4_mn0.04_s0 --scale 1 --n-tilings 4 --model-noise-std 0.04 --seed 0 ) &
( run KDA_x3_T4_s0 --scale 3 --n-tilings 4 --seed 0; run KDA_x3_T1_s0 --scale 3 --n-tilings 1 --seed 0 ) &
wait
log "KDA DA DONE"
