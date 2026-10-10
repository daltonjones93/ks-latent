#!/bin/bash
# 2026-10-10 (user go-ahead): remaining Part II experiments before writing.
#   I7  forecasts from the blended analysis (44^2, K=4 seeds 0-2, K=1 seed 0; every 5 steady cycles, 60-cycle leads)
#   I16 robustness: (a) climatological initial ensemble K=4/K=1; (b) assumed obs noise 0.5x / 2x; (c) 300 cycles 220^2
#   I14 clean timing (wall, peak RSS) K=1/K=4 at 44..440^2, one run at a time on a quiet machine
#   J5  model error at 220^2: PDE filter (lap 1.25, mean flow 1) vs the latent filter's 0.312, one at a time
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i7_i16_i14_j5.log; E=artifacts/experiments/da2d; TAG=p7c12d10_ks2d
log() { echo "$(date '+%F %T') $*" >> $LOG; }
lat() { out=$1; shift; [ -f $E/h9_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --obs-stride 8 --rtps 0.7 --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200 | tr '\n' ' ')"; }
# --- I7 + I16: after J17H (capacity) ---
while pgrep -f "run_j17h_all.sh|run_j17h_rtps06.sh" > /dev/null; do sleep 300; done
log "I7/I16 start"
FC="--forecast-every 5 --forecast-leads 60"
( lat I7_T4_s0 --n-tilings 4 --seed 0 $FC; lat I7_T4_s1 --n-tilings 4 --seed 1 $FC ) &
( lat I7_T4_s2 --n-tilings 4 --seed 2 $FC; lat I7_T1_s0 --n-tilings 1 --seed 0 $FC ) &
( lat I16a_clim_T4 --n-tilings 4 --ic-climatology; lat I16a_clim_T1 --n-tilings 1 --ic-climatology
  lat I16b_R0.5x_T4 --n-tilings 4 --assumed-obs-noise-std 0.05; lat I16b_R2x_T4 --n-tilings 4 --assumed-obs-noise-std 0.2 ) &
lat I16c_L220_300cyc_T4 --n-tilings 4 --L 220 --n-cycles 300 &
wait
log "I7/I16 DONE"
# --- I14: quiet machine (all other experiment queues finished) ---
while pgrep -f "run_j17b|run_j18a|run_j17h|train_kolmogorov_crop|exp_i11_pde_letkf|exp_h9_tiled_da" > /dev/null; do sleep 300; done
log "I14 start (quiet: load $(uptime | sed 's/.*averages: //'))"
for L in 44 88 220 330 440; do for T in 1 4; do out=I14_L${L}_T$T; [ -f $E/h9_${out}.json ] && continue
  OMP_NUM_THREADS=4 /usr/bin/time -l $PY scripts/exp_h9_tiled_da.py --tag $TAG --obs-stride 8 --rtps 0.7 --n-tilings $T --L $L \
    --no-free --out $out > artifacts/logs/h9_$out.log 2>&1
  log "done $out $(grep -E "^$out " artifacts/logs/h9_$out.log | grep -oE "'blend': [0-9.]+") wall=$($PY -c "import json;print(round(json.load(open('$E/h9_$out.json'))['wall_s']))") rss=$(grep 'maximum resident' artifacts/logs/h9_$out.log | awk '{printf "%.2fGB", $1/2^30}') load=$(uptime | sed 's/.*averages: //')"
done; done
log "I14 DONE"
# --- J5: model error at 220^2, one PDE run at a time ---
pde() { tag=$1; shift; [ -f $E/i11_${tag}.json ] && return
  OMP_NUM_THREADS=8 $PY scripts/exp_i11_pde_letkf.py --workers 8 --L 220 --loc-c 16 --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  log "done $tag rc=$? $(grep -E "^$tag " artifacts/logs/i11_${tag}.log | cut -c1-200)"; }
pde J5_L220_lap1.25_est --model-lap-coef 1.25 --rtps 0.8 --estimate-param lap --param-init-std 0.05
pde J5_L220_adv1_est --model-advect-x 1 --rtps 0.9 --estimate-param adv --param-init-std 0.3
log "J5 DONE"
