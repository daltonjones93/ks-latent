#!/bin/bash
# J17H follow-up (2026-10-10): RTPS 0.6 (the physics filter was over-dispersed at 0.7, collapsed at 0.5), all
# orders, both c, seed 0. After each level's main queue finishes, if an RTPS-0.6 config is now the best seed-0
# config for that level, run it on seeds 1-2 (pre-registered selection on seed 0 over the whole grid).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/j17h.log; E=artifacts/experiments/da2d
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
level() { O=$1; P=J17H_p$O
  read A B G <<< "$($PY -c "import json; d=json.load(open('$E/j17h_fit_order$O.json')); print(d['a'], d['b'], d['g'])")"
  MODEL="--model-order $O --model-lap-coef $A --model-hyper-coef $B --model-nl-coef $G"
  pde() { tag=$1; shift; [ -f $E/i11_${tag}.json ] && return
    env $ENV nice -n 10 $PY scripts/exp_i11_pde_letkf.py --workers 2 $MODEL --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
    echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |Error" artifacts/logs/i11_${tag}.log | cut -c1-200)" >> $LOG; }
  for c in 8.25 16; do pde ${P}_c${c}_rtps0.6_s0 --loc-c $c --rtps 0.6; done
  while pgrep -f "run_j17h.sh $O" > /dev/null; do sleep 120; done
  BP=$($PY -c "
import glob, json, math
r = [(json.load(open(f))['steady'].get('analysis', 1e9), f) for f in glob.glob('$E/i11_${P}_*_s0.json')]
r = [x for x in r if math.isfinite(x[0])]; print(min(r)[1].split('i11_')[1][:-5])")
  echo "$(date '+%F %T') $P best over full grid incl. RTPS 0.6: $BP" >> $LOG
  case $BP in *rtps0.6_s0) C=$(echo $BP | sed -E 's/.*_c([0-9.]+)_rtps.*/\1/')
    pde ${BP%_s0}_s1 --loc-c $C --rtps 0.6 --seed 1 & pde ${BP%_s0}_s2 --loc-c $C --rtps 0.6 --seed 2 & wait ;; esac
}
level 5 & level 6 & level 8 & wait
echo "$(date '+%F %T') J17H RTPS0.6 DONE" >> $LOG
