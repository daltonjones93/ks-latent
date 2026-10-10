#!/bin/bash
# J17H fix (2026-10-10 15:30): the joint (a, b, g) estimation runs crashed in a progress print (fixed). Rerun them
# (alone and with bias correction) at each order's best tuning on seed 0; per the pre-registered rule, if an
# estimation config becomes the best seed-0 config for that order, run it on seeds 1-2.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/j17h.log; E=artifacts/experiments/da2d
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
level() { O=$1; P=J17H_p$O
  read A B G <<< "$($PY -c "import json; d=json.load(open('$E/j17h_fit_order$O.json')); print(d['a'], d['b'], d['g'])")"
  MODEL="--model-order $O --model-lap-coef $A --model-hyper-coef $B --model-nl-coef $G"
  pde() { tag=$1; shift; [ -f $E/i11_${tag}.json ] && return
    env $ENV nice -n 5 $PY scripts/exp_i11_pde_letkf.py --workers 2 $MODEL --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
    echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |Error" artifacts/logs/i11_${tag}.log | cut -c1-260)" >> $LOG; }
  EST="--estimate-param abg --param-init-std 0.05 --param-min-std 0.005"
  ME=$E/j16_model_error_hyp$O.npz
  pde ${P}_c16_rtps0.7_abg_s0 --loc-c 16 --rtps 0.7 $EST
  pde ${P}_c16_rtps0.7_abg_bc_q0_s0 --loc-c 16 --rtps 0.7 $EST --model-error-file $ME --bias-correct --q-scale 0
  BP=$($PY -c "
import glob, json, math
r = [(json.load(open(f))['steady'].get('analysis', 1e9), f) for f in glob.glob('$E/i11_${P}_*_s0.json')]
r = [x for x in r if math.isfinite(x[0])]; print(min(r)[1].split('i11_')[1][:-5])")
  echo "$(date '+%F %T') $P best over full grid incl. estimation: $BP" >> $LOG
  case $BP in *abg_bc_q0_s0) for sd in 1 2; do pde ${BP%_s0}_s$sd --loc-c 16 --rtps 0.7 $EST --model-error-file $ME --bias-correct --q-scale 0 --seed $sd & done; wait ;;
              *abg_s0) for sd in 1 2; do pde ${BP%_s0}_s$sd --loc-c 16 --rtps 0.7 $EST --seed $sd & done; wait ;; esac
}
level 5 & level 6 & level 8 & wait
echo "$(date '+%F %T') J17H ABG DONE" >> $LOG
