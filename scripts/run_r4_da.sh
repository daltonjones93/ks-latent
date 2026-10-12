#!/bin/bash
# R4 data assimilation (user 2026-10-12): the latent filter with the small-source models, whatever their D_KY.
# Settings as Paper II (4 tilings, 64 members, RTPS 0.7, GC 1.5, model noise 0.02, 100 cycles, stride 8), with the
# observation error scaled to the same PHYSICAL value as the canonical runs (0.1 x 6.172): the filter's noise is
# in each checkpoint's standardized units, and the small-domain data have smaller stds (44^2: 3.821, 66^2: 4.038).
# Control: p7b_ks2d (same 6-channel per-patch recipe, 264^2 windows, reach 6) at the same settings.
# 440^2 runs one at a time (~17 GB each). Log: artifacts/logs/r4_da.log.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; E=$A/experiments/da2d
log() { echo "$(date '+%F %T') [R4-DA] $*" >> $L/r4_da.log; }
obs() { $PY -c "import torch; ck=torch.load('$A/stage1_ae_kolmogorov_crop_$1.pt', map_location='cpu', weights_only=False); print(round(0.1 * 6.172126667325263 / ck['normalization_std'], 5))"; }
run() { tag=$1; Ls=$2; T=$3; out=R4_${tag}_L${Ls}_T$T; [ -f $E/h9_$out.json ] && return
  o=$(obs $tag)
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $tag --L $Ls --n-tilings $T --rtps 0.7 --obs-stride 8 \
    --obs-noise-std $o --out $out > $L/h9_$out.log 2>&1
  rc=$?; log "$out rc=$rc obs=$o $($PY scripts/r4_summary.py da $E/h9_$out.json)"; }
wait_ckpt() { until [ -f $A/stage2_prop_kolmogorov_crop_$1.pt ]; do sleep 60; done; }
log "start"
wait_ckpt r4c_ks2d_s44_r3_eq
run r4c_ks2d_s44_r3_eq 44 4; run r4c_ks2d_s44_r3_eq 44 1; run p7b_ks2d 44 4
run r4c_ks2d_s44_r3_eq 440 4
wait_ckpt r4a_ks2d_s66_r4_eq; run r4a_ks2d_s66_r4_eq 44 4
run p7b_ks2d 440 4
run r4a_ks2d_s66_r4_eq 440 4
wait_ckpt r4b_ks2d_s66_r5_eq; run r4b_ks2d_s66_r5_eq 44 4; run r4b_ks2d_s66_r5_eq 440 4
log "R4 DA DONE"
