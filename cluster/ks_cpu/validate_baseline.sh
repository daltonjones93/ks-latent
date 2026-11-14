#!/bin/bash
# Pipeline check on the CPU: evaluate the published Paper I/II checkpoints (IC 0) and compare with the paper's numbers.
#   Kolmogorov k_xs01_v2: paper 16.0 / 32.7 / 52.1 (1x / 2x / 3x; truth 15.88 / 30.34 / 56.35)
#   2D KS C2 seed 0:      paper 81.5 / 183.9 / 328.7 (44^2 / 66^2 / 88^2; truth IC 0: 81.77 / 192.50 / 338.3)
set -uo pipefail
. "$HOME/latent_DA/cluster/ks_cpu/common.sh"
mkdir -p $A/experiments/crop
which=${1:-kolm}
if [ "$which" = kolm ]; then
  for ic in ${ICS:-0}; do for sc in 1 2 3; do s0=$(date +%s)
    python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_k_xs01_v2.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_k_xs01_v2.pt \
      --scale $sc --seed $ic --tag val_k_xs01_v2_x${sc}_seed$ic 2>&1 | tail -3
    echo "EVAL kolm x$sc IC$ic: $(( $(date +%s) - s0 )) s"; done; done
else
  AE=$A/stage1_ae_kolmogorov_crop_r13_C2_ks2d_s44big_k48_seed0.pt; PR=$A/stage2_prop_kolmogorov_crop_r13_C2_ks2d_s44big_k48_seed0.pt
  for sc in 1.0 1.5 2.0; do s0=$(date +%s)
    case $sc in 1.0) t=81.77; x="";; 1.5) t=192.50; x="";; 2.0) t=338.3; x="--n-directions 512";; esac
    python scripts/exp_crop_eval_ks2d.py --ae-ckpt $AE --prop-ckpt $PR --scale $sc --seed 0 --tag val_r13C2_x${sc}_seed0 --true-dky $t $x 2>&1 | tail -3
    echo "EVAL ks2d x$sc: $(( $(date +%s) - s0 )) s"; done
fi
