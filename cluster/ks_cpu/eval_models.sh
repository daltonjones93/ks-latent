#!/bin/bash
# Evaluate the retrained propagators of one system: the model's Lyapunov spectrum (D_KY, lambda_1) at the training size and the transfer sizes,
# initial conditions 0-2 (the paper's), for every variant and seed whose checkpoint exists.
#   sbatch ... cluster/ks_cpu/eval_models.sh SYSTEM        SYSTEM: kolm | ks2d | ks1d
set -uo pipefail
. "$HOME/latent_DA/cluster/ks_cpu/common.sh"
mkdir -p $A/experiments/crop
sys=$1
for v in gelu_ln swiglu_rms; do for seed in 0 1 2; do
  tag=mx_${sys}_${v}_s${seed}
  for ic in 0 1 2; do
    case $sys in
      kolm)
        P=$A/stage2_prop_kolmogorov_crop_$tag.pt; [ -f $P ] || continue
        for sc in 1 2 3; do s0=$(date +%s)
          python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $P --scale $sc --seed $ic --tag ${tag}_x${sc}_seed$ic > /dev/null 2>&1
          echo "EVAL $tag x$sc IC$ic rc=$? $(( $(date +%s) - s0 )) s"; done ;;
      ks2d)
        P=$A/stage2_prop_kolmogorov_crop_$tag.pt; [ -f $P ] || continue
        for sc in 1.0 1.5 2.0; do s0=$(date +%s); x=""; t=""
          case "$sc:$ic" in 1.0:0) t=81.77;; 1.0:1) t=80.66;; 1.0:2) t=83.60;; 1.5:0) t=192.50;; 1.5:1) t=185.60;; 1.5:2) t=187.5;; 2.0:0) t=338.3;; esac
          [ "$sc" = 2.0 ] && x="--n-directions 512"
          python scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $P --scale $sc --seed $ic --tag ${tag}_x${sc}_seed$ic ${t:+--true-dky $t} $x > /dev/null 2>&1
          echo "EVAL $tag x$sc IC$ic rc=$? $(( $(date +%s) - s0 )) s"; done ;;
      ks1d)
        P=$A/stage2_prop_patched_full_$tag.pt; [ -f $P ] || continue
        s0=$(date +%s)
        python scripts/eval_1d_transfer.py --tag $tag --ae-ckpt $ks1d_ae --prop-ckpt $P --L 100 200 400 --ic-seed $ic --ic normalized > /dev/null 2>&1
        echo "EVAL $tag L100/200/400 IC$ic rc=$? $(( $(date +%s) - s0 )) s" ;;
    esac
  done
done; done
