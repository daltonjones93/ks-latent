#!/bin/bash
# Raw vs normalized transfer IC, seed 0, sequential (one job at a time).
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
A=artifacts
K05=$A/datasets/kolmogorov_trajectories_default.h5
K10=$A/datasets/kolmogorov_trajectories_alpha0.1.h5
KS2=$A/datasets/ks2d_trajectories_default.h5
run () { tag=$1; shift
  [ -f $A/experiments/icnorm/$tag.json ] && { echo "skip $tag"; return; }
  echo "[$(date +%T)] start $tag"
  nice -n 19 python scripts/exp_ic_normalization_check.py --tag $tag "$@" > $A/logs/icnorm_$tag.log 2>&1 \
    && echo "[$(date +%T)] done $tag" || echo "[$(date +%T)] FAILED $tag"
}
for mode in raw normalized; do
  run s231_x2_$mode --system kolmogorov --ae-ckpt $A/stage1_ae_kolmogorov_full_section231_kolmogorov_localmlp2d_window3.pt \
    --prop-ckpt $A/stage2_prop_kolmogorov_full_section231_kolmogorov_localmlp2d_window3_warmstart_k12_300ep.pt \
    --dataset $K05 --x-scale 2 --ic-mode $mode
  run s233_x2_$mode --system kolmogorov --ae-ckpt $A/stage1_ae_kolmogorov_full_section233_kolmogorov_alpha0p1_localmlp2d_window3.pt \
    --prop-ckpt $A/stage2_prop_kolmogorov_full_section233_kolmogorov_alpha0p1_localmlp2d_window3_warmstart_k12_300ep.pt \
    --dataset $K10 --x-scale 2 --ic-mode $mode
  run d3_x2_$mode --system kolmogorov --ae-ckpt $A/stage1_ae_kolmogorov_full_section231_kolmogorov_localmlp2d_window3.pt \
    --prop-ckpt $A/stage2_prop_kolmogorov_full_section237_kolmogorov_alpha0p05_multiscale_local_mlp2d_warmstart_k12_300ep.pt \
    --dataset $K05 --x-scale 2 --ic-mode $mode
  run ks2d_x1.5_$mode --system ks2d --ae-ckpt $A/stage1_ae_2dks_full_section230_ks2d_localfield_localmlp2d_window3.pt \
    --prop-ckpt $A/stage2_prop_2dks_full_section230_ks2d_localfield_localmlp2d_window3_warmstart_k12_300ep.pt \
    --dataset $KS2 --x-scale 1.5 --y-scale 1.5 --ic-mode $mode
done
echo "[$(date +%T)] ALL DONE"
