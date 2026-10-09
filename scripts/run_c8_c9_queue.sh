#!/bin/bash
# C8/C9 queue (2026-10-03). One task at a time; every task is skipped if its output
# exists, so this can be killed and rerun. Runs alongside the single-core alpha=0.05
# 3x truth run (exp_crop_pipeline.py), which C8's final analysis waits for.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
K05=$A/datasets/kolmogorov_trajectories_default.h5
WIDE=$A/datasets/kolmogorov_wide16_alpha0.05.h5
AE231=$A/stage1_ae_kolmogorov_full_section231_kolmogorov_localmlp2d_window3.pt
D2=$A/stage2_prop_kolmogorov_full_section236_kolmogorov_alpha0p05_localmlp2d_biotsavart_warmstart_k12_300ep.pt
log () { echo "[q $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }

# 1) D2 re-evaluated with the aux-config Lx bug fixed (normalized IC), 2x x 3 seeds
for s in 0 1 2; do
  run $E/icnorm/d2fix_x2_normalized_seed$s.json "D2-fixed x2 seed$s" \
    python scripts/exp_ic_normalization_check.py --system kolmogorov --ae-ckpt $AE231 --prop-ckpt $D2 \
      --dataset $K05 --x-scale 2 --ic-mode normalized --seed $s --tag d2fix_x2_normalized_seed$s \
      >> $A/logs/d2fix.log 2>&1
done

# 2) C8 model side + low-mode autocorrelation
run $E/c8/model_p2_x3_seed0_phys_vectors.npz "C8 model" \
  python scripts/exp_c8_unstable_subspace.py --part model >> $A/logs/c8.log 2>&1
run $E/c8/autocorr_x3.json "C8 autocorr" \
  python scripts/exp_c8_unstable_subspace.py --part autocorr >> $A/logs/c8.log 2>&1

# 3) C9(i): crop-trained, exact global Biot-Savart input (reuses P2's autoencoder)
[ -f $A/stage1_ae_kolmogorov_crop_c9i_a005.pt ] || cp $A/stage1_ae_kolmogorov_crop_p2_a005.pt $A/stage1_ae_kolmogorov_crop_c9i_a005.pt
run $A/stage2_prop_kolmogorov_crop_c9i_a005.pt "C9(i) train" \
  python scripts/train_kolmogorov_crop.py --dataset $WIDE --tag c9i_a005 --backbone local_mlp_2d_biotsavart \
    >> $A/logs/crop_train_c9i_a005.log 2>&1
for sc in 1 2 3; do for s in 0 1 2; do
  run $E/crop/c9i_a005_x${sc}_seed$s.json "C9(i) eval x$sc seed$s" \
    python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_c9i_a005.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_c9i_a005.pt --scale $sc --seed $s --tag c9i_a005_x${sc}_seed$s \
      --truth-json $E/crop/truth_a005_x3_seed0.json >> $A/logs/crop_eval_c9i_a005.log 2>&1
done; done

# 4) C9(ii): box-trained only (the "train small" arm) = D2 + normalized, zero-init velocity input
C9B=section238_kolmogorov_alpha0p05_c9box
run $A/stage2_prop_kolmogorov_full_${C9B}_warmstart_k12_300ep.pt "C9(ii) train" \
  python scripts/train_kolmogorov_localfield_standalone.py --dataset $K05 --tag $C9B --seed 0 \
    --reuse-ae-ckpt $AE231 --backbone local_mlp_2d_biotsavart --attn-window 3 --bs-vel-scale auto \
    --bs-zero-init-embed --stage1-epochs 1 --stage2-epochs 300 --checkpoint-every 20 >> $A/logs/c9ii_train.log 2>&1
for sc in 1 2 3; do for s in 0 1 2; do
  run $E/icnorm/c9ii_x${sc}_seed$s.json "C9(ii) eval x$sc seed$s" \
    python scripts/exp_ic_normalization_check.py --system kolmogorov --ae-ckpt $AE231 \
      --prop-ckpt $A/stage2_prop_kolmogorov_full_${C9B}_warmstart_k12_300ep.pt --dataset $K05 --x-scale $sc \
      --ic-mode normalized --seed $s --tag c9ii_x${sc}_seed$s >> $A/logs/c9ii_eval.log 2>&1
done; done

# 5) C8 analysis, once the truth run has written its result
while [ ! -f $E/crop/truth_a005_x3_seed0.json ]; do sleep 120; done
run $E/c8/c8_result.json "C8 analyze" python scripts/exp_c8_unstable_subspace.py --part analyze >> $A/logs/c8.log 2>&1
log "ALL DONE"
