#!/bin/zsh
# User-directed 2026-08-31, following Section 35's option 7 (data-adaptive
# w_varmatch target): "I like stage 2 (mitigate) 7 the best. is there a
# way to try this with this autoencoder? [stage2_history3_fullprop_wvar005
# _tw16_wspatial005_200ep_mlpmarkovian_k20.log]. should be clear pretty
# quickly whether that fixes the problem."
#
# That log's recipe stalled around val_kmax_mse~0.44-0.51 at k_max=20
# before being interrupted (Section 29's tail). Its AE
# (stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial005_200ep.pt,
# a FRESH 200-epoch AE, not fine-tuned) independently checked to have a
# similarly collapsed channel (min eigenvalue 4.55e-6, condition # 1.3e6 --
# comparable to Section 34's fine-tuned AE), so it's a good test case for
# whether --w-varmatch-adaptive (new capability, ks_latent/config.py
# Stage2TrainingConfig.w_varmatch_adaptive) actually helps, independent of
# whether the original stalled run even had w_varmatch on.
#
# Three matched single-seed runs, same recipe as the stalled run, so the
# ONLY thing varying is the variance-match treatment:
#   (a) no w_varmatch at all (baseline)
#   (b) w_varmatch=0.1, old fixed target=1 (the existing default mechanism)
#   (c) w_varmatch=0.1, --w-varmatch-adaptive (the new fix)
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial005_200ep.pt
RECIPE=(--backbone mlp --mode markovian --epochs 200 --k-max 20 --k-mid 13 --k-warmup-epochs 140 --k-mid-epochs 117 --seed 0)

echo "=== (a) baseline, no w_varmatch ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" "${RECIPE[@]}" \
  --tag wvmadaptive_baseline \
  > artifacts/logs/stage2_wvmadaptive_baseline.log 2>&1

echo "=== (b) w_varmatch=0.1, old fixed target=1 ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" "${RECIPE[@]}" \
  --w-varmatch 0.1 \
  --tag wvmadaptive_fixedtarget \
  > artifacts/logs/stage2_wvmadaptive_fixedtarget.log 2>&1

echo "=== (c) w_varmatch=0.1, --w-varmatch-adaptive (new) ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" "${RECIPE[@]}" \
  --w-varmatch 0.1 --w-varmatch-adaptive \
  --tag wvmadaptive_datatarget \
  > artifacts/logs/stage2_wvmadaptive_datatarget.log 2>&1

echo "=== all 3 runs complete ==="
for f in baseline fixedtarget datatarget; do
  echo "--- $f ---"
  grep "best_val_kmax_mse" artifacts/logs/stage2_wvmadaptive_${f}.log
done
