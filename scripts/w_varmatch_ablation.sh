#!/bin/zsh
# Step 1 of Section 35's plan (user-directed 2026-08-31): "we have two
# options, try to avoid channel collapse in stage 1 or consider removing
# w_varmatch in phase 2... figure out what makes sense to try."
#
# Cheapest, zero-new-code test: on the EXISTING fine-tuned/collapsed AE
# (Section 34), compare Stage 2 mlp/markovian training at the Section 33
# recipe (k_max=8, 68 epochs) across w_varmatch=0.0 vs 0.1 (current
# default used throughout this document), 3 seeds each. Tests how much of
# the ~10x Section 33 gap is attributable to w_varmatch's hardcoded
# target=1.0 actively fighting the collapsed channel (Section 35), before
# investing in any new Stage-1 regularizer code.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE_FINETUNED=artifacts/stage1_ae_patched_full_history3_wspatial005_finetune_from_vitaux_200ep.pt

RECIPE=(--backbone mlp --mode markovian --epochs 68 --k-max 8 --k-warmup-epochs 48 --k-mid 5 --k-mid-epochs 40)

for wvm in 0.0 0.1; do
  tag_wvm=$(echo $wvm | tr -d '.')
  for seed in 0 1 2; do
    echo "=== [finetuned AE] w_varmatch=$wvm seed=$seed ==="
    mamba run -n da_env python scripts/train_stage2_patched.py \
      --ae-checkpoint "$AE_FINETUNED" --seed $seed "${RECIPE[@]}" \
      --w-varmatch $wvm \
      --tag multiseed_wvmablation-wvm${tag_wvm}_seed${seed} \
      > artifacts/logs/stage2_multiseed_wvmablation-wvm${tag_wvm}_seed${seed}.log 2>&1
  done
done

echo "=== all 6 w_varmatch-ablation runs complete -- summarizing ==="
mamba run -n da_env python scripts/summarize_multiseed.py \
  --pattern "artifacts/logs/stage2_multiseed_wvmablation-*.log" \
  > artifacts/logs/wvmablation_summary.log 2>&1
cat artifacts/logs/wvmablation_summary.log
