#!/bin/zsh
# User-directed 2026-09-09, "Section 111": "I don't think the joint
# training idea is going to work, based on these results, maybe we could
# try rerunning 104 euler, with higher derivatives. we just need to limit
# the higher order modes in the pde somehow, otherwise we can't fit
# anything. this pde idea just might not work too, I'm running out of
# ideas."
#
# Pivot back from the pde_head joint-distillation arc (Sections 107-110,
# which consistently showed pde_head learning almost nothing regardless
# of weight -- see docs/sine_transform_pde_plan.md, MSE(pde_head vs aux)
# stuck at ~98% of aux's own step-size MSE in both Section 109 and 110)
# to the ORIGINAL Section 104 architecture: backbone="spectral_pde" as the
# PRIMARY propagator itself (not a side-channel distilled against a free
# model), on this repo's own established L=22/K=24/N_w=64 recipe.
#
# TWO changes on top of Section 104's exact original recipe
# (scripts/section104_shapefloor_wideK_euler.sh):
#
#   1. --spectral-max-order 4 -> 6 ("higher derivatives") -- adds two more
#      derivative channels (w_xxxxx, w_xxxxxx) to field()'s input stack,
#      giving the pointwise MLP a richer local feature basis to work
#      with.
#
#   2. --w-lowpass-rollout ADDED to BOTH phases (did not exist when
#      Section 104 first ran -- built afterward, specifically motivated by
#      visualizing Section 104's own D_KY=22 result, which showed the
#      propagator's free-running rollout populates spurious high-
#      wavenumber content the true attractor never has). This directly
#      answers "we just need to limit the higher order modes in the pde
#      somehow": differentiating amplifies high-wavenumber content by
#      k^n, and at K=24/L=22 the highest kept wavenumber is k~=6.6 --
#      going from order 4 to order 6 means that amplification factor
#      jumps from ~6.6^4=1900 to ~6.6^6=83000, a ~44x increase in how much
#      any residual high-frequency content in z gets amplified before
#      reaching the MLP. Without directly suppressing that, higher-order
#      derivatives make the numerics measurably more fragile, not just
#      richer -- w_lowpass_rollout is the tool built for exactly this.
#
#      Weight calibration (real, computed 2026-09-09, this exact K=24/
#      N_w=64/L=22/max_order=6 config):
#        Phase 1 (fresh init, non-zero-init aux so the term isn't
#        trivially ~0): raw l_lowpass_rollout=247.24, weighted l_pred
#        contribution (w_pred=1.5) = 2.18, ratio ~113x. Targeting roughly
#        2x l_pred's own weighted signal (aggressive, matching the
#        explicit ask to make this term actually bind, not a token
#        nudge): w_lowpass_rollout = 0.02.
#        Phase 2: reuses the w_lowpass_rollout=1.0 calibration already
#        derived earlier this session against a real TRAINED checkpoint
#        at this same K=24/N_w=64/L=22 scale (raw l_lowpass_rollout(power=1)
#        =0.183 vs l_latent(l2)=0.087, ratio ~2.1x there) -- same
#        architecture/scale, not re-derived fresh here.
#
# Everything else IDENTICAL to Section 104's own original recipe: K=24,
# N_w=64, L=22, euler, ode_substeps=3, w_pred=1.5, w_shape_floor=0.1
# (AGGRESSIVE, unchanged), w_lowpass=0.0012 (unchanged, still constrains
# the encoder's own direct z), w_var=w_spatial=w_decorr=w_logdet=0, joint
# Phase-1/Phase-2 design, dt_snap=1.0, same Phase-2 schedule (k_max=12,
# k_mid=8, k_warmup_epochs=56, k_mid_epochs=46, epochs=80).
#
# Verified via a real smoke run before launching (2026-09-09):
# --spectral-max-order 6 --w-lowpass-rollout 0.02 trains cleanly (no NaN)
# at this exact K=24/N_w=64 sizing.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section111_spectralfield_L22_K24_Nw64_maxorder6_lowpassrollout_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (euler, ode_substeps=3, max_order=6, --full-propagator hidden=128/n_blocks=3), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02 (NEW), dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 --spectral-max-order 6 --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0 (NEW), --amp, 80 epochs, then Gate 3/4 ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp --k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80 \
  --w-lowpass-rollout 1.0 --lowpass-rollout-power 1.0 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 1.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dt-snap 1.0 --L 22.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 22.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
wait

echo "=== Section 111 complete ==="
