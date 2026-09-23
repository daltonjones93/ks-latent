#!/bin/zsh
# User-directed 2026-09-03: "would it be possible to have an mlp model
# that takes in fourier features (such as the FNO) and also actual latent
# states, as kind of a hybrid mlp? ... use history = 2." -- new backbone
# 'fourier_mlp' (ks_latent/models/propagator.py _FourierMLPHistoryDeltaBody,
# wired into PropagatorConfig/train_stage2_patched.py), distinct from the
# earlier 'fno_mlp' (which REPLACES the raw state with a learned spectral-
# conv operator, val_kmax_mse=0.257397, did not work well): this backbone
# concatenates, per history state, the RAW latent vector with the real/
# imaginary parts of its first fno_modes rfft frequencies (a FIXED,
# unlearned featurization, not a learned spectral mixing op), flattens
# across n_history=2 states, and feeds the result through the same plain
# residual-MLP body backbone='mlp' uses for its own history mode.
# Motivated by Section 66's discovered latent-index periodicity -- betting
# that handing the network Fourier-domain coordinates directly (on top of,
# not instead of, the raw state) helps more than replacing the raw state
# with them did.
#
# Verified directly before launching: identity-at-init, correct
# rollout_history shapes, bfloat16-autocast-safe (explicit float32 upcast
# for rfft, same bug class already fixed for fno_mlp/logdet_barrier_loss),
# fno_modes clipping to the max available. 8 new unit tests added
# (tests/unit/test_propagator_fourier_mlp.py), full suite re-run.
#
# fno_modes left at default (None = full spectrum, d_latent//2+1=23) --
# raw and Fourier features carry the same information in different bases
# at that setting, giving the network both representations to draw on
# freely rather than restricting to low frequencies only.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section65_mlpmarkovian_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep.pt
TAG=section65_ae_fourier_mlp_history2_k12_300ep

echo "=== Stage 2: fourier_mlp propagator (raw + Fourier features, history=2) on Section 65's AE, --amp, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone fourier_mlp --mode history --n-history 2 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$TAG" \
  > artifacts/logs/stage2_${TAG}.log 2>&1

echo "=== Section 65 fourier_mlp-propagator run complete ==="
tail -3 artifacts/logs/stage2_${TAG}.log
