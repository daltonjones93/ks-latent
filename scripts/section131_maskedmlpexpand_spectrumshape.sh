#!/bin/zsh
# User-directed 2026-09-10. Section 130 (Section 98's vit encoder/decoder +
# masked_mlp_expand propagator + w_local_expansion_floor=0.1/floor=1.5) was
# confirmed NOT good, mechanistically: val_kmax_mse plateaued at 0.6236
# (barely different from Section 128's un-regularized 0.65). Direct
# inspection of the finished checkpoint's FULL Jacobian singular-value
# spectrum (not just the top entry) explained why --
#
#   top 5:    [1.304, 1.194, 1.156, 1.130, 1.049]
#   bottom 5: [0.599, 0.571, 0.559, 0.541, 0.521]
#   singular values >= 1.0:  5 out of 44
#   per-step volume-change factor (product of all 44 singular values): 8.9e-5
#   sum(log(singular values)): -9.32
#
# w_local_expansion_floor only constrains the SINGLE largest singular
# value. Training found the cheapest way to satisfy it: elevate exactly
# that one direction and let the other 43 decay toward 0.5-0.6, keeping
# net catastrophic contraction (~10,000x per step) -- the same collapse
# mechanism as ever, just confined to a lower-dimensional subspace instead
# of applying everywhere.
#
# User's own follow-up, which is exactly the right fix: "can we use the
# regularizer to force some singular vectors to have expansive values
# around 1.5 and others to have contracting values?"
#
# NEW: `propagator_spectrum_shape_loss` (ks_latent/training/losses.py) --
# shapes the WHOLE spectrum instead of just the top entry: floors the top
# `n_expand` singular values toward `expand_target` (1.5, matching Section
# 85's own genuinely-chaotic-propagator benchmark) AND separately floors
# the remaining `d - n_expand` toward a LOWER `contract_floor` (0.7 --
# still permits real net contraction, since KS is dissipative, but
# prevents the runaway 8.9e-5 collapse above). `n_expand` defaults to 11,
# this project's own established L=100 replication target for the number
# of positive Lyapunov exponents (CLAUDE.md section 18) -- an attempt to
# shape the LEARNED propagator toward the shape the TRUE attractor's own
# Lyapunov spectrum is already known to have, not an arbitrary split.
# Shares its Jacobian computation with w_local_expansion_floor via a new
# shared helper (_propagator_step_jacobian_singular_values) -- same
# expensive/once-per-epoch/mode=markovian-only convention. 8 new unit
# tests (exact-match/under-expanded/collapsed-tail/one-sided/gradient-
# flow/real-propagator/n_expand-boundary-validation), full suite re-run
# for regressions; the existing propagator_local_expansion_floor_loss
# tests still pass unchanged after the shared-helper refactor.
#
# Everything else UNCHANGED from Section 128/130: Section 98's exact vit
# encoder+decoder (d_model=56, pos_encoding=linear, attn_window=4,
# token_window=16, pool=token_mlp/dec_pool=token_mlp reduction=8/
# hidden=128), Section 98's exact regularizers (w_decorr=0, w_var=0.02,
# w_spatial=0.04 SIGNED, w_var_floor=0, w_logdet=0.008, w_smooth=0.003, NO
# lambda_z), masked_mlp_expand propagator (--prop-attn-window 3
# --masked-mlp-expand-factor 3), --full-propagator, 200 Stage-1 epochs +
# 300 Stage-2 epochs at the Section 52/98/100/128/130 k12 curriculum.
#
# Verified this turn: a real --profile smoke Stage 1 run completed cleanly
# with the new term firing (n_expand reduced to fit the tiny smoke
# d_latent -- the real run below uses the actual default, 11, valid at
# this run's d_latent=44).
#
# Phase-1-first (matching every local-propagator section this arc): the
# script's own [2/3] step reruns the exact D9-style full-spectrum check
# used to diagnose Section 130's failure, automatically, right after
# Stage 1 finishes -- direct before/after comparison against both 128 (no
# regularizer) and 130 (top-1-only regularizer) before committing to the
# full Stage 2 + Gate 3/4 run.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section131_vitonly_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_spectrumshape_n11_e15_c07_200ep

echo "=== [1/3] Stage 1: Section 98's EXACT vit encoder+decoder (d_model=56, pos_encoding=linear, attn_window=4, token_window=16, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128) + masked_mlp_expand propagator (attn_window=3, expand_factor=3, UNCHANGED from 128/130) + NEW w_spectrum_shape=0.1 (n_expand=11, expand_target=1.5, contract_floor=0.7, n_samples=32, once per epoch), Section 98's EXACT regularizers, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone masked_mlp_expand --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --prop-attn-window 3 --masked-mlp-expand-factor 3 \
  --w-spectrum-shape 0.1 --spectrum-shape-n-expand 11 --spectrum-shape-expand-target 1.5 \
  --spectrum-shape-contract-floor 0.7 --spectrum-shape-n-samples 32 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.04 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.003 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] FULL Jacobian singular-value spectrum check (same test used to diagnose Section 130's failure) + latent covariance spectrum ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.diagnostics import propagator_step_jacobian_spectral_norms, same_time_coupling_diagnostic_signed

AE = '$AE'
PROP = '$AUX'
ae, ae_cfg, _ = load_autoencoder_checkpoint(AE)
prop, prop_cfg, _ = load_propagator_checkpoint(PROP)
ae.eval(); prop.eval()

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
with torch.no_grad():
    z_all = ae.encode(traj.reshape(n*T, NX)).reshape(n, T, -1)

res = propagator_step_jacobian_spectral_norms(prop, z_all, n_samples=200, seed=0)
print('top singular value: median=%.4f  p95=%.4f  min=%.4f  max=%.4f' % (
    np.median(res), np.percentile(res, 95), res.min(), res.max()
))

z0 = z_all[0, 0]
J = jacrev(lambda z: prop.step_one(z.unsqueeze(0)).squeeze(0))(z0)
sv = torch.linalg.svdvals(J)
print('full spectrum at one real point:')
print('  top 5:', sv[:5].tolist())
print('  bottom 5:', sv[-5:].tolist())
print('  singular values >= 1.0:', int((sv >= 1.0).sum()), 'out of', sv.numel())
print('  per-step volume-change factor:', sv.prod().item())
print('  sum(log(singular values)):', torch.log(sv).sum().item())

z0b = z_all[:, 0, :]
with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=60)
sep_start = (traj_roll[:, 5, :] - traj_roll[:, 0, :]).norm(dim=-1).mean().item()
sep_end = (traj_roll[:, -1, :] - traj_roll[:, -6, :]).norm(dim=-1).mean().item()
cross_sample_std_end = traj_roll[:, -1, :].std(dim=0).mean().item()
print('rollout separation steps 0-5: %.4f   steps 54-59: %.4f   cross-sample z.std at final step: %.4f' % (sep_start, sep_end, cross_sample_std_end))

z_np = z_all.reshape(-1, z_all.shape[-1]).numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
d8 = same_time_coupling_diagnostic_signed(z_np, n_null=500, seed=0)
print('D8 signed bandedness=%.4f p=%.4f' % (d8.bandedness_observed, d8.bandedness_p_value))
" > artifacts/logs/jacobiancheck_${TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${TAG}.log

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, --amp, k_max=12, 300 epochs (Section 52/98/100/128/130's shared curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 131 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
