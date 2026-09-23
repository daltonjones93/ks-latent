#!/bin/zsh
# User-directed 2026-09-10. Section 128 (Section 98's vit encoder/decoder +
# NEW masked_mlp_expand propagator, attn_window=3/expand_factor=3) was
# confirmed collapsed by direct test (D9-style Jacobian check on the
# Stage-1 aux propagator, 200 real samples: top singular value pinned at
# median=1.0021, p95=1.0031, min=1.0005, max=1.0035 -- essentially zero
# variation anywhere in phase space, sitting right at the marginal-
# stability boundary rather than showing the stretch-and-fold heterogeneity
# real chaos needs; Stage 2's val_kmax_mse plateaued at ~0.65 and barely
# moved over 30 epochs). The encoder itself did NOT collapse this time
# (z.std stayed healthy, 1.0-1.5, even after a 60-step rollout) -- this is
# the propagator settling at marginal stability uniformly, not the
# Sections-120-127 encoder-collapse mechanism recurring.
#
# Section 129 (identical to 128 but with the ENCODER's own --attn-window
# removed, i.e. fully global/dense attention, isolating whether encoder
# locality mattered) was killed by user direction before finishing --
# "129 is going to give us the same result as 128" -- since the collapse
# signature (marginal-stability-pinned propagator Jacobian) is a property
# of the PROPAGATOR, and the propagator was UNCHANGED between 128 and 129,
# there was no reason to expect a different outcome, and the user was
# right not to spend the compute confirming it.
#
# THIS SECTION instead applies the fix the diagnostic evidence actually
# points at: `w_local_expansion_floor`
# (ks_latent.training.losses.propagator_local_expansion_floor_loss, built
# Section 127) -- a differentiable floor on the propagator's own step-
# Jacobian spectral norm at real z, via vmap(jacrev(...)), the SAME
# technique used to confirm 128's collapse. Applies generically to
# `aux.step_one` (ks_latent/training/loops.py:493) -- no backbone
# restriction, nothing new to build for masked_mlp_expand.
#
# IMPORTANT DIFFERENCE FROM SECTION 127's OWN (unsuccessful) use of this
# same mechanism: Section 127 applied it to a spectral_pde propagator with
# its learned correction PINNED AT ~0 (correction_scale~0,
# --physics-prior-correction-warmup-epochs 100000) -- i.e. testing whether
# the floor loss ALONE, with essentially no learned freedom, could hold
# off collapse. masked_mlp_expand has FULL freedom to learn any local map
# here -- a genuinely more favorable context, not a repeat of an
# already-ruled-out test.
#
# FLOOR VALUE, chosen from the diagnostic evidence, not the mechanism's
# own default: the default floor=1.0 would barely fire here
# (relu(1.0 - 1.002) ~= 0 almost everywhere -- 128's own measured Jacobian
# norms already sit just above 1.0), so it would do essentially nothing.
# Set --local-expansion-floor-value 1.5 instead, chosen to sit close to
# this project's own benchmark for a genuinely chaotic propagator's median
# Jacobian norm (Section 85: median 1.60, p95 1.74) -- i.e. push the
# propagator toward that regime rather than merely off the exact
# marginal-stability line. --w-local-expansion-floor 0.1 matches Section
# 127's own weight (a reasonable, previously-smoke-tested starting point,
# not so large it dominates w_pred/reconstruction).
#
# Everything else UNCHANGED from Section 128: Section 98's exact vit
# encoder+decoder (d_model=56, pos_encoding=linear, attn_window=4,
# token_window=16, pool=token_mlp/dec_pool=token_mlp reduction=8/
# hidden=128), Section 98's exact regularizers (w_decorr=0, w_var=0.02,
# w_spatial=0.04 SIGNED, w_var_floor=0, w_logdet=0.008, w_smooth=0.003, NO
# lambda_z), masked_mlp_expand propagator (--prop-attn-window 3
# --masked-mlp-expand-factor 3), --full-propagator, 200 Stage-1 epochs +
# 300 Stage-2 epochs at the Section 52/98/100/128 k12 curriculum.
#
# Verified this turn: a real --profile smoke Stage 1 run (2 epochs)
# completed with the local-expansion-floor term firing (no errors, no MPS
# svdvals crash -- the CPU round-trip fix from Section 127 still applies),
# followed by a smoke Stage 2 warm-start run, both clean.
#
# Phase-1-first (matching every local-propagator section this arc): check
# the Stage-1 aux propagator's own Jacobian spectral norm distribution
# directly (same D9-style test used to confirm 128's collapse) BEFORE
# committing to the full Stage 2 warm-start + Gate 3/4 run.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section130_vitonly_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_localexpansionfloor015_200ep

echo "=== [1/3] Stage 1: Section 98's EXACT vit encoder+decoder (d_model=56, pos_encoding=linear, attn_window=4, token_window=16, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128) + masked_mlp_expand propagator (attn_window=3, expand_factor=3, UNCHANGED from 128) + NEW w_local_expansion_floor=0.1 (floor=1.5, n_samples=32, once per epoch), Section 98's EXACT regularizers, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone masked_mlp_expand --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --prop-attn-window 3 --masked-mlp-expand-factor 3 \
  --w-local-expansion-floor 0.1 --local-expansion-floor-value 1.5 --local-expansion-floor-n-samples 32 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.04 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.003 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Propagator Jacobian spectral norm check (same D9-style test used to confirm 128's collapse) + latent covariance spectrum ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
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
print('propagator step-Jacobian top singular value: median=%.4f  p95=%.4f  min=%.4f  max=%.4f  frac >= 1.0: %.3f' % (
    np.median(res), np.percentile(res, 95), res.min(), res.max(), float((res >= 1.0).mean())
))

z_np = z_all.reshape(-1, z_all.shape[-1]).numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
d8 = same_time_coupling_diagnostic_signed(z_np, n_null=500, seed=0)
print('D8 signed bandedness=%.4f p=%.4f' % (d8.bandedness_observed, d8.bandedness_p_value))
" > artifacts/logs/jacobiancheck_${TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${TAG}.log

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, --amp, k_max=12, 300 epochs (Section 52/98/100/128's shared curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 130 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
