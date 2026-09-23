#!/bin/zsh
# User-directed 2026-09-10: "135 looks like it also collapsed? maybe we
# should try it with a mlp propagator since the encoder is now localized
# (if I understand correctly)."
#
# Section 135 (local_field encoder + masked_mlp_expand propagator) did NOT
# fail the same way Sections 128-134 did. It's a genuinely different
# failure signature:
#   Stage 1: rollout separation GREW 26.7 -> 157.7 over 60 steps (every
#     prior section's separation SHRANK toward zero) -- divergence, not
#     collapse. Latent covariance nearly singular: min_eig=9.45e-16,
#     cond#=2.06e15 -- at least one channel carries ~zero real variance,
#     a conditioning problem in the ENCODER itself, independent of which
#     propagator it's paired with.
#   Stage 2: partially reined in the blowup (separation ended 6.67->5.06)
#     but landed in a still-wrong regime (sum(log(sv))=-106.6, several
#     near-zero singular values).
#
# User's own reasoning, and it's exactly right: masked_mlp_expand's
# architectural locality was motivated by the OLD (vit) encoder having no
# guaranteed physical ordering to its flat index. Now that local_field
# bakes genuine physical locality into the ENCODER itself, there's no
# obvious reason the PROPAGATOR also needs to be architecturally masked --
# H-PROP's own repeatedly-confirmed finding (this whole project, many
# sections) is that a propagator recovers genuine chaos IFF it has a
# GLOBAL receptive field and UNCONSTRAINED (non-softmax) mixing; every
# local-receptive-field propagator tried so far (local_mlp, masked_mlp,
# node, masked_mlp_wide, masked_mlp_expand) has collapsed regardless of
# mechanism. This section pairs the new architecturally-local encoder with
# `mlp` -- the plain, fully global/unconstrained residual-MLP backbone,
# one of H-PROP's own established winners on every PRIOR (non-local-field)
# encoder tried in this project.
#
# SECOND CHANGE, made before this launch (user-directed, 2026-09-10,
# "are we trying to regularize the spectrum in 134? maybe we shouldn't
# be? is it possible we pushed it towards instability and divergence? we
# should stick with the regularizers from 85 or 98"): Section 135's own
# divergence (rollout separation GREW 26.7->157.7; latent covariance
# nearly singular, cond#=2.06e15) happened under a deliberately MINIMAL
# regularizer set (w_var=0.02, w_logdet=0.008 only -- w_spatial/w_smooth
# dropped, to isolate the new encoder architecture as cleanly as
# possible). That ablation itself is the likely culprit: Section 98's own
# recipe always paired FOUR terms (w_var, w_spatial SIGNED, w_logdet,
# w_smooth) for numerical health on its ViT encoder; a new, unnormalized
# conv-based encoder given only two of those four may simply be
# under-constrained, consistent with what was observed (one latent
# channel with ~zero real variance). This run restores Section 98's FULL
# regularizer package (w_var=0.02, w_spatial=0.04 SIGNED, w_logdet=0.008,
# w_smooth=0.003) on top of the encoder swap -- NOT the spectrum-shape
# mechanism (that was Section 134's own, separate, PROPAGATOR-Jacobian-
# specific regularizer, never used on local_field runs and not implicated
# in this divergence at all -- Section 134's own failure was collapse,
# not divergence, the opposite direction).
#
# Two variables changed from Section 135 in this one launch (encoder-vs-
# propagator isolation already established by 135 itself; this run
# additionally fixes the regularizer-package question before doing
# further encoder/propagator comparisons): --aux-backbone
# masked_mlp_expand -> mlp, AND --w-var/--w-logdet-only -> Section 98's
# full four-term package. Same encoder (local_field, n_sites=32,
# local_channels=3, site_mix_radius=2, n_site_mix_layers=3, hidden=32),
# same --full-propagator/200 Stage-1 epochs/300 Stage-2 epochs at the
# established k12 curriculum.
#
# Verified via a real --profile smoke run of both stages, no errors,
# before launching.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section136_localfield_p32c3_propmlp_200ep

echo "=== [1/4] Stage 1: SAME local_field encoder+decoder as 135 (n_sites=32, local_channels=3, site_mix_radius=2, n_site_mix_layers=3, hidden=32) + mlp propagator (fully global/unconstrained, H-PROP's own established winner) + Section 98's FULL regularizer package (w_var=0.02, w_spatial=0.04 SIGNED, w_logdet=0.008, w_smooth=0.003), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.02 --w-var-floor 0 --w-spatial 0.04 --spatial-signed --w-logdet 0.008 \
  --w-smooth 0.003 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

_spectrum_check() {
  local AE_PATH=$1
  local PROP_PATH=$2
  local LABEL=$3
  mamba run -n da_env python -c "
import h5py, numpy as np, torch
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.diagnostics import propagator_step_jacobian_spectral_norms

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE_PATH')
prop, prop_cfg, _ = load_propagator_checkpoint('$PROP_PATH')
ae.eval(); prop.eval()

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
with torch.no_grad():
    z_all = ae.encode(traj.reshape(n*T, NX)).reshape(n, T, -1)

res = propagator_step_jacobian_spectral_norms(prop, z_all, n_samples=200, seed=0)
print('[$LABEL] top singular value: median=%.4f  p95=%.4f  min=%.4f  max=%.4f' % (
    np.median(res), np.percentile(res, 95), res.min(), res.max()
))

z0 = z_all[0, 0]
J = jacrev(lambda z: prop.step_one(z.unsqueeze(0)).squeeze(0))(z0)
sv = torch.linalg.svdvals(J)
print('[$LABEL] top 5:', sv[:5].tolist())
print('[$LABEL] bottom 5:', sv[-5:].tolist())
print('[$LABEL] singular values >= 1.0:', int((sv >= 1.0).sum()), 'out of', sv.numel())
print('[$LABEL] per-step volume-change factor:', sv.prod().item())
print('[$LABEL] sum(log(singular values)):', torch.log(sv).sum().item())

z0b = z_all[:, 0, :]
with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=60)
sep_start = (traj_roll[:, 5, :] - traj_roll[:, 0, :]).norm(dim=-1).mean().item()
sep_end = (traj_roll[:, -1, :] - traj_roll[:, -6, :]).norm(dim=-1).mean().item()
cross_sample_std_end = traj_roll[:, -1, :].std(dim=0).mean().item()
print('[$LABEL] rollout separation steps 0-5: %.4f   steps 54-59: %.4f   cross-sample z.std final step: %.4f' % (sep_start, sep_end, cross_sample_std_end))

z_np = z_all.reshape(-1, z_all.shape[-1]).numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('[$LABEL] spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
"
}

echo "=== [2/4] FULL Jacobian spectrum check on the STAGE-1 propagator ==="
_spectrum_check "$AE" "$AUX" "stage1" > artifacts/logs/jacobiancheck_stage1_${TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage1_${TAG}.log

echo "=== [3/4] Stage 2: warm-started from Stage 1's own mlp aux, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [4/4] FULL Jacobian spectrum check on the FINAL STAGE-2 propagator ==="
_spectrum_check "$AE" "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log

echo "=== Section 136 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
