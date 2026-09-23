#!/bin/zsh
# User-directed 2026-09-10, following Section 136's genuine chaos recovery
# (D_KY=23.75, n_positive=13, lambda1=0.112 -- in the established L=100
# benchmark band, the first clean success in this whole Sections 128-136
# arc) but with two flagged weaknesses: d_latent=96 (n_sites=32,
# local_channels=3) is more than 2x wider than the classical embedding-
# theory target (2*D_KY~=44 for this attractor), and the latent covariance
# was severely ill-conditioned (cond#=1.4e16, one channel with ~zero real
# variance) -- likely related to why DA skill/calibration (1.84x, 0.23)
# were both notably weaker than this project's best L=100 models
# (3.1-4.2x, 0.3-0.5).
#
# "so can we rerun 136 with latent dimension 44? also increase w_logdet,
# w_spatial and w_smooth for better conditioning."
#
# EXACT 44 is not achievable: d_latent = n_sites * local_channels, and
# n_sites must divide NX=256 = 2^8 (a pure power of 2, tied to the fixed
# dataset's own grid). 44 = 4*11, and 11 is not a power of 2 -- the only
# integer factorizations of 44 with n_sites dividing 256 are (n_sites=4,
# c=11), (n_sites=2, c=22), (n_sites=1, c=44), every one of which forces
# n_sites<=4 -- collapsing the "many small local sites" design Phase 10 is
# actually for (each site would span 64+ grid points, ~25+ physical units,
# destroying the fine local structure the whole architecture exists to
# capture). Used the closest achievable value that preserves reasonable
# site resolution instead: n_sites=16, local_channels=3 (unchanged from
# 135/136) -> d_latent=48, only 4 off from 44. site_mix_radius reduced
# 2->1 (n_site_mix_layers unchanged at 3) to compensate for patch_size
# doubling 8->16 -- keeps the encoder's own receptive field at
# patch_size*(1+layers*radius) = 16*4 = 64 grid points = 25.0 physical
# units, matching Section 135/136's own ~22-25 physical-unit target
# (CLAUDE.md's "2-3 correlation lengths ~20-30 physical units") rather
# than letting it balloon past it now that each site covers more ground.
#
# Regularizers raised for conditioning (Section 98's own values were the
# baseline in 136): w_logdet 0.008 -> 0.03 (this project's own established
# "actual fix" for organic collapse/ill-conditioning -- Section 38 pushed
# cond# from 1e5-1e6 to single digits with this exact mechanism -- gets
# the largest relative jump here since cond#=1.4e16 is far beyond typical
# "unhealthy" territory this project has seen), w_spatial 0.04 -> 0.08,
# w_smooth 0.003 -> 0.01. These are directional first-pass increases, not
# swept/tuned -- if conditioning improves but overshoots into a real
# accuracy cost, dial back; if still ill-conditioned, go further.
#
# Everything else UNCHANGED from Section 136: mlp propagator (fully
# global/unconstrained -- Section 136's own use of it produced the
# significant, NON-tautological D3 bandedness result, since mlp has zero
# architectural locality forced onto it), --full-propagator, 200 Stage-1
# epochs + 300 Stage-2 epochs at the established k12 curriculum. Same
# before/after Jacobian-spectrum + conditioning check as every section in
# this arc; also re-running visualize_rollout.py + Gate 3/4 at the end
# (matching what was done for 136) so this can be compared directly
# against 136's own D_KY/n_positive/lambda1/DA-skill/D3-9 numbers.
#
# Verified via a real --profile smoke run of both stages, no errors,
# before launching. d_latent=48/patch_size=16/encoder-RF=25.0 physical
# units confirmed by direct instantiation.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section137_localfield_p16c3_d48_propmlp_strongerreg_200ep

echo "=== [1/5] Stage 1: local_field encoder+decoder (n_sites=16, local_channels=3, d_latent=48, site_mix_radius=1, n_site_mix_layers=3, hidden=32) + mlp propagator + STRENGTHENED regularizers (w_var=0.02, w_spatial=0.08 SIGNED, w_logdet=0.03, w_smooth=0.01), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 16 --local-field-channels 3 --local-field-mix-radius 1 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.02 --w-var-floor 0 --w-spatial 0.08 --spatial-signed --w-logdet 0.03 \
  --w-smooth 0.01 \
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

echo "=== [2/5] FULL Jacobian spectrum + conditioning check on the STAGE-1 propagator ==="
_spectrum_check "$AE" "$AUX" "stage1" > artifacts/logs/jacobiancheck_stage1_${TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage1_${TAG}.log

echo "=== [3/5] Stage 2: warm-started from Stage 1's own mlp aux, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [4/5] FULL Jacobian spectrum + conditioning check on the FINAL STAGE-2 propagator ==="
_spectrum_check "$AE" "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log

echo "=== [5/5] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) -- directly comparable to Section 136's own numbers ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --rollout-steps 200 --tag "$STAGE2_TAG" \
  > artifacts/logs/visualize_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/visualize_${STAGE2_TAG}.log

mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
P1=$!
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
P2=$!
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
P3=$!
wait $P1 $P2 $P3

echo "=== Section 137 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D4 verdict|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
