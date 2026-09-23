#!/bin/zsh
# User-directed 2026-09-10: "do you think 143 would work with a masked mlp
# propagator (with the expansion we developed today?) I think this may be
# easier for the pde to learn. but hard to get chaos. maybe we can use the
# D3 test for [141] to determine the required window size/receptive
# field." -> measured directly on Section 141's own real D3 coupling
# matrix (genuinely chaotic, D3 p=0.0000): radius 3 (Section 135's own
# masked_mlp_expand window) captures only 40.3% of real coupling mass;
# radius 9 (Sections 128-134's window) only 67.8%; radius 20-30 captures
# 84-90%. Then: "for 3 layers if we set attn_window=20, what is the
# receptive field of masked_mlp_expand?" -> masked_mlp_expand is a FIXED
# 3-layer stack (input_proj/mid_proj/output_proj), each using the SAME
# fractional threshold attn_window/d_latent regardless of that layer's
# own width -- composing 3 such layers gives a TOTAL receptive field of
# 3*attn_window (same additive composition as a 3-layer CNN stack).
# attn_window=20 would give an effective radius of 60 out of d_latent=96
# -- close to fully dense, not the ~20-30 target from the D3 measurement.
# User: "let's try attn_window=8 just for fun. I don't think it will
# work though." -- attn_window=8 gives an effective COMPOSED receptive
# field of 3*8=24, landing right in the 20-30 target range (radius 24
# captures roughly 84-85% of 141's own real coupling mass, interpolating
# the measured table).
#
# This is a genuinely different, properly-sized test of an idea Section
# 135 already tried under-sized (attn_window=3, effective receptive field
# 9, capturing only 40% of real coupling): does giving masked_mlp_expand
# enough width to capture the BULK (not just a third) of the real
# dynamical coupling change the outcome? Prediction (both the user's and
# this project's own H-PROP precedent): still expected to fail to sustain
# chaos, since H-PROP's own repeated finding is that ANY exact
# architectural zero-masking collapses/destabilizes training, not simply
# "insufficient width" -- and 141's own measurement shows even radius 30
# excludes ~10% of real coupling mass, which may be enough on its own.
# Run anyway for a real, data-grounded confirm/refute rather than another
# guess.
#
# Config: local_field (SAME as 136/140/141/143: n_sites=32/
# local_channels=3/site_mix_radius=2/n_site_mix_layers=3/hidden=32) +
# masked_mlp_expand AS THE SOLE PRIMARY PROPAGATOR (attn_window=8,
# expand_factor=3 default -> effective receptive field 24) -- no
# pde_distill/pde_mutual/separate pde_head this time, matching Section
# 135's own no-pde-head design (isolating the propagator's own capacity
# to sustain chaos, not distillation quality). Section 98/141-style
# regularizers (w_var=0.01, w_spatial=0.06 signed, w_logdet=0.01,
# w_smooth=0.006). ONE variable changed from Section 135: attn_window
# 3 -> 8 (properly sized per the D3 measurement, not the previous guess).
#
# Verified via real smoke runs of both stages before this launch.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization,
# and Gate 3/4 suite as every section in this arc.
#
# QUEUED behind Section 143 to avoid GPU contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section144_localfield_maskedmlpexpand_w8_x3_200ep

echo "=== [1/5] Stage 1: local_field (n_sites=32/local_channels=3/site_mix_radius=2/n_site_mix_layers=3/hidden=32) + masked_mlp_expand (attn_window=8, expand_factor=3 -> effective receptive field 24, sized off Section 141's own D3 coupling measurement) AS THE SOLE PRIMARY PROPAGATOR, Section 98/141-style regularizers, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone masked_mlp_expand --mode markovian \
  --prop-attn-window 8 --masked-mlp-expand-factor 3 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.06 --spatial-signed --w-logdet 0.01 --w-smooth 0.006 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

_spectrum_check() {
  local PROP_PATH=$1
  local LABEL=$2
  mamba run -n da_env python -c "
import h5py, numpy as np, torch
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.diagnostics import propagator_step_jacobian_spectral_norms

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
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
print('[$LABEL] singular values >= 1.0:', int((sv >= 1.0).sum()), 'out of', sv.numel())
print('[$LABEL] per-step volume-change factor:', sv.prod().item())

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
_spectrum_check "$AUX" "stage1" > artifacts/logs/jacobiancheck_stage1_${TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage1_${TAG}.log

echo "=== [3/5] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, --amp, k_max=12, 300 epochs (established curriculum) ==="
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
_spectrum_check "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log

echo "=== [5/5] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) ==="
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

echo "=== Section 144 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
