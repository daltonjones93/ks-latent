#!/bin/zsh
# User-directed 2026-09-10: "why don't we try Phase 10's original design
# called for an architecturally-enforced local field (circular-Conv1d,
# channel 0 anchored to a real physical average)."
#
# Context: masked_mlp_expand (Sections 128-134) kept collapsing regardless
# of how its Jacobian spectrum was regularized (flat two-group floor,
# retargeted flat floor, per-rank graded floor -- none held under Stage
# 2's own longer-horizon pressure, or only partially). When asked whether
# the propagator's receptive field (radius 9 out of d_latent=44) was
# fundamentally too narrow, direct evidence pointed the OTHER way: Phase
# 2's own already-measured light-cone bound (v_star=1.261+-0.039 physical
# units/time, docs/RESULTS.md) predicts KS's own true information-
# propagation needs LESS than one site's worth of receptive field at
# dt_snap=1.0 -- radius 9 should be overkill, IF the flat latent index
# corresponded to physical position at all. It doesn't: nothing in the
# vit encoder architecture forces that. This section instead builds
# locality in from the start.
#
# NEW: `--encoder local_field` (KSAutoencoderLocalField,
# ks_latent/models/autoencoder_local_field.py; LocalFieldAutoencoderConfig,
# ks_latent/config.py) -- Phase 10's original design (CLAUDE.md sections
# 12.2.1/12.3), built for the first time this project has ever run it:
#   - Encoder: exact non-overlapping patchify (Conv1d, kernel=stride=
#     patch_size) -> n_site_mix_layers circular Conv1d(kernel=2*radius+1,
#     circular padding) over the SITE axis -> 1x1 conv to the LEARNED
#     residual channels.
#   - GAUGE ANCHOR (not optional per the brief): channel 0 of the latent
#     is NEVER learned -- it is fixed exactly to
#     u.reshape(B,n_sites,patch_size).mean(-1), the site's own local
#     physical average.
#   - Decoder: exact mirror, ending in a shape-matched ConvTranspose1d.
#   - Latent flattened SITE-MAJOR (index = site*local_channels + channel),
#     not channel-major -- so a flat-index window (e.g. masked_mlp_expand's
#     own attn_window) sees a genuine physical neighborhood of sites (all
#     channels included), matching how KS's nonlinear term actually
#     couples multiple state variables AT THE SAME site.
# n_sites=32/local_channels=3 (CLAUDE.md's own "starting point"),
# site_mix_radius=2/n_site_mix_layers=3 gives an encoder receptive field of
# patch_size*(1+n_layers*radius) = 8*7 = 56 grid points = 21.9 physical
# units at L=100 -- inside the brief's own "2-3 correlation lengths ~20-30
# physical units" target (KS cell scale 2*sqrt(2)*pi~=8.9), and far wider
# than the 1.26-physical-unit light-cone bound the PROPAGATOR itself needs.
# 9 new unit tests (exact equivariance under shifts by multiples of
# patch_size, receptive-field measured by gradient masking, the anchored
# channel proven fixed/init-independent, the local_channels=1 degenerate
# case, overfit/reconstructed-spectrum sanity checks, config validation);
# full suite re-run for regressions; both stages smoke-tested end to end.
#
# Propagator: masked_mlp_expand, attn_window=3/expand_factor=3 -- the EXACT
# SAME numeric setting as Sections 128-134, deliberately UNCHANGED, so this
# run isolates exactly one variable (the encoder). What that same number
# now MEANS is different: on this site-major physically-ordered latent, a
# +-3 window at the base layer is a genuine few-site physical neighborhood
# (not an arbitrary slice of an unordered index), and the middle
# (expand_factor=3) layer's own derived radius (attn_window*expand_factor
# = 9, per _MaskedMLPExpandDeltaBody's own math) spans roughly 9/local_
# channels = 3 physical sites -- comfortably wider than the light-cone
# bound, matching Phase 11's own planned stencil-width sweep range {1,2,3,4}.
#
# Regularizers: DELIBERATELY MINIMAL this time (w_var=0.02, w_logdet=0.008
# only -- this project's own cheap, well-established, non-controversial
# anti-collapse insurance, used everywhere; NO w_spectrum_shape/
# w_spectrum_shape_graded). This is the whole point of Phase 10's design:
# if genuine architectural locality + the gauge anchor are what's actually
# needed, they should reduce or eliminate the need for heavy Jacobian-
# shaping regularizer engineering, not just add another layer of it. If
# this still collapses, the graded spectrum-shape mechanism (Section 134)
# can be layered on top as a clean, well-isolated follow-up.
#
# Everything else matches the established curriculum: --full-propagator,
# 200 Stage-1 epochs + 300 Stage-2 epochs at the Section 52/98/100/128-134
# k12 curriculum. Same before/after Jacobian-spectrum check as Sections
# 132-134 (Stage-1 propagator and the final Stage-2 propagator).
#
# QUEUED behind Section 134 (scripts/queue_section135_after_134.sh) to
# avoid GPU contention -- same practice as Section 98 queuing behind 97.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section135_localfield_p32c3_propmaskedmlpexpand_w3_x3_200ep

echo "=== [1/4] Stage 1: NEW local_field encoder+decoder (n_sites=32, local_channels=3, site_mix_radius=2, n_site_mix_layers=3, hidden=32 -- Phase 10's own starting point, gauge-anchored channel 0) + masked_mlp_expand propagator (attn_window=3, expand_factor=3, UNCHANGED from 128-134), MINIMAL regularizers (w_var=0.02, w_logdet=0.008, NO spectrum-shape terms), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone masked_mlp_expand --mode markovian \
  --prop-attn-window 3 --masked-mlp-expand-factor 3 \
  --w-decorr 0 --w-var 0.02 --w-var-floor 0 --w-logdet 0.008 \
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
from ks_latent.analysis.diagnostics import propagator_step_jacobian_spectral_norms, same_time_coupling_diagnostic_signed

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

echo "=== [3/4] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, --amp, k_max=12, 300 epochs (Section 52/98/100/128-134's shared curriculum) ==="
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

echo "=== Section 135 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
