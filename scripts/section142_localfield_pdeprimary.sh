#!/bin/zsh
# User-directed 2026-09-10: "we could also try to train a pde as a
# propagator using this current encoder decoder setup right? that could
# work?" -> "sure build this as 142."
#
# Every prior attempt at a literal, local/interpretable PDE as the SOLE
# PRIMARY propagator (Sections 99-127, on the older spectral_field/ViT
# encoders; Sections 128-134, various masked-propagator mechanisms on a
# vit encoder) collapsed or diverged. Sections 136-141 instead used a
# FREE `mlp` propagator (H-PROP's own established winner) and obtained an
# interpretable closure either by distilling it afterward (Section 138)
# or training it alongside with mutual gradient flow (Sections 139-141).
# This section asks the more literal version of the user's question
# directly: now that `local_field` supplies genuine architectural
# locality in the ENCODER, does a `spectral_pde_raw` closure work as the
# SOLE PRIMARY propagator (no separate free mlp at all)?
#
# Mechanistic prediction, stated before running (falsifiable): likely
# still collapses/diverges, for the SAME reason as every prior local-
# propagator attempt. `spectral_pde_raw`'s self-FFT step has genuine
# global reach, but the actual nonlinear map applied afterward (the
# polynomial body) is POINTWISE and weight-SHARED across every position --
# each output point's nonlinear term depends only on that same point's
# own local derivative stack, never on other points. That is exactly the
# kind of local, information-non-mixing nonlinearity H-PROP's own finding
# says collapses regardless of encoder. Since the propagator's internal
# structure doesn't change based on which encoder feeds it, this is
# expected to fail the same way Sections 99-127's own literal
# spectral_pde-primary attempts did. Run anyway because it is a real,
# previously-untested combination (local_field's genuine locality +
# poly_stable_leading's UV-stability guarantee + a term-order restriction,
# together, as PRIMARY -- never tried together before) and a clean
# confirm/refute either way is worth having.
#
# NEW: `--aux-backbone spectral_pde_raw` wired into
# scripts/train_stage1_patched.py as a genuine PRIMARY propagator choice
# (previously `spectral_pde_raw` was only ever built via `--pde-distill`,
# as a separate pde_head riding alongside a free aux propagator -- the
# underlying model class (`_SpectralPDERawDeltaBody`,
# `ks_latent.models.propagator`) already fully supported this; only the
# CLI's own `--aux-backbone` choices list and aux_spectral_kwargs
# construction were missing it). No spectral_N_w (no target physical
# field resolution to synthesize -- unlike "spectral_pde") and no
# spectral_physics_prior (raises if set -- there is no "true governing
# equation" for an arbitrary learned latent ordering the way there is for
# a genuine w=irfft(z) field). Verified via real smoke runs of both
# stages before this launch.
#
# Config: local_field (SAME as 136/140/141: n_sites=32/local_channels=3/
# site_mix_radius=2/n_site_mix_layers=3/hidden=32) + `--aux-backbone
# spectral_pde_raw` (spectral_K=49, spectral_L=96 matching d_latent's own
# ring convention -- see build_fresh_pdehead_checkpoint.py's identical
# choice for 136/140's own pde_head) + `--spectral-field-kind polynomial
# --spectral-poly-degree 2 --spectral-poly-max-term-order 6` (Section
# 141's own term-order restriction, addressing the w_xxxx*w_xxxx-driven
# blowup found checking 140's pde_head standalone) + `--spectral-poly-
# stable-leading` (the architectural UV-stability sign constraint, NEW to
# this specific combination -- never previously paired with local_field
# or used on a PRIMARY propagator's own training, only tested on the
# earlier spectral_field arc). Section 98/141-style regularizers
# (w_var=0.01, w_spatial=0.06 signed, w_logdet=0.01, w_smooth=0.006) --
# NO --pde-distill/--pde-mutual/separate pde_head at all this time: the
# spectral_pde_raw propagator built here IS the only propagator, full
# stop.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization,
# and Gate 3/4 suite as every section in this arc -- if it collapses, this
# confirms the diagnosis cleanly; if it doesn't, that is the single
# biggest result of the whole arc.
#
# QUEUED behind Section 141 to avoid GPU contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section142_localfield_pdeprimary_termorder6_stableleading_200ep

echo "=== [1/5] Stage 1: local_field encoder (n_sites=32/local_channels=3/site_mix_radius=2/n_site_mix_layers=3/hidden=32) + spectral_pde_raw AS THE SOLE PRIMARY PROPAGATOR (K=49, L=96, polynomial degree=2, max_term_order=6, poly_stable_leading), Section 98/141-style regularizers, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96 --spectral-max-order 4 --spectral-integrator euler \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 6 \
  --spectral-poly-stable-leading \
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

echo "=== [2/5] FULL Jacobian spectrum check on the STAGE-1 propagator (already the primary -- no Stage 2 fine-tune needed to ask 'did it collapse') ==="
_spectrum_check "$AUX" "stage1" > artifacts/logs/jacobiancheck_stage1_${TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage1_${TAG}.log

echo "=== [3/5] Stage 2: warm-started from Stage 1's own spectral_pde_raw aux, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [4/5] FULL Jacobian spectrum check on the FINAL STAGE-2 propagator ==="
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

echo "=== Section 142 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
