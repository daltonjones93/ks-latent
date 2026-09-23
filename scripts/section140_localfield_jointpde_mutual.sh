#!/bin/zsh
# User-directed 2026-09-10: "I would also like to try the joint training
# (non detached version) of the pde distill after this."
#
# Section 139 tested Stage 1's SAFE/detached --pde-distill (the real
# propagator protected) plus Stage 2's own continuation, which is
# UNCONDITIONALLY mutual whenever pde_head continues without
# --freeze-propagator. This section goes one step further: Stage 1 ALSO
# uses `--pde-mutual` (Stage1TrainingConfig.pde_distill_detach_target=
# False), so the distillation loss's gradient reaches the REAL aux
# propagator directly from epoch 0 of Stage 1 onward -- mutual pressure
# through the ENTIRE pipeline (both stages), not just Stage 2's
# continuation. This is the single most aggressive version of "jointly
# train a pde and a propagator at the same time" this project's own CLI
# supports.
#
# Same already-documented risk as Section 139, now applied from the very
# start: `--pde-mutual`'s own CLI help text (in train_stage1_patched.py,
# itself from an earlier round of this same question): "this is the same
# 'pressure toward simplicity' mechanism behind this project's H-PROP
# fixed-point-collapse finding." If Section 139 (Stage 2-only mutual)
# already showed signs of degrading toward collapse, expect this to be at
# least as bad, likely worse, given the pressure now starts a full 200
# epochs earlier and acts on the encoder+aux jointly rather than only the
# already-Stage-1-converged aux.
#
# ONE variable changed from Section 139: --pde-distill --pde-mutual added
# to Stage 1 (weight, field_kind, integrator, regularizers, curriculum,
# Stage-2 command all IDENTICAL to 139, so this is a clean, isolated
# Stage-1-detached-vs-mutual comparison, on top of 139's own
# Stage-2-mutual-vs-138's-frozen comparison).
#
# Verified via a real (--profile full, 2-epoch) dry run of Stage 1 before
# this launch -- pde_distill loss decreases normally with --pde-mutual
# active, no errors.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization,
# Gate 3/4 suite, and k=1,2,4,8 pde-predictiveness evaluation as Section
# 139 -- directly comparable numbers across 136 (free, no pde pressure)
# -> 138 (frozen distillation, zero risk) -> 139 (Stage-2-only mutual)
# -> 140 (fully mutual, this run).
#
# QUEUED behind Section 139.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section140_localfield_p32c3_propmlp_jointpde_mutual_w03_200ep

echo "=== [1/6] Stage 1: Section 136's EXACT encoder/aux/regularizer recipe + --pde-distill --pde-mutual (FULLY MUTUAL from epoch 0, w_pde_distill=0.3, polynomial, euler), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.02 --w-var-floor 0 --w-spatial 0.04 --spatial-signed --w-logdet 0.008 --w-smooth 0.003 \
  --pde-distill --w-pde-distill 0.3 --pde-mutual --pde-field-kind polynomial --pde-poly-degree 2 --pde-integrator euler \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/6] Stage 2: propagator AND pde_head continue together, UNFROZEN (mutual), w_pde_distill=0.3, --amp, k_max=12, 300 epochs (Section 136/139's own curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill 0.3 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
STAGE2_PDEHEAD=artifacts/stage2_pdehead_patched_full_${STAGE2_TAG}.pt

_spectrum_check() {
  mamba run -n da_env python -c "
import h5py, numpy as np, torch
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.diagnostics import propagator_step_jacobian_spectral_norms

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
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
print('singular values >= 1.0:', int((sv >= 1.0).sum()), 'out of', sv.numel())
print('per-step volume-change factor:', sv.prod().item())

z0b = z_all[:, 0, :]
with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=60)
sep_start = (traj_roll[:, 5, :] - traj_roll[:, 0, :]).norm(dim=-1).mean().item()
sep_end = (traj_roll[:, -1, :] - traj_roll[:, -6, :]).norm(dim=-1).mean().item()
cross_sample_std_end = traj_roll[:, -1, :].std(dim=0).mean().item()
print('rollout separation steps 0-5: %.4f   steps 54-59: %.4f   cross-sample z.std final step: %.4f' % (sep_start, sep_end, cross_sample_std_end))
"
}

echo "=== [3/6] FULL Jacobian spectrum check on the FINAL (fully mutually-pressured) propagator -- did it collapse? ==="
_spectrum_check > artifacts/logs/jacobiancheck_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${STAGE2_TAG}.log

echo "=== [4/6] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) -- directly comparable to Sections 136/139's own numbers ==="
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

echo "=== [5/6] Is the FULLY-mutually-trained pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138/139 ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); prop.eval(); pdehead.eval()

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
with torch.no_grad():
    z_all = ae.encode(traj.reshape(n*T, NX)).reshape(n, T, -1)

k_max = 8
z0 = z_all[:, 0, :]
z_true = z_all[:, 1:k_max+1, :]
with torch.no_grad():
    z_prop = prop.rollout(z0, z0, k_max)
    z_pde = pdehead.rollout(z0, z0, k_max)

for k in (1, 2, 4, 8):
    mse_vs_prop = ((z_pde[:, :k] - z_prop[:, :k])**2).mean().item()
    mse_vs_truth = ((z_pde[:, :k] - z_true[:, :k])**2).mean().item()
    prop_mse_vs_truth = ((z_prop[:, :k] - z_true[:, :k])**2).mean().item()
    r2_vs_prop = 1 - mse_vs_prop / z_prop[:, :k].var().item()
    print('k=%d: pde_vs_prop_mse=%.5f (R2=%.4f)   pde_vs_truth_mse=%.5f   prop_vs_truth_mse=%.5f (own propagator error, for reference)' % (
        k, mse_vs_prop, r2_vs_prop, mse_vs_truth, prop_mse_vs_truth
    ))
n_params = sum(p.numel() for p in pdehead.parameters())
print('fully-mutually-trained pde_head param count: %d' % n_params)
" > artifacts/logs/jointpde_eval_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jointpde_eval_${STAGE2_TAG}.log

echo "=== [6/6] Section 140 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
