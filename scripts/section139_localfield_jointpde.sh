#!/bin/zsh
# User-directed 2026-09-10: "we have code to jointly train a pde and a
# propagator at the same time right? I think it would be interesting to
# rerun 136 with this option on, then see if the pde we learn is actually
# predictive of the dynamics."
#
# Confirmed: this project already has two distinct joint-training
# mechanisms, separate from Section 138's after-the-fact frozen
# distillation:
#   1. Stage 1's `--pde-distill` (Sections 107-110): trains `pde_head`
#      (backbone="spectral_pde_raw") ALONGSIDE the real encoder+aux
#      propagator, in the SAME training loop. Default (detached target,
#      Stage1TrainingConfig.pde_distill_detach_target=True): gradient
#      reaches pde_head and, through z, the ENCODER -- the real `aux`
#      propagator itself is fully protected.
#   2. Stage 2's OWN pde_head continuation (train_stage2's docstring):
#      whenever `--init-pdehead-checkpoint` is given WITHOUT
#      `--freeze-propagator`, this is UNCONDITIONALLY mutual (the
#      propagator's own z_pred is NOT detached) -- gradient flows into
#      BOTH propagator and pde_head simultaneously.
#
# IMPORTANT, ALREADY-DOCUMENTED RISK (found in --pde-mutual's own CLI
# help text in train_stage1_patched.py, itself from an earlier round of
# this exact question): "this is the same 'pressure toward simplicity'
# mechanism behind this project's H-PROP fixed-point-collapse finding."
# Sections 120-134 (vit encoder, five different local-propagator
# mechanisms) and Section 135 (local_field encoder + masked_mlp_expand)
# all collapsed/diverged specifically because something pressured the
# propagator toward a simpler/local/describable form; Section 136's own
# success came from having a FULLY FREE propagator with NO such pressure.
# Continuing pde_head into Stage 2 WITHOUT freezing reintroduces exactly
# this category of risk, deliberately, to directly test the user's
# question.
#
# THIS RUN'S DESIGN, chosen to test the question as directly as possible
# while being honest about the risk:
#   Stage 1: EXACT Section 136 encoder/aux-backbone/regularizer recipe
#     (local_field n_sites=32/local_channels=3/site_mix_radius=2/
#     n_site_mix_layers=3/hidden=32, aux-backbone mlp, Section 98's
#     regularizers) PLUS --pde-distill --w-pde-distill 0.3
#     --pde-field-kind polynomial --pde-poly-degree 2 --pde-integrator
#     euler (SAFE/detached form -- genuinely joint/simultaneous training,
#     zero risk to the real propagator at this stage).
#   Stage 2: continues BOTH the propagator (normal, unfrozen,
#     Section-136-identical k12/300-epoch curriculum) AND pde_head
#     (--init-pdehead-checkpoint, no --freeze-propagator -- the
#     unconditionally-mutual, risky mode) at the SAME moderate weight
#     (0.3, deliberately not the stronger 1.0 used in Section 138's safe
#     frozen distillation, given the collapse-risk category this shares
#     with H-PROP's own finding).
#
# Verified via real (--profile full, small epoch count) dry runs of BOTH
# stages before this launch -- Stage 1's pde_distill loss decreases
# normally; Stage 2's val_kmax_mse genuinely changes epoch to epoch
# (confirming the propagator really is still being trained, unlike
# Section 138's frozen mode where it stays exactly constant).
#
# Same before/after Jacobian-spectrum+conditioning check, visualization,
# and Gate 3/4 suite as Sections 136/137 -- directly comparable numbers,
# and the DECISIVE check for whether mutual pde pressure collapsed this
# run the same way it did every masked-propagator attempt. PLUS the same
# k=1,2,4,8 pde_head-predictiveness evaluation Section 138 used, so "is
# the jointly-trained pde actually predictive" gets a real, quantitative
# answer directly comparable to 138's from-scratch-distilled baseline.
#
# QUEUED behind Section 138 (both its 136 and 137 distillations) to avoid
# GPU contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section139_localfield_p32c3_propmlp_jointpde_w03_200ep

echo "=== [1/6] Stage 1: Section 136's EXACT encoder/aux/regularizer recipe + --pde-distill (SAFE/detached, w_pde_distill=0.3, polynomial, euler), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.02 --w-var-floor 0 --w-spatial 0.04 --spatial-signed --w-logdet 0.008 --w-smooth 0.003 \
  --pde-distill --w-pde-distill 0.3 --pde-field-kind polynomial --pde-poly-degree 2 --pde-integrator euler \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/6] Stage 2: propagator AND pde_head continue together, UNFROZEN (mutual, the risky mode), w_pde_distill=0.3, --amp, k_max=12, 300 epochs (Section 136's own curriculum) ==="
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

echo "=== [3/6] FULL Jacobian spectrum check on the FINAL (jointly-pressured) propagator -- did mutual pde pressure collapse it? ==="
_spectrum_check > artifacts/logs/jacobiancheck_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${STAGE2_TAG}.log

echo "=== [4/6] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) -- directly comparable to Section 136's own numbers ==="
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

echo "=== [5/6] Is the JOINTLY-trained pde_head actually predictive? Same k=1,2,4,8 evaluation as Section 138 ==="
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
print('jointly-trained pde_head param count: %d' % n_params)
" > artifacts/logs/jointpde_eval_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jointpde_eval_${STAGE2_TAG}.log

echo "=== [6/6] Section 139 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
