#!/bin/zsh
# User-directed 2026-09-13. Section 167 (--pde-poly-exclude-
# nonconservative, architecturally zeros every even-combined-order two-
# factor term) fixed the mode-0 drift cleanly: no divergence, no
# mode-0 lock-in across any of 10 mixing-check ICs, D_KY=22.67 (as the
# PROPAGATOR's own diagnostic target, not pde_head's). But when pde_head
# itself was tested AS the propagator directly (Section 169, no
# retraining -- just pointing run_analysis_suite.py/run_da_pff.py at
# pde_head's own checkpoint), the result was decisively negative: DA
# skill 0.96x (worse than doing nothing) and pde_head's OWN Lyapunov
# spectrum gave D_KY=0, n_positive=0, lambda1=-0.006 (NEGATIVE) --
# pde_head's free-running dynamics converge to a stable FIXED POINT over
# a long enough horizon, not genuine chaos.
#
# User spotted the visual signature directly in 167's own smooth-field
# GIF: "I also see the decay in the pde trajectory towards 0... would it
# be worth trying to fix the maximum energy in the pde in physical
# space?... force the model to always have an 'energetic' component."
# This is a REAL, distinct mechanism from the mode-0 fix -- the closure
# no longer locks onto an artifact mode, but it IS net-dissipative over
# long rollouts, because nothing in the existing training terms directly
# supervises energy at horizons beyond the short (k<=4-30) windows those
# terms actually see.
#
# Implemented `spatial_energy_floor_loss` (ks_latent/training/losses.py)
# -- a PER-SAMPLE hinge floor on std(z, across the ring index) (a
# DIFFERENT axis/failure-mode from `w_var`'s per-channel-across-batch
# floor or `w_varmatch`'s per-channel-across-different-ICs floor: this
# targets a SINGLE trajectory's own spatial profile decaying toward
# uniform over time). `w_pde_energy_floor` applies this to pde_head's
# OWN unsupervised autoregressive rollout (no real target needed beyond
# the starting state -- NOT constrained by the training window size, can
# run far longer than the distillation rollout's own k=4). `gamma=0.4`
# (real z's own spatial std averages ~0.97, minimum ~0.53). Uses the
# SAME curriculum-ramp safety mechanism as `w_pde_distill_real_rollout`
# (k: 1 -> pde_energy_floor_rollout_k over pde_energy_floor_warmup_
# epochs) -- this is ALSO a genuine multi-step autoregressive rollout
# through pde_head's own dynamics, so the Section 159 postmortem's
# lesson applies identically.
#
# ONLY CHANGE from Section 167: adds --w-pde-energy-floor 0.5
# --pde-energy-floor-gamma 0.4 --pde-energy-floor-rollout-k 30 to BOTH
# stages. Everything else (channel-mean fix, --pde-poly-no-constant,
# --pde-poly-exclude-nonconservative, mutual real-data-only pde_head
# fitting, propagator fully decoupled, doubled --w-spatial/--w-smooth,
# iLED nonlinear-l2+rollout in Stage 2) UNCHANGED from 167.
#
# Verified via: (1) direct unit test of spatial_energy_floor_loss (zero
# penalty when already above gamma, real penalty below it); (2) a real
# (--profile full, --epochs 3-6) dry run of Stage 1 with this exact
# recipe -- no NaN, pde_energy prints and decreases sensibly, ramp
# confirmed active. Stage 2's own dry run was started but killed
# early -- it began contending with Section 168's still-running MPS
# training (a real mistake, caught and fixed immediately by killing the
# dry run, not 168) -- not re-run given the mechanism's identical ramp-
# safety precedent already validated for `w_pde_distill_real_rollout`
# and the loss function's own already-verified math.
#
# This section's whole POINT is to test whether the energy floor
# actually restores genuine chaos in pde_head's own standalone Lyapunov
# spectrum -- Section 169's methodology (pointing run_analysis_suite.py/
# run_da_pff.py directly at pde_head's checkpoint, no retraining needed)
# will be re-run against THIS section's own pde_head once training
# finishes, to check D_KY/lambda1/DA-skill directly.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization
# (including the smooth-field GIF), Gate 3/4 suite, k=1,2,4,8
# pde-predictiveness evaluation, standalone pde_head divergence check,
# and the multi-IC self-spectrum mixing check as Section 167 --
# directly comparable numbers.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section170_energyfloor

echo "=== [1/8] Stage 1: Section 167's recipe + NEW --w-pde-energy-floor 0.5 --pde-energy-floor-gamma 0.4 --pde-energy-floor-rollout-k 30 (per-sample spatial-energy floor on pde_head's own unsupervised rollout): local_field + masked_mlp_expand (attn_window=7, expand_factor=3), --pde-poly-no-constant, --pde-poly-exclude-nonconservative, --w-spatial 0.12 --w-smooth 0.012, pde_head fit SOLELY against real encoder-derived transitions, MUTUAL (--w-pde-distill-real 0.5 --pde-distill-real-mutual -- gradient reaches the encoder; NO --w-pde-distill/--pde-mutual -- propagator never seen) (NO --pde-K -- full spectrum, K=d_latent//2+1=49) --pde-poly-max-term-order 6, w_var=0.01/w_logdet=0.01, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone masked_mlp_expand --mode markovian \
  --prop-attn-window 7 --masked-mlp-expand-factor 3 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --pde-distill --pde-field-kind polynomial --pde-poly-degree 2 \
  --pde-poly-max-term-order 6 --pde-integrator euler --pde-poly-no-constant \
  --pde-poly-exclude-nonconservative \
  --w-pde-distill-real 0.5 --pde-distill-real-mutual \
  --w-pde-energy-floor 0.5 --pde-energy-floor-gamma 0.4 --pde-energy-floor-rollout-k 30 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/8] Stage 2: propagator trains independently (pde_head's loss no longer touches it at all); pde_head continues fitting SOLELY against real data: --w-pde-distill-real 0.5 (single-step) + --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 4 (multi-step, iLED L_forecast) + --w-pde-nonlinear-l2 0.01 (iLED L_non-linearity) + NEW --w-pde-energy-floor 0.5 --pde-energy-floor-gamma 0.4 --pde-energy-floor-rollout-k 30, NO --w-pde-distill, grad_clip active, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill-real 0.5 \
  --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 4 \
  --w-pde-nonlinear-l2 0.01 \
  --w-pde-energy-floor 0.5 --pde-energy-floor-gamma 0.4 --pde-energy-floor-rollout-k 30 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
STAGE2_PDEHEAD=artifacts/stage2_pdehead_patched_full_${STAGE2_TAG}.pt

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
print('[$LABEL] max|z| at final step:', traj_roll[:, -1, :].abs().max().item())

z_np = z_all.reshape(-1, z_all.shape[-1]).numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('[$LABEL] spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
"
}

echo "=== [3/8] FULL Jacobian spectrum + conditioning check on the FINAL propagator ==="
_spectrum_check "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${STAGE2_TAG}.log

echo "=== [4/8] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --rollout-steps 200 --tag "$STAGE2_TAG" \
  > artifacts/logs/visualize_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/visualize_${STAGE2_TAG}.log

mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1

echo "=== [5/8] Is the pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138-141/143/145/153 ==="
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
with torch.no_grad():
    z_prop = prop.rollout(z0, z0, k_max)
    z_pde = pdehead.rollout(z0, z0, k_max)

for k in (1, 2, 4, 8):
    mse_vs_prop = ((z_pde[:, :k] - z_prop[:, :k])**2).mean().item()
    r2_vs_prop = 1 - mse_vs_prop / z_prop[:, :k].var().item()
    print('k=%d: pde_vs_prop_mse=%.5f (R2=%.4f)' % (k, mse_vs_prop, r2_vs_prop))
n_params = sum(p.numel() for p in pdehead.parameters())
print('param count: %d' % n_params)
" > artifacts/logs/jointpde_eval_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jointpde_eval_${STAGE2_TAG}.log

echo "=== [6/8] Standalone pde_head free-running divergence check ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); pdehead.eval()
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][0], dtype=torch.float32)
u0 = traj[0].unsqueeze(0)
with torch.no_grad():
    z0 = ae.encode(u0)
    z_roll = pdehead.rollout(z0, z0, 200)
finite = torch.isfinite(z_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('first non-finite step (out of 200, -1 = never):', first_bad)
for t in [0,10,20,30,40,60,80,100,150,199]:
    v = z_roll[0, t]
    print(t, v.abs().max().item() if torch.isfinite(v).all() else 'nan/inf')
" > artifacts/logs/pdehead_standalone_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/pdehead_standalone_${STAGE2_TAG}.log

echo "=== visualize_pde_smooth_field (smooth Hovmoller + GIF) ==="
mamba run -n da_env python scripts/visualize_pde_smooth_field.py \
  --ae-checkpoint "$AE" --pdehead-checkpoint "$STAGE2_PDEHEAD" \
  --rollout-steps 55 --tag "$STAGE2_TAG" \
  > artifacts/logs/pde_smooth_field_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/pde_smooth_field_${STAGE2_TAG}.log

echo "=== [7/8] pde_head's own self-spectrum mixing check across its standalone rollout, ACROSS MULTIPLE real ICs -- directly measures whether the mode-0 drift is reduced vs. Section 164 ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.spectral_field import encode_to_spectrum

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); pdehead.eval()
K = pdehead_cfg.spectral_K
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    trajs = torch.tensor(f['trajectories'][:5], dtype=torch.float32)
for run_idx in range(5):
    for start in [0, 100]:
        u0 = trajs[run_idx, start].unsqueeze(0)
        with torch.no_grad():
            z0 = ae.encode(u0)
            z_roll = pdehead.rollout(z0, z0, 55).squeeze(0)
        if not torch.isfinite(z_roll).all():
            print('run=%d start=%d: DIVERGED' % (run_idx, start)); continue
        z_hat0 = encode_to_spectrum(z_roll[0:1], K).squeeze(0)
        z_hat54 = encode_to_spectrum(z_roll[54:55], K).squeeze(0)
        e0 = (z_hat0[:K]**2+z_hat0[K:2*K]**2); e54 = (z_hat54[:K]**2+z_hat54[K:2*K]**2)
        top0 = torch.topk(e0,3); top54 = torch.topk(e54,3)
        print('run=%d start=%3d: t0 top3=%s (%.3f) -> t54 top3=%s (%.3f)' % (
            run_idx, start,
            top0.indices.tolist(), (top0.values.sum()/e0.sum()).item(),
            top54.indices.tolist(), (top54.values.sum()/e54.sum()).item(),
        ))
" > artifacts/logs/pdehead_mixing_check_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/pdehead_mixing_check_${STAGE2_TAG}.log

echo "=== [8/8] pde_head AS THE PROPAGATOR directly (Section 169's methodology, no retraining) -- does the energy floor restore genuine chaos? ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PDEHEAD" \
  --tag "${STAGE2_TAG}_pdehead_as_propagator" > artifacts/logs/gate3_analysis_${STAGE2_TAG}_pdehead_as_propagator.log 2>&1
cat artifacts/logs/gate3_analysis_${STAGE2_TAG}_pdehead_as_propagator.log
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PDEHEAD" \
  --tag "${STAGE2_TAG}_pdehead_as_propagator" > artifacts/logs/gate3_da_${STAGE2_TAG}_pdehead_as_propagator.log 2>&1
cat artifacts/logs/gate3_da_${STAGE2_TAG}_pdehead_as_propagator.log

echo "=== Section 170 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
echo "--- pde_head AS PROPAGATOR (the actual question this section asks) ---"
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}_pdehead_as_propagator.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}_pdehead_as_propagator.log || true
