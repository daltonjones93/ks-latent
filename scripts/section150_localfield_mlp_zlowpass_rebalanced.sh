#!/bin/zsh
# User-directed 2026-09-11, following Section 149 (local_field + FREE mlp
# propagator + NEW latent_self_spectrum_lowpass_loss regularizer,
# w_z_lowpass=1e-5 in Stage 1 on the encoder's real z, w_z_lowpass_rollout
# =1e-4 in BOTH Stage 1 (aux rollout) and Stage 2 (propagator rollout,
# 300 epochs) on z_pred).
#
# 149's own numbers were WORSE, not better, than Section 148's baseline:
# D_KY 21.30 (low end of the 21-24 target, vs 148's 23.08), n_positive
# DOWN to 10 (vs 12), DA skill DOWN to 1.16x (vs 148's 1.31x), and
# critically pde_head R^2@k=8 COLLAPSED to 0.504 (vs 148's 0.975) --
# despite the standalone pde_head "never diverging" over 200 steps, its
# own free rollout |z| DECAYS monotonically to a near-fixed-point
# (1.90->1.13->0.89->0.65->0.56->0.49->settles ~0.466), i.e. it achieved
# "stability" by becoming a damped, non-chaotic system, not by learning a
# genuinely stable closure of the real dynamics.
#
# Root cause, diagnosed directly (user's own observation, then confirmed
# by measurement on 149's checkpoint): w_z_lowpass_rollout (weight 1e-4,
# applied to z_pred) got 10x the weight AND ran for the FULL Stage 1 (200
# ep) + Stage 2 (300 ep) duration, while w_z_lowpass (weight 1e-5, applied
# to the TRUE encoded z) only ran during Stage 1's 200 epochs -- Stage 2
# freezes the encoder, so there is structurally no way to keep pressuring
# the true z once Stage 1 ends. Measured directly: real z's self-spectrum
# lowpass loss = 1874.3 vs. the propagator's own rollout z_pred = 698.4 --
# a 2.68x gap. The propagator was pressured into systematically smoother/
# more-damped dynamics than the real system it's supposed to track, not
# because the true dynamics got easier to fit but because of this
# asymmetric regularization schedule. User: "yeah let's do 150 so that
# w_z_lowpass is raised in stage 1 (I think 8e-5), and then eliminate
# w_z_lowpass_rollout in stage 2."
#
# TWO changes from Section 149:
#
# 1. --w-z-lowpass 1e-5 -> 8e-5 (Stage 1, on the encoder's real z) -- 8x
#    stronger, intended to bring the TRUE z's own smoothness pressure up
#    to a comparable order of magnitude to what w_z_lowpass_rollout=1e-4
#    was already applying to z_pred, so there's no longer a large gap for
#    Stage 2 to widen further once the encoder freezes.
#
# 2. Stage 2's --w-z-lowpass-rollout REMOVED entirely (reverts to the
#    default 0.0, off) -- no low-pass pressure at all during Stage 2's
#    300-epoch continuation, removing the single biggest source of the
#    propagator/true-z asymmetry (it ran longest and had gone
#    unopposed).
#
# Stage 1's own --w-z-lowpass-rollout (1e-4, on the aux propagator's SHORT
# rollout) is UNCHANGED -- the user's instruction targeted Stage 1's
# --w-z-lowpass and Stage 2's --w-z-lowpass-rollout specifically, not
# Stage 1's rollout term.
#
# Otherwise IDENTICAL to Section 149: local_field (n_sites=32/
# local_channels=3/site_mix_radius=2/n_site_mix_layers=3/hidden=32) + FREE
# mlp propagator, --pde-distill --pde-mutual --pde-field-kind polynomial
# --pde-poly-degree 2 --pde-poly-max-term-order 6 --pde-integrator euler
# (NO --pde-K -- full spectrum), w_var=0.01/w_spatial=0.06 signed/
# w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, --full-propagator --amp,
# 200 epochs; Stage 2 unfrozen mutual continuation, w_pde_distill=0.5,
# k_max=12, 300 epochs (established curriculum).
#
# Verified via a real (--profile full, 2-epoch) dry run through BOTH
# stages before this launch -- --w-z-lowpass 8e-5 accepted, no
# --w-z-lowpass-rollout in the Stage 2 call, losses compute and decrease
# normally, no errors.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization
# (including the smooth-field GIF), Gate 3/4 suite, k=1,2,4,8
# pde-predictiveness evaluation, and standalone pde_head divergence check
# as Sections 141/143/145/148/149 -- directly comparable numbers.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section150_localfield_mlp_zlowpass_rebalanced_200ep

echo "=== [1/6] Stage 1: local_field + FREE mlp propagator (full attention) + --pde-distill --pde-mutual (NO --pde-K -- full spectrum, K=d_latent//2+1=49) --pde-poly-max-term-order 6, w_var=0.01/w_spatial=0.06/w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, REBALANCED w_z_lowpass=8e-5 (raised from 1e-5)/w_z_lowpass_rollout=1e-4 (unchanged), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.06 --spatial-signed --w-logdet 0.01 --w-smooth 0.006 \
  --pde-distill --w-pde-distill 0.5 --pde-mutual --pde-field-kind polynomial --pde-poly-degree 2 \
  --pde-poly-max-term-order 6 --pde-integrator euler \
  --w-z-lowpass 8e-5 --w-z-lowpass-rollout 1e-4 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/6] Stage 2: propagator AND pde_head continue together, UNFROZEN (mutual), w_pde_distill=0.5, NO w_z_lowpass_rollout (removed -- was 1e-4 in Section 149), --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill 0.5 \
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

echo "=== [3/6] FULL Jacobian spectrum + conditioning check on the FINAL propagator ==="
_spectrum_check "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${STAGE2_TAG}.log

echo "=== [4/6] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) ==="
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

echo "=== [5/6] Is the pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138-141/143/145 ==="
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

echo "=== [6/6] Standalone pde_head free-running divergence check ==="
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

echo "=== Section 150 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
