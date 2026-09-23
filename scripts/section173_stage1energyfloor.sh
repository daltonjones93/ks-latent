#!/bin/zsh
# User-directed 2026-09-14. Section 172 (--w-prop-energy-floor added to
# STAGE 2 ONLY, same Stage-1 checkpoint as 169/171, default 90-epoch
# ramp) collapsed to essentially the SAME fixed point as 169/171 -- D_KY=0,
# lambda1=-0.047 (even slightly worse than 171's -0.040), val_kmax_mse
# final 0.5916 (indistinguishable from 169's 0.591/171's 0.595). Diagnosed
# with the user why: `prop_energy` was still only 0.0012-0.0017 at epoch
# 99 (near the energy floor's OWN full ramp strength) while val_kmax_mse
# was ALREADY frozen at its final value -- the floor's own gradient
# pressure was negligible right when the collapse was forming (epoch
# ~11-20, the instant k reaches 3) and only grew to 0.0055 by epoch 299,
# far too late and far too weak to matter. User's own diagnosis, stated
# directly and confirmed by this data: "if it collapses, like it already
# has, the energy floor will do nothing" -- once the propagator settles
# into a stable dynamical fixed point, it's a genuine attractor, not just
# a loss-landscape minimum; ordinary gradient steps arriving after the
# fact have little reason to climb back out.
#
# THIS section applies the energy floor from the START instead of after
# the fact, on TWO fronts:
#   1. STAGE 1 (previously untouched by this mechanism -- Section 169's
#      Stage 1 never had --w-prop-energy-floor at all): retrained fresh
#      with --w-prop-energy-floor 0.5 --prop-energy-floor-gamma 0.4
#      --prop-energy-floor-rollout-k 30 active throughout, shaping the
#      propagator's own local map to be non-contractive from the start,
#      BEFORE Stage 2's k-curriculum ever exposes it to the collapse-
#      inducing multi-step objective. Default ~60-epoch ramp is fine here
#      -- Stage 1 has no k-curriculum of its own (k_pred stays at 2), so
#      there is no equivalent "epoch 11" urgency to beat.
#   2. STAGE 2: SAME mechanism, but with --prop-energy-floor-warmup-epochs
#      10 (NOT the default ~90) -- reaches full k=30 strength by epoch 10,
#      i.e. BEFORE k_now itself first reaches 3 (epoch 11, per every prior
#      run's own timing) -- directly targeting the timing gap just
#      diagnosed, giving the floor a chance to resist the collapse as it
#      forms rather than fighting an already-established attractor.
#
# Verified via a real (--profile full, --epochs 3-6) dry run of both
# stages with this exact recipe (including the fast 10-epoch Stage-2
# ramp) earlier this session -- no NaN, the finite-value guard fires and
# recovers correctly under aggressive/early full-strength rollout
# stress, same as previously validated.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization,
# Gate 3/4 suite, standalone divergence check, and multi-IC self-spectrum
# mixing check as Sections 169/171/172 -- directly comparable numbers.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section173_stage1energyfloor

echo "=== [1/6] Stage 1: local_field encoder + spectral_pde_raw AS THE PRIMARY PROPAGATOR (Section 167/169's exact field settings) + NEW --w-prop-energy-floor 0.5 --prop-energy-floor-gamma 0.4 --prop-energy-floor-rollout-k 30 ACTIVE FROM THE START (shapes the map to be non-contractive before Stage 2 ever sees it), w_var=0.01/w_spatial=0.12/w_logdet=0.01/w_smooth=0.012/w_channel_mean=0.01, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 \
  --spectral-poly-max-term-order 6 --spectral-integrator euler \
  --spectral-poly-no-constant --spectral-poly-exclude-nonconservative \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --w-prop-energy-floor 0.5 --prop-energy-floor-gamma 0.4 --prop-energy-floor-rollout-k 30 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/6] Stage 2: spectral_pde_raw propagator continues, SAME --w-prop-energy-floor mechanism but with a FAST --prop-energy-floor-warmup-epochs 10 (reaches full strength BEFORE k_now first reaches 3 at epoch ~11, per every prior run's own timing -- directly targets the diagnosed timing gap), k_max=16 (Section 37's curriculum), 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k16_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --w-prop-energy-floor 0.5 --prop-energy-floor-gamma 0.4 --prop-energy-floor-rollout-k 30 \
  --prop-energy-floor-warmup-epochs 10 \
  --amp \
  --epochs 300 --k-max 16 --k-warmup-epochs 210 --k-mid 10 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

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

echo "=== [3/6] FULL Jacobian spectrum + conditioning check on the FINAL propagator -- does early-engagement energy floor actually prevent collapse? ==="
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

echo "=== [5/6] Standalone free-running divergence check ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
ae.eval(); prop.eval()
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][0], dtype=torch.float32)
u0 = traj[0].unsqueeze(0)
with torch.no_grad():
    z0 = ae.encode(u0)
    z_roll = prop.rollout(z0, z0, 200)
finite = torch.isfinite(z_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('first non-finite step (out of 200, -1 = never):', first_bad)
for t in [0,10,20,30,40,60,80,100,150,199]:
    v = z_roll[0, t]
    print(t, v.abs().max().item() if torch.isfinite(v).all() else 'nan/inf')
" > artifacts/logs/standalone_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/standalone_${STAGE2_TAG}.log

echo "=== [6/6] Self-spectrum mixing check across the standalone rollout, ACROSS MULTIPLE real ICs ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.spectral_field import encode_to_spectrum

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
ae.eval(); prop.eval()
K = prop_cfg.spectral_K
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    trajs = torch.tensor(f['trajectories'][:5], dtype=torch.float32)
for run_idx in range(5):
    for start in [0, 100]:
        u0 = trajs[run_idx, start].unsqueeze(0)
        with torch.no_grad():
            z0 = ae.encode(u0)
            z_roll = prop.rollout(z0, z0, 55).squeeze(0)
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
" > artifacts/logs/mixing_check_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/mixing_check_${STAGE2_TAG}.log

echo "=== Section 173 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
