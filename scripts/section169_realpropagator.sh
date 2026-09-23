#!/bin/zsh
# User-directed 2026-09-13/14. Clarifying a misunderstanding: the FIRST
# "Section 169" (scripts/section169_pdehead_as_propagator.sh) was a
# PURE EVALUATION exercise -- pointing the diagnostic scripts (run_
# analysis_suite.py/run_da_pff.py/visualize_rollout.py) at Section 167's
# ALREADY-TRAINED pde_head checkpoint, treating it as a drop-in
# propagator with NO retraining at all. That is NOT what the user meant:
# "why is there no stage 1 log for 169 then? I meant for the pde from
# 167 to be trained as a propagator in stage 1 and 2."
#
# THIS section is the actual intent: `--aux-backbone spectral_pde_raw`
# (matching Section 167's own pde_head field settings EXACTLY --
# field_kind=polynomial, poly_degree=2, poly_max_term_order=6,
# --spectral-poly-no-constant, --spectral-poly-exclude-nonconservative,
# spectral_K=49/spectral_L=96.0=d_latent, the same values Section 167's
# separate pde_head used) as THE PRIMARY propagator itself, trained via
# genuine Stage 1 (reconstruction + prediction loss, real k-curriculum,
# encoder co-adapts) and Stage 2 (k-curriculum rollout against real
# encoded sequences, k_max=12/300 epochs, the established curriculum) --
# NOT a secondary distilled pde_head sitting alongside a separately-
# trained masked_mlp_expand aux. `--aux-backbone spectral_pde_raw` as
# the PRIMARY propagator already has real precedent in this project
# (Section 142, "we could also try to train a pde as a propagator using
# this current encoder decoder setup right? that could work?") -- this
# section is that same mechanism, now combined with Section 167's own
# exact-conservation library fix.
#
# `--spectral-poly-no-constant`/`--spectral-poly-exclude-nonconservative`
# did not exist yet as CLI flags for the MAIN aux (only for the separate
# pde_head's own `--pde-poly-no-constant`/`--pde-poly-exclude-
# nonconservative`) -- added both to train_stage1_patched.py, wired into
# BOTH `aux_spectral_kwargs` construction sites (spectral_pde and
# spectral_pde_raw branches). Verified directly via unit test earlier
# (same underlying `_poly_weight()`/`_nonconservative_term_mask`
# mechanism, already validated for pde_head; this just exposes it for
# the main propagator too).
#
# NO --w-pde-energy-floor/--w-pde-mean-conservation here -- this is a
# deliberately clean, single-variable test of the library fix alone,
# trained under genuine reconstruction+rollout supervision (a
# fundamentally different, and stronger, training signal than Section
# 167's short-horizon real-data distillation) -- if this ALSO shows
# non-chaotic/decaying standalone dynamics, that would isolate the
# problem to the library/field form itself rather than the distillation
# training regime; if it doesn't, that's an important, different result.
#
# QUEUED behind Section 170 (still training on MPS at launch time, per
# user's explicit request: "please queue it after 170") -- this script
# is invoked by scripts/queue_section169_after_170.sh, which polls for
# Section 170's own train_stage1/train_stage2 processes to exit before
# launching this.
#
# Verified via a real (--profile full, --epochs 3 Stage 1 / --epochs 6
# Stage 2) dry run of this exact recipe -- no NaN, both stages train
# normally. Required --spectral-K 49 --spectral-L 96.0 explicitly
# (--spectral-K has no default; --spectral-L's own CLI default of 100.0
# is written for --encoder spectral_field's PHYSICAL domain length, not
# spectral_pde_raw's self-FFT ring convention on a non-spectral encoder
# -- would have been silently wrong, matching pde_head's own default of
# spectral_L=float(d_latent) is the correct choice here).
#
# Same before/after Jacobian-spectrum+conditioning check, visualization
# (including the smooth-field GIF via visualize_pde_smooth_field.py,
# pointed at this run's own propagator directly since there is no
# separate pde_head object), Gate 3/4 suite, standalone divergence
# check, and multi-IC self-spectrum mixing check as prior sections --
# directly comparable numbers, this time measuring the ACTUAL trained
# propagator's own standalone behavior (no more "as-propagator" caveat).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section169_realpropagator

echo "=== [1/6] Stage 1: local_field encoder + spectral_pde_raw AS THE PRIMARY PROPAGATOR (matching Section 167's own pde_head field settings: field_kind=polynomial, poly_degree=2, poly_max_term_order=6, --spectral-poly-no-constant, --spectral-poly-exclude-nonconservative, spectral_K=49/spectral_L=96.0), w_var=0.01/w_spatial=0.12/w_logdet=0.01/w_smooth=0.012/w_channel_mean=0.01, --amp, 200 epochs ==="
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
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/6] Stage 2: spectral_pde_raw propagator continues via genuine k-curriculum rollout against real encoded sequences, --amp, k_max=16 (Section 37's established k_max=16/300ep curriculum: k_mid=10 at epoch 175, k_max by epoch 210), 300 epochs -- user-directed 2026-09-13: 'can you change the phase 2 params to make k go up to 16' ==="
STAGE2_TAG="${TAG}_warmstart_k16_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
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

echo "=== [5/6] Standalone free-running divergence check (this IS the actual propagator now, no separate pde_head) ==="
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

echo "=== Section 169 (real propagator) complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
