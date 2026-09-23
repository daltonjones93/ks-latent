#!/bin/zsh
# User-directed 2026-09-14. Sections 169/171/172/173 (baseline, +varmatch,
# +late w_prop_energy_floor, +early/fast w_prop_energy_floor) ALL FOUR
# converged to the essentially IDENTICAL collapsed fixed point (D_KY=0,
# lambda1 in [-0.049, -0.040], standalone rollout decaying to ~1e-4 by
# step 200) regardless of the anti-collapse mechanism or its timing --
# strong evidence the problem isn't a regularization-weight or timing
# issue at all, but the unconstrained polynomial term library itself:
# gradient descent always finds the same cheap contractive shortcut
# because nothing stops it from finding one.
#
# User proposal: parametrize the propagator as a KS-shaped template
# (u_t = -u*u_x + nu*u_xx + forcing) with nu LEARNABLE but slow-moving,
# arguing this guarantees bounded dynamics by construction. Feedback given
# and agreed: the literal proposed form (2nd-order term + fixed spatial
# forcing, no 4th-order term) would NOT actually be bounded-and-chaotic --
# it's missing the high-wavenumber damping mechanism that's the actual
# reason real KS stays bounded (Re(lambda(k)) = c2*(-k^2) + c4*(k^4) needs
# c4<0 to dominate as k->inf; a fixed spatial forcing term can't provide
# state-dependent high-k damping). Root-cause diagnosis: Sections 169-173
# NEVER constrained w_xx's (order=2) coefficient sign at all -- nothing
# stopped it from drifting positive, which makes Re(lambda(k)) negative at
# EVERY wavenumber (globally damped, not just high-k-damped) -- almost
# certainly the actual collapse mechanism measured identically four times.
#
# Implemented `poly_stable_linear_terms` (ks_latent/models/propagator.py,
# ks_latent/config.py, ks_latent/training/loops.py) -- generalizes the
# existing `poly_stable_leading` trick (which only ever pins ONE
# auto-detected leading-order column's sign) to an explicit SET of linear
# orders, each reparametrized as c_n = -(raw_n)**2 (architecturally
# negative, ALWAYS, regardless of how raw_n moves under gradient descent).
# Used here as {2: -1.0, 4: -1.0}: pins BOTH w_xx's and w_xxxx's
# coefficient signs to match true KS's own -w_xx-w_xxxx (order=2 negative
# supplies the low-wavenumber instability that makes chaos possible at
# all; order=4 negative supplies the high-wavenumber damping that bounds
# it), while leaving their MAGNITUDES fully learnable (raw_n is a genuine
# nn.Parameter, real gradient) -- --spectral-poly-stable-w-xx-w-xxxx.
#
# "Can't make huge steps in nu": new Stage{1,2}TrainingConfig.
# stable_linear_lr_factor (default 0.02, i.e. 1/50th of --lr) puts these
# two scalars in their OWN optimizer param group at training/loops.py's
# AdamW construction -- --stable-linear-lr-factor.
#
# Verified directly before this launch (see conversation): standalone
# _SpectralPDEDeltaBody instantiation confirms physical coefficients
# recover EXACTLY -1.0 at both order=2 and order=4 at init, and both
# receive nonzero gradient after a real forward+backward pass (learnable,
# not frozen -- distinct from --pde-fix-w-xx-w-xxxx's FROZEN analogue,
# which has no counterpart for the main aux propagator until now).
#
# Same field settings as Sections 167/169 otherwise (local_field encoder,
# spectral_pde_raw backbone, K=49/L=96.0, polynomial field_kind degree=2
# max_term_order=6, euler integrator, poly_no_constant +
# poly_exclude_nonconservative retained), fresh Stage 1 (cannot reuse a
# prior checkpoint -- the new mechanism changes what's learnable from the
# very start), Stage 2 with the same k_max=16/300-epoch curriculum as
# 169/171/172/173 for a direct comparison. No w_prop_energy_floor this
# time (the whole point is to test whether the STRUCTURAL fix alone
# prevents collapse, not stacked with another mechanism -- confounding the
# comparison would defeat the purpose).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section174_ks_template_stable_linear

echo "=== [1/6] Stage 1: local_field encoder + spectral_pde_raw propagator, NEW --spectral-poly-stable-w-xx-w-xxxx (w_xx/w_xxxx coefficients LEARNABLE but architecturally sign-constrained, init -1.0/-1.0, matching true KS) + --stable-linear-lr-factor 0.02 (1/50th LR on just those two scalars), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 \
  --spectral-poly-max-term-order 6 --spectral-integrator euler \
  --spectral-poly-no-constant --spectral-poly-exclude-nonconservative \
  --spectral-poly-stable-w-xx-w-xxxx --stable-linear-lr-factor 0.02 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/6] Stage 2: spectral_pde_raw propagator continues (architecture, incl. the stable-linear-terms mechanism, inherited from the Stage-1 checkpoint), same --stable-linear-lr-factor 0.02, k_max=16 (Section 37's curriculum), 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k16_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --stable-linear-lr-factor 0.02 \
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

# Report the fitted physical w_xx/w_xxxx coefficients directly -- did they
# stay near -1.0 (true KS), drift while staying negative, or (if this
# print ever shows a positive value) reveal a bug in the sign guarantee.
body = prop.body.inner if hasattr(prop.body, 'inner') else prop.body
if hasattr(body, 'stable_linear_raw_parameters'):
    import math
    from ks_latent.models.propagator import _polynomial_term_indices
    term_indices = _polynomial_term_indices(body.max_order + 1, body.poly_degree, body.poly_max_term_order)
    w = body._poly_weight()
    char_k = 2.0 * math.pi * body.K / body.L
    for order in sorted(body._stable_linear_positions.values()):
        pos = term_indices.index((order,))
        raw_weight = w[0, pos].item()
        physical = raw_weight / (char_k ** order)
        print('[$LABEL] fitted physical coefficient, order=%d: %.6f' % (order, physical))
"
}

echo "=== [3/6] FULL Jacobian spectrum + conditioning check + fitted w_xx/w_xxxx coefficients on the FINAL propagator ==="
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

echo "=== Section 174 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
