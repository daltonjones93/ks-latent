#!/bin/zsh
# User-directed 2026-09-10: "next I'd like to run 141 again, but with
# max_term_order = 4, and something else to try an prevent divergence" --
# then immediately corrected: "hang on I think I meant max order 5 not 4,
# we have to keep w_xxxx."
#
# Context: Section 141 (max_term_order=6) improved but did not fix
# pde_head's own free-running divergence -- it survived to ~t=80-100
# (vs. Section 140's unrestricted t=57-59) before still blowing up to
# NaN. Section 142 (spectral_pde_raw as sole PRIMARY propagator,
# max_term_order=6 + poly_stable_leading) was killed after two checks
# showed the SAME divergence getting WORSE with more training (first
# non-finite step went from 60 at epoch 39 to 15 at epoch 119, early
# separation growing 39.8 -> 46.2) -- confirming this is a real,
# worsening instability, not something training self-corrects.
#
# TWO changes from Section 141, both aimed directly at the divergence:
#
#   1. --pde-poly-max-term-order 5 (down from 6). max_term_order excludes
#      any monomial whose derivative orders SUM to >= the threshold, and a
#      single index (n,) has "combined order" n itself
#      (_polynomial_term_indices' own docstring) -- so max_term_order=4
#      (the user's first instinct) would have excluded the LINEAR w_xxxx
#      term too (order 4 >= 4), which was the single LARGEST term in every
#      closure extracted so far (138's -1.176, 140's -0.770) and is
#      exactly true KS's own dissipation term -- caught before launch.
#      max_term_order=5 keeps w_xxxx (order 4 < 5) while still excluding
#      every combined-order->=5 cross term (e.g. w_x*w_xxxx, order 5;
#      w_xx*w_xxxx, order 6) -- one step stricter than 141's own
#      keep-through-order-5 threshold of 6, without losing the dominant
#      linear term. Compare directly against 141's own max_term_order=6
#      numbers.
#
#   2. `--w-pde-rollout 0.2` (Stage 2, alongside `--w-pde-distill 0.5`,
#      unchanged) -- "something else to try and prevent divergence".
#      w_pde_distill ("option A") ONLY ever trains pde_head on SINGLE
#      steps from the real propagator's own realized states -- pde_head
#      has never once been trained on its OWN compounding multi-step
#      error, which is exactly the property that's failing (a great
#      single-step fit that still diverges over many free-running steps).
#      w_pde_rollout ("option B") chains pde_head's OWN autoregressive
#      k_now-step rollout and compares the WHOLE trajectory to the
#      propagator's -- directly training multi-step stability instead of
#      only single-step accuracy. Flagged in this project's own docstring
#      as "the riskier direct approach that made backbone='spectral_pde'
#      hard to train as a PRIMARY propagator in Sections 101-106" -- but
#      here pde_head is a SIDE model riding alongside an already-good free
#      propagator (not the primary), and w_pde_distill's own single-step
#      anchor stays active too, so the risk profile should be lower than
#      101-106's context. Weight kept modest (0.2, well below
#      w_pde_distill's 0.5) as a first pass.
#
#   Also newly wired (previously missing): `--pde-poly-stable-leading`
#   (Stage 1's own pde_head construction never exposed this -- added this
#   turn, mirroring `--spectral-poly-stable-leading`'s identical mechanism)
#   -- architecturally forces the leading kept even-order linear
#   coefficient's sign to guarantee UV/high-wavenumber stability. Used
#   here too, on the theory that term-order restriction and rollout
#   training address DIFFERENT failure directions (excessive cross-term
#   growth vs. accumulated compounding error) while stable_leading
#   addresses a third (high-wavenumber runaway) -- belt and suspenders,
#   cheap to include.
#
# Verified via real (--profile full, small-epoch) dry runs of both stages
# before this launch -- pde_distill AND pde_rollout both logged and
# decrease normally, no errors.
#
# Everything else IDENTICAL to Section 141: local_field (n_sites=32/
# local_channels=3/site_mix_radius=2/n_site_mix_layers=3/hidden=32),
# aux-backbone mlp, Section 98/141-style regularizers (w_var=0.01,
# w_spatial=0.06 signed, w_logdet=0.01, w_smooth=0.006), --full-propagator,
# 200+300 epoch curriculum. Same before/after Jacobian-spectrum check,
# visualization, Gate 3/4 suite, k=1,2,4,8 predictiveness eval, AND a
# standalone pde_head free-running divergence check (the same one that
# caught 140/141's own long-horizon blowup) as every section in this arc.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section143_localfield_p32c3_propmlp_jointpde_termorder5_rollout_200ep

echo "=== [1/7] Stage 1: local_field + mlp + --pde-distill --pde-mutual --pde-poly-max-term-order 5 --pde-poly-stable-leading, w_var=0.01/w_spatial=0.06/w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.06 --spatial-signed --w-logdet 0.01 --w-smooth 0.006 \
  --pde-distill --w-pde-distill 0.5 --pde-mutual --pde-field-kind polynomial --pde-poly-degree 2 \
  --pde-poly-max-term-order 5 --pde-poly-stable-leading --pde-integrator euler \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/7] Stage 2: propagator AND pde_head continue together, UNFROZEN (mutual), w_pde_distill=0.5 + w_pde_rollout=0.2, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill 0.5 --w-pde-rollout 0.2 \
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

z_np = z_all.reshape(-1, z_all.shape[-1]).numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
"
}

echo "=== [3/7] FULL Jacobian spectrum + conditioning check on the FINAL propagator ==="
_spectrum_check > artifacts/logs/jacobiancheck_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${STAGE2_TAG}.log

echo "=== [4/7] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) ==="
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

echo "=== [5/7] Is the pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138-141 ==="
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
    r2_vs_prop = 1 - mse_vs_prop / z_prop[:, :k].var().item()
    print('k=%d: pde_vs_prop_mse=%.5f (R2=%.4f)' % (k, mse_vs_prop, r2_vs_prop))
n_params = sum(p.numel() for p in pdehead.parameters())
print('param count: %d' % n_params)
" > artifacts/logs/jointpde_eval_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jointpde_eval_${STAGE2_TAG}.log

echo "=== [6/7] Standalone pde_head free-running divergence check (same test that caught 140/141's own blowup) ==="
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

echo "=== [7/7] Section 143 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
