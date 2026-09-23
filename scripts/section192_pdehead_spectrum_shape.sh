#!/bin/zsh
# User-directed 2026-09-21: "maybe we can try our new regularization
# methods with the joint propagator/pde training from before? where the
# pde is not the propagator." Confirmed precisely before building: this
# means Section 167's DECOUPLED architecture -- a real, flexible
# propagator (`aux`, --aux-backbone masked_mlp_expand) drives the actual
# K-step prediction loss and rollout, while a SEPARATE `pde_head`
# (polynomial/Chebyshev SINDy-style closure) is fit alongside via
# distillation against REAL DATA ONLY (never even seeing aux's own
# rollout in 167's final recipe) -- unlike Sections 175-191, where the
# interpretable PDE (spectral_pde_raw) WAS the propagator, forced to
# simultaneously produce genuine chaos AND stay a short readable term
# library (the exact combination that failed 12 straight times).
#
# Historical facts confirmed directly from old logs before building this
# (not from memory): Section 167's own `aux` (masked_mlp_expand) already
# achieves genuinely correct chaos on its own --
# artifacts/logs/gate3_analysis_section167_..._300ep.log:
# lyapunov_single_state_D_KY = 22.668, dead center of the true 21-24
# benchmark. But `pde_head`'s own standalone rollout
# (pdehead_standalone_section167_..._300ep.log) NEVER diverges yet DECAYS
# toward a fixed point: 1.818 (t=0) -> 1.889 (t=10) -> 0.958 (t=80) ->
# 0.275 (t=199) -- bounded but not chaotic on its own.
#
# User: "please add the new regularizer just to the pde_head. Pick the
# most promising configuration from both sets of experiments and combine
# them." This section:
#   - OLD arc's most-promising configuration = Section 167's EXACT
#     recipe, verbatim (local_field encoder, masked_mlp_expand aux
#     attn_window=7/expand_factor=3, channel-mean fix, doubled
#     --w-spatial/--w-smooth, pde_head field_kind=polynomial degree=2
#     max_term_order=6, --pde-poly-no-constant --pde-poly-exclude-
#     nonconservative, pde_head fit MUTUAL against real data only in
#     Stage 1, propagator fully decoupled by Stage 2, Stage 2's own
#     iLED-style --w-pde-distill-real-rollout/--w-pde-nonlinear-l2,
#     k_max=12/300 epochs). Stage 1 is UNCHANGED from 167 -- the new
#     regularizers are added in STAGE 2 ONLY, matching this arc's own
#     established precedent (w_pde_nonlinear_l2 itself was only ever
#     added in Stage 2, never Stage 1, across 159/160/167).
#   - NEW arc's regularizers = Section 186's two-sided spectrum-shape
#     loss + multistep composed-Jacobian spectrum-shape loss
#     (n_expand=13/expand_target=1.1/contract_floor=0.6, Section 133's
#     own real-Lyapunov-derived targets; weight 0.4 each, matching 186's
#     own calibration) -- newly wired to apply to `pde_head.step_one`
#     ONLY (ks_latent/config.py's new Stage1/2TrainingConfig.w_pde_
#     spectrum_shape/w_pde_spectrum_shape_multistep fields,
#     ks_latent/training/loops.py's new wiring, scripts/train_stage{1,2}_
#     patched.py's new CLI flags -- all added this session). The main
#     `aux` propagator is NEVER touched by these -- it is already
#     correctly chaotic (D_KY=22.67) and this experiment's whole point is
#     to leave it alone while giving pde_head its own independent
#     pressure toward sustained (not decaying) chaos.
#   - pde_head deliberately does NOT get the w_t/w_tt time-derivative
#     library extension (Sections 189-191) -- Section 191 found those
#     features let a closure satisfy short-horizon loss via literal
#     Taylor extrapolation instead of discovering genuine spatial
#     structure, which is exactly the failure mode this experiment is
#     trying to avoid reintroducing.
#
# Verified via a real (--profile full, --epochs 3 for Stage 1 unchanged
# from 167 already verified there; a short Stage 2 dry run of THIS
# section's new flags) before this launch: no NaN, new loss terms
# computed and finite.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section192_pdehead_spectrum_shape

echo "=== [1/2] Stage 1: IDENTICAL to Section 167 (unchanged) -- local_field + masked_mlp_expand (attn_window=7, expand_factor=3), channel-mean fix, pde_head field_kind=polynomial degree=2 max_term_order=6, --pde-poly-no-constant --pde-poly-exclude-nonconservative, pde_head fit MUTUAL against real data only, --amp, 200 epochs ==="
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
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/2] Stage 2: IDENTICAL to 167's own Stage 2 (propagator trains independently, pde_head continues real-data-only fitting + iLED nonlinear-l2) PLUS NEW --w-pde-spectrum-shape 0.4 --pde-spectrum-shape-two-sided + --w-pde-spectrum-shape-multistep 0.4 (both pde_head.step_one ONLY, aux untouched), k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill-real 0.5 \
  --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 4 \
  --w-pde-nonlinear-l2 0.01 \
  --w-pde-spectrum-shape 0.4 --pde-spectrum-shape-n-expand 13 --pde-spectrum-shape-expand-target 1.1 --pde-spectrum-shape-contract-floor 0.6 --pde-spectrum-shape-n-samples 32 --pde-spectrum-shape-two-sided \
  --w-pde-spectrum-shape-multistep 0.4 --pde-spectrum-shape-multistep-k 10 --pde-spectrum-shape-multistep-n-samples 16 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
STAGE2_PDEHEAD=artifacts/stage2_pdehead_patched_full_${STAGE2_TAG}.pt

echo "=== [3/4] pde_head standalone rollout (the direct 167 comparison: did decay-to-fixed-point get fixed?) ==="
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
print('pde_head standalone rollout (200 steps) first non-finite step (-1=never):', first_bad)
for t in [0,10,20,40,60,80,100,150,199]:
    v = z_roll[0,t]
    print(' t=%d max|z|=%s' % (t, v.abs().max().item() if torch.isfinite(v).all() else 'nan/inf'))
print('(Section 167 comparison: 1.818 -> 1.889 -> ... -> 0.958 (t=80) -> 0.275 (t=199), decaying)')
"

echo "=== [4/4] Real Lyapunov spectrum for BOTH aux (sanity check: still ~22.67, untouched) and pde_head (the actual test: did it start sustaining chaos on its own?) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
aux, aux_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); aux.eval(); pdehead.eval()
d_latent = aux.cfg.d_latent

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z01 = ae.encode(traj[:2]).numpy()

print('--- aux propagator (masked_mlp_expand, should be ~unchanged from 167s own 22.67) ---')
res_aux = lyapunov_spectrum_latent_propagator(
    aux, z01, mode='single_state', n_directions=d_latent, n_steps=400, qr_every=5,
    dt_snap=1.0, warmup_steps=50, seed=0, max_abs_state=1e3,
)
print('lambda1:', res_aux.exponents[0], ' n_positive:', res_aux.n_positive, '/', res_aux.n_directions)
try:
    print('D_KY:', res_aux.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)

print()
print('--- pde_head (the actual test: real spectrum, not just bounded-vs-decaying) ---')
res_pde = lyapunov_spectrum_latent_propagator(
    pdehead, z01, mode='single_state', n_directions=d_latent, n_steps=400, qr_every=5,
    dt_snap=1.0, warmup_steps=50, seed=0, max_abs_state=1e3,
)
print('lambda1:', res_pde.exponents[0], ' n_positive:', res_pde.n_positive, '/', res_pde.n_directions)
try:
    print('D_KY:', res_pde.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)
print('spectrum (first 20):', [round(x,4) for x in res_pde.exponents[:20]])
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 192 complete ==="
