#!/bin/zsh
# User-directed 2026-09-23: "do we apply w_var, w_logdet, w_spatial during
# stage 2? we probably should. If we don't please implement this and make
# sure the latest run uses these to prevent collapse."
#
# Checked directly against ks_latent/training/loops.py and both CLI
# scripts before changing anything:
#
# - w_var: Stage 2 does NOT have a literal --w-var (it can't -- the
#   encoder is FROZEN in Stage 2, so a term on the raw encoder output
#   would get zero gradient). It DOES already have the intended Stage-2
#   analogue: --w-varmatch (+ --w-varmatch-adaptive), applied to the
#   PROPAGATOR's own rolled-out z_pred instead -- built 2026-08-29,
#   "motivated by the fixed-point collapse", i.e. for exactly this
#   problem. EXISTED, was simply never turned on in Section 202's launch.
# - w_spatial: Stage 2 already has a --w-spatial flag, literally the same
#   name, "Stage-2 analogue of Stage1TrainingConfig.w_spatial ... applies
#   spatial_coherence_loss to the PROPAGATOR's rolled-out predictions
#   z_pred". EXISTED, also never turned on in Section 202.
# - w_logdet: Stage 2 only had --w-logdet-rollout, and it is HARD-
#   RESTRICTED to backbone="spectral_pde"/"spectral_pde_raw" (it decodes
#   z_pred to physical space via decode_from_spectrum first, machinery
#   only those backbones have). Our propagator is backbone="mlp" -- this
#   flag would silently do nothing for it even if turned on. This WAS a
#   real, genuine gap: no backbone="mlp"/"vit"/etc. propagator had ANY
#   logdet-based anti-collapse term in Stage 2 before today.
#
# Fixed (ks_latent/config.py, ks_latent/training/loops.py,
# scripts/train_stage2_patched.py): added Stage2TrainingConfig.
# w_logdet_rollout_latent (+ logdet_rollout_latent_eps) and its CLI flags
# --w-logdet-rollout-latent/--logdet-rollout-latent-eps -- mirrors Stage
# 1's own --w-logdet exactly (logdet_barrier_loss applied directly to a
# batch of latent vectors, no physical decode needed), just on the
# propagator's rolled-out z_pred instead of the encoder's raw z. General
# to any backbone. Smoke-tested combined with --w-varmatch/--w-spatial/
# --w-spectrum-shape all at once before this launch.
#
# RERUN, user-directed: "please rerun 203 with no rollout in phase 1."
# Confirmed cause of the first attempt's Stage-1 collapse (D_KY=0.00,
# n_positive=0/20 -- see docs/OPEN_QUESTIONS.md): --multistep extends
# Stage 1's own auxiliary-propagator L_pred term to an 8-step rollout
# (k_pred ramped 2->8), which is exactly the long-horizon-MSE mechanism
# already diagnosed as Stage 2's collapse cause -- it was evidently
# reintroducing that same pressure inside Stage 1 itself. `--multistep`
# REMOVED below (this also reverts aux-propagator sizing/checkpoint-
# saving to depend on `--full-propagator` alone, which is kept --
# hidden=128/n_blocks=3 sizing and Stage-2-compatible checkpoint saving,
# unrelated to the rollout curriculum). With `--multistep` gone,
# `Stage1TrainingConfig.k_pred_max` reverts to its own default of 0 (no
# --multistep -> 0, per --k-pred-max's own CLI help) -- genuinely NO
# multi-step rollout term in Stage 1, exactly matching Sections 199/201's
# own successful (non-collapsed) configuration. `--w-spectrum-shape`
# (idea 1, retargeted for L96: n_expand=5, target=1.1, floor=0.6,
# two_sided) is the only thing left on top of that known-good base --
# isolates whether idea 1 ALONE preserves Stage-1 chaos and then also
# protects Stage 2, the original question Section 203 was meant to
# answer before --multistep confounded it.
#
# Otherwise identical to the first attempt: ViT encoder (pos_encoding=
# linear, full attention), d_latent=20, mode=markovian, same Lorenz-96
# N=64/F=4.2/dt_snap=0.1 dataset, same 40/60-epoch schedule, same Stage-2
# geometry regularizers (--w-varmatch/--w-spatial/--w-logdet-rollout-
# latent, the actual fix for this section's own original "do we apply
# w_var/w_logdet/w_spatial in Stage 2" question, unaffected by the
# --multistep bug and kept unchanged).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section203_lorenz96_stage2_geometry_regularizers
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5
SPECTRUM_ARGS=(--w-spectrum-shape 0.4 --spectrum-shape-n-expand 5 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-n-samples 32 --spectrum-shape-two-sided)

echo "=== [1/3] Stage 1: ViT encoder (pos_encoding=linear, full attention), d_latent=20, mode=markovian, NO rollout (--multistep removed) + --w-spectrum-shape (n_expand=5, target=1.1, floor=0.6, two_sided), w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, on Lorenz-96 N=64 F=4.2 dt_snap=0.1, --amp, 40 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 64 --d-latent 20 \
  --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --full-propagator "${SPECTRUM_ARGS[@]}" --amp \
  --epochs 40 --checkpoint-every 4 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

_diagnose() {
  local PROP_PATH=$1
  local LABEL=$2
  mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$PROP_PATH')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('[$LABEL] d_latent:', d_latent)

DATASET = '$DATASET'
with h5py.File(DATASET,'r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, N = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, N)).reshape(n, T, -1)
z0b = z_all[:,0,:]

with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=2000)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('[$LABEL] multi-IC (20) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,50,100,300,600,1000,1500,1999]:
    if t < traj_roll.shape[1]:
        print('[$LABEL] t=%d  max|z| across all 20 ICs: %.4f' % (t, maxz_per_t[t].item()))
final_states = traj_roll[:,-1,:]
pairwise = torch.cdist(final_states, final_states)
print('[$LABEL] final-state pairwise distance (min excl. diag, max, mean):',
      (pairwise + torch.eye(20)*1e9).min().item(), pairwise.max().item(), pairwise.mean().item())

with h5py.File(DATASET,'r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z01 = ae.encode(traj[:2]).numpy()
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('[$LABEL] D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('[$LABEL] D_KY: could not bracket --', e)
print('[$LABEL] (true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
print('[$LABEL] (Section 199 comparison, ViT d20 markovian, no regularizers: stage1 D_KY=1.35, stage2 D_KY=0.00)')
print('[$LABEL] (Section 201 comparison, ViT d20 history n=6, no regularizers: stage1 D_KY=11.94, stage2 D_KY=0.00)')
"
}

echo "=== [2/3] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/3] Stage 2: warm-started from Stage 1, spectrum-shape + varmatch + spatial + logdet-rollout-latent ALL on, NO delta_cap, --amp, k_max=12, 60 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_60ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  "${SPECTRUM_ARGS[@]}" \
  --w-varmatch 0.02 --w-varmatch-adaptive \
  --w-spatial 0.01 --spatial-signed \
  --w-logdet-rollout-latent 0.0035 \
  --amp \
  --epochs 60 --k-max 12 --k-warmup-epochs 42 --k-mid 8 --k-mid-epochs 35 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 203 complete ==="
