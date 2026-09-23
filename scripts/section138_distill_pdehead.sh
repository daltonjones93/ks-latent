#!/bin/zsh
# User-directed 2026-09-10, following the discussion of whether Section
# 136/137's propagator dynamics (genuine chaos recovered, D3 Jacobian
# bandedness significant p=0.0000 on a FULLY FREE mlp propagator -- no
# architectural locality forced anywhere) are now good candidates for
# distillation into an explicit, interpretable local PDE, rather than
# building yet another local/masked PRIMARY propagator (already tried
# twice -- Sections 120-134 on the vit encoder, Section 135 on local_field
# -- and it collapsed/diverged both times, consistent with H-PROP's own
# repeated finding across this whole project: local/masked PRIMARY
# propagators fail regardless of mechanism; only a free, globally-mixing
# one recovers genuine chaos).
#
# "let's set up the test so that we distill from both 136 and 137 so we
# can compare the results"
#
# Reuses this project's OWN existing machinery, unmodified -- `pde_head`
# (Sections 107-110) + `--freeze-propagator` (train_stage2's Phase-3
# mode, docs/sine_transform_pde_plan.md sec 23): fit a SEPARATE,
# interpretable `backbone="spectral_pde_raw"` local-PDE model to match an
# ALREADY-TRAINED, FROZEN main propagator's realized dynamics, with the
# main propagator excluded from the optimizer entirely -- zero risk of
# the distillation objective ever collapsing the thing being distilled,
# unlike every attempt at training a local model as the PRIMARY propagator
# from scratch.
#
# NEW (this section): `scripts/build_fresh_pdehead_checkpoint.py` --
# constructs a FRESH (untrained) spectral_pde_raw pde_head in the exact
# checkpoint format train_stage1_patched.py --pde-distill writes, WITHOUT
# running an entire Stage-1 joint-training job just to seed it (nothing
# about Phase-3 distillation requires pde_head to have been pre-trained
# jointly with anything -- it starts fresh and is trained ONLY against the
# frozen target). field_kind="polynomial" (not "mlp"): the whole point is
# an INTERPRETABLE closure with named coefficients, not another opaque
# network. integrator="euler" (not "etdrk4"): "etdrk4" bakes in KS's TRUE
# physical dispersion relation, which does not hold for local_field's
# arbitrary (site, channel)-structured latent the way it does for a
# genuine truncated-Fourier w=irfft(z) field.
#
# Distills from BOTH Section 136 (local_field p32c3, d_latent=96,
# D_KY=23.75/n_positive=13/lambda1=0.112, skill=1.84x, cond#=1.4e16) and
# Section 137 (local_field p16c3, d_latent=48, stronger w_logdet/w_spatial/
# w_smooth -- own Gate 3/4 numbers checked just before this launched) so
# the two can be compared directly: does the SMALLER, better-targeted
# latent (137) distill into a better-fitting/simpler PDE than the wider,
# more ill-conditioned one (136)?
#
# Verified via a real (non-profile=smoke; --profile smoke's own tiny
# synthetic dimension doesn't match a real checkpoint's actual d_latent)
# 3-epoch dry run against Section 136's real checkpoints before this
# launch -- confirmed pde_distill loss decreases, frozen propagator's own
# val_kmax_mse stays exactly constant across epochs (proof it truly isn't
# being touched).
set -e
cd /Users/daltonjones/Documents/latent_DA

_distill_and_eval() {
  local AE_PATH=$1
  local PROP_PATH=$2
  local D_LATENT=$3
  local LABEL=$4
  local PDEHEAD_FRESH=/tmp/pdehead_fresh_${LABEL}.pt

  echo "=== [$LABEL] Building fresh spectral_pde_raw pde_head (d_latent=$D_LATENT, polynomial, euler) ==="
  mamba run -n da_env python scripts/build_fresh_pdehead_checkpoint.py \
    --d-latent "$D_LATENT" --out "$PDEHEAD_FRESH" \
    --pde-field-kind polynomial --pde-poly-degree 2 --pde-integrator euler

  echo "=== [$LABEL] Phase-3 distillation: pde_head trained against the FROZEN $LABEL propagator, 500 epochs, w_pde_distill=1.0 ==="
  mamba run -n da_env python scripts/train_stage2_patched.py \
    --ae-checkpoint "$AE_PATH" \
    --init-prop-checkpoint "$PROP_PATH" \
    --init-pdehead-checkpoint "$PDEHEAD_FRESH" \
    --freeze-propagator --w-pde-distill 1.0 \
    --profile full --epochs 500 --k-max 8 --k-warmup-epochs 100 \
    --tag "distill_${LABEL}" \
    > artifacts/logs/distill_${LABEL}.log 2>&1
  tail -5 artifacts/logs/distill_${LABEL}.log

  local PDEHEAD_DISTILLED=artifacts/stage2_pdehead_patched_full_distill_${LABEL}.pt

  echo "=== [$LABEL] Evaluating distilled pde_head: rollout MSE vs the frozen propagator AND vs real ground truth, at k=1,2,4,8 ==="
  mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE_PATH')
prop, prop_cfg, _ = load_propagator_checkpoint('$PROP_PATH')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$PDEHEAD_DISTILLED')
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

var_true = z_true.var().item()
for k in (1, 2, 4, 8):
    mse_vs_prop = ((z_pde[:, :k] - z_prop[:, :k])**2).mean().item()
    mse_vs_truth = ((z_pde[:, :k] - z_true[:, :k])**2).mean().item()
    prop_mse_vs_truth = ((z_prop[:, :k] - z_true[:, :k])**2).mean().item()
    r2_vs_prop = 1 - mse_vs_prop / z_prop[:, :k].var().item()
    print('[$LABEL] k=%d: pde_vs_prop_mse=%.5f (R2=%.4f)   pde_vs_truth_mse=%.5f   prop_vs_truth_mse=%.5f (own propagator error, for reference)' % (
        k, mse_vs_prop, r2_vs_prop, mse_vs_truth, prop_mse_vs_truth
    ))

n_params = sum(p.numel() for p in pdehead.parameters())
print('[$LABEL] distilled pde_head param count: %d' % n_params)
" > artifacts/logs/distill_eval_${LABEL}.log 2>&1
  cat artifacts/logs/distill_eval_${LABEL}.log
}

echo "=== Distilling from Section 136 (local_field p32c3, d_latent=96) ==="
_distill_and_eval \
  artifacts/stage1_ae_patched_full_section136_localfield_p32c3_propmlp_200ep.pt \
  artifacts/stage2_prop_patched_full_section136_localfield_p32c3_propmlp_200ep_warmstart_k12_300ep.pt \
  96 "136"

echo "=== Distilling from Section 137 (local_field p16c3, d_latent=48, stronger regularizers) ==="
_distill_and_eval \
  artifacts/stage1_ae_patched_full_section137_localfield_p16c3_d48_propmlp_strongerreg_200ep.pt \
  artifacts/stage2_prop_patched_full_section137_localfield_p16c3_d48_propmlp_strongerreg_200ep_warmstart_k12_300ep.pt \
  48 "137"

echo "=== Section 138 complete -- side-by-side comparison ==="
echo "--- 136 ---"
cat artifacts/logs/distill_eval_136.log
echo "--- 137 ---"
cat artifacts/logs/distill_eval_137.log
