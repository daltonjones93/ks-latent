#!/bin/zsh
# User-directed 2026-09-03, "Section 69": follow-up to the vit-vs-
# fourier_mlp discussion (Sections 65-67) -- "for the next run can we
# make d model = 108 and attn_window = 6. I want to use this for both
# encoder and decoder, keep everything else the same about the ViT. the
# propagator should be the same for stage 1 and 2 and it should be the
# fourier mlp. finally we should use the regularization terms from
# section 64."
#
# REVISED mid-run: the first attempt (attn_window=6) was killed after the
# user said "I don't think the readout mlp makes sense [unrelated aside,
# see below]. also I think the smaller attn_window performs better" --
# this relaunch reverts attn_window back to 4 (Section 52/64's original
# value) while keeping d_model=108, to isolate that one variable
# specifically. The readout='mlp' option added earlier today stays in the
# codebase (off by default, unused either way in this run) since removal
# wasn't explicitly requested, just doubt about the idea.
#
# ViT AE: d_model=108 (up from Section 52/64's 96), attn_window=4
# (Section 52/64's original value, reverted back to from this run's first
# attempt at 6) -- both apply identically to encoder AND decoder already
# (ViTAutoencoderConfig has one shared d_model/attn_window field, not
# separate ones), so no special wiring needed for "both encoder and
# decoder". Everything else about the ViT unchanged from Section 52/64:
# pos_encoding=linear, token_window=16 (overlapping tokenization),
# patch_size=8/pool=mean/n_blocks=3/n_heads=4/mlp_ratio=4 (all CLI/config
# defaults). readout stays 'linear' (the NEW readout='mlp' option added
# today is available but NOT enabled here -- it was requested as a
# capability to have, not asked to be turned on for this specific run).
# Sizing check: d_model=108 -> 1011690 params, matching the earlier
# Section 53 d_model=108 precedent exactly (verified directly).
#
# Propagator: backbone=fourier_mlp, mode=history, n_history=2 (Sections
# 65-67's winning propagator architecture -- raw+Fourier features per
# history state through a plain MLPDeltaBody), hidden=140/n_blocks=2
# (Sections 65-67's Section-52-scale sizing, 111344 params, verified
# directly) -- SAME propagator for Stage 1 (aux, --full-propagator
# sizing) and Stage 2 (warm-started via --init-prop-checkpoint, which
# loads the entire saved PropagatorConfig).
#
# Regularizers ("use the regularization terms from section 64"):
# w_decorr=0, w_var=0.02 (unchanged from 52), w_spatial(signed)=0.02 (up
# from 52's 0.01), w_var_floor=0, w_logdet=0.01 (up from 52's 0.0035),
# lambda_z=0.0075 active from epoch 30 (--reg-start-epoch 30, NOT from
# epoch 0) -- Section 64's exact recipe.
#
# Verified before this launch: real (non-unit-test) 2-epoch Stage 1 + 2
# smoke run with this exact flag combination, confirmed AE params
# (1011690), propagator params (111344, backbone/mode/n_history all
# correct), and a successful Stage 2 --init-prop-checkpoint warm-start.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section69_vit_dmodel108_attn4_fouriermlpprop_section64regs_200ep

echo "=== [1/3] Stage 1: vit AE (d_model=108, attn_window=4, pos_encoding=linear, token_window=16) + fourier_mlp/history2 aux propagator (hidden=140/n_blocks=2, full-propagator sizing), Section 64's regularizers (w_decorr=0, w_var=0.02, w_spatial=0.02 (SIGNED), w_var_floor=0, w_logdet=0.01, lambda_z=0.0075 from epoch 30), NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone fourier_mlp --mode history --n-history 2 \
  --pos-encoding linear --attn-window 4 --token-window 16 --d-model 108 \
  --aux-hidden 140 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.02 --spatial-signed --w-var-floor 0 --w-logdet 0.01 \
  --lambda-z 0.0075 --reg-start-epoch 30 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Latent covariance spectrum + D7/D8 bandedness check ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint
from ks_latent.analysis.diagnostics import same_time_coupling_diagnostic, same_time_coupling_diagnostic_signed
from ks_latent.training.losses import spatial_coherence_loss

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
ae, cfg, _ = load_autoencoder_checkpoint('$AE')
ae.eval()
with torch.no_grad():
    z = ae.encode(traj.reshape(n*T, NX))
z_np = z.numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
print('top 10:', np.array2string(eig[:10], precision=3))
print('bottom 10:', np.array2string(eig[-10:], formatter={'float_kind':lambda x: f'{x:.2e}'}))
d7 = same_time_coupling_diagnostic(z_np, n_null=500, seed=0)
print('D7 bandedness=%.4f p=%.4f' % (d7.bandedness_observed, d7.bandedness_p_value))
d8 = same_time_coupling_diagnostic_signed(z_np, n_null=500, seed=0)
print('D8 signed bandedness=%.4f p=%.4f' % (d8.bandedness_observed, d8.bandedness_p_value))
l_spatial_unsigned = spatial_coherence_loss(z, bandwidth=3.0, signed=False)
l_spatial_signed = spatial_coherence_loss(z, bandwidth=3.0, signed=True)
print('l_spatial (unsigned) raw=%.6f (lower=more coherent)' % l_spatial_unsigned.item())
print('l_spatial (signed) raw=%.6f (lower=more coherent)' % l_spatial_signed.item())
" > artifacts/logs/spectrum_${TAG}.log 2>&1
cat artifacts/logs/spectrum_${TAG}.log

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 69 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
