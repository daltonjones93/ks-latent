#!/bin/zsh
# User-directed 2026-09-05, "Section 96": "let's make run 96 have
# w_smooth = 0, w_spatial = .04. leave the rest the same as 95."
#
# Identical to Section 95 (scripts/section95_wspatial03_wsmooth0015.sh)
# in every respect EXCEPT the two regularizer weights:
#   - w_smooth: 0.0015 -> 0 (disabled entirely).
#   - w_spatial (signed): 0.03 -> 0.04.
#
# Architecture/propagator UNCHANGED from 93/94/95: vit_fourier_hybrid,
# d_model=64/fourier_hidden=95/fourier_blocks=2 (~516K total, ~61% of
# Section 93's size), pool=token_mlp/dec_pool=token_mlp (reduction=8/
# hidden=128), enc_out_modes=8, dec_fno_modes=12, attn_window=4/
# pos_encoding=linear/token_window=16, Section 52's EXACT mlp/markovian
# propagator (hidden=128/n_blocks=3). w_var=0.02, w_var_floor=0,
# w_logdet=0.008, NO lambda_z, w_decorr=0 unchanged.
#
# Context: Section 95 (=94 + w_spatial=0.03/w_smooth=0.0015) regressed
# rollout accuracy (val_kmax_mse=0.0198 vs 94's 0.0123) and did NOT
# improve smoothness, but DID achieve the best DA skill of the whole
# 71-95 arc (skill_free_over_da=3.840, vs 93's previous-best 3.282) and
# a notably higher D8 (0.6733 vs ~0.537). This isolates whether w_smooth
# specifically (as opposed to w_spatial) was responsible for 95's
# accuracy regression -- removing it entirely while pushing w_spatial
# even further tests whether w_spatial alone, without w_smooth's
# compounding pressure, can reach a similarly-high-D8/high-DA-skill
# regime without the accuracy cost.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.024880, no errors) and a 4-epoch Stage 2 warm-start smoke run (both
# completed without error). No code changed this turn (pure hyperparameter
# combination of existing capabilities). Smoke artifacts cleaned up before
# this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section96_vitfourieriffthybrid_dmodel64_fourierhidden95_tokenmlp_propmlpmarkovian52_wspatial04_wsmooth0_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder+decoder (=Section 93/94/95's architecture unchanged: d_model=64/fourier_hidden=95, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128, enc_out_modes=8, dec_fno_modes=12) + Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3), w_decorr=0, w_var=0.02, w_spatial=0.04 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0 (disabled), NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone mlp --mode markovian \
  --d-model 64 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --vit-fourier-fourier-hidden 95 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --vit-fourier-enc-out-modes 8 \
  --fourier-mlp-dec-fno-modes 12 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.04 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0 \
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
print('participation ratio: %.3f (out of d_latent=%d)' % (eig.sum()**2 / (eig**2).sum(), cfg.d_latent))
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/markovian aux, --amp, k_max=12, 300 epochs (Section 52's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 96 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
