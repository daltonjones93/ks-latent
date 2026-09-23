#!/bin/zsh
# User-directed 2026-09-03, "Section 76": "make section 76 the sme
# [same] model with w_var = .0005, w_spatial = .04, w_logdet = .004,
# lambda_z = .0005. also extend stage 2 to 500 epochs, with rollout k up
# to 20." (Supersedes an earlier, never-launched Section 76 plan --
# 75%-sized model with w_var=.03/w_spatial=.02/w_logdet=.02/lambda_z=.0005
# -- discarded in favor of this one.)
#
# "same model" = Section 75's exact architecture/size (masked fourier_mlp
# encoder@attn_window=8 + Fourier-IFFT-readout everywhere, propagator/
# decoder masked@window=22 + Fourier-IFFT-readout, AE hidden=224/
# n_blocks=4 ~1M params, propagator hidden=480/n_blocks=2 ~1M params) --
# Section 75 confirmed the lambda_z hypothesis directly: dropping
# lambda_z from Sections 71/72/74's failing 0.002 down to 0.0002 (all
# else similar) took best_val_kmax_mse from ~0.15-0.29 down to 0.0392,
# BEATING Section 66's own 0.0528 baseline. This run keeps lambda_z in
# that same healthy range (0.0005, still far below the 0.002 that
# failed) while substantially reweighting the other three regularizers:
# w_var 0.01->0.0005 (nearly off), w_spatial(signed) 0.01->0.04 (4x),
# w_logdet 0.008->0.004 (half).
#
# Stage 2 extended per user direction: 300->500 epochs, k_max 12->20.
# Curriculum scaled proportionally from Sections 71-75's established
# 300-epoch/k_max=12 recipe (k_warmup_epochs=210 [70% of total], k_mid=8
# [67% of k_max], k_mid_epochs=175 [58.3% of total]) to preserve the same
# SHAPE of ramp -> dwell-at-k_max structure at the new scale:
#   k_warmup_epochs = 350 (70% of 500)
#   k_mid = 13 (65% of 20, closest integer to 67%)
#   k_mid_epochs = 290 (58% of 500)
# i.e. ramp 2->13 over the first 290 epochs, 13->20 over the next 60,
# then dwell at k=20 for the final 150 epochs (30% of total, matching
# the previous recipe's 90/300=30% dwell fraction exactly).
#
# Verified this turn: a real 4-epoch Stage 1 smoke run
# (val_recon_final=0.021568) with the new regularizer weights, and a
# 6-epoch Stage 2 warm-start smoke run confirming the new curriculum
# flags (--k-max 20 --k-warmup-epochs 350 --k-mid 13 --k-mid-epochs 290)
# are accepted and run without error (val_kmax_mse still large at only 6
# epochs is expected -- the k-ramp barely starts by epoch 6 of 350).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section76_enc8_prop22_dec22_fourierifftreadout_wvar00005_wspatialsigned04_logdet004_lambdaz0005_200ep

echo "=== [1/3] Stage 1: masked fourier_mlp encoder (attn_window=8) + Fourier-IFFT-readout everywhere (decoder/propagator masked@window=22 + Fourier-IFFT, same size as Section 75), w_decorr=0, w_var=0.0005, w_spatial=0.04 (SIGNED), w_var_floor=0, w_logdet=0.004, lambda_z=0.0005 (from epoch 0), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 8 --prop-attn-window 22 --dec-attn-window 22 --fourier-ifft-readout --prop-fourier-ifft \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 4 \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.0005 --w-spatial 0.04 --spatial-signed --w-var-floor 0 --w-logdet 0.004 \
  --lambda-z 0.0005 --reg-start-epoch 0 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked@22+fourier_ifft_readout fourier_mlp/history2 aux propagator, --amp, k_max=20, 500 epochs, --compare-k 12 (added 2026-09-04, user-directed: 'add a readout for val_kequals12_mse, so we can compare these runs more directly' -- pools MSE over just the first 12 steps of the same k_max=20 rollout, directly comparable to Sections 71-75's own k_max=12 val_kmax_mse, since val_kmax_mse alone is inflated at k_max=20 by 8 extra already-diverged late steps -- see Stage2TrainingConfig.compare_k's docstring) ==="
STAGE2_TAG="${TAG}_warmstart_k20_500ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 500 --k-max 20 --k-warmup-epochs 350 --k-mid 13 --k-mid-epochs 290 --compare-k 12 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 76 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
