#!/bin/zsh
# User-directed 2026-09-04, "Section 81": "the run doesn't look very
# good [Section 80]. can we have the same kind of hybrid vit fourier
# mlp, just using the exact fourier mlp from 75 except only using the
# frequency components? make it 1.5 million params total for encoder and
# decoder. call this section 81"
#
# Same overall architecture as Section 80 (ViT for function space +
# Fourier-space branch, SUMMED), but the Fourier-space branch is now
# FourierIFFTBody (ViTFourierHybridAutoencoderConfig.fourier_kind="ifft")
# -- Section 75's OWN Fourier-path mechanism (an MLP predicts frequency-
# domain coefficients, explicitly inverse-transformed via irfft back to
# state space) -- instead of Section 80's new ConservedFourierMLP
# (L1-conservation). "Only using the frequency components": NO raw-
# value/masked path summed in at all (unlike Section 75's own AE, which
# sums a masked raw path alongside FourierIFFTBody) -- just this one
# frequency-domain sub-network, added to the ViT branch's output.
#
# Propagator: still Section 75's EXACT propagator, unchanged (masked
# fourier_mlp, attn_window=22 -- effectively fully dense at d_latent=44
# -- + fourier_ifft_readout, hidden=480/n_blocks=2, mode=history,
# n_history=2).
#
# Sizing ("make it 1.5 million params total for encoder and decoder"):
# verified by direct instantiation sweep this turn -- ViT (d_model=92,
# Section 52's other hyperparameters unchanged) -> 755,834 params.
# FourierIFFTBody-based branch (hidden=270, n_blocks=2, encoder+decoder
# combined) -> 753,604 params. Total AE = 1,509,438 (100.6% of the 1.5M
# target), with the two branches commensurate (ratio 0.997, essentially
# equal). Propagator = 1,005,046 (identical to Section 75's, reused
# unchanged).
#
# Regularizers/curriculum: same as Section 80 -- Section 75's exact
# recipe (w_var=0.01, w_spatial=0.01 signed, w_logdet=0.008,
# lambda_z=0.0002, active from epoch 0) and Stage-2 curriculum
# (k_max=12, 300 epochs, k_warmup_epochs=210/k_mid=8/k_mid_epochs=175).
#
# Verified this turn before launch: 6 new unit tests (fourier_kind="ifft"
# construction, shapes, gradient flow, bfloat16, checkpoint round-trip,
# invalid-kind validation, default-still-"conserved" regression check --
# 18 total in test_autoencoder_vit_fourier_hybrid.py); full unit+
# integration suite (see /tmp/full_test_run25.log); a real 4-epoch
# Stage 1 smoke run -- val_recon_final=0.008821, MUCH cleaner than
# Section 80's ConservedFourierMLP variant (no large initial loss spike:
# 0.75 at epoch 0 here vs. 52.6 there, and converges faster: 0.0088 here
# at 4 epochs vs. Section 80's 0.012 after 40 epochs); a 4-epoch Stage 2
# warm-start smoke run showed a striking early signal --
# val_kmax_mse=0.03355 after only 4 epochs, already BETTER than Section
# 75's own fully-converged 300-epoch result (0.0392). Smoke artifacts
# cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section81_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_section75regs_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder/decoder (ViT d_model=92, Section 52 structure + FourierIFFTBody hidden=270/n_blocks=2, pure frequency components, summed) + Section 75's exact fourier_mlp propagator (attn_window=22, fourier_ifft_readout, hidden=480/n_blocks=2), w_decorr=0, w_var=0.01, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.008, lambda_z=0.0002 (from epoch 0), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone fourier_mlp --mode history --n-history 2 \
  --d-model 92 --pos-encoding linear --attn-window 4 --token-window 16 --pool mean \
  --vit-fourier-fourier-hidden 270 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --prop-attn-window 22 --prop-fourier-ifft \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.01 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --lambda-z 0.0002 --reg-start-epoch 0 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator (Section 75's exact architecture), --amp, k_max=12, 300 epochs (Section 75's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 81 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
