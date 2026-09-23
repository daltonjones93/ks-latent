#!/bin/zsh
# User-directed 2026-09-02, "Section 53": same recipe family as Sections
# 48-52 (mlp/markovian aux, w_decorr=0, w_var_floor=0, NO delta_cap,
# --amp, Stage 2 warm-started, k_max=12, Stage 1 200ep/Stage 2 300ep),
# but with BOTH the aux/Stage-2 propagator AND the ViT encoder/decoder
# made ~20% bigger -- two NEW capabilities added today:
#   --aux-hidden 141 --aux-blocks 3: propagator hidden=141/n_blocks=3
#     gives exactly 1.2x the hidden=128 default's parameter count
#     (133853 vs 111532), matching the precedent already established for
#     Stage 2's own --hidden 141 flag.
#   --d-model 108: encoder/decoder d_model=108 (n_heads=4/n_blocks=3/
#     mlp_ratio=4 unchanged) gives 1011690 total params vs. the d_model=96
#     default's 816342 (~1.24x, the closest achievable ratio to 1.2x with
#     d_model divisible by n_heads=4).
# Both verified directly (real 1-epoch full-profile run, checkpoint
# parameter counts inspected) before launching this real run -- an
# earlier attempt at this section only scaled the propagator and had to
# be killed and redone once that was caught.
# w_var=0.025, w_logdet=0.0045, w_spatial (signed)=0.01.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section53_mlpmarkovian_bigger12x_dmodel108_wvar0025_wspatialsigned01_logdet0045_200ep

echo "=== [1/3] Stage 1: ViT AE (d_model=108, ~1.24x) + mlp/markovian aux (hidden=141/n_blocks=3, 1.2x), w_decorr=0, w_var=0.025, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.0045, NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --d-model 108 --aux-hidden 141 --aux-blocks 3 \
  --w-decorr 0 --w-var 0.025 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0045 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own (1.2x bigger) mlp/markovian aux, NO delta_cap, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 53 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
