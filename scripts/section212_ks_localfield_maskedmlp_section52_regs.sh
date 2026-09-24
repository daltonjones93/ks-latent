#!/bin/zsh
# User-directed 2026-09-24: "use the masked mlp for the next test for
# the propagator. that setup seemed to work. I also wonder if something
# changed in the way we're setting these models up since section 52,
# that's the last test I remember seeing this be successful. launch the
# new experiment then look into it."
#
# Context: Section 211 (local_field + fully-GLOBAL mlp propagator,
# Section 52's regularizers) was badly over-chaotic and apparently
# unbounded (D_KY=74.93, max|z| climbing monotonically across the whole
# 2000-step rollout, never settling) -- the THIRD failure in a row on
# that exact encoder/propagator pairing, across three different
# regularizer recipes. Diagnosis: the encoder has locality architecturally
# baked in (circular convs, gauge anchor), but a fully-connected global
# MLP propagator has zero respect for that structure -- nothing bounds
# how much energy it can pile into one direction over a long
# autoregressive rollout.
#
# --aux-backbone masked_mlp (NOT masked_mlp_expand, the backbone every
# one of the 135-159 arc's diverging checkpoints used): `ks_latent.
# models.propagator._MaskedMLPDeltaBody`, a genuinely local,
# WEIGHT-SHARED, circular-band-masked propagator -- every Linear layer
# stays at d_latent width and only connects each position to others
# within +-attn_window on the d_latent ring (MaskedLinear, exactly-zero
# gradient outside the band, per that class's own docstring). This is
# architecturally the direct propagator-side analogue of what the
# encoder already does -- a genuinely different structural hypothesis
# from Section 211's global MLP, not a regularizer retune.
#
# --attn-window 18: local_field's own encoder receptive field is
# mix_radius(2) * n_mix_layers(3) = 6 SITES each direction; at
# local_channels=3 and site-major flattening, that's roughly 6*3=18
# raw d_latent-index units -- chosen to give the propagator a
# comparable physical receptive field to the encoder's own, not
# separately tuned/swept yet.
#
# Everything else identical to Section 211: local_field at its own
# default sizing (n_sites=32, local_channels=3, d_latent=96),
# mode=markovian, Section 52's regularizers (w_var=0.02, w_spatial=0.01
# signed, w_logdet=0.0035), L=100/NX=256, dt_snap=1.0, 200 epochs.
#
# Verified via a real (--epochs 2) dry run before this launch: recon
# 0.495 -> 0.026 in 2 epochs (val_recon_final=0.0146), ~4.7s/epoch, no
# error.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section212_ks_localfield_maskedmlp_section52_regs

echo "=== [1/2] Stage 1: local_field encoder (n_sites=32, local_channels=3, d_latent=96, own default sizing) + LOCAL masked_mlp propagator (markovian, attn_window=18), Section 52's regularizers (w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035), L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone masked_mlp --mode markovian \
  --attn-window 18 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/2] Stage 1 diagnostics: reconstruction quality + a genuine 2000-step standalone Lyapunov check ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, ae_ckpt = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, prop_ckpt = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent, ' n_sites:', ae_cfg.n_sites, ' local_channels:', ae_cfg.local_channels, ' attn_window:', prop.cfg.attn_window)
print('val_recon_final (from checkpoint):', ae_ckpt.get('val_recon_final'))

DATASET = '$DATASET'
with h5py.File(DATASET,'r') as f:
    traj_val = torch.tensor(f['trajectories'][50:60], dtype=torch.float32)
    traj3 = torch.tensor(f['trajectories'][53], dtype=torch.float32)

with torch.no_grad():
    u_hat, z_all = ae(traj_val.reshape(-1, 256))
    recon_mse = ((u_hat - traj_val.reshape(-1,256))**2).mean().item()
print('held-out recon MSE:', recon_mse)

with torch.no_grad():
    z20 = ae.encode(traj_val.reshape(-1,256)).reshape(10, 251, d_latent)
z0b = z20[:,0,:]
with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=2000)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('multi-IC (10) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,50,100,300,600,1000,1500,1999]:
    if t < traj_roll.shape[1]:
        print('t=%d  max|z| across all 10 ICs: %.4f' % (t, maxz_per_t[t].item()))
final_states = traj_roll[:,-1,:]
pairwise = torch.cdist(final_states, final_states)
print('final-state pairwise distance (min excl. diag, max, mean):',
      (pairwise + torch.eye(10)*1e9).min().item(), pairwise.max().item(), pairwise.mean().item())

with torch.no_grad():
    z01 = ae.encode(traj3[:2]).numpy()
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=1.0, warmup_steps=200, seed=0, max_abs_state=1e4,
)
print('lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)
print('(true L100 target: D_KY in [21,24])')
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 212 Stage 1 complete ==="
