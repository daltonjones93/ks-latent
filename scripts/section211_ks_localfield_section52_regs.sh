#!/bin/zsh
# RESULT (2026-09-24): still badly over-chaotic and apparently unbounded
# -- D_KY=74.93, n_positive=39/96, lambda1=0.00137. max|z| does not
# settle onto any attractor: it climbs monotonically and unboundedly
# across the whole 2000-step rollout (8.07 -> 1019.30, still rising at
# the end, would likely have crossed the 10000 divergence cutoff given
# more steps). Reconstruction is excellent (recon MSE 4.06e-5) -- this
# is purely a standalone-dynamics problem, not a representation
# problem. Third failure in a row on the SAME pairing (local_field
# encoder + fully-GLOBAL mlp propagator), across three different
# regularizer recipes (the undocumented 135-159 arc's own, Section
# 140's, now Section 52's) -- points at something structural in pairing
# a genuinely local encoder with a propagator that has zero
# architectural respect for that locality (nothing bounds how much
# energy a global MLP can pile into one direction over a long
# autoregressive rollout, especially at d_latent=96). Superseded by
# scripts/section212_ks_localfield_maskedmlp_section52_regs.sh
# (--aux-backbone masked_mlp instead of mlp -- a genuinely LOCAL,
# weight-shared, circular-band-masked propagator matching the
# encoder's own spatial structure, user-directed: "use the masked mlp
# for the next test... that setup seemed to work").
#
# User-directed 2026-09-23: "sure, do that. keep the regularizers the
# same as section 52." -- after finding that NONE of the 19 historical
# KS local_field checkpoints on disk (Sections 135-159) survive a
# rigorous, consistent, TODAY-verified 2000-step standalone Lyapunov
# check: 11 diverge outright (max|state| > 10000, all the
# masked_mlp_expand-propagator variants), 5 are wildly over-chaotic
# (D_KY 32-60 against a true target of 21-24 -- including Section 140,
# which a 2-week-old undocumented summary had called "the best result
# of this whole document" at D_KY=22.54, re-measuring at D_KY=36.19
# today), 2 fully collapsed (D_KY=0.00), 1 unbracketable. None land
# anywhere near the target. That whole Sept-9 arc was apparently never
# stress-tested with a genuine long free rollout the way this session
# has insisted on all along for L96.
#
# This section trains a FRESH local_field checkpoint on the CURRENT
# (bug-fixed, enc_out bias=False -- see KSAutoencoderLocalField's own
# docstring for the artifact this fixes) architecture, using:
#  - encoder=local_field at its own established default sizing
#    (n_sites=32, local_channels=3, mix_radius=2, n_mix_layers=3,
#    hidden=32 -> d_latent=96 -- exactly Section 136/140's own "p32c3"
#    sizing, CLAUDE.md brief's own recommended starting point).
#  - aux-backbone=mlp, mode=markovian -- the ONLY local_field propagator
#    variant among all 19 candidates that did NOT diverge outright
#    (Section 136); masked_mlp_expand is excluded entirely here.
#  - NO pde_head, NO w_smooth, NO masked_mlp_expand joint-training
#    complexity -- every one of those in the 135-159 arc diverged.
#  - Regularizers: Section 52's own recipe (w_var=0.02, w_spatial=0.01
#    signed, w_logdet=0.0035, w_decorr=0, w_var_floor=0) -- this is the
#    SAME recipe already validated across many other sections this
#    session (198/199/201/203, L96 side) as a safe, well-behaved
#    baseline, INSTEAD of whatever ad hoc weights (w_logdet 0.008/
#    w_spatial 0.04/0.08/w_smooth 0.003-0.01) the undocumented 135-159
#    arc used, none of which held up under scrutiny.
#
# Verified via a real (--epochs 2) dry run before this launch: recon
# 0.488 -> 0.026 in 2 epochs (val_recon_final=0.013), ~4.3s/epoch on
# this machine, no error.
#
# Stage 1 ONLY this launch (matching this session's own established
# "check chaos survives a genuine long standalone rollout before
# trusting anything built on top of it" discipline) -- diagnostics run
# immediately after training, using the EXACT SAME 2000-step Lyapunov
# check that caught all 19 historical checkpoints' problems, before any
# decision to proceed to Stage 2 or the DA localization experiment.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section211_ks_localfield_section52_regs

echo "=== [1/2] Stage 1: local_field encoder (n_sites=32, local_channels=3, d_latent=96, own default sizing) + plain MLP propagator (markovian), Section 52's regularizers (w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035), L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/2] Stage 1 diagnostics: reconstruction quality + a genuine 2000-step standalone Lyapunov check (the EXACT test that caught all 19 historical local_field checkpoints' problems) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, ae_ckpt = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, prop_ckpt = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent, ' n_sites:', ae_cfg.n_sites, ' local_channels:', ae_cfg.local_channels)
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

echo "=== Section 211 Stage 1 complete ==="
