#!/bin/zsh
# User-directed 2026-09-24: "I would like to run 211 again with the new
# D3 regularizer to see if we can encourage more localization
# structure. you can run this on mps at the same time [as Gate 3/4 on
# 211]."
#
# Exact copy of Section 211's Stage 1 command (local_field encoder at
# its own default sizing -- n_sites=32, local_channels=3, d_latent=96
# -- + plain mlp propagator, markovian, Section 52's regularizer recipe:
# w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035) with ONE addition:
# --w-jacobian-bandedness 0.05 (Section 214's new D3-loss --
# ks_latent.training.losses.propagator_jacobian_bandedness_loss),
# bandwidth=3.0 (matching w_spatial's own default bandwidth, no
# established D3-specific precedent yet -- this is a first attempt, not
# a tuned value). Weight chosen conservatively (0.05, well below
# w_spatial's 0.01... actually ABOVE it, since D3 is a much sparser
# once-per-epoch signal vs w_spatial's every-batch one -- see Stage2/
# Stage1TrainingConfig.w_jacobian_bandedness's docstring for the
# once-per-epoch/expensive-Jacobian convention this shares with
# w_spectrum_shape) specifically BECAUSE this mechanism is completely
# untested at any real weight -- sweep further if this under- or
# over-shoots.
#
# Diagnostics measure BOTH the standard standalone-rollout stability
# check (the exact test that caught every prior local_field checkpoint's
# problems) AND D3's own post-hoc bandedness p-value directly
# (ks_latent.analysis.diagnostics.coupling_graph_diagnostic, the SAME
# correctly-calibrated entry-shuffle statistic scripts/run_diagnostics.py
# Gate 4 reports) -- for a direct before/after comparison against
# Section 211's own Gate 4 result (running in parallel, same launch),
# without waiting for the full Gate 3/4 suite to finish.
#
# Verified via a real (--epochs 2) dry run before this launch: recon
# 0.487 -> 0.030 over 2 epochs (val_recon_final=0.0109), ~5s/epoch, no
# error.
#
# Stage 1 ONLY this launch (matching this arc's own "check before
# going further" discipline, now doubly justified: Section 211 itself
# needed Stage 2 to reach a genuinely bounded attractor from an
# over-chaotic Stage 1 -- this run's own Stage-1-alone numbers should
# NOT be judged against the [21,24] target directly, only its D3
# bandedness score/p-value against Section 211's own Stage-1 baseline,
# and its standalone stability against the same rollout test).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section215_ks_localfield_jacobian_bandedness

echo "=== [1/2] Stage 1: local_field encoder (n_sites=32, local_channels=3, d_latent=96) + plain MLP propagator (markovian), Section 52's regularizers PLUS --w-jacobian-bandedness 0.05 (D3 loss, Section 214), L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --w-jacobian-bandedness 0.05 --jacobian-bandedness-bandwidth 3.0 --jacobian-bandedness-n-samples 32 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/2] Stage 1 diagnostics: reconstruction, standalone stability/Lyapunov, AND D3's own post-hoc bandedness p-value (direct comparison to Section 211's Stage-1 baseline) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.analysis.diagnostics import coupling_graph_diagnostic

DATASET = '$DATASET'
ae, ae_cfg, ae_ckpt = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, prop_ckpt = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent, ' n_sites:', ae_cfg.n_sites, ' local_channels:', ae_cfg.local_channels)
print('val_recon_final (from checkpoint):', ae_ckpt.get('val_recon_final'))

with h5py.File(DATASET,'r') as f:
    trajectories = torch.tensor(f['trajectories'][:60], dtype=torch.float32)
    traj_val = trajectories[50:60]
    traj3 = trajectories[53]

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
print('(true L100 target: D_KY in [21,24] -- NOT directly applicable to a Stage-1-only checkpoint, see this script header)')

# D3: same construction as scripts/run_diagnostics.py's own Gate 4.
import numpy as np
rng = np.random.default_rng(0)
n_runs, T, NX = trajectories.shape
run_idx = rng.integers(0, n_runs, size=150)
t_idx = rng.integers(0, T - 1, size=150)
with torch.no_grad():
    u_prev = trajectories[run_idx, t_idx]
    u_curr = trajectories[run_idx, t_idx + 1]
    z_prev = ae.encode(u_prev)
    z_curr = ae.encode(u_curr)
d3 = coupling_graph_diagnostic(prop.step, z_prev, z_curr, n_null=1000, seed=0)
print('D3 bandedness_observed:', d3.bandedness_observed, ' p_value:', d3.bandedness_p_value)
print('(Section 211 baseline, same statistic, from Gate 4 -- see artifacts/logs/gate4_diagnostics_section211_ks_localfield_section52_regs_warmstart_k12_300ep.log)')
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 215 Stage 1 complete ==="
