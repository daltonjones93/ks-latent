#!/bin/zsh
# User-directed 2026-09-24: "I also wonder if something changed in the
# way we're setting these models up since section 52, that's the last
# test I remember seeing this be successful."
#
# Context: Section 52's own historically-measured result (Gate 3/4,
# artifacts/analysis_suite_full_section52_..._warmstart_k12_300ep.json)
# was genuinely good: D_KY=21.42, lambda1=0.083, matching the true L=100
# target (21-24) almost exactly -- a REAL, reproducible-looking result,
# unlike Section 140's unverified summary. Key finding while investigating
# this: Section 52 ALSO used a fully-GLOBAL --aux-backbone mlp propagator
# (same as Section 211's failed attempt) -- so "give it a local
# propagator" was never really testing what made Section 52 different.
# The two real candidate differences are (a) ENCODER TYPE (Section 52:
# --encoder vit, d_latent=44; Sections 211/212: local_field, d_latent=96)
# and (b) whatever has changed in the SHARED training code (loops.py,
# propagator.py, config.py defaults) between 2026-09-01 (Section 52) and
# today, which would affect EVERY architecture, not just local_field.
#
# This section reruns Section 52's EXACT original Stage 1 CLI command,
# unchanged in every flag, against TODAY's codebase -- the cleanest
# possible regression test: if this still reproduces D_KY~21-24, nothing
# broke in the shared code and the local_field failures are specific to
# that encoder/d_latent; if it does NOT reproduce, something genuinely
# regressed in shared code that affects all architectures.
#
# Verified via a real (--epochs 2) dry run before this launch: recon
# still at ~0.98 after 2 epochs (the known collapse-PLATEAU documented
# in CLAUDE.md brief §5.1 -- escape only happens after ~3000-4000
# gradient steps, i.e. many more epochs; 2 epochs is not enough to judge
# anything, this is expected, not a red flag on its own).
#
# Queued to run AFTER Section 212 finishes (sequential, not parallel,
# to avoid contending for the same MPS device -- this session's own
# established discipline).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section213_ks_vit_section52_repro

echo "=== [1/2] Stage 1: Section 52's EXACT original command -- vit/markovian aux, w_decorr=0, w_var=0.02, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.0035, NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/2] Stage 1 diagnostics: reconstruction quality + a genuine 2000-step standalone Lyapunov check (SAME rigor applied to the local_field attempts) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

DATASET = 'artifacts/datasets/stage1_trajectories_dtsnap1.h5'
ae, ae_cfg, ae_ckpt = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, prop_ckpt = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent)
print('val_recon_final (from checkpoint):', ae_ckpt.get('val_recon_final'))

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
print('(true L100 target: D_KY in [21,24]; Section 52 historical Stage-2 result: D_KY=21.42, lambda1=0.083)')
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 213 Stage 1 complete ==="
