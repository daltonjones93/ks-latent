# KS L=100 Latent DA + Manifold Analysis — Project Handoff

Handoff note for continuing this project in a fresh chat. Upload this file plus
the canonical scripts you're actively editing, and give a one-line reminder of
the current task. Nothing carries over automatically between chats — the
uploaded files are the source of truth.

---

## What the project is

Two-stage fast surrogate + latent data assimilation for the 1D Kuramoto–
Sivashinsky equation at **L=100, NX=1024, d_latent=44**.

- **Stage 1** — patched-transformer autoencoder (physical field ↔ latent z).
- **Stage 2** — latent propagator (advances z one step from a 2-step history).
- **DA** — Natural-Gradient Particle Flow Filter (NAT-PFF) operating in latent
  space; all observation I/O in physical units.

Environment: Apple Silicon (MPS), run via Spyder `%runfile`. Spyder lives in the
`work`/`base` env with the Python interpreter pointed at the `torch` env; that
env has torch, scipy, gudhi, giotto-tda, ripser, spyder-kernels. `pip install`
needs `--break-system-packages`. `matplotlib.use('Agg')` at top of DA scripts.

Workflow: user runs locally, uploads canonical files + result PNGs; assistant
edits in /home/claude, copies to /mnt/user-data/outputs, presents for download.
IMPORTANT: always sync to the user's uploaded files before editing (rebuilding
from memory has caused file-state drift). Do NOT switch models mid-project.

---

## Canonical scripts (the working codebase)

DA pipeline:
- `pff.py` — ParticleFlowFilter class (DET/STO/NAT methods, NAT default).
- `da_pff.py` — DA driver: loads models, generates truth, runs cycles, plots.
- `plot_da_spacetime.py` — Hovmöller / spacetime plots.
- `plot_pff_convergence.py` — per-cycle PFF convergence diagnostics.

Models + training:
- `autoencoder_patched.py` — KSAutoencoderPatched (Stage 1).
- `propagator.py` — LatentPropagator (Stage 2).
- `train_stage1_patched.py`, `train_stage2_patched.py` — training scripts.

Latent-space analysis:
- `latent_sfa.py` — Slow Feature Analysis rotation (viz only).
- `latent_reorder.py` — seriation + DMD reordering (viz only).
- `latent_manifold.py` — intrinsic dimension (two-NN, corr dim) + diffusion maps.
- `latent_lyapunov.py` — Lyapunov spectrum + Kaplan–Yorke dimension.
- `latent_topology.py` — persistent homology (plain Rips).
- `latent_topology_dtm.py` — DTM-filtration persistent homology (density-robust).

Checkpoints (in working dir):
- `stage1_ae_patched_n1024_d44.pt`, `stage2_prop_patched_n1024_d44.pt`
- `sfa_rotation.pt`, `dmd_rotation.pt`, `seriation_order.pt`
- `lyapunov_exponents.npy`

---

## Architecture summary

**Stage 1 autoencoder** (d_model=128, nhead=4, dim_ff=128, GELU, pre-norm):
- Encoder: PatchEmbed MLP (8→256→128 per patch, 128 patches) → local transformer
  within 16 groups of 8 (shared weights) → mean-pool → 16 group tokens → prepend
  8 query tokens → global transformer (2 layers, 24 tokens) → keep 8 queries →
  Linear(8·128→44) = z.
- Decoder: learned 128-token query bank + broadcast-add Linear(44→128) of z →
  transformer (2 layers) → PatchUnembed Linear(128→8) per token → 1024 field.
- Only the patch embed is an MLP; enc_to_latent, dec_from_latent, PatchUnembed,
  propagator in/out projections are single Linear layers.

**Stage 2 propagator** (n_blocks=3, dim_ff=128, hidden=128):
- Input (z_{n-1}, z_n) ∈ R^88 → Linear(88→128) → 3× ResidualMLPBlock (pre-norm:
  x + [LN→Linear→GELU→Dropout→Linear]) → LN → Linear(128→44) = delta →
  z_{n+1} = z_n + delta.
- Design: (1) delta/residual param (persistence prior); (2) zero-init output head
  (identity at init); (3) optional rollout step_noise (error recovery).
- Jacobian ∂z_{n+1}/∂z_n = I + ∂delta/∂z_n → near-identity → small Lyapunov exps.

---

## Loss functions

**Stage 1** (window of 4 snapshots, shift-augmented):
- L_recon = mean over 4 snapshots of ||decode(encode(u)) − u||²/NX.
- L_pred = aux propagator (n_blocks=2, dim_ff=64, trained JOINTLY) rolls
  (z_0,z_1)→ẑ_2,ẑ_3, decoded, vs u_2,u_3. This term flows gradients into the
  encoder → shapes latent to be dynamically predictable (WHY Stage 2 works and
  Lyapunov is clean).
- L_decorr = (1/d²) Σ_{i≠k} C_ik²  (off-diagonal covariance).
- L_var = (1/d) Σ (C_ii − 1)²  (unit-variance diagonal).
- Total: 1.0·recon + 0.5·pred + 0.01·decorr + 0.01·var.
- decorr+var drive C→I → B≈I (WHY natural-gradient PFF is well-conditioned and
  WHY linear reordering methods find no variance structure).
- AdamW lr=1e-3, cosine, 60 epochs, batch 128, grad clip 1.0.

**Stage 2** (frozen AE; 4× shift-augmented latent trajectories):
- Horizon-weighted MSE: L2 = Σ_h w_h e_h, w_h = γ^h/Σγ^h (γ=1 → uniform → mean
  over k rollout steps). e_h = mean squared latent error at horizon h.
- K-curriculum: k ramps 2→16 over first 8 epochs. Input noise annealed 0.10→0.02
  (matches DA-time MODEL_NOISE_STD → WHY DA is robust).
- AdamW lr=3e-4, cosine, 20 epochs, batch 256, best-val-k checkpointing.

---

## PFF / DA key decisions (all in canonical pff.py / da_pff.py)

- Single `ParticleFlowFilter` class. Diagnostic mode auto-activates when
  decoder_phys provided. Methods DET / STO / **NAT (default)**.
- B = PRIOR covariance, computed once from forecast ensemble, FIXED through
  pseudo-time.
- NAT Gauss–Newton metric F = J̄ᵀ R⁻¹ J̄ + B⁻¹ using ensemble-MEAN Jacobian J̄.
  Kernel uses F as metric (Option B). Divergence simplifies via F/F⁻¹ cancellation.
  Prior term −F⁻¹B⁻¹(z^j − z̄). Stochastic term η = √2·L_precond·N_mat·L_kᵀ with
  precond=F⁻¹, using k_bar/N inside Cholesky. robust_cholesky() with CPU float64
  eigh fallback (MPS lacks eigh).
- ds schedule (NAT branch), NO ds_min anywhere:
  ```
  ds_test = min(max(1.0, 1.0/(f.norm()+1e-8)), ds_max)
  if s==0: ds = 1.0
  if ds_test > ds: ds = min(1.1*ds, ds_test)
  else:            ds = max(0.9*ds, ds_test)
  ```
- Convergence early-stop: tol=1e-4 after s>5 on rel change of ensemble mean;
  n_steps default 100.
- **MODEL_NOISE_STD = 0.08** injected at EACH propagator step in
  forecast_ensemble: `z = z + MODEL_NOISE_STD*torch.randn_like(z)`. Tuned so
  forecast spread ≈ analysis RMSE (calibration). Free run gets NO noise.
- RMSE normalization: both rmse_free and rmse_da divide vector norm by sqrt(d)
  → per-dimension RMSE, consistent with spread and convergence-plot RMSE.
- OBS_NOISE=0.1, R physical. vmap/jacrev Jacobians on CPU (MPS gaps).

Latest DA results (MODEL_NOISE_STD=0.08, per-dim RMSE): free run latent RMSE
~1.6/dim, DA ~0.14/dim, spread ~0.11 (well calibrated), skill ~ order 10×.

---

## Spacetime plots (RESOLVED — overwrite scheme)

- Per DA cycle: segment = N_PROP_STEPS rows; LAST forecast row OVERWRITTEN by
  analysis (u_seg[-1]=u_analysis). Truth: u_truth_full[1:], plain N_PROP_STEPS/cyc.
- Only the FULL-resolution plot is produced (per-cycle low-res removed as
  redundant). Saved per cycle: spacetime_full_cycleNNN.png,
  spacetime_latent_full_cycleNNN.png (cycle in filename → not overwritten).
- Full plots use explicit edges + shading='flat'. t_edges = arange(T_full+1)+0.5
  so data row i (physical time i+1) is centered on time i+1; analysis at row 39
  centered on t=40. Boundary lines t_cyc = arange(1,n_cyc+1)*n_prop_steps.
- u_da_history uses decode(mean(z)) (NOT mean(decode(z))) so per-cycle and full
  plots are consistent through the nonlinear decoder.

---

## Manifold analysis — CONCLUSIVE RESULTS

User hypothesis (CONFIRMED): the 44-dim latent is not 44 independent dynamics;
it encodes a ~22-dim attractor (KS Kaplan–Yorke dimension).

Hard-won sampling lesson (latent_manifold.py): temporal correlation corrupts
dimension estimates. Consecutive trajectory points are locally low-dim; thinning
doesn't fix it. ONLY 1 point per run is truly independent. FINAL scheme:
N_RUNS≈10000, one RANDOM on-attractor point per run, each run its own KS spin-up
+ SPINUP_DISCARD=100. Sample-size convergence curve then plateaus.

Estimators (validated on synthetic known-dim data):
- two-NN (Facco 2017): fit central 90% through origin, midpoint CDF (i-0.5)/N.
  Known downward bias at high d (true 22 → ~19). Reads ~18-19 on latent → ~22.
- Correlation dimension: lower bound (~14 on latent, underestimates).
- Diffusion maps: smooth spectrum, no gap → high-dim (no 2-3D structure).

**Lyapunov spectrum (latent_lyapunov.py) — headline result:**
- Benettin algorithm via torch.func.jvp; validated to machine precision on a
  known linear map (needs bounded reference; KS attractor is bounded).
- **D_KY ≈ 21.4**, **11 positive exponents**, cluster near zero ~index 12-14,
  clean negative tail. λ₁ ≈ 0.01/step.
- NOTE: current script uses z→z single-state approx; true phase space is 2d-dim
  (2-step history). Exact spectrum would use the (z_{n-1},z_n) tangent map.

**Topology (latent_topology.py plain Rips; latent_topology_dtm.py DTM):**
- Attractor is topologically TRIVIAL: single connected component (H0 merges
  ~scale 7), H1/H2 all hug the diagonal (noise, not loops/voids).
- DTM (Distance-to-Measure) version added for density-robustness (Gemini
  suggested; assistant had missed it). Two probes: raw 44-dim (sampling-limited
  above H0 at d~22) and diffusion embedding (6-dim, where H1/H2 are resolvable).
  Mass sweep {0.02,0.05,0.1}, H0/H1/H2. Validated on circle (H1) and torus (H1×2).
- DTM RESULT (CONFIRMED, both probes): H0 lifts off the diagonal (connected-blob
  structure), H1 and H2 hug the diagonal at ALL masses and are STABLE across the
  mass sweep — no loop or void ever lifts off to become robust. Holds in both the
  raw 44-dim cloud AND the 6-dim diffusion embedding (the probe where H1/H2 ARE
  resolvable, so travelling-wave phase circles WOULD show up if present). This is
  now a DEFENSIBLE negative result: outlier/density-robust method, its one free
  parameter swept with no change → attractor is genuinely topologically trivial,
  not a sampling artifact. (Birth scales differ between probes — ~0.1-0.25 for
  diffusion coords vs ~10-22 for raw latent — purely the natural distance scale
  of each space, not physical; the diagram shapes are what matter.)
- All features above the diagonal is a mathematical necessity (death≥birth);
  significance = distance FROM diagonal = lifetime. Here all near diagonal.

SFA / DMD / seriation (latent_sfa.py, latent_reorder.py): all viz-only, all
unimpressive — no clean bands, DMD freqs tiny (0-0.012), because the decorr loss
removed 2nd-order structure and travelling-wave coherence lives in PHYSICAL space.

**Synthesis:** every method agrees — latent attractor is a ~22-dim, connected,
topologically-trivial, curved blob with 11 unstable directions. 44 ≈ 2×22 =
Whitney/Sauer-Yorke-Casdagli embedding prediction. This is real high-dim
structure, not visualizable — which is why all 2D/3D viz "failed."

---

## Literature comparison (physical space, L=100)

- Edson, Bunder, Mattner & Roberts 2019 (arXiv:1902.09651, ANZIAM J.): KS
  Lyapunov spectrum; D_KY ≈ 0.226·L − c (linear/extensive for L≳80); exponents
  bounded above ~0.1. → L=100 gives D_KY ~22-23.
- Koopman paper arXiv:1909.00076 states D_KY = 23.2 for L=100 explicitly.
- Inertial manifold: KS has a finite-dim smooth exponentially-attracting
  invariant manifold containing the attractor (textbook example). Dimension ≪
  simulation modes → exactly what the autoencoder exploits.
- Physical vs Kaplan-Yorke: covariant-Lyapunov-vector "physical dimension" <
  D_KY. For L=22, inertial-manifold dim = 8 (Ding/Cvitanović; Yang et al.).
- UPO skeleton: attractor organized by unstable periodic orbits — chaos as a
  walk on the inertial manifold chaperoned by nearby UPOs.

**Model vs literature:** your D_KY ≈ 21.4 vs literature ~22-23 → model reproduces
the true attractor dimension to ~5-8%. Strong validation that the learned latent
dynamics are faithful to the real physics.

Embedding references (verified): Whitney 1936 Ann. Math. 37(3):645-680; Whitney
1944 Ann. Math. 45:220-246; Takens 1981 LNM 898:366-381; Sauer-Yorke-Casdagli
1991 J. Stat. Phys. 65(3-4):579-616 (best cite for "44≈2×22").

---

## Pending / possible next steps

- [DONE] latent_topology_dtm.py run — DTM diagrams (raw + diffusion, mass sweep)
  confirm topologically-trivial attractor (defensible negative result).
- Optional: exact Lyapunov spectrum using the 2-step (z_{n-1},z_n) tangent map
  instead of the z→z single-state approximation.
- Optional: covariant Lyapunov vectors in latent space → "physical dimension"
  vs D_KY comparison (publishable, benchmarks against L=22 d_M=8 result).
- Optional: recurrence analysis / UPO search in latent space (attractor skeleton).
- Confirm DA calibration after MODEL_NOISE_STD=0.08 + per-dim RMSE rerun.
