# CLAUDE_CODE_BRIEF.md — Build spec for `ks-latent`

**Read this whole file before writing any code.**

You are building a research codebase from scratch. It has two jobs:

1. **Reproduce** an existing set of results on latent-space data assimilation and
   manifold analysis for the Kuramoto–Sivashinsky (KS) equation. These currently live in
   an undocumented pile of scripts; we are rebuilding them properly so they are testable,
   configurable, and trustworthy.
2. **Rigorously test a set of new research hypotheses** about whether the learned latent
   space can be given spatial structure, and whether an effective PDE (as opposed to a
   system of coupled ODEs) exists for it.

Job 1 is not a warm-up. If the reproduction targets in §18 are not hit, every conclusion
from job 2 is worthless.

### Reference documents — read in this order

- `docs/PROJECT_HANDOFF.md` — canonical description of the existing results.
- `docs/ML_for_KS_writeup.md` — architectures and loss derivations. **Note:** describes
  an older configuration (`NX=128`, `d=24`, stride 5). The handoff's own canonical config
  is (`L=100`, `NX=1024`, `d=44`). Make both configurable. **Update, 2026-08-29 (see
  §5.1 addendum and `docs/RESULTS.md`): this codebase's actual default is now
  `NX=256`, not the handoff's `NX=1024`** — training reliably collapsed at NX=1024
  and never recovered within a 60-epoch budget, while NX=256 escapes the same
  collapse (~4-5x faster per epoch, no solver-accuracy cost at this domain size).
  `L=100`, `d=44`, and the handoff's architecture/loss recipe are otherwise unchanged.
- `docs/LATENT_PDE_RESEARCH_NOTES.md` — scientific rationale.
- `docs/LATENT_PDE_RESEARCH_NOTES_addendum_2026-08-28.md` — **supersedes parts of the
  main notes.** §12.1 identifies a structural flaw in the h-refinement test as originally
  written; §14.3 gives a revised priority order that this brief follows. Where the two
  documents conflict, **the addendum wins.**
- `reference/latent_locality.py`, `reference/test_locality.py` — a working, tested
  prototype of the constrained optimizer. Now a **contingency** path (Phase X), not part
  of the main line. Read the docstrings; they encode findings you must not re-derive.

---

## 0. Ground rules

These are binding.

1. **No untested numerics.** Every function computing a physical or statistical quantity
   gets a test against an analytic answer, a synthetic system with a known answer, or a
   published number. "It runs" is not evidence.
2. **Fail loudly.** No silent `try/except` around numerical code, no `nan` swallowing.
   If a Cholesky fails, raise with the condition number in the message.
3. **Determinism by default.** Every entry point takes a seed. Two CPU runs with the same
   seed and config must be bitwise identical. MPS does not guarantee this (§1.3) — the
   bitwise test is CPU-only; MPS gets a loose-tolerance reproducibility test.
4. **Config over constants.** Everything in a dataclass config, serialized alongside
   every checkpoint and result.
5. **Smoke path.** Every experiment script supports `--profile smoke`: the full code path
   in under 60 s on CPU at tiny dimensions. CI runs smoke; real runs are launched
   manually.
6. **When a replication target is missed, stop and report.** Do not tune until it matches
   and then move on silently. Write it up in `docs/REPLICATION_LOG.md` and flag it.
7. **Record provenance.** Every artifact gets a sidecar `.meta.json`: git SHA, config
   hash, seed, device, timestamp, wall time, resolved config.
8. **Gates are real.** Phases end with a gate. Report at the gate and wait. If a gate
   fails, write it up rather than working around it.

---

## 1. Environment: use the existing `da_env`

**Do not create a new conda/mamba environment.** The user has a working `da_env` managed
by mamba, already exercised against this pipeline.

### 1.1 What to do

```bash
mamba activate da_env
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Write `scripts/check_env.py` that verifies every import the codebase needs and prints a
table of `package | found | version | required`. Run it first. For anything missing:

```bash
mamba install -n da_env -c conda-forge <pkg>       # prefer conda-forge
pip install <pkg>                                   # only if unavailable on conda-forge
```

Likely already present (per the handoff): `torch`, `scipy`, `numpy`, `matplotlib`,
`gudhi`, `giotto-tda`, `ripser`, `spyder-kernels`.

Likely needed: `pytest`, `pytest-cov`, `hypothesis`, `pyyaml`, `h5py`, `pandas`,
`scikit-learn`, `pysindy`, `PyWavelets`, `persim`, `mpmath`.

**Note from the handoff:** `pip install` in this setup needs `--break-system-packages`.
Try without it first; if pip refuses, add the flag and record that in the README.

After the first successful full run: `mamba env export -n da_env --no-builds >
environment.lock.yml` and commit it. Also commit an `environment.yml` that would recreate
an equivalent env from scratch, as a portability fallback — but do not use it for this
work.

Install the package editable (`pip install -e .` with a `pyproject.toml`) so imports work
from anywhere, including from Spyder.

### 1.2 Device policy (Apple Silicon / MPS)

This gets its own module, `ks_latent/utils/device.py`. **MPS is float32-only and has real
gaps.** The policy:

| Workload | Device | Precision | Why |
|---|---|---|---|
| KS solver / data generation | **CPU (numpy + scipy.fft)** | **float64** | Done once, offline. ETDRK4 needs float64; a float32 spectral solve at `NX=1024` will not give a trustworthy Lyapunov spectrum. **Never put the solver on MPS.** |
| AE / propagator training | **MPS** | float32 | Where the speedup is. |
| Rollouts, forecasts, inference | **MPS** | float32 | |
| Lyapunov (Benettin QR) | **CPU** | **float64** | QR reorthonormalization is precision-critical; MPS has no float64. Keep the `jvp` on MPS if it is faster, push tangent vectors to CPU for the QR, and measure whether the transfer dominates. |
| Covariance, Cholesky, `eigh`, `svd` | **CPU** | **float64** | MPS lacks `eigh`. `robust_cholesky` / `robust_eigh` in `utils/linalg.py` handle this. |
| `jacrev` / `vmap` Jacobians | **CPU** | float32/64 | MPS has known gaps; the handoff already records doing these on CPU. |
| Dimension estimators, persistent homology | CPU | float64 | Library code is CPU anyway. |
| SINDy / PDE-FIND | CPU | float64 | |

`get_device(prefer="auto")` resolves `mps` → `cuda` → `cpu`. Add
`get_compute_device(task: str)` applying the table, so callers say
`get_compute_device("lyapunov")` rather than hardcoding.

### 1.3 MPS engineering rules

1. **Set `PYTORCH_ENABLE_MPS_FALLBACK=1`** in `ks_latent/__init__.py` *before* importing
   torch, and log a warning listing which ops fell back. A silent CPU fallback in an inner
   loop is a 50× slowdown that looks like "the model is slow."
2. **Avoid host–device syncs in inner loops.** Every `.item()`, `.cpu()`, `float(...)`,
   or Python-side `if tensor > x` forces a sync. Accumulate losses as tensors; call
   `.item()` once per epoch, not once per step. This is the single biggest MPS performance
   mistake and the training loop must be written to avoid it.
3. **Do not use `torch.compile`** — immature on MPS. Revisit later.
4. **Unified memory.** `.to(device)` is cheap relative to CUDA but not free. Preload the
   whole training tensor to MPS once rather than per batch (a `(50, T, 1024)` float32
   dataset is small). Use an index-shuffle sampler over an on-device tensor instead of a
   `DataLoader` with workers — multiprocessing gains nothing here and adds copies.
5. **Larger batches help.** Kernel-launch overhead is relatively high; prefer 128–512
   where memory allows.
6. **`torch.fft` on MPS is partial.** Any FFT in a loss or diagnostic must be tested for
   MPS support and routed to CPU on failure. The solver is on CPU already.
7. **Seeding.** `torch.manual_seed`, `torch.mps.manual_seed`, `numpy`, `random`. Assert
   bitwise reproducibility on **CPU only**; use `rtol=1e-4` on MPS.
8. **Benchmark honestly.** `scripts/bench_device.py` times one training epoch on MPS vs.
   CPU for each model. If MPS is not winning for a given model — it often is not, for
   small models dominated by launch overhead — say so and default that model to CPU. Do
   not assume MPS is faster.
9. **Memory.** `torch.mps.empty_cache()` between phases; log
   `torch.mps.current_allocated_memory()` at epoch boundaries.

`tests/unit/test_device_policy.py` asserts the routing table is respected and that
`robust_eigh` / `robust_cholesky` agree with a direct CPU float64 computation to 1e-10.

---

## 2. Repository layout

```
ks-latent/
├── pyproject.toml            # editable install, ruff + pytest config
├── environment.lock.yml      # mamba env export of da_env, generated
├── environment.yml           # portability fallback only
├── Makefile                  # env-check, test, test-fast, smoke, replicate, bench
├── README.md
├── docs/
│   ├── PROJECT_HANDOFF.md
│   ├── ML_for_KS_writeup.md
│   ├── LATENT_PDE_RESEARCH_NOTES.md
│   ├── LATENT_PDE_RESEARCH_NOTES_addendum_2026-08-28.md
│   ├── REPLICATION_LOG.md    # you maintain, auto-generated table
│   ├── RESULTS.md            # you maintain, with pre-registered decision rules
│   └── OPEN_QUESTIONS.md     # you maintain
├── configs/
├── reference/                # latent_locality.py, test_locality.py
├── ks_latent/
│   ├── config.py
│   ├── solver/               # ETDRK4, datasets, filtering
│   ├── models/               # global AE, local field AE, propagators, neural ODE
│   ├── training/             # loops, losses, constraints, schedules
│   ├── da/                   # PFF, obs operators, localization, SEC
│   ├── analysis/             # dimension, lyapunov, topology, diagnostics, spreading, scaling
│   ├── discovery/            # stencil models, PDE-FIND, RG nesting
│   └── utils/                # device, linalg, seeding, io, circstats, plotting
├── scripts/
├── tests/{unit,integration,replication,golden}/
└── artifacts/                # gitignored
```

---

## 3. Phase 1 — KS solver and data generation

### 3.1 Solver

`ks_latent/solver/ks.py`, **numpy + scipy.fft, float64, CPU.**

```
u_t + u u_x + u_xx + u_xxxx = 0,    x ∈ [0, L),  periodic
```

Integrate with **ETDRK4** (Kassam & Trefethen, *SIAM J. Sci. Comput.* 26:1214, 2005). In
Fourier space with `k = 2π n / L`:

```
v_t = L̂ v + N(v),     L̂ = k² − k⁴,     N(v) = −(i k / 2) · FFT( IFFT(v)² )
```

Compute the ETDRK4 coefficients by the **contour-integral trick** — 32 equally spaced
points on a unit circle centred on each `h·L̂` — not the analytic formulas, which lose
catastrophic precision near `L̂ = 0`. Cache per `(L, NX, dt)`.

```python
@dataclass
class KSConfig:
    L: float = 100.0
    NX: int = 1024
    dt: float = 0.05           # solver step
    snapshot_every: int = 5    # -> dt_snap = 0.25
    spinup_time: float = 500.0
    seed: int = 0
```

**`dt_snap` must be recorded in every dataset's metadata.** Lyapunov exponents,
correlation times, and the light-cone bound in Phase 2 are per-step quantities and are
meaningless without it. Most subtle bugs in this project will come from losing `dt_snap`.

**Note (2026-08-29):** Phase 1's Gate 1 (`test_L100_kaplan_yorke`) was validated and
passed at the `NX=1024` shown above (`docs/REPLICATION_LOG.md`); that result stands
as evidence the solver is correct at that resolution and was not rerun. The actual
`ks_latent.config.KSConfig`/`AutoencoderConfig` class defaults were changed to
`NX=256` afterward, for Phase 3+ (the latent-model training pipeline) only — see
the §5.1 addendum and `docs/RESULTS.md`. Phase 1/2's own scripts still use explicit
`NX=1024` and are unaffected by that default change.

### 3.2 Solver tests (`tests/unit/test_solver.py`)

| test | criterion |
|---|---|
| `test_linear_operator_spectrum` | most unstable mode at `k = 1/√2`, wavelength `2√2π ≈ 8.886`, to 1e-10 |
| `test_etdrk4_coefficients_no_cancellation` | coefficients at `L̂ ≈ 0` match an `mpmath` high-precision evaluation to 1e-12 |
| `test_mean_conservation` | `∫u dx` conserved; drift < 1e-8 over 10⁴ steps |
| `test_translation_equivariance` | `integrate(roll(u₀,c)) == roll(integrate(u₀),c)` to 1e-10 |
| `test_convergence_in_dt` | observed order ≥ 3.5 vs. a `dt/8` reference on a short non-chaotic window |
| `test_small_L_decay` | for `L < 2π`, `u → 0` |
| `test_L22_lyapunov` *(slow)* | `λ₁ ≈ 0.043–0.05`, `D_KY ≈ 5.2–5.6` |
| `test_L100_kaplan_yorke` *(slow)* | **`D_KY ∈ [21, 24]`.** Edson et al. give `D_KY ≈ 0.226·L − c`; the Koopman literature quotes 23.2 at L=100. **This is the gate on Phase 1.** |
| `test_energy_spectrum_peak` | time-averaged `\|û(k)\|²` peaks near `k ≈ 1/√2` |

### 3.3 Datasets

`ks_latent/solver/dataset.py`. Two modes, **not interchangeable**:

- **`TrajectoryDataset`** — long correlated runs for training. 60 trajectories, 50/10
  train/val, independent random ICs, spinup discarded. Normalize to zero mean / unit std
  with statistics computed **once on the training split**, stored in metadata.
- **`AttractorPointDataset`** — for dimension and topology. **One point per independent
  run.** `N_RUNS ≈ 10000`, each with its own spinup (`SPINUP_DISCARD=100` snapshots),
  contributing exactly one randomly chosen on-attractor snapshot. Temporal correlation
  corrupts dimension estimates and **thinning does not fix it.**

Write `test_thinning_does_not_fix_correlation` asserting the failure reproduces: two-NN
dimension from a thinned single trajectory must come out materially below the
independent-runs estimate. If this test ever passes trivially, it went blind.

Also `ks_latent/solver/filtering.py`: a Gaussian low-pass `G_ℓ * u` in Fourier space with
width `ℓ` a parameter. Needed in Phases 8 and 10.

Datasets as HDF5 with a `metadata` group: full `KSConfig`, normalization stats, `dt_snap`,
git SHA.

**Gate 1: `test_L100_kaplan_yorke` passes. Do not proceed otherwise.**

---

## 4. Phase 2 — D7: information spreading velocity and the light cone

*(Addendum §12.5, §13.2. Cheap, no training, and it constrains design choices in Phases
10–11 — which is why it comes this early.)*

`ks_latent/analysis/spreading.py`

In spatiotemporal chaos, perturbations spread at finite velocity (velocity-dependent /
comoving Lyapunov exponents; Deissler & Kaneko 1987; Pikovsky & Politi 1998). Define
`v_*` as the largest velocity with a positive comoving exponent.

Implementation: perturb the physical field locally (a narrow bump at `x₀`), integrate the
**tangent** dynamics alongside the nonlinear trajectory, track the spreading of
`|δu(x,t)|`. Two estimators, report both:

1. **Front tracking.** Locate the outermost `x` where `|δu|` exceeds a threshold relative
   to its peak; fit front position vs. `t`. Sweep the threshold; report sensitivity.
2. **Comoving exponent.** `Λ(v) = lim (1/t) ln |δu(x₀ + vt, t)|` over a grid of `v`;
   `v_* = max{v : Λ(v) > 0}`.

Average over ≥ 100 independent base trajectories and perturbation sites.

**Outputs:**

```
light-cone half-width (physical)      = v_* · Δt,   Δt = dt_snap · stride
required stencil half-width (sites)  ≥ v_* · Δt / h
minimum valid localization radius    ≥ max(encoder receptive field, v_* · Δt)
```

Three consequences to write into `docs/RESULTS.md`:

1. **The stencil width in Phase 11 is a prediction, not a hyperparameter.** If the learned
   stencil ends up *wider* than the light cone, the model has learned spurious
   nonlocality; if narrower, it is under-resolved and rollout error will say so.
2. **The temporal stride is a locality lever.** At stride 5, `Δt` is 5× larger than needed,
   multiplying the required stencil width by 5 and hurting the bandedness of the DA
   Jacobians. There is a real trade-off (smaller `Δt` → more rollout steps per DA cycle →
   more accumulated model error). **Sweep the stride; do not inherit it.**
3. The localization bound above is a small, clean, quotable design rule. Implement it as
   an **assertion** in Phase 13, not a comment.

Tests: exact recovery of the known spreading velocity of a synthetic advection–diffusion
tangent problem; **`v_*` must be stride-independent when expressed in physical units** —
a strong correctness check, because if it is not, `dt_snap` is being lost somewhere.

**Gate 2: `v_*` measured with error bars; derived bounds written to `docs/RESULTS.md`.**

---

## 5. Phase 3 — Baseline models (replication)

### 5.1 Stage 1: global patched-transformer AE

`ks_latent/models/autoencoder_patched.py`, per the handoff. `d_model=128, nhead=4,
dim_ff=128`, GELU, pre-norm.

Encoder: `PatchEmbed` MLP (8 → 256 → 128 per patch, 128 patches) → local transformer
within 16 groups of 8 (**shared weights**) → mean-pool → 16 group tokens → prepend 8
learned query tokens → global transformer (2 layers, 24 tokens) → keep the 8 query
outputs → `Linear(8·128 → 44)` = `z`.

Decoder: learned bank of 128 query tokens + broadcast-add `Linear(44→128)` of `z` →
transformer (2 layers) → `PatchUnembed` `Linear(128→8)` per token → 1024-vector.

Only the patch embedding is an MLP; `enc_to_latent`, `dec_from_latent`, `PatchUnembed`,
and the propagator projections are single `Linear` layers.

Loss (window of 4 snapshots, one common cyclic shift per window):
```
L = 1.0·L_recon + 0.5·L_pred + 0.01·L_decorr + 0.01·L_var
```
- `L_recon` = mean over the 4 snapshots of `‖D(E(u)) − u‖²/NX`
- `L_pred` — a **small auxiliary propagator** (`n_blocks=2, dim_ff=64`, ~22k params)
  trained **jointly**, rolling `(z₀,z₁) → ẑ₂, ẑ₃`, decoded, compared to `u₂,u₃` in
  **physical space**. Physical-space targets are essential: latent-space targets move as
  the encoder trains and produce a known collapse degeneracy. Put that in the docstring.
- `L_decorr = (1/d²) Σ_{i≠k} C_ik²`, `L_var = (1/d) Σ (C_ii − 1)²`
- AdamW `lr=1e-3`, cosine, 60 epochs, batch 128, grad clip 1.0.
- The aux propagator is **discarded** after training.

**Known failure mode (found during replication, 2026-08-29): representation
collapse.** Training this exact recipe at the canonical NX=1024/d=44 scale
converged to a fixed point where the encoder maps almost every input to
nearly the same latent vector (cross-sample `z.std` shrinks from ~0.02 to
~0.003–0.007 over training; `L_var` sits near its worst value, ~1.0,
essentially the whole run; val reconstruction plateaus at ~0.97, the
zero-information baseline for unit-variance data). Six ablations — data
sanity, `L_pred`/`L_decorr`/`L_var` removed (recon loss alone), shift
augmentation on/off, grad-clip on/off, learning rate swept over
{1e-4, 1e-3, 3e-3}, decoder-at-init sensitivity to `z` — each reproduce the
identical collapse and are ruled out as the sole cause. At NX=256 the same
collapse occurs but is a transient, not a fixed point: reconstruction
escapes and drops sharply (~0.98 → ~0.13) after roughly 3,000–4,000
gradient steps; the NX=1024 run only completed ~23,000 steps (60 epochs)
and never reached that point.

Two mechanism hypotheses raised during debugging, not yet confirmed or
ruled out — try both before assuming either fixes it:

1. **`dt_snap` too small.** If consecutive snapshots `u(t)` and
   `u(t + dt_snap)` are nearly identical, `z_0 ≈ z_1` and the two-step
   history the propagator/`AuxProp` is meant to exploit degenerates toward
   a single-state map. Try a larger snapshot stride (larger `dt_snap`) and
   check whether the collapse plateau shortens or disappears.
2. **`L_pred` rollout too short to pressure the encoder.** With `K_pred=2`,
   a collapsed (near-constant) `z` makes `AuxProp`'s output collapse too,
   so `L_pred` degenerates to matching the dataset mean — the same failure
   `L_recon` alone has — providing no extra anti-collapse gradient. A
   longer rollout (`K_pred > 2`) compounds prediction error over more steps
   and may generate a stronger, non-degenerate gradient signal even from a
   collapsed starting point. Test by extending the window and rollout
   length in the joint Stage-1 loss.

Record findings from testing these in `docs/RESULTS.md` /
`docs/OPEN_QUESTIONS.md` before changing the canonical training recipe on
the strength of either hypothesis alone.

### 5.2 Stage 2: latent propagator

`(z_{n-1}, z_n) ∈ R^88 → Linear(88→128) → 3× ResidualMLPBlock (pre-norm:
x + [LN→Linear→GELU→Dropout→Linear]) → LN → Linear(128→44) = δ`, then
`z_{n+1} = z_n + δ`.

- **Zero-init the output head** — exactly the identity at step 0, so an initial 16-step
  rollout is constant rather than exploding. `test_propagator_is_identity_at_init`.
- Training: frozen AE, 4× shift-augmented latent trajectories, horizon-weighted MSE
  `w_h = γ^h/Σγ^h` (γ=1 → uniform). K-curriculum 2→16 over the first 8 epochs. Input noise
  annealed 0.10→0.02 (matches `MODEL_NOISE_STD` at DA time; this is why DA is robust).
  AdamW `lr=3e-4`, cosine, 20 epochs, batch 256, best-val-k checkpointing.

**Alternative to try (raised during replication debugging, 2026-08-29): a
genuinely Markovian map.** The two-step history `(z_{n-1}, z_n)` was
motivated as a "discrete velocity proxy," but the KS PDE itself is first
order in time (`u_t = -u u_x - u_xx - u_xxxx`, no second time derivative
anywhere) -- a sufficiently informative encoding of `u(x,t)` alone should
in principle already be Markovian, making the two-step history an
unnecessary crutch (and a possible confound: forcing the encoder to be
useful only in combination with a second lagged copy of itself is a
different, and possibly harder, representational demand than encoding the
state on its own). Add a `mode` flag, read by both the model and the
training loop so they can never disagree about which map is active:

- `mode="two_step"` (original, above): `(z_{n-1}, z_n) -> z_{n+1}`.
- `mode="markovian"`: `M(z_n) -> z_{n+1}`, using only the current state.
  `M` gets its own `backbone` choice:
  - `backbone="mlp"`: identical residual-MLP pattern, just `Linear(44→128)`
    instead of `Linear(88→128)` as the first layer (no other change).
  - `backbone="transformer"`: tokenize `z` into a handful of chunks, run a
    couple of self-attention blocks (optionally with a banded/local
    attention mask over *adjacent latent-channel index*, not physical
    space -- there is no spatial meaning to latent-channel adjacency until
    Phase 10's spatially-organized latent field exists; document this
    plainly rather than implying a locality claim the current
    architecture doesn't support), then project back down.

Both `mode`s keep the zero-init-output-head identity-at-init property.
`mode="markovian"` also changes Stage 1's `L_pred` window: only 1 history
snapshot is needed (not 2), so `L_recon`'s snapshot count shrinks from 4 to
`1 + K_pred` accordingly -- re-derive whatever numeric claims depend on the
window size (e.g. shift-augmentation counts) for this mode.

Two further hypotheses raised in the same debugging session, about *why*
the representation-collapse failure mode (documented above `### 5.2`... see
`docs/RESULTS.md`) might be slow to escape, worth testing alongside or
instead of the mode change:

1. **`dt_snap` too small** relative to the encoder's ability to
   distinguish nearby-in-time states -- try a larger snapshot stride.
2. **`L_pred`'s rollout (`K_pred=2`) too short** to generate a strong
   non-degenerate gradient from a collapsed starting point -- try a longer
   rollout (`K_pred > 2`).

Record findings for all of these against `docs/RESULTS.md` /
`docs/OPEN_QUESTIONS.md` before adopting any one of them as the new
canonical recipe.

### 5.1/5.2 addendum: Ported improvements from a reference implementation (2026-08-29)

The user pointed at a separate, independent implementation of essentially
the same problem at
`/Users/daltonjones/Documents/experiments/ks_latent/` (files: `config.py`,
`data.py`, `models.py`, `regularizer.py`, `train.py`, `evaluate.py`,
`lyapunov.py`, `plots.py`), reporting it trained to visibly lower loss and
higher reconstruction accuracy. It was read in full and compared line by
line against this codebase's Stage-1/Stage-2 training. Below is what
differed, what was adopted, and — just as important — what was
deliberately **not** adopted and why, so nothing is silently dropped.

**Already equivalent (no action needed).** Scalar (not per-feature)
mean/std standardization of the field, computed on the training split only
(`ks_latent/solver/dataset.py` already does this — brief §3.3); a residual
propagator `z_{t+1} = z_t + M(z_t)` with a zero-initialized output head so
the map starts as the identity (`ks_latent/models/propagator.py`, already
in place before this addendum); AdamW with a cosine-decaying LR; a
K-curriculum for the propagator's multi-step training (Stage 2 already
ramps `k` from 2 to `k_max`, matching the reference's unroll-ramp idea).

**Adopted as new defaults (fixes, applied unconditionally).**

1. **`weight_decay` explicit on AdamW.** Both `train_stage1` and
   `train_stage2` previously called `torch.optim.AdamW(params, lr=cfg.lr)`
   with no `weight_decay` argument, which silently uses PyTorch's default
   of `0.01`. The reference project's chosen value, arrived at by
   measurement, is `1e-5` — 1000x smaller. An implicit `0.01` decays every
   parameter every step, including the zero-initialized propagator output
   head that the identity-at-init guarantee depends on, and every
   LayerNorm gain/bias; this is a real latent bug, not a hyperparameter
   preference, since nothing in this codebase's config or tests ever
   intended that value. Fixed by adding `weight_decay: float = 1e-5` to
   both `Stage1TrainingConfig` and `Stage2TrainingConfig` and passing it
   explicitly.
2. **Linear LR warmup before the cosine decay, floored above zero.**
   Plain `CosineAnnealingLR` from epoch 0 exposes a freshly initialized
   (non-identity) encoder/decoder to the full peak LR on step 1. Added
   `warmup_epochs`/`lr_min_factor` fields (defaults 2 / 0.02, matching the
   reference) and a shared `warmup_cosine_lr_lambda` helper in
   `ks_latent/training/loops.py`, used via `LambdaLR` in place of the bare
   `CosineAnnealingLR` in both training loops.

**Adopted as new, off-by-default capabilities (experiments to run, not yet
validated as canonical-recipe changes — ground rule 6: don't fabricate a
recipe change without the numbers to back it).**

3. **Denoising reconstruction training (`Stage1TrainingConfig.noise_std`).**
   Gaussian noise added to the *encoder input* on the reconstruction path
   only (`L_recon = MSE(D(E(x+eps)), x)`); the prediction path and both
   `L_decorr`/`L_var` always see the clean encoding. Motivated by this
   project's own DA use case more than the reference's original
   regularization framing: `E`/`D` are otherwise only ever shown states
   exactly on the attractor, but every state Phase 5's PFF filter hands
   them — an analysis state, an ensemble member — is an off-attractor
   perturbation of one.
4. **Banded latent-index-smoothness + off-band decorrelation regularizer**
   (`RegConfig`, `ks_latent/training/regularizer.py`). Penalizes `z^T B z`
   for a banded graph-Laplacian `B` on the latent index (pulls coordinates
   *near in index* toward similar values — literally what Phase 10's
   backlog item "give the flat latent index physical/spatial meaning" is
   waiting on), paired with an off-band decorrelation term that keeps that
   pull from collapsing every coordinate to the same value (the banded
   term's unconstrained global optimum). This is a genuinely new
   capability, not a replacement for the existing `L_decorr`/`L_var`
   terms: those push the full latent covariance toward the identity but
   are exactly permutation-invariant in the index, so they cannot give the
   index itself any meaning the way a banded term can. Off by default
   (`lambda_z=lambda_decorr=0.0`). **Not a recommended experiment — user
   direction, 2026-08-29:** the reference project's own measurements
   already showed this exact penalty collapsing the latent (participation
   ratio ~1.1 of 8 at `lambda_z >= 5e-3`, documented in
   `regularizer.py`'s module docstring) — i.e. it recreates the failure
   mode this project spent significant effort escaping. Kept in the
   codebase, tested, for possible future use; not to be proposed or turned
   on as a fix for anything.
5. **`Stage2TrainingConfig.latent_loss: "l2" | "l1"`.** The reference
   project (following Linot & Graham, arXiv:2109.00060) found L1 gave
   "visibly better short-time tracking" for the propagator's latent-space
   loss, reasoning that L1 is less dominated by the rare large increments
   a chaotic trajectory produces. Default stays `"l2"` (the brief's
   original, already validated in this codebase's tests).

**Considered and deliberately not ported, with reasons.**

- **The "two-stage" training curriculum** (train `E`/`D` on reconstruction
  *alone*, freeze them, then fit the propagator to the frozen latents with
  a latent-space loss) — the reference project's fix for a *different*
  failure mode it measured (M/D co-adaptation on a joint physical-space
  loss, not full representation collapse). This codebase's Stage-1/Stage-2
  split is already structurally close to this: Stage-2's `train_stage2`
  already does exactly "fit the propagator to frozen Stage-1 latents with
  a latent-space loss." The nearest equivalent of turning the reference's
  "joint" mode into "two_stage" here is setting
  `Stage1TrainingConfig.w_pred=0.0` (dropping the auxiliary propagator's
  prediction loss out of Stage-1 entirely and letting Stage-2 do all the
  dynamics fitting) — already expressible with the existing config, no new
  code needed. Flagged here as a cheap, high-value experiment to run
  **once Gate 3/4's current canonical training finishes** (not run
  alongside it, to avoid confounding the collapse-recipe validation
  already in flight).
- **ViT/CNN encoder tracks, an attention-based propagator over
  per-coordinate tokens, and a "delta" decoder mode**
  (`xhat_{t+1} = x_t + D_delta(z_{t+1})`, a physical-space residual
  prediction) — all real, well-argued ideas in the reference project
  (see its `models.py`), but each is an architecture change to the
  encoder/decoder/propagator that would need its own shape/gradient tests
  and re-validation against this codebase's existing Gate-1/2/3 numbers.
  Given this codebase already has a working, tested patched-transformer
  architecture (brief §5.1) and a separate `mode="markovian"` +
  `backbone="transformer"` propagator option (the addendum directly
  above), these were judged higher-risk-for-the-payoff than the five
  items adopted above and are left here as documented candidates rather
  than implemented.
- **Per-trajectory (not per-snapshot) train/val/test split** — the
  reference project's `data.py` split strategy. `ks_latent/solver/dataset.py`
  already splits by whole trajectory (`n_train`/`n_val` trajectories, not
  snapshots), so there was nothing to change here.

**Controlled before/after result (`docs/RESULTS.md`, "Canonical
NX=256/markovian/dt_snap=1.0 run #2"): items 1-2's hypothesis was NOT
confirmed.** A single-seed, otherwise-identical rerun with only
`weight_decay=1e-5` and the LR warmup applied reached `val_recon_final =
0.815`, against the original defaults' `0.811` -- no improvement, and if
anything a slightly later collapse-escape. Items 1-2 are kept as defaults
anyway on correctness grounds (an unintended 1000x-too-heavy weight decay
has no principled justification independent of its measured effect), but
they should not be cited as having fixed the collapse-escape-speed
problem -- that remains open, see `docs/OPEN_QUESTIONS.md`.

### 5.1/5.2 addendum: MLP encoder/decoder and multistep-rollout comparison (2026-08-29)

Following the reference project's own `L=94, N=128, dt_model=1, d_z=45`
benchmark (`/Users/daltonjones/Documents/experiments/output/latent/
single_run_N128.log`) -- essentially this codebase's exact operating point
-- reaching **val recon MSE ~0.008** against this codebase's **~0.81-0.815**
at the same scale, two more variables from that run were isolated and
ported, user-directed, to test independently and in combination:

1. **`--encoder mlp`** (`KSAutoencoderMLP`, `MLPAutoencoderConfig`,
   `ks_latent/models/autoencoder_mlp.py`): the reference project's plain
   MLP encoder/decoder ("Track A"), `NX -> 512 -> 256 -> 128 -> d_latent`
   with `Linear -> LayerNorm -> GELU` hidden layers and a linear
   bottleneck, exact mirror decoder. Swaps in for
   `KSAutoencoderPatched` (the patched-transformer) with the same external
   interface (`encode`/`decode`/`forward`), so `train_stage1`,
   `eval_stage1_reconstruction`, and `encode_dataset_with_shifts` needed
   zero changes.
2. **`--multistep`** (`Stage1TrainingConfig.k_pred_max`/
   `k_pred_warmup_epochs`, `ks_latent/training/loops.py`'s
   `k_curriculum`): the reference project's `single_run_N128` benchmark
   trained with an 8-step unroll curriculum and a real-sized propagator,
   not the brief's tiny fixed-`k_pred=2` auxiliary one discarded after
   Stage 1. Ported as: size `AuxPropagatorConfig` like the actual Stage-2
   `PropagatorConfig` (`hidden=128, n_blocks=3` instead of `hidden=64,
   n_blocks=2`), and ramp the `L_pred` rollout length linearly from
   `k_pred=2` to `k_pred_max=8` over the first 30% of epochs (starting at
   the full rollout length immediately is unstable -- an untrained
   propagator compounds its own error and the gradient through the chain
   can explode before the one-step map is any good; same reasoning as the
   reference project's own `unroll_ramp_frac`). `k_curriculum` generalizes
   Stage 2's existing `k_min=2` K-curriculum to an arbitrary `k_min`, shared
   by both stages rather than duplicated.

The two flags are independent by design specifically so all four cells of
the 2x2 (encoder x rollout regime) can be run and compared: baseline
(transformer + short rollout, already have `val_recon_final = 0.815` from
the run above), mlp + short, transformer + multistep, mlp + multistep. See
`docs/RESULTS.md` for the results once all four finish.

**Result: the MLP-encoder cell alone closed almost the entire gap
(`docs/RESULTS.md` has the details).** `--encoder mlp` + short rollout
reached `val_recon_final = 0.005587` against the baseline's `0.815` -- a
~146x reduction in MSE from swapping *only* the encoder/decoder
architecture. Just as informative as the number: the training curve
(`artifacts/stage1_history_full_mlp_short.json`) descends smoothly and
monotonically from epoch 0, with **no collapse plateau at all** -- unlike
every patched-transformer run in this document, which spent 30-45 epochs
stuck near `recon~1.0` before any escape. This points at the
patched-transformer's optimization landscape specifically (not the KS
task, the loss, or the domain size) as the main driver of the collapse
this project spent significant effort escaping earlier.

### 5.1/5.2 addendum: ViT-style encoder/decoder (2026-08-29)

The MLP result above raised an obvious follow-up, user-directed: is the
patched-transformer's problem "transformers in general," or something
specific to *this* transformer's design? A third architecture,
`--encoder vit` (`KSAutoencoderViT`, `ViTAutoencoderConfig`,
`ks_latent/models/autoencoder_vit.py`), ports the reference project's
`ViTEncoder`/`ViTDecoder`/`CircularPositionalEncoding` to test this: it is
also transformer-based (so a win here would implicate the
patched-transformer's *specific* design rather than attention itself), but
differs from `KSAutoencoderPatched` in exactly one structural respect that
matters: **positional encoding.**

`KSAutoencoderPatched` has **none, anywhere, on the encoder side.** Its
local transformer mixes each group of `group_size` consecutive patches
with plain self-attention (permutation-invariant without a positional
signal) and then mean-pools -- destroying whatever order information
survived -- before the group tokens ever reach the global transformer,
which also carries no positional encoding. The encoder can route
information by *content* but has no explicit signal for *where* a feature
sits in space. `KSAutoencoderViT` instead adds `CircularPositionalEncoding`
-- a fixed, periodicity-respecting Fourier embedding (KS is on a periodic
domain, so patch 0 and patch `n_tokens-1` must be geometric neighbours,
which a plain/linear positional encoding does not know) -- to every token
*before* any attention layer, on both encoder and decoder, and runs a
single flat attention stack over all `n_tokens` patches directly (no
local-group pre-pooling stage). See `ViTAutoencoderConfig`'s docstring for
the full architecture and the exact math of the positional encoding
(ported faithfully, with its own periodicity/ring-metric tests --
`tests/unit/test_autoencoder_vit.py`).

`patch_size=8` (`n_tokens=32`, matching `AutoencoderConfig`'s own
tokenization exactly) was chosen by the same tokens-per-characteristic-
structure reasoning the reference project used for its own default: at
this codebase's `L=100`/`NX=256`, `dx~0.39`, patch length `8*0.39~3.12`
against the energy-injection lengthscale `2*sqrt(2)*pi~8.89` gives `~2.85`
tokens per structure, matching the reference's own `~3.0` almost exactly.

`decoder_mode="delta"` (the reference project's residual physical-space
decoding option, carried by both its MLP and ViT decoders) was **not**
ported to either -- out of scope for this comparison, and a candidate for
its own follow-up if `--encoder vit`/`--encoder mlp` prove out.

**Not a clean single-variable ablation of positional encoding alone** --
`KSAutoencoderViT` also differs in its bottleneck mechanism (mean/CLS
pooling vs. 8 learned query tokens), how the decoder is seeded from `z`
(a per-token linear readout vs. a shared broadcast vector), the patch
embedding (linear vs. 2-layer MLP), and feedforward width (`mlp_ratio=4`
vs. no expansion). A result here implicates the ViT design as a whole;
isolating positional encoding specifically would mean adding
`CircularPositionalEncoding` into the existing `KSAutoencoderPatched`
directly, not done here.

`--multistep` was dropped from this comparison (user direction,
2026-08-29): `transformer + multistep` (§5.1/5.2 addendum above) showed no
effect over the short rollout, so `mlp + multistep` and `vit + multistep`
were cancelled/not run -- both would only have re-tested a variable
already shown inert.

**Result (`--encoder vit`, short rollout): `val_recon_final = 0.000550`
-- beats both the patched-transformer baseline (~1,482x lower MSE) and
the MLP (~10x lower MSE).** Superseded as the best result by the windowed
variants below (`attn_window=4`: `0.000492` circular / `0.000494` linear
pos_encoding -- statistically indistinguishable from each other). See
`docs/RESULTS.md` for the full ranking and training-curve comparison.

### 5.2 addendum: ViT-style propagator backbone, no bottleneck (2026-08-29)

User-directed follow-up: does the ViT autoencoder's win carry over to the
*propagator*? `LatentPropagator`/`AuxPropagator` already had a
`backbone="transformer"` option (tokenize `z`, add a **learned** absolute
positional embedding, run `nn.TransformerEncoderLayer` blocks, untokenize)
alongside `backbone="mlp"` (the default residual-MLP body). Added
`backbone="vit"` (`_ViTDeltaBody`, `ks_latent/models/propagator.py`):
identical tokenize/untokenize shape, but reuses
`ks_latent.models.autoencoder_vit`'s `CircularPositionalEncoding` (fixed
Fourier ring embedding, not learned) and `ViTBlock` (pre-norm attention +
`mlp_ratio`-expansion MLP, the exact block `KSAutoencoderViT` uses) --
promoted from `_ViTBlock` to a public, shared `ViTBlock` for this reuse.

**"Without dimension reduction/expansion"** (the user's exact phrasing):
unlike `KSAutoencoderViT`, which pools all tokens down to one vector at the
bottleneck and re-expands via a big linear layer, `_ViTDeltaBody` never
pools -- the token grid keeps its full `(n_tokens, token_d_model)` shape
through every block. It is a plain same-shape seq2seq transformer over the
tokenized latent, which is what a propagator (`z -> z`, same dimension in
and out) calls for; the autoencoder's pooling step exists only because it
has to compress to `d_latent`, a constraint that does not apply here.

New config field: `token_mlp_ratio: int = 4` (both `AuxPropagatorConfig`
and `PropagatorConfig`), used only by the `"vit"` backbone. `mode="two_step"`
still stays MLP-only (same restriction as `"transformer"`); `d_latent` must
still be divisible by `n_tokens`. Zero-init on the final projection and the
uniform `.step`/`.step_one`/`.rollout` interface are unchanged, so DA
cycling, the Lyapunov adapters, and the D3 coupling diagnostic all work
with this backbone with no further changes -- verified in
`tests/unit/test_propagator_modes.py` and a Stage-2 training smoke test.

**Same caveat as the encoder's positional-encoding question:** the
`transformer` vs. `vit` backbones differ in more than positional encoding
alone (circular vs. learned embedding, `mlp_ratio` FFN width vs.
`dim_ff=d_model`, and the underlying attention implementation). Now wired
to `--backbone {mlp,transformer,vit}` on `scripts/train_stage2_patched.py`
(smoke-tested end to end for all three); not yet benchmarked against the
real Stage-2 propagator training task at full scale. Benchmarked so far as
the *auxiliary* propagator's backbone during Stage-1 training instead --
see `docs/RESULTS.md`'s "Aux propagator backbone comparison" and
"Ring-windowed attention" sections.

**Follow-up, same day: `pos_encoding="circular"|"linear"`.** User-directed
-- isolates the periodic-vs-non-periodic assumption specifically, holding
the `vit` backbone's block architecture fixed. `"linear"` swaps
`CircularPositionalEncoding` for a new `LinearPositionalEncoding` (the
classic fixed, non-learned Vaswani et al. sinusoidal encoding, but *not*
periodic) and the ring-distance mask for the existing linear-distance one
(`build_local_attention_mask`, the same mask `"transformer"` uses). Same
`ViTBlock`/`mlp_ratio` either way, so this is a cleaner ablation of
periodicity alone than the `transformer`-vs-`vit` comparison above. Applies
to both `KSAutoencoderViT` and `_ViTDeltaBody` (`--pos-encoding` on
`train_stage1_patched.py`). Implemented and unit-tested; not yet run
against real data -- see `docs/RESULTS.md`.

**Result: `pos_encoding="linear"` performs statistically indistinguishably
from `"circular"`** (`val_recon_final` 0.000494 vs 0.000492). **User
decision: canonical Stage-1/Stage-2 recipe changed to `vit` +
`attn_window=4` + `pos_encoding="linear"`**, replacing the
patched-transformer default -- chosen over `"circular"` despite the tie
because most real-world PDEs of interest are not periodic, so a design
that doesn't depend on periodicity generalizes further at no measured
cost here. See `docs/RESULTS.md`'s "Decision: canonical encoder changed"
entry for the full writeup, including a real bug caught while wiring this
up (`run_analysis_suite.py`/`run_da_pff.py`/`run_diagnostics.py` all
hardcoded `KSAutoencoderPatched` regardless of checkpoint architecture,
fixed via a shared `ks_latent.models.load_autoencoder_checkpoint`).
Gate 3/4 are being run against this recipe now.

### 5.3 Tests

`test_autoencoder_shapes`, `test_propagator_shapes`, `test_overfit_single_batch` (loss
< 1e-4 on one batch in 500 steps), `test_aux_propagator_gradient_reaches_encoder` (assert
non-zero encoder grads from `L_pred` — easy to break silently), a golden forward pass for
a fixed seed, and `test_shift_equivariance_learned` which *measures* how approximately
equivariant the trained global encoder is (needed by diagnostic D4).

---

## 6. Phase 4 — Analysis suite (replication)

### 6.1 `analysis/dimension.py`
- **Two-NN (Facco et al. 2017)**: fit the central 90% of the CDF through the origin,
  midpoint convention `(i−0.5)/N`. Document the known downward bias at high `d` (true 22
  reads ≈ 19).
- **Correlation dimension** (Grassberger–Procaccia); document that it is a lower bound.
- **Diffusion maps** with spectrum plotting.

Validate all three on `d`-spheres and `d`-tori for `d ∈ {2,5,10,15,20,22}`. Provide a
**sample-size convergence curve** helper — an estimate that has not visibly plateaued is
not trustworthy and the code should say so.

### 6.2 `analysis/lyapunov.py`
Benettin with QR reorthonormalization, tangent propagation via `torch.func.jvp`. **QR on
CPU float64** (§1.2).

Two modes:
- `single_state` — the `z → z` approximation used originally.
- `two_step` — the **exact** tangent map on the true `2d`-dimensional phase space
  `(z_{n-1}, z_n)`. An open item in the handoff. Implement both and compare.

Report the spectrum, number of positive exponents, `D_KY`, and `λ₁` in **physical time
units** (`λ₁_step / Δt`), checked against the literature bound (KS exponents bounded above
by ≈ 0.1). A mismatch almost always means `dt_snap` was lost.

Validate: machine-precision recovery on a linear map with known eigenvalues; Hénon and
Lorenz-63 exponents to 2%. Assert boundedness of the reference trajectory; raise if it
escapes.

**Write this with a clean interface accepting any lattice model, not just the 44-dim
propagator** — it is reused as the gauge-invariant comparator in Phase 12, which is the
most important use it will have.

### 6.3 `analysis/topology.py`
Persistent homology, plain Rips **and DTM (distance-to-measure) filtration**. Two probes:
the raw 44-dim cloud and a 6-dim diffusion embedding. Sweep DTM mass over
`{0.02, 0.05, 0.1}`; assert the conclusion is stable across the sweep.

Validate on a circle (one persistent `H1`), a torus (two), a 2-sphere (one persistent
`H2`), each with and without outliers to show DTM is earning its keep.

Docstring note: birth/death *scales* differ between probes purely because the natural
distance scale of each space differs; only the *shape* of the diagram — distance from the
diagonal, i.e. lifetime — is meaningful.

---

## 7. Phase 5 — Data assimilation (replication)

`ks_latent/da/pff.py` — one `ParticleFlowFilter` class, methods `DET` / `STO` / `NAT`
(default `NAT`). Diagnostic mode activates when `decoder_phys` is provided.

Preserve exactly:
- `B` is the **prior** covariance, computed once from the forecast ensemble, **fixed**
  through pseudo-time.
- NAT Gauss–Newton metric `F = J̄ᵀR⁻¹J̄ + B⁻¹` with the **ensemble-mean** Jacobian `J̄`.
  Kernel uses `F` as metric. The divergence term simplifies via `F`/`F⁻¹` cancellation —
  **derive this in a docstring**, do not just assert it.
- Prior term `−F⁻¹B⁻¹(z^j − z̄)`.
- Stochastic term `η = √2 · L_precond · N_mat · L_kᵀ`, `precond = F⁻¹`, `k_bar/N` inside
  the Cholesky.
- Step schedule (NAT branch), **no `ds_min` anywhere**:
  ```python
  ds_test = min(max(1.0, 1.0 / (f.norm() + 1e-8)), ds_max)
  if s == 0:            ds = 1.0
  elif ds_test > ds:    ds = min(1.1 * ds, ds_test)
  else:                 ds = max(0.9 * ds, ds_test)
  ```
- Early stop `tol=1e-4` on relative change of the ensemble mean, only after `s > 5`;
  `n_steps` default 100.
- `MODEL_NOISE_STD = 0.08` at **each** propagator step in `forecast_ensemble`. The free
  run gets **no** noise.
- `OBS_NOISE = 0.1`, `R` physical, all observation I/O in physical units.
- RMSE: divide the vector norm by `sqrt(d)` for **both** free and DA runs → per-dimension
  RMSE consistent with the spread.

Spacetime plotting: per cycle the segment is `N_PROP_STEPS` rows with the **last forecast
row overwritten** by the analysis. Truth is `u_truth_full[1:]`. Explicit edges,
`shading='flat'`, `t_edges = arange(T_full+1) + 0.5`. `u_da_history` uses
`decode(mean(z))`, **not** `mean(decode(z))`.

**Tests:** `test_pff_gaussian_linear` — with a linear observation operator and Gaussian
prior, the PFF posterior mean and covariance must match the analytic Kalman answer to 2%
at large ensemble size. This is the one test that proves the filter is correct. Also
`test_pff_no_observations_is_identity`, `test_rmse_normalization_consistency`,
`test_robust_cholesky_fallback`, `test_ds_schedule` (unit test of the recursion against a
hand-computed sequence).

**Gate 3: the §18 replication table.**

---

## 8. Phase 6 — Structure diagnostics D1–D5 (no retraining)

`ks_latent/analysis/diagnostics.py`. Main notes §3.

**D1 — sensitivity maps.** `S[k,x] = E_z |∂D(z)(x)/∂z_k|` and the encoder analogue
`|∂z_k/∂u(x)|`. Use **circular** statistics (resultant-vector length `R`,
`spread = sqrt(-2 ln R)`) — the domain is periodic and linear statistics are wrong.
Output a `(d, NX)` heatmap sorted by centroid plus a per-latent table. Test against a
synthetic decoder built from known localized bumps; assert centroids recover to within one
grid cell.

**D2 — wavenumber content.** FFT each sensitivity map; report spectral centroid and
bandwidth. If latents are narrow-band in `k` rather than narrow in `x`, the latent is
*spectral*, not spatial, and the right model is a mode-coupling ODE system. **Addendum
§14.4 adds a third possibility to test for explicitly:** Wittenberg & Holmes (1999) find
KS is localized in *both* real and Fourier space, so compute a time–frequency / wavelet
localization measure too, not just the two extremes.

**D3 — Jacobian coupling graph.** `A[k,l] = E |∂z_{n+1,k}/∂z_{n,l}|`, plus a nonlinear
alternative (distance correlation; k-NN mutual information) so the conclusion does not
rest on linearization. Then: **is there a permutation making `A` approximately banded or
circulant?** Spectral seriation (Fiedler vector), scored as

```
bandedness(A, π) = Σ_ij A[π_i, π_j] · w(circular_dist(i,j))
```

normalized, **compared against a null distribution of ≥ 1000 random permutations → report
a p-value.** A picture is not a result. Note in the output that the original seriation
attempt used the *covariance*, which is ≈ I by construction because of `L_decorr`, so it
could not have found anything.

**D4 — translation representation.** For each shift `c`, closed-form least squares for
`R(c) = argmin_R E_u ‖E(roll(u,c)) − R E(u)‖²`. Report (i) relative residual of the linear
fit per `c`; (ii) group property `‖R(c₁)R(c₂) − R(c₁+c₂)‖/‖R(c₁+c₂)‖`; (iii) eigenvalues —
if it is a genuine one-parameter group they take the form `e^{i k_j c}`, giving **latent
wavenumbers** `k_j`. If this works even partially it hands us a spatial/spectral
coordinate on the existing latent with zero retraining. Test on a synthetic exactly
equivariant encoder (truncated Fourier projection); assert recovery to 1e-8.

**D5 — local intrinsic dimension vs. patch length.** Two-NN on physical patches of length
`ℓ ∈ {5,10,20,30,40,50}`, using `AttractorPointDataset` sampling. Extensivity predicts
`d_local(ℓ) ≈ 0.226·ℓ + c`; fit the slope with a confidence interval. Validates
extensivity at `L=100` **and** sets the channel count for Phase 10.

`scripts/run_diagnostics.py` runs all five and emits `diagnostics_report.md`.

**Gate 4: `diagnostics_report.md` exists; D3 has a p-value; D4 has a verdict.**

---

## 9. Phase 7 — Distance-free localization on the existing latent

*(Addendum §13.3. **Do this early.** It works on the current checkpoint with no retraining,
gives an immediate DA result, and establishes the honest baseline that Phase 13 must beat.)*

`ks_latent/da/sec.py`

The main notes argue localization needs a distance and the latent has none. True for
Gaspari–Cohn. But a whole family of methods localizes **without any distance**, by
estimating the taper empirically offline:

- **Anderson (2012)**, sampling error correction (SEC), *Mon. Wea. Rev.* 140:2359 — a
  precomputed lookup table mapping sample correlation → expected regression attenuation as
  a function of ensemble size, applied per variable pair. **No distance function
  anywhere.** Primary recipe.
- **Anderson (2007)**, hierarchical ensemble filter, *Physica D* 230:99.
- **Anderson & Lei (2013)**, empirical localization functions, *Mon. Wea. Rev.* 141:4140.

Implement:
1. Build a latent archive (long runs, or the validated 10k-independent-point scheme).
2. Estimate offline either a fixed `d×d` empirical taper `ρ_kl` or an SEC lookup table.
   Implement both; SEC is more standard and more defensible.
3. Run NAT-PFF with `B ∘ ρ` and **sweep `N_ens ∈ {8,16,32,64,128,256}`**.

Report analysis RMSE vs. `N_ens`, with and without the taper, against the free run.

**Frame the result carefully in `docs/RESULTS.md`.** The limitation of an empirical taper
is exactly what motivates Phase 10: it is fitted to one domain size and one training
distribution and encodes no notion of physical space, so it **cannot give L-transfer**.
That is a much better argument for the local latent field than "we cannot localize at
all," because it is a statement about what each method can *extrapolate*.

Baselines to know before making any ensemble-size claim (addendum §14.5):
- **ROAD-EnKF**, arXiv:2301.11961 — latent EnKF with a KS test case; closest direct
  quantitative comparison.
- **Learning Enhanced Ensemble Filters**, arXiv:2504.17836 / *JCP* 2025 — reports beating
  an optimized **LETKF** on KS at both small and large ensemble sizes.
- **LAE-EnKF**, arXiv:2603.06752 — latent autoencoder EnKF; explicitly lists localization
  and inflation as *future work*, which both supports our novelty claim and dates it.
  **Phase 13 is time-sensitive.**

**Gate 5: `N_ens` sweep with and without SEC, plotted, with the extrapolation argument
written down.**

---

## 10. Phase 8 — Interpretation C, decoupled from the latent

*(Addendum §13.4. Delivers a quotable "here is the PDE" answer in about a week with **no
autoencoder at all**, and de-risks the whole program.)*

`ks_latent/discovery/pde_find.py`

Take `ū = G_ℓ * u` (Phase 1 filtering module) for a sweep of filter widths `ℓ`, and fit
`∂_t ū = F(ū, ∂_x ū, ∂²_x ū, ...)` with a standard library.

Why this beats running SINDy on latent channels:
- **no gauge freedom** — `ū` is fixed and interpretable;
- **known answer** to validate against: coarse-grained KS is in the KPZ / stochastic-
  Burgers universality class (Yakhot 1981; Zaleski; recent FRG work identifies KPZ
  `z=3/2`, Edwards–Wilkinson, and an intermediate inviscid-Burgers `z=1` regime), with the
  renormalized viscosity flowing from negative (bare KS) to positive;
- afterwards it becomes the **target** the learned local latent law must reproduce in its
  large-scale sector — a far stronger claim than "we ran SINDy on our channels."

Use `pysindy`, **weak formulation preferred** (far more noise-robust than pointwise
derivative estimation). Defaults from the literature: **larger libraries hurt** term
selection; use ensemble/bootstrap SINDy and report per-term **selection probabilities**,
not a single sparse fit; spurious terms are suppressed by more data, not more
regularization (arXiv:2608.20404).

**Validate before touching filtered data:** the pipeline must recover
`u_t = −u u_x − u_xx − u_xxxx` with correct signs and unit coefficients from raw solver
output, plus heat / Burgers / KdV–Burgers to 1%. If it cannot recover KS itself it cannot
be trusted anywhere. Most important test in the phase.

`ks_latent/analysis/scaling.py`: measure the roughness exponent `α` and dynamic exponent
`z` from structure and correlation functions; compare against KPZ (`α=1/2, z=3/2`) and EW
(`α=1/2, z=2`). **Caveat honestly:** `L=100` may be too small to reach the asymptotic
regime — report the measured crossover scale alongside the exponents rather than claiming
a clean classification.

**Gate 6: KS recovered from physical-space data; filtered-field results and exponents
reported.**

---

## 11. Phase 9 — D6: model-free local closure test

*(Addendum §13.1. Converts "sweep `c ∈ {1,2,3,4}` and see" from a training study into a
prediction that the training study then confirms or refutes.)*

For window half-width `w` and lead time `τ`, estimate the conditional variance

```
V(w, τ) = Var[ u(x, t+τ) | u(·,t) restricted to |x'−x| ≤ w ]
```

with a k-NN conditional-variance estimator, and compare against `V(∞, τ)`.

Two numbers fall out:
- **The width at which `V(w)` saturates** = the effective information horizon. It should
  agree with the Phase 2 light-cone estimate. **Two independent measurements of one
  number — if they agree, that is a strong result; report both regardless.**
- **The residual `V(∞,τ) − V_deterministic`** at large `w`, using only a *single* time
  slice, is the Mori–Zwanzig memory. Its size bounds what a Markovian local model can
  achieve, and therefore how many hidden channels (or delay steps) are needed.

Caveat to handle: k-NN conditional variance in a width-`w` window is a high-dimensional
estimate and will be sample-hungry. **Do it on a low-pass-filtered field first**, where
the effective dimension per window is small, then walk up the filter width.

Output: a predicted `c` and a predicted stencil half-width, written to `docs/RESULTS.md`
**before** Phase 10 runs.

---

## 12. Phase 10 — Local latent field

`ks_latent/models/local_field.py`. Main notes §5, plus addendum §12.3.

### 12.1 Rationale (module docstring)

At `L=100` the KS attractor dimension is *extensive*: `D_KY ≈ 0.226·L`. The state is not a
generic 22-dimensional point; it is a chain of local chaotic units at a fixed
degrees-of-freedom density. A latent **vector** has no continuum limit. A latent **field**
`z_j(t) ∈ R^c` on a periodic lattice `j = 1..P` does.

### 12.2.1 Architecture

- Encoder: stacked **circular** `Conv1d` (`padding_mode='circular'`), striding down to `P`
  sites × `c` channels. Translation equivariance becomes an **architectural guarantee**
  and replaces the shift augmentation. Receptive field ≈ 2–3 correlation lengths (KS cell
  scale `2√2π ≈ 8.9`, so ≈ 20–30 physical units) — **and must satisfy the Phase 2
  light-cone bound.**
- Decoder: mirror, transposed conv or pixel-shuffle, overlap-add.
- Latent shaped `(P, c)`, **not flattened**.

**Gauge anchor (addendum §12.3) — implement this, it is not optional.** Force **channel 0
to be a fixed, h-independent physical quantity**: the local average of `u` over the site's
support, or a fixed low-pass filter of `u` evaluated at site centres. Channels `1..c−1`
are free learned residuals. This gauge-fixes at least one component so it is directly
comparable across resolutions in Phase 12, makes the leading-order behaviour of `F`
interpretable and checkable against the Phase 8 KPZ/Burgers result, and turns a validation
hope into an automatic check.

**Hybrid option** (implement, ablate): `z = (few global scalars, local field)`. KS with
periodic BC conserves `∫u dx` and the translation phase is a global object; do not force
the local field to carry global constraints.

Sizing configs:

| P | h = L/P | dof/site ≈ 0.226h | c | total |
|---|---|---|---|---|
| 16 | 6.25 | 1.41 | 4 | 64 |
| **32** | **3.13** | **0.71** | **2, 3** | **64, 96** |
| 64 | 1.56 | 0.35 | 1, 2 | 64, 128 |

`P=32, c=3` is the starting point. The config validator must assert `h ≪ 8.9` for the
continuum limit (h≈3 comfortable, h≈6 marginal) and `c` at least the value predicted by
D6. With `c=1` the latent is essentially a local coarse-graining of `u`, and
coarse-grained KS is **not closed** (Mori–Zwanzig memory); extra channels are the local
hidden variables that restore Markovianity. **Sweep `c ∈ {1,2,3,4}` and compare against
D6's prediction.**

## 12.2.2 Architecture (local attention variant)

Encoder: patch the field into `P` non-overlapping segments of length `h = L/P` (`P=32`
recommended start). Each patch gets a linear embed to `d_model`, plus a **circular
positional encoding** (position on the ring, not absolute index — see equivariance note
below). Then stack `n_layers` of **windowed self-attention**: each patch attends only to
patches within a fixed radius `r` on the periodic index (its `2r+1`-patch neighborhood),
implemented with an additive attention mask of `-inf` outside the window, or with
`torch.nn.functional.scaled_dot_product_attention` given an explicit banded boolean mask
built once from `circular_dist(i,j) <= r`. Do **not** implement this by physically
gathering neighbor tokens per query (that breaks batched matmul efficiency); mask the full
`P×P` attention matrix instead — `P` is small enough (16–64) that this costs nothing.

After `n_layers` of windowed attention, project each patch token to `c` channels:
`Linear(d_model → c)`. Latent shape `(P, c)`, **not flattened** — same as the conv
version.

Decoder: mirror — broadcast each `(P, c)` site to a learned bank of per-patch query
tokens, run windowed self-attention (same radius `r`, or a wider one — ablate), then
`Linear(d_model → h)` per patch to reconstruct the segment, concatenated back to the full
field.

**Receptive field is `n_layers · r` patches**, i.e. `n_layers · r · h` in physical units.
Set this to satisfy the Phase 2 light-cone bound the same way as the conv version:
`n_layers · r · h ≥ v_* · Δt` and ≈ 2–3 correlation lengths (`2√2π ≈ 8.9` physical units).
Unlike a fixed-kernel conv stack, `r` and `n_layers` are independent knobs — sweep both,
not just one, since attention lets you trade depth for width at fixed total receptive
field, and that trade-off itself is worth reporting.

**Translation equivariance is not automatic here and must be engineered in, unlike the
circular-conv version — this is the main thing this variant costs you.** Requirements:
- Positional encoding must be **relative**, not absolute: use a circular relative
  position bias added to the attention logits (a learned or fixed function of
  `circular_dist(i,j)`, shared across all query-key pairs at that distance), not an
  absolute per-patch embedding. Absolute positional embeddings break equivariance
  outright.
- All attention weights (Q/K/V projections, the relative-position bias table, the output
  projection) must be **shared across patches** — the same windowed-attention block
  applied at every site, exactly analogous to weight-sharing across sites in the conv
  version. Do not give each patch its own parameters.
- With both conditions met, shifting the input by `s` patches shifts the attention
  pattern by `s` patches exactly (the mask and relative bias are themselves
  shift-invariant), giving the same architectural equivariance guarantee the conv version
  gets from `padding_mode='circular'`.

**Gauge anchor (§12.3) still applies unchanged**: channel 0 of `c` is fixed to the local
physical average of `u` over the patch, non-learned; channels `1..c−1` are free.

### Tests (replace the conv-specific ones)

- `test_windowed_attention_mask_is_circular_banded` — assert the mask is `True` exactly
  where `circular_dist(i,j) <= r`, including wraparound at the `P-1`/`0` boundary.
- `test_local_attention_exact_equivariance` — `E(roll(u, s·h)) == roll_sites(E(u), s)` to
  1e-6, **exactly**. This is the direct analogue of `test_circular_conv_exact_equivariance`
  and carries the same weight: if it fails, the relative-position bias or weight-sharing
  is wrong and the architecture does not have the property the whole program depends on.
- `test_relative_position_bias_shared` — assert the bias table (or bias function) has no
  per-patch-index parameters, only per-distance ones.
- `test_receptive_field` — as before, but compute it as `n_layers · r` patches by gradient
  masking (do not just trust the analytic `n_layers·r` formula, since attention's
  effective receptive field can be narrower than its nominal one if the learned attention
  weights concentrate near the query) and assert it against the Phase 2 bound.
- `test_anchored_channel_is_fixed`, `test_local_field_reconstruction`,
  `test_reconstructed_spectrum` — unchanged from the conv version (§12.4).
- write tests and evaluation to compare models from 12.2.1 and 12.2.2

### Note for `docs/RESULTS.md`

Record the comparison against the conv variant honestly if both are tried: attention
gives you an *adaptive* receptive field (the effective window can be narrower than `r` if
the learned weights concentrate locally) versus the conv's *fixed* one, which is a
genuinely different inductive bias worth reporting on its own — it may reveal whether the
true interaction range is sharp (conv-like) or graded (attention-like), which is itself
informative about the physics.


### 12.3 Wavelet control baseline

*(Addendum §14.4.)* Implement a **wavelet-packet local basis** at matched dimension as a
non-learned control for the whole architecture, using `PyWavelets`. Wittenberg & Holmes
(1999) built localized low-dimensional KS models this way 25 years ago. If a plain wavelet
basis does nearly as well as the learned encoder, that is important to know early; if the
learned one wins, the wavelet baseline quantifies by how much. Cheap; run it as a first-
class ablation, not an afterthought.

### 12.4 Tests

- `test_circular_conv_exact_equivariance` — `E(roll(u, c·stride)) == roll_sites(E(u), c)`
  to 1e-6, **exactly**, not approximately. If this fails the architecture is wrong.
- `test_receptive_field` — measure empirically by gradient masking, assert it matches the
  analytic value from kernel sizes and strides, and assert it satisfies the Phase 2 bound.
- `test_anchored_channel_is_fixed` — channel 0 equals the specified filter of `u` to
  numerical precision and does **not** change under training.
- `test_local_field_reconstruction` and `test_reconstructed_spectrum`. **Warning:** AROMA
  (arXiv:2406.02176, App. C.7) reports KS as an explicit failure case for local
  neural-field latents, with the *decoder* as the bottleneck and recon MSE ~1e-10–1e-12
  needed for accurate spectrum reconstruction. **Evaluate the time-averaged energy
  spectrum, not just MSE.**

**Gate 7: exact-equivariance test passes; reconstruction spectrum acceptable; wavelet
control measured.**

---

## 13. Phase 11 — Stencil propagator (a hypothesis test, not a hyperparameter search)

`ks_latent/discovery/stencil.py`

Shared local update, same weights at every site:
```
ż_j = f( z_{j-w}, ..., z_j, ..., z_{j+w} )
```
reparametrized in centred finite differences: `ż = F(z, δ¹z, δ²z, δ³z, δ⁴z)`.

**Sweep the half-width `w ∈ {1,2,3,4}` and the temporal stride together.** The width at
which rollout error saturates is a *measurement* of the effective interaction range and
must be compared against the Phase 2 light-cone prediction `w ≥ v_* Δt / h`. Two
independent routes to one number; report agreement or disagreement prominently.

Direct precedent to cite and compare against: **Bar-Sinai, Hoyer, Hickey & Brenner (2019)**,
"Learning data-driven discretizations for PDEs," *PNAS* 116(31):15344 — learned
finite-difference stencils on deliberately coarse grids with explicit h-dependence. The
closest existing thing to this, though in physical rather than latent variables.
Kochkov et al. (2021), *PNAS*, is the same lineage.

---

## 14. Phase 12 — The PDE/ODE verdict, done correctly

**Read addendum §12.1–§12.4 before writing any of this.** The h-refinement test in the
main notes §5.4 is **not well-posed** and must not be implemented as written.

### 14.1 The problem

Training separate autoencoders at `P=32` and `P=64` produces **two different latent
variables**. Nothing ties channel 2 of the coarse model to channel 2 of the fine model.
The local latent is defined only up to a site-wise invertible map `z_j → φ(z_j)` (channel
gauge, `GL(c)` at minimum, generally a nonlinear diffeomorphism) and a relabelling /
half-site shift of the lattice. `F` transforms covariantly under these. So "is `F` the same
at both h?" compares coordinate representations, not dynamics — **a negative answer would
be uninformative and a positive answer suspicious.**

### 14.2 Fix C — gauge-invariant observables (run this FIRST)

Before any coefficient-level comparison, compare quantities invariant under smooth changes
of latent coordinates:

| Observable | Why invariant | What it tests |
|---|---|---|
| **Full Lyapunov spectrum** of the lattice model | invariant under smooth invertible coordinate change | whether the *dynamics* agree, gauge-free |
| **`D_KY` per unit length** | ditto, plus extensivity | whether the model is extensive at ≈ 0.226 |
| **Coupling decay vs. physical distance** (Jacobian of `F` converted to physical units) | measured in length, not site index | whether the interaction range is an h-independent physical length |
| **Decoded two-point correlations / structure functions / energy spectrum** | computed in physical space | whether field statistics are reproduced |
| **Comoving Lyapunov / spreading velocity `v_*`** | physical velocity | whether information propagates at the right speed |

The Lyapunov row is the strong one, and **the machinery already exists** (Phase 4,
validated to machine precision). Matching Lyapunov spectral density across two site
spacings is a legitimate continuum-limit test that sidesteps the gauge problem entirely.
**Run this before attempting any coefficient comparison.**

### 14.3 Fix A — nested coarse-graining (an RG flow)

`ks_latent/discovery/rg_nesting.py`

Train **once** at the finest spacing `P_fine`. Then *define* coarser latents by a **fixed,
non-learned** local pooling operator `R`:

```
z^{(P/2)}_j := R( z^{(P)}_{2j}, z^{(P)}_{2j+1} )
```

with `R` a plain average (or a linear map learned once and then frozen across the sweep).
Fit a stencil model `F_h` at each level. The question becomes:

> **Does the sequence `F_h` converge as h → 0 under a fixed coarse-graining map?**

This is literally a real-space RG flow of the effective local law, and it is well-posed
because all levels share a coordinate system by construction. It reframes the deliverable
as "**what is the RG flow of the local latent law, and does it have a fixed point?**" A
fixed point *is* the continuum limit. Non-convergence with a measurable flow rate is a
quantitative negative result rather than a shrug. It is also **cheaper** — one autoencoder
instead of three.

**Risk to measure, not assume:** the fixed pooling `R` may destroy closure at coarse levels
even if a good coarse latent exists (the optimal coarse latent need not be a linear pool of
the fine one). Compare `F_h` fitted on pooled latents against `F_h` fitted on an
independently trained coarse latent, and **report the gap.** This is addendum open
question 7.

### 14.4 Pre-registration

Write the decision rule into `docs/RESULTS.md` **before** running: state the threshold on
each consistency measure that will be treated as "the continuum limit exists." Both
verdicts are publishable — write the report so a negative result reads as the answer to
the research question, not as a failure.

**Gate 8: gauge-invariant comparison run and reported; RG flow measured; verdict written
against the pre-registered rule.**

---

## 15. Phase 13 — Localized latent DA and L-transfer

`ks_latent/da/localization.py`

- **Gaspari–Cohn** tapering on the latent lattice index (standard piecewise quintic;
  `test_gaspari_cohn_positive_definite`).
- Banded observation Jacobian `J = ∂h(D(z))/∂z`; exploit bandedness with banded solvers.
- `F = J̄ᵀR⁻¹J̄ + B⁻¹` becomes banded → **measure and report condition number and wall time
  vs. the dense global-latent baseline.**
- Localized PFF kernel bandwidth.

**Assertion, not a comment:**
```
localization_radius >= max(encoder_receptive_field, v_star * dt)
```
Both quantities are measured (Phase 10 and Phase 2). Raise if violated. Stating this
design rule is a small but clean theoretical contribution.

### The two headline experiments

1. **Ensemble-size scaling.** Sweep `N_ens ∈ {8,16,32,64,128,256}` for (a) global latent,
   no localization; (b) global latent + **SEC** (the Phase 7 baseline); (c) local latent
   field + Gaspari–Cohn. **(c) must beat (b), not merely beat (a)** — that is what makes
   the result credible. Hypothesis: required ensemble size scales with the *local*
   dimension (~1–3 per site) rather than the global attractor dimension (~22).
2. **L-transfer.** Train the local model at `L=100`; run forecasting **and DA** at `L=200`
   and `L=400` by adding lattice sites, **no retraining**. Check rollout statistics, energy
   spectrum, and `D_KY/L` (should stay ≈ 0.226). The global baseline and the SEC taper
   **cannot do this at all** — that contrast is the result. **Highest-value experiment in
   the plan.**

---

## 16. Phase 14 — Optional extensions (only after Gate 8)

- **Latent neural ODE** replacing the discrete propagator, with the `−Ah` damping
  stabilizer (`A = γ δ_ij σ_i(h)`) from Constante-Amores, Linot & Graham, arXiv:2408.03135.
  Makes the continuum-limit statement cleaner (compare `∂_t z`, not one-step maps) and the
  Lyapunov analysis exact.
- **UPO / ECS search in latent space.** Not speculative — arXiv:2408.03135 finds ECS in a
  latent model that converge in the full system, yielding 17 previously unknown solutions.
  A *local* latent would let us look for **localized** coherent structures, a genuinely
  different question.

---

## 17. Phase X — Contingency: constrained latent locality

**Only if Phases 10–12 stall.** The addendum idea ledger lists this as *Held — second-best
to architecture*. Do not build it as part of the main line.

If invoked: fold in `reference/latent_locality.py` properly. Baseline first — replace
`L_decorr + L_var` with a **banded circulant covariance target** `‖Corr(z) − C_target‖²_F`
with `C_target` Gaspari–Cohn in the index (one line, no dual variables, nontrivial minimum
so no collapse). Only then the three-constraint augmented Lagrangian (smoothness +
variance floor + **far-pair decorrelation**).

Findings from the prototype that must not be re-derived:
- **The far-pair decorrelation constraint is not optional.** Smoothness + variance floor
  alone collapses effective rank to ≈ 1: the variance floor blocks *shrinkage* collapse but
  not *rank* collapse.
- **EMA the constraint value in the dual update.** `λ ← max(0, λ + μĝ)` with noisy
  minibatch `ĝ` is biased *upward* because the rectifier is convex, producing a slow
  multiplier ratchet. Write a test demonstrating the bias.
- **Calibrate `M`, anneal downward.** Start feasible; starting infeasible drives
  multipliers up fast.
- **Monitor effective rank** (participation ratio of covariance eigenvalues), not
  per-coordinate variance — the collapsed solution had *large* per-coordinate variance.

Whatever the outcome, **run D3 on the result.** Banded covariance with a dense Jacobian
means the route failed at the thing that matters.

---

## 18. Replication targets (acceptance criteria)

`tests/replication/`, `@pytest.mark.slow`. The gate on everything.

| quantity | target | tolerance |
|---|---|---|
| Full-PDE `D_KY` at `L=100` | 21–24 | must fall in range |
| Full-PDE `D_KY` at `L=22` | 5.2–5.6 | must fall in range |
| Latent `D_KY` (Benettin, single-state) | ≈ 21.4 | ±1.5 |
| Positive latent exponents | 11 | ±2 |
| Latent two-NN dimension | 18–19 (biased estimate of ~22) | ±2 |
| Latent correlation dimension | ≈ 14 (lower bound) | ±3 |
| Latent topology | `H0` lifts off the diagonal; `H1`, `H2` hug it at **all** DTM masses | qualitative, stable across the sweep |
| Diffusion map spectrum | smooth, **no gap** | qualitative |
| Free-run latent RMSE (per-dim) | ≈ 1.6 | ±0.3 |
| DA analysis RMSE (per-dim) | ≈ 0.14 | ±0.04 |
| Ensemble spread | ≈ 0.11 | ±0.03 |
| Spread/RMSE calibration | ≈ 0.8 | 0.6–1.2 |
| DA skill vs. free run | ≈ 10× | ≥ 6× |
| Stage-1 val reconstruction | comparable to the recorded run | log and compare |
| `λ₁` in physical units | ≤ 0.1 (literature bound) | must satisfy |
| `v_*` stride-independence | same physical value across strides | ±10% |

Each test writes its measured value and target to `docs/REPLICATION_LOG.md` with a
pass/fail marker, regenerated on every run.

---

## 19. Testing strategy

Four tiers by pytest mark:

- **`unit`** (< 1 s) — shapes, gradients, math identities, schedules, config validation.
- **`integration`** (< 60 s) — smoke profile of every training loop and experiment.
- **`replication`** (`slow`) — §18.
- **`golden`** — stored reference arrays; regenerating requires an explicit
  `--update-golden` flag producing a reviewable diff.

Additional requirements:

- **Property-based tests** (`hypothesis`) for numerical utilities: linalg fallbacks,
  circular statistics, Gaspari–Cohn, finite-difference stencils.
- **Every estimator is validated on synthetic data with a known answer before touching real
  data.** No exceptions — dimension, Lyapunov, homology, SINDy, scaling exponents,
  conditional variance, spreading velocity.
- **Test the failure modes.** Where we know something breaks (thinned trajectories
  underestimate dimension; the quadratic penalty collapses; smoothness + variance floor
  collapses rank; naive h-comparison is gauge-dependent), write a test asserting the
  failure reproduces. If it stops reproducing, either we fixed something or the test went
  blind — both need investigation.
- Coverage ≥ 85% on `ks_latent/`, excluding plotting.

---

## 20. Deliverables

1. The package, `make test-fast` green in under three minutes on `da_env`.
2. `docs/REPLICATION_LOG.md` — auto-generated §18 table.
3. `docs/RESULTS.md` — narrative results, with pre-registered decision rules stated
   **before** the corresponding results.
4. `scripts/` entry points per phase, all supporting `--profile smoke`.
5. `artifacts/` with figures and JSON summaries, each with `.meta.json`.
6. `docs/OPEN_QUESTIONS.md` — anything unresolved, ambiguous, or internally inconsistent in
   the source documents. One is known already: the writeup and handoff disagree on `NX` and
   `d`. Carry forward the addendum's open questions 7–10.
7. `scripts/bench_device.py` output: an honest MPS-vs-CPU table per model.

---

## 21. Order of work and gates

Follows addendum §14.3. Do not reorder without saying why.

| Phase | Content | Gate |
|---|---|---|
| 0–1 | Skeleton, env check, device policy, linalg, solver, datasets | **`test_L100_kaplan_yorke` passes** |
| 2 | D7: `v_*`, light cone, stencil and localization bounds | bounds written to `RESULTS.md` |
| 3–5 | Baseline AE, propagator, analysis suite, PFF DA | **§18 replication table** |
| 6 | Diagnostics D1–D5 | `diagnostics_report.md`; D3 p-value; D4 verdict |
| 7 | SEC / empirical localization + `N_ens` sweep | sweep plotted; extrapolation argument written |
| 8 | PDE-FIND on filtered physical field; KPZ check | **KS recovered from physical data** |
| 9 | D6 model-free closure test | predicted `c` and stencil width, written *before* Phase 10 |
| 10 | Local latent field AE (anchored channel 0) + wavelet control | exact equivariance; spectrum acceptable |
| 11 | Stencil propagator; width and stride sweeps | learned width vs. light-cone prediction |
| 12 | Gauge-invariant comparison, then RG-nested h-sweep | **verdict against the pre-registered rule** |
| 13 | Localized latent DA; `N_ens` vs. Phase 7 baseline; L-transfer | — |
| 14 | Optional: neural ODE, ECS search | — |
| X | Contingency: constrained locality | only if 10–12 stall |

Report at each gate before continuing.

---

## 22. Prior-art warnings to respect in all writing

*(Addendum §14.4.)* This has a pre-neural ancestor. **Read before writing any claims:**

- **Wittenberg & Holmes (1999)**, "Scale and space localization in the Kuramoto–Sivashinsky
  equation," *Chaos* 9(2):452 — wavelet analysis showing KS dynamics are well localized in
  **both real and Fourier space**, with distinct large-scale (slow, Gaussian), active-scale
  (travelling-wave / heteroclinic) and small-scale (intermittent) regimes.
- **"Spatially Localized Models of Extended Systems"** (Springer chapter, Holmes/Wittenberg
  lineage — **verify authors and year**) — wavelet projections building localized
  low-dimensional KS models on large domains, arguing that short periodized systems are
  minimal models for chaotic dynamics in arbitrarily large domains. That is essentially our
  L-transfer claim, made ~25 years earlier by other means.

**"We show KS admits local reduced models" is not available as a claim.** "We show a
*learned* local latent field enables localization and transfer in nonlinear latent DA" is.

Also read before claiming novelty: **Constante-Amores, Linot & Graham, arXiv:2410.01238**
(patch-decomposed AE + NODE on KS and Kolmogorov flow) and **arXiv:2408.03135** (pipe flow;
the regime contrast — a minimal flow unit is non-extensive by construction, which is
exactly why a global latent vector works there and does not scale here — is our
one-sentence positioning statement).

Citations in the source documents marked "verify" have **not** been confirmed against
primary sources. Check them before they appear in any output.
