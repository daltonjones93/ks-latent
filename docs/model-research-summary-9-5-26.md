# ks-latent Model Research Summary (2026-09-05)

**Scope**: synthesizes the entire project history — the ~70 unnumbered/numbered
experiments in `docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md` (Phases 1-3 plus
Sections 1-70), the fourier_mlp/ViT-hybrid arc in
`docs/HANDOFF_2026-09-04_FOURIER_HYBRID.md` (Sections 71-83), and this
session's own work (Sections 84-88, plus the new `w_smooth` regularizer and
`scripts/analyze_latent_smoothness.py` diagnostic). Every number below is
sourced from a logged Gate 3/4 run or this document's own extraction pass;
where a figure wasn't logged, it's marked "not logged" rather than guessed.

**Task**: identify what's been learned about which encoder, decoder, and
propagator architectures work best and why; propose falsifiable hypotheses
for the mechanisms involved; recommend untested architectures worth trying;
and name the best 3 models for PDE-modeling readiness, faithful dynamics
fidelity, and stability.

---

## 1. Executive summary

Four **largely independent axes** keep reappearing across ~88 sections of
experiments, and almost every surprising result in this project is one axis
moving while the others don't:

1. **Reconstruction/rollout accuracy** (`val_recon`, `val_kmax_mse`).
2. **Dynamical fidelity** — does the learned propagator's own attractor match
   the true KS system's chaos (`D_KY≈22-23`, `n_positive` positive Lyapunov
   exponents, `λ1`)?
3. **Latent geometric quality** — covariance conditioning (`cond#`),
   participation ratio, D3/D6/D7/D8 structural diagnostics, D4 translation
   equivariance, and (new this session) real-trajectory smoothness
   (step size / curvature / propagator Jacobian norm).
4. **Data-assimilation skill** (`skill_free_over_da`).

The single most-repeated finding in the whole project is that **(3) does not
predict (1) or (2) in either direction** — the best-conditioned latent tried
(Section 78, `cond#=11.7`) has mediocre rollout fit, and the worst-conditioned
latent tried (Section 52, `cond#=1.59e6`) has the *best* rollout fit and DA
skill of the entire project. Don't use conditioning number as a proxy for
anything except itself.

For the propagator specifically, one clean, falsifiable, repeatedly-confirmed
mechanism explains almost every collapse-vs-chaos result in the project (see
§3): **only architectures with a global (full-`d_latent`-reach), unnormalized
mixing operation recover the true chaotic attractor; anything local, or
anything global-but-softmax-normalized, collapses to a fixed point.**

---

## 2. Encoder / decoder: what works and why

### 2.1 What was tried

| Family | Best instance | `val_recon` | Notes |
|---|---|---|---|
| Original patched transformer (`AutoencoderConfig`, NX=1024) | — | 0.969 (≈ mean-baseline) | **Collapsed outright** — origin problem motivating the whole project. Fixed by shrinking NX to 256 and switching to `mode="markovian"`. |
| ViT-for-function-space (`vit`), globally mean-pooled | Section 52 (`d_model=96`, `attn_window=4` token-space, `pos_encoding=linear`) | 0.000292 | **Best raw rollout MSE (0.0256) and DA skill (3.123) of the entire project.** Catastrophic conditioning (`cond#=1.59e6`) but *also* the smoothest real-trajectory embedding measured this session. |
| `masked_mlp` as encoder/decoder itself (hard architectural locality) | Section 17 | 0.002043 (~10x worse) | Highest `D_KY`/`n_positive` of the whole document (22.26/13) but the worst reconstruction of any non-toy AE tried. Locality is structurally biased, not strictly local (compounds across stacked layers). |
| Banded pooling (`pool="banded"`, soft-local token→latent map) | Sections 24-25 | — | Only architecture besides masked_mlp-encoder to get all of D3/D6/D7 significant simultaneously, but 8-33x worse rollout accuracy than the dense-encoder best. |
| `fourier_mlp` (masked raw-value path + dense/ifft Fourier path, summed) | Section 75 | ~0.00003-0.00005 | `cond#=20.98` (well-conditioned), `D_KY=21.089`, `val_kmax_mse=0.0392`. First AE family combining good conditioning with real chaos fidelity. |
| `vit_fourier_hybrid` (ViT + FourierIFFTBody, summed) | Section 81 → 85 | ~0.00003 | Best of the fourier_mlp lineage. Section 85 (this session): `val_kmax_mse=0.0279`, `D_KY=21.668` (closest to true of this lineage), `cond#=24.0`, **D4 genuine translation equivariance**, and the smoothest embedding of the fourier_mlp/hybrid family. |
| `d_latent` scaling (56, 64) on the hybrid AE | Sections 82, 88 | worse each time | 56: `val_kmax_mse=0.084` but *better* `D_KY=22.16` and *better* DA skill (1.625) than 81 — not uniformly worse, just worse on rollout. 64 (this session, heavier regularizers): `val_kmax_mse=0.12-0.17`, participation ratio collapsed to 2.6/64 — clearly regressed. |

### 2.2 Hypothesis (falsifiable)

> **H-ENC: Soft regularization on a globally-connected encoder beats hard
> architectural locality, for both accuracy and measured structure.**
> A ViT (or ViT+Fourier hybrid) encoder with global mean-pooling, trained with
> balanced `w_spatial`/`w_logdet` (and, newly, `w_smooth`), achieves *both*
> better reconstruction/rollout accuracy *and* comparable-or-better structural
> diagnostics (D7 bandedness, D4 equivariance) than an encoder with masked/
> banded architectural locality built in.

**Evidence for**: Section 17 (masked_mlp encoder) and Sections 24-25 (banded
pooling) both pay a real, consistent accuracy tax (8-33x) for their locality,
without reliably beating soft-regularized models (75/81/85) on D7/D8
significance. Section 39 (`w_spatial=0.05, w_logdet=0.005` on a fully dense
ViT) achieved `D7 p=0.0000` at essentially no accuracy cost — this recipe is
the direct ancestor of every Section 75-88 recipe.

**How to falsify it**: train a hard-local encoder (banded pooling or
masked_mlp) with the *same* soft regularizers (`w_spatial`+`w_logdet`+
`w_smooth`) added on top, at the current best hyperparameters. If it closes
the accuracy gap to within ~2x of the dense-encoder best while keeping its
structural edge, H-ENC is wrong; if the gap persists, H-ENC stands.

### 2.3 Decoder-specific note

The `fourier_ifft_readout` mechanism (an MLP predicts frequency-domain
coefficients, then an explicit, mathematically exact `irfft` maps back to
state space, instead of an arbitrary linear readout) is a small but
consistent structural upgrade across the whole fourier_mlp lineage (Sections
74-88) — it never *hurt* accuracy relative to the plain-linear-readout
predecessor and gives the decoder a literal, inspectable frequency-domain
interpretation. Worth keeping as a default whenever the fourier_mlp/hybrid
family is used at all.

---

## 3. Propagator: what works and why

### 3.1 The central result — the 2×2 receptive-field/normalization grid

Sections 5, 9, and 10 together ran (unintentionally, over three sections) a
complete 2×2 grid crossing **receptive field width** (local vs. global) with
**normalized mixing** (softmax attention vs. unconstrained linear/spectral):

| | Softmax-normalized (attention) | Unconstrained (linear/spectral) |
|---|---|---|
| **Local** | `vit`, `attn_window=4` → **collapsed** (D_KY=0) | `local_mlp` (local conv, no softmax) → **collapsed** (D_KY=0) |
| **Global** | `vit`, full attention → **collapsed** (D_KY=0) | `mlp` (dense) → **chaotic** (D_KY=20.92-22.14); `fno_vit` (FNO then attention) → **chaotic** (D_KY=20.92) |

Two earlier mechanistic explanations were proposed and then **explicitly
falsified by the next experiment**:

- *"Softmax = convex combination = structurally contractive"* — falsified
  because plain `mlp` (no softmax anywhere) recovers chaos even more cleanly
  than `fno_vit` did, with no attention-related mechanism to appeal to at all.
- *"Unconstrained-to-amplify vs. softmax-constrained"* — falsified because
  `mlp` has the exact same freedom to amplify as FNO's measured singular
  values (up to 2.5x), yet needs no such measurement to justify its win.

The surviving pattern, confirmed directly via `local_mlp`'s
receptive-field-boundedness test (Section 10) and reconfirmed via
`masked_mlp_warm_start`'s dense endpoint (Section 11, `attn_window=None`
recovering `D_KY=20.51` after a purely-local pretrain phase): **only the
conjunction of global reach AND unconstrained mixing works.** Neither
property alone is sufficient — global+softmax collapses just as surely as
local+anything.

### 3.2 Hypothesis (falsifiable)

> **H-PROP: A latent propagator recovers the true chaotic KS attractor
> (`D_KY≈22`, positive Lyapunov spectrum) if and only if its architecture
> contains at least one operation with (a) full `d_latent` receptive field
> and (b) mixing weights not constrained to a bounded/normalized simplex
> (i.e., not softmax-attention).**

**Evidence for**: every propagator satisfying (a)+(b) in this project's
history recovered chaos (`mlp`, `fno_vit`, `masked_mlp` at `attn_window=None`,
every `fourier_mlp`-family propagator in Sections 71-88 via its dense/global
Fourier-features path). Every propagator failing (a) or (b) collapsed
(`vit` at any window including full/global, `local_mlp`, `masked_mlp` at any
finite window).

**A second, narrower mechanism** (Section 43) partially explains *why*
softmax specifically biases toward contraction: softmax-weighted aggregation
is a convex combination of value vectors (weights sum to 1), giving attention
an architecture-level ceiling on amplification a dense Linear layer lacks.
But this is a contributing factor, not the deciding variable — H-PROP's
global-reach requirement is what actually separates every success from every
failure in the grid.

**Falsifiable predictions for future runs** (see §5 for the concrete
proposed experiments): `fno_mlp` (FNO + plain per-token MLP, literally zero
attention) should succeed — this removes attention from a known-successful
family entirely, isolating whether attention was ever load-bearing in
`fno_vit`'s own success. A `cnn` backbone with a **full** (non-local) kernel
width should also succeed, since a wide circular convolution is global +
unconstrained + additionally translation-equivariant (a property neither
`mlp` nor `fourier_mlp` has by construction). Neither has been tried.

### 3.3 Once chaos is recovered, accuracy/DA-skill/smoothness differentiate the winners

Once an architecture clears the H-PROP bar, essentially every "successful"
run in this project's history lands in a fairly tight `D_KY≈20.9-22.3` band —
the real differentiation between Section 52, 75, 81, 85 is accuracy, DA
skill, conditioning, and (newly) smoothness, not "how chaotic." This matches
finding #10 in the extraction (chaos richness, accuracy, DA skill, and D4
equivariance are four largely-independent axes) — a model doesn't need to be
"more chaotic" to be better, it needs to be more accurate/smoother/better
assimilated while staying inside the chaotic regime at all.

### 3.4 A propagator's learned weights don't transfer across AEs

Section 31 directly tested warm-starting an already-well-converged
propagator onto a *different*, same-architecture-family AE: it never
recovered, staying ~10x worse than that exact propagator's own converged
value on its original AE, even well past the point the original needed to
converge. **A trained propagator is tightly fit to its own AE's specific
latent coordinate geometry and is not a portable prior** — any architecture
search that changes the encoder needs to re-fit (or at least meaningfully
fine-tune) the propagator from scratch, not just transplant a good one.

---

## 4. Regularizers: what fights what

| Regularizer | Targets | Collapse mode it fixes / risks |
|---|---|---|
| `w_var`/`decorr_var_loss` | pulls every channel toward variance 1 | Fights healthy strong channels too; diluted gradient over `d` channels — not sufficient alone against organic collapse (Section 34). |
| `w_var_floor` (3a, VICReg-style) | per-channel std floor | **Provably inert** against correlation-driven collapse (Section 38: exactly zero loss at convergence, while the covariance's smallest eigenvalue was still catastrophic) — only fixes marginal-variance collapse, structurally blind to joint-eigenvalue collapse. |
| `w_logdet` (3b, log-det barrier) | whole covariance eigenspectrum | The actual fix for this project's real, organically-occurring collapse mode (correlation-driven). Pushed `cond#` from 10^5-10^6 to single digits in Section 38. **Use this, not `w_var_floor`, for anti-collapse.** |
| `w_spatial` (signed, `spatial_coherence_loss`) | rewards banded correlation between nearby latent-index channels | Genuinely opposed to `w_logdet` (isotropy vs. banded correlation pull the same eigenspectrum in opposite directions, Section 39) — navigable at the right ratio (`w_spatial=0.05, w_logdet=0.005` in the original scale; `0.01`/`0.008` in the 71-88 arc's rescaled recipe), but pushing either too far in isolation breaks the other. At high absolute weight (Section 47: `w_spatial_signed=0.08`) it can degrade the Lyapunov spectrum itself, not just conditioning (`D_KY` crashed to 7.97). |
| `lambda_z` (`RegConfig`, banded-value mechanism, **flagged, never recommended**) | pulls index-nearby coordinates toward *equal values* | Degenerate global optimum = literal collapse to a constant; mechanistically different from `w_spatial` (value-proximity vs. correlation) but explicitly not-recommended since 2026-08-29. **Correction found this session**: the `--lambda-z` CLI flag genuinely is this mechanism (an earlier project doc wrongly claimed otherwise) — Sections 75/81/82 all had a small dose (0.0002-0.0003) active, below the reference project's documented 5e-3 collapse threshold. Section 85 dropped it entirely with no downside. |
| `w_smooth` (temporal smoothness, **new this session**) | real-trajectory step size + curvature, `z_t` vs. `z_{t+1}` | Genuinely different axis from `w_spatial` (cross-sectional, same-instant) and from the propagator's own Jacobian norm (mixes wanted chaotic expansion with unwanted roughness). **Architecture-dependent effect, not yet explained**: improved *every* axis simultaneously on the `vit_fourier_hybrid`/history-mode lineage (Section 85 vs. 81), but made *every* axis worse on the `vit`/mlp-markovian lineage (Sections 86, 87 vs. 52) even though its own narrow objective (reduced curvature) was separately confirmed working in both cases. |

**Hypothesis (falsifiable), on the `w_smooth` architecture-dependence**:

> **H-SMOOTH: `w_smooth`'s benefit depends on whether the propagator has
> access to more than one past state.** A `mode="history"` propagator (81/85,
> `n_history=2`) can recover local rate-of-change information from its own
> multi-state input even as the encoder is pushed smoother, while a
> `mode="markovian"` propagator (52/86/87, single-state input) loses exactly
> that information when the encoder is smoothed, since a smoother embedding
> compresses the very step-to-step differences a markovian map needs to infer
> the system's velocity.

**How to test directly**: train Section 52's *exact* ViT+mlp architecture
with `mode="history"` (`n_history=2`) instead of `markovian`, both with and
without `w_smooth`, holding every other hyperparameter fixed. If the
history-mode version shows a clean multi-axis win (matching 85's pattern)
while a fresh markovian control (matching 86/87) regresses again, H-SMOOTH is
confirmed; if history-mode also regresses, the effect is about `vit_fourier_hybrid`
specifically, not `mode`, and a different hypothesis is needed.

---

## 5. Architectures / experiments worth trying next

In priority order, each tied to a specific open question or falsifiable
hypothesis above rather than a speculative "new idea":

1. **`fno_mlp` propagator (FNO + plain per-token MLP, zero attention).**
   Directly tests H-PROP by removing attention entirely from a
   known-successful family (`fno_vit`) — cheap, and the backbone already
   exists in the codebase, unused for a real Lyapunov/chaos-fidelity check.
2. **`node` (neural-ODE) propagator, evaluated for Lyapunov/D_KY fidelity.**
   This is the most literal "PDE model" already in the codebase (continuous
   `dz/dt = f_theta(z)`, RK4-integrated) and directly serves the project's
   stated end goal, yet it has *never* been run through Gate 3/4 — only
   flagged for an MPS circular-padding performance issue at wide kernel
   widths. Worth a real evaluation at a narrow-then-progressively-wider
   kernel, watching for the same local→global collapse transition H-PROP
   predicts.
3. **`cnn` propagator with a FULL (non-local) kernel width.** A third,
   independent test of H-PROP, and — unlike `mlp`/`fourier_mlp` — genuinely
   translation-equivariant by construction, which could combine with D4
   equivariance for a more literally PDE-like propagator than anything
   tried so far.
4. **The propagator-capacity confound, isolated for real.** Flagged
   repeatedly (Section 34, Section 78, the 2026-09-04 handoff) but never
   actually launched: train a *larger* propagator (`--aux-hidden`/
   `--aux-blocks` increased proportionally to `d_latent`) on the SAME
   `d_latent=56` or `64` frozen AE that currently underperforms (Sections 82,
   88) to see whether it was propagator capacity, not `d_latent` itself,
   causing the regression.
5. **A clean, single-variable `w_smooth` sweep** (e.g. 0.0001/0.0003/0.001/
   0.003) on both the 81-hybrid and 52-markovian bases *without* also
   changing `w_var`/`w_spatial`/`lambda_z` at the same time — Sections 85-87
   all confounded `w_smooth` with 2-3 other simultaneous changes, so the true
   Pareto frontier for `w_smooth` alone is still unknown on either
   architecture.
6. **Re-run Gate 3/4 + the new smoothness diagnostic on Section 16
   (`nodeltacap_e2e`).** Its raw numbers (`val_kmax_mse=0.006546`,
   `D_KY=21.99`, `|D_KY-22|=0.01` — both the best in the entire project) were
   never checked against the D8/smoothness tooling built in Sections 71-88.
   If it's *also* smooth and well-behaved, it may be a stronger base than
   52/81/85 for everything; if it's rough or poorly conditioned, that's a
   useful data point on its own.
7. **Test H-SMOOTH directly** (Section 52's architecture under
   `mode="history"`, with and without `w_smooth`) — see §4.

---

## 6. Best 3 models, by criterion

Recommending three genuinely different models rather than picking one
overall winner, since the project's own evidence (§1, §3.3) says accuracy,
dynamical fidelity, and geometric/stability quality are separate axes.

### 6.1 Faithful dynamics modeling — **Section 16 (`nodeltacap_e2e`)**

- `val_kmax_mse = 0.006546` — best raw rollout accuracy of the **entire**
  project (roughly 4x better than Section 52, the best of the 71-88 arc).
- `D_KY = 21.99`, `|D_KY - 22| = 0.01` — closest to the true benchmark of
  any checkpoint in the whole document.
- `n_positive = 12`, `λ1 = +0.0885`, DA `skill_free_over_da = 3.43`.
- D7 significance confirmed under the corrected null (Section 22 retroactive
  check, `p=0.018`) — real, not spurious, local structure.
- **Caveat**: predates the `w_spatial`/`w_logdet` recipe (Section 39) and
  the D4/smoothness tooling entirely — never checked for translation
  equivariance or real-trajectory smoothness. Recommended as the top pick
  for raw dynamical fidelity, but re-running it through the modern
  diagnostics (§5, item 6) is the natural next step before fully trusting it
  over Section 52 for anything beyond accuracy/`D_KY`/DA skill.
- **Runner-up**: Section 52 — `val_kmax_mse=0.0256`, `D_KY=21.419`,
  `skill_free_over_da=3.123` (best DA skill of the modern 71-88 arc), and
  independently confirmed as the *smoothest* embedding of that comparison
  set. Weaker than `nodeltacap_e2e` on raw accuracy/`D_KY` but has the
  modern diagnostic suite already run.

### 6.2 PDE modeling readiness — **Section 85**

- `val_kmax_mse = 0.0279`, `D_KY = 21.668` (closest to true of the
  `vit_fourier_hybrid` lineage), `cond# = 24.0` (well-conditioned).
- **D4: "Genuine (approximately) equivariant translation representation
  found"** — a literal PDE-relevant symmetry (KS is translation-invariant on
  its periodic domain); Section 52 does *not* have this property.
- Smoothest embedding **and** smoothest learned dynamics of the whole
  `vit_fourier_hybrid` family measured this session: step size/curvature
  both lower than Section 81, and the lowest propagator-Jacobian norm
  (median 1.60, p95 1.74) of the compared set — the least locally-jagged
  trained dynamics found.
- D7/D8 spatial coherence confirmed significant (`p=0.0000`).
- This is the single model in the whole project combining *smoothness*,
  *equivariance*, and *good accuracy/D_KY* simultaneously — exactly the
  profile needed before attempting to literally fit a continuous governing
  equation to the latent trajectory.
- **Runner-up**: Section 81 (same architecture family, one step back on
  every axis — `val_kmax_mse=0.0286`, `D_KY=20.914`, also D4-equivariant).

### 6.3 Stability — **Section 85** (again — see caveat below)

- Lowest propagator-Jacobian p95 (1.74) of the entire compared set,
  including Section 52 (2.90) and Section 81 (1.75) — the least prone to a
  sharp, locally-expansive blow-up at any single step.
- Free-running rollout error saturates cleanly around the expected
  `√2` bound rather than diverging (consistent with every well-behaved
  checkpoint in this project — `delta_cap` was conclusively shown
  unnecessary for this, Sections 15-16, so a low Jacobian norm from the
  *learned dynamics itself* is the real signal, not an architectural safety
  net).
- **Caveat**: Section 85 is a single run from this session, not yet
  stress-tested across seeds or long-horizon autonomous rollouts the way
  Section 52/81 have been (indirectly) via repeated reuse across many
  downstream sections. If a second, independently-trained instance of
  Section 85's recipe is wanted before fully trusting this pick, treat
  **Section 81** as the safer, more battle-tested alternative — nearly as
  smooth, and it's been the base architecture for 4 further experiments
  (82, 83, 85) without ever producing a numerical instability or divergence.

---

## 7. How to verify anything in this document

```bash
cd /Users/daltonjones/Documents/latent_DA
# Sections 71-88 (this project's own logs):
grep -E "val_kmax_mse|D_KY|n_positive|skill_free_over_da" artifacts/logs/gate3_analysis_section{75,81,82,85,86,87}_*.log
cat artifacts/logs/gate3_da_section85_*_warmstart_k12_300ep.log
mamba run -n da_env python scripts/analyze_latent_smoothness.py   # re-run the smoothness comparison table

# Sections 1-70 (earlier history): docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md,
# docs/RESULTS.md (Phases 1-3), docs/OPEN_QUESTIONS.md
```

This document's Section 1-70 material was produced by a full read of
`docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md` (4449 lines) plus `docs/RESULTS.md`
and `docs/OPEN_QUESTIONS.md`; `docs/LATENT_PDE_EXPERIMENTS.md`,
`docs/LATENT_PDE_RESEARCH_NOTES.md`, and `docs/ML_for_KS_writeup.md` were
**not** read in full for this pass — if a recommendation here seems to
conflict with something in those files, re-check them directly before
trusting this document over them.
