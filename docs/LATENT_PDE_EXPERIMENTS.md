# Latent PDE Experiments: a concrete plan

**Status:** proposal, not yet run. First written 2026-09-02 off the
`section52` checkpoint; revised same day twice more after user pushback —
see the two notes below, both still load-bearing.

**Revision 1.** The first version argued that because the encoder pools
globally (`pool="mean"`), no propagator could discover local dynamics, and
made an architectural rebuild the necessary next step. **That was wrong** —
it conflated the encoder's receptive field (which physical locations feed
`z_k`) with the propagator's dynamical locality (whether
`∂z_{n+1,k}/∂z_{n,l}` is banded). §3 keeps the corrected reasoning: a
globally-supported encoding can still produce maximally local dynamics
(§3.1's Fourier counterexample), so global pooling does not by itself rule
anything out.

**Revision 2 (user-directed, 2026-09-02): the coordinate is fixed, not
searched.** Revision 1's fix over-corrected into proposing a search over
alternative coordinates (permutations, rotations, learned reparametrizations)
before testing for local dynamics at all. **User direction: don't do that.**
The whole point of `w_spatial_signed` was to train the encoder to produce
local structure *in its native, already-emitted index* — and that structure
is already independently confirmed (D8, and visually in the Hövmoller
figure). The question this document now answers is narrower and more
direct: **does a local PDE fit the dynamics in the latent's current
ordering, as-is, with no coordinate search.** §3.3 records the
broader-search idea as considered and explicitly out of scope, per repo
convention for documenting rejected alternatives, rather than deleting the
reasoning entirely.

**Relationship to other docs.** `CLAUDE_CODE_BRIEF.md` §11-16 (Phases 9-14)
lays out the long-range program: a local latent field, a stencil
propagator, a gauge-invariant PDE/ODE verdict, localized DA and L-transfer.
Almost none of that code exists (`ks_latent/discovery/`, Phase 8's
`pde_find.py`, Phase 9's closure test, and `ks_latent/models/
local_field.py` don't exist). This document doesn't replace that plan — it
sequences the near-term, native-index experiments that test whether a PDE
exists at all before any of Phase 10-12's machinery is built.

**Why now.** `docs/figures/latent_hovmoller_section52_..._300ep.png` shows
the `section52` propagator's free rollout tracking the true encoded
trajectory's banded structure for roughly the first third of the window,
then drifting. That structure is not incidental: `section52` is the
endpoint of a deliberate 9-run sweep (`docs/PHASE2_ARCHITECTURE_
EXPERIMENTS.md` §§44-52) that ramped a *signed* local-coherence loss
(`spatial_coherence_loss(..., signed=True)`, the training-time counterpart
of the D8 diagnostic) while a log-det barrier held off collapse. This
document is about whether that induced structure supports a PDE, tested
directly in the coordinate the loss was built to shape.

---

## 1. What a latent PDE actually requires

| # | Requirement | Precise statement | Status today |
|---|---|---|---|
| **R1** | **A coordinate** | fixed, by decision, to the native latent index — the one `spatial_signed` shapes and D8 measures | **settled by decision, not search** |
| **R2** | **A resolved field** | along the native index, the state is smooth enough that `δ¹z, δ²z, …, δ⁴z` are signal, not sampling noise (`\|δ¹z\| ≪ \|z\|`) | **never measured** |
| **R3** | **Bounded interaction range** | `∂ż_j/∂z_i` decays with `\|i−j\|` on a scale `≪` the ring, in the native index | **not established (D3, §2.1)** — this document's central open question |
| **R4** | **Homogeneity** | the *same* `F` acts at every site (weight sharing), else it is a lattice ODE with site-dependent coefficients, not a PDE | untested |
| **R5** | **Closure** | `ż_j` is determined by `{z_{j±w}}` at one time (plus at most short memory) | untested (Phase 9 never built) |
| **R6** | **A resolved time derivative** | `∂_t z` is recoverable from the data cadence to useful accuracy | marginal — `dt_snap=1.0`, §2.4 |

Two further requirements are real but **downstream of the PDE claim, not
part of it**:

| # | Requirement | Needed for | Needs |
|---|---|---|---|
| **R7** | **Physical localization** | Phase 13's Gaspari–Cohn DA tapering (a *physical distance* between latent coordinates) | bounded **decoder** receptive field |
| **R8** | **Size-extensibility / equivariance** | Phase 13's L-transfer (run at `L=200` with no retraining) | an architecturally weight-shared, receptive-field-bounded **encoder** |

Nothing in R1-R6 requires R7/R8, and this document doesn't chase them —
they belong to the DA/L-transfer program (brief Phase 13), not to "does a
PDE fit the current latent."

**R2 is the cheapest decisive test and has never been run.** A 4th-order
difference operator on an index-white field is pure noise amplification;
the whole discovery program is meaningless without it.

### 1.1 The formulation this document pursues

`∂_t z_j = F(z_j, δ¹z_j, δ²z_j, δ³z_j, δ⁴z_j)`, `j` = the native latent
index, `h` a nominal unitless spacing. This is exactly what
`spatial_signed` is implicitly aiming at, and it is the only formulation
this document tests. Two related formulations exist and are **not**
pursued here, kept for completeness:

- **A physical-coordinate version**, using D1 decoder-sensitivity centroids
  `X_k` as physical locations instead of the raw index `j`. Would give
  derivatives physical units and direct comparability to Interpretation
  C's KPZ/Burgers prediction, but the centroids are currently weak
  (median resultant length `R = 0.131`, §2.1) and this isn't the ordering
  the user wants tested — noted here only because it's what a bounded-
  receptive-field encoder (Block C) would eventually unlock.
- **Interpretation C** (the effective large-scale PDE of KS itself, no
  autoencoder) — brief Phase 8, independent of the latent entirely, useful
  later as a validation target for whatever coefficients fall out (§5,
  Block C.5).

---

## 2. Where the project stands

`docs/RESULTS.md` stops at Gate 4. Everything below is from
`docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md` §§34-43, the
`docs/diagnostics_report_section4x/5x_*.md` files, and checkpoint
metadata — none of it folded into `RESULTS.md` yet.

### 2.0 The current best models

| quantity | `section52` | `section53` (newer, bigger: `d_model=108`) |
|---|---|---|
| Stage-1 final recon | 0.000308 | — |
| Stage-2 `val_kmax_mse` (k=12) | 0.0256 | **0.0186** (better) |
| Latent `D_KY` | **21.42** (target 21.4 ± 1.5) | not run |
| positive exponents / `λ1` | 11 / 0.083 | not run |
| DA skill (free/DA) | 3.12× | **2.13× (worse)** |
| DA analysis RMSE | 0.283 (target 0.14 ± 0.04) | 0.440 |

**Note the inversion**: `section53` improved 12-step prediction MSE by 27%
and got *worse* at DA (skill 3.12× → 2.13×, RMSE 0.283 → 0.440) —
first-party evidence for the "weather vs. climate" caveat already raised in
§43 of the architecture doc. **`val_kmax_mse` is not a sufficient objective
for this program**, and no step below should be graded on it alone (§6's
decision table requires attractor statistics for every cell).

### 2.1 What's already known about the native index

| evidence | result | reading |
|---|---|---|
| **D8** signed same-time bandedness, native index | **0.6466, p = 0.0000**; monotone in the training weight across §§48-52: 0.046 → 0.19 → 0.47 → 0.57 → **0.65** | the loss works: the native index carries genuine, tunable, signed near-neighbour *correlation* |
| **D1** decoder-sensitivity centroids | span 12.3-94.3 of `L=100`; mean gap **1.91** vs. **2.27** for a uniform tiling; median resultant length `R = 0.131` | channels have *some* preferred location, but weakly |
| **D3** propagator Jacobian bandedness | 0.2920, **p = 0.184 — not significant**, flat (0.26-0.30) across the *entire* §§44-52 sweep | the dense `mlp` propagator's Jacobian, in this index, is not banded |

**The gap this document exists to close:** D8 (same-instant correlation) is
strong and real; D3 (one-step dynamical Jacobian) has never shown
bandedness, across nine training runs that all used the same dense `mlp`
propagator. §3.1 explains why that null result is not conclusive on its
own — a dense propagator with no locality bias has no particular reason to
develop a banded Jacobian even when the underlying structure would support
one — and §5's Block B is the direct test: does a propagator *architecturally
restricted* to a narrow window of the native index match the dense one's
accuracy?

### 2.2 The encoder's receptive field: relevant to R7/R8, not to this document's question

`section52`/`section53` use `pool="mean"`:

```
NX=256, patch_size=8            -> n_tokens = 32, each token = 3.12 physical units
attn_window=4, n_blocks=3       -> attention reach = 12 tokens = 37.5 physical units
pool="mean", pool_window=1      -> GLOBAL mean over all 32 tokens, then Linear(d_model -> 44)
```

so every `z_k` is nominally a function of all of `u` — D1 confirms this is
not merely nominal (median sensitivity support ≈32 physical units, vs. the
light cone's 1.26). **This number bounds R7/R8 (§1), not R1-R6.** §3.1
gives the concrete reason: bounded encoder support is not a precondition
for the *dynamics*, expressed in whatever coordinate the encoder settles
on, to be local.

### 2.3 R2, R4, R5: never measured

- **R2 (resolvedness)** — nobody has looked at the latent's power spectrum
  *along the native index*, or at `‖δ¹z‖/‖z‖`, `‖δ²z‖/‖z‖`. §5.A.1.
- **R4 (homogeneity)** — untested; `masked_mlp` (per-site weights) vs.
  `cnn`/`node` (weight-shared) is a direct test, §5.B.2.
- **R5 (closure)** — untested. The brief's Phase 9 model-free conditional-
  variance test was never built. The codebase's own D6 diagnostic
  ("temporal coherence structure") **is a different thing** that reused
  the same letter — the brief's D6 is `V(w,τ) = Var[u(x,t+τ) | window]`,
  and it does not exist. §5.A.3.

### 2.4 R6 (time derivative)

`dt_snap = 1.0` with `λ1 = 0.083` means one snapshot ≈ 0.08 Lyapunov
times — fine for *dynamics*, marginal for a *derivative*: a one-sided
difference `(z_{n+1} − z_n)/Δt` at `Δt = 1.0` is first-order accurate over
an interval in which the state visibly changes. Any regression-based PDE
fit (§5.B.1) should be run on a trajectory re-generated at a finer
`snapshot_every` (solver runs at `dt = 0.05`; `snapshot_every = 2-4` gives
`dt_snap = 0.1-0.2` at negligible cost), same encoder applied to it. Data-
generation change, not a model change — skipping it will silently bias
every fitted coefficient.

### 2.5 Tooling that exists and has never been benchmarked

- `PropagatorConfig(backbone="masked_mlp" | "node" | "cnn")` — all three
  implemented, unit-tested, and with **zero real-data results anywhere in
  the repository**. `"node"` (`dz/dt = f_θ(z)`, weight-shared circular-conv
  vector field, RK4) was added 2026-08-31 explicitly as "the most literal
  reading of 'model the latent variable with a PDE'" and then never run.
- `ks_latent/analysis/spreading.py` — `comoving_exponents()` is fully
  generic, so measuring a light cone *in the latent index* is glue code
  over already-validated numerics. §5.A.2.
- `pysindy` 1.7.5 is installed and importable today.

### 2.6 The known chaos-collapse hazard

Two independent first-party observations: `PropagatorConfig.delta_cap`'s
docstring (a propagator collapsing to a single fixed point, `D_KY = 0`,
under plain rollout-MSE training, because global contraction is the
cheapest way to bound rollout error), and §43's condition-number blowup
with an unconstrained dense `mlp` aux. Monitor on every new propagator
below — **a local propagator removes global mixing but does nothing to
prevent local contraction**, and `val_kmax_mse` will look *better*, not
worse, as a model contracts.

---

## 3. Why testing the native index directly is legitimate (and what's out of scope)

### 3.1 Global receptive field does not preclude local dynamics

Verified directly against this project's own operator, not just asserted.
Take KS's linear term, `u_t = u_xx`, discretized on the same `d=44`-point
periodic ring the latent lives on. One Euler step is tridiagonal
(bandwidth-1) in the physical index — banded by construction. Apply the
discrete Fourier transform, `w = Fz`, and re-express the same update:

```
physical-index encoder (identity): off-band(w=1) fraction of matrix mass = 0.0000   [banded by construction]
Fourier-index encoder (global per-coefficient): off-band(w=0) fraction  = 0.0000   [diagonal]
```

Every DFT coefficient is, by construction, a function of every physical
grid point — 100% receptive field — yet the transformed dynamics is not
merely local, it's exactly diagonal. **This is why the encoder's global
pooling doesn't pre-judge the answer**: it's entirely possible for
`section52`'s trained, `spatial_signed`-shaped index to carry local
dynamics despite the encoder's support being broad, and the only way to
know is to test it directly (§5) rather than infer it from the pooling
architecture in either direction.

### 3.2 What genuinely does need a bounded, architectural receptive field

Keep three questions separate:

1. **"Does a local dynamical law exist in the native index?"** (R1-R6, this
   document) — does not require a bounded encoder receptive field, per
   §3.1. Tested directly in §5.
2. **"Can we localize DA updates?"** (R7) — genuinely requires a bounded
   **decoder** receptive field, since Gaspari–Cohn tapering needs a
   physical distance between latent coordinates. Real, separate, not
   pursued here.
3. **"Can we run at `L=200` with no retraining?"** (R8, L-transfer) —
   genuinely requires the encoder to be architecturally size-extensible
   and weight-shared. Real, separate, not pursued here.

### 3.3 Considered and out of scope: searching for a different coordinate

An earlier draft of this document proposed, before testing the native
index at all, searching over alternative coordinates — a bandedness-
maximizing rotation of `z`, a DMD/Koopman eigenbasis of the propagator's
linearization, or a small jointly-trained nonlinear reparametrization in
the style of Champion, Lusch, Kutz & Brunton (2019)'s SINDy-autoencoder or
Kemeth et al. (2022)'s emergent-space construction. **User direction,
2026-09-02: not pursued.** The native index is not an arbitrary label to
be replaced — it's the specific coordinate `w_spatial_signed` was built to
shape, already verified (D8, and visually) to carry real local structure;
the question at hand is whether *that* structure supports a PDE, not
whether some other coordinate would score better. Recorded here, rather
than deleted, so it isn't silently rediscovered later: if §5's native-index
tests come back negative, this is the documented next thing to consider,
not a default.

---

## 4. Literature grounding

`docs/LATENT_PDE_RESEARCH_NOTES.md` §10 and `CLAUDE_CODE_BRIEF.md` §22
cover the core references (Kemeth et al. 2022; Bar-Sinai/Hoyer/Hickey/
Brenner 2019; Kochkov et al. 2021; Champion et al. 2019; Constante-Amores/
Linot/Graham 2024/2026; AROMA 2024; Wittenberg & Holmes 1999). What follows
is what's specifically relevant to the native-index tests in §5.

- **Bar-Sinai, Hoyer, Hickey & Brenner (2019)**, *PNAS* 116(31):15344, and
  **Kochkov et al. (2021)**, *PNAS* 118(21) — the methodological precedent
  for §5.B.2: the interaction width is a *measurement* (swept, checked for
  saturation), not a hyperparameter.
- **Brandstetter, Worrall & Welling (2022)**, ICLR, arXiv:2202.03376 —
  classical finite-difference/finite-volume/WENO schemes are special cases
  of local message passing; locality is what buys generalization across
  discretizations. The cleanest external argument that L-transfer (a
  separate, R8 concern, §3.2) should follow from locality itself, should
  that program be revisited later.
- **Cheng, Dong, Schönlieb & Aviles-Rivero (2025)**, "PDE Solvers Should Be
  Local" (FINO), arXiv:2509.26186 — global-mixing operators (spectral
  convolution, attention — this project's `vit`/`fno_vit` propagator
  backbones) oversmooth sharp local dynamics at higher cost; strict
  locality plus a learnable explicit time-stepping scheme gives better
  rollout stability, with a composition bound from one-step to
  long-horizon error. **Read the stability result with care and do not
  cite it as a chaos fix** — the bound is Lipschitz-based, and a Lipschitz
  constant below 1 is exactly the globally contractive regime `delta_cap`
  exists to prevent (§2.6). Supports the locality bet in Block B; doesn't
  address reproducing a positive Lyapunov spectrum.
- **"Watch your neighbors: Training statistically accurate chaotic systems
  with local phase space information"** (2026), arXiv:2605.14405 — trains
  chaotic surrogates so *local expansion/contraction statistics* match, not
  just pointwise trajectory error. A candidate second anti-collapse
  mechanism alongside `delta_cap` if §5.B.3 shows one is needed. Its
  "local" means local in *phase space*, a different notion from this
  document's index locality — don't conflate them.
- **"Stability analysis of chaotic systems in latent spaces"**, *Nonlinear
  Dynamics* (2024), doi:10.1007/s11071-024-10712-w — a convolutional-
  autoencoder latent can preserve the underlying system's Lyapunov
  exponents, readable directly off the latent model.
- **arXiv:2608.20404 (2026)**, "Robust Discovery of Coarse-Grained
  Continuum Equations from Microscopic Dynamics" — practical guidance for
  the latent SINDy library (§5.B.1): larger libraries hurt, use ensemble/
  bootstrap fits and report per-term selection probabilities, expect
  spurious terms suppressed by more data rather than more regularization.
- **AROMA (2024)**, arXiv:2406.02176 App. C.7 — KS as an explicit failure
  case for local neural-field latents, decoder-limited, needing recon MSE
  ~1e-10-1e-12 for an accurate energy spectrum. Not directly in scope here
  (no architecture change is proposed), but the right caveat if R7/R8 are
  revisited later.
- **Warming & Hyett (1974)**, "The modified equation approach to the
  stability and accuracy analysis of finite-difference methods,"
  *J. Comput. Phys.* 14:159-179 — the classical technique B.4.1 uses to go
  from a known-form discrete stencil to the continuum operator (and its
  truncation-error correction) it's consistent with, by formal Taylor
  expansion in `h`. Deterministic, not a regression.
- **Thaler, Paehler & Adams (2019)**, "Sparse Identification of Truncation
  Errors" (SITE), *J. Comput. Phys.* 397:108851 — automates Warming &
  Hyett's expansion with SINDy-style sparse regression, identifying
  truncation-error terms from simulation data. Direct precedent/tooling for
  B.4.1.
- **Long, Lu, Ma & Dong (2018)**, "PDE-Net: Learning PDEs from Data," ICML,
  arXiv:1710.09668, and **Long, Lu & Dong**, "PDE-Net 2.0," *J. Comput.
  Phys.* 399:108925 — moment-matrix constraints forcing a learned local
  kernel to be a consistent, specified-order finite-difference
  approximation of a chosen derivative by construction. B.4.2's answer to
  "how do you regularize the black-box-stencil version of this problem."
- **Raissi & Karniadakis (2018)**, "Hidden Physics Models: Machine Learning
  of Nonlinear Partial Differential Equations," *J. Comput. Phys.*
  357:125-141 — Gaussian-process regression as the regularized backbone for
  reconstructing a smooth field from discrete samples with no assumed
  stencil form; notably demonstrated on KS among the paper's own examples.
  B.4.3's method if 1-2 both prove awkward.

**Not pursued, per §3.3:** Champion, Lusch, Kutz & Brunton (2019), *PNAS*
116(45):22445 (SINDy-autoencoder); Kemeth et al. (2022), *Nat. Commun.*
13:3318 (emergent-space PDEs); Lusch, Kutz & Brunton (2018), *Nat. Commun.*
9:4950 (deep Koopman embeddings); Williams, Kevrekidis & Rowley (2015),
*J. Nonlinear Sci.* 25(6):1307 (Extended DMD) — all methodologically sound
precedent for *finding* a coordinate, kept here as the citation trail for
§3.3's "considered, not pursued" note rather than as active method
references.

**Prior-art position, unchanged.** Constante-Amores, Linot & Graham
(arXiv:2410.01238) remains the closest work (patch-decomposed AE + shared
NODE on KS, motivated by the same extensivity argument, architecturally
local from the start rather than testing an existing globally-pooled
encoder). Differentiators: (i) the continuum-limit test done
gauge-invariantly, (ii) latent-space localization for DA, (iii)
L-transfer — all deferred here as R7/R8 concerns, §3.2.

---

## 5. The experiments

Two blocks. Block A costs no training and can settle whether Block B is
worth running at all. Costs are on the scale already in `docs/RESULTS.md`
(Stage-1 ~1400-3200 s, Stage-2 minutes, on this machine's MPS).

### Block A — Zero-training audit of the native index (days, no GPU-hours)

Everything here runs on the frozen `section52` checkpoint (and, where
cheap, `section53`), in the native latent index, no coordinate change.

**A.1 — Is the latent a resolved field along its index? (R2)**
Compute, on encoded validation trajectories: the power spectrum of `z`
along the native index (rfft over the 44-ring, time-averaged), and the
ratios `‖δ¹z‖/‖z‖`, `‖δ²z‖/‖z‖`, `‖δ⁴z‖/‖z‖` for centred circular
difference operators. Compare against 100 random-permutation nulls — a
*control* confirming any structure found is a property of the native
index specifically, not an artifact that would appear under any
relabeling.
*Decision rule:* the index-spectrum must decay by at least a factor of 10
from lowest to highest index-wavenumber, and `‖δ¹z‖/‖z‖ < 0.5`, outside the
permutation-null band, for a difference-operator library to be meaningful.

**A.2 — The latent light cone (R3, an a priori width prediction)**
Perturb one latent coordinate on an on-attractor state, roll the trained
propagator forward alongside the unperturbed trajectory, record
`ln|δz_j(t)|` as a `(T, 44)` profile in the native index, and feed it to
`ks_latent.analysis.spreading.comoving_exponents(times, log_abs_profiles,
x0=j0, L=44, v_grid=…)` — generic, already validated; the wraparound guard
(`|v|·t ≤ 0.45L`) applies unchanged. Report `Λ(v)`, `v_*^latent`
(sites/step) via `find_v_star`, and the front-tracking cross-check,
averaged over ≥100 sites and base states.
*Why this matters:* a **direct** measurement of interaction range in the
dynamics — stronger than D3's static Jacobian bandedness, which is a
one-step linearization and (per its own docstring) blind to structured
broadband coupling. Gives `w_predicted = v_*^latent · 1 step`, which
Block B's width sweep then confirms or refutes: two independent estimates
of one number, the pattern Phase 2 already used for `v_*`.
*Decision rule:* `v_*^latent < 3` sites/step ⇒ a genuinely local latent
light cone exists in the native index; `v_*^latent ≳ 10` ⇒ information
crosses the ring in a few steps (try reducing `dt_snap`, §2.4, since the
light cone scales with `Δt`, before concluding the native index can't
support a narrow stencil).

**A.3 — Model-free closure test (R5; brief Phase 9, never built)**
`V(w, τ) = Var[z_j(t+τ) | z_{j−w..j+w}(t)]` by k-NN conditional variance,
swept over `w` and `τ`, against `V(∞, τ)`, in the native index. Two
numbers fall out: the `w` at which `V` saturates (the information horizon,
checked against A.2's light cone — a third independent estimate of the
same quantity), and the residual at large `w` (the Mori–Zwanzig memory,
bounding what any Markovian local model can achieve). Run also on a
coarse-grained physical field at 44 points as a calibration control, where
the answer is essentially known.
*Naming collision:* the brief's D6 is this test; the codebase's D6 is
"temporal coherence structure," a different diagnostic that reused the
letter. Name this one `closure_test`, not D6.

**A.4 — Encoder/decoder receptive field, for R7/R8 only**
Measure the receptive field empirically by gradient masking (brief §12.4's
`test_receptive_field` methodology), feed to the already-implemented
`minimum_localization_radius(v_star, dt, stride, encoder_receptive_field)`.
Record in `docs/RESULTS.md`, explicitly scoped: this gates R7/R8 (DA
localization, L-transfer, out of scope per §3.2/3.3), **not** A.1-A.3.

> **Gate A.** Write A.1-A.4 into `docs/RESULTS.md` with the decision rules
> above. A pass on A.1/A.2 is the go-ahead for Block B's stencil sweep. A
> fail is a real, complete, reportable answer in its own right — "the
> native index carries same-time correlation (D8) but not resolved,
> bounded-range dynamics" — and per §3.3 the documented next step, if this
> is revisited, is the coordinate search, not something to default into
> now.

### Block B — Local propagator sweep, in the native index

**B.1 — Direct PDE regression on encoded data (no propagator at all)**
The most direct possible test, and it does not require training anything.
Re-generate a trajectory at `snapshot_every` giving `dt_snap ≈ 0.1` (§2.4),
encode it, and fit `∂_t z_j = F(z, δ¹z, δ²z, δ³z, δ⁴z)` in the native
index by **weak-form** SINDy (`pysindy` 1.7.5, already installed) with
coefficients **shared across sites** (R4 imposed, then relaxed in B.2 as a
test), sweeping the stencil width used to build the library.
*Validate the pipeline first, on data with a known answer* — non-negotiable,
the brief's own rule (§10, Phase 8, "the most important test in the
phase"): the same code must recover `u_t = −uu_x − u_xx − u_xxxx` and
heat/Burgers/KdV–Burgers from raw solver output to 1% before it is pointed
at any latent. That validated module (`ks_latent/discovery/pde_find.py`)
is a prerequisite of this plan and does not exist yet.
*Report* per-term selection probabilities over bootstrap folds, not a
single sparse fit (arXiv:2608.20404), plus the fraction of `∂_t z`
variance explained as a function of stencil width — the model-free
version of B.2's saturation width.

**B.2 — Local propagator sweep, with the controls that make it mean something**
Freeze the `section52` AE. Train Stage 2 with `--backbone {masked_mlp,
cnn, node}` sweeping the radius over `{1, 2, 3, 4, 6, 8, 11}` in the
native index, `--delta-cap` on, otherwise identical to `section52`'s
recipe:

```
python scripts/train_stage2_patched.py --profile full \
  --ae-checkpoint artifacts/stage1_ae_patched_full_section52_..._200ep.pt \
  --backbone cnn --cnn-kernel-size {2w+1} --delta-cap {cap} --k-max 12 --epochs 300
# and --backbone node --attn-window {w} --ode-substeps 4
# and --backbone masked_mlp --attn-window {w}
```

**Four control arms, without which a positive result proves nothing:**

1. **Random index permutation** — same sweep, but on a randomly permuted
   latent index (a null, not a search for a better one). If a narrow-window
   model does just as well there, the "locality" found is an artifact of
   latent redundancy, not structure specific to the trained ordering.
   (`ks_latent/models/permuted_autoencoder.py` and the existing
   `diagnostics_report_permuted_*` runs are precedent.)
2. **`w_spatial = 0` AE** — the same sweep on a checkpoint trained without
   the coherence loss, isolating what `spatial_signed` actually bought.
3. **Coarse-grained physical field at 44 points** — the same stencil
   family fit to `G_ℓ * u` sampled at 44 points
   (`ks_latent/solver/filtering.py` exists). At `h = 2.27 ≪ 8.89` this is
   still a well-resolved KS field, so a narrow stencil *must* work: the
   pipeline's known-answer upper bound, calibrating every number in the
   sweep.
4. **Dense `mlp` baseline** — `section52` itself, `val_kmax_mse = 0.0256`.

**Sweep the history length jointly with the width** (`mode="markovian"` vs.
`mode="history"`, `n_history ∈ {2, 3}` — both already exist and were both
used in §§44-52). R5 predicts a trade-off: a narrower stencil should need
more memory, since truncating space and truncating memory are the same
Mori–Zwanzig approximation seen from two sides. A 2-D `(w × n_history)`
table is the deliverable, not a 1-D width curve.

**B.3 — Chaos preservation, monitored per cell, not at the end**
For every `(backbone, w, n_history)` cell that passes on accuracy, run
`scripts/run_analysis_suite.py` and record `D_KY`, `n_positive`, `λ1`,
plus long-rollout attractor statistics: latent marginal/joint
distributions, temporal autocorrelation, and the **decoded time-averaged
energy spectrum**. The `section52`-vs-`section53` inversion (§2.0) is the
standing warning that these can move opposite to `val_kmax_mse`. If
`delta_cap` proves insufficient at narrow widths, arXiv:2605.14405's
local-phase-space training signal is the next lever — **new code, to be
built only if this step shows it is needed**, not speculatively.

> **Gate B.** A width at which accuracy saturates, with the light-cone
> prediction from A.2 and the closure width from A.3 next to it, and the
> four controls' numbers alongside. Three independent estimates of one
> number is the strongest form this result can take; report agreement or
> disagreement prominently either way. Also report the `masked_mlp`
> (per-site) vs. `cnn`/`node` (weight-shared) gap: a large gap in favour of
> `masked_mlp` is evidence **against** R4, i.e. evidence for "lattice ODE
> with site-dependent coefficients" rather than "a PDE."
>
> **A pass here is most of the deliverable this document was asked for: a
> local PDE, `ż = F(z, δ¹z, …, δ⁴z)`, fit in the latent's current
> ordering.** B.4 below is what turns the winning propagator into an
> explicit continuous-`X` differential operator — it's the answer to "how
> do we actually get from a learned stencil to a continuum PDE," not
> optional polish.

**B.4 — From a fixed-window stencil to a continuum operator**
Two genuinely different problems hide under "extract the PDE," and they
need different tools. Which applies depends on which backbone wins B.2.

1. **The winning stencil has an (approximately) known closed form —
   `masked_mlp`/`cnn`, small window, near-linear residual.** Recovering
   the continuum operator is then a deterministic Taylor expansion, not a
   regression: write each neighbor as `z_{j+k} = z(x_j + kh)`, expand the
   stencil in powers of `h` treating `h` as a formal small parameter, and
   match order by order. This is **modified equation analysis** (Warming &
   Hyett, 1974, *J. Comput. Phys.* 14:159 — the classical technique for
   asking what PDE a discrete scheme actually solves, truncation error
   included) and it requires no regularizer, because nothing is being
   inferred from scattered data — the stencil's coefficients are already
   known, and the expansion is exact order by order. The leading correction
   term is itself diagnostic (numerical-diffusion/dispersion-type
   artifacts, in the classical language, would show up here as spurious
   extra terms the training introduced beyond the "intended" derivative).
   An automated, data-driven version of exactly this exists and is directly
   reusable: **Thaler, Paehler & Adams (2019)**, "Sparse Identification of
   Truncation Errors" (SITE), *J. Comput. Phys.* 397:108851 — SINDy plus
   modified-equation analysis, built to identify these correction terms
   from simulation data rather than by hand.
2. **The winning stencil is a genuinely nonlinear black-box network over
   the window — the `node` backbone, or a `cnn`/`masked_mlp` whose fitted
   map isn't close to linear.** A bare MLP over a window has no built-in
   notion of "this channel means `∂²_X`," so modified-equation matching has
   no target to lock onto — this is the actual ill-posed inverse problem:
   many different continuous operators are consistent with the same finite
   window of samples, and picking one needs a regularizer. Rather than
   solving that inverse problem after training, the precedented fix builds
   the regularizer into the architecture: **PDE-Net** (Long, Lu, Ma & Dong,
   ICML 2018, arXiv:1710.09668; PDE-Net 2.0, *J. Comput. Phys.* 399:108925)
   constrains each learned local kernel via a **moment-matrix constraint**
   — a linear condition on the kernel's discrete moments
   (`Σ_k kᵅ q_k = c_α`) forcing it, by construction, to be a consistent
   order-`p` finite-difference approximation of a *specified* derivative,
   while still letting the network learn corrections beyond the textbook
   coefficients. If the `node`/`cnn` backbone wins B.2, refit its local
   kernel with these moment constraints (rather than a free kernel) as a
   direct test of whether the learned correction beyond the standard
   stencil is large or small.
3. **A third, weaker-assumption version, if 1-2 both prove awkward:**
   reconstruct a smooth field `z(X)` from the 44 discrete site values with
   *no* structural assumption on the stencil at all — classical minimum-
   norm/RKHS interpolation (a natural cubic spline minimizing `∫(z'')²`
   subject to matching the data is the simplest instance). The
   PDE-discovery-specific version is **Raissi & Karniadakis (2018)**,
   "Hidden Physics Models," *J. Comput. Phys.* 357:125-141 — Gaussian-
   process regression as the regularized backbone for exactly this
   recovery, notably demonstrated on KS itself among their own examples.

**Scope note, so this isn't oversold:** all three routes give a continuous
differential operator that is *consistent with the fit at the single
resolution `h = L/44` this document trains at* — a legitimate, literal
answer to "what continuum PDE does this stencil correspond to." None of
them are a substitute for actually verifying the same operator would be
recovered at a different `h`, which is the stronger, resolution-independent
continuum-limit claim brief §14.2-14.3 makes — that still needs the
multi-resolution training this document's scope (§3.3) excludes.

Once a continuum form is in hand, compare its large-scale sector against
Interpretation C's KPZ/Burgers prediction (brief §10) — the natural
write-up step once B.4 has produced explicit coefficients, not a new
experiment.

---

## 6. Consolidated pre-registered decision rules

Stated before any result, per `CLAUDE_CODE_BRIEF.md` §0 rule 6 and §20
item 3. A miss gets written up, not tuned past.

| # | Test | Rule |
|---|---|---|
| A.1 | index power spectrum | ≥10× decay low→high index-wavenumber **and** `‖δ¹z‖/‖z‖ < 0.5`, outside the random-permutation null band |
| A.2 | latent light cone | `v_*^latent < 3` sites/step ⇒ local; `≳ 10` ⇒ not local at this `Δt` |
| A.3 | closure width | `w` at which `V(w,τ)` saturates, compared against A.2 and B.2 |
| A.4 | receptive field (R7/R8 only) | measured, fed to `minimum_localization_radius`, recorded — not a gate on A.1-A.3 or Block B |
| B.2 | accuracy | `val_kmax_mse` within 2× of `section52`'s 0.0256 (≤ 0.051) at the saturating width |
| B.2 | controls | permuted-index and `w_spatial=0` arms must be **worse**; if not, the structure is illusory |
| B.2 | homogeneity (R4) | `masked_mlp` ≫ `cnn`/`node` at the same width ⇒ evidence against a homogeneous law |
| B.3 | chaos preserved | `D_KY ∈ [21, 24]`, `n_positive = 11 ± 2`, `λ1 ≤ 0.1`, decoded energy spectrum tracking truth |
| B.4 | continuum operator recovered | modified-equation expansion (or moment-constrained refit) converges to a stable coefficient set across bootstrap folds; large/unstable leading-correction terms ⇒ report as such, not smoothed over |

---

## 7. What to write down regardless of outcome

Populate a "Latent PDE" section of `docs/RESULTS.md` (not this file) with
each step's measured numbers against the rules in §6, as they complete.
Both possible outcomes are complete, useful answers:

- **A pass through Block B** ⇒ a local PDE fit in the current latent
  ordering, with rollout accuracy, chaos-preservation, and (via B.4) an
  explicit continuum differential operator — the deliverable this document
  was asked for.
- **B.4 finds the leading correction term is large or unstable across
  folds** ⇒ still report the leading-order operator, but flag explicitly
  that the fit is "a local law at `h = L/44`," not evidence the same
  coefficients would hold at another resolution — a real, bounded claim,
  not a failure.
- **A.1/A.2 fail, or B.2's controls fail** ⇒ "the native, `spatial_signed`-
  shaped index carries same-time correlation (D8) but not resolved,
  bounded-range dynamics (D3/A.2), or the apparent locality doesn't survive
  the permutation-null control" — a real, complete negative result, and
  the documented next step (§3.3's coordinate search, or R7/R8's
  architectural program, §3.2) only if the user chooses to revisit either.

Finally: `docs/RESULTS.md` has not been updated since Gate 4, while
§§34-53 of `docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md` and the
`section4x/5x` runs contain the project's current best results (including
`section53`, which has no analysis-suite run at all). Folding those in —
with the `section52`/`section53` weather-vs-climate inversion (§2.0)
stated explicitly — is a prerequisite for this plan's Gate A report to sit
in a document that reflects reality.
