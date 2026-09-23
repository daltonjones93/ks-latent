# ks-latent Research Summary

**Scope**: the complete project history — ~114 numbered experimental sections
plus several unnumbered phases, spanning `docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md`,
`docs/HANDOFF_2026-09-04_FOURIER_HYBRID.md`, `docs/model-research-summary-9-5-26.md`,
`docs/sine_transform_pde_plan.md`, `docs/LATENT_PDE_RESEARCH_NOTES.md`,
`docs/LATENT_PDE_EXPERIMENTS.md`, `docs/RESULTS.md`, `docs/PROJECT_HANDOFF.md`,
`docs/ML_for_KS_writeup.md`, `docs/OPEN_QUESTIONS.md`, `docs/REPLICATION_LOG.md`,
and every `scripts/sectionN_*.sh` run's own logs and diagnostics reports. Every
number below is sourced from a logged Gate 3/4 run or one of these documents;
nothing is estimated or guessed.

**A note on completeness**: this document intentionally reports failures and
inconclusive results at the same level of detail as successes. The PDE-modeling
effort in particular (§6) is mostly negative — that is itself the headline
finding, not a gap in the write-up.

---

## 1. Executive summary

Four things were attempted across this project. Two succeeded clearly, one
succeeded with important caveats, and one — modeling the latent dynamics as an
explicit local PDE — did not work as originally attempted (Sections 99-127),
for a reason that is well understood — **but was later resolved by a change
in approach (Sections 128-140, §6.13): a genuinely local, interpretable
closure is achievable, just not as a from-scratch jointly-trained PRIMARY
propagator. See point 4's own update note and §6.13 for the full account
before treating the pessimism below as final.**

1. **Chaotic latent-space dynamics can be reliably reproduced.** A single
   mechanism (**H-PROP**, §3) explains essentially every success and failure
   in the whole propagator search: a latent propagator recovers the KS
   system's true chaotic attractor **if and only if** it contains at least one
   operation with (a) full-latent-dimension reach and (b) mixing weights that
   are not constrained to a normalized/softmax simplex. Every architecture
   satisfying both recovered chaos (`D_KY` within the correct benchmark band);
   every architecture missing either property collapsed to a fixed point
   (`D_KY=0`), including global *softmax-attention* architectures — global
   reach alone is not sufficient.

2. **Data assimilation works well on top of these models.** The best latent
   propagators give DA skill ratios (`rmse_free/rmse_da`) of 3-4x with good
   calibration, cheaply, because the latent dimension is tiny relative to the
   physical field (§5).

3. **Regularization can impose real, useful structure on the latent space**
   without sacrificing accuracy — banded same-instant correlation (D7/D8),
   approximate translation equivariance (D4), and reliable anti-collapse
   conditioning (`w_logdet`) are all achievable simultaneously on a globally-
   connected encoder, and *hard* architectural locality (masked/banded
   encoders) is never necessary and costs 8-33x in accuracy (**H-ENC**, §4).
   Latent geometric quality (conditioning number, D-diagnostics) does **not**
   predict rollout accuracy or DA skill in either direction — they are
   independent axes.

4. **Modeling the latent dynamics as an explicit local PDE has not
   succeeded**, across five structurally different attempts (§6): a literal
   pointwise-PDE propagator architecture (spectral derivatives + shared
   response function), joint distillation of a separate PDE-form "student"
   alongside a free-running "teacher" propagator, a native-latent-index
   locality search on an already-trained good propagator, and eight variants
   of an interpretable polynomial-coefficient PDE (degree, dimensionality,
   normalization strength, an architectural eigenvalue bound, integrator
   choice). The most successful-looking single run among these (Section 104,
   `D_KY≈22`) turned out to be a **spurious, non-physical result** once
   cross-checked (see the benchmark correction below); every other one of
   these attempts either collapsed outright (`D_KY=0`) or diverged to NaN.
   **The working hypothesis, arrived at independently twice** — once
   theoretically in this project's very first PDE planning document
   (2026-08-28, before any spectral_pde experiment ran) and once empirically
   via the unrelated H-PROP propagator search — **is that these two findings
   are in direct tension**: H-PROP says a propagator needs *global,
   unconstrained* mixing to sustain chaos, but a PDE's nonlinear term is by
   definition a *local, pointwise* function of the field and its spatial
   derivatives — and every one of the attempts above shares a structural
   feature: the learned function must *discover* chaotic dynamics from
   scratch, for which a fixed point is always the cheapest available answer.
   **The one variant that changes this incentive rather than fighting its
   symptoms** — a physics-informed prior baking in the *exact* analytic KS
   equation, so the network's "lazy" solution is correct physics rather than
   collapse — was tested properly for the first time this session (a prior
   attempt had used the wrong numerical integrator and produced immediate
   NaN for unrelated reasons, see §6.8). A completely untrained instance of
   this design showed genuine, persistent spatiotemporal structure, unlike
   every other attempt in this document — but **real joint training
   destroyed that structure within 20-40 epochs every time, across six
   mechanistically distinct anti-collapse interventions** (§6.9). The
   isolation diagnostic (Section 124) traced this to the *encoder*, not the
   dynamics: even with the propagator pinned to be exactly the true
   equation (nothing learned in the dynamics at all), training still
   collapsed it, because pointwise prediction loss on a chaotic target
   structurally rewards an encoder that stops distinguishing between real
   states. This unifies with H-PROP itself: a local/fixed propagator gives
   the (encoder, propagator) pair no escape route from that incentive
   *except* collapsing the encoder, while H-PROP's own globally-mixing
   winners apparently have enough alternative freedom that they never need
   to. This project's conclusion at this point is that a local/PDE-form
   propagator trained jointly with an encoder via pointwise prediction loss
   has run into a structural wall, not a tuning problem.

   **Update (Sections 128-140, §6.13): the wall is specific to where the
   locality lives, not to local/interpretable closures existing at all.**
   Three more regularizer generations on a local *propagator* (Sections
   128-134) confirmed the wall is real and not a tuning gap. Moving the
   locality into the *encoder* instead (`local_field`, Phase 10's original
   design, built for the first time in Section 135) and pairing it with a
   fully free propagator (Section 136) produced the first genuine,
   benchmark-matching chaos recovery with a *non-tautological* significant
   local-dynamics signal (D3) this project has measured. From there, an
   interpretable 22-parameter polynomial closure was obtained two ways —
   distilling a frozen, already-chaotic propagator after the fact (Section
   138: R²=0.92 at an 8-step rollout horizon), and, better still, co-training
   the encoder, propagator, and closure together with mutual gradient flow
   from epoch 0 (Section 140: R²=0.96, and simultaneously this project's
   best accuracy, `lambda1`, and DA skill/calibration numbers of the whole
   `local_field` line). Both routes independently rediscover true KS's own
   linear dissipation signs. The two-sentence version: a local closure of
   this system's dynamics is achievable, but it has to be given genuine
   physical locality by the *encoder*, and trained either as a distillation
   target or jointly with full co-adaptation — never as a from-scratch,
   jointly-trained-with-pointwise-loss PRIMARY propagator, which keeps
   failing for the H-PROP reason above regardless of encoder.

**Important benchmark correction, made while writing this document**: the KS
attractor's true Kaplan-Yorke dimension `D_KY` is **domain-length-dependent**
and this project's own memory had, until now, conflated the two values in
use: `D_KY≈22.5` is the true benchmark for `L=100` (the original/canonical
domain, used throughout Sections 1-98); `D_KY≈5.2-5.6` is the true benchmark
for `L=22` (this repo's own "Gate 1" domain, used throughout the entire
PDE-modeling arc, Sections ~99-114). Both are reproducible from the actual
ground-truth solver (`docs/REPLICATION_LOG.md`) and corroborated by two
independent project documents. This matters concretely: Section 104's
headline `D_KY=22.0007` result was reported at the time as a success, but at
`L=22` the true target is ~5.2-5.6 — `D_KY≈22` is **~4x too chaotic**, not a
match. Cross-checking against topological dimension estimates on the same
run's real encoded data (`two_nn_dimension≈4.5`, `correlation_dimension≈3.1`,
both near the true ~5.4) showed the *encoder* was faithful; the *propagator's
free rollout* was exploring a non-physical, over-excited regime. Every
`D_KY=0` collapse result in this document is unaffected by this correction
(zero is zero regardless of benchmark).

---

## 2. Background and methods

**Original goal**: build a fast surrogate — an autoencoder (AE) that
compresses a KS PDE state into a low-dimensional latent `z` (`d ≪ N_x`), plus
a learned propagator that advances `z` forward in time — cheap enough to run
nonlinear data assimilation (an Ensemble/Particle Flow Filter) directly in
latent space. This DA use case is the project's primary stated end goal.

**Reference implementation** (`docs/ML_for_KS_writeup.md`): an earlier
codebase (`L=100`, `N_x=1024/128`, `d=24-44`, Set-Transformer/Perceiver-style
AE + residual-MLP propagator) established the training conventions this
project inherited throughout: delta/residual parameterization with a
zero-init output head (exact identity at init, so an early rollout can't
explode), a two-state `(z_{n-1}, z_n)` input as a velocity proxy, a rollout-
horizon curriculum (`k` ramped 2→16 over training), best-val-`k`
checkpointing, and — critically — a **physical-space pushforward loss**
(`w_pred`: decode the propagator's prediction and compare to the true next
*state*, not to a moving latent target). This last point fixes a real,
documented failure mode: an AE trained on reconstruction alone produces a
latent where the MSE-optimal next-step prediction degenerates toward
persistence (the conditional distribution of `z_{n+1}` given the past is
broad and centered near `z_n` on a chaotic attractor), so joint AE+propagator
training via `w_pred` has been load-bearing throughout the whole project.

**A second, later objective**: the PDE-modeling effort (§6) originates from a
separate, later request — `docs/LATENT_PDE_RESEARCH_NOTES.md` (2026-08-28)
documents this as "Peter Jan's request for 'a PDE that models the latent
variables,'" layered on top of the original DA-surrogate goal rather than
being it. That planning document disambiguates three distinct things "a PDE
for the latent variables" could mean: (A) a literal governing
ODE/inertial-manifold-form system; (B) an *emergent-space* local PDE (per
Kemeth et al. 2022, arXiv:2012.12738 — discovering new, PDE-admitting
coordinates rather than assuming the existing latent has them); (C) a
coarse-grained effective PDE in the KPZ/stochastic-Burgers family, a genuine
known theoretical target. Its own read was that the request was closest to
(C), approached as if it were (B).

**KS solver / physical setup**: `ks_latent/solver/ks.py` uses ETDRK4
(exponential time differencing, RK4) — Fourier space, diagonal linear
operators, pseudo-spectral nonlinear term. Two domain lengths are used
throughout, with **different true chaos benchmarks** (see the correction
above): `L=100` (`D_KY≈22.5`) for most of the architecture search (Sections
1-98), `L=22` (`D_KY≈5.2-5.6`) for the PDE-modeling arc (Sections ~99-114+).

**Diagnostic tooling**, reused throughout: Gate 3 (`run_analysis_suite.py`,
`run_da_pff.py`) computes the Lyapunov spectrum (two independent methods,
`single_state`/`two_step`) and `D_KY`, plus DA skill (`rmse_free/rmse_da`,
calibration ratio); Gate 4 (`run_diagnostics.py`) runs a battery of
structural tests: D3 (one-step propagator-Jacobian bandedness), D4
(translation-equivariance search), D6/D7/D8 (same-instant spatial coherence
at various representations), D9 (real-trajectory step-size/curvature/
Jacobian-norm smoothness). `visualize_rollout.py`'s Hovmöller plots
(physical + latent space) are the ground-truth check for collapse — this
project's own established practice is to **never trust `val_kmax_mse` alone**
as evidence of health; a suspiciously tiny value is confirmed or refuted via
Hovmöller visualization and Gate 3's Lyapunov spectrum every time.

---

## 3. Chaos recovery: the propagator architecture search (H-PROP)

### 3.1 The central result

Sections 5, 9, and 10 together ran a complete 2×2 grid crossing **receptive
field width** (local vs. global) with **normalization** (softmax attention vs.
unconstrained linear/spectral mixing):

| | Softmax-normalized (attention) | Unconstrained (linear/spectral) |
|---|---|---|
| **Local** | `vit`, `attn_window=4` → **collapsed** (`D_KY=0`) | `local_mlp` → **collapsed** (`D_KY=0`) |
| **Global** | `vit`, full attention → **collapsed** (`D_KY=0`) | `mlp` → **chaotic** (`D_KY=20.9-22.1`); `fno_vit` → **chaotic** (`D_KY=20.9`) |

Two earlier candidate explanations were proposed and then explicitly
falsified by the next experiment: "softmax = convex combination = structurally
contractive" (falsified — plain `mlp`, no softmax anywhere, recovers chaos
even more cleanly than `fno_vit`) and "unconstrained-to-amplify vs.
softmax-constrained" (falsified — `mlp` needs no measured amplification
justification the way FNO's singular values do). The surviving, repeatedly
confirmed pattern:

> **H-PROP**: a latent propagator recovers the true chaotic KS attractor if
> and only if its architecture contains at least one operation with (a) full
> `d_latent` receptive field, and (b) mixing weights not constrained to a
> bounded/normalized simplex (i.e. not softmax attention). Neither property
> alone is sufficient.

Confirmed directly via `local_mlp`'s receptive-field-boundedness test
(Section 10) and reconfirmed via `masked_mlp_warm_start`'s dense endpoint
(Section 11, `attn_window=None` recovering `D_KY=20.51` after a purely-local
pretrain phase). Every propagator satisfying (a)+(b) in this project's entire
history recovered chaos (`mlp`, `fno_vit`, `masked_mlp` at `attn_window=None`,
every `fourier_mlp`-family propagator); every propagator failing (a) or (b)
collapsed (`vit` at any window including full/global, `local_mlp`,
`masked_mlp` at any finite window). A second, narrower mechanism (Section 43)
partially explains *why* softmax specifically biases toward contraction
(convex-combination amplification ceiling), but is a contributing factor, not
the deciding variable.

Once an architecture clears the H-PROP bar, essentially every successful run
lands in a fairly tight `D_KY≈20.9-22.3` band (at `L=100`) — differentiation
between winners is accuracy, DA skill, conditioning, and smoothness, not "how
chaotic." Chaos richness, accuracy, DA skill, and D4 equivariance are four
largely independent axes.

### 3.2 `delta_cap` conclusively ruled out as the active ingredient

Every early winning MLP-family result used `PropagatorConfig.delta_cap`
(`delta = delta_cap * tanh(raw_delta/delta_cap)`, an architectural residual
bound, motivated as a "stretch-and-fold" mechanism). A direct ablation
(Sections 15-16) found the opposite of the working hypothesis: **removing
`delta_cap` from both Phase 1 and Phase 2 gave the single best result of that
whole investigation** (`val_kmax_mse=0.006546`, ~2x better than any
`delta_cap` variant; `D_KY=21.99`, `n_positive=12`; best DA skill/calibration
ratios measured). Conclusion, stated directly in the project's own record:
"`delta_cap` is not required anywhere in the pipeline to recover chaotic,
accurate, DA-skillful latent dynamics... it was never the active ingredient
behind the collapse/no-collapse distinction." H-PROP's global+unconstrained
mixing requirement is the real explanation. (This ablation was independently
re-confirmed relevant this month: an early guess during the PDE-modeling arc
that `delta_cap` might rescue a collapsing polynomial-PDE propagator, §6.6,
was withdrawn on rediscovering this exact prior result before wasting a
training run on it.)

### 3.3 A trained propagator does not transfer across encoders

Section 31: warm-starting an already-converged propagator onto a different
(same-family) AE never recovered — stayed ~10x worse than that exact
propagator's own converged value, even well past the point the original
needed to converge. A propagator is tightly fit to its own AE's specific
latent coordinate geometry; any encoder architecture search needs to re-fit
the propagator from scratch each time, not transplant a converged one.

---

## 4. Encoder/decoder architecture search (H-ENC)

| Family | Best instance (`val_recon`/key metric) | Notes |
|---|---|---|
| Original patched transformer (`NX=1024`) | 0.969 (≈mean-baseline) | Collapsed outright — the origin problem motivating the whole project; fixed by shrinking `NX` to 256 and `mode="markovian"`. |
| `vit`, globally mean-pooled | Section 52: `val_recon=0.000292` | Best raw rollout MSE (0.0256) and DA skill (3.12) up to that point. Catastrophic conditioning (`cond#=1.6e6`) but also the smoothest real-trajectory embedding measured. |
| `masked_mlp` (hard architectural locality, encoder+decoder) | Section 17 | Highest `D_KY`/`n_positive` of its era (22.26/13) but ~10x worse reconstruction than any non-toy dense AE. |
| Banded pooling (soft-local token→latent) | Sections 24-25 | Only architecture besides masked_mlp to get D3/D6/D7 all significant simultaneously — but 8-33x worse rollout accuracy than the dense-encoder best. |
| `fourier_mlp` (masked raw-value path + dense/ifft Fourier path, summed) | Section 75 | `cond#=20.98` (well-conditioned), `D_KY=21.09`, `val_kmax_mse=0.039` — first family combining good conditioning with real chaos fidelity. |
| `vit_fourier_hybrid` | Section 85 | `val_kmax_mse=0.0279`, `D_KY=21.67`, `cond#=24.0`, genuine D4 translation equivariance, smoothest embedding of its family. |

> **H-ENC**: soft regularization on a globally-connected encoder
> (`w_spatial`+`w_logdet`, later `+w_smooth`) beats hard architectural
> locality, on *both* accuracy and measured structure. Sections 17/24-25 pay
> a real, consistent 8-33x accuracy tax for locality without reliably beating
> soft-regularized models (75/81/85) on structural significance.

`fourier_ifft_readout` (an MLP predicts frequency-domain coefficients, then a
mathematically exact `irfft` maps to state space, instead of an arbitrary
linear readout) was a small, consistently non-negative accuracy upgrade
across the whole `fourier_mlp` lineage — kept as a default there.

**Sections 89-98** (continued tuning of the `vit_fourier_hybrid` family, all
2026-09-05): a run of direct user-directed architecture/regularizer
iterations — Fourier-branch truncation and pooling changes (89-90),
ViT/Fourier parameter-split rebalancing (91, beaten by 92's `token_mlp`
pooling variant: `val_kmax_mse=0.0135` vs `0.0162`), model-size scale-downs
(93), a `w_spatial`/`w_smooth` sweep that twice regressed accuracy without
improving smoothness (95, 96 — confirming `w_spatial` alone, not `w_smooth`,
drove the cost), and a final size reduction (97). **Section 98** then removed
the Fourier branch entirely — a plain `vit` encoder/decoder (`d_model=56`,
`token_mlp` pooling) + Section 52's `mlp`/markovian propagator, at the user's
own direction ("remove the fourier mlp... don't change anything else") to
isolate how much it was actually contributing. Result: `val_kmax_mse=0.0332`,
`D_KY=22.12`, `n_positive=13`, `λ1=0.092` — solidly chaotic **at L=100**, and
the baseline the PDE-modeling arc subsequently launched from.
**Caution reflected in this document but not always observed in this
project's own section-script comments**: because Section 98 is an `L=100`
run, its `D_KY≈22` is **not** a valid direct chaos-fidelity comparison point
for the later `L=22` PDE-arc runs (§6) — the correct `L=22` target is
`D_KY≈5.2-5.6` (see the benchmark correction in §1).

---

## 5. Regularizers and data assimilation

### 5.1 Regularizers: what fights what

| Regularizer | Targets | Verdict |
|---|---|---|
| `w_var` / `decorr_var_loss` | pulls every channel toward variance 1 | Fights healthy strong channels too; not sufficient alone against organic collapse. |
| `w_var_floor` (VICReg-style per-channel std floor) | marginal per-channel variance | **Provably inert** against correlation-driven collapse (Section 38: exactly zero loss at convergence while the covariance's smallest eigenvalue was still catastrophic) — structurally blind to joint-eigenvalue collapse. |
| `w_logdet` (log-det barrier, `-logdet(Cov(z)+εI)/d`) | whole covariance eigenspectrum | **The actual fix** for this project's real, organically-occurring collapse mode. Pushed `cond#` from 1e5-1e6 to single digits (Section 38). |
| `w_spatial` (signed spatial coherence) | banded same-instant correlation between nearby latent-index channels | Genuinely opposed to `w_logdet` (isotropy vs. banded pull the same eigenspectrum in opposite directions) — navigable at the right ratio, but too far in isolation can degrade the Lyapunov spectrum itself (Section 47: `D_KY` crashed to 7.97 at `w_spatial_signed=0.08`). |
| `lambda_z` (banded `RegConfig`, value-proximity) | pulls index-nearby coordinates toward *equal values* | **Never recommended** — degenerate global optimum is literal collapse to a constant. (The `--lambda-z` CLI flag genuinely is this mechanism; an earlier handoff doc wrongly claimed otherwise.) |
| `w_smooth` (temporal smoothness) | real-trajectory step size + curvature | Architecture-dependent: helped every axis on `mode="history"` propagators, hurt every axis on `mode="markovian"` propagators (**H-SMOOTH**: a markovian map loses needed velocity information when the encoder is smoothed; a history-mode map can recover it from its multi-state input). |
| `w_shape_floor` (one-sided, real-data-calibrated per-mode energy floor) | `spectral_field` encoder's own `z`, penalizes falling *below* the true field's empirical per-mode energy share | Introduced for the PDE arc specifically (§6.3) — the first regularizer in the whole project calibrated from real physical data rather than a fixed hyperparameter target. |
| `w_lowpass` / `w_lowpass_rollout` | penalizes high-wavenumber energy in `z` / the propagator's own rolled-out `z_pred`, `∝k^power` | Introduced after Section 104 showed the propagator's free rollout driving high-index channels to unphysical amplitudes that nothing previously constrained. |
| `w_logdet_physical` (this week, §6.7) | full-covariance log-det barrier applied to `decode_from_spectrum(z)` (physical space) instead of `z` directly | Mathematically shown to be a **mild reweighting** of `w_logdet`, not a distinct mechanism, when the transform is a genuine (non-square, truncated) linear embedding — see §6.7 for the full derivation. Independently, `docs/sine_transform_pde_plan.md` §15 (2026-09-07) had already tried and explicitly rejected the naive version of this idea for the same reason, before this week's re-derivation. |

### 5.2 Data assimilation

DA skill (`skill_free_over_da = rmse_free/rmse_da`) tracks the best
propagators closely: 3.1-4.2x skill ratios with calibration ratios
(`spread/rmse`) in a healthy 0.3-0.5 range across the best `L=100` models
(Sections 16, 52, 75, 81, 85). This confirms the core DA-surrogate premise
(§2): once a propagator clears H-PROP's chaos-recovery bar, it is cheap and
effective to assimilate in latent space. DA metrics were **not** systematically
re-run for the `L=22` PDE-arc's collapsed runs beyond confirming the same
collapse signature Gate 3's Lyapunov spectrum already shows.

### 5.3 Best models, by criterion (at L=100, the architecture-search domain)

- **Dynamics fidelity**: Section 16 (`nodeltacap_e2e`) — `val_kmax_mse=0.006546`
  (best raw accuracy of the whole architecture search), `D_KY=21.99`
  (`|D_KY-22.5|` closest of any checkpoint), `n_positive=12`, `λ1=+0.0885`,
  `skill_free_over_da=3.43`. Caveat: predates the `w_spatial`/`w_logdet`
  recipe and the D4/smoothness tooling entirely.
- **PDE-modeling readiness / stability**: Section 85 — `val_kmax_mse=0.0279`,
  `D_KY=21.67`, `cond#=24.0`, genuine D4 translation equivariance, lowest
  propagator-Jacobian norm of its compared set (median 1.60, p95 1.74) — the
  single model combining smoothness, equivariance, and good accuracy
  simultaneously, the profile wanted before attempting to fit a governing
  equation. (In practice, the PDE arc launched from Section 98's simpler
  architecture instead, not Section 85 — see §6.3.)

---

## 6. The PDE-modeling effort

This is the least successful and most extensively probed part of the project.
Five structurally different approaches were tried; all either collapsed,
produced a result later found to be spurious, or required a non-local
component doing the real dynamical work. This section presents them
chronologically.

### 6.1 Theoretical prediction, before any experiment ran

`docs/LATENT_PDE_RESEARCH_NOTES.md` (2026-08-28) — the project's very first
PDE planning document — identified the core obstruction from first
principles, before any `spectral_pde` code existed: the existing
(architecture-search-winning) latent is produced by a **globally-supported**
encoder (every `z_k` depends on every physical point `x`) — "locality was
architecturally destroyed on purpose," in that document's own words, by the
very encoders H-ENC (§4) shows work best. Its recommended program — a
literal local latent *field* (circular-Conv1d encoder/decoder, patch-local
and translation-equivariant by construction, plus a shared local stencil
propagator and an explicit h-refinement/continuum-limit falsifiability test)
— was **never built**. It remains a well-specified, untried alternative (see
§7).

A separate, useful methodological point from this document, verified
directly against this project's own KS operator: a globally-supported
encoder does **not** by itself preclude locally-acting dynamics — the DFT of
a tridiagonal (banded) physical-space heat-equation update is exactly
diagonal despite every DFT coefficient depending on every physical point.
This is what legitimizes testing the existing, globally-pooled latent
directly for local structure rather than requiring an architecturally-local
encoder as a prerequisite.

### 6.2 Direct empirical locality tests on an already-good propagator

Two independent, already-completed tests probed whether local dynamics exist
in or can be imposed on an existing well-trained latent (Section 98's own
propagator, §4):

- **Post-hoc SINDy** (`scripts/fit_latent_pde.py`, sparse local regression
  directly on Section 98's frozen real latent trajectories): `R²≈0.005`,
  `D_KY=0` — a clean, hard failure. The representation was never pressured
  toward local-fittability, so this isn't surprising in hindsight, but it
  rules out "a local PDE was hiding in the existing latent all along."
- **Section 100** (`masked_mlp_wide`, an architecturally-local propagator
  sized to match Section 98's dense propagator's *active* parameter count
  exactly, `attn_window=4`, paired with a matching local-windowed
  ViT encoder/decoder): `val_kmax_mse=0.280` (~8.5x worse than Section 98's
  0.033), `cond#=7.9e15` (essentially singular). No Gate 3 Lyapunov run was
  logged (Stage 2 itself performed too badly to be worth it) — but this
  result is exactly what H-PROP predicts, and the script's own pre-registered
  expected-result note said as much before it ran.

`docs/LATENT_PDE_EXPERIMENTS.md` proposed a more systematic version of this
test (a `{masked_mlp, cnn, node}` × window-radius sweep on Section 52's
frozen AE, with permutation/no-`w_spatial`/coarse-grained control arms,
grounded in an 8-requirement taxonomy — R1-R8 — for "does a latent PDE
exist") — **this specific plan was never executed** (its own status line:
"proposal, not yet run"). Its one already-measured (not proposed) finding
worth keeping: across the Sections 44-52 sweep, **D8 (same-instant signed
bandedness) is strong and tunable** (0.65, `p=0.0000`, monotone across the
sweep) while **D3 (one-step propagator-Jacobian bandedness) is never
significant** (flat 0.26-0.30, `p=0.18`, across all 9 runs) — same-instant
correlation exists in the native latent index, but the *dynamics* have never
shown measured locality in any run tried.

### 6.3 The spectral-derivative PDE propagator (Sections 99-106)

**Architecture**: `encoder_kind="spectral_field"` — a global ViT
encoder/decoder producing a genuine spatially-indexed field `w`, composed
with a fixed, non-learned rFFT truncated to `K` modes (`z = rfft(w)[:K]`).
Paired with `backbone="spectral_pde"`: synthesizes analytically **exact**
spatial derivatives `w, w_x, w_xx, ..., w^(n)` from the same truncated
spectrum via `(ik)^n` diagonal multipliers (no finite differences, no
learned filters), feeds the derivative stack through a small **shared
pointwise** function to predict `w_t` — literally the functional shape of a
real local PDE, in the same basis the true KS solver itself uses (ETDRK4:
diagonal linear operators in Fourier space, pseudo-spectral nonlinearity).
Closest prior art: PDE-Net/PDE-Net 2.0 (Long et al.), with exact spectral
derivatives in place of PDE-Net's approximate finite-difference-like
convolution kernels.

**The H-PROP tension was pre-registered, not discovered later.** The founding
proposal for this architecture (2026-09-06) states up front, before any
experiment ran: "the resulting propagator... is still a spatially local,
weight-shared, pointwise map in the same architectural family as
`node`/`local_mlp`/`masked_mlp` — every one of which has collapsed to a fixed
point in this project's own prior experiments... This proposal is best
understood as a sharper, more principled test of *why* that keeps happening,
not a guaranteed escape from it." The open question was specifically whether
giving a local architecture numerically *exact* derivative features
(matching the true solver's own basis) would let it succeed where naive
local architectures (arbitrary conv/masked-linear, with no reason to
discover real derivative structure) had failed.

**Section 101** (first real run, frozen-encoder design, `L=22`, `K=8`,
`N_w=32`, `dt_snap=0.2`): all three integrators (rk4/etdrk4/euler)
**collapsed** (`D_KY=0`, `n_positive=0`), via different mechanisms
(`λ1=-0.649` rk4, `≈-2e-7` etdrk4 — marginal, `-0.455` euler).

**Section 102**: fixed a real design flaw — the encoder in Section 101 never
saw the actual `spectral_pde` propagator during training (only a throwaway
generic `mlp`), confounding the collapse result. Enabled real joint Phase-1
training of encoder + `spectral_pde` propagator together.

**Section 103** (`K` widened to 15 at the user's explicit objection to an
arbitrary `K=8` truncation, `dt_snap` restored to the canonical 1.0,
`ode_substeps=3`): still collapsed, but *meaningfully less so* than Section
101 — `λ1` improved from -0.455 to **-0.030** (~15x closer to neutral),
`skill_free_over_da` improved from 8.17 to 1.90, and the propagator's own
Jacobian norm moved from contractive (0.945) to slightly expansive (1.19). A
real integrator bug was found and fixed here too: `euler` had been hardcoded
to take exactly one step regardless of `ode_substeps` (rk4/etdrk4 already
sub-stepped correctly).

**`spectral_shape_floor_loss`** (introduced here, user: "I really just want a
regularizer to prevent the latent state from collapsing"): a **one-sided**
floor, calibrated from real training data's own empirical per-mode energy
share, penalizing `z` falling *below* that physically-required minimum,
never for exceeding it (so legitimate compensation for discarded high-mode
structure isn't fought). Notably, applying `w_logdet` to the *physical*
reconstruction `decode_from_spectrum(z)` instead of `z` directly was tried
and explicitly rejected at this point (2026-09-07) — verified mathematically
equivalent to applying it on `z`, plus a harmful rank-deficiency artifact
from the truncation (`Cov(w)`'s smallest eigenvalues matched
`N_w - 2K` exactly). This is the same conclusion this project's most recent
session (§6.7) independently re-derived a week later before building
`w_logdet_physical` — worth noting as a case where a rejected idea was
revisited without initially checking prior art, then correctly re-derived.

**Section 104** (`K=24`, `N_w=64`, `w_shape_floor=0.1`, deliberately
aggressive): the first non-collapsed `spectral_pde` result —
`D_KY=22.0007`, `n_positive=11`, `λ1=+0.079`, `D9` Jacobian genuinely
expansive (1.74), verified not a numerical artifact via a bounded 300-step
rollout check. **This was reported as a milestone at the time and is not
one** — per the benchmark correction (§1), `D_KY≈22` at `L=22` is ~4x too
high relative to the true `~5.2-5.6` target. Cross-checking against
topological dimension estimates on the same run's real encoded data
(`two_nn_dimension≈4.5`, `correlation_dimension≈3.1`, both near the true
value) showed the encoder itself was faithful, but the free-running
propagator was exploring a non-physical, over-excited regime — confirmed via
Hovmöller visualization showing persistent high-wavenumber streaks the true
`L=22` attractor never has (the true `L=22` field is smooth by construction:
only wavenumbers 1-3 fall inside the instability band, vs. ~16 at `L=100`).
`w_lowpass_rollout` was introduced specifically in response to this finding,
to constrain the propagator's own rolled-out high-mode energy — nothing
previously did.

**Section 106**: exact repeat of 104's recipe, retrained from scratch, with
`w_lowpass_rollout` active from Phase 1 (0.015→1.0), explicit goal of
bringing `D_KY` down toward the true ~5-6. **Result: inconclusive** — the run
was interrupted (`Interrupted system call`) partway through Phase 2, before
reaching Gate 3. Its Phase-1 covariance spectrum already showed concerning
signs before the interruption: participation ratio 6.18 (out of 48 latent
dimensions), `cond#≈3e16`, and `val_kmax_mse` frozen flat at ~0.0094 across
visible epochs despite the rollout horizon `k` growing 2→6 — the same
flat/frozen signature later collapses (§6.6) also show. No clean verdict is
available; this run should be re-launched to completion if the `spectral_pde`/
MLP-field line of investigation is revisited.

**Section 105** (correction, found later while investigating §6.6's pivot):
not actually undocumented — it exists as raw logs/checkpoints with no saved
script (invisible to a `scripts/sectionN_*.sh` search). It attempted
`--spectral-physics-prior` with `K=24`, `euler` integrator, `w_shape_floor=0.3`
— and produced `loss=nan` from **epoch 0**. Root-caused directly (see §6.8):
this is a textbook numerical-stiffness violation, not a training/collapse
issue. `physics_prior` bakes in the *exact* `-w_xxxx` term, which is
famously stiff; explicit Euler's stability limit at `K=24`, `L=22` is
`dt < 2/k_max^4 ≈ 0.0009`, but the actual substep used (`ode_substeps=3`,
`dt_snap=1.0`) is `0.333` — roughly 370x too large. `etdrk4` exists
specifically to handle this stiffness via an exact exponential/contour-
integral treatment (the same formula the real KS solver itself uses); Section
105 simply used the wrong integrator for this feature.

**`spectral_physics_prior`** (built, §6.4): bakes in the exact analytic KS
right-hand side as a fixed baseline (the network learns only a *correction*),
motivated by Lyapunov-exponent invariance under smooth coordinate change — if
`w` faithfully reparametrizes the true state, baking in exact KS dynamics
should inherit genuine chaos "essentially for free." Verified exactly at
zero-init (0.0 max absolute difference from the analytic RHS). Section 105's
only real attempt used the wrong integrator (above) and never produced a
usable result — see §6.8 for the follow-up that finally tested this
properly.

### 6.4 `pde_head`: joint distillation alongside a free propagator (Sections 107-110)

After Sections 101-106's repeated collapse, the user's own framing shifted:
"this new framework seems incredibly hard to train" (direct quote,
2026-09-08). The redesign: keep training a **free, unconstrained** `aux`
propagator as primary (the kind H-PROP says reliably produces chaos), while a
**separate** `spectral_pde`-form propagator (`pde_head`) is trained
*alongside* it to match its dynamics — reasoned as structurally different
from the failed primary-propagator attempts, since `pde_head` is never
responsible for its own stable multi-step numerical integration under
gradient pressure, only for a single-step regression fit (Stage 1, detached
target). A Stage 2 extension made the loss mutual (gradient flows into both
the propagator and `pde_head`) and added an autoregressive-rollout variant
(`w_pde_rollout`) — explicitly flagged as "the riskier direct approach that
made `spectral_pde` incredibly hard to train as a primary propagator."

- **Section 107** (`L=40`, mutual, `w=0.8`) and **Section 108** (two Phase-3
  "freeze the propagator, refine `pde_head` alone" attempts): both killed or
  plateaued unhelpfully, superseded by better-motivated designs.
- **Sections 109/110** (from-scratch, **detached** target, `w=0.008/0.016`):
  the propagator itself stayed genuinely healthy in both
  (`D_KY≈21.7-22.2`, matching Section 98's own — L=100 — baseline), but
  direct measurement showed `pde_head` learned essentially nothing:
  `MSE(pde_head vs aux) / aux's own step-size MSE ≈ 0.98` in both runs —
  indistinguishable from a trivial persistence baseline. Doubling the weight
  (109→110) did not improve this at all, while measurably *degrading* the
  propagator's own accuracy 2.5-3x. **This result is what ended the
  joint-distillation line of investigation** — user's own verdict: "I don't
  think the joint training idea is going to work... this pde idea just might
  not work too, I'm running out of ideas."

`spectral_pde_raw` (a generalization letting `pde_head` compute its own
self-FFT of any encoder's raw `z`, treating the latent index as a spatial
coordinate on a periodic ring) was built to let `pde_head` sit on top of the
project's best pre-PDE architecture (Section 95's `vit_fourier_hybrid`). A
real instability was found while building it: `etdrk4`'s baked-in linear
operator (`k²-k⁴`, correct for genuine `spectral_field` z) is an unjustified,
empirically harmful assumption for an arbitrary self-FFT'd latent ordering —
a real run diverged (loss `0.57→268.0` over 5 epochs) with etdrk4 while
`euler` trained stably (`0.51→0.48`) on the identical setup; the CLI default
was changed to `euler` for this backbone accordingly.

### 6.5 Interpretable polynomial-coefficient PDE (Sections 111-113, this session)

Motivated by the same collapse pattern recurring: "instead of using a neural
network to parametrize the pde, expand it as a polynomial (degree 1 or 2) in
all the derivative terms, then directly learn the coefficients" — trading the
shared MLP's black-box correction for a single shared linear layer over a
monomial library built from the derivative stack (`[w, w_x, ..., w^(n)]`),
so the learned weights directly *are* the local PDE's coefficients. Two
features were added for physical plausibility and stability:

- **Combined-order truncation** (`max_term_order`): excludes any monomial
  whose derivative orders sum to `≥` a threshold (e.g. `w_xxx·w_xxx`,
  combined order 6) — KS's own true terms (`w·w_x` order 1, `w_xx` order 2,
  `w_xxxx` order 4) all survive comfortably under a threshold of 5.
- **Fixed-scale per-order normalization**: each derivative order divided by
  `char_k^n` (a fixed, non-learned constant) before the polynomial library is
  built. This fixed a real, precisely diagnosed NaN blowup: the unnormalized
  degree-2 library's raw output reached ~1.3e7 at nominal coefficient scale
  (a controlled scale test isolated this to ~875,000x before the fix, ~14.5x
  after).

**Section 111** (degree=1, purely linear): killed immediately on review — a
linear PDE structurally cannot represent KS's own nonlinear term `-w·w_x`,
and the linear part of KS is not chaotic on its own.

**Section 112** (degree=2, `max_term_order=5`, `w_shape_floor=0.1`):
**collapsed** — `val_kmax_mse=0.0015` (vs. Section 98's 0.033), confirmed via
Hovmöller (every latent channel snaps to a constant at t=0, sustained 200
steps) and Gate 3 (`D_KY=0.0`, `n_positive=0`, `λ1=-0.40`). Diagnosis: the
two stability fixes needed to stop the NaN blowup (normalization, term
truncation) both shrink the model's dynamic range/hypothesis space — plausibly
making "contract to a point" the cheapest solution again, the same H-PROP
mechanism recurring in a new architectural guise. Section 104's own original
(unnormalized, MLP field_kind) run at the identical `w_shape_floor=0.1` did
*not* collapse — only the polynomial, normalized, truncated form did.

**Section 113** (`w_shape_floor` escalated 0.1→0.3, plus `w_varmatch`/
`w_varmatch_adaptive` and `noise_step`, two anti-collapse loss terms built
earlier in the project specifically for this failure mode): **collapsed
worse** — `val_kmax_mse=0.0006` (lower than Section 112's own collapsed
value), crashing from 0.124 to 0.0006 within a single Phase-2 epoch, flat
thereafter despite the rollout horizon growing 5→12. Gate 3: `D_KY=0.0`,
`λ1=-0.317`. **This was avoidable**: `docs/train_stage2_patched.py`'s own
code comments already documented, from 2026-08-29 (before any `spectral_pde`
work), that `w_varmatch`+`noise_in` had already been tried and had already
failed to prevent this exact fixed-point collapse on a *different*
propagator backbone — these tools were picked for Section 113 without first
checking that history.

A `delta_cap` fix was proposed next based on the same file's own docstring
framing ("gives the model an architectural fold"), then **withdrawn before
launching** on rediscovering §3.2's own prior ablation (Sections 15-16),
which had already conclusively shown `delta_cap` is not load-bearing and
removing it produced the best result of that entire investigation — avoided
repeating a result the project already had.

### 6.6 `w_logdet_physical` and its rollout variants (Sections 114-116) — three attempts, three failure modes, no success

Proposed as a genuinely new lever (not yet tried in this arc): apply the
existing `w_logdet` mechanism's log-det anti-collapse barrier to
`decode_from_spectrum(z)` — the encoder's own physical-space field
reconstruction — instead of `z` (the rFFT coefficients) directly, in **Stage
1** (encoder training), leaving Stage 2 (the propagator's own rollout
training) bare, isolating this term's own effect.

**A worked mathematical check, done before implementing**: if `z→w` were a
square orthonormal map, `logdet(Cov(w)+εI)` would be *exactly* equal to
`logdet(Cov(z)+εI)` (since `det(AAᵀ)=1` for orthonormal `A`) — the two losses
would be identical, not just similar, and there'd be no point computing both.
Here the map is genuinely non-square (`z∈R^{48}`, `w∈R^{64}`, a real
truncation, `K=24 < N_w//2+1=33`), so two real (if modest) effects survive:
(1) `Cov(w)` is rank-deficient in the ambient 64-dim space — the extra
`ε·I` eigenvalues in unreachable directions are a fixed, `z`-independent
additive constant, not real gradient signal; (2) rFFT/irFFT's non-uniform
Parseval weighting (DC/Nyquist weight 1, other modes weight 2, `1/N_w`
scaling) makes the "live" part of the `w`-space penalty a **reweighted**
version of the `z`-space one, not a rotated copy. Net honest assessment: a
mild per-mode reweighting, not a fundamentally different mechanism — this
matches (and was cross-checked against, after the fact) `sine_transform_pde_plan.md`
§15's own independent rejection of the naive version of this same idea a
week earlier (§6.3), for the identical reason.

Weight was calibrated via smoke testing rather than guessed: an initial
`w_logdet_physical=0.1` measurably hurt Phase-1 reconstruction convergence
(`val_recon_final=0.168` after 3 epochs vs. ~0.024-0.037 for every prior
smoke test at the same budget); rescaled to `0.01` (matching this project's
own previously-calibrated `w_logdet` range, 0.0035-0.02) restored normal
convergence (`val_recon_final=0.024`) while a Phase-2 continuation smoke test
stayed in a healthy `val_kmax_mse` 0.05-0.09 range (no early collapse
signature, unlike Sections 112/113's own smoke tests).

**Result: collapsed.** The caveat flagged before launching turned out to be
exactly right — Section 114's Phase-2 `val_kmax_mse` froze at ~0.0053 from
epoch 26 through the final epoch 79, essentially flat despite the rollout
horizon `k` growing 6→12 (the same "not growing with horizon" signature
every other collapsed run in this arc has shown). Gate 3 confirmed it:
`D_KY=0.0`, `n_positive=0`, `λ1=-0.472`. A Stage-1-only, encoder-side
regularizer has no direct hold on the propagator's own free rollout during
Phase 2, where the encoder is already frozen — precisely the gap the
caveat predicted.

**Section 115**: the user's own diagnosis of the gap, stated directly —
"`w_logdet_physical` should apply to the encoder in stage 1, and the rollout
from the propagator in stage 1, and also to the rollout in stage 2" — led to
the same log-det-on-physical-space mechanism being applied in all three
places at once: (1) `w_logdet_physical` (Stage 1, encoder's real `z`,
unchanged from 114), (2) a new `w_logdet_physical_rollout`
(`Stage1TrainingConfig`, added directly mirroring the existing
`w_lowpass`/`w_lowpass_rollout` pairing — applies the same barrier to the
JOINT-trained aux propagator's own short Phase-1 rollout `z_pred`), and (3)
`w_logdet_rollout` (`Stage2TrainingConfig` — already built during this same
session but left unused in Section 114 — applies it to the real
propagator's extended Phase-2 rollout).

**A new diagnostic technique used from here on**: rather than waiting for
the full Phase 1 + Phase 2 + Gate 3 pipeline, `visualize_rollout.py` was run
directly on Phase-1's own mid-training checkpoints (`train_stage1_patched.py
--full-propagator` writes one every `--checkpoint-every` epochs) — isolating
whether collapse is already present after Phase 1 alone, before Stage 2 ever
touches the propagator. This directly tests whether Stage 2 is the cause of
collapse, as the user suspected. Applied retroactively to **Section 114**
first: its finished Phase-1 checkpoint (before Section 114's own Stage 2 had
even been designed) already showed a **complete, extreme collapse** — the
decoded rollout saturates to a single constant (physical space) and an
extreme constant in z-scored latent space within the first step, for the
entire 200-step horizon. **Stage 2 was not the cause for 114** — Phase 1
alone already fully collapsed; `w_logdet_physical` (Stage-1, encoder-only)
had no visible effect on the propagator's own dynamics at all.

Section 115's own mid-Phase-1 checkpoint (epoch ~119/200) showed something
different: not a literal frozen constant, but a persistent, high-frequency,
spatially fine-grained oscillation that keeps moving for the whole rollout
instead of freezing — still completely unphysical (no resemblance to the
true smooth, large-scale traveling-wave structure), but a qualitatively
different failure mode. Read at the time: `w_logdet_physical_rollout` (which
directly penalizes the Phase-1 rollout's own covariance losing rank) may be
doing *something* — a literal frozen constant is disfavored — but the
propagator may be satisfying that constraint via spurious high-wavenumber
noise instead of genuine dynamics, since nothing in the loss cares *which*
modes carry the required variance.

**Section 116** tested a direct fix for that gap, per user direction: (1)
`--spectral-K 5` (`d_latent=10`, down from Section 115's 24/48) — removing
the physically-unjustified high-wavenumber modes from the representation
architecturally, rather than relying on a loss term to suppress them after
the fact (only wavenumbers `m=1,2,3` fall inside the true instability band
at `L=22`); (2) `w_logdet_physical_rollout` escalated 5x (0.01→0.05).
Section 115 was killed at the user's call before completing ("it's not
going to work") once 116 was smoke-tested and launched. **Section 116's own
mid-Phase-1 checkpoint (epoch ~69/200) also collapsed** — the same complete,
extreme saturation-to-a-constant signature as Section 114, confirmed in
z-scored latent space (the physical-space plot looked merely "low-amplitude"
only because of the raw, unnormalized color scale — normalized, it is the
identical hard fixed-point signature). Killed by the user before completing
Phase 1.

**Net result across Sections 114-116**: three attempts at a log-det barrier
in physical space, at two very different latent dimensionalities (48 and
10) and rollout-penalty weights (0/0.01/0.05), produced three different
*flavors* of failure (extreme-high saturation, spurious high-frequency
oscillation, extreme-low/near-zero saturation) but never genuine chaotic
dynamics. None of the three completed the full Phase 1 + Phase 2 + Gate 3
pipeline to a final `D_KY` (114 did; 115 and 116 were killed on strong
mid-Phase-1 collapse evidence alone) — but the mid-training Phase-1-only
diagnostic technique itself is a useful addition to this project's toolkit:
it can rule out a recipe in minutes from a checkpoint that already exists,
without waiting for Stage 2 or Gate 3 at all.

### 6.7 Divergence instead of collapse, and a first attempt to bound it (Sections 117-119)

Section 117 (K=5, degree=3 — polynomial degree generalized this session from
a hand-written degree-1/2 case split to a general construction supporting any
degree — `poly_norm_power=0.5`, a new knob weakening the per-order
normalization, plus a new `w_var_physical` term mirroring `w_logdet_physical`
but with `decorr_var_loss`'s marginal-variance mechanism) produced something
new: **not collapse, but outright numerical divergence** — the rollout
tracked real-looking structure for ~20 steps, then went to NaN. Diverging
got *worse* with more training (NaN by step 22 at epoch 39, by step 9 at
epoch 79), ruling out "training will fix it." The user's own framing:
"I'd rather see divergence and some kind of accurate modeling than just
collapsing to a point. we can regularize the divergence easier than the
collapse."

This motivated a direct mathematical question: **can the eigenvalues of the
induced differential operator be bounded?** The polynomial's linear
(degree-1) part is a genuine constant-coefficient operator, so each Fourier
mode is an eigenfunction with `λ(k) = Σ_n c_n(ik)^n`. Only even `n`
contributes to `Re(λ(k))` (odd orders are dispersive, not growth/decay), and
boundedness as `k→∞` reduces to one condition: the coefficient on the
*highest* kept even-order derivative must have a specific sign — negative
when that order is `≡0 mod 4` (e.g. order 4, matching KS's own `-w_xxxx`),
but **positive** when it's `≡2 mod 4` (e.g. order 2, ordinary diffusion) — a
sign flip from the `(ik)^n` period-4 cycle that the initial implementation
got wrong for the untested case (caught and fixed the same session, before
it could bite an experiment: verified the current code chooses the sign
generally, not hardcoded). Implemented as `poly_stable_leading`: a hard
reparametrization (`c = sign·(raw)²`) that architecturally guarantees this
one coefficient's sign regardless of training, without suppressing
instability at any other wavenumber (still needed for genuine chaos).

**Section 118** (117's recipe + `poly_stable_leading`): fixed the divergence
completely (full 200-step rollout, no NaN) — but fell straight back into a
full, stark collapse to a fixed point, same magnitude as every earlier
collapsed run. **Section 119** (117's recipe + `etdrk4` instead of `euler`,
no `poly_stable_leading`, motivated by the same numerical-stiffness argument
independently confirmed in §6.8 below): also fixed the divergence, and also
collapsed, just as starkly, by epoch 19. **Both fixes closed the divergence
escape route only to have the model fall into the collapse escape route
instead** — consistent with H-PROP: removing one "cheap" failure mode
doesn't create any positive incentive toward genuine dynamics if the
underlying optimization problem (discover chaos from scratch via a
local, weight-shared function) is unchanged.

### 6.8 Pivot: `spectral_physics_prior`, tested properly for the first time

Every attempt in §6.3-6.7 shares a structural feature: the learned function
is asked to *discover* chaotic dynamics from scratch, and a fixed point is
always the cheapest available answer to that optimization problem. Rather
than trying another variant of the same setup, this session returned to
`spectral_physics_prior` (built 2026-09-08, §6.3/6.4) — bakes in the *exact*
analytic KS right-hand side as a fixed, non-learned baseline; the learned
polynomial adds only a residual correction, zero at initialization. This
changes the incentive structure directly: the "lazy" solution (correction
≈ 0) is no longer a collapsed fixed point — it is the *exact true KS
equation*, genuinely chaotic by construction. Training only has to preserve
that property while adapting to the encoder's actual `z`, not invent chaos
from nothing.

**Why this hadn't worked before**: Section 105 (§6.3) had tried
`--spectral-physics-prior` once already, with `euler` — and produced
`loss=nan` from epoch 0. Root-caused directly this session, independent of
any training dynamics: `physics_prior` bakes in the exact `-w_xxxx` term,
which is numerically stiff; explicit Euler's stability limit at `K=24`,
`L=22` is `dt < 2/k_max^4 ≈ 0.0009`, but the actual substep used is `0.333` —
a ~370x CFL violation, a textbook numerical-analysis failure, not a
collapse/training issue. `etdrk4` handles this exact stiffness via the same
exponential/contour-integral formula the real KS solver itself uses (the
whole reason the reference solver uses ETDRK4 in the first place).

**Verified directly before spending any training compute** (the single most
informative check in this whole arc): built a completely fresh,
zero-init, `physics_prior`+`etdrk4` propagator — literally zero propagator
training — and paired it with Section 104's own already-trained AE
checkpoint. Result: genuine spatiotemporal structure in both physical and
latent space — persistent, moving, traveling-wave-like boundaries for the
full 200-step rollout, qualitatively unlike every flat, static collapse seen
in Sections 112-119 (some amplitude saturation was visible, expected since
Section 104's encoder was never trained alongside this exact physics).
**Section 120** launched a real joint Phase-1/Phase-2 training run from
here (`K=24`, `N_w=64`, `etdrk4`, `physics_prior=True`, polynomial
correction degree=2, standard normalization, Section 104's own
`w_shape_floor=0.1` baseline, no other new regularizers — isolating this
mechanism's own effect cleanly). Smoke-tested clean: no NaN, and Phase-2's
`val_kmax_mse≈0.093` across 5 smoke epochs — far above the suspiciously tiny
values (~0.0005-0.005) every collapsed run in this arc showed at the
equivalent stage. **A real mid-training checkpoint (epoch 19/200) told a
worse story than the smoke test suggested**: the rollout visibly *damped*
the built-in chaos — decaying from the untrained baseline's persistent,
full-amplitude traveling-wave motion into a temporally-static (though still
spatially non-trivial) pattern within ~20 epochs. Killed before completing.

**Diagnosis** (user-prompted: "how do we preserve the chaotic structure and
nudge it in the direction we want?"): this is a known failure mode in
learning chaotic systems from short-horizon MSE loss. A chaotic system
amplifies any state error exponentially, so an imprecise encoder `z`
(always true early in joint training) turns into large prediction error
within a few steps under real chaotic dynamics — and damping is the
cheapest way for gradient descent to reduce that error, since a contracting
system's errors shrink regardless of input quality while a genuinely
chaotic one's grow regardless of model quality. Even starting from the
*exact* KS equation, `w_pred`'s pointwise loss directly rewards a learned
correction that damps the built-in chaos away.

**Section 121, a homotopy/continuation fix** (the user's own proposal,
translated directly: "could we progressively add systems we know are
chaotic? ... sort of a dynamic system gradient descent"): a new mechanism,
`correction_scale` (`ks_latent.models.propagator._SpectralPDEDeltaBody`),
scales the learned correction's contribution —
`field(z) = exact_KS(z) + correction_scale·learned_correction(z)` — and a
new `Stage1TrainingConfig.physics_prior_correction_warmup_epochs` ramps it
linearly from `0.0` to `1.0` over the first 150 of 200 Phase-1 epochs
(held at `1.0` thereafter, matching Stage 2's own default). At
`correction_scale=0`, the propagator is *exactly* the true KS equation,
completely untouched by the learned correction — giving the encoder a
long runway to converge under real, undamped chaotic dynamics before the
correction has enough room to find the damping shortcut. Verified directly
(not just smoke-tested): forcing `correction_scale` to `0.0` vs `1.0` on the
same randomly-initialized model produces different, both-finite outputs,
confirming the gate actually works; full test suite (238 tests) passes, no
regressions. Launched in place of Section 120. **Result: still collapsed**
— a mid-training checkpoint (epoch 39/200, `correction_scale≈0.26`, i.e. a
74%-exact-physics/26%-learned-correction mix) showed a near-total collapse
to a saturated constant in normalized latent space, barely better than the
un-warmed-up Section 120. That a mix still dominated by the exact equation
collapsed this much raised a sharper question: is the *encoder* drifting
away from being a faithful physical-field representation under joint
training (nothing besides plain reconstruction loss anchors `z`'s Fourier
semantics to what `physics_prior` assumes), which would undermine the
prior's exactness guarantee even at low `correction_scale`?

**Section 122: is the ViT itself part of the problem?** User's read: "we've
artificially constrained the encoder quite a bit... the vit might not be
the right model for this." `SpectralFieldAutoencoderConfig` (previously
hard-coded to wrap a ViT) was generalized to accept a plain, fully-connected
`KSAutoencoderMLP` as the field-producing inner model instead (no
attention, tokenization, or windowing at all — every output position can
depend on every input position with no architectural locality bias) —
`--spectral-field-inner mlp`. Both inner models expose an identical
`encode`/`decode` interface, so this is a clean swap; kept Section 121's
correction-scale warmup active on top (testing the encoder change as an
addition, not a replacement). Verified: 23 new/updated unit tests for the
`mlp` branch, full suite (631 tests) passes. Smoke-tested clean, and
notably ~4x faster per epoch than the ViT (4.9s vs ~17-20s at this sizing).
**Result: still collapsed** — a mid-training checkpoint (epoch 79/200,
`correction_scale≈0.54`) showed the same qualitative signature as every
ViT-based attempt: the rollout settles into a temporally static (though
still spatially non-trivial) pattern within ~10 steps. **This rules out the
ViT's own architecture as the primary cause** — the same failure persists
regardless of encoder family, pointing back to the loss function itself
(`w_pred`'s pointwise MSE) as the dominant, architecture-independent driver.

**Section 123: ramp `w_pred` itself, not just the correction.** User's
direct proposal: "why don't we just have a schedule that slowly ramps up
w_pred loss." A new `Stage1TrainingConfig.w_pred_warmup_epochs` applies the
same homotopy idea one level up — linearly ramps the *effective* `w_pred`
weight from `0.0` to its full value over the first `N` epochs, so the
encoder gets real pressure-free time to become a faithful reconstruction of
the physical field before any prediction loss touches it at all, rather
than only delaying the learned correction's own growth. Launched with both
warmups together (`--w-pred-warmup-epochs 40` alongside the existing
150-epoch correction-scale warmup, MLP inner encoder kept). **Result: still
collapsed** — a mid-training checkpoint (epoch 19/200) already showed the
familiar static-pattern signature, despite `correction_scale≈0.13` (mostly
pure physics) at that point — a sharper failure than any of the previous
attempts, since so little of the learned correction was active yet.

**Section 124, a decisive isolation diagnostic**: is the collapse coming
from the learned correction at all, or from the *encoder* drifting
independent of it? `physics_prior_correction_warmup_epochs` was set to
100,000 (`correction_scale` stays at ≈0.002 for an entire 200-epoch run —
indistinguishable from exactly 0), with `w_pred` at full weight from epoch
0 (no warmup) — the propagator is *literally the exact KS equation* for the
whole run, nothing ever learned in the dynamics at all, while the encoder
trains under full prediction pressure. **Result: still collapsed**, by
epoch 19, identical signature. Since there is no learned correction to
blame here at all, this conclusively implicates the *encoder itself*: under
`w_pred` pressure, with nothing in plain reconstruction loss preventing it,
the encoder is free to map temporally-diverse real states to similar,
nearby `z` values — which trivially minimizes prediction error against a
fixed, exact propagator regardless of whether the real underlying dynamics
are chaotic. **The collapse is a representational phenomenon in the
encoder, not something happening in the learned dynamics at all** — a
distinct failure mode from every other one this document has described,
and one none of the dynamics-focused fixes (warmups, `poly_stable_leading`,
integrator choice, encoder architecture family) could have touched.

**Section 125, testing the direct fix**: `w_logdet_physical` (built earlier
this session, §6.6 — the full-covariance log-det anti-collapse barrier,
applied to `decode_from_spectrum(z)` computed from real encoded data, not
any rollout) had been left off in every `physics_prior` run so far to
isolate the mechanism cleanly. Added back (`0.01`, its previously-calibrated
weight) on top of Section 124's exact diagnostic setup (`correction_scale`
still pinned at ≈0, `w_pred` still at full weight from epoch 0) — this
directly targets the just-diagnosed mechanism (real-data encoder collapse)
rather than anything about the dynamics or the correction. Smoke-tested
clean (`val_recon_final=0.021` after 5 epochs, no divergence). **Result:
still collapsed** — a mid-training checkpoint (epoch 29/200) showed the
identical static-pattern signature. This clarified *why* a batch-level
covariance check doesn't help here: a batch mixes many unrelated snapshots
from different times/trajectories, so it can show full aggregate diversity
across the dataset while still flattening any *specific* real trajectory
segment locally in time — a different, more local granularity of collapse
than a global rank check can see.

**Section 126, a temporal-structure-specific fix**: user's direct proposal
— "build and test that, but design it for w space and for z space where
w = irfft(z)." A new mechanism, `temporal_expansion_floor_loss`
(`ks_latent/training/losses.py`), is a one-sided floor (same design
convention as `spectral_shape_floor_loss`/`w_shape_floor` — real-data-
calibrated, never penalizes exceeding it) on how close together, in
representation space, real states some fixed number of real steps apart
(`temporal_floor_lag`, default 1) are allowed to become. The reference
floor (`reference_temporal_separation`) is computed *once* from real
ground-truth data via the same fixed transform used everywhere else
(`encode_to_spectrum`/`decode_from_spectrum`), entirely independent of the
current encoder — no circularity, the target doesn't move as the encoder
trains. Built both scopings requested, as independent weights:
`w_temporal_floor_z` (z-space, the rFFT coefficients directly) and
`w_temporal_floor_w` (w-space, `decode_from_spectrum(z)`) — the same
z-vs-physical distinction this project draws elsewhere (`w_logdet` vs
`w_logdet_physical`). Verified directly (not just smoke-tested): a
synthetic test confirmed collapsed (constant-over-time) states get heavily
penalized, states matching the real reference distribution get only a
small residual loss, and states *more* separated than the reference get
**exactly zero** loss (confirming the one-sided floor never fights
legitimate expansion). 8 new unit tests
(`tests/unit/test_temporal_expansion_floor_loss.py`) cover construction,
gradient flow, one-sidedness, and `lag>1`; full suite passes. Launched
(both terms at `0.01`, on top of Section 124/125's exact diagnostic setup
— `correction_scale` still pinned at ≈0, isolating whether this
regularizer alone can keep the encoder's own representation from
collapsing with zero learned dynamics at all). **Result: still
collapsed** (full 200-epoch run completed, `val_recon_final=0.0089`; the
rollout was already flat by epoch 39). This clarified exactly why a
Stage-1, real-data-only regularizer — no matter how well-targeted at the
representation — can't be sufficient here: it only checks properties of
real, single-lag-apart data pairs, never actually running the propagator
at all, so it cannot constrain what happens once the model is iterated
purely autoregressively for many steps with no re-anchoring to real data.

**Section 127, querying the dynamics directly.** The user asked a
clarifying question that reframed the problem: "we could try a rollout
variant, but it's the propagator already collapsing in stage 1? so it's
sort of irrelevant?" The resolution: with `correction_scale` pinned near
zero, the propagator's own step has no free parameters to adjust — but
the *encoder* still chooses where in z-space real states land, and a
fixed nonlinear map can be locally expansive in one region of phase space
and contractive in another. A rollout-style check isn't irrelevant, but a
cheaper and more direct version of the same idea is: does the propagator's
own step-Jacobian, evaluated at the encoder's real z, have a largest
singular value `>1` (local expansion) or `<1` (local contraction) — the
literal definition of a positive local Lyapunov exponent. This project's
own Gate 4 D9 diagnostic (`propagator_step_jacobian_spectral_norms`,
`prop_jacobian_med`) already measures exactly this quantity post-hoc, via
`torch.func.vmap(jacrev(...))`; `propagator_local_expansion_floor_loss`
turns it into a differentiable training-time regularizer with the same
technique. `floor=1.0` needs no real-data calibration at all (unlike the
temporal floor) — "does not contract" is an absolute criterion. Verified
directly: a genuinely expanding toy map gets exactly zero loss, a
genuinely contracting one gets penalized by the exact expected amount, and
gradient flows correctly back through a real `spectral_pde` propagator's
own step into `z`. Found and fixed a real device gap along the way:
`torch.linalg.svdvals` has no MPS kernel — moved just that one op to CPU
(a differentiable device transfer, cheap given `d≈48`). The full Jacobian
computation is expensive (~0.5-1s per call, measured directly, versus
~4.5-8s for an entire epoch otherwise), so it's applied only *once per
epoch*, on a small 32-sample subsample, rather than every batch — a real
`--amp` smoke run confirmed this keeps per-epoch overhead negligible.
5 new unit tests pass; full suite re-verified for regressions. Launched
(weight `0.1`, on top of Section 124-126's same diagnostic setup —
`correction_scale` still pinned at ≈0, temporal-floor terms dropped this
round to isolate this mechanism's own effect). **Result: still
collapsed** — a mid-training checkpoint (epoch 39/200) showed the same
extreme saturation-to-a-constant signature as Section 114/116/etc, this
time in latent space too (a more complete collapse than Section 126's own
milder, spatially-non-trivial pattern). Killed by the user before
completing, with the explicit framing: "please take stock. I don't know
if there is a way to make this work."

### 6.9 Taking stock: what the encoder-collapse sub-arc (Sections 120-127) established

**Six mechanistically distinct interventions, all targeting the same
diagnosed failure, all failed the same way.** Section 124's isolation
diagnostic is the load-bearing result here: with the learned correction
pinned at exactly zero — the propagator *is* the exact analytic KS
equation, nothing learned in the dynamics at all — the rollout still
collapsed once the encoder trained under real `w_pred` pressure. Since
there was no learned dynamics to blame, this conclusively located the
failure in the *encoder*: nothing in plain reconstruction loss stops it
from mapping temporally-diverse, causally-connected real states to nearly
the same `z`, which trivially reduces prediction error against a chaotic
target (any imprecision in a chaotic system's state gets amplified
exponentially; a collapsed representation's errors don't, so gradient
descent prefers it). Every subsequent attempt targeted this exact
mechanism from a different angle, and every one failed identically:
delaying the correction's growth (§6.8, `correction_scale` warmup),
delaying `w_pred` itself (§6.8, `w_pred_warmup_epochs`), replacing the ViT
encoder with a plain MLP (ruling out encoder architecture family as the
cause), a batch-level covariance-rank check (`w_logdet_physical`), a
real-data pairwise temporal-separation floor in both z-space and w-space
(`temporal_expansion_floor_loss`), and a differentiable local-expansion
floor directly querying the propagator's own step-Jacobian at real data
(`propagator_local_expansion_floor_loss`). All six are regularizers added
*on top of* `w_pred`, fighting a consistent bias rather than removing it —
and none gave the encoder+propagator pair a genuinely different, non-
collapsing way to reduce the loss.

**A theory unifying this with the original H-PROP finding.** H-PROP
(§3) says a propagator needs global, unconstrained mixing to sustain chaos
under training; anything local collapses. The encoder-collapse mechanism
found here is the same story one level up: when the *propagator* has no
freedom at all (physics_prior, correction pinned to zero), the only
remaining place gradient descent can go to reduce prediction error on a
chaotic system is the *encoder*, and it takes that escape route every
time, regardless of which regularizer is added. A genuinely unconstrained
propagator (`mlp`, `fourier_mlp` — the H-PROP winners) apparently gives
the *joint* (encoder, propagator) system enough alternative ways to fit
the data that it never needs to collapse the encoder to do so. A local or
exactly-fixed propagator removes that alternative, making encoder collapse
the only cheap answer left — and no amount of penalizing the *symptom*
changes the underlying *incentive*.

**Is there a way to make this work?** Two genuinely different directions
might, but both are real redesigns, not tunable regularizers, and neither
is guaranteed:
- Replace `w_pred`'s pointwise MSE with a loss matching long-run
  *statistics* (energy spectrum, invariant measure) instead of trajectory-
  by-trajectory prediction — the established fix in the turbulence-closure
  ML literature for training surrogates of chaotic systems, since pointwise
  MSE is arguably the wrong objective in principle for a chaotic target,
  not merely under-regularized.
- Freeze the encoder *permanently* on reconstruction alone, with `w_pred`
  never touching it at all, attaching dynamics only afterward — untested
  in this exact (hard, permanent) form; the warmup version tried here
  still eventually exposed the encoder to prediction pressure, which is
  exactly when collapse set in each time.

Given the consistency of the failure across six independent angles, this
project's own conclusion at this point is that a local/PDE-form propagator
trained jointly with an encoder via pointwise prediction loss has run into
a real, structural wall — not a tuning problem this sub-arc's remaining
budget is likely to solve. **External validation**: AROMA (arXiv:2406.02176)
independently reports KS as an explicit failure case for this general class
of approach — chaotic spectral fidelity needed reconstruction MSE
~1e-10-1e-12, with the *decoder* as the bottleneck — consistent with this
project's own experience, not merely this project's bad luck with KS
specifically.

### 6.10 If not a PDE, then what? Extracting physics from latent geometry without one

The user's own reframing of the underlying goal: "the whole idea would be
to use the latent dynamics geometry to learn something about the physics
of the larger system... it's possible we could do this without learning a
pde." This section catalogs what this project has *already* achieved in
that broader sense (no PDE involved), and what well-precedented,
previously-identified-but-never-completed next steps remain — all
recoverable from this project's own earlier notes
(`docs/LATENT_PDE_RESEARCH_NOTES.md`, `docs/PROJECT_HANDOFF.md`,
`docs/LATENT_PDE_EXPERIMENTS.md`), not new speculation.

**Already achieved, no PDE required:**
- **Embedding-dimension validation**: the latent's dimensionality (`44 ≈
  2×22`, twice the attractor's own `D_KY`) matches what classical embedding
  theory (Whitney 1936/1944; Takens 1981; Sauer-Yorke-Casdagli 1991)
  predicts for a faithful embedding of a 22-dimensional attractor — a real,
  rigorous mathematical validation of the learned representation, with zero
  governing equation involved.
- **`D_KY`/Lyapunov spectrum matching published KS literature**: this
  project's own measured `D_KY` (both the `L=100` and `L=22` benchmarks, see
  §1's correction) agrees with independent sources — Edson, Bunder, Mattner
  & Roberts (2019, arXiv:1902.09651): `D_KY≈0.226·L−c`; a Koopman-analysis
  paper (arXiv:1909.00076) independently states `D_KY=23.2` at `L=100`.
- **D4 translation equivariance** emerging in some architectures is itself
  a physical statement (KS *is* translation-invariant on its periodic
  domain) that nothing forced the model to discover.
- **Topology**: DTM/persistent-homology analysis (already run, Gate 3) found
  the `L=100` latent attractor genuinely topologically trivial — a
  defensible negative result in its own right, not merely "visualization
  failed."

**A specific prior failure, already diagnosed as self-inflicted and
reversible.** This project already tried DMD/SFA/seriation once (an
earlier, non-`spectral_field` latent) as a no-PDE way to extract modes/
frequencies — found "unimpressive... DMD freqs tiny (0-0.012)." The
project's own contemporaneous diagnosis identifies two specific, fixable
causes, neither of which is "KS admits no such structure":
1. That latent was aggressively **decorrelated** (`w_decorr`/`w_var`
   driving `Cov(z)→I`) — DMD/SFA need exactly the correlational structure
   this destroys. (Sections 104-127's own `spectral_field`/`physics_prior`
   latents all use `--w-decorr 0`, so this specific confound may not even
   apply to them.)
2. KS's continuous translation symmetry means the attractor is a **group
   orbit** — travelling-wave coherence lives in a phase/drift variable that
   was never separated from the underlying shape, so any mode-finding
   method looking at the raw (unreduced) state sees drift noise, not
   structure.

**Concrete next steps, in priority order** (all previously identified in
this project's own notes as "optional"/"future work," none completed):

1. **Method of slices / symmetry reduction** (Budanur, Cvitanović,
   Davidchack & Siminos; see also Cvitanović, Davidchack & Siminos 2010,
   *SIAM J. Appl. Dyn. Syst.* 9(1):1-33, arXiv:0709.2944) — quotient out
   continuous translation *before* any further analysis, splitting the
   state into a "shape" component (stationary in the reduced frame,
   turning travelling waves into relative equilibria) and a single phase
   variable. Directly targets cause 2 above. Flagged in this project's own
   notes as "cheap, high value," never implemented.
2. **Redo DMD/Koopman mode analysis**, ideally after step 1, on a latent
   that is *not* aggressively decorrelated (e.g. the `spectral_field`
   latents from the PDE arc, or a `w_decorr=0`/low-`w_var` H-PROP-winning
   propagator). Real KS-specific precedent exists for this being fruitful:
   arXiv:1909.00076 (Koopman analysis of KS, independently reproducing
   `D_KY`); arXiv:2310.10745 (Mori-Zwanzig latent-Koopman closure, "a
   KS-specific treatment").
3. **Covariant Lyapunov vectors → "physical dimension" vs. `D_KY`
   comparison** — flagged in this project's own notes as well-defined,
   tractable, and "publishable," never completed. Needs no new training,
   only post-hoc analysis of an already-converged model.
4. **Unstable-periodic-orbit (UPO) / recurrence search in the latent** —
   characterize the attractor by its skeleton of near-recurrent states
   rather than a governing equation. This is the core of Cvitanović's own
   published research program for KS specifically (Ding, Chaté, Cvitanović,
   Siminos & Takeuchi 2016, "Estimating the dimension of an inertial
   manifold from unstable periodic orbits," *PRL* 117:024101) — a
   genuinely different, well-established notion of "understanding the
   physics" that doesn't require any explicit equation, local or global.
   Flagged in this project's own notes as "optional," never attempted.
5. **Mori-Zwanzig (MZ) reframing**: if a reduced description doesn't close
   instantaneously (as this whole arc found — a local closure keeps
   collapsing or diverging), that failure to close *is* the memory term in
   MZ theory, not a bug — model it explicitly (auxiliary latent memory
   variables, or a short delay-embedding window) instead of forcing an
   instantaneous local closure. Directly reframes this arc's central
   negative result as an expected, principled outcome rather than a wall.

**If a from-scratch local architecture is still of interest** (rather than
a post-hoc analysis of an already-good model): Constante-Amores, Linot &
Graham (arXiv:2410.01238, *Phys. Rev. E* 2026) already do patch-decomposed
AE + neural-ODE local models on KS (and Kolmogorov flow) with shared
weights across patches — the closest existing prior art to "build locality
in from the start" rather than retrofitting an accuracy-optimized encoder.
Required reading before any further attempt in that specific direction,
both to avoid duplicating their contribution and because they may already
report how (or whether) they handled the same encoder-collapse mechanism
this arc just spent Sections 120-127 discovering independently.

**Explicitly available but previously deprioritized** (per direct user
instruction, 2026-09-02, in the context of the now-abandoned coordinate-
search-for-local-PDE-discovery effort — worth reconsidering given the
present reframing is a different goal than that one was): Champion, Lusch,
Kutz & Brunton (2019), *PNAS* 116(45):22445 (SINDy-autoencoder); Kemeth et
al. (2022), *Nat. Commun.* 13:3318 (emergent-space PDEs); Lusch, Kutz &
Brunton (2018), *Nat. Commun.* 9:4950 (deep Koopman embeddings); Williams,
Kevrekidis & Rowley (2015), *J. Nonlinear Sci.* 25(6):1307 (Extended DMD).

---

### 6.11 The real motivation: weather-dynamics modeling and data assimilation

Per the user (2026-09-09): the actual purpose of this KS-latent project is
to develop, on a cheap and well-understood testbed, autoencoder + latent-
dynamics methodology that will later be applied to real weather dynamics
(for a collaborator, "Peter Jan") and to data assimilation specifically.
This section is written with that end goal in mind — it asks, for each
piece of real published precedent, "does this validate something this
project has already built, or does it point at a gap this project should
close before the jump to weather?"

**KS/Lorenz-96 as an accepted low-dimensional testbed for DA and latent
methods aimed at weather — this project's basic strategy is standard
practice, not an idiosyncratic simplification.** The 1D KS equation and the
Lorenz-96 system are both routinely used as the intermediate step between
"toy chaos" and "operational NWP" when developing new DA algorithms:
comparative sequential-DA studies use KS directly as the test case (e.g.
"Comparative study of sequential data assimilation methods for the
Kuramoto-Sivashinsky equation"); a closely-related effort — Boudier,
Fillion, Gratton, Gürol & Sadok, "Discovery of interpretable structural
model errors by combining Bayesian sparse regression and data assimilation:
a chaotic Kuramoto-Sivashinsky test case" (arXiv:2110.00546, *Chaos*
32:061105, 2022) — combines DA with sparse regression on KS specifically to
recover *interpretable* model-error terms, i.e. almost exactly this
project's own "PDE-discovery via DA-adjacent tooling" ambition, on the same
equation, from a group that does DA for operational NWP (CERFACS/Meteo-
France). This paper is worth reading directly before the DA phase begins —
it is the closest single precedent for "KS + interpretability + DA" as one
combined program, and it will say plainly whether their structural-error
discovery ran into the same collapse-under-chaotic-MSE failure mode
Sections 120-127 just spent this whole arc diagnosing.

**Latent-space data assimilation is an established, active subfield —
directly validating the project's Stage-1/Stage-2 latent-DA split as a
real methodology, not a toy exercise:**
- **Peyron, Fillion, Gürol, Marchais, Gratton, Boudier & Goret (2021),
  "Latent space data assimilation by using deep learning," *QJRMS*
  147(740):3759** (already in this project's own
  `docs/LATENT_PDE_RESEARCH_NOTES.md` reference list, confirmed real and
  re-verified this turn) — builds an ensemble transform Kalman filter with
  model error directly in an autoencoder's latent space (ETKF-Q-Latent),
  tested on an augmented Lorenz-96 system, and shows it beats the
  model-space ETKF-Q on both cost and accuracy. This is the closest
  possible structural precedent for what this project's own
  `ks_latent`/DA scripts already do (see §5.2) — it validates doing DA
  *in the learned latent*, at exactly this project's scale (low-dimensional
  chaotic toy system), before scaling to real atmospheric fields.
- **Fan, Bai, Fei, Xiao, Chen, Liu, Qu, Ling & Gentine, "Physically
  consistent global atmospheric data assimilation with machine learning in
  latent space," *Science Advances* (2026), arXiv:2502.02884** — the direct
  scale-up of the Peyron idea to real, operational, multivariate global
  atmospheric fields: Latent Data Assimilation (LDA) performing Bayesian DA
  in a latent space learned by an autoencoder trained on real reanalysis
  data, reporting that latent-space assimilation improves both analysis
  quality and forecast skill over model-space DA, and remains robust even
  when the autoencoder is trained on imperfect forecasts. This is the paper
  that answers "does the Peyron approach still work at weather scale" —
  yes — and is the most directly relevant single citation for the
  eventual Peter Jan hand-off, since it is doing, on real weather data,
  precisely the pipeline shape (encode → assimilate in latent → decode)
  this project has been building and testing on KS.
- **A companion/follow-on line, "AE-O2L" (Monthly Weather Review 153(8),
  2025)** — an autoencoder-observation-to-latent-space network explicitly
  designed for interpretability of the background/observation assimilation
  step, i.e. addressing the same "black-box latent DA" concern this
  project should anticipate being raised about its own latent.
- **Fan, Xiao, Qu, Nathaniel, Ling, Fei, Bai & Gentine, "Learning more
  physically realistic dynamics in machine-learning based weather
  forecasting with latent-space constraints," arXiv:2510.04006 (2025)** —
  reformulates ML weather-forecast training as a 4D-Var problem and moves
  the *loss* itself into latent space (rather than raw grid MSE), reporting
  more physically realistic rollouts and better preserved fine-scale
  structure. This is directly relevant to the encoder-collapse mechanism
  this project just diagnosed (§6.9): it is independent, real-weather-scale
  evidence that where you put the loss (state space vs. latent space)
  materially changes what a chaotic system's training dynamics reward or
  punish — the same axis this project's own w_pred-vs-collapse finding
  turned on, but observed by a different group, on a different chaotic
  system, at far larger scale. Worth reading in detail: if their latent-
  space loss formulation avoids this project's own collapse failure mode
  (or hits an analogous one), that is either a validated fix or an
  independent confirmation of the same problem, either of which is exactly
  what's needed to decide the next architectural move here.
- **LD-EnSF (arXiv:2411.19305)**: synergizes latent dynamics with ensemble
  score-based filters for fast DA under sparse observations — another
  concrete latent-space DA design pattern (ensemble + score-based, as
  opposed to Peyron's ETKF-Q or Fan et al.'s variational/4D-Var framing),
  worth having as a second reference architecture when the DA design phase
  starts, since it makes a different assumption (score-based generative
  prior vs. Gaussian ensemble) about the latent's error distribution.

**Koopman/DMD as a bridge between "physics from latent geometry" (§6.10)
and "weather forecasting," independent of this project's own KS-specific
Koopman citations:**
- **KODA (arXiv:2409.19518)** — separates a time series into a Koopman-
  operator-modeled physical component (via a Fourier-domain filter) plus a
  learnable residual/recursive model for what the Koopman operator can't
  capture, and reports it beating existing methods on forecasting
  benchmarks *including weather data specifically*. This is a concrete,
  already-realized version of the Mori-Zwanzig reframing already proposed
  in §6.10 point 5 (explicit residual/memory term instead of forcing full
  closure into one operator) — but demonstrated at weather scale, which
  makes it a good target architecture if the MZ-reframing direction is
  pursued with the eventual weather application in mind from the start.
- **"Deep Learning for Koopman Operator Estimation in Idealized Atmospheric
  Dynamics" (arXiv:2409.06522)** — Koopman-operator learning applied
  directly to idealized (not full-GCM) atmospheric dynamics, i.e. a testbed
  positioned exactly between this project's KS work and Fan et al.'s
  full-scale reanalysis-trained models. Good next rung on the ladder if the
  Koopman/DMD redo recommended in §6.10 point 2 is pursued with weather as
  the eventual target, rather than stopping at KS.

**What this means concretely for the Peter Jan hand-off, stated plainly:**
the field-level conclusion of this arc (§6.9) — that a jointly-trained
local closure collapses the encoder under chaotic pointwise MSE — is not
a KS-specific curiosity. Fan et al. (2510.04006) independently observing
that *where* the loss is computed (grid space vs. latent space) changes
learned-dynamics realism at full weather scale is the same axis. Before
committing the weather/DA effort to a specific architecture, it would be
worth explicitly checking whether Fan et al.'s latent-space-loss
formulation is a real fix for this project's own collapse mechanism (by
testing an analogous "loss computed after re-encoding the prediction,
rather than on raw predicted values" formulation on KS first, since it is
so much cheaper to iterate on here) — turning this project's own negative
result into a concrete, testable connection to the literature already
being invoked for the hand-off, rather than treating KS and weather as two
disconnected problems that happen to share an architecture family.

---

### 6.12 What Sections 99-127 actually built: a portable toolkit for any chaotic latent-dynamics project

User question (2026-09-10): "is there anything we can build with any of the
section _ experiments. I feel like we must have learned something that can
extend to chaotic systems, weather and DA." Sections 99-127 were, by their
own final verdict (§6.9), a **failed attempt at a specific artifact** (a
literal, local, interpretable latent PDE). But a failed artifact is not the
same as a failed arc. Five things came out of it that are not KS-specific
and are ready to reuse, unmodified in spirit, on the next chaotic system —
including a weather emulator. Each entry below names the exact code and
exactly what carries over.

**1. A pre-flight protocol for "will jointly training this dynamics model
collapse my encoder," runnable before spending real compute.** The
single most expensive lesson of this whole arc (six independent
interventions, Sections 120-127, each smoke-tested and each failing the
same way) reduces to one diagnostic question, and Section 124 shows how to
ask it cheaply: **freeze the dynamics model to something known-correct
(or maximally expressive) and check whether the encoder still collapses.**
If it does, the encoder/loss combination is the problem, not the dynamics
architecture — no amount of regularizing the dynamics model will fix it.
Concretely, before committing to a constrained/local/physically-structured
dynamics model on any new chaotic system (weather included):
   - Train the encoder+decoder jointly with a **maximally expressive**
     propagator first (H-PROP's own winning condition: global receptive
     field, unconstrained mixing — `mlp`/`fno_vit`-style, not attention
     with softmax). If this collapses too, the problem is upstream of the
     dynamics model entirely (loss placement, data, or encoder capacity).
   - Only then swap in the constrained/interpretable dynamics model you
     actually want, with the encoder initialized from the run above. If
     collapse reappears *only* now, it is caused specifically by removing
     the propagator's expressive "escape route" — exactly the mechanism
     diagnosed in §6.9 — and the fix has to touch the dynamics model's
     capacity/parameterization, not the encoder's regularizers.
   This protocol is generic to any project pairing an autoencoder with a
   constrained dynamics model on a chaotic target, and it would have
   saved most of Sections 120-126 (four of which regularized the
   *symptom* — real-data separation floors, log-determinant barriers —
   rather than running this one isolation check first).

**2. A cheap, differentiable "is this propagator still chaotic" health
check, usable as either a training regularizer or a five-minute post-hoc
gate.** `propagator_local_expansion_floor_loss`
(`ks_latent/training/losses.py`, Section 127) computes the propagator's
own per-sample step-Jacobian spectral norm at real encoded states via
`torch.func.vmap(jacrev(...))` — the same technique
`ks_latent.analysis.diagnostics.propagator_step_jacobian_spectral_norms`
already uses post-hoc for Gate 4's D9 diagnostic. A genuinely chaotic
system's propagator must be locally expansive (top singular value > 1)
somewhere in phase space; a model that has quietly collapsed to a fixed
point or a contraction will fail this instantly, and it is orders of
magnitude cheaper than a full Lyapunov-spectrum computation (Benettin QR,
CPU float64, the project's existing Gate 3 machinery). For a weather
emulator this is directly applicable as a one-line training-time canary:
run it every few epochs on real (or reanalysis) states and alarm if the
top singular value across the sampled states drops toward 1 — the exact
signature this arc watched happen to KS, generalized to any learned
propagator over any latent representation. (Caveat carried over honestly:
Section 127 itself did not, on its own, prevent collapse when the
propagator's own correction was pinned near zero — the mechanism failed
as an *anti-collapse loss* at floor=1.0 in that specific isolation setup.
Its value demonstrated here is as a **diagnostic/canary**, not as a proven
fix; do not re-claim it as a solved anti-collapse regularizer.)

**3. The physics-prior-plus-learned-correction pattern, with a
continuation schedule, is the same shape as the dominant real
hybrid-weather-model paradigm — and this project already has a working,
tested implementation of it.** `_SpectralPDEDeltaBody`'s
`physics_prior=True` mode (`field() → prior + correction_scale *
correction`, `ks_latent/models/propagator.py`) plus
`physics_prior_correction_warmup_epochs`'s linear ramp
(`ks_latent/training/loops.py`) is structurally identical to what
operational hybrid weather models do: start from a known, trusted physical
operator (here, KS's own `-w_xxxx` term and the rest of the true
equation; in a weather model, a numerical dynamical core or an existing
physical parameterization) and let a learned correction fade in on top of
it, rather than learning the whole map from scratch. This project's own
negative result (§6.9) narrows exactly what this pattern is and is not
good for: it did not, by itself, prevent encoder collapse under a chaotic
pointwise-MSE loss (that failure is upstream, in the loss/encoder
interaction, not in the physics-prior mechanism). But the homotopy
mechanism itself — ramp a *known-good* fallback down to zero while a
learned term ramps up, rather than initializing the learned term at full
strength from epoch 0 — is a real, reusable stabilization technique for
*any* hybrid physics+ML training, independent of whether the collapse
problem is fixed, and is directly reusable for a weather closure/
correction model.

**4. A portable technique for guaranteeing UV (small-scale) numerical
stability in a learned local operator.** `poly_stable_leading`
(`ks_latent/models/propagator.py`, Sections 117-119): when a learned local
operator is parameterized as a polynomial in spatial-derivative order
(as any local finite-difference-style closure or stencil model is), the
sign of the leading even-order linear coefficient determines whether
`Re(eigenvalue) → ±∞` as wavenumber → ∞ — i.e. whether the model is
UV-stable or UV-explosive by construction. The fix here (architecturally
constrain that one coefficient's sign via `-1.0 if highest_even_order % 4
== 0 else 1.0`, derived from `Re((ik)^n)`'s period-4 cycle, not just
"always negative" — a real bug this arc caught and fixed) is a general
numerical-analysis trick for **any** learned local closure model with a
polynomial or finite-difference parameterization, including learned
subgrid-stress/parameterization schemes for weather and climate models,
where guaranteed-stable small-scale behavior is exactly the property a
naively-parameterized learned closure is most likely to lack and most
dangerous to lack silently (a closure that is UV-unstable in an offline
KS test just NaNs a training run; the same failure mode inside an online
weather model's dynamical core is a blown-up forecast).

**5. A validated architecture-search result usable off the shelf for any
spatially-organized, roughly-periodic (or at least locally-structured)
scientific field.** The canonical-recipe decision already reached in this
project (§5.2 addendum, `docs/RESULTS.md`) — `vit` encoder/decoder with
windowed local attention (`attn_window=4`) and a **fixed, non-learned**
positional encoding beating both a hand-designed patched-transformer and
a plain global MLP by 2-3 orders of magnitude in reconstruction MSE, with
`pos_encoding="linear"` performing statistically indistinguishably from
the periodicity-aware `"circular"` variant — is itself evidence worth
carrying forward: a fixed (not learned) positional signal plus **local**
(not global) attention windowing was the decisive architectural choice,
not periodicity-awareness per se. That is a directly actionable prior for
designing a weather-field encoder: windowed/local attention over a
lat-lon (or icosahedral) grid with a fixed positional encoding — matching
what large operational ML weather models (e.g. Pangu-Weather's 3D
windowed attention, GraphCast's local message passing) already converge
on independently — rather than the global-attention or global-MLP
alternatives this project's own search ruled out at KS scale.

**6. Track accuracy, conditioning, and smoothness as three separate axes
— never let one substitute for the others.** Section 52 (`vit`, globally
mean-pooled) produced this project's best-ever raw rollout accuracy and DA
skill up to that point (rollout MSE 0.0256, `skill_free_over_da=3.12`) and
the smoothest real-trajectory embedding measured to that point (§4) —
while simultaneously having catastrophic linear-algebra conditioning
(`cond#=1.6e6`, effectively numerically singular). A DA system built on a
Kalman-type update — this project's own PFF filter (§5.2, §7), and the
Peyron/Fan latent-DA methods cited in §6.11 — inverts a covariance-derived
matrix every cycle; a near-singular latent covariance is a real
correctness risk (unstable updates, spuriously amplified innovations) that
a prediction-loss or even a DA-skill number will not surface on its own,
since Section 52 scored well on both anyway. Report all three (raw
accuracy, `cond#`, and a smoothness/D9-style diagnostic) for any
weather-latent candidate architecture — this project's own leaderboard-
style comparison tables (§4, §5.3) exist specifically because no single
number was ever sufficient to pick a winner.

**7. Ablate exactly one component at a time, even when it is expensive to
be that disciplined.** Section 98's own instruction ("remove the fourier
mlp... don't change anything else") is the cleanest single-variable
ablation in the project on a multi-part hybrid architecture — Sections
89-97 had been progressively tuning `vit_fourier_hybrid`'s Fourier branch,
sizing, and regularizer weights together. Stripping the Fourier branch
alone and holding everything else fixed produced a result
(`D_KY=22.12`, `n_positive=13`, `val_kmax_mse=0.0332`) that became this
project's actual PDE-arc launch baseline (§6.3) — i.e. "was the added
complexity worth it" only became answerable once it was isolated this way.
Sections 95-96's `w_spatial`/`w_smooth` sweep applies the same discipline
to regularizer weights: it took two separate confirming runs to establish
that `w_spatial` alone, not `w_smooth`, was responsible for a real
accuracy regression (§5.1) — a conclusion a joint sweep could not have
produced cleanly. The same protocol is directly reusable at weather scale,
where a full training run costs far more than at KS scale, which makes
disciplined single-variable ablation *more* valuable there, not less.

**8. Same-instant spatial correlation and dynamical (Jacobian) locality
are different properties — regularizing or architecting for one does not
get you the other for free.** Across the Sections 44-52 sweep
(`docs/LATENT_PDE_EXPERIMENTS.md`'s own measured, not merely proposed,
result), same-instant signed bandedness (D8 — how correlated nearby
latent channels are *at one moment*) was strong and tunable (0.65,
`p=0.0000`, moving monotonically across the sweep) while one-step
propagator-Jacobian bandedness (D3 — whether the *dynamics update* only
couples nearby channels) stayed flat and non-significant (0.26-0.30,
`p=0.18`) across every one of the 9 runs, including the ones with the
strongest D8 bandedness. A latent that "looks" spatially organized at a
glance — correlated neighbors, a banded covariance — is not evidence its
*dynamics* are local; that has to be measured directly (D3-style Jacobian
analysis, or item 2's local-expansion-floor technique adapted into a
bandedness probe), never inferred from instantaneous structure. For a
weather latent this means a spatially-coherent-looking embedding (e.g.
from item 5's windowed-attention recipe) is not by itself grounds to
assume the learned propagator can be made local or patch-decomposed —
that remains a separate, harder claim needing its own direct test, exactly
the test Section 100's `masked_mlp_wide` comparison (§6.2) ran and failed
(8.5x worse accuracy, `cond#=7.9e15`, matching H-PROP's own prediction).

**9. Post-hoc structure-mining on an accuracy-optimized latent doesn't
work — structure has to be trained in, not discovered after the fact.**
`scripts/fit_latent_pde.py`, run directly on Section 98's frozen, real,
already-well-trained latent trajectories (sparse local regression,
SINDy-style), returned `R²≈0.005` and `D_KY=0` (§6.2) — a clean,
unambiguous failure. This representation was never given any training
pressure toward local fittability, so in hindsight this isn't surprising,
but it is still a decisive, useful rule-out: it forecloses "maybe a local
PDE is hiding in an existing accuracy-optimized weather-emulator latent,
waiting to be mined post-hoc." If interpretable local structure is wanted
for the weather work, it needs to be trained for explicitly — via
architecture (item 5), or a loss term with items 1-2's collapse-risk
protocol run first — not searched for afterward in a latent that was
never asked to have it.

**What did *not* generalize, stated for balance:** the actual interpretable
local PDE this arc set out to build (Sections 99-127's headline goal) did
not work, and the diagnosed reason — a local/fixed dynamics model gives
the optimizer no cheaper way to reduce chaotic-target loss than collapsing
the encoder (§6.9) — is itself a warning that would apply with at least
equal force to a weather-scale attempt at a literal, local, physically-
interpretable closure trained jointly with reconstruction+prediction loss.
Items 1-2 above exist specifically to catch that failure early and cheaply
at weather scale, rather than after a full-cost training run, the way it
was caught here only after Sections 120-126 had each already run to
completion. **Update (§6.13): a later sub-arc (Sections 128-140) found a
real, working resolution to exactly this problem — a local, interpretable
closure that also works — by changing WHERE the locality is enforced, not
by tuning the propagator further. Read §6.13 before treating this
paragraph's pessimism as final.**

---

### 6.13 Resolution: a genuinely local, interpretable closure — obtained by moving locality into the encoder (Sections 128-140)

This sub-arc directly revisits §6.9's conclusion and, unlike Sections
99-127, ends in a real success — not by tuning the local propagator
further, but by changing which component carries the locality constraint.

**Sections 128-134: re-confirming H-PROP with a richer local architecture,
then finding the regularizer wasn't the fix either.** `masked_mlp_expand`
(a new 3-layer bottleneck-expanded masked propagator, receptive-field
radius 9 of 44) collapsed the same way every prior local propagator had
(§6.9's own list), regardless of anti-collapse mechanism tried against
it: a top-singular-value-only floor (`propagator_local_expansion_floor_loss`,
Section 127/130) fixed the top direction while the other 43 collapsed
freely (measured directly: 5/44 singular values `>=1.0`, per-step volume
factor `8.9e-5`); a two-group floor calibrated against the wrong
reference statistic (Section 131-133, conflating an instantaneous
per-point singular-value distribution with the asymptotic Lyapunov
exponent it superficially resembles); a full per-rank floor calibrated
against a real, robustly-measured reference spectrum
(`propagator_graded_spectrum_shape_loss`, Section 134, `w=0.5` in both
Stage 1 and Stage 2) — still collapsed under Stage 2's own longer-horizon
training (6/44 `>=1.0`, volume factor `7.98e-5`), even though it looked
healthy right after Stage 1 alone (12/44 `>=1.0`, volume factor `0.66`).
Three regularizer generations, one consistent outcome: the *propagator's*
own architectural locality is the thing that keeps failing, not the
richness or calibration of whatever anti-collapse term rides alongside
it.

**A receptive-field-width red herring, ruled out with this project's own
prior measurement.** Before pivoting further, Phase 2's already-measured
light-cone bound (`v_star=1.261+-0.039` physical units/time, `docs/RESULTS.md`)
was checked directly against `masked_mlp_expand`'s own radius-9 window:
at `dt_snap=1.0`, true KS information only propagates `~1.26` physical
units per step — an order of magnitude *narrower* than radius 9 already
provided, if the flat latent index corresponded to physical position at
all. It doesn't (nothing in a ViT-pooled encoder enforces that), which
pointed at the real culprit directly: not window width, but the flat
index's lack of any architectural tie to physical space.

**Section 135: moving locality into the encoder instead — Phase 10's
original design, built for the first time.** `KSAutoencoderLocalField`
(`ks_latent/models/autoencoder_local_field.py`, `LocalFieldAutoencoderConfig`)
implements the brief's own circular-Conv1d local field with a mandatory
gauge anchor (channel 0 of every site is *exactly* the site's own local
physical average of `u`, never learned) — a genuine `(n_sites,
local_channels)` field, flattened site-major so a flat-index window
means a real physical neighborhood. Paired with `masked_mlp_expand`
again (same propagator, new encoder): this diverged rather than
collapsed — rollout separation *grew* (`26.7 -> 157.7` over 60 steps,
the opposite sign from every 128-134 failure) and the latent covariance
was nearly singular (`cond#=2.06e15`). A different failure mode, but
still a failure, on the *same* propagator architecture on a *different*
encoder.

**Section 136: free propagator, decisive success.** Same `local_field`
encoder, propagator swapped to plain `mlp` (fully global, unconstrained
— H-PROP's own established winner). Genuine, benchmark-matching chaos:
`D_KY=23.75` (target `21-24`), `n_positive=13` (target `11+-2`),
`lambda1=0.112`. Textbook error-growth curve (smooth rise, saturates
cleanly at `sqrt(2)`, oscillates there indefinitely). Most importantly:
**D3 (propagator-Jacobian bandedness) came out significant (`p=0.0000`)
on a propagator with zero architectural locality forced anywhere** — the
first time in this whole project that a fully free propagator showed
genuine, non-tautological dynamical locality, plausibly because the
*encoder* now gives the data itself a locally-organized structure for an
unconstrained model to discover on its own. Weaknesses: DA skill only
`1.84x`/calibration `0.230` (below this project's best `L=100` models,
`3.1-4.2x`/`0.3-0.5`), and the latent covariance was severely
ill-conditioned (`cond#=1.4e16`, `d_latent=96` — more than 2x the
classical embedding-theory target of `2*D_KY~=44` for this attractor).

**Section 137: chasing the conditioning problem, at a real cost.**
Shrunk toward the embedding-theory target (`d_latent=48`, closest
achievable given `NX=256=2^8` forces `n_sites` to a power of 2) and
raised Section 98's regularizer weights (`w_logdet` `0.008->0.03`,
`w_spatial` `0.04->0.08`, `w_smooth` `0.003->0.01`). Result: genuinely
better on several raw axes (`lambda1=0.0955`, inside the literature
bound; DA skill `2.41x`, better than 136) — but conditioning did **not**
improve (`cond#=1.56e16`, unchanged in order of magnitude) and **D3 lost
significance** (`p=0.199`). A real, informative trade-off: better raw
accuracy is not the same axis as "how local/simple the learned dynamics
are," and pushing on regularizer weight alone bought one at a measurable
cost to the other.

**Section 138: does D3 significance actually predict distillability? Yes,
decisively.** Built `scripts/build_fresh_pdehead_checkpoint.py` (seeds a
fresh, untrained, **interpretable** `backbone="spectral_pde_raw"`,
`field_kind="polynomial"` pde_head — 22 parameters — without needing a
full Stage-1 job to produce one) and distilled it, frozen-target Phase-3
style (`train_stage2`'s own `--freeze-propagator` mode, zero risk to the
model being distilled), against both 136 and 137's finished propagators.
Rollout-horizon R² (pde_head vs. its own frozen target, `k=1,2,4,8`):

| k | 136 (D3 significant) | 137 (D3 not significant) |
|---|---|---|
| 1 | 0.988 | 0.946 |
| 2 | 0.978 | 0.873 |
| 4 | 0.956 | 0.682 |
| 8 | **0.920** | **0.367** |

A 22-parameter closure captures 92% of the variance in an 8-step rollout
of 136's genuinely chaotic 96-dimensional MLP propagator; the same tiny
model craters to 37% on 137, despite 137's own propagator being *more*
accurate against ground truth in isolation (`prop_vs_truth_mse=0.003` vs.
136's `0.027`). D3's significance is not noise — it is a real, predictive
signal about how local/distillable the learned dynamics are, independent
of the propagator's own raw accuracy.

**Sections 139-140: joint (not just distilled) training — mutual pressure
on the propagator, made safe by the encoder's own locality.** This
project already had two joint-training mechanisms, both pre-dating this
sub-arc: Stage 1's `--pde-distill` (pde_head trained alongside the real
propagator; detached target by default — the real propagator fully
protected) and a `--pde-mutual` flag whose own CLI help text, from an
earlier round of this exact question, already named the risk plainly:
"this is the same 'pressure toward simplicity' mechanism behind this
project's H-PROP fixed-point-collapse finding." Section 139 (Stage 1
detached, Stage 2 continuation unfrozen — unconditionally mutual by this
codebase's own design) did not collapse (D3 stayed significant, `D_KY`
stayed in-band) but was a mild step backward on every axis (`val_kmax_mse`
`0.082->0.186`, DA skill `1.84x->1.61x`, distilled R² at `k=8`
`0.920->0.836` — worse than 138's *frozen* distillation of the very
propagator it rode alongside). Section 140 went further still —
`--pde-mutual` in Stage 1 too, mutual pressure from epoch 0 through both
stages — and, contrary to the risk this mechanism carries everywhere
else in the project, came out **better on every measured axis**:

| | 136 (free) | 139 (Stage-2-only mutual) | 140 (fully mutual) |
|---|---|---|---|
| val_kmax_mse | 0.082 | 0.186 | **0.060** |
| `D_KY` / `lambda1` | 23.75 / 0.112 | 24.29 / 0.127 | 22.54 / **0.073** |
| DA skill / calibration | 1.84x / 0.230 | 1.61x / 0.132 | **2.08x / 0.241** |
| D3 | significant | significant | significant |
| distilled pde R² @ k=8 | — | 0.836 | **0.964** |

Best raw accuracy, best `lambda1` (comfortably inside the literature
bound), best DA skill and calibration of the whole sub-arc, *and* the
best-fitting distilled closure yet measured. The likely reason 139
degraded while 140 improved: 139 applied mutual pressure only after
Stage 1 had already converged to a solution optimized *without* it — a
shock to an already-settled optimum — whereas 140 let encoder,
propagator, and pde_head co-adapt together from the very first epoch,
finding a jointly-compatible solution rather than being pushed off one.
D1 (decoder sensitivity map) on 140 independently confirms the encoder's
own site structure directly, not just architecturally: centroids march
across the domain in tight triples matching `local_channels=3` exactly
(`1.26, 1.58, 1.79 | 4.46, 4.74, 4.92 | 7.53, 7.82, 8.05 | ... | 98.1,
98.4, 98.7` across all 32 sites).

**The extracted closures themselves are cross-run consistent — a strong
sign this is a real structure, not an optimization accident.** Both
138's from-scratch frozen distillation (of 136) and 140's fully mutual
joint training (a different encoder/propagator run entirely) arrived at
the *same qualitative closure shape*:

| term | 138 (distilled, frozen) | 140 (jointly trained) |
|---|---|---|
| `w_xxxx` | **-1.176** (dominant linear) | **-0.770** (dominant linear) |
| `w_xx` | -0.760 | -0.368 |
| `w_xxx` | +0.870 | +0.365 |
| `w_x` | +0.463 | +0.196 |
| dominant nonlinear term | `w_xxxx*w_xxxx` (-0.712) | `w_xx*w_xxxx` (-0.397) |
| `w*w_x` (the classical KS term) | -0.082 (small, correctly signed) | -0.006 (small, correctly signed) |

Both runs independently rediscover true KS's own linear dissipation
signs (`w_xxxx` and `w_xx` both negative, matching `-w_xx-w_xxxx`
exactly) and both find the classical advective nonlinearity `w*w_x`
present with the physically correct sign but small — the dominant
nonlinearity in both is instead high-derivative-order self/cross terms
(`w_xxxx^2`-like and `w_xx*w_xxxx`-like), consistently across two
separately-trained models. This is a genuine, reproducible local closure
of `local_field`'s own learned field, not the classical KS equation
restated, but not arbitrary either.

**Bottom line, directly updating §6.9's pessimism**: a literal, local,
interpretable closure of KS's latent dynamics is achievable — just not
as a jointly-trained-from-scratch PRIMARY propagator (that keeps failing,
now confirmed on a second encoder architecture too, for what looks like
a structural reason connected to H-PROP, not a tuning gap). It has to be
obtained either by distilling a free, already-chaotic propagator after
the fact (Section 138), or by co-training everything together with the
encoder itself supplying genuine architectural locality from the start
(Section 140, the best result of this whole document). The second is
better on every axis measured so far.

---

## 7. Open questions and untried directions

- **The local-latent-field program** (`docs/LATENT_PDE_RESEARCH_NOTES.md`
  §5, §6.1): a circular-Conv1d encoder/decoder with a shared local-stencil
  propagator, plus an explicit h-refinement continuum-limit test and an
  L-transfer test (train at one domain size, evaluate at a larger one with no
  retraining). This is the one approach in the project's history designed
  from the start to sidestep the locality-vs-global-mixing tension rather
  than run into it, and it has never been built.
- **`spectral_physics_prior`**: built, verified at zero-init, never run in a
  real experiment. Genuinely open, not negative.
- **`cnn`/`node` backbones with a full (non-local) kernel width**: proposed
  (`docs/LATENT_PDE_EXPERIMENTS.md`) as a systematic locality sweep, never
  executed. Would be a third, independent, translation-equivariant-by-
  construction test of H-PROP.
- **`fno_mlp` propagator** (FNO + plain per-token MLP, zero attention):
  would isolate whether attention was ever load-bearing in `fno_vit`'s own
  H-PROP success — cheap, backbone already exists, never run through Gate 3/4.
- **Section 106, re-run to completion**: interrupted before Gate 3; its
  early covariance-collapse signs (participation ratio 6/48, `cond#≈3e16`)
  make its true outcome genuinely unclear rather than negative.
- **Section 16 (`nodeltacap_e2e`)**, the best raw-accuracy model in the
  entire architecture search, was never re-run through the D4/smoothness
  tooling built later (Sections 71-88) — unknown whether it is also smooth
  and well-conditioned or just accurate.
- **A clean, single-variable `w_smooth` sweep**, isolated from simultaneous
  `w_var`/`w_spatial`/`lambda_z` changes — every prior test confounded it
  with 2-3 other changes at once.
- **The propagator-capacity confound** (flagged repeatedly, never isolated):
  train a *larger* propagator on the same frozen AE that currently
  underperforms at larger `d_latent`, to separate `d_latent` itself from
  propagator capacity as the cause of regression.

---

## 8. How to verify anything in this document

```bash
cd /Users/daltonjones/Documents/latent_DA

# Gate 3/4 results for any section:
grep -E "val_kmax_mse|D_KY|n_positive|skill_free_over_da" \
  artifacts/logs/gate3_analysis_section<N>_*.log

# Ground-truth D_KY benchmarks (both domain lengths, reproducible):
cat docs/REPLICATION_LOG.md

# Re-run the smoothness/conditioning comparison:
mamba run -n da_env python scripts/analyze_latent_smoothness.py

# Confirm/deny collapse for any run (never trust val_kmax_mse alone):
mamba run -n da_env python scripts/visualize_rollout.py --tag <section_tag>
```

Primary sources for this document: `docs/model-research-summary-9-5-26.md`
(Sections 1-88, read in full and used as the structural backbone for §3-5),
`docs/sine_transform_pde_plan.md` (1732 lines, read in full — §6.1-6.4),
`docs/LATENT_PDE_RESEARCH_NOTES.md` and `docs/ML_for_KS_writeup.md` (read in
full — §2, §6.1), `docs/LATENT_PDE_EXPERIMENTS.md` (read in full — §6.2,
§7), `docs/PROJECT_HANDOFF.md`/`docs/OPEN_QUESTIONS.md`/`docs/REPLICATION_LOG.md`
(read in full — §1 benchmark correction, §2), plus this session's own
first-hand work on Sections 111-120 (§6.5-6.9), direct verification of
Section 106's interrupted status, Section 105's actual (previously
undocumented) NaN-at-epoch-0 attempt and its stiffness root cause, and
Section 120's pre-training rollout verification, all against the actual
logs, checkpoints, and scripts rather than any secondhand summary.
