# Spectral-Latent PDE Discovery: Analysis and Execution Plan

**Status: PROPOSAL — not yet implemented.** Written 2026-09-06 in response to a
user request to analyze a new architectural idea before building anything.
Nothing in this document has been coded; see "Staged execution plan" for what
would happen on approval.

**DECIDED 2026-09-06 (user-confirmed): use rFFT/irFFT, not a literal sine
transform** — see §2.1 for the reasoning (a pure sine transform doesn't
close under odd-order derivatives, which KS's nonlinear term needs). This
also settles §7 decision 1. Implementation proceeds on this basis; the rest
of this document's analysis is written in terms of rFFT throughout.

**IMPLEMENTED 2026-09-06** (Stages 1-2 of §8's execution plan; Stages 0/3/4
— data generation at a reduced `L`, and actually training/evaluating the
resulting system — not yet run):
- `ks_latent/models/spectral_field.py`: the fixed rFFT transform +
  `(i*k)^n` derivative-synthesis helpers. Unit-tested against a known
  analytic sine/cosine field's hand-computed derivatives up to order 4
  (exact to float32 precision) and round-trip invariants.
- `ks_latent/models/autoencoder_spectral_field.py` /
  `SpectralFieldAutoencoderConfig` (`encoder_kind="spectral_field"`):
  wraps `KSAutoencoderViT` (`pool` in `("none", "local")`) with the fixed
  transform. Registered in `ks_latent.models`' `build_autoencoder`/
  `load_autoencoder_checkpoint` dispatch.
- `ks_latent/models/propagator.py`'s `_SpectralPDEDeltaBody`
  (`PropagatorConfig(backbone="spectral_pde")`): synthesizes the exact
  derivative stack, runs a shared pointwise MLP for `w_t`, integrates via
  one of THREE options (`spectral_integrator`) — see §5a for the full
  comparison against the real solver's own method:
  - `"euler"`/`"rk4"`: generic explicit integration of the ENTIRE
    right-hand side (reusing `"node"`'s RK4 loop structure). Verified
    exactly identity-at-init for physically-valid `z`.
  - `"etdrk4"` (added 2026-09-06, user-directed: "how closely does this
    method mirror the actual method we use to integrate the KS system"):
    mirrors `ks_latent/solver/ks.py`'s own ETDRK4 solver directly, reusing
    its `etdrk4_coefficients` function — KS's true linear symbol
    `Lhat(k)=k^2-k^4` is fixed (not learned) and integrated EXACTLY via
    `exp(dt*Lhat)`; only the nonlinear residual is learned. Verified: at
    `zero_init`, reduces EXACTLY to `exp(Lhat)*z` (the linearized-KS map),
    and this holds identically regardless of `ode_substeps` (an `exp(a)`
    composition identity, checked directly, not assumed).
- `ks_latent.training.losses.low_pass_spectral_loss`
  (`Stage1TrainingConfig.w_lowpass`/`.lowpass_power`), wired into
  `train_stage1`.
- CLI: `train_stage1_patched.py --encoder spectral_field --spectral-K
  --spectral-L --w-lowpass --lowpass-power`; `train_stage2_patched.py
  --backbone spectral_pde --spectral-K/-N-w/-L/-max-order/-integrator
  {euler,rk4,etdrk4}` (K/N_w/L default to the paired checkpoint's own
  values when not given).
- 50 new unit tests (transform correctness, identity-at-init/exact-linear-
  reduction, gradient flow, config validation) — full suite at 554 passed
  (up from 504). Real end-to-end CLI smoke tests (`--profile full
  --epochs 4`, canonical `L=100` dataset, all three integrators) ran
  cleanly through both stages.

Not yet done: generating a reduced-`L` dataset, screening its ground-truth
`D_KY`, or training/evaluating this architecture for real (§8 Stages 0, 3,
4) — this implementation pass built and tested the capability; no
experiment has been run with it yet.

## 0. TL;DR verdict

The core idea — encode into a spatially-organized field, transform it into a
truncated spectral representation, synthesize **analytically exact** spatial
derivatives from that representation, and let a small pointwise MLP learn the
map `(w, w_x, w_xx, ..., w^(n)) -> w_t` — is sound, has strong precedent
(closest analog: PDE-Net / PDE-Net 2.0, Long et al. 2018/2019), and is *more*
principled than this session's earlier SINDy attempt for reasons explained
below. It is also, almost eerily, close to what this codebase's own KS
solver already does (`ks_latent/solver/ks.py`'s ETDRK4 integrator: Fourier
space, diagonal derivative operators, pseudo-spectral nonlinear term) — this
project would essentially be learning the KS solver's own right-hand side
from data, in the same functional form the true equation is written in.

One part of the literal spec needs to change for the math to close: a
**pure sine transform doesn't work** for odd-order derivatives (explained in
§2.1). The fix is to use the **same real-FFT (rFFT) representation this
codebase already uses everywhere else** (`FourierIFFTBody`, `SpectralConv1d`,
the solver itself) instead of a literal sine transform. Everything else in
the proposal survives this substitution unchanged — "sine transform" and
"truncated rFFT" play the exact same conceptual role (fixed, invertible,
frequency-ordered basis); rFFT is just the version that actually supports
all the derivative orders requested.

The most serious open risk isn't mathematical, it's empirical: the resulting
propagator, however good its inputs, is still a **spatially local, weight-shared,
pointwise map** in the same architectural family as `node`/`local_mlp`/`masked_mlp`
— every one of which has collapsed to a fixed point (`D_KY=0`) in this
project's own prior experiments (H-PROP, and this session's Section 100,
which finished mid-write-up of this doc — see §4.3). This proposal is best
understood as a sharper, more principled test of *why* that keeps happening,
not a guaranteed escape from it. §4 discusses this directly.

## 1. The idea, restated precisely

1. A **global** encoder/decoder pair `E`/`D` maps the full physical state
   `x` (length `NX`) to/from an intermediate field `w` (length `N_w`,
   spatially indexed — i.e. `w` is a genuine local field, not a globally
   pooled vector). "Global" here refers to `E`/`D`'s own receptive field
   (full attention, no windowing) — the locality constraint in this design
   lives entirely in the *propagator*, not the autoencoder.
2. `w` is mapped to the latent `z` by a **fixed, non-learned, invertible
   (up to truncation) spectral transform** — literally the coefficients of
   `w` in some orthogonal basis (sine, in the original proposal).
3. Because `z` are coefficients in a known basis, spatial derivatives
   `w, w_x, w_xx, ..., w^(n)` (n ~ 5-6) can be computed **analytically** at
   every grid point of `w` by a diagonal operation on `z` followed by
   synthesis back to physical space — not estimated by finite differences
   or learned convolution filters.
4. A small MLP, applied identically (shared weights, translation-equivariant)
   at every spatial point, maps that point's local derivative stack to
   `w_t` (and optionally `w_tt`) at that point.
5. This MLP **is** the implicit PDE: `w_t = f_theta(w, w_x, w_xx, ..., w^(n))`
   pointwise, no explicit symbolic form imposed, but exactly the functional
   *shape* a real local PDE has.
6. Integrate forward (Euler or better), decode `w(t+1) -> x_hat(t+1)`.

Two proposed experimental conditions, in the user's message:
- Encoder/decoder: Section 98's plain-ViT architecture, but with
  `attn_window` effectively removed (set to 44, which is >= half of
  Section 98's `n_tokens=32`, i.e. full/global attention) — confirming
  point 1 above (global AE, local propagator only).
- Drop `w_spatial`/`w_var` entirely; add a new regularizer penalizing
  high-frequency content of `z` proportional to frequency (a low-pass
  penalty).
- Use a smaller domain length `L` and `dt_snap=0.2` (vs. the canonical
  `L=100`/`dt_snap=1.0`) for the first validation runs.

## 2. Mathematical analysis

### 2.1 Sine transform vs. Fourier transform — the one required change

A pure discrete sine transform (DST) represents `w` as `sum_k c_k
sin(k*pi*x/L)`. Differentiating:

- `d/dx sin(k*pi*x/L) = (k*pi/L) cos(k*pi*x/L)` — **leaves the sine basis**.
- `d^2/dx^2 sin(k*pi*x/L) = -(k*pi/L)^2 sin(k*pi*x/L)` — **stays** in the sine
  basis (diagonal eigenvalue, since `sin(k*pi*x/L)` is an eigenfunction of
  `d^2/dx^2` under Dirichlet BCs).

So **even-order** derivatives (`w`, `w_xx`, `w_xxxx`) are diagonal, exact,
and stay purely in terms of the sine coefficients `z` — this part of the
proposal works exactly as described. But **odd-order** derivatives (`w_x`,
`w_xxx`, `w_xxxxx`) map sine coefficients to an entirely different (cosine)
coefficient set that a pure-sine `z` does not contain. Since KS's actual
nonlinear term is `u*u_x` (needs the *first* derivative specifically), and
the user's proposal explicitly wants derivatives "up to 5 or 6," a pure
sine-only `z` cannot supply about half the requested feature set from `z`
alone.

The fix that is both mathematically clean and requires zero new
infrastructure: use the **real FFT** (`torch.fft.rfft`/`irfft`) instead of a
pure sine transform. In the complex-exponential (Fourier) basis, `d/dx`
is diagonal at **every** order: `d^n/dx^n e^{ikx} = (ik)^n e^{ikx}`. One
multiplier array `(ik)^n`, applied to the same `z`, gives every derivative
order with no basis-switching and no missing odd orders. This also:

- **Matches the real boundary condition.** KS is periodic
  (`x in [0, L)`); a DST assumes `w(0) = w(L) = 0` (Dirichlet), which is not
  actually true of the field being represented, forcing the encoder to
  waste representational capacity forcing artificial zero endpoints or
  learn to work around a distorted basis. rFFT's implicit periodicity is
  the *correct* assumption here (it's what the solver itself already
  assumes).
- **Reuses tested code.** `ks_latent/models/propagator.py`'s
  `FourierIFFTBody` already implements "run an MLP on `rfft` features,
  predict output-mode coefficients, `irfft` back to physical space" —
  the derivative-multiplier step is a small, well-scoped addition to an
  existing, unit-tested pattern (multiply by `(ik)^n` before the `irfft`,
  instead of learning the output coefficients directly).
- **Matches the project's own solver exactly.** `ks_latent/solver/ks.py`'s
  ETDRK4 integrator already represents KS in Fourier space with wavenumbers
  `k = 2*pi*n/L` and a diagonal linear operator `Lhat(k) = k^2 - k^4`
  (`u_xx` and `u_xxxx`'s Fourier symbols) plus a pseudo-spectral nonlinear
  term computed by transform -> pointwise square -> transform back — i.e.
  **this proposal, in Fourier form, is structurally the same computational
  pattern the ground-truth solver already uses**, just with a *learned*
  pointwise combiner MLP standing in for the hand-derived `-0.5*(u^2)_x -
  u_xx - u_xxxx` formula. This is the strongest point in favor of the whole
  approach: if it works, the learned MLP is directly, term-for-term
  comparable to a known closed-form answer, using the exact same feature
  basis the true solver computes in.

**Recommendation: implement this with rFFT, not a literal sine transform.**
If there's a specific reason for wanting Dirichlet boundary behavior
(none is stated in the request), say so and we can revisit — but I don't see
one, and rFFT dominates on every axis (correctness, code reuse, and direct
interpretability against the known solver).

### 2.2 Truncation, `d_latent`, and what "low-pass regularizer" should mean

`z` should be a **truncated** rFFT spectrum (keep the lowest `K` modes,
`K < N_w/2+1`) for the same reason `FourierIFFTBody.out_modes`/Section 89's
`enc_out_modes` truncate today: dimensionality reduction, and a *hard*,
structural smoothness guarantee (dropped modes are exactly, not
approximately, absent). The proposed low-pass regularizer then acts as a
*soft* penalty on top of that hard truncation, pushing energy toward the
very lowest few of the kept modes — this is not redundant with truncation,
it's a second, finer-grained lever (truncation sets a hard ceiling; the
regularizer shapes the energy distribution below it).

"Penalizes ... proportional to their frequency" is most literally
`L_lowpass = w_lowpass * sum_k |k| * |z_k|^2` (linear-in-`k` weighting). The
more classical choice in the numerical-PDE/regularization literature is
`sum_k k^2 * |z_k|^2` (an `H^1` Sobolev seminorm — literally "the energy of
the gradient," since by Parseval this quantity equals `integral (dw/dx)^2
dx`). I'd expose the exponent as a hyperparameter (`w_lowpass * sum_k
k^p * |z_k|^2`) and try both `p=1` (as literally requested) and `p=2`
(classical Sobolev) rather than assume the former is obviously right.

Worth noting explicitly: this mechanism is mathematically the *same
operation* as the **dealiasing / hyperviscosity filters** that real
pseudo-spectral PDE codes apply to suppress spurious high-wavenumber energy
generated by pointwise nonlinear products (see §2.3) — so this regularizer
is doing double duty, serving both the "smooth enough to fit a PDE to"
goal and a classical numerical-stability role.

### 2.3 A numerical subtlety worth tracking: aliasing

The proposal computes derivative *fields* by synthesizing back to `N_w`
physical grid points, then applies a nonlinear (MLP) pointwise operation —
exactly the classical "pseudo-spectral" pattern the solver itself uses for
`u*u_x`. Classical spectral codes must guard against **aliasing**: a
pointwise product of two band-limited (`<=K` modes) signals produces energy
up to `2K`, which folds back (aliases) into the kept `<=K` modes as
corruption unless the physical grid has enough points to resolve `2K`
(the standard "2/3 rule": keep `<=N_w/3` modes, or oversample). Concretely:
if `z` keeps `K` modes and the physical synthesis grid also has only `~K`
points, the MLP's implicit nonlinear combination will alias.

This is a real numerical consideration, but likely **not** a blocking one
for a first attempt, for a reason a hand-derived spectral solver doesn't
have: this whole system is trained end-to-end against real trajectory
data, so the optimizer can (and likely will) partially absorb aliasing
error into the learned encoder/decoder/MLP rather than it being a fatal,
uncontrolled instability the way it would be in a fixed hand-coded
scheme. Recommendation: keep `N_w` comfortably larger than `K` (e.g.
`N_w >= 3K`, matching the classical 2/3 rule) as a cheap precaution, and add
a diagnostic (energy in modes `> K` of the *predicted* `w_t` field, before
any re-truncation) to check whether aliasing is actually showing up in
practice, rather than assuming either way.

### 2.4 Is the pointwise MLP expressive enough?

The true KS RHS is `w_t = -0.5*(w^2)_x - w_xx - w_xxxx`, i.e. a *linear*
combination of `w_xx`, `w_xxxx`, plus one genuinely nonlinear term
(quadratic in `w`, via `w*w_x` after expanding `(w^2)_x = 2*w*w_x`). A plain
MLP with smooth activations is a universal approximator and can represent
this in principle, but may be sample-inefficient at discovering a bilinear
product from scratch. Worth trying as an ablation (not a blocker for the
first attempt): augment the MLP's raw derivative-stack input with a few
explicit low-order product features (e.g. `w*w_x`, `w^2`), in the same
spirit SINDy-style libraries already use — gives the optimizer a head
start on the one term that isn't already linear-diagonal in the derivative
features.

## 3. Precedent in the literature

- **PDE-Net (Long, Lu, Ma, Dong, 2018) and PDE-Net 2.0 (Long et al., 2019).**
  The closest direct analog. Learns constrained convolutional filters that
  approximate spatial derivative operators up to a given order, then feeds
  the resulting derivative stack at each point through a shared nonlinear
  "response function" (a small NN, "SymNet" in the 2.0 version) to predict
  `u_t` pointwise — structurally identical to what's proposed here, with
  the difference that this proposal computes derivatives *exactly* via a
  spectral transform instead of *approximately* via constrained finite
  convolution kernels. That's a meaningful, well-motivated upgrade: spectral
  differentiation has no discretization error for band-limited functions,
  whereas a fixed-stencil convolution has bounded polynomial order of
  accuracy.
- **SINDy for PDEs / PDE-FIND (Rudy, Brunton, Proctor, Kutz, 2017).** Builds
  a fixed library of candidate derivative/polynomial terms at each point and
  fits a *linear*, sparse combination reproducing `u_t`. This session
  already tried a close cousin of this on Section 98's (globally pooled,
  non-spatial) latent and it failed cleanly (R^2~0.005, fitted map collapses
  to `D_KY=0`) — see §4.2 for why this new proposal is a substantively
  different, more principled attempt, not a repeat of that failure.
- **DeepMoD (Both, Vermarien, Kusters, 2021).** Uses a coordinate-based
  network (PINN-style) plus automatic differentiation to get exact
  derivatives, then sparse regression to discover the governing PDE. Same
  "get exact derivatives, then fit the RHS" pattern, different derivative
  mechanism (autodiff on a continuous ansatz vs. spectral synthesis here).
- **Classical pseudo-spectral time-stepping for stiff PDEs, specifically
  Kassam & Trefethen's ETDRK4 (2005)** — this is not just adjacent
  precedent, it is **literally the method `ks_latent/solver/ks.py` already
  implements** to generate every dataset this project uses. That solver
  already represents the true KS state in Fourier space, computes `u_xx`/
  `u_xxxx` as diagonal multipliers (`Lhat(k) = k^2 - k^4`), and computes the
  nonlinear term pseudo-spectrally (transform down, square pointwise,
  transform back up, `g(k) = -ik/2`). The proposal in this document is, in
  essence, asking whether the same computational *pattern* the true solver
  uses can have its one nonlinear, non-diagonal step (`g * FFT(IFFT(v)^2)`)
  replaced by a learned pointwise MLP acting on a richer derivative-feature
  stack, operating on a *learned* intermediate field `w` instead of the raw
  state `u`. That's a clean, well-posed research question with a
  ready-made ground truth to compare against.
- **Fourier Neural Operator (Li et al., 2020) — already used elsewhere in
  this codebase (`backbone="fno_vit"`/`"fno_mlp"`, `use_fno`).** Also
  operates in Fourier space with a learned per-mode linear operator, but
  does *not* expose explicit derivative features or aim for symbolic
  interpretability — it's a strong black-box operator learner. Worth
  naming explicitly so this proposal isn't mistaken for "FNO again": the
  novel/valuable part here is the *explicit derivative featurization*
  (interpretability — after training, the pointwise MLP is directly
  readable as an approximate PDE) and the *exactness* of the spectral
  derivative computation, not raw representational power (which FNO
  already provides and this project has already tried in several forms).
- **Sobolev-norm / spectral-bias regularization** (widely used across
  operator learning and PINN literature to bias networks toward smooth
  solutions; e.g. Sobolev training, and the FNO literature's own use of
  `H^1`-type loss terms) — direct precedent for the proposed low-pass
  regularizer, and for reading it as an `H^1`/`H^{1/2}` Sobolev seminorm
  penalty depending on the chosen exponent (§2.2).
- **"Neural spectral methods"** is an active, if newer, research direction
  (networks that operate natively in a spectral basis with differentiation
  as multiplication, in the spirit of classical spectral PDE solvers) —
  flagged as a category worth a literature search before/during
  implementation rather than a specific citation I'm confident enough in to
  name precisely here.

## 4. Relationship to this project's own findings

### 4.1 The H-PROP tension (must be addressed head-on, not glossed over)

This project's own accumulated finding (H-PROP, `docs/model-research-summary-9-5-26.md`):
a propagator recovers chaos **if and only if** it has a global (full
`d_latent`) receptive field *and* unconstrained mixing; every local-reach
architecture tried (`local_mlp`, `masked_mlp`, `node`) has collapsed to a
fixed point regardless of the specific local mechanism. The proposed
pointwise derivative-MLP propagator is, in terms of raw information flow,
**also a spatially local, weight-shared, pointwise map** — each output
point's value depends only on that point's own (locally-supported, if
`K` is modest) derivative stack, nothing else. Per H-PROP's stated pattern,
a fresh collapse to `D_KY=0` (or a severe conditioning blowup, as just
observed in Section 100 — see §4.3) is a live, real possibility, not a
remote one.

The counter-argument, and the reason this is still worth trying rather than
dismissing on H-PROP grounds alone: **the true KS PDE is itself exactly this
kind of local, pointwise map** (`u_t` at a point depends only on `u` and its
spatial derivatives *at that point*) — chaos in the real system is an
emergent, collective property of many identical local rules coupled through
the field's own spatial structure, not something that requires a nonlocal
right-hand side. So H-PROP's pattern so far doesn't prove local propagators
*can't* produce chaos; it may instead mean every local architecture tried
so far was never actually constrained to compute anything resembling a
genuine derivative (an arbitrary small conv kernel or per-position masked
linear map has no reason to discover the specific differential structure
that makes KS chaotic, and training may simply have found it easier to
settle on a safe contractive/fixed-point solution instead). Feeding the
network numerically **exact** derivatives, in the same feature basis the
real solver uses, is arguably the most direct test available of whether
"locality" itself was ever really the obstruction, or whether it was
"locality without the right features."

**This should be stated as an explicit, first-class hypothesis of the
experiment, with a clear failure mode anticipated in advance** — not
discovered as a surprise after another collapse.

### 4.2 Why this differs from today's failed SINDy attempt

`scripts/fit_latent_pde.py` (built and tested against Section 98 earlier
today) failed cleanly: `R^2~0.005`, fitted map collapses to `D_KY=0`. Three
concrete reasons this new proposal is a materially different attempt, not
a repeat:

1. **The feature library was built on the wrong object.** SINDy's
   circular-shift stencil operated on Section 98's `z`, whose "adjacent
   index" ordering has no guaranteed physical meaning (the propagator
   module's own docstrings flag this explicitly — it's an *assumed*
   ordering, correct only once a genuinely spatially-organized latent
   exists). This proposal deliberately constructs a real spatial field `w`
   (via `pool="none"`/`"local"`) *before* transforming, so "neighboring"
   and "derivative" have their literal, intended meaning.
2. **Linear vs. nonlinear combiner.** SINDy fits a small *linear* sparse
   combination over a fixed 8-term library. This proposal uses a general
   MLP — closer to PDE-Net 2.0's SymNet than classical SINDy.
3. **Post-hoc vs. end-to-end.** SINDy was fit *after* the fact on an
   already-trained, frozen checkpoint's induced `z`-dynamics — a
   representation optimized for a completely different (unconstrained)
   propagator, not for being easy to fit a local PDE to. This proposal
   trains the whole encoder-transform-derivative-MLP-decoder pipeline
   jointly from scratch, so the encoder is directly pressured (via the
   rollout loss) to produce a `w`/`z` that this specific constrained
   propagator can actually work with.

### 4.3 Section 100 (finished during the writing of this document)

Independently, Section 100 (launched just before this request, testing a
different way of architecturally enforcing propagator locality — a
windowed masked-linear "single wide hidden layer") finished mid-write-up:
`cond#=7.9e15` (most latent directions collapsed to numerically zero
variance), `val_kmax_mse=0.28` (vs. Section 98's 0.033) — a clear collapse,
worse than Section 99's. This is the same H-PROP pattern again, on yet
another local mechanism (masked-linear rather than conv or the sine/Fourier
approach proposed here). It's additional evidence for the risk in §4.1, but
for the reason given there (Section 100's masked-linear layer has no
particular reason to have discovered anything resembling a real
derivative operator), it doesn't pre-empt this new proposal — if anything
it sharpens the case that *what* local information the propagator is given
matters, which is exactly this proposal's bet.

## 5. Proposed architecture (modular breakdown)

Two new, independent pieces, both maximizing reuse of existing tested code:

1. **A thin spectral-transform wrapper**, e.g.
   `ks_latent.models.autoencoder_spectral_field.SpectralFieldAutoencoder`:
   wraps an existing `KSAutoencoderViT` configured with `pool="none"` or
   `"local"` (already gives a genuine per-site scalar/small-vector field —
   no new AE code needed for the `x <-> w` half) and adds `encode(x) =
   rfft_truncate(vit.encode(x))`, `decode(z) = vit.decode(irfft_pad(z))`.
   The transform itself is fixed (no learned parameters) — literally
   `torch.fft.rfft`/`irfft` plus a mode-count truncation, following
   `FourierIFFTBody`'s existing pattern for the complex-to-real handling.
   **Note**: Section 98's own `pool="token_mlp"` is *not* compatible with
   this design (it discards spatial position entirely via its flatten
   step) — the encoder/decoder must use `pool="none"`/`"local"` instead,
   which is a real (small) deviation from "use the ViT encoder/decoder
   from 98 unchanged" worth flagging back to the user.
2. **A new propagator backbone**, e.g. `backbone="spectral_pde"`:
   given `z`, synthesize `w, w_x, ..., w^(n)` via `(ik)^m` multipliers +
   `irfft` (generalizing `FourierIFFTBody`'s existing complex-coefficient
   handling); stack the `n+1` fields into a `(B, N_w, n+1)` tensor; run a
   small shared-weight pointwise MLP (same "no positional encoding, same
   weights at every site" design already used by `TokenMLPBlock`/
   `backbone="fno_mlp"`) to get `w_t` (and optionally `w_tt`); integrate
   forward (start with explicit Euler at `dt=0.2`, matching the empirical
   finite-difference target directly; consider reusing
   `_NeuralODEDeltaBody`'s existing fixed-step RK4 loop as a refinement
   once Euler is validated, since KS is genuinely stiff and RK4 is already
   implemented and tested in this codebase for exactly this kind of local
   vector field).
3. **New loss term**: `L_lowpass = w_lowpass * sum_k k^p * |z_k|^2`
   (`p` a hyperparameter, try 1 and 2), replacing `w_spatial`/`w_var`
   entirely per the user's request — and worth noting explicitly *why*
   dropping those makes sense here, not just "because asked": `w_var`/
   `w_spatial` exist to *induce* decorrelated, spatially organized latent
   channels in an otherwise unconstrained latent; a truncated orthogonal
   spectral basis already **has** decorrelated, ordered channels by
   construction, so those regularizers' whole purpose is structurally
   satisfied for free. `w_logdet` (the fix for per-channel collapse from
   `docs`/memory) addresses a *different* failure mode and is worth keeping
   available, at least as an option to fall back on if the new architecture
   shows the same per-channel collapse this project has hit before.

## 5a. The three `spectral_integrator` options, and how closely each mirrors the real solver

**Added 2026-09-06** in response to the user asking how closely this
design mirrors `ks_latent/solver/ks.py`'s own ETDRK4 solver, and whether
splitting off a "linear operator" makes the system linear (it does not —
see the Q&A at the end of this section). All three are implemented in
`_SpectralPDEDeltaBody`, selectable via `PropagatorConfig.spectral_integrator`
/ `train_stage2_patched.py --spectral-integrator`, so they can be compared
directly.

**`"euler"`** (the original proposal) and **`"rk4"`**: both hand `field`
(the pointwise MLP) the ENTIRE right-hand side, including KS's own stiff
linear diffusion term (`-u_xx - u_xxxx`, eigenvalues `~-k^4`), and step it
with a generic explicit integrator. This is a real mismatch with the true
solver: the true solver never uses an explicit scheme on this term
specifically *because* it's stiff (a naive explicit step needs `dt` small
relative to `1/k_max^4` to stay stable — the entire motivation for
"exponential time differencing" in the first place, see Kassam & Trefethen
2005). These two options are the "generic ODE integrator, let the network
figure everything out" baseline.

**`"etdrk4"`** (added 2026-09-06, this is the "mirror the real methodology
as closely as possible" option): splits the dynamics the same way the true
PDE does, `z_t = Lhat(k)*z + N(z)`:
- `Lhat(k) = k^2 - k^4` — KS's own true, exactly-known linear symbol,
  computed for the `spectral_K` kept modes and **fixed, not learned** (a
  real, zero-parameter physical prior, not a black box guess).
- Integrated **exactly** via the integrating factor `exp(dt*Lhat)` —
  unconditionally stable for this term regardless of step size, removing
  the stiffness problem entirely rather than working around it.
- `N(z)` (the genuinely nonlinear residual) is evaluated at 4 RK-like
  stages and combined via `E/E2/Q/f1/f2/f3` — the SAME coefficients
  `ks_latent.solver.ks.etdrk4_coefficients` (Kassam & Trefethen's
  contour-integral computation) produces, reused directly rather than
  reimplemented, then combined via literally the same formula
  `ks_latent.solver.ks.step` uses. `N(z)` itself is computed the same
  pseudo-spectral way the real solver computes its own nonlinear term:
  synthesize the exact derivative stack in physical space (`field`'s
  input), evaluate the pointwise MLP there, transform the result back to
  frequency space (`encode_to_spectrum`) — matching the real solver's own
  `_nonlinear` (physical-space squaring, back to frequency space) step for
  step, with the hand-coded formula replaced by the learned one.

The only thing genuinely different from the real solver is *which*
function computes `N(z)` — a hand-derived closed form there, a trained
pointwise MLP here. Everything else (the wavenumber convention, the linear
symbol, the exact exponential integration, the RK combination coefficients,
the pseudo-spectral evaluation pattern for the nonlinear term) is the same
computation, reused directly rather than re-derived.

**Consequence for `zero_init`**: for `"euler"`/`"rk4"`, zeroing the MLP's
output gives an exact identity map (`z_next = z`), matching every other
propagator backbone's init convention. For `"etdrk4"`, zeroing the MLP
gives `N(z)=0` identically, so the map reduces to `z_next = exp(Lhat)*z` —
the EXACT linearized-KS map (real growth below the `k=1/sqrt(2)`
instability threshold, real decay above it), not a no-op. This is a
deliberate, arguably *better* starting point: training begins from an
architecture that already correctly implements linearized KS, with only
the nonlinear correction left to learn, rather than from a trivial
fixed-point map with everything left to learn. Verified directly (not
merely asserted) in `tests/unit/test_propagator_spectral_pde.py`: the
reduction to `exp(Lhat)*z` holds exactly regardless of how many
`ode_substeps` are used (since `exp(a)` composed `n` times equals
`exp(n*a)` for a diagonal exponent — a real, checkable numerical
invariant, not just a design intention).

**Q&A: does splitting off a linear operator make the system linear?** No.
KS is `u_t = -u*u_x - u_xx - u_xxxx`. `Lhat(k)=k^2-k^4` only ever
represents the `-u_xx - u_xxxx` piece. The `-u*u_x` term — genuinely
nonlinear, built from `u^2` in physical space — is never folded into
`Lhat`; it is `N(z)`, computed separately and pseudo-spectrally, and stays
exactly as nonlinear as it ever was. The full system remains
`z_t = Lhat*z + N(z)` with `N` nonlinear in `z`. What's linear is only the
isolated sub-problem "hold `N` fixed for an instant and integrate the
linear part" — a standard operator-splitting technique (exponential time
differencing): solve exactly the piece you can solve exactly, and treat
explicitly only the piece you can't. It does not linearize KS; it removes
an already-solved sub-problem (and the stiffness that sub-problem would
otherwise impose on an explicit scheme) from what the network has to
discover.

## 6. Data prerequisites

Fully supported by existing code, no new solver work needed:
`ks_latent.solver.dataset.generate_trajectory_dataset` takes an arbitrary
`KSConfig(L, NX, dt, snapshot_every, spinup_time, seed)`. The existing
canonical dataset uses `dt=0.05, snapshot_every=20` (`dt_snap=1.0`); the
requested `dt_snap=0.2` is `snapshot_every=4` at the same internal solver
step (`dt=0.05`) — a trivial parameter change, not a new solver capability.

Two things worth doing *before* committing to a specific reduced `L`
(both cheap, both reusing `ks_latent.analysis.lyapunov.lyapunov_spectrum_ks`,
which already computes a ground-truth Lyapunov spectrum directly from the
raw PDE solver, independent of any latent model):
- **Screen a few candidate `L` values** (e.g. 22, 36, 50 — small-`L` KS is a
  well-studied low-dimensional-chaos regime in the classical literature,
  though I'd rather verify the resulting `D_KY` directly with this
  project's own tool than rely on a possibly-misremembered exact citation)
  and pick one with a modest, clearly nonzero ground-truth `D_KY` (e.g.
  somewhere in 2-6) — informative enough to be a real test of chaos
  recovery, small enough to be a fast, cheap validation loop before
  scaling to the canonical `L=100`/`D_KY~22`.
- Decide whether `NX` should shrink proportionally with `L` (to preserve
  the same tokens-per-characteristic-wavelength ratio `ViTAutoencoderConfig`
  already reasons about) or stay fixed — recommend shrinking proportionally,
  simplest way to keep every existing sizing convention meaningful.

## 7. Open design decisions (my recommendation, for confirmation)

1. **Sine transform -> rFFT.** Recommended, see §2.1. (Requires user
   sign-off since it's a literal change to the spec, even though I think
   it's clearly correct.)
2. **Truncation count `K`** for `z`: a new hyperparameter, larger than the
   project's usual `d_latent~44` compression target probably needed to give
   the derivative-synthesis step enough resolution — needs a direct
   parameter sizing pass once `N_w` is chosen (§2.3's aliasing-margin
   constraint, `N_w >= 3K`, is the binding relationship).
3. **Low-pass exponent `p`**: try both 1 (literal ask) and 2 (classical
   Sobolev `H^1`).
4. **1st- vs. 2nd-order-in-time**: recommend starting 1st-order only
   (`w_t` alone) — true KS is exactly first-order in time, matching this
   project's own existing `mode="markovian"` rationale; add `w_tt` only as
   a follow-up ablation if 1st-order underperforms.
5. **Euler vs. RK4 vs. ETDRK4 integration** at `dt=0.2`: all three are now
   implemented (§5a) and directly comparable via `spectral_integrator`.
   Given the stiffness argument in §5a, I'd actually recommend `"etdrk4"`
   as the FIRST one to try in Stage 3 (§8), not a fallback if the others
   show instability — it mirrors the real solver most closely and removes
   the stiff linear term from what the network has to discover at all.
   `"euler"`/`"rk4"` remain useful baselines to quantify how much that
   distinction actually matters empirically.
6. **Explicit product features** (`w*w_x`, `w^2`) alongside the raw
   derivative stack: worth an ablation given §2.4, not required for a
   first attempt.
7. **`w_logdet`**: recommend keeping available (not necessarily on) as a
   guard against the per-channel-collapse failure mode this project has
   hit before, distinct from what the low-pass regularizer addresses.
8. **Reduced `L`**: pick empirically via the screening in §6, not from
   memory of the literature.

## 8. Staged execution plan (only on approval)

Each stage has a cheap go/no-go check before investing in the next.

**Stage 0 — data & ground truth (cheap, ~minutes of solver time).**
Generate a small trajectory dataset at a screened reduced `L`, `dt_snap=0.2`.
Compute its ground-truth `D_KY` via `lyapunov_spectrum_ks` directly on the
raw solver (no model involved yet) — this is the number every later stage's
result gets compared against.

**Stage 1 — spectral derivative module, unit-tested in isolation.**
Implement the `(ik)^n` multiplier + `irfft` derivative synthesis. Validate
against a *known* analytic test field (e.g. a truncated sine/cosine
combination with hand-computable derivatives) to machine precision, fully
independent of any trained model — a strong, cheap correctness gate before
anything is trained.

**Stage 2 — autoencoder only (no propagator).**
Train the `SpectralFieldAutoencoder` (encode/transform/truncate/
inverse-transform/decode) on reconstruction alone, `pool="none"`/`"local"`,
global attention. Check reconstruction quality and the actual energy
spectrum of `z` (is energy really concentrated at low `k`, with or without
`L_lowpass` on yet?) before ever touching the propagator.

**Stage 3 — propagator, small-`L` regime.**
Train the derivative-MLP propagator end-to-end (Stage-1-style joint
training, then Stage-2-style rollout fine-tuning, mirroring this project's
existing two-stage convention) on the reduced-`L` dataset. Compare
`val_kmax_mse`/reconstruction against a same-size `mlp`/markovian baseline
propagator on the *same* reduced-`L` data (a fair, apples-to-apples
comparison this project hasn't had for any prior local-propagator attempt —
every past comparison was implicitly against the canonical `L=100` `mlp`
baseline's numbers). Compute the trained system's own `D_KY` (reusing
`lyapunov_spectrum_latent_propagator`) and compare directly against Stage
0's ground truth.

**Go/no-go after Stage 3**: if `D_KY` is meaningfully nonzero and in the
right ballpark relative to Stage 0's ground truth, this is a genuinely
positive, interesting result — proceed to inspect the learned pointwise
MLP directly (does it look anything like the true KS RHS, e.g. does it
respond roughly linearly to `w_xx`/`w_xxxx` and pick up a `w*w_x`-like
sensitivity?) and consider scaling `L` toward the canonical 100. If it
collapses (`D_KY~0`, or a conditioning blowup like Sections 99/100), that's
still a real, informative result for H-PROP (§4.1) — worth understanding
*why* (is the propagator itself contractive, or did the encoder collapse
`z`'s variance first?) before deciding whether to iterate on this design or
treat H-PROP's locality finding as more fully confirmed.

**Stage 4 — only if Stage 3 succeeds: scale to canonical `L=100`.**

## 9. Risk register (summary)

| Risk | Where discussed | Mitigation |
|---|---|---|
| Pure sine transform doesn't close under odd derivatives | §2.1 | Use rFFT instead |
| H-PROP: local propagators have collapsed every time so far | §4.1, §4.3 | Treat as an explicit, anticipated hypothesis; compare against a ground-truth `D_KY` at the same (reduced) `L`, not just "did it collapse" |
| Aliasing from pointwise nonlinear combination of truncated spectrum | §2.3 | Keep `N_w >= 3K`; add an energy-in-dropped-modes diagnostic |
| MLP may be sample-inefficient at discovering the quadratic `w*w_x` term | §2.4 | Ablation: explicit product features |
| New dataset generation (reduced `L`, `dt_snap=0.2`) | §6 | Fully supported by existing `generate_trajectory_dataset`/`KSConfig`, no new solver code |
| `pool="token_mlp"` (Section 98's actual pool mode) is incompatible with this design | §5 | Use `pool="none"`/`"local"` instead — flagged as a deviation from the literal spec |

## 10. Success criteria

A genuinely positive result: `D_KY` on the reduced-`L` system stays
meaningfully above 0 and in the right range relative to the raw solver's
own ground-truth spectrum (Stage 0), at rollout accuracy at least in the
neighborhood of a same-size unconstrained `mlp`/markovian baseline trained
on the *same* reduced-`L` data — plus, ideally, a trained pointwise MLP that
is at least loosely recognizable as an approximation to the true KS RHS
when inspected directly (e.g. via partial-derivative sensitivity analysis
of the trained MLP itself). Anything short of the `D_KY` criterion is still
a real, informative negative result for the project's H-PROP finding, not
a wasted experiment — the specific way it fails (contraction vs. variance
collapse vs. aliasing-driven noise) would itself be useful information.

## 11. Section 101 execution log (2026-09-07): running Stages 0-3 for real

User-directed: "so now we have two new options for training the pde
propagator, correct? can we try both and see what we can learn. please do
this, document all your choices and run as many diagnostic as possible to
determine what we can improve." "Two new options" = `spectral_pde` with
`spectral_integrator="rk4"` (the original proposal, generic explicit
integration) vs. `spectral_integrator="etdrk4"` (§5a, mirrors the real
solver's own integration scheme). Launched via
`scripts/section101_spectral_pde_L22.sh`; every choice below is also
documented inline in that script's header.

**Design: one shared AE, two propagators.** Train ONE `spectral_field`
autoencoder (Stage 1), then train TWO Stage-2 `spectral_pde` propagators
on top of its frozen `z` — one per integrator. This is the correct
controlled comparison: identical `z` representation, identical MLP
capacity/curriculum, differing *only* in integration scheme — isolating
exactly the variable this experiment is about.

**Choice: `L=22`, not the canonical `L=100`.** Reuses this repo's own
already-validated Gate 1 benchmark
(`tests/replication/test_gate1_kaplan_yorke.py::test_L22_lyapunov`):
ground-truth `D_KY` in `[5.2, 5.6]`, `lambda_1` in `[0.043, 0.05]` — a
real, low-dimensional-but-genuinely-chaotic regime with a precisely known
answer already established in this codebase, so §6's suggestion to screen
candidate `L` values ourselves turned out to be unnecessary. `NX` kept at
256 (this project's `--profile full` default, not Gate 1's own 128) —
purely more over-resolved, never a correctness issue.

**Choice: `dt_snap=0.2`** (`snapshot_every=4` at the solver's `dt=0.05`),
per the user's original request. `trajectory_time=50.0` (not the canonical
250.0) chosen to give the same per-run snapshot count (250) as every
canonical dataset in this project at the finer spacing — a scoping choice
to keep dataset size comparable to precedent, not a claim that 50 physical
time units is generous (the L=22 system's own decorrelation time is
`~1/lambda_1~=22`, so one run spans only ~2.3 such times; `n_train=50`
independent, separately-spun-up runs is relied on for diversity rather
than within-run length). Dataset:
`artifacts/datasets/stage1_trajectories_L22_dtsnap02.h5` (generated in
9.1s).

**Choice: `K=8` kept rFFT modes, `N_w=32` (`patch_size=8`).** At `L=22`,
`K=8` keeps wavenumbers up to `k=2*pi*7/22~=2.0`, well beyond KS's own
instability range (`0<k<1`, peak growth at `k=1/sqrt(2)~=0.71`) — several
spare modes beyond the handful of genuinely unstable/energetic ones this
low-dimensional system has. `N_w=32` is comfortably above `3*K=24`, the
§2.3 dealiasing margin.

**Choice: `w_var=0`, `w_spatial=0`, `w_decorr=0`, `w_logdet=0`.** The first
three per §5, point 3 (a truncated orthogonal rFFT basis is already
decorrelated/organized by construction — nothing left for these to
usefully induce). `w_logdet=0` is an extension beyond the user's literal
request, reasoned through this pass: `w_logdet` exists to fight
*pathological* per-channel variance collapse, but a real KS field's
amplitude spectrum genuinely, physically decays with wavenumber — higher
kept modes SHOULD have smaller variance than low ones, and `w_logdet`
would fight that correct structure rather than an actual failure.
Verified via the covariance-spectrum diagnostic (script step 2), not
merely assumed safe.

**Choice: `w_lowpass=0.003`, `lowpass_power=1.0`.** Verified by direct
computation on a fresh, untrained `SpectralFieldAutoencoderConfig(K=8,
L=22.0)` and 256 real states from this dataset: at init, `l_recon=1.190`,
`l_lowpass(power=1)=54.045` (ratio ~45.4x). `w_lowpass=0.003` gives a
weighted contribution of ~0.162 (~14% of `l_recon`'s own magnitude at
init) — gentle, not dominant, matching this project's established
regularizer-calibration practice. `power=1.0` matches the user's literal
"proportional to frequency" request (`power=2`, the classical `H^1`
Sobolev seminorm, remains a documented follow-up ablation, not run here).

**Choice: propagator sizing `hidden=128`, `n_blocks=3`, IDENTICAL for both
integrators** — matches this project's long-established `mlp`/markovian
convention, and keeps capacity from being a confound between the two runs.

**Choice (REVISED after direct timing): `k_max=12`/`k_mid=8`
(canonical, not the originally-planned 24), `ode_substeps=1` (not 4),
`epochs=80` (not 300), Stage 2 runs sequential not parallel.** Directly
timed before committing: `k_max=4`/`ode_substeps=4` came back at ~60s/epoch;
even `k_max=12`/`ode_substeps=1` (run in PARALLEL, contending for one MPS
device) still cost ~75s/epoch — the originally-planned
`k_max=24`/`ode_substeps=4`/300-epoch run projected to ~30 HOURS per
propagator, clearly infeasible. Revised:
- `ode_substeps=1` for BOTH — not just a compute cut: ETDRK4's whole
  selling point (§5a) is that it does NOT need fine sub-stepping for
  stability, unlike explicit RK4. Comparing both at the SAME
  `ode_substeps=1` is exactly the comparison that matters — if plain
  `"rk4"` struggles at this coarseness while `"etdrk4"` doesn't, that IS
  the result, not a confound to fix.
- `k_max=12`/`k_mid=8`, reverting to canonical rather than the
  unaffordable `k_max=24` — the "shorter fraction of the system's own
  Lyapunov time than canonical `dt_snap=1.0` gets" caveat from the
  original plan (§7, point 5) STANDS (`12*0.2=2.4` physical time units,
  ~0.11 Lyapunov times here) — an accepted limitation of this pass.
- `epochs=80` (`k_warmup_epochs=56`, `k_mid_epochs=46`, same ~70%/58%
  ratios as canonical) — justified empirically: 2-epoch timing runs
  already showed `val_kmax_mse` dropping from ~0.015-0.017 to
  ~0.002-0.004, i.e. a genuinely fast-converging landscape on this
  simpler, low-dimensional system, not merely truncated early.
- Stage 2 runs sequential — a fairer wall-clock comparison (no mutual MPS
  contention), and the direct reason the timing check itself came back
  slower than hoped.

**Fixed along the way: two Gate 3/4 scripts hardcoded `L=100.0`.**
`run_da_pff.py` and `run_diagnostics.py` both hardcoded
`KSConfig(L=100.0, ...)` in their real (`--ae-checkpoint`-driven) setup
paths, regardless of the checkpoint's actual training `L` — silently
wrong for this experiment (would have compared an `L=22`-trained
propagator's DA skill/diagnostics against ground truth generated at
`L=100`). Both now accept `--L` (default 100.0, unchanged behavior for
every prior run using the canonical domain). `run_analysis_suite.py`
needed no patch (never constructs its own `KSConfig` in the real path,
only reads `--dataset`/`--points-dataset` directly) — it just needed an
`L=22` `--points-dataset`, generated once, since it has no
auto-generation fallback at all (raises `FileNotFoundError` otherwise):
`artifacts/datasets/attractor_points_L22.h5`.

**Status**: launched as one background job (Stage 1 -> spectrum check ->
Stage 2 rk4 -> Stage 2 etdrk4 -> Gate 3/4 on both, sequential). Results to
be appended here once complete.

**Result (rk4, frozen-encoder design): `D_KY=0.0`, `n_positive=0`,
`lambda_1=-0.649`.** A genuine collapse to a fixed point — despite
excellent short-horizon rollout accuracy (`val_kmax_mse=0.00035`) and
clean D1-D9 diagnostics (D3 not significant/dense coupling, D4 clean
translation equivariance). This is exactly the H-PROP pattern again:
good short-term numbers, zero actual chaos. Also caught and fixed two
real bugs surfaced by this run's D1/D7 diagnostics on the always-zero
DC-imaginary channel — see the code-fix note below.

**Result (etdrk4, frozen-encoder design): `D_KY=0.0`, `n_positive=0`,
`lambda_1≈-2e-7` (numerically zero, not clearly negative like rk4's
-0.649).** Also a collapse, but a qualitatively different one — rk4
converged to a clearly CONTRACTING fixed point, while etdrk4 converged to
an almost perfectly NEUTRAL/marginal map (lambda_1 essentially exactly 0
to numerical precision, not negative). Both give zero chaos, but the
*way* they fail differs, which is itself informative. DA skill
(`skill_free_over_da=2.85`) is real but more modest than rk4's (6.76),
consistent with etdrk4's near-neutral (rather than strongly contracting)
free-running dynamics not diverging as fast even without assimilation.
D3/D7/D8 not significant (dense coupling); D4 clean translation
equivariance, same as rk4.

**Fixed along the way: two diagnostics crashed on a genuinely dead latent
channel.** `decoder_sensitivity_map` (D1) raised on an all-zero sensitivity
weight vector; `same_time_coupling_diagnostic`'s Fiedler-permutation
eigendecomposition (D7) then failed to converge on a NaN-poisoned
correlation matrix (`np.corrcoef` divides by a zero-variance channel's
stddev). Both are generic gaps, not specific to this architecture — any
encoder with a collapsed/dead channel could hit them. Fixed generically:
`circular_weighted_stats` now returns a well-defined degenerate result
(`resultant_length=0`, `centroid=nan`, `spread=inf`) instead of raising on
zero total weight; `same_time_channel_correlation`/`_signed` now replace
`nan` correlation entries with `0` before any downstream eigendecomposition.
Tests added for both.

## 12. Section 102 (2026-09-07): correcting a real design flaw — joint Phase-1 training

User-directed, after seeing Section 101's `rk4` result: "I don't think
this is a good idea. I want the encoder and decoder pair to be trained
specifically to encode data that can be transformed accurately by the
rk4/etdrk4/euler method. therefore each of those should be present in
phase 1 and then we can use the same network in phase 2 as we extend the
rollout."

This is a real, valid critique of Section 101's design, not just a
preference: there, the encoder/decoder were trained against a cheap,
generic `mlp`/markovian propagator (used only for Stage 1's own `L_pred`
regularizer) and then FROZEN before any of the three real `spectral_pde`
propagators ever touched it — the encoder had no pressure to produce a
`z` actually well-suited to spectral-derivative-based propagation
specifically. Section 101's `D_KY=0` result is confounded by this: we
don't know how much of the collapse is intrinsic to the architecture
versus an artifact of training the representation in isolation from the
propagator that would consume it.

**New capability built to fix this**: `AuxPropagatorConfig` (Stage 1's own
propagator config) now supports `backbone="spectral_pde"` directly —
previously this backbone was Stage-2 (`PropagatorConfig`) only, matching
`"node"`/`"cnn"`'s existing precedent of being full-propagator-only. Added
every field `PropagatorConfig` already had for this backbone
(`spectral_K`/`spectral_N_w`/`spectral_L`/`spectral_max_order`/
`spectral_integrator`), plus a new `ode_substeps` field
`AuxPropagatorConfig` never needed before this. `aux_cfg_to_propagator_cfg`
forwards all of them. `train_stage1_patched.py` gained
`--aux-backbone spectral_pde` plus
`--spectral-max-order`/`--spectral-integrator`/`--ode-substeps` CLI flags.
Verified via a real smoke run before launching: Stage 1 now jointly trains
the AE with a REAL, full-sized (`--full-propagator`, `hidden=128`/
`n_blocks=3` — matching Phase 2's own sizing exactly, so warm-starting is
a like-for-like continuation, not a resize) `spectral_pde` propagator, and
Stage 2 correctly warm-starts from that checkpoint via
`--init-prop-checkpoint` to extend the rollout further.

**Design**: three full pipelines (one per integrator: `rk4`, `etdrk4`,
`euler`), each Phase 1 (joint AE + real propagator, Stage 1's own short
`k_pred=2` `L_pred` rollout — a cheap regularizing signal, not the main
training) followed by Phase 2 (warm-start, ramp `k_pred=2 -> k_max=12`
over 80 epochs — identical curriculum to Section 101, for direct
comparability). Same `L=22` dataset, same `K=8`/`N_w=32`/`L=22.0`, same
regularizers as Section 101 — ONLY the joint-vs-frozen-encoder question
differs between the two sections, isolating exactly that variable.
Launched via `scripts/section102_joint_spectral_pde_L22.sh`.

**Status**: launched as one background job covering all three integrators
sequentially. Section 101's `etdrk4` Gate 3/4 (frozen-encoder design) was
left running in parallel rather than killed — its result remains a useful
comparison point for exactly this joint-vs-frozen question. Results to be
appended here once complete.

**Result (etdrk4, frozen-encoder design, added for completeness): `D_KY=0.0`,
`n_positive=0`, `lambda_1≈-2e-7`** (numerically zero/marginal, not clearly
contracting like rk4's -0.649 — a qualitatively different kind of
collapse). DA skill `skill_free_over_da=2.85` (real but more modest than
rk4's 6.76). D3/D7/D8 not significant; D4 clean translation equivariance.

**Result (euler, frozen-encoder design): `D_KY=0.0`, `n_positive=0`,
`lambda_1=-0.455`** (contracting, like rk4, though less strongly).
`skill_free_over_da=8.17` (the highest of the three). D3/D7/D8 not
significant; D4 clean translation equivariance; D9 `prop_jacobian_med
=0.945`.

**Full Section 101 summary (all three, frozen-encoder design) — every
propagator collapsed to a fixed point, but via visibly different
mechanisms:**

| | rk4 | etdrk4 | euler |
|---|---|---|---|
| `D_KY` | 0.0 | 0.0 | 0.0 |
| `lambda_1` | -0.649 | ≈-2e-7 (marginal) | -0.455 |
| `skill_free_over_da` | 6.76 | 2.85 | 8.17 |

This is the complete frozen-encoder baseline. Section 12/13's joint-
training redesign (and now Section 14's larger-`K` design) are the direct
follow-ups testing whether either fix changes this outcome.

## 13. Queued next (2026-09-07): stop hard-truncating `K`, let `w_lowpass` do the real work

User-directed, questioning the `K=8` choice directly: "why are we
truncating at 8? that seems arbitrary? why not use the whole spectrum we
can but also have a low pass filter, that was the whole reason for the
filter." Correct and sharper than my own framing up to this point.

**Why `K=8` wasn't literally arbitrary, but defeated the filter's purpose
anyway**: it was sized to respect the classical spectral dealiasing margin
(Orszag's 3/2 rule: avoiding nonlinear-aliasing corruption of the kept
modes needs roughly `N_w >= 3*K`; `32 >= 3*8=24` held with some room). But
`w_lowpass` was built specifically so the model could decide how much
spectrum it needs, rather than us guessing a cutoff up front — hard-
truncating to a value close to our own a priori guess about the system's
dimension makes the soft mechanism redundant, which is exactly what got
flagged two exchanges earlier (§11's "is the low-pass filter unnecessary
at these smaller dimensions" discussion) and is the same root cause.

**Follow-up Q&A this turn: does having an exact/closed-form derivative
expression eliminate the aliasing concern, so the dealiasing margin can
just be dropped?** Partially, not entirely, worth being precise: the
derivatives themselves are exact regardless of `K`/`N_w` -- no argument
there, and that was never what the margin was protecting. The margin
protects the NONLINEAR MLP step specifically (`field`, computing `w_t`
from the derivative stack) -- like any nonlinear operation, it can produce
output frequency content above what the input had, and representing that
at only `N_w` points can fold high content back into the kept low modes if
`N_w` is too small relative to `K`. Exact derivatives don't touch this;
it's a separate, downstream step.

That said, the classical 3/2 rule is derived for a FIXED, hand-coded
nonlinearity in a NON-ADAPTIVE numerical scheme, where aliasing error is
silent and uncorrectable. That isn't this system: (a) `w_lowpass` already
discourages the model from needing sharp high-k corrections in the first
place, indirectly bounding how much aliasing-prone content the MLP has
reason to produce; (b) real aliasing corruption during TRAINING would
show up as measurable instability/worse loss, not silently; (c) the whole
system can adapt around a soft violation of the margin in a way a fixed
classical solver cannot. Conclusion: treat the classical rule as a
starting heuristic, not a hard a priori design constraint -- pick a large
`K` at a more modest `N_w` increase than "double everything to keep the
3/2 margin exactly," and ADD A DIRECT DIAGNOSTIC (energy in the discarded
modes `K..N_w/2` of the MLP's raw output, each step, before truncation)
to check empirically whether aliasing is actually showing up, rather than
assuming either way.

**Concrete next design** (queued, to launch once Section 102's `rk4`
Phase 1 + Phase 2 finish, so it doesn't contend with that run for GPU):
- Grow `N_w` and `K` together, `K` close to the practical maximum the
  (relaxed) margin allows rather than a guess based on assumed system
  dimension -- e.g. `N_w=64`, `K~20` (covering wavenumbers out to
  `k≈6`, far beyond anything physically relevant for this `L=22` system's
  own instability/dissipation range) -- exact values to be finalized at
  launch time via direct computation, same convention as every other
  sizing decision this project makes.
- Recalibrate `w_lowpass`'s weight at the new `K` (the raw penalty
  magnitude scales with mode count, so the old 0.003 calibrated for
  `K=8` will not transfer directly -- redo the init-time magnitude check
  against `l_recon`, same method as before).
- Add the aliasing-energy diagnostic described above (new, not yet
  implemented) to monitor whether the relaxed margin is actually a
  problem in practice.
- Use the JOINT Phase-1-then-Phase-2 design from Section 12 (not
  Section 101's frozen-encoder one) from the start, now that it's built
  and verified.

**Two more changes added by the user, same queued experiment:**

1. **`dt_snap` back to `1.0`** (the project's own canonical value, not
   Section 101/102's `0.2`) — "to discourage collapse to a single point."
   A real, well-motivated hypothesis for *why* Sections 101/102 collapsed:
   at `dt_snap=0.2`, consecutive snapshots are already very close together,
   so a lazy near-identity/contracting map already achieves low prediction
   error — training has little pressure to do anything else. At
   `dt_snap=1.0`, consecutive snapshots differ meaningfully, so a trivial
   contracting solution incurs real, visible loss, giving genuine training
   pressure against exactly the collapse mode observed. Requires a NEW
   `L=22` dataset at `dt_snap=1.0` (`snapshot_every=20` at the solver's own
   `dt=0.05` — this is now literally the project's canonical
   `snapshot_every`, no special-casing needed; `trajectory_time=250.0`,
   the canonical value, giving the canonical 250 snapshots/run).
2. **Sub-step the integrator within each now-larger `dt_snap=1.0` interval**
   ("we might consider integrating rk4 multiple times in that time step")
   — directly maps to the EXISTING `ode_substeps` field (already built,
   already used by `"node"` and `"rk4"`/`"etdrk4"`'s own sub-stepping
   loop). At `dt_snap=0.2`, `ode_substeps=1` gave a per-substep size of
   `0.2`; matching that same per-substep fineness at `dt_snap=1.0` means
   `ode_substeps=5` (`1.0/5=0.2`) — a reasonable default, applied
   identically to `rk4`/`etdrk4`/`euler` for a fair comparison (`euler`
   itself still ignores `ode_substeps`, always a single full step, per
   its own design — see `_SpectralPDEDeltaBody`'s docstring; this remains
   a known, deliberate asymmetry, not something this change fixes).
   **Compute cost warning**: this is a substantial increase, not a free
   change — `ode_substeps=5` means 5x the `field()` (MLP) evaluations per
   rollout step versus Section 102's `ode_substeps=1`, on top of `k_max=12`
   now covering the full canonical `dt_snap=1.0*12=12` physical time units
   (vs. `2.4` before) rather than a shrunk one — genuinely the RIGHT thing
   to test, but plausibly several hours per propagator at Section 102's
   per-epoch timings. Flagging this now rather than discovering it
   mid-run.
3. **`w_pred=1.5`** in Stage 1 (`Stage1TrainingConfig.w_pred`, default
   `0.5`) — directly reinforces the Section 12 joint-training fix's own
   purpose: weight the `L_pred` term (the one that pressures the encoder
   to co-adapt with the real propagator) more heavily relative to pure
   reconstruction, now that Stage 1 is actually training the real
   propagator instead of a throwaway generic one. Already a plain CLI
   flag (`--w-pred`), no new capability needed.

## 14. Section 103 (2026-09-07): launched — K=15, dt_snap=1.0, ode_substeps=3, w_pred=1.5

User: "kill the rk4 training and move on to the experiment we discussed."
Section 102's mid-flight `rk4` run (K=8/dt_snap=0.2/ode_substeps=1) was
killed (Phase 1 had already finished and its checkpoint remains on disk,
superseded but harmless). Combined all changes discussed above into one
launch: `K=15` (up from 8, `N_w=32` unchanged — `d_latent=30`),
`dt_snap=1.0` (new dataset generated, `snapshot_every=20`,
`trajectory_time=250.0`, both literally this project's own canonical
values), `ode_substeps=3` for `rk4`/`etdrk4` (a user-directed compute-cost
trim from the originally-discussed 5, via this session's own cost
estimate showing ~12-15 hours for all three integrators at
`ode_substeps=5`), `w_pred=1.5` in Phase 1. `w_lowpass` recalibrated at
`K=15` (was tuned for `K=8`): verified by direct computation on the new
`dt_snap=1.0` dataset, `l_recon=1.471`, `l_lowpass(power=1)=30.289` at
init (ratio 20.6x) — `w_lowpass=0.007` gives the same ~14%-of-`l_recon`
target fraction Sections 101/102 used. Order: `euler`, `rk4`, `etdrk4`
(user-directed, cheapest first — `euler` ignores `ode_substeps` entirely,
unaffected by that change).

Uses Section 12's joint Phase-1/Phase-2 design throughout (not Section
101's frozen-encoder one). Verified via a real smoke run (4-epoch Phase 1
+ 2-epoch Phase 2, all four changes combined, `euler` integrator) before
the real launch — both phases ran cleanly. Launched via
`scripts/section103_wideK_dtsnap1_L22.sh`.

**On the dealiasing margin specifically** (`K=15` at `N_w=32` violates the
classical `N_w>=3K` rule by a wide margin, `45 > 32`): deliberately not
respected here, per the §13 discussion — treated as an empirical question
to watch via existing diagnostics (e.g. the covariance-spectrum step
already run after Phase 1) rather than a hard constraint. The dedicated
energy-in-discarded-modes diagnostic discussed in §13 was NOT built this
pass — flagged as a follow-up if the covariance spectrum or training
behavior suggests it's actually needed.

**Status**: launched as one background job, all three integrators
sequential. Mid-flight, a real Euler-integrator bug was found and fixed
(see §14a) — the `euler` leg was killed and restarted cleanly with the
fix; `rk4`/`etdrk4` were unaffected (they already sub-stepped correctly).

**Result (`euler`, corrected Section 103 design): still `D_KY=0.0`,
`n_positive=0` — but meaningfully less collapsed than before:**

| | Section 101 `euler` (frozen encoder, K=8, dt=0.2) | Section 103 `euler` (joint, K=15, dt=1.0) |
|---|---|---|
| `D_KY` | 0.0 | 0.0 |
| `lambda_1` | -0.455 | **-0.030** (~15x closer to neutral) |
| `skill_free_over_da` | 8.17 | **1.90** (free-running nearly matches DA-assisted) |
| D3 (Jacobian bandedness) | not significant | **significant, p=0.003** (new — structured coupling appeared) |
| D4 (translation equivariance) | clean | **lost** ("no clean linear representation") |
| D9 `prop_jacobian_med` | 0.945 | **1.19** (median now slightly expansive) |

None of the fixes (joint training, `K=15`, `dt_snap=1.0`, `w_pred=1.5`,
sub-stepped Euler) flipped this into genuine chaos, but together they
moved the propagator substantially toward neutral/marginally-expansive
rather than strongly contracting — real, measurable progress, just not
over the `D_KY>0` line yet. D3 turning significant and D4's equivariance
breaking are new wrinkles worth tracking in `rk4`/`etdrk4`'s results too,
not yet understood as good or bad.

## 14a. Euler sub-stepping fix (2026-09-07)

User: "wait so euler is just the super basic one step euler step? couldn't
we also integrate euler over multiple steps?" `_SpectralPDEDeltaBody`'s
`"euler"` branch was hardcoded to always take exactly one full
`dt_snap`-sized step, completely ignoring `ode_substeps` — unlike `"rk4"`/
`"etdrk4"`, which already sub-step correctly. Fixed: `"euler"` now loops
`ode_substeps` times with `h=1/ode_substeps` each, re-encoding between
sub-steps exactly like `"rk4"` does (`ode_substeps=1`, the default, is
unchanged — exactly the original single-step formula, so every prior
`euler` result before this fix remains valid on its own terms).

Follow-up user question, resolved by direct computation rather than
assumption: "how do we recalculate spatial derivatives [between
sub-steps]? ... which is exactly the same derivatives we started with[?]"
Checked directly (small test propagator): substep 1's `z` is exactly
`z` at substep 0 (an exact `decode`-then-`encode` round trip), so the
FIRST substep genuinely does start from the same derivatives — but
subsequent substeps are NOT frozen (max abs diff between substep 0 and 1's
`z` was ~3.7, not zero). What carries forward is only the portion of the
pointwise MLP's response that lands within the kept `K` modes — ~12% of
its output energy sits OUTSIDE the kept modes at each substep and is
discarded on re-encoding. Also checked (correcting an earlier hedge): `rk4`
and `etdrk4` have the SAME truncation-at-every-evaluation property, and
actually MORE of it (4 stage evaluations per sub-step vs. Euler's 1) — this
is a property of the whole architecture's re-encoding pattern, not
something specific to Euler.

4 new tests added (identity-at-init across multiple substep counts, exact
match to the pre-fix single-step formula at `ode_substeps=1`, confirmation
sub-stepping changes the result rather than being a no-op, gradient flow).
Full suite: 559 passed.

## 15. `spectral_shape_floor_loss` (2026-09-07): a collapse regularizer calibrated from real data

User: "I really just want to come up with a regularizer to prevent the
latent state from collapsing," after `w_var`/`w_logdet`/`w_decorr` (and,
checked directly and rejected, `w_logdet` applied to `decode_from_spectrum(z)`
— mathematically equivalent to applying it on `z` directly plus a harmful
rank-deficiency artifact, since `decode_from_spectrum` is a pure LINEAR
map and can't manufacture new degrees of freedom, verified directly:
`cov(w)`'s two smallest eigenvalues were `~1e-7`, exactly matching
`N_w-2K=2`) all turned out to fight the ALREADY-CORRECT natural spectral
energy decay of a real KS field, having no principled way to distinguish
"healthy decay" from "pathological collapse."

**Design: a one-sided floor calibrated from the real training data's own
spectrum**, not a hand-picked absolute threshold. `reference_mode_energy(u,
K)` computes the TRUE field's own empirical per-mode energy proportions
among its lowest `K` modes (u's own `rfft` mode `m` sits at the exact same
physical wavenumber `k=2*pi*m/L` regardless of `u`'s grid resolution, so
this needs no rescaling to compare against `z`'s own modes) — a physically-
grounded reference for what a non-collapsed latent's per-mode energy SHARE
should be. `spectral_shape_floor_loss(z, K, p_ref)` then penalizes `z`'s
own per-mode share falling BELOW that reference, but NEVER for exceeding
it.

**Resolving "where does the energy go" directly** (`u` has far more modes
than `z` keeps): the encoder plausibly needs to fold some of `u`'s
discarded higher-mode structure into the low modes it keeps, so `z`'s kept
modes may legitimately need MORE energy than the bare reference suggests,
not the same amount. An exact shape-match penalty would fight exactly that
necessary compensation. The one-sided floor resolves this directly: extra
"compensation" energy can land in any kept mode, above its own floor,
completely unpenalized by construction — only genuine collapse (falling
short of the physically-required minimum) is penalized.

Implemented in `ks_latent/training/losses.py`
(`reference_mode_energy`/`spectral_shape_floor_loss`), wired into
`Stage1TrainingConfig.w_shape_floor` and `train_stage1` (`p_ref` computed
ONCE from real training data at the start of training, not per-batch —
a fixed, non-learned target), CLI flag `--w-shape-floor` on
`train_stage1_patched.py`. 7 new tests (reference computation correctness
against a hand-computed single-mode signal, zero loss at exact match,
penalized below floor, NOT penalized when at-or-above floor everywhere,
gradient flow). Verified end-to-end via a real CLI smoke run
(`encoder_kind=spectral_field`, `--w-shape-floor 1.0`, 4 epochs) — trains
cleanly.

Magnitude calibrated on real `L=22`/`K=15` data at init: `l_recon=1.469`,
`l_shape_floor=37.730` (ratio 25.7x) — `w_shape_floor~=0.005-0.006` would
give the same ~14%-of-`l_recon` target fraction every other regularizer
this session used, matching the same calibration convention (not yet run
in a real experiment as of this note). The reference profile itself
(`p_ref`), computed on real `L=22` data, shows exactly the expected
physical shape: mode 0 (DC) `~0` (correct, zero-mean data), mode 2
dominant (`0.438` — its wavenumber `k=0.571` sits right at this system's
own peak linear-instability wavenumber `k=1/sqrt(2)~=0.707`), then a
steep, physically-sensible decay through mode 14.

## 16. Section 104 (2026-09-07): `w_shape_floor` live, `K=24`/`N_w=64`, `euler` only

User: "kill it. I want to try the euler mode with the
specreal_shape_floor_loss. set it at whatever value you think will make a
difference, and let's see if we can get something that doesn't collapse.
also increase the latent space dimension a bit and the number of modes
for z." Section 103's `rk4` leg was killed mid-Phase-2 (its own `euler`
result already in hand from §14, `D_KY=0` but `lambda_1=-0.030`, real
progress not yet over the line).

Two changes bundled together, as requested:
- **`N_w=64` (up from 32), `K=24` (up from 15)** — `d_latent=48`, close to
  this project's canonical 44. `NX=256` stays evenly divisible
  (`patch_size=4`), `K=24` comfortably under the `N_w=64` max of 33.
- **`w_shape_floor=0.1`, set AGGRESSIVELY, not the gentle ~14% convention.**
  User: "set it at whatever value you think will make a difference."
  Verified at init on this exact config: `l_recon=1.674`,
  `l_shape_floor=17.167` (ratio 10.3x) — `w_shape_floor=0.1` gives a
  contribution of ~1.72, roughly EQUAL to `l_recon`'s own magnitude — a
  deliberately bold weighting, matching the explicit intent to really test
  whether this mechanism can prevent collapse, not just nudge at it.

**`w_lowpass` recalibrated for the new `K`** (its raw magnitude scales with
mode count AND how high the kept wavenumbers reach — at `K=24` the highest
kept mode's wavenumber is `~6.6` vs `K=15`'s `~4.0`, so the sum grows much
faster than `K` alone suggests): at this config's init, the OLD
`w_lowpass=0.007` would have given ratio 115x `l_recon` — wildly dominant.
`w_lowpass=0.0012` restores the usual ~14% target.

Everything else unchanged from Section 103's `euler` leg (`dt_snap=1.0`,
`ode_substeps=3`, `w_pred=1.5`, `w_var=w_spatial=w_decorr=w_logdet=0`,
joint Phase-1/Phase-2 design, same `k_max=12`/epoch schedule). `euler`
only this pass. Verified via a real smoke run (4-epoch Phase 1 + 2-epoch
Phase 2) before launching — both phases ran cleanly at the new sizing with
`--w-shape-floor 0.1` active. Launched via
`scripts/section104_shapefloor_wideK_euler.sh`.

**Status**: launched as one background job. Results to be appended here
once complete.

**RESULT — the first genuine positive `D_KY` in this entire spectral_pde
investigation**: `D_KY=22.0007`, `n_positive=11`, `lambda_1=+0.079`
(POSITIVE). `val_kmax_mse=0.00244`. `skill_free_over_da=3.43`. D9
`prop_jacobian_med=1.74`/`p95=2.09` (genuinely, substantially expansive —
consistent with real chaos, unlike every prior run's near-1 or sub-1
values). D3 not significant (`p=0.173`, dense coupling). D4 lost clean
equivariance (same pattern as Section 103's `euler`).

Verified this is not a numerical artifact before treating it as real:
(a) `single_state` Lyapunov mode uses the FULL, uncapped `n_directions=
d_latent=48` (only the separate `two_step` mode hit its 20-direction cap
and failed, which is documented, expected behavior, not a red flag); (b)
directly rolled the trained propagator forward 300 steps from a real
initial condition — `z`'s norm and `u_hat`'s amplitude both stay bounded
throughout (comparable to or smaller than the real data's own amplitude
range), no NaN/Inf — genuine bounded dynamics, not divergence being
masked by the Benettin algorithm's `max_abs_state` safety clamp.

**Important caveat, found by cross-checking against an independent
estimate**: `D_KY=22` is suspiciously close to this project's *canonical
L=100* result, not this system's own L=22 ground truth (`D_KY∈[5.2,5.6]`,
this repo's own Gate 1 benchmark). Cross-checked against the SAME run's
topology-based dimension estimates on the real ENCODED data (not the
propagator's own free rollout): `two_nn_dimension=4.48`,
`correlation_dimension=3.10` — both reasonably close to the true ~5.4
ground truth. This is a real, informative discrepancy: the ENCODER
appears to be doing a reasonably faithful job (real data, once encoded,
occupies a genuinely low-dimensional manifold close to the true
attractor's own dimension) — but the PROPAGATOR's own free-running
dynamics explore a substantially higher-dimensional, more chaotic regime
than the real data's own manifold, while still staying bounded. Plausible
explanation: `w_shape_floor`'s one-sided design only prevents the ENCODER
from letting a mode's energy fall below the real-data floor; nothing
constrains the PROPAGATOR from exciting those same modes MORE than real
trajectories ever do, if that happens to help satisfy the (short-horizon,
`k_max=12`) rollout-accuracy objective it was actually trained on.

**Bottom line**: a real, significant milestone — the first genuine escape
from this whole project's H-PROP fixed-point-collapse pattern for this
architecture family, with corroborating evidence across multiple
independent diagnostics (D9's expansive Jacobian, D3's dense coupling,
direct bounded-rollout verification) — but the specific `D_KY=22` value
likely reflects the free-running propagator exploring "extra," not
necessarily physically faithful, degrees of freedom beyond the true
attractor's own ~5-dimensional structure, rather than an exact
reproduction of true KS chaos at `L=22`. Worth a direct visual check
(`visualize_rollout.py`/`make_latent_gif.py`) as a natural follow-up, not
yet done.

## 17. `spectral_physics_prior` (2026-09-08): bake in the exact true KS dynamics, learn only a correction

User: "what if we had another variant that assumes that the latent space
follows exactly the KS dynamics with some learned correction term (the
correction term should be some nonlinear pde in terms of the spatial
derivatives). That way we already have chaos in the formulation."

**Motivation, stated precisely**: `"etdrk4"` alone only bakes in the
LINEAR part of KS exactly (`Lhat=k^2-k^4` via `exp(dt*Lhat)`); the linear
part alone is not chaotic on its own (unstable modes there just grow
without bound — it's specifically the nonlinear term `-w*w_x` that folds
that instability into a bounded chaotic attractor). This option bakes in
the FULL exact right-hand side, so the pointwise MLP only has to learn a
CORRECTION on top of real physics, not the whole dynamics from scratch.

**Why this should actually work, not just sound appealing**: Lyapunov
exponents are invariant under a smooth, invertible change of coordinates.
If the learned intermediate field `w` is a reasonably faithful (close to
invertible — which good reconstruction already pressures the encoder
toward) reparametrization of the true physical state, baking in exact KS
dynamics for `w` should inherit KS's own genuine chaos essentially for
free, rather than requiring training to discover it.

**Implementation, integrator-aware to avoid double-counting**: `field()`
now optionally adds a FIXED baseline before the learned correction:
- `"etdrk4"`: only `-w*w_x` (the nonlinear term) — the linear part
  `-w_xx-w_xxxx` is already handled exactly, separately, via `exp(dt*Lhat)`;
  baking it into `field()` too would apply it twice.
- `"euler"`/`"rk4"`: the FULL `-w*w_x - w_xx - w_xxxx` (these integrators
  have no separate linear treatment).

Verified directly (not just asserted): at `zero_init` (the learned
correction MLP zeroed), `field()` matches the analytic KS RHS EXACTLY (0.0
max abs diff) for all three integrators, computed with the correct
integrator-specific formula. `zero_init=True` no longer means identity —
the propagator now implements genuine (nonzero) KS dynamics from the very
start of training, a deliberately different, more informative
initialization than every other option.

**Caveat, stated directly**: KS is not scale-invariant (rescaling
`u -> c*u` scales the nonlinear and linear terms differently), so this
baseline is most exact if `w`'s own learned amplitude roughly matches the
scale the equation was derived in. Nothing currently pins that exactly —
reconstruction loss gives indirect pressure toward a "physically
reasonable" scale, but the correction MLP may need to absorb some scale
mismatch, which it can do in principle (a general nonlinear function) but
less elegantly than if `w` were already well-scaled.

Implemented as `PropagatorConfig`/`AuxPropagatorConfig.spectral_physics_prior`
(mirrored on both so Phase 1 joint training can use it too, not just
Phase 2), `_SpectralPDEDeltaBody.field`'s new baked-in-baseline logic,
CLI flag `--spectral-physics-prior` on both `train_stage1_patched.py` and
`train_stage2_patched.py`, requires `spectral_max_order >= 4`. 8 new
tests (exact match to the analytic RHS at zero-init for all three
integrators via `pytest.mark.parametrize`, non-identity at init, gradient
flow through the correction, config validation, correction genuinely adds
on top of the prior). Verified end-to-end via a real CLI smoke run
(`--aux-backbone spectral_pde --spectral-integrator etdrk4
--spectral-physics-prior`, Phase 1 + Phase 2 warm-start) — trains cleanly,
config persists correctly through the saved checkpoint. Not yet run in a
real experiment as of this note — a natural next step once Section 104
finishes, most promising paired with `etdrk4` specifically (see the
`lambda_1` comparison across integrators in §14 for why: `etdrk4` already
landed closest to neutral under the worst prior conditions, so it has the
least distance left to cover).

## 18. Section 104's `D_KY=22` diagnosed: spurious high-mode energy, not genuine L=22 chaos

`scripts/visualize_rollout.py`'s Hovmöller triptychs (physical-space, always
raw `u_true` vs. decoded rollout; and — after the fix in §18a — latent-space,
now `w=decode_from_spectrum(z)` rather than raw `z`) showed Section 104's
free-running rollout populates persistent, narrow high-wavenumber streaks in
both spaces that the true L=22 attractor never has, while error growth
saturates cleanly near `sqrt(2)` (genuine bounded chaos, not divergence).
Cross-checked against topology-based dimension estimates on the same run
(`two_nn_dimension≈4.5`, `correlation_dimension≈3.1`, both near this
project's own Gate 1 L=22 benchmark of `D_KY∈[5.2,5.6]` —
`tests/replication/test_gate1_kaplan_yorke.py::test_L22_lyapunov`; `D_KY=22`
is the CANONICAL L=100 answer, not L=22's — see `project_ks_true_dky_benchmark`
memory, which needs the L=22-specific correction noted here). The raw
physical ground truth is itself remarkably smooth at L=22 (confirmed
directly, not assumed): KS's linear instability band is `0<k<1`, and only
discrete wavenumbers `k_m=2*pi*m/L` are allowed on a periodic domain — at
`L=22` only `m=1,2,3` (`k≈0.29,0.57,0.86`) fall inside that band before
`k_4≈1.14` exits it, vs. roughly 16 at `L=100`. Far fewer independently-
growing directions → genuinely low-dimensional, smooth-looking attractor,
not a truncation artifact of the K=24 rFFT reconstruction (verified by
comparing directly against the untouched raw-`u` physical Hovmöller, which
shows the same smoothness). So `D_KY≈5-6` is the physically correct target
for L=22, and Section 104's `D_KY=22` was inflated specifically by the
propagator's own spurious high-wavenumber content.

### 18a. `plot_latent_hovmoller` now plots `w=irfft(z)`, not raw `z`

User: "when you plot the latent_hovmoller plots can you plot the irfft of
the states z, instead of z itself." Raw `z` is just `K` real + `K`
imaginary rFFT coefficients with no spatial meaning — a "hard-saturated
high-index channel" there is one Fourier coefficient blowing up, not
visible spatial structure. `plot_latent_hovmoller` (`scripts/
visualize_rollout.py`) now decodes both panels via `decode_from_spectrum
(z, K, N_w)` whenever `ae_cfg` carries `K`/`N_w` (`encoder_kind=
"spectral_field"`; falls back to the original raw-`z` behavior otherwise),
turning it into a genuine `w`-space Hovmöller directly comparable to
`plot_physical_hovmoller`'s own `u`-space triptych. `--reorder-by d7`'s
cosmetic column permutation is skipped in the decoded case (a physical
position axis already has inherent meaning, nothing to reorder).

### 18b. `w_lowpass_rollout`: constrain the PROPAGATOR's own rolled-out high-mode energy

Stage-1 (`Stage1TrainingConfig.w_lowpass_rollout`/`lowpass_rollout_power`)
and Stage-2 (`Stage2TrainingConfig.w_lowpass_rollout`/`lowpass_rollout_power`)
analogues of `w_lowpass`, but applied to the PROPAGATOR's own rolled-out
`z_pred` (Stage 2) or the joint-training aux propagator's short `z_pred`
(Stage 1) rather than the encoder's direct `z` — directly targets what §18
found: the encoder's own `z` already looked reasonable, nothing previously
constrained what the PROPAGATOR itself does once it starts free-running.
Reuses `low_pass_spectral_loss` unchanged. CLI: `--w-lowpass-rollout`/
`--lowpass-rollout-power` on both `train_stage1_patched.py` and
`train_stage2_patched.py`.

**Section 106** (queued/running as of this note): repeat of Section 104's
exact recipe (K=24, N_w=64, `w_shape_floor=0.1`, `euler`, joint training),
Phase 1 RETRAINED FROM SCRATCH (not reusing Section 104's checkpoints —
user explicitly corrected an earlier draft of this plan that tried to reuse
them, since `w_lowpass_rollout` needs to shape the encoder from the start,
not fight an already-fixed Phase-1 representation) with `w_lowpass_rollout
=0.015` (Phase 1) and `=1.0` (Phase 2) active, both calibrated directly
against real data (Phase 1: ~1x `l_recon`/`l_pred` at fresh init; Phase 2:
~2x `l_latent` against Section 104's own trained checkpoint as a proxy) —
"somewhat aggressive," per user direction, not the gentle ~14% convention.
Goal: bring `D_KY` down from 22 toward the physically correct ~5-6.

## 19. `pde_head` distillation (2026-09-08): a hybrid that keeps training easy but still yields a PDE

User, after Section 101-106's `backbone="spectral_pde"`-as-primary-propagator
struggles ("this new framework seems incredibly hard to train"): "however
learning an mlp to map latent states to future latent states doesn't
readily admit a pde. is there a hybrid between these approaches that we
could learn the evolution of the system more effectively, but still easily
derive the correct pde describing the latent system evolution based on our
solution?"

**Why not post-hoc SINDy on a frozen good model** (the first, rejected
proposal): already tried and already failed, precisely (§4.2): `fit_latent_pde.py`
fit on a FROZEN Section 98 checkpoint got `R^2~0.005`, collapsed to `D_KY=0`
— the representation was never pressured toward local-fittability, so
nothing guaranteed one existed to find.

**The actual design**: keep training the FREE, unconstrained `aux`
propagator as primary (whatever backbone reliably produces chaos per
H-PROP — e.g. `"mlp"` — driven by the real `l_pred` rollout loss,
unchanged). Train a SECOND propagator, `pde_head` (always `backbone=
"spectral_pde"`), ALONGSIDE it, but give it an easy job: match `aux`'s own
realized ONE-STEP prediction (`z_pred[:, 0]`, already computed, `.detach()`ed
— no gradient back into `aux`, which stays free to fit real dynamics
uncorrupted by pressure to look local), via `pde_head.step_one(z_start)` — a
single ordinary forward pass, never an autoregressive rollout under
gradient pressure. This is the load-bearing structural difference from
what made `backbone="spectral_pde"`-as-primary so hard (Sections 101-106):
`pde_head` is never responsible for its own stable multi-step numerical
integration under gradient pressure, only for a single-step regression fit.
Gradient from this term reaches `pde_head`'s own weights AND, through
`p_pred`'s dependence on `z_start`, the ENCODER — pressuring it toward a
representation the local-derivative-MLP form can actually fit, jointly and
concurrently (not after the fact on a fixed representation, which is
exactly reason (3) §4.2 already identified as the earlier attempt's flaw).

**Tradeoff, stated directly**: `pde_head` is graded against `aux`'s own
dynamics, not the true KS system directly, so any bias in `aux` becomes a
bias in the extracted PDE; and a good single-step fit doesn't guarantee a
stable multi-step integration of `pde_head` alone (untested until actually
tried — the natural next validation once this trains).

**Implementation**: `train_stage1(..., pde_head=...)` (`ks_latent/training/
loops.py`) — optional second `AuxPropagator`, always constructed by the
caller as `backbone="spectral_pde"`; `pde_head.parameters()` added to the
optimizer alongside `ae`/`aux`; new batch-loop term (gated on `cfg.
w_pde_distill > 0` and `pde_head is not None`, alongside every other
optional regularizer): `z_start` = the same starting state `aux`'s own
rollout used (`z_hist_in[:, -1]` in `"history"` mode, `z_curr_in`
otherwise), `g_target = z_pred[:, 0].detach()`, `p_pred = pde_head.step_one
(z_start)`, `l_pde_distill = reconstruction_loss(p_pred, g_target)`.
`Stage1TrainingConfig.w_pde_distill: float = 0.0` (off by default).
`train_stage1_patched.py`: `--pde-distill` (requires `--encoder
spectral_field`; rejects `--aux-backbone spectral_pde` as redundant/
contradictory — the whole point is a second head alongside a FREE aux),
`--w-pde-distill`, `--pde-hidden`/`--pde-n-blocks`/`--pde-integrator`
(default `etdrk4`, the numerically safest per §17/Section 105's stiffness
lesson, though a single default-`euler` substep with a small non-physics-
prior correction is stable regardless)/`--pde-ode-substeps`/`--pde-max-order`/
`--pde-physics-prior`. Saved as a separate checkpoint,
`stage1_pdehead_<profile><tag>.pt`, loadable via the ordinary
`load_propagator_checkpoint` (verified via a real round-trip: load, call
`.step_one`, finite output) — this is the artifact to eventually inspect
`field()` on directly for the extracted PDE's actual learned form.

**Tests**: `tests/unit/test_pde_head_distillation.py` — directly verifies
the gradient-isolation property the whole design rests on (`aux` gets
EXACTLY zero gradient from this term; `pde_head` and the ENCODER both get
nonzero gradient), by replicating `train_stage1`'s exact code path outside
the full loop. `tests/integration/test_training_loops_smoke.py` — real
`train_stage1(..., pde_head=...)` call trains end-to-end and reduces loss;
a second test confirms `pde_head` present but `w_pde_distill=0` reproduces
`pde_head=None`'s history EXACTLY (no accidental coupling). Verified via a
real CLI smoke run (`--pde-distill --w-pde-distill 0.5`) end-to-end,
including the checkpoint round-trip. Not yet run in a real experiment as of
this note — natural next step once Section 106 finishes and the GPU is
free.

## 20. `backbone="spectral_pde_raw"` (2026-09-08): generalize `pde_head` to ANY encoder, via z's own self-FFT

User, after §19: "I'm confused, are you saying the encoder and decoder are
also operating in fourier space?" (answered: no -- `KSAutoencoderSpectralField`'s
learned ViT sub-network operates entirely in physical space on both ends;
the fixed rFFT/irFFT is a bolted-on wrapper specific to that ONE encoder
kind, not something every encoder does). Then: "I think this hybrid/joint
training idea is really good, but we need to refine it. I would like to be
able to use the encoder and decoder and propagator from 95. but as an
additional step, within the propagator, I want to take the fourier
transform of the latent states, and train the spectral pde with this
information (since this gives us closed form derivatives.) This then
becomes part of the loss ... Is this possible in the current setup?"

**The refinement**: §19's `pde_head` required a dedicated `spectral_field`
encoder, since `_SpectralPDEDeltaBody` needs its input to already BE a
truncated rFFT spectrum. This generalizes it to work on the raw `z` of ANY
encoder (Section 95's own `vit_fourier_hybrid` + `mlp`/markovian propagator
included, `artifacts/stage1_{ae,prop}_full_section95_...pt`, confirmed
`d_latent=44`) by having `pde_head` take its OWN self-FFT of `z` first:
`z`'s `d_latent` indices are treated as a spatial coordinate on a periodic
ring of circumference `L` (the same "native index as space" convention
`spatial_coherence_loss`/`circular_weighted_stats` already use elsewhere in
this project, extended here to actually synthesize closed-form derivatives
along it, not just measure correlation banding) -- `z_hat =
encode_to_spectrum(z, K)`, fed through the ordinary, UNCHANGED
`_SpectralPDEDeltaBody` machinery, then `decode_from_spectrum` back to the
same raw `d_latent` space `aux` operates in.

**User decisions** (asked directly, both confirmed): (1) Section 95's
encoder+propagator should NOT be warm-started/frozen -- "train from scratch
with this method," with the goal stated explicitly: "to have an encoder and
decoder that can accurately model the underlying geometry and
dimensionality of the system and to have a propagator that both accurately
rolls out latent space over time but can also be modeled by a pde." (2)
`K=d_latent//2+1` (no truncation -- keep the entire self-spectrum) and
`L=d_latent` (unit index spacing) as the default self-FFT parameters, both
genuinely arbitrary (the latent index has no natural physical length
scale) but reusing this project's existing convention.

**Honest risk, named directly** (not glossed over): treating the latent
INDEX as a spatial coordinate is exactly what the earlier, already-failed
post-hoc SINDy attempt did (§4.2), and exactly the "search over alternative
coordinates" idea §3.3 recorded as "considered, not pursued" on 2026-09-02.
The bet this time is the same as §19's: JOINT, not frozen -- gradient from
the distillation loss can push the encoder+`aux` toward a `z` ordering that
genuinely admits local derivative structure, rather than hoping a fixed,
arbitrary ordering already has it.

**Implementation**: `_SpectralPDERawDeltaBody` (`ks_latent/models/
propagator.py`) -- a thin wrapper composing an internal, unmodified
`_SpectralPDEDeltaBody`. New `backbone="spectral_pde_raw"` on both
`PropagatorConfig`/`AuxPropagatorConfig` (mode='markovian' only, same as
`"spectral_pde"`; no `d_latent==2*spectral_K` constraint -- `spectral_K` is
now a purely internal self-FFT parameter, independent of `d_latent`; no
`spectral_physics_prior` option -- raises `ValueError` if set, since no
"true governing equation" exists for an arbitrary learned ordering).
`train_stage1_patched.py --pde-distill` now builds this backbone
generically for `pde_head` regardless of `--encoder`; new `--pde-K`/
`--pde-L` flags (defaulting to the user-confirmed `d_latent//2+1`/
`d_latent`); the old `--encoder spectral_field` requirement is gone (the
redundancy check against `--aux-backbone` now covers both `spectral_pde`
and `spectral_pde_raw`).

**Real finding while testing this (2026-09-08), corrected the default**:
`integrator="etdrk4"` bakes in `Lhat=k^2-k^4` -- KS's TRUE physical
dispersion relation, a correct fact for a genuine `spectral_field` z, but
an unjustified, empirically HARMFUL assumption for an arbitrary self-FFT'd
latent ordering. Verified directly: a real `train_stage1(..., pde_head=...)`
run with `--pde-integrator etdrk4` diverged (loss `0.57 -> 268.0` over 5
epochs); the IDENTICAL setup with `--pde-integrator euler` trained stably
(`0.51 -> 0.48`). `train_stage1_patched.py`'s `--pde-integrator` default
changed to `"euler"` for this backbone specifically (unlike `"spectral_pde"`'s
own `etdrk4`-favoring default) -- `etdrk4` remains selectable to
deliberately test whether a `k^2-k^4`-like prior helps a specific learned
ordering, just not assumed safe.

**Tests**: `tests/unit/test_propagator_spectral_pde_raw.py` (11 tests --
shape, identity-at-init for `euler`, non-identity for `etdrk4`, gradient
flow, arbitrary `d_latent` not tied to `2*K`, config validation including
the `spectral_physics_prior` rejection). `tests/unit/
test_pde_head_distillation.py` and `tests/integration/
test_training_loops_smoke.py`'s `pde_head` tests updated to use a PLAIN
(non-spectral) encoder + `spectral_pde_raw`, matching what the CLI now
actually builds for any `--encoder`. Verified end-to-end via a real CLI
smoke run using Section 95's EXACT architecture recipe (`vit_fourier_hybrid`
+ `mlp`/markovian, `d_model=64`/`fourier_hidden=95`/etc., `--full-propagator`)
plus `--pde-distill --w-pde-distill 0.05` -- trains cleanly, no NaN,
checkpoints written correctly.

**Magnitude calibration** (real L=100 data, Section 95's exact architecture,
fresh init): both `aux` (zero-init `mlp`) and `pde_head` (zero-init
`spectral_pde_raw`) reduce to EXACT identity at init, so raw
`l_pde_distill=0` there -- not a usable calibration point. With `pde_head`
at its default (non-zero) random init instead (approximating early-training
divergence from identity): `l_recon=1.681`, raw `l_pde_distill=0.092`
(~5.5% of `l_recon`). This is a rough, not a precise, calibration --
`l_pde_distill`'s actual scale during a real run depends on how far `aux`'s
own predictions have already diverged from `z`, which changes throughout
training as both co-evolve -- but suggests `w_pde_distill` in the
`1.0-3.0` range for a moderate (not negligible, not dominant) starting
contribution, roughly matching this project's usual ~14%-of-primary-loss
gentle-regularizer convention.

**Not yet run as a real experiment** — next step: launch a "Section 107"
using Section 95's exact architecture (fresh init, not warm-started per
user direction), `--pde-distill --w-pde-distill` at the calibrated
starting weight, and monitor: (1) whether `aux`'s own rollout accuracy
(`l_pred`/Gate 3-4) holds up as well as Section 95's original run despite
the added joint pressure: (2) `pde_head`'s own single-step distillation
fit quality over training (does it improve, i.e. is `z` actually becoming
more locally-PDE-fittable); (3) qualitatively, whether `pde_head`'s learned
`field()` shows recognizable structure when probed directly.

## 21. `pde_head` continues into Stage 2 (2026-09-08): options A and B, MUTUAL not detached

User: "is the pde_head trained during phase 2 as well. we should try to
get pde rollout to have decent performance" then, after being presented
two options (A: extend the safe single-step distillation; B: train
`pde_head` on its own autoregressive rollout, the riskier direct
approach): "please implement both option a and b, and we can try both.
at the end of the day, we're really just training the propagator right,
the pde_head training is acting as a regularization term."

**Key structural point that changes the design from Stage 1's own
mechanism**: Stage 2 (`train_stage2`) never touches the encoder --
`train_sequences` is already-encoded, frozen `z` (from `encode_dataset_
with_shifts` on a frozen Stage-1 AE). Stage 1's `pde_head` loss detaches
its target specifically so gradient reaches the ENCODER (the thing being
regularized there) without corrupting `aux`. In Stage 2 there is no
encoder gradient path at all -- a detached target would leave NOTHING for
the loss to regularize except `pde_head` in isolation, which contradicts
the user's own framing ("pde_head training is acting as a regularization
term" [on the propagator]). So Stage 2's version is deliberately MUTUAL:
`propagator`'s own `z_pred` is NOT detached, so gradient flows into BOTH
`propagator` (genuinely regularized toward dynamics a local PDE can also
explain) and `pde_head` (learns to track it).

**Option A** (`Stage2TrainingConfig.w_pde_distill`, safe): at EVERY one of
the `k_now` steps of `propagator`'s own realized rollout, `pde_head.
step_one` is evaluated from the SAME state `propagator` started that step
from (`z_states = cat([z_start], z_pred[:, :-1])`), compared to
`propagator`'s own realized next state. `pde_head`'s own gradient is
always a single-step regression -- verified directly (`tests/unit/
test_pde_head_stage2_mutual.py::test_option_a_pde_head_gradient_is_single_step_only`)
that the batched formula used in the training loop exactly matches
`k_now` independent `step_one` calls, i.e. NOT a chain through `pde_head`'s
own output.

**Option B** (`Stage2TrainingConfig.w_pde_rollout`, risky): `pde_head.
rollout(z_start, z_start, k_now)` -- a genuine autoregressive chain
through `pde_head` itself -- compared to `propagator`'s own `z_pred` over
the whole trajectory. Verified directly that this really is a chain
(`test_option_b_is_a_true_autoregressive_chain`: matches step-by-step
manual chaining exactly). This is the same mechanism that made
`backbone="spectral_pde"` "incredibly hard to train" as a PRIMARY
propagator in Sections 101-106 (stiffness, collapse) -- offered so both
can be tried and directly compared, now that Stage 1's own distillation
may have already pressured `z` toward being more locally-PDE-fittable
than those earlier attempts ever started from.

Either or both may be active simultaneously (independent weights, both
0.0/off by default).

**Implementation**: `train_stage2(..., pde_head=...)` -- new optional
param, `pde_head.parameters()` added to the optimizer alongside
`propagator`'s own. `train_stage2_patched.py --init-pdehead-checkpoint`
loads a `pde_head` checkpoint (the same `stage1_pdehead_*.pt` format
Stage 1 writes) to CONTINUE training it here (not construct fresh --
continuity was the whole point: "is the pde_head trained during phase 2
as well"); `--w-pde-distill`/`--w-pde-rollout` gate the two terms. Saved
separately as `stage2_pdehead_patched_<profile><tag>.pt` (periodic +
final, mirroring `propagator`'s own checkpoint convention), loadable via
the ordinary `load_propagator_checkpoint`.

**Tests**: `tests/unit/test_pde_head_stage2_mutual.py` (4 tests -- mutual
nonzero gradient for both options independently, option A's single-step-
only property, option B's genuine-chaining property, all verified by
direct comparison against hand-computed reference formulas, not just
"runs without crashing"). `tests/integration/test_training_loops_smoke.py`
-- real `train_stage2(..., pde_head=...)` call with BOTH options active
simultaneously trains end-to-end (finite loss); a second test confirms
`pde_head` present but both weights `0.0` reproduces `pde_head=None`'s
history EXACTLY (no accidental coupling). Verified end-to-end via a real
CLI round-trip: Phase 1 (`--pde-distill`) writes `stage1_pdehead_full_
....pt`, Phase 2 (`--init-pdehead-checkpoint ... --w-pde-distill 1.0
--w-pde-rollout 0.5`) loads and continues it, trains cleanly (no NaN,
`loss` finite every epoch), writes `stage2_pdehead_patched_full_....pt`,
confirmed loadable and callable afterward.

Full suite: 617 passed (up from 611) after this addition, no regressions.

## 22. Stage 1's own `pde_head` loss can also be made mutual (2026-09-08)

User: "I think the loss should be mutual for stage 1 training too. We
always want the propagator to have dynamics that can be easily modeled
by the pde_head right? tell me if I'm wrong." Agreed in principle --
detaching only ever shapes the encoder; `aux`'s own computation could
stay arbitrarily non-local regardless of how nice `z` looks, so if the
actual goal is a propagator whose OWN dynamics are PDE-describable,
gradient must reach `aux` directly. Real risk, stated back to the user
before building it: this is exactly the "pressure toward simplicity"
mechanism behind H-PROP (every propagator pressured toward simple/local
has collapsed to a fixed point) -- if `pde_head` is weak or lazily fit
(plausible early in training, near-identity at `zero_init`), a mutual
loss could pull `aux` toward matching that impoverished target instead
of real dynamics.

**Implementation, built as an explicit toggle (not a silent replacement
of the existing, tested, protected default)**: `Stage1TrainingConfig.
pde_distill_detach_target: bool = True` (default = unchanged, protected
behavior). `train_stage1`'s distillation block now conditionally detaches
`g_target` on this flag instead of unconditionally. CLI: `--pde-mutual`
on `train_stage1_patched.py` (sets it `False`). Tests: `tests/unit/
test_pde_head_distillation.py::test_aux_receives_nonzero_gradient_when_mutual`
verifies `aux` genuinely receives gradient when mutual (the existing
`test_aux_receives_zero_gradient_from_distillation_loss` continues to
verify the unchanged default). Verified via a real CLI smoke run
(`--pde-distill --w-pde-distill 0.5 --pde-mutual`) -- trains cleanly.

## 23. Training-log visibility + "Phase 3": frozen-propagator `pde_head` refinement (2026-09-08)

User: "should we add a phase 3 that refines the pde with the propagator
fixed? basically how should we extract the pde? also can we report the
pde loss in the logs for training, just to see if it's going down?"

**Logging**: both `train_stage1` and `train_stage2` now track and report
the RAW (unweighted) `pde_distill` (and, in Stage 2, `pde_rollout`) loss
per epoch -- printed in the verbose per-epoch line and written into each
epoch's `history` dict entry, gated on `pde_head is not None` (and the
corresponding weight `> 0`) so it's silently absent when unused, matching
every other optional term's convention. Reporting the RAW (not
weight-multiplied) value specifically so its trend is comparable across
runs using different `--w-pde-distill` settings.

**"Phase 3" answers "how should we extract the pde" directly**: Phases
1-2's `pde_head` losses exist to SHAPE the encoder/propagator toward
being PDE-describable WHILE they're still being fit to real data --
necessarily a compromise, since `aux` is also being pulled toward real
accuracy at the same time. Once `aux` is fully trained and its dynamics
are considered final, the cleanest way to get the best possible PDE fit
to those FIXED dynamics is a dedicated final stage with literally zero
competing pressure on `aux` -- ordinary teacher-frozen distillation
(train the teacher first, then distill into the student against a fixed
target, the standard pattern this is named after).

**Implementation**: `train_stage2(..., freeze_propagator=True)` --
`propagator` is kept in `.eval()` (importantly, this required an explicit
fix: `eval_stage2_kmax` unconditionally restores `.train()` on exit,
which would silently re-enable dropout on the "frozen" teacher every
epoch -- `train_stage2` now re-applies `.eval()` immediately after that
call when `freeze_propagator=True`) and every one of its parameters is
set `requires_grad_(False)` (so its own rollout builds no autograd graph
at all -- free, not just harmless) and excluded from the optimizer
entirely; only `pde_head` trains. The mutual-vs-detached distinction from
§21 is moot here -- `propagator` gets zero gradient regardless of
anything about the target, since it was never in the optimizer to begin
with. Requires `pde_head` (raises `ValueError` otherwise -- nothing to
train). CLI: `--freeze-propagator` on `train_stage2_patched.py` (requires
`--init-pdehead-checkpoint`).

**Tests**: `tests/unit/test_pde_head_freeze_propagator.py` (4 tests --
`ValueError` without `pde_head`; `propagator`'s state dict is BYTE-
IDENTICAL before/after training while `pde_head`'s genuinely changes;
`val_kmax_mse` stays EXACTLY constant across epochs with `dropout=0.3`
specifically to make a silent `.train()` leak obvious if the
`eval_stage2_kmax` fix regressed; `pde_distill`/`pde_rollout` both
present and finite in every history entry). Verified via a real CLI
round-trip (Phase 1 `--pde-distill` -> Phase 2 `--freeze-propagator
--init-pdehead-checkpoint ... --w-pde-distill 1.0 --w-pde-rollout 0.5`):
`val_kmax_mse` identical (0.378409) every one of 3 epochs, confirming the
propagator is genuinely frozen; `pde_distill`/`pde_rollout` both logged
and finite, dropping toward ~0 within 3 epochs on this tiny smoke setup.

Full suite: 622 passed (up from 617) after §22-23, no regressions.
`scripts/section107_L40_pdehead_mutual.sh` updated to a 4-stage pipeline
(Phase 1 mutual -> Phase 2 continue -> Phase 3 freeze-and-refine -> Gate
3/4 on Phase 2's own propagator), queued to launch once the new L=40
datasets finish generating.
