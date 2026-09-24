# Open Questions

## From the source documents

1. **NX / d disagreement.** `docs/ML_for_KS_writeup.md` describes NX=128,
   d=24, stride 5. `docs/PROJECT_HANDOFF.md` (canonical) uses L=100, NX=1024,
   d=44. This codebase defaults to the handoff's configuration everywhere
   and makes both configurable (brief top matter).

2. **Missing addendum file.** `docs/LATENT_PDE_RESEARCH_NOTES_addendum_2026-08-28.md`
   is referenced throughout `CLAUDE_CODE_BRIEF.md` (§12.1, §12.3, §13.1-13.4,
   §14.3-14.5, §14.4 prior-art) but was not present in the working directory
   at the start of this build. Proceeded using the brief's inline quotes of
   the addendum's operative content, per user direction. If the full file
   becomes available, re-check Phases 2, 6, 9, 10, 12 sections against it
   for anything not already quoted in the brief.

3. **Missing `reference/` contingency prototype.** `reference/latent_locality.py`
   and `reference/test_locality.py` (Phase X contingency, brief §17) are not
   present either. Not needed unless Phases 10-12 stall and Phase X is
   invoked; flagged here so it isn't forgotten if that happens.

4. **Prior-art citations marked "verify" in the source docs** (brief §22,
   e.g. the "Spatially Localized Models of Extended Systems" Springer
   chapter authorship/year) have not been confirmed against primary sources.
   Do this before any of these citations appear in `docs/RESULTS.md` or
   similar external-facing output.

## Carried forward from the addendum (per the brief's citations of it)

- **Open question 7** (§14.3): whether a stencil law fitted on a fixed
  linear pooling of the fine latent matches one fitted on an independently
  trained coarse latent -- addressed in Phase 12's RG-nesting comparison,
  not yet run.
- Open questions 8-10: not independently knowable without the addendum
  text itself; nothing in the brief's quotes names them specifically.
  Revisit if the addendum surfaces.

## Phase 1 implementation notes worth carrying forward

- `test_convergence_in_dt` (brief §3.2) needed a specific IC/domain choice
  to measure cleanly: a small-L, low-amplitude, fast-decaying regime showed
  a spuriously low apparent order (~1-3, not approaching 4) because the
  nonlinear forcing became negligible relative to floating-point noise at
  fine dt, not because of a solver bug -- confirmed independently against
  `scipy.integrate.solve_ivp(DOP853, rtol=1e-13)` on the same IC, which
  shows the true ETDRK4 order approaching 4 cleanly. The test now uses
  L=22, a smooth two-mode IC of moderate amplitude, over t_final=1.0.
- `test_thinning_does_not_fix_correlation` reproduces cleanly at L=100
  (true D_KY ~22) but not at L=22 (true D_KY ~5), where the effect is
  swamped by two-NN's own sampling noise at n=800 points. Also found (and
  documented in the test) a *separate* two-NN failure mode at very short
  thinning strides (<~1 physical time unit): near-duplicate temporal
  neighbors give mu=r2/r1 close to 1, which is numerically unstable in the
  origin-forced regression and inflates the estimate rather than deflating
  it -- worth knowing before Phase 4 tunes stride choices by eye.
- The two-NN estimator (`ks_latent/analysis/dimension.py`) currently
  implements only the two-NN method itself, validated on d-spheres for
  d in {2,5,10} and a d=3 uniform cube. Correlation dimension, diffusion
  maps, and the full d in {2,5,10,15,20,22} validation matrix are Phase 4
  scope (brief §6.1) and not yet built.
- `ks_latent/analysis/lyapunov.py`'s generic `benettin()` was built now
  (Phase 1 needs it for Gate 1) rather than in Phase 4 where the brief
  formally introduces it. It already supports the "clean interface
  accepting any lattice model" requirement (brief §6.2) via the
  `step_with_tangent_fn(state, tangent) -> (state_next, tangent_next)`
  signature; Phase 4 still needs to add the `single_state`/`two_step`
  torch-propagator adapters and the physical-time-units reporting wrapper
  for that model.

## Phase 3 implementation notes / inferred choices

- **Local-transformer depth (`AutoencoderConfig.n_local_layers`) is not
  stated in the brief.** Only the global transformer's depth is explicit
  ("2 layers, 24 tokens"). Defaulted to 1 layer; if the real handoff
  codebase used more, `docs/RESULTS.md`'s Stage-1 val-reconstruction number
  should be compared against the recorded run with this in mind before
  concluding a mismatch is a bug.
- **AE transformer dropout defaulted to 0.0** (not stated in the brief for
  Stage-1; only the Stage-2 propagator's dropout=0.1 is explicit in
  PROJECT_HANDOFF.md).
- **Aux-propagator hidden width** inferred as `hidden == dim_ff == 64`
  (matching the pattern PROJECT_HANDOFF.md states explicitly for the main
  Stage-2 propagator, `hidden == dim_ff == 128`), giving ~25.6k params
  against the brief's "~22k" -- close enough to trust the architecture
  pattern, but if this ever needs to match a real checkpoint's param count
  exactly, re-derive from that checkpoint rather than this estimate.
- **Noise-anneal and horizon-curriculum schedule shapes**: linear anneal
  for Stage-2 input noise (brief only says "annealed 0.10->0.02", no shape
  specified); the K-curriculum formula `2 + floor(min(e/E_warm,1)*(K_max-2)+0.5)`
  is taken directly from `docs/ML_for_KS_writeup.md` §4.4, which does give
  the exact formula.
- **Stage-1 window-sampling epoch definition**: implemented as a true
  index-shuffle epoch over *all* valid `(run, start)` windows in the
  training split (brief §1.3.4's explicit MPS guidance), not a fixed
  steps-per-epoch random sampler -- this determines what "one epoch" means
  when comparing wall-clock/epoch numbers against any other implementation.

- **Deferred experiments from the "Ported improvements" comparison
  (2026-08-29, CLAUDE_CODE_BRIEF.md §5.1/5.2 addendum)**, not yet run:
  - `Stage1TrainingConfig(w_pred=0.0)` as a cheap stand-in for the
    reference project's "two-stage" (`E`/`D`-then-`M`) training
    curriculum -- already expressible with existing config, deliberately
    not run alongside the Gate 3/4 canonical-recipe validation to avoid
    confounding it.
  - `noise_std > 0.0` (denoising reconstruction training) -- implemented
    and unit-tested but not yet validated at real-data scale.
  - The `RegConfig` banded-smoothness/off-band-decorrelation penalty
    (`ks_latent/training/regularizer.py`) is implemented and unit-tested
    but **not a recommended experiment -- user direction, 2026-08-29.**
    The reference project it was ported from already measured it
    collapsing the latent (participation ratio ~1.1 of 8 at
    `lambda_z >= 5e-3`), i.e. it recreates this project's own
    already-escaped collapse failure mode. Kept in the codebase for
    possible future use, not to be proposed as a fix.
  - `Stage2TrainingConfig(latent_loss="l1")` for the propagator's
    latent-space loss.
  - Architecture ideas explicitly considered and **not** ported: a
    ViT/CNN encoder track, an attention-based propagator tokenized per
    latent coordinate, and a physical-space "delta" decoder mode
    (`xhat_{t+1} = x_t + D_delta(z_{t+1})`) -- see the brief addendum for
    the full reasoning on each.
  - **In progress (2026-08-29, CLAUDE_CODE_BRIEF.md §5.1/5.2 second
    addendum), user-directed:** `--encoder mlp` (plain-MLP encoder/decoder,
    `KSAutoencoderMLP`) and `--multistep` (real-sized propagator + an
    8-step rollout curriculum in Stage 1, `Stage1TrainingConfig.k_pred_max`)
    on `scripts/train_stage1_patched.py`, tested independently and combined
    against the reference project's own `L=94`-scale benchmark showing
    ~100x lower reconstruction MSE. Code implemented and unit/integration
    tested; the four-cell comparison run is in flight -- see
    `docs/RESULTS.md` for results once it finishes.

## Phase 4 implementation notes / findings

- **Two-NN dimension is biased in *opposite directions* on spheres vs. tori
  at high true dimension.** Sphere sweep (N=6000): true d=22 reads ~18.5
  (downward bias, matching the brief's documented "true 22 reads ~19"
  almost exactly). Torus sweep (same N): true d=22 reads ~28 (upward bias,
  confirmed repeatable across seeds, `tests/unit/test_dimension.py`). This
  is a genuine, repeatable property of the estimator on product-topology
  manifolds, not a bug -- worth remembering before trusting a two-NN number
  on any manifold whose topology isn't sphere-like. The real 44-dim latent
  attractor is reported topologically trivial / connected-blob-like in
  `docs/PROJECT_HANDOFF.md`, i.e. much closer to the sphere case, so this
  caveat is about estimator robustness in general rather than a live risk
  for this project's actual data.
- **Correlation dimension underestimates much more severely than two-NN at
  high d** (N=6000: ~11.5 at true d=22 on a sphere, vs two-NN's ~18.5) --
  consistent with the brief's "document that it is a lower bound," but the
  gap is large enough that the two estimators should never be expected to
  agree numerically at d>~15; only their *ordering* and *monotonic trend
  with true d* are meaningful cross-checks at that scale.
- **Diffusion maps validated via spectral degeneracy + rotation-invariant
  embedding radius**, not by comparing individual eigenvectors to a known
  parameterization (eigenvectors within a degenerate eigenspace are only
  defined up to an arbitrary rotation from `eigh`). Circle: 2 leading modes;
  2-sphere: 3 (matching l=1 spherical harmonics); 2-torus: ~4, with looser
  degeneracy (the product-manifold spectrum doesn't factor as cleanly under
  an isotropic Gaussian-kernel bandwidth as the sphere's does).
- **DTM "earning its keep" only shows up with individually-scattered
  outliers**, not a single far-away outlier cluster (which plain Rips
  already handles fine, since it only inflates one region of the diagram).
  Both `docs/CLAUDE_CODE_BRIEF.md`'s Phase 4 test design and the outlier
  construction in `test_dtm_outperforms_plain_rips_on_scattered_outliers`
  reflect this.

## Phase 5 implementation notes / inferred choices

- **The exact original NAT-PFF formula could not be reconstructed** --
  `docs/PROJECT_HANDOFF.md` names quantities from the original script
  ("N_mat", "L_k", "k_bar/N") without enough detail to rebuild the exact
  stochastic-term formula, and no `reference/pff.py` exists in this
  directory. See `docs/RESULTS.md`'s Phase 5 section for the full account:
  an independently-derived preconditioned-Langevin algorithm was built
  instead, satisfying every literal, checkable spec in the brief (F metric,
  prior term, ds schedule, noise/RMSE conventions) and verified against the
  brief's own stated correctness bar (`test_pff_gaussian_linear`, exact
  Kalman recovery). Full derivation in `ks_latent/da/pff.py`'s docstring.
- **`ds_max` has no stated default in the brief** (only the recursion is
  given). Defaulted to 10.0; this only changes how fast the adaptive step
  size accelerates once the flow is nearly converged, not the fixed point
  the filter converges to.
- **`grad_j` re-linearizes `h` at the *current* ensemble mean every
  pseudo-time step** (a proper iterative Gauss-Newton scheme, needed for
  nonlinear `h`), while the *prior* term is anchored to the *initial*
  (fixed) forecast mean `z0_bar` throughout -- these are two different
  reference points by design, both appearing in the brief's formulas
  (`z0_bar` in "prior term ... (z^j - z_bar)", the ensemble-mean Jacobian
  implicitly re-evaluated each step). For a linear `h` (the one
  case that's actually tested) this distinction is moot, since the
  Jacobian is the same everywhere.
- **A real numerical-stability bug was found and fixed during
  development**: naive Euler-Maruyama discretization of the Langevin SDE is
  unstable once the adaptive `ds` schedule grows past ~2, inflating the
  analysis covariance by >500% in `test_pff_gaussian_linear` before the
  fix. Replaced with the exact Ornstein-Uhlenbeck transition kernel
  (unconditionally stable). Worth knowing if anyone later "simplifies" the
  integrator back to plain Euler.

## Phase 6 implementation notes / inferred choices

- **D3's Jacobian coupling matrix is `d z_{n+1,k}/d z_{n,l}` holding
  `z_{n-1}` fixed at its sampled value** (brief's literal formula names
  only `z_n`, not `z_{n-1}`), matching Phase 4's `single_state` convention
  for consistency across diagnostics that need "the" propagator Jacobian.
  This is a partial, not full, Jacobian of the true 2-step map -- if D3's
  bandedness conclusion is ever surprising, check whether the `z_{n-1}`
  branch (available via the `two_step` two_step machinery in
  `lyapunov.py`) changes the picture.
- **D3's bandedness kernel** `w(dist) = exp(-dist^2 / (2*bandwidth^2))` and
  its `bandwidth` parameter (default 3.0 sites) are not specified in the
  brief beyond "a normalized sum weighted by a function of circular
  distance" -- the specific decay shape/scale is a free choice that
  determines how sharply "banded" must mean "banded" to score well; if D3's
  verdict looks sensitive to this, sweep `bandwidth` and report the range.
- **D2's wavelet-energy-entropy measure** is one reasonable choice among
  several the addendum's "third possibility" (a joint space-scale
  localization measure) could mean; implemented as normalized Shannon
  entropy of the `pywt.wavedec` coefficient energy distribution (`db4`
  wavelet, periodization mode). A different wavelet family or a raw
  wavelet-scalogram visualization would be a reasonable alternative if this
  measure doesn't discriminate well on the real trained model.
- **D4's shift grid** (`range(0, NX, NX//16)`, i.e. 16 shifts spanning the
  domain) and **D5's patch-length grid** (brief's own `{5,10,20,30,40,50}`
  for the full run) are both configurable in `scripts/run_diagnostics.py`;
  the smoke profile uses much smaller values appropriate to its tiny `NX`.

## Lorenz-96 generalization test

See `docs/RESULTS.md`'s "Lorenz-96 generalization test" section for the
full finding and citations; unresolved items only, here:

- **Stochastic closure, untested.** Lu/Lin/Chorin's discrete Mori-Zwanzig
  decomposition for L96 splits the needed memory term into a deterministic
  history function `Phi` plus a stochastic component `xi`. This project
  has only tried the deterministic half (`mode=two_step`/`history`,
  Sections 198/201) -- a stochastic latent propagator (sampled noise
  injected per step, not just at rollout start) has never been built or
  tried on either KS or L96. If longer history (Section 201) does not
  close the D_KY gap to L96's true 11.3, this is the next structurally-
  motivated thing to try, not a bigger/deeper deterministic network.
- **Is latent-space DA even the right framing for L96?** The spectral-gap
  finding means L96's own theoretical motivation for a *small* latent
  reduction is weaker than KS's (no inertial manifold to exploit). L96 is
  also the field's standard testbed for *covariance localization*
  specifically because it is high-dimensional but only locally correlated
  in its native (unreduced) coordinates -- classical LETKF/EnKF already
  handles that case well without any dimension reduction. Worth an
  explicit decision at some point: is the L96 line's deliverable "does our
  latent method also work here" (comparable framing to KS), or "how much
  does a system need an inertial manifold before latent DA beats
  classical localized DA" (a different, arguably more useful question,
  answered either way regardless of whether L96's own D_KY number ever
  closes the gap to 11.3)? Not yet decided.
- **Rayleigh-Benard: what aspect ratio/Ra gives genuine chaos?** The
  solver is validated (critical-Ra test, linear-growth-rate test both
  pass), but the first real dataset (Ra=3e4, aspect ratio locked to the
  validation box's `2*sqrt(2)`) settled into an exact steady 2-roll
  state, not the chaotic/oscillatory regime requested. A quick check at
  4x wider aspect ratio showed a much slower, still-unresolved transient
  (not yet confirmed chaotic either). Open: sweep aspect ratio and/or Ra
  (a longer run, or several short ones) to find where genuine sustained
  time-dependence sets in, before generating a full training dataset.
  See `docs/RESULTS.md`'s "Rayleigh-Benard convection: solver build" entry.
- **Peyron et al.'s "augmented" Lorenz-96 variant** (QJRMS 2021,
  arXiv:2104.00430) has not been read in detail -- only its abstract-level
  framing (tested on an L96 variant constructed to have latent structure,
  not raw L96) is recorded in `docs/RESULTS.md`. If the L96 line continues,
  reading the actual augmentation they used (and whether it's a fair
  comparison point or a fundamentally different question) is worth doing
  before citing it further.
- **Section 203 (idea 1 + idea 2 combined: `--w-spectrum-shape` +
  `--multistep`, both on markovian) found Stage 1 ITSELF collapsed**
  (`D_KY=0.00, n_positive=0/20`, `val_recon_final=0.0025` -- trained
  cleanly, no nan/errors, just landed non-chaotic) before Stage 2 ever
  ran. Every prior bare-Stage-1 L96 run produced genuine chaos (199:
  1.35, 201: 11.94), so this is a new, unexpected regression. Likely
  cause (not yet confirmed): `--multistep` extends STAGE 1's own
  auxiliary-propagator loss to an 8-step rollout -- exactly the kind of
  longer-horizon multi-step MSE term already diagnosed as the Stage-2
  collapse mechanism. If so, the working hypothesis "pushing rollout-
  fitting into Stage 1's jointly-regularized regime is safer than
  Stage 2's propagator-only regime" was wrong, or at least insufficient
  at this rollout length -- collapse pressure from long-horizon MSE may
  operate inside Stage 1's own objective too, not just Stage 2's. Killed
  before Stage 2 ran (uninformative to continue from an already-
  collapsed Stage 1); user paused this line 2026-09-23 to return to the
  Rayleigh-Bénard work. **Next step when resumed**: rerun Section 203
  WITHOUT `--multistep` (keep only `--w-spectrum-shape`) to isolate
  whether spectrum-shape alone preserves Stage-1 chaos and then also
  protects Stage 2 -- the original idea 1, tested in isolation.
- **Section 203 RERUN (2026-09-23, `--multistep` removed) confirms the
  `--multistep` diagnosis for Stage 1, but Stage 2 still collapsed even
  with a dedicated anti-collapse regularizer stack.** Stage 1 (ViT,
  markovian, `--w-spectrum-shape` n_expand=5/target=1.1/floor=0.6/
  two_sided, `--w-var 0.02`, `--w-spatial 0.01` signed, `--w-logdet
  0.0035`, NO `--multistep`) recovered to `D_KY=2.117, n_positive=1/20,
  lambda1=0.00315` -- confirms `--multistep` (not spectrum-shape, not the
  other Stage-1 regularizers) was the cause of the first attempt's
  Stage-1 collapse (`D_KY=0.00`), and is better than the bare Section 199
  baseline (`D_KY=1.35`). Stage 2 was then warm-started from this good
  checkpoint with `--w-varmatch 0.02` adaptive + `--w-spatial 0.01` signed
  + the newly-built `--w-logdet-rollout-latent 0.0035` (general-backbone
  logdet anti-collapse term on the propagator's rolled-out `z_pred`,
  built specifically for this rerun -- see
  `scripts/section203_lorenz96_stage2_geometry_regularizers.sh` header) --
  and STILL collapsed: `D_KY=0.0, n_positive=0/20, lambda1=-0.00235`
  (slightly negative). This is the **fourth** independent confirmation of
  Stage-2 collapse (197, 199, 201, 203) and the **first time it survived a
  regularizer stack purpose-built to prevent it** -- `w_varmatch`,
  `w_spatial`, and `w_logdet_rollout_latent` together were not enough.
  Narrows the remaining untried candidate to the pde_head
  self-rollout/spectrum-shape regularizer applied directly to Stage 2's
  *main* propagator (see next entry) -- everything else in the current
  regularizer toolkit has now been tried and failed on L96.
- **Stage-2 collapse-to-zero is now confirmed a structural property of
  the training objective, not a fixable-by-better-inputs symptom, and not
  fixable by the variance/spatial/logdet regularizer toolkit tried so
  far.** Section 201 (2026-09-23) started Stage 2 from a Stage-1
  checkpoint whose D_KY=11.94 essentially matched the true system's 11.3,
  and Stage 2 still collapsed it to exactly 0.00; Section 203's rerun
  (above) then showed that adding `w_varmatch` + `w_spatial` +
  `w_logdet_rollout_latent` on top of a good Stage-1 checkpoint (D_KY=
  2.117) *still* collapses to 0.00. Four independent confirmations total
  (197, 199, 201, 203; local_field and ViT encoders; 0 and 5 steps of
  propagator history; with and without geometry regularizers) that pure
  k-step supervised `horizon_weighted_latent_loss` training destroys
  autonomous chaos regardless of how good the dynamics were beforehand or
  what anti-collapse terms are added on top. Never tried: applying the
  pde_head self-rollout/spectrum-shape regularizers (Sections 192-193,
  built to discourage Jacobian contraction along a self-generated
  rollout, so far only used on the separate `pde_head` distillation
  target) directly to Stage 2's *main* propagator, on either KS or L96.
  This is now the single remaining direct candidate fix in the existing
  toolkit and has not been attempted on either system.
- **`--w-spectrum-shape-self` (the candidate above) was finally attempted
  (Section 205, L96 N=16, x+x' augmented state, mode=history) but the
  test was confounded: Stage 1 ITSELF collapsed under the ViT encoder/
  propagator** (`D_KY=0.00, n_positive=0/8`, despite healthy
  reconstruction, `val_recon_final=0.0836`) before Stage 2 (where
  `w_spectrum_shape_self` actually applies) ever ran; Stage 2 was
  cancelled rather than run against an already-collapsed input. Section
  204 (identical setup, plain MLP encoder/decoder/propagator instead of
  ViT) was killed mid-Stage-1 by user direction before it could serve as
  the controlled comparison, though its own quick 2-epoch dry run showed
  no immediate collapse signal (not enough epochs to be conclusive, but a
  contrast worth noting against Section 205's decisive negative). This
  points at the ViT-backbone propagator itself, not the regularizer
  stack, as the likely proximate cause, consistent with `ks_latent.
  models.propagator._LocalMLPDeltaBody`'s own documented KS-side finding
  ("every self-attention-based propagator tried... collapsed... every
  architecture WITHOUT self-attention... recovered rich chaos"). **Next
  step if this line resumes**: finish/rerun Section 204's plain-MLP
  variant to completion (same dataset/regularizers) as the actual
  controlled test of `--w-spectrum-shape-self` -- it still has not been
  tested against a Stage 2 that starts from a genuinely chaotic Stage-1
  checkpoint.
- **Correction to the entry above: the "ViT propagator collapses"
  diagnosis was wrong for this experiment.** Section 205 had used
  `--aux-backbone vit` by mistake (a misreading of "replace the encoder
  and decoder with the ViT" -- `--aux-backbone` is a separate flag from
  `--encoder`, caught directly by the user: "wait I thought we were
  using the mlp as the propagator?"). Section 206 reran with the
  intended `--encoder vit --aux-backbone mlp` and got the SAME slow,
  poorly-converging reconstruction curve as Section 205 (epoch 0/10/20/
  30 recon 0.43/0.15/0.10/0.09 either way, vs. Section 201's 0.11/0.004/
  0.002/final 0.003 at the SAME `--aux-backbone vit --mode history`).
  Since both propagator backbones inherit the identical bad Stage-1
  latent, the propagator is not the shared bottleneck; both Section 205
  and 206 are uninformative about `--w-spectrum-shape-self` for the same
  underlying reason (bad Stage 1), not because of the propagator choice.
  Likely real cause (see `docs/RESULTS.md`'s correction entry for the
  full reasoning): concatenating `x`/`x'` into one flat 32-dim vector
  before ViT patch-tokenization gives the encoder no structural signal
  that tokens 2-3 are "the same sites, a different quantity" rather than
  "further along the ring" -- confounded with N=16's fewer tokens (4 vs.
  201's 8) and `d_latent=8` (vs. 20), none separated in this experiment.
  Section 206 killed (user-directed) before completion. **Revised next
  step if this line resumes**: encode `x`/`x'` as two CHANNELS per site
  (shape `(N,2)`) instead of concatenating into a longer sequence,
  before re-attempting `--w-spectrum-shape-self` on this operating
  point -- Section 204's plain-MLP variant (which does not tokenize by
  patch and so may not share this specific failure mode) remains the
  other untried, cheaper thing to finish first.
