# Literature Review & Findings Summary

**Status: draft, started 2026-09-23.** Part 1 summarizes this project's own
results across its three test systems (KS, Lorenz-96, Rayleigh-Bénard).
Part 2 is a literature review of directly related prior work, organized by
theme. Citations marked `[verify]` have not yet been independently
confirmed; everything else was checked against a primary source (arXiv,
journal page, or DOI) before being included here.

---

## Part 1: Findings Summary

### 1. Kuramoto-Sivashinsky (KS)

**System.** `u_t + u u_x + u_xx + u_xxxx = 0`, periodic domain, solved with
ETDRK4 (`ks_latent/solver/ks.py`). Canonical config `L=100`, and — after a
documented, deliberate deviation from the original handoff spec —
`NX=256` rather than `NX=1024` for the latent-modeling phases specifically
(the solver itself was validated at `NX=1024`; `NX=256` was adopted only
for AE/propagator training after `NX=1024` reliably collapsed within
budget and `NX=256` did not).

**Phase 1-2 gates (replication).** Solver validated against the known
`D_KY` scaling law at `L=100` (target 21-24) and `L=22` (target 5.2-5.6);
both passed. Light-cone/spreading-velocity bound (D7) measured and used
to set stencil-width design rules for later phases.

**The architecture search (the bulk of this project's KS work).** The
original patched-transformer autoencoder reliably collapsed to a
near-constant latent representation during Stage-1 training at the
canonical scale. A long comparative search across encoder architectures
found:
- A plain MLP encoder/decoder ("Track A") eliminated the collapse
  entirely and reduced validation reconstruction MSE by ~146x versus the
  patched-transformer baseline, with no collapse plateau in the training
  curve at all.
- A ViT-style encoder with an explicit circular positional encoding (the
  patched-transformer had none) did even better (~1,482x lower MSE than
  the baseline), implicating the ORIGINAL architecture's specific design
  (not "transformers in general," since the ViT variant also uses
  attention) as the root cause of the collapse.
- Windowed/local attention (`attn_window`) and periodic vs. non-periodic
  positional encoding were both swept; a linear (non-periodic) positional
  encoding performed statistically indistinguishably from the circular
  one, and was adopted as canonical specifically because it does not
  depend on periodicity and would generalize further to non-periodic
  target systems (directly relevant to the later Rayleigh-Bénard work).
- **Canonical recipe as of this writing**: `vit` encoder, `attn_window=4`,
  `pos_encoding="linear"`, dense-MLP markovian propagator.

**The Stage-2 collapse phenomenon (a major, independently-rediscovered
finding).** Across many regularizer/architecture variants, a consistent
pattern emerged: Stage 1 (encoder + propagator trained jointly, with
reconstruction and geometry regularizers) can produce a propagator with
genuine, healthy chaos (Jacobian singular values properly distributed,
Lyapunov spectrum matching the true system), but Stage 2 (propagator-only
fine-tuning on pure k-step supervised MSE, no competing objective)
reliably collapses that chaos toward a fixed point or a simple
limit-cycle-like state — while simultaneously IMPROVING short/medium-term
forecast accuracy. This was diagnosed as a structural property of pure
MSE-based multi-step supervised loss (it has no term rewarding sensitivity
to initial conditions, and any residual model imperfection is penalized
more if amplified by chaos than if damped, creating a systematic bias
toward contraction) rather than a bug in any one recipe. A dedicated
regularizer (`w_spectrum_shape`, shaping the propagator's own per-step
Jacobian singular-value spectrum toward a target split of expansive/
contracting directions) was built specifically to counter this and
validated on KS.

**pde_head / spectral-PDE line.** A parallel research thread attempted to
fit an explicit, interpretable local PDE/stencil form to the learned
latent dynamics (`backbone="spectral_pde"`/`"spectral_pde_raw"`,
polynomial and Fourier-based propagator bodies, "PDE distillation" losses
tying a separate interpretable head to the main propagator). Produced a
family of spectrum-shape and energy-floor regularizers later reused
directly in the Lorenz-96 line.

### 2. Lorenz-96 (L96)

**System.** `dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`, a periodic
ring of N coupled ODEs (Lorenz 1996), built as a second chaotic testbed to
test generalization of the KS-line findings. Solver validated against
exact analytic identities (fixed point, energy conservation, trace
identity, closed-form fixed-point spectrum) and the classic N=40/F=8
literature Lyapunov exponent.

**Extensivity confirmed.** `n_positive/N` and `D_KY/N` ratios at F=8
nearly match between N=64 and N=256, directly confirming L96 is a
genuinely extensive chaotic system, the same qualitative property KS has.

**F controls chaos strength non-monotonically.** A sweep at N=64 found
F=2,3 non-chaotic, F=4 barely chaotic, narrow periodic windows embedded
in the transition region (F=4.4, F=4.9 collapse to D_KY=0 despite
chaotic neighbors), F=8 fully turbulent. F=4.2 was selected as this
project's primary L96 operating point specifically because its true
D_KY≈11.3 lands in a comparable range to KS's own replication targets.

**Headline finding: L96 lacks KS's spectral gap.** Direct measurement of
the FULL Lyapunov spectrum of both TRUE systems (no neural network) at
matched state dimension (N=64) found KS's spectrum has a sharp gap (sum
of all 64 exponents = -4180, a few near-neutral directions then a
steeply diving tail to -96) while L96's does not (sum = exactly -64 by
construction, decaying almost linearly to only -3.5). Mechanism: KS's
linear operator (`k^2 - k^4`) is wavenumber-selective, damping high-k
modes as the fourth power of k -- the literal content behind KS's known
finite-dimensional inertial manifold. L96's only linear term is a
spatially uniform `-x_i` damping, with no analogous scale-selective
mechanism. **Practical consequence**: a small-`d_latent` MARKOVIAN
reduction of L96 is not "harder than KS" -- it may be asking for
something structurally unavailable, since there is no rigorous basis for
assuming any subset of L96's directions is dynamically disposable the
way KS's high-k tail is.

**d_latent-vs-true-D_KY sizing is a dominant lever.** Undersized
`d_latent` relative to the true D_KY produces degenerate/unbounded
propagators; adequately-to-generously oversized `d_latent` is necessary
(not sufficient) for genuine bounded chaos.

**Memory/history closes most of the gap.** At matched `d_latent=20`,
increasing propagator history from 0 steps (`mode=markovian`, D_KY=1.35)
to 1 step (`mode=two_step`, D_KY=2.01) to 5 steps (`mode=history`,
`n_history=6`, D_KY=11.94) produced a STEP CHANGE, not a gradual trend --
closing almost the entire gap to the true system's D_KY=11.3. This is a
direct, empirical confirmation of the Mori-Zwanzig/spectral-gap
reasoning above (see Part 2.C).

**The extended state is not "just more dimensions."** A direct objection
(does `n_history=6 * d_latent=20 = 120` raw numbers just exceed the
original system's own N=64?) was checked via linear participation ratio
of the stacked history vector across real trajectories: raw dimension
120, but EFFECTIVE dimension only ~8.75 (vs. 5.87 for the raw N=64
physical state, 5.45 for a single encoded snapshot) -- consecutive
encoded states along a smooth chaotic flow are highly redundant, exactly
consistent with Takens delay-embedding theory. The raw parameter count
overstates the genuine information content by more than an order of
magnitude.

**Stage-2 collapse reproduces on L96, independent of architecture,
memory length, and Stage-1 chaos quality.** Confirmed three times
(local_field encoder/0 history, ViT/0 history, ViT/5 steps history) --
including once starting from a Stage-1 checkpoint whose D_KY=11.94
essentially MATCHED the true system, which still collapsed to exactly
0.00 after Stage 2. This is now a cross-system (KS + L96), cross-
architecture, cross-memory-length finding, strengthening the case that
it is a structural property of the k-step MSE training objective itself.
A fix attempt (porting KS's `w_spectrum_shape` regularizer to L96,
combined with extended Stage-1 rollout) is in progress as of this
writing (Section 203).

**Literature context and open strategic question.** L96 was literally
built by Lorenz as an NWP/DA testbed (its `F=8` deliberately chosen to
match real atmospheric predictability statistics), unlike KS, which does
not appear in the weather/DA literature. This raises a genuine open
question for the project (see Part 2.B/2.E): is latent-space DA even the
right framing for a system like L96, given it is also the field's
standard testbed for classical covariance LOCALIZATION specifically
because it needs no dimension reduction to be tractable?

### 3. Rayleigh-Bénard convection (RBC)

**System.** 2D Boussinesq convection, vorticity-streamfunction form, built
as a genuinely 2D, non-periodic-in-one-axis bridge system between L96 and
a real convection-resolving model (see Part 2.D). Free-slip/isothermal
boundary conditions chosen specifically so the whole solver reduces to a
pure FFT method (no Chebyshev machinery) and so the exact textbook
critical Rayleigh number (`Ra_c = 27*pi^4/4`, Chandrasekhar 1961) is
reachable for validation.

**Solver validated**: the real nonlinear pseudospectral solver reproduces
the analytic linear growth rate to 3% at both a sub- and super-critical
Ra, checked at the exact textbook critical wavenumber/Ra pair. 14 unit
tests, all passing.

**Two genuine numerical bugs found and fixed** en route to a stable
target-regime (Ra=3e4, Pr=0.7) integration: (1) explicit treatment of the
buoyancy coupling term is a hidden stiffness bug at large Ra (fixed via
an exact implicit 2x2 Crank-Nicolson solve of the full linear system);
(2) a boundedness safety check compared raw, transform-normalization-
dependent spectral coefficients against a fixed threshold, producing
false "blowup" errors on runs that were genuinely bounded in physical
space (fixed by checking the physical velocity field instead).

**Demonstration dataset generated and visualized** (5 train + 2 val
trajectories, 64x64, Ra=3e4/Pr=0.7) -- textbook-correct convection
structure (mushroom-shaped thermal plumes, counter-rotating rolls).

**Open finding**: at the validation aspect ratio (chosen to exactly match
the critical wavenumber, so only one pair of convection rolls fits), the
flow settles into an essentially EXACT steady state (relative variation
`~1e-8`), not the intended chaotic/oscillatory regime. A quick check at
4x wider aspect ratio showed a much slower, unresolved transient. Not yet
determined whether a wider box, higher Ra, or longer integration (or some
combination) is the right lever -- see Part 2.D for the relevant
literature on the onset of chaos in extended 2D convection (the "Busse
balloon").

### 4. Cross-cutting findings

1. **The spectral-gap/inertial-manifold criterion is a general, portable
   design rule**, independent of any one system: before assuming a
   small-latent Markovian reduction will work on a new target system,
   check whether its true dynamics have a wavenumber-selective (or more
   generally, scale-selective) linear damping mechanism. KS has one; L96
   does not; this directly explains why L96 needed history/memory in a
   way KS never did.
2. **Pure k-step supervised MSE training structurally destroys chaos**,
   confirmed across two systems, multiple architectures, and multiple
   propagator-memory lengths. This looks like a general property of the
   loss family, not a fixable-by-better-architecture symptom -- the
   productive directions are either an explicit anti-collapse
   regularizer (shape the propagator's own Jacobian spectrum) or a
   change of loss family entirely (a genuinely probabilistic/ensemble
   propagator scored with a proper scoring rule, not yet tried on either
   system).
3. **Reconstruction quality and dynamical fidelity are separate axes.**
   Repeatedly, an architecture change that dramatically improved
   reconstruction MSE (e.g., ViT vs. local_field on L96) did NOT improve,
   and sometimes hurt, the resulting propagator's intrinsic chaos --
   these two properties do not automatically track each other and must
   both be measured, not inferred from one another.

---

## Part 2: Literature Review

*(Verified 2026-09-23 against primary sources -- arXiv/DOI links below.
See individual entries in `docs/RESULTS.md` and `docs/OPEN_QUESTIONS.md`
for the specific conversational context each citation was found in.)*

### A. Chaotic PDE/ODE reduced-order & latent dynamics modeling

- Kassam, A.-K. & Trefethen, L. N. (2005), "Fourth-Order Time-Stepping
  for Stiff PDEs," *SIAM Journal on Scientific Computing* 26(4):1214-1233,
  [DOI:10.1137/S1064827502410633](https://doi.org/10.1137/S1064827502410633).
  The ETDRK4 integration scheme this project's own KS solver uses.
- Foias, C., Nicolaenko, B., Sell, G. R. & Temam, R. (1988), "Inertial
  manifolds for the Kuramoto-Sivashinsky equation and an estimate of
  their lowest dimension," *Journal de Mathématiques Pures et Appliquées*
  67:197-226 (earlier 1985 French-language precursor: "Variétés
  inertielles pour l'équation de Kuramoto-Sivashinsky"). Established KS's
  finite-dimensional inertial manifold; the rigorous mathematical content
  behind why a small Markovian latent reduction can work for KS in
  principle (see Part 1.2's spectral-gap finding).
- Edson, R. A., Bunder, J. E., Mattner, T. W. & Roberts, A. J. (2019),
  "Lyapunov exponents of the Kuramoto-Sivashinsky PDE," *ANZIAM Journal*.
  KS extensivity/Kaplan-Yorke dimension scaling result (`D_KY ~
  0.226*L`), this project's own Phase 1 replication target.
- Bar-Sinai, Y., Hoyer, S., Hickey, J. & Brenner, M. P. (2019), "Learning
  data-driven discretizations for partial differential equations," *PNAS*
  116(31):15344-15349,
  [DOI:10.1073/pnas.1814058116](https://doi.org/10.1073/pnas.1814058116).
  Learned finite-difference stencils on deliberately coarse grids; the
  closest existing work to this project's own (unbuilt) Phase 11
  stencil-propagator plan.
- Kochkov, D., Smith, J. A., Alieva, A., Wang, Q., Brenner, M. P. &
  Hoyer, S. (2021), "Machine learning-accelerated computational fluid
  dynamics," *PNAS* 118(21),
  [DOI:10.1073/pnas.2101784118](https://doi.org/10.1073/pnas.2101784118).
  Same lineage as Bar-Sinai et al.
- Constante-Amores, C. R., Linot, A. J. & Graham, M. D.,
  [arXiv:2410.01238](https://arxiv.org/abs/2410.01238), "Data-driven
  prediction of large-scale spatiotemporal chaos with distributed
  low-dimensional models" -- distributed/patch-decomposed low-dimensional
  models tested on KS and related spatiotemporal chaos.
- Constante-Amores, C. R., Linot, A. J. & Graham, M. D.,
  [arXiv:2408.03135](https://arxiv.org/abs/2408.03135), "Dynamics of a
  Data-Driven Low-Dimensional Model of Turbulent Minimal Pipe Flow"
  (Re=2500) -- the minimal-flow-unit contrast this project uses for its
  own positioning (Part 2.E). Note: the specific `-Ah` damping-stabilizer
  detail attributed to this paper in earlier project notes has NOT been
  independently confirmed from the abstract/search results alone --
  verify against the paper's own text before restating that detail.
- Wittenberg, R. W. & Holmes, P. (1999), "Scale and space localization in
  the Kuramoto-Sivashinsky equation," *Chaos* 9(2):452,
  [DOI:10.1063/1.166419](https://doi.org/10.1063/1.166419). KS is
  localized in BOTH real and Fourier space; a pre-neural-network
  precedent for this project's own local-latent-field research
  direction.

**Linot & Graham (UW-Madison; Linot now UCLA Mechanical & Aerospace
Engineering) -- the closest existing prior-art lineage to this project's
own methodology,** user-directed literature search 2026-09-23 (verified
against arXiv/journal pages). Their whole program is essentially this
project's own approach (autoencoder dimension reduction + a learned
ODE/map on the resulting latent coordinates, applied to KS and related
spatiotemporal chaos) predating it by several years -- reading these
papers directly, not just their abstracts, before any future publication
claim is essential.

- Linot, A. J. & Graham, M. D. (2020), "Deep learning to discover and
  predict dynamics on an inertial manifold," *Phys. Rev. E* 101:062209,
  [arXiv:2001.04263](https://arxiv.org/abs/2001.04263). Hybrid linear +
  nonlinear (autoencoder) dimension reduction onto KS's own inertial
  manifold, with translation invariance and energy conservation built
  into the formalism; substantially outperforms linear reduction alone.
  **The single most directly relevant prior paper to this project's own
  KS/L96 "inertial manifold / spectral gap" finding** (`docs/RESULTS.md`)
  -- read this one first if reading only one.
- Linot, A. J. & Graham, M. D. (2022), "Data-driven reduced-order
  modeling of spatiotemporal chaos with neural ordinary differential
  equations," *Chaos* 32:073110,
  [arXiv:2109.00060](https://arxiv.org/abs/2109.00060). Autoencoder finds
  manifold coordinates, a neural-ODE learns the dynamics on them; applied
  to KS at multiple domain sizes, finds dimension reduction improves
  forecast performance relative to ambient-space prediction. Already an
  indirect influence on this codebase before this literature search: its
  L1-vs-L2 latent-loss finding motivated `Stage2TrainingConfig.
  latent_loss` (`ks_latent/config.py`).
- Pérez De Jesús, C. E., Linot, A. J. & Graham, M. D. (2024), "Building
  symmetries into data-driven manifold dynamics models for complex
  flows: application to two-dimensional Kolmogorov flow," *Phys. Rev.
  Fluids*, [DOI:10.1103/ts3k-flx6](https://doi.org/10.1103/ts3k-flx6),
  [arXiv:2312.10235](https://arxiv.org/abs/2312.10235). "Symmetry
  charting" -- building continuous translation and discrete
  rotation/shift-reflect symmetries directly into the autoencoder +
  neural-ODE framework for 2D Kolmogorov flow. Directly relevant to this
  project's own 2D Rayleigh-Bénard work: a template for how a genuinely
  2D chaotic flow's known symmetries (RBC has periodic-x translation,
  same as this project's own solver) could be built into a future
  encoder rather than left to shift-augmentation alone.
- Zeng, K., Pérez De Jesús, C. E., Fox, A. J. & Graham, M. D. (2023),
  "Autoencoders for discovering manifold dimension and coordinates in
  data from complex dynamical systems," *Machine Learning: Science and
  Technology* (IOP),
  [arXiv:2305.01090](https://arxiv.org/abs/2305.01090). Same lab, not a
  Linot paper. An autoencoder architecture (implicit regularization +
  internal linear layers + weight decay) that AUTOMATICALLY estimates
  the correct latent dimension from data, rather than it being chosen a
  priori. Directly relevant to this project's own "d_latent vs. true
  D_KY sizing is a dominant lever" finding on L96 (Part 1.2) -- this
  paper is a candidate method for doing that sizing principled rather
  than by sweep.
- Guo, A. & Graham, M. D. (2025), "Blending data and physics for
  reduced-order modeling of systems with spatiotemporal chaotic
  dynamics," [arXiv:2507.21299](https://arxiv.org/abs/2507.21299). Same
  lab, not a Linot paper. A hybrid physics+data reduced-order model
  (PI-DManD): the full-order vector field is projected onto the
  autoencoder-discovered invariant manifold and then corrected by data
  (or used as a Bayesian prior updated with data), tested on KS and the
  complex Ginzburg-Landau equation; the hybrid approach roughly halves
  the one-Lyapunov-time forecast error versus the data-only method.
  Directly relevant to this project's own open question about whether a
  stochastic/hybrid closure term (Part 2.C's Mori-Zwanzig `Phi + xi`
  decomposition) could resolve the Lorenz-96 memory requirement more
  principled than pure history-length sweeps.
- Vinograd, M. Y. & Clark di Leoni, P. (2024), "Reduced Representations
  of Rayleigh-Bénard Flows via Autoencoders," *Journal of Fluid
  Mechanics*,
  [arXiv:2410.01496](https://arxiv.org/abs/2410.01496). NOT a Linot/
  Graham-group paper, found via the same search. Convolutional
  autoencoders on the 2D RBC temperature field at Pr=1, Ra from 10^6 to
  10^8 (turbulent regime, well beyond this project's own Ra=3e4). The
  direct precedent for "autoencoder dimension reduction applied to RBC
  specifically" -- worth reading before any further RBC latent-modeling
  work, both for its own findings and for what Ra/resolution regime it
  considers tractable.

### B. Latent-space data assimilation

- Peyron, Fillion, Gürol, Marchais, Gratton, Boudier & Goret (2021),
  "Latent Space Data Assimilation by using Deep Learning," QJRMS,
  [arXiv:2104.00430](https://arxiv.org/abs/2104.00430). ETKF-Q in a
  learned latent space, tested on an AUGMENTED Lorenz-96 system
  specifically constructed to possess exploitable latent structure --
  not raw L96. The direct precedent for this project's own "latent DA +
  L96" question; the augmentation detail is itself evidence that raw L96
  does not hand a latent-DA method exploitable structure for free.
- Chen, Sanz-Alonso & Willett, "Reduced-Order Autodifferentiable
  Ensemble Kalman Filters,"
  [arXiv:2301.11961](https://arxiv.org/pdf/2301.11961) (ROAD-EnKF).
  Tests Lorenz-63 (embedded in high dimension), Burgers, and KS -- NOT
  Lorenz-96, correcting an earlier assumption. KS keeps appearing as the
  natural target for this method class in the literature; L96 does not,
  plausibly for the same structural (spectral-gap) reasons this project
  measured directly.
- "Latent Auto-encoder Ensemble Kalman Filter for Nonlinear Data
  Assimilation," [arXiv:2603.06752](https://arxiv.org/html/2603.06752) --
  explicitly lists localization and inflation as future work.
- "Learning Enhanced Ensemble Filters,"
  [arXiv:2504.17836](https://arxiv.org/html/2504.17836v3) -- reports
  beating an optimized LETKF on Lorenz-96, KS, and Lorenz-63.
- "Physically Consistent Global Atmospheric Data Assimilation with
  Machine Learning in Latent Space," Science Advances,
  [arXiv:2502.02884](https://arxiv.org/pdf/2502.02884) -- real global-
  atmosphere-scale latent DA, well past this project's toy-system scale;
  evidence the overall approach is an active, currently-published
  direction.
- "A Novel Latent Space Data Assimilation Framework with Autoencoder-
  Observation to Latent Space (AE-O2L) Network," Monthly Weather Review
  153(8), 2025.
- "Data Assimilation in the Latent Space of a Convolutional Autoencoder,"
  Springer -- `[verify exact venue/citation]`, same core idea as this
  project's own program, applied to a (likely simpler) test system.

### C. Closure, memory, and the Mori-Zwanzig formalism for reduced
   chaotic systems

- Lu, Lin & Chorin (the "optimal prediction"/discrete Mori-Zwanzig line,
  Chorin group, ~2015-era) -- `[verify exact citation]`. Formally derived
  that a correct discrete-time closure for Lorenz-96 needs a memory term
  with a deterministic history-dependent part AND a stochastic part.
  This project has only tried the deterministic-history half; the
  stochastic half is untested (see `docs/OPEN_QUESTIONS.md`).
- Xu & Chen, "Intrinsic Instantaneous Coarse-to-Fine Recoverability in
  the Lorenz-96 System,"
  [arXiv:2607.08323](https://arxiv.org/abs/2607.08323) (July 2026) -- an
  independent, very recent, information-theoretic confirmation of this
  project's own spectral-gap finding: L96's closure is strongly
  nonuniform, and recoverability DECREASES as F increases (more chaos,
  less closure).

### D. Weather/convection-relevant modeling

- Bryan, CM1 (Cloud Model 1), NCAR -- the standard idealized non-
  hydrostatic cloud-resolving model for 2D vertical-slice deep-convection
  studies (squall lines, thunderstorms); the likely target of the user's
  advisor's own stated research interest. Multi-field (velocity
  components, potential temperature, pressure perturbation, moisture
  species) and non-periodic in the vertical -- two axes of complexity
  this project's Rayleigh-Bénard build was designed to bridge toward
  before attempting CM1 directly.
- Weisman & Klemp (1982) -- the classic idealized sounding-plus-warm-
  bubble squall-line setup `[verify exact citation]`, standard technique
  for 2D idealized convection studies of the kind CM1 runs.
- "Comparing storm resolving models and climates via unsupervised
  machine learning," Scientific Reports (2023) -- a VAE reducing storm-
  resolving-model vertical-velocity fields to a low-dimensional latent
  space, finding it separates into distinct tropical-convection regimes;
  direct precedent for latent-space analysis of convection specifically.

### E. Positioning / prior-art warnings (what NOT to claim as novel)

Carried over from `CLAUDE_CODE_BRIEF.md` §22, restated here for the
literature review's own completeness:

- **"We show KS admits local reduced models" is not an available claim**
  -- Wittenberg & Holmes (1999) and the associated wavelet-projection
  literature made essentially this argument ~25 years earlier by
  different means. This project's own claim must be narrower: "a
  LEARNED local latent field enables localization and transfer in
  nonlinear latent DA."
- The minimal-flow-unit contrast (Constante-Amores/Linot/Graham,
  arXiv:2408.03135) is this project's own one-sentence positioning
  statement: a minimal flow unit is non-extensive by construction, which
  is exactly why a global latent vector works there and does not scale
  to KS/L96/RBC's own extensive regimes.
