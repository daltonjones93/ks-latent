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

**Whitney/Takens/Sauer-Yorke-Casdagli -- the classical embedding
theorems, and the precise statement behind "D_KY = n implies embedding
dimension 2n+1."** User-directed 2026-09-23: Peter Jan reportedly
believes he discovered this relationship. He did not -- it is one of
the most-cited results in all of nonlinear dynamics, dating to 1981
(generalized to fractal attractors in 1991):

- Takens, F. (1981), "Detecting Strange Attractors in Turbulence," in
  D. Rand & L.-S. Young (eds.), *Dynamical Systems and Turbulence,
  Warwick 1980*, Lecture Notes in Mathematics vol. 898,
  Springer-Verlag, pp. 366-381. The original embedding theorem: for a
  smooth dynamical system whose attractor is a compact manifold of
  (integer) dimension `d`, a delay-coordinate map built from `m >= 2d+1`
  generic time-delayed samples of a generic scalar observable is,
  generically, an embedding (i.e. a smooth, invertible reconstruction of
  the full attractor and its dynamics, up to a diffeomorphism) -- the
  literal origin of the "2d+1" rule. `Takens's theorem` on Wikipedia is
  a reasonable non-paywalled starting point for the statement.
- Sauer, T., Yorke, J. A. & Casdagli, M. (1991), "Embedology," *Journal
  of Statistical Physics* 65:579-616,
  [DOI:10.1007/BF01053745](https://doi.org/10.1007/BF01053745). The
  generalization that makes the rule apply to genuinely fractal/chaotic
  attractors (Takens's original theorem assumes a smooth manifold, i.e.
  an integer dimension): for a compact set `A` with box-counting
  dimension `d_box(A)`, if `n` is an integer STRICTLY GREATER THAN
  `2*d_box(A)`, then almost every delay-coordinate map (in the sense of
  prevalence) is an embedding of `A`. The smallest such integer is
  `floor(2*d_box(A)) + 1` -- "2n+1" is the standard safe rounding of
  this when `d_box(A) = n` is itself an integer or close to one, exactly
  the form of the claim in question.
- Whitney, H. (1936), "Differentiable Manifolds," *Annals of
  Mathematics* 37(3):645-680. The purely topological precursor (smooth
  compact `d`-manifolds embed in `R^(2d+1)`, with no dynamical-systems
  content) that both of the above build on.

**The precise caveat that matters for using this with Kaplan-Yorke
dimension specifically**: the theorems above are stated in terms of
BOX-COUNTING (or a closely related fractal) dimension, not the
Kaplan-Yorke dimension `D_KY` this project computes from the Lyapunov
spectrum. `D_KY` is used as a *practical proxy* for the attractor's true
fractal/information dimension via the Kaplan-Yorke conjecture (Kaplan &
Yorke, 1979) -- empirically well-supported and used throughout this
project's own replication targets, but a conjecture, not a theorem, and
known to fail in constructed counterexamples. So the fully precise
chain is `D_KY ~= d_box` (conjectured) `=> embedding dimension > 2*D_KY`
(Sauer-Yorke-Casdagli) `=> 2*D_KY + 1 is a safe integer choice`
(rounding) -- three well-established steps, not one, but none of them
new. **This project's own results already independently corroborate the
same relationship empirically, without originally citing this as the
reason**: Section 201's L96 result (Part 1.2) -- 5 steps of history
closing the chaos gap where 0-1 steps failed -- is a direct delay-
embedding experiment in exactly this sense (stacking time-lagged copies
of a lossy `d_latent`-dimensional observable to reconstruct the full
dynamics), and the "d_latent must comfortably exceed true D_KY" sizing
lever found across this project's whole L96 line is the single-snapshot
special case of the same theorem.

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

**Racca & Magri (Cambridge/Imperial, MagriLab) -- the direct prior-art
answer to "is matching the true system's Lyapunov exponents/Kaplan-Yorke
dimension in a learned latent space novel?"** User-directed search
2026-09-23 (a collaborator's advisor believed this specific result --
recovering the true system's own Lyapunov dimension in a latent/reduced
representation -- was a novel finding; it is not, and has not been for
several years). See Part 2.E for the direct statement of this.

- Racca, A., Doan, N. A. K. & Magri, L. (2023), "Predicting turbulent
  dynamics with the convolutional autoencoder echo state network,"
  *Journal of Fluid Mechanics* 975:A2,
  [DOI:10.1017/jfm.2023.716](https://doi.org/10.1017/jfm.2023.716),
  [arXiv:2211.11379](https://arxiv.org/abs/2211.11379). A convolutional
  autoencoder finds a low-dimensional latent representation of KS (L=22
  -- this project's own second Phase-1 replication target), an echo
  state network propagates the dynamics on it, and the paper explicitly
  reports that this pipeline "accurately infers the Lyapunov exponents
  and covariant Lyapunov vectors (CLVs) in this low-dimensional manifold
  for different attractors." This is precisely the claim in question --
  reproducing the true system's own Lyapunov spectrum (and hence
  Kaplan-Yorke dimension) from a learned latent representation -- done,
  published, peer-reviewed, on the exact system (KS) this project itself
  uses.
- Özalp, E. & Magri, L. (2025), "Stability analysis of chaotic systems in
  latent spaces," *Nonlinear Dynamics* 113:13791-13806,
  [DOI:10.1007/s11071-024-10712-w](https://doi.org/10.1007/s11071-024-10712-w),
  [arXiv:2410.00480](https://arxiv.org/abs/2410.00480). A direct
  follow-up, entirely dedicated to this question (latent-space stability/
  Lyapunov analysis of chaotic systems, again including KS), from the
  same group. Removes any doubt that recovering Lyapunov dimension in a
  latent space is an open or novel question as of this writing.

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

- **"Recovering the true system's own Lyapunov exponents / Kaplan-Yorke
  dimension in a learned latent space is a novel finding" is NOT an
  available claim, and should not be presented as one in any paper or
  talk.** Directly answers a question raised 2026-09-23: a collaborator's
  advisor believed their own replication of this result (matching
  Lyapunov dimension between a chaotic system and its latent
  representation) was novel. It is not -- **Racca, Doan & Magri (2023),
  *JFM* 975:A2** (Part 2.A above) explicitly reports "accurately infers
  the Lyapunov exponents and covariant Lyapunov vectors ... in this
  low-dimensional manifold," on KS, published and peer-reviewed two-plus
  years before this note was written, with **Özalp & Magri (2025),
  *Nonlinear Dynamics*** as a direct, entirely-dedicated follow-up on
  the same question. This project's OWN replication targets
  (`CLAUDE.md`/`docs/REPLICATION_LOG.md`: "Latent D_KY (Benettin,
  single-state) ~= 21.4") already implicitly assume matching Lyapunov
  dimension in a latent space is achievable and is being used as a
  correctness CHECK on this project's own pipeline, not presented
  anywhere as this project's own novel contribution -- that framing is
  correct and should be kept. **If a paper or talk is being prepared
  that presents latent-space Lyapunov-dimension matching itself as the
  novel result, stop and re-scope it before proceeding** -- the novel
  content, if any, has to be in what is DONE WITH that matching (this
  project's own strongest candidate, per Part 3.2-3.4, is the
  spectral-gap-as-predictor and cross-system generalization angles, not
  the matching itself).
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
- **The Linot/Graham lineage (Part 2.A) is close enough to this
  project's own approach that overlap risk is real, not hypothetical.**
  Linot & Graham (2020) already reduces KS onto its inertial manifold
  with an autoencoder; this project's own "KS has a spectral gap, hence
  a small Markovian reduction can work" framing may already be present,
  explicitly or implicitly, in that paper's own reasoning. **Do not
  claim the spectral-gap/inertial-manifold argument as novel until the
  full text (not the abstract) of Linot & Graham (2020) and Pérez De
  Jesús, Linot & Graham (2024) have been read.** See Part 3 below for
  what is more likely to survive that check.

---

## Part 3: Recommended Research and Publication Directions

Written after the literature search above, calibrated against it --
these are ranked by how much of each idea's novelty survives the
Linot/Graham overlap risk just flagged, not by how interesting each
idea is in isolation. **None of these are ready to write up yet**: the
two most promising threads (spectral-gap-as-predictor, Stage-2 collapse)
both have an unresolved experiment sitting in `docs/OPEN_QUESTIONS.md`
that needs to land before the finding is actually a finding.

### 3.1 Do this first, before any of the below: read four papers in full

Not a research direction, a prerequisite. Abstracts were verified for
Part 2's citations; full texts were not read. Before drafting anything
publication-shaped:

1. Linot & Graham (2020) -- checks whether "KS's inertial manifold is
   *why* small-Markovian-latent reduction works, and this is testable by
   comparing against a system that lacks one" is already their own
   framing.
2. Pérez De Jesús, Linot & Graham (2024) -- checks whether the
   symmetry-charting approach already addresses what this project's own
   local-latent-field plan (`CLAUDE_CODE_BRIEF.md` Phase 10) was going
   to attempt, which would change that phase from "build it" to "apply
   their method."
3. Guo & Graham (2025) -- checks whether their hybrid physics+data
   correction already covers the ground this project's own "does L96
   need a stochastic closure term, not just more history" open question
   (Part 2.C, `docs/OPEN_QUESTIONS.md`) is asking.
4. Xu & Chen (2026) -- already read closely enough to cite specifics in
   `docs/RESULTS.md`, but re-read alongside Linot & Graham (2020)
   specifically to check whether ITS "coarse-to-fine recoverability"
   framing and this project's own spectral-gap framing are actually the
   same underlying idea in different language, which would change how
   the two should be cited relative to each other.

### 3.2 Most promising candidate: the spectral-gap criterion as a
    quantitative predictor, not just a qualitative pass/fail

**Current state**: a qualitative diagnosis (KS has a gap, L96 doesn't;
the gap's presence/absence tracks whether a small Markovian latent
reduction works at all) plus exactly one data point suggesting the gap's
ABSENCE predicts a quantitative need (Section 201: 5 steps of history
closed most of L96's chaos gap, at matched d_latent). That is a
demonstration, not yet a predictive theory.

**What would make this a real contribution**: a quantitative relationship
between a measurable property of the true system's spectral gap (e.g.
the ratio of the trace to the largest exponent, or the wavenumber at
which damping crosses some threshold) and the MINIMUM propagator memory
length needed to recover a target fraction of the true D_KY. Concrete
experiment: sweep `n_history` on L96 systematically (not just 0/1/5 as
done so far) at 2-3 different F values with different degrees of
"gaplessness," and check whether the memory length needed scales with
some closed-form function of each system's own measured spectrum shape.
If a clean relationship exists, this becomes a genuinely new, portable
design rule ("here is how much memory your propagator needs, computed
from the true system's linear operator alone, before you train
anything") -- publishable on its own, likely at a venue like *Chaos* or
*Physical Review Fluids/E* given the Linot/Graham precedent for exactly
those venues.

**Blocking**: needs the Section 201-style history sweep repeated at
several F values, and (per 3.1) needs the Linot & Graham (2020)/Xu & Chen
(2026) overlap check done first, since a version of "gap size predicts
required memory" may already exist in the Mori-Zwanzig / optimal-
prediction literature (Part 2.C) under different vocabulary
(memory-kernel decay rate vs. spectral gap).

### 3.3 Second candidate: Stage-2 collapse as a general critique of
    k-step MSE training for chaotic surrogate models, with a validated fix

**Current state**: the phenomenon (pure multi-step MSE propagator
training destroys measured chaos while improving forecast skill) is
demonstrated across two systems, several architectures, and multiple
propagator-memory lengths -- a genuinely broad empirical base. The
mechanistic explanation (MSE's asymmetric penalty: amplified error costs
more than damped error, biasing any imperfect model toward contraction)
is plausible but not independently verified against the literature.
**Section 203's own fix attempt is still unresolved** (`--multistep`
appears to have reintroduced the same collapse pressure inside Stage 1
itself -- see `docs/OPEN_QUESTIONS.md`), so there is not yet a validated
fix to report, only a validated problem.

**What would make this a real contribution**: (a) the isolation
experiment already queued in `docs/OPEN_QUESTIONS.md`
(`--w-spectrum-shape` alone, no `--multistep`) actually run and shown to
preserve chaos through Stage 2 on L96, not just KS; (b) a genuine
comparison against a non-MSE loss family -- a probabilistic/ensemble
propagator scored with a proper scoring rule (CRPS or an energy score),
motivated directly by the Mori-Zwanzig `Phi + xi` decomposition (Part
2.C) -- since a fix via loss-family change, if it works, is a stronger
and more general result than a fix via one more regularizer term; (c) a
literature check specifically for the "blurry forecast" / "regression to
the mean" phenomenon in ML weather forecasting (video prediction and
precipitation nowcasting have well-known versions of this under names
like "double penalty problem" -- not yet searched for this project, a
real gap in Part 2 as written) to confirm whether the MECHANISM claimed
here is already established elsewhere, even if the specific
cross-system/cross-architecture chaos-measurement evidence is new.

### 3.4 Third candidate: a three-system ladder (KS -> L96 -> RBC) as
    the paper's actual organizing structure

Rather than publishing the spectral-gap finding or the Stage-2 finding
in isolation, the strongest single narrative this project is positioned
to tell -- not obviously duplicated by anything found in Part 2 -- is
the deliberate three-system progression itself: a 1D periodic system
with a scale-selective linear damping (KS), a 1D periodic system without
one (L96), and a 2D system with a genuinely bounded (non-periodic)
axis (RBC), each chosen specifically to isolate one structural axis at a
time (spectral gap presence/absence, then dimensionality/boundary type),
with a consistent measurement protocol (Lyapunov spectrum matching, not
just forecast skill) applied throughout. Vinograd & Clark di Leoni
(2024) is the closest existing RBC-autoencoder precedent but works at a
single system (RBC only, Ra=1e6-1e8, no comparison against a
spectral-gap-bearing system); nothing found in Part 2 runs this specific
three-system comparative design. **Blocking**: RBC's own chaos regime is
still unresolved (`docs/OPEN_QUESTIONS.md`) -- there is no third rung on
the ladder yet, only a solver capable of building one.

### 3.5 Honest assessment of what is NOT yet a contribution

- The RBC solver itself, however carefully validated, is not
  independently publishable -- Vinograd & Clark di Leoni (2024) and the
  broader pseudospectral-RBC literature already establish the method
  class; this project's own solver is an implementation, not a new
  numerical method.
- "We built a latent model of KS/L96/RBC" alone is not a claim -- Linot
  & Graham's own line already does this, earlier, on KS specifically,
  with more architectural sophistication (symmetry-equivariant encoders,
  automatic dimension discovery) than this project has built.
- The d_latent-vs-true-D_KY sizing finding (Part 1.2) is very likely
  already implicit in, or superseded by, Zeng et al. (2023)'s automatic
  dimension-discovery method (Part 2.A) -- this project swept d_latent by
  hand; their method estimates it directly from data. Citing their
  method as the principled alternative to this project's own sweep-based
  approach is more defensible than presenting the sweep finding as novel.

### 3.6 Suggested venues, once one of 3.2-3.4 has a resolved result

*Chaos* (AIP) and *Physical Review Fluids*/*E* both have direct
Linot/Graham precedent for exactly this material (Part 2.A). *Machine
Learning: Science and Technology* (IOP) is the right venue if the
methodological/diagnostic-criterion framing (3.2) is the lead result
rather than the physical-systems framing. If the DA angle (Part 2.B) is
developed further (e.g. an actual latent-DA experiment on L96 or RBC,
not yet attempted -- `docs/RESULTS.md`), *QJRMS* is the direct venue
given the Peyron et al. (2021) precedent already sits there.

---

## Part 4: Assessment of Peter Jan's Proposed Directions

User-directed 2026-09-23: is anything Peter Jan proposes in
`docs/LATENT_PDE_RESEARCH_NOTES.md` (his own request for "a PDE that
models the latent variables," written 2026-08-28, updated 2026-09-12)
novel? That document is itself unusually self-aware about prior art --
it has its own risk table (its §8) and a 30-citation related-work log
(its §10) that already catch most of the overlap. What follows confirms
and sharpens that self-assessment against this document's own Part 2,
rather than re-deriving it from nothing.

### 4.1 Not novel

- **Interpretation A** (inertial-form neural ODE + SINDy on the latent)
  -- the source document itself calls this "low-risk and quick," i.e.
  applying existing Linot & Graham tooling (Part 2.A). Not claimed as
  novel there either.
- **Interpretation B's general framework** (manifold-learn an emergent
  spatial coordinate, then fit a PDE local in it) -- this is Kemeth,
  Bertalan, Thiem, Dietrich, Moon, Laing & Kevrekidis (2022), *Nat.
  Commun.* 13:3318, arXiv:2012.12738, which the source document itself
  calls "the central reference." The general idea is theirs.
- **The core local-latent-field architecture** (patch-decomposed field +
  shared-weight local stencil propagator, on KS) -- Constante-Amores,
  Linot & Graham (arXiv:2410.01238, already in Part 2.A) do this on KS
  *and* 2D Kolmogorov flow, "motivated explicitly by attractor dimension
  scaling linearly with domain size." The source document's own risk
  table marks this overlap "High -- read this first" and already narrows
  what might be left over to exactly the three items in 4.2 below.
- **Matching Lyapunov exponents/D_KY in a learned latent space** -- see
  Part 2.E: Racca, Doan & Magri (2023) and Özalp & Magri (2025) already
  did this. Worth noting explicitly: the source document itself already
  cites the Özalp & Magri paper (its §10, dated 2026-09-12, via its PMC
  ID) -- this was on record in this project's own history before the
  advisor's claim of novelty that prompted Part 2.E's entry, not a new
  finding from this literature search.
- **"If D_KY = n, the latent dimension should be `2n+1`-dimensional"**
  -- see Part 2.A's new entry above: this is Takens (1981) generalized
  to fractal attractors by Sauer, Yorke & Casdagli (1991), i.e.
  44-and-35-year-old results respectively, among the most-cited papers
  in all of nonlinear dynamics. Not a discovery -- textbook material,
  covered in essentially every time-series-embedding course and
  reference (e.g. Kantz & Schreiber, *Nonlinear Time Series Analysis*).
  Also already on this project's own reading list before this question
  was asked: `docs/LATENT_PDE_RESEARCH_NOTES.md` §10 itself lists
  "Whitney (1936, 1944); Takens (1981); Sauer–Yorke–Casdagli (1991) --
  embedding" under "Already in the project handoff."

### 4.2 Still open, ranked

1. **Latent-space localization for ensemble/particle-flow DA** --
   expanded into a full proposal in 4.3 below. The strongest candidate.
2. **The h-refinement multi-resolution consistency test** (source
   document §5.4): train the same local stencil law at 2-3 site
   spacings and check whether the *same* function fits all of them,
   after the usual h-scaling of the finite-difference operators -- a
   sharper, more falsifiable criterion for "is this actually a PDE"
   than anything found cited elsewhere. Provisional on confirming
   Constante-Amores/Linot/Graham didn't already run an equivalent
   consistency check, since their architecture is shared-weight by
   construction and might make this a natural thing for them to have
   checked too.
3. **L-transfer** (train at one domain size, run at a larger one with
   zero retraining): moderate confidence for the same reason as #2 -- a
   shared-weight patch architecture is transferable by construction, so
   arXiv:2410.01238 may already demonstrate something like this even if
   not framed the same way.
4. **KPZ/Burgers universality as an independent validation target for a
   learned latent PDE**: the physics (Yakhot 1981 onward) is decades
   old, not novel; using it as a ground-truth check on a *learned*
   latent operator specifically is a reasonable methodological choice,
   not a headline finding on its own.

### 4.3 Full proposal: latent-space localization for ensemble/particle-
    flow data assimilation

**Claim being tested**: a local latent field (a genuinely spatial latent
representation, not a flat vector) enables covariance/kernel
localization in latent-space DA, letting required ensemble size scale
with *local* per-site dimension rather than the full attractor
dimension -- and this specifically has NOT been demonstrated in the
latent-DA literature as of this writing. **Status as of 2026-09-23: all
three papers found during this search that plausibly could have scooped
this have now been read in full and ruled out** -- Chandravamsi et al.
(global manifold, different problem entirely: shock multimodality, not
localization), Guerrieri et al. (physical-space localization of a
different algorithm, no learned representation at all), and Pasmans et
al. (localization appears exactly once, as an unimplemented future-work
sentence, in a low-dimensional test regime where it wouldn't be
load-bearing anyway) -- see each entry below for the specific check.
This is the closest this proposal will get to a clean "as far as a
real, non-exhaustive search can tell" novelty statement; it is not a
substitute for a professional literature search before submission.

**Broader implication if this works** (user-directed 2026-09-23, asked
before any experiment was run -- worth having on record independent of
the outcome):

1. **It resolves a real tension in the current ML+DA literature, not
   just a KS/L96-specific gap.** Every latent-DA paper found in this
   review (Peyron et al.; ROAD-EnKF; LAE-EnKF; Pasmans et al. -- three
   of the four read in full, Part 2.B/4.3) uses a GLOBAL latent vector,
   because that is what a standard autoencoder naturally produces. None
   of them localize. A positive result shows that "cheap, low-dimensional
   learned surrogate" and "localizable" are not fundamentally in
   tension -- the encoder architecture just has to preserve spatial
   structure instead of collapsing to a flat vector. That is a general
   design principle, exportable beyond this project's own three systems,
   not a one-off result.
2. **It relocates the ensemble-size win classical localized DA already
   achieved in physical space (the LETKF revolution of the 2000s) into
   a learned, cheap reduced-order surrogate.** Required ensemble size
   would scale with local per-site dimension (~1-3 for KS/L96) rather
   than the full attractor dimension (~11-22), inside a model cheap
   enough to run large ensembles of in the first place -- the entire
   reason a reduced-order surrogate was wanted to begin with.
3. **The L-transfer corollary is the practically unique part.** A
   local-field surrogate transfers to a larger domain with ZERO
   retraining, being shared-weight by construction. No current ML-based
   reduced-order DA method in Part 2.B can do this -- they are all tied
   to the exact state dimension they were trained on. SEC cannot do it
   either (fit to one training distribution). This is the one result in
   the whole proposal that only the local-latent approach can produce,
   and per the brief's own framing (§7) is the argument that turns "we
   cannot localize at all" into "here is what generalizes further,"
   which is the stronger claim.
4. **It is the direct, concrete bridge to the weather/CM1 conversation**
   (`docs/RESULTS.md`'s "Weather-relevance context" entry). Localization
   is literally why ensemble DA is operationally feasible at NWP scale
   at all. A validated result here becomes the justification for
   attempting the same on Rayleigh-Benard/CM1-scale convection later,
   rather than a hopeful analogy with nothing behind it.

A negative result (local-latent + Gaspari-Cohn fails to beat SEC) is
still informative, not a wasted experiment: it would establish that
distance-free empirical localization is the more robust choice for
systems at this scale, and the L-transfer property alone -- which SEC
structurally cannot ever have -- would remain the sole surviving
argument for the architecturally heavier local-field approach.

**What this project already has that supports pursuing it** (checked
directly against the codebase before writing this, not assumed):

- A **measured, quotable localization-radius design rule** already
  exists and is validated: `minimum_localization_radius = max(
  encoder_receptive_field, v_star * Delta_t)`
  (`ks_latent.analysis.spreading.minimum_localization_radius`), derived
  from Phase 2's information-spreading-velocity measurement (Gate 2,
  passed, `docs/RESULTS.md`). This is a real, already-published-quality
  theoretical contribution independent of whether the DA experiment
  below is ever run.
- A **working local latent field architecture** (`ks_latent/models/
  autoencoder_local_field.py`) exists and has now been validated on TWO
  systems, not one -- KS (the original architecture search) and Lorenz-
  96 (Section 196, `docs/RESULTS.md`'s Lorenz-96 entry) -- giving it a
  genuine spatial/lattice index a Gaspari-Cohn taper could act on
  directly.
- The **PFF (particle flow filter) DA machinery is built and validated**
  (`ks_latent/da/pff.py`, Phase 5/Gate 3, `docs/RESULTS.md`) against the
  analytic Kalman-filter answer in the linear-Gaussian case -- the
  correctness baseline the whole DA line depends on.
- The **distance-free empirical localization (SEC) baseline is already
  coded** (`ks_latent/da/sec.py`, brief Phase 7) -- the honest
  comparator `CLAUDE_CODE_BRIEF.md` itself pre-registers as the bar the
  local-latent approach must beat, not merely beat no-localization.

**What is missing -- this is a ready-to-run experiment, not a completed
result.** Checked directly against `docs/RESULTS.md`: there is no Phase
7 entry (the SEC/`N_ens` sweep baseline has not been run or reported)
and no Phase 13 entry (the actual localized-latent-DA `N_ens` sweep,
compared against the SEC baseline, has not been attempted). Nothing in
this document should be read as claiming the localization result
itself -- only that the prerequisite pieces are unusually far along for
how little of the actual experiment has been run.

**Literature-check status**: ~~Read Pasmans et al. (2025, below) IN FULL
before drafting anything~~ -- **done, 2026-09-23: confirmed clear.** All
three papers flagged during this search (Chandravamsi et al., Guerrieri
et al., Pasmans et al.) have now been read in full and ruled out as
prior art for this proposal. The experimental design below is the
actual remaining blocker.

**Experimental design (2026-09-23), checked directly against the
codebase before writing -- not assumed.** What already exists and can
be reused as-is: `ks_latent/da/pff.py` (validated NAT-PFF, Phase 5/Gate
3), `ks_latent/da/sec.py` (SEC), `ks_latent/da/cycling.py` +
`scripts/run_da_pff.py` (DA cycling driver -- currently hardcoded to a
single global KS setup, no localization hook), and trained L96
local-field checkpoints (Sections 196/197, `n_sites=8, channels=2,
d_latent=16`, flattened SITE-MAJOR -- confirmed by reading
`ks_latent/models/autoencoder_local_field.py` directly:
`z.reshape(B, n_sites, channels)`, i.e. `site = i // channels`, the
detail that matters for indexing the taper correctly). What does NOT
exist yet -- real code, not just new config: `ks_latent/da/
localization.py` (zero "Gaspari" hits anywhere in the codebase,
confirmed by grep), and any N_ens sweep result at all (Phase 7 and
Phase 13 both absent from `docs/RESULTS.md`).

Staged, each gated on the last:

- **Stage 0 -- build the missing infrastructure.**
  `gaspari_cohn_taper(distance, c)` (standard piecewise quintic) +
  `build_latent_taper_matrix(n_sites, channels, c)`, using CIRCULAR site
  distance (both KS and L96 are periodic domains) and respecting the
  confirmed site-major flattening. Wire a localizer hook into PFF's `B`
  computation (Schur product on the ensemble covariance before it enters
  `F = J-bar^T R^-1 J-bar + B^-1`, per the brief's own Phase 13 spec).
  Generalize `run_da_pff.py`: accept `--encoder local_field`,
  `--localizer {none,sec,gaspari_cohn}`, sweep `--n-ensemble`
  automatically, support both KS and the existing L96 checkpoints
  without the current L=100 hardcoding.
- **Stage 1 -- Phase 7 baseline (never run before): SEC sweep on the
  EXISTING GLOBAL latent.** `N_ens` in {8,16,32,64,128,256}, with and
  without SEC, analysis RMSE vs. `N_ens`. One methodological question to
  resolve empirically rather than assume: use a Stage-1-only checkpoint
  (genuine intrinsic chaos, weaker short-horizon forecast) or a Stage-2
  checkpoint (collapsed intrinsic chaos, better short-horizon forecast)?
  DA cycling only ever runs short free-run segments between analysis
  updates, so the Stage-2 checkpoint may be the fairer comparator
  despite its collapsed standalone Lyapunov spectrum -- run both, report
  which is actually the right baseline rather than guessing.
- **Stage 2 -- Phase 13 core: same sweep on the local-field
  architecture with real Gaspari-Cohn tapering.** Reuses Section
  196/197's already-trained checkpoints; no new training needed for a
  first pass.
- **Stage 3 -- pre-registered decision rule (already stated in the
  brief, restated here as this proposal's own bar, not decided after
  the fact): the local-field + Gaspari-Cohn result must beat the Stage
  1 SEC curve, not merely beat no-localization.** SEC is a real,
  competitive method (Anderson 2012), not a strawman -- this is what
  makes a pass credible.
- **Stage 4 -- if Stage 3 passes: L-transfer, the headline result.**
  Take the N=64-trained local-field model, run DA at a larger L96 `N`
  (e.g. 128) by adding lattice sites with zero retraining. Compare
  against the global-latent model (structurally cannot even run at a
  different `N`) and SEC (fit to one training distribution, cannot
  transfer either) -- neither comparator can do this at all, which is
  the actual contrast worth publishing.

The DA cycling itself should be cheap (no gradient descent, just running
an already-trained model) -- the real cost is Stage 0's code, not
compute.

**Additional related literature**, found via this specific search
(2026-09-23), not already in Part 2:

- Pasmans, I., Chen, Y., Finn, T. S., Bocquet, M. & Carrassi, A. (2025),
  "Ensemble Kalman filter in latent space using a variational
  autoencoder pair," *QJRMS*,
  [DOI:10.1002/qj.70070](https://doi.org/10.1002/qj.70070),
  [arXiv:2502.12987](https://arxiv.org/abs/2502.12987) (already cited in
  `docs/LATENT_PDE_RESEARCH_NOTES.md` §10 as a preprint; now published).
  **Read in full 2026-09-23 -- does NOT overlap with 4.3, resolving the
  earlier conflicting search signal.** Their focus is entirely
  non-Gaussianity/constrained-variable handling (e.g. sea ice
  concentration bounded in [0,1], Mohr-Coulomb stress constraints) via a
  VAE latent mapping -- not localization. The Gaspari-Cohn reference
  appears exactly once, in the Discussion, as a FUTURE-WORK suggestion
  for how the method might scale to higher dimensions ("future work
  should also investigate... an approach similar to the application of
  covariance localisation using convolution (Gaspari and Cohn, 1999)")
  -- never implemented or tested anywhere in the paper. Consistent with
  this: their own twin-experiment test system (a "simple circular
  model") has an ensemble-size-to-state-dimension ratio of 32 -- the
  paper states this explicitly as "much higher than the << 1 ratios
  typical of operational forecasting systems" -- i.e. a regime where
  localization is not remotely load-bearing, unlike KS/L96/RBC.
- Poterjoy, J. (2016), "A Localized Particle Filter for High-Dimensional
  Nonlinear Systems," *Monthly Weather Review* 144:59-76,
  [DOI:10.1175/MWR-D-15-0163.1](https://doi.org/10.1175/MWR-D-15-0163.1).
  Classical (physical-space, not latent) localized particle filter --
  the non-latent comparator this project's own PFF localization result
  should be positioned against, not just the SEC/no-localization
  contrast.
- Guerrieri, J. M., Pulido, M., Miyoshi, T., Amemiya, A. & Ruiz, J. J.
  (2026), "Localization in the mapping particle filter," *Nonlinear
  Processes in Geophysics* 33:33,
  [DOI:10.5194/npg-33-33-2026](https://doi.org/10.5194/npg-33-33-2026).
  **Read in full 2026-09-23 -- does NOT overlap with 4.3.** Two
  localization schemes (alpha: local kernel, global state update;
  beta: full local mapping, physical partitioning) for the Mapping
  Particle Filter (MPF, an SVGD-based particle-flow method -- a
  different algorithm from this project's own NAT-PFF), tested on the
  two-scale Lorenz-96 system. Entirely in PHYSICAL state space: the
  state vector is the raw L96 grid, Gaspari-Cohn-style distance decay
  (they cite Gaspari & Cohn 1999 directly) is applied to physical
  grid-point distance. No autoencoder, no learned latent representation,
  no dimension reduction anywhere in the paper -- same category as
  Poterjoy (2016) above, the classical non-latent comparator. **Notable:
  this paper was reviewed by Peter Jan van Leeuwen himself** (see its
  Review statement) -- direct, current confirmation of his own active
  engagement with localized-particle-flow-filter research, consistent
  with `docs/LATENT_PDE_RESEARCH_NOTES.md`'s own framing of the
  localization argument as "in Peter Jan's own language." One genuinely
  useful, citable result to reuse in 4.3's own writeup: their Figs. 5-6
  show the optimal localization radius tracks directly whether ensemble
  size exceeds or falls short of the number of positive Lyapunov
  exponents (20 particles vs. ~16-19 exponents needs looser localization;
  10 particles needs tighter) -- a clean, quantitative illustration of
  exactly the "ensemble size should scale with effective, not full,
  dimension" argument 4.3 is built on, done in physical space rather
  than latent space.
- Chandravamsi, H., Hu, H., Thiagarajan, P. et al. (2026), "Feature-
  preserving Latent-EnKF for Data Assimilation of Flows with Shocks,"
  [arXiv:2606.12559](https://arxiv.org/abs/2606.12559), Johns Hopkins
  Dept. of Mechanical Engineering. **Read in full 2026-09-23 -- does
  NOT overlap with 4.3, checked directly, not just by abstract.** A
  different problem entirely: EnKF's Gaussian assumption breaks when
  shock-location uncertainty makes ensemble statistics multimodal,
  producing spurious oscillations; their fix performs the analysis
  update in a learned, GLOBAL low-dimensional latent manifold (a single
  shared decoder back to physical space -- no spatial/lattice structure)
  specifically to preserve sharp shock/contact-discontinuity features
  through the update. "Gaspari": 0 mentions. "taper": 0 mentions.
  "localization": exactly 1, in the Introduction, as generic background
  on EnKF in general -- not implemented or tested here. Their own
  Conclusion frames the contribution as extending latent EnKF to
  discontinuous flows and lists 3D/other physical systems as future
  work; no mention of localization or ensemble-size scaling. Ruled out
  as prior art for 4.3.
