# Future applications: where the local-latent-field methodology could go next

Status: speculation written 2026-10-13 at the user's request ("write a md file and speculate how we can use our
methodology there"). Nothing here has been run. Citations marked *(verify)* are from memory and must be
checked before they appear in any paper. Not part of Papers I or II.

## 0. What "our methodology" is, in four pieces

1. **Per-patch, weight-shared encoder/decoder (reach 0) plus a local propagator** (`local_mlp`, reach =
   layers × window), trained on **windows of one large simulation** with a **forced halo** (Paper I,
   Prop. 2), so that the trained weights are independent of domain size and run zero-shot on larger domains.
2. **The reach rule and the box check**: a small box suffices only if the propagator reach is below half
   the box, and a truth-only spectral test (`scripts/exp_box_check.py`) says whether the new domain
   contains scales the box does not. This is also the diagnostic for when it will *not* work (Z13).
3. **Validation by gauge-invariant dynamics**: full Lyapunov spectra, D_KY per unit size, λ₁, plus
   rollout statistics, with decision rules fixed before the run (R9a, Z8).
4. **Exactly local ensemble DA in the latent field** (Paper II): per-patch encoder ⇒ block-diagonal
   latent observation operator ⇒ exact per-site local analysis, ensemble size set by local dimension,
   and the **post-remedy model-error axis** that says when a physics model with estimated parameters
   beats a learned model (Paper II §6; Z17 on L96 suggests the crossover scales with the learned model's
   *own* one-cycle error, 1–1.5×).

What we have learned about where it fails (Z13/Z14, R15, Z16): systems with a nonlocal velocity/pressure
coupling (Kolmogorov at weak drag) lose large-scale energy when run larger than the training box; more
data, longer rollouts, memory channels and velocity inputs did not fix it; strictly local systems (KS, L96)
transfer cleanly. Every application below is therefore a question of *how local the physics is at the
scales the box resolves*.

The first three candidates are ordered by how directly they map onto what we already have; §4 (coupled
phase oscillators) was added later and is the cleanest control experiment for the nonlocality question,
and §5 is the CM1 roadmap (transient dynamics, no CM1 data needed).

---

## 1. Cardiac electrophysiology: data assimilation of spiral-wave chaos

### The problem
Fibrillation is spatiotemporal chaos of an excitable medium: spiral waves break up into persistent,
self-sustained defect turbulence, with an attractor dimension that grows with tissue size (extensive;
see Strain & Greenside 1998 *(verify)*, and the cardiac spiral-breakup literature, Qu, Weiss & Garfinkel
2000 *(verify)*). Clinical observations are sparse and noisy surface voltage (optical mapping in
experiments; catheters/ECG in patients); the gating variables and the intracellular state are hidden;
ionic model parameters are uncertain. Ensemble DA with a physics model exists (Hoffman, Cherry, Fenton
et al., Chaos 2016, LETKF for 3D reentrant waves *(verify)*; Marcotte & Grigoriev *(verify)*).

### Why our methodology fits
- **Local physics**: the monodomain model is reaction–diffusion, strictly local, translation-invariant in
  homogeneous tissue. No nonlocal inversion, so the Kolmogorov failure mode should not appear.
- **Hidden fields are the generic situation**: only voltage is observed. This is exactly our
  hidden-field case in Paper II §6 (J17B; the learned filter handled hidden ε by latent channels where the
  physics filter, lacking the field, was tied or worse). Latent channels per site are the natural
  carrier for gating variables without writing down an ionic model.
- **Model error is big and mostly structural** (the ionic model is a caricature; parameters are
  patient-specific and drift), which is the regime where our §6 axis says the learned model wins, and
  where estimation of a few parameters (e.g. conductances) is the physics baseline.
- **Exact localization matters**: tissue models have large state (10⁴–10⁶ sites), ensembles are small,
  and defects are localized objects.

### Plan (cheap first)
1. **Proxy systems**: Barkley model (2 fields) and Aliev–Panfilov or Fenton–Karma (3 fields) in 2D,
   periodic and no-flux; cost per run minutes on one CPU core with explicit FD. Parameters in the
   spiral-breakup regime (check extensivity: D_KY vs L by Lyapunov, as for KS).
2. **Transfer test (Paper I protocol)**: train on windows of one large simulation (e.g. 8× the spiral
   wavelength), evaluate at 1×, 2×, 3× the training box, D_KY and λ₁ vs truth, decision rule R9a.
   Pre-register that failures at larger sizes are *not* expected (strictly local). A failure would point
   at sharp-front resolution, not nonlocality; report which.
3. **DA test (Paper II protocol)**: observe the voltage only, on a sparse regular network (every 4th–8th
   cell, noise 0.1 × std) or a catheter-like patch; compare (a) latent LETKF, (b) physics LETKF with
   wrong ionic parameters, with and without online estimation, over a sweep of structural error
   (e.g. a different restitution curve). Report on the post-remedy axis, with the crossover expressed
   relative to the learned model's own one-cycle error (Z17).
4. **Make it harder**: no-flux boundaries (use interior windows plus a boundary indicator channel),
   heterogeneous conductivity (fibrosis map as an input field, as the forcing is an input in
   Kolmogorov), fibre anisotropy (the `local_field_2d_aniso` model exists).

### Risks
- Sharp wavefronts and thin action-potential upstrokes: the 0.1-std noise and 16-grid-point patches that
  work for KS may be too coarse; the encoder may need smaller patches (reach 0 with patch 2–4).
- Excitable systems are not smooth flows: Lyapunov spectra of the *model* may be less robust (stiff,
  threshold dynamics). Use many ICs and report the spread, as for Kolmogorov (single-truth spread ±20%).
- Clinical relevance needs 3D, anisotropy and patient geometry; the 2D proxy only establishes the method.

### What a paper would claim
"A latent field trained on a small simulated tissue patch transfers to larger tissue and supports
exactly local ensemble DA from voltage-only observations; the learned model beats a mis-specified
ionic model once structural error exceeds about its own one-cycle error." Cheapest of the three, and
the closest to the hidden-field results we already have.

---

## 2. Minimal flow units in wall-bounded turbulence

### The problem
Jiménez & Moin (1991) showed that turbulence near a wall is sustained in a *minimal* periodic box (a few
hundred wall units across, ~100 in the spanwise direction for the streaks), and that the statistics in
the near-wall region match larger boxes. The minimal flow unit (MFU) is the standard way to study the
near-wall cycle, exact coherent structures and dimension (Hamilton, Kim & Waleffe 1995 *(verify)* for
plane Couette flow; Flores & Jiménez 2010 on hierarchies of minimal units in the logarithmic layer
*(verify)*). Its limits are well known: very-large-scale motions in the outer layer need larger boxes, and
the box size changes the statistics in the log layer. The attractor dimension of channel turbulence is
extensive in the box size (Keefe, Moin & Kim 1992 *(verify)*).

### Why our methodology fits
- The MFU assumption *is* "a small periodic box represents a large domain". Our reach test and box check
  give a quantitative, per-scale answer to "is this box big enough?", and the Lyapunov-spectrum comparison
  at 2–4× the box is a gauge-invariant version of the usual "do the statistics converge" check.
- Wall-parallel planes map directly onto our 2D pipeline: take planes at a few wall distances
  (e.g. y⁺ = 5, 15, 30, 60) of (u, v, w) as **channels per site** of an (x, z) latent field.
  This is exactly Kolmogorov-flow-in-a-plane with more channels.
- A cheap, genuine test of the "nonlocal pressure" hypothesis: in the near-wall region the pressure
  Poisson coupling acts over wall units that the streak spacing resolves, so a model trained on windows
  in the near-wall layer *might* transfer, while an outer-layer model should fail. A graded failure
  with y⁺ would be the clean mechanism test we lack on Kolmogorov (Z13 could only say "consistent with").

### Plan
1. **Data**: plane Couette (Re ≈ 400, MFU of Hamilton–Kim–Waleffe) or channel Re_τ = 180 minimal box,
   with a spectral solver (Dedalus, Channelflow, or Diablo *(verify availability)*). Wide-box data
   at 2–4× the MFU from the same code. Public alternative: Johns Hopkins Turbulence Database channel
   (Re_τ ≈ 1000), which gives wide wall-parallel planes without running a solver.
2. **Baselines**: box check on the truth (R_new/R_same) as a function of y⁺ *before* training; the
   prediction is that it flags the outer layer and passes the buffer layer.
3. **Training**: windows of the wide simulation in (x, z), forced halo, c channels per site,
   propagator reach set by the box check; compare with training in the MFU itself.
4. **Evaluation**: λ₁ and D_KY per unit area at 1×, 2×, 4×, one-point statistics (mean profile,
   Reynolds stresses), streak spacing and spectrum. Pre-register: per-y⁺ pass/fail like R9a.
5. **DA variant** (optional): observe wall quantities only (wall shear stress, pressure), the standard
   realistic sensor set, and infer the plane at y⁺ = 15; this is the "wall-sensor estimation" problem
   in flow control. The latent filter's block-diagonal observation operator is natural there.

### Risks
- **3D**: the incompressibility constraint and the wall-normal direction make this a larger system than
  anything we have run; the plane-with-channels reduction is a modelling choice that loses the
  wall-normal coupling, so the latent propagator must carry it implicitly (memory/hidden structure).
- The mean flow (shear) and the pressure are the nonlocal parts; a model that cannot represent
  pressure will fail on the same large-scale mechanism as Kolmogorov, which is *informative* but not a
  success. Decide in advance that this outcome is a result.
- Data size: 3D DNS data are large; the JHTDB route avoids running the solver but constrains the grid
  and sampling.

### What a paper would claim
"Where is a minimal flow unit enough?": a per-y⁺ transfer map, and with it, an independent test of
whether the failure of small-box training is due to the nonlocal pressure coupling. Higher risk, highest
conceptual payoff; a negative result in the outer layer would still be publishable as a diagnostic.

---

## 3. Shearing-box simulations in astrophysical disks

### The problem
Local simulations of accretion disks use a **shearing box**: a small Cartesian patch co-rotating with the
disk at a radius, with shear-periodic boundaries in the radial direction and periodic in azimuth
(Hawley, Gammie & Balbus 1995 *(verify)*). Magnetorotational-instability (MRI) turbulence in them is
known to depend on box size, aspect ratio, resolution and the magnetic Prandtl number
(Fromang & Papaloizou 2007; Bodo et al. 2011; Shi, Krolik & Hirose 2010 *(verify all)*), and the
zero-net-flux case has a long-running convergence controversy. How large a box is needed for the
transport coefficient (α) to be box-independent is an open practical question for the field.

### Why our methodology fits
- The same question as the MFU: does a small periodic patch represent a large disk? Our protocol gives
  a transfer test and a box check on the *same* simulation, with the exact Lyapunov spectrum of the
  learned model as a validation, which this literature does not use.
- The structure is a lattice of local patches: patch = (x, y) cell with vertical points (in stratified
  boxes) or fields (B, v, ρ) as channels, as in the atmospheric-column picture.
- Strongly extensive MHD turbulence with a **large-scale** component (mean-field dynamo, the net
  vertical flux): this is where we expect the Kolmogorov-type failure, so it is a stringent test of the
  large-scale-energy diagnostic (|k| < 2 share) developed in Z13/Z14.
- DA is the less natural half: observations of disks are line-of-sight and unresolved. The methodology
  here is primarily surrogate modelling and transfer, not assimilation.

### Plan
1. **Start 2D**: the 2D shearing sheet (hydrodynamic, shallow-water–like vortices, or 2D MRI with a
   vertical field) with shear-periodic boundaries. This needs one new piece of code: shear-periodic
   halo handling (a shift by `S t L_x` in the radial wrap), which replaces the circular padding and
   forced-halo gather. Cost: a day, with a unit test for the shifted wrap.
2. **Real target**: 3D unstratified, zero-net-flux or net-vertical-flux MRI box with Athena++ or
   Dedalus *(verify)*. Train on windows of a wide box (e.g. 8×8×1 H), test at 2× and 4× in the plane.
3. **Metrics**: α_Maxwell+Reynolds, magnetic/kinetic energy ratio, vertical power spectra, plus
   λ₁ and D_KY. Pre-register: α within 20% at 2× and 4× and the energy spectra match.

### Risks
- **Shear-periodic boundary**: translation invariance holds only in the sheared frame; the model must be
  trained and run in that frame. This is a code change, not a research risk.
- **Expense**: 3D MHD boxes are the costliest data of the three; Athena++ runs of a few hundred
  orbits at modest resolution are hours on a cluster, not on a laptop.
- **MRI is itself sensitive to box size, resolution and dissipation** (the known controversy), so
  "the model does not match a bigger box" may reflect the physics, not the method. We would need a
  truth at the larger size *from the same code*, and the 20% decision rule can be unreachable if the
  truth itself varies that much (compare the ±20% single-truth spread we already saw on Kolmogorov).
- No clean exact-conservation structure in the learned model: ∇·B is not guaranteed zero. A decoder
  that outputs the vector potential would fix this by construction.

### What a paper would claim
"A learned local model trained on small shearing-box windows and tested at larger boxes reproduces the
transport and spectra, or fails at identified scales." Most speculative and most expensive; best treated
as a Paper III follow-on once the cheaper two exist.

---

## 4. Networks of coupled phase oscillators (Kuramoto and relatives)

Added 2026-10-14 at the user's request, with the user's reservation that the power-grid application may not
fit because grids are "more tightly connected". That reservation is largely right and is worked through in
§4.8; the lattice and small-world versions of the problem are the interesting part.

### 4.1 The problem
N phase oscillators θ_i(t) ∈ S¹ coupled through a graph:

    dθ_i/dt = ω_i + Σ_j K_ij · h(θ_j − θ_i − α),     h(φ) = sin φ  (Kuramoto–Sakaguchi)

Variants that matter here: heterogeneous natural frequencies ω_i; phase lag α; a coupling function with
higher harmonics (Daido: h = sin φ + a sin 2φ); and inertia (second order: m θ̈_i + θ̇_i = …), which adds a
hidden velocity per oscillator. The regimes that interest us are *spatiotemporally chaotic*, not
synchronised:

- **Phase turbulence in oscillator lattices** (Sakaguchi, Shinomoto & Kuramoto 1987 *(verify)*): the
  continuum limit of locally coupled oscillators with phase lag is the **Kuramoto–Sivashinsky equation**
  itself (Kuramoto's own derivation of KS was a phase reduction of a reaction–diffusion system near a Hopf
  bifurcation; Kuramoto 1984 *(verify)*). So this system is *microscopically below* Paper I's testbed.
- **Chimera states and chimera chaos on rings with nonlocal coupling** (Kuramoto & Battogtokh 2002;
  Abrams & Strogatz 2004; Wolfrum & Omel'chenko 2011, who also computed Lyapunov spectra, found them
  extensive in N, and showed chimeras at finite N are chaotic and can be transient *(all verify)*).
- **Inertial (second-order) Kuramoto** on lattices: robust chaos and hidden velocities
  (Tanaka, Lichtenberg & Oishi 1997 *(verify)*).

Caution on what is *not* chaotic: with **identical** frequencies, α = 0, symmetric K and first-order sine
coupling the flow is a **gradient system** and cannot be chaotic. With heterogeneous ω "phase chaos" does
exist even in the plain all-to-all model for N ≥ 4 (Popovych, Maistrenko & Tass 2005 *(verify)*), but
that is low-dimensional chaos at small N. Extensive, N-scaling chaos on a lattice is usually obtained with
a phase lag α ≠ 0, a second harmonic, or inertia, and any experiment here should say which it is using.

### 4.2 Why our methodology fits (and where it needs more)
| Our ingredient | What it maps to |
|---|---|
| Per-patch weight-shared encoder, reach 0 | One oscillator (or a block of s sites) per latent site; encode (cos θ, sin θ), not θ |
| `local_mlp` propagator, reach = layers × window | Interaction range R (sites) of the coupling kernel; reach ≥ R is the design rule |
| Forced halo, window training | Windows of one long ring; transfer N → 2N → 4N with the same K(r) |
| Lyapunov spectrum / D_KY density | Gauge-invariant check; extensive in N in the regimes above |
| Exactly local analysis | Observation of a subset of oscillators' phases → block-diagonal latent H |

New ingredients that Paper I and II did not need:
1. **A circular state.** Encode (cos θ, sin θ) as two channels; the decoder returns (c, s) and θ = atan2.
   Observation noise should be von Mises or wrapped Gaussian, and DA innovations are wrapped differences.
2. **A global rotation gauge** (θ → θ + c). Compare phases only through differences and the order parameter's
   modulus, never absolute angles, exactly as D_KY is gauge-invariant in the latent.
3. **Quenched disorder as an input channel.** If ω_i varies, ω_i is a fixed per-site field that must be
   supplied, the same role forcing plays in Kolmogorov flow. Transfer to larger N then means a larger draw
   from the same ω distribution.
4. **A conserved topological quantity on a ring** (winding number of twisted states, the analog of ∫u).
   (cos, sin) loses it; carry it as a global scalar if the regime has twisted states.

### 4.3 The key design question: what does "local" mean on a graph?
The Kuramoto coupling is **pairwise**, so on the *graph* it is exactly one hop. The failure we saw in
Kolmogorov flow is a different animal (an elliptic solve, not a sum). Four cases, in increasing
difficulty, with predictions and the reason for each:

| Coupling | Index-local model (our current architecture) | Graph-aware model (message passing) |
|---|---|---|
| Ring, finite range R sites | **Works if reach ≥ R** | Works, equivalent |
| Ring, kernel with exponential tail | Works with a truncated reach; error set by the tail beyond reach (testable) | Same |
| Watts–Strogatz, rewiring p | **Fails gradually with p**: shortcut partners are outside any index window | Works at reach 1 hop, but see below |
| All-to-all (mean field) | Fails | **Works** with a normalised mean aggregator: f(θ_i, mean_j g(θ_j)) |

Two consequences worth stating before running anything:

- The index-local failure under rewiring is a **controlled mechanism test of nonlocality** with a single
  knob p and an unchanged local rule. In Kolmogorov we could only say the failure is "consistent with"
  Biot–Savart; here the cause is built in, so the *shape* of the failure curve versus p is the result
  (smooth, or a threshold near p·k·N ~ 1 shortcut per window).
- The Paper I rule "reach < half the box" has a **graph analog that gets harsher**: reach in hops < ½ the
  graph diameter. A small-world graph has diameter ~ log N, so a handful of hops already spans the whole
  graph. For pairwise Kuramoto coupling this is harmless (1 hop is exact), but a multi-layer propagator
  with reach 3 hops *is* the whole network, so the "small box represents large domain" idea loses its
  meaning. This is a real limit on what "zero-shot transfer in N" can claim on small-world graphs.

*(Conjecture, not checked against the Kolmogorov data: a mean aggregator handles the mean-field case because
the interaction is a position-independent sum. A 1/r kernel would also be a pairwise sum but needs the
distance as an input, so the aggregator would have to be kernel-aware. R15 supplied velocity to a
local model and did not fix Kolmogorov, so this is a hypothesis and the mean-field test is where it is
checked, not assumed.)*

### 4.4 Plan (cheap first; same code path as the L96 control, Z16)

**K0. Regime scan, truth only (hours).** Choose (α or Daido a or inertia m, R/N, ω distribution) so that
the dynamics are chaotic and *stay* chaotic. Report for N ∈ {64, 128, 256}: λ₁, number of positive
exponents, D_KY and D_KY/N, and the **lifetime of the chaotic state** (chimera collapse to synchrony at
small N is a documented risk). Gate: D_KY/N is constant to ±10% across N and no collapse within 50× the
run length. If the plain ring fails the gate, switch to the inertial lattice or phase turbulence with
heterogeneous ω.
- Integrator: RK4 at dt ≈ 0.05/ω_max-scale, float64; exact tangent from the sparse Jacobian
  K_ij · h′(θ_j − θ_i − α) (cost O(N·R) per step, like `tangent_rhs` for L96). Chimera exponents are small
  (≈ 10⁻² or lower), so runs of 10⁴–10⁵ time units are needed for converged spectra; report running D_KY.

**K1. Compression measurement (an hour).** D_KY/N decides whether a compressed latent is possible. L96 was
near-incompressible (0.68 N) so the latent had to be as large as the state. If a Kuramoto regime has
D_KY/N ≈ 0.1–0.3, then c channels per patch of s oscillators with c/s < 1 is a real test of whether the
latent can be smaller than the state, which L96 could not answer. Rule: the smallest c/s at which D_KY is
within ±10% and λ₁ within ±15% at 2× and 4× N, 3 seeds.

**K2. Ring transfer (about a day).** Train on windows of one N=512 ring (or a periodic N=64 ring), evaluate
at N = 64/128/256/512 with no retraining. Registered rule (inherited from L96 so it is not tuned): D_KY
within ±10% of the truth mean (each IC within ±20% given chimera variability) and λ₁ within ±15%, plus
rollout statistics (order-parameter distribution, structure function) and valid time in Lyapunov times.
Controls: the trivial extensive baseline (D_KY/N × N), and the box check (a spectral test of the truth
for scales the training box does not hold).

**K3. Rewiring sweep (about a day).** Watts–Strogatz with k neighbours per side, p ∈ {0, 0.01, 0.03, 0.1,
0.3, 1}, shortcut *density* held fixed as N grows. Two models per p: index-local and graph-aware. Report
D_KY error versus p at 2× and 4× N for each. Pre-register the prediction that the index-local error is
monotone in p and the graph-aware error is flat; a non-monotone curve, or a graph-aware failure, would
be the interesting outcome.

**K4. Mean-field hybrid (a day).** Global Kuramoto, or global + local, with the order parameter
R e^{iψ} as a two-channel global input (the "few global scalars + local field" option in the original
brief, §12.2.1). Test transfer in N and whether the model's order parameter has the correct finite-size
fluctuations (variance ∝ 1/N, a good independent check). Note for DA: the *forecast* now has a dense
dependence through R, but the exactness argument of Paper II concerns the **observation operator**, which
stays block-diagonal; only the *sampling error* of the global scalar needs a separate look.

**K5. DA, the Z17 analog (a few days).**
- Observations: noisy phases (von Mises) of a fraction f ∈ {0.25, 0.5, 1} of oscillators; and the
  harder "voltage-like" case, only cos θ_i, which loses the sign of sin θ and is the hidden-field case
  (second-order variant: phases observed, velocities hidden).
- Physics filter: local ETKF with the *wrong* model, with K (and α) estimated online; a learned filter in
  the latent field. Same winner rule as Paper II: ≥10% lower 3-seed mean and lower on every seed.
- Error axes: **parametric** (K or α off by 10–30%, which estimation should absorb), **structural**
  (a missing second harmonic a, or a coupling delay, which estimation cannot absorb), and in-between
  (heterogeneous ω unknown). Prediction from Z17: physics wins the parametric axis, learned wins the
  structural one once structural error exceeds about 1–1.5× the learned model's own one-cycle error.
- Technical points: circular means and wrapped innovations in the ETKF; localization distance on the
  graph, not the index, for small-world graphs; spread floors as in Paper II.

**K6. (Optional, later) Cross-scale check.** Train on the oscillator lattice, then compare the latent
model's large-scale statistics with KS on the matched continuum scale. This would be the first time our
two papers' systems are tested against one another, but it needs a clean map from (α, K, R) to the KS
coefficients and is a research question in itself.

### 4.5 Risks
- **Weak or transient chaos.** Chimera chaos is slow (small λ₁), sensitive to N and initial conditions,
  and can collapse. Truth runs may be expensive to converge. Mitigation: K0 gate and an inertial or
  heterogeneous fallback.
- **Phase slips and the circle.** A latent in R² per oscillator is fine; a phase *difference* near ±π is
  a discontinuity in θ-space, so train on (cos, sin) and score in phase differences wrapped to (−π, π].
- **Finite-size and N-dependence of the truth** itself: the order-parameter fluctuations scale as 1/√N,
  so a model trained on small N sees a noisier mean field than it will at large N (an in-distribution
  check for K4, and a place where "equal D_KY density" might be too forgiving).
- **Heterogeneity.** Frozen ω_i makes the system non-translation-invariant, so "the same weights at
  every site" is only valid conditional on the ω field; at transfer the ω draw changes and truth spread
  increases (budget more initial conditions and ω draws, as for the Kolmogorov truth spread of ±20%).
- **Conflating tests.** Index-local and graph-aware models differ in more than the graph input (parameter
  count, aggregation). Keep their channel and layer counts matched, or the K3 comparison is not clean.

### 4.6 What a paper would claim
"For locally and mean-field coupled phase oscillators in chaotic regimes, a weight-shared latent field
trained on a small network reproduces the Lyapunov spectrum density on larger ones; transfer degrades
smoothly with the fraction of non-local links for an index-local model and not for a graph-aware one,
giving a controlled test of the locality requirement." The rewiring sweep is the most distinctive
element, because it isolates the locality assumption that Kolmogorov flow could only suggest.

### 4.7 Where it sits relative to the other candidates
Cheapest after the cardiac proxy and a close cousin of Z16/Z17 (ODE lattice, exact tangent, no PDE
solver), so a first K0–K2 pass is about a day of work. Scientifically it is the cleanest nonlocality
control we could run, and the only candidate with a built-in link to Paper I's own system.

### 4.8 The power-grid application: an honest assessment
Power systems are the obvious engineering motivation (the swing equation is a second-order Kuramoto
model: m_i θ̈_i + d_i θ̇_i = P_i − Σ_j |V_i V_j B_ij| sin(θ_i − θ_j)). The user's reservation is correct,
and it is worth being specific about why, because some of it is fixable and some is not.

What does not fit:
1. **Normal operation is not chaotic.** A grid is designed to sit at a stable synchronous state with small
   perturbations. There is no extensive chaos, so the Lyapunov-spectrum validation that carries Papers I
   and II has nothing to validate (D_KY would be 0 and the spectrum is stable). Lyapunov-validated
   transfer is therefore not available; one would need a fault or oscillation regime (inter-area
   oscillations, cascading failure transients), which is a transient, not an attractor.
2. **No translation invariance.** Buses differ in inertia, damping, line impedance and degree. Weight
   sharing across sites is only meaningful with these as input features, which makes the model a graph
   neural network, not our patch encoder.
3. **Small diameter.** Transmission grids have low mean degree (about 2.5–3) but the graph diameter is
   modest and electrical coupling spreads quickly, so a few hops cover a large fraction of the network.
   The "small box represents a large domain" argument is weak (see §4.3: reach in hops against the
   diameter).
4. **Established competition.** Dynamic state estimation with EKF/UKF/ensemble filters and PMU data is a
   mature field; the bar for a learned filter is a model-based estimator with online parameter
   estimation, which is exactly Paper II's comparison but in a field with decades of tuned baselines.

What could still fit:
- **DA with unmodelled dynamics, not Lyapunov transfer.** Grid models omit load dynamics, inverter-based
  resources, and control loops; a learned latent residual in a *local* neighborhood of buses is a structural
  error that parameter estimation cannot absorb. That is the Z17 situation. The measurable claim is
  "learned filter beats the physics filter once structural error exceeds its own one-cycle error",
  on synthetic large grids with hidden dynamics.
- **Ensemble size.** Grids with 10³–10⁴ buses make the ensemble-size argument of Paper II relevant:
  local analysis needs ensemble size set by neighborhood dimension, not network size. Whether
  a graph-local analysis with a k-hop radius is accurate is a concrete, checkable claim.
- **Large synthetic test cases** (e.g. the ACTIVSg synthetic grids at 500–10,000 buses *(verify)*) give
  realistic topologies without proprietary data.

Verdict: a weak fit for the Lyapunov-transfer story and a moderate fit for the DA story. If pursued, treat
it as the K5 experiment on a graph with grid-like statistics (low degree, long thin loops, heterogeneous
parameters) and **not** as a transfer paper; and do it *after* K3, because K3 tells us what a graph-aware
model can and cannot do on a non-lattice network before we meet real heterogeneity.

### 4.9 What a failure here would tell us
- If K2 fails on a plain ring with R below reach, the problem is the circular encoding or the chaos
  regime, not locality, so fix the setup before reading anything into it.
- If K2 passes and K3's index-local error does **not** rise with p, locality of the *index window* is not
  what matters, which would weaken the explanation we gave for Kolmogorov.
- If K4 passes with the order parameter as a global input, a finite set of global moments is a tractable
  kind of nonlocality, which would delimit exactly what R15's failure did and did not rule out.

---

## 5. A roadmap toward CM1 squall lines without CM1 data (transient dynamics)

Added 2026-10-14 at the user's request. Context: the end goal (Paper III) is latent DA on CM1 squall-line
data. We have no CM1 data and the user cannot generate it on the laptop. This section builds a **model
hierarchy**, the standard atmospheric-science device: each rung adds one CM1 ingredient, every rung is
cheap, and a failure localises to the ingredient just added. Literature citations are from memory and
marked *(verify)*.

### 5.1 The methodological problem that none of the earlier sections solves
A squall line is a **transient**, not a statistically stationary attractor: it spins up, propagates, and
decays over hours. Papers I and II validate with Benettin Lyapunov spectra and D_KY per unit size, which
require a stationary, ergodic attractor. None of KS, L96, Kolmogorov flow, or the oscillator lattices in §4
forces us to change the protocol; this is the first place where it must change. The hierarchy below is
therefore chosen so that **every rung is a transient we can generate in bulk** (hundreds of events per
rung), and so that the new validation protocol is developed and debugged on cheap cases first.

### 5.2 Proposed validation protocol for transients (to be pre-registered before T1 runs)
1. **Event ensembles.** Randomise the initial condition and environment (bubble amplitude and size,
   background stratification, shear, rotation rate) over a stated range; hold out whole parameter regions.
2. **Event diagnostics instead of attractor statistics:** front (gust-front) position and speed, cold-pool
   strength, rain total and accumulated-rain histogram, kinetic-energy spectrum at fixed times, and
   conservation errors (mass, energy where conserved).
3. **Finite-time Lyapunov exponents (FTLE).** Keep our gauge-invariant dynamics check by computing the
   FTLE distribution over the event window, for the truth and for the model, from an ensemble of
   initial conditions. Compare distributions (mean and quantiles) instead of a spectrum. Short windows are
   noisy, so a distribution is the object, not a single value.
4. **Skill relative to the truth's own spread.** Forecast error versus lead time, normalised by the
   spread among truth runs from perturbed ICs (the predictability floor). A model that is "within the
   predictability floor" is as good as a perfect model can be.
5. **Time-shift and event-count transfer** (the transient analogue of domain transfer, see 5.4).
Decision rules (R-style): fix tolerances before looking; e.g. front position within one grid cell or 5% of
the travelled distance at 2×, FTLE mean within 15%, error below the truth-spread floor at the lead time
that matters. Both verdicts are publishable.

### 5.3 The ladder
| Rung | System | New ingredient vs. the rung below | Transient? | Cost on a laptop |
|---|---|---|---|---|
| T0 | Rotating shallow water (RSW) | Fast gravity waves, geostrophic balance, hidden velocity (observe h) | Adjustment, dam break, forced-dissipative turbulence | hours |
| **T1** | **Modified RSW with convection and rain** (Würsch & Craig 2014; Kent, Bokhove & Tobias 2017 *(verify)*) | A convective trigger when h exceeds a threshold, and rain as a hidden variable; designed as a toy for convective-scale DA | Triggered, decaying events | 1–2 days for a validated solver |
| **T2** | **2D density current / cold pool** (Straka et al. 1993 benchmark *(verify)*) | Buoyancy, pressure (elliptic or acoustic), vertical structure, Kelvin–Helmholtz rotors, a cold pool | One event over about 900 s | a few days |
| T3 | 2D moist bubble with warm-rain microphysics (Bryan & Fritsch 2002 *(verify)*) | Latent heating, microphysical thresholds, positivity | Yes | about a week |
| T4 | Cold pool in vertical shear (Rotunno, Klemp & Weisman 1988 *(verify)*) | Gust-front lifting, line propagation, RKW balance | Yes, with a long-lived line | extension of T2/T3 |
| T5 | 2D idealised squall line (WRF `em_squall2d_x`-style, or a 2D CM1 case if a build ever becomes possible) | Full combination, domain large enough for line organisation | Yes | needs external code |

The lower rungs can be solved with small purpose-written solvers (finite volume or pseudo-spectral); the
upper rungs need validation against published benchmark numbers (front position and temperature extrema
at fixed times for T2) before any learning is attempted.

### 5.4 What carries over from Papers I–II, and the transient version of each
| Existing element | Transient version |
|---|---|
| Reach rule, "small box suffices" | Train on one event in a small box; run **several independent events in a large box** with no retraining; far-apart events must evolve independently (locality), and the **collision of two gust fronts** is the physically interacting case (new convection is triggered) |
| Forced halo (Prop. 2) | Lateral boundary conditions of a limited-area model (CM1's inflow/outflow) are exactly an externally supplied halo |
| Hidden-variable latent filter | Observe rain or reflectivity-like fields, infer θ, pressure and velocity |
| Z17 crossover (physics + parameter estimation vs. learned) | Physics model with wrong microphysics or trigger threshold vs. learned filter; parametric vs. structural error |
| Box check | Truth-only test of whether a larger domain holds scales or events the small box does not |
| D_KY per unit size | Replaced by FTLE distributions (5.2); extensivity is tested through event counts, not D_KY |

### 5.5 Rung-by-rung plan

**T0, rotating shallow water.** Doubly periodic, forced-dissipative (a drive at intermediate wavenumbers
and a large-scale drag) for a stationary turbulent state, plus transients: Rossby adjustment and a
Gaussian-hump dam break. Checks: gravity-wave speed √(gH) against the snapshot interval (the Phase 2
rule: reach ≥ c·Δt/h); conservation of mass and potential vorticity along Lagrangian paths (to
tolerance); divergence-free balanced part after adjustment. DA: observe h only, infer (u, v). Expected:
transfer in domain size works as for KS because the system is hyperbolic, with no elliptic solve.
Main new risk: **two time scales** (slow balanced flow and fast waves), which can force a very small
snapshot interval or a latent split into balanced and wave parts.

**T1, modified RSW with convection and rain.** The reason for choosing it: the modelling community
built it for convective-scale DA experiments *(verify)*, so there are published baselines for the DA
side (ensemble filter behaviour under model error, observation of rain). Questions: can a weight-shared
local latent learn a threshold trigger (a discontinuous source) and non-negative rain? Does the learned
filter beat a physics filter with a misspecified threshold? Failure mode to watch: a smooth MLP
propagator rounding the threshold, giving early or late triggering. A direct measurement of this
(trigger time and place versus truth) is part of the validation set.

**T2, 2D density current.** Dry, no moisture, x–z, periodic or open in x, rigid plates in z. Truth solver
must reproduce the benchmark's front position and extrema at fixed times at the benchmark resolution
(with its prescribed eddy viscosity; the problem is not converged without it, which is a feature: it
removes the sub-grid question). Training set: randomised bubble amplitude, radius, depth and
background stability, many events. Tests: (i) held-out amplitude and size; (ii) domain transfer with
several pools; (iii) FTLE distribution over the event; (iv) front position bias, which is the
first thing a smoothing MLP will damage. Vertical structure enters as **columns-as-channels** (the same
design as the minimal-flow-unit plan, §2): the encoder sees a full column at each x-site, so reach is
purely horizontal.

**T3, 2D moist bubble.** Add moisture variables with simple warm-rain (Kessler-type) microphysics. New
requirements: positivity of the moisture and rain mixing ratios (a constrained decoder, e.g.
softplus), threshold physics at saturation, and **log-like observation operators** (reflectivity is a
log of hydrometeors, bounded below by "no echo", so the innovation distribution is non-Gaussian near
zero). DA: reflectivity and radial-velocity-like observations of a subset of columns.

**T4, cold pool in shear.** Add background shear and test RKW's prediction that a line is long-lived when
the cold pool's circulation balances the shear's. Here the model must hold a propagating coherent
structure for many advective times, so error in propagation speed accumulates linearly with time: the
first rung where *phase drift* is the dominant failure and needs its own metric.

**T5.** A 2D idealised squall line. Needs external code (WRF idealised 2D case, or CM1 in 2D, which is
small compared to 3D but still needs a Fortran/NetCDF build); keep it as the target, not a prerequisite.
If it ever becomes possible, the first step is a truth-only box check and the event protocol of 5.2.

### 5.6 Risks
- **Sharp fronts and shocks.** An MLP propagator trained with MSE will smooth them; phase and amplitude
  errors of the front may dominate everything else. Consider a front-aware loss or a conservative
  (flux-form) latent update; measure front error first.
- **Positivity and conservation.** Rain, moisture and mass must remain non-negative / conserved; a
  decoder constraint is cheap, a conservative propagator is harder.
- **Stiffness from fast waves** sets the snapshot interval and so the number of rollout steps per event.
- **Short trajectories and distribution shift.** Training on many short events means a model can be good
  at the typical event and wrong on rare triggers; hold out whole regimes, not random snapshots.
- **Benchmarks as a trap.** Matching a published benchmark at one resolution is not evidence of
  generality; test at a second resolution and a second parameter region.
- **No CM1 until much later.** Every rung is a proxy; the roadmap claims only that ingredients are
  tested in isolation, not that results transfer to CM1.

### 5.7 What would change the plan
- If T0 fails on balance (the model drifts off the slow manifold), the latent probably needs an explicit
  balanced/wave split before any convective rung.
- If T1 shows the trigger cannot be learned without a hand-built indicator channel, state that and
  supply the indicator as a physical input, as ω_i is supplied in §4.
- If T2's front error is large but FTLE statistics are fine, the dynamics are right and the
  regression is wrong: change the loss, not the architecture.

---

## 6. Suggested order and shared infrastructure

| Step | System | New code | Cost | Depends on |
|---|---|---|---|---|
| 1 | Barkley/Aliev–Panfilov 2D, transfer + DA | reaction-diffusion solver, no-flux window | ~1 week | existing 2D pipeline |
| 2 | Plane Couette or channel wall-parallel planes, y⁺ map | data loader/solver, multi-height channels | 2–4 weeks | step 1 not required |
| 3 | 2D shearing sheet | shear-periodic halo | ~1 week | 2D pipeline |
| 4 | 3D MRI shearing box | HPC data, 3D patches | months | step 3 |
| 1b | Kuramoto lattice/ring: K0 regime scan, K1 compression, K2 transfer | RK4 + exact sparse tangent, circular encoding | ~1 day | Z16 code path |
| 1c | Rewiring sweep K3, mean-field K4, DA K5 | graph-aware propagator, wrapped innovations | ~1 week | K2 |
| 5 | Power-grid DA on synthetic grids | GNN with edge features | weeks | K3, K5 |
| C0 | Rotating SWE: waves, balance, h-only DA (CM1 roadmap T0) | periodic RSW solver | days | existing 2D pipeline |
| C1 | Modified RSW with convection and rain (T1) | trigger and rain variables, positivity | 1–2 weeks | C0 |
| C2 | 2D density current (T2), then moist bubble (T3), cold pool in shear (T4) | x–z solver, benchmark validation, FTLE protocol | weeks | C1 |

Shared requirements across all candidates: (i) a truth-only box check *before* any training, (ii) a
Lyapunov-spectrum validator at 1×, 2×, 3× or more, with several initial conditions to expose truth
spread, (iii) the decision rules written down before running (R9a, the 10% winner rule, and the Z17
statement of the crossover relative to the learned model's own one-cycle error), (iv) one section on
what failed.

## 7. What would change our minds

- If the cardiac proxy transfers and the DA ordering follows Paper II, the methodology has a natural
  application and a hidden-field story that is stronger than anything in KS.
- If the wall-turbulence planes fail in a y⁺-graded way, we have the missing mechanism test for
  nonlocality; if they pass everywhere, the nonlocal-pressure explanation for Kolmogorov is in doubt
  and Z13's "consistent with" should be weakened.
- If the 2D shearing sheet needs more than the shear-periodic fix (e.g. the epicyclic/density-wave
  coupling is effectively nonlocal), the rule "reach < half the box" would need a modification for
  long-range wave propagation, which would also be worth saying.
- If the Kuramoto rewiring sweep shows no dependence on p for the index-local model, locality of the index
  window is not the controlling factor, and our explanation of the Kolmogorov failure (nonlocal coupling)
  needs rethinking. If it rises smoothly with p while the graph-aware model stays flat, that explanation
  gains a controlled test it has not had (§4.3, K3).
- If a Kuramoto regime has D_KY/N well below 1 and the latent can be compressed there (K1), the L96 result
  ("latent as large as the state") is a property of L96's incompressibility and not a limit of the method.
- If the transient-validation protocol of §5.2 cannot separate a good model from a smoothed one (front
  position and FTLE both pass for a model that visibly smears the front), the protocol is the problem and
  needs a front-sensitive diagnostic before any CM1-facing claim is made.
