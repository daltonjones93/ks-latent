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

The three candidates are ordered by how directly they map onto what we already have.

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

## 4. Suggested order and shared infrastructure

| Step | System | New code | Cost | Depends on |
|---|---|---|---|---|
| 1 | Barkley/Aliev–Panfilov 2D, transfer + DA | reaction-diffusion solver, no-flux window | ~1 week | existing 2D pipeline |
| 2 | Plane Couette or channel wall-parallel planes, y⁺ map | data loader/solver, multi-height channels | 2–4 weeks | step 1 not required |
| 3 | 2D shearing sheet | shear-periodic halo | ~1 week | 2D pipeline |
| 4 | 3D MRI shearing box | HPC data, 3D patches | months | step 3 |

Shared requirements across all three: (i) a truth-only box check *before* any training, (ii) a
Lyapunov-spectrum validator at 1×, 2×, 3× or more, with several initial conditions to expose truth
spread, (iii) the decision rules written down before running (R9a, the 10% winner rule, and the Z17
statement of the crossover relative to the learned model's own one-cycle error), (iv) one section on
what failed.

## 5. What would change our minds

- If the cardiac proxy transfers and the DA ordering follows Paper II, the methodology has a natural
  application and a hidden-field story that is stronger than anything in KS.
- If the wall-turbulence planes fail in a y⁺-graded way, we have the missing mechanism test for
  nonlocality; if they pass everywhere, the nonlocal-pressure explanation for Kolmogorov is in doubt
  and Z13's "consistent with" should be weakened.
- If the 2D shearing sheet needs more than the shear-periodic fix (e.g. the epicyclic/density-wave
  coupling is effectively nonlocal), the rule "reach < half the box" would need a modification for
  long-range wave propagation, which would also be worth saying.
