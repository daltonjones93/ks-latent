# Latent-Space PDE vs. ODE: Research Notes

**Project:** KS L=100 latent DA + manifold analysis
**Topic:** Peter Jan's request for "a PDE that models the latent variables"
**Status:** Living document — ideas, assessments, related work, and decision criteria.
**Started:** 2026-08-28

---

## 0. Executive summary

- **You are right about the current model.** The existing 44-dim latent admits no PDE
  description, for four independent reasons (§2). Any "PDE" fitted to it today would
  be a fiction.
- **But the goal is salvageable, and the loophole is extensivity.** At L=100 the KS
  attractor dimension is *extensive*: D_KY ≈ 0.226·L. That means the state is not
  really a 22-dimensional point, it is a **field of weakly-coupled local units at a
  density of ~0.22 degrees of freedom per unit length**. A latent *vector* has no
  continuum limit. A latent *field* does. This is the entire reframing.
- **Don't impose locality by penalty. Impose it architecturally.** Your penalty
  collapsed for a structural reason, not an optimizer reason (§4). Fixes exist for
  the penalty route, but the patch-local encoder is strictly better.
- **There is a crisp, falsifiable criterion for "is it a PDE?"**: train the latent
  lattice model at two different site spacings h and ask whether the *same* function
  of (z, δz, δ²z, …) fits both. If yes → continuum limit exists → PDE. If no →
  lattice ODE. Either answer is publishable (§5.4).
- **This is not a detour from the DA goal — it is the missing piece.** A local latent
  field is what makes **localization** possible in latent-space DA. Without it, your
  PFF ensemble size has to scale with the full attractor dimension. This is the
  strongest argument to make to Peter Jan (§6).
- **Do the cheap diagnostics first** (§3). They cost days, not weeks, and answer the
  question "does hidden spatial structure already exist?" empirically before any
  retraining.

---

## 1. Disambiguating what "a PDE for the latent variables" could mean

Three different things get called this. They have different answers.

### Interpretation A — "give me the governing equations of the latent state"
Mathematically this is the **inertial form**. KS on a periodic domain has a
finite-dimensional, smooth, exponentially attracting inertial manifold; the dynamics
restricted to it is *by definition* a finite system of coupled ODEs, not a PDE
(Foias–Sell–Temam; Constantin–Foias–Nicolaenko–Temam). Your instinct is correct and
this is the textbook answer.

Deliverable if this is what he wants: a neural ODE in latent space plus a SINDy
symbolic approximation. Tooling already exists and two of the relevant papers are
already in the project folder (Linot & Graham). This is low-risk and quick.

### Interpretation B — "find a spatial coordinate in which the latent dynamics is local, then write a PDE in it"
This is a real, established research program: **emergent-space PDE learning**
(Kemeth, Bertalan, Thiem, Dietrich, Moon, Laing & Kevrekidis, *Nat. Commun.* 13:3318,
2022, "Learning emergent partial differential equations in a learned emergent space",
arXiv:2012.12738). Their setting is almost exactly your objection: a system of coupled
agents/oscillators with **no obvious spatial coordinate**, all-to-all coupling. They
use manifold learning to extract emergent coordinates, then learn a PDE that is *local
in those coordinates*.

Notably they explicitly address the apparent paradox you'd raise — fine-scale dynamics
with long-range coupling, coarse model with local coupling — and their framing is the
honest one: the learned operator is not "the true physics," it's a parsimonious
parametrization of the long-time dynamics on a lower-dimensional slow manifold.

**This is the single most relevant citation for Peter Jan's request.** It gives the
idea legitimacy, and it also gives you a template to argue about.

### Interpretation C — "what is the effective large-scale PDE of KS?"
This one has a known answer, and it is genuinely interesting. Coarse-grained KS at
large scales is in the **KPZ / stochastic-Burgers universality class** (Yakhot 1981;
Zaleski; Sneppen et al.; recent functional-RG work identifies KPZ z=3/2, Edwards–
Wilkinson, and an intermediate inviscid-Burgers z=1 regime). Writing u = ∂ₓh, the
long-wavelength h obeys an effective noisy KPZ equation with a *positive* renormalized
viscosity even though the bare KS viscosity is negative.

Why this matters to you: it is a **ground-truth validation target**. If you build a
coarse latent field and extract an effective PDE, you have a known answer to check
against in the large-scale sector. That converts a vague "let's find a PDE" into a
falsifiable experiment.

**My read of Peter Jan's actual goal:** he wants Interpretation C (learn something new
about KS from the reduced description), and is asking for it via Interpretation B.
Worth confirming with him directly — the answer changes the plan a lot.

---

## 2. Why the *current* 44-dim latent admits no PDE

Four independent obstructions. Any one of them is fatal on its own.

1. **No ordering.** The latent index k ∈ {1..44} is arbitrary — it's a permutation
   gauge freedom of the last Linear layer. A PDE needs a spatial coordinate; you have
   a label.
2. **Global support.** The encoder mean-pools 128 patches into 16 group tokens and
   then runs a *global* transformer with 8 query tokens. Every z_k depends on every
   x. There is no receptive-field structure to inherit locality from. Locality was
   architecturally destroyed on purpose.
3. **Whitened by construction.** `L_decorr` + `L_var` drive Cov(z) → I. You have
   explicitly removed all second-order structure among latent coordinates. This is
   also the correct explanation for why SFA / DMD / seriation all came back
   unimpressive — you optimized that structure away. You cannot then ask for it back.
4. **Finite dimension without extensivity.** A PDE is the continuum limit of a lattice
   as spacing → 0 with the *number* of variables → ∞. A fixed 44-vector has no such
   limit. This is exactly your point and it is correct.

**Note that obstruction 3 is self-inflicted and reversible**, and obstruction 2 is an
architecture choice, not a law. Only 1 and 4 are structural — and 4 dissolves under
the extensivity reframing below.

---

## 3. Cheap diagnostics to run FIRST (days, no retraining)

These empirically answer "is there hidden spatial structure I can exploit?" before
committing weeks to a rebuild. Run all four.

### D1 — Decoder sensitivity maps (highest value/effort ratio)
For each latent coordinate k, compute over many on-attractor states:

```
S_k(x) = E_z [ | ∂D_ψ(z)(x) / ∂z_k | ]
```

Then, using **circular** statistics on the periodic domain, compute the centroid and
angular spread of S_k.

- Broad, delocalized S_k → confirms §2.2, latents have no spatial support. (Expected.)
- Narrow S_k → each latent already owns a region of x, and you can order them by
  centroid *for free*.

Do the same for the encoder: `∂z_k/∂u(x)`. Cost: one `jacrev` pass, you already have
this machinery from the PFF Jacobians.

### D2 — Wavenumber content of the sensitivity maps
FFT each S_k. If each latent is narrow-band in wavenumber rather than in position,
then you have a *spectral* latent, not a spatial one. That's an equally good answer,
and it points to a mode-coupling ODE system (approximate inertial form) rather than a
real-space PDE. Either outcome is informative — this diagnostic cannot fail to say
something.

### D3 — Propagator Jacobian coupling graph (do this instead of covariance seriation)
Your `latent_reorder.py` seriation failed because it acted on covariance, which is ≈ I
by construction. Use the **dynamical** coupling instead:

```
A_kl = E_attractor | ∂z_{n+1,k} / ∂z_{n,l} |
```

(or a nonlinear dependence measure — distance correlation, mutual information, or
transfer entropy — if you want to avoid linearization). Then ask: **is there a
permutation of the 44 indices under which A is approximately banded / circulant?**
Spectral seriation on A, scored against a null distribution of random permutations, so
you get a p-value rather than a picture. This is the direct test of "does hidden
locality exist in the dynamics?" and it is cheap.

### D4 — Translation representation on the latent (my favorite)
You trained with cyclic-shift augmentation, so the encoder is *approximately*
translation-equivariant. If so, the translation group must act on the latent somehow.
Fit, for each shift c:

```
R(c) = argmin_R  E_u || E_φ(Roll(u,c)) − R E_φ(u) ||²
```

Then check: (a) how well does a *linear* R(c) fit? (b) does R(c₁)R(c₂) ≈ R(c₁+c₂),
i.e. is it a one-parameter group? (c) diagonalize R — the eigenvalues should be
e^{i k_j c} for a set of **latent wavenumbers** k_j.

If this works even partially, **you get a spatial/spectral coordinate on the existing
latent with zero retraining**, and Peter Jan's question becomes answerable immediately:
sort latents by k_j, and the dynamics in that basis is a mode-coupling system you can
try to write as a PDE in Fourier space. Given the shift augmentation and the reported
learned equivariance, I'd put decent odds on this partially succeeding.

### D5 — Local intrinsic dimension vs. patch length (reuses existing code)
Take patches of the *physical* field of length ℓ, run your two-NN estimator from
`latent_manifold.py` on them, and plot d_local(ℓ). Extensivity predicts a straight line
with slope ≈ 0.22 (plus an offset from boundary/overlap effects). This both validates
the extensivity assumption at L=100 and **tells you how many latent channels per site
you need** in §5. Precedent: Zeng, Pérez-De-Jesús, Fox & Graham (*MLST* 5:025053, 2024)
apply IRMAE-WD to local patches exactly this way.

---

## 4. Why your smoothness penalty collapsed, and how to fix it if you keep that route

### Diagnosis
`λ Σᵢ (z_i − z_{i+1})²` has a **global minimizer at total collapse** (all z equal), and
collapse is *inside the feasible set*. This is a property of the objective's geometry,
not of the optimizer. So:

> **No solver — augmented Lagrangian, primal-dual, ADMM, MDMM — will fix this.**
> Changing the solver changes the path, not the location of the minimum.

Second, subtler problem: you are penalizing differences along an **arbitrary index**.
Smoothness in a meaningless coordinate is not a structural constraint; it is just
shrinkage toward the mean. The network's cheapest response to shrinkage is collapse.
Both observations are consistent with exactly what you saw.

### Fix 1 — make the penalty scale-invariant (do this first, it's one line)
Use a normalized Dirichlet energy / Rayleigh quotient:

```
R(z) = Σᵢ ||z_{i+1} − z_i||²  /  Σᵢ ||z_i − z̄||²
```

Collapse leaves R unchanged (0/0 → regularize the denominator), so the trivial
solution is no longer a minimizer. This is precisely the graph-Laplacian / SFA
objective, applied along the latent index instead of along time. This alone may make
your original experiment behave.

### Fix 2 — make collapse impossible, not merely unattractive
Replace `L_var = (1/d)Σ(C_ii − 1)²` with the **VICReg one-sided hinge**:

```
L_var = (1/d) Σ_i max(0, 1 − std(z_i))²
```

A two-sided quadratic can be traded off against the smoothness term; a hinge cannot be
satisfied by shrinking. Better still: insert a **hard whitening layer** (Cholesky /
batch-whitening of the latent), so Cov(z) = I is an architectural identity and the
smoothness penalty can only act on the residual rotation.

### Fix 3 — learn the permutation, don't fix it
If you insist on real-space smoothness with a global encoder, the ordering must be a
free variable. Use a Gumbel–Sinkhorn / soft-permutation relaxation, or an optimal-
transport relaxation, and penalize smoothness *under the learned order*. Otherwise
you're demanding the network satisfy a constraint under an arbitrary labeling.

### On the constrained formulation |z_i − z_{i+1}| ≤ m
Fine in principle, with three corrections:
- Constrain the **normalized** quantity, or add a simultaneous **lower bound on
  variance**. Otherwise the feasible set still contains collapse and the constraint is
  vacuous.
- Use **MDMM** (Platt & Barr, NIPS 1988, "Constrained differential optimization") —
  damped dual ascent — rather than plain dual ascent. This is the standard stabilizer
  for Lagrangian constraints in nonconvex NN training and it substantially reduces
  multiplier oscillation. The primal-dual gap is not the practical problem here;
  multiplier blow-up on early infeasibility is.
- **Anneal m *downward*, not upward.** Start loose (feasible, small multipliers) and
  tighten — standard continuation/homotopy. Starting at small m means starting deeply
  infeasible, which drives λ up fast and produces exactly the instability you saw.

**Honest recommendation:** treat Fix 1 + Fix 2 as a *diagnostic* — if a scale-invariant
smoothness objective can be driven low without wrecking reconstruction, hidden locality
exists and the architectural route will work easily. If it can't, that's evidence too.
But don't build the research program on penalties.

---

## 4b. The inequality-constrained formulation: evaluation + tested implementation

*(Added after direct evaluation of the |z_i − z_{i+1}| ≤ M idea. Companion code:
`latent_locality.py`, toy validation in `test_locality.py`.)*

### 4b.1 The idea is better than the penalty, for a precise reason

I under-credited this in §4. The inequality constraint is **structurally different**
from the quadratic penalty, not just a softer version of it. Using the Rockafellar
augmented-Lagrangian form for `g ≤ 0`:

```
P(g) = (1/2μ) [ max(0, λ + μ g)² − λ² ] ,      dP/dg = max(0, λ + μ g)
```

the derivative is **exactly zero** whenever the constraint holds with slack. The
quadratic penalty's gradient never vanishes, so it pulls toward collapse forever; the
ALM term switches itself off. That is the correct instinct and it's why this is worth
implementing.

The residual danger is a **transient** one: while infeasible, the gradient of the
constraint term points along the shrink direction, and collapse is an *absorbing state*
— once collapsed the constraint is permanently inactive, λ decays to zero, and only
reconstruction can climb back out of a latent the AE has already reorganized around.
That matches what you observed.

### 4b.2 Empirical finding: the variance floor is NOT sufficient

Tested on a toy band-limited-field autoencoder (NX=64, d=16, 6000 steps), three
regimes, target ratio M² = 0.64 (uncorrelated neighbours would give ≈ 2.0):

| regime | recon | neighbour ratio | **effective rank** (/16) | min std |
|---|---|---|---|---|
| unconstrained | 0.0092 | 2.08 | 11.92 | 0.49 |
| quadratic penalty | 0.0094 | 0.43 | **2.24** | 0.08 |
| ALM, smoothness + variance floor only | 0.0214 | 0.19 | **1.23** | 1.66 |
| ALM, smoothness + variance floor + **far-pair decorrelation** | 0.0092 | 0.37 | **6.60** | 1.16 |

The third row is the important one. Adding a per-coordinate variance floor
`Var(z_i) ≥ 1` blocks **shrinkage** collapse — min std went *up* to 1.66 — but the
effective rank fell to 1.23. The optimizer simply made all d coordinates the *same*
unit-variance variable. Both constraints were satisfied with slack.

> **Lesson: a per-coordinate variance floor does not prevent collapse, because the
> degenerate solution of a smoothness constraint is rank-1, not zero.** You must
> constrain the *spread* of the latent, not the scale of each coordinate.

The fix is a third constraint family: an upper bound on the rms correlation of each
coordinate with its **non**-neighbours,

```
q_i = Σ_j Wf_ij ρ_ij²  −  ρ_max²  ≤  0
```

With that added, the constrained run hits the smoothness target **at zero
reconstruction cost** (0.0092, identical to unconstrained) while retaining effective
rank 6.60. All multipliers ended at exactly 0 — feasible with slack, zero residual
gradient pressure, which is the property claimed in §4b.1.

Effective rank 6.6 < 11.9 is *not* a failure. Imposing neighbour correlation
necessarily lowers rank; a banded covariance has lower effective rank than the identity
by construction. That is the constraint doing its job. **Effective rank (participation
ratio of the covariance eigenvalues) is the single monitor to watch — not per-coordinate
variance, which is actively misleading here.**

### 4b.3 What the three constraints jointly say

Read together, the constraint set is:

- near pairs: strongly correlated (ρ ≳ 0.7 at M = 0.8)
- far pairs: nearly uncorrelated (ρ ≤ 0.15)
- each coordinate: unit variance

That is a specification of a **banded circulant covariance** — a target correlation
*length* on the latent index rather than a target correlation *matrix* of I. Which
suggests a simpler alternative worth trying alongside the constrained version:

> **Replace `L_decorr` + `L_var` with a single matching loss `||Corr(z) − C_target||²`
> where C_target is circulant with a Gaspari–Cohn-shaped correlation function of the
> index distance.**

This has a nontrivial minimum (so no collapse), gives the latent a ring topology and a
correlation length, keeps B well-conditioned for the natural-gradient PFF, and is
*exactly* the covariance structure localization wants. It's one line and no dual
variables. I'd run it as the baseline that the constrained method has to beat.

### 4b.4 Answering "why did he add that constraint?"

The decorrelation + unit-variance losses are not a representation-learning choice — the
reason is recorded in your own handoff: they drive `Cov(z) → I`, hence `B ≈ I`, which is
**why the natural-gradient PFF is well-conditioned**. It's a DA-side conditioning
decision. That also explains the note in the same file that "linear reordering methods
find no variance structure" — the losses removed it.

So it should not simply be relaxed. The banded-circulant target in §4b.3 is the right
move: it preserves the conditioning (a well-scaled banded circulant is perfectly
well-conditioned; the eigenvalues are the DFT of the correlation function, and a
Gaspari–Cohn kernel is positive definite with a controllable spectral floor) while
adding exactly the structure you want. You can pick the correlation length to set the
condition number explicitly.

### 4b.5 Implementation notes that matter

Collected in `latent_locality.py`; the non-obvious ones:

1. **Rectification bias in the dual update.** `λ ← max(0, λ + μ ĝ)` with a noisy
   minibatch `ĝ` is biased *upward* relative to using `E[g]`, because the rectifier is
   convex. With per-batch constraint estimates this produces a slow multiplier ratchet
   and eventually collapse. **Use an EMA of g in the dual step.** This is implemented
   and is probably the single most important practical detail.
2. **Alternating, not simultaneous.** `dual_every = 20` primal steps per dual step,
   which is closer to true ALM and much less noisy. Simultaneous GDA works too if
   `lr_dual ≪ lr_primal`, but the alternating version was more stable.
3. **Calibrate M, don't guess it.** The module runs 20 batches unconstrained, measures
   the model's natural neighbour ratio, and sets `M_start` at the 90th percentile — so
   training *starts feasible*. Then anneal **downward** geometrically. Starting at a
   small M means starting deeply infeasible, which drives λ up fast; that is the most
   likely proximate cause of the instability you hit.
4. **Warmup.** No constraint pressure at all for the first ~2000 steps; let the AE find
   a reasonable reconstruction first.
5. **Cap the multipliers** (`λ_max`) and grow μ slowly with a ceiling.
6. **Scale-free smoothness.** Constrain `E[(z_i − z_j)²] / (local mean variance)`, not
   the raw difference, so the ratio can't be reduced by rescaling.
7. **Damping (MDMM, Platt & Barr 1988).** Not implemented here since the EMA + slow
   dual rate proved sufficient in the toy, but it's the next lever if the multipliers
   oscillate on the real problem.

### 4b.6 The ordering problem, handled

Constraining adjacent pairs under an arbitrary index is asking the network to satisfy a
constraint under a meaningless labelling (§2.1). `latent_locality.py` offers two routes:

- `FixedRingGraph(d, perm)` — supply an order, e.g. from decoder-sensitivity centroids
  (diagnostic D1). Principled if D1 gives you localized latents.
- `LearnedRingGraph(d)` — assign each latent coordinate a **learnable angle φ_k on a
  circle** and build the adjacency from a periodic kernel of angular distance. The
  ordering becomes a discovered quantity, and gradients flow to φ. This is a cheap,
  differentiable stand-in for the Kemeth et al. emergent-space construction, and it
  matches the periodic domain.

The learned-position version is self-regularizing in the right direction: spreading the
φ's out *reduces* the constraint burden, so there's no need for an extra repulsion term.
In the toy it recovered an ordering close to the identity (which was the correct answer
there, since the latent had no reason to permute).

### 4b.7 Honest verdict on this route

- It **works** in the sense that the constraint can be satisfied at no reconstruction
  cost with the latent staying full-ish rank. Row 4 of the table is a genuine success.
- It is **still second-best to architectural locality** (§5), because it buys you a
  correlation structure on the latent index, not local *dynamics*. Correlation is not
  causation: nothing in these constraints forces `∂z_{n+1,i}/∂z_{n,j}` to be banded, and
  bandedness of the Jacobian is what a PDE actually requires. **Test that explicitly
  with diagnostic D3 after training** — if the constrained model gives you banded
  covariance but a dense Jacobian, the route has failed at the thing that matters and
  you should stop and go to §5.
- Your point about local (sliding-window) attention for obstruction §2.2 is right, and
  it composes with this: sliding-window attention gives a finite receptive field, and
  the ring constraint then aligns the latent index with that window structure. That
  combination is a reasonable intermediate step between the current model and a full
  patch-local rebuild, and it preserves more of your existing architecture.

**Suggested experimental order:** (i) banded-circulant covariance target (§4b.3, one
line, no dual variables) → (ii) if that's not enough, the full constrained version → in
both cases (iii) run D3 to check whether the *Jacobian* became banded. Only (iii) tells
you whether any of this served the PDE goal.

---

## 5. The recommended program: a local latent *field*

### 5.1 The reframing
Extensivity (D_KY ≈ 0.226·L for L ≳ 80, Edson–Bunder–Mattner–Roberts 2019) says the
KS state at L=100 is *not* a generic 22-dimensional object. It is a chain of local
chaotic units with a fixed dof density. So:

> A latent **vector** has no continuum limit. A latent **field** z_j(t) ∈ R^c on a
> periodic lattice j = 1..P does. Give the latent a spatial index by *construction*,
> and every objection in §2 disappears at once.

### 5.2 Architecture
- Encoder: circular `Conv1d` stack (exact translation equivariance — this also
  *replaces* your shift augmentation with an architectural guarantee), striding down
  to P sites × c channels. Receptive field ~2–3 correlation lengths.
- Decoder: mirror, transposed conv / pixel-shuffle, overlap-add.
- Latent shaped `(P, c)`, **not** flat.
- **Hybrid option:** KS with periodic BC conserves ∫u dx, and translation phase is a
  global object. Consider `z = (few global scalars, local field)` so the local part
  isn't forced to carry global constraints. Worth a small ablation.

### 5.3 Sizing (from extensivity, not guesswork)
dof density ≈ 0.22 per unit length; L = 100; site spacing h = L/P.

| P | h | dof/site ≈ 0.22h | suggested c | total latent | notes |
|---|---|---|---|---|---|
| 16 | 6.25 | 1.4 | 4 | 64 | h borderline vs. cell scale |
| 32 | 3.13 | 0.69 | 2–3 | 64–96 | **recommended starting point** |
| 64 | 1.56 | 0.34 | 1–2 | 64–128 | good for h-refinement test |

Two competing constraints:
- **Continuum limit needs h ≪ correlation length.** The KS cellular scale is
  ℓ₀ = 2√2 π ≈ 8.9. So h ≈ 3 is comfortable; h ≈ 6 is marginal.
- **Closure needs c large enough.** With c = 1 the latent is essentially a local
  coarse-graining of u, and coarse-grained KS is *not closed* — you get memory
  (Mori–Zwanzig). Extra channels are precisely the local hidden variables that restore
  Markovianity.

This gives a genuinely nice result to report: **the minimum c at which the local
dynamics closes is a measurable property of KS**, i.e. "how many local hidden variables
does the coarse-grained description need?" Sweep c ∈ {1,2,3,4} and report rollout error.

### 5.4 The stencil propagator and the falsifiable PDE test
Learn a **shared** local update:

```
ż_j = f( z_{j-1}, z_j, z_{j+1} )        (or a wider stencil / small circular CNN)
```

Same weights at every site (spatial homogeneity → translation equivariance of the
dynamics). Then reparametrize the stencil in terms of centered finite differences:

```
ż = F( z, δ¹z, δ²z, δ³z, δ⁴z )
```

**The test for "is it a PDE?":**

> Train at two (ideally three) site spacings h. If the *same* F — after the usual
> h-scaling of the difference operators — fits all of them, the continuum limit exists
> and you have a PDE: ∂_t z = F(z, ∂_X z, ∂²_X z, …). If F must change with h, you have
> a lattice dynamical system, and the honest answer to Peter Jan is "coupled ODEs on a
> lattice, not a PDE."

This is the crisp criterion your question was missing, and it makes the project
succeed-either-way. Sweep the stencil width too: how far the coupling has to reach
before rollout error saturates is a direct measurement of the **nonlocality of the
effective law**.

### 5.5 Symbolic extraction
Once F is a fixed function of (z, δ^m z), run **PDE-FIND / SINDy** (Rudy, Brunton,
Proctor & Kutz, *Sci. Adv.* 2017) with a library of products of channels and their
spatial derivatives — with the coefficient vector **shared across sites**. Compare with
SINDy-autoencoder (Champion, Lusch, Kutz & Brunton, *PNAS* 116:22445, 2019), which does
the joint coordinate + equation discovery but in the global-latent-vector setting.

### 5.6 Validation targets
1. Does the leading channel behave like a low-pass filtered u? (Sanity.)
2. Does the large-scale sector of the learned equation reproduce **KPZ/Burgers**
   scaling (α = 1/2, z = 3/2)? This is Interpretation C and the real prize.
3. **L-transfer:** train the local model at L=100, then run it at L=200 or L=400 by
   simply adding lattice sites — no retraining. If the rollout statistics and D_KY
   density are right, you have demonstrated you learned a genuine *local law*, not a
   fit to one domain. **This is the headline result if it works.**
4. D_KY of the local model at L=100 should still be ≈ 21–23, matching your current
   result and the literature.

---

## 6. Why this serves the DA goal (the argument to make to Peter Jan)

This is the part that turns "PDE curiosity" into "DA necessity," and it is in Peter
Jan's own language.

**Latent-space DA currently cannot localize.** Localization — tapering spurious
long-range ensemble correlations — is the single technique that lets ensemble DA work
with N_ens ≪ state dimension. It requires a *distance* between state variables. Your
current latent coordinates have no distance, so you cannot localize, and your PFF
ensemble size must scale with the full attractor dimension (~22).

**A local latent field restores all of it:**
- Gaspari–Cohn tapering on the latent lattice index → a localized B.
- Local observations inform only nearby latent sites → the observation Jacobian
  J = ∂h(D_ψ(z))/∂z becomes **banded**, with bandwidth set by the encoder receptive
  field.
- Your NAT-PFF metric `F = J̄ᵀR⁻¹J̄ + B⁻¹` becomes banded → cheaper Cholesky, better
  conditioning, and `robust_cholesky` fallbacks get less exercise.
- The PFF kernel bandwidth can itself be made local.
- Ensemble size then scales with *local* dimension (~1–3 per site), not global.
- You inherit the L-transfer property: assimilate at L=200 with a model trained at
  L=100.

Caveat to state explicitly: latent localization is only valid because the encoder
receptive field is finite. **The receptive field sets the minimum admissible
localization radius** — that's a clean, quotable design rule and a nice small
theoretical contribution in itself.

---

## 7. Alternative and complementary directions

### 7.1 Symmetry reduction before encoding (cheap, high value)
Use the **method of slices** (Budanur, Cvitanović, Davidchack, Siminos) to quotient out
continuous translation before the autoencoder. This:
- explains, and fixes, why SFA/DMD/seriation found nothing (the attractor is a group
  orbit; travelling-wave coherence lives in the phase you never separated);
- removes one dimension;
- cleanly splits **phase** (one global drift variable) from **shape**;
- makes travelling waves → relative equilibria → fixed points, so your topological
  analysis becomes much sharper.

Reference for the state-space geometry: Cvitanović, Davidchack & Siminos, *SIAM J.
Appl. Dyn. Syst.* 9(1):1–33 (2010).

### 7.2 Spectral ("Fourier") latent instead of real-space
Rather than a real-space lattice, constrain latents to be narrow-band in wavenumber
(D2 will tell you whether they already are). KS in Fourier is exactly a mode-coupling
ODE system:

```
ẑ_k' = (k² − k⁴) ẑ_k − (ik/2) Σ_m ẑ_m ẑ_{k−m}
```

A nonlinear-Galerkin / approximate-inertial-form latent in this spirit is arguably more
faithful to KS than a real-space stencil, and connects directly to the inertial-manifold
literature. The "PDE" then reads off as the mode-coupling structure. This is a genuine
alternative answer to Peter Jan's question and it may be *easier* than the spatial one.

### 7.3 Local delay embedding instead of extra channels
Instead of c > 1 channels per site, use time-delays of a *scalar* local latent (Takens,
applied locally). More interpretable, connects to your existing 2-step-history
propagator, and gives a second independent estimate of the local closure dimension.

### 7.4 Neural ODE latent
Switch the discrete propagator for a neural ODE, so "the ODE system" is explicit rather
than implicit in a map. Makes the continuum-limit statement in §5.4 cleaner (you compare
∂_t z, not one-step maps) and makes your Lyapunov analysis exact rather than a
one-step approximation. Both Linot & Graham papers already in the project folder are
the direct template.

### 7.5 Mori–Zwanzig framing
If c = 1 fails to close, that failure *is* the MZ memory term. Explicitly modeling it
(auxiliary local memory variables, or a short delay window) turns a bug into a
contribution. See MZ latent-Koopman-closure work (arXiv:2310.10745) for a KS-specific
treatment.

---

## 8. Honest risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| **Prior art overlap.** Constante-Amores, Linot & Graham (arXiv:2410.01238, *Phys. Rev. E* 2026) already do patch-decomposed AE + NODE local models on KS and Kolmogorov flow, with shared weights across patches. | **High — read this first.** | Their contribution is the distributed ROM. Yours would be (i) continuum-limit / PDE extraction with h-refinement consistency, (ii) latent localization for ensemble/particle-flow DA, (iii) L-transfer. Confirm which of these they leave open before committing. |
| Reconstruction degrades with a local encoder. | Medium | AROMA (arXiv:2406.02176) reports KS as an explicit failure case: chaotic spectra needed recon MSE ~1e-10–1e-12, and the *decoder* was the bottleneck. Budget for this; measure spectra, not just MSE. |
| No continuum limit exists at any affordable h. | Medium | This is a *result*, not a failure — reported with the quantitative h-refinement criterion, it answers Peter Jan's question definitively. Frame it that way in advance. |
| Global constraints (conserved ∫u, phase) don't fit a purely local latent. | Low–Medium | Hybrid latent (§5.2). |
| Time cost of a full v6 rebuild. | Medium | Diagnostics in §3 first (days). Only rebuild if D3/D4/D5 are encouraging. |

**Overall assessment:** the request as literally posed is not well-posed, and you should
say so plainly. The reframed version — *local latent field → effective coarse-grained
equation → localization-enabled latent DA → transfer across L* — is well-posed, novel
enough, physically motivated, has a known validation target (KPZ), and directly serves
the DA objective rather than competing with it. I'd advocate for it.

---

## 9. Proposed sequence

| # | Task | Cost | Gate |
|---|---|---|---|
| 1 | Diagnostics D1–D5 (§3) on existing checkpoints | 2–4 days | Do these regardless |
| 2 | Read arXiv:2410.01238, Kemeth 2022, re-read Linot & Graham | 1–2 days | Novelty check |
| 3 | Conversation with Peter Jan: which interpretation (A/B/C) does he want? | — | Determines everything |
| 4 | Scale-invariant smoothness diagnostic (§4 Fix 1+2) on current AE | 1–2 days | Evidence for/against hidden locality |
| 5 | Local-latent AE v6, P=32, c ∈ {1,2,3,4}, circular convs | 1–2 weeks | Compare recon, rollout, D_KY vs. current |
| 6 | Shared-stencil propagator; stencil-width sweep | 1 week | Nonlocality measurement |
| 7 | h-refinement consistency test (P = 16/32/64) | 1 week | **The PDE/ODE verdict** |
| 8 | PDE-FIND on F; KPZ scaling check | 1 week | Interpretation C |
| 9 | Localized latent PFF; ensemble-size sweep; L=200 transfer | 1–2 weeks | The DA payoff |

**Go / no-go for the PDE claim:** if D3 and D4 show no recoverable locality *and* the
h-refinement test in step 7 fails at two resolutions, the defensible answer is
"coupled ODEs on a lattice, not a PDE" — and that is worth writing up.

---

## 10. Related work log

### Directly on "PDE without a spatial coordinate"
- **Kemeth, Bertalan, Thiem, Dietrich, Moon, Laing & Kevrekidis (2022)**, "Learning
  emergent partial differential equations in a learned emergent space," *Nat. Commun.*
  13:3318. arXiv:2012.12738. — **The central reference.** Manifold learning to build an
  emergent spatial coordinate for coupled agents with no natural space, then learn a PDE
  local in that coordinate. Also captures collective bifurcations.
- **Kemeth et al. (2018)**, "An emergent space for distributed data with hidden internal
  order through manifold learning," *IEEE Access* 6:77402. — Predecessor.
- **Arbabi, Kevrekidis et al.**, "Particles to Partial Differential Equations
  Parsimoniously," arXiv:2011.04517. — Coarse-grained PDE discovery from fine-scale
  data, linked to equation-free multiscale computation.
- **Reinbold & Grigoriev (2019)**, "Data-driven discovery of PDE models with latent
  variables," *Phys. Rev. E* 100:022219.
- **Kevrekidis et al. (2003)**, equation-free / coarse-grained multiscale computation,
  *Commun. Math. Sci.* 1(4):715. — The "patch dynamics" lineage; the conceptual
  ancestor of ODE-lattice ↔ PDE transfer.

### Local / patch-decomposed reduced-order models (closest prior art)
- **Constante-Amores, Linot & Graham (2024/2026)**, "Data-driven prediction of
  large-scale spatiotemporal chaos with distributed low-dimensional models,"
  arXiv:2410.01238, *Phys. Rev. E*. — **Read first.** Patches + shared autoencoders +
  shared NODEs on KS and 2D Kolmogorov flow; up to 2 orders of magnitude dimension
  reduction; motivated explicitly by attractor dimension scaling linearly with domain
  size. Closest thing to §5.
- **Zeng, Pérez-De-Jesús, Fox & Graham (2024)**, "Autoencoders for discovering manifold
  dimension and coordinates in data from complex dynamical systems," *MLST* 5:025053,
  arXiv:2305.01090. — IRMAE-WD; applies it to **local patches** as well as global data.
  Direct precedent for diagnostic D5.
- **Linot & Graham (2022)**, "Data-driven reduced-order modeling of spatiotemporal chaos
  with neural ODEs," *Chaos* 32:073110. — *(already in project folder)*
- **Linot & Graham**, "Deep learning to discover and predict dynamics on an inertial
  manifold." — *(already in project folder)*
- **AROMA (2024)**, "Preserving Spatial Structure for Latent PDE Modeling with Local
  Neural Fields," arXiv:2406.02176. — Spatially-structured latents via local neural
  fields. **Appendix C.7 documents KS as a failure case** — read it as a warning about
  decoder-limited reconstruction of chaotic spectra.
- **Hybrid Latent Representations for PDE Emulation**, NeurIPS 2025. — Coarsened PDE
  fields augmented with spatially structured latents; explicitly motivated by the fact
  that unstructured latents "cannot enforce local interactions as inductive biases."
  Same diagnosis as §2 here.
- **Latent Neural PDE Solver (LNS)**, *JCP* 2025. — Latent dynamics on a coarser mesh
  rather than an unstructured vector.

### Effective large-scale behavior of KS (validation target)
- **Yakhot (1981)**, *Phys. Rev. A* 24:642. — Large-scale properties of KS.
- **Zaleski (1989)**; **Sneppen et al. (1992)**; **Jayaprakash, Hayot & Pandit** — KPZ
  universality of coarse-grained KS; 2D version via explicit coarse-graining.
- **Minami & Sasa (2018)**, *J. Stat. Phys.* 173:120. — Effective model for universal
  behavior of noisy KS.
- **Functional RG treatments (2026)**, arXiv:2605.23364 and arXiv:2607.15784. — Three
  scaling regimes of KS: KPZ (z=3/2), Edwards–Wilkinson, and inviscid-Burgers (z=1).
  Effective viscosity flows from negative (KS) to positive (KPZ). Recent and directly
  relevant if you pursue Interpretation C.

### Equation discovery
- **Rudy, Brunton, Proctor & Kutz (2017)**, "Data-driven discovery of PDEs" (PDE-FIND),
  *Sci. Adv.* 3:e1602614.
- **Champion, Lusch, Kutz & Brunton (2019)**, "Data-driven discovery of coordinates and
  governing equations," *PNAS* 116:22445. — SINDy autoencoder; joint coordinate +
  equation discovery, global latent version.
- **Long, Lu, Ma & Dong**, PDE-Net (ICML 2018) and PDE-Net 2.0 (*JCP* 399:108925).
- **arXiv:2608.20404 (2026)**, "Robust Discovery of Coarse-Grained Continuum Equations
  from Microscopic Dynamics." — PDE-SINDy data/library/noise scaling study; useful
  practical guidance (larger libraries *hurt*; spurious terms suppress with more data).

### Constrained optimization in NN training
- **Platt & Barr (1988)**, "Constrained differential optimization," NIPS. — MDMM /
  damped dual ascent; the standard stabilizer for §4.
- **VICReg** (Bardes, Ponce & LeCun, 2022) — variance hinge + covariance term; the
  collapse-proof replacement for the current `L_var`.
- **Barlow Twins** (Zbontar et al., 2021) — redundancy-reduction alternative.

### Symmetry and state-space geometry of KS
- **Cvitanović, Davidchack & Siminos (2010)**, "On the state space geometry of the KS
  flow in a periodic domain," *SIAM J. Appl. Dyn. Syst.* 9(1):1–33. arXiv:0709.2944.
- **Budanur, Cvitanović et al.** — method of slices / symmetry reduction.
- **Ding, Chaté, Cvitanović, Siminos & Takeuchi (2016)**, "Estimating the dimension of
  an inertial manifold from unstable periodic orbits," *PRL* 117:024101.

### Latent-space DA (context for §6)
- **Peyron et al. (2021)**, "Latent space data assimilation by using deep learning,"
  *QJRMS* 147:3759.
- **D-LSPF (2024)**, "The deep latent space particle filter for real-time data
  assimilation with UQ," *Sci. Rep.* — Latent particle filtering with transformer
  latent time-stepping. Closest DA analogue to your PFF setup.
- **EnKF in latent space using a VAE pair**, arXiv:2502.12987 — includes a useful table
  surveying ML+DA latent approaches, and explicitly frames localization/inflation as
  among the challenges latent methods are meant to address. **Check whether anyone has
  done latent-space *localization*; I did not find it, which is a good sign for §6.**
- **Adjoint-based optimization with quantized local ROMs**, arXiv:2603.05531 —
  variational DA for KS with *local* ROMs. Overlaps with the DA-side motivation; worth
  reading for positioning.

### Lyapunov/stability analysis and linearization in learned latent spaces (2026-09-12 addition)
- **"Stability analysis of chaotic systems in latent spaces"**, PMC11982125. — CAE-ESN
  (convolutional autoencoder + echo state network) computes Lyapunov exponents, D_KY, and
  covariant Lyapunov vectors *directly in the learned latent space* for KS and Kolmogorov
  flow, 1.5% mean error vs. the reference spectrum. **Direct parallel to this project's own
  Gate 3/4 Benettin-in-latent-space work** — read for how they validate latent Lyapunov
  exponents against ground truth, and whether their architecture/training suggests
  anything for the standalone-pde-divergence problem seen across Sections 153–158.
- **arXiv:2503.14702**, "Learning Chaos In A Linear Way" (Poincaré Flow Neural Network,
  PFNN). — Linearizes chaotic evolution in a learned finite-dimensional feature space
  (Koopman-like), explicitly targeting long-term contraction/measure-invariance rather
  than short-horizon autoregressive accuracy, on the argument that exponential error
  growth makes short-horizon fit a poor optimization target for chaotic systems.
  Relevant to the short-horizon-fit-vs-standalone-stability tension found across every
  `pde_head` variant tried this arc (L1, Chebyshev, fixed-linear-terms) — a genuinely
  different mechanism (linearizing map) for the same tension.
- **arXiv:2309.05812**, "Interpretable Learning of Effective Dynamics for Multiscale
  Systems" (iLED; Menier, Kaltenbach, Yagoubi, Schoenauer & Koumoutsakos). — Autoencoder
  + Mori-Zwanzig/Koopman-grounded **linear** latent operator + a neural **nonlinear
  closure** correction term; tested on FitzHugh-Nagumo, KS, and cylinder flow. **Closely
  relevant to this arc's own dominant-linear-term ↔ periodicity diagnosis**: their
  finding is that the latent dynamics decompose into a dominant linear part plus a
  smaller nonlinear correction, which is structurally the same split
  `pde_coeff_l1_linear_only`/`poly_fixed_linear_terms` were built to separate and
  control (Sections 153–158) — read for how they keep the nonlinear closure well-behaved
  under standalone rollout, since that is exactly where this arc's own closures
  (Sections 155–158) have failed.

### Already in the project handoff (retained for completeness)
- Edson, Bunder, Mattner & Roberts (2019), *ANZIAM J.* — KS Lyapunov spectrum;
  D_KY ≈ 0.226·L. **This is the quantitative basis for the extensivity argument in §5.**
- Whitney (1936, 1944); Takens (1981); Sauer–Yorke–Casdagli (1991) — embedding.

---

## 11. Open questions to resolve

1. **Which interpretation (A/B/C) does Peter Jan actually want?** Ask directly. If A,
   the project is a 2-week neural-ODE + SINDy exercise. If B or C, it's the §5 program.
2. Does D4 (translation representation) succeed? If yes, there may be a fast path to a
   spectral-coordinate answer with no rebuild at all.
3. What exactly does arXiv:2410.01238 leave open? Novelty hinges on this.
4. Has anyone done localization in a learned latent space? Preliminary searching says
   no. If confirmed, §6 is a paper on its own.
5. Is the minimum closure channel count c a stable, reportable number for KS?
6. Does the L=100 → L=200 transfer work? This is the single highest-value experiment
   in the whole plan.

---

*Maintained alongside `PROJECT_HANDOFF.md` and `ML_for_KS_writeup.md`.*
