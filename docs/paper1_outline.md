> **SUPERSEDED (2026-10-08).** This outline guided the first draft. Its numbers are stale (see `docs/paper1_review.md` §10 and `docs/paper1/decisions.md`); the paper itself (`docs/paper1/`) and `docs/full_paper_tasks.md` are the current sources. Do not quote numbers from this file.

# Paper 1 — detailed outline

**Working title.** *Local latent fields for extensive spatiotemporal chaos: zero-shot domain
transfer, training on windows, and exactly localized latent data assimilation*

Alternative titles:
- *Learning the local rule: weight-shared latent fields that transfer across domain size in
  extensive chaos, and what that buys data assimilation*
- *Train on a window, deploy at any size: local latent surrogates for spatiotemporal chaos*

**Source material.** Spine = `ks_local_field_ltransfer.tex` Section 15 (`sec:consolidation`,
Phases 0–5) and the publication plan it points to (Section 14, `sec:status_plan`). Supporting
numbers from the rest of the tex record and `docs/paper1_tasks.md`. Every number below is
copied from those documents or the result JSONs in `artifacts/`; section labels in
`[brackets]` point to the tex record where the full table lives.

**Deliberate deviations from the §14/§15 outline** (reasons in parentheses):
1. *Lead with the scaling claim, not single-point $D_{KY}$ matches* (user decision 2026-10-07,
   `sec:threats`: Kolmogorov truths carry 4–13% seed spread; 1D KS is the only system where
   percent-level matching is trustworthy).
2. *Promote the DA section from "1D demonstration" to a co-headline.* The 2D results
   (block-diagonal Jacobian, exact per-site LETKF, dense filter inoperable at 88²) are
   stronger and more novel than the §14 plan anticipated.
3. *Group all "what did not work" results into one section* rather than scattering them;
   they are a coherent second contribution (training distribution > architecture; MSE
   damping; coupling makes transfer worse).
4. *Drop from paper 1:* transfer-error-as-physics-diagnostic and drag-sweep (TD1–TD7, P8,
   α=0), residual hierarchy (R1–R6), self-similarity (C16–C18), SWE, Kolmogorov DA,
   per-patch-vs-full-reach DA (E9/E10). These are paper 2. Keep one paragraph in the
   Discussion pointing at the diagnostic idea.

---

## Decisions log

- **2026-10-07 (user):** (1) Treat the existing single-seed α=0.1 3× truth (56.35) as
  truth; the paused seeds 1–2 are not resumed. (2) Main-text α=0.1 Kolmogorov model is
  `k_xs01_v5` (k_max=16, w_logdet=0.006, w_var=0.005, ε=1e-3; 3× 2.3%, spread 1.2).
  (3) Target venue *Chaos* (revtex4-2, `aip,cha`). §3 Method and Appendix B written
  (`docs/paper1/`).
- **2026-10-07 (found while writing §3):** the 2D DA script deployed the 2D KS model at
  5× the training resolution (see W0 below). §7's 2D skill/calibration numbers must be
  re-run before they are used; the computational results (exact block-diagonal Jacobian,
  colouring speed-up, dense-filter OOM at d=38,400) are unaffected.

- **2026-10-08:** §4 (Systems, data and ground truth) and Appendix C written
  (`docs/paper1/sections/04_systems.tex`, `appendix/C_ground_truth.tex`, figure script
  `scripts/paper_figures/figC1_truth_convergence.py`). User decision: existing truth runs
  are the reference; amend later if more truths are run. Convention adopted: mean of
  multiple runs where they exist (α=0.05 2×: 48.59; α=0.1 2×: 30.87), matched-IC truths at
  α=0.05 1× (22.46/20.52/21.63), single runs elsewhere. **Correction found:** the true 1D
  D_KY at L=100 is 22.5 (Gate 1, `docs/REPLICATION_LOG.md`), not 22.14 (the research
  tex's E1 table put the model's value in the truth column) — model native error is 1.6%.
  **New quantitative finding for §4/§6:** 40-unit Kolmogorov truths differ from their
  120-unit continuations by −17% to +23% (mean +10%; 4 of 5 high), so single-seed
  40-unit truths at 1×/3× may be biased upward — relevant to reading the 3× "underestimate".

- **2026-10-08:** §5 (1D results) written, `docs/paper1/sections/05_results_1d.tex`
  (Table III + Fig. 3). Uses truth 22.5 at L=100 (model error 1.6%); notes the ~20%
  decoded-amplitude undershoot and the L=25 kernel-wrap failure.

- **2026-10-08:** §6 (2D results) written, `docs/paper1/sections/06_results_2d.tex`
  (Table IV + Figs 5, 6). Notes: at α=0.1 box training is fine at 2× (1.9%) and window
  training regresses there (18.8%); the α=0.05 main model `c19_e0d0_c6` uses the original
  k_max=12 (not long rollouts) — §3's canonical-recipe paragraph corrected to say k_max is
  12–32 by model; 2D KS density flat 83.3–83.6 per 44² box at all three sizes; at 3× the
  α=0.025 model falls below α=0.05, unlike the truth.

## 0. Target, format, length

- **Venue candidates:** *Chaos* (AIP) or *Physica D* (dynamics audience, values Lyapunov
  validation); *J. Comput. Phys.* (method audience); *Nonlinear Processes in Geophysics* /
  *JAMES* if the DA section ends up leading. Recommend writing for *Chaos*/*Physica D*:
  the Lyapunov-level validation is the unusual strength, and both take long papers.
- **Length:** ~25–30 pages main text + ~15 pages appendix. 9–10 main figures, 5–6 main
  tables.
- **TeX layout** (new directory, separate from the research record):
  ```
  docs/paper1/
    main.tex              % preamble, \input of sections, bibliography
    macros.tex            % \DKY, \dlat, \E, \D, \F, theorem envs, etc.
    sections/01_intro.tex ... sections/11_conclusion.tex
    appendix/A_architecture.tex ... appendix/F_reproducibility.tex
    refs.bib
    figures/              % PDFs produced by scripts/paper_figures/*.py
  scripts/paper_figures/  % one script per figure, reads artifacts/, writes PDF + .meta.json
  ```
- **Class/packages:** start with `article` (11pt, 1in margins) + `amsmath, amssymb, amsthm,
  mathtools, booktabs, siunitx, graphicx, subcaption, xcolor, tikz, natbib, hyperref,
  cleveref, microtype`; trivially swappable to `revtex4-2` (Chaos) or `elsarticle`
  (Physica D/JCP) once the venue is fixed. Theorem environments: `proposition`, `lemma`,
  `remark`. Figures: vector PDF, one consistent palette, Computer Modern/STIX text in
  figures to match the body.

---

## 1. Claims (what the paper must support)

Calibrated to the evidence actually in hand.

- **C1 — Architecture.** A latent *field* $z\in\mathbb{R}^{P\times c}$ built only from
  weight-shared, finite-reach local operators has a parameter count independent of domain
  size and is exactly translation-equivariant, so one trained model can be instantiated at
  any domain size by adding sites. A global latent vector cannot even be evaluated at a
  new size.
- **C2 — Zero-shot transfer reproduces extensive scaling.** 1D KS: trained at $L=100$,
  $D_{KY}$ within 2% of independent ground truth at $L=50,200,400$ (0.60% at $4\times$),
  three training seeds agree to 0.1%. 2D KS: within ~3% at $1.5\times$ per side and
  extensive density at $2\times$ per side (88², no truth yet). Kolmogorov flow:
  $D_{KY}$ scaling with channel width reproduced across three drags to within the truth's
  own measured seed spread.
- **C3 — The training distribution decides transfer, not the architecture.** The same
  architecture trained on a small periodic box fails under transfer (24–50% error); trained
  on windows (crops) of one large domain with forced-halo rollouts it transfers (1–6%).
  Adding cross-patch coupling while box-training makes transfer *worse* (D2 95%, D3 99%).
  Window training is also the realistic setting (regional data of an open flow).
- **C4 — A per-patch autoencoder suffices and makes latent DA exactly local.** Zero
  encoder/decoder reach with $c=6$ matches or beats full reach (C19/Phase 0). Its decoder
  Jacobian is exactly block-diagonal, so (i) the observation Jacobian costs $c$ forward
  passes at any domain size, (ii) a per-site (Hunt et al.) LETKF is the exact local
  problem, (iii) memory/cost scale with area. The dense covariance-localized filter is
  killed by the OS at 88² ($d=38{,}400$); the local filter runs in <1 GB.
- **C5 — Localization becomes load-bearing as the domain grows** (1D): at fixed ensemble
  size, unlocalized DA goes from competitive at $L=100$ to worse than the free run at
  $L=200$–$400$ for every $N_{ens}\le 256$, under both PFF and LETKF; an empirically fitted
  SEC taper cannot be carried to the new size (indefinite covariance at 5/6 sizes at
  $L=200$, 6/6 at $L=400$).
- **C6 — Honest negative results and failure modes**: SEC beats GC at the trained size;
  softmax attention propagators collapse; MSE rollouts damp chaos (a $k_{\max}\times$
  log-det-barrier interaction, not one knob); spectral regularizer hurts at $3\times$;
  learned consensus/exact Biot–Savart input do not help; short-horizon training produces
  spurious multistable regimes (P9); truth-side $D_{KY}$ uncertainty limits what can be
  claimed.

**Claims to avoid** (prior art or not supported): "KS admits local reduced models"
(Wittenberg & Holmes 1999); "first patch-trained surrogate of large chaotic domains"
(Pathak et al. 2018; Constante-Amores et al. 2024); "first zero-shot domain transfer"
(arXiv:2606.14597); any percent-level Kolmogorov $D_{KY}$ match stated without truth error
bars; "first demonstration that a learned latent reproduces an extensive scaling law" —
keep only after the literature verification pass.

---

## 2. Section-by-section outline (paper order)

### Abstract (~200 words) — write last
Problem (global latent vectors cannot scale with extensive chaos) → method (weight-shared
local latent field, window/crop training with forced halos, per-patch autoencoder) →
results (1D KS 0.6–2% at up to 4×; 2D KS and Kolmogorov scaling reproduced; box vs window
24–50% vs 1–6%) → DA (exact block-diagonal observation Jacobian; per-site LETKF runs where
the dense filter cannot; localization becomes necessary with size) → honest scope
(negative result at fixed size; truth uncertainty).

### 1. Introduction (~2 pages)
1. Latent surrogates for spatiotemporal chaos are usually one global vector (Linot &
   Graham; Vlachas et al.; ROAD-EnKF; LAE-EnKF). For *extensive* chaos,
   $D_{KY}\propto$ volume (Ruelle 1982; Manneville 1985; Egolf & Greenside 1994), so a
   fixed-size latent has no continuum limit and cannot be evaluated at another size.
2. Two practical consequences: (a) re-training per domain size; (b) ensemble DA in a
   global latent needs $N_{ens}\sim$ global dimension, and localization has no distance to
   act on (Patil et al. 2001 "local low dimensionality"; Ott et al. 2004).
3. Idea: make the latent a field with a local, weight-shared rule; then transfer is "apply
   the same rule on more sites", and localization has a physical distance.
4. What is hard (and what the paper is really about): the rule must be learned from data
   that exhibit the large-domain local statistics (window training), the propagator must
   stay chaotic under free rollout (MSE damping), and DA must stay local *through the
   decoder* (per-patch autoencoder).
5. Contributions list C1–C6 (bulleted, one line each, with the headline number).
6. Positioning sentence: minimal flow units are non-extensive by construction, which is
   why global latents work there (Constante-Amores, Linot & Graham 2024b) and not here.
7. Roadmap.

### 2. Background and related work (~2.5 pages)
Keep tight in the main text; the long version can go to an appendix if a venue wants it.
- 2.1 Extensive chaos and Lyapunov dimension (KS, 2D KS, Kolmogorov flow with drag).
- 2.2 Data-driven surrogates of spatiotemporal chaos: global latents, local/parallel
  reservoirs, patch-decomposed AEs, learned discretizations/closures, neural operators and
  domain-size generalization, rollout stability and "blurring".
- 2.3 Latent-space and localized ensemble DA.
- 2.4 Pre-neural local models of KS (Wittenberg & Holmes) — state what is and is not new.
- Closing paragraph: table of closest prior work vs. features (adapt `tab:td_prior` in
  `sec:td_literature`: local/equivariant, zero-shot $L$, Lyapunov validation, window
  training, localized latent DA).
Full reference plan in §5 below.

### 3. Method (~5 pages) — the mathematical core
**3.1 Notation and setting.** Periodic domain $\Omega_L$ (1D length $L$, 2D $L_x\times L_y$),
grid spacing $\Delta x$ fixed, snapshot interval $\Delta t$; state $u_n$; site lattice
$\Lambda_P$ with $P$ sites of physical size $h$ (patch $= h/\Delta x$ grid points).

**3.2 Local latent field.** Encoder $E:u\mapsto z$, $z_j=e_\theta(u|_{B_{r_e}(j)})\in
\mathbb{R}^c$; decoder $D$ with reach $r_d$; propagator
$$z^{n+1}_j = z^n_j + f_\theta\big(\{z^n_k\}_{|k-j|\le w}\big),$$
$f_\theta$ = `local_mlp` (windowed *unconstrained linear* token mixer + per-site MLP, no
softmax, zero-initialized output so $F=\mathrm{id}$ at init). Gauge anchor: channel 0 =
patch average of $u$ (1D/early 2D); note canonical 2D model's per-patch $c=6$.
- **Proposition 1 (equivariance and size-independence).** If every layer is a
  weight-shared operator of finite reach on the periodic lattice, then
  $E\circ T_{s h} = S_s\circ E$, $F\circ S_s=S_s\circ F$, and $|\theta|$ does not depend on
  $P$. *Proof:* two lines (convolution commutes with shifts; kernel size fixed).
- **Corollary (transfer map).** Holding $\Delta x$ and $h$ fixed, the trained $\theta$
  defines a model on $\Omega_{L'}$ for every $L'$ that is a multiple of $h$; this is the
  zero-shot transfer evaluated throughout.
- **Remark (extensivity of local models).** A translation-invariant lattice map with
  finite reach has a Lyapunov spectrum with a thermodynamic limit (Ruelle 1982; coupled
  map lattice results, e.g. Isola et al. 1990 — verify), so $D_{KY}^{\text{model}}(L)
  =\rho_\theta L + o(L)$. Consequence: the model is extensive *by construction*; it can
  match the truth's density $\rho$ but cannot reproduce finite-size super- or
  sub-extensivity. This explains why transfer error tracks the truth's deviation from
  extensivity (G2: 4.5%↔4.5–5.9%, 18.3%↔19.4%; P1/P2 signs). Heuristic, not a theorem.

**3.3 Receptive fields and the light cone.** $R_{\text{enc}}=r_e n_e$, $R_{\text{prop}}=w
n_\ell$ sites per step, $R_k = R_{\text{enc}}+kR_{\text{prop}}+R_{\text{dec}}$; measured
by Jacobian masking, not trusted from the formula. Physical bound
$h R_{\text{prop}}\gtrsim v_*\Delta t$ (comoving Lyapunov / information speed; Deissler &
Kaneko 1987, Pikovsky & Politi 1998); localization radius
$r_{\text{loc}}\ge\max(R_{\text{enc}}h, v_*\Delta t)$, which reduces to $v_*\Delta t$ when
encoder reach is 0.

**3.4 Training on windows of one large domain (crop training).** One large simulation;
random crops; Stage A (autoencoder) loss masked to sites whose receptive field lies in the
crop (margin $R_{\text{enc}}+R_{\text{dec}}$, zero for per-patch); Stage B (propagator)
with a *forced halo* of $R_{\text{prop}}$ sites overwritten by the true encoding at every
rollout step, loss on the free interior only.
- **Proposition 2 (forced-halo exactness).** If halo $\ge R_{\text{prop}}$, the k-step
  free-interior prediction equals the prediction of the same weights on any larger periodic
  domain whose sites outside the interior follow the true trajectory (a limited-area model
  with exact lateral boundary data). With halo $< R_{\text{prop}}$ equality fails. *Proof*
  by induction on $k$ (light-cone argument). Unit-tested to $10^{-6}$ (1D) and $10^{-5}$
  (2D), including the failure case.
- Why it fixes box training: no periodic seam to exploit (the box-trained propagator's
  6-site reach covers 75% of an 8-site ring), and local statistics are those of the large
  domain.
- 2D KS: crops in both directions; Kolmogorov: $x$-strips with full periodic $y$, $y$
  offsets aligned to the forcing period.

**3.5 Objective and the MSE damping problem.** Stage B loss
$\sum_{k\le K}\|\hat z_k - z_k\|^2$ on the interior with $K$ ramped $2\to k_{\text{mid}}\to
k_{\max}$; input-noise anneal. Stage A regularizers: per-site variance floor
$w_{\text{var}}$ and log-det barrier $-w_{\text{logdet}}\log\det(\mathrm{Cov}(z)+\epsilon I)$.
- **Lemma 3 (MSE contracts beyond the predictability horizon).** The minimizer of
  $\mathbb{E}\|\Phi(z_0)-z_k\|^2$ is $\mathbb{E}[z_k\mid z_0]$, and by the law of total
  variance $\mathrm{Var}(\mathbb{E}[z_k\mid z_0]) = \mathrm{Var}(z_k)-\mathbb{E}\,
  \mathrm{Var}(z_k\mid z_0)$, strictly smaller once $k$ exceeds the predictability time.
  A propagator trained to this target is pushed toward contraction along the most
  unstable directions → systematic $D_{KY}$ under-estimation that grows with $k_{\max}$.
  Standard argument (cite "blurring" in ML weather: Bonavita 2024; Lam et al. 2023), stated
  as a lemma because it organizes §8.
- **Remark (why $\epsilon$ in the log-det barrier matters).** $\partial/\partial\lambda_i
  [-\log(\lambda_i+\epsilon)] = -1/(\lambda_i+\epsilon)$; measured smallest eigenvalues of
  $\mathrm{Cov}(z)$ at init are $0.8$–$4.4\times10^{-3}$ (10 seeds), so $\epsilon=10^{-3}$
  vs $0$ changes the barrier's force on the weakest directions by O(1). Explains why
  $\epsilon$ interacts with $k_{\max}$ in the B1 ablation (§8.4).
- Why $w_{\text{var}}$ is still needed: the log-det barrier only penalizes near-singular
  covariance and actually rewards all channels growing together, so with
  $w_{\text{var}}=0$ the encoder and decoder rescale against each other ($l_{\text{var}}$
  $831\to2.5\times10^8$ while reconstruction stays flat). $w_{\text{var}}$ anchors the
  scale; $w_{\text{logdet}}$ prevents correlated collapse. Different jobs.

**3.6 Evaluation protocol.** Benettin QR Lyapunov spectra (float64, CPU), Kaplan–Yorke
$D_{KY}=k+\sum_{i\le k}\lambda_i/|\lambda_{k+1}|$; model spectra from the propagator's
tangent map; IC normalization with training statistics (bug found and fixed:
`sec:icnorm`); 3 IC seeds per size; free-rollout statistics (variance, energy split by
scale) as an early-warning diagnostic. Truth protocol: 200-unit spinup, 120-unit averaging
with running $D_{KY}$ recorded at $t=20,\dots,120$; multi-seed where available.

**3.7 Latent ensemble DA** (can also live in §7.1). Observations $y=H(u)+\eta$ of the
physical field; latent filters with $h=H\circ D$. NAT particle flow filter (Hu & van
Leeuwen 2021) and stochastic EnKF with Schur-product localization (Houtekamer & Mitchell)
in 1D; per-site LETKF with R-localization (Hunt et al. 2007) in 2D. Gaspari–Cohn taper on
site distance (toroidal in 2D). SEC (Anderson 2012) as the distance-free baseline.
- **Proposition 4 (block-diagonal observation Jacobian).** If $r_d=0$, then
  $\partial (H\circ D)_i/\partial z_j=0$ unless observation $i$ lies in patch $j$.
  *Corollary 4a (colouring):* the full $(m,d)$ Jacobian is recovered exactly from $c$ JVPs
  with tangents $e_k\otimes\mathbf{1}_{\text{sites}}$ (Curtis–Powell–Reid colouring), at
  any domain size. *Corollary 4b (exact locality):* the per-site LETKF analysis at site $j$
  depends only on observations and ensemble columns inside the localization disk — no
  truncation of $h$ is involved (unlike any decoder with $r_d>0$).
- Ensemble-size heuristic: localized filter needs $N_{ens}\gtrsim\rho\,|A_{\text{loc}}|$
  (local dimension inside the taper support), e.g. 2D KS $0.043\times137\approx6$;
  unlocalized needs $N_{ens}\gtrsim D_{KY}$ (82/193/~330 at 44²/66²/88²).

### 4. Systems and ground truth (~2.5 pages)
- 4.1 **1D KS** $u_t+uu_x+u_{xx}+u_{xxxx}=0$; ETDRK4 with contour-integral coefficients;
  $L=100$, $NX=256$, $\Delta t=1.0$. Validation gate: $D_{KY}\in[21,24]$ at $L=100$
  (measured 22.14), $[5.2,5.6]$ at $L=22$.
- 4.2 **2D KS** $u_t+\nabla^2u+\nabla^4u+\tfrac12|\nabla u|^2=0$; mean-mode projection
  (the 2D nonlinearity drives the $k=0$ mode); $44^2$ box, $NX=128$; wide training domain
  $6\times44$ per axis ($768^2$). Truths: $81.77$ ($44^2$, 150 dirs), $192.5$ ($66^2$).
  Density $0.0422\to0.0442$ per unit area.
- 4.3 **Kolmogorov flow with linear drag** $\omega_t+u\cdot\nabla\omega=\nu\nabla^2\omega
  -\alpha\omega+f(y)$, $n=4$, $\nu=0.01$ ($Re\approx46$–51 in Vinograd et al.'s
  convention), $\alpha\in\{0.025,0.05,0.1\}$; laminar fixed-point and symmetry tests;
  wide domain $16\times2\pi$ by $2\pi$. Transfer grows $L_x$ only (channel widening; state
  explicitly — `sec:xy_growth_asymmetry`).
- 4.4 **How precise is the truth?** (new, important) Table of all truth values with seed
  counts and averaging windows; convergence curves (running $D_{KY}$). Key facts:
  - 1D KS: sub-percent reliability.
  - Kolmogorov $\alpha=0.05$: $1\times$ per-IC 22.46/20.52/21.63 (mean 21.54); $2\times$
    46.62/46.57/52.58 (mean 48.59, spread 12.4%); old 40-unit truths spanned 38.7–56.8
    (47%); $3\times$ 75.22 (1 seed, 40-unit).
  - $\alpha=0.1$: $1\times$ 15.88 (1 seed); $2\times$ 30.34/31.47/30.80 (mean 30.87,
    spread 3.7%); $3\times$ 56.35 (seeds 1–2 paused at 46% — finish before writing §6).
  - $\alpha=0.025$: single seeds 25.23/51.51/82.75.
  - Consequence: report Kolmogorov results as scaling with error bars; per-IC scoring at
    $1\times$ (`sec:rescoring_1x`).

### 5. Results I — 1D KS: zero-shot transfer (~2.5 pages)
- Checkpoint (`tab:checkpoint`): $P=16,c=3,d=48$; $\lambda_1=0.088$; 12/48 positive;
  $D_{KY}=22.14$; D3 bandedness 0.4313 ($p<10^{-4}$). $d=96$ sibling had 13/96 positive
  exponents (capacity not binding).
- **Table: transfer vs independent ground truth** (`tab:e1_result`): $L=50$ 11.37/11.06,
  $L=200$ 45.46/44.61 (1.87%), $L=400$ 89.66/89.13 (0.60%); density 0.2214–0.2230 vs
  truth 0.2242–0.2273.
- **Seed robustness** (`tab:e4_result`): $0.2214\pm0.0001$ at $L=100$, $0.2222\pm0.0007$
  at $L=200$, 3 training seeds.
- **Where it breaks:** $L=25$ — truth weakly chaotic (1.56), model collapses to 0 (4 sites
  < receptive field). A clean, explained lower limit.
- Energy spectrum peak: $L=200$ 0.660/0.660; $L=400$ 0.675/0.691. Decoded rollouts bounded
  ($\max|u|$ 2.6–2.9 vs 3.35–3.47 — note the amplitude under-shoot honestly).
- Architecture note: 1D uses the earlier full-reach $c=3$ design; the 2D canonical design
  (per-patch $c=6$) was not retrained in 1D (decision D1); state it.
- Optional (if computed): normalized Lyapunov spectra $\lambda_i$ vs $i/L$ overlaid for
  $L=100,200,400$ (H1, spectral density). Data partly exist (truth exponents in
  `e1_groundtruth_lyapunov_L*.json`; model spectra need extraction/recompute).

### 6. Results II — 2D: the training distribution decides (~5 pages)
**6.1 Box training fails under transfer; coupling makes it worse.**
- 2D KS box-trained: native 13.5%, $66^2$ 24.4% (3 seeds, tight).
- Kolmogorov box-trained: $\alpha=0.05$ $2\times$ 50.0% (46.8% normalized IC);
  $\alpha=0.1$ $3\times$ 19.4%. D1 window narrowing non-monotone (11.8%/7.3%); D2 exact
  Biot–Savart input 95%; D3 learned regional scale 99%; C9(ii) box-trained collapse.
- Mechanism: propagator reach (6 sites) covers 75% of the 8-site ring → learns wraparound;
  coupling paths only ever see whole-ring inputs.
**6.2 Window training fixes it — same architecture.**
- 2D KS (P7, original arch): native 13.5%→1.2%, $66^2$ 24.4%→5.7%; crops cut in both
  directions (no sample contains a whole periodic direction).
- Kolmogorov (P1/P2): $\alpha=0.1$ 8.6/20.8/1.3% (the 20.8% at $2\times$ now inside the
  rescored truth band — recheck against 30.87 mean); $\alpha=0.05$ 7.3/10.4(vs 49.8)/6.3%.
- **Figure 5** (bar chart, box vs window, all systems/sizes).
**6.3 The canonical recipe** (Phase 0–2 of §15): per-patch AE ($r_e=r_d=0$, $c=6$), long
free rollouts.
- Phase 0 gate (`sec:phase0_results`): per-patch $c=6$ vs P2 three-seed means
  8.7/8.3/2.9% vs 7.3/9.7/6.3% (old truths); 2nd training seed 3.0% at $3\times$;
  control (full reach, $c=6$) 15.3/6.8/4.5% → reach removal helps at equal $c$.
- 2D KS canonical (`p7b_ks2d`, 2 IC seeds): $44^2$ 83.19/84.03 (1.7/2.8%), $66^2$
  187.97/187.74 (2.4/2.5%), $88^2$ 331.98/334.57 — density 0.043 at all three sizes,
  inside the pre-registered 327–345 bracket (the original P7's 215.9 at $88^2$ was a
  training artefact). **Headline 2D number.**
- Kolmogorov canonical family across drag (choose one model per $\alpha$ for the main
  text; full table in appendix):
  - $\alpha=0.05$: `c19_e0d0_c6` (rescored vs 48.59: 4.8% at $2\times$; 2.9% at $3\times$)
    or `k_xs005_noreg` (3× 5.1%). Recommend `c19_e0d0_c6` (canonical AE, 2 training seeds).
  - $\alpha=0.1$: `k_xs01_v5` (3× 2.3%, spread 1.2) or `k_xs01_v7` (1× 2.7%) — present
    both in the B1 ablation; main text uses whichever the decision in §8.4 picks.
  - $\alpha=0.025$: `k_xs0025` (2× 8.1% sd 0.8; 1× 24.8% with large IC scatter) —
    report as the hard regime.
**6.4 The scaling picture (central figure).** $D_{KY}$ vs area (or $L_x$) for all three
systems and all drags, truth with seed error bars, model points with IC spread, dashed
extensive lines; lower panel $D_{KY}$/area. Message: models are extensive by construction
(Remark in §3.2) and match the truth's density within its uncertainty; deviations track
the truth's own finite-size non-extensivity (G2 tracking, P1/P2 signs).
**6.5 The chaos lives where a local model can see it.** C8: unstable-subspace scale
content truth vs model (variance beyond box 9.1% vs 10.0%; 34 vs 36 positive exponents);
C16: 31.5 of 36 unstable dimensions at $|k|\in[2,8)$ around forcing $k=4$. Explains why a
few-site reach suffices.

### 7. Results III — data assimilation (~5 pages)
**7.1 1D: localization becomes necessary as the domain grows.**
- $L=200$, $N_{ens}=64$: none RMSE 1.652 / skill 0.69 (DA hurts) vs GC 0.812 / 1.34
  (`tab:da_transfer`).
- Sweep (`tab:e2_result`, `tab:e2_result_L400`): GC/none ratio >1 at every
  $N_{ens}\in[8,256]$ at $L=200$ and $400$; at $L=400$ unlocalized skill never exceeds 1.
- PFF vs LETKF (`tab:e7_result`, `_L400`): agree within ~0.15 at all 24 points; same
  ratio shape. Result not an artefact of the analysis scheme.
- SEC cannot be carried to the new size: indefinite covariance at 5/6 ensemble sizes
  ($L=200$), 6/6 ($L=400$); at the one runnable point GC wins 0.814 vs 0.903.
- **Negative result at fixed size** (`tab:sec_vs_gc`): at $L=100$ SEC on the global latent
  beats GC on the local field at all 6 sizes (5–25%). Scope the claim accordingly.
**7.2 2D: per-patch decoder → exact, scalable local filter.**
- Block-diagonal decoder Jacobian (unit test: exactly zero off-block).
- Colouring timing (T1): $44^2/66^2/88^2$, `jacrev` 0.150/0.477/0.994 s vs coloured
  0.023/0.044/0.065 s (6.4/10.8/15.3×), error $\le4\times10^{-16}$; speed-up grows because
  each reverse pass grows with the decoder while colouring uses a fixed $c=6$ passes.
- Dense vs per-site LETKF (T2/T3, $N_{ens}=64$, 20 cycles): $44^2$ RMSE 0.591/0.623,
  skill 1.99/1.88, CPU 533/516 s; $66^2$ 0.655/0.676, 1.82/1.76, CPU 1108/1061 s;
  $88^2$ dense killed by the OS twice (RSS 17–29 GB, swap to 24.6 GB), local <1 GB,
  2061 s, RMSE 0.704, skill 1.67, calibration 0.84.
- Transfer costs the filter little: skill 1.88→1.76→1.67 from native to $88^2$ at fixed
  settings (H5 within the 25% gate).
**7.3 Steady state and tuning.** 100-cycle run (T5): first ~15 cycles are a transient;
skill plateaus 3.0–3.6×; calibration overshoots to 1.3–1.4 (overdispersion); inflation
neutral (T4: 1.987/1.985/1.983). Model-noise sweep (E1): 0.08→0.04→0.02 trades calibration
(1.34→1.01→0.83) against skill (2.91→2.44→2.17). Ensemble sweep, native (E4): $N_{ens}$
16 collapses (calib 0.15), 32 under-dispersive, 64 best calibrated (0.80), 128 overshoots.
Frame: an untuned first pass in a sparse (1 obs per 4 sites), high-dimensional
($D_{KY}=82$–330) regime; $2$–$4\times$ is typical for such problems — avoid importing the
1D 10× target.

### 8. What does not work, and why (~3.5 pages)
- 8.1 **Softmax attention propagators collapse** to a fixed point regardless of window;
  unconstrained linear mixing does not (1D search). 2D: windowed-attention site mixer
  (Sec. 234) trains to 3× lower loss but transfers 6× worse (28.8% vs 4.5%) — flexibility
  vs inductive bias.
- 8.2 **Coupling while box-training** (D2, D3, C4, C9(ii)) — worse transfer; learned
  consensus (C7) and exact velocity input (C9(i)) with window training: no gain (9.2%,
  11.4% vs 6.3% at $3\times$).
- 8.3 **Short-horizon training gives spurious regimes** (P9: $D_{KY}$ 65–107 from IC alone
  at one size, energy-spectrum drift; true flow not multistable). Long free rollouts (P9b)
  cut $3\times$ seed spread 22%→2% but introduced a consistent undershoot. Free-rollout
  statistics catch this before Lyapunov evaluation.
- 8.4 **MSE damping: a two-knob interaction** (B1 ablation, `sec:k_xs01_ablation`).
  Table v1→v7: $k_{\max}$ 32→16 fixes 3× (15.4→7.6%) but destabilizes 2×; $w_{\text{var}}=0$
  → scale degeneracy; $w_{\text{rank}}$ inert (bit-identical training); $\epsilon$ in the
  log-det barrier: at $k_{\max}=16$, $\epsilon=10^{-3}\to0$ moves 3× 2.3%→9.8% and 1× 8.9%→2.7%.
  No configuration wins at every size. Tie to Lemma 3 and the $\epsilon$ remark.
  Spectral-band regularizer: A/B shows it hurts at $3\times$ (9.9% vs 5.1%). Training
  diversity (4→8 trajectories): no improvement at $3\times$.
- 8.5 **The autoencoder needs capacity, not context** (C19): reach 0 at $c=3$ costs 9%
  dimension per site at every size (transfer unaffected); $c=6$ restores it.

### 9. Discussion and limitations (~2 pages)
- What transfer requires: (i) extensive physics at the deployment scale (checkable from
  ground truth at two sizes — practical design rule); (ii) a training distribution that
  contains large-domain local statistics; (iii) a propagator that remains chaotic under
  free rollout.
- Transfer error as a physics probe (one paragraph; paper 2): local models are extensive
  by construction, so their transfer error measures the truth's non-extensivity.
- Limitations, stated plainly: Kolmogorov truth uncertainty and single-seed truths at
  $1\times$/$3\times$; Kolmogorov transfer is channel-widening only; one training seed for
  most 2D models; 1D and 2D canonical architectures differ; MPS non-determinism (~5–6%
  run-to-run in DA numbers); DA with synthetic regular observations and additive Gaussian
  model noise; no direct $88^2$ 2D-KS truth; no physical-space DA baseline; AROMA's KS
  failure case and spectrum-reconstruction bar — report our reconstruction MSE and spectra.
- Outlook: RBC (extensive spiral-defect chaos, Karimi & Paul), limited-area NWP, CM1;
  learned stencil closure / RG flow (H3) as interpretability.

### 10. Conclusions (~0.5 page) — write last

### Appendices
- **A. Architecture and hyper-parameters** — full table per system (patch size, $c$,
  hidden width, reaches, window, layers, training steps, LR schedule, curriculum, noise
  anneal, regularizer weights, crop sizes, halo, dataset sizes, $\Delta t$).
- **B. Proofs** — Prop. 1, Prop. 2 (induction), Lemma 3, Prop. 4 + colouring corollary,
  local-LETKF exactness.
- **C. Ground-truth protocol** — Benettin settings, warm-up bug and fix (sign flip for
  Kolmogorov without warm-up), running-$D_{KY}$ convergence plots, seed tables.
- **D. Full evaluation tables** — every model × size × IC seed (generate from
  `artifacts/experiments/crop/*.json`), rescoring history (A1/A2/A7).
- **E. DA details** — PFF/LETKF settings, observation networks, taper construction,
  SEC implementation, per-cycle curves, failures (E6/E1c memory blow-ups, unexplained).
- **F. Reproducibility** — code layout, seeds, device policy (solver/Lyapunov/linear
  algebra float64 CPU; training MPS), tests (equivariance, forced-halo exactness,
  block-diagonal Jacobian, Kalman validation), compute budget, data/code availability.

---

## 3. Figures (main text)

Each figure gets a script `scripts/paper_figures/figNN_*.py` reading `artifacts/`. Status:
**[data]** = numbers exist; **[extract]** = data exist but need pulling/re-running a cheap
diagnostic; **[new]** = needs a new run.

1. **Concept schematic** (TikZ): (a) global vector vs local field; (b) transfer = more
   sites, same weights; (c) crop with forced halo and free interior; (d) block-diagonal
   decoder → local observation operator. [none needed]
2. **Systems gallery**: 1D KS space–time; 2D KS snapshot; Kolmogorov vorticity on the
   $16\times2\pi$ wide domain with a crop box overlaid. [extract from datasets]
3. **1D transfer**: (a) $D_{KY}$ vs $L$, truth vs model, $L=25$–400, with $0.226L$ line;
   (b) $D_{KY}/L$ with 3 training seeds; (c) optional Lyapunov spectra vs $i/L$. [data;
   (c) extract]
4. **1D rollout at $L=400$**: truth vs transferred-model space–time (decoded), and time-
   averaged energy spectra at $L=100,200,400$. [extract]
5. **Box vs window**: grouped bars of transfer error, all systems/sizes, box / box+coupling
   (D2, D3) / window / canonical. [data]
6. **Scaling (centerpiece)**: $D_{KY}$ vs area for 2D KS and Kolmogorov at 3 drags, truth
   error bars, model IC spread; lower panel density. [data; finish A8 3× truths]
7. **2D rollouts at transfer size**: 2D KS $88^2$ and Kolmogorov $3\times$ — truth vs
   model snapshots and energy spectra. [extract]
8. **1D DA**: skill vs $N_{ens}$ at $L=200$ and 400, none vs GC, PFF and LETKF; SEC
   failure markers; inset $L=100$ SEC vs GC. [data: `e2_*.csv`, `e3_*.csv`, `e7_*.csv`]
9. **2D DA**: (a) decoder Jacobian sparsity image; (b) Jacobian timing vs size; (c) CPU
   time & peak memory, dense vs local, with OOM marker at 88²; (d) per-cycle skill and
   calibration (100 cycles); (e) $N_{ens}$ and noise sweeps. [data: `artifacts/da_pff_2d_*`;
   memory trace from T3 log]
10. **Failure modes**: (a) B1 ablation as a size × variant error heat-map; (b) free-rollout
    variance vs lead time for short/long $k_{\max}$ (illustrates Lemma 3) [extract];
    (c) P9 energy-share drift vs truth. [data/extract]

Appendix figures: truth running-$D_{KY}$ convergence; per-IC $1\times$ scatter (models
track IC dependence of the truth); C19 reach/channel ablation; animation stills.

**Main tables.** T1 systems & truths (with seeds/window); T2 1D transfer; T3 2D transfer
(canonical models); T4 DA summary (1D and 2D); T5 B1 ablation; T6 prior-work feature
matrix.

---

## 4. Order to write the sections

Rationale: write what the results force first, then what frames them; the framing
(intro/abstract/conclusion) depends on decisions made while writing results.

1. **§3 Method** (+ Appendix B proofs). Fixes notation, propositions, and the vocabulary
   every later section uses. Decide the canonical-model terminology here.
2. **§4 Systems and ground truth** (+ Appendix C). Needs the A8 truth runs; writing it
   forces the "how precise is the truth" position that all results sections cite.
3. **Figure scripts 3, 5, 6, 8, 9** (data already exist) — make figures before prose for
   §5–7 so text describes what is plotted.
4. **§5 1D results** — cleanest, numbers final.
5. **§6 2D results** — needs the Kolmogorov model-per-α decision (§8.4 ablation feeds it).
6. **§7 DA** (+ Appendix E).
7. **§8 Failure modes** — mostly assembling; write after §6 so cross-references settle.
8. **§2 Related work** (+ verification pass on every citation, see §5 below).
9. **§9 Discussion/limitations.**
10. **§1 Introduction.**
11. **Appendices A, D, F** (tables generated by script).
12. **§10 Conclusions**, then **Abstract**, then title.

---

## 5. Literature review plan

Status flags: **[✓]** cited in the tex record with details checked there; **[k]** standard
reference, details from memory — verify DOI/pages; **[v]** needs verification of authors,
venue, or content before citing. Every 2025–2026 arXiv item is **[v]**.

### 5.1 Extensive chaos, KS, Lyapunov analysis
- Kuramoto & Tsuzuki 1976 (Prog. Theor. Phys.); Sivashinsky 1977 (Acta Astronaut.) [k]
- Manneville 1985, extensive chaos in KS (Lecture Notes in Physics 230) [k]
- Ruelle 1982, large-volume limit of Lyapunov spectra (CMP 87) [✓]
- Egolf & Greenside 1994, dimension and correlation length (Nature 369) [✓]
- Edson, Bunder, Mattner & Roberts 2019, Lyapunov exponents of KS (ANZIAM J.) [✓]
- Koopman analysis reporting $D_{KY}\approx23.2$ at $L=100$, arXiv:1909.00076 [v: authors]
- Cvitanović, Davidchack & Siminos 2010, KS state-space geometry at $L=22$ (SIADS) [k]
- Yang, Takeuchi, Ginelli, Chaté & Radons 2009, effective dimension of extended systems
  (PRL 102:074102) [k] — directly relevant to "physical vs. inactive modes"
- Benettin et al. 1980 (Meccanica); Kaplan & Yorke 1979 [k]; Ginelli et al. 2007 CLVs [k]
- Deissler & Kaneko 1987; Pikovsky & Politi 1998 (comoving Lyapunov / light cone) [k]
- Isola, Politi, Ruffo & Torcini 1990, Lyapunov spectra of coupled map lattices [v]
- Wittenberg & Holmes 1999, scale and space localization in KS (Chaos 9) [✓]; the
  "spatially localized models of extended systems" chapter [v: authors/year]
- Kassam & Trefethen 2005 ETDRK4 [✓]
- Karimi & Paul 2012 (extensive chaos in RBC, PRE 85:046201 — tex says 2010, check) [v];
  Paul et al. 2007 [✓]; Egolf et al. 2000 [✓]

### 5.2 2D turbulence and Kolmogorov flow
- Kraichnan 1967 [✓]; Boffetta & Ecke 2012 [✓]; Danilov & Gurarie 2000 [✓]
- Smith & Yakhot 1993; Chertkov et al. 2007 [✓] (condensates); Mishra et al. 2015 [✓]
- Meshalkin & Sinai 1961 (laminar Kolmogorov stability) [k]
- Platt, Sirovich & Fitzmaurice 1991 [✓]; Chandler & Kerswell 2013 [✓]
- Grappin & Léorat 1991 [✓]; Yamada & Ohkitani 1988 [✓]
- Vinograd, Cullen & Clark Di Leoni 2026, dimensional regimes (arXiv:2602.08960) [✓ read]
- Cleary & Page 2025, attractor dimension vs Re (arXiv:2505.21361) [✓]

### 5.3 Data-driven surrogates of spatiotemporal chaos
- Pathak, Hunt, Girvan, Lu & Ott 2018, parallel local reservoirs (PRL 120) [✓]
- Vlachas et al. 2018 (Proc. R. Soc. A); Vlachas et al. 2020 (Neural Netw.); Vlachas,
  Arampatzis, Uhler & Koumoutsakos 2022, learning effective dynamics (Nat. Mach. Intell.) [k]
- Linot & Graham 2020 (PRE, inertial manifold dimension); Linot & Graham 2022 (Chaos,
  neural ODE ROMs) [k]
- Constante-Amores, Linot & Graham 2024a (arXiv:2410.01238, distributed low-dim models)
  [✓ read]; 2024b (arXiv:2408.03135, minimal pipe flow) [✓]
- Özalp & Magri 2025, stability analysis in latent spaces (Nonlinear Dyn.) [✓]
- Lusch, Kutz & Brunton 2018 (Koopman autoencoders) [k]
- Bar-Sinai, Hoyer, Hickey & Brenner 2019 [✓]; Kochkov et al. 2021 [✓, check transfer claim]
- Li et al. 2021 FNO (ICLR); Kovachki et al. 2023 (JMLR) [k]
- Zero-shot generalization of transformer neural operators to larger domains,
  arXiv:2606.14597 [v: authors]; "False promise of zero-shot super-resolution",
  arXiv:2510.06646 [v]
- Guan et al. 2022 (JCP, transfer to higher Re) [v]; Frezat et al. 2022 [v];
  Brenowitz & Bretherton 2019 [✓]; Wang, Yuval & O'Gorman 2022 [✓]
- AROMA 2024 (arXiv:2406.02176, KS failure case for local neural fields) [v: authors]
- Equivariant dynamics models: Wang, Walters & Yu 2021 (ICLR) [k]; Cohen & Welling 2016 [k]
- Fully convolutional training on crops: Long, Shelhamer & Darrell 2015 [k]
- **Rollout stability, invariant measures, blurring** (needed for §3.5/§8.4):
  Brandstetter, Worrall & Welling 2022 (pushforward trick) [k]; Lippe et al. 2023
  PDE-Refiner [k]; Jiang et al. 2023, training neural operators to preserve invariant
  measures (NeurIPS) [k]; Schiff et al. 2024 DySLIM (ICML) [k]; Platt et al. 2023,
  constraining chaos / invariants in RC training (Chaos) [v]; Chattopadhyay &
  Hassanzadeh 2023 long-term stability [v]; Lam et al. 2023 GraphCast (Science) [k];
  Bonavita 2024, limitations of ML weather models (GRL) [k]; Price et al. 2025 GenCast
  (Nature) [k]
- **Limited-area ML weather** (window training analogue): Oskarsson et al. 2023/2024
  Neural-LAM [v]; Nipen et al. 2024 stretched-grid regional models [v]
- Collapse regularizers: VICReg (Bardes, Ponce & LeCun 2022) [k] for $w_{\text{var}}$ /
  decorrelation lineage

### 5.4 Ensemble DA, localization, latent DA
- Evensen 1994; Burgers, van Leeuwen & Evensen 1998 [k]
- Houtekamer & Mitchell 1998, 2001 [k]; Hamill, Whitaker & Snyder 2001 [k]
- Gaspari & Cohn 1999 (QJRMS) [k]
- Patil, Hunt, Kalnay, Yorke & Ott 2001, local low dimensionality (PRL 86) [k] — key
  precedent for "ensemble size set by local dimension"
- Ott et al. 2004 LEKF (Tellus A); Hunt, Kostelich & Szunyogh 2007 LETKF (Physica D) [k]
- Anderson 2007 (hierarchical filter, Physica D); Anderson 2012 SEC (MWR) [✓];
  Anderson & Lei 2013 (MWR) [k]
- Hu & van Leeuwen 2021, particle flow filter (QJRMS) [k]; Pulido & van Leeuwen 2019 [k];
  Daum & Huang 2008 [k]
- Carrassi, Bocquet, Bertino & Evensen 2018 review (WIREs) [k]; Bocquet et al. 2020 /
  Brajard et al. 2020, DA + ML [k]
- Latent DA: Peyron et al. 2021 (QJRMS) [v]; Amendola et al. 2021 [v]; Mack et al. 2020 [v];
  ROAD-EnKF, Chen et al. 2023 (arXiv:2301.11961) [v: authors]; Learning Enhanced Ensemble
  Filters (arXiv:2504.17836) [v]; LAE-EnKF (arXiv:2603.06752) [v]
- Jacobian colouring: Curtis, Powell & Reid 1974; Coleman & Moré 1983 [k]

### 5.5 Verification pass (do while writing §2)
- Resolve every [v]; fetch abstracts for every 2025–2026 arXiv item and confirm the claim
  we attribute to it.
- Targeted searches still owed (task G1): patch/window-trained PDE surrogates; latent
  localized EnKF; zero-shot domain-size transfer in chaotic surrogates; learned local
  rules validated by Lyapunov spectra. Record search date and terms in Appendix F.

---

## 6. Work to finish before or while writing

Ordered by what blocks which section.

| # | Item | Blocks | Cost | Status |
|---|---|---|---|---|
| W0 | **Re-run 2D DA at a correct deployment.** `run_da_pff_2d_ltransfer.py` takes its native config from the crop checkpoint (NX=640, 40×40 sites) and pairs it with L=44, so each site spans 1.1 units instead of the 5.5 used in training (`exp_crop_eval_ks2d.py` correctly uses NX=128, 8×8 sites at L=44). Fix: set L from the trained Δx (L=220 for 40×40 sites; transfer to 330/440 keeps d=9,600/21,600/38,400). Re-run T2/T3/T5, E1, E4. | §7 | ~1 day of runs | **blocking for §7** |
| W1 | ~~Resume α=0.1 3× truth seeds 1–2~~ — user decision: use the existing single-seed 56.35 | — | — | closed |
| W2 | α=0.1 1× truth seeds 1–2; α=0.05 3× seeds 1–2 at T=120 | §4, Fig 6 | 1× cheap; 3× ~2 h each | A8 to-do |
| W3 | Main-text α=0.1 model: `k_xs01_v5` (user decision) | §6 | — | closed |
| W4 | Figure scripts (Figs 3, 5, 6, 8, 9) — written 2026-10-08, `scripts/paper_figures/fig0{3,5,6,8,9}_*.py` + `_style.py`; PDFs and `.meta.json` provenance in `docs/paper1/figures/`. Fig 9(f) memory panel re-plots automatically once the L=220/330/440 runs finish (re-run the script). | §5–7 | — | done (9f pending Block C) |
| W5 | Extract model Lyapunov spectra at L=100/200/400 for Fig 3c | §5 | <1 h | optional |
| W6 | Free-rollout variance vs lead time, short/long $k_{\max}$ (Fig 10b) | §8 | <1 h | extract |
| W7 | Reconstruction MSE + energy spectra table (AROMA bar) | §9 | <1 h | extract |
| W8 | 2D DA $N_{ens}$ sweep at a transfer size (E5 / H4 proper) | §7 | ~3 h | optional; can scope claim to N=64 |
| W9 | Seam diagnostic (H6) on existing analyses | §7 | <1 h | optional, reviewers may ask |
| W10 | Second IC seed set / third seed for `p7b_ks2d` (C2) | §6 | ~1 h | optional |
| W11 | Literature verification pass | §2 | ~half day | not started |
| W12 | Direct 88² 2D-KS truth (~500 directions) | §6 | many CPU hours | optional; state as limitation |

Minimal path to a complete draft: W0, W2, W4, W11. Everything else strengthens but does not
block.
