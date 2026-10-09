# Critical review of paper 1 (`docs/paper1/`), 2026-10-08

Reviewer: Claude (Fable 5.1), working from the outline, the written sections (§3–§6, App. B–C), the task list, the research record, the code and the result files. Every substantive claim below is tagged **[verified: …]** (I read the file, ran the command, or re-derived the number) or **[inferred]** (my reading, not checked against a primary source). Three verification passes were delegated and their reports are folded in: artifact numbers (≈190 numeric claims in §4–§6 and App. C, all re-read from JSON/CSV/HDF5 metadata), code-vs-text (every mechanism claim in §3 and App. B), and literature (every citation and the closest prior work). Nothing in the repository was edited; no runs were started.

Tags on recommendations: **impact** high/medium/low, **effort** hours/days/compute.

---

## 1. Executive summary

The core of the paper is sound and worth publishing in *Chaos*: a weight-shared local latent field, trained once, reproduces the extensive Kaplan–Yorke dimension of 1D KS to ≤3% over an 8× range of domain sizes and reproduces the scaling of $D_{KY}$ in 2D KS and drag-damped Kolmogorov flow within the (large) truth uncertainty; the per-patch decoder makes latent DA exactly local; and the 2D DA re-run shows a 64-member localized filter keeping its skill (≈4×) while $d_{lat}$ grows from 384 to 21,600 and $D_{KY}$ from ≈83 to ≈4,700, with zero retraining. The numbers in the written sections reproduce from the artifacts to the last digit, with one arithmetic slip. The problems are in what is *claimed around* those numbers. In priority order:

1. **The 1D DA transfer results (C5, §7.1 plan, Fig. 8b–c, record E2/E3/E7) are confounded by observation density, and must be re-run before they can carry the claim "localization becomes necessary as the domain grows."** `scripts/run_da_pff_ltransfer.py:76` scales the observation stride with $L$, so the *number* of observations (32) is held fixed while the physical spacing goes 3.1 → 6.25 → 12.5 units at $L$ = 100/200/400 **[verified: script line 76 and the `obs_stride` column of `e2_*.csv`, `e7_*.csv`: 16 at L=200, 32 at L=400 vs. 8 at L=100]**. At $L=400$ the spacing exceeds the KS pattern wavelength 8.9. The 2D probe already showed that DA is worse than the free run at 11-unit spacing **[verified: `docs/paper1_tasks.md:496–499`]**. The decline of localized skill (2.7 → 1.34 → 1.05) and the unlocalized collapse are therefore not separable from observation starvation. The 2D runs hold density fixed (stride in grid points at fixed $\Delta x$ **[verified: `run_da_pff_2d_ltransfer.py:113–117`]**) and show the clean version of the same claim (unlocalized N=64 skill 2.31 → 0.94 → 0.83 at 44²/66²/88²; local 4.23 → 3.86 at 44²→330²). Fix: re-run the 1D sweeps with `--obs-stride-at-old-L` divided by the scale (minutes per point), and lead §7 with the 2D result. **impact high / effort hours.**

2. **The SEC "cannot be carried to the new size" claim (C5, C6, record E3) is an implementation artifact and is wrong as stated.** `ks_latent/da/sec.py` applies Anderson's correction as an *elementwise map of the whole sample correlation matrix* (`sec_localize_covariance`), which does not preserve positive semi-definiteness; with $N<d$ the result is indefinite, which is exactly the failure pattern in `e3_*.csv` (fails for N ≤ 128 at d=96, runs at N=256) **[verified: sec.py lines 60–75; CSV]**. Anderson (2012) applies SEC to per-observation regression coefficients in a serial filter, where no PSD requirement arises. Moreover the SEC table depends only on $N$ (`build_sec_table(n_ens)`), not on $d$ or $L$, so SEC is *trivially* transferable across domain size; the non-transferable object is the fixed empirical taper (method 2 in the same file). Reframe: "a matrix-form SEC becomes indefinite when N<d; a serial SEC would not; a fixed empirical taper has no notion of size." Cheap fix: project the corrected matrix to the PSD cone (clip eigenvalues) or implement serial SEC, and re-run E3 at fixed density. **impact high / effort hours.**

3. **The 2D DA section should become the DA headline, framed around a fixed-size ensemble at 56× the training area, not around memory.** The corrected steady-state table **[verified: `scripts/summarize_da2d.py` output]**: per-site local filter N=64 skill 4.23/4.08/4.00/3.84/3.86 at L=44/66/88/220/330; dense+GC 3.65/3.54/3.47/3.40; dense unlocalized 2.31/0.94/0.83. Two things the outline still says must go: "local filter runs in <1 GB" (measured peak RSS 1.7/4.5/4.9 GB at 44²/66²/88² (of which 0.5/≈1/1.2 GB is the truth trajectory) and 18.7 GB at L=330 (9.9 GB truth) with the leak fixed; the re-measurement is pending and no `M_*` files exist yet) and "dense filter killed at 88²" (now a memory-accounting statement, as the record itself concedes). Also report honestly that the per-site filter is *slower* than the dense one at d ≤ 1536 (CPU 50 vs 23 s at L=44) and only wins at d=9600 (1450 vs 1914 s), and that at N=16 the per-site filter (1.8) is much worse than dense+GC (3.8). **impact high / effort hours (text) + the pending memory re-measurement.**

4. **§3.3's light-cone argument is physically misdirected for these systems and is contradicted by the project's own measurement.** $v_*$ for 1D KS was measured at 1.26 units/time **[verified: `docs/RESULTS.md` Phase 2]**, so $v_*\Delta t$ = 1.3 units at $\Delta t=1$, against a propagator reach of 6 sites = 37.5 units and a Gaspari–Cohn support radius of 37.5 units. The comoving-Lyapunov velocity bounds the spread of *perturbation growth*, not the domain of dependence of a parabolic one-step map, which is infinite with an exponentially decaying kernel; the relevant length is a correlation/Green's-function length (the learned stencil's $R^2$ saturates at w≈3 sites ≈ 19 units ≈ 2 pattern wavelengths **[verified: `docs/RESULTS.md` Phase 11 table]**). Rewrite §3.3 around the measured interaction range; keep $v_*\Delta t$ as a (weak) lower bound and state that it is not binding. Eq. (loc_bound) as written is decorative. **impact medium / effort hours.**

5. **Lemma 3's mechanism needs a quantitative check that the paper can do from its own numbers, and a second, sharper mechanism is missing.** All training rollouts are *shorter* than one Lyapunov time: $k_{\max}\Delta t\lambda_1$ = 1.1 (1D KS), 0.53 (2D KS), 0.2–0.8 (Kolmogorov) **[inferred from Tables tab:truth_runs and §6.3]**, and the input noise (0.02–0.10) needs ≈ 4 e-folds ≈ 40 time units to reach climatology, so "beyond the predictability horizon" is not reached. The law-of-total-variance argument still applies at every $k>0$ (the variance deficit is $\approx\sigma^2 e^{2\lambda_1 k\Delta t}$ along the leading direction), so the lemma is right but the prose is wrong. The missing mechanism is that *training with input noise is a Jacobian penalty* (Bishop 1995: $\mathbb{E}\|\Phi(z+\xi)-y\|^2 \approx \mathbb{E}\|\Phi(z)-y\|^2 + \sigma^2\mathbb{E}\|J_\Phi\|_F^2$), i.e. the noise anneal directly penalizes finite-time growth. This explains P9b's undershoot (end noise raised to 0.05) more directly than rollout length. **impact medium / effort hours.**

6. **Framing.** The strongest single result is not "zero-shot transfer" in the abstract (equivariant operators generalize to larger domains, and Carretero-González et al. 1999 showed the thermodynamic-limit Lyapunov spectrum of a lattice can be predicted from a small lattice) but the *combination*: a learned local rule, validated by full Lyapunov spectra at four sizes in three systems, whose latent is exactly local for DA. The title should name both halves and the Lyapunov validation; the abstract should lead with 1D KS numbers, then the 2D scaling, then the fixed-ensemble DA result. Recommended title and abstract in §2. **impact high / effort hours.**

7. **Truth-side weaknesses a referee will attack first:** 2D KS truths are single 40-unit runs (≈7 Lyapunov times) while the paper interprets a 4.6% density difference between them as possibly real (§6.4); the α=0.1 3× headline (2.3%) is scored against a single 40-unit truth that App. C says is biased +10% on average; the L=25 "weakly chaotic" truth has $\lambda_1 = 0.0013$ over 1200 units, indistinguishable from zero. Fixes in §4 and §8 (two 2D-KS seeds at 44²/66² and finishing the two paused α=0.1 3× seeds are the best value per CPU-hour). **impact medium / effort compute (hours).**

8. **The paper is internally consistent on numbers (one slip: "94–100%" should be "93–100%", `06_results_2d.tex:44`), but the outline is stale in several places** (22.14 as the L=100 truth; "within 2%" for 1D; 4.8% vs 7.7% for c19_e0d0_c6 at 2×; "<1 GB"; "killed at 88²"; T2/T3 timing numbers from invalid runs). Section 10 lists them.

---

## 2. Assessment of significance

### 2.1 What is new, what is not

**Not available as novelty claims** (the outline already lists some; I confirm and add):
- "KS admits local reduced models" — Wittenberg & Holmes 1999 **[verified by literature pass]**.
- "A learned model of KS reproduces the Lyapunov spectrum" — Pathak et al. 2017 (*Chaos* 27, 121102) computed the KS Lyapunov spectrum from a reservoir; Vlachas et al. 2020 (*Neural Netw.* 126) compared Lyapunov spectra of RC/BPTT models of KS at L=200; Özalp, Margazoglou & Magri 2023 (*Chaos* 33) and Özalp & Magri 2025 did it in a learned latent space, as did Racca, Doan & Magri 2023 (*JFM* 975) **[Pathak, Vlachas, Özalp verified by the literature pass; Racca from the project's literature file; see §6]**.
- "Parallel/patch-local predictors of large chaotic domains" — Pathak et al. 2018; Constante-Amores, Linot & Graham 2024. Note the real difference: Pathak's reservoirs are *not* weight-shared (each has its own random reservoir and trained readout), so they cannot be transferred by adding sites; Constante-Amores et al. train per domain. Say this explicitly; it is your cleanest differentiator.
- "Translation-equivariant surrogates deployed zero-shot on larger domains" — arXiv:2606.14597 and, earlier, the convolutional-closure literature (Bar-Sinai et al. 2019 ran a 2π-trained model on a 10× domain; Kochkov et al. 2021 report equal performance on a 2× larger Kolmogorov domain; Stachenfeld et al. 2022 likewise for KS, qualitatively) **[verified; see §6]**.
- "Ensemble size is set by local dimension" — Patil et al. 2001; Ott et al. 2004; Hunt et al. 2007; and the quantitative form, $N_{ens} > n_0+1$ unstable–neutral directions for an unlocalized EnKF, Bocquet & Carrassi 2017 / Bocquet et al. 2017 **[verified by literature pass]**.
- "Extensive chaos has a thermodynamic-limit Lyapunov density; small lattices predict it" — Livi, Politi & Ruffo 1986; Isola et al. 1990; Carretero-González et al. 1999 (*PRL* 83, 3633: "Thermodynamic limit from small lattices of coupled maps") **[verified by literature pass]**. The last is the direct physics ancestor of your transfer result and is not cited anywhere in the repository.

**Genuinely new, in combination** (and each should be stated at exactly this width):
1. A *learned* local latent rule whose zero-shot extensions are validated against full Benettin spectra of the true PDE at 4 domain sizes (1D) and 2–3 sizes (2D), in three systems, with the extensive density reproduced to 1.5% (1D) and within truth uncertainty (2D). Nobody has done the Lyapunov-level, multi-size validation of an assembled learned model.
2. The box-vs-window result: at equal architecture, training on a small periodic box fails under transfer in two 2D systems (24–38%, with coupling 93–100%) and window training with a forced halo fixes it (1–8%). This is a practical, reproducible finding for the surrogate community and is new in this form.
3. The per-patch decoder ⇒ exactly block-diagonal observation Jacobian ⇒ exact per-site local analysis with $c$ JVPs at any size. Simple, correct, and stronger than what ROAD-EnKF/LAE-EnKF offer (both list localization as future work).
4. The empirical DA result that follows: a 64-member localized filter in a learned latent keeps skill ≈4× from $D_{KY}\approx 83$ to $\approx 4{,}700$ (56× area) with no retraining, while the unlocalized filter at the same N dies by 2.25× area. This is the first latent-DA demonstration at that scale ratio that I know of.
5. The honest ablation that MSE rollout length and the log-det offset $\epsilon$ interact and that no configuration wins at all sizes (B1 v1–v7). Useful; keep short.

### 2.2 Who cares, and is *Chaos* right

- Nonlinear-dynamics readers (Chaos, Physica D) will value items 1 and 5 and the Lyapunov validation; they will be indifferent to the DA cost story and skeptical of Kolmogorov truths with 12% seed spread.
- DA/geophysics readers (MWR, QJRMS, NPG, JAMES) will value items 3–4 and will immediately ask about observation density, inflation tuning, deterministic vs perturbed-observation updates, and a physical-space baseline.
- Scientific-ML readers will value item 2 and the "training distribution decides" lesson.

*Chaos* is a reasonable home (Pathak 2017 and Hunt-lineage DA papers have appeared there), **provided** §7 is written so a dynamics referee can follow it without DA background (define skill, spread, localization in two sentences each) and the DA claims are scoped to "in a learned latent space". If the authors want the DA community to read it, Physica D or NPG would be equally defensible; I would not move to a DA journal, because the Lyapunov-level validation is the unusual strength and the DA skill numbers (≈4×) are modest by DA standards.

### 2.3 Recommended framing

**Title (recommended):** *Local latent fields for extensive spatiotemporal chaos: zero-shot domain transfer validated by Lyapunov spectra, and exactly local ensemble data assimilation*

(Alternatives: *A learned local rule for extensive chaos transfers across domain size and keeps latent data assimilation local*; the outline's third alternative "Train on a window, deploy at any size" undersells the Lyapunov validation.)

**Abstract sketch (~200 words):**
> Latent surrogates of spatiotemporally chaotic PDEs are usually a single global vector, whose dimension is fixed at training time. In extensive chaos the number of active degrees of freedom grows with the domain, so such a surrogate cannot even be evaluated at a new size. We make the latent a field on a lattice of sites, built only from weight-shared operators of finite reach, so that one trained model is instantiated at any domain size by adding sites. We test whether such models reproduce extensive chaos by computing full Lyapunov spectra of the learned maps and of the PDEs. Trained at L=100, a 1D Kuramoto–Sivashinsky model reproduces the true Kaplan–Yorke dimension to within 3% from L=50 to L=400 (0.6% at 4×), with three training seeds agreeing to 0.1%. In 2D KS and in Kolmogorov flow with linear drag, models trained on windows of one large simulation with forced lateral boundaries reproduce the growth of D_KY with domain size within the uncertainty of the reference spectra, whereas the same architecture trained on a small periodic box fails by 24–38%. A per-patch autoencoder makes the latent observation operator exactly local, so a per-site localized ensemble filter is exact and its cost per site is independent of domain size; a 64-member filter keeps its skill while the latent grows 56-fold. We report what did not work: unlocalized filters degrade with size, matrix-form sampling-error correction becomes indefinite, and squared-error rollouts damp chaos.

**Contributions list (replace outline C1–C6):**
- C1 unchanged (architecture, Prop. 1).
- C2: 1D ≤3% over 8× with 3 seeds; 2D KS extensive at 2× per side; Kolmogorov scaling within truth uncertainty at three drags. Drop "within 2%" (L=50 is 2.7%).
- C3 unchanged in substance; state the wraparound mechanism as a hypothesis.
- C4: per-patch ⇒ exact locality ⇒ per-site filter; *promote* the 56× fixed-ensemble result into C4; drop "<1 GB" and "killed by the OS".
- C5: restate after the fixed-density 1D re-run; in the meantime use the 2D unlocalized degradation (2.31 → 0.94 → 0.83) as the evidence.
- C6: keep attention collapse, MSE damping × ε, spectral regularizer; **remove** "SEC cannot be carried to the new size"; replace with "matrix-form SEC becomes indefinite when N < d".

---

## 3. Framing and structure

### 3.1 The story the data support

The outline leads with "zero-shot transfer of extensive chaos" and co-headlines DA. I agree with the two-headline structure but would change what each headline says:

- **Headline A (dynamics):** *A local rule learned from data reproduces the extensive Lyapunov dimension at sizes it never saw, provided it is learned from windows that contain large-domain statistics.* The 1D result carries the precision; the 2D results carry the "training distribution decides" lesson; the Kolmogorov results carry the "scaling within truth uncertainty" caveat. The Remark on extensivity (§3.2) is the organizing idea and should be stated once, carefully (see §4.3 below), then used as the lens for every transfer result.
- **Headline B (DA):** *Because the latent is a field and the decoder is per-patch, the latent observation operator is exactly local, so a fixed-size localized ensemble filter works at any domain size.* Evidence: the 2D sweep (fixed-density observations, 100 cycles, steady state): local N=64 skill 4.2→3.9 from 44² to 330² while D_KY grows 56×; unlocalized N=64 2.3→0.9→0.8 by 88². The 1D results then become a *second analysis scheme and second system* confirmation (PFF and EnKF agree), once re-run at fixed observation density.

What should *not* be a headline: memory/OOM, the colouring speed-up (a nice engineering corollary, one paragraph), and "SEC cannot transfer".

### 3.2 Recommended section order (changes from the outline in bold)

1. Introduction — as outlined, with the one-sentence positioning statements: Carretero-González et al. 1999 (thermodynamic limit from small lattices) is the physics ancestor; Pathak 2018 / Constante-Amores 2024 are the surrogate ancestors (not weight-shared / not transferred); Patil 2001 / Bocquet–Carrassi 2017 are the DA ancestors.
2. Background — tight. **Add two paragraphs:** (i) thermodynamic limit of Lyapunov spectra in lattices (Livi–Politi–Ruffo 1986; Isola 1990; Carretero-González 1999; Takeuchi–Ginelli–Chaté 2009 for collective modes the local model will miss); (ii) ensemble size vs unstable subspace (Bocquet & Carrassi 2017; Carrassi et al. 2022 review).
3. Method — as written, **with §3.3 rewritten** (interaction range, not light cone), **Lemma 3 reworded** (holds at every k; add the noise-as-Jacobian-penalty statement), and **the per-site filter named correctly** (a local EnKF with perturbed observations and R-localization, not an LETKF).
4. Systems and ground truth — as written; **add the 2D-KS truth uncertainty honestly** (single 40-unit runs; no claim about a 4.6% density change).
5. 1D transfer — as written; **soften L=25** (λ₁=0.0013 is not resolvable from 0 in a 1200-unit window).
6. 2D transfer — as written; **§6.4 must not read the 4.6% rise as real**; **§6.5 is Kolmogorov-only, say so.**
7. DA — **lead with 2D** (exact locality → per-site filter → fixed N across 56× → unlocalized degrades → cost/memory honestly), then 1D at fixed observation density (PFF vs EnKF agree; GC vs none), then the fixed-size negative result (SEC on the global latent beats GC on the local field at L=100), then the SEC-indefiniteness observation, correctly attributed.
8. What does not work — **shorten to ~2 pages**: attention collapse (one paragraph), MSE damping × ε (table + Lemma 3 reference), spectral regularizer and diversity (one paragraph each), P9 regimes (one paragraph with the rollout-statistics diagnostic as the take-home).
9. Discussion — add "what transfer requires" (as outlined) plus **the interaction-range design rule** (encoder reach must cover the unstable band; P6 result) and the limitations list, extended by: observation density, perturbed-observation filter, no inflation tuning, no physical-space baseline, 2D-KS truth uncertainty.
10. Conclusions.

Appendices: A (architecture table — must include the 1D model's actual reach, see §4.2), B (proofs; see §4.3 for scoping), C (as written, plus 2D-KS caveat), D (full tables), E (DA details incl. observation networks in physical units at every size), F (reproducibility).

---

## 4. Correctness issues

### 4.1 Numbers (artifact pass)

All ≈190 numeric claims in §4–§6 and App. C were re-read from the artifacts **[verified: delegated pass; scratch scripts in the scratchpad `verify_numbers/`]**. Findings:

| item | file:line | status | fix |
|---|---|---|---|
| "errors of 94–100%" for coupling models | `06_results_2d.tex:44` | **mismatch**: largest survivor is `icnorm/c9ii_x3_seed0.json` 4.93 vs 75.22 = 93.4% | write "93–100%" |
| "windows of 1200–2000 time units span more than 100 Lyapunov times" | `04_systems.tex:40,134` | true for L ≥ 100 (180, 119, 123); exactly 100 at L=50; **1.6 at L=25** (λ₁=0.0013) | "for L ≥ 50, 100 or more; at L=25 the exponent is too small to resolve" |
| "Lyapunov times are 8–15 time units … 3–5 of them" | `04_systems.tex:139–140` | from the truth λ₁ the range is 6.2–16.4 and 2.4–6.4 | widen |
| mean-of-errors vs error-of-mean | `tab:transfer2d` | paper uses mean of per-IC errors consistently; `paper1_outline.md:352` quotes error-of-mean (4.8% at 2× for c19_e0d0_c6 vs 7.7% in the paper) | state the convention in the table caption; fix the outline |
| 1D model numbers (λ₁=0.088, 12/48, 13/96, 0.00/11.06/22.14, seed-1/2 L=200 values, spectrum peaks, max|u|) | `05_results_1d.tex` | **record-only**: no JSON artifact; `fig03_1d_transfer.py` hard-codes them | regenerate from checkpoints before submission (W5 covers the spectra) — or at least store a JSON with provenance |
| old 40-unit α=0.05 2× truth 56.84 | App. C | record-only (survives only as `true_D_KY` in crop JSONs) | fine, but say so in App. D |
| ε=10⁻³ for `k_xs01_v5` | `06_results_2d.tex:103` | not recorded in the log/meta; consistent with the script default | record it in App. A from the code default, with the commit SHA |
| kmax=32 α=0.05 comparison "6.2/5.2/7.2%" | `06_results_2d.tex:105` | matches `k_xs005_noreg`; model not named | name it |

Everything else — Tables tab:systems, tab:truth, tab:transfer1d, tab:transfer2d, tab:running_dky, tab:truth_runs, the per-patch/full-reach/decoder-only ablation numbers, C8/C16 numbers — reproduces to the printed precision. The crop JSONs carry stale `true_D_KY`/`rel_err_pct` fields (23.26 / 56.84 / 30.34); the paper correctly ignores them, and App. D should say so to pre-empt a reader who opens the files.

### 4.2 Claims that are wrong, overstated, or unsupported

**(a) C5 and §7.1 (1D): the observation density falls with L.** Described in the executive summary. The effect size is large enough to explain the whole absolute decline: GC skill 2.7 (L=100, spacing 3.1) → 1.34 (L=200, spacing 6.25) → 1.05 (L=400, spacing 12.5) **[verified: `da_sweep_phaseD3_local224.csv`, `e2_*.csv`]**, while in 2D at fixed spacing the local filter's skill is flat from 44² to 330². The *ratio* GC/none at fixed L is still informative, and the sentence "unlocalized DA is worse than the free run at L=200–400 for every N ≤ 256" may survive, but it must be re-measured. Note also that the record's `tab:da_transfer` is labelled n_prop_steps=3 while its numbers (0.812/1.34) are identical to the n_prop_steps=5 E2 row **[verified: record lines 327–336 vs 549]**; one of the two labels is wrong.

**(b) C5/C6: "SEC cannot be carried to the new size."** Wrong for the reasons in the executive summary. The honest and still-useful statement: "SEC implemented as a correction of the full sample correlation matrix is not guaranteed positive semi-definite; for N < d it was indefinite at 5/6 (L=200) and 6/6 (L=400) ensemble sizes, so a serial (regression-space) implementation is required to compare it fairly at transfer sizes; the fixed empirical taper (archive correlation) is the method that genuinely has no notion of domain size." Also note that at L=100 the SEC comparison was *global latent + SEC* vs *local latent + GC* — two different models — so the "negative result at fixed size" conflates architecture with localizer; the record says this (`ltransfer.tex:664–667`), the paper should too.

**(c) §7.2/C4: "<1 GB" and "killed by the OS."** Both superseded by the leak fixes; measured local-filter peak RSS is 1.7/4.5/4.9 GB at 44²/66²/88² and 18.7 GB at 330² (of which 9.9 GB is the truth trajectory) **[verified: da2d JSON `peak_rss_after_truth_GB`, `peak_rss_GB`]**. The filter-only increment (≈1.2, 3.7, 8.9 GB at d = 384, 1536, 21,600) is sub-linear in d but far above the O(N·d) one would expect, so the "memory ∝ area" claim is not yet demonstrated. Report the dense filter's cost as accounting (3 × 11.8 GB at d=38,400) and the local filter's as measured, after the pending re-measurement.

**(d) §3.3 light cone.** See executive summary item 4. The measured $v_*$ (1.26) makes $v_*\Delta t_{\rm win}$ = 6.3 units for the 5-step 1D window and 0.63 units (!) for the 2D KS window (Δt=0.1, 5 steps, assuming a similar $v_*$) against GC supports of 37.5 and 33 units. Eq. (loc_bound) is never within a factor of 5 of binding. Either drop it or present it as "a lower bound that is not the operative constraint here; the operative constraint is the interaction range measured by the stencil fit / D6".

**(e) §6.4: "the model cannot follow the truth's 4.6% rise between 44² and 66²."** The two 2D-KS truths are single 40-unit runs; the Kolmogorov 40-unit runs deviated from their 120-unit continuations by −17% to +23% **[verified: App. C]**. A 4.6% difference between two such runs is noise. Say "indistinguishable from extensive within the truth uncertainty."

**(f) §5 "where transfer stops": L=25 truth is "weakly chaotic, λ₁ = 0.001".** A finite-time exponent of 0.0013 from a 1200-unit window is not distinguishable from zero (the natural error scale of a finite-time LE over 1200 units, ≈ 1/√(T/τ_corr), is of order 0.01). KS near L≈25 is close to a non-chaotic window in L. The contrast "truth chaotic, model not" is therefore not established; the model's failure to be chaotic may be correct. Reword as "at L=25 the true dynamics are at most marginally chaotic and the transferred model is not chaotic; the lattice (4 sites) is narrower than the kernel (7), so this size is outside the model's domain of validity by construction."

**(g) §6.3 "costs nothing if the channel count is raised."** The per-patch c=6 AE's reconstruction is worse (interior val 0.0118 vs 0.0089 for P7, ≈30% **[verified: record 5622]**), and the record's own C19 table shows that the three-seed means are *equal* at 1×/2× and better at 3×, not uniformly better. The paper's wording ("at least as accurate") is fine; "costs nothing" is not. Also the AROMA reconstruction-spectrum bar (W7) is still owed.

**(h) §6.5 "no evidence that remaining errors come from missing long-range coupling."** C8/C16 are Kolmogorov α=0.05 3× only (P2 model, not the canonical one) **[verified: `c8/c8_result.json` uses p2_a005]**. State the model and drag; do not generalize to 2D KS or α=0.025.

**(i) Outline C2 "within 2% at L=50,200,400".** L=50 is 2.7%. The paper says "below 3%"; the outline/abstract must match.

**(j) §3.6 "the number of tangent directions always exceeds k+1."** Check whether this is asserted in code or chosen by hand (code pass, §4.4). At 88² the model evaluation used 512 of 768 directions for D_KY ≈ 333 (k ≈ 170 positive) — fine, but say "chosen so that" rather than "always".

### 4.3 Propositions and lemma

- **Prop. 1 (equivariance, size independence).** Correct and worth keeping, but it is a two-line observation; keep the proof in App. B as is. Two scoping notes: (i) equivariance holds for shifts by whole patches only (say so in the statement; the proof uses it); (ii) the proviso P ≥ 2r+1 per axis should be in the statement, not only in the proof, because it is what kills L=25. Also state explicitly that LayerNorm over channels is per-site (it is) and that no global pooling/normalization exists anywhere, because any batch- or domain-level statistic would silently break both claims.
- **Remark (local models are extensive).** As written it over-claims in both directions. "For a translation-invariant lattice map of finite range, the distribution of Lyapunov exponents per site converges as P→∞" is *not* a theorem for general nonlinear lattice maps; it is proved for weakly coupled CMLs (Bunimovich–Sinai 1988 is about SRB measures, not the Lyapunov density) and observed numerically (Livi–Politi–Ruffo 1986; Isola et al. 1990). Cite those and say "expected and numerically established". Second, "it cannot represent finite-size departures" is false as stated: a local model has its own o(|Ω|) corrections (your L=25 collapse is one), set by its reach; what is true is that its corrections are *tied to the reach* and need not match the truth's. Suggested wording in §9.
- **Prop. 2 (forced-halo exactness).** Correct; the proof is essentially the definition of reach. Keep, but call it a Lemma and keep the useful part: the remark that the mismatch is confined to a strip of width R_prop. A referee may ask for a quantitative statement; the cheap one is in the record (Variant A at short horizon measures the mismatch directly) — report that number if it exists, or say it was not measured.
- **Lemma 3 (squared error favours contraction).** Correct. Fix the surrounding text (executive summary item 5): the deficit exists at all k, grows like σ²e^{2λ₁kΔt} until saturation, and none of the training horizons exceed one Lyapunov time, so "beyond the predictability horizon" is not the regime. Add the Bishop (1995) noise-injection equivalence as part (b): with input noise ξ, the loss expands to the clean loss plus σ²‖J_Φ‖²_F, a direct penalty on finite-time growth. This gives a *testable* prediction the data already bear on: P9b raised terminal noise to 0.05 and undershoot appeared; `k_xs01` with 0.02 still undershoot at 3× under k_max=32 — so both knobs act. A cheap experiment (noise fixed at 0.02 vs 0.05 at fixed k_max) would separate them (see §8).
- **Prop. 4 (block-diagonal Jacobian).** Correct. Part (b) describes the per-site perturbed-observation analysis; the paper calls it "the local-analysis form of Hunt et al. 2007". Hunt's LETKF is a deterministic ensemble transform; yours is a per-site stochastic EnKF with R-localization. Call it "per-site local EnKF (R-localization, perturbed observations)". This matters because (i) the small-N behaviour you observe (local N=16: 1.8 vs dense+GC N=16: 3.8) is the classic perturbed-observation penalty (Whitaker & Hamill 2002) and would likely vanish with an ETKF update, and (ii) R- and B-localization are not equivalent (Sakov & Bertino 2011; Greybush et al. 2011), which is a cleaner explanation of the dense-vs-local gap than anything in the record.
- **A stronger statement that is available.** Combine Prop. 1 with the extensivity remark into a single corollary that is actually checkable: *if* the model's per-site Lyapunov density converges, then D_KY^model(L)/L → ρ_θ, and the *finite-size correction of the model is O(R_prop h / L)*; the data (model density 0.2212 at L=50 with R_prop h/L = 0.75, 0.2214 at L=100, 0.2230, 0.2228) are consistent with a correction that is already small at L=100 and could be plotted against R_prop h/L. That turns the Remark into a figure.

### 4.4 Methodology

- **Finite-time D_KY from one or three ICs.** Adequate for 1D (3 ICs, 1200–2000 units, converged). Not adequate for 2D KS (one IC, 40 units) and for the single-seed Kolmogorov 1×/3× points; the paper says so for Kolmogorov but not for 2D KS. The minimal fix is in §8.
- **Per-step exponents vs physical time.** Model exponents are per snapshot step divided by Δt; the PDE's are per unit time; comparing D_KY is fine (dimensionless) and λ₁ is fine after division. State in §3.6 that the model's tangent map is that of the *latent* map, so the spectrum compares the latent dynamics to the PDE's, not the decoded field's; the equality of D_KY across the decoder is a (reasonable) smooth-conjugacy assumption, which fails where the decoder is not an immersion. One sentence suffices.
- **Gauge-anchored channel.** The 1D model and early 2D models anchor channel 0 to the patch mean; the canonical per-patch c=6 model — does it? The code pass (below) reports the checkpoint config. If the canonical 2D models do *not* anchor, §3.2's "Channel 0 is not learned" must be scoped to the 1D model.
- **IC normalization.** The bug (raw ICs, 77% → 23% for 2D KS) is fixed and documented; App. D should list which evaluations predate the fix (only Fig. 5's α=0.1 box-trained 3× point, per the caption).
- **Truth windows are shorter than model evaluations** (Kolmogorov truth 40 units vs model 400 units **[verified: record 5084–5090]**): a referee will note the asymmetry; state it and, where matched-IC truths exist, use them.
- **DA metrics.** "Skill" = free-run RMSE / analysis RMSE; "steady state" = cycles 51–100; calibration = spread/RMSE; RMSE normalized per dimension. All fine, but each must be defined in §3.7 (a Chaos referee will not know them). The free run is unperturbed and starts from the truth encoding, so skill ≈ 1 for the first ~15 cycles is by construction; show per-cycle curves (Fig. 9d) and say why the mean over 100 cycles is misleading.
- **No inflation tuning.** All 2D runs use inflation 1.0 and model noise 0.08; every configuration is under-dispersed (0.25–0.46). The comparison "local vs dense vs none" at one untuned setting is acceptable if stated as such, but the unlocalized filter's collapse at N=64 is partly an inflation question; a referee from the DA side will ask for a short inflation sweep at L=44 (cheap).
- **Reproducibility of DA numbers.** ±5–6% run-to-run on MPS at fixed seed **[verified: record 733–757]**. Fine if stated; the 2D seed-1 repeats (4.30 vs 4.23 etc.) give a direct estimate and should be quoted.

### 4.5 Code-vs-text findings (delegated pass; file:line in the report are from the current tree)

| claim in text | what the code does | consequence |
|---|---|---|
| §3.6 "All QR factorizations and tangent arithmetic are carried out in double precision on the CPU." | True for the PDE steppers. For *models*, `ks_latent/analysis/lyapunov.py:540–541` casts state and tangent to float32 for the `jvp`; only the QR (every 10 steps) is float64. | Reword. Add one float64 re-run of a model spectrum (CPU, cheap) to show the negative/near-zero exponents are unaffected; D_KY depends on exponents near zero. **[verified]** |
| §3.6 "A model trajectory starts from the encoding of a true on-attractor state, standardized with the training statistics." | True for the 2D scripts (`exp_crop_eval*.py`). The 1D transfer evaluations (`scripts/run_ltransfer_test.py:83,101`, used by E1/E4) encode the **raw** solver field (σ = 1.32); the icnorm fix covered 2D only. | The 1D numbers probably survive (the exponents are measured after a 200-step warm-up and already agree with truth to 1–3%), but the sentence is false for 1D and the evaluation must be re-run standardized (minutes). **[verified]** |
| §3.5 Eq. (stageB) and §3.6 canonical recipe | `k_xs01_v5` was trained with `w_rank = 0.02` (one-sided per-site log-det floor on the rollout). The record shows the term never fired (v6a bit-identical), so it is inert, but it is in the config. | Say so in App. A: "trained with w_rank=0.02, which never activated (control run identical to machine precision)". **[verified: checkpoint config; record 9289–9319]** |
| §3.6 "1D model … earlier two-stage schedule" | Stage 1 also used `--w-spatial 0.01 --spatial-signed` (a signed latent spatial-coherence regularizer, `losses.py:216`). Not described anywhere in the paper. | Describe it in App. A; it is likely part of why the 1D latent is so banded (D3 = 0.43). Not the banned `lambda_z`. **[verified: `scripts/section224_…sh:36–46`]** |
| §3.4 "a companion test that a halo of R_prop − 1 breaks it"; App. B "realizes this with η = R_prop − 1" | `test_forced_halo_too_small_breaks_exactness` uses η = R_prop − 2 (halo 2 vs reach 4). The 1e-6 "1D" test is an x-strip crop of a 2D propagator. | Change the text to match, or change the test (trivial). **[verified]** |
| §3.3 Eq. (loc_bound) "the same reasoning bounds the localization radius" and outline "implement as an assertion" | No assertion exists in `ks_latent/` or `scripts/`; `v_star` is only measured in Phase 2 for 1D KS. | Either implement it (one line) or present it as a design rule. **[verified]** |
| §3.7 "per-site local ensemble Kalman filter in the local-analysis form of Hunt et al." | `letkf_local.py:131–200`: per-site perturbed-observation EnKF with R-localization; no ensemble transform. | Rename; see §4.3. **[verified]** |
| §3.7 RMSE (undefined in the written sections; defined in `cycling.py:199–200`, `rmse.py:13–18`) | RMSE is **latent-space** per-dimension: ‖mean(z) − E(u_truth)‖/√d; spread is per-dimension latent std. | State this explicitly; add physical-space RMSE of the decoded mean (the brief's own convention for plots) at least at one size. A DA referee will not accept skill measured only in the model's own coordinates. **[verified]** |
| §4.3 Re convention "of Vinograd et al." | Their Re has no drag term (literature pass). The code exposes Re = 1/ν. | Define your own convention and say it reduces to theirs at α=0. **[verified by literature pass]** |
| §3.2 "Channel 0 is not learned" (canonical 2D) | The anchor is unconditional in both AE classes; the canonical c=6 checkpoints have 5 learned + 1 anchor channels. | Fine; say "5 learned channels and the anchor" in App. A. **[verified]** |
| Prop. 1 at L=25 | `Conv1d(kernel 7, padding 3, circular)` on 4 sites runs and equals explicit periodic extension; the resize loaders do not refuse P < 2r+1. | Matches the remark. Consider making the loader warn. **[verified]** |
| Kolmogorov crops "y offsets aligned to the forcing period" (outline) | `y_align=1` for all three canonical checkpoints and x-strips keep full y; alignment only exists in the 2D-crop branch (P9). | The paper's text is right; the outline is stale. **[verified]** |

---

## 5. Under-emphasized or missing results

Each: where it lives → why it matters → where it should go. Impact/effort in brackets.

1. **Dense+GC saturates by N = 16 (skill 3.77/3.63/3.54 at 44²/66²/88²), the per-site filter by N ≈ 32–64; the paper's own heuristic Eq. (nens_local) predicted ≈ 6 for 2D KS.** (`summarize_da2d.py`; outline §3.7.) This is a quantitative success of the local-dimension argument and the cleanest DA number in the paper. → §7.2, one sentence plus Fig. 9c. [high / hours]
2. **Observation-spacing threshold in 2D DA** (`docs/paper1_tasks.md:496–499`; `artifacts/experiments/da2d/probe/`): 11-unit spacing → DA worse than free run; 5.5 → 1.1; 2.75 → 3.6–4.2; pattern wavelength 8.9. This is a result (DA needs to resolve the unstable scale) and it is the diagnosis of the 1D confound. → §7.2 (a small panel) and §9. [high / hours]
3. **The unlocalized filter's collapse at fixed density tracks the number of positive exponents**: n₊ = 43/101/≈170 at 44²/66²/88²; unlocalized N=128 gives 3.47 / 2.22 / 1.11 and N=64 gives 2.31 / 0.94 / 0.83 (`summarize_da2d.py`; `tab:truth_runs`). Plot skill vs N/n₊ and cite Takeda & Miyoshi 2026 (m* = N₊+1) and Bocquet & Carrassi 2017. → §7.2 Fig. 9d with n₊ marked. [medium / hours]
4. **Learned stencil closure (RESULTS.md "Phase 11", Section 224/225):** R² of a shared quadratic stencil saturates at w = 3 sites (≈19 units) for the 1D latent, 90% at full width; sparse selection keeps 49/252 terms, mostly anchor↔hidden couplings. This is the only direct measurement of the *interaction range* of the learned rule, and it is what §3.3 should be built on; it also supports the hidden-channel/Mori–Zwanzig rationale for c > 1. → §3.3 (one sentence + number) and a short appendix or a §9 paragraph. [medium / hours; data exist]
5. **D3 Jacobian bandedness of the 1D propagator** (0.43, p < 10⁻⁴ vs ≥1000 permutations; `tab:checkpoint` in the record). Direct evidence that the learned latent dynamics are banded in site index. → App. A table row or §5.1 one sentence. [low / hours]
6. **Models track but amplify the truth's IC dependence at native size** (record 5092–5123: per-IC truths 22.46/20.52/21.63; both P2 and C19 reproduce the IC-1 dip; model scatter ±10% vs truth ±4%). This is the honest reason the 1× numbers are what they are and supports the "scaling, not points" framing. → §6.3 or App. C.4, with the small table. [medium / hours]
   **CORRECTION (2026-10-08, task A2, `docs/paper1/decisions.md`):** a round-off-level perturbation (float32 → float64) moves a single model run's D_KY at Kolmogorov 1× by up to 15% (IC 2: 26.04 → 21.99), so the per-IC model scatter is finite-time sampling noise. Do not use this item.
7. **The model can be *less* IC-sensitive than the truth** (P2 at 2×: 0.3% spread vs 12% truth spread; record 5146–5153). Under Lemma 3 this is what a contracted model looks like; worth one sentence in §8.4 as a diagnostic. [low / hours]
8. **k_max × ε interaction** (record `sec:k_xs01_ablation`): keep as the §8.4 table (v1–v7), but add the two truth-free training metrics (final `val_kmax`, free-rollout variance) per variant so the reader can see that *they do not predict the truth errors* — which is itself the point. [medium / hours]
9. **Per-site vs dense at small N** (local N=16: 1.82/1.83/2.09 vs dense+GC 3.77/3.63/3.54; crossover at N≈32). Report it; explain by perturbed-observation sampling noise (Whitaker & Hamill 2002) and R- vs B-localization (Sakov & Bertino 2011; Greybush et al. 2011); offer the ETKF update as the fix (§8). [medium / hours]
10. **Under-dispersion of the corrected 2D DA** (spread/RMSE 0.25–0.46 everywhere; model-noise sweep 0.02/0.04/0.08/0.16 → skill 4.93/4.79/4.23/3.40, calibration 0.25/0.31/0.35/0.33). Note that the earlier "over-dispersion" (T5) came from the invalid-resolution runs and must not appear. → §7.3. [medium / hours]
11. **Transfer error tracks the truth's non-extensivity** (P1/P2 signs, G2). Already slated for one Discussion paragraph; keep it there, but phrase it through the corrected Remark (the model's finite-size corrections are set by its reach). [low]
12. **Reconstruction quality and decoded spectra (AROMA bar, W7).** Not in the paper; the per-patch c=6 AE's interior reconstruction MSE (0.0118 for 2D KS; 0.0082–0.0092 for Kolmogorov) is 8 orders of magnitude above AROMA's 10⁻¹⁰ "necessary for accurate spectrum reconstruction" and yet D_KY is right. That contrast is interesting and defuses the AROMA objection *only if* you show decoded energy spectra. → §9 and App. D. [medium / hours]
13. **The free-rollout statistics diagnostic** (variance, energy split; caught P9 and the original 88² artefact). Present once in §3.6 and use it as the truth-free model-selection criterion (see §7, objection 2). [medium / hours]

---

## 6. Missing connections and literature

Verified by the literature pass unless tagged otherwise. For each: what it is → how it changes the paper.

**Thermodynamic limit of Lyapunov spectra / predicting large systems from small ones.**
- Livi, Politi & Ruffo, J. Phys. A 19, 2033 (1986) [partially verified]; Isola, Politi, Ruffo & Torcini, Phys. Lett. A 143, 365 (1990) [verified]; **Carretero-González, Ørstavik, Huke, Broomhead & Stark, Chaos 9, 466 (1999)** [verified]: sub-system Lyapunov spectra rescaled to predict the full spectrum, D_KY and entropy; its PRL companion (83, 3633) is about statistics, not spectra. Takeuchi, Ginelli & Chaté, PRL 103, 154103 (**2009**) [verified] on collective modes; Takeuchi et al., PRL 107, 124101 (2011) on extensive vs sub-extensive chaos. → Cite in §3.2 Remark and §2; replace Bunimovich–Sinai (SRB measures) as the support for "the Lyapunov density per site converges". Position your result as "the learned-rule analogue of Carretero-González et al.".
- Engelken, Wolf & Abbott, PRR 5, 043044 (2023) [partially verified]: random RNNs have extensive chaos with a size-invariant spectrum. → one sentence; a learned network being extensive is not surprising per se, which strengthens the point that the *density* is the content.

**Surrogates validated by Lyapunov spectra (all single size).** Pathak et al., Chaos 27, 121102 (2017) (L = 60, 26 exponents; **not** L = 200) [verified]; Vlachas et al., Neural Netw. 126 (2020) [verified]; Özalp, Margazoglou & Magri, Chaos 33, 093107 (2023) and Özalp & Magri, Nonlinear Dyn. 113 (2025) [verified]; Racca, Doan & Magri, JFM 975, A2 (2023) [from the project's own literature file]; Kobayashi, Nakai & Saiki, J. Phys. Complexity 5, 025024 (2024) [verified]. → §2.2; your novelty is *multi-size, assembled-model* validation.

**Zero-shot larger domains with local convolutional surrogates.** Bar-Sinai et al., PNAS 2019 (trained on 2π, run on 10× for Burgers; KS in SI) [verified]; Kochkov et al., PNAS 2021 ("essentially the exact same performance" on a 2× larger Kolmogorov domain, "because they only rely upon local features") [verified]; Stachenfeld et al., ICLR 2022 (KS to larger domains, qualitative; 3D "imperfect and unreliable") [verified]; de Villeroché et al., arXiv:2606.14597 (Bocquet group; non-chaotic benchmarks, 3-step rollouts) [verified]; Li/Ye et al., CMAME 427 (2024) local neural operator [verified]. → §2.2 and C1: the mechanism is known; what is new is the Lyapunov-level test of it in extensive chaos, and the failure of box training.

**Pathak et al. 2018 specifics** [verified from the full text]: identical reservoirs shared across sites for μ = 0 (translation invariance exploited), but a new reservoir set at each L; D_KY values 23/43/85/167/338 at L = 100–1600 are of the *true* KS (with μ = 0.01 inhomogeneity) and no reservoir exponents were computed. → §2.2 must not say Pathak computed model spectra at L=200 or transferred across L; the correct contrast is "shared weights, retrained per size; no model spectrum".

**Limited-area modelling.** Davies, QJRMS 102, 405 (1976) [partially verified]; Warner, Peterson & Treadon, BAMS 78, 2599 (1997) [verified]; Neural-LAM (Oskarsson et al. 2023) and Adamov et al., arXiv:2504.09340 (2025) [partially verified]. → §3.4: the forced halo is a Davies relaxation zone of zero width; Prop. 2 is the statement that with exact boundary data the LAM is exact. This also tells a referee why the "mismatch strip" matters: it is the classical LAM boundary-error problem.

**Ensemble size vs unstable subspace.** Bocquet & Carrassi, Tellus A 69, 1304504 (2017); Gurumoorthy et al., SICON 55, 741 (2017); Bocquet et al., SIAM/ASA JUQ 5, 304 (2017); Trevisan & Uboldi, JAS 61, 103 (2004); Carrassi et al. 2022 (Springer chapter, arXiv:2010.07063) [verified]; **Takeda & Miyoshi, NPG 33, 335 (2026)**: minimum ensemble size m* = N₊ + 1 without localization [verified]; Bergemann & Reich, QJRMS 136 (2010) [partially verified]. → §3.7 "How large an ensemble is needed" should cite these and replace "N_ens ≳ D_KY" by "N_ens > n₊ + 1 is necessary without localization (Takeda & Miyoshi); with localization the bound involves the local unstable dimension (Patil, Ott)". Your 2D data (item 5.3) can be plotted against n₊.

**Localization formulations.** Sakov & Bertino, Comput. Geosci. 15, 225 (2011) "Relation between two common localisation methods for the EnKF" and Greybush et al., MWR 139, 511 (2011) [both inferred, not verified in this pass]: R-localization and B-localization are not equivalent; Whitaker & Hamill, MWR 130, 1913 (2002) [inferred]: perturbed-observation filters need larger ensembles than square-root filters. → explains the per-site vs dense gap at N=16 and motivates the ETKF update.

**Latent DA.** ROAD-EnKF is published (Inverse Problems 39, 124001, 2023) and includes KS; it declines localization [verified]. LAE-EnKF (Tong, Wang & Yan 2026) has no KS and lists localization as future work [verified]. Si & Chen, Latent-EnSF, ICLR 2025 [partially verified]: notes that local structure is lost on encoding so LETKF-type methods lose their advantage — a quotable motivation for the per-patch decoder. Pasmans et al., arXiv:2502.12987 (2025) latent ETKF, no localization [verified]. Peyron et al., QJRMS 147, 3759 (2021) [verified]. → §2.3 and §7.

**Training with noise = Jacobian penalty.** Bishop, Neural Computation 7, 108 (1995) [inferred, standard]; Sanchez-Gonzalez et al., ICML 2020 (noise injection for rollout stability) [inferred]. → Lemma 3 part (b); explains the P9b undershoot.

**MSE emulators under-estimate the leading exponent.** arXiv:2604.21097 (adversarial-OT emulators; L96 LLE 1.615 vs 2.334 for an MSE emulator) and Ainslie et al., arXiv:2608.16084 (Jacobian eigenanalysis of KS emulators) [both verified; authors of the first not captured]. Invariant-measure training: Jiang et al., NeurIPS 2023; Schiff et al. (DySLIM), ICML 2024; Platt et al., Chaos 33, 103107 (2023) [verified]. → §3.5/§8.4: the observation is not new; the lemma's statement and the ε interaction are.

**Comoving exponents.** Kaneko, Physica D 23, 436 (1986) and Deissler & Kaneko 1987 define v_*; Pikovsky & Politi, Nonlinearity 11, 1049 (1998) is about Lyapunov-vector localization, not comoving exponents [verified]; Lepri, Politi & Torcini, J. Stat. Phys. 82 (1996) "Chronotopic Lyapunov analysis" [partially verified]. → fix the citation in §3.3; and the parabolic-PDE domain-of-dependence point in §4.2(d).

**Kolmogorov flow.** Vinograd et al. is PRFluids 11, 034402 (2026); their Re has no drag [verified]; Musacchio & Boffetta, PRE 89, 023004 (2014) for periodic Kolmogorov conventions [partially verified]. Cleary & Page's dimension is from autoencoders, not D_KY [verified]. → §4.3 wording.

**Bibliographic corrections** (all verified): Takeuchi–Ginelli–Chaté is 2009; Karimi & Paul PRE 85, 046201 (2012) is "Quantifying spatiotemporal chaos in Rayleigh–Bénard convection" (their "extensive chaos" paper is Chaos 20, 043105 (2010), Lorenz-96); the L = 22 value 5.2 is Edson et al., not Cvitanović et al.; Wittenberg & Holmes' "Spatially localized models…" is Nonlinear Dyn. 25, 111 (2001); Nipen et al. has a 2025 AIES version; Coleman & Moré (SIAM J. Numer. Anal. 20, 187, 1983) is the graph-colouring statement to cite with CPR 1974; Edson et al. give D_KY ≈ 0.226L − 0.160 for L ≳ 80 and 5.198 at L=22 (so your 5.26 is 1.2% high — fine, but quote their number).

**Not found by either pass**: any learned surrogate tested for D_KY growing linearly with L across several sizes; any latent EnKF with provable locality; any forced-halo crop training on chaotic PDEs. These are the three claims that can be stated as new.

---

## 7. Anticipated referee objections, ranked

1. **"Your 1D DA degradation with L is observation starvation."** (See §4.2a.) *Response:* re-run E2/E7 at fixed spacing (3.1 units) at L = 200/400; present 2D (fixed density) first. If the re-run reverses the unlocalized-vs-GC ratio, say so and drop C5's 1D half. [must; hours of compute]
2. **"The α = 0.1 main-text model was chosen by looking at the truth."** `k_xs01_v5` is the best of seven variants at 3× and among the worst at 1×/2×; v7 has the best validation loss (0.02105 vs 0.02262) **[verified: tasks file]**. *Response:* pre-register a truth-free selection rule (validation rollout loss, or free-rollout statistics vs the *training* data) and report the selected model in the main text; show v4/v5/v7 together in §8.4 with the rule stated. The same applies, more mildly, to using `c19_e0d0_c6` (k_max = 12) at α = 0.05 while the recipe says long rollouts. [must; hours]
3. **"Single 40-unit truths at Kolmogorov 1×/3× and for 2D KS, and you interpret 2–5% differences."** *Response:* finish the two paused α = 0.1 3× seeds (checkpointed at 46%, ≈2–3 h each) and run two more 2D-KS seeds at 44² and 66² (≈1–3 h each); until then, never read a difference below ~15% at those points as anything. Remove "cannot follow the truth's 4.6% rise". [must; compute]
4. **"SEC is not a matrix taper; your 'SEC fails' is your implementation."** *Response:* §4.2b; implement serial SEC or PSD projection and re-run E3 at fixed density, or restate the claim narrowly. [must; hours]
5. **"This is not an LETKF; small-N performance is poor; no inflation; RMSE is in your own latent coordinates."** *Response:* rename; add the deterministic ETKF update per site (the per-site problem is already in ensemble space; ~50 lines) and a short inflation sweep at L=44; report decoded-field RMSE. [should; hours–day]
6. **"Memory and cost claims."** The dense filter was never observed to fail after the leak fix; the local filter is slower than dense at d ≤ 1536 and uses more memory. *Response:* report measured numbers (CPU 23/55/87/1914 s dense vs 50/218/174/1450 s local; peak RSS) and the d² accounting; promise nothing about "<1 GB". [must; hours]
7. **"Lemma 3 is a textbook fact and your horizons never exceed a Lyapunov time."** *Response:* §4.3; add the Jacobian-penalty part; quantify k_max Δt λ₁ in the text; add the cheap noise-vs-k_max separation experiment if time allows. [should; hours (+ optional 1 h training)]
8. **"The light-cone bound is irrelevant at your settings."** *Response:* §4.2d; rewrite §3.3 around the measured interaction range (stencil fit, encoder-reach rule from P6/C16). [should; hours]
9. **"Why does box training fail? The wraparound story is a hypothesis."** *Response:* say so (the record does). A cheap direct test exists: box-train on a 16-site box (reach covers 37%) at α = 0.1 and check whether transfer recovers; or evaluate the box-trained model's one-step error on crops of the wide domain (no training). [nice; hours–1 training]
10. **"Only four training trajectories; the 2D-KS training set is 100 time units total."** *Response:* report the 8-trajectory `k_xs005_v2` result (4.4% at 2× vs 4.8%; 1× spread worse) as a data-size sensitivity; state the data sizes plainly in Table I. [should; hours]
11. **"Architecture differs between 1D and 2D, and the 1D model has an undocumented regularizer and raw-IC evaluation."** *Response:* document `w_spatial`, re-evaluate 1D with standardized ICs; keep the architecture difference as a stated limitation (the 2.7 h retrain is optional). [must (re-eval) / nice (retrain)]
12. **"Float32 tangent propagation for the model spectra."** *Response:* one float64 control run per system. [should; hours]
13. **"No physical-space DA baseline; synthetic regular observations."** *Response:* state as limitations; a PDE-ensemble EnKF at L=44 (2D KS, 128², N=64, 100 cycles) is feasible in hours if you want one number. [nice; compute]
14. **"Kolmogorov transfer is channel widening only."** Already stated; keep it in the abstract too. [must; text]
15. **"What is new relative to Bar-Sinai/Kochkov/Pathak?"** *Response:* §2.1 wording. [must; text]

---

## 8. Proposed additional work

Compute on the stated laptop (MPS training 20–60 min; 1D DA points ≈1–3 min; 2D DA 100-cycle runs ≈ 25 min (L=44) to ≈ 1 h (L=88), ≈ 1 h (L=220) per point; 2D truth runs 0.5–10 h CPU).

| # | item | question answered | expected impact on the paper | compute | pre-registered decision rule | priority |
|---|---|---|---|---|---|---|
| 1 | Re-run 1D E2/E7 (PFF & EnKF, none vs GC, N ∈ {8…256}) at L = 200, 400 with `--obs-stride-at-old-L` chosen so the physical spacing stays 3.1 units (stride 8 in grid points at every L) | Is the 1D localization-necessity result real at fixed observation density? | Decides C5's 1D half and the content of Fig. 8 | 24 points × 2 sizes × ~2 min ≈ 2 h | If at N=64 unlocalized skill < 1 and GC/none > 1.5 at both sizes, C5 stands as written; if unlocalized skill ≥ 2, restate C5 as "localization raises skill by ×(ratio) at fixed N; absolute skill is set by observation density" | must |
| 2 | Re-evaluate the 1D model at L = 25–400 with standardized ICs (`run_ltransfer_test.py` fix) | Do the §5 numbers survive the IC bug? | Protects Table III; removes a false sentence | 5 runs × minutes | Accept if every D_KY moves < 1%; otherwise replace the table | must |
| 3 | SEC: PSD-project the corrected covariance (clip eigenvalues) **and/or** implement serial SEC; re-run E3 at fixed density at L = 200 | Is "SEC fails at transfer size" a method property or an implementation property? | Rewrites the C5/C6 SEC sentence either way | hours of code + 12 points × 2 min | If serial/PSD SEC runs and beats GC, drop the "cannot transfer" claim; if it runs and loses, state "GC beats SEC at the transfer size"; the indefiniteness result is reported as a matrix-form artefact regardless | must |
| 4 | Finish the two paused α = 0.1 3× truth seeds (`truth_a0p1_x3_seed{1,2}_T120.ckpt.npz`, 46% done) | Is the 2.3% headline at 3× real or a single-seed accident? | Fixes the only Kolmogorov number quoted below 5% | ≈2.5 h CPU each, parallel | Report the 3-seed mean ± spread; if v5's error vs the mean exceeds the spread, say "within/outside truth uncertainty" accordingly; no model re-selection | must |
| 5 | Two more 2D-KS truth seeds at 44² (150 dirs) and 66² (300 dirs), 40 units, or one seed each at 80 units with running D_KY | How uncertain are the 2D-KS truths? | Removes the "4.6% rise" over-reading; gives error bars to Fig. 6a | 44²: ~1–2 h; 66²: ~4–8 h CPU | Report mean ± spread; the model is "consistent" if inside the spread | should |
| 6 | Per-site ETKF (deterministic transform) variant of `LETKFLocal2D`; re-run N ∈ {8,16,32,64} at L = 44 | Is the small-N weakness of the per-site filter the perturbed-observation penalty? | Likely turns item 5.9 from a weakness into a clean result; makes the "Hunt et al." citation accurate | ~half day code; 4 × 25 min | If ETKF at N=16 reaches ≥ 3 (dense+GC level), adopt it as the per-site filter in the paper | should |
| 7 | Inflation sweep {1.0, 1.05, 1.1, 1.2} for dense-none, dense-GC, local at L = 44, N = 64, 100 cycles | Does untuned inflation drive the under-dispersion / unlocalized collapse? | Pre-empts the standard DA objection | 12 × 25 min = 5 h | If the unlocalized N=64 skill at L=44 changes by > 30% with inflation, redo the L=66/88 unlocalized points with the best inflation | should |
| 8 | Physical-space RMSE of decoded analysis mean (post-hoc from saved per-cycle states if any; otherwise re-run one point per size with state saving) | Is skill in physical space comparable to latent-space skill? | Required for DA readers | hours | Report both; if they differ by > 20%, discuss | should |
| 9 | Float64 control of one model Lyapunov spectrum per system (CPU) | Does float32 tangent propagation bias D_KY? | Supports §3.6 wording | 3 × ≤1 h | Accept if D_KY changes < 0.5% and n₊ unchanged | should |
| 10 | Lyapunov spectra overlay λ_i vs i/L for L = 100/200/400 (model) and truth (W5; H1) | Is the whole spectral density transferred, not just D_KY? | Strengthens §5 with Fig. 3c; connects to Carretero-González | < 1 h (data exist) | If model and truth densities agree within 5% over the positive part, say so; report where they diverge (collective/negative modes) | should |
| 11 | Clean memory re-measurement of local vs dense (pending `run_da2d_memfix.sh`) + a memory profile of `LETKFLocal2D` at d = 1536 | Why is the local filter's RSS 3–9 GB above the truth baseline? | Decides whether "cost ∝ area" can be claimed | pending + hours | Claim "memory ∝ P" only if filter-only RSS is linear in d within 30% over 384–21,600 | must |
| 12 | Noise vs rollout-length separation: train `k_xs01` with k_max = 16 at terminal noise 0.05 (one run) | Which knob drives the undershoot: horizon or input noise? | Decides the wording of Lemma 3 / §8.4 | 1 training (~1 h) + 9 evals (~2 h) | If 3× error rises by > 5 points at fixed k_max when noise doubles, the Jacobian-penalty mechanism is dominant | nice |
| 13 | Decoded energy spectra + reconstruction MSE table (W7) | Does the per-patch AE reproduce spectra despite 10⁻² reconstruction MSE? | Defuses AROMA | < 1 h | Report; no gate | should |
| 14 | Wraparound test: box-train α = 0.1 on a 16-site box (reach covers 37% of the ring) and evaluate at 2×/3× | Is the box-training failure the periodic seam? | Converts the §6.1 mechanism from hypothesis to evidence | 1 training + 6 evals ≈ 3 h | If 3× error falls below 10%, the seam explanation is supported; if it stays ~20%, the "small-box statistics" explanation wins | nice |
| 15 | L = 37.5 (6 sites) 1D truth + transfer | Where does transfer actually stop? | Replaces the unresolvable L = 25 point | truth ~1 h, eval minutes | Report; no gate | nice |
| 16 | 2D-KS rollout-statistics diagnostic (C1b) | Does the truth-free diagnostic pass for the headline 2D model? | Supports using it as the model-selection rule (objection 2) | ~1 h code + minutes | Pass if variance and band shares within 15% of truth | should |
| 17 | 88² 2D-KS truth (≥ 500 directions) | Direct truth for the flagship 2D number | Nice but expensive; items 5 and 10 are better value | many CPU hours | — | nice |

---

## 9. Concrete edits to the written sections

Line numbers refer to the current files. Every edit here is text-only (effort: hours); its impact is that of the §4 item it implements.

### `sections/03_method.tex`

**§3.2 Remark (lines 167–181).** Before: "For a translation-invariant lattice map of finite range, the distribution of Lyapunov exponents per site converges as P→∞; this is the thermodynamic limit of the Lyapunov spectrum [ruelle1982, bunimovichsinai1988]. … A local model can therefore match the true density ρ, but it cannot represent finite-size departures of the true system from Eq. (extensive)." After: "For translation-invariant lattice maps of finite range the Lyapunov spectrum is expected, and numerically found, to have a thermodynamic limit: the exponents per site converge to a density as P→∞ [livi1986, isola1990, carretero1999chaos; rigorous results exist for weakly coupled maps]. Hence D_KY^model(Ω) = ρ_θ|Ω| + o(|Ω|) with ρ_θ set by the weights. A local model can match ρ, but its finite-size corrections are set by its own reach (R_prop h), not by the physics, so it need not reproduce the true system's departures from Eq. (extensive) at sizes comparable to that reach. We use this as an interpretive principle …"

**§3.3 (lines 199–217).** Replace the light-cone paragraph. After: "What sets R_prop physically is the range over which one snapshot interval couples sites. For a dissipative parabolic PDE the one-step map has unbounded domain of dependence with an exponentially decaying kernel, so the operative length is an interaction range, not a wave speed. We measure it two ways: by fitting a shared quadratic stencil to the trained latent dynamics, whose explained variance saturates at a half-width of 3 sites (≈19 physical units, about two pattern wavelengths) in 1D (Appendix X), and by the encoder-reach ablation of Sec. 6.3/8.5, where an encoder reach of 2 sites fails and 6 sites succeeds, consistent with the unstable band |k| ∈ [2,8) of Sec. 6.5. The spreading velocity v_* of perturbations [kaneko1986, deisslerkaneko1987], measured at 1.26 length units per time unit for 1D KS, gives a much weaker lower bound v_*Δt ≈ 1.3 units per step; it is not the binding constraint at our settings. The same reasoning bounds the localization radius below by the interaction range plus the forecast window's spread, which is far inside the Gaspari–Cohn support radius we use (37.5 units in 1D, 33 in 2D)." Drop Eq. (loc_bound) or keep it with the sentence "which, with R_enc = 0 and our windows, is < 7 units and never binds."

**§3.5 after Lemma 3 (lines 347–359).** Before: "The second term … grows with k … until it reaches the climatological variance. A propagator trained toward Φ* at large k is therefore rewarded …" After: "The second term is non-zero at every k > 0: along the leading direction it grows as σ²e^{2λ₁kΔt} from the input noise until it saturates. None of our training horizons exceeds one Lyapunov time (k_max Δt λ₁ ≈ 1.1 for 1D KS, 0.5 for 2D KS, 0.2–0.8 for Kolmogorov flow), so the effect is a bias that grows with k_max rather than a collapse to the climatological mean. A second mechanism acts in the same direction: with input noise ξ the expected loss expands as E‖Φ(z)−z^k‖² + σ²E‖∂Φ/∂z‖²_F + O(σ⁴) [bishop1995], so the noise anneal is a direct penalty on finite-time growth. Both predict a systematic under-estimate of D_KY that grows with k_max and with σ, which is what we observe (Sec. 8.4; the P9b run, which raised terminal noise to 0.05, undershoots by 11–15%)."

**§3.6 (lines 393–399).** After: "For a model, tangent vectors are propagated through F by forward-mode automatic differentiation in single precision, re-orthonormalized every 10 steps by a double-precision QR on the CPU; a double-precision control run changes D_KY by [X]% (Appendix C). … A model trajectory starts from the encoding of a true on-attractor state standardized with the training statistics [after item 2 of §8 is done; until then: '…(2D); the 1D evaluations encoded the unstandardized field, see Appendix C']."

**§3.7 (lines 445–449).** Before: "a per-site local ensemble Kalman filter in the local-analysis form of [hunt2007] …" After: "a per-site local ensemble Kalman filter: each site is updated from the observations within the taper's support, with observation-error variances divided by the taper value (R-localization, as in [hunt2007]) and the same perturbed-observation update as above; unlike the LETKF it does not use an ensemble transform." Add after the itemize: "Analysis error is reported as the per-dimension RMSE between the analysis mean and the encoded truth in latent space, ‖z̄ − E(u)‖/√d, and skill as the ratio of the unperturbed free run's RMSE to the analysis RMSE over the second half of the cycles; [physical-space RMSE of the decoded mean is given in Appendix E]."

**§3.7 "How large an ensemble" (lines 486–502).** Add: "Without localization a deterministic ensemble filter needs at least n₊ + 1 members to span the unstable–neutral subspace [bocquet2017, takeda2026]; here n₊ = 43, 101 and ≈170 at 44², 66² and 88²."

**Canonical recipe (lines 366–382).** Add: "The α = 0.1 model was additionally trained with a one-sided per-site log-determinant floor (w_rank = 0.02) that never activated during training (a control without it is identical to machine precision)." And for the 1D model: "with a signed spatial-coherence regularizer on the latent (weight 0.01, Appendix A)."

### `sections/04_systems.tex`

- Lines 97–100: "In the Reynolds-number convention of [kolmogorovregimes2026]" → "With Re = √(νn²+α)/ν, which reduces to the convention of [kolmogorovregimes2026] at α = 0, these are …".
- Lines 40, 134: "more than 100 Lyapunov times" → "100 or more Lyapunov times for L ≥ 50 (at L = 25, λ₁ = 0.0013 and the window covers fewer than two)".
- Lines 139–140: "8–15 time units … 3–5 of them" → "6–16 time units … 2–6 of them".
- Lines 154–157 (2D KS): add "Each is a single 40-unit run; by analogy with the Kolmogorov runs of Appendix C.3, differences of 10–20% between such runs are expected, so the 4.6% difference between the two densities is not significant."
- Line 44: cite Edson et al. for 5.2 at L = 22, not Cvitanović et al.

### `sections/05_results_1d.tex`

- Lines 77–85 (L = 25): rewrite per §4.2(f): "At L = 25 the true finite-time exponent, 0.0013 over 1200 time units, is not distinguishable from zero and the density (0.062) is far below the extensive value; the transferred model, whose 4-site lattice is narrower than its 7-site kernel, is not chaotic at all. This size is outside the model's domain of validity by construction (Appendix B), and the truth is at most marginally chaotic, so L = 25 bounds where transfer can be tested rather than where it fails."
- Lines 98–106: add "The 1D model was trained with an additional latent spatial-coherence regularizer (Appendix A)."
- Table caption: state "mean of per-IC errors" convention where applicable (one IC here) and that the L = 100 value is the trained-size model.

### `sections/06_results_2d.tex`

- Line 44: "94–100%" → "93–100%".
- Lines 81–83: "costs nothing if the channel count is raised" → "is at least as accurate once the channel count is raised, at a 30% higher reconstruction error (Appendix D)".
- Lines 104–105: name `k_xs005_noreg`.
- Lines 146–153: replace "cannot follow the truth's 4.6% rise … needs an 88² truth" with "the two truths differ by 4.6%, within the uncertainty of single 40-unit runs (Sec. 4.4), so the truth is consistent with extensive scaling; the model's 88² value lies on the extensive line through both."
- Lines 171–176: "For Kolmogorov flow at α = 0.05, 3×, we decomposed …" add "(the window-trained c = 3 model P2; the canonical model was not analysed)".
- Table IV caption: add "errors are means of per-IC relative errors; at α = 0.05, 1×, each IC is scored against its matched truth".
- §6.3 rollout-length paragraph: state that the main-text α = 0.1 model was chosen by [the rule you adopt]; if it remains v5, say it was chosen for its 3× accuracy after the ablation of Sec. 8.4 and that v7 has the lower validation loss.

### `appendix/B_proofs.tex`

- Prop. 1 statement (in §3): add "for shifts by whole patches" and "provided every lattice axis has at least 2r+1 sites for every kernel radius r".
- Lines 138–144: "with η = R_prop − 1" → "with η = R_prop − 2" (or change the test).
- Remark after Lemma 3 (lines 174–184): keep; add the Bishop expansion as a second remark.
- Prop. 4(b): say "perturbed-observation analysis with R-localization" explicitly (it already does in the proof body; make the statement match).

### `appendix/C_ground_truth.tex`

- Add a C.3b paragraph on 2D-KS truth uncertainty (single runs; expected ±10–20%).
- Add the model-side Benettin settings (float32 jvp, QR every 10 steps, 200-step warm-up, 2000 steps = 400 time units for Kolmogorov) and the float64 control.
- Line 107: "published values" → "earlier values (56.84 survives only in the evaluation files)".

### `refs.bib`

Fix/add per §6: Takeuchi 2009; Karimi & Paul title; Carretero-González Chaos 1999; Livi 1986; Isola 1990; Kaneko 1986 (already); drop Pikovsky & Politi 1998 for v_* (or cite it for vector localization only); Bocquet & Carrassi 2017; Takeda & Miyoshi 2026; Davies 1976; Warner 1997; Bishop 1995; Whitaker & Hamill 2002; Sakov & Bertino 2011; ROAD-EnKF journal version; Coleman & Moré 1983; Wittenberg & Holmes 2001 (Nonlinear Dyn.); Vinograd PRFluids 2026.

---

## 10. Minor issues

- `main.tex`: `\input{sections/01_introduction}` etc. are commented out; `app:architecture` is referenced throughout but no `appendix/A_architecture.tex` exists; `\ref{sec:failure}`, `sec:da_results`, `sec:fail_*`, `sec:discussion` are dangling until those sections exist (the PDF compiles with `??`).
- Table II (`tab:truth`): "Kolmogorov, α=0.1 … density 15.9/15.4/18.8" is D_KY per 2π box; say "per box" in the column head; the 1D "density" column mixes per-length with per-area without a unit.
- Table IV header "1.5× / 2×" and "2× / 3×" is hard to read; split 2D KS and Kolmogorov into two sub-tables or label columns by system.
- Fig. 3(a): the L = 25 "collapses" annotation sits on top of the truth marker; move it. Fig. 3(b) y-range 0.216–0.228 exaggerates a 1.5% offset; say so in the caption or widen.
- Fig. 5: the 0.3% floor on a log axis makes "canonical 2D KS 1×" look like a bar; state the floor in the caption (done) but consider a linear axis with a broken scale for the 93–100% bars.
- Fig. 6(b,d): truth error bars are "spread of runs"; at 1×/3× single runs have no bar, which visually suggests precision; add a shaded ±15% band for single-run truths per App. C.
- Fig. 8: panels (b),(c) must state the observation spacing (6.25 and 12.5 units) in the caption until re-run; the "SEC failed" markers at skill ≈ 0 are misleading (no value exists) — use a hatched region or annotate.
- Fig. 9: (c) and (d) have different y-ranges; (e) spread/RMSE is only shown for the local filter — add dense; (f) is an empty placeholder (fine until the re-measurement; do not ship empty). (a) axis "observation" vs "latent coordinate (site-major)" fine; mark the 6-column block width.
- Notation: R_enc h (sites × length) vs r_loc (length) vs c (channels) vs c_GC (taper parameter in sites) — c is overloaded; rename the GC parameter (e.g. ℓ_GC).
- "Kolm." abbreviation in Table IV; "2D KS" vs "2D Kuramoto–Sivashinsky" inconsistent.
- §3.1 "scalar field u" — Kolmogorov vorticity is scalar, fine; but "dissipative PDE ∂_t u = N(u)" conflicts with the Kolmogorov notation N for forcing wavenumber n; fine as long as n is lower-case everywhere (it is).
- §3.2 "The output projection has no bias, since a free per-channel offset … drifts" — give the one-line reason the record gives (it aliases onto one Fourier mode of the flattened index).
- §4.1 "snapshot interval Δt = 1.0" vs §3.1 "snapshot interval Δt": in Kolmogorov Δt = 0.2 and in 2D KS 0.1; Table I has them; fine.
- Outline stale items: §4.1 "measured 22.14"; C2 "within 2%"; §6.3 "4.8% at 2×" for c19_e0d0_c6; §7.2 "<1 GB", "killed by the OS", T2/T3 CPU times (invalid runs); §7.3 "calibration overshoots to 1.3–1.4" (invalid runs); Fig. 9 "memory trace from T3 log".
- `docs/paper1_tasks.md` E6/E1c "unexplained memory spike" is now explained (leak); mark it.
- Record `tab:da_transfer` caption vs E2 caption: n_prop_steps 3 vs 5 with identical numbers.
- The Chaos "lead paragraph" (quotation environment) and the abstract are empty; fine for a draft.

---

## 11. Disagreements with recorded decisions

1. **"Treat the existing single-seed α = 0.1 3× truth (56.35) as truth; the paused seeds 1–2 are not resumed."** Disagree. The 3× point is the only Kolmogorov number quoted below 5% (v5: 2.3%), it is the number that selected v5 over v4/v7, and Appendix C's own analysis says 40-unit single runs are biased +10% on average with −17/+23% excursions. The two checkpoints are 46% done; finishing them costs ≈2–3 CPU-hours each and can run in the background. Evidence: `tab:running_dky`; `paper1_tasks.md` A8 note. [impact high / compute hours]
2. **Main-text α = 0.1 model = `k_xs01_v5`.** Disagree with the *selection procedure*, not necessarily the model: v5 was chosen for its 3× truth error after seven variants were evaluated against the truth; v7 has the better validation loss and the best 1× error. Adopt a truth-free rule (validation rollout loss, or free-rollout statistics against training data), state it, and show v4/v5/v7 in §8.4. If the rule picks v7, the 3× error becomes 9.8% and the "scaling within uncertainty" framing still holds. Evidence: `paper1_tasks.md` B1-v7 table. [impact high / hours]
3. **"The 1D DA results are not affected by these bugs."** True for the resolution and autograd bugs; false in effect, because of the observation-density scaling (§4.2a). The re-run is cheap. [impact high / hours]
4. **"Existing truth runs are the reference; more may be added later."** Agree for the paper's mechanics, but the 2D-KS truths need at least one more seed before §6.4 says anything about the 4.6% difference. [impact medium / compute]
5. **"Per-patch vs full-reach decoder in DA → paper 2."** Agree. But the per-site-vs-dense small-N gap (item 5.9) is in paper 1's data and must be explained there; the ETKF variant is the cheap fix. [impact medium / hours]
6. **Venue.** Agree with *Chaos*, conditional on §7 being written for a dynamics audience (definitions of skill/spread/localization in-line) and the DA claims being scoped as "in a learned latent space".
7. **"Lead with scaling for Kolmogorov."** Agree, and go one step further: state what would falsify it (model density outside the truth's seed spread at two of three sizes) and show the n₊ counts, which are less noisy than D_KY.

---

## 12. Open questions I could not resolve

- Whether the 1D §5 numbers change under standardized ICs (item 8.2 resolves it in minutes).
- Whether the 1D unlocalized-vs-GC ordering survives at fixed observation density (item 8.1).
- Why the per-site filter's peak RSS is 3–9 GB above the truth-generation baseline at d = 1536–21,600 (pending `run_da2d_memfix.sh`; a line profile of `LETKFLocal2D.analyze` would answer it).
- Whether the box-training failure is the periodic seam or the small-box statistics (item 8.14).
- The provenance of the Re convention √(νn²+α)/ν (not Vinograd et al.).
- The L = 440 DA point (running at the time of review; `da2d_C_L440_local_N64_s0` has no JSON yet).
- Whether `w_spatial = 0.01` in the 1D recipe matters for transfer (never ablated).
