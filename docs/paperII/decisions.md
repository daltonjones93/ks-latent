# paperII decisions log

Paper II (data assimilation; target JAMES), split from `docs/paper1` per `docs/split_paper_tasks.md`.

Every decision taken before the split, with its evidence, is in
[`docs/paper1/decisions.md`](../paper1/decisions.md) and still applies here. This file records
only decisions taken during and after the split (cuts, moves, renumbering, new results), in the
same format: what was decided, why, and the artifact that holds every number quoted.

---

## 2026-10-11 — S0: split decisions (user)

- S0.1 split line: Paper I = dynamics and transfer; Paper II = data assimilation. The per-patch
  decoder stays in Paper I with one forward pointer; the block-diagonal Jacobian proposition
  (paper1 Prop. 3) and its proof move to Paper II, as its Prop. 1.
- S0.2 journals: Paper I → Chaos; Paper II → JAMES (QJRMS fallback).
- S0.3 the 1D DA results stay in Paper II (subsection + appendix).
- S0.4 Paper I is finished first and posted to arXiv; Paper II cites the preprint.
- S0.5 Paper II gets a Kolmogorov-flow DA section: latent filter only, noisy-trained 12-channel
  model, no physics-based comparator (Block SK).

## 2026-10-11 — S1: build infrastructure

Figures and tables are generated into `docs/paperII/{figures,tables}` by `make paperII-assets` under
content names (`scripts/paper_figures/_style.py: content_name`); `make paperII` builds the PDF,
runs `check_numbers.py --paper paperII` and fails on undefined or duplicate references. The
generated table bodies are byte-identical to paper1's (checked at the split).

## 2026-10-11 — S4 first pass (S4.3–S4.9, S4.11–S4.13, S4.16, S4.17; S4.10, S4.19 partly)

What was written, and the choices made in writing it:

- **Section layout.** §2 Background (filters and localization with the n₊ and extensivity paragraph;
  latent DA; imperfect models and remedies; what is new, with a new prior-work table); §3 Method
  (latent field restated without proofs, citing Paper I Props. 1–2; per-site filter; paper1 Prop. 3
  is now **Prop. 1**; shifted tilings; ensemble size; comparator); §4 System, model and ground
  truth (2D KS, the 12-channel model's validation, 1D KS); §5 tilings and sizes (paper1 §7.1–7.4);
  §6 physics comparison (paper1 §7.5 moved as is, P7/P9 placeholders kept); §7 noisy training;
  §8 forecasts, robustness, density, cost; §9 the 1D case. Appendices: A settings (model table,
  analysis schemes, comparator), B proof of Prop. 1, C ground truth (short; cites Paper I), D DA
  details (networks, Jacobian cost, 1D, round trip, run-to-run spread), E reproducibility.
- **Paper I citation key** `jones2026transfer` (in preparation) in `refs.bib`; text says
  "Paper~I" after the first `\citet`.
- **Prior-work table (S4.3).** Columns: latent DA / localized / other sizes / wrong model
  (compared with a filter that uses a physical model with model error and its remedies; ∘ =
  compared only with the true-model filter). The entries for peyron2021, chen2023roadenkf,
  pasmans2025, tong2026laeenkf and si2025latentensf follow paper1's §2.3 statements (global
  latents, no localization); bach2025 and brajard2020/bocquet2020 are marked as not latent DA and
  compared only with true-model filters. **All prior rows must be checked against the papers
  before submission (G-pass)**; the least certain are bach2025's "localized" and every
  "wrong model" entry.
- **Validation of the DA model (S4.6), new evidence.** `tabV_model_validation.py` →
  `tables/model_validation.tex`. New runs today: free-rollout statistics of the noisy-trained
  (`p7c12d10_ks2d`) and clean-trained (`p7c12_ks2d`) 12-channel models at 44² and 66², seeds 0–1
  (`scripts/exp_rollout_stats_ks2d.py --no-truth`;
  `artifacts/experiments/rollout_stats/p7c12{d10,}_ks2d_ks2d_n1000.json`); the truth columns
  come from the p7b run file, which integrates the PDE from the same initial conditions.
  Findings stated in §4.2: D_KY +3.6% / −2.1% (seed-0 truth), n₊ inside the truth range; **λ₁ of
  the noisy-trained model (0.221, 0.217) is above the truth range (0.165–0.194, 0.189–0.204)**;
  free-rollout std 5–7% high; slightly too much energy at |k| ≥ 1.4 (0.85% vs 0.74%, 0.69% vs 0.54%).
- **1D model is not per-patch** (R_enc = R_dec = 6): §4.3 says Prop. 1 does not apply in 1D and the 1D
  filters localize the latent covariance instead.
- **Model spectra of the 12-channel model use 384 / 512 tangent directions** (from the run JSONs),
  not all d_lat as paper1 App. C states for the main-text models; App. C here says so.
- **Training time of the 12-channel model:** ~7 h, almost all of it Stage B, on a shared workstation
  (`artifacts/logs/crop_train_p7c12d10_ks2d.log`).
- **New Fig. 1** (`scripts/paper_figures/fig_concept_da.py` → `figures/concept_da.pdf`): (a) schematic
  of the per-patch latent field, (b) the measured |J| at 44², (c) tiling lattices, (d) blend
  weights. Panels (b)–(d) duplicate Fig. da_2d(a) and Fig. tilings(a,b); dropping those panels
  needs Paper II variants of `fig09_p2_da.py` and `fig13_tilings.py` (S4.19, left open).
- **New assets in the Paper II lists of the Makefile:** `tabA_architecture` (writes
  `architecture_da`: 1D model and 12-channel model), `tabC_truth` (`running_ks2d`),
  `tabV_model_validation`, `fig_concept_da`, `fig02_systems` (writes `systems_ks2d`, the 2D KS panel
  alone).
- **Number check:** 0 UNMATCHED, 0 SUSPECT; two EXPLAINED entries (the 2D KS densities 0.0424 and
  0.0434, means of the truth runs over the area).

## 2026-10-11 — J17G: the physics-based filter's tuning grid was incomplete (user approved the fix)

The J17G runs (`scripts/run_j17g.sh`) tuned the wrong-model filter over c ∈ {8.25, 16} × RTPS ∈ {0.5,
0.7, 0.9, 1.0}, narrower than the grid App. D states for every physics-based filter (c ∈ {8.25, 16, 24} ×
RTPS ∈ {0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, multiplicative inflation 1.1/1.2), and ran the true-model
filter only at the wrong model's best setting. At δ = 0.2 that true-model filter reached 0.469
(0.468/0.476/0.463), worse than the latent filter's 0.341, with spread/error 3.3, so the comparison was
limited by the physics filter's tuning. Results before the fix (provisional, not for the paper):
δ = 0.1: physics 0.328 (0.324/0.326/0.333; best c8.25 RTPS 0.7, no remedy helped), latent 0.310
(0.300/0.307/0.323) → tie (−5.5%); δ = 0.2: physics 0.489 (0.480/0.496/0.492; c8.25 RTPS 0.9 + a,b,g
estimation), latent 0.341 (0.341/0.345/0.337) → latent wins (−30%). `scripts/run_j17g_tune.sh` completes
the grid for both filters (true-model runs tagged `true_`), reruns the remedies at the new best tuning and
the best configuration on seeds 1–2; driver `run_j17g_tune_all.sh` (δ = 0.2 and 0.1, then 0.3; log
`artifacts/logs/j17g_tune.log`). P7 is written from these tuned results only.

## 2026-10-11 — Prior-work table checked against the papers (user: check them yourself; arXiv HTML)

| work | source read | latent DA | localized | other sizes | wrong model |
|---|---|---|---|---|---|
| peyron2021 | arXiv:2104.00430 PDF | ✓ (ETKF-Q-L, 40-dim latent of a 400-dim augmented L96) | — (none mentioned) | — | ∘ (full-space ETKF-Q with the true L96 dynamics and an additive model-error term σ_Q) |
| chen2023roadenkf | arXiv:2301.11961 HTML | ✓ | — ("we do not pursue the covariance localization approach") | — | — (baselines SINDy-AE, AD-EnKF: learned models) |
| pasmans2025 | arXiv:2502.12987 HTML | ✓ (VAE pair) | — (convolutional VAE as future work) | — | ∘ (ETKF with the same physical model; conceptual circle model) |
| tong2026laeenkf | arXiv:2603.06752 HTML | ✓ (learned linear latent dynamics) | — (future work; baselines localized) | — | ∘ (physical-space EnKF with the nonlinear model; "true" not stated explicitly, twin setting) |
| si2025latentensf | arXiv:2409.00127 HTML | ✓ | — (only the Latent-LETKF baseline) | — | ∘ (filters share the same model: shallow-water solver / FourCastNet) |
| bach2025 | arXiv:2504.17836v4 HTML | — (physical-space learned gain correction) | **✓ (learned distance-based localization; was "—")** | — (transfer across ensemble size only) | ∘ |
| brajard2020 / bocquet2020 | arXiv:2001.01520 PDF | — | — (N = 30, "localization unnecessary") | — | ∘ (DA with true L96: 0.34 vs surrogate 0.80, i.e. 2.4×; interpolation 2.32) |

Changes: bach2025 localized → ✓; the "wrong model" entry of peyron2021, pasmans2025, tong2026laeenkf and
si2025latentensf → ∘ (they compare with a true-model filter, which the column's ∘ denotes). §2.2 text: the
Latent-EnSF statement softened to its own wording ("may lose the local structure"), and bach2025 described as a
learned gain correction with learned distance localization, at its training size. bocquet2020 not re-read (grouped
with brajard2020; same framework).

## 2026-10-11 — Correction: the latent filter's update is perturbed-observation, not a transform

paper1 (§3.8, App. E.2) and the first Paper II draft said every 2D experiment uses the deterministic
per-site ensemble transform. The code says otherwise: `scripts/exp_h9_tiled_da.py` (every 2D latent run)
uses `LETKFLocal2D`, the per-site perturbed-observation EnKF with R-localization, and the decision log of
2026-10-08 (D4) records the deterministic `LETKFLocal2DETKF` as "not adopted" (equal skill to within 0.15
at every ensemble size, but N = 16 below the adoption threshold). Paper II §2.1, §3.2, §5.1, §5.4 and App. A
now describe the latent filter as a per-site ensemble Kalman filter in the local-analysis form of Hunt et al.
with perturbed observations; the physics-based comparator (`letkf_phys.py`) is a deterministic LETKF, as
stated. Prop. 1(b) covers both updates, so nothing else changes. paper1 is frozen and keeps the error.
Found while writing the Kolmogorov filter (which uses the same `LETKFLocal2D`).

## 2026-10-11 — Kolmogorov DA: tilings shifted in x only (SK4 smoke)

With the 6-channel α = 0.05 model at 1× (30 cycles, vorticity units): interpolation 0.197, one tiling 0.222,
four tilings in x and y 0.415, four in x only 0.174. The y-shifted tilings' analyses fail (0.656, 0.665) while
their decoder floors are only modestly worse (0.20–0.21 vs 0.165): Kolmogorov flow is invariant to y-shifts
only by whole forcing periods (32 grid points), a half-patch y-shift is a quarter period, and the strip-trained
propagator, which never saw that phase, produces wrong dynamics. Every Kolmogorov run uses x-only offsets
(0, 4, 8, 12 grid points for four tilings). The Kolmogorov section must say why the tilings differ from 2D KS.
Smoke outputs: `artifacts/experiments/kda/k_smoke_c19_*.json`.

## 2026-10-12 — P7, P9, Kolmogorov section, 1D sentence written

- **P7 (J17G, tuned physics filter, full App. D grid):** δ = 0.1: physics 0.328 (0.324–0.333, c8.25 RTPS 0.7,
  no remedy helped) vs latent 0.310 (0.300–0.323) → tie (−5.5%, lower on every seed); δ = 0.2: 0.433
  (0.427–0.440, c8.25 RTPS 0.8 + a,b,g est.) vs 0.341 → latent 1.27×; δ = 0.3: 0.485 (0.470–0.496, same
  setting) vs 0.376 → latent 1.29×. True-model filter (also fully tuned): 0.087, 0.087, 0.096. State-dependent
  one-cycle errors 0.094 / 0.194 / 0.323 (0.15 / 0.31 / 0.49 σ_obs, `j16_model_error_gani*.json`): the tie
  sits just below the σ_obs/6 crossover and the wins above it. The tuned-only column is seed 0 only (marked ∗);
  the 256-member contingency was not needed. Table and Fig. model_error regenerated with three J17G rows
  (squares in panel a).
- **P9 (J5, 220², seed 0):** physics with online estimation 0.421 (∇² 1.25) and 0.446 (advection 1) vs latent
  0.312 (`h9_D10_L220_I1_T4`): ratios 1.35 / 1.43, as at 44² (1.35 / 1.40).
- **Kolmogorov (SK6):** 12-channel per-patch model `k_xs005_c12d10` (noisy strips, k_max 32). The D_KY check
  in `exp_crop_eval.py` compares against stale hard-coded truths (23.26 at 1×, 56.84 at 2×; the reported
  "15.7%" at 2× is against 56.84). The paper uses Paper I's truth table instead: 22.4 / 47.9 / 69.8 against
  means 21.5 / 48.6 and single run 75.2 → +4% / −1% / −7%. DA: 1× 0.115 (0.105–0.124) vs interpolation 0.202,
  one tiling 0.147; 3× 0.113 vs 0.201, one tiling 0.137; spread/error 0.52–0.58 (four tilings), 0.70–0.73 (one).
- **1D:** the per-patch model (`pp1d_c6d10`) assimilates ~2× worse (0.84–0.96 vs 0.43–0.45 at N = 64, GC);
  the current 1D model stays; one sentence in §4.3.

## R8 (2026-10-12) — referee points without new runs (internal_referee_report.md)

R6 (ensemble budget) and R7 (parameter-estimation stall) are not in this entry: their runs are in progress
and the P7 sentence "same observations and ensemble resources" is left for R6.

1. **Training source** (M12/R8): §3.1 "Training on windows" now says the 2D KS windows are 40×40 sites (220²)
   of one 264² simulation and that the assimilation runs to 440², larger than the training simulation; §4.1
   adds that 330² and 440² exceed the training domain and every domain above 220² exceeds the largest window.
2. **"A sixth of the observation error"** (M3c): stated as a result of this configuration (one system, network,
   cycle length, ensemble size, learned model) in §6, abstract, key point 3, PLS, §1 contributions, §11, §12 and
   the Fig. model_error caption, with the range 0.08–0.2 (between the largest physics win/tie and the smallest
   latent win) and the latent filter's dependence on density (0.226 / 0.769 at spacing 1.4 / 5.5). Abstract:
   "confirms this crossover" → "is consistent with". Conclusions no longer present the number as the quantity
   that transfers; the quantity to estimate (the state-dependent one-cycle error relative to σ_obs) does.
   Key point 3: "In our tests the learned filter beat tuned physics filters once their state-dependent model
   error passed a sixth of the observation error" (137 characters).
3. **Learned model's one-cycle error** (M3c): 0.19 field units, blended forecast from the encoded truth at lead
   1 cycle (`tables/forecast.tex`, column "True state", mean over seeds 0–2, `fig16_forecasts.py`
   `true_init_blend`), including the blended decoder floor 0.13–0.14. Replaces "is small" in §5.3; §6 states it
   is larger than the state-dependent error of a physics model it ties with (damping order 6, 0.12), so the
   crossover reflects how each filter handles its model's error, not the forecast errors alone. Not yet put on
   the Fig. model_error axis (wait for R7, which regenerates that figure).
4. **Proposition 1 framing** (M5): new paragraph after the proposition: elementary, restores the property a
   physical-space LETKF with point observations has by construction; its value is what it enables (no
   truncation of H, tilings, every domain size); "exact" refers to the observation operator, the filter stays a
   localized finite-ensemble approximation with a gain linearized at the ensemble mean; holds for grid-point or
   within-patch observations, not for spatially integrated ones. "is exact" → "needs no truncation of it" in the
   abstract, §1 and §12; §11 "makes it exact" → "makes the observation operator exactly local". Prop. 1(b) now
   names site s itself (minor 18). The title is unchanged (it says "exactly local", which is accurate for the
   observation operator); flagged for the authors.
5. **Per-patch decoder cost** (M5): §11 "What locality buys" adds the price (lattice-tied error), the R_dec = 1
   alternative (banded, c·3² colours, neighbours in each per-site analysis), the 1D evidence (per-patch ~2×
   worse) and that the 2D R_dec = 1 model was not trained.
6. **Noisy-training gain vs the decision rule** (M6): 44² three-seed means 0.312 (noisy) vs 0.327 (clean),
   4.5% lower and lower on every seed: below the 10% threshold, a tie by the rule. §7 "it is the better for it"
   → "it loses nothing by it" plus the rule statement; §1 contribution 5 and §11 say "at least as good"
   (4–5%, below the threshold).
7. **Observation network aligned with the lattice** (M9): limitation bullet (stride 8, patch 16, offsets 8; every
   tiling sees the same observation geometry; irregular, non-commensurate and integrated observations untested).
8. **Sparse-network numbers** (minor 1): the 300-cycle runs' JSON "steady" is the mean of their last half
   (cycles 151–300: 0.798 at stride 16, 0.308 at 220²), not the paper's window. `tabE_p2_robustness.py` now
   reports error and calibration over cycles 51–300 for the long runs (0.776 / 1.05 at stride 16; 0.309 / 0.97 at
   220²) and says so in the row header; §8 adds 0.776. 0.769 (100-cycle run, cycles 51–100) is unchanged.
9. **Cubic missing process** (M3e): not mentioned in the paper before. Since §6 states the decision rule was
   pre-registered, the stopped pre-registered case is now reported in one passage of §6 "Design": β = 1.1e-3,
   one-cycle error 0.055 (0.050 state-dependent; `j16_model_error_cubA.json`), physics filter without remedies
   0.190 on seed 0 (`i11_J17A_pde_c8.25_rtps0.7_s0.json`), a physics win consistent with the crossover; the
   state-dependent part never exceeded 0.060 in the β scan (`j16_model_error_cubscan_*.json`), the truth
   stopping being chaotic first; replaced by the anisotropic process.
10. **Notation** (minor 13–17): latent observation operator h → H (site length keeps h, as in Paper I); Kalman
   gain K → G; wavenumber magnitude K → k = |k|; tangent-vector count K → n_v (App. C); comparator localization
   half-width c → L_loc; advection speed c → U; damping order p → γ (text and Fig. model_error labels, regenerated;
   its label overlaps at x ≈ 0.1–0.4 remain for R7, which regenerates the figure); KY density ρ → ρ_KY (ρ stays the taper); RTPS weight α → α_R (α stays the
   Kolmogorov drag); blend weights w_k → ω_k (w stays the propagator radius); App. B uses App. A's U_s
   (observations) and N_s (sites).
11. **Figure/table order** (minor 23): Fig. da_2d moved before Fig. tilings (now Figs. 3, 4) and Table sizes before
   Table members (now Tables 3, 4); the §5.4 long-run sentence cites Sec. 8 instead of Table robustness, so main
   tables are first cited in order 1–8 and figures 1–8 (appendix tables A1, D1 are cited early, as appendix
   floats).
12. **Fig. da_2d(a)** (minor 25): log y-axis (0.05–5) so the failing unlocalized filters (2.26 at 66², 3.40 at
   88²) are in the frame; the 128-member unlocalized runs are added (`h9_K1u_L*_T4_N128`). Paper I's version of
   the figure is unchanged (`P2` switch).
13. **Kolmogorov model in App. A** (minor 8): `tabA_architecture.py` adds `k_xs005_c12d10` as a third column of
   the Paper II table (Paper I's dynamics table drops "Part II" columns, unchanged); App. A text says three
   models; §10 cites the table.
14. **Prior work** (M10, M4): verified and added `pathak2018prl` (PRL 120, 024102), `pathak2018hybrid` (Chaos 28,
   041101) and `penny2022` (JAMES 14, e2021MS002843; Lorenz-96 only, RNN-ETKF/4D-Var, LETKF-like localization
   at 40 variables, baselines use the perfect model, no transfer across sizes — from the arXiv HTML). Cited in
   §2.2 (local learned models; learned surrogate assimilated with domain localization), §2.3 (hybrid physics +
   learned correction, not run here), §2.4 and §11 (limitation). Table 1 gains a Penny et al. row (latent DA:
   hidden state of a recurrent network; localized; no other sizes; compared with the true model only).
Also: App. A "(decisions log, D4)" removed (minor 22). Abstract kept at 250 words (math counted as one word).

## P2-3 cross-paper consistency pass (2026-10-13)

Aligned Paper II with Paper I's rewritten text (Z11–Z14 framing):
- Intro (01): Paper I's 2D KS transfer now "small periodic box or windows … within 3.5% at 44² and 66², and
  Kolmogorov flow needs the window route" (was "trained on windows … 0.6–2%").
- Background (02): Kolmogorov transfer qualified "when trained on windows of a larger simulation".
- Method (03): "Training on windows" no longer says windows are used "not on a small periodic box, so that no training
  sample contains a whole periodic direction" (Paper I now shows the small box works for 2D KS); says Paper I trains on
  either and this paper uses windows, the route that also serves Kolmogorov flow (Paper I, Sec. VI C).
- Kolmogorov (10): Paper I section reference Sec. 4 → Sec. IV C (Paper I numbers sections in Roman).
- System (04) and App. C: third 66² truth run (IC 2, 187.5) added: three runs 185.6/187.5/192.5, mean 188.5 (was two,
  189.0); density 0.0433 (was 0.0434), 44²–66² difference 2.2% (was 2.5%); spread "about ±2%"; the model's 188.5 is
  "inside the range of the three". λ1 truth range at 66² becomes 0.189–0.217, so the model's 0.217 is at the top of
  it (text no longer says it exceeds the truth at both sizes). Paper I's window model numbers updated to its current
  table (+1.6%, −0.5%).
- Regenerated tables via tabC_truth.py and tabV_model_validation.py (PAPER_OUT/PAPER_TABLES → docs/paperII):
  truth_runs (+66² IC 2, +α = 0.1 1× ICs 1–2 at T120), running_ks2d (+66² IC 2), model_validation (λ1 range).
- check_numbers.py: EXPLAINED entry for the 66² density moved from 0.0434 to 0.0433.
- Checked unchanged: companion citations and titles in both refs.bib, shared macros, Prop. 1/2 numbering, Table III
  (truth) reference, Paper I's statements about Paper II.
- Not touched: §6, model_error table/figure (main session).

## 2026-10-13 — Internal referee report 2: minor items and grayscale (fork)

- Item 17: App. A wrong-model list gains the missing process (−(δ/2)(u_x²−u_y²), (a,b,g) estimated from (1,1,1))
  and the stopped cubic case; the R7 floor settings were already added to App. A by the main session. App. E
  lists `run_j17g*.sh`, `run_r7.py`, `run_r7b.py`.
- Item 20: Table 2 caption "two at 66²" → three. Item 21: free-run skill sentence now says the table gives the
  free-run error and skill follows from it. Item 22: Eq. 7 counts D_KY; added that counting n₊ (0.022 per unit
  area) halves the estimate to ~19, saturation (~32) lies between, so the estimate holds to a factor of two.
- Item 23: intro now says the 3.5% is Paper I's models and gives this paper's model (+3.6%, −2.1%).
- Item 24: Sec. 7 lists the five sizes explicitly (66² not run) and "4–5%" → "4–5.3%".
- Item 25: Sec. 10 points to the per-site filter (Sec. da_method) on the tilings (Sec. tilings_method).
- Item 26: forecast caption says λ₁ = 0.165 is the smallest of the three truths and that the mean would make
  every lead in Lyapunov times ~9% larger.
- Item 27: prior-work caption notes the wrong-model comparison is 2D KS only.
- Items 28–29: Fig. da_2d (d) caption explains bar shading; panel (c) title uses d_lat.
- Item 32: `[draft]` kept — it is the AGU submission mode of the class (line numbers, spacing), not a stray
  flag. Placeholders left for the user: Open Research (repository + DOI), Acknowledgments (funding, CRediT).
- Item 33: skipped. 975 overfull boxes appear identically with and without `[draft]`, spread over every
  paragraph (largest 109 pt): a property of the class/line numbering, not the text; AGU typesets the final.
- M10: systems_ks2d and da_snapshot use cividis (monotone lightness, sign readable in gray); Paper I's
  `systems` figure (same script) keeps RdBu_r. 1d_da panel (a): "global + SEC" is now a filled
  down-triangle with a dashed line ("global, none" stays an open up-triangle, solid).
