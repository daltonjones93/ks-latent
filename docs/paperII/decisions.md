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
