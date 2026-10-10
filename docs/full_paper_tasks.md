# Full paper: atomic task list

Sources: `docs/paper1_review.md` (the critical review, 2026-10-08), `docs/paper1_outline.md`,
`docs/paper1/main.tex` and its sections. This file supersedes `docs/paper1_tasks.md` for
everything that remains.

**Decision: build on `docs/paper1/`.** The revtex skeleton, §3–§6, App. B–C, `macros.tex`,
`refs.bib`, the figure scripts and their provenance sidecars are all reusable. Starting over
would cost a day and gain nothing. The paper becomes one manuscript in two parts:

- **Part I — dynamics.** A learned local latent rule, trained once, reproduces the extensive
  Lyapunov dimension at domain sizes it never saw. Window training with a forced halo is what
  makes this work in 2D. Validated against full Lyapunov spectra in three systems.
- **Part II — data assimilation.** A per-patch decoder makes the latent observation operator
  exactly local, so a fixed-size localized ensemble filter keeps its skill at any domain size.

Part I is written from data already in hand. Part II needs the follow-up runs in Block D
first, then is written in Block E. Blocks A–C and Block D are independent: start Block D's
runs in the background on day 1 and write Part I while they run.

Status key: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` dropped. Each task has an ID,
dependencies, an estimated effort, and a done-criterion. Tasks are ordered so that each one
can be picked up alone. "Figure tasks" follow the figure contract in §F.

Conventions that apply to every task:
- Never edit a number into prose by hand; every number comes from an artifact or a script
  output, and figures are produced by `scripts/paper_figures/*.py` with a `.meta.json`.
- Never read a difference below the truth's measured uncertainty as a model effect.
- Pre-register a decision rule before any run whose result changes a claim.
- Do not touch `RegConfig.lambda_z` / `--lambda-z`.
- Report every negative result that is referenced anywhere in the paper.

---

## Block A — Fix Part I's data (no new training, all cheap)

### A1 `[x]` Re-evaluate the 1D model at L = 25, 50, 100, 200, 400 with standardized initial conditions
- Why: `scripts/run_ltransfer_test.py:83,101` encodes the raw solver field (σ = 1.32); §3.6
  claims standardized ICs (review §4.5). The 1D Table III numbers are also record-only, with no
  JSON artifact.
- Do: add `--normalize-ic` (dataset mean/std from `stage1_trajectories_dtsnap1.h5` metadata)
  to `run_ltransfer_test.py`; run the Section-224 checkpoint at all five sizes, 1 IC seed each,
  plus seeds 1–2 of the E4 training seeds at L = 100 and 200; write one JSON per run under
  `artifacts/experiments/e1_model/` with `D_KY`, `lambda1`, `n_positive`, `spectrum`.
- Depends on: nothing. Effort: 1 h code, ~1 h runs (CPU, can run in background).
- Decision rule (pre-registered): accept the existing Table III if every D_KY moves < 1%;
  otherwise Table III and Fig. 3 are regenerated from the new JSONs and §5 is reworded.
- Done when: five JSONs exist, `fig03_1d_transfer.py` reads them instead of hard-coded values.
- **Done 2026-10-08.** `scripts/eval_1d_transfer.py`; 23 JSONs in `artifacts/experiments/e1_model/`
  (3 ICs × 5 sizes, training seeds 1–2, raw-IC checks); summary
  `e1_model/summary_1d_transfer.json` (`scripts/summarize_1d_transfer.py`). Rule verdict:
  **regenerate** (L=50 moved 1.6%). Table III is now generated
  (`docs/paper1/tables/tab03_transfer1d.tex`), Fig. 3 reads the summary. Model − truth:
  −0.7 / −1.1 / −1.8 / −0.3% at L = 50/100/200/400. Findings: the "20% amplitude undershoot" was
  a units error; L = 25 is IC-dependent (one IC chaotic); details in `docs/paper1/decisions.md`.

### A2 `[x]` Float64 control of one model Lyapunov spectrum per system
- Why: model tangents are propagated in float32 (`lyapunov.py:540–541`); §3.6 says double.
- Do: add a `--tangent-dtype float64` option to the model Benettin adapter; run 1D L = 100,
  2D KS 44², Kolmogorov α = 0.05 1× (IC seed 0) in float64; record D_KY, n₊, λ₁ side by side.
- Depends on: nothing. Effort: 1 h code, ≤ 3 h CPU.
- Decision rule: if D_KY changes < 0.5% and n₊ is unchanged, state "float32 propagation,
  double-precision QR; a float64 control changes D_KY by < 0.5%". Otherwise re-run every
  model spectrum in float64 (then budget a day of CPU).
- Done when: a table of (system, dtype, D_KY, n₊, λ₁) is saved to `artifacts/experiments/lyap_dtype_control.json`.
- **Done 2026-10-08.** `tangent_dtype` option + 3 unit tests; `scripts/lyap_dtype_control.py`;
  table in `artifacts/experiments/lyap_dtype_control.json`. Three-IC means: float64 − float32 =
  0.00% / −0.06% (1D L=100/200), +0.30% (2D KS 44²), −2.2% (Kolmogorov α=0.05 1×, IC range 25%).
  **Rule deviation, recorded:** single-run comparisons measure chaotic sampling noise (two
  float64 runs of one configuration differ by 1.8%), so the test used is mean difference vs IC
  spread; no precision bias, no float64 re-run. Side finding: Kolmogorov 1× per-IC model
  scatter is sampling noise (review §5 item 6 withdrawn). See `docs/paper1/decisions.md`.

### A3 `[x]` Finish the two paused α = 0.1, 3× truth seeds
- Why: the 3× point is the only Kolmogorov number quoted below 5%; the single 40-unit truth
  is biased +10% on average (App. C). Checkpoints are at t = 64/140 in
  `artifacts/experiments/crop/truth_a0p1_x3_seed{1,2}_T120.ckpt.npz`.
- Do: resume both with `run_truth_kolmogorov.py` (same T = 120 running-average protocol).
- Depends on: nothing. Effort: ≈ 2.5 h CPU each, parallel. Run at normal priority.
- Decision rule: Table II's 3× entry becomes the 3-seed mean with its spread; no model is
  re-selected on this basis.
- Done when: `truth_a0p1_x3_seed{1,2}_T120.json` exist and Table II is regenerated.

### A4 `[x]` Two more 2D-KS truth seeds at 44² and one at 66²
- Why: both 2D-KS truths are single 40-unit runs; §6.4 currently reads a 4.6% density
  difference as possibly real.
- Do: `exp_g1`-style runs with new IC seeds, 150 directions at 44² (seeds 1, 2) and 300 at
  66² (seed 1), with running D_KY recorded at t = 10, 20, 30, 40.
- Depends on: nothing. Effort: 44²: ~1–2 h each; 66²: ~4–8 h. Background, one at a time.
- Decision rule: model is "consistent with the truth" if inside the seed spread; report the
  spread as the error bar in Fig. 6(a,c).
- Done when: three new JSONs exist; `fig06_scaling.py` draws error bars from them.
- **Done 2026-10-08.** `scripts/run_truth_ks2d.py` (same protocol + checkpoint + running D_KY);
  `artifacts/experiments/truth_ks2d/`. 44²: 81.77 / 80.66 / 83.60 (mean 82.01); 66²: 192.50 /
  185.60 (mean 189.05). Density 0.0424 → 0.0434 (+2.5%, inside the run spread; the "4.6% rise"
  was single-run noise). Running estimates still move 3–5% over the last 20 units. All truth
  consumers now read run JSONs via `scripts/paper_figures/_truths.py` (Table II, Table IV,
  Figs. 5, 6). §4.2, §4.4, §6.1–6.4 updated. Model consistent: 2.0% / 0.6% at 44²/66².

### A5 `[x]` Lyapunov spectra overlay (density vs i/L) for 1D, model and truth
- Why: shows the whole spectral density transfers, not only D_KY; connects to
  Carretero-González et al. 1999.
- Do: from A1's JSONs (model spectra at L = 100, 200, 400) and `e1_groundtruth_lyapunov_L*.json`
  (truth spectra), plot λ_i against i/L; one panel; report where model and truth diverge
  (collective / strongly negative modes).
- Depends on: A1. Effort: 1 h.
- Done when: `fig04_1d_spectra.pdf` exists (becomes Fig. 3(c) or Fig. 4).
- **Done 2026-10-08.** `scripts/paper_figures/fig04_1d_spectra.py` → `fig04_1d_spectra.pdf`
  (now Fig. 4 in §5) + `artifacts/experiments/e1_model/spectra_overlay_1d.json`. Truth spectra
  extended to the model's 0.48·L directions (`e1_groundtruth_lyapunov_L{25,50,100,200,400}_fullspec*.json`).
  Positive part: mean |Δλ| 0.002–0.003; positive-exponent density 0.12–0.125 (model) vs
  0.12–0.135 (truth). The model departs only in strongly contracting directions (i/L > 0.22),
  which do not enter D_KY. Extra truth realizations also give every 1D size ≥ 2 truth runs.

### A6 `[x]` Decoded energy spectra and reconstruction MSE table (the AROMA bar)
- Why: AROMA reports KS as a failure case for local neural fields; our reconstruction MSE is
  10⁻² yet D_KY is right. Show the decoded spectra.
- Do: for the 1D model (L = 100, 200, 400), `p7b_ks2d` (44², 66², 88²) and `k_xs01_v5` (1–3×):
  time-averaged energy spectrum of free rollouts vs truth; interior reconstruction MSE and
  relative spectral error per band. Write `artifacts/experiments/spectra/*.json`.
- Depends on: nothing. Effort: 2 h code, 1 h runs.
- Done when: a table for App. D and `fig07_rollouts.pdf` input data exist.
- **Done 2026-10-08.** `scripts/eval_spectra_recon.py` (1000-snapshot truth, free rollout,
  reconstruction; band energies) → `artifacts/experiments/spectra/{ks1d_x1,2,4; ks2d_x1,1.5,2;
  kolm_x1,2,3}.json`; App. D table `docs/paper1/tables/tabD_spectra.tex`
  (`scripts/paper_figures/tabD_spectra.py`). 1D: recon error 0.01% of variance, rollout bands
  within ~10%. Kolmogorov: recon 1.3%; beyond-box energy 0.3–0.6% (model) vs 0.07–0.14% (truth).
  **2D KS: recon 2.6–2.9%; the per-patch decoder moves ~1.5% of energy to |k| ≥ 1.4 (patch
  seams, review H6).** *Corrected by A9:* the single-run "large-scale build-up at 66²/88²" (51–55%
  at |k| < 0.35, std +11/19%) does not survive a second IC; only the seam excess is robust.

### A7 `[x]` Learned-stencil interaction-range numbers into an artifact
- Why: §3.3 is being rewritten around the measured interaction range (review §4.2d in /Users/daltonjones/Documents/latent_DA/docs/paper1_review.md); the
  stencil R² table lives only in `docs/RESULTS.md`.
- Do: re-run `scripts/fit_local_stencil_pde.py` on the Section-224 latent for w ∈ {0,…,8};
  save `artifacts/experiments/stencil_r2_1d.json`.
- Depends on: nothing. Effort: 1 h.
- Done when: JSON exists; used by §3.3 text and an appendix figure.
- **Done 2026-10-08.** `fit_local_stencil_pde.py --out-json`; `artifacts/experiments/
  stencil_r2_1d_deg{1,2}.json`. Quadratic stencil val R² 0.152/0.578/0.816/0.882/0.900/0.904/
  0.904/0.904/0.903 at w = 0…8 sites (h = 6.25): saturates at w = 4 (25 units), 88% at w = 3
  (19 units). Linear stencil ≤ 0.025 at every width (closure needs the nonlinearity).
  Reproduces the record's table.

### A8 `[x]` Record the exact training configuration of every model in the paper
- Why: App. A does not exist; `k_xs01_v5` has `w_rank=0.02` (inert), the 1D model has
  `w_spatial=0.01` signed, ε defaults differ by model.
- Do: script `scripts/paper_figures/tabA_architecture.py` that reads each checkpoint's config
  and writes `docs/paper1/tables/tabA_architecture.tex` (one row per model: patch, c, hidden,
  reaches, w, n_ℓ, d_m, steps, k_mid/k_max, noise, w_var, w_logdet, ε, w_rank, w_spatial,
  crop sizes, halo, dataset, trajectories).
- Depends on: nothing. Effort: 2 h.
- Done when: the table compiles and matches `.meta.json`/checkpoint configs.
- **Done 2026-10-08.** `scripts/paper_figures/tabA_architecture.py` → `docs/paper1/tables/
  tabA_architecture.tex` (5 models × 18 settings) + full resolved configs in
  `artifacts/experiments/model_configs.json`; new `appendix/A_architecture.tex` included in
  `main.tex`, compiles. 2D values read from checkpoints and crop-training logs; 1D training
  values from its launch script + code defaults (checkpoints store architecture only).
  Footnotes: 1D `w_spatial` 0.01 signed; `k_xs01_v5` `w_rank` 0.02 (inert).

### A9 `[x]` Free-rollout statistics diagnostic for 2D KS
- Why: it is the truth-free diagnostic the paper proposes in §3.6 and uses for model selection
  (review objection 2); it exists for Kolmogorov only.
- Do: port `exp_rollout_stats_kolmogorov.py` to 2D KS; run `p7b_ks2d` at 44²/66²/88².
- Depends on: nothing. Effort: 1 h code, minutes to run.
- Done when: `artifacts/experiments/rollout_stats/p7b_ks2d_*.json` exist.
- **Done 2026-10-08.** `scripts/exp_rollout_stats_ks2d.py` → `artifacts/experiments/rollout_stats/
  p7b_ks2d_ks2d_n1000.json` (44²/66²/88² × IC seeds 0, 1; model vs same-IC truth vs training
  data). Model and truth agree within IC scatter on std and large-scale share at every size;
  the seam band (|k| ≥ 1.4) is 1.7–2.4% of energy vs 0.5–0.8% (truth), in every run. For 2D KS a
  training-data reference is valid only for small-scale bands (the 264² domain's large-scale
  share is 77%, a size effect); it flags the seam excess. 100-unit windows give ±10–20%
  scatter in large-scale statistics, so model selection by these statistics needs ≥ 2 ICs.

### A10 `[x]` Choose the main-text α = 0.1 model by a truth-free rule
- Why: `k_xs01_v5` was picked for its 3× truth error after seven variants were scored on the
  truth; v7 has the lower validation loss (review objection 2 and §11.2).
- Do: write the rule first (recommended: lowest final validation rollout loss among variants
  whose free-rollout statistics are within 15% of the training data), apply it to v4/v5/v7,
  record the choice and the losing variants' numbers in App. D.
- Depends on: A9 (only for the rollout-statistics part; Kolmogorov statistics already exist).
  Effort: 1 h.
- Done when: `docs/paper1/decisions.md` records the rule, the date, and the chosen model.

- **Done 2026-10-08.** Rule pre-registered in `docs/paper1/decisions.md` before evaluation
  (eligibility: free-rollout vorticity std and forcing-band share within 15% of training data at
  1–3×, 2 ICs; selection: lowest k=8 normalized forecast error on the held-out validation
  trajectory, model on the full 32π×2π domain). `scripts/select_alpha01_model.py`; results in
  `artifacts/experiments/a10_selection/` (`decision.json`). All six eligible; **selected
  `k_xs01_v2`** (k8 error 0.0191 vs 0.0196–0.0204; ≥ 4.6 paired SE ahead). Truth errors
  0.9 / 5.9 / 7.6% at 1/2/3× (v5 was 8.8 / 9.9 / 2.3%). Main-text model is read from
  `decision.json` by `_style.MAIN_A01` (Figs. 5–6, Tables IV and A); §6.3 and App. A describe
  the rule. The 3× truth will change when A3 lands; no re-selection.

---

## Block B — Write Part I (text)

Each task is one section or one subsection; each ends with a compile.

### B1 `[x]` Rewrite §3.3 (locality) around the interaction range
- Use review §9 "§3.3" wording; cite Kaneko 1986 / Deissler & Kaneko 1987 for v_*; quote the
  measured v_* = 1.26 and the stencil saturation at 3 sites (A7); drop or demote Eq. (loc_bound).
- Depends on: A7. Effort: 2 h.

### B2 `[x]` Fix §3.2 Remark and Prop. 1 statement
- Remark: cite Livi 1986, Isola 1990, Carretero-González 1999 (Chaos); "finite-size corrections
  are set by the model's reach". Prop. 1: add "shifts by whole patches" and the P ≥ 2r+1 proviso
  to the statement.
- Depends on: nothing. Effort: 1 h.

### B3 `[x]` Fix Lemma 3's surrounding text and add the noise-penalty remark
- Quantify k_max Δt λ₁ per system; add Bishop (1995) expansion as a remark; tie P9b to it.
- Depends on: nothing. Effort: 2 h.

### B4 `[x]` Fix §3.6 (evaluation) wording
- Float32 propagation + float64 QR (A2 result); standardized ICs (A1 result); model window
  lengths (2000 steps = 400 time units for Kolmogorov); "directions chosen so that D_KY is
  bracketed, enforced by an exception".
- Depends on: A1, A2. Effort: 1 h.

### B5 `[x]` Update §3.6 canonical-recipe paragraph and point to App. A
- Mention w_rank (inert) and w_spatial; state that k_max differs by model and why.
- Depends on: A8, A10. Effort: 1 h.
- **Done 2026-10-08.** w_spatial mentioned for 1D; w_rank not mentioned in §3 because the
  A10-selected model does not use it (it stays in the §8 ablation table). "Fixed before the
  final round" removed (contradicted by A10).

### B6 `[x]` Fix §4 (systems)
- Re convention: define your own, reduce to Vinograd et al. at α = 0; Lyapunov-time wording
  (100+ for L ≥ 50; 6–16 units; 2–6 windows); 2D-KS truth uncertainty paragraph (A4);
  Edson et al. for L = 22; Table II regenerated from A3/A4.
- Depends on: A3, A4. Effort: 2 h.
- **Done 2026-10-08 except A3.** Rewritten: rows D_KY / D_KY per box / n₊ per box (n₊ as a
  third row, not a secondary axis); ±15% band on single-run truths; open = truth, filled =
  model as in Fig. 3; system hues and drag markers from `_style.py`; A4 seeds included.
  **Awaiting A3**: rerun the script; the α = 0.1 3× band becomes a three-run error bar; recheck
  the caption sentence "models track the truth at α = 0.1 and 0.05".
- **Done 2026-10-08 except the A3-dependent part** (see "Awaiting A3" below): Lyapunov-time
  wording, Re convention (Vinograd et al.'s definition applied to our forcing amplitude, equal
  to theirs at α=0), Edson et al. for L=22 and the L≥80 law, 2D-KS uncertainty, convention
  paragraph rewritten (multi-run means, matched-IC scoring removes truth noise only).

### B7 `[x]` Fix §5 (1D)
- L = 25 paragraph (marginal truth, lattice narrower than kernel); add the w_spatial sentence;
  Table III from A1 JSONs; add the spectra overlay paragraph (A5).
- Depends on: A1, A5. Effort: 2 h.

### B8 `[x]` Fix §6 (2D)
- "93–100%"; "at least as accurate, at 30% higher reconstruction error"; name `k_xs005_noreg`;
  §6.4 "consistent with extensive scaling within truth uncertainty"; §6.5 scope to P2 at
  α = 0.05, 3×; Table IV caption convention; model-selection sentence (A10); add the per-IC
  amplification result (review §5.6) as a short paragraph.
- Depends on: A4, A10. Effort: 3 h.
- **Done 2026-10-08 except the A3-dependent numbers** (listed under "Awaiting A3"). Per-IC
  "amplification" paragraph **not added**: withdrawn by A2 (sampling noise); §4.4 now states that
  matched-IC scoring removes truth noise only. Added a "Beyond the dimension" paragraph (A6/A9:
  Kolmogorov band energies, 2D KS amplitude within IC scatter, patch seams at |k| ≥ 1.4).
  §6.5 scoped to P2 at α=0.05, 3×. References `app:tables` (created in B12).

### B9 `[x]` Write §8 "What does not work" (Part I version, ~2 pages)
- Attention collapse; MSE damping × ε table (v1–v7 with validation loss and rollout-statistics
  columns); spectral regularizer A/B; training-set size (`k_xs005_v2`); P9 spurious regimes and
  the rollout-statistics diagnostic. Keep the DA negative results for Part II.
- Depends on: A9, A10. Effort: 4 h.
- **Done 2026-10-08.** `sections/08_failure_modes.tex` (in `main.tex`): attention (1D collapse;
  2D windowed-attention encoder, raw-IC caveat), coupling/data size under window training,
  short-rollout regimes (P9 → P9b), MSE damping × ε with generated Table
  (`tables/tab08_alpha01_ablation.tex`, `scripts/paper_figures/tab08_alpha01_ablation.py`;
  paper v1–v6 = record k_xs01, v2, v3, v4, v5, v7). All errors recomputed against current truths.
  Record correction: k_xs005_noreg 3× is 7.2%, not 5.1%. The 1D attention-collapse statement
  rests on the record (no single artifact lists every attention run).

### B10 `[x]` Write §2 Background and related work (~2.5 pages)
- Use review §6 verbatim as the source list; include the prior-work feature table (local /
  zero-shot L / Lyapunov validation / window training / localized latent DA) with Pathak 2018,
  Constante-Amores 2024, Bar-Sinai 2019, Kochkov 2021, Stachenfeld 2022, de Villeroché 2026,
  Özalp & Magri 2025, Carretero-González 1999, ROAD-EnKF, LAE-EnKF, Latent-EnSF.
- Depends on: B13 (bib). Effort: 1 day.
- **Done 2026-10-08.** `sections/02_background.tex` (in `main.tex`): extensive chaos and
  Lyapunov analysis (incl. Carretero-González 1999 as the lattice precedent), learned surrogates
  (global latents, single-size Lyapunov validation, local models, larger-domain deployment,
  limited-area models, rollout training), latent/localized DA, novelty statement; Table
  `tab:prior_work` with the verified characterizations from the literature pass.

### B11 `[x]` Write §1 Introduction (~2 pages)
- Follow outline §2.1 with the corrected positioning sentences (review §3.2 item 1) and the
  contributions list from review §2.3.
- Depends on: B10, and Part II's headline numbers (E-block) for the DA bullet; write the DA
  bullet last.
- Effort: 1 day.
- **Done 2026-10-08 except the DA bullet's numbers** (`TODO(Part II, block E)` in
  `sections/01_introduction.tex`; see "Awaiting Part II" below). Also awaits A3 (the
  "1–9% in Kolmogorov flow" range).

- **Done 2026-10-08**: DA bullet written from §7 (90% of skill kept 44²→440², linear memory, unlocalized fails beyond the native size, 1D ensemble need doubles with L, localized 16 at every size). Also fixed the stale box-trained range (19–38% → 15–38% in three of four cases, after A3).
### B12 `[x]` Write App. A (architecture tables), App. D (full evaluation tables)
- App. A from A8; App. D generated by a script from `artifacts/experiments/crop/*.json`
  (every model × size × IC, with the truth convention and the stale-field warning).
- Depends on: A8. Effort: 3 h.
- **Done 2026-10-08.** App. A from A8 (+ selection rule, A10). App. D:
  `appendix/D_tables.tex` with generated tables `tables/tabD_{ks2d,kolm01,kolm005,kolm0025}.tex`
  (`scripts/paper_figures/tabD_evaluations.py`: 28 models + box-trained baselines, current truth
  convention) and `tabD_spectra.tex` (A6); per-IC CSV `artifacts/experiments/appendix_D_evaluations.csv`
  (201 rows, incl. the stale JSON truth fields, flagged). Wide tables now use `\widetabular`
  (macros.tex) and table bodies `\tabinput`; no overfull tables remain.

### B13 `[x]` Bibliography verification pass and `refs.bib` fixes
- Apply review §6 "bibliographic corrections"; add every new reference; resolve every [v] and
  [standard] tag by checking DOI/pages; record the search date and terms in App. F.
- Depends on: nothing. Effort: half a day.
- **Done 2026-10-08.** Every entry status-tagged; all verified/checked except Livi 1986
  (partial) and Long 2015 (standard). ~50 entries added for §1/§2/Part II. Search log in
  `docs/paper1/decisions.md` (B13 entry); App. F (G5) should copy it.

### B14 `[x]` Write App. C additions
- 2D-KS uncertainty paragraph; model-side Benettin settings; float64 control (A2); running
  D_KY for the new truth seeds (A3, A4).
- Depends on: A2, A3, A4. Effort: 2 h.
- **Done 2026-10-08 except the A3 rows.** `scripts/paper_figures/tabC_truth.py` generates the
  truth-run list, the Kolmogorov running-D_KY table and a new 2D KS running table; App. C gains
  "Independent realizations" (thread-count trajectories), "Convergence and spread of the 2D KS
  truths" and "Model spectra and their precision" (A2 table). A3 rows appear on rerun; the
  deviation statistics sentence is on the "Awaiting A3" list.

---

### Awaiting A3 (α = 0.1, 3× truth seeds 1–2) — **done 2026-10-08**
All items below applied after the runs landed (3-run mean 53.47; selected model's 3× error
7.6% → 2.6%); details in `docs/paper1/decisions.md`, A3 entry. B11 stays `[~]` for Part II only.
Automatic on rerun (read the run JSONs through `_truths.py`): Table II
(`tab02_tab03.py`), Table IV (`tab04_transfer2d.py`), Figs. 5 and 6, and (Block C) Fig. 10
(`fig10_failure.py`), App. figures `figC1_truth_convergence.py` and `figC2_appendix_panels.py`.
Manual text edits:
- [x] §4.4 convention paragraph: remove the `TODO(A3)` comment; α=0.1 3× becomes a three-run
  mean; update the "remaining sizes rest on a single run" sentence.
- [x] §6.3 "Selecting a model without the truth": the selected model's 3× error (7.6% now) and
  v5's (2.3% now) change; the selection itself is not revisited.
- [x] §6.3 "Results": the "1–9% at α=0.1 and α=0.05" range.
- [x] §6.4 scaling paragraph: α=0.1 truth density at 3× (18.8 per box now) and the sentence
  "the truth's rise at 3× rests on a single 40-unit run".
- [x] App. C: running D_KY table/figure gains two α=0.1 3× rows; 40- vs 120-unit deviation
  statistics (−17% to +23%, mean +10%) are recomputed with them (B14).
- [x] decisions.md: A3 entry, and whether the +10% bias of 40-unit truths holds at 3×.
- [x] §1 contributions bullet "1–9% in Kolmogorov flow at the two stronger drags".
- [x] Block C captions/text: Fig. 6 caption ("track the truth at α = 0.1 and 0.05"); §8.4
  "raising it from 16 to 32 at α = 0.1 lowers it by 3–10 points" and Fig. 10 caption "no row is
  lightest in every column"; App. C Fig. `fig:truth_convergence` caption deviation range and
  count (TODO(A3) comment). Then mark C6, C7, C8 `[x]`.
- [x] §8.4 text: the α=0.1 3× numbers (v1 15.4%, v2 7.6%, v3/v4 12–14%, v5 2.3%, v6 9.8%) and
  the claim that a different variant is best at each size; Table VIII regenerates itself
  (`tab08_alpha01_ablation.py`).

---

### Awaiting Part II (Block D runs, then Block E)
- [x] §1 contributions, last bullet: replace `TODO(Part II, block E)` with the headline DA
  numbers (fixed-N skill from 44² to 440², unlocalized degradation, cost/memory).
- [x] §1 roadmap sentence and §9 cross-references once §7 exists (`sec:da_results`). (done 2026-10-08)

---

## Block C — Part I figures

Follow §F (figure contract). One script per figure in `scripts/paper_figures/`, each writing a
PDF and a `.meta.json`. Keep `_style.py` as the single source of palette and sizes.

### C1 `[x]` Fig. 1 — concept schematic (TikZ or matplotlib)
- Four panels: (a) global vector vs local field; (b) transfer = same weights, more sites;
  (c) crop with margin, forced halo, free interior; (d) per-patch decoder → block-diagonal
  observation Jacobian → per-site analysis. Hand-drawn geometry, no data.
- Effort: half a day. Done when it reads without the caption.
- **Done 2026-10-08.** `scripts/paper_figures/fig01_concept.py` → `figures/fig01_concept.pdf`
  (matplotlib, 2×2). In §1 as `fig:concept`, referenced from the paragraph that introduces the
  latent field.

### C2 `[x]` Fig. 2 — systems gallery
- 1D KS space–time (L = 100, 250 time units); 2D KS snapshot on the 264² training domain with
  one 40×40-site crop outlined and its halo/interior shaded; Kolmogorov vorticity on the
  16×2π strip with one 48-site crop outlined. Shared colormap for the two 2D panels; colorbar
  once. Pull fields from the HDF5 datasets.
- Effort: 3 h.
- **Done 2026-10-08.** `fig02_systems.py`; one shared diverging colormap for all three panels
  (fields standardized; the 2D datasets are stored standardized). Kolmogorov panel at
  α = 0.05. In §4 as `fig:systems`.

### C3 `[x]` Fig. 3 — 1D transfer (regenerate from A1)
- (a) D_KY vs L with truth and model, 0.226L line, L = 25 marked as "outside validity"
  rather than "collapses"; (b) D_KY/L with the three training seeds, y-range stated in the
  caption; (c) spectra overlay from A5. Move the L = 25 annotation off the marker.
- Depends on: A1, A5. Effort: 2 h.
- **Done 2026-10-08.** Annotation moved to the empty lower right ("outside validity (4 sites <
  7-site kernel)"); legend order truth / seed 0 / seeds 1–2; caption states the (b) y-range and
  no longer says the L = 25 model "loses all chaos" (A1: IC-dependent). Panel (c) stays a
  separate figure (Fig. 4, `fig04_1d_spectra.py`, A5), which reads better at full width.

### C4 `[x]` Fig. 4 — rollouts and spectra at transfer sizes
- Top row: truth vs transferred model space–time at L = 400 (1D). Middle: 2D KS 88² truth
  vs model snapshots. Bottom: time-averaged energy spectra, model vs truth, for 1D (100, 200,
  400) and 2D KS (44², 66², 88²), from A6.
- Depends on: A6. Effort: 4 h.
- **Done 2026-10-08** as `fig07_rollouts_spectra.py` (`fig:rollouts`, in §6, cited from §5 and
  §6): 2×3 grid, rows 1D L = 400 and 2D KS 88², columns PDE / model / spectra. Fields from
  `eval_spectra_recon.py --fields-out` (new flag; writes `spectra/fields/*.npz`, not the JSON).
  Found and fixed on the way: the Kolmogorov A6 spectra used v5, not the A10-selected v2 —
  re-run with v2, §6 numbers and Table `tab:spectra` updated (decisions.md, Block C).

### C5 `[x]` Fig. 5 — box vs window (revise)
- Linear error axis with a broken scale for the 93–100% bars, or two panels (0–40% and
  90–100%); state the floor only if a log axis is kept; per-IC dots retained.
- Effort: 2 h.
- **Done 2026-10-08.** Linear axis broken between 42% and 88%; caption rewritten (no floor).
  Regenerates itself after A3 (α = 0.1, 3× bars).

### C6 `[x]` Fig. 6 — scaling (revise)
- Add seed error bars from A3/A4; shaded ±15% band for single-run truths; mark n₊ on a
  secondary axis or in a small table inset; unify marker semantics with Fig. 3.
- Depends on: A3, A4. Effort: 2 h.

### C7 `[x]` Fig. 10 — failure modes
- (a) v1–v7 ablation as a size × variant error heat-map with validation loss annotated;
  (b) free-rollout variance vs lead time for k_max = 12/16/32 (illustrates Lemma 3);
  (c) P9 energy-share drift vs truth (from `rollout_stats/`).
- Depends on: A9. Effort: 4 h.
- **Done 2026-10-08 except A3.** `fig10_failure.py` (`fig:failure`, in §8). (a) heat map with
  k_max and val. column; (b) **changed**: the variance-vs-lead diagnostic was run
  (`scripts/exp_forecast_variance.py`) and shows no amplitude contraction for any k_max (all
  within 2.3% to 64 steps), so (b) shows signed D_KY error of three k_max pairs instead; the
  variance result is stated in §8.4 (decisions.md, Block C); (c) P9 large-scale energy share
  at 2×, PDE vs k_max 12 vs k_max 32. **Awaiting A3**: rerun; recheck the §8.4 "3–10 points"
  range for v2 → v1 and the caption claim "no row is lightest in every column".

### C8 `[x]` App. figures
- Truth running-D_KY convergence (exists; add the new seeds); per-IC scatter at 1× (models vs
  matched truths); C19 reach/channel ablation bars; stencil R² vs half-width (A7).
- Depends on: A3, A4, A7. Effort: 3 h.
- **Done 2026-10-08 except A3.** `figC1_truth_convergence.py` rewritten (house style; one
  panel per system and size, reads every run that recorded a running D_KY, so the α = 0.1 3×
  panel appears by itself; 2D KS panels added; now a `figure*`). `figC2_appendix_panels.py`
  (`fig:appendix_panels`, App. D): (a) per-IC scatter at α = 0.05 1×, (b) C19 reach/channel
  ablation, (c) stencil R² vs half-width. **Awaiting A3**: rerun figC1; fix the caption's
  "−17% to +23%, four of five above" (TODO(A3) comment in `C_ground_truth.tex`).

### C9 `[x]` Fig. 11 — rollout filmstrips and PDE–model correlation (2D systems) — merged into Fig. 7
- Fig. 7 already has the 1D L = 400 space–time and one 2D KS 88² snapshot; this adds what it
  lacks: (a) 2D KS 88² PDE vs model at four times from a common encoded IC, (b) Kolmogorov
  α = 0.1 at 3× (Lx = 6π) PDE vs model at first and last stored time, (c) pattern correlation
  of model vs PDE against time (dotted line = 1/λ₁ for 2D KS). Message: pointwise agreement is
  lost after a few Lyapunov times, statistics persist; the 2D KS model panels also show faint
  site-block seams worth a sentence in §6.
- **Script written and run 2026-10-08**: `scripts/paper_figures/fig11_rollout_filmstrip.py` →
  `figures/fig11_rollout_filmstrip.pdf` on the real fields (`spectra/fields/{ks2d_x2,kolm_x3}.npz`,
  regenerated at 16:25–16:27 by another job with a dense time axis). Result: 2D KS correlation
  leaves 1 near 1/λ₁ and falls to ~0.1 by t≈20; Kolmogorov plateaus near 0.5 (forced pattern).
- **TODO:** (1) check the model panels' faint site-block seams (visible in both systems) and
  say what they are in the caption; (2) add to §6 as `fig:rollout_filmstrip` with a caption and
  one paragraph (the Kolmogorov 0.5 plateau needs a sentence); (3) optionally add a DA companion
  (truth / ensemble mean / error at L = 440 with observation sites) once D0 lands.
- Effort: 1 h plus run time.

- **Merged into C4's Fig. 7, 2026-10-08** (decided with the user: a separate filmstrip would
  mostly repeat Fig. 7(d,e)). Fig. 7 (`fig07_rollouts_spectra.py`) gained a Kolmogorov α = 0.1,
  3× row (PDE / model / kinetic-energy spectra at 1×–3×, v2) and a full-width panel of pattern
  correlation vs t·λ₁ for all three systems (Kolmogorov as anomaly correlation: the PDE's
  forcing-locked mean y-profile removed). Correlation falls below 0.5 after 2.85 (1D), 2.75
  (2D KS) and 2.34 (Kolmogorov) Lyapunov times; late level −0.01 / 0.14 / 0.18
  (`spectra/fields/pattern_correlation_summary.json`). New fields: `spectra/fields/kolm_x3.npz`
  (400 steps, stride 2); `ks2d_x2.npz` re-dumped at stride 5 (identical to the old stride-50
  dump at t = 35). `fig11_rollout_filmstrip.py` is retired (left on disk, not used).
---

## Block D — Part II follow-up runs (start on day 1, in the background)

> **2026-10-09:** the shifted-tiling filter (H9 → Block I) supersedes the single-tiling 2D
> accuracy numbers. D2, D5–D10 are rerun under Block I (table at the top of Block I); D1, D3,
> D4 and the dense/unlocalized points stand.

All DA runs: 100 cycles, steady state = cycles 51–100, report per-cycle arrays, seed 0 unless
stated. Queue them in one orchestration script with logging to a file (never `/dev/null`).
Large domains (≥ 220²) one at a time; they reach ~20 GB.

**Launched 2026-10-08 16:11 (while A3 finishes):** every Block D run that needs no code
change, in `scripts/run_blockD_nocode.sh` (log `artifacts/logs/blockD_nocode.log`, outputs
`artifacts/experiments/da2d/da2d_D*`). Small lane, normal priority: D2 2D half (strides 4/16/32
at 44²), D9 at 88², D5 (9 runs). Large lane, `nice 10`, one domain at a time: D9 at 220², D8
(4 runs), D7 seed 1 at 220² and 330². Still need code first: D1, D2 1D half, D3, D4, D6
(then D10); D5's local row should be repeated with the D4 variant if D4 is adopted.

### D0 `[x]` Ingest the finished queue
- `da2d_C_L440_local_N64_s0.json` and the `M_*` memory re-measurements have landed. Run
  `scripts/summarize_da2d.py`, record the L = 440 steady-state skill and the clean peak RSS per
  size in `docs/paper1/decisions.md`; profile `LETKFLocal2D.analyze` at d = 1536 if filter-only
  RSS is not within 30% of linear in d (review §8.11).
- Effort: 1 h (+ 2 h if a profile is needed).

- **Done 2026-10-08**: L = 440 steady-state skill 3.90; local filter RSS linear in d (no profile needed); decisions.md.
### D1 `[x]` 1D DA sweeps at fixed observation spacing
- Why: `run_da_pff_ltransfer.py:76` scales the stride with L (review §4.2a).
- Do: add `--obs-spacing-units` (default 3.125) that sets the stride in grid points from Δx;
  re-run E2 (PFF) and E7 (EnKF), none vs GC, N ∈ {8,16,32,64,128,256}, at L = 200 and 400;
  write `e2b_*.csv`, `e7b_*.csv`.
- Decision rule (pre-registered): C5's 1D half stands if, at N = 64, unlocalized skill < 1 and
  GC/none ≥ 1.5 at both sizes; if unlocalized skill ≥ 2, restate C5 as a fixed-N ratio claim
  and move the obs-density result (D2) to the front.
- Effort: 1 h code; 48 points × ~2 min ≈ 2 h unattended.

- **Done 2026-10-08**: fixed-spacing sweeps, PFF and EnKF, L = 200/400. Rule: GC/none ≥ 1.5 holds everywhere; unlocalized < 1 at N = 64 holds at L = 400, not at L = 200 (1.1–1.3, the rule's uncovered middle case). Unlocalized threshold tracks n₊ + 1. decisions.md.
### D2 `[x]` Observation-spacing sweep, 1D and 2D
- Why: the spacing threshold (DA fails above the pattern wavelength) is a result and the
  diagnosis of the 1D confound (review §5.2).
- Do: 1D at L = 200, GC, N = 64, spacing ∈ {1.6, 3.1, 6.25, 12.5}; 2D at 44², local, N = 64,
  spacing ∈ {1.4, 2.75, 5.5, 11} (the 2.75/5.5/11 probes exist in `da2d/probe/`, re-run under
  the current code for consistency).
- Effort: 8 runs, ~2 h.

- **Done 2026-10-08**: both halves; DA fails once the observation spacing reaches the site length (1D 6.25, 2D 5.5). decisions.md.
### D3 `[x]` SEC: PSD-projected and serial variants
- Do: (i) `sec_localize_covariance_psd` — eigenvalue clipping after the elementwise correction;
  (ii) serial SEC: apply the correction to each observation's regression coefficients in a
  serial perturbed-observation EnKF (new small class, same `analyze` interface). Unit tests:
  both reduce to the plain filter when the table is the identity; serial SEC matches the
  Kalman posterior for a linear-Gaussian case at large N.
- Then re-run E3 (SEC vs GC on the local latent) at L = 200 with D1's spacing.
- Decision rule: report whichever of (i)/(ii) runs at every N; "GC beats SEC at the transfer
  size" or "SEC beats GC" is stated from that; the indefiniteness result is reported as a
  property of the matrix form.
- Effort: half a day code; 12 points × 2 min.

- **Done 2026-10-08**: SEC-PSD and serial SEC EnKF both run at every N; GC beats both below N = 256. Raw matrix SEC fails 5/6. decisions.md.
### D4 `[x]` Deterministic per-site ensemble transform (ETKF) variant of `LETKFLocal2D`
- Why: the per-site filter at N = 16 scores 1.8 vs 3.8 for dense+GC; the perturbed-observation
  update is the likely cause (review §5.9).
- Do: add `method="letkf_local2d_etkf"` (Hunt et al. 2007 transform in ensemble space per
  site, same R-localization, same compact Jacobian); unit test against the Kalman posterior;
  run N ∈ {8,16,32,64,128} at 44² and N = 64 at 88².
- Decision rule: adopt the ETKF variant as "the per-site filter" if its N = 16 skill ≥ 3 and
  N = 64 skill is within 10% of the perturbed-observation version; otherwise report both.
- Effort: half a day code; 6 runs, ~3 h.

- **Done 2026-10-08**: ETKF variant coded and tested; N = 64 within 2% of perturbed-obs, N = 16 = 1.87 < 3 → not adopted, report both (decisions.md).
### D5 `[x]` Inflation sweep at 44²
- Do: inflation ∈ {1.0, 1.05, 1.1, 1.2} for dense-none, dense-GC, local (D4 variant if
  adopted), N = 64.
- Decision rule: if unlocalized N = 64 skill at 44² changes by > 30%, re-run the 66²/88²
  unlocalized points with the best inflation; otherwise state "inflation neutral" with numbers.
- Effort: 12 runs ≈ 5 h unattended.

- **Done 2026-10-08**: inflation neutral (≤ 1% unlocalized, ≤ 9% any filter); no reruns. Repeat local row if D4 adopted.
### D6 `[x]` Physical-space RMSE
- Do: in `cycling.py`, optionally decode the analysis mean and the free run each cycle and
  record physical RMSE against `u_truth` (and the observed-site RMSE separately); re-run one
  point per size (local N = 64 at 44², 88², 220², 330²) with the flag on.
- Decision rule: report both metrics; if they differ by > 20% explain (decoder error floor).
- Effort: 2 h code; ~2.5 h runs.

- **Code done 2026-10-08; 44², 88² done** (`[~]`): physical skill 6.24 / 4.78 vs latent 4.23 / 4.00; decoder floor ≈ 40% of analysis error variance. 220²/330² queued behind the large lane (`run_blockD_nocode.sh d6`).
- **Done 2026-10-09**: 220², 330² landed. Physical analysis RMSE 1.05 / 0.99 / 1.01 / 1.02 (44²–330²), at observed points 0.79–0.82, decoder floor 0.65–0.69; physical skill 4.5–6.2 vs latent 3.8–4.2. §7.2 updated.
### D7 `[x]` Seed repeats for the headline 2D points
- Do: seed 1 for local N = 64 at 220² and 330²; seed 1 for dense-none N = 64 at 44² and 88²
  (exists for some); record run-to-run spread.
- Effort: ~3 h unattended (large domains one at a time).

- **Done 2026-10-09**: seed 1 at 220² 3.90 (3.84), 330² 3.88 (3.86); per-site seed spread 0.02–0.15 at every size 44²–330² (§7.2, App. E).
### D8 `[x]` Ensemble-size sweep at a transfer size
- Do: local filter N ∈ {8,16,32,128} at 220² (N = 64 exists).
- Effort: 4 runs ≈ 2 h.

- **Done 2026-10-08**: per-site filter at 220², N = 8/16/32/64/128: 1.06 / 1.80 / 3.28 / 3.84 / 3.64 — the same curve as at 44²–88² (32 members slightly lower). §7.2 and `tab07_2d_ensemble` updated.
### D9 `[x]` Model-noise sweep completion
- Do: noise ∈ {0.02, 0.04, 0.16} at 44² exists; add 0.04 at 88² and 220² to show the
  skill/calibration trade-off is size-independent.
- Effort: 2 runs ≈ 1.5 h.

- **Done 2026-10-08**: noise 0.04 gives +13–14% skill, −0.03 calibration at 44², 88², 220² alike.
### D10 `[x]` Seam diagnostic (H6)
- Do: on saved analyses from D6, ratio of mean |∇u| across patch edges to inside patches, for
  analysis / forecast / truth; one number per size.
- Effort: 2 h.

- **44², 88² done 2026-10-08** (`[~]`): edge/interior gradient ratio truth 1.0, recon 3.6, analysis 3.8, free 3.0. Rerun `scripts/exp_d10_seams.py` when D6 220²/330² land.
- **Done 2026-10-09**: edge/interior gradient ratio at 44², 88², 220², 330²: truth 1.00, reconstruction 3.6, analysis 3.8, free run 3.0 — size-independent (§7.2).
### D13 `[~]` Accuracy: match observation interpolation → continued as Block H (2026-10-09)
- Finding: the 2D latent filter (1.05 physical RMSE at 44²) is worse than cubic interpolation of the
  observations (0.83). Details and every test in decisions.md ("Accuracy study").
- Done: R / model-noise sweep (best 0.97); hybrid physical analysis (0.975); variational recentring
  (0.956 with localized ensemble B); low-pass of the decoded analysis matches interpolation (0.81–0.82
  vs 0.83). Gradient-matching AE loss: trades seams for resolved-scale error (worse total).
- Next: (1) evaluate `p7c12_ks2d` (12-channel per-patch AE, Stage A done) with
  `scripts/exp_recon_bands.py p7b_ks2d p7c12_ks2d`; retrain `p7c12g3_ks2d` if wanted; (2) if the
  floor drops, Stage B (~2 h) and the 44² DA test; (3) else decoder reach 1 (filter generalization);
  (4) whatever the outcome: add the interpolation baseline to §7.2 / Fig. 9 and reframe Part II's
  claims accordingly (pending user decision).

### D11 `[→I11]` (optional) Physical-space baseline filter on the true PDE at 44²
- Do: per-site EnKF on the 128² PDE state with a 64-member ensemble of true integrations,
  same observations; one point.
- Effort: 1–3 days code; ~1 h run. Do only if time remains after D1–D10; otherwise state as a
  limitation.

### D12 `[→I19]` (optional) Kolmogorov DA at 1× and 3×
- Do: Kolmogorov truth generation and strip geometry in the 2D DA script; local N = 64, one
  point per size.
- Effort: 1–2 days code; hours of runs. Paper-2 material if dropped.

---

## Block E — Write Part II (after Block D)

### E1 `[x]` Decide Part II's headline table and figure from D0–D9
- Write `docs/paper1/decisions.md` entry: which filter variant is "the per-site filter",
  which inflation/noise setting is canonical, and the final steady-state skill table
  (sizes 44²–440², N = 64, local / dense+GC / dense-none) with seed spreads.
- Effort: 2 h.

- **Done**: headline = Table `tab:da_2d_sizes` + Fig. 9 (fixed 64-member per-site filter, 44² → 440²).
### E2 `[x]` Write §7.1 — exact locality (Prop. 4, colouring, per-site filter definition)
- Name the filter correctly; define skill, spread, calibration, steady state, and both RMSE
  metrics in-line for a dynamics audience; cite Takeda & Miyoshi 2026 and Bocquet & Carrassi
  2017 for the ensemble-size bound.
- Effort: 3 h.

- **Done 2026-10-08**: `sections/07_data_assimilation.tex` §7.1 (protocol, filter names, exactness, metrics incl. physical RMSE, n₊ bound with measured n₊); §7 now in `main.tex` with TODO(E3/E4/E5) placeholders. 'Appendix~E' is cited by name until E6 exists.
### E3 `[x]` Write §7.2 — fixed ensemble across 56× area (2D headline)
- Table: skill vs size for the three filters; text on n₊ vs N; saturation by N = 16 for
  dense+GC vs the heuristic; small-N behaviour (D4); under-dispersion and the noise trade-off.
- Depends on: E1. Effort: 4 h.

- **Drafted 2026-10-08** (`[~]`): §7.2 with generated `tab07_2d_sizes`/`tab07_2d_ensemble` (`tab07_2d_da.py`); headline = fixed per-site N = 64 from 44² to 440² (skill 4.23 → 3.90). E1 decision taken in passing: headline table = Table `tab:da_2d_sizes`. TODO(D7, D8) in the text: seed-1 at 220²/330² and N = 16/32/128 at 220² fill in on rerun; revisit wording then.
- **Done 2026-10-09**: all TODO(D7/D6/D10) placeholders resolved.
### E4 `[x]` Write §7.3 — observation density and the 1D results
- D2's spacing threshold; D1's fixed-spacing sweeps (PFF and EnKF agree); GC vs none ratio;
  the fixed-size negative result (SEC on the global latent beats GC on the local field at
  L = 100, with the two-architecture caveat and the 75%-of-ring taper note); D3's SEC result
  stated as a property of the matrix form.
- Depends on: D1–D3. Effort: 4 h.

- **Done 2026-10-08**: §7.3 with `tab07_1d_da` (`tab07_1d_da.py`): spacing threshold (D2), fixed-spacing 1D sweeps (D1), L = 100 SEC caveats, D3 as a property of the matrix form.
### E5 `[x]` Write §7.4 — cost and memory, honestly
- Measured CPU and clean peak RSS per size (D0); the dense filter's d² accounting; where the
  per-site filter overtakes the dense one; what remains unexplained.
- Depends on: D0. Effort: 2 h.

- **Drafted 2026-10-08** (`[~]`): §7.4 with `tab_da_cost` (`tab_da_cost.py`). TODO: clean per-site memory at 330² from D6/D7 when they land; per-site loop not profiled (stated).
- **Done 2026-10-09**: 330² memory in text and `tab_da_cost`.
### E6 `[x]` Write App. E — DA details
- Filter equations, observation networks in physical units at every size, taper construction,
  SEC implementations, inflation sweep, per-cycle curves, run-to-run spread.
- Depends on: D1–D9. Effort: 4 h.

- **Drafted 2026-10-08** (`[~]`): `appendix/E_da_details.tex` (networks table, filter equations incl. ETKF and both SEC forms, inflation table, seed spread, transient). TODO(D7) seed spread at 220²/330²; per-cycle curves figure belongs to Block F.
- **Done 2026-10-09**: seed spread at every size.
### E7 `[x]` Write §9 Discussion and limitations
- What transfer requires; the interaction-range design rule; transfer error as a physics probe
  (one paragraph, paper 2); limitations list from review §3.2 item 9; outlook (RBC, LAM, CM1).
- Depends on: B-block, E2–E5. Effort: half a day.

- **Done 2026-10-08**: `sections/09_discussion.tex` (what transfer requires, four conditions; interaction range as the design length incl. the narrow-reach P6 failures; transfer error as a physics probe; what locality buys in DA; limitations list; outlook without citations, TODO(G-pass) to add verified ones). Included in `main.tex`; no undefined references remain.
### E8 `[x]` Write §10 Conclusions, the abstract, the Chaos lead paragraph, and the title
- Use review §2.3 as the starting text; every number in the abstract traced to a table.
- Depends on: everything above. Effort: 3 h.

---

## Block F — Part II figures

- **Done 2026-10-08**: title = review §2.3's recommendation ("…zero-shot domain transfer validated by Lyapunov spectra, and exactly local ensemble data assimilation"); three Key Points (128–132 characters); final abstract (every number traced to Table III, Table IV, `tab:da_2d_sizes`, Fig. 5 — noted in a comment); Plain Language Summary (serves as the Chaos lead paragraph at submission); new `sections/10_conclusions.tex`. Checked while writing: "the weakest drag departs most from extensive" was false (α = 0.05 rises most) — removed from §10 and softened in §9; "selected model among the most accurate" made exact (lowest mean error of the six α = 0.1 variants, 3.1%). Number check rerun: 0 unmatched.
### F1 `[x]` Fig. 8 — 1D DA (regenerate from D1–D3)
- (a) L = 100 trained size: global none / global+SEC / local none / local+GC; (b) L = 200 and
  (c) L = 400 at fixed spacing: none vs GC, PFF and EnKF; SEC shown as its own curve where it
  runs, hatched region where it does not; observation spacing in every panel title.
- Depends on: D1, D3. Effort: 3 h.

- **Done 2026-10-08**: `fig08_1d_da.py` rewritten from the D1/D3 JSONs (steady-state skill, fixed spacing), panel (a) from the L = 100 sweeps (whole-run skill, stated); truth n₊+1 marked. In §7.3 as `fig:da_1d`. Found while doing it: §7.1/§7.3 had quoted the model's n₊ (25, 49–50) as the truth's (26–27, 50–51) — corrected; decisions.md.
### F2 `[x]` Fig. 9 — 2D DA headline
- (a) decoder Jacobian sparsity image with the 6-column block marked; (b) skill vs domain
  area for local / dense+GC / dense-none at N = 64, 44²→440², with n₊ and D_KY on a top axis;
  (c) skill vs N at 44², 88², 220² for the local filter; (d) per-cycle skill and spread/RMSE
  at 330² (100 cycles); (e) CPU time and clean peak RSS vs d, dense vs local, with the dense
  d² accounting line; (f) observation-spacing sweep (D2) with the pattern wavelength marked.
- Depends on: D0–D8. Effort: 1 day.

- **Built 2026-10-08** (`[~]`): `fig09_2d_da.py` → `fig:da_2d` in §7.2, six panels: (a) dense reverse-mode Jacobian at 44² (`scripts/exp_f2_jacobian_sparsity.py`; off-block entries exactly 0); (b) skill vs area, per-site / dense+GC / unlocalized 64 and 128; (c) skill vs N at 44², 88², 220² + dense GC; (d) per-cycle error, spread, free run at 330²; (e) filter memory vs d with ∝d and ∝d² guides; (f) skill vs observation spacing / site length, 2D and 1D. CPU time stays in Table `tab:da_cost` (one axis per panel). Deviations from the spec: no n₊ top axis (n₊ in the caption), per-cycle panel shows errors and spread (same units) rather than skill and calibration.
- **TODO — finalize after the 330² runs (~20:30)**: rerun `tab_da_cost.py`, then `fig09_2d_da.py` (panel (e) gains the clean 330² per-site memory point from D6/D7); look at the figure again; then mark `[x]`.
- **Done 2026-10-09**: final with the 330² memory point (4.30 GB; sub-linear from 220²: ×1.28 for ×2.25 area); labels in (e), (f) fixed. Text/abstract/intro/conclusions changed from "memory linear" to "no faster than the area".
### F3 `[x]` Fig. 11 — DA snapshot at a transfer size
- Truth, free forecast, analysis (decoded mean), and analysis error at 330², one cycle in
  steady state; shared colormap; observation sites overlaid on the truth panel.
- Depends on: D6. Effort: 3 h.

- **Done 2026-10-09**: `fig11_da_snapshot.py` → `fig:da_snapshot` in §7.2: truth / free run / analysis / error at 330², cycle 78, full domain and a 55×55 window with observation sites and patch edges. The free-run panel shows why the skill reference is weak (mean of decorrelated members keeps only the large scales); §7.1 now says skill is measured against the free ensemble mean.
### F4 `[x]` Fig. 12 — Jacobian timing and colouring (small, could be an appendix figure)
- Re-run `exp_t1_jacobian_timing.py` at the corrected deployment; reverse mode vs coloured
  vs d.
- Effort: 1 h.

- **Done 2026-10-08**: new `scripts/exp_f4_jacobian_timing.py` (the old T1 script sized the domain from the crop grid and scaled the stride with L) at the deployment grid, stride 8, 44²–440²; `fig12_jacobian_timing.py` → App. E (`fig:jacobian_timing`, `app:da_jacobian`) and one sentence in §7.4. Coloured = reverse mode exactly (diff 0) up to 132²; coloured 1.3 → 49 ms (slope 0.81 in d), reverse 7 → 415 ms by 132² (slope 1.84). 4 threads, run while the large DA lane was busy, so absolute times are indicative.
### F5 `[ ]` Animations for the supplement / talk
- 2D KS 88² truth vs model rollout; DA cycling at 330² (truth / analysis / error). Frames
  from `docs/anim_frames/` tooling; MP4 + a 6-frame filmstrip figure for the supplement.
- Effort: half a day.

---

## Block G — Polish and submission

### G1 `[x]` Figure consistency pass
- Every figure: same fonts as body (serif, 8.5 pt), one palette (`_style.py`), truth always in
  neutral ink, model in one hue per system, identical marker semantics across Figs. 3/6/9,
  axis labels with units, panel letters, colorbars once per figure, no text smaller than 7 pt,
  vector PDF, column widths exact (3.37 / 6.75 in). Check each in grayscale and at 50% zoom.
- Effort: half a day.

- **Done 2026-10-08**: figures are now designed at the layout's text width (`_style.COL2` = 5.5 in for the AGU draft class, 397.5 pt; switch to 3.37/7.0 for REVTeX at submission) and included at natural size, centred, so every font prints at its designed size; all explicit font sizes below 7 pt raised to 7 (contract minimum). Then every figure re-rendered and inspected: titles shortened where they collided with panel letters (details moved to captions/axis labels), legends moved off data or replaced by direct labels (Fig. 9 b, e, f), Fig. 8's legend deduplicated and markers made distinct for grayscale (none ▽/△, GC ○/□), Fig. 4/6/12 legends moved below the axes, Fig. 10 validation column spaced. Checked in grayscale at half size (Figs. 6, 8, 10). Palette, truth-in-ink, open-truth/filled-model and one-hue-per-system semantics were already in place from Block C. `fig11_rollout_filmstrip.pdf` is not used by the paper.
### G2 `[x]` Number-provenance pass
- Script `scripts/paper_figures/check_numbers.py`: extract every number in the `.tex` sources
  that also appears in a table/figure script and assert equality; list record-only numbers.
- Effort: half a day.

- **Done 2026-10-08**: `scripts/paper_figures/check_numbers.py` → `docs/paper1/number_check.md`. 524 decimals in the text: 410 printed in generated tables, 109 equal to a value in the 527 result files at the printed precision, 5 unmatched — all checked by hand and explained in the script (a literature value, two axis limits, a D9 skill, a memory ratio). Limitation stated in the report: a match is consistency, not proof of source; the check catches numbers that match nothing. Integers are not checked. Rerun after every text or result change (it is fast).
### G3 `[x]` Internal-consistency pass against the review
- Walk review §4.2, §4.5, §9, §10 line by line and tick each item; delete `paper1_outline.md`'s
  stale numbers or mark the file as superseded.
- Effort: 3 h.

- **Done 2026-10-08**: `docs/paper1/review_checklist.md` (every item of review §4.2, §4.5, §9, §10 with status). Fixed in this pass: forced-halo failure test now covers η = R−1 and R−2 (text matched); Prop. 4(b) statement and proof cover both per-site updates; App. C "chosen to exceed" and "earlier values"; Table IV headers; `paper1_outline.md` and `paper1_tasks.md` marked superseded. Deferred to G4: output-projection bias explanation, table abbreviations.
### G4 `[x]` Writing pass
- One idea per sentence; define every symbol at first use; no claim without a number or a
  citation; every negative result stated once; cut §8 to ≤ 2 pages; cut repeated caveats.
- Effort: 1 day.

- **Done 2026-10-08**: §8 rewritten from 1306 to 1145 words incl. captions (~850 body, ≤ 2 pages), every number stated once, selection-rule robustness kept; symbols defined at first use ($n_+$, $\kmid$/$\kmax$ in §3; EnKF, PFF, GC in §7.1; "Kolm." expanded in the Table II caption for all tables); the two items deferred from G3 (no-bias explanation in §3.2; abbreviation) done; duplicate "finite-time estimate is random" sentence in §3.6 merged, free-rollout screen now says training data or truth; §5/§6 tightened (ambiguous "either", per-patch paragraph). Number check: 0 unmatched. Sections written today (§7, §9, §10) were drafted to the one-idea-per-sentence rule and were spot-checked, not rewritten.
### G5 `[x]` App. F — reproducibility
- Code layout, seeds, device policy, test list (equivariance, forced halo, block-diagonal
  Jacobian, colouring, Kalman checks, geometry), compute budget, data/code availability
  statement, commit SHA, environment lock.
- Effort: 3 h.

- **Done 2026-10-08**: `appendix/F_reproducibility.tex` (code and provenance, numerics/devices/seeds incl. FFT-thread realizations, versions, the 132 tests the claims rest on — run and passing, compute costs from run files), included in `main.tex`; Open Research text points to it. TODO(G8) in the appendix: tagged release commit and archive DOI — the results were produced from a dirty working tree on top of 2cd9da0, so a release commit is required before submission.
### G6 `[~]` Compile in `reprint` (two-column) layout and fix layout issues
- Floats, table widths, equation breaks, overfull boxes; `latexmk -pdf` clean with no `??`.
- Effort: 3 h.

- **Done in part 2026-10-08** (user request): `main.tex` now follows AIP's REVTeX 4.2 template (`aipsamp.tex`): `aip,cha,reprint` two-column layout, short running title, provisional abstract and Chaos lead paragraph (TODO(E8)), acknowledgments, Author Declarations (conflict of interest) and Data Availability (TODO(G5)) before the appendices, `hidelinks`. All overfull boxes fixed in `reprint` (wide tables → `table*`, long displays split); `preprint` also compiles (two small §2 overflows). Left for G6: float placement at the end of the appendices (pages of tables with white space).
- **Superseded 2026-10-08** (user request): layout switched to the AGU class used by van Leeuwen, arXiv:2601.01795 (`docs/paper1/PJ2019.cls` = agujournal2019, `draft` option, apacite author-year citations via `natbibapa`, Key Points + Abstract + Plain Language Summary, Open Research Section and Acknowledgments before the references). Peter Jan van Leeuwen added as second author (CSU affiliation from his paper). A `ruledtabular` shim in `macros.tex` keeps the REVTeX tables; floats use `[htbp]`. The class's ragged-right draft mode emits ~1000 harmless overfull warnings (the reference paper emits 510 the same way). TODO: Dalton Jones's affiliation; acknowledgments; both authors to confirm the conflict-of-interest sentence; target journal (Chaos would need REVTeX again).
- **2026-10-08** (user): both authors at CSU (corresponding: daltjone@colostate.edu); no acknowledgments (section removed); no conflict of interest (stated); **target journal Chaos** — the AGU-style layout is kept for drafting at the user's request; switching back to REVTeX (`aip,cha`) is needed for the submission package (G8).
### G7 `[ ]` External read
- Send the PDF to one dynamics reader and one DA reader; collect objections; address each in
  `docs/paper1/decisions.md`.
- Effort: a week of calendar time.

### G8 `[ ]` Submission package
- Chaos cover letter; suggested referees; arXiv version with the same figures; supplement
  (animations, App. D tables as CSV); Zenodo/GitHub release of code and the artifact JSONs
  used by every figure.
- Effort: half a day.

---

## Block H — Part II accuracy: make the latent filter beat the no-model baselines

Started 2026-10-09 from the finding (D13) that the 2D latent filter's analysis (physical RMSE
1.0 at 44²) is worse than interpolating the observations (spline 0.83; statistical
interpolation with the climatological covariance 0.80). Part II's scaling result stands; this
block is about accuracy. All numbers: 2D KS, 44², per-site filter, N = 64, observation spacing
2.75 (stride 8, 4 per site), observation noise 0.62 field units (0.1 standardized), 100 cycles,
steady state = cycles 51–100, physical RMSE in field units (truth std 4.0). Scripts:
`scripts/exp_obs_interp_baseline.py`, `exp_hybrid_da.py --diag --save-prior`, `exp_oi_bound.py`,
`exp_recon_bands.py`, `run_gc_sweep.sh`, `run_acc2.sh`; results in `artifacts/experiments/da2d/`
(`da2d_ACC*`, `da2d_GC_*`, `da2d_B2_*`, `hybrid_*`, `interp_baseline_L44.json`).

### H0 Observations (what the diagnostics established)

**O1 — Error budget by wavenumber band** (error variance, field units²; observation Nyquist and
patch wavenumber both ≈ 1.14):

| |k| band | 6-ch decoder floor | 12-ch decoder floor | spline interp. | current analysis |
|---|---|---|---|---|
| [0, 0.6) | 0.002 | 0.001 | 0.095 | 0.214 |
| [0.6, 1.14) | 0.037 | 0.002 | 0.259 | 0.340 |
| [1.14, 1.6) | 0.136 | 0.011 | 0.307 | 0.198 |
| ≥ 1.6 | 0.267 | 0.063 | 0.028 | 0.295 |
| total → RMSE | 0.44 → 0.67 | 0.08 → 0.28 | 0.69 → 0.83 | 1.05 → 1.02 |

The latent represents the observed scales almost perfectly (6-ch floor 0.04 below the Nyquist)
yet the analysis carries 0.55 there: the filter, not the representation, fails at observed scales.
Above the Nyquist the analysis is already at the 6-ch floor, and there the latent beats
interpolation in [1.14, 1.6) — the only band where a model can beat the observations.

**O2 — Ensemble span.** Only 20% of the prior error variance lies in the span of the 63
decoded ensemble perturbations (global least squares, `hybrid_diag_mn0.08`). The perturbation
spectrum is large-scale (band shares 0.66 / 0.31 / 0.07 / 0.11), so the directions are not the
wrong shape; there are too few of them. One-step bound √(1−0.20) × prior 1.09 = 0.98 = the
observed analysis (0.975–1.06).

**O3 — Localization.** At 44² the Gaspari–Cohn half-width 3 (support radius 6 sites on an 8 × 8
lattice) hardly localizes; the per-site filter there is in effect a global 64-member filter.
Tightening to half-width 1–2 gains only 2% (0.967 → 0.947). At one observation per site (stride
16) the filter gives 2.83 against 3.00 for interpolation, i.e. the sparse regime is where a
model-based filter starts to win, but only just.

**O4 — Fitting the observations harder makes the total worse.** Error at the observed points is
0.81 > noise 0.62, so the prior is over-weighted; but inflation 1.5–3, smooth physical-space model
noise (std 0.1–0.2, length 4–8 grid points) and the physical-space hybrid analysis with inflation
all lower the error at the observed points (to 0.60) while raising the total (to 1.24). An
optimal linear analysis with the prior error's own (isotropic) covariance reaches only 0.92;
with the ensemble perturbation spectrum 0.945. The prior error has a near-white component
(RMSE 0.31 above |k| = 1.14, the decoder seams) that every observation sees and that any
covariance-based update spreads as if it were signal, and its large-scale part is
non-stationary (an isotropic OI on it over-corrects). Changing the filter alone cannot beat
interpolation on this decoder.

**O5 — The decoder is the lever.** A per-patch autoencoder with 12 channels per site (same
training, `p7c12_ks2d`, Stage A) has floor 0.28 vs 0.67, and its seam band is 0.063 vs 0.267.
Seam-penalty losses on the 6-channel decoder (gradient matching, `--w-grad`) only trade seam
error for resolved-scale error (0.78–0.85 total): 6 channels per 16 × 16 patch is a capacity
limit.

**O6 — Model noise / R.** Model noise 0.08 → 0.02 improves the analysis 1.05 → 0.97 (every size);
assuming a larger observation error in the filter (0.15–0.2) fixes calibration (0.35 → 0.8) but not
accuracy. The free-run skill reference in the paper (≈ 4) is a weak reference; the paper must say
so (done in §7.1) and show the no-model baselines.

### H0b The argument (short)

1. *Subspace bound.* A square-root/transform ensemble update moves the mean within
   span(X_f): e_a = (I − P)e_f + P e_f', so ‖e_a‖ ≥ ‖(I − P)e_f‖ = √(1−f)‖e_f‖ with f the span
   fraction. Measured f = 0.20 ⇒ ≥ 0.98. Localization raises the effective rank, but only if the
   error is local; the seam component is everywhere.
2. *Representation error as observation error.* Innovation d = ε + H e_L + H e_S with e_S the
   decoder's small-scale error (white-like, σ_S = 0.31): R_eff = r² + σ_S² (0.62² → 0.69²), and,
   worse, any gain K that spreads innovations also spreads e_S. Hence O4. With the 12-channel
   decoder σ_S ≈ 0.07 and the term vanishes.
3. *Where a model can win.* With m = 256 observations and ≈ 200 resolved modes, interpolation is
   already near-optimal at |k| < 1.14. A model-based filter can only beat it through (a) a prior
   that is better than climatology at the observed scales and (b) the scales the observations do
   not resolve, where only the model carries information. (b) requires a decoder whose floor is
   below the truth's energy there (12-ch: 0.011 vs truth ≈ 0.31 in [1.14, 1.6)); (a) requires an
   ensemble that spans the prior error (f ≫ 0.2) — hybrid static-plus-ensemble covariances are the
   standard remedy.
4. *Prediction to test (H1).* Same filter, 12-channel model: analysis ≈ √(0.1 + 0.011 + 0.063) ≈
   0.4–0.45 if the observed-scale error reaches interpolation's level (0.35 var), 0.6–0.7 if it
   stays where it is. Either beats 0.80. If it does not, the prior error at observed scales is the
   remaining problem (→ H5).

### H1 `[~]` Twelve-channel 2D KS model: train, screen, transfer, assimilate
- Stage B for `p7c12_ks2d` running since 2026-10-09 (nice 10; ~2–2.5 h). Then: free-rollout
  screen; `exp_crop_eval` at 44², 66², 88² (2 ICs); `exp_recon_bands.py`; DA at 44² stride 8 and
  16 (`run_acc2.sh`-style, `--phys-rmse`), scored against `interp_baseline_L44.json` and the
  climatological OI baseline (`exp_oi_bound.py`).
- Pre-registered decision rule: adopt the 12-channel model as the canonical 2D KS model if (i) its
  transfer errors at 44², 66² are within the truth spread (≤ 2.5%), (ii) its free-rollout
  statistics pass the §3.6 screen, and (iii) its analysis RMSE at stride 8 is ≤ 0.80 (the no-model
  bar). If (iii) fails but (i)–(ii) hold, keep it and go to H5. If (i) fails, report it in §8 and
  keep the 6-channel model.
- If adopted: rerun Block D's 2D runs (A/B/C sizes, D2, D5–D9; ~1 day, large domains one at a
  time) and Part I's 2D KS numbers (Tables IV, D; Figs. 5–7).

### H2 `[→I18]` No-model baselines into the paper
- Add spline and climatological-OI baselines (`exp_obs_interp_baseline.py`, `exp_oi_bound.py`
  last line) to §7.2 and Fig. 9(b)/(f) at every size and observation density; state in §7.1 that
  skill against the free run is not an accuracy measure. Do this whatever H1 gives.

### H3 `[→I10]` "Why" figure for §7: error anatomy
- Panels: (a) error variance by band for decoder floor, interpolation, analysis (O1); (b) prior
  error spectrum vs ensemble perturbation spectrum and the span fraction per cycle (O2);
  (c) error at observed points vs total error across the batch-2 settings (O4). From existing
  JSONs; half a day. This turns the accuracy limitation into a result.

### H4 `[→I5]` The regime where the filter should win: sparse or noisy observations
- With the adopted model: stride 16 and 32 (interpolation 3.0 and 4.0 — it fails), and
  observation noise 0.1 / 0.2 / 0.4 at stride 8 (interpolation degrades in proportion; a good
  prior should not). 8 runs at 44², then the best point at 220². Expect this to be the headline
  accuracy result; the dense-and-accurate network is interpolation's home turf.

### H5 `[dropped]` Ensemble covariance: raise the span fraction
- (a) Hybrid covariance: static latent B from the training latents (localized) + ensemble B, in
  `var_recenter` (`cov="hybrid"`), the EnVar remedy for rank deficiency; (b) 128 members with the
  12-channel model; (c) ensemble of lagged analysis increments as extra directions. Score by span
  fraction (`exp_hybrid_da.py --diag`) and analysis RMSE. Only after H1.

### H6 `[dropped]` Decoder reach 1 (fallback if H1 fails on the floor)
- Reach-1 decoder keeps a local Jacobian (3 × 3-site blocks, 54 forward passes); generalize
  `LETKFLocal2D` to that block structure; retrain. 1–2 days.

### H7 `[→I11]` Physical-space ensemble filter on the PDE at 44² (was D11)
- Now justified: it gives the ceiling for a filter with a perfect model at this network. A local
  ETKF on the 128² grid with 64 PDE members; `exp_hybrid_da.py`'s per-patch ETKF already does the
  analysis step on a grid, so only the PDE ensemble forecast is new (~half a day). Report next to
  the latent filter and the no-model baselines.

### H9 `[~]` Overlapping tilings: remove the seams with no retraining
- Idea (user, 2026-10-09): encode/decode the field on several tilings shifted by a fraction of a
  patch and blend the decodings, so every point is taken mostly from a tiling where it sits
  mid-patch. No retraining needed: the AE was trained on crops at arbitrary pixel offsets and the
  propagator is translation-equivariant, so shifted encodings are in distribution.
- **Measured (`scripts/exp_overlap_recon.py`, p7b_ks2d, 44², 20 snapshots; error power by band
  |k| < 0.6 / 0.6–1.14 / 1.14–1.6 / > 1.6):**
  | reconstruction | RMSE | bands |
  |---|---|---|
  | 1 tiling (current) | 0.665 | 0.002 0.037 0.136 0.267 |
  | 4 tilings (half-patch), plain average | 0.411 | 0.001 0.024 0.056 0.088 |
  | 4 tilings, Hann partition of unity | **0.340** | 0.005 0.033 0.049 0.029 |
  | 16 tilings (quarter-patch), plain average | 0.311 | 0.001 0.023 0.044 0.029 |
  The decoder floor halves (0.665 → 0.340), and the seam band (> 1.6) drops 9×. The floor is now
  well under interpolation (0.83) and climatological OI (0.80), which removes O1's ceiling with
  the existing model; the 12-channel model (H1) can be blended the same way.
- **DA result, cheap form (a), 2026-10-09 (`scripts/exp_h9_tiled_da.py`, 44², N=64 per tiling,
  stride 8, obs noise 0.62 field units, GC c=1.5 sites, model noise 0.02, steady = cycles 51–100,
  analysis RMSE in field units; truth std 4.0):**
  | | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | current filter (1 tiling) | 0.944 | 0.880 | 0.908 |
  | **4 tilings, independent filters, Hann-blended** | **0.572** | **0.549** | **0.566** |
  | spline interpolation (no model) | 0.829 | 0.813 | 0.819 |
  | decoder floor, 1 tiling / 4 blended | 0.69 / 0.35 | 0.65 / 0.34 | 0.66 / 0.35 |
  Controls (seed 0): 4 ensembles on the SAME tiling, averaged: 0.943; one tiling with N=256:
  0.949, so the gain is from moving the seams, not from 4× the members. Feeding the blended
  analysis back (re-encode on every tiling, `--feedback`) is worse: 0.811 (c=1.5), 0.907 (c=3);
  the projection costs more than it gains, so tilings run independently.
  **The latent filter now beats interpolation by 30–33% and climatological OI (0.80) by ~29%,
  with no retraining, at 4× the cost.** Still to do: (d) the same at 220² (scaling claim must
  survive the blend); (e) sparse network (stride 16, needs offsets 0/8 with a stride-16 network —
  offsets not a multiple of the stride: generalize the observation-site map); (f) the 12-channel
  model (H1) blended.
- Sparse network (stride 16, 2026-10-09, seed 0): single 3.37, blended **1.86**, interpolation 3.01.
  Continued as **Block I**.
- Original plan: (a) **DA, cheap form**: run the existing per-site filter independently on the
  four tilings (four latent ensembles, each exact and local), report the Hann-blended analysis
  in physical space; 4× cost. (b) **Joint form, "DA on the intersections"**: state = the four
  tilings' latents, observation operator = the blended decoder; still exactly local (each
  observation touches ≤ 4 sites, one per tiling), coloured Jacobian with 4× the JVPs; the
  analysis couples the tilings only through the overlaps. Re-encode the blended analysis on each
  tiling at every cycle so the four ensembles stay consistent (a projection; measure its error).
  (c) If (a)/(b) beat 0.80 at stride 8, this goes into §7 as the accuracy fix, with the
  floor table above, and the exactness statement extended to the multi-tiling operator.

### H8 `[→I18]` Rewrite Part II's claims after H1–H4
- Abstract, Key Points, §1 bullet, §7.2, §10: scaling and exactness stay; accuracy stated
  against the no-model baselines with the regime where the filter wins named explicitly.

---

> **Superseded by Block I (2026-10-09):** H2→I18, H3→I10, H4→I5, H7→I11, H8→I18; H1 continues as
> I12. Dropped: H5 (the blend lifts the accuracy limit H5 was aimed at; calibration is I6) and H6
> (fallback for a floor problem the blend has removed).

## Block I — Shifted-tiling filter: make it Part II's method (from H9, 2026-10-09)

**Where we are.** Running the existing per-site latent filter on K = 4 tilings shifted by half a
patch, and blending the K analyses in physical space with Hann weights, cuts the 2D KS analysis
error at 44² from 0.88–0.94 to 0.55–0.57 (3 seeds). That beats spline interpolation (0.81–0.83)
and climatological OI (0.80) by ~30%, with no retraining and at K× the cost (H9 table). Controls
rule out "more members" as the cause: 4 ensembles on one tiling give 0.943, and 1 × 256 members
gives 0.949. On the sparse network (stride 16, one observation per site length, where D2 found
that DA fails) the blend gives 1.86 against interpolation's 3.01 and the single tiling's 3.37
(seed 0, GC c = 1.5 not retuned). Weak points so far:
- the blended ensemble is under-dispersive (spread/RMSE 0.37 at stride 8, 0.22 at stride 16);
- scaling with the blend is untested beyond 44²;
- there is no perfect-model ceiling yet.

Everything in this block uses `scripts/exp_h9_tiled_da.py` (generalized 2026-10-09: offsets
need not be multiples of the observation stride; K ∈ {1, 2, 4, 16}; Hann or uniform weights;
inflation; free run; blended spread; error at observed points; peak RSS). Runs go through
`scripts/run_h9_queue.sh` (log `artifacts/logs/h9_queue.log`), outputs
`artifacts/experiments/da2d/h9_<tag>.json`. Defaults unless stated: N = 64 per tiling, model
noise 0.02, GC half-width c = 1.5 sites, observation noise 0.1 standardized (0.62 field units),
100 cycles, steady state = cycles 51–100, physical RMSE in field units.

### Which Block D runs must be redone with the blended filter

| D task | Rerun? | Why / where |
|---|---|---|
| D1 1D fixed-spacing sweeps | No | 1D uses the full-reach AE (no per-patch seams); tiling does not apply. 1D accuracy vs interpolation is checked separately in I15. |
| D2 observation-spacing sweep (2D) | **Yes** → I5 | The "DA fails at spacing ≥ site length" result is overturned at stride 16 (1.86 vs interpolation 3.01). |
| D3 SEC vs GC | No | Relative comparison of localizers (1D, dense latent); unaffected. |
| D4 per-site ETKF | No | Not adopted; blending is orthogonal to the update form. |
| D5 inflation | **Yes** → I6 | The blended ensemble has its own calibration (0.37); tuning must be redone. |
| D6 physical RMSE at 44²–330² | **Yes** → I1 | This is the headline accuracy-vs-size table. |
| D7 seed repeats | **Yes** → I1 | Seed 1 at 220², 330² with the blend. |
| D8 ensemble-size sweep | **Yes** → I3 | The "required N is size-independent" claim must hold for the method we report. |
| D9 model-noise sweep | **Yes** → I6 | Folded into the calibration retune. |
| D10 seam diagnostic | **Yes** → I9 | Edge/interior gradient ratio of the blended analysis (expect ≈ 1). |
| Dense / unlocalized points (A/B/C sizes) | No | The n₊ > N failure concerns the latent ensemble's rank, not the seams. I1(d) adds one blended unlocalized point at 44² to show blending does not rescue it. |
| Latent-space skill tables | Recompute | Report physical RMSE and the no-model baselines first, with skill against the free run second (it is a weak reference, H0 O6). |

### I0 `[ ]` Pre-registered adoption rule (written before the 220² result, 2026-10-09 06:40)
- Adopt the K = 4 blended filter as "the per-site filter" of Part II if:
  1. its 220² steady-state RMSE is ≤ 0.80 (beats climatological OI);
  2. that value is ≤ 1.15× its 44² value (0.575);
  3. the I1 sizes 88²–330² satisfy (1)–(2) too.
- If (2) fails but (1) holds, report the accuracy gain and the scaling separately (scaling with the single-tiling filter, accuracy with the blend at each size).
- If (1) fails at 220², the blend is a 44² result only. Report it in §7 as such and find out why (I8).

### I1 `[~]` Scaling with the blend (D6/D7 rerun)
- **12-channel Part II runs (RTPS 0.7, stride 8, seed 0):** 88²: K = 4 0.321, K = 1 0.457 (interpolation
  0.808), the same as at 44². **K = 4 across sizes (2026-10-09):** 44² 0.338, 88² 0.321, 220² 0.326,
  330² 0.340, **440² 0.326** (interpolation 0.808–0.824). K = 1: 0.457 / 0.467 / 0.480 at 88² / 220² /
  330². Accuracy is flat over a 100× area: **I0 conditions 1–3 met with the 12-channel model.** The 440²
  wall time includes a ~4 h machine-sleep pause (15:30–19:20), so do not use it for I14 timing.
- (a) K = 4 and K = 1 at 88², 220², 330², seed 0, same settings as 44². 220² was launched
  2026-10-09 06:40 (`run_h9_queue.sh l220`). One large domain at a time.
- (b) Seed 1 at 220², 330², and seeds 1–2 at 88².
- (c) **440², required** (the abstract's 44²→440² claim must be restated with the blended filter or the range shortened). Measured peak RSS is 4.7 GB at 220² (K = 4), so ~19 GB at 440² is expected: feasible, run alone. The single-tiling C run reached ~20 GB, but the h9 script
  stores no history, so first measure peak RSS at 330².
- (d) One unlocalized blended point at 44² (dense filter on each tiling), N = 64.
- **220² done 2026-10-09 (seed 0):** 1 tiling 0.902, 4 tilings blended **0.560**, interpolation 0.820;
  free-run skill 5.07 → 8.17; blended spread/RMSE 0.375. Wall time 293 s (K = 1) and 675 s (K = 4),
  serial in one process, 2.3× rather than 4× because the truth integration is shared. Peak RSS
  4.3 GB (K = 1) and 4.7 GB (K = 4). **I0 conditions 1 and 2 pass at 220²** (0.560 ≤ 0.80, and
  ≤ 1.15 × 0.575); condition 3 (88², 330²) is pending.
- Report at each size: the blended and single RMSE, interpolation and climatological OI at that
  size, free-run skill, calibration, wall time and peak RSS. Apply I0.

### I2 `[~]` Equal-cost comparisons — is the blend an efficient use of members?
- **12-channel, Part II settings (RTPS 0.7), 2026-10-09** (`P2_L44_I2_*`; seeds 0/1/2; spread/RMSE):
  - 1 × 64: 0.483 / 0.459 / 0.458 (0.80–0.84);
  - 2 × 32: 0.395 / 0.376 / 0.373 (0.85–0.90);
  - **4 × 16: 0.366 / 0.344 / 0.337 (0.71–0.77)**, 25% better than 1 × 64 at equal members;
  - 16 × 4: 0.96 / 0.84 / 0.83, collapsed (0.10). Four members per tiling is too few, so the
    split has a floor.
- **44² done 2026-10-09** (3 seeds, each paired on one cached truth; seed 2's spinup is now
  cached too, because an uncached 200-unit spinup is not bitwise reproducible across runs, as
  chaos amplifies floating-point differences):
  | total 64 members | seed 0 | seed 1 | seed 2 | mean |
  |---|---|---|---|---|
  | 1 tiling × 64 | 0.951 | 0.879 | 0.859 | 0.896 |
  | 2 tilings × 32 | 0.712 | 0.670 | 0.657 | 0.680 |
  | 4 tilings × 16 | **0.632** | **0.613** | **0.596** | **0.614** |
  | interpolation | 0.829 | 0.813 | 0.813 | |
  At a fixed total of 64 members, 4 × 16 is 31% better than 1 × 64, and better than interpolation
  on every seed. At a fixed 256: 4 × 64 0.575 against 1 × 256 0.949. **Splitting the members
  across tilings is a better use of them than one larger ensemble.**
- **220² done 2026-10-09** (seed 0, total 64): 1 × 64 0.902, 2 × 32 0.681, 4 × 16 **0.623**
  (interpolation 0.820). The same ordering and the same ratios as at 44² (0.951 / 0.712 / 0.632). Wall
  time 293 / 308 / 406 s, peak RSS 4.2–4.3 GB: at a fixed total of members, the blend costs ~1.4×.
- At a fixed total of 64 members: 1 × 64, 2 × 32 (K = 2, diagonal offset) and 4 × 16.
- At a fixed total of 256 members: 1 × 256 (done, 0.949) and 4 × 64 (0.575).
- At equal wall time: the K tilings are independent, so they run in parallel (K processes); report serial and parallel wall time.
- Run at 44² and 220².
- This decides whether the paper can say "efficient": if 4 × 16 beats 1 × 64, the blend is better per member as well as per run.

### I3 `[ ]` Ensemble-size sweep with the blend (D8 rerun)
- N per tiling ∈ {8, 16, 32, 64, 128}, K = 4, at 44² and 220².
- Claim to test: the N needed for a given RMSE is the same at both sizes, as for the single tiling.
- Plot the RMSE against the total member count, with the single-tiling curve and interpolation.

### I4 `[~]` Number of tilings and blend weights
- **12-channel, RTPS 0.7, N = 64 per tiling, seed 0** (`P2_L44_I4_*`; spread/RMSE):
  - K = 2: 0.387 (0.90);
  - **K = 4 Hann: 0.338 (0.85)**;
  - K = 16 Hann: 0.291 (0.49);
  - K = 4 uniform: 0.334 (0.58);
  - K = 16 uniform: 0.275 (0.35).
  More tilings or uniform weights buy accuracy at the cost of calibration (independent spreads
  average down). **Part II default: K = 4 Hann**, which is calibrated. K = 16 is reported as the
  accuracy end of the trade-off.
- **44² done 2026-10-09** (seed 0, N = 64 per tiling; analysis RMSE; wall time for 100 cycles):
  | | stride 8 | stride 16 | wall |
  |---|---|---|---|
  | K = 1 | 0.951 | 3.368 | 13 s |
  | K = 2 (diagonal) | 0.692 | 2.248 | 20 s |
  | K = 4 Hann | 0.575 | 1.856 | 31 s |
  | K = 4 uniform | 0.629 | **1.421** | 29 s |
  | K = 16 Hann (quarter-patch) | **0.427** | **1.144** | 102 s |
  | interpolation | 0.829 | 3.008 | |
  K = 4 is not the knee: 16 tilings improve a further 26% (dense network) and 38% (sparse network),
  far beyond the decoder floor's change (0.35 → 0.30). This is consistent with I8: the
  grid-tied *filter* errors average down, not just the seams. Hann weights win at stride 8 and
  uniform weights at stride 16. With sparse observations the grid-tied filter error dominates the
  seams, and uniform weights average more tilings per point. **Follow-ups:**
  - weight family w ∝ hann^p, p ∈ {0, 0.5, 1}, at K = 4 and 16;
  - K = 16 at 220² (the cost is ~3.5× K = 4, still linear in area);
  - K = 16 in I2's equal-cost test (16 × 4 members).
  Calibration worsens with K (spread/RMSE 0.24 at K = 16), since independent spreads average
  down faster than the errors (I6).
- K ∈ {1, 2, 4, 16} with Hann weights, plus K = 4 with uniform weights.
- Run at 44², strides 8 and 16.
- Output: an accuracy-vs-cost curve, and whether 16 quarter-patch tilings are worth 4× more than 4 (the floor goes from 0.35 to 0.31 only).
- Expect K = 4 to be the knee.

### I5 `[ ]` Observation regime (replaces H4; D2 rerun) — the likely headline
- **WARNING (2026-10-09 08:27): at stride 16 the filter has not reached a steady state.**
  Errors drift upward over all 100 cycles: slow filter divergence from under-dispersion. 10-cycle
  block means (12-channel):
  - 1 tiling: 0.56 → 0.90 → 1.21;
  - tiling 0 inside the 16-tiling run: 0.57 → 1.20 → 2.50;
  - 16-tiling blend: 0.35 → 0.47 → 0.65;
  - stride 8, for comparison: flat (0.26–0.29).
  Single runs at stride 16 also vary strongly with the noise realization (tiling 0 gives 1.09 vs
  1.87 in two otherwise identical runs). **Every stride-16 number recorded so far (6-channel 1.86 /
  1.14, 12-channel 1.09 / 1.00 / 0.63) is a cycles-51–100 average of a drifting error; do not
  quote it.** For I5:
  - 300 cycles;
  - a drift test (mean over cycles 151–225 vs 226–300; call a run steady only if they differ by
    < 10%);
  - 3 seeds per cell;
  - run I21 (inflation / RTPS) at stride 16 *first*, because it is the standard cure for this
    divergence.
- Grid: stride ∈ {4, 8, 16, 32} × observation noise ∈ {0.1, 0.2, 0.4} standardized, at 44², K = 1 and K = 4.
- GC c is retuned per stride, c ∈ {1.5, 2, 3}, because sparse networks need wider localization.
- Baselines at every point: spline interpolation, and climatological OI (`exp_oi_bound.py`, generalized to stride and noise).
- Then the two most informative points at 220².
- Pre-registered expectation: the blend's margin over interpolation grows with observation spacing and noise. If it does not, say so; the dense, accurate network is interpolation's best case, and the paper should show the full map.
- Figure: RMSE ratio (blend / best no-model baseline) on the stride × noise grid.

### I6 `[ ]` Calibration retune (D5, D9 rerun)
- The blended spread is 0.37 of the RMSE. The cause is that independent filters' spreads average down while their errors partly do not.
- Sweep at 44², stride 8 and 16:
  - inflation {1.0, 1.1, 1.2, 1.5};
  - model noise {0.02, 0.04, 0.08};
  - assumed R ×{1, 1.5, 2};
  - post-blend spread rescaling: inflate the blended perturbations by √K_eff, with K_eff measured from inter-tiling member correlation. This only rescales the output ensemble, nothing inside the filters.
- Report the best RMSE and the best calibration separately, and the trade-off between them. Do not claim a calibrated ensemble unless one of these delivers 0.7–1.2.

### I7 `[ ]` Forecasts from the blended analysis
- From steady-state analyses (cycles 51–100), forecast each tiling's ensemble mean freely and blend, out to ~10 Lyapunov times (λ₁ of 2D KS at 44²: from Table truth).
- Compare against: forecasts from the single-tiling analysis; forecasts from the true state encoded once (decoder-limited ceiling); persistence; climatology.
- Report the lead time at which error reaches 50% and 100% of the climatological std.
- A DA paper is judged by whether the analysis improves forecasts; this shows whether the 40% analysis gain survives the dynamics or is only a smoother picture at t = 0.

### I8 `[~]` Why the blend works: error mechanics (math + measurement)
- **44² done 2026-10-09** (`h9_L44_I89_T4.json`, stride 8; error variance by band
  [0, 0.6) / [0.6, 1.14) / [1.14, 1.6) / ≥ 1.6):
  - tiling 0: 0.170 / 0.274 / 0.181 / 0.285;
  - Hann blend: 0.077 / 0.118 / 0.087 / 0.050;
  - mean inter-tiling error correlation: 0.26 / 0.48 / 0.32 / 0.12.
  **The gain is at every scale, not only the seam band.** The tilings' analysis errors are
  weakly correlated even at large scales, so most of the single-tiling analysis error is
  *grid-tied* (it depends on where the patch lattice sits); the same-tiling control (0.943) shows it is
  not sampling noise. Seams are its most visible part. Re-encoding the blended mean on one tiling
  gives 0.85 (vs 0.575): feedback re-imposes one tiling's grid-tied error, which explains the
  `--feedback` result. (The w^T C w "prediction" for the uniform average is an algebraic identity,
  not a test; the content is C's correlations.) Write-up frame: translation-averaging of a
  shift-variant estimator (cycle spinning), seams as one instance.
- **220² done 2026-10-09** (`h9_L220_I89_T4.json`): tiling 0 0.143 / 0.242 / 0.160 / 0.269, blend
  0.067 / 0.120 / 0.077 / 0.049, inter-tiling correlation 0.24 / 0.48 / 0.31 / 0.12, re-encoded blend 0.82.
  The same mechanics as at 44², to two digits: size-independent.
- With Σ_k w_k = 1, the blended error is e_b = Σ_k w_k e_k, and per band, var(e_b) = w^T C w, where C is the inter-tiling error covariance.
- Measure C per wavenumber band, and the correlation of tiling errors in the seam band versus the large scales.
- Prediction: the seam band is near-uncorrelated across tilings, so it is reduced roughly 1/K-fold, while the large scales are near-fully correlated and unchanged. Check that w^T C w reproduces the measured blend RMSE.
- Also explain why feedback hurt (0.811): measure the error of re-encoding the blend on one tiling. The prediction is that it returns the 0.69 single-tiling floor, so feedback reinjects the seams every cycle.
- Write as a short proposition plus measurement for App. E.

### I9 `[~]` Seam diagnostic on blended analyses (D10 rerun)
- **44² done 2026-10-09** (`seams_I9.json`, 10 steady-state fields; edge/interior gradient ratio
  on tiling 0's patch grid, half-shifted grid in brackets): truth 1.02 (0.99); single-tiling analysis
  3.70 (0.90); blended analysis 1.02 (1.00) at stride 8 and 0.91 (0.88) at stride 16; blended floor
  1.03 (1.01). **The seams are gone at both grids, not moved.** 220² (6 fields): truth 1.00 (1.00),
  single 3.67 (0.92), blend 1.02 (1.01). Fig. 11 redo still to do (fields in `h9_L*_I89_*_fields.npz`).
- Add `--save-fields` to `exp_h9_tiled_da.py` (truth, single and blended analyses at a few cycles).
- Compute the edge/interior gradient ratio with `exp_d10_seams.py` at 44² and 220². Expect ≈ 1.0, against 3.8 for the single tiling.
- Use the same fields to redo Fig. 11 (snapshot) with a blended-analysis column. The seams disappearing is the most convincing picture of the method.

### I10 `[ ]` Error-anatomy figure (H3 updated)
- One figure, error variance by wavenumber band, for:
  - the decoder floor, single and blended;
  - interpolation;
  - climatological OI;
  - the single and blended analyses;
  - the perfect-model ceiling (I11).
- Add the span-fraction panel from H3, recomputed for the blended prior.

### I11 `[x]` Perfect-model ceiling (H7, now required)
- **Code done 2026-10-09.**
  - `ks_latent/da/letkf_phys.py`: local ETKF on the grid, 4 × 4 analysis blocks, R-localization
    with Gaspari–Cohn, optional multiplicative inflation and RTPS.
  - `tests/unit/test_letkf_phys.py`: 4 tests. Without localization it matches the Kalman mean and
    covariance to 1e-10; distant blocks are untouched; RTPS with α = 1 restores the prior spread.
  - `scripts/exp_i11_pde_letkf.py`: the ensemble is forecast with the PDE itself (batched ETDRK4).
    It assimilates the same truth and the same observation-noise draws as `exp_h9_tiled_da.py`, so
    the comparison is paired.
  - Smoke test (10 cycles): analysis 0.16, spread/RMSE 1.09.
- **Sweep launched 2026-10-09** (`scripts/run_i11_sweep.sh`, log `artifacts/logs/i11_sweep.log`):
  strides 8 and 16 × GC half-width c ∈ {4, 8.25, 12} units × RTPS ∈ {0, 0.5, 0.9}, plus no
  localization with RTPS 0.5; 20 runs, ~3.5 min each. **The ceiling is the best steady RMSE over
  this declared grid** (stated before the results).
- 08:03: the stride-8 best so far sits at the edge of the grid. Errors are 0.131 / 0.096 / 0.083
  for c = 4 / 8.25 / 12, all with RTPS 0, and RTPS only hurts (over-dispersion). So the grid is
  extended upward, to c ∈ {16, 24} and no localization with RTPS 0, at both strides
  (`run_i11_sweep2.sh`, runs after the first sweep). The ceiling is quoted over the extended grid,
  and the extension is stated.
  **Implication for the latent filter:** its GC c was tuned without the blend (best 1.5–2 sites,
  H0 O3). With grid-tied errors averaged out, wider c may now help: add c ∈ {2, 3, 4} sites to the
  I21 sweep.
- **Sweep complete 2026-10-09 09:25** (100 cycles each; ceiling = best steady RMSE over the declared +
  extended grid):
  - **stride 8: 0.076** (c = 24 units; 0.077 at 16, 0.083 at 12, 0.096 at 8.25, 0.131 at 4);
  - **stride 16: 0.213** (c = 24; 0.218 at 16, 0.214 at 12, 0.243 at 8.25, 0.348 at 4).
  Calibration 0.85–1.13 at the best settings. RTPS hurts at every c (over-dispersion; RTPS 0.9 is
  catastrophic). **Unlocalized collapses even with the true model:** 0.314 (stride 8) and 0.529
  (stride 16), spread/RMSE 0.24–0.31, because 64 members are not enough for n₊ ≈ 43. This
  independently supports Part II's localization argument.
  Framing (user): a floor, not a yardstick; one table row.
- Physical-space local ETKF on the 128² PDE with 64 true-PDE members, same observations, at strides 8 and 16, at 44².
- This is the only way to say how good 0.57 is, as a percentage of the best achievable with this network and ensemble size. Report "the latent filter reaches X% of the perfect-model error reduction over interpolation".
- `exp_hybrid_da.py`'s per-patch physical ETKF is the analysis step; the PDE ensemble forecast is new (~half a day).

### I12 `[ ]` Twelve-channel model with the blend (H1 extension)
- **Blended floor measured 2026-10-09** (Stage A AE): 0.276 (1 tiling) / 0.157 (4) / 0.112 (16),
  seam band 0.063 → 0.011 → 0.007. Stage B finishing; then the DA runs below.
- **Pivot test launched 2026-10-09 ~08:05** (`scripts/run_i12_pivot.sh`, log
  `artifacts/logs/i12_pivot.log`; it starts by itself when Stage B finishes, ~08:30):
  - blended DA at 44² (K = 1, 4, 16; strides 8 and 16; seed 0; K = 4 seeds 1–2), ~10 min;
  - then Lyapunov transfer at 44² and 66² (H1 rule (i)), ~1.5 h.
  **Pre-registered pivot rule:** switch the canonical 2D KS model to `p7c12_ks2d` if:
  1. its K = 4 blended analysis beats the 6-channel K = 4 by ≥ 15% at both strides (seed 0), and on
     all three seeds at stride 8;
  2. its D_KY transfer errors at 44² and 66² are within the truth spread (≤ 2.5%; the 6-channel
     model gives 1.7% and 2.4%).
  - If (1) holds and (2) fails: use 12 channels for Part II only, say so explicitly, and keep
    6 channels for Part I.
  - If (1) fails: stay with 6 channels.
  - **DECIDED 2026-10-09 08:25 (user): Part II switches to `p7c12_ks2d`.** The first result,
    K = 4 at stride 8, gives 0.345 against 0.575 for 6 channels. **Part I keeps the 6-channel
    model** unless the 12-channel model shows significant benefits there (Lyapunov transfer at 44²
    and 66², running in the pivot script; report against the 6-channel 1.7% / 2.4%).
    Model-specific Block I runs (I1, I2, I3, I4, I5, I21; I9/I10 figures) are redone with 12
    channels; the 6-channel results stay recorded as the method on the Part I model.
  - **Part I check done 2026-10-09 08:34** (`artifacts/experiments/crop/p7c12_ks2d_x*_seed0.json`): D_KY
    error at 44² is 2.1% (12-channel) vs 1.7% (6-channel); at 66² 1.1% vs 2.4%. The two models are
    mixed, both within the truth spread, one seed and one IC each: **no significant benefit, so Part I
    keeps the 6-channel model**, per the user's rule. The 12-channel numbers can go in an appendix
    sentence ("a 12-channel sibling gives 2.1% and 1.1%") as a capacity robustness check.
  - **User decision 2026-10-09:** two models are acceptable if necessary. Both are in the same
    architecture class, and the 12-channel model only adds capacity, which DA may need.
  - Rule (1) compares the two models at the 6-channel model's untuned settings. Its verdict is
    re-checked once I20/I21 have tuned the localization radius and cycle length, applying the best
    settings to both models (I20/I21 started on the 6-channel model at ~08:10).
  - Meanwhile: no model-specific 6-channel runs (I1 rest, I3, I5) until the decision.
    Model-independent work continues: I11, and the code for I21–I24 and I27.
- Once H1's Stage B is done: compute the blended floor (`exp_overlap_recon.py p7c12_ks2d`), then K = 1 and K = 4 DA at 44², strides 8 and 16.
- If the 12-channel model passes H1's rules (i)–(ii), it becomes canonical: rerun I1, I3 and I5 with it.
- Prediction from H0b plus H9: blended floor ≈ 0.15–0.2; analysis ≈ 0.4.

### I13 `[ ]` Joint multi-tiling filter (option 2) — only if needed
- State = all K tilings' latents; observation operator = the blended decoder.
- Each observation touches at most K sites (one per tiling), so it stays exactly local. The coloured Jacobian costs K × c JVPs.
- Pursue only if I1/I5 leave a gap and I8 shows correlated filter errors that a joint update could reduce. The feedback result (0.811) argues against any variant that re-encodes the blend.

### I14 `[ ]` Cost and memory (§7.4 rerun)
- From I1's records: wall time and peak RSS against area for K = 1 and K = 4, plus parallel wall time (K processes).
- Update `tab_da_cost` and the "memory grows no faster than the area" sentence with the blend's numbers.

### I15 `[ ]` 1D accuracy against interpolation
- From the D1 GC N = 64 runs at L = 200, 400, compute the 1D physical analysis RMSE against spline interpolation of the same observations, at fixed spacing.
- If the 1D filter already beats interpolation (full-reach AE, no seams), say so; it supports the seam diagnosis. If not, report it and diagnose before writing any 1D accuracy sentence.

### I16 `[ ]` Robustness
- (a) Spin-up from a climatological initial ensemble (no knowledge of the truth): cycles to steady state, K = 1 vs K = 4.
- (b) Misspecified observation error: the filter assumes 0.5× and 2× the true noise.
- (c) A 300-cycle run at 220²: no drift or divergence.
- (d) Three seeds at the I5 headline point.

### I17 `[ ]` Positioning and citations (verify every one against the primary source)
- Cycle spinning / translation-invariant denoising: Coifman & Donoho (1995). This is the prior art for averaging shifted block transforms; present the blend as this idea applied to a patch latent, not as a new algorithm.
- Overlapping local analysis domains in operational ensemble filters, e.g. LETKF local volumes (Ott et al. 2004; Hunt et al. 2007) and domain-localized EnKFs with overlap and blending. Check how they treat overlap.
- Latent-DA baselines ROAD-EnKF (arXiv:2301.11961) and LAE-EnKF (arXiv:2603.06752): compare setups and numbers only where the configurations match; otherwise state that the setups differ.

### I18 `[→J9]` Rewrite Part II (absorbs H2, H8)
- Proposed §7 order:
  - 7.1 setting and exactness: Prop. 4 extended. Each tiling's filter is exactly local and the blend is a fixed linear map in physical space, so the composite stays local, at K× the cost.
  - 7.2 accuracy at 44² against interpolation, climatological OI and the perfect-model ceiling, with the controls (I2) and the error anatomy (I10).
  - 7.3 scaling with domain size (I1, I3).
  - 7.4 observation regime (I5).
  - 7.5 forecasts (I7).
  - 7.6 cost (I14).
- Abstract, Key Points, §1 bullet, §9 limitations (calibration, I6; the single 2D system), §10.
- Figures: Fig. 9 rerun with the blend; new error-anatomy figure (I10); Fig. 11 with a blended column (I9); I5 ratio map.

### I19 `[ ]` (optional) Kolmogorov DA at one size with the blend (was D12)
- Evidence that the method is not specific to KS. Only after I1–I11.

### Block I additions (2026-10-09, after the first I11 results and the user's questions)

**Measured today, for these tasks:**
- **12-channel AE with overlapping tilings** (`exp_overlap_recon.py p7c12_ks2d`, Stage A AE, 44²):
  decoder floor 0.276 (1 tiling), 0.157 (4, Hann), 0.112 (16), against 0.665 / 0.340 / 0.311 for
  the 6-channel model. The 12-channel blended DA waits for Stage B (I12).
- **Error budget per cycle, prior → analysis** (44², stride 8, steady state):
  - perfect-model filter (I11, c = 4, no RTPS): 0.143 → 0.131;
  - latent, 1 tiling: 0.992 → 0.952;
  - latent, 4 tilings blended (`h9_L44_T4_prior.json`): 0.646 → 0.575.
  All three analyses are only 4–11% below their priors. **Correction (08:20, after I22's
  measurement):** I first read this as "the learned model's per-cycle error is the bottleneck".
  The data do not support that:
  - The latent model error of one cycle, measured on an independent truth, is small: rms 0.038
    per latent coordinate (climatological std 0.84), i.e. smaller than the isotropic noise the
    runs already inject (0.045 accumulated).
  - The relative error growth over a cycle is similar for both filters: latent blend
    0.575 → 0.646 (+12%), perfect model 0.083 → 0.092 (+11%, c = 12).
  So the latent filter does not lose to new model error each cycle. It equilibrates at a higher
  level because the analysis cannot remove a persistent error component:
  - the blended decoder floor, 0.35 of the 0.575;
  - error outside the ensemble span (H0 O2: span fraction 0.20);
  - a gain that is too small (spread 0.3 of the error).
  The levers are therefore the representation (I12, 12 channels) and the ensemble's span and
  spread (I21), more than the propagator.
- **Spread vs error** (spread/RMSE): perfect model 1.12; latent 0.29 (1 tiling), 0.37 (4).
  The latent filter believes its prior is 3–4× more accurate than it is, so its gain is too small.
  With a perfect model no inflation is needed: RTPS 0.5 over-disperses it (1.60) and costs accuracy
  (0.131 → 0.166).
- **Gaussianity** (`scripts/exp_i_gaussianity.py` → `gaussianity_L44.json`):
  - climatological u: skewness 0.03, excess kurtosis −0.21;
  - latent coordinates per channel: |skew| 0.10, kurtosis −0.38;
  - one-cycle prior ensembles (16 members): mean |skew| 0.40 (latent) and 0.42 (physical).
  For n = 16 the sampling std of the skewness is ≈ 0.6, so the priors show **no detectable
  non-Gaussianity in either space** at this cycle length. This regime cannot show a non-Gaussian
  advantage either way.

### I20 `[x]` Cycle length: assimilate more often
- **Done 2026-10-09** (6-channel model, K = 4, 44², window 50 time units; propagator steps between
  analyses 1 / 2 / 5 / 10):
  - stride 8: 0.551 / 0.558 / **0.575** / 0.595;
  - stride 16: 2.347 / 2.005 / **1.856** / 2.016.
  Weak at dense spacing (−4% for 5× more analyses). Harmful at sparse spacing: every-step analyses
  fit the few observed points (error there 0.32) without spreading the information, a symptom of
  the too-small spread (I21). **Keep 5 steps.** Not repeated for 12 channels unless I21 changes the
  spread.
- The forecast sets the error, and latent forecast error grows quickly, so shorter cycles should
  help the latent filter more than the perfect-model one.
- Sweep n_prop_steps ∈ {1, 2, 5, 10} (Δt between analyses 0.1–1.0), K = 1 and K = 4 latent, and
  the I11 filter at its best setting.
- Report analysis RMSE against Δt for each, plus the number of analyses per time unit (cost).
- Caveat: the observations then come more often too. Report it as "same network, more frequent";
  it is not a free improvement.

### I21 `[ ]` Under-dispersion is an accuracy problem, not only a calibration one (amends I6)
- **Localization part done 2026-10-09** (6-channel, K = 4; GC c = 1.5 / 2 / 3 / 4 sites):
  - stride 8: 0.575 / 0.582 / 0.602 / 0.640;
  - stride 16: 1.856 / 2.017 / 2.173 / 2.181.
  Wider is worse at both strides, unlike the perfect-model filter, which improves up to c = 12
  units ≈ 2.2 sites and beyond. The latent ensemble's long-range correlations are less
  trustworthy. RTPS code is done (`LETKFLocal2D(rtps=…)`, 2 tests; `--rtps` on
  `exp_h9_tiled_da.py`); the inflation/RTPS sweep runs with the 12-channel model.
- **12-channel sweep done 2026-10-09** (K = 4, 44², seed 0; `scripts/run_i21_c12.sh`, drift via
  `scripts/drift_check.py`).
  - **Stride 8, 100 cycles, error (spread/RMSE):**
    - none 0.345 (0.55);
    - RTPS 0.3 0.338 (0.64), 0.5 0.336 (0.73), **0.7 0.338 (0.86)**;
    - inflation 1.2 0.337 (0.56), 1.5 0.342 (0.59).
    **Pre-registered rule met: RTPS 0.5–0.7 gives calibration 0.73–0.86 at no accuracy cost.**
    Adopt RTPS 0.7 at stride 8. Multiplicative inflation does not calibrate.
  - **Stride 16, 300 cycles (mean over cycles 151–300; drift = cycles 226–300 vs 151–225):**
    - none 2.06 (levels off near 2.1 after ~200 cycles);
    - RTPS 0.3 1.74 (steady), 0.5 1.25 (drifting, +18%), 0.7 1.43 (+26%);
    - inflation 1.2 1.96, 1.5 1.95;
    - interpolation 2.89.
    RTPS slows the sparse divergence but does not stop it. No sparse setting is final.
- With spread at 0.3 of the error, the gain is too small. The earlier finding that inflation
  makes it worse (H0 O4: the observed points improve, the total gets worse) was for one tiling,
  where the inflated covariance spread grid-tied errors. The blend averages those away, so the
  conclusion may flip.
- Code: add RTPS / RTPP to `LETKFLocal2D` (one option, as in `letkf_phys.py`).
- Sweep, K = 4 and 16, strides 8 and 16: multiplicative inflation {1, 1.2, 1.5, 2}, RTPS
  {0.5, 0.9}, model noise {0.02, 0.05, 0.1}.
- Pre-registered: if any setting gets the blended spread/RMSE into 0.7–1.2 at no more than 5%
  accuracy cost, adopt it. If calibration and accuracy improve together, that is the headline
  form of I6.

### [Paper 2] I22 `[ ]` Model-error statistics from training data (replace isotropic 0.02 noise)
- **Measured 2026-10-09** (`scripts/exp_i22_model_error.py`, `ks_latent/da/model_error.py` + 4 tests;
  independent truth, seed 3; 25,600 pooled site samples): one-cycle residual rms 0.038 per
  coordinate, bias rms 0.0017, neighbour correlation 0.13 (block-diagonal Q is a fair
  approximation), Q eigenvalues 0.0003–0.0027. **Q is smaller than the noise already injected**
  (0.045), so data-estimated model error will *not* by itself raise the spread. The
  under-dispersion does not come from unrepresented random model error. Still worth one run
  (Q × {1, 4, 16}) to see whether Q's *shape* beats isotropic noise of equal size. Priority
  drops from A to B for the KS paper. For CM1 the estimation code is ready and the question
  stays open: a CM1 surrogate's error may be much larger.
- The perfect-model filter is calibrated because its spread includes all the error growth. The
  latent filter's model error (propagator and decoder) is represented only by isotropic latent
  noise of std 0.02.
- Estimate Q from data: one-cycle residuals r = E(u_{t+Δ}) − M^5(E(u_t)) on training
  trajectories. Because the latent field is local and homogeneous, Q per site is a c × c block
  plus nearest-neighbour cross-blocks, estimated by pooling over sites.
- Draw the additive model noise from N(0, Q) (or a scaled Q) in `forecast_ensemble`.
- Report: spread/RMSE, analysis RMSE, and whether Q's scale transfers to 220² unchanged. It
  should, since it is a per-site statistic, and it makes a clean sentence for the paper.
- This is standard in DA (additive model error estimated offline); here locality makes Q
  site-local and size-independent.

### [Paper 2] I23 `[ ]` Ensemble observation perturbations instead of the mean Jacobian
- `LETKFLocal2D` forms HBHᵀ from the decoder Jacobian at the ensemble mean (coloured JVPs). The
  standard nonlinear EnKF uses the ensemble itself: Y = h(Z) − mean, and Cov(z_s, Y_used).
- With the per-patch decoder this stays exactly local: each observation depends only on its own
  site. It handles the decoder's nonlinearity across the ensemble, and drops the c JVPs per
  cycle, though it needs N decodes, which the filter already does for the innovations.
- Code: a `method="letkf_local2d_ens"` variant plus a unit test (for a linear decoder it equals the
  Jacobian version). Test at 44² with K = 1 and 4.

### I24 `[~]` "No equations needed": framing, the right baseline, and an honest cost comparison
- **Code done 2026-10-09** (`scripts/exp_i24_phys_emulator.py`):
  - learned model used as a physical-space emulator (encode on K tilings, propagate, decode, blend);
  - analysis by the I11 physical local ETKF;
  - same truth and observation draws as the latent runs; supports `--truth-lap-coef` (I28).
  - Smoke test (10 cycles): 0.49, against interpolation 0.78.
  - **Sweep running** (`scripts/run_i24.sh`): K ∈ {4, 1} × c ∈ {8.25, 16} units × RTPS ∈ {0, 0.5, 0.7},
    seed 0; then 3 seeds at the best setting.
- **Sweep done 2026-10-09 (seed 0, stride 8).** Best physical-space analysis with the learned model:
  - K = 4: **0.378** (c = 8.25, RTPS 0.5, calibration 1.19);
  - K = 1: 0.505 (c = 8.25, RTPS 0.5).
  Latent analysis (Part II method): K = 4 0.338 (calibration 0.85); K = 1 0.46–0.48. **The latent
  analysis is ~11% more accurate at fixed model** and better calibrated. The latent space earns its
  place in the analysis, not only in the forecast (single seed; seeds 1–2 at the best settings to
  come).
- **Framing (for I18).** The latent filter needs trajectory data of the full state, not the
  equations. I11 measures what that costs: 0.13 (equations known) vs 0.43–0.56 (learned). Say
  both halves: no equations, but full-state training data, and a 3–4× accuracy price at this
  network.
- **The fair equation-free competitor is not the PDE filter** but a learned physical-space
  emulator inside a physical-space LETKF (Brajard et al. 2020; Bocquet et al. 2020 — verify).
  Cheapest available form, no training: the latent model used as a physical emulator. Forecast
  on K tilings, decode and blend each member, run the physical local ETKF (`letkf_phys.py`, the
  same tuned c as I11), then re-encode on each tiling.
  - If it beats the latent analysis, the analysis belongs in physical space and the latent space
    is a forecast device.
  - If not, the latent analysis is earning its place.
  - Either answer is publishable. (The earlier hybrid, one tiling, gave 0.975; the re-encoding
    caveat from I8 applies.)
- **Cost and memory against the physical LETKF.** Neither filter ever forms a covariance matrix.
  The LETKF works in ensemble space per analysis block, N × N, with memory N × n_grid, so it is
  also linear in area. **The latent method has no asymptotic scaling advantage over a
  physical-space LETKF, and the paper must not claim one.** Measured differences to report:
  - state per ensemble member: 6 numbers per 16 × 16 patch (43× smaller), × K tilings;
  - local analyses: 64 sites vs 1024 blocks at 44²;
  - forecast cost: propagator vs PDE solver per cycle (time both).
  For 2D KS the spectral PDE is cheap, so the forecast saving is modest. It would matter for an
  expensive model, but say "would", not "does".
- **Where the advantage is real**, and how to position it in I18:
  1. Against other latent DA (ROAD-EnKF, LAE-EnKF, global-latent PFF): physical localization
     becomes exact and meaningful in latent space, and the same model transfers across sizes.
  2. Against physical-space DA: no equations needed, at a measured accuracy price.

### [Paper 2] I25 `[ ]` Non-Gaussian errors: what the latent formulation can and cannot claim
- **Honest status.** The filters used (LETKF and variants) are Gaussian. The latent formulation
  does not by itself make the analysis non-Gaussian, and in today's regime the priors are
  Gaussian to within sampling error in both spaces (measured above).
- **Plausible advantages, to test, not to claim:**
  1. **Local particle methods.** Exact locality plus compression give a per-site problem of
     dimension c = 6 with about 16 local observations, instead of 256 grid points per patch.
     Particle-filter degeneracy grows with the effective dimension of the local update, so
     localized particle filters (Poterjoy 2016 MWR; Farchi & Bocquet 2018 NPG review — verify)
     and the codebase's particle flow filter (`da/pff.py`, made per-site) become more tractable
     in the latent field.
  2. **The decoder as a learned anamorphosis.** A Gaussian latent prior maps to a non-Gaussian
     physical one; a test is whether latent priors are closer to Gaussian than physical priors
     in a regime where the latter are not.
- **Regimes that would show it:**
  - longer cycles (Δt = 2–5 time units, priors leave the linear regime);
  - a nonlinear observation operator (e.g., observe u² or |∇u|, or thresholded or bounded data);
  - heavy-tailed observation errors (Student-t, ν = 3).
- **Test plan:** in one such regime, first rerun the Gaussianity diagnostic. Then compare the
  physical LETKF (perfect model), the latent LETKF with the blend, and a per-site latent PFF.
- **Scope:** probably Paper 2. For Paper 1, one paragraph in §9 stating the argument and that
  it is untested. Optionally one demonstration with a nonlinear observation operator, if a
  per-site PFF can be written in a day.

### I26 `[ ]` Analysis granularity — the link between I11's best localization and K
- **Correction 07:50:** the first sweep point suggested the perfect-model filter prefers a GC
  half-width finer than one site, but c = 8.25 units (= the latent filter's 1.5 sites) gives 0.096
  against 0.131 at c = 4. Revisit this task after the sweep; the granularity argument below may not
  hold. The latent filter cannot localize below one site: c < 1.5 sites was
  worse (H0 O3).
- The K-tiling blend analyses each point in K differently placed sites (offsets p/2 for K = 4,
  p/4 for K = 16), a finer effective analysis granularity than one site. That may be part of why
  K = 16 helps.
- Test: K = 16 at the I11 sweep's best c expressed in sites, and K = 16 with c ∈ {1, 1.5, 2}
  sites. If the best latent c decreases with K, write this up as the mechanism (with I8).

### Steering note: Paper 3 is CM1 squall-line DA (user, 2026-10-09)
The end goal is to apply this method to CM1 output for a squall line, where the DA system has no
closed-form evolution equations: the learned surrogate is the model. What that changes for Part II:
- **I11 (perfect-model filter) is a twin-only diagnostic.** It measures the price of learning the
  model. It is not a method we can use in Paper 3.
- **Up-weighted, because they carry over:**
  - I22 (model-error statistics from data): a CM1 surrogate will be imperfect;
  - I23 (ensemble observation perturbations): radar reflectivity and radial velocity are nonlinear
    observation operators;
  - I24 (equation-free framing and forecast cost): against CM1 itself the cost argument becomes real;
  - I25 (non-Gaussian): hydrometeors are bounded and skewed;
  - plus I27 and I28 below.
- **Down-weighted:** results that rely on periodicity, homogeneity or extensivity specifically
  (D_KY-based arguments for the DA part; pooling statistics over all sites).

### [Paper 2] I27 `[ ]` Shifted tilings on a non-periodic domain
- The blend uses periodic rolls, but a squall-line domain is not periodic. On a bounded domain,
  shifted tilings leave partial patches at the edges.
- Options:
  - (a) Each tiling covers only whole patches inside the domain. Renormalize the Hann weights by
    their local sum, so points near the boundary are served by fewer tilings.
  - (b) Pad by reflection or with a halo from a larger window, which is what crop training already
    does.
- Test on 2D KS with a non-periodic window cut from the 220² truth, e.g. 88² inside 220²:
  - the decoder floor and the blended DA error as a function of distance from the boundary;
  - the same with the boundary strip treated as observed.
- The crop-trained model was trained on non-periodic windows, so this should need no retraining.

### I28 `[~]` Imperfect-model twin (the CM1 situation in miniature)
- **Code done 2026-10-09:**
  - `modified_operator2d` and an optional `op=` argument on `integrate2d` (`ks_latent/solver/ks2d.py`;
    3 tests in `tests/unit/test_solver2d_modified.py`: a = 1 is bitwise standard, k* = √(a/2), truths
    differ);
  - `ks_latent/solver/twin.py`: cached truth on the modified attractor, after 100 units of adjustment
    from the standard spin-up;
  - `--truth-lap-coef` on `exp_h9_tiled_da.py` and `exp_i11_pde_letkf.py`. Truth: u_t + a∇²u + ∇⁴u +
    ½|∇u|² = 0; both filters use a = 1.
- **Stage 1 running** (`scripts/run_i28.sh`): a ∈ {1.1, 1.25}, seed 0, stride 8.
  - latent filter (12-channel, K = 4) with RTPS {0.5, 0.7, 0.9};
  - PDE filter (c = 16) with RTPS {0, 0.3, 0.5, 0.7}; each filter gets its own tuning.
  Stage 2: 3 seeds at each filter's best setting.
  - 3-cycle smoke test at a = 1.2: PDE filter 0.49 vs latent 0.74 (meaningless at 3 cycles).
- **Stage 1 results 2026-10-09 12:15** (seed 0, stride 8, 100 cycles; analysis RMSE):
  - **a = 1.1:**
    - latent (K = 4), RTPS 0.5 / 0.7 / 0.9: 0.587 / 0.554 / **0.524**;
    - PDE filter (wrong equation), RTPS 0 / 0.3 / 0.5 / 0.7: 3.11 (diverged) / 1.12 / 0.655 / **0.342**;
    - interpolation 0.963.
  - **a = 1.25:**
    - latent, RTPS 0.5 / 0.7 / 0.9: 1.202 / 1.071 / **0.939**;
    - PDE, RTPS 0 / 0.3 / 0.5 / 0.7: 5.18 / 4.36 / 1.49 / **0.642**;
    - interpolation 1.191.
  **Reading:**
  - A tuned wrong-equation PDE filter still beats the latent filter, by ~1.5× at both levels. The gap
    shrinks from 4.5× with a perfect model.
  - The PDE filter is far more fragile: it diverges or loses to interpolation unless its RTPS is
    ≥ 0.5–0.7. The latent filter beats interpolation at every setting tried at a = 1.1, and at
    RTPS ≥ 0.7 at a = 1.25.
  - Both optima sit at the edge of their sweeps. Stage 1b extends them: latent RTPS 1.0 and
    inflation 1.2 on top; PDE RTPS 0.8 / 0.9.
- **Realism upgrade (user, 2026-10-09):** the parametric error above is the most benign kind for
  the physics filter: right structure, one wrong coefficient. Real model error is structural. Added:
  - **Option 1, resolution error:** `--model-nx M` on `exp_i11_pde_letkf.py`. The PDE filter runs
    the correct equation on an M² grid; observations at the same physical points; scored after
    exact spectral upsampling (Nyquist dropped, round trip 1e-16). 64² barely under-resolves KS
    (smoke: 0.18 vs 0.13). 32² (~6.5 points per characteristic wavelength) is the realistic case.
  - **Option 2, missing processes in the truth only:**
    - uniform mean-flow advection (`--truth-advect-x c`), exact as a Fourier phase shift;
    - fixed terrain-like forcing f(x, y) (`--truth-forcing-amp`): smooth random field,
      correlation length 5. Amplitude 0.2 raises the field std 3.9 → 4.2, with a fixed pattern of
      std 1.9; amplitude 0.4 raises it to 5.0.
    `integrate2d(advect=, forcing=)` and `twin_truth()` in `ks_latent/solver/twin.py`; tests:
    advection = exact roll, zero forcing = standard, and both together are rejected.
    Parametric-only truths reuse the original cache, so every I28 run shares one truth.
  - **REDESIGN (user, 2026-10-09 ~13:00) — the realistic comparison.** The truth is the standard
    2D KS, i.e. the system whose data the latent model was trained on (as a CM1 surrogate is trained
    on the real system's data). The **physics filter's model is wrong**: wrong coefficient
    (`--model-lap-coef`), spurious mean flow (`--model-advect-x`), spurious terrain forcing
    (`--model-forcing-amp`), or coarse grid (`--model-nx`, option 1, already this design). This is
    option 3 without retraining. The latent filter's numbers are the standard-truth runs (0.338 /
    0.322 / seed 2). The truth-modified option-2 runs were killed and deleted (they had both models
    wrong, a different question). The stage-1/1b parametric runs above remain as the "both models
    wrong" variant. Running: `scripts/run_i28d.sh` (4 lanes). `step_batch` wrong-model terms are
    tested equal to the solver's (`test_pde_filter_step_batch_matches_solver_with_model_errors`).
- **REDESIGNED I28 RESULTS 2026-10-09 14:17** (seed 0, stride 8, 100 cycles; PDE filter c = 16 units;
  analysis RMSE at RTPS 0 / 0.5 / 0.7 / 0.9; best in bold; spread/RMSE at best).
  **Latent filter: 0.338 / 0.322 / 0.320** (seeds 0–2, RTPS 0.7, default settings, calibration ≈ 0.85).
  Interpolation 0.83. Correct PDE 0.076.
  | PDE model error | RTPS 0 | 0.5 | 0.7 | 0.9 | best vs latent |
  |---|---|---|---|---|---|
  | coefficient 1.1 | 4.35 | 0.90 | **0.325** (1.59) | 0.46 | tie |
  | coefficient 1.25 | 4.86 | 1.96 | **0.538** (1.38) | 0.59 | latent 1.6× |
  | spurious mean flow 1 | 4.66 | 3.53 | 1.37 | **0.698** (3.43) | latent 2.1× |
  | spurious mean flow 2 | 5.01 | 3.78 | 2.30 | **1.25** (2.04) | latent 3.7×; PDE worse than interpolation |
  | spurious forcing 0.2 | 1.88 | 0.42 | **0.251** (1.67) | 0.44 | PDE 1.35× |
  | spurious forcing 0.4 | 3.75 | 0.80 | **0.313** (1.34) | 0.45 | tie |
  | coarse grid 64² | **0.077** (1.13) | 0.12 | 0.23 | — | PDE 4.4× (64² resolves KS) |
  | coarse grid 32² | 0.52 | **0.187** (0.94) | 0.24 | — | PDE 1.8× |
  **Reading.**
  - Against a physics model with wrong *dynamics* (coefficient, spurious advection or forcing), the
    learned filter ties or wins in 5 of 6 cases, by up to 3.7×. The physics filter wins only for
    mild forcing (1.35×).
  - Against *resolution* error the physics filter keeps the edge: 32² is still 1.8× better, because
    KS's energy-carrying scales are resolved at 32². This would not carry over to convection-
    resolving CM1, but it is a fair loss here.
  - Robustness: the physics filter diverges without RTPS in every wrong-dynamics case. Its best
    RTPS varies with the error (0.7 or 0.9), it is overdispersed at its best (1.3–3.4), and one
    step off its optimum costs 2–4×. The latent filter uses one default setting throughout and is
    calibrated.
  - Caveats: one seed for the PDE runs; RTPS grid coarse (0.6/0.8 untested); the latent model was
    trained on clean data (I33 tests noisy training).
  - Paper sentence (draft): "Against an ensemble filter run with a physics model whose dynamics
    are modestly wrong, the learned latent filter matches or exceeds its accuracy in most cases
    and needs no model-error tuning; a physics model whose only error is resolution keeps its
    advantage."
  - Next: seeds 1–2 at each PDE best setting; RTPS 0.6/0.8 refinement for the close cases
    (coefficient 1.1, forcing 0.2/0.4, 32²).
  - **Option 3 (planned): retrain the latent model on the truth system's data.** This is the
    equation-free advantage the CM1 claim rests on.
  - Running: `scripts/run_i28c.sh`.
  Honest framing for Part II: model error narrows the learned model's accuracy deficit a lot and
  the learned filter is more robust to tuning, but a physics model with modest errors remains
  more accurate in this twin.
- **Tier A (user, 2026-10-09).** The perfect-model filter (I11) is a floor, not a yardstick: no perfect
  model exists for CM1 or most real systems. The paper's claims rest on the no-model baselines and
  I24, with I28 as the realism check; I11 appears as one table row ("price of learning the model
  in an idealized twin").
- The truth comes from a slightly different system than the one the surrogate was trained on:
  2D KS with a modified coefficient (e.g., a damping term or a different ratio of the ∇² and ∇⁴
  terms), or Kolmogorov flow at a different drag.
- Compare: the latent filter with model noise from I22, the same with isotropic noise, and
  interpolation. The perfect-model filter, now run with the wrong equations, is an informative
  extra: the learned model is not alone in being wrong.
- This is the closest Part II can get to Paper 3's setting, where the surrogate is fit to CM1 and
  the truth is the atmosphere (or a different CM1 configuration).

### I29 `[ ]` What lies outside the ensemble span? (diagnostic before any training change)
- The "80% outside the span" figure (H0 O2) is a pessimistic measure: a *global* least-squares fit
  of the whole 128² prior error with 63 decoded perturbations, for one tiling, before the blend. A
  local filter only needs the error spanned *locally*. Measure before retraining anything:
  1. **Local span fraction:** per site, fit the prior error in the site's analysis window (GC
     support) with the ensemble perturbations restricted to that window, then average.
  2. **The same for the perfect-model ensemble (I11)**, which works well. If its global span
     fraction is also low but its local one is high, the global number was never the diagnosis.
  3. **Decomposition of the blended prior error** into representation error
     (D(E(u)) − u, unreachable by any latent ensemble) and latent error (D(z_f) − D(E(u))), each
     with its local span fraction.
- Decision:
  - If the out-of-span part is mostly representation error, the fix is the decoder: I12, then I30(b).
  - If it is latent error, the fix is how the ensemble is generated: I30(c).
- Effort: ~2 h code, minutes to run. `exp_hybrid_da.py --diag` has the global version.

### [Paper 2] I30 `[ ]` Training-side options (only after I29 says which part dominates)
- (a) **Capacity:** the 12-channel model (I12, running). Lowers the representation part directly
  (floor 0.276 → 0.112 with 16 tilings).
- (b) **Blend-aware autoencoder fine-tuning.** Add a loss on the Hann-blended reconstruction over
  K random sub-patch offsets, keeping the single-tiling loss. The decoders then learn errors that
  cancel across tilings, not just errors that happen to. Exact locality is untouched: each tiling
  is still per-patch. ~Half a day of code, ~2 h fine-tuning from the current checkpoint; then
  Stage 2 (always run Stage 2).
- (c) **Stochastic, calibrated propagator.** Per-site noise inputs, trained with an ensemble
  score (energy score or CRPS) over k-step rollouts against encoded truth. Ensemble spread and
  its directions then come from the learned forecast-error distribution instead of isotropic
  latent noise; this targets the span and the calibration at once. 1–2 days. Highly relevant to
  CM1, where a deterministic surrogate's spread will be wrong.
- **(d) tested 2026-10-09** (`scripts/exp_i30_propagator_degradation.py`, 12-channel, stride 16,
  RTPS 0.5, 300 cycles). One-cycle propagator error against E(PDE(D(z))) from the same state:
  - encoded truth 0.059;
  - **analysis means 0.085–0.088 (1.5×)**;
  - truth + random latent noise of the analysis error's size (0.45): **0.56–0.67 (10×)**.
  Off-manifold distance |E(D(z)) − z| of analysis means is 0.09, against 0.08 for the truth, and
  does not grow. So analysis *means* stay where the propagator works, and their degradation does
  not drive the sparse divergence. But random perturbations, i.e. what ensemble *members* carry
  (model noise, RTPS), wreck it. **Fine-tune launched 2026-10-09 08:47** (user):
  - `p7c12n15_ks2d` = same AE plus `p7c12_ks2d`'s propagator fine-tuned 2000 steps;
  - constant input noise 0.15, k = 32 throughout, lr 1e-4;
  - new `--init-prop / --noise-start / --prop-lr / --k-full` options on `train_kolmogorov_crop.py`;
  - log `artifacts/logs/crop_train_p7c12n15_ks2d.log`.
  Then rerun: the degradation test, K = 4 at stride 8 and 16 (RTPS 0.5, 300 cycles), and a free
  rollout screen, since noise training can damp the dynamics (Lemma 1).
- **Fine-tune evaluated 2026-10-09 09:43** (`p7c12n15_ks2d`, seed 0, K = 4):
  - stride 8, RTPS 0.7: 0.350 (original 0.338), calibration 0.76 (0.86);
  - **stride 16, RTPS 0.5, 300 cycles: 0.87, bounded**: block means stay 0.7–1.1 and end at 0.78
    (original 1.25, still climbing to 1.34);
  - stride 16 without RTPS: 2.27 (original 2.06). **Neither fix works alone; the combination does.**
  - D_KY at 44²: 82.3 against true 81.8 (**0.6%**; original 2.1%), so no damping from noise training.
  - Degradation test: from analysis means 0.095 (truth 0.062); from random perturbations 0.67,
    *unchanged* from the original (0.67).
  **Correction to the I30(d) reading:** the random-perturbation number does not measure "the
  propagator breaks". E(PDE(D(z))) lies on the manifold whatever z is, so any off-manifold part
  of z that the propagator carries forward counts as error. That number mainly says neither
  propagator projects perturbations back onto the manifold. So the fine-tune's sparse-observation
  gain is real (one seed), but its mechanism is **not** established by this test.
  Next: 2 more seeds at stride 16 (RTPS 0.5, both propagators), and a projection baseline
  (z ← E(D(z)) on members after the model noise), which tests the manifold idea directly.
- **3-seed check, 2026-10-09 10:30** (12-channel, K = 4, stride 16, RTPS 0.5, 300 cycles; median blend
  error over cycles 151–300; interpolation 2.79–2.89):
  - original propagator: 1.25 (drift +18%) / **blow-up** (seed 1, from cycle ~142) / 1.60 (+20%);
  - noise fine-tuned: 0.84 / **blow-up** (seed 1, from cycle ~239) / 1.40;
  - **member projection** z ← E(D(z)) after each forecast, original propagator: **1.46 / 1.54 / 1.66,
    steady on every seed** (drift ≤ 5%), with no new model.
  - Projection without RTPS: 2.62.
  Seed 0's fine-tune success was partly luck. RTPS 0.5 at sparse observations inflates members into
  regions where neither propagator is bounded; projection keeps them bounded. Note: a blow-up in one
  tiling was averaged silently into the blend. `exp_h9_tiled_da.py` now stops a run when any tiling's
  error exceeds 5× the field std and records `diverged_at_cycle`.
  Next: fine-tuned + projection, and the I31 gates (on = original; off = project or fine-tuned),
  3 seeds each.
- (d) **Robustness to off-manifold analysis states.** Analysis latents need not lie in the
  encoder's image. Train the propagator and decoder on perturbed latents ("analysis-like"
  inputs; Stage 2 already anneals input noise 0.10 → 0.02, so check first whether raising its
  floor helps). Cheap to test via a Stage 2 rerun.
- (e) **DA-aware end-to-end training:** train the latent model to maximize the filter's
  observation likelihood (ROAD-EnKF's approach, arXiv:2301.11961). The strongest form, but heavy.
  Paper 2/3, not Paper 1.
- Not proposed: anything that adds decoder reach (it breaks the exact block-diagonal Jacobian,
  which is Part II's structural result), and the banded latent regularizer (known to collapse).

### [Paper 2] I31 `[~]` Manifold-gated propagators (user idea, 2026-10-09)
- **Gate option 1 implemented** (`ks_latent/da/gated.py`, 4 tests; `--gate [--gate-off-tag]` on
  `exp_h9_tiled_da.py`). It is not learned: per-site round-trip score ‖E(D(z)) − z‖ with threshold
  d₀ = 95th percentile of clean scores on the independent seed-3 truth (0.132), width 0.008. About 8%
  of site-states are routed off.
- **3 seeds, stride 16, RTPS 0.5, 300 cycles (median over 151–300; drift; spread/RMSE):**
  - original: 1.25 / blow-up / 1.60;
  - fine-tuned: 0.84 / blow-up / 1.40;
  - projection: 1.46 / 1.54 / 1.66 (calibration 0.33–0.39);
  - fine-tuned + projection: 1.52 / 1.59 / 1.52;
  - gate original/projection: 1.48 / 1.40 / 1.46;
  - **gate original/fine-tuned: 1.15 (+43%, cal 0.88) / 0.74 (+7%, 1.02) / 1.32 (+6%, 0.41)**.
  The MoE gate is the most accurate (mean ≈ 1.07 vs ≈ 1.55 for projection), has no blow-ups, and is
  best calibrated on 2 of 3 seeds. Seed 0 is still rising late. Interpolation 2.79–2.89.
  Next: 600-cycle runs (gate, projection; 3 seeds), and the gate at stride 8.
- **Problem.** Training the propagator on noisy inputs makes it robust to off-manifold states, but a
  squared-error loss on noisy inputs favours contraction (Lemma 1), which damps the dynamics and
  can collapse the spread. The noise-free propagator keeps the dynamics but fails badly on
  perturbed states: 10× its usual error for random latent perturbations of the analysis-error size
  (I30(d)).
- **Idea.** Route each state, mixture-of-experts style, between an accurate propagator M_c (the
  current one) and a noise-robust one M_r (the I30(d) fine-tune), using a gate that measures
  closeness to the data manifold:
  z' = g ⊙ M_c(z) + (1 − g) ⊙ M_r(z), with g ∈ [0, 1] **per site**.
  - The gate must be local and weight-shared (computed from a site's neighbourhood), or exact
    locality and size transfer are lost.
  - A soft blend avoids members jumping between models.
- **Gate options, cheapest first:**
  1. No training: the per-site round-trip distance ‖E(D(z)) − z‖ (truth ≈ 0.08, analysis means
     ≈ 0.09, members ≈ 0.10 at 44²), passed through a logistic with threshold and width set from the
     clean-encoding distribution.
  2. Per-site density: Mahalanobis distance in the c-dimensional site latent (and neighbours),
     fitted on clean encodings.
  3. A learned per-site classifier: clean encodings vs perturbed ones. Or train the gate jointly
     with M_c and M_r, a real mixture of experts.
- **Test (after the fine-tune lands; both models then exist, so option 1 needs no training):**
  - K = 4 DA at strides 8 and 16 (300 cycles, drift test), against M_c alone and M_r alone;
  - error, spread/RMSE and the I30(d) degradation numbers;
  - a free-rollout Lyapunov check (D_KY) of the gated model, which must stay at M_c's value on
    clean trajectories.
- **Watch:** the gate will mostly fire on ensemble *members* (means stay near the manifold). Members
  set the spread, and a contracting M_r shrinks it, worsening under-dispersion. Calibration is
  the key output, not just RMSE.
- **Baselines it must beat:**
  - (a) project members back after the model noise, z ← E(D(z)), with no second model;
  - (b) model noise drawn from the data-estimated, on-manifold Q (I22) instead of isotropic latent
    noise;
  - (c) M_r alone.
- **Prior art to check:** mixture-of-experts routing; hybrid physics/ML models that switch on ML
  uncertainty or out-of-distribution detection; manifold-projection steps in latent DA.
- **Relevance to Paper 3:** CM1 analysis states will be further from a surrogate's training
  distribution than KS states are, so an explicit in/out-of-distribution routing may matter more
  there.

### [Paper 2] I32 `[~]` Tiling-reset failsafe (user, 2026-10-09)
- **Observation.** Over 600 cycles at stride 16 (RTPS 0.5), every method eventually degrades:
  - projection 1.91 / 2.22 (seeds 0/1, cycles 301–600);
  - the MoE gate diverged at cycle 593 (seed 0).
  Blow-ups always start in ONE tiling while the others stay healthy: at divergence the tiling errors
  were 1.25 / 0.70 / **18.6** / 0.66.
- **Fix** (`--tiling-reset` on `exp_h9_tiled_da.py`). It uses no truth:
  1. Each cycle, compute each tiling's disagreement with the median of the other tilings' analysis
     means.
  2. If a disagreement exceeds 3× the median disagreement and 1.0 field units, drop that tiling from
     this cycle's blend.
  3. Re-initialize it from the blend of the healthy tilings, with members re-encoded in its frame.
  Resets are logged as (cycle, tiling, disagreement, median). Unlike `--feedback` (every tiling,
  every cycle, which hurt: 0.811), it acts only on outliers, rarely. A single-model filter has no
  analogue: the K tilings are a built-in consistency check.
- Healthy disagreement at stride 8 is ≈ 0.34, so the defaults should never fire there (checked by
  `L44_C12GR_st8`).
- **Running 2026-10-09 10:50** (`scripts/run_reset.sh`): MoE gate + reset and original + reset;
  stride 16, RTPS 0.5, 600 cycles, seeds 0–2; plus gate + reset at stride 8.
- **Results 2026-10-09 11:08** (stride 16, RTPS 0.5, 600 cycles; mean blend error over cycles 301–600;
  interpolation 2.74–2.87):
  - projection: 1.91 / 2.22 / 1.93;
  - MoE gate alone: **diverged (cycle 593)** / 2.07 / 1.14;
  - original + reset: 1.60 / 1.81 / 1.52, **bounded on all seeds** (seed 1 blew up without reset);
  - **gate + reset: 1.17 / 1.66 / 1.58** (mean 1.47), bounded on all seeds.
  - Resets fired: original + reset 1 / 7 / 3; gate + reset 2 / 0 / 0.
  - **Reproducibility caveat:** gate + reset on seeds 1 and 2 fired no reset, so it is the same
    configuration as gate alone, yet gave 1.66 vs 2.07 and 1.58 vs 1.14. The long sparse runs are
    not run-to-run reproducible: tiny floating-point differences (multithreaded BLAS / thread
    count) grow chaotically over 600 cycles. The dense stride-8 pair matched exactly. So
    between-method differences at stride 16 / 600 cycles below ~±0.5 are within noise.
    Comparisons need ≥ 5 seeds and fixed threads (OMP_NUM_THREADS=1 or deterministic torch);
    blow-up vs no blow-up remains a robust distinction.
  At stride 8: 0 resets in 100 cycles, error unchanged (0.341). The reset prevents every blow-up
  seen so far at no cost where nothing is wrong. Remaining issue: slow late creep in all methods
  (error still rising in the last 100–200 cycles), so long-run sparse cycling is contained, not
  solved.
- Paper framing if it works: the tilings serve as an ensemble of models (I8) *and* as a divergence
  detector; worth a sentence in §7 and possibly a figure panel showing a reset event.

### Scope decision (user, 2026-10-09 ~11:10): Part II vs. the DA follow-up paper
**Part II of Paper 1:**
- exact locality and per-site filtering;
- shifted-tiling blending (I8, I9, I2, I4);
- scaling with domain size (I1, I3);
- dense-observation accuracy and calibration with the 12-channel model (RTPS 0.7);
- no-model baselines and the perfect-model floor (I11);
- the equation-free baseline (I24);
- **the imperfect-model twin (I28: kept in Part II, user)**;
- cost (I14); figures and rewrite (I9, I10, I18).
Sparse observations get one honest paragraph: long-run sparse cycling is limited by the learned
model's stability; failsafes exist and are promising (pointer to the follow-up).

**Moved to the DA follow-up paper (Paper 2; bridge to CM1, Paper 3):**
- I30 (noise fine-tuning, projection);
- I31 (manifold-gated MoE propagators);
- I32 (tiling-reset failsafe);
- the sparse half of I5 (stride ≥ 16, long cycling);
- I25 (non-Gaussian, nonlinear observations);
- I22 (model-error Q), I23 (ensemble observation perturbations), I27 (non-periodic tilings).
Their results so far stay recorded here. Before any claim in that paper, the sparse runs need ≥ 5
seeds and deterministic threading (see the I32 reproducibility caveat).

### I33 `[x]` Latent model trained on noisy data (user, 2026-10-09) — Part II
- **Why.** In the I28 redesign the latent model was trained on clean, exact data from the true
  system, which is idealized. Measurement noise is far better characterized than model error, and
  a squared-error loss on noisy targets learns the clean conditional mean (noise2noise). So a
  learned model trained only on noisy data should stay close to the clean one, while a physics
  model's structural error has no such remedy.
- **Existing latent-noise training is different.** It only adds noise to the propagator's inputs;
  the training data itself was clean.
- **Setup.** `--data-noise σ` on `train_kolmogorov_crop.py`: one fixed Gaussian realization added
  to training and validation fields; no clean data anywhere in training. Run `p7c12d10_ks2d`,
  σ = 0.1 (= the DA observation noise), same recipe as `p7c12_ks2d`, launched 2026-10-09 ~13:50
  (nice 10). Auto-evaluation `scripts/run_i33_eval.sh`: K = 4 seeds 0–2 and K = 1 at stride 8,
  RTPS 0.7; D_KY at 44²; always scored against the CLEAN truth.
- **RESULT 2026-10-09 20:59 — benign; if anything slightly better.** Scored against the clean truth,
  stride 8, RTPS 0.7:
  | | seed 0 | seed 1 | seed 2 | K = 1 | decoder floor (1 / 4 tilings) |
  |---|---|---|---|---|---|
  | noisy-data model `p7c12d10` | **0.324** | **0.310** | **0.302** | 0.457 | 0.251 / 0.137 |
  | clean-data model `p7c12` | 0.338 | 0.322 | 0.320 | 0.46–0.48 | 0.287 / 0.164 |
  Better on all three seeds (by 4–6%) with a lower decoder floor. This is consistent with
  squared-error training on noisy targets learning the clean conditional mean, plus a
  regularizing effect. D_KY at 44²: 84.7 vs 81.8 (**3.6%**; clean 12-channel 2.1%, 6-channel
  1.7%): slightly more dimension error, still within ~4%. Note it as a caveat (DA does not
  depend on it).
  **Consequence for the claims:** "trained on clean data" can be replaced by "trained only on
  data with observation-level noise" for the model-error comparison.
- **Next:** σ = 0.3 (~2–3 h of training) to map where noise starts to hurt; optionally make
  `p7c12d10` Part II's model. Decide after J1/J2.
- **Caveat for the paper.** For CM1 the training data will be high-resolution simulations or
  reanalysis (with their own model error), not noisy full fields. This test is a step toward
  realism, not the end.

### Block I priorities and estimates (2026-10-09, with the Paper 3 / CM1 steering)
Part II stays in Paper 1 for now; it may later become its own paper. "Machine" is unattended run
time, with large domains one at a time; "work" is coding, analysis and writing.

| Pri. | Task | Machine | Work | Why |
|---|---|---|---|---|
| A | I1 scaling 88², 330², 440², seeds (completes I0) | ~2.5 h | small | The scaling claim must hold for the method we report. |
| A | I12 12-channel model: H1 checks + blended DA | ~2–3 h | ~1 h | Floor 0.11 with 16 tilings. If adopted, I1/I3/I5 rerun (+~4 h machine). |
| A | I21 inflation / RTPS with the blend | ~1 h | ~0.5 h | Spread is 0.3 of the error; likely an accuracy gain, not just calibration. |
| B | I22 model error Q from training residuals | ~0.5 h | done | Measured: Q is smaller than the existing noise, so it is not the under-dispersion cause. One shape-vs-isotropic run left; code ready for CM1. |
| A | I24 equation-free baseline (latent model as a physical emulator in the physical LETKF) + cost timing | ~1 h | ~3 h | Answers "why latent?" — the reviewer question. |
| A | I5 observation spacing × noise grid | ~2.5 h | ~1 h | Likely headline (sparse observations: a third of interpolation's error). |
| A | I11 finish sweep (running) | ~1 h | small | Ceiling for the accuracy statement. |
| A | I9 Fig. 11 redo, I14 cost table, I17 citations | — | ~0.5 day | Paper. |
| A | I18 rewrite §7, abstract, key points, §9, §10 | — | ~2 days | Paper. |
| B | I23 ensemble observation perturbations | ~1 h | ~3 h | Nonlinear observations (radar) in Paper 3. |
| A | I28 imperfect-model twin | ~2 h | ~2 h | The fair comparison: the physical model is wrong too, as for CM1 and any real model. Promoted to A (user, 2026-10-09). |
| B | I27 non-periodic blend | ~1 h | ~3 h | Needed for CM1; could move to Paper 3. |
| B | I20 cycle length | ~1 h | small | Cheap. The forecast sets the error. |
| B | I3 ensemble-size sweep | ~1 h | small | Completes the "efficient ensemble" story. |
| B | I7 forecasts from analyses | ~1 h | ~2 h | Reviewers will ask. |
| B | I10 error-anatomy figure | — | ~0.5 day | Explains the mechanism. |
| B | I15 1D vs interpolation | ~0.5 h | ~1 h | Consistency of claims. |
| B | I31 manifold-gated propagators (after the I30(d) fine-tune) | ~1 h | ~3 h | Gate option 1 needs no training; tests calibration and D_KY, not just RMSE. |
| C | I13 joint filter, I16 robustness, I19 Kolmogorov DA, I26 granularity, I25 experiment (I25 paragraph is A, ~1 h) | — | — | Only if time; I25 experiment → Paper 2/3. |

Totals:
- A tier: ~12 h machine (+~4 h if the 12-channel model is adopted) and ~3.5 days of work, including the rewrite.
- B tier: ~8 h machine and ~2 days of work.
- Suggested sequence:
  1. Overnight: I1 and I5.
  2. Day 1: I12, I21, I22 code.
  3. Day 2: I24, I22 runs, I23.
  4. Days 3–5: figures and the rewrite, with B-tier runs in the background.

**Queued 2026-10-09 06:45** (`scripts/run_h9_queue2.sh`, waits for the 220² pair; log
`artifacts/logs/h9_queue.log`): I8+I9 at 44² (strides 8, 16; `--diag --save-fields`), I2 at 44²
(1×64, 2×32, 4×16, seeds 0–2), I4 at 44² (K = 2, 16, uniform-weight 4; strides 8, 16), then 220²:
I2 (4×16, 2×32) and I8+I9 (K = 4). Seam ratios via `scripts/exp_i9_seams.py` → `seams_I9.json`.

**Order:**
1. I1(a) at 220², running now; apply I0.
2. I2, I4, I8 and I9 at 44². These are fast, about 1 h in total.
3. I5 (~2 h) and I6 (~1 h).
4. I11: half a day of code.
5. I12, when H1 finishes.
6. I1(b, c), I3 and the I5 points at 220², one large run at a time, overnight.
7. I7, I10, I14 and I15.
8. I18.

I13, I16 and I19 can wait.

---

## Block J — Finalize Part II: learned latent DA vs physics-based DA under model error (2026-10-09)

**New perspective (user, 2026-10-09).** We will never have a perfect dynamical model. A learned
latent model trained on data from the true system is compared against a physics-based ensemble
filter whose model is (modestly) wrong. Measurement noise is far better characterized than model
error, so it can be accounted for when training the learned model (I33). Together with exact
locality and size transfer, this is Part II's central argument.

What Part II now claims (to be confirmed by J1–J5):
1. Exact locality and size transfer:
   - one model and one filter setting from 44² to 440²;
   - 12-channel, K = 4: 0.338 / 0.321 / 0.326 / 0.340 at 44² / 88² / 220² / 330²;
   - interpolation 0.81–0.83.
2. Shifted-tiling blending (cycle spinning on the patch lattice):
   - ~30% lower error at no retraining;
   - at equal total members, 4 × 16 beats 1 × 64 by 25%;
   - the seams vanish;
   - the mechanism is grid-tied error.
3. Accuracy and calibration with dense observations:
   - ≈ 0.32–0.34, about 2.5× better than interpolation and climatological OI;
   - spread/RMSE ≈ 0.85 with RTPS 0.7;
   - the latent analysis beats a physical-space analysis with the same model (I24: 0.338 vs 0.378).
4. **Against physics-based DA with model error (I28 redesign):**
   - wrong dynamics: the latent filter ties or wins in 5 of 6 cases (up to 3.7×) with no model-error tuning;
   - the physics filter diverges without RTPS, has a narrow tuning window, and is overdispersed at its best;
   - resolution-only error: the physics filter keeps its edge (KS resolved at 32²);
   - perfect model: 0.076 (the floor, one table row).
5. Limitation, stated: long sparse-observation cycling is limited by the learned model's stability;
   failsafes exist (pointer to the DA follow-up paper).

### J1 `[x]` Seeds for the PDE filter (model-error comparison)
- Seeds 1–2 for the PDE filter at its best RTPS for each of the 8 model errors (coefficient
  1.1/1.25, mean flow 1/2, forcing 0.2/0.4, grid 64²/32²): 16 runs, ~3–5 min each with thread caps
  (`OMP_NUM_THREADS=2`, `--workers 2`; never uncapped, see the 12:40 oversubscription incident).
- Latent seeds 0–2 exist (0.338 / 0.322 / 0.320).
- Report mean ± range per case. A case counts as a "tie" if the ranges overlap.
- ~1.5 h machine.

### J2 `[x]` Fair tuning of the PDE filter
- RTPS 0.6 and 0.8 for every case, and 1.0 for the mean-flow cases, where 0.9 was best (edge of
  grid).
- Localization c ∈ {8.25, 24} at the best RTPS for the closest cases (coefficient 1.1, forcing 0.2
  and 0.4, 32²).
- Multiplicative inflation 1.1 / 1.2 as an alternative to RTPS for one case per error type.
- Rule (pre-registered): the PDE filter's number in the paper is its best over this whole grid on
  seed 0, then confirmed on seeds 1–2. The latent filter keeps its single default.
- ~2 h machine.

- **J1 + J2 launched 2026-10-09 14:36** (`scripts/run_j12.sh`, 2 lanes, nice 10, 2 threads, so the J4
  evaluation always takes priority; log `artifacts/logs/j12.log`). Per case:
  1. RTPS 0.6 / 0.8 (1.0 for the mean-flow cases);
  2. the best RTPS is picked automatically from all seed-0 c = 16 runs;
  3. c ∈ {8.25, 24} at that RTPS, and inflation 1.1 / 1.2 without RTPS;
  4. seeds 1–2 at the best RTPS (c = 16).
  If J2 finds a better c or inflation for a case, rerun its seeds at that setting before quoting it.

- **J1/J2 RESULTS 2026-10-09 22:15** (c = 16, best RTPS, seeds 0/1/2; Part II latent filter
  `p7c12d10` 0.324 / 0.310 / 0.302, mean 0.312):
  - coefficient 1.1: RTPS 0.7, 0.325 / 0.323 / 0.318 → **tie**;
  - coefficient 1.25: RTPS 0.8, 0.475 / 0.488 / 0.480 → **latent 1.5×**;
  - mean flow 1: RTPS 0.9, 0.698 / 0.707 / 0.697 → **latent 2.2×**;
  - mean flow 2: RTPS 1.0, 1.175 / 1.112 / 1.133 → **latent 3.7×**;
  - forcing 0.2: RTPS 0.7, 0.251 / 0.249 / 0.244 → **PDE 1.26×**;
  - forcing 0.4: RTPS 0.7, 0.313 / 0.310 / 0.315 → **tie**;
  - 64² grid: RTPS 0, 0.077 / 0.088 / 0.085 → PDE;
  - 32² grid: RTPS 0.6, 0.184 / 0.180 / 0.180 → **PDE 1.7×**.
  Seed spread < 0.02 in every case. Multiplicative inflation alone diverges or is far worse in every
  wrong-dynamics case.
  - **Fair-tuning follow-up:** c = 8.25 beat c = 16 on seed 0 for mean flow 1 (0.584), mean flow 2
    (0.863) and forcing 0.4 (0.297). Per the pre-registered rule these are reported at c = 8.25 once
    seeds 1–2 exist. Queued `scripts/run_j1b.sh` (after J16): seeds 1–2 at c = 8.25; c = 4 for the
    mean-flow cases (8.25 is the grid edge); J16(a) mean-flow parameter estimation at c = 8.25.
  - The latent filter still wins the mean-flow cases at c = 8.25 (0.58 / 0.86 vs 0.31).

### J3 `[ ]` The latent side of the comparison at full strength
- Also report K = 16 (0.29, the accuracy end; calibration 0.49) as the latent filter's accuracy
  ceiling, so the reader sees both ends of its trade-off.
- Latent seeds are done.
- ~0.5 h.

### J4 `[x]` I33 — the latent model trained on noisy data (done 2026-10-09; see I33)
- `p7c12d10_ks2d`, data noise 0.1; auto-evaluation queued.
- If its DA error stays within ~15% of the clean model's, replace "trained on clean data" in the
  claims with "trained on noisy data". Then train σ = 0.3 (~1 h) to map the degradation.
- If it degrades badly, say so; this is the most likely reviewer objection to claim 4.

### J5 `[ ]` (B) Model error and transfer together
- One or two wrong-dynamics cases (coefficient 1.25, mean flow 1) at 220²: the PDE filter on 640²
  against the latent filter (0.326). Shows that claim 4 holds at a transferred size.
- PDE ensemble at 640² is ~25× the 44² cost (~1–2 h per run), so run overnight, one at a time.

### J6 `[ ]` Figure: learned vs physics-based DA under model error (new §7 figure)
- **Panel (a):** analysis RMSE per model-error case (8 cases), for:
  - the PDE filter at its best setting, with seed ranges;
  - the latent filter (seed range, horizontal band);
  - the perfect-model PDE floor (dashed);
  - interpolation (dotted).
- **Panel (b):** tuning sensitivity, i.e. RMSE against RTPS for the PDE filter in three
  representative cases, log y-axis, marking divergence; the latent filter's single default as a point.
- **Panel (c, optional):** spread/RMSE at the best settings.
- Data: `artifacts/experiments/da2d/i11_I28W_*`, `i11_I28R_*`, `h9_L44_C12_st8_rtps0.7_s*`.
- Script: `scripts/paper_figures/fig13_model_error.py`, following the §F figure contract.

### J7 `[ ]` Tables for Part II (regenerated with the 12-channel model)
- `tab07_2d_da` (scaling): 12-channel, K = 1 and 4, 44²–440², with interpolation, climatological
  OI, spread/RMSE, wall time and peak RSS (I1, I14).
- New `tab07_model_error`: the J1/J2 numbers.
- `tab07_2d_ensemble` (equal cost / number of tilings): I2, I4.
- `tabE_da`: settings (RTPS 0.7, Hann blend, K = 4, c = 1.5 sites).

### J8 `[ ]` Write the new §7 subsection: "Learned versus physics-based models under model error"
- **Framing to carry in J8** (user, 2026-10-09). The result is significant because it combines
  four things:
  1. a latent model learned **only from noisy data** of the system (J15 / I33: never sees a clean
     field or an equation);
  2. that **matches or beats** a physics-based ensemble filter whose dynamics are modestly wrong, in
     5 of 6 wrong-dynamics cases;
  3. with **one default setting**, while the physics filter needs per-case tuning and diverges
     without it;
  4. **at any domain size** (44²–440², no retraining).
  State it in that order, then the caveats.
- **Caveats that must appear in §7.5 (not only in §9):**
  - the physics filter had RTPS only, and J16 adds parameter estimation / additive model error;
    word the claim according to J16's decision rule;
  - the model-error sizes were chosen, not calibrated to a real model (illustrative); real
    structural errors such as convection microphysics are often larger and non-parametric;
  - the learned model's training data is complete noisy fields, plentiful, from the exact system,
    whereas real data is sparse observations or reanalysis;
  - resolution-only error favours the physics filter here (KS is resolved at 32²);
  - one system, twin experiments.
- **Table row:** the clean-data model `p7c12_ks2d` (0.338 / 0.322 / 0.320), showing that noisy
  training costs nothing (slightly better).
- **Design:** the truth is the system the latent model was trained on. The physics filter's model
  has a parametric error (∇² coefficient), a structural error (spurious mean flow, spurious
  terrain-like forcing), or a resolution error (coarse grid). Each filter is tuned fairly (J2).
- **Results:** J6 figure, J7 table.
- **Robustness:** divergence without RTPS, the tuning window, overdispersion at the best setting.
- **Honest scope:**
  - resolution error favours the physics model here, because KS is smooth;
  - the latent model's own data-related error (I33);
  - one system;
  - twin experiments.
- The "perfect model" result appears as the floor only.
- ~1 day.

### J9 `[ ]` Restructure and rewrite §7 (absorbs I18)
- Proposed order:
  - 7.1 setting, exact locality and the per-site filter. Extend the exactness statement: each
    tiling's filter is exactly local and the blend is a fixed linear map, so the composite stays
    local.
  - 7.2 shifted-tiling blending: method, mechanism (I8 grid-tied error), seams (I9), equal-cost (I2), K and weights (I4).
  - 7.3 accuracy and calibration at dense observations: no-model baselines, RTPS, latent vs
    physical-space analysis (I24), error anatomy (I10).
  - 7.4 transfer and scaling: 44²–440² (I1), ensemble size (I3).
  - **7.5 learned versus physics-based models under model error (J8).**
  - 7.6 observation density, the 1D results, and the sparse-observation limitation paragraph,
    pointing to the follow-up.
  - 7.7 cost and memory (I14).
- Replace every 6-channel Part II number with 12-channel ones. Keep the 6-channel single-tiling
  results only where they explain the motivation (H0 diagnosis).
- ~1.5 days.

### J10 `[ ]` Two-model statement
- Part I uses `p7b_ks2d` (6 channels); Part II uses `p7c12_ks2d` (12 channels, same architecture
  class, more capacity for DA).
- Text in §7.1 and App. A; App. D sentence: the 12-channel sibling's D_KY transfer errors are
  2.1% / 1.1% at 44² / 66² (robustness of Part I to capacity).
- ~1 h.

### J11 `[ ]` Abstract, Key Points, Plain Language Summary, §1 bullet, §9 discussion, §10 conclusions
- **Carry the J8 framing into these** (user, 2026-10-09):
  - Abstract: one sentence. "Trained only on noisy data, the latent model matches or exceeds an
    ensemble filter run with a modestly wrong physical model, without model-error tuning, and keeps
    its accuracy on domains up to 100× larger." Word it per J16's outcome.
  - Key Point 3: fold in "learned only from noisy data".
  - Plain Language Summary: "we never have a perfect model of the weather; a model learned from
    (noisy) data can do as well."
  - §1: add the motivation (no perfect dynamical model in practice; measurement noise is better
    understood than model error, so it can be handled in training).
  - §9: the caveat list from J8, CM1 as the target (Paper 3), and J16 if it is not yet done.
  - §10: one sentence.
- Abstract and Key Point 3: accuracy against no-model baselines, transfer to 440², and the
  model-error comparison sentence.
- §9:
  - CM1 motivation (no perfect model; learned surrogate from data);
  - the noise-trained model (I33);
  - resolution vs dynamics error;
  - the sparse-observation limitation and follow-up paper;
  - the reproducibility note for long sparse runs.
- §10: one paragraph for Part II.
- ~0.5 day.

### J12 `[ ]` App. E and App. F updates
- **App. E:**
  - tiling blend (Hann weights, K, offsets);
  - RTPS (Whitaker & Hamill 2012);
  - the physical-space LETKF used as the comparator (blocks, R-localization, c);
  - wrong-model definitions (equations for each case);
  - noisy-data training (I33).
- **App. F:**
  - new scripts: `exp_h9_tiled_da.py`, `exp_i11_pde_letkf.py`, `exp_i24_phys_emulator.py`,
    `run_i28d.sh`, `run_partII_c12.sh`, `train_kolmogorov_crop.py --data-noise`;
  - modules: `letkf_phys.py`, `ks_latent/solver/twin.py`, `gated.py` (follow-up).
- ~0.5 day.

### J13 `[ ]` Citations (verify every one against the primary source)
- Cycle spinning: Coifman & Donoho 1995.
- RTPS: Whitaker & Hamill 2012.
- LETKF: Hunt et al. 2007; Ott et al. 2004.
- Additive and multiplicative inflation for model error: e.g. Houtekamer & Mitchell; Anderson.
- Data-driven DA: Brajard et al. 2020; Bocquet et al. 2020.
- Noise2noise: Lehtinen et al. 2018.
- Latent DA: ROAD-EnKF, LAE-EnKF.
- ~0.5 day.

### J14 `[ ]` Number provenance and consistency
- Add every new Part II number to `check_numbers.py` (EXPLAINED dict where needed), regenerate the
  tables from JSONs only, rerun G2/G3-style passes on §7.
- ~0.5 day.

### J15 `[~]` **Part II switches to the noisy-data model `p7c12d10_ks2d` (user, 2026-10-09 ~21:10)**
- **Decision.** Part II's latent model is the 12-channel model trained ONLY on data with one fixed
  realization of observation-level noise (σ = 0.1, I33). It is slightly more accurate than the
  clean-trained `p7c12_ks2d` on every seed (0.302–0.324 vs 0.320–0.338) and is the more realistic
  setting: real training data is noisy, and measurement noise is better characterized than model
  error. This strengthens the model-error claim (J8).
- Part I keeps `p7b_ks2d` (6 channels, clean).
- `p7c12_ks2d`'s Part II results stay recorded as the clean-data reference: one sentence and a
  table row ("training on clean data instead gives …").
- **Reruns, all with `scripts/run_partII_d10.sh`** (launched 21:10; log `artifacts/logs/partII_d10.log`).
  Settings are unchanged (K = 4, Hann, RTPS 0.7, c = 1.5 sites, stride 8, 100 cycles).
  - [~] RTPS check 0.5 / 0.9 at 44². Pre-registered: keep 0.7 unless another value is > 5% better
    and calibrated (0.7–1.2).
  - [x] 44² K = 4 seeds 0–2, K = 1 (I33: 0.324 / 0.310 / 0.302; 0.457).
  - [~] I2 equal members (1 × 64, 2 × 32, 4 × 16; seeds 0–2).
  - [~] I4 K = 2 / 16, uniform weights.
  - [~] I3 ensemble size N ∈ {8, 16, 32, 128} at 44²; {16, 32, 128} at 220².
  - [~] I8 / I9 / I10 diagnostics and saved fields at 44² and 220² (for figures).
  - [~] I24 physical-space analysis with the same model (seeds 0–2, K = 4; K = 1).
  - [~] I1 scaling 88² / 220² / 330² / 440², K = 4 and K = 1; seed 1 at 220² and 330².
  - [~] D_KY at 66² for the Part II model (44²: 3.6%).
  - [ ] I14 cost and memory: take wall time / peak RSS from these runs, excluding any run
    interrupted by machine sleep (check `pmset -g log`), or rerun one size cleanly.
  - [ ] J5 (optional, model error + transfer at 220²): use this model.
  - [ ] J8 / J9 / J11 text and J6 / J7 figures and tables: quote `p7c12d10` numbers throughout Part II.
- **Lane A results (44², seeds 0/1/2; error / spread-to-error), 2026-10-09 21:31:**
  - RTPS 0.5 / 0.7 / 0.9: 0.322 / 0.324 / 0.340, so **0.7 is kept** (pre-registered rule).
  - **I2, equal 64 members:**
    - 1 × 64: 0.457 / 0.431 / 0.418 (0.86–0.93);
    - 2 × 32: 0.376 / 0.359 / 0.350 (0.92–0.99);
    - **4 × 16: 0.352 / 0.334 / 0.318 (0.77–0.85)**, 23–24% better than 1 × 64.
  - **I4 (N = 64 per tiling):** K = 2 0.369 (0.97); K = 4 0.324 (0.85); K = 16 0.287 (0.51); K = 4
    uniform 0.313 (0.63). Same trade-off as before.
  - **I3 (K = 4, N per tiling):** 8 → 0.450 (0.53); 16 → 0.352 (0.77); 32 → 0.330 (0.87); 64 → 0.324;
    128 → 0.322 (0.94). Saturates by N ≈ 32.
  - **I8:** inter-tiling error correlation by band 0.42 / 0.39 / 0.41 / 0.11 (more correlated than the
    clean model's 0.26 / 0.48 / 0.32 / 0.12 at large scales).
  - **I9 seams (edge/interior gradient ratio):** truth 1.02; single tiling **1.73** (clean model 3.7:
    noisy training roughly halves the seams by itself); blend 1.02.
  - **I24, physical-space analysis with the same model:** 0.354 / 0.345 / 0.338 (calibration
    1.22–1.27), against latent 0.324 / 0.310 / 0.302 (0.85). **The latent analysis is 5–10% better on
    every seed and better calibrated** (clean model: 11%). K = 1: physical 0.461 vs latent 0.457.
  - **D_KY:** 44² 3.6%, **66² 2.1%** (188.5 vs 192.5; 6-channel model 2.4%).
  - Scaling so far: 88² 0.304 (K = 1 0.423); 220² 0.312 (K = 1 0.434).
  - **Not rerun:** the PDE-filter model-error runs (J1 / J2 / I28: independent of the latent
    model); the perfect-model floor (I11).
- **Later (user: "we can do 1 later"):** train σ = 0.3 (`--data-noise 0.3`, ~2–3 h) to map
  where data noise starts to hurt; one table row in §7.5.
- Paper text: state that Part II's model never sees clean data, and give D_KY caveats (44²: 3.6%,
  66²: pending) in App. D/E.

### J16 `[~]` Give the physics filter its standard model-error remedies (reviewer-critical; run when time allows)
- **Code done 2026-10-09 (runs not yet queued):**
  - `ks_latent/da/param_est.py`, `global_param_update`: global ETKF update of an augmented scalar
    parameter, with RTPS and a spread floor. 5 tests: the mean and variance equal the ensemble
    Kalman regression to 1e-10; RTPS 1 restores the spread; the floor is respected.
  - `exp_i11_pde_letkf.py`:
    - `--estimate-param {lap,adv}` (`--param-init-std`, `--param-rtps`, `--param-min-std`), with
      member-specific operators or advection speeds;
    - `--model-error-file` plus `--bias-correct` / `--q-scale`.
  - `scripts/exp_j16_estimate_pde_model_error.py`: offline one-cycle residuals on the independent
    seed-3 truth, giving bias, spectral shape and sigma. A perfect-model control gives a residual
    of 1e-15.
  - Also fixed: dotted tags were truncated by `Path.with_suffix`.
- **Smoke tests (20 cycles, seed 0):**
  - coefficient 1.25 + estimation: the estimate moves 1.25 → 1.12 (true 1.0, still converging);
    analysis 0.372 vs 0.475 for the best fixed-parameter setting;
  - mean flow 1 + estimation: the estimate moves 1 → ~0.08 (true 0) in ~10 cycles; 0.452 vs 0.698;
  - forcing 0.2: the residual is almost entirely bias (0.101 of 0.1014); bias correction gives
    0.297 vs 0.305 over 20 cycles.
  **Expectation: parameter estimation recovers much of the physics filter's accuracy for the
  parametric and advection cases**, so the pre-registered rule may narrow the claim to
  non-parameterizable structural error. That is still the realistic CM1 case, and it is honest.
- **Runs QUEUED 2026-10-09 21:25** (`scripts/run_j16.sh`, log `artifacts/logs/j16.log`). It waits for
  J1/J2, reruns the 12 J2 forcing runs that crashed at 21:13–21:19 from a since-fixed print bug,
  estimates the model-error files, then runs the plan below (seeds 0–2, 100 cycles; best RTPS from J2):
  - (a) coefficient 1.1 / 1.25 and mean flow 1 / 2 with `--estimate-param`;
  - (b) forcing 0.2 / 0.4 and 32² with estimated `--bias-correct` ± `--q-scale` {0.5, 1, 2};
    estimate the files first: `exp_j16_estimate_pde_model_error.py --out <case>`.
  ~40 runs, ~3 h at low priority.
- **Why.** In I28/J1/J2 the physics filter's only model-error treatment is RTPS. Operational and
  research DA also estimate model error online. A parametric or advection error is exactly what
  parameter estimation can recover, so the physics filter could close much of the gap there. Without
  this test the model-error claim is open to the objection "you didn't let the physics filter
  correct itself".
- **Do** (`exp_i11_pde_letkf.py`, new options; seed 0 first, then seeds 1–2 at the best setting):
  - (a) **State-augmented parameter estimation** for the parametric and advection cases: append the
    ∇² coefficient a (cases coefficient 1.1 / 1.25) or the mean-flow speed c (cases mean flow 1 / 2)
    to each member as a global scalar. Members draw it from N(wrong value, σ_p²) initially. The
    LETKF updates it through its ensemble covariance with the observations; use a global (not
    localized) update for the scalar, plus parameter-spread relaxation so it does not collapse.
    Standard references: Anderson (2001) and Ruiz, Pulido & Miyoshi (2013) on parameter estimation
    with EnKFs (verify).
  - (b) **Additive model-error noise** for the forcing and resolution cases: add N(0, Q) to each
    member's forecast, with Q estimated offline from one-cycle model-minus-truth residuals on an
    independent trajectory (as `ks_latent/da/model_error.py` does for the latent model). Tune Q's
    scale.
  - (c) Optional: offline bias correction, i.e. subtract the mean one-cycle residual.
  Unit test: the parameter estimator recovers a known coefficient in a perfect-model twin to < 2%.
- **Decision rule (pre-registered):**
  - If (a) brings the physics filter below the latent filter in the parametric/advection cases,
    restate the claim as "for structural errors that cannot be parameterized" and lead with the
    forcing and resolution cases.
  - If it does not, the claim stands, with "even with online parameter estimation".
- **Effort:** ~1 day of code (state augmentation in `LocalETKFPhys2D`, Q estimation for the PDE),
  ~3 h of runs.

### J18 `[ ]` Training on noisy data helps DA: make it a highlighted result (user, 2026-10-10)
**Why.** The model trained only on σ = 0.1 noisy data (`p7c12d10_ks2d`) beats the clean-data model at every
domain size (4 tilings, seed 0):

| Domain | 44² | 88² | 220² | 330² | 440² |
|---|---|---|---|---|---|
| Clean-data model | 0.338 | 0.321 | 0.326 | 0.340 | 0.326 |
| Noisy-data model | 0.324 | 0.304 | 0.312 | 0.326 | 0.311 |

44², seeds 0–2: 0.338/0.322/0.320 → 0.324/0.310/0.302; spread/error 0.92–0.99; seam ratio 1.94 → 1.73 at matched settings (J18c);
D_KY within 2–4%. Consistent but modest (~4–5%); the strong message is that realistic noisy training
costs nothing and helps a little. Paper: a short highlighted subsection plus a figure, not the headline.

- **J18a σ = 0.3 model** (deferred earlier). Train `p7c12d30_ks2d` (same recipe, `--data-noise 0.3`), then
  DA at 44² (K = 4 seeds 0–2, K = 1) and one larger domain (220²). Shows whether more noise helps more,
  plateaus, or hurts: turns one point into a trend. ~2.5 h training + ~1 h runs. Run after J17B's
  second training (one MPS training at a time).
- **J18b mechanism check.** For each model (clean vs σ = 0.1, and σ = 0.3 when available): take true
  states, nudge them slightly off the attractor the way filter members are (latent noise like the
  analysis spread, and physical-space noise of matched size), and measure how far the encoder → decoder
  round trip moves them, and whether it moves them TOWARD the truth (distance to the clean state
  before vs after). If the noisy-trained model pulls perturbed fields back closer, that explains the DA
  gain. ~20 min, no training.
- **J18c seam figure.** The error ratio at patch seams, 3.7 (clean) → 1.73 (noisy), as a figure: error
  vs position within a patch for both models (single tiling), plus the K = 4 blend. The most visible
  effect of noisy training.
- **J18b DONE (2026-10-10, `exp_j18b_roundtrip.py`, 40 states of the independent seed-3 truth, 44²).** Ratio =
  rms(round trip − truth) / rms(perturbed − truth); < 1 means the round trip moves the field toward the truth.

  | Perturbation (standardized std) | clean-data model | noisy-data model (σ = 0.1) |
  |---|---|---|
  | none: reconstruction error | 0.046 | **0.039** |
  | white grid-point noise 0.05 / 0.1 / 0.2 / 0.4 | 0.95 / 0.53 / 0.35 / 0.29 | **0.81 / 0.44 / 0.29 / 0.23** |
  | smooth noise (4-point correlation) 0.05 / 0.1 / 0.2 / 0.4 | 1.35 / 1.09 / 1.02 / 0.99 | 1.26 / 1.07 / 1.01 / 0.99 |
  | latent noise 0.05 / 0.1 / 0.2 | 1.04 / 1.01 / 1.00 | 1.05 / 1.02 / 1.00 |

  - Both models remove grid-scale noise; the noisy-trained one removes ~15–20% more, and reconstructs
    clean states 15% better.
  - Neither pulls SMOOTH or latent-space perturbations back toward the attractor (ratio ≈ 1): those pass
    through unchanged. So the mechanism is better small-scale denoising and reconstruction, **not** a
    general projection onto the attractor. The paper should say exactly that.
- **J18c DONE (2026-10-10, `fig_j18c_seams.py`, `docs/figures/j18c_seams.png`; clean-model field runs
  `h9_J18c_clean_T1/T4`).** Matched settings (44², seed 0, RTPS 0.7):
  - **Correction:** the gradient seam ratio (edge/interior, truth = 1.02) is **1.94 (clean) → 1.73 (noisy)**
    for a single tiling, not 3.7 → 1.73. The 3.7 came from an earlier clean-model measurement under
    different settings. At matched settings the excess over truth falls from 0.92 to 0.71 (−23%).
  - With 4 tilings the seams vanish for both models (blend 1.02 = truth).
  - Error profile across a patch (edge/interior RMSE): 1.44 → 1.38 single tiling; 0.98 / 0.99 with 4 tilings.
  - Paper: noisy training reduces seams modestly; tilings remove them. The seam figure belongs with the
    tilings result more than with the noisy-training result.
- **J18a queued** (`run_j18a.sh`, caffeinated): trains after J17B's two models (~19:00), runs ~23:00.
- **Expected / implications:** if J18b shows a pull toward the attractor, the paper can state the
  mechanism (the encoder learns a projection onto the attractor, which the filter benefits from); with
  J18a the claim becomes "noisy training is a regularizer for latent DA, optimal near σ ≈ X".

### J17 `[ ]` Missing dynamics: no setting of the physics model's parameters reproduces the truth (user, 2026-10-09)
**Why.** In J1/J2/J16 every physics-model error is either a wrong parameter (which online parameter
estimation can recover) or a fixed pattern (which bias correction can remove). The fairest, and the
most CM1-like, comparison is a truth containing a process that **no configuration of the physics
model can represent**, so the physics filter has a state-dependent error that no remedy removes. The
learned model is trained on data from that true system, as a CM1 surrogate would be. This requires
a new truth system **and retraining the latent model on its data**.

**Common protocol for both variants:**
- The physics filter is the standard 2D KS PDE filter, given ALL J16 remedies:
  - J2-style tuning (RTPS, c);
  - bias correction and additive model error estimated offline on an independent seed-3 truth of
    the new system;
  - online estimation of the ∇² coefficient as a *surrogate* parameter (the best-fitting KS, which
    is what a practitioner would tune).
- The latent filter uses the Part II settings (K = 4, Hann, RTPS 0.7, c = 1.5 sites, stride 8),
  a 12-channel model trained ONLY on σ = 0.1 noisy data from the new system, never re-tuned.
- Baselines: interpolation; the perfect-model PDE filter (true equations) as the floor.
- Seeds 0–2. Pre-register the decision rule and the process strength before any DA run.

#### Variant A — a missing state-dependent process that the data fully determine (do first)
Truth: u_t = KS(u) − β u³ (local, nonlinear, saturating). The KS family has no u³ term, and it is
state-dependent and odd in u, so its mean contribution is ~0 and **bias correction can remove almost
none of it**. It depends only on the current field, so a learned model can in principle capture it
completely.

- **A1 Choose β (pre-registered, no DA).** Set β so that rms(β u³) ≈ 15% of the rms KS tendency on
  the standard attractor. Also report 30% as a second level for A9. Check that the new system stays
  chaotic and bounded over 2000 time units.
  - *Expected:* β ~ 10⁻³ in field units (to be measured).
- **A2 Solver.** Add `cubic_coef` to `ks_latent/solver/ks2d.py` (an explicit term in N, like the
  forcing) and the matching option to `twin.py`, `exp_i11_pde_letkf.py` (`--model-cubic-coef` for the
  perfect-model floor) and `exp_j16_estimate_pde_model_error.py`. Tests:
  - β = 0 equals the standard solver;
  - the term's tendency matches a direct evaluation;
  - a spatially constant field decays as du/dt = −β u³ (analytic).
- **A3 Characterize the new truth.** Field std, energy spectrum and D_KY at 44² (truth Lyapunov
  pipeline) against standard KS, so the paper can state how different the system is.
- **A4 Training data.** `scripts/gen_ks2d_wide_dataset.py` with the new term (same size as
  `ks2d_wide6`: 5 trajectories, 264², 768², 25 time units, dt_snap 0.1). ~30 min.
- **A5 Train the latent model** (`p7c12d10_cubA`, `--data-noise 0.1`, same recipe): ~5 h.
  Screens: free-rollout statistics; D_KY at 44² against A3 within the usual ~4%.
- **A6 Physics-filter runs.** KS model (no u³): RTPS sweep; bias correction ± additive noise; ∇²
  coefficient estimation as surrogate. Seeds 0–2 at the best. ~3 h.
- **A7 Latent-filter runs.** K = 4 seeds 0–2, K = 1, plus a 220² transfer check. ~1 h.
- **A8 Floor and baseline.** Perfect-model PDE filter (with u³) and interpolation.
- **A9 (if A6–A7 are close) Strength sweep:** the 30% level (one more dataset and model) to locate
  the crossover process strength.
- **Expected outcome:**
  - The physics filter's error stays above its usual 0.08–0.3, and bias correction and parameter
    estimation recover little, because the error is state-dependent and not KS-shaped.
  - The latent filter stays near its usual 0.30–0.35 if training captures the term.
  - Most likely the latent filter wins after all remedies, provided the process is strong enough;
    A9 finds the threshold.
- **Implications for the paper (Part II, §7.5):**
  - If the latent filter wins: the headline structural-error result. "When the equations lack a
    process that the data contain, a latent model learned only from noisy data outperforms a
    physics-based filter equipped with online parameter estimation, bias correction and additive
    model error, and does so at any domain size."
  - If it does not: report the crossover strength. The claim becomes "beyond a missing-process
    strength of X% of the tendency."
  - Either way it replaces the parametric cases as the section's main comparison, with those kept
    as the easier cases.

#### J17A progress and pre-registration (2026-10-10, written before any J17A DA run)
- **A2 done.** `ks2d.cubic_term2d` (−β P[u′³] on the mean-removed field u′, so the system keeps
  KS's u → u + c invariance and the k = 0 gauge fix stays exact); `cubic=` on `step2d`,
  `integrate2d`, `spinup2d`, `step_with_tangent2d` (exact tangent −3β P[u′² δu′]), `twin_truth`,
  `gen_ks2d_wide_dataset.py --cubic`, `exp_h9_tiled_da.py --truth-cubic`,
  `exp_i11_pde_letkf.py --truth-cubic/--model-cubic-coef/--norm-std`,
  `exp_j16_estimate_pde_model_error.py --truth-cubic`. Tests (`test_solver2d_modified.py`, 12 pass):
  β = 0 bitwise equals the standard solver; the term equals a direct evaluation and is invariant
  under u → u + c; single-mode projection −(3/4)βA³ (analytic); tangent vs finite difference
  (1e-6); batched PDE-filter stepper equals the solver; advect + cubic refused.
  - Note: a "constant field decays as −βu³" test is not meaningful here, because the mean is gauge.
- **A1 β (measured on the standard attractor, 44², seed 3, 40 snapshots over 200 units).**
  - rms KS tendency 1.278; rms P[u′³] 168.8; field std 3.75.
  - 15% level: β = 1.14 × 10⁻³. **Pre-registered β = 1.1 × 10⁻³** (14.5%).
  - 30% level (A9): β = 2.2 × 10⁻³.
  - Boundedness and chaos check (2000 units, twin-start separation): `j17a_check_bounded.py`,
    results in `artifacts/experiments/da2d/j17a_bounded_beta*.json`. **Passed at both levels:**
    - β = 1.1e-3: field std 3.14–3.39 per 100-unit window over 2000 units (standard KS 3.75, i.e.
      13% lower), max |u| 13.3; twin starts 1e-8 apart separate to 0.66 in 100 units (chaotic).
    - β = 2.2e-3: std 2.92–3.07 (20% lower), max 10.1; separation 1e-8 → 0.06 in 100 units (still
      chaotic, weaker growth).
  - Truth D_KY at 44² (A3): `j17a_truth_lyap.py`, G1 settings (150 directions), running.
- **Paper wording for the J17A process (user, 2026-10-10).** Describe −βu′³ as "a Swift–Hohenberg-type
  cubic saturation absent from the forecast model". Points to make:
  - −u³ is the classic saturating term of the Swift–Hohenberg equation, u_t = ru − (1 + ∇²)²u − u³,
    which shares KS's ∇²/∇⁴ structure (citation to verify).
  - It has no linear part, so KS's linear instability (fastest-growing wavenumber 1/√2) is unchanged;
    it only saturates large amplitudes.
  - The system stays chaotic: field std −13% (β = 1.1e-3) and −20% (2.2e-3); twin-start growth
    comparable to standard KS (A1); truth D_KY from A3.
  - No literature search was done (user decision): do not claim the KS + cubic system is new or that
    it was used before; present it purely as a controlled missing process.
- **Pre-registered decision rule.**
  - Physics filter: the best configuration over the A6 grid (RTPS, c; bias correction ± Q;
    ∇²-coefficient estimation), **selected on seed 0**, then run on seeds 0–2.
  - Latent filter: Part II settings unchanged (K = 4, Hann, RTPS 0.7, c = 1.5 sites, stride 8),
    seeds 0–2, model `p7c12d10_cubA`, no tuning.
  - Metric: steady-state analysis RMSE (cycles 51–100), field units, same truths, obs and noise
    (σ_obs = 0.1 × the new training-data std, via `--norm-std`).
  - **Latent wins** if its 3-seed mean is ≥ 10% below the physics filter's AND it is lower on every
    seed. **Physics wins** under the mirror condition. Otherwise **tie** → run A9 (β = 2.2 × 10⁻³).
- **Launched 2026-10-10 ~10:00:** A4 data (5 trajectories in parallel, `--only`) and the A1
  boundedness checks at both β levels.

#### J17A interim results (2026-10-10, 11:40)
- **A3 truth D_KY (β = 1.1e-3): 76.1** (standard KS 81.8, −7%), λ₁ = 0.154, 38 positive exponents.
- **A4 data std** (raw field) 4.05; σ_obs = 0.405.
- **One-cycle model error of the KS model** (seed 3, rms field units): 0.055 (bias 0.022). Smaller than
  the J16 forcing-0.2 case (0.10), which the physics filter handled.
- **A6 tuning, seed 0, no remedies:** best 0.190 (c = 8.25, RTPS 0.7); 0.192 at c = 16. Already below the
  latent filter's usual ~0.31 and its decoder floor ~0.25 → the 15% level is a physics win (pending
  formal completion).
- **Lesson:** tendency share (15%) is the wrong strength measure; what matters is the one-cycle forecast
  error relative to σ_obs.
- **β scan of the one-cycle error and chaos** (44², seed 3; separation of starts 1e-6 apart after 100 units):

  | β | one-cycle error (bias) | field std | separation after 100 u | regime |
  |---|---|---|---|---|
  | 1.1e-3 | 0.055 (0.022) | 3.27 | grows to O(1) | chaotic |
  | 2.2e-3 | 0.074 (0.042) | 3.00 | grows | chaotic (weaker) |
  | 3.3e-3 | — | 2.95 | 5.8e-4 | weakly chaotic (λ ≈ 0.06) |
  | 4.4e-3 | 0.102 (0.087) | 2.65 | 2.7e-4 | weakly chaotic |
  | 6.6e-3 | 0.103 (0.101) | 2.49 | 9e-7 (no growth) | **not chaotic** |
  | 1e-2 | 0.167 (0.167) | 2.41 | 2e-8 | **steady pattern** |

  The cubic process **cannot reach a one-cycle error of 0.2–0.3 while staying chaotic.** At large β the
  error becomes almost pure bias (a frozen pattern), which bias correction removes exactly. So the
  user-approved "strong level by forecast error" (option 2) is not achievable with −βu³.
- **Launched:** the pre-registered A9 level β = 2.2e-3 (`run_j17a30.sh`: data, then training after the 15%
  model, physics A6/A8, latent A7; truth D_KY via `j17a_truth_lyap.py`).
- **Open decision (user):** for a large missing-process error that stays chaotic, a different process
  is needed (state-dependent, not saturating), or Variant B (hidden memory).

#### J17C (stopped) and J17H — structurally wrong physics model, all coefficients estimable (2026-10-10)
- **J17A stopped (user).** The −βu³ truth cannot reach a large one-cycle error while chaotic (β scan above).
- **J17C stopped (user).** Truth = standard KS, physics model = KS + spurious u³. Rejected: if its
  coefficient is estimated it can go to zero, which recovers KS exactly (the J16 parametric case).
- **J17H design (user-approved).** Truth = standard 2D KS (so the latent filter's existing result, 0.312
  over seeds 0–2 with `p7c12d10_ks2d`, needs no retraining). The physics model is
  u_t = −a∇²u − b(−∇²)^{p/2}u − (g/2)|∇u|², with damping order **p ≠ 4** (main level p = 6, ∇⁶
  hyperdiffusion, a standard modelling choice in weather/climate models).
  - **All coefficients (a, b, g) are estimated jointly online** (`--estimate-param abg`,
    `param_est.global_params_update`), starting from the offline best fit.
  - **No (a, b, g) reproduces KS**: the growth-rate curve aK² − bK^p has the wrong shape for p ≠ 4.
  - The order p is the model's structure (the modeller's choice), not an estimated parameter.
- **Code:** `ks2d.hyper_operator2d`; `exp_i11_pde_letkf.py --model-order/--model-hyper-coef/--model-nl-coef`,
  `--estimate-param abg` (per-member operators and nonlinearity, b kept > 0);
  `exp_j16_estimate_pde_model_error.py` model options; `exp_j17h_fit.py` (offline best fit on the
  independent seed-3 truth: minimum one-cycle error = the irreducible structural error); `run_j17h.sh <p>`.
  Tests: p = 4, a = b = 1 equals the standard operator; order-6 symbol; the nonlinearity coefficient
  obeys KS's scaling symmetry u → u/g; the joint update equals the column-wise scalar updates and the
  joint ensemble Kalman regression; per-column floors.
- **Pre-registered decision rule (before any J17H DA run).** Physics filter: best configuration over
  tuning (RTPS × c), joint (a, b, g) estimation, bias correction ± Q, and estimation + bias
  correction, selected on seed 0, then run on seeds 0–2. Latent: 0.312 (existing, untouched). **Latent
  wins** if the physics 3-seed mean is ≥ 10% higher and higher on every seed; mirror for a physics win;
  otherwise tie. Strength is reported as the offline irreducible one-cycle error; levels p = 5, 6, 8
  give a curve if they differ.

#### Variant B — a hidden process with memory (realistic; probably the DA follow-up paper)
Truth: u_t = KS(u) + ε v, and v_t = −v/τ + κ ∇²u (or κ u). Here v is an unobserved field with its own
slow dynamics, τ ≈ 1–5 time units (longer than a 0.5-unit DA cycle). v is a memory integral of past
u, not a function of the current u, so **neither model can represent it fully**:
- the physics filter lacks it entirely;
- the learned model sees only u and is Markovian, so it captures v only as far as v leaves a trace in
  u's spatial structure.
This mirrors unobserved moisture or microphysics in CM1.

- **B1 Design (pre-registered).** Choose κ, ε, τ so that rms(ε v) ≈ 15% of the KS tendency, and v's
  autocorrelation time is ≥ 2 DA cycles. Check boundedness and chaos.
- **B2 Coupled solver.** A two-field ETDRK4 (u, v), with v's linear part −1/τ and its explicit part
  κ∇²u (or κ u); u's explicit part gains + ε v. Tests:
  - ε = 0 gives the standard u;
  - for a prescribed u, v matches the analytic linear-ODE solution;
  - energy and boundedness over long runs.
- **B3 Characterize the truth.** u statistics against KS; v autocorrelation; how much of v is
  predictable from the current u (regress v on local u features; this bounds what a Markovian
  learned model can capture).
- **B4 Training data.** u only (v saved separately for diagnostics, never used in training);
  σ = 0.1 noise.
- **B5 Train the latent model** (`p7c12d10_hidB`): ~5 h. Optional ablation: the `history` /
  `two_step` propagator modes, giving the learned model access to the past, which can carry the
  memory.
- **B6 Physics filter.** KS without v, plus all J16 remedies. Optional extra: a physics model that
  includes v but with a wrong τ (structurally right, parametrically wrong).
- **B7 Latent-filter runs and floor.** The perfect-model filter with (u, v) observing u only, with v
  inferred through ensemble covariances.
- **Expected outcome:** both filters are imperfect. The learned filter's error rises above its usual
  ~0.32 by the unpredictable part of v (B3 bounds it); the physics filter's rises by the whole εv
  effect. The learned filter is likely ahead, by less than in A. History-aware propagators should
  narrow its gap.
- **Implications for the paper:** the most realistic comparison, with error on both sides. Best
  placed in the DA follow-up paper together with history-aware propagators, unless the result is
  crisp and the paper has room. In Part II it would be one paragraph pointing forward. It directly
  motivates the CM1 study (Paper 3).

#### J17B progress (2026-10-10)
- **B2 code done** (`ks_latent/solver/ks2d_memory.py`; tests in `tests/unit/test_ks2d_memory.py`: ε = 0 equals
  the standard solver, free decay of v is exp(−t/τ), a tiny single mode matches the 2×2 matrix exponential
  for both couplings, batching, multi-field LETKF). Wired into `gen_ks2d_wide_dataset.py --memory`,
  `exp_h9_tiled_da.py --truth-memory`, `exp_i11_pde_letkf.py --truth-memory/--model-memory` (floor: the
  model carries v, initialised at 0, updated through its covariance with observed u),
  `exp_j16_estimate_pde_model_error.py --truth-memory`. Smoke-tested (4 cycles) on all three paths.
- **B1/B3 scan** (`j17b_scan.py`, 44², seed 3; τ = 2, C = ∇²). "One-cycle" = KS model's error (bias part):

  | ε, κ | u std | chaos (1e-6 → after 100 u) | rms εv / KS tendency | v autocorr. | R² v ~ local u | one-cycle |
  |---|---|---|---|---|---|---|
  | 0.05, +1 | 3.33 | 4.7 | 0.15 | 5.5 | 0.83 | 0.063 (0.040) |
  | 0.1, +1 | 2.93 | 0.13 | 0.33 | 6.0 | 0.84 | 0.114 (0.081) |
  | 0.3, +1 / 1, +1 | 1.43 / 0.03 | none | — | — | — | chaos killed (delayed diffusion) |
  | identity, +1 | 20–61 | none | — | — | — | runaway to a steady state |
  | 0.05, −1 | 4.29 | 5.3 | 0.11 | 4.5 | 0.80 | 0.079 (0.047) |
  | **0.1, −1** | 4.78 | 6.8 | 0.20 | 4.5 | 0.79 | **0.165 (0.095)** |
  | **0.2, −1** | 5.78 | 7.1 | 0.32 | 4.0 | 0.77 | **0.377 (0.203)** |

  κ < 0 (a delayed anti-diffusion: driving) stays strongly chaotic and reaches large, mostly
  state-dependent errors; v's memory is 4–4.5 time units (8–9 DA cycles); ~20% of v's variance is not
  linearly predictable from the current local u (the part a Markovian learned model cannot see).
  **Pre-registered levels (user go-ahead, 2026-10-10): (ε, τ, κ) = (0.1, 2, −1) and (0.2, 2, −1), C = ∇².**
- **Pre-registered decision rule (before any J17B DA run).** Physics filter (KS model, no v): best over
  tuning (RTPS × c), bias correction ± Q, ∇²-coefficient estimation (surrogate), and estimation + bias
  correction, selected on seed 0, then seeds 0–2. Latent filter: Part II settings, untuned, model
  `p7c12d10_memB<ε>` trained only on σ = 0.1 noisy u data. Same obs (σ = 0.1 × the new data std) and
  truths. **Latent wins** if its 3-seed mean is ≥ 10% lower and lower on every seed; mirror for a physics
  win; else tie. Floor: physics filter with the true (u, v) model, v unobserved.
- **Launched 2026-10-10 ~13:45:** `run_j17b_all.sh` (caffeinated).

**Effort:**
- Variant A: ~0.5 day of code, ~30 min of data, ~5 h of training, ~4 h of runs, plus ~5 h per extra
  strength level for A9.
- Variant B: ~1 day of code, ~30 min of data, ~5 h of training (+5 h per ablation), ~4 h of runs.
- Order: A1–A8 after J16; B after A.

**Order:**
1. J1 and J2 (runs, ~3–4 h).
2. J4 (lands ~15:05).
3. J7 and J6 (figures and tables).
4. J10.
5. J9, which includes J8.
6. J11, J12, J13, J14.

J5 runs overnight if wanted.

**Estimate:** ~4–6 h of machine time (plus J5 overnight) and ~4 days of writing.

---

## §F Figure contract (applies to every figure task)

- Produced by one script in `scripts/paper_figures/`, reading only `artifacts/` and writing
  `docs/paper1/figures/<name>.pdf` plus `<name>.meta.json` (inputs, git SHA, timestamp).
- Sizes: single column 3.37 in, double column 6.75 in; heights chosen so no figure exceeds
  60% of a page.
- Palette from `_style.py`: truth in neutral ink; 1D KS, 2D KS, Kolmogorov each a fixed hue
  across the whole paper; drag values distinguished by marker and lightness, never hue alone.
- Error bars: seed spread when several runs exist; a shaded ±15% band for single-run 2D
  truths; stated in every caption.
- Every panel: letter, axis labels with units, a legend or direct labels; log axes only for
  ratios or when the data span > 2 decades, with the floor stated.
- Space–time and field plots: `shading='flat'` with explicit edges, one shared colormap per
  figure, symmetric limits for signed fields, colorbar once.
- Captions say what is plotted and what the reader should see, in that order.
- Each figure is checked in grayscale and at 50% zoom before it is accepted.

---

## Suggested order of execution

Day 1: start A3, A4 (background CPU) and D0, D1, D2 (background); do A1, A2, A7, A8.
Days 2–4: B1–B9, C3, C5, C6, C8; D3, D4 (code), then queue D4–D9 runs.
Days 5–7: B10, B12, B13, B14, C1, C2, C4, C7; D5–D10 runs finish.
Week 2: E1–E6, F1–F4; B11; A5/A6 figures if not done.
Week 3: E7, E8, F5, G1–G6.
Week 4: G7 external read, G8.

Minimal path to a complete draft: A1, A3, A8, A10, B1–B10, B12–B14, C3, C5, C6, D0–D4, D6,
E1–E8, F1, F2, G1–G6. Everything else strengthens but does not block.
