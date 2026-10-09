# Critical review of paper 1: prompt

You are reviewing a research project that is about to become a journal paper. Your job is
to make the paper as strong and as honest as possible. Read the material below, check its
claims against the code and the result files, and write a single review document
(`docs/paper1_review.md`, format at the end) containing:

- whether the paper is framed correctly, and if not, how to reframe it;
- whether the results are significant, why, and to whom;
- connections to other work or other ideas that we are missing;
- results already in the repository that we are under-emphasizing or have overlooked;
- weaknesses a referee would attack, with the fix for each;
- concrete changes to the existing draft (text, structure, figures, claims);
- additional experiments or analyses worth running, prioritized by value per unit of compute.

Be critical. Praise is not useful here; precise, evidence-backed disagreement is. If a
claim in the draft is wrong, overstated, or unsupported, say so and point to the evidence.
If the paper's main story is not its best story, propose the better one.

---

## 1. The project in one paragraph

We build latent surrogates of spatiotemporally chaotic PDEs whose latent state is a
*field* (c channels per site on a lattice), built only from weight-shared, finite-reach
operators. Because the parameters do not depend on the number of sites, one trained model
can be deployed zero-shot on larger domains by adding sites. We test whether such models
reproduce extensive chaos (Kaplan–Yorke dimension D_KY growing with domain size) in three
systems — 1D Kuramoto–Sivashinsky (KS), 2D KS, and 2D Kolmogorov flow with linear drag —
validating against true Lyapunov spectra. We train 2D models on *windows (crops) of one
large domain* with a forced halo, use a *per-patch* autoencoder (zero encoder/decoder reach)
whose decoder Jacobian is exactly block-diagonal, and use the resulting exactly local
observation operator for localized ensemble data assimilation (DA) in latent space. The
target venue is *Chaos* (AIP).

---

## 2. What to read, in this order

All paths are relative to the repository root (`/Users/daltonjones/Documents/latent_DA`).

1. **`docs/paper1_outline.md`** — the plan for the paper: claims C1–C6, section outline,
   figure list, literature plan, decisions log, and the open work table. Read the
   "Decisions log" carefully; it records choices the author has already made.
2. **`docs/paper1/main.tex`** and the written sections it inputs:
   `sections/03_method.tex`, `04_systems.tex`, `05_results_1d.tex`, `06_results_2d.tex`,
   `appendix/B_proofs.tex`, `appendix/C_ground_truth.tex`, plus `macros.tex`, `refs.bib`.
   Figures are in `docs/paper1/figures/` (scripts in `scripts/paper_figures/`). Compile
   with `latexmk -pdf main.tex` in `docs/paper1/` if you want to see the typeset draft.
   Sections 1, 2, 7–10 and Appendices A, D–F are not written yet.
3. **`docs/paper1_tasks.md`** — the task list and the detailed history of recent
   experiments (Kolmogorov α=0.1 training variants v1–v7, truth re-runs, DA bug fixes).
4. **`docs/ks_local_field_ltransfer.tex`** (~9,800 lines) — the full research record. It
   is chronological and contains many hypotheses, dead ends and superseded readings.
   Treat it as a lab notebook, not as settled conclusions: later sections often revise
   earlier ones, and several earlier numbers have been corrected (see §4 below). The
   sections most relevant to the paper are: the 1D results and experiments E1–E7 (lines
   ~233–1150), "Training on crops of a large domain" (~3420+), C19 and Phase 0 (~4170,
   ~4955–5170), P7 (~5490–5850), the publication plan and consolidation plan (~7580–8490),
   the DA hypotheses section (`sec:c19_da_hypotheses`, ~7720–8300), the rescoring and
   ablation sections (~8740–9340), threats to validity and related work (~9440–9560).
   Ideas the author has set aside for a second paper are in "Transfer error as a physics
   diagnostic" (~2189), the residual hierarchy (~2841), and the SWE section (~9345).
5. Supporting documents, skim as needed: `docs/RESULTS.md` (long 1D history),
   `docs/OPEN_QUESTIONS.md`, `docs/LITERATURE_REVIEW_AND_FINDINGS.md`,
   `docs/spatially_structured_latent_DA_FMM_research_idea.md` (a recent multiscale idea),
   `docs/advisor_summary.tex`, `docs/oct_1_exp.md`, `docs/REPLICATION_LOG.md`.
6. **Code**, when you need to check a claim: models in `ks_latent/models/`
   (`autoencoder_local_field_2d.py`, `propagator.py`), crop training in
   `ks_latent/training/crop.py` and `scripts/train_kolmogorov_crop.py`, DA in
   `ks_latent/da/` (`letkf.py`, `letkf_local.py`, `jacobian_coloring.py`, `cycling.py`,
   `geometry2d.py`, `localization.py`, `pff.py`, `sec.py`), Lyapunov code in
   `ks_latent/analysis/lyapunov.py`, solvers in `ks_latent/solver/`, evaluation scripts
   `scripts/exp_crop_eval.py`, `scripts/exp_crop_eval_ks2d.py`,
   `scripts/run_da_pff_2d_ltransfer.py`, `scripts/run_truth_kolmogorov.py`. Tests are in
   `tests/`.
7. **Results**: `artifacts/experiments/` — crop evaluations in `crop/*_x{1,2,3}_seed*.json`
   (field `model_D_KY`), truths in `crop/truth_*.json` and the `true_dky_*`, `e1_*`,
   `g1_*` files, box-trained baselines in `icnorm/` and `c4/`, 1D DA sweeps in
   `e2_*.csv`, `e3_*.csv`, `e7_*.csv`, and the re-run 2D DA in `da2d/` (summarize with
   `python3 scripts/summarize_da2d.py`). Truth values used in the paper are in Table II
   of `sections/04_systems.tex`.

---

## 3. Decisions already made (challenge only with strong evidence)

- Lead the quantitative D_KY claim with 1D KS, where the truths are accurate to about 1%.
  For Kolmogorov flow, claim that the models reproduce the *scaling* of D_KY with domain
  size within the truth's measured uncertainty, not single-point matches.
- Existing truth runs are the reference (means where several exist; matched initial
  conditions at α=0.05, 1×). More truth runs may be added later.
- The main-text Kolmogorov α=0.1 model is `k_xs01_v5`; α=0.05 is `c19_e0d0_c6`;
  α=0.025 is `k_xs0025`; 2D KS is `p7b_ks2d`.
- Out of scope for paper 1 (planned for paper 2): transfer error as a physics diagnostic,
  the drag sweep toward α=0, the residual hierarchy, self-similarity, shallow-water
  equations, Kolmogorov DA, per-patch vs full-reach decoder in DA.
- Venue: *Chaos*. Negative results are reported, not hidden. Decision rules are
  pre-registered before experiments where possible.
- Never propose the banded latent-index smoothness regularizer (`RegConfig.lambda_z`,
  CLI `--lambda-z`); it is known to collapse the latent.

If you think one of these decisions is wrong, say so in a clearly marked section and give
the evidence; do not silently write the review as though it were different.

---

## 4. Corrections and bugs found recently (do not re-derive; do check their consequences)

- **1D truth at L=100 is 22.5**, not 22.14 (the record's E1 table put the model value in
  the truth column; now corrected). Model error at the trained size is 1.6%.
- **40-unit Kolmogorov truths are not converged.** 120-unit continuations of the same
  trajectories move by −17% to +23% from their 40-unit values (4 of 5 high, mean +10%).
  Single-seed 40-unit truths at 1× and 3× may be biased upward, which would make models
  appear to under-estimate D_KY. See `appendix/C_ground_truth.tex`.
- **The 2D DA runs before 2026-10-08 are invalid as model measurements.** The DA script
  deployed the 2D KS model at 5× its training resolution (sites 1.1 units wide instead of
  5.5). Fixed (`ks_latent/da/geometry2d.py`); the old outputs are quarantined in
  `artifacts/da2d_INVALID_resolution_bug/`. All RMSE/skill/calibration numbers in the
  record's `sec:c19_da_hypotheses` T1–T5 and in `paper1_tasks.md` waves E1/E4 are invalid.
- **Two autograd memory bugs in the DA filters** (graphs retained across cycles, and
  within one analysis across sites) inflated every memory figure; fixed and verified
  numerically inert (per-cycle RMSE identical to 3e-8). Memory numbers from before the
  fix, including the "dense filter killed at 88²" claim, must not be used.
- **Observation spacing must resolve the 2D KS pattern scale (~8.9).** At 11-unit spacing
  DA is worse than the free run; at 2.75 units skill is ~4. The re-run uses 2.75 units.

### Corrected 2D DA results so far (2D KS, `p7b_ks2d`, 100 cycles, steady state = cycles 51–100)

| | L=44 (d=384) | L=66 (d=864) | L=88 (d=1536) | L=220 (d=9600) | L=330 (d=21600) |
|---|---|---|---|---|---|
| per-site local filter, N=64, skill | 4.23 (seed 1: 4.30) | 4.08 (3.93) | 4.00 (3.96) | 3.84 | 3.86 |
| dense + Gaspari–Cohn, N=64 | 3.65 | 3.54 | 3.47 | 3.40 | — |
| dense, no localization, N=64 | 2.31 (2.54) | 0.94 (0.98) | 0.83 (0.86) | — | — |

Further: dense+GC saturates by N=16 (~3.5–3.8 at every L); without localization N=128
reaches 3.5 at L=44 but only 1.1 at L=88; every configuration is under-dispersed
(spread/RMSE ≈ 0.25–0.46), and additive model noise does not fix it (noise sweep at L=44:
0.02/0.04/0.08/0.16 → skill 4.93/4.79/4.23/3.40, spread/RMSE 0.25/0.31/0.35/0.33). L=440
and a clean memory re-measurement are still running; check `artifacts/experiments/da2d/`
for whatever has landed. The 1D DA results (record lines ~312–1146; files `e2/e3/e7`)
are not affected by these bugs.

---

## 5. Questions to answer

Work through all of these; the review should say which ones matter most.

**Framing and significance**
1. What is the single most important result in this body of work? Is it what the
   outline currently leads with (zero-shot transfer of extensive chaos), or something
   else (e.g. "the training distribution decides transfer", exact locality of latent DA,
   a fixed-size ensemble staying skilful while D_KY grows ~57×, Lyapunov-level validation
   itself)? Propose the title and abstract-level framing you would use.
2. Who is the audience (nonlinear dynamics, scientific ML, data assimilation /
   geophysics), and does the current structure serve them? Is *Chaos* the right venue?
3. Which results are genuinely new relative to the closest prior work (parallel local
   reservoir computing, Pathak et al. 2018; patch-decomposed autoencoders with neural
   ODEs, Constante-Amores, Linot & Graham 2024; zero-shot transformer neural operators on
   larger domains, arXiv:2606.14597; latent ensemble filters such as ROAD-EnKF and
   LAE-EnKF; Wittenberg & Holmes 1999 on localized models of KS; Patil et al. 2001 and
   Ott et al. 2004 on local low dimensionality)? Which claims would a referee say are
   known? Search the literature if you can, and say what you checked. Is there any other work our results are connected to that should be highlighted?

**Correctness and rigor**
4. Spot-check the numbers in the written sections against the result files and code.
   Report any number that does not reproduce.
5. Are the propositions and the lemma in §3 / Appendix B correct, correctly scoped, and
   worth including? Is anything stated as a theorem that is really a heuristic, or vice
   versa? Is there a stronger or more useful statement available (for example about
   extensivity of local models, or the localization radius)?
6. Is the evaluation methodology sound: finite-time D_KY from a few initial conditions,
   tangent directions vs. bracket, comparing a learned map's exponents (per snapshot step)
   with the PDE's, the role of the gauge-anchored channel, initial-condition
   normalization? What would a careful referee demand?
7. Is the treatment of truth uncertainty adequate, and is "match the scaling" a defensible
   claim given the data? What is the minimal additional truth computation that would
   most strengthen the Kolmogorov results?

**Missed results and connections**
8. Search the record, `RESULTS.md`, `OPEN_QUESTIONS.md` and the artifacts for results that
   deserve a place in the paper but are not in the outline. Candidates to evaluate
   (not an exhaustive list): the learned local stencil closure and sparse selection
   (H3, "Section 224/225"); the 1D Jacobian-bandedness diagnostic; the observation that
   models amplify the true initial-condition dependence at native size; the C8/C16
   unstable-subspace scale analysis; transfer error tracking the truth's deviation from
   extensivity (G2, P1/P2); the k_max × log-det-barrier interaction (`sec:k_xs01_ablation`);
   the observation-spacing threshold in 2D DA; the per-site vs dense filter behaviour at
   small ensembles; the under-dispersion of the corrected 2D DA.
9. What connections to other fields are we missing (e.g. renormalization / coarse-graining,
   the thermodynamic limit of Lyapunov spectra and coupled-map lattices, information
   propagation speed and comoving exponents, limited-area weather prediction, fully
   convolutional training on crops, local ensemble transform Kalman filters at operational
   scale, invariant-measure-preserving training for chaotic surrogates)? Which would
   change what we claim or how we explain it?

**Weaknesses and what to do about them**
10. List the strongest objections a referee could raise, ranked by severity, each with
    the cheapest convincing response (text change, re-analysis of existing data, or a
    new run).
11. Are there internal inconsistencies between the outline, the written sections, the
    task list and the record (different numbers for the same quantity, models described
    with the wrong recipe, claims that a later section contradicts)?

**Additional work**
12. Propose additional experiments or analyses. For each give: the question it answers,
    what result would change the paper and how, an honest compute estimate on a single
    Apple-silicon laptop (48 GB RAM; training runs are 20–60 min on MPS; a 2D truth
    Lyapunov run is 0.5–10 h on CPU), and a pre-registered decision rule. Prefer analyses
    of existing artifacts over new training.

---

## 6. How to work

- **Verify, don't trust.** The record contains superseded readings. Before citing a
  number, find it in a result file or recompute it; before describing what code does,
  read the code. In the review, mark every substantive claim as **[verified: file/line or
  command]** or **[inferred]**.
- You may run short, read-only analyses on existing artifacts (Python with
  `mamba run -n da_env python ...`; anything under ~10 minutes). Put scratch scripts in a
  scratch directory, not in the repository.
- **Do not** start training runs, truth runs or DA runs, and do not kill or interfere with
  running processes (a DA queue, `scripts/run_da2d_rerun.sh` / `run_da2d_memfix.sh`, may be
  running). Propose such runs in the review instead.
- **Do not edit** the paper, the record, the outline, the task list, code or results.
  Your only output is `docs/paper1_review.md`. Suggested text changes go in the review as
  quoted before/after passages with the file and approximate line.
- Do not invent citations. If you cite a paper, give enough detail to find it and say
  whether you checked it; flag anything unverified.
- Prefer depth on the few issues that matter most over breadth on many small ones, but
  do list the small ones too (in their own section).

---

## 7. Output: `docs/paper1_review.md`

Use this structure:

1. **Executive summary** (≤ 1 page): the 5–8 most important findings and recommendations,
   in priority order.
2. **Assessment of significance**: what is new, what is not, why it matters and to whom;
   your recommended framing (title, one-paragraph abstract sketch, list of contributions).
3. **Framing and structure**: recommended changes to the paper's story and section order,
   with reasons. Include a revised outline if you recommend restructuring.
4. **Correctness issues**: numbers that do not reproduce, overstated or unsupported claims,
   problems with the propositions/lemma, methodological concerns. Each with evidence and
   a fix.
5. **Under-emphasized or missing results**: each with where it lives (file/section), why
   it matters, and where it should go in the paper.
6. **Missing connections and literature**: each with how it changes the paper.
7. **Anticipated referee objections**: ranked, each with the recommended response.
8. **Proposed additional work**: a table with columns — item, question answered, expected
   impact on the paper, compute estimate, decision rule, priority (must / should / nice).
9. **Concrete edits to the written sections**: before/after passages for §3–§6 and the
   appendices, grouped by file.
10. **Minor issues**: typos, notation, figure details, small inconsistencies.
11. **Disagreements with recorded decisions** (if any): what, why, evidence.

Tag every recommendation with an impact (high / medium / low) and an effort (hours /
days / requires new compute). End with a short list of the open questions you could not
resolve and what would resolve them.
