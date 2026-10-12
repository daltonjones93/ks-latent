# Splitting `docs/paper1` into two papers — atomic tasks

Written 2026-10-11. `docs/paper1/` (≈30,000 words, 19 figures, 21 tables, six appendices) is kept
untouched as the reference. Two new, self-contained papers are built from it:

| | Working name | Directory | Content | Audience / journal (decision S0.2) |
|---|---|---|---|---|
| **Paper I** | *Local latent fields for extensive spatiotemporal chaos: zero-shot domain transfer validated by Lyapunov spectra* | `docs/paperI/` | the model, window training, transfer in 1D and 2D, failure modes (old Part I: §1–6, §8, App. A–D, F) | physics / nonlinear dynamics (Chaos, the stated target) |
| **Paper II** | *Exactly local ensemble data assimilation in a learned latent field* (working) | `docs/paperII/` | the filter, shifted tilings, one model from 44² to 440², learned vs physics-based filters, noisy training (old Part II: §3.8, §7, App. E) | data assimilation (candidates: QJRMS, MWR, JAMES, NPG) |

Naming: the DA follow-up on sparse-observation stability stays "Paper 2" and the CM1 squall-line
study "Paper 3" (`full_paper_tasks.md`); the split papers are "Paper I" and "Paper II", matching the
Part I / Part II language already in the text.

Conventions: one task = one commit-sized unit with an input, an output and a check. Effort is for one
person. `[ ]` open, `[~]` in progress, `[x]` done. Numbers in the new papers are never typed from
memory: every table body is `\tabinput` of a generated file and every figure comes from a script.

---

## S0 Decisions (user; before any writing)

- **S0.1 `[x]` Confirmed 2026-10-11 (user). The split line.** Paper I = dynamics and transfer; Paper II = data assimilation.
  The per-patch decoder (§6.3) stays in Paper I as an architecture result with one forward pointer;
  Proposition 3 (block-diagonal Jacobian) and its proof move to Paper II. ~5 min.
- **S0.2 `[x]` Target journals and formats (decided 2026-10-11).** Paper I: Chaos. Paper II: **JAMES**
  (AGU): the scope is learned models in Earth-system modelling and DA, the AGU class is already in
  use, there is no word cap, and the CM1 follow-up will address the same readership; QJRMS is the
  fallback. Both drafts keep the current class; the length budgets below still apply. Original text: Paper I: Chaos (REVTeX `aip,cha`; the AGU class was kept
  only for drafting) or keep AGU style. Paper II: pick one of QJRMS, MWR, JAMES, NPG; this fixes
  length (MWR ≈ 7,500 words main text; QJRMS no hard limit, ~8,000 typical; JAMES none), figure count,
  citation style and whether an appendix or supplement holds the details. ~15 min. Until decided the
  drafts use the current AGU class with a length budget of ≤ 9,000 words of main text and ≤ 8 figures each.
- **S0.3 `[x]` (a) chosen 2026-10-11. The 1D DA results (§7.7, Fig. 8, App. E.5).** They use the 1D model, a single tiling and
  different filters (PFF, stochastic EnKF, SEC). Options: (a) a short subsection + appendix in Paper II
  (recommended: the "n₊ doubles with the domain" and "GC beats SEC" results are the clearest
  evidence that a latent with a geometry is what makes localization work); (b) appendix only; (c) drop.
- **S0.4 `[x]` Agreed 2026-10-11. Submission order and cross-citation.** Paper II cites Paper I for the model's transfer
  validation. Recommended: finish Paper I first and post it to arXiv so Paper II can cite a preprint;
  Paper II is otherwise written to stand alone (its own model description, its own validation table).
- **S0.5 `[x]` Kolmogorov flow in Paper II (decided 2026-10-11): latent filter only**, with a
  noisy-trained 12-channel model, no physics-based comparator. Block SK below.
- **S0.6 `[x]` Pending results that Paper II waits for:** J17G (P7, running), J5 (P9, running), the
  plain-language-summary sentence (removed 2026-10-11). Paper II's §5 is written against P7's outcome.

---

## S1 Infrastructure (so both papers build from the same results without touching `paper1`)

- **S1.1 `[x]` Figure output directory per paper.** `scripts/paper_figures/_style.py`: `OUT` from
  the environment (`PAPER_OUT`, default `docs/paper1/figures`); `save()` unchanged otherwise.
  Check: regenerating any paper1 figure with the default still writes to `docs/paper1/figures`. ~20 min.
- **S1.2 `[x]` Table output directory per paper.** Same for every `tab*.py` (`PAPER_TABLES`, default
  `docs/paper1/tables`); sidecar `.meta.json` written next to the table. ~30 min.
- **S1.3 `[x]` Figure numbering per paper.** Figure scripts save under their *content* name
  (`concept`, `systems`, …); each paper's `main.tex` includes by content name, so the same script
  serves both papers without renumbering; the "figNN_" prefixes are dropped in the new directories
  (keep them in `paper1`). Check: `ls docs/paperI/figures` shows content names only. ~30 min.
- **S1.4 `[x]` `check_numbers.py --paper paperI|paperII`.** Paths from the argument; the Part II strict
  class applies to Paper II's sections; writes `docs/<paper>/number_check.md`. ~20 min.
- **S1.5 `[x]` Skeletons.** `docs/paperI/{main.tex,macros.tex,refs.bib,sections/,appendix/,figures/,tables/}`
  and the same for `paperII`; `macros.tex` and `refs.bib` copied (not symlinked) so each paper can
  diverge; `PJ2019.cls` copied. `main.tex` compiles empty with the class. ~20 min.
- **S1.6 `[x]` Build and check targets.** `make paperI`, `make paperII` (latexmk + `check_numbers`
  + undefined-reference grep), `make paper1` unchanged. ~20 min.
- **S1.7 `[x]` Decision log.** `docs/paperI/decisions.md` and `docs/paperII/decisions.md` start with a
  pointer to `paper1/decisions.md` and record only split-time decisions. ~10 min.


**S1 done 2026-10-11.** What exists now:
- `_style.py`: `OUT`/`TABLES` from `PAPER_OUT`/`PAPER_TABLES` (default paper1), `content_name()`
  (prefix and `p2_` stripped outside paper1, overrides for vague names: `fig09_p2_da` → `da_2d`,
  `tabD_*` → `eval_*`, `tab02_truth` → `truth_dky`), and `write_table()`, which every table script
  now uses (tex + `.meta.json` with SHA, time, inputs). All 25 paper1 table bodies were byte-identical
  before and after the refactor; 15 tables gained a sidecar they lacked.
- `check_numbers.py --paper paper1|paperI|paperII`; EXPLAINED entries are per paper (carry them
  over when a number moves); the strict class covers every file of paperII.
- Skeletons `docs/paperI` (sections 01 introduction … 07 failure modes, 08 discussion,
  09 conclusions; App. A–E) and `docs/paperII` (01 introduction … 05 tilings and sizes,
  06 physics comparison, 07 noisy training, 08 forecasts/cost, 09 one dimension, 10 Kolmogorov,
  11 discussion, 12 conclusions; App. A–E). Every stub names its task and its paper1 source, and
  renders a red `\placeholder`. Stubs carry no `\label`, so moved text brings its own labels.
- `make paperI|paperII|paper1` (latexmk, number check, undefined/duplicate-reference check,
  count of placeholders left) and `make paperI-assets|paperII-assets` (tables, then figures, into
  the paper's directories; asset lists in the Makefile). Assets regenerate in ~35 s for both.
  Figure 1 is still the full four-panel version in both until S3.8/S4.17.
- `tests/unit/test_paper_paths.py`: content names, no collisions, env redirect, sidecar.
- `docs/paperI/decisions.md`, `docs/paperII/decisions.md`.

---

## S2 Content map (what goes where; the reference for every task below)

Source → destination. "Rewrite" = new prose with the same facts; "move" = copy then edit for context.

| paper1 source | words | Paper I | Paper II |
|---|---|---|---|
| main.tex: title, key points, abstract, PLS | 780 | rewrite (S3.1) | rewrite (S4.1) |
| §1 Introduction | 1,340 | rewrite: drop para 2 (DA motivation) to one sentence, drop the two DA bullets, drop "two parts" para (S3.2) | rewrite from scratch (S4.2) |
| §2.1 Extensive chaos | | move | one paragraph in §2 (density ρ, n₊ grows with the domain) |
| §2.2 Learned surrogates | | move | two sentences (cite Paper I) |
| §2.3 Latent and localized DA | | drop | move, expand slightly (S4.3) |
| §2.4 What is new + Table 1 | | rewrite without the "local DA" column (S3.3) | new prior-work table with DA columns (S4.3) |
| §3.1 Setting and notation | | move | compact version in §2/§3 (S4.4) |
| §3.2 Local latent fields, Props 1–2, Cor., Remark | | move | one-page restatement of the architecture (no proofs), cite Paper I (S4.4) |
| §3.3 Receptive fields | | move | two sentences (reach of the 12-channel model) |
| §3.4 Window training, Prop 3 | | move | one paragraph (what the Paper II model was trained on) |
| §3.5 Objective, Lemma, remarks | | move | one paragraph: the noisy-training objective (S4.4) |
| §3.6 Canonical recipe | | move | the Paper II recipe as a table row (App. A') |
| §3.7 Evaluation | | move | the D_KY check of the Paper II model only |
| §3.8 Latent ensemble DA, Prop 3 | 1,300 | drop (one forward pointer in §6.3) | move → Paper II §3 "Method" (S4.5) |
| §4 Systems (three), Table 2, Fig. 2 | 1,730 | move | 2D KS only (eq., parameters, truth D_KY and n₊); 1D KS if S0.3 = (a) (S4.6) |
| §5 1D transfer, Figs 3–4, Table 3 | 930 | move | — |
| §6 2D transfer, Figs 5–7, Table 4 | 2,280 | move; §6.3 per-patch paragraph keeps one forward pointer (S3.5) | — |
| §7.1 Setting | | — | move (S4.7) |
| §7.2 Shifted tilings, Fig. 13, Table members | | — | move (S4.8) |
| §7.3 Accuracy at native size | | — | move (S4.8) |
| §7.4 44²→440², Fig. 9, Fig. 11, Table sizes | | — | move (S4.9) |
| §7.5 vs physics, Fig. 14, Table model error | | — | move; P7/P9 (S4.10) |
| §7.6 Noisy training | | — | move (S4.11) |
| §7.7 Forecasts/robustness/density, Fig. 16, Tables | | — | move (S4.12) |
| §7.8 1D results, Fig. 8 | | — | per S0.3 (S4.13) |
| §7.9 Cost, Table cost | | — | move (S4.12) |
| §8 Failure modes, Fig. 10, Table 8 | 1,150 | move | — |
| §9 Discussion paras 1–4 (transfer, reach, parameters, probe) | | move (S3.6) | — |
| §9 paras 5–6 (locality in DA, learned vs physics) | | — | move (S4.14) |
| §9 Limitations | | items 1–5 | item 6, expanded (S4.14) |
| §9 Outlook | | rewrite: dynamics only | rewrite: DA / CM1 (S4.14) |
| §10 Conclusions | 500 | paras 1, 2, 4 (S3.7) | para 3 expanded (S4.15) |
| App. A Architecture, Table A | | move; drop the 12-channel column | App. A': 12-channel column + noisy training + filter settings (S4.16) |
| App. B Proofs 1–2 + Lemma | | move | App. B': proof of Prop. 3 only (S4.16) |
| App. C Ground truth | | move | one subsection: the 2D KS truths used (44², 66²; n₊), cite Paper I for the protocol (S4.16) |
| App. D Tables, Fig. C2 | | move (drop the 12-channel rows? keep: they document the DA model's dynamics) | the 12-channel model's rows as a validation table (S4.16) |
| App. E DA details, Fig. 12 | 2,180 | — | move (S4.16) |
| App. F Reproducibility | | move, minus "Part II code" | "Part II code" + a short reproducibility section (S4.16) |
| Fig. 1 concept (a–d) | | panels a–c (S3.8) | panel d + tilings schematic → new Fig. 1 (S4.17) |

Length after the split (main text, estimate): Paper I ≈ 11,500 words → trim to ≤ 9,000 (S3.9);
Paper II ≈ 9,500 (7,000 from §7 + method + intro) → trim to ≤ 8,500 (S4.18).

---

## S3 Paper I (`docs/paperI/`)

- **S3.1 `[x]` Front matter.** Title (drop "and exactly local ensemble data assimilation"); three key
  points (replace KP3 by a failure-modes or selection-rule point); abstract ≤ 250 words without the
  DA sentences; PLS without the DA sentences. Check: no `\ref` to a DA label. ~1 h.
- **S3.2 `[x]` §1 Introduction.** Keep paras 1, 3, 4 (why a field; three things that matter → two);
  reduce the DA paragraph to one sentence ("a companion paper [Paper II] builds a localized ensemble
  filter on this latent"); contributions list without the two DA bullets; drop the "two parts"
  paragraph. Target ≤ 1,000 words. ~1.5 h.
- **S3.3 `[x]` §2 Background.** §2.1–2.2 moved; §2.3 dropped; §2.4 rewritten with Table 1 minus the
  "local DA" column and minus the DA-only rows (chen2023roadenkf, tong2026laeenkf, brajard2020,
  bocquet2020 unless kept for the Lyapunov column). Regenerate the table by hand (it is static). ~1 h.
- **S3.4 `[x]` §3 Method without §3.8.** Move §3.1–3.7; renumber; the first paragraph lists
  Secs. 5–7 as the results; delete every "Part I"/"Part II" phrase (26 occurrences across the paper,
  `grep -n "Part~I"`). Check: builds; Props 1–2, Cor. 1, Lemma 1 numbered 1–2, 1, 1. ~1 h.
- **S3.5 `[x]` §4–§6 and §8 moved.** §4 whole; §5 whole; §6 whole with the per-patch paragraph's last
  sentence changed to "and, as the companion paper shows, makes the latent observation operator
  exactly local"; §8 whole. Fix cross-references (`sec:da_results` → a `\citet{paperII}` entry in
  `refs.bib` with "submitted"). ~1 h.
- **S3.6 `[x]` §8 Discussion (old §9).** Paras 1–4; limitations items 1–5; outlook rewritten for the
  dynamics (other extensive systems; forecasting; what a learned model's transfer error says about
  extensivity). Drop "What locality buys in DA" and "When to prefer a learned model". ~1 h.
- **S3.7 `[x]` §9 Conclusions (old §10).** Paras 1, 2 and the closing rule; the DA paragraph replaced by
  one sentence pointing to Paper II. ~30 min.
- **S3.8 `[x]` Figures.** Fig. 1 → panels (a)–(c) only (`fig01_concept.py` gains `--panels abc`);
  Figs. 2–7, 10, C1, C2 regenerated into `docs/paperI/figures` under content names; renumber
  references. Check each at natural size; captions no longer mention Part II. ~1.5 h.
- **S3.9 `[x]` (9,609 words; see paperI/decisions.md) Length pass.** Target ≤ 9,000 words main text. Candidates: §2.2 (surrogates) −300;
  §3.5 remarks −200; §4.4 (truth precision) → App. C −400; §6.3 −300; §8 −200. Record the cuts in
  `paperI/decisions.md`. ~2 h.
- **S3.10 `[x]` Appendices.** A (Table A without the 12-channel column; note that the DA model is in
  Paper II), B (Props 1–2, Lemma), C, D (keep the 12-channel rows with a footnote "the model of
  Paper II"), F (minus "Part II code"; compute paragraph without the DA timings). ~1 h.
- **S3.11 `[x]` Bibliography.** `refs.bib` pruned to the entries cited; add the Paper II entry. Check:
  `bibtex` warnings zero; no DA-only references remain unless cited. ~30 min.
- **S3.12 `[x]` Number check.** `check_numbers.py --paper paperI`: zero UNMATCHED; every table body
  generated. ~30 min.
- **S3.13 `[x]` Read-through for self-containment.** A reader who has not seen Paper II must not meet
  an undefined object: grep for "filter", "assimilation", "tiling", "observation" and confirm each
  occurrence is motivated locally or points to Paper II. ~1 h.
- **S3.14 `[x]` Journal formatting (after S0.2).** If Chaos: switch to REVTeX `aip,cha,reprint`
  (the 2026-10-08 `aipsamp` port exists in git history), lead paragraph, figure widths 3.37/7.0 in.
  ~2 h.

---

## S4 Paper II (`docs/paperII/`)

- **S4.1 `[x]` Front matter.** Title; three key points (tilings; one model and one setting from 44²
  to 440²; the crossover against physics-based filters, phrased after P7); abstract ≤ 250 words
  that states the model in one sentence and cites Paper I; PLS (restore the learned-vs-physics
  sentence only from J17G, per the user's 2026-10-11 instruction). ~1.5 h.
- **S4.2 `[x]` §1 Introduction (new).** Order: (i) ensemble DA needs localization, and in extensive
  chaos the number of unstable directions grows with the domain (one paragraph, with the Lorenz-96
  and n₊+1 citations from old §1–2); (ii) latent DA with a global vector has no distance, so
  localization is unavailable and recent latent filters omit it (old §1 para 2 + §2.3's second
  paragraph); (iii) the local latent field of Paper I in three sentences, with its transfer
  validation quoted (D_KY within 0.6–2% in 2D KS); (iv) what this paper shows: exact locality of the
  observation operator, shifted tilings, size invariance, the comparison with physics-based filters,
  noisy training; (v) scope: twin experiments in 2D KS; the CM1 target as motivation (one sentence).
  Contributions as a bulleted list (old §1's two DA bullets, updated by P7). Target ≤ 1,100 words. ~2 h.
- **S4.3 `[x]` §2 Background.** Old §2.3 moved and slightly expanded (what LETKF does; RTPS,
  parameter estimation, bias correction as the remedies; cycle spinning); new prior-work table with
  columns: learned model / latent DA / localized / other sizes / compared with a physics-based filter;
  rows: chen2023roadenkf, tong2026laeenkf, peyron2021, pasmans2025, bach2025, si2025latentensf,
  brajard2020, bocquet2020, this work. ~1.5 h.
- **S4.4 `[x]` §3.1 The local latent field (restated).** One page: lattice, channels, encoder/decoder
  per patch, propagator with reach, equivariance (statement only, "Paper I, Prop. 1"), window
  training (one paragraph), the noisy-training objective (one paragraph), the 12-channel model's
  recipe (table row in App. A'). No proofs. Include Fig. 1 (S4.17). ~2 h.
- **S4.5 `[x]` §3.2–3.5 The filter (old §3.8 moved).** Per-site LETKF; Proposition 1 (block-diagonal
  Jacobian; old Prop. 3) with proof in App. B'; shifted tilings and Eq. (blend); the N_ens argument;
  the physics-based comparator. Renumber: Prop. 3 → Prop. 1. ~1 h.
- **S4.6 `[x]` §4 System and ground truth.** 2D KS only: equation, parameters at 44²–440², ETDRK4 in
  one sentence, the truth's D_KY and n₊ at 44² and 66² with their spread (from old §4.3, §4.4, App. C.4),
  and a validation table of the Paper II model (D_KY error 3.6% / 2.1%, reconstruction error,
  free-rollout statistics; from App. D's 12-channel rows). Figure: the 2D KS panel of old Fig. 2 as a
  one-column figure. If S0.3 = (a), a short paragraph on 1D KS and its model. ~1.5 h.
- **S4.7 `[x]` §5.1 Setting (old §7.1).** Move; replace "Part I" by "Paper I"; the model paragraph
  shortens because §3.1 now describes it. ~30 min.
- **S4.8 `[x]` §5.2–5.3 Tilings and accuracy (old §7.2–7.3).** Move with Fig. (tilings) and the
  members table. ~30 min.
- **S4.9 `[x]` §5.4 One model from 44² to 440² (old §7.4).** Move with Fig. (da_2d), Fig. (snapshot)
  and the sizes table; the "without localization" paragraph stays here. ~30 min.
- **S4.10 `[x]` §6 Learned versus physics-based filters (old §7.5).** Move; fill P7 (J17G) and P9 (J5)
  when they land; regenerate the model-error table and Fig. (model_error) with the J17G rows; the
  J17B paragraph stays as the non-Markov case (the 2026-10-11 framing). ~2 h after the runs.
- **S4.11 `[x]` §7 Trained only on noisy data (old §7.6).** Move; the round-trip appendix with it. ~30 min.
- **S4.12 `[x]` §8 Forecasts, robustness, observation density, cost (old §7.7, §7.9).** Move; Fig.
  (forecasts), the robustness and cost tables. ~45 min.
- **S4.13 `[x]` The 1D results (old §7.8, Fig. 8, App. E.5).** Per S0.3: (a) §9 "The one-dimensional
  case" + App.; (b) App. only with two sentences in §5.4; (c) drop and delete the SEC background. ~1 h.
- **S4.14 `[x]` §10 Discussion.** Old §9 paras "What locality buys" and "When to prefer a learned
  model"; limitations item 6 expanded into a list (twin experiments; one system; dense regular
  observations; chosen model errors; complete noisy training fields; Kolmogorov not assimilated;
  sparse-observation stability → Paper 2); outlook: CM1 (Paper 3), limited-area models, non-periodic
  domains (I27), model-error statistics (I22). ~1.5 h.
- **S4.15 `[x]` §11 Conclusions.** Old §10 para 3 expanded to three short paragraphs: exactness and
  tilings; size invariance and the ensemble; the crossover. ~45 min.
- **S4.16 `[x]` Appendices.** A' settings (12-channel model, noisy training, filter and comparator
  settings; from old Table A's last column and App. E.2–E.3); B' proof of Prop. 1 (old App. B.4);
  C' the 2D KS truths used (short; cite Paper I); D' old App. E.1, E.4, E.6, E.7 and (per S0.3) E.5;
  E' reproducibility ("Part II code" + numerics/seeds in short + compute). ~2 h.
- **S4.17 `[x]` Fig. 1 (new).** Panels: (a) the latent field and the per-patch decoder; (b) the
  block-diagonal observation Jacobian (old Fig. 9a, measured); (c) the four shifted tilings and the
  Hann weights (old Fig. 13a–b). Reuse `fig01_concept.py` panel (d) and `fig13_tilings.py`; one new
  script `fig_concept_da.py`. ~2 h.
- **S4.18 `[x]` Length pass.** Target ≤ 8,500 words main text. Candidates: §5.3 Gaussianity → App.;
  §8 observation density → App. with one sentence; the cost section → one paragraph + table. ~1.5 h.
- **S4.19 `[x]` Figures regenerated** into `docs/paperII/figures` under content names: tilings, da_2d,
  snapshot, model_error, forecasts, (1d_da), jacobian_timing, systems_ks2d. Captions without
  "Part I/II". ~1 h.
- **S4.20 `[x]` Bibliography.** `refs.bib` pruned to the entries cited; add Paper I (arXiv id after
  S0.4). ~30 min.
- **S4.21 `[x]` Number check.** `check_numbers.py --paper paperII` with the strict class on every
  section; zero UNMATCHED and zero SUSPECT; the by-hand trace of 2026-10-11
  (`paper1/k11_part2_verification.md`) re-applied to the moved text. ~1 h.
- **S4.22 `[x]` Self-containment read-through.** Every symbol (𝓛, c, p, R_enc, Ω, ρ, n₊, D_KY) defined
  before use; no `\ref` to a Paper I label; the model's transfer is quoted, not assumed. ~1 h.
- **S4.23 `[x]` Journal formatting (after S0.2).** Class, citation style, figure widths, supplement
  vs appendix, data-availability statement. ~2 h.


**S4 first pass done 2026-10-11** (S4.3–S4.9, S4.11–S4.13, S4.16, S4.17; S4.10 and S4.19 partly; details
in `docs/paperII/decisions.md`). `make paperII` builds (48 pages, no undefined references; number check
0 UNMATCHED, 0 SUSPECT). Left: S4.1, S4.2, S4.14, S4.15 (front matter, introduction, discussion,
conclusions: after J17G), S4.10's P7/P9 and the J17G rows of the model-error table and figure, SK6,
S4.18 (sections now ~9,800 words with LaTeX, before introduction and discussion), S4.19's duplicate
panels (Fig. da_2d(a) and Fig. tilings(a,b) repeat Fig. 1(b)–(d)), S4.20–S4.23.

**S4 completed 2026-10-12.** Front matter (key points, abstract 249 words, PLS with the learned-vs-physics
sentence from J17G), introduction (~880 words), discussion (~770) and conclusions (~310) written; main text
~9,270 words (target 9,500 per the user's 2026-10-11 decision, which supersedes the 8,500 above; the
Gaussianity paragraph moved to App. D). Duplicate panels dropped: Paper II variants of `fig09_p2_da.py`
(no Jacobian panel; legend in the bottom row) and `fig13_tilings.py` (no schematic/weights; 2x2), selected by
`PAPER_OUT` containing `paperII`, with all panel references relabelled. `refs.bib` pruned to the 47 cited
entries. Number check 0 UNMATCHED / 0 SUSPECT; every K11 correction confirmed in the moved text; the new text
traced by hand. Self-containment: SEC and Hann defined at first use; "vanishes on its patch edges" corrected
to "is smallest on" (the weight is sin^2((t+1/2)pi/p), never zero). JAMES: journal name, Open Research
statement, Acknowledgments. Left for the user: repository/DOI (S6.4), funding and CRediT (S6.3), Paper I's
arXiv id for `refs.bib` (S0.4).

---

## SK Kolmogorov-flow data assimilation for Paper II (decided 2026-10-11)

Purpose: show that the filter is not specific to KS, on a second system whose chaos is organized
by a forcing (anisotropic, a fixed channel height). Latent filter only; the physics-based comparison
stays a KS result. α = 0.05, the drag whose canonical models are best documented in Paper I.

- **SK1 `[x]` Model.** `k_xs005_c12d10`: the canonical α = 0.05 x-strip recipe (`k_xs005_noreg`
  flags) with c = 12 and training noise 0.1, i.e. the Paper II KS recipe. Queued
  (`scripts/run_kda_train.sh`, caffeinated): starts after the third J17G training (~13:30), ~3 h,
  then D_KY at 1×/2×/3× (seed 0) against the α = 0.05 truths. Check: D_KY within the spread of the
  6-channel canonical models (Paper I Table); if far worse, fall back to the clean 6-channel model.
- **SK2 `[x]` Kolmogorov truth in `exp_h9_tiled_da.py`.** `--system kolmogorov --alpha 0.05 --scale s`:
  strip geometry (48s × 8 sites, widened in x only), vorticity truth from `spinup_kolmogorov` /
  `integrate_kolmogorov` with a cache per (scale, seed), snapshot spacing matching the model's
  training `dt_snap`, standardization by the checkpoint's statistics. Unit test: the truth step
  equals the solver's; the deployment grid matches the training spacing in x and y. ~0.5 day.
- **SK3 `[x]` Tilings and observations on a strip.** Offsets {0, p/2} in x and y on a non-square
  periodic lattice; observation stride 8 (four per site); interpolation baseline on the strip;
  decoder floors; the blend's partition of unity on a rectangle (test: weights sum to 1). ~0.5 day.
- **SK4 `[x]` Smoke and calibration.** 10-cycle smoke at 1×; check the cycle length (number of
  propagator steps per cycle) against the Kolmogorov Lyapunov time (6–16 time units) so a cycle is
  a comparable fraction of a Lyapunov time to the KS runs; keep the KS filter settings (GC 1.5 sites,
  RTPS 0.7, 64 members, noise 0.02) unless the smoke diverges. ~2 h.
- **SK5 `[x]` Runs.** 1×: one and four tilings, seeds 0–2; 3×: one and four tilings, seed 0; 100
  cycles. Queue with caffeinate. ~3–4 h unattended.
**SK2–SK5 status, 2026-10-11.** Code: `scripts/exp_k_tiled_da.py` (Kolmogorov truth from the training
metadata at the training spacing, rectangular strip, same per-site filter and Hann blend as 2D KS) and
`ks_latent/da/tilings.py` (offsets, blend weights, per-frame observation sites; `tests/unit/test_tilings.py`,
13 tests). Cycle = 5 model steps = 1.0 time unit = 0.08 Lyapunov times (λ₁ = 0.084 at α = 0.05, 1×), as in 2D
KS. **Smoke (6-channel c19_e0d0_c6, 1×, 30 cycles, vorticity units):** interpolation 0.197; one tiling 0.222;
4 tilings shifted in x and y 0.415, because the y-shifted tilings fail (0.656, 0.665; their decoder floors are
only 0.20–0.21 vs 0.165): a y-shift by half a patch is a quarter forcing period, outside training, and the
propagator's dynamics break; 2 tilings in x 0.190; **4 tilings in x only 0.174** (floor 0.125), spread/error
0.35 (under-dispersed). Decision: Kolmogorov tilings are shifted in x only (default). The decoder floor
(12% of the field std, 4% in 2D KS) is the main limit for the 6-channel model. SK5 queued
(`scripts/run_kda_da.sh`, caffeinated): after the KDA training and D_KY check, 1× four tilings seeds 0–2 and one
tiling, 3× four and one tiling, plus RTPS 0.9 and model noise 0.04 at 1× (log `artifacts/logs/kda_da.log`).
- **SK6 `[x]` Results in Paper II.** A subsection "A second system: Kolmogorov flow" (≤ 400 words):
  error vs interpolation and the decoder floors, one tiling vs four, calibration, 1× vs 3×; one row
  block in the sizes table (or its own small table, `tabK_kolm_da.py`); one panel (vorticity truth,
  analysis, error at 3×) added to the snapshot figure or as a separate figure. Remove "Kolmogorov
  flow was not assimilated" from the limitations. ~0.5 day.

## S1D Per-patch 1D model for Paper II (decided 2026-10-11)

Purpose: make the 1D experiments of Paper II use the same kind of model as the 2D ones (per-patch
encoder and decoder, so Prop. 1 holds in 1D; noisy training at 0.1), instead of Paper I's 1D model
(reach 6, clean data), which §4.3 currently has to describe as an exception.

- **S1D.1 `[x]` Option A (queued, `scripts/run_1d_perpatch.sh`, caffeinated, log `artifacts/logs/pp1d.log`).**
  Paper I's 1D recipe with encoder/decoder reach 0, c = 6, no spatial penalty, noisy data
  (`artifacts/datasets/stage1_trajectories_dtsnap1_noise0.1.h5`, `scripts/make_noisy_dataset.py`).
  Starts when the last J17G training is done; then D_KY at L = 100/200/400 from three ICs
  (`eval_1d_transfer.py --tag pp1d_c6d10`) and the full 1D DA set (`run_blockD_1d.sh`, TAGP=pp1d_c6d10_,
  results in `artifacts/experiments/d1_pp1d_c6d10_`). Check: D_KY within the truth spread at every L
  (Paper I's 1D model is within 1.2%); reconstruction; free rollouts.
- **S1D.2 `[x]` Option B code (written 2026-10-11, not run).** Window training on one wide 1D domain:
  `scripts/gen_ks1d_wide.py` (L = 1600, 4+1 trajectories of 250, standardized with the L = 100 statistics),
  `ks_latent/training/crop1d.py` (Stage A on 1D windows; Stage B reuses the 2D forced-halo trainer with one
  site in y), `scripts/train_ks1d_crop.py` (saves checkpoints in the L = 100 format; `eval_1d_transfer.py
  --tag pp1d_crop_c6d10`). Tests `tests/unit/test_crop1d.py` (1D forced-halo exactness, halo one site
  short breaks it; bank layout; seam wrap; size independence). Run only if A transfers poorly or the
  user wants 2D-identical training.
- **S1D result (2026-10-12): keep the current 1D model.** Per-patch c6 model (pp1d_c6d10) transfers well (D_KY/L 0.22 at L=100/200/400) but its DA is ~2x worse: PFF+GC N=64 0.958 (L=200) / 0.910 (L=400) vs 0.427 / 0.425 for the current model; LETKF+GC N=64 0.837 / 0.882 vs 0.453 / 0.452. S1D.3 reduces to one sentence in §4.3 (a per-patch 1D model was tried and assimilates worse, so Prop. 1 is not used in 1D). Option B not run.
- **S1D.3 `[x]` Update Paper II with the per-patch 1D results** (after S1D.1, or S1D.2 if run): §4.3 (the
  1D model is per-patch and noisy-trained; drop the "Prop. 1 does not apply" sentence; its D_KY
  validation, a row block in Table `model_validation`); §9 and App. D (regenerate `fig08_1d_da.py`,
  `tab07_1d_da.py` from the new runs; re-check "ensemble doubles with the domain" and "GC beats SEC");
  App. A (`tabA_architecture_da`: the new 1D column); App. B/§3 if the 1D filter changes. Optional: run
  the 2D per-site LETKF with two shifted tilings in 1D, so 1D and 2D use the same filter (needs a 1D
  path in `exp_h9_tiled_da.py`). Decision log entry in `docs/paperII/decisions.md`.

---

## S5 Consistency between the two papers

- **S5.1 `[x]` Shared facts agree.** One script, `scripts/paper_figures/shared_facts.py`, prints the
  numbers both papers quote (2D KS truth D_KY and n₊ at 44²/66²; the 12-channel model's D_KY error;
  the 1D model's transfer error if used; σ_obs in field units) from the result files; both papers'
  tables are generated from it or checked against it. ~1 h.
- **S5.2 `[x]` Cross-citations.** Paper I cites Paper II once (§6.3 and the conclusions); Paper II
  cites Paper I in §1, §3.1 and §4. Each citation names what is taken from the other paper. ~20 min.
- **S5.3 `[x]` No duplicated figures.** The only shared graphic is the 2D KS snapshot panel; old Fig. 9a
  (Jacobian sparsity) appears only in Paper II. ~10 min.
- **S5.4 `[x]` `full_paper_tasks.md` closed out.** Add a final entry pointing to this file; Blocks K11–K12
  continue here as S3.12/S4.21 and S6.3. ~15 min.


**Paper I finishing pass, 2026-10-11** (user: "please do that"; details in `docs/paperI/decisions.md`):
S3.9 9,609 → 9,167 words (stopped ~170 above target; Chaos has no hard limit). S3.14: `main.tex` is
now REVTeX 4.2 `aip,cha,reprint,floatfix` with a lead paragraph, Data availability and Author
declarations; the AGU draft is kept as `main_agu.tex`. Figures stay at their generated widths
(3.37 / 5.5 in), placed in `figure`/`figure*`, rather than being regenerated at 7.0 in. The CRediT line is a
placeholder for the coauthor. S5.1: `scripts/paper_figures/shared_facts.py` (both papers agree; it
caught the 66² truth mean, 189.05, printed as 189.1 in both: now 189.0). S5.2: Paper I cites the
companion in 9 places, each naming what it takes; the 1D pointer in §5 no longer assumes which 1D
model Paper II uses. S5.3: the only shared graphic is the 2D KS snapshot. S6.1 done for Paper I
(grayscale: hatching in `box_vs_window` and `appendix_panels`(b), markers and line styles in
`truth_convergence`, dashed k_max = 32 in `failure`(c)); Paper II's figures still to check. S6.2:
the only references added in the split are the two companion papers. Left: S6.3 hand-off
(user), S6.4 release (tag + DOI; user).

---

## R Internal referee reviews and the 2D transfer question (2026-10-12)

Two independent referee-style reviews (fresh agents, full read) recommended **major revision** for both papers;
reports: `docs/paperI/internal_referee_report.md`, `docs/paperII/internal_referee_report.md`. Verify each point
against the data before changing text. Verified so far: R1's premise (2D training windows, 40×40 sites = 220²,
are larger than the 44²/66² evaluation domains; Kolmogorov windows 48 sites ≈ 6 boxes vs ≤ 3 evaluated), R6's
ensemble counts, R7's stalled estimation. Not confirmed: the "35% low at 88²" point refers to the older `p7_ks2d`
(215.9); the canonical `p7b_ks2d` gives 332–335 against the extensive prediction 0.043·88² ≈ 333.

User decision (2026-10-12): the 2D method is "train on windows of one larger simulation, then deploy at other
sizes", acceptable because it is how a CM1 model would be trained; but first try training on subsets of the 44²
data (R4).

- **R1 `[ ]` Paper I 2D framing.** Replace "zero-shot transfer to larger domains" in 2D by "trained on windows of one
  large simulation, deployed on domains of other sizes without retraining"; keep the true small-to-large claim for
  1D (L = 100 box to L = 400). State the source-domain size wherever 2D results appear: abstract, key points,
  Fig. 1 caption, introduction, contributions, conclusions, Table A. Depends on R4's outcome.
- **R2 `[ ]` Paper I disclosures (verify each first).** Precision vs truth uncertainty ("0.6–2%" against ±20% single
  truths, mean of absolute errors); λ₁ of the 2D models 12–29% above the truth (1D +4–9%) not reported; the older
  88² value (`p7`, 215.9) vs canonical; the order in which the truth-free selection rule was written and applied;
  Lemma 1 framed as the law of total variance with its limits; "no drift" vs Table XVI (std 1.19× at 88²); a
  global-latent or CNN baseline (or say why not); the 1D interaction-range measurement confounded by encoder reach;
  ~30 minor items in the report.
- **R3 `[ ]` Beyond-source check.** Free run of the canonical 2D KS model at 440² (beyond the 264² source) against a
  true 440² simulation: field std, energy spectrum, correlation length. A few hours; supports Paper I and Paper II.
- **R4 `[ ]` 2D small-source experiment (user, 2026-10-12).** c19-style per-patch model (encoder and decoder reach
  0, c = 6) trained on windows of the 44² data (`ks2d_trajectories_default.h5`, 24 + 6 trajectories × 25 time
  units). The propagator's reach must be short enough that a window of the 8-site box does not wrap onto itself:
  reach 2 (window 1 × 2 layers, halo 2, 8-site crop, 4×4 interior) and reach 3 (window 3 × 1 layer, halo 3, 2×2
  interior). Control: the same short-reach propagator trained on 264² windows, to separate the data from the reach.
  Evaluate D_KY at 44², 66² (truths) and 88² (extensive prediction ≈ 333); then the latent filter at 44²–440².
  The earlier box-trained model (full reach 6 on the 8-site ring, periodic wrap) failed by 14–23%.
- **R4 first result (2026-10-12).** `r4c_ks2d_s44_r3` (44² data, c = 6, reach 3, batches 32/64): D_KY 83.1 at 44²
  (+1.3% vs the 82.0 mean), 187.1 at 66² (−1.0% vs 189.0), 329.0 at 88² (−1.2% vs the extensive 333; canonical p7b
  332–335). Train small, run large works in 2D KS once the propagator's window cannot wrap onto itself. The 6-channel
  decoder floor at 44² is 0.66–0.69 for every 6-channel model (canonical p7b too), 0.25 for 12 channels: a channel
  effect. Filter at 44²: 0.618 (p7b at the same settings 0.579; interpolation 0.829). Queue
  (`scripts/run_r4_queue2.sh`): equal-work C, D (c = 12, noise 0.1, 44² data; filter at 44² and 440²), p7b at 440²,
  A, B.
- **R4 log of decisions (2026-10-12, user-approved).** (1) First C used default batches; an equal-work rerun (batches
  scaled to the canonical sites per step) gave D_KY within the IC spread of the first (82.8/183.1/324.6 vs
  83.1/187.1/329.0), so all later models use default batches. (2) A and B (66² source, reach 4/5) dropped as
  unnecessary once C worked; replaced by the reach-6 wrap control (`r4w_ks2d_s44_r6_wrap`, `--periodic-full`) and C
  training seeds 1–2. Kolmogorov models moved ahead of those (they decide the Paper I rewrite). (3) D (c = 12, noise
  0.1, 44² box, reach 3) filters at 0.408 at 44² (RTPS 0.7; 0.395 at RTPS 0.9) vs 0.324 for the Paper II model, with
  an equally good decoder (floor 0.233/0.132 vs 0.251/0.137): the gap is the propagator. Two candidate causes, tested
  in order: **data** (D saw 24 box trajectories, the Paper II model ~6× more area) → `r4e_ks2d_s44big_r3_c12d10`
  on 144 new 44² trajectories (`scripts/gen_ks2d_box_large.py`, `scripts/run_r4_dbig.sh`); **reach** (3 vs 6) →
  if data does not close it, c = 12 models with reach 5 on 66² (12 sites) and reach 6 on 77² (14 sites, the smallest
  box without a wrap at reach 6), data generated in advance (`scripts/run_r4_gen_reach.sh`: 64 × 66², 48 × 77²,
  about the same area as the 44² set). If reach 5–6 reaches ~0.32, Paper II pivots to that small box (440² is then
  25–40× the training area).
- **R4 fallback (user, 2026-10-12).** Paper I adopts C (and the small-box Kolmogorov model if it transfers) regardless
  of D. If neither more 44² data nor reach 5–6 (66²/77²) brings D near the Paper II model, Paper II keeps its current
  model (windows of 264²) and states that route; the small-box filter result is then at most a remark.
- **R4K `[~]` Kolmogorov small-source test (user, 2026-10-12; `scripts/run_r4k_kolmogorov.sh`, after the R4 training
  lane).** 1× box data (α = 0.05), crops 8 sites in x with full height, reach 3, equal-work batches; c = 6 (Paper I)
  and c = 12 with data noise 0.1 (Paper II). D_KY at 1×/2×/3× against the per-IC truths; the tiled filter at 1× and
  3×. Then α = 0.1 (box data exists) and α = 0.025 (box data to generate).
- **R9 `[ ]` Paper I rewrite (user, 2026-10-12: "a complete rewrite for paper 1").** If R4/R4K hold: 2D and Kolmogorov
  trained on the small periodic box with a propagator reach below half the box, transferring like 1D; the old
  box-trained failure explained by the wrap-around (reach 6 on an 8-site ring), the window-trained models of a large
  domain kept as the alternative route. Needs: second training seeds of the adopted small-source models, all three
  drags, second ICs at every size, a truth at 88² (or the extensive prediction stated as such), the λ₁ comparison, the
  interaction-range argument for the reach, new figures/tables, and every R2 disclosure. Paper II then uses the
  small-source 12-channel model where it matches the current one (or adds it as "trained on 44², assimilated on
  440²").
- **R9a `[ ]` Paper I decision rule (written 2026-10-12 12:30, before the 66²/88² and Kolmogorov 2×/3× evals came in;
  the 44² results were already known).** Adopt the small-box paradigm for a system if, for that system's adopted model:
  (i) **2D KS:** D_KY within 5% of the truth mean (82.0 / 189.0; 88² against the extensive prediction ≈ 333, stated as
  such until R10 item 13) at 44², 66² and 88², on every training seed (3 seeds); (ii) **Kolmogorov (α = 0.05):** D_KY
  at 1×, 2× and 3× within the per-IC truth spread (about ±20%; R2) of the multi-IC truth mean, and the mean over ICs
  within 10%. If both systems pass: full rewrite (R9/R10/R11). If only 2D KS passes: rewrite the 2D KS parts, keep the
  window-trained Kolmogorov model and say so. If Kolmogorov passes only for the c = 12 noisy model rather than c = 6:
  adopt whichever passes, log it in `docs/paperI/decisions.md`, and use that model for both papers' Kolmogorov results.
  If neither system passes: Paper I keeps the window-trained models and gains only the R2 disclosures, the seeds and the
  wrap control as a limitations result. The wrap control (`r4w`, reach 6 on the full periodic box, 44²: 83.3, +1.9%)
  does not by itself establish wrap-around as the cause of the old failure; it is read alongside its 66² and 88² evals
  before the paper says so.
- **R4 scheduling change (2026-10-12 15:10, user: "prioritize any jobs that will help us make that decision").**
  (a) `run_r4_dbig.sh` gated the 144-trajectory D retrain (r4e) on "TRAIN LANE DONE", i.e. on every eval including
  Kolmogorov 3×, which would have delayed the Paper II data test by about a day; replaced by `run_r4_dbig2.sh`
  (starts at once; its 440² run waits only for D's own 440² result). (b) c = 6 Kolmogorov 3× (seed 0) started in
  parallel with 2×, rather than after it; the queue skips it when its json exists. (c) c = 12 Kolmogorov 1× eval
  (6.3 h, 150% CPU) paused with SIGSTOP (pid 14910; resume with `kill -CONT 14910`): c = 6 is the model R9a tests
  first; c = 12 is the contingency. No result discarded.
- **R10 `[ ]` Paper I rerun and reanalysis inventory (if R4/R4K hold; user 2026-10-12: "complete rewrite").**
  Everything in Paper I that rests on the 2D KS or Kolmogorov models, mapped from the figure/table generators and
  the sections. The 1D results (§5, Figs. 1d_transfer/1d_spectra, Table transfer1d) are unaffected: the 1D model is
  already trained on a small periodic box (L = 100, 16 sites) with reach 6 < half the ring, i.e. it obeys the new rule.
  *Training runs (MPS):*
  1. Adopted small-source models, per system: 2D KS (44² source) and Kolmogorov at α = 0.1, 0.05, 0.025 (1× box
     source; α = 0.025 box data must first be generated, `kolmogorov_trajectories_*` recipe: 30 trajectories × 40 time
     units). At least **three training seeds** each (Paper I had one for most 2D models; the referee's point).
  2. **Causal control for the old box failure:** the same small-source recipe with the propagator reach 6 on the
     8-site box (`--periodic-full`, wraps onto itself) vs reach 3 (does not). Without it "the wrap-around caused the
     failure" is a hypothesis; the old box-trained models also differ in encoder reach and recipe.
  3. **Reach sweep on the small box:** reach 1, 2, 3 (and 4 with a one-site wrap) at 44² and at the Kolmogorov 1×
     box — the measured interaction range that justifies the chosen reach (replaces the 1D-only interaction-range
     argument the referee called confounded; Paper I §3.3 says reach-1 propagators fail in Kolmogorov flow, which
     must be reconciled with the new results).
  4. Optional: k_max ablation for the small-source recipe (does it still need long rollouts? Paper I §7.3–7.4).
  5. Decide whether the α = 0.1 training-variant study (Table alpha01_ablation, Fig. failure(a), the truth-free
     selection rule) is redone with small-source models or kept as a window-trained study; if the canonical recipe
     changes, apply the selection rule to the new variants and **state its pre-registration date honestly**.
  *Evaluations (CPU):*
  6. D_KY, n₊, λ₁ for every adopted model and seed at every size: 2D KS 44², 66², 88²; Kolmogorov 1×, 2×, 3×;
     **three IC seeds per size** (the consolidation runs used two), all directions where d_lat allows (88²: 512 of
     2304, as before, stated in App. C).
  7. λ₁ against the truth for every model (2D models' λ₁ was 12–29% high and unreported) — a column in Table
     transfer2d and in App. D.
  8. Free-rollout statistics of the new models (`exp_rollout_stats_ks2d.py`, `exp_rollout_stats_kolmogorov.py`):
     field std, band energies, drift over the rollout (replaces the "no drift" claim that Table XVI contradicts).
  9. Reconstruction and spectra (`eval_spectra_recon.py` → Table eval_spectra) for the new models.
  10. Unstable-subspace scale content (`exp_c8_unstable_subspace.py`, App. D.2) for the new Kolmogorov model at 3×.
  11. Matched-IC comparison at native size (App. C.6) for the new Kolmogorov models.
  12. The canonical window-trained models stay as a reported alternative: keep their numbers, add seeds only if
      they remain in the main text.
  *Truth runs (CPU, long):*
  13. **2D KS at 88²** (none exists; the 88² checks use the extensive prediction ≈ 333): one or two 40-unit runs,
      ~400 directions — about a day each.
  14. Kolmogorov truths that rest on one 40-unit run (α = 0.1 at 1×; α = 0.05 at 3×; α = 0.025 at 1×, 2×, 3×):
      extend to 120 units or add ICs, as R2 requires (40-unit runs deviated by −17% to +23%).
  15. Truth-side λ₁ for the λ₁ comparison (already in the truth JSONs; tabulate).
  *Figures (regenerate into `docs/paperI` via `make paperI-assets`; `docs/paper1` stays frozen):*
  16. concept_transfer (Fig. 1): panel (c) shows window training on a large simulation → small periodic box with a
      forced halo and a reach shorter than half the box.
  17. systems (Fig. 2): panels (b), (c) show the 264² domain and the 16 × 2π strip → the 44² and 1× training boxes.
  18. box_vs_window (Fig. 5): becomes wrap vs no-wrap (old box, reach-6 control, reach-3 small source, window-trained).
  19. scaling (Fig. 6): new models, three training seeds, three ICs.
  20. rollouts_spectra (Fig. 7): the new models at 88² and 3×.
  21. failure (Fig. 10): per item 5.
  22. appendix_panels (Fig. C2): per-IC scatter and reach/channel ablation with the new models (reach sweep).
  23. truth_convergence (Fig. C1): add the new truth runs (88², extended Kolmogorov).
  *Tables (regenerate):* 24. transfer2d (Table IV, + λ₁ and seeds); 25. truth_dky / truth_runs / running_* (new
  truths); 26. eval_ks2d, eval_kolm01/005/0025 (App. D, all new models and controls); 27. eval_spectra;
  28. architecture_dynamics (Table A: training domain, window, reach, batches, data); 29. alpha01_ablation per item 5.
  *Text (rewrite):* 30. abstract, key points, lead paragraph, title (check "zero-shot domain transfer" still fits);
  31. §1 introduction and contributions; 32. §2.3 what is and is not new; 33. §3.3 receptive fields (reach rule:
  reach < half the training domain; interaction range from item 3); 34. §3.4 training on windows → training on a
  small periodic box with a forced halo (Prop. 2 unchanged; the window of a periodic box), and windows of a large
  domain as the alternative; 35. §3.6 canonical recipe; 36. §4 data (training sets: 44², 1× boxes; compute in
  minutes); 37. §6 rewritten (box with wrap fails, without wrap transfers; windows of a large domain also work);
  38. §7 failure modes reframed (the coupling section's "more non-local information makes it worse" now has a cause);
  39. §8 discussion: "what transfer requires" (condition 2 becomes the reach/wrap rule), limitations, outlook;
  40. §9 conclusions; 41. App. A, C (model-spectra precision, matched ICs), D, E (compute, seeds);
  42. every R2 disclosure (precision, λ₁, selection-rule chronology, Lemma 1 framing, drift, baselines statement).
  *Cross-paper and checks:* 43. Paper II text that quotes Paper I (§2.2, §3.1 "window training", §4.2, §5.1;
  "D_KY within 0.6–2%", "windows of 264²") and the Paper II model decision (R9); 44. `shared_facts.py` updated for
  the new facts; 45. `check_numbers.py --paper paperI` 0 UNMATCHED/0 SUSPECT; 46. grayscale/zoom check of the new
  figures (S6.1); 47. citation check for any new reference; 48. `docs/paperI/decisions.md` entries for every change;
  49. a fresh internal referee pass on the rewritten paper before the coauthor hand-off.
- **R11 `[ ]` Two training routes, one rule (user, 2026-10-12; add if R4/R4K work out).** Keep both routes in
  Paper I and present them as one principle: *the propagator must never read a site that is not forced from data
  within its reach.* (a) **Small periodic box → large domain**: satisfied when reach < half the box; cheap data,
  for systems that are extensive at the box size and homogeneous (KS, Kolmogorov, periodic shallow water).
  (b) **Windows of a large simulation → full domain**: satisfied by the forced halo (Prop. 2); the general route for
  inhomogeneous or organized systems that a small box cannot host and for data that are windows of larger dynamics
  (CM1 squall lines: cold pool, environmental gradients, mesoscale organization; radar/reanalysis regions) — the
  limited-area-model setting. The old box failure is then the rule violated (reach 6 wraps an 8-site ring), shown by
  the R4 wrap control. Add: a side-by-side table of the two routes per system (D_KY errors, seeds, training cost);
  a test of where only windows work (candidate: Kolmogorov α = 0.025, whose truth density per box rises with width,
  i.e. not extensive at the box size); a discussion paragraph on choosing the route (measure the interaction range
  first: for shallow water, gravity waves make the per-step range wave speed × Δt, which may force a large box and so
  favour windows; for CM1, windows). Paper II: one filter model (D if it matches the current 12-channel model), with
  one 440² run of the other route's model to show the filter does not depend on how the model was trained; the
  outlook ties the window route to CM1 (Paper 3).
- **R5 `[ ]` Optional: source-size sweep** (windows of 132² and 88² sources) — how small the training simulation can
  be; useful for planning CM1 runs.
- **R6 `[ ]` Paper II ensemble budget.** P7 says "same observations and ensemble resources": false (latent 4 × 64 vs
  physics 64). Fix the text; report matched-budget latent results (4 × 16: 0.318–0.352 at 44²); run 4 × 16 latent
  filters for J17G δ = 0.1/0.2/0.3.
- **R7 `[ ]` Paper II parameter estimation stalls.** Parameter spread hits its 0.005 floor by cycle 30–40; estimates
  stick (∇² 1.25 → 1.17; advection 2 → 0.40; advection 1 → 0.07). Rerun the estimation cases with parameter inflation
  (higher floor or parameter RTPS) and RTPS retuned with each remedy on; regenerate Table/Fig. model_error; recheck
  the crossover.
- **R8 `[ ]` Paper II other points.** State the training source (windows of 264²) and that 440² goes beyond it;
  crossover x-axis (after bias correction, before estimation) and where the hidden-field cases sit; "a sixth of the
  observation error" stated as a one-system result; the learned model's own one-cycle error (0.19) vs the crossover
  (0.10); Prop. 1 framed by what exactness enables (tilings, size invariance), not as deep; the per-patch decoder's
  cost and the overlapping-decoder alternative; noisy-training gain vs the run-to-run threshold; prior work to verify
  (hybrid physics + learned correction, e.g. Pathak et al. 2018; Penny et al. 2022, JAMES); the partially run cubic
  missing-process experiment; observation network aligned with the patch lattice; sparse-observation numbers
  (0.769 / 0.798 / 0.77) inconsistent; notation reuse (c, p, K, h, ρ, U_s/N_s); figure/table order; Fig. da_2d(a)
  clipping; Kolmogorov model settings in App. A.

---

## S6 Finishing

- **S6.1 `[x]` Grayscale and 50 %-zoom check** of every figure in both papers (contract §F). ~1 h.
- **S6.2 `[x]` Citation verification** for any reference added during the split (the B13 search-log
  convention in `paper1/decisions.md`). ~1 h.
- **S6.3 `[~]` Decisions log and coauthor hand-off.** Both `decisions.md` files complete; send the
  coauthor Paper I first (S0.4), then Paper II once P7/P9 are in. ~30 min.
- **S6.1/S6.2 done 2026-10-12.** Paper II figures checked in grayscale at 72 dpi: two fixes (the uniform-average
  bars in `tilings`(b) hatched; the four-tiling decoder floor in `da_2d`(a) dash-dot); the diverging colour maps of
  `da_snapshot` and `systems_ks2d` keep their sign only in colour, as in Paper I. Citations: the only entries added
  or changed since the split are the two companion papers (`jones2026transfer`, `jones2026da`), whose titles match
  the papers; every other entry is byte-identical to the B13-verified `paper1/refs.bib`.
- **S6.4 `[ ]` Release.** Tag the commit; fill the TODO(G8) archive line in both reproducibility
  appendices. ~30 min.

---

## Order and estimate

| Phase | Tasks | Effort | Blocked by |
|---|---|---|---|
| 0 | S0.1–S0.6 | 0.5 h (user) | — |
| 1 | S1.1–S1.7 | 2.5 h | S0.1 |
| 2 | S3.1–S3.13 (Paper I) | ~13 h | phase 1 |
| 3 | S4.1–S4.9, S4.11–S4.19, S4.22 (Paper II, all but §6) | ~20 h | phase 1, S0.3 |
| 4 | S4.10, S4.20–S4.21 | ~4 h | J17G, J5 |
| K | SK1–SK6 (Kolmogorov DA) | ~2 days + ~7 h runs | SK1 training (~16:30) |
| 5 | S5, S3.14, S4.23, S6 | ~8 h | S0.2, phases 2–4 |

Phases 2 and 3 are independent and can be interleaved; Paper I can be finished and posted (S0.4)
while J17G runs.
