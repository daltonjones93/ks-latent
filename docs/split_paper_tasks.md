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
- **R4 result log, 2026-10-12 19:15.** (1) r4e (D on 144 trajectories): 44² filter 0.393 (RTPS 0.7 and 0.9) vs D 0.408/0.395:
  data does not close the gap to the Paper II model (0.324). (2) r4f (66² box, reach 5, c = 12): 0.418 (RTPS 0.7),
  0.405 (0.9): a longer reach does not help either. (3) **D at 440²: 1.104, calibration 0.26, worse than interpolation
  (0.819)**; C_eq at 440²: 0.813. Small-box models' filters do not hold up at 440² even though their D_KY transfers.
  (4) **Kolmogorov c = 6 small-box model at 2×: D_KY 25.4 (12 positive) vs truth 46.6 (−45%), λ₁ 0.070** (1× was 22.3,
  −0.9%). Against R9a the Kolmogorov criterion fails at 2×. Hypothesis (not tested): reach 3 on an 8-site box
  spans 7 of 8 sites, so the Kolmogorov propagator can use box-scale information that does not exist in a larger
  domain, whereas 2D KS interactions are local at that range. The c = 12 model (contingency in R9a) is being
  evaluated at 1× (resumed) and then 2×; the c = 6 3× eval is left running to confirm. Duplicate 3× eval started by
  the queue (pid 21667) killed.
- **R12 `[ ]` Making the small-box paradigm work for Kolmogorov flow (user, 2026-10-12).** The α = 0.05 small-box
  model (`r4k_c6_a005_s1x_r3`: 1× box 2π × 2π, 128², 8 × 8 sites, reach 3, c = 6, forced-halo windows in x, full
  periodic y) gives D_KY 22.3 at 1× (−0.9%) but 25.4 at 2× (truth 46.6, −45%; 12 positive exponents vs ~22; λ₁ 0.070
  vs 0.124 at 1×). Training does not use the wrap-around (windows with a forced halo), so the R4 wrap hypothesis does not
  apply. **Hypothesis:** at weak drag energy reaches the largest scale the domain allows; a 2× domain has active scales
  (wavelength 4π) that a 2π box never contains, so a model learned there cannot produce them. Supporting: the truth's
  D_KY per box width rises with width (22.5 / 23.3 / ~25 at 1× / 2× / 3×), i.e. α = 0.05 is not fully extensive at
  the box size. Tasks, in order:
  1. (plan) **Diagnosis** (cheap): free-rollout statistics of the model and the true PDE at 1× and 2×
     (`exp_rollout_stats_kolmogorov.py`: energy share beyond the native box, |k| < 2 share, field std; new
     `box_period_corr_late`, the correlation of the field with itself shifted by 2π, ≈ 1 if the 2× rollout has locked
     into copies of the box). Confirms or rules out the large-scale hypothesis.
  1. `[x]` **Diagnosis result (2026-10-12 19:46; 1000 model steps, same IC as the true PDE; late half):**
     1×: model ≈ truth (vorticity std 1.456 vs 1.404; |k| < 2 energy share 0.116 vs 0.096; forcing-scale share
     0.537 vs 0.546). 2×: the model is **not** locked into copies of the box (shift-by-2π correlation 0.07; truth
     0.41) and does **not** lack the new large scales much (energy at wavelength 4π: 0.046 vs truth 0.065). Instead it
     drifts into a state with **3× too much large-scale energy** (|k| < 2 share 0.420 vs 0.146), less forcing-scale
     activity (0.449 vs 0.536) and lower std (1.226 vs 1.406): a spurious large-scale (condensate-like) state that
     damps the small-scale chaos, consistent with the low λ₁ (0.070) and n₊ = 12. **The "missing scales" hypothesis
     is not supported; the failure is a large-scale drift of the model in the larger domain.** Implication for item 2:
     stronger drag damps large scales, so α = 0.1 still tests the right thing (whether the drift is drag-limited);
     item 3 (larger box) would expose the model to these large-scale modes in training.
  2. `[ ]` **Same recipe at α = 0.1** (stronger drag, shorter largest active scale; box data and 1×/2×/3× truths
     exist): train `r12_c6_a01_s1x_r3`, rollout statistics at 1×/2×/3×, D_KY at 2× (IC seed 1, T120 truth 31.5),
     1×, 3×. Pass: R9a's Kolmogorov criterion. If α = 0.1 transfers and α = 0.05 does not, Paper I gains a physical
     rule (the box must contain the largest dynamically active scale) and α = 0.05 becomes a predicted failure.
  3. `[ ]` (decide later) **Larger source box at α = 0.05**: train on a 2× box, test at 3× (needs 2× box data; check
     whether `kolmogorov_wide3_alpha0.05.h5` is a periodic 3× box). Tests the rule in item 2 directly.
  4. `[~]` c = 12 noisy model (R9a contingency), already being evaluated; not expected to supply missing scales.
  5. Low priority: more 1× data (cannot add scales larger than the box; more data did not help 2D KS DA);
     a multiscale architecture (a coarse global field alongside the local one): out of scope for Papers I/II.
  Script: `scripts/run_r12.sh` (items 1–2). To free cores, the c = 6 α = 0.05 3× eval (pid 20213), which cannot
  change the R9a outcome after the 2× failure, is paused (`kill -CONT 20213` to resume).
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
- **R8 `[~]` Paper II other points** (2026-10-12: done without new runs, `docs/paperII/decisions.md` R8 entry: training
  source; crossover as a one-configuration result; one-cycle error 0.19; Prop. 1 framing and "exact"; per-patch decoder
  cost; noisy-training gain vs the rule; aligned network as a limitation; sparse numbers (long runs over cycles 51–300);
  cubic case reported; notation; float order; Fig. da_2d(a) log axis; Kolmogorov model in App. A; Pathak 2018 ×2 and
  Penny 2022 verified and cited. **Left:** the one-cycle error on Fig. model_error's axis and the label overlaps there
  (with R7's regeneration); P7's "ensemble resources" sentence (R6); the title (authors' call)).
  Original list: State the training source (windows of 264²) and that 440² goes beyond it;
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

---

## Block Z — Where we are and the minimum viable path to both papers (written 2026-10-12 20:45)

This block supersedes the planning parts of R4–R12 where they conflict. Numbers are D_KY (Lyapunov dimension)
unless marked as filter error (analysis RMSE in units of the field std, 3 decimals).

### Z1. What we learned on 2026-10-12

1. **2D KS, small box works for dynamics.** Model C (c = 6, reach 3, trained on 44² boxes) gives D_KY 81.4–83.8 at
   44² over 3 training seeds (truth 81.8), 183.1–187.1 at 66² (truth mean 189.0; −4 to −5% against the one 66² IC
   truth 192.5) and 323.8–329.0 at 88² (extensive prediction ≈ 333; no truth). Paper I's window-trained model:
   83.6 / 187.9 / 333.3. This is the real upscaling test Paper I's referee asked for (Paper I M1).
2. **Wrap control.** Reach 6 on the full periodic 44² box (wraps onto itself) still gives 83.3 at 44²; its 66²
   eval is pending. The old box failure is therefore *not yet* explained by the wrap-around; treat it as a
   recipe difference with no isolated cause unless the 66² eval fails.
3. **Small-box models do not filter well, and do not transfer as filters.** At 44²: C 0.599, D (c = 12, noisy)
   0.395–0.408, D on 4× more data 0.393, D with reach 5 on a 66² box 0.405–0.418; Paper II model 0.324. At 440²:
   C 0.813, D 1.104 (interpolation 0.819). Data and reach are ruled out as causes of the 44² gap. **Paper II keeps
   its model.** (p7b 440² control in the same setup is running; needed only for an optional paragraph.)
4. **Kolmogorov, small box.** α = 0.05: D_KY 22.3 at 1× (−0.9%) but 25.4 at 2× (truth 46.6, −45%). Diagnosis:
   at 2× the model drifts into a spurious large-scale state (|k| < 2 energy share 0.420 vs truth 0.146, lower
   forcing-scale activity and λ₁), not a lack of large scales. α = 0.1: the 2× rollout matches the truth
   (|k| < 2 share 0.093 vs 0.081; std 1.534 vs 1.518; forcing-scale share 0.628 vs 0.584); D_KY pending. If α = 0.1
   transfers, Paper I gets a small-box story in all three systems plus a *diagnosed* failure at weak drag.
   c = 12 noisy α = 0.05 model at 1×: 24.8 (+10.5%), worse than c = 6.
5. **Paper II ensemble budget (R6, done).** Latent filter at 4 tilings × 16 members (64 states, = physics filter's
   64), J17G, 3 seeds: δ = 0.1 0.332 (physics 0.328: tie); δ = 0.2 0.355 (0.433: latent 1.22×); δ = 0.3 0.395
   (0.485: 1.23×). At 4 × 64 the ratios were 1.27× / 1.29×. The latent advantage survives equal budget.
6. **Paper II parameter estimation (R7, partial).** A spread floor of 0.02 un-sticks the estimate (∇² case:
   1.25 → ≈ 1.0, previously stuck at 1.17) and, with state RTPS 0.6, the physics filter error falls from 0.423 to
   ≈ 0.16–0.17 at cycle 70. Expect: with working estimation, the physics filter *wins* the parameter-error cases; the
   latent filter's advantage is confined to missing-process (structural) error, where no parameter can fix the
   physics model. Paper II's model-error narrative must be rewritten around that split.
7. **Machine load.** Load average ≈ 177 on 18 cores: the D_KY evals each start one thread per core. Everything
   launched from now on sets `OMP_NUM_THREADS` (2–4).

### Z2. Running now, and why

| Job | Why | Expected |
|---|---|---|
| α = 0.1 small-box D_KY at 2× (IC 1, truth 31.5), 1×; then 3× (IC 1) | decides Paper I's Kolmogorov small-box story | 2×: overnight; 3×: tomorrow |
| α = 0.1 rollout statistics at 3× | early read on 3× before D_KY | tonight |
| C seeds 1–2 and wrap control at 66², then 88² | Paper I: ≥3 training seeds at every size; wrap control's meaning (Z1.2) | tonight–tomorrow |
| R7 parameter-inflation reruns (24 seed-0, then 12 more) | Paper II referee: estimation stall | tomorrow morning |
| p7b at 440² (R4 DA lane) | optional Paper II paragraph (Z4, optional) | tonight |
| r4f D_KY, then r4g (77², reach 6) | closes the reach question for Paper II | low priority; can be stopped |
| c = 12 α = 0.05 Kolmogorov evals; α = 0.05 c = 6 3× (paused, pid 20213); c = 6 IC-1 1× | R9a contingency / confirmation | low priority; stop if cores are needed |

**All remaining runs queued (`scripts/run_z.py`, started 2026-10-12 20:48, log `artifacts/logs/z.log`; pool of 3 at
3 threads, MPS trainings serial):**
- truths: 2D KS 88² IC 0 (K = 400; ~1 day; M1); 2D KS 66² IC 2 (K = 300; third 66² IC, M3); Kolmogorov α = 0.1 1×
  ICs 1, 2 (T120; the 1× truth was one 40-unit run).
- α = 0.1 small box (`r12_c6_a01_s1x_r3`): D_KY 2× ICs 0, 2; 3× ICs 0, 2; 1× ICs 1, 2 (after their truths); training
  seeds 1, 2 (+ 2× IC 1 and 1× IC 0 each). With run_r12.sh (2× IC 1, 1× IC 0, 3× IC 1): 3 ICs × 3 sizes for seed 0,
  3 training seeds at 2×.
- 2D KS IC 2 at 44², 66², 88² for the small-box C (seed 0) and the window-trained p7b (M3: 3 ICs per cell).
- rollout statistics of C at 44²/66²/88² with truth (M8 drift statement).
Not queued, on purpose: α = 0.1 seeds 1–2 at 3× (add if seed 0 passes and time allows); C seeds 1–2 at IC 2.
Note: the R4 queue's 66² evals compare both ICs to 192.5 (IC 0's truth); use the per-IC truths (192.50, 185.60, and
IC 2 when its truth lands) in the tables.

### Z3. Minimum viable path — Paper I (*Chaos*)

**Framing (R11, "two routes, one rule").** A weight-shared local latent model transfers to larger domains when the
propagator never reads a site that is not forced from data. Route (a), a small periodic box (1D L = 100; 2D KS 44²;
Kolmogorov 1× at α = 0.1 if it passes); route (b), windows of a larger simulation (the existing canonical 2D models).
State plainly that the window-trained models are evaluated on domains *smaller* than their windows (referee M1(a)),
and that route (a) is the genuine upscaling test (M1(b)). The window-trained models stay as they are; **no full
retraining (R10 is replaced by the list below).**

*Must do — compute* (in this order; each eval with `OMP_NUM_THREADS=4`):
1. Finish C seeds 1–2 at 66² and 88² (running). 3 training seeds × the existing ICs.
2. More ICs for C seed 0 (M3: ≥ 3 ICs per cell; 5 if time allows) at 44² and 66², where truths exist.
3. α = 0.1 small-box: D_KY at 1×, 2×, 3× (running); if it passes R9a, train 2 more seeds (≈ 15 min each) and
   evaluate at 2× (and 3× if affordable).
4. 2D KS truth at 88² (M1, ~1 day CPU). Start as soon as the load drops. If not done in time, report 88² against
   the extensive prediction, labelled as such, next to the trivial extensive baseline (M7).
5. Rollout statistics for the α = 0.05 failure (done) and for C at 88² (2D drift statement, M8; existing script).

*Must do — analysis and writing (no new runs):*
6. Trivial extensive baseline next to the transfer table (M7); λ₁ against truth for every 2D model (M4, from
   existing JSONs); interval statements instead of "0.6–2%" (M3).
7. Narrow Lemma 1 (motivating identity), Proposition 2 ("exact equivariance"), and the selection rule (one case
   study, honest chronology) (M5, M6, M12).
8. Kolmogorov extensivity statements made consistent (M11), using the R12 rollout spectra as the support the
   referee asked for: weak drag → large-scale drift in the model; the truth's per-box D_KY rises with width.
9. Box-vs-window section (M2) rewritten: the old box-trained models (different recipe) failed; the small box with
   the canonical local recipe works; the cause of the old failure is not isolated unless the wrap control fails at
   66². Remove causal language otherwise.
10. Figures: concept (Fig. 1) panel for the two routes; scaling figure with the small-box points and seeds;
    one new panel for the α = 0.05 vs α = 0.1 rollout spectra at 2×. Tables: transfer table gains the small-box rows.
11. Learned baseline (M7): argue explicitly — the global latent model (Phase 3, D_KY 21.4 at L = 100) cannot be run
    at another size at all, which is the comparison that matters; a physical-space local CNN is listed as future
    work. (Optional if time: none.)
12. Cross-checks: `check_numbers.py --paper paperI`, decisions log, fresh internal referee pass.

*Deferred (not needed for submission):* α = 0.025 small box; reach sweep; k_max ablation; retraining the window
models with seeds; Kolmogorov truths beyond what α = 0.1 needs; larger-source-box test at α = 0.05 (R12.3).

**User decision (2026-10-12 20:45): if α = 0.1 passes R9a, Paper I keeps the small-box story** (all three systems on
route (a), with the window-trained models as route (b)).

**Decision point (tomorrow, when α = 0.1 2× lands):** pass R9a → Kolmogorov small-box story in Paper I (items 3, 8,
10). Fail → small-box story for 1D and 2D KS only; Kolmogorov stays window-trained and the α = 0.05/0.1 results
go to the discussion as a limitation.

### Z4. Minimum viable path — Paper II (JAMES)

The model, the 44²/440² results, the tilings, the Kolmogorov section and the 1D section stay. R8 is done.
1. R6 text: replace "same observations and ensemble resources" with the matched-budget numbers (Z1.5). ~30 min.
2. R7: when the reruns finish, regenerate Table/Fig. model_error (with the learned model's 0.19 marked), rewrite
   §6's narrative: physics filter with working estimation wins parameter errors; latent filter wins structural
   (missing-process) errors; recheck the crossover and every abstract/key-point/conclusion sentence that depends
   on it. ~half a day.
3. Cross-paper consistency with Paper I's new framing (R10 item 43: "windows of 264²", "D_KY within …"). ~1 h,
   after Paper I's text settles.
4. Optional: one paragraph on small-box models (D_KY transfers, filter does not: 0.813 / 1.104 at 440²) once the
   p7b 440² control is in. Decide when the number is in.
5. Final checks: build, `check_numbers.py --paper paperII`, grayscale, internal referee re-read of §6.

### Z5. Order and estimate

| When | Paper I | Paper II |
|---|---|---|
| Tonight | evals running (Z2) | R7 running; R6 text (Z4.1) |
| Tomorrow | α = 0.1 decision; ICs + seeds evals; start 88² truth | R7 → §6 rewrite, table/figure (Z4.2) |
| Day 2–3 | writing items 6–11 | cross-paper pass; final checks → coauthor |
| Day 3–4 | figures, checks, internal referee → coauthor | — |

Both papers go to the coauthor together or Paper II first; Paper II cites Paper I as "submitted".
User items outstanding: CRediT and funding statements, DOI/archive (S6.4), arXiv IDs.
**Titles (user, 2026-10-12): keep both current titles unchanged** (Paper I: "Local latent fields for extensive
spatiotemporal chaos: zero-shot domain transfer validated by Lyapunov spectra"; Paper II: "Exactly local ensemble data
assimilation in a learned latent field"). Do not propose title changes.

### Z6. Load reduction and new results (2026-10-12 21:20)

- **Heat (user: "It's been hot all day").** Stopped for good: D/r4e/r4f D_KY evals, α = 0.05 small-box Kolmogorov
  evals (c = 6 3× and IC 1, c = 12 2×), C_eq 88² IC 1, the r4e and reach (r4g) queues, the R4K Kolmogorov filter.
  **Paused (SIGSTOP; `scripts/resume_paused.sh` resumes all):** the Block Z queue (`run_z.py` and its truths, α = 0.1
  training seed 2 and evals), α = 0.1 1× IC 0 eval, C seeds 1–2 and wrap-control 88² evals. Still running: α = 0.1 2×
  IC 1 eval (background QoS, efficiency cores), R7 (3 × 2 threads), p7b 440² filter. Load 175 → ~40.
- **66² results (IC 0, truth 192.50; truth mean 189.05):** C seed 1 183.6 (−4.6%); **C seed 2 169.5 (−12.0%; −10.3%
  against the mean): fails R9a's 5% rule**; wrap control (reach 6, full periodic box) 182.9 (−5.0%), i.e. the
  wrap-around does not break transfer at 66² either, so it is **not** the cause of the old box failure (Z3.9: no
  causal language). With seed 0 at 183.5/187.1, the small-box 2D KS model transfers on 2 of 3 training seeds within
  5%; seed 2 needs its 88² eval and IC 1 before the R9a call. R9a as written requires every seed; report the spread
  honestly (one seed −12%) rather than relax the rule after the fact.
- **R13 `[~]` C seed-2 fix (user, 2026-10-12 21:20: "a slightly longer rollout and more training data").**
  `scripts/run_c2.sh`, applied to **all three seeds** (not just seed 2), two recipes:
  C2 = 144 trajectories (4× C's data) + k_max 48 (k_mid 24) + 12000 propagator steps;
  C3 = C2 + `--periodic-full` (whole 8 × 8 box supervised: reach 3 < half the box, no self-wrap; C's forced-halo
  windows supervise only a 2 × 2 free interior per sample, so C3 gets 16× more supervised sites per sample).
  Read-out written before the runs: 66² IC 0 for all six models; then 44² IC 0 and 66² IC 1 for the better recipe;
  R9a decides. Caveat for the paper: choosing a recipe on 66² uses a test-size truth; the held-out checks are 66²
  IC 1–2 and 88² (truth pending), and C, C2 and C3 are all reported. Load-capped: serial MPS training, ≤ 2 evals at
  3 threads. Trainings ≈ 20–30 min each (~3 h for six), each 66² eval a few hours.
- **Load, 21:26 (user: fan still on).** Further paused: two of R7's three lanes (R7 now runs one case at a time) and
  the p7b 440² filter (`scripts/resume_paused.sh` resumes them too). R13 re-launched with one eval at a time at 2
  threads. Running: α = 0.1 2× IC 1 eval (efficiency cores), one R7 run, one R13 training (GPU), later one R13 eval.

### Z7. Overnight results (2026-10-13 05:30) and ramp-up

- **Kolmogorov α = 0.1 small box, D_KY at 2×:** IC 0 29.5 (truth 30.34, −2.8%), IC 1 30.4 (truth 31.47, −3.3%).
  Within R9a so far (per-IC ±20%, mean within 10%); 3× and the extra seeds/ICs pending. **On track for the small-box
  story in Paper I (user decision, Z3).**
- **R13, 66² IC 0 (truth 192.50; truth mean 189.05):** C2 seeds 0–2: 185.1 / 182.6 / 185.3 (−3.9 / −5.2 / −3.8%;
  vs mean −2.1 / −3.4 / −2.0%). C3: 179.9 / 185.4 / 186.6 (−6.5 / −3.7 / −3.0%). C (old): 183.5 / 183.6 / 169.5.
  **C2 (more data, k_max 48, 12000 steps) is the better recipe and fixes seed 2**; per the pre-registered read-out,
  C2 is the 2D KS small-box model; its 44² IC 0, 66² IC 1 and 88² evals are running (`scripts/run_c2b.sh`). All
  models sit 2–5% low at 66² IC 0; report as a signed bias.
- **R7, seed 0 (best per case; original in parentheses):** ∇² 1.25: 0.166 (0.423); advection 1: 0.228 (0.438);
  advection 2: 0.348 (0.604; two cells still running); J17G δ = 0.1/0.2/0.3: 0.323/0.421/0.471 (0.328/0.427/0.470).
  Parameter inflation fixes the parameter-error cases (physics + estimation now clearly better) and changes nothing for
  the missing-process cases, where the latent filter (0.310/0.341/0.376; 4×16: 0.332/0.355/0.395) still ties or
  wins. Lower RTPS (0.6) is best once the parameter spread is maintained. Seeds 1–2 follow.
- **Ramp-up (user: machine recovered):** all paused jobs resumed (`resume_paused.sh`); R13b added (2 evals at 3 threads).
- To switch in Block Z's queue: the extra-IC evals for "C" (r4c, seed 0) are superseded by C2 seed 0 (in R13b).

### Z8. Kolmogorov flow in Paper I: decision rule, framing, and the fallback (written 2026-10-13 05:50, before R14 results)

**User direction (2026-10-13).** If the C2/C3-recipe Kolmogorov models transfer at α = 0.1 (3× included), test them at
α = 0.05. If they transfer at α = 0.05, or come fairly close, keep the small-box story for every system, with a note
that weaker friction lets the cascade build larger-scale dynamics that a small box captures less well. If they do
not, keep Kolmogorov flow in Paper I using the window-trained models (train once, run on every domain size).

**Runs (R14, `scripts/run_r14.sh`, started 05:39).** 4× larger 1× box data at α = 0.05 and 0.1
(`gen_kolmogorov_box_large.py`, 96 + 6 trajectories each); C2 and C3 recipes at both drags (seed 0); for each:
rollout statistics at 1×/2×/3× with the true PDE (early read, about an hour after training), D_KY at 2× (IC 1, T120
truth) and 3× (IC 1 at α = 0.1; IC 0 at α = 0.05, its only 3× truth). Plus truth-side statistics at 1×/2× for
α = 0.025 and α = 0.05 (window-trained models `k_xs0025`, `k_xs005`; logs `r14_rollout_k_xs*.log`), so the
large-scale share of the *true* flow is known at all three drags.

**Thresholds (fixed now).** *Transfer* = R9a (each IC within ±20% of its truth, mean over ICs within 10%).
*Fairly close* = signed error within ±20% at both 2× and 3×, and a 2× large-scale (|k| < 2) energy share within a
factor 1.5 of the truth's (the old small-box model: −45% and 0.420 vs 0.146). Anything worse = fail. C2 and C3 are
both reported; picking the better one uses test-size truths, which the paper will say.

**Why a fallback is not a retreat: the physical reading we already have.** The true flow's own statistics depend on
the box size at weak drag. Large-scale (|k| < 2) energy share, truth, 1× → 2×: α = 0.1 0.087 → 0.081 (box-size
independent); α = 0.05 0.096 → 0.146 (+50%). Per-box truth D_KY, 1×/2×/3×: α = 0.1 15.9/15.4/17.8; α = 0.05
21.5/24.3/25.1; α = 0.025 25.2/25.8/27.6. Weaker linear drag arrests the upscale energy transfer at a larger scale,
so a larger domain hosts large-scale motion that the 1× box cannot contain, and the 1× box is no longer a
representative sample of the larger flow's local statistics. This also answers referee M11 (the inverse-cascade and
extensivity claims need support): the |k| < 2 share and the per-box D_KY are that support.

**Recommended framing in either case: one rule, a pre-training check, two routes.**
- *Rule (unchanged):* the propagator never reads a site that is not forced from data within its reach.
- *Pre-training check (new, cheap, truth-side):* a small box is a valid training domain when the true flow's local
  statistics (field variance, large-scale energy share) are the same in the box and in a box twice as large. It costs
  two short PDE runs and no Lyapunov spectra. It passes for 1D KS (L = 100), 2D KS (44²; we should run it, see
  below) and Kolmogorov α = 0.1, and fails for Kolmogorov α = 0.05 (and, we expect, 0.025). It is the practical
  design rule for CM1 too: if a candidate box's statistics depend on its size (organized convection, cold pools),
  use windows of a larger simulation.
- *Route (a), small box,* where the check passes; *route (b), windows,* where it fails.

**Case A — C2/C3 transfers at α = 0.05 (or is fairly close).** One recipe, one route for every system. The α
dependence becomes a graded result: transfer error grows as drag weakens, in the direction and roughly in proportion
to the box-size dependence of the true statistics. Write it as "the small box captures the cascade less completely
as friction weakens", quoting the signed errors next to the truth-side |k| < 2 shares. α = 0.025: report the
window-trained results, and say the check predicts the small box degrades further there (one small-box run at
α = 0.025 is cheap and would turn this prediction into a test; add if time allows). Keep the window-trained models in
an appendix as route (b), showing that the same rule also covers windows.

**Case B — C2/C3 fails at α = 0.05.** Kolmogorov uses route (b) at α = 0.05 and 0.025 (existing window-trained
results: train once, run at 1×, 2× and 3×), route (a) at α = 0.1 (small box, passes), and reports the small-box
failure at α = 0.05 *as a test of the pre-training check*, not as an anomaly: the check flags it, the rollout
diagnosis shows the mechanism (spurious large-scale drift, |k| < 2 share 0.420 vs 0.146), and the window route,
which sees the large scales in training, transfers.
- *Benefits.* (1) All three drags stay in the paper, with results and figures that already exist; little new
  compute. (2) The failure becomes evidence for a stated, checkable precondition. A method that works everywhere
  says less about when it works than one with a predicted boundary; the referee asked for discriminating tests (M7)
  and support for the cascade claims (M11), and this provides both. (3) It shows both routes on one system, which is
  exactly the decision a CM1 user faces (Paper 3), and matches Paper II, whose Kolmogorov filter model is
  window-trained at α = 0.05.
- *Costs and how to contain them.* (1) Referee M1 still applies to route (b): the windows (48 sites, 6× the box)
  are larger than the evaluation domains (≤ 3×), so route (b) shows "one model for every domain size", not
  upscaling. Say so plainly; the upscaling evidence is route (a) (1D, 2D KS, Kolmogorov α = 0.1), plus Paper II's
  2D KS assimilation at 440² beyond its 264² parent. (2) Two recipes in one paper invite a cherry-picking
  objection. Counter it by stating the check before the results, applying it to every system, and reporting the
  route-(a) failure rather than omitting it. (3) The window-trained Kolmogorov models have one training seed and
  their own weaknesses (α = 0.025 errors of 8–25%; single-run truths at some sizes). Report them as they are; seeds
  for them are deferred (Z3). (4) The abstract gets one clause longer ("…where a small domain is representative;
  otherwise from windows of a larger one").

**Recommendation.** Use the "one rule, a pre-training check, two routes" framing whatever R14 shows; it is honest in
both cases and costs nothing extra in case A. Decide case A or B on the thresholds above. Two cheap additions make it
solid: (i) the truth-side check for 2D KS (44² vs 88² statistics, no Lyapunov; add after R14's runs) and for 1D
(L = 100 vs 200, existing truth data); (ii) if case A, one small-box run at α = 0.025.
**Unchanged:** C2 is the 2D KS small-box model (Z7); α = 0.1 adopts whichever of the C, C2 and C3 recipes the paper
uses for 2D KS (C2), so that one recipe serves every system, if K-C2 passes at α = 0.1; otherwise r12 (C recipe),
with the recipe difference stated.

### Z9. R14 verdict: case B for Kolmogorov in Paper I (2026-10-13 08:40)

**Result, against the Z8 thresholds (fixed before the runs).** Both C2-recipe and C3-recipe small-box models fail at
α = 0.05:

| model (α = 0.05) | D_KY 2× (truth 46.6) | D_KY 3× (truth 75.2) | \|k\| < 2 share 1× (truth 0.096) | 2× (truth 0.146) | box-period corr 2× (truth 0.41) |
|---|---|---|---|---|---|
| KC2 (4× data, k 48, 12000 steps) | 28.3 (−39%) | 30.3 (−60%) | 0.154 | 0.441 (3.0×) | −0.01 |
| KC3 (KC2 + whole box supervised) | pending | pending | 0.214 | 0.359 (2.5×) | 0.23 |
| old small box (R12, c = 12) | ≈ −45% | — | — | 0.420 | — |
| window-trained `k_xs005` (route b) | (existing) | (existing) | 0.088 | 0.173 (1.18×) | 0.38 |
| window-trained `k_xs0025`, α = 0.025 (truth 1× 0.144, 2× 0.277) | (existing) | (existing) | 0.196 | 0.375 (1.35×) | 0.21 (truth 0.21) |

Neither is "fairly close" (the share criterion fails by itself), so **case B**: Kolmogorov stays in Paper I; α = 0.1
uses the small box (route a), α = 0.05 and 0.025 the window-trained models (route b, train once, run at 1×/2×/3×);
the small-box failure at α = 0.05 is reported with its mechanism (spurious large-scale energy, 2.5–3× the truth's,
in both recipes). More data and a longer rollout did not help, which is itself informative: the failure is not
under-training, it is what the box contains. The window models stay within 1.2–1.35× of the truth's large-scale
share at 2×, which supports the reading. KC3 D_KY evals and the α = 0.1 KC2/KC3 runs continue (the α = 0.1 ones
choose between the C2 recipe and r12 for route a, per Z8).

**Caveat on the Z8 "pre-training check" (found 08:30, before using it in text).** The truth-side large-scale share
also grows with box size for 2D KS (lowest band |k| < 0.35, 44² → 88²: 0.379 → 0.473, +25%, one run, r4c rollout
statistics), and 2D KS transfers. So "share the same in a box twice as large" separates the systems only with a
threshold chosen after the fact (between +25% and +50%). Likely reason: a band share mixes energy *per mode* with the
*number* of modes, which grows with the box; KS has a flat low-k spectrum, so a bigger box adds modes at the same
density. Test running (`scripts/exp_box_check.py`, truth only, 3 ICs per system, logs `box_check_*.log`): per-mode
spectral density at the 2× box's new scale (k1/2) vs the 1× box's largest scale (k1), R_new, for 2D KS and
Kolmogorov α = 0.1/0.05/0.025. **Use rule:** if R_new separates {2D KS, α = 0.1} from {α = 0.05, 0.025} with a gap
across ICs, the check is stated in Paper I as a per-mode-density check, with the numbers. If not, drop the
"pre-training check" claim; present the α = 0.05 failure as an observed limit with its mechanism, and the two routes
as "small box where it transfers, windows otherwise" with no predictive claim. (Caveat on the measurement: the
k1 shell of the 1× Kolmogorov box has 4 modes, so seed-to-seed scatter is large; averaging over 3 ICs and 200
snapshots.)

**Paper II (same morning).** R6 done: latent filter at matched budget (N 16 × 4 tilings) 0.332 / 0.355 / 0.395 at
δ = 0.1/0.2/0.3 vs physics 0.328 / 0.432 / 0.483 → tie, latent, latent. R7 done (3 seeds): physics with working
estimation ∇² 1.25 0.164, adv1 0.225, adv2 0.345 (latent 0.312: physics, physics, latent ~1.1× — check rule);
J17G unchanged. R7b (started 08:15) applies the same fix to the other five estimation rows (∇² 1.1, damping 5/6,
hidden ε 0.1/0.2) so every "est." row uses one remedy; §6 and the table are rewritten once after R7b.
**Load.** 2D KS truths (88² IC 0, 66² IC 2) had been starved at nice 20 (75 s CPU in 11 h); restarted 08:14 at normal
priority. Three 88² evals of superseded models (C seeds 1–2, wrap control) paused (PIDs in
`artifacts/logs/paused_r4c_r4w_88.txt`).

### Z10. Non-critical-path work on the Kolmogorov small-box failure (2026-10-13, user discussion)

**Diagnosis so far.** MSE damping is not the mechanism: at 1× the KC2 model holds the truth's energy and variance for
1000 steps (≈ 20× its training horizon), and longer rollouts did not hurt (−45% → −39% at 2×). The loss of small-scale
activity happens only in larger domains, together with energy piling up at |k| < 2 over the rollout. Working
hypothesis: vorticity dynamics are nonlocal through Biot–Savart (the advecting velocity depends on vorticity
beyond the model's reach); weak drag puts energy at scales larger than the 1× box. 2D KS is local in u (gradients
only), which would explain why it transfers although its large-scale share also grows with box size.

**Queued (side runs, low priority, one at a time, `scripts/run_r15.sh`, tag [R15]).**
- Reach test (`scripts/exp_reach_velocity.py`, truth only, α = 0.1/0.05/0.025, 1×/2×/3×): unexplained velocity
  fraction F(R) from vorticity within R = 1–12 sites. Prediction: at the model reach, F grows with box size at
  α = 0.05/0.025 and not at α = 0.1. It also says how far a model would need to see.
- Option 1: velocity (u, v) as encoder input channels, decoder reconstructs all three (`r15_KV`).
- Option 2: statistics regularizer on 2× rollouts using only 1× data (vorticity variance and forcing-band share,
  which are box-size independent in the truth; choice informed by 2× truth, to be stated) (`r15_KS`).
- Not tried, user decision: loss reweighting / probabilistic objectives (option 3); a parameter-free Biot–Savart
  term on the anchor channel (option 4: exact at any size, but not local, so it gives up the locality that makes the
  method scale; cancelled).
- Larger reach does not help route (a): reach must stay below half the training box (≤ 3 sites in the 1× box), and
  the box contains no scales larger than itself. It can help route (b).

**R15 implemented (commit 3cd4368, `scripts/run_r15.sh`, queued 08:50, one job at a time on efficiency cores, ~2–3 h each).** `r15_KV` (`--input-fields wuv`: encoder sees ω, u, v, velocity computed once from the whole field, decoder reconstructs all three), `r15_KS` (`--w-stats 1.0`, `LargeDomainStats`: same weights run on 2× mosaics of 1× states, persistent pool, 16 no-grad + 4 grad steps, penalty on vorticity variance and forcing-band share vs 1× decoded truth; reuses KC2's Stage A, so KS vs KC2 differs only in the propagator loss; weight set from loss magnitudes, not tuned), `r15_KVS` (both; last, easy to drop). Evals: rollout statistics 1/2/3×, D_KY 2× IC 1 and 3× IC 0. Tests: `tests/unit/test_r15_kolmogorov_options.py` (7) plus 147 related tests pass; old checkpoints bitwise unchanged.
**Prior attempts to cite (found by the R15 agent).** C9(ii) gave a box-trained propagator the exact global Biot–Savart velocity and collapsed under transfer (D_KY ≈ 3 at 2×); option 4 would have repeated it (backbone `local_mlp_2d_biotsavart`). The old `w_spectral` regularizer matched band shares at training size and was worse at 3×. Option 1 differs (velocity encoded once, then carried locally); option 2 differs (free 2× rollouts, size-independent statistics only).

**Ideas for later (speculative, not queued).**
- *Pretrain on the small box, then fine-tune on a larger domain with extra latent channels that carry large-scale
  (nonlocal) quantities, still updated only from neighbouring sites.* Stays local (finite reach), so transfer and
  local DA are kept; remote influence travels as state at up to reach/Δt per step. Rough timescale check at
  α = 0.05: reach ≈ 2.4 length units per 0.2 time units ≈ 12 units per time unit, so crossing a 3× box (≈ 19 units)
  takes ≈ 1.6 time units, against a large-scale drag time 1/α = 20. So signals can keep up. Measures how much
  large-domain data is needed once the small box has done most of the work, which is the CM1 question (large runs
  are expensive). Watch for channel collapse when fine-tuning (resolved before with `--w-logdet`).
- *History in the latent state (delay embedding).* Mori–Zwanzig: the effect of unseen variables (remote or
  sub-site) shows up as memory, so past states can stand in for them. Costs: the phase space grows (Lyapunov needs the
  two-step tangent mode, which exists for the global model), and DA must update lagged states consistently. Free
  learned channels (above) can store the same information without those costs; history is the natural ablation.

**Z10a. Memory channels for the C2-style α = 0.05 model (user idea, 2026-10-13; corrected same day; written down,
not a priority, not queued).** Question: if each site carries a few learned memory channels that only the
propagator writes, does the small-box model stop piling energy into large scales at 2×/3×? No overlapping patches.
- *State.* Per site: the c = 6 encoder channels as now (a function of the current field only; channel 0 the anchor;
  the decoder reads only these) plus m < c memory channels (m = 1–2) that the encoder never produces. Encoding a field
  gives memory = 0 ("unpopulated"); memory fills only as the propagator runs over a history of states. The
  propagator maps (z_t, m_t) → (z_{t+1}, m_{t+1}) with the same local_mlp_2d reach, mixing both sets of channels.
  It is a learned recurrent hidden state per site (not a delay embedding of past encoded states, which was the
  first version of this note), and the map on the extended state is Markovian.
- *Training (KC2 recipe on the 1× box, `kolmogorov_box1x_large_a0.05.h5`, 4× data, k_max 48, 12000 steps, reach 3).*
  Windows of W + k encoded states: a warm-up of W steps with every site's encoder channels forced from data while the
  memory evolves from 0, then the usual forced-halo rollout of k steps (halo encoder channels forced from data, halo
  memory left to the model). Loss on the encoder channels only; gradients through the warm-up (short BPTT).
- *Open design point, to settle in code and a test.* The forced-halo exactness argument (Paper I Prop. 2) covers
  forced data; memory has no data, so halo memory near the window edge is computed without the sites beyond the
  window, and memory can carry information sideways by one reach per step. Two options: (i) memory updates read only
  their own site's memory plus neighbours' encoder channels, so memory never spreads sideways by itself and the
  window computation stays exact given a long enough halo; (ii) allow sideways spread and size the halo for the
  warm-up length, accepting an edge error that a test measures. (i) first.
  *Transfer (2026-10-13 follow-up).* Both options are local and shared across sites, so both run at any domain size. They differ in training. In (i), a site's memory depends only on encoder channels within one reach of it, over its history: a bounded neighbourhood. So window training stays exact if the window adds one reach of forced data beyond the halo, and the half-box rule still holds. In (ii), memory relays memory, so its dependence spreads by one reach per step without bound. In the periodic 1× box, information can then travel round the box through halo memory (which no data overwrites) and come back, so the model can learn box-periodic artefacts, the failure the reach rule exists to prevent. So (i) for small-box training; (ii) only where training data contains the large scales (window training, or fine-tuning on a larger domain, Z10 idea 1), with the halo sized for warm-up + rollout length. Consequence (corrected after the user's question): in (i), information in memory still moves sideways over time, routed through the encoder channels (memory_j → z_j → neighbours' z → their memory), at up to one reach per step, the same light cone as z. A site's memory integrates its neighbours' history, and those neighbours were earlier influenced by sites further away, so far-field information can accumulate in memory over many steps. What (i) forbids is memory relaying memory directly. During training the forced halo overwrites z with data, cutting the route at the window edge; that is what keeps training exact. At inference on a large domain the route is open.
- *Evaluation.* Encode a truth spin-up history, run the warm-up with forced encoder channels to populate memory, then
  roll out freely. Read-out first: rollout statistics at 1×/2×/3× against the truth, as in Z9: the 2× |k| < 2 share
  (truth 0.146; KC2 0.441; window model 0.173) and vorticity std (truth 1.41; KC2 1.15). Go to D_KY only if the 2×
  share is within 1.5× of the truth's (the Z8 "fairly close" share criterion). D_KY is then a standard Benettin run on
  the extended state (z, m), since the map is Markovian; the memory adds directions that are probably contracting.
  In DA, memory is an unobserved part of the ensemble state, updated through ensemble covariances.
- *What exists / what is new.* Exists: forced-halo training and history-window banks (`ks_latent/training/crop.py`,
  J17B's `forced_halo_rollout_history` as a template), local_mlp_2d. New: the extended-state propagator (m extra
  channels in and out, zero memory at encode), the warm-up in training and in `exp_rollout_stats_kolmogorov.py`, and
  tests (m = 0 reproduces KC2 exactly; locality/equivariance of the extended map; option (i) exactness on windows).
  About 4–6 h of code and tests, 25 min training, 1 h rollout statistics. Low priority, after the R15 runs.
