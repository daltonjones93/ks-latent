# paperI decisions log

Paper I (dynamics and transfer; target Chaos), split from `docs/paper1` per `docs/split_paper_tasks.md`.

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

Figures and tables are generated into `docs/paperI/{figures,tables}` by `make paperI-assets` under
content names (`scripts/paper_figures/_style.py: content_name`); `make paperI` builds the PDF,
runs `check_numbers.py --paper paperI` and fails on undefined or duplicate references. The
generated table bodies are byte-identical to paper1's (checked at the split).

## 2026-10-11 — S3: Paper I content (phase 2)

Text moved from paper1 §1–6, §8–10 and App. A–D, F, then edited per S3.1–S3.13.

- **Front matter (S3.1).** Title without "and exactly local ensemble data assimilation". Key points:
  KP3 is now the failure modes / truth-free selection (128 characters). Abstract 240 words, without
  the DA sentences; it adds the limited-area proof and the two rollout failure modes. PLS without DA.
- **DA material removed, with one pointer each** to the companion paper (`jones2026da`, "in
  preparation" in refs.bib): §1 para 2 and the two DA contribution bullets; §2.3 (latent and
  localized DA); Table 1's "local DA" column and the chen2023roadenkf/tong2026laeenkf row; §3.8;
  the light-cone/localization sentence of §3.3; §9's "What locality buys" and "When to prefer a
  learned model"; the DA limitation item; Conclusions para 3; App. B.4 (Prop. 3 proof and the
  tilings paragraph); the Part II code subsection, the DA test items and DA timings of App. F.
- **Kept, rephrased:** brajard2020/bocquet2020 stay in Table 1 for the Lyapunov column, now cited
  in §2.2. The 440² deployment is quoted once (§3.2 parameter paragraph) as a fact from the
  companion paper, with the correct training domain (264², not 44²).
- **Correction.** The Kolmogorov seed-to-seed spread is 4–12% (inputs 3.7, 9, 11.3, 12.4%);
  paper1 §3.7 says 4–13%. paper1 is frozen and keeps it; Paper I says 4–12% in §3.7 and §4.4.
- **Figure 1** is panels (a)–(c) (`fig01_concept.py` now writes `fig01_concept_transfer`; the
  2 × 2 version is pixel-identical to before). **Table A** is `tabA_architecture_dynamics`
  (no 12-channel column, no training-noise row).
- **Length (S3.9).** Running text 10,181 → 9,609 words (texcount, sections only). Cuts: §2.2
  condensed (all citations kept), §3.5 remarks shortened and the noise-penalty expansion moved to
  App. B, §3.7 ground-truth paragraph reduced to a pointer, §4.4 condensed with the per-run
  Kolmogorov spreads moved to App. C.3, §6.3 per-patch paragraph tightened (the c = 6 full-reach
  control is summarized; its numbers are in Fig. C2b), §8 w_var = 0 sentence dropped (stated in
  §3.5), Discussion paras 1–2 condensed. Reaching 9,000 needs a content decision (candidates: §6.5
  to an appendix, ~250 words; the parameter-count paragraph; Kolmogorov density numbers in §6.4
  that Fig. 6 shows); Chaos has no hard limit.
- **Bibliography (S3.11).** 29 DA-only entries pruned; 57 cited; zero bibtex warnings.
- **Number check (S3.12).** 0 UNMATCHED, 0 SUSPECT; the Fig. 3 axis-range entries ported to
  EXPLAINED for paperI.

## 2026-10-11 — Correction: tangent directions of the model spectra

§3.7 and App. C said every model spectrum uses all d_lat tangent directions. Nine of the 207
evaluations in `artifacts/experiments/appendix_D_evaluations.csv` use fewer: the 2D KS models at
88² (512 of 1536) and the 12-channel models (384 at 44², 512 at 66²). Their D_KY is unaffected:
Benettin with K vectors gives the leading K exponents, and the computation raises an error unless
the cumulative sum turns negative within them (largest D_KY among the nine: 334.6 < 512). Both
sentences now say "all d_lat, or 384–512 where d_lat is larger". Found while writing Paper II App. C.

## 2026-10-11 — S3.9 length pass, second round (user: "please do that")

Running text 9,609 → 9,167 words (texcount, sections only; captions and inline math excluded). Cuts:
- §6.5 "Why a model of finite reach suffices" → App. D, "Scale content of the unstable subspace"
  (`app:chaos_scale`); the Discussion's interaction-range paragraph points to it.
- §3.2 parameter paragraph reduced to the counts and the contrast with a global latent; the
  440² example (12-channel model, 1.6 M grid values, 76,800 latents) now appears only in the
  Discussion's "parameters describe the law" paragraph, which was itself condensed.
- §6.4 Kolmogorov densities: only the two numbers that carry an argument kept (α = 0.1, 3×:
  17.4 vs 17.8; α = 0.025, 3×: 23.7 vs 27.6); the rest are in Fig. 6 and Table 4.
- §6.4 "Beyond the dimension": the large-scale-energy caveat (modes longer than the native box at
  2–3 times the truth's but < 0.5% of the energy) and the reconstruction percentages dropped (both
  in App. D tables); the patch-seam defect kept.
- §6.3 rollout-length paragraph: the k_xs005_noreg numbers left to App. D.
- §7.2 coupling: numbers summarized as ranges (6.7–11.4% with coupling vs 6.3–7.4% without; data
  doubling moves errors by < 3 points); all values in App. D.
- §4.4 and Limitations: the 2D KS truth values and the Kolmogorov 40-vs-120-unit numbers are no
  longer repeated (they are in §4.2 and §4.4).
- §3.5: the closing "trade-off" paragraph merged into the Stage B paragraph.
Stopped ~170 words above 9,000: further cuts would remove results or the Lemma's discussion,
and Chaos has no hard limit. Every cut is reversible from git (commit after afa4310).

## 2026-10-11 — S3.14 Chaos format; S5/S6 checks

- `main.tex` → REVTeX 4.2 (`aip,cha,reprint,amsmath,amssymb,nofootinbib,floatfix`), numbered
  references (aipnum4-2). Lead paragraph adapted from the plain language summary (~140 words).
  Key points and the PLS live on in `main_agu.tex` (AGU class, same sections), not built by `make`.
  Data availability and Author declarations (conflict of interest; CRediT roles: placeholder for
  the coauthor). All 26 floats placed (checked against the aux labels).
- Figures keep their generated widths (single column 3.37 in, double 5.5 in) inside
  `figure`/`figure*`; regenerating every figure at 7.0 in would change paper1's shared scripts.
- **Corrections found by the shared-facts check:** the mean of the two 66² truths is 189.05, printed
  "189.1" in §4.2 (rounded from the table's two decimals); now 189.0 in both papers (paper1 frozen,
  not changed). The 6-channel model's "1.7% and 2.4%" in App. D are single-IC errors against the
  matched truth run, while Table 4's 2.0% and 0.6% are means over ICs against the mean truths; App. D
  now says so, and Paper II §4.2 too.
- §5's pointer to the companion paper now says the 1D DA uses "a model transferred in the same way",
  which holds whether Paper II keeps the full-reach 1D model or switches to the per-patch one (S1D).
- Grayscale (S6.1): hatching added to `box_vs_window` and `appendix_panels`(b), whose orange and
  aqua bars have equal luminance; marker and line style per seed in `truth_convergence`; the
  k_max = 32 curve of `failure`(c) dashed. Regenerated into docs/paperI only.
- The outlook's TODO(G-pass) removed: its examples (convection cells, nested limited-area models)
  make no factual claim that needs a citation.
