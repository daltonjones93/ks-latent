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
