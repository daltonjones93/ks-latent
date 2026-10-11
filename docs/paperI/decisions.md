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
  (paper1 Prop. 4) and its proof move to Paper II, as its Prop. 1.
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
