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

## 2026-10-13 — Rewrite per Block Z11/Z3 (paperI-writing agent)

Framing (Z11, user): small-box training (route a) transfers for 1D and 2D KS (2D KS: recipe C2,
`r13_C2_ks2d_s44big_k48_seed{0,1,2}`); Kolmogorov uses the window-trained models (route b:
`k_xs01_v2` [A10 selection], `c19_e0d0_c6`, `k_xs0025`) at all three drags; the small-box Kolmogorov
failure is reported as an observed limit with its rollout signature. Title unchanged.

Decisions and changes:
- **"One rule" stated as a procedure, not a necessity.** The 1D model (whole periodic box, no halo)
  and the reach-6 wrap control (`r4w`, transfers at 66²: 182.9) do not obey "never read an unforced
  site", so the paper says: one forced-halo procedure turns a small box or windows into the same
  limited-area problem (Prop. 2), and keeping reach < half box is a design choice, not a proven
  necessity.
- **No pre-training check** (coordinator, Z13): the box check failed its pre-registered rule (2D KS
  R_new 2.66 ≈ α=0.1 2.75). The Kolmogorov failure is an observed limit; the Biot–Savart reading is
  "consistent with" (reach test: unexplained velocity fraction beyond 3 sites grows 1×→2× by +50% /
  ×2 / ×2.7 at α=0.1/0.05/0.025, but stays ≤5%). 3× reach values: TODO.
- **Truth registry** (`scripts/paper_figures/_truths.py`): added the α=0.1 1× T120 runs (seeds 1, 2:
  12.89, 15.18) and placeholders for 2D KS 66² IC 2 and 88² IC 0 (enter automatically when the JSONs
  exist). Consequence: the α=0.1 1× reference fell 15.88 → 14.65, the selected window model's 1×
  error rose 0.9% → +9.0%, and in Table alpha01_ablation the selected variant v2 now ranks third of
  six by mean |error| (v4 4.7%, v5 5.1%, v2 5.8%). The selection rule is therefore presented as a
  case study with an honest chronology (M6), not as "chose the lowest-error variant"; removed from
  the abstract.
- **Signed errors + ranges in the main text** (M3, minor 4): new Table IV (`tab04_transfer2d.py`
  rewritten, Paper I only; refuses to write into docs/paper1) with both routes, the trivial extensive
  baseline (M7) and per-evaluation ranges; appendix tables state "mean of absolute errors".
- **λ1 vs truth** (M4): new Table `lambda1` (same script). 2D KS +12–18% (small box), +17–20%
  (windows); Kolmogorov windows +16–30% at 1×/2×; 1D +4–9% now stated. KS entropy density not
  computed (needs full spectra; not in the JSONs).
- **Trivial extensive baseline** (M7): column in Table transfer1d (+0.5/−0.4/+0.4%) and rows in
  Table IV; text says transfer of D_KY tests the density where the truth is extensive and
  discriminates only where it is not (Kolmogorov).
- **Lemma 1** (M5): retitled "Law of total variance for rollout targets", stated as a motivating
  identity; the damping is a hypothesis "consistent with" three confounded pairs; Remark (noise)
  "dominated by"; abstract/conclusions softened. New label `rem:mse_limits` in App. B.
- **Prop. 2 / equivariance** (M12): "elementary consequence of locality", certifies the training loss,
  not deployment accuracy; "equivariant under shifts by whole patches" in the contributions.
- **M8 drift**: "no drift" claim replaced. New rollout statistics for C2 seed 0 (3 ICs × 44²/66²/88²;
  `artifacts/experiments/rollout_stats/r13_C2_ks2d_s44big_k48_seed0_ks2d_n1000.json`, run 2026-10-13,
  ~10 min CPU): truth std 3.7→4.0→4.3 and |k|<0.35 share 0.38→0.46→0.51; the small-box model stays
  flat (std ratio 0.87–1.12; share 0.49/0.45/0.42), the window model drifts up (1.03/1.11/1.19).
  Neutral (mean) mode in the model spectrum discussed.
- **M9**: interaction-range caveats (encoder reach 6; linear-stencil R² unexplained); 2D reaches not
  chosen this way (window 6, small box 3 = largest below half the 8-site box).
- **M10**: the 1D coherence penalty described (App. A); sensitivity not measured (stated).
- **M11**: one consistent statement: the Kolmogorov truth is not extensive at these sizes (per-box
  D_KY rises 9–22% from 1× to 3×; |k|<2 share rises); "condenses into a mean state" removed.
- **M1a**: both window models' windows are stated to be larger than every evaluation domain; the
  small-box 2D KS models are the train-small-run-large test.
- **M2.5**: old full-reach box failure: cause not isolated; wrap-around ruled out as sole cause by the
  reach-6 wrap control. Coupling experiments (D2/D3/C9) moved to §7.2 with descriptive labels.
- **Learned baselines** (M7): argued in §8 (global latent cannot run at another size; local
  physical-space surrogate left open).
- **Recipe selection disclosure**: C2 chosen over C/C3 on the 66² IC-0 truth (test size); held-out
  evaluations named; all recipes in App. D.
- Minor: ρ≈0.226 everywhere; spin-up 100–500; 7.3 "40×40 windows of a (6×2π)² domain"; Fig. 8(c)
  2× vs 3× and trace length; attention claim scoped to 1D; App. A seeds; App. C new α=0.1 1× runs.
- **Figures**: new `fig05_routes.py` (Paper I only; replaces box_vs_window in PAPERI_FIGS): signed
  errors by route (hollow bars for n=1) and the |k|<2 rollout share vs width. `fig06_scaling.py`:
  small-box markers (blue). `tabA_architecture.py`: small-box columns in Paper I's
  architecture_dynamics only (paper1/Paper II tables unchanged). `tabD_evaluations.py`: small-box
  rows; truth at 88² once it exists. `tab02_tab03.py`: 88² row once it exists; 1D baseline column.
- Placeholders: `\todo{...}` (red, defined in macros.tex) — grep `\\todo` for the list.

## 2026-10-13 — Internal referee report 2: fixes

Source: scratchpad paperI_referee_2.md (A1–A7, B1–B16, N0/N8, figures). Title unchanged (user decision).

- **A1 λ₁.** All λ₁ statements regenerated from `tables/lambda1.tex`: abstract ("8–36% too high at the
  training size and twice it, within 6% at 3×, 5–18% too low at α=0.025"), intro contribution 3, Sec. VI A
  (0.219 vs 0.203, +8%), Sec. VI C (+16 to +30%), Discussion Limitations, Conclusions (8–18%), Sec. VII
  (sign qualified for α=0.025).
- **A2 data vs reach.** Not resolved in text: the main session is training a reach-3 window model at α=0.05
  (`scripts/run_a2_reach3.sh`, tag `a2_c19_e0d0_c6_r3`). The confound is stated in the abstract ("they also
  have twice the propagator reach"), lead paragraph, intro, Sec. VI C, Discussion, Limitations and
  Conclusions; "windows are needed" removed everywhere. Three `\todo{A2: ...}` placeholders.
- **A3 non-extensivity.** "Window models follow the truth's rising density" removed (Secs. VI C, VI E,
  VIII, Fig. 6 caption); replaced by the density slopes (models +5–9%, truth +16–22%) and the honest reading;
  the Discussion now names the size of the α=0.05 small-box failure as the discriminating evidence.
- **A4 selective reporting.** Table IV and Figs. 5, 6 now report every small-box model the same way at both
  drags (α=0.1: recipe C seed 0 row + pooled row of seeds 1–2, C2, C3; α=0.05: recipe C row + pooled C2, C3
  row; figures pool all). Text: "two of five within 4% at 2×", every model misses at 3× (21–34%).
  The after-the-fact reach-fraction threshold (0.033) is deleted.
- **A5.** No claim of small-box failure at α=0.025; the drag ordering rests on two drags, stated.
- **A6 stale numbers.** ε ablation (v5 2.6% → v6 11.4%); extensive baseline "2.25 times … 2.1%"; wrap
  control −3.0%; Discussion 188.5 within 2.4% (single evaluations 3.2%); App. C three 66² runs, 2.2%,
  spread −1.6/+1.9% and −1.6/+2.1%; Sec. IV "±2%"; App. D −0.3%; Sec. III parameter counts (75,975 small box,
  ~135,000 windows).
- **A7 forced halo.** Necessary on windows, optional on a periodic box (C3, wrap control, 1D) — Sec. III
  paragraph plus one sentence on what a 2-site-interior rollout trains; removed from the abstract headline;
  contribution 2 and the novelty paragraph rewritten.
- **B1** Intro states what is compared in 2D (D_KY, n₊, λ₁; full spectra in 1D); conclusions "exponents".
- **B3** α=0.025 truth bias noted in Sec. VI C. **B5** Fig. 5(e) now uses the main-text α=0.05 window model
  (`c19_e0d0_c6` rollout stats, run 2026-10-13); text quotes its shares (0.19/0.25/0.20 vs 0.10/0.15/0.22).
- **B6** abstract/intro/conclusions: k_max effect "in three confounded comparisons … lowered/tended to".
- **B7** Fig. 7 caption softened (2D KS window model 19% more energetic at 88²).
- **B8** KS "gradients small" marked as an assumption (Sec. IV and VI B), with the 2D KS roughening citation
  (Jayaprakash, Hayot & Pandit 1993, Crossref-verified); 08 reach sentence rewritten.
- **B9** Table IV caption gives initial conditions per seed. **B10** global-model D_KY 21.4 removed (it was
  the replication target, not a measured value in this codebase). **B11** Barbosa & Gauthier (Chaos 2022)
  added (Crossref-verified); Constante-Amores et al. reworded to "reported tests are at the training sizes"
  (abstract confirms shared patch functions; other-size runs not verifiable from the abstract).
- **B12** truth-run table caption: 1D runs not independent ICs; the two L=100 runs differ in K and window.
- **B13** round-off divergence: "a few hundred time units (~35 Lyapunov times)" in App. C and App. E.
- **B14** reconstruction normalization stated in the Table spectra caption. **N8** single-IC vs three-IC
  difference stated in App. D.
- **B4** p7 at 88² (215.9) sentence added in Sec. VI A.
- **B16** small-box n₊ deficit (fewer but faster unstable directions) in Sec. VI A.
- **Not done:** B2 (more ICs: compute), B15 (attention
  counts: provenance not at hand), Fig. 4 0.222 marker, Fig. 7 reference curve, Fig. 2(c) strip, ±15% band
  asymmetry (caption now says it bounds downward uncertainty only).
- **Figures:** Fig. 6 small-box markers half-filled (grayscale); Fig. 7(j) line styles; Fig. C1 legend
  markers/linestyles match panels; Fig. 4(b) and Fig. 2(b) y-label padding; Fig. 5 pooling and ylim.
- **Lead paragraph** 146 words; abstract 205 words. **Acknowledgments** section added (placeholder).
