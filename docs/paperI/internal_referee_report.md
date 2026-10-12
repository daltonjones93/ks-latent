# Referee report — *Chaos* (AIP)

**Manuscript:** "Local latent fields for extensive spatiotemporal chaos: zero-shot domain transfer validated by Lyapunov spectra" (Jones & van Leeuwen)
**Sources reviewed:** `docs/paperI/main.tex`, `sections/01–09`, `appendix/A–E`, all `tables/*.tex`, every figure in the compiled `main.pdf` (rendered at 110 dpi). I spot-checked against `artifacts/experiments/appendix_D_evaluations.csv`, `artifacts/experiments/spectra/ks2d_x*.json`, `artifacts/experiments/a10_selection/decision.json` and `docs/paper1/decisions.md` (A10). I checked Paper II (`docs/paperII`) only for consistency between the two papers.

Severity labels: **[FATAL]** means I would reject unless it is fixed. **[SERIOUS]** means the claims must change or new evidence is needed. **[SHOULD FIX]**. **[NITPICK]**.

---

## 1. Summary and recommendation

**Claims.** The authors replace the usual single latent vector with a *latent field*: a few channels at each site of a patch lattice. The encoder, decoder and propagator are weight-shared and have finite reach, so one set of parameters runs on a periodic domain of any size. A 1D Kuramoto–Sivashinsky (KS) model trained at L=100 is transferred zero-shot to L=50–400. Its own Lyapunov spectrum gives a Kaplan–Yorke dimension D_KY 0.2–1.2% below the truth, and three training seeds give the same result. In 2D KS and in Kolmogorov flow with linear drag, models trained on an 8×8-site periodic box transfer poorly. Models trained on forced-halo windows of one large simulation match D_KY to 0.6–2% (2D KS) and 1–9% (Kolmogorov, two stronger drags). The authors also claim three further results: a proof that window training is exactly limited-area training; a lemma saying squared-error rollouts bias D_KY downward; and a selection rule that needs no ground truth.

**Recommendation: major revision.**

The 1D result is clean and worth publishing in *Chaos*. Its strengths are converged truths, a factor of 8 in domain size, three seeds and agreement of the whole spectrum. The idea of judging learned surrogates by their own Lyapunov spectra across domain sizes is also a useful standard for the community.

The 2D part, which carries the paper's main message ("the training distribution decides transfer"), does not support the claims as written:

1. **In 2D, the window-trained models are trained on windows *larger* than every domain they are evaluated on.** The 2D results therefore never test "train small, deploy large".
2. **The box-versus-window comparison has confounds and very little statistical power.**
3. **The headline percentages are finer than the precision of the reference truths and of the model estimates.**
4. **The "Lyapunov spectra" validation in 2D reduces to D_KY and n₊.** The leading exponents of the canonical models are 12–29% too high, and the paper does not report this.

Two further claims are presented more strongly than the evidence allows: the mechanism behind the squared-error damping (Lemma 1), and the truth-free selection rule. Both appear in the abstract.

None of this is beyond repair. Some of it needs new computation, mainly longer or matched truth windows, more initial conditions, and one genuine upscaling test in 2D. The rest needs rewriting.

**Strengths to keep:**
- The truth uncertainty is quantified honestly (Sec. IV D, App. C), including running-D_KY tables. Few ML-surrogate papers do this.
- The 1D section (Sec. V), with spectra collapsing per unit length (Fig. 4) and seed replication.
- The candid failure-modes section and the "What is and is not new" paragraph (02_background.tex:112–125).
- Provenance and generated tables (App. E). The structural propositions are also backed by unit tests.

---

## 2. Major concerns

### M1. [FATAL as framed; fixable] In 2D, "zero-shot transfer to larger domains" is never tested

- **2D KS.** Windows are 40×40 sites (`tables/architecture_dynamics.tex:13`) with h=5.5, so 220×220 physical units. The free interior is 28×28 sites. The evaluation domains are 44², 66² and 88², i.e. 8, 12 and 16 sites per side (04_systems.tex:94–95).
- **Kolmogorov flow.** Windows are 48 sites wide, 36 of them free (`architecture_dynamics.tex:13`). The evaluation channels are 8, 16 and 24 sites wide (1×, 2×, 3×).

Every window-trained 2D evaluation is therefore on a periodic domain *smaller* than the open window the model was trained on. The data come from a 264² or 32π×2π parent simulation, which is 3–16 times larger than any evaluation domain.

The box-trained models, by contrast, are trained at 8×8 and evaluated at larger sizes. The box-vs-window comparison (06_results_2d.tex:16–72, Fig. 5) therefore compares two opposite directions of extrapolation:
- small periodic box → large periodic domain, versus
- large open window → small periodic domain.

What the window result shows is that a rule learned in the interior of a large domain also works on small periodic tori. That is interesting, but it is not the "train once on a small domain, run on any larger one" claim made in:
- the abstract (main.tex:34–41: "runs on any larger domain by adding sites", "miss the dimension of larger domains … while models trained on windows … are within …");
- the lay summary (main.tex:55–60);
- the Introduction (01_introduction.tex:23–25, 35–39, 79–89);
- Fig. 1(b) (deployed lattice larger than the trained one);
- Table I ("Other sizes ✓, Windows ✓" for this work, 02_background.tex:147);
- the Conclusions (09_conclusions.tex:15–21, 37–38: "a model trained once is a model for every domain size").

Nowhere does the paper state that the windows are larger than the evaluation domains. The only genuine 2D upscaling is the 12-channel sibling at 440² in Paper II (08_discussion.tex:50–53), and it has no Lyapunov truth.

It also weakens the motivation (01_introduction.tex:15–17: "every new domain size requires new data and new training"): the window recipe needs a large-domain simulation first.

**Required:** Do one of the following.
- (a) Reframe all 2D claims accurately, e.g. "transfer from open windows of a large simulation to periodic domains of other sizes". Remove "larger" from the 2D statements in the abstract, introduction and conclusions.
- (b) Preferably, add a real upscaling test. Two options:
  - Train windows from a parent domain no larger than 2× the base box and evaluate at 3×–4×.
  - Evaluate the existing canonical models on domains larger than the parent, with truths. For 2D KS at 88², a truth with K≈400–512 is feasible: the 66² truth used K=300, and Appendix E quotes model spectra at 88² in 7 minutes. The paper itself says (04_systems.tex:100–101) that no truth was computed there.

### M2. [SERIOUS] The box-versus-window evidence has confounds, is thin, and is partly selective

1. **Single initial conditions.** Every Kolmogorov box-trained number is n=1 IC (`tables/eval_kolm01.tex:2`, `tables/eval_kolm005.tex:2`). The α=0.1, 3× value comes from an *unstandardized* IC (Fig. 5 caption, 06_results_2d.tex:202–204). The paper itself says single-run model D_KY scatters by 25% at native size (03_method.tex:470; App. C table, 25%). By its own standard (03_method.tex:470–471; 04_systems.tex:180–181: "treat differences of 10–15% … as within the uncertainty"), the 38% (n=1) and 15% box failures are at or barely beyond the noise. The headline "up to 38%" (main.tex:39; 09_conclusions.tex:16) rests on one IC of one model.
2. **Box models also fail at their own size.** 2D KS has 13.7% error at 44² (06_results_2d.tex:22) and Kolmogorov α=0.1 has 23% at 1× (06_results_2d.tex:27–28). These are poorer models, not specifically transfer failures. The 2D KS box model degrades only from 13.7% to 22.7%. Meanwhile the canonical α=0.05 model is *worse* at 1× (8.9%) than the box model (3.7%) (Fig. 5c).
3. **The abstract compares unlike things.** It sets box-trained models (full reach, c=3, k_max=12) against *canonical* models (per-patch, c=6, k_max 16–32). The same-architecture window results are weaker: 0.9%/3.9% for 2D KS, and 8.6%/18.8%/4.0% at α=0.1. At α=0.1, 2×, the same-architecture window model is *worse* than the box model (1.9% → 18.8%, 06_results_2d.tex:66–69).
4. **A relevant result is left out.** The same-architecture window model `p7_ks2d` gives D_KY=215.9 at 88² (`tables/eval_ks2d.tex:3`). That is about 35% below the extensive extrapolation (≈330) the authors use to vouch for the canonical model at the same size (06_results_2d.tex:124–126). It appears in Appendix Table XII but not in Sec. VI B or Fig. 5. If the extensivity extrapolation is good enough to validate the canonical model at 88², it must also be applied to p7.
5. **The proposed mechanism is not tested and is contradicted by the 1D model.** The text blames wrap-around self-interaction: "R_prop=6 on an 8-site ring … sees 75%" (03_method.tex:244–249; 06_results_2d.tex:30–33). Yet the successful 1D model is trained on a 16-site ring with R_enc=R_dec=R_prop=6 (`architecture_dynamics.tex:8–10`). Its propagator alone covers 13/16 sites per step, and encoder + propagator + decoder exceed the ring. The paper never explains why wrap-around ruins 2D and not 1D (06_results_2d.tex:6–7 only asserts it). Note also that the true PDE on a periodic box has the same wrap-around; the argument needs to be made precise.
6. **No control for data volume or diversity.** The amount of box training data is never stated; I could not find it anywhere in Secs. IV, VI or App. A. The "twice as many trajectories" control (07_failure_modes.tex:41–44) is for window training only (4→8 trajectories, one seed). It does not justify "the amount of data does not limit accuracy" in general.

**Required:**
- Report box training data.
- Re-evaluate all box models from ≥3 standardized ICs.
- Add a control that separates the candidate causes: many independent small boxes with matched data volume; box training with a propagator whose reach is ≪ box size; window training from a parent of the box size.
- Show same-architecture comparisons in the abstract.
- Include p7 at 88² in the main text.

### M3. [SERIOUS] The stated precision is finer than the truth or model estimates support

- **Kolmogorov truths.**
  - Five configurations rest on one 40-unit run (04_systems.tex:176–178).
  - 40-unit values differ from 120-unit values by −17% to +23%, and are biased high on average by +9% (App. C:154–158).
  - Even the 120-unit runs have not converged. For example, α=0.1 2× seeds 1 and 2 are still decreasing at t=120 (33.1→31.5) and α=0.05 2× seed 0 too (47.6→46.6) (`tables/running_kolm.tex:2,5–6`).
  - Converged values from different ICs at α=0.05, 2× still differ by 13% (46.6 vs 52.6; App. C:163–165).
- **2D KS truths** use 40-unit windows that are still moving 3–5% over the last 20 units (`tables/running_ks2d.tex`). Yet the 2D KS claim is "0.6–2%".
- **Model and truth use mismatched averaging windows.** Model spectra use 2000 steps: 200 time units in 2D KS and 400 in Kolmogorov (03_method.tex:452–453). Truths use 40 (or 120) units. Because short-window truths are biased high, the comparison is systematically biased toward negative model errors. That bias is about as large as the reported errors (e.g. α=0.025, all single 40-unit truths, model 8–25% low).
- **The reported error metric hides large per-IC deviations.** "Mean of the per-IC relative errors" (Table V caption, 06_results_2d.tex:134–137; App. D:9–10) is in fact the mean of *absolute* per-IC errors (`appendix_D_evaluations.csv`). Individual model runs deviate by:
  - α=0.05, 1×: −1.5 / +4.7 / +20.4%;
  - α=0.05, 2×: +4.3 / −0.1 / −18.6%;
  - α=0.1, 2×: +0.1 / +15.9 / +1.6%;
  - α=0.025, 1×: −47.8 / +8.2 / −18.4%.

  With n=2–3 ICs, the standard errors of these means are several percentage points.
- **Consequence.** "Within 1–9%" (main.tex:41; 01_introduction.tex:86; 09_conclusions.tex:18) and "within 0.6–2%" are not statistically distinguishable from errors two or three times larger. The authors partly acknowledge this (04_systems.tex:178–182), but the abstract and conclusions quote the point estimates without qualification.

**Required:**
- Compute model and truth D_KY over matched windows with matched warm-ups. Either compute the model's 40/120-unit finite-time D_KY from the same ICs, or extend the truths.
- Use ≥5 ICs per cell (model spectra cost about 7 minutes, App. E:67).
- Report confidence intervals (e.g. bootstrap over ICs and truth runs), and say for each cell whether the model–truth difference is significant.
- Replace "1–9%" in the abstract with a statement that reflects the uncertainty.

### M4. [SERIOUS] In 2D, "validated by Lyapunov spectra" means D_KY and n₊ only, and λ₁ is substantially wrong

Only for 1D is the spectrum shown (Fig. 4). For the canonical 2D models, the main text reports D_KY and n₊ (Fig. 6). From `appendix_D_evaluations.csv` against the truth λ₁ in `tables/truth_runs.tex`, the leading exponent of the canonical models is systematically **too high**:

| Configuration | λ₁ error |
|---|---|
| 2D KS, 44² | +15% |
| 2D KS, 66² | +12% |
| Kolmogorov α=0.1, 1× / 2× / 3× | +29% / +17% / +2% |
| Kolmogorov α=0.05, 1× / 2× / 3× | +28% / +16% / +6% |
| Kolmogorov α=0.025 | −5% to −18% |

In 1D, λ₁ is also high: 0.094–0.098 against 0.090, i.e. +4–9% (05_results_1d.tex:17–19), against a truth precise to about 1%.

So matching D_KY involves compensation between too-strong growth and too-strong contraction. In 1D this is visible: the model's exponents are more negative beyond i/L≈0.2 (Fig. 4a), and the departure begins before the D_KY crossing at 0.22, not "beyond i/L≈0.22" as stated at 05_results_1d.tex:110.

This matters in three ways:
- The title promises spectra.
- The Lemma-1 narrative predicts *damped* growth rates (main.tex:43–44; 03_method.tex:371–383), but the canonical models have *excess* growth.
- For data assimilation (Paper II), λ₁ sets error growth more directly than D_KY.

**Required:**
- Show 2D spectra per base box (as Fig. 4) for truth and model.
- Report λ₁ and the Kolmogorov–Sinai entropy density (sum of positive exponents per unit area) alongside D_KY.
- Discuss the λ₁ overestimate and reconcile it with Lemma 1 and Sec. VII D.

### M5. [SERIOUS] Lemma 1 and the "squared-error damping" claim are stated more strongly than the evidence allows

- **Lemma 1 (03_method.tex:357–369) is the law of total variance.** Remark B (App. B:181–190) concedes that it does not describe the map the training converges to. The step from "the conditional mean has less variance" to "contraction along the most unstable directions" and then to "an underestimate of D_KY that grows with k_max" (03_method.tex:374–383) is a heuristic, not a consequence.
- **The authors' own observation contradicts the lemma's literal prediction.** The lemma predicts reduced variance of the predictions. The authors find "every model we checked … keeps its vorticity variance within 2.3%" (07_failure_modes.tex:80–84), and that k_max does not order the variance ratios (`docs/paper1/decisions.md`, C7(b)). The paper then moves the claimed effect to "growth of perturbations", which the lemma does not address.
- **The empirical support is three confounded pairs** (Fig. 8b; 07_failure_modes.tex:76–79):
  - per-patch α=0.05: k_max 12→32 *and* Stage B 6000→9000 steps;
  - 2D-crop: k_max *and* terminal noise 0.02→0.05, and the k_max=12 model was the multistable one, so "lowering D_KY" partly reflects removing the spurious regime;
  - v2→v1: k_max *and* w_ld 0.005→0.0035.

  The cleanest pair changes D_KY by 0.4–6 points, comparable to the IC scatter (M3).
- **Remark 3 (03_method.tex:385–397)** also confounds noise and k_max ("the one run that raised both").
- **The abstract and conclusions** state it as established: "a bias that follows from the structure of the loss" (main.tex:44; 09_conclusions.tex:26).

**Required:**
- Rephrase as a hypothesis "consistent with" the lemma.
- Or run a clean ablation: k_max ∈ {8, 16, 32} with noise, regularizers and training steps fixed, ≥2 seeds and ≥5 ICs; report λ₁ and D_KY.
- Ideally, add a control with a loss that removes the conditional-mean bias (e.g. a CRPS/ensemble or invariant-measure loss). The paper cites jiang2023, schiff2024 and platt2023 but tests none of them.

### M6. [SERIOUS] The truth-free selection rule is a single retrospective case and is described inaccurately

- **It was not blind.** The rule was written after all variants had already been scored against truths. Sec. VI C mentions "the variant previously chosen for its 3× accuracy" (06_results_2d.tex:118–119), and the decision log (`docs/paper1/decisions.md`, A10) says it was motivated by that. The computed metric was not seen in advance, but the authors knew which variants did well. "Chosen without any reference to the truth" (09_conclusions.tex:19–20; Table V caption) needs this context.
- **The free-rollout statistics did not select anything.** All six variants passed the eligibility screen (06_results_2d.tex:114; `decision.json`: max deviation 1.9–6.3% against a 15% tolerance). The selection was made entirely by the 8-step physical-space forecast error. The abstract's "a selection rule built on [free-rollout statistics] chose … the one with the lowest mean error" (main.tex:45–46) misdescribes this.
- **This contradicts Sec. VII A**, where short-horizon losses "predicted neither failure" (07_failure_modes.tex:29–30).
- **The margin is narrow.**
  - Forecast errors span only 0.0191–0.0204, a 7% spread (`tables/alpha01_ablation.tex`).
  - v2 beats the runner-up by 2.6%, against a 2% tie threshold (App. A:44).
  - In D_KY, v2's mean |error| (3.1%) is 1.4–1.7 points below v4 and v6. Per-cell IC half-ranges are up to ±7% (`tables/eval_kolm01.tex:5–9`), so this ranking is not resolved.
- **One application** (one system, one drag, six variants) cannot establish a "model-selection rule" (01_introduction.tex:92–93; 02_background.tex:124–125).

In fairness, the rank correlation between the k=8 forecast error and the mean D_KY error across the six variants is high (Spearman about 0.9). Presented as a case study, this is a nice observation.

**Required:**
- Describe accurately when the rule was written relative to the truth results.
- Remove the claim from the abstract, or test the rule at a second system or drag.
- Report the uncertainty of the D_KY ranking.

### M7. [SERIOUS] What does the transfer test discriminate? No null model or learned baseline

Remark 1 (03_method.tex:174–194) says a local model is extensive by construction. Once its density at the training size is right, transfer of D_KY to larger L follows almost automatically. A trivial predictor illustrates this: D_KY(L) = (D_KY^truth(100)/100)·L gives 44.98 at L=200 (truth 45.18, −0.4%) and 89.96 at L=400 (truth 89.56, +0.4%). That is at least as accurate as the model (−1.2%, −0.2%; Table IV). The Lyapunov transfer test therefore mostly tests the density at the trained size plus locality. It does not show that the learned rule is "the right one" (08_discussion.tex:56–58).

Tests that would actually discriminate:
- finite-size departures from extensivity: L=25–50 in 1D, Kolmogorov 1×–3× where the truth is not extensive, or microextensive oscillations in D_KY(L) (Tajima & Greenside 2002, *to verify*);
- comoving exponents or v_* of the model (v_* is measured for the PDE at 03_method.tex:230–237, but not for the model);
- the localization of covariant Lyapunov vectors;
- two-point correlations or structure functions after transfer;
- forecast skill against simple baselines (Fig. 7j has none).

There is also **no learned baseline**:
- no global-latent model at the training size (to show what the field buys at the same size);
- no physical-space local CNN or learned coarse simulator (Stachenfeld 2022, Kochkov 2021) evaluated by the same Lyapunov protocol.

Without the latter, the case for a *latent* field, as opposed to any weight-shared local surrogate, rests only on the novelty table (Table I). Note that Table I credits Bar-Sinai/Kochkov with "Local ✓, Other sizes ✓"; the only difference is that they did not compute Lyapunov spectra, which this paper could do for them.

**Required:**
- State the trivial extensive baseline next to Table IV.
- Add at least one discriminating diagnostic beyond D_KY.
- Add at least one learned baseline, or argue explicitly why none is needed.

### M8. [SERIOUS] 2D KS: the decoded field drifts with domain size, and the text says it does not

Sec. VI D says: "in 2D KS the field amplitude and large-scale energy share are within the scatter between initial conditions, with no drift" (06_results_2d.tex:177–179). But Table XVI (`tables/eval_spectra.tex:6–8`) shows:

| Size | std (model/truth) | Lowest-band energy, model vs truth |
|---|---|---|
| 44² | 1.03 | — |
| 66² | 1.11 | — |
| 88² | 1.19 | 54.8% vs 45.4% |

`artifacts/experiments/spectra/ks2d_x2.json` gives max|u| 22.6 (model) against 16.3 (truth) at 88². Fig. 7(e) visibly shows a field offset to positive values relative to Fig. 7(d). Two physical points:
- The 2D KS field roughens: the training-domain standard deviation is 6.17 against 3.85–4.06 on the evaluation boxes (`ks2d_x*.json`, `normalization_std` vs `field_std_truth`). The large-scale statistics of u are therefore *not* size-independent. This is known KPZ-type behaviour of 2D KS (Procaccia et al. 1992; Jayaprakash, Hayot & Pandit 1993; Boghosian, Chow & Hwa 1999 — *to verify*).
- The truth projects out k=0 in the base and tangent dynamics (04_systems.tex:84–87). The model's anchor channel carries the patch mean, and nothing removes the corresponding neutral or drifting direction from its latent spectrum.

**Required:**
- Correct the text.
- Report rollout statistics from ≥3 ICs.
- Discuss whether the model respects the u→u+const invariance, and how neutral directions (the mean mode, translations) are handled in the model's spectrum compared with the truth's. With D_KY≈16 at Kolmogorov 1×, one spurious near-zero exponent is a 5% effect.

### M9. [SHOULD FIX → SERIOUS if it remains a design rule] The "interaction range" measurement is confounded and was not used for the 2D designs

- **It is measured on the latent of the 1D model** (03_method.tex:213–228; Fig. 10c), whose encoder already has reach 6 sites. Each latent site therefore already contains physical information from ±6 sites, so the latent interaction range (3–4 sites) is a property of this encoder as much as of KS.
- **It is measured only in 1D.** The 2D reaches (six sites) were not chosen this way. Yet the Discussion and Conclusions present "measure the interaction range … give the model a reach that covers it" as *the* rule (08_discussion.tex:29–44; 09_conclusions.tex:33–36).
- **The linear stencil's R² of ≤2.5% needs explaining.** A linear stencil should capture the KS linear operator, which dominates short-time increments. Is this a consequence of the latent coordinates? It needs a sentence.
- **Compare with the trained propagator.** The quadratic stencil saturates at R²=0.90 even with the whole ring. What does the trained propagator achieve? Without this, "saturation" may reflect the model class rather than the physics.

**Required:** Measure the range on the physical field (or on the per-patch 2D latents) and say how the 2D reaches were chosen.

### M10. [SHOULD FIX] The 1D and 2D results are different recipes, and the 1D one uses non-canonical ingredients

The clean 1D result comes from a different recipe (03_method.tex:423–431; 05_results_1d.tex:123–125):
- whole-domain training;
- full reach;
- c=3;
- a joint auxiliary propagator;
- a "signed latent spatial-coherence penalty (weight 0.01)", disclosed only in a table footnote (`appendix/A_architecture.tex:26`).

The 2D recipe is never tested in 1D. Paper II (`docs/paperII/sections/04_system.tex:125–127`) mentions a per-patch 1D model with D_KY/L≈0.22. Including its transfer table would let one recipe cover all three systems.

Please also describe the spatial-coherence penalty (form, sign, what "signed" means) and show the 1D model's sensitivity to it.

### M11. [SHOULD FIX] Kolmogorov flow: the extensivity and inverse-cascade statements are inconsistent and not supported

- **Contradictory framing.** Sec. IV C says the per-box dimension changes "by at most about 20% … comparable to the uncertainty" (04_systems.tex:133–137), i.e. effectively extensive. The Discussion says it "rises with the channel width at every drag" (08_discussion.tex:11–12) and uses this to excuse transfer errors. The latter is false at α=0.1, 1×→2× (15.9→15.4; `tables/truth_dky.tex:11–12`).
- **The model is not extensive either.** Per box at α=0.1 it gives 16.0, 16.4 and 17.4, which sits awkwardly with Remark 1 and the "transfer error as a probe" paragraph (08_discussion.tex:62–71).
- **The inverse-cascade argument lacks support.** At Re≈43–51 (04_systems.tex:126), forced at n=4, there is little scale separation. The claims that "without drag the flow condenses into a domain-scale mean state" (04_systems.tex:136–137) and that the weakest drag is where "the inverse cascade is least arrested" (08_discussion.tex:66–69) are asserted without a figure or a citation for this regime.

**Required:** Either show spectra or condensate diagnostics, or soften these statements. Settle on one statement about whether the truth is extensive, and apply it consistently.

### M12. [SHOULD FIX] Proposition 2 and "exact equivariance" are oversold

- **Proposition 2** (03_method.tex:272–279; App. B:93–151) follows by induction from locality. It is correct but elementary. It certifies the *limited-area* training loss, while the models are deployed on *periodic* domains where the halo is not forced (App. B:153–161). The proposition guarantees nothing about deployment accuracy. "We prove that this window training is exactly the training of a limited-area model" (main.tex:41–42) should not be a headline claim of the abstract.
- **Equivariance** holds only for shifts by whole patches (03_method.tex:140–147). The PDE has continuous translation symmetry. The 2D seams at patch edges (06_results_2d.tex:179–185; Fig. 7f, energy 2–4 decades above the truth beyond 2π/h) show the consequence. "Exactly translation-equivariant" (01_introduction.tex:71) needs the qualifier "under shifts by whole patches".

### M13. [SHOULD FIX] Comparing a latent spectrum with a physical one

The paper recognizes this (03_method.tex:455–458; 08_discussion.tex:84–87). But the argument "the agreement of the spectra themselves supports [injectivity]" is circular.

Note that in 1D d_lat=48 against 2·D_KY+1≈46 (05_results_1d.tex:6–8). The latent is only just large enough to embed the attractor; with c=2 it would not be. Please comment.

A direct check would make the comparison more convincing: estimate λ₁ in physical space from twin decoded trajectories (perturb, decode, measure separation), or verify E∘D ≈ id on the attractor.

### M14. Novelty and missing prior work (all "*to verify*" — please check before citing)

The positioning is mostly fair. Some closely related work is missing:

- **Extensivity and finite-size deviations:**
  - Tajima & Greenside, PRE 66, 017205 (2002), "microextensive" chaos in KS;
  - Fishman & Egolf, PRL 96, 054103 (2006), deviations from extensivity as a probe of building blocks. This is directly relevant to 08_discussion.tex:62–71, which presents "transfer error as a probe of the physics" as new.
- **Learned local/translation-symmetric surrogates of KS with Lyapunov validation:**
  - Barbosa & Gauthier, *Chaos* 32, 093137 (2022), next-generation RC with local states and translation symmetry for KS;
  - Platt et al., *Neural Networks* (2022), systematic RC study including Lyapunov exponents;
  - Lu, Hunt & Ott, *Chaos* 28, 061104 (2018).
- **Size generalization of local models:** Yehudai et al., ICML 2021, "From local structures to size generalization in GNNs". It makes essentially the thesis of Sec. VI: transfer fails when the distribution of local neighbourhoods differs between training and deployment.
- **Kolmogorov-flow Lyapunov analysis:** Inubushi, Kobayashi, Takehiro & Yamada, PRE 85, 016331 (2012); Platt, Sirovich & Fitzmaurice (1991).
- **2D KS large-scale (KPZ) behaviour:** see M8.
- **Learned limited-area models with boundary forcing** beyond Oskarsson 2023, e.g. recent kilometre-scale ML LAM work.
- **Learned effective dynamics:** Vlachas et al., *Nat. Mach. Intell.* (2022).

The bibliography also lists `devilleroche2026`, `sakarvadia2026`, `kolmogorovregimes2026` and the Wittenberg & Holmes Springer chapter (`wittenberg2001`, which the project brief flagged as "verify authors and year"). Please confirm all of them against primary sources.

---

## 3. Minor concerns

### Internal inconsistencies

1. **Spin-up times.** [SHOULD FIX] 03_method.tex:474 says truth "spin-up of 100–200 time units". App. C:58–59 says 500 for 1D KS (200 at L=22), and App. E:33–34 says 100–500.
2. **When round-off realizations diverge.** [SHOULD FIX] App. C:65–66 says thread-count realizations diverge "after a few hundred time units"; App. E:35–36 says "after a few Lyapunov times". These are different by an order of magnitude (λ₁≈0.09).
3. **1D "independent" truths are not independent ICs.** [SHOULD FIX] All 1D truths use IC seed 0 and differ only by round-off (FFT thread count; App. C:63–71; `tables/truth_runs.tex:1–11`). They share the spin-up and part of the averaging window, so their half-range understates the uncertainty. Also, L=100 lists two runs (K=48, 1200 units; K=35, 2000 units). By the paper's own rule, runs differing only in K "count as one" (App. C:68–70). Please clarify whether they differ in thread count. Using genuinely independent IC seeds would cost about 35 minutes per L=400 truth.
4. **Mean of absolute vs signed errors.** [SHOULD FIX] Table V caption (06_results_2d.tex:134–137), App. D:9–10 and 07_failure_modes.tex:8–9 say "mean of the per-IC relative errors", but the numbers are means of |error|. Say so, and give signed means too. Fig. 5 also plots |error| and loses the sign: box models under-predict.
5. **Fig. 8(c) size.** [SHOULD FIX] 07_failure_modes.tex:55–57 describes the spurious state "at 3×" (D_KY 65–107), but Fig. 8(c) (caption 07_failure_modes.tex:140–144, panel title "regime at 2×") shows 2×. The k_max=32 trace is also only 200 time units long, against 400 for the others.
6. **Crop dimensions for Kolmogorov.** [SHOULD FIX] 07_failure_modes.tex:51 says "A Kolmogorov model trained on 40×40-site crops". The Kolmogorov training strip described in Sec. IV C / Table II is only 8 sites high, so 40×40 crops are impossible. Per the decision log, this model (p9) was trained on a separate 6×6-box dataset, which is never described in Sec. IV or Table II. Please add it.
7. **"Covers 75%".** [NITPICK] 03_method.tex:247–248 and 06_results_2d.tex:30–31: with R_prop=6 on an 8-site ring, the 13-site stencil wraps, so every site sees 100% of the ring (some sites twice). "75%" is reach/size, which is not what "covers" means.
8. **Where the 1D spectra depart.** [NITPICK] 05_results_1d.tex:110–111 says the model departs "beyond i/L≈0.22 … these do not enter D_KY". Fig. 4(a) shows departure from about 0.18–0.20, inside the range that sets D_KY.
9. **λ₁ in 1D.** [NITPICK] 05_results_1d.tex:17–20: λ₁ model 0.094–0.098 against 0.090 (+4–9%) is reported without comment, although the truth is precise to about 1% (see M4).
10. **Reconstruction error normalization.** [NITPICK] Two different normalizations appear without explanation. The 2D KS validation reconstruction MSE is 1.2×10⁻² (Table VII, `tables/architecture_dynamics.tex:27`), while the Discussion and Table XVI give 2.6–2.9% "of the variance" (08_discussion.tex:90; `eval_spectra.tex:6–8`). The ratio is (6.17/3.85)², i.e. training std versus box std. Say which variance is meant.
11. **Density value.** [NITPICK] 03_method.tex:54–55 gives "ρ≈0.22 per unit length" while every other place uses 0.226 (02_background.tex:20; 04_systems.tex:50). Use one value.
12. **Fig. 5 caption.** [NITPICK] 06_results_2d.tex:194–196: "Every window-trained and canonical evaluation is below 25%" omits 88² (p7: −35% against extensive scaling) and α=0.025.
13. **Training seeds.** [NITPICK] Table VII caption: "All runs use … training seed 0" (A_architecture.tex:24) does not cover the seed-1/2 1D models and the α=0.05 s1 model cited in the text.
14. **12-channel comparison.** [NITPICK] App. D:22–30 compares the 12-channel model against single ICs ("one initial condition each") while making claims at the level of 1–3%.

### Clarity and notation

15. **Internal run codes.** [SHOULD FIX] Lab run codes leak into the text and tables:
    - "D2", "D3", "C9" (06_results_2d.tex:40–42, 197–199);
    - `p7_ks2d`, `k_xs01_v7` labelled "variant v6", "same architecture as p2/p7" (`tables/eval_*.tex`; App. D:85–91).

    Replace them with descriptive labels. As described, D2 and C9 ("exact Biot–Savart velocity input" vs "box-trained model with the exact velocity input") are indistinguishable. Explain the difference.
16. **Softmax-attention failure.** [SHOULD FIX] 07_failure_modes.tex:17–30 gives no numbers for 1D: how many attention propagators, which radii, which training settings? The 2D evidence is one unstandardized IC (footnote 1). This supports a design choice stated as fact at 03_method.tex:125–129 ("Every self-attention propagator we tried collapsed"). Either add a small table or soften the statement.
17. **Ramp schedule.** [NITPICK] 03_method.tex:346–350 says the rollout ramp reaches k_max "and held there for the last 30%"; Table VII says "to k_max by 70%". These are consistent, but stating it once would avoid confusion.
18. **Remark 3 inequality.** [NITPICK] 03_method.tex:388–392: ‖J‖_F² ≥ σ₁² is fine, but "penalizes exactly the expansion" overstates it, because the Frobenius norm penalizes all singular values. Say "is dominated by" if most singular values are below 1.
19. **"Light cone".** [NITPICK] The term is used (03_method.tex:252–253, Eq. 4) for the reach bound R_k, then separately for v_*. Use different words.
20. **Deferred definitions.** [NITPICK] "Base box" and "1×" are used in Fig. 2 and Sec. IV C before Table II defines the Kolmogorov sizes. "Native" is undefined in Table V.

### Figures and tables

21. **Fig. 2(c).** [NITPICK] The strip is very thin at print size; the y-ticks (0, 2π) crowd. Consider a taller inset of one window.
22. **Fig. 4.** [NITPICK] In (b) the y-label (1/L)Σλ is rendered vertically and collides with the tick labels. The shared legend overlaps the x-axis labels. "0.226" is drawn as a line although the model crosses at 0.222; add both.
23. **Fig. 5.** [SHOULD FIX] Show signed errors or mark the sign. Distinguish n=1 bars visually. Bars with one IC and bars with three ICs currently look identical.
24. **Fig. 6.** [NITPICK] The ±15% band for single-run truths is a choice. App. C:154–158 suggests the 40-unit truths are biased (+9%), not only noisy, so a symmetric band is misleading.
25. **Fig. 7.** [NITPICK] One colourbar labelled u/σ serves the KS fields and the Kolmogorov vorticity. Label it generically or give each its own. Panel (j) needs a reference, e.g. persistence, or the decorrelation of two PDE runs from slightly perturbed ICs, which gives the same 2–3 Lyapunov-time scale by construction.
26. **Fig. 8(a).** [NITPICK] The heatmap has no colourbar.
27. **Table XVI.** [NITPICK] The caption mixes "model rollout" (low band) and "reconstruction" (high band) in the last two columns. Give both for both bands, and add the number of ICs.
28. **Fig. 9.** [NITPICK] The float "is stuck" in the log (`main.log`, around line 722); check its placement in the final PDF.

### Submission hygiene

29. **Placeholder.** [SHOULD FIX] main.tex:91 renders a red "[CRediT author contributions]" placeholder in the PDF.
30. **Data availability.** [SHOULD FIX] main.tex:81–85 says "upon reasonable request". For an ML-surrogate paper whose claims rest on trained checkpoints and result files, a public archive (Zenodo DOI) is strongly expected. The commented note (App. E:21–22) says results came from "a working tree … with uncommitted changes". Tag a release before submission.
31. **Date.** [NITPICK] main.tex:28 uses `\date{\today}`.

### Consistency with Paper II

No contradictions found in the numbers I checked:
- 2D KS truths 80.7/81.8/83.6 and 185.6/192.5;
- the 12-channel model at 84.7 and 188.5;
- the 1D truths 45.2 and 89.6 with n₊ 26–27 and 51;
- the Kolmogorov 3× truth of 75.2.

Two points:
- Paper II uses `jones2026transfer` with this paper's exact title, and this paper cites `jones2026da` with Paper II's exact title. Good.
- Paper II's introduction (`paperII/sections/01_introduction.tex:37–40`) repeats "reproduces the true D_KY to within 0.6–2% at 44² and 66²". If M3 changes how that number is stated, update Paper II too.

---

## 4. The five changes that would most improve the chance of acceptance

Ordered by impact relative to effort.

1. **Reframe the 2D "transfer" honestly, and add one real upscaling test** (M1, M2). In the abstract, introduction, Fig. 1, Table I and the conclusions, state that the 2D windows are larger than the evaluation domains. Then either:
   - train windows from a small parent (≤2× box) and evaluate at 3–4×, or
   - compute a 2D KS truth at 88² (K≈400–512) and Kolmogorov truths at ≥8× width, and evaluate the existing models there.

   Include p7 at 88² in the main text. Mostly writing, plus a few days of compute.
2. **Make the statistics match the claims** (M3). Compute model D_KY over the same finite windows and warm-ups as the truths, from the same ICs. Use ≥5 ICs per model cell and genuinely independent IC seeds for the 1D truths. Report confidence intervals and signed errors. Replace the "0.6–2%" and "1–9%" headline figures with interval statements. Moderate compute: model spectra take about 7 minutes each.
3. **Show more than D_KY in 2D** (M4, M8). Add per-area spectra (as Fig. 4), λ₁ and the KS entropy density for the canonical 2D models. Discuss the +12–29% λ₁ excess, correct the "no drift" statement for 2D KS at 88², and explain how neutral modes are treated. Mostly analysis of existing files.
4. **Narrow the mechanistic and methodological claims** (M5, M6, M12). Recast Lemma 1 as a motivating identity and say explicitly that it does not predict the trained map. Run one clean k_max ablation with noise, regularizers and steps fixed, and ≥2 seeds. Present the selection rule as a single case study, with an accurate account of when it was written. Drop Proposition 2 and the selection rule from the abstract, or qualify them. Mostly writing plus one small training sweep.
5. **Add baselines and controls that discriminate** (M2.5–6, M7). Report the trivial extensive-scaling baseline next to Table IV. Add one box control at matched data volume and ≥3 standardized ICs, plus one control that isolates wrap-around (small reach on the box). Add one learned baseline: a physical-space local CNN, or a global latent at the training size, judged by the same Lyapunov protocol. Cite the extensivity and local-surrogate literature in M14. This is the largest effort, but it is what turns "this recipe worked" into "this is why".
