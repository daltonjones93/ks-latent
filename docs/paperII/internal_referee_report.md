# Referee report: "Exactly local ensemble data assimilation in a learned latent field" (Jones & van Leeuwen), submitted to JAMES

Line references are to `docs/paperII/` (for example `sections/06_physics_comparison.tex:49`). Where I checked a number against the result files, the file is named under `artifacts/experiments/da2d/`. Severity labels: **fatal** (the claim cannot stand as written), **serious** (must be addressed for acceptance), **should fix**, **nitpick**.

---

## 1. Summary and recommendation

**Claims.** The authors run ensemble data assimilation in the latent space of the "local latent field" model of a companion paper (Paper I): a few channels per site of a patch lattice, with weight-shared operators of finite reach. They make four claims.
1. With a decoder that reconstructs each patch from its own site only, the latent observation operator is block-diagonal, so a per-site, R-localized EnKF in latent space is "exact" (Proposition 1).
2. Running four filters on lattices shifted by half a patch and blending the results ("shifted tilings") lowers the analysis error by about 30%.
3. In 2D Kuramoto–Sivashinsky (KS) twin experiments, one noisy-trained model with one filter setting gives an analysis error of 0.30–0.33 on every domain from 44² to 440². About 32 members per tiling suffice, while an unlocalized latent filter fails once the number of unstable directions exceeds the ensemble size.
4. Against a physical-space LETKF with deliberately wrong models, given RTPS tuning, online parameter estimation and offline bias correction, the latent filter wins once the physics model's "state-dependent one-cycle error" exceeds about a sixth of the observation error. A missing anisotropic process of increasing strength is offered as confirmation.

**Recommendation: major revision.** Claims 2 and 3 are well supported within the twin-experiment setting. The controls are unusually careful: equal total membership, four ensembles on one tiling, the decoder floor, a physical-space analysis using the same learned model, an unlocalized baseline, a start from climatology, misspecified R, and 300-cycle runs. Provenance is exemplary.

The third key point and the comparison it rests on (claim 4) are not adequately supported, for four reasons:
- **The comparator's remedies were not tuned together with its inflation.** The result files show that online parameter estimation stalled with its spread collapsed onto the floor, at parameter values far from the truth. These are exactly the cases counted as "latent wins".
- **Ensemble resources differ.** The latent filter uses 4 × 64 members against the comparator's 64, although the text says the two filters have "the same … ensemble resources".
- **The x-axis of the crossover is inconsistent with the remedies applied.** It measures the error left after bias correction but before parameter estimation. The hidden-field cases, which lie to the right of the claimed crossover and are not latent wins, are set aside after the fact.
- **The threshold "a sixth of the observation error" is stated more generally than the evidence allows.** It is a property of one observing network, one cycle length and one learned model.

Separately, "exact" (title, abstract, key points, conclusions) overstates what Proposition 1 gives. It restores the locality of a grid-point observation operator that physical-space LETKF has by construction, and the localized filter remains an approximation.

None of this is beyond repair. Rerunning the comparator with jointly tuned remedies and equal membership, and reframing the crossover, would make the paper publishable. As it stands, the third key point, part of the abstract and the last paragraph of the conclusions are not supported by the evidence.

**Genuine strengths.**
- The equal-membership and same-tiling controls (`sections/05_tilings_and_sizes.tex:73-80`) convincingly separate offset diversity from ensemble size.
- The spectral decomposition of inter-tiling error correlation (`sections/05_tilings_and_sizes.tex:82-96`, Fig. 3) explains why cycle spinning works here.
- The test of a physical-space analysis using the same learned model (`sections/05_tilings_and_sizes.tex:166-178`) is the right control and is rarely done.
- The limitations list (`sections/11_discussion.tex:46-74`) is candid.
- Full reproducibility tooling (`appendix/E_reproducibility.tex`).

---

## 2. Major concerns

### M1. The physics-based comparator's remedies were not tuned jointly, and its parameter estimation stalled. **Fatal for the crossover claim as stated.**

The text says each physics filter is "given, and tuned on seed 0 over, the RTPS weight and the localization radius; online estimation …; and bias correction" (`sections/06_physics_comparison.tex:33-38`; `sections/03_method.tex:218-226`; `appendix/A_settings.tex:156-170`). The result files show that the remedies were run only at the RTPS and localization chosen for the filter *without* remedies, which are heavily inflated settings for a wrong model. Inflation was never retuned once a remedy removed most of the model error.

- **Bias-corrected forcing.** The state-dependent error left is only 0.008–0.016 (`tables/model_error.tex:5-6`), so this model is nearly perfect, yet the filter reaches 0.225. That is three times the perfect-model 0.076, at spread/error 1.82 (`i11_J16b_frc0.4_rtps0.7_bc_q{0,0.5,1,2}_s0.json`: 0.229–0.236 regardless of q). The perfect-model filter is best at RTPS 0 with calibration 1.05 (`i11_L44_st8_c24_rtps0.json`). The forcing-with-bias-correction run kept RTPS 0.7.
- **∇² coefficient 1.1 with estimation.** The estimate converges to 1.005, yet the error is 0.229 at calibration 1.85 (`i11_J16a_lap1.1_rtps0.7_s0.json`). Three different "physics wins" all sit at exactly 0.225 (`tables/model_error.tex:5-7`), which points to a configuration-imposed floor rather than to model error.
- **∇² coefficient 1.25 with estimation (a "latent win").** The parameter ensemble collapses onto the 0.005 spread floor by about cycle 30 and stalls at a ≈ 1.17 against the true 1.0 (`i11_J16a_lap1.25_rtps0.8_s0.json`, per-cycle `theta_mean`/`theta_std`). Calibration is 3.3. About 70% of the parameter error is never removed.
- **Advection with estimation (two "latent wins").** The speed stalls at 0.068 (true 0) and at 0.395 (true 0) for speeds 1 and 2, again at the spread floor, with calibration 4.7 and 6.4 (`i11_J16a_adv{1,2}_*_s0.json`).

Advection speed and the ∇² coefficient lie *inside* the model family. A working state-augmentation scheme should recover them almost exactly; the authors' own 1.1 case shows it can. These cases are therefore not evidence about "the part that parameter estimation and bias correction cannot reach" (`sections/06_physics_comparison.tex:133-135`). They are evidence that this implementation did not converge: spread relaxation of 0.9 plus a floor of 0.005, with no parameter-spread inflation (cf. Aksoy et al. 2006; Ruiz et al. 2013, already cited), and inflation fixed at a setting tuned for the uncorrected model.

The same pattern appears in the structural cases:
- **Damping order 8.** Joint (a, b, g) estimation was run only at c = 16, RTPS 0.7 (`i11_J17H_p8_c16_rtps0.7_abg_s0.json`, 0.425). The best tuned setting was c = 8.25, RTPS 0.6 (0.420), where estimation was never tried.
- **Missing process δ = 0.2 and 0.3.** Estimation was run only at RTPS 0.8 and 0.9. The error falls from 0.9 to 0.8 (0.480 → 0.427 at δ = 0.2; 0.520 → 0.470 at δ = 0.3), so the optimum may lie at or below the edge of what was tried. The chosen configurations have spread/error 2.0–2.1.
- **Localization grid edge.** The localization grid {8.25, 16, 24} has its optimum at the smallest value (8.25) for every structural case (`appendix/A_settings.tex:156-157`). That is the edge of the grid, and tighter localization usually helps under model error.

The authors' own decision log records that an earlier, incomplete physics tuning grid had given a δ = 0.2 ratio of 1.43 (physics 0.489); completing the grid brought it to 1.27 (physics 0.433). It also produced a *true-model* filter at 0.469 instead of 0.087 (`docs/paperII/decisions.md:79-92`). The comparison is clearly sensitive to comparator tuning.

**Required:**
- Retune RTPS (or multiplicative/additive inflation) and localization *with each remedy switched on*.
- Fix the parameter-estimation spread handling: for example, an inflated parameter spread or a larger floor, and a demonstration that a = 1.25 → 1.0 and speed → 0 are recovered.
- Report the estimated parameters and their spreads for every case, not only for the hidden field (`sections/06_physics_comparison.tex:83`).
- Consider adaptive inflation (Anderson 2009; Miyoshi 2011; Kotsuki et al. 2017 for adaptive RTPS; to verify). It is the standard remedy for unknown model error and removes the "narrow RTPS window" argument (`sections/06_physics_comparison.tex:114-121`).

I expect at least the a = 1.25 and advection cases to become physics wins. That would leave order 8 and δ = 0.2/0.3 as the only latent wins.

### M2. Unequal ensemble resources. **Serious.**

The comparator runs 64 members (`appendix/A_settings.tex:127`). The latent filter runs four tilings of 64, so 256 member forecasts and four analyses per cycle. Yet the text states "The comparison is between two filters with the same observations and ensemble resources" (`sections/06_physics_comparison.tex:110-111`). The equal-cost latent configurations are already in the paper:
- 4 × 16 members: mean 0.335 over seeds 0–2.
- 1 × 64 members: mean 0.435 (`tables/members.tex:12-14`).

Against 1 × 64, the a = 1.25 (0.420), order 8 (0.409) and advection 1 (0.436) cases become ties or physics leads. Against 4 × 16, the latent advantages shrink to about 1.22–1.30. The crossover moves.

**Required:**
- Either run the comparator with 256 members, or compare against 4 × 16 latent members, and report both.
- Correct the sentence at `sections/06_physics_comparison.tex:110-111`.
- If cost is the argument (latent members are cheap), make it explicit with wall times for both filters. The paper gives none for the physics filter (`sections/08_forecasts_cost.tex:99-120`).

### M3. The crossover's x-axis, the post-hoc exclusion of the hidden-field cases, and the "sixth of the observation error" generalization. **Serious.**

**(a) The x-axis does not match the remedies.** "State-dependent one-cycle error" is the standard deviation of r − r̄, the residual after bias correction but *before* parameter estimation (`appendix/A_settings.tex:169-170`; `sections/06_physics_comparison.tex:60-72`). The outcomes, however, use the best remedy, which includes parameter estimation. For the 1.1 coefficient the x-value (0.077) overstates the error that remains after estimation. For a = 1.25 and the advection cases, the x-value counts as "uncorrectable" an error that estimation can, in principle, remove completely. The abstract and conclusions describe the axis as "the part that bias correction cannot remove" (`main.tex:61-62`; `sections/12_conclusions.tex:28-29`). `sections/06_physics_comparison.tex:133-135` and `sections/11_discussion.tex:32-34` call it what "parameter estimation and bias correction cannot reach". These are different quantities. Plot the one-cycle error of the model *as actually run*, with the estimated parameters.

**(b) The hidden-field cases contradict the rule and are excluded post hoc.** On the stated axis they lie at about 0.25 and 0.56 of the observation error. That is to the right of the crossover, yet both are ties with the physics filter ahead on every seed (`tables/model_error.tex:14-15`; diamonds in Fig. 6a).

The authors exclude them because the physics error there "is largely parametric" (`sections/06_physics_comparison.tex:73-84`). But the a = 1.25 and advection cases are equally parametric, and they are *included* as latent wins. The rule is applied asymmetrically. The Fig. 6 caption ("above it the latent filter does", `sections/06_physics_comparison.tex:169-171`) is contradicted by its own diamonds.

**(c) The crossover cannot be general in units of the observation error.** Its location is where the comparator's error rises past the latent filter's fixed 0.31, and that depends on:
- the observation spacing: the latent error is 0.226 at spacing 1.4 and 0.769 at spacing 5.5 (`sections/08_forecasts_cost.tex:33-46`);
- the cycle length, since one-cycle error grows with T;
- the ensemble size;
- the learned model's quality.

The comparison was run at one density, one cycle length and one domain size (plus two spot checks at 220²). The key point ("beats tuned physics-based filters once their uncorrectable model error exceeds about a sixth of the observation error", `main.tex:44`) and the conclusions ("That number … is the quantity to estimate before replacing a physical model", `sections/12_conclusions.tex:30-34`) read as a transferable rule.

**Required:**
- Either repeat the comparison at a second observation density and a second cycle length, or restate the crossover as specific to this configuration.
- A more interpretable normalization would be the learned model's own one-cycle error. Table 6 implies 0.19 in field units at lead 1 from the encoded truth, including a decoder floor of about 0.13 (`tables/forecast.tex:2`). That is *larger* than the 0.10 crossover. Why does a physics model with a smaller one-cycle error lose? The honest candidate answer is the comparator's handling of model error (M1), not the models themselves. The paper states the latent model's one-cycle error "is small" without giving a number (`sections/05_tilings_and_sizes.tex:151-154`). Give it, and put it on the axis of Fig. 6a.

**(d) The confirmatory test is weak.**
- Three levels of δ, all at one size and density.
- The "tie at 0.15" is a 5.5% latent lead on every seed.
- In the first set, order 6 at 0.20 of the observation error is a tie with physics ahead.
- Taken together, the "crossover" lies somewhere between about 0.12 and 0.31 of the observation error.
- "Confirms this crossover" (`main.tex:63`) should be "is consistent with".
- Report the ratio-equals-1 point by interpolation with an uncertainty, rather than "about a sixth".

**(e) A pre-registered missing-process experiment is unreported.** The repository's planning notes describe a missing cubic process (u′³ saturation), pre-registered on 2026-10-10 and designed as "the headline structural-error result" (`docs/full_paper_tasks.md:2280-2412`). Its physics-filter tuning was run (`i11_J17A_pde_*`; best 0.190 at seed 0 against the latent filter's usual ~0.31) and then stopped. Since the paper states that the decision rule "was written down before the runs" (`sections/06_physics_comparison.tex:39-41`), every pre-registered comparison should be reported, including stopped ones and the reason they were stopped. A supplement is enough. This case is probably consistent with the crossover, so reporting it costs nothing.

### M4. The comparison omits the obvious third option: a physical model corrected by a learned term trained on the same data. **Serious for a JAMES readership.**

The latent model is trained on complete, noisy fields of the true system. With those same data, the standard alternative is to keep the physics model and learn a state-dependent correction. Examples (to verify): Farchi et al. 2021, QJRMS; Brajard et al. 2021, Phil. Trans.; Chen et al. 2022, JAMES (FV3-GFS); Wikner et al. 2020/2021, Chaos (hybrid reservoir); Arcomano et al. 2022/2023, JAMES/GRL (hybrid SPEEDY). That alternative removes exactly the state-dependent residual that the crossover is built on.

The offline bias correction used (`sections/03_method.tex:222-226`) is the state-independent special case. Without a hybrid comparator, "where the structural error of the physical model is large … a local model learned from noisy data is the better forecast model" (`sections/12_conclusions.tex:32-34`) is not established. At minimum, discuss this alternative and temper the conclusions. Ideally, run a local learned correction to the physics model using the same weight-shared architecture.

### M5. "Exact" is overstated, and Proposition 1 is close to trivial. **Serious (framing).**

The proposition says that if each patch is decoded from its own site and observations are grid-point values, then observation i depends only on z_{s(i)} (`sections/03_method.tex:119-135`; proof `appendix/B_proof.tex:12-19`). That is immediate from the architecture. Part (a) is a standard Curtis–Powell–Reid column colouring. Part (b) states that the local analysis needs no truncation of h. That is precisely the property a physical-space LETKF with point observations has by construction, so the proposition restores parity rather than adding anything beyond it. Specific issues:

- **The abstract's sentence is misleading.** "a per-site ensemble Kalman filter in the latent space is exact" (`main.tex:54`; also `sections/01_introduction.tex:46-47` and `sections/12_conclusions.tex:9`). A DA reader will take "exact" to mean Bayes-exact or Kalman-exact. The filter is localized (the Gaspari–Cohn half-width of 1.5 sites removes true covariances), has finite N and perturbed observations, and uses a gain *linearized* at the ensemble mean (`appendix/A_settings.tex:70`, "the gain uses J"). Say instead: "the localized analysis requires no truncation of the latent observation operator", or "the observation operator is exactly local".
- **The Jacobian and its colouring are a self-created need.** A standard LETKF forms Y = h(Z) − mean h(Z) with one decoder pass per member. Because h_i depends only on z_{s(i)}, that is exactly local too, with no Jacobian at all. Contribution 1's emphasis on "as many forward passes as there are channels" (`sections/01_introduction.tex:65-69`; Fig. D1) is therefore not a real cost advantage: the Jacobian costs 2.4–119 ms against 35 s per cycle (`sections/08_forecasts_cost.tex:117-120`). Justify the linearized gain (the paper reports Gaussian priors, `appendix/D_da_details.tex:109-114`), or drop the emphasis. The dense reverse-mode baseline in Fig. D1 is a strawman.
- **Exactness holds only for local observation operators.** Many Earth-system observations are spatially integrated or nonlocal: radiances, GNSS-RO, radar beams with finite volumes, precipitation accumulations. The property does not carry over to them, and the paper should say so (`sections/11_discussion.tex:55-60` discusses density only).
- **The per-patch decoder's cost to DA is under-examined.** A per-patch 1D model "assimilates about twice as badly" as the overlapping-decoder model (0.84–0.96 against 0.43–0.45, `sections/04_system.tex:125-128`). The 2D single-tiling seam error is what the tilings then repair. Proposition 1 already notes that a decoder with R_dec = 1 keeps the local analysis exact if N_s includes neighbouring sites (`sections/03_method.tex:139-143`; `appendix/B_proof.tex:50-52`). An R_dec = 1 2D model with one tiling is the natural comparator for "four per-patch tilings". Without it, the 30% gain may be largely the repair of a problem the design choice created.
- **Title.** "Exactly local" in the title should be reconsidered.

### M6. "Trained only on noisy data" overstates the difficulty. **Serious (framing).**

Training fields are complete 768² grids with white noise of 0.1 times the training standard deviation per grid point, one fixed realization (`sections/03_method.tex:76-83`). The KS pattern wavelength is about 26 grid points. White noise at 1% of the variance per point is almost perfectly separable by any spatial smoothing; a 16×16 patch average reduces it a further 16-fold. This is a mild denoising regularizer, and the 4–5% gain (`sections/07_noisy_training.tex:6-12`) is consistent with that. It says little about learning from the sparse, irregular, biased observations that the motivating application (`sections/01_introduction.tex:98-101`; `sections/11_discussion.tex:76-83`) would provide.

The key points (`main.tex:43`) and the PLS ("the learned model can be trained on noisy data alone", `main.tex:76`) will mislead non-specialists. The limitation is stated only at `sections/11_discussion.tex:64-67`.

In addition, the 4–5% improvement is 0.012–0.018 in absolute terms (`sections/07_noisy_training.tex:8-12`). By the authors' own criterion, "differences below about 0.02 are within run-to-run variation" (`appendix/D_da_details.tex:121-122`). Either show that the paired-seed difference is significant (for example, the 220²–440² runs, where 10-cycle means vary by 1% or less) or soften "is the better for it" (`sections/07_noisy_training.tex:7`) and contribution 5 (`sections/01_introduction.tex:88-90`).

### M7. Ensemble-size argument: inconsistent density and a weak test. **Should fix.**

Eq. (3) (`sections/03_method.tex:195-205`) is used with ρ = 0.043 per unit area, the *D_KY* density (`sections/05_tilings_and_sizes.tex:207-209`), and with ρ = 0.22 per unit length in 1D, also a D_KY density. Elsewhere the paper argues the requirement is the number of *unstable* directions, n_+ + 1 (`sections/02_background.tex:28-32`), and uses the n_+ density for the unlocalized filter ("about 170 at 88²", `sections/05_tilings_and_sizes.tex:215`). The n_+ density is about half (43/1936 ≈ 0.022), giving about 19 rather than 37. The integration area (the full Gaspari–Cohn support, where weights are tiny) is also a choice.

The saturation curve is soft: 16 members are 9% worse, 32 members 2% worse (`tables/members.tex:5-8`). "Consistent with that saturation" is therefore not a test. Use one density consistently, show the predicted number with its range, and describe the result as qualitative.

The unlocalized-filter claim, "fails once the attractor has more unstable directions than the ensemble has members" (`main.tex:58-59`; `sections/12_conclusions.tex:21-22`), rests on three sizes and two ensemble sizes. The unlocalized filter also used the localized filter's RTPS and model noise without retuning (`sections/05_tilings_and_sizes.tex:211-224`). Tune it, or state that it was not tuned. "Degrades well before it fails" also deserves mention: at 44², 64 members is above n_+ and still 60% worse.

### M8. Calibration of the blended ensemble is a by-product of how members are paired across tilings. **Should fix.**

Blended member e pairs member e of each independently cycled tiling (`appendix/A_settings.tex:52-54`). Its spread is therefore √(Σ_k w_k² σ_k²). It omits the between-tiling disagreement of the means, and it depends purely on the weight arithmetic:
- four ensembles on one tiling: 0.44–0.48 ≈ 0.9/√4;
- uniform weights: 0.63;
- sixteen tilings: 0.51 (`tables/members.tex:19,26-27`).

That four Hann-weighted tilings come out at 0.92–0.99 is a coincidence of how fast error and spread fall. It is not evidence of a calibrated posterior, yet it drives the choice of K = 4 and of Hann weights ("four is the calibrated choice", `sections/05_tilings_and_sizes.tex:68-71`). Uniform weights are *more* accurate (0.313 against 0.324).

**Required:**
- Define the blended uncertainty in a principled way, for example including the between-tiling variance.
- Show rank histograms and spread–skill by wavenumber band and by position in the patch. Edges and centres differ by 30%, Fig. 3a.
- Report prior (forecast) RMSE and spread alongside the analysis values, as is standard in DA papers. The prior RMSE appears once, in prose (`sections/05_tilings_and_sizes.tex:151`).

The same applies to the comparator, whose "best" configurations are 1.4–6 times over-dispersed (`sections/06_physics_comparison.tex:119-120`). Under M1, that is a symptom of mis-tuning rather than an intrinsic property of physics filters, which is how `sections/11_discussion.tex:36-39` reads it.

### M9. The observing system is aligned with the patch lattice and is the most favourable case. **Should fix.**

Observations are every 8 grid points with 16-point patches and tiling offsets of 8 (`tables/networks.tex:2-7`). Every tiling therefore sees the same observation geometry relative to its patches (observations at patch corners and centres). The asymmetric error profile in Fig. 3a, lower at position 0 (observed) than at 15, suggests that part of the "lattice-tied" error is tied to the observation geometry, not only to the decoder.

Show at least one irregular or randomly placed network, and one network not commensurate with the patch. This matters for real observations and for the claim that the tilings' gain is a decoder effect. At one observation per site the filter's error (0.77) exceeds the observation error (0.62) (`sections/08_forecasts_cost.tex:36-46`; `tables/robustness.tex:19`). The physics comparator was not run at that density, so the crossover's dependence on density (M3c) is untested.

### M10. Novelty and positioning: missing prior work. **Should fix (to verify).**

The following directly bear on "local learned model + localized EnKF + one model at any size + learning from noisy data" and are not cited (`docs/paperII/refs.bib` has 47 entries; none of these):
- **Pathak et al. 2018, PRL**: parallel, local reservoir computers for large KS domains. This is the original "local learned model scales with domain" result; only Pathak et al. 2017 is cited, for Lyapunov spectra.
- **Penny et al. 2022, JAMES**: RNN/reservoir surrogates integrated with LETKF and 4D-Var, localized and scalable, trained from noisy data. In JAMES and very close in spirit.
- **Wikner et al. 2021, Chaos**: hybrid ML + knowledge-based model trained through DA.
- **Arcomano et al. 2022/2023**: local hybrid ML with SPEEDY.
- **Chattopadhyay et al. 2023, JCP**: deep-learning surrogate inside EnKF in high dimensions.
- **Constante-Amores, Linot & Graham 2024 (arXiv:2410.01238)**: patch-decomposed autoencoder + NODE on KS and Kolmogorov flow.
- **Latent-space DA for global weather**: Melinc & Zaplotnik 2024, QJRMS; Fan et al. 2025, arXiv:2502.02884. Relevant to the claim that latent DA "has so far used a global latent vector" (`main.tex:48`; `sections/01_introduction.tex:21-31`).
- **Shifted tilings**: weight interpolation and overlapping-domain averaging in LETKF (Yang et al. 2009, QJRMS), and the overlap-tile / test-time shift-averaging practice in CNNs, besides cycle spinning.
- **Model-error remedies (M1, M4)**: Hamill & Whitaker 2005, MWR; Anderson 2009; Miyoshi 2011; Aksoy et al. 2006 (parameter-spread inflation).

Table 1 (`sections/02_background.tex:118-140`) should include Penny et al. 2022 and Pathak et al. 2018, which would change the "Localized" and "Other sizes" columns. The claim "What we add is their combination, made exact" (`sections/02_background.tex:110-116`) should be re-examined in their light.

### M11. Relevance and framing for JAMES. **Should fix.**

All quantitative results are for a non-dimensional, statistically homogeneous, periodic, univariate 2D PDE observed densely at grid points. The motivation ("convection-permitting simulations of organized convection", `sections/01_introduction.tex:98-101`; `sections/11_discussion.tex:76-83`) is far from the evidence. For a JAMES audience the discussion needs to address the following.

- **Inhomogeneity breaks the shift trick.** Terrain, coastlines and the diurnal cycle break translation invariance. Kolmogorov flow already shows that tilings must respect the model's symmetries and fail otherwise (`sections/10_kolmogorov.tex:17-23`). What happens to weight-sharing and tilings in an inhomogeneous domain?
- **Multivariate, balanced and 3D states.** What a "site" and "channels" mean for these states, and whether per-patch encoding preserves balance.
- **Nonlocal observation operators** (M5).
- **Training data.** For a real system, the learned model would be trained on analyses produced with the wrong physics model, which inherit its errors. The comparison in Sec. 6 has no counterpart there. This deserves an explicit paragraph rather than one bullet (`sections/11_discussion.tex:64-67`).
- **Cost against numerical models.** This is usually the main argument for ML surrogates, and the paper is silent on it.
- **The claim about observation errors.** "the errors of the observations are well characterized" (`sections/01_introduction.tex:57-58`; `sections/06_physics_comparison.tex:8-9`; `sections/11_discussion.tex:40-41`) is contestable for convective-scale DA, where representativeness error dominates. Soften it.

### M12. Logical gaps and claims that outrun the evidence elsewhere. **Should fix.**

- **"Not tuned" versus selected at 44².** "none was tuned per domain size, per seed or per experiment" (`sections/05_tilings_and_sizes.tex:37-39`) sits beside an RTPS sweep at 44² used to *select* 0.7 (`sections/05_tilings_and_sizes.tex:156-158`). State plainly that the settings were selected at 44² and carried over. Also say how the Gaspari–Cohn half-width (1.5 sites) and the model noise (0.02) were chosen.
- **"One setting" versus retraining.** "the learned model needs no tuning for the error of the model it replaces: one setting served every case" (`sections/11_discussion.tex:36-39`). For the hidden and missing cases the latent model was *retrained* on each truth (`sections/06_physics_comparison.tex:97-98`). That is far more effort than tuning RTPS. In the main comparison the latent model saw the true system. The robustness comparison in Fig. 6b also shows the physics filter over RTPS 0–1, but the latent filter only over 0.5–0.9 (`tables/robustness.tex:12-15`). Show the latent filter over the same range.
- **Unsupported reason for 12 channels.** "We use 12 channels rather than the 6 of Paper I because the filter benefits from the extra capacity" (`sections/04_system.tex:77-78`) is unsupported in the paper. Give the 6-channel DA numbers.
- **"As close to the truth".** "A clean-trained model … is as close to the truth" (`sections/04_system.tex:90-91`), yet its free-rollout standard deviation is 11% high (4.06 against 3.66) and its λ₁ at 66² is 0.243 against 0.189–0.204 (`tables/model_validation.tex:5-6`).
- **Kolmogorov flow.** It supports only "the filter runs". There is no comparator, the ensemble is under-dispersed (0.52–0.58) and was not retuned, 3× is one seed, and the truth at 3× is a single run. Paper I and the authors' notes report single 40-unit Kolmogorov truths varying by ±20% (`tables/running_kolm.tex`). Comparing D_KY at 3× (−7%) against one run is therefore weak (`sections/10_kolmogorov.tex:10-12`). "The ingredients carry over to any system whose chaos is extensive" (`sections/11_discussion.tex:82-83`) overreaches.
- **The 1D SEC result is reported selectively.** At the training size L = 100, SEC on a global latent vector *beat* Gaspari–Cohn on the local field by 3–19% (`appendix/D_da_details.tex:69-73`; Fig. 8a). The main text states "A distance beats a distance-free correction" (`sections/09_one_dimension.tex:23-35`) without mentioning it. The confounds offered are reasonable, but Fig. 8a shows this result in a main-text figure the main text never discusses. Mention it in Sec. 9.
- **Steady-state windows and statistical power.** 50 steady-state cycles are 25 time units, about 4–5 Lyapunov times. At 44², 10-cycle means vary by ±6% (`appendix/D_da_details.tex:124`). Report a standard error for the steady-state means (block bootstrap over cycles, combined with seeds), and use it in the decision rule rather than the "lower on every seed" heuristic alone. Run the model-error comparison at 220², where spatial averaging is large, for more than two cases.

---

## 3. Minor concerns

**Numbers and internal consistency**
1. **Sparse-network run, three different numbers.** The spacing 5.5 run appears as 0.769 in the text (`sections/08_forecasts_cost.tex:37`), 0.798 in Table 7 (`tables/robustness.tex:19`), and as window means "0.77 / 0.77 / 0.79". The result files give 0.769 (100 cycles) and a 300-cycle "steady" value of 0.798, which is not the mean of cycles 51–300 (0.776). State the window used for the long runs in Table 7's caption (`sections/08_forecasts_cost.tex:78-83`). Likewise 0.308 against 0.31 at 220². *Should fix.*
2. **The "true equations" row is one seed repeated.** It is produced by repeating one tag three times (`scripts/paper_figures/tab07_p2_model_error.py`, CASES[0]), so "mean over seeds 0–2" in the caption does not apply to it. *Nitpick.*
3. **The 64² grid row has zero model error.** Its one-cycle error is 0.000/0.000 (`tables/model_error.tex:3`). It is a resolved model, not a wrong one, and should not count as a "physics win" case (`sections/06_physics_comparison.tex:27,47`). On the log axis of Fig. 6a it is plotted at an arbitrary x of about 3 × 10⁻³. *Should fix.*
4. **Calibration ranges differ by context.** "calibration stays between 0.92 and 0.98" across sizes (`sections/05_tilings_and_sizes.tex:194`), while the 44² seeds give 0.92–0.99 (`sections/05_tilings_and_sizes.tex:155`). Fine, but the seed range at other sizes is not shown. *Nitpick.*
5. **Exact versus single precision.** Fig. D1's caption says the two Jacobians "agree exactly" (`appendix/D_da_details.tex:49-50`); the text says "to single precision (largest difference 2 × 10⁻⁷)" (`appendix/D_da_details.tex:36-37`). *Nitpick.*
6. **Thread count.** Three threads for cost (`sections/08_forecasts_cost.tex:100`) against four threads for the Jacobian timing (`appendix/D_da_details.tex:47`). *Nitpick.*
7. **Paper I's author list differs between papers.** `refs.bib:477-481` lists "Jones, Dalton and van Leeuwen, Peter Jan"; Paper I's own `main.tex` lists Dalton Jones alone. It is "In preparation". JAMES will need the companion manuscript available to reviewers. *Should fix.*
8. **Kolmogorov 12-channel model not listed.** It is used in Sec. 10 but appears in no configuration table. Table A1 lists only the 1D and 2D KS models (`appendix/A_settings.tex:9-13`; `tables/architecture_da.tex`), and "the two models used here" is therefore wrong. *Should fix.*
9. **"Trained on noisy fields" is not universal.** The 1D model of Sec. 9 is clean-trained (`tables/architecture_da.tex:29`), while "its model never sees a clean field" (`sections/05_tilings_and_sizes.tex:9-10`) and the PLS imply this holds throughout. Make the scope explicit. *Nitpick.*
10. **Two meanings of "standardized units".** The observation error is "0.1 in standardized units (0.62 in field units)" (`sections/05_tilings_and_sizes.tex:26-27`), so the training standard deviation on the 264² domain is about 6.2. The deployed 44² field standard deviation is 3.66–3.91 (`tables/model_validation.tex:2`). So "observation-level noise of 0.1" is 0.1 of the *training* standard deviation but about 0.17 of the local field's. In Kolmogorov flow it is "0.1 of the standard deviation" (`sections/10_kolmogorov.tex:13`). State which standard deviation, and report the field standard deviation per domain size: the free-run error varies from 4.3 to 10.1 (`tables/sizes.tex`), which hints at size- and realization-dependent large-scale variance. *Should fix.*
11. **Abstract length.** The abstract is about 254 words including LaTeX tokens (`main.tex:47-65`), at or just over JAMES's 250-word limit. *Nitpick.*
12. **Placeholders remain.** Open Research and Acknowledgments still contain them (`main.tex:102-109`). The AGU data policy makes the Open Research section mandatory at submission. *Should fix before submission.*

**Notation clashes and undefined symbols**
13. **c** means channels (`sections/03_method.tex:28-32`), localization half-width ("($c=16$)", `sections/06_physics_comparison.tex:116`; `appendix/A_settings.tex:124,157`), and advection speed ("$-c\,\partial_x u$", `appendix/A_settings.tex:136`). *Should fix.*
14. **p** means patch size (`sections/03_method.tex:21`, Eq. 2) and damping order (`sections/06_physics_comparison.tex:28-31`; `appendix/A_settings.tex:144`). *Should fix.*
15. **K** means number of tilings (`sections/03_method.tex:156`), Kalman gain (`appendix/A_settings.tex:67,81`), wavenumber magnitude (`sections/04_system.tex:18`), and number of tangent vectors (`appendix/C_ground_truth.tex:9,17`). *Should fix.*
16. **h** means site length (`sections/03_method.tex:22`) and the latent observation operator (`sections/03_method.tex:95`). **ρ** means D_KY density (`sections/02_background.tex:39`; Eq. 3) and the Gaspari–Cohn taper (`sections/03_method.tex:102`; `appendix/A_settings.tex:36`). **α** means RTPS weight and Kolmogorov drag. **w** means propagator radius and blend weights. *Should fix.*
17. **U_s and N_s swap roles** between `appendix/A_settings.tex:72-76` (U_s = observations, N_s = sites) and `appendix/B_proof.tex:31-32` (O_s = observations, U_s = sites). *Should fix.*
18. **Proposition 1(b) omits s itself.** The statement says the analysis depends "only on the ensemble at the sites that own an observation within the taper support of s" (`sections/03_method.tex:129-133`); the proof correctly says U_s ∪ {s} (`appendix/B_proof.tex:45-46`). *Nitpick.*
19. **"dim" used before definition** (`sections/03_method.tex:142`; defined at `appendix/B_proof.tex:7`). λ₁ is used in Sec. 4.1 before it is defined (`sections/04_system.tex:42`). MLP (`sections/03_method.tex:52`), EnKF (Table D2 header, Fig. 8 legend), PFF (`appendix/D_da_details.tex:88`) and GC (Fig. 8 legend) are not defined in the text. *Nitpick.*
20. **Imprecise cost statement.** "Neither the cost of the colouring nor the memory of the per-site filter depends on the domain size" (`sections/03_method.tex:143-145`). The *number* of passes does not, but each pass's cost grows almost linearly (Fig. D1, slope 0.88). *Nitpick.*
21. **Paper I section numbering.** "(Paper I, Sec. 4)" (`sections/10_kolmogorov.tex:6`): Paper I numbers this section IV C. *Nitpick.*
22. **Internal decision log cited in the text.** "decisions log, D4" (`appendix/A_settings.tex:99-101`) refers to an internal document. Give the numbers or remove it. *Should fix.*

**Figures and tables**
23. **Order of first citation.** Fig. 4 (da_2d) is cited (`sections/05_tilings_and_sizes.tex:65`) before Fig. 3 (tilings, line 84), and Table 4 (sizes, line 51) before Table 3 (members, line 65). AGU requires numbering in order of first citation. *Should fix.*
24. **Fig. 6a.** Point labels overlap at x ≈ 0.1–0.4 ("0.3", "p = 8", "0.2", "a = 1.25", "p = 6", "0.1") and are unreadable at print size. The diamonds contradict the caption (M3b). Fig. 6b plots values at RTPS 0 although the caption says RTPS 0 "diverged or was worse than interpolation" (`sections/06_physics_comparison.tex:168`); say what is plotted there. Add the latent filter's own RTPS curve. *Should fix.*
25. **Fig. 4a.** The unlocalized curve leaves the frame at 66², the 88² values (3.40, 1.76) are not visible, and the 128-member unlocalized runs are absent although the text discusses them (`sections/05_tilings_and_sizes.tex:216-221`). Use a broken or log axis, or an inset. In Fig. 4c the dashed spread line is indistinguishable from the error line. *Should fix.*
26. **Fig. 5.** The caption says "errors on their own colour scale" (`sections/05_tilings_and_sizes.tex:260-261`), but the top-row error panels (d, e) appear to share the field colour bar (labelled "u/σ"). Panel text is very small. *Nitpick.*
27. **Fig. 1.** Panel (a) text is near-illegible. In (d) the x-axis "grid point (p = 16)" with ticks 0 and 25 is unclear. *Nitpick.*
28. **Fig. 8a** uses a different protocol (40 cycles, whole-run skill, a different global-vector model) from (b, c), and is not discussed in the main text (M12). "Skill" as free-run error over analysis error (`sections/09_one_dimension.tex:11`) depends on the free-run error. Prefer RMSE normalized by climatological standard deviation, or give both. *Should fix.*
29. **Table 5.** The "Tuned" column is blank for the hidden cases. The "Outcome" column mixes a ratio with the latent error. Add a column with the estimated parameter value and spread, and one with the comparator's calibration. *Should fix.*
30. **Table 2.** The model's D_KY is "from one initial condition (seed 0)", while the truth spans three runs (`sections/04_system.tex:97-105`). Paper I reports means over initial conditions. A multi-IC model mean would be more comparable. *Nitpick.*
31. **Unsupported "×" factors.** "four times better" (`sections/06_physics_comparison.tex:12-13,111-112,131`; `sections/11_discussion.tex:23-24`). For δ = 0.3 the ratio of the physics filter with the true model to the wrong-model physics filter is 5.1, so "about four times better than either" is loose. *Nitpick.*
32. **Overstated opening.** "in high-dimensional systems they work only because of localization" (`sections/01_introduction.tex:6-10`) overstates: EnKFs work unlocalized when N exceeds the unstable dimension, as the paper itself shows. *Nitpick.*

---

## 4. The five changes that would most improve the chance of acceptance (ordered by impact, then effort)

1. **Redo the physics-based comparison properly (M1, M2).** Retune inflation and localization with each remedy switched on, including smaller localization radii. Fix the parameter-spread handling and show that a = 1.25 and the advection speed are recovered. Report estimated parameters and calibration. Run the comparator with 256 members, or compare against latent 4 × 16. This is moderate effort with existing scripts, and it is decisive for the third key point.
2. **Reframe or generalize the crossover (M3).** Plot the error of the model *as run* (after estimation) and add the learned model's own one-cycle error to the axis. Include the hidden-field and pre-registered cubic cases consistently. Either repeat the comparison at a second observation density and cycle length, or restate "a sixth of the observation error" as specific to this configuration. Rewrite the third key point, the abstract (`main.tex:60-63`) and `sections/12_conclusions.tex:26-34` accordingly.
3. **Tone down "exact" and "trained only on noisy data" (M5, M6)** in the title, key points, abstract, PLS and conclusions. Explain that exactness means "no truncation of a local observation operator", that it holds only for local observations, and that the training data are complete fields with weak, easily separable noise. Low effort, high impact on reviewer trust.
4. **Add the missing baselines and prior work (M4, M10).** At minimum, discuss physics-plus-learned-correction hybrids and cite Pathak et al. 2018, Penny et al. 2022 (JAMES), Wikner et al. 2021 and the hybrid SPEEDY work, and revise Table 1. Ideally, add one hybrid comparator and an R_dec = 1, one-tiling 2D model (M5) to show the tilings are not just repairing the per-patch design.
5. **Strengthen the DA diagnostics (M8, M9, M12).** Use a principled blended spread; give rank histograms and spread–skill by scale; report prior RMSE; give standard errors of the steady-state means; test one irregular, non-aligned observation network; and give the latent filter's robustness over the same RTPS range as the comparator. Also fix the minor inconsistencies above, notably the sparse-run numbers, the notation clashes, the missing Kolmogorov model configuration, figure order and the placeholders.
