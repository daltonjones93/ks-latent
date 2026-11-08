# CM1 programme review: audit, literature position, and a research plan

*Written 2026-11-07 (Fable 5.1 review). Sources: `docs/cm1_plan.md`, `docs/cm1_notes.md`, `docs/cm1_status/cm1_status.tex`, the Paper I/II abstracts, and the code in `ks_latent/cm1`. Every number below is copied from the notes or plan; where a result is still running, it is marked **pending**. Literature citations in §4–5 were gathered by five parallel search agents on 2026-11-07 and spot-checked; anything marked *(unverified)* must be checked against the primary source before it appears in a paper.*

---

## 0. One-paragraph summary

In three days of cluster work (2026-11-04 → 07) the project produced: (i) a **121-storm CM1 2D squall-line dataset** at 1-min output across three ensembles, plus four CM1 source patches (per-member noise seeds, two-thermal initiation, a tiny-perturbation twin field, 10-s output reruns); (ii) a **local latent surrogate** (column autoencoder, 108× compression, 32 channels per 2 km site; stencil propagator, 1-min step) whose deterministic forecasts keep skill 0.35 of persistence from 30 min to 2 h on unseen storms, and a stochastic variant with calibrated spread; (iii) a **localized latent LETKF** that assimilates surface θ′, learned reflectivity and Doppler radial velocity with no access to CM1's equations: −42% cycled error on 22/22 unseen storms, better than every equation-free baseline tried, unchanged in a held-out environment and in an 800 km two-storm domain without retraining; (iv) a **predictability and dynamics audit** (CM1 twin-run doubling times of 3–8 min; the surrogate's finite-time Lyapunov spectrum; dispersion relation; travelling-wave solutions) showing the MSE-trained surrogate is far less chaotic than CM1 and why; (v) **interpretation results** (latent regimes, 30-min rain-onset prediction, nested-dropout-ordered channels, an explicit quadratic latent PDE that forecasts as well as the neural propagator but is dense); and (vi) a **pre-registered claims protocol** (G0) with six claims, five met and one half met. Paper III (equation-free convective-scale latent DA) is publishable now modulo the open items in §3. Two further papers are within reach (§5): the emulator/predictability story and the interpretable latent-law story. The most promising new directions (§6) are sparse-observation DA where realistic error growth should start to matter, the nonlinear (particle-flow) latent filter for the reflectivity operator, the sparsified law with ordered channels, and the move to 3D.

---

## 1. What was done: an audit by activity

### 1.1 Simulation (CM1 r21.1, Cashew CPU cluster)

| Item | What | Status / numbers | Why it matters |
|---|---|---|---|
| Base case | 2D x–z squall line, 400 km × 20 km, Δx 250 m, 62 stretched levels, Morrison 2-moment, no Coriolis/surface fluxes, open lateral boundaries, grid moving at 12.5 m/s, 6 h | from the original 10 members (5-min output) | The nature run for all twin experiments |
| Output-interval test | 1-min rerun of member 1 | successive-increment correlation +0.37 at 1 min, −0.53 at 5 min; column states cycle in ~10 min | 5-min output under-samples the convective cycle; **1-min output adopted** |
| Build validation (B1) | rebuilt CM1 vs supplied runs | bitwise identical at t = 0; differences 50× below member spread, reach member spread by ~2 h | A rebuilt run is a new member, not a bit-copy |
| Patch 1: `seed_var1` | per-member random-noise seed | all 121 new members have independent noise; original 10 shared one noise field | Independent truths for DA; honest learning curves |
| `ens_A` | 31 members, control sounding, σθ 0.1 K perturbations below 500 m | 64 GB GrADS; 15 GB float16 cache | Training core |
| `ens_A2` | 30 members, control sounding, new seeds | | Independent truths; learning-curve extension |
| `ens_B` | 60 members, BL moisture offset −2.0…+0.25 g/kg × shear factor 0.6–1.4, after two 4-member pilots | +1.0 g/kg is not a squall line (domain-wide convection) | Environment variety; held-out-corner test |
| Patch 2: `two_storms_var2` | two identical thermals var2 apart | `ens_T2` 8 members (nx 3200, 800 km); `ens_T2b` 8 more **(running, Cashew 750365–6)** | Real two-storm truths for transfer (T2) |
| Patch 3: `tiny_pert_var3` | hash-based ±var3 K θ field | 8 pairs at 1e-3 K, 2 at 1e-4 K, 4 h (`ens_L2_*`) | CM1 error-growth ground truth (L2) |
| 10-s reruns | 2 storms, 3 h | every 6th frame bit-identical to 1-min output | Time resolution of tendency fits (FP3) |
| Cost | ~90 min/member/6 h on one core; 20-job cap handled by packing | | Cost table for the paper (G9, not yet written) |

**Assessment.** The simulation side is careful: bitwise checks on every patch, pilots before ensembles, supersaturation guards, provenance in manifests. The one structural limitation is that everything is 2D, one storm archetype, one microphysics scheme, and one initiation mechanism (warm bubble). The data-only analyses (reach 8–12 km per 5 min; learning-curve saturation at ~28 storms) are unusual and valuable: they let design choices (stencil reach, localization radius, ensemble count) be *measured* rather than tuned.

### 1.2 Representation (the autoencoder)

| Experiment | Result | Reading |
|---|---|---|
| Column AE, 8-cell patches → 32 channels (`full_v1`) | recon nMSE 0.022 all / 0.071 storm; divergence ratio 0.236 vs truth 0.224 | 108× compression keeps anelastic mass balance |
| 16 vs 32 channels (D4) | 32 better on recon and 45–60 min | adopted |
| Joint vs per-variable encoder (D7) | joint: w RMSE 0.42 vs 0.83 m/s, 5-min nMSE 0.092 vs 0.118 | variables must be encoded together (capacity confound noted) |
| 64 channels / 1 km sites (SM1) | recon −36% / −31%, forecasts unchanged at 30–60 min, **DA worse (+0.023, 21/22 storms)** | representation is not the forecast bottleneck; more latent dims hurt a 32-member filter |
| PCA-32 per site (I-PCA) | recon 2.7× worse; forecasts 44/15/16% worse; divergence ratio 0.61; **DA +0.100 (22/22), worse than surrogate-EnOI and obs-only** | the nonlinear AE is load-bearing; the "latent ≈ PCA" probe finding was a linear-readout artefact |
| 2D field AEs with z as a coordinate (FP1/FP1b) | at the column AE's size, 2× worse; low-compression field models forecast well to 5 min then diverge | flattening the column into channels is the better compression |
| Nested dropout (FPc arm 3) | first channel alone: recon 0.30 (vs 1.10 unordered); 8 channels 0.086; skill unchanged | an importance-ordered latent at no cost; 5 trailing channels unused |
| SM0 band analysis | recon keeps 70–100% of 4–16 km variance and the extremes; < 4 km band partly lost (θ′ 0.27–0.31) | the representation floor sits below 4 km |

**Assessment.** The autoencoder work is thorough and the negative results (PCA, bigger AE, field AE) are as informative as the positive ones. Two points deserve more thought than they have had: (a) *why* a richer latent hurts DA — the natural hypothesis is sampling error in 64 dims per site with 32 members, which predicts the penalty should vanish with N = 128 and could be tested in an afternoon (I1); (b) nested dropout's success suggests the *effective* local dimension is well below 32, consistent with D_KY ≈ 2–12 for the whole domain (L1) — the ordered latent is the right tool to measure it.

### 1.3 Dynamics (the propagator) and training

| Experiment | Result |
|---|---|
| Stencil CNN, reach 3 sites/min, zero-init residual (identity at init) | baseline |
| Rollout length k ≤ 45 vs 20 (Stage B) | 60-min nMSE 0.319 → 0.249; 1–5 min slightly worse |
| Uniform-k sampling (D3) | recovers short leads: 0.040/0.092 at 1/5 min |
| Width 512 (D4) | worse beyond 15 min → data-limited |
| Learning curve (D5) | 7 → 14 → 28 storms: 0.271/0.230/0.196 at 30 min; 28 → 58: ≤ 4% |
| **Stage C** (joint fine-tune on decoded 45-step rollouts, D12) | **0.214/0.332/0.351 → 0.187/0.316/0.310**; cold-pool bias −17 → −14%; rain FSS 0.70 → 0.77 (ens_A); same pattern on ens_B and the pooled run |
| Averaging 4 propagators (D14a) | 5–10% at all leads; incompatible with Stage C's per-model latent |
| 2-frame history (D13) | best long-lead fronts; nothing once averaged; dropped |
| Environment conditioning (D11), mirrors (D9) | neutral / hurts |
| Attention propagator (M1a/M1c) | +12% at 5 min at 256 wide (half of it capacity), equal at 30–60; no DA gain; transfers slightly worse |
| Stochastic propagator, noise channels + CRPS Stage C | CRPS 30% lower, spread/error 0.93, cold-pool bias halved, mean error 7–12% worse |
| SM2a/b/c, SM3 (energy score, 8 noise channels, 30k steps, spectral loss) | sharper small scales, same mean-accuracy gap; spectral loss worse |
| Flow-matching propagator (SM4 → SM4b) | one-step-trained pilot worse; with rollout Stage C: CRPS within 5%, 2–2.5× sharper < 4 km, overshoots mesoscale condensate at 60 min; SM4c's 8 fixes all fail the pre-registered rule |
| Pooled 93-storm model (`full_v1_stagec`) | 0.213/0.365/0.344 at 5/30/60 min on 22 test storms (true boundaries); held boundaries cost 1–2% |
| Leads to 2 h (H2) | skill 0.357/0.361/0.352/0.350 at 30/60/90/120 min; error at 2 h < persistence at 5 min |
| Advection extrapolation (H1a) | worse than persistence (1.2) in the storm-following frame |

**Assessment.** The headline emulator result is solid and unusually well controlled (held boundaries, 22 storms × 8 starts, true-vs-held check). The deepest finding is the SM0/L1/L2 triangle: *the reconstruction is sharp, the deterministic forecast is smooth, the surrogate is far less chaotic than CM1, and no objective tweak closes the stochastic model's accuracy gap.* This is the conditional-mean problem of MSE training, documented here more completely than I have seen elsewhere for a convective system (spectra by band, extremes, Lyapunov spectrum, twin-run growth, all on one model). It is also the clearest open problem: the flow-matching model shows sharpness *can* be bought at equal CRPS, but nothing yet reproduces CM1's growth from small differences (SM5: every stochastic model jumps to saturation-level spread at step 1).

Two methodological cautions for the paper: single seeds in most model screens (the conv seed spread was measured once, ±0.003, which is reassuring); and nMSE across runs with different training sets is not comparable because the normalizer is refit (noted in D5, correctly handled by reporting skill vs persistence).

### 1.4 Dynamics audit: Lyapunov, spreading, dispersion, travelling waves, twin runs

| Measurement | Result |
|---|---|
| L1, finite-time Lyapunov spectrum of the surrogate (6208 dims, 6 storms) | along its own run: 1 ± 1 positive, λ1 ≈ 0.001/min (doubling ~13 h), D_KY ≈ 2; along the encoded CM1 trajectory: 4–5 positive, λ1 0.005–0.010/min (doubling 1–2 h), D_KY 9–12 |
| L5, perturbation spreading | 3.2–3.3 km/min, same in storm and far field (network coupling, not convection); data-only reach 1.6–2.4 km/min; network max 6 |
| L2, CM1 twin runs (1e-3 K) | growth 0.13 / 0.21 / 0.08 per min at 5–15 / 15–30 / 30–60 min (doubling 5.4 / 3.4 / 8.4 min); 10³× in the first hour; saturation at independent-storm difference by 2–3 h; 1e-4 K pairs agree |
| L2c, rough intrinsic floor | half the twin nMSE ≈ 0.22 / 0.35 at 2 / 3 h of storm life vs forecasts 0.27 / 0.36 — **but twins start at initiation; L2b (perturb at 60-min restart) needed before claiming "at the floor"** |
| SM5 | deterministic surrogate does not amplify a tiny latent perturbation at all over 3 h; stochastic samples start at 0.19 and follow the saturation curve from ~30 min |
| Q1, dispersion of the quiescent state | short waves damped (−0.064/min at 4–8 km), long waves weakly unstable (+0.02/min > 32 km), no fast gravity-wave branch (|c| ≤ 0.46 vs NH/π = 2.46 km/min); storm-forced far-field waves 15–35% too slow |
| Q2, travelling waves | propagator has a near-exact travelling-wave solution next to every mature state (residual 0.10); speed correlation with CM1 0.61; decoded C/Δu = 1.3–3.6, uncorrelated with shear |
| Q3, regeneration period | none at the domain level in model or CM1 |
| FPc dispersion (explicit law) | damped at 4–8 km, weakly growing at 32–128 km, phase speeds to 2.5–3.2 km/min at 128 km |

**Assessment.** This is the part of the programme most likely to interest dynamicists and the least likely to have been done before on a convective emulator (see §4). The gravity-wave finding (Q1) is a genuine counter-example to "MSE gets the organized dynamics right" and deserves a dedicated figure. The honest caveat about L2c (initiation-phase twins) is right; L2b is cheap (~27 CPU-h) and would turn a suggestive comparison into a quotable claim.

### 1.5 Data assimilation

**Method.** Per-site latent LETKF (Hunt et al. 2007 form, ensemble-space, R-localization with Gaspari–Cohn on the site lattice, plain distances for open boundaries), RTPS, additive latent noise; observation operators on decoded columns: surface θ′ (linear), learned composite-reflectivity head (1.6 dBZ RMSE on decoded columns), radial velocity (4/3-Earth beams, 3 tilts). Tested against the Kalman update to 1e-10, locality to 2c, RTPS spread restoration.

**Results (G0 setup: 22 unseen storms × 3 seeds, 32 random members, 5-min cycles over 3 h, held boundaries, settings tuned on 6 validation storms):**

| Claim | Threshold | Result | Status |
|---|---|---|---|
| 1 DA works | ≥ 10% below no-DA, CI excl. 0, ≥ 80% storms, front < 1 km | 0.273 ± 0.022 vs 0.469 ± 0.061 (−42%), 22/22, front 0.38 km | met |
| 2 Calibrated | spread/error 0.7–1.3; rank histogram not U (χ², p > 0.01) | 0.71; perturbed-member histogram mildly U (outer bins 1.11–1.34× flat), χ² rejects | **half met** |
| 3 Beats static / obs-only | ≥ 10% below each | −23% surrogate EnOI, −53% physical EnOI (worse than no DA), −16% obs-only (analysis) | met |
| 4 Generalizes | gain in held-out env ≥ half in-dist; 30-min skill within 25% | gain 59% vs 48%; skill within 3% | met |
| 5 Forecasts | skill < 0.5 at 30, < 1 at 60, beats advection | 0.357 / 0.361; advection 1.21 / 1.25 | met |
| 6 Transfer | localized 2-storm/1-storm ≤ 1.1 at every N; unlocalized larger at N = 8 | 0.99 / 1.00 (N 8/32); unlocalized 1.08 / 1.04 | met (8 storms, 1 seed; 16 storms × 3 seeds **pending**) |

**Other DA findings.**
- Observation types are complementary: θ′ pins the cold pool and gust front; reflectivity the condensate aloft; reflectivity alone *degrades* θ′ (0.73 → 0.96 K) through spurious correlations; radial velocity is the single most valuable addition (−30% vs −18%); w is barely constrained by anything (1.10 → 1.00 m/s).
- Spread collapses without additive noise; noise 0.1 (latent units) is the validation optimum and gives 0.71 spread/error; the stochastic model gives 1.04 with RTPS 0.5 at a 7% accuracy cost; 4 propagators roughly double the deterministic spread but do not replace inflation.
- Latent vs physical-space analysis on the same ensemble: 0.236 vs 0.245 (latent lower on 9/12, within noise).
- Shifted tilings −4% at 4× cost; the control shows −0.8% is seams, the rest is averaging (→ ensemble size is the cheap route).
- True vs held lateral boundaries: 10% in cycled DA (1–2% in forecasts).
- G3: forecasts from analyses beat no-DA and persistence-of-analysis at every lead; FSS(≥ 35 dBZ) 0.93 vs 0.85 / 0.84 at 50 min; front 3 km vs 7–8 km.
- G2b (**pending**): pre-registered persistence-ensemble and drawn-analog baselines came out catastrophically worse than no DA (2.2–14 nMSE) on validation; a disclosed post-hoc extension (zero/low noise persistence; mean-of-10 analog) is queued (Saidie 9061–9072).
- Attention propagator and 64-channel AE give no DA gain / a loss; PCA latent far worse.

**Assessment.** The DA core is the strongest part of the programme: large effect sizes, every storm, pre-registered thresholds, validation-tuned settings, a reproduced headline (0.274 on rerun), and the right set of baselines. Three things a referee will press on, in order: (i) **calibration**: 0.71 and a mild U are "slightly underdispersive", not "calibrated"; the paper should say so, show the histogram, and either report the stochastic model (1.04) as the calibrated option or add adaptive inflation (G4). (ii) **Baselines**: the G2b failure mode is itself informative (a model-free ensemble random-walks off the manifold), but the paper needs the extended grid to show the baselines were given a fair chance. (iii) **The twin setting**: perfect model, dense synthetic observations, 2D. The held-out-environment result partly answers "perfect model"; nothing yet answers "dense observations" (G8/G8b).

### 1.6 Interpretation

| Study | Result |
|---|---|
| P1 probes | latent carries the same information as PCA-32 of the columns (linear readout) and the gap closes to ≤ 0.01 with MLP probes; rain vs cloud water separable (AUC 0.95) though inputs lump them |
| P2 regimes | 8 GMM regimes with clear physics (convective core 3.5% of sites, heavy trailing rain, light rain, dry cold pool, anvil, 3 quiet), each persisting 0.88–0.93 over 5 min |
| P3 event prediction (hard version: quiet sites near an active storm) | rain onset AUC 0.977 / 0.952 at 5 / 30 min; neighbourhood context worth 0.08–0.13 AUC; latent motion adds up to 0.03; the surrogate's own forecast is competitive; transfers to two-storm runs (−0.01 to −0.05) |
| P4 superposed epoch | onset probability separates from matched controls already at −30 min (0.08 vs 0.03–0.06), rising to 0.79 at −5 min |
| P5 traversal | the onset direction is a deepening cloud with ice at 8–10 km, not low-level drizzle |
| P7 storm-scale | pooled local latent as good as physical baselines for 30-min-ahead totals (no unified latent needed) |
| FP0/FP0b, laws on hand-built fields | Dη/Dt = −0.80 ∂B/∂x + 28.5 ∇²η + 1.07 η w dlnρ/dz (known −1, +1); fall speed 5.7 m/s; condensation 0.77 (known 1) with a Lagrangian regressor; all four pre-registered checks pass |
| FPc, explicit latent PDE | skill 0.377/0.362 (within 10% of the neural propagator), residual share 1–3%, **dense** (30–47k of 52.7k terms > 1e-3), no pruning knee; nested dropout orders channels at no cost; a domain-mean input G⟨z⟩ is as large in norm as A0 (0.90) but buys only 1.5–2% skill: locality stands |
| FPd, SINDy-AE-style sparsification during training | **pending** (Saidie 9032–9038) |

**Assessment.** The P-study is well designed (baselines, the trivial-far-field trap caught and fixed, gauge caveats stated). Its most quotable result is the 30-min rain-onset lead with a local latent neighbourhood. FP0 is a clean known-answer validation that makes the later law-finding credible. FPc's "accurate but dense" is the expected outcome for a chaotic system with a generic quadratic library; FPd is the real test.

### 1.7 Methodology

Strengths: pre-registration (G0 + three addenda) before headline runs; validation-storm tuning; paired statistics with CIs; every negative result recorded; explicit disclosure of the one post-hoc change (addendum 3b); bitwise provenance on the simulation side; unit tests against analytic answers for every numerical component. Weaknesses: single seeds in most model screens; the predictability-floor comparison (L2c) is not yet apples-to-apples; T2 is unpaired with 8 storms; the learned-vs-hand-set noise comparison was run with RTPS tuned for the deterministic model (the follow-up fixed this); CRPS-vs-MAE comparisons favour ensembles by construction (noted).

---

## 2. Key numbers in one place

| Quantity | Value |
|---|---|
| Storms simulated / used for training / test | 121 (+16 two-storm, +20 twin) / 93 / 22 |
| Latent | 200 sites × 32 ch = 6400 (108× compression of 434 × 1600) |
| Forecast skill vs persistence, 5 / 30 / 60 / 120 min | 0.22 / 0.36 / 0.36 / 0.35 |
| Front displacement, 30 / 60 min | 0.6 / 1.4 km (persistence 6 / 11.5) |
| Cold-pool intensity bias at 60 min | −14 to −20% (deterministic), −10% (stochastic) |
| CRPS skill, stochastic model | 0.34–0.39 to 2 h; spread/error 0.86–0.93 |
| CM1 error doubling (twin runs) | 3.4–8.4 min; saturation 2–3 h |
| Surrogate λ1 (along truth) | 0.005–0.010 /min; D_KY 9–12 |
| Cycled DA nMSE, headline / no DA | 0.273 / 0.469 |
| Spread/error; rank outer-bin ratio | 0.71; 1.11–1.34 |
| vs surrogate EnOI / physical EnOI / obs-only | −23% / −53% / −16% |
| Held-out environment DA gain / in-distribution | −59% / −48% |
| Two-storm / single error ratio, localized / not (N = 8) | 0.99 / 1.08 |
| Latent-law residual share; dense terms | 1–3%; 30–47k of 52.7k |

---

## 3. What is solid, what is fragile, what is missing

**Solid (would survive a hostile referee):**
1. The emulator's skill, locality and sample-efficiency results (held boundaries, many starts, pooled vs specialized, learning curve).
2. The DA headline, claim 3 baselines, and the held-out-environment result, all pre-registered and validation-tuned.
3. The localized transfer to real two-storm CM1 runs (claim 6), once the 16-storm × 3-seed extension lands.
4. The SM0 diagnosis (smoothing is the objective, not the representation) and the PCA/64-channel negative results.
5. The CM1 twin-run growth rates (a dataset result independent of any ML).

**Fragile or incomplete:**
1. **Calibration** (claim 2 half met). Fix: report the stochastic model as the calibrated configuration and/or run adaptive inflation (G4); show the histogram.
2. **"Forecasts near the predictability floor."** Needs L2b (perturb at a 60-min restart). Until then, state it as suggestive only.
3. **Baselines for forecast models (G2b).** Pending; the disclosed grid extension is the right move.
4. **Dense observations only.** No sparse-observation or cycle-length sensitivity (G8/G8b). This is also where the deterministic-vs-stochastic question would be decided.
5. **Boundaries.** A 10% boundary effect in cycled DA with held boundaries is a 2D/open-boundary artefact; I4 (self-forecast edges) would quantify how much is recoverable.
6. **Cold-pool bias.** −15–20% is a physically meaningful systematic error (RKW balance, propagation). It is reported but not fixed; Stage C and the stochastic model reduce it.
7. **Single microphysics / archetype / 2D.** Only more simulation fixes this.

**Missing from Paper III as outlined:** the cost table (G9, cheap), a non-local baseline (S1/H1b: what locality buys), the literature review (G12), the ensemble-size curve (G6), and the localization-vs-reach sweep (G7). S1 is the one I would not submit without: "local is better" is currently an argument supported indirectly (transfer, sample efficiency), not a head-to-head result.

---

## 4. Literature position

Five parallel searches (latent DA; convective emulators and sharpness; storm-scale radar DA; interpretable latent laws; predictability, extensivity and equation-free baselines) returned ~150 items; the ones that bear on our claims are below. The agents worked from abstracts and bibliographic records, not full texts. Items they could not confirm are marked *(unverified)*; everything must be checked against the primary source before it is cited (G12/H5 remain open tasks). Where an agent's report contradicted the brief I gave it (wrong author lists, wrong arXiv ids, wrong venues), I have used the agent's corrected version.

### 4.1 Latent-space and learned-surrogate data assimilation

**What exists.** Autoencoder-latent ensemble filters are now a well-populated line: Amendola et al. 2020 (arXiv:2012.12056, convolutional AE + LSTM + OI on indoor CO₂); Peyron et al. 2021 (QJRMS 147:3759, ETKF-Q in an AE latent with a neural surrogate, augmented Lorenz-96); Chen, Sanz-Alonso & Willett 2023, ROAD-EnKF (Inverse Problems 39(12), arXiv:2301.11961: learned latent surrogate + decoder fitted through an EnKF likelihood, tested on Lorenz-63 embeddings, Burgers and KS); Pasmans et al. 2025 (QJRMS, arXiv:2502.12987, ETKF in a VAE latent with a second latent for innovations); Fan et al. 2025/26 (arXiv:2502.02884, Science Advances; and HLOBA, arXiv:2603.04395: global atmospheric latent DA at ~3% of EnVar cost, element-wise latent uncertainty); Tong, Wang & Yan 2026, LAE-EnKF (arXiv:2603.06752, linear latent dynamics with generalization bounds). Latent variational DA: Mack et al. 2020 (CMAME 372), Melinc & Zaplotnik 2024 (QJRMS 150:2273) and 2025 (arXiv:2506.11968, a neural B that is near-diagonal in latent space). Generative/score-based DA: Rozet & Louppe 2023 (NeurIPS), Appa (Andry et al. 2025, arXiv:2504.18720, latent diffusion at 0.25°), SLAMS (Qu et al. 2024), Latent-EnSF (Si & Chen, ICLR 2025, arXiv:2409.00127), Manshausen et al. 2025 (JAMES 17, km-scale station DA with diffusion), DiffDA (Huang et al. 2024), FuXi-DA (Xu et al. 2024/25), and end-to-end Aardvark Weather (Allen et al. 2025, Nature 641:1172). Data-driven models inside classical filters: Hamilton, Berry & Sauer 2016 (PRX 6:011021, "Ensemble Kalman filtering without a model", Kalman–Takens, Lorenz-63), Lguensat et al. 2017 (MWR 145:4093, analog DA), Brajard et al. 2020 (J. Comput. Sci. 44:101171) and Bocquet et al. 2020 (joint model/DA learning), Penny et al. 2022 (JAMES 14, e2021MS002843: reservoir computing + DA scaled by *physical-space* domain localization), the Arcomano/Szunyogh hybrid-SPEEDY + LETKF line (arXiv:2509.22465, *authors unverified*), and "Learning Enhanced Ensemble Filters" (Bach et al., JCP 547:114550, 2026, arXiv:2504.17836: a learned set-transformer filter that beats an optimized LETKF on L96/KS/L63). Localization theory: Anderson 2012 (MWR 140:2359, sampling-error correction, distance-free), Wang et al. 2023 (JAMES 15(10), CNN-learned localization), Al-Ghattas & Sanz-Alonso 2024 (Information & Inference, arXiv:2208.03246: small ensembles suffice when the prior covariance has low *effective dimension*), Majda & Tong 2016 (arXiv:1606.09321).

**Where we sit.** Every latent-EnKF paper above uses a *global* latent vector (or a flattened convolutional latent) and either no localization or a diagonal/learned-B surrogate for it. Penny et al. localize in physical space with a reservoir model. Nobody we found localizes a learned latent *field* with physical site coordinates and Gaspari–Cohn, nor shows that such a system transfers to a larger domain without retraining; LAE-EnKF reportedly lists localization as future work (*not independently confirmed*). The "latent analysis ≈ physical analysis" comparison at identical ensembles, the PCA-vs-AE DA comparison, the 32-vs-64-channel degradation, and the deterministic+noise vs CRPS-stochastic comparison have no precedent we found. The 64-channel result has a natural theoretical home in Al-Ghattas & Sanz-Alonso's effective-dimension bound and Majda & Tong's filtering dimension, which no latent-DA paper has used in this way.

**Verdict.** The *method family* (AE latent + ensemble filter + learned surrogate) is established and must be cited as such (Peyron, ROAD-EnKF, Pasmans, Fan). Our novelty is specific and defensible: (i) localization on a learned latent lattice with physical coordinates; (ii) training-free domain transfer of the whole DA system; (iii) convective-scale, radar-like observations through learned operators; (iv) the controlled ablations. Hamilton, Berry & Sauer 2016 is the right ancestor for the phrase "equation-free EnKF" and must appear in the introduction.

### 4.2 ML emulation of convection and the sharpness problem

**What exists.** Km-scale emulators: StormCast (Pathak et al. 2024, arXiv:2408.10958; Science Advances 2026: diffusion emulation of the 3 km HRRR at 1-h steps, realistic cold pools and updrafts, no DA), HiRO-ACE (Perkins et al. 2025, arXiv:2512.18224, stochastic coarse emulator + diffusion downscaler trained on a 3 km GSRM), STRATA (*2026 NeurIPS poster, unverified*: local 3D neighbourhood attention on 4.9 km SCREAM output). Column-wise CRM-trained parameterizations: Rasp, Pritchard & Gentine 2018 (PNAS), Brenowitz & Bretherton 2018/2019, Yuval & O'Gorman 2020. Local learned simulators and their generalization: Bar-Sinai et al. 2019 (PNAS 116:15344), Kochkov et al. 2021 (PNAS), Stachenfeld et al. 2022 (ICLR), Sanchez-Gonzalez et al. 2020, NeuralGCM (Kochkov et al. 2024, Nature 632:1060). Blurring and its fixes: Subich et al. 2025 (ICML, arXiv:2501.19374, modified spherical-harmonic loss raising GraphCast's effective resolution ~8×), FastNet (arXiv:2509.17601), GenCast (Price et al. 2025, Nature), AIFS-CRPS (Lang et al. 2024, arXiv:2412.15832, fair-CRPS ensemble with noise), FourCastNet 3 (arXiv:2507.12144, CRPS + spectral CRPS), ArchesWeatherGen (Couairon et al. 2024, flow matching on residuals of a deterministic model), CorrDiff (Mardani et al. 2025, Commun. Earth Environ. 6:124: UNet mean + diffusion residual). ML models are too smooth and too non-chaotic: Bonavita 2024 (GRL), Selz & Craig 2023 (GRL 50(20): Pangu does not reproduce the butterfly effect) and 2026 (JGR-MLC, DOI 10.1029/2025JH001180: some models show early growth but through numerical noise), Chen et al. 2026 (arXiv:2609.18489: four probabilistic models misrepresent upscale energy transfer; spatially uncorrelated perturbations accumulate excess small-scale energy). Rollout training: Brandstetter et al. 2022 (ICLR, pushforward trick), PDE-Refiner (Lippe et al. 2023, NeurIPS). Latent interpretation of convection: Behrens et al. 2022 (JAMES, VED separating convective regimes), Mooers et al. 2020/2023, Mangipudi et al. 2021. Nowcasting baselines: pysteps (Pulkkinen et al. 2019, GMD), RainNet (Ayzel et al. 2020), DGMR (Ravuri et al. 2021, Nature), NowcastNet (Zhang et al. 2023, Nature).

**Where we sit.** The agent found *no* published full-state neural emulator of an idealized CRM squall line or supercell (a search negative, not proof). Our Stage C result (joint fine-tuning on decoded 45-step rollouts as the largest single gain) is a fuller version of the pushforward trick and is not reported in this form. The comparison of MSE, CRPS+noise-channels, energy score, spectral loss and conditional flow matching with rollout fine-tuning in one storm-scale setting, with the finding that none closes the ensemble-mean gap, has no counterpart; Subich et al. and FCN3 report spectral fixes working at global scale, which is the contrast to state. Our finite-time Lyapunov spectrum of a storm-scale emulator vs CM1 twin runs confirms Selz & Craig at a scale they did not test, and is the first full-spectrum (6208-dim) computation for a weather-type emulator that the agent could find. The cold-pool bias and the gravity-wave phase-speed deficit are concrete numbers StormCast reports only qualitatively. Behrens/Mooers cluster regimes; none reports a forward-looking probe (our 30-min rain-onset AUC 0.95).

**Verdict.** Cite StormCast as the nearest emulator and the Subich/FCN3/CorrDiff/ArchesWeatherGen line as the sharpness prior art; frame our smoothing result as "confirms Bonavita / Selz & Craig / Chen et al. at storm scale, with the mechanism isolated (objective, not representation)". Two of the agents' suggestions should be adopted before publication: check that the surrogate's positive exponents survive a float64 / CPU re-run (Selz & Craig 2026's numerical-noise caveat), and add a kinetic-energy-transfer diagnostic to the SM5 comparison (Chen et al. 2026).

### 4.3 Storm-scale radar DA

**What exists.** Snyder & Zhang 2003 (MWR 131:1663, first storm-scale EnKF OSSE: supercell recovered after ~30 min of radial-velocity scans, inflation enhanced spurious cells); Tong & Xue 2005 (MWR 133:1789, reflectivity as a highly nonlinear operator; updates to some fields withheld in early cycles); Dowell et al. 2011 (MWR 139:272: reflectivity speeds development but gives low-confidence cold pools); Dowell & Wicker 2009 (JTECH 26:911, additive noise in precipitating regions); Sobash & Wicker 2015 (MWR 143:3067: repeated coherent additive noise biases the cold pool and tropopause; an adaptive version gated by reflectivity innovations matches the best hand-tuned noise); Aksoy, Dowell & Snyder 2009/2010 (MWR 137:1805; 138:1273: prior fits 3–6 m/s and 7–10 dBZ; "no-precipitation" reflectivity suppresses spurious cells; 30-min forecasts from analyses lose skill in tens of minutes); Sobash & Stensrud 2013 (MWR 141:3691: horizontal localization cut-offs 6/12/18 km, larger horizontal + smaller vertical best); Whitaker & Hamill 2012 (MWR 140:3078, RTPS); Anderson 2009 (Tellus A 61:72, adaptive inflation); Wheatley et al. 2015 and Jones et al. 2016 (Wea. Forecasting 30:1795; 31:297, Warn-on-Forecast radar + satellite); Torn, Hakim & Snyder 2006 (MWR 134:2490, lateral boundaries in limited-area EnKF); Trevisan & Uboldi 2004 (JAS 61:103) and Bocquet & Carrassi 2017 (Tellus A 69, assimilation in the unstable subspace). ML in radar DA: Frerix et al. 2021 (ICML, learned *inverse* observation operator), Stefanelli, Zaplotnik & Skok 2025 (arXiv:2512.18289: a CNN reflectivity operator in ALADIN 3D-Var, 5.99 → 3.47 dBZ), Geer 2021 (Phil. Trans. A, ML vs DA framing).

**Where we sit.** Several of our DA findings are *rediscoveries* and should be presented as consistency checks, not results: reflectivity degrading low-level thermodynamics through spurious correlations (Dowell et al. 2011; Tong & Xue 2005), radial velocity constraining wind while reflectivity constrains hydrometeors (Snyder & Zhang; Aksoy), spread collapse cured by additive noise (Dowell & Wicker), RTPS, and the boundary effect (Torn et al.). Our 5 km localization is tighter than Sobash & Stensrud's optimum (12–18 km horizontal at 2–3 km grid spacing) but on a 250 m grid with 2 km sites; the N5 sweep will settle whether the reach rule or their result predicts ours. A learned reflectivity operator in DA now has a precedent (Stefanelli et al. 2025, variational, grid space); ours is in a latent ensemble filter. The agent found no idealized squall-line OSSE reporting percent error reductions in a standard protocol, no published w analysis errors to compare with our "w barely constrained" (Potvin & Wicker 2012 is the place to look), and no storm-scale precedent for a particle flow filter. Sobash & Wicker's innovation-gated adaptive noise is the obvious template for N2 and for the mild U in our rank histogram.

**Verdict.** The DA *results* are novel in their setting (equation-free, latent, localized, learned operators), but the paper must position every observation-type and spread finding against the 2003–2015 EnKF literature, which already knows them. The honest framing is: "a learned local surrogate reproduces the known phenomenology of storm-scale radar EnKF without the equations."

### 4.4 Interpretable latent dynamics

**What exists.** SINDy (Brunton, Proctor & Kutz 2016, PNAS), PDE-FIND (Rudy et al. 2017, Sci. Adv.), SINDy autoencoder (Champion et al. 2019, PNAS 116:22445, low-dimensional demonstrations only), weak SINDy (Messenger & Bortz 2021, JCP 443), ensemble SINDy (Fasel et al. 2022, Proc. R. Soc. A), SINDy-PI (Kaheman et al. 2020), deep delay autoencoders (Bakarji et al. 2023, Proc. R. Soc. A), Bayesian SINDy-AE with spike-and-slab priors (Gao & Kutz 2024, Proc. R. Soc. A 480), LaSDI (Fries et al. 2022, deliberately non-sparse latent ODEs), PDE-Net (Long et al. 2018, derivative-constrained filters), universal differential equations (Rackauckas et al. 2020), stabilized neural ODEs with a sparse-linear + dense-nonlinear split (Linot et al. 2023, JCP), Linot & Graham 2021/22 (KS neural ODEs), Constante-Amores, Linot & Graham 2024 (arXiv:2410.01238: patch-shared AE + NODE on KS and Kolmogorov flow, the nearest architecture to ours) and 2025 (JFM: 17 new relative periodic orbits found in a latent model of pipe flow), Page, Brenner & Kerswell 2021 (PRF 6:034402, latent Fourier analysis of an equivariant AE), arXiv:2609.28981 (*unverified*: localized recurrent solutions in a patch latent of Kolmogorov turbulence). Ordered representations: Rippel, Gelbart & Adams 2014 (ICML; nested dropout = PCA for a semi-linear AE), Matryoshka representation learning (Kusupati et al. 2022, NeurIPS), PCA-AE (Ladjal et al. 2019; Pham et al. 2022), Rolinek et al. 2019 (VAEs pursue PCA directions). Closures and stochastic/memory terms: Lin & Lu 2021 (JCP 424, Wiener projections / Koopman–Mori–Zwanzig, KS and stochastic Burgers), Gupta & Lermusiaux 2021 (Proc. R. Soc. A, neural delay closures), Gilani, Giannakis & Harlim 2021 (Physica D), Callaham et al. 2021 (Proc. R. Soc. A, Langevin regression), Boninsegna, Nüske & Clementi 2018 (JCP, stochastic SINDy). Geophysical equation discovery: Zanna & Bolton 2020 (GRL), Jakhar et al. 2024 (JAMES: discovered closures accurate offline but unstable in rollout; filtered truth depends on the filter), Beucler et al. 2021 (PRL, hard conservation constraints). Squall-line theory for our C/Δu result: Rotunno, Klemp & Weisman 1988 (JAS 45:463), Weisman & Rotunno 2004 (JAS 61:361), Bryan, Knievel & Parker 2006 (MWR 134:2772).

**Where we sit.** Our FPc design is a UDE / SINDy-AE hybrid with the Linot et al. 2023 linear–nonlinear split, trained the Champion way; it is not architecturally new. What the agent could not find: any SINDy-AE-style law for an atmospheric or convective model; any systematic report that such a law comes out *dense* with no pruning knee at this scale (the literature reports sparse successes on low-dimensional systems; LaSDI gives up sparsity by design; Jakhar et al. report offline-accurate/rollout-unstable closures, which is our pruned-law blow-up in a latent setting); nested dropout used anywhere in scientific ML or to order the terms of a latent law; a linearization/dispersion analysis of a convective emulator with a quantified gravity-wave deficit; or recovery of known CM1 budgets (η, fall-out, condensation) along semi-Lagrangian trajectories as a benchmark. The C/Δu = 1.3–3.6 result, uncorrelated with shear, sits against RKW's "optimal" C/Δu ≈ 1 and Bryan et al.'s finding that modern models differ systematically from the original theory; it needs a held-out shear sweep before it is interpreted.

**Verdict.** Position the law work on (a) the atmospheric setting, (b) the measured density / no-knee negative result, (c) ordered channels as a sparsity prior, and (d) the dynamical diagnostics (dispersion, travelling waves), not on the local latent field or the AE+law architecture, both of which have precedents (Constante-Amores et al.; Champion et al.). Methods to adopt from the search: the weak form and ensemble/Bayesian term-selection probabilities (Messenger & Bortz; Fasel; Gao & Kutz) instead of a single L1 fit; Langevin regression or a Wiener-projection memory term for the residual R (Callaham; Lin & Lu); PDE-Net-style derivative-constrained stencils or Beucler-style conservation constraints as candidate fixes for the gravity-wave deficit and the pruning blow-up; Page et al.'s latent Fourier analysis to unify dispersion, travelling waves and the missing wave branch.

### 4.5 Predictability, extensivity, transfer and baselines

**What exists.** Predictability: Lorenz 1969 (Tellus 21:289), Lorenz 1982, Zhang, Snyder & Rotunno 2003 (JAS 60:1173), Zhang et al. 2007 (JAS 64:3579, three-stage error growth; convective-scale saturation in ~1 h), Hohenegger & Schär 2007 (BAMS 88:1783: doubling ~4 h at cloud-resolving scale vs ~40 h synoptic; tangent-linear validity ~1.5 h), Melhauser & Zhang 2012 (JAS 69:3350, practical vs intrinsic predictability of a squall line / bow echo), Selz & Craig 2015 (MWR 143:813, e-folding ~0.5 h early), Durran & Gingrich 2014 (JAS 71:2476), Durran & Weyn 2016 (BAMS, "Thunderstorms do not get butterflies"), Weyn & Durran 2017 (JAS 74) and 2018 (JAS 75:3331, spread grows faster at higher resolution), Sun & Zhang 2016 (JAS 73:1419), Judt 2018 (JAS 75:1477), Rotunno, Snyder & Judt 2023 (JAS 80:63, up-amplitude vs upscale). Extensive chaos: Egolf et al. 2000 (Nature 404:733, Rayleigh–Bénard), Karimi & Paul 2012 (PRE 85:046201), Pikovsky & Politi 1998 (Nonlinearity 11:1049) and 2016 (book), Deissler & Kaneko 1987 (Phys. Lett. A 119:397, velocity-dependent exponents), Patil et al. 2001 (PRL 86:5878, local low dimensionality), Yoon, Ott & Szunyogh 2010 (JAS 67, localization vs information propagation), Ott et al. 2004 / Hunt et al. 2007 (LEKF/LETKF), Bocquet & Carrassi 2017. ML-emulator dynamics: Selz & Craig 2023, Vonich & Hakim 2024 (GRL, sensitivity through GraphCast), Hakim & Masanam 2023/24 (arXiv:2309.10867, dynamical tests of Pangu). Transfer: Wittenberg & Holmes 1999 (Chaos 9:452), Constante-Amores et al. 2024, Bar-Sinai, Kochkov, Stachenfeld (none tests a larger domain explicitly — *check Stachenfeld before citing it for transfer*). Baselines: Hamilton, Berry & Sauer 2016, Lguensat et al. 2017, Zhao & Giannakis 2016 (Nonlinearity 29:2888, kernel-weighted analog ensembles beat single analogs). Data scaling: Rasp & Thuerey 2021, ClimaX, Aurora (*scaling numbers unverified*).

**Where we sit.** Three cautions and two opportunities. Cautions: (1) our 3–8 min doubling times are two orders of magnitude shorter than Hohenegger & Schär's ~4 h; the metric (whole-domain normalized RMS of all channels, 0.001 K perturbation, measured from initiation) and the system (2D, 1-min) differ, so the number must be presented with the metric attached and compared with Weyn & Durran 2017 and Melhauser & Zhang 2012, which the agent could not read in full; (2) "half the twin difference" as a deterministic floor is *our* heuristic — no source defines it — so the paper should derive it (two independent draws give 2σ² error variance; the conditional mean halves it) and call it a heuristic; (3) Durran & Gingrich argue the 0.001 K regime is not practically relevant, so the floor should also be computed for realistic-amplitude perturbations (L2b with 0.1–0.5 K). Opportunities: (a) the agent found no full Lyapunov spectrum or D_KY for any neural weather emulator, and no extensivity measurement (D_KY vs domain size / storm count) for a convection-resolving model — L3 (two-storm spectrum) would be the first; (b) the reach rule "localization ≥ information reach, measured from data before training" is an empirical, model-free version of Yoon, Ott & Szunyogh's argument and should be presented that way.

**Verdict.** The transfer result (claim 6) has precedents in spirit (Wittenberg & Holmes; Constante-Amores) but none in kind: no convective emulator, no interacting new configuration, no DA. The analog/persistence baselines must be introduced via Hamilton et al. 2016 and Lguensat et al. 2017, and the extended analog (mean of k neighbours) is what Zhao & Giannakis would predict works better than a single draw — which is the right justification for addendum 3b.

---

## 5. What is novel and what is worth publishing

### 5.1 Novelty ledger

| Result | Precedent | Verdict |
|---|---|---|
| Localized LETKF on a learned latent *field* with physical site coordinates | none found (global latents: Peyron, ROAD-EnKF, Pasmans, Fan; physical-space localization with ML models: Penny et al.) | **novel** |
| Training-free transfer of a surrogate *and* its DA to a 2× domain with interacting storms; ensemble-per-storm invariance | Wittenberg & Holmes (idea), Constante-Amores et al. (forecast-only, KS/Kolmogorov) | **novel in kind** |
| Equation-free convective-scale DA with learned reflectivity + Doppler operators | Stefanelli et al. 2025 (learned reflectivity operator, variational, grid space); Hamilton et al. 2016 (equation-free EnKF, Lorenz-63) | **novel in setting** |
| Controlled ablations: PCA vs AE in DA; 32 vs 64 channels hurts; deterministic+noise vs CRPS-stochastic; latent vs physical analysis | none found | **novel**; theory hook in Al-Ghattas & Sanz-Alonso |
| Full-state neural emulator of an idealized CRM squall line with locality-based sample efficiency and a learning curve | StormCast (HRRR, 1-h, diffusion, global); none on idealized CRMs | **novel** (search negative) |
| Stage C (decoded multi-step fine-tuning) as the largest gain; cold-pool bias quantified | pushforward trick (Brandstetter), PDE-Refiner | incremental but reportable |
| Smoothing isolated to the objective (SM0) + Lyapunov spectrum vs CM1 twins + no objective tweak closes the gap | Bonavita; Selz & Craig 2023/26; Chen et al. 2026 (all global) | **confirms at storm scale with the mechanism isolated**; first full spectrum of a weather-type emulator found |
| Gravity-wave phase-speed deficit and travelling-wave C/Δu from the emulator's linearization | Page et al. 2021 (latent Fourier), Constante-Amores 2025 (RPOs in latent pipe flow) | **novel diagnostics**; interpretation of C/Δu needs RKW-style sweep |
| CM1 twin-run doubling times of 3–8 min | Hohenegger & Schär (~4 h, different metric), Weyn & Durran 2017 | quotable but must carry the metric; check W&D and M&Z |
| Explicit quadratic latent PDE matching the neural propagator; *dense*, no pruning knee | Champion et al. (low-dim successes), LaSDI (non-sparse by design), Jakhar et al. (offline-accurate, rollout-unstable) | **novel negative result** for the atmospheric setting |
| Nested dropout to order latent channels under a latent law | Rippel et al. 2014 (no scientific-ML use found) | **novel use** |
| Known-law recovery from CM1 along semi-Lagrangian trajectories (η budget, fall speed, condensation) | PDE-FIND Lagrangian sensors | methods contribution |
| 30-min rain-onset prediction from a local latent neighbourhood | Behrens, Mooers (regime clustering only) | **novel**, modest |
| Reflectivity degrades θ′; Vr most valuable; additive noise; RTPS; boundary effect | Dowell et al.; Snyder & Zhang; Dowell & Wicker; Whitaker & Hamill; Torn et al. | **rediscoveries**: present as consistency checks |

### 5.2 Publication plan

**Paper III — "Equation-free localized ensemble data assimilation of a squall line with a learned local latent surrogate" (MWR or JAMES; QJRMS as alternative).** Ready to draft once N1–N3 (G2b, T2 extension, calibration fix, non-local baseline) are in. Contributions in order: localized latent LETKF on a learned lattice; learned radar operators; the six pre-registered claims; the transfer result; the ablations (PCA, channels, stochastic). Known phenomenology (observation complementarity, additive noise, boundaries) goes in as validation against the 2003–2015 EnKF literature. State plainly: 2D, perfect-model twin, dense synthetic observations, one microphysics; calibration 0.71 with a mild U, or 1.04 with the stochastic model at 7% cost. The equation-free framing must cite Hamilton, Berry & Sauer 2016 and Lguensat et al. 2017; the latent-DA framing Peyron 2021, ROAD-EnKF, Pasmans 2025, Fan 2025/26, LAE-EnKF; the architecture Constante-Amores et al. 2024.

**Paper IV — "What a learned local emulator of a cloud-resolving model knows: skill, sample efficiency, chaos and sharpness" (JAMES or AIES).** Contributions: the emulator and its learning curve; Stage C; the SM0 → L1/L5 → L2/SM5 chain (smoothing is the objective; the surrogate is 20–60× less chaotic than CM1; stochastic/flow models restore sharpness and calibration but not growth); the dispersion relation and travelling waves (Q1/Q2); the predictability floor (after L2b); the probes (P). Needs: L2b, the float64 Lyapunov check, a KE-transfer diagnostic, M4 (growth from small differences) as the constructive ending or an honest open problem, and a quantitative comparison with StormCast's cold-pool/updraft statements.

**Paper V — "Can a cloud-resolving model's latent dynamics be written down? Ordered channels, explicit latent PDEs and the density of learned laws" (Chaos, JFM Rapids, or Proc. R. Soc. A; possibly combined with the FP0 known-answer benchmark).** Contingent on FPd/M5. Either outcome is a paper: a sparse law on ordered channels with named physics, or the measured minimum residual share at ≤ 500 terms plus the no-knee result and the ordered-channel dimension estimate, framed with Jakhar et al. (offline-accurate / rollout-unstable) and LaSDI (sparsity is not the right prior for coarse-grained closures).

**Short notes worth considering.** (i) The CM1 twin-run growth rates and the two-thermal/tiny-perturbation patches as a community resource (GMD or MWR note), after L2b. (ii) Localization vs information reach measured from data (a methods note, Yoon-Ott-Szunyogh in empirical form), after N5.

**Not worth a paper on its own:** the attention/capacity results (fold into IV as an ablation), tilings (fold into III as a control), SM4c's negative results (fold into IV), the PCA probes (fold into IV).

---

## 6. Research programme

The programme below is ordered by expected value per unit of cluster time, with the near-term items chosen to close Paper III and the mid-term items chosen to open the two follow-on papers. Every study has a hypothesis, a design, a pre-registered reading, and a cost estimate (Saidie H200-hours or Cashew core-hours). Standing constraints: ≤ 2 Saidie GPUs; no CM1-based EnKF (the premise is DA without equations); no banded channel-index regularizer; settings tuned on the 6 validation storms only.

### 6.1 Near term (close Paper III; 1–2 weeks)

**N1. Finish the pre-registered runs.** G2b extended grid (Saidie 9061–9072), T2 extension (Cashew 750368), FPc arm 4, FPd. No new design; record under the existing rules.

**N2. Calibration without hand tuning (G4).**
- *Hypothesis:* innovation-gated adaptive additive noise (Sobash & Wicker 2015's reflectivity-innovation gating, moved to the latent) or Anderson-2009 spatially varying inflation brings spread/error into 0.9–1.1 and flattens the rank histogram without hurting cycled nMSE by more than 3%. Sobash & Wicker also warn that repeated coherent additive noise biases the cold pool; check the cold-pool bias of the analyses under each scheme.
- *Design:* deterministic Stage C, additive noise scaled per site by the running innovation statistic; three variants (global scalar, per-site, per-variable-group); 6 validation storms to pick one; 22 test storms × 3 seeds.
- *Reading (pre-register):* adopt if spread/error ∈ [0.9, 1.1], outer-bin ratio < 1.1, and paired cycled nMSE ≥ −3% of the headline. Report the stochastic model (1.04) as the alternative either way.
- *Cost:* ~3 GPU-h.

**N3. Non-local baseline (S1 = H1b).** The missing head-to-head.
- *Hypothesis:* a global-latent model of matched parameter count (a 1D U-Net or a ViT over the 400 km domain, bottlenecked to one 6400-dim vector) trained on the same 93 storms is worse at 30–60 min, needs more storms to reach the local model's skill, cannot ingest an 800 km input, and cannot be localized in DA (SEC-style or none).
- *Design:* two architectures; learning curve at 7 / 14 / 28 / 93 storms; forecast table; DA with (a) no localization, (b) Anderson-2012 sampling-error correction on the latent covariance, vs the local model's 5 km localization at N = 8 / 32.
- *Reading:* the claim "locality buys X" is the measured difference; no threshold needed, but report per-N.
- *Cost:* ~6 GPU-h (S55 already provides the global-latent code path).

**N4. Ensemble size and the 64-channel puzzle (G6 + I1).**
- *Hypothesis:* the 64-channel AE's DA penalty (+0.023 at N = 32) vanishes at N = 128 and the 32-channel model's error falls by ≥ 4% from N = 32 to 128 (the tiling control predicts this).
- *Design:* N = 8 / 16 / 32 / 64 / 128 for both AEs, noise re-tuned per N on validation.
- *Reading:* if the 64-ch penalty is gone at N ≥ 64, the "richer latent hurts" result is a sampling-error statement and should be framed as such (and connects to the local-dimension argument).
- *Cost:* ~4 GPU-h.

**N5. Localization vs reach (G7).** Sweep localization 2 / 5 / 10 / 20 / 40 km against the measured information reach (~2 km/min × 5-min cycle = 10 km) and the decoder reach. *Reading:* the optimum should sit at or above the reach rule; report whether Paper II's rule predicts it, and compare with Sobash & Stensrud 2013's 12–18 km optimum at storm scale and Yoon, Ott & Szunyogh 2010's information-propagation argument. ~1 GPU-h.

**N8. Two literature-driven checks before any emulator claim.** (a) Re-run the L1 Lyapunov spectrum in float64 on CPU for 2 storms and confirm the positive exponents are not numerical noise (Selz & Craig 2026 found exactly this artefact in some AI models). (b) Add a kinetic-energy spectral-transfer diagnostic to SM5 (Chen et al. 2026 show spatially uncorrelated perturbations pile up small-scale energy; our noise channels share the site structure, so this is a chance to show the opposite). Both are CPU afternoons.

**N6. Cost table (G9).** Wall time per forecast hour and per cycle, surrogate vs CM1 (CPU core-hours and GPU-seconds). One afternoon.

**N7. L2b: perturb at a 60-min restart.** 8 pairs at 1e-3 K and 8 pairs at a realistic 0.3 K (Durran & Gingrich 2014: the tiny-perturbation regime is not the practical one), applied after reading a 60-min restart. Gives the intrinsic floor for forecasts started at 1 h, i.e. the one our skill numbers can be compared with; report the doubling times with the metric stated and alongside Weyn & Durran 2017 and Melhauser & Zhang 2012. ~60 CPU-h on Cashew. Turns §1.4's suggestive floor into a claim or kills it. Also the natural place for L3 (Lyapunov spectrum on the two-storm runs: does D_KY double?), which no convective-model study has measured.

### 6.2 Mid term (open Paper IV and V; 1–2 months)

**M1. Sparse observations and cycle length (G8b).** *The highest-value scientific experiment not yet run.*
- *Hypothesis (assimilation in the unstable subspace):* as observations thin (2 → 4 → 8 → 16 km) and cycles lengthen (5 → 10 → 15 → 30 min), the advantage of the deterministic model with hand-set additive noise over the stochastic model (and over the flow-matching model) shrinks and then reverses, because the filter increasingly depends on the forecast ensemble's spread lying in the growing directions, which additive isotropic noise does not provide and which an under-chaotic deterministic model cannot generate.
- *Design:* 4 × 4 grid of spacing × cycle, three forecast models, noise/RTPS re-tuned on validation per cell; 22 storms × 3 seeds; scores: cycled nMSE, spread/error, w and qv gains, FSS(35 dBZ) of forecasts from analyses.
- *Reading:* the crossover cell, if any. If the stochastic/generative models win at 8–16 km, that is a result about *which kind* of surrogate convective-scale DA needs where radar coverage is poor — stronger than any accuracy number.
- *Cost:* ~15 GPU-h.

**M2. Nonlinear latent filter (E6).** A domain-localized particle flow filter (Paper I/II's `ParticleFlowFilter`, per-site like the LETKF) for the reflectivity operator and bounded, bimodal condensate.
- *Hypothesis:* PFF analyses have lower reflectivity and condensate error and bimodal spread where rain is intermittent; the gain is largest in M1's sparse cells.
- *Design:* global tapered PFF first (6200 × 6200 is feasible on an H200), then per-site PFF; compare with the headline LETKF on 22 × 3; also run on M1's sparsest cell.
- *Reading:* adopt if condensate RMSE improves ≥ 10% with CI excluding 0 at no cycled-nMSE cost.
- *Cost:* code ~3 days; runs ~10 GPU-h.

**M3. Mean-anchored stochastic model (I6 / SM4c idea 5).** Members = deterministic step of the mean + flow-sampled residual perturbations; this is the CorrDiff / ArchesWeatherGen residual design (Mardani et al. 2025; Couairon et al. 2024) at the 1-min step. *Hypothesis:* keeps deterministic mean accuracy (0.354 at 60 min) with the flow model's sharp samples and calibration. *Reading:* CRPS ≤ stochastic model's, skill within 2% of deterministic, SM0 sharpness ≥ SM4b's at 30 min, 4–16 km variance within 0.8–1.2 at 60 min. ~4 GPU-h. If it works it becomes the DA ensemble model for M1.

**M4. Growth from small differences (the SM5 gap).** No current surrogate reproduces CM1's 3–8 min doubling; stochastic ones jump to saturation. *Hypothesis:* a flow/diffusion propagator trained on *pairs* of nearby CM1 twin trajectories (the L2 data) with a divergence-matching term learns state-dependent, direction-selective growth. *Design:* add a pair-divergence loss on the L2 twin pairs (10 pairs × 4 h = ~2400 frame pairs); test with SM5's protocol. *Reading:* sample-pair growth rate within a factor 2 of CM1's at 15–60 min while CRPS stays within 5%. ~6 GPU-h plus ~100 CPU-h for more twin pairs if needed. This is speculative but is the one experiment that would address the Lyapunov mismatch directly.

**M5. Ordered channels + sparse law (FPd follow-on).**
- *Hypothesis:* with nested-dropout-ordered channels, a law on the first k channels (k = 4, 8, 12), with the remaining channels treated as a stochastic closure, reaches ≤ 200 terms at ≤ 10% skill loss at 30 min; the leading terms are advection/diffusion of channels that probes identify with cold-pool depth/strength and column condensate.
- *Design:* FPd recipe on the nested AE, truncated to k channels before the law; term selection by ensemble/weak SINDy inclusion probabilities (Fasel et al. 2022; Messenger & Bortz 2021) or a spike-and-slab prior (Gao & Kutz 2024) rather than a single L1 fit; the residual R replaced by a Langevin-regression noise term (Callaham et al. 2021) or a short memory term (Lin & Lu 2021) as the closure for the dropped channels; probes (P1 machinery) on the leading channels; dispersion relation and travelling-wave speed of the sparse law vs Q1/Q2, with Page et al. 2021's latent Fourier analysis on z(x,t) to tie them together.
- *Reading (pre-register):* success = ≤ 200 terms, residual share < 0.1, 30-min skill within 10% of the dense law, all k channels used (ablation > 1e-3), and ≥ 2 leading channels with probe R² > 0.8 to named physical quantities.
- *Cost:* ~6 GPU-h.

**M6. Q4/Q5/Q7: mechanisms from the surrogate.** Response kernels (decoder ∘ propagator ∘ encoder linear responses to a cold anomaly behind the gust front; gravity-wave forcing by the updraft), covariant Lyapunov vectors decoded to physical patterns vs where L2 twin differences grow first, and in-silico interventions (strengthen the cold pool along a probe direction, read the gust-front speed) checked against RKW. Cheap (CPU), and it is the content of Paper IV's "what the emulator knows" section. ~50 CPU-h plus 3–4 targeted CM1 runs for validation.

**M7. Third environment axis (X1).** `ens_C`: ~30 members varying CAPE/lapse rate and/or initiation (cold pool instead of warm bubble), held out entirely. *Reading:* H3-style; this is the only way to widen the generalization claim beyond two parameters. ~1 day on Cashew, then ~2 GPU-h.

**M8. Realistic observations (G8).** Range-limited radar, blocked sector, surface network 5–20 km, operator error (learned vs perfect reflectivity). Feeds Paper III's limitations section or Paper IV. ~6 GPU-h.

### 6.3 Long term (the project's stated goal; 3–12 months)

**L1. 3D CM1 squall lines.** The 2D model is a 1D latent field; 3D makes it a 2D latent field of column sites (Paper II's 2D machinery exists: block-diagonal Jacobians, per-site LETKF, tilings). The costs scale as: CM1 3D at 1 km × 250 m is ~10³× a 2D run; a 20-member 3D ensemble is weeks of Cashew time. Design the ensemble around what the learning curve says (~28 storms per environment) and the output interval the convective cycle needs (1–2 min). Pre-register the same six claims. This is also where the Warn-on-Forecast comparison becomes meaningful.

**L2. Real observations (OSSE → nature run with a different model → real radar).** Step 1: nature run from a *different* CRM configuration (different microphysics, resolution) to break the perfect-model setting; step 2: WRF or CM1 3D with a real case and real radar data as the truth proxy. Each step needs the learned observation operators retrained on the new truth generator or on real data.

**L3. Multi-scale latent (P8) only if needed.** The pooled local latent already matches physical baselines for storm-scale targets; revisit only if 3D storm-scale organization (bow echoes, line-end vortices) is poorly captured.

### 6.4 Sequencing and GPU budget

Week 1: N1–N2, N5–N7 (≈ 10 GPU-h + 30 CPU-h). Week 2: N3, N4 (≈ 10 GPU-h). Weeks 3–6: M1, M3, M5 (≈ 25 GPU-h), M6 (CPU). Weeks 6–10: M2 (code-heavy), M4, M7, M8. Paper III drafting starts after week 2; Paper V (laws) after M5; Paper IV (emulator/predictability) after M4/M6 and L2b.

---

## 7. Risks

1. **Referee: "it's 2D and a twin."** Mitigations in order of cost: realistic observations (M8), different-model nature run (L2 step 1), ens_C (M7), 3D (L1). Say plainly in Paper III what is and is not claimed.
2. **Calibration slips.** If N2 fails, the paper should lead with the stochastic model's 1.04 at a 7% cost and present the deterministic result as the accuracy-optimal variant.
3. **The sparse-observation result goes the "wrong" way** (hand-set noise still wins at 16 km / 30 min). That is still publishable — it says realistic dynamics are not needed for convective DA at these scales — but then M4 loses priority.
4. **The sparse law never gets sparse.** Then the publishable statement is the minimum residual share at ≤ 500 terms and the ordered-channel dimension estimate; pair it with the FP0 known-answer results as a methods paper.
5. **Compute contention.** Everything above fits in ≤ 2 Saidie GPUs; the 3D step does not fit in the current Cashew allocation and needs a plan.
