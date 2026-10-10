# Part II narrative (Block K0, 2026-10-10)

The skeleton of §7, the Part II sentences of the abstract, Key Point 3, the §1 bullet and the §10
paragraph. Every number comes from a JSON in `artifacts/experiments/da2d/` (named in brackets).
Units: physical RMSE in field units. σ_obs = 0.1 × the training-data std (6.17) = **0.62 field units** (the 44² truth's own field std is ≈ 3.75; corrected 2026-10-10 — earlier drafts said ≈ 0.40).
Steady state = cycles 51–100. Placeholders `[[Pn]]` are listed at the end with the run that fills them.

**Setting.** Part II uses `p7c12d10_ks2d`: the Part I architecture with 12 channels per site, trained
only on fields carrying one fixed realization of observation-level noise (σ = 0.1). Filter: per-site
local ETKF in latent space, R-localization (GC half-width 1.5 sites), RTPS 0.7, 64 members per tiling,
observations every 2.75 length units (half a patch), 5 model steps per cycle, 100 cycles. One setting
everywhere; nothing is tuned per domain size or per experiment.

---

## Claim 1 — Shifted tilings: the new DA mechanism (lead with it)

*Statement.* Running K independent latent filters on patch lattices shifted by half a patch, and
blending their decoded analyses with a Hann partition of unity, cuts the analysis error by ~30% with
no retraining, removes the patch seams, and uses members better than one larger ensemble.

*Evidence.*
- 44², seed 0: one tiling 0.457 → four tilings **0.324**; seeds 1–2: 0.431 → 0.310, 0.420 → 0.302
  [`h9_I33_d10_T{1,4}_s*`]. 440²: 0.434 → 0.311 [`h9_D10_L440_I1_T{1,4}`].
- More tilings: K = 2 0.369, K = 4 0.324, K = 16 **0.287** (spread/error 0.97 → 0.92 → 0.51: accuracy
  vs calibration trade-off; K = 4 is the calibrated default) [`h9_D10_L44_I4_*`].
- Equal total of 64 members: 1 × 64 0.457 / 0.431 / 0.418; 4 × 16 **0.352 / 0.334 / 0.318** (23–24%
  better on every seed) [`h9_D10_L44_I2_*`].
- **Control (Part II model, 2026-10-10):** four ensembles of 64 on the SAME tiling, blended the same way:
  0.450 / 0.423 / 0.412 (spread/error 0.44–0.48), i.e. no better than one tiling (0.457 / 0.431 / 0.418)
  and badly under-dispersed, against 0.324 / 0.310 / 0.302 (0.92–0.99) for four shifted tilings
  [`h9_K2b_ctrl_T4_s*`]. The gain comes from the offsets, not from more members or from averaging.
- Seams (edge/interior gradient ratio, truth 1.02): one tiling 1.73, blend **1.02** [`seams_I9.json`].
- Mechanism: the single-tiling analysis error is tied to the patch lattice; the tilings' errors are
  only weakly correlated (0.11–0.42 by wavenumber band) so they average down [`h9_D10_L44_diag_T4`].
- Exactness is preserved: each tiling's filter is exactly local (Prop. blockdiag) and the blend is a
  fixed linear map. Cost: ×K serially, ×1 with K processes.
- Context: this is cycle spinning (Coifman & Donoho 1995) on the patch lattice, applied to the
  analysis rather than to a denoiser.

## Claim 2 — One model and one setting from 44² to 440²

*Statement.* The same model and filter keep the same accuracy over a hundredfold area and latent
dimension (d = 768 → 76 800), with a calibrated ensemble and a fixed number of members.

*Evidence.* [`h9_D10_L*_I1_T*`, `h9_I33_d10_*`]

| size | K = 1 | K = 4 (seed 0, seed 1) | spread/error | interpolation | decoder floor 1 / 4 tilings |
|---|---|---|---|---|---|
| 44² | 0.457 | 0.324, 0.310 (seed 2: 0.302) | 0.92–0.99 | 0.829 | 0.251 / 0.137 |
| 88² | 0.423 | 0.304 | 0.98 | 0.808 | 0.242 / 0.129 |
| 220² | 0.434 | 0.312, 0.308 | 0.96 | 0.820 | 0.247 / 0.134 |
| 330² | 0.452 | 0.326, 0.316 | 0.92 | 0.815 | 0.254 / 0.142 |
| 440² | 0.434 | 0.311 | 0.96 | 0.819 | 0.246 / 0.134 |

- Climatological OI 0.80. Free run 4.2–5.3 (skill ≈ 14–17).
- Members per tiling: 8 → 0.450, 16 → 0.352, 32 → 0.330, 64 → 0.324, 128 → 0.322 at 44²; 16 → 0.329,
  32 → 0.315, 128 → 0.311 at 220²: saturated by ≈ 32 at both sizes [`h9_D10_L{44,220}_I3_N*`].
- Unlocalized comparison (same model and filter, taper ≡ 1): [[P1]].
- Cost: [[P2]].
- Long-run stability at 220²: [[P3]].

## Claim 3 — The latent space earns its place in the analysis

*Statement.* With the same learned model, analysing in latent space is more accurate and better
calibrated than analysing in physical space.

*Evidence.* The learned model as a physical-space emulator (encode on 4 tilings, propagate, decode,
blend) inside a tuned physical-space local ETKF: 0.354 / 0.345 / 0.338 (spread/error 1.22–1.27),
against the latent analysis 0.324 / 0.310 / 0.302 (0.92–0.99), better on every seed; one tiling 0.461
vs 0.457 [`i24_D10_I24_*`]. Reason to state: the latent analysis never re-encodes a blended or
off-attractor field (re-encoding the blend costs accuracy, I8).

## Claim 4 — Trained only on noisy data, and better for it

*Statement.* The model never sees a clean field. Training on noisy fields costs nothing and
improves the filter slightly at every domain size.

*Evidence.*
- Clean-data vs noisy-data model, K = 4, seed 0: 44² 0.338 → 0.324, 88² 0.321 → 0.304, 220² 0.326 →
  0.312, 330² 0.340 → 0.326, 440² 0.326 → 0.311; 44² seeds 1–2 0.322 → 0.310, 0.320 → 0.302.
- Single-tiling seams at matched settings 1.94 → 1.73 [`h9_J18c_clean_T*`, `seams_I9.json`].
- Mechanism (J18b) [`j18b_roundtrip.json`]: the noisy-trained autoencoder reconstructs clean states 15%
  better (0.039 vs 0.046 standardized) and removes 15–20% more grid-scale noise (round-trip ratio 0.44
  vs 0.53 at noise 0.1), but neither model pulls smooth or latent-space perturbations back toward the
  attractor (ratio ≈ 1). Say "denoising and reconstruction", not "projection onto the attractor".
- D_KY of the Part II model: 3.6% (44²), 2.1% (66²) from the truth.
- Training-noise trend: [[P4]].

## Claim 5 — Against physics-based DA with an imperfect model and every standard remedy

*Statement.* A physics-based local ETKF (same observations; RTPS and localization tuned per case;
online parameter estimation; bias correction; additive model error) beats the latent filter when its
model error is small or fixable, and loses when the error is large or structural. The crossover is
set by the physics model's irreducible one-cycle forecast error relative to the observation error.

*Evidence* (3-seed means; latent 0.312, range 0.302–0.324) [`i11_I28*`, `i11_J16*`, `i11_J17H*`]:

| physics model error | one-cycle error: total / state-dependent part | physics, best incl. remedies | outcome |
|---|---|---|---|
| none (true equations) | 0 / 0 | 0.076 | floor |
| 64² grid | 0.000 / 0.000 | 0.083 | physics |
| 32² grid | 0.037 / 0.036 | 0.18 (bias correction no help) | physics 1.7× |
| ∇² coefficient 1.1 | 0.082 / 0.077 | 0.225 (with estimation) | physics 1.4× |
| fixed forcing 0.2 / 0.4 | 0.101 / 0.008; 0.203 / 0.016 | 0.225 / 0.225 (bias correction) | physics 1.4× |
| ∇² coefficient 1.25 | 0.210 / 0.199 | 0.420 (with estimation) | latent 1.35× |
| spurious mean flow 1 / 2 | 0.78 / 0.72; 1.53 / 1.42 | 0.44 / 0.55–0.61 (with estimation) | latent 1.4× / 1.8× |
| damping order 5 / 6 / 8, (a, b, g) estimated online | 0.072 / 0.069 (p = 5); 0.129, 0.210 total (p = 6, 8; state part [[P5b]]) | seed 0 tuned: 0.296 / 0.351 / 0.432 | [[P6]] |
| hidden memory field ε = 0.1 / 0.2 (own latent model) | 0.165 / 0.377 | [[P7]] | [[P7]] |

- Robustness: the physics filter diverges or loses to interpolation outside a narrow RTPS window
  (e.g. damping order 6: 0.351 at RTPS 0.7, 1.8–2.1 at 0.9–1.0, 0.52–0.72 at 0.5); its best setting is
  over-dispersed (spread/error 1.7–2.0). The latent filter uses one default throughout.
- Bias correction does nothing for the state-dependent errors (orders 5/6/8: 0.297 / 0.361 / 0.440 vs
  0.296 / 0.351 / 0.432 without).
- **Organizing variable (found 2026-10-10 15:15 from the P5 files):** the *state-dependent* part of the
  physics model's one-cycle error (the std of the residual after removing its mean, which bias
  correction removes). Every case where it is ≤ 0.077 (64², 32², forcing 0.008–0.016, coefficient 1.1,
  order 5) is a physics win or tie; every case where it is ≥ 0.2 (coefficient 1.25, mean flow, order 8
  pending) is a latent win. Order 6 (total 0.129) sits in between. So the crossover is near 0.1 field
  units ≈ 0.16 σ_obs (σ_obs = 0.62 field units: a sixth of the observation error). Caveat: for the estimated cases the offline value is at
  the wrong starting parameter; estimation lowers the effective error, so the true crossover can only
  be lower than these points suggest for those rows. The forcing cases show why the total error is the
  wrong axis: 0.2 total, but almost all bias.
- Crossover statement: [[P8]] (fill the order-6/8 and memory points, then state the threshold).
- Same comparison at 220²: [[P9]].

## Claim 6 — Honest frame

- The latent filter's single-tiling floor is the decoder's (0.24–0.25); with four tilings the blended
  decoder floor is 0.13, so the remaining 0.31 is mostly filter error, not reconstruction. A correct
  or correctable physics model gets far below it (0.076–0.23).
- The method is for systems without a trustworthy model; no equations are needed, but complete
  (noisy) training fields are. CM1 (Paper 3) is the target.
- Long cycling with sparse observations (stride 16) does not reach a steady state within 100 cycles
  for any variant; it is limited by the learned model's stability, and the failsafes (noise-trained
  propagator, manifold gate, tiling reset) belong to the follow-up paper. [[P10]]
- Twin experiments; one 2D system; model errors chosen, not calibrated to a real model.
- Forecasts from the analyses: [[P11]].
- Robustness to the initial ensemble and to a misspecified R: [[P12]].

---

## Draft text for the summary places (fill placeholders first)

**Abstract (Part II, 3 sentences).** "With a per-patch decoder the latent observation operator is
exactly local, and running independent filters on shifted copies of the patch lattice and blending
their analyses removes the patch seams and cuts the analysis error by a third without retraining. A
model trained only on noisy fields, with one filter setting, analyses 2D Kuramoto–Sivashinsky fields
at 0.31 of the field units against 0.82 for interpolating the same observations, on every domain from
44² to 440². [[P8-abstract]]"
- P8-abstract options: (A) "It outperforms a physics-based ensemble filter equipped with online
  parameter estimation and bias correction whenever the physical model's structure is wrong";
  (B) "It outperforms such a filter once the physical model's one-cycle forecast error exceeds about
  [x] of the observation error"; (C) "It is competitive with such a filter, and better when the
  physical model's error is large".

**Key Point 3 options** (≤ 140 characters): (A/B) "A latent filter learned only from noisy data keeps
its accuracy from 44² to 440² and beats a physics filter with a wrong model"; (C) "Shifted tilings cut
latent filter error by a third without retraining, and one setting holds over a hundredfold area".

**§10 paragraph (4 sentences).** Claims 1, 2, 4, 5 in that order, worded per the chosen branch.

---

## Placeholders to fill when the runs finish

| id | what | run / file | expected |
|---|---|---|---|
| P1 | unlocalized vs localized, N = 64/128, K = 1/4, at 44²/66²/88² | `run_k1_unloc.sh` → `h9_K1u_*`, `h9_K1l_*` | ~21:00 |
| P2 | wall time and peak memory vs area, K = 1 and 4 | I14 → `h9_I14_*` | ~03:00 |
| P3 | 300-cycle run at 220² (drift or not) | I16c → `h9_I16c_L220_300cyc_T4` | ~20:30 |
| P4 | σ = 0.3 training-noise point (44² seeds 0–2, 220²) and its round-trip ratios | J18a → `h9_J18a_*`, `j18b_roundtrip.json` | ~23:00 |
| P5 | ~~one-cycle error for coefficient 1.1/1.25, mean flow 1/2, 64²~~ FILLED 15:08 (K10b) | `j16_model_error_*.json` | done |
| P5b | state-dependent part for orders 6 and 8 (`j16_model_error_hyp{6,8}.json`, written by run_j17h.sh) | J17H | ~16:00 |
| P6 | damping orders 5/6/8: 3-seed best incl. (a,b,g) estimation and RTPS 0.6; outcome per the pre-registered rule | J17H → `i11_J17H_*`, `j17h.log` | ~18:30 |
| P7 | hidden memory ε = 0.1 / 0.2: physics best incl. remedies, floor (true (u,v) model), latent K = 4 seeds 0–2, in RMSE and RMSE/σ_obs | J17B → `i11_J17B_*`, `h9_J17B_*` | ~16:30 / ~20:00 |
| P8 | crossover sentence: the one-cycle error (in σ_obs units) where the latent filter starts to win; abstract branch A/B/C | P5–P7 | after P7 |
| P9 | coefficient 1.25 and mean flow 1 at 220² (physics with estimation vs latent 0.312) | J5 → `i11_J5_*` | ~06:00 |
| P10 | observation density with the Part II model, strides 4/8/16 (+300 cycles at 16) | K10(a) → `h9_K10a_*` | ~19:30 |
| P11 | forecast error vs lead (blend, single tiling, true init, persistence, climatology) | I7 → `h9_I7_*` (`forecast` key) | ~20:30 |
| P12 | climatological initial ensemble; assumed R 0.5× / 2× | I16a/b → `h9_I16a_*`, `h9_I16b_*` | ~20:30 |

## Decisions recorded with K0 (user, 2026-10-10)
- K1(1): the unlocalized-filter failure is rerun with the Part II model (`run_k1_unloc.sh`), not quoted
  from the 6-channel runs.
- K1(2): the 1D results stay in §7 only if they are remarkably good for the thesis, otherwise App. E.
  Assessment: they are on-thesis and distinctive in two places — the unlocalized filter's required
  ensemble doubles with every doubling of the domain while the localized one needs 16 members at
  every size, and distance-based localization in the latent field beats distance-free sampling error
  correction by up to 3.5× (the direct answer to "why give the latent a geometry at all?"). Their
  absolute skill (2.3–3.0) is modest and they use the old skill metric and the PFF/EnKF filters.
  **Recommendation: keep one compact §7 subsection with those two results and Fig. 8; move the
  observation-spacing numbers, the L = 100 sweep and the SEC implementation details to App. E.**
- Correction found while writing K0: the "≈ 0.25 decoder floor" is the single-tiling floor. With four
  tilings the blended decoder floor is 0.13, so the 0.31 analysis is mostly filter error. K0 claim 6
  and K5.3 must say this (the earlier "the floor is the decoder's" framing applies to K = 1 only).

## Filled 2026-10-10 ~16:00 (results that landed)
- **P6 (partial), J17H joint (a, b, g) estimation, seed 0** (after the print-bug rerun): order 5 0.266, order 6
  0.323, order 8 0.425 (tuned only: 0.296 / 0.351 / 0.432; RTPS 0.6 c 8.25: 0.289 / 0.356 / 0.420). Estimation
  helps orders 5–6 by ~10%, order 8 by 2%. Seeds 1–2 running for the best configs. Provisional reading:
  order 5 physics wins (~0.27 vs 0.31), order 6 tie (0.323 vs 0.324 on seed 0), order 8 latent 1.35×.
  Consistent with the crossover at state-dependent one-cycle error ≈ 0.1 (orders 5/6/8: 0.069/0.123/0.201).
- **P10 observation density (K10a, 44², seed 0):** stride 4 (1.4 units): 4 tilings 0.226 (cal. 0.86), 1 tiling
  0.333, interpolation 0.539. Stride 8: 0.324 / 0.457 / 0.829. Stride 16 (5.5 units, one observation per
  site): 0.769 (cal. 1.06) / 0.990 / interpolation 3.008 — 4× better than interpolation. 300 cycles at
  stride 16: 50-cycle block means 0.55, 0.77, 0.72, 0.82, 0.72, 0.86 — bounded, slowly fluctuating,
  possibly a weak upward trend; no divergence (the 6-channel/clean-model drift warning does not reproduce
  with the Part II model over 300 cycles).
- **P11 forecasts (I7, 44², 3 seeds, 10 starts each):** blended-analysis forecast error at leads of 1 / 5 /
  10 / 20 / 40 / 60 cycles (0.5 time units each): seed 0 0.38 / 0.79 / 1.42 / 2.87 / 4.66 / 4.93, single
  tiling 0.51 / 0.92 / 1.62 / 3.33 / 5.19 / 5.49, forecast from the encoded truth 0.21 / 0.56 / 1.08 / 2.40
  / 4.57 / 4.95, persistence 0.75 / 3.00 / 4.58 / 5.44, field std ≈ 3.8–4.2. The blend's advantage persists:
  26% at 1 cycle, 12–14% at 10–20 cycles (≈ 0.8–1.6 Lyapunov times), gone by ~40 cycles. Same on seeds 1–2.
  → branch A ("a better initial condition, not only a smoother picture").
- **P12 robustness (I16a/b):** a climatological initial ensemble (no knowledge of the truth) starts at 1.33
  and reaches the steady state within ~8 cycles; its steady error is identical (0.324) to the run started
  from the encoded truth: the filter forgets its initial ensemble. Assumed observation error 0.5× / 2× the
  true: 0.351 (cal. 0.58) / 0.338 (cal. 1.40) vs 0.324 (0.92): 4–8% worse, calibration moves as expected.
- Still pending: P1 (~21:00), P2 (overnight), P3 (I16c 300 cycles at 220², running), P4 (~23:00), P7 (J17B:
  ε = 0.1 model trained 15:54, latent runs next; ε = 0.2 later), P9 (overnight).
