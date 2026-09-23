# Diagnostics Report (Phase 6, D1-D9)

Profile: `full`

## D1: sensitivity maps

| latent k | centroid (x) | resultant length R | circular spread |
|---|---|---|---|
| 20 | 0.792 | 0.000 | 4.070 |
| 25 | 2.046 | 0.004 | 3.319 |
| 23 | 2.868 | 0.022 | 2.761 |
| 24 | 2.883 | 0.014 | 2.914 |
| 8 | 3.010 | 0.015 | 2.904 |
| 9 | 3.309 | 0.011 | 3.002 |
| 22 | 3.457 | 0.011 | 2.993 |
| 7 | 3.890 | 0.012 | 2.971 |
| 6 | 3.983 | 0.006 | 3.194 |
| 10 | 4.485 | 0.010 | 3.049 |
| 21 | 4.879 | 0.005 | 3.279 |
| 0 | 6.092 | 0.080 | 2.249 |
| 1 | 9.064 | 0.007 | 3.171 |
| 19 | 10.333 | 0.002 | 3.513 |
| 16 | 11.966 | 0.010 | 3.034 |
| 3 | 12.252 | 0.001 | 3.715 |
| 2 | 12.405 | 0.006 | 3.209 |
| 18 | 14.271 | 0.004 | 3.336 |
| 12 | 14.333 | 0.013 | 2.958 |
| 26 | 14.787 | 0.001 | 3.677 |
| 17 | 14.847 | 0.006 | 3.226 |
| 5 | 16.097 | 0.002 | 3.495 |
| 4 | 16.106 | 0.002 | 3.522 |
| 13 | 16.202 | 0.026 | 2.699 |
| 11 | 17.472 | 0.004 | 3.301 |
| 14 | 17.639 | 0.034 | 2.602 |
| 29 | 19.180 | 0.051 | 2.438 |
| 27 | 20.043 | 0.016 | 2.873 |
| 28 | 20.438 | 0.031 | 2.638 |
| 15 | nan | 0.000 | inf |

## D2: wavenumber content

| latent k | spectral centroid \|k\| | spectral bandwidth | wavelet energy entropy |
|---|---|---|---|
| 0 | 0.3079 | 2.1814 | 0.4021 |
| 1 | 0.1124 | 0.2488 | 0.3264 |
| 2 | 0.2170 | 0.4861 | 0.3804 |
| 3 | 0.3367 | 0.7404 | 0.4751 |
| 4 | 0.4475 | 0.9923 | 0.4934 |
| 5 | 0.5742 | 1.3446 | 0.4958 |
| 6 | 0.7449 | 2.0260 | 0.5056 |
| 7 | 1.0977 | 3.2804 | 0.5176 |
| 8 | 1.8771 | 5.3083 | 0.5090 |
| 9 | 2.1296 | 5.2654 | 0.5493 |
| 10 | 1.8542 | 5.0094 | 0.5320 |
| 11 | 1.2508 | 4.2586 | 0.4873 |
| 12 | 0.6577 | 3.1119 | 0.4408 |
| 13 | 0.4153 | 2.2559 | 0.4349 |
| 14 | 0.3639 | 1.7403 | 0.3923 |
| 15 | nan | nan | -0.0000 |
| 16 | 0.1127 | 0.2521 | 0.3262 |
| 17 | 0.2210 | 0.4845 | 0.3789 |
| 18 | 0.3359 | 0.7420 | 0.4755 |
| 19 | 0.4480 | 0.9933 | 0.4925 |
| 20 | 0.5749 | 1.3443 | 0.4964 |
| 21 | 0.7483 | 2.0144 | 0.5053 |
| 22 | 1.1008 | 3.3003 | 0.5172 |
| 23 | 1.8316 | 5.2540 | 0.5066 |
| 24 | 2.1364 | 5.2759 | 0.5483 |
| 25 | 1.8721 | 5.0105 | 0.5344 |
| 26 | 1.2467 | 4.2040 | 0.4917 |
| 27 | 0.6585 | 3.0913 | 0.4429 |
| 28 | 0.3720 | 2.0807 | 0.4310 |
| 29 | 0.3619 | 1.7989 | 0.3880 |

Mean wavelet energy entropy: 0.449 (0=maximally localized, 1=delocalized).

## D3: Jacobian coupling graph

Bandedness (Fiedler-seriated): **0.6309**
**p-value vs. 1000 random permutations: 0.0030**
Verdict: SIGNIFICANT bandedness
Discovered ordering (Fiedler permutation): `[15, 0, 3, 5, 6, 7, 13, 18, 25, 28, 26, 10, 20, 9, 8, 4, 27, 1, 11, 22, 24, 23, 14, 19, 17, 16, 12, 2, 21, 29]`

Note (brief §8 D3): the original seriation attempt used the *covariance*, which is approximately I by construction because of L_decorr, so it could not have found anything -- this diagnostic uses the propagator Jacobian instead.

## D4: translation representation

| shift c | relative residual |
|---|---|
| 0 | 0.0000 |
| 16 | 0.0006 |
| 32 | 0.0010 |
| 48 | 0.0013 |
| 64 | 0.0016 |
| 80 | 0.0017 |
| 96 | 0.0018 |
| 112 | 0.0020 |
| 128 | 0.0020 |
| 144 | 0.0020 |
| 160 | 0.0019 |
| 176 | 0.0017 |
| 192 | 0.0015 |
| 208 | 0.0014 |
| 224 | 0.0011 |
| 240 | 0.0007 |

Mean relative residual: 0.0014
Mean group-property error (||R(c1)R(c2) - R(c1+c2)|| / ||R(c1+c2)||): 0.4567
**Verdict: No clean linear translation representation found in the existing latent**

## D5: local intrinsic dimension vs. patch length

| patch length | d_local (two-NN) |
|---|---|
| 5 | 2.212 |
| 10 | 2.477 |
| 20 | 3.001 |
| 30 | 3.211 |
| 40 | 3.374 |
| 50 | 3.449 |

Fitted slope: 0.0274 (95% CI [0.0149, 0.0400]); extensivity prediction: 0.226. Intercept: 2.2452.

## D6: temporal coherence structure (not part of the original brief)

Bandedness (Fiedler-seriated): **0.3044**
**p-value vs. 1000 random permutations: 0.6050**
Verdict: NOT significant (no detectable temporal coherence structure)
Discovered ordering (Fiedler permutation): `[22, 6, 11, 9, 25, 14, 26, 12, 4, 19, 13, 7, 2, 0, 23, 5, 16, 8, 20, 3, 27, 18, 29, 17, 21, 10, 24, 1, 28, 15]`

Note (user-directed 2026-08-30, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 8): D3's Jacobian-bandedness is blind to a real coupling structure through a broadband-but-still-ring-respecting operator (e.g. an FNO spectral conv with many active modes). D6 instead looks for emergent spatial structure directly in the raw ENCODED DATA -- lagged, windowed cross-correlation (window=100, max_lag=5) between latent channels along real trajectories through time, a statistic the L_decorr regularizer's same-time covariance constraint does not touch -- independent of any one propagator.

## D7: same-time channel correlation (not part of the original brief)

Bandedness (Fiedler-seriated): **0.6140**
**p-value vs. 1000 random permutations: 0.9970**
Verdict: NOT significant (no detectable same-time correlation structure)
Discovered ordering (Fiedler permutation): `[1, 17, 29, 18, 28, 0, 3, 14, 8, 10, 12, 13, 25, 16, 5, 26, 11, 21, 20, 24, 22, 6, 9, 7, 23, 4, 27, 19, 2, 15]`

Note (user-directed 2026-08-31, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 22): D3's own docstring claims the naive same-time covariance 'is approximately I by construction because of L_decorr, so it could not have found anything' -- this was checked empirically against real checkpoints and found to be an overstatement (mean off-diagonal |correlation| ~0.1, max ~0.5 on the checkpoints tested, not remotely flat). D7 seriates that real same-instant |Pearson correlation| matrix directly -- no propagator, no time lag, the most literal reading of 'do neighboring latent variables vary together.'

## D8: same-time SIGNED channel correlation (not part of the original brief)

Signed bandedness (fixed index order, no permutation search): **-0.0022**
**p-value vs. 1000 random relabelings: 0.7400**
Verdict: NOT significant (no detectable same-sign local coherence)

Note (user-directed 2026-09-01): D7 uses |correlation|, so an anti-correlated near-neighbor scores identically to a correlated one -- it cannot distinguish 'coupled' from 'coherent'. D8 uses SIGNED correlation and scores the ACTUAL, fixed latent index order directly (no Fiedler search -- that assumes non-negative edge weights, which a signed correlation matrix does not satisfy), against a null of random relabelings of the same channels' signed pairwise correlations. This is the diagnostic-side counterpart to Stage1/Stage2TrainingConfig.spatial_signed.

## D9: real-trajectory smoothness (not part of the original brief)

| measure | median | p95 |
|---|---|---|
| step size \|\|z_t+1 - z_t\|\| (raw) | 0.0530 | 0.1434 |
| step size (normalized by trajectory RMS radius) | 0.0117 | 0.0317 |
| curvature \|\|z_t+1 - 2z_t + z_t-1\|\| (raw) | 0.0193 | 0.0892 |
| curvature (normalized) | 0.0043 | 0.0197 |
| propagator step-Jacobian spectral norm | 1.1879 | 1.4554 |

Note (user-directed 2026-09-04/05, docs/HANDOFF_2026-09-04_FOURIER_HYBRID.md and scripts/analyze_latent_smoothness.py): the encoder-side step size/curvature (rows 1-4) is a propagator-independent measure of whether the ENCODER's real-trajectory placement of physical states is smooth (motivated directly by a GIF-visible 'huge jumps' observation on Section 82, later confirmed quantitatively by this exact measure). The propagator step-Jacobian spectral norm (row 5, when supported) is the trained propagator's own local Lipschitz constant, sampled at real points on the attractor -- a mixture of WANTED chaotic expansion (KS genuinely has positive Lyapunov exponents) and UNWANTED roughness, so a lower value is not unconditionally better, unlike rows 1-4.
