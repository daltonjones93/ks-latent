# Diagnostics Report (Phase 6, D1-D9)

Profile: `full`

## D1: sensitivity maps

| latent k | centroid (x) | resultant length R | circular spread |
|---|---|---|---|
| 12 | 0.842 | 0.134 | 2.006 |
| 7 | 1.072 | 0.058 | 2.387 |
| 2 | 4.324 | 0.119 | 2.063 |
| 13 | 6.647 | 0.043 | 2.509 |
| 0 | 7.537 | 0.086 | 2.215 |
| 4 | 12.041 | 0.026 | 2.704 |
| 10 | 12.931 | 0.082 | 2.238 |
| 15 | 13.592 | 0.123 | 2.049 |
| 5 | 14.329 | 0.087 | 2.211 |
| 6 | 14.688 | 0.055 | 2.411 |
| 14 | 14.804 | 0.165 | 1.899 |
| 1 | 15.466 | 0.026 | 2.697 |
| 11 | 16.821 | 0.186 | 1.836 |
| 9 | 18.927 | 0.124 | 2.043 |
| 8 | 19.944 | 0.070 | 2.308 |
| 3 | 21.633 | 0.097 | 2.161 |

## D2: wavenumber content

| latent k | spectral centroid \|k\| | spectral bandwidth | wavelet energy entropy |
|---|---|---|---|
| 0 | 0.0359 | 0.2543 | 0.3762 |
| 1 | 0.0335 | 0.2811 | 0.3840 |
| 2 | 0.0538 | 0.3018 | 0.3744 |
| 3 | 0.0360 | 0.3148 | 0.3831 |
| 4 | 0.1325 | 0.5218 | 0.4012 |
| 5 | 0.0507 | 0.2882 | 0.3692 |
| 6 | 0.0278 | 0.2438 | 0.3856 |
| 7 | 0.0238 | 0.2302 | 0.3696 |
| 8 | 0.0846 | 0.3649 | 0.3777 |
| 9 | 0.0532 | 0.3292 | 0.3807 |
| 10 | 0.0594 | 0.3328 | 0.3914 |
| 11 | 0.1013 | 0.3823 | 0.3782 |
| 12 | 0.0546 | 0.3048 | 0.3716 |
| 13 | 0.0491 | 0.2987 | 0.3852 |
| 14 | 0.0973 | 0.4409 | 0.3812 |
| 15 | 0.0414 | 0.2884 | 0.3755 |

Mean wavelet energy entropy: 0.380 (0=maximally localized, 1=delocalized).

## D3: Jacobian coupling graph

Bandedness (Fiedler-seriated): **0.6000**
**p-value vs. 1000 random permutations: 0.9340**
Verdict: NOT significant (dense/unstructured coupling)
Discovered ordering (Fiedler permutation): `[0, 4, 8, 2, 9, 7, 10, 1, 5, 13, 6, 11, 3, 12, 14, 15]`

Note (brief §8 D3): the original seriation attempt used the *covariance*, which is approximately I by construction because of L_decorr, so it could not have found anything -- this diagnostic uses the propagator Jacobian instead.

## D4: translation representation

| shift c | relative residual |
|---|---|
| 0 | 0.0000 |
| 16 | 0.2206 |
| 32 | 0.2725 |
| 48 | 0.2767 |
| 64 | 0.2817 |
| 80 | 0.2986 |
| 96 | 0.2818 |
| 112 | 0.2820 |
| 128 | 0.2797 |
| 144 | 0.2691 |
| 160 | 0.3009 |
| 176 | 0.3004 |
| 192 | 0.2924 |
| 208 | 0.2830 |
| 224 | 0.2886 |
| 240 | 0.2215 |

Mean relative residual: 0.2593
Mean group-property error (||R(c1)R(c2) - R(c1+c2)|| / ||R(c1+c2)||): 0.4781
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

Bandedness (Fiedler-seriated): **0.5017**
**p-value vs. 1000 random permutations: 0.2720**
Verdict: NOT significant (no detectable temporal coherence structure)
Discovered ordering (Fiedler permutation): `[11, 10, 4, 12, 5, 3, 2, 13, 1, 9, 6, 8, 14, 15, 7, 0]`

Note (user-directed 2026-08-30, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 8): D3's Jacobian-bandedness is blind to a real coupling structure through a broadband-but-still-ring-respecting operator (e.g. an FNO spectral conv with many active modes). D6 instead looks for emergent spatial structure directly in the raw ENCODED DATA -- lagged, windowed cross-correlation (window=100, max_lag=5) between latent channels along real trajectories through time, a statistic the L_decorr regularizer's same-time covariance constraint does not touch -- independent of any one propagator.

## D7: same-time channel correlation (not part of the original brief)

Bandedness (Fiedler-seriated): **0.5350**
**p-value vs. 1000 random permutations: 0.0330**
Verdict: SIGNIFICANT bandedness
Discovered ordering (Fiedler permutation): `[15, 14, 7, 6, 0, 8, 9, 5, 1, 13, 12, 4, 3, 10, 11, 2]`

Note (user-directed 2026-08-31, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 22): D3's own docstring claims the naive same-time covariance 'is approximately I by construction because of L_decorr, so it could not have found anything' -- this was checked empirically against real checkpoints and found to be an overstatement (mean off-diagonal |correlation| ~0.1, max ~0.5 on the checkpoints tested, not remotely flat). D7 seriates that real same-instant |Pearson correlation| matrix directly -- no propagator, no time lag, the most literal reading of 'do neighboring latent variables vary together.'

## D8: same-time SIGNED channel correlation (not part of the original brief)

Signed bandedness (fixed index order, no permutation search): **0.3532**
**p-value vs. 1000 random relabelings: 0.0000**
Verdict: SIGNIFICANT signed bandedness

Note (user-directed 2026-09-01): D7 uses |correlation|, so an anti-correlated near-neighbor scores identically to a correlated one -- it cannot distinguish 'coupled' from 'coherent'. D8 uses SIGNED correlation and scores the ACTUAL, fixed latent index order directly (no Fiedler search -- that assumes non-negative edge weights, which a signed correlation matrix does not satisfy), against a null of random relabelings of the same channels' signed pairwise correlations. This is the diagnostic-side counterpart to Stage1/Stage2TrainingConfig.spatial_signed.

## D9: real-trajectory smoothness (not part of the original brief)

| measure | median | p95 |
|---|---|---|
| step size \|\|z_t+1 - z_t\|\| (raw) | 0.8511 | 2.1413 |
| step size (normalized by trajectory RMS radius) | 0.0864 | 0.2170 |
| curvature \|\|z_t+1 - 2z_t + z_t-1\|\| (raw) | 0.4068 | 1.4688 |
| curvature (normalized) | 0.0414 | 0.1493 |
| propagator step-Jacobian spectral norm | 2.1332 | 3.3328 |

Note (user-directed 2026-09-04/05, docs/HANDOFF_2026-09-04_FOURIER_HYBRID.md and scripts/analyze_latent_smoothness.py): the encoder-side step size/curvature (rows 1-4) is a propagator-independent measure of whether the ENCODER's real-trajectory placement of physical states is smooth (motivated directly by a GIF-visible 'huge jumps' observation on Section 82, later confirmed quantitatively by this exact measure). The propagator step-Jacobian spectral norm (row 5, when supported) is the trained propagator's own local Lipschitz constant, sampled at real points on the attractor -- a mixture of WANTED chaotic expansion (KS genuinely has positive Lyapunov exponents) and UNWANTED roughness, so a lower value is not unconditionally better, unlike rows 1-4.
