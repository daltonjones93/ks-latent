# Diagnostics Report (Phase 6, D1-D9)

Profile: `full`

## D1: sensitivity maps

| latent k | centroid (x) | resultant length R | circular spread |
|---|---|---|---|
| 9 | 2.656 | 0.001 | 3.767 |
| 0 | 11.421 | 0.202 | 1.788 |
| 13 | 14.395 | 0.001 | 3.833 |
| 14 | 15.167 | 0.001 | 3.722 |
| 7 | 16.664 | 0.002 | 3.540 |
| 6 | 17.007 | 0.001 | 3.651 |
| 5 | 17.228 | 0.001 | 3.777 |
| 15 | 17.319 | 0.002 | 3.590 |
| 3 | 17.715 | 0.001 | 3.691 |
| 1 | 18.097 | 0.001 | 3.740 |
| 12 | 18.214 | 0.001 | 3.750 |
| 11 | 19.298 | 0.001 | 3.678 |
| 10 | 19.520 | 0.003 | 3.447 |
| 2 | 19.944 | 0.000 | 3.920 |
| 4 | 20.027 | 0.001 | 3.682 |
| 8 | nan | 0.000 | inf |

## D2: wavenumber content

| latent k | spectral centroid \|k\| | spectral bandwidth | wavelet energy entropy |
|---|---|---|---|
| 0 | 0.5441 | 2.6191 | 0.4155 |
| 1 | 0.1155 | 0.2567 | 0.3265 |
| 2 | 0.2280 | 0.5102 | 0.3640 |
| 3 | 0.3410 | 0.7521 | 0.4804 |
| 4 | 0.4483 | 0.9879 | 0.4731 |
| 5 | 0.5538 | 1.2232 | 0.4927 |
| 6 | 0.6564 | 1.4568 | 0.4928 |
| 7 | 0.7503 | 1.6812 | 0.5013 |
| 8 | nan | nan | -0.0000 |
| 9 | 0.1149 | 0.2573 | 0.3271 |
| 10 | 0.2313 | 0.5089 | 0.3609 |
| 11 | 0.3409 | 0.7518 | 0.4803 |
| 12 | 0.4484 | 0.9867 | 0.4729 |
| 13 | 0.5555 | 1.2250 | 0.4933 |
| 14 | 0.6575 | 1.4586 | 0.4922 |
| 15 | 0.7502 | 1.6814 | 0.5011 |

Mean wavelet energy entropy: 0.417 (0=maximally localized, 1=delocalized).

## D3: Jacobian coupling graph

Bandedness (Fiedler-seriated): **0.9485**
**p-value vs. 1000 random permutations: 0.8000**
Verdict: NOT significant (dense/unstructured coupling)
Discovered ordering (Fiedler permutation): `[8, 1, 15, 2, 5, 11, 6, 14, 12, 0, 3, 4, 9, 10, 13, 7]`

Note (brief §8 D3): the original seriation attempt used the *covariance*, which is approximately I by construction because of L_decorr, so it could not have found anything -- this diagnostic uses the propagator Jacobian instead.

## D4: translation representation

| shift c | relative residual |
|---|---|
| 0 | 0.0000 |
| 16 | 0.0030 |
| 32 | 0.0035 |
| 48 | 0.0032 |
| 64 | 0.0029 |
| 80 | 0.0034 |
| 96 | 0.0035 |
| 112 | 0.0029 |
| 128 | 0.0024 |
| 144 | 0.0031 |
| 160 | 0.0033 |
| 176 | 0.0034 |
| 192 | 0.0030 |
| 208 | 0.0032 |
| 224 | 0.0035 |
| 240 | 0.0025 |

Mean relative residual: 0.0029
Mean group-property error (||R(c1)R(c2) - R(c1+c2)|| / ||R(c1+c2)||): 0.0136
**Verdict: Genuine (approximately) equivariant translation representation found**

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

Bandedness (Fiedler-seriated): **0.4859**
**p-value vs. 1000 random permutations: 0.6170**
Verdict: NOT significant (no detectable temporal coherence structure)
Discovered ordering (Fiedler permutation): `[8, 14, 2, 13, 4, 9, 0, 7, 5, 3, 12, 1, 10, 15, 11, 6]`

Note (user-directed 2026-08-30, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 8): D3's Jacobian-bandedness is blind to a real coupling structure through a broadband-but-still-ring-respecting operator (e.g. an FNO spectral conv with many active modes). D6 instead looks for emergent spatial structure directly in the raw ENCODED DATA -- lagged, windowed cross-correlation (window=100, max_lag=5) between latent channels along real trajectories through time, a statistic the L_decorr regularizer's same-time covariance constraint does not touch -- independent of any one propagator.

## D7: same-time channel correlation (not part of the original brief)

Bandedness (Fiedler-seriated): **0.7749**
**p-value vs. 1000 random permutations: 0.9880**
Verdict: NOT significant (no detectable same-time correlation structure)
Discovered ordering (Fiedler permutation): `[9, 3, 4, 5, 6, 13, 14, 10, 11, 7, 15, 2, 12, 0, 1, 8]`

Note (user-directed 2026-08-31, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 22): D3's own docstring claims the naive same-time covariance 'is approximately I by construction because of L_decorr, so it could not have found anything' -- this was checked empirically against real checkpoints and found to be an overstatement (mean off-diagonal |correlation| ~0.1, max ~0.5 on the checkpoints tested, not remotely flat). D7 seriates that real same-instant |Pearson correlation| matrix directly -- no propagator, no time lag, the most literal reading of 'do neighboring latent variables vary together.'

## D8: same-time SIGNED channel correlation (not part of the original brief)

Signed bandedness (fixed index order, no permutation search): **-0.0024**
**p-value vs. 1000 random relabelings: 0.6980**
Verdict: NOT significant (no detectable same-sign local coherence)

Note (user-directed 2026-09-01): D7 uses |correlation|, so an anti-correlated near-neighbor scores identically to a correlated one -- it cannot distinguish 'coupled' from 'coherent'. D8 uses SIGNED correlation and scores the ACTUAL, fixed latent index order directly (no Fiedler search -- that assumes non-negative edge weights, which a signed correlation matrix does not satisfy), against a null of random relabelings of the same channels' signed pairwise correlations. This is the diagnostic-side counterpart to Stage1/Stage2TrainingConfig.spatial_signed.

## D9: real-trajectory smoothness (not part of the original brief)

| measure | median | p95 |
|---|---|---|
| step size \|\|z_t+1 - z_t\|\| (raw) | 0.0079 | 0.0212 |
| step size (normalized by trajectory RMS radius) | 0.0223 | 0.0600 |
| curvature \|\|z_t+1 - 2z_t + z_t-1\|\| (raw) | 0.0006 | 0.0026 |
| curvature (normalized) | 0.0016 | 0.0073 |
| propagator step-Jacobian spectral norm | 0.8913 | 0.8966 |

Note (user-directed 2026-09-04/05, docs/HANDOFF_2026-09-04_FOURIER_HYBRID.md and scripts/analyze_latent_smoothness.py): the encoder-side step size/curvature (rows 1-4) is a propagator-independent measure of whether the ENCODER's real-trajectory placement of physical states is smooth (motivated directly by a GIF-visible 'huge jumps' observation on Section 82, later confirmed quantitatively by this exact measure). The propagator step-Jacobian spectral norm (row 5, when supported) is the trained propagator's own local Lipschitz constant, sampled at real points on the attractor -- a mixture of WANTED chaotic expansion (KS genuinely has positive Lyapunov exponents) and UNWANTED roughness, so a lower value is not unconditionally better, unlike rows 1-4.
