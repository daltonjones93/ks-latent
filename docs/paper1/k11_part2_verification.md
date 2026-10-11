# K11: Part II number provenance check

Scope: `sections/07_data_assimilation.tex` (full), DA sentences of `main.tex` (abstract, key points, PLS),
`sections/01_introduction.tex`, `sections/09_discussion.tex` (the two DA paragraphs), `sections/10_conclusions.tex`,
and `appendix/E_da_details.tex`. P7 and P9 placeholders ignored. Checked 2026-10-11 against
`artifacts/experiments/da2d/*` and `artifacts/*` (`da2d/` paths abbreviated below).

**Summary: 192 numbers (or tight groups) verified / 19 mismatched or problematic / 3 unsourced.**

Generated tables: all seven Part II tables (`tab07_p2_sizes`, `tab07_p2_members`, `tab07_p2_model_error`,
`tab07_p2_forecast`, `tabE_p2_robustness`, `tab_da_cost`, `tabE_p2_networks`) were regenerated from the current
JSONs into a scratch directory with copies of their generators. All seven match the committed `.tex`
byte for byte, so every table cell is current. None of them reads an old-model `da2d_*` file. No number in the
text comes from the superseded 6-channel filter, except where the text says it does (App. E l.98-99 "earlier
single-tiling filter", l.205-207 "earlier version", l.209-213 "earlier comparison").

## Mismatches and problems

| file:line | text | source says | suggested correction | severity |
|---|---|---|---|---|
| 07:86-88 | "at equal cost the tilings are $23$--$24\%$ more accurate on every seed" | 1-(4x16)/(1x64): 22.98, 22.42, 23.92 % (`h9_D10_L44_I2_T4_N16_s{0,1,2}` blend 0.3518/0.3344/0.3184 vs `h9_I33_d10_T1_s0`, `h9_D10_L44_I2_T1_N64_s{1,2}` 0.4566/0.4311/0.4184) | "$22$--$24\%$" | rounding |
| 07:188 | "The latent analysis is $5$--$10\%$ more accurate on every seed" | 1-latent/hybrid: 8.42, 10.05, 10.65 % (`i24_D10_I24_T4_c8.25_rtps0.5_s{0,1,2}` analysis 0.354/0.345/0.338 vs `h9_I33_d10_T4_s*` 0.324/0.310/0.302) | "$8$--$11\%$" | wrong number |
| 07:184 | physical-space analysis "tuned over its localization radius and RTPS" | for `p7c12d10_ks2d` only one configuration exists (`i24_D10_I24_T*_c8.25_rtps0.5*`). The localization x RTPS sweep (`i24_I24_T{1,4}_c{8.25,16}_rtps{0,0.5,0.7}`) was run with the clean-trained `p7c12_ks2d`. Values quoted are correct for that one configuration | "with the localization radius and RTPS tuned on the clean-trained model" (or run the sweep with d10) | wrong config |
| 07:236 | "$3.40$ and $1.77$ with four tilings" (88^2, unlocalized, 128 members) | `h9_K1u_L88_T4_N128` steady.blend = 1.76470 | "$1.76$" | rounding |
| 07:401 | "At its best setting it is over-dispersed, with spread over error $1.7$--$2.0$" | order 6 at its best settings: tuned (c16, RTPS 0.7) calib 1.75/1.70/1.63; est. (abg) 1.73/1.69/1.59 (`i11_J17H_p6_c16_rtps0.7[_abg]_s*`). Across all wrong-model best configs: 0.96-1.29 (grids), 1.14-1.22 (order 8), 1.74-1.85 (forcing, 10% coeff.), 3.0-3.3 (25% coeff.), 4.4-6.4 (advection) | for order 6: "$1.6$--$1.75$"; or drop the range and say "over-dispersed (spread over error up to $6$)" | wrong number |
| 07:399-400 | "multiplicative inflation without RTPS diverged or was far worse in every wrong-model case" | `i11_J2_*_inf{1.1,1.2}`: 32^2 grid 0.191 at inf 1.2 vs 0.184 tuned (4% worse); 64^2 grid 0.093 vs 0.077 (21% worse); no inflation runs for damping order or hidden field. "Far worse" holds for forcing, coefficient and advection only | "...was far worse in every case with a wrong coefficient, forcing or advection, and slightly worse on the coarse grids" | wrong claim |
| 07:465-466 | KY "within $3.6\%$ and $2.1\%$ ... both within the truth's own spread" | `crop_eval_p7c12d10_ks2d.log`: 84.73 at 44^2. Truth runs at 44^2: 80.66, 81.77, 83.60 (tabC_truth_runs); 84.73 lies above the range (3.3% above their mean 82.01). The percentages are against the seed-0 truth only (81.77, 192.5). 66^2 (188.45) and the clean model (83.46, 190.37) are inside the range | "the clean-trained one within the truth's spread, the noisy-trained one just outside it at $44^2$"; optionally say the percentages are against one truth run | wrong claim |
| main.tex:83-85 (PLS) | "a surrogate learned from noisy measurements does as well as a filter built on equations that are slightly wrong" | §7.5 and tab07_p2_model_error: with slightly wrong equations the physics filter wins by 1.19-3.75x (10% coefficient 1.38x, forcing 1.38-1.39x, grids, order 5). Ties occur only at order 6 and the hidden field | e.g. "is beaten by a filter built on equations that are slightly wrong, but does better once..." | wrong claim |
| 07:665-666 | peak memory "grows more slowly than the area (slope $0.75$)" | least-squares on log axes over 44^2-440^2 (`h9_I14_L*_T{1,4}` peak_rss_GB): 0.677 (K=1), 0.682 (K=4). About 0.73-0.77 only if 44^2 is dropped (88^2-440^2) | "slope $0.68$" | wrong number |
| AppE:210-211 | earlier L=100 comparison: SEC on the global vector "with errors $5$--$25\%$ lower" | `artifacts/da_sweep_phaseC2_global.csv` (sec) vs `da_sweep_phaseD3_local224.csv` (gaspari_cohn), rmse_da: SEC lower by 4.6, 3.1, 11.7, 15.7, 18.8, 18.6 % at N=8...256. No metric in these files gives 5-25% (skill ratio gives 5-42%) | "$3$--$19\%$" | wrong number |
| 07:383 | estimated nabla^2 coefficient "settles at $1.12$--$1.20$" | theta_mean, seed 0: 1.120 (eps 0.1), 1.197 (eps 0.2). Seeds 1-2: 1.099, 1.092 / 1.179, 1.167 (`i11_J17B_e*_pde_c8.25_rtps0.7_lap_s*`) | "$1.09$--$1.20$" (or say "on seed 0") | wrong config |
| 07:367-368 | "every case at $0.2$ or above (the $25\%$ coefficient, the advection, damping order 8)" | state-dependent one-cycle error of the 25% coefficient = 0.199 (`j16_model_error_lap1.25.json` sigma) | "at about $0.2$ or above" | rounding |
| 07:473 | "reconstructs clean states $15\%$ more accurately ($0.039$ against $0.046$)" | `j18b_roundtrip.json`: 0.0392 vs 0.0456. That is 14.0% lower error (the clean model's is 16.3% higher) | "$14\%$" | rounding |
| 07:59-60 | "The run without assimilation has errors of $4.2$--$5.3$" | free run over all h9 Part II size runs: 4.19-5.29, but 10.06 at 330^2 seed 0 (`h9_D10_L330_I1_T4` free). The exception is explained only in the Table VII caption | add "(10.1 for one truth at $330^2$, Table ...)" or "typically" | wrong number (omission) |
| main.tex:64 (abstract); 10:36-37 | "errors of $0.31$ ... from $44^2$ to $440^2$" / "an error of $0.31$ ... on every domain" | four tilings, all sizes and seeds: 0.302-0.326 (tab07_p2_sizes); seed 0: 0.304-0.326. The introduction (01:128) gives "$0.30$--$0.33$" | "$0.30$--$0.33$" (consistent with Sec. 1), or "about $0.31$" | rounding |
| 09:84-85 | "about 32 members per tiling at every size" | the member sweep exists only at 44^2 and 220^2 (`h9_D10_L44_I3_N*`, `h9_D10_L220_I3_N*`) | "at both sizes tested" | wrong config |
| AppE:252-253 | "is within $10\%$ of its steady value from cycle 4 at every size" | true for 66^2-440^2 (3-4). At 44^2 single-cycle errors leave the +-10% band until cycles 42-51 (per-cycle noise; `h9_I33_d10_T4_s*` per_cycle.blend). Table tabE_p2_robustness gives "1" for the same run because it counts the first cycle at or below 1.1x steady | "first within 10% by cycle 4 at every size" or "from cycle 4 at $66^2$ and beyond" | wrong claim |
| 07:426-427 (Table caption, model error) | "$^\dagger$A configuration run on seed 0 only was lower still; its seeds 1--2 are pending." | no row of the regenerated tab07_p2_model_error.tex carries a dagger | drop the dagger sentence | stale |
| 07:605 / tabC_truth_runs | "$L=400$, where $n_+=50$--$51$" | the 50 is from `e1_groundtruth_lyapunov_L400.json` (n_positive 50, D_KY 89.66). Table C lists the two L=400 runs as 51 and 51 (from `_fullspec*`, also 89.66 / 89.46). Text and Table C disagree | make Table C and the text use the same runs (either "50--51" in both or "51") | wrong config (internal) |

Lower-priority notes, not counted as mismatches:
- Lyapunov-time conversions (07:32 "about 0.08 Lyapunov times", Table tab07_p2_forecast, 07:506-510) use the seed-0 truth lambda_1 = 0.165. The three 44^2 truth runs give 0.165, 0.188 and 0.194 (mean 0.182), which would scale every Lyapunov-time value by +10%.
- Fig. 11 caption (07:273) "cycle 99 of 100": the stored index is 99 in a 0-based array (`h9_K10e_L440_T4_fields_fields.npz` cycle = [50, 74, 99]), i.e. the last of 100 cycles. Say "cycle 100 of 100", or keep 99 if cycles are counted from 0 elsewhere.
- 07:71/75 "about $30\%$ lower on every seed": exact values 29.1, 28.0, 27.7 % ("about 28--29%" would be exact). Acceptable as written.
- 07:612-618 "up to $3.5$ against the serial one": holds only when the serial SEC EnKF is compared with the GC **EnKF** (3.52 at N=32). Against GC-PFF it is 3.83 (N=32). The 1.7 against the projected form is PFF vs PFF (1.70 at N=16). Consider "(same filter)" for clarity.

## Unsourced

| file:line | text | what was searched |
|---|---|---|
| 07:231 | "about $170$ at $88^2$" (unstable directions) | no truth Lyapunov run at 88^2 in tabC_truth_runs, `artifacts/experiments/e1_*`, `da2d/`, or the task log (grep "170", "88"). It agrees with extrapolating the 44^2 density (43/44^2 x 88^2 = 172) | say "an extrapolated $\approx170$" |
| AppE:97-99 | stochastic per-site filter "same skill to within $0.15$ ... at about $40\%$ higher CPU time" (earlier single-tiling filter) | grep of full_paper_tasks.md and decisions.md for "40%", "CPU time", "within 0.15"; no matching JSON identified in `da2d/` (old `da2d_*` files not opened). Labeled as an earlier filter in the text. **Search was light (App. E)** |
| 07:194-196 | "16-member one-cycle priors" (Gaussianity) | `gaussianity_L44_p7c12d10_ks2d.json` has the skewness and kurtosis values but does not record the member count. The values themselves are verified |

## Decimal convention (RMSE to 3 decimals in running text)

Sec. 7 running text: lines 59 (0.076), 71-73, 75-76, 80, 83, 86-87, 169, 185-187, 208, 220-221, 312 (0.076), 318-319,
346-348, 353-354, 357-358, 385-389, 459-460, 463, 480-483, 524, 529-531. Lines 364 and 367 are one-cycle errors
(0.016, 0.000-0.077) and line 473 is a standardized reconstruction error; 224 (0.043) is a density, not an RMSE.
App. E: 127 (0.076), 250 (0.302-0.324); 146 and 244 are one-cycle and reconstruction errors.
Note: 0.076 rounds to 0.08 at 2 decimals, which loses the "four times" comparison's precision.
Several comparisons (seeds, 5-10% gaps, ties) also depend on the third decimal.

## Verified (file:line -> source)

Abbreviations: `S` = `da2d/h9_<tag>.json` steady.<field>; `I` = `da2d/i11_<tag>.json` steady; T-sizes = tab07_p2_sizes,
T-mem = tab07_p2_members, T-me = tab07_p2_model_error, T-rob = tabE_p2_robustness, T-cost = tab_da_cost (all regenerated, identical).

Sec. 7
- 07:23 KY 3.6% (44^2), 2.1% (66^2) -> `logs/crop_eval_p7c12d10_ks2d.log` 3.626, `crop_eval_p7c12d10_ks2d_x1.5.log` 2.102 (vs one truth run; see note above). Part I 6-channel comparable: `crop_eval_p7b_ks2d.log` 1.7-2.8%.
- 07:32 0.5 time units, 5 steps -> args n_prop_steps 5; 0.08 Lyapunov times = 0.5 x 0.165.
- 07:34 spacing 2.75, four obs per site -> 44/128 x 8; m=256 over 64 sites (tabE_p2_networks).
- 07:36 0.62 field units -> `interp_baseline_L44.json` obs_noise_std_field 0.6172; `j18b_roundtrip.json` field_units_std 6.172.
- 07:54/158/211 interpolation 0.81-0.83 -> S.interp 0.808-0.829 over all sizes/seeds.
- 07:55/159 OI 0.80 -> re-ran `scripts/exp_oi_bound.py` (read-only): 0.798; the value hard-coded in `exp_h9_tiled_da.py:440`.
- 07:59/312/408 true-PDE filter 0.076 -> I `L44_st8_c24_rtps0` 0.0763. "Four times" 0.312/0.076 = 4.1.
- 07:71-72 one tiling 0.457/0.431/0.418, four 0.324/0.310/0.302 -> S `I33_d10_T1_s0`, `D10_L44_I2_T1_N64_s{1,2}`, `I33_d10_T4_s{0,1,2}`.
- 07:75 two tilings 0.369, sixteen 0.287 -> S `D10_L44_I4_T2`, `D10_L44_I4_T16`.
- 07:76 calibration 0.86 / 0.92-0.99 / 0.51 -> calib_blend 0.864; 0.920, 0.967, 0.987; 0.512.
- 07:80 uniform 0.313 at 0.63 -> S `D10_L44_I4_T4u` 0.313, 0.629.
- 07:83-84 same tiling 0.450/0.423/0.412, calib 0.44-0.48 -> S `K2b_ctrl_T4_s*` (0.438-0.476).
- 07:86-87 4x16 0.352/0.334/0.318; 1x64 values -> S `D10_L44_I2_T4_N16_s*`, `D10_L44_I2_T1_N64_s*`.
- 07:92 edges 0.49-0.58, centres 0.38 -> recomputed from `h9_D10_L44_fields_T1_fields.npz` (as fig13): 0.485 / 0.583, min 0.383; also `j18c_seams.json`.
- 07:93-94, 102, 469 seam ratio 1.73, truth 1.02, blend 1.02, clean 1.94 -> `seams_I9.json`: D10_L44_fields_T1 tiling0 1.727, truth 1.017, D10_L44_diag_T4 blend 1.017, J18c_clean_T1 1.940.
- 07:97-98 inter-tiling correlation 0.37-0.42 below 1.6, 0.11 above, both sizes -> `h9_D10_L44_diag_T4`, `h9_D10_L220_diag_T4` steady.diag.mean_offdiag_corr (0.395-0.420, 0.371-0.422; 0.113, 0.110).
- 07:100-101 variance factor 1.4-1.9 and 3.9 -> diag tiling0_var/blend_var at 44^2: 1.67, 1.86, 1.42; 3.91.
- 07:111, 662 four tilings 2.5-2.9x one -> wall_s ratios `h9_I14_L*_T4/T1`: 2.89, 2.72, 2.48, 2.75, 2.80.
- 07:158 0.30-0.32 over three seeds -> 0.302-0.324.
- 07:162 at observed points 0.31-0.33 -> S.blend_obs 0.333, 0.323, 0.311; half of 0.62.
- 07:164 prior 0.34-0.39 -> S.prior_blend 0.385, 0.361, 0.338.
- 07:168 calibration 0.92-0.99 -> as above.
- 07:169-171 RTPS 0.5/0.7/0.9: 0.322/0.324/0.340, calib 0.78/0.92/1.16; within 1% -> S `D10_L44_rtps0.5`, `I33_d10_T4_s0`, `D10_L44_rtps0.9`; 0.324/0.322 = +0.6%.
- 07:173-175 floor 0.24-0.25 (one), 0.13-0.14 (blend), "about half", analysis 0.30-0.33 -> S.floor1 0.239-0.254, floorK 0.128-0.142, all sizes/seeds; 0.251/0.457 = 0.55.
- 07:185-187 hybrid 0.354/0.345/0.338, calib 1.22-1.27, one tiling 0.461 vs 0.457 -> `i24_D10_I24_T4_c8.25_rtps0.5_s*`, `i24_D10_I24_T1_c8.25_rtps0.5` (calib 1.220, 1.252, 1.269; 0.4605).
- 07:193-196 skewness 0.20, kurtosis -0.07, priors 0.38 / 0.40, noise about 0.6 -> `gaussianity_L44_p7c12d10_ks2d.json` clim_z 0.201/-0.070, prior_z 0.382, prior_u 0.402; SD of sample skewness for n=16 = 0.56.
- 07:207 d_lat 768-76,800, factor 100 -> 12 x (8^2 ... 80^2).
- 07:208-209 0.324, 0.304, 0.312, 0.326, 0.311; second seeds within 0.01 -> T-sizes; 0.308 vs 0.312, 0.316 vs 0.326 (diff 0.0098).
- 07:210 calibration 0.92-0.98 -> T-sizes seed 0: 0.916-0.984.
- 07:211-212 one tiling 0.42-0.46; floor 0.13-0.14 -> T-sizes 0.423-0.457; 0.129-0.142.
- 07:220-221 members 0.450/0.352/0.330/0.324/0.322 and 0.329/0.315/0.312/0.311 -> T-mem (i), S `D10_L44_I3_N*`, `D10_L220_I3_N*`.
- 07:224-225 density 0.043, radius 16.5, about 37 -> tab02_truth 0.0424/0.0434; 2 x 1.5 sites x 5.5; pi x 16.5^2 x 0.043 = 36.8 (Eq. nens_local).
- 07:231 n_+ 42-44 at 44^2, 93-101 at 66^2 -> tabC_truth_runs.
- 07:232-233 unlocalized 44^2: 0.51/0.37 (four), 0.69/0.50 (one), 0.32 -> S `K1u_L44_T4_N{64,128}` 0.514/0.370, `K1u_L44_T1_N*` 0.693/0.498.
- 07:234-235 66^2: 2.26, 3.22, interp 0.82, 128 members 0.49 / 0.67 -> S `K1u_L66_*` 2.263, 3.221, 0.816, 0.492, 0.666.
- 07:236-237 88^2: 3.40 (four, 64), 4.07 and 2.59 (one) -> S `K1u_L88_*` 3.400, 4.069, 2.589.
- 07:239 localized 0.32, 0.31, 0.30 at 44, 66, 88 -> 0.324, 0.307, 0.304.
- 07:242-243, 525 300 cycles at 220^2: 0.31/0.31/0.31, calib 0.97 -> S `I16c_L220_300cyc_T4` per_cycle means 0.312/0.309/0.308; 0.968.
- 07:313 latent 0.31 -> three-seed mean 0.3122.
- 07:318 0.312 (0.302-0.324) -> mean of `I33_d10_T4_s*`.
- 07:346 10% coefficient est. 0.225 -> T-me; `I J16a_lap1.1_rtps0.7_s*`.
- 07:347 forcing bc 0.225 both -> T-me (`J16b_frc*`).
- 07:348 grids 0.083 and 0.181; bc no help -> T-me (best = tuned; `J16b_M32_*` 0.186/0.182/0.180).
- 07:349-350 25%: 0.48 -> 0.42, 1.35x -> T-me 0.481, 0.420; 0.4198/0.3122 = 1.345.
- 07:351-352 advection 0.44, 0.55; 1.4, 1.8x -> T-me 0.436, 0.553; 1.40, 1.77.
- 07:353 orders 5/6/8 tuned 0.283, 0.343, 0.409 -> T-me (`J17H_p5_c8.25_rtps0.6`, `J17H_p6_c16_rtps0.7`, `J17H_p8_c8.25_rtps0.6`).
- 07:354-356 est. 0.263, 0.322; wins 1.19x, tie, loses 1.31x -> T-me (`J17H_p{5,6}_c16_rtps0.7_abg_s*`).
- 07:357-358 bc on seed 0: 0.297, 0.361, 0.440 -> I `J17H_p{5,6,8}_c16_rtps0.7_bc_q0_s0`.
- 07:363-364 forcing 0.4: 0.20 total, 0.016 state-dependent -> `j16_model_error_frc0.4.json` residual_rms 0.203, sigma 0.016.
- 07:366-367 below 0.08 cases, 0.000-0.077 -> j16 sigma: M64 0.000, M32 0.036, frc 0.008/0.016, lap1.1 0.077, hyp5 0.069.
- 07:369 order 6 at 0.12; crossover near 0.1, a sixth -> hyp6 sigma 0.123; 0.1/0.617 = 0.16.
- 07:376 tau = 2, memory 4-4.5 time units, 8-9 cycles; about 20% unpredictable -> `j17b_scan_mem_e{0.1,0.2}_t2_k-1_lap.json` v_autocorr_time 4.5 / 4.0, R2 0.79 / 0.77; truth_memory "0.1,2,-1" in the run args.
- 07:385-387 latent 0.476 (0.465-0.483), 0.628 (0.620-0.641) -> S `J17B_e*_lat_T4_s*`; physics 0.442 (0.436-0.447), 0.599 (0.596-0.604) -> I `J17B_e*_pde_c8.25_rtps0.7_lap_s*`; ties 0.93 / 0.95.
- 07:388-389 true equations with v unobserved 0.405 and 0.493 -> I `J17B_e*_floor_c8.25_rtps0.7_s*` means 0.405, 0.493.
- 07:398-399 order 6 RTPS sweep 0.72/0.43/0.35/0.93/1.27 at c=16 -> I `J17H_p6_c16_rtps{0.5,0.6,0.7,0.9,1.0}_s0`.
- 07:459-461 clean 0.338/0.321/0.326/0.340/0.326; 4-5% -> S `P2_L44_I4_T4`, `P2_L88_I1_T4`, `P2_L{220,330,440}_I1_T4`; 4.2, 5.3, 4.2, 4.0, 4.7 %.
- 07:462 calibration 0.85-0.90 vs 0.92-0.98 -> clean 0.846-0.903.
- 07:463 clean seeds 0.338/0.322/0.320 -> S `P2_L44_I4_T4`, `L44_C12_st8_rtps0.7_s{1,2}`.
- 07:465 clean KY 2.1% / 1.1% -> `crop_eval_p7c12_ks2d.log` 2.063, 1.106.
- 07:468-469 floors clean 0.27-0.29 / 0.15-0.16, noisy 0.24-0.25 / 0.13-0.14 -> S floor1/floorK of the P2_* and D10/I33 runs (0.270-0.287, 0.150-0.164).
- 07:474-475 15-20% more grid-scale noise removed; 44% vs 53% -> `j18b_roundtrip.json` white ratios 0.444 / 0.531 (after-noise 15-21% lower at 0.05-0.4).
- 07:480-483 noise-0.3 model 0.343/0.332/0.328, 220^2 0.334; comparisons -> S `J18a_d30_T4_s*`, `J18a_d30_L220_T4_s0`, `P2_L220_I1_T4` 0.326.
- 07:484 KY within 1.4% -> `crop_eval_p7c12d30_ks2d.log` 1.409.
- 07:485-487 floors 0.22-0.23 / 0.12-0.13, calibration 1.06-1.10 -> S `J18a_d30_*` (0.224-0.231, 0.121-0.129; 1.059-1.104).
- 07:504-506 26% after one cycle, 15-17% at 0.4-1.7 Lyap. times -> tab07_p2_forecast ratio 0.74; 0.83, 0.85, 0.83.
- 07:506-510 half-std 1.26 (1.16-1.40), 1.10, 1.62, 0.33; std 2.39, 2.06, 2.70, 0.74 -> recomputed from `h9_I7_T4_s*` forecast (first lead with error >= 0.5 clim / clim at that lead, per seed, mean over seeds, x 0.5 x 0.165): blend 15/14/17 cycles -> 1.265; tiling0 1.10; true state 1.6225; persistence 0.33; std 2.39, 2.0625, 2.695, 0.7425.
- 07:510-511 "a third to a half" -> (1.26-1.10)/(1.62-1.10) = 0.31; (2.39-2.06)/(2.70-2.06) = 0.52.
- 07:519 from 1.33, within 10% in 8 cycles -> S `I16a_clim_T4` per_cycle.blend[0] 1.330, first <= 0.356 at cycle 8 (T-rob).
- 07:521-522 difference decays ~10x per 10 cycles, 1e-8 by cycle 100 -> |per_cycle blend(I16a_clim_T4) - (I33_d10_T4_s0)|: 1.1 -> 1.5e-2 (c10) -> 1.3e-3 (c20) -> 4.3e-7 (c50) -> 4.4e-9 (c100).
- 07:523-525 R misspecified: 0.351 and 0.338, +4-8%, calibration 0.58 and 1.40 -> S `I16b_R0.5x_T4`, `I16b_R2x_T4`; +8.3%, +4.4%.
- 07:529-531 spacing 1.4: 0.226 / 0.333 / interp 0.539 -> S `K10a_st4_T{4,1}`; `interp_baseline_L44.json` stride 4 0.538.
- 07:531-532 spacing 5.5: 0.769 / 0.990 / 3.01, factor of four -> S `K10a_st16_T{4,1}`; interp 3.008; 3.9.
- 07:533 300 cycles 0.77, 0.77, 0.79 -> T-rob, S `K10a_st16_T4_300` per-cycle means 0.769/0.767/0.788.
- 07:598 spacing 3.125 -> tabE_p2_networks.
- 07:602 skill 2.3-3.0 with localization, N >= 16 -> tab07_1d_da (2.32-2.99).
- 07:604 n_+ 26-27 at L=200 -> tabC_truth_runs; e1_groundtruth_lyapunov_L200*.
- 07:604-606 marginal at 64 / works at 128 (L=200); fails at 64, marginal 128, works 256 (L=400) -> tab07_1d_da PFF none: 1.32, 2.65; 0.76, 1.41, 2.69.
- 07:607 filters differ by at most 0.4 -> largest PFF-EnKF gap in tab07_1d_da 0.37.
- 07:617 factor up to 1.7 against projected form -> `da_pff_ltransfer_d1_pff_L200_gaspari_cohn_nens16` / `e3_sec_vs_gc_ltransfer_d3_L200_pff_sec_psd_N16` = 1.70.
- 07:617-618 up to 3.5 against serial; serial useless below 128 -> EnKF-GC / serial SEC 3.52 at N=32; serial skill 0.72-0.94 for N <= 64.
- 07:655-657 9 s, 1266 s, 26 s, 3542 s, 35 s per cycle -> `h9_I14_L{44,440}_T{1,4}` wall_s; 3542/100.
- 07:659 slope 1.05 both -> log-log LSQ of wall_s over 5 sizes: 1.06 (K=1), 1.05 (K=4).
- 07:660 slope 1.3 above 220^2 -> 220-440: 1.27 (K=1), 1.36 (K=4).
- 07:665 0.7 GB, 14.6, 17.2 GB -> peak_rss_GB 0.717, 14.63, 17.24.
- 07:667 truth 6.6 GB -> 1280^2 x 501 x 8 B = 6.57 GB (T-cost).
- 07:668 47 GB -> 76,800^2 x 8 B = 47.2 GB (decimal).
- 07:671-672 2.4 ms, 119 ms -> `jacobian_timing_F4_p7c12d10_ks2d.json` t_coloured_s 0.00239, 0.1189.
- Tables VII-XI and Table tab:da_cost cells -> regenerated, identical (see top).

main.tex, Sec. 1, Sec. 9, Sec. 10
- main:66-68 crossover about a sixth of the observation error -> as 07:369.
- main:81 "a hundred times larger" -> 44^2 to 440^2.
- 01:125, 10:34-35 about 30% lower with tilings -> 27.7-29.1 %.
- 01:128-129 0.30-0.33, 0.82 -> T-sizes 0.302-0.326; interp 0.808-0.829.
- 01:131-137 loses when small or correctable, wins when large or structural, a sixth -> T-me.
- 09:91-92 blended floor less than half of the analysis error -> 0.137/0.324 = 0.42; max over sizes 0.44.
- 09:96-98 correct equations four times more accurate -> 0.312/0.076 = 4.1.
- 09:109-110 noisy-trained slightly better -> 07:459-463.

App. E
- E:11 two obs per site in 1D, four in 2D -> tabE_p2_networks (64 obs / 32 sites; 256 / 64).
- E:13 R = 0.01 I, 0.62 -> as 07:36.
- E:36 GC half-width 1.5 sites, zero at 3 -> args gc_c 1.5.
- E:41-43 RTPS 0.7, 5 steps, noise 0.02, IC 0.1 -> h9 args rtps, n_prop_steps, model_noise_std, ic_perturb_std.
- E:48 offsets {0,8}^2, 16-point patch -> tabE_p2_networks offsets.
- E:125-127 64 members, smooth IC perturbation 0.5 over 4 points, c=24 no RTPS gives 0.076 -> i11 args ic_perturb_std 0.5, ic_perturb_len 4.0; I `L44_st8_c24_rtps0`.
- E:133 most unstable wavenumber sqrt(a/2) -> growth rate aK^2 - K^4 (analytic).
- E:144 fitted (a,b,g) (0.640, 0.634, 0.996), (0.436, 0.426, 0.980), (0.210, 0.203, 0.938) -> run args of `i11_J17H_p{5,6,8}_c16_rtps0.7_abg_s0` (model_lap_coef, model_hyper_coef, model_nl_coef).
- E:146 one-cycle errors 0.072, 0.129, 0.210 -> `j16_model_error_hyp{5,6,8}.json` residual_rms (T-me).
- E:147-148 tau=2, kappa=-1, eps 0.1/0.2 -> truth_memory "0.1,2,-1".
- E:154-155 tuning grid includes c=4 for advection, inflation 1.1 / 1.2 -> files `i11_J1b_adv*_c4_*`, `i11_J2_*_inf{1.1,1.2}`.
- E:177 max difference 2e-7 up to 132^2 -> max_abs_diff 2.38e-7.
- E:178-179 grows more slowly than d; 2.4 ms / 119 ms -> log-log slope 0.85.
- E:180 reverse mode near square, 32x slower at 132^2 -> slope 1.84 (768-6912); 0.374/0.0117 = 32.1.
- E:202-203 obs density skill 5.2, 2.9, 1.2, 1.0 at 1.6, 3.1, 6.25, 12.5 -> `da_pff_ltransfer_d2_pff_L200_gc_N64_sp{1.5625,6.25,12.5}` 5.19, 1.23, 1.04; d1 pff L200 GC N64 2.86.
- E:206 earlier localized skill about 1.3 -> `da_pff_ltransfer_e2_L200_gaspari_cohn_nens64` 1.34 (stride 16).
- E:212 taper non-zero on 11 of 16 sites -> half-width 3 sites: |d| <= 5.
- E:214 matrix-form SEC indefinite at five of six sizes -> `e3_sec_vs_gc_sweep_L200.csv` FAILED rows (robust_cholesky indefinite) and decisions.md; only N=256 matrix-form JSON exists.
- E:240-244 round-trip ratios (white 0.95/0.53/0.35/0.29 and 0.81/0.44/0.29/0.23; smooth 1.35/1.09/1.02/0.99 and 1.26/1.07/1.01/0.99; latent 1.00-1.05; recon 0.046 / 0.039) -> `j18b_roundtrip.json`.
- E:250-251 three seeds 0.302-0.324; second seeds within 0.01 -> as 07:208.
- E:254 10-cycle means vary +-6% (44^2), +-2% (88^2), <= +-1% (220^2+) -> per_cycle blend, max |dev| 5.9%, 2.3%, 0.9%.

Skipped or light: App. E l.45 (climatology ensemble construction), l.137 (forcing correlation length), l.158/163
(parameter-estimation spreads, settings only), l.107-111 (SEC table construction settings), and "forty true states" (l.237, not in the JSON).
