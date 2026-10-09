# Review checklist (task G3, 2026-10-08)

Every item of `docs/paper1_review.md` §4.2, §4.5, §9 and §10, checked against the current text.
✓ = already done (task named), **fixed** = changed in this pass, n.a. = no longer applies.

## §4.2 Claims
| item | status |
|---|---|
| (a) 1D DA observation density falls with L | ✓ D1/D2: fixed spacing 3.125, §7.3 + Fig. 8 + `tab:da_1d`; earlier doubling-spacing result stated as such |
| (b) "SEC cannot be carried to the new size" | ✓ D3: indefiniteness is a property of the matrix form; PSD and serial forms run; GC beats both (§7.3) |
| (c) "<1 GB", "killed by the OS" | ✓ gone; measured filter-only memory, linear in d (§7.4, `tab:da_cost`, Fig. 9e) |
| (d) light-cone bound | ✓ B1: §3.3 rebuilt on the interaction range; Eq. loc_bound removed; §9 design rule |
| (e) 4.6% rise between 44² and 66² | ✓ B8: densities consistent with extensive within truth spread |
| (f) L = 25 "weakly chaotic" | ✓ A1/B7: "at most marginally chaotic", outside the model's validity |
| (g) "costs nothing" | ✓ "at least as accurate" |
| (h) C8/C16 scope | ✓ §6.5 names the model (window-trained c = 3) and drag |
| (i) "within 2%" (outline) | ✓ not in the paper; outline marked superseded |
| (j) "K always exceeds" | **fixed**: App. C "is chosen to exceed" |

## §4.5 Code vs text
| item | status |
|---|---|
| model tangents float32 / QR float64 | ✓ B4 + A2 (float64 control, App. C) |
| 1D evaluations encoded raw field | ✓ A1 (standardized re-evaluation) |
| `w_rank` on the α = 0.1 variant | ✓ Table VIII footnote †; v2 (selected) has none |
| 1D signed spatial-coherence regularizer | ✓ App. A footnote a, §5 |
| halo test is R_prop − 2, text says R_prop − 1 | **fixed**: test parametrized over halo = R−1 and R−2 (`test_forced_halo_too_small_breaks_exactness`, 27 crop tests pass); §3.4 and App. B now say both |
| localization-radius assertion | ✓ n.a. (Eq. removed; design rule in §3.3/§9) |
| per-site filter name | ✓ §3.7, §7.1 (local stochastic EnKF; ETKF variant named separately) |
| RMSE definition, physical RMSE | ✓ §7.1 metrics; D6 physical RMSE (§7.2, 44²/88²; larger sizes pending) |
| Re convention | ✓ B6 (own definition, coincides with Vinograd et al. at α = 0) |
| channel-0 anchor in canonical 2D | ✓ unconditional in both AE classes; App. A "5+1"; unit test `test_gauge_anchor_is_exact_physical_patch_average` |
| Prop. 1 at L = 25 | ✓ remark + proviso in the statement |
| y-alignment (outline) | ✓ outline marked superseded |

## §9 Concrete edits
| file | item | status |
|---|---|---|
| 03 | §3.2 Remark wording | ✓ B2 |
| 03 | §3.3 interaction range | ✓ B1 |
| 03 | Lemma 3 text + noise-as-Jacobian penalty | ✓ B3 (`rem:noise_penalty`) |
| 03 | §3.6 precision, standardized ICs | ✓ B4 |
| 03 | §3.7 filter name, RMSE/skill definitions | ✓ B4/E2 (definitions in §7.1) |
| 03 | n₊ + 1 bound with numbers | ✓ §7.1 (truth n₊ for 2D and 1D) |
| 03 | canonical recipe: w_rank, 1D spatial term | ✓ B5 / App. A / Table VIII |
| 03 | Prop. 4(b) statement matches proof | **fixed**: statement names R-localization and both updates; proof covers the transform update |
| 04 | Re convention, Lyapunov-time counts, 6–16 units | ✓ B6 |
| 04 | 2D KS single-run caveat | ✓ B6/B14 (now 3 and 2 runs; App. C subsection) |
| 04 | Edson et al. for L = 22 | ✓ |
| 05 | L = 25 rewrite, spatial regularizer, table convention | ✓ B7 |
| 06 | 93–100%, "costs nothing", name k_xs005_noreg, 4.6%, P2 scope, Table IV caption, selection rule | ✓ B8, A10 |
| B | Prop. 1 proviso; η in the failure test; Bishop remark; Prop. 4(b) | ✓ B2/B3; **fixed** η and Prop. 4(b) |
| C | 2D KS uncertainty; model Benettin settings; "published values" | ✓ B14; **fixed** "earlier values" |
| refs | verification list | ✓ B13 |

## §10 Minor
| item | status |
|---|---|
| commented inputs, missing App. A, dangling refs | ✓ all sections and App. A–F in; no undefined references |
| Table II density units | ✓ caption gives per length / per area / per box |
| Table IV header | **fixed**: "first size (native), second size, third size" |
| Fig. 3 annotation / y-range | ✓ C3 |
| Fig. 5 log floor | ✓ C5 (linear, broken axis) |
| Fig. 6 single-run truths | ✓ C6 (±15% band) |
| Fig. 8 spacing, "SEC failed" markers | ✓ F1 (fixed spacing; crosses at the failed sizes, labelled) |
| Fig. 9 placeholder panels | ✓ F2 (new six-panel figure; finalize after 330² runs) |
| notation: c overloaded for the GC parameter | n.a. (the taper is given as "half-width 3 sites" in the text; no symbol) |
| "Kolm." abbreviation, "2D KS" naming | kept for tables (space); G4 writing pass |
| reason for no bias in the output projection | deferred to G4 (current sentence is correct, not fully explained) |
| outline stale numbers | **fixed**: `paper1_outline.md` marked superseded |
| `paper1_tasks.md` memory-spike note | **fixed**: marked superseded with the explanation |
| record `tab:da_transfer` n_prop_steps label | n.a. to the paper (1D DA re-run in D1) |
| empty lead paragraph / abstract | ✓ E8 |
