# Part 4.3 implementation steps: latent-space localization for DA

Companion to `docs/LITERATURE_REVIEW_AND_FINDINGS.md` Part 4.3. That
document states the claim, the novelty check (three papers read in full,
all ruled out), the broader implication, and a high-level Stage 0-4
outline. This document breaks that outline into small, individually
checkable steps, each with a concrete pass/fail test, meant to be worked
through one at a time -- check a box, run its test, move to the next.

**Pre-registered decision rule (already stated in Part 4.3, restated
here so it's next to the steps that produce the number it judges):**
local-field + Gaspari-Cohn must beat the SEC baseline (Phase B/C below),
not merely beat no-localization. SEC is a real, competitive method
(Anderson 2012), not a strawman.

## Status snapshot (2026-09-24, keep this updated as steps complete)

- [x] **Stage 0 infrastructure is built and tested.**
  `ks_latent/da/localization.py` (`gaspari_cohn_taper`,
  `build_latent_taper_matrix`, 13 unit tests), wired into
  `scripts/run_da_pff.py` via `--localizer {none,sec,gaspari_cohn}`.
  Reuses `ks_latent/da/pff.py`'s pre-existing (previously unwired)
  `localize_fn` hook and `ks_latent/da/sec.py`'s Schur-product mechanism.
- [x] **A validated GLOBAL-latent KS checkpoint exists**: Section 213
  (`--encoder vit`, `d_latent=44`, Section 52's regularizer recipe).
  Stage 2 (`warmstart_k12_300ep`): `D_KY=21.55`, `lambda1=0.086`,
  `n_positive=11/44`, standalone 2000-step rollout genuinely bounded
  (`max|z|` oscillates 2.4-3.4, never diverges).
- [x] **NEW LEADING CANDIDATE: Section 224 (`local_mlp`, d_latent=48) -- beats 223 on every axis while using HALF the latent dimensions, still satisfies both Phase A and Phase F's L-transfer requirement.** Ten
  candidates measured:
  | tag | Stage-1 propagator/regularizers | Stage-2 `D_KY` | bounded? | D3 bandedness (Stage 2) |
  |---|---|---|---|---|
  | 211 | global `mlp`, Section 52 recipe only | 23.04 | yes | 0.1883 |
  | 215 | global `mlp` + `--w-jacobian-bandedness` alone | -- (Stage 2 killed, Stage-1-only divergence got WORSE: `D_KY=90.64`) | no (Stage 1) | 0.9938 (Stage 1, not Stage 2) |
  | 216 | global `mlp` + `--w-jacobian-bandedness` + `--w-jacobian-diagonal-bound` | 22.59 | yes | 0.2146 |
  | 217 | LOCAL `masked_mlp`, no D3 loss | -- (Stage 2 could not rescue it: NaN by step ~350) | no | 0.2970 (Stage 2, but on a diverging checkpoint -- unusable) |
  | 218 | LOCAL `masked_mlp` + `--w-prop-magnitude-ceiling` + `--multistep` | -- (bounded through t=300, but still diverges to ~2e11 by t=1999) | no | 0.3661, p=0.0000 (Stage 2, still on a diverging checkpoint -- unusable) |
  | 219 | LOCAL `masked_mlp` + `--w-multistep-growth-ceiling` + narrowed `attn_window=9` + `--multistep` | -- (Stage 2 NaN by step 254 -- FASTER than 218) | no | 0.4052, p=0.0000 (Stage 2, highest yet, still on a diverging checkpoint -- unusable) |
  | 220 | LOCAL `masked_mlp` + `--w-multistep-growth-barrier` (log-barrier, Stage 1 only), plain k=2 | -- (Stage-2-without-barrier stalled/diverged into the thousands at k=12) | no | n/a (never reached a usable Stage 2) |
  | 221 | LOCAL `masked_mlp_expand` (3-layer expand/wide/contract variant), site radius 4 (`attn_window=4`), `--w-multistep-growth-barrier` Stage 1 only, NO barrier in Stage 2 | 22.68 | yes -- `max\|z\|` in `[8.8,9.6]` across the full 2000-step rollout | 0.3916, p=0.0000 |
  | 222 | LOCAL `site_conv` (NEW backbone, reuses `KSAutoencoderLocalField`'s own circular-conv design as a propagator), clean baseline recipe | -- (bounded to t=600, then diverges to 1.8e31 by t=1000) | no | 0.2840, p=0.0000 (Stage 2, on a checkpoint that ultimately diverges) |
  | 223 | LOCAL `local_mlp` (pre-existing backbone, never previously validated), `d_latent=96`, tokens aligned to sites (`n_tokens=32`), effective receptive field 6 sites, clean baseline recipe | 22.66 | yes -- `max\|z\|` in `[7.0,7.9]` across the FULL 2000-step rollout, Stage 1 ALONE also bounded | 0.3091, p=0.0000 |
  | **224** | LOCAL `local_mlp`, `d_latent=48` (n_sites=16, local_channels=3 -- HALF of 223's), otherwise identical recipe | **22.14** | **yes -- `max\|z\|` in `[4.0,4.7]` across the FULL 2000-step rollout, tighter than 223, Stage 1 ALONE also bounded even more tightly** | **0.4313, p=0.0000 -- highest of any validated checkpoint** |
  **224 (`local_mlp`, `d_latent=48`) is the new leading candidate: beats
  223 on EVERY metric** (D3 0.431 vs 0.309, `best_val_kmax_mse` 0.016
  vs 0.042, tighter standalone bounds) **while using half the raw
  latent dimensions** -- `n_positive` stayed almost identical (12/48 vs
  13/96, i.e. nearly the same effective chaotic dimensionality, just
  packed more efficiently). Suggests 223's own `d_latent=96` carried
  more redundant coordinate budget than the dynamics actually needed.
  Still verified L-transferable in principle (same `local_mlp`
  architecture as 223, whose parameter-count independence from
  `n_sites` was already confirmed) though not re-verified at this
  specific size. Not yet known how far the "smaller is better" trend
  continues -- worth testing progressively smaller `d_latent` to find
  where quality actually starts to degrade. 222 (`site_conv`) remains a
  real second-tier alternate (excellent Stage-2 convergence, 600 clean
  standalone steps, but ultimately diverged by t~1000).
  **Action before treating 224 as truly frozen: run the full Gate 3/4
  diagnostic suite (D1-D8, not just the lighter D3-only check used
  during this search) on it, same as was done for 211, AND actually run
  the F2/F3 L-transfer tests below** (structural eligibility is
  necessary but not sufficient -- nothing has yet been run at a
  different `L` for any candidate).
- [x] **D3/D7 measured on Section 211 (Gate 4, full D1-D8 report):**
  `docs/diagnostics_report_section211_..._warmstart_k12_300ep.md` --
  D3=`0.1883` (p=0.0000), D7=`0.2415` (p=0.0000), D8=`0.2694`
  (p=0.0000), all significant. Gate 4 not yet run on Section 216 (only
  the lighter `coupling_graph_diagnostic`-only D3 check was) -- run the
  full Gate 4 on whichever checkpoint is finally frozen.

---

## Phase A: pick and freeze the two checkpoints this whole proposal rests on

- [x] **A1. Record final checkpoint comparison and freeze `LOCAL_AE`/
  `LOCAL_PROP`.** Superseded across the full 10-candidate table in the
  status snapshot above (211 through 224). **FROZEN (2026-09-25):
  `LOCAL_AE`/`LOCAL_PROP` = Section 224** (`local_field` encoder,
  `n_sites=16, local_channels=3, d_latent=48` + `local_mlp` propagator),
  superseding Section 216 -- confirmed bounded 2000-step standalone
  rollout (`max|z|` `4.0-4.7`, the tightest of any candidate),
  `D_KY=22.14` in `[21,24]`, D3=`0.4313` (highest of any validated
  checkpoint this investigation). Also serves as `TRANSFER_PROP` for
  Phase F (verified structurally L-transferable). Plots (`error_growth`/
  `latent_hovmoller`/`physical_hovmoller`/GIF) confirm healthy chaotic
  saturation at `sqrt(2)` relative RMSE, not divergence or collapse --
  see `docs/RESULTS.md`.
- [x] **A2. Freeze `GLOBAL_AE`/`GLOBAL_PROP` = Section 213.** Already
  validated (see status snapshot). No further action, just naming it
  so later steps can refer to it consistently.
- [x] **A3. Confirm `LOCAL_AE` is genuinely a `KSAutoencoderLocalField`
  instance** (not a fallback/mismatched load). **Test:** `isinstance(ae,
  KSAutoencoderLocalField)` -- this is exactly the guard
  `scripts/run_da_pff.py --localizer gaspari_cohn` already raises on if
  false, so this step is really "run the guard once on purpose and
  confirm it does NOT raise."

## Phase B: make the DA pipeline itself trustworthy before sweeping anything

Motivation: the one real DA run attempted so far this session (against
Section 140, since abandoned) showed skill barely above 1.0x and
calibration ~0.02 -- far below the brief's own `~10x`/`~0.8` targets.
Before trusting ANY `N_ens` sweep number, confirm the DA pipeline itself
behaves sanely on the validated checkpoints. This phase has no direct
counterpart in Part 4.3's Stage 0-4 list -- it is a prerequisite the
original outline assumed away.

- [x] **B1. Baseline DA sanity check on `GLOBAL_AE`/`GLOBAL_PROP`,
  `--localizer none`.** Swept `--n-prop-steps` in `{1,3,5,8,12}` at
  `--n-ensemble 32`.
  **Result: `n_prop_steps=5` chosen.** `calibration_spread_over_rmse
  =0.42` (in `[0.4,1.5]`), `skill_free_over_da=2.73` (best of all five,
  clears the `>2` bar). Higher `n_prop_steps` (8, 12) does not help
  further -- `rmse_free` saturates near the attractor's own natural
  scale (~1.2-1.26) as the free run loses all skill, so skill only
  degrades (1.49, 1.38) while calibration doesn't improve. **Test
  PASSED.**
- [x] **B2. Repeat B1's sweep on `LOCAL_AE`/`LOCAL_PROP`.**
  **Originally run against Section 216 (`d_latent=96`)**: needed a wide
  sweep (`n_prop_steps` in `{1,3,5,6,8}` x `n_ensemble` in
  `{32,64,128,256}`) and only ever reached `n_ensemble=128,
  n_prop_steps=6` -> `skill=1.64` (never cleared the `>2` bar), with
  real instability at other ensemble sizes -- see `docs/RESULTS.md`'s
  original Phase B writeup for the full detail (kept for the historical
  record, no longer the operative checkpoint).
  **RE-RUN 2026-09-25 against Section 224 (`d_latent=48`, the new
  frozen `LOCAL_AE`/`LOCAL_PROP`)**: `n_ensemble=32` (matching the
  global checkpoint's own setting) already gave real skill (1.24-1.71
  across `n_prop_steps` in `{1,3,5,6}`, a big improvement over 216's
  near-zero/negative skill at the same ensemble size). Widening to
  `n_ensemble=64`: `n_prop_steps=3` gives `calibration=0.467` (in
  `[0.4,1.5]`) and `skill=3.27` -- **clears both bars, and beats the
  GLOBAL checkpoint's own best result (2.73).** `n_ensemble=128,
  n_prop_steps=3` also clears both bars (`calibration=0.56`,
  `skill=2.97`). **Test PASSED, decisively, unlike the original 216
  run** -- the smaller/cleaner `d_latent=48` checkpoint DAs better even
  WITHOUT localization, not just looking better standalone. Chosen:
  `n_ensemble=64, n_prop_steps=3`.
- [x] **B3. Record the chosen `n_prop_steps` (and any other DA
  hyperparameter tuned in B1/B2) for each checkpoint.**
  `GLOBAL_AE`/`GLOBAL_PROP`: `n_ensemble=32, n_prop_steps=5`.
  `LOCAL_AE`/`LOCAL_PROP` (Section 224): `n_ensemble=64, n_prop_steps=3`.
  These are now fixed for Phase C/D's sweeps (`n_ensemble` itself still
  varies across the `{8,...,256}` sweep set in those phases -- what's
  fixed here is `n_prop_steps` and the general sanity that the pipeline
  produces a real, bounded DA effect at each checkpoint's own natural
  operating scale).

## Phase C: Phase 7 baseline -- SEC sweep on the GLOBAL latent

- [x] **C1. Build a thin sweep driver.** DONE --
  `scripts/run_da_ensemble_sweep.py`: loops `run_da_pff.py` over
  `--n-ensemble` x `--localizer`, collects results into one CSV.
  Records a `FAILED` row (with the error) rather than aborting the
  whole sweep when one combination crashes (found for real: SEC at
  very small `N_ens` relative to `d_latent` can produce a genuinely
  singular ensemble covariance -- an informative result, not a harness
  bug). Smoke-tested end to end before the real run.
- [x] **C2. Run the real sweep.** DONE -- `N_ens in {8,16,32,64,128,256}`,
  `n_prop_steps=5` (B3's chosen value), `n_cycles=40`, `GLOBAL_AE`/
  `GLOBAL_PROP` (Section 213). Result in `artifacts/da_sweep_
  phaseC2_global.csv` and `docs/figures/phase4_3_decision_rule_sweep.png`.
  **SEC clearly helps at small `N_ens`** (`rmse_da` `2.550->0.506` at
  `N=8`, `1.663->0.422` at `N=16`) and stays at or below no-localization
  throughout the sweep -- confirms the literature's own claimed
  benefit on this checkpoint. See `docs/RESULTS.md`'s Phase C/D/E entry
  for the full table.

## Phase D: Phase 13 core -- Gaspari-Cohn sweep on the LOCAL field

- [x] **D1. One-cycle integration test before any real sweep.** DONE --
  checked directly against Section 224's own real forecast covariance:
  at every `gc_c` tested, far-apart site pairs (beyond support radius
  `2*gc_c`) are EXACTLY zero after tapering, near pairs retain real
  covariance structure. Test PASSED.
- [x] **D2. Sweep `--gc-c` alone.** DONE -- swept `{1.0,1.5,2.0,3.0,4.0,5.0}`
  at `n_ensemble=64`. Cross-checked against Phase 2's own measured
  light-cone bound: at `dt_snap=1.0`, `minimum_localization_radius=6`
  sites (encoder/propagator receptive field dominates; light-cone term
  negligible), requiring `gc_c>=3.0` -- matched the empirical result
  cleanly (`gc_c>=3.0` clearly beat no-localization; `gc_c=1.0`, below
  the floor, did not). Chose `gc_c=3.0`. Test PASSED.
- [x] **D3. Run the full `N_ens` sweep.** DONE -- same `N_ens` set and
  `n_prop_steps=5` as Phase C, `gc_c=3.0`, Section 224. Result in
  `artifacts/da_sweep_phaseD3_local224.csv` and the same plot as C2.
  **Gaspari-Cohn dramatically beats no-localization on the local
  latent** (`rmse_da` `1.649->0.530` at `N=8`, `0.772->0.421` at
  `N=32`) but stays 5-25% above SEC's own curve throughout -- see
  Phase E below for the decision-rule verdict this produces.

## Phase E: the actual decision

- [x] **E1. Apply the pre-registered rule mechanically.** DONE --
  `scripts/check_4_3_decision_rule.py`, 8 unit tests (all-pass,
  all-fail, mixed, exact-tie, failed-run-skipped, CSV round-trip) pass
  before being trusted on real data. Run against the real Phase C2/D3
  output: **0 PASS, 6 FAIL** at every `N_ens in {8,16,32,64,128,256}`.
- [x] **E2. Write the verdict into `docs/RESULTS.md`.** DONE -- **the
  verdict is negative**: local-field + Gaspari-Cohn (Section 224,
  `gc_c=3.0`) does NOT beat SEC on the global latent (Section 213) at
  any tested ensemble size, though it dramatically beats no-localization
  on the local latent itself, confirming the mechanism works, just not
  enough to catch a strong SEC baseline. Full table, plot
  (`docs/figures/phase4_3_decision_rule_sweep.png`), and discussion of
  what remains untried (different `n_prop_steps`, finer `gc_c` sweep,
  `N_ens`-dependent `gc_c`) in `docs/RESULTS.md`'s Phase C/D/E entry.
  Per Part 4.3's own pre-registered framing, this negative result is
  still informative and publishable -- it leaves the L-transfer
  property (Section 224/`local_mlp`, verified structurally, Phase F
  below) as the sole surviving argument for the local-field approach.

## Phase F: L-transfer -- the headline result, and a real blocking gap

**Read this whole phase before starting it.** `LOCAL_PROP` (Section
211/215) uses a fully-connected GLOBAL `mlp` propagator backbone.
That architecture's weight matrices are tied to a FIXED `d_latent` --
if the domain length `L` changes, `n_sites` (and therefore `d_latent =
n_sites * local_channels`) changes too, and a fixed-shape `mlp` simply
cannot run on a different input/output dimension at all. **The
L-transfer claim structurally requires a translation-equivariant,
weight-shared propagator** (`backbone` in `{local_mlp, cnn, node,
site_conv}` -- anything whose parameter count doesn't depend on
`n_sites`, VERIFIED for `local_mlp`/`site_conv`, not yet checked for
`cnn`/`node` -- `masked_mlp`/`masked_mlp_expand` do NOT qualify despite
being "local," see below), not the propagator Phase A originally
validated (Section 216, later superseded by Section 223 -- see status
snapshot above). **Plain
`masked_mlp` was tried and ruled out for good (Sections 217/218/219,
2026-09-24)**: Section 217's standalone divergence was so severe it
crashed the Lyapunov computation outright (`max|z|` reached `1e30`),
and Stage 2 could NOT rescue it (NaN by step ~350). Section 218 then
tried a propagator growth ceiling during Stage 1 plus a longer Stage-1
rollout -- Stage 2 held bounded noticeably longer (through t=300 vs.
217's ~350-step collapse to NaN) and reached a higher *significant* D3
score (0.3661 vs 0.2970), but still diverged to `~2e11` by t=1999.
Section 219 then tried a ceiling on the composed 10-step Jacobian's
top singular value plus a narrowed `attn_window` (9, down from 18) --
highest D3 yet at that point (0.4052) but Stage 2 diverged even
FASTER (NaN by step 254). Section 220 tried a safeguarded log-barrier
penalty instead of a squared hinge (Stage 1 only) -- calmest Stage-1
trajectory of the whole arc, but Stage 2 (with no barrier) stalled and
then diverged into the thousands once `k` reached its full schedule.

**Section 221 (2026-09-24) FINALLY produced a bounded, validated
`masked_mlp`-family checkpoint, using `masked_mlp_expand`** (the
"other" 3-layer expand/stay-wide/contract masked-mlp variant, NOT
plain `masked_mlp`) at site radius 4 (`--attn-window 4` -- verified
empirically to be a clean 1:1 site-radius mapping for this backbone,
unlike plain `masked_mlp`'s `window = 3*site_radius`), with Section
220's Stage-1-only log-barrier growth control and no D3 loss. Stage 2
converged to a checkpoint with `max|z|` bounded in `[8.8,9.6]` across
the FULL 2000-step standalone rollout, `D_KY=22.68` (in target range),
and D3=`0.3916` (p=0.0000) -- the highest of any validated checkpoint
at that point, beating even Section 216. **This is a strong Phase-A
candidate but does NOT satisfy this phase's own L-transfer
requirement** -- checked and corrected the same day: `masked_mlp_expand`'s
layers are masked DENSE matrices whose parameter shape is tied to
`d_latent` (verified: total params quadruple when `n_sites` doubles),
not a real weight-shared convolution, so a checkpoint trained at one
`L` cannot even be loaded at another.

**Section 222/223 (2026-09-24, same day) then solved F1 outright.**
User: "could we use a model like the encoder from 221 as a propagator?"
-> built `backbone="site_conv"` (reuses `KSAutoencoderLocalField`'s own
circular-conv site-mixing design directly as a propagator). Section
222 (clean baseline recipe) converged beautifully in Stage 2
(`val_kmax_mse=0.076`) and held bounded for 600 standalone steps before
diverging -- close, not quite there. User: "can we try the setup from
222 with the local mlp too? just for fun" -> Section 223
(`backbone="local_mlp"`, pre-existing but never previously validated,
tokens aligned to sites) **succeeded completely**: bounded the FULL
2000-step rollout (Stage 1 alone AND Stage 2), `D_KY=22.66` (in
range), D3=`0.3091` (beats 216), AND verified structurally
L-transferable (parameter count IDENTICAL -- 31,651=31,651 -- at
`n_sites=32` vs. `64`). Then, after discussing why `d_latent=96` isn't
trying to match the global design's `44` (a local dof-density argument,
not a global-attractor-dimension one) and being asked whether
`d_latent=50` might work, user said "run 224 with d_latent = 48"
(the nearest clean value). **Section 224 (same `local_mlp` recipe,
`n_sites=16, local_channels=3 = 48`) beat 223 on EVERY metric while
using HALF the latent dimensions**: D3=`0.4313` (highest of any
validated checkpoint this investigation), `best_val_kmax_mse=0.016`
(best of any local-field checkpoint), tighter standalone bounds
(`max|z|` `4.0-4.7` vs 223's `7.0-9.1`), `D_KY=22.14` (still in range).
**`local_mlp` at `d_latent=48` (Section 224) is `TRANSFER_PROP`,
pending the F2/F3 tests below and full Gate 3/4 verification.**

- [x] **F1. Train and validate a size-transferable local propagator.**
  DONE -- Section 224 (`backbone="local_mlp"`, `d_latent=48`, tokens
  aligned to `n_sites=16`, effective receptive field 6 sites): Stage-2
  standalone 2000-step rollout bounded (`max|z|` `4.0-4.7`, the
  tightest of any candidate), `D_KY=22.14` (in `[21,24]`), verified
  L-transferable (same architecture family as 223, whose parameter-
  count independence from `n_sites` was directly confirmed). Call this
  `TRANSFER_PROP`. Section 223 (`d_latent=96`, same backbone) and
  `site_conv` (Section 222) remain as alternates if `TRANSFER_PROP`
  doesn't hold up under F2/F3 or full Gate 3/4 scrutiny.
- [x] **F2. Architecture-level transfer test, no DA yet.** DONE --
  built `load_autoencoder_checkpoint_resized`/`load_propagator_
  checkpoint_resized` (`ks_latent/models/__init__.py`, 8 unit tests)
  and `scripts/run_ltransfer_test.py`. Ran `LOCAL_AE`/`TRANSFER_PROP`
  (Section 224) at `L=200` (2x) AND `L=400` (4x), zero retraining.
  **Both tests PASSED**: energy spectrum peaks near `k~1/sqrt(2)` at
  both sizes (L=200: exact match, true=model=0.660; L=400: ~2% apart),
  AND `D_KY/L` extensivity holds tightly (`L=100`: 0.2214, `L=200`:
  0.2231, `L=400`: 0.2228 -- all close to the true KS constant ~0.226).
  Full numbers in `docs/RESULTS.md`'s Phase F2 entry.
- [x] **F3. Full localized-DA-at-larger-L test.** DONE --
  `scripts/run_da_pff_ltransfer.py` (reuses `run_da_pff.py`'s own DA
  cycling machinery with the resized loaders). At `L=200`, zero
  retraining, `n_ensemble=64` UNCHANGED from the `L=100` operating
  point: **unlocalized DA actively breaks** (`skill=0.69`, worse than
  free-running -- the raw latent dimension doubled along with `L`) --
  **Gaspari-Cohn (same `gc_c=3.0` as `L=100`, still valid since `h` is
  held fixed by the transfer) rescues it to real positive skill=1.34**,
  more than doubling the RMSE improvement. This is Part 4.3's central
  ensemble-size-scaling claim demonstrated directly, not just argued.
  Also confirmed directly (not just asserted): feeding correctly-
  resolved `L=200` data into `GLOBAL_AE`'s fixed-`d_latent=44` encoder
  raises an immediate shape mismatch -- structurally cannot even
  attempt this test. Found (not yet fixed) a real gap along the way:
  `run_da_pff.py`'s own `--L` flag silently generates wrong-resolution
  ground truth rather than raising, if used directly without also
  reasoning about `NX` -- see `docs/RESULTS.md`'s Phase F3 entry.
- [x] **F4. Write up F1-F3 in `docs/RESULTS.md`.** DONE -- see the
  Section 224/Phase F2/Phase F3 entries; explicit throughout about
  which comparators could and couldn't even attempt the transfer test.

---

## Notes on scope and ordering

- Phases are meant to be done roughly in order (A -> B -> C -> D -> E ->
  F), but B1/B2, and C/D's own internal steps, can run concurrently on
  MPS if compute allows (per this session's own established practice) --
  ordering matters for what depends on what, not necessarily for wall-
  clock scheduling.
- Every phase that produces a headline number (C2, D3, F2, F3) has an
  explicit boundedness/sanity check attached, not just the summary
  metric -- this project has now been burned twice (Section 140, the
  Sept-9 local_field arc broadly) by trusting a summary number without
  checking the underlying standalone trajectory. Do not skip those
  checks to save time.
- If Phase D2 finds NO `gc_c` beats no-localization, or Phase E's rule
  fails at every `N_ens`, that is a valid, documentable stopping point
  (Part 4.3's own text: SEC winning is still informative, and the
  L-transfer argument alone may be worth writing up even then) -- this
  is not a plan that only has one acceptable outcome.
