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
- [x] **A validated LOCAL-FIELD KS checkpoint exists -- three candidates
  measured, `TENTATIVE PICK: Section 216`.**
  | tag | Stage-1 regularizers | Stage-2 `D_KY` | bounded? | D3 bandedness (Stage 2) |
  |---|---|---|---|---|
  | 211 | Section 52 recipe only | 23.04 | yes | 0.1883 |
  | 215 | + `--w-jacobian-bandedness` alone | -- (Stage 2 killed, Stage-1-only divergence got WORSE: `D_KY=90.64`) | no (Stage 1) | 0.9938 (Stage 1, not Stage 2) |
  | **216** | + `--w-jacobian-bandedness` + `--w-jacobian-diagonal-bound` | **22.59** | **yes** | **0.2146** |
  216 matches 211's chaos quality with measurably better D3 bandedness,
  entirely from Stage-1-only regularization (neither D3 loss was active
  during Stage 2). Before fully freezing this pick: Section 216's own
  Stage 2 never had the D3 losses active either -- try keeping
  `--w-jacobian-bandedness`/`--w-jacobian-diagonal-bound` on THROUGH
  Stage 2 too (cheap, already wired) to see if bandedness pushes higher
  still without losing bounded chaos, before calling this final.
- [x] **D3/D7 measured on Section 211 (Gate 4, full D1-D8 report):**
  `docs/diagnostics_report_section211_..._warmstart_k12_300ep.md` --
  D3=`0.1883` (p=0.0000), D7=`0.2415` (p=0.0000), D8=`0.2694`
  (p=0.0000), all significant. Gate 4 not yet run on Section 216 (only
  the lighter `coupling_graph_diagnostic`-only D3 check was) -- run the
  full Gate 4 on whichever checkpoint is finally frozen.

---

## Phase A: pick and freeze the two checkpoints this whole proposal rests on

- [ ] **A1. Record final Section 211 vs 215 comparison.** Table: recon
  MSE, Stage-1 D3 bandedness p-value, Stage-2 `D_KY`/`lambda1`,
  standalone-rollout boundedness (yes/no), for both. Pick one, call it
  `LOCAL_AE`/`LOCAL_PROP` for the rest of this document.
  **Test:** the chosen checkpoint's Stage-2 standalone 2000-step
  rollout has `D_KY` in `[21,24]` and `max|z|` bounded (does not grow
  monotonically across the whole rollout) -- exactly the check that
  disqualified all 19 historical checkpoints and Section 211's own
  Stage-1-alone number. Do not proceed past Phase A without this.
- [ ] **A2. Freeze `GLOBAL_AE`/`GLOBAL_PROP` = Section 213.** Already
  validated (see status snapshot). No further action, just naming it
  so later steps can refer to it consistently.
- [ ] **A3. Confirm `LOCAL_AE` is genuinely a `KSAutoencoderLocalField`
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

- [ ] **B1. Baseline DA sanity check on `GLOBAL_AE`/`GLOBAL_PROP`,
  `--localizer none`.** Sweep `--n-prop-steps` (try 1, 3, 5) at a fixed
  moderate `--n-ensemble` (e.g. 32) to find a setting where the free
  ensemble actually develops meaningful spread between analyses (the
  earlier Section 140 attempt's likely failure mode: `n_prop_steps=1`
  gave the ensemble no time to diverge before each analysis re-collapsed
  it).
  **Test:** `calibration_spread_over_rmse` lands roughly in `[0.4, 1.5]`
  (loosely around the brief's `~0.8` target -- not requiring an exact
  match, just "not absurdly small/large") AND `skill_free_over_da > 2`
  (DA is doing SOMETHING, not indistinguishable from the free run).
- [ ] **B2. Repeat B1's sweep on `LOCAL_AE`/`LOCAL_PROP`.** Different
  encoder, different latent scale (`d=96` vs `d=44`) -- the good
  `n_prop_steps` may differ.
  **Test:** same criterion as B1, independently on the local-field
  checkpoint.
- [ ] **B3. Record the chosen `n_prop_steps` (and any other DA
  hyperparameter tuned in B1/B2) for each checkpoint** -- these become
  fixed for every sweep below; changing them between the SEC and
  Gaspari-Cohn runs would invalidate the Phase D comparison.

## Phase C: Phase 7 baseline -- SEC sweep on the GLOBAL latent

- [ ] **C1. Build a thin sweep driver** (`scripts/run_da_ensemble_sweep.sh`
  or similar): loops `run_da_pff.py` over `--n-ensemble` in
  `{8,16,32,64,128,256}` x `--localizer` in `{none, sec}`, on
  `GLOBAL_AE`/`GLOBAL_PROP`, collecting each run's JSON output into one
  combined table (a small Python script reading the per-run
  `artifacts/da_pff_*.json` files is enough -- no need for a fancier
  aggregator).
  **Test:** at smoke scale (`--n-ensemble` in `{4,8}` only, few cycles),
  the sweep runs end to end with no crash and produces a well-formed
  table with one row per `(N_ens, localizer)` pair.
- [ ] **C2. Run the real sweep** (full `N_ens` set, `B3`'s chosen
  `n_prop_steps`, enough `n_cycles` for the RMSE estimate to stop moving
  much cycle-to-cycle -- check by eye on one run before committing to
  the full sweep).
  **Test/plot:** RMSE-vs-`N_ens` curve, with and without SEC. Report
  plainly whether SEC helps at small `N_ens` here (the literature's own
  claimed benefit) -- if it doesn't on this checkpoint, say so; this
  curve is the actual Phase-7 baseline result Part 4.3 flags as "never
  run before," not a foregone conclusion.

## Phase D: Phase 13 core -- Gaspari-Cohn sweep on the LOCAL field

- [ ] **D1. One-cycle integration test before any real sweep**:
  run a SINGLE DA cycle with `--localizer gaspari_cohn` on `LOCAL_AE`/
  `LOCAL_PROP` and directly inspect the localized `B` matrix (add a
  `--dump-localized-B path.npy` style debug hook, or just print it in a
  scratch script) to confirm sites far apart on the ring are actually
  near-zero after tapering, not just "the run didn't crash."
  **Test:** for a chosen `gc_c`, `B_localized[site_i, site_j]` for
  `circular_distance(i,j) > 2*gc_c` sites is within numerical noise of
  zero (this directly re-uses `test_taper_matrix_local_window_actually_
  localizes`'s logic, just checked against a REAL forecast covariance,
  not a synthetic one).
- [ ] **D2. Sweep `--gc-c` alone** at one fixed, moderate `N_ens` (reuse
  whatever `N_ens` looked stable in Phase C) to find a good localization
  radius before committing to the full 2-D sweep. Cross-check the
  chosen radius against Phase 2's own measured light-cone bound
  (`ks_latent.analysis.spreading.minimum_localization_radius`) if that
  has been measured for KS at this `dt_snap`/propagator -- if not
  measured, note it as a gap, don't skip the comparison silently.
  **Test:** at least one `gc_c` value gives RMSE below the no-
  localization baseline at this `N_ens` (Phase B's own free/DA numbers
  already established a no-localization reference point). If NO `gc_c`
  beats no-localization here, stop and investigate before Phase D3 --
  that would be a genuinely important negative result on its own.
- [ ] **D3. Run the full `N_ens` sweep** at D2's chosen `gc_c`, same
  `N_ens` set and `n_prop_steps` as Phase C (for a fair comparison).
  **Test/plot:** RMSE-vs-`N_ens`, three curves on one plot -- no
  localization, SEC (Phase C), Gaspari-Cohn (this step).

## Phase E: the actual decision

- [ ] **E1. Apply the pre-registered rule mechanically.** A short script
  (`scripts/check_4_3_decision_rule.py` or similar) that reads Phase
  C/D's result tables and asserts, at every `N_ens` in the sweep (or at
  least the small-`N_ens` regime where localization is supposed to
  matter most): `rmse_gaspari_cohn[N] <= rmse_sec[N]`. Print PASS/FAIL
  per `N_ens`, not just an overall verdict -- a rule that only holds at
  some ensemble sizes is itself informative.
  **Test:** running this script against Phase C/D's actual output files
  produces a table of PASS/FAIL by `N_ens`; a hand-constructed synthetic
  pair of CSVs (one where local clearly wins, one where it clearly
  loses) should make the script report PASS/FAIL correctly -- i.e. the
  DECISION SCRIPT ITSELF gets a unit test before being trusted on real
  data.
- [ ] **E2. Write the verdict into `docs/RESULTS.md`**, either way,
  with the plot and the PASS/FAIL table. A negative result here (SEC
  wins) is still publishable -- Part 4.3's own text already frames why
  (SEC can't do L-transfer; that argument only matters if this
  comparison is reported honestly regardless of outcome).

## Phase F: L-transfer -- the headline result, and a real blocking gap

**Read this whole phase before starting it.** `LOCAL_PROP` (Section
211/215) uses a fully-connected GLOBAL `mlp` propagator backbone.
That architecture's weight matrices are tied to a FIXED `d_latent` --
if the domain length `L` changes, `n_sites` (and therefore `d_latent =
n_sites * local_channels`) changes too, and a fixed-shape `mlp` simply
cannot run on a different input/output dimension at all. **The
L-transfer claim structurally requires a translation-equivariant,
weight-shared propagator** (`backbone` in `{masked_mlp, local_mlp, cnn,
node}` -- anything whose parameter count doesn't depend on `n_sites`),
not the propagator Phase A just validated. This was the ORIGINAL
motivation for trying `--aux-backbone masked_mlp` (Section 212, launched
then superseded mid-investigation by the Section-52-regression
question) -- it needs to be revisited specifically for this phase, not
skipped.

- [ ] **F1. Train and validate a size-transferable local propagator**
  on `LOCAL_AE`'s own encoder (or a fresh matching one): `local_field`
  encoder + `masked_mlp` (or `local_mlp`/`cnn`/`node`) propagator,
  same regularizer recipe, same Stage 1 -> Stage 2 pipeline Phase A
  used. Call the result `TRANSFER_PROP`.
  **Test:** identical bar to A1 -- Stage-2 standalone 2000-step rollout,
  `D_KY` in `[21,24]`, `max|z|` bounded. Do not skip this because
  Section 212 "seemed to work" in an earlier smoke test -- it was never
  run through Stage 2 or checked this rigorously.
- [ ] **F2. Architecture-level transfer test, no DA yet.** Run
  `LOCAL_AE`/`TRANSFER_PROP` (both trained at `L=100`) at a LARGER `L`
  (e.g. `L=200`, by re-deriving `n_sites`/`NX` for the new `L` and
  confirming the encoder/propagator accept the new shapes with zero
  retraining) in pure free-running forecast mode.
  **Test:** the decoded field at `L=200` has a plausible energy
  spectrum (peaks near the same physical wavenumber `k~1/sqrt(2)` KS's
  own linear instability predicts, brief `test_energy_spectrum_peak`'s
  criterion) AND the standalone `D_KY` measured at `L=200` divided by
  `200` is within a reasonable tolerance of the `L=100` model's own
  `D_KY/100` ratio (the extensivity check, `D_KY ~ 0.226*L`) -- this is
  the SAME `D_KY`-per-unit-length constant this project has used
  throughout, now checked for transfer, not just at one fixed `L`.
- [ ] **F3. Full localized-DA-at-larger-L test.** Run the actual DA
  cycling pipeline (Phase D's own `--localizer gaspari_cohn` setup) at
  `L=200` using the `L=100`-trained `LOCAL_AE`/`TRANSFER_PROP`, zero
  retraining.
  **Test:** report RMSE/skill/calibration at `L=200`, and explicitly
  confirm `GLOBAL_AE`/`GLOBAL_PROP` and the SEC taper CANNOT even be
  evaluated at `L=200` at all (different `d_latent`/different fitted
  taper) -- this contrast (one method transfers, two structurally
  cannot) is the actual headline result Part 4.3 is after, not just a
  good RMSE number in isolation.
- [ ] **F4. Write up F1-F3 in `docs/RESULTS.md`**, explicit about
  which comparators could and couldn't even attempt the `L=200` test.

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
