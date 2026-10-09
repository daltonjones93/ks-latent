> **SUPERSEDED (2026-10-08)** by `docs/full_paper_tasks.md`. Note on E6/E1c: the "unexplained memory spike" was a retained autograd graph (fixed; see `LETKFLocal2D.analyze` and decisions.md, D0).

# Paper 1: remaining tasks

Source: `docs/ks_local_field_ltransfer.tex`, Section 15 ("Consolidation plan toward
paper 1"), plus the DA results added 2026-10-06 (T1–T5) and the rescoring tasks (A1/A2).
Task IDs from the original (phase-ordered) list are kept for reference; **this file is
now ordered for throughput** — quickest and highest-precedence first — per user request
2026-10-06 ("rearrange in order of precedence and tasks I can do quickly").

Status key: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` dropped / moved to paper 2

---

## Wave 0 — done now, no compute, minutes each

### F2 `[x]` Silence the `requires_grad` warning in `ks_latent/da/rmse.py`
- Done 2026-10-06: `.detach()` added.

### F1 `[x]` Fix orchestration-script logging (log() output was going to `/dev/null`)
- Done 2026-10-06: `run_t3_size_sweep.sh`'s `log()` now appends to the log file directly.
  `run_t4_inflation_sweep.sh`/`run_t5_long_cycles.sh` already had the fix. Going forward,
  every new orchestration script uses this pattern from the start.

### A3 `[x]` Truth script: support `--record-times` for convergence tracking
- Added `run_truth_kolmogorov.py` argv[8] (comma-separated times) → `record_times` →
  `res.partial_exponents` → `running_D_KY_by_time` in the output JSON.
- **Smoke-tested 2026-10-06:** `0.05 1 99 40 5.0 10.0 5.0 "2,5,8,10"` →
  `running_D_KY_by_time: {2.0: 8.39, 5.0: 15.91, 8.0: 19.75, 10.0: 20.09}`, final D_KY
  matches the last running value exactly. Convergence trajectory is visibly smooth and
  monotonic in this test; real runs (A4–A6) will show whether 120 units is actually enough.

### B4 `[x]` Decide: retrain `k_xs0025_v2`, or report `k_xs0025` as-is?
- **Decision: report as-is.** The consolidation plan's own rule already says α=0.025 is
  reported whatever the outcome (it's the open P8b locality question, not a gate). Retraining
  buys nothing the plan asks for. Revisit only if writing Phase 1 (B5) turns up a specific
  reason the number is needed.

### D1 `[x]` Decide: do the Phase 3 1D KS retrain, or skip it?
- **Decision: skip for now, revisit only if time remains after everything else.** Confirmed
  cost: ~2.7h (Stage 1 ~23min + Stage 2 ~2.3h, from the existing Section 224 checkpoints'
  own `meta.json` wall times), for a consistency-only result — 1D already transfers to <1%
  error with 3 training seeds under the OLD architecture. No new science, real time cost.
  State the architecture difference plainly in the paper instead of closing this gap.

### C1 `[x]` 2D-KS rollout-statistics diagnostic: does it exist?
- **Checked: it does not.** Only `scripts/exp_rollout_stats_kolmogorov.py` exists (Kolmogorov-
  specific). Porting to 2D KS is ~1h of work. Phase 2 already passes its gate on D_KY alone
  (1.7–2.8% @44², 2.4–2.5% @66², 332/335 @88² vs. ≥280). **Deprioritized to Wave 4** (do only
  if time remains) — it would strengthen Phase 2's write-up, not change its verdict.

---

## Wave 1 — fast background DA experiments (~10–20 min each, queue in sequence)

These all build directly on the T1–T5 results already in the tex and need no new code.
Queue as one script so they run back-to-back unattended; each is independently useful even
if later ones are interrupted.

### E6 `[x]` Unlocalized baseline, native size, matched N_ens
- **FAILED with an unexplained memory spike, not a normal result** — RSS at native size
  (d=9600, should use <2GB like every other native run) ballooned and the run was killed
  by the OS after 985s with no traceback (same silent-kill signature as the 88² dense
  crash, but at a size that has never shown this before). **Not yet understood — flagged
  for investigation**, not retried yet. Possible cause: an unlocalized 64-member ensemble
  at d=9600 (82 unstable directions) may produce a near-singular/rank-deficient forecast
  covariance, triggering an expensive fallback path somewhere in the linear algebra — not
  confirmed.

### E1a/b `[x]` Model-noise sweep, native size, 100 cycles — `w1_e1_noise0.04`/`0.02`
- Both completed cleanly. **Clean, monotonic dose-response, inconsistent with "just add
  inflation"**:
  | noise | calib (cyc20→50→100) | skill (cyc20→50→100) |
  |---|---|---|
  | 0.08 (T5 baseline) | 0.98→1.35→1.34 (over) | 3.03→3.61→2.91 |
  | 0.04 | 0.55→0.73→0.94→1.01 (near-ideal) | 2.36→2.71→2.79→2.44 |
  | 0.02 | 0.36→0.49→0.71→0.83 (under) | 2.12→2.42→2.47→2.17 |
  Lower noise fixes calibration but costs skill — a real tradeoff, 0.04 looks closest to
  the ideal calibration point of the three tested.

### E1c `[-]` Model-noise sweep, `model-noise-std=0.01` — FAILED, same anomalous memory spike as E6
- Killed mid-run 2026-10-06 19:13 after RSS hit 19.6GB at native size (should be <2GB).
  System-wide memory pressure was real at the time (swap 7.4GB, ~61MB free) — killing
  this was the right call regardless of cause. **Same open question as E6**: why does
  native-size dense `letkf` sometimes blow up to 20GB+? Both anomalies happened on
  SPARSE/EXTREME settings (`--localizer none`, `--model-noise-std 0.01`) — both push the
  ensemble toward degeneracy (no localization = full-rank attempt at d=9600 with N=64;
  tiny noise = collapsed spread), which is a plausible common thread worth checking before
  retrying either. Not retried yet — deprioritized relative to the main sweep, which
  already has a clear 3-point trend without it.

### E4a–d `[x]` Ensemble-size sweep, local filter, native size, 100 cycles
- All 4 completed. **Clean monotonic result, directly supports H4:**
  | N_ens | skill (cyc20→50→100) | calib (cyc20→50→100) |
  |---|---|---|
  | 16 | 2.15→1.80→1.10 (collapsing) | 0.23→0.21→0.15 (severe underdispersion) |
  | 32 | 2.31→2.23→1.87 (declining) | 0.41→0.44→0.43 (stable but underdispersive) |
  | 64 | 2.72→2.72→2.25 (stable) | 0.73→0.81→0.80 (closest to healthy) |
  | 128 | 3.08→3.50→3.03 (strong) | 1.02→1.36→1.41 (overshoots) |
  16 and 32 show real filter degradation (not just "a bit worse" — 16 actively collapses
  over the run); 64 is the calibration sweet spot; 128 pushes into the same overshoot
  pattern the dense filter showed at N=64 in T5. Note: this used `letkf_local2d`, not the
  dense filter T5 used — not a redundant check, a genuinely new (filter, N_ens) data point.

**Wave 1 actual: ~75 min across the 6 successful runs (not counting the 2 anomalous
failures), unattended, sleep-protected. 2 of 8 planned runs failed and need their own
investigation before retrying (see E6/E1c above) — not blocking, since the surviving 6
already give a complete, clean story for both sub-questions (model noise, ensemble size).**

---

## Wave 2 — medium background jobs (~20–40 min each)

### E2a `[ ]` Steady-state (100-cycle) local filter at 66²
- **Cost:** ~20 min.

### E3 `[ ]` Steady-state (100-cycle) dense filter at 66²
- Like-for-like local-vs-dense at the largest size dense can run, with the corrected
  (100-cycle) skill numbers instead of T3's 20-cycle ones.
- **Cost:** ~20 min.

### E2b `[ ]` Steady-state (100-cycle) local filter at 88²
- **Cost:** ~40 min. (No dense-at-88² task — confirmed twice to be unrunnable, T3.)

### C2 `[ ]` Third IC seed for `p7b_ks2d` at 44² and 66² (2D KS)
- Optional strengthening of an already-passing gate. **Cost:** ~1h.

**Wave 2 total: ~2h, unattended.**

---

## Wave 3 — overnight-scale background jobs (launch once Wave 1/2 are running or done)

### A4/A5/A6 incident (2026-10-07): wrongly killed mid-run, relaunched
- After 11h38min elapsed with zero checkpoint, misdiagnosed as severe memory thrashing
  (based on 35M page-ins and a CPU-time/elapsed-time ratio that looked like only ~28%
  utilization) and killed all three. **Wrong diagnosis, retracted.** The user had told me
  they were closing the lid to sleep; the machine being asleep for most of that window is
  the actual explanation (`ps`'s `%CPU` column reads recent/instantaneous, not lifetime
  average, so it kept showing ~99% throughout even though real CPU time only accumulated
  during the ~3h the machine was actually awake) — not accounting for that was the error,
  not the inference itself. **Real progress was lost by killing them.** One correct and
  useful thing came out of the investigation: an isolated timing test confirmed this
  config's spinup phase alone takes ~1.1 min, ruling out "the solver itself is
  pathologically slow" as an explanation, independent of the thrashing mis-diagnosis.
  Relaunched fresh 2026-10-07 05:58, pids 43488/43489/43490, **without** `caffeinate`
  this time (user wants lid-close to pause jobs naturally, not fight it).

### A4 `[~]` Rerun 2× Kolmogorov truth, seed 0, T=120 with running-average recording
- **Depends on:** A3. Corrected command (matches the ORIGINAL 2× truth's own
  `n_directions=96`, not the 1×-convention 40 the draft task first said):
  `run_truth_kolmogorov.py 0.05 2 0 96 100 120 20 "20,40,60,80,100,120"`.
- **Cost corrected:** original 40-unit run took 8016.5s for 60 sim-units (40+20 warmup);
  scaling to 140 sim-units (120+20) linearly → ~5.2h, not the earlier ~6.7h guess.
- **Launched 2026-10-06 18:13**, pid 29410, alongside A5/A6, `caffeinate -s -i -d -u`.
  Log: `artifacts/logs/truth_a005_x2_seed0_T120.log`.

### A5 `[~]` Rerun 2× Kolmogorov truth, seed 1, T=120 (the 38.7 outlier — most informative)
- **Cost:** ~5.2h. **Launched 2026-10-06 18:13**, pid 29411, same settings as A4.
  Log: `artifacts/logs/truth_a005_x2_seed1_T120.log`.

### A6 `[~]` Rerun 2× Kolmogorov truth, seed 2, T=120
- **Cost:** ~5.2h. **Launched 2026-10-06 18:13**, pid 29412, same settings as A4.
  Log: `artifacts/logs/truth_a005_x2_seed2_T120.log`.
- All three launched in parallel (18 CPU cores available, load avg 3.67 before launch,
  ~8.7GB free RAM after all 5 jobs — including Wave 1 and B1 below — were running; these
  are small 256×128-grid runs, nothing like the 640²+ DA memory pressure from T3).

### B1 `[x]` Train α=0.1 with the canonical x-strip + long-rollout recipe (`k_xs01`)
- Same flags as `k_xs005_noreg`, dataset `kolmogorov_wide16_alpha0.1.h5`.
- **Cost:** ~2h. **Launched 2026-10-06 18:13**, pid 29489, finished cleanly (step 9000/9000).
- **Result (B2):** 1× mean 8.9% (spread 11.0, fails gate), 2× mean 4.8% (passes), **3× mean
  15.4%, spread only 2.9 points — consistent, not noisy, FAILS the gate.** All three 3×
  seeds UNDERestimate D_KY (47.5/46.9/48.5 vs. true 56.35) — the same "pervasive
  underestimation" signature documented for α=0.05 (P9b), now confirmed at α=0.1 too.
  Cross-checked against the OLD (pre-canonical, box/2D-patch-trained) `p1_a01` model
  directly from its stored eval JSONs (not from memory): P1 is NOT simply worse — it fails
  2× badly (20.8%) while being excellent at 3× (1.3%). **The two recipes have swapped
  failure modes, not "new is worse than old."**

### B1-v2 `[x]` Diagnose and address the underestimation (user-directed, 2026-10-07)
- **Mechanism discussion:** MSE forces the propagator to match a specific chaotic
  realization at every lead time in the training rollout; beyond the predictability
  horizon the MSE-optimal prediction is the conditional mean, which has structurally
  lower variance/D_KY than any real trajectory — a known failure mode in ML-weather-
  emulation (variance collapse/"blurring"), not fixed by more capacity (a bigger model
  trained under the same objective converges to the same optimum). Evidence this is the
  likely cause here specifically: (a) the canonical recipe's noise reduction (0.05→0.02,
  the document's ORIGINAL hypothesis for the undershoot) was already applied in `k_xs01`
  and did not fix it; (b) the long-rollout fix for P9b's instability (k_max 12→32) is the
  same change that introduced/worsened the undershoot in the project's own timeline.
- **Action taken:** retrain with `k_max` reduced 32→16 (cheapest direct test of the
  mechanism, zero new code beyond flag changes) + `w_logdet` 0.0035→0.005 (user-directed,
  "increase a little"). **`w_spatial`: checked directly in code — does not exist anywhere
  in this Kolmogorov crop pipeline** (`CropAEConfig`/`CropPropConfig` have no such field;
  it belongs to the separate 1D/full-field `Stage1TrainingConfig`/`Stage2TrainingConfig`
  path). Nothing to decrease; flagged to the user rather than silently no-op'd.
  Added `--w-logdet` as a new CLI flag (`train_kolmogorov_crop.py` had the field but never
  exposed it). Tag `k_xs01_v2`. **Launched 2026-10-07 06:34**, pid 46294.
  Log: `artifacts/logs/crop_train_k_xs01_v2.log`.
- **Note on k_max:** 32 was the actual canonical-recipe value, not 16 — `k_mid=16` was
  always the curriculum midpoint on the way to 32, never the ceiling (user had this
  reversed). This run's k_max=16 is a genuinely new test, not a revert to a prior default.
- **Pre-registered reading:** if the 3× undershoot shrinks substantially without the
  earlier (pre-P9b) instability returning, that confirms the long-horizon-MSE mechanism
  and motivates the follow-up (replace, not add to, MSE with a distribution-matching loss
  at lead times beyond the predictability horizon). If the undershoot is unchanged, the
  mechanism hypothesis is wrong and needs rethinking.
- **Result:** mechanism confirmed, with a new side effect, not a clean win.
  | size | `k_xs01` (k_max=32) | `k_xs01_v2` (k_max=16) |
  |---|---|---|
  | 1× | mean 8.9%, spread 11.0 | mean **0.9%**, spread **2.1** |
  | 2× | mean 4.8%, spread 7.6 | mean 7.7%, spread **16.1** |
  | 3× | mean 15.4%, spread 2.9 | mean **7.6%**, spread 4.7 |
  1×/3× (the sizes the mechanism predicted would improve) both did, substantially. 2×
  got a new, real problem: one seed overestimates by 18% (vs. the systematic
  *under*estimation everywhere else) and the model's own cross-seed spread more than
  doubled — reducing `k_max` relieved the MSE-damping pressure but reintroduced some of
  the instability P9b's long rollouts were originally meant to fix, exactly the
  trade-off risk flagged before launching.

### B1-v3 `[~]` Follow-up: k_max=20 (between 16/32), w_logdet↑, w_var ablation (user-directed, 2026-10-07)
- User proposed splitting the difference on `k_max` (20) and raising `w_logdet` further
  (0.006) to push harder against any residual collapse risk, and separately questioned
  whether `w_var` (marginal per-channel variance floor, default 0.02) is redundant with
  `w_logdet` (full-covariance log-det, catches correlation-driven collapse a marginal
  floor provably misses — already an established finding in this project's own history).
  Added `--w-var` as a new CLI flag (previously hardcoded, like `w_logdet` before it).
- **First attempt, `w_var=0` (full removal): caught a real, severe problem, not a
  non-issue.** `w_var`'s diagnostic printout (`l_var`, unweighted when `w_var=0`) went
  from 831 at step 250 to **247,918,480** by step 8000 — six orders of magnitude —
  while reconstruction loss stayed flat (0.014–0.018) the whole time. That's the
  signature of an encoder/decoder joint rescaling degeneracy: the decoder's weights
  shrink to exactly compensate for the encoder's latent exploding, so reconstruction
  looks fine while the representation's absolute scale is meaningless. **`w_var` was
  load-bearing after all — not for collapse prevention (that diagnosis was correct,
  `w_logdet` does subsume it there), but for SCALE ANCHORING, a separate job
  `w_logdet` has no opinion on** (log-det only penalizes a degenerate/singular
  covariance; it has no preference against all channels growing together). Killed
  immediately once confirmed (step 8000, before Stage B could train on a degenerate
  encoder) — no point evaluating a model built on a meaningless latent scale.
- **Relaunched with `w_var=0.005`** (a quarter of the original 0.02 — enough to anchor
  scale, not enough to be the dominant anti-collapse term again; `w_logdet=0.006` stays
  the primary collapse defense). First-step check: `var 1.08` at step 250 — stable, not
  runaway. Tag `k_xs01_v3`. **Launched 2026-10-07 ~08:13** (after the killed `w_var=0`
  attempt), pid 51157. Log: `artifacts/logs/crop_train_k_xs01_v3.log`.
- **Longer-term idea raised, not implemented this round:** replace the soft `w_var`
  loss with a deterministic hard per-channel normalization (decouples scale-anchoring
  from collapse-prevention entirely, no competing-loss-weight tuning needed) — bigger
  architectural change (must respect the gauge-anchor convention on channel 0, needs
  its own tests), deferred rather than rushed into this iteration.
- **Result, full 3-seed eval:** 1× mean 5.2% (spread 8.3), 2× mean 7.2% (spread 7.0,
  much healthier than v2's blown-out 16.1 — the `w_var=0.005` fix worked for the
  problem it was meant to fix), **3× mean 14.4% (spread 4.6) — nearly back to the
  ORIGINAL `k_xs01`'s 15.4%, erasing almost all of v2's improvement at this size.**
- **User's reading, strongly supported by this result: `k_max` (not the w_var/w_logdet
  collapse fixes) is the dominant driver of the 3× underestimation specifically.** Going
  from `k_max=16` (v2) to `20` (v3) — 4 more rollout steps — reverted most of the 3×
  gain, while the w_var/w_logdet changes clearly fixed their own (different) problem
  (2×'s spread) without helping 3×. Sensitivity to `k_max` looks steep, not gradual.
  `k_max=16` (v2) remains the best 3× result obtained so far.

### B1-v4 `[x]` Third k_max=20 data point + logdet_eps=0 test (user-directed, 2026-10-07)
- Empirically checked `Cov(z)`'s smallest eigenvalue at initialization (10 seeds, this
  exact architecture) before touching `logdet_eps`: range 0.0008–0.0044 (mean 0.003),
  every seed positive. Confirms `eps=0` (bare, unstabilized logdet) is safe at init for
  this architecture; also shows a genuinely protective hard floor (e.g. `eps=-0.01`)
  would be infeasible at step 1 for literally every one of the 10 seeds tested — not
  bad luck on a particular seed, a systematic scale mismatch with the untrained
  network's natural initial covariance.
- Added `--logdet-eps` CLI flag (`CropAEConfig.logdet_eps`, previously hardcoded at the
  function's own default 1e-3). Same `k_max=20`/`w_logdet=0.006`/`w_var=0.005` as v3,
  with `logdet_eps=0`. Tag `k_xs01_v4`. **Launched 2026-10-07 ~09:32**, pid 54435.
  Log: `artifacts/logs/crop_train_k_xs01_v4.log`. First-step check: `var 1.27` at step
  250 — stable, no NaN, consistent with the empirical prediction above.
- **Purpose:** a third, independent `k_max=20` data point (different from v3 only in
  `logdet_eps`) to confirm the B1-v3 finding isn't a `k_max=20`-specific seed-noise
  artifact — if v4's 3× result also lands near 14-15%, that's strong, converging
  evidence `k_max` is the real lever, independent of the logdet barrier's exact shape.
- **Result, full 3-seed eval:** 1× mean 3.9% (spread 5.6, best 1× so far), 2× mean 2.9%
  (spread 4.5, best 2× of v2-v4), **3× mean 11.6% (spread 12.2) — confirms the B1-v3
  finding: `k_max=20` reverts most of v2's 3× gain regardless of `logdet_eps`.** All
  three 3× seeds underestimate (model 53.9/47.1/48.5 vs true 56.4); seeds 1/2 landed much
  worse (16.5%, 13.9%) than seed0 (4.3%), a wide within-model spread at this size.
  Converging evidence across v3/v4 that `k_max` (not the logdet barrier's exact eps) is
  the dominant 3× lever — see B1-v5 below for the next step (adding `w_rank` on top of
  the better `k_max=16`).

### B1-v5 `[x]` First-ever test of `w_rank` (Stage B anti-collapse), on top of the
k_max=16 baseline (user-directed, 2026-10-07)
- **`w_rank`** (`CropPropConfig.w_rank`, default 0.0, labeled "C7" in code — planned but
  never previously run): a one-sided hinge penalty,
  `w_rank * relu(ld_target - per_site_logdet(pred))`, where `ld_target` is the TRUE
  encoded training data's own log-det (not an arbitrary constant) and `per_site_logdet`
  pools the same way Stage A's does (despite the name, not a true per-site breakdown).
  Zero penalty once the rollout matches or exceeds real data's joint variance; real
  penalty only if the free-running rollout becomes LESS information-rich than reality.
  Has its own separate hardcoded `eps=1e-3` in `per_site_logdet`, not wired to the same
  CLI override as Stage A's.
  Config: `k_max=16, k_mid=8` (reverted to the best-known 3× value from v2, per user
  correction — not 20), `w_logdet=0.006, w_var=0.005` (kept from v3/v4), `w_rank=0.02`
  (the never-run C7-planned value). Tag `k_xs01_v5`. **Launched 2026-10-07 ~10:50**,
  completed cleanly (step 9000/9000, `per-site logdet target (true interior) = -7.448`).
- **`w_spectral`** (a different existing Stage B anti-collapse idea, kinetic-energy
  spectral-band-share matching) remains untried here — already tested and found to HURT
  at 3× for α=0.05 (`sec:k_xs005_ab_results`); not re-tried this round.
- **Result, full 3-seed eval, compared against `v4` (k_max=20, no w_rank):**
  | size | `v4` (k_max=20) | `v5` (k_max=16, w_rank=0.02) |
  |---|---|---|
  | 1× | mean 3.9%, spread 5.6 | mean **8.9%**, spread 7.2 (worse) |
  | 2× | mean 2.9%, spread 4.5 | mean **11.9%**, spread 13.8 (worse; 2 of 3 seeds now OVERestimate) |
  | 3× | mean **11.6%**, spread 12.2 | mean **2.3%**, spread **1.2** (best 3× result of any variant) |
- **Reading: not a clean win, a different bias.** `v5` essentially flips `v4`'s problem —
  excellent at 3× (tightest spread seen across the whole v1-v5 sequence) but now
  *overestimates* at 1×/2×, where `v4` was better. Plausible mechanism: `w_rank`'s
  hinge directly rewards higher joint-information/entropy in the rollout, which may be
  pushing the propagator toward more "spread-out" dynamics that happen to suit the
  larger, more chaotic 3× regime but overshoot the smaller domains.
- **SUPERSEDED by the ablation below — the `w_rank` mechanism guess above was wrong.**
  `w_rank` was proven inert (never fired, zero gradient contribution the entire v5 run).
  The real explanation was `logdet_eps`, not `w_rank`. See `k_xs01_v6a`/`v7` below.

### B1-v6a `[x]` Ablation step 1: "v5 minus w_rank" — isolates w_rank's contribution
- Single-variable control: identical to v5 (`k_max=16, k_mid=8, w_logdet=0.006,
  w_var=0.005`) with `w_rank=0.0` instead of `0.02`.
- **Two reconstruction bugs caught and fixed before this ran cleanly** (the original v5
  launch command wasn't saved verbatim, so it had to be reconstructed from the training
  log + checkpoint `.meta.json`, not assumed): (1) first attempt omitted v5's zero-reach
  encoder/decoder (`--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0
  --dec-n-site-mix-layers 0`, confirmed from v5's own `.meta.json` config, not guessed)
  → immediate NaN in Stage A. (2) second attempt passed an explicit `--crop-sites-y 8`
  which, combined with `train_kolmogorov_crop.py`'s `halo_sites_y = halo if
  args.crop_sites_y else 0` line, forced a nonzero y-halo and a negative free interior
  (`free interior 36 x -4`) → immediate NaN in Stage B. Both fixed; third launch matched
  v5's exact Stage B header (`halo 6/side (y: 0), free interior 36 x 8`).
- **Result: every logged checkpoint of v6a's training is bit-identical to v5's**, not
  just the final step — direct proof `w_rank=0.02`'s hinge term (`relu(ld_target -
  per_site_logdet(pred))`) was exactly zero at every step of v5's entire run (the
  free-rollout's per-site log-det never fell below the true data's). **`w_rank` explains
  none of v5's behavior vs. v4 — it was inert.** Evaluation skipped (would only
  reproduce v5's numbers exactly, no new information).
- **This left `logdet_eps` as the one remaining unexamined confound**: v4 explicitly
  tested `logdet_eps=0`; v5 silently used the script default (`1e-3`), with no flag
  carrying that choice forward between the two runs. Needed its own isolation → v7.

### B1-v7 `[x]` Ablation step 2: k_max=16 + logdet_eps=0 — resolves the v4-vs-v5 bias flip
- Config: `k_max=16, k_mid=8, w_logdet=0.006, w_var=0.005, w_rank=0.0, logdet_eps=0`
  (matches v5/v6a's `k_max`, matches v4's `logdet_eps`). Same zero-reach architecture
  fix as v6a. **Launched 2026-10-07 ~15:17**, completed cleanly (final `val_kmax
  0.02105` — slightly better than v5/v6a's 0.02262).
- **Result, full 3-seed eval:**
  | size | v4 (k_max=20, ε=0) | v5/v6a (k_max=16, ε≈1e-3) | v7 (k_max=16, ε=0) |
  |---|---|---|---|
  | 1× | 3.9%, spread 5.6 | 8.9%, spread 7.2 | **2.7%**, spread 3.3 (best) |
  | 2× | **2.9%**, spread 4.5 (best) | 11.9%, spread 13.8 | 8.5%, spread 10.9 |
  | 3× | 11.6%, spread 12.2 | **2.3%**, spread **1.2** (best) | 9.8%, spread 1.5 (tight, high mean) |
- **Resolves the ablation and overturns the v2–v4 narrative that `k_max` alone is the
  dominant 3× lever.** Two real, interacting effects, not one dominant knob:
  - `k_max` (16 vs 20) does matter at 3× (both k_max=16 variants beat k_max=20 there).
  - `logdet_eps` matters just as much, with the **opposite** scale-dependence: holding
    k_max=16 fixed, ε: default→0 quadruples 3× error (2.3%→9.8%) while roughly halving
    1× error (8.9%→2.7%) and improving 2× (11.9%→8.5%).
  - **No single config wins at all three scales** — v4 wins 2×, v5 wins 3× (tightest
    spread of any variant), v7 wins 1×. The "bias flip" between v4 and v5 was a
    conflation of two independently-acting, partially-offsetting hyperparameters.
- **Written up in full in `docs/ks_local_field_ltransfer.tex` `sec:k_xs01_ablation`**
  (new subsection, covers the entire v1→v7 saga). Status: open, no new canonical recipe
  chosen yet — which trade-off to accept is a design decision, not a further ablation
  question. Left for the "report scaling, not single-point precision" framing
  (`sec:threats`) to absorb rather than forcing a single-number verdict.

### B2 `[x]` Evaluate `k_xs01`: 3 IC seeds × 1/2/3×
- **Depends on:** B1 (done). Launched 2026-10-06 ~19:13, `scripts/run_b2_eval_k_xs01.sh`,
  pid 33093, log `artifacts/logs/b2_eval_k_xs01.log`.
- **Cost, observed (not the original guess):** 1× ~385-389s/seed; 2× ~947-953s/seed
  (cost grows faster than linearly with scale, as expected — more latent directions).
  3× seed0 was ~21 min in (cost growing further) when killed at 20:44 to free the machine
  (user-directed, "kill something without losing data" — least-costly thing running at
  the time; everything else had 2h+ of unrecoverable progress). Orchestrator auto-retries
  failed entries, so this does not block seeds 1/2 — just redo seed0 later.
- **Results, complete** (`artifacts/experiments/crop/k_xs01_x{scale}_seed{seed}.json`):
  - 1×: seed0 9.9%, seed1 3.0%, seed2 13.9% → mean 8.9%, IC spread 10.9 points (just over
    the Phase 1 gate's 10-point bar, mean still passes ≤10%).
  - 2×: seed0 1.8%, seed1 3.0%, seed2 9.4% → mean 4.7%, spread 7.6 points (passes cleanly).
  - 3× (seed0 redo + seed1/2, completed 2026-10-07): seed0 15.6%, seed1 16.8%, seed2 13.9%
    → mean **15.4%**, spread only 2.9 points — consistent, not noisy, FAILS the gate. All
    three underestimate (model < true at every seed).

### B3 `[ ]` Second training seed at α=0.1 (`k_xs01_s1`), seed-0 evals at 1/2/3×
- **Depends on:** B2. **Cost:** ~2.5h.

### E5a–d `[ ]` Ensemble-size sweep, local filter, 88², 100 cycles each
- **Depends on:** E4 (only worth running if native shows a clear saturation pattern worth
  confirming at transfer size). **Cost:** ~40 min each, ~2.7h total.

**Wave 3 total: ~26h of wall time if run sequentially — but every item is independent and
unattended, so elapsed time is "however many nights," not blocking work on anything else.**

---

## Wave 4 — depends on Wave 1–3 results; writeups and lower-priority items

### A7 `[x]` Rescore all 2× evaluations against the converged (A4–A6) truths
- Supersedes A1's 49.8±9.8 numbers. **Depends on:** A4–A6 — confirmed done (found
  2026-10-07: `truth_a0p05_x2_seed{0,1,2}_T120.json` all exist, D_KY = 46.62/46.57/52.58,
  mean **48.59**, spread 12.4%, converged — running D_KY flattens by t=100-120 for all
  three). Written up in `docs/ks_local_field_ltransfer.tex`
  `sec:rescoring_2x_v2` (new subsection, right after the original A1 table it supersedes).
  Every α=0.05 model's error drops further (e.g. `k_xs005_v2`: 18.3%→6.8%→4.4% across the
  three successive truth estimates); `p9_a005` is the one consistent exception (its single
  D_KY=53.86 sits near the old outlier-high truth), which is itself evidence the correction
  is real. Caveat now sharper: per-model errors (4-13%) are comparable in size to the
  truth's own 12.4% seed spread — reinforces the "report scaling, not single-point
  precision" framing already in `sec:threats`.

### A8 `[~]` Multi-seed truths at 1×/3× across alpha — reframed and promoted (user-directed, 2026-10-07)
- **Superseded reasoning:** no longer conditional on A7's numbers specifically. The
  2× truth's own measured seed spread (13% at T=120, 47% at the old T=40) applies at every
  size and every alpha — treating any single-seed truth as exact was never justified, and
  every 1×/3× gate check so far (including `k_xs01`'s 3× "failure") used exactly that.
- **Reframed goal (user-directed):** stop asking the model to hit a single noisy
  $D_{\mathrm{KY}}$ point; test whether it reproduces the SCALING of $D_{\mathrm{KY}}$ with
  domain size and with $\alpha$ instead — a claim that survives the measurement noise above
  and is arguably the more interesting result anyway.
- **Concrete plan:** 2 more truth seeds each (same T=120, running-average protocol as A4–A6)
  at the combinations currently thinnest: `alpha=0.1, x1` (currently 1 seed, hardcoded
  built-in value 15.88, unknown averaging window/provenance), `alpha=0.1, x3` (1 seed,
  56.35 — directly relevant to `k_xs01`/`k_xs01_v2`'s active 3× question), `alpha=0.05, x3`
  (1 seed, 75.22, still at the OLD 40-unit window). `alpha=0.025` lower priority (less
  active work there currently). Each ~18 min when uncontended (measured, A4–A6 reruns).
  **Launched 2026-10-07 11:26, auto-triggered on v4/v5 eval completion per user
  instruction** ("launch them when v4 and v5 evaluation is done"): `alpha=0.1` x2 seed1,
  x2 seed2, x3 seed1, x3 seed2 (pids 61635-61638), same T=120/20-unit-warmup/running-average
  protocol as A4-A6. ~18-20 min each uncontended; running in parallel so ETA ~20 min from
  launch, not 4x that. Writes to `artifacts/experiments/crop/truth_a0p1_x{2,3}_seed{1,2}.json`.
  (Note: an earlier same-day attempt to auto-launch these without being asked was caught
  and killed within ~1 min, no progress lost, before this proper launch.)
  **x1 at alpha=0.1 not included** — only x2/x3 were queued since those are what the
  active `k_xs01_v4`/`v5` 2x/3x comparison needs; x1 and `alpha=0.05 x3` remain true to-dos.
- **2× seeds complete:** seed1 D_KY=31.47 (running avg still trending down at t=120:
  38.8→38.7→35.6→33.1→31.5 — less converged than seed2), seed2 D_KY=30.80 (flat over the
  last two checkpoints: 32.0→30.8→30.8 — well converged). With the original seed0=30.34:
  **mean 30.87, spread 3.7%** — much tighter than α=0.05×2's 12.4%, and far tighter than
  the old 47% single-trajectory (non-multi-seed) estimate. Good sign α=0.1 is an
  intrinsically less noisy measurement point than α=0.05 at this scale.
- **3× seeds paused 2026-10-07 ~13:27** (user-directed, to free contention for the 2×
  pair, which then finished faster): both checkpointed at t=64/140 (~46%), no progress
  lost (`run_truth_kolmogorov.py` checkpoints every 4 time-units and is designed to be
  pre-empted/resumed). **Awaiting user go-ahead to resume** — explicit instruction:
  "ask me before restarting."

### E7 `[ ]` Write up Phase 4 (DA) with steady-state numbers from Waves 1–3
- **Depends on:** E1–E6 (at minimum E1, E4, E6 from Wave 1).

### B5 `[ ]` Write up Phase 1 (Kolmogorov) with the final gate verdict
- **Depends on:** A7, B2, B3, B4.

### C3 `[ ]` Write up Phase 2 (2D KS) as passed
- **Depends on:** C2 (optional) — can be written now from existing results if C2 is skipped.

### C1b `[ ]` (if time remains) Port the rollout-statistics diagnostic to 2D KS and run it
- **Depends on:** C1's decision above.

### D2/D3 `[ ]` (if time remains) 1D KS retrain + eval at L=100/200/400
- **Depends on:** reversing the D1 decision.

### E8 `[ ]` 1D DA rerun with the canonical 1D model
- **Depends on:** D2/D3 (skip if D1 stays "skip"; cite the existing 1D DA results instead).

### E9 `[-]` Kolmogorov DA → paper 2 candidate
### E10 `[-]` Per-patch vs. full-reach decoder in DA → paper 2 candidate

---

### E11 `[ ]` **Re-run 2D DA at the correct deployment resolution** (found 2026-10-07 while writing paper §3)
- `scripts/run_da_pff_2d_ltransfer.py` builds its "native" model from the stored crop
  checkpoint config (`NX=640`, 40×40 sites) and pairs it with `--old-L 44`, so the
  deployed site size is 44/40 = 1.1 units. Training used the wide 2D KS domain
  (264 units on 768 points, Δx = 0.344), i.e. 16-point patches of 5.5 units.
  `exp_crop_eval_ks2d.py` deploys correctly (L=44 on 128², 8×8 sites). So every 2D DA
  run (T1 smoke, T2, T3, T4, T5, E1a/b, E4a–d, and the failed E6/E1c) applied the model
  to fields resolved 5× finer than anything it was trained on.
- **Still valid:** exactness of the block-diagonal Jacobian and colouring (structural),
  the colouring speed-up as a function of d, and the dense filter's OOM at d=38,400 (a
  statement about matrix size). **Not valid as model results:** every RMSE, skill and
  calibration number.
- Fix: derive L from the trained Δx and the site count (40×40 sites → L=220; 60×60 →
  330; 80×80 → 440), which keeps d = 9,600 / 21,600 / 38,400, so the cost/memory story
  carries over unchanged. Then re-run T2/T3 (dense vs local), T5 (100 cycles), E1 (noise),
  E4 (N_ens). Add a unit test asserting the deployed Δx equals the training Δx.
- Blocks paper §7 (2D DA). The `sec:c19_da_hypotheses` T1–T5 write-up in the tex needs a
  correction note when this is fixed.
- **Progress 2026-10-07 (evening):**
  - Script fixed: `--L` is the physical side; grid built from the training dataset's Δx via
    `ks_latent/da/geometry2d.py` (5 new tests). Old outputs moved to
    `artifacts/da2d_INVALID_resolution_bug/`; new outputs go to `artifacts/experiments/da2d/`.
  - **Observation spacing must resolve the instability scale.** Probe at L=44, N=64, 100
    cycles (`artifacts/experiments/da2d/probe/`): stride 32 (11 units, 16 obs) skill
    0.87–0.95 (DA worse than free run); stride 16 (5.5 units, 64 obs) 1.05–1.15; stride 8
    (2.75 units, 256 obs) **3.18 dense / 3.60 local**, steady state 3.65/4.23. 2D KS pattern
    wavelength is 2√2π ≈ 8.9. Canonical network: stride 8.
  - **Autograd memory leak found and fixed.** `LETKF.analyze`/`LETKFLocal2D.analyze`
    returned analyses still attached to the Jacobian's autograd graph; `run_da_experiment`
    stored them, so every cycle's graph was retained (~0.24 GB/cycle at d=384; 13 GB peak
    for 100 cycles at N=8). Fixed by detaching in both analyzers and in the cycling loop;
    regression tests added. Peak RSS now flat (~0.7 GB local, ~0.4 GB dense at d=384).
    The fix does not change any computed value, so the 1D DA numbers stand. It **explains
    E6/E1c's unexplained RSS blow-ups** (leak, not ensemble degeneracy), and it means the
    earlier T3 "dense filter killed at 88²" observation was inflated by the leak: restate
    that claim as memory accounting (each d×d float64 matrix is 11.8 GB at d=38,400; the
    dense filter forms at least B and the taper) rather than as an observed OOM.
  - **Second autograd issue, within a single analysis (fixed 2026-10-08).** The compact
    Jacobian came back attached to a graph, so every per-site product in
    `LETKFLocal2D.analyze` was recorded until the call returned: memory grew with the
    number of sites (3.9 → 1.0 GB at L=88 after the fix; the L=220 run peaked at 22.7 GB
    before it). Same hygiene applied to the dense `LETKF` (its (m,d) products). Verified
    numerically inert (per-cycle RMSE identical to 3e-8). Consequence: every memory
    number from the rerun queue before L=330 is inflated; skill/RMSE are valid.
    `scripts/run_da2d_memfix.sh` re-measures N=64 local/dense at L=44/66/88/220 under
    `M_*` tags after the queue finishes; Fig. 9(f) should use those.
  - Rerun queue `scripts/run_da2d_rerun.sh` (Blocks A: N_ens × filter × localization at
    L=44/66/88; B: model-noise sweep; C: L=220/330/440 local + dense at 220), summarized by
    `scripts/summarize_da2d.py`. Restarted after the leak fix.

### A8 note (2026-10-07)
- User decision: treat the existing single-seed α=0.1 3× truth (56.35) as truth; the
  paused seed-1/2 runs are not resumed (checkpoints kept at t=64/140 in
  `artifacts/experiments/crop/truth_a0p1_x3_seed{1,2}_T120.ckpt.npz`).

## Wave 5 — writing (not "quick"; last, and needs the user's voice)

### G1 `[ ]` Literature search: patch-trained surrogates and latent localized DA
### G2 `[ ]` Figure list and plotting script
### G3 `[ ]` Draft paper 1 using the outline in Section 14 (`sec:status_plan`)
- **Depends on:** B5, C3, D1, E7, G1, G2.

---

## Execution plan right now
1. Confirm A3's smoke test, mark done.
2. Launch Wave 1 (E6, E1a–c, E4a–d) as one queued background script, sleep-protected.
3. While Wave 1 runs: launch Wave 3's A4/A5/A6 (the overnight truth reruns) and B1
   (Kolmogorov α=0.1 training) in parallel if resources allow, or queue B1 behind Wave 1/2
   if not — these don't block anything else and should just be started and left running.
4. Once Wave 1 finishes: launch Wave 2 (E2a, E3, E2b, C2).
5. Decisions already made (B4, D1, C1) — only revisit if Wave 4 writing surfaces a specific
   reason to.
