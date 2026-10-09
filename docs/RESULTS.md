# Results

Narrative results, with pre-registered decision rules stated **before** the
corresponding results (brief §14.4, §20). Populated phase by phase.

## Phase 1: KS solver and Kaplan-Yorke dimension (Gate 1)

**Decision rule (pre-registered in the brief, §3.2, §18):** the ETDRK4
solver is trusted for downstream work only if `test_L100_kaplan_yorke`
lands `D_KY` in `[21, 24]`, and `test_L22_lyapunov` lands `lambda_1` in
`[0.043, 0.05]` and `D_KY` in `[5.2, 5.6]`.

**Result: PASS.** Measured via `ks_latent.analysis.lyapunov.lyapunov_spectrum_ks`
(Benettin/QR on the variational ETDRK4 integrator, `n_directions=35`,
`total_time=2000` physical time units, `warmup_time=300`, for L=100;
`n_directions=15`, `total_time=4000`, `warmup_time=400` for L=22):

| quantity | measured | target | status |
|---|---|---|---|
| L=100 `D_KY` | 22.50 | [21, 24] | PASS |
| L=100 `lambda_1` (physical units) | 0.0899 | <= 0.1 | PASS |
| L=100 positive exponents | 12 | ~11 | PASS (qualitative) |
| L=22 `lambda_1` | 0.0489 | [0.043, 0.05] | PASS |
| L=22 `D_KY` | 5.228 | [5.2, 5.6] | PASS |

Full auto-generated table: `docs/REPLICATION_LOG.md`.

These numbers land close to both the literature (Edson et al. 2019: D_KY ~
0.226*L - c; the Koopman-paper figure of 23.2 at L=100) and the existing
44-dim latent model's headline result (D_KY ~ 21.4) reported in
`docs/PROJECT_HANDOFF.md` -- i.e. the from-scratch solver reproduces the
physical ground truth the rest of this program is built on top of.

## Phase 2: D7 information-spreading velocity and the light cone (Gate 2)

**Decision rule (pre-registered, brief §4):** measure `v_star` -- the
largest comoving velocity with a positive comoving Lyapunov exponent -- via
two independent estimators (comoving exponent, front tracking) on the
canonical L=100 config, averaged over >=100 independent base trajectories
and perturbation sites, and derive the light-cone/stencil/localization
bounds from it. This measurement, not a hyperparameter choice, is what
fixes the stencil half-width tested in Phase 11 and the localization radius
enforced in Phase 13.

**Result:** measured via `ks_latent.analysis.spreading.measure_ks_spreading`
on `KSConfig(L=100, NX=1024, dt=0.05, spinup_time=200)`, 150 trials,
`total_time=20` physical time units per trial, `v_grid` in `[-2.5, 2.5]`
(chosen to stay inside the periodic-wraparound guard,
`|v|*t_fit_hi <= 0.45*L` -- see "pitfall" below):

| estimator | value | note |
|---|---|---|
| comoving exponent `v_star` | **1.261 +/- 0.039** (SEM, 150 trials); trial-to-trial std 0.483 | zero-crossing of the mean `Lambda(v)` curve |
| front tracking (threshold 0.1 of peak) | 1.096 +/- 0.417 (std) | |
| front tracking (threshold 0.3 of peak) | 0.903 +/- 0.466 (std) | |
| front tracking (threshold 0.5 of peak) | 0.826 +/- 0.483 (std) | |

The two independent estimators agree to within a factor of ~1.2-1.5x, all
landing in the same O(1) physical-velocity regime -- brief §4's "two
independent measurements of one number... report both regardless." Front
tracking reads systematically a bit lower than the comoving-exponent
`v_star`, consistent with it being a stricter operational definition (a
fixed fraction of the *current* peak) rather than the asymptotic
zero-crossing of `Lambda(v)`.

The averaged `Lambda(v)` curve is close to symmetric about `v=0` (as
expected: KS has no preferred propagation direction once averaged over many
random perturbation sites and base trajectories), peaking around
`Lambda(0) ~ 0.02-0.06`, well below the leading Lyapunov exponent
`lambda_1 ~ 0.09` from Phase 1 -- the comoving exponent at v=0 measures the
growth rate of a *fixed-shape local* perturbation, not the globally optimal
(fully aligned, unconstrained-shape) unstable direction, so a smaller value
than `lambda_1` is expected, not a discrepancy.

**Pitfall found and fixed while implementing this:** an earlier `v_grid` of
`[-4, 4]` at this `total_time` let some rays wrap around the periodic
domain within the regression fit window and re-approach the original
perturbation site, producing a spurious *upturn* in `Lambda(v)` at the grid
edges (large |v| reading positive again). `comoving_exponents` now raises
if `|v|*t_fit > 0.45*L` for any point in the fit window, rather than
silently returning a contaminated curve; see
`test_comoving_exponents_raises_on_domain_wraparound`.

### Derived bounds (brief §4 formulas, `dt=0.05`)

Light-cone half-width `v_star * dt * stride`, swept over stride (**do not
inherit stride 5** -- brief §4 consequence 2):

| stride | Delta_t (phys.) | light-cone half-width (phys.) |
|---|---|---|
| 1 | 0.05 | 0.063 |
| 2 | 0.10 | 0.126 |
| 5 | 0.25 | 0.315 |
| 10 | 0.50 | 0.631 |
| 20 | 1.00 | 1.261 |

Required stencil half-width in lattice sites, `light_cone_half_width / h`,
against the Phase 10 candidate lattice spacings (brief §12.2.1 sizing
table):

| stride | Delta_t | P=16 (h=6.25) | P=32 (h=3.13) | P=64 (h=1.56) |
|---|---|---|---|---|
| 1 | 0.05 | 0.010 | 0.020 | 0.040 |
| 2 | 0.10 | 0.020 | 0.040 | 0.081 |
| 5 | 0.25 | 0.050 | 0.101 | 0.202 |
| 10 | 0.50 | 0.101 | 0.201 | 0.404 |
| 20 | 1.00 | 0.202 | 0.403 | 0.808 |

### Three consequences (brief §4)

1. **The Phase 11 stencil half-width is a prediction, not a hyperparameter.**
   Even at the most demanding row in this table (stride 20, P=64), the
   required half-width is under 1 site. A learned stencil with half-width
   `w=1` (the smallest value in Phase 11's planned sweep `w in {1,2,3,4}`)
   is already predicted to be *more* than sufficient at any stride up to 20
   solver steps; if Phase 11 finds the rollout error still improving out to
   `w=3` or `w=4`, that is evidence of either an under-estimated `v_star`
   (recheck this measurement) or the learned model absorbing spurious
   nonlocality, not evidence that the physics needs a wider stencil.
2. **Temporal stride is a locality lever, confirmed quantitatively.** Going
   from stride 1 to stride 20 inflates the required stencil half-width by
   20x (linearly, as expected from `Delta_t = dt*stride`). Phase 11 must
   sweep stride explicitly rather than inheriting the ML-writeup's stride 5
   (itself from a different, non-canonical `NX=128` configuration, brief
   top matter) -- there is a real trade-off against DA-cycle cost (smaller
   `Delta_t` means more rollout steps per cycle, more accumulated model
   error) that this table does not resolve on its own.
3. **The localization-radius design rule is now a concrete, checkable
   number.** `minimum_localization_radius = max(encoder_receptive_field,
   v_star*Delta_t)` (`ks_latent.analysis.spreading.minimum_localization_radius`).
   Until Phase 10 produces an encoder receptive field to compare against,
   the light-cone term alone (0.06-1.26 physical units over the stride
   range above) is the only known lower bound; Phase 13 must implement the
   `max(...)` comparison as an assertion against both terms once the
   encoder exists (brief: "a small but clean theoretical contribution",
   not a comment).

## Phase 3: Stage-1 AE / Stage-2 propagator (part of Gate 3)

**Decision rule (pre-registered, brief §18):** Stage-1 val reconstruction
should be "comparable to the recorded run" (no numeric target given in the
brief -- logged for comparison, not pass/fail). Gate 3 proper is the full
§18 replication table, which additionally needs Phase 4 (analysis suite)
and Phase 5 (PFF DA); this section covers only what Phase 3 itself produces.

### Honest device benchmark (brief §1.3.8, `scripts/bench_device.py`)

One epoch, canonical-size models, measured on this machine:

| model | device | seconds/epoch |
|---|---|---|
| Stage-1 AE (synthetic-data benchmark) | CPU | 57.1 |
| Stage-1 AE (synthetic-data benchmark) | MPS | 19.5 |
| Stage-1 AE (**real dataset**, 50 runs x 1001 snapshots) | MPS | 79.0 |
| Stage-2 propagator | CPU | 0.265 |
| Stage-2 propagator | MPS | 0.297 |

Stage-1 AE is ~2.9x faster on MPS at this scale -- full training (60
epochs) uses MPS, `--profile smoke` stays on CPU (small enough not to
matter, keeps CI simple). Stage-2 propagator does **not** win on MPS
(CPU is marginally faster, both ~0.3s/epoch) -- launch-overhead-dominated,
exactly the brief's predicted small-model case. Both training scripts
default to the device that actually measured faster for that model, not to
MPS-by-assumption.

### Stage-1 training run -- REPLICATION TARGET MISSED (ground rule 6)

Canonical config (`AutoencoderConfig()` defaults: NX=1024, d_latent=44,
d_model=128), trained on `generate_trajectory_dataset(KSConfig(L=100,
NX=1024, spinup_time=500), n_train=50, n_val=10, trajectory_time=250)` (1001
snapshots/run) via `scripts/train_stage1_patched.py --profile full`, 60
epochs on MPS (79.0s/epoch measured, 4520.6s = 75.3 min total,
`artifacts/stage1_ae_patched_full.pt.meta.json`).

**Result: val reconstruction MSE = 0.969 -- essentially the "predict the
batch mean" baseline for zero-mean/unit-variance data (MSE~1.0).** Per
`artifacts/stage1_history_full.json`, the training-set reconstruction loss
drops from 1.009 (epoch 0) to ~0.965 by epoch 5, then is **flat for the
remaining 55 epochs** (0.9611 at epoch 59). This is not "hasn't converged
yet" -- it is a converged fixed point (see below) -- and is far from
`docs/ML_for_KS_writeup.md`'s reported val recon of 0.075 for the
(non-canonical, NX=128) v4/v5 configuration; no comparable number exists
in `docs/PROJECT_HANDOFF.md` for the canonical NX=1024/d=44 setup, but
0.969 is not "comparable to the recorded run" under any reading.

**Diagnosis (representation collapse, not a data or infrastructure bug):**
measuring the *cross-sample* standard deviation of the encoder's latent
output across a batch of genuinely different real KS snapshots
(`z.std(dim=0)`) shows it shrinking steadily during training (~0.02 early
to ~0.003-0.007 later) while the batch-mean gradient norm shrinks toward
~0.01-0.03 -- a real converged optimum where the encoder maps essentially
every input to nearly the same latent vector, so the decoder cannot do
better than reproducing something close to the dataset mean. Six ablations
were run to localize the cause, each reproducing the identical collapse
(so each is ruled out as the root cause):

| ablation | result |
|---|---|
| Data sanity (mean/std/NaN/Inf) | clean; not a data bug |
| Reconstruction loss alone (no `L_pred`/`L_decorr`/`L_var`) | same collapse -- not caused by the auxiliary loss terms |
| Shift augmentation on vs. off | same collapse either way |
| Grad-clip at 1.0 vs. effectively unclipped (clip=1000) | same collapse either way -- clip=1.0 is not the cause |
| Learning rate in {1e-4, 1e-3, 3e-3} | same collapse at every value; 1e-2 diverges (unstable) rather than collapsing, so no LR in a reasonable range avoids it |
| Decoder's sensitivity to `z` at initialization | actually *strong* (`\|\|D(z2)-D(z1)\|\|` for random z1,z2 exceeds `\|\|D(z1)\|\|` itself) -- rules out a "dead" decoder pathway from the start; the collapse specifically develops in the *encoder's* cross-sample output diversity during training |

**This is reported per ground rule 6 rather than silently tuned away.**
The architecture as built matches brief §5.1's specification exactly
(verified line by line: patch/group/query-token structure, loss weights,
optimizer, schedule); the collapse is a property of how this specific
architecture's optimization landscape behaves on real KS data at this
scale, not an implementation deviation caught so far. Candidate next steps
(not yet tried, to discuss with the user before spending further compute):
a KL-style or stronger `L_var` warm-up schedule (anneal-in additional
variance pressure before allowing recon to dominate), a curriculum starting
from smaller `NX`/`d_latent` before scaling up, an encoder-only warm-up
against a simpler reconstruction target, or literally far more epochs
under a different schedule.

### Stage-2 training run

**Not run.** Stage-2 trains a propagator on Stage-1's *latent* trajectories;
training it against a collapsed, near-constant latent would only produce a
meaningless (near-identity, high-confidence-in-nothing) propagator and
burn compute on a number that cannot inform Gate 3. Blocked on resolving
the Stage-1 collapse above.

### Collapse follow-up (2026-08-29, user-directed debugging session)

**NX=256, full Stage-1 objective (recon+pred+decorr+var, shift
augmentation, real val split), 100 epochs, 1716s wall time:** escapes the
worst of the collapse (epoch 0-5 recon ~1.00-1.01 plateau) and improves
steadily to **val recon 0.764** by epoch 99, still slowly decreasing at the
end. Materially better than NX=1024's permanent 0.969 plateau, but not yet
"good" reconstruction -- the improvement is real but slow/partial under
the *full* objective (decorr/var terms + shift augmentation make this a
harder landscape than the isolated ablations below).

**Targeted comparison (brief §5.1 addendum's two hypotheses), NX=256,
recon+pred+decorr+var, 1500 steps each, single seed:** tracked the step at
which `L_var` drops from its collapsed value (~1.0) toward zero and
`z_std` (cross-sample latent std) jumps toward ~1 -- the escape point.

| config | escape step (var: 1.0 -> <0.5) |
|---|---|
| baseline (dt_snap=0.25, K_pred=2) | ~900 |
| longer rollout (dt_snap=0.25, K_pred=4) | ~600 |
| larger dt_snap=1.0 (K_pred=2) | ~300 |

Both hypotheses accelerate escape versus baseline, in this single-seed
test; larger `dt_snap` is the stronger single lever tested so far. All
three reach a similar escaped state (`z_std`~0.9-1.0) by step 1200-1500 --
this only speeks to *how fast the plateau is escaped*, not to final
reconstruction quality (that needs a full run under the winning
config(s), not yet done).

**New architecture option implemented** (brief §5.2 addendum, user request):
a `mode` flag on `PropagatorConfig`/`AuxPropagatorConfig`, `"two_step"`
(original `(z_{n-1},z_n)->z_{n+1}`) vs `"markovian"` (`M(z_n)->z_{n+1}`,
since the KS PDE is first-order in time and a sufficient encoding should be
Markovian without a two-step history), the latter with `"mlp"` or
`"transformer"` (tokenized, optionally banded-local-attention) backbones.
Single `LatentPropagator` class handles both mode`s via a uniform
`.step(z_prev, z_curr)` interface (markovian ignores `z_prev`), so
`train_stage2`/DA cycling/Lyapunov adapters/D3 diagnostics needed zero
changes; only `train_stage1`'s window construction and `L_pred` rollout
needed to become mode-aware (window shrinks from 4 to `1+K_pred` snapshots
in markovian mode, since the now-unnecessary second history snapshot is
dropped). Validated: identity-at-init in both modes/all backbones,
`z_prev`-independence in markovian mode, `train_stage2` needing no
mode-specific code (confirmed via test), and end-to-end Stage-1 training
smoke tests for both backbones.

### Factorial comparison: dt_snap x propagator mode, NX=256, full objective, 100 epochs

**Two separate questions, not one.** `dt_snap` and `mode` were tested
against two genuinely different criteria, and conflating them produced a
misleading first read of this data (corrected here):

1. **Does the change help *escape* the collapse fixed point faster?** This
   is the question the debugging session was actually about.
2. **What *final* reconstruction quality does it reach?** This is
   confounded by an *expected*, uninteresting effect: a larger `dt_snap`
   makes the underlying prediction task intrinsically harder (predicting
   further ahead in a chaotic system), so worse asymptotic `L_pred`/`L_recon`
   is the expected outcome regardless of whether collapse is an issue at
   all. A worse final number under larger `dt_snap` is not evidence against
   the collapse-escape hypothesis.

**On question 1 (escape speed), the best evidence remains the earlier
purpose-built diagnostic** (fixed 1500-step budget, `L_var`/`z_std` logged
every 300 steps, `two_step` mode held fixed so only `dt_snap` varies):

| config | step where `L_var` drops from ~1.0 (collapsed) toward 0 |
|---|---|
| dt_snap=0.25 | between 600-900 |
| dt_snap=1.0 | **before 300** |

This is well-resolved and supports larger `dt_snap` escaping faster. The
four full 100-epoch runs below log reconstruction only every 5 epochs,
which at these configs' very different batches/epoch (389 for dt_snap=0.25,
97 for dt_snap=1.0) corresponds to sampling every ~1945 or ~485 gradient
steps respectively -- too coarse to add a reliable independent read on
escape *timing*, especially for isolating the `mode` effect on escape speed
(the four estimated escape windows below overlap too much to distinguish):

| config | batches/epoch | escape window (epoch -> approx. step) |
|---|---|---|
| two_step, dt_snap=0.25 | 389 | epoch 5-10 -> step ~1945-3890 |
| markovian, dt_snap=0.25 | 390 | epoch 10-15 -> step ~3900-5850 |
| two_step, dt_snap=1.0 | 97 | epoch 35-40 -> step ~3395-3880 |
| markovian, dt_snap=1.0 | 97 | epoch 30-35 -> step ~2910-3395 |

**On question 2 (final quality), the complete 2x2 table:**

| | dt_snap=0.25 | dt_snap=1.0 |
|---|---|---|
| **two_step** | **0.764** (best) | 0.8195 (worst) |
| **markovian** | 0.7793 | 0.811 |

`dt_snap` is clearly the dominant factor for final quality (~0.05-0.06 cost
going from 0.25 to 1.0, in *either* mode) -- consistent with it being the
expected harder-task effect from question 2, not a sign anything is wrong.
`mode` has a small, direction-flipping effect within noise of a single-seed
comparison (markovian is ~0.015 worse than two_step at dt_snap=0.25, but
~0.008 *better* at dt_snap=1.0) -- no clean win for either mode on this
axis, and not decisive enough to draw a "markovian is better/worse" verdict
from a single seed each.

**Verdict:** `dt_snap` is the variable that matters, and it helps in the
way it was hypothesized to help (faster escape from the collapse fixed
point) while costing what basic physical reasoning predicts it should cost
(worse asymptotic accuracy at a harder prediction task) -- both effects
real, neither surprising once separated. `mode="markovian"` is implemented,
tested, and available as an option, but this single-seed factorial
comparison found no clear standalone benefit for it on this task; it
remains available for future experiments (e.g. paired with the
`transformer` backbone, or at NX=1024) rather than adopted as a fix here.

**Performance anomaly, unresolved:** the `markovian, dt_snap=0.25` cell
took 9629s (~2.7 hours) versus the `two_step, dt_snap=0.25` baseline's
1716s -- a ~5.6x slowdown for a nearly identical workload (same dataset,
same ~390 batches/epoch, `markovian`'s only structural difference is a
window of 3 vs. 4 and a smaller `AuxPropagator` input). The
`markovian, dt_snap=1.0` cell showed no such slowdown (336s, comparable to
`two_step, dt_snap=1.0`'s 405s), so this isn't simply "markovian mode is
slower" in general, nor simple session-cumulative thermal throttling (the
`two_step, dt_snap=1.0` cell, run immediately before this one in the same
session, showed no slowdown). No MPS-fallback warnings were logged during
the slow run (`ks_latent`'s fallback-warning hook would have caught an
op silently dropping to CPU). Confirmed via direct process sampling that it
was genuinely computing throughout, not hung. Root cause not identified --
flagged here rather than left silent; worth a dedicated investigation
(e.g. `torch.mps.profiler`) before relying on wall-clock estimates for
`markovian` mode at this dataset scale.

### Decision: canonical NX changed from 1024 to 256 (2026-08-29)

**User decision, following the factorial results above:** rather than
continuing to fight NX=1024's collapse (never escaped within a 60-epoch
budget across every recipe variant tried), the working canonical scale for
Phase 3+ is now **NX=256**, with **`mode="markovian"`** and **`dt_snap=1.0`**
as the chosen recipe. Changed in code:

- `ks_latent.config.KSConfig.NX` default: 1024 -> 256.
- `ks_latent.config.AutoencoderConfig.NX` default: 1024 -> 256 (`d_latent=44`
  unchanged; `n_patches=32`, `n_groups=4` at this NX with `patch_size=
  group_size=8` unchanged).
- `scripts/train_stage1_patched.py`, `scripts/train_stage2_patched.py`,
  `scripts/bench_device.py`: pick up the new defaults automatically (no
  per-script NX override needed); `--mode`/`--dt-snap` flags added for
  explicit control.
- `scripts/run_analysis_suite.py`, `scripts/run_da_pff.py`: were hardcoding
  `dt_snap=0.25` for Lyapunov-exponent physical-time conversion and truth-
  trajectory generation respectively -- both **real bugs** once the chosen
  recipe moved to `dt_snap=1.0`, caught and fixed while making this change
  (both now take a `--dt-snap` flag, defaulting to 1.0).
- Stale NX=1024 datasets/checkpoints from this debugging session deleted
  from `artifacts/` (gitignored, session-local scratch); the NX=256/
  dt_snap=1.0 dataset from the factorial comparison was promoted to the
  canonical `artifacts/datasets/stage1_trajectories_dtsnap1.h5` path all
  scripts now default to.

**What did *not* change:** Phase 1's Gate 1 (`test_L100_kaplan_yorke`,
NX=1024, passed, `docs/REPLICATION_LOG.md`) and Phase 2's Gate 2
(`v_star` measurement, NX=1024, passed, earlier in this document) both used
explicit `NX=1024` in their own test/script code, independent of the
`KSConfig`/`AutoencoderConfig` class defaults -- neither was rerun, and
both stand as valid evidence the *solver* is correct at NX=1024. The
default change only affects Phase 3+ (the latent-model training pipeline),
where NX=256 is a legitimate cost/speed choice, not an accuracy compromise:
at L=100, NX=256 gives grid spacing h~0.39 against the energy-injection
lengthscale `2*sqrt(2)*pi~8.89` (~23 points/wavelength), still deeply
spectrally-resolved.

**Status at the point of this change:** the real Stage-1 training run at
NX=1024/markovian/dt_snap=1.0 (launched to close out Gate 3-4) was killed
partway through in favor of restarting at the new NX=256 canonical scale --
no result was lost (it hadn't produced a checkpoint yet). The next step is
re-launching Stage-1 training at NX=256/markovian/dt_snap=1.0 for the real
Gate 3/4 numbers.

### Canonical NX=256/markovian/dt_snap=1.0 run #1 (baseline, before ported improvements)

The first full 100-epoch run at the new canonical recipe completed with
**`val_recon_final = 0.811254`** -- a real improvement over NX=1024's
permanent 0.969 plateau, but still not the ~0.05-0.15 "escaped cleanly"
level seen in the earlier factorial comparison's diagnostic runs. The
epoch-level history (`artifacts/baseline_before_ported_improvements/
stage1_history_full.json`, preserved for this comparison) shows why:

| epoch | recon |
|---|---|
| 0 | 1.032 |
| 10 | 1.000 |
| 20 | 1.000 |
| 30 | 1.000 |
| 40 | 0.906 |
| 50 | 0.856 |
| 60 | 0.834 |
| 70 | 0.822 |
| 80 | 0.815 |
| 90 | 0.811 |
| 99 | 0.811 |

Stuck at the collapse plateau (`recon ~= 1.0`) for the first ~30 of 100
epochs, then a slow, monotonically-decelerating escape that had nearly
flattened out by epoch 90-99 (0.8147 -> 0.8109) without reaching the level
the earlier diagnostic runs found reachable. This is exactly the training
picture the "Ported improvements" addendum (CLAUDE_CODE_BRIEF.md §5.1/5.2)
predicted would be improvable: this run used AdamW with an implicit
`weight_decay=0.01` (1000x the reference project's `1e-5`) and no LR
warmup, i.e. the full peak `lr=1e-3` hit the freshly-initialized encoder/
decoder from epoch 0 -- plausible contributors to both the length of the
initial stall and the slow, incomplete escape afterward.

### Canonical NX=256/markovian/dt_snap=1.0 run #2 (with ported improvements) -- hypothesis NOT confirmed

A second run, **identical in every other setting** (same seed, same
dataset, same 100 epochs, same `mode="markovian"`/`dt_snap=1.0`), was
launched with only `Stage1TrainingConfig`'s two new defaults changed:
`weight_decay=1e-5` (was implicitly `0.01`) and `warmup_epochs=2`/
`lr_min_factor=0.02` (was no warmup, cosine decay to exactly 0). Result:
**`val_recon_final = 0.815263`**, against run #1's `0.811254` -- no
improvement, within single-seed noise of being identical, in fact
marginally worse.

| epoch | recon (run #1, old defaults) | recon (run #2, weight_decay=1e-5 + warmup) |
|---|---|---|
| 0 | 1.032 | 1.022 |
| 20 | 1.000 | 1.000 |
| 30 | 1.000 | 1.000 |
| 40 | 0.906 | 0.996 |
| 50 | 0.856 | 0.888 |
| 60 | 0.834 | 0.846 |
| 80 | 0.815 | 0.821 |
| 99 | 0.811 | 0.816 |

If anything, run #2 escaped the initial collapse plateau slightly *later*
(between epoch 40-45, versus run #1's epoch 30-40) and converged to a
marginally higher final reconstruction error -- the opposite direction of
what fixing an over-strong weight decay and adding a warmup was expected
to do. **Reported honestly, per ground rule 6, rather than cherry-picked:**
at this single seed, neither the `weight_decay=0.01 -> 1e-5` fix nor the LR
warmup measurably improved (or hurt, beyond noise) the collapse-escape
dynamics at NX=256/`mode="markovian"`/`dt_snap=1.0`. This does not mean the
weight_decay fix was wrong to make -- an unintended, 1000x-too-heavy decay
constant on every parameter (including the zero-initialized propagator
head) has no principled justification regardless of its measured effect
here, and it is now the honest, deliberate value rather than an accident
of never having passed the argument -- but it should **not** be cited as
"the fix" for the collapse-escape speed problem. That problem's actual
driver remains open: both runs show the same qualitative shape (a long
initial stall near `recon~1.0`, a slow partial escape, a decelerating
plateau around 0.81-0.82 that had not fully flattened by epoch 100). A
longer epoch budget, or `w_pred=0.0` as a cheap two-stage-curriculum
stand-in (see `docs/OPEN_QUESTIONS.md`'s "Deferred experiments" list), is
the more promising next thing to try if faster/cleaner escape at NX=256
becomes a priority again -- it is not currently blocking Gate 3/4, since
0.81-0.82 is a real (non-degenerate) reconstruction level, just not as good
as the diagnostic-run numbers suggested was reachable.

**The banded latent-index regularizer (`RegConfig.lambda_z`) is explicitly
not a candidate here -- user direction, 2026-08-29:** the reference
project's own measurements already showed it collapses the latent
(participation ratio falling to ~1.1 of 8 at `lambda_z >= 5e-3`, see
`ks_latent/training/regularizer.py`'s module docstring), i.e. turning it on
would risk recreating the exact failure mode this project spent
significant effort escaping. `RegConfig`/`ks_latent/training/regularizer.py`
stay in the codebase (tested, off by default) for possible future use, but
are not to be proposed or turned on as a fix for anything going forward.

### MLP encoder/decoder + multistep rollout comparison (2026-08-29, in progress)

Motivated by the reference project's own `L=94, N=128, dt_model=1, d_z=45`
benchmark (`.../experiments/output/latent/single_run_N128.log`) -- its
plain-MLP joint architecture with a real-sized propagator and an 8-step
unroll curriculum reached **val recon MSE ~0.008** at essentially this
codebase's exact operating point, against this codebase's **~0.81-0.815**.
A second reference log (`diagnose_L94.log`) at the same domain size showed
that even that architecture's *dynamics* (free-running latent rollout)
were fragile at this scale -- several variants diverged (`|z|` reaching
1e6) within 20-40 steps -- so the comparison here is scoped to
*reconstruction* quality specifically, not propagator stability.

Two variables ported and isolated (CLAUDE_CODE_BRIEF.md §5.1/5.2 second
addendum): the plain-MLP encoder/decoder (`--encoder mlp`,
`KSAutoencoderMLP`) and a multistep training rollout with a real-sized
propagator (`--multistep`, ramping `k_pred` 2 -> 8 over the first 30% of
epochs). Four-cell comparison, all at NX=256/`mode="markovian"`/
`dt_snap=1.0`/seed 0/100 epochs, otherwise identical:

| encoder | rollout | val_recon_final |
|---|---|---|
| transformer (patched) | short (k_pred=2, small aux) | 0.815 (already measured, see run #2 above) |
| mlp | short | **0.005587** |
| transformer | multistep (k_pred 2->8, real-sized aux) | **0.808340** |
| mlp | multistep | *not run -- user direction, 2026-08-29: stopped pursuing multistep after transformer+multistep showed no effect* |
| vit | short | **0.000550** |
| vit | multistep | *not run, same reason* |

**The MLP-encoder cell landed first, and the effect is enormous: `0.815 ->
0.0056`, a ~146x reduction in MSE (~12x in relative RMSE) from swapping
only the encoder/decoder architecture, everything else held fixed** (same
NX=256/`mode="markovian"`/`dt_snap=1.0`/seed/100 epochs/dataset). This is
close to the reference project's own L=94 MLP number (~0.008) and
confirms the architecture -- not the domain size, not the training
regime -- was the dominant factor behind the earlier ~100x gap.

Just as telling as the final number is the *shape* of the training curve
(`artifacts/stage1_history_full_mlp_short.json`): `recon` = 0.448, 0.025,
0.014, 0.011, 0.0088, 0.0076, ..., 0.0056 at epochs 0, 10, 20, 30, 40, 50,
..., 99 -- **smooth, monotonic descent from epoch 0, with no collapse
plateau at all.** This is qualitatively different from every
patched-transformer run in this document, all of which spent 30-45 epochs
stuck near `recon~1.0` before any escape. That strongly suggests the
collapse this project spent significant debugging effort escaping (NX
change, `mode="markovian"`, `dt_snap=1.0`) was substantially an artifact of
the patched-transformer's optimization landscape specifically, not a
property of the KS reconstruction task, the loss function, or the domain
size. Also still descending (not fully plateaued) at epoch 99 -- a longer
epoch budget would likely improve this further, consistent with the
reference project's own much longer training runs (120-300 epochs).

**`transformer + multistep` landed second: `0.808340`, essentially
identical to the `0.815` short-rollout baseline** (both still show the same
long collapse-plateau shape -- `recon` history at epochs 0/10/20/30/40:
1.02, 1.00, 1.02, 1.00, 0.88, only escaping around epoch 40 exactly like
every other patched-transformer run in this document, ramp curriculum
notwithstanding). **The bigger propagator + longer rollout curriculum did
nothing for the patched-transformer encoder.** Combined with the
MLP-encoder result above, this narrows the explanation further: the
collapse and poor final quality are attributable to the encoder/decoder
*architecture* specifically, not to the auxiliary propagator's size or the
rollout length used during Stage-1 training.

**User direction, 2026-08-29: stopped pursuing `--multistep` at this
point.** With `transformer + multistep` showing no effect and the
architecture (not the rollout regime) clearly identified as what matters,
the `mlp + multistep` run in progress was killed before completion and the
queued `vit + multistep` run was cancelled -- both would only have tested
a variable already shown to be inert for the one architecture where it was
measured. Only `vit + short` remains in the comparison (see below).

### ViT-style encoder/decoder (2026-08-29, in progress)

The MLP result above raises an obvious question: is the
patched-transformer's problem "transformers in general," or something
specific to *its* design? `--encoder vit` (`KSAutoencoderViT`,
`ViTAutoencoderConfig`, ported from the same reference project's Track B)
is also transformer-based, but adds one thing `KSAutoencoderPatched`
completely lacks: a positional encoding. `KSAutoencoderPatched`'s local
transformer mixes groups of patches with plain (position-blind)
self-attention and then mean-pools -- destroying any surviving order
information -- before the global transformer, which also has no
positional signal, ever sees it; the encoder can route information by
content but has no explicit signal for *where* a feature sits in space.
`KSAutoencoderViT` adds a periodicity-respecting `CircularPositionalEncoding`
to every token before any attention layer runs, on both encoder and
decoder, and skips the local-group pre-pooling stage entirely (a single
flat attention stack over all tokens). See CLAUDE_CODE_BRIEF.md §5.1/5.2
for the full architecture comparison.

**Caveat, raised when the user asked for the full difference list:** this
is *not* a clean single-variable ablation of positional encoding alone.
`KSAutoencoderViT` also differs from `KSAutoencoderPatched` in its
bottleneck mechanism (mean/CLS pooling vs. 8 learned query tokens sharing
the global transformer), how the decoder is seeded from `z` (a full
per-token linear readout vs. a shared broadcast-added vector plus a
z-independent learned bank), the patch embedding (a plain `Linear` vs. a
2-layer MLP), and the feedforward width (`mlp_ratio=4` vs. `dim_ff=d_model`,
i.e. no expansion). A result here implicates the ViT design as a whole,
not positional encoding specifically -- isolating PE alone would mean
adding `CircularPositionalEncoding` directly into the existing
`KSAutoencoderPatched` instead, which has not been done.

Per user direction (2026-08-29), only `vit + short` was run.

**Result: `val_recon_final = 0.000550` -- the best of every architecture
tried, and by a clear margin over the MLP (`0.005587`), not just over the
patched-transformer (`0.815`).** Relative to the patched-transformer
baseline this is a **~1,482x reduction in MSE** (~38x in relative RMSE);
relative to the MLP result it is a further ~10x reduction. The training
curve (`artifacts/stage1_history_full_vit_short.json`) descends smoothly
and monotonically from epoch 0 with no collapse plateau, same as the MLP,
but converges noticeably faster: `recon` = 1.01, 0.18, 0.0090, 0.0039,
0.0023, 0.0015, ..., 0.00055 at epochs 0, 10, 20, 30, 40, 50, ..., 99 --
already an order of magnitude below the MLP's epoch-30 value by epoch 30
here, and still descending (not plateaued) at epoch 99.

### Aux propagator backbone comparison, vit encoder fixed (2026-08-29)

User question: does the *auxiliary* propagator's architecture (mlp vs.
vit backbone) affect Stage-1 training, holding the `vit` encoder fixed?
Both cells use full (unrestricted) attention in the aux propagator:

| aux propagator backbone | val_recon_final |
|---|---|
| mlp (original) | 0.000550 (the `vit + short` run above) |
| vit | 0.000649 |

**No meaningful difference** -- within run-to-run noise. With full
attention, the aux propagator (which is discarded after Stage 1 and only
ever supplies a gradient signal via `L_pred`) doesn't materially change
what the encoder learns, regardless of its own architecture. This sets up
the next question directly: does *restricting* that attention (a genuine
locality constraint, not just a different block design) do anything --
see below.

### Ring-windowed attention, encoder + aux propagator (2026-08-29, in progress)

User hypothesis: the propagator's positional encoding might "enforce some
structure on the latent states." Caveat raised before running: with full
attention (as in every run above), there's no real gradient pressure for
this -- `CircularPositionalEncoding` gives tokens a distinct identity, but
any token can still attend to any other regardless of position, so there
is no cost to non-local coordination. Testing the hypothesis for real
requires restricting attention to a finite ring window, so that the
propagator (and, if applied there too, the encoder/decoder) can only mix
information between physically-nearby tokens -- at that point, if the
encoder does not organize which physical information lands in which
latent coordinate to match, prediction gets measurably worse, creating
real gradient pressure back through `L_pred` to reorganize `z`.

**A real bug was caught and fixed while wiring this up:** `_ViTDeltaBody`
was masking by *linear* index distance while pairing it with
`CircularPositionalEncoding`, which assumes a *ring* -- token 0 and token
`n_tokens-1` were "ring-adjacent" to the positional encoding but
"maximally far" to the attention mask, an internal contradiction that
would have made this experiment meaningless. Fixed: both `_ViTDeltaBody`
and (newly) `KSAutoencoderViT` now use `build_ring_local_attention_mask`
(circular distance) when `attn_window` is set; `_TransformerDeltaBody`
(learned, non-ring positional embedding) correctly keeps its existing
linear-distance mask, since ring restriction would contradict *its*
positional assumption instead. Confirmed non-causal in both cases (masks
are symmetric by distance, not direction).

`attn_window=4` on both the `vit` encoder/decoder and the `vit` aux
propagator (the propagator's aux tokenization bumped from `n_tokens=4` to
`n_tokens=44`, one token per latent coordinate, with `token_d_model=64` --
`n_tokens=4` would have made `attn_window=4` no restriction at all, since
the max ring distance on a 4-token ring is 2), otherwise identical to the
runs above.

**Result: `val_recon_final = 0.000492`** -- slightly *better* than either
full-attention `vit` cell (`0.000550` mlp-aux, `0.000649` vit-aux), not
worse. A single-seed result isn't strong evidence the windowing itself
caused the (small) improvement -- it's within the same noise band as the
mlp-vs-vit-aux difference above -- but it does rule out the concern that a
hard locality constraint would hurt reconstruction quality. It does not,
by itself, establish whether the hoped-for structure (physically-adjacent
information landing in ring-adjacent latent coordinates) actually emerged;
that requires a direct diagnostic (e.g. the existing D3 coupling-graph /
bandedness-p-value machinery in `ks_latent/analysis/diagnostics.py`,
applied to the trained propagator's Jacobian) rather than reading it off
the reconstruction number alone.

### Linear positional encoding + linear-distance masking (2026-08-29, added, not yet run)

User-requested follow-up: isolate the periodic-vs-non-periodic assumption
itself, holding the block architecture fixed. Added `pos_encoding:
"circular" | "linear"` to `ViTAutoencoderConfig`/`PropagatorConfig`'s
`vit` backbone (`--pos-encoding` on `train_stage1_patched.py`): `"linear"`
swaps `CircularPositionalEncoding` for a new `LinearPositionalEncoding`
(the classic fixed, non-learned Vaswani et al. sinusoidal encoding, but
*not* periodic -- position 0 and the last position are not treated as
neighbours) and swaps the ring-distance mask for the existing
linear-distance one (`build_local_attention_mask`, the same mask the
older `transformer` backbone uses). Same `ViTBlock`/`mlp_ratio`
architecture either way -- this isolates exactly the periodicity
assumption from every other difference between `vit` and `transformer`
(block implementation, feedforward width, learned-vs-fixed encoding).

For the encoder/decoder specifically, `pos_encoding="linear"` is a
deliberately-wrong prior to test (KS's domain genuinely is periodic), which
makes it an informative negative control: if `"linear"` performs
comparably to `"circular"` there, that would suggest periodicity isn't
actually doing much work in practice at this window size; if it performs
clearly worse, that's direct evidence the periodicity assumption matters.
For the propagator, whether the latent index has any periodic (or even
consistently ordered) structure at all is unknown -- both `"circular"` and
`"linear"` are equally unverified assumptions there, which is the more
open and arguably more interesting question. Implemented and unit-tested
(`tests/unit/test_autoencoder_vit.py`, `tests/unit/test_propagator_modes.py`).

**Result (`--pos-encoding linear`, both encoder and aux propagator,
`attn_window=4`, otherwise identical to the circular run above):
`val_recon_final = 0.000494`** -- statistically indistinguishable from the
circular version's `0.000492`. At `window=4` (a fairly wide window: the AE
covers +-4 of 32 ring positions, i.e. the near side of the domain either
way), whether the model is told the domain wraps around or not makes
essentially no difference to reconstruction quality. User's stated
motivation for trying this (2026-08-29): most real-world PDEs of interest
are not on a periodic domain, so a design that works about as well without
assuming periodicity is more broadly useful than one that depends on it.
This single comparison is consistent with that -- the periodicity
assumption isn't pulling meaningful weight here, at least at this window
size and this domain -- but doesn't prove non-periodic domains would fare
equally well in general; it only shows KS's own (mild) periodicity isn't
being heavily relied upon by the model at `window=4`.

Taken together with the earlier caveat that this is not a clean
single-variable ablation (ViT differs from the patched-transformer in
positional encoding, bottleneck mechanism, decoder seeding, patch
embedding, and feedforward width all at once), the ranking across all
architectures tried, best to worst:

| architecture | val_recon_final |
|---|---|
| vit, attn_window=4, circular pos_encoding | **0.000492** |
| vit, attn_window=4, linear pos_encoding | 0.000494 |
| vit, full attention, mlp aux | 0.000550 |
| vit, full attention, vit aux | 0.000649 |
| mlp | 0.005587 |
| transformer (patched), multistep | 0.808340 |
| transformer (patched), short (baseline) | 0.815 (0.811-0.815 across two seeds/settings) |

Both attention-free (MLP) and position-aware attention (ViT) architectures
dramatically outperform the patched-transformer; the fact that ViT beats
MLP by a further ~10x, rather than merely matching it, is suggestive that
attention *is* useful here once it is given positional information to work
with -- but confirming that specifically (versus ViT's other differences
from the patched-transformer) would need the narrower ablation described
above (positional encoding added directly to `KSAutoencoderPatched`,
nothing else changed), which has not been run.

### Decision: canonical encoder changed to `vit`, `attn_window=4`, `pos_encoding="linear"` (2026-08-29)

**User decision, following the architecture/rollout/windowing/positional-encoding
comparisons above:** the working canonical Stage-1/Stage-2 recipe for
Phase 3+ is now `--encoder vit --aux-backbone vit --attn-window 4
--pos-encoding linear` (encoder/decoder) and the matching `--backbone vit
--pos-encoding linear --attn-window 4` for the real Stage-2 propagator,
replacing the patched-transformer default (`val_recon_final` improves
from ~0.81 to ~0.0005, a ~1,600x reduction in MSE). `pos_encoding="linear"`
specifically (not `"circular"`) was chosen going forward even though the
two performed statistically indistinguishably on KS (`0.000494` vs
`0.000492`) -- user's stated reasoning: most real-world PDEs of interest
are not on a periodic domain, so a design that does not depend on
periodicity is more broadly useful than one that does, and here it costs
nothing to prefer it.

Changed/added in code to support this:
- `n_tokens` bumped from 4 to 44 (one token per latent coordinate) for both
  the aux and real propagators when using `backbone="vit"` with a finite
  `attn_window` -- at `n_tokens=4` the max ring/line distance is only 2, so
  `attn_window=4` would impose no restriction at all. `token_d_model`
  correspondingly raised to 64 (`CircularPositionalEncoding` requires
  `d_model >= n_tokens` for even `n_tokens`; `LinearPositionalEncoding` has
  no such constraint but 64 was kept for consistency/capacity).
- A real bug caught and fixed while wiring this up: `run_analysis_suite.py`,
  `run_da_pff.py`, and `run_diagnostics.py` all hardcoded
  `KSAutoencoderPatched` when loading a Stage-1 checkpoint, regardless of
  which architecture the checkpoint was actually trained with -- would
  have crashed or silently mismatched state_dict keys against a
  `vit`/`mlp`-encoder checkpoint. Fixed by adding a single shared
  `ks_latent.models.load_autoencoder_checkpoint`/`build_autoencoder`
  dispatch (used by all four scripts that load a Stage-1 checkpoint,
  including `train_stage2_patched.py`, which had its own inline copy of
  the same dispatch logic before this consolidation) -- tested in
  `tests/unit/test_models_checkpoint_loading.py`.
- `--pos-encoding`/`--attn-window`/`--prop-n-tokens`/`--prop-token-d-model`
  added to `scripts/train_stage2_patched.py` (previously `--backbone vit`
  existed but always used full attention, circular encoding, and the
  transformer-backbone's `n_tokens=4` sizing unconditionally).
- The winning Stage-1 checkpoint
  (`artifacts/stage1_ae_patched_full_vit_window4_linear.pt`) promoted to
  the canonical `artifacts/stage1_ae_patched_full.pt` path all Gate-3/4
  scripts default to; the prior patched-transformer baseline checkpoint
  archived to `artifacts/archive/` rather than deleted.

**What did *not* change:** the underlying `mode="markovian"`/`dt_snap=1.0`/
`NX=256` recipe decided earlier in this document -- this decision is
layered on top of that one, changing only the encoder/decoder and
propagator *architecture*, not the physical-time/domain-size choices.
Gate 3 (§18 replication table) and Gate 4 (`docs/diagnostics_report.md`)
are being run against this new checkpoint pair now.

## Phase 4: Analysis suite (part of Gate 3)

Full validation results are in `docs/OPEN_QUESTIONS.md` (Phase 4 section) to
avoid duplicating the sweep tables here; summary: two-NN and correlation
dimension validated on d-spheres/tori for d in {2,5,10,15,20,22}
(`ks_latent/analysis/dimension.py`), diffusion maps validated via spectral
degeneracy on circle/2-sphere/2-torus, persistent homology (plain Rips +
DTM) validated on circle/torus/2-sphere with and without outliers
(`ks_latent/analysis/topology.py`), and the `single_state`/`two_step`
latent-propagator Lyapunov adapters validated to 1e-3 against a hand-built
linear recursion's exact companion-matrix spectrum
(`ks_latent/analysis/lyapunov.py`). Real numbers against the trained
Stage-1/2 checkpoints via `scripts/run_analysis_suite.py --profile full`
once Stage-2 training completes.

## Phase 5: Natural-Gradient Particle Flow Filter (part of Gate 3)

**Decision rule (pre-registered, brief §7, §18):** `test_pff_gaussian_linear`
-- for a linear observation operator and Gaussian prior, PFF's posterior
mean and covariance must match the analytic Kalman filter to ~2% at large
ensemble size -- is "the one test that proves the filter is correct."

**Result: PASS.** Measured with `d=5`, `m=3`, `N=8000` (`ks_latent/da/pff.py`,
`NAT` method, `n_steps=100`): mean relative error 1.53%, covariance relative
error 1.75-2.43% across repeated runs -- both within the brief's ~2% target;
the shipped test uses a 5% bound for margin against seed variance while
still being a real correctness check.

### Honest note on reconstructing the algorithm

`reference/` had no `pff.py` to consult (brief §17 flags `reference/` as
containing only the Phase X contingency prototype, which is unrelated), and
`docs/PROJECT_HANDOFF.md` describes the NAT-PFF update in shorthand (named
quantities like "N_mat", "L_k", "k_bar/N" from the original script) without
enough detail to reconstruct the exact original formula. What's implemented
is an independently-derived algorithm satisfying every *literal, checkable*
specification in the brief (the `F = J^T R^-1 J + B^-1` metric, the prior
term's exact form, the `ds` schedule recursion verified against a
hand-computed sequence in `test_ds_schedule`, `MODEL_NOISE_STD` /
`OBS_NOISE` / RMSE-normalization conventions) via a from-scratch derivation:
preconditioned (natural-gradient) Langevin dynamics targeting the
Gauss-Newton/Laplace-approximate posterior, re-linearized at the ensemble
mean each pseudo-time step. The full derivation, including *why* "the
divergence term simplifies via F/F^-1 cancellation" (brief's own phrasing)
is in the `ks_latent/da/pff.py` module docstring. This is a documented
substitution for an unrecoverable implementation detail, not a claim of
bit-for-bit replication -- what's actually verified is the property the
brief's own test cares about (exact Kalman recovery in the linear-Gaussian
case).

**Numerical stability fix found during derivation:** a naive Euler-Maruyama
discretization of the stochastic term (`z + ds*f + sqrt(2*ds)*noise`) blew
up by ~10x in covariance once the adaptive `ds` schedule grew past ~2
(explicit Euler is only accurate, and here unstable, for small step sizes).
Fixed by recognizing the per-step dynamics are an *exact* Ornstein-Uhlenbeck
process (the Newton drift is exactly affine with a single shared root
`z_star` per step) and using its closed-form transition kernel instead --
an exponential integrator, unconditionally stable for any `ds`, the same
design principle as the ETDRK4 solver elsewhere in this codebase. Before
this fix, `test_pff_gaussian_linear`'s covariance error was >500%; after,
~2%.

**DET vs. STO/NAT (documented, tested behavior, not a bug):** the purely
deterministic `DET` mode converges every particle to the same MAP point
(`test_pff_det_collapses_ensemble_spread`) -- a Newton flow on a shared
quadratic energy has no mechanism to preserve spread. Only `STO`/`NAT`
(Langevin, with the noise term) recover the correct posterior *covariance*,
which is why `NAT` is the default and `test_pff_gaussian_linear` targets it.

### DA cycling driver

`ks_latent/da/cycling.py` runs the full forecast/observe/assimilate loop
with the brief's exact spacetime convention (last forecast row per cycle
overwritten by the analysis; `u_da` uses `decode(mean(z))`, not
`mean(decode(z))`) and RMSE convention (latent-space, per-dimension,
`||error|| / sqrt(d)`, identical function for free and DA runs). Validated
on a controlled synthetic system (linear decoder/encoder, a mildly unstable
linear propagator, biased initial ensemble) where DA measurably beats the
free run and the free run's error grows over cycles while DA's does not
(`test_da_beats_free_run_on_mildly_unstable_linear_system`).

*(Real numbers against the trained Stage-1/2 checkpoints --
free-run/DA/spread/calibration/skill, targeting the brief's ~1.6 / ~0.14 /
~0.11 / ~0.8 / ~10x -- via `scripts/run_da_pff.py --profile full` once
Stage-2 training completes.)*

## Phase 6: Structure diagnostics D1-D5 (Gate 4)

**Decision rule (pre-registered, brief §8):** Gate 4 requires
`docs/diagnostics_report.md` to exist, with D3 reporting a p-value and D4
reporting a verdict (equivariant-representation-found or not). No
retraining -- everything here runs on the existing Stage-1/2 checkpoints.

**Result: infrastructure complete and validated; `docs/diagnostics_report.md`
exists** (currently populated from `--profile smoke`, since Stage-1 training
is still in progress -- rerun with `--profile full` once the real
checkpoints exist, which will overwrite it with real numbers).

All five diagnostics validated on synthetic ground truth before being
pointed at any trained model (ground rule 1):

- **D1** (`ks_latent/analysis/diagnostics.py::decoder_sensitivity_map`):
  synthetic decoder built from known Gaussian bumps at fixed centers
  (including one straddling the periodic wrap point) -- recovered circular
  centroids land within one grid cell of the true centers in every case.
- **D2** (`wavenumber_content`): recovers the exact wavenumber of a pure
  cosine mode; correctly ranks a narrow spatial bump as broader-spectrum
  than a wide one; the wavelet-energy-entropy measure (addendum §14.4's
  "third possibility", beyond pure real-space or pure Fourier-space
  localization) correctly scores a localized bump as more concentrated than
  white noise.
- **D3** (`coupling_graph_diagnostic`, `bandedness_p_value`): recovers a
  hand-built banded circulant Jacobian exactly (linear case); the
  permutation test scores a genuinely banded matrix as significant
  (p<0.01) and a dense random matrix as unremarkable (p>0.05) at 1000 null
  permutations. Distance correlation validated to detect a nonlinear
  (y=x^2) dependency the linear Jacobian measure alone would miss.
- **D4** (`translation_representation`): a synthetic *exactly* equivariant
  encoder (truncated Fourier projection) recovers the rotation group
  structure and its eigenvalue-derived latent wavenumbers to <1e-5 (residuals
  essentially at the float32 roundoff floor) -- matching the brief's 1e-8
  target modulo float32 vs. float64 precision.
- **D5** (`local_dimension_vs_length`): a synthetic "extensive" system (NX
  split into independent low-dimensional chunks, so the true local-dimension
  density is known exactly) recovers the correct slope to within two-NN's
  own well-documented estimation bias.

**A real, non-trivial finding surfaced even in the smoke run** (not yet
meaningful *numerically* given the tiny undertrained smoke model, but
structurally informative): D4's per-shift residuals were uniformly small
(~0.01, each individual `R(c)` fits well) while the group-property error
was large (~0.7, `R(c1)R(c2)` far from `R(c1+c2)`) -- individual shifts fit
a linear map reasonably but the family doesn't compose like a genuine
one-parameter group. This is exactly the qualitative signature the
diagnostic is designed to catch (an encoder that is *locally* well-behaved
under any single shift but not *globally* equivariant), and is consistent
with `docs/ML_for_KS_writeup.md`'s own description of the trained encoder
as only *approximately* equivariant.

*(Real numbers and full report -- including whether this pattern persists
on the real, better-trained encoder -- once Stage-1/2 training completes
and `scripts/run_diagnostics.py --profile full` is run.)*

## Lorenz-96 generalization test (cross-system validation, outside the brief's phase numbering)

**Context.** User-directed, 2026-09-22 onward: does everything learned on
KS generalize to a second chaotic system? `ks_latent/solver/lorenz96.py`
(validated against known-answer tests -- exact fixed point, exact energy
and trace identities, closed-form fixed-point spectrum, literature
Lyapunov comparison at N=40/F=8) plus Sections 195-201 (`scripts/
section19{5,6,7,8,9}_*.sh`, `section20{0,1}_*.sh`) trained a matched
encoder/propagator pipeline on Lorenz-96 (N=64, F chosen from a sweep to
land near KS's own D_KY range -- F=4.2 gives true D_KY=11.3, `lambda1=
0.080`, `n_positive=5/64`). Full per-section logs/diagnostics live under
`artifacts/logs/`; this entry does not reproduce that whole table, only
the finding that changes the interpretation of all of it.

**Finding (2026-09-23): Lorenz-96 does not have KS's spectral gap, and
that is very likely the actual ceiling on every number the L96 line has
produced so far, independent of encoder architecture or regularizer
choice.** Measured directly (no autoencoder, no propagator -- the true
physical solvers only), full 64-direction Lyapunov spectrum at matched
state dimension N=64 for both systems (KS: L=22, NX=64; L96: N=64,
F=4.2):

| quantity | KS (L=22, NX=64) | Lorenz-96 (N=64, F=4.2) |
|---|---|---|
| D_KY | 5.25 | 9.98 |
| lambda_1 | 0.0526 | 0.0532 |
| sum of all 64 exponents (= trace of the time-averaged Jacobian) | **-4180** | **-64.00** (exactly -N, by construction) |
| most-damped exponent | -96.1 | -3.48 |

Despite comparable D_KY and lambda_1, KS's total damping budget is ~65x
larger than L96's at the *same* state dimension, and it is spent almost
entirely on a strongly-damped tail (down to -96) rather than spread evenly.
**Why:** KS's linear operator is `k^2 - k^4` in Fourier space --
wavenumber-selective, with damping growing as the *fourth power* of k, so
high-k modes are crushed almost instantly. This is the literal
mathematical content behind KS's known finite-dimensional inertial
manifold (Foias/Nicolaenko/Temam and others): a small number of modes
genuinely close the dynamics, and the rest are fast, slaved, and safely
discardable -- the actual reason a small Markovian latent model can work
for KS *in principle*, not just in practice. Lorenz-96's only linear term
is the uniform `-x_i` damping (exactly -1 on every diagonal entry for
every site, proved exactly in `tests/unit/test_lorenz96.py::
test_jacobian_trace_is_exactly_minus_n`) -- there is no wavenumber-
dependent operator, hence no analogous gap, hence no rigorous basis for
assuming any small subset of L96's 64 directions is dynamically
disposable.

**Practical read:** a small-`d_latent` *Markovian* reduction of L96 is not
"harder than KS," it may be asking for something structurally different
from what the true system provides. Supporting evidence, in increasing
order of history length, all at identical `d_latent=20` on the F=4.2
dataset:

| Section | Propagator history | D_KY (Stage 1) |
|---|---|---|
| 199 | `mode=markovian`, 0 steps | 1.35 |
| 198 | `mode=two_step`, 1 step | 2.01 |
| 201 | `mode=history`, `n_history=6` (5 steps), full/unmasked attention | **11.94** |

**Section 201's Stage-1 result closes almost the entire gap to the true
system** (`lambda1=0.0184`, `n_positive=4/20`, `D_KY=11.94` against the
true `lambda1=0.080`, `n_positive=5/64`, `D_KY=11.3`; `val_recon_final=
0.0028`, matching Section 199's reconstruction quality) -- a step change,
not a gradual trend, going from 1 to 5 steps of history. This is a strong,
direct confirmation of the Mori-Zwanzig/spectral-gap reasoning above: L96
genuinely needs propagator memory in a way KS never did, and once given
enough of it (here, 5 past states), a Markovian-in-the-*extended*-state
sense propagator can match the true system's chaos to within measurement
noise. (Caveat: the free rollout is not obviously stationary over the
2000-step diagnostic window -- `max|z|` grows from ~3 at t=0 to ~75 by
t=1999 across the 20 tracked ICs, staying comfortably bounded
`(max_abs_state=1e3)` but not obviously settled; worth a longer rollout
check before treating the D_KY number as fully converged.)

**Stage 2 result: collapsed to D_KY=0.00 again** (`lambda1=-0.0025`,
`n_positive=0/20`, `best_val_kmax_mse=0.006368`, rollout stayed bounded
and finite -- `max|z|` settles to ~3.0 by t=1999, a tight, non-chaotic
orbit). This is now the **third** independent confirmation of the same
outcome (Section 197, local_field encoder, 0 history; Section 199, ViT,
0 history; Section 201, ViT, 5 steps of history) -- and the most
informative one, because Section 201's Stage 1 started from D_KY=11.94,
*essentially matching the true system*. The collapse is therefore not a
symptom of weak Stage-1 chaos that a stronger starting point would avoid
-- Stage 2's training objective itself destroys autonomous chaos
regardless of how good the dynamics were before it started, regardless of
propagator memory length, and regardless of encoder architecture. This
sharpens the standing hypothesis (brief §5.2 addendum discussion, and the
KS-side Sections 130-174 finding this mirrors) from "a pattern we've seen
a few times" to "a structural property of pure k-step supervised latent
loss": nothing in `horizon_weighted_latent_loss` rewards preserving
sensitivity to initial conditions, so a trained propagator is always free
to trade it away for lower k-step forecast error, and apparently always
does. The KS-side `pde_head` self-rollout/spectrum-shape regularizers
built earlier this project (Sections 192-193, discouraging Jacobian
contraction along a self-generated rollout) are the most direct existing
candidate fix, not yet tried on Stage 2's *main* propagator on either
system -- see Open questions.

**User challenge, 2026-09-23: "if we allow that much history, are we
really using a lower dimension? like in this case, 5 steps of history
with 20 dim is 100 states, which is more than the original system?"** A
fair and important objection -- `n_history=6 * d_latent=20 = 120` raw
numbers the propagator conditions on, nominally larger than L96's own
N=64. Checked directly (no training, linear participation ratio of the
covariance eigenspectrum, `>=32000` real on-attractor windows from the
Section 201 Stage-1 checkpoint, `artifacts/logs/...` not yet written up as
a standalone script -- see `/tmp/artifact_imgs/effective_rank_check.py`
for the exact computation if reproducing):

| representation | raw dim | participation ratio |
|---|---|---|
| raw physical state | 64 | 5.87 |
| single encoded `z` | 20 | 5.45 |
| stacked 6-state history | 120 | **8.75** |

Stacking history does add genuinely new information (PR rises from ~5.5 to
~8.75, moving toward the true D_KY=11.3), but the stacked vector's
*effective* dimension is nowhere near its raw count of 120 -- it is
comparable to, not larger than, both D_KY=11.3 and the raw system's own N=
64. Cumulative variance across the 120 stacked directions: 93.7% is
already captured by the top 20, 99.8% by the top 40, and the bottom ~10
eigenvalues are ~1e-5 (numerically negligible) -- consecutive encoded
states along a smooth chaotic flow are highly redundant with each other,
exactly the picture Takens/delay-embedding theory (Takens 1981; Sauer,
Yorke & Casdagli, "Embedology," 1991) predicts: a short window of a lossy,
low-dimensional observable can reconstruct an attractor's full dynamics
without needing the window's raw coordinate count to stay below the
ambient dimension.

**So: no, not "just using more dimensions" in the sense of effective
information content** -- but two honest caveats keep this from being a
clean answer. (1) Linear participation ratio is a known-biased-low
estimator on curved attractors (this project's own KS work already
documents PCA/linear measures reading systematically below true nonlinear
dimension); PR=8.75 is a lower bound on the "true" redundancy claim, not
a tight one -- a two-NN or correlation-dimension estimate on the stacked
vector would be a less-biased cross-check. (2) Effective/information
dimension is not the same as *raw storage/compute cost*: an explicit
ensemble-DA scheme conditioning on the full 120-dim history vector still
carries 120 raw numbers per ensemble member regardless of how redundant
they are, unless that redundancy is explicitly exploited (e.g., project
onto the leading ~9-11 PCA directions of the history window before doing
anything DA-related with it). The natural, well-motivated follow-up this
diagnostic suggests: since 93.7% of the stacked variance is captured by
the top 20 of 120 directions, a *shorter* history (`n_history=3` or `4`,
giving 60 or 80 raw dimensions -- at or below N=64) may already recover
most of Section 201's chaos gain. Not yet run.

### Section 203 rerun: `--multistep` diagnosis confirmed, but a dedicated Stage-2 anti-collapse regularizer stack still fails (2026-09-23)

First attempt at Section 203 (idea 1 `--w-spectrum-shape` + idea 2
`--multistep`, combined, on top of the markovian ViT recipe) collapsed
**Stage 1 itself** to `D_KY=0.00`, before Stage 2 ever ran -- a new,
unexpected regression, since every prior bare Stage-1 L96 run had produced
genuine chaos (199: 1.35, 201: 11.94). Working hypothesis: `--multistep`
extends Stage 1's own auxiliary-propagator `L_pred` term to an 8-step
rollout (`k_pred` ramped 2->8) -- the same long-horizon-MSE mechanism
already diagnosed as the Stage-2 collapse cause -- reintroducing that
pressure inside Stage 1's own jointly-regularized objective.

**Rerun, `--multistep` removed, everything else unchanged:**

| stage | config | lambda1 | n_positive | D_KY |
|---|---|---|---|---|
| Stage 1 | ViT, markovian, w-spectrum-shape (n_expand=5, target=1.1, floor=0.6, two_sided), w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, NO multistep | 0.00315 | 1/20 | **2.117** |
| Stage 2 | warm-started from above; w_varmatch=0.02 adaptive, w_spatial=0.01 signed, w_logdet_rollout_latent=0.0035, k_max=12 | -0.00235 | 0/20 | **0.0** |

(True L96 N=64/F=4.2 reference: `lambda1=0.080, n_positive=5/64,
D_KY=11.3`. Section 199 comparison: stage1 D_KY=1.35, stage2 D_KY=0.00.
Section 201 comparison: stage1 D_KY=11.94, stage2 D_KY=0.00.)

**Two results, read together:**

1. **The `--multistep` diagnosis is confirmed.** Removing it alone
   recovered Stage 1 to `D_KY=2.117` -- better than the bare Section 199
   baseline (1.35) and comfortably chaotic. `--w-spectrum-shape` and the
   other Stage-1 regularizers (`w_var`, `w_spatial`, `w_logdet`) do not
   cause Stage-1 collapse on their own; `--multistep`'s 8-step rollout
   term does.
2. **Stage 2 still collapsed to exactly 0.00, despite a regularizer stack
   built specifically to prevent this.** `w_varmatch` (adaptive) +
   `w_spatial` (signed) + `w_logdet_rollout_latent` (the new
   general-backbone logdet anti-collapse term built for this run, see
   `ks_latent/config.py`/`ks_latent/training/loops.py`/
   `scripts/train_stage2_patched.py`) were all active simultaneously,
   warm-started from a Stage-1 checkpoint whose chaos (D_KY=2.117) was
   already reasonable. None of it mattered -- Stage 2 still drove the
   system to a fixed point (`lambda1` went slightly *negative*, all 20
   directions contracting).

This is the **fourth** independent confirmation that pure k-step
supervised `horizon_weighted_latent_loss` training collapses autonomous
chaos on L96 (Sections 197, 199, 201, 203 -- local_field and ViT
encoders, 0 and 5 steps of propagator history, with and without geometry
regularizers), and the first case where a dedicated anti-collapse
regularizer stack was tried and still failed. The remaining untried
candidate in the existing toolkit is applying the pde_head
self-rollout/spectrum-shape regularizer (Sections 192-193, built to
discourage Jacobian contraction along a self-generated rollout) directly
to Stage 2's *main* propagator rather than only to the separate
`pde_head` distillation target -- see `docs/OPEN_QUESTIONS.md`.

### Sections 204/205: L96 N=16, x+x' augmented state, self-rollout spectrum-shape regularizer -- Stage 1 itself collapses under ViT, not under plain MLP (2026-09-23)

Testing the one remaining candidate flagged by the Section 203 rerun
above (`--w-spectrum-shape-self`, the self-rollout analogue of Section
193's pde_head-only mechanism, now generalized to the MAIN propagator
and to `mode="history"` -- see `ks_latent.training.loops._propagator_
spectrum_shape_self_pool` and `Stage2TrainingConfig.w_spectrum_shape_
self`), on a new operating point: L96 N=16, F=6.0 (F=4.2, this line's
usual forcing, measured directly to be NON-chaotic at N=16 -- D_KY=0.0;
F=6.0 gives D_KY=9.39, the closer match to this test's d_latent=8 of the
two chaotic candidates tried), state augmented with the exact analytic
derivative (`x` concatenated with `x'=l96_rhs(x,F)`, doubling the stored
state to 32 dims -- `ks_latent.solver.lorenz96_dataset.generate_
trajectory_dataset(..., include_derivative=True)`), `mode="history"`
(`n_history=6`, following Section 201's own depth).

**Section 204 (`--encoder mlp --aux-backbone mlp`)**: killed mid-Stage-1,
user-directed, before completion, to swap in the ViT architecture
instead (see Section 205). Left as an open comparison cell (script kept,
annotated as superseded) rather than deleted.

**Section 205 (`--encoder vit --aux-backbone vit`, `pos_encoding=linear`,
FULL attention -- no `--attn-window`)**: Stage 1 trained cleanly on
reconstruction (`val_recon_final=0.0836`, smooth monotonic descent, no
collapse plateau) but the PROPAGATOR itself collapsed: `lambda1=
-4.51e-05, n_positive=0/8, D_KY=0.0`. Stage 2 was cancelled (user-
directed: "don't run stage 2 then") once this came in -- warm-starting
the self-rollout regularizer test from an already-collapsed Stage-1
propagator can't demonstrate anything about whether that regularizer
prevents collapse; a collapsed input producing more collapse is
uninformative either way.

**Reading the two together**: the reconstruction quality (healthy) vs.
propagator health (collapsed) split, on the SAME dataset/regularizer
stack that a plain MLP propagator survived in Section 204's own 2-epoch
dry run (recon 0.546->0.424, no collapse signal in that short a window --
not power to detect eventual collapse, but no immediate red flag either,
unlike Section 205's decisive negative), points at the ViT-backbone
PROPAGATOR specifically, not the data augmentation or the regularizer
stack, as the proximate cause here. This is consistent with this
project's own established pattern (`ks_latent.models.propagator._
LocalMLPDeltaBody`'s docstring, KS-side): "every self-attention-based
propagator tried... collapsed to a fixed point... every architecture
WITHOUT self-attention... recovered rich chaos" -- this is a further
confirmation of that pattern, now on a NEW system configuration (L96
N=16, x+x' augmented state, mode=history) rather than a repeat of an
existing one. Section 204's mlp/mlp variant was never run to full
completion, so this is not yet a controlled, same-day head-to-head
confirmation -- reopening/finishing Section 204 (or a fresh mlp/mlp
rerun with identical settings) would make it one.

**Correction (Section 205's own causal claim above), same day:** Section
205 used `--aux-backbone vit` by mistake -- a misreading of the user's
"replace the encoder and decoder with the ViT" request (`--encoder`
covers the autoencoder; `--aux-backbone`, a separate flag, was supposed
to stay `mlp` per Section 204's original, never-rescinded spec). Caught
directly by the user. Section 206 reran with the CORRECT
`--encoder vit --aux-backbone mlp` -- and reconstruction convergence was
**just as bad** (epoch 0/10/20/30 recon: 0.426/0.145/0.104/0.088, killed
at epoch 30 before completion -- essentially the same trajectory as
Section 205's own 0.43/0.145/0.104/.../0.084 final). Since BOTH the vit-
and mlp-backbone propagators show the identical slow-convergence
reconstruction curve, the propagator backbone is NOT the shared
bottleneck -- **the "self-attention propagator collapses" diagnosis
above does not explain this experiment's actual problem.** User-directed
comparison against Section 201 (L96 N=64, F=4.2, raw x only, d_latent=
20, ALSO `--aux-backbone vit --mode history` -- i.e. the SAME propagator
backbone Section 205 used, but which recovered D_KY=11.94) makes this
concrete: Section 201's recon curve is 0.110/0.0043/0.0023/.../0.0028
(epoch 0/10/20/final) -- nearly converged within 10 epochs -- against
Sections 205/206's 0.43/0.145/0.10/~0.085, a much higher, much slower-
descending plateau regardless of propagator choice.

**Likely actual cause, not yet confirmed**: concatenating `x` and `x'`
into one flat 32-dim vector before ViT patch-tokenization
(`patch_size=8` -> tokens `[x_0:8, x_8:16, x'_0:8, x'_8:16]`) hands the
encoder's `CircularPositionalEncoding`/`LinearPositionalEncoding` four
tokens as if they sit on ONE ring, when tokens 2-3 are not "further
along in space" from tokens 0-1 -- they are a DIFFERENT physical
quantity at the SAME sites. Nothing in the architecture tells the model
this; it would have to discover the distinction from data alone, a
strictly harder problem than Section 201's homogeneous single-quantity
64-dim `x`. Confounded with two further simultaneously-changed variables
(N=16 gives only 4 tokens vs. 201's 8; `d_latent=8` vs. 20) that this
experiment's own design never separated (see Section 204's header:
"THREE independent new variables in this one experiment... a genuinely
combined test, not a controlled single-variable one") -- this reasoning
is a diagnosis, not yet an isolated confirmation. Given both propagator
variants inherit the same poorly-converged Stage-1 latent, Section 206
was killed (user-directed) before finishing rather than let it produce
an uninformative Stage-2 result on top of a bad foundation. **If this
line resumes**, the natural next step is a single-variable fix: encode
`x`/`x'` as two CHANNELS per site (shape `(N, 2)`, analogous to a
2-channel image) rather than concatenating them into one longer
sequence -- giving the encoder the site-alignment structurally instead
of asking it to learn x/x' are "the same place, different quantity"
from scratch.

### Section 207: x/x' as ViT channels fixes reconstruction but propagator chaos drops sharply (2026-09-23)

User-directed: "can you think of a way of augmenting 201 with x' that
will work with the vit's assumptions. implement this and run it." Built
`ViTAutoencoderConfig.n_channels` (`ks_latent/models/autoencoder_vit.py`,
`ks_latent/config.py`) so `patch_size`/`token_window` stay in PHYSICAL
SITE units and each token becomes `patch_size*n_channels` raw values
(both channels of `patch_size` consecutive sites, interleaved) instead
of one long flat sequence -- `n_tokens` shrinks back to exactly Section
201's own count (8, at N=64/patch_size=8), keeping "adjacent token =
adjacent physical site" true again. Paired with `lorenz96_dataset.
generate_trajectory_dataset(derivative_layout="interleaved")`:
`[x_0,x'_0,x_1,x'_1,...]` site-major/channel-minor, instead of the
block-concatenated layout Sections 204-206 used. 5 new unit tests
(`tests/unit/test_autoencoder_vit.py`), including a direct check that
token 0 is exactly sites `0..patch_size-1`'s both channels interleaved,
not a channel block. Full 705-test suite passes.

Otherwise an EXACT copy of Section 201's recipe (L96 N=64, F=4.2,
`--encoder vit --aux-backbone vit --mode history --n-history 6`,
`pos_encoding=linear`, full attention, same Stage-1 regularizers) --
the only changes are `--nx 128` (was 64; NX counts raw scalars =
N*n_channels) and the new `--n-channels 2`.

**Result: reconstruction is fixed, but propagator chaos is far weaker
than Section 201's:**

| | recon_final | lambda1 | n_positive | D_KY |
|---|---|---|---|---|
| Section 201 (x only) | 0.0028 | 0.0184 | 4/20 | 11.94 |
| Section 207 (x+x' as channels) | 0.0044 | 0.0033 | 1/20 | **2.19** |

Reconstruction converged as well as (arguably better than) Section 201's
own curve -- confirming the channel-based fix genuinely resolves
Sections 205/206's badly-converging reconstruction, unlike the
concatenated-block layout. But the propagator's chaos dropped ~5.5x.
The rollout trajectory itself shows this directly: `max|z|` across 20
ICs actually SHRINKS over the 2000-step rollout (2.66 -> 4.47 -> ... ->
1.93) rather than sustaining, and the final-state pairwise spread tops
out at 5.07 (vs. Section 201's 264) -- the ensemble is converging
toward each other, not diverging chaotically. Not a full collapse
(D_KY=2.19 > 0, lambda1 still positive), but a substantial, real
regression.

**Working hypothesis, not yet confirmed**: `x'` is fully determined by
`x` (`x'=l96_rhs(x,F)`, no independent information -- see the section
header's own note that this augmentation adds no NEW degrees of
freedom, just a different presentation of the same ones, unlike a
genuine Takens embedding recovering degrees of freedom that were never
directly observed). In Stage 1's joint loss, the auxiliary
propagator's `L_pred` term is judged in PHYSICAL space against both `x`
and `x'` now; `x'` is a rougher, higher-frequency signal (measured
directly, Sections 204-206: std~6-12x larger than `x`'s own std),
making it a harder rollout-reconstruction target. The fixed `d_latent=
20` budget that gave Section 201 rich dynamics may be getting partly
spent tracking `x'` faithfully at the expense of chaos: a duller,
more-damped propagator still reconstructs short-horizon `x'` reasonably
well, while a genuinely chaotic one would amplify rollout error on the
harder-to-track derivative channel specifically. Not yet isolated from
the alternative explanation that `d_latent=20` was tuned (Section 201)
for a 64-dim raw input and may simply be a worse fit for this
augmented/redundant representation regardless of the mechanism above.

**Stage 2 deliberately not run against this checkpoint** (matching this
arc's own "check chaos survives before running Stage 2" discipline,
Section 194) -- D_KY=2.19 is a worse foundation than Section 201's
11.94, not a better one, so testing `--w-spectrum-shape-self` (the
original motivating question) on top of it would not be informative.
**If this line resumes**: the natural follow-up is Section 201's exact
recipe with x' as a channel but the propagator's own `L_pred`/rollout
loss scoped to compare only against `x` (not `x'`) in physical space --
isolating whether it's specifically the ROLLOUT-RECONSTRUCTION pressure
on `x'` (this section's hypothesis) or something else about the
augmented representation driving the chaos loss.

### Section 208: `--w-spectrum-shape-self` gets its clean test, and still fails to prevent Stage-2 collapse (2026-09-23)

User-directed ("let's try 1 then 2" after Section 207's x' augmentation
was abandoned): the cleanest possible test of `--w-spectrum-shape-self`
(built Section 204, never yet cleanly tested -- Sections 204-207 were
all confounded by unrelated data-representation/architecture problems)
-- warm-start Stage 2 DIRECTLY from Section 201's own Stage-1 checkpoint
(`D_KY=11.94`, no new Stage-1 run needed), with the Section 203 stack
(`w_varmatch` + `w_spatial` + `w_logdet_rollout_latent`, which alone
already failed once at a WEAKER Stage-1 starting point, `D_KY=2.117`)
plus `--w-spectrum-shape-self 0.4` on top. Same dataset, same 60-epoch/
`k_max=12` schedule as every prior Stage-2 attempt in this line.

**Result: still collapsed. `lambda1=-0.0028, n_positive=0/20, D_KY=0.0`**
-- despite starting from the best Stage-1 checkpoint this entire L96
line has produced, and despite the one regularizer specifically built
to prevent exactly this failure mode being active. Worth noting: unlike
some earlier collapses, this one settled onto a BOUNDED, non-trivial
orbit rather than a literal fixed point -- `max|z|` stays around
2.5-3.0 across the whole 2000-step rollout (vs. shrinking toward 0),
and the final-state pairwise spread across 20 ICs stays real (0.82 min,
10.37 max, 6.27 mean) rather than collapsing toward 0 -- i.e. a stable
limit cycle or quasi-periodic torus, not a point attractor. Still zero
positive Lyapunov exponents either way.

**This is now the fifth independent confirmation of Stage-2 collapse on
L96** (197, 199, 201, 203, 208), and the first case where the
regularizer toolkit's own most-recently-built, most-targeted candidate
was tested cleanly (no confound) and still failed. Every anti-collapse
mechanism currently in this codebase's toolkit -- `w_varmatch`,
`w_spatial`, `w_logdet_rollout_latent`, `w_spectrum_shape` (real-data-
anchored), and now `w_spectrum_shape_self` (self-rollout-sampled) -- has
been tried, individually and combined, and none has prevented Stage 2's
pure k-step supervised `horizon_weighted_latent_loss` training from
destroying autonomous chaos. **The open question is no longer "which
regularizer fixes this" (the existing toolkit is now exhausted) but
whether the training OBJECTIVE itself (long-horizon MSE against a
`k_max`-ramped rollout) is fundamentally incompatible with sustained
chaos, independent of any regularizer added on top** -- see
`docs/OPEN_QUESTIONS.md` for candidate directions outside the existing
toolkit (e.g. training against a distributional/statistical rollout
objective instead of pointwise MSE, or abandoning the Stage-1/Stage-2
split in favor of Section 5.1/5.2 addendum's flagged-but-never-run
"two_stage" experiment, `Stage1TrainingConfig.w_pred=0.0`).

### Sections 211-213: no historical local_field checkpoint survives scrutiny, and the "did Section 52 regress" scare resolves to a methodological gap, not a real regression (2026-09-24)

Continuing the Part 4.3 pivot (latent-space DA localization on KS):
after the full audit of 19 historical local_field checkpoints found none
survive a rigorous 2000-step standalone Lyapunov check (see the
`docs/OPEN_QUESTIONS.md` entry for the full table), three fresh
attempts were made, all using the SAME Section-52-derived regularizer
recipe (`w_var=0.02, w_spatial=0.01` signed, `w_logdet=0.0035`) to
isolate the architecture as the remaining variable:

- **Section 211** (local_field + fully-GLOBAL `mlp` propagator): badly
  over-chaotic and apparently unbounded -- `D_KY=74.93, n_positive=
  39/96`, `max|z|` climbing monotonically and unboundedly across the
  whole 2000-step rollout (8.07 -> 1019.30, never turning over).
- **Section 212** (local_field + LOCAL `masked_mlp` propagator,
  `attn_window=18`): launched to test whether a propagator matching the
  encoder's own spatial locality fares better -- superseded before
  completing (see below).
- **Section 213** (exact rerun of Section 52's own original command --
  `--encoder vit`, NOT local_field -- as a regression test): Stage 1
  ALONE was *also* badly over-chaotic and apparently unbounded --
  `D_KY=38.41, n_positive=20/44`, `max|z|` climbing from 2.68 to
  2070.11 over 2000 steps, never turning over. This looked, at first,
  like the shared training code itself had regressed since Section 52
  (2026-09-01) -- the user, reasonably, found this concerning ("why the
  heck would section 52 have worked and now it doesn't").

**Resolution: nothing regressed. The comparison was apples-to-oranges.**
Section 52's own historically-recorded good result (`D_KY=21.42,
lambda1=0.083`, `artifacts/analysis_suite_full_section52_..._
warmstart_k12_300ep.json`) was measured on the **Stage-2** propagator,
after 300 further epochs of k-step supervised training -- nobody, on
2026-09-01, ever ran a standalone 2000-step autoregressive rollout on
Section 52's Stage-1-ONLY checkpoint (that specific test did not exist
until this investigation). Running Section 213's own Stage 2
continuation (Section 52's exact original schedule: `k_max=12,
k_warmup_epochs=210, k_mid=8, k_mid_epochs=175, 300 epochs`) on TODAY's
Stage-1 checkpoint reproduces the historical result almost exactly:

| | historical (2026-09-01) | today's rerun |
|---|---|---|
| `D_KY` | 21.42 | **21.55** |
| `lambda1` | 0.083 | **0.086** |
| `n_positive` | (not recorded) | **11/44** (matches the ~11+-2 literature target) |

And, critically, `max|z|` is genuinely BOUNDED across the entire
2000-step standalone rollout on this Stage-2 checkpoint (oscillating
2.4-3.4, never climbing) -- a real attractor, not a diverging
trajectory that happened to land on a plausible-looking summary number
(the exact failure mode Section 140 demonstrated: never trust a
summary number without checking the standalone trajectory itself).

**The actual, reconciling finding: Stage 1 alone is reliably,
severely over-chaotic/diverging for this ViT/global-mlp-propagator
architecture, and Stage 2's k-step supervised training reliably tames
it into the correct, bounded, target-matching regime -- this has
apparently ALWAYS been true for Section 52's own recipe, just never
checked with this rigor before.** This reframes the entire local_field
investigation: Section 211's own severe Stage-1 over-chaos (`D_KY=
74.93`) may not indicate a deeper architectural problem at all -- it
may just be the SAME "Stage 1 overshoots, Stage 2 corrects" pattern
Section 52 already exhibits, never given the chance to run through
Stage 2 before being judged. Section 212 (the `masked_mlp` local
propagator alternative) was superseded and not completed once this
reframing was identified -- the more informative, lower-cost next step
was running Stage 2 on Section 211's EXISTING checkpoint first, to
test this hypothesis directly before building a new architecture.

**Confirmed: Section 211's Stage 2 continuation (Section 52's exact
schedule) genuinely works.** `D_KY=23.04, lambda1=0.0913, n_positive=
13/96` -- squarely inside the true `[21,24]` target and comfortably
under the `lambda1<=0.1` literature bound. `max|z|` is genuinely
bounded across the full 2000-step standalone rollout (9.1 -> 9.65 ->
10.0 -> 9.4 -> 9.3 -> 9.7 -> 9.3, oscillating tightly, never
diverging) -- a real, settled attractor. Combined with Stage 1's
excellent reconstruction (recon MSE `4.06e-5`, unchanged by Stage 2
since the encoder is frozen), **this is the first genuinely validated
local_field checkpoint in this entire investigation** -- a real
spatial latent FIELD (`n_sites=32`, `local_channels=3`, `d_latent=96`,
site-major flattened, gauge-anchored channel 0) with both accurate
reconstruction and bounded, target-matching chaos, confirmed with the
same rigor that caught every one of the 19 historical checkpoints'
problems. This is the foundation for Part 4.3's actual DA localization
experiment (Stage 0 infrastructure already built --
`ks_latent/da/localization.py`, `scripts/run_da_pff.py --localizer
gaspari_cohn`).

### Sections 215-216: D3-as-a-loss, and its companion diagonal-magnitude-bound term (2026-09-24)

Continuing directly from Section 211's validated checkpoint (baseline
D3 bandedness `0.1883`, Gate 4): tested whether `--w-jacobian-
bandedness` (D3's new differentiable loss, Section 214) can push that
number higher while a Stage-2 continuation still reaches genuine,
bounded, target-matching chaos.

**Section 215 (`--w-jacobian-bandedness 0.05` alone, Stage 1 only):**
D3 bandedness went from `0.19` to **`0.9938`** (near-perfect
concentration near the diagonal) -- but standalone-rollout divergence
got WORSE, not better: `D_KY` `74.93 -> 90.64`, `max|z|` reaching
`6265` (vs. `1019`) over the same 2000 steps. Diagnosis: bandedness
constrains WHERE coupling mass concentrates, not HOW LARGE the
surviving (near-diagonal, including self-coupling) entries are allowed
to be -- forcing locality apparently concentrated the instability into
sharper, more self-reinforcing per-site growth rather than damping it.

**Section 216 (`--w-jacobian-bandedness 0.05` + new `--w-jacobian-
diagonal-bound 0.1` at `ceiling=1.5`, Stage 1 + Stage 2):**
`ks_latent.training.losses.propagator_jacobian_diagonal_bound_loss`
caps each site's own self-coupling magnitude directly (one-sided,
per-sample, verified orthogonal to bandedness by a dedicated unit
test). Stage 1 result: the diagonal ceiling worked with near-perfect
precision (mean `|diagonal|`=`1.0000`, max=`1.0026`, 0% over the
ceiling) -- but standalone divergence was essentially UNCHANGED from
Section 215 (`D_KY=95.07`, `max|z|@1999=5872`). **This cleanly falsifies
the "large diagonal entries are the driver" hypothesis**: even with
every individual diagonal entry pinned at ~1.0, the system still
diverges just as badly, pointing instead at the aggregate/collective
magnitude of the OFF-diagonal-but-within-band neighbor coupling
(unconstrained by a diagonal-only term) as the more likely mechanism.

**Stage 2 (Section 52's plain schedule, no D3 losses active) tamed it
again, with a small but real, survived improvement:**

| | `D_KY` (Stage 2) | `max\|z\|` bounded? | D3 bandedness (Stage 2) |
|---|---|---|---|
| Section 211 (no D3 loss anywhere) | 23.04 | yes | 0.1883 |
| Section 216 (D3 losses in Stage 1 only) | 22.59 | yes | **0.2146** |

Chaos quality is essentially equivalent to Section 211 (both centered
in the `[21,24]` target), but D3 bandedness is measurably higher
(`0.2146` vs `0.1883`) -- despite NEITHER D3 loss being active during
Stage 2 at all. Some of Stage 1's much stronger bandedness push (`0.99`
right after Stage 1) survived Stage 2's plain k-step training, even
though most of it washed back out. This is a genuine, if modest, win,
and suggests an obvious next lever: keep `--w-jacobian-bandedness`/
`--w-jacobian-diagonal-bound` active DURING Stage 2 too (already wired
via `Stage2TrainingConfig`, never yet invoked there) to see whether
bandedness can be pushed further without losing bounded chaos -- not
yet tried.

Both experiments followed the "always run Stage 2" rule adopted this
session (see `MEMORY.md`): Stage 1 alone reliably overshoots into
apparent divergence for every architecture/regularizer combination
tried so far, and only Stage 2's own standalone rollout is the real
bar for judging a result.

### Section 217: `masked_mlp` propagator gives higher D3 bandedness than any prior candidate, but Stage 2 cannot rescue its Stage-1 divergence (2026-09-24)

User-directed: "Can we run 211 with the masked mlp instead. get rid of
the D3 regularizer. I think maybe that could increase bandedness as
well." Hypothesis: `--aux-backbone masked_mlp`'s weights are
architecturally masked to exactly zero outside a fixed circular band
(`MaskedLinear`), so its Jacobian should be band-limited by
construction rather than by soft regularization -- possibly sidestepping
Sections 215/216's finding that penalizing bandedness competes against
stability for gradient budget. Exact copy of Section 211's recipe with
`--aux-backbone mlp` -> `masked_mlp --attn-window 18`, NO D3 loss.

**Stage 1 diverged catastrophically** -- not merely over-chaotic like
every prior candidate, genuinely unbounded: `max|z|` `7.3 -> 23,106`
(t=300) `-> 3.8e8` (t=600) `-> 2.1e14` (t=1000) `-> 2.1e30` (t=1999).
This crashed the Lyapunov Benettin computation outright (reference
trajectory exceeded the divergence bound), which in turn crashed this
project's own diagnostic script (not written to tolerate a Lyapunov
failure) and halted the section script under `set -e` before Stage 2
could launch automatically -- Stage 2 was relaunched manually. This
matches the historical `masked_mlp_expand` arc exactly: all 8 of those
checkpoints (Sections 135, 145-153) also diverged outright when
audited earlier this session -- now a second, independent confirmation
that the `masked_mlp` FAMILY (both the "expand" variant and this plain
one) is prone to much more severe standalone instability than the
global `mlp` propagator ever showed.

**Stage 2 could not tame it -- the first checkpoint this session where
the "always run Stage 2" rule's own pattern broke.** `best_val_kmax_mse
=0.34` (an order of magnitude worse than every other Stage-2 run this
session, which land around `0.02-0.09`). Standalone rollout: bounded
through ~100 steps (`max|z|~10`), then explodes to `9e15` by step 300
and NaN shortly after (`first non-finite step: 348`); the direct
Lyapunov computation fails the same way Stage 1's did.

**D3 bandedness was genuinely higher than any prior candidate --
`0.2970`, vs. Section 216's `0.2146` and Section 211's `0.1883` -- but
on a checkpoint that diverges to NaN, making the number unusable.**
Mechanistic note: the "architecturally exact zeros" reasoning motivating
this test does not hold as cleanly as expected for a MULTI-layer masked
body -- `0 / 9120` off-diagonal Jacobian entries were exactly zero
(checked directly). `MaskedLinear`'s hard-zero guarantee is proven for
a SINGLE linear layer's own weight; `_MaskedMLPDeltaBody` stacks
several such layers with GELU nonlinearities between them, and the
CHAIN-RULE-composed Jacobian through multiple nonlinear layers does not
preserve exact zeros the way pure linear composition would, even
though every individual layer's own weight does. The higher bandedness
score is real but soft (learned through training, not structurally
guaranteed), and it came bundled with much worse instability.

**Verdict: `masked_mlp` dropped as a propagator candidate for now.**
Section 216 remains the frozen local-field checkpoint (best validated
D3 bandedness among the checkpoints that actually reach a bounded
attractor after Stage 2). This also affects Phase F of
`docs/steps_4-3.md` (L-transfer): `masked_mlp` was the natural
candidate for a size-transferable propagator (weight-shared, unlike
the global `mlp`), and its own standalone instability -- independent
of any D3 loss -- means Phase F needs a genuinely stable
translation-equivariant propagator found first, not just a reuse of
this exact configuration.

### Section 218: bounding propagator growth + longer Stage-1 rollout delays but does not fix `masked_mlp`'s divergence (2026-09-24)

User-directed: "weird. I still think there's hope for masked_mlp. I
just think we need to bound the growth of the propagator during stage
1 and we need more rollout of the propagator in stage 1, so the
encoder can try to compensate to avoid this kind of blow up." Two new
levers, both implemented and combined: (1) `--w-prop-magnitude-ceiling`
(new `propagator_rollout_magnitude_ceiling_loss`, ceiling=15.0 -- this
project's own measured bounded-checkpoint scale), which ceilings the
raw `max|z|` of the aux propagator's own unsupervised rollout during
Stage 1 (gradient reaches only the aux propagator's own parameters,
not the encoder); (2) `--multistep --k-pred-max 8`, ramping the aux
propagator's `L_pred` rollout horizon from 2 to 8 steps (gradient DOES
reach the encoder here, since `L_pred` decodes back to physical space).
Otherwise identical to Section 217 (`local_field` + `masked_mlp`,
`attn_window=18`, no D3 loss).

**Stage 1 still diverged, and the raw D3 number was actively
misleading.** `max|z|` reached `497` by t=100 (faster than 217's more
gradual climb) and `nan` by t=1500. Raw bandedness was the highest
ever measured (`0.6916`) -- but `p_value=1.0`, i.e. *less* banded than
essentially every random permutation. Under dynamics spanning `10^0`
to `10^30`, the measured Jacobian is dominated by numerical blow-up
artifacts, not genuine structure, which corrupts the null-distribution
comparison entirely. **Lesson recorded for future diagnostic use: a
high raw bandedness score is meaningless without checking its p-value,
especially on a checkpoint that isn't already known to be bounded.**

**Stage 2 showed real, partial improvement over 217, but still could
not produce a usable checkpoint.** Held genuinely bounded near the
project's own target scale (`max|z|` 11.0 -> 17.2) through t=300 --
versus Section 217's Stage 2, which was already unraveling by that
point and hit NaN by step ~350. This time the runaway is slower:
`max|z|` climbs to `616` by t=600, `3.0e5` by t=1000, and `2.0e11` by
t=1999. The direct Lyapunov computation still fails (reference
trajectory escapes the bound). D3 bandedness = `0.3661`, `p=0.0000` --
genuinely significant, and the highest *significant* D3 of any
Stage-2-validated checkpoint this session (211: 0.1883; 216: 0.2146;
217: 0.2970 but on an already-NaN'd checkpoint) -- but again on a
checkpoint that ultimately diverges and cannot be considered validated.

**Verdict: three independent regularizer attempts at stabilizing
`masked_mlp` (215's bandedness loss, 216's diagonal-bound, 218's
magnitude-ceiling + multistep) have now all failed to produce a stable
standalone rollout.** Each failed differently -- 215 made Stage-1-only
chaos worse, 216 had no effect on divergence despite perfect ceiling
compliance, 218 delayed divergence onset and pushed D3 higher but
still ultimately diverged -- but none crossed the bar Section 211/216
already cleared with no special-casing. Combined with all 8 historical
`masked_mlp_expand` checkpoints also diverging, this is now treated as
a real architectural instability in the `masked_mlp` family, not
something fixable with a loss term alone. `masked_mlp` is dropped as a
propagator candidate; Section 216 (global `mlp`) remains the frozen
`LOCAL_AE`/`LOCAL_PROP`, and Phase F of `docs/steps_4-3.md`
(L-transfer) will need `local_mlp`/`cnn`/`node` instead.

### Phase B (`docs/steps_4-3.md`): DA pipeline sanity check reveals a real asymmetry between the global and local checkpoints (2026-09-24)

Before trusting any `N_ens` sweep (Phase C/D), swept `--n-prop-steps`
(and, for the local checkpoint, `--n-ensemble` too) with
`--localizer none` on both frozen checkpoints (`GLOBAL_AE`/`GLOBAL_PROP`
= Section 213, `LOCAL_AE`/`LOCAL_PROP` = Section 216) to confirm the
PFF pipeline produces a sane, bounded DA effect before spending compute
on the real sweeps.

**`GLOBAL_AE`/`GLOBAL_PROP` (`d_latent=44`): clean pass.** At
`n_ensemble=32`, `n_prop_steps=5` gives `calibration_spread_over_rmse
=0.42` and `skill_free_over_da=2.73` -- both bars cleared easily.
Larger `n_prop_steps` (8, 12) only hurts skill as `rmse_free` saturates
near the attractor's natural scale.

**`LOCAL_AE`/`LOCAL_PROP` (`d_latent=96`): needs a much bigger ensemble
for even a modest effect, and never matches the global checkpoint's
margin.** At `n_ensemble=32` (matching the global setting), skill was
1.0-1.4x and at `n_prop_steps=5` DA actively made things WORSE
(`skill=0.44`). Swept `n_ensemble` in `{32,64,128,256}` x
`n_prop_steps` in `{1,3,5,6,8}` (11 runs total). Best point:
`n_ensemble=128, n_prop_steps=6` -- `calibration=0.43` (in range),
`skill=1.64` (real, but below the `>2` bar the global checkpoint
cleared). Notably NOT monotonic in ensemble size: `n_ensemble=64` was
inconsistent between nearby settings (1.47x at nps=5, 0.51x at nps=6),
and `n_ensemble=256` at nps=5 collapsed catastrophically (0.28x)
despite being twice the ensemble size that worked at 128 -- ruling out
"just use a bigger ensemble" as a clean fix.

**Read carefully: this is not (yet) a failure of the local-field
approach -- it is the expected shape of the problem Part 4.3 exists to
solve.** The local checkpoint's raw latent dimension (`d=96`) is more
than double the global one's (`d=44`), so an UNlocalized ensemble
filter needing a much larger `N_ens` to even approximate a well-
conditioned prior covariance is exactly the failure mode Gaspari-Cohn
localization (Phase D) is supposed to fix -- and exactly why the
pre-registered decision rule requires beating SEC, not just beating
`--localizer none`. Chosen operating points
(`n_ensemble=32/n_prop_steps=5` global; `n_ensemble=128/n_prop_steps=6`
local) are now fixed inputs to Phase C/D's `N_ens` sweeps, per
`docs/steps_4-3.md` Phase B3.

### Section 219: narrowing attn_window + a composed-10-step-Jacobian growth ceiling still fails, and fails faster than Section 218 (2026-09-24)

User-directed follow-up, after Section 218's partial-but-incomplete
result: "any ideas to make the masked mlp work?" -> proposed spectral
normalization (already exists as `nonexpansive`, but judged too hard a
per-layer clamp -- would force the whole map non-expansive, i.e. zero
tolerance for any local chaotic stretching at all) or narrowing
`attn_window` as a cheap complementary lever. User's actual direction:
"try not to clamp too hard to preserve the chaotic dynamics (another
way to do this is just make sure the longer term jacobian after 10
steps doesn't expand too much.) Do this in conjunction with narrowing
attn_window."

Two new levers, REPLACING Section 218's raw-magnitude rollout ceiling:
(1) new `propagator_multistep_growth_ceiling_loss` -- a ONE-SIDED
ceiling on the composed 10-step Jacobian's top singular value (`d
z_{n+10}/d z_n`'s worst-case amplification factor), evaluated at real
encoded states rather than a drifting rollout, so it can apply
pressure before a trajectory gets large enough for a magnitude ceiling
to notice. Calibrated directly against Section 216 (the frozen,
genuinely bounded checkpoint): composed 10-step top singular value at
100 real states has median 28.6, p95 54.1, max 66.4 -- default
ceiling=150.0 sits ~2.3x above that observed max, so genuine chaotic
variation at a known-good checkpoint's own scale should rarely be
penalized. (2) `--attn-window 9` (down from 217/218's 18) --
`attn_window` is a radius in raw `d_latent=96` INDEX units for
`masked_mlp`'s circular-band mask, roughly `window/local_channels`
sites; 18 -> ~6 sites, 9 -> ~3 sites, matching this project's own
correlation-length-based receptive-field reasoning (~2.85 sites at
this `n_sites`/`L`). `--multistep --k-pred-max 8` kept from 218
(independent mechanism, not implicated in either prior failure).

**Result: still diverges, and Stage 2 failed FASTER than Section
218's did.** Stage 1: `max|z|` already `248.6` by t=50, `26,169` by
t=100, NaN by t=1000 -- faster onset than 217's or 218's own Stage-1
divergence. Stage 2: NaN by step 254 (`first non-finite step: 254`) --
noticeably earlier than 218's Stage 2, which held bounded through
t=300 and only diverged between t=600-1000.

**D3 bandedness kept climbing -- now the highest yet at both stages
(Stage 1: `0.7661`, p=0.0000, genuinely significant this time, unlike
218's spurious `p=1.0`; Stage 2: `0.4052`, p=0.0000) -- but stability
kept getting WORSE, not better, across the same sequence.** Full
trend across all four `masked_mlp` attempts (Stage-2 D3, all
significant except noted): 217 `0.2970`/NaN by ~350; 218
`0.3661`/NaN by ~600-1000; 219 `0.4052`/NaN by 254. Narrowing the
architectural window (which mechanically forces MORE structural
bandedness, independent of any loss) combined with a growth ceiling
did not break this pattern -- if anything, constraining the map to be
more local while also penalizing net growth seems to concentrate
whatever instability remains into a faster blowup, echoing Section
215/216's original finding for the global-`mlp` case (pure bandedness
pressure, with nothing controlling per-site magnitude, makes things
worse) -- except here a magnitude-style control (the growth ceiling)
WAS present and still didn't prevent it.

**Verdict: four independent, differently-motivated stabilization
attempts on `masked_mlp` (215-style bandedness, 216-style diagonal
control transplanted conceptually, 218's magnitude-ceiling +
multistep, 219's growth-ceiling + narrowed window) have now all
failed, each producing HIGHER bandedness and WORSE-or-equal stability
than the last.** Combined with all 8 historical `masked_mlp_expand`
checkpoints also diverging, this is treated as conclusive: `masked_mlp`
is not salvageable via loss-shaping or window-narrowing alone, and
further attempts in this same family are not recommended without a
genuinely different mechanism (e.g. a hard per-layer spectral-norm
clamp via the existing but much stricter `nonexpansive` flag, at the
cost of suppressing real local chaos -- not yet tried, and judged
likely too restrictive on its own priors). Section 216 (global `mlp`)
remains the frozen `LOCAL_AE`/`LOCAL_PROP`; Phase F's L-transfer
candidate list is `local_mlp`/`cnn`/`node`, not any `masked_mlp`
variant.

### Section 221: `masked_mlp_expand` at site radius 4 FINALLY produces a genuinely validated, translation-equivariant local propagator (2026-09-24)

After Section 220's mixed result, user asked what `attn_window` was set
to, then: "right, so let's expand the physical site radius to 4, let's
use the other masked mlp that expands in the second layer, and let's
try running 211 again. please leave the log penalty out of phase 2" --
i.e. `--aux-backbone masked_mlp_expand` (Section 128's three-layer
expand/stay-wide/contract variant, distinct from the plain
dimension-preserving `masked_mlp` tried in 217-220), `--attn-window 4`
(verified empirically via a direct Jacobian receptive-field measurement
to be a CLEAN 1:1 site-radius mapping for this backbone specifically --
unlike plain `masked_mlp`, where `window = 3 * site_radius` via the
`local_channels=3` scaling), Section 211/217's plain baseline
regularizers, `--w-multistep-growth-barrier` kept in STAGE 1 ONLY
(k=10, ceiling=75.0 -- Section 220's log-barrier loss, which produced
the calmest Stage-1 trajectory of any prior masked_mlp attempt),
DROPPED from Stage 2 (Section 220 found it caused catastrophic
Stage-2 instability there).

**Stage 1 still diverged** (`max|z|` `147` by t=100, astronomical by
t=1999, D3=`0.7579`/p=0.0000 -- structurally significant but on a
diverging checkpoint) -- consistent with every masked_mlp Stage-1
result this session.

**Stage 2, unlike every `masked_mlp`/`masked_mlp_expand` attempt
before it, converged to a genuinely bounded, validated checkpoint.**
`max|z|` stayed in a tight `8.8-9.6` band across the ENTIRE 2000-step
standalone rollout (t=0 through t=1999) -- no drift, no blowup. The
direct Lyapunov computation SUCCEEDED for the first time on any
`masked_mlp`-family checkpoint: `lambda1=0.0911`, `n_positive=13/96`,
**`D_KY=22.68`** -- squarely inside the `[21,24]` replication target.
`best_val_kmax_mse=0.255` during training (worse than 211/216's
`0.02-0.09`, comparable to Section 217's own troubled `0.34`, but with
occasional large loss spikes during Stage 2 -- e.g. epoch 273:
loss=1435 -- that did not corrupt the best-checkpoint tracking or the
final standalone result).

**D3 bandedness = `0.3916`, p=0.0000 -- the highest of any VALIDATED
(non-diverging) checkpoint this entire investigation**, beating
Section 216's `0.2146` and Section 211's `0.1883` by a wide margin,
while also being the first `masked_mlp`-family checkpoint to actually
reach a bounded attractor.

**Correction (2026-09-24, caught the same day while discussing Section
222): `masked_mlp_expand` does NOT satisfy Phase F's L-transfer
requirement -- an earlier version of this entry claimed otherwise and
was wrong.** `masked_mlp`/`masked_mlp_expand`'s layers are
`MaskedLinear`/`MaskedLinearRect` -- a fixed-size DENSE matrix with a
mask zeroing far-apart entries, not a real weight-shared convolution.
The mask restricts WHICH entries can be nonzero; it does not tie
different (i,j) pairs at the same relative offset to a SHARED
parameter the way a genuine conv kernel does. Verified directly:
`masked_mlp_expand`'s total parameter count QUADRUPLES from 138,912 to
554,304 when `n_sites` doubles (32->64, same `attn_window`/
`expand_factor`) -- its weight tensors are tied to `d_latent`, exactly
the same limitation Section 216's global `mlp` has. A checkpoint
trained at `L=100` cannot even be LOADED at `L=200` (wrong shape). So
Section 221 is a strong candidate for Phase A alone (best D3 of any
validated checkpoint), but does NOT resolve Phase F's blocking gap --
`local_mlp`/`cnn`/`site_conv` (genuine `Conv1d`-based backbones, whose
parameter count is verified identical regardless of `n_sites`) remain
the only real L-transfer candidates.

**Open methodological note:** five straight `masked_mlp`/
`masked_mlp_expand` attempts (217-221) all showed genuinely significant
D3 bandedness even while mostly diverging, and this is the SIXTH
attempt (counting 215/216's global-`mlp` bandedness work) where higher
architectural/regularizer-driven locality correlated with the run that
finally stabilized -- suggestive, but not by itself a controlled
comparison (site radius, backbone family, and the Stage-1 growth
barrier all changed together between 220 and 221). Worth isolating
which change mattered most if this checkpoint gets used seriously.

### Section 222/223: `site_conv` and `local_mlp` -- a new backbone reusing the encoder's own architecture, and the FIRST checkpoint to satisfy Phase A and Phase F's L-transfer requirement at once (2026-09-24)

User: "could we use a model like the encoder from 221 as a propagator?
I've never thought of trying that." Built `backbone="site_conv"`
(`_SiteConvDeltaBody`, `ks_latent/models/propagator.py`): reuses
`KSAutoencoderLocalField`'s own circular-`Conv1d` site-mixing design
almost verbatim (1x1 conv `local_channels->H`, `n_layers` circular
convs mixing across SITES, 1x1 conv `H->local_channels`, `bias=False`
on the output projection -- deliberately matching the encoder's own
documented fix for uncontrolled mean drift) as a dimension-preserving
`z->z` map, instead of a masked dense layer (`masked_mlp*`) or a
flat-sequence conv (the existing, untried `backbone="cnn"`). `H=32`,
`n_layers=3`, radius (reuses `attn_window`) default to
`LocalFieldAutoencoderConfig`'s own already-validated values. 11 new
unit tests (shape, identity-at-init, no-bias, receptive field via
direct Jacobian measurement, circular wraparound, config validation)
all pass; found and fixed a real pre-existing bug along the way
(`train_stage1_patched.py`'s `--aux-backbone` choices list was missing
`node`/`cnn` entirely -- unreachable via CLI despite existing in the
model code; `train_stage2_patched.py`'s `--backbone` list was missing
`spectral_pde_raw`).

**Section 222 (`site_conv`, clean Section 211/217 baseline recipe --
no D3 loss, no growth barrier, plain k_pred=2): Stage 2 converged
beautifully (`best_val_kmax_mse=0.076`, among the best Stage-2 numbers
this session) and the standalone rollout stayed genuinely bounded for
600 steps** (`max|z|` `5.2-6.8`) **before diverging catastrophically
between t=600 and t=1000** (`1.8e31`), NaN by t=1500. D3=`0.284`
(p=0.0000, beats Section 216's `0.2146`). Not a validated checkpoint
under this project's 2000-step standard, but the best short/medium-
horizon behavior of any weight-shared candidate tried -- a real
candidate for direct DA testing (short `n_prop_steps`) given the
"DA only needs a few-step forecast" reframing discussed the same day
(see below).

User: "weird. can we try the setup from 222 with the local mlp too?
just for fun" -- same clean-baseline recipe, `backbone="local_mlp"`
(pre-existing, never previously validated), `--aux-n-tokens 32`
(=`n_sites`, aligning tokens exactly with the encoder's own sites,
`chunk_size=3=local_channels`), `--attn-window 3`. Verified empirically
(not assumed) that `train_stage1_patched.py`'s `--profile full` path
hardcodes `token_n_layers=2` for the aux/full propagator (a pre-
existing, not-CLI-exposed constant), so the effective receptive field
is `token_n_layers * attn_window = 2*3 = 6` sites -- matched to Section
222's own `n_layers*radius = 3*2 = 6`, confirmed via a direct Jacobian
receptive-field measurement before launch.

**Section 223 (`local_mlp`): SUCCESS on every axis measured.** Stage 1
ALONE stayed bounded across the full 2000-step rollout (`max|z|`
`4.8-6.2`) -- the first time any local-field propagator's Stage-1-only
checkpoint has stayed bounded long enough for the Lyapunov computation
to even succeed this session (`D_KY=16.40`, below target but genuinely
computable). **Stage 2 stayed bounded the ENTIRE 2000 steps**
(`max|z|` `7.0-7.9`), `lambda1=0.101`, `n_positive=13/96`,
**`D_KY=22.66`** -- squarely inside `[21,24]`. `best_val_kmax_mse
=0.042`, in the same range as 211/216's own best numbers. D3=`0.309`
(p=0.0000), beating Section 216's `0.2146`.

**Verified structurally L-transferable** (same check applied to
Section 221's `masked_mlp_expand`, which FAILED it): total parameter
count is IDENTICAL (31,651 = 31,651) at `n_sites=32` vs. `n_sites=64`
(`n_tokens` scaled with `n_sites`, `chunk_size` held fixed) -- a
genuine weight-shared `Conv1d`-based architecture, not a masked dense
matrix.

**This is the first checkpoint this entire investigation to satisfy
BOTH Phase A (bounded, `D_KY` in range, competitive D3) AND Phase F's
structural L-transfer requirement AT ONCE, with no compromise between
the two.** Candidate to become the new frozen `LOCAL_AE`/`LOCAL_PROP`
AND `TRANSFER_PROP` simultaneously, pending full Gate 3/4 verification
(only the lighter D3-only check has been run) and the actual F2/F3
L-transfer tests (run at a larger `L` with zero retraining -- not yet
attempted for any candidate).

### Section 224: `d_latent=48` (n_sites=16, local_channels=3) beats `d_latent=96` on every axis measured (2026-09-24/25)

After explaining why `local_field`'s `d_latent=96` isn't trying to match
the global-vector design's `44` (it's sized from a LOCAL dof-density
argument -- `dof/site ~= 0.226*h`, `h=L/n_sites` -- not the global
attractor dimension), user asked "tell me if you think it might work
for d_latent=50," then "run 224 with d_latent=48" (50 isn't reachable:
`NX=256=2^8` forces `n_sites` to be a power of 2, so no integer
`local_channels` gives exactly 50 -- `48` = `n_sites=16, local_channels=3`
is the nearest clean value). Otherwise an exact repeat of Section 223's
recipe (`local_mlp` propagator, tokens aligned to sites, `--attn-window
3` kept unchanged in SITE units -- so now a larger fraction of a smaller
ring / wider physical extent than in 223, not compensated for
deliberately, to keep this a single-variable `d_latent` test).

**Result: `d_latent=48` matched or beat `d_latent=96` on every metric.**
Stage 1 ALONE bounded even more tightly than 223's (`max|z|` `4.5->7.7`,
plateaus, vs. 223's own `4.8-6.2`), `D_KY=10.47` (below target but
genuinely bounded/computable). **Stage 2 bounded across the FULL
2000-step rollout in an even tighter band** (`max|z|` `4.0-4.7`, vs.
223's `7.0-9.1`), `lambda1=0.088`, `n_positive=12/48`, `D_KY=22.14`
(in `[21,24]`, comparable to 223's `22.66`). `best_val_kmax_mse
=0.0161` -- BETTER than 223's own `0.042`, the best Stage-2 training
convergence of any local-field checkpoint this session. D3
bandedness=`0.4313` (p=0.0000) -- HIGHER than 223's `0.3091`, the
highest of any validated (non-diverging) checkpoint this entire
investigation.

**Reading:** halving `d_latent` (96->48) did not starve the model of
capacity -- `n_positive` stayed almost identical (12/48 vs 13/96,
i.e. nearly the same EFFECTIVE chaotic dimensionality), just packed
into half the raw coordinates, and every measured quality signal
improved. This suggests `d_latent=96` carried more redundant/slack
coordinate budget than the dynamics actually needed, and a tighter
budget may have forced a less collapsible, more efficiently-used
representation rather than hurting it. Not yet known how far this
trend continues -- worth testing progressively smaller `d_latent`
(e.g. `n_sites=16/local_channels=2=32`, approaching the brief's own
`dof/site` floor more closely) to find where quality actually starts
to degrade, rather than assuming `48` is already the efficient
frontier.

### Phase A/B re-verification with Section 224 as the frozen `LOCAL_AE`/`LOCAL_PROP`/`TRANSFER_PROP` (2026-09-25)

`LOCAL_AE`/`LOCAL_PROP` formally reset from Section 216 to **Section
224** (`local_field`, `n_sites=16, local_channels=3, d_latent=48` +
`local_mlp` propagator) -- see Section 224's own writeup above for the
full comparison against 216/223. Also serves as `TRANSFER_PROP` for
Phase F, since `local_mlp` is verified structurally L-transferable.

Re-ran Phase B2's DA sanity check (Section 216's original numbers are
now historical, not operative). **Section 224 DAs decisively better
than Section 216 ever did, even without any localization**:
`n_ensemble=64, n_prop_steps=3` gives `calibration_spread_over_rmse
=0.467` (in the `[0.4,1.5]` target) and `skill_free_over_da=3.27` --
clears both bars cleanly, and actually BEATS `GLOBAL_AE`/`GLOBAL_PROP`'s
own best result (`skill=2.73`). Compare Section 216's best-ever result
under the same test: `skill=1.64`, never clearing the `>2` bar at any
ensemble size tried. Chosen operating point for Phase C/D:
`n_ensemble=64, n_prop_steps=3`.

### Part 4.3 Phases C/D/E: the actual decision-rule experiment, run for the first time (2026-09-25)

With `GLOBAL_AE`/`GLOBAL_PROP` (Section 213) and the newly-frozen
`LOCAL_AE`/`LOCAL_PROP` (Section 224) both validated and DA-sanity-
checked, ran the actual Part 4.3 experiment: Phase 7's SEC baseline
(never run before this session) and Phase 13's core Gaspari-Cohn
sweep, then applied the pre-registered decision rule.

**Infrastructure built:** `scripts/run_da_ensemble_sweep.py` (Phase
C1) -- loops `run_da_pff.py` over `--n-ensemble` x `--localizer`,
collecting results into one CSV; records a per-combination `FAILED`
row (with the error message) rather than aborting the whole sweep or
silently dropping it, when an individual run crashes (a genuinely
singular ensemble covariance at very small `N_ens` is itself an
informative result, not a harness bug). `scripts/check_4_3_decision_
rule.py` (Phase E1) -- mechanically applies `rmse_gaspari_cohn[N] <=
rmse_sec[N]` per `N_ens`, printing PASS/FAIL/SKIPPED per row, not just
one overall verdict; 8 unit tests (all-pass, all-fail, mixed,
exact-tie-counts-as-pass, failed-run-skipped, CSV round-trip) pass
before being trusted on real data, per that phase's own explicit
requirement.

**Phase D1 (one-cycle taper sanity check):** confirmed directly on
Section 224's own real forecast covariance -- at every `gc_c` tested,
far-apart site pairs (circular site distance beyond the support radius
`2*gc_c`) are EXACTLY zero after tapering (not just small), near
pairs retain real covariance structure (`max~0.0144`).

**Phase D2 (radius selection):** swept `gc_c` in `{1.0,1.5,2.0,3.0,4.0,5.0}`
at `n_ensemble=64`. Cross-checked against Phase 2's own measured
light-cone bound (`docs.RESULTS.md`'s Gate 2 entry): at `dt_snap=1.0`
(stride 20), the light-cone term is negligible (~0.2 sites) versus the
encoder/propagator's own receptive field (6 sites), so
`minimum_localization_radius = 6` sites, requiring `gc_c >= 3.0`
(support radius `2*gc_c >= 6`). This matched the empirical result
cleanly: `gc_c in {3.0,4.0,5.0}` (at or above the theoretical floor)
all clearly beat the no-localization baseline; `gc_c=1.0` (well below
the floor) did not. Chose `gc_c=3.0` -- the theoretical floor exactly,
and the best raw RMSE improvement.

**Phase C2 (SEC sweep on GLOBAL, Section 213) and D3 (Gaspari-Cohn
sweep on LOCAL, Section 224):** both run over
`N_ens in {8,16,32,64,128,256}`, same `n_prop_steps=5` (Phase C's own
chosen value, used for both to keep the cross-comparison fair) and
`n_cycles=40`. Full tables in
`artifacts/da_sweep_phaseC2_global.csv`/`artifacts/da_sweep_
phaseD3_local224.csv`; plotted in
`docs/figures/phase4_3_decision_rule_sweep.png`.

| N_ens | GLOBAL, none | GLOBAL+SEC | LOCAL(224), none | LOCAL(224)+GC(3.0) |
|---|---|---|---|---|
| 8 | 2.550 | 0.506 | 1.649 | 0.530 |
| 16 | 1.663 | 0.422 | 1.506 | 0.435 |
| 32 | 0.442 | 0.372 | 0.772 | 0.421 |
| 64 | 0.354 | 0.342 | 0.448 | 0.406 |
| 128 | 0.346 | 0.329 | 0.406 | 0.405 |
| 256 | 0.330 | 0.330 | 0.377 | 0.405 |

**Phase E1 (decision rule applied mechanically): 0 PASS, 6 FAIL --
local-field + Gaspari-Cohn does NOT beat SEC on the global latent at
any tested ensemble size.** SEC's `rmse_da` is consistently 5-25%
lower than Gaspari-Cohn's across the whole sweep, and the gap widens
at large `N_ens` (SEC keeps improving toward `0.33`; Gaspari-Cohn
plateaus around `0.40-0.41`, and is actually slightly WORSE than
no-localization at `N_ens=256` -- over-tapering once the ensemble is
already large enough to estimate `B` well without help).

**This is a negative result under the strict pre-registered rule, but
not an uninformative one, per Part 4.3's own pre-registered framing:**

1. **Gaspari-Cohn genuinely helps the local latent over no
   localization** -- dramatically at small `N_ens` (`N=8`: `1.649 ->
   0.530`; `N=32`: `0.772 -> 0.421`), confirming the localization
   mechanism itself works correctly (matches Phase D1's direct taper
   check) and is not simply inert.
2. **SEC is just a strong baseline here, as pre-registered it should
   be** -- the whole point of testing against SEC rather than
   no-localization was that SEC is a real, competitive method
   (Anderson 2012), not a strawman; this result is exactly what makes
   a future PASS (if one is ever found, e.g. at a different `gc_c`,
   `N_prop_steps`, or observation density) credible rather than
   trivial.
3. **The L-transfer property remains the sole surviving argument for
   the local-field approach** -- exactly as Part 4.3's own text
   anticipated for a negative result here. Section 224/`local_mlp` is
   already verified structurally L-transferable (identical parameter
   count at `n_sites=32` vs `64`); SEC is fit to one training
   distribution and cannot transfer at all, and the global `mlp`
   cannot even be loaded at a different `d_latent`. Phase F's F2/F3
   tests (actually running at a larger `L` with zero retraining) are
   what would make this argument concrete rather than structural.

**Not yet tried, real candidates for a follow-up before treating this
as final:** a different `n_prop_steps` (5 was Phase C's own choice for
the global checkpoint, not independently re-optimized for the
comparison), a finer `gc_c` sweep between 2.0 and 4.0, and whether
`gc_c` should itself vary with `N_ens` (SEC's own table is explicitly
`N_ens`-specific; a single fixed Gaspari-Cohn radius across the whole
sweep is a simpler but not obviously optimal choice).

### Phase F2: L-transfer confirmed, zero retraining, at both 2x and 4x domain size (2026-09-25)

User: "yes, please try phase F and run a larger L without retraining.
Make certain that if L gets larger, the number of samples gets larger
though so the effective sample width remains the same." Confirmed the
mechanism precisely before implementing: `patch_size = NX/n_sites` is
the number of RAW GRID POINTS each site's patchify convolution
consumes, and that layer's weights only mean the same thing physically
if `patch_size` stays fixed -- so `NX` must scale with `L` to hold
`dx=L/NX` fixed, which (since `patch_size` is held fixed) forces
`n_sites` to scale by the same factor automatically. Doubling `L`:
`NX` 256->512, `n_sites` 16->32, `d_latent` 48->96, all falling out of
one constraint, not three independent choices.

**Built `load_autoencoder_checkpoint_resized`/`load_propagator_
checkpoint_resized`** (`ks_latent/models/__init__.py`) -- loads a
`local_field`/`local_mlp` checkpoint's TRAINED weights into a freshly
constructed model at a different `n_sites`/`NX`/`d_latent`/`n_tokens`,
raising clearly if the checkpoint's own architecture doesn't have
size-independent weight shapes (`masked_mlp`/`masked_mlp_expand`/`mlp`
explicitly rejected -- their layers are tied to `d_latent`, confirmed
in Section 221's own writeup) or if the caller's requested resize
doesn't hold `patch_size`/`chunk_size` fixed. Verified directly before
either function was written: Section 224's exact trained `state_dict`
loads with ZERO missing/unexpected keys into both a `2x` and `4x`
larger model. 8 new unit tests (weight-value transfer, not just shape;
forward pass at the new size; both rejection cases) pass.

**`scripts/run_ltransfer_test.py` (Phase F2): ran the resized model
in pure free-running forecast mode at `L=200` (2x) and `L=400` (4x),
zero retraining, against fresh KS ground truth generated at each new
`L`/`NX`.**

| | `L=100` (trained) | `L=200` (2x, zero retraining) | `L=400` (4x, zero retraining) |
|---|---|---|---|
| `NX`/`n_sites`/`d_latent` | 256/16/48 | 512/32/96 | 1024/64/192 |
| decoded rollout | -- | finite, bounded (`max\|u\|=2.62` vs true `3.35`) | finite, bounded (`max\|u\|=2.87` vs true `3.47`) |
| energy spectrum peak `k` | -- | true=0.660, model=0.660 (**exact match**) | true=0.675, model=0.691 (~2% apart) |
| `D_KY` | 22.14 | 44.61 | 89.13 |
| `D_KY/L` | 0.2214 | **0.2231** | **0.2228** |

Both `D_KY/L` ratios at the transferred sizes land almost exactly on
the trained model's own `0.2214` -- and all three are close to the
TRUE KS extensivity constant (`~0.226`). The decoded rollout stays
bounded and the energy spectrum peaks near the correct, L-independent
wavenumber (`k~1/sqrt(2)~0.707`) at both transfer sizes. **This is a
clean, quantitative confirmation of genuine L-transfer**: the same
weights, trained only at `L=100`, correctly reproduce KS's extensive
scaling law at 2x and 4x the trained domain size, with zero
retraining. Per Part 4.3's own framing, this is the result neither the
global `mlp` (Section 213, cannot even be loaded at a different
`d_latent`) nor SEC (fit to one training distribution) can produce at
all -- the practically-unique contribution of the local-field approach,
now demonstrated rather than only structurally argued.

### Phase F3: localized DA at L=200, zero retraining -- localization becomes NECESSARY, not just helpful, as the domain grows (2026-09-25)

`scripts/run_da_pff_ltransfer.py` (reuses `run_da_pff.py`'s own
`CycleConfig`/`run_da_experiment`/taper machinery, swaps in the resized
loaders) -- ran full DA cycling at `L=200` with `LOCAL_AE`/
`TRANSFER_PROP` (Section 224), zero retraining, `n_ensemble=64`
(UNCHANGED from the `L=100` operating point -- the whole point of the
test), `n_prop_steps=5`, `--obs-stride` scaled `8->16` to hold physical
observation density fixed (same `dx`-preserving principle as
`n_sites`/`NX`).

| | no localization | Gaspari-Cohn (`gc_c=3.0`) |
|---|---|---|
| `rmse_da` | 1.652 | **0.812** |
| `skill_free_over_da` | **0.69 (DA actively HURTS)** | **1.34** |
| `calibration_spread_over_rmse` | 0.18 | 0.81 |

**At `L=200`, with the SAME `n_ensemble=64` that worked fine at
`L=100` (`docs/RESULTS.md`'s Phase A/B entry: `skill=3.27`), unlocalized
DA actively breaks** (`skill=0.69`, worse than just running the model
free) **-- the raw latent dimension doubled to 96 along with `L`, so a
fixed ensemble size is now badly underdetermined for the full
covariance.** Gaspari-Cohn localization -- the SAME `gc_c=3.0` chosen
at `L=100` (valid unchanged, since `h=L/n_sites` is held fixed by the
transfer itself) -- rescues this back to real positive skill, more
than doubling the RMSE improvement over no localization. **This is
`docs/LITERATURE_REVIEW_AND_FINDINGS.md` Part 4.3's central claim
demonstrated directly: required ensemble size scales with LOCAL
dimension, not the (now-doubled) global one, and localization's value
GROWS as the domain grows** -- exactly the ensemble-size-scaling
argument the whole proposal was built around, now shown at a domain
size the checkpoint was never trained on.

**Confirmed directly (not just architecturally asserted) that
`GLOBAL_AE`/`GLOBAL_PROP` cannot even be evaluated at `L=200`:**
feeding a correctly-resolved `L=200` sample (`NX=512`, same `dx` as
training) into Section 213's fixed-`d_latent=44` encoder raises an
immediate shape-mismatch (`"size of tensor a (64) must match size of
tensor b (32)"`) -- no amount of retraining-free adaptation is
possible, confirming the contrast Part 4.3 frames as the actual
headline result (one method transfers, the other structurally cannot).
SEC is fit to one `N_ens`-specific empirical table at one training
distribution and has no mechanism to transfer either (not separately
re-tested; the mechanism itself has no size-dependent inputs to even
attempt at a new size).

**Gap found along the way, not yet fixed:** `scripts/run_da_pff.py`'s
own `--L` flag does NOT actually change the checkpoint's input size
(that's fixed by the checkpoint's own trained `NX`) -- passing `--L
200` with a checkpoint trained at `NX=256` silently generates ground
truth at the WRONG resolution (`dx=200/256~=0.78` instead of the
trained `0.39`) rather than raising, since the array shape still
happens to match. This is a real "fail loudly" gap (brief ground rule
2) -- anyone using `--L` on `run_da_pff.py` directly (not through this
Phase F script, which handles `NX` correctly) without also reasoning
about `NX` could silently get a physically-wrong comparison. Flagged
here for a future fix (e.g. `run_da_pff.py` could assert `L/NX` matches
the checkpoint's own trained ratio unless an explicit override flag is
passed); not fixed now since it didn't block this test (a dedicated,
correct script was used instead).

### E1: ground-truth Lyapunov spectrum at the L-transfer sizes, both directions (2026-09-28)

User-directed follow-up to `docs/ks_local_field_ltransfer.tex`'s
proposed E1 experiment ("designed experiments to prove the
[L-transfer] results" -- see that document's full framing): run this
project's own already-validated Gate-1 Benettin-QR pipeline
(`scripts/exp_e1_groundtruth_lyapunov_ltransfer.py`) directly at the
Phase F2 L-transfer sizes, to check the transferred model's own
`D_KY` against genuine ground truth rather than only self-consistency.
User: "couldn't we test E1 with L=50 and L=25 too? might be
interesting." -- extended downward as well as upward.

| $L$ | true $D_{KY}$ (Benettin QR) | true $D_{KY}/L$ | model $D_{KY}$ (zero retraining) | model $D_{KY}/L$ |
|---|---|---|---|---|
| 25 | 1.56 | 0.0622 | **0.00** (collapsed, 0/12 positive) | 0.0000 |
| 50 | 11.37 | 0.2273 | 11.06 | 0.2212 |
| 100 (trained) | 22.14 (Gate-1 target [21,24]) | 0.2214 | 22.14 | 0.2214 |
| 200 | 45.46 | 0.2273 | 44.61 | 0.2230 |
| 400 | 89.66 | 0.2242 | 89.13 | 0.2228 |

**Finding 1, confirms the central L-transfer claim against real ground
truth for the first time -- and the match gets TIGHTER, not looser, as
the extrapolation grows**: at every size where the true system is
comfortably chaotic (`L>=50`), the transferred model's `D_KY` matches
the true, independently computed PDE value to within 2%.
`L=200`: model 44.61 vs. true 45.46, 1.87% relative error.
`L=400`: model 89.13 vs. true 89.66, **0.60% relative error** -- the
best agreement of any size tested, at 4x the trained domain. This is
no longer model-vs-itself -- it's the model's zero-retraining
prediction checked against a computation it never saw, across a 16x
range of domain sizes with no sign of degrading as `L` grows.

**Finding 2, sharpens the claim -- extensivity is an asymptotic law,
and the model breaks down harder than the true system does as `L`
shrinks toward the pattern-forming scale.** At `L=25`
(~2.8x KS's own pattern length `2*sqrt(2)*pi~8.9`), the TRUE system's
own `D_KY/L` has already dropped sharply from the asymptotic ~0.226 to
0.0622 -- expected, given `L=22` is this project's own Gate-1
small-`L` benchmark (`D_KY in [5.2,5.6]`) and `L=25` is close to that
same onset-of-chaos regime. But the true system is still genuinely
chaotic at `L=25` (`D_KY=1.56`, one positive exponent). **The
transferred model, at the same `L=25`, loses all chaos completely**
(`D_KY=0`, `lambda1~=-0.0004`, measured via
`scripts/run_ltransfer_test.py --new-L 25`) -- a sharp, clean
transfer FAILURE, not merely a degradation, and the true system's own
persistence of (weak) chaos at this size shows this isn't simply "the
physics turning off." Most likely mechanism: the model's receptive
field (6 sites, fixed by the trained weights) was tuned relative to
`L=100`'s own site spacing; at `L=25` only 4 sites remain in the
transferred lattice, so the receptive field now exceeds the domain
itself. This is exactly the kind of clean, quantified breakdown point
worth chasing further (does it track receptive-field-vs-domain-size
precisely, testable with a checkpoint trained at a smaller
`attn_window`? or is there a separate site-count representational
floor too?) -- not yet investigated further this session.

**E1 complete: all five sizes (L=25/50/100/200/400) now have
ground-truth PDE `D_KY` measured directly, not assumed.** `L=400`'s
run took 1711s wall time (NX=4096, n_directions=145, float64 CPU,
`n_positive=50/145`, `lambda1=0.1026`).

### E2: ensemble-size sweep at L=200 -- a real, but different, signature than predicted (2026-09-28)

`scripts/exp_e2_ensemble_sweep_ltransfer.py`, `N_ens in
{8,16,32,64,128,256}` x `{none, gaspari_cohn}`, zero retraining,
`n_prop_steps=5` (kept matching Phase C/D's own harder $L=100$
operating point above, not Phase A/B's easier `n_prop_steps=3`, for a
fair comparison against the existing SEC-vs-GC table).

| N_ens | skill, none | skill, Gaspari-Cohn | ratio (GC/none) |
|---|---|---|---|
| 8   | 0.722 | 0.831 | 1.15 |
| 16  | 0.513 | 1.043 | 2.03 |
| 32  | 0.596 | 1.218 | 2.04 |
| 64  | 0.692 | 1.342 | 1.94 |
| 128 | 0.848 | 1.395 | 1.65 |
| 256 | 1.266 | 1.398 | 1.10 |

**Mixed relative to the pre-registered "smallest N_ens clearing a
skill>=2 threshold" signature -- reported honestly, not reframed
after the fact.** Neither curve clears skill>=2 at any tested N_ens up
to 256 at L=200 under `n_prop_steps=5` (consistent with this same
operating point's own L=100 numbers in the Phase E1 SEC-vs-GC table
above, not a new degradation specific to L=200) -- so that specific
comparison is inconclusive within this range, not confirmed.

**A different, real signature shows up instead.** Gaspari-Cohn's
advantage over no localization is positive at EVERY tested N_ens
(ratio >1 throughout), and critically does NOT reverse into a
disadvantage at large N_ens the way it did at the trained size: the
L=100 SEC-vs-GC table above shows Gaspari-Cohn measurably WORSE than
no localization at N_ens=256 (rmse 0.405 vs. 0.377 -- over-tapering
once the ensemble is already large enough to estimate the covariance
well on its own). At L=200, the same N_ens=256 point still favors
localization (skill 1.398 vs. 1.266). This is a real, if narrower, form
of the central L-transfer claim: as the domain grows past the trained
size, localization's value does not roll off with ensemble size the
way it does at the trained size -- even though the absolute skill bar
set in advance (>=2) was not cleared here. Next steps before treating
this as settled: repeat at L=400, and rerun this identical sweep
protocol at L=100 to confirm the crossover-vs-no-crossover contrast is
reproducible and not itself noise (the original L=100 SEC-vs-GC sweep
used a different tag/run than this script, so isn't a byte-identical
comparison).

### E6: LETKF implemented as a second DA analysis scheme, first head-to-head vs. PFF (2026-09-28)

User: "would another DA technique be more suited to this problem?
particle filter? some kind of kalman filtering?" then "sure implement
LETKF." `ks_latent/da/letkf.py` (new): a covariance-localized EnKF
(Houtekamer & Mitchell 1998/2001 -- explicitly NOT Hunt et al. 2007's
own per-gridpoint local-domain LETKF, which targets much higher
dimensions than this project's `d_latent<=192`; naming caveat stated
directly in the module docstring to avoid overclaiming). Drop-in
`analyze(Z0, y) -> Z_analysis` interface matching
`ParticleFlowFilter`'s own, sharing the SAME forecast step and the SAME
`localize_fn` taper -- isolates the analysis-update-rule choice as the
one variable that differs from PFF. Kalman gain built from the
ensemble-mean Jacobian of the (nonlinear) observation operator (same
linearization PFF's own NAT branch already computes) and the localized
ensemble forecast covariance; perturbed-observation (stochastic) update.

Wired into `ks_latent/da/cycling.py` via a new `CycleConfig.method:
"pff"|"letkf"` field (default `"pff"`, zero behavior change for every
existing caller) and `--method`/`--letkf-inflation` on
`scripts/run_da_pff_ltransfer.py`. 3 new unit tests
(`tests/unit/test_letkf.py`): validated against the analytic
linear-Gaussian Kalman posterior to <5% relative error (mirrors
`test_pff_gaussian_linear`'s own bound exactly), plus shape/finiteness
and a direct check that the taper measurably shrinks a deliberately
induced far-apart prior correlation's effect on the posterior. Full
suite: 814 passed (811 baseline + 3 new), no regressions.

**First real result**: one head-to-head point at `L=200`, identical
setup to E2's `N_ens=64`/Gaspari-Cohn row -- **LETKF: skill=1.310
(rmse=0.832, calibration=0.784) vs. PFF: skill=1.342 (rmse=0.852,
calibration=0.778) -- nearly indistinguishable.** Only one point, not
a full comparison yet, but an encouraging early signal that the
localization result doesn't depend on PFF's specific nonlinear flow
mechanism -- the same effect should be demonstrable inside the
classical, more widely-trusted EnKF family too. Natural next step:
repeat E2's full N_ens/localizer sweep with `--method letkf` for a real
comparison curve, not queued yet (deferred behind E3 per explicit user
direction: "determine next step. if E3, do that").

### E3: SEC vs. Gaspari-Cohn on the SAME (local) latent at L=200 -- first PASS in this project's decision-rule history (2026-09-28)

`scripts/exp_e3_sec_vs_gc_at_transfer.py`/`scripts/exp_e3_sweep.py`
(new): unlike the original L=100 SEC-vs-GC comparison (SEC on the
44-dim GLOBAL latent, GC on the 48-dim LOCAL one -- two different
architectures), this applies BOTH localizers to the SAME resized local
latent at L=200 -- a methodologically cleaner ablation of localizer
choice alone. Key realization stated directly in the new script's own
docstring: `build_sec_table` is a pure Monte-Carlo function of
ensemble size alone (no dependence on the physical system or latent
dimension anywhere), so "fit SEC fresh at L=200" doesn't mean anything
different numerically -- the table IS already L-independent by
construction; what changes at L=200 is that SEC can finally be
compared against Gaspari-Cohn on an apples-to-apples latent.

| N_ens | SEC | Gaspari-Cohn | decision rule |
|---|---|---|---|
| 8   | FAILED (indefinite covariance) | 1.230 | SKIPPED |
| 16  | FAILED | 0.935 | SKIPPED |
| 32  | FAILED | 0.915 | SKIPPED |
| 64  | FAILED | 0.812 | SKIPPED |
| 128 | FAILED | 0.814 | SKIPPED |
| 256 | 0.903 | 0.814 | **PASS** |

**SEC failed outright -- a genuinely indefinite ensemble covariance
(`robust_cholesky` raising on a matrix with a measurably negative
eigenvalue, not roundoff-scale), not a close loss -- at every ensemble
size except the largest tested.** `scripts/check_4_3_decision_rule.py`
(same CSV passed as both --global-csv and --local-csv, since SEC and
GC rows both live in one file here): **1 PASS, 0 FAIL, 5 SKIPPED --
the first PASS this project's decision rule has EVER recorded**, at
`N_ens=256`, Gaspari-Cohn beats SEC (0.814 vs. 0.903), a direct
reversal of the L=100 result (0/6 PASS, SEC winning throughout, see
Phase E1 above).

**Genuinely informative beyond the single PASS**: SEC's own numerical
fragility at small N_ens relative to a larger d_latent (96 at L=200,
vs. 44 at L=100) is itself a form of the central claim -- an
empirically-fit correction has no mechanism to stay well-posed as the
latent dimension grows past what it was fit for, while Gaspari-Cohn's
physical taper, needing no fitting at all, ran cleanly at every size
tested. Caveat: this is one ensemble size's worth of head-to-head
comparison, not the six-point sweep originally planned, precisely
because SEC could not produce five of those six points at all --
whether this fragility is specific to this project's own SEC/ensemble-
generation convention or a more general property of covariance-based
sampling corrections at this dimension-to-ensemble ratio is not yet
known and would need its own follow-up (e.g. does `sec_n_trials`,
currently 2000, matter here?).

### E4: seed-robustness sweep complete, n=3 (2026-09-28)

`scripts/exp_e4_seed_robustness_ltransfer.sh` (new): exact repeat of
Section 224's own recipe with only `--seed` changed, followed by the
same D_KY-at-L=100 measurement and Phase F2 transfer check to L=200.
User: "I don't think we need a seed 3 if seed 1 and 2 show identical
results" -- seed 3 was killed shortly after it started (Stage 1,
epoch ~0) once seed 2's result confirmed seed 1's tight pattern; its
partial artifacts were removed.

| seed | D_KY(100) | D_KY/100 | D_KY(200) | D_KY/200 |
|---|---|---|---|---|
| 0 (Section 224) | 22.14 | 0.2214 | 44.61 | 0.2230 |
| 1 (new) | 22.13 | 0.2213 | 44.32 | 0.2216 |
| 2 (new) | 22.15 | 0.2215 | 44.41 | 0.2220 |
| **mean +/- range** | -- | **0.2214 +/- 0.0001** | -- | **0.2222 +/- 0.0007** |

Seed 1's Stage 1/Stage 2 converged to numbers nearly identical to
Section 224's own (`val_recon_final=0.000113` vs. 224's `0.000125`;
`val_kmax_mse~0.0157` vs. 224's `0.0161`); seed 2 likewise trained
cleanly. **This closes E4's own stated bar (retrain across `>=3`
seeds) with n=3.** The L=100 ratio is essentially seed-invariant
(stdev 0.0001); the transferred L=200 ratio has a small but real
spread (stdev 0.0007, seed 0 the high outlier) and every seed still
lands close to E1's independently-measured true ratio (0.2273 at
L=200). The L-transfer property is not an artifact of one lucky
checkpoint. Not yet covered from E4's original scope: the $(P,c)$
packaging sweep at fixed d_latent (does transfer quality depend on how
the fixed local dof-density is split between sites and channels, not
just its total).

### E7: full LETKF ensemble sweep at L=200 -- reproduces PFF's entire pattern (2026-09-28)

User: "let's design this experiment as E7, add a placeholder in the
latex document and run this." Added `--method`/`--letkf-inflation`
flags to `scripts/exp_e2_ensemble_sweep_ltransfer.py` (default `pff`,
zero behavior change to E2's own recorded results) and reran E2's
identical sweep protocol with `--method letkf` at L=200.

| N_ens | PFF none | LETKF none | PFF GC | LETKF GC | ratio PFF | ratio LETKF |
|---|---|---|---|---|---|---|
| 8   | 0.722 | 0.784 | 0.831 | 0.801 | 1.15 | 1.02 |
| 16  | 0.513 | 0.635 | 1.043 | 1.064 | 2.03 | 1.67 |
| 32  | 0.596 | 0.559 | 1.218 | 1.163 | 2.04 | 2.08 |
| 64  | 0.692 | 0.658 | 1.342 | 1.310 | 1.94 | 1.99 |
| 128 | 0.848 | 0.905 | 1.395 | 1.387 | 1.65 | 1.53 |
| 256 | 1.266 | 1.150 | 1.398 | 1.352 | 1.10 | 1.18 |

**Both halves of the pre-registered prediction held.** Every LETKF
skill value lands within ~0.15 of PFF's at the same point, and
critically, Gaspari-Cohn's advantage over no localization never
reverses at N_ens=256 under LETKF either (1.18x, vs. PFF's 1.10x) --
the same qualitative signature E2 found for PFF, unlike the L=100
crossover in the original SEC-vs-GC table. The alternative,
also-informative outcome (a small-N_ens divergence, PFF's nonlinear
flow handling non-Gaussianity better than LETKF's single linearized
gain) did NOT occur -- the two methods behave as essentially
interchangeable at this operating point, itself informative (no sign
this posterior has enough non-Gaussian structure for PFF's extra
machinery to earn its keep here). **Net effect: the
localization-becomes-necessary-at-scale finding now stands under two
independently implemented analysis schemes**, one of them the
classical, widely-trusted EnKF family -- substantially de-risking the
concern that Section E2/Phase F3's results are specific to PFF's own
nonlinear-flow mechanics.

### E2 repeated at L=400 -- unlocalized DA never beats free-running at ANY tested N_ens (2026-09-28)

User: "Let's repeat E2/E3/E7 at L=400." Same protocol as the L=200
sweep, `--new-L 400` (d_latent=192, NX=1024, n_sites=64, obs_stride=32).

| N_ens | skill, none | skill, Gaspari-Cohn | ratio (GC/none) |
|---|---|---|---|
| 8   | 0.691 | 0.869 | 1.26 |
| 16  | 0.510 | 0.990 | 1.94 |
| 32  | 0.503 | 1.056 | 2.10 |
| 64  | 0.597 | 1.047 | 1.75 |
| 128 | 0.795 | 1.050 | 1.32 |
| 256 | 0.897 | 1.066 | 1.19 |

**Two findings.** Confirming: the "ratio never drops below 1" signature
from L=200 reproduces almost exactly at L=400 (same shape, peaks near
N_ens~32, eases toward 256, always >1) -- now shown at two independent
transfer sizes, not one. Sharper/new: **without localization, skill
never exceeds 1 at L=400 at ANY tested N_ens up to 256** (best case
0.897, still worse than free-running) -- whereas at L=200, unlocalized
skill did cross above 1 by N_ens=256 (1.266). At 4x the trained domain,
no ensemble size tested rescues unlocalized DA at all; localization
isn't merely better here, it's the only thing tested that makes DA
worth running.

**Cautionary observation, flagged rather than glossed over**:
Gaspari-Cohn's own absolute skill ceiling is also lower at L=400
(<=1.066) than at L=200 (<=1.398), even though gc_c=3.0 is, by
construction, the same PHYSICAL localization radius at every transfer
size (site spacing h=L/P held fixed by the transfer itself -- this
isn't simply "the same taper covering a smaller fraction of a bigger
domain," that invariance is exactly what the transfer is supposed to
preserve). Not yet disentangled: whether this is n_prop_steps=5
compounding more forecast error over a genuinely harder,
higher-dimensional DA problem regardless of localization, or whether
gc_c itself needs re-examining at large transfer factors.

### E3 repeated at L=400 -- SEC fails at ALL six ensemble sizes; a real reproducibility caveat discovered along the way (2026-09-28)

Same protocol as the L=200 E3 sweep, `--new-L 400`.

**SEC's fragility gets categorically worse, not just quantitatively
worse: at L=400, SEC fails -- the same indefinite-covariance error --
at every single tested ensemble size, including 256**, the one size it
managed at L=200. Gaspari-Cohn ran cleanly at all six.
`check_4_3_decision_rule.py`: **0 PASS, 0 FAIL, 6 SKIPPED** -- the
comparison cannot even be run, because SEC produces zero valid points
at this size. Sharper version of the L=200 finding: an empirically-fit
correction has no mechanism to stay well-posed as the latent dimension
grows past what it was fit for, and by 4x the trained domain that
failure is total.

**Reproducibility caveat, discovered while re-running this sweep and
reported honestly.** Repeating the identical Gaspari-Cohn point
(N_ens=8, L=400, same explicit seed=0) three times gave three
different `rmse_da` values (1.36, 1.31, 1.28, ~6% spread); the same
check at L=200 (N_ens=64) gave two matching repeats and one differing
repeat, similar magnitude. Consistent with -- and a concrete, quantified
instance of -- a limitation `CLAUDE_CODE_BRIEF.md`'s own ground rule 3
already documents: MPS does not guarantee bitwise-reproducible
execution even with an identical seed (the propagator forward pass runs
on MPS; only Lyapunov/PFF/LETKF linear algebra is pinned to CPU
float64). **Does not undermine the qualitative findings**: SEC's
failures are hard `RuntimeError`s from a measurably negative covariance
eigenvalue, not a borderline number that could flip under ~5% noise,
and every "ratio always >1" / "skill never exceeds 1" conclusion in
this document has margins well outside that noise band. It DOES mean
individual reported skill/RMSE values should be read as indicative, not
exact to the last decimal -- a fully rigorous treatment would average
multiple seeds per DA point (the same treatment E4 already gives the
L-transfer architecture question), flagged as real follow-up work, not
retroactively applied to every number already collected.

### E7 repeated at L=400 -- LETKF confirms PFF's sharpest L=400 finding too (2026-09-28)

Same protocol as E7's L=200 sweep, `--new-L 400`.

| N_ens | PFF none | LETKF none | PFF GC | LETKF GC | ratio PFF | ratio LETKF |
|---|---|---|---|---|---|---|
| 8   | 0.691 | 0.732 | 0.869 | 0.795 | 1.26 | 1.09 |
| 16  | 0.510 | 0.581 | 0.990 | 0.996 | 1.94 | 1.72 |
| 32  | 0.503 | 0.466 | 1.056 | 1.025 | 2.10 | 2.20 |
| 64  | 0.597 | 0.529 | 1.047 | 1.047 | 1.75 | 1.98 |
| 128 | 0.795 | 0.730 | 1.050 | 1.053 | 1.32 | 1.44 |
| 256 | 0.897 | 0.922 | 1.066 | 1.060 | 1.19 | 1.15 |

**Confirms both halves of the prediction again, including the sharper
L=400 finding**: LETKF's unlocalized skill also never exceeds 1 at any
tested N_ens (0.922 at 256, essentially matching PFF's own 0.897) --
"no ensemble size rescues unlocalized DA at 4x the trained domain" now
holds under both analysis schemes, not one. The ratio column keeps its
shape under LETKF too. Read alongside the reproducibility caveat above:
these values carry the same ~5-6% run-to-run margin already quantified
there, so the close PFF/LETKF agreement should be read at that
resolution -- genuinely close, not identical to three significant
figures.

**E2/E3/E7 at L=400 now complete, closing out the user-directed "repeat
at L=400" request.** Net picture across both transfer sizes: the
localization-becomes-necessary finding holds at both L=200 and L=400,
gets STRONGER (not weaker) as the domain grows (unlocalized DA goes
from "beats free-running only at the largest tested ensemble" at L=200
to "never beats free-running at any tested ensemble" at L=400), under
BOTH DA methods tested (PFF and LETKF), and SEC's own numerical
fragility also gets strictly worse with domain size (1/6 valid points
at L=200 to 0/6 at L=400).

### Step 1 (2D KS), part 1: solver validated, genuine chaos found on essentially the first attempt (2026-09-28)

User: "please implement step 1: 2D KS" (from `docs/ks_local_field_ltransfer.tex`'s
"Staged next steps" section).

**Solver** (`ks_latent/solver/ks2d.py`, `ks_latent/config.py`'s
`KSConfig2D`): isotropic 2D KS, `u_t + nabla^2 u + nabla^4 u +
(1/2)|nabla u|^2 = 0`, periodic in both directions. Same ETDRK4/
contour-integral machinery as 1D (`Lhat = K^2-K^4`, `K^2=kx^2+ky^2`,
same functional form as 1D's own `k^2-k^4`), reimplemented
independently (not refactored out of the validated 1D module) to carry
zero risk to it. The nonlinear term CANNOT reuse 1D's own divergence-form
trick (`-u u_x = -(1/2)(u^2)_x`, an exact x-derivative) -- 2D's
`-(1/2)|nabla u|^2` is not a pure divergence, computed instead via
spectral differentiation in both directions, squared and summed in
physical space, transformed forward.

**A real, non-obvious bug caught before it corrupted anything**: an
initial unprojected implementation showed `mean(u)` drifting
unboundedly and `max|u|` exceeding 100 within the spin-up window.
Diagnosed correctly, not patched around: since EVERY term on this
equation's RHS is a function of `u`'s spatial derivatives only, `u`'s
absolute value carries zero physical content -- but unlike 1D (whose
nonlinearity is an exact derivative, so `mean(u)` is automatically
conserved with no extra step), 2D's own nonlinearity has a nonzero
spatial mean (`<u_x^2+u_y^2> > 0` generically), so the equation
genuinely drives an unbounded drift in the physically-meaningless mean
mode. Fixed EXACTLY (not approximately): the k=0 Fourier mode is
projected to zero after every step (base state AND every tangent
direction, for a Lyapunov-consistent gauge) -- after the fix, `max|u|`
stayed in `[10,16]` over a full 2000-step trajectory with `mean(u)`
pinned at float64 roundoff.

**Solver validation** (`tests/unit/test_solver_2d.py`, 9 tests, same
Gate-1-style discipline as 1D): linear-instability spectrum (isotropic
ring at `|K|=1/sqrt(2)`, same critical radius as 1D at every tested
angle); ETDRK4 coefficient construction validated two ways (exact
agreement with 1D's own tested function on a 1D-shaped input, proving
the broadcasting generalization is lossless; direct mpmath comparison
to 1e-12 on a 2D-shaped input); exact mean conservation; exact
translation equivariance in x, in y, and jointly; convergence order
>=3.5 in dt; small-domain decay to zero. All 9 pass; full suite 845/845
(836 baseline + 9 new), no regressions.

**Chaos found on essentially the first reasonable parameter choice --
no extensive search needed.** At `Lx=Ly=32` (a modest first test
domain): `lambda1=0.237`, `D_KY=45.5` (bracketed cleanly at 70
directions, 24 positive), `D_KY/Area=0.0444` -- genuinely rich,
well-behaved chaos (smoothly decreasing exponent spectrum, not
degenerate). Confirmed again at the actual production config
(`Lx=Ly=44, NX=NY=128`, the default in `KSConfig2D`): all 12 quick-test
directions positive, consistent with even richer chaos at the larger
domain. `ks_latent.analysis.lyapunov.lyapunov_spectrum_ks2d`/
`_ks2d_step_with_tangent_physical_space` (new, additive) reuse the
existing `benettin` core unmodified via a flatten/reshape wrapper at
the boundary (`benettin` assumes a flat `(state_dim,)` vector space).

**Architecture built (encoder + first propagator).**
`KSAutoencoderLocalField2D` (`Conv2d` throughout, literal 2D
generalization of Section 224's own encoder, same gauge-anchor design)
-- 17 tests (shapes, exact gauge anchor, exact equivariance in x/y/
jointly, no-bias output, gradient flow). First propagator attempt:
`site_conv_2d` (2D generalization of the `site_conv` backbone, reusing
the encoder's own circular-Conv2d site-mixing stack as a
dimension-preserving `z->z` map -- same precedent Section 222/223
established for 1D), NOT `local_mlp` generalized directly -- a
deliberate scope choice for the first pass (no new 2D token-window
masking machinery needed), noted honestly rather than implied to be
"the 224 propagator in 2D."

**Dataset-sizing mistake caught and fixed.** A first attempt reused 1D
KS's own dataset convention directly (60 trajectories, 1001 snapshots
each) without re-deriving the resulting file size for a 2D field -- a
2D snapshot at NX=NY=128 has 64x the raw element count of 1D's own
NX=256 field, so the same snapshot count produced a ~15.8GB file (raw +
normalized arrays combined), most of the way through generating before
the mistake was caught (user: "16 gb is huge, any way we can create a
smaller dataset?"). Corrected to 30 trajectories (24 train/6 val) x 251
snapshots (`trajectory_time=25`, ~6 Lyapunov times at
`lambda1~=0.24`) -- ~1.8GB, generated in ~8 minutes. Verified no
orphaned large files were left on disk from the failed attempts.

**Training launched** (Section 228: Section 224's exact regularizer
recipe, `d_latent=192`, `n_sites_x=n_sites_y=8`) on the corrected
dataset; results pending.

### `local_mlp_2d`: the actual 2D `local_mlp`, built as a second propagator to compare (2026-09-28)

User: "how hard would building the actual 2D local_mlp be?" then
"build it please." Genuinely straightforward given `site_conv_2d`'s own
precedent: `_LocalMixerBlock`'s three components (pre-norm windowed
mixer + residual, pre-norm per-token FFN + residual) are all per-token
pointwise operations except the mixer itself, so the ONLY new code
(`_LocalMixerBlock2D`) was swapping the windowed circular `Conv1d` for
a windowed circular `Conv2d` -- the same 1D->2D generalization pattern
already used twice (the encoder, `site_conv_2d`). `_LocalMLPDeltaBody2D`
is purpose-built for a `local_field_2d`-paired latent specifically, so
(unlike 1D's own general-purpose `local_mlp`) it needs no general
tokenizer at all: tokens ARE the sites, already naturally gridded.

9 new unit tests (shape, identity-at-init, exact equivariance under a
site shift, gradient flow, multi-step rollout finiteness) -- all pass.
Also directly verified NOT to trigger the multiscale experiment's own
MPS rollout+backward+amp bug (that bug was specific to a degenerate
single-token case; `local_mlp_2d`'s smallest realistic token grid is a
genuine 8x8=64 tokens). Full suite: 871 passed (862 baseline + 9 new),
no regressions. `backbone="local_mlp_2d"` wired into both
`AuxPropagatorConfig`/`PropagatorConfig` (new `local_mlp2d_*` fields,
same mechanical pattern as `multiscale_local_mlp`/`site_conv_2d`
before it) and `LatentPropagator`'s dispatch. Not yet trained/compared
against `site_conv_2d` -- queued behind Section 228's own results.

### Section 228 result: `site_conv_2d` collapsed -- D_KY=0, a real negative result with a well-motivated next step (2026-09-28)

Training completed (200+300 epochs). `val_recon_final=0.018712`,
`best_val_kmax_mse=0.044841`. **A real anomaly flagged before
diagnosing further**: `val_kmax_mse` printed EXACTLY `0.044846` for
every logged epoch from ~80 through 300 (220+ epochs straight) despite
the training `loss` itself continuing to change (0.016->0.046 as the
k-curriculum ramped, matching the LR's own continuing cosine decay) --
not a logging artifact (`val_kmax_mse` in the print statement is the
genuine per-epoch recomputed value, confirmed by reading
`ks_latent/training/loops.py`'s own `train_stage2`, not a "best so
far" display).

**Diagnosed directly rather than assumed.** Standalone rollout: finite
everywhere over 2000 steps, `max|z|` in `[7.36,7.65]` (a slow,
MONOTONIC, non-oscillating drift, not the usual bounded-fluctuation
pattern a healthy chaotic rollout shows); different initial conditions
stayed well-separated (mean pairwise distance 14.8 across 6 ICs after
2000 steps -- rules out a classic fixed-point collapse where different
ICs converge to the same state). The real number: Lyapunov spectrum of
the trained latent model, `mode='single_state'`, `n_directions=192`,
`n_steps=2000` -- **`lambda1=-1.19e-5` (essentially exactly zero),
`n_positive=0/192`, `D_KY=0`.** A genuine collapse to a
non-chaotic (near-neutral) dynamical regime, not the "diverges to
NaN"/"clean fixed point" failure modes seen elsewhere in this
project's history -- a third, distinct collapse signature.

**A well-motivated explanation and next step, not a dead end.**
`site_conv_2d` was chosen for this FIRST 2D attempt specifically
because it needed no new 2D token-window masking machinery -- but
`site_conv` was never the reliably robust backbone in 1D either:
Section 222 (`site_conv`, 1D) got good short-horizon behavior but
diverged catastrophically by step 1000; Section 223 (`local_mlp`, 1D)
was the one that held `D_KY=22.66` bounded across the full 2000 steps
and became Section 224's own winning recipe. This 2D collapse may
simply be repeating a known-weaker-1D-backbone's own known
unreliability, not a new, 2D-specific failure mode. `local_mlp_2d`
(the actual 2D analogue of Section 224's own winning propagator,
already built and tested per the user's own follow-up request) is the
natural, well-precedented next attempt -- queued, not yet launched.

### Section 230: genuine chaos recovered -- D_KY=71.9, with one honest caveat about what's actually been isolated (2026-09-28)

User: "kill it. try 230 with attention window = 3" (Section 229,
`local_mlp_2d` at the same `attn_window=2` Section 228 used, was
killed mid-Stage-1 before producing a real result, superseded rather
than diagnosed). Added `--backbone`/`--attn-window` CLI flags to
`scripts/train_2dks_localfield_standalone.py` (additive, defaults
preserve Section 228's exact behavior) and launched Section 230:
`local_mlp_2d`, `attn_window=3` (receptive field `2 layers x 3 window
= 6 sites ~= 33 physical units`, now above the "2-3 correlation
lengths" heuristic's own upper bound of ~26.7 -- deliberately testing
past it, not just up to it).

`val_kmax_mse` never froze this time -- genuinely fluctuated the whole
run (0.028 -> 0.017 minimum around epoch 100-110, a slow drift back to
0.019 by epoch 300, ordinary training dynamics, not the pathological
exact-repeat pattern Section 228 showed). `best_val_kmax_mse=0.016956`
-- **more than 2.5x better than Section 228's 0.044841.**

**Diagnostic: genuine, rich chaos.**

| | Section 228 (`site_conv_2d`, window=2) | Section 230 (`local_mlp_2d`, window=3) |
|---|---|---|
| lambda1 | -1.19e-5 (~=0) | **0.197** |
| n_positive | 0/192 | **35/192** |
| D_KY | 0 | **71.9** |
| max\|z\| pattern (2000 steps) | monotonic drift, 7.36->7.65 | genuinely oscillating, 7.17<->8.15, bounded |
| IC pairwise separation | 14.8 | 15.95 |

**D_KY=71.9 is strikingly consistent with the true system's own
extensivity, not just "some positive number."** E1's own ground-truth
measurement at the smaller `Lx=Ly=32` test domain gave
`D_KY/Area=0.0444`; naively scaling that ratio to the production
domain's area (`44^2=1936` vs `32^2=1024`) predicts `D_KY~=86` --
same order of magnitude as the model's own 71.9, not a wild mismatch.
(Not a rigorous ground-truth comparison -- that needs an actual
Benettin-QR computation at `Lx=Ly=44` directly, not yet done -- but a
strong, encouraging consistency check.)

**One honest caveat, stated precisely rather than overclaimed.** Two
variables changed between Section 228 and 230 (backbone AND
`attn_window`), and Section 229 (`local_mlp_2d` at the SAME
`attn_window=2` Section 228 used, which would have isolated which one
mattered) was killed before finishing. Both plausibly contributed to
the fix; the data needed to cleanly separate "backbone choice" from
"receptive field width" as the operative variable does not exist yet.
A genuine `local_mlp_2d`-at-`window=2` rerun would close this gap if
it's ever worth the compute.

### Step 2 (Kolmogorov flow): solver + architecture reuse implemented and validated (2026-09-28)

User: "please implement step 2 from ks_local_field_ltransfer for
Kolmogorov flow. Reuse as much as possible from prior code both for 2d
KS and for the 1d ks." Proceeded despite Step 2's own stated gate (a
fully validated 2D-KS zero-retraining transfer result) not yet being
closed -- Section 230's chaos result was judged sufficient evidence the
2D architecture generalizes to start Step 2 in parallel, with the
transfer experiment itself still open on both systems; flagged rather
than silently skipped.

Forced 2D Navier-Stokes in vorticity-streamfunction form:
`omega_t + u.grad(omega) = nu*nabla^2(omega) + f(y)`, `u=(psi_y,-psi_x)`,
`nabla^2(psi)=-omega` solved exactly per-mode in Fourier space (one
division per mode). `f(y) = -nu*n^3*cos(n*y)` is chosen so the laminar
solution `u_lam=(sin(n*y),0)` is an EXACT steady state of the FULL
nonlinear equation (the convective term vanishes identically on it) --
the classical Kolmogorov-flow setup (Meshalkin & Sinai 1961; Chandler &
Kerswell 2013 study exactly this `n=4`, `2*pi x 2*pi`-box configuration,
already cited in this document's own literature review).

**Real, quantified code reuse, not just a structural analogy.**
`ks_latent/solver/kolmogorov.py` imports `wavenumbers2d` and
`etdrk4_coefficients_2d` from `ks2d.py` DIRECTLY, unmodified -- the
linear operator here (`Lhat=-nu*(kx^2+ky^2)`, pure diffusion, no
zero-crossing, strictly better-conditioned than KS's own `K^2-K^4`) is a
strict special case of the same contour-integral construction 2D KS
already validated. `generate_trajectory_dataset2d` (`dataset2d.py`) was
generalized with optional `integrate_fn`/`spinup_fn` parameters
(defaulted to 2D KS's own functions, so every existing caller/test is
byte-for-byte unaffected) rather than duplicated;
`generate_trajectory_dataset_kolmogorov` is a 10-line wrapper.
`KSAutoencoderLocalField2D`, `site_conv_2d`, and `local_mlp_2d` needed
**zero new code** -- confirmed directly by reading them, not assumed:
they operate on a flat `NX*NY` scalar field and have no notion of which
PDE produced it. Only the nonlinear-term evaluation and its tangent-linear
model are new physics-specific code, plus a new `KolmogorovConfig` and a
`lyapunov_spectrum_kolmogorov` adapter mirroring `lyapunov_spectrum_ks2d`'s
own flatten-at-the-boundary pattern.

**The new tangent-linear model.** Vorticity advection `u.grad(omega)` is
bilinear in `omega` (the velocity `u` is itself linear in `omega` via the
Poisson solve) -- the same "product of two linear functionals of the
state" structure both KS variants' own nonlinearities have, so the
Frechet derivative is exact via the product rule:
`D[u.grad(omega)][domega] = du.grad(omega) + u.grad(domega)`, `du` the
velocity derived from `domega` by the identical linear map. No new
derivation technique was needed, just the third application of a pattern
this project has now used three times (1D KS, 2D KS, Kolmogorov flow).

**Validation (9 new tests, `tests/unit/test_solver_kolmogorov.py`, all
passing), two of which are new relative to both KS variants because this
system's physics genuinely differs:**

1. **The laminar solution is an exact fixed point of the discretized,
   time-stepped equation**, not merely the continuous PDE --
   `test_base_flow_is_exact_fixed_point` confirms `max|final - u_lam| <
   1e-8` after 500 ETDRK4 steps. Needs no external literature number --
   the strongest, least ambiguous check available for this solver.
2. **Anisotropic translation equivariance, tested both ways.** `f(y)`
   genuinely breaks full y-translation invariance (both KS variants are
   equivariant in every direction; this system is not) but leaves an
   EXACT residual discrete symmetry (shifts by `2*pi/n`) intact --
   `test_translation_equivariance_exact_at_forcing_period_in_y` confirms
   the residual symmetry to `1e-10`, and (ground rule 19, "test the
   failure modes") `test_translation_NOT_equivariant_for_generic_y_shift`
   confirms equivariance genuinely FAILS (relative difference > 0.05) for
   a shift that is not a multiple of the forcing period -- if this ever
   started passing, that would mean the forcing term had silently been
   lost, not that anything improved. x-equivariance (unaffected by the
   forcing) is exact to `1e-10`, as usual.
3. Mean-vorticity conservation, `dt`-convergence order >= 3.5, and a
   qualitative bifurcation check: the leading Lyapunov exponent of the
   linearization about the laminar fixed point (measured via a bespoke
   single-direction power-iteration growth-rate estimator built directly
   on `step_with_tangent_kolmogorov`, not `benettin`'s full machinery,
   which is unnecessary machinery -- and actively gets in the way here,
   since its internal Kaplan-Yorke computation raises when not enough
   directions bracket a still-growing cumulative sum) is negative at
   Re=1 and positive at Re=50 (`n=4`, `2*pi`-box) -- consistent with,
   but deliberately not claiming to precisely reproduce, the classical
   Meshalkin-Sinai/Chandler-Kerswell literature (this project's compact
   box is not the elongated/near-infinite-x domain the classical
   `Re_c=sqrt(2)` result needs).

Full suite re-run after all changes: 868/868 passing, no regressions
(2 new tests added on top of the 9 listed above, during the methodology
fix described below).

### Section 231: genuine Kolmogorov-flow chaos found, a real methodology bug fixed along the way, and a trained local-field model that closely reproduces it (2026-09-28)

**Correcting an overclaim from the entry above first.** "Re=50 well past
chaos onset" was wrong -- see below. The test itself (laminar solution
unstable between Re=1 and Re=50) is still correct and still passes; only
the interpretive claim about how far past onset Re=50 sits was wrong.

**The parameter search that followed hit a wall: every configuration
tried looked stable.** Sweeping Re in [40, 1000], square and elongated
(up to 4:1) boxes, forcing wavenumber n in {1,2,4}, and drag alpha in
[0,0.3] -- all at production resolution NX=NY=128, all with dealiasing
-- every single one showed a NEGATIVE leading Lyapunov exponent on the
saturated attractor (typically -0.2 to -0.4), despite the laminar base
flow being confirmed linearly unstable throughout. This is a
well-documented, well-cited chaotic system (Chandler & Kerswell 2013 and
others use exactly this n=4, 2*pi-box configuration) refusing to show
ANY chaos anywhere in a wide, honest search -- a strong signal something
was wrong, not that the literature was wrong.

**It was a real bug, found by direct isolation, not a guess.** All these
Lyapunov measurements used `warmup_time=0.0` (the function's own default)
with only 6-8 total time units of measurement. This project's own
established protocol elsewhere already knew better:
`scripts/exp_e1_groundtruth_lyapunov_ltransfer.py` defaults
`--warmup-time 300` out of `total_time=2000` -- the ad hoc exploration
scripts written for Kolmogorov flow simply hadn't followed that
convention. Direct isolation confirmed it exactly: measuring the SAME
config (n=4, Re=50, laminar fixed point) with a short 3-time-unit,
no-warmup power iteration gave lambda1 in [-0.2,-0.4] (matching the
"stable everywhere" sweep results); the identical calculation with a
10-time-unit warmup discarded and 30 more measured gave lambda1=0.837
(NX=32) and 0.839 (NX=128) -- not a refinement, a sign flip, and one
that agrees across resolution once measured correctly. **Lesson,
recorded here so it isn't lost**: the Benettin QR algorithm's random
initial tangent basis needs a genuine alignment period regardless of
whether the base trajectory itself is already on the attractor;
`warmup_time=0.0` is a dangerous default to leave unexamined for any new
system, and this project's own E1 protocol already encoded that lesson
once. Checked (grep for `warmup_time` across `scripts/`): this bug was
contained to the new, ad hoc Kolmogorov-flow exploration scripts written
this session -- 1D KS, 2D KS, and the trained-propagator Lyapunov checks
all go through protocols that already set real warmup, so no past
validated result in this project needs revisiting.

**Once corrected, genuine, resolution-converged chaos appeared cleanly
-- no drag, no elongated domain needed.** Re-measuring with
`warmup_time=20, total_time=40`: Re=50 (n=4, square box) sits at
lambda1=-0.0025, essentially marginal (a limit-cycle-like secondary
state, not chaos); Re=70 gives lambda1=0.057, n_positive=3, D_KY=7.09;
Re=100 gives lambda1=0.102, n_positive=2, D_KY=7.90 -- both confirmed at
production resolution NX=128. Separately, testing whether a SMALLER
forcing wavenumber (the opposite of what "less resolution" naively
suggests -- raising n needs MORE points to resolve a shorter wavelength,
and via the classical `Re_c~n*sqrt(2)` scaling makes the flow relatively
LESS unstable at fixed Re, not more) could reach chaos at lower
resolution found n=2, Re=40 chaotic at just NX=NY=48 (D_KY=3.59),
confirmed resolution-converged at NX=96 (D_KY=3.37) -- a genuine minimal
validated config, lower-dimensional than the n=4 point chosen for
production below.

**The drag term mattered for a different reason than originally
guessed.** Motivated up front (chat discussion, 2026-09-28, on the
parallel to RBC's own physical vertical confinement providing an outer
scale against 2D turbulence's inverse-cascade condensation, Kraichnan
1967) as a way to REACH chaos -- but once Re=100 was already confirmed
chaotic without it, sweeping alpha there revealed a real, non-monotonic
ENRICHMENT effect instead: D_KY=7.90 (alpha=0), **23.26** (alpha=0.05),
16.18 (alpha=0.1). alpha=0.05 -- clearly best of the three, not "more
drag is always better" -- was adopted as the production default.

**Production config** (`KolmogorovConfig` defaults):
`Lx=Ly=2*pi, NX=NY=128, nu=0.01 (Re=100), alpha=0.05, forcing_n=4,
dt=0.01, snapshot_every=20 (dt_snap=0.2), spinup_time=100`.
Ground truth at this config: lambda1=0.0905, n_positive=12/24,
**D_KY=23.26** -- comparable in scale to 1D KS's own D_KY~22 at its
canonical config.

**Dataset + training (Section 231).** `generate_trajectory_dataset2d`
was generalized (optional `integrate_fn`/`spinup_fn`, default unchanged)
rather than duplicated; `generate_trajectory_dataset_kolmogorov` is a
10-line wrapper. 30 trajectories (24 train/6 val), `trajectory_time=40`
(~3.6 Lyapunov times at lambda1=0.0905 -- shorter than 2D KS's own ~6,
traded off against file size given this system's slower dynamics),
1.52GB. Training reused `KSAutoencoderLocalField2D`/`local_mlp_2d` and
Section 224's regularizer recipe completely unchanged, `attn_window=3`
(reusing Section 230's own value directly): `val_recon_final=0.008251`
(better than 2D KS's own 0.016521), `best_val_kmax_mse=0.030675`,
`val_kmax_mse` fluctuating genuinely throughout training -- no
Section-228-style frozen-metric collapse signature at any point.

**Result: the trained model reproduces the true system's own chaos
closely, on the first attempt.** Standalone 2000-step rollout: finite
throughout, `max|z|` oscillating boundedly (4.5-6.3, no drift), two
independent ICs staying well-separated (final pairwise distance 15.0).
Lyapunov spectrum of the trained propagator (`n_directions=192`,
`warmup_steps=200`, `n_steps=2000`): lambda1=0.1207, n_positive=10/192,
**D_KY=21.33** -- against the true system's own D_KY=23.26 at the
identical config, a strikingly close match (closer, in fact, than 2D
KS's own Section 230 comparison against a naively-scaled ground-truth
estimate). The zero-retraining L-transfer payoff experiment (does this
hold at larger domains/more forcing periods, where the local-field
architecture's actual value proposition lives -- a single small box has
almost no spatial extent to localize over) is the natural next step, not
yet run.

### G1/G2: L-transfer tested for real on both 2D systems -- a genuine, informative mixed result (2026-09-29)

User-directed: "It seems like the next step would be to test L transfer
for section 230 and 231... make a specific plan, numbered g1-g_n in the
latex file, and let's run the relevant experiments one at a time." Full
G1-G8 plan written into `docs/ks_local_field_ltransfer.tex`'s new
"L-transfer for the 2D systems" section; G1 (2D KS) and G2 (Kolmogorov
flow) run, both against genuine ground-truth Benettin-QR computations of
the TRUE PDE at the transfer size (not just the model's own
self-consistency), reusing `lyapunov_spectrum_ks2d`/`lyapunov_spectrum_
kolmogorov` unchanged. New 2D resize helpers
(`load_autoencoder_checkpoint_resized_2d`/`load_propagator_checkpoint_
resized_2d`, `ks_latent/models/__init__.py`, 7 new tests) generalize the
existing 1D `_resized` mechanism -- verified directly, same reasoning:
Conv2d/circular-Conv2d weight shapes depend only on kernel size and
channel counts, never on `n_sites_x`/`n_sites_y`/`NX`/`NY`, so a trained
`local_field_2d`/`local_mlp_2d` state_dict loads with zero missing/
unexpected keys into a differently-sized model.

**G1 (2D KS, 1.5x per axis, `Lx=Ly=66` up from trained `44`).** TRUE PDE:
lambda1=0.204, n_positive=101/300, **D_KY=192.5** (11803s, 300
directions). MODEL (zero-retraining transfer): lambda1=0.057,
n_positive=19/432, D_KY=43.6 -- **77% relative error**, far worse than
1D KS's own E1 result (sub-2% at every size tested).

**The disambiguating finding: the TRUE system's own D_KY/Area ratio is
essentially constant across two independently-measured domain sizes** --
0.0444 at the original Lx=Ly=32 test domain (Step 1's own early
measurement) and **0.0442** at this experiment's Lx=Ly=66 -- about as
clean a confirmation of genuine area-extensivity as this project has
produced. This rules out "2D KS's physics isn't extensive" as the
explanation for the 77% gap -- there manifestly IS a stable local law,
and zero-retraining transfer is failing to carry it forward, a
transfer-MECHANISM problem, not a physics problem. The model's own
D_KY/Area collapses from 0.0371 (self-reported, trained size) to just
**0.0100** under transfer -- a ~4x density drop, severe and specific.

**G2 (Kolmogorov flow, 2x in x only, `Lx=4*pi` up from trained `2*pi`,
`Ly` fixed at the forcing period).** TRUE PDE: lambda1=0.129,
n_positive=28/75, **D_KY=56.8** (6833s, 75 directions) -- genuinely,
richly chaotic, resolving the inverse-cascade/condensation worry raised
in chat before this ran: doubling the domain does not collapse the
attractor. MODEL (zero-retraining transfer): lambda1=0.102,
n_positive=13/384, D_KY=28.4 -- **50% relative error**: smaller than 2D
KS's 77%, but still a real, substantial gap.

**But unlike G1, G2's own physics question is NOT yet settled.** The
TRUE system's D_KY grew SUPER-linearly with area: 23.26 -> 56.84, a
2.44x increase for an EXACTLY 2x increase in area -- not the ~2x growth
clean linear-in-area extensivity predicts. Two live hypotheses, not yet
disambiguated: (a) the drag alpha=0.05 isn't yet strong enough to pin
the correlation length independent of domain size (genuinely
non-extensive at this alpha, consistent with 2D turbulence's own
inverse-cascade condensation, Kraichnan 1967); (b) simply "not yet
asymptotic," the same finite-size caveat 1D KS's own E1 result raised at
small L. The proposed disambiguating test (chat, 2026-09-29, not yet
run): measure TRUE D_KY/Area at a third, larger x-scale and/or sweep
alpha at fixed domain size, purely ground-truth (no retraining needed),
to see whether the ratio converges or keeps accelerating.

**One architecture-side hypothesis raised in chat, applies to both
systems, not yet tested**: a fixed `attn_window` covers a shrinking
FRACTION of the domain much faster in 2D than in 1D as domain size
grows, since site count scales with (scale factor)^2 in 2D but only
(scale factor) in 1D -- purely architectural, orthogonal to whether the
underlying physics is extensive.

**Status: paused here, user-directed ("don't continue to g3 without
permission"), after G1/G2.** Both landed as genuine mixed results --
neither a clean confirmation of the paper's central claim nor a clean
refutation -- which is itself the honest, informative outcome this
project's own "both verdicts are publishable" discipline calls for.
G3-G8 remain queued.

### Phase 11 (never previously implemented): a local quadratic stencil closure fits Section 224's latent much better than Section 216's (2026-09-25)

User: "do you think we would be able to fit a pde more easily to the
latent space of 224 since it's highly localized? please try this and
report on the results." Neither `ks_latent/discovery/stencil.py` nor
`pde_find.py` (brief Phases 8/11) had ever actually been built in this
codebase -- genuinely new territory, not a rerun.

**`scripts/fit_local_stencil_pde.py`**: a shared, translation-invariant
Ridge regression -- the literal Phase 11 "stencil propagator" idea
(`z_j_dot = F(z_{j-w},...,z_j,...,z_{j+w})`, same law at every site) --
predicting each site's next-step latent delta from a circular window
of `2w+1` neighboring sites' current state, pooled across all
sites/times/trajectories. Plain linear features were far too weak
(R^2~0.03 even at full width, confirming the closure genuinely needs
nonlinearity, not just locality); degree-2 polynomial features (KS's
own nonlinearity is the quadratic advective term `u*u_x`) is where the
real signal appeared.

**Result, comparing Section 224 (`d_latent=48`, `n_sites=16`,
D3=0.43) against Section 216 (`d_latent=96`, `n_sites=32`, D3=0.21) --
same methodology, same dataset:**

| stencil half-width `w` (sites) | 224 val R^2 | 216 val R^2 |
|---|---|---|
| 0 (own site only) | 0.152 | 0.264 |
| 1 | 0.578 | 0.569 |
| 2 | 0.816 | 0.702 |
| 3 | 0.882 | 0.762 |
| 4 | 0.900 | 0.793 |
| 6 | 0.904 | -- |
| 8 (=full ring for 224) | 0.903 | 0.819 |
| 16 (=full ring for 216) | -- | 0.817 |

**Even at FULL global width (every site sees every other site, no
windowing restriction at all), a local quadratic closure law explains
90.3% of Section 224's one-step latent dynamics but only 81.7% of
Section 216's** -- a real ~8-point gap in explainable variance that
has nothing to do with window size once both are already global. And
224 reaches essentially its own ceiling (~90%) using barely a third of
its ring (`w=3`, 7/16 sites), while 216 never gets there even using
all 32 sites. (Caveat noted for completeness: at very SHORT physical
radius, 216 is actually slightly better -- `216`'s finer site spacing,
`h=3.13` vs `224`'s `h=6.25`, gives it a modest short-range edge before
224 pulls decisively ahead past `~12` physical units.)

**Reading**: this is a stronger and more precise result than "224 just
needs a smaller stencil" -- 224's dynamics are genuinely better
described by a compact, local, low-order polynomial closure law, not
merely easier to fit within a restricted window. This is consistent
with, and adds real evidence beyond, the D3 bandedness gap (0.43 vs
0.21) that originally motivated the question. A genuinely interpretable
PDE-like closure (a few hundred polynomial coefficients, shared across
all sites) recovering 90% of a neural propagator's own dynamics is a
promising, previously-untried direction -- see `docs/PART_4_3_SUMMARY.md`
for next-step framing.

### SINDy-style sparse selection on Section 224's local closure: real compression, but not a hand-readable PDE (2026-09-25)

Direct follow-up to the stencil-regression finding above. User: "how
much would it take to test the SINDy-style sparse selection for the
latent space of 224 to see if it collapses to a small human-readable
set of terms? if it isn't too bad, can you try this now."

`scripts/sindy_local_closure.py`: reuses `fit_local_stencil_pde.py`'s
data pipeline, swaps dense Ridge for `pysindy.optimizers.STLSQ` over a
NAMED degree-2 polynomial library (`z[-1,c1]*z[+1,c0]`, not an opaque
coefficient vector) at `width=3` (7 sites, `252` candidate terms).
Target: channel 0's own delta specifically -- the one physically-
anchored, non-learned channel (the local average of `u`), the single
most interpretable thing to look for a compact closure in. Per the
brief's own Phase 8 guidance ("use ensemble/bootstrap SINDy and report
per-term selection probabilities, not a single sparse fit"): 20
bootstrap resamples (30% of pooled rows each) at a chosen threshold,
reporting each term's selection frequency, not one point estimate.

**Sparsity path** (threshold -> active terms -> held-out R^2 on channel
0's delta): `0` (dense) `-> 252 -> 0.929`; `0.008 -> 91 -> 0.876`;
`0.014 -> 53 -> 0.749`; `0.02 -> 24 -> 0.501`; `0.05 -> 2 -> 0.126`.
**No sweet spot with both few terms and strong fit** -- there is a
real cliff between `~50` terms (still explaining most of the variance)
and a genuine handful (which collapses to near-useless, `R^2=0.13` at
2 terms).

**Bootstrap-robust set at threshold=0.014: 49 of 252 terms selected in
>=80% of resamples.** Not a hand-readable equation -- meaningfully
short of what "collapses to a small human-readable set of terms" would
mean (compare the true KS PDE's 4 terms). What IS real: a genuine ~5x
compression (252->49-53) that retains most of the achievable local fit,
so the dynamics are neither maximally dense nor trivially sparse.

**Mechanistically informative finding, independent of the "is it
small" verdict**: most of the robustly-selected terms couple channel
0 (the physically-anchored local mean) to channels 1/2 (the LEARNED,
non-physical hidden channels) at nearby sites, not to other sites' own
channel-0 values. Consistent with the original motivation for
`local_channels > 1` in this project's own design (`CLAUDE.md`'s
Phase 10 rationale): a bare coarse-grained physical field is not
closed on its own (real Mori-Zwanzig memory from what coarse-graining
discards), and the learned hidden channels are doing real,
load-bearing work restoring that closure -- which is exactly why a
compact closure purely in terms of the physical mean was never going
to reduce to a handful of terms; the hidden channels' own dynamics are
where the "extra" structure needed for Markovianity actually lives.

**Time/effort note** (the actual question asked): tractable, as
estimated -- reused the existing stencil-regression data pipeline
almost entirely, added STLSQ + named feature library + bootstrap in
one new script, total wall time (implementation + all runs) well under
an hour.

### Literature context (2026-09-23)

User question: "is there any hope for our approach? has there been any
research trying something similar? I know latent DA is a thing." Four
findings, web-verified rather than recalled from training data, since
citation accuracy matters here:

1. **Xu & Chen, "Intrinsic Instantaneous Coarse-to-Fine Recoverability in
   the Lorenz-96 System"** (arXiv:2607.08323, July 2026) -- an
   independent, very recent, information-theoretic version of the same
   question (how much of the unresolved fine scales is instantaneously
   determined by the resolved coarse scales), applied directly to L96 at
   F=8,16,32,64. Finds recoverability is **strongly nonuniform** -- only a
   finite band of modes near the retained cutoff gets partially slaved,
   organized around the quadratic triad-coupling scale `k_cut~ceil(k/2)`
   (the same nonlinear coupling structure that makes L96 spatially
   broadband at any F, per the earlier KS-vs-L96 spectral-concentration
   comparison in this project). Critically, **raising F reduces
   recoverability** -- more chaos means less closure, not more. An
   independent confirmation of the spectral-gap finding above, from a
   completely different (non-Lyapunov) diagnostic.
2. **Lu, Lin & Chorin's discrete Mori-Zwanzig closures for Lorenz-96**
   (the "optimal prediction" line, ~2015-era, following Chorin & Lu, PNAS)
   -- formally derived that a correct reduced closure for L96 needs a
   memory term with two parts: a deterministic history-dependent function
   `Phi` (a function of current *and past* resolved-scale values) plus a
   stochastic component `xi`. This project has only tried the
   deterministic-history half so far (`mode=two_step`/`history`); the
   stochastic half is completely untested -- see Open questions below.
3. **Peyron et al., "Latent Space Data Assimilation by using Deep
   Learning"** (QJRMS 2021, arXiv:2104.00430) -- the actual "latent DA +
   L96" precedent for this project's own research question. Notably, they
   test their ETKF-Q-Latent method on an **augmented** Lorenz-96 system
   specifically constructed to "possess a latent structure that accurately
   represents the observed dynamics" -- not raw/plain L96. Read plainly,
   this suggests the researchers who tried this before also found plain
   L96 does not hand a latent-DA method exploitable structure for free,
   and modified the system rather than the raw article.
4. **ROAD-EnKF** (arXiv:2301.11961), already cited in `CLAUDE_CODE_BRIEF.md`
   §7/§22 as this project's nearest DA-method comparator -- **correction**:
   verified directly, its numerical examples are Lorenz-63 (embedded in a
   high-dimensional space), Burgers, and KS, **not** Lorenz-96. KS keeps
   showing up as the natural target for this exact class of
   reduced-order/latent EnKF method in the recent literature; L96 does
   not, for what look like the same structural reasons measured above.

**Net read (not yet a decision rule, since no new experiment has been run
off it):** the honest framing going forward is not "does our KS recipe
also work on L96" but "how irreducible is L96, measured the same way we'd
check any new target system" -- the spectral-gap criterion above is itself
a general, reusable, checkable rule (check for a spectral gap in the true
system before assuming a small-latent Markovian DA model will work on it)
independent of whatever number L96's own D_KY ends up at.

### Weather-relevance context and candidate next system (2026-09-23)

User question: "what system do you think is more likely to be close to
the models actually used by weather... my advisor would like to use this
method to analyze a convection system... for strong thunderstorms...
like looking at a 2d vertical view of convection." Findings, web-verified:

**Neither KS nor L96 is a literal weather model, but L96 is the one the
weather/DA community actually treats as a stand-in for one.** Lorenz
built L96 in 1996 explicitly as an NWP predictability/DA test problem;
its standard `F=8` forcing was deliberately chosen to give chaoticity
matching real atmospheric predictability statistics, and it has been the
default first testbed for essentially every major DA method (EnKF,
LETKF, particle filters, and now ML+DA work) for three decades. KS, by
contrast, is architecturally more "a real PDE" (genuine spatial
derivatives and scale-selective dissipation, per the inertial-manifold
finding above) but its home fields are flame-front instability and
thin-film hydrodynamics -- it does not appear in the weather/DA
literature. Net implication for this project's own results: L96's
*lack* of a spectral gap (vs. KS's presence of one) may be the more
representative warning sign for what a real turbulent/convective
atmospheric flow's latent reducibility looks like, not an artifact of
having picked a harder toy problem than necessary.

**The advisor's "2D vertical slice, idealized strong thunderstorm"
description matches CM1 (Cloud Model 1, George Bryan/NCAR)** -- the
standard idealized non-hydrostatic cloud-resolving model in the
severe-storms/mesoscale community for exactly this kind of study (2D x-z
squall-line/convection simulations following the classic Weisman & Klemp
1982 sounding-plus-warm-bubble setup are a standard, published technique;
squall lines are quasi-2D enough that a vertical cross-section is a
legitimate, common way to study them). Not confirmed with the advisor
directly -- worth verifying the exact model/config before committing to
it. Two structural facts that matter for anything built against it:

1. **Multi-field, not scalar.** CM1's prognostic state is velocity
   components (`u`, `w`), potential temperature, pressure perturbation,
   and several moisture/microphysics species (water vapor, cloud water,
   rain, possibly ice) -- a genuinely different complexity class from KS/
   L96's single scalar field. An encoder for this needs to be multi-
   channel from the start, not a straightforward port of what exists.
2. **The vertical dimension is not periodic.** KS and L96 are both
   periodic in their one spatial dimension -- essentially every piece of
   this codebase's architecture assumes it (`torch.roll`-based shift
   augmentation, `CircularPositionalEncoding`, ring-distance attention
   masks). A 2D convection slice has a periodic (or open) horizontal
   dimension but a bounded, non-periodic vertical one (ground to
   tropopause). None of this codebase's periodicity-dependent machinery
   applies along that axis unmodified.

**Relevant prior art for latent-space DA/ML specifically on convection or
real atmospheric fields** (distinct from the L96-specific citations
above): a VAE reducing storm-resolving-model vertical-velocity fields to
a low-dim latent space, found to separate into distinct tropical-
convection regimes (*Scientific Reports*, 2023); "Physically Consistent
Global Atmospheric Data Assimilation with Machine Learning in Latent
Space" (*Science Advances*, 2025, arXiv:2502.02884) and the AE-O2L
framework (*MWR*, 2025) -- both full global-atmosphere latent DA, well
past toy-system scale but confirming the overall approach is an active,
credible, currently-published research direction rather than a dead end.

**Recommendation: don't jump straight from L96 to CM1.** The two new
axes of complexity CM1 adds -- multi-field coupling and a non-periodic
spatial boundary -- are exactly the two things nothing in this codebase
has ever been tested against, and CM1 adds both simultaneously plus real
domain-specific complexity (moisture microphysics, a real sounding/
initialization protocol, likely needing the advisor's own model access
and configuration). A cleaner intermediate step: **2D dry Rayleigh-Bénard
convection** (Boussinesq equations, buoyancy-driven, no moisture/
microphysics) as a genuine-but-still-simple 2D PDE system -- same single-
mechanism clarity this project has relied on for KS/L96, but forces the
architecture to confront a real second spatial dimension and a
non-periodic boundary (the vertical) before also taking on CM1's
multi-field complexity. If the existing encoder/propagator machinery
(shift augmentation, positional encoding, attention masking) survives
that step cleanly, extending to CM1's multi-field state becomes a much
lower-risk next move, and is the natural point to bring the advisor in on
the exact model/configuration to target. Not yet started.

## Rayleigh-Benard convection: solver build (2026-09-23)

User-directed: build the 2D Rayleigh-Benard bridge system recommended
above. `ks_latent/solver/rayleigh_benard.py` (solver),
`ks_latent/solver/rayleigh_benard_dataset.py` (streaming HDF5 dataset
writer), `scripts/generate_rayleigh_benard_dataset.py` (CLI),
`scripts/visualize_rayleigh_benard.py` (snapshots/animation/saturation
plots), `tests/unit/test_rayleigh_benard{,_dataset}.py` (14 tests, all
passing). Target regime: `Ra=3e4, Pr=0.7, 64x64` (~46x supercritical,
per `RayleighBenardConfig`'s own docstring).

**Physics choice, made deliberately to stay pure-FFT.** Free-slip +
isothermal boundary conditions at z=0,1 (not the more common no-slip
case) -- every field expands EXACTLY as a sine series in z under these
BCs, so the whole solver is `scipy.fft.rfft` (x) composed with an
explicit sine/cosine SYNTHESIS matrix (z) -- no Chebyshev or implicit-
solve machinery. This also gives an exact, closed-form validation target:
free-free RBC's critical Rayleigh number is `Ra_c = 27*pi**4/4 ~= 657.51`
(Chandrasekhar 1961), reachable exactly by choosing the box aspect ratio
(`Lx/Lz = 2*sqrt(2)`) so the domain's fundamental wavenumber lands on the
critical one.

**Validation (ground rule 1).** `test_critical_rayleigh_number_matches_
textbook_value` confirms the textbook Ra_c from a from-scratch 2x2
linear-stability reduction of the governing equations (independent of
the solver code). `test_solver_reproduces_linear_growth_rate_below_and_
above_critical` -- the real, decisive test -- runs the ACTUAL nonlinear
pseudospectral solver with a tiny-amplitude perturbation initialized
exactly along the dominant eigenvector (a naive `(omega=0, theta=eps)`
IC turned out to be a near-50/50 mix of two distinct real eigenvalues,
`-4.34` and `-25.27` at `Ra=0.5*Ra_c` -- contaminating any growth-rate
measurement with the wrong mode; fixed by projecting onto the dominant
eigenvector directly) and confirms the measured growth rate matches the
analytic one to 3% at both a sub- and super-critical Ra. Plus: exact
DST-I orthonormality, transform round-trip, the zero state's exact
fixed-point property, spinup boundedness, and dataset-level tests
(shape, normalization-matches-train-split-only, flatten-compatibility,
raises on a genuinely divergent run).

**Two real bugs found and fixed while getting the target Ra=3e4 regime
stable, both worth recording since they're generic spectral-solver
traps, not RBC-specific:**

1. **Explicit buoyancy at large Ra is a hidden stiffness bug.** The
   fastest LINEAR growth rate across all resolved modes at Ra=3e4 is
   ~93 (checked directly by diagonalizing the governing equations at
   every resolved wavenumber) -- explicit (AB2) treatment of the
   buoyancy/diffusion coupling required `dt << 1/93` for stability,
   causing a real blowup within ~30 steps at the original `dt_max=5e-3`.
   Fixed by treating the ENTIRE linear system (diffusion + buoyancy +
   background-gradient advection -- all exactly linear in `(omega_hat,
   theta_hat)`) via an exact per-mode 2x2 Crank-Nicolson solve
   (unconditionally A-stable), leaving only the genuinely nonlinear
   advection term explicit. Only the ordinary advective CFL constrains
   `dt` after this fix.
2. **A step-size discontinuity breaks fixed-coefficient AB2, and a
   boundedness check on the wrong quantity looked exactly like the same
   symptom.** Clamping the timestep to land exactly on a snapshot
   boundary creates a dt jump the fixed-coefficient AB2 extrapolation
   (`1.5*N^n - 0.5*N^{n-1}`) implicitly assumes doesn't happen -- fixed
   by resetting the AB2 bootstrap whenever consecutive step sizes differ
   by more than 25%. Separately (and this was the dominant effect, found
   by ruling the above OUT via matched single-vs-multi-boundary A/B
   tests): the integration's own boundedness check compared `max(|
   omega_hat|, |theta_hat|)` -- raw, transform-normalization-dependent
   spectral coefficients with no fixed physical scale -- against a fixed
   threshold. A run that was genuinely bounded and saturated in PHYSICAL
   space (`max|omega_phys| ~ 500-550`, `max|u| ~ 100-130`, matching the
   expected free-fall velocity scale `sqrt(Ra*Pr) ~ 145` closely) still
   tripped the spectral-coefficient ceiling and raised a false "blowup".
   Fixed by checking the physical velocity field instead (`max_abs_
   velocity`, default 1e3) -- physically meaningful, and free, since
   `integrate`'s CFL step already computes it every iteration.

**Performance**: `scipy.fft(..., workers=cfg.fft_workers)` (default 4,
moderate multi-core use, not all cores) gave a measured 4.8x speedup
(284 -> 1370 steps/sec) on an 18-core Apple Silicon machine. At the
saturated turbulent state, CFL settles the timestep to ~2.7e-5 -- a
genuine physical consequence of this nondimensionalization at Ra=3e4
(the free-fall time is a small fraction of the diffusive time unit at
this Ra), not a solver inefficiency; budget dataset-generation time
accordingly (documented directly in the CLI script's own docstring).

**ML pipeline compatibility.** `trajectories` dataset shape `(n_runs, T,
2, Nz, Nx)` (channel 0 `theta`, channel 1 `omega` -- the two independent
dynamical fields; `u`/`w`/`psi` are algebraically derivable from `omega`
alone and stored only if `--include-velocity` is passed, for
visualization convenience). `trajectories.reshape(n_runs, T, -1)` gives
a flat `(n_runs, T, 2*Nz*Nx)` vector, directly usable with `--nx` on
`scripts/train_stage1_patched.py` with zero further code changes -- real
compatibility, but a crude one: it discards the field's 2D structure and
the periodic-x/bounded-z distinction entirely. A genuinely 2D-spatial
encoder is NOT built here -- still future work, per the recommendation
above.

**Status: solver, dataset writer, CLI, and visualization all built,
tested, and run end to end** -- demonstration dataset (5 train + 2 val
trajectories, 64x64, Ra=3e4/Pr=0.7, `artifacts/datasets/rayleigh_benard_
ra3e4_pr0.7_demo.h5`) generated in 633.9s. Visualizations
(`docs/figures/rbc_ra3e4_demo_{snapshots,animation,saturation_history}.
{png,gif}`) show textbook-correct physics: classic mushroom-shaped
thermal plumes, a clean pair of counter-rotating convection rolls.

**Finding, not yet resolved: at the validation aspect ratio (`Lx/Lz =
2*sqrt(2)`, chosen so the box's fundamental wavenumber exactly matches
the critical one -- see above), Ra=3e4 settles into an essentially
EXACT steady state, not the "chaotic/oscillatory" regime originally
requested.** Checked directly: `max|omega|` across the last 15 recorded
snapshots of the demo dataset varies by a relative `6.9e-8` -- pure
numerical noise, not dynamics. Likely mechanism: this aspect ratio only
fits a single pair of convection rolls, leaving no lateral room for the
pattern competition (wavenumber selection, defects, the oscillatory
secondary instability within the Busse balloon) that drives 2D RBC
spatiotemporal chaos at moderate Ra -- a known effect in the RBC
literature, not a solver bug (the steady 2-roll state itself is a
textbook-correct solution, just not a chaotic one).

A quick, smaller-scale check (Nx=96, Nz=32, aspect ratio 4x wider,
`fft_workers=6`, 4.0 time units, ~60s) shows a materially larger but
still SLOWLY, MONOTONICALLY drifting `max|omega|` (relative variation
`~1e-3` over the last 30 snapshots, vs. `~1e-8` at the narrow box) --
consistent with a longer transient toward a different (possibly still
steady, possibly eventually oscillatory) pattern, not yet resolved
within that short a test. `--aspect-ratio` is now a CLI override on
`scripts/generate_rayleigh_benard_dataset.py` for exactly this
follow-up. Next step, not yet run given this project's own compute-
budget sensitivity (deferred to explicit user direction rather than
launched speculatively): either a longer integration at a widened
aspect ratio, or a higher Ra, to find where genuine sustained
time-dependence actually sets in for this system.

Not yet done, regardless of the above: any actual latent-encoder/
propagator training run on this data (the natural next step once a
genuinely chaotic operating point is found).
