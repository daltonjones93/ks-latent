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
