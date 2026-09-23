# Phase 2 Architecture Experiments: Chasing the Propagator Collapse

**Status:** Living document, started 2026-08-30. Records every architecture/training
change tried to fix the Stage-2 propagator's fixed-point collapse, the exact
configuration used, and the exact numeric result -- so nothing has to be re-derived
or re-run to answer "did we try X, and what happened."

**Headline finding so far (Section 9)**: the best result of the entire
investigation is not either "Solution" originally proposed -- it's a plain
residual-MLP propagator (`backbone="mlp"`, the brief's original default,
no attention, no FNO, no ensemble), trained with the same
restructured-AE/`delta_cap`/slow-k-curriculum recipe as everything else in
this document. It recovers the same rich chaos as the FNO+ViT hybrid
(`D_KY~21` both) while being 7x more accurate on rollout and ~2x better on
DA calibration. The pattern that has held up across every architecture
tried: **every propagator using softmax self-attention for token-mixing
collapsed (`D_KY=0`), regardless of window size (local or global); every
one that didn't (`mlp`, `fno_vit`) recovered chaos.** See Section 9 for
the full comparison and Section 5 for how this was discovered.

---

## 0. The problem, precisely

The trained Stage-2 latent propagator, run autonomously (no re-anchoring to data),
converges to a **single fixed point** from every tested initial condition within
~100-200 steps. Measured consequences:

- Lyapunov spectrum: `D_KY = 0`, `n_positive = 0` (no chaos at all), `lambda1 < 0`
  (contracting). The reference project's own prior work on this same problem
  achieved genuine chaos (`D_KY ~= 21.4`, 11 positive exponents, matching the
  literature value `D_KY ~= 22-23` for KS at L=100) -- see `docs/PROJECT_HANDOFF.md`.
- DA ensemble collapse: `calibration = spread/rmse` far below the ideal ~1.0
  (ensemble members converge toward each other, exactly what a contracting map
  autonomously does to any ensemble).

Root cause (established via direct evidence, not speculation): plain MSE training
on a `k_max`-step rollout window has no way to penalize what the learned map does
*beyond* that window. For a chaotic target, the MSE-optimal prediction past a few
Lyapunov times is the conditional mean -- on a chaotic attractor, that is the
climatological average, a fixed point independent of the input. This is a property
of **the loss objective given the training horizon**, not of any one architecture
choice, which is why architecture-only fixes (delta/residual parameterization,
zero-init) are necessary for stability but not sufficient to prevent it.

## 1. What was already tried (same-day, prior to this document)

All against the canonical Stage-1 AE (`vit` encoder/decoder, `attn_window=4`,
`pos_encoding=linear`) unless noted. `val_kmax_mse` values are **not** comparable
across different `k_max` settings (evaluated over a longer/harder horizon at
higher `k_max`).

| # | Change | k_max | val_kmax_mse | D_KY | lambda1 | DA calibration |
|---|---|---|---|---|---|---|
| 1 | Original markovian vit propagator, straight linear k-curriculum | 16 | 0.6916 | -- | -- | -- |
| 2 | Two-segment k-curriculum (dwell at low k) | 32 | 0.8132 | 0 | -0.106 | 0.066 |
| 3 | + `w_varmatch=0.1` (rollout variance-matching loss) | 32 | 0.8132 | (same run as #2; stacked) | | |
| 4 | + `noise_step` (genuine per-step rollout noise, not just input noise) | 32 | ~0.888 | (killed before Gate3/4) | | |
| 5 | + `delta_cap=0.5` (tanh-saturated residual, "stretch-and-fold" architecture) | 32 | ~0.89 | (killed before Gate3/4, GPU contention) | | |
| 6 | `mode="history"` propagator (n=3 states, joint spatio-temporal ViT attention), same k_max=32 recipe | 32 | (superseded before finishing) | | | |
| 7 | `mode="history"`, **much slower k-curriculum, k_max capped at 8** (long dwell at k=2..5) | 8 | 0.4099 | **0** | **-0.057** | **0.112** |

Row 7 is the best result to date: `lambda1` moved from -0.106 to -0.057 (~46%
closer to zero), DA calibration nearly doubled (0.066 -> 0.112). Still non-chaotic.
Full run parameters for row 7:

```
python scripts/train_stage2_patched.py --profile full --mode history --n-history 3 \
  --ae-checkpoint artifacts/stage1_ae_patched_full.pt \
  --backbone vit --pos-encoding linear --attn-window 4 \
  --prop-n-tokens 44 --prop-token-d-model 64 \
  --device mps --epochs 68 --k-warmup-epochs 48 --k-mid 5 --k-mid-epochs 40 \
  --k-max 8 --w-varmatch 0.1 \
  --noise-in-start 0.15 --noise-in-end 0.05 --noise-step-start 0.05 --noise-step-end 0.01 \
  --delta-cap 0.5
```
Checkpoint: `artifacts/stage2_prop_patched_full_v5_history3_kmax8_68ep.pt` (backed up).

### 1.1 Stage-1 side-experiments (same-day)

| # | Change | val_recon_final | Note |
|---|---|---|---|
| A | Canonical (vit, attn_window=4, pos_encoding=linear, mean pool) | 0.000494 | baseline |
| B | `pool="local"` (windowed mean pool, pool_window=8) | 0.000984 | 2x worse than global pool |
| C | `pool="local_attn"` (learned local attention pool, pool_window=8) | ~similar to B | learned pooling didn't close the gap much |
| D | `w_pred=1.0, k_pred_max=4` (heavier + longer joint dynamics-shaping term) | 0.000770 | worse recon, dynamics effect on Stage-2 not fully isolated |
| E | `token_window=16` (overlapping-patch tokenization, encoder only) | **0.000413** | best single-change result |
| F | `token_window=16 + k_pred_max=4` | 0.000476 | combining E with D partially erodes E's gain |
| G | `mode="history"` aux (n=3) + `w_var=0.15` + `token_window=16` | **FAILED -- shortcut collapse** | see §1.2 |
| H | `mode="history"` aux (n=3) + `w_var=0.05` + `token_window=16` | **0.000440** | second-best result; fixed G |

### 1.2 A caught failure mode: `w_var` shortcut collapse

Raising `Stage1TrainingConfig.w_var` from 0.01 to 0.15 (15x) caused `recon` to sit
completely flat at ~1.0 (the variance of the normalized data) for 30+ epochs --
diagnosed as a **shortcut collapse**: the encoder can satisfy "each latent
dimension has unit variance across the batch" with input-independent noise,
without doing the harder job of actually encoding the input. Variance alone does
not force the code to carry real information; only an invariance/reconstruction
term does that (the same reason VICReg-style objectives need variance *and*
invariance terms together, never variance alone at high weight). Fixed by using
`w_var=0.05` (5x, not 15x) instead -- see row H above. **Lesson: `w_var` has a
real ceiling before it stops being a mild regularizer and starts being an easy
out.** Recommend never exceeding ~0.05-0.10 without checking recon isn't flat in
the first ~20 epochs.

## 2. New direction (this document), user-directed 2026-08-30

Three changes requested together:

1. **Restructure the phased training.** Instead of Phase 1 training a small
   *auxiliary* propagator alongside the AE (discarded after Stage 1) and Phase 2
   training the *real* propagator from scratch, Phase 1 should train the **same,
   full-sized** propagator jointly with the AE from the start. Phase 2 then
   *fine-tunes that same propagator* for longer rollouts, rather than starting
   over. Rationale (user's stated hypothesis): starting Phase 2 from scratch may
   be part of why collapse is "inevitable" in the current setup -- a propagator
   that already has some joint-training-shaped structure before the long-rollout
   phase begins might behave differently.
2. **Solution 1: ensemble propagator.** An ensemble of M small variations of the
   propagator body. Each member gets diagonal Gaussian noise injected at every
   step (variance proportional to that latent dimension's own variance, no
   off-diagonal/cross-dimension noise). The ensemble prediction is a (possibly
   learned, z-dependent) weighted mean of the members.
3. **Solution 2: FNO + ViT hybrid.** A Fourier Neural Operator (spectral
   convolution via FFT, truncated to low modes) combined with ViT-style blocks
   on the tokenized latent -- a genuinely different modeling paradigm, motivated
   by FNO's use in physics/PDE surrogates and the hope that spectral structure
   preserves spatial relationships in a way that helps the latent's dynamics.

User's explicit framing: "I suspect the current training will also collapse" --
this is stated as a real possibility going in, not something to avoid reporting.
Every experiment below is documented regardless of outcome.

---

## 3. Phased-training restructure: implementation

Landed in `scripts/train_stage1_patched.py` and `scripts/train_stage2_patched.py`.

**Prerequisite fixes** (`ks_latent/config.py`, `ks_latent/models/propagator.py`):
`AuxPropagatorConfig` was missing a `delta_cap` field (present only on
`PropagatorConfig`), needed so Phase 1 can train a propagator that is
architecture-identical to Phase 2's, `delta_cap` included. Added
`delta_cap: float | None = None` with the same `> 0` validation
`PropagatorConfig` has. The private `_aux_cfg_to_propagator_cfg` helper (the
existing `AuxPropagatorConfig -> PropagatorConfig` converter, already used
internally by `AuxPropagator.__init__`) was made public
(`aux_cfg_to_propagator_cfg`) since `train_stage1_patched.py` now needs to
call it directly. Fast suite (235 tests) re-run clean after both changes.

**`train_stage1_patched.py --full-propagator`** (new flag): forces
Stage-2-standard sizing (`hidden=128, n_blocks=3`) on the Phase-1 aux
propagator regardless of `--multistep`, and accepts `--prop-delta-cap` to set
`AuxPropagatorConfig.delta_cap`. After `train_stage1` finishes, the trained
aux propagator is converted via `aux_cfg_to_propagator_cfg` and saved as its
own checkpoint, `artifacts/stage1_prop_{profile}{tag}.pt`, in the exact
`{"prop_state_dict", "prop_config"}` format `train_stage2_patched.py` already
uses for its own output -- this is the only contract that matters, since it's
what makes the checkpoint loadable by Phase 2 without any format-conversion
code on that side.

**`train_stage2_patched.py --init-prop-checkpoint <path>`** (new flag): loads
`prop_config`/`prop_state_dict` from the given checkpoint instead of building
a fresh `PropagatorConfig` from this script's own `--backbone`/`--mode`/
`--n-history`/`--attn-window`/etc CLI flags (those are ignored when this flag
is given -- fine-tuning requires the identical architecture the checkpoint
was trained with). The existing `--k-max`/`--epochs`/`--w-varmatch`/
`--noise-*`/`--delta-cap` schedule flags are unaffected and now serve as the
fine-tuning schedule applied on top of the loaded weights. Also added
`--tag` (stage2 never had one) so a fine-tuning run's output checkpoint
doesn't clobber a from-scratch comparison run's.

**Validation** (smoke profile, CPU/MPS, `d_latent=8`): ran
`train_stage1_patched.py --profile smoke --full-propagator --prop-delta-cap 0.5`,
confirmed it wrote `stage1_prop_smoke_....pt`, then ran
`train_stage2_patched.py --profile smoke --init-prop-checkpoint stage1_prop_smoke_....pt`,
confirmed the printed architecture (`backbone=mlp, mode=markovian`) matched
what Phase 1 had trained, and that fine-tuning proceeded normally
(`val_kmax_mse` fell from 0.0012 to 0.00003 over 2 epochs against a
consistent d_latent=8 AE). Smoke artifacts deleted after validation --
this was a round-trip correctness check, not a real experiment.

**Full-scale run launched** (2026-08-30, PID 45440/45441, MPS): Phase 1,
best-known Stage-1 recipe (`--encoder vit --aux-backbone vit --mode history
--n-history 3 --pos-encoding linear --attn-window 4 --token-window 16
--aux-n-tokens 44 --aux-token-d-model 64 --w-var 0.05`) plus the two new
flags (`--full-propagator --prop-delta-cap 0.5`), tag
`history3_fullprop_wvar005_tw16`. 100 epochs, ~9.6s/epoch on MPS. Recon
trajectory tracking cleanly downward (no `w_var` shortcut-collapse
signature -- recon at epoch 60/100 was 0.000794, well below the ~1.0 flat
line row G hit) -- log: `artifacts/logs/stage1_history3_fullprop_wvar005_tw16.log`.
Phase 2 fine-tune (`--init-prop-checkpoint
artifacts/stage1_prop_full_history3_fullprop_wvar005_tw16.pt`, same k-max=8
slow-curriculum recipe as row 7) to follow once Phase 1 completes; result
will be appended here.

## 4. Solution 1: ensemble propagator

Implemented as a new class, `ks_latent.models.ensemble_propagator.
EnsemblePropagator`, configured by a new `EnsemblePropagatorConfig`
(`ks_latent/config.py`) -- not folded into `PropagatorConfig` itself, since
an ensemble wraps a *member* architecture rather than being one itself
(`EnsemblePropagatorConfig.member: PropagatorConfig`).

**Design, matching the user's spec exactly except one documented
deviation:**
- `cfg.n_members` independently-initialized copies of `cfg.member`'s
  architecture (a full `LatentPropagator` each -- same class used
  everywhere else in this codebase, so every backbone/mode combination
  that already exists is available to ensemble members for free).
- Diagonal Gaussian noise injected into each member's input at every step,
  std `= noise_std_frac * sqrt(Var(z_i))` per latent dimension `i` -- no
  off-diagonal covariance, exactly as specified. `Var(z_i)` is measured
  from the encoded training set (`train_seq.var(dim=(0,1))` in
  `train_stage2_patched.py`) and pushed into the module via
  `set_latent_var(...)` right after construction, since it's data, not a
  hyperparameter (configs must stay plain/hashable for provenance).
- Combined output is a weighted mean of the `n_members` predictions,
  weights from a small softmax gate conditioned on the (un-noised)
  reference state `z` -- `learned_weights=True` (default) makes the gate
  a real `Linear -> GELU -> Linear` network; `learned_weights=False` uses
  a fixed uniform `1/n_members` mean instead.
- **Documented deviation**: each member computes `z_next_m = z +
  capped_delta_m(body_m(z + eps_m))` -- the noise perturbs only the INPUT
  to the residual body, not the residual's own additive base. A literal
  reading of the spec (noise added to "the current input") could also mean
  perturbing the additive base too (`z_next_m = (z+eps_m) +
  delta_m(z+eps_m)`), but that breaks `zero_init`'s identity-at-init
  property that every other propagator in this codebase preserves exactly
  (verified this breaks it empirically -- see below) -- with the additive
  base perturbed, a freshly-initialized ensemble outputs `z + eps_m`
  instead of `z`, i.e. hasn't learned anything yet but is already
  non-identity. Kept the clean-base design since it's the only one
  consistent with `test_propagator_is_identity_at_init`'s invariant
  everywhere else, and is functionally equivalent for what the noise is
  *for* (giving each member a different view when computing its
  correction), while being strictly better-behaved at the start of
  training.

**Drop-in compatibility**: `EnsemblePropagator` exposes the same
`.mode`/`.step_one`/`.step`/`.rollout` (markovian) or
`.step_history`/`.rollout_history` (history) interface as `LatentPropagator`,
plus `.cfg.d_latent`/`.cfg.n_history` proxy properties on
`EnsemblePropagatorConfig` (delegating to `.member`) -- these are the only
attributes Gate 3/4 (`lyapunov.py`, `run_diagnostics.py`,
`run_analysis_suite.py`, `run_da_pff.py`) and `train_stage2`'s loop read off
a propagator/its config, confirmed by grep before writing this. No changes
needed to any of those consumers.

**A real bug caught and fixed while wiring this up**: every Gate 3/4 script
hardcoded `prop = LatentPropagator(prop_cfg)` when loading a checkpoint --
exactly the bug class `ks_latent/models/__init__.py`'s
`load_autoencoder_checkpoint` was already added to prevent on the
autoencoder side (silently wrong class for a config type that doesn't
match). Fixed by adding the analogous `build_propagator_from_config`/
`load_propagator_checkpoint` to `ks_latent/models/__init__.py`, dispatching
on `type(prop_cfg)` (`EnsemblePropagatorConfig` -> `EnsemblePropagator`,
`AuxPropagatorConfig` -> `AuxPropagator`, else `PropagatorConfig` ->
`LatentPropagator`), and updating `run_da_pff.py`, `run_diagnostics.py`,
`run_analysis_suite.py`, and `train_stage2_patched.py`'s
`--init-prop-checkpoint` loader to use it. Also renamed
`LatentPropagator._capped_delta` to public `capped_delta` since
`EnsemblePropagator` needs to call it directly on each member (to implement
the clean-base/noised-delta split above) without reaching into a
double-underscore-adjacent "private" method.

**CLI**: `train_stage2_patched.py --ensemble` wraps whatever architecture
`--backbone`/`--mode`/etc already select into an ensemble;
`--ensemble-n-members` (default 5), `--ensemble-noise-std-frac` (default
0.05), `--ensemble-learned-weights`/`--ensemble-uniform-weights` (default
learned), `--ensemble-gate-hidden` (default 32). Ignored when
`--init-prop-checkpoint` is set (that loads whatever architecture, ensemble
or not, the checkpoint already has).

**Testing**: `tests/unit/test_ensemble_propagator.py` (11 tests) --
config validation (`n_members<1`, negative `noise_std_frac`, `two_step`
member all raise), `.cfg.d_latent`/`.cfg.n_history` proxying,
identity-at-init for both markovian and history modes *regardless of
noise_std_frac* (this is the test that caught the additive-base bug during
development: an earlier implementation that called `member.step_one(z +
eps)` directly, rather than reaching into `member.body`/`capped_delta`
separately, failed this test with a large mismatch -- exactly the
deviation discussed above, caught before it reached a real training run),
shape/wrong-mode-raises checks for both interfaces, uniform-weights-equals-
plain-mean-of-members when `learned_weights=False` and `noise_std_frac=0`,
learned gate starts exactly uniform at init (zero-initialized gate output
layer), and `set_latent_var` shape validation + noise scaling. Fast suite:
246/246 passing after this addition (was 235 before this document's work
began).

**Smoke-tested end-to-end** (CPU, `--profile smoke --ensemble`): ran
`train_stage2_patched.py --profile smoke --ensemble --device cpu`, trained 2
epochs, printed `latent_var range [1.99e-05, 1.18e-04]` (confirming
`set_latent_var` picked up real, non-unit, anisotropic per-dimension
variance from the encoded smoke dataset, not a placeholder), saved a
checkpoint, reloaded it via `load_propagator_checkpoint` (confirmed
`isinstance(prop, EnsemblePropagator)` and `type(prop_cfg) is
EnsemblePropagatorConfig`), and ran a `.rollout(...)` on the reloaded model
-- correct shape, no errors. Also confirmed all three Gate 3/4 scripts still
import and parse `--help` cleanly after the refactor. Smoke artifacts
deleted after validation (round-trip correctness check, not a real
experiment).

**Phase 1 restructure run completed** (2026-08-30): 100 epochs,
`val_recon_final = 0.000401` -- a new best Stage-1 recon result, beating row
H's 0.000440 and row E's 0.000413 (see Section 1.1's table). No shortcut
collapse (recon fell smoothly from 1.006 to 0.0004, never plateaued flat).
Wrote `artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16.pt`
and `artifacts/stage1_prop_full_history3_fullprop_wvar005_tw16.pt`.

**Phase 2 fine-tune launched** immediately after (PID 46331, MPS):
`--init-prop-checkpoint artifacts/stage1_prop_full_history3_fullprop_wvar005_tw16.pt`,
same slow k-curriculum recipe as row 7 (`--k-max 8 --k-warmup-epochs 48
--k-mid 5 --k-mid-epochs 40 --w-varmatch 0.1`, same noise schedule), tag
`fullprop_finetune`. ~13s/epoch, 68 epochs (~15 min total). Log:
`artifacts/logs/stage2_fullprop_finetune.log`. Result (Gate 3/4 included)
will be appended here once complete.

**A cosmetic bug caught and fixed while launching this**: the run-start log
line printed `args.backbone` (the CLI flag, defaulting to `"mlp"`) instead
of the actually-loaded architecture whenever `--init-prop-checkpoint` was
used -- harmless (training used the correct loaded config throughout; only
the log line was wrong) but confusing when reading logs later. Fixed to
print `type(prop_cfg).__name__` instead.

**Phase 2 fine-tune result**: `best_val_kmax_mse = 0.383520` (vs row 7's
0.4099 at the same `k_max=8` recipe, trained from scratch -- an ~6.4%
improvement from starting Phase 2 with a Phase-1-jointly-trained
propagator instead of a random one). Checkpoint:
`artifacts/stage2_prop_patched_full_fullprop_finetune.pt`.

**Gate 3/4 results (row 8)**:

| Metric | Row 2 (original) | Row 7 (history, kmax=8) | **Row 8 (full-propagator restructure)** |
|---|---|---|---|
| `val_kmax_mse` | 0.8132 | 0.4099 | **0.3835** |
| `D_KY` | 0 | 0 | **0** |
| `lambda1` | -0.106 | -0.057 | **-0.046** |
| DA calibration (spread/rmse) | 0.066 | 0.112 | **0.175** |

Still **not chaotic** (`D_KY=0`, `n_positive=0`) -- the fixed-point collapse
persists under the restructured phased training, confirming the user's own
prediction going in ("I suspect the current training will also collapse").
But every scalar that measures "how close to the edge of chaos" moved
further in the right direction than any single prior change: `lambda1`
is the closest to zero yet (56% closer to zero than row 2, 19% closer than
row 7), and DA calibration nearly tripled versus row 2 (0.066 -> 0.175) and
is 56% higher than row 7's own already-best 0.112. Full Gate 3 JSON:
`artifacts/analysis_suite_full.json`, `artifacts/da_pff_full.json` (saved
before being overwritten by later runs -- see the ensemble/FNO+ViT results
below for their own equivalents). D3 (Jacobian coupling bandedness) stayed
significant (0.9888, p=0.0000) and D4 found no clean linear translation
representation -- both consistent with prior runs, not new findings from
this restructure specifically.

**Full-scale ensemble training run completed** (2026-08-30, tag
`ensemble5_vit_markovian`): 5 members, each a `vit`/`markovian` propagator
(the same architecture size as row 2/3 -- `attn_window=4`,
`pos_encoding=linear`, `prop_n_tokens=44`, `token_d_model=64`,
`delta_cap=0.5`), `noise_std_frac=0.05`, learned gate, same slow
k-curriculum recipe as row 7/8 (`k_max=8`, `w_varmatch=0.1`, same noise
schedule), trained on top of the same restructured-pipeline AE
(`stage1_ae_patched_full_history3_fullprop_wvar005_tw16.pt`). Chose
`mode="markovian"` rather than `"history"` for the members specifically to
keep the 5x member-count compute multiplier tractable -- an ensemble of 5
`history`-mode members (the row 7/8 architecture) would have been
prohibitively slow. `best_val_kmax_mse = 0.4599` -- worse than row 8's
single deterministic fine-tuned propagator (0.3835), but not a fair
apples-to-apples comparison (different member architecture AND trained
from scratch rather than restructured/fine-tuned). Checkpoint:
`artifacts/stage2_prop_patched_full_ensemble5_vit_markovian.pt`.

**A wall-clock anomaly observed, not yet explained**: most epochs took the
expected ~17-35s, but a meaningful fraction randomly took 90-1100s (10-60x
slower) throughout the run, including late epochs long after the
concurrently-running Gate-3/4 CPU jobs (row 8's analysis) had finished --
so CPU contention from those jobs is NOT a sufficient explanation, though
it may have contributed to the early instances. No errors, no change in
loss trajectory -- purely a wall-clock effect, most likely macOS
thermal/power-management throttling on a laptop during a long unattended
run (untested hypothesis; not investigated further since it doesn't affect
correctness, only turnaround time). Documented as a caveat on any future
run's wall-clock estimates on this machine, not as a bug in the code.

**Gate 3 results** (tag `ensemble5`): DA cycling calibration
(`spread/rmse`) = **0.1026** -- worse than row 8's 0.175, better than row
2's 0.066; `skill_free_over_da` = 1.266 (free-running rollout actually
scored slightly *better* than the DA-corrected one on this metric, which
is a DA-quality question more than a chaos question).

**A real bug caught and fixed running the Lyapunov spectrum on this
checkpoint**: it crashed --
`RuntimeError: vmap: called random operation while in randomness error
mode` -- because `EnsemblePropagator._noise` calls `torch.randn` on every
`.step()`, and the Lyapunov tangent-map linearization
(`ks_latent/analysis/lyapunov.py`) computes the propagator's Jacobian via
`torch.func.vmap(jvp(...))`, which requires the wrapped function to be
deterministic. Lyapunov exponents are only well-defined for a deterministic
map in the first place (a genuinely stochastic per-step map needs random-
dynamical-systems theory, e.g. frozen/coupled noise paths across the
reference and perturbed trajectories -- out of scope here). Fixed with a
`noise_disabled()` context manager on `EnsemblePropagator` (returns exact
zeros from `_noise` while active) that `lyapunov.py`'s three tangent-map
adapters (`_make_single_state_step_fn`/`_make_two_step_step_fn`/
`_make_history_step_fn`) now wrap ONLY around the `vmap(jvp(...))` call --
the reference trajectory's own advancement (`z_next = g(z)`, computed just
before that block) still uses the model's real, noisy dynamics, so the
Lyapunov exponent reported is the local stability of the deterministic
"skeleton" map, evaluated along the actual (noisy) trajectory the ensemble
visits -- not the stability of a wholly noise-free model. Dispatch is
duck-typed (`getattr(propagator, "noise_disabled", None)`), so every other
propagator in this codebase (already deterministic) is unaffected. Added a
regression test (`test_single_state_wiring_on_ensemble_propagator` in
`tests/unit/test_lyapunov_latent.py`) that runs the Lyapunov adapter
end-to-end on a small `EnsemblePropagator` and asserts a finite spectrum
comes back. Fast suite: 258/258 passing (was 257 before this fix's test).
Gate 3 Lyapunov re-run launched immediately after this fix; result below.

**Result -- the first positive Lyapunov exponent of the entire session**:
`single_state` mode: `D_KY = 1.011`, `n_positive = 1`, `lambda1 = +0.00023`.
`two_step` mode (a self-consistency check on the same underlying map):
`D_KY = 0`, `lambda1 = -0.00081`. Every other architecture tried this
session (original, k-curriculum, `delta_cap`, `history` mode, the
restructured full-propagator, and the FNO+ViT propagator below) came back
strictly negative / `D_KY=0`. This is genuinely marginal, not a
breakthrough: `lambda1` is barely distinguishable from zero in either
direction, and the two evaluation modes disagree on its sign -- "just
barely, fragile-ly chaotic at best," nothing like the reference project's
`D_KY~21.4`. Plausible interpretation: the per-member noise injection
(genuinely present at eval/rollout time too, not just training -- see
`noise_disabled()`'s docstring) prevents literal convergence to a single
point the way one deterministic contracting map does, nudging the
ensemble's effective dynamics to the immediate edge of chaos rather than
solidly into it. Not yet verified for robustness (a different seed, or
more Lyapunov directions/longer warmup, could easily flip the sign back to
negative) -- flagged as a fragile, suggestive-not-conclusive result.

**A real overwrite bug caught while running this**: `run_analysis_suite.py`/
`run_da_pff.py`/`run_diagnostics.py` had no `--tag` (unlike
`train_stage1/2_patched.py`), so their fixed output filenames
(`analysis_suite_full.json`, `da_pff_full.json`,
`docs/diagnostics_report.md`) were silently overwritten by each new Gate
3/4 run -- row 8's own `da_pff_full.json`/`analysis_suite_full.json` were
already clobbered by this ensemble run's Gate 3 before this was caught
(the printed JSON survived in this document and in
`artifacts/logs/gate3_*_fullprop_finetune.log`, so no data was actually
lost, but the raw JSON artifact is gone). Fixed by adding `--tag` to all
three scripts, same convention as the training scripts. All Gate 3/4 runs
from this point on use a tag matching the checkpoint's own tag.

## 5. Solution 2: FNO + ViT hybrid propagator

New `backbone="fno_vit"` on `PropagatorConfig`/`AuxPropagatorConfig`
(`markovian` mode only, same original scope `"transformer"`/`"vit"` had
before `"history"` was added specifically for `"vit"`). Implemented in
`ks_latent/models/propagator.py` alongside the other backbone bodies (not a
separate file, unlike the ensemble wrapper -- this plugs into
`LatentPropagator.__init__`'s existing per-backbone dispatch rather than
composing other propagators).

**Design**: tokenizes `z` exactly like `_ViTDeltaBody` (same
`circular_overlap_tokenize`/`token_embed`/positional-encoding machinery,
reusing `n_tokens`/`token_d_model`/`token_window`/`pos_encoding`/
`attn_window`), then runs two stages on the token grid, `(B, n_tokens,
d_model)`:

1. `fno_n_layers` `_FNOLayer` blocks: each does a `SpectralConv1d` (FFT
   along the token axis, multiply the lowest `fno_modes` frequency
   components by a learned per-mode `(d_model, d_model)` complex weight,
   zero every higher frequency, inverse FFT) summed with a pointwise
   `Conv1d(kernel_size=1)` skip path, then `GELU` -- the standard FNO layer
   design (Li et al. 2020). `fno_modes=None` (default) keeps every rfft
   mode (no truncation, i.e. the spectral conv is just an exact global
   linear mixing across tokens at that point, with `GELU` supplying the
   only nonlinearity).
2. `token_n_layers` `ViTBlock` attention blocks (identical to
   `_ViTDeltaBody`'s) on the FNO output.
3. `LayerNorm` + `Linear(d_model -> chunk_size)`, zero-initialized (same
   `zero_init` discipline as every other body -- delta is exactly zero at
   init regardless of what the FNO/ViT stages computed, since the final
   Linear's weight and bias are the zero matrix/vector).

An FFT is inherently a periodicity assumption (it treats its input as one
period of a periodic signal) -- so the FNO half's circularity is
unconditional, independent of `pos_encoding`, which only governs the ViT
half's own positional encoding/attention masking (`"circular"` vs
`"linear"`, exactly as for `_ViTDeltaBody`).

**Testing**: extended `tests/unit/test_propagator_modes.py` (11 new tests)
-- `two_step`/`history` + `fno_vit` both raise (same pattern as
`transformer`/`vit`'s own restrictions), `d_latent % n_tokens` divisibility
enforced, `fno_n_layers`/`fno_modes` must be positive, `fno_vit` added to
the existing parametrized shape and identity-at-init tests (covers
`.step`/`.rollout` shapes and exact identity-at-init regardless of internal
FNO/ViT computation), a direct `SpectralConv1d` shape test, and a linearity
check (`conv(x1+x2) == conv(x1)+conv(x2)` when `modes` covers every
frequency -- catches FFT/mode-multiply/iFFT round-trip bugs that a
shape-only test would miss), plus shape tests with and without mode
truncation. Fast suite: 257/257 passing (was 246 after the ensemble
addition, +11 here).

**CLI**: `train_stage2_patched.py --backbone fno_vit` (also added to
`train_stage1_patched.py --aux-backbone`'s choices, for
`--full-propagator` runs), `--fno-modes` (default `None`), `--fno-n-layers`
(default 2), reusing the existing `--prop-n-tokens`/`--prop-token-d-model`/
`--attn-window`/`--pos-encoding`/`--token-window` flags for the shared
tokenization/ViT half.

**Smoke-tested end-to-end** (CPU, `--profile smoke --backbone fno_vit`):
trained 2 epochs, saved a checkpoint, reloaded it via
`load_propagator_checkpoint` (confirmed `cfg.backbone == "fno_vit"`,
correct `fno_modes`/`fno_n_layers`), ran `.rollout(...)` on the reloaded
model -- correct shape, no errors. Smoke artifacts deleted after
validation.

**Full-scale FNO+ViT training run completed** (tag `fno_vit_markovian`, same
recipe as the ensemble's members -- `mode="markovian"`, `attn_window=4`,
`pos_encoding=linear`, `prop_n_tokens=44`, `token_d_model=64`,
`delta_cap=0.5`, `fno_n_layers=2`, `fno_modes=None`, same slow k-curriculum,
`w_varmatch=0.1`): `best_val_kmax_mse = 0.1863` -- **the best rollout
accuracy of the entire session** by a wide margin (vs 0.3835 for the plain
fine-tuned `vit` propagator, 0.4599 for the ensemble). ~2x fewer epoch-time
seconds than the ensemble at comparable epochs (no 5x member multiplier).
Checkpoint: `artifacts/stage2_prop_patched_full_fno_vit_markovian.pt`.

**Gate 3/4 results** (tag `fno_vit`): DA calibration (`spread/rmse`) =
**0.1714** -- essentially tied with row 8's 0.175 (the best non-ensemble
result so far), far better than the ensemble's 0.1026.
`skill_free_over_da = 2.14` (free-running scored notably *worse* than
DA-corrected here, unlike the ensemble's case -- consistent with a more
"normal"/deterministic model that benefits more from assimilation).

**D3 (Jacobian coupling bandedness) -- directly answering the "does FNO
find real latent structure" question from the same conversation, Section 5's
user discussion**: `0.3504` (still significant vs random permutations,
p=0.0000, but dramatically LOWER than the plain `vit`/`history`
propagator's `0.9888` measured earlier this session). This is genuine
evidence AGAINST the "FNO discovered/reinforced real local latent
structure" hypothesis: if FNO's win came from exploiting real index-
locality, bandedness should have stayed high or increased; instead it
dropped by nearly 3x while still being non-random. The more likely
explanation: FNO's spectral-conv layer is a shift-equivariant (circulant)
linear map REGARDLESS of mode truncation, which is both a strong
regularizer (smaller effective hypothesis class than a dense mixing layer)
and, at `fno_modes=None` (all modes kept, as run here), an efficient GLOBAL
mixing operator -- `mode 0` alone connects every token in one matmul. Both
properties plausibly explain the val_kmax_mse win without requiring the
latent channel ordering to carry genuine physical meaning.

**A real bug caught running Lyapunov on this checkpoint**: crashed with
`ValueError: Not enough Lyapunov directions computed to bracket D_KY
(cumulative sum of all 20 exponents is still non-negative)` inside the
`"two_step"` sub-mode. Root cause: for a `mode="markovian"` propagator (not
`"two_step"`), `_make_two_step_step_fn`'s 2d-dim embedding
`f(z_prev,z_curr)=(z_curr, step_one(z_curr))` is provably redundant -- its
true spectrum is the underlying d-dim map's exponents PLUS exactly `d`
trivial zero exponents from the shift-register block. In floating point,
those "exactly zero" exponents come out as small positive-or-negative
noise; if enough land noise-positive, the Kaplan-Yorke bracketing search
can need more than the `n_directions=min(2d,20)=20` cap even though the
correct, non-redundant `"single_state"` mode (which always uses the full
`n_directions=d`) succeeds. Not a sign of unusually rich chaos -- a
numerical-noise artifact of an intentionally redundant self-consistency
check applied to a model where it isn't the primary diagnostic. **Fixed**:
`run_analysis_suite.py` now wraps each Lyapunov sub-mode
(`single_state`/`two_step`/`history`) in its own `try/except ValueError`,
recording `lyapunov_{mode}_error` instead of losing every other (expensive:
dimension, topology, DA) result in the report to one sub-mode's exception
-- this was a real, silent data-loss risk for every future Gate 3 run, not
specific to FNO+ViT. Re-run launched immediately after the fix; full
`single_state`/`two_step` spectrum to be appended here once complete.

**RESULT -- genuinely rich chaos, matching the reference project's own
benchmark almost exactly**: `single_state` mode: `D_KY = 20.92`,
`n_positive = 11`, `lambda1 = +0.0753`. The reference project's own prior
achieved result, cited throughout this document and CLAUDE_CODE_BRIEF.md as
the target: `D_KY ~= 21.4`, 11 positive Lyapunov exponents. **This is the
first architecture in this entire investigation (this document and every
prior session) to land in that range** -- every other variant tried came
back at `D_KY=0` or the ensemble's fragile `D_KY~1`. `two_step` mode still
hits the known numerical-noise artifact described above (harmless --
`single_state` is the correct primary mode for a `markovian` propagator,
and it succeeded cleanly with a large, unambiguous margin: `lambda1=+0.075`
is not a borderline number the way the ensemble's `+0.0002` was).

This is a landmark result for the whole "why does the propagator collapse"
investigation: an FNO+ViT `markovian` propagator (no history, no ensemble,
`delta_cap=0.5`, `w_varmatch=0.1`, the same slow k-curriculum used
throughout this document) recovers rich chaos that every attention-only
architecture tried this session could not, despite scoring the best
`val_kmax_mse` (0.1863) of the session at the same time -- i.e. it is not a
tradeoff where better short-horizon accuracy came at the cost of collapsing
harder; both improved together here. Whether this generalizes (a different
seed, the `history`-mode FNO+ViT variant now running in Section 7, or a
non-`fno_vit` propagator built on the same FNO-augmented AE) is the
immediate open question -- see Section 7's results once available.

**Robustness check, `--seed 7`** (same checkpoint, different Lyapunov RNG
seed -- different random tangent-vector initialization and warmup
trajectory point): `D_KY = 20.96`, `n_positive = 11`, `lambda1 = 0.0752` --
essentially identical to the `seed=0` result (`20.92`, `11`, `0.0753`).
Not a fluke of one random initialization.

## 6. FNO + ViT encoder/decoder (user-directed, 2026-08-30 follow-up)

Following the D3 bandedness discussion above, the user asked: "implement
the FNO for the encoder and decoder in conjunction with the ViT structure
as well as in the propagator... any other features that we've found help
with performance in past runs can be added at your discretion." Key
framing from that conversation, preserved here: the encoder/decoder's token
axis is genuine physical position on KS's periodic domain (unlike the
propagator's latent-channel-index axis, whose periodicity is an unverified
ordering assumption) -- so applying FNO there is more physically justified
than the propagator case, and this codebase already trains with
cyclic-shift augmentation (`encode_dataset_with_shifts`), consistent with
treating physical translation as a real symmetry.

**Implementation**: `ViTAutoencoderConfig.use_fno` (+ `fno_modes`,
`fno_n_layers`) in `ks_latent/config.py`. `SpectralConv1d`/`FNOLayer`
(the propagator's FNO building blocks) were MOVED from
`ks_latent/models/propagator.py` to `ks_latent/models/autoencoder_vit.py`
(renamed public, `_FNOLayer` -> `FNOLayer`) specifically so
`KSAutoencoderViT` could reuse them without a circular import
(`propagator.py` already imports `ViTBlock`/positional encodings FROM
`autoencoder_vit.py`) -- `propagator.py` now imports `FNOLayer` back from
there instead of defining its own copy. Also added a small shared
`apply_fno_layers(h, layers)` helper (handles the transpose to/from
`FNOLayer`'s channels-first convention; a no-op for an empty
`ModuleList`), used by the encoder, the decoder, AND the propagator's
`_FNOViTDeltaBody`/`_FNOViTHistoryDeltaBody` bodies, replacing three
near-duplicate transpose blocks with one.

`KSAutoencoderViT.encode()`: FNO layers run right after the positional
encoding, BEFORE any CLS-token concatenation (a CLS token has no ring
position, same reasoning that already excludes it from the positional
encoding and gets it special-cased in the attention mask) and before the
ViT attention blocks. `decode()`: symmetric, after `dec_pos`, before
`dec_blocks`. Both are unconditional no-ops (`apply_fno_layers` on an
empty `ModuleList`) when `use_fno=False`.

**Propagator side, extended to `mode="history"`**: the FNO+ViT propagator
backbone (Section 5) was markovian-only; the user's requested recipe needs
`n_history=3`, so added `_FNOViTHistoryDeltaBody` (the `fno_vit` analogue of
`_ViTHistoryDeltaBody`): runs `fno_n_layers` FNO blocks over the spatial
`n_tokens` axis INDEPENDENTLY PER TIME-SLICE (not jointly over space+time --
"FNO preserves spatial relationships" was the framing throughout, not
temporal ones, which the joint spatio-temporal attention already handles),
then the same joint space+time `ViTBlock` attention `_ViTHistoryDeltaBody`
uses. `_validate_propagator_mode_backbone` updated: `mode="history"` now
accepts `backbone in ("vit", "fno_vit")` instead of only `"vit"`.

**Testing**: `tests/unit/test_autoencoder_vit.py` (+8 tests: empty-vs-built
FNO ModuleLists, shape round-trip with/without mode truncation, CLS-pool
compatibility, end-to-end training convergence, config validation).
`tests/unit/test_propagator_modes.py`: replaced the now-stale
"`history`+`fno_vit` raises" test with one confirming it constructs and
produces correct shapes, plus identity-at-init tests for both
`fno_vit`+`markovian` and `fno_vit`+`history`, plus a
`mode="history"`+unsupported-backbone-still-raises test (renamed/kept for
the genuinely-unsupported case, e.g. `"transformer"`). Fast suite:
268/268 passing (was 258 after the ensemble Lyapunov fix).

**Smoke-tested end-to-end, full pipeline in one command**: `--encoder vit
--ae-fno --ae-fno-n-layers 1 --aux-backbone fno_vit --mode history
--n-history 3 --aux-fno-n-layers 1 --full-propagator --prop-delta-cap 0.5`
(Phase 1) followed by `--init-prop-checkpoint` (Phase 2 fine-tune) on CPU,
smoke profile -- confirmed the saved Phase-1 propagator checkpoint reports
`PropagatorConfig` with `backbone="fno_vit"` and loads/fine-tunes correctly
end to end. Smoke artifacts deleted after validation.

## 7. Full combined run: FNO+ViT everywhere, restructured phased training

Launched per the user's explicit recipe (2026-08-30): Phase 1 trains the
full FNO+ViT propagator jointly with an FNO+ViT encoder/decoder, Phase 2
fine-tunes that same propagator for the longer k-curriculum -- combining
every improvement found productive so far into one run.

```
python scripts/train_stage1_patched.py --profile full \
  --encoder vit --ae-fno --ae-fno-n-layers 2 \
  --aux-backbone fno_vit --mode history --n-history 3 --aux-fno-n-layers 2 \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --aux-n-tokens 44 --aux-token-d-model 64 \
  --w-var 0.05 --full-propagator --prop-delta-cap 0.5 \
  --tag fno_everywhere_history3_wvar005
```

Choices made at the user's discretion, all carried over from prior
best-known results in this document: `token_window=16` (best single
Stage-1 change, Section 1.1 row E), `pos_encoding=linear`/`attn_window=4`
(the canonical recipe throughout this document), `w_var=0.05` (Section
1.2's fix for the shortcut-collapse ceiling), `delta_cap=0.5`
("stretch-and-fold" architectural bound, Section 1), `n_history=3` (the
user's explicit request, "current step and the last two steps"),
`aux_n_tokens=44`/`aux_token_d_model=64` (one token per latent coordinate,
Section 1's row 7 recipe). Phase 2 fine-tune (same slow k-curriculum as
rows 7/8: `k_max=8`, `k_warmup_epochs=48`, `k_mid=5`, `k_mid_epochs=40`,
`w_varmatch=0.1`, same noise schedule) to follow once Phase 1 completes.
Result, Gate 3/4, and comparison against rows 7/8/the standalone FNO+ViT
propagator (Section 5) to be appended here.

**Phase 1 completed -- a real reconstruction regression**:
`val_recon_final = 0.0644` -- roughly 160x WORSE than the same recipe
without `--ae-fno` (0.000401, Section 2's Phase-1-restructure run). Adding
FNO to the encoder/decoder substantially hurt reconstruction quality in
this run (loss curve descended much more slowly and plateaued far higher;
see `artifacts/logs/stage1_fno_everywhere_history3_wvar005.log`). Not yet
root-caused -- candidates to check before trying again: LR too high/low for
the added FNO parameters, more epochs needed (the loss was still visibly
decreasing at epoch 99, unlike the non-FNO run which had flattened out by
epoch 60), or a genuine architectural mismatch between global spectral
mixing and the mean-pooling bottleneck. Phase 2 fine-tune launched anyway
(on this worse AE) since the propagator's rollout dynamics are the primary
object of interest here, not reconstruction quality per se -- but this
result should be read with that caveat. Checkpoints:
`artifacts/stage1_ae_patched_full_fno_everywhere_history3_wvar005.pt`,
`artifacts/stage1_prop_full_fno_everywhere_history3_wvar005.pt`.

**Phase 2 fine-tune completed**: `best_val_kmax_mse = 0.2631` -- worse than
the standalone FNO+ViT propagator's `0.1863` (built on the good, non-FNO
AE), consistent with the worse encoder handicapping the propagator's
achievable accuracy. DA calibration: `0.1909` (comparable to, slightly
better than, the standalone FNO+ViT's `0.1714`). Checkpoint:
`artifacts/stage2_prop_patched_full_fno_everywhere_finetune.pt`.

**Lyapunov result: collapsed.** `D_KY=0, n_positive=0, lambda1=-0.033`
(`mode="history"` -- only `single_state`/`two_step` don't apply). Unlike
the standalone FNO+ViT propagator (Section 5, `D_KY~21`), this combination
-- FNO in the encoder AND `mode="history"` (joint spatio-temporal
attention across 3 states x 44 tokens = 132 tokens) in the propagator, on
top of the reconstruction-regressed AE -- did not recover chaos. Confounded
(FNO-encoder reconstruction quality AND markovian-vs-history mode both
differ from Section 5's winning recipe at once, so this doesn't cleanly
isolate which change caused the collapse) but consistent with the emerging
attention-mediated-collapse pattern: this is the only `fno_vit`-backbone
run in the whole document that still uses substantial ViT joint-attention
(132 tokens, `mode="history"`), and it's also the only `fno_vit` run that
collapsed.

## 8. D6: temporal coherence structure (user-directed follow-up, 2026-08-30)

Prompted by a direct methodological objection to Section 5's D3 comparison:
D3's Jacobian-bandedness score measures whether the propagator's learned
map is LOCAL in latent-index space. An FNO spectral-conv layer is a
shift-equivariant (circulant) operator regardless of how many Fourier modes
it keeps -- at `fno_modes=None` (every mode active, as run in Section 5),
its `mode=0` term alone already mixes every token in one matmul. So a
broadband (non-banded) Jacobian is fully consistent with the map still
respecting the latent's ring topology; bandedness specifically detects a
narrow local kernel, not "any operator that respects the ring's symmetry."
The 0.99 -> 0.35 drop measured in Section 5 is therefore NOT conclusive
evidence against real latent structure, only against a *narrow* one -- exactly
the gap the user identified. Framing for the new diagnostic: motivated by
the FNO+ViT propagator's `D_KY~21` result (Section 5, confirmed robust
across seeds), if the rollout is reasonably accurate (`val_kmax_mse=0.1863`,
the best of the session) AND genuinely chaotic, that suggests real,
exploitable latent structure may exist -- worth testing directly in the
data itself, independent of any one propagator's Jacobian.

**Does a suitable technique exist? Yes** -- the same idea used for
"functional connectivity" networks in neuroscience and coherent-structure
detection from unlabelled sensor arrays in fluid mechanics: build a
similarity matrix between channels from **lagged, windowed cross-
correlation** (not simultaneous cross-covariance -- `L_decorr` already
forces that to be approximately diagonal by construction, which is why the
original covariance-based seriation attempt mentioned in `jacobian_
coupling`'s docstring found nothing), then find the best low-dimensional
(ideally near-ring) arrangement of channels via the graph Laplacian's
leading eigenvectors -- the same spectral-seriation machinery `D3` already
uses (`_fiedler_permutation`/`bandedness_p_value`), reused here verbatim on
a new kind of input matrix. Crucially, this measures a genuinely different
statistic than same-time covariance: whether `z_i` and `z_j` move together
ALONG a trajectory through time (possibly with a short lag, reflecting a
finite propagation speed, physically analogous to how nearby points in the
true PDE field are correlated with a short time lag), not whether they are
correlated across independent samples at one instant -- `L_decorr` does not
constrain this at all.

**Implementation** (`ks_latent/analysis/diagnostics.py`, new "D6" section --
not part of the original brief's D1-D5): `local_temporal_coherence(Z,
window, max_lag)` builds the `(d, d)` coherence matrix (median, over many
overlapping windows and trajectories, of the best-|lag| windowed
correlation between each channel pair); `laplacian_eigenmap(A,
n_components)` generalizes D3's `_fiedler_permutation` (which uses only the
single leading eigenvector) to a full low-dimensional embedding coordinate
per channel; `temporal_coherence_diagnostic(...)` wires both together with
D3's existing `bandedness_p_value` for the significance test. Wired into
Gate 4 (`run_diagnostics.py`) as a new D6 report section, using the
*genuine, ordered-in-time* encoded trajectory (unlike D3's independent
`(u_prev, u_curr)` pairs).

**A real calibration bug caught while testing this**: taking the max over
many lags (`2*max_lag+1`) of a correlation estimated from too few samples
per window is a multiple-comparisons problem with a real, systematic
positive bias -- `window=50, max_lag=10` gave ~0.3 "coherence" between
PURE independent noise channels (a false-positive rate for
`bandedness_p_value` well above nominal). Not a bug in `bandedness_p_value`
itself (reused verbatim from D3, already relied on there) -- a property of
this specific estimator. Fixed by using `window >> 2*max_lag+1` (verified
empirically: `window=200, max_lag=5` calibrates correctly); documented as
explicit guidance in `local_temporal_coherence`'s docstring, and
`run_diagnostics.py` uses `window=100, max_lag=5` at full profile
accordingly.

**Testing**: `tests/unit/test_diagnostics_d6.py` (9 tests) -- shape/
symmetry/validation checks, a synthetic AR(1)-based "ring" trajectory
(chosen over a sum-of-sinusoids construction after that first attempt
failed: a purely periodic/deterministic signal shifted by any amount stays
~perfectly correlated once the lag search covers the shift, so it can't
actually test whether NEARBY channels are distinguishable from FAR ones --
an AR(1) process's finite memory/correlation length is what makes that
distinction meaningful) where ring-adjacent channels are confirmed more
coherent than far ones, `temporal_coherence_diagnostic` finds significant
bandedness on that same ring structure and does NOT on pure independent
noise (at the corrected `window`/`max_lag` calibration above), and
`laplacian_eigenmap` shape/orthogonality checks. Fast suite: 277/277
passing (was 268 before this addition).

**Result, run on the FNO+ViT propagator's own AE-encoded dataset** (the
`vit`-only, non-FNO encoder from Section 2/5 -- the one whose latent
achieved `D_KY~21`): **bandedness = 0.2037, p-value = 0.0050 -- SIGNIFICANT.**
Full report: `docs/diagnostics_report_fno_vit_d6.md`.

This directly supports the user's hypothesis. Independent of any
propagator's learned Jacobian, the raw encoded latent trajectory itself
shows a statistically significant emergent low-dimensional organization:
channels that are "nearby" in the Fiedler-discovered ordering genuinely
co-move over time (via lagged, windowed correlation) more than far-apart
ones, well beyond what 1000 random channel permutations of the same matrix
produce. This does not contradict D3's finding that the FNO+ViT
propagator's Jacobian is broadband (0.35, not narrowly local like the
plain `vit` propagator's 0.99) -- it shows the *data itself* carries real
structure that a broadband-but-ring-respecting operator (which is exactly
what an FNO spectral conv is, regardless of mode truncation) could
plausibly be exploiting even without a narrowly-local Jacobian. Together,
D3 and D6 tell a coherent story: real spatial/temporal structure exists in
the latent (D6), and the FNO+ViT propagator's chaos-recovering win likely
comes from a GLOBAL operator that still respects that structure, not from
learning a spatially LOCAL map the way the plain ViT/attention-window
propagators do.

**Important clarification, caught immediately on reflection**: D6 depends
only on `ae.encode` (a genuine, ordered-in-time trajectory through the
FIXED, frozen encoder) -- it does not touch the propagator checkpoint at
all. This means the 0.204/p=0.005 result above is not specific to the
FNO+ViT propagator; it is a property of the SHARED AE
(`stage1_ae_patched_full_history3_fullprop_wvar005_tw16.pt`) that BOTH the
plain `vit`/history propagator (row 8, `D_KY=0`) and the FNO+ViT
propagator (Section 5, `D_KY~21`) were trained on identically. **The
significant temporal-coherence structure was already present in the latent
before either propagator was trained on it** -- so its mere existence is
NOT sufficient for a propagator to recover chaos (row 8's propagator had
access to the exact same structured latent and still collapsed to `D_KY=0`).
The FNO+ViT architecture specifically was needed to exploit it. This
sharpens the user's original hypothesis usefully: it's not "does structure
exist" (yes, robustly) but "which propagator architectures can exploit
it" -- consistent with the D3 finding that FNO+ViT's Jacobian is a
broadband-but-ring-respecting operator, unlike the attention-only
propagators' narrower, more local (but ultimately fixed-point-collapsing)
maps.

**Not yet done**: running D6 against the FNO-augmented encoder/decoder from
Section 7 (once available) to see whether adding FNO to the encoder itself
changes (strengthens, weakens, or reshapes) the structure D6 detects,
versus the plain ViT encoder used here.

**Fixed a report gap and visualized the discovered ordering directly**
(user follow-up, "does D6 find this ordering, is there an easy way to find
it"): yes -- `TemporalCoherenceResult.permutation` (the Fiedler-vector
ordering) and `.embedding` (a 2D spectral-graph coordinate per channel,
`laplacian_eigenmap`) were already computed but not written to the
markdown report; fixed `run_diagnostics.py` to print the permutation and
save `artifacts/diagnostics_arrays{tag}.npz` (raw `C`, both permutations,
the embedding) for further use/plotting.

**Visualized the embedding and the reordered coherence matrix directly --
the honest picture is more modest than "a clean ring"**: the 2D embedding
is dominated by 3 outlier channels (indices 5, 10, 28), each far from the
rest in its own direction (a "hub-and-spoke" pattern), while the other ~41
channels sit in one tight, poorly-differentiated cluster near the origin --
not the smooth loop a clean 1D ring structure would produce (a true ring
graph's 2nd/3rd Laplacian eigenvectors trace a circle; this doesn't).
The Fiedler-reordered coherence heatmap is visually almost
indistinguishable from the unreordered one -- no obvious diagonal band
emerges to the eye. This is consistent with the statistics: `bandedness
observed = 0.204` is a real, significant (`p=0.005`) effect versus random
permutations, but it is a WEAK effect size relative to D3's own numbers
throughout this document (`0.35`-`0.99`) -- real, not a strong or visually
obvious structure. Fair summary: there is significant, exploitable-in-
principle structure beyond random chance, concentrated more in a few
distinctive channels than in a clean global spatial organization; it is
not the dramatic "latent has a hidden physical geometry" picture the
D_KY~21 result might suggest on its own.

**The real mechanism, resolved (user-directed follow-up)**: the user
correctly objected that D6's weak/hub-dominated structure is hard to
square with FNO+ViT's dramatic `D_KY~21` win, and pointed out something
this document had not accounted for: **plain ViT propagators with FULL
(unrestricted) attention were already tried in an earlier session and also
collapsed** (`D_KY=0`) -- ruling out "FNO just gives a wider receptive
field" as the explanation, since global self-attention already has
unlimited reach and still failed. Combined with the fact that the FNO+ViT
propagator in Section 5 *also* used `attn_window=4` on its ViT half (the
only architectural change from the plain, collapsing `vit`/`attn_window=4`
propagator was ADDING the FNO layers in front of that same locally-windowed
attention), this gives three data points: local attention alone
(collapsed), global attention alone (collapsed, per the user), FNO (global)
+ local attention together (chaotic). Reach is not the variable that
matters.

**Mechanistic hypothesis, confirmed directly on the trained weights, no
new training run needed**: self-attention's softmax normalization makes
every output a convex combination of value vectors (weights sum to 1) --
structurally biased toward averaging/contraction regardless of window
width. FNO's spectral-conv weights have no such constraint; a per-mode
complex `(64, 64)` matrix can freely amplify. Checked directly: extracted
`stage2_prop_patched_full_fno_vit_markovian.pt`'s two `FNOLayer`s' learned
per-mode weight matrices and took the largest singular value at each of
the 23 rfft modes --

| | modes with max singular value > 1 | max amplification |
|---|---|---|
| FNO layer 0 | 22 / 23 | **2.50x** (highest at the highest frequencies) |
| FNO layer 1 | 19 / 23 | 1.42x |
| layer 0's parallel skip path (`Conv1d`, kernel 1) | -- | 1.37x |

Nearly every frequency mode learned genuine signal AMPLIFICATION, not
mixing/averaging -- exactly the local "stretch" a chaotic map needs to
produce positive Lyapunov exponents, and exactly what attention's convexity
constraint forbids by construction. **Revised conclusion**: FNO+ViT's win
is not primarily about exploiting real spatial/index structure in the
latent (D6 found only a weak signal, and the hypothesis-motivating premise
-- that discovering structure would explain the win -- doesn't actually
need to be true for this explanation to hold). It is that FNO's
architecture is *unconstrained* to learn amplification, while every
attention-based propagator tried this session (local or global window) is
structurally biased toward the contraction that causes the fixed-point
collapse, independent of receptive field size. This reframes the "Solution
2" motivation after the fact: the win came from architectural freedom to
stretch, not from "Fourier space preserving spatial relationships" as
originally hypothesized -- a genuinely different (and more mechanistically
satisfying) explanation than the one motivating the experiment.

**This explanation did not survive its own next test.** The user pushed
back again, correctly: a plain residual MLP (`backbone="mlp"`) has no
softmax anywhere either -- every weight matrix is exactly as free to
amplify as FNO's -- and an earlier (not fully documented in this file)
observation was that MLP backbones also collapsed. Rather than rely on
that undocumented prior result, re-ran it directly: same AE
(`stage1_history3_fullprop_wvar005_tw16`), same Phase-2 recipe as the
winning FNO+ViT run (`delta_cap=0.5`, `w_varmatch=0.1`, identical slow
k-curriculum and noise schedule), swapping only `--backbone mlp`, tag
`mlp_markovian_control`. Trained on CPU (per this codebase's own documented
device policy: the small residual-MLP propagator is launch-overhead
dominated and doesn't benefit from MPS) in ~10 minutes.

**Result: plain MLP matches FNO+ViT's chaos almost exactly, and beats it
on every other metric.**

| | `val_kmax_mse` | `D_KY` | `n_positive` | `lambda1` | DA calibration |
|---|---|---|---|---|---|
| Plain ViT (`attn_window=4`) | 0.3835 | 0 | 0 | -0.046 | 0.175 |
| FNO+ViT | 0.1863 | 20.92 | 11 | 0.0753 | 0.1714 |
| **Plain MLP (control)** | **0.0280** | **20.92** | **12** | **0.0976** | **0.3768** |

(`D3` bandedness for the MLP control: `0.3514`, essentially the same as
FNO+ViT's `0.35` -- both are architectures with no locality constraint at
all, unlike the `attn_window`-restricted `vit` propagators, consistent
with bandedness mostly reflecting architectural connectivity rather than
discovered structure, as discussed above. `D6` is unchanged, `0.2037`/
`p=0.005`, since it depends only on the shared AE.)

**Revised conclusion, superseding the amplification/convexity story
above**: MLP has no softmax and recovers chaos even more cleanly than FNO
does, so "freedom to amplify vs. softmax-constrained" cannot be the
deciding factor either -- MLP and FNO both have that freedom, and only one
of them needed it demonstrated via measured singular values to make the
case. The pattern that has actually held up across every architecture
tried this session: **every propagator using softmax self-attention for
token-mixing (`vit` backbone, local OR global window) collapsed to
`D_KY=0`; every one that didn't (`mlp`, and `fno_vit`, where the FNO layer
runs before its own internal attention) recovered chaos** -- and the
simplest attention-free option won outright, on every metric, including
against the architecture that was specifically designed to fix the
attention-based collapse. This suggests attention itself (regardless of
receptive field) was actively counterproductive for this task, not merely
insufficiently expressive -- and that a substantial share of this
session's architecture exploration (`transformer`/`vit`/`fno_vit`
backbones, and the ensemble, which used a `vit` member architecture) may
have been fighting a problem that the brief's original, much simpler
`mlp` backbone never had. Full mechanistic "why does attention specifically
cause this" is still open -- see the open questions below.

---

## 9. Session summary: every propagator architecture tried, and the MLP control

All results in this document, on the same restructured-AE/`delta_cap=0.5`/
slow-k-curriculum recipe (Section 1's row 7/8 schedule) unless noted, sorted
best-to-worst by `D_KY`:

| Architecture | Attention? | `val_kmax_mse` | `D_KY` | `n_positive` | `lambda1` | DA calib. |
|---|---|---|---|---|---|---|
| **MLP (control, Section 9)** | no | **0.0280** | **20.92** | **12** | **0.0976** | **0.3768** |
| FNO+ViT (Section 5) | yes (after FNO) | 0.1863 | 20.92 | 11 | 0.0753 | 0.1714 |
| Ensemble, 5x `vit`/markovian members (Section 4) | yes | 0.4599 | ~1.01 (fragile, mode-dependent) | 1 | ~0.0002 | 0.1026 |
| FNO+ViT on FNO-augmented AE (Section 7) | yes (after FNO) | 0.2631 | pending | pending | pending | 0.1909 |
| Restructured full-propagator, `vit`/history (Section 2) | yes | 0.3835 | 0 | 0 | -0.046 | 0.175 |
| Original `vit`/history, `k_max=8` (Section 1 row 7) | yes | 0.4099 | 0 | 0 | -0.057 | 0.112 |
| Original `vit`, two-segment k-curriculum (row 2) | yes | 0.8132 | 0 | 0 | -0.106 | 0.066 |
| Plain `vit`, global attention (no window) | yes | -- | 0 | 0 | -- | -- |

(Last row: an earlier session's result, not re-run/re-verified with the
current best AE/recipe -- included because it's what motivated ruling out
"receptive field size" as the deciding variable; see Section 5's
discussion. Everything else in this table used the identical AE,
`stage1_ae_patched_full_history3_fullprop_wvar005_tw16`, except the two
FNO-augmented-AE rows.)

**The clean pattern**: every architecture using softmax self-attention for
token-mixing (`vit` backbone, `transformer` backbone was never actually
re-tested on this recipe but shares the same self-attention mechanism)
collapsed regardless of window size; every architecture that avoids it
(`mlp`, and `fno_vit`, whose FNO layer runs before its internal attention)
recovered rich chaos. Plain MLP -- no attention at all -- did best on every
metric, including against the architecture (FNO+ViT) specifically built to
work around the collapse.

**Open questions, not yet resolved**:
- *Why* does self-attention specifically cause this collapse, mechanistically?
  The "softmax = convex combination = contraction" hypothesis (Section 5)
  doesn't survive MLP's result (MLP has no such constraint either, yet also
  works) -- it can't be the whole story. Untested alternative: model
  capacity/parameter-count differences between the tokenized attention
  architectures (44 tokens, 64-dim, multi-head) and the plain MLP
  (`hidden=128`, 3 residual blocks) could matter more than the attention
  mechanism per se -- a smaller/simpler model might be less able to
  represent the globally-MSE-optimal collapsed solution as easily, or the
  optimizer might reach a different basin purely from different effective
  learning dynamics. Not yet isolated.
- Does the `history` mode (`n_history=3`, the user's originally-requested
  recipe) recover chaos with an MLP-family body, or does the same collapse
  pattern hold once history/joint-attention is reintroduced? (Plain `mlp`
  backbone only supports `mode="markovian"`/`"two_step"`, not `"history"`
  in the current codebase -- would need `_MLPHistoryDeltaBody`, not yet
  implemented, to test directly.)
- Given this, is further ensemble/FNO-in-encoder work still the right
  priority, versus a systematic sweep of the *inductive bias that actually
  matters* (attention vs. none) holding capacity/parameter-count fixed?

**`mode="history"` + `backbone="mlp"`, implemented and launched
(user-directed follow-up)**: the user asked directly whether the same
MLP win holds for their originally-requested `n_history=3` recipe --
answering the first open question above required implementing it, since
the codebase only supported `backbone in ("vit", "fno_vit")` for
`mode="history"`. Added: generalizes `"two_step"` (fixed `n_history=2`,
flattened `2*d`-dim input to `_MLPDeltaBody`) to an arbitrary history
length -- `LatentPropagator.step_history` now flattens `(B, n_history, d)`
to `(B, n_history*d)` before calling `_MLPDeltaBody` when
`backbone="mlp"`, reusing the exact same class (no new body needed,
unlike `_ViTHistoryDeltaBody`/`_FNOViTHistoryDeltaBody`, which need
their own classes because they tokenize the multi-state input rather than
flattening it). `_validate_propagator_mode_backbone` updated to accept
`mlp` for `mode="history"`. 4 new tests (shape/identity-at-init for
`history`+`mlp`, mirroring the existing `fno_vit`+`history` tests). Fast
suite: 279/279 passing (was 277). Smoke-tested the full pipeline (Phase 1
full-propagator with `--aux-backbone mlp --mode history` -> Phase 2
`--init-prop-checkpoint` fine-tune) end to end on CPU before launching.

**Launched**: exactly the `stage1_history3_fullprop_wvar005_tw16` recipe
(same AE settings: `--encoder vit --pos-encoding linear --attn-window 4
--token-window 16 --w-var 0.05`), with `--aux-backbone mlp` in place of
`--aux-backbone vit` (dropping the now-inapplicable `--aux-n-tokens`/
`--aux-token-d-model`, which only apply to tokenized backbones), tag
`history3_fullprop_wvar005_tw16_mlpaux`.

**Phase 1 completed**: `val_recon_final = 0.000482` -- essentially
identical to the `vit`-aux version's `0.000401` (as expected: the joint
`L_pred` term's backbone choice barely touches AE reconstruction quality).
**Phase 2 fine-tune was launched, then killed mid-run** (~28 epochs in) at
the user's explicit request, in favor of prioritizing the `markovian`
(not `history`) variant and the new `local_mlp` backbone below -- no
result to report for this specific `history`+`mlp` combination yet;
picking it back up is a candidate for later, not abandoned.

## 10. `backbone="local_mlp"`: isolating receptive field from softmax (user-directed, 2026-08-30)

The user raised a sharp objection to Section 9's "attention specifically
causes collapse" framing: plain MLP has no softmax either, and recovers
chaos even better than FNO+ViT, so "freedom from softmax constraints"
alone can't be the deciding factor (Section 9 already concedes this). The
user then proposed the natural next experiment directly: a propagator with
a LOCAL receptive field (bounded connectivity, like a banded/masked
weight matrix) but NO attention/softmax at all -- "this might force the
latent states to have more structure." This fills in the one missing cell
of a 2x2 grid this document had otherwise fully populated:

| | softmax (attention) | no softmax |
|---|---|---|
| **local receptive field** | `vit`, `attn_window=4` -- collapsed | **`local_mlp` (this section)** |
| **global receptive field** | `vit`, full attention -- collapsed | `mlp`, `fno_vit` -- both chaotic |

If `local_mlp` ALSO recovers chaos, softmax specifically is implicated
(receptive field width doesn't matter). If it collapses like the
attention-based propagators, receptive field width is the real variable,
not softmax -- and `mlp`/`fno_vit`'s wins would be better attributed to
having *global* reach than to avoiding softmax per se.

**Implementation** (`ks_latent/models/propagator.py`): `_LocalMixerBlock` --
one block of a LOCAL, UNCONSTRAINED linear token-mixer (circular
`nn.Conv1d`, kernel width `2*window+1`, no normalization of any kind)
followed by the same per-token feedforward MLP `ViTBlock` uses -- pre-norm
residual connections around each, mirroring `ViTBlock`'s structure exactly
so the token-mixing mechanism is the only thing that changes.
`_LocalMLPDeltaBody` stacks `token_n_layers` of these, tokenizing/
detokenizing exactly like `_ViTDeltaBody`. No positional encoding: a
circular convolution is already translation-equivariant by construction
(same property `fno_vit` has, deliberately, not an oversight). Reuses the
existing `attn_window` config field as the conv kernel radius (now
REQUIRED, not optional, for this backbone -- there is no "global
`local_mlp`": that degenerates to a dense/circulant mixer already covered
by `mlp`/`fno_vit`). `markovian` mode only for now (same original scope
`transformer`/`vit` had before `history` was added). Added to
`_VALID_PROPAGATOR_BACKBONES` and both config classes' validation
(`d_latent % n_tokens` divisibility, `attn_window is not None` requirement).

**Testing**: 6 new tests in `tests/unit/test_propagator_modes.py` --
config validation (`attn_window` required, divisibility, `two_step`/
`history` both still correctly rejected), shape + identity-at-init, and a
direct receptive-field-boundedness check (perturbing latent index 0 and
confirming, via autograd, that only tokens within the conv's kernel radius
show a nonzero gradient -- verifies the mixer is genuinely local, not just
initialized-looking-local). Fast suite: 285/285 passing (was 279).
Smoke-tested the full pipeline (Phase 1 full-propagator with
`--aux-backbone local_mlp --mode markovian --attn-window 1` -> Phase 2
`--init-prop-checkpoint` fine-tune) end to end on CPU before launching.

**Launched**: `--encoder vit --aux-backbone local_mlp --mode markovian
--pos-encoding linear --attn-window 4 --token-window 16 --w-var 0.05
--full-propagator --prop-delta-cap 0.5`, tag
`fullprop_wvar005_tw16_localmlp_markovian` -- the same AE recipe as
Section 2/5/9's winning runs, `attn_window=4` matching the collapsing
`vit` propagator's own window size (so the ONLY variable that changes
versus that collapsed baseline is softmax vs. plain linear token-mixing),
`mode="markovian"` per the user's stated preference ("a markovian MLP
makes the most sense"). Result to be appended here once training and
Gate 3/4 complete.

**A real bug caught before it invalidated the experiment**: the first
launch omitted `--aux-n-tokens`/`--aux-token-d-model`, silently falling
back to `train_stage1_patched.py`'s DEFAULTS (`n_tokens=4`,
`token_d_model=32`) instead of the `n_tokens=44` (one token per latent
coordinate) convention every other tokenized-backbone experiment in this
document uses. With only 4 tokens on a ring, the maximum possible circular
distance is `floor(4/2)=2` -- so `attn_window=4` covered every pair of
tokens, making the "local" conv mixer secretly GLOBAL (a `Conv1d` with
`kernel_size=2*4+1=9` on only 4 positions wraps around more than twice),
completely defeating the point of the experiment (testing locality
specifically). Caught by inspecting the saved checkpoint's actual config
(`n_tokens=4, attn_window=4, chunk_size=11`) before Phase 2 got far into
training -- killed both the Phase 1 checkpoint's downstream Phase 2 run
and relaunched Phase 1 from scratch with `--aux-n-tokens 44
--aux-token-d-model 64` explicitly set, matching every other experiment's
convention so `attn_window=4` is genuinely restrictive (a token can only
reach 4 of its 44 neighbours, not all of them).

**Phase 1 (corrected) completed**: `val_recon_final = 0.000530`, consistent
with every other run on this AE recipe.

**A robustness check requested alongside Phase 2**: the user asked
whether the plain-MLP-recovers-chaos result (Section 9) was specific to
the one AE it happened to be tested on, or holds on a different (but
same-recipe) AE too. Launched two Phase 2 runs from this new AE in
parallel: (a) `local_mlp`'s own fine-tune via `--init-prop-checkpoint`
(MPS), and (b) a fresh, from-scratch plain-MLP propagator -- NOT loaded
from any checkpoint, exactly the `stage2_mlp_markovian_control.log`
training scheme (`backbone=mlp`, `mode=markovian`, same `delta_cap`/
k-curriculum/noise schedule) -- pointed at this new AE instead (CPU).

**Results**:

| | `val_kmax_mse` |
|---|---|
| `local_mlp` fine-tune | **0.5204** -- worse than even the collapsed plain `vit` propagator (0.3835) |
| Plain MLP, fresh, on this new AE | **0.0250** -- matches the original control's `0.0280` almost exactly |

The plain-MLP result is confirmed robust: it is not a fluke of one
particular AE checkpoint's random initialization -- an independently
Phase-1-trained AE (different joint aux propagator, `local_mlp` instead of
`vit`/history, during Phase 1) gives essentially the same excellent
result. `local_mlp`, by contrast, performed poorly on rollout accuracy --
worse than every attention-based propagator tried this session. Per the
user's instruction, Gate 3/4 (`D_KY`/chaos) was run ONLY on `local_mlp`
(the plain-MLP-on-this-AE run was not expected to add information beyond
confirming the robustness check above, and was skipped to save compute).
`local_mlp`'s Gate 3/4 result is below.

**RESULT: collapsed.** `D_KY=0, n_positive=0`. `single_state` mode:
`lambda1=-0.0747`; `two_step` mode (independent cross-check): `lambda1=
-0.0748` -- the two agree almost exactly, a strongly contracting,
unambiguous fixed-point collapse. DA calibration: `0.0947` (one of the
worst of the session). `val_kmax_mse=0.5204` (worse than every
attention-based propagator tried, including the ones that also
collapsed). D3 bandedness: `0.8803` (high, close to the collapsed plain
`vit` propagator's `0.99` -- and the discovered Fiedler ordering nearly
matches the ORIGINAL consecutive latent index order, confirming the
architecture is doing exactly what it was designed to: genuine local
coupling in latent-index space, not a scrambled hidden one). D6 (data-only,
independent of propagator): `0.2038`, consistent with every other run on
this AE recipe.

**This settles the question the experiment was designed to answer.**
Combined with every other result in this document, the full 2x2 grid is
now:

| | softmax (attention) | no softmax |
|---|---|---|
| **local receptive field** | collapsed (`vit`, `attn_window=4`) | **collapsed (`local_mlp`, this section)** |
| **global receptive field** | collapsed (`vit`, full attention) | chaotic (`mlp`, `fno_vit`) |

**Receptive field width is the deciding variable, not softmax
normalization.** Neither "global" nor "no softmax" is sufficient on its
own -- global self-attention still collapsed, and a local-but-unconstrained
linear mixer (this section) also collapsed. Only the combination (global
AND unconstrained -- plain `mlp`'s dense layers, or `fno_vit`'s spectral
convolution) recovers chaos. The user's original intuition motivating this
experiment -- that forcing a local receptive field "might force the latent
states to have more structure" -- turned out to be exactly backwards for
this system: locality (regardless of the mixing mechanism) reproduces the
collapse; only genuinely global, freely-parameterized mixing escapes it.
This also retroactively explains why `fno_vit`'s local ViT-attention half
(still using `attn_window=4`) didn't prevent it from recovering chaos: the
FNO layer running BEFORE that attention already gave the map global reach,
and evidently that's what mattered, not what the (still-local,
still-softmax) attention layers downstream did with it.

**Effective receptive field, verified directly on the trained weights**
(not just the config): `attn_window=4` (conv kernel radius) x
`token_n_layers=2` (stacked local-mixer blocks) gives a theoretical max
reach of `4*2=8` tokens in either direction, exactly like receptive-field
growth in any stacked CNN. Confirmed empirically via autograd on the
actual trained checkpoint: perturbing latent token 0 produces a nonzero
gradient at every token within circular distance 8 (17 of 44 tokens,
~39% of the ring) and EXACTLY zero beyond it -- the architecture is
genuinely, verifiably local, not just nominally so.

## 11. `masked_mlp`: literal local-to-full weight-space warm start (user-directed, 2026-08-30)

Despite Section 10's definitive result (local receptive field collapses
regardless of softmax), the user proposed a distinct, constructive idea
rather than abandoning locality entirely: use a LOCAL propagator to shape
the latent during Phase 1 (hypothesis: a local auxiliary propagator's
`L_pred` gradient might induce a more spatially-organized encoder, even
though a local propagator alone can't do the final job), then hand off to
a FULL (dense, global) propagator for the real rollout task -- warm-started
directly from the local model's weights, since "this should just have
zeros off diagonal."

**Why `local_mlp` (Section 10) can't do this literally**: it tokenizes `z`
and uses a `Conv1d` token-mixer over a `(n_tokens, token_d_model)`
representation -- a fundamentally different parameterization from
`_MLPDeltaBody`'s flat `d_latent -> hidden -> d_latent` residual MLP, with
no direct weight-space correspondence between the two. A literal warm
start needs local and full to be the SAME architecture, differing only in
which weight entries are allowed to be nonzero.

**Implementation** (`ks_latent/models/propagator.py`): `backbone=
"masked_mlp"` -- a residual MLP, dimension-preserving throughout (every
layer is `d_latent -> d_latent`, ignoring `cfg.hidden`, so "position i"
keeps a consistent latent-index meaning through the whole network, which
masking requires), with every `Linear` replaced by `MaskedLinear`: a
`Linear(dim, dim)` with a fixed circular-band mask applied to its weight
on every forward pass, zeroed once at construction. `attn_window` (reused
again) sets the mask radius; `None` gives a fully dense, unmasked layer --
the "full mlp" endpoint.

**Why this makes the warm start literal, not approximate**: a masked
entry's gradient is `d(loss)/d(weight[i,j]) = d(loss)/d((weight*mask)[i,j])
* mask[i,j]`, which is EXACTLY zero whenever `mask[i,j]=0`. So an off-band
entry, zeroed at construction, receives exactly zero gradient for as long
as the module is trained with a finite window -- it is still exactly zero
after training, not "small" or "arbitrary leftover noise from
initialization." `masked_mlp_warm_start(local_prop, full_cfg, perturb_std,
seed)` builds a fresh dense (`attn_window=None`) module with IDENTICAL
parameter shapes, copies every weight across via `load_state_dict` (the
mask itself is a non-persistent buffer, excluded from `state_dict()`, so
this is a plain, exact weight copy), then adds `Gaussian(0, perturb_std)`
noise ONLY to the entries the local model's mask had excluded -- unlocking
them from their previously-frozen-at-zero state for further training,
leaving the on-band (already-trained) entries untouched.

**A real bug caught by the receptive-field test before it invalidated
anything**: the first implementation used `nn.LayerNorm(d_latent)` (pre-
norm, matching `ResidualMLPBlock`'s convention) inside each residual
block. `LayerNorm` normalizes across its ENTIRE feature axis -- here, the
physical latent-index axis -- computing one shared mean/variance over ALL
positions. That silently reintroduces a fully global coupling between
every position, completely independent of the mask, defeating the entire
purpose of this backbone. Caught immediately by the receptive-field-
boundedness test (mirroring Section 10's for `local_mlp`): the observed
nonzero-gradient set covered every position, not just the expected
`4*window*n_blocks` band. Fixed by removing `LayerNorm` from this backbone
entirely (documented in `_MaskedMLPResidualBlock`'s docstring as a
deliberate omission, not an oversight) -- there is no position-local
normalization available here (each position carries a single scalar, not
a feature vector, so there is nothing to normalize "per position" against).

**Testing**: 8 new tests in `tests/unit/test_propagator_modes.py` --
`MaskedLinear` zeros its off-band weights at construction and they receive
exactly zero gradient after a backward pass; `window=None` is fully dense;
shape/identity-at-init; `attn_window=None` is explicitly ALLOWED (unlike
`local_mlp`); a receptive-field-boundedness check (this is the test that
caught the `LayerNorm` bug -- run at `d_latent=20` specifically, since the
`local_mlp` test's `d_latent=8` would have made the bug invisible: reach
`4*1*1=4` already covers an 8-ring entirely by coincidence, silently
passing even with the bug present); and `masked_mlp_warm_start` copies
on-band weights exactly and perturbs only the off-band (previously
provably-zero) ones, plus its config-mismatch guards. Fast suite: 292/292
passing (was 285). Smoke-tested the full pipeline (Phase 1
`--full-propagator` with `masked_mlp`/`attn_window=1` -> Phase 2
`--masked-mlp-warm-start`, confirmed the output checkpoint's config shows
`attn_window=None`) end to end on CPU before launching.

**CLI**: `train_stage1_patched.py --aux-backbone masked_mlp` (with
`--attn-window` set, `--full-propagator` to save it);
`train_stage2_patched.py --masked-mlp-warm-start <local-checkpoint-path>
--masked-mlp-perturb-std <default 0.01>` (mutually exclusive with
`--init-prop-checkpoint`, which preserves architecture rather than
deliberately changing it).

**Launched**: Phase 1, `--aux-backbone masked_mlp --mode markovian
--attn-window 4` (matching this document's established window size),
`--epochs 150` (the user's specified duration for the local/structure-
building phase), same AE recipe as every other experiment
(`--encoder vit --pos-encoding linear --token-window 16 --w-var 0.05
--full-propagator --prop-delta-cap 0.5`), tag `maskedmlp_local_150ep`.
Phase 2 (`--masked-mlp-warm-start`, then the established slow k-curriculum
fine-tuning recipe) to follow once Phase 1 completes. Result to be
appended here.

**Phase 1 completed**: `val_recon_final = 0.000244` -- the best
reconstruction of the ENTIRE session (the extra 50 epochs, 150 vs the
usual 100, clearly helped; the joint `masked_mlp` `L_pred` term did not
hurt reconstruction at all).

**Phase 2 (warm-start) completed**: `val_kmax_mse = 0.2753` -- much worse
than the from-scratch plain-MLP control's `0.0280` (~10x), suggesting the
warm-start's small perturbation (`std=0.01`) may not "unlock" the
off-band weights as effectively as full random initialization would; the
optimizer may be stuck exploring only a narrow neighborhood around the
local model's solution rather than freely finding a better global one. DA
calibration: `0.1201` (poor, consistent with the accuracy shortfall). D3
bandedness: only marginally significant now (`p=0.038`, much weaker than
every other run). D6 (data-only): **NOT significant** (`p=0.217`) -- the
first AE in this document NOT to show the temporal-coherence structure
Section 8 found on two other same-recipe AEs; this AE's `masked_mlp`
joint training may have organized the latent differently.

**RESULT -- chaos recovered anyway, despite the accuracy shortfall**:
`single_state` mode: `D_KY=20.51, n_positive=8, lambda1=+0.0686`. This is
the THIRD independent architecture in this document to land in the
reference project's `D_KY~21` range (after the from-scratch plain `mlp`
and `fno_vit`), reached via a completely different training path (local
`masked_mlp` for 150 epochs, then warm-started to a dense `masked_mlp`,
rather than training a dense architecture from scratch). This is further
evidence for Section 9's pattern -- {global receptive field, unconstrained
weights} -- being what matters for recovering chaos, independent of HOW a
model with that property was arrived at. It also decouples "how chaotic"
from "how accurate": this model is far less accurate (`val_kmax_mse=
0.275`) and far worse calibrated (`0.120`) than the from-scratch MLP
control, yet reaches comparably rich chaos -- rollout accuracy at a fixed
horizon and the qualitative dynamical character of the attractor appear to
be fairly independent axes.

**Open question this raises**: does the LOCAL pretraining phase actually
matter here, or would 150+68 epochs of plain (never-masked) `mlp` training
from scratch on this same AE do just as well or better? The `perturb_std`
choice (0.01) is also untested for sensitivity -- a larger perturbation
(closer to full reinitialization of the off-band weights) might close the
accuracy gap with the from-scratch control while keeping the local
phase's hypothesized benefit.

---

## 12. Isolating `history`: `mlp` + `mode="history"` (user-directed, 2026-08-30)

Both of the two best results in this document so far (`fno_vit`, Section 5,
and the from-scratch plain `mlp`, Section 9) were built on the SAME
underlying AE -- `stage1_ae_patched_full_history3_fullprop_wvar005_tw16.pt`
-- while every OTHER same-recipe AE (the `local_mlp`/`masked_mlp`
experiments' own AEs, Sections 10/11) did not reproduce this. The user
asked the natural question: is there something special about that
specific AE, and is it the `mode="history"` (joint `n_history=3` `L_pred`
during Phase 1) that explains it, as opposed to the `vit` backbone used
there? This is directly testable: an `mlp`+`history` Phase 1 run (same AE
recipe, `--aux-backbone mlp --mode history --n-history 3`) had ALREADY
been trained earlier this session (before being redirected to
`local_mlp`/markovian) and its checkpoint was still on disk -- its Phase 2
fine-tune had been killed mid-run, so it just needed to be resumed.

**Phase 2 fine-tune (resumed, completed)**: `val_kmax_mse = 0.02577` --
actually BETTER than the from-scratch plain-`mlp`-markovian control's
`0.0280`.

**Gate 3/4 results -- the best of the entire session**:

| Metric | This run (`mlp`+`history`) | Best prior (`mlp`, markovian) |
|---|---|---|
| `val_kmax_mse` | **0.0258** | 0.0280 |
| `D_KY` | **22.14** | 20.92 |
| `n_positive` | **13** | 12 |
| `lambda1` | **+0.111** | +0.098 |
| DA calibration | 0.325 | **0.377** |
| D6 p-value | **0.0000** | (not run on this AE) |

`D_KY=22.14` EXCEEDS the reference project's own benchmark (`D_KY~21.4`,
cited throughout this document as the target) -- the highest `D_KY`,
`n_positive`, and `lambda1` of the entire session. DA calibration is
slightly below the markovian `mlp` control's own best, but still the
second-best of the whole document. D3 bandedness: `p=0.065` (marginal, not
quite significant at the conventional threshold).

**A real numerical limitation caught and resolved**: the initial Gate 3
run hit the SAME "not enough Lyapunov directions" bracketing failure
described in Section 5/7 -- but for `mode="history"` specifically, there
is no `single_state` fallback the way markovian propagators have (history
mode's own state space, `n_history*d_latent = 132`-dimensional, is the
only option), so `run_analysis_suite.py`'s built-in cap
(`n_directions=min(n_hist*d, 20)=20`) returned NO usable result at all,
just a caught, gracefully-reported error (Section 5's resilience fix
prevented a crash but not the missing result). Given how strong the other
metrics were, re-ran the Lyapunov computation directly (not through the
Gate 3 script, which hardcodes the 20-direction cap for `history` mode)
with the FULL `n_directions=132` -- this resolved cleanly to the result
above. **Follow-up**: `run_analysis_suite.py`'s hardcoded `min(n_hist*d,
20)` cap for history mode should probably be raised (or made
configurable) given this -- a real result was sitting one CLI-level
limitation away from being found automatically.

**Conclusion**: this strongly supports the user's hypothesis. The
`vit` backbone was not what made
`stage1_ae_patched_full_history3_fullprop_wvar005_tw16.pt` special --
`mode="history"` (joint multi-step `L_pred` during Phase 1) reproduces
(and slightly exceeds) the same magic with a completely different
propagator FAMILY (`mlp`, not `vit`) for the auxiliary/full propagator.
Combined with Section 9's finding that softmax attention (not history)
was what caused collapse in `vit`-backbone propagators, the emerging
picture is: `history`-mode joint training during Phase 1 shapes the
latent in a way that benefits ANY sufficiently unconstrained/global
downstream propagator (`mlp`, `fno_vit`), independent of that
propagator's own architecture family -- while `local_mlp`'s and
`masked_mlp`'s OWN Phase-1 joint training (markovian, not history) did NOT
reproduce it, suggesting the history-mode joint training itself,
not just "any AE from this general recipe," is the active ingredient.

## 13. Pushing the best recipe further: more epochs in both phases (user-directed, 2026-08-30)

Given `mlp` markovian and `mlp`+`history` were now the two best results,
both built on `stage1_ae_patched_full_history3_fullprop_wvar005_tw16`, the
user asked to re-run that exact sequence with more epochs in both phases,
to see if the "best recipe" scales further. Doubled: Phase 1 `--epochs
100 -> 200` (tag `history3_fullprop_wvar005_tw16_200ep`); Phase 2 (fresh,
from-scratch `mlp`/markovian, matching `mlp_markovian_control`'s exact
scheme) `--epochs 68->136 --k-warmup-epochs 48->96 --k-mid-epochs
40->80` (tag `mlp_markovian_control_200ep`), preserving the k-curriculum's
proportions rather than just training longer at `k_max`.

**Phase 1 (200 epochs) result**: `val_recon_final = 0.000156` -- a NEW
best reconstruction for the entire session (previous best: `masked_mlp`'s
150-epoch run, `0.000244`).

**Phase 2 (136 epochs) result**: `val_kmax_mse = 0.0311` -- slightly
WORSE than the original 68-epoch control's `0.0280`; more epochs did not
improve rollout accuracy here despite the better underlying AE.

**Gate 3/4 results**:

| Metric | This run (200ep AE + 136ep `mlp`) | 68ep control | `mlp`+`history` (Section 12) |
|---|---|---|---|
| `val_kmax_mse` | 0.0311 | **0.0280** | 0.0258 |
| `D_KY` | 21.78 | 20.92 | **22.14** |
| `n_positive` | 11 | 12 | **13** |
| `lambda1` | **+0.133** (highest of the session) | +0.098 | +0.111 |
| DA calibration | 0.263 | **0.377** | 0.325 |
| D4 verdict | **Genuine translation representation found** | none | none |
| D6 p-value | 0.399 (not significant) | (not run) | **0.0000** |

**A genuinely new finding: D4 found a real translation representation for
the first time in this entire document.** Every other run (dozens, across
every architecture and AE tried) returned D4's "No clean linear
translation representation found in the existing latent" verdict; this is
the first to return "Genuine (approximately) equivariant translation
representation found." This is a directly-confirmed structural result
(the latent supports an actual equivariant linear representation of
physical translation), not an inferred/indirect one like D3's bandedness
or D6's temporal coherence -- and it appeared on the LEAST temporally-
coherent AE by D6's measure (`p=0.399`, not significant), suggesting D4
and D6 are capturing different, not obviously correlated, kinds of
structure. Not yet understood why more epochs specifically produced this
-- worth a targeted follow-up (e.g. does the 100-epoch version of the same
AE show partial movement toward this, or is 200 epochs a genuine
threshold effect).

**Takeaway**: this is now a fourth confirmed chaos recovery on top of the
same "global, unconstrained propagator" pattern (`D_KY` in the `20.5-22.1`
range across `mlp` markovian, `mlp`+`history`, `fno_vit`, and
`masked_mlp`-warm-start), with the highest single-run `lambda1` of the
session -- but more training did not uniformly improve every metric (
rollout accuracy and DA calibration both slipped slightly versus the
68-epoch control), reinforcing Section 11's observation that accuracy,
calibration, and "how chaotic" are three fairly independent axes, not one
that improves in lockstep with more training.

## 14. Two follow-ups: more epochs for the markovian control, and `mlp`+`history` at 200 epochs

Two more experiments launched per the user's request, both building on
Section 13's 200-epoch AE (`stage1_ae_patched_full_history3_
fullprop_wvar005_tw16_200ep`, the current best-recon checkpoint):

1. **More epochs for the from-scratch markovian `mlp` control**: same AE,
   same training scheme, doubling again -- `--epochs 272
   --k-warmup-epochs 192 --k-mid-epochs 160` (proportionally scaled from
   the 136-epoch version in Section 13, which was itself double the
   original 68). Tag `mlp_markovian_control_ae200ep_prop272ep`.
2. **`mlp`+`history` at 200 epochs**: repeats Section 12's `--aux-backbone
   mlp --mode history --n-history 3` full-propagator recipe, but at
   `--epochs 200` (matching Section 13's AE epoch count, instead of
   Section 12's original 100) -- a fresh Phase 1 AE+propagator, tag
   `history3_fullprop_wvar005_tw16_mlpaux_200ep`. Verified before
   launching that the existing (100-epoch) `mlp`+`history` propagator
   checkpoint already used the full Stage-2-standard sizing (`hidden=128,
   n_blocks=3`, matching what a from-scratch `--backbone mlp` propagator
   gets) -- `--full-propagator` was already doing this correctly, so no
   code changes were needed to satisfy "make sure it's big enough." Once
   Phase 1 completes, Phase 2 will fine-tune it via
   `--init-prop-checkpoint` with the established slow k-curriculum recipe
   (same as Section 12).

Both launched in parallel (Task 1 on top of the already-encoded AE is
cheap; Task 2 needs to re-run Phase 1 from scratch) -- no contention
observed.

**Task 1 (272 epochs, from-scratch markovian `mlp`) result**:
`val_kmax_mse = 0.0305` -- essentially unchanged from the 136-epoch
version's `0.0311` (within noise); more epochs plateaued here. Gate 3/4:
`D_KY=21.14, n_positive=12, lambda1=+0.112` (consistent chaos, matching
the 136-epoch run closely), DA calibration `0.304` (between the 136-epoch
run's `0.263` and the 68-epoch control's `0.377`), D4 verdict **"Genuine
translation representation found" again** (expected -- D4 depends only on
the AE, unchanged from Section 13's 200-epoch AE), D6 `p=0.399` (also
unchanged, same AE). This confirms the D4 finding is a real, reproducible
property of the 200-epoch AE itself, not a fluke tied to one particular
propagator's training run.

**Task 2 (`mlp`+`history`, 200 epochs) Phase 1 result**:
`val_recon_final = 0.000163` -- nearly matching the `vit`-aux 200-epoch
AE's `0.000156` (both are the two best reconstructions of the session).

**Phase 2 fine-tune result**: `val_kmax_mse = 0.0305` -- comparable to the
other 200-epoch-AE variants, slightly worse than the original 100-epoch
`mlp`+`history`'s `0.0258`.

**Gate 3/4**: `D_KY=21.71, n_positive=12, lambda1=+0.0957` (consistent
chaos). DA calibration `0.230` (lower than the 100-epoch version's
`0.325`). D3 `p=0.0010` (significant). **D4: no translation representation
found** -- unlike the `vit`-aux 200-epoch AE (Section 13), this `mlp`-aux
200-epoch AE does NOT show it, confirming that finding is specific to the
particular AE/training path, not a generic property of "any AE trained for
200 epochs." D6 `p=0.629` (not significant).

**A real bug fix validated**: this run automatically resolved the
Lyapunov spectrum on the FIRST try, using the `n_dirs = n_hist * d` fix
from Section 12 (previously hardcoded to `min(n_hist*d, 20)`, which is
what forced a manual re-run there) -- confirms the fix works as intended
for future `history`-mode Gate 3 runs.

**Working observation/hypothesis, user-noted (2026-08-31), not yet
rigorously isolated**: using a `vit` (`attn_window=4`) auxiliary
propagator during Phase 1 seems to impart more structure to the latent
variables than an `mlp` auxiliary does, even though the `vit`-family
propagator itself goes on to collapse when used as the actual Stage-2
propagator (Section 9). Circumstantial evidence so far: the ONE AE in this
entire document where D4 found a genuine translation-equivariant
representation (Section 13's 200-epoch run) was trained with a `vit`-aux
Phase 1 (`stage1_history3_fullprop_wvar005_tw16_200ep`); the same-epoch-
count `mlp`-aux AE (Section 14, immediately above) did NOT show it. D6
(temporal coherence) also first appeared significant on `vit`-aux AEs
(Sections 5/8) before later `mlp`/`local_mlp`/`masked_mlp`-aux AEs gave
mixed (sometimes significant, sometimes not) results. Not yet a controlled
comparison (AE recipe, epoch count, and aux backbone haven't been varied
one-at-a-time systematically against this specific hypothesis) -- flagged
here as a real pattern worth a targeted follow-up, not a confirmed
conclusion.

## 15. Is `delta_cap` the secret to recovering chaos? Phase-2-only ablation (user-directed, 2026-08-31)

Every winning MLP-family result in this document so far used
`delta_cap=0.5` during Phase 2. The user asked directly: "I wonder if the
delta_cap term is actually the secret to recovering chaotic latent space
dynamics... can we try a phase two training using an mlp without
delta_cap." Launched Phase 2 (`--backbone mlp --mode markovian`, no
`--prop-delta-cap`) on top of the 200-epoch `mlp`+`history`-aux AE from
Section 14, tag `mlp_no_deltacap`, using the same slow k-curriculum recipe
(`--k-max 8 --k-warmup-epochs 48 --k-mid 5 --k-mid-epochs 40
--w-varmatch 0.1`, 68 epochs).

**Result: removing `delta_cap` did not cause instability and gave the
best accuracy of the entire session.**

- **Training**: `best_val_kmax_mse = 0.013186` -- clearly better than
  every `delta_cap=0.5` MLP variant tried (`0.0258`, `0.0305`, `0.0311`
  across Sections 12-14). No divergence or instability at any point in the
  68-epoch run despite the propagator's output being architecturally
  unbounded.
- **Gate 3 (Lyapunov)**: `single_state` mode: `D_KY=21.19, n_positive=11,
  lambda1=+0.0795` -- solidly chaotic, in the same range as every
  `delta_cap` winner. (`two_step` mode failed with "not enough directions"
  -- a pre-existing sub-mode limitation, not evidence against chaos here,
  since `single_state` succeeded cleanly.)
- **Gate 3 (DA)**: `rmse_da=0.213, rmse_free=0.898, spread=0.0906,
  calibration_spread_over_rmse=0.424, skill_free_over_da=4.20` -- both the
  best calibration ratio AND the best DA skill ratio seen in this document
  (previous best calibration was `0.325`, Section 12; previous best skill
  ratios were in the 2-3x range).
- **Gate 4 (diagnostics)**: `D3 p=0.0000` (the most significant banded
  structure seen this session), `D4`: no clean translation representation
  found (consistent with every `mlp`-aux AE), `D6 p=0.0050` (significant --
  one of only a few `mlp`-aux runs where D6 came back significant).

**Conclusion**: `delta_cap` is not the secret to recovering chaos during
Phase 2 fine-tuning -- removing it produced a strictly better result on
every metric measured (accuracy, DA skill, calibration, D3/D6 structure)
with equivalent chaos. The real active ingredients identified earlier in
this document (global + unconstrained receptive field, `mode="history"`
during Phase 1) remain the explanation; `delta_cap` looks like an
architectural safety net that was never actually load-bearing for this
particular AE/training regime. Whether it matters for *other* backbones or
*less*-well-conditioned training runs is untested.

## 16. Does `delta_cap` matter during Phase 1's auxiliary propagator training? (user-directed, 2026-08-31)

Section 15's ablation tests whether `delta_cap` is load-
bearing during PHASE 2 fine-tuning, on an AE whose own Phase-1 aux
propagator training already used `delta_cap=0.5`. The natural follow-up:
does it matter during PHASE 1's joint aux training itself? Launched a
fresh Phase 1 run -- `--full-propagator --aux-backbone mlp --mode
markovian` (the full, Stage-2-standard-sized `mlp` aux, matching the
user's "the full mlp instead of a small auxiliary propagator"), same AE
recipe otherwise (`--encoder vit --pos-encoding linear --attn-window 4
--token-window 16 --w-var 0.05`), but WITHOUT `--prop-delta-cap` at all --
tag `fullprop_wvar005_tw16_mlpaux_markovian_no_deltacap`. Phase 2 fine-
tuned it via `--init-prop-checkpoint` (tag `nodeltacap_e2e`, same slow
k-curriculum recipe: `--epochs 68 --k-max 8 --k-warmup-epochs 48 --k-mid 5
--k-mid-epochs 40 --w-varmatch 0.1`), inheriting Phase 1's `delta_cap=None`
automatically -- so this is a clean "no `delta_cap` anywhere in the
pipeline, AND the same full-sized propagator carried through both phases"
test (unlike Section 15, where the no-`delta_cap` Phase-2 propagator was
built fresh rather than warm-started from a Phase-1-trained checkpoint).

**Result: this is the best result of the entire document.**

| Metric | `nodeltacap_e2e` (this run) | Section 15 (`mlp_no_deltacap`, Phase-2-only) |
|---|---|---|
| `val_kmax_mse` | **0.006546** | 0.013186 |
| `D_KY` | **21.99** | 21.19 |
| `n_positive` | **12** | 11 |
| `lambda1` | +0.0885 | +0.0795 |
| DA calibration | 0.397 | **0.424** |
| DA skill (`rmse_free/rmse_da`) | 3.43 | **4.20** |
| D3 p-value | 0.191 (not significant) | **0.0000** (highly significant) |
| D4 verdict | no translation rep | no translation rep |
| D6 p-value | 0.0070 (significant) | 0.0050 (significant) |

`val_kmax_mse=0.006546` is roughly 2x better than any other propagator in
this document, with chaos fully intact (`D_KY=21.99`, still exceeding the
reference project's `D_KY~21.4` benchmark) and DA skill in the same strong
range as every other `delta_cap`-free winner. The one notable regression
is D3 (bandedness): significant in Section 15's Phase-2-only ablation but
NOT here, where `delta_cap` was also absent from Phase 1. Since Phase 1's
own joint training is what shapes the raw encoded latent that D3 probes,
this suggests `delta_cap` presence during PHASE 1 (not Phase 2) may
matter for the banded latent structure specifically, even though it does
not matter for accuracy, chaos, or DA skill -- a nuance worth flagging
rather than a contradiction of Section 15's conclusion.

**Conclusion**: `delta_cap` is not required anywhere in the pipeline to
recover chaotic, accurate, DA-skillful latent dynamics -- removing it from
BOTH phases produced the single best accuracy result of the entire
investigation. This closes out the user's `delta_cap` line of inquiry:
it was never the active ingredient behind the collapse/no-collapse
distinction found earlier in this document (global+unconstrained
receptive field, `mode="history"` during Phase 1); it appears to be, at
most, a minor contributor to D3-style banded structure in the raw latent,
not to dynamical richness.

**Queued next** (user-directed, explicitly deferred until this experiment
completed): try `masked_mlp` as the ENCODER/DECODER itself (not just the
propagator) -- not yet started.

## 17. `masked_mlp` as the ENCODER/DECODER itself (user-directed, 2026-08-31)

New architecture: `encoder_kind="masked_mlp"`
(`ks_latent/models/autoencoder_masked_mlp.py`,
`MaskedMLPAutoencoderConfig`), mirroring `KSAutoencoderMLP`'s
`NX -> hidden -> d_latent -> reversed(hidden) -> NX` structure but with
every `Linear` replaced by `MaskedLinearRect` -- a generalization of the
propagator's `MaskedLinear` (Section 11) to RECTANGULAR weight matrices,
since here layer widths change across the stack (`NX=256 -> 512 -> 256 ->
128 -> 44`, unlike the propagator's dimension-preserving `d_latent ->
d_latent` layers throughout). Both axes of every masked layer are treated
as points on the SAME normalized circular ring (`idx/dim` fraction), so a
single `mask_window` (in `d_latent`-ring units, matching `attn_window`'s
convention elsewhere) is meaningful across layers of differing width. No
`LayerNorm` (same global-coupling leak `_MaskedMLPResidualBlock` was built
to avoid in Section 11). `mask_window=None` reduces to `KSAutoencoderMLP`
module-for-module.

**Receptive field verified BEFORE launching (same methodology as every
other backbone in this document)**: with `mask_window=4` and the real
`hidden=(512,256,128)` config, perturbing one physical input point shows
encoder latent coord 20 depends on **183 of 256** physical inputs (~71%);
perturbing one latent coord shows decoder output 128 depends on **31 of
44** latent coords (~70%). This is NOT a tightly local receptive field --
masking a single layer bounds one hop, but stacking 3-4 masked layers
compounds the reach the same way `local_mlp`'s reach grew with
`token_n_layers` (Section 10) and `masked_mlp`-as-propagator's grew with
`n_blocks` (Section 11): each layer's fixed fractional window adds to the
total spread, and with `hidden` shrinking from 512 down to 44 across 4
masked layers, ~70% of the ring is reachable end-to-end despite every
individual layer being locally masked. This is reported honestly rather
than re-tuned to force tighter locality (per this document's established
practice of measuring, not assuming, actual receptive fields) -- the
resulting architecture is accurately described as "structurally biased
toward local connectivity, with a majority-but-not-full end-to-end
receptive field," not "local" in the strict sense `local_mlp`/`masked_mlp`
achieved as PROPAGATOR backbones (where the latent dimension stayed fixed
throughout, making a tight per-block reach bound meaningful).

**Recipe**: paired with the current best propagator recipe (Section 16) --
`--aux-backbone mlp --mode markovian --full-propagator`, NO
`--prop-delta-cap`, `--w-var 0.05`, `--ae-mask-window 4` -- tag
`maskedmlp_encoder_mlpaux_markovian_no_deltacap`. Smoke-tested end-to-end
on CPU and the fast test suite (292 tests) verified passing before
launch. Phase 2 fine-tuned the Phase-1-trained full propagator via
`--init-prop-checkpoint` (same recipe as Section 16), tag
`maskedmlp_encoder_e2e`, followed by Gate 3/4.

**Result: the masked_mlp encoder did not help, and its one intended
benefit (spatial locality) came out WORSE than the plain dense
encoders.**

- Phase 1 reconstruction: `val_recon_final = 0.002043` -- roughly an
  order of magnitude worse than the `vit` encoder's best runs
  (`~0.0002-0.0004`), consistent with the encoder being architecturally
  constrained relative to a fully dense one.
- Phase 2: `val_kmax_mse = 0.012631` -- mid-pack, close to Section 15's
  `0.013186` but well behind Section 16's `0.006546`.
- Gate 3 (DA): `rmse_da=0.348, rmse_free=1.208, calibration=0.262,
  skill=3.47` -- middling.
- **D3 (spatial coherence -- the whole point of this experiment):
  bandedness `0.4659`, `p=0.0000`.** Technically still "significant"
  against the random-permutation null, but the bandedness VALUE itself is
  the worst of every real (non-collapsed) run in this document except
  `local_mlp`'s degenerate case -- worse than every plain dense `mlp`/`vit`
  encoder (`0.30-0.37` range). The architectural bias toward local
  connectivity did not translate into better same-instant spatial
  structure; if anything it went the wrong direction.
- D4: no clean translation representation (consistent with every
  non-`vit`-aux AE in this document).
- D6 (temporal coherence): `p=0.0000` -- the best (most significant) D6
  result of the entire session.
- **Gate 3 (Lyapunov): `D_KY=22.26, n_positive=13, lambda1=+0.0958`** --
  this is actually the HIGHEST `D_KY` and `n_positive` of the entire
  investigation, edging out `history3_mlpaux`'s `22.14/13` (tied on
  `n_positive`, slightly ahead on `D_KY`). A genuinely interesting split
  result: worst spatial coherence (D3) and mediocre accuracy/DA, but the
  single richest recovered chaotic attractor by this measure.

**Conclusion**: baking a local receptive field directly into the
encoder/decoder -- as opposed to using one for the PROPAGATOR (Section
11) -- did not reproduce or improve on the spatial-coherence gains seen
elsewhere in this document, and cost meaningfully on reconstruction
accuracy. Combined with the receptive-field measurement above (the
architecture is only ~70% local end-to-end after compounding across
layers, not genuinely local), the most defensible reading is that this
particular way of imposing locality on a dimension-CHANGING map (physical
grid to a much smaller latent grid) doesn't preserve enough
position-consistent structure for the mask to pay off, unlike the
dimension-PRESERVING case (`masked_mlp` as a propagator, Section 11) where
the same physical/latent index has one fixed meaning throughout.

---

## 18. Current best-model recommendation (interim, 2026-08-31)

Across every architecture/ablation tried, no single run dominates on
every metric. **Evaluation criterion, clarified by the user (2026-08-31):
the goal is NOT to maximize `D_KY` -- it's accuracy plus recovering the
true attractor's dimension, which the reference project puts at `~22`. So
the right column to optimize is `|D_KY - 22|` (closest, not highest),
alongside accuracy.** (D3/D6 columns below predate Section 22's
calibration fix and are superseded -- see that section for corrected
values; they're left here for the accuracy/`D_KY` comparison, which they
don't affect.)

| Model | `val_kmax_mse` | `D_KY` | `\|D_KY-22\|` | `n_positive` | DA calibration | DA skill |
|---|---|---|---|---|---|---|
| **`nodeltacap_e2e`** (§16) | **0.00655 (best)** | 21.99 | **0.01 (best)** | 12 | 0.397 | 3.43 |
| `maskedmlp_encoder_e2e` (§17) | 0.01263 | 22.26 | 0.26 | 13 | 0.262 | 3.47 |
| `mlp_no_deltacap` (§15) | 0.01319 | 21.19 | 0.81 | 11 | **0.424** | **4.20** |
| `history2_e2e` (§19) | 0.01356 | 21.66 | 0.34 | 12 | 0.300 | 2.26 |
| `history3_mlpaux` 100ep (§12) | 0.0258 | 22.14 | 0.14 | 13 | 0.325 | 3.96 |
| `history3_mlpaux_200ep` (§14) | 0.0305 | 21.71 | 0.29 | 12 | 0.230 | 2.88 |

**Recommendation: `nodeltacap_e2e`** (Section 16 -- full-sized `mlp` aux
propagator, `mode=markovian`, `delta_cap` removed from BOTH Phase 1 and
Phase 2, on the `mlp`+`history` 200-epoch AE) -- and under the clarified
criterion this is now an even cleaner win than originally stated: it is
simultaneously the BEST accuracy (by ~2x over everything else) AND the
CLOSEST `D_KY` to the true attractor's `~22` (off by only `0.01`, not just
"exceeding" some benchmark) of every model in this document. No tradeoff
between the two criteria that matter -- this one dominates on both. DA
skill (`3.43x`) is solidly mid-pack, not the best (`mlp_no_deltacap` has
`4.20x`), but not a weak point either.

**Runners-up, for reference**:
- **`mlp_no_deltacap`** if DA calibration/skill specifically is the
  priority metric (best of the document on both) -- at the cost of ~2x
  worse accuracy and `D_KY` noticeably farther from 22 (`21.19`, off by
  `0.81`).
- **`maskedmlp_encoder_e2e`** is close on both accuracy and `|D_KY-22|`
  and is the one checkpoint with genuinely-surviving (not miscalibrated)
  D3/D6 structure (Section 22) -- a reasonable second choice if latent
  spatial structure specifically matters for downstream work (e.g. the
  PDE-in-latent-space idea discussed earlier).

## 19. History-length sweep: does `n_history=2` or pure `markovian` reproduce the 200-epoch `mlp`+`history` result? (user-directed, 2026-08-31)

Section 12/14 established that `mode="history"` (`n_history=3`) during
Phase 1's joint training, not the `vit` backbone, is what made the
original best AE special. The natural next question: how much history is
actually needed? Launched two new Phase 1 runs, EXACT same recipe as
`history3_fullprop_wvar005_tw16_mlpaux_200ep` (Section 14) -- `--encoder
vit --pos-encoding linear --attn-window 4 --token-window 16 --w-var 0.05
--full-propagator --prop-delta-cap 0.5 --epochs 200 --aux-backbone mlp` --
varying only the history setting:

1. `--mode history --n-history 2`, tag
   `history2_fullprop_wvar005_tw16_mlpaux_200ep`
2. `--mode markovian` (no history at all), tag
   `markovian_fullprop_wvar005_tw16_mlpaux_200ep`

Both chained into their own Phase 2 fine-tune (`--init-prop-checkpoint`,
same slow k-curriculum recipe as every other Phase-2 fine-tune in this
document) and Gate 3/4, run sequentially (not in parallel) to avoid MPS
contention between the two Phase-1/Phase-2 training jobs -- Gate 3/4
itself runs on CPU and does not contend with MPS training, so those still
run as 3 parallel jobs per tag as usual.

**Results** (evaluation criterion per the user's 2026-08-31 clarification:
accuracy plus `|D_KY - 22|`, not maximizing `D_KY`):

| Tag | `val_kmax_mse` | `D_KY` | `\|D_KY-22\|` | `n_positive` | DA calibration | DA skill |
|---|---|---|---|---|---|---|
| `history2_e2e` (`n_history=2`) | 0.01356 | 21.66 | 0.34 | 12 | 0.300 | 2.26 |
| `markovian_e2e` (no history) | **0.00911** | 21.78 | **0.22** | **13** | **0.384** | **2.95** |
| (reference: `history3_mlpaux_200ep`, original `n_history=3` recipe, Section 14) | 0.0305 | 21.71 | 0.29 | 12 | 0.230 | 2.88 |

**Surprising conclusion: pure `markovian` (no history at all) beats BOTH
`n_history=2` AND the original `n_history=3` recipe** on accuracy and on
`|D_KY-22|`, at this exact matched recipe (`--encoder vit --attn-window 4
--token-window 16 --w-var 0.05 --full-propagator --prop-delta-cap 0.5
--epochs 200`, `--aux-backbone mlp`, only the history setting varied).
This significantly undercuts Sections 12/14's earlier conclusion that
`mode="history"` during Phase 1 was a load-bearing "active ingredient" --
at matched recipe and epoch count, history-mode is not just unnecessary
here, it's outperformed by no history at all. The likely resolution: the
original `mlp`+`history` result's strength (Section 12, `D_KY=22.14`) came
from a specific 100-epoch checkpoint that happened to train well, and this
203-epoch, delta_cap=0.5, full-propagator-recipe head-to-head suggests
history-mode's apparent advantage doesn't reproduce cleanly across
epoch-count/recipe variations -- more likely run-to-run/checkpoint
variance than a genuine causal effect of `mode="history"` itself. D3/D6/D7
diagnostics (corrected calibration, Section 22) found no significant
structure in either AE (`history2_e2e`: D3 `p=0.011`, D6 `p=0.013`,
D7 `p=0.088`, actually a mixed/borderline picture; `markovian_e2e`: D3
`p=0.441`, D6 `p=0.301`, D7 `p=0.313`, cleanly non-significant across the
board).

## 20. Permutation-informed local propagator (user-directed, 2026-08-31)

**The idea, as proposed**: "train an autoencoder with good spatial
coherence, then after stage 1, we apply D3, to determine the best
permutation for local spatial coherence. We could then apply this
permutation to all latent states, and train a stage 2 propagator with
these permuted, hopefully spatially coherent states" -- using a
`masked_mlp` with a small receptive field as the proof of concept.

**Does it make sense? Yes, with one important correction to how D3 works.**
D3 (`jacobian_coupling` + `bandedness_p_value`,
`ks_latent/analysis/diagnostics.py`) is NOT a direct map from the
encoder/decoder to physical space -- it is computed from a trained
PROPAGATOR's Jacobian, `A[k,l] = E|d z_{n+1,k} / d z_{n,l}|`, then
Fiedler-seriated and scored for bandedness against a random-permutation
null. So "apply D3 after Stage 1" concretely means: take Phase 1's own
(already-trained, full-sized) propagator, compute its Jacobian coupling
graph, and find the reordering that makes it most banded. This is exactly
available, and in fact we already have the single best example of it on
file: `history3_mlpaux_200ep` (Section 14) has the best genuinely-
significant D3 result in the whole document (bandedness `0.3081`,
`p=0.001`), computed from its Stage-2, chaos-recovering, `mlp`+`history`
propagator's Jacobian -- a better source of evidence than a Phase-1,
not-yet-chaos-tuned propagator's Jacobian would be, so this experiment
reuses that existing permutation rather than recomputing a fresh one.

**Why there's real reason to expect this could matter**: the saved Fiedler
permutation is essentially scrambled relative to the raw latent index
order --
`[39,40,15,18,21,7,19,8,9,12,37,1,10,42,0,6,33,31,38,30,16,3,23,13,43,5,32,
4,17,20,25,22,14,35,41,28,26,36,34,27,24,29,2,11]`, nowhere close to
`[0,1,2,...,43]`. Every previous local-receptive-field propagator in this
document (`local_mlp`, Section 10; `masked_mlp`, Section 11) imposed
locality in that RAW, arbitrary index order and collapsed. Those
experiments never actually tested "local in the ordering where the
learned dynamics are genuinely most local" -- they tested "local in
whatever order `encode()` happens to emit," which per this permutation is
a materially different constraint. This gives a concrete, mechanistic
alternative explanation for the earlier collapses that is directly
testable.

**Why to stay cautious**: the bandedness improvement found is real but
modest -- `0.31` under the best-found reordering vs. `~0.35-0.37` typical
for an unconstrained propagator's Jacobian under the SAME Fiedler
procedure (Section 18's table) -- not a dramatic tightening. A meaningful
fraction of coupling mass likely still sits outside any narrow band even
under the best permutation found. It's also plausible that KS's true
reduced-order dynamics has intrinsically non-local mode coupling (the
cubic/convective nonlinearity mixes distant wavenumbers in a Galerkin
truncation) that no single coordinate permutation can fully localize -- in
which case a narrow-window propagator would still fail here, which would
itself be a clean, useful negative result ruling out "wrong basis" as the
explanation for the earlier collapses.

**Implementation** (completed, ready to launch automatically once Section
19's sweep finishes -- no AE retraining required anywhere):
- `ks_latent/models/permuted_autoencoder.py` (new): `PermutedAutoencoder`,
  a pure `nn.Module` wrapper around any existing (unmodified, weights
  untouched) autoencoder -- `encode()` reindexes its output by a fixed
  permutation, `decode()` un-reindexes before calling through. Since
  `decode(z) == decode(P^-1(Pz))` for any permutation `P`, this is an
  exact, lossless relabeling: verified bit-for-bit reconstruction equality
  against the unwrapped AE in `tests/unit/test_permuted_autoencoder.py`
  (8 new tests, all passing). Also proxies `.cfg` so code reading
  `ae.cfg.d_latent` directly (`run_diagnostics.py`,
  `ks_latent/training/loops.py`) keeps working unmodified.
- `--latent-permutation <path>` CLI flag added to `train_stage2_patched.py`,
  `run_analysis_suite.py`, `run_da_pff.py`, `run_diagnostics.py` (every
  script that loads an AE checkpoint) -- accepts a bare `.npy` array or a
  `diagnostics_arrays{tag}.npz` (reads its `d3_permutation` key directly,
  the exact format `run_diagnostics.py` already writes).
- Smoke-tested end-to-end on the CLI (not just unit tests): a real
  `train_stage2_patched.py --backbone masked_mlp --attn-window 2
  --latent-permutation ...` run completed successfully on MPS; `--help`
  verified on all 4 scripts.

**Recipe queued**: on `history3_mlpaux_200ep`'s existing AE (unmodified),
two Stage-2 runs at `--backbone masked_mlp --attn-window 2` (a
deliberately small receptive field, per the user's "smaller receptive
field just as a proof of concept"), same slow k-curriculum recipe as
every other Stage-2 comparison in this document:
1. `permuted_maskedmlp_w2` -- WITH `--latent-permutation` (the D3 Fiedler
   order above).
2. `maskedmlp_control_w2` -- WITHOUT it (natural/raw latent order) -- a
   same-AE, same-window control isolating the permutation's own effect
   from "yet another masked_mlp run."

Both followed by Gate 3/4 (with the same `--latent-permutation` flag
applied to the permuted run's Gate 3/4, so D3/Lyapunov/DA all evaluate
consistently in the coordinate system the propagator was actually trained
in). Chained to launch automatically once Section 19's history-length
sweep pipeline finishes (to avoid MPS contention) via a detached script
polling that pipeline's PID.

**Result: a clean negative -- the permutation did not help, at all.**

| Tag | `val_kmax_mse` | `D_KY` | `n_positive` | DA skill | D3 (p) | D6 (p) | D7 (p) |
|---|---|---|---|---|---|---|---|
| `permuted_maskedmlp_w2` (D3-permuted) | 0.5937 | **0.0 (collapsed)** | 0 | 1.18 | 0.0000 | 0.981 (NS) | 0.555 (NS) |
| `maskedmlp_control_w2` (natural order) | 0.5959 | **0.0 (collapsed)** | 0 | 1.18 | 0.0000 | 0.982 (NS) | 0.546 (NS) |

Both runs collapsed to a fixed point (`D_KY=0`, negative `lambda1`), with
essentially IDENTICAL accuracy, DA skill, and diagnostics regardless of
whether the D3-derived permutation was applied. Note also `val_kmax_mse
~0.59` is far worse than any other propagator in this document (the next
worst, `local_mlp`, was nowhere near this bad) -- `attn_window=2` is
evidently too narrow for `masked_mlp` on this AE to do anything useful at
all, permuted or not. **Conclusion: reordering the latent by a (since
Section 22, statistically unconfirmed) "best" permutation does not rescue
a genuinely narrow receptive field from collapse -- locality itself is the
limiting factor here, not the choice of basis.** This is consistent with
the broader finding running through Sections 9-11: `local_mlp`/`masked_mlp`
collapsed regardless of softmax, and now regardless of latent reordering
too. Given Section 22's finding that this particular permutation's
"significance" didn't survive correct calibration anyway, this negative
result should be read as "an exploratory permutation choice failed to
help," not as strong evidence against the permutation-informed idea in
general -- Section 23 (using the ONE permutation that DID survive proper
calibration) is the more decisive test of that.

## 21. A purely empirical, propagator-free permutation: D6 (user-directed, 2026-08-31)

Follow-up question: "is there no way we could take a rollout of latent
variables, computed using the autoencoder and actual progression of the
system, then empirically determine local coherence (neighboring states
covariance vs global covariance or something?) ... This seems more stable
than the current approach [Section 20's D3-based one]."

**This already exists in this codebase, under a different name: D6.**
`local_temporal_coherence` + `temporal_coherence_diagnostic`
(`ks_latent/analysis/diagnostics.py`, added Section 8) does exactly what
was described, with no propagator anywhere in the computation: take real
encoded trajectories `z(t) = encode(u(t))` from the actual system's true
rollout, and for every pair of latent channels compute the best-lag
windowed correlation across many overlapping time windows (median
aggregation over windows for robustness against a chaotic trajectory's
non-stationarity), Fiedler-seriate that channel-by-channel matrix, and
score its bandedness against 1000 random permutations -- the same
statistical machinery D3 uses, applied to a completely different
(propagator-free) matrix. The user's intuition about why this should be
"more stable" is exactly right: D6's permutation is a property of the
AUTOENCODER'S OWN LATENT GEOMETRY plus the true dynamics, not of any one
downstream propagator's idiosyncratic, possibly-overfit Jacobian -- D3's
permutation could in principle differ across different propagators trained
on the identical latent space, while D6's does not depend on a propagator
at all.

**Which AE/permutation to use**: unlike Section 20 (which reused
`history3_mlpaux_200ep`'s D3 result because it was the best SIGNIFICANT D3
in the document), the best D6 result in the entire document belongs to a
DIFFERENT checkpoint -- the original 100-epoch `history3_mlpaux`
(Section 12): `D6 p=0.0000`, the strongest (most significant) temporal-
coherence result found anywhere, versus `history3_mlpaux_200ep`'s D6
`p=0.629` (not significant at all). So this experiment pairs the D6
permutation with ITS OWN source AE (`stage1_ae_patched_full_
history3_fullprop_wvar005_tw16_mlpaux.pt`, unmodified, no retraining),
not the 200-epoch one used in Section 20. Sanity check: the saved D6
permutation for this AE,
`[39,37,7,43,42,20,27,38,9,6,12,23,22,11,18,3,10,15,0,30,31,21,24,1,25,33,
17,16,40,14,4,32,28,35,2,34,19,8,36,29,13,41,26,5]`, is both far from the
identity AND meaningfully different from that same AE's own D3 permutation
`[15,3,18,10,26,0,16,9,39,1,42,13,41,14,33,6,8,19,38,7,28,23,37,20,29,17,
32,22,5,11,35,24,31,25,43,36,12,34,27,4,2,40,30,21]` -- confirming D3 and
D4 are picking up genuinely different structure (dynamical-Jacobian
coupling vs. raw-data temporal coherence), not just rediscovering the same
ordering twice.

**Implementation**: generalized `load_latent_permutation` (`ks_latent/
models/permuted_autoencoder.py`) to take a `key` argument (`"d3_permutation"`
default, or `"d6_permutation"`), and added a matching `--latent-permutation-
key` CLI flag to all 4 scripts alongside `--latent-permutation` -- no new
wrapper logic needed, `PermutedAutoencoder` itself is already agnostic to
where its permutation came from. Added `test_load_latent_permutation_from_
npz_d6_key` to the test suite (301 tests passing).

**Recipe queued** (same structure as Section 20, for direct comparability):
on `history3_mlpaux`'s (100ep) existing AE, `--backbone masked_mlp
--attn-window 2`, standard slow k-curriculum recipe:
1. `d6_permuted_maskedmlp_w2` -- WITH `--latent-permutation ... 
   --latent-permutation-key d6_permutation`.
2. `maskedmlp_control_100ep_w2` -- WITHOUT it (natural order, same 100ep
   AE) -- control.

Both followed by Gate 3/4. Chained to launch automatically once Section
20's pipeline finishes (to avoid MPS contention).

**Result: the same clean negative as Section 20.**

| Tag | `val_kmax_mse` | `D_KY` | DA skill | D3 (p) | D6 (p) | D7 (p) |
|---|---|---|---|---|---|---|
| `d6_permuted_maskedmlp_w2` (D6-permuted) | 0.5535 | **0.0 (collapsed)** | 1.16 | 0.0000 | 0.236 (NS) | 0.819 (NS) |
| `maskedmlp_control_100ep_w2` (natural order) | 0.5550 | (Lyapunov still computing at time of writing) | 1.20 | 0.0000 | 0.226 (NS) | 0.834 (NS) |

Both collapsed again, permuted and unpermuted essentially indistinguishable
on every metric (accuracy, DA skill, D3/D6/D7 diagnostics) -- the same
outcome as Section 20's D3-permuted pair, now confirmed with a completely
different permutation (D6-derived, propagator-free) and a different AE
(`history3_mlpaux`'s 100-epoch checkpoint rather than the 200-epoch one).
Between Sections 20 and 21, this document now has FOUR masked_mlp-with-
small-window data points on real AEs (D3-permuted, D6-permuted, and two
natural-order controls), and all four collapse identically regardless of
latent ordering. **This is a fairly strong, consistent signal that
`attn_window=2` is simply too narrow for `masked_mlp` to work AT ALL on
these AEs, independent of which permutation (if any) is applied** -- the
receptive-field width itself, not the choice of basis, is the limiting
factor. Section 23 (the one permutation that survived Section 22's
calibration correction, on an AE that is ALSO architecturally local by
construction) is the one remaining chance for this general idea to show a
different outcome.

## 22. Correction: D3/D6's shared statistical test was miscalibrated at the real `d=44` scale -- every prior "significant" claim in this document is superseded

User pushback that led here: "I thought D6 was for temporal coherence, not
spatial? I want latent variables to have some local structure where
neighboring latent variables vary together." This is the genuinely
SAME-INSTANT question (D6 is temporal/lagged), leading to a new diagnostic
(D7, below) -- and building it surfaced a serious pre-existing calibration
bug in the statistical machinery D3 and D6 both already relied on.

**The bug**: `bandedness_p_value` (used by D3 and D6, unchanged since
before this session) tests "does the Fiedler-vector-optimized ordering of
matrix `A` score higher than a RANDOM (unoptimized) ordering of the SAME
`A`." Since the Fiedler vector is specifically chosen (via spectral
relaxation) to concentrate `A`'s largest entries near the diagonal, it will
nearly always beat an unoptimized ordering -- even on PURE SYMMETRIC NOISE
with no true structure at all. Direct empirical test: `d=44` (the actual
latent dimension used everywhere in this project) pure-noise matrices gave
`p<0.05` in 6 of 8 random seeds (nominal false-positive rate should be
~5%, i.e. ~0.4 of 8). **This was never caught because the existing D3/D6
calibration tests only ever checked `d=12`, not the real `d=44`.**

**The fix**: `bandedness_p_value_entry_shuffle` (new,
`ks_latent/analysis/diagnostics.py`, next to `bandedness_p_value`) instead
shuffles `A`'s off-diagonal VALUES among themselves (preserving symmetry,
the diagonal, and the exact multiset of observed magnitudes -- critically,
this means real and null matrices always have the same total mass, unlike
an earlier attempt at fixing this via raw-data column-shuffling, which
changed that and gave nonsensical results, including `p=1.0` on data with
GENUINE strong planted structure) and re-optimizes a FRESH Fiedler
ordering for each shuffled surrogate before scoring it. Verified: 0/6
seeds `p<0.05` on `d=44` pure noise (correct calibration), while still
detecting planted ring structure cleanly at `d=44` (`p=0.0000` on a
`rho=0.85` circulant-covariance test case). D3 (`coupling_graph_diagnostic`,
`coupling_graph_diagnostic_history`) and D6 (`temporal_coherence_diagnostic`)
were both switched over to this corrected null -- a direct drop-in swap,
since the fix operates on the already-computed `(d,d)` matrix regardless of
what produced it (propagator Jacobian for D3, lagged correlation for D6),
not on the raw underlying data. 7 new tests added
(`tests/unit/test_diagnostics_d7.py`), 308 total passing.

**The damage, quantified** -- every D3/D6 p-value on real checkpoints,
recomputed with the corrected null (permutations themselves are UNCHANGED,
only p-values -- `_fiedler_permutation` is deterministic given `A` and
doesn't depend on the null-testing method):

| Checkpoint | D3 (old → corrected) | D6 (old → corrected) |
|---|---|---|
| `history2_e2e` | 0.0040 → **0.3420** | 0.0100 → **0.3680** |
| `history3_mlpaux` | 0.0680 → **0.7520** | 0.0000 → **0.2180** |
| `history3_mlpaux_200ep` | 0.0000 → **0.4760** | 0.6480 → 0.9860 |
| `local_mlp` (markovian) | 0.0000 → **0.0000** (survives) | 0.0000 → 0.1860 |
| **`maskedmlp_encoder_e2e`** | 0.0000 → **0.0000** (survives) | 0.0000 → **0.0000** (survives) |
| `maskedmlp_warmstart` | 0.0320 → **0.3120** | 0.2260 → 0.8940 |
| `mlp_control` | 0.0000 → **0.2100** | 0.0080 → **0.4140** |
| `mlp_control_200ep` | 0.0000 → **0.0020** (survives, weaker) | 0.4180 → 0.9580 |
| `mlp_control_272ep` | 0.0000 → **0.0120** (survives, weaker) | 0.4180 → 0.9580 |
| `mlp_no_deltacap` | 0.0000 → **0.0040** (survives, weaker) | 0.0080 → **0.4140** |
| `nodeltacap_e2e` | 0.2040 → 0.6580 | 0.0060 → **0.4360** |

**Nearly every "significant D3/D6" claim made in Sections 5-21 of this
document does NOT survive correction and should be treated as superseded**
-- including the two results Sections 20/21's queued permutation
experiments were specifically built around (`history3_mlpaux_200ep`'s D3,
`history3_mlpaux`'s D6). The clear exception, standing out sharply: **
`maskedmlp_encoder_e2e` (Section 17) is the ONLY checkpoint whose D3 AND D6
BOTH remain robustly significant after correction** -- along with a
handful of `mlp`-propagator-only D3 results that survive in weakened form.
This makes physical sense in hindsight: `maskedmlp_encoder_e2e` is the one
AE in this document whose ENCODER/DECODER is architecturally biased toward
local connectivity by construction (Section 17's `masked_mlp` encoder), so
finding genuine (not just apparent) coupling structure there, while the
fully-dense `vit`/`mlp` encoders mostly don't, is a coherent, expected
result rather than a coincidence. Consistent with this: its D3-discovered
permutation, `[20,21,18,15,22,23,19,17,25,13,16,26,14,24,27,29,11,10,28,
30,12,31,9,32,8,7,33,6,5,34,4,36,35,43,3,0,1,2,37,38,42,39,40,41]`, is
visibly much CLOSER to the identity ordering (many short adjacent runs:
`20,21`; `22,23`; `9,10,11`; `0,1,2,3`; `37,38`; `39,40,41`) than every
other checkpoint's near-fully-scrambled discovered permutation -- i.e. this
encoder's own latent index already carries real, near-correct spatial
meaning, exactly what Sections 10/11/17's local-receptive-field
architectural bias was trying to induce.

**D7 computed retroactively across every AE checkpoint** (user-directed
follow-up, 2026-08-31): D7 needs only the AE (encode real trajectories,
no propagator involved at all), so it's cheap to compute after the fact
for every distinct AE in this document, corrected calibration throughout:

| AE (-> sections using it) | D7 bandedness | D7 p-value | Significant? |
|---|---|---|---|
| `history3_mlpaux` (100ep, §12) | 0.2934 | 0.9180 | no |
| `history3_mlpaux_200ep` (§14, base of §15) | 0.3490 | 0.1980 | no |
| `history3_vitaux_100ep` (§5 `fno_vit`, §9 `mlp_control`) | 0.2975 | 0.4920 | no |
| `history3_vitaux_200ep` (§13 `mlp_control_200/272ep`) | 0.3481 | 0.4300 | no |
| `local_mlp` AE (§10) | 0.3054 | **0.0240** | **yes** |
| `masked_mlp` local/warmstart AE (§11) | 0.3211 | 0.8100 | no |
| **`maskedmlp_encoder_e2e` AE (§17)** | **0.4099** | **0.0000** | **yes (strongest)** |
| **`nodeltacap_e2e` AE (§16 -- the recommended best model)** | 0.3045 | **0.0180** | **yes** |
| `history2_e2e` AE (§19) | -- | 0.0880 | marginal |
| `markovian_e2e` AE (§19) | -- | 0.3130 | no |

Only 3 of 10 AEs show genuinely significant same-time spatial coherence.
Two consistent with the D3/D6 story above (`local_mlp`, whose propagator
is itself architecturally local and evidently shaped the AE's own latent
geometry during Phase 1's joint training; `maskedmlp_encoder_e2e`, by far
the strongest, consistent with its D3/D6 also surviving). The third is
new and worth calling out on its own: **`nodeltacap_e2e` -- Section 18's
recommended best overall model -- has genuine, non-spurious same-time
spatial structure**, something its own (non-significant) D3 result missed
entirely. This is a reassuring finding for the recommendation: the model
that wins on accuracy and `|D_KY-22|` also turns out to have real
(D7-detectable) local latent structure, not just competitive numbers with
nothing underneath. Every AE built on a fully-dense `vit` encoder with a
plain- or history-mode propagator (the `history3_*` and `history2/
markovian_e2e` rows) shows no significant same-time coherence at all --
consistent with the emerging picture that local structure only appears
when something in the pipeline (an architecturally-local propagator or
encoder, or specifically the no-`delta_cap` recipe) actively pushes toward
it, rather than emerging for free.

**What this does NOT affect**: Section 18's best-model recommendation was
built on accuracy (`val_kmax_mse`), Lyapunov spectrum (`D_KY`/`n_positive`/
`lambda1`), and DA calibration/skill -- none of which touch
`bandedness_p_value` -- so that ranking and recommendation stand
unchanged. D4 (translation representation) is a separate, unrelated
statistical test (linear-map residual thresholding, not Fiedler-seriation
bandedness) and is also unaffected.

**Revised plan**: given `maskedmlp_encoder_e2e` is now the only checkpoint
with genuinely-surviving significance, it is the best-motivated candidate
for the permutation-informed local-propagator idea -- better-motivated
than either of Sections 20/21's already-queued choices. Section 23 (next)
queues this as a third, additional experiment rather than replacing the
in-flight ones (their results remain informative as exploratory data
points regardless of the calibration issue -- Sections 20/21 test whether
SOME empirically-derived permutation helps a narrow propagator avoid
collapse; that experimental question doesn't strictly require its input
permutation to have a "significant" p-value to be worth running, though it
is no longer the strongest-motivated choice available).

## 23. The best-motivated permutation experiment: `maskedmlp_encoder_e2e`'s surviving D3 structure (2026-08-31)

Following directly from Section 22: `maskedmlp_encoder_e2e` (Section 17 --
`masked_mlp` encoder/decoder, `mlp` aux propagator, markovian, no
`delta_cap`) is the only checkpoint in this document whose D3 (and D6)
bandedness survives the corrected calibration. Its D3-discovered
permutation is reused directly from the already-saved
`artifacts/diagnostics_arrays_maskedmlp_encoder_e2e.npz` (`d3_permutation`
-- unaffected by the calibration fix, since the permutation itself is
just `_fiedler_permutation(A)`, computed the same way before and after).

**Recipe queued** (same structure as Sections 20/21): on
`maskedmlp_encoder_e2e`'s existing AE
(`stage1_ae_patched_full_maskedmlp_encoder_mlpaux_markovian_no_deltacap.pt`,
unmodified), `--backbone masked_mlp --attn-window 2`, standard slow
k-curriculum recipe:
1. `d3_permuted_maskedmlp_encoder_w2` -- WITH `--latent-permutation ...
   --latent-permutation-key d3_permutation`.
2. `maskedmlp_control_encoderAE_w2` -- WITHOUT it (natural order, same
   AE) -- control.

One caveat worth flagging in advance: since this permutation is already
close to the identity ordering (see Section 22), this experiment may end
up closer to a null result BY CONSTRUCTION -- permuted and unpermuted may
behave similarly simply because the permutation doesn't move very much.
That would still be informative (it would mean this AE's own latent index
already carries most of the exploitable local structure, with or without
explicit reordering).

Both followed by Gate 3/4. Chained to launch automatically once Section
21's pipeline finishes (to avoid MPS contention).

**Result: collapsed again -- 6 of 6 `attn_window=2` `masked_mlp` runs in
this document now collapse, regardless of permutation source or AE.**

| Tag | `val_kmax_mse` | `D_KY` | DA skill | D3 (p) | D6 (p) | D7 (p) |
|---|---|---|---|---|---|---|
| `d3_permuted_maskedmlp_encoder_w2` | 0.5589 | **0.0 (collapsed)** | 1.55 | 0.0000 | **0.0010 (sig)** | **0.0000 (sig)** |
| `maskedmlp_control_encoderAE_w2` (natural order) | 0.5428 | **0.0 (collapsed)** | 1.94 | 0.0000 | **0.0000 (sig)** | **0.0000 (sig)** |

The caveat flagged in advance (permutation close to identity, so permuted
vs. unpermuted might not differ much) is confirmed, but the deeper result
is stronger than that: BOTH conditions here have genuinely significant D6
AND D7 -- real, non-artifact local structure in this AE's latent space,
in EITHER the permuted or the natural ordering -- and it still didn't
matter. The propagator collapsed anyway. Combined with Sections 20/21's
four collapses (using ultimately-non-significant permutations), this
closes out the permutation-informed-locality question for this document:
**across every combination tried -- two permutation sources, two AEs, one
of which has genuine, statistically-confirmed local structure -- an
`attn_window=2` `masked_mlp` propagator has never once avoided collapse.
The receptive field width itself is the bottleneck, not the latent
ordering.** A wider window might behave differently (untested at this
window size specifically with a real permutation), but "find the right
basis" as a fix for narrow-receptive-field collapse is not supported by
any of this document's evidence.

## 24. Banded pooling on `nodeltacap_e2e`'s own architecture, plus cross-AE propagator transplant (user-directed, 2026-08-31)

New request: take `nodeltacap_e2e`'s exact recipe (Section 16 -- the
current best-recommended model), but replace the `vit` encoder's global
`pool="mean"` with a LEARNED, circular-band-MASKED pooling operator ("zero
off the 4th diagonal"), lower `attn_window` from 4 to 3, keep the same aux
propagator settings Phase 1 originally used, and for Phase 2 warm-start
from `nodeltacap_e2e`'s OWN already-trained propagator rather than a fresh
one.

**New architecture implemented**: `pool="banded"`
(`ks_latent/models/autoencoder_vit.py`'s `KSAutoencoderViT`,
`ViTAutoencoderConfig.pool_bandwidth`) -- replaces `"mean"`'s global
average + dense `Linear(d_model, d_latent)` with `MaskedLinearRect`
(reusing the exact class the `masked_mlp` encoder, Section 17, already
uses) applied directly on the token axis: each `z_k` is a LEARNED
combination of only its nearby tokens (within `pool_bandwidth`, in
`d_latent`-ring units -- `4`, per the user's "zero off the 4th diagonal"),
not a dense function of a global average. Decoding mirrors this with an
independently-learned banded map back out to `n_tokens`. Unlike
`pool="local"` (fixed uniform mean over a non-overlapping window), this is
a learned, overlapping-band weighting with no averaging constraint --
genuinely different mechanism, not a renaming of an existing option.
`--pool banded --pool-bandwidth <N>` CLI flags added to
`train_stage1_patched.py`. Verified: forward pass shapes correct,
`MaskedLinear`'s off-band entries stay exactly zero through training
(regression test mirroring Section 17's), 312 tests passing.

**Receptive field, checked before launching (same discipline as every
other backbone in this document)**: at the real config (`pool_bandwidth=4`,
`attn_window=3`), perturbing one physical input shows encoder latent[20]
depends on 184/256 (~72%) physical inputs; perturbing one latent shows
decoder output[128] depends on 31/44 (~70%) latent coords -- essentially
the same ~70% figure Section 17's `masked_mlp` encoder showed. Same
underlying reason: the pooling matrix ITSELF is genuinely banded, but the
`n_blocks=3` ViT attention layers running BEFORE pooling (even restricted
to `attn_window=3`) already mix tokens together across a wider span before
the banded map ever sees them, so end-to-end locality is again
"structurally biased, not strict" -- reported honestly rather than
re-tuned to claim more than is true.

**Recipe queued**:
1. Phase 1: `--encoder vit --pool banded --pool-bandwidth 4 --attn-window 3
   --pos-encoding linear --token-window 16 --w-var 0.05 --full-propagator
   --aux-backbone mlp --mode markovian` (no `--prop-delta-cap` -- matching
   `nodeltacap_e2e`'s own Phase 1 recipe exactly except for the pooling
   mechanism and the lowered `attn_window`), tag
   `bandedpool4_attn3_mlpaux_markovian_no_deltacap`.
2. Phase 2: rather than fine-tuning this new Phase 1's own (freshly,
   less-trained) propagator, warm-start via `--init-prop-checkpoint`
   pointing at `nodeltacap_e2e`'s OWN FINAL, fully-trained propagator
   (`stage2_prop_patched_full_nodeltacap_e2e.pt`) -- this tests whether an
   already-good propagator, transplanted onto a DIFFERENT (more
   locally-structured-by-construction) latent space via ordinary Phase 2
   fine-tuning, adapts well, a genuine cross-AE transfer test. Standard
   slow k-curriculum recipe, tag
   `bandedpool4_attn3_from_nodeltacap_e2e_prop`.

Followed by Gate 3/4. Chained to launch automatically once Section 23's
pipeline finishes (to avoid MPS contention).

**Result: it did NOT collapse -- the only positive (non-collapsed) result
among every local/banded architecture tried in Sections 20-24.**

| Metric | `bandedpool4_attn3_from_nodeltacap_e2e_prop` | `nodeltacap_e2e` (recommended best, §16) |
|---|---|---|
| Phase 1 reconstruction | 0.007037 | ~0.0002-0.0004 (typical dense `vit`) |
| `val_kmax_mse` | 0.076687 | **0.006546 (best)** |
| `D_KY` | 19.81 (single_state) / 19.85 (two_step) | **21.99** |
| `\|D_KY-22\|` | 2.19 | **0.01 (best)** |
| `n_positive` | 10 | 12 |
| `lambda1` | 0.080 | 0.089 |
| DA calibration | 0.271 | 0.397 |
| DA skill | 2.81 | 3.43 |
| D3 | **sig, p=0.0000** | NS, p=0.191 |
| D6 | **sig, p=0.0000** | sig, p=0.007 |
| D7 | **sig, p=0.0000** | sig, p=0.018 |

The transplanted propagator (already fully trained on `nodeltacap_e2e`'s
own, differently-organized latent space) successfully ADAPTED to this new,
architecturally-local latent space via ordinary Phase 2 fine-tuning and
came out genuinely chaotic (`D_KY~19.8`, comfortably positive `n_positive`
and `lambda1`) -- a real, nontrivial result given that every
`attn_window=2` `masked_mlp` propagator in Sections 20/21/23 collapsed
outright. This is also, along with `maskedmlp_encoder_e2e`, one of only
two checkpoints in the entire document with D3, D6, AND D7 all
simultaneously significant -- the most comprehensive evidence of genuine
local latent structure found anywhere in this investigation.

**But it does not beat `nodeltacap_e2e` on the criteria that matter for
this project** (accuracy, `|D_KY-22|`): worse accuracy by ~12x, and
`D_KY` noticeably farther from the true attractor's `~22` (`19.81` vs
`21.99`). The honest reading: banked/local architectural constraints
(this section's `pool="banded"`, Section 17's `masked_mlp` encoder)
reliably produce real, statistically-confirmed local structure and CAN
support chaotic dynamics when paired with a sufficiently capable
propagator (unlike a genuinely-narrow-window propagator, which collapses
regardless) -- but they consistently cost real accuracy and move `D_KY`
away from the target, a real tradeoff rather than a free win. Across this
whole document, no locally-structured architecture has yet beaten
`nodeltacap_e2e`'s combination of best accuracy and closest `D_KY` to 22 --
that recommendation stands.

---

## 25. Banded pooling, take two: wider `attn_window`, wider band, more epochs (user-directed, 2026-08-31)

Follow-up on Section 24: same design (Phase 1 trains a fresh `pool="banded"`
AE with the same full-sized `mlp`/markovian/no-`delta_cap` aux propagator
recipe; Phase 2 warm-starts from `nodeltacap_e2e`'s OWN already-trained
propagator, transplanting it onto this new latent space), varying three
things per the user's request:
1. `--attn-window 4` (back to `nodeltacap_e2e`'s original value, up from
   Section 24's `3`).
2. `--pool-bandwidth 5` (one wider than Section 24's `4`).
3. More epochs in both phases -- Phase 1 `--epochs 200` (double Section
   24's 100), Phase 2 `--epochs 136 --k-warmup-epochs 96 --k-mid-epochs 80`
   (double Section 24's 68/48/40, the same doubling convention used for
   every other "more epochs" request in this document, e.g. Section 13).

Tags: `bandedpool5_attn4_mlpaux_markovian_no_deltacap_200ep` (Phase 1),
`bandedpool5_attn4_from_nodeltacap_e2e_prop_200ep` (Phase 2 + Gate 3/4).
Launched immediately (no queue -- Section 24's chain had already fully
completed).

**Result: a substantial improvement over Section 24 on every metric that
matters.**

| Metric | Section 25 (bandwidth 5, `attn_window=4`, 200/136ep) | Section 24 (bandwidth 4, `attn_window=3`, 100/68ep) | `nodeltacap_e2e` (recommended best) |
|---|---|---|---|
| Phase 1 reconstruction | **0.003367** | 0.007037 | ~0.0002-0.0004 |
| `val_kmax_mse` | **0.051523** | 0.076687 | **0.006546** |
| `D_KY` | **21.786** | 19.810 | 21.99 |
| `\|D_KY-22\|` | **0.21** | 2.19 | **0.01** |
| `n_positive` | 12 | 10 | 12 |
| `lambda1` | 0.119 | 0.080 | 0.089 |
| DA skill | 2.88 | 2.81 | 3.43 |
| D3/D6/D7 | all sig, `p=0.0000` | all sig, `p=0.0000` | 2/3 sig |

Widening the band by one, restoring `attn_window=4`, and doubling both
phases' epoch counts closed roughly 90% of the `D_KY` gap to
`nodeltacap_e2e` (from `2.19` off-target down to `0.21`) and improved
accuracy by ~33%, while fully preserving the comprehensive D3/D6/D7
significance that made Section 24 interesting in the first place. This is
now the SECOND-BEST `\|D_KY-22\|` in the entire document (only
`nodeltacap_e2e` itself is closer), combined with the strongest structural
evidence of genuine local latent organization found anywhere. It still
trails `nodeltacap_e2e` by ~8x on raw accuracy, so the recommendation does
not change -- but the gap this specific architectural family has to close
to be competitive is now much smaller than Section 24 suggested, and this
looks like the more promising direction to push further (wider band/window
and more epochs helped substantially; it's not yet clear which of the
three changes mattered most, or whether pushing further in the same
direction would keep helping).

---

## 26. Spatial coherence as a training loss, first real test (user-directed, 2026-08-31)

Following the user's request ("turn the D7 diagnostic into a loss ... push
the model towards creating spatial coherence in addition to accurate
reconstruction and propagation"), implemented `spatial_coherence_loss`
(`ks_latent/training/losses.py`) -- a differentiable, per-batch analogue of
D7 that rewards the CURRENT (fixed, not permuted) latent index order for
having circular-band same-time correlation structure. Full design,
precedent, and a real bug caught and fixed along the way (an early version
whose true global optimum was pure decorrelation, not locality -- see the
function's docstring) are documented there. Wired in as
`Stage1TrainingConfig.w_spatial`/`spatial_bandwidth` (off by default) +
`--w-spatial`/`--spatial-bandwidth` CLI flags. **Explicit precedent flagged
before running anything real**: `RegConfig.lambda_z`, a closely related
existing mechanism in this codebase, is documented as collapsing the
latent even with a built-in anti-collapse counter-term -- this is
mechanically different (correlation-based, not value-proximity-based) and
verified not to share `lambda_z`'s exact degenerate optimum, but the
broader risk class (cheap "coherence" via locally redundant channels, at a
real cost to effective latent capacity) has not been ruled out. 317 tests
passing, live-smoke-tested.

**First real test, exactly as requested**: the EXACT original
`history3_fullprop_wvar005_tw16` recipe (Sections 2/3/5/9's `vit`-encoder,
`vit`-aux, `mode=history`, `n_history=3`, `delta_cap=0.5` recipe -- the
first-ever "special" AE in this document), with `--w-spatial 0.05` added
and nothing else changed, so the EXISTING `history3_fullprop_wvar005_tw16`
log/checkpoint serves directly as the control (no need to retrain it).
Tag `history3_fullprop_wvar005_tw16_wspatial005`. Pipeline:
1. Phase 1 (`--full-propagator --prop-delta-cap 0.5 --w-spatial 0.05`,
   otherwise identical CLI to the original).
2. A quick, propagator-free D7 check comparing this AE against the
   original control immediately after Phase 1 finishes (cheap -- no need
   to wait for Phase 2/Gate 3/4 to answer "did spatial coherence
   increase").
3. Phase 2: changed per user follow-up request -- rather than
   `--init-prop-checkpoint` warm-starting from this run's own `vit`-aux/
   history-mode Phase-1 propagator (matching the original recipe exactly),
   use a FRESH `--backbone mlp --mode markovian` propagator with no
   `--delta-cap` (all 3 are `train_stage2_patched.py`'s own defaults, so no
   flags needed beyond `--ae-checkpoint`) -- this pairs `w_spatial`'s
   effect on the AE with the single best-performing propagator recipe
   found in this entire document (Section 16's `nodeltacap_e2e`), rather
   than the older `vit`/history-mode one, and directly tests whether
   `w_spatial` helps or hurts THAT combination specifically. Same slow
   k-curriculum recipe otherwise (`--k-max 8 --k-warmup-epochs 48
   --k-mid 5 --k-mid-epochs 40 --w-varmatch 0.1`). Tag
   `history3_fullprop_wvar005_tw16_wspatial005_mlpmarkovian_nodeltacap`.
4. Gate 3/4 (Lyapunov spectrum is the critical one here -- checking
   whether `w_spatial=0.05` collapsed the propagator's ability to recover
   chaos, per the precedent above).

Launched immediately in parallel with Section 25's still-running Gate 3/4
(user-directed -- Gate 3/4 is CPU-only and Phase 1 training is MPS-only,
so no correctness conflict, only a possible wall-clock slowdown from
shared-machine contention, which was accepted).

**Phase 1 result**: `val_recon_final = 0.000594` -- barely worse than the
original (no-`w_spatial`) recipe's typical `~0.0002-0.0004`, i.e. no sign
of the feared accuracy cost at this weight.

**D7 quick-check result (the direct answer to "did the loss work"):**

| | D7 bandedness | D7 p-value |
|---|---|---|
| `w_spatial=0.05` (this run) | **0.4996** | **0.0000 (significant)** |
| control (`history3_fullprop_wvar005_tw16`, no `w_spatial`) | 0.2975 | 0.492 (not significant) |

**The loss works as intended**: adding `w_spatial=0.05` took this AE from
no detectable same-time spatial coherence at all to the strongest D7
bandedness score found anywhere in this document, with essentially no
reconstruction cost. This is the first clean, direct evidence that
`spatial_coherence_loss` does what it was built to do.

**A real bug caught by the user reviewing the log**: the first Stage-2
launch omitted `--epochs 68`, silently falling back to
`train_stage2_patched.py`'s own default of `20` -- with `--k-warmup-epochs
48` set, the k-curriculum never got past its first segment (stuck at
`k=3`, `best_val_kmax_mse=0.070976`) before the (too-short) run ended.
Fixed by relaunching Stage 2 via `--init-prop-checkpoint` on that same
(otherwise perfectly good) checkpoint, this time with the correct full
recipe (`--epochs 68 --k-max 8 --k-warmup-epochs 48 --k-mid 5
--k-mid-epochs 40 --w-varmatch 0.1`) so it actually reaches `k=8` -- tag
`history3_fullprop_wvar005_tw16_wspatial005_mlpmarkovian_nodeltacap_k8`.
Gate 3/4 chained to launch automatically once this finishes.

**Final result: `w_spatial=0.05` clears all three of the user's bars
(coherence, accuracy, chaos) -- the first case in this document where the
loss produced a genuinely competitive model, not just a coherence/accuracy
tradeoff.**

| Metric | `w_spatial=0.05` (k8, this run) | `nodeltacap_e2e` (recommended best, §16) |
|---|---|---|
| `val_kmax_mse` | 0.020858 | **0.006546** |
| `D_KY` | 21.351 | 21.99 |
| `\|D_KY-22\|` | 0.649 | **0.01** |
| `n_positive` | 12 | 12 |
| `lambda1` | 0.0965 | 0.0885 |
| DA calibration | 0.365 | 0.397 |
| DA skill | **3.82** | 3.43 |
| D3 | NS, `p=0.876` | NS, `p=0.191` |
| D6 | **sig, `p=0.0000`** | sig, `p=0.007` |
| D7 | **sig, `p=0.0000`** | sig, `p=0.018` |

`w_spatial=0.05` took this AE's D6/D7 significance from nothing detectable
(the control, `p=0.492`/not computed) to the strongest same-time and
temporal coherence seen on any `vit`-based AE in the document, WHILE
landing within a competitive range of `nodeltacap_e2e` on accuracy, chaos,
and DA skill (even edging it out slightly on DA skill, `3.82` vs `3.43`).
It does not beat the recommendation outright (worse accuracy, farther from
`D_KY=22`), but this is the cleanest positive evidence yet that
`spatial_coherence_loss` can induce genuine local structure without
paying the steep accuracy cost architectural approaches (Sections 17,
20-25) have consistently required -- the `RegConfig.lambda_z` collapse
precedent that motivated caution here did not materialize at this weight.
A natural next step (not yet run): sweep `w_spatial` a bit higher to see
whether coherence keeps improving before any collapse signature appears.

---

## 27. Rollout visualization: latent and physical space-time plots (user-directed, 2026-08-31)

Requested independent of any specific experiment's outcome ("if we see
[coherence, accuracy, chaos], I would like a way to visualize the
evolution of latent variables over time ... a 2d heatmap, with position on
the x axis and time on the y axis ... rollout performance and the whole
model's trajectory"). New script: `scripts/visualize_rollout.py`, adapted
from the reference project's plotting code
(`/Users/daltonjones/Documents/experiments/ks_latent/plots.py`'s
`plot_latent_rollout_spacetime`/`plot_rollout_spacetime`/
`plot_error_vs_lead_time`) but retargeted to this codebase's own model
interfaces (`load_autoencoder_checkpoint`/`load_propagator_checkpoint`,
`ae.encode`/`decode`, `LatentPropagator.rollout`/`rollout_history`).

Given any `--ae-checkpoint`/`--prop-checkpoint` pair and a real trajectory
from `--dataset`, seeds a free-running rollout from the trajectory's first
2 (or `n_history`, for history-mode propagators) true steps, propagates
autoregressively for `--rollout-steps`, decodes every predicted state, and
produces three figures (`docs/figures/*.png`):
1. **Latent Hovmoller** (`latent_hovmoller<tag>.png`): the requested
   position(index)-vs-time heatmap, as a truth/rollout/difference triptych,
   each latent coordinate z-scored by the TRUTH's own mean/std (not the
   rollout's) so the difference panel reads as "error in units of that
   coordinate's own climatological spread" rather than being swamped by a
   few loud, larger-scale coordinates.
2. **Physical Hovmoller** (`physical_hovmoller<tag>.png`): the classic KS
   space-time plot -- true trajectory, decoded model rollout, and their
   difference, position on x, time on y -- "the whole model's trajectory."
3. **Error growth** (`error_growth<tag>.png`): relative RMSE (physical
   space) vs. lead time, with the theoretical saturation level (`sqrt(2)`
   for a zero-mean bounded field) marked.

A diverging rollout (any non-finite state) is detected, masked blank past
the divergence point (not left to crash `pcolormesh` or saturate the
colour scale), and noted in the panel title and printed summary -- same
robustness convention the reference project's plotting code uses.

**Validated end-to-end against `nodeltacap_e2e`** (the recommended best
model, Section 18) at `--rollout-steps 200`: both Hovmoller plots show
the model tracking the true trajectory closely for roughly the first
75-100 time units (visually indistinguishable structure) before visibly
diverging -- exactly the qualitative signature expected of a genuinely
chaotic system with real short-term predictive skill, not a fluke.
Error growth rises smoothly from 0 and saturates oscillating around
`sqrt(2)~1.41`, the correct bounded-chaotic-attractor signature (matches
`plot_error_vs_lead_time`'s own documented expectation in the reference
project). Divergence-masking path also verified against a synthetic
propagator engineered to blow up to non-finite values partway through a
rollout (correctly detected and masked, no crash).

**Also validated against `maskedmlp_encoder_e2e`** (the checkpoint with the
strongest statistically-confirmed D3/D6/D7 coherence in the document): both
Hovmoller plots look qualitatively similar to `nodeltacap_e2e`'s (good
short-term tracking, then chaotic divergence) -- but the raw (natural-
order) latent Hovmoller does NOT show an obviously smooth/banded look to
the naked eye, just one persistently high-variance column (index ~13-16).
This is an important, honest observation: D3/D6/D7's statistical detection
of "neighboring channels correlate" is a subtler pattern than raw visual
smoothness and is often not obvious without reordering.

**`--reorder-by d7` added** (user-directed: "try the reordering by the D7
permutation"): reindexes the latent Hovmoller's columns (cosmetic only --
does not change the model's computation, unlike `PermutedAutoencoder`) by
a Fiedler permutation computed fresh from that SAME rollout's own
`z_true` via `same_time_coupling_diagnostic`. Tested on
`maskedmlp_encoder_e2e`: computing D7 from only this one 200-step rollout
(vs. the ~20-trajectory pooled sample used for the document's main D7
table) gives a weaker, marginal signal (bandedness `0.27`, `p=0.042` vs.
the full-sample `p=0.0000`) and correspondingly subtle visual improvement
-- some neighboring columns show shared coloring bands, but it is not a
dramatic transformation. Small-sample D7 estimates should be read with
this caveat in mind.

**Applied to Section 26's `w_spatial=0.05` result** once its Gate 3/4
confirmed it cleared all three of the user's bars (coherence, accuracy,
chaos -- see Section 26's final table): this is the most visually striking
latent Hovmoller in the document -- smooth, elongated coherent bands
spanning several ADJACENT indices are visible by eye in the raw (natural,
non-reordered) index order, unlike every other checkpoint tested (which
needed reordering, and even then only subtly, to show any visual
coherence). This is a meaningful qualitative difference from the
architectural approaches: `spatial_coherence_loss` shapes the loss with
respect to the FIXED index order directly, so it induces visible
smoothness in that same natural order, whereas D3/D6/D7 "significant but
architecturally-induced" structure (Sections 17, 24-25) only reveals
itself statistically. Figures:
`docs/figures/{latent,physical,error_growth}_hovmoller_wspatial005_k8.png`.

---

## 28. Pushing `w_spatial` further: 300/300 epochs, `k_max=40`, same `mlp`/markovian propagator in both phases (user-directed, 2026-08-31)

Following Section 26's clean positive result, user-directed follow-up:
`w_spatial=0.04` (slightly lower than Section 26's `0.05`), MUCH longer
training (`300` epochs each phase, vs. `100`/`68`), a MUCH longer rollout
curriculum (`k_max=40`, vs. `8`), and -- after an initial launch used the
wrong recipe and was corrected mid-flight -- the SAME `mlp`/markovian
propagator architecture in BOTH phases, with Phase 2 FINE-TUNING Phase 1's
own trained propagator (`--init-prop-checkpoint`) rather than training a
fresh one from scratch.

**A real launch mistake, caught and corrected before much time was lost**:
the first attempt used the ORIGINAL `history3_fullprop_wvar005_tw16`
recipe's aux settings (`--aux-backbone vit --mode history --n-history 3`,
matching Section 26's base case) with a fresh from-scratch `mlp`/markovian
Phase 2 -- i.e. Section 26's exact design, just scaled up. The user then
clarified mid-launch that they specifically wanted the SAME propagator
architecture and weights carried through both phases this time (`mlp`/
markovian trained during Phase 1 via `--aux-backbone mlp --mode markovian
--full-propagator`, matching `nodeltacap_e2e`'s own Phase 1 recipe --
Section 16 -- rather than the older `vit`/history-mode one), with Phase 2
fine-tuning that exact propagator rather than starting fresh. The
already-running (wrong-recipe) job was killed within ~1 minute of
starting (epoch 0 only) and relaunched correctly -- negligible wasted
compute.

**Recipe** (tag `fullprop_wvar005_tw16_mlpaux_markovian_no_deltacap_wspatial004_300ep`):
1. Phase 1: `nodeltacap_e2e`'s exact Phase 1 recipe (`--encoder vit
   --aux-backbone mlp --mode markovian --pos-encoding linear
   --attn-window 4 --token-window 16 --w-var 0.05 --full-propagator`, no
   `--prop-delta-cap`) plus `--w-spatial 0.04 --epochs 300`.
2. A quick D7 check (same pattern as Section 26) against the `nodeltacap_e2e`
   AE itself as control (the closest existing no-`w_spatial` comparison
   with this exact aux/mode combination).
3. Phase 2: `--init-prop-checkpoint` on THIS run's own Phase-1-trained
   `mlp`/markovian propagator (fine-tuning, not from-scratch), with a
   k-curriculum scaled up proportionally from the document's standard
   `k_max=8` recipe to `k_max=40` (5x): `--epochs 300 --k-max 40
   --k-warmup-epochs 210 --k-mid 25 --k-mid-epochs 175 --w-varmatch 0.1`
   (`k_warmup_epochs`/`k_mid`/`k_mid_epochs` all scaled by the same 5x
   factor relative to `k_max`, preserving the standard recipe's relative
   curriculum shape -- see this section's git-blame-adjacent commit for
   the exact ratios: `k_warmup_epochs/epochs~0.70`,
   `k_mid_epochs/k_warmup_epochs~0.83`, `k_mid/k_max~0.625`, all matched).
   Tag `..._wspatial004_300ep_k40`.
4. Gate 3/4 on the result.

Launched (PID confirmed running, `mlp`-aux Phase 1 training at ~5.6s/epoch
-- notably faster than the `vit`-aux recipe's ~10s/epoch, so despite 3x the
epoch count this leg should finish in a comparable or shorter wall-clock
time).

**Phase 1 result: excellent** -- `val_recon_final = 0.000131`, the best
reconstruction in this entire document (better than `nodeltacap_e2e`'s own
Phase 1). D7 quick-check: bandedness `0.5258`, `p=0.0000` -- also the
strongest D7 result of the whole `w_spatial` series, beating even Section
26's `0.4996`.

**Phase 2: killed by the user, a real and informative anomaly.** Fine-
tuning this run's own Phase-1-trained `mlp`/markovian propagator via
`--init-prop-checkpoint` (the SAME warm-start mechanism `nodeltacap_e2e`'s
own Stage 2 used successfully) got stuck: `val_kmax_mse` started at `1.27`
at epoch 0 and never dropped meaningfully below `~1.0` through `k=9`,
essentially the no-skill baseline the entire time. Direct comparison:
`nodeltacap_e2e`'s own Stage 2 (identical warm-start mechanism, same
backbone/mode, no `w_spatial`) started at `val_kmax_mse=0.070` at epoch 0
and had already reached `~0.007` by `k=7-8` -- roughly 100x better at the
same point. This is NOT a training-schedule artifact (both start at epoch
0 with the same peak LR) -- something about the specific Phase-1-trained
propagator this run produced (jointly trained with `w_spatial=0.04` for
300 epochs) is broken for Stage 2 fine-tuning, DESPITE the paired encoder
itself looking excellent by every other measure (reconstruction, D7).
Killed before wasting further compute or reaching Gate 3/4 on a result
that was clearly not going to be meaningful. Not yet root-caused -- a real
open question is whether `w_spatial` training shapes the Phase-1
propagator's OWN weights (not just the encoder) into some state that
doesn't transfer well to Stage 2's fine-tuning objective, or whether this
is a one-off. Section 29 (next) tests a different combination that
sidesteps this specific failure mode by not fine-tuning this propagator at
all.

---

## 29. Same scale-up, but back to `vit`/history-mode Phase 1 with a fresh (not fine-tuned) `mlp`/markovian Phase 2 (user-directed, 2026-08-31)

Direct response to Section 28's Phase 2 anomaly: "we will run the same
training as 28 but will use the same vit backbone for the aux propagator
as here [the original `history3_fullprop_wvar005_tw16` recipe]. Then we
will replace this with the markovian mlp in stage 2." This is exactly
Section 26's design (`vit`/history-mode Phase 1, fresh from-scratch
`mlp`/markovian Phase 2 -- NOT fine-tuned, since the two backbones are
architecturally incompatible for `--init-prop-checkpoint` warm-starting
anyway) scaled up to Section 28's longer schedule and lower `w_spatial`.
This also sidesteps Section 28's specific failure mode by construction:
there is no propagator being fine-tuned across phases here at all, so a
bad Phase-1 propagator (if that's what happened in Section 28) can't
poison Phase 2.

**Recipe** (tag `history3_fullprop_wvar005_tw16_wspatial004_300ep`):
1. Phase 1: EXACT original recipe (`--encoder vit --aux-backbone vit
   --mode history --n-history 3 --pos-encoding linear --attn-window 4
   --token-window 16 --aux-n-tokens 44 --aux-token-d-model 64 --w-var 0.05
   --full-propagator --prop-delta-cap 0.5`) plus `--w-spatial 0.04
   --epochs 300`.
2. Quick D7 check against the original (no-`w_spatial`) control.
3. Phase 2: FRESH `--backbone mlp --mode markovian` (no
   `--init-prop-checkpoint` -- a new propagator, matching Section 26's
   approach). **Changed mid-flight per user request**: `k_max` lowered
   from `40` to `20` (`--k-mid` rescaled proportionally from `25` to `13`
   to keep it under the new `k_max`; `--epochs 300
   --k-warmup-epochs 210 --k-mid-epochs 175` unchanged, since those are
   epoch counts, not k-values). Tag `..._mlpmarkovian_k20`.
4. Gate 3/4.

**How the mid-flight change was made without losing progress**: Phase 1
was already at epoch ~90/300 when the request came in. Rather than
restarting, killed only the wrapper pipeline SCRIPT (a plain `kill`, no
signal propagation to children since it was launched via `nohup`) --
verified the actual Phase 1 training process (a further-descendant `mamba
run` -> `python` chain) kept running completely undisturbed, since bash/
zsh does not automatically signal a foreground child when its own parent
script dies. A second script was launched that waits on that same
still-running training process's PID directly, then proceeds with the
corrected (`k_max=20`) Phase 2 once it actually finishes -- zero wasted
compute, zero interruption to the in-progress epoch.

**Then the user's machine crashed unexpectedly at Phase 1 epoch ~190/300**,
killing every process (the `nohup`/detached-process protection above
only helps against a wrapper SCRIPT dying, not against the whole machine
going down) with no checkpoint to recover from -- this codebase only wrote
checkpoints at the very END of a training run, so all ~190 epochs of
progress were lost.

**New capability built in direct response**: `--checkpoint-every N`
(added to both `train_stage1_patched.py` and `train_stage2_patched.py`,
default `0` = off). `train_stage1`/`train_stage2`
(`ks_latent/training/loops.py`) each gained an `on_epoch_end` callback
hook, called after every `N`-th completed epoch; the scripts use it to
overwrite a SINGLE rolling `*_checkpoint.pt` file (not one file per
checkpoint, to bound disk usage) in the same format as the final
checkpoint. Deliberately NOT a full resume mechanism (no optimizer/
scheduler state, so resuming restarts the LR schedule from scratch rather
than continuing it exactly) -- just enough to avoid losing an entire run
to a crash. Verified live on both scripts (smoke profile,
`--checkpoint-every 1`): checkpoint file written and overwritten each
epoch as expected, 317 tests still passing.

**Relaunched from scratch** with parameters also revised per the user's
follow-up request: `--w-spatial 0.05` (back from `0.04`), `--epochs 200`
(down from `300`, both phases), `--checkpoint-every 10` (new), tag
`history3_fullprop_wvar005_tw16_wspatial005_200ep`. Phase 2's k-curriculum
epoch counts rescaled proportionally for the shorter schedule
(`--k-warmup-epochs 140 --k-mid-epochs 117`, from `210`/`175` at `300`
epochs -- both scaled by `200/300`; `--k-max 20 --k-mid 13` unchanged,
since those are k-values, not epoch counts). Confirmed running.

**Automatic cleanup added** (user-directed follow-up: "write code to delete
unused checkpoints after these runs complete, just keep the best one"):
both scripts now delete their own rolling `*_checkpoint.pt` file(s) right
after writing the real final checkpoint -- only reached on a normal
(non-crashed) exit, which is exactly when the periodic file is safely
superseded and safe to remove. Verified live on both scripts (smoke
profile): checkpoint written each epoch, then removed with a printed
`removed superseded mid-training checkpoint ...` line once the final
checkpoint exists. 317 tests still passing. Note: this run's already-in-
progress Phase 1 process loaded the OLDER code (without the cleanup)
before this change landed, so its own periodic checkpoint will need a
manual `rm` once it finishes -- Phase 2, launched fresh afterward by the
same pipeline script, will pick up the new cleanup code automatically.

**Phase 1 result: excellent again.** `val_recon_final = 0.000230`. D7
quick-check: bandedness `0.5147`, `p=0.0000` -- strongly significant,
essentially matching Section 26's original `0.05` result (`0.4996`) and
confirming the loss's effect reproduces cleanly at this recipe. AE
checkpoint: `stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial005_200ep.pt`.

**Phase 2 killed by the user before finishing** (at epoch 75/200,
`val_kmax_mse~0.47`, not looking promising -- though not necessarily
broken, just not yet evaluated to completion) in favor of a bigger,
much-longer run -- see Section 30.

---

## 30. Same AE, much longer Stage 2 (1000 epochs) with a 1.2x-larger `mlp` propagator (user-directed, 2026-08-31)

"kill the stage 2 training running right now. I want it to run for 1000
epochs. same AE setup. Also give the markovian mlp 1.2 times as many
parameters." Reuses Section 29's Phase 1 AE unchanged (no retraining) --
only the Phase 2 propagator's SIZE and TRAINING LENGTH change.

**New capability added**: `--hidden`/`--n-blocks` CLI flags for
`--backbone mlp` in `train_stage2_patched.py` (previously there was no way
to override `PropagatorConfig`'s `hidden=128`/`n_blocks=3` defaults from
the command line at all). Computed the exact scaling needed: with
`n_blocks=3` and `d_latent=44` held fixed, the default `hidden=128`
propagator has `111,532` parameters; `hidden=141` gives `133,853` --
almost exactly `1.2x` (`ratio=1.2001`), found by direct construction +
parameter count rather than an analytic approximation (the residual-MLP
body's parameter count is dominated by the `2 * n_blocks * hidden^2` term
from the blocks' two square `Linear(hidden,hidden)` layers, so the right
`hidden` isn't a simple `sqrt(1.2)` scaling of the base width once the
fixed `input_proj`/`output_proj`/`LayerNorm` terms are accounted for).
317 tests passing, smoke-tested live.

**Recipe**: `--ae-checkpoint <Section 29's AE> --backbone mlp --mode
markovian --hidden 141 --epochs 1000 --k-max 20 --k-warmup-epochs 700
--k-mid 13 --k-mid-epochs 585 --w-varmatch 0.1 --checkpoint-every 10`
(k-curriculum epoch counts scaled by `1000/200 = 5x` from Section 29's
`140`/`117`, preserving the same relative curriculum shape; `k-max`/
`k-mid` unchanged since those are k-values, not epoch counts). Tag
`history3_fullprop_wvar005_tw16_wspatial005_200ep_mlpmarkovian_hidden141_1000ep`.
Confirmed running (`~0.9-1.0s/epoch` at `k=2`, will grow as `k` ramps up
through the curriculum).

**Killed by the user at epoch 251/1000** (`k=7`, `val_kmax_mse~0.62`,
looking stuck relative to how fast propagator training on this general
recipe usually goes) after noticing a puzzling contrast with Section 26's
`_k8` result (which reached `val=0.021` by its own epoch 68) -- see
Section 31 for the follow-up isolation test this prompted, and the actual
resolution (a warm-start artifact, not a property of this run). A rolling
checkpoint from epoch 249 was preserved
(`..._hidden141_1000ep_checkpoint.pt`) if this configuration is revisited
later.

---

## 31. Isolating whether the fast Section 26 convergence was about the propagator's weights or the AE's latent space (user-directed, 2026-08-31)

The apparent mystery -- Section 26's `_k8` run reaching `val_kmax_mse=0.02`
while Section 30 was stuck around `0.6` at a comparable point -- turned out
to have a mundane explanation on inspection: `_k8`'s log opens with
`--init-prop-checkpoint: loaded architecture from
...mlpmarkovian_nodeltacap.pt`, meaning it was NOT trained from scratch --
it continued from an already-trained propagator (Section 26's own earlier,
`--epochs`-bug-truncated 20-epoch run, `best_val_kmax_mse=0.070976`) rather
than a fresh random initialization. Section 30, by contrast, has no
`--init-prop-checkpoint` line at all and starts at the genuine cold-start
value (`val_kmax_mse=1.004` at epoch 0). Once that's accounted for, the
original "mystery" mostly dissolves -- but a real, well-posed follow-up
question remains: does a propagator's LEARNED WEIGHTS transfer usefully
across different AEs' latent spaces, or is the earlier fast convergence
specific to the exact AE it was trained alongside? User-directed test:
"train another stage 2 run the same as [Section 30's 1000-epoch run], but
initialize the propagator from the model from [the original 100-epoch-AE,
20-epoch, `best_val=0.071` checkpoint]. this should isolate if it's
something to do with the latent space or with the model/initialization."

**Setup**: `--ae-checkpoint` Section 30's AE (200-epoch, `w_spatial=0.05`)
+ `--init-prop-checkpoint` the ORIGINAL 100-epoch-AE-trained, 20-epoch
propagator checkpoint, same k-curriculum/epoch budget as Section 30
(`--epochs 1000 --k-max 20 --k-warmup-epochs 700 --k-mid 13
--k-mid-epochs 585 --w-varmatch 0.1 --checkpoint-every 10`). One
consequence worth flagging explicitly: `--init-prop-checkpoint` loads the
EXACT architecture from the checkpoint, so this run necessarily uses
`hidden=128` (the original default), not Section 30's `hidden=141` -- the
only way to warm-start from those exact weights at all, and actually the
correct choice for a clean isolation test (same propagator, only the AE
differs). Tag
`history3_fullprop_wvar005_tw16_wspatial005_200ep_initFrom100epAEprop_1000ep`.

**Result (killed by the user at epoch 105/1000, conclusive enough)**: the
warm-started weights never recovered. Trajectory: `val~0.8` at epoch 18,
still `val~0.67-0.68` at epoch 101-105 -- barely moving over 90+ further
epochs, and nowhere near the `0.071` this exact propagator had already
reached on the AE it was originally trained with. Given 105 epochs is
already well past the point where the original 20-epoch run had converged
to `0.071`, and this run is still 10x worse, the weights are not just
"slow to re-adapt" -- they are essentially starting over.

**Conclusion (user's own read, and the evidence supports it): it's the
latent space, not the propagator's weights/initialization.** The fast
Section 26 `_k8` convergence was specific to the AE/propagator pairing --
a propagator's learned one-step map is tuned to its particular AE's latent
coordinate system (basis, scaling, whatever `w_spatial`/training-recipe-
specific structure that AE has) closely enough that it does NOT transfer
to a different AE's latent space, even one built from a very similar
recipe (same architecture family, same `w_spatial=0.05`, just 100 vs 200
Phase-1 epochs). This rules out "any reasonable initialization gets you
most of the way there" as the explanation, and points back to something
about the specific 100-epoch AE's latent space that made it unusually easy
for an `mlp`/markovian propagator to fit -- still not identified (worth a
future look at whether that AE's D3/D6/D7 structure, or something else
about it, is the actual reason), but now correctly scoped as a property of
THAT AE rather than a generically-transferable good initialization.

---

## 32. Fine-tuning the AE jointly with its already-converged propagator, then reusing that same propagator for a longer Stage 2 (user-directed, 2026-08-31)

Follow-up on Section 31's finding: "I would like to fine tune the
autoencoder from [Section 26's `_k8` run] for 200 epochs (same training as
stage 1 for that run, just initialize with this and the aux propagator
from this run.) Then I want to take that, and use the model from [the
same `_k8` run] to initialize another stage 2 ... extend to k=16." The
idea: since Section 26's `mlp`/markovian propagator already converged
very well specifically on this AE (`val=0.021`), does JOINTLY fine-tuning
the AE further -- guided by that already-good propagator via Stage 1's
`L_pred` term, rather than a small/fresh aux -- produce an even more
mutually-adapted AE, which the SAME propagator can then be pushed further
on via a longer Stage 2 (`k_max` doubled to `16`)?

**New capability required and added**: `--init-ae-checkpoint`/
`--init-aux-checkpoint` for `train_stage1_patched.py` (previously, Stage 1
could only ever build a fresh, randomly-initialized AE and aux propagator
-- there was no way to CONTINUE joint training from existing weights at
all). Implementation notes:
- `--init-ae-checkpoint` loads via the existing `load_autoencoder_checkpoint`,
  overriding whatever `--encoder`/`--pool`/etc. flags would otherwise have
  built (those become inert once this is given, same convention
  `train_stage2_patched.py --init-prop-checkpoint` already established).
- `--init-aux-checkpoint` loads ANY Stage-2-format propagator checkpoint
  directly as the aux. This works with NO conversion needed because
  `AuxPropagator` is architecturally identical to `LatentPropagator` for
  the same effective config (`AuxPropagator.__init__` just converts its
  own config via `aux_cfg_to_propagator_cfg` and delegates to
  `LatentPropagator.__init__`) -- a Stage-2 `LatentPropagator`'s
  `state_dict()` loads directly into the constructed object, and
  `train_stage1`'s loop only ever calls the methods both classes share
  (`.mode`, `.cfg.n_history`, `.rollout`/`.rollout_history`).
- A real correctness bug caught and fixed while wiring this in: the
  periodic-checkpoint and final-save code paths both saved
  `"encoder_kind": args.encoder` literally -- correct when the AE is built
  fresh from that same flag, but WRONG once `--init-ae-checkpoint`
  overrides the actual architecture (if `--encoder` wasn't ALSO passed to
  match, the saved checkpoint's dispatch tag would silently point at the
  wrong class for its own `state_dict`, corrupting the checkpoint). Fixed
  by reusing the SOURCE checkpoint's own recorded `encoder_kind` instead of
  re-deriving it from the CLI flag. Also fixed the same "wrong architecture
  label in the startup print line" cosmetic issue `--init-prop-checkpoint`
  already had fixed on the Stage-2 side. Verified round-trip correct
  (smoke-scale): saved checkpoint's `encoder_kind='vit'` matches the
  actually-loaded `KSAutoencoderViT`, propagator config `backbone='mlp'
  mode='markovian'` matches the loaded Stage-2 checkpoint. 317 tests
  passing.

**A mid-flight correction on which "aux propagator from this run" was
meant**: the first launch read "initialize with this and the aux
propagator from this run" as referring to `_k8`'s STAGE 2 propagator
(`mlp`/markovian) -- and indeed ran ~2x faster per epoch (`~5s` vs
`~9.9s`) as a direct, confirmable consequence (a dense `mlp`/markovian
body is far cheaper per step than an attention-based `vit`/history-mode
one). The user then clarified: they meant the ORIGINAL `vit`/history-mode
aux propagator that was jointly trained ALONGSIDE this AE during ITS OWN
Stage 1 run (`stage1_history3_fullprop_wvar005_tw16_wspatial005.log`,
saved via that run's own `--full-propagator` mechanism as
`stage1_prop_full_history3_fullprop_wvar005_tw16_wspatial005.pt`) -- NOT
the unrelated Stage-2-trained `mlp` one. Confirmed directly from that
checkpoint's saved `aux_config`: `backbone='vit', mode='history',
n_history=3`, matching the original recipe exactly. The first (wrong-aux)
run was killed within ~3 minutes (epoch ~35/200) and relaunched correctly
-- negligible wasted compute.

**Recipe** (corrected, tag `history3_wspatial005_finetune_from_vitaux_200ep`):
1. Stage 1 continuation: `--encoder vit --pos-encoding linear
   --attn-window 4 --token-window 16 --w-var 0.05 --w-spatial 0.05`
   (matching the original AE's training recipe -- these loss weights still
   apply even though the architecture flags they'd otherwise also govern
   are inert here) + `--init-ae-checkpoint <Section 26's AE>
   --init-aux-checkpoint <that SAME run's own vit/history aux propagator>
   --epochs 200 --checkpoint-every 10`. Confirmed running at `~9.9s/epoch`
   (back to `vit`-aux speed, as expected).
2. Quick D7 check: fine-tuned AE vs. the pre-fine-tune AE.
3. Stage 2 (tag `..._initFromK8prop_k16`): `--ae-checkpoint` the NEWLY
   fine-tuned AE from step 1, `--init-prop-checkpoint` Section 26's `_k8`
   `mlp`/markovian propagator (reused directly, unchanged -- per the
   user's explicit instruction, this part of the plan was always meant to
   use the `mlp` propagator, just not for the Stage 1 fine-tuning step),
   `--epochs 200 --k-max 16` (doubled from the original `_k8`'s `8`) with
   the k-curriculum epoch counts scaled by the same ratios as the original
   recipe (`--k-warmup-epochs 141 --k-mid 10 --k-mid-epochs 118`, from
   `48`/`5`/`40` at `k_max=8`/`68` total epochs).
4. Gate 3/4.

**Stage 2 (step 3) also converged worse than the original `_k8` run** --
`val_kmax_mse~0.27` around epoch 140-144/200 at `k=15-16`, versus the
original `_k8`'s `val=0.021` at `k=8`/epoch 47-68 -- despite warm-starting
from that exact same well-converged propagator, now paired with a version
of the SAME AE that has only been further fine-tuned (jointly, with its
own original aux), not swapped for a different one. User's hypothesis:
"I wonder if it's just an initialization sort of thing for the mlp" --
i.e. maybe the warm-started weights are actively a BAD starting point for
this specific (slightly shifted, post-fine-tuning) latent basis, rather
than the latent space itself having become harder to fit in general.

**Clean test queued**: another Stage 2 run on this EXACT SAME fine-tuned
AE, same `--backbone mlp --mode markovian` and k-curriculum
(`k_max=16`), but a FRESH random init instead of `--init-prop-checkpoint`
-- tag `..._freshmlp_k16`. If fresh-init converges well (like the original
`_k8` did), that points to the WARM-STARTED WEIGHTS being a bad match for
this specific fine-tuned basis (an initialization problem). If fresh-init
also struggles, that points back to the fine-tuned latent space itself
having gotten harder for an `mlp`/markovian propagator to fit in general,
regardless of starting point -- a meaningfully different, more concerning
finding, since it would mean fine-tuning the AE (even guided by its own
original aux) made the resulting latent space WORSE for this propagator
family, not just differently-shaped. Queued to launch automatically once
the warm-started run's training finishes.

**Final results, all four pieces landed:**

- **Stage 1 fine-tune**: `val_recon_final = 0.000154` (excellent, even
  better than the pre-fine-tune AE's own reconstruction).
- **D7 quick-check**: fine-tuned AE bandedness `0.5161, p=0.0000` vs.
  pre-fine-tune `0.4996, p=0.0000` -- both strongly significant, fine-
  tuning slightly increased it further.
- **Stage 2 (warm-started from the `_k8` propagator, `k_max=16`)**:
  `best_val_kmax_mse = 0.224951` -- far worse than the original `_k8`'s
  `0.021` at half the `k_max`.
- **Fresh-init comparison (same AE, same recipe, no warm start)**:
  `best_val_kmax_mse = 0.227688` -- **essentially IDENTICAL to the warm-
  started result** (`0.2250` vs `0.2277`, well within run-to-run noise).
- **Gate 3/4 on the warm-started run**: `D_KY=20.28, n_positive=11,
  lambda1=0.080` (still solidly chaotic, just less accurate); DA
  `rmse_da=0.364, skill=2.75`; **D3 `p=0.962` (not significant)**; **D4:
  genuine (approximately) equivariant translation representation found**
  -- only the SECOND checkpoint in this entire document to earn that
  verdict (the first was Section 13's 200-epoch `vit`-aux AE); D6/D7 both
  `p=0.0000` (strongly significant).

**This settles the "initialization vs. latent space" question for this
specific `k_max=16` recipe: it's the latent space, not the initialization.**
Warm-started and fresh-init propagators land at essentially the same
(mediocre) accuracy, ruling out "these particular pretrained weights are a
bad fit" as the explanation. Combined with the excellent D4/D6/D7
structural results, there's a real, notable TENSION here worth stating
plainly: **this fine-tuned AE has the best/most comprehensive structural
diagnostics of any checkpoint in the entire document (D4 AND D6 AND D7 all
positive -- no other checkpoint achieves all three) yet is measurably
HARDER for an `mlp`/markovian propagator to fit accurately than several
much-less-structured AEs.** This raises a real, open question worth
flagging for future investigation: does strong spatial/temporal coherence
somehow make the propagator's job intrinsically harder (e.g. more
entangled cross-channel information that a simple per-step markovian map
struggles to disentangle), or is this specific to something about the
`k_max=16` recipe/this particular fine-tuning path? Section 33's
systematic multi-seed sweep (queued, using the STANDARD `k_max=8` recipe
instead) will help separate "is this AE harder in general" from "is
`k_max=16` specifically harder."

---

## 33. Systematic multi-seed sweep: is the original `_k8` convergence reliable, or was it luck? (user-directed, 2026-08-31)

"can we run like 10 random init markovian mlp stage 2 processes for both
[the fine-tuned AE] and [the original AE]. I want to see if all of them
converge in the second case, and all don't converge in the first case. or
if we just got lucky on the first run or something. set up the
infrastructure to test this systematically."

**Infrastructure built**:
- `scripts/summarize_multiseed.py` (new): parses `best_val_kmax_mse = ...`
  out of a set of `stage2_multiseed_{group}_seed{N}.log` files (matched by
  a glob `--pattern`), groups by `{group}`, and reports per-group min/
  median/mean/max plus a convergence count against `--threshold` (default
  `0.05` -- comfortably above every well-converged result and below every
  stuck/collapsed one seen anywhere in this document). Handles a
  crashed/incomplete run gracefully (reports `MISSING/INCOMPLETE` rather
  than crashing the summary itself). Verified against synthetic log files
  before relying on it for real results.
- A sweep script looping `--seed 0` through `9` for EACH of the two AEs
  (`stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial005.pt`,
  the original; `stage1_ae_patched_full_history3_wspatial005_finetune_from_vitaux_200ep.pt`,
  the fine-tuned one from Section 32), all FRESH random inits (no
  `--init-prop-checkpoint` anywhere in this sweep -- deliberately not
  reusing the already-running single fresh-init test from Section 32,
  which used a different, longer `k_max=16`/`200`-epoch recipe), all using
  the STANDARDIZED recipe matching the original `_k8` run exactly
  (`--backbone mlp --mode markovian --epochs 68 --k-max 8
  --k-warmup-epochs 48 --k-mid 5 --k-mid-epochs 40 --w-varmatch 0.1`) so
  every one of the 20 runs is directly comparable to that historical
  result and to each other. Runs sequentially (not parallel) to avoid MPS
  contention and keep timing comparisons clean; 20 runs x ~68 epochs x
  ~1.3-2s/epoch is expected to take on the order of 30-40 minutes total.
  Queued behind the currently-running Section 32 fresh-init test to avoid
  overlapping MPS usage.

**Design note on why the recipe is `k_max=8`/`68` epochs, not the `k_max=16`/
`200` epochs of Section 32's own tests**: this deliberately reproduces the
ORIGINAL `_k8` run's exact recipe so the question "does the ORIGINAL
convergence reproduce reliably across seeds" is answered on its own exact
terms, not conflated with the separate (already-suspected-relevant)
question of whether the longer/`k=16` recipe itself is harder. If the
`original` group's 10 seeds mostly converge here, that directly confirms
`_k8`'s result was not a fluke, on a fair, matched footing.

**Killed early, per direct user instruction, once the question was already
answered**: "looking at the preliminary results from the multiseed test,
it seems clear that the prior run that created the autoencoder for [the
`_k8` run] has latent states that are inherently more predictable... kill
the multiseed test, we've answered the question." Only the `original`
group had progressed far enough to produce results (3/10 seeds complete,
a 4th in progress when killed; the `finetuned` group's queue position
meant it never started):

| seed | `best_val_kmax_mse` (original AE) |
|---|---|
| 0 | 0.023200 |
| 1 | 0.022781 |
| 2 | 0.022678 |

All three land within a tight `0.0227-0.0232` band, matching the
historical `_k8` result almost exactly -- conclusive enough on its own
that the original AE's fast convergence is a reliable property of that
run, not a lucky seed. This directly motivated Section 34: since it's now
established that the two latent spaces genuinely differ in how
predictable they are for an `mlp`/markovian propagator, the next question
is *why*, investigated there with training-free diagnostics that don't
depend on any specific propagator's fitting/init luck at all.

---

## 34. What makes the original AE's latent space more predictable? A near-collapsed channel, not a global smoothness difference (user-directed, 2026-08-31)

"the prior run that created the autoencoder for [the `_k8` run] has latent
states that are inherently more predictable. The question is what is
different about this run versus the fine tuned auto encoder... look at
the latent space and try to figure out what makes that run more
predictable than the other and how we can replicate it."

**Method**: `scripts/analyze_latent_predictability.py` (new). Encodes the
SAME real trajectories (`artifacts/datasets/stage1_trajectories_dtsnap1.h5`,
60 runs x 251 steps) with each AE and compares five completely
TRAINING-FREE measures -- no propagator fitting, no gradient descent, no
random init, so nothing here can be a fitting-luck artifact:

1. Relative one-step latent jump size (`||z(t+1)-z(t)||` vs. attractor
   spread).
2. Closed-form ridge-regression linear map residual, fit SEPARATELY at
   horizons `k in {1,2,4,8,16}` (not an iterated one-step map -- a direct
   per-horizon fit, matching how Stage 2's curriculum actually trains on
   k-step targets).
3. KNN local stretching factor: for many near-neighbor pairs `z(t)_i,
   z(t)_j`, the ratio `||z(t+1)_i - z(t+1)_j|| / ||z(t)_i - z(t)_j||` -- a
   local Lipschitz-like estimate of the true one-step map.
4. Decoder Jacobian operator norm `||d decode(z)/dz||` at sampled
   attractor points, via power iteration on `J^T J` using
   `torch.autograd.functional.jvp`/`vjp` (needed
   `with sdpa_kernel(SDPBackend.MATH):` around the whole per-sample
   power-iteration body -- the same CPU double-backward-through-attention
   limitation already hit once before in `ks_latent/analysis/lyapunov.py`;
   first draft only wrapped the inner loop and left the final post-loop
   `jvp` call outside the `with` block, which still crashed -- fixed by
   widening the `with` scope to cover the entire per-sample computation).
5. **Latent covariance eigenspectrum** (added after 1-4 came back
   inconclusive): top/min eigenvalue, condition number, and effective
   rank (participation ratio) of the covariance of all encoded states.

**Results, original vs. fine-tuned AE**:

| diagnostic | original (fast, `val~0.023`) | fine-tuned (slow, `val~0.22`) |
|---|---|---|
| relative one-step jump size | 1.1837 | 1.2395 |
| KNN stretch median / p90 | 1.0139 / 1.1529 | 1.0185 / 1.1694 |
| decoder Jacobian norm | 11.65 | 13.65 |
| linear-map rel. MSE, k=1 | 0.9509 | 0.9641 |
| linear-map rel. MSE, k=2 | 0.9358 | 0.9373 |
| linear-map rel. MSE, k=4 | 0.8420 | 0.8372 |
| linear-map rel. MSE, k=8 | 0.6628 | 0.6510 |
| linear-map rel. MSE, k=16 | 0.4985 | 0.4934 |
| **effective rank** | 10.11 / 44 | 10.34 / 44 |
| **top eigenvalue** | 6.115 | 5.881 |
| **min eigenvalue** | **2.04e-5** | **2.42e-6** |
| **condition number** | **3.0e5** | **2.4e6** |

**Diagnostics 1-4 are a dead end**: every one of them is within ~15% of
the other AE, several even *favor* the fine-tuned AE slightly (k=4/8/16
linear residual). None of these training-free, scale-robust measures of
the true one-step/multi-step dynamics comes close to explaining a ~10x
gap in actual `mlp`-propagator convergence quality. This is itself a
useful negative result: the divergence is NOT a simple smoothness/
local-Lipschitz/global-linear-predictability property of the underlying
dynamics.

**Diagnostic 5 finds the real difference**: both AEs have nearly
identical *effective* rank (~10/44 dimensions carry real variance; the
top-15 eigenvalues of the two spectra track each other closely), but the
fine-tuned AE's SINGLE weakest channel has collapsed to `2.42e-6` --
roughly **8.5x smaller** than the original AE's weakest channel
(`2.04e-5`), giving an ~8x worse overall condition number (`2.4e6` vs.
`3.0e5`). This is invisible to the scale-robust diagnostics above (ridge
regression's global regularization and KNN's raw distances don't care how
close to zero one channel's variance gets).

**Correction (see Section 35): this is not quite the right mechanism.**
`Stage2TrainingConfig.w_varmatch` (`ks_latent/training/losses.py`'s
`decorr_var_loss`, `l_var = ((diag - 1.0) ** 2).mean()`) does NOT match
the propagator's predicted variance to the AE's true per-channel
variance -- it matches every channel to a HARDCODED target of `1.0`,
regardless of that channel's actual scale. For a channel whose true
variance is `2.42e-6`, this loss actively pulls the propagator's
prediction AWAY from the correct near-zero value and toward `1.0` --  a
direct conflict with the primary latent-matching loss on exactly the
channel that's collapsed, not merely "a numerically delicate near-zero
target to chase." See Section 35 for the full mechanism and the resulting
options.

**Is this a progressive effect of fine-tuning, not specific to the `vit`
aux?** A leftover crash-insurance checkpoint survived from the FIRST
(wrong-aux, later killed per Section 32's correction) fine-tuning attempt
-- `artifacts/stage1_ae_patched_full_history3_wspatial005_finetune_from_k8prop_200ep_checkpoint.pt`,
confirmed via its own metadata to be epoch 69 of a `mode=markovian,
backbone=mlp` aux fine-tune (not the `vit`/history aux used in the run
that actually completed and produced the AE analyzed above). Checking its
latent covariance:

| checkpoint | min eigenvalue | condition # |
|---|---|---|
| original (epoch 0 of fine-tuning) | 2.04e-5 | 3.0e5 |
| wrong-aux fine-tune, epoch 69 (killed) | 1.45e-5 | 4.1e5 |
| correct-aux (`vit`) fine-tune, epoch 200 (completed) | 2.42e-6 | 2.4e6 |

The min eigenvalue barely moves through epoch 69 (`2.0e-5 -> 1.5e-5`) but
then drops by another ~6x between epoch 69 and 200. This is consistent
with **progressive, roughly monotonic channel collapse over the course of
extended joint fine-tuning**, independent of which aux backbone drives
it (seen starting under the `mlp` aux before that run was killed, and
completing under the `vit` aux) -- i.e. this looks like a generic
consequence of continuing joint AE+aux optimization for a long time past
where reconstruction/structural quality (D4/D6/D7, all BETTER on the
fine-tuned AE) has already converged, not something specific to the `vit`
aux itself.

**Note on the standing no-banded-latent-regularizer guidance**: this is a
genuine, organically-emerging latent collapse, found WITHOUT
`RegConfig.lambda_z` enabled anywhere in this pipeline -- consistent with
that regularizer being flagged as actively causing collapse, but showing
here that prolonged joint fine-tuning alone, with no explicit
anti-collapse pressure at all, can produce the same failure mode. The fix
proposed below is deliberately NOT a return to that regularizer.

**How to replicate the good property (per the user's explicit ask)**:
since the collapse is progressive and the good structural diagnostics
(D4/D6/D7) were ALREADY strong on the original 100-epoch AE before any
extra fine-tuning, the straightforward fix is **early stopping on a
latent-health signal, not a fixed epoch budget**:
- Track the minimum per-channel latent variance (or, equivalently, the
  covariance's minimum eigenvalue / condition number) as a monitored
  quantity during Stage-1 (fine-tuning or otherwise) training, the same
  way `val_kmax_mse` or reconstruction loss is already tracked.
- Stop extending joint fine-tuning once that minimum starts dropping much
  below its value at the checkpoint being fine-tuned FROM (e.g. more than
  ~2-3x below the starting min eigenvalue), rather than committing to a
  fixed large epoch count (200) chosen only for reconstruction/structural
  convergence.
- This is a monitoring/early-stopping change, not a new loss term -- it
  does not touch `RegConfig.lambda_z` or add any new regularizer.
- A cheap way to sanity-check a candidate AE checkpoint BEFORE spending a
  full Stage-2 training run on it: run
  `scripts/analyze_latent_predictability.py` and look at the min
  eigenvalue / condition number alone -- the other four diagnostics in
  this section turned out not to matter, but this one predicted the ~10x
  Stage-2 convergence gap that motivated this whole investigation.

---

## 35. Options for fixing the collapse: mechanism review and a prioritized plan (user-directed, 2026-08-31)

"we have two options, try to avoid channel collapse in stage 1 or
consider removing w_varmatch in phase 2... maybe we can add a term like
Combined Barrier / Log-Det Regularization that aims to limit condition
number collapse. I guess w_varmatch should be doing this in the first
stage of training though, so we could also try increasing its weight as
well, or have its weight on a schedule that increases over time."

### Correcting one premise first: `w_varmatch` is a Stage-2 (propagator) loss, not a Stage-1 one

Checked directly (`ks_latent/config.py:900`, `Stage2TrainingConfig.
w_varmatch`, used at `ks_latent/training/loops.py:439-445`; no field of
that name exists on `Stage1TrainingConfig`). It's applied to the
PROPAGATOR's rolled-out predictions during Stage 2, never to the
encoder's own output during Stage 1 -- there is currently no anti-collapse
pressure on the encoder itself anywhere in the pipeline. So "w_varmatch
should be doing this in stage 1" doesn't apply literally, but the
underlying intuition (something should be defending channel variance) is
exactly right -- it just isn't currently wired to the stage where the
collapse actually happens.

### The real `w_varmatch` mechanism -- and why "increase its weight" is likely to backfire

`decorr_var_loss` (`ks_latent/training/losses.py:33-41`):
```python
l_var = ((diag - 1.0) ** 2).mean()   # diag = per-channel Var(z_pred) across the batch
```
This targets a HARDCODED `1.0` for every latent channel, unconditionally
-- it was never data-adaptive to begin with (per its own `config.py`
docstring: "Target is fixed at 1... not independently configurable"). It
was added (2026-08-29) to fix a specific, different failure: autonomous
rollout collapsing every initial condition to the *same fixed point*
(`lambda1<0`, `D_KY=0`) -- `Var(z_pred)->0` ACROSS DIFFERENT INITIAL
CONDITIONS is the signature of that failure, and pushing every channel's
cross-IC variance toward 1 directly opposes it.

For a channel whose TRUE encoded variance is `2.42e-6` (Section 34), this
same mechanism is now actively wrong: it drags that channel's prediction
toward `1.0`, fighting the primary per-step latent-matching loss (which
correctly wants that channel near its true near-zero value) on exactly
the one channel that's collapsed. **Increasing `w_varmatch`'s weight,
or ramping it up over training, makes this specific conflict WORSE, not
better** -- it more forcefully drags the collapsed channel toward the
wrong target. This is the opposite of what the user's instinct ("it
should already be handling this, maybe turn it up") would predict, and
is worth flagging precisely because it's the kind of fix that would look
reasonable and make things quietly worse.

### Is `w_spatial` itself an accomplice in the Stage-1 collapse?

`spatial_coherence_loss` (`ks_latent/training/losses.py:44-127`) z-scores
each channel by its OWN std before computing the correlation matrix
(`z_n = z_c / std.clamp_min(1e-6)`, line 110-111) -- so, unlike a raw
covariance-based score, a channel's bandedness contribution is NOT
naturally suppressed just because its raw variance is small; after
normalization a nearly-collapsed-but-not-exactly-zero channel behaves
like any other unit-variance channel. The function's own docstring
already flags the closest thing to this risk it could find and rule out:
literal full collapse (every channel identical) scores only `mean(W)`, a
mediocre value, not the maximum -- but the docstring is explicit that
"locally-redundant-but-not-fully-collapsed optima remain possible, a
softer version of the same class of risk `lambda_z` fell into" and was
never ruled out. A channel that becomes a small-amplitude, highly
correlated near-copy of a neighbor (not literally zero, not literally
identical) is exactly that un-ruled-out soft failure mode, and would
score WELL under this loss (post-normalization, a small-scale copy of a
neighbor is indistinguishable from a full-scale one). Consistent with
this: even the ORIGINAL (100-epoch, not-yet-fine-tuned) AE already has
its smallest eigenvalue `2.04e-5` -- orders of magnitude below its
largest (`6.1`) -- suggesting this pressure was present and active from
early in training, not something that appeared abruptly during the extra
200 fine-tuning epochs; the extra epochs likely just let an
already-present trend run much further. Both AEs were trained with
`w_spatial=0.05`.

This is a genuinely different mechanism from the banned
[[no_banded_latent_regularizer]] (`RegConfig.lambda_z`): that one pulls
INDEX-NEARBY coordinates toward equal VALUES directly (a per-sample
proximity penalty); `spatial_coherence_loss` rewards CORRELATION
structure across a batch instead, and was deliberately built and
unit-tested to avoid `lambda_z`'s specific known failure (literal full
collapse). But per its own docstring, it was never verified against this
SOFTER failure mode, and the eigenspectrum evidence above is consistent
with it being exactly that.

### Full option menu

**A. Prevent the collapse in Stage 1**

1. *Early stopping via a monitored latent-health signal* (already
   proposed in Section 34): track min covariance eigenvalue / condition
   number during training (including during fine-tuning), stop extending
   once it degrades much past its starting point. Cheapest option, zero
   new loss terms, but reactive rather than preventive -- doesn't change
   why the drift happens, just limits how far it's allowed to go.
2. *Reduce or schedule down `w_spatial` specifically for extended joint
   fine-tuning.* If confirmed as a real contributor (see "recommended
   first experiment" below), decaying `w_spatial` toward 0 over a
   fine-tune's later epochs would let the aux propagator keep adapting to
   the encoder without continuing to pay the encoder for a redundancy
   trade its own docstring warned about.
3. *Add an explicit anti-collapse regularizer to Stage 1* (the user's
   Combined Barrier / Log-Det idea). Two concrete forms, genuinely
   different from `RegConfig.lambda_z` (that one enforces index-locality
   value proximity; these enforce a variance/eigenvalue FLOOR with no
   notion of index adjacency at all -- compatible with `w_spatial`'s goal
   rather than competing with it):
   - **3a. Per-channel variance floor** (VICReg-style):
     `L_floor = mean_i(relu(gamma - std_i)^2)` for a small target
     `gamma` (e.g. 0.1-0.5 relative to typical latent scale). Cheap,
     well-established elsewhere (self-supervised learning uses exactly
     this to prevent representation collapse), and directly targets the
     specific failure observed here (one channel's OWN marginal variance
     collapsing). Doesn't catch a purely correlation-driven joint
     eigenvalue collapse (a channel with normal marginal variance but
     near-total linear dependence on another), but that's not what was
     observed in Section 34 -- the marginal per-channel spectrum itself
     is what collapsed.
   - **3b. Full covariance log-det / eigenvalue barrier**:
     `L_logdet = -logdet(Cov(z) + eps*I)`, maximizing the entropy of a
     Gaussian fit to the batch's latent states, pushing the WHOLE
     spectrum away from zero rather than just the diagonal. Catches
     joint/correlation-driven collapse too (more principled), at the cost
     of being untested in this codebase (no local precedent the way
     VICReg's variance term has) and needing careful weight tuning so it
     doesn't just inflate noise directions. `d=44`, batch~256 makes the
     `logdet`/Cholesky itself computationally trivial -- the risk is
     tuning/novelty, not cost.
   - Recommendation: try 3a first. It's simpler, cheaper, better
     understood, and matches the specific failure mode actually observed;
     reach for 3b only if 3a proves insufficient (e.g. eigenvalues
     collapse via correlation even with healthy marginal variances).
4. *Don't jointly fine-tune the encoder for 200 more epochs at all* --
   freeze the AE and fine-tune only the aux propagator against it, or
   simply shorten the fine-tune. If the original motivation for the
   200-epoch fine-tune was to improve the aux propagator's own fit or
   push D4/D6/D7 further, and D4/D6/D7 were ALREADY strong on the
   100-epoch original AE (per Section 32), it's worth asking whether that
   much additional joint drift was ever necessary. This is the
   lowest-engineering-cost option of all -- no new code, just a
   different training recipe -- and worth keeping in mind as a fallback
   even if 3a is implemented.

**B. Mitigate the mismatch in Stage 2**

6. *Remove `w_varmatch` entirely* (the user's proposed option). Simplest
   possible change (already a `--w-varmatch 0.0` CLI flag, zero new
   code). Risk: it was added to fix a real, different, previously
   observed failure (autonomous rollout converging every IC to the same
   fixed point). Removing it outright without re-checking that failure
   mode on this AE/propagator pairing risks reopening it silently.
7. *Make `w_varmatch`'s target data-adaptive* instead of a hardcoded
   `1.0` -- compute each channel's true variance once from the Stage-1
   training data and match rolled-out predictions to THAT vector.
   Preserves the original anti-fixed-point-collapse motivation (matching
   real cross-IC diversity is the actual thing that distinguishes
   "healthy ensemble" from "converged point," regardless of the target's
   absolute scale) while removing the active conflict on the collapsed
   channel identified above. Mechanically better-targeted than either 6
   (loses the real protection entirely) or the user's tentative
   "increase the weight" idea (actively worsens the conflict, per the
   mechanism above) -- this is the recommended Stage-2-side fix if a
   Stage-2-only fix is wanted at all.

### Recommended plan, in order

1. **Fast, cheap, no new code, run first**: on the EXISTING fine-tuned/
   collapsed AE, compare Stage 2 `mlp`/markovian training at the
   Section 33 recipe (`k_max=8`, 68 epochs) across `--w-varmatch 0.0` vs.
   `0.1` (current default used throughout this document), a few seeds
   each. This directly tests how much of the ~10x gap is attributable to
   the active target-mismatch conflict alone, with zero implementation
   risk, before deciding whether options 3/7 are worth building at all.
   Also re-check autonomous free rollout stability (the Lyapunov/`D_KY`
   diagnostic `w_varmatch` was built to fix) on the `w_varmatch=0.0`
   condition specifically, so removing it isn't evaluated on Stage-2 fit
   quality alone while silently reopening the older failure.
2. **If (1) shows `w_varmatch` conflict explains a meaningful share of
   the gap**: implement option 7 (data-adaptive target) as the
   production fix rather than leaving it at 0 permanently, since 0
   forfeits real protection against a real, previously-observed failure
   mode.
3. **Regardless of (1)'s outcome, in parallel or after**: implement
   option 3a (per-channel variance floor in Stage 1) at a small weight,
   and re-run the 200-epoch fine-tune with periodic latent-covariance
   logging (`--checkpoint-every`, already available) so the min-eigenvalue
   trajectory can be watched across the whole fine-tune, not just judged
   at the end -- this is the only way to directly confirm the floor is
   working and to pick a sensible stopping point/weight.
4. **Keep option 4 in reserve**: if 3a doesn't fully prevent the drift,
   or introduces its own side effects, the simplest fallback is just not
   fine-tuning the AE that long in the first place.
5. **De-prioritized**: option 3b (log-det/condition-number barrier) as a
   fallback only if 3a proves insufficient; increasing `w_varmatch`'s
   weight or putting it on an increasing schedule, per the mechanism
   analysis above.

Step 1 is being launched now as `scripts/w_varmatch_ablation.sh` (new).

---

## 36. Implementing option 7 (data-adaptive `w_varmatch` target), and a surprising ablation result that reframes the whole investigation (user-directed, 2026-08-31)

"I like stage 2 (mitigate) 7 the best. is there a way to try this with
this autoencoder? [`stage2_history3_fullprop_wvar005_tw16_wspatial005
_200ep_mlpmarkovian_k20.log`]. should be clear pretty quickly whether that
fixes the problem."

### Result of Section 35's Step 1 (the cheap `w_varmatch=0.0` vs `0.1` ablation, launched before this message): a bigger surprise than expected

At the Section 33 recipe (`k_max=8`, 68 epochs) on the fine-tuned/collapsed
AE, 3 seeds each:

| `w_varmatch` | seed 0 | seed 1 | seed 2 | converged (<=0.05)? |
|---|---|---|---|---|
| `0.0` | 0.036852 | 0.036809 | 0.035997 | 3/3 |
| `0.1` | 0.036358 | 0.036547 | 0.035724 | 3/3 |

**Both conditions converge comfortably, and are statistically
indistinguishable from each other.** This means the fine-tuned/collapsed
AE is NOT inherently hard to fit at `k_max=8` -- Section 33's original
`_k8` sweep was killed before the `finetuned` group ever ran, so this is
the first real data point on that AE at that short horizon, and it
overturns the implicit assumption carried since Section 33 that the
fine-tuned AE is uniformly harder to fit regardless of curriculum length.
It is specifically the LONGER curricula (Section 32's `k_max=16`, ~0.22;
this section's `k_max=20`, stalled ~0.44-0.51) where the huge gap
appears. This is consistent with the collapsed channel being a
compounding-error problem: a short 8-step rollout doesn't give whatever
conflict exists (structural or `w_varmatch`-related) enough steps to
blow up, while a 16-20 step rollout does. This makes the user's requested
test on the `k_max=20` AE exactly the right next experiment -- the short
`k_max=8` regime this ablation used cannot distinguish the fixes at all.

### Implementation

- `ks_latent/training/losses.py`: `decorr_var_loss` gained a `var_target:
  Tensor | float = 1.0` parameter (default preserves the exact old
  behavior); `l_var = ((diag - var_target) ** 2).mean()`.
- `ks_latent/config.py`: `Stage2TrainingConfig.w_varmatch_adaptive: bool =
  False` (new field, default off).
- `ks_latent/training/loops.py`'s `train_stage2`: when `w_varmatch > 0
  and w_varmatch_adaptive`, computes `varmatch_target =
  train_sequences.reshape(-1, d).var(dim=0)` once before the epoch loop
  (the AE's own real per-channel variance on its encoded training data)
  and passes it into every `decorr_var_loss` call in place of the
  hardcoded `1.0`.
- `scripts/train_stage2_patched.py`: new `--w-varmatch-adaptive` flag
  (only meaningful with `--w-varmatch > 0`).
- All 310 existing unit tests still pass; new behavior is strictly
  additive/opt-in (default `False` reproduces prior behavior exactly).

### Test launched: the AE and recipe the user asked about

`stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial005_200ep.pt`
(Section 29's fresh, NOT fine-tuned, 200-epoch AE) independently checked
via `scripts/analyze_latent_predictability.py`'s covariance diagnostic and
found to have a similarly collapsed channel to Section 34's fine-tuned AE
despite never being fine-tuned at all -- min eigenvalue `4.55e-6`,
condition number `1.3e6` (vs. the original 100-epoch AE's `2.04e-5`/
`3.0e5`). This is a second, independent confirmation that the collapse is
a function of total joint-training duration in general (200 fresh epochs
alone reproduces it), not something specific to the fine-tuning procedure
Section 34 originally studied.

Its Stage 2 run at `--backbone mlp --mode markovian --epochs 200
--k-max 20 --k-mid 13 --k-warmup-epochs 140 --k-mid-epochs 117` stalled
around `val_kmax_mse~0.44-0.51` before being interrupted (Section 29's
tail) -- exact `w_varmatch` setting used in that original run unconfirmed
(not recorded in the surviving mid-training checkpoint or log), so
`scripts/w_varmatch_adaptive_test.sh` (new) reruns the SAME recipe from
scratch, same seed, three ways for a clean isolated comparison:
  (a) no `w_varmatch` at all (baseline),
  (b) `w_varmatch=0.1`, old fixed target=1,
  (c) `w_varmatch=0.1`, `--w-varmatch-adaptive` (the new fix).

Results to be appended here once complete.

---

## 38. Testing the two new anti-collapse regularizers: variance-floor fails silently, log-det barrier fixes the problem completely (user-directed, 2026-08-31)

"try a couple of stage 1 runs for 120 epochs using the two new
regularizers we developed. make their weight .04 in each case... Use the
same encoder decoder and aux propagator model as
[stage2_multiseed_wvmablation-wvm00_seed0.log]. stage 2 of training in
both of these cases should use a markovian mlp, w_varmatch = 0.
k_max = 12."

**Setup**: two FRESH (not initialized from any existing checkpoint) Stage-1
runs, 120 epochs, the canonical `history3_fullprop_wvar005_tw16_wspatial005`
architecture (`vit` encoder/decoder, `vit`/history aux, `w_var=0.05`,
`w_spatial=0.05`, `full_propagator`, `prop_delta_cap=0.5` -- the same
architecture underlying `wvmablation-wvm00_seed0`'s AE), one with
`--w-var-floor 0.04` (option 3a), one with `--w-logdet 0.04` (option 3b).
Each followed by a fresh `mlp`/markovian Stage 2 run, `w_varmatch=0`,
`k_max=12`, 100 epochs. `var_floor_gamma`/`logdet_eps` left at their
documented defaults (`0.1`/`1e-3`) -- only the requested top-level weight
varied.

**A pre-existing, unrelated bug found and fixed while sanity-checking the
new CLI flags**: a literal (unescaped) `%` in an old `--pool-bandwidth`
help string (`"Requires d_latent % n_sites == 0..."`) broke argparse's
`--help` formatting entirely (`ValueError: unsupported format character
'n'`) for the whole script, not just that flag -- fixed by escaping to
`%%`. Pre-existing, unrelated to today's regularizers; caught only because
`--help` was run as a sanity check before committing to a real launch.

### Run A (variance floor, `w_var_floor=0.04`): silently inactive, not just "too weak"

Final spectrum: `min_eig=1.56e-5, cond#=4.09e5` -- essentially in line with
what UNREGULARIZED training would produce anyway at 120 epochs
(log-linear interpolation between the known unregularized 100-epoch
[`2.04e-5`] and 200-epoch [`4.55e-6`] baselines predicts `~1.5e-5` at 120
epochs with no regularizer at all). Stage 2 (`k_max=12`): `best_val_kmax_
mse=0.070503` -- a real improvement over the badly-collapsed AEs at
similar/longer horizons (`~0.22-0.36`), but well short of full
convergence.

**Direct inspection of the final checkpoint's actual loss-term values**
explains why, and it is NOT simply "the weight needed to be bigger":

| term | raw value | weighted (this run's weight) |
|---|---|---|
| `variance_floor_loss` (w=0.04) | **0.000000** | 0.000000 |
| `decorr_var_loss`'s `l_var` (existing, w=0.01) | 0.001760 | 0.000018 |
| `spatial_coherence_loss` (w=0.05) | 0.382757 | 0.019138 |
| `decorr_var_loss`'s `l_decorr` (w=0.01) | 0.074333 | 0.000743 |
| `logdet_barrier_loss` (not active this run) | 2.419515 | -- |

`variance_floor_loss` is EXACTLY zero -- genuinely inactive, not
underpowered. Checked why directly: every channel's own MARGINAL std is
`0.95-0.98` (all comfortably above `gamma=0.1`), while the full
covariance's smallest EIGENVALUE is still `1.56e-5`
(std-equivalent `0.004`). This is only possible if channels are healthy
individually but nearly linearly redundant with each other -- exactly the
correlation-driven collapse `logdet_barrier_loss`'s docstring and its
unit test (`test_logdet_barrier_reacts_to_correlation_driven_collapse_
unlike_variance_floor`, Section 36... wait, added alongside Section 36's
other new-loss tests) were designed around, now confirmed to be the
REAL mechanism in an actual trained AE, not just a hypothetical. Since
`variance_floor_loss` only ever looks at the diagonal (marginal per-
channel variance), no weight increase could have helped here -- the
constraint it enforces was simply never violated. This is a genuine,
useful negative result: option 3a is the wrong tool for THIS specific
failure mode (though it may still matter for a more naive marginal-
variance collapse elsewhere).

### Run B (log-det barrier, `w_logdet=0.04`): collapse fully eliminated

Final spectrum: `min_eig=0.559, cond#=6.51, top_eig=3.64` -- unlike
ANYTHING else seen in this entire document. Every prior AE (including the
"good" original 100-epoch one) had a condition number in the `10^5-10^6`
range; this one's whole spectrum is nearly flat (bottom-10 eigenvalues
range `0.56-0.77`, close to the top eigenvalue itself). The regularizer's
own value on this checkpoint reacted strongly throughout training
(`2.42` on Run A's still-collapsed checkpoint, for reference, confirming
it "sees" the problem `variance_floor_loss` cannot).

**Stage 2 (`k_max=12`, fresh `mlp`/markovian, `w_varmatch=0`, 100 epochs):
`best_val_kmax_mse = 0.037618`** -- matching the best result anywhere in
this entire investigation (the original `_k8` run's `~0.037`, Section 26)
almost exactly, at a LONGER horizon (`k_max=12` vs. `8`), on a freshly
regularized AE. This isn't just "collapse avoided" -- it fully resolves
the practical problem that motivated the whole Section 34-38 investigation:
a latent space that a short, cheap `mlp`/markovian Stage-2 propagator can
actually fit well, without needing to limit the curriculum to `k_max=8`
or accept a `~0.22-0.44` plateau at longer horizons.

### Conclusion and recommendation

Between the two options laid out in Section 35, **the log-det/eigenvalue
barrier (3b) is the clear winner for this project's actual failure mode**,
and should become the standard Stage-1 anti-collapse mechanism going
forward (`--w-logdet 0.04`, or nearby, alongside the existing
`w_var`/`w_spatial` recipe) -- NOT the per-channel variance floor (3a),
which this real test showed to be structurally blind to the specific
(correlation-driven) way this project's AEs actually collapse. Given how
decisive and clean this result is, it's worth re-running the ORIGINAL
190/200/300-epoch-scale recipes with `--w-logdet` enabled from the start
to confirm the fix holds at the longer training durations where collapse
was originally observed to compound (Section 36's dose-response finding),
not just at the shorter 120-epoch scale tested here.

---

## 39. Retuning `w_spatial`/`w_logdet` to recover spatial coherence without losing the collapse fix (user-directed, 2026-08-31)

"section 38 lacks the same visual spatial coherence of the other runs
with w_spatial=.05. what do you think happened? the log det term
dominated the loss?"

**Diagnosis** (checked directly on real checkpoints, not just reasoned
abstractly): `l_spatial` really is worse on the Section 38 AE (`0.545`)
than the pre-log-det original (`0.396`). Mechanism: `logdet_barrier_loss`
is minimized by maximizing `det(Cov)` for a given total variance -- by
AM-GM, that's maximized when the eigenvalues are as EQUAL as possible,
i.e. it explicitly rewards an isotropic/decorrelated covariance.
`spatial_coherence_loss` wants the opposite: correlated NEARBY channels
(banded structure), which necessarily makes some eigenvalue directions
smaller relative to others. The two terms pull on the same underlying
quantity (the covariance's eigenvalue distribution) in genuinely opposite
directions -- not just independent noise competing for loss budget.
Likely compounded by training dynamics: early on, before `w_spatial` has
established banded structure, the AE tends to drift toward the same
correlated-redundant configuration Section 38's Run A diagnosis found,
so `l_logdet`'s gradient starts large and forcefully pushes toward
decorrelation before `spatial_coherence_loss` gets much of a foothold.

**First attempt (killed before completing)**: `--w-spatial 0.065
--w-logdet 0.025`, 200 epochs -- raising `w_spatial`'s pull while lowering
`w_logdet`'s. Killed by the user partway through Stage 1 in favor of a
sharper hypothesis: "I think we should just lower w_logdet honestly, it
changed the condition number hugely before" -- i.e. don't fight the
tension by raising both simultaneously, just weaken the term that's
winning it.

### `--w-spatial 0.05 --w-logdet 0.005`, 120 epochs: the best result of the whole investigation

| | Section 38 (`w_logdet=0.04`) | this run (`w_logdet=0.005`) | original (no logdet) |
|---|---|---|---|
| min eigenvalue | 0.559 | **0.081** | 2.04e-5 |
| condition # | 6.5 | 62.4 | 3.0e5 |
| top eigenvalue | 3.64 | 5.05 | 6.12 |
| `l_spatial` (lower=better) | 0.545 | **0.411** | 0.396 |
| D7 bandedness / p | -- | 0.5246 / 0.0000 | -- |
| Stage-2 `val_kmax_mse` (`k_max=12`) | 0.0376 | **0.0303** | (n/a, collapsed) |

Both goals achieved simultaneously: condition number still ~1000-4000x
better than any unregularized AE, spatial coherence back close to the
original baseline, AND the best Stage-2 result anywhere in this entire
document. **Gate 3/4, full results**: `λ1=0.0985`, `D_KY=21.24`,
`n_positive=10` (genuinely chaotic, no fixed-point signature); D3
p=0.189 (n.s.), D4 genuine translation-equivariant representation, D6
p=0.0000, D7 p=0.0000; DA cycling `rmse_da=0.289`, `rmse_free=0.896`,
`spread/rmse=0.321`, `skill_free_over_da=3.10` -- all healthy, comparable
to the best historical baselines without the scale-artifact inflation
seen in Section 38's own DA numbers (`rmse_da=0.872` there, likely
inflated by that run's much more isotropic/rescaled covariance making
raw per-dimension latent RMSE not apples-to-apples across runs -- see
Section 38's discussion). Visualization (`docs/figures/latent_hovmoller_
history3_fullprop_wvar005_tw16_wspatial005_logdet0005_120ep_
mlpmarkovian_k12.png`) shows clear, visually obvious banded structure
in the ground-truth encoding (neighboring latent indices co-varying in
coherent blocks) that the model rollout tracks closely.

---

## 40. Pushing further: `--w-spatial 0.085 --w-logdet 0.001` -- a clean tradeoff curve

"the same stage 1 and stage 2 training as before but with w_spatial .085
and logdet = .001"

| | `w_logdet=0.005` (Section 39) | `w_logdet=0.001` (this run) |
|---|---|---|
| min eigenvalue | 0.081 | 1.59e-4 |
| condition # | 62.4 | **3.78e4** (600x worse, still ~10x better than unregularized) |
| `l_spatial` | 0.411 | **0.368** (best of everything tried, better than the original 0.396) |
| D7 bandedness / p | 0.5246 / 0.0000 | 0.5113 / 0.0000 |
| Stage-2 `val_kmax_mse` (`k_max=12`) | **0.0303** | 0.0625 |

Pushing `w_logdet` lower keeps buying spatial coherence, but conditioning
degrades fast (600x jump in condition number between `0.005` and
`0.001`), and Stage-2 fit quality tracks conditioning, not `l_spatial`,
as the thing that actually matters for propagator fitting -- `0.005`
remains the better all-around pick on that basis.

**Gate 3/4**: `λ1=0.0964`, `D_KY=21.36`, `n_positive=12`; D3 p=0.964
(n.s.), D4 genuine equivariance, D6/D7 both p=0.0000. DA cycling:
`rmse_da=0.199`, `rmse_free=0.790`, `spread/rmse=0.397`,
`skill_free_over_da=3.98` -- nominally the BEST DA numbers of any run
tested, which is a genuine open question against the `val_kmax_mse`
ordering: this AE's overall latent scale (`top_eig=6.0`) sits closer to
the original/historical baseline than `0.005`'s (`top_eig=5.05`), so part
of the apparent DA improvement could be the same raw-per-dimension-RMSE
scale artifact flagged in Section 38/39 rather than a real forecasting
advantage -- not yet disentangled. Visualization: final relative RMSE at
`t=201` = 1.39 (same "past the chaos horizon" range as every other run,
not informative on its own).

---

## 41. `--w-spatial 0.06 --w-logdet 0.002`, 140 epochs, 200-epoch Stage 2 -- killed before completion

Queued as a natural next point on the curve (between `0.005` and `0.001`,
with a longer Stage-2 budget to see if the harder `k_max=12` curriculum
converges further given more epochs). Killed by the user before Stage 1
completed; no checkpoint or results produced.

---

## 42. Summary of 2026-08-31's investigation, and a forward-looking idea: modeling the latent state as a PDE

**The arc, start to finish**: Section 34 found that two AEs with nearly
identical structural diagnostics (D4/D6/D7) and nearly identical
training-free one-step/multi-step predictability statistics could still
differ by ~10x in actual Stage-2 `mlp`/markovian propagator convergence,
traced to a single latent channel's variance collapsing to the numerical
noise floor -- invisible to every diagnostic except a direct look at the
latent covariance's eigenspectrum. Section 35 laid out options; Section 36
found the effect is progressive with total joint-training duration
regardless of aux backbone, and specific to longer `k_max` curricula
(short `k_max=8` recipes converge fine even on a collapsed AE). Section 38
tested two new Stage-1 anti-collapse regularizers side by side and found
the obvious-seeming one (a per-channel variance floor) provably does
nothing for this project's actual failure mode (a correlation-driven,
not marginal-variance-driven, collapse), while a log-det/eigenvalue
barrier fixes it completely -- taking the condition number from
`10^5-10^6` (every AE trained without it) down to single digits, and
Stage-2 convergence at `k_max=12` down to `~0.03-0.04`, matching or
beating the best result anywhere in this document (previously only
achieved at the much shorter `k_max=8`). Sections 39-41 then found and
partially explored a genuine tension between `w_logdet` (rewards an
isotropic covariance) and `w_spatial` (rewards banded local correlation,
which is NOT isotropic) -- **`--w-spatial 0.05 --w-logdet 0.005`** is the
best balance found so far: condition number `62.4` (still ~1000x better
than unregularized), `l_spatial` back near the original baseline, and
the best Stage-2 result of the whole investigation (`val_kmax_mse=
0.0303` at `k_max=12`). This should be adopted as the new standard
Stage-1 recipe going forward, superseding the plain `w_var`/`w_spatial`
recipe used throughout Sections 1-33.

**Forward-looking idea (user-proposed)**: "I think the latent variable
structure we've induced with logdet and wspatial should allow us to
model the latent variable with a pde." This is well-motivated by what's
actually been measured: `spatial_coherence_loss`/D7 specifically reward
and confirm BANDED, index-LOCAL correlation structure (`bandedness_score
~0.51-0.52`, `p=0.0000` throughout Sections 39-40) -- i.e. the latent
vector now behaves like a discretized field over a 1D domain (its own
index axis) with genuine local coupling between neighbors, much like the
physical KS state it's encoding. A generic dense `mlp` propagator (used
throughout this investigation) has no inductive bias reflecting that
locality at all -- every latent-index pair gets an independent weight,
identical to how it would treat a latent space with no spatial meaning.
A PDE-style (local-stencil) propagator, by contrast, would directly
exploit exactly the structure now being deliberately engineered.

**Concretely, and cheaply testable**: `PropagatorConfig(backbone=
"masked_mlp", attn_window=...)` already exists in this codebase (used
earlier in `maskedmlp_encoder_e2e`, Section 22-23) -- a locally-masked
MLP that only lets each latent index's update depend on a narrow window
of index-neighboring inputs, i.e. already a discrete local-stencil
propagator in spirit. Worth trying directly on the Section 39 AE (the
`w_spatial=0.05`/`w_logdet=0.005` checkpoint) in place of the dense `mlp`
backbone Sections 38-41 all used, now that the latent space has been
deliberately shaped to have real local structure for it to exploit --
previously, using `masked_mlp` on a latent space with no confirmed
locality would have been an architectural bet with no evidence behind
it; now there is direct, repeated, statistically significant evidence
(D7) that the bet is well-founded. A true continuous-PDE-form propagator
(e.g. a small fixed stencil + explicit finite-difference-style update,
or a narrow 1D convolution) would be a further, more committed step in
the same direction, worth trying only after `masked_mlp` confirms the
locality is exploitable at all.

---

## 43. Contraction vs. chaos: why an unconstrained-mixing aux propagator (mlp) may be prone to condition-number blowup where softmax-attention (vit) isn't (user-directed, 2026-09-01)

Motivated directly by Section 42's anomaly: a Stage-1 run using an `mlp`/
markovian aux (the first time this whole investigation used a non-`vit`,
non-attention aux) produced a latent covariance with `top_eigenvalue=51.04`
-- roughly 10x every other checkpoint in this document (`~3-6` range) --
and the resulting warm-started Stage-2 propagator was catastrophically bad
(`val_kmax_mse` stuck at `~2.7-3.0`, vs. the usual `~0.03-0.08`), even with
`w_logdet=0.005` active. Section 42b re-runs the identical recipe with the
aux swapped back to `vit`/history (`n_history=2`) to test directly whether
the aux's own architecture family is the deciding factor.

**Two hypotheses considered for why `vit` might behave differently, and
which one the evidence favors**:

1. *"The `vit` aux is smaller, so it can't overfit/produce exploding
   modes."* Checked directly by instantiating both at the actual sizes
   used (`hidden=128, n_blocks=3` for `mlp`; `n_tokens=44,
   token_d_model=64, token_n_layers=2` for `vit`/history):
   `mlp` aux = **111,532** params, `vit` aux = **100,418** params -- only
   a ~10% difference, nowhere near enough to explain a ~10x eigenvalue
   gap. Raw model size is not the likely explanation.

2. *"The `vit` aux is inherently more regularized against high condition
   numbers"* -- true, but the more precise mechanism is **softmax
   normalization's non-expansiveness**: the `mlp` aux's
   `input_proj = Linear(44, 128)` mixes all 44 latent dimensions with
   completely unconstrained weights -- nothing architectural stops
   gradient descent from learning a matrix that disproportionately
   amplifies variance along one direction, and Section 42's recipe
   (`w_pred=1.0`, doubled; `k_pred_max=4` rollout pressure; `w_var=0`,
   `w_decorr=0`; only a small `w_logdet=0.005` pushing back) removed
   nearly every counterweight that existed in every other recipe in this
   document. `vit`'s self-attention, by contrast, aggregates via
   softmax-weighted combinations of value vectors -- softmax weights sum
   to 1 by construction, so every token update is a CONVEX COMBINATION of
   existing values. That is an architecture-level ceiling on how much the
   mixing step alone can amplify any single direction; a dense `Linear`
   has no analogous constraint.

**Connection to contraction vs. chaos, and to this project's own prior
findings**: the user's follow-up observation -- "contraction is good for
short-term prediction but terrible for replicating chaotic behavior in
the long run (they just contract to a single degenerate value)" -- is
exactly right, and is not new to this session. Chaos requires at least
one positive Lyapunov exponent (local divergence of nearby trajectories,
bounded globally via stretch-and-fold); a contractive map (all local
Lyapunov exponents negative) gives good short-horizon accuracy precisely
because it damps out errors, but is mathematically incompatible with a
genuine chaotic attractor, since any two initial conditions eventually
converge to the same trajectory. Plain MSE training on bounded rollouts
has a structural bias toward exactly this: "hedge toward a fixed point"
is a cheap way to minimize squared error unless something specifically
opposes it. **This project already has direct, first-party evidence of
this failure mode**: `PropagatorConfig.delta_cap`'s own docstring
(Sections 5-9 of this document) describes an earlier propagator collapsing
to a single global fixed point (`D_KY=0`) under plain MSE training,
motivating the `delta_cap`/"stretch-and-fold" architectural fix -- the
same phenomenon now appearing again in a different architecture (softmax
attention's tendency toward contraction generally, vs. Section 42's
unconstrained-mixing `mlp` swinging the opposite way toward instability).

**External literature** (held loosely on exact citations, offered as
research *themes* rather than verified references): this tension is a
recognized issue in chaotic-system surrogate modeling, often framed as a
"weather vs. climate" tradeoff -- a model can achieve strong short-term
(weather-like) forecast skill while systematically failing to reproduce
long-term (climate-like) attractor statistics, including underestimating
the true Lyapunov spectrum. The reservoir-computing line of work
associated with Pathak, Ott, and collaborators is the most directly
relevant research area recalled here, notably because it commonly uses
the Kuramoto-Sivashinsky equation itself as a benchmark system (the same
PDE this project models) and explicitly evaluates learned/hybrid models
on Lyapunov-spectrum/attractor-statistics fidelity rather than short-term
error alone, for exactly this reason.

**Status**: Section 42b (`vit`/history `n_history=2` aux, otherwise
identical recipe to Section 42) is the direct empirical test of hypothesis
2 above -- if its resulting AE's top eigenvalue lands back in the normal
`~3-6` range, that is strong support for the softmax/non-expansiveness
mechanism over a generic "vit is more regularized" story. Results to be
appended once that run completes.

---

**NOTE (2026-09-02)**: Sections 44-52 (the `--amp`/bfloat16 rollout, the
D8 signed-bandedness diagnostic and `spatial_signed` loss variant, and
the `w_spatial_signed` weight sweep from 0.002 through 0.01 across
Sections 47-52) were run and reported to the user in-conversation but
were never appended to this file -- a documentation gap discovered while
writing up Section 53 below. That backfill is still owed; the real
per-run numbers exist in `artifacts/logs/`, `docs/diagnostics_report_*.md`,
and `artifacts/analysis_suite_full_*.json` for each tag and should be
reconstructed from those files (not from memory) if/when it is done.
Headline results that are verified and safe to state now: Section 47
(`w_spatial_signed=0.08`) caused a catastrophic collapse (`min_eig=2.09e-6`,
`cond#=1.07e7`, `D_KY` crashed to 7.97 -- the first time the Lyapunov
spectrum itself, not just the covariance, visibly degraded); Section 48
(`w_spatial_signed=0.002`) was the best result of the sweep; Stage-2 fit
quality (`val_kmax_mse`) stayed remarkably flat (~0.017-0.019) across
roughly five orders of magnitude of covariance conditioning degradation
before finally breaking down at `cond#~1.6e6` (`w=0.01`, Sections 52-53).

## 53. Both model components scaled ~1.2x (not just the propagator): ViT `d_model=108`, aux propagator `hidden=141`/`n_blocks=3` (user-directed, 2026-09-02)

The user asked for "the models" (plural) to be made 20% bigger; an
earlier attempt only scaled the aux/Stage-2 propagator's
`--aux-hidden`/`--aux-blocks` (a flag that already existed from a prior
section) and left the ViT encoder/decoder at its default `d_model=96` --
caught directly by the user ("did you just make the propagator bigger? I
want a bigger encoder and decoder as well"), which required killing the
in-progress run and adding a new `--d-model` CLI override to
`train_stage1_patched.py` before relaunching.

**Sizing**: `d_model` must stay divisible by `n_heads=4`. Direct
instantiation/param-counting picked the closest achievable ratio to 1.2x:

| `d_model` | encoder+decoder params | ratio vs. default (96 -> 816342) |
|---|---|---|
| 100 | 879154 | 1.077 |
| 104 | 944270 | 1.157 |
| **108** | **1011690** | **1.239** |
| 112 | 1081414 | 1.325 |

`d_model=108` was used. The aux propagator used the existing
`hidden=141, n_blocks=3` (133853 params vs. the 111532 default, exactly
1.2x). Both figures were re-verified against real (non-smoke) checkpoint
files before the full launch, not just computed in the abstract.

**Recipe**: otherwise identical to the Section 48-52 family --
`vit`/`mlp`+markovian aux, `w_decorr=0`, `w_var_floor=0`, no `delta_cap`,
`--amp`, Stage 2 warm-started via `--init-prop-checkpoint`, `k_max=12`,
Stage 1 200 epochs / Stage 2 300 epochs. Loss weights (also user-specified
for this run): `w_var=0.025`, `w_logdet=0.0045`, `w_spatial_signed=0.01`.

**Result**:

| Metric | Value |
|---|---|
| Stage 1 `val_recon_final` | 0.000215 |
| latent `cond#` | 1.62e6 |
| latent `min_eig` | 9.25e-6 |
| D7 p-value | 0.0000 |
| D8 (signed) p-value | 0.0000 |
| Stage 2 `val_kmax_mse` | 0.018573 |
| rollout final relative RMSE (t=201) | 1.53 |
| `rmse_da` | 0.4398 |
| `rmse_free` | 0.9373 |
| `skill_free_over_da` | 2.13 |
| `lambda1` | 0.1065 |
| `D_KY` | 21.22 |
| `n_positive` | 11 |
| D3 p-value | 0.071 |
| D4 | no clean linear translation representation found |
| D6 p-value | 0.0000 |

**Correction (2026-09-02)**: the paragraph below originally stated the
true KS attractor's `D_KY` is `~4-5`; the user corrected this -- the true
value is `D_KY~22`. That changes the reading of this section's own
result substantially: `D_KY=21.22` is actually very close to the true
attractor's dimension, not "far above" it as originally written. This
also means several earlier sections' Gate 3/4 commentary in this document
(and in-conversation reports) that judged high `D_KY`/`n_positive` values
as straightforward evidence of Lyapunov-spectrum degradation should be
treated with caution -- the correct benchmark to compare against is
`D_KY~22`, not the low single-digit figure that was assumed. `lambda1`
and `n_positive` benchmarks have not been independently re-confirmed and
should not be taken as settled either.

**Interpretation**: scaling both components up ~1.2-1.24x did not change
the latent covariance conditioning already seen at this same
`w_spatial_signed=0.01` weight in Section 52 (`cond#=1.59e6` there vs.
`1.62e6` here, essentially unchanged), and Stage-2 fit quality improved
somewhat (`val_kmax_mse` 0.0186 vs. 0.0256). With `D_KY=21.22` now read
against the correct `D_KY~22` benchmark, this run's Lyapunov spectrum is
close to the true attractor's dimension rather than degraded -- so the
capacity increase (or this weight configuration generally) does not show
the same clear spectral-degradation signature that Section 47's
`w_spatial_signed=0.08` collapse did. The `cond#=1.6e6`
conditioning level itself may still be worth improving, but it is not
demonstrated here to be harming attractor fidelity the way it was
previously assumed to. Section 54 (launched immediately after, same
larger models, same recipe as Section 52 but `w_var` reduced
0.02 -> 0.0015 and `w_logdet` reduced 0.0035 -> 0.000325) is still a
useful follow-up test of the regularizer-weight-balance question, but its
interpretation should also be revisited against the corrected `D_KY`
benchmark once it completes.

---

**NOTE (2026-09-03)**: Sections 54-65 (the model-size-scaling sweep at
0.8x/1.2x, the `dt_snap=0.25` experiment, `RegConfig.start_epoch`-delayed
`lambda_z` tests, and the vit-vs-mlp/history propagator comparison on
Section 65's AE) were run and reported to the user in-conversation but,
like the 44-52 gap noted above, were never appended to this file. Still
owed; reconstruct from `artifacts/logs/`, `docs/diagnostics_report_*.md`,
and `artifacts/analysis_suite_full_*.json` if/when done, not from memory.

## 66. Verified: the latent index shows genuine periodic (ring) structure, not just local circular coherence (user-directed, 2026-09-03)

While reviewing `scripts/make_latent_gif.py`'s output for Section 52
(`docs/figures/latent_state_evolution_section52_mlpmarkovian_wvar002_
wspatialsigned01_logdet0035_200ep.gif`), the user noticed the latent
state looked "almost approximately periodic" across the latent index and
asked for this to be verified quantitatively rather than taken on visual
impression alone.

**Verification** (same AE checkpoint and ground-truth trajectory as the
GIF, `z_true` from `ae.encode`, same-time Pearson correlation matrix
across the batch of 201 snapshots):

- `corr(z_0, z_43)` (the two ends of the 44-dim latent index) = **0.6999**
  -- nearly identical to the average correlation between LINEARLY
  adjacent indices (distance 1), **0.7167**.
- The full correlation-vs-LINEAR-distance curve is not a monotonic decay
  (what a purely local, non-periodic path-graph structure would produce)
  but a smooth, symmetric **U-shape**: 0.717 at distance 1, crossing zero
  near distance 11, bottoming out at **-0.624 at distance 24** (within 2
  of `d/2=22`, i.e. exact half-way around a 44-element ring), then rising
  back up to 0.700 by distance 43 (the far endpoint). This is the
  textbook signature of a single-low-order-Fourier-mode-like periodic
  structure in the latent index, not sampling noise.

**Why this is plausible, and the one part that is genuinely non-obvious**:
`spatial_coherence_loss` (backing `w_spatial_signed`, active in Section
52 and nearly every section since) already weights its correlation
target by a Gaussian kernel over CIRCULAR index distance (`w(dist) =
exp(-dist^2/(2*bandwidth^2))`, `bandwidth=3` by default) -- the same
circular convention D3/D6/D7's `bandedness()` uses, deliberately matching
the KS system's own periodic boundary condition (`u_t + u*u_x + u_xx +
u_xxxx = 0`, `x` in `[0, L)`, periodic -- `ks_latent/solver/ks.py`). So
index-0-correlates-with-index-43-the-way-it-correlates-with-index-1 is a
direct, INTENDED consequence of that loss term, not a surprise on its
own. What is NOT directly rewarded, and is therefore the genuinely
emergent part of this finding: at `bandwidth=3` the Gaussian kernel's
weight at distance ~22 is essentially zero, so nothing in the loss
rewards or penalizes whatever correlation structure exists at that
distance. The network nonetheless produced a strong, clean ANTI-
correlation there (-0.624) rather than noise -- i.e. a globally coherent
periodic organization emerged from a loss that only ever looks at local
neighbors.

**Implication**: the encoder has organized its latent index to
faithfully mirror the physical domain's own periodic topology, not just
locally (which the loss explicitly asks for) but globally (which it does
not). This is a meaningful piece of evidence that `w_spatial_signed`
training is accomplishing something more structurally significant than
"make nearby channels correlated" -- it appears to be recovering
something close to a genuine circular/Fourier-mode-like coordinate system
for the latent index, consistent with the codebase's Phase 10 backlog
goal of giving the flat latent index physical/spatial meaning. Two
follow-ups worth doing: (1) check whether this periodicity signature is
present (and how strong) across the rest of the `w_spatial_signed` sweep
(Sections 47-65) to see whether it scales with the weight the way D8
bandedness does, or saturates/appears independently of it; (2) if it
holds up, this directly informs the periodic-latent-regularizer idea
discussed the same day (a `BandedSmoothness`-style ring-generalized
Laplacian, `lambda_z_ring`) -- that regularizer would be reinforcing
structure the network already tends to find on its own under
`w_spatial_signed`, rather than fighting for it from scratch.

---

**NOTE (2026-09-04)**: Sections 67-70 were run in this session but their
launch scripts/docstrings were not preserved with the same level of
verified detail as the entries below (a pre-existing gap in the same
spirit as the 54-65 note above). Roughly: they explored variations on
masked `fourier_mlp` attention windows and Fourier-readout placement,
setting up the window-sizing and `fourier_ifft_readout` conventions that
Section 71 onward uses directly. Sections 71 onward are documented in
full below, with every number pulled directly from the corresponding
`artifacts/logs/{stage1,stage2,spectrum,gate3,gate4}_section{N}_*.log`
file rather than from memory.

**Standing background regularizer for context**: `w_pred`/`k_pred`
(default 0.5/2, the Stage-1 auxiliary-propagator loss) is active in every
run below without being explicitly set on the command line unless noted.

## Section 71: masked `fourier_mlp`, narrow attention window (attn=4), Section 66 regularizer recipe

Masked `fourier_mlp` AE (~1.0M params) + masked `fourier_mlp` propagator
(~1.0M params), `attn_window=4` (narrow -- not yet "effectively fully
dense"), `mode=history`/`n_history=2`, `lambda_z=0.0002` from epoch 0, 200
Stage-1 epochs + 300 Stage-2 epochs (`k_max=12`, warmstarted).

- Stage 1: `val_recon_final = 0.000068`.
- Stage 2: did not settle -- `val_kmax_mse` still oscillating around
  0.23-0.24 at epoch 185-187/300 (best checkpoint not separately
  recorded in this log). Substantially worse than Section 75's eventual
  0.039 at the same `k_max=12`.
- Latent spectrum: `cond#=310`, `top_eig=21.9`, D7 bandedness=0.212 (p<0.001),
  D8 signed bandedness=0.816 (p<0.001) -- strong same-time coupling
  structure, but Stage-2 rollout quality was poor. This narrow-window
  configuration was the starting point later widened to "effectively
  fully dense" windows in Section 74/75.

## Section 72: masked AE (attn=12) + dense propagator, `fourier_mlp`

Same AE/propagator family, AE `attn_window=12`, propagator moved toward
denser masking. Section 66 regularizer recipe, `lambda_z=0.0002`.

- Stage 1: `val_recon_final = 0.000078`.
- Stage 2: `best_val_kmax_mse = 0.148534` (better than 71, still far off
  75's later 0.039).
- Latent spectrum: `cond#=365`, `top_eig=22.6`, D8 signed
  bandedness=0.825 (p<0.001).

## Section 73: masked encoder (attn=4) + dense propagator + IFFT decoder

Attempted decoupling the decoder onto a pure `FourierIFFTBody` readout
while keeping a narrow masked encoder. **Killed before completing Stage
2** -- no `stage2_section73_*.log` or `spectrum_section73_*.log` exists.
Superseded by Section 74's corrected "dense means wider `attn_window`,
not dropping the raw-value path" implementation.

## Section 74: `fourier_ifft_readout`, `attn_window=8`/`prop_attn_window=22`/`dec_attn_window=22`

First working version of the "effectively fully dense" propagator/decoder
convention (`attn_window=22 = d_latent//2` at `d_latent=44`, verified by
direct mask-all-True check) plus `fourier_ifft_readout` (the masked path's
final layer is replaced by an explicit `irfft` of predicted frequency
coefficients -- this is the `FourierIFFTBody` mechanism, later reused
directly in Sections 75-83). Section 66 regularizer recipe,
`lambda_z=0.0002`.

- Stage 1: `val_recon_final = 0.000064`.
- Stage 2: `best_val_kmax_mse = 0.181539` (did not fully settle by epoch
  299/300, still trending down: 0.278 in the final logged epochs before
  best-checkpoint restore).
- Latent spectrum: `cond#=382`, `top_eig=23.3`, D8 signed
  bandedness=0.831 (p<0.001).
- This established the `attn_window=8`(enc)/`22`(prop)/`22`(dec) +
  `fourier_ifft_readout` architecture that Section 75 then paired with a
  substantially revised regularizer recipe to get the session's best
  single-architecture result to date.

## Section 75: same architecture as 74, revised regularizers -- **the reference point for the rest of the session**

Identical architecture to Section 74 (`attn_window=8/22/22`,
`fourier_ifft_readout`, masked `fourier_mlp` AE ~1.0M params +
`fourier_mlp` propagator ~1.0M params, `mode=history`/`n_history=2`).
Regularizers revised to: `w_decorr=0`, `w_var=0.01`, `w_spatial=0.01`
(**signed**), `w_var_floor=0`, `w_logdet=0.008`, `lambda_z=0.0002` (all
from epoch 0). This is the recipe referred to throughout the rest of this
document as **"Section 75's regularizers"** or **"Section 75's recipe."**

- Stage 1: `val_recon_final = 0.000035` (best reconstruction of the
  family to this point).
- Stage 2 (`k_max=12`, `k_warmup_epochs=210`, `k_mid=8`/`k_mid_epochs=175`,
  300 epochs): **`best_val_kmax_mse = 0.039243`** -- the best Stage-2
  rollout result achieved with the pure masked-`fourier_mlp` architecture
  this session, and (per the user, in the plan-discussion that follows)
  qualitatively "the rollout was the best I've seen, it really captured
  the dynamics the best I've observed."
- Latent spectrum: **`cond#=20.98`** (an order of magnitude better than
  71/72/74), `top_eig=9.68`, participation-ratio-adjacent bottom-10
  eigenvalues all `>0.46` (no near-collapsed directions), D7
  bandedness=0.272 (p<0.001), D8 signed bandedness=**0.341** (much lower
  than 71/72/74's ~0.82-0.83 -- less redundant, more spread-out latent
  geometry at the same `w_spatial` mechanism, attributable to the
  `w_logdet`/other-regularizer combination, not `w_spatial` alone).
- Gate 3 full analysis suite (`run_analysis_suite.py`): **`D_KY =
  21.089`**, `n_positive = 12`, `lambda1 = 0.0913` -- within ~4% of the
  true KS attractor benchmark `D_KY ~ 22` (see
  `project_ks_true_dky_benchmark` memory), the best Lyapunov-spectrum
  fidelity achieved this session at the time.
- **This run is the reference/baseline for every subsequent section**
  (76-83) and for the ViT-hybrid comparison (80-83).

## Section 76: extended Stage-2 curriculum (`k_max=20`), stronger regularizers

Same AE/propagator architecture as 75. Regularizers pushed much harder:
`w_var=0.00005` (down), `w_spatial=0.4` (signed, 40x Section 75's 0.01),
`w_logdet=0.004` (down), `lambda_z=0.0005` (2.5x). Stage 2 extended to
`k_max=20`, 500 epochs.

- Stage 1: `val_recon_final = 0.000060`.
- Stage 2: **did not converge** -- `val_kmax_mse` still ~0.91-0.92 and
  `val_k12_mse` ~0.40-0.41 at epoch 96-98/500, an order of magnitude worse
  than Section 75. The much larger `w_spatial=0.4` collapsed the
  propagator's learnability (consistent with the
  `no_banded_latent_regularizer`/collapse-diagnosis memory -- **do not
  push `w_spatial` this high**).
- Latent spectrum: `cond#=462`, `top_eig=52.1` (much larger raw scale --
  consistent with weaker `w_var`), D8 signed bandedness=0.846 (highest of
  the session -- `w_spatial=0.4` did drive up same-time coupling, but at
  the cost of propagator learnability). This run motivated backing
  `w_spatial` down substantially in 77.

## Section 77: `w_spatial x3` (0.03, not 0.4), `k_max=20`, otherwise Section 75's recipe

More moderate `w_spatial` increase than 76 (0.01 -> 0.03, 3x rather than
40x), keeping `lambda_z=0.0002` and the rest of Section 75's recipe,
extended to `k_max=20`/500 epochs.

- Stage 1: `val_recon_final = 0.000047`.
- Stage 2: `val_kmax_mse` ~0.51, `val_k12_mse` ~0.233-0.235 at
  epoch 274-276/500 -- much better than 76 but still notably worse than
  75's 0.039 (though not a fair comparison since `k_max` differs; see
  `compare_k`/`val_k12_mse` discussion in the new summary doc).
- Latent spectrum: `cond#=122`, `top_eig=22.8`, D8 signed
  bandedness=0.696 -- intermediate between 75 (0.34) and 76 (0.85), as
  expected from the intermediate `w_spatial`. Evidence that even a
  moderate `w_spatial` increase (3x) degrades propagator fit relative to
  75's 0.01, reinforcing the standing "don't raise `w_spatial`" memory.

## Section 78: `d_latent=64` (was 44), same recipe as 77

Grew the latent dimension from 44 to 64 (`prop_attn_window` raised
22->32 = new `d_latent//2` saturation point, verified mask-all-True) to
test whether more room reduces the correlation-driven collapse at higher
`w_spatial`. `w_spatial=0.015`, `w_logdet=0.016` (2x), `lambda_z=0.0002`,
`k_max=20`/500 epochs.

- Stage 1: `val_recon_final = 0.000038`.
- Stage 2: `val_kmax_mse` ~0.385, `val_k12_mse` ~0.142-0.143 at epoch
  376-378/500 -- the best `val_k12_mse` among 76/77/78's `k_max=20` runs,
  but still notably worse than Section 75's `val_kmax_mse=0.039` at
  `k_max=12` directly (not an apples-to-apples comparison, but
  directionally consistent with a propagator-capacity ceiling).
- Latent spectrum: **`cond#=11.7`** (best-conditioned latent of the whole
  masked-`fourier_mlp` family), `top_eig=7.36`, **participation ratio =
  37.4 out of 64** (excellent spread -- no near-collapsed directions), D8
  signed bandedness=0.211 (lowest/best-behaved of the sweep). **This is
  the "excellent encoder diagnostics, mediocre Stage-2 fit" case** flagged
  in Problem Solving: hypothesized (not yet directly tested) that the
  SAME propagator capacity (`hidden=480/n_blocks=2`) now has to predict a
  50% larger output space, and that is the bottleneck rather than the
  encoder geometry. A Phase-0 disambiguation test (grow the propagator to
  match) was scaffolded (`--nonexpansive`/`--fourier-ifft-readout` CLI
  flags added to `train_stage2_patched.py`) but never launched -- still
  outstanding, see the new summary doc's Open Questions.

## Section 79: Section 77's recipe, `w_pred x2` (1.0, was 0.5)

Isolated the effect of the Stage-1 auxiliary-propagator loss weight
(`w_pred`), testing "would increasing this force the embedding to be more
'propagatable'?" Identical to Section 77 otherwise (`w_spatial=0.03`,
`d_latent=44`, `k_max=20`/500 epochs).

- Stage 1: `val_recon_final = 0.000057`.
- Stage 2: `best_val_kmax_mse = 0.329930` (at that best epoch,
  `val_k12_mse = 0.112912`) -- better `best_val_kmax_mse` than 77's final
  logged value, though 77's run wasn't fully converged either. Final
  epoch 499/500: `val_kmax_mse=0.508`, `val_k12_mse=0.238`.
- Latent spectrum: essentially identical to 77 (`cond#=122.6`,
  `top_eig=22.9`, participation ratio 3.6/44, D8 signed
  bandedness=0.697) -- `w_pred` doubling had negligible effect on the
  static encoder geometry, as expected (it only touches the Stage-1
  auxiliary-propagator loss, not the AE's own reconstruction/regularizer
  terms). Effect on Stage-2 fit was modest and not clearly better than
  simply not touching `w_pred` -- deprioritized in favor of the
  ViT-hybrid direction that followed.

## Section 80: first ViT+Fourier hybrid attempt (`ConservedFourierMLP`, L1-conservation)

First implementation of the hybrid architecture: `encode(u) =
vit.encode(u) + fourier_encoder(fourier_features(u))` (and mirrored for
decode), where `vit` is Section 52's ViT-for-function-space architecture
and `fourier_encoder`/`fourier_decoder` is `ConservedFourierMLP` -- a new
module enforcing "the absolute value of each output of each layer sums to
the same value as the original sum of absolute values of the frequency
coefficients" (an L1 conservation law per-layer, checked at every
intermediate layer, not just the final output), as an alternative to
spectral-norm-based non-expansiveness. `d_model=72`, `fourier_hidden=246`,
`fourier_blocks=3`, sized to roughly match Section 75's ~1M AE params.
Section 75's exact regularizer recipe and Stage-2 curriculum.

- Stage 1: rocky optimization -- large initial loss spike (52.6 at epoch
  0) though it did converge reasonably by 40 epochs in earlier smoke
  testing (`val_recon~0.012`). Full 200-epoch Stage-1 log exists but
  **no Stage 2 was completed** (`stage2_section80_*.log` does not exist)
  -- per the user, "the run doesn't look very good," and the direction
  was revised to Section 81 (swap `ConservedFourierMLP` for
  `FourierIFFTBody`) before a real Stage-2 run was launched.

## Section 81: ViT + `FourierIFFTBody` hybrid, "only using the frequency components" -- **the architecture-comparison headline result**

Same hybrid summation structure as 80, but the Fourier branch is now
`FourierIFFTBody` (Section 75's own frequency-domain mechanism: an MLP
predicts frequency-domain coefficients, explicitly inverse-transformed via
`irfft` back to state space) instead of `ConservedFourierMLP` -- and with
NO raw-value/masked path at all in this branch (unlike Section 75's AE,
which sums a masked raw-value path alongside its own `FourierIFFTBody`
readout). `d_model=92`, `fourier_hidden=270`/`fourier_blocks=2`, sized to
hit ~1.5M total AE params (755,834 ViT + 753,604 Fourier branch =
1,509,438 measured directly from the checkpoint). Propagator: Section
75's exact `fourier_mlp` propagator, completely unchanged (~1.0M params).
Regularizers/curriculum: Section 75's exact recipe and Stage-2 curriculum
(`k_max=12`, 300 epochs).

- Stage 1: `val_recon_final = 0.000049` (comparable to 75's 0.000035, same
  order of magnitude).
- Stage 2: **`best_val_kmax_mse = 0.028618`** -- meaningfully better than
  Section 75's `0.039243` at the identical `k_max=12` curriculum (both
  fully converged at epoch 298-299/300).
- Latent spectrum: `cond#=22.3` (essentially matching Section 75's
  20.98), `top_eig=10.1`, participation ratio 14.3/44, D8 signed
  bandedness=0.341 (essentially identical to Section 75's 0.341) --
  encoder geometry is very similar to Section 75's despite the different
  architecture.
- **Gate 3 full analysis suite**: `D_KY = 20.914`, `n_positive = 12`,
  `lambda1 = 0.0917` -- essentially matching Section 75's `D_KY=21.089`
  (both ~4-5% below the true benchmark `D_KY~22`), i.e. Lyapunov-spectrum
  fidelity is preserved, not traded away for the better rollout MSE.
- **Gate 3 DA skill** (`run_da_pff.py`): `rmse_da=0.7096`,
  `rmse_free=0.9259`, `spread=0.0880`,
  `calibration_spread_over_rmse=0.1241`, `skill_free_over_da=1.3049` (DA
  meaningfully beats free-running, as expected).
- **Gate 4 diagnostics**: D3 p-value=0.4480 (no anomalous structure
  flagged), **D4 verdict: "Genuine (approximately) equivariant
  translation representation found"**, D6/D7/D8 p-values all 0.0000
  (strong, significant same-time coupling/bandedness structure, as with
  every other section in this family).
- **Caveat raised by the user and not yet resolved**: "we don't really
  know if the hybrid is the improvement, the models did get larger"
  (1.509M vs 1.002M params, a 1.5x size difference) -- see Section 83.

## Section 82: Section 81's architecture at `d_latent=56`, `w_spatial x1.5`, `lambda_z x1.5`

Same ViT+`FourierIFFTBody` hybrid AE and `fourier_mlp` propagator family
as 81, three changes: `d_latent` 44->56 (`prop_attn_window` raised
22->28, the new saturation point, verified mask-all-True), `w_spatial`
0.01->0.015 (signed, 1.5x), `lambda_z` 0.0002->0.0003 (1.5x). Motivation
(user): "I think the next test then, assuming 81 looks better than 75, is
to increase w_spatial for 81 to .015 and lambda_z to .0003 ... we could
try this with embedding dimension 56" -- explicitly gated on 81 beating
75, which was confirmed mid-training (81 vs 75 at matched curriculum
point k_now=7: 0.035 vs 0.053) before this run was launched. AE
1,552,374 params (ViT 792,278 + fourier_encoder 379,948 + fourier_decoder
380,148), propagator 1,029,610 params.

- **Status at last check: Stage 1 in progress, epoch 130/200,
  `recon=0.000103`, trending down smoothly** (no Stage 2 results yet).
  This section was still running when this document was last updated --
  check `artifacts/logs/stage1_section82_dlatent56_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wspatial015_lambdaz0003_200ep.log`
  (and the corresponding `stage2_..._warmstart_k12_300ep.log` once Stage 2
  starts) for current numbers.
- 4-epoch smoke tests before the real launch showed `val_recon_final =
  0.006143` (Stage 1) and `val_kmax_mse = 0.024543` (Stage 2, 4 epochs)
  -- an even stronger early signal than Section 81's own smoke result
  (0.0336), though smoke-test trajectories are not reliable predictors of
  final converged results (see Section 83's own smoke-vs-real
  discrepancy below).

## Section 83: controlled size comparison -- grow Section 75's ORIGINAL architecture to Section 81's param count

Directly answers the "models did get larger" confound from Section 81.
Section 75's architecture, completely unchanged (masked `fourier_mlp` AE,
`attn_window=8/22/22`, `fourier_ifft_readout`, Section 75's exact
propagator and regularizer recipe) **except** `fourier_mlp_hidden`
224->282 to hit Section 81's exact param count (1,509,368 measured vs.
81's 1,509,438 -- ratio 1.0000). If this run's `best_val_kmax_mse` still
trails Section 81's 0.0286, that is evidence for the hybrid architecture
itself; if it closes the gap or matches, capacity/size was the real
explanation for 81's improvement over 75.

- **Status: queued, not yet launched.** A chain-waiter process is armed
  to launch this automatically once Section 82's full pipeline (Stage
  1+2) finishes running. Check for
  `artifacts/logs/stage1_section83_fouriermlp_1_5m_hidden282_section75regs_200ep.log`
  to see if it has started.
- A 4-epoch smoke test before queuing showed `val_recon_final=0.014574`
  (Stage 1) and `val_kmax_mse=0.188` by epoch 3 (Stage 2, 4 epochs) --
  notably WORSE than both Section 75's own smoke trajectory at this size
  and Section 81's hybrid smoke trajectory, an early (not conclusive at
  just 4 epochs) signal that scaling up this specific architecture's raw
  capacity alone does not help and may hurt, consistent with Sections
  77/78's earlier finding that more capacity within the same
  masked-`fourier_mlp` family does not straightforwardly improve fit.
  **Not yet confirmed with the real 200+300-epoch run.**

---
