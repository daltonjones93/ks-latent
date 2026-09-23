# ks-latent

Latent-space data assimilation and manifold analysis for the 1D Kuramoto-Sivashinsky
(KS) equation. See `CLAUDE_CODE_BRIEF.md` for the full build spec and
`docs/PROJECT_HANDOFF.md` / `docs/ML_for_KS_writeup.md` / `docs/LATENT_PDE_RESEARCH_NOTES.md`
for the scientific background this reproduces and extends.

## Status

Phase 0-1 complete: repo skeleton, device policy, ETDRK4 solver, dataset
generation, and the raw-PDE Kaplan-Yorke dimension. **Gate 1 passed** --
see `docs/REPLICATION_LOG.md`.

Phase 2 complete: D7 information-spreading velocity / light-cone bounds.
**Gate 2 passed** -- `v_star = 1.261 +/- 0.039` (SEM, L=100), derived
stencil/localization bounds written to `docs/RESULTS.md`.

**Phase 3: replication target missed at the brief's original defaults,
root-caused, and resolved by changing the canonical recipe (ground rule
6).** The full-scale Stage-1 training run (60 epochs, brief's original
NX=1024/`mode="two_step"`/`dt_snap=0.25`, real data) **converged to a
representation collapse** -- the encoder maps nearly every input to the
same latent vector, val reconstruction stuck at 0.969 (essentially the
"predict-the-mean" baseline). Six ablations (data, loss terms, shift
augmentation, grad clipping, learning rate, decoder-at-init sensitivity)
reproduced the identical collapse, ruling each out as the sole cause.
Follow-up debugging (user-directed) found NX=1024 itself never escapes the
collapse within any epoch budget tried, while **NX=256 does** -- a
2x2 factorial over `dt_snap` x propagator `mode` then isolated `dt_snap`
as the variable that measurably speeds up escaping the collapse (not
`mode`, which was chosen anyway for physical-modeling reasons: the KS PDE
is first-order in time, so `two_step`'s two-history-snapshot design is an
unmotivated crutch). **Canonical recipe changed, 2026-08-29** (documented
in `CLAUDE_CODE_BRIEF.md` §5.1 addendum and `docs/RESULTS.md`): `NX=256`,
`mode="markovian"`, `dt_snap=1.0` -- config defaults and all training
scripts updated accordingly (catching and fixing two real bugs along the
way: two analysis/DA scripts were hardcoding the old `dt_snap=0.25` for
physical-time conversions). Gate 3/4 training at the new canonical scale
is the immediate next step.

Phase 4 complete: analysis suite extended with correlation dimension,
diffusion maps, the full d in {2,5,10,15,20,22} sphere/torus validation
sweep, the `single_state`/`two_step` latent-propagator Lyapunov adapters
(exact `torch.func.jvp`-based tangent map, validated to 1e-3 against a
hand-built linear recursion's companion-matrix spectrum), and persistent
homology (plain Rips + DTM filtration) validated on circle/torus/2-sphere
with and without outliers. `scripts/run_analysis_suite.py` ties it together
once Stage-1/2 checkpoints exist.

Phase 5 complete: Natural-Gradient Particle Flow Filter
(`ks_latent/da/pff.py`) -- an independently-derived preconditioned-Langevin
algorithm (the exact original formula wasn't recoverable from the source
docs; see `docs/RESULTS.md`/`docs/OPEN_QUESTIONS.md` for the full account),
verified against the brief's stated correctness bar: exact Kalman recovery
for a linear observation operator (`test_pff_gaussian_linear`, ~2% at
N=8000). Found and fixed a real numerical-stability bug along the way
(naive Euler-Maruyama blew up the analysis covariance by >500% once the
adaptive step schedule grew past ~2; fixed with an exact
Ornstein-Uhlenbeck/exponential integrator). Full DA cycling driver
(`ks_latent/da/cycling.py`) with the brief's exact spacetime/RMSE
conventions, validated to measurably beat a free run on a controlled
synthetic system. `scripts/run_da_pff.py` runs the real replication
experiment once Stage-1/2 checkpoints exist.

Phase 6 complete: structure diagnostics D1-D5
(`ks_latent/analysis/diagnostics.py`) -- sensitivity maps with circular
statistics, wavenumber content plus a wavelet-based joint space-scale
localization measure, the Jacobian coupling graph with spectral seriation
and a permutation-test p-value (validated: p<0.01 for a genuinely banded
matrix, p>0.05 for dense random, at 1000 nulls), the closed-form
translation-representation fit (validated to <1e-5 on an exactly
equivariant synthetic encoder), and local-dimension-vs-patch-length.
`scripts/run_diagnostics.py` emits `docs/diagnostics_report.md` (Gate 4);
already run once in `--profile smoke` and surfaced a real, sensible
qualitative finding (see `docs/RESULTS.md`) even on the tiny toy model.

Phase 7 (partial): sampling-error-correction localization
(`ks_latent/da/sec.py`, brief §9) -- Anderson (2012) Monte Carlo SEC table
plus a fixed-empirical-taper alternative, both exposed as a single
`Callable[[Tensor],Tensor]` `localize_fn` interface plugged into
`ParticleFlowFilter`/`CycleConfig`, validated both in isolation (correct
shrinkage direction/magnitude) and end-to-end (SEC-localized small-ensemble
DA analysis measurably closer to the true-prior Kalman answer than the raw
prior). **Not yet done:** the brief's Gate 5 deliverable, the `N_ens`
sweep script (with/without SEC, plotted, extrapolation argument written
down).

**All of Phases 3-7's core library code is built and tested (149 fast
tests) and independently validated on synthetic ground truth. Gate 3 (the
full §18 replication table) and Gate 4's real numbers are both blocked on
training a real (non-collapsed) Stage-1/Stage-2 checkpoint pair at the new
canonical recipe above** -- Phases 4-6's own machinery, and Phase 7's
localization, are ready to run the moment that checkpoint exists.

## Environment

This project uses the existing `da_env` mamba environment; do not create a
new one.

```bash
mamba activate da_env
python scripts/check_env.py       # verify every import; installs missing packages as printed
pip install -e . --no-deps        # editable install so `import ks_latent` works anywhere
```

Packages not on conda-forge (`giotto-tda`, `pysindy`) were installed via
`pip install giotto-tda pysindy` inside `da_env`. `pip` did not need
`--break-system-packages` here (conda envs are not externally managed); the
brief flags that flag as a possible requirement on some setups. One
side effect worth knowing about: installing `giotto-tda` downgraded
`numpy` (2.4.6 -> 1.26.4) and `scikit-learn` (1.9.0 -> 1.3.2) via pip's
resolver; both still work fine with everything else in the environment.
`pysindy` additionally needed `setuptools<81` pinned (it imports the
now-removed-by-default `pkg_resources`).

After any environment change: `mamba env export -n da_env --no-builds >
environment.lock.yml` and commit it. `environment.yml` is a from-scratch
portability fallback only -- not used for actual work in this repo.

## Running tests

```bash
make test-fast   # unit + integration, excludes @pytest.mark.slow -- CI tier
make test        # everything, including slow replication tests
make smoke       # integration tier only
make replicate   # slow replication tests + regenerates docs/REPLICATION_LOG.md
```

Every experiment script (once they exist, from Phase 3 on) supports
`--profile smoke`: the full code path in under 60s on CPU at tiny
dimensions.

## Layout

See `CLAUDE_CODE_BRIEF.md` §2 for the intended full layout. As of Phase 1:

- `ks_latent/config.py` -- `KSConfig` and config-hashing.
- `ks_latent/utils/` -- `device.py` (MPS/CPU routing table), `linalg.py`
  (CPU float64 `robust_eigh`/`robust_cholesky`), `seeding.py`, `io.py`
  (provenance sidecars), `replication.py` (§18 result recording).
- `ks_latent/solver/` -- `ks.py` (ETDRK4 solver + tangent/variational
  propagator for Lyapunov exponents), `filtering.py` (Gaussian low-pass),
  `dataset.py` (`TrajectoryDataset` / `AttractorPointDataset` generation).
- `ks_latent/analysis/` -- `lyapunov.py` (generic Benettin/QR spectrum,
  reused across the raw PDE, the trained latent propagator via
  `single_state`/`two_step` `torch.func.jvp` adapters, and Phase 12's
  gauge-invariant comparator), `dimension.py` (two-NN, correlation
  dimension, diffusion maps, sphere/torus sampling), `topology.py`
  (persistent homology, plain Rips + DTM filtration), `spreading.py` (D7:
  comoving-exponent and front-tracking light-cone velocity estimators, plus
  the light-cone/stencil/localization bound formulas reused in Phases 11
  and 13).
- `ks_latent/models/` -- `autoencoder_patched.py` (Stage-1
  `KSAutoencoderPatched`), `propagator.py` (`LatentPropagator`,
  `AuxPropagator`, shared `ResidualMLPBlock`).
- `ks_latent/training/` -- `losses.py` (recon/pred/decorr/var,
  horizon-weighted MSE), `data.py` (on-device windowing + shift
  augmentation, brief §1.3.4), `loops.py` (`train_stage1`, `train_stage2`).
- `ks_latent/da/` -- `pff.py` (`ParticleFlowFilter`, NAT-PFF), `forecast.py`
  (`forecast_ensemble`, model-noise injection), `cycling.py`
  (`run_da_experiment`, the full cycling driver), `rmse.py`
  (`per_dim_rmse`, shared by free/DA runs).
- `ks_latent/analysis/diagnostics.py` -- D1 (`decoder_sensitivity_map`,
  `encoder_sensitivity_map`), D2 (`wavenumber_content`), D3
  (`coupling_graph_diagnostic`, `jacobian_coupling`, `nonlinear_coupling`,
  `bandedness_p_value`), D4 (`translation_representation`), D5
  (`local_dimension_vs_length`).
- `ks_latent/utils/circstats.py` -- `circular_weighted_stats` (D1's
  periodic-domain-aware centroid/spread).
- `scripts/train_stage1_patched.py`, `scripts/train_stage2_patched.py`,
  `scripts/bench_device.py`, `scripts/run_analysis_suite.py`,
  `scripts/run_da_pff.py`, `scripts/run_diagnostics.py` -- Phase 3-6 entry
  points, each with `--profile smoke`.
- `tests/unit/`, `tests/replication/` -- see below.
- `scripts/check_env.py`, `scripts/generate_replication_log.py`.

## Known deviations / open items

See `docs/OPEN_QUESTIONS.md`.
