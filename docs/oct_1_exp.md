# Oct 1 2026 experiment plan — does the Kolmogorov local propagator need genuine long-range coupling?

## 0. Background and the question being tested

Section 231/233's Kolmogorov-flow propagator (`backbone="local_mlp_2d"`, trained via
`scripts/train_kolmogorov_localfield_standalone.py`) is a weight-shared circular-conv
site-mixer: receptive field = `n_layers * window` sites. The deployed Stage-2 propagator
(`AuxPropagatorConfig(local_mlp2d_n_layers=2, ...)`, hardcoded in the training script) with
`--attn-window 3` (the CLI-exposed default) gives a receptive field of `2*3 = 6` sites on
the trained `n_sites_x=n_sites_y=8` ring — **75% of the domain**. The encoder side
(`site_mix_radius=2, n_site_mix_layers=3`, also hardcoded) gives `3*2 = 6` sites, also 75%.
Neither is a small local neighborhood; both are closer to "nearly every site can reach
nearly every other site in one forward pass" than to a genuine PDE stencil.

The user's hypothesis, stated directly: 2D incompressible Kolmogorov flow is driven by
local differential relationships (`omega_t + u.grad(omega) = nu*lap(omega) - alpha*omega +
f(y)`), so a genuinely local, tileable model *should* work the same way 1D KS's did — and
the reason it hasn't is that we are not modeling the interaction between tiles accurately
enough (i.e. the model's effective coupling range is too narrow, or the wrong kind, to
capture what 2D flow actually needs).

Checked against the repo and the physics, both halves of that are true, with one added
wrinkle:

1. **We are not currently testing "genuinely local."** The 75%-of-domain receptive field
   above means every existing Kolmogorov result (G2/G2b/G3/T1, including the alpha=0.1 drag
   fix) was produced by a model that, at *train* size, already had near-global reach. We
   have not yet separated "the drag fix works because it screens the long-range coupling"
   from "the drag fix works and the model's near-global coverage was doing some of the
   work too." Experiment D1 below isolates this.
2. **The physical reason 2D flow needs long range, that 1D KS does not, is a specific,
   known term: the elliptic streamfunction/velocity recovery `lap(psi) = -omega`, `u =
   (psi_y, -psi_x)`.** In Fourier space this is a `1/k^2` divide — an exact, instantaneous,
   distance-independent coupling between vorticity at every scale and velocity everywhere
   in the domain. 1D KS has no such term (`u_t = -u u_x - u_xx - u_xxxx` is built entirely
   from local derivatives). This is the one piece of the physics that *cannot* be made
   local no matter how wide a learned window is used, and it is strongest at small `k`
   (large scale) — exactly where the inverse cascade pushes energy and exactly what linear
   drag `alpha` screens.
3. **Linear drag is therefore best understood as *screening*, not *removing*, the
   nonlocal term.** It shortens the correlation length `ell_alpha ~ (eps/alpha^3)^(1/2)`
   below the domain size, so a local model can get away with not representing the `1/k^2`
   coupling explicitly, as long as `ell_alpha` stays short relative to `L`. That is
   consistent with G2's own finding that the alpha=0.1 fix was flat to 2x but drifted again
   at 3x.

This document plans three experiments, in strict sequence, each one run only if the
previous one's pre-registered decision rule says it is needed:

- **D1 (diagnostic, cheap, no new code):** retrain Section 233's exact recipe with a
  genuinely sub-domain propagator receptive field and see whether alpha=0.1 transfer
  survives. Tells us whether the near-global coverage above was load-bearing.
- **D2 (targeted architectural fix, moderate new code):** inject the *exact*, non-learned
  elliptic velocity-recovery term as an extra input to the local mixer, computed by FFT at
  whatever grid size the model is evaluated at (so it is correct and resizable by
  construction, like the gauge-anchor channel already is). Tests whether supplying the one
  provably-nonlocal piece of the physics directly removes the need for drag at all (i.e.
  works at the alpha values where even a broad-receptive-field local model failed).
- **D3 (general architectural fix, largest new code):** extend the already-drafted
  FMM/multiscale-pyramid idea (`docs/spatially_structured_latent_DA_FMM_research_idea.md`,
  `ks_latent/models/autoencoder_multiscale_local_field.py`, currently 1D-autoencoder-only)
  to a genuinely multiscale 2D *propagator*, approximating long-range coupling without
  assuming a closed-form elliptic operator exists — the thing to reach for once Kolmogorov
  flow's known fix (D2) has been tried, if a PDE-agnostic version of the same idea is
  wanted (every chaotic PDE with an elliptic constraint — Boussinesq convection, the full
  Navier-Stokes pressure projection, quasi-geostrophic flow — has the same problem, but
  not always a `1/k^2` closed form as clean as 2D vorticity's).

Gating: D2 is implemented only if D1's decision rule says the near-global coverage was
load-bearing (or if D1 passes but we want to push past drag's known limits, e.g. working
at the original alpha=0.05 or an even bigger domain, as an upgrade rather than a repair).
D3 is implemented only if D2 fails, is architecture/PDE-specific in a way that defeats its
purpose for the broader project, or if D1+D2 together leave an unexplained residual gap.

---

## 1. Experiment D1 — diagnostic: train with a genuinely sub-domain receptive field

### 1.1 Design

Single-variable change from Section 233's exact recipe: reduce the propagator's
`attn_window` from 3 to **1**. Everything else identical: `n_sites_x=n_sites_y=8`,
`local_channels=3`, `alpha=0.1` dataset
(`artifacts/datasets/kolmogorov_trajectories_alpha0.1.h5`), encoder architecture
unchanged (`site_mix_radius=2, n_site_mix_layers=3`, both hardcoded in the training
script, left as-is), `backbone="local_mlp_2d"`, 200 Stage-1 epochs / 300 Stage-2 epochs,
seed 0.

This changes the propagator's receptive field from `2*3=6` sites (75% of the trained
8-site ring) to `2*1=2` sites (**25%** of the ring) — a single, cleanly isolated knob,
touching only the dynamics model, not the encoder/decoder. `d_latent` is unchanged
(`8*8*3=192`), so this reuses the *exact* same Lyapunov/transfer cost profile as every
existing T1/G2/G3 run — no new compute-cost risk, no need to recompute truth D_KY (the
existing `artifacts/experiments/t1/truth_x{1,2,3}_seed{0,1,2}.json` apply unchanged, since
alpha, Lx, Ly, NX, NY are all untouched).

No new code is required: `--attn-window` is already an exposed CLI flag on
`scripts/train_kolmogorov_localfield_standalone.py`, and `scripts/exp_g3_kolmogorov_alpha_transfer.py`'s
resize logic already assumes exactly this `(n_sites_x=8, NX=128)` trained shape, unchanged
by this experiment.

A single `window=1` point is a deliberate choice over a sweep, to keep this cheap; if the
result is ambiguous (see decision rule), the natural next point is `window=2` (50% of the
ring) before concluding anything, not a full sweep.

### 1.2 Tag and artifacts

- Tag: `section235_kolmogorov_alpha0p1_localmlp2d_window1`
- Stage-1 AE: `artifacts/stage1_ae_kolmogorov_full_section235_kolmogorov_alpha0p1_localmlp2d_window1.pt`
- Stage-2 propagator: `artifacts/stage2_prop_kolmogorov_full_section235_kolmogorov_alpha0p1_localmlp2d_window1_warmstart_k12_300ep.pt`
- Transfer results: `artifacts/experiments/g3_kolmogorov_alpha_transfer_d1_window1_x{1,2,3}_ic{0,1,2}.json`
  (x1 only needs `ic0`, as a native-size collapse sanity check, matching T1's own economy)
- Summary: `artifacts/experiments/d1/d1_summary.json`

### 1.3 Commands (orchestrated by `scripts/exp_d1_sparse_window_transfer.py`, see §1.4)

```bash
# Train (resumable: --resume skips Stage 1 if its checkpoint already exists)
python scripts/train_kolmogorov_localfield_standalone.py \
  --dataset artifacts/datasets/kolmogorov_trajectories_alpha0.1.h5 \
  --tag section235_kolmogorov_alpha0p1_localmlp2d_window1 \
  --seed 0 --stage1-epochs 200 --stage2-epochs 300 --checkpoint-every 20 \
  --attn-window 1 --resume

# Transfer evaluation (x1 sanity check + x2/x3 at 3 IC seeds each, matching T1's own grid)
python scripts/exp_g3_kolmogorov_alpha_transfer.py \
  --ae-ckpt artifacts/stage1_ae_kolmogorov_full_section235_kolmogorov_alpha0p1_localmlp2d_window1.pt \
  --prop-ckpt artifacts/stage2_prop_kolmogorov_full_section235_kolmogorov_alpha0p1_localmlp2d_window1_warmstart_k12_300ep.pt \
  --x-scale {1,2,3} --alpha 0.1 --seed {0,1,2} --tag d1_window1_x{size}_ic{seed}
```

### 1.4 Orchestration script

`scripts/exp_d1_sparse_window_transfer.py` (new), mirroring `scripts/exp_t1_error_bars.py`'s
conventions exactly: single-threaded BLAS/FFT/torch env vars set before any heavy import,
`os.nice(19)`, a lockfile so this can never run concurrently with T1 or anything else, and
a `(name, done_fn, run_fn)` task list so the whole thing is preemptable — rerunning the
same command after a kill just skips whatever already has output on disk. Tasks, in order:

1. `train` — runs the training command above; skipped if the Stage-2 checkpoint exists.
2. `eval x1 ic0` — the native-size collapse sanity check.
3. `eval x2 ic{0,1,2}`, `eval x3 ic{0,1,2}` — the actual transfer test, 3 seeds each (same
   discipline as T1, since a single-seed transfer number is not trustworthy per this
   project's own IC-robustness precedent).
4. `summarize` — loads all of the above plus the existing truth means from
   `artifacts/experiments/t1/t1_summary.json` (or directly from
   `artifacts/experiments/t1/truth_x{size}_seed{seed}.json` if the summary file is stale)
   and Section 233's own existing `m0` numbers
   (`artifacts/experiments/g3_kolmogorov_alpha_transfer_section233_{native,2x,2x_seed1,2x_seed2,3x}.json`),
   computes D1's relative transfer error at each size, and prints/writes the comparison
   table plus the decision-rule verdict below.

### 1.5 Pre-registered decision rule (written before running, per project ground rule 6)

Let `err_D1(x)` = D1's mean relative D_KY transfer error at size `x` (2 or 3), and
`err_233(x)` = Section 233's (`window=3`, `m0`) existing relative error at the same size
(already measured: ~4.5-6% at x2 across 3 model seeds' worth of IC checks, ~18-19% at x3).

- **PASS ("coverage was not load-bearing; drag alone explains the transfer fix"):**
  D1's x1 check shows no collapse (positive-exponent count and D_KY within the same
  order of magnitude as Section 233's native value — i.e. this is still a working
  chaotic model, not a degenerate one), **and** `err_D1(2) <= err_233(2) + 3 percentage
  points` **and** `err_D1(3) <= err_233(3) + 5 percentage points`. If this holds, the
  25%-of-domain model transfers about as well as the 75%-of-domain one, which means the
  broad coverage in every prior Kolmogorov result was incidental, not necessary — the
  linear-drag screening mechanism is doing the real work. D2/D3 become optional
  extensions (pushing past drag's own limits) rather than required repairs.
- **FAIL ("coverage was load-bearing"):** D1 collapses at x1 (near-fixed-point, D_KY near
  0), **or** `err_D1(2)` / `err_D1(3)` exceed the PASS thresholds above by a wide margin
  (say, more than double `err_233` at the same size). This would mean the near-global
  reach in every existing result was doing real work beyond what drag screening alone
  explains, and a genuinely local model needs either the exact nonlocal term (D2) or an
  approximate multiscale one (D3) to match it.
- **AMBIGUOUS:** errors worse than PASS but not as bad as FAIL (roughly 1-2x `err_233`).
  In this case, run the `window=2` intermediate point (50% of ring) before concluding
  anything, rather than treating a single ambiguous data point as a verdict.

### 1.6 What D1 does *not* test

D1 narrows the propagator's own mixing window but changes nothing about what information
is *available* to it — the encoder's receptive field, and therefore what each site's
latent vector can "see" about its physical neighborhood, is unchanged. If D1 fails, it is
evidence the *dynamics* need wider reach, not direct evidence about representation
capacity. This is deliberate (isolating the dynamics question is the whole point) but
should be stated plainly in the writeup rather than overclaimed.

---

## 2. Experiment D2 — exact elliptic (streamfunction/Biot-Savart) feedback term

**Only implemented if D1 is FAIL or AMBIGUOUS-after-window=2, or if D1 is PASS and the
goal shifts from "repair alpha=0.1 transfer" to "remove the need for drag at all."**

### 2.1 Design

2D vorticity-velocity recovery (`lap(psi) = -omega`, `u = (psi_y, -psi_x)`) is linear,
diagonal in Fourier space, and known in closed form: `psi_hat(k) = omega_hat(k) / |k|^2`
(with the `k=0` mode set to zero — mean vorticity is dynamically irrelevant and the
operator is singular there). This is the one term in the whole problem that genuinely
cannot be made local, and it is known exactly, so it does not need to be learned at all —
only evaluated, at whatever grid the model happens to be running on. Because `k = 2*pi*n/L`
is defined in physical units, this evaluates correctly at any domain size by construction,
the same resizability argument already used for the gauge-anchor channel.

Implementation, in `ks_latent/models/propagator.py`:

1. **New propagator backbone `"local_mlp_2d_biotsavart"`**, a new class
   `_LocalMLPDeltaBody2DBiotSavart` mirroring `_LocalMLPDeltaBody2D` exactly (same
   `_LocalMixerBlock2D` stack, same zero-initialized `token_unembed`, same
   identity-at-init property), with one addition before `token_embed`:
   - Take channel 0 of the current `(Px, Py, local_channels)` token grid (the
     non-learned, gauge-anchored physical average — for Kolmogorov flow this is a local
     block-average of vorticity `omega`).
   - Treat it as `omega` sampled on an `(n_sites_x, n_sites_y)` periodic grid of physical
     size `Lx x Ly`. 2D FFT it, divide by `-|k|^2` (zeroing `k=0`), invert to get `psi_hat`,
     differentiate in Fourier space (`u_hat = i*ky*psi_hat`, `v_hat = -i*kx*psi_hat`),
     inverse FFT to real `(u, v)` fields at site resolution.
   - Concatenate `(u, v)` as two extra fixed (non-learned, non-trainable) channels,
     projected through a small `Linear` into `token_d_model` **in addition to** (summed
     with, matching `_LocalMLPDeltaBody2D`'s existing `token_embed`) the usual
     `local_channels`-based embedding, before the usual mixer stack runs unchanged.
   - This is a deterministic function of the current state, differentiable through (plain
     FFT ops), and does not touch the final zero-initialized output projection, so
     identity-at-init is unaffected.
2. **Config.** Add `biot_savart_Lx: float = 2*pi`, `biot_savart_Ly: float = 2*pi` to
   `AuxPropagatorConfig`/`PropagatorConfig` (used only by this backbone), and wire the new
   backbone name into both configs' validation and `LatentPropagator`'s/`AuxPropagator`'s
   dispatch (~`propagator.py:4099-4108`, alongside the existing `"local_mlp_2d"` /
   `"local_mlp_2d_aniso"` branches).
3. **Resize loader.** `ks_latent/models/__init__.py`'s `load_propagator_checkpoint_resized_2d`
   must also rescale `biot_savart_Lx`/`biot_savart_Ly` by the same ratio it already applies
   to `n_sites_x`/`n_sites_y` (i.e. `new_Lx = old_Lx * (new_n_sites_x / old_n_sites_x)`,
   matching `exp_g3`'s own convention of scaling `Lx` and `n_sites_x` together) — otherwise
   the Fourier `k`-grid used inside the new backbone would silently be wrong after a
   resize, which would be a far worse failure mode than loudly raising.
4. **Training.** The autoencoder is untouched by this change (only the propagator's
   *input*, not anything about how `u` is encoded/decoded, changes), so **Section 233's
   existing Stage-1 AE checkpoint (seed 0) is reused as-is** — only Stage 2 (the
   propagator) needs retraining. Requires a small addition to
   `scripts/train_kolmogorov_localfield_standalone.py` (or a dedicated
   `scripts/train_kolmogorov_biotsavart_standalone.py`): a `--reuse-ae-ckpt <path>` flag
   that loads a given AE checkpoint and skips Stage 1 entirely, rather than training a new
   one under the same tag.

### 2.2 Tests (new, before any training run per project ground rule 1)

- `test_biot_savart_feedback_translation_equivariant` — shifting the input field by whole
  patches shifts the computed `(u, v)` feedback (and hence the delta) by the same amount,
  to the same tolerance the existing `test_circular_conv_exact_equivariance`-style tests use.
- `test_biot_savart_k0_mode_zeroed` — assert the `k=0` component of `psi_hat` is exactly
  zero regardless of the mean of the input field (the operator is singular there; silently
  leaving it as `0/0 -> nan` would violate ground rule 2, fail loudly otherwise).
- `test_biot_savart_identity_at_init` — `zero_init=True` still gives the exact identity map
  at init, same as every other backbone.
- `test_biot_savart_resize_rescales_Lx` — loading a checkpoint resized to `x_scale=2`
  produces a `k`-grid matching `Lx_new = 2*Lx_old`, checked against a direct
  `np.fft.fftfreq`-based reference.
- `test_biot_savart_matches_analytic_taylor_green` — on a synthetic single-mode
  (Taylor-Green-like) vorticity field with a known analytic streamfunction/velocity, the
  computed `(u, v)` must match the closed-form answer to near machine precision. This is
  the one test that proves the elliptic solve itself is implemented correctly, independent
  of anything about training — ground rule 1, "no untested numerics."

### 2.3 Evaluation and decision rule

Run the same `exp_g3`-style transfer sweep, but the primary test is **not** re-proving
alpha=0.1 works (D1 may have already shown that needs no fix) — it is whether this
backbone recovers transfer **at alpha values where even the broad-receptive-field
(window=3) local model already failed**, i.e. the original alpha=0.05 dataset (G2's own
~50% x2 transfer error) or, if that succeeds, alpha=0 (no drag at all, the hardest case,
where the inverse cascade runs unchecked). Re-run the same x1 collapse sanity check, x2/x3
at 3 IC seeds, same decision-rule structure as D1 (collapse check, then relative error vs.
the existing un-fixed alpha=0.05 numbers as the baseline to beat, rather than vs. Section
233's already-fixed alpha=0.1 numbers).

**Pre-registered success criterion:** D2 is a confirmed fix if, at alpha=0.05, its x2
transfer error drops from G2's own ~50% down to single digits (matching what drag-at-0.1
already achieves by a different mechanism) *without* needing drag increased to 0.1 to get
there. A result that only works at alpha=0.1 (where D1 may have already shown a plain
local model suffices) would not be informative about whether the elliptic feedback is
doing anything D1 doesn't already explain.

---

## 3. Experiment D3 — multiscale/FMM-style propagator pyramid

**Only implemented if D2 fails, or succeeds but in a way judged too PDE-specific (the
closed-form `1/k^2` operator is particular to 2D vorticity-velocity recovery; other
long-range-coupled PDEs this project may eventually touch — Boussinesq convection, 3D
Navier-Stokes pressure projection, quasi-geostrophic flow — have an analogous elliptic
constraint but not always as clean a closed form), or if a PDE-agnostic version of the
same fix is wanted on its own merits.**

### 3.1 Design

Extend the already-drafted FMM/multiscale idea
(`docs/spatially_structured_latent_DA_FMM_research_idea.md`, and its already-implemented
1D-autoencoder-only instantiation `ks_latent/models/autoencoder_multiscale_local_field.py`,
`MultiscaleLocalFieldAutoencoderConfig`) in two ways neither of those currently cover:

1. **2D, not 1D.** `autoencoder_multiscale_local_field.py` is `Conv1d`-based
   (`ks_latent.config.MultiscaleLocalFieldAutoencoderConfig`, fine/regional/global scales
   over a 1D KS field). A new `ks_latent/models/autoencoder_multiscale_local_field_2d.py`
   /`MultiscaleLocalFieldAutoencoderConfig2D` ports the same fine -> `AvgPool1d` ->
   regional -> `AvgPool1d` -> global pattern to `Conv2d`/`AvgPool2d`, mirroring how
   `autoencoder_local_field_2d.py` already ports the single-scale `autoencoder_local_field.py`.
2. **The propagator, not just the autoencoder.** Neither existing multiscale file touches
   dynamics — they are compression-only. A new propagator backbone
   `"multiscale_local_mlp_2d"` runs a `_LocalMixerBlock2D` stack **at each scale**
   independently (fine/regional/global, each own weights, same `_LocalMixerBlock2D` class
   reused across scales as it already is across sites), with a lift-pool/broadcast-upsample
   exchange between adjacent scales before and after each scale's own mixing pass
   (reusing `autoencoder_multiscale_local_field.py`'s own pool/lift/sum pattern, ported to
   2D), so information genuinely moves between scales during a single propagator step, not
   just during encoding.

**The one design rule that makes this a genuine FMM analogue rather than "one more coarse
layer":** pooling factors are fixed in **physical units**, not site counts, so that
transferring to a larger domain adds *more pyramid levels* (depth grows ~`log2(L)`) rather
than stretching an existing level's receptive field past where it was trained — the same
reasoning that makes the single-scale models' site-mixing weight-shared-and-transferable
in the first place, generalized to multiple scales.

### 3.2 Implementation steps (high-level, to be detailed in their own plan once reached)

1. `MultiscaleLocalFieldAutoencoderConfig2D` (config.py) and
   `KSAutoencoderMultiscaleLocalField2D` (new model file) — direct 2D port of the existing
   1D multiscale autoencoder, each scale gauge-anchored from `u` directly (not from a
   lower scale's hidden features) exactly as the 1D version already does.
2. `_MultiscaleLocalMLPDeltaBody2D` (propagator.py) — per-scale `_LocalMixerBlock2D`
   stacks plus the pool/lift/sum cross-scale exchange, zero-init preserved at the final
   per-scale output projections.
3. Resize loader support (`load_autoencoder_checkpoint_resized_2d`-equivalent and
   `load_propagator_checkpoint_resized_2d`-equivalent for the multiscale configs) — the
   part that actually has to add pyramid levels on resize rather than just stretching
   existing ones, so this is the one piece of loader logic that's genuinely new, not a
   copy of the existing resize code.
4. Tests mirroring Section 3 of this doc's D2 tests, generalized to multiple scales, plus
   a new one specific to the design rule above:
   `test_resize_adds_pyramid_level_rather_than_stretching` — resizing past a fixed
   physical-unit threshold must increase the number of active scales, not change any
   existing scale's own pooling factor.
5. Full retrain (both AE and propagator; unlike D2, this is a structurally new
   autoencoder, so Stage 1 cannot be skipped) and the same `exp_g3`-style evaluation
   sweep, at the same alpha values D2 was tested against.

### 3.3 Decision rule

Same structure as D2 (collapse sanity check at x1, relative transfer error vs. the
unfixed-local baseline at alpha=0.05/0), plus a direct head-to-head against D2 if D2 ran
first: does the general multiscale mechanism match D2's exact-physics fix at the one PDE
where the exact fix is known, as a check that the approximation isn't leaving a lot on the
table even where it has no informational advantage?

---

## 4. Summary of what's being launched now

Only **D1** is implemented and launched as part of this plan's own execution — it requires
zero new code, reuses every existing script and the existing truth-D_KY measurements, and
its result determines whether D2/D3 are necessary repairs or optional extensions. D2 and
D3 are planned in enough detail to start immediately once D1's decision rule calls for
them, but neither is implemented yet.
