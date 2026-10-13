# Plan: minimal reproduction repository for Paper I and Paper II

Written 2026-10-13 from a planning-agent survey of this repo (traced from the paper builds, the
table/figure `.meta.json` inputs, `docs/split_paper_tasks.md` Z-blocks, driver scripts, checkpoint
sidecars and result-JSON `args`). Condensed; the decisions and file lists are complete.

## 0. Key decisions

- `artifacts/` is gitignored: today the papers cannot be rebuilt from git. Both App. E say results came
  from a dirty tree on top of `2cd9da0`.
- New repo keeps the package name `ks_latent`, the script file names and a flat `scripts/` (provenance
  mapping, sibling-script imports and pickled checkpoints keep working).
- Artifact tiers:

| Tier | What | Reproduced how | Shipped |
|---|---|---|---|
| A | Final models (§1) | retrained + re-evaluated by the new repo | code in repo; checkpoints on Zenodo |
| B | α = 0.1 selection variants (`k_xs01`, `_v3`, `_v4`, `_v5`, `_v7`) | same code path, optional target | Zenodo |
| C | Historical/appendix ablations (consensus/Biot–Savart backbones, global ViT, section231/233, p2/p9/c7b/c9i/k_a005, 1D global Phase C, pp1d, text-only controls) | not reproducible from the slim repo | frozen results bundle on Zenodo |

- Checkpoints are 70–490 KB each (Tier A+B ≈ 15 MB); results JSON/CSV/figure npz ≈ 350 MB (excl. Benettin
  `.ckpt.npz` ≈ 830 MB); spin-up cache 71 MB; datasets ≈ 93 GB (never shipped, regenerated).
- Freeze first: wait for the 88² truth, the R15 decision and the window-seed evals; commit; tag
  `paper-freeze-2026-10`.

## 1. Final-model registry (stage 3 must retrain)

| Paper | Tag(s) | Dataset | Recipe / driver | Train wall |
|---|---|---|---|---|
| I+II | `section224_ks_localfield_d48_localmlp` (+`_warmstart_k12_300ep`); seeds `exp_e4_seed1/2_*` | `stage1_trajectories_dtsnap1.h5` | `section224_ks_localfield_d48_localmlp.sh`, `exp_e4_seed_robustness_ltransfer.sh <seed>` | 0.4 h + 8.0 h each |
| I | `r13_C2_ks2d_s44big_k48_seed{0,1,2}` | `ks2d_box44_large.h5` | `run_c2.sh` | ≈1 h each |
| I, II | `p7b_ks2d` | `ks2d_wide6.h5` | from checkpoint config: `--crop-sites 40 --crop-sites-y 40 --n-offsets 16 --local-channels 6` + $MIX `--k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000` | 2.1 h |
| I | `k_xs01_v2`, `_s1`, `_s2` | `kolmogorov_wide16_alpha0.1.h5` | `run_kseeds.sh`: $MIX `--k-max 16 --k-mid 8 --noise-end 0.02 --prop-steps 9000 --w-logdet 0.005` | 0.3–0.4 h |
| I | `c19_e0d0_c6`, `_s1`, `_s2` | `kolmogorov_wide16_alpha0.05_orig5.h5` | `run_c19a_p9_p8_v2.sh` / `run_kseeds.sh`: $MIX | 0.2 h |
| I | `k_xs0025`, `_s1`, `_s2` | `kolmogorov_wide16_alpha0.025_orig5.h5` | `run_k_xstrip0025.sh` / `run_kseeds.sh`: $MIX `--k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000` | 0.6–1.2 h |
| I (fail) | `r12_c6_a01_s1x_r3` (+`_seed1/2`) | `kolmogorov_trajectories_alpha0.1.h5` | `run_r12.sh` | 0.2–0.3 h |
| I (fail) | `r4k_c6_a005_s1x_r3`, `r4k_c12d10_a005_s1x_r3` | `kolmogorov_trajectories_default.h5` | `run_r4k_kolmogorov.sh` | 0.1–0.2 h |
| I (fail) | `r14_KC{2,3}_a{0.05,0.1}_seed0` | `kolmogorov_box1x_large_a{α}.h5` | `run_r14.sh` | 0.3–0.4 h |
| I (1 sentence) | `r16_KM_a{0.05,0.1}_seed0` | same | `run_r16.sh` (stage A copied from KC2; hazard E9) | 0.65 h |
| I (A2) | `a2_c19_e0d0_c6_r3` | `kolmogorov_wide16_alpha0.05.h5` | `run_a2_reach3.sh` (stage A copied from c19) | — |
| II | `p7c12d10_ks2d`, `p7c12_ks2d`, `p7c12d30_ks2d` | `ks2d_wide6.h5` | as p7b, `--local-channels 12 --data-noise {0.1,–,0.3}` (E7) | 7.0 / 2.2 / ? h |
| II | `p7c12d10_memB{0.1,0.2}`, `p7c12d10_gani{0.1,0.2,0.3}` | `ks2d_memB*_wide6.h5`, `ks2d_gani*_wide6.h5` | `run_j17b.sh`, `run_j17g.sh` | 2.3–5.1 h |
| II | `k_xs005_c12d10` | `kolmogorov_wide16_alpha0.05.h5` (8+2) | `run_kda_train.sh` | 0.6 h |

$MIX = `--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --local-channels 6`.
All 2D models: backbone `local_mlp_2d` (markovian); 1D: `local_mlp` + `local_field` encoder; R16: `memory_prop`.
Text-only models (`r13_C3_*`, `r4c/w/d/e/f_*`, `p7c12n15`, `pp1d_c6d10`, R15) → Tier C unless a decision says otherwise.

## 2. `ks_latent`: keep / prune

AST import closure from 47 entry scripts + `paper_figures`: 62 of 76 files reached. Drop outright:
`analysis/{diagnostics,dimension,spreading,topology}.py`, `discovery/`,
`solver/{filtering,lorenz96,lorenz96_dataset,rayleigh_benard,rayleigh_benard_dataset}.py`,
`training/crop1d.py`, `utils/{circstats,replication}.py`; `da/model_error.py` (cited in Paper II App. E:
keep with its test or edit App. E). Most of the reachable set comes via barrel imports
(`models/__init__.py`, `train_stage1_patched.py`, `lyapunov.py` → `swe.py`, `propagator.py` →
`autoencoder_vit`/`spectral_field`/`spectral_norm`).

Third-party: numpy, scipy, torch, h5py, matplotlib (+ mpmath for tests; sklearn only Tier C). Not needed:
gudhi, ripser, persim, giotto-tda, pysindy, pywavelets, tqdm.

| Module | Lines | Action |
|---|---|---|
| `config.py` | 5752 | prune → ~800–1000; keep class names (pickles): `KSConfig`, `KSConfig2D`, `KolmogorovConfig`, `LocalFieldAutoencoderConfig(2D)`, `AuxPropagatorConfig`, `PropagatorConfig`, `Stage1/2TrainingConfig`, `RegConfig`, `config_hash` |
| `models/propagator.py` | 4742 | prune → ~600–800: `_LocalMixerBlock(2D)`, `_LocalMLPDeltaBody(2D)`, `LatentPropagator` (markovian), `AuxPropagator`, builders, local attention mask; conditional: history body (likely drop), `_biot_savart_velocity` (only if R15 stays). state_dict keys must not change |
| `models/__init__.py` | 547 | prune → ~200 (local_field / local_field_2d builders and resized loaders) |
| `models/autoencoder_local_field{,_2d}.py`, `kolmogorov_fields.py`, `memory_prop.py` | | keep |
| other `models/*` | ~2400 | drop |
| `training/crop.py` | 716 | keep core (gathers, CropAE/Prop configs, train_crop_ae/prop, encode banks, forced_halo_rollout, logdet, spectral_band_shares); R15 parts only if R15 stays |
| `training/memory.py`, `data.py`, `regularizer.py` | | keep |
| `training/loops.py` | 2399 | prune → ~700–900 (section224 stage 1/2 path only; drop pde-head code) |
| `training/losses.py` | 1354 | prune → ~250 (recon, decorr/var, logdet, signed spatial coherence, horizon weights) |
| `analysis/lyapunov.py` | 712 | keep, drop SWE path |
| `solver/{ks,ks2d,kolmogorov,ks2d_memory,twin,dataset,dataset2d}.py` | | keep (incl. wrong-model operators) |
| `da/{letkf_local,letkf_phys,geometry2d,jacobian_coloring,tilings,forecast,param_est}.py` | | keep (Paper II 2D) |
| `da/{pff,sec,serial_enkf,localization,letkf,cycling,rmse}.py` | | keep (1D DA); drop `var_recenter`, `gated` |
| `utils/{device,io,seeding,linalg}.py`, `__init__.py` | | keep |

Result: ~26.9k → ~9–10k lines. Add CI checks: `tools/check_closure.py` (no unreachable module, all imports
work, `compileall scripts`); coverage-guided function pruning over the smoke run + checkpoint loads;
golden checkpoint-parity test (CPU float64 outputs of every Tier-A checkpoint on a seeded input, bitwise).

## 3. Scripts by stage (commands generated from manifests; original file names kept)

- **Stage 1 data.** New `gen_ks1d_dataset.py` (~30 lines; no generator exists; check config_hash 20a3d7edea7b,
  std 1.3160951909); `gen_ks2d_wide_dataset.py` (`--scale 6 [--memory eps,2,-1 | --ganiso δ] --only i`, seed-base
  900); `gen_ks2d_box_large.py` (144+6, seed-base 1000); `gen_kolmogorov_wide_dataset.py` (α 0.1 4+1; α 0.05 8+2
  and `_orig5` assembled from parts 0–4; α 0.025 `_orig5`; add `--only`); `gen_kolmogorov_box_large.py` (96+6);
  `gen_kolmogorov_dataset.py` (24+6, α 0.05 default and 0.1). Spin-up/twin cache: ship on Zenodo or new
  `make_spinup_cache.py`.
- **Stage 3 training.** `train_kolmogorov_crop.py` (all 2D; keep `--data-noise`, `--periodic-full`, `--memory-*`,
  R15 flags only if kept; drop consensus/Biot–Savart/rank/spectral/y-align); `train_stage1_patched.py` +
  `train_stage2_patched.py` pruned to the section224 path (flags from `section224_ks_localfield_d48_localmlp.sh`).
  Exact per-tag argv in `manifests/models.yaml`, generated from checkpoint configs, logs as fallback; config diff
  after each retrain (as `run_kseeds.sh verify()`).
- **Stage 4 evaluation.** `exp_crop_eval.py`, `exp_crop_eval_ks2d.py` (D_KY grids per driver);
  `run_truth_kolmogorov.py`, `run_truth_ks2d.py`, `exp_g1_2dks_groundtruth_transfer.py` (or freeze),
  `exp_e1_groundtruth_lyapunov_ltransfer.py` (1D); `eval_1d_transfer.py` + `summarize_1d_transfer.py`;
  `eval_spectra_recon.py`; `exp_rollout_stats_{kolmogorov,ks2d}.py`; `exp_reach_velocity.py`; `exp_box_check.py`;
  `exp_ic_normalization_check.py`; `select_alpha01_model.py`. Paper II: `exp_h9_tiled_da.py` (+ da helpers),
  `exp_i11_pde_letkf.py` (+ `run_r7.py`, `run_r7b.py`, M2/M5 drivers), `exp_j16_estimate_pde_model_error.py`,
  `exp_j17h_fit.py`, `exp_j18b_roundtrip.py`, `exp_i9_seams.py` + `exp_d10_seams.py`,
  `exp_f2_jacobian_sparsity.py`, `exp_f4_jacobian_timing.py`, `exp_k_tiled_da.py`, 1D DA via
  `run_blockD_1d.sh` → `exp_e2_ensemble_sweep_ltransfer.py`, `run_da_pff_ltransfer.py` (+ `run_da_pff.py`,
  `run_ltransfer_test.py`), `exp_e3_sec_vs_gc_at_transfer.py`, `run_da_ensemble_sweep.py`.
- **Stage 5 figures/tables.** Keep `_style.py`, `_truths.py`, `shared_facts.py`, `check_numbers.py` and the
  Makefile lists for paperI/paperII (tables before figures). Drop paper1-only scripts (`fig05_box_vs_window`,
  `fig09_2d_da`, `fig11_rollout_filmstrip`, `tab07_2d_da`, `tabE_da`).

## 4. Tests to keep

Solvers (`test_solver*`, `test_ks2d_memory`, `test_ks2d_j17s_terms`, `test_dataset` minus dimension,
`test_training_data`); architecture/Props 1–2 (`test_autoencoder_local_field{,_2d}`, `test_propagator_local_mlp_2d`,
`test_ltransfer_resized_loading{,_2d}`, `test_memory_prop`); window training (`test_crop_training` core);
Lyapunov (`test_lyapunov`, `_checkpoint`, `_latent`); DA (`test_letkf_local`, `test_jacobian_coloring`,
`test_da_geometry2d`, `test_tilings`, `test_letkf_phys`, `test_param_est`, `test_letkf`, `test_pff`, `test_sec`,
`test_sec_pff_integration`, `test_serial_enkf`, `test_localization`, `test_da_cycling`); 1D loops (rewrite against
the local-field AE); infrastructure. New: checkpoint parity, manifest commands parse, results-bundle completeness,
no-paper1-default. Drop ~60 others.

## 5. Layout, environment, entry points, compute

```
<new repo>/  ks_latent/ (pruned)  scripts/ (flat) + scripts/paper_figures/
  manifests/{datasets,models,paperI_runs,paperII_runs}.yaml, results_bundle.txt
  pipelines/run_manifest.py (--stage --paper --profile --only -j), pipelines/smoke/
  tools/{check_closure,fetch_zenodo,verify_numbers,convert_checkpoints}.py
  tests/{unit,golden}  paper/{paperI,paperII}  Makefile environment.yml environment.lock.yml pyproject.toml
  README.md REPRODUCE.md CITATION.cff LICENSE
```

README: what it reproduces; three levels (assets from Zenodo results in minutes; verify from released
checkpoints in hours; full rebuild in days); install; data/checkpoints; artifact → script → manifest map; compute
notes; known limits; tests/licence/citation.

Environment (clean spec; old lock mixes conda/pip numpy and sklearn): python 3.11.15, scipy 1.17.1, h5py 3.16.0,
matplotlib 3.11.1, mpmath, pytest, pyyaml; pip numpy 1.26.4, torch 2.13.0. Test macOS-arm64 (MPS) and linux-64.

Make targets: `test`, `smoke`, `fetch-{results,checkpoints,cache}`, `paper{I,II}-assets`, `paper{I,II}-verify`,
`paper{I,II}-full`, `paper{I,II}-{data,train,eval}`.

Compute (upper bounds, shared machine): full Paper I ≈ 40 GPU-h + ≈ 200 CPU-h (critical path: 88² truth,
~1–1.5 days); Paper II ≈ 30 GPU-h + ≈ 170 CPU-h (physics tuning grids dominate). Assets from Zenodo < 5 min each;
verify tier ≈ 4–8 CPU-h (I), 3–6 (II).

## 6. Hazards

| # | Hazard | Mitigation |
|---|---|---|
| E1 | `/opt/homebrew/...python` in ~80 drivers; `/Users/daltonjones` in config/regularizer/docstrings/several scripts; relative `artifacts/` paths incl. caches | `ks_latent.paths` (env `KSL_ARTIFACTS`), manifests replace drivers, CI grep gate |
| E2 | `_style.py` defaults output to `docs/paper1`; `check_numbers.py` defaults to paper1 | require `PAPER_OUT`/`PAPER_TABLES`; drop paper1 branches |
| E3 | `_style._main_a01()` falls back to `k_xs01_v5` (selected is `k_xs01_v2`) | hardcode + assert against decision.json |
| E4 | glob-dependent tables/figures (`tabD_evaluations`, `figC1/C2`, `tabC`, `_truths.REGISTRY`, `check_numbers`) | pin explicit file lists; fail on missing |
| E5 | DA truths start from `artifacts/cache/*`; twin/memory solvers raise if missing; order-dependent, races | ship cache; `make_spinup_cache.py` run serially first |
| E6 | PDE trajectories depend on FFT thread count; not recorded | record threads in provenance; set per manifest entry; tolerance-based verification |
| E7 | `p7c12d10_ks2d` data noise only in training log; `tabA` parses logs | full argv in manifest; model cards per checkpoint |
| E8 | α 0.05/0.025 wide16 datasets overwritten 2026-10-06 (seed-0 models trained on 4+1 `_orig5`) | name datasets explicitly; assemble both from the same parts; check norm std per checkpoint |
| E9 | `train_kolmogorov_crop.py` silently reuses an existing stage-A file; R16/A2 copy stage A | explicit `--init-ae`; refuse reuse without `--resume`; encode in manifest |
| E10 | data-dependent selection inside drivers (J17 `best()`, α 0.1 selection, C2 vs C3) | reproduce grids, freeze chosen configs in manifest, keep selection code |
| E11 | orphaned producers: `true_dky_kolmogorov_alpha0.1_*_transfer.json`, `pattern_correlation_summary.json`, `da_sweep_phaseC2_global.csv`, ad-hoc `h9_K2b_ctrl`, `h9_J18c`, `i11_L44_st8` | freeze in bundle as `producer: frozen`; optionally re-derive |
| E12 | non-regenerable inputs: `replication_results.jsonl`, `a10_selection/*`, `appendix_D_evaluations.csv` | ship frozen |
| E13 | code not at a commit; checkpoint git_sha values vary | commit + tag `paper-freeze-2026-10` before migrating; cite tags in App. E |
| E14 | checkpoints pickle `ks_latent.config.*` classes; `weights_only=False` | keep names; `convert_checkpoints.py` → state_dict + config.json, load with `weights_only=True`; originals on Zenodo |
| E15 | evaluation opens the training h5 for metadata; figures read dataset snapshots | embed metadata in converted checkpoints; extract figure snapshots to a small npz |
| E16 | seeds/rng offsets; MPS not bitwise | document; don't change offsets; statistical retrain verification; optional `DEVICE=cpu` |
| E17 | no LICENSE; code "ported from a reference implementation" (`~/Documents/experiments/ks_latent`); AGU/REVTeX class files | confirm authorship; MIT/BSD + CITATION.cff; keep class files out of code licence; scrub paths |
| E18 | retraining ≈ 65 GPU-h, not bitwise | Zenodo: checkpoints; results bundle; cache + snapshots; never datasets |
| E19 | sibling-script imports (`exp_i9_seams`→`exp_d10_seams`, j16/j17h→i11, `run_r7b`→`run_r7`, pff_ltransfer→run_da_pff/run_ltransfer_test) | keep `scripts/` flat with original names |
| E20 | paper App. E cites removed modules/queues | update both App. E to the new layout and DOIs |

## 7. Migration steps and verification

- **M0 freeze (0.5 d + compute):** wait for 88² truth, R15, window seeds; commit; tag; rebuild both papers;
  `check_numbers` 0 unmatched; snapshot tables/figures as reference.
- **M1 manifests (1–1.5 d, old repo):** `tools/build_manifest.py` (training argv from checkpoint configs, DA
  commands from JSON args, eval commands from crop JSONs, `results_bundle.txt` = meta inputs ∪ text-number
  sources ∪ globs ∪ frozen files).
- **M2 bundles (0.5 d):** convert checkpoints; results/cache/snapshot bundles; sha256; Zenodo drafts.
- **M3 skeleton (0.5 d):** copy closure + scripts + figures + tests; pytest green before pruning.
- **M4 prune (3–4 d; main risk):** barrel imports → delete modules → propagator.py → config.py → loops/losses →
  training-script argparse → eval/DA script branches; rerun closure, tests, parity after each step.
- **M5 paths/outputs (1 d):** E1–E4, E7, E9, E15; provenance (threads, git_sha, argv) in every JSON.
- **M6 pipelines/Makefile (1 d)** with smoke profile.
- **M7 verification (1 d + compute).** M8 docs/release (0.5 d).
- **Effort:** ≈ 9–11 person-days + ≈ 1 day T1 (CPU) + 1–2 days T2 (overlapping).

Verification tiers:
- **Smoke** (`make smoke`, < 15 min CPU): micro-datasets, 20-step trainings, tiny evals and DA, all paper scripts
  on a fixture bundle.
- **T0** (minutes, blocks release): rebuild all assets from the frozen results; tables byte-identical; figure
  inputs identical and raster diff small; `check_numbers` 0/0.
- **T1** (≈ 8–14 CPU-h): recompute key numbers from released checkpoints with matching threads and shipped cache,
  e.g. C2 66² IC 0 185.1 (±0.5%), k_xs01_v2 2× (±0.5%), KC2 α 0.05 2× 28.3 (±1%), 1D L = 100 D_KY (±0.5%), box
  check R_new 2.66 (±2%), reach F(3) (±5%), Paper II 44² filter 0.324 (±0.003), J17G δ 0.1 matched budget
  0.332 vs 0.328 (±0.005), KDA 0.115 (±0.003), 1D PFF+GC N 64 L 200 0.427 (±0.01), R7 ∇² 1.25 ≈ 0.166 (±0.01).
- **T2** (optional, statistical): retrain r13_C2 s0, k_xs01_v2, c19_e0d0_c6, r14_KC2_a0.05 — config diff OK and
  within stored seed spread (KC2 must still fail); optionally p7c12d10 + 3-seed filter within ±0.01.
- **T3:** full rebuild, statistical match within reported spreads; mismatches reported in REPRODUCE.md.

Critical files: `scripts/train_kolmogorov_crop.py` (stage-A reuse), `ks_latent/models/propagator.py`,
`ks_latent/config.py`, `scripts/exp_h9_tiled_da.py`, `scripts/exp_i11_pde_letkf.py`,
`scripts/paper_figures/{_style,check_numbers}.py`, `Makefile`.
