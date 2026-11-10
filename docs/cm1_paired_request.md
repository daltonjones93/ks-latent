# Request: more paired CM1 runs for Paper III (growth matching)

**Ask.** About 24 additional restart-and-perturb storm sets on the 2D squall line (same configuration as `cm1data/`, 1600 x 62 cells, dx 250 m), plus roughly 40 core-hours and 40 GB of scratch. Nothing new is needed from CM1 itself; the work is re-running existing members from t = 0 with restart files and then branching.

## Why
Paper III's surrogate propagator is much less unstable than CM1. Its leading Lyapunov exponent over 15-30 min is 0.006-0.015 per min against CM1's 0.12-0.25 (the 4 paired storms we have). A fine-tune that matches the growth of CM1's own perturbation differences (note 5n, `docs/cm1_notes.md`) raised the exponent on 3 of 4 held-out storms to 0.06-0.135 per min, but it overfit: held-out growth loss fell only 45% while the training-storm loss fell to ~0.01, and the held-out control forecast got 23-30% worse. Four storms is too few. This request is the data to find out whether more storms fix that.

## What exists and what is missing
- The 61 base members (`ens_A` 31, `ens_A2` 30) are on Cashew in `~/cm1_bundle`. They were run without restart files, so a paired set needs a re-run to 7260 s that writes a restart at 7200 s (bitwise reproducible; verified, `cluster/cm1/bump_smoke.sh`).
- Four storms are already done (`memA029`, `memA030`, `memB001`, `memB002`; `ens_P` = base to 7260 s, `ens_Q` = branches to 10800 s).
- The patched build (`bump_restart_var5`) with a restart-time bump and a tiny hashed field is built (`build_l3`).

## Proposed set
Per storm, one re-run to 7260 s with a restart, then branches that run 1 h each (1-min output):

| branch | purpose |
|---|---|
| control | reference |
| tiny field, 1e-3 K, 2 hash seeds | growth target |
| tiny field, 1e-4 K, 2 hash seeds | growth target at a second amplitude |

That is 5 branches per storm, no bumps (the bump branches test the response, which this fine-tune does not need). Two seeds per amplitude give a measure of how much of the growth is the particular perturbation. 24 storms from the remaining `ens_A`/`ens_A2` members, with 20 for training and 4 held out. The 4 current storms stay as an extra held-out set.

**Cost (scaled from measurements, not re-measured for the branches).** Stage 1 for the 4 existing storms took 23 min on 4 CPUs (~0.4 core-hours per storm). A 1 h branch at the measured ~14.5 core-min per member-hour is ~0.25 core-hours, so ~1.25 per storm, ~1.7 per storm in total, ~40 core-hours for 24 storms. Output for the existing 4 storms with 9 branches was 15 GB; 5 branches with fewer variables kept should be ~1.5 GB per storm, ~36 GB. Wall time: with 20 CPUs about 2-3 h, plus the cache build (~3 min per storm).

## Optional extensions (each adds cost, not required)
1. **A second restart time**, e.g. 5400 s as well as 7200 s (set `rstfrq = 1800`, which writes restarts at every 1800 s; `make_paired.py` takes a fixed t0 and needs a small change). CM1's growth is concentrated 15-30 min after the perturbation, so the storm's stage at restart matters. Doubles the branch cost.
2. **New regimes (needs the supercomputer, or someone running new base members):** soundings or shear outside the current ensemble. The one storm where the fine-tune failed to transfer (`memB001`) differs from the others; I have not established why. If the 24 new storms come back with the same failure, that would show a regime effect and not a data-quantity one, and new regimes would be the next request.

## What I would do with it, and what would count as success
Train the growth-matching fine-tune on 20 storms and test on the other 4 and the existing 4. Same pre-registered rule as note 5n: held-out window growth loss down by at least 50%, the held-out leading Lyapunov at 15-30 min at or above 0.06 per min on both folds, and the held-out control anchor MSE up by less than 20%. If that is not met with 20 storms, I would stop pursuing growth matching as a training term and report the growth gap as a property of the surrogate class.

## What I need
1. Agreement to use `coe_cpu` on Cashew for ~40 core-hours (or the same on another cluster).
2. ~40 GB of scratch on Cashew, or another place to put the branch output until the cache is built (the raw output can be deleted afterwards; the cache is a few GB).
3. If a postdoc runs it: the kit is `cluster/cm1/` (`paired_stage1.sh`, `paired_make.sh`, `paired_cache.sh`); I will adapt the member list and branch set and test it on 2 storms first.
