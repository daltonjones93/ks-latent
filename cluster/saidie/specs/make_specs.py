"""Write the JSON specs for the Fable-review section-6 DA sweeps (docs/cm1_fable_review.md; G0 addendum 4).
Run from the repo root: python cluster/saidie/specs/make_specs.py"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SET = "--n-members 32 --random-members --seeds 0 1 2 --obs-set th+refl+vr --boundary persistence --forecast-from"
STOCH_FROZEN = "/home/c838596879/cm1_runs/m3_stoch_frozen_stagec"


def setn(n):
    return SET.replace("--n-members 32", f"--n-members {n}")


specs = {
    # N5: localization radius against the reach rule (curve: test at every radius)
    "n5_loc": [{"name": "n5_loc", "run": "full_v1_stagec", "args": f"--modes latent {SET} --model-noise 0.1",
                "grid": {"--loc-km": [2, 5, 10, 20, 40]}, "test": "all"}],
    # N2: innovation-driven adaptive additive noise (base level tuned on validation)
    "n2_adapt": [{"name": f"n2_adapt_{k}", "run": "full_v1_stagec",
                  "args": f"--modes latent {SET} --adaptive-noise {k}",
                  "grid": {"--model-noise": [0.05, 0.1, 0.2]}, "test": "pick"} for k in ("global", "site")],
    # N4: ensemble size for the 32- and 64-channel autoencoders (noise re-tuned per N)
    "n4_ens": [{"name": f"n4_{lab}_N{n}", "run": run, "args": f"--modes latent {setn(n)}",
                "grid": {"--model-noise": [0.05, 0.1, 0.15]}, "test": "pick"}
               for lab, run in (("c32", "full_v1_stagec"), ("c64", "sm1_c64_stagec")) for n in (8, 16, 32, 64, 93)],
    # M1: observation spacing x cycle length; deterministic + tuned noise, stochastic + tuned RTPS
    "m1_sparse": [{"name": f"m1_{lab}_obs{km}_cyc{cy}", "run": run,
                   "args": f"--modes latent {SET} --obs-km {km} --refl-km {km} --vr-km {km} --cycle-min {cy}" + extra,
                   "grid": grid, "test": "pick"}
                  for km in (2, 4, 8, 16) for cy in (5, 10, 15, 30)
                  for lab, run, extra, grid in (
                      ("det", "full_v1_stagec", "", {"--model-noise": [0.05, 0.1, 0.2]}),
                      ("stoch", "full_v1_stagec_crps", " --model-noise 0", {"--rtps": [0.5, 0.9]}))],
    # M3: mean-anchored ensemble (deterministic mean, stochastic perturbations on the same frozen latent)
    "m3_anchored": [{"name": "m3_anchored", "run": "full_v1_stagec",
                     "args": f"--modes anchored {SET} --model-noise 0 --stoch-run {STOCH_FROZEN}",
                     "grid": {"--rtps": [0.0, 0.5, 0.9]}, "test": "pick"},
                    {"name": "m3_stoch_frozen_alone", "run": "m3_stoch_frozen_stagec",
                     "args": f"--modes latent {SET} --model-noise 0", "grid": {"--rtps": [0.0, 0.5, 0.9]}, "test": "pick"}],
}
for k, v in specs.items():
    (HERE / f"{k}.json").write_text(json.dumps(v, indent=1))
    print(k, len(v), "studies")
