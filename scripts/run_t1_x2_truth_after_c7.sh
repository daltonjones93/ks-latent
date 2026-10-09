#!/bin/bash
# Queued 2026-10-03 (user: "queue it after C7"): T1's 2x true-PDE D_KY seeds 0-2,
# which decide whether the 2x dip in D_KY/Area (and P1's 2x miss) is real.
# Waits for the C7 pipeline (and anything it waits on) to finish; preemptable via
# T1's own Benettin checkpoints and done-file skipping.
while pgrep -f "exp_c7_pipeline.py|exp_crop_pipeline.py|run_icnorm_ks2d_seeds.sh|run_ks2d_seeds_after" > /dev/null; do sleep 60; done
cd "$(dirname "$0")/.."
python - <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("t1", "scripts/exp_t1_error_bars.py")
t1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t1)
for s in t1.SEEDS:
    if t1.truth_result_path(2, s).exists():
        t1.log(f"skip truth x2 seed{s} (done)"); continue
    t1.run_truth(2, s)
t1.log("T1 x2 truth seeds done")
print(t1.summarize()["truth"]["x2"])
PY
