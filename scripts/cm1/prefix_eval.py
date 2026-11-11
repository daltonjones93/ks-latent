"""5s: evaluate pruned (prefix_m) runs on the 5r cases: free rollout error and truncation floor at lags 5/15/30 per variable.
Compares with the unpruned model's numbers in prefix_nested.json (same members and start times).
  python scripts/cm1/prefix_eval.py --runs ~/cm1_runs/fpc_nested_p8_prop ... --source cache:... --out ~/cm1_da/prefix_ft
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from prop_clamp import roll, var_err, LAGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--n-test", type=int, default=10); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads); t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cpu")
    files = member_files(a.source); res = {"vars": list(VARS), "lags": list(LAGS), "runs": {}}
    for run in a.runs:
        distilled = run.endswith(":distilled"); run = Path(run.replace(":distilled", "")).expanduser()
        ae, prop, norm, cfg, split = load_run(run, dev)
        if distilled:                                       # the checkpoint after the distillation phase, before the CM1 fine-tune
            ae.load_state_dict(torch.load(run / "ae_distilled.pt", map_location=dev)["state_dict"])
            prop.load_state_dict(torch.load(run / "prop_distilled.pt", map_location=dev)["state_dict"])
        halo, p, m_sites = cfg["stage_b"]["halo_sites"], ae.cfg.patch, ae.cfg.reach_sites
        names = split["test"][: a.n_test]; n = max(LAGS); cases = []
        for nm in names:
            d = ColumnData([load_member(files[nm])], norm, dev)
            with torch.no_grad():
                z = encode_members(ae, d)[0]
            for t0 in [int(x) for x in a.t0.split(",")]:
                if t0 + n + 1 > z.shape[0]:
                    continue
                u_true = torch.stack([d.full(0, t0 + k)[0] for k in LAGS])
                ev = lambda st: var_err(ae, torch.stack([st[k] for k in LAGS]), u_true, d.Z, halo, m_sites, p).tolist()
                cases.append({"member": nm, "t0": t0, "floor": ev(z[t0:t0 + n + 1]), "free": ev(roll(prop, z, t0, n, halo, [], "free")),
                              "persistence": var_err(ae, z[t0][None].expand(len(LAGS), -1, -1), u_true, d.Z, halo, m_sites, p).tolist()})
        res["runs"][run.name + (":distilled" if distilled else "")] = {"prefix_m": ae.cfg.prefix_m, "cases": cases}
        print(f"[prefix_eval] {run.name} m={ae.cfg.prefix_m} {len(cases)} cases ({time.time() - t_start:.0f}s)", flush=True)
    json.dump(res, open(out / "prefix_ft.json", "w")); print("[prefix_eval] done")


if __name__ == "__main__":
    main()
