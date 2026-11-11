"""Pre-registration 5r: prefix truncation of the nested-dropout latent at inference. Only the first m channels evolve; channels >= m are
(a) MEAN: set to their dataset mean in the initial state, the halo and after every step, (b) FROZEN at the t0 value, (c) FORCED to the encoded truth (oracle).
Errors are against the true field at lags 5/15/30 per variable, with the truncation floor decode(truth with channels >= m at the mean). m = 1..32.
  python scripts/cm1/prop_prefix.py --run ~/cm1_runs/fpc_nested_stagec --source cache:~/cm1_cache,~/cm1_cache_B --out ~/cm1_da/prefix_nested
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import _with_true_halo  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from prop_clamp import roll, var_err, LAGS  # noqa: E402

MS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


@torch.no_grad()
def roll_mean(prop, zt, t0, n, halo, tail, mean):
    s = zt[t0]; out = [s]
    for j in range(n):
        seen = _with_true_halo(s, zt[t0 + j], halo)
        s = _with_true_halo(prop(seen[None])[0], zt[t0 + j + 1], halo).clone()
        s[:, tail] = mean[tail]
        out.append(s)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--names-prefix", default=""); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads); t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo, p, m_sites, c = cfg["stage_b"]["halo_sites"], ae.cfg.patch, ae.cfg.reach_sites, ae.cfg.latent_channels
    files = member_files(a.source)
    names = sorted(n for n in files if n.startswith(a.names_prefix))[: a.n_test] if a.names_prefix else split["test"][: a.n_test]
    zs, ds = {}, {}
    for nm in names:
        ds[nm] = ColumnData([load_member(files[nm])], norm, dev)
        with torch.no_grad():
            zs[nm] = encode_members(ae, ds[nm])[0]
    mean = torch.cat([zs[nm][60:].reshape(-1, c) for nm in names]).mean(0)
    res = {"args": vars(a), "names": names, "vars": list(VARS), "lags": list(LAGS), "ms": MS, "cases": []}
    n = max(LAGS)
    for nm in names:
        d, z = ds[nm], zs[nm]
        for t0 in [int(x) for x in a.t0.split(",")]:
            if t0 + n + 1 > z.shape[0]:
                continue
            u_true = torch.stack([d.full(0, t0 + k)[0] for k in LAGS])
            ev = lambda st: var_err(ae, torch.stack([st[k] for k in LAGS]), u_true, d.Z, halo, m_sites, p).tolist()
            case = {"member": nm, "t0": t0, "persistence": var_err(ae, z[t0][None].expand(len(LAGS), -1, -1), u_true, d.Z, halo, m_sites, p).tolist(),
                    "free": ev(roll(prop, z, t0, n, halo, [], "free")), "m": {}}
            for m in MS:
                tail = list(range(m, c))
                zt = z.clone(); zt[:, :, tail] = mean[tail]
                case["m"][str(m)] = {"floor": ev(zt[t0:t0 + n + 1] if False else torch.stack([zt[t0 + k] for k in range(n + 1)])) if tail else ev(z[t0:t0 + n + 1]),
                                     "mean": ev(roll_mean(prop, zt, t0, n, halo, tail, mean)) if tail else case["free"],
                                     "frozen": ev(roll(prop, z, t0, n, halo, tail, "frozen")) if tail else case["free"],
                                     "forced": ev(roll(prop, z, t0, n, halo, tail, "forced")) if tail else case["free"]}
            res["cases"].append(case)
            print(f"[prefix] {nm} t0={t0} ({time.time() - t_start:.0f}s)", flush=True)
    json.dump(res, open(out / "prefix.json", "w")); print("[prefix] done")


if __name__ == "__main__":
    main()
