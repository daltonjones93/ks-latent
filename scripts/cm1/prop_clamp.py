"""Pre-registration 5o (item 5): channel clamping in propagator rollouts.

From a real state at t0, roll the propagator n steps (halo = the true latents) and decode at lags 5, 15, 30, with the latent channels in K
(a) FREE (none held), (b) FROZEN at their t0 value at every site, (c) FORCED to the encoded true trajectory (teacher forcing of K only).
Per physical variable v (u, w, th, qv, prs, qliq, qice) the error is the mean squared error of the decoded field against the true field in
normalised units over all cells and levels; the floor is decode(encode(truth)). Frozen damage of K = err_frozen / err_free (> 1: the dynamics
of the other channels depend on K's evolution); forced gain = err_forced / err_free (< 1: K's own evolution is information the other channels
lack). Channel sets: single channels 0,1,2,3,4,6,8,12,16,24,31 and the groups 0-3, 8-11, 28-31.
  python scripts/cm1/prop_clamp.py --run ~/cm1_runs/fpc_nested_stagec --source cache:~/cm1_cache,~/cm1_cache_B --out ~/cm1_da/clamp_nested
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import _with_true_halo  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

SINGLES = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 31]
GROUPS = {"0-3": [0, 1, 2, 3], "8-11": [8, 9, 10, 11], "28-31": [28, 29, 30, 31]}
LAGS = (5, 15, 30)


@torch.no_grad()
def roll(prop, z, t0, n, halo, K, mode):
    """z (T,P,c) true latents. Returns states (n+1, P, c) from t0; channels K frozen / forced / free."""
    if prop.cfg.n_hist != 1:
        raise ValueError("one-frame history only")
    s = z[t0]; out = [s]
    for j in range(n):
        seen = _with_true_halo(s, z[t0 + j], halo)
        s = _with_true_halo(prop(seen[None])[0], z[t0 + j + 1], halo)
        if K and mode == "frozen":
            s = s.clone(); s[:, K] = z[t0][:, K]
        elif K and mode == "forced":
            s = s.clone(); s[:, K] = z[t0 + j + 1][:, K]
        out.append(s)
    return torch.stack(out)


@torch.no_grad()
def var_err(ae, zs, u_true, Z, halo, m_sites, p):
    """zs (len(LAGS), P, c), u_true (len(LAGS), C, NX) -> (len(LAGS), V) mean squared error per variable (interior sites)."""
    dec = ae.decode(zs)
    sl = slice((halo + m_sites) * p, dec.shape[-1] - (halo + m_sites) * p)
    e = ((dec - u_true) ** 2)[..., sl]
    return e.reshape(e.shape[0], len(VARS), Z, -1).mean((2, 3)).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--names-prefix", default="", help="use cached members whose name starts with this instead of the run's test split")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t_start = time.time(); torch.set_num_threads(a.threads)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo, p = cfg["stage_b"]["halo_sites"], ae.cfg.patch
    files = member_files(a.source)
    names = sorted(n for n in files if n.startswith(a.names_prefix))[: a.n_test] if a.names_prefix else split["test"][: a.n_test]
    sets = {f"c{k}": [k] for k in SINGLES if k < ae.cfg.latent_channels}
    sets.update({f"g{k}": v for k, v in GROUPS.items()})
    res = {"args": vars(a), "names": names, "sets": sets, "vars": list(VARS), "lags": list(LAGS), "cases": []}
    for name in names:
        d = ColumnData([load_member(files[name])], norm, dev)
        with torch.no_grad():
            z = encode_members(ae, d)[0]
        T = z.shape[0]; n = max(LAGS)
        for t0 in [int(x) for x in a.t0.split(",")]:
            if t0 + n + 1 > T:
                continue
            u_true = torch.stack([d.full(0, t0 + k)[0] for k in LAGS])
            m_sites = ae.cfg.reach_sites
            ev = lambda st: var_err(ae, torch.stack([st[k] for k in LAGS]), u_true, d.Z, halo, m_sites, p)
            case = {"member": name, "t0": t0,
                    "floor": ev(torch.stack([z[t0 + k] for k in range(n + 1)])).tolist(),
                    "persistence": var_err(ae, z[t0][None].expand(len(LAGS), -1, -1), u_true, d.Z, halo, m_sites, p).tolist(),
                    "free": ev(roll(prop, z, t0, n, halo, [], "free")).tolist(), "frozen": {}, "forced": {}}
            for nm, K in sets.items():
                case["frozen"][nm] = ev(roll(prop, z, t0, n, halo, K, "frozen")).tolist()
                case["forced"][nm] = ev(roll(prop, z, t0, n, halo, K, "forced")).tolist()
            res["cases"].append(case)
            print(f"[clamp] {name} t0={t0} done ({time.time() - t_start:.0f}s)", flush=True)
    json.dump(res, open(out / "clamp.json", "w"))
    print("[clamp] done")


if __name__ == "__main__":
    main()
