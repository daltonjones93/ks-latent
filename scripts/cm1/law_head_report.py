"""FPg: how closed is the dense law on its n leading channels, and what is the short PDE?

For a run (any law propagator), on the encoded states of the train and test storms (frames f0:f1, interior sites):
  F_i(z)      the dense law's tendency of channel i < n (exact; depends on every channel)
  closure     R2 of an OLS fit of F_i on the library of the first n channels only,
                [1, z_j, d1 z_j, d2 z_j, z_j z_k (j<=k), z_j d1z_k], j,k < n   (the head's library),
              fitted on train storms, scored on test storms. This is the post-hoc version of the head and works on any run;
  head        for a run trained with --law-head-n: R2 of the learned head g_i against F_i, and against the data's centred
              difference; F_i against the data for reference.
Also prints channel 0's equation in the n leading channels (largest terms, as coefficient x channel scale).
  python scripts/cm1/law_head_report.py --run ~/cm1_runs/fpg_n4_w1_stagec --source cache:A+A2+B --n 4 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.leadlaw import site_derivs  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402


def library(zn, d1, d2):
    """(N, n) fields -> (N, n_features), names."""
    n = zn.shape[1]
    cols, names = [np.ones(len(zn))], ["1"]
    for tag, a in (("z", zn), ("d1z", d1), ("d2z", d2)):
        for j in range(n):
            cols.append(a[:, j]); names.append(f"{tag}{j}")
    for j in range(n):
        for k in range(j, n):
            cols.append(zn[:, j] * zn[:, k]); names.append(f"z{j}*z{k}")
    for j in range(n):
        for k in range(n):
            cols.append(zn[:, j] * d1[:, k]); names.append(f"z{j}*d1z{k}")
    return np.stack(cols, 1), names


def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--frames", default="30:330")
    a = ap.parse_args()
    t0 = time.time(); torch.set_num_threads(4)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    prop.eval()
    law, n = prop.law, a.n
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    f0, f1 = map(int, a.frames.split(":"))

    def rows(names):
        R = []
        for nm in names:
            d = ColumnData([load_member(files[nm])], norm, dev)
            with torch.no_grad():
                z = encode_members(ae, d)[0].cpu().numpy().astype(np.float64)
            hi = min(f1, z.shape[0] - 2)
            zt = torch.as_tensor(z[f0:hi], dtype=torch.float32)
            with torch.no_grad():
                F = law.tendency(zt)[..., :n].double().numpy()
                G = law.head.tendency(zt[..., :n]).double().numpy() if law.head is not None else None
            zn = z[f0 - 1: hi + 1, :, :n]
            d1, d2 = (x for x in site_derivs(zn[1:-1], 2))
            data = 0.5 * (zn[2:] - zn[:-2])
            sl = slice(halo, -halo)
            fl = lambda v: v[:, sl].reshape(-1, n)
            R.append({"F": fl(F), "G": None if G is None else fl(G), "data": fl(data), "z": fl(zn[1:-1]),
                      "d1": fl(d1), "d2": fl(d2)})
        return R
    cat = lambda Rs, k: np.concatenate([r[k] for r in Rs])
    tr, te = rows(split["train"][: a.n_train]), rows(split["test"][: a.n_test])
    Xtr, names = library(*(cat(tr, k) for k in ("z", "d1", "d2")))
    Xte, _ = library(*(cat(te, k) for k in ("z", "d1", "d2")))
    sc = Xtr.std(0); sc[sc == 0] = 1
    res = {"run": a.run, "n": n, "channels": {}}
    print(f"[head] run {a.run}, n {n}, library {len(names)} terms per channel", flush=True)
    for i in range(n):
        Ftr, Fte = cat(tr, "F")[:, i], cat(te, "F")[:, i]
        coef, *_ = np.linalg.lstsq(Xtr, Ftr, rcond=None)
        row = {"closure_R2_train": r2(Ftr, Xtr @ coef), "closure_R2_test": r2(Fte, Xte @ coef),
               "F_vs_data_R2_test": r2(cat(te, "data")[:, i], Fte)}
        if law.head is not None:
            Gte = cat(te, "G")[:, i]
            row["head_vs_F_R2_test"] = r2(Fte, Gte)
            row["head_vs_data_R2_test"] = r2(cat(te, "data")[:, i], Gte)
        top = np.argsort(-np.abs(coef * sc))[:6]
        row["top_terms_ols"] = [(names[j], float(coef[j])) for j in top]
        res["channels"][str(i)] = row
        print(f"[head] ch{i}: closure R2 train {row['closure_R2_train']:.3f} test {row['closure_R2_test']:.3f}; "
              f"F vs data {row['F_vs_data_R2_test']:.3f}"
              + (f"; head vs F {row['head_vs_F_R2_test']:.3f}, head vs data {row['head_vs_data_R2_test']:.3f}" if law.head is not None else "")
              + "; " + " ".join(f"{c:+.3g} {nm}" for nm, c in row["top_terms_ols"]), flush=True)
    res["mean_closure_R2_test"] = float(np.mean([v["closure_R2_test"] for v in res["channels"].values()]))
    print(f"[head] mean closure R2 test over the {n} channels: {res['mean_closure_R2_test']:.3f}")
    json.dump(res, open(out / "head.json", "w"), indent=1)
    print(f"[head] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
