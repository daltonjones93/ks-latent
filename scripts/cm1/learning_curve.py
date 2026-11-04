"""How much CM1 data does a local model need? Learning curve on the locality test.

Fixed window: columns within +-8 exact, farther ones block-averaged to +-48
(12 km half-width, where skill saturated in locality_test_pooled). Train on
k of members 1-7 (several random draws per k), validate on 8, test on 9-10.
Optionally add mirror images of the training members: with icor=0 the
equations are symmetric under x -> -x, u -> -u, so a mirrored member is a
valid storm (in mirrored shear). Skill = 1 - MSE/MSE(persistence) for the
5-min PC increment of the centre column.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import cm1_io as c

torch.set_num_threads(2)
K, STRIDE, INNER, BLOCK, W = 32, 8, 8, 8, 48
POOL, VAL, TEST = c.MEMBERS[:7], c.MEMBERS[7:8], c.MEMBERS[8:]
SIZES, DRAWS = [1, 2, 3, 5, 7], 3
smoke = "--smoke" in sys.argv
if smoke:
    SIZES, DRAWS, STRIDE = [1, 2], 1, 32
U_IDX = c.VARS.index("u")

raw = {r: c.load_run(r) for r in POOL + VAL + TEST}
def mirror(d):
    X = d["X"][..., ::-1].copy(); X[:, U_IDX] *= -1
    return {"X": X, "cref": d["cref"][:, ::-1].copy()}
runs = dict(raw)
runs.update({r + "_m": mirror(raw[r]) for r in POOL})

std = np.array([np.concatenate([raw[r]["X"][:, v].ravel() for r in POOL]).std() for v in range(len(c.VARS))])
def cols(r):
    X = runs[r]["X"] / std[None, :, None, None]
    T, V, Z, NX = X.shape
    return X.transpose(0, 3, 1, 2).reshape(T, NX, V * Z)
# PCA fitted on all 7 pool members (unmirrored) for every k, so k changes only the regression data
sub = np.concatenate([cols(r)[:, ::4].reshape(-1, len(c.VARS) * 62) for r in POOL])
mu = sub.mean(0)
P = np.linalg.svd(sub - mu, full_matrices=False)[2][:K].T
PC = {r: ((cols(r) - mu) @ P).astype(np.float32) for r in runs}
NX = PC[POOL[0]].shape[1]
sites = np.arange(W, NX - W, STRIDE)
inner = sites[:, None] + np.arange(-INNER, INNER + 1)[None, :]
offs = [np.arange(INNER + 1 + b * BLOCK, INNER + 1 + (b + 1) * BLOCK) for b in range((W - INNER) // BLOCK)]

def build(members):
    Xs, Ys, A = [], [], []
    for r in members:
        p, cref = PC[r], runs[r]["cref"]
        for t in range(p.shape[0] - 1):
            f = [p[t][inner].reshape(len(sites), -1)]
            for o in offs:
                f += [p[t][sites[:, None] + o].mean(1), p[t][sites[:, None] - o].mean(1)]
            Xs.append(np.concatenate(f, 1))
            Ys.append(p[t + 1, sites] - p[t, sites])
            A.append((cref[t, sites] > 10) | (cref[t + 1, sites] > 10))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(A)

def skill(pred, Y, A):
    e, e0 = ((pred - Y) ** 2).sum(1), (Y**2).sum(1)
    return {n: float(1 - e[m].sum() / e0[m].sum()) for n, m in (("all", np.ones_like(A)), ("active", A))}

def ridge(Xtr, Ytr, Xva, Yva, Ava):
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc = (Xtr - xm).astype(np.float64)
    G, B = Xc.T @ Xc, Xc.T @ (Ytr - ym)
    best = None
    for lam in [1e-2, 1e0, 1e2, 1e3, 1e4, 1e5]:
        Wt = np.linalg.solve(G + lam * len(Xc) * 1e-4 * np.eye(len(G)), B)
        s = skill((Xva - xm) @ Wt + ym, Yva, Ava)["all"]
        if best is None or s > best[0]:
            best = (s, Wt)
    return lambda X: (X - xm) @ best[1] + ym

def mlp(Xtr, Ytr, Xva, Yva, Ava, epochs=30):
    xm, xs, ys = Xtr.mean(0), Xtr.std(0) + 1e-6, Ytr.std(0) + 1e-6
    f = lambda X: torch.tensor((X - xm) / xs, dtype=torch.float32)
    Xt, Yt, Xv = f(Xtr), torch.tensor(Ytr / ys, dtype=torch.float32), f(Xva)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 256), torch.nn.GELU(),
                              torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, Yt.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = -np.inf, None
    # same number of gradient steps for every k (fair comparison across data sizes)
    steps = 6000 if not smoke else 100
    for s_ in range(steps):
        i = torch.randint(0, len(Xt), (512,))
        loss = ((net(Xt[i]) - Yt[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s_ + 1) % min(300, steps) == 0:
            with torch.no_grad():
                sc = skill(net(Xv).numpy() * ys, Yva, Ava)["all"]
            if sc > best:
                best, state = sc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return lambda X: net(f(X)).detach().numpy() * ys

Xva, Yva, Ava = build(VAL)
Xte, Yte, Ate = build(TEST)
rng = np.random.default_rng(0)
rows, t0 = [], time.time()
for k in SIZES:
    draws = [POOL[:k]] if k == len(POOL) else [list(rng.choice(POOL, k, replace=False)) for _ in range(DRAWS)]
    for d in draws:
        for mir in (False, True):
            tr = d + ([r + "_m" for r in d] if mir else [])
            Xtr, Ytr, _ = build(tr)
            row = {"k": k, "members": d, "mirror": mir, "n_train": int(len(Xtr)),
                   "ridge": skill(ridge(Xtr, Ytr, Xva, Yva, Ava)(Xte), Yte, Ate),
                   "mlp": skill(mlp(Xtr, Ytr, Xva, Yva, Ava)(Xte), Yte, Ate)}
            rows.append(row)
            print(f"k={k} mirror={mir!s:5} ridge act {row['ridge']['active']:.3f}  mlp act {row['mlp']['active']:.3f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
out = c.CACHE / ("learning_curve_smoke.json" if smoke else "learning_curve.json")
out.write_text(json.dumps({"window_half_width_km": W * 0.25, "rows": rows}, indent=1))
print("wrote", out)
