"""Model-free locality test on CM1 (analysis 4; the CM1 analogue of D6).

Question: at the 5-min output interval, how wide a window of columns at time t
is needed to predict one column at t+1? If skill saturates at a modest
half-width w, a weight-shared local model with that stencil is viable.

Setup
  state per column: 7 variables x 62 levels (cm1_io), one scalar std per var
  compress each column to its top-K principal components (fit on train members)
  sample  : member m, time t, site i (interior, stride SITE_STRIDE)
  features: PCs of columns i-w .. i+w at time t
  target  : PC increment of column i from t to t+1
  skill   : 1 - MSE(pred) / MSE(persistence), on held-out members,
            reported for active columns (cref > 10 dBZ at t or t+1) and quiet ones
  models  : ridge (linear) and a small MLP (nonlinear)
Split by whole member: train 1-7, val 8, test 9-10.
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
rng = np.random.default_rng(0)
K = 32
SITE_STRIDE = 8
WS = [0, 1, 2, 4, 8, 16, 32]
WMAX = max(WS)
TRAIN, VAL, TEST = c.MEMBERS[:7], c.MEMBERS[7:8], c.MEMBERS[8:]
smoke = "--smoke" in sys.argv
# --pooled: columns within +-INNER stay exact; farther ones are averaged in blocks
# of BLOCK (a lower bound on wide-window skill, cheap enough to reach 32 km)
pooled = "--pooled" in sys.argv
INNER, BLOCK = 8, 8
if pooled:
    WS = [8, 16, 32, 48, 64, 96, 128]
    WMAX = max(WS)
if smoke:
    WS, TRAIN, VAL, TEST, SITE_STRIDE = [0, 4], c.MEMBERS[:2], c.MEMBERS[2:3], c.MEMBERS[3:4], 32

t0 = time.time()
runs = {r: c.load_run(r) for r in TRAIN + VAL + TEST}
# per-variable scalar std over train members
std = np.array([np.concatenate([runs[r]["X"][:, v].ravel() for r in TRAIN]).std() for v in range(len(c.VARS))])
print("per-variable std:", dict(zip(c.VARS, std.round(3))))

def cols(r):  # (T, NX, V*Z) normalized columns
    X = runs[r]["X"] / std[None, :, None, None]
    T, V, Z, NX = X.shape
    return X.transpose(0, 3, 1, 2).reshape(T, NX, V * Z)

# PCA of columns on a train subsample
sub = np.concatenate([cols(r)[:, ::4].reshape(-1, len(c.VARS) * 62) for r in TRAIN])
mu = sub.mean(0)
U, S, Vt = np.linalg.svd(sub - mu, full_matrices=False)
ev = S**2 / (S**2).sum()
cum = np.cumsum(ev)
pca_info = {f"k_for_{int(p*100)}pct": int(np.searchsorted(cum, p) + 1) for p in (0.9, 0.95, 0.99, 0.999)}
pca_info["var_frac_top_K"] = float(cum[K - 1])
print("column PCA:", pca_info)
P = Vt[:K].T  # (V*Z, K)

def pcs(r):
    return ((cols(r) - mu) @ P).astype(np.float32)  # (T, NX, K)

PC = {r: pcs(r) for r in runs}
# how much of the one-step increment variance the top-K subspace holds (test members)
inc_full = np.concatenate([np.diff(cols(r), axis=0).reshape(-1, P.shape[0]) for r in TEST])
pca_info["increment_var_frac_in_top_K"] = float(((inc_full @ P) ** 2).sum() / (inc_full**2).sum())
del inc_full
print("increment variance in top-K:", pca_info["increment_var_frac_in_top_K"])
NX = PC[TRAIN[0]].shape[1]
sites = np.arange(WMAX, NX - WMAX, SITE_STRIDE)

def build(members, w):
    Xs, Ys, A = [], [], []
    for r in members:
        p, cref = PC[r], runs[r]["cref"]
        T = p.shape[0]
        if pooled and w > INNER:
            inner = sites[:, None] + np.arange(-INNER, INNER + 1)[None, :]
            nb = (w - INNER) // BLOCK
            offs = [np.arange(INNER + 1 + b * BLOCK, INNER + 1 + (b + 1) * BLOCK) for b in range(nb)]
        idx = sites[:, None] + np.arange(-w, w + 1)[None, :]       # (S, 2w+1)
        for t in range(T - 1):
            if pooled and w > INNER:
                f = [p[t][inner].reshape(len(sites), -1)]
                for o in offs:
                    f.append(p[t][sites[:, None] + o[None, :]].mean(1))
                    f.append(p[t][sites[:, None] - o[None, :]].mean(1))
                Xs.append(np.concatenate(f, 1))
            else:
                Xs.append(p[t][idx].reshape(len(sites), -1))
            Ys.append(p[t + 1, sites] - p[t, sites])
            A.append((cref[t, sites] > 10) | (cref[t + 1, sites] > 10))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(A)

def skill(pred, Y, A):
    e, e0 = ((pred - Y) ** 2).sum(1), (Y**2).sum(1)
    out = {}
    for name, m in (("all", np.ones_like(A)), ("active", A), ("quiet", ~A)):
        out[name] = float(1 - e[m].sum() / e0[m].sum())
    return out

def ridge(Xtr, Ytr, Xva, Yva, Ava):
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = (Xtr - xm).astype(np.float64), (Ytr - ym).astype(np.float64)
    G, B = Xc.T @ Xc, Xc.T @ Yc
    best = None
    for lam in [1e-2, 1e0, 1e2, 1e3, 1e4]:
        Wt = np.linalg.solve(G + lam * len(Xc) * 1e-4 * np.eye(len(G)), B)
        s = skill((Xva - xm) @ Wt + ym, Yva, Ava)["all"]
        if best is None or s > best[0]:
            best = (s, lam, Wt)
    return lambda X: (X - xm) @ best[2] + ym, best[1]

def mlp(Xtr, Ytr, Xva, Yva, Ava, epochs=30):
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-6
    ys = Ytr.std(0) + 1e-6
    f = lambda X: torch.tensor((X - xm) / xs, dtype=torch.float32)
    Xt, Yt, Xv = f(Xtr), torch.tensor(Ytr / ys, dtype=torch.float32), f(Xva)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 256), torch.nn.GELU(),
                              torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, Yt.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = -np.inf, None
    for ep in range(epochs if not smoke else 2):
        perm = torch.randperm(len(Xt))
        net.train()
        for b in range(0, len(Xt), 512):
            i = perm[b:b + 512]
            loss = ((net(Xt[i]) - Yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            s = skill(net(Xv).numpy() * ys, Yva, Ava)["all"]
        if s > best:
            best, state = s, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    def pred(X):
        with torch.no_grad():
            return net(f(X)).numpy() * ys
    return pred

res = {"pca": pca_info, "K": K, "site_stride": SITE_STRIDE, "dx_km": 0.25, "dt_s": 300,
       "split": {"train": TRAIN, "val": VAL, "test": TEST}, "by_w": {}}
for w in WS:
    Xtr, Ytr, _ = build(TRAIN, w)
    Xva, Yva, Ava = build(VAL, w)
    Xte, Yte, Ate = build(TEST, w)
    rp, lam = ridge(Xtr, Ytr, Xva, Yva, Ava)
    mp = mlp(Xtr, Ytr, Xva, Yva, Ava)
    r = {"half_width_km": w * 0.25, "n_train": int(len(Xtr)), "active_frac_test": float(Ate.mean()),
         "ridge": skill(rp(Xte), Yte, Ate), "ridge_lambda": lam, "mlp": skill(mp(Xte), Yte, Ate)}
    res["by_w"][w] = r
    print(f"w={w:3d} ({w*0.25:5.2f} km)  ridge {r['ridge']}  mlp {r['mlp']}  [{time.time()-t0:.0f}s]", flush=True)
    del Xtr, Ytr

out = c.CACHE / ("locality_test_smoke.json" if smoke else
                 "locality_test_pooled.json" if pooled else "locality_test.json")
res["pooled"] = {"inner": INNER, "block": BLOCK} if pooled else None
out.write_text(json.dumps(res, indent=1))
print("wrote", out)
