"""Plan P: what the CM1 latent field knows. Site-level targets from raw CM1 output, event detection,
neighbourhood features, probes and their scores, k-means regimes. Everything here is plain numpy / torch
so it can be tested on synthetic data with known answers.

Conventions: a "site" is one latent site = `patch` grid columns (8 x 250 m = 2 km); site series are (T, P).
Events are ONSETS: a site is a candidate at time t only if the quantity has stayed below the threshold for
the previous `quiet` frames (inclusive of t), and it is labelled 1 if the threshold is reached within
(t, t + tau]. Already-active sites are never candidates, so persistence of an ongoing event cannot score.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------- targets from raw CM1 output
def site_reduce(a: np.ndarray, patch: int, how: str = "max") -> np.ndarray:
    """(T, NX) -> (T, P): reduce each site's `patch` columns."""
    T, NX = a.shape
    P = NX // patch
    b = a[:, :P * patch].reshape(T, P, patch)
    return {"max": b.max, "min": b.min, "mean": b.mean}[how](axis=2)


def site_targets(raw: dict, z_m: np.ndarray, patch: int, theta_sfc_pert: np.ndarray) -> dict:
    """Per-site (T, P) diagnostics from `ks_latent.cm1.io.read_grads` output (raw CM1 fields):
    w_max (m/s, column max), rain_rate (mm/h, surface `prate`), qr_low (g/kg, max below 1 km), qc_max
    (g/kg), qs_aloft / qg_aloft / qi_aloft (g/kg, max above 4 km), dbz_max (column max), th_sfc_min (K,
    surface theta' from the model's transform, `theta_sfc_pert` (T, NX))."""
    T = min(raw["winterp"].shape[0], theta_sfc_pert.shape[0])
    low, high = z_m < 1000.0, z_m > 4000.0
    colmax = lambda v, sel=slice(None): raw[v][:T][:, sel].max(axis=1)       # (T, NX)
    out = {"w_max": site_reduce(colmax("winterp"), patch),
           "rain_rate": site_reduce(raw["prate"][:T].astype(np.float64) * 3600.0, patch),
           "qr_low": site_reduce(colmax("qr", low) * 1e3, patch),
           "qc_max": site_reduce(colmax("qc") * 1e3, patch),
           "qs_aloft": site_reduce(colmax("qs", high) * 1e3, patch),
           "qg_aloft": site_reduce(colmax("qg", high) * 1e3, patch),
           "qi_aloft": site_reduce(colmax("qi", high) * 1e3, patch),
           "dbz_max": site_reduce(colmax("dbz"), patch),
           "th_sfc_min": site_reduce(theta_sfc_pert[:T], patch, "min")}
    return {k: v.astype(np.float32) for k, v in out.items()}


# ---------------------------------------------------------------- events
def event_candidates(series: np.ndarray, thr: float, tau: int, quiet: int, above: bool = True):
    """Onset labelling on a (T, P) series. Returns (t, p, label): candidates are (t, p) with the series on
    the quiet side of `thr` for frames t - quiet + 1 ... t, and t + tau inside the record; label 1 if the
    threshold is crossed within (t, t + tau]. `above=False` for quantities that must DROP below thr
    (e.g. surface theta' for cold-pool arrival)."""
    act = series >= thr if above else series <= thr
    T, P = act.shape
    c = np.cumsum(np.concatenate([np.zeros((1, P), int), act.astype(int)]), axis=0)   # c[t] = active count in [0, t)
    ts, ps, ys = [], [], []
    for t in range(quiet - 1, T - tau):
        quiet_ok = (c[t + 1] - c[t + 1 - quiet]) == 0                   # inactive on t-quiet+1 .. t
        future = (c[t + tau + 1] - c[t + 1]) > 0                        # active somewhere in t+1 .. t+tau
        p = np.nonzero(quiet_ok)[0]
        ts.append(np.full(p.size, t)); ps.append(p); ys.append(future[p])
    if not ts:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    return np.concatenate(ts), np.concatenate(ps), np.concatenate(ys)


def near_active(series: np.ndarray, thr: float, half: int, above: bool = True) -> np.ndarray:
    """(T, P) bool: some site within +-half sites is active (the event is happening nearby)."""
    act = series >= thr if above else series <= thr
    T, P = act.shape
    c = np.concatenate([np.zeros((T, 1), int), np.cumsum(act.astype(int), axis=1)], axis=1)   # c[:, p] = count in [0, p)
    lo = np.clip(np.arange(P) - half, 0, P); hi = np.clip(np.arange(P) + half + 1, 0, P)
    return (c[:, hi] - c[:, lo]) > 0


def onsets(series: np.ndarray, thr: float, quiet: int, above: bool = True) -> list[tuple[int, int]]:
    """(t, p) of onsets: active at t after `quiet` inactive frames."""
    act = series >= thr if above else series <= thr
    T, P = act.shape
    out = []
    for p in range(P):
        run = 0
        for t in range(T):
            if act[t, p]:
                if run >= quiet:
                    out.append((t, p))
                run = 0
            else:
                run += 1
    return out


# ---------------------------------------------------------------- features
def window_features(F: np.ndarray, ts: np.ndarray, ps: np.ndarray, half: int) -> np.ndarray:
    """F (T, P, k) site features -> (n, (2 half + 1) k): the window of sites p - half ... p + half at time t,
    edge sites repeated (open domain)."""
    T, P, k = F.shape
    off = np.arange(-half, half + 1)
    idx = np.clip(ps[:, None] + off[None, :], 0, P - 1)                  # (n, 2h+1)
    return F[ts[:, None], idx].reshape(ts.size, -1)


def pca_fit(X: np.ndarray, k: int):
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k], (S[:k] ** 2) / (S ** 2).sum()


# ---------------------------------------------------------------- probes and scores
def standardize_fit(X: np.ndarray):
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd > 1e-8, sd, 1.0)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, epochs: int = 300, seed: int = 0) -> dict:
    """L2-regularized logistic regression, full batch L-BFGS (torch, float64), on standardized features."""
    mu, sd = standardize_fit(X)
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    torch.manual_seed(seed)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    pos = float(yt.mean().clamp(1e-6, 1 - 1e-6))
    with torch.no_grad():
        b.fill_(np.log(pos / (1 - pos)))
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=epochs, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"mu": mu, "sd": sd, "w": w.detach().numpy(), "b": float(b.detach())}


def predict_logistic(m: dict, X: np.ndarray) -> np.ndarray:
    z = ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]
    return 1.0 / (1.0 + np.exp(-z))


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> dict:
    mu, sd = standardize_fit(X)
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + l2 * len(y) * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return {"mu": mu, "sd": sd, "w": w, "b": ym}


def predict_ridge(m: dict, X: np.ndarray) -> np.ndarray:
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC by the rank-sum formula, ties averaged."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    sv = np.asarray(s)[order]
    i = 0
    while i < len(sv):                       # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier_skill(y: np.ndarray, p: np.ndarray, base_rate: float) -> float:
    y = np.asarray(y, float)
    bs = ((p - y) ** 2).mean()
    ref = ((base_rate - y) ** 2).mean()
    return float(1 - bs / ref) if ref > 0 else float("nan")


# ---------------------------------------------------------------- regimes
def kmeans(X: np.ndarray, k: int, iters: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means (k-means++ init). Returns (centres (k, d), labels (n,))."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
        Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    return C, np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
