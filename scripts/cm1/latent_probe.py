"""Plan P (P0-P5, P7): what the CM1 latent field knows, and what in it predicts convective events.

P0  targets per 2 km site and minute from the RAW CM1 output (species, surface rain rate, w, reflectivity)
    and from the model's surface theta'; onsets of rain, updrafts, snow aloft, cold-pool arrival.
P1  diagnostic probes (lead 0): ridge from the site latent / its window, against PCA-32 of the physical
    columns; rain vs cloud water separability (the model's inputs lump them).
P2  regime atlas: k-means on site latents, composite physics per regime, transitions over 5 min.
P3  event prediction within tau = 5/10/20/30 min for currently QUIET sites, from latent (+ its 5-min
    motion), PCA-32 and a 6-number physical summary on the same windows, and the surrogate's own forecast.
    Windows widen with the lead (full window to +-8 sites, ring means beyond, out to ~2 tau km).
P4  position vs motion vs neighbourhood ablation; superposed-epoch trajectories aligned at rain onset
    against matched non-onset controls.
P5  latent traversals along the probe directions (decode and look at the column).
P7  storm-scale: pooled latents -> total rain, cold-pool minimum, max w, now and 30 min ahead.
Probes are trained on training storms of the run and scored on its test storms (and on --extra storms,
e.g. the two-storm T2 runs). Writes p.json and figures to --out.
  python scripts/cm1/latent_probe.py --run ~/cm1_runs/full_v1_stagec --bundle ~/cm1_bundle --out ~/cm1_da/p_v1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run  # noqa: E402
from ks_latent.cm1.data import ColumnData, member_from_cm1io  # noqa: E402
from ks_latent.cm1.io import load_grads_run, read_grads  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.probe import (auc, brier_skill, event_candidates, fit_logistic, fit_mlp, fit_ridge, kmeans, near_active, onsets, predict_mlp,  # noqa: E402
                                 pca_fit, predict_logistic, predict_ridge, r2, site_reduce, site_targets,
                                 window_features)
from ks_latent.cm1.train import encode_members  # noqa: E402

EVENTS = {  # name: (target, threshold, above?)
    "rain_onset": ("rain_rate", 1.0, True), "updraft_5": ("w_max", 5.0, True),
    "snow_aloft": ("qs_aloft", 0.1, True), "cold_pool": ("th_sfc_min", -2.0, False)}
SURROGATE = {"rain_onset": "liq_low", "updraft_5": "w_max", "snow_aloft": "ice_aloft", "cold_pool": "th_sfc"}
LEADS = (5, 10, 20, 30)
QUIET = 20


def raw_dir(bundle: Path, name: str) -> Path:
    for pre, ens in (("ens_A2_", "ens_A2"), ("ens_B_", "ens_B"), ("ens_T2_", "ens_T2")):
        if name.startswith(pre):
            return bundle / ens / "members" / name[len(pre):]
    return bundle / "ens_A" / "members" / name


def summary_feats(X: np.ndarray, norm, z_m, patch) -> np.ndarray:
    """(T, V, Z, NX) physical -> (T, P, 6): column-max w, column-max condensate, low-level liquid, surface
    theta', low-level qv', column-max |u'|; the 'what a forecaster would look at' baseline."""
    vs = norm.vars
    g = lambda v: X[:, vs.index(v)]
    low = z_m < 1000
    cond = (np.clip(g("qliq"), 0, None) ** 2 + np.clip(g("qice"), 0, None) ** 2).max(axis=1)
    f = [site_reduce(g("w").max(axis=1), patch), site_reduce(cond, patch),
         site_reduce((np.clip(g("qliq")[:, low], 0, None) ** 2).max(axis=1), patch),
         site_reduce(g("th")[:, 0], patch, "min"), site_reduce(g("qv")[:, low].mean(axis=1), patch, "mean"),
         site_reduce(np.abs(g("u")).max(axis=1), patch)]
    return np.stack(f, -1).astype(np.float32)


def context(F: np.ndarray, ts, ps, tau: int) -> np.ndarray:
    """Full window to +-min(tau, 8) sites, then ring means (left and right separately) over 9-16 and 17-32
    sites when the lead allows (info travels ~1 site per min)."""
    full = min(tau, 8)
    parts = [window_features(F, ts, ps, full)]
    P = F.shape[1]
    for lo, hi in ((9, 16), (17, 32)):
        if tau < lo:
            break
        for sgn in (-1, 1):
            idx = np.clip(ps[:, None] + sgn * np.arange(lo, hi + 1)[None], 0, P - 1)
            parts.append(F[ts[:, None], idx].mean(axis=1))
    return np.concatenate(parts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bundle", required=True, help="CM1 bundle with the raw ensembles (ens_A, ens_A2, ens_B, ens_T2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=30, help="training storms used to fit the probes")
    ap.add_argument("--n-test", type=int, default=0, help="use only the first n test storms (0: all; for a quick check)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra test storms, e.g. ens_T2_mem001 ... (raw only)")
    ap.add_argument("--max-train-samples", type=int, default=150000)
    ap.add_argument("--near-sites", type=int, default=10,
                    help="P3/P4 'near' candidates: the event is active within +- this many sites (20 km) - the hard question")
    ap.add_argument("--surrogate-every", type=int, default=3, help="start-time spacing (frames) for the surrogate baseline")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mlp", action="store_true",
                    help="I-PCA (2): also fit small MLP probes (P1; P3 near-storm events for latent / latent+motion / pca)")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    run = Path(a.run).expanduser(); bundle = Path(a.bundle).expanduser()
    ae, prop, norm, cfg, split = load_run(run, dev)
    halo, patch = cfg["stage_b"]["halo_sites"], ae.cfg.patch
    z_m = BaseState.from_sounding().z
    rng = np.random.default_rng(0)
    tr = split["train"]
    train_names = ([n for n in tr if n.startswith("mem")][:a.n_train // 3] + [n for n in tr if n.startswith("ens_A2_")][:a.n_train // 3]
                   + [n for n in tr if n.startswith("ens_B_")][:a.n_train - 2 * (a.n_train // 3)])
    test_names = list(split["test"])[: a.n_test or None]
    log = lambda s: print(s, flush=True)
    log(f"[P] {run.name}: {len(train_names)} training storms for the probes, {len(test_names)} test storms, extra {a.extra}")

    # ---------------- P0: per-storm latents, physical features, targets ----------------
    D = {}
    pca = {}
    def load(name, mode: str):
        """mode 'pca_sample': return a subsample of site columns (first pass, to fit the PCA);
        'train' / 'test': store latents, PCA features, summary and targets (test also keeps the data)."""
        d = raw_dir(bundle, name)
        run_raw = load_grads_run(d / "cm1out_s.ctl")
        raw = read_grads(d / "cm1out_s.ctl")
        mem = member_from_cm1io(run_raw, name)
        X = mem.X.transpose(0, 2, 3, 1).astype(np.float32)                 # (T, V, Z, NX) physical
        cd = ColumnData([mem], norm, dev)
        with torch.no_grad():
            z = encode_members(ae, cd)[0].cpu().numpy()                      # (T, P, c)
        tgt = site_targets(raw, z_m, patch, X[:, norm.vars.index("th"), 0])
        u = cd.u[0].float().numpy()                                          # (T, NX, C) normalized
        T, NX, C = u.shape
        P = NX // patch
        cols = u[:, :P * patch].reshape(T, P, patch * C)                     # the site's physical columns
        if mode == "pca_sample":
            return cols[::10].reshape(-1, patch * C)
        D[name] = {"z": z, "pca": ((cols - pca["mu"]) @ pca["Vt"].T).astype(np.float32),
                   "vel": np.concatenate([np.zeros_like(z[:5]), z[5:] - z[:-5]]),
                   "sum": summary_feats(X, norm, z_m, patch), "tgt": tgt, "T": T, "P": P}
        if mode == "test":
            D[name]["cd"] = cd
        del cols, u, X, raw, run_raw
        log(f"  [P0] {name}: T {T}, P {P}, events: " + ", ".join(
            f"{e} {len(onsets(tgt[v], thr, QUIET, ab))}" for e, (v, thr, ab) in EVENTS.items()) + f" [{time.time() - t_start:.0f}s]")
    buf = [load(n, "pca_sample") for n in train_names[:: max(1, len(train_names) // 6)]]
    pca["mu"], pca["Vt"], ev = pca_fit(np.concatenate(buf), 32)
    del buf
    for n in train_names:
        load(n, "train")
    for n in test_names + list(a.extra):
        load(n, "test")
    res = {"train": train_names, "test": test_names, "extra": list(a.extra), "pca_var_explained": float(ev.sum())}
    log(f"[P0] physical PCA-32 explains {100 * ev.sum():.1f}% of the site-column variance")

    # ---------------- P1: diagnostic probes ----------------
    P1 = {}
    diag = {"log_rain_rate": ("rain_rate", np.log1p), "w_max": ("w_max", None), "log_qr_low": ("qr_low", np.log1p),
            "log_qs_aloft": ("qs_aloft", np.log1p), "log_qg_aloft": ("qg_aloft", np.log1p), "th_sfc_min": ("th_sfc_min", None)}
    def samples(names, n_per):
        out = []
        for n in names:
            T, P = D[n]["T"], D[n]["P"]
            t = rng.integers(5, T, n_per); p = rng.integers(0, P, n_per)
            out.append((n, t, p))
        return out
    def feats(sm, key, half):
        return np.concatenate([window_features(D[n][key], t, p, half) for n, t, p in sm])
    def targ(sm, v):
        return np.concatenate([D[n]["tgt"][v][t, p] for n, t, p in sm])
    s_tr, s_te = samples(train_names, 4000), samples(test_names, 4000)
    for fname, (key, half) in {"latent_site": ("z", 0), "latent_window2": ("z", 2), "pca_site": ("pca", 0),
                               "pca_window2": ("pca", 2), "summary_window2": ("sum", 2)}.items():
        Xtr, Xte = feats(s_tr, key, half), feats(s_te, key, half)
        for tname, (v, f) in diag.items():
            ytr, yte = targ(s_tr, v), targ(s_te, v)
            if f is not None:
                ytr, yte = f(np.clip(ytr, 0, None)), f(np.clip(yte, 0, None))
            m = fit_ridge(Xtr, ytr, l2=1e-3)
            P1.setdefault(tname, {})[fname] = r2(yte, predict_ridge(m, Xte))
            if a.mlp and fname != "summary_window2":
                P1[tname][fname + "_mlp"] = r2(yte, predict_mlp(fit_mlp(Xtr, ytr, "reg"), Xte))
    # rain vs cloud water among liquid-bearing low-level sites (the model input is total liquid only)
    def rain_vs_cloud(sm, key, half):
        X, y = [], []
        for n, t, p in sm:
            liq = D[n]["tgt"]["qr_low"][t, p] + 0.0
            qc = D[n]["tgt"]["qc_max"][t, p]
            sel = (liq + qc) > 0.2
            X.append(window_features(D[n][key], t[sel], p[sel], half)); y.append(liq[sel] > qc[sel])
        return np.concatenate(X), np.concatenate(y)
    for fname, (key, half) in {"latent_window2": ("z", 2), "pca_window2": ("pca", 2), "summary_window2": ("sum", 2)}.items():
        Xtr, ytr = rain_vs_cloud(samples(train_names, 20000), key, half)
        Xte, yte = rain_vs_cloud(samples(test_names, 20000), key, half)
        if ytr.size > 100 and 0 < ytr.mean() < 1 and yte.size > 50:
            P1.setdefault("rain_vs_cloud_auc", {})[fname] = auc(yte, predict_logistic(fit_logistic(Xtr, ytr), Xte))
            if a.mlp and fname != "summary_window2":
                P1["rain_vs_cloud_auc"][fname + "_mlp"] = auc(yte, predict_mlp(fit_mlp(Xtr, ytr, "clf"), Xte))
    res["P1_r2"] = P1
    log("[P1] test R^2 (AUC for rain_vs_cloud):")
    for tname, row in P1.items():
        log(f"   {tname:18s} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))

    # ---------------- P2: regime atlas ----------------
    s2 = samples(test_names, 3000)
    Z2 = feats(s2, "z", 0); mu2, sd2 = Z2.mean(0), Z2.std(0) + 1e-6
    K = 8
    C, lab = kmeans((Z2 - mu2) / sd2, K, seed=0)
    comp = {}
    for v in ("w_max", "rain_rate", "qr_low", "qc_max", "qs_aloft", "qg_aloft", "th_sfc_min", "dbz_max"):
        y = targ(s2, v)
        comp[v] = [float(y[lab == k].mean()) for k in range(K)]
    frac = [float((lab == k).mean()) for k in range(K)]
    # transitions over 5 minutes
    trans = np.zeros((K, K))
    for n in test_names[:10]:
        z = D[n]["z"]; T, P = z.shape[:2]
        t = rng.integers(0, T - 5, 2000); p = rng.integers(0, P, 2000)
        assign = lambda Zs: np.argmin(((((Zs - mu2) / sd2)[:, None] - C[None]) ** 2).sum(-1), axis=1)
        l0, l1 = assign(z[t, p]), assign(z[t + 5, p])
        np.add.at(trans, (l0, l1), 1)
    trans = trans / np.maximum(trans.sum(1, keepdims=True), 1)
    res["P2"] = {"K": K, "fraction": frac, "composites": comp, "transition_5min": trans.tolist()}
    log("[P2] regimes (fraction | mean w_max, rain mm/h, qr_low, qs_aloft, qg_aloft, th_sfc_min):")
    for k in np.argsort([-comp["w_max"][k] for k in range(K)]):
        log(f"   regime {k}: {frac[k]:.3f} | w {comp['w_max'][k]:6.2f} rain {comp['rain_rate'][k]:6.2f} qr {comp['qr_low'][k]:5.2f} "
            f"qs {comp['qs_aloft'][k]:5.2f} qg {comp['qg_aloft'][k]:5.2f} th {comp['th_sfc_min'][k]:6.2f}   stay 5 min {trans[k, k]:.2f}")

    # ---------------- P3 / P4: event prediction ----------------
    def cand(names, ev, tau, near=None):
        """Quiet candidates; with near = True only those with the event active within +-near_sites now
        (the hard question: the storm is close, does it start HERE within tau?)."""
        v, thr, ab = EVENTS[ev]
        out = []
        for n in names:
            S = D[n]["tgt"][v]
            ts, ps, ys = event_candidates(S, thr, tau, QUIET, ab)
            ok = ts >= 5
            if near:
                ok &= near_active(S, thr, a.near_sites, ab)[ts, ps]
            out.append((n, ts[ok], ps[ok], ys[ok]))
        return out
    def subsample(cs, maxn):
        n_all = sum(len(c[1]) for c in cs)
        pos = sum(int(c[3].sum()) for c in cs)
        keep_neg = min(1.0, (maxn - min(pos, maxn // 3)) / max(n_all - pos, 1))
        out = []
        for n, ts, ps, ys in cs:
            k = ys | (rng.random(ys.size) < keep_neg)
            out.append((n, ts[k], ps[k], ys[k]))
        return out, keep_neg
    def build(cs, kind, tau):
        Xs = []
        for n, ts, ps, ys in cs:
            if kind == "latent":
                Xs.append(context(D[n]["z"], ts, ps, tau))
            elif kind == "latent+motion":
                Xs.append(np.concatenate([context(D[n]["z"], ts, ps, tau), context(D[n]["vel"], ts, ps, tau)], 1))
            elif kind == "latent_site_only":
                Xs.append(window_features(D[n]["z"], ts, ps, 0))
            elif kind == "latent_site+motion":
                Xs.append(np.concatenate([window_features(D[n]["z"], ts, ps, 0), window_features(D[n]["vel"], ts, ps, 0)], 1))
            elif kind == "pca":
                Xs.append(context(D[n]["pca"], ts, ps, tau))
            elif kind == "summary":
                Xs.append(context(D[n]["sum"], ts, ps, tau))
        return np.concatenate(Xs)
    P3, models = {}, {}
    kinds = ("latent", "latent+motion", "pca", "summary", "latent_site_only", "latent_site+motion")
    for ev, near in [(e, nr) for e in EVENTS for nr in (True, False)]:
        evk = ev if near else ev + "_all_quiet"
        for tau in LEADS:
            ctr, keep = subsample(cand(train_names, ev, tau, near), a.max_train_samples)
            cte = cand(test_names, ev, tau, near)
            ytr = np.concatenate([c[3] for c in ctr]); yte = np.concatenate([c[3] for c in cte])
            if ytr.sum() < 30 or yte.sum() < 10:
                log(f"   [P3] {evk} tau {tau}: too few events ({int(ytr.sum())} train, {int(yte.sum())} test)")
                continue
            base = float(yte.mean())
            row = {"n_test": int(yte.size), "events_test": int(yte.sum()), "base_rate": base}
            for kind in kinds:
                m = fit_logistic(build(ctr, kind, tau), ytr, l2=1e-3)
                p = predict_logistic(m, build(cte, kind, tau))
                # undo the negative subsampling in the probabilities (prior correction) for the Brier score
                odds = p / np.maximum(1 - p, 1e-12) * keep
                pc = odds / (1 + odds)
                row[kind] = {"auc": auc(yte, p), "bss": brier_skill(yte, pc, base)}
                if kind == "latent+motion" and near:
                    models[(ev, tau)] = m
                if a.mlp and near and kind in ("latent", "latent+motion", "pca"):
                    pm = predict_mlp(fit_mlp(build(ctr, kind, tau), ytr, "clf"), build(cte, kind, tau))
                    om = pm / np.maximum(1 - pm, 1e-12) * keep
                    row[kind + "_mlp"] = {"auc": auc(yte, pm), "bss": brier_skill(yte, om / (1 + om), base)}
            P3.setdefault(evk, {})[str(tau)] = row
            log(f"   [P3] {evk:20s} tau {tau:2d}: base {base:.3f}  AUC " + "  ".join(f"{k} {row[k]['auc']:.3f}" for k in row
                                                                                if isinstance(row[k], dict))
                + f"  [{time.time() - t_start:.0f}s]")
    # surrogate baseline: roll the latent from each start, decode, and keep the running extreme over leads 1..tau
    # of each event's decoded quantity; one rollout per start serves every event and lead
    tau_max = max(LEADS)
    def surrogate_cache(n):
        d = D[n]; cd = d["cd"]; z = torch.as_tensor(d["z"])
        vs = norm.vars; Zl = cd.Z
        chs = torch.tensor(norm.channel_scale(Zl), dtype=torch.float32)
        low = torch.as_tensor(z_m < 1000); high = torch.as_tensor(z_m > 4000)
        starts = np.arange(5, d["T"] - tau_max, a.surrogate_every)
        out = {k: np.zeros((starts.size, tau_max, d["P"]), np.float32) for k in set(SURROGATE.values())}
        for si, t0 in enumerate(starts):
            with torch.no_grad():
                lat = rollout_from(prop, z, int(t0), tau_max, halo, boundary="persistence")[:, 0]
                dec = ae.decode(lat) * chs[None, :, None]                    # (tau, C, NX) physical
            q = lambda v: dec[:, vs.index(v) * Zl:(vs.index(v) + 1) * Zl]
            f = {"liq_low": (q("qliq").clamp(min=0) ** 2)[:, low].amax(dim=1), "w_max": q("w").amax(dim=1),
                 "ice_aloft": (q("qice").clamp(min=0) ** 2)[:, high].amax(dim=1), "th_sfc": -q("th")[:, 0]}
            for k, v in f.items():
                run = torch.cummax(v, dim=0).values.numpy()                  # running extreme over the leads
                out[k][si] = site_reduce(run, patch)
        return starts, out
    sur = {n: surrogate_cache(n) for n in test_names}
    log(f"[P3] surrogate forecasts done [{time.time() - t_start:.0f}s]")
    for ev in EVENTS:
        for tau in LEADS:
            if ev not in P3 or str(tau) not in P3[ev]:
                continue
            sc, yy = [], []
            for n, ts, ps, ys in cand(test_names, ev, tau, True):
                starts, cache = sur[n]
                pos = {int(t0): i for i, t0 in enumerate(starts)}
                k = np.array([t in pos for t in ts])
                if k.any():
                    si = np.array([pos[t] for t in ts[k]])
                    sc.append(cache[SURROGATE[ev]][si, tau - 1, ps[k]]); yy.append(ys[k])
            if sc:
                sc, yy = np.concatenate(sc), np.concatenate(yy)
                P3[ev][str(tau)]["surrogate_forecast"] = {"auc": auc(yy, sc), "n": int(sc.size)}
                log(f"   [P3] {ev:10s} tau {tau:2d}: surrogate forecast AUC {auc(yy, sc):.3f} ({sc.size} cases)")
    # extra storms (e.g. two-storm T2): latent+motion probe trained above
    for ev in EVENTS:
        for tau in LEADS:
            m = models.get((ev, tau))
            if m is None or not a.extra:
                continue
            cx = cand(list(a.extra), ev, tau, True)
            y = np.concatenate([c[3] for c in cx])
            if y.sum() >= 10:
                P3[ev][str(tau)]["extra_latent+motion_auc"] = auc(y, predict_logistic(m, build(cx, "latent+motion", tau)))
    res["P3"] = P3

    # ---------------- P4: superposed epoch at rain onset ----------------
    ev, tau = "rain_onset", 10
    if (ev, tau) in models:
        m = models[(ev, tau)]
        win = np.arange(-30, 11)
        trajs_e, trajs_c = [], []
        for n in test_names:
            v, thr, ab = EVENTS[ev]
            S = D[n]["tgt"][v]
            ons = [(t, p) for t, p in onsets(S, thr, QUIET, ab) if t + win[0] >= 5 and t + win[-1] < D[n]["T"]]
            for t, p in ons:
                ts = t + win
                ps = np.full(win.size, p)
                trajs_e.append(predict_logistic(m, build([(n, ts, ps, None)], "latent+motion", tau)))
                # matched control: a quiet site at the same time whose physical summary is nearest, no rain within 30 min
                nearby = near_active(S, thr, a.near_sites, ab)[t]
                cands = [q for q in range(D[n]["P"]) if q != p and nearby[q] and (S[t - QUIET:t + 31, q] < thr).all()]
                if cands:
                    sm = D[n]["sum"][t - 10]
                    q = cands[int(np.argmin(((sm[cands] - sm[p]) ** 2).sum(-1)))]
                    trajs_c.append(predict_logistic(m, build([(n, ts, np.full(win.size, q), None)], "latent+motion", tau)))
        if trajs_e and trajs_c:
            E, Cc = np.array(trajs_e), np.array(trajs_c)
            me, se = E.mean(0), E.std(0) / np.sqrt(len(E))
            mc, sc_ = Cc.mean(0), Cc.std(0) / np.sqrt(len(Cc))
            sep = win[np.nonzero((me - mc) > 2 * np.sqrt(se ** 2 + sc_ ** 2))[0]]
            res["P4_epoch"] = {"lags": win.tolist(), "event_mean": me.tolist(), "control_mean": mc.tolist(),
                               "n_events": len(E), "n_controls": len(Cc), "first_separation_lag": int(sep.min()) if sep.size else None}
            fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
            ax.plot(win, me, label=f"rain onset at lag 0 (n = {len(E)})"); ax.fill_between(win, me - 2 * se, me + 2 * se, alpha=0.3)
            ax.plot(win, mc, label=f"matched controls (n = {len(Cc)})"); ax.fill_between(win, mc - 2 * sc_, mc + 2 * sc_, alpha=0.3)
            ax.axvline(0, color="0.5", lw=1, ls=":"); ax.set_xlabel("minutes relative to onset")
            ax.set_ylabel("P(rain within 10 min), latent probe"); ax.legend(frameon=False, fontsize=7)
            fig.savefig(out / "fig_p4_epoch.png", dpi=150)
            log(f"[P4] superposed epoch: {len(E)} onsets, {len(Cc)} controls; first separation (2 SE) at lag "
                f"{res['P4_epoch']['first_separation_lag']} min")

    # ---------------- P5: traversals along the probe directions ----------------
    def traverse(direction, name):
        n = test_names[0]; d = D[n]
        T, P = d["T"], d["P"]
        cand_sites = [(t, p) for t in range(60, T - 1, 30) for p in range(halo + 2, P - halo - 2)
                      if d["tgt"]["rain_rate"][t, p] < 0.1 and d["sum"][t, p, 1] > 0.05][:200]
        if not cand_sites:
            return None
        prof = {}
        Zl = d["cd"].Z
        chs = torch.tensor(norm.channel_scale(Zl), dtype=torch.float32)
        for alpha in (-2, -1, 0, 1, 2):
            acc = 0
            for t, p in cand_sites:
                z = torch.as_tensor(d["z"][t]).clone()
                z[p] = z[p] + alpha * torch.as_tensor(direction, dtype=z.dtype)
                with torch.no_grad():
                    u = ae.decode(z[None])[0] * chs[:, None]
                acc = acc + u[:, p * patch:(p + 1) * patch].mean(-1).numpy()
            prof[alpha] = acc / len(cand_sites)
        vs = norm.vars
        fig, axs = plt.subplots(1, 4, figsize=(8, 3), sharey=True, constrained_layout=True)
        for ax, v in zip(axs, ("w", "th", "qliq", "qice")):
            i = vs.index(v)
            for alpha, pr in prof.items():
                x = pr[i * Zl:(i + 1) * Zl]
                ax.plot(np.clip(x, 0, None) ** 2 if v in ("qliq", "qice") else x, z_m / 1000, label=f"{alpha:+d}σ")
            ax.set_title(v); ax.set_ylim(0, 14)
        axs[0].set_ylabel("z (km)"); axs[-1].legend(fontsize=6, frameon=False)
        fig.suptitle(f"Moving dry storm sites along the {name} direction (decoded mean column)", fontsize=8)
        fig.savefig(out / f"fig_p5_traverse_{name}.png", dpi=150)
        return {str(k): v.tolist() for k, v in prof.items()}
    # direction = the site-centre weights of the probe (standardized space), mapped back to latent units
    c = D[test_names[0]]["z"].shape[-1]
    if ("rain_onset", 10) in models:
        m = models[("rain_onset", 10)]
        full = min(10, 8); centre = slice(full * c, (full + 1) * c)
        dvec = m["w"][centre] / m["sd"][centre]
        dvec = dvec / np.linalg.norm(dvec) * np.median(np.linalg.norm(D[test_names[0]]["z"], axis=-1)) * 0.25
        traverse(dvec, "rain_onset")
    mw = fit_ridge(feats(s_tr, "z", 0), targ(s_tr, "w_max"), l2=1e-3)
    dw = mw["w"] / mw["sd"]; dw = dw / np.linalg.norm(dw) * np.median(np.linalg.norm(D[test_names[0]]["z"], axis=-1)) * 0.25
    traverse(dw, "updraft")

    # ---------------- P7: storm scale from pooled latents ----------------
    def pooled(n, key):
        d = D[n]; F = d[key]; act = d["sum"][..., 1] > 0.05                  # condensate-bearing sites
        T = d["T"]
        mean_all = F.mean(1); std_all = F.std(1); mx = F.max(1); mn = F.min(1)
        m_act = np.array([F[t][act[t]].mean(0) if act[t].any() else np.zeros(F.shape[-1]) for t in range(T)])
        n_act = act.sum(1, keepdims=True).astype(np.float32)
        return np.concatenate([mean_all, std_all, mx, mn, m_act, n_act], 1)
    def storm_targets(n):
        g = D[n]["tgt"]
        return {"total_rain": g["rain_rate"].sum(1), "th_min": g["th_sfc_min"].min(1), "w_max": g["w_max"].max(1)}
    P7 = {}
    for lag in (0, 30):
        for key in ("z", "pca", "sum"):
            Xtr = np.concatenate([pooled(n, key)[30:D[n]["T"] - lag] for n in train_names])
            Xte = np.concatenate([pooled(n, key)[30:D[n]["T"] - lag] for n in test_names])
            for tname in ("total_rain", "th_min", "w_max"):
                ytr = np.concatenate([storm_targets(n)[tname][30 + lag:] for n in train_names])
                yte = np.concatenate([storm_targets(n)[tname][30 + lag:] for n in test_names])
                P7.setdefault(f"{tname}_lag{lag}", {})[key] = r2(yte, predict_ridge(fit_ridge(Xtr, ytr, l2=1e-2), Xte))
    res["P7_r2"] = P7
    log("[P7] storm-scale R^2 from pooled features (latent z | physical PCA | physical summary):")
    for k, row in P7.items():
        log(f"   {k:18s} " + "  ".join(f"{kk} {vv:.3f}" for kk, vv in row.items()))

    (out / "p.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"[P] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
