"""Track FP, FP1: train 2D latent-field autoencoders (`ks_latent.cm1.field_ae`) over a sweep of field counts m,
latent-grid strides and decoders, on the reference run's split and normalizer; data loaded once.

For every configuration "m:stride_x:stride_z:decoder" (e.g. 4:1:1:jet) it trains on the reference run's
training storms (storm-weighted random windows, full height), keeps the checkpoint with the lowest
validation MSE, and scores the test storms with the same `reconstruction_report` the column autoencoders
use (nMSE over all and storm columns, physical RMSE, divergence ratio, condensate mass), plus a
rediscovery probe: pointwise MLP R^2 of hand-built physical fields -- theta_il', q_t' (conserved under
phase change), the perturbation vorticity eta', buoyancy B, and the non-conserved theta', q_v' -- read
from phi (interpolated to the data grid) and height at the same point, fitted on half the test storms
and scored on the other half. What reconstruction alone makes the latent keep; a dynamics loss (FP2) is
what should favour conserved combinations.
  python scripts/cm1/train_field_ae.py --profile full --ref-run ~/cm1_runs/full_v1 \\
      --source cache:~/cm1_cache+~/cm1_cache_A2+~/cm1_cache_B --out ~/cm1_runs/f1 \\
      --configs 1:1:1:jet 2:1:1:jet 2:1:1:point 4:2:1:jet --device cuda
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from ks_latent.cm1.data import (ColumnData, Normalizer, load_member, positive_channel_mask,  # noqa: E402
                                split_members, synthetic_members)
from ks_latent.cm1.evaluate import reconstruction_report  # noqa: E402
from ks_latent.cm1.field_ae import FieldAE, FieldAEConfig  # noqa: E402
from ks_latent.cm1.physics import CP, G, RD, BaseState  # noqa: E402
from ks_latent.cm1.probe import fit_mlp, predict_mlp  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402

LV, LS = 2.501e6, 2.834e6


def parse_config(s: str, V: int, Z: int, width: int) -> FieldAEConfig:
    m, sx, sz, dec = s.split(":")
    return FieldAEConfig(n_vars=V, n_levels=Z, m=int(m), stride_x=int(sx), stride_z=int(sz), decoder=dec, width=width)


def physical_targets(x: np.ndarray, norm: Normalizer, base: BaseState | None) -> dict:
    """Hand-built fields from one normalized frame (C, NX) -> {name: (Z, NX)}, physical units."""
    V, Z = len(norm.vars), x.shape[0] // len(norm.vars)
    X = x.reshape(V, Z, -1) * norm.scale[:, None, None]
    g = {v: X[i] for i, v in enumerate(norm.vars)}
    ql = sum(g[k] ** 2 for k in ("qliq", "qc", "qr") if k in g) / 1000.0
    qi = sum(g[k] ** 2 for k in ("qice", "qi", "qs", "qg") if k in g) / 1000.0
    if base is not None and len(base.z) == Z:
        th0, z = base.theta0[:, None], base.z
        pi0 = ((base.rho0 * RD * base.theta0 / 1e5) ** (RD / (CP - RD)))[:, None]
    else:                                                  # synthetic smoke data: nominal profiles
        th0, z, pi0 = np.full((Z, 1), 300.0), np.arange(Z) * 300.0, np.ones((Z, 1))
    out = {"theta": g["th"], "qv": g["qv"],
           "theta_il": g["th"] - (LV * ql + LS * qi) / (CP * pi0),
           "qt": g["qv"] + 1000.0 * (ql + qi),
           "eta": np.gradient(g["u"], z, axis=0) - np.gradient(g["w"], 250.0, axis=1),
           "B": G * (g["th"] / th0 + 0.608e-3 * g["qv"] - ql - qi)}
    return out


def probe(ae: FieldAE, d: ColumnData, norm: Normalizer, base, dev, n_per_storm: int, seed: int) -> dict:
    """Pointwise MLP R^2 of physical fields from (phi at the point, height features), storm columns."""
    rng = np.random.default_rng(seed)
    Xs, Ys, S = [], {}, []
    for m in range(len(d.u)):
        frames = rng.choice(np.arange(d.T[m] // 6, d.T[m]), size=min(6, d.T[m] - d.T[m] // 6), replace=False)
        for t in frames:
            u = d.full(m, int(t), device=dev)
            with torch.no_grad():
                phi = ae.encode(u)
            Z, NX = ae.cfg.n_levels, u.shape[-1]
            ph = torch.nn.functional.interpolate(phi, size=(phi.shape[-2] * ae.cfg.stride_z, phi.shape[-1] * ae.cfg.stride_x),
                                                 mode="bilinear", align_corners=False)[0, :, :Z, :NX].cpu().numpy()
            act = (d.cref[m][int(t)] > 10).numpy()
            kk, ii = np.nonzero(np.broadcast_to(act[None], (Z, NX)))
            if not len(kk):
                continue
            sel = rng.choice(len(kk), size=min(n_per_storm // len(frames), len(kk)), replace=False)
            kk, ii = kk[sel], ii[sel]
            zf = ae.zfeat.cpu().numpy()[:, kk].T
            Xs.append(np.concatenate([ph[:, kk, ii].T, zf], axis=1))
            for name, f in physical_targets(u[0].cpu().numpy(), norm, base).items():
                Ys.setdefault(name, []).append(f[kk, ii])
            S.append(np.full(len(kk), m))
    X, S = np.concatenate(Xs), np.concatenate(S)
    tr = S % 2 == 0
    out = {}
    for name, ys in Ys.items():
        y = np.concatenate(ys)
        mdl = fit_mlp(X[tr], y[tr], "reg", epochs=40, seed=seed)
        p = predict_mlp(mdl, X[~tr])
        out[name] = float(1 - np.mean((y[~tr] - p) ** 2) / np.var(y[~tr]))
    return out


def train_one(cfg, d_tr, d_va, pos, a, dev, log) -> tuple:
    torch.manual_seed(a.seed)
    ae = FieldAE(cfg, pos).to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=a.lr, weight_decay=1e-5)
    warm = min(500, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    gen = torch.Generator().manual_seed(a.seed)
    vgen = torch.Generator().manual_seed(a.seed + 1)
    val = [d_va.windows(a.batch, a.window, 1, vgen)[0][:, 0] for _ in range(a.val_batches)]
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    acc = torch.zeros((), device=dev)
    for step in range(1, a.steps + 1):
        u = d_tr.windows(a.batch, a.window, 1, gen)[0][:, 0].to(dev, non_blocking=True)
        loss = ((ae(u)[0] - u) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt.step(); sched.step()
        acc += loss.detach()
        if step % a.val_every == 0 or step == a.steps:
            ae.eval()
            with torch.no_grad():
                v = float(np.mean([((ae(x.to(dev))[0] - x.to(dev)) ** 2).mean().item() for x in val]))
            ae.train()
            tr_loss = float(acc) / (step - (hist[-1]["step"] if hist else 0))
            acc.zero_()
            hist.append({"step": step, "train": tr_loss, "val": v})
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in ae.state_dict().items()}
            log(f"  [{cfg.m}:{cfg.stride_x}:{cfg.stride_z}:{cfg.decoder}] step {step}/{a.steps} train {tr_loss:.4f} "
                f"val {v:.4f} [{time.time() - t0:.0f}s]")
    ae.load_state_dict(best_state)
    return ae.eval(), hist, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smoke", "full"], default="full")
    ap.add_argument("--ref-run", help="column run whose split and normalizer are used (full profile)")
    ap.add_argument("--source", help="cache:DIR[+DIR...] (full profile)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", required=True, help="m:stride_x:stride_z:decoder")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--probe-points", type=int, default=20000, help="per storm (0: no probe)")
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    dev = torch.device(a.device)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    if a.profile == "smoke":
        mem = synthetic_members(6, T=12, NX=128, Z=8)
        norm = Normalizer.fit(mem)
        tr, va, te = split_members([m.name for m in mem], 1, 2)
        byname = {m.name: m for m in mem}
        get = lambda n: byname[n]
        a.steps, a.val_every, a.window, a.batch, a.val_batches = min(a.steps, 40), 20, 64, 4, 2
        a.width = min(a.width, 16)
        base = None
    else:
        from da_twin import member_files
        ref = Path(a.ref_run).expanduser()
        norm = Normalizer.from_dict(json.loads((ref / "normalizer.json").read_text()))
        sp = json.loads((ref / "results.json").read_text())["split"]
        tr, va, te = sp["train"], sp["val"], sp["test"]
        files = member_files(a.source)
        missing = [n for n in tr + va + te if n not in files]
        if missing:
            raise ValueError(f"members missing from {a.source}: {missing[:5]}")
        get = lambda n: load_member(files[n])
        base = BaseState.from_sounding()
    te = te[: a.max_test or None]
    log(f"[FP1] {len(tr)} train / {len(va)} val / {len(te)} test storms; loading training data on {dev}")
    d_tr = ColumnData([get(n) for n in tr], norm, dev)
    d_va = ColumnData([get(n) for n in va], norm, torch.device("cpu"))
    V, Z = d_tr.V, d_tr.Z
    pos = positive_channel_mask(Z, norm.vars)
    res = {"ref_run": a.ref_run, "split": {"train": tr, "val": va, "test": te}, "runs": {}}
    cfgs = [parse_config(s, V, Z, a.width) for s in a.configs]
    trained = {}
    for s, cfg in zip(a.configs, cfgs):
        ae, hist, best = train_one(cfg, d_tr, d_va, pos, a, dev, log)
        n_par = sum(p.numel() for p in ae.parameters())
        torch.save({"config": cfg.__dict__, "state_dict": ae.state_dict()}, out / f"field_ae_{s.replace(':', '_')}.pt")
        trained[s] = ae
        res["runs"][s] = {"config": cfg.__dict__, "numbers_per_point": cfg.numbers_per_point, "parameters": n_par,
                          "best_val": best, "history": hist}
    del d_tr
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    log(f"[FP1] scoring {len(te)} test storms")
    reps = {s: [] for s in trained}
    for n in te:
        d = ColumnData([get(n)], norm, torch.device("cpu"))
        for s, ae in trained.items():
            with torch.no_grad():
                reps[s].append(reconstruction_report(ae, d, norm))
    d_te = ColumnData([get(n) for n in te], norm, torch.device("cpu")) if a.probe_points else None
    for s, ae in trained.items():
        rs = reps[s]
        r = res["runs"][s]
        r["nmse_all"] = float(np.mean([x["nmse_all"] for x in rs]))
        r["nmse_storm"] = float(np.mean([x["nmse_storm"] for x in rs if "nmse_storm" in x]))
        r["rmse_phys_storm"] = {v: float(np.mean([x["rmse_phys_storm"][v] for x in rs if "rmse_phys_storm" in x]))
                                for v in norm.vars}
        r["physics"] = {p: float(np.mean([x["physics"][p] for x in rs])) for p in rs[0]["physics"]}
        if d_te is not None and len(te) >= 2:
            r["probe_r2"] = probe(ae, d_te, norm, base, dev, a.probe_points, a.seed)
    (out / "f1.json").write_text(json.dumps(res, indent=1))
    write_provenance(out / "f1.json", config=FieldAEConfig(), seed=a.seed, device=str(dev),
                     wall_time_s=time.time() - t_start, extra={"args": vars(a)})
    names = ["theta_il", "qt", "eta", "B", "theta", "qv"]
    log("\n[FP1] test storms (reference, column autoencoder full_v1: 0.065 numbers/point, nMSE 0.022, storm 0.071, div 0.236; truth div 0.224)")
    log("%-14s %8s %8s %8s %8s  %s" % ("config", "num/pt", "nMSE", "storm", "div", "probe R2 " + " ".join(names)))
    for s, r in res["runs"].items():
        pr = r.get("probe_r2", {})
        log("%-14s %8.3f %8.4f %8.4f %8.3f  %s" % (s, r["numbers_per_point"], r["nmse_all"], r["nmse_storm"],
                                                 r["physics"].get("div_ratio_recon", float("nan")),
                                                 " ".join(f"{pr.get(k, float('nan')):.2f}" for k in names)))
    log(f"[FP1] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
