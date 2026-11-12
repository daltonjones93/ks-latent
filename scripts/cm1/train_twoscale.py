"""6a: train the small-scale half of the two-scale model (ks_latent/cm1/twoscale.py) on top of a frozen large-scale run.

Stage A  small-scale autoencoder on the residual r = u - D_L(E_L(u)) (computed on the fly per window; margins where the window edge
         corrupts the large model's reconstruction are excluded), MSE + the usual anti-collapse terms.
Stage S  stochastic small-scale propagator, autoencoders frozen: from the encoded true residual at t, S samples are rolled k steps
         with the large-scale latent TEACHER-FORCED (the true encoded z_L at t and t+1, one-way coupling) and the true small-scale
         latent on a halo; the decoded residual samples are scored against the true residual with the fair CRPS + w_energy x energy
         score + w_spec x log-spectrum mismatch, k ramped to --k-max.
  python scripts/cm1/train_twoscale.py --large-run ~/cm1_runs/fpc_nested_p8_dft --source cache:A+A2+B --out ~/cm1_runs/ts_v1 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.training.losses import decorr_var_loss, logdet_barrier_loss  # noqa: E402
from ks_latent.training.crop import warmup_cosine  # noqa: E402
from ks_latent.cm1.models import ColumnAEConfig, _with_true_halo, build_autoencoder  # noqa: E402
from ks_latent.cm1.train import crps_elements, energy_score, spectral_loss  # noqa: E402
from ks_latent.cm1.twoscale import SmallProp, SmallPropConfig, save_twoscale  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402


def interior(x, m):
    return x[..., m:x.shape[-1] - m] if m else x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--large-run", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--small-channels", type=int, default=16); ap.add_argument("--small-hidden", type=int, default=256)
    ap.add_argument("--noise-channels", type=int, default=8); ap.add_argument("--cond-channels", type=int, default=8)
    ap.add_argument("--prop-hidden", type=int, default=256); ap.add_argument("--prop-layers", type=int, default=2)
    ap.add_argument("--a-steps", type=int, default=6000); ap.add_argument("--a-lr", type=float, default=1e-3)
    ap.add_argument("--s-steps", type=int, default=8000); ap.add_argument("--s-lr", type=float, default=3e-4)
    ap.add_argument("--k-max", type=int, default=30); ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--w-energy", type=float, default=1.0); ap.add_argument("--w-spec", type=float, default=0.1)
    ap.add_argument("--batch-a", type=int, default=32); ap.add_argument("--batch-s", type=int, default=8)
    ap.add_argument("--window-sites", type=int, default=48); ap.add_argument("--halo", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=0, help="0: all training members of the large run's split")
    ap.add_argument("--device", default="auto"); ap.add_argument("--data-device", default="same"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); torch.manual_seed(a.seed)
    dev = get_device(a.device); ddev = dev if a.data_device == "same" else torch.device(a.data_device)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.txt", "a")
    def log(s):
        print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    large = Path(a.large_run).expanduser()
    ae_l, prop_l, norm, cfg, split = load_run(large, dev)
    for q in list(ae_l.parameters()) + list(prop_l.parameters()):
        q.requires_grad_(False)
    files = member_files(a.source)
    tr_names = split["train"][: a.n_train or None]
    tr = ColumnData([load_member(files[n]) for n in tr_names], norm, ddev)
    va = ColumnData([load_member(files[n]) for n in split["val"]], norm, ddev)
    log(f"[ts] large {large.name}; {len(tr_names)} training / {len(split['val'])} validation storms, data on {ddev} ({time.time() - t0:.0f}s)")
    p = ae_l.cfg.patch
    C, Zl = tr.C, tr.Z
    ae_s = build_autoencoder(ColumnAEConfig(n_channels=C, n_levels=Zl, patch=p, hidden=a.small_hidden, latent_channels=a.small_channels), None).to(dev)
    rl = 2 * ae_l.cfg.reach_sites                                  # sites where the window edge corrupts D_L(E_L(u))
    W = a.window_sites * p
    g = torch.Generator().manual_seed(a.seed); vg = torch.Generator().manual_seed(a.seed + 99)

    @torch.no_grad()
    def split_window(u):                                          # u (B, F, C, W) -> z_L (B, F, P, cl), r (B, F, C, W)
        B, Fr = u.shape[:2]
        zl = ae_l.encode(u.reshape(B * Fr, C, W))
        r = u.reshape(B * Fr, C, W) - ae_l.decode(zl)
        return zl.reshape(B, Fr, a.window_sites, -1), r.reshape(B, Fr, C, W)

    # ---------------- Stage A: small-scale autoencoder on the residual ----------------
    mA = (rl + 2 * ae_s.cfg.reach_sites) * p
    drawA = lambda data, gen, B: data.windows(B, W, 1, gen, 0.5)[0].to(dev)
    valA = [split_window(drawA(va, vg, a.batch_a))[1][:, 0] for _ in range(6)]
    opt = torch.optim.AdamW(ae_s.parameters(), lr=a.a_lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, a.a_steps, 200, 0.02))
    best, best_state, histA = float("inf"), None, []
    for step in range(a.a_steps):
        ae_s.train()
        r = split_window(drawA(tr, g, a.batch_a))[1][:, 0]
        z = ae_s.encode(r)
        l_rec = F.mse_loss(interior(ae_s.decode(z), mA), interior(r, mA))
        zs = z[:, mA // p: z.shape[1] - mA // p].reshape(-1, z.shape[-1])
        loss = l_rec + 0.02 * decorr_var_loss(zs)[1] + 0.0035 * logdet_barrier_loss(zs, eps=1e-3)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ae_s.parameters(), 1.0); opt.step(); sch.step()
        if (step + 1) % 500 == 0 or step == a.a_steps - 1:
            ae_s.eval()
            with torch.no_grad():
                v = np.mean([F.mse_loss(interior(ae_s(x)[0], mA), interior(x, mA)).item() for x in valA])
                vr = np.mean([interior(x, mA).pow(2).mean().item() for x in valA])
            histA.append({"step": step + 1, "train": l_rec.item(), "val": float(v), "val_frac_resid_var_explained": float(1 - v / vr)})
            log(f"  [A] {step + 1}/{a.a_steps} recon {l_rec.item():.5f} val {v:.5f} (residual var {vr:.5f}, explained {1 - v / vr:.3f}) [{time.time() - t0:.0f}s]")
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in ae_s.state_dict().items()}
    ae_s.load_state_dict(best_state); ae_s.eval()
    for q in ae_s.parameters():
        q.requires_grad_(False)

    # ---------------- Stage S: stochastic small-scale propagator ----------------
    small = SmallProp(SmallPropConfig(small_channels=a.small_channels, cond_channels=a.cond_channels, noise_channels=a.noise_channels,
                                      hidden=a.prop_hidden, layers=a.prop_layers)).to(dev)
    h = max(a.halo, small.cfg.reach_sites)
    mS = h + ae_s.cfg.reach_sites + rl                            # interior margin in sites for the scored residual
    S, ck = a.samples, a.cond_channels

    @torch.no_grad()
    def encode_window(u):
        zl, r = split_window(u)
        B, Fr = u.shape[:2]
        zs = ae_s.encode(r.reshape(B * Fr, C, W)).reshape(B, Fr, a.window_sites, -1)
        return zl, zs, r

    def roll(zl, zs, k, gen):
        B = zl.shape[0]
        zt = zs.repeat(S, 1, 1, 1)                                # (S*B, F, P, cs), sample-major
        zlr = zl[..., :ck].repeat(S, 1, 1, 1)
        st, outs = zt[:, 0], []
        for j in range(k):
            seen = _with_true_halo(st, zt[:, j], h)
            st = _with_true_halo(small(seen, zlr[:, j], zlr[:, j + 1], gen=gen), zt[:, j + 1], h)
            outs.append(st)
        pred = torch.stack(outs, 1)                               # (S*B, k, P, cs)
        dec = ae_s.decode(pred.reshape(S * B * k, a.window_sites, -1)).reshape(S, B, k, C, W)
        return interior(dec, mS * p)

    def loss_on(u, k, gen):
        zl, zs, r = encode_window(u)
        dec = roll(zl, zs, k, gen)
        tgt = interior(r[:, 1:k + 1], mS * p)
        l_crps = crps_elements(dec, tgt).mean()
        l_e = energy_score(dec, tgt) if a.w_energy else dec.new_zeros(())
        l_s = spectral_loss(dec.flatten(1, 2), tgt.flatten(0, 1)) if a.w_spec else dec.new_zeros(())
        spread = dec.std(0).pow(2).mean().sqrt(); err = (dec.mean(0) - tgt).pow(2).mean().sqrt()
        return l_crps + a.w_energy * l_e + a.w_spec * l_s, {"crps": l_crps.item(), "energy": float(l_e), "spec": float(l_s),
                                                           "spread_over_err": float(spread / err)}
    drawS = lambda data, gen, k: data.windows(a.batch_s, W, k + 1, gen, 0.5)[0].to(dev)
    valS = [drawS(va, vg, a.k_max) for _ in range(3)]
    opt = torch.optim.AdamW(small.parameters(), lr=a.s_lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, a.s_steps, 200, 0.05))
    best, best_state, histS = float("inf"), None, []
    for step in range(a.s_steps):
        small.train()
        k_now = int(round(2 + (a.k_max - 2) * min(1.0, step / (0.3 * a.s_steps))))
        k = int(torch.randint(1, k_now + 1, (1,), generator=g))
        loss, st = loss_on(drawS(tr, g, k), k, g)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(small.parameters(), 1.0); opt.step(); sch.step()
        if (step + 1) % 500 == 0 or step == a.s_steps - 1:
            small.eval()
            with torch.no_grad():
                vs = [loss_on(x, a.k_max, torch.Generator().manual_seed(7 + i)) for i, x in enumerate(valS)]
            v = float(np.mean([x[0].item() for x in vs]))
            rec = {"step": step + 1, "k": k, "train_loss": loss.item(), **{f"train_{kk}": vv for kk, vv in st.items()},
                   "val_loss": v, **{f"val_{kk}": float(np.mean([x[1][kk] for x in vs])) for kk in st}}
            histS.append(rec)
            log(f"  [S] {step + 1}/{a.s_steps} k {k} loss {loss.item():.4f} | val (k={a.k_max}) {v:.4f} crps {rec['val_crps']:.4f} "
                f"energy {rec['val_energy']:.4f} spec {rec['val_spec']:.3f} spread/err {rec['val_spread_over_err']:.2f} [{time.time() - t0:.0f}s]")
            if v < best:
                best, best_state = v, {kk: t.detach().clone() for kk, t in small.state_dict().items()}
    small.load_state_dict(best_state)
    save_twoscale(out, large, ae_s, small, {"args": vars(a)})
    (out / "history.json").write_text(json.dumps({"stage_a": histA, "stage_s": histS}, indent=1))
    log(f"[ts] done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
