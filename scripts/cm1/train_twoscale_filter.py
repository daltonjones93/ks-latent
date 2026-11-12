"""6b: FILTER-based two-scale model with small -> large feedback (ks_latent/cm1/twoscale.py FilterTwoScaleAE / FeedbackNet).

u_L = G u (Gaussian low-pass in x, --sigma-km), u_S = u - u_L. The large scales are carried by a frozen large run (its autoencoder encodes
u_L; its law propagator, optionally fine-tuned in stage J, advances z_L); the small scales u_S by their own autoencoder and a stochastic
propagator. Stages:
  A  small-scale autoencoder on u_S (MSE + anti-collapse terms)
  S  small-scale stochastic propagator, large latent teacher-forced (fair CRPS + energy + log-spectrum on the decoded u_S samples)
  J  joint: free rollouts of the coupled model z = [z_L, z_S] with the zero-initialized feedback z_S -> z_L (one step of z_L sees z_S(t)),
     full-field fair CRPS + energy + spectrum of the decoded samples D_L(z_L) + D_S(z_S) against u, plus --w-large x MSE of the low-passed
     sample mean against u_L. Trains the small propagator and the feedback (and the large law propagator with --unfreeze-large).
Windows are drawn with a margin of 3 sigma on each side, filtered, and cropped, so the filter has no edge effect inside the window.
  python scripts/cm1/train_twoscale_filter.py --large-run ~/cm1_runs/fpc_nested_p8_dft --source cache:A+A2+B --out ~/cm1_runs/tf_v1 --device cpu
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
from ks_latent.cm1.models import ColumnAEConfig, _with_true_halo, build_autoencoder  # noqa: E402
from ks_latent.cm1.train import crps_elements, energy_score, spectral_loss  # noqa: E402
from ks_latent.cm1.twoscale import (FeedbackNet, FilterTwoScaleAE, SmallProp, SmallPropConfig, TwoScaleProp,  # noqa: E402
                                    gaussian_lowpass, save_twoscale)
from ks_latent.training.crop import warmup_cosine  # noqa: E402
from ks_latent.training.losses import decorr_var_loss, logdet_barrier_loss  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402


def interior(x, m):
    return x[..., m:x.shape[-1] - m] if m else x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--large-run", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-km", type=float, default=5.0)
    ap.add_argument("--small-channels", type=int, default=32); ap.add_argument("--small-hidden", type=int, default=256)
    ap.add_argument("--noise-channels", type=int, default=8); ap.add_argument("--cond-channels", type=int, default=8)
    ap.add_argument("--prop-hidden", type=int, default=256); ap.add_argument("--prop-layers", type=int, default=2)
    ap.add_argument("--a-steps", type=int, default=8000); ap.add_argument("--a-lr", type=float, default=1e-3)
    ap.add_argument("--s-steps", type=int, default=4000); ap.add_argument("--s-lr", type=float, default=3e-4)
    ap.add_argument("--j-steps", type=int, default=3000); ap.add_argument("--j-lr", type=float, default=1e-4)
    ap.add_argument("--k-max", type=int, default=15); ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--w-energy", type=float, default=1.0); ap.add_argument("--w-spec", type=float, default=1.0)
    ap.add_argument("--w-large", type=float, default=1.0)
    ap.add_argument("--no-feedback", action="store_true"); ap.add_argument("--unfreeze-large", action="store_true")
    ap.add_argument("--batch-a", type=int, default=32); ap.add_argument("--batch-s", type=int, default=4)
    ap.add_argument("--window-sites", type=int, default=64); ap.add_argument("--halo", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=0)
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
    log(f"[tf] large {large.name}; filter sigma {a.sigma_km} km; {len(tr_names)} training / {len(split['val'])} validation storms ({time.time() - t0:.0f}s)")
    p = ae_l.cfg.patch
    C, Zl = tr.C, tr.Z
    ae_s = build_autoencoder(ColumnAEConfig(n_channels=C, n_levels=Zl, patch=p, hidden=a.small_hidden, latent_channels=a.small_channels), None).to(dev)
    WS = a.window_sites
    W = WS * p
    pad = p * int(np.ceil(3 * a.sigma_km / 0.25 / p))
    g = torch.Generator().manual_seed(a.seed); vg = torch.Generator().manual_seed(a.seed + 99)

    def draw(data, gen, B, Fr):
        """windows (B, Fr, C, W + 2 pad) -> cropped u, u_L, u_S, each (B, Fr, C, W)"""
        u = data.windows(B, W + 2 * pad, Fr, gen, 0.5)[0].to(dev)
        ul = gaussian_lowpass(u, a.sigma_km)
        sl = slice(pad, pad + W)
        return u[..., sl], ul[..., sl], (u - ul)[..., sl]

    # ---------------- Stage A: small-scale autoencoder on u_S ----------------
    mA = 2 * ae_s.cfg.reach_sites * p
    valA = [draw(va, vg, a.batch_a, 1)[2][:, 0] for _ in range(6)]
    opt = torch.optim.AdamW(ae_s.parameters(), lr=a.a_lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, a.a_steps, 200, 0.02))
    best, best_state, histA = float("inf"), None, []
    for step in range(a.a_steps):
        ae_s.train()
        us = draw(tr, g, a.batch_a, 1)[2][:, 0]
        z = ae_s.encode(us)
        l_rec = F.mse_loss(interior(ae_s.decode(z), mA), interior(us, mA))
        zs = z[:, mA // p: z.shape[1] - mA // p].reshape(-1, z.shape[-1])
        loss = l_rec + 0.02 * decorr_var_loss(zs)[1] + 0.0035 * logdet_barrier_loss(zs, eps=1e-3)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ae_s.parameters(), 1.0); opt.step(); sch.step()
        if (step + 1) % 500 == 0 or step == a.a_steps - 1:
            ae_s.eval()
            with torch.no_grad():
                v = np.mean([F.mse_loss(interior(ae_s(x)[0], mA), interior(x, mA)).item() for x in valA])
                vr = np.mean([interior(x, mA).pow(2).mean().item() for x in valA])
            histA.append({"step": step + 1, "train": l_rec.item(), "val": float(v), "val_frac_var_explained": float(1 - v / vr)})
            log(f"  [A] {step + 1}/{a.a_steps} recon {l_rec.item():.5f} val {v:.5f} (u_S var {vr:.5f}, explained {1 - v / vr:.3f}) [{time.time() - t0:.0f}s]")
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in ae_s.state_dict().items()}
    ae_s.load_state_dict(best_state); ae_s.eval()
    for q in ae_s.parameters():
        q.requires_grad_(False)

    small = SmallProp(SmallPropConfig(small_channels=a.small_channels, cond_channels=a.cond_channels, noise_channels=a.noise_channels,
                                      hidden=a.prop_hidden, layers=a.prop_layers)).to(dev)
    h = max(a.halo, small.cfg.reach_sites)
    S, ck = a.samples, a.cond_channels
    mS = h + ae_s.cfg.reach_sites + 2 * ae_l.cfg.reach_sites          # interior margin in sites for scored fields
    ae2 = FilterTwoScaleAE(ae_l, ae_s, a.sigma_km).eval()

    @torch.no_grad()
    def encode_window(ul, us):
        B, Fr = ul.shape[:2]
        zl = ae_l.encode(ul.reshape(B * Fr, C, W)).reshape(B, Fr, WS, -1)
        zs = ae_s.encode(us.reshape(B * Fr, C, W)).reshape(B, Fr, WS, -1)
        return zl, zs

    # ---------------- Stage S: stochastic small-scale propagator, large latent teacher-forced ----------------
    def roll_s(zl, zs, k, gen):
        B = zl.shape[0]
        zt = zs.repeat(S, 1, 1, 1); zlr = zl[..., :ck].repeat(S, 1, 1, 1)
        st, outs = zt[:, 0], []
        for j in range(k):
            seen = _with_true_halo(st, zt[:, j], h)
            st = _with_true_halo(small(seen, zlr[:, j], zlr[:, j + 1], gen=gen), zt[:, j + 1], h)
            outs.append(st)
        dec = ae_s.decode(torch.stack(outs, 1).reshape(S * B * k, WS, -1)).reshape(S, B, k, C, W)
        return interior(dec, mS * p)

    def loss_s(batch, k, gen):
        u, ul, us = batch
        zl, zs = encode_window(ul, us)
        dec = roll_s(zl, zs, k, gen)
        tgt = interior(us[:, 1:k + 1], mS * p)
        l_c = crps_elements(dec, tgt).mean()
        l_e = energy_score(dec, tgt) if a.w_energy else dec.new_zeros(())
        l_p = spectral_loss(dec.flatten(1, 2), tgt.flatten(0, 1)) if a.w_spec else dec.new_zeros(())
        spread = dec.std(0).pow(2).mean().sqrt(); err = (dec.mean(0) - tgt).pow(2).mean().sqrt()
        return l_c + a.w_energy * l_e + a.w_spec * l_p, {"crps": l_c.item(), "energy": float(l_e), "spec": float(l_p), "spread_over_err": float(spread / err)}

    def run_stage(name, params, steps, lr, loss_fn, draw_fn, val_batches, k_ramp_frac=0.3, extra_model=None):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, steps, 200, 0.05))
        best, best_state, hist = float("inf"), None, []
        mods = [small] + ([extra_model] if extra_model is not None else [])
        for step in range(steps):
            small.train()
            k_now = int(round(2 + (a.k_max - 2) * min(1.0, step / (k_ramp_frac * steps))))
            k = int(torch.randint(1, k_now + 1, (1,), generator=g))
            loss, st = loss_fn(draw_fn(tr, g, k), k, g)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
            if (step + 1) % 500 == 0 or step == steps - 1:
                small.eval()
                with torch.no_grad():
                    vs = [loss_fn(x, a.k_max, torch.Generator().manual_seed(7 + i)) for i, x in enumerate(val_batches)]
                v = float(np.mean([x[0].item() for x in vs]))
                rec = {"step": step + 1, "k": k, "train_loss": loss.item(), "val_loss": v, **{f"val_{kk}": float(np.mean([x[1][kk] for x in vs])) for kk in st}}
                hist.append(rec)
                log(f"  [{name}] {step + 1}/{steps} k {k} loss {loss.item():.4f} | val (k={a.k_max}) {v:.4f} " + " ".join(f"{kk} {rec['val_' + kk]:.3f}" for kk in st) + f" [{time.time() - t0:.0f}s]")
                if v < best:
                    best = v
                    best_state = [{kk: t.detach().clone() for kk, t in m.state_dict().items()} for m in mods]
        for m, stt in zip(mods, best_state):
            m.load_state_dict(stt)
        return hist

    drawS = lambda data, gen, k: draw(data, gen, a.batch_s, k + 1)
    valS = [draw(va, vg, a.batch_s, a.k_max + 1) for _ in range(3)]
    histS = run_stage("S", list(small.parameters()), a.s_steps, a.s_lr, loss_s, drawS, valS)

    # ---------------- Stage J: joint rollouts with feedback ----------------
    fb = None if a.no_feedback else FeedbackNet(a.small_channels, ck).to(dev)
    prop2 = TwoScaleProp(prop_l, small, fb).eval()
    if a.unfreeze_large:
        for q in prop_l.parameters():
            q.requires_grad_(True)
    mL = max(mS, int(np.ceil(3 * a.sigma_km / 0.25 / p)) + 1)      # low-pass of the decoded window has an edge effect of 3 sigma

    def loss_j(batch, k, gen):
        u, ul, us = batch
        zl, zs = encode_window(ul, us)
        z = torch.cat([zl, zs], -1)                                  # (B, F, P, cl + cs) true encoded trajectory
        B = z.shape[0]
        zt = z.repeat(S, 1, 1, 1)
        st, outs = zt[:, 0], []
        for j in range(k):
            seen = _with_true_halo(st, zt[:, j], h)
            st = _with_true_halo(prop2(seen, gen=gen), zt[:, j + 1], h)
            outs.append(st)
        dec_full = ae2.decode(torch.stack(outs, 1).reshape(S * B * k, WS, -1)).reshape(S, B, k, C, W)
        dec = interior(dec_full, mS * p); tgt = interior(u[:, 1:k + 1], mS * p)
        l_c = crps_elements(dec, tgt).mean()
        l_e = energy_score(dec, tgt) if a.w_energy else dec.new_zeros(())
        l_p = spectral_loss(dec.flatten(1, 2), tgt.flatten(0, 1)) if a.w_spec else dec.new_zeros(())
        lm = interior(gaussian_lowpass(dec_full.mean(0), a.sigma_km), mL * p)
        l_L = (lm - interior(ul[:, 1:k + 1], mL * p)).pow(2).mean()
        spread = dec.std(0).pow(2).mean().sqrt(); err = (dec.mean(0) - tgt).pow(2).mean().sqrt()
        return l_c + a.w_energy * l_e + a.w_spec * l_p + a.w_large * l_L, {"crps": l_c.item(), "energy": float(l_e), "spec": float(l_p),
                                                                        "large_mse": float(l_L), "spread_over_err": float(spread / err)}
    params_j = list(small.parameters()) + ([] if fb is None else list(fb.parameters())) + (list(prop_l.parameters()) if a.unfreeze_large else [])
    valJ = [draw(va, vg, a.batch_s, a.k_max + 1) for _ in range(3)]
    extra = torch.nn.ModuleList([m for m in (fb, prop_l if a.unfreeze_large else None) if m is not None]) if (fb is not None or a.unfreeze_large) else None
    histJ = run_stage("J", params_j, a.j_steps, a.j_lr, loss_j, drawS, valJ, extra_model=extra) if a.j_steps else []

    save_twoscale(out, large, ae_s, small, {"split": "filter", "sigma_km": a.sigma_km, "args": vars(a)}, fb=fb, prop_l=prop_l if a.unfreeze_large else None)
    (out / "history.json").write_text(json.dumps({"stage_a": histA, "stage_s": histS, "stage_j": histJ}, indent=1))
    log(f"[tf] done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
