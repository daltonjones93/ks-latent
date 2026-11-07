"""Curiosity run (user, 2026-11-07): KS Section 55's model, minimally adapted to CM1.

Section 55 (scripts/section55_bigger_models_lower_reg.sh) was a GLOBAL latent: a ViT autoencoder over patch tokens
(linear positional encoding, attention window 4, overlapping token window 16, d_model 108, mean pooling) to one
latent vector, a Markovian residual-MLP propagator on that vector, and Stage 1 losses recon + 0.5 x decoded
2-step prediction + 0.015 L_var + 0.01 signed spatial coherence over the latent INDEX (neighbouring indices
correlated: the index becomes a pseudo-coordinate) + 0.003 log-det barrier; Stage 2 trains the propagator alone on
frozen latents, warm-started from Stage 1's, k ramped to 12, input noise 0.1 -> 0.02.

Adaptations, and only these: the field is a whole CM1 frame, 1600 columns x 434 channels flattened site-major /
channel-minor (`KSAutoencoderViT`'s `n_channels`), so each 8-column patch token carries 8 x 434 numbers (200
tokens); the latent has d = 2800 entries (248x compression; KS: 44; user direction: 128 was far too small); mean pooling would
cap the latent's rank at d_model = 108, so the encoder uses the codebase's `token_mlp` pooling (per-token FFN to 15
numbers, flatten the 200 tokens, MLP to d) -- still one global vector; the propagator is 2048 wide (Section 55's
3.2 d would be ~0.4 B parameters); the covariance-based losses (L_var, log-det, signed coherence) are computed on a
random contiguous block of 128 indices per step (192 samples per batch cannot give a full-rank 2800 x 2800
covariance); batches of 64 windows of 3 frames; step budgets instead of epochs.
Known mismatch kept for minimality: the overlapping tokenizer wraps 4 columns around the domain edges (KS is
periodic; CM1's quiet far field makes this mostly harmless). Same split and normalizer as full_v1.

Outputs: reconstruction nMSE (all / storm columns), decoded forecast skill (nMSE / persistence) at 5 / 30 / 60 min
on the test storms (no boundary forcing: a global model has none), the latent's signed bandedness (D8) against
random-permutation nulls, and Section 55's figure: encoded truth vs latent rollout vs difference, z-scored by the
truth's own mean and std per channel, plus the latent correlation matrix.
  python scripts/cm1/sec55_cm1.py --ref-run ~/cm1_runs/full_v1 --source cache:A+B+C --out ~/cm1_runs/sec55_cm1
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from ks_latent.cm1.data import ColumnData, Normalizer, load_member, split_members, synthetic_members  # noqa: E402
from ks_latent.config import PropagatorConfig, ViTAutoencoderConfig  # noqa: E402
from ks_latent.models.autoencoder_vit import KSAutoencoderViT  # noqa: E402
from ks_latent.models.propagator import build_propagator  # noqa: E402
from ks_latent.training.losses import decorr_var_loss, logdet_barrier_loss, spatial_coherence_loss  # noqa: E402


def flat(u: torch.Tensor) -> torch.Tensor:
    """(..., C, W) variable-major columns -> (..., W * C) site-major / channel-minor."""
    return u.transpose(-1, -2).reshape(*u.shape[:-2], -1)


def lr_lambda(steps: int, warm: int = 500):
    return lambda s: (s + 1) / warm if s < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))


@torch.no_grad()
def encode_storm(ae, d: ColumnData, m: int, dev, chunk: int = 32) -> torch.Tensor:
    out = []
    for t in range(0, d.T[m], chunk):
        out.append(ae.encode(flat(d.full(m, slice(t, t + chunk), device=dev))))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smoke", "full"], default="full")
    ap.add_argument("--ref-run")
    ap.add_argument("--source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-latent", type=int, default=2800, help="2800: 694 400 / 2800 = 248x compression")
    ap.add_argument("--pool", default="token_mlp", help="'mean' (Section 55) caps the latent's rank at d_model = 108; "
                    "'token_mlp' (per-token FFN -> flatten -> MLP) gives a full-rank global latent of any size")
    ap.add_argument("--token-mlp-reduction", type=int, default=7, help="per-token compression d_model // r (108 // 7 = 15)")
    ap.add_argument("--token-mlp-hidden", type=int, default=2800)
    ap.add_argument("--prop-hidden", type=int, default=2048, help="Section 55 used ~3.2 d; 3.2 x 2800 would be ~0.4 B parameters")
    ap.add_argument("--cov-block", type=int, default=128,
                    help="covariance-based losses on a random contiguous block of this many latent indices per step "
                         "(a batch's 192 samples cannot give a full-rank 2800 x 2800 covariance)")
    ap.add_argument("--steps1", type=int, default=20000)
    ap.add_argument("--steps2", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k2", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    if a.profile == "smoke":
        mem = synthetic_members(6, T=40, NX=64, Z=4)
        norm = Normalizer.fit(mem)
        tr, va, te = split_members([m.name for m in mem], 1, 2)
        by = {m.name: m for m in mem}
        get = lambda n: by[n]
        a.steps1 = a.steps2 = 20
        a.batch, a.d_latent, a.token_mlp_hidden, a.prop_hidden, a.cov_block = 8, 40, 40, 32, 16
    else:
        from da_twin import member_files
        ref = Path(a.ref_run).expanduser()
        norm = Normalizer.from_dict(json.loads((ref / "normalizer.json").read_text()))
        sp = json.loads((ref / "results.json").read_text())["split"]
        tr, va, te = sp["train"], sp["val"], sp["test"]
        files = member_files(a.source)
        get = lambda n: load_member(files[n])
    d_tr = ColumnData([get(n) for n in tr], norm, dev)
    d_va = ColumnData([get(n) for n in va], norm, torch.device("cpu"))
    C, W, d = d_tr.C, d_tr.NX, a.d_latent
    vcfg = ViTAutoencoderConfig(NX=W * C, n_channels=C, patch_size=8, token_window=16, d_model=108 if a.profile == "full" else 16,
                                n_heads=4, n_blocks=3, attn_window=4, pos_encoding="linear", pool=a.pool, d_latent=d,
                                token_mlp_reduction=a.token_mlp_reduction, token_mlp_hidden=a.token_mlp_hidden)
    ae = KSAutoencoderViT(vcfg).to(dev)
    pcfg = PropagatorConfig(d_latent=d, hidden=a.prop_hidden, n_blocks=3, mode="markovian", backbone="mlp")
    aux = build_propagator(pcfg).to(dev)
    log(f"[S55] {len(tr)} train / {len(va)} val / {len(te)} test storms; frame {W} x {C} = {W * C} numbers -> latent {d} "
        f"({W * C / d:.0f}x); autoencoder {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M, propagator "
        f"{sum(p.numel() for p in aux.parameters()) / 1e6:.2f} M parameters")
    # ---- Stage 1: autoencoder + auxiliary propagator, Section 55's loss ----
    params = list(ae.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.steps1))
    g = torch.Generator().manual_seed(a.seed)
    vg = torch.Generator().manual_seed(a.seed + 1)
    val = [flat(d_va.windows(min(a.batch, 16), W, 3, vg)[0]) for _ in range(4)]

    def s1_losses(x):
        B = x.shape[0]
        z = ae.encode(x.reshape(B * 3, -1)).reshape(B, 3, d)
        rec = F.mse_loss(ae.decode(z.reshape(B * 3, d)), x.reshape(B * 3, -1))
        z1 = aux.step_one(z[:, 0]); z2 = aux.step_one(z1)
        pred = F.mse_loss(ae.decode(torch.cat([z1, z2])), torch.cat([x[:, 1], x[:, 2]]))
        zs = z.reshape(-1, d)
        if a.cov_block and a.cov_block < d:          # a random contiguous block of indices (keeps index neighbours)
            i0 = int(torch.randint(0, d - a.cov_block + 1, (1,), generator=g))
            zs = zs[:, i0:i0 + a.cov_block]
        _, l_var = decorr_var_loss(zs)
        return rec, pred, l_var, spatial_coherence_loss(zs, bandwidth=3.0, signed=True), logdet_barrier_loss(zs, eps=1e-3)

    hist1, best, best_state = [], float("inf"), None
    for step in range(1, a.steps1 + 1):
        ae.train(); aux.train()
        x = flat(d_tr.windows(a.batch, W, 3, g)[0]).to(dev)
        rec, pred, l_var, l_sp, l_ld = s1_losses(x)
        loss = rec + 0.5 * pred + 0.015 * l_var + 0.01 * l_sp + 0.003 * l_ld
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if step % max(1, a.steps1 // 20) == 0 or step == a.steps1:
            ae.eval(); aux.eval()
            with torch.no_grad():
                vr = float(np.mean([F.mse_loss(ae(v.reshape(-1, W * C).to(dev))[0], v.reshape(-1, W * C).to(dev)).item() for v in val]))
            hist1.append({"step": step, "rec": rec.item(), "pred": pred.item(), "l_var": l_var.item(), "l_spatial": l_sp.item(),
                          "l_logdet": l_ld.item(), "val_rec": vr})
            log(f"  [S55-1] step {step}/{a.steps1} rec {rec.item():.4f} pred {pred.item():.4f} var {l_var.item():.3f} "
                f"spatial {l_sp.item():.3f} logdet {l_ld.item():.2f} val rec {vr:.4f} [{time.time() - t_start:.0f}s]")
            if vr < best:
                best = vr
                best_state = ({k: v.detach().clone() for k, v in ae.state_dict().items()},
                              {k: v.detach().clone() for k, v in aux.state_dict().items()})
    ae.load_state_dict(best_state[0]); aux.load_state_dict(best_state[1]); ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    # ---- Stage 2: propagator alone on frozen latents, warm-started, k ramped 2 -> k2 ----
    Ztr = [encode_storm(ae, d_tr, m, dev) for m in range(len(d_tr.u))]
    del d_tr
    Zva = [encode_storm(ae, d_va, m, dev) for m in range(len(d_va.u))]
    prop = build_propagator(pcfg).to(dev)
    prop.load_state_dict(aux.state_dict())
    opt = torch.optim.AdamW(prop.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.steps2))

    def windows(Z, B, k, gen):
        m = torch.randint(0, len(Z), (B,), generator=gen)
        t = [int(torch.randint(0, Z[i].shape[0] - k, (1,), generator=gen)) for i in m.tolist()]
        return torch.stack([Z[i][s:s + k + 1] for i, s in zip(m.tolist(), t)])

    def roll(z0, k, noise=0.0, gen=None):
        z, outs = z0, []
        for _ in range(k):
            if noise:
                z = z + noise * torch.randn(z.shape, generator=gen, device="cpu").to(z.device)
            z = prop.step_one(z); outs.append(z)
        return torch.stack(outs, 1)

    hist2, best, best_state = [], float("inf"), None
    vwin = windows(Zva, 64, a.k2, torch.Generator().manual_seed(5))
    for step in range(1, a.steps2 + 1):
        frac = step / a.steps2
        k = min(a.k2, 2 + int((a.k2 - 2) * frac / 0.7))
        noise = 0.1 + (0.02 - 0.1) * frac
        prop.train()
        w = windows(Ztr, 128, k, g)
        loss = F.mse_loss(roll(w[:, 0], k, noise, g), w[:, 1:])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); opt.step(); sched.step()
        if step % max(1, a.steps2 // 20) == 0 or step == a.steps2:
            prop.eval()
            with torch.no_grad():
                v = F.mse_loss(roll(vwin[:, 0], a.k2), vwin[:, 1:]).item()
                vp = F.mse_loss(vwin[:, :1].expand(-1, a.k2, -1), vwin[:, 1:]).item()
            hist2.append({"step": step, "k": k, "loss": loss.item(), "val_k": v, "val_persistence": vp})
            log(f"  [S55-2] step {step}/{a.steps2} k {k} loss {loss.item():.4f} val (k={a.k2}) {v:.4f} persistence {vp:.4f} "
                f"[{time.time() - t_start:.0f}s]")
            if v < best:
                best, best_state = v, {kk: t.detach().clone() for kk, t in prop.state_dict().items()}
    prop.load_state_dict(best_state); prop.eval()
    torch.save({"ae_cfg": vcfg.__dict__, "ae": ae.state_dict(), "prop_cfg": pcfg.__dict__, "prop": prop.state_dict()}, out / "sec55_cm1.pt")
    # ---- evaluation on the test storms ----
    leads = [5, 30, 60] if a.profile == "full" else [1, 3]
    t0s_min = [60, 120, 180, 240] if a.profile == "full" else [5]
    acc = {L: np.zeros(4) for L in leads}
    rec_all = rec_st = n_all = n_st = 0.0
    Zte, d_keep = [], None
    for n in te:
        dt_ = ColumnData([get(n)], norm, torch.device("cpu"))
        dtm = dt_.dt / 60
        Z = encode_storm(ae, dt_, 0, dev)
        Zte.append(Z.cpu())
        with torch.no_grad():
            for t in range(0, dt_.T[0], 4):
                u = flat(dt_.full(0, t))[0]
                e = ((ae.decode(Z[t:t + 1])[0].cpu() - u) ** 2).reshape(W, C).mean(1)
                act = dt_.cref[0][t] > 10
                rec_all += float(e.sum()); n_all += W
                rec_st += float(e[act].sum()); n_st += int(act.sum())
            for t0m in t0s_min:
                t0 = int(round(t0m / dtm))
                nl = int(round(max(leads) / dtm))
                if t0 + nl >= dt_.T[0]:
                    continue
                zr = roll(Z[t0:t0 + 1], nl)[0]
                u0 = flat(dt_.full(0, t0))[0]
                for L in leads:
                    j = int(round(L / dtm))
                    tr_ = flat(dt_.full(0, t0 + j))[0]
                    fc = ae.decode(zr[j - 1:j])[0].cpu()
                    ef = ((fc - tr_) ** 2).reshape(W, C).mean(1); ep = ((u0 - tr_) ** 2).reshape(W, C).mean(1)
                    act = dt_.cref[0][t0 + j] > 10
                    acc[L] += [float(ef.sum()), float(ep.sum()), float(ef[act].sum()), float(ep[act].sum())]
        if d_keep is None:
            d_keep = (n, dt_, Z)
        log(f"[S55] {n} scored [{time.time() - t_start:.0f}s]")
    res = {"args": vars(a), "stage1": hist1, "stage2": hist2, "test": te,
           "recon_nmse_all": rec_all / n_all, "recon_nmse_storm": rec_st / max(n_st, 1),
           "skill": {str(L): acc[L][0] / acc[L][1] for L in leads},
           "skill_storm": {str(L): acc[L][2] / max(acc[L][3], 1e-12) for L in leads}}
    zall = torch.cat(Zte).numpy()
    try:
        from ks_latent.analysis.diagnostics import same_time_coupling_diagnostic_signed
        d8 = same_time_coupling_diagnostic_signed(zall[::2], n_null=200, seed=0)
        res["D8_signed_bandedness"] = float(d8.bandedness_observed)
        res["D8_p_value"] = float(d8.bandedness_p_value)
    except Exception as e:  # diagnostic only; report why it is missing
        res["D8_error"] = repr(e)
    (out / "sec55_cm1.json").write_text(json.dumps(res, indent=1))
    # ---- Section 55's figure on the first test storm ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name, dt_, Z = d_keep
    t0 = int(round((60 if a.profile == "full" else 2) / (dt_.dt / 60)))
    n_roll = min(200, Z.shape[0] - t0 - 1)
    with torch.no_grad():
        zr = roll(Z[t0:t0 + 1], n_roll)[0].cpu().numpy()
    zt = Z[t0 + 1:t0 + 1 + n_roll].cpu().numpy()
    mu, sd = zt.mean(0), zt.std(0) + 1e-8
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    tt = np.arange(1, n_roll + 1) * dt_.dt / 60
    for ax, arr, title, cmap in ((axs[0], (zt - mu) / sd, "ground truth  encode(u_true)", "RdBu_r"),
                                 (axs[1], (zr - mu) / sd, "model rollout", "RdBu_r"),
                                 (axs[2], (zr - zt) / sd, "difference", "PuOr_r")):
        im = ax.pcolormesh(np.arange(d), tt, arr, cmap=cmap, vmin=-3, vmax=3, shading="nearest", rasterized=True)
        ax.set_title(title, fontsize=10); ax.set_xlabel("latent index")
        fig.colorbar(im, ax=ax)
    axs[0].set_ylabel(f"minutes after {t0 * dt_.dt / 60:.0f} min")
    cc = np.corrcoef(zall.T)
    im = axs[3].imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    axs[3].set_title("latent correlation (test storms)", fontsize=10); axs[3].set_xlabel("latent index")
    fig.colorbar(im, ax=axs[3])
    fig.suptitle(f"Section 55 model on CM1 ({name}, unseen): latent-space rollout, z-scored by the truth's own mean/std",
                 fontsize=11)
    fig.savefig(out / "fig_sec55_cm1_hovmoller.png", dpi=150)
    sk, ss = res["skill"], res["skill_storm"]
    log(f"\n[S55] reconstruction nMSE all {res['recon_nmse_all']:.4f} / storm {res['recon_nmse_storm']:.4f} "
        f"(column model full_v1: 0.022 / 0.071)")
    log(f"[S55] forecast skill (nMSE / persistence) at {leads} min: all {' '.join('%.3f' % sk[str(L)] for L in leads)}, "
        f"storm {' '.join('%.3f' % ss[str(L)] for L in leads)} (column model full_v1_stagec: 0.213 / 0.365 / 0.344, "
        f"true boundaries)")
    if "D8_signed_bandedness" in res:
        log(f"[S55] signed latent-index bandedness {res['D8_signed_bandedness']:.3f} (p = {res['D8_p_value']:.3f} against "
            f"random index orders)")
    log(f"[S55] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
