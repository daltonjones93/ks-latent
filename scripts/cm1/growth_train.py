"""Growth-matching fine-tune of the propagator on the paired CM1 runs (pre-registration 5n).

The paired runs give, for each storm, a control trajectory and two tiny-perturbation branches (uniform 1e-3 / 1e-4 K).
Their difference D_cm1(k) (rms over all normalised channels) grows at 0.1-0.25 per minute after ~15 min; the model's
perturbation does not grow (leading Lyapunov 0.006-0.015 per min). Loss, on windows of n steps from lag k0 of a storm:
  growth  : (log[D_m(j)/D_m(0)] - log[D_cm1(k0+j)/D_cm1(k0)])^2 averaged over j = 1..n, where D_m is the rms difference
            of the DECODED rollouts of the branch state (encoded CM1 branch frame at k0) and of the control state
            (halo = the control's true latents, as in the paired comparison). Ratios, so the autoencoder floor on the
            perturbation amplitude (retention 0.05-0.2) does not enter.
  anchor  : MSE of the decoded control rollout against the CM1 control frames (the forecast must not get worse).
  prox    : w_prox * squared distance of the propagator parameters from the starting ones.
Only the propagator is trained (autoencoder frozen). Storms are split into train and held-out; the evaluation reports the
same two quantities on the held-out storms before and after, for windows at k0 = 10..40.
  python scripts/cm1/growth_train.py --run ~/cm1_runs/d34_c32_kuniform --base ~/cm1_cache_P --branches ~/cm1_cache_Q \
      --manifest ~/cm1_bundle/ens_Q/manifest.json --train memA029,memB001 --test memA030,memB002 --out ~/cm1_runs/d34_growth_f1
"""
from __future__ import annotations
import argparse, copy, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import forced_halo_rollout  # noqa: E402
from paired_compare import frame_at  # noqa: E402

AMPS = ("tiny1e-3", "tiny1e-4")


@torch.no_grad()
def enc_all(ae, u, chunk=8):
    return torch.cat([ae.encode(u[i:i + chunk]) for i in range(0, u.shape[0], chunk)])


def load_storm(short, a, ae, norm, man, n):
    info = man["members"][short]
    base = load_member(next(Path(a.base).expanduser().glob(f"*__{short}.npz")))
    br = {nm: load_member(Path(a.branches).expanduser() / f"ens_Q__{b['dir']}.npz") for nm, b in info["branches"].items()}
    T0, dt = man["t0_s"], base.dt
    dev = torch.device("cpu")
    d_base = ColumnData([base], norm, dev)
    ctrl = br["ctrl"]; dctrl = ColumnData([ctrl], norm, dev)
    u_c = torch.stack([d_base.full(0, frame_at(base, T0))[0]] + [dctrl.full(0, frame_at(ctrl, T0 + k * dt))[0] for k in range(1, n + 1)])
    S = {"name": short, "u_c": u_c, "z_c": enc_all(ae, u_c), "z_t": {}, "D": {}}
    for nm in AMPS:
        mem = br[nm]; dm = ColumnData([mem], norm, dev)
        u_t = torch.stack([u_c[0]] + [dm.full(0, frame_at(mem, T0 + k * dt))[0] for k in range(1, n + 1)])
        S["z_t"][nm] = enc_all(ae, u_t)
        S["D"][nm] = ((u_t - u_c) ** 2).mean((1, 2)).sqrt().numpy()
        del u_t
    return S


def window_losses(ae, prop, items, nroll, halo):
    """items: list of (storm, amp, k0). Returns growth loss, anchor loss, and per-window diagnostics."""
    z_c = torch.stack([S["z_c"][k0:k0 + nroll + 1] for S, _, k0 in items])
    w0 = z_c.clone()
    wp = z_c.clone()
    wp[:, 0] = torch.stack([S["z_t"][amp][k0] for S, amp, k0 in items])
    B, _, P, c = z_c.shape
    r0 = forced_halo_rollout(prop, w0, nroll, halo); rp = forced_halo_rollout(prop, wp, nroll, halo)
    z0 = torch.cat([w0[:, :1], r0], 1); zp = torch.cat([wp[:, :1], rp], 1)
    dec = lambda z: ae.decode(z.reshape(-1, P, c)).reshape(B, nroll + 1, -1, ae_nx(ae, P))
    d0, dp = dec(z0), dec(zp)
    Dm = ((dp - d0) ** 2).mean((2, 3)).sqrt()
    lm = torch.log(Dm[:, 1:] + 1e-12) - torch.log(Dm[:, :1] + 1e-12)
    lc = torch.as_tensor(np.stack([np.log(S["D"][amp][k0 + 1:k0 + nroll + 1] + 1e-12) - np.log(S["D"][amp][k0] + 1e-12)
                                   for S, amp, k0 in items]), dtype=lm.dtype)
    L_g = ((lm - lc) ** 2).mean()
    u_true = torch.stack([S["u_c"][k0 + 1:k0 + nroll + 1] for S, _, k0 in items])
    L_a = ((d0[:, 1:] - u_true) ** 2).mean()
    return L_g, L_a, {"slope_m": (lm[:, -1] / nroll).detach(), "slope_c": (lc[:, -1] / nroll)}


def ae_nx(ae, P):
    return P * ae.cfg.patch


def evaluate(ae, prop, storms, nroll, halo, k0s):
    out = {"growth_loss": [], "anchor": [], "slope_m": [], "slope_c": []}
    with torch.no_grad():
        for S in storms:
            for amp in AMPS:
                items = [(S, amp, k0) for k0 in k0s]
                Lg, La, d = window_losses(ae, prop, items, nroll, halo)
                out["growth_loss"].append(float(Lg)); out["anchor"].append(float(La))
                out["slope_m"] += d["slope_m"].tolist(); out["slope_c"] += d["slope_c"].tolist()
    return {"growth_loss": float(np.mean(out["growth_loss"])), "anchor_mse": float(np.mean(out["anchor"])),
            "median_model_log_slope_per_step": float(np.median(out["slope_m"])),
            "median_cm1_log_slope_per_step": float(np.median(out["slope_c"]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--base", required=True); ap.add_argument("--branches", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=6); ap.add_argument("--nroll", type=int, default=12)
    ap.add_argument("--w-anchor", type=float, default=1.0); ap.add_argument("--w-prox", type=float, default=0.0)
    ap.add_argument("--k0-lo", type=int, default=10); ap.add_argument("--k0-hi", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lags", type=int, default=60)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    run = Path(a.run).expanduser(); out = Path(a.out).expanduser()
    ae, prop, norm, cfg, split = load_run(run, torch.device("cpu"))
    halo = cfg["stage_b"]["halo_sites"]
    assert prop.cfg.n_hist == 1, "growth_train assumes a one-frame history"
    for q in ae.parameters():
        q.requires_grad_(False)
    man = json.loads(Path(a.manifest).expanduser().read_text())
    t0 = time.time()
    tr = [load_storm(s, a, ae, norm, man, a.lags) for s in a.train.split(",")]
    te = [load_storm(s, a, ae, norm, man, a.lags) for s in a.test.split(",")]
    print(f"[growth] loaded {len(tr)}+{len(te)} storms ({time.time() - t0:.0f}s)", flush=True)
    k0s = list(range(a.k0_lo, a.k0_hi + 1, 5))
    before = {"train": evaluate(ae, prop, tr, a.nroll, halo, k0s), "test": evaluate(ae, prop, te, a.nroll, halo, k0s)}
    print("[growth] before", json.dumps(before), flush=True)
    p0 = [q.detach().clone() for q in prop.parameters()]
    opt = torch.optim.AdamW(prop.parameters(), lr=a.lr, weight_decay=0.0)
    hist = []
    for step in range(a.steps):
        prop.train()
        items = [(tr[rng.integers(len(tr))], AMPS[rng.integers(2)], int(rng.integers(a.k0_lo, a.k0_hi + 1))) for _ in range(a.batch)]
        Lg, La, _ = window_losses(ae, prop, items, a.nroll, halo)
        prox = sum(((q - q0) ** 2).sum() for q, q0 in zip(prop.parameters(), p0))
        loss = Lg + a.w_anchor * La / max(before["train"]["anchor_mse"], 1e-9) + a.w_prox * prox
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); opt.step()
        if (step + 1) % 50 == 0 or step == a.steps - 1:
            prop.eval()
            ev = {"step": step + 1, "train_batch_growth": float(Lg), "train_batch_anchor": float(La),
                  "test": evaluate(ae, prop, te, a.nroll, halo, k0s)}
            hist.append(ev)
            print(f"[growth] {step + 1}/{a.steps} Lg {float(Lg):.3f} La {float(La):.5f} | test growth_loss "
                  f"{ev['test']['growth_loss']:.3f} anchor {ev['test']['anchor_mse']:.5f} slope_m "
                  f"{ev['test']['median_model_log_slope_per_step']:.3f} (cm1 {ev['test']['median_cm1_log_slope_per_step']:.3f}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    prop.eval()
    after = {"train": evaluate(ae, prop, tr, a.nroll, halo, k0s), "test": evaluate(ae, prop, te, a.nroll, halo, k0s)}
    out.mkdir(parents=True, exist_ok=True)
    for f in run.iterdir():
        if f.is_file() and f.name not in ("prop.pt", "prop.pt.meta.json"):
            shutil.copy2(f, out / f.name)
    ck = torch.load(run / "prop.pt", map_location="cpu"); ck["state_dict"] = prop.state_dict()
    torch.save(ck, out / "prop.pt")
    json.dump({"args": vars(a), "before": before, "after": after, "history": hist}, open(out / "growth_train.json", "w"), indent=1)
    print("[growth] after", json.dumps(after)); print("[growth] done")


if __name__ == "__main__":
    main()
