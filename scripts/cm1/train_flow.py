"""SM4: train the flow-matching propagator (`ks_latent.cm1.flow`) on a trained run's frozen latents, then score
it against the run's deterministic and stochastic propagators on the same test cases (held boundaries).

Data and split are the base run's (its results.json); the autoencoder is the base run's, frozen. Training
pairs are latent windows (z_n, z_{n+1}) from the training storms (storm-centred half the time, as Stage B);
the loss leaves out the windows' outer `reach_sites`. Writes flow.pt, config.json, history.json and
results.json (same layout as train_cm1.py's, so compare_runs.py reads it) to --out.
  python scripts/cm1/train_flow.py --run ~/cm1_runs/full_v1_stagec --stoch ~/cm1_runs/full_v1_stagec_crps \
      --source cache:A+B+C --out ~/cm1_runs/sm4_flow
"""
import argparse
import dataclasses
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
from ks_latent.cm1.evaluate import forecast_report  # noqa: E402
from ks_latent.cm1.flow import FlowPropagator, FlowPropConfig  # noqa: E402
from ks_latent.cm1.train import encode_members, latent_windows  # noqa: E402
from ks_latent.training.crop import warmup_cosine  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="base run: autoencoder, normalizer, split, deterministic propagator")
    ap.add_argument("--stoch", default="", help="stochastic run to compare with")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--width", type=int, default=48, help="window width in sites")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-flow", type=int, default=8)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = get_device(a.device)
    torch.manual_seed(a.seed)
    run = Path(a.run).expanduser()
    ae, prop_det, norm, cfg, split = load_run(run, dev)
    for p_ in ae.parameters():
        p_.requires_grad_(False)
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    log = lambda s: print(s, flush=True)
    data = {k: ColumnData([load_member(files[n]) for n in split[k]], norm, dev) for k in ("train", "val")}
    z = {k: encode_members(ae, d) for k, d in data.items()}
    act = {k: d.active for k, d in data.items()}
    inc = torch.cat([(zz[1:] - zz[:-1]).reshape(-1, zz.shape[-1]) for zz in z["train"]])
    scale = inc.std(dim=0).clamp(min=1e-6).tolist()
    fcfg = FlowPropConfig(latent_channels=z["train"][0].shape[-1], hidden=a.hidden, flow_hidden=a.hidden,
                          n_flow=a.n_flow, delta_scale=scale)
    flow = FlowPropagator(fcfg).to(dev)
    log(f"[SM4] flow propagator: {sum(p.numel() for p in flow.parameters()) / 1e6:.2f} M parameters, reach "
        f"{fcfg.reach_sites} sites/step, {a.n_flow} Euler steps; {len(z['train'])} training storms")
    if a.width <= 2 * fcfg.reach_sites + 4:
        raise ValueError("window too narrow for the reach")
    opt = torch.optim.AdamW(flow.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, a.steps, 500, 0.02))
    g, vg = torch.Generator().manual_seed(a.seed), torch.Generator().manual_seed(a.seed + 999)
    patch = ae.cfg.patch
    draw = lambda k, gen: latent_windows(z[k], act[k], patch, a.batch, a.width, 2, gen, 0.5).to(dev)
    val = [draw("val", vg) for _ in range(8)]
    hist, best, best_state = [], float("inf"), None
    for step in range(a.steps):
        flow.train()
        w = draw("train", g)
        loss = flow.loss(w[:, 0], w[:, 1], gen=g, margin=fcfg.reach_sites)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % max(1, a.steps // 20) == 0 or step == a.steps - 1:
            flow.eval()
            with torch.no_grad():
                v = float(np.mean([flow.loss(x[:, 0], x[:, 1], gen=torch.Generator().manual_seed(1),
                                             margin=fcfg.reach_sites).item() for x in val]))
            hist.append({"step": step + 1, "train": float(loss.item()), "val": v})
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in flow.state_dict().items()}
            log(f"  [SM4] step {step + 1}/{a.steps} loss {loss.item():.4f} val {v:.4f} [{time.time() - t_start:.0f}s]")
    flow.load_state_dict(best_state)
    flow.eval()
    torch.save({"state_dict": flow.state_dict(), "cfg": dataclasses.asdict(fcfg), "base_run": str(run)}, out / "flow.pt")
    (out / "config.json").write_text(json.dumps({"flow": dataclasses.asdict(fcfg), "base_run": str(run), "args": vars(a)}, indent=1))
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    del data, z
    # ---- evaluation on the test storms, held boundaries; same cases for every model ----
    test = ColumnData([load_member(files[n]) for n in split["test"]], norm, dev)
    zt = encode_members(ae, test)
    dt = test.dt
    n_lead, every = int(round(60 * 60 / dt)), int(round(60 * 60 / dt))
    if every + n_lead >= min(test.T):   # short runs (smoke): shrink the evaluation to fit
        n_lead = every = max(1, min(test.T) // 3)
        log(f"  WARNING: test runs have only {min(test.T)} frames; evaluating {n_lead} leads every {every} frames")
    t0s = list(range(every, min(test.T) - n_lead, every))
    rows = {}
    with torch.no_grad():
        rows["flow"] = forecast_report(ae, flow, test, zt, norm, t0s, n_lead, halo, n_samples=a.eval_samples, boundary="persistence")
        rows["deterministic"] = forecast_report(ae, prop_det, test, zt, norm, t0s, n_lead, halo, boundary="persistence")
        if a.stoch:
            ae_s, prop_s, norm_s, _, _ = load_run(Path(a.stoch).expanduser(), dev)
            rows["stochastic"] = forecast_report(ae_s, prop_s, test, encode_members(ae_s, test), norm_s, t0s, n_lead, halo,
                                                 n_samples=a.eval_samples, boundary="persistence")
    lead = [round(s / 60) for s in rows["flow"]["lead_s"]]
    idx = [lead.index(L) for L in (5, 30, 60) if L in lead] or [0, len(lead) - 1]
    print(f"\n[SM4] test storms, held boundaries: skill (nMSE/persistence) | CRPS skill | spread/error at leads "
          f"{[lead[i] for i in idx]} min")
    for k, r in rows.items():
        sk = np.array(r["nmse"]) / np.array(r["nmse_pers"])
        cr = np.array(r["crps"]) / np.array(r["crps_pers"])
        se = r.get("spread_err")
        print(f"   {k:13s} " + " ".join("%.3f" % sk[i] for i in idx) + "  |  " + " ".join("%.3f" % cr[i] for i in idx)
              + "  |  " + (" ".join("%.2f" % se[i] for i in idx) if se else "  -"))
    results = {"split": split, "dt_s": dt, "forecast": rows["flow"], "comparison": rows,
               "reconstruction": json.loads((run / "results.json").read_text())["reconstruction"]}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    write_provenance(out / "results.json", config=fcfg, seed=a.seed, device=str(dev), wall_time_s=time.time() - t_start,
                     extra={"args": vars(a)})
    print(f"[SM4] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
