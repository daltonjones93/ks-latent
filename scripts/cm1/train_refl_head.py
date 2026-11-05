"""E1b: train the composite-reflectivity operator for a trained CM1 run and score it.

Trains `ReflectivityHead` on true columns of the run's training members (targets: CM1's own composite
reflectivity), validates on its validation member, and reports RMSE (dBZ) on the test members for
(i) true columns and (ii) decoded columns decode(encode(u)) - the input DA gives it - over all columns
and over echo columns (truth > 10 dBZ). Saves `refl_head.pt` and `refl_head.json` in the run directory.
  python scripts/cm1/train_refl_head.py --run ~/cm1_runs/d34_c32_kuniform --source cache:~/cm1_cache
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig  # noqa: E402
from ks_latent.training.crop import warmup_cosine  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def windows(data: ColumnData, B: int, W: int, gen):
    u, mem, t0, x0 = data.windows(B, W, 1, gen, active_frac=0.5)
    cref = torch.stack([data.cref[int(m)][int(t), int(x):int(x) + W] for m, t, x in zip(mem, t0, x0)])
    return u[:, 0], cref.clamp(min=0)


@torch.no_grad()
def score(head, ae, data: ColumnData, dev, decoded: bool, stride: int = 6):
    se = n = se_e = n_e = 0.0
    for m in range(len(data.u)):
        for t in range(0, data.T[m], stride):
            u = data.full(m, t, device=dev)
            if decoded:
                u = ae(u)[0]
            pred = head(u)[0].cpu()
            tru = data.cref[m][t].clamp(min=0)
            e = (pred - tru) ** 2
            se += float(e.sum()); n += e.numel()
            echo = tru > 10
            se_e += float(e[echo].sum()); n_e += int(echo.sum())
    return {"rmse_all": (se / n) ** 0.5, "rmse_echo": (se_e / max(n_e, 1)) ** 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    dev = get_device(a.device)
    torch.manual_seed(a.seed)
    run = Path(a.run).expanduser()
    ae, _, norm, cfg, split = load_run(run, dev)
    files = member_files(a.source)
    load = lambda names: ColumnData([load_member(files[n]) for n in names], norm, dev)
    tr, va, te = load(split["train"]), load(split["val"]), load(split["test"])
    head = ReflectivityHead(ReflHeadConfig(n_channels=tr.C)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, a.steps, 200, 0.02))
    g, vg = torch.Generator().manual_seed(a.seed), torch.Generator().manual_seed(a.seed + 1)
    val = [windows(va, a.batch, a.width, vg) for _ in range(8)]
    best, best_state = float("inf"), None
    for step in range(a.steps):
        u, y = windows(tr, a.batch, a.width, g)
        loss = F.mse_loss(head(u.to(dev)), y.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                v = sum(F.mse_loss(head(u.to(dev)), y.to(dev)).item() for u, y in val) / len(val)
            print(f"  step {step + 1}/{a.steps} train RMSE {loss.item() ** 0.5:.2f} dBZ  val {v ** 0.5:.2f} dBZ  [{time.time() - t0:.0f}s]", flush=True)
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    res = {"val_rmse_dbz": best ** 0.5, "test_true_columns": score(head, ae, te, dev, False),
           "test_decoded_columns": score(head, ae, te, dev, True), "steps": a.steps}
    torch.save({"state_dict": head.state_dict(), "cfg": dataclasses.asdict(head.cfg)}, run / "refl_head.pt")
    (run / "refl_head.json").write_text(json.dumps(res, indent=1))
    write_provenance(run / "refl_head.pt", config=head.cfg, seed=a.seed, device=str(dev), wall_time_s=time.time() - t0)
    print(f"[refl] test RMSE, true columns: all {res['test_true_columns']['rmse_all']:.2f}, echo {res['test_true_columns']['rmse_echo']:.2f} dBZ; "
          f"decoded columns: all {res['test_decoded_columns']['rmse_all']:.2f}, echo {res['test_decoded_columns']['rmse_echo']:.2f} dBZ")


if __name__ == "__main__":
    main()
