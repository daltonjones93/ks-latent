"""x-z sections of truth, autoencoder reconstruction and forecast for chosen variables of a test member.

Microphysics in the model state: liquid condensate (cloud water + rain) and ice condensate (cloud ice +
snow + graupel), stored as sqrt(g/kg) and plotted here as mixing ratio in g/kg. Rain and snow are not
separate model variables. The forecast is from the encoded truth at --t0-min for --lead-min, true state at
the domain edges; with --average-props the decoded forecasts of several propagators (sharing the
autoencoder) are averaged, as in the best model so far.
  python scripts/cm1/plot_sections.py --run ~/cm1_runs/d34_c32_kuniform --average-props R1 R2 R3 \\
      --source cache:~/cm1_cache --member mem029 --out sections.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import StencilPropagator, StencilPropConfig, rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402

STYLE = {"qliq": ("liquid water: cloud + rain (g/kg)", "Blues", (0, 2.0), True),
         "qice": ("ice: cloud ice + snow + graupel (g/kg)", "Purples", (0, 2.0), True),
         "th": ("θ′ (K)", "RdBu_r", (-8, 8), False), "w": ("w (m/s)", "RdBu_r", (-10, 10), False),
         "qc": ("cloud water (g/kg)", "Blues", (0, 1.5), True), "qr": ("rain (g/kg)", "Blues", (0, 2.5), True),
         "qi": ("cloud ice (g/kg)", "Purples", (0, 0.5), True), "qs": ("snow (g/kg)", "Purples", (0, 1.5), True),
         "qg": ("graupel (g/kg)", "Purples", (0, 2.5), True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--average-props", nargs="*", default=[])
    ap.add_argument("--source", required=True)
    ap.add_argument("--member", default="mem029")
    ap.add_argument("--t0-min", type=float, default=60.0)
    ap.add_argument("--lead-min", type=float, default=60.0)
    ap.add_argument("--vars", nargs="+", default=["qliq", "qice"])
    ap.add_argument("--zmax-km", type=float, default=14.0)
    ap.add_argument("--n-samples", type=int, default=8, help="samples of a stochastic propagator")
    ap.add_argument("--one-sample", action="store_true",
                    help="stochastic propagator: also show a single sample (sharpness) next to the sample mean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = get_device(a.device)
    run = Path(a.run).expanduser()
    ae, prop, norm, cfg, split = load_run(run, dev)
    props = [prop]
    for d in a.average_props:
        d = Path(d).expanduser()
        p2 = StencilPropagator(StencilPropConfig(**json.loads((d / "config.json").read_text())["prop"])).to(dev)
        p2.load_state_dict(torch.load(d / "prop.pt", map_location=dev)["state_dict"]); p2.eval()
        props.append(p2)
    files = member_files(a.source)
    data = ColumnData([load_member(files[a.member])], norm, dev)
    z = encode_members(ae, data)[0]
    dt = data.dt
    t0, n = int(round(a.t0_min * 60 / dt)), int(round(a.lead_min * 60 / dt))
    halo = cfg["stage_b"]["halo_sites"]
    stoch = any(p.cfg.noise_channels for p in props)
    with torch.no_grad():
        lats = [rollout_from(p, z, t0, n, halo, a.n_samples if p.cfg.noise_channels else 1,
                             gen=torch.Generator().manual_seed(0))[-1] for p in props]          # (S, P, c) each
        fc = sum(ae.decode(l).mean(0) for l in lats) / len(props)                                 # (C, NX)
        one = ae.decode(lats[0][:1])[0] if (stoch and a.one_sample) else None
        truth = data.full(0, t0 + n, device=dev)[0]
        recon = ae(truth[None])[0][0]
    Z = data.Z
    zk = BaseState.from_sounding().z / 1000.0
    x = (np.arange(data.NX) + 0.5) * 0.25
    what = (f" (average of {len(props)} propagators)" if len(props) > 1 else
            f" (stochastic: mean of {a.n_samples} samples)" if stoch else " (single propagator)")
    rows = [(f"truth, t = {(t0 + n) * dt / 3600:.1f} h ({a.member})", truth), ("autoencoder reconstruction of the truth", recon),
            (f"forecast from t = {t0 * dt / 3600:.1f} h, lead {a.lead_min:.0f} min" + what, fc)]
    if one is not None:
        rows.append(("one sample of the stochastic forecast", one))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(len(rows), len(a.vars), figsize=(7.2 * len(a.vars), 2.35 * len(rows)), sharex=True, sharey=True,
                           constrained_layout=True, squeeze=False)
    for j, v in enumerate(a.vars):
        lab, cmap, (lo, hi), squared = STYLE[v]
        if v not in norm.vars:
            raise ValueError(f"{v} is not a variable of this model ({norm.vars})")
        sc = float(norm.scale[norm.vars.index(v)])
        sl = slice(norm.vars.index(v) * Z, (norm.vars.index(v) + 1) * Z)
        for i, (title, field) in enumerate(rows):
            F = field[sl].double().cpu().numpy() * sc
            if squared:
                F = np.clip(F, 0, None) ** 2          # sqrt(g/kg) -> g/kg
            m = ax[i, j].pcolormesh(x, zk, F, cmap=cmap, vmin=lo, vmax=hi, shading="auto", rasterized=True)
            ax[i, j].set_title(title if len(a.vars) == 1 else f"{title}" if j == 0 else title, fontsize=9)
            ax[i, j].set_ylim(0, a.zmax_km)
            if j == 0:
                ax[i, j].set_ylabel("z (km)")
        ax[-1, j].set_xlabel("x (km)")
        fig.colorbar(m, ax=ax[:, j], label=lab, shrink=0.8)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
