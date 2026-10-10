"""J18b (2026-10-10): does training on noisy data make the encoder->decoder round trip pull off-attractor
fields back toward the truth? For each model tag: true 44^2 states (independent seed-3 truth, standardized
with that model's stats), perturbed (a) in physical space by smooth zero-mean noise of std s and (b) in latent
space (decode(E(u) + s_z N)), then passed through D(E(.)). Reports rms distances to the clean truth before
and after the round trip; ratio < 1 = the round trip pulls the field back toward the attractor.
Usage: exp_j18b_roundtrip.py TAG [TAG ...]; writes artifacts/experiments/da2d/j18b_roundtrip.json (merged)."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from ks_latent.config import KSConfig2D
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.models import load_autoencoder_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d

def smooth(rng, n, NX, length=4.0):
    xi = rng.standard_normal((n, NX, NX)); f = np.fft.fftfreq(NX)
    k = np.exp(-2 * (np.pi * length) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    xi = np.fft.ifft2(np.fft.fft2(xi) * k).real; xi -= xi.mean(axis=(1, 2), keepdims=True)
    return xi / xi.std(axis=(1, 2), keepdims=True)

out_p = Path("artifacts/experiments/da2d/j18b_roundtrip.json")
res = json.loads(out_p.read_text()) if out_p.exists() else {}
for tag in sys.argv[1:]:
    AE = f"artifacts/stage1_ae_kolmogorov_crop_{tag}.pt"
    ck = torch.load(AE, map_location="cpu", weights_only=False)
    patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
    g = deployment_grid_2d(44.0, training_dx(ck["train_dataset"]), patch)
    ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=g.n_sites, new_n_sites_y=g.n_sites,
                                                new_NX=g.NX, new_NY=g.NX)[0].eval()
    mean, std = float(ck["normalization_mean"]), float(ck["normalization_std"])
    cfg = KSConfig2D(Lx=44., Ly=44., NX=g.NX, NY=g.NX, dt=0.02, snapshot_every=5, spinup_time=200., seed=3)
    u0 = np.load(f"artifacts/cache/ks2d_spinup_L44_NX{g.NX}_dt0.02_T200_seed3.npy")
    U = (integrate2d(u0, cfg, 40 * 25, snapshot_every=25)[1:] - mean) / std  # 40 states, 0.5 apart
    n = len(U)
    enc = lambda F: ae.encode(torch.tensor(F.reshape(len(F), -1), dtype=torch.float32))  # noqa: E731
    dec = lambda Z: ae.decode(Z).double().numpy().reshape(len(Z), g.NX, g.NX)  # noqa: E731
    rms = lambda A: float(np.sqrt(np.mean(A ** 2)))  # noqa: E731
    r = {}
    with torch.no_grad():
        Z = enc(U); R0 = dec(Z)
        r["clean_roundtrip_err"] = rms(R0 - U)
        rng = np.random.default_rng(0)
        for s in (0.05, 0.1, 0.2, 0.4):
            P = U + s * smooth(rng, n, g.NX)
            R = dec(enc(P))
            W = rng.standard_normal(U.shape); W -= W.mean(axis=(1, 2), keepdims=True)
            Pw = U + s * W / W.std(axis=(1, 2), keepdims=True)  # white (grid-point) noise, as the training noise
            Rw = dec(enc(Pw))
            r[f"white_s{s}"] = {"before": rms(Pw - U), "after": rms(Rw - U), "ratio": rms(Rw - U) / rms(Pw - U)}
            r[f"phys_s{s}"] = {"before": rms(P - U), "after": rms(R - U), "ratio": rms(R - U) / rms(P - U),
                               "excess_over_clean": rms(R - U) - r["clean_roundtrip_err"]}
        for sz in (0.05, 0.1, 0.2):
            gz = torch.Generator().manual_seed(1)
            P = dec(Z + sz * torch.randn(Z.shape, generator=gz))
            R = dec(enc(P))
            r[f"lat_s{sz}"] = {"before": rms(P - U), "after": rms(R - U), "ratio": rms(R - U) / rms(P - U)}
    r["field_units_std"] = std
    res[tag] = r
    print(tag, json.dumps(r))
out_p.write_text(json.dumps(res, indent=1))
