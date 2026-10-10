"""J17B B1/B3 (2026-10-10): characterize a hidden-memory truth (ks2d_memory.py) at 44^2, seed 3 (never assimilated).
Reports: u std (vs standard KS 3.75); chaos (growth of a 1e-6 perturbation over 100 units); rms(eps v) / rms KS
tendency; v autocorrelation time (1/e); how much of v a MARKOVIAN model could infer from the current u (R^2 of
a linear regression of v on local features u, lap u, lap^2 u, |grad u|^2, u^2 at the same point, fitted on the
first half and scored on the second half). The KS model's one-cycle error comes from
exp_j16_estimate_pde_model_error.py --truth-memory. Usage: j17b_scan.py eps,tau,kappa[,coupling]"""
import json, sys
import numpy as np
from scipy.fft import fft2, ifft2
from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import _nonlinear2d, get_operator2d
from ks_latent.solver.ks2d_memory import MemoryParams, integrate_memory, memory_truth
f = sys.argv[1].split(",")
mp = MemoryParams(float(f[0]), float(f[1]), float(f[2]), f[3] if len(f) > 3 else "lap")
cfg = KSConfig2D(Lx=44., Ly=44., NX=128, NY=128, dt=0.02, snapshot_every=5, spinup_time=200., seed=3)
u0, v0 = memory_truth(cfg, 3, mp)
U, V = integrate_memory(u0, v0, cfg, 10000, mp, snapshot_every=25)  # 200 units, every 0.5
U2, _ = integrate_memory(u0 + 1e-6 * np.random.default_rng(0).standard_normal(u0.shape), v0, cfg, 5000, mp,
                         snapshot_every=5000)
sep = float(np.sqrt(np.mean((U[200] - U2[-1]) ** 2)))
op = get_operator2d(cfg)
W = fft2(U[::20], axes=(-2, -1)); W[:, 0, 0] = 0
T = op.Lhat * W + np.stack([_nonlinear2d(w, op.KX, op.KY) for w in W]); T[:, 0, 0] = 0
ks_tend = float(np.sqrt(np.mean(ifft2(T, axes=(-2, -1)).real ** 2)))
ev = float(np.sqrt(np.mean((mp.eps * V) ** 2)))
Vc = V - V.mean()
ac = [float(np.mean(Vc[:-l] * Vc[l:]) / np.mean(Vc * Vc)) if l else 1.0 for l in range(0, 41)]
tau_ac = next((0.5 * l for l, c in enumerate(ac) if c < np.exp(-1)), None)
K2 = op.KX ** 2 + op.KY ** 2
def feats(u):
    w = fft2(u); ux = ifft2(1j * op.KX * w).real; uy = ifft2(1j * op.KY * w).real
    return np.stack([u, ifft2(-K2 * w).real, ifft2(K2 ** 2 * w).real, ux ** 2 + uy ** 2, u ** 2, np.ones_like(u)], -1)
idx = np.arange(1, len(U), 4)
Fa = np.concatenate([feats(U[i]).reshape(-1, 6) for i in idx]); va = np.concatenate([V[i].reshape(-1) for i in idx])
h = len(va) // 2
coef = np.linalg.lstsq(Fa[:h], va[:h], rcond=None)[0]
r2 = float(1 - np.mean((Fa[h:] @ coef - va[h:]) ** 2) / np.var(va[h:]))
out = {"mem": sys.argv[1], "u_std": float(U.std()), "sep_100u": sep, "rms_eps_v": ev, "rms_ks_tendency": ks_tend,
       "ratio": ev / ks_tend, "v_autocorr_time": tau_ac, "R2_v_from_local_u": r2}
open(f"artifacts/experiments/da2d/j17b_scan_{mp.tag()}.json", "w").write(json.dumps(out, indent=1))
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}))
