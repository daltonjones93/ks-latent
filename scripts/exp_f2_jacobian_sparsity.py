"""Task F2(a): the observation Jacobian J = d h(z)/dz, h = H o D, of the 2D KS model at its
native 44^2 deployment (per-patch decoder, 64 sites x 6 channels; observations on the regular
stride-8 network of the DA runs, 256 points), evaluated at the encoding of a spun-up state.
Computed densely by reverse mode (torch.func.jacrev) -- not by the colouring the filter uses --
so the figure shows the structure the colouring relies on rather than assuming it.
Rows are sorted by the site that owns each observation. Writes
artifacts/experiments/da2d/jacobian_sparsity_L44.npz.
"""
import numpy as np
import torch

from ks_latent.da.jacobian_coloring import obs_site_map_2d
from ks_latent.models import load_autoencoder_checkpoint_resized_2d

AE = "artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt"
NX, NS, STRIDE = 128, 8, 8
ck = torch.load(AE, map_location="cpu", weights_only=False)
ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=NS, new_n_sites_y=NS, new_NX=NX, new_NY=NX)[0].eval()
u0 = np.load("artifacts/cache/ks2d_spinup_L44_NX128_dt0.02_T200_seed0.npy")
u = torch.tensor(((u0 - ck["normalization_mean"]) / ck["normalization_std"]).reshape(1, -1), dtype=torch.float32)
with torch.no_grad():
    z = ae.encode(u)[0]


def h(zz):
    return ae.decode(zz.unsqueeze(0))[0].reshape(NX, NX)[::STRIDE, ::STRIDE].reshape(-1)


J = torch.func.jacrev(h)(z).detach().numpy()  # (m, d)
site = obs_site_map_2d(NX, NX, NS, NS, STRIDE, STRIDE).numpy()
order = np.argsort(site, kind="stable")
J = J[order]
c = z.shape[0] // (NS * NS)
off_block = np.array([[J[i, j] for j in range(J.shape[1]) if j // c != site[order][i]] for i in range(J.shape[0])])
print(f"J {J.shape}, channels {c}; max |J| off the site blocks = {np.abs(off_block).max():.3e}, "
      f"max |J| = {np.abs(J).max():.3e}")
np.savez("artifacts/experiments/da2d/jacobian_sparsity_L44.npz", J=J, site=site[order], channels=c,
         max_off_block=float(np.abs(off_block).max()), max_abs=float(np.abs(J).max()))
