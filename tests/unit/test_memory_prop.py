"""Memory-channel propagator (Z10a, ks_latent/models/memory_prop.py): reduces to KC2's local_mlp_2d with m = 0,
window training is exact against the large-domain forced reference for option (i) (and not for option (ii)), and the
map is equivariant under site rolls."""
import pytest
import torch

from ks_latent.config import AuxPropagatorConfig
from ks_latent.models.memory_prop import MemoryLocalProp2D, MemoryPropConfig, resized_memory_config
from ks_latent.models.propagator import AuxPropagator
from ks_latent.training.memory import forced_halo_rollout_memory

pytestmark = pytest.mark.unit
C, M, R = 3, 2, 2  # channels, memory channels, reach (window R, one layer)


def _mem(Px, Py, m=M, relay=False, seed=0):
    torch.manual_seed(seed)
    p = MemoryLocalProp2D(MemoryPropConfig(Px, Py, C, m, token_d_model=8, n_layers=1, window=R, mlp_ratio=2,
                                           relay=relay, zero_init=False)).double().eval()
    return p


def test_zero_memory_reproduces_local_mlp_2d_bitwise():
    Px, Py = 6, 5
    cfg = AuxPropagatorConfig(d_latent=Px * Py * C, mode="markovian", backbone="local_mlp_2d", attn_window=R,
                              local_mlp2d_n_sites_x=Px, local_mlp2d_n_sites_y=Py, local_mlp2d_local_channels=C,
                              local_mlp2d_token_d_model=8, local_mlp2d_n_layers=1, token_mlp_ratio=2, zero_init=False)
    torch.manual_seed(1)
    ref = AuxPropagator(cfg).double().eval()
    mem = _mem(Px, Py, m=0)
    ren = {"token_embed": "z_embed", "blocks": "z_blocks", "norm": "z_norm", "token_unembed": "z_unembed"}
    sd = {}
    for k, v in ref.state_dict().items():
        k = k.removeprefix("body.")
        head = k.split(".")[0]
        sd[ren[head] + k[len(head):]] = v
    mem.load_state_dict(sd, strict=True)
    z = torch.randn(4, Px * Py * C, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(ref.step_one(z), mem.step_one(z))


def _big_reference(prop, z_true, W, k, lo, hi):
    """Large periodic domain, every site's z outside [lo, hi) (x) forced to the truth before every step (all sites
    during the W warm-up steps); memory free everywhere from 0. Returns the interior predictions."""
    B, T, Px, Py, c = z_true.shape
    mem = z_true.new_zeros(B, Px, Py, prop.cfg.memory_channels)
    for t in range(W):
        _, mem = prop.step_zm(z_true[:, t], mem)
    state = z_true[:, W].clone()
    out = []
    for j in range(k):
        inp = z_true[:, W + j].clone()
        inp[:, lo:hi] = state[:, lo:hi]
        state, mem = prop.step_zm(inp, mem)
        out.append(state[:, lo:hi])
    return torch.stack(out, 1)


@pytest.mark.parametrize("relay,margin,exact", [(False, R, True), (False, 0, False), (True, R, False)])
def test_window_rollout_exactness(relay, margin, exact):
    P_big, Py, W, k, x0, Pi = 24, 4, 3, 4, 5, 2
    big = _mem(P_big, Py, relay=relay)
    halo = R + margin  # forced z: halo (reach) + margin (one more reach so halo memory sees data)
    Pu = Pi + 2 * halo
    win = MemoryLocalProp2D(resized_memory_config(big.cfg, Pu, Py)).double().eval()
    win.load_state_dict(big.state_dict())
    torch.manual_seed(3)
    z_true = torch.randn(2, W + 1 + k, P_big, Py, C, dtype=torch.float64)
    with torch.no_grad():
        ref = _big_reference(big, z_true, W, k, x0 + halo, x0 + halo + Pi)
        got = forced_halo_rollout_memory(win, z_true[:, :, x0:x0 + Pu], W, k, halo)
    err = (got - ref).abs().max().item()
    assert (err < 1e-12) == exact, err


def test_site_roll_equivariance():
    Px, Py = 7, 6
    p = _mem(Px, Py)
    x = torch.randn(2, Px * Py * (C + M), dtype=torch.float64)
    roll = lambda v: torch.roll(v.reshape(2, Px, Py, C + M), shifts=(2, -1), dims=(1, 2)).reshape(2, -1)
    with torch.no_grad():
        assert torch.allclose(p.step_one(roll(x)), roll(p.step_one(x)), atol=1e-12)


def test_extend_and_warmup_shapes():
    p = _mem(4, 3)
    z = torch.randn(2, 5, 4 * 3 * C, dtype=torch.float64)
    x = p.extend(z[:, 0])
    assert x.shape == (2, p.cfg.d_latent) and torch.equal(p.encoder_part(x), z[:, 0])
    with torch.no_grad():
        xw = p.warmup(z)
    assert torch.equal(p.encoder_part(xw), z[:, -1]) and p.split(xw)[1].abs().max() > 0
