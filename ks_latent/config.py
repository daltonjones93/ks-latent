"""Configuration dataclasses. Every experiment script constructs its state from
these, never from bare module-level constants (ground rule 4).

Every config is a plain, hashable dataclass so it can be serialized verbatim
into a run's provenance sidecar (see `ks_latent.utils.io.write_provenance`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass


def config_hash(cfg) -> str:
    """Stable short hash of a dataclass config's contents.

    Used to key artifacts to the exact config that produced them, and to
    detect when a cached/reused artifact was built from a stale config.
    """
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class KSConfig:
    """Physical + numerical setup for the KS solver (brief §3.1).

    The handoff's originally-stated canonical configuration is L=100,
    NX=1024. **Changed to NX=256 as this codebase's working default,
    2026-08-29** (user decision, see docs/RESULTS.md "Collapse follow-up"
    and CLAUDE_CODE_BRIEF.md §5.1 addendum): Stage-1 AE training at
    NX=1024 reliably collapsed and never recovered in a 60-epoch budget,
    while NX=256 escapes the same collapse and trains ~4-5x faster: for
    L=100, NX=256 is still deeply spectrally-resolved (grid spacing
    h=L/NX~0.39 against the energy-injection lengthscale
    2*sqrt(2)*pi~8.89, i.e. ~23 points per characteristic wavelength), so
    this is a cost/speed choice, not an accuracy compromise on the solver
    side. Phase 1's Gate 1 (`test_L100_kaplan_yorke`) was originally
    validated at the literal NX=1024 and passed
    (docs/REPLICATION_LOG.md); that result stands as evidence the solver
    itself is correct at the higher resolution, independent of this
    default change. The older `docs/ML_for_KS_writeup.md` configuration
    (NX=128, stride 5 relative to a different dt) remains reproducible by
    overriding fields, and was never the default in this codebase either.
    """

    L: float = 100.0
    NX: int = 256
    dt: float = 0.05
    snapshot_every: int = 5
    spinup_time: float = 500.0
    seed: int = 0

    @property
    def dt_snap(self) -> float:
        """Physical time between stored snapshots.

        This is the single most load-bearing derived quantity in the whole
        codebase (brief §3.1): Lyapunov exponents, comoving-exponent
        velocities, and the light-cone stencil bound in Phase 2 are all
        per-physical-time quantities computed from per-snapshot data, and
        are meaningless if `dt_snap` is dropped or hardcoded elsewhere.
        Always read it from here, never recompute `dt * snapshot_every`
        inline at a call site.
        """
        return self.dt * self.snapshot_every

    def replace(self, **kwargs) -> "KSConfig":
        return dataclasses.replace(self, **kwargs)


@dataclass(frozen=True)
class Lorenz96Config:
    """Physical + numerical setup for the Lorenz-96 solver (added
    2026-09-22, user-directed: "I would like to change gears a little bit
    and try to test our findings in terms of downprojection and latent
    space dynamics on another system. in this case the Lorenz 96 system.
    can you code up a working Lorenz 96 system in 256 dimensions").

    The classic Lorenz (1996) model:

        dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F,   i = 1..N,

    indices taken mod N (a single periodic ring of N coupled ODEs, NOT a
    discretized PDE -- there is no continuum limit as N->inf the way KS's
    own NX is a grid resolution converging to a fixed physical system;
    here N is genuinely a system-SIZE parameter, more directly analogous
    to KS's domain length L than to its NX). Designed by Lorenz explicitly
    as a minimal, analyzable model of atmospheric advection: the
    quadratic term mimics nonlinear advection and conserves the
    quadratic "energy" `sum(x_i**2)` exactly on its own (see
    `nonlinear_energy_conservation` in `ks_latent/solver/lorenz96.py`'s
    module docstring for the identity and its proof) -- the SAME
    structural property (-u*u_x conserving `int(u**2)dx` up to boundary
    terms) that motivates comparing it against KS at all. `-x_i` is
    linear damping, `F` is a constant external forcing that drives the
    system away from the trivial fixed point `x_i=F` once large enough.

    `F=8.0` (this class's default) is Lorenz's own original choice and
    the overwhelmingly standard one in the literature; it produces robust
    chaos for the classic `N=40` case and, per the same
    convection/advection mechanism, remains chaotic at essentially any
    `N` comfortably larger than the instability's own local coupling
    range (empirically confirmed for this project's own `N=256` via
    `tests/unit/test_lorenz96.py::test_l96_n256_is_chaotic` -- lambda_1
    measured directly with this project's own Benettin/QR code, not
    assumed from a citation for this specific N).

    Unlike KS, there is no stiff high-order term (no `x_xxxx`-like
    operator): standard RK4 (see `ks_latent/solver/lorenz96.py`)
    integrates it accurately and efficiently with no need for an
    exponential-time-differencing scheme. Still CPU/float64-only (brief
    §1.2's device policy: solver/data-generation work is always CPU,
    float64, regardless of which specific system is being integrated)."""

    N: int = 256
    F: float = 8.0
    dt: float = 0.01
    snapshot_every: int = 10
    spinup_time: float = 50.0
    seed: int = 0

    @property
    def dt_snap(self) -> float:
        """Physical (model) time between stored snapshots -- see
        `KSConfig.dt_snap`'s own docstring for why this must always be
        read from here, never recomputed inline: every downstream
        Lyapunov/rate quantity is per-`dt_snap`, not per-solver-step."""
        return self.dt * self.snapshot_every

    def replace(self, **kwargs) -> "Lorenz96Config":
        return dataclasses.replace(self, **kwargs)


@dataclass(frozen=True)
class RayleighBenardConfig:
    """Physical + numerical setup for the 2D Boussinesq Rayleigh-Benard
    convection solver (added 2026-09-23, user-directed: build a second
    genuinely-2D chaotic testbed, following `docs/RESULTS.md`'s own
    "candidate next system" recommendation -- a bridge between L96's 1D
    periodic chain and a real convection-resolving model, forcing the
    architecture to confront a real second spatial dimension and a
    non-periodic (bounded) boundary before taking that on).

    Nondimensionalized by depth `Lz` (fixed at 1.0 -- always the unit of
    length here), thermal diffusion time `Lz**2 / kappa`, and the imposed
    top-bottom temperature difference (Chandrasekhar 1961's classic
    convention). Vorticity-streamfunction form, `theta` the temperature
    PERTURBATION from the linear conductive profile `T_bg = 1 - z`:

        (1/Pr) * (d(omega)/dt + u . grad(omega)) = laplacian(omega) + Ra * d(theta)/dx
        d(theta)/dt + u . grad(theta) = laplacian(theta) + w
        laplacian(psi) = -omega,   u = d(psi)/dz,   w = -d(psi)/dx

    **Boundary conditions: free-slip (stress-free) and isothermal at
    z=0,1** -- `psi=0`, `d^2(psi)/dz^2=0` (equivalently `omega=0`), and
    `theta=0` at both boundaries. Chosen deliberately (not the more
    common no-slip/rigid case) because these three homogeneous conditions
    are satisfied EXACTLY, term by term, by expanding every field in a
    pure sine series in z -- `f(x,z,t) = sum_n f_n(x,t) sin(n*pi*z)` --
    since `sin(n*pi*0)=sin(n*pi*1)=0` for every n, and differentiating a
    sine series twice in z stays a sine series. Combined with a Fourier
    series in the periodic x direction, the WHOLE solver reduces to
    `scipy.fft.rfft`/`irfft` (x) composed with `scipy.fft.dst`/`idst`
    type-1 (z) -- genuinely pseudospectral, no Chebyshev/implicit-solve
    machinery needed, matching this project's existing KS/L96 solvers'
    own FFT-only convention exactly.

    This choice also gives an EXACT, closed-form validation target
    (ground rule 1): free-free Rayleigh-Benard has critical Rayleigh
    number `Ra_c = 27*pi**4/4 ~= 657.5113`, attained at horizontal
    wavenumber `k_c = pi/sqrt(2)` (Chandrasekhar 1961, ch. II). `Lx` is
    set via `aspect_ratio = Lx/Lz` so the domain's FUNDAMENTAL periodic
    wavenumber (`2*pi/Lx`) lands exactly on `k_c` -- `aspect_ratio =
    2*pi/k_c = 2*sqrt(2)` -- so the textbook critical Ra is exactly
    reachable by this specific finite periodic box, not just the
    continuum limit. `tests/unit/test_rayleigh_benard.py`'s linear-
    stability test is the primary correctness gate, directly analogous to
    KS's own `test_L100_kaplan_yorke`.

    `Ra=3e4, Pr=0.7` (this class's defaults) is ~46x supercritical -- a
    genuinely nonlinear, time-dependent 2D convection regime, not a
    single steady roll.
    """

    Ra: float = 3.0e4
    Pr: float = 0.7
    Nx: int = 64
    Nz: int = 64
    Lz: float = 1.0
    aspect_ratio: float = 2.0 * 1.4142135623730951  # 2*sqrt(2), matches k_c exactly
    dt0: float = 2e-4
    cfl_target: float = 0.3
    dt_min: float = 1e-7
    dt_max: float = 2e-3
    # `cfl_check_every=1` (added/found 2026-09-23, NOT the originally
    # tried 5): at this project's target Ra=3e4, the fastest linear growth
    # rate across all resolved modes is ~93 -- checked directly -- so the
    # transition from tiny-amplitude linear growth into nonlinear
    # saturation happens over just a handful of steps, with velocities
    # growing by large factors step to step. Checking CFL only every 5
    # steps let dt stay stale (too large) through that transition and
    # caused a genuine NUMERICAL blowup (not a physical one -- confirmed
    # directly: the same run with cfl_check_every=1 saturates cleanly into
    # a bounded oscillating state, omega~500-570, u~85-100, w~110-130,
    # theta~0.5-0.6, held for thousands of steps with no sign of
    # divergence). velocities_physical() is cheap; checking every step
    # costs essentially nothing next to the pseudospectral nonlinear term.
    cfl_check_every: int = 1
    snapshot_dt: float = 0.05
    spinup_time: float = 5.0
    perturbation_amplitude: float = 1e-3
    dealias_fraction: float = 2.0 / 3.0
    seed: int = 0
    fft_workers: int = 4  # scipy.fft(..., workers=) -- moderate multi-core use by default,
    # not all cores (brief's own "Apple Silicon Mac... multi-core CPU vectorization" request,
    # balanced against not saturating the machine when other work may be running alongside it).

    @property
    def Lx(self) -> float:
        """Horizontal domain width, set so the box's fundamental periodic
        wavenumber `2*pi/Lx` exactly equals the free-free critical
        wavenumber `k_c = pi/sqrt(2)` -- see the class docstring."""
        return self.aspect_ratio * self.Lz

    @property
    def dt_snap(self) -> float:
        """Physical time between stored snapshots -- see `KSConfig.
        dt_snap`'s own docstring for why this must always be read from
        here. Here it is a direct config input (`snapshot_dt`), not
        `dt * snapshot_every`, because the timestep itself is adaptive
        (CFL-limited); `integrate` shrinks its last sub-step before each
        snapshot boundary to land on it exactly."""
        return self.snapshot_dt

    def replace(self, **kwargs) -> "RayleighBenardConfig":
        return dataclasses.replace(self, **kwargs)


@dataclass(frozen=True)
class AutoencoderConfig:
    """Stage-1 patched-transformer autoencoder architecture (brief §5.1).

    `NX=256` (not the handoff's originally-stated NX=1024) is this
    codebase's working default as of 2026-08-29 -- see `KSConfig`'s
    docstring and docs/RESULTS.md for why (NX=1024 training collapsed and
    never recovered; NX=256 escapes the same collapse and trains
    ~4-5x faster). `n_patches=NX/patch_size=32` and `n_groups=4` at this
    default, still evenly divisible with `patch_size=group_size=8`
    unchanged. `d_latent=44` still matches the handoff. Override NX/d_latent
    (and patch_size/group_size, kept as divisors) to reproduce either the
    handoff's original NX=1024 or the older ML_for_KS_writeup.md NX=128
    configuration instead.

    `n_local_layers` is *not* stated explicitly in the brief (only the
    global transformer's depth, "2 layers", is) -- defaulted to 1 and
    recorded as an inferred choice in docs/OPEN_QUESTIONS.md.
    """

    NX: int = 256
    patch_size: int = 8
    group_size: int = 8
    d_model: int = 128
    nhead: int = 4
    dim_ff: int = 128
    patch_embed_hidden: int = 256
    n_local_layers: int = 1
    n_global_layers: int = 2
    n_query_tokens: int = 8
    d_latent: int = 44
    dropout: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.NX % self.patch_size != 0:
            raise ValueError(f"NX={self.NX} must be divisible by patch_size={self.patch_size}")
        if self.n_patches % self.group_size != 0:
            raise ValueError(
                f"n_patches={self.n_patches} must be divisible by group_size={self.group_size}"
            )

    @property
    def n_patches(self) -> int:
        return self.NX // self.patch_size

    @property
    def n_groups(self) -> int:
        return self.n_patches // self.group_size


@dataclass(frozen=True)
class MLPAutoencoderConfig:
    """Plain MLP encoder/decoder ("Track A"), added 2026-08-29, ported from
    a reference implementation found at
    `/Users/daltonjones/Documents/experiments/ks_latent/models.py`
    (`MLPEncoder`/`MLPDecoder`) as an alternative to `AutoencoderConfig`'s
    patched-transformer -- see CLAUDE_CODE_BRIEF.md §5.1/5.2 "Ported
    improvements" addendum for the motivating comparison (that reference
    project's own L=94/d_z=45 run, essentially our exact operating point,
    reached val reconstruction MSE ~0.008 with this architecture against
    this codebase's ~0.81 with the patched-transformer).

    Encoder: `NX -> hidden[0] -> hidden[1] -> ... -> d_latent`, each hidden
    layer `Linear -> LayerNorm -> GELU`, linear (unconstrained) bottleneck.
    Decoder: the exact mirror, `d_latent -> reversed(hidden) -> NX`, linear
    output. `hidden=(512, 256, 128)` matches the reference's default at
    every domain size it tried (`N` in `{64, 96, 128, 288}`) -- it did not
    scale hidden widths with `N`, so this is kept unscaled here too rather
    than invented.
    """

    NX: int = 256
    hidden: tuple[int, ...] = (512, 256, 128)
    d_latent: int = 44


@dataclass(frozen=True)
class LocalFieldAutoencoderConfig:
    """Phase 10's original architecturally-local latent FIELD (CLAUDE.md
    §12.2.1/§12.3), built 2026-09-10 (Section 135), user-directed: "why
    don't we try Phase 10's original design called for an architecturally-
    enforced local field (circular-Conv1d, channel 0 anchored to a real
    physical average)" -- after `masked_mlp_expand` (Sections 128-134), a
    LOCAL propagator paired with a GLOBALLY-pooled ViT encoder, kept
    collapsing regardless of how its Jacobian spectrum was regularized.
    That arc's own §6.12 diagnosis (this project's research notes) already
    flagged the likely reason: nothing forces the flat latent index used
    by `masked_mlp_expand`'s masking to correspond to physical position at
    all -- a "receptive field of 9 index slots" may not be 9 physically
    NEARBY slots. This config instead builds locality in from the start:
    the latent is architecturally a FIELD `(n_sites, local_channels)`, not
    an arbitrary flat vector, so index adjacency IS physical adjacency by
    construction.

    Encoder: `u: (B, NX)` -> patchify (`Conv1d(1, hidden, kernel_size=
    patch_size, stride=patch_size)`, `patch_size = NX // n_sites`, an exact
    non-overlapping tiling -- no padding needed) -> `n_site_mix_layers`
    circular `Conv1d(hidden, hidden, kernel_size=2*site_mix_radius+1,
    padding=site_mix_radius, padding_mode='circular')` + GELU (mixing
    NEIGHBORING SITES, exactly translation-equivariant under any integer
    site shift, hence under any shift of `u` by a multiple of `patch_size`)
    -> `Conv1d(hidden, local_channels-1, kernel_size=1)` for the LEARNED
    residual channels. Decoder is the exact mirror, ending in
    `ConvTranspose1d(hidden, 1, kernel_size=patch_size, stride=patch_size)`
    (the shape-exact inverse of the patchify step).

    GAUGE ANCHOR (§12.3, "not optional"): channel 0 of the `(n_sites,
    local_channels)` latent is NEVER produced by the learned conv stack --
    it is fixed, exactly, to `u.reshape(B, n_sites, patch_size).mean(-1)`,
    the site's own local physical average of `u`. This gauge-fixes at
    least one channel so it is directly comparable across configs/
    resolutions, and makes the leading-order behavior of any downstream
    propagator interpretable (channel 0's own dynamics is literally the
    coarse-grained field equation). `local_channels=1` is the degenerate
    "pure coarse-graining" case (CLAUDE.md: "the latent is essentially a
    local coarse-graining of u... coarse-grained KS is NOT closed", the
    Mori-Zwanzig-memory motivation for `local_channels > 1`'s extra hidden
    channels).

    Public `encode`/`decode` use the SAME flat `(B, NX) <-> (B, d_latent)`
    interface every other AE in this codebase does (drop-in for
    `train_stage1`/`encode_dataset_with_shifts`/Stage-2 scripts) --
    `d_latent = n_sites * local_channels`, flattened SITE-MAJOR (`z_flat[:,
    site*local_channels + ch] = z[:, site, ch]`), not channel-major: this
    is what makes a flat-index masked propagator (e.g. `masked_mlp_expand`)
    see a window of nearby indices as a window of nearby SITES (all
    channels included) rather than a window confined to one channel's own
    far-flung site range -- the physically sensible locality structure,
    matching how KS's own nonlinear term couples multiple state variables
    AT THE SAME site.

    Sizing (CLAUDE.md §12.2.1's own table): `n_sites=32` ("the starting
    point"), `local_channels=3` at `NX=256` gives `patch_size=8`,
    `h=patch_size*dx=3.125` physical units at `L=100` -- matching that
    table's own `P=32` row exactly. `site_mix_radius=2`,
    `n_site_mix_layers=3` (defaults) give an encoder receptive field of
    `patch_size*(1 + n_site_mix_layers*site_mix_radius) = 8*7 = 56` grid
    points `= 21.9` physical units -- inside the brief's own target of
    "2-3 correlation lengths... ~20-30 physical units" (KS cell scale
    `2*sqrt(2)*pi ~= 8.9`), and comfortably wider than Phase 2's own
    measured light-cone bound at `dt_snap=1.0` (`v_star*dt_snap = 1.26`
    physical units, `docs/RESULTS.md`) -- the encoder needs to SEE a
    stretch of the field to represent its shape; the much smaller
    light-cone number instead bounds how wide a PROPAGATOR's own stencil
    needs to be, a separate question this config does not answer by
    itself.
    """

    NX: int = 256
    n_sites: int = 32
    local_channels: int = 3
    site_mix_radius: int = 2
    n_site_mix_layers: int = 3
    hidden: int = 32

    @property
    def patch_size(self) -> int:
        return self.NX // self.n_sites

    @property
    def d_latent(self) -> int:
        return self.n_sites * self.local_channels

    def __post_init__(self):
        if self.NX % self.n_sites != 0:
            raise ValueError(f"NX={self.NX!r} must be divisible by n_sites={self.n_sites!r}")
        if self.local_channels < 1:
            raise ValueError(f"local_channels must be >= 1, got {self.local_channels!r}")
        if self.site_mix_radius < 0:
            raise ValueError(f"site_mix_radius must be >= 0, got {self.site_mix_radius!r}")
        if self.n_site_mix_layers < 0:
            raise ValueError(f"n_site_mix_layers must be >= 0, got {self.n_site_mix_layers!r}")
        if self.hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {self.hidden!r}")


@dataclass(frozen=True)
class MaskedMLPAutoencoderConfig:
    """Locally-receptive-field MLP encoder/decoder (added 2026-08-31,
    user-directed: "try using a masked mlp for the encoder and decoder"),
    isolating whether a LOCAL receptive field baked into the encoder/decoder
    themselves (as opposed to `masked_mlp`'s use as a PROPAGATOR backbone,
    Section 11 of docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md) helps or hurts
    the recovery of chaotic latent dynamics.

    Same `NX -> hidden -> d_latent -> reversed(hidden) -> NX` structure as
    `MLPAutoencoderConfig`, but every `Linear` is a `MaskedLinearRect`
    (`ks_latent/models/autoencoder_masked_mlp.py`): a fixed 0/1 mask zeroes
    out entries whose two endpoints are farther than `mask_window` apart on
    a shared normalized circular ring (every layer's index axis, regardless
    of its width, is treated as `d_latent` evenly-spaced points around the
    KS domain's circular boundary -- so `mask_window` is in the SAME units
    as `attn_window` elsewhere in this codebase, e.g. `attn_window=4` on
    the `d_latent=44` latent ring). `mask_window=None`: fully dense (reduces
    to `KSAutoencoderMLP`'s architecture, module-for-module). No
    `LayerNorm` anywhere (see `MaskedLinearRect`'s docstring -- same
    global-coupling leak `_MaskedMLPResidualBlock` was built to avoid).
    """

    NX: int = 256
    hidden: tuple[int, ...] = (512, 256, 128)
    d_latent: int = 44
    mask_window: int | None = 4


@dataclass(frozen=True)
class FourierMLPAutoencoderConfig:
    """Hybrid Fourier+MLP encoder/decoder ("fourier_mlp" encoder kind,
    added 2026-09-03, user-directed: "I would like to try section 66
    using the fourier_mlp as the encoder, decoder and propagator" --
    direct follow-up to `PropagatorConfig`'s `backbone="fourier_mlp"`,
    which won this session's propagator-architecture comparison on
    Section 65's AE (`val_kmax_mse=0.0559`, vs. 0.0684 for `mlp`/history5
    and a dynamically-collapsed 0.2574/`D_KY=5.69` for `backbone=
    "fno_mlp"`) -- this applies the same idea (raw values PLUS a fixed,
    unlearned Fourier featurization, not replacing them) to the
    encoder/decoder role instead of just the propagator.

    Encoder: `u` (`NX` raw physical values) concatenated with the real and
    imaginary parts of `u`'s first `enc_fno_modes` `rfft` frequencies
    (`2*enc_fno_modes` numbers, periodic in PHYSICAL space -- KS's domain
    genuinely is periodic, `ks_latent/solver/ks.py`, so this featurization
    is exactly correct here, unlike the propagator's latent-index case
    where periodicity is an ordering assumption backed only by Section
    66's empirical finding) -- fed through a plain residual-MLP body
    (`ks_latent.models.propagator.MLPDeltaBody`, the same one every other
    `backbone="mlp"`/`"fourier_mlp"` propagator uses, reused here via
    cross-module import) to `d_latent` outputs, linear (unconstrained).

    Decoder: the mirror -- `z` (`d_latent` raw values) concatenated with
    the real/imaginary parts of `z`'s first `dec_fno_modes` `rfft`
    frequencies (periodic in LATENT-INDEX space, betting on Section 66's
    finding generalizing to a fresh AE trained with this architecture
    itself, not just measured post-hoc on a `vit` AE) -- fed through
    another `MLPDeltaBody` to `NX` outputs, linear (unconstrained).

    `enc_fno_modes`/`dec_fno_modes`: `None` (default) uses the full
    available spectrum (`NX//2+1`/`d_latent//2+1`) -- raw and Fourier
    features then carry the same real information in different bases,
    giving the network both representations to draw on freely. `hidden`/
    `n_blocks` size BOTH the encoder's and decoder's `MLPDeltaBody`
    identically (mirroring `PropagatorConfig`'s own `hidden`/`n_blocks`
    convention, since this reuses that exact class).

    `attn_window` (added 2026-09-03, user-directed, extending
    `PropagatorConfig.backbone="fourier_mlp"`'s own masked option to the
    encoder/decoder role -- see that field's docstring for the underlying
    idea): `None` (default) keeps the fully-dense `MLPDeltaBody` above.
    Set to a finite radius to instead split each of the encoder's and
    decoder's raw-value paths into a masked contribution (circular-band,
    same radius convention as `attn_window` elsewhere -- physical
    position and latent index treated as points on the same normalized
    ring, `ref_dim=d_latent`, exactly `ViTAutoencoderConfig.pool=
    "banded"`'s convention) SUMMED with a fully dense contribution from
    the Fourier-coefficient path (all coefficients, unrestricted -- same
    "use all the fourier coefficients" reasoning as the propagator case).
    See `ks_latent.models.autoencoder_fourier_mlp`'s masked path classes
    for the exact structure.

    `dec_use_ifft` (added 2026-09-03, user-directed: "a dense inverse
    fourier mlp for the decoder where the inverse fourier mlp applied the
    ifft not the fft" -- tried alongside `attn_window` set for the encoder
    only, i.e. asymmetric masking, since this field always forces the
    DECODER to the fully-dense structure regardless of `attn_window`
    (which then applies to the encoder only)). `False` (default): decoder
    keeps the forward-`rfft`-featurized behavior described above
    (`_fourier_features`, treating `z` as a spatial-like signal and
    extracting ITS OWN frequency content as an auxiliary feature -- an odd
    fit for a decode step, kept only for backward compatibility). `True`:
    replaces that feature with `_inverse_fourier_features`, which instead
    treats the RAW LATENT `z` itself as the non-negative-frequency half of
    a Hermitian spectrum (imaginary part zero) and applies `irfft` to
    reconstruct an `NX`-length physical-domain signal directly -- i.e. the
    decoder gets a literal "invert the latent as if it were the field's
    own low-frequency Fourier coefficients" base reconstruction, summed
    into a single dense `MLPDeltaBody` alongside raw `z` (never masked,
    unlike the `attn_window`-driven decoder path -- there is no
    circular-band structure to mask here since the ifft output already
    mixes all latent coordinates by construction). Kept additive with raw
    `z` (not replacing it) for the same safety reason `_MaskedRectPath`
    keeps a raw-value path alongside the Fourier path elsewhere in this
    class: a pure-frequency-domain-only encoder/decoder (`spectral_mlp`,
    tried and fully reverted 2026-09-03) diverged catastrophically in
    Stage 2 (`val_kmax_mse` ~170-190).
    """

    NX: int = 256
    d_latent: int = 44
    hidden: int = 128
    n_blocks: int = 3
    enc_fno_modes: int | None = None
    dec_fno_modes: int | None = None
    dropout: float = 0.0
    attn_window: int | None = None
    dec_use_ifft: bool = False
    # Added 2026-09-03, user-directed: "apply an inverse fft to the
    # frequency component output of the fourier mlp to map back to state
    # space in the encoder and propagator and decoder". Highest priority
    # of the three decoder options (overrides dec_use_ifft for the
    # decoder). False (default, unchanged behavior): encoder/decoder's
    # Fourier-features-only sub-network reads out to state space via a
    # plain unconstrained MLPDeltaBody, as before. True: that sub-network
    # instead becomes a FourierIFFTBody (ks_latent.models.propagator) --
    # an MLPDeltaBody that predicts frequency-domain coefficients,
    # explicitly inverse-transformed (irfft) back to state space, rather
    # than an arbitrary linear layer learning that mapping. (A genuinely
    # complex-valued version of FourierIFFTBody's internals -- complex
    # nn.Linear layers + a ModReLU nonlinearity -- was tried and reverted
    # same day: diverged over a realistic training horizon even after
    # adding a stabilizing complex norm; see FourierIFFTBody's own
    # docstring for the full account.) The decoder gets the SAME
    # two-network structure as the encoder
    # under this option -- a masked raw-value path (_MaskedRectPath, at
    # dec_attn_window, falling back to attn_window if dec_attn_window is
    # None) SUMMED with this FourierIFFTBody, still additive/non-
    # interacting, exactly like the encoder's masked_raw + Fourier path
    # (corrected 2026-09-03, user-directed: "I want the propagator and
    # the decoder to have the same structure as the encoder... but I want
    # the attn_window to be much larger" -- an earlier version of this
    # option wrongly dropped the decoder's/propagator's raw-value path
    # entirely under "dense" instead of just widening its window).
    fourier_ifft_readout: bool = False
    # "fourier_ifft_readout" only: the decoder's masked raw-value path
    # radius, independent of the encoder's `attn_window` (added
    # 2026-09-03, user-directed, see fourier_ifft_readout's docstring --
    # "I want the attn_window to be much larger" for the decoder/
    # propagator than the encoder's). None (default) falls back to
    # `attn_window`'s value (single-flag convenience, matching every other
    # window in this class defaulting to reuse attn_window unless
    # overridden).
    dec_attn_window: int | None = None
    # Added 2026-09-04, user-directed: "would there be a way to constrain
    # the fourier_mlp to be nonexpansive" -- ViT's attention is
    # structurally non-expansive (softmax outputs are convex combinations
    # of value vectors); this gives fourier_mlp's plain linear layers an
    # analogous guarantee. False (default, unchanged behavior). True:
    # every Linear inside the encoder's/decoder's masked raw-value path
    # (_MaskedRectPath) and Fourier-coefficient path (FourierIFFTBody) is
    # spectral-normalized (operator norm <=1), residual connections
    # become damped/averaged (0.5*(x+h), not x+h -- a plain residual is
    # NOT non-expansive even when its branch is), and rfft/irfft use
    # norm="ortho" (an isometry). The masked+Fourier SUM also becomes an
    # AVERAGE (see KSAutoencoderFourierMLP's docstring) -- summing two
    # <=1-Lipschitz branches only guarantees <=2-Lipschitz; averaging
    # keeps the combined map itself <=1-Lipschitz. See
    # ks_latent.models.propagator.ResidualMLPBlock's docstring for the
    # full mechanism and the documented output_proj/zero_init compromise
    # (the very last linear readout of each body is deliberately left
    # unconstrained, since spectral_norm divides by the weight's
    # estimated largest singular value -- exactly zero, hence NaN, for an
    # all-zero zero_init weight).
    nonexpansive: bool = False


@dataclass(frozen=True)
class ViTAutoencoderConfig:
    """ViT-style encoder/decoder ("Track B"), added 2026-08-29, ported from
    the same reference implementation's `ViTEncoder`/`ViTDecoder`/
    `CircularPositionalEncoding` (`/Users/daltonjones/Documents/
    experiments/ks_latent/models.py`) -- see CLAUDE_CODE_BRIEF.md §5.1/5.2
    "ViT-style encoder/decoder" addendum for the motivating comparison
    against this codebase's existing patched-transformer
    (`AutoencoderConfig`/`KSAutoencoderPatched`).

    The key structural difference from the patched-transformer, and the
    reason this is worth trying as its own option rather than a tweak to
    the existing one: `KSAutoencoderPatched` has **no positional encoding
    anywhere on the encoder side**. Its local transformer mixes each group
    of `group_size` consecutive patches with plain self-attention (which is
    permutation-invariant without a positional signal) and then
    mean-pools -- destroying whatever order information survived -- before
    the group tokens ever reach the global transformer, which also carries
    no positional encoding. The encoder can route information by *content*
    but has no explicit signal for *where* a feature sits in space. This
    architecture instead adds `CircularPositionalEncoding` -- a fixed,
    periodicity-respecting Fourier embedding (KS is on a periodic domain,
    so patch 0 and patch `n_tokens-1` must be geometric neighbours, which a
    plain/linear positional encoding does not know) -- to every token
    *before* any attention layer runs, on both the encoder and the
    decoder, and is a single flat attention stack over all `n_tokens`
    patches directly (no local-group pre-pooling stage).

    Encoder: patchify -> `Linear(patch_size -> d_model)` -> add circular
    positional encoding -> `n_blocks` pre-norm attention+MLP blocks -> pool
    (`"mean"` over tokens, or `"cls"` via a learned aggregation token) ->
    `Linear(d_model -> d_latent)`.
    Decoder: `Linear(d_latent -> n_tokens*d_model)` (a full per-token linear
    readout of the whole code, unlike the patched-transformer's
    broadcast-add of a single shared vector) -> add circular positional
    encoding -> `n_blocks` blocks -> `Linear(d_model -> patch_size)` per
    token, reshaped back to `NX`.

    `patch_size=8` -> `n_tokens=32` at `NX=256`: chosen by the same
    tokens-per-characteristic-structure reasoning the reference project
    used to pick its own default (`P=4` at their `N=128`/`L=94`, ~3.0
    tokens per structure) -- at this codebase's `L=100`/`NX=256`,
    `dx=100/256~0.39`, patch length `8*0.39~3.12`, and the energy-injection
    lengthscale `2*sqrt(2)*pi~8.89` gives `8.89/3.12~2.85` tokens per
    structure, matching their ratio almost exactly. Also equal to
    `AutoencoderConfig`'s own `patch_size=8` and `n_patches=32`, so the two
    architectures tokenize the field identically and differ only in what
    they do with the tokens.
    """

    NX: int = 256
    patch_size: int = 8
    d_model: int = 96
    n_heads: int = 4
    n_blocks: int = 3
    mlp_ratio: int = 4
    dropout: float = 0.0
    # "mean" (default)/"cls": global aggregation -- every z_k is a dense
    # function of every token, i.e. of the whole physical field
    # (LATENT_PDE_RESEARCH_NOTES.md §2's "global support" obstruction).
    #
    # "none" (added 2026-08-29, user-directed): skip pooling entirely. The
    # encoder applies one *shared, per-token* `Linear(d_model -> c)` (same
    # weights at every token, `c = d_latent // n_tokens`) instead of pooling
    # then a dense `Linear(d_model -> d_latent)`, giving a genuine local
    # latent field `(n_tokens, c)` (flattened to `(n_tokens*c,) = (d_latent,)`
    # only for interface compatibility with the rest of the codebase, which
    # expects a flat latent vector) instead of a globally-mixed vector. The
    # decoder mirrors this: a shared per-token `Linear(c -> d_model)`
    # replaces the dense `Linear(d_latent -> n_tokens*d_model)`. Requires
    # `d_latent % n_tokens == 0`.
    #
    # "local" (added 2026-08-29, user-directed): the compromise between the
    # two above -- mean-pool each contiguous, non-overlapping group of
    # `pool_window` tokens into one "site" (`n_sites = n_tokens //
    # pool_window` sites), then apply the same shared per-site Linear head
    # as "none" (`local_channels = d_latent // n_sites` per site). Each
    # z_k still depends only on a *bounded* (`pool_window`-token) receptive
    # field, not the whole domain, but `pool_window` gives many more
    # divisors of a target `d_latent` to work with than "none"'s rigid
    # `d_latent == n_tokens * c` (e.g. `d_latent=44` doesn't divide
    # `n_tokens=32` at all, but `pool_window=2` gives `n_sites=16`, and
    # `d_latent=32` at `c=2` works). Decoding "un-pools" by broadcasting
    # each site's vector to all `pool_window` tokens in its group
    # (`repeat_interleave`) before the decoder's windowed attention blocks
    # run. `pool="none"` is exactly `pool="local"` with `pool_window=1`.
    #
    # "local_attn" (added 2026-08-29, user-directed): same windowing as
    # "local" (`pool_window`, `n_sites`, `local_channels` all identical),
    # but the aggregation itself is a *learned, content-adaptive* local
    # attention pool instead of a fixed uniform mean -- one learned query
    # vector, shared across all sites (translation-equivariant), does
    # cross-attention over just the `pool_window` tokens in its own site
    # (the same mechanism "cls" already uses for *global* aggregation,
    # here restricted to a local window). Motivated by "local"'s measured
    # ~2x reconstruction penalty vs "mean"/"cls": this isolates whether
    # that cost came from uniform averaging discarding information, or
    # from the window size itself being the bottleneck. Decoding is
    # identical to "local" (mean-pool has no learned parameters to mirror
    # on the way back out either way, so both un-pool via
    # `repeat_interleave`).
    #
    # No CLS token is compatible with "none"/"local"/"local_attn" (there is
    # no single aggregation step for it to feed).
    pool: str = "mean"
    pool_window: int = 1
    d_latent: int = 44
    # `pool = "banded"` (added 2026-08-31, user-directed): replaces
    # "mean"'s global-average + dense Linear(d_model, d_latent) with a
    # LEARNED but circular-band-MASKED linear map straight from the
    # `n_tokens` axis to `d_latent` (reusing
    # `ks_latent.models.autoencoder_masked_mlp.MaskedLinearRect`, the same
    # rectangular circular-band mask already used for the `masked_mlp`
    # encoder, Section 17) -- each latent channel k is a learned
    # combination of only the `pool_bandwidth`-nearby tokens (in
    # `d_latent`-ring units, matching `attn_window`'s convention), not
    # every token equally the way global mean-pooling is. Decoding mirrors
    # this with its own independently-learned banded map back out to
    # `n_tokens`. Unlike `pool="local"` (fixed uniform mean over a
    # contiguous, non-overlapping token window), this is a LEARNED,
    # OVERLAPPING-band weighting -- see
    # docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 24.
    pool_bandwidth: int | None = None

    # `pool="gated"` (added 2026-09-05, user-directed: "I think 2 is the
    # best option to try first" -- soft/gated attention pooling, "A
    # Structured Self-Attentive Sentence Embedding," Lin et al. 2017):
    # replaces `pool="mean"`'s fixed uniform average over tokens with a
    # LEARNED weighted combination -- one small `Linear(d_model, 1)` gate
    # scores each token, softmax over the token axis gives the weights.
    # Strictly more expressive than "mean" (which is the special case of
    # uniform weights) at near-zero extra parameter cost (~d_model+1
    # params for the gate; the existing post-pool `enc_out`/`readout` is
    # reused unchanged). Decoding is IDENTICAL to "mean"/"cls" (the same
    # dense `Linear(d_latent, n_tokens*d_model)` expand) -- gated pooling
    # only changes the ENCODE aggregation step.
    #
    # `pool="token_mlp"` (added 2026-09-05, user-directed: "how big would
    # [flatten+dense] end up being? ... I don't want most of our
    # parameter count dominated by this mlp ... have a ffn that maps (B,
    # n_tokens, d_model) to (B, n_tokens, d_model // 8) ... and then
    # another mlp that takes the flattened vector and maps to d_latent"):
    # a parameter-BOUNDED alternative to naive flatten-then-dense pooling
    # (whose `Linear(n_tokens*d_model, d_latent)` would otherwise scale
    # with `n_tokens*d_model*d_latent` -- verified by direct computation to
    # reach ~163K-962K params at this project's typical n_tokens=32/
    # d_model=92-136 scale, comparable to or larger than an entire
    # Fourier branch in the `vit_fourier_hybrid` architecture). Two
    # stages: (1) a per-token FFN (shared weights across tokens, same
    # broadcasting `enc_out` already relies on) reduces `d_model ->
    # compressed_dim = d_model // token_mlp_reduction` BEFORE flattening,
    # keeping the flatten step's width bounded regardless of `d_model`;
    # (2) a 2-layer MLP maps the flattened `(n_tokens*compressed_dim)`-dim
    # vector to `d_latent`. Unlike "gated"/"mean"/"cls", every token's
    # full (compressed) content reaches the readout -- nothing is
    # averaged away, only dimensionality-reduced per token first.
    # Decoding mirrors this exactly in reverse (2-layer MLP expands
    # `d_latent` to the flattened space, then a per-token FFN expands
    # `compressed_dim` back to `d_model`) -- see `KSAutoencoderViT`'s
    # `enc_token_ffn`/`enc_flatten_mlp`/`dec_flatten_mlp`/`dec_token_ffn`.
    token_mlp_reduction: int = 8
    token_mlp_hidden: int = 128

    # `pool="local_token_mlp"` (added 2026-09-06, Section 100, user-directed:
    # "Let's include a token mlp in that encoder that respects the local
    # attention window"): a WINDOWED variant of `pool="token_mlp"` -- the
    # same per-token FFN (`d_model -> compressed_dim = d_model //
    # token_mlp_reduction`, shared weights across tokens) feeds a LEARNED,
    # circular-band-MASKED map straight from the `n_tokens` axis to
    # `d_latent` (`MaskedLinearRect`, `window=attn_window` measured in
    # `n_tokens`-ring units -- the SAME window value already restricting
    # this encoder's own attention, so the pooling step "respects" it
    # exactly rather than approximating it with a different bandwidth)
    # instead of `token_mlp`'s fully dense flatten-then-MLP. Each latent
    # channel k therefore depends only on the `attn_window`-nearby tokens'
    # (compressed) content, not every token in the field -- unlike
    # `token_mlp`, whose flatten step sees every token regardless of
    # `attn_window`. Structurally identical to `pool="banded"` (also a
    # `MaskedLinearRect` on the token axis + a `Linear(_, 1)` channel
    # readout) with the compressed per-token representation substituted for
    # the raw `d_model` one -- see `KSAutoencoderViT`'s `enc_local_pool`/
    # `enc_local_readout`/`dec_local_pool`/`dec_local_expand_readout`.
    # Requires `attn_window` to be set (mirrors `pool="banded"`'s
    # `pool_bandwidth` requirement -- a `window=None` "local_token_mlp"
    # would just be `token_mlp` with an extra readout step for no reason).
    # Reuses `token_mlp_reduction` (no separate field).

    # None = full (global) attention among tokens (default). A finite value
    # restricts attention to `+-attn_window` neighbours, measured according
    # to `pos_encoding` below -- added 2026-08-29, user-directed, to test
    # whether a hard locality constraint (rather than just the positional
    # encoding alone) is needed to meaningfully organize the latent's
    # structure.
    attn_window: int | None = None
    # "circular" (default): CircularPositionalEncoding + ring-distance
    # masking -- correct for this module's tokens, which sit on KS's
    # genuinely periodic domain. "linear" (added 2026-08-29, user-directed):
    # LinearPositionalEncoding (fixed, non-learned, but *not* periodic) +
    # linear-distance masking -- deliberately tests a non-periodic
    # assumption against a domain that actually is periodic, to see whether
    # the periodicity assumption specifically matters or any fixed local
    # encoding does about as well. Same ViTBlock/mlp_ratio architecture
    # either way -- only the positional encoding and its paired mask swap,
    # isolating exactly this one variable (unlike the `transformer` vs
    # `vit` backbone comparison, which differs in several respects at once).
    pos_encoding: str = "circular"
    # None (default): non-overlapping patches, unchanged behavior --
    # tokens tile the field exactly at `patch_size` spacing. Set >
    # `patch_size` (added 2026-08-29, user-directed) for OVERLAPPING patch
    # tokenization: each token's encoder *input* becomes a `token_window`-
    # wide slice of the field, circularly padded and centered on its own
    # `patch_size`-wide slot (stride stays `patch_size`, so there are still
    # exactly `n_tokens` of them). Precedent: PVTv2/T2T-ViT's overlapping
    # patch embedding, PatchTST's overlapping-stride patching for 1D time
    # series. Motivation: a hard, non-overlapping patch boundary arbitrarily
    # splits any physical feature straddling it between two independent
    # patch embeddings with no shared context; a wider input window per
    # token smooths that artifact out at the tokenization step itself,
    # before any attention runs. Only the encoder's *input* tokenization
    # changes -- the decoder's reconstruction (`dec_out` -> reshape) is
    # untouched, so no overlap-add step is needed. See
    # `circular_overlap_tokenize`'s docstring for the exact construction.
    token_window: int | None = None
    # `n_channels` (added 2026-09-23, Section 207, user-directed: "can you
    # think of a way of augmenting 201 with x' that will work with the
    # vit's assumptions" -- Sections 205/206 found concatenating x/x' into
    # one flat NX-dim vector before ViT patch-tokenization was likely the
    # real cause of that experiment's badly-converging reconstruction: the
    # encoder's positional encoding treats all n_tokens as one ring, but a
    # token built from the x' BLOCK is not "further along in space" from a
    # token built from the x block -- it is the SAME physical sites, a
    # different quantity, and nothing in the architecture told it that).
    #
    # `n_channels=1` (default, unchanged behavior): `NX` raw scalars are
    # `NX` physical positions, one channel each, exactly as before.
    #
    # `n_channels=k>1`: `NX` raw scalars are interpreted as `NX/k`
    # physical SITES of `k` channels each, laid out SITE-MAJOR/CHANNEL-
    # MINOR (`[site_0_ch_0, site_0_ch_1, ..., site_0_ch_{k-1}, site_1_ch_0,
    # ...]` -- e.g. for k=2, x/x': `[x_0, x'_0, x_1, x'_1, ...]`, exactly
    # `np.stack([x, x'], axis=-1).reshape(-1)`, NOT the block-concatenated
    # `[x_0..x_{N-1}, x'_0..x'_{N-1}]` layout Sections 204-206 used).
    # `patch_size` stays in SITE units (a token still covers `patch_size`
    # consecutive PHYSICAL sites, same physical receptive field as
    # `n_channels=1`), so `n_tokens = (NX/n_channels)/patch_size` is
    # smaller than the `n_channels=1` case at the same `NX`/`patch_size`
    # -- e.g. NX=128 (64 sites x 2 channels), patch_size=8 -> n_tokens=8,
    # matching the `n_channels=1`/N=64 case's own token count exactly,
    # unlike Section 205/206's approach (NX=32 flat, 4 tokens, only 2 of
    # which were even "x tokens"). Each token's raw vector becomes
    # `patch_size*n_channels` long (all channels of its `patch_size`
    # sites, interleaved) instead of `patch_size` -- `enc_proj`/`dec_out`
    # are sized accordingly (`KSAutoencoderViT.__init__`) -- so the SAME
    # positional encoding / attention mask (which only ever see the
    # `n_tokens` axis, now genuinely "one entry per physical location,
    # all channels together") keeps its "adjacent token = adjacent
    # physical site" meaning intact; nothing about the ring assumption
    # is violated by adding channels this way, unlike concatenation.
    # Requires `NX % n_channels == 0` and `(NX // n_channels) % patch_size
    # == 0` (validated below).
    n_channels: int = 1
    # FNO+ViT hybrid encoder/decoder (added 2026-08-30, user-directed:
    # "implement the FNO for the encoder and decoder in conjunction with
    # the ViT structure" -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md
    # Section 6). When True, `fno_n_layers` FNO spectral-conv layers
    # (`ks_latent.models.autoencoder_vit.FNOLayer`) run on the tokenized
    # sequence right after the positional encoding, BEFORE the `n_blocks`
    # ViT attention blocks, on both the encoder and decoder. Unlike the
    # propagator's `backbone="fno_vit"` (where the FFT's periodicity
    # assumption is over the *latent channel index*, unverified), here the
    # token axis is genuine physical position on KS's periodic domain --
    # the periodicity assumption is actually correct, so this is the more
    # physically justified place to try FNO. Default False (unchanged
    # behavior).
    use_fno: bool = False
    # `use_fno` only. Number of low-frequency Fourier modes each spectral-
    # conv layer keeps; None (default) keeps every rfft mode (n_tokens // 2
    # + 1, no truncation) -- see PropagatorConfig.fno_modes's docstring for
    # the same field's meaning on the propagator side.
    fno_modes: int | None = None
    # `use_fno` only. Number of stacked FNO layers before the ViT blocks.
    fno_n_layers: int = 2
    # "linear" (default): the encoder's post-pool head (and mirror
    # decoder input) is a single `Linear(d_model, d_latent)`, unchanged
    # behavior. "mlp" (added 2026-09-03, user-directed: "after the mean
    # pool, there is a linear map from 96 to 44 ... can we have the
    # option to replace that with a simple one layer mlp"): replaces it
    # with `Linear(d_model, d_latent) -> ReLU -> Linear(d_latent,
    # d_latent)`. Only meaningful for `pool in ("mean", "cls")` (the
    # `Linear(d_model, d_latent)` in question doesn't exist in that shape
    # for the other pooling modes -- `__post_init__` enforces this).
    readout: str = "linear"
    # `dec_pool` (added 2026-09-05, user-directed: "for the decoder, I
    # would also like to ... use global mean pooling for the vit" --
    # asked while `pool="local"` for the ENCODER, i.e. genuinely
    # asymmetric pooling): `None` (default) mirrors `pool` -- unchanged
    # behavior for every existing recipe. When set to a different value
    # than `pool`, `KSAutoencoderViT.decode` uses THIS mode for its own
    # expansion/un-pooling step while `encode` keeps using `pool` --
    # e.g. `pool="local"` (a hard, bounded-receptive-field encoder,
    # intended to organize/smooth the latent) paired with
    # `dec_pool="mean"` (a fully global, maximally expressive decoder,
    # so the decode path isn't ALSO bottlenecked by the same locality
    # constraint that's deliberately imposed on the encoder). See
    # `KSAutoencoderViT.__init__`'s `dec_pool_mode` / `decode`'s own
    # branching -- `dec_expand`/`dec_expand_readout`/the banded `dec_pool`
    # module are built from THIS effective mode, not unconditionally
    # from `pool`. Must be one of the same choices as `pool`, or `None`.
    dec_pool: str | None = None

    @property
    def dec_pool_mode(self) -> str:
        """The effective decoder pooling mode -- `dec_pool` if set, else
        mirrors `pool` (unchanged behavior when `dec_pool` is left at its
        default `None`)."""
        return self.dec_pool if self.dec_pool is not None else self.pool

    def __post_init__(self):
        if self.n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {self.n_channels!r}")
        if self.NX % self.n_channels != 0:
            raise ValueError(f"NX={self.NX} must be divisible by n_channels={self.n_channels}")
        if (self.NX // self.n_channels) % self.patch_size != 0:
            raise ValueError(
                f"NX/n_channels={self.NX // self.n_channels} (physical sites) must be divisible "
                f"by patch_size={self.patch_size}, got NX={self.NX}, n_channels={self.n_channels}"
            )
        if self.token_window is not None and self.token_window < self.patch_size:
            raise ValueError(
                f"token_window={self.token_window} must be >= patch_size={self.patch_size}"
            )
        if self.pool not in (
            "mean", "cls", "none", "local", "local_attn", "banded", "gated", "token_mlp", "local_token_mlp",
        ):
            raise ValueError(
                f"pool must be 'mean', 'cls', 'none', 'local', 'local_attn', 'banded', 'gated', "
                f"'token_mlp', or 'local_token_mlp', got {self.pool!r}"
            )
        if self.readout not in ("linear", "mlp"):
            raise ValueError(f"readout must be 'linear' or 'mlp', got {self.readout!r}")
        if self.readout == "mlp" and self.pool not in ("mean", "cls", "gated"):
            raise ValueError(
                f"readout='mlp' is only implemented for pool in ('mean', 'cls', 'gated'), "
                f"got pool={self.pool!r}"
            )
        if self.token_mlp_reduction < 1:
            raise ValueError(f"token_mlp_reduction must be >= 1, got {self.token_mlp_reduction!r}")
        if self.pool == "banded" and self.pool_bandwidth is None:
            raise ValueError("pool='banded' requires pool_bandwidth to be set")
        if self.pool_bandwidth is not None and self.pool != "banded":
            raise ValueError("pool_bandwidth is only used with pool='banded'")
        if self.pool == "local_token_mlp" and self.attn_window is None:
            raise ValueError(
                "pool='local_token_mlp' requires attn_window to be set (its token-axis "
                "circular-band window, in n_tokens-ring units) -- a window=None "
                "'local_token_mlp' is just 'token_mlp' with an extra readout step for no "
                "reason; use pool='token_mlp' for the fully dense flatten."
            )
        if self.pool == "none" and self.pool_window != 1:
            raise ValueError(
                f"pool='none' means pool_window=1 (no pooling); got pool_window="
                f"{self.pool_window!r}. Use pool='local'/'local_attn' for pool_window > 1."
            )
        if self.pool in ("none", "local", "local_attn"):
            if self.pool_window < 1:
                raise ValueError(f"pool_window must be >= 1, got {self.pool_window!r}")
            if self.n_tokens % self.pool_window != 0:
                raise ValueError(
                    f"pool={self.pool!r} requires n_tokens={self.n_tokens} (= NX/patch_size) "
                    f"to be divisible by pool_window={self.pool_window!r}."
                )
            if self.d_latent % self.n_sites != 0:
                raise ValueError(
                    f"pool={self.pool!r} requires d_latent={self.d_latent} to be divisible by "
                    f"n_sites={self.n_sites} (= n_tokens // pool_window), since each site gets "
                    f"an equal, fixed number of local channels c = d_latent // n_sites."
                )
        if self.dec_pool is not None:
            if self.dec_pool not in (
                "mean", "cls", "none", "local", "local_attn", "banded", "gated", "token_mlp", "local_token_mlp",
            ):
                raise ValueError(
                    f"dec_pool must be 'mean', 'cls', 'none', 'local', 'local_attn', 'banded', "
                    f"'gated', 'token_mlp', 'local_token_mlp', or None (mirror pool), got {self.dec_pool!r}"
                )
            if self.dec_pool == "local_token_mlp" and self.attn_window is None:
                raise ValueError(
                    "dec_pool='local_token_mlp' requires attn_window to be set (same reason "
                    "as pool='local_token_mlp')."
                )
            if self.dec_pool in ("none", "local", "local_attn") and self.pool not in ("none", "local", "local_attn"):
                # The pool_window/n_sites/local_channels compatibility checks above only ran
                # if `pool` itself uses one of these modes -- re-check here for the case where
                # only `dec_pool` does (dec_pool != pool).
                if self.n_tokens % self.pool_window != 0:
                    raise ValueError(
                        f"dec_pool={self.dec_pool!r} requires n_tokens={self.n_tokens} to be "
                        f"divisible by pool_window={self.pool_window!r}."
                    )
                if self.d_latent % self.n_sites != 0:
                    raise ValueError(
                        f"dec_pool={self.dec_pool!r} requires d_latent={self.d_latent} to be "
                        f"divisible by n_sites={self.n_sites} (= n_tokens // pool_window)."
                    )
            if self.dec_pool == "banded" and self.pool != "banded" and self.pool_bandwidth is None:
                raise ValueError("dec_pool='banded' requires pool_bandwidth to be set")
        if self.pos_encoding not in ("circular", "linear"):
            raise ValueError(f"pos_encoding must be 'circular' or 'linear', got {self.pos_encoding!r}")
        if self.fno_n_layers < 1:
            raise ValueError(f"fno_n_layers must be >= 1, got {self.fno_n_layers!r}")
        if self.fno_modes is not None and self.fno_modes < 1:
            raise ValueError(f"fno_modes must be >= 1, got {self.fno_modes!r}")

    @property
    def n_tokens(self) -> int:
        return (self.NX // self.n_channels) // self.patch_size

    @property
    def n_sites(self) -> int:
        """Number of pooled local sites for `pool in ("none", "local",
        "local_attn")` (`n_tokens // pool_window`; equals `n_tokens` when
        `pool_window=1`, i.e. for `pool="none"`)."""
        return self.n_tokens // self.pool_window

    @property
    def local_channels(self) -> int:
        """`c` in the `(n_sites, c)` local latent field for `pool in
        ("none", "local")` (`d_latent // n_sites`)."""
        return self.d_latent // self.n_sites


_VALID_PROPAGATOR_MODES = ("two_step", "markovian", "history")
_VALID_PROPAGATOR_BACKBONES = (
    "mlp", "transformer", "vit", "fno_vit", "fno_mlp", "fourier_mlp",
    "local_mlp", "masked_mlp", "masked_mlp_wide", "masked_mlp_expand", "node", "cnn",
    "spectral_pde", "spectral_pde_raw",
)


def _validate_propagator_mode_backbone(mode: str, backbone: str) -> None:
    if mode not in _VALID_PROPAGATOR_MODES:
        raise ValueError(f"mode must be one of {_VALID_PROPAGATOR_MODES}, got {mode!r}")
    if backbone not in _VALID_PROPAGATOR_BACKBONES:
        raise ValueError(f"backbone must be one of {_VALID_PROPAGATOR_BACKBONES}, got {backbone!r}")
    if mode == "two_step" and backbone in (
        "transformer", "vit", "fno_vit", "fno_mlp", "fourier_mlp",
        "local_mlp", "masked_mlp", "masked_mlp_wide", "masked_mlp_expand", "node", "cnn",
        "spectral_pde", "spectral_pde_raw",
    ):
        raise ValueError(
            f"backbone={backbone!r} is only implemented for mode='markovian' "
            "(brief §5.2 addendum); the two_step map stays an MLP."
        )
    if mode == "history" and backbone not in ("mlp", "vit", "fno_vit", "fourier_mlp"):
        raise ValueError(
            f"mode='history' (added 2026-08-29, user-directed: 'the original model for this "
            f"project used a propagator with information from the current step and a step in "
            f"the past... design a ViT model that takes in several states from the past'; "
            f"backbone='mlp' added 2026-08-30, generalizing 'two_step' to an arbitrary "
            f"history length -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 9) is "
            f"only implemented with backbone in ('mlp', 'vit', 'fno_vit', 'fourier_mlp'), "
            f"got backbone={backbone!r}."
        )


def _validate_spectral_field_kind(
    field_kind: str, poly_degree: int, poly_max_term_order: int | None = None,
    poly_norm_power: float = 1.0,
) -> None:
    """Shared by `backbone="spectral_pde"`/`"spectral_pde_raw"` validation
    in both `AuxPropagatorConfig` and `PropagatorConfig` (added 2026-09-09,
    see `PropagatorConfig.spectral_field_kind`'s docstring)."""
    if field_kind not in ("mlp", "polynomial", "chebyshev", "forced_burgers"):
        raise ValueError(
            f"spectral_field_kind must be 'mlp', 'polynomial', 'chebyshev', or "
            f"'forced_burgers', got {field_kind!r}"
        )
    if field_kind in ("polynomial", "chebyshev") and poly_degree not in (1, 2, 3):
        raise ValueError(f"spectral_poly_degree must be 1, 2, or 3, got {poly_degree!r}")
    if poly_max_term_order is not None and poly_max_term_order < 1:
        raise ValueError(
            f"spectral_poly_max_term_order must be >= 1 (or None, unrestricted), "
            f"got {poly_max_term_order!r} -- a value < 1 would exclude even the "
            f"constant term's combined order 0 is always kept, but any value < 1 "
            f"would also exclude every linear term (order >= 0), leaving nothing "
            f"but the constant."
        )
    if poly_norm_power < 0:
        raise ValueError(f"spectral_poly_norm_power must be >= 0, got {poly_norm_power!r}")


@dataclass(frozen=True)
class ViTFourierHybridAutoencoderConfig:
    """ViT (function-space) + Fourier-MLP (frequency-space) hybrid
    encoder/decoder (`encoder_kind="vit_fourier_hybrid"`), added
    2026-09-04, user-directed: "I don't think the spectral norm is worth
    pursuing based on what we're seeing. I want to try the following
    hybrid network for the encoder and decoder I want a Vit for function
    space, the same structure as section 52 plus I want a Fourier mlp
    that only acts on frequency space ... Then just add the output of the
    ViT and the Fourier mlp I described." See `ks_latent.models.
    autoencoder_vit_fourier_hybrid`'s module docstring and
    `ConservedFourierMLP`'s docstring for the full architecture (`vit`:
    an ordinary `ViTAutoencoderConfig`, Section 52's exact structure;
    `fourier_hidden`/`fourier_blocks`: the size of the purely-frequency-
    domain `ConservedFourierMLP` branch, summed with the ViT branch's
    output rather than replacing any part of it) and the L1-"conservation
    law" mechanism (explicitly requested as a simpler alternative to
    spectral normalization, which was tried first for a different
    architecture and abandoned after crippling reconstruction capacity --
    see `FourierMLPAutoencoderConfig.nonexpansive`'s docstring).

    `enc_fno_modes`/`dec_fno_modes`: `None` (default) uses the full
    available spectrum (`NX//2+1`/`d_latent//2+1`), same convention as
    `FourierMLPAutoencoderConfig`. These are the INPUT side (how many of
    the raw signal's own frequencies get fed as features into the Fourier
    branch) -- distinct from `enc_out_modes`/`dec_out_modes` below (the
    OUTPUT side).

    `enc_out_modes`/`dec_out_modes` (added 2026-09-05, `fourier_kind="ifft"`
    only, user-directed: "can we explicitly penalize higher frequency
    terms in the irfft matrix? or even truncate these completely? that
    way the embedding will be inherently smoother"): truncates how many
    frequency coefficients `FourierIFFTBody` is allowed to PREDICT before
    its `irfft` (`FourierIFFTBody.out_modes` -- see that class's
    docstring; `torch.fft.irfft` zero-pads any missing high-frequency
    modes automatically, so this is an exact, not approximate, low-pass
    truncation of that branch's own contribution). `None` (default) keeps
    the full spectrum (`d_latent//2+1`/`NX//2+1`), unchanged behavior.
    CAVEAT (do not oversell this as guaranteeing a smooth `z`): `encode(u)
    = vit.encode(u) + fourier_encoder(feats)` sums an UNCONSTRAINED `vit`
    branch with the (now band-limited) Fourier branch -- truncating only
    `fourier_encoder`'s own output modes does not stop the `vit` branch
    from injecting arbitrary high-frequency content into the summed `z`.
    Only meaningful with `fourier_kind="ifft"` (`ConservedFourierMLP` has
    no `irfft`/output-mode concept at all) -- raises if set otherwise.

    `fourier_kind` (added 2026-09-04, user-directed same day, Section 81:
    "can we have the same kind of hybrid vit fourier mlp, just using the
    exact fourier mlp from 75 except only using the frequency
    components"): `"conserved"` (default) = `ConservedFourierMLP`, the L1-
    "conservation law" mechanism above. `"ifft"` = `ks_latent.models.
    propagator.FourierIFFTBody` instead -- Section 75's own Fourier-path
    mechanism (an MLP predicts frequency-domain coefficients, explicitly
    inverse-transformed via `irfft` back to state space), used here with
    NO raw-value/masked path summed in (unlike Section 75's AE, which
    sums a masked raw path alongside it) -- "only using the frequency
    components." Both share the identical `forward(feats) -> output`
    interface, so `KSAutoencoderViTFourierHybrid`'s `encode`/`decode`
    need no branching of their own -- only which class gets constructed
    for `fourier_encoder`/`fourier_decoder` changes."""

    vit: ViTAutoencoderConfig
    fourier_hidden: int = 128
    fourier_blocks: int = 3
    enc_fno_modes: int | None = None
    dec_fno_modes: int | None = None
    fourier_kind: str = "conserved"
    enc_out_modes: int | None = None
    dec_out_modes: int | None = None

    def __post_init__(self):
        if self.fourier_kind not in ("conserved", "ifft"):
            raise ValueError(f"fourier_kind must be 'conserved' or 'ifft', got {self.fourier_kind!r}")
        if self.fourier_kind == "conserved" and (self.enc_out_modes is not None or self.dec_out_modes is not None):
            raise ValueError(
                "enc_out_modes/dec_out_modes require fourier_kind='ifft' -- ConservedFourierMLP "
                "has no irfft/output-mode concept to truncate."
            )

    @property
    def d_latent(self) -> int:
        """Mirrors `self.vit.d_latent` -- many call sites (`train_stage1`'s
        regularizer setup, Gate 3/4 scripts) generically read `ae_cfg.
        d_latent`/`ae_cfg.NX` regardless of encoder architecture; this
        config nests those under `.vit` instead of duplicating them at
        the top level, so this property (and `NX` below) keep that
        generic access working transparently."""
        return self.vit.d_latent

    @property
    def NX(self) -> int:
        return self.vit.NX


@dataclass(frozen=True)
class SpectralFieldAutoencoderConfig:
    """`encoder_kind="spectral_field"` (added 2026-09-06,
    docs/sine_transform_pde_plan.md, user-directed): a **global** encoder/
    decoder pair (a plain `KSAutoencoderViT`, `pool` in `("none", "local")`
    so its output is a genuine spatially-indexed scalar field `w` of length
    `N_w = vit.d_latent`, not a globally-mixed vector) composed with a
    FIXED, non-learned real-FFT transform: `z = rfft(w)[:K]`, represented as
    `2*K` real numbers (`K` real parts concatenated with `K` imaginary
    parts, same convention `ks_latent.models.propagator.FourierIFFTBody`
    already uses). `decode(z)` zero-pads back to the full `N_w//2+1`-mode
    rFFT spectrum and calls `irfft`, then the ViT decoder.

    This is the encoder/decoder half of the "spectral-latent PDE discovery"
    design (see docs/sine_transform_pde_plan.md for the full analysis and
    motivation) -- paired with `PropagatorConfig(backbone="spectral_pde")`,
    which synthesizes analytically exact spatial derivatives of `w` from
    this SAME truncated spectrum and feeds them through a shared pointwise
    MLP, so the propagator's implicit "PDE" and this class's `z` are
    defined in the same basis.

    Originally proposed as a literal sine transform; changed to rFFT
    2026-09-06 (user-confirmed) because a pure sine series does not close
    under ODD-order derivatives (needed for KS's `u*u_x` nonlinear term --
    see the plan doc §2.1) while rFFT closes under every order via a single
    `(i*k)^n` diagonal multiplier, and matches both the actual periodic
    boundary condition and `ks_latent/solver/ks.py`'s own ETDRK4
    representation.

    `L` (added here, not on `ViTAutoencoderConfig`): the physical domain
    length the wavenumbers `k = 2*pi*m/L` are computed against -- MUST match
    the `KSConfig.L` the training dataset was actually generated with, or
    every derivative synthesized downstream (both here, trivially, since
    this class itself only needs `L` for documentation/consistency-checking
    purposes -- the transform/inverse-transform pair is degree-of-freedom-
    preserving regardless of `L` -- and, critically, in
    `PropagatorConfig(backbone="spectral_pde")`'s derivative synthesis) will
    be silently scaled wrong.

    `vit.d_latent` is reused as `N_w`, the physical resolution of the
    intermediate field `w` -- a deliberate field-name overload documented
    here rather than adding a new field to `ViTAutoencoderConfig` (every
    other consumer of `ViTAutoencoderConfig` already treats `d_latent` as
    "the length of whatever `encode()` returns," which is exactly what `w`
    is from `KSAutoencoderViT`'s own point of view; this class's own
    `d_latent` property below returns `2*K`, the length `z` actually is,
    which is what the rest of the codebase reads generically)."""

    # `mlp` (added 2026-09-09, user-directed: "we've artificially
    # constrained the encoder quite a bit. Why don't we let the encoder be
    # a general mlp and see if that measurably changes things. the vit
    # might not be the right model for this" -- after Section 121 showed
    # even a mostly-exact-physics mix collapsing quickly, raising the
    # possibility that the ViT's own architectural constraints on the
    # encoder are what's preventing `z` from staying a genuinely faithful
    # physical-field representation under joint training): mutually
    # exclusive with `vit` -- EXACTLY ONE of the two must be set. `mlp`
    # uses a plain, fully-connected `KSAutoencoderMLP` (no attention, no
    # tokenization, no windowing -- every output position can depend on
    # every input position with no architectural locality bias at all) in
    # place of the ViT as the field-producing inner model; its own
    # `d_latent` is reused as `N_w` (the intermediate field's physical
    # resolution), the same field-name-overload convention `vit.d_latent`
    # already uses. No `pool`/`local_channels`/etc validation applies in
    # this branch -- `KSAutoencoderMLP.encode()` already returns a plain
    # dense `(B, d_latent)` vector, treated here as the physical field `w`
    # exactly the way `vit.encode()` with `pool="none"` is.
    vit: ViTAutoencoderConfig | None = None
    mlp: MLPAutoencoderConfig | None = None
    K: int = 1
    L: float = 100.0

    def __post_init__(self):
        if (self.vit is None) == (self.mlp is None):
            raise ValueError(
                "SpectralFieldAutoencoderConfig requires EXACTLY ONE of `vit`/`mlp` to be "
                f"set, got vit={self.vit!r}, mlp={self.mlp!r}."
            )
        if self.vit is not None:
            if self.vit.pool not in ("none", "local"):
                raise ValueError(
                    f"SpectralFieldAutoencoderConfig requires vit.pool in ('none', 'local') "
                    f"so encode() returns a genuine spatially-indexed field -- got "
                    f"vit.pool={self.vit.pool!r}. ('token_mlp'/'mean'/etc. discard spatial "
                    f"position, incompatible with treating the output as a physical field to "
                    f"spectrally transform.)"
                )
            if self.vit.dec_pool_mode not in ("none", "local"):
                raise ValueError(
                    f"SpectralFieldAutoencoderConfig requires dec_pool_mode in ('none', 'local'), "
                    f"got {self.vit.dec_pool_mode!r}"
                )
            if self.vit.local_channels != 1:
                raise ValueError(
                    f"SpectralFieldAutoencoderConfig requires vit.local_channels == 1 (a genuine "
                    f"SCALAR field w, one real number per physical site) -- got "
                    f"local_channels={self.vit.local_channels!r}. Adjust pool_window/d_latent so "
                    f"d_latent // n_sites == 1."
                )
        n_freq = self.N_w // 2 + 1
        if self.K < 1 or self.K > n_freq:
            raise ValueError(
                f"K={self.K!r} must be in [1, N_w//2+1={n_freq!r}] (N_w={self.N_w!r}, "
                f"the number of rFFT modes a length-N_w real field has)."
            )
        if self.L <= 0:
            raise ValueError(f"L must be > 0, got {self.L!r}")

    @property
    def N_w(self) -> int:
        """Physical resolution of the intermediate field `w` -- see this
        class's docstring for why this reuses `vit.d_latent`/`mlp.d_latent`."""
        return self.vit.d_latent if self.vit is not None else self.mlp.d_latent

    @property
    def d_latent(self) -> int:
        """The actual length of `z` (`2*K`: `K` real + `K` imaginary parts)
        -- NOT `vit.d_latent`/`mlp.d_latent` (see `N_w` above), so generic
        call sites that read `ae_cfg.d_latent` (regularizer sizing, Gate
        3/4 scripts) get the right number."""
        return 2 * self.K

    @property
    def NX(self) -> int:
        return self.vit.NX if self.vit is not None else self.mlp.NX


@dataclass(frozen=True)
class AuxPropagatorConfig:
    """Tiny auxiliary propagator used only inside Stage-1 training for
    L_pred, then discarded (brief §5.1: "~22k params").

    `mode` (brief §5.2 addendum, added 2026-08-29): the KS PDE is first
    order in time, so a sufficiently informative encoding of `u(x,t)` alone
    should in principle be Markovian -- `"two_step"` is the original
    `(z_{n-1}, z_n) -> z_{n+1}` design ("velocity proxy"); `"markovian"` is
    `M(z_n) -> z_{n+1}`, taking only the current state. `backbone` selects
    the architecture of `M` for `"markovian"` mode (`"mlp"`: the same
    residual-MLP pattern as `two_step`, just with a `d`-dim rather than
    `2d`-dim input; `"transformer"`: a couple of self-attention blocks with
    a *learned* positional embedding over a tokenized latent, optionally
    with a banded/local attention mask; `"vit"`, added 2026-08-29: the same
    tokenization, but with the `KSAutoencoderViT`-style block (circular,
    not learned, positional encoding; pre-norm attention + `mlp_ratio`-
    expansion MLP) and **no pooling/bottleneck step** -- see
    `ks_latent/models/propagator.py`'s `_ViTDeltaBody` for why "local"/
    "circular" here are only a proxy for locality until Phase 10's
    spatially-organized latent field exists). `two_step` is only
    implemented with the `mlp` backbone.
    """

    d_latent: int = 44
    hidden: int = 64
    n_blocks: int = 2
    dropout: float = 0.0
    zero_init: bool = True
    mode: str = "two_step"
    backbone: str = "mlp"
    # Transformer/vit-backbone-only fields (brief §5.2 addendum):
    n_tokens: int = 4
    token_d_model: int = 32
    token_nhead: int = 2
    token_n_layers: int = 2
    token_mlp_ratio: int = 4  # "vit" backbone only
    # None = full (global) attention among tokens ("transformer"/"vit"/
    # "fno_vit" backbones). "local_mlp" backbone (added 2026-08-30,
    # user-directed -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section
    # 10) reuses this SAME field as its local circular-conv kernel radius
    # instead ("local receptive field, but no softmax" -- isolates
    # receptive-field-width from the attention-softmax question) and
    # REQUIRES it to be set (there is no "global local_mlp": that is just
    # a dense/circulant mixer, already covered by "mlp"/"fno_vit").
    attn_window: int | None = None
    # "vit" backbone only (added 2026-08-29, user-directed): "circular"
    # (default) = CircularPositionalEncoding + ring-distance attn_window
    # masking. "linear" = LinearPositionalEncoding (fixed, non-learned, but
    # *not* periodic) + linear-distance masking -- same ViTBlock/mlp_ratio
    # either way, isolating exactly the periodic-vs-non-periodic assumption
    # (unlike the transformer-vs-vit backbone comparison, which differs in
    # several respects at once). See `ks_latent.models.propagator._ViTDeltaBody`.
    pos_encoding: str = "circular"
    # "vit" backbone only (added 2026-08-29, user-directed): overlapping-
    # patch tokenization, same construction as `ViTAutoencoderConfig.
    # token_window` (see its docstring) -- each token's input becomes a
    # `token_window`-wide, circularly-padded slice of `z` centered on its
    # own `chunk_size`-wide output slot, instead of an exact non-overlapping
    # tile. None (default) = unchanged (non-overlapping) behavior.
    token_window: int | None = None
    # mode="history" only; see PropagatorConfig's docstring. Carried through
    # by aux_cfg_to_propagator_cfg. `train_stage1`'s loop gathers
    # n_history-length windows and calls `.rollout_history` when
    # `aux.mode == "history"` (added 2026-08-29).
    n_history: int = 3
    # See PropagatorConfig's docstring. Carried through by
    # aux_cfg_to_propagator_cfg (added 2026-08-30, user-directed: Phase 1
    # can now train the *same, full-sized* propagator used in Phase 2,
    # architecture-identical down to this field, instead of a small
    # discarded aux -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md).
    delta_cap: float | None = None
    # `delta_cap_relative` (added 2026-09-17, Section 184, user-directed
    # after Sections 180-183 ALL converged to the identical failure
    # signature -- sustained growth to a large, unphysical scale, never
    # settling to bounded chaos at the correct scale -- regardless of
    # fixed vs. learnable A/beta, hyperviscosity, removing the forcing
    # MLP, or a regularizer guaranteeing genuine instability survives.
    # The one thing common to all four: `delta_cap` is a FIXED ABSOLUTE
    # per-step bound, not scaled to the state's own magnitude -- once
    # real instability is present, growth continues until whatever scale
    # makes that fixed bound negligible in RELATIVE terms, rather than
    # reaching a genuine dynamical balance. When `True`, `capped_delta`
    # uses `delta_cap * ||z_ref||` (the reference state's own per-sample
    # norm) as the cap instead of the bare `delta_cap` constant -- same
    # tanh-saturation shape, but the allowed step size now grows
    # proportionally with the state's own scale instead of staying fixed.
    # `False` (default): unchanged absolute-cap behavior. Only has an
    # effect when `delta_cap` is also set.
    delta_cap_relative: bool = False
    # "fno_vit" backbone only (added 2026-08-30, user-directed "Solution 2"
    # -- see PropagatorConfig's docstring and
    # docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 5). Number of
    # low-frequency Fourier modes each spectral-conv layer keeps; None
    # (default) keeps every rfft mode (n_tokens // 2 + 1, no truncation).
    fno_modes: int | None = None
    # "fno_vit" backbone only. Number of stacked FNO spectral-conv layers
    # before the token_n_layers ViT attention blocks.
    fno_n_layers: int = 2
    # "fourier_mlp" backbone only -- see PropagatorConfig.fourier_ifft_readout's
    # docstring (identical semantics, carried through to the saved full
    # propagator by aux_cfg_to_propagator_cfg).
    fourier_ifft_readout: bool = False
    # "fourier_mlp" backbone only -- see PropagatorConfig.nonexpansive's
    # docstring (identical semantics, carried through to the saved full
    # propagator by aux_cfg_to_propagator_cfg).
    nonexpansive: bool = False
    # "spectral_pde" backbone only (added 2026-09-07, user-directed: "I want
    # the encoder and decoder pair to be trained specifically to encode
    # data that can be transformed accurately by the rk4/etdrk4/euler
    # method. therefore each of those should be present in phase 1" --
    # unlike "node"/"cnn", which remain Stage-2 (full-propagator)-only,
    # this backbone's fields are mirrored here on the AUX config too, so
    # Stage 1's joint training can co-adapt the encoder/decoder with the
    # REAL propagator architecture from the start, rather than a
    # throwaway generic "mlp" aux that the encoder never actually has to
    # be compatible with -- see PropagatorConfig's own docstring for the
    # full semantics of each field, identical here.
    spectral_K: int | None = None
    spectral_N_w: int | None = None
    spectral_L: float = 100.0
    spectral_max_order: int = 4
    spectral_integrator: str = "euler"
    # "spectral_pde" ("rk4"/"etdrk4" integrators) only -- see
    # PropagatorConfig.ode_substeps's "node" docstring section for the
    # identical semantics (number of sub-steps per unit dt_snap). Mirrored
    # here (PropagatorConfig already has this field for its own
    # "node"/"cnn" use; AuxPropagatorConfig never needed it before
    # "spectral_pde" became aux-usable).
    ode_substeps: int = 1
    # "spectral_pde" only -- see PropagatorConfig.spectral_physics_prior's
    # docstring for the full semantics, identical here. Mirrored so Phase 1
    # joint training can use this baseline too, not just Phase 2.
    spectral_physics_prior: bool = False
    # "spectral_pde"/"spectral_pde_raw" only -- see PropagatorConfig.
    # spectral_field_kind's docstring for the full semantics, identical
    # here. Mirrored so Phase 1 joint training can use the polynomial form
    # too, not just Phase 2.
    spectral_field_kind: str = "mlp"
    spectral_poly_degree: int = 2
    # "spectral_pde"/"spectral_pde_raw", field_kind="polynomial" only
    # (added 2026-09-09, user-directed: "really only let the combined
    # degree of the terms be less than 5 (so w_xxx * w_xxx or
    # w_xxx*w_xxxx would have 0 coefficients since they have combined
    # degree 6, 7 respectively)"): excludes any monomial whose derivative
    # ORDERS SUM to `>= this value` from the polynomial library entirely
    # (equivalent to, but more efficient than, keeping the term with a
    # permanently-zero coefficient) -- see
    # `ks_latent.models.propagator._polynomial_term_indices`'s docstring
    # for the exact filtering rule and the physical motivation (KS's own
    # true equation has combined order <=4 on every term; a high-combined-
    # order cross term like `w_xxx*w_xxxx` has no obvious physical
    # justification). `None` (default) is unrestricted -- every monomial
    # up to `spectral_poly_degree` is kept, unchanged behavior.
    spectral_poly_max_term_order: int | None = None
    # "spectral_pde"/"spectral_pde_raw", field_kind="polynomial" only
    # (added 2026-09-09, user-directed: "less normalization for the
    # polynomial"): generalizes the per-order normalization exponent from
    # a fixed `n` to `n*spectral_poly_norm_power` -- `1.0` (default) is
    # the original strength (fixes a real NaN blowup, see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s docstring);
    # `<1.0` weakens it (high-order derivative channels keep more raw
    # dynamic range relative to low-order ones before the shared linear
    # layer sees them); `0.0` disables normalization entirely. Weakening
    # this re-introduces real blowup risk -- verify via a direct scale
    # test before trusting a value below `1.0`, not just a short smoke
    # run.
    spectral_poly_norm_power: float = 1.0
    # "spectral_pde"/"spectral_pde_raw", field_kind="polynomial" only
    # (added 2026-09-09, user-directed: "is there a way to regularize or
    # bound the eigenvalues of the differential operator induced by the
    # pde?"): forces the coefficient on the HIGHEST kept even-order linear
    # derivative term to whichever side of zero guarantees the induced
    # linear operator's eigenvalue real part goes to -inf as wavenumber ->
    # inf (bounded/well-posed) regardless of what training does elsewhere
    # -- negative for a leading order divisible by 4 (e.g. order 4,
    # w_xxxx, KS's own sign), POSITIVE for a leading order that is 2 mod 4
    # (e.g. order 2, w_xx, ordinary diffusion) -- see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody.__init__`'s
    # docstring for the full eigenvalue derivation and why the required
    # sign flips with the (ik)^n period-4 cycle. Does NOT suppress
    # instability at low/mid wavenumber (still needed for genuine chaos)
    # -- a targeted fix for runaway high-wavenumber blowup, not another
    # collapse-inducing constraint. OFF by default (False, unchanged
    # behavior).
    spectral_poly_stable_leading: bool = False
    # `spectral_poly_no_constant` (added 2026-09-11, user-directed: "we
    # should just force the constant to be 0 during training"): removes
    # BOTH additive-constant avenues -- the Linear layer's own `bias`
    # (bias=False, no parameter at all) and the `()` term's own weight
    # column (architecturally zeroed every forward pass, so it never
    # contributes and never receives gradient -- same mechanism
    # `spectral_poly_stable_leading`'s own reparametrized column already
    # relies on). Motivated by a direct finding: a closure with
    # `pde_coeff_l1_linear_only` already zeroing every linear term still
    # produced a near-static standalone rollout (its own self-spectrum
    # barely moved over 55 steps), with the constant term absorbing 61%
    # of the coefficient mass -- suspected of anchoring the dynamics near
    # a near-invariant configuration. OFF by default (False, unchanged
    # behavior).
    spectral_poly_no_constant: bool = False
    # `spectral_poly_fixed_linear_terms` (added 2026-09-11, user-directed:
    # "assume the pde always had -w_xx-w_xxxx, we'll just learn the rest
    # of the terms around this"): maps derivative order `n` (for the
    # single-index term `(n,)`, e.g. `{2: -1.0, 4: -1.0}` for w_xx/w_xxxx)
    # to a FIXED, non-learned PHYSICAL coefficient value -- see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s own docstring
    # for the exact mechanism (a plain constant substituted every forward
    # pass, disconnected from the underlying parameter, which therefore
    # never receives gradient and stays frozen). Every OTHER term
    # (including w*w_x and every higher cross/product term) stays fully
    # learned. `None` (default): no terms fixed, unchanged behavior.
    spectral_poly_fixed_linear_terms: dict[int, float] | None = None
    # `spectral_poly_exclude_nonconservative` (added 2026-09-12, user-
    # directed: "please try to exclude every even-combined-order two-
    # factor term from the library"): true KS conserves int(u)dx EXACTLY
    # (every term in -u*u_x-u_xx-u_xxxx is a total x-derivative). For a
    # two-factor product term d_i*d_j, repeated integration by parts shows
    # its spatial mean is EXACTLY zero for any state when i+j is odd
    # (reduces to an exact total derivative, e.g. w_xx*w_xxx), and
    # generically NONZERO when i+j is even (reduces to a perfect square
    # (w^(m))^2, m=(i+j)/2 -- e.g. w_x*w_x, the largest measured violator
    # in Section 164's own fit at -0.043). Architecturally zeros every
    # such even-combined-order length-2 term's raw contribution (see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody._poly_weight`'s
    # docstring for the full derivation) -- an EXACT structural fix,
    # unlike the soft `w_pde_mean_conservation` penalty (Section 165),
    # which needs a heuristically-tuned weight and only matches the
    # constraint on the training batch, not for every state. Requires
    # `spectral_poly_degree<=2` -- raises otherwise (the rule is not
    # derived for length>=3 terms). `False` (default): unchanged behavior.
    spectral_poly_exclude_nonconservative: bool = False
    # `spectral_poly_stable_linear_terms` (added 2026-09-14, Section 174,
    # user-directed): user proposal -- parametrize the propagator as a
    # KS-shaped template (`u_t = -u*u_x + nu*u_xx + ...`) with `nu`
    # LEARNABLE but sign-guaranteed and slow-moving, so boundedness is
    # architectural rather than hoped-for. Maps derivative order `n` (for
    # the single-index term `(n,)`) to an INITIAL physical coefficient
    # (must be `< 0`, e.g. `{2: -1.0, 4: -1.0}` for w_xx/w_xxxx, matching
    # true KS's own dissipation operator at init) -- unlike
    # `spectral_poly_fixed_linear_terms` above, this does NOT freeze the
    # value: the coefficient is reparametrized as `-(raw)**2` (`raw` a
    # genuine, trainable `nn.Parameter`), so it always stays negative but
    # its magnitude keeps learning. See
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s own docstring
    # for the full mechanism and motivation (directly targets the root
    # cause diagnosed across Sections 169-173: nothing ever constrained
    # w_xx's sign, so the linear operator was free to become globally
    # damping at every wavenumber -- the actual fixed-point collapse
    # measured identically in all four of those runs). Combine with
    # `Stage{1,2}TrainingConfig.stable_linear_lr_factor` to give these
    # specific parameters a much smaller learning rate than the rest of
    # the network (the "can't make huge steps in nu" requirement).
    # `None` (default): mechanism off, unchanged behavior.
    spectral_poly_stable_linear_terms: dict[int, float] | None = None
    # `spectral_poly_time_deriv` (added 2026-09-18, Section 189, user-
    # directed: "can we incorporate time derivatives into the pde
    # polynomial? might give us a richer expression. We can approximate
    # them using rollout terms potentially"): appends a finite-difference
    # time-derivative feature, `w_t_fd = (w - w_prev) / spectral_poly_
    # time_deriv_dt_snap`, to the polynomial/Chebyshev library -- see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s own
    # `poly_time_deriv` docstring for the full mechanism (where `w_prev`
    # comes from at training vs. rollout time, and why `step_one`'s
    # single-argument regularizer call sites stay unaffected). Requires
    # `spectral_field_kind in ('polynomial', 'chebyshev')` -- validated in
    # `_SpectralPDEDeltaBody.__init__`. `False` (default): unchanged
    # behavior (no extra library variable, `step()` still discards
    # `z_prev` for markovian mode exactly as before).
    spectral_poly_time_deriv: bool = False
    spectral_poly_time_deriv_dt_snap: float = 1.0
    # `spectral_poly_time_deriv2` (added 2026-09-18, Section 190, user-
    # directed: "that might be the next thing to try, incorporate u_tt"):
    # appends a SECOND finite-difference feature, `w_tt_fd = (w - 2*w_prev
    # + w_prev2) / spectral_poly_time_deriv_dt_snap**2` (the standard
    # 3-point BACKWARD second difference -- only past states, no future
    # state needed, matching the rollout-terms framing) -- in ADDITION to
    # `spectral_poly_time_deriv`'s own `w_t`, independent flags (either
    # may be on without the other). Needs a genuine `z_prev2` (state two
    # steps back), threaded through `LatentPropagator.step()`/`.rollout()`
    # -- see `ks_latent.models.propagator._SpectralPDEDeltaBody`'s
    # `poly_time_deriv` docstring in `__init__` for the shared mechanism.
    # `False` (default): unchanged behavior.
    spectral_poly_time_deriv2: bool = False
    # `spectral_burgers_nu_init` (added 2026-09-14, Section 175, user-
    # directed -- a revision of their own Section 174 proposal after the
    # "learnable KS template" mechanism above turned out numerically
    # expensive, ~10-15h for a full Stage-2 run, since even a SIGN-
    # guaranteed (not frozen) `-w_xxxx` term is stiff): `spectral_field_
    # kind="forced_burgers"` only. Initial value for `nu` (real,
    # UNCONDITIONALLY stabilizing viscosity -- reparametrized as
    # `softplus(raw)`, always > 0 regardless of how raw moves under
    # training) in `w_t = -w*w_x + nu*w_xx + g_theta(w, w_x, ...)`, where
    # `g_theta` is a learned, shared-pointwise MLP forcing term (the
    # user's own words: "the forcing function should be a learned output
    # from an mlp depending on the latent state"). Default `1.0` matches
    # true KS/Burgers' own unit-scale diffusion coefficient. See
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s own docstring
    # for the full mechanism and why dropping the 4th-order term entirely
    # (rather than sign-constraining it, Section 174's approach) is both
    # MORE robust (ordinary positive-coefficient diffusion damps every
    # wavenumber unconditionally -- no low-k/high-k sign cancellation to
    # get right) and far cheaper (removes the stiffest eigenvalue in the
    # operator, ~10x relaxation of the forward-Euler stability
    # constraint).
    spectral_burgers_nu_init: float = 1.0
    # `spectral_burgers_beta_max` (added 2026-09-14, Section 175, same
    # day, user-directed: after the first "forced_burgers" version --
    # hardcoded advection at exactly -1.0 -- diverged to nan by epoch
    # 4/k=6 under a real gradual ramp: "why don't we give -u*u_x a
    # coefficient term too, if that's the cause of the instability".
    # `spectral_field_kind="forced_burgers"` only. The nonlinear advection
    # term has its OWN amplitude-dependent CFL-type explicit-Euler
    # stability constraint, entirely independent of nu -- this cannot be
    # fixed by tuning nu alone. `beta = spectral_burgers_beta_max *
    # tanh(raw_beta)`, a genuine learnable parameter but architecturally
    # bounded to `[-beta_max, +beta_max]` regardless of how raw_beta
    # moves under training (same hard-cap philosophy as `delta_cap`).
    # `raw_beta` starts at exactly `0` (matching `zero_init` -- the
    # propagator starts as pure diffusion+forcing, discovering advection
    # strength gradually via the same low-LR `stable_linear_lr_factor`
    # group `nu` uses). Default `1.0` matches true KS/Burgers' own
    # unit-scale advection coefficient as the outer bound.
    spectral_burgers_beta_max: float = 1.0
    # `spectral_burgers_nonlinear_nu` (sketched 2026-09-14, same section,
    # user-directed follow-up: "should we consider creating a pde that
    # isn't a polynomial? it could be a more generic nonlinear function of
    # the derivatives"). `spectral_field_kind="forced_burgers"` only.
    # Generalizes the CONSTANT `nu` to a genuinely nonlinear, state-
    # dependent diffusion coefficient `nu(w, w_x, ..., w^(max_order))`
    # (analogous to real nonlinear-diffusion PDEs, e.g. the porous medium
    # equation) via its OWN independent small MLP trunk, rather than a
    # bare unconstrained `field_kind="mlp"` correction -- nu(x) =
    # softplus(trunk(x)) stays architecturally POSITIVE AT EVERY POINT
    # regardless of training, the same unconditional guarantee the
    # constant-nu case has, just spatially/state-varying. Zero-init
    # preserving: the trunk's final weight is zeroed but its bias is set
    # so nu(x) == spectral_burgers_nu_init exactly everywhere at
    # construction (identical to the constant-nu case's own init value).
    # The entire nu-trunk goes into `stable_linear_lr_factor`'s slow-LR
    # group (see `_SpectralPDEDeltaBody.stable_linear_raw_parameters`'s
    # docstring for why it has its own trunk rather than sharing one with
    # the forcing MLP). `False` (default): unchanged constant-nu behavior.
    # NOT YET LAUNCHED as a real training run at the time this was
    # written -- implemented and unit-verified (CPU-only: zero-init
    # recovers the constant-nu case exactly, nu(x) stays positive under
    # extreme inputs, gradient reaches the trunk) while Section 175 (the
    # constant-nu version) was still training on MPS.
    spectral_burgers_nonlinear_nu: bool = False
    # `spectral_burgers_kernel_instability` (added 2026-09-14, Section 177,
    # user-directed after reading Sakaguchi, "A Simple Model for
    # Spatio-Temporal Chaos in an Unstable Burgers Equation," Prog. Theor.
    # Phys. 103 (2000) 703). `spectral_field_kind="forced_burgers"` only.
    # Sections 175/176's forced_burgers propagators had NOTHING
    # destabilizing at all -- fitted nu/beta barely moved from their
    # (stabilizing) init values, pure damping, hence their collapse
    # (lambda1=-0.269, the worst of the whole arc). Sakaguchi's own
    # unstable Burgers equation gets its chaos-sustaining instability from
    # a NONLOCAL kernel on w_xx, which in Fourier space is an exactly
    # diagonal multiplier `Lhat(k) = -(A*exp(-k^2/width^2) + nu) * k^2` --
    # REPLACES the diffusion term entirely with this (A=0 reduces to the
    # constant-nu case exactly, so this is a strict generalization, not a
    # different starting point). `A` (learnable, architecturally bounded
    # to `[-spectral_burgers_kernel_A_max, +spectral_burgers_kernel_A_max]`
    # via tanh, any sign, starts at 0) is the missing destabilizing
    # ingredient -- the Gaussian envelope means its magnitude stays
    # BOUNDED as k grows (unlike a k^4 term, which is unconditionally
    # unbounded and forced Section 174's ode_substeps~64), while `nu`
    # (learnable, always > 0 via softplus, same mechanism as the constant
    # case) guarantees UNCONDITIONAL high-wavenumber damping regardless of
    # what A/width do, matching Sakaguchi's own stability requirement
    # exactly (g(k)+nu < 0 for small k, g(k) -> 0 rapidly for large k).
    # See ks_latent.models.propagator._SpectralPDEDeltaBody's own
    # docstring for the full derivation. `False` (default): unchanged
    # constant/nonlinear-nu behavior.
    spectral_burgers_kernel_instability: bool = False
    spectral_burgers_kernel_A_max: float = 1.0
    spectral_burgers_kernel_width_init: float = 1.0
    # `spectral_burgers_forcing_max` (added 2026-09-14, Section 178,
    # user-directed after Sections 176/177 both diverged -- standalone
    # rollout max|z| growing ~linearly, 2.5->51 over 100 steps -- while
    # their physically-constrained terms (nu/beta/kernel A) stayed near
    # their safe init values. `spectral_field_kind="forced_burgers"` only.
    # Unlike nu/beta/A, the forcing MLP had NO structural guarantee at
    # all -- it is never trained on states outside Stage 1's own
    # on-attractor k=2 window, so its behavior off-distribution (exactly
    # where an autoregressive rollout ends up once anything drifts) is
    # unconstrained extrapolation. TWO guarantees now applied to the
    # forcing's raw output, in this order: (1) bounded magnitude via
    # `spectral_burgers_forcing_max*tanh(raw/spectral_burgers_forcing_max)`
    # (same cap philosophy as `delta_cap`/`beta`); (2) the CAPPED result's
    # own spatial mean is then subtracted, giving an EXACTLY zero-mean
    # forcing (architecturally guaranteed regardless of training) --
    # matching true KS's own exact mass conservation, the primary fix for
    # the specific near-uniform drift observed in 176/177 (a systematic,
    # same-sign forcing every step is exactly what produces that kind of
    # steady growth). See ks_latent.models.propagator._SpectralPDEDeltaBody's
    # own docstring for the full derivation, including why the final
    # result is bounded by 2x this value (not exactly this value) once
    # the mean-subtraction is applied. Default `1.0`.
    spectral_burgers_forcing_max: float = 1.0
    # `spectral_burgers_kernel_A_fixed` (added 2026-09-16, Section 180,
    # user-directed ablation: "is there any way to get kernel A to
    # activate in 178 and 179? initialize differently? any ideas?").
    # `spectral_field_kind="forced_burgers"` with
    # `spectral_burgers_kernel_instability=True` only. 178/179 both
    # measured `A` drifting slightly NEGATIVE under gradient descent
    # (toward MORE damping, not less), meaning the local k-step MSE
    # gradient actively opposes activating the kernel's destabilizing
    # capacity -- a warm-start init alone would likely just get walked
    # back to ~0 under the slow-LR group over many epochs. This is the
    # clean causal test instead: when set (not None), `A` is FIXED at
    # this exact value (a buffer, not a learnable parameter, no tanh
    # reparametrization -- there is no training dynamic to guard against
    # when it can't move), architecturally forcing a genuine low-k
    # unstable band regardless of what gradient descent would have
    # chosen. Isolates whether the forcing MLP simply compensates for a
    # forced-unstable `A` (reproducing the same frozen-band collapse,
    # meaning the kernel was never the bottleneck) or whether real chaos
    # emerges once the model can no longer avoid a destabilizing linear
    # term. `None` (default): unchanged learnable-A behavior.
    spectral_burgers_kernel_A_fixed: float | None = None
    # `spectral_burgers_beta_fixed` (added 2026-09-16, Section 181,
    # user-directed follow-up to the kernel-A ablation: "if we remove
    # delta_cap what would happen?" -> [explained: real KS-class
    # saturation comes from the advection term cascading energy from
    # unstable low-k modes to damped high-k modes; beta has ALSO stayed
    # near 0 in every one of 175-180's fitted values, so with only A
    # forced unstable, delta_cap was the ONLY thing bounding growth,
    # producing an artificial freeze at a larger scale instead of genuine
    # nonlinear saturation] -> "sure try that"). Exactly analogous to
    # `spectral_burgers_kernel_A_fixed`: when set, `beta` (the advection
    # coefficient) is a FIXED buffer, not a learnable parameter -- forces
    # the nonlinear advection term to genuinely engage regardless of what
    # gradient descent would have chosen. Tests whether real
    # energy-cascade saturation (bounded chaos at the correct physical
    # scale) emerges once BOTH the destabilizing (A) and the
    # saturating-nonlinearity (beta) mechanisms are active
    # simultaneously, rather than delta_cap alone doing all the
    # (artificial) bounding work. `None` (default): unchanged learnable-
    # beta behavior.
    spectral_burgers_beta_fixed: float | None = None
    # `spectral_burgers_kernel_mu_init` (added 2026-09-16, Section 182,
    # user-directed: "let's keep the sakaguchi setup, add a
    # hyperviscosity term and get rid of the forcing mlp term"). Section
    # 181 (kernel A AND beta both forced active) made things WORSE, not
    # better (RMSE climbing monotonically to ~32 over 200 steps, never
    # saturating) -- hypothesis: real KS-class saturation needs 4th-order
    # hyperviscosity to absorb energy the advection term cascades to high
    # k (the "forced_burgers" family, Section 175 onward, only ever had
    # 2nd-order diffusion). Extends Lhat(k) with a genuine `-mu*k^4` term,
    # `mu=softplus(raw)` always > 0 (real damping, never anti-damping),
    # matching true KS's own `-w_xxxx`. `None` (default): term absent
    # entirely, unchanged behavior. See
    # ks_latent.models.propagator._SpectralPDEDeltaBody's docstring for
    # the default init value's derivation (chosen to respect this
    # backbone's ode_substeps=1 forward-Euler stability budget).
    spectral_burgers_kernel_mu_init: float | None = None
    # `spectral_burgers_no_forcing` (added 2026-09-16, Section 182, same
    # user direction). Removes the forcing MLP term entirely (no
    # input_proj/blocks/final_ln/output_proj built at all, field()
    # returns a plain zero for it) -- leaving ONLY the hardcoded/
    # constrained physical terms (advection, diffusion, kernel
    # instability, hyperviscosity), matching Sakaguchi's own equation (1)
    # structurally (no separate learned correction term at all). Tests
    # whether the forcing MLP (still substantial in magnitude even
    # mean-zero, ~0.7-1.0 in Sections 178-181) was itself contributing to
    # or masking the dynamics, rather than the pure physical terms alone
    # being incapable of bounded chaos. `False` (default): unchanged
    # forcing-MLP behavior.
    spectral_burgers_no_forcing: bool = False
    # `spectral_burgers_kernel_A_init`/`spectral_burgers_beta_init` (added
    # 2026-09-16, Section 183, user-directed: "let's initialize with the
    # same parameters but let A and beta be trainable"). Only used when
    # `spectral_burgers_kernel_A_fixed`/`spectral_burgers_beta_fixed` are
    # `None` (learnable case) -- sets the LEARNABLE parameter's own
    # starting point via inverse-tanh (instead of always starting at 0),
    # e.g. `-2.0`/`-1.0` to start from the same genuinely-destabilizing
    # values Sections 180-182 hard-FIXED, but now letting gradient
    # descent adjust them freely from there. Requires
    # `|spectral_burgers_kernel_A_init| < spectral_burgers_kernel_A_max`
    # (respectively beta) strictly, since atanh diverges at +-1 -- e.g.
    # raise spectral_burgers_kernel_A_max to 3.0 for an A_init of -2.0.
    # `0.0` (default): unchanged zero-init behavior.
    spectral_burgers_kernel_A_init: float = 0.0
    spectral_burgers_beta_init: float = 0.0
    # "masked_mlp_expand" backbone only (added 2026-09-10, Section 128,
    # user-directed: "in the middle layer of the mlp, expand the dimension
    # 3x"). Sizes that middle (hidden) layer's width as `expand_factor *
    # d_latent` -- see `ks_latent.models.propagator._MaskedMLPExpandDeltaBody`'s
    # docstring for the full architecture and the exact derivation of why
    # this, combined with `attn_window` (reused as the layer's circular-band
    # window, referenced in `d_latent`-ring units throughout, same
    # convention as `masked_mlp_wide`), gives an effective one-sided
    # neighbor radius of `attn_window * expand_factor` inside the widened
    # middle layer. Default `3` matches the request this backbone was built
    # for exactly.
    masked_mlp_expand_factor: int = 3

    def __post_init__(self):
        _validate_propagator_mode_backbone(self.mode, self.backbone)
        if self.mode == "history" and self.n_history < 2:
            raise ValueError(f"mode='history' requires n_history >= 2, got {self.n_history!r}")
        if self.backbone in ("transformer", "vit", "fno_vit", "local_mlp") and self.d_latent % self.n_tokens != 0:
            raise ValueError(
                f"d_latent={self.d_latent} must be divisible by n_tokens={self.n_tokens} "
                f"for the {self.backbone!r} backbone"
            )
        if self.pos_encoding not in ("circular", "linear"):
            raise ValueError(f"pos_encoding must be 'circular' or 'linear', got {self.pos_encoding!r}")
        if self.token_window is not None and self.token_window < self.d_latent // self.n_tokens:
            raise ValueError(
                f"token_window={self.token_window} must be >= chunk_size="
                f"{self.d_latent // self.n_tokens} (= d_latent // n_tokens)"
            )
        if self.delta_cap is not None and self.delta_cap <= 0:
            raise ValueError(f"delta_cap must be > 0, got {self.delta_cap!r}")
        if self.backbone == "fno_vit" and self.fno_n_layers < 1:
            raise ValueError(f"fno_n_layers must be >= 1 for backbone='fno_vit', got {self.fno_n_layers!r}")
        if self.fno_modes is not None and self.fno_modes < 1:
            raise ValueError(f"fno_modes must be >= 1, got {self.fno_modes!r}")
        if self.backbone == "local_mlp" and self.attn_window is None:
            raise ValueError(
                "backbone='local_mlp' requires attn_window to be set (its local-conv kernel "
                "radius) -- there is no 'global local_mlp'; use 'mlp' or 'fno_vit' for that."
            )
        if self.backbone == "spectral_pde":
            if self.spectral_K is None or self.spectral_N_w is None:
                raise ValueError(
                    "backbone='spectral_pde' requires spectral_K and spectral_N_w to be set"
                )
            if self.d_latent != 2 * self.spectral_K:
                raise ValueError(
                    f"backbone='spectral_pde' requires d_latent == 2*spectral_K "
                    f"(got d_latent={self.d_latent!r}, spectral_K={self.spectral_K!r})"
                )
            n_freq = self.spectral_N_w // 2 + 1
            if self.spectral_K < 1 or self.spectral_K > n_freq:
                raise ValueError(
                    f"spectral_K={self.spectral_K!r} must be in [1, spectral_N_w//2+1={n_freq!r}]"
                )
            if self.spectral_L <= 0:
                raise ValueError(f"spectral_L must be > 0, got {self.spectral_L!r}")
            if self.spectral_max_order < 0:
                raise ValueError(f"spectral_max_order must be >= 0, got {self.spectral_max_order!r}")
            if self.spectral_integrator not in ("euler", "rk4", "etdrk4"):
                raise ValueError(
                    f"spectral_integrator must be 'euler', 'rk4', or 'etdrk4', "
                    f"got {self.spectral_integrator!r}"
                )
            _validate_spectral_field_kind(
                self.spectral_field_kind, self.spectral_poly_degree, self.spectral_poly_max_term_order,
                self.spectral_poly_norm_power,
            )
            if self.spectral_physics_prior and self.spectral_max_order < 4:
                raise ValueError(
                    "spectral_physics_prior requires spectral_max_order >= 4 "
                    "(needs w_x for the nonlinear term and w_xx/w_xxxx for the linear "
                    "one, except under 'etdrk4' where the linear part is handled "
                    "separately -- max_order>=4 is required uniformly regardless, for "
                    "a consistent feature set across integrators)"
                )
        if self.backbone == "spectral_pde_raw":
            # See _SpectralPDERawDeltaBody's docstring: works on ANY
            # encoder's raw z (no d_latent==2*spectral_K constraint --
            # spectral_K here is purely an internal self-FFT truncation
            # parameter, independent of d_latent).
            if self.spectral_K is None:
                raise ValueError("backbone='spectral_pde_raw' requires spectral_K to be set")
            n_freq = self.d_latent // 2 + 1
            if self.spectral_K < 1 or self.spectral_K > n_freq:
                raise ValueError(
                    f"spectral_K={self.spectral_K!r} must be in [1, d_latent//2+1={n_freq!r}] "
                    f"(backbone='spectral_pde_raw' treats d_latent={self.d_latent!r} as the "
                    f"field length for its own self-FFT)"
                )
            if self.spectral_L <= 0:
                raise ValueError(f"spectral_L must be > 0, got {self.spectral_L!r}")
            if self.spectral_max_order < 0:
                raise ValueError(f"spectral_max_order must be >= 0, got {self.spectral_max_order!r}")
            if self.spectral_integrator not in ("euler", "rk4", "etdrk4"):
                raise ValueError(
                    f"spectral_integrator must be 'euler', 'rk4', or 'etdrk4', "
                    f"got {self.spectral_integrator!r}"
                )
            _validate_spectral_field_kind(
                self.spectral_field_kind, self.spectral_poly_degree, self.spectral_poly_max_term_order,
                self.spectral_poly_norm_power,
            )
            if self.spectral_physics_prior:
                raise ValueError(
                    "spectral_physics_prior is not defined for backbone='spectral_pde_raw' -- "
                    "there is no 'true governing equation' for an arbitrary, LEARNED latent "
                    "ordering the way there is for a genuine w=irfft(z) physical field "
                    "(backbone='spectral_pde' only)."
                )


@dataclass(frozen=True)
class PropagatorConfig:
    """Stage-2 latent propagator (brief §5.2: L_b=3, H=128, dropout=0.1).

    See `AuxPropagatorConfig`'s docstring for `mode`/`backbone` (identical
    semantics; both propagators in this codebase share the same
    architecture family and mode flag).

    `delta_cap` (added 2026-08-29, user-directed): architectural bound on
    the per-step residual, `delta = delta_cap * tanh(raw_delta /
    delta_cap)`, applied inside `LatentPropagator.step`/`step_one` -- NOT a
    training-only augmentation (unlike `Stage2TrainingConfig`'s
    `noise_in`/`noise_step`); it changes what the model can output at both
    train and eval/inference time. Motivated by a real failure: MSE
    training on bounded-horizon rollouts has no way to keep predictions
    from diverging except by biasing the learned map toward contraction
    (eigenvalues inside the unit circle) everywhere, which was measured to
    collapse the trained propagator to a single global fixed point
    (D_KY=0) instead of reproducing KS's genuinely chaotic latent dynamics
    (the reference project's own achieved D_KY~21.4, 11 positive Lyapunov
    exponents -- see docs/PROJECT_HANDOFF.md). Real chaotic maps stay
    bounded via *stretch-and-fold* (local expansion + a nonlinear fold),
    not stretch-and-contract; capping the step size architecturally
    provides the "fold" so training no longer needs to suppress local
    expansion just to avoid blowup. `tanh(x/delta_cap)*delta_cap ~= x` for
    small `x`, so the zero-init identity-at-init property is unaffected
    (`raw_delta=0` at init regardless of `delta_cap`). Default `None`
    (off, unchanged behavior); must be `> 0` if set.

    `mode="history"`/`n_history` (added 2026-08-29, user-directed: "the
    original model for this project used a propagator with information
    from the current step and a step in the past... design a ViT model
    that takes in several states from the past, and propagate them into
    the future"): generalizes `"two_step"` (fixed at exactly 2 states, MLP
    only) to an arbitrary-length history `z_hist` of shape `(B, n_history,
    d_latent)`, oldest to newest. Only implemented with `backbone="vit"`
    (`_ViTHistoryDeltaBody`): each of the `n_history` states is tokenized
    (shared spatial embedding + a learned per-time-step offset), all
    `n_history * n_tokens` tokens attend jointly (space *and* time), and
    only the *most recent* time-slice's tokens are projected back out to a
    `chunk_size`-wide delta -- the older states are context the attention
    can draw on, not additional outputs. `n_history=3` (default) is "2
    past states + current", the case explicitly requested first. Uses
    `.step_history(z_hist)`/`.rollout_history(z_hist, k)` instead of the
    fixed-arity `.step(z_prev, z_curr)`/`.rollout(...)` every other mode
    uses -- calling the latter on a `mode="history"` propagator raises
    (fail loudly, brief ground rule 2), since there is no well-defined way
    to fit an arbitrary-length history into a 2-argument signature.

    `backbone="node"` (Neural ODE, added 2026-08-31, user-directed --
    Phase 2 architecture doc Section 42/43, the most literal reading of
    "model the latent variable with a PDE"): parameterizes `dz/dt =
    f_theta(z)` with a translation-equivariant, weight-shared local
    circular-convolution vector field (same locality-without-softmax
    reasoning as `"local_mlp"`, but genuinely weight-tied across the
    latent-index axis, unlike `"masked_mlp"`'s per-position-independent
    masked weights), integrated via fixed-step RK4 over `ode_substeps`
    sub-steps spanning one unit of the dataset's own snapshot-index
    interval (every dataset this project uses is built at `dt_snap=1.0`,
    and every OTHER backbone already implicitly learns `z_n -> z_n+1` in
    that same unit, so no literal physical `dt` needs threading through
    here). Reuses `attn_window` as the vector field's conv kernel radius
    (required, not `None` -- same convention as `"local_mlp"`) and
    `hidden`/`n_blocks` as the vector field's channel width/residual
    block count. `zero_init=True` (default) zeros the vector field's
    final projection, so `f_theta(z)=0` everywhere at init and RK4
    collapses to the exact identity -- matching every other backbone's
    identity-at-init property.

    `backbone="masked_mlp_wide"` (added 2026-09-06, Section 100,
    user-directed: "For the propagator let's use a single layer mlp with
    more parameters with limited attention window (say like 4). Please add
    the proper number of parameters to match the size of the propagator in
    98, off diagonal terms (from the attention window) shouldn't count."):
    a SINGLE hidden layer (one `MaskedLinearRect(d_latent, hidden,
    attn_window)` expansion, one `GELU`, one `MaskedLinearRect(hidden,
    d_latent, attn_window)` contraction back down -- `n_blocks` is not used,
    unlike every other backbone here, since "single layer" is structural,
    not a configurable depth of 1) -- structurally the direct "wide but
    shallow" counterpart to `"masked_mlp"` (which stays at `d_latent` width
    throughout but can stack `n_blocks` residual layers). Both
    `MaskedLinearRect`s share `attn_window` as their circular-band window,
    referenced against `d_latent`'s own ring (same convention as
    `"masked_mlp"`'s `MaskedLinear`) -- masked-off (out-of-band) weight
    entries are exactly zero for the entire time this module is trained
    (see `MaskedLinearRect`'s docstring), so they never affect the forward
    computation regardless of `hidden`.
    Sizing (`hidden`): because `attn_window=4` at `d_latent=44` only
    activates ~18% of each `MaskedLinearRect`'s entries (a `+-4`-token band
    out of a 44-ring), matching another backbone's ACTIVE (non-zero)
    parameter count at this narrow a window requires a MUCH wider hidden
    layer than that backbone's own hidden width -- e.g. matching
    Section 98's `"mlp"`/`hidden=128`/`n_blocks=3` propagator's 111,532
    total params active-for-active requires `hidden~=6558` here (~51x
    wider), verified by direct search over `_circular_band_mask_rect`'s
    exact nonzero count. This is an explicit, deliberate design
    consequence of "off diagonal terms ... shouldn't count" (an
    apples-to-apples ACTIVE-parameter comparison), not a sizing mistake --
    flagged here because a >6000-wide single hidden layer is unusually
    wide relative to this project's typical 128-256 propagator hidden
    widths, purely a byproduct of how sparse a `+-4` window is at
    `d_latent=44`. `zero_init=True` (default) zeros the final
    `MaskedLinearRect`'s weight/bias, matching every other backbone's
    identity-at-init property. Requires `mode="markovian"` and
    `attn_window` to be set (same convention as `"local_mlp"`/`"node"`).

    `backbone="masked_mlp_expand"` (added 2026-09-10, Section 128,
    user-directed: "I want to use a masked mlp with three layers,
    attention_window=3 for the propagator... in the middle layer of the
    mlp, expand the dimension 3x, but in this expanded dimensions, still
    respect the attention window (I guess it would be 9 in that case, so
    only a very limited number of neighboring values interact.)"): THREE
    `MaskedLinearRect` layers (`input_proj`: `d_latent -> hidden`,
    `mid_proj`: `hidden -> hidden` -- "the middle layer" -- `output_proj`:
    `hidden -> d_latent`), `hidden = masked_mlp_expand_factor * d_latent`
    (default factor `3`), all sharing `attn_window` referenced against
    `d_latent`'s own ring (`ref_dim=d_latent`, same convention as
    `"masked_mlp_wide"`). This makes the user's own "I guess it would be
    9" arithmetic exact, not approximate: `mid_proj`'s one-sided neighbor
    radius, expressed in `hidden`-index units, works out to `attn_window *
    masked_mlp_expand_factor` (`3 * 3 = 9` at the defaults) -- see
    `ks_latent.models.propagator._MaskedMLPExpandDeltaBody`'s docstring
    for the full derivation. `n_blocks`/`hidden` are unused (three fixed
    layers, not a configurable depth or width -- `masked_mlp_expand_factor`
    is the only width knob). `zero_init=True` (default) zeros
    `output_proj`'s weight/bias. Requires `mode="markovian"` and
    `attn_window` to be set (same convention as `"masked_mlp_wide"`).

    `backbone="cnn"` (added 2026-09-01, user-directed: "try a CNN for a
    propagator too... based on the best features of the current MLP...
    1d in the spatial dimension, with width 16"): the same
    `input_proj -> n_blocks residual blocks -> final LayerNorm ->
    output_proj (zero-init)` structure `"mlp"` uses, but every
    cross-index `Linear` inside the residual blocks replaced by a
    circular `Conv1d` of kernel size `cnn_kernel_size` (a literal kernel
    WIDTH, unlike `"node"`/`"local_mlp"`/`"masked_mlp"`'s `attn_window`,
    which is a RADIUS -- `2*window+1`). Reuses `hidden`/`n_blocks` for the
    conv channel width/residual block count, same as `"mlp"`. Genuinely
    translation-equivariant (weight-shared conv kernels), unlike
    `"masked_mlp"`. See `ks_latent.models.propagator._CNNDeltaBody`'s
    docstring for the full design rationale, including a direct warning
    about `"node"`'s wide-circular-padding MPS slowdown this backbone was
    deliberately kept cheap (no RK4 sub-stepping) to avoid, but was not
    proven immune to until directly timed.

    `backbone="fno_mlp"` (added 2026-09-03, user-directed: "a FNO based MLP
    might work better for the propagator" -- direct follow-up to
    docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 66's finding that the
    latent index already exhibits genuine periodic/single-low-order-
    Fourier-mode structure under `w_spatial_signed` training): the FNO half
    of `"fno_vit"` (`fno_n_layers` `FNOLayer` spectral-conv blocks, global
    low-frequency mixing across the circular latent index) with the ViT
    attention stage REPLACED by `token_n_layers` plain per-token MLP blocks
    (`ks_latent.models.autoencoder_vit.TokenMLPBlock`: pre-norm residual
    `x + MLP(LN(x))`, no attention) instead of `token_n_layers` `ViTBlock`s.
    Cheaper than `"fno_vit"` (no attention at all) and, unlike every other
    backbone here except `"node"`/`"cnn"`, genuinely translation-equivariant:
    NO positional encoding is used (the `pos_encoding` field is accepted for
    CLI uniformity but ignored) -- the per-token MLP is applied identically
    at every token position (shared weights), and the FNO spectral-conv path
    is exactly translation-equivariant by construction (multiplication in
    Fourier space), so the whole body has no way to distinguish one token
    position from another except through the data itself. This mirrors the
    KS PDE's own spatial homogeneity (no explicit `x`-dependence in `u_t +
    u*u_x + u_xx + u_xxxx = 0`) and is a direct bet that the periodic
    structure Section 66 found the network already produces is the RIGHT
    inductive bias to build in architecturally, rather than only encourage
    via a loss term. Reuses `token_n_layers`/`token_mlp_ratio` (MLP block
    count/hidden-ratio) and `fno_modes`/`fno_n_layers` (spectral-conv mode
    count/layer count) fields; `token_nhead`/`attn_window`/`pos_encoding`
    are accepted but unused (no attention exists to apply them to).

    `backbone="fourier_mlp"`, `mode="history"` OR `mode="markovian"`
    (added 2026-09-03, user-directed: "an mlp model that takes in fourier
    features (such as the FNO) and also actual latent states, as kind of
    a hybrid mlp"; `mode="markovian"` support added 2026-09-04,
    user-directed -- "make the fourier mlp markovian" -- Section 84):
    unlike `"fno_mlp"` (whose FNO layers are a LEARNED spectral-conv
    operator replacing the raw latent state entirely), this backbone
    computes a fixed, unlearned Fourier FEATURIZATION -- for each of the
    `n_history` states (`mode="history"`) or the single current state
    (`mode="markovian"`, internally `n_history=1`, no separate history
    states to featurize -- there is only ever "the current state" in
    that mode), `torch.fft.rfft` along the latent index, keeping
    the first `min(fno_modes, d_latent//2+1)` frequencies' real and
    imaginary parts (`2*fno_modes` numbers) -- and concatenates it
    alongside (not instead of) that state's raw `d_latent` values. Every
    history state's [raw || Fourier] features are flattened together into
    one vector and fed through the same plain residual-MLP body
    `backbone="mlp"` uses for its history mode (`MLPDeltaBody`, reusing
    `hidden`/`n_blocks`) -- no attention, no tokenization, no positional
    encoding; `n_tokens`/`token_d_model`/`token_nhead`/`token_n_layers`/
    `token_mlp_ratio`/`pos_encoding`/`token_window` are all unused.
    `n_history` itself is unused/ignored under `mode="markovian"` (fixed
    at 1 internally by `LatentPropagator`, not read from `cfg.n_history`,
    exactly like every other markovian backbone ignores `n_history`).

    `attn_window` (added 2026-09-03, user-directed: "use all the fourier
    coefficients, but only let real variables interact with their
    neighbors"): `None` (default) keeps the fully-dense behavior above.
    Set to a finite radius to instead split into two independent, SUMMED
    paths -- a circular-band-MASKED path (`_MaskedMLPDeltaBody`, same
    mechanism as `backbone="masked_mlp"`) on the CURRENT state's raw
    values only, restricting which raw indices can influence each other,
    plus a fully DENSE path on every history state's Fourier features
    (all of them, unrestricted -- a single Fourier coefficient summarizes
    the whole state already, so index-masking it has no analogous
    meaning). See `_FourierMLPHistoryDeltaBody`'s docstring for the exact
    split.

    Motivated directly by docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md
    Section 66's discovered latent-index periodicity: handing the network
    low-frequency Fourier-domain coordinates directly as EXTRA input
    features (on top of the raw state, not replacing it) removes the need
    for a plain MLP to reconstruct that structure implicitly from raw
    index values alone -- a bet that giving a useful coordinate system
    directly beats making the network rediscover it, similar in spirit to
    how positional encodings are handed to attention models rather than
    learned from nothing. `fno_modes=None` (default) uses every available
    frequency (`d_latent//2+1`), i.e. the raw-vs-Fourier features carry
    the same real information content in different bases -- pass an
    explicit smaller `fno_modes` to restrict to low frequencies only, the
    tighter version of the "recover the periodic structure" bet.
    """

    d_latent: int = 44
    hidden: int = 128
    n_blocks: int = 3
    dropout: float = 0.1
    zero_init: bool = True
    mode: str = "two_step"
    backbone: str = "mlp"
    n_tokens: int = 4
    token_d_model: int = 32
    token_nhead: int = 2
    token_n_layers: int = 2
    token_mlp_ratio: int = 4  # "vit" backbone only
    attn_window: int | None = None  # "local_mlp" backbone: reused as its conv kernel radius, required (not None) -- see AuxPropagatorConfig's docstring
    pos_encoding: str = "circular"  # "vit" backbone only; see AuxPropagatorConfig's docstring
    delta_cap: float | None = None
    # `delta_cap_relative` (added 2026-09-17, Section 184, user-directed
    # after Sections 180-183 ALL converged to the identical failure
    # signature -- sustained growth to a large, unphysical scale, never
    # settling to bounded chaos at the correct scale -- regardless of
    # fixed vs. learnable A/beta, hyperviscosity, removing the forcing
    # MLP, or a regularizer guaranteeing genuine instability survives.
    # The one thing common to all four: `delta_cap` is a FIXED ABSOLUTE
    # per-step bound, not scaled to the state's own magnitude -- once
    # real instability is present, growth continues until whatever scale
    # makes that fixed bound negligible in RELATIVE terms, rather than
    # reaching a genuine dynamical balance. When `True`, `capped_delta`
    # uses `delta_cap * ||z_ref||` (the reference state's own per-sample
    # norm) as the cap instead of the bare `delta_cap` constant -- same
    # tanh-saturation shape, but the allowed step size now grows
    # proportionally with the state's own scale instead of staying fixed.
    # `False` (default): unchanged absolute-cap behavior. Only has an
    # effect when `delta_cap` is also set.
    delta_cap_relative: bool = False
    token_window: int | None = None  # "vit" backbone only; see AuxPropagatorConfig's docstring
    n_history: int = 3  # mode="history" only; total states including current
    # "fno_vit" backbone only (added 2026-08-30, user-directed "Solution 2"
    # for the fixed-point collapse: "a combination of a Fourier neural
    # operator and a vision transformer... in Fourier space we preserve
    # spatial relationships" -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md
    # Section 5 and `ks_latent.models.propagator._FNOViTDeltaBody`).
    # Number of low-frequency Fourier modes each spectral-conv layer keeps;
    # None (default) keeps every rfft mode (n_tokens // 2 + 1, no
    # truncation).
    fno_modes: int | None = None
    # "fno_vit" backbone only. Number of stacked FNO spectral-conv layers
    # before the token_n_layers ViT attention blocks.
    fno_n_layers: int = 2
    # "node" backbone only: number of fixed-step RK4 sub-steps per
    # snapshot-index interval -- see this class's "node" docstring section.
    ode_substeps: int = 4
    # "cnn" backbone only: circular Conv1d kernel WIDTH (not a radius,
    # unlike attn_window) -- see this class's "cnn" docstring section.
    cnn_kernel_size: int = 16
    # "fourier_mlp" backbone only (added 2026-09-03, user-directed: "apply
    # an inverse fft to the frequency component output of the fourier mlp
    # to map back to state space ... in the propagator"). False (default,
    # unchanged behavior): the Fourier-features-only sub-network
    # (_FourierMLPHistoryDeltaBody's fourier_body, or its entire dense
    # body when attn_window is None) reads out to d_latent via a plain
    # unconstrained MLPDeltaBody, as before. True: that sub-network
    # instead becomes a FourierIFFTBody (ks_latent.models.propagator) --
    # an MLPDeltaBody that predicts frequency-domain coefficients,
    # explicitly inverse-transformed (irfft) back to d_latent. (A
    # genuinely complex-valued version of FourierIFFTBody's internals was
    # tried and reverted same day -- diverged over a realistic training
    # horizon even after adding a stabilizing complex norm; see
    # FourierIFFTBody's own docstring for the full account.) SUMMED
    # with the existing masked raw-value path (`masked_body`, at radius
    # `attn_window`) whenever `attn_window` is not None, exactly the same
    # two-network structure as `FourierMLPAutoencoderConfig`'s encoder
    # (corrected 2026-09-03, user-directed: "I want the propagator and the
    # decoder to have the same structure as the encoder... but I want the
    # attn_window to be much larger" -- an earlier version of this option
    # wrongly dropped the raw-value path entirely when `attn_window` was
    # `None` ("dense"), instead of the intended meaning: keep the same
    # masked-raw + Fourier-irfft structure, just with a much larger
    # `attn_window` than the encoder's). `attn_window=None` still means no
    # raw-value path at all (pure Fourier+irfft) -- to get the two-network
    # structure, set `attn_window` to a large but finite radius.
    fourier_ifft_readout: bool = False
    # "fourier_mlp" backbone only (added 2026-09-04, user-directed: "would
    # there be a way to constrain the fourier_mlp to be nonexpansive").
    # False (default, unchanged behavior). True: every Linear in
    # masked_body/fourier_body is spectral-normalized, residual
    # connections become damped/averaged (0.5*(x+h)), rfft/irfft use
    # norm="ortho", and the masked+Fourier SUM becomes an AVERAGE instead
    # -- see ks_latent.models.propagator.ResidualMLPBlock's and
    # _FourierMLPHistoryDeltaBody's docstrings for the full mechanism
    # (including the documented output_proj/zero_init compromise: the
    # very last linear readout of each body stays unconstrained, since
    # spectral_norm is incompatible with an exact-zero zero_init weight).
    nonexpansive: bool = False
    # `backbone="spectral_pde"` only (added 2026-09-06,
    # docs/sine_transform_pde_plan.md): `d_latent` must equal `2*spectral_K`
    # (this backbone's `z` is a truncated rFFT spectrum, `spectral_K` real +
    # `spectral_K` imaginary parts, the SAME convention
    # `SpectralFieldAutoencoderConfig` uses). `spectral_N_w` is the physical
    # grid length derivative fields are synthesized at (must match the
    # paired autoencoder's own `N_w`); `spectral_L` the physical domain
    # length (must match the training dataset's `KSConfig.L`) -- both
    # required, not inferrable from `d_latent` alone.
    spectral_K: int | None = None
    spectral_N_w: int | None = None
    spectral_L: float = 100.0
    # Highest spatial derivative order synthesized (0..spectral_max_order
    # inclusive, so the shared pointwise MLP sees `spectral_max_order + 1`
    # features per point: w, w_x, w_xx, ..., w^(spectral_max_order)).
    # Default 4 matches true KS's own governing equation exactly (u, u_xx,
    # u_xxxx, plus u_x for the nonlinear term); the user's request allows up
    # to 5 or 6 for extra headroom.
    spectral_max_order: int = 4
    # "euler": forward Euler, sub-stepped over ode_substeps steps of size
    # h=1/ode_substeps each (added 2026-09-07, user-directed: "couldn't we
    # also integrate euler over multiple steps?" -- previously hardcoded
    # to always take a single full step regardless of ode_substeps,
    # inconsistent with "rk4"/"etdrk4" and silently ignoring whatever
    # ode_substeps a config claimed). ode_substeps=1 (the default) is
    # exactly the original single-step form (w(t+1) = w(t) + field(z(t))),
    # matching the proposal's literal (w(t+1)-w(t))/dt finite-difference
    # training target. "rk4": reuses the same fixed-step RK4 integration
    # `backbone="node"`'s `_NeuralODEDeltaBody` already implements
    # (`ode_substeps` sub-steps spanning one unit of dt_snap), operating on
    # `w` between substeps and re-encoding to `z` (via the same fixed
    # rFFT-truncate transform the paired autoencoder uses) to evaluate
    # `field` at each RK stage.
    #
    # "etdrk4" (added 2026-09-06, docs/sine_transform_pde_plan.md, user-
    # directed: "how closely does this method mirror the actual method we
    # use to integrate the KS system in the code? ... being as close as
    # possible to that methodology will give us the best chance for
    # success"): mirrors `ks_latent/solver/ks.py`'s own ETDRK4 solver
    # (Kassam & Trefethen 2005) directly, reusing its `linear_operator`/
    # `etdrk4_coefficients` functions -- splits the dynamics into the
    # EXACTLY-known, diagonal LINEAR part (`Lhat(k) = k^2 - k^4`, the
    # Fourier symbol of `-u_xx - u_xxxx`, fixed at KS's own true value,
    # never learned) integrated via the exact integrating factor
    # `exp(dt*Lhat)`, plus the pointwise MLP's output standing in for the
    # solver's hand-coded nonlinear term `N(v)` (both are evaluated at
    # 4 RK stages combined via the SAME `E/E2/Q/f1/f2/f3` coefficients the
    # real solver uses -- see `_SpectralPDEDeltaBody`'s docstring for the
    # exact correspondence). Unlike "euler"/"rk4" (which ask the MLP to
    # learn the ENTIRE right-hand side, including the stiff linear part,
    # through a generic explicit integrator with no stiffness protection),
    # this removes that burden entirely -- the MLP only has to learn the
    # nonlinear residual, and the scheme is unconditionally stable for the
    # linear part regardless of step size, exactly like the ground-truth
    # solver. Reuses `ode_substeps` for sub-stepping (each substep's own
    # `dt` used to compute the ETDRK4 coefficients once at construction,
    # not re-derived per forward pass -- `Lhat` is fixed, not trainable).
    # NOTE this backbone's usual "identity at init" property does NOT hold
    # for this option: with the MLP zeroed (`zero_init=True`), `N(v)=0`
    # identically, so the propagator reduces to `z_next = exp(Lhat)*z` --
    # the EXACT linearized-KS map, not a no-op -- see
    # `_SpectralPDEDeltaBody`'s docstring.
    spectral_integrator: str = "euler"
    # `spectral_physics_prior` (added 2026-09-08, docs/sine_transform_pde_plan.md,
    # user-directed: "what if we had another variant that assumes that the
    # latent space follows exactly the KS dynamics with some learned
    # correction term ... That way we already have chaos in the
    # formulation"): bakes the EXACT true KS right-hand side into `field`'s
    # output as a fixed baseline, with the pointwise MLP now learning only
    # a CORRECTION on top of it, rather than the whole dynamics from
    # scratch. Motivated by a real, checkable fact from dynamical systems
    # theory: Lyapunov exponents are invariant under a smooth, invertible
    # change of coordinates -- if the learned intermediate field `w` is a
    # reasonably faithful (close to invertible, which good reconstruction
    # already pressures the encoder toward) reparametrization of the true
    # physical state, then baking in the exact KS dynamics for `w` should
    # inherit KS's own genuine chaos, essentially "for free," rather than
    # requiring training to discover it -- unlike `"etdrk4"` alone (which
    # only bakes in the LINEAR part; the linear part alone is not chaotic,
    # unstable modes there just grow unboundedly -- it's specifically the
    # nonlinear term that folds that instability into a bounded chaotic
    # attractor).
    #
    # Integrator-aware to avoid double-counting with `"etdrk4"`'s own exact
    # linear treatment (see `_SpectralPDEDeltaBody.field`'s docstring for
    # the exact formulas): for `"etdrk4"`, only the nonlinear term `-w*w_x`
    # is baked in (the linear part `-w_xx-w_xxxx` is already handled
    # separately, exactly, via `exp(dt*Lhat)` -- baking it into `field`
    # TOO would apply it twice). For `"euler"`/`"rk4"` (which have no
    # separate linear treatment), the FULL exact RHS `-w*w_x - w_xx -
    # w_xxxx` is baked in.
    #
    # Requires `spectral_max_order >= 4` (needs `w_x` for the nonlinear
    # term, `w_xx`/`w_xxxx` for the linear one). `zero_init=True` (the
    # correction MLP's output zeroed) now means `field(z)` returns EXACTLY
    # the analytic KS right-hand side at init, not zero and not
    # `exp(Lhat)*z` alone -- a genuinely different, more informative
    # starting point than every other option here, verified directly (not
    # merely asserted) in this module's tests.
    #
    # CAVEAT (stated directly, not hidden): the KS equation is not
    # scale-invariant (rescaling `u -> c*u` scales the nonlinear and linear
    # terms differently), so this baseline is most exact if `w`'s own
    # learned amplitude roughly matches the scale the equation was derived
    # in -- nothing currently pins that exactly; the correction MLP can in
    # principle absorb a scale mismatch (it is a general nonlinear
    # function) but less elegantly than if `w` were already well-scaled.
    spectral_physics_prior: bool = False
    # See AuxPropagatorConfig.spectral_field_kind's docstring for the full
    # semantics (identical here) -- this class's own version, near
    # AuxPropagatorConfig's earlier-declared copy, carries the canonical
    # docstring; this is Stage-2/PropagatorConfig's mirror of it.
    spectral_field_kind: str = "mlp"
    spectral_poly_degree: int = 2
    # "spectral_pde"/"spectral_pde_raw", field_kind="polynomial" only
    # (added 2026-09-09, user-directed: "really only let the combined
    # degree of the terms be less than 5 (so w_xxx * w_xxx or
    # w_xxx*w_xxxx would have 0 coefficients since they have combined
    # degree 6, 7 respectively)"): excludes any monomial whose derivative
    # ORDERS SUM to `>= this value` from the polynomial library entirely
    # (equivalent to, but more efficient than, keeping the term with a
    # permanently-zero coefficient) -- see
    # `ks_latent.models.propagator._polynomial_term_indices`'s docstring
    # for the exact filtering rule and the physical motivation (KS's own
    # true equation has combined order <=4 on every term; a high-combined-
    # order cross term like `w_xxx*w_xxxx` has no obvious physical
    # justification). `None` (default) is unrestricted -- every monomial
    # up to `spectral_poly_degree` is kept, unchanged behavior.
    spectral_poly_max_term_order: int | None = None
    # "spectral_pde"/"spectral_pde_raw", field_kind="polynomial" only
    # (added 2026-09-09, user-directed: "less normalization for the
    # polynomial"): generalizes the per-order normalization exponent from
    # a fixed `n` to `n*spectral_poly_norm_power` -- `1.0` (default) is
    # the original strength (fixes a real NaN blowup, see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s docstring);
    # `<1.0` weakens it (high-order derivative channels keep more raw
    # dynamic range relative to low-order ones before the shared linear
    # layer sees them); `0.0` disables normalization entirely. Weakening
    # this re-introduces real blowup risk -- verify via a direct scale
    # test before trusting a value below `1.0`, not just a short smoke
    # run.
    spectral_poly_norm_power: float = 1.0
    # "spectral_pde"/"spectral_pde_raw", field_kind="polynomial" only
    # (added 2026-09-09, user-directed: "is there a way to regularize or
    # bound the eigenvalues of the differential operator induced by the
    # pde?"): forces the coefficient on the HIGHEST kept even-order linear
    # derivative term to whichever side of zero guarantees the induced
    # linear operator's eigenvalue real part goes to -inf as wavenumber ->
    # inf (bounded/well-posed) regardless of what training does elsewhere
    # -- negative for a leading order divisible by 4 (e.g. order 4,
    # w_xxxx, KS's own sign), POSITIVE for a leading order that is 2 mod 4
    # (e.g. order 2, w_xx, ordinary diffusion) -- see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody.__init__`'s
    # docstring for the full eigenvalue derivation and why the required
    # sign flips with the (ik)^n period-4 cycle. Does NOT suppress
    # instability at low/mid wavenumber (still needed for genuine chaos)
    # -- a targeted fix for runaway high-wavenumber blowup, not another
    # collapse-inducing constraint. OFF by default (False, unchanged
    # behavior).
    spectral_poly_stable_leading: bool = False
    # `spectral_poly_no_constant` (added 2026-09-11, user-directed: "we
    # should just force the constant to be 0 during training"): removes
    # BOTH additive-constant avenues -- the Linear layer's own `bias`
    # (bias=False, no parameter at all) and the `()` term's own weight
    # column (architecturally zeroed every forward pass, so it never
    # contributes and never receives gradient -- same mechanism
    # `spectral_poly_stable_leading`'s own reparametrized column already
    # relies on). Motivated by a direct finding: a closure with
    # `pde_coeff_l1_linear_only` already zeroing every linear term still
    # produced a near-static standalone rollout (its own self-spectrum
    # barely moved over 55 steps), with the constant term absorbing 61%
    # of the coefficient mass -- suspected of anchoring the dynamics near
    # a near-invariant configuration. OFF by default (False, unchanged
    # behavior).
    spectral_poly_no_constant: bool = False
    # `spectral_poly_fixed_linear_terms` (added 2026-09-11, user-directed:
    # "assume the pde always had -w_xx-w_xxxx, we'll just learn the rest
    # of the terms around this"): maps derivative order `n` (for the
    # single-index term `(n,)`, e.g. `{2: -1.0, 4: -1.0}` for w_xx/w_xxxx)
    # to a FIXED, non-learned PHYSICAL coefficient value -- see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s own docstring
    # for the exact mechanism (a plain constant substituted every forward
    # pass, disconnected from the underlying parameter, which therefore
    # never receives gradient and stays frozen). Every OTHER term
    # (including w*w_x and every higher cross/product term) stays fully
    # learned. `None` (default): no terms fixed, unchanged behavior.
    spectral_poly_fixed_linear_terms: dict[int, float] | None = None
    # `spectral_poly_exclude_nonconservative` (added 2026-09-12, user-
    # directed: "please try to exclude every even-combined-order two-
    # factor term from the library"): true KS conserves int(u)dx EXACTLY
    # (every term in -u*u_x-u_xx-u_xxxx is a total x-derivative). For a
    # two-factor product term d_i*d_j, repeated integration by parts shows
    # its spatial mean is EXACTLY zero for any state when i+j is odd
    # (reduces to an exact total derivative, e.g. w_xx*w_xxx), and
    # generically NONZERO when i+j is even (reduces to a perfect square
    # (w^(m))^2, m=(i+j)/2 -- e.g. w_x*w_x, the largest measured violator
    # in Section 164's own fit at -0.043). Architecturally zeros every
    # such even-combined-order length-2 term's raw contribution (see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody._poly_weight`'s
    # docstring for the full derivation) -- an EXACT structural fix,
    # unlike the soft `w_pde_mean_conservation` penalty (Section 165),
    # which needs a heuristically-tuned weight and only matches the
    # constraint on the training batch, not for every state. Requires
    # `spectral_poly_degree<=2` -- raises otherwise (the rule is not
    # derived for length>=3 terms). `False` (default): unchanged behavior.
    spectral_poly_exclude_nonconservative: bool = False
    # `spectral_poly_stable_linear_terms` (added 2026-09-14, Section 174,
    # user-directed): user proposal -- parametrize the propagator as a
    # KS-shaped template (`u_t = -u*u_x + nu*u_xx + ...`) with `nu`
    # LEARNABLE but sign-guaranteed and slow-moving, so boundedness is
    # architectural rather than hoped-for. Maps derivative order `n` (for
    # the single-index term `(n,)`) to an INITIAL physical coefficient
    # (must be `< 0`, e.g. `{2: -1.0, 4: -1.0}` for w_xx/w_xxxx, matching
    # true KS's own dissipation operator at init) -- unlike
    # `spectral_poly_fixed_linear_terms` above, this does NOT freeze the
    # value: the coefficient is reparametrized as `-(raw)**2` (`raw` a
    # genuine, trainable `nn.Parameter`), so it always stays negative but
    # its magnitude keeps learning. See
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s own docstring
    # for the full mechanism and motivation (directly targets the root
    # cause diagnosed across Sections 169-173: nothing ever constrained
    # w_xx's sign, so the linear operator was free to become globally
    # damping at every wavenumber -- the actual fixed-point collapse
    # measured identically in all four of those runs). Combine with
    # `Stage{1,2}TrainingConfig.stable_linear_lr_factor` to give these
    # specific parameters a much smaller learning rate than the rest of
    # the network (the "can't make huge steps in nu" requirement).
    # `None` (default): mechanism off, unchanged behavior.
    spectral_poly_stable_linear_terms: dict[int, float] | None = None
    # `spectral_poly_time_deriv` (added 2026-09-18, Section 189, user-
    # directed: "can we incorporate time derivatives into the pde
    # polynomial? might give us a richer expression. We can approximate
    # them using rollout terms potentially"): appends a finite-difference
    # time-derivative feature, `w_t_fd = (w - w_prev) / spectral_poly_
    # time_deriv_dt_snap`, to the polynomial/Chebyshev library -- see
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s own
    # `poly_time_deriv` docstring for the full mechanism (where `w_prev`
    # comes from at training vs. rollout time, and why `step_one`'s
    # single-argument regularizer call sites stay unaffected). Requires
    # `spectral_field_kind in ('polynomial', 'chebyshev')` -- validated in
    # `_SpectralPDEDeltaBody.__init__`. `False` (default): unchanged
    # behavior (no extra library variable, `step()` still discards
    # `z_prev` for markovian mode exactly as before).
    spectral_poly_time_deriv: bool = False
    spectral_poly_time_deriv_dt_snap: float = 1.0
    # `spectral_poly_time_deriv2` (added 2026-09-18, Section 190, user-
    # directed: "that might be the next thing to try, incorporate u_tt"):
    # appends a SECOND finite-difference feature, `w_tt_fd = (w - 2*w_prev
    # + w_prev2) / spectral_poly_time_deriv_dt_snap**2` (the standard
    # 3-point BACKWARD second difference -- only past states, no future
    # state needed, matching the rollout-terms framing) -- in ADDITION to
    # `spectral_poly_time_deriv`'s own `w_t`, independent flags (either
    # may be on without the other). Needs a genuine `z_prev2` (state two
    # steps back), threaded through `LatentPropagator.step()`/`.rollout()`
    # -- see `ks_latent.models.propagator._SpectralPDEDeltaBody`'s
    # `poly_time_deriv` docstring in `__init__` for the shared mechanism.
    # `False` (default): unchanged behavior.
    spectral_poly_time_deriv2: bool = False
    # `spectral_burgers_nu_init` (added 2026-09-14, Section 175, user-
    # directed -- a revision of their own Section 174 proposal after the
    # "learnable KS template" mechanism above turned out numerically
    # expensive, ~10-15h for a full Stage-2 run, since even a SIGN-
    # guaranteed (not frozen) `-w_xxxx` term is stiff): `spectral_field_
    # kind="forced_burgers"` only. Initial value for `nu` (real,
    # UNCONDITIONALLY stabilizing viscosity -- reparametrized as
    # `softplus(raw)`, always > 0 regardless of how raw moves under
    # training) in `w_t = -w*w_x + nu*w_xx + g_theta(w, w_x, ...)`, where
    # `g_theta` is a learned, shared-pointwise MLP forcing term (the
    # user's own words: "the forcing function should be a learned output
    # from an mlp depending on the latent state"). Default `1.0` matches
    # true KS/Burgers' own unit-scale diffusion coefficient. See
    # `ks_latent.models.propagator._SpectralPDEDeltaBody`'s own docstring
    # for the full mechanism and why dropping the 4th-order term entirely
    # (rather than sign-constraining it, Section 174's approach) is both
    # MORE robust (ordinary positive-coefficient diffusion damps every
    # wavenumber unconditionally -- no low-k/high-k sign cancellation to
    # get right) and far cheaper (removes the stiffest eigenvalue in the
    # operator, ~10x relaxation of the forward-Euler stability
    # constraint).
    spectral_burgers_nu_init: float = 1.0
    # `spectral_burgers_beta_max` (added 2026-09-14, Section 175, same
    # day, user-directed: after the first "forced_burgers" version --
    # hardcoded advection at exactly -1.0 -- diverged to nan by epoch
    # 4/k=6 under a real gradual ramp: "why don't we give -u*u_x a
    # coefficient term too, if that's the cause of the instability".
    # `spectral_field_kind="forced_burgers"` only. The nonlinear advection
    # term has its OWN amplitude-dependent CFL-type explicit-Euler
    # stability constraint, entirely independent of nu -- this cannot be
    # fixed by tuning nu alone. `beta = spectral_burgers_beta_max *
    # tanh(raw_beta)`, a genuine learnable parameter but architecturally
    # bounded to `[-beta_max, +beta_max]` regardless of how raw_beta
    # moves under training (same hard-cap philosophy as `delta_cap`).
    # `raw_beta` starts at exactly `0` (matching `zero_init` -- the
    # propagator starts as pure diffusion+forcing, discovering advection
    # strength gradually via the same low-LR `stable_linear_lr_factor`
    # group `nu` uses). Default `1.0` matches true KS/Burgers' own
    # unit-scale advection coefficient as the outer bound.
    spectral_burgers_beta_max: float = 1.0
    # `spectral_burgers_nonlinear_nu` (sketched 2026-09-14, same section,
    # user-directed follow-up: "should we consider creating a pde that
    # isn't a polynomial? it could be a more generic nonlinear function of
    # the derivatives"). `spectral_field_kind="forced_burgers"` only.
    # Generalizes the CONSTANT `nu` to a genuinely nonlinear, state-
    # dependent diffusion coefficient `nu(w, w_x, ..., w^(max_order))`
    # (analogous to real nonlinear-diffusion PDEs, e.g. the porous medium
    # equation) via its OWN independent small MLP trunk, rather than a
    # bare unconstrained `field_kind="mlp"` correction -- nu(x) =
    # softplus(trunk(x)) stays architecturally POSITIVE AT EVERY POINT
    # regardless of training, the same unconditional guarantee the
    # constant-nu case has, just spatially/state-varying. Zero-init
    # preserving: the trunk's final weight is zeroed but its bias is set
    # so nu(x) == spectral_burgers_nu_init exactly everywhere at
    # construction (identical to the constant-nu case's own init value).
    # The entire nu-trunk goes into `stable_linear_lr_factor`'s slow-LR
    # group (see `_SpectralPDEDeltaBody.stable_linear_raw_parameters`'s
    # docstring for why it has its own trunk rather than sharing one with
    # the forcing MLP). `False` (default): unchanged constant-nu behavior.
    # NOT YET LAUNCHED as a real training run at the time this was
    # written -- implemented and unit-verified (CPU-only: zero-init
    # recovers the constant-nu case exactly, nu(x) stays positive under
    # extreme inputs, gradient reaches the trunk) while Section 175 (the
    # constant-nu version) was still training on MPS.
    spectral_burgers_nonlinear_nu: bool = False
    # `spectral_burgers_kernel_instability` (added 2026-09-14, Section 177,
    # user-directed after reading Sakaguchi, "A Simple Model for
    # Spatio-Temporal Chaos in an Unstable Burgers Equation," Prog. Theor.
    # Phys. 103 (2000) 703). `spectral_field_kind="forced_burgers"` only.
    # Sections 175/176's forced_burgers propagators had NOTHING
    # destabilizing at all -- fitted nu/beta barely moved from their
    # (stabilizing) init values, pure damping, hence their collapse
    # (lambda1=-0.269, the worst of the whole arc). Sakaguchi's own
    # unstable Burgers equation gets its chaos-sustaining instability from
    # a NONLOCAL kernel on w_xx, which in Fourier space is an exactly
    # diagonal multiplier `Lhat(k) = -(A*exp(-k^2/width^2) + nu) * k^2` --
    # REPLACES the diffusion term entirely with this (A=0 reduces to the
    # constant-nu case exactly, so this is a strict generalization, not a
    # different starting point). `A` (learnable, architecturally bounded
    # to `[-spectral_burgers_kernel_A_max, +spectral_burgers_kernel_A_max]`
    # via tanh, any sign, starts at 0) is the missing destabilizing
    # ingredient -- the Gaussian envelope means its magnitude stays
    # BOUNDED as k grows (unlike a k^4 term, which is unconditionally
    # unbounded and forced Section 174's ode_substeps~64), while `nu`
    # (learnable, always > 0 via softplus, same mechanism as the constant
    # case) guarantees UNCONDITIONAL high-wavenumber damping regardless of
    # what A/width do, matching Sakaguchi's own stability requirement
    # exactly (g(k)+nu < 0 for small k, g(k) -> 0 rapidly for large k).
    # See ks_latent.models.propagator._SpectralPDEDeltaBody's own
    # docstring for the full derivation. `False` (default): unchanged
    # constant/nonlinear-nu behavior.
    spectral_burgers_kernel_instability: bool = False
    spectral_burgers_kernel_A_max: float = 1.0
    spectral_burgers_kernel_width_init: float = 1.0
    # `spectral_burgers_forcing_max` (added 2026-09-14, Section 178,
    # user-directed after Sections 176/177 both diverged -- standalone
    # rollout max|z| growing ~linearly, 2.5->51 over 100 steps -- while
    # their physically-constrained terms (nu/beta/kernel A) stayed near
    # their safe init values. `spectral_field_kind="forced_burgers"` only.
    # Unlike nu/beta/A, the forcing MLP had NO structural guarantee at
    # all -- it is never trained on states outside Stage 1's own
    # on-attractor k=2 window, so its behavior off-distribution (exactly
    # where an autoregressive rollout ends up once anything drifts) is
    # unconstrained extrapolation. TWO guarantees now applied to the
    # forcing's raw output, in this order: (1) bounded magnitude via
    # `spectral_burgers_forcing_max*tanh(raw/spectral_burgers_forcing_max)`
    # (same cap philosophy as `delta_cap`/`beta`); (2) the CAPPED result's
    # own spatial mean is then subtracted, giving an EXACTLY zero-mean
    # forcing (architecturally guaranteed regardless of training) --
    # matching true KS's own exact mass conservation, the primary fix for
    # the specific near-uniform drift observed in 176/177 (a systematic,
    # same-sign forcing every step is exactly what produces that kind of
    # steady growth). See ks_latent.models.propagator._SpectralPDEDeltaBody's
    # own docstring for the full derivation, including why the final
    # result is bounded by 2x this value (not exactly this value) once
    # the mean-subtraction is applied. Default `1.0`.
    spectral_burgers_forcing_max: float = 1.0
    # `spectral_burgers_kernel_A_fixed` (added 2026-09-16, Section 180,
    # user-directed ablation: "is there any way to get kernel A to
    # activate in 178 and 179? initialize differently? any ideas?").
    # `spectral_field_kind="forced_burgers"` with
    # `spectral_burgers_kernel_instability=True` only. 178/179 both
    # measured `A` drifting slightly NEGATIVE under gradient descent
    # (toward MORE damping, not less), meaning the local k-step MSE
    # gradient actively opposes activating the kernel's destabilizing
    # capacity -- a warm-start init alone would likely just get walked
    # back to ~0 under the slow-LR group over many epochs. This is the
    # clean causal test instead: when set (not None), `A` is FIXED at
    # this exact value (a buffer, not a learnable parameter, no tanh
    # reparametrization -- there is no training dynamic to guard against
    # when it can't move), architecturally forcing a genuine low-k
    # unstable band regardless of what gradient descent would have
    # chosen. Isolates whether the forcing MLP simply compensates for a
    # forced-unstable `A` (reproducing the same frozen-band collapse,
    # meaning the kernel was never the bottleneck) or whether real chaos
    # emerges once the model can no longer avoid a destabilizing linear
    # term. `None` (default): unchanged learnable-A behavior.
    spectral_burgers_kernel_A_fixed: float | None = None
    # `spectral_burgers_beta_fixed` (added 2026-09-16, Section 181,
    # user-directed follow-up to the kernel-A ablation: "if we remove
    # delta_cap what would happen?" -> [explained: real KS-class
    # saturation comes from the advection term cascading energy from
    # unstable low-k modes to damped high-k modes; beta has ALSO stayed
    # near 0 in every one of 175-180's fitted values, so with only A
    # forced unstable, delta_cap was the ONLY thing bounding growth,
    # producing an artificial freeze at a larger scale instead of genuine
    # nonlinear saturation] -> "sure try that"). Exactly analogous to
    # `spectral_burgers_kernel_A_fixed`: when set, `beta` (the advection
    # coefficient) is a FIXED buffer, not a learnable parameter -- forces
    # the nonlinear advection term to genuinely engage regardless of what
    # gradient descent would have chosen. Tests whether real
    # energy-cascade saturation (bounded chaos at the correct physical
    # scale) emerges once BOTH the destabilizing (A) and the
    # saturating-nonlinearity (beta) mechanisms are active
    # simultaneously, rather than delta_cap alone doing all the
    # (artificial) bounding work. `None` (default): unchanged learnable-
    # beta behavior.
    spectral_burgers_beta_fixed: float | None = None
    # `spectral_burgers_kernel_mu_init` (added 2026-09-16, Section 182,
    # user-directed: "let's keep the sakaguchi setup, add a
    # hyperviscosity term and get rid of the forcing mlp term"). Section
    # 181 (kernel A AND beta both forced active) made things WORSE, not
    # better (RMSE climbing monotonically to ~32 over 200 steps, never
    # saturating) -- hypothesis: real KS-class saturation needs 4th-order
    # hyperviscosity to absorb energy the advection term cascades to high
    # k (the "forced_burgers" family, Section 175 onward, only ever had
    # 2nd-order diffusion). Extends Lhat(k) with a genuine `-mu*k^4` term,
    # `mu=softplus(raw)` always > 0 (real damping, never anti-damping),
    # matching true KS's own `-w_xxxx`. `None` (default): term absent
    # entirely, unchanged behavior. See
    # ks_latent.models.propagator._SpectralPDEDeltaBody's docstring for
    # the default init value's derivation (chosen to respect this
    # backbone's ode_substeps=1 forward-Euler stability budget).
    spectral_burgers_kernel_mu_init: float | None = None
    # `spectral_burgers_no_forcing` (added 2026-09-16, Section 182, same
    # user direction). Removes the forcing MLP term entirely (no
    # input_proj/blocks/final_ln/output_proj built at all, field()
    # returns a plain zero for it) -- leaving ONLY the hardcoded/
    # constrained physical terms (advection, diffusion, kernel
    # instability, hyperviscosity), matching Sakaguchi's own equation (1)
    # structurally (no separate learned correction term at all). Tests
    # whether the forcing MLP (still substantial in magnitude even
    # mean-zero, ~0.7-1.0 in Sections 178-181) was itself contributing to
    # or masking the dynamics, rather than the pure physical terms alone
    # being incapable of bounded chaos. `False` (default): unchanged
    # forcing-MLP behavior.
    spectral_burgers_no_forcing: bool = False
    # `spectral_burgers_kernel_A_init`/`spectral_burgers_beta_init` (added
    # 2026-09-16, Section 183, user-directed: "let's initialize with the
    # same parameters but let A and beta be trainable"). Only used when
    # `spectral_burgers_kernel_A_fixed`/`spectral_burgers_beta_fixed` are
    # `None` (learnable case) -- sets the LEARNABLE parameter's own
    # starting point via inverse-tanh (instead of always starting at 0),
    # e.g. `-2.0`/`-1.0` to start from the same genuinely-destabilizing
    # values Sections 180-182 hard-FIXED, but now letting gradient
    # descent adjust them freely from there. Requires
    # `|spectral_burgers_kernel_A_init| < spectral_burgers_kernel_A_max`
    # (respectively beta) strictly, since atanh diverges at +-1 -- e.g.
    # raise spectral_burgers_kernel_A_max to 3.0 for an A_init of -2.0.
    # `0.0` (default): unchanged zero-init behavior.
    spectral_burgers_kernel_A_init: float = 0.0
    spectral_burgers_beta_init: float = 0.0
    # "masked_mlp_expand" backbone only (added 2026-09-10, Section 128,
    # user-directed: "in the middle layer of the mlp, expand the dimension
    # 3x"). Sizes that middle (hidden) layer's width as `expand_factor *
    # d_latent` -- see `ks_latent.models.propagator._MaskedMLPExpandDeltaBody`'s
    # docstring for the full architecture and the exact derivation of why
    # this, combined with `attn_window` (reused as the layer's circular-band
    # window, referenced in `d_latent`-ring units throughout, same
    # convention as `masked_mlp_wide`), gives an effective one-sided
    # neighbor radius of `attn_window * expand_factor` inside the widened
    # middle layer. Default `3` matches the request this backbone was built
    # for exactly.
    masked_mlp_expand_factor: int = 3

    def __post_init__(self):
        _validate_propagator_mode_backbone(self.mode, self.backbone)
        if self.mode == "history" and self.n_history < 2:
            raise ValueError(f"mode='history' requires n_history >= 2, got {self.n_history!r}")
        if (
            self.backbone in ("transformer", "vit", "fno_vit", "fno_mlp", "local_mlp")
            and self.d_latent % self.n_tokens != 0
        ):
            raise ValueError(
                f"d_latent={self.d_latent} must be divisible by n_tokens={self.n_tokens} "
                f"for the {self.backbone!r} backbone"
            )
        if self.pos_encoding not in ("circular", "linear"):
            raise ValueError(f"pos_encoding must be 'circular' or 'linear', got {self.pos_encoding!r}")
        if self.delta_cap is not None and self.delta_cap <= 0:
            raise ValueError(f"delta_cap must be > 0, got {self.delta_cap!r}")
        if self.token_window is not None and self.token_window < self.d_latent // self.n_tokens:
            raise ValueError(
                f"token_window={self.token_window} must be >= chunk_size="
                f"{self.d_latent // self.n_tokens} (= d_latent // n_tokens)"
            )
        if self.backbone in ("fno_vit", "fno_mlp") and self.fno_n_layers < 1:
            raise ValueError(f"fno_n_layers must be >= 1 for backbone={self.backbone!r}, got {self.fno_n_layers!r}")
        if self.fno_modes is not None and self.fno_modes < 1:
            raise ValueError(f"fno_modes must be >= 1, got {self.fno_modes!r}")
        if self.backbone == "local_mlp" and self.attn_window is None:
            raise ValueError(
                "backbone='local_mlp' requires attn_window to be set (its local-conv kernel "
                "radius) -- there is no 'global local_mlp'; use 'mlp' or 'fno_vit' for that."
            )
        if self.backbone == "node" and self.attn_window is None:
            raise ValueError(
                "backbone='node' requires attn_window to be set (its local vector field's "
                "conv kernel radius) -- there is no 'global node'; use 'mlp' or 'fno_vit' for that."
            )
        if self.backbone == "masked_mlp_wide" and self.attn_window is None:
            raise ValueError(
                "backbone='masked_mlp_wide' requires attn_window to be set (its "
                "MaskedLinearRect band window) -- there is no 'global masked_mlp_wide'; "
                "use 'mlp' for that."
            )
        if self.backbone == "masked_mlp_expand" and self.attn_window is None:
            raise ValueError(
                "backbone='masked_mlp_expand' requires attn_window to be set (its "
                "MaskedLinearRect band window, shared by all three layers) -- there is no "
                "'global masked_mlp_expand'; use 'mlp' for that."
            )
        if self.backbone == "masked_mlp_expand" and self.masked_mlp_expand_factor < 1:
            raise ValueError(
                f"masked_mlp_expand_factor must be >= 1, got {self.masked_mlp_expand_factor!r}"
            )
        if self.backbone == "node" and self.ode_substeps < 1:
            raise ValueError(f"ode_substeps must be >= 1, got {self.ode_substeps!r}")
        if self.backbone == "cnn" and self.cnn_kernel_size < 1:
            raise ValueError(f"cnn_kernel_size must be >= 1, got {self.cnn_kernel_size!r}")
        if self.backbone == "spectral_pde":
            if self.mode != "markovian":
                raise ValueError("backbone='spectral_pde' requires mode='markovian'")
            if self.spectral_K is None or self.spectral_N_w is None:
                raise ValueError(
                    "backbone='spectral_pde' requires spectral_K and spectral_N_w to be set"
                )
            if self.d_latent != 2 * self.spectral_K:
                raise ValueError(
                    f"backbone='spectral_pde' requires d_latent == 2*spectral_K "
                    f"(got d_latent={self.d_latent!r}, spectral_K={self.spectral_K!r})"
                )
            n_freq = self.spectral_N_w // 2 + 1
            if self.spectral_K < 1 or self.spectral_K > n_freq:
                raise ValueError(
                    f"spectral_K={self.spectral_K!r} must be in [1, spectral_N_w//2+1={n_freq!r}]"
                )
            if self.spectral_L <= 0:
                raise ValueError(f"spectral_L must be > 0, got {self.spectral_L!r}")
            if self.spectral_max_order < 0:
                raise ValueError(f"spectral_max_order must be >= 0, got {self.spectral_max_order!r}")
            if self.spectral_integrator not in ("euler", "rk4", "etdrk4"):
                raise ValueError(
                    f"spectral_integrator must be 'euler', 'rk4', or 'etdrk4', "
                    f"got {self.spectral_integrator!r}"
                )
            _validate_spectral_field_kind(
                self.spectral_field_kind, self.spectral_poly_degree, self.spectral_poly_max_term_order,
                self.spectral_poly_norm_power,
            )
            if self.spectral_physics_prior and self.spectral_max_order < 4:
                raise ValueError(
                    "spectral_physics_prior requires spectral_max_order >= 4 "
                    "(needs w_x for the nonlinear term and w_xx/w_xxxx for the linear "
                    "one, except under 'etdrk4' where the linear part is handled "
                    "separately -- max_order>=4 is required uniformly regardless, for "
                    "a consistent feature set across integrators)"
                )
        if self.backbone == "spectral_pde_raw":
            if self.mode != "markovian":
                raise ValueError("backbone='spectral_pde_raw' requires mode='markovian'")
            # See _SpectralPDERawDeltaBody's docstring: works on ANY
            # encoder's raw z (no d_latent==2*spectral_K constraint --
            # spectral_K here is purely an internal self-FFT truncation
            # parameter, independent of d_latent).
            if self.spectral_K is None:
                raise ValueError("backbone='spectral_pde_raw' requires spectral_K to be set")
            n_freq = self.d_latent // 2 + 1
            if self.spectral_K < 1 or self.spectral_K > n_freq:
                raise ValueError(
                    f"spectral_K={self.spectral_K!r} must be in [1, d_latent//2+1={n_freq!r}] "
                    f"(backbone='spectral_pde_raw' treats d_latent={self.d_latent!r} as the "
                    f"field length for its own self-FFT)"
                )
            if self.spectral_L <= 0:
                raise ValueError(f"spectral_L must be > 0, got {self.spectral_L!r}")
            if self.spectral_max_order < 0:
                raise ValueError(f"spectral_max_order must be >= 0, got {self.spectral_max_order!r}")
            if self.spectral_integrator not in ("euler", "rk4", "etdrk4"):
                raise ValueError(
                    f"spectral_integrator must be 'euler', 'rk4', or 'etdrk4', "
                    f"got {self.spectral_integrator!r}"
                )
            _validate_spectral_field_kind(
                self.spectral_field_kind, self.spectral_poly_degree, self.spectral_poly_max_term_order,
                self.spectral_poly_norm_power,
            )
            if self.spectral_physics_prior:
                raise ValueError(
                    "spectral_physics_prior is not defined for backbone='spectral_pde_raw' -- "
                    "there is no 'true governing equation' for an arbitrary, LEARNED latent "
                    "ordering the way there is for a genuine w=irfft(z) physical field "
                    "(backbone='spectral_pde' only)."
                )


@dataclass(frozen=True)
class EnsemblePropagatorConfig:
    """Ensemble propagator (user-directed 2026-08-30, "Solution 1" -- see
    docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 4): `n_members`
    independently-initialized propagators sharing one architecture
    (`member`), combined into a single step/rollout via a (optionally
    learned, `z`-dependent) weighted mean.

    Motivation: plain MSE training of a single deterministic propagator on a
    bounded rollout horizon is optimized by collapsing to the conditional-
    mean fixed point (see `PropagatorConfig.delta_cap`'s docstring for the
    full argument). User's proposed fix: inject independent, diagonal
    Gaussian noise into each member's input at every rollout step (variance
    proportional to that latent dimension's OWN variance -- no off-diagonal/
    cross-dimension covariance), so each member sees a slightly different
    local view of the same state; the ensemble's output is the (possibly
    learned) weighted mean of their predictions.

    Each member computes `z_next_m = z + capped_delta_m(body_m(z + eps_m))`
    -- note the noise perturbs only the INPUT to the residual body, not the
    residual's own additive base. This is a deliberate implementation
    choice (not part of the user's literal spec) needed to preserve
    `zero_init`'s identity-at-init property exactly regardless of
    `noise_std_frac`: every member's delta is exactly zero at init (its
    output head is zero-initialized, same as every other propagator in this
    codebase), so `z_next_m == z` at init even though `eps_m != 0` --
    without this, injecting noise directly into the additive base would
    make even a freshly-initialized ensemble non-identity, breaking the
    invariant `test_propagator_is_identity_at_init` checks for every other
    propagator here.

    `latent_var` (per-dimension `Var(z_i)`, shape `(d_latent,)`) is NOT part
    of this config (configs must stay plain/hashable for provenance -- see
    this module's docstring) -- it is data, not a hyperparameter, computed
    from the encoded training set and passed to `EnsemblePropagator.
    set_latent_var(...)` after construction. Defaults to all-ones (isotropic
    unit noise) until set, which is almost certainly NOT the latent's true
    (non-unit, generally anisotropic) per-dimension scale.

    `member.mode` must be `"markovian"` or `"history"` -- noise-injection
    semantics for `"two_step"` are not implemented (that mode is legacy;
    see `PropagatorConfig`'s docstring).
    """

    member: PropagatorConfig = PropagatorConfig()
    n_members: int = 5
    noise_std_frac: float = 0.05
    learned_weights: bool = True
    gate_hidden: int = 32

    @property
    def d_latent(self) -> int:
        """Proxies `member.d_latent` -- every Gate 3/4 consumer (Lyapunov,
        D3 coupling, DA cycling) reads `propagator.cfg.d_latent` directly,
        so `EnsemblePropagator` is a drop-in replacement without touching
        any of that code."""
        return self.member.d_latent

    @property
    def n_history(self) -> int:
        """Proxies `member.n_history` -- see `d_latent`'s docstring; same
        drop-in-compatibility reasoning for `train_stage2`'s loop and every
        Gate 3/4 script's `mode == "history"` branch."""
        return self.member.n_history

    def __post_init__(self):
        if self.n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {self.n_members!r}")
        if self.noise_std_frac < 0:
            raise ValueError(f"noise_std_frac must be >= 0, got {self.noise_std_frac!r}")
        if self.member.mode not in ("markovian", "history"):
            raise ValueError(
                f"EnsemblePropagatorConfig.member.mode must be 'markovian' or 'history', "
                f"got {self.member.mode!r} -- noise-injection semantics for 'two_step' "
                f"are not implemented."
            )


@dataclass(frozen=True)
class Stage1TrainingConfig:
    """Brief §5.1: AdamW lr=1e-3, cosine, 60 epochs, batch 128, grad clip 1.0,
    loss = 1.0*recon + 0.5*pred + 0.01*decorr + 0.01*var.

    `weight_decay`/`warmup_epochs`/`lr_min_factor`/`noise_std` added
    2026-08-29, ported from a reference implementation found at
    `/Users/daltonjones/Documents/experiments/ks_latent/` (see
    CLAUDE_CODE_BRIEF.md §5.1 "Ported improvements" addendum for the full
    rationale and docs/RESULTS.md for what was and wasn't adopted):

    - `weight_decay` fixes a real latent bug: passing no `weight_decay` to
      `torch.optim.AdamW` silently uses PyTorch's default of 0.01, which is
      1000x the reference project's deliberately-chosen 1e-5 and decays
      *every* parameter every step -- including the zero-initialized
      propagator output head that the identity-at-init property depends on,
      and every LayerNorm gain/bias. Defaulting it here makes the choice
      explicit and matches the reference's measured-good value.
    - `warmup_epochs`/`lr_min_factor`: linear LR warmup before the cosine
      decay, floored at `lr_min_factor * lr` rather than decaying to exactly
      zero. Plain `CosineAnnealingLR` from epoch 0 exposes a freshly
      initialized (non-identity) encoder/decoder to the full peak LR
      immediately; a short warmup is cheap insurance against that.
    - `noise_std`: Gaussian noise added to the *encoder input* on the
      reconstruction path only (`L_recon = MSE(D(E(x+eps)), x)`), target and
      prediction path unaffected. Not a generic regularizer -- it is
      motivated by this project's own DA use case (Phase 5's PFF filter):
      `E`/`D` are otherwise only ever shown states exactly on the attractor,
      but every state a filter hands them (an analysis state, an ensemble
      member) is an off-attractor perturbation of one. Default 0.0
      (off, matches the brief's original recipe exactly); nonzero values are
      an experiment to run, not yet validated as a canonical-recipe change.
    - `k_pred_max`/`k_pred_warmup_epochs` (added 2026-08-29, ported from
      the same reference project's multi-step/unrolled training): `0`
      (default) preserves the brief's original fixed-`k_pred` behavior.
      When `k_pred_max > k_pred`, the rollout length used for `L_pred` is
      ramped linearly from `k_pred` up to `k_pred_max` over
      `k_pred_warmup_epochs` epochs (same curriculum shape as Stage 2's
      existing K-curriculum), instead of jumping straight to the long
      rollout -- starting at the full length is unstable (an
      undertrained propagator compounds its own error and the gradient
      through the chain can explode before the one-step map is any good).
      This is meant to be paired with a *larger* `aux` propagator (sized
      like the real Stage-2 `PropagatorConfig`, not the brief's tiny
      ~22k-param auxiliary one) -- a small aux propagator has no spare
      capacity to fit an 8-step rollout well. See
      CLAUDE_CODE_BRIEF.md §5.1/5.2 addendum for the motivating comparison
      and `scripts/train_stage1_patched.py --multistep`.
    """

    lr: float = 1e-3
    epochs: int = 60
    batch_size: int = 128
    grad_clip: float = 1.0
    w_recon: float = 1.0
    w_pred: float = 0.5
    w_decorr: float = 0.01
    w_var: float = 0.01
    # `spatial_coherence_loss` (added 2026-08-31, user-directed: "turn the
    # D7 diagnostic into a loss ... push the model towards creating
    # spatial coherence"). OFF BY DEFAULT (0.0) -- read
    # `ks_latent.training.losses.spatial_coherence_loss`'s docstring before
    # setting this above 0: `RegConfig.lambda_z`, a closely related
    # mechanism, is documented as collapsing the latent in the reference
    # project even with an explicit anti-collapse counter-term, and this
    # carries a plausibly similar (though mechanistically different) risk.
    # Test at a small weight with full Gate 3/4 monitoring before trusting
    # any result.
    w_spatial: float = 0.0
    spatial_bandwidth: float = 3.0
    # Added 2026-09-01, user-directed (backs D8 -- see
    # ks_latent.analysis.diagnostics.signed_bandedness and
    # spatial_coherence_loss's "signed" docstring section): False (default)
    # rewards coupling regardless of sign (D7's statistic); True rewards
    # SAME-SIGN local coherence specifically. Only meaningful with
    # w_spatial > 0.
    spatial_signed: bool = False
    # `w_jacobian_bandedness` (added 2026-09-24, Section 214, user-
    # directed: "let's make D3 into a loss, since this seems like it's
    # the most important statistic to improve our chances at being able
    # to localize as in 4.3 in the literature review document"). D3's
    # own differentiable analogue -- see `ks_latent.training.losses.
    # propagator_jacobian_bandedness_loss`'s docstring for the full
    # mechanism and why D3 (dynamical Jacobian coupling) matters MORE
    # than D7/w_spatial (same-time encoder coherence) for Part 4.3's
    # Gaspari-Cohn localization proposal specifically: a taper on the
    # latent's SITE structure only makes physical sense if the
    # PROPAGATOR also respects that locality, not just the encoder's
    # instantaneous representation. `mode="markovian"` only (same
    # restriction as `w_spectrum_shape`); evaluated on `aux.step_one`
    # directly (no rollout), same once-per-epoch/outside-autocast/
    # expensive-Jacobian convention as `w_spectrum_shape`. OFF by
    # default (0.0) -- genuinely new, untested at scale; test with full
    # Gate 3/4 monitoring (and, critically for this use case, a real
    # standalone Lyapunov check -- see docs/OPEN_QUESTIONS.md/RESULTS.md
    # for why short-horizon training diagnostics alone have repeatedly
    # missed real long-rollout instability in this project) before
    # trusting any result from this.
    w_jacobian_bandedness: float = 0.0
    jacobian_bandedness_bandwidth: float = 3.0
    jacobian_bandedness_n_samples: int = 32
    # Two anti-collapse regularizers (added 2026-08-31, user-directed --
    # Phase 2 architecture doc Section 35/38, options 3a/3b): both OFF by
    # default (0.0). `ks_latent.training.losses.variance_floor_loss`
    # (3a, VICReg-style per-channel std floor) and `.logdet_barrier_loss`
    # (3b, full covariance log-det/eigenvalue barrier) target the same
    # observed failure the existing `w_var`/`decorr_var_loss` doesn't fully
    # prevent: one latent channel's variance collapsing many orders of
    # magnitude below the rest over extended joint training. Unlike
    # `w_var` (pulls every channel toward variance exactly 1, fighting
    # naturally-strong channels too), both of these only activate against
    # actual collapse -- see each loss function's own docstring for the
    # mechanism and the 3a-vs-3b tradeoff.
    w_var_floor: float = 0.0
    var_floor_gamma: float = 0.1
    w_logdet: float = 0.0
    logdet_eps: float = 1e-3
    # `w_logdet_physical` (added 2026-09-09, user-directed, after Sections
    # 112/113's polynomial-field_kind propagator both collapsed (Gate 3:
    # D_KY=0.0): "add some w_logdet too to the w states (irfft of latent
    # frequency state z), I think that may prevent collapse as well" --
    # then clarified this means STAGE 1, on the encoder's own z (not
    # Stage 2's rolled-out z_pred, see Stage2TrainingConfig.w_logdet_rollout
    # for that separate variant). Same `logdet_barrier_loss` mechanism as
    # `w_logdet` above, but applied to `decode_from_spectrum(z, K, N_w)` --
    # the exact irfft reconstruction of the PHYSICAL field `w` -- instead
    # of `z` (the rFFT coefficients) directly. `w_logdet` already keeps the
    # spectral-coefficient covariance full-rank; this is a distinct
    # constraint in physical space (every grid point's value across the
    # batch), which could catch a collapse mode that's degenerate in
    # physical space but not (yet) in the coefficient covariance, or simply
    # give the encoder a more directly physically-interpretable pressure
    # against every state converging to the same field shape. Only
    # meaningful when `ae.cfg` carries `K`/`N_w` attributes (i.e.
    # `encoder_kind="spectral_field"`), duck-typed the same way `w_lowpass`
    # reads `ae.cfg.K`/`.L`. OFF by default (0.0).
    w_logdet_physical: float = 0.0
    logdet_physical_eps: float = 1e-3
    # `w_logdet_physical_rollout` (added 2026-09-09, user-directed, after
    # Section 114 -- w_logdet_physical alone, encoder-side only -- showed
    # the same frozen/not-growing-with-horizon val_kmax_mse signature as
    # every other collapsed run: "the way I see it w_logdet_physical should
    # apply to the encoder in stage 1, AND the rollout from the propagator
    # in stage 1, AND also to the rollout in stage 2"). Stage-1 analogue of
    # `w_lowpass`/`w_lowpass_rollout`'s own pairing: `w_logdet_physical`
    # above targets the ENCODER's own `z` (real data snapshots);
    # `w_logdet_physical_rollout` targets the JOINT-TRAINED aux propagator's
    # own short rolled-out `z_pred` instead (already computed every batch
    # whenever `--full-propagator` is active, same convention as
    # `w_lowpass_rollout`), mapped to physical space via
    # `decode_from_spectrum` before the log-det barrier is applied -- so
    # the propagator's own rollout is pressured, from Phase 1 onward, not
    # to collapse the field's physical shape across a training batch,
    # exactly the collapse axis `w_logdet_physical` alone cannot reach
    # (that term only ever sees real encoder states, never touches the
    # propagator's own free predictions). See also
    # `Stage2TrainingConfig.w_logdet_rollout` for the Stage-2 (post-Phase-1)
    # counterpart. OFF by default (0.0).
    w_logdet_physical_rollout: float = 0.0
    logdet_physical_rollout_eps: float = 1e-3
    # `w_var_physical` (added 2026-09-09, user-directed: "add a w_var term
    # for the irfft of the latent state z"): `decorr_var_loss`'s per-
    # channel variance-vs-target term (`w_var`'s own mechanism, encoder's
    # `z` in spectral-coefficient space), applied instead to
    # `decode_from_spectrum(z)` -- the encoder's own physical-space field
    # reconstruction -- same "physical" scoping convention as
    # `w_logdet_physical`. Targets a DIFFERENT collapse signature than
    # `w_logdet_physical`: marginal per-physical-point variance floor
    # (VICReg-style), rather than the full-covariance-rank barrier --
    # cheaper to compute and optimize, but (per `variance_floor_loss`'s own
    # documented limitation) structurally blind to a correlation-driven
    # collapse that keeps marginal variance up while the joint covariance
    # still loses rank. `var_target` fixed at `1.0` (uniform, matching
    # `w_var`'s own default) -- no adaptive per-point target computed from
    # data for this first pass. OFF by default (0.0).
    w_var_physical: float = 0.0
    # `physics_prior_correction_warmup_epochs` (added 2026-09-09,
    # user-directed: "how do we preserve the chaotic structure and nudge
    # it in the direction we want? could we progressively add systems we
    # know are chaotic? ... sort of a dynamic system gradient descent" --
    # backbone="spectral_pde" with spectral_physics_prior=True only):
    # linearly ramps `aux.body.correction_scale` from `0.0` (epoch 0) to
    # `1.0` (this many epochs in, then held at 1.0 for the rest of
    # training) -- a homotopy/continuation schedule between "exactly the
    # true KS equation, no learned contribution at all" and "the full
    # learned correction is active". Motivated by a real, observed failure
    # mode the same day: a physics_prior propagator that starts genuinely
    # chaotic (zero_init) visibly DAMPED that chaos within the first ~20
    # of 200 epochs -- diagnosed as short-horizon MSE prediction loss
    # rewarding contraction whenever the encoder's z is imprecise (always
    # true early in joint training), since a damped system's errors shrink
    # regardless of input quality while a genuinely chaotic one's grow
    # regardless of model quality. Ramping the correction in slowly gives
    # the encoder a long runway to converge to an accurate representation
    # UNDER real (untouched) chaotic dynamics before the learned
    # correction gets enough room to find that damping shortcut. Default
    # `0` (off, unchanged behavior -- correction_scale stays at its own
    # default `1.0` throughout).
    physics_prior_correction_warmup_epochs: int = 0
    # `w_pred_warmup_epochs` (added 2026-09-09, user-directed: "why don't
    # we just have a schedule that slowly ramps up w_pred loss" -- the
    # same homotopy idea as `physics_prior_correction_warmup_epochs`,
    # applied one level up: instead of (or alongside) delaying the
    # LEARNED CORRECTION's own growth, delay the PREDICTION loss itself,
    # so the encoder trains on pure reconstruction first and only
    # gradually comes under pressure from `w_pred`'s pointwise MSE term --
    # the mechanism diagnosed as rewarding contraction whenever z is still
    # imprecise. Linearly ramps the EFFECTIVE `w_pred` weight from `0.0`
    # (epoch 0) to `cfg.w_pred` (this many epochs in, then held there) --
    # `cfg.w_pred` itself is unchanged; this only affects the schedule.
    # Default `0` (off, unchanged behavior -- w_pred is active at its full
    # value from epoch 0, exactly as before).
    w_pred_warmup_epochs: int = 0
    # `w_temporal_floor_z`/`w_temporal_floor_w` (added 2026-09-09,
    # user-directed: "build and test that, but design it for w space and
    # for z space where w = irfft(z). I want to test both" -- after
    # Section 124's diagnostic showed the collapse is an ENCODER
    # phenomenon (a propagator that is LITERALLY the exact analytic KS
    # equation, zero learned dynamics at all, still collapsed once the
    # encoder trained under real w_pred pressure), and Section 125's
    # `w_logdet_physical` -- a BATCH-level, cross-snapshot covariance-rank
    # check -- did NOT fix it (a batch mixing many unrelated snapshots can
    # show full aggregate diversity while still flattening any SPECIFIC
    # real trajectory segment, a different, more local granularity of
    # collapse)): a direct, TEMPORAL-STRUCTURE-SPECIFIC fix -- a one-sided
    # floor (see `ks_latent.training.losses.temporal_expansion_floor_loss`)
    # on how close together, in representation space, real states
    # `temporal_floor_lag` real steps apart are allowed to become. The
    # reference floor is computed ONCE from real ground-truth data via the
    # SAME fixed transform used everywhere else (`reference_temporal_
    # separation`), entirely independent of the current encoder.
    # `w_temporal_floor_z` applies this in `z`-space (the rFFT
    # coefficients directly); `w_temporal_floor_w` applies the identical
    # mechanism in `w`-space (`decode_from_spectrum(z)`, the physical
    # field) -- the same z-vs-physical scoping distinction this project
    # draws elsewhere (`w_logdet` vs `w_logdet_physical`, `w_var` vs
    # `w_var_physical`). Either or both may be nonzero; both OFF by
    # default (0.0).
    w_temporal_floor_z: float = 0.0
    w_temporal_floor_w: float = 0.0
    # Shared by both of the above -- how many real dt_snap steps apart the
    # compared states are. Default `1` (consecutive real snapshots).
    temporal_floor_lag: int = 1
    # `w_local_expansion_floor` (added 2026-09-09, user-directed, after a
    # decisive question -- "we could try a rollout variant, but it's the
    # propagator already collapsing in stage 1? so it's sort of
    # irrelevant?" -- clarified what's actually worth testing: with
    # `physics_prior`'s learned correction pinned near zero, the
    # propagator's OWN step has no free parameters to adjust, but the
    # ENCODER still chooses WHERE in z-space real states land, and a fixed
    # nonlinear map can be locally expansive in one region of phase space
    # and contractive in another): a DIFFERENTIABLE, one-sided floor
    # (`ks_latent.training.losses.propagator_local_expansion_floor_loss`)
    # on the propagator's own per-sample step-Jacobian spectral norm
    # (largest singular value), evaluated at real encoded states -- the
    # exact same quantity Gate 4's own D9 diagnostic
    # (`propagator_step_jacobian_spectral_norms`, `prop_jacobian_med`)
    # measures post-hoc, turned into a training-time regularizer via
    # `torch.func.vmap(jacrev(...))`. This is the first regularizer in the
    # whole investigation that actually queries the DYNAMICS (not just
    # real-data separation or batch-level covariance rank) -- it directly
    # rewards the encoder for placing real states where the dynamics
    # (learned or not) expand rather than contract, the literal definition
    # of a positive local Lyapunov exponent. `floor` defaults to `1.0`
    # (the absolute expand-vs-contract boundary, no real-data calibration
    # needed). EXPENSIVE (~0.5-1s per call at K=24, measured directly) --
    # applied only ONCE PER EPOCH (on a small subsample of that epoch's
    # first batch), not every batch, to keep the added cost reasonable.
    # `mode="markovian"` only (matches the underlying diagnostic's own
    # restriction). OFF by default (0.0).
    w_local_expansion_floor: float = 0.0
    local_expansion_floor_value: float = 1.0
    local_expansion_floor_n_samples: int = 32
    # `w_spectrum_shape` (added 2026-09-10, Section 131, user-directed: "can
    # we use the regularizer to force some singular vectors to have
    # expansive values around 1.5 and others to have contracting values?"):
    # a direct fix for `w_local_expansion_floor`'s own diagnosed failure
    # mode (see `ks_latent.training.losses.propagator_local_expansion_floor_loss`'s
    # docstring) -- constraining only the TOP singular value let the other
    # `d-1` collapse toward zero (measured directly on a real Section 130
    # checkpoint: only 5/44 singular values >= 1.0, per-step volume-change
    # factor 8.9e-5). `propagator_spectrum_shape_loss` instead floors the
    # top `spectrum_shape_n_expand` singular values toward
    # `spectrum_shape_expand_target` (expansive) AND floors the remaining
    # `d - spectrum_shape_n_expand` toward `spectrum_shape_contract_floor`
    # (a lower floor that still permits net contraction -- KS is
    # dissipative -- but prevents runaway collapse of the tail). Same
    # EXPENSIVE/once-per-epoch/mode="markovian"-only convention as
    # `w_local_expansion_floor` (shares its underlying Jacobian computation
    # via `_propagator_step_jacobian_singular_values`). OFF by default
    # (0.0).
    w_spectrum_shape: float = 0.0
    # Defaults to this project's own established `L=100` replication target
    # for the number of positive Lyapunov exponents (`CLAUDE.md` §18:
    # "Positive latent exponents: 11, +-2") -- an attempt to shape the
    # LEARNED propagator's Jacobian spectrum toward the shape the TRUE
    # attractor's own Lyapunov spectrum is already known to have, not an
    # arbitrary split.
    spectrum_shape_n_expand: int = 11
    spectrum_shape_expand_target: float = 1.5
    spectrum_shape_contract_floor: float = 0.7
    spectrum_shape_n_samples: int = 32
    # `w_spectrum_shape_graded` (added 2026-09-10, Section 134, user-
    # directed: "keep target higher? or increase the weight" ->
    # investigation showed the real spectrum is a smooth graded decline
    # across all d ranks, not two flat groups (see
    # `ks_latent.training.losses.propagator_graded_spectrum_shape_loss`'s
    # docstring for the full per-rank measurement) -- a per-rank floor
    # generalizing `w_spectrum_shape`'s two-group one. `spectrum_shape_
    # graded_reference_path` must point to a `.npy` file of shape
    # `(d_latent,)`, descending -- typically produced by
    # `scripts/compute_reference_spectrum.py` from a trusted checkpoint's
    # own real spectrum. Loaded ONCE, not recomputed per epoch. Same
    # expensive/once-per-epoch/mode="markovian"-only convention as
    # `w_spectrum_shape` (shares its underlying Jacobian computation). OFF
    # by default (0.0); `w_spectrum_shape` and `w_spectrum_shape_graded`
    # may be used independently or together (not mutually exclusive).
    w_spectrum_shape_graded: float = 0.0
    spectrum_shape_graded_reference_path: str | None = None
    spectrum_shape_graded_n_samples: int = 32
    # `spectrum_shape_two_sided` (added 2026-09-18, Section 186, user-
    # directed: "I think we should try the two sided approach"). Section
    # 185's own trained checkpoint measured top singular values 3-4x ABOVE
    # `spectrum_shape_expand_target`, so `w_spectrum_shape`'s one-sided
    # floor (`relu(target - sv)^2`) was already fully satisfied there and
    # contributed zero corrective gradient for the entire run -- see
    # `ks_latent.training.losses.propagator_spectrum_shape_loss`'s
    # docstring. Setting this True switches BOTH groups to a plain
    # squared-error match (`(sv - target)^2`), which penalizes exceeding
    # the target as much as falling short -- the only way this mechanism
    # can pull an already-excessive singular value back down. Applies to
    # `w_spectrum_shape` only (not `w_spectrum_shape_graded`, which is a
    # separate, off-by-default mechanism). OFF by default (False), matching
    # this module's existing one-sided default.
    spectrum_shape_two_sided: bool = False
    # `w_spectrum_shape_multistep` (added 2026-09-18, Section 186, user-
    # directed: "is there a way to force the pde's singular vectors ... to
    # go from expansive to contractive and back again? ... if the expansive
    # singular vectors all feed into other expansive singular vectors in
    # the evolution of the system, we will see runaway growth"). The
    # one-step spectrum-shape losses above constrain only the ONE-STEP
    # Jacobian's singular VALUES -- a map can look fine at every single
    # point along its trajectory and still blow up if the same expansive
    # subspace keeps re-feeding into itself instead of mixing into a
    # contracting one over successive steps (no cheap differentiable
    # handle on singular VECTORS themselves exists, so this targets the
    # same failure mode indirectly: see
    # `ks_latent.training.losses.propagator_multistep_spectrum_shape_loss`'s
    # docstring for the full mechanism and why persistent self-feeding is
    # caught by constraining the COMPOSED k-step Jacobian's own spectrum
    # toward `expand_target**k`/`contract_floor**k` rather than the
    # one-step spectrum). Always two-sided (a ceiling is the entire point).
    # Independent of `w_spectrum_shape`/`w_spectrum_shape_graded` -- may be
    # used together. OFF by default (0.0). `spectrum_shape_multistep_k`
    # defaults to matching `k_pred_max` at call sites unless overridden.
    w_spectrum_shape_multistep: float = 0.0
    spectrum_shape_multistep_k: int = 10
    spectrum_shape_multistep_n_samples: int = 16
    # `temporal_smoothness_loss` (added 2026-09-04, user-directed: "is
    # there a way to encode the test function here
    # analyze_latent_smoothness.py as a regularization term ... implement
    # that"). OFF BY DEFAULT (0.0) -- unlike `w_spatial` (which rewards a
    # CORRELATION structure between latent-index channels at a single
    # instant and cannot trivially collapse to a fixed point), this term's
    # global optimum in isolation genuinely IS latent collapse (every
    # state mapped to the same vector has zero step size). Read
    # `ks_latent.training.losses.temporal_smoothness_loss`'s docstring
    # before setting this above 0; test at a small weight with full
    # Gate 3/4 monitoring, same as every other regularizer here.
    w_smooth: float = 0.0
    smooth_curvature_weight: float = 1.0
    # `low_pass_spectral_loss` (added 2026-09-06, docs/sine_transform_pde_plan.md,
    # user-directed: "add a regularizer that acts as a low pass filter.
    # Namely, it penalizes the higher frequency terms of z proportional to
    # their frequency"): only meaningful when `ae.cfg` carries `K`/`L`
    # attributes (i.e. `encoder_kind="spectral_field"`, `z` is literally a
    # truncated rFFT spectrum) -- penalizes `sum_k k^lowpass_power *
    # |z_k|^2` (`k` the physical angular wavenumber `2*pi*m/L`), pushing
    # energy toward the lowest few kept modes on top of the hard truncation
    # at `K` modes. `lowpass_power=1` matches the literal "proportional to
    # frequency" request; `lowpass_power=2` is the classical `H^1` Sobolev
    # seminorm (`sum_k k^2|z_k|^2 = integral (dw/dx)^2 dx` by Parseval) --
    # see `ks_latent.training.losses.low_pass_spectral_loss`'s docstring.
    # OFF by default (0.0).
    w_lowpass: float = 0.0
    lowpass_power: float = 1.0
    # Stage-1 analogue of `Stage2TrainingConfig.w_lowpass_rollout` (added
    # 2026-09-08, docs/sine_transform_pde_plan.md, user-directed: "don't
    # reuse 104's phase 1 checkpoints, we need to retrain phase 1 with a
    # lowpass regularization term as well"). Unlike `w_lowpass` above
    # (penalizes the ENCODER's direct `z`), this penalizes `z_pred` --
    # the joint-training AUX PROPAGATOR's own short (`k_pred`-step)
    # rolled-out predictions, always computed every batch regardless of
    # aux backbone/size (`--full-propagator` only changes ITS SIZE, not
    # whether `z_pred` exists). Motivated by the same Section 104
    # visualization finding as `w_lowpass_rollout`: nothing previously
    # constrained the PROPAGATOR's own induced high-mode energy at
    # EITHER stage, only the encoder's. Applying it starting in Phase 1
    # (not just Phase 2, as the first `w_lowpass_rollout` attempt did)
    # lets the encoder and the real full-sized propagator co-adapt to the
    # constraint from the start, rather than a Phase-1-trained encoder
    # having a fixed-afterward representation that a Phase-2-only
    # constraint must then fight from a frozen starting point. OFF by
    # default (0.0).
    w_lowpass_rollout: float = 0.0
    lowpass_rollout_power: float = 1.0
    # `spectral_shape_floor_loss` (added 2026-09-07, docs/sine_transform_pde_plan.md,
    # user-directed: "I really just want to come up with a regularizer to
    # prevent the latent state from collapsing" -- following the finding
    # that `w_var`/`w_logdet`/`w_decorr` all fight the ALREADY-CORRECT
    # natural spectral energy decay of a real KS field, having no way to
    # distinguish "healthy decay" from "pathological collapse"). Only
    # meaningful when `ae.cfg` carries `K`/`L` attributes (i.e.
    # `encoder_kind="spectral_field"`). A ONE-SIDED floor: precompute
    # (once, from real training data, via `ks_latent.training.losses.
    # reference_mode_energy`) the TRUE field's own empirical per-mode
    # energy proportions among its lowest `K` modes -- a physically-
    # grounded reference for what a non-collapsed latent's per-mode energy
    # SHARE should be, calibrated from the real PDE rather than guessed --
    # then penalize `z`'s own per-mode share falling BELOW that reference,
    # never for exceeding it (see `spectral_shape_floor_loss`'s docstring
    # for why: the encoder plausibly needs to fold some of `u`'s discarded
    # higher-mode structure into the low modes it keeps, so `z`'s modes may
    # legitimately need MORE energy than the bare reference suggests, not
    # the same amount -- an exact shape-match would fight exactly that
    # necessary compensation). OFF by default (0.0).
    w_shape_floor: float = 0.0
    # `pde_head` distillation (added 2026-09-08, docs/sine_transform_pde_plan.md,
    # user-directed: "joint training seems pretty smart. can you implement
    # this idea" -- the hybrid discussed there: keep training the FREE,
    # unconstrained `aux` propagator as primary (it's the one that reliably
    # produces chaos, per H-PROP, and it's the one `l_pred` actually
    # optimizes), while a SEPARATE, always `backbone="spectral_pde"`
    # `pde_head` (passed to `train_stage1` directly, not part of this
    # config -- construction lives in the calling script) is trained
    # alongside it via a single-step distillation loss against `aux`'s own
    # (detached) realized one-step prediction. Weight on that loss -- see
    # `train_stage1`'s own docstring for the full mechanism and why this
    # differs from the earlier, already-failed post-hoc SINDy attempt
    # (§4.2 of the plan doc: frozen checkpoint, `R^2~0.005`, `D_KY=0`).
    # Meaningless (silently has no effect) unless `train_stage1` is also
    # given a non-`None` `pde_head`. OFF by default (0.0).
    w_pde_distill: float = 0.0
    # `pde_distill_detach_target` (added 2026-09-08, user-directed: "I
    # think the loss should be mutual for stage 1 training too. We always
    # want the propagator to have dynamics that can be easily modeled by
    # the pde_head right?"): `True` (default, UNCHANGED behavior) detaches
    # `aux`'s own realized step before comparing to `pde_head`'s
    # prediction -- gradient reaches only `pde_head` and, through `z`, the
    # ENCODER; `aux` stays fully protected. `False` makes the loss MUTUAL
    # (same convention Stage 2's own `pde_head` continuation always uses,
    # see `Stage2TrainingConfig`'s docstring -- there it's not even a
    # choice, since Stage 2 has no encoder to shape) -- gradient ALSO
    # reaches `aux` directly, genuinely pressuring it toward dynamics
    # `pde_head` can describe, not just an encoder that could support one.
    # Real risk, named directly: this is exactly the "pressure toward
    # simplicity" mechanism behind this project's own H-PROP finding
    # (every propagator pressured to look simple/local has collapsed to a
    # fixed point) -- if `pde_head` is weak or lazily fit (plausible early
    # in training, near-identity at `zero_init`), a mutual loss could pull
    # `aux` toward matching that impoverished target instead of real
    # dynamics. Monitor `l_pred`/Gate 3 `D_KY` specifically when this is
    # `False`, the same way every other regularizer here gets watched.
    pde_distill_detach_target: bool = True
    # `w_pde_distill_real` (added 2026-09-11, user-directed: "just train on
    # one step at a time from the true latent dynamics derived from the
    # encoder. but you can train on the whole batch of latent states from
    # the encoder at a time"): a SEPARATE single-step distillation term,
    # independent of `w_pde_distill` -- instead of matching `aux`'s own
    # predicted next state (`w_pde_distill`'s target), matches the REAL,
    # actually-encoded next state from data. Every consecutive pair
    # already present in the current batch's `z_win` (shape `(b, window,
    # d)`, already computed for `l_recon`/`l_pred` -- no new forward
    # passes needed) is used as an independent (state, real-next-state)
    # training example: `pde_head.step_one(z_win[:,:-1])` vs.
    # `z_win[:,1:]`, averaged over ALL such pairs in the batch at once.
    #
    # This is NOT the same idea as the already-failed `fit_latent_pde.py`
    # SINDy attempt (docs/sine_transform_pde_plan.md sec 4.2, a FROZEN,
    # closed-form regression using noisy finite-difference-estimated time
    # derivatives directly from data, R^2~0.005, collapsed to D_KY=0) --
    # this is gradient descent on the actual discrete `step_one` map
    # (whatever integrator: euler/rk4/etdrk4, no derivative-estimation
    # noise since both sides are actual encoded VALUES, not finite-
    # difference derivative estimates), can run jointly with encoder/
    # propagator training rather than frozen post-hoc, and inherits this
    # session's normalization/stability machinery (`poly_norm_power`,
    # `poly_stable_leading`, term-order restriction) the old attempt
    # never had. Still worth watching for the same failure mode though.
    # OFF by default (0.0), untested at scale -- verify smoke first.
    w_pde_distill_real: float = 0.0
    # `pde_distill_real_detach` (added 2026-09-11): mirrors
    # `pde_distill_detach_target`'s own convention. `True` (default)
    # detaches `z_win` for this term specifically -- gradient reaches only
    # `pde_head`'s own parameters, the encoder stays free to represent the
    # dynamics however it wants, uncorrupted by pressure to look
    # PDE-fittable via this specific route. `False` lets gradient also
    # reach the encoder through both the "state" and "target" sides of
    # each pair, pressuring it toward a representation this specific
    # single-step map can already predict well -- same "pressure toward
    # simplicity" risk class as every other mutual/non-detached option
    # here; monitor Gate 3 D_KY if set False.
    pde_distill_real_detach: bool = True
    # `w_pde_coeff_l1` (added 2026-09-10, user-directed: "I wonder if we
    # could impose some sparsity using l1 norm on the polynomial
    # coefficients of the pde"): penalizes `pde_head.body`'s own
    # `poly_coeffs.weight.abs().sum()` -- the SAME tensor whose entries are
    # this project's directly-interpretable PDE coefficients (see
    # `ks_latent.training.loops.pde_head_poly_coeffs`). Meaningless unless
    # `pde_head` is non-`None`, `cfg.w_pde_distill > 0` (or, in Stage 2,
    # `w_pde_rollout > 0`) and `pde_head` was built with
    # `field_kind="polynomial"` -- raises otherwise (fail loudly), since
    # `field_kind="mlp"` has no coefficient vector to sparsify. Only ever
    # applied to `pde_head`'s own coefficients, never to `aux`/`propagator`
    # itself -- this is a sparsity prior on the DISTILLED closure, not a
    # constraint on the free-running dynamics that produce the chaos in
    # the first place (would otherwise repeat this project's own H-PROP
    # "pressure toward simplicity collapses the primary propagator"
    # finding). Motivation: this arc's own extracted closures (Sections
    # 138/140/141/143) are already dominated by one or two terms (`w_xxxx`,
    # `w_xx`) with several small nonzero coefficients on higher cross
    # terms that plausibly aren't load-bearing -- L1 is the standard way
    # to let training itself zero out ones that don't help the fit, rather
    # than reading a dense coefficient table and guessing which terms are
    # noise. OFF by default (0.0), untested at scale -- verify smoke first.
    w_pde_coeff_l1: float = 0.0
    # `pde_coeff_l1_linear_only` (added 2026-09-11, user-directed: "please
    # try using the w_pde_coeff_l1 machinery to penalize single index
    # terms"): `False` (default, unchanged behavior) applies `w_pde_coeff_l1`
    # to EVERY coefficient. `True` restricts it to only the LINEAR (single-
    # derivative-index) terms -- see `ks_latent.training.loops.
    # pde_head_linear_coeffs`'s docstring for the full motivation: this
    # session found every closure dominated by a linear term produced
    # simple, near-periodic standalone dynamics (exact Fourier eigenmodes
    # of a linear operator), while closures dominated by a genuine
    # nonlinear term (w*w_x, w_x*w_xx) looked visibly richer -- this flag
    # lets L1 pressure specifically discourage linear-term dominance
    # without penalizing (and thus without directly suppressing) the
    # nonlinear terms the fit might need to lean on instead.
    pde_coeff_l1_linear_only: bool = False
    # `w_pde_distill_real_rollout`/`pde_distill_real_rollout_k` (added
    # 2026-09-12, user-directed: "directly replicate iLED's stabilization
    # mechanism and incorporate it into a rollout term to pde_distill
    # loss" -- iLED = arXiv:2309.05812, "Interpretable Learning of
    # Effective Dynamics for Multiscale Systems"): iLED trains its latent
    # dynamics with a MULTI-STEP forecast loss (`L_forecast`/
    # `L_rec_forecast`: integrate the learned model forward from a real
    # starting state and compare EVERY step to the real trajectory), not
    # just a 1-step distillation -- `w_pde_distill_real` above is exactly
    # iLED's would-be 1-step special case; this is the genuine multi-step
    # generalization. Rationale: a closure trained ONLY on 1-step targets
    # has no gradient pressure against exponential blowup under its own
    # repeated (autoregressive) integration -- which is exactly this
    # arc's own observed failure mode (Sections 155-158: every richer-
    # capacity `pde_head` variant diverged EARLIER under standalone
    # rollout, even as short-horizon fit improved). This term supplies
    # that missing pressure directly: `pde_head.rollout(...)` for
    # `pde_distill_real_rollout_k` steps from a REAL starting state
    # (`z_win`/`windows`, never `aux`/`propagator`'s own predictions),
    # compared at EVERY step against the REAL ground-truth states already
    # present in the training window -- reuses `pde_head`'s OWN chained
    # integrator (unlike `w_pde_distill_real`'s single `step_one` call),
    # so gradient now genuinely sees multi-step compounding, but the
    # TARGET stays real data throughout (never a moving/co-adapting
    # target), avoiding the specific "incredibly hard to train"
    # stiffness `w_pde_rollout` ("option B", distilling against
    # `propagator`'s own rollout) was flagged for in Sections 101-106.
    # `pde_distill_real_rollout_k` (default 4): deliberately short relative
    # to `k_pred_max`/`k_max` -- iLED's own forecast horizons are modest,
    # not deep unrolls; must be `<= max(k_pred, k_pred_max)` (Stage 1) or
    # `<= k_max` (Stage 2), enforced at the top of the training loop (fail
    # loudly if violated -- the window isn't sized to supply more real
    # future steps than that). Respects `pde_distill_real_detach`'s
    # existing convention (same flag, no separate one added). OFF by
    # default (0.0), untested at scale -- verify smoke first.
    w_pde_distill_real_rollout: float = 0.0
    pde_distill_real_rollout_k: int = 4
    # `pde_distill_real_rollout_warmup_epochs` (added 2026-09-12, found
    # necessary empirically the same day): ramps the ACTUAL rollout length
    # used each epoch from `1` up to `pde_distill_real_rollout_k` over this
    # many epochs (`ks_latent.training.loops.k_curriculum`, the same
    # mechanism `k_pred_max`'s own ramp uses), rather than applying the
    # full target horizon from epoch 0. NOT optional in practice: a direct
    # smoke test found that applying the full `pde_distill_real_rollout_k`
    # immediately, against a freshly-initialized (or only lightly warm-
    # started) `pde_head`, can produce a divergent (inf/nan) rollout on
    # the VERY FIRST batch -- which poisons the ENTIRE combined loss (one
    # nan term makes the whole sum nan) and corrupts the optimizer's
    # (AdamW) running moment estimates PERMANENTLY for every parameter,
    # not just pde_head's -- silently wrecking the rest of the run. `0`
    # (default) means NO ramp (`k_curriculum` returns `k_max` immediately)
    # -- matches `pde_distill_real_rollout_k`'s naive/unsafe behavior, kept
    # as the literal default for config-field consistency, but callers
    # MUST set this whenever `w_pde_distill_real_rollout > 0` (see
    # `train_stage1_patched.py`/`train_stage2_patched.py`'s own CLI
    # wiring, which defaults it to `max(1, round(0.3*epochs))` -- the SAME
    # 30%-of-training convention `k_pred_max`'s own warmup already uses,
    # for the identical reason: "starting at the full rollout length
    # immediately is unstable -- an untrained [closure] compounds its own
    # error").
    pde_distill_real_rollout_warmup_epochs: int = 0
    # `w_pde_nonlinear_l2` (added 2026-09-12, user-directed alongside the
    # rollout term above -- see its docstring for the full iLED
    # background): the OTHER half of iLED's stabilization mechanism.
    # iLED's linear operator `A_theta` is reparametrized `W - W^T -
    # diag(|w|)` (provably non-positive eigenvalues); this project's
    # analogous, STRONGER mechanism is `poly_fixed_linear_terms`/
    # `poly_stable_leading` (already implemented, Sections 155-158 already
    # use `poly_fixed_linear_terms`) -- so no new code was needed to
    # replicate that half. iLED separately penalizes `||Psi_1(z,h)||^2`
    # (their nonlinear closure's own squared magnitude) to keep it from
    # growing large enough to dominate/destabilize the stable linear part
    # -- THIS is the half that had no analogue here until now. Penalizes
    # `ks_latent.training.loops.pde_head_nonlinear_output(pde_head,
    # z).pow(2).mean()` (see `_SpectralPDEDeltaBody.nonlinear_field`'s
    # docstring for exactly what's isolated: every term of length>=2 in
    # the polynomial/Chebyshev library, i.e. everything except the
    # constant and the single-derivative linear terms). Evaluated on
    # real, detached on-attractor states only (never mutual with the
    # encoder/propagator) -- this is a prior on `pde_head`'s OWN closure
    # alone, deliberately not another "pressure toward simplicity" channel
    # into the primary dynamics (same reasoning as `w_pde_coeff_l1`).
    # Requires `field_kind` in `("polynomial", "chebyshev")` -- raises
    # otherwise (fail loudly), same convention as `w_pde_coeff_l1`. OFF by
    # default (0.0), untested at scale -- verify smoke first.
    w_pde_nonlinear_l2: float = 0.0
    # `w_pde_spectrum_shape`/`w_pde_spectrum_shape_multistep` (added
    # 2026-09-21, Section 192, user-directed: "maybe we can try our new
    # regularization methods with the joint propagator/pde training from
    # before? where the pde is not the propagator... please add the new
    # regularizer just to the pde_head"): pde_head analogues of
    # `w_spectrum_shape`/`w_spectrum_shape_multistep` (Sections 131/186,
    # applied to the MAIN propagator there) -- see those fields'
    # docstrings and `ks_latent.training.losses.propagator_spectrum_
    # shape_loss`/`propagator_multistep_spectrum_shape_loss` for the full
    # mechanism. Motivation: Section 167's own pde_head (masked_mlp_
    # expand as the real, DECOUPLED propagator -- already genuinely
    # chaotic, D_KY=22.67, dead center of the true 21-24 benchmark --
    # with pde_head fit separately against real data only) never
    # diverges standalone but DECAYS toward a fixed point (1.82 -> 0.28
    # over 200 steps) rather than sustaining chaos on its own. Applying
    # these regularizers to `pde_head.step_one` specifically (NOT the
    # main `aux`/propagator, which is already healthy and should not be
    # touched) gives pde_head direct pressure toward a genuinely
    # expansive-then-contracting Jacobian spectrum, independent of
    # whatever the real propagator is doing. Evaluated on real, detached
    # on-attractor states (same convention as `w_pde_nonlinear_l2`/`w_pde_
    # mean_conservation` above -- never mutual with the encoder; this is
    # a prior on pde_head's OWN closure alone). Requires `pde_head` to be
    # built (`--pde-distill`); raises via the same `kernel_Lhat`-style
    # `getattr`/mode checks the main-propagator version already uses.
    # OFF by default (0.0) for both.
    w_pde_spectrum_shape: float = 0.0
    pde_spectrum_shape_n_expand: int = 13
    pde_spectrum_shape_expand_target: float = 1.1
    pde_spectrum_shape_contract_floor: float = 0.6
    pde_spectrum_shape_n_samples: int = 32
    pde_spectrum_shape_two_sided: bool = False
    w_pde_spectrum_shape_multistep: float = 0.0
    pde_spectrum_shape_multistep_k: int = 10
    pde_spectrum_shape_multistep_n_samples: int = 16
    # `w_pde_spectrum_shape_self`/`w_pde_spectrum_shape_multistep_self`
    # (added 2026-09-21, Section 193, user-directed: "did you use the
    # rollout regularization for the pde_head where we tried to
    # discourage collapse and chaos by looking at the jacobian? maybe
    # that's something we should add to stage 2 or even stage 3 if we
    # train the pde on its own"). Gap in `w_pde_spectrum_shape`/`w_pde_
    # spectrum_shape_multistep` above: both are evaluated ONLY on real,
    # encoder-derived on-attractor states (z_win/windows) -- they shape
    # pde_head's Jacobian correctly NEAR THE TRUE ATTRACTOR, but say
    # nothing about states pde_head's OWN free rollout actually visits if
    # it drifts off the true manifold -- exactly the region where Section
    # 167's pde_head decayed toward a fixed point over its own 200-step
    # standalone rollout, a region the real-data-anchored version never
    # samples from. These two fields apply the SAME two losses
    # (`propagator_spectrum_shape_loss`/`propagator_multistep_spectrum_
    # shape_loss`, same shared `pde_spectrum_shape_n_expand`/`expand_
    # target`/`contract_floor`/`two_sided` targets above) but evaluated
    # at states drawn from `pde_head`'s OWN self-generated rollout
    # instead: roll `pde_spectrum_shape_self_rollout_k` steps forward
    # from a batch of real starting states (`torch.no_grad()`, cheap --
    # no backprop through the rollout chain itself, matching every other
    # once-per-epoch Jacobian regularizer's "evaluate AT a batch of
    # points" convention, not "backprop through how we got there"), take
    # the final (fully detached) state of each self-rollout as the
    # evaluation pool. A NaN/inf self-rollout (plausible mid-training, an
    # unsupervised free-running rollout of a partially-converged closure)
    # skips this batch's contribution entirely rather than poisoning the
    # whole loss -- same discipline as `w_pde_energy_floor`'s own
    # documented fix for exactly this failure mode. Independent, genuinely
    # separate weights from the real-data versions above (both may be on
    # at once) so the two can be attributed independently. OFF by default
    # (0.0) for both -- NOT yet launched; the decision to enable these was
    # deferred until Section 192's own real-data-only version's results
    # (Stage 1 + Stage 2 diagnostics) are in.
    w_pde_spectrum_shape_self: float = 0.0
    pde_spectrum_shape_self_rollout_k: int = 20
    pde_spectrum_shape_self_n_samples: int = 16
    w_pde_spectrum_shape_multistep_self: float = 0.0
    # `w_pde_mean_conservation` (added 2026-09-12, user-directed: "can you
    # recommend a fix to prevent drift toward mode-0"): true KS conserves
    # `int u dx` EXACTLY -- every term in `-u*u_x - u_xx - u_xxxx` is a
    # total x-derivative, which integrates to zero over a periodic
    # domain. A freely-fit polynomial closure has no such guarantee.
    # Measured directly on Section 164's own trained pde_head: its
    # standalone rollout was broadband at t=0 but 72-86% concentrated in
    # the self-FFT DC/mean mode by t=54, and `field(z).mean(dim=-1)`
    # (the predicted dz/dt's own spatial mean) across 500 real states was
    # `-0.00031 +- 0.0045` -- small but systematically nonzero. Two
    # concrete sources found in that fit: the bias/intercept (0.0065, a
    # flat additive offset with no counterpart in true KS -- see
    # `spectral_poly_no_constant`, which zeros this architecturally) and
    # the `w_x*w_x` term (coefficient -0.043, the largest term that is
    # NOT a total x-derivative -- `w_x^2 >= 0` pointwise everywhere, so
    # its contribution to the mean never oscillates out the way a
    # genuine total-derivative term does, e.g. `w_xx*w_xxx =
    # d/dx(w_xx^2/2)` integrates to exactly zero). Penalizes
    # `ks_latent.training.loops.pde_head_field_output(pde_head,
    # z).mean(dim=-1).pow(2).mean()` directly -- a general prior mirroring
    # true KS's own exact conservation law, rather than hand-suppressing
    # whichever specific terms happen to violate it (that set can shift
    # as other regularizers/weights change). Evaluated on real, detached
    # on-attractor states only (same convention as `w_pde_nonlinear_l2`)
    # -- a prior on `pde_head`'s OWN closure alone, not another pressure
    # channel into the encoder/propagator. Works for any `field_kind`
    # (unlike `w_pde_nonlinear_l2`, which requires polynomial/chebyshev).
    # OFF by default (0.0), untested at scale -- verify smoke first.
    w_pde_mean_conservation: float = 0.0
    # `w_pde_energy_floor` (added 2026-09-13, user-directed: viewing
    # pde_head's own free-running smooth-field GIF (Section 167, the
    # --pde-poly-exclude-nonconservative run), "I also see the decay in
    # the pde trajectory towards 0... would it be worth trying to fix the
    # maximum energy in the pde in physical space?... force the model to
    # always have an 'energetic' component"): penalizes
    # `ks_latent.training.losses.spatial_energy_floor_loss` on pde_head's
    # OWN autoregressive rollout (see that function's docstring for the
    # full mechanism -- a PER-SAMPLE spatial-std-across-the-ring floor, a
    # different axis/failure-mode entirely from `w_var`'s per-channel-
    # across-batch floor or `Stage2TrainingConfig.w_varmatch`'s per-
    # channel-across-different-ICs floor: this targets a SINGLE
    # trajectory's own spatial profile decaying toward uniform over time,
    # exactly what Section 167's standalone rollout showed -- no
    # divergence, no mode-0 lock-in, but the smoothed field's peak
    # amplitude steadily decayed over 200 steps). Purely a property of
    # pde_head's own dynamics, so this rollout is UNSUPERVISED (no real
    # target beyond the single starting state -- pde_head rolls forward
    # entirely on its own for `pde_energy_floor_rollout_k` steps) and
    # detached from the encoder/propagator, same convention as
    # `w_pde_nonlinear_l2`/`w_pde_mean_conservation`. `pde_energy_floor_
    # gamma` (default 0.4): real encoded z's own spatial std averages
    # ~0.97, minimum ~0.53 (measured directly on Section 167's real
    # data) -- the floor sits clearly below that range, not at it (a
    # floor against total collapse, not a target to pull every step
    # toward, same hinge-loss philosophy as `variance_floor_loss`).
    # OFF by default (0.0), untested at scale -- verify smoke first.
    w_pde_energy_floor: float = 0.0
    pde_energy_floor_gamma: float = 0.4
    # `pde_energy_floor_rollout_k` (default 30): since this term is
    # UNSUPERVISED (no real future states needed, unlike `pde_distill_
    # real_rollout_k`), it is NOT constrained by the training window
    # size -- can be, and should be, much longer than the distillation
    # rollout's own target (4), since the observed decay compounds over
    # tens to hundreds of steps, well past what a short supervised
    # rollout would ever directly pressure.
    pde_energy_floor_rollout_k: int = 30
    # `pde_energy_floor_warmup_epochs` (default 0 = no ramp): SAME
    # empirically-necessary safety mechanism as `pde_distill_real_
    # rollout_warmup_epochs` -- this is ALSO a genuine multi-step
    # autoregressive rollout through pde_head's own (possibly
    # undertrained) dynamics, so applying the full target horizon from
    # epoch 0 risks the identical inf/nan-and-permanently-corrupt-the-
    # optimizer failure mode found empirically in Section 159's
    # postmortem. Callers MUST set this whenever `w_pde_energy_floor > 0`
    # (see `train_stage1_patched.py`/`train_stage2_patched.py`'s own CLI
    # wiring, which defaults it to `max(1, round(0.3*epochs))`, the same
    # convention `k_pred_max`/`pde_distill_real_rollout_warmup_epochs`
    # already use).
    pde_energy_floor_warmup_epochs: int = 0
    # `w_prop_energy_floor` (added 2026-09-14, user-directed): generalizes
    # `w_pde_energy_floor` to the PRIMARY propagator (`aux` here) directly,
    # for the case where `aux` itself IS the closure under test (e.g.
    # `--aux-backbone spectral_pde_raw`, no separate `pde_head`) --
    # Sections 169/171 found that setup ALSO decays toward a fixed point
    # (Lyapunov lambda1=-0.04, D_KY=0, standalone rollout decaying to
    # ~0.0007 by step 199), the SAME failure mode `w_pde_energy_floor` was
    # built for, just on `aux` instead of a separate `pde_head`. Critically,
    # `Stage2TrainingConfig.w_varmatch` (the existing anti-collapse term)
    # was tried FIRST and did NOT prevent this: `w_varmatch` only ever sees
    # the SUPERVISED training rollout, capped at `k_max` -- it cannot guard
    # against a collapse rate that is slow enough to look acceptable within
    # that horizon but still decays to ~0 well beyond it. This term is
    # UNSUPERVISED (no real target, no window-size constraint) for exactly
    # that reason -- same mechanism/rationale as `w_pde_energy_floor`,
    # applied to `aux.rollout` instead of `pde_head.rollout`, penalizing
    # `ks_latent.training.losses.spatial_energy_floor_loss` on `aux`'s own
    # ramped (`prop_energy_floor_warmup_epochs`, same ramp-safety
    # convention as `pde_energy_floor_warmup_epochs`/`pde_distill_real_
    # rollout_warmup_epochs`) autoregressive rollout from a real detached
    # starting state. Gradient reaches `aux`'s own parameters fully (that
    # IS the propagator being trained here) but not the encoder (the
    # starting state is detached). OFF by default (0.0), untested at scale
    # -- verify smoke first.
    w_prop_energy_floor: float = 0.0
    prop_energy_floor_gamma: float = 0.4
    prop_energy_floor_rollout_k: int = 30
    prop_energy_floor_warmup_epochs: int = 0
    # `stable_linear_lr_factor` (added 2026-09-14, Section 174, user-
    # directed: "we can't make huge steps in nu, otherwise the dynamics
    # will change pretty drastically"): multiplies `lr` for exactly the
    # parameters exposed by `aux.stable_linear_raw_parameters()`
    # (`AuxPropagatorConfig.spectral_poly_stable_linear_terms`'s raw
    # coefficients) -- a SEPARATE, much smaller optimizer param group, not
    # a manual gradient clip, so it doesn't fight Adam's own per-parameter
    # adaptive scaling. `1.0` (default): no effect, unchanged behavior
    # (also a no-op whenever `stable_linear_raw_parameters()` returns an
    # empty list, i.e. the mechanism is off).
    stable_linear_lr_factor: float = 0.02
    # `w_kernel_unstable_floor`/`kernel_unstable_target_modes`/
    # `kernel_unstable_margin` (added 2026-09-16, Section 183, user-
    # directed: "we should also increase the regularizer that tries to
    # keep at least 13 unstable modes in the pde, or if that doesn't
    # exist, implement it" -- no such regularizer existed). Only has an
    # effect when the propagator exposes `kernel_Lhat()` (`spectral_
    # field_kind="forced_burgers"` with `spectral_burgers_kernel_
    # instability=True`). `kernel_unstable_floor_loss` (ks_latent.
    # training.losses) architecturally-independent-of-data hinge penalty
    # keeping the lowest `kernel_unstable_target_modes` Fourier modes
    # (excluding DC, always exactly 0) above `kernel_unstable_margin` in
    # `Lhat(k)` -- i.e. genuinely growing/unstable. `w_kernel_unstable_
    # floor=0.0` (default): off, unchanged behavior.
    w_kernel_unstable_floor: float = 0.0
    kernel_unstable_target_modes: int = 13
    kernel_unstable_margin: float = 0.0
    # `w_z_lowpass`/`w_z_lowpass_rollout` (added 2026-09-11, user-directed:
    # "I would like to penalize higher frequencies in the fourier
    # transform of the latent space. This may help make the pde dynamics
    # more easily fit and smooth"): generalizes `w_lowpass`/
    # `w_lowpass_rollout` (which require `encoder_kind="spectral_field"`,
    # since they assume `z` already IS a truncated rFFT spectrum) to ANY
    # encoder's raw `z`, via `ks_latent.training.losses.
    # latent_self_spectrum_lowpass_loss` -- computes `z`'s own self-FFT
    # (treating the `d_latent` index as a periodic ring, the same
    # convention `backbone="spectral_pde_raw"` already uses) and penalizes
    # its high-frequency energy. `w_z_lowpass`: applied to the encoder's
    # own `z` (reconstruction path). `w_z_lowpass_rollout`: applied to the
    # (full, `--full-propagator`) aux propagator's own rolled-out
    # `z_pred`, mirroring `w_lowpass_rollout`'s Stage-1 convention. Both
    # OFF by default (0.0). See `z_lowpass_K`/`z_lowpass_L` for the
    # self-FFT truncation/ring-length parameters (shared by both terms).
    w_z_lowpass: float = 0.0
    w_z_lowpass_rollout: float = 0.0
    z_lowpass_power: float = 1.0
    z_lowpass_rollout_power: float = 1.0
    # `None` (default): derived at call time as `K = d_latent//2+1`
    # (no truncation of z's own self-spectrum -- matches a
    # `spectral_pde_raw` pde_head's own default `spectral_K`, so the
    # encoder is pressured toward exactly the spectral shape a pde_head
    # trained on the same `z` will see) and `L = d_latent` (matches that
    # pde_head's own default `spectral_L`). Set explicitly to target a
    # DIFFERENT self-FFT truncation/ring length than whatever pde_head
    # (if any) is also in use.
    z_lowpass_K: int | None = None
    z_lowpass_L: float | None = None
    # `w_channel_mean` (added 2026-09-11, user-directed: "implement the
    # root fix"). `encoder_kind="local_field"` only. Penalizes each free
    # residual channel (1..local_channels-1) for having a nonzero mean
    # ACROSS SITES -- see `ks_latent.training.losses.
    # local_field_channel_mean_loss`'s docstring for the full motivation
    # (a measured, real period-`local_channels` flattening artifact that
    # was found to dominate a trained pde_head's own self-spectrum) and
    # for why this is a soft, loss-level penalty rather than an
    # architectural forward-pass fix (the latter requires a global
    # reduction that breaks the encoder's core locality guarantee). OFF
    # by default (0.0), untested at scale -- verify smoke first.
    w_channel_mean: float = 0.0
    # Linear regularizer-weight decay (added 2026-09-02, user-directed:
    # "maybe it would be worth initially having regularizer parameters
    # high and then decaying them overtime to increase prediction
    # performance" -- motivated by the observed tradeoff where raising
    # w_var/w_logdet/w_spatial improves latent covariance conditioning
    # but measurably worsens downstream DA/rollout metrics, e.g. Sections
    # 56 vs. 58 in docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md). All three OFF
    # by default (`None` = no decay, unchanged behavior: the weight stays
    # fixed at `w_var`/`w_logdet`/`w_spatial` for the whole run). When set,
    # the corresponding weight linearly decays from its own `w_var`/
    # `w_logdet`/`w_spatial` value (epoch 0) to `w_var_end`/`w_logdet_end`/
    # `w_spatial_end` (the final epoch, `cfg.epochs - 1`) -- see
    # `ks_latent.training.loops.linear_decay`. Decaying to a NONZERO floor
    # (not all the way to 0) is deliberate: this project's latent collapse
    # is correlation-driven, not just variance-shrinkage-driven (see
    # docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md's collapse investigation), so
    # fully removing the anti-collapse pressure late in training risks the
    # latent drifting back toward collapse right when regularization is
    # weakest.
    w_var_end: float | None = None
    w_logdet_end: float | None = None
    w_spatial_end: float | None = None
    k_pred: int = 2
    k_pred_max: int = 0
    k_pred_warmup_epochs: int = 0
    weight_decay: float = 1e-5
    warmup_epochs: int = 2
    lr_min_factor: float = 0.02
    noise_std: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.w_spectrum_shape_graded > 0 and self.spectrum_shape_graded_reference_path is None:
            raise ValueError(
                "w_spectrum_shape_graded > 0 requires spectrum_shape_graded_reference_path "
                "to be set (a .npy file of shape (d_latent,), typically from "
                "scripts/compute_reference_spectrum.py)"
            )


@dataclass(frozen=True)
class Stage2TrainingConfig:
    """Brief §5.2: AdamW lr=3e-4, cosine, 20 epochs, batch 256, K-curriculum
    2->16 over 8 epochs, input noise annealed 0.10->0.02, best-val-k
    checkpointing.

    `weight_decay`/`warmup_epochs`/`lr_min_factor`/`latent_loss` added
    2026-08-29 -- see `Stage1TrainingConfig`'s docstring for the shared
    rationale on the first three. `latent_loss`: the reference
    implementation (following Linot & Graham, arXiv:2109.00060) found L1
    gave "visibly better short-time tracking" than L2 for the propagator's
    latent-space loss, on the grounds that L1 is less dominated by the rare
    large increments a chaotic trajectory produces. Default stays `"l2"`
    (the brief's original, exactly matching the loss already validated in
    this codebase's tests); `"l1"` is an untested-at-scale alternative to
    try.

    `k_mid`/`k_mid_epochs` (added 2026-08-29, user-directed): split the
    K-curriculum's linear ramp into two segments -- `2 -> k_mid` over the
    first `k_mid_epochs` epochs, then `k_mid -> k_max` over the remaining
    `k_warmup_epochs - k_mid_epochs` -- so training dwells longer on short
    rollout horizons before the harder long-horizon regime. `k_mid=None`
    (default) is the original single-segment linear ramp. See
    `k_curriculum`'s docstring for the rationale (the plateau observed
    training the vit/attn_window=4/linear propagator with a straight 2->16
    ramp over 8 epochs).

    `w_varmatch` (added 2026-08-29, user-directed): weight on a variance-
    matching term applied to the *rolled-out predictions* `z_pred`,
    computed via `decorr_var_loss`'s `l_var` (only the per-dimension-
    variance-vs-1 term, not its off-diagonal `l_decorr` term, and unrelated
    to the separate banded `RegConfig.lambda_z` smoothness regularizer).
    Motivated by a concrete failure found running this propagator
    autonomously (no re-anchoring to data): every tested initial condition
    converged to the *same single fixed point* within ~100-200 steps --
    consistent with the measured Lyapunov spectrum (`lambda1<0`, `D_KY=0`)
    and the DA ensemble collapse (`spread/rmse=0.066`) from Gate 3. A
    collapsing rollout has `Var(z_pred_i) -> 0` across different initial
    conditions as they converge to the same point, so penalizing that
    directly opposes the failure mode. Default `0.0` is off (unchanged
    behavior). Target is fixed at 1 by default (matching `decorr_var_loss`'s
    own default) unless `w_varmatch_adaptive` is set.

    `w_varmatch_adaptive` (added 2026-08-31, user-directed -- Phase 2
    architecture doc Section 35, option 7): if `True`, replace the
    hardcoded uniform target of `1` above with the AE's own REAL
    per-channel latent variance (computed once from `train_sequences`
    at the start of `train_stage2`). Motivated by a follow-up finding: a
    fine-tuned AE's latent can have one channel collapse to a true
    variance many orders of magnitude below `1` (observed `~2.4e-6`) --
    for that channel, the fixed target of `1` actively fights the primary
    latent-matching loss instead of complementing it, since it pulls the
    prediction toward a value the true data never takes. Data-adaptive
    targets preserve the original anti-fixed-point-collapse motivation
    (matching real cross-IC diversity, whatever scale it's actually at)
    without that conflict. Default `False` (unchanged behavior) since
    this changes what `w_varmatch>0` actually optimizes for existing
    recipes.

    `noise_step_start`/`noise_step_end` (added 2026-08-29, user-directed):
    linearly-annealed Gaussian noise std passed as `LatentPropagator.
    rollout`'s `step_noise` -- injected at *every* autoregressive step
    during training, not just the rollout's starting pair (that's
    `noise_in_start`/`noise_in_end` above, which only touches the first
    step). Ported from `docs/ML_for_KS_writeup.md` §4.2's `sigma_step`.
    Tried as the direct fix after `k_max=32` + `w_varmatch=0.1` +
    increased `noise_in` all failed to prevent the fixed-point collapse
    above: neither of those levers ever exposes the model to off-manifold
    states more than once per training window, so this is the one that
    actually forces repeated recovery-from-error within even a short
    rollout. Default `0.0` for both is off (unchanged behavior).
    """

    lr: float = 3e-4
    epochs: int = 20
    batch_size: int = 256
    # `grad_clip` (added 2026-09-12, found necessary empirically): unlike
    # `Stage1TrainingConfig`, `train_stage2` never clipped gradients at
    # all -- a real, general safety gap, not specific to any one loss
    # term. Confirmed directly: Section 159's `w_pde_distill_real_rollout`
    # term overflowed to inf/nan mid-training (even with its own
    # curriculum ramp), and with no gradient clipping the resulting NaN
    # permanently corrupted `pde_head`'s own weights (every coefficient
    # became `nan`) for the rest of the 300-epoch run. Same default
    # (`1.0`) and mechanism (`torch.nn.utils.clip_grad_norm_`) as
    # `Stage1TrainingConfig.grad_clip` -- applied unconditionally (not
    # gated on `pde_head`), since an unclipped propagator/latent-loss
    # gradient can in principle explode on its own too.
    grad_clip: float = 1.0
    k_max: int = 16
    k_warmup_epochs: int = 8
    k_mid: int | None = None
    k_mid_epochs: int = 0
    noise_in_start: float = 0.10
    noise_in_end: float = 0.02
    noise_step_start: float = 0.0
    noise_step_end: float = 0.0
    gamma: float = 1.0
    weight_decay: float = 1e-5
    warmup_epochs: int = 2
    lr_min_factor: float = 0.02
    latent_loss: str = "l2"
    w_varmatch: float = 0.0
    w_varmatch_adaptive: bool = False
    # Stage-2 analogue of Stage1TrainingConfig.w_spatial (added 2026-09-01,
    # user-directed): applies the same spatial_coherence_loss, but to the
    # PROPAGATOR's rolled-out predictions z_pred rather than the encoder's
    # own output -- rewards a propagator whose own predictions preserve the
    # banded/local correlation structure Stage 1 already induced in z,
    # instead of only relying on Stage 1 to have produced it. Off by
    # default (0.0), same collapse-risk precedent as Stage 1's own
    # w_spatial -- see spatial_coherence_loss's docstring before raising
    # this above 0.
    w_spatial: float = 0.0
    spatial_bandwidth: float = 3.0
    # See Stage1TrainingConfig.spatial_signed's docstring -- identical
    # meaning, applied to Stage 2's own w_spatial term.
    spatial_signed: bool = False
    # Stage-2 analogue of Stage1TrainingConfig.w_jacobian_bandedness --
    # see that field's docstring and `ks_latent.training.losses.
    # propagator_jacobian_bandedness_loss`'s docstring for the full
    # mechanism (D3's differentiable analogue). Evaluated directly on
    # `propagator.step_one` using real, UNNOISED encoded states drawn
    # from the current batch's own windows -- no rollout involved, same
    # convention as Stage 2's own `w_spectrum_shape`. `mode="markovian"`
    # only. OFF by default (0.0).
    w_jacobian_bandedness: float = 0.0
    jacobian_bandedness_bandwidth: float = 3.0
    jacobian_bandedness_n_samples: int = 32
    # `encoder_kind="spectral_field"`/`backbone="spectral_pde"` only (added
    # 2026-09-08, docs/sine_transform_pde_plan.md, user-directed after
    # visualizing Section 104's D_KY=22 result: "it seems like the next
    # logical step would be to try to limit higher modes energy in the
    # propagator with some kind of soft constraint. this is where the
    # [low]-pass filter idea might work"). Stage-2 analogue of
    # Stage1TrainingConfig.w_lowpass, but applied to the PROPAGATOR's own
    # rolled-out predictions `z_pred` rather than the encoder's direct
    # output -- Section 104's visualizations showed the encoder's own z
    # (Stage 1's w_lowpass target) looked physically reasonable, but the
    # PROPAGATOR, once run freely, drove the high-index latent channels to
    # hard-saturated, unphysical amplitudes (visible as a much finer-
    # grained, higher-wavenumber spatial pattern than the true L=22
    # attractor's own smooth, large-scale chaos) -- nothing previously
    # constrained the propagator's OWN induced high-mode energy, only the
    # encoder's. `K`/`L` are read from `propagator.cfg.spectral_K`/
    # `.spectral_L` (duck-typed, same convention as Stage 1's `ae.cfg.K`/
    # `.L` read) -- only meaningful for `backbone="spectral_pde"`. OFF by
    # default (0.0).
    w_lowpass_rollout: float = 0.0
    lowpass_rollout_power: float = 1.0
    # `w_logdet_rollout` (added 2026-09-09, user-directed, after Section 113
    # -- w_varmatch/noise_step, applied directly to z_pred -- still
    # collapsed (D_KY=0.0): "add some w_logdet too to the w states (irfft
    # of latent frequency state z), I think that may prevent collapse as
    # well"). Stage-2 analogue of `logdet_barrier_loss` (Stage 1's own
    # `w_logdet`, applied to the encoder's raw `z`), but applied to the
    # PROPAGATOR's own rolled-out predictions -- and, per the user's own
    # framing, in PHYSICAL space rather than spectral-coefficient space:
    # `z_pred` is first mapped through `decode_from_spectrum` (the exact
    # irfft reconstruction `w_hat = irfft(z_pred)`, same helper `field()`
    # and `synthesize_derivatives` already use) before the full-covariance
    # log-det barrier is computed on THOSE physical-space states. This is a
    # strictly different (and more general) mechanism than `w_varmatch`
    # (which only constrains z_pred's MARGINAL per-channel variance, in
    # spectral-coefficient space): a collapsing rollout drives every
    # physical grid point toward the same constant field regardless of
    # initial condition, which shows up as the full covariance of `w_hat`
    # across the batch losing rank -- exactly what `logdet_barrier_loss`
    # penalizes, and unlike a spectral-space check, this reacts to
    # collapse in the field's actual physical shape (amplitude AND
    # location), not just its Fourier-coefficient magnitudes. `K`/`N_w`
    # are read from `propagator.cfg.spectral_K`/`.spectral_N_w` (duck-
    # typed, same convention as `w_lowpass_rollout` above) -- only
    # meaningful for `backbone="spectral_pde"`/`"spectral_pde_raw"`. OFF by
    # default (0.0).
    w_logdet_rollout: float = 0.0
    logdet_rollout_eps: float = 1e-3
    # `w_logdet_rollout_latent` (added 2026-09-23, Section 203, user-
    # directed: "do we apply w_var, w_logdet, w_spatial during stage 2? we
    # probably should. If we don't please implement this"). Stage 1's own
    # `w_logdet` applies `logdet_barrier_loss` DIRECTLY to the encoder's
    # raw `z` -- no physical-space decode needed, since `logdet_barrier_
    # loss` only ever needed a batch of vectors (see its own docstring;
    # the mechanism that actually resolved this project's own "latent
    # channel collapse from fine-tuning" failure, per CLAUDE.md's project
    # memory). `w_logdet_rollout` above generalizes that to Stage 2's
    # rolled-out `z_pred`, but ONLY for `backbone="spectral_pde"`/
    # `"spectral_pde_raw"` (it requires `decode_from_spectrum`, which only
    # those backbones have). Every other backbone (`mlp`, `vit`,
    # `transformer`, `masked_mlp`, ...) had NO Stage-2 logdet-based
    # anti-collapse term at all until this field: applies `logdet_barrier_
    # loss` directly to `z_pred` itself (mirroring Stage 1's own `w_logdet`
    # exactly, just on the propagator's rolled-out predictions instead of
    # the encoder's raw output), general to any backbone. Distinct from,
    # and may be combined with, `w_logdet_rollout` above -- they react to
    # collapse in different spaces (latent covariance vs. decoded physical
    # field shape) and neither subsumes the other. OFF by default (0.0).
    w_logdet_rollout_latent: float = 0.0
    logdet_rollout_latent_eps: float = 1e-3
    # `w_spectrum_shape` (added 2026-09-10, Section 132, user-directed:
    # "so it looks like it collapsed in stage 2? please wire that fix you
    # mentioned into stage 2"): Stage-2 analogue of
    # `Stage1TrainingConfig.w_spectrum_shape`
    # (`ks_latent.training.losses.propagator_spectrum_shape_loss`) -- see
    # that field's docstring for the mechanism. Added here specifically
    # because Section 131 found that applying this ONLY in Stage 1 is not
    # enough: a real checkpoint's propagator Jacobian was healthy right
    # after Stage 1 (volume-change factor 0.68, 13/44 singular values
    # >= 1.0) but had fully re-collapsed by the end of Stage 2's own
    # UNREGULARIZED 300-epoch continuation (5/44 >= 1.0, volume factor
    # 5.3e-5) -- Stage 2 is a separate propagator-only fine-tune (frozen
    # AE) with no anti-collapse pressure of this kind at all, and it is
    # also the LONGER-horizon phase (k ramping to k_max), i.e. exactly
    # where this pressure is needed most. Unlike `w_spatial`/
    # `w_lowpass_rollout`/`w_logdet_rollout` above (which all act on the
    # rolled-out PREDICTIONS `z_pred`), this acts on `propagator.step_one`
    # directly via real (unnoised) encoded states drawn from the current
    # batch's own windows -- there is no rollout involved, matching Stage
    # 1's own usage exactly. `mode="markovian"` only (same restriction as
    # Stage 1's copy). OFF by default (0.0).
    w_spectrum_shape: float = 0.0
    spectrum_shape_n_expand: int = 11
    spectrum_shape_expand_target: float = 1.5
    spectrum_shape_contract_floor: float = 0.7
    spectrum_shape_n_samples: int = 32
    # Stage-2 analogue of Stage1TrainingConfig.w_spectrum_shape_graded --
    # see that field's docstring for the full mechanism and motivation
    # (Section 134). Same file-path/once-per-epoch convention.
    w_spectrum_shape_graded: float = 0.0
    spectrum_shape_graded_reference_path: str | None = None
    spectrum_shape_graded_n_samples: int = 32
    # Stage-2 analogues of Stage1TrainingConfig.spectrum_shape_two_sided /
    # w_spectrum_shape_multistep -- see those fields' docstrings (Section
    # 186) for the full mechanism and motivation.
    spectrum_shape_two_sided: bool = False
    w_spectrum_shape_multistep: float = 0.0
    spectrum_shape_multistep_k: int = 10
    spectrum_shape_multistep_n_samples: int = 16
    # `w_spectrum_shape_self` (added 2026-09-23, Section 204, user-
    # directed: "I want to try the self rollout spectrum-shape
    # regularizer" -- the candidate flagged in docs/OPEN_QUESTIONS.md
    # after Section 203's rerun: `w_spectrum_shape` above, plus
    # `w_varmatch`/`w_spatial`/`w_logdet_rollout_latent`, all evaluated
    # on real, on-attractor states, still collapsed the MAIN propagator
    # to D_KY=0.00 on L96. Direct analogue of `w_pde_spectrum_shape_self`
    # (Section 193), which exists ONLY for the separate `pde_head`
    # distillation target -- this is the first time the SAME self-
    # rollout-sampled mechanism is applied to the propagator that is
    # actually cycled/decoded/used everywhere else in this project.
    # Motivation (see `w_pde_spectrum_shape_self`'s docstring for the
    # original diagnosis): `w_spectrum_shape` above is evaluated ONLY on
    # real, encoder-derived on-attractor states -- it shapes the
    # propagator's Jacobian correctly NEAR THE TRUE ATTRACTOR, but says
    # nothing about states the propagator's OWN free rollout actually
    # visits once it starts drifting toward a fixed point, which is
    # exactly the region a real-data-anchored loss never samples from.
    # Uses `_propagator_spectrum_shape_self_pool`
    # (`ks_latent.training.loops`): rolls the propagator forward
    # `spectrum_shape_self_rollout_k` steps under `torch.no_grad()` from
    # a real starting window, takes the FINAL state (or, `mode="history"`,
    # the final `n_hist`-length window) as the evaluation pool, then
    # applies `propagator_spectrum_shape_loss` there WITH gradient --
    # cheap (no backprop through the rollout chain itself), same "evaluate
    # AT a batch of self-visited points" convention as
    # `w_pde_spectrum_shape_self`. Shares `spectrum_shape_n_expand`/
    # `expand_target`/`contract_floor`/`two_sided` above (the same target
    # shape, just sampled from a different state distribution) rather
    # than duplicating them, matching how `w_pde_spectrum_shape_self`
    # reuses `pde_spectrum_shape_n_expand` etc. GENERALIZED to
    # `mode="history"` (unlike `w_spectrum_shape`, which is `mode=
    # "markovian"` only) -- `_propagator_spectrum_shape_self_pool`
    # flattens the trailing `(n_hist, d)` window to a single vector before
    # differentiating through `step_history`, mirroring `step_history`'s
    # own internal flatten for `backbone="mlp"`. `mode="two_step"` still
    # unsupported (same restriction as every other spectrum-shape loss in
    # this project). OFF by default (0.0); may be combined with
    # `w_spectrum_shape` (independent weights, real-data-anchored vs.
    # self-rollout-sampled, same as the pde_head pair).
    w_spectrum_shape_self: float = 0.0
    spectrum_shape_self_rollout_k: int = 20
    spectrum_shape_self_n_samples: int = 16
    # `pde_head` continuation into Stage 2 (added 2026-09-08, see
    # docs/sine_transform_pde_plan.md §21, user-directed: "is the pde_head
    # trained during phase 2 as well. we should try to get pde rollout to
    # have decent performance" then "please implement both option a and b
    # ... at the end of the day, we're really just training the propagator
    # right, the pde_head training is acting as a regularization term").
    # `pde_head` (a backbone="spectral_pde_raw" propagator) is passed
    # directly to `train_stage2`, not constructed from this config -- these
    # two weights only gate whether/how its loss is added, mirroring every
    # other optional term's convention here. DELIBERATELY MUTUAL (unlike
    # Stage 1's own w_pde_distill, whose target is detached): Stage 2 never
    # touches the encoder (already-frozen z), so a detached target would
    # leave nothing for this loss to regularize except pde_head in
    # isolation -- here gradient reaches BOTH propagator and pde_head. See
    # train_stage1's docstring for why Stage 1's own version stays
    # detached, and train_stage2's for the full mechanism of both options
    # below. Both OFF by default (0.0); either or both may be nonzero.
    #
    # w_pde_distill ("option A", safe): single-step match between
    # pde_head.step_one and propagator's own realized next state, evaluated
    # at EVERY position along propagator's own rollout -- pde_head's own
    # gradient is always a single-step regression, never backprop through
    # its own multi-step chain.
    w_pde_distill: float = 0.0
    # w_pde_rollout ("option B", riskier): pde_head's OWN autoregressive
    # k_now-step rollout (chained through pde_head itself) vs. propagator's
    # -- the direct approach that made backbone="spectral_pde" hard to
    # train as a PRIMARY propagator in Sections 101-106. Offered so both
    # can be tried and compared, per user direction.
    w_pde_rollout: float = 0.0
    # See `Stage1TrainingConfig.w_pde_distill_real`'s docstring for the
    # full mechanism -- here it uses the pre-computed, FROZEN real `z`
    # sequences already available in Stage 2 (`train_sequences`/windows
    # thereof), which have NO dependence on the propagator at all -- so
    # this term is completely isolated from whatever the propagator is
    # doing; it trains pde_head purely against real encoded transitions.
    # `pde_distill_real_detach` is a no-op in Stage 2 (the encoder is
    # already frozen and nothing else depends on `z_win`'s graph), kept
    # only for CLI/config symmetry with Stage 1.
    w_pde_distill_real: float = 0.0
    pde_distill_real_detach: bool = True
    # See `Stage1TrainingConfig.w_pde_distill_real_rollout`'s docstring
    # for the full mechanism (same `pde_head.rollout` chained-real-target
    # idea) -- here using Stage 2's own `windows`/`n_hist`, requires
    # `pde_distill_real_rollout_k <= k_max` (enforced at the top of
    # `train_stage2`).
    w_pde_distill_real_rollout: float = 0.0
    pde_distill_real_rollout_k: int = 4
    # See `Stage1TrainingConfig.pde_distill_real_rollout_warmup_epochs`'s
    # docstring -- SAME mechanism, same "found necessary empirically"
    # finding (a divergent first-batch rollout against an undertrained
    # pde_head poisons the whole loss and the optimizer's moment
    # estimates permanently). Stage 2's own `--init-pdehead-checkpoint`
    # may already be reasonably conditioned from Stage 1, but the ramp
    # costs nothing and removing the exact failure mode found empirically
    # is worth keeping unconditionally.
    pde_distill_real_rollout_warmup_epochs: int = 0
    # See `Stage1TrainingConfig.w_pde_nonlinear_l2`'s docstring for the
    # full iLED (arXiv:2309.05812) rationale -- identical mechanism here.
    w_pde_nonlinear_l2: float = 0.0
    # `w_pde_spectrum_shape`/`w_pde_spectrum_shape_multistep` (added
    # 2026-09-21, Section 192, user-directed: "maybe we can try our new
    # regularization methods with the joint propagator/pde training from
    # before? where the pde is not the propagator... please add the new
    # regularizer just to the pde_head"): pde_head analogues of
    # `w_spectrum_shape`/`w_spectrum_shape_multistep` (Sections 131/186,
    # applied to the MAIN propagator there) -- see those fields'
    # docstrings and `ks_latent.training.losses.propagator_spectrum_
    # shape_loss`/`propagator_multistep_spectrum_shape_loss` for the full
    # mechanism. Motivation: Section 167's own pde_head (masked_mlp_
    # expand as the real, DECOUPLED propagator -- already genuinely
    # chaotic, D_KY=22.67, dead center of the true 21-24 benchmark --
    # with pde_head fit separately against real data only) never
    # diverges standalone but DECAYS toward a fixed point (1.82 -> 0.28
    # over 200 steps) rather than sustaining chaos on its own. Applying
    # these regularizers to `pde_head.step_one` specifically (NOT the
    # main `aux`/propagator, which is already healthy and should not be
    # touched) gives pde_head direct pressure toward a genuinely
    # expansive-then-contracting Jacobian spectrum, independent of
    # whatever the real propagator is doing. Evaluated on real, detached
    # on-attractor states (same convention as `w_pde_nonlinear_l2`/`w_pde_
    # mean_conservation` above -- never mutual with the encoder; this is
    # a prior on pde_head's OWN closure alone). Requires `pde_head` to be
    # built (`--pde-distill`); raises via the same `kernel_Lhat`-style
    # `getattr`/mode checks the main-propagator version already uses.
    # OFF by default (0.0) for both.
    w_pde_spectrum_shape: float = 0.0
    pde_spectrum_shape_n_expand: int = 13
    pde_spectrum_shape_expand_target: float = 1.1
    pde_spectrum_shape_contract_floor: float = 0.6
    pde_spectrum_shape_n_samples: int = 32
    pde_spectrum_shape_two_sided: bool = False
    w_pde_spectrum_shape_multistep: float = 0.0
    pde_spectrum_shape_multistep_k: int = 10
    pde_spectrum_shape_multistep_n_samples: int = 16
    # `w_pde_spectrum_shape_self`/`w_pde_spectrum_shape_multistep_self`
    # (added 2026-09-21, Section 193, user-directed: "did you use the
    # rollout regularization for the pde_head where we tried to
    # discourage collapse and chaos by looking at the jacobian? maybe
    # that's something we should add to stage 2 or even stage 3 if we
    # train the pde on its own"). Gap in `w_pde_spectrum_shape`/`w_pde_
    # spectrum_shape_multistep` above: both are evaluated ONLY on real,
    # encoder-derived on-attractor states (z_win/windows) -- they shape
    # pde_head's Jacobian correctly NEAR THE TRUE ATTRACTOR, but say
    # nothing about states pde_head's OWN free rollout actually visits if
    # it drifts off the true manifold -- exactly the region where Section
    # 167's pde_head decayed toward a fixed point over its own 200-step
    # standalone rollout, a region the real-data-anchored version never
    # samples from. These two fields apply the SAME two losses
    # (`propagator_spectrum_shape_loss`/`propagator_multistep_spectrum_
    # shape_loss`, same shared `pde_spectrum_shape_n_expand`/`expand_
    # target`/`contract_floor`/`two_sided` targets above) but evaluated
    # at states drawn from `pde_head`'s OWN self-generated rollout
    # instead: roll `pde_spectrum_shape_self_rollout_k` steps forward
    # from a batch of real starting states (`torch.no_grad()`, cheap --
    # no backprop through the rollout chain itself, matching every other
    # once-per-epoch Jacobian regularizer's "evaluate AT a batch of
    # points" convention, not "backprop through how we got there"), take
    # the final (fully detached) state of each self-rollout as the
    # evaluation pool. A NaN/inf self-rollout (plausible mid-training, an
    # unsupervised free-running rollout of a partially-converged closure)
    # skips this batch's contribution entirely rather than poisoning the
    # whole loss -- same discipline as `w_pde_energy_floor`'s own
    # documented fix for exactly this failure mode. Independent, genuinely
    # separate weights from the real-data versions above (both may be on
    # at once) so the two can be attributed independently. OFF by default
    # (0.0) for both -- NOT yet launched; the decision to enable these was
    # deferred until Section 192's own real-data-only version's results
    # (Stage 1 + Stage 2 diagnostics) are in.
    w_pde_spectrum_shape_self: float = 0.0
    pde_spectrum_shape_self_rollout_k: int = 20
    pde_spectrum_shape_self_n_samples: int = 16
    w_pde_spectrum_shape_multistep_self: float = 0.0
    # See `Stage1TrainingConfig.w_pde_mean_conservation`'s docstring for
    # the full rationale (true KS's exact `int u dx` conservation law) --
    # identical mechanism here.
    w_pde_mean_conservation: float = 0.0
    # See `Stage1TrainingConfig.w_pde_energy_floor`'s docstring for the
    # full rationale and mechanism -- identical here. Same UNSUPERVISED
    # convention (no real target, no window-size constraint).
    w_pde_energy_floor: float = 0.0
    pde_energy_floor_gamma: float = 0.4
    pde_energy_floor_rollout_k: int = 30
    pde_energy_floor_warmup_epochs: int = 0
    # See `Stage1TrainingConfig.w_prop_energy_floor`'s docstring for the
    # full rationale (Sections 169/171's fixed-point collapse, w_varmatch
    # unable to see far enough ahead) -- identical mechanism here, applied
    # to `propagator.rollout` (Stage 2's own primary propagator).
    w_prop_energy_floor: float = 0.0
    prop_energy_floor_gamma: float = 0.4
    prop_energy_floor_rollout_k: int = 30
    prop_energy_floor_warmup_epochs: int = 0
    # See `Stage1TrainingConfig.stable_linear_lr_factor`'s docstring for
    # the full rationale -- identical mechanism here, applied to
    # `propagator.stable_linear_raw_parameters()` (Stage 2's own primary
    # propagator).
    stable_linear_lr_factor: float = 0.02
    # `w_kernel_unstable_floor`/`kernel_unstable_target_modes`/
    # `kernel_unstable_margin` (added 2026-09-16, Section 183, user-
    # directed: "we should also increase the regularizer that tries to
    # keep at least 13 unstable modes in the pde, or if that doesn't
    # exist, implement it" -- no such regularizer existed). Only has an
    # effect when the propagator exposes `kernel_Lhat()` (`spectral_
    # field_kind="forced_burgers"` with `spectral_burgers_kernel_
    # instability=True`). `kernel_unstable_floor_loss` (ks_latent.
    # training.losses) architecturally-independent-of-data hinge penalty
    # keeping the lowest `kernel_unstable_target_modes` Fourier modes
    # (excluding DC, always exactly 0) above `kernel_unstable_margin` in
    # `Lhat(k)` -- i.e. genuinely growing/unstable. `w_kernel_unstable_
    # floor=0.0` (default): off, unchanged behavior.
    w_kernel_unstable_floor: float = 0.0
    kernel_unstable_target_modes: int = 13
    kernel_unstable_margin: float = 0.0
    # See `Stage1TrainingConfig.w_z_lowpass`'s docstring for the full
    # mechanism (same `latent_self_spectrum_lowpass_loss` helper). Here
    # applied to the PROPAGATOR's own rolled-out `z_pred` (Stage 2 has no
    # encoder to apply the non-rollout `w_z_lowpass` term to). OFF by
    # default (0.0).
    w_z_lowpass_rollout: float = 0.0
    z_lowpass_rollout_power: float = 1.0
    z_lowpass_K: int | None = None
    z_lowpass_L: float | None = None
    # See `Stage1TrainingConfig.w_pde_coeff_l1`'s docstring for the full
    # rationale (same mechanism, same `pde_head_poly_coeffs` helper) --
    # added here 2026-09-10 for the same reason, applied to whichever
    # `pde_head` is continued into Stage 2. OFF by default (0.0).
    w_pde_coeff_l1: float = 0.0
    # See `Stage1TrainingConfig.pde_coeff_l1_linear_only`'s docstring.
    pde_coeff_l1_linear_only: bool = False
    # Added 2026-09-04, user-directed: "can we add a readout for
    # val_kequals12_mse, so we can compare these runs more directly" --
    # `eval_stage2_kmax` pools its MSE over EVERY step of the k_max-length
    # rollout (`z_pred`/`z_true` both `(B, k_max, d)`, one `.mean()` over
    # batch+k+d together), so runs at different `k_max` are not directly
    # comparable: KS's chaotic error growth (Section 75 measured
    # lambda1~=0.091/step, an ~11-step e-folding time) means a `k_max=20`
    # run's pooled MSE includes 8 extra, already-saturated late steps a
    # `k_max=12` run's never sees, inflating it even for an equally good
    # model. `compare_k` (`None` default, off, unchanged behavior): when
    # set, ALSO pools the MSE over just the first `compare_k` steps of
    # that SAME rollout (no extra rollout computed) and logs/prints it as
    # `val_k{compare_k}_mse` alongside the full `val_kmax_mse` -- e.g.
    # `compare_k=12` on a `k_max=20` run gives a number directly
    # comparable to another run's own `k_max=12` `val_kmax_mse`. Must be
    # `<= k_max` if set.
    compare_k: int | None = None
    seed: int = 0

    def __post_init__(self):
        if self.latent_loss not in ("l1", "l2"):
            raise ValueError(f"latent_loss must be 'l1' or 'l2', got {self.latent_loss!r}")
        if self.compare_k is not None and not (1 <= self.compare_k <= self.k_max):
            raise ValueError(f"compare_k={self.compare_k!r} must be in [1, k_max={self.k_max!r}]")
        if self.k_mid is not None and not (2 <= self.k_mid <= self.k_max):
            raise ValueError(f"k_mid={self.k_mid!r} must be in [2, k_max={self.k_max!r}]")
        if self.k_mid is not None and not (0 <= self.k_mid_epochs <= self.k_warmup_epochs):
            raise ValueError(
                f"k_mid_epochs={self.k_mid_epochs!r} must be in [0, k_warmup_epochs={self.k_warmup_epochs!r}]"
            )
        if self.w_varmatch < 0:
            raise ValueError(f"w_varmatch must be >= 0, got {self.w_varmatch!r}")
        if self.w_logdet_rollout_latent < 0:
            raise ValueError(
                f"w_logdet_rollout_latent must be >= 0, got {self.w_logdet_rollout_latent!r}"
            )
        if self.w_spectrum_shape_graded > 0 and self.spectrum_shape_graded_reference_path is None:
            raise ValueError(
                "w_spectrum_shape_graded > 0 requires spectrum_shape_graded_reference_path "
                "to be set (a .npy file of shape (d_latent,), typically from "
                "scripts/compute_reference_spectrum.py)"
            )
        if self.w_logdet_rollout < 0:
            raise ValueError(f"w_logdet_rollout must be >= 0, got {self.w_logdet_rollout!r}")
        if self.w_spectrum_shape_self < 0:
            raise ValueError(f"w_spectrum_shape_self must be >= 0, got {self.w_spectrum_shape_self!r}")


@dataclass(frozen=True)
class RegConfig:
    """Optional banded latent-index-smoothness + off-band decorrelation
    regularizer (added 2026-08-29, ported from the same reference
    implementation as the training-config fields above; see
    `ks_latent/training/regularizer.py` for the full derivation).

    Off by default (`lambda_z=0.0`, `lambda_decorr=0.0`) -- this is a new
    optional capability, not a change to the canonical recipe. Its point is
    orthogonal to `Stage1TrainingConfig`'s existing `w_decorr`/`w_var`
    terms: those push the latent covariance toward the identity (full
    decorrelation, unit variance) but are exactly permutation-invariant in
    the latent index, so they cannot give the index itself any meaning. This
    regularizer instead penalizes `z^T B z` for a banded `B`, pulling
    coordinates that are *near in index* toward similar values -- literally
    the missing ingredient the codebase's own Phase 10 backlog item ("give
    the flat latent index physical/spatial meaning") is waiting on -- with a
    companion off-band decorrelation term that keeps that pull from
    collapsing every coordinate to the same value (see docstring in
    `regularizer.py` for why the banded term alone has that degenerate
    global optimum, and measured evidence of it happening in the reference
    project at `lambda_z >= 5e-3`).
    """

    lambda_z: float = 0.0
    lambda_decorr: float = 0.0
    bandwidth: int = 3
    decay: float = 0.5
    apply_to_propagated: bool = True
    # Delayed activation (added 2026-09-02, user-directed: "only add it
    # after the first 30 epochs"). Default 0 = active from epoch 0
    # (unchanged behavior). When > 0, `train_stage1` skips this penalty
    # entirely for epochs `< start_epoch`, then applies it normally
    # (uniform weight, no ramp) from `start_epoch` on -- i.e. the encoder
    # trains unconstrained by this term at first and only starts being
    # pulled toward index-adjacent smoothness partway through training.
    start_epoch: int = 0
