# Part 4.3 (latent-space localization for DA): results summary

*Prepared 2026-09-25 for discussion with Peter Jan. Full detail in
`docs/RESULTS.md` and `docs/steps_4-3.md`; novelty check in
`docs/LITERATURE_REVIEW_AND_FINDINGS.md` Part 4.3.*

## TL;DR

**The headline claim (local latent field + Gaspari-Cohn beats SEC at
the trained domain size) failed under our own pre-registered test.**
What survived, and is genuinely new and cleanly demonstrated, is
narrower: **the local-field architecture transfers to a larger domain
with zero retraining, and at that larger domain, localization stops
being merely helpful and becomes load-bearing** — something neither
comparator (a global latent vector, or SEC) can do at all. That's a
real result, but on its own it's a demonstration of a mechanism, not
yet a complete paper. See "Is this publishable?" below for the honest
version of that question, and "Where I'd put more effort" for what
would close the gap.

## What was tested

**The claim** (`docs/LITERATURE_REVIEW_AND_FINDINGS.md` Part 4.3): a
genuinely spatial ("local field") latent representation, instead of a
flat vector, should let ensemble/particle-flow DA localize the way
classical LETKF does — required ensemble size scaling with *local*
per-site dimension rather than the full attractor dimension. Novelty
check: three candidate prior-art papers (Chandravamsi et al.,
Guerrieri et al., Pasmans et al.) read in full and ruled out — none
localize a *learned* latent representation. Not a substitute for a
professional search before submission.

**Pre-registered decision rule**: local-field + Gaspari-Cohn must beat
SEC (Anderson 2012, a real distance-free empirical-localization
method, not a strawman) — not merely beat no localization.

**What we built to test it**: two validated KS (`L=100`) checkpoints —
a global-vector one (ViT encoder, `d_latent=44`) and, after a long
architecture search (`docs/RESULTS.md`), a local-field one
(`n_sites=16, local_channels=3, d_latent=48`, `local_mlp` propagator)
— genuinely bounded 2000-step standalone chaos, `D_KY=22.14` against a
true target of `21–24`, best dynamical-coupling-locality score (D3
bandedness) of any checkpoint produced this investigation. Gaspari-Cohn
taper infrastructure, an SEC baseline sweep driver, and a mechanically-
applied decision-rule script (unit-tested before trusting on real
data).

## Results

**Phase 7/13 core experiment, `L=100`**: SEC sweep on the global
latent vs. Gaspari-Cohn sweep on the local field, `N_ens` in
`{8,...,256}`.

| N_ens | GLOBAL+SEC | LOCAL+Gaspari-Cohn | decision rule |
|---|---|---|---|
| 8 | 0.506 | 0.530 | FAIL |
| 32 | 0.372 | 0.421 | FAIL |
| 128 | 0.329 | 0.405 | FAIL |
| 256 | 0.330 | 0.405 | FAIL |

**0 PASS, 6 FAIL, at every ensemble size tested.** SEC wins throughout,
by 5–25%, and the gap widens at large `N_ens`. Gaspari-Cohn *does*
dramatically help the local latent over no localization at all
(`N=8`: rmse `1.65→0.53`) — the mechanism works, it's just not enough
to catch a strong SEC baseline at the domain size both were fit to.

**L-transfer, zero retraining** (the local field's structural
advantage no comparator shares): the same `L=100`-trained weights,
loaded at `L=200` (2×) and `L=400` (4×) by re-deriving `n_sites`/`NX`
to hold physical resolution fixed —

- Energy spectrum peaks at the correct, domain-independent wavenumber
  at both sizes.
- `D_KY/L` extensivity holds tightly: `0.2214` (trained) → `0.2231`
  (`L=200`) → `0.2228` (`L=400`), all near KS's true extensivity
  constant (`~0.226`).
- **At `L=200`, with `N_ens` held at the same value that worked fine
  at `L=100`: unlocalized DA actively breaks** (skill `0.69`, worse
  than just running the model with no correction — the raw latent
  dimension doubled along with the domain). **Gaspari-Cohn — same
  radius parameter, unchanged — rescues it to real positive skill
  (`1.34`)**, more than doubling the RMSE improvement.
- Confirmed directly, not just architecturally asserted: feeding
  correctly-resolved `L=200` data into the global checkpoint's
  fixed-size encoder raises an immediate shape-mismatch crash. SEC has
  no size-dependent object to even attempt transferring.

## Is this publishable?

**Not as originally scoped.** The proposal's own headline claim — beat
SEC at a fixed domain size — is a clean negative result under our own
pre-registered rule. A paper built around "local + Gaspari-Cohn beats
SEC" doesn't exist in this data.

**There is a narrower, real claim that does hold up**, and it's
exactly what Part 4.3's own text predicted a negative result would
leave standing: *SEC cannot transfer to a new domain size at all; the
local field can, with zero retraining, and localization is what makes
that transfer usable rather than merely interesting.* That's a
genuine, clean, quantitatively demonstrated mechanism. But as it
stands it's one data point at one new size (`L=200`, one `N_ens`, one
`gc_c`) plus a qualitative confirmation at `L=400` — a demonstration,
not yet a swept, defended result. I don't think it clears the bar for
submission yet, but I think it's the most promising thread we have,
and closing the gaps below is a bounded amount of work, not a new
research direction.

**Weaknesses a reviewer would raise immediately:**
1. The `L=200` DA result is a single illustrative point, not a sweep —
   Phase C/D's own rigor (full `N_ens` curve, `gc_c` selection
   cross-checked against a theoretical light-cone bound) hasn't been
   repeated at the new size.
2. SEC gets a freshly-fit table *per `N_ens`* at `L=100`; Gaspari-Cohn
   used one fixed radius across the whole sweep. Not obviously unfair
   (`gc_c` was chosen from a real physical bound, not tuned per point),
   but a reviewer will ask whether Gaspari-Cohn was given its best shot.
3. Everything here is KS only. The literature review's own Part 3.4
   argues the project's strongest publishable *structure* is a
   multi-system ladder (KS → L96 → RBC), not a single-system result.
4. Only one architecture (`local_mlp`) and one training recipe were
   swept to find a working local-field checkpoint — the search that
   got us here (`docs/RESULTS.md`) tried and discarded many
   configurations before finding one that worked, which is normal but
   means the result isn't yet shown to be robust to that choice.

## Where I'd put more effort, ranked

1. **Fit SEC fresh on the local latent AT `L=200`** (cheap — SEC only
   needs a new empirical table, not retraining the model) and compare
   against Gaspari-Cohn-on-local at `L=200`. This is the single most
   informative missing experiment: it asks whether Gaspari-Cohn's
   *physical* prior is what's earning its keep at the new size, or
   whether any localization — even a naive one — would do, which
   directly determines how strong a claim we can make.
2. **Run the full `N_ens` sweep at `L=200`** (and maybe `L=400`),
   not just one point, so the "localization becomes necessary" claim
   is a curve, not an anecdote.
3. **Let `gc_c` vary with `N_ens`** at `L=100` (untested) — SEC's own
   comparator is inherently `N_ens`-adaptive; a fixed-radius Gaspari-
   Cohn may be leaving performance on the table specifically at the
   ensemble sizes where it currently loses most.
4. **If this is going in a paper at all, it likely needs the L96 rung**
   of the ladder (already has a validated local-field checkpoint per
   earlier work) to show the L-transfer mechanism isn't a KS
   idiosyncrasy.

## Other directions from the literature review, if 4.3 doesn't pan out

Two other candidates were identified as "still open" alongside 4.3 in
`docs/LITERATURE_REVIEW_AND_FINDINGS.md` Part 3, ranked there below
4.3 but not investigated further this session:

- **Spectral-gap criterion as a quantitative predictor** (§3.2): does
  a measurable property of a system's true spectral gap predict how
  much propagator memory (`n_history`) is needed for a small latent
  model to recover chaos? One suggestive KS-vs-L96 data point exists;
  needs a systematic `n_history` sweep across several L96 forcing
  values to become a real result. Flagged as needing a Mori-Zwanzig/
  optimal-prediction literature check first (a memory-kernel-decay
  version of this idea may already exist under different vocabulary).
- **Stage-2 chaos collapse as a general critique of k-step MSE
  training** (§3.3): the phenomenon (multi-step MSE training destroys
  measured chaos while *improving* forecast skill) is demonstrated
  across two systems and several architectures — a broad empirical
  base — but the proposed fix (`--multistep`) is itself unresolved
  (`docs/OPEN_QUESTIONS.md`), so there's a validated problem but not
  yet a validated fix. Needs either the queued isolation experiment
  run on L96 too, or a genuine comparison against a non-MSE loss
  family (CRPS/energy score).

Neither has had anywhere near the investigation Part 4.3 got this
session — raising them mainly so the choice not to pursue them further
is a decision, not an accident.

## Would this work for Rayleigh-Bénard, and toward NCAR's CM1?

**RBC's extensivity is the real open question, not a detail to defer.**
KS's whole L-transfer story rests on one clean, measured scaling law
(`D_KY ~ 0.226*L`). This project's RBC work doesn't have that yet —
there is no validated, Lyapunov-spectrum-matched chaotic baseline for
RBC at all (`docs/OPEN_QUESTIONS.md`: RBC's own chaos regime is still
unresolved — a solver exists, the third rung of the KS→L96→RBC ladder
does not). That's the actual prerequisite before asking whether
localized DA works on it, not a formality to skip.

**If RBC does exhibit horizontally-extensive chaos at fixed Rayleigh
number/gap (plausible, not yet checked against this project's own
solver), the growth axis has to be the horizontal one specifically.**
Plate separation sets Ra — it is not a "make the domain bigger" knob
the way KS's `L` is; the periodic horizontal extent is. That is
actually a good structural fit for "train small, deploy large": a
wider convecting domain (and eventually a real cloud field) grows
horizontally while the vertical column structure per site stays a
fixed-size local encoding — directly analogous to `local_channels` per
site here, just needing more of them than KS's 2–3, since a full
temperature + velocity vertical profile carries more structure than a
scalar coarse-grained average.

**What it would actually take**, not a parameter change on the current
code: a genuinely 2D local field (`Conv2d`, circular in the periodic
horizontal axis, a bounded — not circular — treatment in the vertical),
a gauge-anchor convention re-derived for a multi-field state (T, u, w,
not one scalar), and Phase 2's own spreading-velocity/receptive-field
measurement redone for RBC's own correlation length (which likely
depends on Ra, not a fixed geometric constant the way KS's `k~1/sqrt(2)`
is). Then the whole Phase A–F arc this document summarizes would need
repeating on that new system.

**CM1 is a much bigger jump than RBC is from KS, and I'd resist
treating it as a short hop.** CM1 is compressible/anelastic, moist,
3D, with real microphysics and surface/radiation forcing — RBC is
Boussinesq, idealized, and dry. RBC is a reasonable *structural*
stepping stone (same "extensive horizontal chaos over a bounded
vertical column" shape), but the physics gap from RBC to CM1 is larger
than the gap from KS to RBC. I'd treat it as its own rung, earning its
own validated chaotic baseline before the localization question is
asked of it, the same caution this project applied to KS before
trusting any DA result off of it.

**Net read**: worth pursuing, and the localization mechanism is a
genuinely plausible fit for RBC's specific geometry — but this is a
multi-step research program on its own (validate RBC's chaotic
attractor → build the 2D local-field encoder/decoder → remeasure the
light-cone bound → rerun this session's Phase A–F arc), not an
extension of the current KS result.
