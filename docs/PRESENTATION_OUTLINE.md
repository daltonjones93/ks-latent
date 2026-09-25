# Presentation outline: latent-space localization for DA (for Peter Jan)

Slide-by-slide plan. Each entry: title, the 3-4 things it must say, and
the visual it should carry. Build HTML slides from this one at a time.
Target ~13 content slides + title — tight, not exhaustive.

---

**1. Title**
"Localized Latent Fields for Data Assimilation: KS, Lorenz-96, and a Path to Weather"
Subtitle: what we set out to test, one line. Date, your name.

**2. The research question**
- Learned reduced-order models for chaotic PDEs are cheap but usually a flat vector — no spatial structure, no localization, no domain-size transfer.
- Question: does a genuinely *spatial* latent field fix all three at once?
- Visual: global vector vs. local field, side-by-side schematic (one box vs. a row of boxes).

**3. Two architectures, one key difference**
- Global: ViT/MLP encoder → single `d_latent` vector, `d=44` for KS.
- Local: circular-conv encoder → `(n_sites, local_channels)` field, site-major, one physically-anchored channel per site.
- Visual: diagram of the local_field encoder (patchify → circular conv stack → anchor+residual channels).

**4. The architecture search: what it took to get a stable local propagator**
- Many backbones tried: `masked_mlp`/`masked_mlp_expand` (locally masked dense layers) — repeatedly unstable despite 4+ regularizer attempts, and later shown structurally non-transferable anyway (params tied to `d_latent`).
- `local_mlp` (genuine weight-shared conv token-mixer) — first attempt that fully worked.
- One-line moral: architectural locality (real weight sharing) beat regularized locality (masks on a dense layer) on every axis.
- Visual: small table, backbone vs. outcome (bounded? L-transferable? best D3?).

**5. Training/regularization lessons**
- Log-barrier growth penalty > squared-hinge penalty for preventing blowup (barrier repels approach, hinge only reacts after crossing).
- "Always run Stage 2" — Stage-1-only chaos checks are unreliable; only the Stage-2, standalone 2000-step rollout is trustworthy.
- Smaller `d_latent` (48 vs 96) beat larger on every metric — less redundant coordinate budget forced a cleaner representation.
- Visual: the error-growth / physical Hovmöller plot for 224 (already generated, `docs/figures/`).

**6. Section 224: the validated local-field checkpoint**
- Numbers: `D_KY=22.14` (target 21-24), bounded the *entire* 2000-step rollout, best dynamical-locality score (D3=0.43) of any checkpoint this project has produced.
- Visual: the 4-panel summary figure (`artifacts/figures/section224_summary.png` equivalent) — rollout stability, D3 coupling heatmap, spacetime plot, Lyapunov spectrum.

**7. Headline result: zero-retraining domain transfer**
- Same trained weights, loaded at 2x and 4x the trained domain size (`L=200`, `L=400`) — zero retraining.
- Energy spectrum peak matches; `D_KY/L` extensivity holds tightly (0.221 → 0.223 → 0.223).
- Frame as: this is the one thing the global-vector model *cannot even attempt* (confirmed via an actual shape-mismatch crash).
- Visual: `D_KY/L` bar chart across the three sizes, flat/consistent.

**8. DA result: when does localization actually help?**
- At `L=100`: local + Gaspari-Cohn does NOT beat SEC (0/6 on the pre-registered decision rule) — a clean negative on the headline DA claim.
- At `L=200` (new, zero retraining): unlocalized DA *breaks* (skill 0.69, worse than free-running); Gaspari-Cohn rescues it to skill 1.34.
- The honest framing: localization's value is domain-size-dependent, and only the local field can even be tested at the larger size.
- Visual: the RMSE-vs-N_ens sweep plot (`docs/figures/phase4_3_decision_rule_sweep.png`).

**9. Literature review: what's actually publishable**
- Novelty check done (3 candidate papers read in full, ruled out) — real gap exists in ML+DA literature.
- Honest verdict: not publishable as originally scoped (the SEC comparison is negative); the transfer+localization-becomes-necessary result is real but currently one data point, not a swept result.
- Two other candidate directions flagged in the lit review, not yet pursued: spectral-gap criterion, Stage-2 chaos-collapse critique.
- Visual: none needed, or a simple traffic-light table (claim → publishable now? / needs more work / dead end).

**10. Where current methods struggle: Lorenz-96**
- L96 has no spectral gap (unlike KS) — a small Markovian latent doesn't recover chaos on it the way it does on KS.
- Suggestive finding: adding propagator memory (history) recovers most of the missing chaos — points at gap-presence as the predictive variable, not yet a validated quantitative law.
- Frame as: the KS result may not generalize for free — this is the concrete open question standing in the way.
- Visual: simple 2-system comparison table (KS vs L96: gap? memory needed? chaos recovered?).

**11. A promising side-result: interpretable local closures**
- Fit a shared, local, quadratic stencil law to 224's latent (Ridge): 90% of the dynamics explained at full global width, vs 82% for a less-local checkpoint — real, measurable structure.
- SINDy-style sparse selection: doesn't collapse to a handwritten equation, but a genuine ~5x compression exists, and the surviving terms are mechanistically informative (physical channel needs the learned hidden channels — real closure memory).
- Frame as: a second possible publishable angle, orthogonal to the DA question.
- Visual: the sparsity-path chart (R² vs. #terms).

**12. Extending to real weather: RBC as a stepping stone**
- Why RBC: same "extensive chaos along a periodic axis" shape, closer to a real cloud field than KS.
- What's missing: RBC has no validated chaotic baseline in this project yet — that's the actual prerequisite, not a formality.
- The growth axis has to be horizontal (plate separation sets Ra, isn't a free "bigger domain" knob) — good structural match for train-small/deploy-large.
- Path: validate RBC chaos → 2D local field (circular horizontal, bounded vertical) → repeat this whole arc → then NCAR CM1 as its own, much bigger rung.
- Visual: a 3-rung ladder diagram (KS → RBC → CM1), each with its own validation gate.

**13. Summary and recommendation**
- What's solid: architecture + training recipe, L-transfer mechanism (novel, demonstrated).
- What's not yet publishable: the DA-beats-SEC claim, as scoped.
- Recommended next step(s): pick one — (a) fit SEC fresh on the local latent at `L=200` for a fair localization comparison at the size where it matters, or (b) pursue the interpretable-PDE angle further, or (c) start the RBC validation baseline.
- Visual: none — text summary slide, maybe icons per bullet.

**14. (Optional backup) Key numbers appendix**
- One dense table: every checkpoint (211, 216, 221-224) with D_KY, D3, bounded?, L-transferable?, DA skill. For Q&A, not the main flow.
