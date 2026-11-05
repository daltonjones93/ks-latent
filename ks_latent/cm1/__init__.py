"""Latent local-field models for the CM1 2D squall line (Paper III).

The KS / Kolmogorov local latent field, adapted to an x-z cloud-model slice:
each x-column (variables x levels) is the channel vector at one grid cell,
locality is horizontal only, and boundaries are open (no periodic padding).
See docs/cm1_notes.md for the data findings that set the design.
"""
