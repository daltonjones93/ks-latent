"""Part II Table tab:da_model_error (Block K2c, 2026-10-10): learned vs physics-based filters under model error.

Per case: the physics model's one-cycle error (total / state-dependent, from j16_model_error_<case>.json, an
independent seed-3 trajectory); the physics filter tuned only (RTPS, localization) and its best configuration
including remedies (online parameter estimation, bias correction +- additive noise). Selection follows the
pre-registered rule: the configuration with the lowest seed-0 error among those run on seeds 0-2; a dagger marks
a case where a seed-0-only configuration was lower still (seeds 1-2 pending). Outcome per the rule: a filter wins
if its 3-seed mean is >= 10% lower and it is lower on every seed; otherwise tie.
Latent filter: p7c12d10_ks2d, seeds 0-2 (0.324 / 0.310 / 0.302); for the hidden-field cases (J17B) the latent
model trained on that system (p7c12d10_memB<eps>), compared in its own field units.
Writes tables/tab07_p2_model_error.tex (+ .meta.json) and da2d/model_error_summary.json (for Fig. 14).
"""
import glob
import json
import math
import re

import numpy as np

from _style import EXP, ROOT, write_table

D = EXP / "da2d"
inputs = []


def ana(tag):
    p = D / f"i11_{tag}.json"
    if not p.exists():
        p = D / f"h9_{tag}.json"
        if not p.exists():
            return None
        inputs.append(str(p.relative_to(ROOT)))
        s = json.loads(p.read_text())["steady"]
        return s.get("blend")
    inputs.append(str(p.relative_to(ROOT)))
    e = json.loads(p.read_text())["steady"].get("analysis")
    return e if e is not None and math.isfinite(e) else None


def seeds(tags):
    v = [ana(t) for t in tags]
    return v if all(x is not None for x in v) else None


def j17(prefix, sub):  # J17H/J17B configs: {name: [s0, s1, s2]} from the file names
    out = {}
    for f in glob.glob(str(D / f"i11_{prefix}_*_s0.json")):
        base = f.split("i11_")[1][:-8]
        if not re.search(sub, base):
            continue
        out[base] = [f"{base}_s{k}" for k in range(3)]
    return out


LAT = [0.324, 0.310, 0.302]
CASES = [  # label, one-cycle file, tuned configs, remedy configs (name -> seed tags)
    ("true equations", None, {"c24": ["L44_st8_c24_rtps0"] * 3}, {}),
    ("$64^2$ grid", "M64", {"r0": ["I28R_M64_pde_rtps0", "J1_M64_rtps0_s1", "J1_M64_rtps0_s2"]}, {}),
    ("$32^2$ grid", "M32", {"r0.6": ["I28R_M32_pde_rtps0.6", "J1_M32_rtps0.6_s1", "J1_M32_rtps0.6_s2"]},
     {"bc": [f"J16b_M32_rtps0.6_bc_q0_s{k}" for k in range(3)]}),
    ("forcing $0.2$", "frc0.2", {"r0.7": ["I28W_frc0.2_rtps0.7", "J1_frc0.2_rtps0.7_s1", "J1_frc0.2_rtps0.7_s2"]},
     {"bc": [f"J16b_frc0.2_rtps0.7_bc_q2_s{k}" for k in range(3)]}),
    ("forcing $0.4$", "frc0.4", {"c16": ["I28W_frc0.4_rtps0.7", "J1_frc0.4_rtps0.7_s1", "J1_frc0.4_rtps0.7_s2"],
                                 "c8.25": ["J2_frc0.4_c8.25_rtps0.7", "J1b_frc0.4_c8.25_rtps0.7_s1", "J1b_frc0.4_c8.25_rtps0.7_s2"]},
     {"bc": [f"J16b_frc0.4_rtps0.7_bc_q0.5_s{k}" for k in range(3)]}),
    ("$\\nabla^2$ coefficient $1.1$", "lap1.1", {"r0.7": ["I28W_lap1.1_rtps0.7", "J1_lap1.1_rtps0.7_s1", "J1_lap1.1_rtps0.7_s2"]},
     {"est.": [f"J16a_lap1.1_rtps0.7_s{k}" for k in range(3)]}),
    ("damping order $5$", "hyp5", "J17H_p5", None),
    ("damping order $6$", "hyp6", "J17H_p6", None),
    ("$\\nabla^2$ coefficient $1.25$", "lap1.25", {"r0.8": ["I28W_lap1.25_rtps0.8", "J1_lap1.25_rtps0.8_s1", "J1_lap1.25_rtps0.8_s2"]},
     {"est.": [f"J16a_lap1.25_rtps0.8_s{k}" for k in range(3)]}),
    ("damping order $8$", "hyp8", "J17H_p8", None),
    ("advection $1$", "adv1", {"c16": ["I28W_adv1_rtps0.9", "J1_adv1_rtps0.9_s1", "J1_adv1_rtps0.9_s2"],
                               "c8.25": ["J2_adv1_c8.25_rtps0.9", "J1b_adv1_c8.25_rtps0.9_s1", "J1b_adv1_c8.25_rtps0.9_s2"]},
     {"est.": [f"J16a_adv1_rtps0.9_s{k}" for k in range(3)], "est. ": [f"J16a_adv1_c8.25_rtps0.9_s{k}" for k in range(3)]}),
    ("advection $2$", "adv2", {"c16": ["I28W_adv2_rtps1.0", "J1_adv2_rtps1.0_s1", "J1_adv2_rtps1.0_s2"],
                               "c8.25": ["J2_adv2_c8.25_rtps1.0", "J1b_adv2_c8.25_rtps1.0_s1", "J1b_adv2_c8.25_rtps1.0_s2"]},
     {"est.": [f"J16a_adv2_rtps1.0_s{k}" for k in range(3)], "est. ": [f"J16a_adv2_c8.25_rtps1.0_s{k}" for k in range(3)]}),
]


def remedy_name(base):
    if "abg" in base and "bc" in base: return "est.+bc"
    if "abg" in base or "_lap" in base and "J17B" in base: return "est." + ("+bc" if "bc" in base else "")
    if "bc" in base: return "bc"
    return "tuned"


def pick(cfgs):
    best, flag = None, False
    for name, tags in cfgs.items():
        v = seeds(tags)
        s0 = ana(tags[0])
        if v is not None and (best is None or v[0] < best[1][0]):
            best = (name, v)
    for name, tags in cfgs.items():
        s0 = ana(tags[0])
        if s0 is not None and best is not None and seeds(tags) is None and s0 < best[1][0] - 1e-9:
            flag = True
    return best, flag


def outcome(phys, lat):
    mp, ml = np.mean(phys), np.mean(lat)
    if mp >= 1.1 * ml and all(p > l for p, l in zip(phys, lat)):
        return f"latent ${mp / ml:.2f}\\times$"
    if ml >= 1.1 * mp and all(l > p for p, l in zip(phys, lat)):
        return f"physics ${ml / mp:.2f}\\times$"
    return f"tie (${mp / ml:.2f}$)"


def fmt(v):
    return "---" if v is None else (f"${np.mean(v):.3f}$" + ("" if len(set(v)) == 1 else f" (${min(v):.3f}$--${max(v):.3f}$)"))


rows = ["% Generated by scripts/paper_figures/tab07_p2_model_error.py -- do not edit by hand."]
summary = []
for lab, oc, tuned, rem in CASES:
    if isinstance(tuned, str):  # J17H
        allc = j17(tuned, ".")
        tuned = {k: v for k, v in allc.items() if not re.search("abg|bc", k)}
        rem = {k: v for k, v in allc.items() if re.search("abg|bc", k)}
    one = "---"
    stdp = None
    if oc:
        p = D / f"j16_model_error_{oc}.json"
        if p.exists():
            inputs.append(str(p.relative_to(ROOT)))
            j = json.loads(p.read_text()); one = f"${j['residual_rms']:.3f}$ / ${j['sigma']:.3f}$"; stdp = j["sigma"]
    else:
        one = "$0$ / $0$"; stdp = 0.0
    t, tf = pick(tuned)
    b, bf = pick({**tuned, **rem})
    rname = "---" if b is None or b[0] in tuned else remedy_name(b[0]) if b[0].startswith("J17") else b[0].strip()
    out = outcome(b[1], LAT) if b else "---"
    if lab == "true equations":
        out = "floor"
    dag = "$^\\dagger$" if (tf or bf) else ""
    rows.append(f"{lab} & {one} & {fmt(t[1] if t else None)} & {fmt(b[1] if b else None)}{dag} & {rname} & {out} \\\\")
    summary.append({"case": lab, "one_cycle_state": stdp, "tuned": t and t[1], "best": b and b[1], "remedy": rname, "outcome": out})

# J17B: hidden slow field, own latent model and units
for eps in ("0.1", "0.2"):
    allc = j17(f"J17B_e{eps}_pde", ".")
    tuned = {k: v for k, v in allc.items() if not re.search("lap|bc", k)}
    rem = {k: v for k, v in allc.items() if re.search("lap|bc", k)}
    lat = seeds([f"J17B_e{eps}_lat_T4_s{k}" for k in range(3)])
    flo = seeds([f"J17B_e{eps}_floor_{c}" for c in [re.sub(r"^J17B_e[0-9.]+_floor_", "", f.split("i11_")[1][:-8])
                                                    for f in glob.glob(str(D / f"i11_J17B_e{eps}_floor_*_s0.json"))][:1] for c in [c + f"_s{k}" for k in range(3)]]) if glob.glob(str(D / f"i11_J17B_e{eps}_floor_*_s0.json")) else None
    p = D / f"j16_model_error_memB{eps}.json"
    one = "---"
    if p.exists():
        j = json.loads(p.read_text()); one = f"${j['residual_rms']:.3f}$ / ${j['sigma']:.3f}$"
    t, _ = pick(tuned); b, _ = pick({**tuned, **rem})
    if b and lat:
        out = outcome(b[1], lat).replace("latent", "latent").replace("physics", "physics")
        out += f"; latent ${np.mean(lat):.3f}$"
    else:
        out = "\\placeholder{P7}"
    rows.append(f"hidden field $\\varepsilon={eps}$ & {one} & {fmt(t[1] if t else None)} & {fmt(b[1] if b else None)} & "
                f"{'---' if not b or b[0] in tuned else remedy_name(b[0])} & {out} \\\\")
    summary.append({"case": f"hidden field eps={eps}", "latent": lat, "floor": flo, "tuned": t and t[1], "best": b and b[1]})

write_table("tab07_p2_model_error", "\n".join(rows) + "\n", inputs)
(D / "model_error_summary.json").write_text(json.dumps(summary, indent=2))
print("\n".join(rows))
