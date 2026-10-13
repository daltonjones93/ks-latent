"""Appendix A table (task A8): architecture and training configuration of every main-text model.

Sources, per field:
  * 2D models (crop-trained): everything from the checkpoints themselves (`ae_config`,
    `crop_ae_config`, `prop_config`, `crop_prop_config`, `train_dataset`) plus the
    training-set shape printed on the first line of artifacts/logs/crop_train_<tag>.log.
  * 1D model (Section 224): architecture from the checkpoints; training settings are not
    stored in them, so they come from the launch script
    (scripts/section224_ks_localfield_d48_localmlp.sh, explicit flags) and, for every setting
    that script leaves at its default, from Stage1TrainingConfig / Stage2TrainingConfig in
    ks_latent/config.py (the full profile overrides only the fields the script passes).
    The Stage-1 learning rate (1e-3) is confirmed by the training log.
Writes docs/paper1/tables/tabA_architecture.tex and artifacts/experiments/model_configs.json
(the full resolved configs with a source string per model).
"""
import json
import re

import torch

from _style import ART, EXP, MAIN_A01, ROOT, write_table
from ks_latent.config import Stage1TrainingConfig, Stage2TrainingConfig

MODELS_2D = [("2D KS", "p7b_ks2d"), ("2D KS, Part~II", "p7c12d10_ks2d"), ("Kolm., $\\alpha=0.1$", MAIN_A01),
             ("Kolm., $\\alpha=0.05$", "c19_e0d0_c6"), ("Kolm., $\\alpha=0.025$", "k_xs0025"),
             ("Kolm., Part~II", "k_xs005_c12d10")]  # R8: the Kolmogorov DA model of Paper II, Sec. 10


def load(p):
    return torch.load(ART / p, map_location="cpu", weights_only=False)


def as_dict(cfg):
    return {k: v for k, v in vars(cfg).items()} if not isinstance(cfg, dict) else cfg


def train_shape(tag):
    line = next(l for l in (ROOT / f"artifacts/logs/crop_train_{tag}.log").read_text().splitlines()
                if l.startswith("[crop]") and "train (" in l)
    m = re.search(r"train \((\d+), (\d+), (\d+), (\d+)\) val \((\d+),", line)
    return {"n_train": int(m[1]), "n_snapshots": int(m[2]), "grid": (int(m[3]), int(m[4])), "n_val": int(m[5])}


cols, record = {}, {}
# ---------------- 1D ----------------
ae1 = load("stage1_ae_patched_full_section224_ks_localfield_d48_localmlp.pt")
pr1 = load("stage2_prop_patched_full_section224_ks_localfield_d48_localmlp_warmstart_k12_300ep.pt")
a, p = as_dict(ae1["ae_config"]), as_dict(pr1["prop_config"])
s1, s2 = Stage1TrainingConfig(), Stage2TrainingConfig()
cols["1D KS"] = {
    "patch (grid points)": a["NX"] // a["n_sites"], "site length $h$": 100.0 / a["n_sites"],
    "channels $c$ (learned + anchor)": f"{a['local_channels']-1}+1", "AE hidden width": a["hidden"],
    "encoder reach $r_e n_e$": f"{a['site_mix_radius']}$\\times${a['n_site_mix_layers']}",
    "decoder reach $r_d n_d$": f"{a['site_mix_radius']}$\\times${a['n_site_mix_layers']}",
    "propagator $w$, $n_\\ell$, $d_m$": f"{p['attn_window']}, {p['token_n_layers']}, {p['token_d_model']}",
    "training data": "periodic, $L=100$ (whole box)",
    "trajectories (train+val) $\\times$ length": "50+10 $\\times$ 250",
    "window (sites)": "---", "halo (sites)": "---",
    "Stage A length": "200 epochs", "Stage A learning rate": f"{s1.lr:g}", "Stage A batch": f"{s1.batch_size}",
    "Stage A $w_{\\rm var}$ / $w_{\\rm ld}$": "0.02 / 0.0035$^a$",
    "log-det offset $\\epsilon$": f"{s1.logdet_eps:g}",
    "Stage A auxiliary loss": f"$w_{{\\rm pred}}$ {s1.w_pred:g}, $k=2$",
    "Stage B length": "300 epochs", "Stage B learning rate": f"{s2.lr:g}", "Stage B batch": f"{s2.batch_size}",
    "$k$ curriculum": "2$\\to$8$\\to$12",
    "input noise": f"{s2.noise_in_start:g}$\\to${s2.noise_in_end:g}",
    "$w_{\\rm rank}$": "---",
    "validation recon. MSE": f"{ae1['val_recon_final']:.1e}", "validation $k_{\\max}$ MSE": f"{pr1['best_val_kmax_mse']:.3f}",
}
record["section224"] = {"source": "checkpoints + scripts/section224_ks_localfield_d48_localmlp.sh + "
                                  "Stage{1,2}TrainingConfig defaults", "ae_config": a, "prop_config": p,
                        "stage1": {"epochs": 200, "lr": s1.lr, "batch_size": s1.batch_size, "w_var": 0.02,
                                   "w_logdet": 0.0035, "logdet_eps": s1.logdet_eps, "w_spatial": 0.01,
                                   "spatial_signed": True, "w_decorr": 0.0, "w_pred": s1.w_pred, "k_pred": 2},
                        "stage2": {"epochs": 300, "lr": s2.lr, "batch_size": s2.batch_size, "k_max": 12, "k_mid": 8,
                                   "k_mid_epochs": 175, "k_warmup_epochs": 210, "noise_in_start": s2.noise_in_start,
                                   "noise_in_end": s2.noise_in_end, "weight_decay": s2.weight_decay}}
# ---------------- 2D ----------------
def nparams(sd):
    return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))


def pcount(ae, pr):
    """Trainable parameters (autoencoder, propagator, total), read from the checkpoints. They do not depend on
    the domain: the same tensors are used on every deployment grid."""
    a, p = nparams(ae["ae_state_dict"]), nparams(pr["prop_state_dict"])
    f = lambda n: f"{n:,}".replace(",", "{,}")  # noqa: E731
    return {"parameters, autoencoder": f(a), "parameters, propagator": f(p), "parameters, total": f(a + p)}


def data_noise(tag, ae):
    """Std of the fixed Gaussian noise added to the training fields (I33). Stored in checkpoints from 2026-10-10;
    for earlier ones read from the training log (the log line is written only when noise was added)."""
    if "data_noise" in ae:
        return ae["data_noise"]
    log = ROOT / f"artifacts/logs/crop_train_{tag}.log"
    if log.exists():
        import re as _re
        m = _re.search(r"added fixed Gaussian data noise, std ([0-9.]+)", log.read_text())
        if m:
            return float(m.group(1))
    return 0.0


for name, tag in MODELS_2D:
    ae, pr = load(f"stage1_ae_kolmogorov_crop_{tag}.pt"), load(f"stage2_prop_kolmogorov_crop_{tag}.pt")
    a, ca, p, cp = as_dict(ae["ae_config"]), ae["crop_ae_config"], as_dict(pr["prop_config"]), pr["crop_prop_config"]
    sh = train_shape(tag)
    ks2d = "ks2d" in ae["train_dataset"]
    dt_snap = 0.1 if ks2d else 0.2
    px = a["NX"] // a["n_sites_x"]
    h = (264.0 / 48) if ks2d else (2 * 3.141592653589793 / 8)
    crop = f"{ca['crop_sites']}$\\times${ca['crop_sites_y']}" if ca["crop_sites_y"] else f"{ca['crop_sites']}$\\times$full $y$"
    hy = cp.get("halo_sites_y")
    halo = f"{cp['halo_sites']}" + ((" ($x$, $y$)" if hy == cp["halo_sites"] else f"/{hy}") if hy else " ($x$ only)")
    kmid_pct = round(100 * cp["k_mid_frac"])
    cols[name] = {
        "patch (grid points)": f"{px}$\\times${a['NY'] // a['n_sites_y']}", "site length $h$": round(h, 2),
        "channels $c$ (learned + anchor)": f"{a['local_channels']-1}+1", "AE hidden width": a["hidden"],
        "encoder reach $r_e n_e$": f"{a['site_mix_radius']}$\\times${a['n_site_mix_layers']}",
        "decoder reach $r_d n_d$": f"{a['dec_site_mix_radius']}$\\times${a['dec_n_site_mix_layers']}",
        "propagator $w$, $n_\\ell$, $d_m$": f"{p['attn_window']}, {p['local_mlp2d_n_layers']}, {p['local_mlp2d_token_d_model']}",
        "training data": "periodic, $264^2$" if ks2d else "periodic, $32\\pi\\times2\\pi$",
        "trajectories (train+val) $\\times$ length": f"{sh['n_train']}+{sh['n_val']} $\\times$ {round((sh['n_snapshots']-1)*dt_snap)}",
        "window (sites)": crop, "halo (sites)": halo,
        "Stage A length": f"{ca['steps']} steps", "Stage A learning rate": f"{ca['lr']:g}", "Stage A batch": f"{ca['batch']}",
        "Stage A $w_{\\rm var}$ / $w_{\\rm ld}$": f"{ca['w_var']:g} / {ca['w_logdet']:g}",
        "log-det offset $\\epsilon$": f"{ca.get('logdet_eps', 1e-3):g}",
        "Stage A auxiliary loss": "none",
        "Stage B length": f"{cp['steps']} steps", "Stage B learning rate": f"{cp['lr']:g}", "Stage B batch": f"{cp['batch']}",
        "$k$ curriculum": f"2$\\to${cp['k_mid']}$\\to${cp['k_max']}",
        "input noise": f"{cp['noise_in_start']:g}$\\to${cp['noise_in_end']:g}",
        "$w_{\\rm rank}$": f"{cp.get('w_rank', 0.0):g}" + ("$^b$" if cp.get("w_rank", 0.0) else ""),
        "validation recon. MSE": f"{ae['val_recon_interior']:.1e}", "validation $k_{\\max}$ MSE": f"{pr['best_val_kmax_mse']:.3f}",
        "training-data noise": f"{data_noise(tag, ae):g}",
        **pcount(ae, pr),
    }
    record[tag] = {"source": "checkpoints + crop_train log", "ae_config": a, "crop_ae_config": ca,
                   "prop_config_relevant": {k: p[k] for k in ("backbone", "attn_window", "local_mlp2d_n_layers",
                                                              "local_mlp2d_token_d_model", "token_mlp_ratio", "zero_init",
                                                              "mode", "d_latent")},
                   "crop_prop_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cp.items()},
                   "train_dataset": ae["train_dataset"], "train_shape": sh,
                   "k_mid_reached_at_pct_of_steps": kmid_pct}

cols["1D KS"].update(pcount(ae1, pr1))
for c in cols.values():
    c.setdefault("training-data noise", "0")


# Printed row labels (2026-10-11, user: capitalize every label, consistently across rows and columns).
LABELS = {
    "patch (grid points)": "Patch (grid points)", "site length $h$": "Site length $h$",
    "channels $c$ (learned + anchor)": "Channels $c$ (learned + anchor)", "AE hidden width": "Autoencoder hidden width",
    "encoder reach $r_e n_e$": "Encoder reach $r_e n_e$", "decoder reach $r_d n_d$": "Decoder reach $r_d n_d$",
    "propagator $w$, $n_\\ell$, $d_m$": "Propagator $w$, $n_\\ell$, $d_m$", "training data": "Training domain",
    "trajectories (train+val) $\\times$ length": "Trajectories (train + val) $\\times$ length",
    "window (sites)": "Window (sites)", "halo (sites)": "Halo (sites)",
    "log-det offset $\\epsilon$": "Stage A log-det offset $\\epsilon$",
    "$k$ curriculum": "Stage B rollout length $k$", "input noise": "Stage B input noise $\\sigma$",
    "$w_{\\rm rank}$": "Stage B rank penalty $w_{\\rm rank}$",
    "validation recon. MSE": "Validation reconstruction MSE", "validation $k_{\\max}$ MSE": "Validation $k_{\\max}$-step MSE",
    "training-data noise": "Training-data noise", "parameters, autoencoder": "Parameters, autoencoder",
    "parameters, propagator": "Parameters, propagator", "parameters, total": "Parameters, total",
}


def cap(v):
    v = str(v)
    return v[0].upper() + v[1:] if v[:1].islower() else v


def render(names, rows, labels=None):
    lines = ["% Generated by scripts/paper_figures/tabA_architecture.py -- do not edit by hand.",
             "Setting & " + " & ".join(labels or names) + " \\\\", "\\hline"]
    for r in rows:
        lines.append(LABELS.get(r, cap(r)) + " & " + " & ".join(cap(cols[n][r]) for n in names) + " \\\\")
    return lines


names = list(cols)
rows = list(cols[names[1]])
lines = render(names, rows)
write_table("tabA_architecture", "\n".join(lines) + "\n", main_alpha01_model=MAIN_A01)
# Paper I (docs/split_paper_tasks.md S3.10): the dynamics models only, without the data-assimilation
# model of Paper II; the training-noise row is then 0 everywhere and is dropped.
write_table("tabA_architecture_dynamics",
            "\n".join(render([n for n in names if "Part" not in n],
                             [r for r in rows if r != "training-data noise"])) + "\n",
            main_alpha01_model=MAIN_A01)
# Paper II (S4.16; R8 2026-10-12 adds the Kolmogorov model): the three models its filters use, the 1D model of
# Paper I (Sec. 9), the 12-channel noisy-trained 2D KS model (every 2D experiment) and the 12-channel
# noisy-trained Kolmogorov model (alpha = 0.05, Sec. 10).
write_table("tabA_architecture_da",
            "\n".join(render(["1D KS", "2D KS, Part~II", "Kolm., Part~II"], rows,
                              labels=["1D KS", "2D KS, 12 ch.", "Kolmogorov, 12 ch."])) + "\n")
(EXP / "model_configs.json").write_text(json.dumps(record, indent=2, default=str))
print("\n".join(lines))
