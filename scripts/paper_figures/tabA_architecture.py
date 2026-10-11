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
             ("Kolm., $\\alpha=0.05$", "c19_e0d0_c6"), ("Kolm., $\\alpha=0.025$", "k_xs0025")]


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
    "training data": "whole periodic box, $L=100$",
    "trajectories (train+val) $\\times$ length": "50+10 $\\times$ 250",
    "crop / halo (sites)": "--- / ---",
    "Stage A": f"200 ep., lr {s1.lr:g}, batch {s1.batch_size}",
    "Stage A regularizers": "$w_{\\rm var}$ 0.02, $w_{\\rm ld}$ 0.0035, $w_{\\rm sp}$ 0.01$^a$",
    "log-det offset $\\epsilon$": f"{s1.logdet_eps:g}",
    "Stage A aux. loss": f"$w_{{\\rm pred}}$ {s1.w_pred:g}, $k=2$",
    "Stage B": f"300 ep., lr {s2.lr:g}, batch {s2.batch_size}",
    "$k$ curriculum": "2$\\to$8$\\to$12",
    "input noise": f"{s2.noise_in_start:g}$\\to${s2.noise_in_end:g}",
    "$w_{\\rm rank}$": "---",
    "val. recon / val. $k_{\\max}$ MSE": f"{ae1['val_recon_final']:.1e} / {pr1['best_val_kmax_mse']:.3f}",
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
    """Trainable parameters (autoencoder + propagator), read from the checkpoints. They do not depend on the
    domain: the same tensors are used on every deployment grid."""
    a, p = nparams(ae["ae_state_dict"]), nparams(pr["prop_state_dict"])
    return f"{a:,} + {p:,} = {a + p:,}".replace(",", "{,}")


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
    halo = f"{cp['halo_sites']}" + (f"/{cp['halo_sites_y']}" if cp.get("halo_sites_y") else " ($x$ only)")
    kmid_pct = round(100 * cp["k_mid_frac"])
    cols[name] = {
        "patch (grid points)": f"{px}$\\times${a['NY'] // a['n_sites_y']}", "site length $h$": round(h, 2),
        "channels $c$ (learned + anchor)": f"{a['local_channels']-1}+1", "AE hidden width": a["hidden"],
        "encoder reach $r_e n_e$": f"{a['site_mix_radius']}$\\times${a['n_site_mix_layers']}",
        "decoder reach $r_d n_d$": f"{a['dec_site_mix_radius']}$\\times${a['dec_n_site_mix_layers']}",
        "propagator $w$, $n_\\ell$, $d_m$": f"{p['attn_window']}, {p['local_mlp2d_n_layers']}, {p['local_mlp2d_token_d_model']}",
        "training data": "$264^2$ periodic" if ks2d else "$32\\pi\\times2\\pi$ periodic",
        "trajectories (train+val) $\\times$ length": f"{sh['n_train']}+{sh['n_val']} $\\times$ {round((sh['n_snapshots']-1)*dt_snap)}",
        "crop / halo (sites)": f"{crop} / {halo}",
        "Stage A": f"{ca['steps']} steps, lr {ca['lr']:g}, batch {ca['batch']}",
        "Stage A regularizers": f"$w_{{\\rm var}}$ {ca['w_var']:g}, $w_{{\\rm ld}}$ {ca['w_logdet']:g}",
        "log-det offset $\\epsilon$": f"{ca.get('logdet_eps', 1e-3):g}",
        "Stage A aux. loss": "none",
        "Stage B": f"{cp['steps']} steps, lr {cp['lr']:g}, batch {cp['batch']}",
        "$k$ curriculum": f"2$\\to${cp['k_mid']}$\\to${cp['k_max']}",
        "input noise": f"{cp['noise_in_start']:g}$\\to${cp['noise_in_end']:g}",
        "$w_{\\rm rank}$": f"{cp.get('w_rank', 0.0):g}" + ("$^b$" if cp.get("w_rank", 0.0) else ""),
        "val. recon / val. $k_{\\max}$ MSE": f"{ae['val_recon_interior']:.1e} / {pr['best_val_kmax_mse']:.3f}",
        "training-data noise": f"{data_noise(tag, ae):g}",
        "parameters (AE + propagator)": pcount(ae, pr),
    }
    record[tag] = {"source": "checkpoints + crop_train log", "ae_config": a, "crop_ae_config": ca,
                   "prop_config_relevant": {k: p[k] for k in ("backbone", "attn_window", "local_mlp2d_n_layers",
                                                              "local_mlp2d_token_d_model", "token_mlp_ratio", "zero_init",
                                                              "mode", "d_latent")},
                   "crop_prop_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cp.items()},
                   "train_dataset": ae["train_dataset"], "train_shape": sh,
                   "k_mid_reached_at_pct_of_steps": kmid_pct}

cols["1D KS"]["parameters (AE + propagator)"] = pcount(ae1, pr1)
for c in cols.values():
    c.setdefault("training-data noise", "0")


def render(names, rows):
    lines = ["% Generated by scripts/paper_figures/tabA_architecture.py -- do not edit by hand.",
             " & " + " & ".join(names) + " \\\\", "\\hline"]
    for r in rows:
        lines.append(r + " & " + " & ".join(str(cols[n][r]) for n in names) + " \\\\")
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
(EXP / "model_configs.json").write_text(json.dumps(record, indent=2, default=str))
print("\n".join(lines))
