"""10b: summary figure of the scale blend against the average, probability matching and one member (from postproc_test.json).

  python scripts/cm1/plot_blend_summary.py --data docs/fpe_report/data/postproc_test.json --out docs/fpe_report/figs_pool/fig_blend_summary.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHOW = [("mean", "average of 32 members", "#2A78C2", "-", "o"), ("blend10", "10 km scale blend", "#1F8A4B", "-", "s"),
        ("blend20", "20 km scale blend", "#7BC08E", "--", "s"), ("pm", "probability-matched average", "#B5446E", "-.", "^"),
        ("member", "one member", "#9AA0A6", ":", "D")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.data)); S = d["summary"]; lags = d["lags"]; bands = d["bands_km"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(wspace=0.28, left=0.05, right=0.99, top=0.84, bottom=0.25))
    for k, lab, c, ls, mk in SHOW:
        ax[0].plot(lags, [S[f"{k}|{L}"]["rmse"] for L in lags], ls, color=c, marker=mk, ms=6, lw=2, label=lab)
        ax[2].plot(lags, [S[f"{k}|{L}"]["fss"]["w3@10"] for L in lags], ls, color=c, marker=mk, ms=6, lw=2)
        p = S[f"{k}|60"]["power"]
        ax[1].plot(range(len(bands)), p, ls, color=c, marker=mk, ms=6, lw=2)
    ax[0].set_title("Error (lower is better)", fontsize=10); ax[0].set_ylabel("RMSE, normalized units"); ax[0].set_xlabel("lead (min)")
    ax[1].axhline(1, color="k", lw=0.8, ls=":")
    ax[1].set_xticks(range(len(bands))); ax[1].set_xticklabels(["0.5-2", "2-5", "5-10", "10-20", "20-50", ">50"])
    ax[1].set_yscale("log"); ax[1].set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2]); ax[1].set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax[1].minorticks_off()
    ax[1].set_title("Realism at +60 min: power / truth's power (1 = realistic)", fontsize=10); ax[1].set_xlabel("wavelength band (km)")
    ax[2].set_title("Updraft placement: FSS of |w| > 3 m/s, 10 km window", fontsize=10); ax[2].set_xlabel("lead (min)"); ax[2].set_ylabel("FSS")
    for x in ax:
        x.grid(alpha=0.25)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Single-field products from the pool of 4 models x 8 members, 22 test storms x 3 starts (66 cases)", fontsize=11)
    fig.savefig(a.out, dpi=150); plt.close(fig)
    print(a.out)


if __name__ == "__main__":
    main()
