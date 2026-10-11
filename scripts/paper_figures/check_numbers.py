"""Task G2 (docs/full_paper_tasks.md): number-provenance check for the paper text.

Every decimal number in the LaTeX sources (sections, appendices, abstract; comments stripped) is
classified:
  TABLE     printed verbatim in a generated table body (docs/paper1/tables/*.tex), which its
            generator script reads from the result files;
  SOURCE    equal, at the printed precision, to a value in the result files the paper uses
            (raw, or x100 as a percentage, or 100*(v-1) as a relative change), with the number
            of distinct files it matches -- many matches means the match is weak evidence;
  UNMATCHED found in neither: a derived quantity (ratio, difference, average) or a stale or
            mis-copied number. These are the ones to check by hand.
Integers are not checked (sizes, counts and settings are too ambiguous to match). Sources: the
inputs recorded in docs/paper1/figures/*.meta.json, every JSON under artifacts/experiments/
{crop,truth_ks2d,spectra,da2d,e1_model,forecast_variance,a10_selection,rollout_stats}, the 1D DA
JSONs (artifacts/da_pff_ltransfer_d*.json, artifacts/e3_sec_vs_gc_ltransfer_d3_*.json), the
Appendix D CSV and a few summary JSONs; numeric lists longer than 20 (time series, spectra) are
skipped. Writes docs/<paper>/number_check.md.

`--paper` picks the paper (default paper1, the reference draft; paperI and paperII are the split
papers of docs/split_paper_tasks.md). The strict Part II class (SUSPECT, below) applies to the
Part II files of paper1 and to every file of paperII.
Usage: python scripts/paper_figures/check_numbers.py [--paper paper1|paperI|paperII]
"""
import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--paper", default="paper1", choices=("paper1", "paperI", "paperII"))
PAPER = ap.parse_args().paper
P1 = ROOT / "docs" / PAPER  # the paper being checked (name kept from when there was one paper)
EXP = ROOT / "artifacts/experiments"
NUM = re.compile(r"(?<![\w.\\])(\d+\.\d+)(?![\w.])")

# Unmatched numbers checked by hand (2026-10-08) and why they are right; a number listed here is
# reported as EXPLAINED instead of UNMATCHED. Keep the reasons specific. Keys are per paper because
# file names differ; carry an entry over to paperI/paperII when the number moves there.
EXPLAINED_BY_PAPER = {"paper1": {
    ("sections/04_systems.tex", "0.048"): "literature value (Cvitanovic et al. 2010 leading exponent at L=22)",
    ("sections/05_results_1d.tex", "0.2125"): "axis range of Fig. 3(b), quoted in the caption",
    ("sections/05_results_1d.tex", "0.2325"): "axis range of Fig. 3(b), quoted in the caption",
    ("sections/07_data_assimilation.tex", "4.55"): "D9 steady-state skill at 88^2 (4.549), da2d_D9_L88_local_N64_noise0.04_s0.json",
    ("sections/07_data_assimilation.tex", "5.96"): "ratio of filter-only RSS 220^2/88^2 (3.35/0.5625), cost_summary.json",
}, "paperI": {
    ("sections/04_systems.tex", "0.048"): "literature value (Cvitanovic et al. 2010 leading exponent at L=22)",
    ("sections/05_results_1d.tex", "0.2125"): "axis range of Fig. 3(b), quoted in the caption",
    ("sections/05_results_1d.tex", "0.2325"): "axis range of Fig. 3(b), quoted in the caption",
}, "paperII": {}}
EXPLAINED = EXPLAINED_BY_PAPER[PAPER]

# ---------- source values ----------
files = set()
for m in glob.glob(str(P1 / "figures/*.meta.json")):
    for f in json.loads(Path(m).read_text()).get("inputs", []):
        p = Path(f) if Path(f).is_absolute() else ROOT / f
        if p.suffix in (".json", ".csv") and p.exists():
            files.add(p)
for sub in ("crop", "truth_ks2d", "spectra", "da2d", "e1_model", "forecast_variance", "a10_selection", "rollout_stats"):
    files.update(Path(f) for f in glob.glob(str(EXP / sub / "*.json")))
files.update(Path(f) for f in glob.glob(str(ROOT / "artifacts/da_pff_ltransfer_d*.json")))
files.update(Path(f) for f in glob.glob(str(ROOT / "artifacts/e3_sec_vs_gc_ltransfer_d3_*.json")))
files.update(Path(f) for f in glob.glob(str(EXP / "*.json")))
files.add(EXP / "appendix_D_evaluations.csv")

index = defaultdict(set)  # printed string -> set of files

# K11 (2026-10-11): strict Part II check. Numbers in the Part II text may only come from the Part II
# results (the 12-channel model p7c12*, its filters, the physics comparator, the 1D DA files) or the
# Part II tables; a match only in Part I or superseded-filter files (da2d_*) is reported as SUSPECT.
P2_TEX = {"sections/07_data_assimilation.tex", "appendix/E_da_details.tex", "appendix/F_reproducibility.tex"}
P2_TABLES = ("tab07_p2_", "tab_da_cost", "tabE_p2_", "tab07_1d_da")


def strict(texfile: str) -> bool:
    return PAPER == "paperII" or (PAPER == "paper1" and texfile in P2_TEX)


def p2_table(name: str) -> bool:
    return PAPER == "paperII" or (PAPER == "paper1" and name.startswith(P2_TABLES))
P2_SRC = re.compile(r"(da2d/(h9_|i11_|j16_|j17|j18|jacobian_timing_F4_p7c12|cost_summary|seams|gaussianity|i24|i30|probe|leaky|hybrid))"
                    r"|p7c12|da_pff_ltransfer|e3_sec_vs_gc|truth_ks2d")


def add_value(v, src):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
        return
    for x in (v, 100 * v, 100 * (v - 1), abs(100 * (v - 1))):
        for p in range(0, 5):
            index[f"{abs(x):.{p}f}"].add(src)


def walk(o, src):
    if isinstance(o, dict):
        for v in o.values():
            walk(v, src)
    elif isinstance(o, list):
        if len(o) <= 20:
            for v in o:
                walk(v, src)
    else:
        add_value(o, src)


for f in sorted(files):
    src = str(f.relative_to(ROOT))
    try:
        if f.suffix == ".json":
            walk(json.loads(f.read_text()), src)
        elif f.suffix == ".csv":
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    for v in row.values():
                        try:
                            add_value(float(v), src)
                        except (TypeError, ValueError):
                            pass
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue

table_tokens, p2_table_tokens = set(), set()
for t in glob.glob(str(P1 / "tables/*.tex")):
    toks = NUM.findall(Path(t).read_text())
    table_tokens.update(toks)
    if p2_table(Path(t).name):
        p2_table_tokens.update(toks)

# ---------- text numbers ----------
tex_files = sorted(glob.glob(str(P1 / "sections/*.tex")) + glob.glob(str(P1 / "appendix/*.tex")) + [str(P1 / "main.tex")])
rows, counts = [], defaultdict(int)
for tf in tex_files:
    for ln, line in enumerate(Path(tf).read_text().splitlines(), 1):
        code = re.sub(r"(?<!\\)%.*", "", line)
        code = re.sub(r"\\(ref|label|cite[tp]?|includegraphics|tabinput|input)\{[^}]*\}", "", code)
        code = re.sub(r"\[[\d.]+(in|pt|cm|em)\]|[\d.]+\\(linewidth|textwidth|columnwidth)", "", code)
        for tok in NUM.findall(code):
            if tok in table_tokens:
                status, n = "TABLE", 0
            elif tok in index:
                status, n = "SOURCE", len(index[tok])
            elif (str(Path(tf).relative_to(P1)), tok) in EXPLAINED:
                status, n = "EXPLAINED", 0
            else:
                status, n = "UNMATCHED", 0
            if strict(str(Path(tf).relative_to(P1))) and status in ("TABLE", "SOURCE") and \
                    tok not in p2_table_tokens and not any(P2_SRC.search(f) for f in index.get(tok, ())):
                status, n = "SUSPECT", len(index.get(tok, ()))
            counts[status] += 1
            rows.append((Path(tf).relative_to(P1), ln, tok, status, n,
                         sorted(index.get(tok, []))[:2] if status == "SOURCE" else []))

out = [f"# Number-provenance check (task G2): {PAPER}", "",
       f"Generated by `scripts/paper_figures/check_numbers.py --paper {PAPER}`; regenerate after every text or "
       "result change.", "",
       f"Decimals checked: {sum(counts.values())} — TABLE {counts['TABLE']}, SOURCE {counts['SOURCE']}, "
       f"UNMATCHED {counts['UNMATCHED']}, SUSPECT {counts['SUSPECT']}, EXPLAINED {counts['EXPLAINED']}. SOURCE matches with many files (n ≥ 10) are weak evidence.", "",
       "Caveat: a decimal such as 2.3 matches something somewhere by chance, so TABLE/SOURCE show that "
       "a number is consistent with the results, not that it came from the intended source. The check "
       "is designed to catch numbers that match nothing — the usual signature of a stale or mis-copied "
       "value.", "",
       "## Unmatched (check by hand)", "", "| file | line | number |", "|---|---|---|"]
out += [f"| {f} | {ln} | {tok} |" for f, ln, tok, s, n, _ in rows if s == "UNMATCHED"]
out += ["", "## Suspect (Part II text; matches only Part I or superseded-filter results)", "",
        "| file | line | number | files | e.g. |", "|---|---|---|---|---|"]
out += [f"| {f} | {ln} | {tok} | {n} | {', '.join(sorted(index.get(tok, []))[:2])} |"
        for f, ln, tok, s, n, _ in rows if s == "SUSPECT"]
out += ["", "## Explained (checked by hand)", "", "| file | line | number | why |", "|---|---|---|---|"]
out += [f"| {f} | {ln} | {tok} | {EXPLAINED[(str(f), tok)]} |" for f, ln, tok, s, n, _ in rows if s == "EXPLAINED"]
out += ["", "## Weak matches (SOURCE, n ≥ 10 files)", "", "| file | line | number | files |", "|---|---|---|---|"]
out += [f"| {f} | {ln} | {tok} | {n} |" for f, ln, tok, s, n, _ in rows if s == "SOURCE" and n >= 10]
out += ["", "## Matched (SOURCE, n < 10 files)", "", "| file | line | number | files | e.g. |", "|---|---|---|---|---|"]
out += [f"| {f} | {ln} | {tok} | {n} | {', '.join(ex)} |" for f, ln, tok, s, n, ex in rows if s == "SOURCE" and n < 10]
(P1 / "number_check.md").write_text("\n".join(out) + "\n")
print(PAPER, dict(counts), "sources:", len(files), "files")
