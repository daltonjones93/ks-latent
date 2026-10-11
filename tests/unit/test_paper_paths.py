"""Output paths of the paper figure/table scripts (docs/split_paper_tasks.md S1.1-S1.3).

paper1 keeps its numbered names; the split papers get content names, which must be unique so
that no two scripts write the same file of a paper."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PF = ROOT / "scripts/paper_figures"


@pytest.fixture()
def style(monkeypatch):
    monkeypatch.syspath_prepend(str(PF))
    sys.modules.pop("_style", None)
    return importlib.import_module("_style")


def test_paper1_names_unchanged(style):
    assert style.content_name("fig09_p2_da", style._P1_FIGS) == "fig09_p2_da"
    assert style.content_name("tab07_p2_sizes", style._P1_TABLES) == "tab07_p2_sizes"


@pytest.mark.parametrize("old,new", [
    ("fig01_concept", "concept"), ("figC1_truth_convergence", "truth_convergence"),
    ("fig09_p2_da", "da_2d"), ("tab07_p2_sizes", "sizes"), ("tabA_architecture", "architecture"),
    ("tab_da_cost", "da_cost"), ("tabE_p2_robustness", "robustness"), ("tabD_ks2d", "eval_ks2d"),
])
def test_content_names(style, tmp_path, old, new):
    assert style.content_name(old, tmp_path) == new


def test_content_names_unique(style, tmp_path):
    for kind in ("figures", "tables"):
        old = sorted({p.name.split(".")[0] for p in (ROOT / "docs/paper1" / kind).iterdir()
                      if p.suffix in (".pdf", ".tex")})
        new = [style.content_name(n, tmp_path) for n in old]
        dup = {n for n in new if new.count(n) > 1}
        assert not dup, f"{kind}: content-name collision {dup}"


def test_env_redirects_output(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_OUT", str(tmp_path / "f"))
    monkeypatch.setenv("PAPER_TABLES", str(tmp_path / "t"))
    monkeypatch.syspath_prepend(str(PF))
    sys.modules.pop("_style", None)
    s = importlib.import_module("_style")
    out = s.write_table("tab07_p2_sizes", "x \\\\\n", ["a.json", "a.json"], summary={"k": 1})
    assert out == (tmp_path / "t" / "sizes.tex").resolve() and out.read_text() == "x \\\\\n"
    import json
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["inputs"] == ["a.json"] and meta["summary"] == {"k": 1} and meta["table"] == "sizes"
    sys.modules.pop("_style", None)
