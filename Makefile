.PHONY: env-check test test-fast smoke replicate bench

ENV := da_env
RUN := mamba run -n $(ENV)

env-check:
	$(RUN) python scripts/check_env.py

test-fast:
	$(RUN) python -m pytest tests/unit tests/integration -m "not slow" -v

test:
	$(RUN) python -m pytest tests -v

smoke:
	$(RUN) python -m pytest tests/integration -v

replicate:
	$(RUN) python -m pytest tests/replication -v -m slow
	$(RUN) python scripts/generate_replication_log.py

bench:
	$(RUN) python scripts/bench_device.py

# ---------------------------------------------------------------- papers
# paper1 is the frozen reference draft; paperI / paperII are the split papers
# (docs/split_paper_tasks.md). `make <paper>` builds the PDF and runs the checks;
# `make <paper>-assets` regenerates that paper's tables and figures from the result files
# (tables first: fig14 and the cost text read summaries the table scripts write).
.PHONY: paper1 paperI paperII paperI-assets paperII-assets
PY := $(RUN) python
PF := scripts/paper_figures

PAPERI_TABLES := tab02_tab03 tab04_transfer2d tab08_alpha01_ablation tabA_architecture \
	tabC_truth tabD_evaluations tabD_spectra
PAPERI_FIGS := fig01_concept fig02_systems fig03_1d_transfer fig04_1d_spectra fig05_routes \
	fig06_scaling fig07_rollouts_spectra fig10_failure figC1_truth_convergence figC2_appendix_panels
PAPERII_TABLES := tab07_1d_da tab07_p2_model_error tab07_p2_tables tabE_p2 tabE_p2_robustness tab_da_cost \
	tabA_architecture tabC_truth tabV_model_validation
PAPERII_FIGS := fig_concept_da fig02_systems fig08_1d_da fig09_p2_da fig11_da_snapshot fig12_jacobian_timing \
	fig13_tilings fig14_model_error fig16_forecasts

define build_paper
	cd docs/$(1) && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /dev/null \
		|| { grep -E -A3 "^!" main.log; exit 1; }
	$(PY) $(PF)/check_numbers.py --paper $(1)
	@! grep -E "Reference .* undefined|Citation .* undefined|multiply defined" docs/$(1)/main.log \
		|| { echo "$(1): undefined or duplicate references (above)"; exit 1; }
	@grep -c "placeholder" docs/$(1)/sections/*.tex docs/$(1)/appendix/*.tex docs/$(1)/main.tex \
		| awk -F: '{n += $$2} END {print "$(1): " n " lines with \\placeholder left"}'
endef

define assets
	@for s in $(2); do echo "== $$s"; PAPER_OUT=$(CURDIR)/docs/$(1)/figures PAPER_TABLES=$(CURDIR)/docs/$(1)/tables \
		$(PY) $(PF)/$$s.py > /dev/null || exit 1; done
endef

paper1:
	$(call build_paper,paper1)
paperI:
	$(call build_paper,paperI)
paperII:
	$(call build_paper,paperII)
paperI-assets:
	$(call assets,paperI,$(PAPERI_TABLES) $(PAPERI_FIGS))
paperII-assets:
	$(call assets,paperII,$(PAPERII_TABLES) $(PAPERII_FIGS))
