PYTHON ?= python3
# Lets every target work in a fresh clone, before `make setup` has been run.
export PYTHONPATH := $(CURDIR)/src

.PHONY: all setup test lint data exp1 exp2 exp3 exp4 exp5 figures report clean fast

all: data exp1 exp2 exp3 exp4 exp5 figures report

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src scripts tests

# Experiment 0: download the pinned data and record its provenance.
data:
	$(PYTHON) scripts/00_fetch_data.py

# Experiment 1: does model flexibility buy out-of-sample accuracy?
exp1:
	$(PYTHON) scripts/01_model_comparison.py

# Experiment 2: is the accuracy worth anything after trading costs?
exp2:
	$(PYTHON) scripts/02_portfolios.py

# Experiment 3: factor alphas, block bootstrap, and multiple testing.
exp3:
	$(PYTHON) scripts/03_inference.py

# Experiment 4: do asset-pricing tests behave with many assets?
exp4:
	$(PYTHON) scripts/04_high_dimensional.py

# Experiment 5: ablations, regimes, and alternative design choices.
exp5:
	$(PYTHON) scripts/05_robustness.py

figures:
	$(PYTHON) scripts/06_figures.py

# Rebuild the research report from whatever is currently in results/.
report:
	$(PYTHON) scripts/07_report.py

# End-to-end smoke run: minutes rather than an hour, same code path.
fast:
	$(PYTHON) scripts/00_fetch_data.py
	$(PYTHON) scripts/01_model_comparison.py --fast
	$(PYTHON) scripts/02_portfolios.py
	$(PYTHON) scripts/03_inference.py
	$(PYTHON) scripts/04_high_dimensional.py --quick
	$(PYTHON) scripts/05_robustness.py --skip-ablation
	$(PYTHON) scripts/06_figures.py
	$(PYTHON) scripts/07_report.py

clean:
	rm -rf results reports/figures reports/tables data/processed
