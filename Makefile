.PHONY: setup setup-full validate test compile smoke readiness prepare-b2 prepare-b3 ablation robustness contrasts

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-ci.txt

setup-full: setup
	$(PYTHON) -m pip install -r requirements-ml.txt
	$(PYTHON) -m pip install -r gabriel-scrapper/requirements.txt
	$(PYTHON) -m pip install -r gabriel-curadoria/requirements.txt
	$(PYTHON) -m pip install -r gabriel-telegram/requirements.txt

validate:
	$(PYTHON) -m cinebot_ml.readiness ci

test:
	$(PYTHON) -m unittest discover -s tests

compile:
	$(PYTHON) -m compileall -q cinebot_ml tests

smoke:
	$(PYTHON) -m unittest tests.test_experiment_readiness.ExperimentReadinessTests.test_smoke_b0_a_b5_em_dados_minimos -v

readiness:
	$(PYTHON) -m cinebot_ml.readiness full

prepare-b2:
	@test -n "$(SPLIT_MANIFEST)" || (echo "Informe SPLIT_MANIFEST=<arquivo>" && exit 2)
	$(PYTHON) -m cinebot_ml.artifacts build-b2 --split-manifest "$(SPLIT_MANIFEST)"

prepare-b3:
	$(PYTHON) train_ml.py

ablation:
	$(PYTHON) -m cinebot_ml.analysis.ablation_execution

robustness:
	$(PYTHON) -m cinebot_ml.analysis.robustness_execution

contrasts:
	$(PYTHON) -m cinebot_ml.analysis.confirmatory_contrasts
