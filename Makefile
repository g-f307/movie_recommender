.PHONY: setup setup-full validate test compile smoke readiness prepare-b2 prepare-b3 ablation robustness contrasts discovery-cost external-validation final-evidence

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

discovery-cost:
	$(PYTHON) -m cinebot_ml.analysis.discovery_cost

external-validation:
	$(PYTHON) -m cinebot_ml.analysis.external_validation
final-evidence:
	$(PYTHON) -m cinebot_ml.analysis.external_validation --output-dir "$$(mktemp -d /tmp/movie-recommender-final-evidence-XXXXXX)/evidence"
	cd reports/ACM_SAC_2027_Article_Template && latexmk -pdf -interaction=nonstopmode -halt-on-error movie_recommender_draft.tex
