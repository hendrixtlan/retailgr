SHELL := /bin/bash
PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
export PYTHONPATH := src

# Iceberg with no JVM jars and no services: a SQL catalog beside the warehouse
# root and the pyiceberg engine. This is the configuration that made the
# lakehouse backend executable at all on a machine that cannot reach Maven.
ICEBERG_LOCAL := --set warehouse.backend=iceberg \
	--set warehouse.iceberg.engine=pyiceberg \
	--set warehouse.iceberg.uri= --set warehouse.iceberg.warehouse=

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel

.PHONY: install
install: $(BIN)/python ## Create the virtualenv and install the project
	$(BIN)/pip install -e ".[dev,serving]"

.PHONY: install-all
install-all: $(BIN)/python ## Install with Kafka and Redis clients too
	$(BIN)/pip install -e ".[dev,all]"

.PHONY: test
test: ## Run the unit tests (fast)
	$(BIN)/python -m pytest -q -m "not slow"

.PHONY: test-all
test-all: ## Run every test, including the end-to-end Spark pipeline
	$(BIN)/python -m pytest -q

.PHONY: lint
lint: ## Check formatting and imports
	$(BIN)/ruff check src tests

.PHONY: mutants
mutants: ## Break each load-bearing claim and check a test notices
	$(BIN)/python scripts/mutation_check.py

.PHONY: data
data: ## Generate the synthetic dataset
	$(BIN)/python -m retailgr.cli generate-data

.PHONY: pipeline
pipeline: ## bronze -> silver -> gold sequences (config granularity)
	$(BIN)/python -m retailgr.cli ingest
	$(BIN)/python -m retailgr.cli silver
	$(BIN)/python -m retailgr.cli sequences --granularity config --variant config

.PHONY: stage1
stage1: ## Data, pipeline, every granularity variant, popularity + SASRec + HSTU
	$(BIN)/python -m retailgr.cli experiment

.PHONY: ablate
ablate: ## Switch off HSTU components one at a time and compare
	$(BIN)/python -m retailgr.cli ablate --variant config

.PHONY: convergence
convergence: ## Fixed epochs against early stopping, both scored on test
	$(BIN)/python -m retailgr.cli convergence --seeds 13 17 23

.PHONY: retention
retention: ## Prune each layer past its retention window (dry run first)
	$(BIN)/python -m retailgr.cli retention --dry-run

.PHONY: consent-cost
consent-cost: ## Measure what honouring consent costs the model, at several opt-in rates
	$(BIN)/python -m retailgr.cli consent-cost --seeds 0 1 2
	$(BIN)/python -m retailgr.cli consent-cost --seeds 0 1 2 --correlate

.PHONY: movielens
movielens: ## Fetch MovieLens and run the model experiment on real timestamps
	bash scripts/fetch_movielens.sh data/raw
	$(BIN)/python -m retailgr.cli experiment --dataset ml1m --variants config \
		--set privacy.consent.enforce=false \
		--model-configs sasrec_ml1m.yaml hstu_ml1m.yaml \
		--set clean.bot_events_per_day=100000 --set clean.max_events_per_user=100000 \
		--set sequences.max_len=200 --output artifacts/ml1m

# -- serving ------------------------------------------------------------------

.PHONY: bundle
bundle: ## Train a model and export a serving bundle
	$(BIN)/python -m retailgr.cli export-model --variant config \
		--model-config hstu_small.yaml --output artifacts/bundle

.PHONY: bootstrap
bootstrap: ## Replay silver through the broker into the online store
	$(BIN)/python -m retailgr.cli bootstrap --variant config

.PHONY: serve
serve: ## Run the recommendations API (bootstraps the in-memory store first)
	$(BIN)/python -m retailgr.cli serve --bootstrap

.PHONY: bench
bench: ## Measure the serving latency budget, stage by stage
	$(BIN)/python -m retailgr.cli bench --requests 500

.PHONY: audit
audit: ## Check the portability claim; exits non-zero if it fails
	$(BIN)/python -m retailgr.cli audit

.PHONY: stack-iceberg-local
stack-iceberg-local: ## Run the pipeline on Iceberg with no JVM jars and no services
	$(BIN)/python -m retailgr.cli ingest    $(ICEBERG_LOCAL)
	$(BIN)/python -m retailgr.cli silver    $(ICEBERG_LOCAL)
	$(BIN)/python -m retailgr.cli sequences $(ICEBERG_LOCAL)

.PHONY: sizing
sizing: ## Measure what a serving pod needs: memory, cold start, throughput
	$(BIN)/python -m retailgr.cli sizing

.PHONY: stage2
stage2: bundle bench ## Bundle, then measure the serving path end to end

# -- infrastructure -----------------------------------------------------------

.PHONY: up
up: ## Start MinIO, Iceberg catalog, Kafka, Schema Registry and Redis
	docker compose -f docker/docker-compose.yml up -d

.PHONY: down
down: ## Stop the local stack
	docker compose -f docker/docker-compose.yml down

.PHONY: stack-iceberg
stack-iceberg: ## Run the experiment against the Iceberg lakehouse
	$(BIN)/python -m retailgr.cli experiment --backend iceberg

.PHONY: stack-kafka
stack-kafka: ## Run the streaming path against real Kafka and Redis
	$(BIN)/python -m retailgr.cli bootstrap \
		--set streaming.broker=kafka --set online_store.backend=redis

.PHONY: clean
clean: ## Remove generated data, warehouse tables and reports
	rm -rf data/warehouse data/warehouse_demo data/tmp artifacts
