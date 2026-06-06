.DEFAULT_GOAL := help

# ── Variables ──────────────────────────────────────────────────────────────────
VENV          := .venv
SYSTEM_PYTHON := $(shell command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null)
PYTHON        := $(VENV)/bin/python3
PIP           := $(PYTHON) -m pip
BLACK         := $(PYTHON) -m black
FLAKE8        := $(PYTHON) -m flake8
SRC           := training monitoring api dashboard

.PHONY: help install hooks \
        up down build logs \
        train drift \
        format lint check test \
        version clean

# ── Help ───────────────────────────────────────────────────────────────────────
help: ## Show available targets
	@grep -E '^[a-zA-Z_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────
install: ## Create .venv (if needed) and install all dependencies
	@test -f $(PYTHON) || $(SYSTEM_PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r training/requirements.txt -r requirements-dev.txt
	$(PIP) install pre-commit black==24.4.2 flake8==7.1.0 flake8-pyproject

hooks: ## Install pre-commit hooks (run once after clone)
	pre-commit install --hook-type pre-commit --hook-type pre-push

# ── Docker ─────────────────────────────────────────────────────────────────────
up: ## Start all services detached
	docker compose up -d

down: ## Stop and remove all containers
	docker compose down

build: ## Build api and dashboard Docker images
	docker compose build

logs: ## Tail logs from all services
	docker compose logs -f

# ── ML ─────────────────────────────────────────────────────────────────────────
train: ## Train model and register to MLflow (requires: make up first)
	@echo "Waiting for MLflow..."; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		$(PYTHON) -c "import urllib.request; urllib.request.urlopen('http://localhost:5000', timeout=2)" 2>/dev/null && break; \
		echo "  not ready yet ($$i/10), retrying in 3s..."; \
		sleep 3; \
		if [ "$$i" = "10" ]; then echo "Error: MLflow not reachable after 30s. Is 'make up' running?"; exit 1; fi; \
	done
	MLFLOW_TRACKING_URI=http://localhost:5000 $(PYTHON) training/train.py

test: ## Run the test suite
	$(PYTHON) -m pytest tests/ -v

drift: ## Run drift detection report against local stack
	MLFLOW_TRACKING_URI=http://localhost:5000 \
	POSTGRES_HOST=localhost \
	$(PYTHON) monitoring/drift_report.py

# ── Code quality ───────────────────────────────────────────────────────────────
format: ## Auto-format code with Black
	$(BLACK) $(SRC)

lint: ## Lint with Flake8
	$(FLAKE8) $(SRC)

check: ## Check formatting + linting without modifying files (same as CI)
	$(BLACK) --check $(SRC)
	$(FLAKE8) $(SRC)

# ── Version ────────────────────────────────────────────────────────────────────
version: ## Show current project version
	@$(PYTHON) -c "exec(open('__version__.py').read()); print(__version__)"

# ── Clean ──────────────────────────────────────────────────────────────────────
clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -f .coverage
