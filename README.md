# ML Monitoring Platform

A production-grade MLOps pipeline demonstrating the full model lifecycle: training, serving, monitoring, and automated retraining.

---

## What This Proves

| Concern | How It's Addressed |
|---|---|
| **Model tracking** | Every training run logs parameters, metrics, and the model artifact to MLflow. Reproducible by run ID. |
| **Model registry** | The trained model is registered in the MLflow Model Registry with versioning and stage transitions. |
| **Production serving** | FastAPI loads the *latest registered model* at startup and serves real-time predictions with sub-50ms latency. |
| **Observability** | Every prediction (features, label, confidence, timestamp) is written to PostgreSQL for auditing and drift analysis. |
| **Data drift detection** | Evidently AI compares the last 7 days of live predictions to the training distribution daily. |
| **Automated retraining** | GitHub Actions runs the drift check on a cron; if drift is detected, it retrains and registers a new model version automatically. |
| **Operational dashboard** | Streamlit surfaces KPIs (predictions today, avg confidence, drift status) and a volume chart — no manual SQL needed. |
| **Reproducible infra** | Docker Compose wires all services together with named volumes so data survives restarts. |

---

## Architecture

```
┌──────────────┐     POST /predict      ┌─────────────────┐
│   Client     │ ─────────────────────► │  FastAPI (:8000) │
└──────────────┘                        └────────┬────────┘
                                                 │ log prediction
                                      ┌──────────▼──────────┐
                                      │   PostgreSQL (:5432) │
                                      └──────────┬──────────┘
                                                 │
              ┌──────────────────────────────────┤
              │                                  │
   ┌──────────▼──────────┐           ┌───────────▼────────┐
   │  Airflow (:8080)    │           │  Streamlit (:8501) │
   │  drift_detection    │           │  Dashboard          │
   │  DAG (daily)        │           └────────────────────┘
   └──────────┬──────────┘
              │ fetch artifacts
   ┌──────────▼──────────┐
   │  MLflow (:5000)     │
   │  tracking + registry│
   └─────────────────────┘
              ▲
              │ register model
   ┌──────────┴──────────┐
   │  training/train.py  │◄── GitHub Actions (on drift)
   └─────────────────────┘
```

---

## Stack

- **scikit-learn** — RandomForest binary classifier (breast cancer dataset)
- **MLflow** — experiment tracking, artifact storage, model registry
- **FastAPI** — REST prediction API with automatic OpenAPI docs
- **Evidently AI** — statistical data drift detection
- **Apache Airflow** — scheduled DAG for daily drift checks
- **PostgreSQL** — prediction log store and Airflow metadata DB
- **Streamlit** — operational dashboard
- **Docker Compose** — local orchestration with persistent volumes
- **GitHub Actions** — CD pipeline (versioned image builds) + drift-triggered retraining
- **pre-commit** — local hooks: Black formatting, Flake8 linting, version gate on push
- **Makefile** — developer shortcuts for docker, training, drift, and code quality

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for training and local dev tooling)
- Make (pre-installed on Mac/Linux; on Windows use WSL or Git Bash)

### 1. Clone and install dev tooling

```bash
git clone <repo-url>
cd ML-Monitoring-Platform
make install   # creates .venv, installs training deps + Black/Flake8
make hooks     # installs pre-commit hooks (run once after clone)
```

### 2. Start the infrastructure

```bash
make up
# wait ~10s for postgres to be healthy
```

### 3. Train and register the model

```bash
make train
```

This logs metrics, saves `training_stats.json` as an artifact (used for drift reference), and registers `breast-cancer-model` in the MLflow registry.

### 4. All services are already running

`make up` starts everything. Check status with `make logs`.

| Service | URL |
|---|---|
| FastAPI (API docs) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Airflow | http://localhost:8080 (admin/admin) |
| Streamlit Dashboard | http://localhost:8501 |

### 5. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

### 6. GitHub Actions setup

**Secrets required** (Settings → Secrets and variables → Actions):
- `MLFLOW_TRACKING_URI` — public URL of your MLflow instance
- `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

**CD pipeline** (`cd.yml`): triggers on every push to `main`. Validates that `__version__.py` was bumped, creates an annotated git tag, then builds and pushes the `api` and `dashboard` Docker images to GHCR in parallel.

**Drift & retrain** (`drift_retrain.yml`): runs daily at 06:00 UTC. If drift is detected, retrains and registers a new model version. Trigger manually via **Actions → Drift Check & Auto-Retrain → Run workflow**.

**Before pushing to main:** bump `__version__.py` or the pre-push hook (and the CD workflow) will block the push.

---

## Local Development

```bash
make help       # list all available commands

make format     # auto-format all Python with Black
make lint       # run Flake8
make check      # check formatting + linting (same gates as CI)

make up         # start all services
make down       # stop all services
make build      # rebuild api + dashboard images
make logs       # tail service logs

make train      # retrain model (MLflow must be running)
make drift      # run drift report against local stack

make version    # show current version from __version__.py
make clean      # remove __pycache__ and build artifacts
```

---

## Component Deep-Dive

### `training/train.py`
Loads the sklearn breast cancer dataset, trains a `StandardScaler → RandomForestClassifier` pipeline, logs all params and metrics to MLflow, uploads `training_stats.json` (feature means/stds) for drift reference, and registers the model.

### `api/main.py`
FastAPI app that lazy-loads the latest registered model from MLflow on the first request. Every call to `POST /predict` writes the features, prediction, confidence, and timestamp to the `predictions` table in PostgreSQL.

### `monitoring/drift_report.py`
Fetches the training reference distribution from MLflow artifacts, queries the last 7 days of live predictions from PostgreSQL, runs an Evidently `DataDriftPreset` report, and exits with code 1 if drift is detected (used as a CI gate).

### `dags/drift_detection_dag.py`
Airflow DAG that runs `drift_report.py` daily and stores the result in an Airflow Variable so the dashboard can read it.

### `dashboard/app.py`
Streamlit app showing: total predictions today, average confidence, drift status (yes/no), prediction volume bar chart, and a confidence scatter plot. Refreshes every 60 seconds from PostgreSQL.

### `docker-compose.yml`
Brings up PostgreSQL, MLflow, FastAPI, Airflow (webserver + scheduler + init), and Streamlit. PostgreSQL and MLflow artifacts use named volumes so data persists between `docker compose down` / `up` cycles.

### `.github/workflows/drift_retrain.yml`
Runs `drift_report.py` on a daily cron. If drift is detected (exit code 1), it triggers the retrain job which runs `training/train.py` and registers a new model version in MLflow.

---

## Key Design Decisions

- **Pipeline wrapping the model**: the `StandardScaler` is baked into the MLflow artifact, so the API never needs to know about feature scaling — inference is a single `model.predict()` call.
- **Synthetic reference reconstruction**: rather than storing the full training set (569 rows), only means and stds are stored in MLflow. The drift checker reconstructs a Gaussian reference — sufficient for Evidently's statistical tests.
- **No LLM dependencies**: pure MLOps stack. Every component is deterministic and infrastructure-focused.
- **Single PostgreSQL instance, two databases**: `mlmonitoring` for predictions, `airflow` for Airflow metadata. Reduces container count while keeping schemas isolated.
