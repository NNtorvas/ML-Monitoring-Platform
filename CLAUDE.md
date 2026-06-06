# CLAUDE.md — ML Monitoring Platform

## Project Purpose

Portfolio MLOps project demonstrating the full model lifecycle: training → serving → monitoring → automated retraining. Target audience: ML engineering recruiters. Keep everything clean and completable — no over-engineering.

## Stack

| Layer | Tool | Version |
|---|---|---|
| Model | scikit-learn RandomForest | 1.4.1.post1 |
| Tracking | MLflow | 3.13.0 |
| Serving | FastAPI + uvicorn | 0.110.0 |
| Drift detection | Evidently AI | 0.4.22 |
| Scheduling | Apache Airflow | 2.8.1 |
| Database | PostgreSQL | 15 |
| Dashboard | Streamlit | 1.32.2 |
| Infra | Docker Compose | - |
| CI/CD | GitHub Actions | - |
| Formatting | Black | 24.4.2 |
| Linting | Flake8 | 7.1.0 |
| Hooks | pre-commit | 4.6.0 |
| Dev automation | Make | - |

## File Map

```
training/train.py          ← train + log to MLflow + register model
api/main.py                ← FastAPI: POST /predict, GET /health
monitoring/drift_report.py ← Evidently drift check, exits 1 on drift
dags/drift_detection_dag.py← Airflow DAG (daily drift check)
dashboard/app.py           ← Streamlit: KPIs + charts from PostgreSQL
sql/init.sql               ← predictions table schema
sql/init_airflow.sql       ← creates airflow database
docker-compose.yml         ← all services + volumes
.env                       ← all environment variables
__version__.py             ← semver string (bump before every push to main)
pyproject.toml             ← Black + Flake8 config
Makefile                   ← dev shortcuts: up/down/train/drift/check/etc.
scripts/check_version_bump.py ← pre-push local version gate
.pre-commit-config.yaml    ← hooks: hygiene + Black + Flake8 + version gate
.github/workflows/drift_retrain.yml ← CI: daily drift check → retrain
.github/workflows/cd.yml            ← CD: push to main → tag + push images
.github/workflows/_prep.yml         ← reusable: validate semver + git tag
.github/workflows/_build-push.yml   ← reusable: parallel api + dashboard → GHCR
README.md                  ← recruiter-facing documentation
docs/OVERVIEW.md           ← plain-English guide: what it does, technologies, example
docs/DEEP_DIVE.md          ← full technical study guide
```

## Running the Stack

```bash
# Step 1: install dev tooling (once after clone)
make install
make hooks

# Step 2: start everything
make up

# Step 3: train once (after MLflow is healthy, ~10s)
make train

# Raw docker commands still work if Make is unavailable:
# docker compose up -d postgres mlflow
# MLFLOW_TRACKING_URI=http://localhost:5000 python training/train.py
# docker compose up -d
```

## Key Design Decisions to Preserve

**Pipeline-as-artifact**: The sklearn `StandardScaler` is baked into the `Pipeline` saved to MLflow. The API calls `model.predict(raw_features)` — no preprocessing logic in the API. Do not break this apart.

**JSONB for features**: `predictions.input_features` is `JSONB` storing the full feature vector. Evidently reads it with `json.loads(row["input_features"])`. Do not change to separate columns without updating `drift_report.py`.

**Synthetic reference**: The drift checker reconstructs a Gaussian reference from `training_stats.json` (means/stds). The artifact is logged at training time. If `training_stats.json` format changes, update both `train.py` and `drift_report.py`.

**Lazy model loading**: The FastAPI `_model` global is populated on first `/predict` call. This is intentional — allows the API to start before MLflow has a registered model.

**Fail-safe logging**: Prediction DB writes are wrapped in try/except and failures are logged but not raised. The prediction is returned regardless. Do not change this to block on DB failures.

## Environment Variables

All config flows via env vars (12-factor). Defaults point to localhost for running outside Docker:

```python
os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.getenv("POSTGRES_HOST", "localhost")
os.getenv("MLFLOW_MODEL_NAME", "breast-cancer-model")
```

Inside Docker, docker-compose.yml overrides these with service names (`mlflow`, `postgres`).

## Ports

| Port | Service |
|---|---|
| 5432 | PostgreSQL |
| 5000 | MLflow |
| 8000 | FastAPI |
| 8080 | Airflow |
| 8501 | Streamlit |

## Database Schema

```sql
-- mlmonitoring database
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_features JSONB NOT NULL,    -- full 30-feature vector
    prediction INTEGER NOT NULL,      -- 0=malignant, 1=benign
    confidence FLOAT NOT NULL         -- predict_proba score for predicted class
);
```

## Model Registry

Model registered as `breast-cancer-model`. The API loads `models:/breast-cancer-model/latest`. After retraining, a new version is automatically registered — but the running API container must be restarted to pick it up.

## GitHub Actions Secrets Required

| Secret | Value |
|---|---|
| `MLFLOW_TRACKING_URI` | Public URL of deployed MLflow instance |
| `POSTGRES_HOST` | Hostname of deployed PostgreSQL |
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |

## No LLMs

This project uses no LLM APIs (no OpenAI, no Anthropic). Pure MLOps. Do not add LLM dependencies.

## Extending

- To add a new model: use a different `MLFLOW_MODEL_NAME` and add model routing to the API
- To add ground truth labels: `ALTER TABLE predictions ADD COLUMN ground_truth INTEGER`
- To add Prometheus metrics: `pip install prometheus-fastapi-instrumentator` and expose `/metrics`
- To replace Streamlit: the dashboard only reads from PostgreSQL — swap freely

## Common Issues

**API returns 500 on first request**: MLflow doesn't have a registered model yet. Run `training/train.py` first.

**Airflow DAG not appearing**: The `dags/` directory is bind-mounted. Make sure the scheduler is running (`docker compose ps`).

**PostgreSQL health check failing**: Container is still initializing. Wait 10-15 seconds and retry.

**Drift always detected**: Synthetic reference is Gaussian; if live predictions cluster differently, drift fires. This is expected behavior — it means the live data genuinely differs from the training distribution assumption.

## CI/CD

**CD workflow** (`.github/workflows/cd.yml`): fires on every push to `main`. Calls `_prep.yml`
(semver validation + annotated git tag), then `_build-push.yml` (parallel GHCR pushes for `api`
and `dashboard`). Uses registry layer caching — incremental rebuilds are fast.

**Version gate**: `__version__.py` at the project root is the single source of truth. Both the
pre-push hook (`scripts/check_version_bump.py`) and `_prep.yml` enforce that the version is
strictly greater than the latest git tag. Always bump this file before pushing to `main`.

**Image names**:
- `ghcr.io/{owner}/ml-monitoring-platform-api:{version}`
- `ghcr.io/{owner}/ml-monitoring-platform-dashboard:{version}`

Both are also tagged with the short commit SHA and `latest` (on default branch only).

**Pre-commit hooks** run automatically on `git commit` (Black, Flake8, file hygiene) and
`git push` (version gate). Install once with `make hooks`. Run manually with `pre-commit run --all-files`.

**Makefile targets** (run `make help` for the full list): `up`, `down`, `build`, `logs`, `train`,
`drift`, `format`, `lint`, `check`, `version`, `clean`, `install`, `hooks`.
