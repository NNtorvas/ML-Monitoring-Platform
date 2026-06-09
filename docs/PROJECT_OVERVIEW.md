# ML Monitoring Platform — Project Overview

## What It Is

An end-to-end MLOps platform covering the full model lifecycle: **train → serve → monitor → retrain**. A RandomForest classifier on the breast cancer dataset serves as the model, but the infrastructure is the substance of the project — the goal is a working, observable, self-maintaining ML system.

**The problem it addresses:** Most ML projects end at `model.fit()`. This platform answers what comes after: how do you version experiments, serve a model as an API, detect when real-world data drifts from the training distribution, and automatically retrain — without manual intervention?

---

## Architecture

```
Training ──► MLflow Registry ──► FastAPI (serve predictions)
                                         │
                                         ▼
                                    PostgreSQL (log predictions)
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                       Airflow DAG           Streamlit Dashboard
                      (daily drift check)    (KPIs + charts)
                              │
                         drift detected?
                              ▼
                       GitHub Actions
                       (retrain job → new MLflow model version)
```

---

## Stack

| Layer | Tool | Rationale |
|---|---|---|
| Model | scikit-learn RandomForest | Native `predict_proba`, Pipeline support, ~96% accuracy without hyperparameter tuning |
| Experiment tracking | MLflow 3.x | Self-hosted, Docker-native, first-class sklearn integration |
| Serving | FastAPI | Async, Pydantic request validation, automatic OpenAPI docs |
| Drift detection | Evidently AI | Per-feature statistical tests (KS, chi-squared), single `Report` call |
| Scheduling | Apache Airflow | Retries, run history, conditional task execution |
| Database | PostgreSQL | Single instance shared by predictions log, MLflow metadata, and Airflow metadata |
| Dashboard | Streamlit | Operational dashboard in ~50 lines of Python |
| Infra | Docker Compose | All 5 services + named volumes in one file |
| CI/CD | GitHub Actions | Drift check → conditional retrain; CD builds to GHCR on push to main |

---

## Services & Ports

| Service | Port | Role |
|---|---|---|
| PostgreSQL | 5432 | Predictions + MLflow metadata + Airflow metadata |
| MLflow | 5000 | Experiment tracker + model registry |
| FastAPI | 8000 | `POST /predict`, `GET /health` |
| Airflow | 8080 | DAG management UI |
| Streamlit | 8501 | Operational dashboard |

---

## Key Design Decisions

### Pipeline-as-artifact
The `StandardScaler` is baked into the sklearn `Pipeline` object saved to MLflow. The API calls `model.predict(raw_features)` with no preprocessing logic. This eliminates training-serving skew: the same transformation applied during training is guaranteed to run during inference, because it is the same object.

### JSONB for feature storage
`predictions.input_features` is a `JSONB` column storing the full 30-feature vector as a JSON array. This keeps the schema stable across model versions — if the feature set changes, no `ALTER TABLE` is needed. Evidently also reads the column directly without any mapping code.

### Lazy model loading
The FastAPI `_model` global starts as `None` and is populated on the first `/predict` call. Docker Compose starts all containers simultaneously, so eager loading at startup would race with MLflow model registration. The trade-off is a slow first request; `GET /health` works regardless.

### Fail-safe prediction logging
Database writes are wrapped in `try/except`. If PostgreSQL is unavailable, the prediction is returned to the caller anyway and the error is logged. The prediction is the primary function; logging is a side effect and should not block it.

### Synthetic Gaussian reference for drift detection
Rather than saving the full training set (which scales to GBs in real projects), `train.py` logs `training_stats.json` containing per-feature means and standard deviations. The drift checker reconstructs a 500-sample Gaussian reference from these stats and passes it to Evidently. The trade-off: the Gaussian approximation degrades if the true training distribution is heavily skewed or multimodal. For this dataset it is sufficient.

### Exit-code-driven CI branching
`drift_report.py` exits with code `1` on drift, `0` otherwise. GitHub Actions captures this via shell `&&`/`||` into a named job output, which the downstream retrain job reads as a conditional. This converts a binary Python signal into a clean, readable CI branch without relying on job failure state.

### 12-factor configuration
All service configuration flows through `os.getenv("VAR", "localhost_default")`. The localhost defaults make every script runnable outside Docker. `docker-compose.yml` overrides with Docker service names; GitHub Actions uses repository secrets for deployed environments.

---

## Data Flow

1. Client sends `POST /predict` with 30 floats (breast cancer feature vector)
2. API lazy-loads model: `mlflow.sklearn.load_model("models:/breast-cancer-model/latest")`
3. sklearn Pipeline runs `StandardScaler.transform()` → `RandomForestClassifier.predict()`
4. Response (`prediction`, `confidence`, `label`, `timestamp`) returned to client
5. Feature vector + result inserted into `predictions` as a JSONB row
6. At midnight, Airflow DAG runs `drift_report.py`:
   - Downloads `training_stats.json` from MLflow artifacts
   - Reconstructs Gaussian reference (500 samples per feature)
   - Queries the last 7 days of `predictions`
   - Runs Evidently `DataDriftPreset` (Kolmogorov-Smirnov per numerical feature)
   - Dataset flagged as drifted if > 50% of features drift (reduces false positives)
   - Exits 0 (stable) or 1 (drift detected)
7. If drift detected → GitHub Actions retrain job runs `train.py`, registers a new model version in MLflow

---

## Model Details

- **Dataset**: `sklearn.datasets.load_breast_cancer` — 569 samples, 30 numerical features, binary labels (0 = malignant, 1 = benign)
- **Split**: 80/20 with `stratify=y` — the dataset is ~63% benign / ~37% malignant; stratification preserves this ratio in both splits
- **Pipeline**: `StandardScaler` → `RandomForestClassifier(n_estimators=100, max_depth=6)`
- **Metrics tracked**: accuracy, F1, ROC-AUC (all logged to MLflow per run)
- **Model URI**: `models:/breast-cancer-model/latest`

---

## CI/CD

**On push to `main`**:
1. `_prep.yml` — validates that `__version__.py` is strictly greater than the latest git tag, then creates an annotated tag
2. `_build-push.yml` — parallel Docker builds for `api` and `dashboard`, pushed to GHCR tagged with version, short SHA, and `latest`

**Daily cron (06:00 UTC)**:
- `drift_retrain.yml` — runs the drift check; if drift is detected, runs `training/train.py` to register a new model version

**Pre-commit hooks (local)**: Black formatting, Flake8 linting, version bump gate (`scripts/check_version_bump.py`)

**Image names**:
- `ghcr.io/{owner}/ml-monitoring-platform-api:{version}`
- `ghcr.io/{owner}/ml-monitoring-platform-dashboard:{version}`

---

## Possible Extensions

- **Ground truth labels**: `ALTER TABLE predictions ADD COLUMN ground_truth INTEGER` — enables tracking model accuracy over time alongside drift signals
- **Prometheus + Grafana**: expose `/metrics` from the API for production-grade dashboards and alerting thresholds
- **SHAP values**: store per-prediction feature attributions alongside predictions for model explainability
- **Champion/challenger promotion**: evaluate the retrained model on a holdout set before promoting it to the `Production` stage in the MLflow registry
- **Async database writes**: replace `psycopg2` with `asyncpg` + connection pooling for high-throughput serving
- **Multi-model routing**: add a `model_name` field to the request and a dict-based model cache in the API
