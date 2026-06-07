# ML Monitoring Platform — Technical Reference

## What It Does

End-to-end MLOps platform that covers the full model lifecycle:

```
Train → Register → Serve → Monitor → Auto-Retrain
```

Trains a breast cancer classifier (scikit-learn RandomForest), serves predictions via REST API,
tracks model drift in production, and triggers automated retraining when drift is detected.

---

## Architecture Overview

```
┌────────────┐     POST /predict     ┌──────────────┐
│   Client   │ ───────────────────→  │  FastAPI API  │
└────────────┘                       └──────┬───────┘
                                            │ loads model
                                     ┌──────▼───────┐
                                     │    MLflow    │  ← model registry + experiment tracking
                                     └──────────────┘
                                            │ logs prediction
                                     ┌──────▼───────┐
                                     │  PostgreSQL  │  ← stores predictions + features (JSONB)
                                     └──────┬───────┘
                                            │ reads live data
                                     ┌──────▼───────┐
                                     │   Evidently  │  ← drift detection (vs. training reference)
                                     └──────┬───────┘
                                            │ triggered by
                                     ┌──────▼───────┐
                                     │   Airflow    │  ← daily DAG: drift check → retrain if drift
                                     └──────────────┘
                                            │
                                     ┌──────▼───────┐
                                     │  Streamlit   │  ← dashboard: KPIs + charts from PostgreSQL
                                     └──────────────┘
```

---

## Tech Stack

| Concern           | Tool                    | Notes                                      |
|-------------------|-------------------------|--------------------------------------------|
| ML model          | scikit-learn 1.9+       | RandomForestClassifier, breast cancer data |
| Preprocessing     | sklearn Pipeline        | StandardScaler baked in — no logic in API  |
| Experiment tracking | MLflow 3.13           | Logs params, metrics, artifacts, model     |
| Model serving     | FastAPI + uvicorn       | POST /predict, GET /health                 |
| Drift detection   | Evidently AI 0.4.22     | Compares live vs. synthetic Gaussian ref   |
| Orchestration     | Apache Airflow 2.8.1    | Daily drift → conditional retrain DAG     |
| Database          | PostgreSQL 15           | Predictions table with JSONB feature store |
| Dashboard         | Streamlit 1.32.2        | Reads directly from PostgreSQL             |
| Infrastructure    | Docker Compose          | All services containerized                 |
| CI/CD             | GitHub Actions          | Drift check, image build/push to GHCR      |
| Code quality      | Black + Flake8          | Enforced via pre-commit hooks              |

---

## Services & Ports

| Port  | Service    |
|-------|------------|
| 5432  | PostgreSQL |
| 5000  | MLflow UI  |
| 8000  | FastAPI    |
| 8080  | Airflow    |
| 8501  | Streamlit  |

---

## Key Data Flows

### Training
```
train.py
  → fit Pipeline(StandardScaler + RandomForest) on breast cancer dataset
  → log params + metrics to MLflow
  → save training_stats.json (means/stds) as MLflow artifact
  → register model as "breast-cancer-model" in MLflow registry
```

### Prediction
```
POST /predict  {features: [...30 floats...]}
  → lazy-load model from MLflow on first call
  → model.predict(raw_features)  ← scaler inside the pipeline
  → write to PostgreSQL: timestamp, JSONB features, prediction, confidence
  → return {prediction, confidence, model_version}
```

### Drift Detection
```
Airflow DAG (daily)
  → drift_report.py
      → fetch last N predictions from PostgreSQL
      → reconstruct reference distribution from training_stats.json
      → run Evidently DatasetDriftReport
      → exit 0 = no drift, exit 1 = drift detected
  → if drift: trigger retrain task → re-run train.py
```

---

## Database Schema

```sql
CREATE TABLE predictions (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_features  JSONB       NOT NULL,   -- full 30-feature vector
    prediction      INTEGER     NOT NULL,   -- 0 = malignant, 1 = benign
    confidence      FLOAT       NOT NULL    -- predict_proba score
);
```

---

## CI/CD Pipeline

```
git push → main
  ├── version gate (semver must be bumped)
  ├── annotated git tag created
  └── parallel Docker image builds → GHCR
        ├── ghcr.io/{owner}/ml-monitoring-platform-api:{version}
        └── ghcr.io/{owner}/ml-monitoring-platform-dashboard:{version}

GitHub Actions (scheduled daily)
  └── drift check → retrain if needed
```

---

## Key Design Decisions

| Decision                      | Why it matters                                                  |
|-------------------------------|------------------------------------------------------------------|
| Pipeline-as-artifact          | Scaler travels with model; API stays stateless                  |
| JSONB feature storage         | Full feature vector preserved; Evidently can reconstruct inputs |
| Synthetic Gaussian reference  | No need to store training data; means/stds are enough           |
| Lazy model loading in API     | API starts before MLflow has a registered model                 |
| Fail-safe DB writes           | DB failure never blocks a prediction response                   |
| 12-factor env config          | Same codebase runs locally and in Docker with zero changes      |

---

## Quick Start

```bash
make install    # install dev tooling
make hooks      # install pre-commit hooks
make up         # start all services
make train      # train + register model (after MLflow is up ~10s)
# → API at http://localhost:8000
# → MLflow at http://localhost:5000
# → Dashboard at http://localhost:8501
```
