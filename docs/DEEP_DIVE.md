# ML Monitoring Platform — Deep Dive

A complete technical breakdown of every design decision, alternative considered, and data flow in this project. Written to build a solid mental model, not just describe what exists.

---

## Table of Contents

1. [The Problem This Solves](#1-the-problem-this-solves)
2. [System Architecture](#2-system-architecture)
3. [Data Flow — End to End](#3-data-flow--end-to-end)
4. [Component 1: Training (`training/train.py`)](#4-component-1-training)
5. [Component 2: MLflow — Tracking & Registry](#5-component-2-mlflow--tracking--registry)
6. [Component 3: FastAPI Serving (`api/main.py`)](#6-component-3-fastapi-serving)
7. [Component 4: PostgreSQL — Prediction Logging](#7-component-4-postgresql--prediction-logging)
8. [Component 5: Evidently — Drift Detection (`monitoring/drift_report.py`)](#8-component-5-evidently--drift-detection)
9. [Component 6: Airflow DAG (`dags/drift_detection_dag.py`)](#9-component-6-airflow-dag)
10. [Component 7: Streamlit Dashboard (`dashboard/app.py`)](#10-component-7-streamlit-dashboard)
11. [Component 8: GitHub Actions CI (`drift_retrain.yml`)](#11-component-8-github-actions-ci)
12. [Component 9: Docker Compose](#12-component-9-docker-compose)
13. [Cross-Cutting Design Decisions](#13-cross-cutting-design-decisions)
14. [What Could Go Wrong in Production](#14-what-could-go-wrong-in-production)
15. [How to Extend This Project](#15-how-to-extend-this-project)

---

## 1. The Problem This Solves

Most ML tutorials end at `model.fit()`. The challenge is:

- **Reproducibility**: can you re-run the exact experiment from 3 months ago?
- **Serving**: how does the model get from a `.pkl` file to an API endpoint?
- **Observability**: what are users actually sending to your model? Is the model still working?
- **Data drift**: the world changes. The distribution of inputs your model sees in production will eventually diverge from the distribution it was trained on.
- **Retraining**: when drift is detected, how do you retrain and deploy without manual intervention?

This project builds an opinionated, complete answer to all five using standard industry tools.

---

## 2. System Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              TRAINING                        │
                        │  sklearn breast cancer dataset               │
                        │  → StandardScaler + RandomForestClassifier   │
                        │  → MLflow run (params + metrics + artifact)  │
                        │  → Model Registry: breast-cancer-model v1    │
                        └───────────────────┬─────────────────────────┘
                                            │ registers model
                                            ▼
┌──────────────┐   POST /predict   ┌────────────────────┐    pull latest model
│   Any client │ ─────────────────►│  FastAPI (:8000)   │◄──────────────────────┐
│  (curl, app) │                   └────────┬───────────┘                       │
└──────────────┘                            │                                    │
                                            │ INSERT prediction row              │
                                            ▼                                    │
                                   ┌────────────────────┐          ┌────────────┴───────┐
                                   │  PostgreSQL (:5432) │          │   MLflow (:5000)   │
                                   │  predictions table  │          │  tracking + registry│
                                   └────────┬───────────┘          └────────────────────┘
                                            │                                    ▲
                              ┌─────────────┴──────────────┐                    │
                              │                             │                    │
                   ┌──────────▼──────────┐    ┌────────────▼─────────┐          │
                   │  Airflow (:8080)    │    │  Streamlit (:8501)   │          │
                   │  drift_detection   │    │  Dashboard            │          │
                   │  DAG @ 00:00 daily │    │  KPIs + charts        │          │
                   └──────────┬──────────┘    └──────────────────────┘          │
                              │ drift detected?                                  │
                              ▼                                                  │
                   ┌──────────────────────┐                                      │
                   │  GitHub Actions      │                                      │
                   │  retrain job         │──────────────────────────────────────┘
                   └──────────────────────┘  registers new model version
```

### Services and ports summary

| Service | Port | Role |
|---|---|---|
| PostgreSQL | 5432 | Prediction log + Airflow metadata |
| MLflow | 5000 | Experiment tracker + model registry |
| FastAPI | 8000 | Prediction REST API |
| Airflow Webserver | 8080 | DAG management UI |
| Streamlit | 8501 | Operational dashboard |

---

## 3. Data Flow — End to End

Understanding this one complete journey is the key to understanding the whole system.

### Journey: A single prediction request

**Step 1 — Client sends a POST request**
```json
POST http://localhost:8000/predict
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, ...]
}
```
These 30 floats are the features from the breast cancer dataset: mean radius, mean texture, mean perimeter, etc.

**Step 2 — FastAPI loads the model (first request only)**
```python
model_uri = "models:/breast-cancer-model/latest"
model = mlflow.sklearn.load_model(model_uri)
```
MLflow resolves `latest` to the highest-version model in the registry, downloads the artifact from `/mlflow/artifacts/`, and deserializes the sklearn `Pipeline` object. Subsequent requests reuse the cached `_model` global.

**Step 3 — The sklearn Pipeline runs inference**
```python
features = np.array([17.99, 10.38, ...]).reshape(1, -1)
prediction = model.predict(features)        # → [0] (malignant)
proba = model.predict_proba(features)       # → [[0.87, 0.13]]
confidence = proba[0][prediction[0]]        # → 0.87
```
The `Pipeline` object internally calls `StandardScaler.transform()` then `RandomForestClassifier.predict()`. This is why the scaler is baked into the pipeline — the API does not need to know about feature scaling.

**Step 4 — Prediction is logged to PostgreSQL**
```sql
INSERT INTO predictions (timestamp, input_features, prediction, confidence)
VALUES ('2024-03-15T14:23:01Z', '[17.99, 10.38, ...]', 0, 0.87);
```
`input_features` is stored as JSONB — a binary JSON format that allows indexing and querying inside the array later if needed.

**Step 5 — Response returned to client**
```json
{
  "prediction": 0,
  "confidence": 0.87,
  "label": "malignant",
  "timestamp": "2024-03-15T14:23:01.234Z"
}
```

**Step 6 — Airflow runs at midnight**
The DAG calls `drift_report.py`, which:
- Downloads `training_stats.json` from MLflow (feature means + stds from training)
- Reconstructs a synthetic reference DataFrame (500 Gaussian samples)
- Queries the last 7 days of rows from `predictions`
- Unpacks each row's JSONB features into a DataFrame
- Runs Evidently's `DataDriftPreset`
- Returns True/False

**Step 7 — GitHub Actions (separate, daily cron)**
If drift is detected (exit code 1), the `retrain` job runs `training/train.py`, which logs a new MLflow run and registers a new model version. The FastAPI container will pick this up on its next restart (or can be extended to hot-reload).

---

## 4. Component 1: Training

**File**: `training/train.py`

### What it does

```python
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0=malignant, 1=benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, ...)),
])
```

### Why RandomForest?

| Model | Pros | Cons | Why not chosen here |
|---|---|---|---|
| **RandomForest** | Handles feature scaling internally (via splits), interpretable feature importance, robust to outliers | Slower inference than linear models | **Chosen** — good accuracy + `predict_proba` gives calibrated confidence |
| Logistic Regression | Very fast, interpretable coefficients | Requires linear separability | Simpler, but less representative of real MLOps |
| SVM | Good with high-dimensional data | No native probability output (needs `CalibratedClassifierCV`) | Complicates the confidence scoring |
| XGBoost | Often best accuracy | External dependency, more config | Overkill for a demo dataset |
| Neural Net | Flexible | Needs more data, harder to explain | Wrong tool for tabular data at this scale |

RandomForest wins here because the point is **infrastructure correctness**, not model accuracy. RF gives `predict_proba` natively, handles the feature scale itself inside the Pipeline (though we add a StandardScaler anyway to demonstrate the pattern), and produces reasonable accuracy (~96%) without tuning.

### Why `stratify=y` in train_test_split?

The breast cancer dataset is imbalanced: ~63% benign, ~37% malignant. Without `stratify=y`, a random split might put most malignant cases in training, making the test accuracy misleadingly high. `stratify=y` guarantees each split has the same class ratio as the full dataset.

### Why a sklearn Pipeline instead of separate scaler + model?

**The wrong way:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Later, in the API:
X_scaled = scaler.transform(X_test)  # ← WHERE IS THIS SCALER?
pred = model.predict(X_scaled)
```
This creates a **two-artifact problem**: you have to serialize and load both the scaler and the model separately, and keep them in sync. If you update the model, you need to remember to also deploy the updated scaler.

**The right way (Pipeline):**
```python
pipeline = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())])
pipeline.fit(X_train, y_train)  # scaler learns stats from X_train only
# Saved as ONE artifact. API calls pipeline.predict(raw_features) — done.
```

The Pipeline ensures:
- `fit()` on training data only (prevents data leakage)
- `transform()` is always applied before `predict()` automatically
- One artifact in MLflow, one `load_model()` call in the API

### Why save `training_stats.json`?

The drift detector needs a **reference distribution** — what normal data looks like. Two options:

**Option A: Save the entire training set**
```python
X_train.to_csv("training_data.csv")
mlflow.log_artifact("training_data.csv")
```
569 rows × 30 features. Small here, but for real datasets this is thousands of MB.

**Option B: Save summary statistics**
```python
stats = {
    "feature_means": X_train.mean().tolist(),
    "feature_stds": X_train.std().tolist(),
}
```
This is what we do. The drift checker reconstructs a Gaussian approximation:
```python
samples = rng.normal(loc=mean, scale=std, size=500)
```
This is an approximation (real data is rarely perfectly Gaussian), but it's sufficient for Evidently's statistical tests (Kolmogorov-Smirnov, Wasserstein distance). The trade-off: if the training distribution is highly skewed or multimodal, this approximation degrades. For the breast cancer dataset it works well.

**Option C: Store reference in PostgreSQL**
Keep a `reference_data` table populated at training time. Eliminates the Gaussian approximation. Adds DB dependency to the training script. Viable for production — not chosen here to keep training decoupled from the DB.

---

## 5. Component 2: MLflow — Tracking & Registry

### The two roles MLflow plays

MLflow does two distinct things in this project:

**Role 1: Experiment Tracker**
Stores the metadata of every training run: parameters, metrics, timestamps, and artifact paths. Think of it as a structured `git log` for model training.

```
Run ID: a3f7c2b1...
├── Parameters
│   ├── n_estimators: 100
│   ├── max_depth: 6
│   └── random_state: 42
├── Metrics
│   ├── accuracy: 0.9649
│   ├── f1: 0.9726
│   └── roc_auc: 0.9971
└── Artifacts
    ├── model/         ← the serialized Pipeline
    └── training_stats.json
```

**Role 2: Model Registry**
A separate concept — a named, versioned catalog of models promoted from runs.

```
Model Registry: "breast-cancer-model"
├── Version 1  ← run a3f7c2b1, Staging
├── Version 2  ← run 9d4e1a0f, Production
└── Version 3  ← run f2b8c3d1, None (just registered)
```

### Why MLflow over alternatives?

| Tool | Pros | Cons |
|---|---|---|
| **MLflow** | Open source, self-hosted, sklearn/pytorch/tf native integrations, model registry included | UI is basic, registry transitions are manual | **Chosen** |
| Weights & Biases | Beautiful UI, great for teams | SaaS (costs money), external dependency |
| Neptune.ai | Similar to W&B | Same SaaS concern |
| DVC | Git-native versioning, works with any storage | More complex setup, no built-in registry |
| SageMaker Experiments | AWS-native | Vendor lock-in |

MLflow wins for self-hosted portfolio projects because it runs in Docker with zero external dependencies and has first-class sklearn integration via `mlflow.sklearn.log_model()`.

### How `models:/breast-cancer-model/latest` works

When the API calls `mlflow.sklearn.load_model("models:/breast-cancer-model/latest")`:
1. MLflow client contacts the tracking server at `http://mlflow:5000`
2. Queries the registry for all versions of `breast-cancer-model`
3. Picks the one with the highest version number (`latest` = highest version, not necessarily "Production" stage)
4. Resolves the artifact path: `mlflow-artifacts/0/a3f7c2b1.../artifacts/model`
5. Downloads the artifact to a temp directory
6. Unpickles the sklearn Pipeline

**Stage aliases**: MLflow also supports `models:/name/Production` and `models:/name/Staging`. You'd set these via the UI or `client.transition_model_version_stage()`. We use `latest` to keep the demo simple — in production you'd promote to `Production` stage explicitly.

### MLflow backend configuration

```yaml
mlflow:
  command: >
    mlflow server
    --backend-store-uri postgresql://mluser:mlpassword@postgres/mlmonitoring
    --default-artifact-root /mlflow/artifacts
```

Two storage layers:
- **`--backend-store-uri`**: Where run metadata (params, metrics) is stored. We use PostgreSQL — the same instance as the prediction logs, just different tables. Alternative: SQLite file (simpler but not concurrent-safe).
- **`--default-artifact-root`**: Where binary artifacts (the model pickle, json files) are stored. We use a Docker volume (`mlflow-artifacts:/mlflow/artifacts`). Alternative: S3 bucket, GCS bucket — required for multi-machine deployments.

---

## 6. Component 3: FastAPI Serving

**File**: `api/main.py`

### Lazy model loading

```python
_model = None

def get_model():
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    return _model
```

The model is loaded on the first prediction request, not at startup. Why?

**Problem with eager loading at startup:**
Docker Compose starts all services simultaneously. When the API container starts, MLflow might not have the model registered yet (especially if you forgot to run `training/train.py`). Eager loading would crash the API container immediately.

**Alternative — startup event:**
```python
@app.on_event("startup")
async def load_model():
    get_model()
```
This loads eagerly but after the container is running, so it doesn't crash Docker. The API would return 500s until the model is loaded. Better UX for health checks but same race condition problem.

**What we chose — true lazy loading:**
First `/predict` call triggers the load. The `/health` endpoint always works, even before the model exists. The trade-off: the first prediction is slow (model download). Acceptable for a demo; unacceptable for latency-sensitive production (where you'd use `startup` + retry logic).

### Request validation with Pydantic

```python
class PredictRequest(BaseModel):
    features: list[float]
```

Pydantic automatically:
- Rejects non-numeric values with a 422 error
- Coerces integers to floats
- Returns a structured error body, not a Python traceback

We add an explicit check:
```python
if len(request.features) != 30:
    raise HTTPException(status_code=422, detail=f"Expected 30 features, got {len(request.features)}")
```
This gives a clear error instead of a cryptic numpy reshape error.

### Why not async database writes?

```python
# We do this (synchronous):
conn = get_db_conn()
with conn:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO predictions ...", ...)
conn.close()
```

`psycopg2` is synchronous. For a FastAPI app (which is async), the correct approach for high throughput is `asyncpg` with connection pooling:

```python
# What production would look like:
import asyncpg
pool = await asyncpg.create_pool(...)
async with pool.acquire() as conn:
    await conn.execute("INSERT INTO predictions ...", ...)
```

We use synchronous `psycopg2` here because:
1. It's simpler — fewer abstractions to explain
2. The breast cancer dataset demo has low throughput needs
3. The logging failure is non-fatal (we catch exceptions and return the prediction anyway)

In production with high concurrency, synchronous DB calls in async handlers cause thread starvation. Always use `asyncpg` or run psycopg2 calls via `asyncio.run_in_executor`.

### The `/health` endpoint

```python
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}
```

This is intentionally minimal. It does not check if the model is loaded or if PostgreSQL is reachable. A more production-grade health check:

```python
@app.get("/health")
async def health():
    checks = {"api": "ok"}
    try:
        conn = get_db_conn()
        conn.close()
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"

    try:
        get_model()
        checks["model"] = "ok"
    except Exception:
        checks["model"] = "error"

    status_code = 200 if all(v == "ok" for v in checks.values()) else 503
    return JSONResponse(checks, status_code=status_code)
```

We kept it simple — the deep check is an extension exercise.

---

## 7. Component 4: PostgreSQL — Prediction Logging

**File**: `sql/init.sql`

### Schema design

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_features JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    confidence FLOAT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp);
```

**Why JSONB for `input_features`?**

The 30 breast cancer features could be stored as 30 separate columns:
```sql
-- Option A: wide table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    mean_radius FLOAT,
    mean_texture FLOAT,
    ... -- 28 more columns
);
```
This is perfectly valid but:
- Schema changes require `ALTER TABLE ADD COLUMN` (painful in production)
- If you swap to a different model with different features, you need a new table
- Querying is more complex for bulk analysis

JSONB stores the features as binary JSON:
```sql
input_features = '[17.99, 10.38, 122.8, ...]'
```
Later you can extract individual features:
```sql
SELECT (input_features->0)::float AS mean_radius FROM predictions;
```
Or pass the whole array to Python for Evidently without any column mapping.

The trade-off: JSONB queries are slower than typed column queries for single-feature lookups. For our use case (bulk export for drift analysis), it's faster because we SELECT the entire JSONB column without parsing.

**Why `TIMESTAMPTZ` (timestamp with time zone)?**

Store timestamps in UTC. `TIMESTAMP` without timezone stores local time, which becomes ambiguous when servers change timezone or daylight saving kicks in. `TIMESTAMPTZ` converts to UTC at write time and back to local time at read time. Always use `TIMESTAMPTZ` in new schemas.

**Why the index on timestamp?**

```sql
CREATE INDEX idx_predictions_timestamp ON predictions (timestamp);
```

The drift detector's query:
```sql
SELECT input_features FROM predictions WHERE timestamp >= %s
```
Without the index, PostgreSQL scans every row. With the index, it jumps directly to the rows in the last 7 days. At 10,000 predictions/day × 7 days = 70,000 rows, a sequential scan takes ~10ms and an index scan takes ~0.5ms. At 1M rows/day, the difference is seconds vs. milliseconds.

### Why one PostgreSQL instance with two databases?

```
PostgreSQL
├── database: mlmonitoring  ← predictions table + MLflow tracking tables
└── database: airflow       ← Airflow metadata (DAG runs, task states, variables)
```

Alternative: run two separate PostgreSQL containers. That's cleaner isolation but doubles the container count and resource usage. Since both databases are low-traffic for a demo, sharing a PostgreSQL instance is fine.

The init script trick:
```sql
-- init_airflow.sql
SELECT 'CREATE DATABASE airflow OWNER mluser'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
```
PostgreSQL init scripts run once (on first container start, when the data volume is empty). The `\gexec` pipes the result of the SELECT as a command — this creates the `airflow` database only if it doesn't exist, making the script idempotent.

---

## 8. Component 5: Evidently — Drift Detection

**File**: `monitoring/drift_report.py`

### What is data drift?

Your model was trained on data from January 2024. By December 2024, the real-world distribution of inputs has shifted — maybe tumor measurement technology improved and values are systematically different, or your user base changed.

Data drift ≠ model degradation, but drift is an early warning signal. If inputs look different from training data, predictions become unreliable even if accuracy metrics haven't fallen yet (because you might not have ground truth labels yet).

### What Evidently actually computes

```python
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
```

`DataDriftPreset` runs a statistical test on each feature to determine if the current distribution significantly differs from the reference. The specific test depends on the column type:

| Column type | Default test | What it measures |
|---|---|---|
| Numerical (our case) | Kolmogorov-Smirnov | Maximum difference between CDFs |
| Categorical | Chi-squared | Difference in category frequencies |
| High-cardinality | Wasserstein distance | "Earth mover's distance" between distributions |

**Kolmogorov-Smirnov in plain English:**
Plot the cumulative distribution of a feature in reference data and in current data. KS distance is the maximum vertical gap between those two curves. If this gap is above a threshold (default p-value < 0.05), the distributions are "significantly different."

```
Reference CDF:     ___/‾‾‾
Current CDF:     ___/‾‾‾       ← shifted right (values are larger)
KS distance:         ↕         ← this gap, if large enough → drift
```

**Dataset-level drift:** Evidently flags the entire dataset as drifted if more than a threshold fraction of features drift (default: >50% of features). This avoids false alarms from single noisy features.

### The synthetic reference trick

```python
data = {
    name: rng.normal(loc=mean, scale=max(std, 1e-9), size=500)
    for name, mean, std in zip(FEATURE_NAMES, means, stds)
}
reference = pd.DataFrame(data)
```

We sample 500 synthetic points from a Gaussian matching the training distribution. Why 500? Enough for KS test to have statistical power; small enough to be fast. The `max(std, 1e-9)` guards against features with zero variance (constant features) — numpy would otherwise crash.

**Limitation:** Real distributions are often skewed or bimodal. A Gaussian approximation works reasonably for the breast cancer dataset but would be wrong for, e.g., income data (log-normal) or event counts (Poisson). For production: save the actual training data, or save percentiles (5th, 25th, 50th, 75th, 95th) for better distribution reconstruction.

### Why exit code matters

```python
if __name__ == "__main__":
    drift = run_drift_check()
    sys.exit(1 if drift else 0)
```

UNIX convention: exit code 0 = success, non-zero = failure. GitHub Actions treats any non-zero exit as a step failure. By exiting with code 1 on drift, the CI step "fails", which we use as a conditional trigger for retraining:

```yaml
- name: Run drift check
  id: drift
  run: |
    python monitoring/drift_report.py
    DRIFT_EXIT=$?
    if [ $DRIFT_EXIT -eq 1 ]; then
      echo "drift_detected=true" >> "$GITHUB_OUTPUT"
    fi
```

This is a clean, shell-native way to communicate binary state between a Python script and a CI system.

---

## 9. Component 6: Airflow DAG

**File**: `dags/drift_detection_dag.py`

### What Airflow is and isn't

Airflow is a **workflow orchestrator** — it runs Python functions on a schedule and tracks success/failure. It is not a stream processor, not a message queue, and not a web server (it has a web UI but it's for management, not serving).

Key concepts:
- **DAG** (Directed Acyclic Graph): a collection of tasks with dependencies
- **Operator**: the type of task (PythonOperator, BashOperator, etc.)
- **Scheduler**: Airflow component that triggers DAGs when their schedule is due
- **Webserver**: Airflow UI for viewing DAG runs, logs, task states

### Why LocalExecutor instead of CeleryExecutor?

```yaml
AIRFLOW__CORE__EXECUTOR: LocalExecutor
```

| Executor | How it works | When to use |
|---|---|---|
| **LocalExecutor** | Runs tasks in parallel subprocesses on the same machine | Single machine, moderate task volume |
| CeleryExecutor | Distributes tasks to worker machines via message queue (Redis/RabbitMQ) | Multi-machine, high task volume |
| KubernetesExecutor | Each task is a Kubernetes pod | K8s clusters, variable workloads |
| SequentialExecutor | One task at a time, in-process | SQLite backend, development only |

LocalExecutor is the right choice here: one machine, one DAG, one task per run. CeleryExecutor would require adding Redis to docker-compose, adding a Celery worker container, and configuring broker URLs — all overhead for no benefit.

### The XCom pattern

```python
context["ti"].xcom_push(key="drift_detected", value=drift_detected)
```

XCom (cross-communication) is how Airflow tasks share small values between each other. A downstream task could read this:
```python
drift = context["ti"].xcom_pull(task_ids="run_drift_check", key="drift_detected")
```

We also write to an Airflow Variable:
```python
Variable.set("drift_detected", str(drift_detected))
```
Variables persist across DAG runs (unlike XComs which are per-run). The Streamlit dashboard could query this Variable via the Airflow REST API or a shared database read. In our implementation, we use a file flag `/tmp/drift_detected.txt` as a simpler alternative.

### Why not just use a cron job instead of Airflow?

A Linux cron job could run the drift script:
```
0 0 * * * python /path/to/drift_report.py
```

Airflow adds:
- **Visibility**: UI shows run history, logs, success/failure
- **Retries**: `retries=1, retry_delay=timedelta(minutes=5)` — automatic retry on failure
- **Backfill**: if the server was down yesterday, you can manually trigger a "catch-up" run
- **Dependencies**: in more complex DAGs, Task B only runs if Task A succeeded
- **Alerting**: email/Slack on failure via Airflow connections

For a single daily task, cron is simpler. For any workflow with 2+ steps or that needs visibility, Airflow is worth it.

---

## 10. Component 7: Streamlit Dashboard

**File**: `dashboard/app.py`

### Why Streamlit over Flask/React?

| Tool | Pros | Cons |
|---|---|---|
| **Streamlit** | Python-only, built-in charts, instant hot-reload, zero frontend knowledge | Limited interactivity, not suitable for complex UIs | **Chosen** |
| Flask + Chart.js | Full control over UI | Need HTML/CSS/JS, much more code |
| Grafana | Production-grade dashboards, datasource plugins | YAML config, not code-first |
| Dash (Plotly) | More interactive than Streamlit, still Python | More boilerplate than Streamlit |
| Metabase | SQL-driven BI tool | Not code-first, separate infra |

Streamlit wins for portfolio demos: a recruiter can see a working dashboard from 50 lines of Python.

### The caching pattern

```python
@st.cache_data(ttl=60)
def load_predictions(days: int = 30) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    ...
```

`@st.cache_data(ttl=60)` caches the function's return value for 60 seconds. Without caching, Streamlit re-runs the entire script on every user interaction (slider move, button click), hitting the database every time. With `ttl=60`, the DB is queried at most once per minute.

### Plotly for charts

```python
import plotly.express as px
fig = px.bar(volume, x="date", y="count")
st.plotly_chart(fig, use_container_width=True)
```

Why Plotly over Streamlit's built-in `st.bar_chart()`?

`st.bar_chart()` is simpler but offers no control over labels, colors, or tooltips. `plotly.express` gives hover tooltips, custom axes, and color encoding from 1-2 lines, which looks significantly more polished.

---

## 11. Component 8: GitHub Actions CI

**File**: `.github/workflows/drift_retrain.yml`

### The conditional job pattern

```yaml
jobs:
  drift-check:
    outputs:
      drift_detected: ${{ steps.drift.outputs.drift_detected }}

  retrain:
    needs: drift-check
    if: needs.drift-check.outputs.drift_detected == 'true'
```

This is the key pattern: Job B (`retrain`) only runs if Job A (`drift-check`) sets a specific output. The output is set via `$GITHUB_OUTPUT` in a bash step:

```bash
echo "drift_detected=true" >> "$GITHUB_OUTPUT"
```

Why this is better than failing on drift:
- If `drift-check` job fails (exit 1), GitHub marks it failed and `retrain` is skipped as blocked
- By capturing the exit code and converting to a string output, we can distinguish "drift detected (expected)" from "script crashed (unexpected)"
- The `needs.drift-check.outputs.drift_detected == 'true'` condition is explicit and readable

### Secrets vs. env vars

```yaml
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

GitHub Actions secrets are encrypted at rest, masked in logs, and only available to workflows in the same repository. They're the right place for credentials and external URLs.

The alternative — hardcoding the MLflow URL — would expose your internal services to anyone who views the workflow file. Never do this.

### Why the workflow has two separate pip installs

```yaml
# drift-check job:
pip install mlflow evidently pandas psycopg2-binary scikit-learn

# retrain job:
pip install -r training/requirements.txt
```

Each job runs on a fresh GitHub-hosted runner (a new Ubuntu VM). There's no shared state between jobs. The retrain job uses the requirements file; the drift job installs inline because it's simpler than maintaining a separate `monitoring/requirements.txt`.

---

## 12. Component 9: Docker Compose

**File**: `docker-compose.yml`

### Health checks and `depends_on`

```yaml
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U mluser -d mlmonitoring"]
    interval: 5s
    retries: 10

mlflow:
  depends_on:
    postgres:
      condition: service_healthy
```

Without `condition: service_healthy`, `depends_on` only waits for the container to *start*, not for PostgreSQL to be *ready to accept connections*. PostgreSQL takes a few seconds to initialize. A bare `depends_on: postgres` causes MLflow to crash on first connection attempt because PostgreSQL is still starting.

`pg_isready` is a PostgreSQL utility that returns 0 when the server is accepting connections. The health check runs it every 5 seconds, up to 10 times (50 seconds total). This is enough for any reasonable machine.

### The x-airflow-common YAML anchor

```yaml
x-airflow-common: &airflow-common
  image: apache/airflow:2.8.1
  environment: ...
  volumes: ...

services:
  airflow-webserver:
    <<: *airflow-common
    command: webserver

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
```

`&airflow-common` defines a YAML anchor (a reusable block). `<<: *airflow-common` merges it into the service definition. Without this, the image, environment variables, and volumes would be duplicated 3 times (init, webserver, scheduler). YAML anchors are Docker Compose's answer to DRY — no actual language feature, just YAML syntax.

### Named volumes vs. bind mounts

```yaml
volumes:
  postgres-data:         # named volume
  mlflow-artifacts:      # named volume

services:
  postgres:
    volumes:
      - postgres-data:/var/lib/postgresql/data   # named volume
      - ./sql/init.sql:/docker-entrypoint-initdb.d/01_init.sql  # bind mount
```

| Type | What it is | Persists across `down`/`up`? | Visible on host? |
|---|---|---|---|
| Named volume | Docker-managed directory | Yes | No (in Docker's internal storage) |
| Bind mount (`./host:container`) | Direct link to host path | Yes (it's just a folder) | Yes |

We use named volumes for PostgreSQL data and MLflow artifacts so they survive `docker compose down` / `docker compose up` cycles. We use bind mounts for:
- `./sql/*.sql` → init scripts (Docker copies these into the container on first start)
- `./dags` → Airflow reads these at runtime, so bind mount allows editing DAGs without rebuilding
- `./monitoring` → same reason as dags

**Gotcha**: named volumes are NOT deleted by `docker compose down`. You need `docker compose down -v` to delete them. This is intentional — protect your data.

---

## 13. Cross-Cutting Design Decisions

### 1. The Pipeline-as-artifact pattern

The single most important decision: bake preprocessing into the model artifact.

**Before (wrong):**
```
Training:  scaler.fit(X) → scaler.pkl  +  model.fit(X_scaled) → model.pkl
Serving:   load scaler.pkl, load model.pkl, scale then predict
```

**After (right):**
```
Training:  pipeline.fit(X) → pipeline.pkl (contains scaler + model)
Serving:   load pipeline.pkl, predict (scaling happens internally)
```

This eliminates an entire class of bugs: "training-serving skew" where the serving code applies a different transformation than what the model was trained with.

### 2. Configuration via environment variables

Every service reads its configuration from environment variables, not from hardcoded values or config files:

```python
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
```

The default (`"localhost"`) makes the script runnable outside Docker. Inside Docker, the environment variables are set by `docker-compose.yml`. For GitHub Actions, they come from secrets.

This follows the [12-factor app methodology](https://12factor.net/config): store config in the environment, not in code.

### 3. Fail-safe prediction logging

```python
try:
    conn = get_db_conn()
    with conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO predictions ...", ...)
    conn.close()
except Exception as e:
    logger.error(f"Failed to log prediction to DB: {e}")
    # No re-raise — we still return the prediction
```

If PostgreSQL is down, the API continues serving predictions — it just doesn't log them. The alternative — raising an exception and returning 500 — would make the API completely unavailable whenever PostgreSQL has trouble. For a model serving system, logging is a side effect; the prediction is the core function.

### 4. JSONB for flexible feature storage

The `input_features JSONB` column stores the entire feature vector as a JSON array. This allows:
- Storing features from different model versions without schema migrations
- Querying individual features with PostgreSQL JSON operators: `input_features->0` for the first feature
- Passing the entire row to Evidently without any column-mapping code

---

## 14. What Could Go Wrong in Production

Understanding failure modes is as important as understanding happy paths.

### Model staleness

The API caches the model in memory:
```python
_model = None
def get_model():
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model(...)
    return _model
```

After retraining registers a new model version, the running API still serves the old one. Fixes:
1. **Container restart**: `docker compose restart api` — simplest, causes ~2s downtime
2. **Model version check endpoint**: `GET /model-version` returns current version; CI can trigger a restart when the version changes
3. **Background refresh thread**: check for new versions every N minutes and reload in the background

### Drift detection false positives

The KS test at p=0.05 means there's a 5% chance of flagging drift when the distribution hasn't actually changed. With 30 features, even if none drifted, you'd expect 1-2 false positives per run. Evidently's dataset-level drift flag (>50% of features must drift) mitigates this, but it's not zero.

Mitigation: raise the threshold (`stattest_threshold=0.01`) or require multiple consecutive drift detections before retraining.

### Single-node Airflow

Our Airflow setup (LocalExecutor, single webserver+scheduler container) has no fault tolerance. If the container crashes, scheduling stops. For production: use CeleryExecutor with multiple workers and a separate scheduler.

### Secrets in `.env`

The `.env` file contains real passwords. It's committed to the repo by default. Add `.env` to `.gitignore` and provide a `.env.example` with placeholder values.

---

## 15. How to Extend This Project

### Add a second model

1. Train with a different `MLFLOW_EXPERIMENT_NAME` and `MLFLOW_MODEL_NAME`
2. Add a `model_name` field to `PredictRequest`
3. Use a dict cache: `_models = {}; _models[name] = mlflow.sklearn.load_model(...)`

### Add ground truth labels

When actual outcomes are known (e.g., biopsy results), store them and compute accuracy over time:
```sql
ALTER TABLE predictions ADD COLUMN ground_truth INTEGER;
```
Track accuracy as a time series in the dashboard — this is "model performance monitoring" vs. "data drift monitoring".

### Add Prometheus + Grafana

Instead of Streamlit, expose metrics via a `/metrics` endpoint:
```python
from prometheus_client import Counter, Histogram
PREDICTION_COUNT = Counter("predictions_total", "Total predictions", ["label"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
```
Then Grafana scrapes Prometheus for production-grade dashboards with alerting.

### Add model explainability

```python
import shap
explainer = shap.TreeExplainer(model.named_steps["classifier"])
shap_values = explainer.shap_values(features)
```
Store SHAP values alongside predictions to understand which features drove each prediction — critical for regulated industries.

### Replace GitHub Actions retraining with a proper trigger

Rather than a cron in CI, use an Airflow DAG that:
1. Detects drift
2. Triggers retraining via an MLflow Projects run
3. Evaluates the new model on a holdout set
4. Only promotes to `Production` stage if the new model is better than the current one

This is the MLOps "champion/challenger" pattern.
