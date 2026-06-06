"""
Train a binary classifier on the breast cancer dataset and register it in MLflow.
Run this once locally (or in CI) before starting the stack:
  pip install -r training/requirements.txt
  python training/train.py
"""

import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "breast-cancer-classifier")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "breast-cancer-model")

mlflow.set_tracking_uri(TRACKING_URI)

# Ensure the experiment uses mlflow-artifacts:/ so artifact uploads go through the
# tracking server HTTP proxy (required when training runs outside Docker).
# If a stale experiment exists with a local-filesystem artifact root, rename it out
# of the way so set_experiment creates a fresh one with the correct URI.
_client = mlflow.MlflowClient()
_exp = _client.get_experiment_by_name(EXPERIMENT_NAME)
if _exp is not None:
    if _exp.lifecycle_stage == "deleted":
        _client.restore_experiment(_exp.experiment_id)
        _exp = _client.get_experiment_by_name(EXPERIMENT_NAME)
    if not str(_exp.artifact_location or "").startswith("mlflow-artifacts"):
        _client.rename_experiment(_exp.experiment_id, f"{EXPERIMENT_NAME}_legacy")

mlflow.set_experiment(EXPERIMENT_NAME)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 4,
    "random_state": 42,
}

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(**params)),
    ]
)

with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Save training feature names and distribution stats for drift detection
    training_stats = {
        "feature_names": list(data.feature_names),
        "feature_means": X_train.mean().tolist(),
        "feature_stds": X_train.std().tolist(),
    }
    with open("training_stats.json", "w") as f:
        json.dump(training_stats, f)
    mlflow.log_artifact("training_stats.json")

    model_info = mlflow.sklearn.log_model(
        pipeline,
        name="model",
        registered_model_name=MODEL_NAME,
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Metrics: {metrics}")
    print(f"Model registered as: {MODEL_NAME}")
    print(f"Model URI: {model_info.model_uri}")
