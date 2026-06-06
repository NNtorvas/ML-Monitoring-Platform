"""
Run Evidently drift detection comparing last 7 days of predictions
against the training distribution fetched from MLflow artifacts.
Returns exit code 1 if drift is detected (used by CI to trigger retraining).
"""

import os
import sys
import json
import tempfile
import logging
from datetime import datetime, timezone, timedelta

import mlflow
import pandas as pd
import psycopg2
import psycopg2.extras
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "breast-cancer-model")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "mluser"),
    "password": os.getenv("POSTGRES_PASSWORD", "mlpassword"),
    "dbname": os.getenv("POSTGRES_DB", "mlmonitoring"),
}

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


def fetch_training_reference() -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME)
    if not versions:
        raise RuntimeError(f"No registered versions found for model '{MODEL_NAME}'")

    run_id = versions[0].run_id
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="training_stats.json",
            dst_path=tmpdir,
        )
        with open(local_path) as f:
            stats = json.load(f)

    # Reconstruct a synthetic reference DataFrame from mean/std
    rng = __import__("numpy").random.default_rng(42)
    n = 500
    data = {
        name: rng.normal(loc=mean, scale=max(std, 1e-9), size=n)
        for name, mean, std in zip(FEATURE_NAMES, stats["feature_means"], stats["feature_stds"])
    }
    return pd.DataFrame(data)


def fetch_recent_predictions(days: int = 7) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    conn = psycopg2.connect(**DB_CONFIG)
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT input_features FROM predictions WHERE timestamp >= %s",
                (cutoff,),
            )
            rows = cur.fetchall()
    conn.close()

    if not rows:
        logger.warning("No predictions found in the last %d days", days)
        return pd.DataFrame(columns=FEATURE_NAMES)

    records = [json.loads(row["input_features"]) for row in rows]
    return pd.DataFrame(records, columns=FEATURE_NAMES)


def run_drift_check() -> bool:
    reference = fetch_training_reference()
    current = fetch_recent_predictions()

    if current.empty:
        logger.info("No current data — skipping drift check")
        return False

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report_path = "/tmp/drift_report.html"
    report.save_html(report_path)
    logger.info(f"Drift report saved to {report_path}")

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    logger.info(f"Drift detected: {drift_detected}")
    return drift_detected


if __name__ == "__main__":
    drift = run_drift_check()
    print(f"DRIFT_DETECTED={drift}")
    sys.exit(1 if drift else 0)
