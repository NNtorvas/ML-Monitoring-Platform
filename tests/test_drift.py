import json
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from monitoring.drift_report import (
    fetch_training_reference,
    fetch_recent_predictions,
    run_drift_check,
    FEATURE_NAMES,
)

TRAINING_STATS = {
    "feature_names": FEATURE_NAMES,
    "feature_means": [float(i) for i in range(30)],
    "feature_stds": [1.0] * 30,
}


def _download_side_effect(run_id, artifact_path, dst_path):
    path = os.path.join(dst_path, "training_stats.json")
    with open(path, "w") as f:
        json.dump(TRAINING_STATS, f)
    return path


@pytest.fixture
def mlflow_client():
    version = MagicMock()
    version.run_id = "run-abc123"
    client = MagicMock()
    client.get_latest_versions.return_value = [version]
    return client


def test_fetch_reference_shape(mlflow_client):
    with patch("monitoring.drift_report.mlflow.set_tracking_uri"), patch(
        "monitoring.drift_report.mlflow.MlflowClient", return_value=mlflow_client
    ), patch(
        "monitoring.drift_report.mlflow.artifacts.download_artifacts", side_effect=_download_side_effect
    ):
        df = fetch_training_reference()

    assert df.shape == (500, 30)
    assert list(df.columns) == FEATURE_NAMES


def test_fetch_reference_no_versions(mlflow_client):
    mlflow_client.get_latest_versions.return_value = []
    with patch("monitoring.drift_report.mlflow.set_tracking_uri"), patch(
        "monitoring.drift_report.mlflow.MlflowClient", return_value=mlflow_client
    ):
        with pytest.raises(RuntimeError, match="No registered versions"):
            fetch_training_reference()


def test_fetch_predictions_unpacks_jsonb():
    sample = [float(i) for i in range(30)]
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = [{"input_features": json.dumps(sample)}]

    with patch("monitoring.drift_report.psycopg2.connect", return_value=mock_conn):
        df = fetch_recent_predictions()

    assert df.shape == (1, 30)
    assert list(df.columns) == FEATURE_NAMES


def test_fetch_predictions_empty():
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = []

    with patch("monitoring.drift_report.psycopg2.connect", return_value=mock_conn):
        df = fetch_recent_predictions()

    assert df.empty
    assert list(df.columns) == FEATURE_NAMES


def test_run_drift_check_no_data():
    with patch("monitoring.drift_report.fetch_training_reference"), patch(
        "monitoring.drift_report.fetch_recent_predictions", return_value=pd.DataFrame(columns=FEATURE_NAMES)
    ):
        result = run_drift_check()
    assert result is False


def test_run_drift_check_no_drift():
    reference = pd.DataFrame(np.ones((500, 30)), columns=FEATURE_NAMES)
    current = pd.DataFrame(np.ones((100, 30)), columns=FEATURE_NAMES)
    mock_report = MagicMock()
    mock_report.as_dict.return_value = {"metrics": [{"result": {"dataset_drift": False}}]}

    with patch("monitoring.drift_report.fetch_training_reference", return_value=reference), patch(
        "monitoring.drift_report.fetch_recent_predictions", return_value=current
    ), patch("monitoring.drift_report.Report", return_value=mock_report):
        result = run_drift_check()

    assert result is False


def test_run_drift_check_drift_detected():
    reference = pd.DataFrame(np.ones((500, 30)), columns=FEATURE_NAMES)
    current = pd.DataFrame(np.ones((100, 30)) * 100, columns=FEATURE_NAMES)
    mock_report = MagicMock()
    mock_report.as_dict.return_value = {"metrics": [{"result": {"dataset_drift": True}}]}

    with patch("monitoring.drift_report.fetch_training_reference", return_value=reference), patch(
        "monitoring.drift_report.fetch_recent_predictions", return_value=current
    ), patch("monitoring.drift_report.Report", return_value=mock_report):
        result = run_drift_check()

    assert result is True
