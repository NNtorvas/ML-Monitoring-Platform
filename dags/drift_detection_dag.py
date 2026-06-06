"""
Airflow DAG: daily drift detection.
Runs monitoring/drift_report.py and stores the result in an Airflow Variable
so the dashboard can surface it without hitting the DB on every refresh.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run_drift_check(**context):
    import sys

    sys.path.insert(0, "/opt/airflow")

    from monitoring.drift_report import run_drift_check

    drift_detected = run_drift_check()
    Variable.set("drift_detected", str(drift_detected))
    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    return drift_detected


with DAG(
    dag_id="drift_detection",
    default_args=default_args,
    description="Daily data drift check using Evidently",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["monitoring", "drift"],
) as dag:

    drift_task = PythonOperator(
        task_id="run_drift_check",
        python_callable=_run_drift_check,
    )
