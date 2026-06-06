import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.express as px
import psycopg2
import psycopg2.extras
import streamlit as st

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "mluser"),
    "password": os.getenv("POSTGRES_PASSWORD", "mlpassword"),
    "dbname": os.getenv("POSTGRES_DB", "mlmonitoring"),
}

st.set_page_config(page_title="ML Monitor", layout="wide")
st.title("ML Monitoring Dashboard")
st.caption(f"Last refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


@st.cache_data(ttl=60)
def load_predictions(days: int = 30) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT timestamp, prediction, confidence
                    FROM predictions
                    WHERE timestamp >= %s
                    ORDER BY timestamp
                    """,
                    (cutoff,),
                )
                rows = cur.fetchall()
        conn.close()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame(columns=["timestamp", "prediction", "confidence"])


df = load_predictions()

today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    today_df = df[df["timestamp"] >= today_start]
else:
    today_df = df

# Drift flag — set by Airflow DAG via a shared file or env var fallback
drift_flag_path = "/tmp/drift_detected.txt"
drift_detected = False
try:
    with open(drift_flag_path) as f:
        drift_detected = f.read().strip().lower() == "true"
except FileNotFoundError:
    drift_detected = os.getenv("DRIFT_DETECTED", "false").lower() == "true"

# KPI row
col1, col2, col3 = st.columns(3)
col1.metric("Predictions Today", len(today_df))
col2.metric(
    "Avg Confidence Today",
    f"{today_df['confidence'].mean():.2%}" if not today_df.empty else "N/A",
)
drift_label = "YES" if drift_detected else "NO"
drift_delta = "Retraining may be needed" if drift_detected else "Model looks healthy"
col3.metric("Drift Detected", drift_label, drift_delta)

st.divider()

# Volume chart
if not df.empty:
    st.subheader("Prediction Volume (last 30 days)")
    df["date"] = df["timestamp"].dt.date
    volume = df.groupby("date").size().reset_index(name="count")
    fig = px.bar(volume, x="date", y="count", labels={"date": "Date", "count": "Predictions"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confidence Over Time")
    fig2 = px.scatter(
        df,
        x="timestamp",
        y="confidence",
        color=df["prediction"].map({0: "malignant", 1: "benign"}),
        labels={"confidence": "Confidence", "timestamp": "Time", "color": "Class"},
        opacity=0.6,
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No predictions logged yet. Hit POST /predict on the API to generate data.")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()
