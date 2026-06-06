import os
import json
import logging
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="ML Monitoring API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        model_uri = f"models:/{MODEL_NAME}/latest"
        logger.info(f"Loading model from {model_uri}")
        _model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully")
    return _model


def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    label: str
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = get_model()

    if len(request.features) != 30:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 30 features, got {len(request.features)}",
        )

    features = np.array(request.features).reshape(1, -1)

    try:
        prediction = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        confidence = float(proba[prediction])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    timestamp = datetime.now(timezone.utc).isoformat()
    label = "malignant" if prediction == 0 else "benign"

    try:
        conn = get_db_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions (timestamp, input_features, prediction, confidence)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (timestamp, json.dumps(request.features), prediction, confidence),
                )
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log prediction to DB: {e}")

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        label=label,
        timestamp=timestamp,
    )
