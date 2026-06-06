CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_features JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    confidence FLOAT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp);
