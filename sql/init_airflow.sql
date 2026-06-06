-- Create airflow database if it doesn't exist
-- Postgres init scripts run as the POSTGRES_USER
SELECT 'CREATE DATABASE airflow OWNER mluser'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
