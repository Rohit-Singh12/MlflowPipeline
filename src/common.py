import mlflow, os

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
mlflow.set_tracking_uri(MLFLOW_URI)
