set -e

CURRENT_DIR=$(pwd | sed 's#/c/#C:/#')
source "${CURRENT_DIR}/venv/Scripts/activate"

export MLFLOW_PORT=5000
export MLFLOW_ARTIFACTS_DIR="${CURRENT_DIR}/artifacts"

mkdir -p "${MLFLOW_ARTIFACTS_DIR}"

echo "Using MLflow: $(which mlflow)"

echo "sqlite:///${CURRENT_DIR}/mlflow.db"

# CRITICAL FIX: The --default-artifact-root MUST use the file:// scheme
# to ensure the MLflow client knows how to handle the path.
mlflow server \
    --backend-store-uri "sqlite:///${CURRENT_DIR}/mlflow.db" \
    --default-artifact-root "file:///${MLFLOW_ARTIFACTS_DIR}" \
    --host 127.0.0.1 \
    --port ${MLFLOW_PORT}