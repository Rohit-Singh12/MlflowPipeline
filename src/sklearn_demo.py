# Basic Tracking - experiments, runs, params, metrics, artifacts, model registry
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from common import MLFLOW_URI

# set experiment name for tracking
mlflow.set_experiment('sklearn iris classifier v2')
with mlflow.start_run() as run:
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
    n_estimators = 10
    mlflow.log_param('model_type', 'Random Forest')
    mlflow.log_param('n_estimators', n_estimators)

    # Model training
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Accuracy
    acc = model.score(X_test, y_test)
    mlflow.log_metric('accuracy', acc)

    #Infer model signature
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model, 
        artifact_path='rf_model', 
        registered_model_name="iris_rf",
        signature=signature,
        input_example=X_train[:5],
        tags={"purpose": "iris classification"})
    print("Run id: ", run.info.run_id)

client = MlflowClient()
model_version_infos = client.get_latest_versions(name="iris_rf", stages=["None", "Staging", "Production", "Archived"])
version = model_version_infos[0].version
client.update_model_version(
    name="iris_rf",
    version=version,
    description="Updated model version description"
)

#add tags
client.set_registered_model_tag(
    name="iris_rf",
    key="owner",
    value="data_science_team"
)

#### Serve using
### mlflow models serve -m "models:/iris_rf/version" -p 1234 --no-conda
