import mlflow
import numpy as np
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from common import MLFLOW_URI

mlflow.set_experiment("Transformer Text Classification")
# mlflow.transformers.autolog() Disabled for manual logging

# Configurations
MODEL='distilbert-base-uncased'
TASK='sentiment-analysis'

mlflow.log_param("model", MODEL)
mlflow.log_param("task", TASK)

# Load tokenizer and pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
text_classifier = pipeline(TASK, model=model, tokenizer=tokenizer)

# Sample data
texts = ["MLflow is great for tracking.", "Hugging Face makes NLP easy."]
labels = [1, 1]  # 1: Positive sentiment

if mlflow.active_run():
    print("WARNING: Ending previously active MLflow run.")
    mlflow.end_run()

with mlflow.start_run() as run:
    # Training step (mocked for demonstration)
    # In practice, you would fine-tune the model here

    # Predictions
    predictions = text_classifier(texts)
    predicted_labels = [1 if pred['score'] > 0.5 else 0 for pred in predictions]
    # Evaluation
    accuracy = accuracy_score(labels, predicted_labels)
    model_config = {"model": MODEL, "task": TASK}
    mlflow.log_metric("accuracy", accuracy)
    signature = infer_signature(
        texts,
        mlflow.transformers.generate_signature_output(text_classifier, texts),
        params=model_config
    )
    
    # Log model
    model_info = mlflow.transformers.log_model(
        transformers_model=text_classifier,
        name='text-classifier',
        signature=signature,
        input_example=texts[:2],
        model_config=model_config,
        registered_model_name='TransformerTextClassifier',
        task=TASK
    )
    model_uri = model_info.model_uri
    print(f"Model logged at: {model_uri}")
    
# LOAD AND PREDICT
print("MODEL URI ", model_uri)
loaded_model = mlflow.transformers.load_model(model_uri)
test_data = ["This is good movies", "This is really bad"]
predictions = loaded_model(test_data)
print(predictions)
