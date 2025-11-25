# MLflow Pipeline

A comprehensive demonstration of **MLflow** for end-to-end machine learning lifecycle management. This project showcases best practices for tracking experiments, logging various model types (traditional ML, neural networks, transformers, and LLM agents), managing model registries, and deploying models in production environments.

## Table of Contents

- [Overview](#overview)
- [Setup & Installation](#setup--installation)
- [MLflow Server Setup](#mlflow-server-setup)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [Experiment Tracking & Logging](#experiment-tracking--logging)
- [Model Logging & Registration](#model-logging--registration)
  - [Traditional Models (Scikit-learn)](#traditional-models-scikit-learn)
  - [Neural Networks (PyTorch)](#neural-networks-pytorch)
  - [Transformers](#transformers)
  - [LLM Agents](#llm-agents)
- [Model Registry & Versioning](#model-registry--versioning)
- [Model Loading & Deployment](#model-loading--deployment)
- [Running the Examples](#running-the-examples)

---

## Overview

This project demonstrates:
- ✅ **MLflow Tracking**: Log experiments, metrics, parameters, and artifacts
- ✅ **Model Management**: Log and register models across different frameworks
- ✅ **Model Registry**: Version, stage, and manage models in production
- ✅ **Multiple Model Types**: Scikit-learn, PyTorch (SimpleNN, CNN), Transformers, LLM Agents
- ✅ **Agent Workflows**: Advanced iterative agent patterns with LLMs (Google Gemini)
- ✅ **Model Serving**: Load and deploy models for inference

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Bash (for running server setup script)
- Virtual environment manager (venv)

### Installation Steps

1. **Clone the repository**:
```bash
git clone <repository-url>
cd MlflowPipeline
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirement.txt
```

### Dependencies

The project uses the following key libraries:
- **mlflow**: MLflow tracking and model management
- **scikit-learn**: Traditional machine learning models
- **torch & torchvision**: PyTorch neural networks
- **transformers**: Hugging Face pre-trained models
- **langchain & langchain-google-genai**: LLM agent frameworks
- **pandas, numpy**: Data manipulation

---

## MLflow Server Setup

MLflow uses a backend store to persist experiment data and an artifact store to manage model files.

### Starting the MLflow Server

Run the provided setup script:

```bash
bash start_mlflow_server.sh
```

This script:
- Activates the Python virtual environment
- Creates an SQLite database for backend storage
- Sets up a local artifact directory for model storage
- Starts the MLflow server on `http://127.0.0.1:5000`

### MLflow Server Configuration

The server is configured with:
- **Backend Store**: SQLite database (`sqlite:///mlflow.db`)
- **Artifact Store**: Local filesystem with `file://` scheme
- **Host**: `127.0.0.1`
- **Port**: `5000`

### Accessing MLflow UI

Once the server is running, open your browser and navigate to:
```
http://127.0.0.1:5000
```

You'll see:
- **Experiments**: All tracked experiments
- **Runs**: Individual experiment runs with metrics, parameters, and artifacts
- **Model Registry**: Registered models with versions and stages

---

## Project Structure

```
MlflowPipeline/
├── src/
│   ├── common.py                      # MLflow URI configuration
│   ├── sklearn_demo.py               # Traditional ML with Scikit-learn
│   ├── pytorch_demo.py               # Neural networks with PyTorch
│   ├── transformer_demo.py           # Transformer models (Hugging Face)
│   ├── agent_demo.py                 # LLM agent implementation
│   ├── agent_demo_main.py            # Main agent workflow execution
│   ├── story_agent_Inference.py      # Agent inference wrapper
│   └── data/                         # Dataset directory (FashionMNIST)
├── artifacts/                        # Local artifact storage
├── requirement.txt                   # Python dependencies
├── start_mlflow_server.sh           # MLflow server startup script
└── README.md                        
```

---

## Core Concepts

### 1. **MLflow Tracking URI**

All scripts connect to the MLflow tracking server via:

```python
import mlflow
import os

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
mlflow.set_tracking_uri(MLFLOW_URI)
```

This is centralized in `common.py` and imported by all demo scripts.

### 2. **Experiments**

Experiments group related runs under a common namespace:

```python
mlflow.set_experiment('sklearn iris classifier v2')
```

### 3. **Runs**

A run represents a single execution of your training code:

```python
with mlflow.start_run() as run:
    # Training code here
    run_id = run.info.run_id
```

---

## Experiment Tracking & Logging

### Logging Parameters

Parameters are hyperparameters or configuration values:

```python
mlflow.log_param('model_type', 'Random Forest')
mlflow.log_param('n_estimators', 10)
mlflow.log_param('learning_rate', 0.001)
```

### Logging Metrics

Metrics are performance measurements:

```python
mlflow.log_metric('accuracy', 0.95)
mlflow.log_metric('loss', 0.05)
mlflow.log_metric('f1_score', 0.93)
```

### Logging Artifacts

Artifacts are output files (plots, models, data):

```python
mlflow.log_dict(result_dict, artifact_file="results.json")
```

### Logging Tags

Tags add metadata to runs:

```python
mlflow.set_tag('Dependencies', 'torch, torchvision, sklearn')
mlflow.set_tag('purpose', 'fashion mnist classification')
```

---

## Model Logging & Registration

Each demo script logs and registers models to MLflow's Model Registry with proper signatures, input examples, and metadata.

### Traditional Models (Scikit-learn)

**File**: `src/sklearn_demo.py`

Demonstrates logging a Random Forest classifier on the Iris dataset:

```python
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

mlflow.set_experiment('sklearn iris classifier v2')

with mlflow.start_run() as run:
    # Train model
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param('n_estimators', 10)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric('accuracy', accuracy)
    
    # Infer signature for model validation
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Register model
    mlflow.sklearn.log_model(
        model,
        artifact_path='rf_model',
        registered_model_name='iris_rf',
        signature=signature,
        input_example=X_train[:5],
        tags={'purpose': 'iris classification'}
    )
```

**Key Features**:
- Automatic signature inference
- Input examples for documentation
- Versioning in Model Registry
- Tags for organization

---

### Neural Networks (PyTorch)

**File**: `src/pytorch_demo.py`

Demonstrates training and logging two neural network architectures on FashionMNIST:

#### SimpleNN (Multi-layer Perceptron)

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

#### CNN (Convolutional Neural Network)

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # CNN forward pass...
```

**Logging PyTorch Models**:

```python
mlflow.set_experiment("Pytorch Classification Experiment")

with mlflow.start_run() as run:
    model = SimpleNN()
    
    # Log hyperparameters
    mlflow.log_param('learning_rate', 0.001)
    mlflow.log_param('epochs', 10)
    mlflow.log_param('batch_size', 64)
    
    # Training loop...
    
    # Log metrics
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1_score', f1)
    
    # Infer signature from actual predictions
    signature = infer_signature(
        sample_inputs.numpy(),
        model(sample_inputs).detach().numpy()
    )
    
    # Register model
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path='pytorch_fashion_mnist_model',
        registered_model_name='simplenn_fashion_mnist',
        signature=signature,
        input_example=sample_inputs.numpy()[:5],
        tags={'purpose': 'fashion mnist classification'}
    )
```

**Key Features**:
- Autolog support with `mlflow.pytorch.autolog()`
- Multi-metric tracking during training
- Proper tensor-to-numpy conversion for signatures
- Model-specific logging methods

---

### Transformers

**File**: `src/transformer_demo.py`

Demonstrates logging a pre-trained transformer model for text classification:

```python
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

mlflow.set_experiment("Transformer Text Classification")

with mlflow.start_run() as run:
    # Load pre-trained model and tokenizer
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    
    # Log parameters
    mlflow.log_param('model', model_name)
    mlflow.log_param('task', 'sentiment-analysis')
    
    # Make predictions
    texts = ["MLflow is great!", "This is bad."]
    predictions = classifier(texts)
    
    # Log metrics
    mlflow.log_metric('accuracy', accuracy)
    
    # Generate signature for transformers
    signature = infer_signature(
        texts,
        mlflow.transformers.generate_signature_output(classifier, texts),
        params={'model': model_name}
    )
    
    # Register transformer model
    mlflow.transformers.log_model(
        transformers_model=classifier,
        name='text-classifier',
        signature=signature,
        input_example=texts[:2],
        registered_model_name='TransformerTextClassifier',
        task='sentiment-analysis'
    )
```

**Key Features**:
- Pre-trained model support
- Pipeline-based interface
- Automatic dependency tracking
- Task-specific configurations

---

### LLM Agents

**Files**: `src/agent_demo.py`, `src/agent_demo_main.py`, `src/story_agent_Inference.py`

This is a sophisticated example of an iterative agent workflow using Google's Gemini model:

#### Architecture

The agent demonstrates:
1. **Agent Workflow**: Iterative story writing with self-evaluation
2. **Chat History Management**: Maintaining conversation state
3. **Tool Calling**: Agent decides to evaluate or improve stories
4. **MLflow Tracing**: Capture all agent operations

#### Agent Workflow

```python
@mlflow.trace(span_type=SpanType.AGENT)
def agent_workflow(topic: str) -> Dict[str, Any]:
    '''
    Orchestrates the agent workflow:
    1. Writer agent writes a story
    2. Evaluator agent provides feedback
    3. Loop until score >= 6 or max iterations
    '''
    
    with mlflow.start_run(run_name="Agent Story Writer Run"):
        append_to_chat_history("user", f"Write a story about: {topic}")
        
        for iteration in range(5):  # Max 5 iterations
            # 1. Generate story
            story = make_gemini_chat(writer_prompt, chat_history)
            append_to_chat_history("model", story)
            
            # 2. Evaluate story
            evaluation = make_gemini_chat(evaluation_prompt, chat_history)
            parsed_eval = json.loads(evaluation)
            
            # 3. Check quality
            if parsed_eval['score'] >= 6:
                return {'message': 'Success', 'story': story}
            
            # 4. Continue improvement loop
            append_to_chat_history("user", f"Improve based on: {parsed_eval['comments']}")
```

#### LLM Call Tracing

```python
@mlflow.trace(span_type=SpanType.LLM)
def make_gemini_call(query: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=query
    )
    return response.text
```

#### Model Serialization

```python
class StoryAgentModel(mlflow.pyfunc.PythonModel):
    """Custom model wrapper for the story agent."""
    
    def predict(self, context, model_input: pd.DataFrame):
        topic = model_input["topic"].iloc[0]
        result = agent_workflow(str(topic))
        return pd.Series([result.get("story", "")])

# Register the agent
mlflow.pyfunc.log_model(
    artifact_path='story_agent_model',
    python_model=StoryAgentModel(),
    registered_model_name='story-agent',
    input_example=pd.DataFrame({"topic": ["Sample topic"]}),
    pip_requirements=[
        'mlflow', 'google-genai', 'python-dotenv', 'pandas'
    ]
)
```

**Key Features**:
- **MLflow Tracing**: Captures all spans and traces
- **Agent Patterns**: Agentic loops with state management
- **LLM Integration**: Direct API calls to LLMs
- **Custom Models**: `pyfunc` wrapper for arbitrary Python code
- **Dependency Management**: Explicit pip requirements

---

## Model Registry & Versioning

### Registering Models

Models are registered with `registered_model_name` parameter:

```python
mlflow.sklearn.log_model(
    model,
    artifact_path='rf_model',
    registered_model_name='iris_rf'
)
```

### Serving Models
Registered model can be served using mlflow.pyfunc.load_model
```
import mlflow
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.pyfunc.load_model('model:/<name>/<version>')
```

### Model Stages

The Model Registry supports lifecycle stages:

- **None**: Development/unstarred version
- **Staging**: Pre-production testing
- **Production**: Active serving models
- **Archived**: Retired models

### Retrieving Model Versions

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all versions
model_versions = client.get_latest_versions(
    name='iris_rf',
    stages=['None', 'Staging', 'Production', 'Archived']
)

# Get specific version
version = model_versions[0].version
```

### Managing Model Versions

Update version descriptions and transition stages:

```python
# Update version description
client.update_model_version(
    name='iris_rf',
    version=version,
    description='Updated model version description'
)

# Add tags to registered model
client.set_registered_model_tag(
    name='iris_rf',
    key='owner',
    value='data_science_team'
)

# Transition stage (requires aliases in newer MLflow versions)
client.transition_model_version_stage(
    name='iris_rf',
    version=version,
    stage='Production'
)
```

---

## Model Loading & Deployment

### Loading Models from Registry

```python
import mlflow

# Load specific model version
model = mlflow.sklearn.load_model(f"models:/iris_rf/1")

# Or use alias
model = mlflow.sklearn.load_model("models:/iris_rf/Production")

# Make predictions
predictions = model.predict(X_test)
```

### Framework-Specific Loading

**Scikit-learn**:
```python
model = mlflow.sklearn.load_model(model_uri)
```

**PyTorch**:
```python
model = mlflow.pytorch.load_model(model_uri)
model.eval()
predictions = model(inputs)
```

**Transformers**:
```python
classifier = mlflow.transformers.load_model(model_uri)
results = classifier(texts)
```

**Custom Models (pyfunc)**:
```python
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(input_data)
```

### Model Serving with MLflow

Serve models using the MLflow Models CLI:

```bash
# Serve specific version
mlflow models serve -m "models:/iris_rf/1" -p 1234 --no-conda

# Serve production stage
mlflow models serve -m "models:/iris_rf/Production" -p 5001
```

Make predictions via HTTP:

```bash
curl -X POST http://127.0.0.1:1234/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["col1", "col2"], "data": [[1, 2]]}}'
```

---

## Running the Examples

### 1. Start MLflow Server

```bash
bash start_mlflow_server.sh
```

### 2. Run Individual Demos

**Scikit-learn Demo** (Random Forest on Iris):
```bash
cd src
python sklearn_demo.py
```

Expected output:
- Run ID printed to console
- Model registered as `iris_rf` in Model Registry
- Metrics logged: accuracy
- Parameters logged: model type, n_estimators

**PyTorch Demo** (SimpleNN & CNN on FashionMNIST):
```bash
python pytorch_demo.py
```

Expected output:
- Two models registered: `simplenn_fashion_mnist`, `cnn_fashion_mnist`
- Training progress printed (epochs and F1 scores)
- Metrics: accuracy, precision, recall, f1_score

**Transformer Demo** (Text Classification):
```bash
python transformer_demo.py
```

Expected output:
- Model registered as `TransformerTextClassifier`
- Predictions printed
- Model loaded and inference demonstrated

**Agent Demo** (Story Writing with Gemini):

First, set up environment variables:
```bash
# Create .env file in project root
GOOGLE_API_KEY=your_google_api_key_here
```

Then run:
```bash
python agent_demo_main.py
```

Expected output:
- Generated story about specified topic
- Story evaluation and iterative improvements
- Agent artifacts logged (workflow result, final story)
- Model registered as `story-agent`

### 3. Explore MLflow UI

Navigate to `http://127.0.0.1:5000` to:
- View all experiments and runs
- Compare metrics across runs
- Inspect model parameters and artifacts
- Check Model Registry for all registered models and versions

---

## Best Practices

### 1. Always Use Signatures

```python
signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, ..., signature=signature)
```

Signatures document input/output schemas for model consumers.

### 2. Include Input Examples

```python
mlflow.sklearn.log_model(
    model,
    ...,
    input_example=X_train[:5]
)
```

Helps with model serving and documentation.

### 3. Use Tags for Organization

```python
mlflow.set_tag('team', 'data_science')
mlflow.set_tag('model_type', 'production')
mlflow.set_tag('framework', 'sklearn')
```

### 4. Log All Dependencies

For custom models, explicitly list requirements:

```python
pip_requirements=['mlflow', 'sklearn', 'pandas', 'numpy']
```

### 5. Use Context Managers for Runs

```python
with mlflow.start_run() as run:
    # Your code here
    pass
# Run automatically ends
```

### 6. Leverage MLflow Autolog

Many frameworks support automatic logging:

```python
mlflow.pytorch.autolog()
mlflow.sklearn.autolog()
mlflow.transformers.autolog()
```

---

## Troubleshooting

### MLflow Server Connection Issues

**Problem**: `ConnectionError: Failed to connect to MLflow server`

**Solution**:
```bash
# Verify server is running
bash start_mlflow_server.sh

# Check tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

### Model Loading Errors

**Problem**: `MlflowException: Model not found`

**Solution**:
```python
# Use correct model URI format
model_uri = "models:/model_name/version_number"  # Correct
# or
model_uri = "runs:/run_id/artifact_path"  # Correct
```

### Artifact Storage Issues

**Problem**: Artifacts not saving to expected location

**Solution**: Ensure `default-artifact-root` uses `file://` scheme in server startup

---

## Resources

- [MLflow Documentation](https://mlflow.org/docs/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Model Signatures](https://mlflow.org/docs/latest/models.html#model-signature)
- [Transformers Integration](https://mlflow.org/docs/latest/transformers-integration.html)
- [Google Gemini API](https://ai.google.dev/)
