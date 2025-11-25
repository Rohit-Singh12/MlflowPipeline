# Registering the model with Pytorch Mlflow, Experiments, logging params, metrics and artifacts, 
# Model registry and versioning,
# signature, tags and dependencies

import mlflow
from common import MLFLOW_URI
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from mlflow.models.signature import infer_signature
mlflow.set_experiment("Pytorch Classification Experiment")
mlflow.pytorch.autolog()

# Load training and test datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

BATCH_SIZE = 64

#Hyperparameters
LR = 0.001
EPOCHS = 10

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
criterion = nn.CrossEntropyLoss()

# Define a simple neural network model
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

# Define a CNN network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

print("Starting training with SimpleNN model...")
# Training the model with SimpleNN   
with mlflow.start_run() as run:
    
    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mlflow.set_tag("Dependencies", "torch, torchvision, sklearn, mlflow")
    mlflow.log_param("model_type", "SimpleNN")
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dataset", "FashionMNIST")
    mlflow.log_param("Training_data_size", len(training_data))
    mlflow.log_param("Test_data_size", len(test_data))
    mlflow.log_param("optimizer", type(optimizer).__name__)
    mlflow.log_param("criterion", type(criterion).__name__)
    
    for epoch in range(EPOCHS):
        model.train()
        # loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        accuracy, precision, recall, f1 = evaluate(model, test_dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], F1 Score: {f1:.4f}")
        
    accuracy, precision, recall, f1 = evaluate(model, test_dataloader)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    #Infer model signature
    sample_inputs, _ = next(iter(test_dataloader))
    signature = infer_signature(sample_inputs.numpy(), model(sample_inputs).detach().numpy())
    
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="pytorch_fashion_mnist_model",
        registered_model_name="simplenn_fashion_mnist",
        signature=signature,
        input_example=sample_inputs.numpy()[:5],
        tags={"purpose": "fashion mnist classification", "model_type": "Simple NN"},
    )
    
    print("Run id: ", run.info.run_id)

print("Starting training with CNN model...")
# Training the model with CNN   
with mlflow.start_run() as run:
    
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mlflow.set_tag("Dependencies", "torch, torchvision, sklearn, mlflow")
    mlflow.log_param("model_type", "CNN")
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dataset", "FashionMNIST")
    mlflow.log_param("Training_data_size", len(training_data))
    mlflow.log_param("Test_data_size", len(test_data))
    mlflow.log_param("optimizer", type(optimizer).__name__)
    mlflow.log_param("criterion", type(criterion).__name__)
    
    for epoch in range(EPOCHS):
        model.train()
        # loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        accuracy, precision, recall, f1 = evaluate(model, test_dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], F1 Score: {f1:.4f}")
        
    accuracy, precision, recall, f1 = evaluate(model, test_dataloader)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    #Infer model signature
    sample_inputs, _ = next(iter(test_dataloader))
    signature = infer_signature(sample_inputs.numpy(), model(sample_inputs).detach().numpy())
    
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="pytorch_fashion_mnist_cnn_model",
        registered_model_name="cnn_fashion_mnist",
        signature=signature,
        input_example=sample_inputs.numpy()[:5],
        tags={"purpose": "fashion mnist classification", "model_type": "CNN"},
    )
    
    print("Run id: ", run.info.run_id)
    


