# Deployment Basics for Deep Learning Models

Deploying a deep learning model means **making it available for real-world use** outside of training.  
For example, turning your trained model into a **web API, mobile app, or embedded system**.


## 🔑 Key Concepts

1. **Training vs. Inference**
   - **Training**: Model learns patterns from data (compute-heavy, done offline).
   - **Inference**: Using the trained model to make predictions (must be fast & efficient).

2. **Deployment Goals**
   - **Speed**: Predictions should be quick.
   - **Scalability**: Handle many users at once.
   - **Portability**: Run on different platforms (cloud, edge devices, mobile).
   - **Maintainability**: Easy to update when a new model version is available.


## ⚙️ Common Deployment Approaches

### 1. **Local Deployment**
- Save the trained model and load it in a script for offline use.
- Example: A Python script that loads a `.pt` (PyTorch) or `.h5` (Keras) file.

```python
import torch
model = torch.load("model.pt")
model.eval()
prediction = model(torch.randn(1, 3, 224, 224))
```

## Web API Deployment

- Expose the model as a REST API so others can send requests.

- Popular frameworks: Flask, FastAPI, Django.

```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("model.pt")
model.eval()

@app.post("/predict")
def predict(data: list):
    input_tensor = torch.tensor(data)
    output = model(input_tensor).tolist()
    return {"prediction": output}

```

## Cloud Deployment

Use cloud providers for scalability.

- AWS SageMaker

- Google AI Platform

- Azure ML

### Handles large-scale traffic and monitoring automatically.
