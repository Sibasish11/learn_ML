# Logistic Regression

Logistic Regression is a **classification algorithm** used to predict the probability that a given input belongs to a particular class. Unlike Linear Regression, it is used for **categorical outcomes** (e.g., yes/no, spam/ham).


## 1. Concept
- Predicts probabilities using the **sigmoid function**:
  
\[
P(y=1|x) = \frac{1}{1 + e^{-(β_0 + β_1x_1 + … + β_nx_n)}}
\]

- Output is between **0 and 1**.
- Threshold (e.g., 0.5) is applied to classify into classes.


## 2. When to Use
- Binary classification (spam vs not spam, disease vs no disease).
- Multi-class classification (using extensions like One-vs-Rest or Softmax).


## 3. Implementation in Python

### Example: Predicting if a student passes based on study hours
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "pass":  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# Features and target
X = df[["hours"]]
y = df["pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

```

## 4. Visualization of the Sigmoid Function

```python

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.grid()
plt.show()

```

## 5. Evaluation Metrics
For classification problems, we evaluate using:

- Accuracy

- Precision

- Recall

- F1-score

- ROC-AUC curve

## 6. Advantages

Easy to implement and interpret

Outputs probabilities, not just classes

Works well for linearly separable classes

## 7. Limitations

- Assumes linear decision boundary

- Not effective for non-linear relationships

- Sensitive to outliers and multicollinearity

✅ Summary
Logistic Regression is a powerful baseline algorithm for classification. It is simple, interpretable, and widely used in real-world problems like spam detection, medical diagnosis, and fraud detection.
