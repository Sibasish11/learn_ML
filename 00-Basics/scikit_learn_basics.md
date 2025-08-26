# Introduction to Scikit-learn

[Scikit-learn](https://scikit-learn.org/) is one of the most popular Python libraries for **machine learning**.  
It provides simple and efficient tools for data analysis, preprocessing, and predictive modeling.

## 1. Installing Scikit-learn:

```bash
pip install scikit-learn
```

## 2. Importing Scikit-learn:

```python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

```

## 3. Dataset Example:
- Scikit-learn includes several built-in datasets.

```python

from sklearn.datasets import load_iris

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

```

## 4. Train-Test Split:

```python

X = iris.data   # features
y = iris.target # labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


```

## 5. Simple Linear Regression Example

- from sklearn.linear_model import LinearRegression

# Example dataset
```python

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict([[6]])

print("Prediction for x=6:", y_pred)

```

## 6. Classification Example (Logistic Regression on Iris)

```python

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

```

## 7. Preprocessing Example (Scaling Data)

```python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled Data:\n", X_scaled[:5])

```

## 8. Model Evaluation

```python

from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

```

### ✅ Summary

- With Scikit-learn, you can:

- Load datasets (load_iris, load_digits, etc.)

- Split data (train_test_split)

- Build models (regression, classification, clustering, etc.)

- Preprocess data (scaling, encoding, normalization)

- Evaluate models (accuracy, confusion matrix, MSE)
