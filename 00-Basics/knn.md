# K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a **simple, non-parametric, supervised learning algorithm** used for **classification** and **regression**. It makes predictions based on the majority (classification) or average (regression) of the nearest data points.


## 1. Concept
- KNN assumes that **similar data points exist close to each other**.
- It calculates the **distance** between a query point and all data points.
- Selects the **K nearest neighbors** and makes a decision:
  - **Classification** → majority vote of neighbors
  - **Regression** → average of neighbors’ values


## 2. Distance Metrics
Common distance functions used in KNN:
- **Euclidean Distance**:  
  \[
  d(p, q) = \sqrt{\sum (p_i - q_i)^2}
  \]
- Manhattan Distance
- Minkowski Distance
- Cosine Similarity


## 3. Choosing K
- Small **K** → sensitive to noise (overfitting).
- Large **K** → smoother decision boundary but may underfit.
- Use **cross-validation** to choose optimal K.

## 4. Implementation in Python

### Classification Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "score": [10, 20, 30, 40, 50, 60, 70, 80],
    "passed": [0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["study_hours", "score"]]
y = df["passed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

```

#### Regression Example

```python

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# KNN Regression
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)

y_pred_reg = knn_reg.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred_reg))
5. Visualization (2D Classification)
python
Copy code
import matplotlib.pyplot as plt
import numpy as np

# Create mesh for decision boundary
x_min, x_max = X["study_hours"].min() - 1, X["study_hours"].max() + 1
y_min, y_max = X["score"].min() - 10, X["score"].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X["study_hours"], X["score"], c=y, edgecolors="k")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("KNN Decision Boundary")
plt.show()

```

## 6. Advantages

Simple and intuitive

No training phase (lazy learner)

Works well with small datasets

Non-parametric (no assumptions about data distribution)

## 7. Limitations

Computationally expensive for large datasets

Sensitive to irrelevant features and scaling

Needs proper choice of K

Not suitable for high-dimensional data (curse of dimensionality)

## 8. Improving KNN

- Feature scaling (Standardization/Normalization)

- Use weighted KNN (closer neighbors have higher influence)

- Dimensionality reduction (PCA, t-SNE)

## ✅ Summary

KNN is a simple yet effective algorithm for classification and regression.
It relies on the idea that similar things exist in close proximity.
Despite its simplicity, KNN can perform well for small to medium-sized problems when data is properly scaled.

yaml
Copy code
