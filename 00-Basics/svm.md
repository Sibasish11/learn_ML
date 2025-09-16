# Support Vector Machine (SVM)

Support Vector Machine (SVM) is a powerful **supervised learning algorithm** used for both **classification** and **regression** tasks. It is especially effective in high-dimensional spaces.


## 1. Concept
- SVM finds the **optimal hyperplane** that separates different classes in the dataset.
- The hyperplane is chosen to maximize the **margin** between support vectors (data points closest to the boundary).
- Can handle both **linear** and **non-linear** classification using **kernel functions**.


## 2. When to Use
- Binary and multiclass classification
- Text classification (spam detection, sentiment analysis)
- Image classification
- Problems where the number of features is large compared to the number of samples


## 3. Kernel Functions
SVM can transform data into higher dimensions using kernels:
- **Linear Kernel** → works for linearly separable data
- **Polynomial Kernel** → works for polynomial decision boundaries
- **RBF (Radial Basis Function)** → popular for non-linear data
- **Sigmoid Kernel** → behaves like a neural network activation


## 4. Implementation in Python

### Classification Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "age": [25, 30, 45, 35, 40, 50, 23, 33],
    "income": [50000, 60000, 80000, 120000, 70000, 150000, 40000, 90000],
    "buys": [0, 0, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df[["age", "income"]]
y = df["buys"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM with RBF kernel
model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

```

## 5. Visualization (2D Example)

```python

import matplotlib.pyplot as plt
import numpy as np

# Create mesh for plotting decision boundary
x_min, x_max = X["age"].min() - 1, X["age"].max() + 1
y_min, y_max = X["income"].min() - 10000, X["income"].max() + 10000
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 5000))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X["age"], X["income"], c=y, edgecolors="k")
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("SVM Decision Boundary")
plt.show()

```
## 6. Evaluation Metrics

- Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC

- Regression (SVR): MAE, MSE, RMSE, R² score

## 7. Advantages

- Works well in high-dimensional spaces

- Effective for small- to medium-sized datasets

- Flexible with kernel functions

- Robust to overfitting (with proper regularization)

## 8. Limitations

Training can be slow on very large datasets

Choosing the right kernel and parameters can be tricky

Less interpretable compared to decision trees

## 9. Hyperparameters to Tune

C: Regularization parameter (higher C = less regularization, risk of overfitting)

kernel: Linear, RBF, Poly, Sigmoid

gamma: Controls influence of a single training example (higher gamma = tighter fit)

### ✅ Summary
Support Vector Machines are versatile, powerful models for both classification and regression.
With proper kernel selection and parameter tuning, they can achieve high accuracy, especially in high-dimensional spaces.
