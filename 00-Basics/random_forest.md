# Random Forest

Random Forest is an **ensemble learning algorithm** that builds multiple decision trees and combines their outputs for more accurate and stable predictions. It is widely used for both **classification and regression** tasks.

## 1. Concept
- Builds many decision trees (a "forest").
- Each tree is trained on a **random subset** of the data and features (Bagging + Feature Randomness).
- Final prediction:
  - **Classification** → Majority vote of trees
  - **Regression** → Average of tree predictions

## 2. When to Use
- Classification problems (fraud detection, customer churn, disease classification).
- Regression problems (price prediction, demand forecasting).
- When accuracy is more important than interpretability.

## 3. Implementation in Python

### Classification Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

```

## 4. Feature Importance:

- Random Forest provides a measure of feature importance.

```python

import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest")
plt.show()

```

## 5. Evaluation Metrics

- Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC

- Regression: MAE, MSE, RMSE, R² score

## 6. Advantages

- Handles large datasets well

- Works with both categorical and numerical data

Less prone to overfitting than a single decision tree

Provides feature importance

## 7. Limitations
More complex and less interpretable than a single decision tree

Computationally intensive with many trees

Not ideal for very high-dimensional sparse data (e.g., text data)

## 8. Hyperparameters to Tune

**n_estimators** : Number of trees.

**max_depth** : Maximum depth of trees.

**min_samples_split** : Minimum samples required to split a node.

**max_features** : Number of features to consider for the best split.

✅ Summary
Random Forest is a robust and versatile algorithm that reduces overfitting and improves accuracy by combining multiple decision trees. It is one of the most widely used algorithms in practical machine learning.
