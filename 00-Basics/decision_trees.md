# Decision Trees

Decision Trees are **supervised learning algorithms** used for both classification and regression tasks. They split the dataset into branches based on feature values, forming a tree-like structure where each internal node represents a decision, and each leaf node represents an outcome.


## 1. Concept
- At each step, the algorithm chooses the **best feature and threshold** to split the data.
- Splitting criteria:
  - **Gini Impurity**
  - **Entropy / Information Gain**
  - **Mean Squared Error** (for regression)

Example (Classification):

```python

Is Age < 30?
├── Yes → Student?
│ ├── Yes → Buys
│ └── No → Doesn’t Buy
└── No → Income > 50k?
├── Yes → Buys
└── No → Doesn’t Buy

```

## 2. When to Use
- Classification tasks (spam detection, loan approval, disease prediction).
- Regression tasks (predicting house prices, sales forecasting).
- When **interpretability** is important.



## 3. Implementation in Python

### Classification Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

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

# Train model
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Plot tree
plt.figure(figsize=(8,6))
plot_tree(model, feature_names=["age", "income"], class_names=["No", "Yes"], filled=True)
plt.show()

# Predictions
print("Predictions:", model.predict(X_test))

```

## 4. Evaluation Metrics:
- Classification: Accuracy, Precision, Recall, F1-score

- Regression: MAE, MSE, RMSE, R² score

## 5. Advantages

- Easy to understand and interpret

- Works with both numerical and categorical data

- No need for feature scaling

- Captures non-linear relationships

## 6. Limitations

- Prone to overfitting

- Small changes in data can lead to different splits (unstable)

- Bias towards features with more categories

## 7. Avoiding Overfitting

- Limit tree depth (max_depth)

- Minimum samples required to split (min_samples_split)

- Prune unnecessary branches

- Use ensemble methods (Random Forest, Gradient Boosting)

### ✅ Summary
Decision Trees are simple yet powerful models for classification and regression. They provide excellent interpretability but require careful tuning to avoid overfitting.
