# Linear Regression

**Linear Regression** is one of the most fundamental and widely used algorithms in Machine Learning. It is used to model the relationship between a dependent variable (target) and one or more independent variables (features).

## 1. Concept
- **Simple Linear Regression**: Models the relationship between one feature (X) and a target (y).
- **Multiple Linear Regression**: Models the relationship between multiple features (X1, X2, …, Xn) and a target (y).
- Equation:

\[
y = β_0 + β_1x_1 + β_2x_2 + … + β_nx_n + ε
\]

Where:
- \( β_0 \): Intercept
- \( β_i \): Coefficients
- \( ε \): Error term


## 2. When to Use
- Predicting **continuous values** (house price, salary, temperature, etc.)
- When there is a **linear relationship** between features and target


## 3. Implementation in Python

### Example: Predicting House Prices
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "area": [650, 800, 900, 1200, 1500],
    "price": [70000, 85000, 95000, 130000, 160000]
}
df = pd.DataFrame(data)

# Split into features and target
X = df[["area"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted prices:", y_pred)

# Visualization
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Linear Regression Example")
plt.show()

```

## 4. Evaluation Metrics

#### To check the performance of Linear Regression, we use:

- Mean Absolute Error (MAE)

- Mean Squared Error (MSE)

- Root Mean Squared Error (RMSE)

- R² Score

```python

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

```

## 5. Advantages

Easy to understand and implement

Computationally efficient

Works well for linearly separable data

## 6. Limitations
- Assumes linear relationship between variables

- Sensitive to outliers

- Cannot capture complex non-linear patterns

## ✅ Summary
Linear Regression is the go-to algorithm for predicting continuous values when the relationship between features and target is approximately linear. 
