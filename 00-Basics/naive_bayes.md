# Naive Bayes Classifier

Naive Bayes is a **probabilistic machine learning algorithm** based on **Bayes’ Theorem** with the assumption of independence among features. It is widely used for **classification tasks**, especially in text processing.


## 1. Concept

**Bayes’ Theorem:**

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

- **Naive assumption**: Features are independent of each other given the class.
- Despite this assumption, it performs well in many practical applications.


## 2. Types of Naive Bayes
- **Gaussian Naive Bayes** → Continuous data (assumes normal distribution).
- **Multinomial Naive Bayes** → Discrete counts (e.g., word frequencies in text).
- **Bernoulli Naive Bayes** → Binary/boolean features.


## 3. When to Use
- Text classification (spam detection, sentiment analysis)
- Document categorization
- Medical diagnosis
- Real-time predictions (fast and efficient)


## 4. Implementation in Python

### Gaussian Naive Bayes Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "age": [25, 30, 45, 35, 40, 50, 23, 33],
    "salary": [50000, 60000, 80000, 120000, 70000, 150000, 40000, 90000],
    "buys": [0, 0, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df[["age", "salary"]]
y = df["buys"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Multinomial Naive Bayes (Text Classification)

```

```python

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["I love this product", "This is terrible", "Amazing experience", "Worst purchase ever"]
labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Prediction
test_texts = ["I love it", "Not good"]
X_test = vectorizer.transform(test_texts)
print(model.predict(X_test))

```
## 5. Advantages

- Simple and fast

- Works well with high-dimensional data (e.g., text classification)

- Requires small training data

- Robust to irrelevant features

## 6. Limitations
- Assumes feature independence (not always true in real-world data)

- Struggles with correlated features

- Poor estimates if probability values are very small (can use Laplace smoothing)

## 7. Key Improvements
- Laplace Smoothing → Prevents zero probabilities.

- Feature selection → Helps reduce correlation between features.

- Hybrid models → Combining Naive Bayes with other classifiers.

## ✅ Summary:

Naive Bayes is a fast, simple, and surprisingly effective classification algorithm.
It is particularly powerful for text classification and works well as a baseline model for many tasks.
