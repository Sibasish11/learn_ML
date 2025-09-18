{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Random Forest Classifier\n",
    "\n",
    "Random Forest is an **ensemble learning method** that builds multiple decision trees and merges them to get more accurate and stable predictions.\n",
    "\n",
    "It reduces overfitting compared to a single Decision Tree."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Sample Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = {\n",
    "    \"age\": [25, 30, 45, 35, 40, 50, 23, 33, 48, 29],\n",
    "    \"income\": [50000, 60000, 80000, 120000, 70000, 150000, 40000, 90000, 110000, 55000],\n",
    "    \"buys\": [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]\n",
    "}\n",
    "df = pd.DataFrame(data)\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[[\"age\", \"income\"]]\n",
    "y = df[\"buys\"]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Training Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = RandomForestClassifier(n_estimators=100, criterion=\"entropy\", random_state=42)\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Predictions & Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"Classification Report:\\n\", classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "importances = model.feature_importances_\n",
    "feature_names = X.columns\n",
    "\n",
    "sns.barplot(x=importances, y=feature_names)\n",
    "plt.title(\"Feature Importance in Random Forest\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Key Notes\n",
    "- Random Forest builds multiple decision trees and averages their results.\n",
    "- Reduces overfitting compared to single trees.\n",
    "- Can handle missing values and categorical features reasonably well.\n",
    "- Useful for both **classification** and **regression**."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

