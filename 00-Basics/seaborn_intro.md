# Introduction to Seaborn

Seaborn is a **statistical data visualization library** built on top of **Matplotlib**.  
It provides a **high-level interface** for drawing attractive and informative graphics.  

Seaborn works especially well with **Pandas DataFrames**.


## 1. Importing Seaborn

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

## 2. Basic Dataset Example
Seaborn includes built-in datasets. Let’s use the famous Iris dataset.
```python
df = sns.load_dataset("iris")
print(df.head())
```

## 3. Distribution Plot

```python
sns.histplot(df["sepal_length"], bins=20, kde=True)
plt.title("Distribution of Sepal Length")
plt.show()
```
## 4. Scatter Plot with Seaborn

```python
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
plt.title("Sepal vs Petal Length by Species")
plt.show()
```

## 5. Box Plot

```python
sns.boxplot(x="species", y="sepal_width", data=df)
plt.title("Boxplot of Sepal Width by Species")
plt.show()
```
## 6. Violin Plot

```python
sns.violinplot(x="species", y="petal_length", data=df, palette="muted")
plt.title("Violin Plot of Petal Length")
plt.show()
```

## 7. Pair Plot

```python
sns.pairplot(df, hue="species")
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()
```

## 8. Heatmap (Correlation Matrix)

```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

### ✅ Summary

- With Seaborn, you can easily create:

- Distribution plots (histplot, kdeplot)

- Scatter and categorical plots (scatterplot, boxplot, violinplot)

- Pair plots for relationships

- Heatmaps for correlation matrices

- Seaborn is ideal for EDA (Exploratory Data Analysis) and works seamlessly with Pandas.
