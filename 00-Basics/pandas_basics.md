# Pandas Basics

Pandas is a Python library built on top of NumPy for **data analysis and manipulation**.  
It provides two main data structures:

- **Series** → 1D labeled array
- **DataFrame** → 2D table (rows & columns)

Pandas is widely used in **Machine Learning, Data Science, and Analytics** workflows.

This notebook introduces the **essential Pandas operations**.

# Import pandas and numpy
```python
import pandas as pd
import numpy as np
```

## 1. Creating Series

A Pandas Series is like a one-dimensional array with labels (index).

# From a list
```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print("Series:\n", s)
```
# From a dictionary
```python
data = {'Math': 90, 'Science': 85, 'English': 88}
s2 = pd.Series(data)
print("\nSeries from dict:\n", s2)
```

## 2. Creating DataFrames

DataFrames are like Excel tables: rows + columns.

# From a dictionary of lists
```python
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [24, 27, 22],
    'Score': [85, 90, 88]
}
df = pd.DataFrame(data)
print("DataFrame:\n", df)
```

# From NumPy array

```python
arr = np.arange(9).reshape(3, 3)
df2 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print("\nDataFrame from NumPy:\n", df2)
```

## 3. Inspecting Data

Common methods to understand your dataset.

```python

print(df.head())     # First 5 rows
print(df.tail(2))    # Last 2 rows
print(df.info())     # Data types & nulls
print(df.describe()) # Summary statistics
print(df.shape)      # Rows, Columns

```

## 4. Indexing and Selecting Data

# Column access
```python
print(df['Name'])
print(df[['Name', 'Score']])
```

# Row access with loc (label) and iloc (position)
```python
print(df.loc[0])   # First row by label
print(df.iloc[1])  # Second row by position
```
# Slicing rows
```
print(df[1:3])
```

## 5. Filtering Data

# Conditional filtering
```python
print(df[df['Age'] > 23])
```

# Multiple conditions

```python
print(df[(df['Age'] > 23) & (df['Score'] > 85)])
```

## 6. Adding and Modifying Columns

```python

# Add new column
df['Passed'] = df['Score'] > 86
print(df)

# Modify existing column
df['Score'] = df['Score'] + 5
print("\nAfter modifying Score:\n", df)

```

## 7. Handling Missing Data

```python

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, np.nan, 22, 28],
    'Score': [85, 90, np.nan, 75]
}
df_missing = pd.DataFrame(data)
print(df_missing)

```
# Fill missing values

```python
print("\nFill NA with mean:\n", df_missing.fillna(df_missing.mean(numeric_only=True)))
```

# Drop rows with missing values

```python
print("\nDrop NA rows:\n", df_missing.dropna())
```

## 8. Grouping and Aggregation

```python

data = {
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'Employee': ['A', 'B', 'C', 'D', 'E'],
    'Salary': [50000, 40000, 55000, 60000, 42000]
}
df_group = pd.DataFrame(data)

```

# Group by department
```python

print(df_group.groupby('Department')['Salary'].mean())

```

# Multiple aggregations
```python
print(df_group.groupby('Department')['Salary'].agg(['mean', 'max', 'min']))
```

## 9. Merging and Joining DataFrames
```python
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Score': [85, 90, 88]})
```
# Merge on ID
```python
print(pd.merge(df1, df2, on='ID', how='inner'))  # inner join
print(pd.merge(df1, df2, on='ID', how='left'))   # left join
```

## 10. Reading and Writing Files

# Save to CSV
```python
df.to_csv('students.csv', index=False)
```

# Read from CSV
```python
df_loaded = pd.read_csv('students.csv')
print(df_loaded)
```

# ✅ Summary

In this notebook, we covered:
- Creating **Series** and **DataFrames**
- Inspecting data with `head()`, `info()`, `describe()`
- Indexing, slicing, and filtering
- Adding/modifying columns
- Handling missing values
- Grouping and aggregation
- Merging/joining DataFrames
- Reading and writing CSV files

Pandas is the **backbone of data analysis in Python**.  
Mastering it will make you comfortable working with real-world datasets.
