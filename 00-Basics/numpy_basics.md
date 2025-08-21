# NumPy Basics

NumPy (Numerical Python) is the core library for numerical and scientific computing in Python.  

It provides support for:
- Multidimensional arrays
- Mathematical functions
- Linear algebra
- Random number generation  

This notebook introduces the **essential NumPy operations** that every Machine Learning student should know.


# Import NumPy
```python
import numpy as np
```

## 1. Creating Arrays

You can create NumPy arrays from Python lists or using built-in functions.

# From a Python list
```python
a = np.array([1, 2, 3, 4, 5])
print("Array from list:", a)
```

# Using arange (start, stop, step)
```python
b = np.arange(0, 10, 2)
print("Array using arange:", b)
```

# Using linspace (start, stop, number of points)
```python
c = np.linspace(0, 1, 5)
print("Array using linspace:", c)
```

# Zeros and Ones

```python
d = np.zeros((2, 3))
e = np.ones((2, 3))
print("Zeros:\n", d)
print("Ones:\n", e)
```

# Identity matrix
```python
f = np.eye(3)
print("Identity Matrix:\n", f)
```

## 2. Array Properties

Each array has attributes that describe its shape and data.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Array:\n", arr)
print("Shape:", arr.shape)      # Rows, Columns
print("Dimensions:", arr.ndim)  # Number of dimensions
print("Size:", arr.size)        # Total number of elements
print("Data type:", arr.dtype)  # Data type of elements
```
## 3. Indexing and Slicing

You can access elements and subarrays just like Python lists, but with more flexibility.

```python
arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

print("Element at (0,0):", arr[0, 0])
print("First row:", arr[0, :])
print("Second column:", arr[:, 1])
print("Submatrix:\n", arr[0:2, 1:3])

```
# Boolean indexing
```python
print("Elements greater than 50:", arr[arr > 50])
```

## 4. Array Operations

NumPy supports element-wise operations and broadcasting.

```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])


print("Addition:", x + y)
print("Subtraction:", x - y)
print("Multiplication:", x * y)
print("Division:", x / y)
print("Dot Product:", np.dot(x, y))

```

# Broadcasting
```python
z = np.array([1, 2, 3])
print("Broadcasted Addition:\n", arr + z)
```

## 5. Useful Functions

NumPy includes many mathematical and statistical functions.

```python
arr = np.array([1, 2, 3, 4, 5])

print("Mean:", np.mean(arr))
print("Standard Deviation:", np.std(arr))
print("Sum:", np.sum(arr))
print("Max:", np.max(arr))
print("Min:", np.min(arr))
```

# Reshape
```python
matrix = np.arange(1, 10).reshape(3, 3)
print("Reshaped matrix:\n", matrix)
```

# Transpose
```python
print("Transpose:\n", matrix.T)
```

## 6. Random Numbers

NumPy has a powerful random module.

# Random integers between 0 and 10

```python
rand_ints = np.random.randint(0, 10, size=(3, 3))
print("Random Integers:\n", rand_ints)
```
# Random floats between 0 and 1

```python
rand_floats = np.random.rand(2, 4)
print("Random Floats:\n", rand_floats)
```
# Normal distribution
```python
normal_dist = np.random.randn(5)
print("Normal Distribution:", normal_dist)
```

# ✅ Summary

In this notebook, we covered:
- Creating arrays (`array`, `arange`, `linspace`, `zeros`, `ones`)
- Array properties (`shape`, `ndim`, `size`, `dtype`)
- Indexing, slicing, boolean indexing
- Array operations and broadcasting
- Useful mathematical/statistical functions
- Random number generation

NumPy is the **foundation of data science & ML** workflows in Python. 
Mastering it will make working with datasets and models much easier.
