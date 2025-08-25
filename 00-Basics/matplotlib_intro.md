# Introduction to Matplotlib

Matplotlib is a **data visualization library** in Python.  
It is widely used for creating **static, interactive, and animated plots**.  
The most common module is **`pyplot`**, which provides a MATLAB-like plotting interface.

---

## 1. Importing Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np
```
2. Basic Line Plot
```python
x = np.linspace(0, 10, 100)   # 100 points between 0 and 10
y = np.sin(x)

plt.plot(x, y)
plt.title("Basic Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

3. Adding Style and Labels

```python
y2 = np.cos(x)

plt.plot(x, y, label="sin(x)", color="blue", linestyle="--")
plt.plot(x, y2, label="cos(x)", color="red", marker="o")
plt.title("Sine and Cosine Waves")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()
```
4. Scatter Plot
```python
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color="green", marker="x")
plt.title("Random Scatter Plot")
plt.show()
```
5. Bar Chart
```python
categories = ["A", "B", "C", "D"]
values = [5, 7, 3, 8]

plt.bar(categories, values, color="orange")
plt.title("Bar Chart Example")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()
```
6. Histogram

```python
data = np.random.randn(1000)  # 1000 random numbers (normal distribution)

plt.hist(data, bins=30, color="purple", edgecolor="black")
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

7. Pie Chart

```python
sizes = [30, 20, 25, 25]
labels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Fruit Distribution")
plt.show()
```
8. Subplots

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x, y1, "b")
axes[0].set_title("Sine Wave")

axes[1].plot(x, y2, "r")
axes[1].set_title("Cosine Wave")

plt.tight_layout()
plt.show()
```

## ✅ Summary
In this file , i've covered:

- Creating line plots, scatter plots, bar charts, histograms, pie charts

- Adding labels, titles, legends, and grid

- Using subplots for multiple charts

