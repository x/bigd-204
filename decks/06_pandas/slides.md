---
layout: cover
---

# Pandas

---
layout: image-right
image: decks/06_pandas/image.png
---

## Why Pandas?

### Performance
Built on top of NumPy’s fast numerical arrays.

### Convenience
Offers intuitive data structures.

### Functionality
Tons of built-in data manipulation methods.

### Compatibility
Works with libraries like matplotlib, Seaborn, scikit-learn.

---
layout: image-right
image: decks/06_pandas/image-2.png
backgroundSize: contain
---

## Tablular Data

### Structured Data
Tabular data is organized into rows and columns, making it easy to read and analyze.

### Consistent Data Types
Each column represents a specific attribute (e.g., age, price), while each row is a single record.

### Scalable
Works with small datasets (CSV files) to big data (SQL databases, distributed systems).

---
layout: image-right
image: decks/06_pandas/image-13.png
backgroundSize: contain
---

## The Sumer Data Tables
- The Sumerians (circa 3100 BCE) used clay tablets to record economic transactions.
- Tablets contained summaries, calculations, and even forecasting elements.
- Babylonian numbers were base 60.

<div class="invert-when-dark-mode">

![alt text](./image-14.png)

</div>

---

## Tabular Data in Python

### List-of-Lists

```python
data = [
    ["name", "city", "age"],
    ["Alice", "New York", 25],
    ["Bob", "Los Angeles", 30],
    ["Charlie", "Chicago", 35],
]

age_idx = data[0].index("age")
city_idx = data[0].index("city")  # corrected from your original code
print(data[2][city_idx])
```

### List-of-Dictionaries

```python
data = [
    {"name": "Alice", "city": "New York", "age": 25},
    {"name": "Bob", "city": "Los Angeles", "age": 30},
    {"name": "Charlie", "city": "Chicago", "age": 35},
]

print(data[1]["age"])
```

---

## Series and DataFrames

Pandas provides two fundamental data structures:

1. **Series**  
   - A one-dimensional labeled array.

   ```python
   import pandas as pd
   s = pd.Series([0.25, 0.5, 0.75, 1.0])
   print(s)
   ```

2. **DataFrame**  
   - A two-dimensional labeled data structure, akin to a table.

   ```python
   df = pd.DataFrame({
       'area': [423967, 695662, 141297],
       'population': [38332521, 26448193, 19651127]
   })
   print(df)
   ```

These structures handle the vast majority of typical use cases in finance, social sciences, statistics, etc.

---
layout: image-right
image: decks/06_pandas/table_series.svg
backgroundSize: 50%
---

## Pandas Series

A one-dimensional labeled array.

#### Key Attributes
- **values**: The underlying NumPy array
- **index**: The associated array of labels
- **dtype**: The data type of the values

```python {monaco-run} {autorun: false}
import pandas as pd

s = pd.Series([0.25, 0.5, 0.75, 1.0])
print(s)
```

---

## Series Indexing

A Series can be **indexed** by integer position:

```python {monaco-run} {autorun: false}
s = pd.Series([0.25, 0.5, 0.75, 1.0], index=[0, 1, 2, 3])
print(s[1])
```

Or by a string label:

```python {monaco-run} {autorun: false}
s = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(s['b'])
```

---
layout: image-right
image: decks/06_pandas/table_dataframe.svg
backgroundSize: contain
---

## Pandas DataFrame

A two-dimensional array with flexible row indices and column names (`df`, `foo_df`)

#### Attributes
- **values**: The underlying 2D NumPy array
- **columns**: Index of column labels
- **index**: Index of row labels

```python {monaco-run} {autorun: false}
import pandas as pd

area = pd.Series({'CA': 423967, 'TX': 695662, 'NY': 141297, 'FL': 170312, 'IL': 149995})
population = pd.Series({'CA': 38332521, 'TX': 26448193, 'NY': 19651127, 'FL': 19552860, 'IL': 12882135})
df = pd.DataFrame({'area': area, 'population': population})
print(df)
```

---

## You Can Perform Math Between Series

Because pandas is built on top of NumPy, you can do basic math operations on DataFrames and Series:

```python {monaco-run} {autorun: false}
import pandas as pd

df = pd.DataFrame({
    'area': [423967, 695662, 141297, 170312, 149995],
    'population': [38332521, 26448193, 19651127, 19552860, 12882135]
}, index=['CA', 'TX', 'NY', 'FL', 'IL'])

density = df['population'] / df['area']
print(density)
```

---
layout: two-cols-header-2
---

## And Then We Can Add It to the DataFrame

::left::

DataFrames act like dictionaries, so we can add a new column as long as the indices match:

```python
df['density'] = df['population'] / df['area']
print(df)
```
```
      area  population    density
CA  423967    38332521  90.413926
TX  695662    26448193  38.018740
NY  141297    19651127 138.195210
FL  170312    19552860 114.806121
IL  149995    12882135  85.883763
```

::right::

We can also modify existing columns.

```py
df['age'] = [25, 30, 35, 40, 45]  # Example column
df['age'] = df['age'] + 1
print(df)
```
```
      name  age         city
0    Alice   26     New York
1      Bob   31  Los Angeles
2  Charlie   36      Chicago
```

---
layout: two-cols-two-rows-header
---

## DataFrames Are The Best of Both Worlds

::left-top::

### Like a Dict-of-Lists

```python
df["area"]
```
```
CA  423967
TX  695662
NY  141297
FL  170312
IL  149995
```

::left-bottom::

<br>
```py
df["area"]["TX"]
```
```
695662
```

::right-top::

### Like a List-of-Dicts
```py
df.loc['TX']
```
```
area            695662
population    26448193
Name: TX, dtype: int64
```

::right-bottom::
<br>

```py
df.loc['TX']['area']
```
```
695662
```


---

## Constructing DataFrames

DataFrames can be constructed literally from Python data structures:

```python
import pandas as pd

df1 = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
df2 = pd.DataFrame([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
df3 = pd.DataFrame({'a': [1, 3], 'b': [2, 4]})
```

They can also be constructed from Series:

```python
df4 = pd.DataFrame({
    'a': pd.Series([1, 3]),
    'b': pd.Series([2, 4])
})
```

And the most common way we’ll use pandas in class is to read CSV files<br>(it’s better than python's built-in `csv` module):

```python
df_csv = pd.read_csv('data.csv')
```

---
layout: two-cols-header-2
---


## Viewing Data

::right::

Use `.head()` to view the first few rows.
```py
df.head(2)
```
```
    name  age         city
0  Alice   25     New York
1    Bob   30  Los Angeles
```

::left::

Use `.info()` to get a concise summary of the DataFrame.
```py
df.info()
```
```
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   name    3 non-null      object
 1   age     3 non-null      int64 
 2   city    3 non-null      object
dtypes: int64(1), object(2)
memory usage: 200.0+ bytes
```

---

 ## Simple Aggregate Methods

In addition to normal list operations, `Series` (or columns of a DataFrame) have methods for aggregation:
```py
>>> df['age'].mean()
30.0
>>> df['age'].sum()
35
```

When you use these methods on an entire `DataFrame`, they typically return a `Series` (one result per column):
```py
>>> df.mean(numeric_only=True)
age          31.0
salary    60000.0
```

---

## More Statistical Tools in Pandas

### Measures of Central Tendency

```python {monaco-run} {autorun: false}
import pandas as pd

values = pd.Series([7, 2, 3, 4, 5, 6, 7, 7, 9, 4])
mean = values.mean()
median = values.median()
mode = values.mode()

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode.to_list())  # mode can return multiple values
```

---

## Correlation and Covariance

```python {monaco-run} {autorun: false}
import pandas as pd

data = {
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
}
df_stats = pd.DataFrame(data)

corr = df_stats['x'].corr(df_stats['y'])
cov = df_stats['x'].cov(df_stats['y'])

print("Correlation:", corr)
print("Covariance:", cov)
```

---

## Grouping Data

Pandas has a powerful method for grouping data:

```python {monaco-run} {autorun: false}
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Jill'],
    'age': [25, 30, 35, 32],
    'city': ['New York', 'Los Angeles', 'Los Angeles', 'New York']
})
print(df.groupby('city')['age'].mean())
```


--- 

## Selecting Multiple Columns

Selecting multiple columns returns a `DataFrame`.

```py
df_subset = df[['area', 'population']]
df_subset
```
```
      name         city
0    Alice     New York
1      Bob  Los Angeles
2  Charlie      Chicago
```

---

## Filtering Data

In addition to selecting columns, you can Filter rows based on booleans (the results of conditionals):
```python
older_than_30 = df['age'] > 30
df[older_than_30]
```
```
      name  age     city
2  Charlie   35  Chicago
3    Alice   32  New York
```

By combining multiple conditions, you can filter data more precisely:
```python
is_new_york = df.index == 'NY'
df[older_than_30 & is_new_york]
```
```
      name  age      city
3    Alice   32  New York
```

---
layout: image-right
image: decks/06_pandas/pandas_scatter.png
backgroundSize: contain
---

## Visualizing Data with Pandas

Pandas has built-in plotting capabilities.
- Use `df.plot()` to create simple plots.
- Use `df.plot(kind='bar')` to create a bar plot.
- Use `df.plot(kind='hist')` to create a histogram.
- Use `df.plot(kind='scatter')` to create a scatter plot.