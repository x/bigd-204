---
layout: section
---

# Feature Engineering

---
layout: image-right
image: decks/10_feature_engineering/image.png
backgroundSize: contain
---

## What is Feature Engineering?

Feature engineering is the process of **using domain knowledge to create or transform variables** (features) from raw data into formats that help machine learning models learn better.  

### Goal
1. **Improve model performance** and interpretability.  
2. **Reduce complexity** by transforming or encoding data more effectively.  
3. **Leverage domain insights** so that the features *best capture underlying patterns*.  

---
layout: image-right
image: decks/10_feature_engineering/image-1.png
backgroundSize: contain
---

## Why is Feature Engineering Important?

1. **Enhances Model Performance**  
   Good features can significantly improve accuracy and reduce error.  

2. **Reduces Data Complexity**  
   Proper transformations can help algorithms converge faster and more reliably.  

3. **Domain Knowledge**  
   Adds “expert insight” that purely algorithmic approaches might miss.  

---

## Polynomial Features

### Polynomial Transformation

Create new features by raising existing numeric features to a power (e.g., $x^2, x^3$).

### Interaction Term

Multiply or combine features (e.g., $x_1 \times x_2$, $x_1 / x_2$).  

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

<br>

_We've already seen something similar to this in Mean Squared Error. The polynomial degree is the same concept, but applied to the features themselves._

---
layout: image-right
image: decks/10_feature_engineering/image-2.png
backgroundSize: contain
---

## Interaction Features

Combine features in pairs or more to capture potential relationships.  

### Examples
- Multiplying two features: $x_1 \times x_2$ 
- Dividing features: $x_1 / x_2$ 
- Combining text-based features (e.g., bigrams, trigrams in NLP).  

### Real-World Examples
- `price_per_sqft` = `price` / `sqft`
- `BMI` = `weight` / `height` $^2$
- `force` = `mass` $\times$ `acceleration`



---
layout: image-right
image: decks/10_feature_engineering/image-3.png
backgroundSize: contain
---

## Normalization & Standardization

### Normalization

Scale features to a fixed range. Useful for algorithms that rely on distances or magnitude (e.g., KNN, neural networks)

$$
x_{norm} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$


### Standardization

Transform features to have mean $0$ and variance $1$. Useful for linear models, logistic regression, SVM.

$$
x_{std} = \frac{x - \mu}{\sigma}
$$

---

## De-Skewing & Transformation

Reduce skewness, handle outliers, or stabilize variance.  

### Log Transform

$x' = \log(x + 1)$

<small>(“+1” to handle zero or near-zero values)</small>

### Box-Cox / Yeo-Johnson

More general transformations that handle negative and zero values gracefully.  

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
X_trans = pt.fit_transform(X)
```

---

## One-Hot Encoding (Categorical Data)

Converts each category into a separate binary feature (0 or 1) so numerical algorithms can process them.

This is closely related to bag-of-words in NLP.

```python {monaco-run} {autorun: false}
data = pd.DataFrame({"color": ["red", "blue", "green", "red"], "values": [5, 2, 3, 8]})
print("Before encoding:")
print(data)

data = pd.get_dummies(data, columns=["color"])
print("\nAfter encoding:")
print(data)
```

---

## Binning (Discretization)

Converting continuous features into categories by splitting the range into intervals (bins).  
### Example

Age → [0-18], [19-35], [36-50], [51+].  

### Usage

- Helps capture *non-linear relationships* or reduce outlier impact.  
- But can cause loss of granularity.

---
layout: image-right
image: decks/10_feature_engineering/image-4.png
backgroundSize: contain
---

## Dealing with Missing Data

### Simple Methods
Replace missing values with mean, median, or mode.  

### Advanced Methods
KNN imputation, iterative imputer, or domain-specific rules.  

### Indicator Variables
Create a binary feature “was_missing” to capture whether data was missing.

---
layout: image-right
image: decks/10_feature_engineering/image-5.png
backgroundSize: contain
---

## Date/Time Feature Engineering

### Extract Components
Day of week, month, hour, whether it’s a holiday, etc.  

### Cyclical Encoding

When dealing with periodic data (hours in a day, days in a week, months in a year), transform them into sine/cosine pairs.
$$
  x_{sin} = \sin\Bigl(\frac{2\pi \times \text{hour}}{24}\Bigr), 
  \quad
  x_{cos} = \cos\Bigl(\frac{2\pi \times \text{hour}}{24}\Bigr)
$$

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
```

---
layout: image-right
image: decks/07_nlp/nlp-3.png
backgroundSize: contain
---

## Text Feature Engineering
As we saw in NLP, text data has multiple transformations into numerical features.

### Bag-of-Words
Converts text into counts of each word.  

### TF-IDF
Accounts for frequency across documents.  

### Embeddings
Word2Vec / GloVe / BERT embeddings for deeper semantic representation.
