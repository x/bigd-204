---
layout: section
---

# Unsupervised Machine Learning

---
layout: image-right
image: decks/11_machine_learning_unsupervised/image-4.png
backgroundSize: contain
---

## Unsupervised Learning

Trains algorithms on *unlabeled* data to find patterns, structures, or anomalies.  

### Goal
- Discover hidden relationships or groupings.
- Preprocessing step for exploratory analysis.

### Applications
- **Exploratory data analysis**: Gain insights before building predictive models.  
- **Feature engineering**: Transform features for better performance in supervised tasks.  
- **Pattern Recognition**: Identify clusters, outliers, or trends.

---
layout: section
hideInToc: true
---

# Clustering

---
layout: image-right
image: decks/11_machine_learning_unsupervised/image-5.png
backgroundSize: contain
---

## What is Clustering?

Grouping data points so that points in the same cluster are more similar to each other than those in different clusters.  

- **Applications**:  
  - **Market Segmentation**: Grouping customers with similar buying patterns.  
  - **Document Clustering**: Grouping news articles or research papers by topic.  
  - **Image Segmentation**: Identifying different regions in images (e.g., for medical imaging).

---
layout: two-cols-header-2
image: decks/11_machine_learning_unsupervised/image.gif
backgroundSize: contain
---

## K-Means Clustering

::left::

1. **Initialize** $k$ cluster centroids randomly.
2. **Assign** each data point to its nearest centroid.
3. **Recompute** each centroid as the mean of points in its cluster.
4. **Repeat** until cluster assignments stop changing (convergence).

**Objective Function**:
$$
\min \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2 
$$
where $\mu_i$ is the centroid of cluster $C_i$.

::right::

In Python, we can use the `KMeans` class from `sklearn` to perform K-Means clustering.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)
```

**Key Consideration**:  
- **Scaling** your features w/ Feature Engineering removes bias towards high-variance features.
- **Choosing $k$** can be guided by the “elbow method” or silhouette score.

---
layout: image
image: decks/11_machine_learning_unsupervised/image.gif
backgroundSize: contain
---

---
layout: image
image: decks/11_machine_learning_unsupervised/image-3.png
backgroundSize: contain
---

---
layout: image
image: decks/11_machine_learning_unsupervised/image-1.png
backgroundSize: contain
---

---
layout: image
image: decks/11_machine_learning_unsupervised/image-2.png
backgroundSize: contain
---

---
layout: image
image: decks/11_machine_learning_unsupervised/image-7.png
backgroundSize: contain
---

---
layout: section
hideInToc: true
---

# Anomaly Detection

---
layout: image-right
image: decks/11_machine_learning_unsupervised/image-9.png
backgroundSize: contain
---

## What is Anomaly Detection?

Identifying *outliers* or *rare events* in data that differ significantly from the norm.

### Applications

- Data cleaning: Remove outliers before training models.
- Fraud detection: Identify unusual credit card transactions.
- Security: Detect malicious network activity.

---
layout: image-right
image: decks/11_machine_learning_unsupervised/image-10.png
backgroundSize: contain
---

## Traditional Outlier Detection

### Using percentiles
- $x < p5$ or $x > p95$

### Using IQR
- $IQR = p75 - p25$
- $x < Q_1 - 1.5 * IQR$
- $x > Q_3 + 1.5 * IQR$

### Using z-scores
- $z = \frac{x - \bar{x}}{stddev}$
- $z > 3$ or $z < -3$


---
layout: image-right
image: decks/11_machine_learning_unsupervised/image-6.png
backgroundSize: contain
---

## ML-Based Methods

### Isolation Forest
- Randomly select a feature and split data; anomalies are isolated *faster* than normal points.  

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.02, random_state=42)
preds = iso_forest.fit_predict(X)
```

### DBSCAN
<p class="subheading"><small>Density-Based Spatial Clustering of Applications w/ Noise</small></p>

- Finds core samples of high density and expands clusters from them.  
- Points in low-density regions are considered anomalies (noise).

---
layout: section
hideInToc: true
---

# Principal Component Analysis
### (PCA)

---
layout: image-right
image: decks/11_machine_learning_unsupervised/image.png
backgroundSize: contain
---

## What is PCA?

PCA identifies the directions of greatest variance in the data and projects the data onto these directions, creating new features called **principal components**.

### Goal
  - Capture the bulk of variance in fewer dimensions.  
  - Eliminate redundancy or noise in high-dimensional data.

<br>

> **Feature Engineering**: PCA can serve as a *feature extraction* method, creating new uncorrelated features (principal components) from existing ones.

---
layout: image-right
image: decks/11_machine_learning_unsupervised/image-11.png
backgroundSize: contain
---

## PCA Steps

1. **Standardize the Data**  
   Use mean = 0, variance = 1
2. **Compute the Covariance Matrix**  
   Measures how features vary together.
3. **Eigenvalues & Eigenvectors**  
   Eigenvectors point to directions of maximum variance.
4. **Project the Data**  
   Transform original data onto these new axes.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_std)
```


---
layout: header-link
hideInToc: true
---

# Exercise

[https://shorturl.at/GtDys](https://shorturl.at/GtDys)

(Let's do this one together)

---
layout: header-link
hideInToc: true
---

# Exercise

[https://shorturl.at/CIMXx](https://shorturl.at/CIMXx)

(Do this one on your own)