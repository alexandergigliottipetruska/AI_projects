## K-Nearest Neighbors (KNN)

### Overview
This section implements **K-Nearest Neighbors (KNN)** from scratch. The model includes the hyperparameter $k$, which determines how many neighbors are used for prediction.

Two distance metrics are supported:
- **Euclidean distance**
- **Cosine distance** (computed as $1 - \text{cosine similarity}$)

KNN is a **non-parametric** and **instance-based** learning algorithm.

---

## Method

Given an input point $x$, the algorithm finds the $k$ closest training examples:

$$
\{x^{(1)}, \dots, x^{(k)}\} = \arg\min_{x^{(i)} \in \text{training set}} \text{distance}(x^{(i)}, x)
$$

The predicted label is:

$$
\hat{y} = \text{majority}(t^{(1)}, \dots, t^{(k)})
$$

---

## Distance Metrics

### Euclidean Distance

$$
\|x^{(a)} - x^{(b)}\|_2 = \left( \sum_{j=1}^{d} (x_j^{(a)} - x_j^{(b)})^2 \right)^{1/2}
$$

---

### Cosine Similarity

$$
\text{cosine}(x^{(a)}, x^{(b)}) = \frac{x^{(a)} \cdot x^{(b)}}{\|x^{(a)}\|_2 \|x^{(b)}\|_2}
$$

Cosine distance:

$$
d_{\text{cos}}(x^{(a)}, x^{(b)}) = 1 - \text{cosine}(x^{(a)}, x^{(b)})
$$

---

## Notes
- KNN does not learn parameters during training
- Smaller $k$ → more flexible but noisier
- Larger $k$ → more stable but less sensitive
- Feature scaling is important
