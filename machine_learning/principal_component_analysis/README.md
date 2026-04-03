## Principal Component Analysis (PCA)

### Overview
This section implements **Principal Component Analysis (PCA)** from scratch. PCA is a dimensionality reduction technique that reduces the number of features in a dataset while preserving the most important information.

It transforms correlated features into a smaller set of uncorrelated ones by computing:
- **Eigenvalues**: measure the importance (variance) of each component
- **Eigenvectors**: define the directions of the new subspace

The top $K$ components with the highest eigenvalues are selected, and the data is linearly projected onto them.

---

## Objective Function

The objective is to **maximize the projected variance** via the empirical covariance matrix:

$$
\overset{\scriptscriptstyle\wedge}{\Sigma} = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \overset{\scriptscriptstyle\wedge}{\mu})(x_i - \overset{\scriptscriptstyle\wedge}{\mu})^T
$$

where:
- $x_i$ is the $i$-th data point
- $\overset{\scriptscriptstyle\wedge}{\mu}$ is the empirical mean
- The optimal subspace is spanned by the top $K$ eigenvectors of $\overset{\scriptscriptstyle\wedge}{\Sigma}$

---

## Projection and Reconstruction

Projecting $x$ onto the subspace yields a code (representation) $z$:

$$
z = U^T(x^{(i)} - \overset{\scriptscriptstyle\wedge}{\mu})
$$

where $U$ is the orthonormal basis of the subspace. The original data can be reconstructed from $z$ via:

$$
\overset{\scriptscriptstyle\sim}{x} = \overset{\scriptscriptstyle\wedge}{\mu} + Uz
$$

---

## Equivalent Objectives

A key property of PCA is that the following three objectives are equivalent and yield the same solution:

**1. Minimize reconstruction error:**
$$\min_{U} \frac{1}{N}\sum^{N}_{i=1}\lVert x^{(i)}-\overset{\scriptscriptstyle\sim}{x}^{(i)}\rVert^2$$

**2. Maximize variance of the reconstruction:**
$$\max_{U} \frac{1}{N}\sum^{N}_{i=1}\lVert \overset{\scriptscriptstyle\sim}{x}^{(i)} - \overset{\scriptscriptstyle\wedge}{\mu} \rVert^2$$

**3. Maximize variance of the representation:**
$$\max_{U} \frac{1}{N}\sum^{N}_{i=1}\lVert z^{(i)} \rVert^2$$

All three objectives lead to the same optimal $U$, the top $K$ eigenvectors of $\overset{\scriptscriptstyle\wedge}{\Sigma}$.

---

## Notes
- PCA assumes that variance is a reliable proxy for information which this holds for Gaussian-like data but may not for nonlinear structures.
- The covariance matrix is symmetric positive semi-definite, so its eigenvectors are orthogonal and eigenvalues are non-negative
- This implementation uses **eigendecomposition** of the covariance matrix directly
- Time complexity: $O(nd^2 + d^3)$ for computing the covariance matrix and its eigendecomposition, where $n$ is the number of samples and $d$ is the number of features
