# Linear, LASSO, and Ridge Regression

## Overview
This project implements the following regression models from scratch:

- **Linear Regression**
- **LASSO Regression (L1 regularization)**
- **Ridge Regression (L2 regularization)**

All models are trained using **mini-batch gradient descent** with **Mean Squared Error (MSE)** loss. Closed-form solutions are also provided where applicable.

---

## Linear Regression

A linear regression model is defined as:

$$
y = w^T x + b
$$

Using augmented notation (absorbing the bias into $w$), the objective function is:

$$
E(w) = \frac{1}{2N} \|Xw - t\|_2^2
$$

where:
- $X \in \mathbb{R}^{N \times d}$ is the feature matrix  
- $w \in \mathbb{R}^d$ are the parameters  
- $t \in \mathbb{R}^N$ are the targets  

### Closed-form solution

$$
w = (X^T X)^{-1} X^T t
$$

However:
- Computing $(X^T X)^{-1}$ costs $O(d^3)$  
- The matrix may be non-invertible in practice  

---

## Gradient Descent Optimization

Instead, we optimize using gradient descent:

$$
w \leftarrow w - \alpha \nabla E(w)
$$

where:

$$
\nabla E(w) = \frac{1}{N} X^T (Xw - t)
$$

---

## Regularization

To improve generalization and prevent overfitting, regularization terms are added.

---

### LASSO Regression (L1)

$$
J(w) = E(w) + \lambda \sum_{j=1}^{d} |w_j|
$$

Gradient:

$$
\nabla J(w) = \frac{1}{N} X^T (Xw - t) + \lambda \, \text{sign}(w)
$$

where:

$$
\text{sign}(w_j) =
\begin{cases}
1 & \text{if } w_j > 0 \\
0 & \text{if } w_j = 0 \\
-1 & \text{if } w_j < 0
\end{cases}
$$

**Key property:** LASSO can drive weights exactly to zero → performs feature selection.

---

### Ridge Regression (L2)

$$
J(w) = E(w) + \frac{\lambda}{2} \sum_{j=1}^{d} w_j^2
$$

Gradient:

$$
\nabla J(w) = \frac{1}{N} X^T (Xw - t) + \lambda w
$$

**Key property:** Ridge shrinks weights but does not set them exactly to zero.

---

## Mini-batch Gradient Descent

Due to large dataset size, training uses mini-batch gradient descent:

- Each **epoch** = full pass through the dataset  
- Data is **shuffled** each epoch  
- Split into batches of fixed size  
- Final batch may be smaller  

### Advantages
- Faster computation  
- Scales to large datasets  
- Faster convergence in practice  

### Trade-off
- Noisy gradients  
- Smaller batches → higher variance  

---

## Notes
- Gradient-based optimization is preferred for large-scale or high-dimensional problems  
- Regularization strength $\lambda$ is tuned  
- (Optional) Bias term may be excluded from regularization
