## Logistic Regression (Binary & Multinomial)

### Overview
This section implements **logistic regression** for both:
- **Binary classification** (sigmoid)
- **Multiclass classification** (softmax)

Models are trained using **gradient descent** with **cross-entropy loss** and optional **L2 regularization**.

---

## Binary Logistic Regression

The model is defined as:

$$
z = w^T x
$$

$$
y = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

where $\sigma(z)$ maps real-valued inputs to $(0,1)$.

### Prediction
- $y > 0.5 \rightarrow$ class 1  
- $y < 0.5 \rightarrow$ class 0  

---

### Binary Cross-Entropy Loss

$$
L_{BCE} = -t \log(y) - (1 - t)\log(1 - y)
$$

Dataset-level objective:

$$
E(w) = \frac{1}{N} \sum_{i=1}^N L_{BCE}^{(i)}
$$

---

## Multinomial Logistic Regression (Softmax)

For $K > 2$ classes:

$$
z = Wx
$$

$$
y_k = \frac{e^{z_k}}{\sum_{m=1}^{K} e^{z_m}}
$$

This defines a probability distribution:
- $0 \le y_k \le 1$
- $\sum_{k=1}^K y_k = 1$

---

### Categorical Cross-Entropy Loss

$$
L_{CE} = -\sum_{k=1}^K t_k \log(y_k) = -t^T \log(y)
$$

Dataset-level objective:

$$
E(W) = \frac{1}{N} \sum_{i=1}^N L_{CE}^{(i)}
$$

where $t$ is a one-hot encoded vector.

---

## Optimization

Logistic regression has **no closed-form solution**, so parameters are learned via gradient descent:

$$
w \leftarrow w - \alpha \nabla E(w)
$$

For both binary and softmax regression, the gradient simplifies to:

$$
\nabla E(w) = \frac{1}{N} X^T (y - t)
$$

---

## Regularization (L2)

To reduce overfitting, L2 regularization is added:

$$
E(w) = E_{\text{data}}(w) + \frac{\lambda}{2} \|w\|_2^2
$$

Gradient becomes:

$$
\nabla E(w) = \frac{1}{N} X^T (y - t) + \lambda w
$$

---

## Notes
- Binary logistic regression uses the **sigmoid** function  
- Multiclass logistic regression uses **softmax**  
- Both are trained via **cross-entropy loss**  
- Gradient simplifies elegantly to $X^T(y - t)$  
