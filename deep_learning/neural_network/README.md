[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexandergigliottipetruska/AI_projects/blob/main/deep_learning/neural_network/5_neural_network_fashion_mnist.ipynb)
## Neural Network Implementation

### Overview
This section implements a fully-connected **Neural Network** from scratch. The architecture consists of:
- an input layer  
- one hidden layer  
- an output layer  

The model is trained using **mini-batch gradient descent with Adam optimization**.  
Regularization techniques include:
- **Dropout**
- **Weight decay (L2 regularization)**  

The following activation functions are used:
- **ReLU** (hidden layer)
- **Softmax** (output layer)

---

## Forward and Backpropagation

### Single Datapoint

#### Forward Propagation

$$
m = W^{(1)}x + b^{(1)}
$$

$$
h = \text{ReLU}(m)
$$

$$
z = W^{(2)}h + b^{(2)}
$$

$$
y = \text{softmax}(z)
$$

$$
L = L_{\text{CE}}(y, t)
$$

---

#### Backpropagation

$$
\overline{z} = y - t
$$

$$
\overline{W^{(2)}} = \overline{z} \, h^T
$$

$$
\overline{b^{(2)}} = \overline{z}
$$

$$
\overline{h} = (W^{(2)})^T \overline{z}
$$

$$
\overline{m} = \overline{h} \odot \text{ReLU}'(m)
$$

$$
\overline{W^{(1)}} = \overline{m} \, x^T
$$

$$
\overline{b^{(1)}} = \overline{m}
$$

---

### Mini-batch Formulation

#### Forward Propagation

$$
M = X (W^{(1)})^T + b^{(1)}
$$

$$
H = \text{ReLU}(M)
$$

$$
Z = H (W^{(2)})^T + b^{(2)}
$$

$$
Y = \text{softmax}(Z)
$$

$$
E = \frac{1}{N} \sum_{i=1}^{N} L_{\text{CE}}(y^{(i)}, t^{(i)})
$$

---

#### Backpropagation

$$
\overline{Z} = \frac{1}{N}(Y - T)
$$

$$
\overline{W^{(2)}} = (\overline{Z})^T H
$$

$$
\overline{b^{(2)}} = (\overline{Z})^T \mathbf{1}
$$

$$
\overline{H} = \overline{Z} W^{(2)}
$$

$$
\overline{M} = \overline{H} \odot \text{ReLU}'(M)
$$

$$
\overline{W^{(1)}} = (\overline{M})^T X
$$

$$
\overline{b^{(1)}} = (\overline{M})^T \mathbf{1}
$$

---

## Adam Optimization

The **Adam** optimizer combines momentum and adaptive learning rates.

### First and Second Moment Estimates

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla L_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla L_t)^2
$$

---

### Bias Correction

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
\quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

---

### Parameter Update

$$
w_{t+1} = w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

---

## Weight Initialization and Gradient Clipping

### Initialization

- **Xavier Initialization** (for sigmoid/softmax):

$$
x = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \quad
W \sim \text{Uniform}(-x, x)
$$

- **He/Kaiming Initialization** (for ReLU):

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n}}\right)
$$

---

### Gradient Clipping

To prevent exploding gradients, gradients are clipped using:

$$
g \leftarrow \eta \frac{g}{\|g\|}
$$

when $\|g\| > \eta$.

---

## Dropout

Dropout randomly deactivates hidden units with probability $p$:

$$
h' =
\begin{cases}
0 & \text{with probability } p \\
\frac{h}{1 - p} & \text{otherwise}
\end{cases}
$$

This prevents co-adaptation and improves generalization.

During evaluation, dropout is disabled.

---

## Weight Decay

Weight decay adds an $L_2$ penalty to the loss:

$$
E(w) = E_{\text{data}}(w) + \frac{\lambda}{2}\|w\|_2^2
$$

This discourages large weights and helps prevent overfitting.
