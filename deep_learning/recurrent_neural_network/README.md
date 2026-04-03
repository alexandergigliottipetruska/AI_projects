## Recurrent Neural Network (RNN)

### Overview
This section implements a **Recurrent Neural Network (RNN)** from scratch using NumPy, including **Backpropagation Through Time (BPTT)**. The RNN consists of:
- **Input layer**: receives the input at each timestep
- **Recurrent layer**: maintains a hidden state across the sequence
- **Output layer**: produces predictions from the hidden state

The input is of shape $(batch\_size, seq\_length, num\_features)$.

---

## Forward Propagation

At each timestep $t$, the hidden state and output are computed as:

$$M_t = X_tW_{hx}^T+H_{t-1}W_{hh}^T+b_h$$
$$H_t = \tanh(M_t)$$
$$O_t = H_tW_{qh}^T+b_q$$

where:
- $X_t$ is the input at timestep $t$
- $H_{t-1}$ is the hidden state from the previous timestep
- $W_{hx}, W_{hh}, W_{qh}$ are the input, recurrent, and output weight matrices
- $b_h, b_q$ are the hidden and output biases

The loss is averaged across all $T$ timesteps:

$$L=\frac{1}{T}\sum^T_{t=1}l(O_t, Y_t)$$

---

## Backpropagation Through Time (BPTT)

Gradients are computed by unrolling the network across all timesteps and propagating errors backwards. At each timestep:

$$\overline{O}_t=\frac{1}{T}(Y_t-O_t)$$
$$\overline{H}_t=\overline{M}_{t+1}W_{hh}^T + \overline{O}_tW_{qh}$$
$$\overline{M}_t=\overline{H}_t\odot(1-\tanh^2(M_t))$$
$$\overline{X}_t=\overline{M}_tW_{hx}$$

Parameter gradients are accumulated across all timesteps:

$$\overline{W}_{qh}=\sum^T_{t=1}(\overline{O}_t)^TH_t$$
$$\overline{W}_{hh}=\sum^T_{t=1}(\overline{M}_t)^TH_{t-1}$$
$$\overline{W}_{hx}=\sum^T_{t=0}(\overline{M}_t)^TX_t$$
$$\overline{b}_{q}=\sum^T_{t=1}(\overline{O}_t)^T\mathbf{1}$$
$$\overline{b}_{h}=\sum^T_{t=1}(\overline{M}_t)^T\mathbf{1}$$

**Note:** Since the task is prediction, the output layer gradients $\overline{W}_{qh}$ and $\overline{b}_q$ use only the final timestep $T$, with no summation.

---

## Vanishing Gradients

Vanishing gradients occur when gradients shrink exponentially as they are backpropagated through many timesteps, effectively preventing earlier layers from receiving useful updates. As a result, the model fails to learn long-range dependencies and loses information from earlier parts of the sequence.

---

## Notes
- This is a **non-modular** implementation, the full forward and backward pass is implemented directly without abstractions
- Vanilla RNNs are sensitive to sequence length due to vanishing gradients so the performance degrades significantly for long sequences
- Gated architectures such as **LSTMs** and **GRUs** were designed specifically to address vanishing gradients by controlling the flow of information through learnable gates
- This implementation uses **MSE loss**, suited for sequence prediction tasks
- Time complexity per forward pass: $O(T \cdot H^2)$ where $T$ is sequence length and $H$ is hidden size, dominated by the recurrent weight multiplication $W_{hh}$
