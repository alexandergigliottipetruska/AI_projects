## Long Short-Term Memory (LSTM)

### Overview
This section implements a **Long Short-Term Memory (LSTM)** network, a gated recurrent architecture designed to address the **vanishing gradient problem** in sequential models. Unlike vanilla RNNs, LSTMs maintain both:
- **Hidden state** $h_t$: short-term representation used for output
- **Cell state** $c_t$: long-term memory that persists across timesteps

The architecture introduces learnable **gates** to control the flow of information:
- **Forget gate**: removes irrelevant information
- **Input gate**: selects new information to store
- **Output gate**: determines what information to expose

The input is of shape $(batch\_size, seq\_length, num\_features)$.

---

## Forward Propagation

At each timestep $t$, the LSTM computes:

$$
\begin{aligned}
i_t &= \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
f_t &= \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
g_t &= \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
o_t &= \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

where:
- $x_t$ is the input at timestep $t$
- $h_{t-1}$ is the previous hidden state
- $c_{t-1}$ is the previous cell state
- $i_t, f_t, o_t \in [0,1]$ are gate activations (sigmoid)
- $g_t$ is the **candidate memory** (tanh)
- $W, b$ are learnable parameters
- $\odot$ denotes element-wise multiplication

The **cell state update**:

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t
$$

can be interpreted as:
- **forgetting** old information ($f_t \odot c_{t-1}$)
- **adding** new candidate information ($i_t \odot g_t$)

The hidden state is a gated version of the cell state:

$$
h_t = o_t \odot \tanh(c_t)
$$

---

## Vanishing Gradients

Vanishing gradients occur in vanilla RNNs due to repeated multiplication of gradients by values less than 1, causing exponential decay across timesteps.

LSTMs mitigate this through:
- **Additive memory updates** in the cell state
- **Learnable gates** that regulate information flow
- A **near-linear gradient path** through $c_t$

This enables LSTMs to:
- Capture long-range dependencies
- Maintain stable gradients during training
- Learn from long sequences effectively

---

## Notes
- The cell state provides a **persistent memory pathway** across timesteps
- The candidate values $g_t$ represent **new information proposals** before gating
- LSTMs introduce additional parameters compared to RNNs, increasing computational cost but significantly improving performance on long sequences
- Common applications include **language modeling, time-series forecasting, and sequence prediction**
- Time complexity per forward pass: $O(T \cdot H^2)$, similar to RNNs but with larger constant factors due to multiple gates
