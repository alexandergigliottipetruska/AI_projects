## K-Means Clustering

### Overview
This section implements the **K-Means** clustering algorithm from scratch. The model includes the following hyperparameters:
- number of clusters $K$
- random seed
- maximum number of iterations
- tolerance (for convergence)

The goal of K-Means is to minimize the **within-cluster sum of squared distances** (also called **inertia**).

---

## Objective Function

$$
\min_{\{m_k\}, \{r^{(n)}\}}
\sum_{n=1}^{N} \sum_{k=1}^{K}
r_k^{(n)} \, \| x^{(n)} - m_k \|_2^2
$$

where:
- $x^{(n)}$ is the $n$-th data point  
- $m_k$ is the $k$-th cluster center  
- $r^{(n)}$ is a one-hot assignment vector  

---

## Algorithm

The algorithm proceeds as follows:

1. **Initialization**  
   Randomly initialize cluster centers $m_1, \dots, m_K$

2. **Repeat until convergence**:
   - **Assignment step**
   - **Update (refitting) step**

---

## Assignment Step

Each data point is assigned to the nearest cluster center:

$$
r_k^{(n)} =
\begin{cases}
1 & \text{if } k = \arg\min_{j} \| x^{(n)} - m_j \|_2^2 \\
0 & \text{otherwise}
\end{cases}
$$

---

## Update Step (Refitting)

Cluster centers are updated as the mean of assigned points:

$$
m_k =
\frac{
\sum_{n=1}^{N} r_k^{(n)} x^{(n)}
}{
\sum_{n=1}^{N} r_k^{(n)}
}
$$

---

## Optimization Perspective

K-Means can be viewed as **block coordinate descent**, alternating between:

- minimizing the objective with respect to assignments $r^{(n)}$
- minimizing with respect to cluster centers $m_k$

If assignments are fixed, the optimal centers are means.  
If centers are fixed, assignments are nearest neighbors.

---

## Convergence

The algorithm iterates until:
- the change in inertia is below a threshold (tolerance), or  
- the maximum number of iterations is reached  

---

## Notes
- The objective is **non-convex**, so K-Means can converge to local minima  
- Initialization matters significantly  
- This implementation uses **random initialization**, unlike  
  **k-means++**, which improves stability
- Time complexity per iteration: $O(NKd)$
