# Test GCN models

## Prerequsite

`algorithm.git` repo is used for middle-ware.

## Data

Data object is defined in `utility.load_data.Data`.

### Adjacent Matrix

`plain_adj` is as follows.

$$
\mathbf{A} = 
\begin{bmatrix}
\mathbf{0} && R \\
\mathbf{R}^T && \mathbf{0}
\end{bmatrix}
$$

`norm_adj` is as follows where $\mathbf{D}$ is the diagonal matrix whose element means the number of edges.

$$
\mathbf{A}_{norm} = \mathbf{D}^{-1} (\mathbf{A} + \mathbf{I})
$$

