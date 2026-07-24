# PCA Sync Task
The **Principal Component Analysis (PCA)** sync task projects real U8 or FP32 data onto a reduced
set of orthogonal modes. Given an input tensor $X \in\mathbb{R}^{F \times H \times W}$, the task
reshapes it into a sample matrix $X_{(N \times F)}$ with $N = H \cdot W$, forms the symmetric covariance

$$
C = X^\mathrm{H} X,
$$

computes the eigenpairs associated with indices `begin` (inclusive) through `end` (exclusive), and finally produces the principal component responses

$$
Y = X \, V_{[\text{begin}:\text{end}]},
$$

where the columns of $V$ are the selected eigenvectors. 

!!! warning
    The input is not mean-centered before decomposition.

!!! note
    Eigenvalues are returned in ascending order. To capture the most energetic components, choose indices close to `F` (for example, `begin = F - K`, `end = F` to keep the `K` largest modes). 

## Inputs
This task has a single input of shape `(F, H, W)`, where:

- `F`: number of features (e.g., frames or spectral slices)
- `H`: height (rows)
- `W`: width (columns)

The dtype must be unsigned 8-bit (`uint8`) or real 32-bit (`float32`), and the tensor must reside
in device memory. The U8 path reads the original input directly and converts values during the
cuBLASDx covariance and projection operations; it does not materialize an FP32 input cube. The
FP32 cuBLAS path remains available as a reference implementation. Complex FP32 and U16 PCA are
not supported.

## Outputs
This task produces a single output of shape `((end - begin), H, W)`, where:

- `end - begin`: number of retained principal components
- `H`: height (same as input)
- `W`: width (same as input)

The dtype of the output tensor is real 32-bit (`float32`) and it is written to device memory.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\pca_settings.md"
