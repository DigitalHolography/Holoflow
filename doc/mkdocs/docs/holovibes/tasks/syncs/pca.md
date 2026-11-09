# PCA Sync Task
The **Principal Component Analysis (PCA)** sync task projects real FP32 data onto a reduced set of orthogonal modes. Given an input tensor $X \in\mathbb{R}^{F \times H \times W}$, the task reshapes it into a sample matrix $X_{(N \times F)}$ with $N = H \cdot W$, forms the symmetric covariance

$$
C = X^\mathrm{H} X,
$$

computes the eigenpairs associated with indices `begin` (inclusive) through `end` (exclusive), and finally produces the principal component responses

$$
Y = X \, V_{[\text{begin}:\text{end}]},
$$

where the columns of $V$ are the selected eigenvectors. 

The **Local PCA** time transform applies the same uncentered product at every pixel. For a
`window_size` of 5 and `F = 16`, it gathers a 5×5×16 neighborhood, forms a 16×16 covariance
matrix from the 25 spatial samples, and projects only the center pixel's 16-frame signal. The
spatial window advances by one pixel. `Time Stride` advances the 16-frame input window; setting it
to 1 produces overlapping windows starting one frame apart. Spatial neighborhoods wrap around
both image borders. Local PCA uses
CUDA batched matrix products and solves pixels in memory-bounded tiles; `window_size = 0` retains
the whole-frame behavior described above.
With **Space Transform: None**, the processed view displays the absolute component responses on
the source pixel grid, without Fresnel or Angular Spectrum propagation.

!!! warning
    The input is not mean-centered before decomposition.

!!! note
    Eigenvalues are returned in ascending order. To capture the most energetic components, choose indices close to `F` (for example, `begin = F - K`, `end = F` to keep the `K` largest modes). 

## Inputs
This task has a single input of shape `(F, H, W)`, where:

- `F`: number of features (e.g., frames or spectral slices)
- `H`: height (rows)
- `W`: width (columns)

The dtype must be real 32-bit (`float32`) and the tensor must reside in device memory. Local PCA
requires contiguous input. Complex
FP32 PCA is not yet supported by the cuSolverDx path.

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
--8<-- "docs/schemas/syncs/pca_settings.md"
