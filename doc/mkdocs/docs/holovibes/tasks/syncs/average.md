# Average Sync Task
The **Average** sync task computes the arithmetic mean of a 3D tensor along a configurable axis on the GPU. Given an input tensor $X \in \mathbb{C}^{B \times H \times W}$ (or $\mathbb{R}^{B \times H \times W}$), the task collapses the selected axis by averaging the elements between the indices `start` (inclusive) and `end` (exclusive):

$$
Y[\mathbf{i}] = \frac{1}{N} \sum_{n=\text{start}}^{\text{end}-1} X[\mathbf{i}_{k \leftarrow n}], \qquad N = \text{end} - \text{start}
$$

where $k$ is the axis being reduced.

!!! note
    Axis labels follow the `(B, H, W)` layout: axis `0` averages across batches/frames, axis `1` across rows, and axis `2` across columns. The output retains the other two dimensions and replaces the reduced one with size `1`.

!!! warning
    For integer inputs (`uint8`/`uint16`), the averaged value is truncated when cast back to the original dtype.

## Inputs
This task has a single input of shape `(B, H, W)`, where:

- `B`: number of frames or batches
- `H`: height (rows)
- `W`: width (columns)

Valid dtypes are `uint8`, `uint16`, `float32`, and `complex32`. The tensor must reside in device memory.

## Outputs
This task has a single output whose shape matches the input except that the reduced axis is set to `1`:

- axis `0` -> `(1, H, W)`
- axis `1` -> `(B, 1, W)`
- axis `2` -> `(B, H, 1)`

The dtype of the output tensor matches the input dtype. The tensor is written to device memory.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\average_settings.md"
