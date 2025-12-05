# Fresnel Diffraction Sync Task
The **Fresnel Diffraction** sync task models near-field propagation of a complex wavefront by applying the single-FFT Fresnel transform on the GPU. For each frame, it multiplies the sampled input field $U_0(x, y)$ by the quadratic phase factor $\exp\!\left(i \frac{\pi}{\lambda z} (x^2 + y^2)\right)$ and performs a 2D forward FFT across the spatial axes:

$$
U_z(f_x, f_y) \propto \mathcal{F}\left\{ U_0(x, y) \cdot \exp\!\left(i \frac{\pi}{\lambda z} (x^2 + y^2)\right) \right\}.
$$

The proportionality constant and output-plane chirp that appear in the analytical Fresnel integral are intentionally omitted. Positive `z` propagates the field forward, negative `z` back-propagates it. 

See [Wikipedia](https://en.wikipedia.org/wiki/Fresnel_diffraction) for more details on the Fresnel diffraction integral.

!!! warning
    Set the propagation distance `z` to a non-zero value. A zero distance would yield an infinite phase term during lens generation, producing `NaN` coefficients and undefined output values.

## Inputs
This task expects a single complex tensor of shape `(B, H, W)`:

- `B`: batch size (number of frames)
- `H`: number of rows
- `W`: number of columns

The dtype must be `complex32` (`CF32`) and the tensor must reside in device memory.

## Outputs
One output tensor is produced with the same shape `(B, H, W)`, dtype `complex32`, and device memory location as the input. The data contains the Fresnel-propagated field samples at distance `z` up to a global complex scaling factor.

## Inplace
This task has an inplace relationship between its input and output.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\fresnel_diffraction_settings.md"
