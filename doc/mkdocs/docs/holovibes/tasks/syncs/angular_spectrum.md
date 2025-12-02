# Angular Spectrum Sync Task
The **Angular Spectrum Method (ASM)** models free-space propagation of a wavefield by decomposing it into plane waves and applying a phase shift in the spatial frequency domain.

Given a scalar complex field $U(x, y, 0)$ at $z = 0$, its propagated field at distance $z$ is:

$$
U(x, y, z) = \mathcal{F}^{-1} \{ \mathcal{F}\{U(x, y, 0)\} \cdot H(f_x, f_y, z) \}
$$

where
$\mathcal{F}$ and $\mathcal{F}^{-1}$ are the 2D Fourier and inverse Fourier transforms,
and the **transfer function** $H(f_x, f_y, z)$ is:

$$
H(f_x, f_y, z) = \exp\left( i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2} \right)
$$


where:

* $\lambda$: wavelength
* $f_x, f_y$: spatial frequencies
* $z$: propagation distance

See [Angular Spectrum Method - Wikipedia](https://en.wikipedia.org/wiki/Angular_spectrum_method) and [Angular Spectrum Method - LibreTexts](https://phys.libretexts.org/Bookshelves/Optics/BSc_Optics_(Konijnenberg_Adam_and_Urbach)/06%3A_Scalar_diffraction_optics/6.04%3A_Angular_Spectrum_Method) for more details.

!!! note
    This task can perform an optional **frequency domain filtering** step specified by the **optional** `filter` setting. See the [Filter 2D](filter2d.md) task documentation for more details on the available filter types and settings.


## Inputs
This task has a single input of shape `(B, H, W)`, where:

- `B`: batch size
- `H`: height of the input field
- `W`: width of the input field

The dtype of the input tensor must be complex 32-bit (`complex32`).

The memory location of the input tensor must be device memory.

## Outputs
This task has a single output of shape `(B, H, W)`, where:

- `B`: batch size (same as input)
- `H`: height of the output field (same as input)
- `W`: width of the output field (same as input)

The dtype of the output tensor is complex 32-bit (`complex32`).

The memory location of the output tensor is device memory.

## Inplace
This task has an inplace relationship between its input and output.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\angular_spectrum_settings.md"