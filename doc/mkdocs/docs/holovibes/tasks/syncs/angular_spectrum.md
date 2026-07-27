# Angular Spectrum Sync Task
The **Angular Spectrum Method (ASM)** models free-space propagation of a wavefield by decomposing it into plane waves and applying a phase shift in the spatial frequency domain.

Given a scalar real or complex field $U(x, y, 0)$ at $z = 0$, its propagated field at distance $z$ is:

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

!!! note
    The optional `padding` setting centers the input in a larger zero-filled grid before
    propagation. The requested width and height must be at least the input dimensions, and the
    number of pixels added on each axis must be even.

## Inputs
This task has a single input tensor of rank 2 or higher. The last two dimensions are the propagated spatial dimensions:

- `H`: height of the input field
- `W`: width of the input field

Any leading dimensions are treated as batch dimensions. The dtype of the input tensor must be 32-bit real (`float32`) or complex 32-bit (`complex32`).

The memory location of the input tensor must be device memory.

## Outputs
This task has a single complex 32-bit (`complex32`) output. Without padding, its shape matches the
input. With padding, the last two dimensions are replaced by the requested padded height and width;
all leading batch dimensions are preserved.

The memory location of the output tensor is device memory.

## Inplace
Without padding, this task has an inplace relationship between its input and output for complex
input. Padding and real input require a separate complex output.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs/schemas/syncs/angular_spectrum_settings.md"
