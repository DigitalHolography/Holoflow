<!-- # FresnelDiffractionSettings -->

<!-- **Title:** FresnelDiffractionSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Fresnel diffraction propagation task.
 -->

**Example:**

```json
{
    "lambda": 5.32e-07,
    "dx": 3.45e-06,
    "dy": 3.45e-06,
    "z": 0.1,
    "skip_phase_shift": true
}
```

| Property             | Pattern | Type   | Deprecated | Definition | Title/Description                                                         |
| -------------------- | ------- | ------ | ---------- | ---------- | ------------------------------------------------------------------------- |
| + [lambda](#lambda ) | No      | number | No         | -          | Wavelength in meters. Must satisfy:<br />- lambda > 0                     |
| + [dx](#dx )         | No      | number | No         | -          | Pixel pitch in meters along X. Must satisfy:<br />- dx > 0<br />- dx = dy |
| + [dy](#dy )         | No      | number | No         | -          | Pixel pitch in meters along Y. Must satisfy:<br />- dy > 0<br />- dy = dx |
| + [z](#z )           | No      | number | No         | -          | Propagation distance in meters. Can be positive or negative.              |
| - [skip_phase_shift](#skip_phase_shift ) | No      | boolean | No         | -          | When true, omit the output-plane quadratic phase factor after the FFT. When false, multiply the FFT result by exp(i*pi*(x^2+y^2)/(lambda*z)). |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 13:11:10 +0200
-->
