# AngularSpectrumSettings

<!-- **Title:** AngularSpectrumSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Angular Spectrum propagation task.
 -->

**Example:**

```json
{
    "lambda": 5.32e-07,
    "dx": 3.45e-06,
    "dy": 3.45e-06,
    "z": 0.1,
    "filter": {
        "r_inner": 120,
        "r_outer": 220,
        "s_inner": 16,
        "s_outer": 32
    }
}
```

| Property             | Pattern | Type   | Deprecated | Definition | Title/Description                                                                                                                                     |
| -------------------- | ------- | ------ | ---------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [lambda](#lambda ) | No      | number | No         | -          | Wavelength in meters. Must satisfy:<br />- lambda > 0                                                                                                 |
| + [dx](#dx )         | No      | number | No         | -          | Pixel pitch along X in meters. Must satisfy:<br />- dx > 0<br />- dx == dy                                                                            |
| + [dy](#dy )         | No      | number | No         | -          | Pixel pitch along Y in meters. Must satisfy:<br />- dy > 0<br />- dy == dx                                                                            |
| + [z](#z )           | No      | number | No         | -          | Propagation distance in meters. Positive values propagate forward; negative values back-propagate.                                                    |
| - [filter](#filter ) | No      | object | No         | -          | Optional bandlimiting filter expressed in pixels. Must satisfy:<br />- r_inner >= 0<br />- r_outer >= r_inner<br />- s_inner >= 0<br />- s_outer >= 0 |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 11:09:40 +0200
-->