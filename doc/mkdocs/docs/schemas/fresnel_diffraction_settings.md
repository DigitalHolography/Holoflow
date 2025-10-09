# FresnelDiffractionSettings

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
    "z": 0.1
}
```

| Property             | Pattern | Type   | Deprecated | Definition | Title/Description                                                         |
| -------------------- | ------- | ------ | ---------- | ---------- | ------------------------------------------------------------------------- |
| + [lambda](#lambda ) | No      | number | No         | -          | Wavelength in meters. Must satisfy:<br />- lambda > 0                     |
| + [dx](#dx )         | No      | number | No         | -          | Pixel pitch in meters along X. Must satisfy:<br />- dx > 0<br />- dx = dy |
| + [dy](#dy )         | No      | number | No         | -          | Pixel pitch in meters along Y. Must satisfy:<br />- dy > 0<br />- dy = dx |
| + [z](#z )           | No      | number | No         | -          | Propagation distance in meters. Can be positive or negative.              |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-09 at 17:39:35 +0200
-->