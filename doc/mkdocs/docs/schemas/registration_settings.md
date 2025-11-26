<!-- # RegistrationSettings

- [1. Property `RegistrationSettings > radius`](#radius)

**Title:** RegistrationSettings

|                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |

**Description:** Settings for the image registration task used in holography doppler analysis. -->

**Example:**

```json
{
    "radius": 0.8
}
```

| Property             | Pattern | Type   | Deprecated | Definition | Title/Description                                                                                                                                                                                                                                          |
| -------------------- | ------- | ------ | ---------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [radius](#radius ) | No      | number | No         | -          | Normalized radius of the circular mask applied to the region of interest. Defines the elliptical area used for cross-correlation computation. Must satisfy:<br />- 0.0 ≤ radius ≤ 1.0<br />- Radius is normalized relative to the smallest image dimension |

<!-- ## <a name="radius"></a>1. Property `RegistrationSettings > radius`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

**Description:** Normalized radius of the circular mask applied to the region of interest. Defines the elliptical area used for cross-correlation computation. Must satisfy:
- 0.0 ≤ radius ≤ 1.0
- Radius is normalized relative to the smallest image dimension

| Restrictions |     |
| ------------ | --- |
| **Minimum**  | N/A |
| **Maximum**  | N/A |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-11-26 at 15:57:03 +0100 -->
