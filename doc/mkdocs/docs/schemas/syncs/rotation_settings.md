<!-- # RotationSettings

**Title:** RotationSettings

|                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |

**Description:** Settings for the Rotation sync task. -->

**Example:**

```json
{
    "angle": 90,
    "axis": "Z"
}
```

| Property           | Pattern | Type              | Deprecated | Definition | Title/Description                                                                                                            |
| ------------------ | ------- | ----------------- | ---------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| + [angle](#angle ) | No      | enum (of integer) | No         | -          | Rotation angle in degrees. Must satisfy:<br />- angle % 90 == 0<br />- angle % 360 != 0<br />Only 90, 180, or 270 are valid. |
| + [axis](#axis )   | No      | enum (of string)  | No         | -          | Axis of rotation. For 2D tensors only 'Z' is allowed. Supported values are: 'X', 'Y', 'Z'.                                   |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-11-25 at 14:57:17 +0100 -->
