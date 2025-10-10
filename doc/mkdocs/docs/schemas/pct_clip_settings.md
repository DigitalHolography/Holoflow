# PctClipSettings

<!-- **Title:** PctClipSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the percentile clipping sync task.
 -->

**Example:**

```json
{
    "min_pct": 0.5,
    "max_pct": 99.5,
    "roi": {
        "cx": 0.5,
        "cy": 0.5,
        "rx": 0.45,
        "ry": 0.45,
        "angle": 0.0
    }
}
```

| Property               | Pattern | Type   | Deprecated | Definition | Title/Description                                                                                                                                              |
| ---------------------- | ------- | ------ | ---------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [min_pct](#min_pct ) | No      | number | No         | -          | Lower percentile (0-100) used to clip values. Must satisfy:<br />- min_pct >= 0<br />- min_pct < max_pct                                                       |
| + [max_pct](#max_pct ) | No      | number | No         | -          | Upper percentile (0-100) used to clip values. Must satisfy:<br />- max_pct <= 100<br />- max_pct > min_pct                                                     |
| + [roi](#roi )         | No      | object | No         | -          | Elliptical region of interest expressed in normalized coordinates. Must satisfy:<br />- 0 <= cx <= 1<br />- 0 <= cy <= 1<br />- 0 < rx <= 1<br />- 0 < ry <= 1 |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 11:09:40 +0200
-->