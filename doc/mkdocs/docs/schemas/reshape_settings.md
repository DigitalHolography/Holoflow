<!-- # ReshapeSettings -->

<!-- **Title:** ReshapeSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Reshape sync task.
 -->

**Example:**

```json
{
    "shape": [
        1,
        512,
        512
    ]
}
```

| Property           | Pattern | Type             | Deprecated | Definition | Title/Description                                                                                                                                                                                   |
| ------------------ | ------- | ---------------- | ---------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [shape](#shape ) | No      | array of integer | No         | -          | Desired output shape expressed as a list of dimensions. Must satisfy:<br />- shape has at least one dimension<br />- no dimension is zero<br />- product(shape) equals the number of input elements |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 13:11:11 +0200
-->