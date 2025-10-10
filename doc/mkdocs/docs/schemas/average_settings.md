<!-- # AverageSettings -->

<!-- **Title:** AverageSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Average sync task.
 -->

**Example:**

```json
{
    "axis": 0,
    "start": 0,
    "end": 64
}
```

| Property           | Pattern | Type    | Deprecated | Definition | Title/Description                                                                                       |
| ------------------ | ------- | ------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| + [axis](#axis )   | No      | integer | No         | -          | Axis along which to compute the mean. Must satisfy:<br />- axis in [0, 2]                               |
| + [start](#start ) | No      | integer | No         | -          | Inclusive index where the average begins. Must satisfy:<br />- start >= 0<br />- start < end            |
| + [end](#end )     | No      | integer | No         | -          | Exclusive index where the average stops. Must satisfy:<br />- end > start<br />- end <= size along axis |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 13:11:10 +0200
-->