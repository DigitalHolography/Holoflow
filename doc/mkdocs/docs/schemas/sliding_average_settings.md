# SlidingAverageSettings

<!-- **Title:** SlidingAverageSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the SlidingAverage async task.
 -->

**Example:**

```json
{
    "target_capacity": 32,
    "window_size": 8
}
```

| Property                               | Pattern | Type    | Deprecated | Definition | Title/Description                                                                                                         |
| -------------------------------------- | ------- | ------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| + [target_capacity](#target_capacity ) | No      | integer | No         | -          | Number of extra frames the circular buffer can hold beyond the averaging window. Must satisfy:<br />- target_capacity > 0 |
| + [window_size](#window_size )         | No      | integer | No         | -          | Number of frames included in each running average. Must satisfy:<br />- window_size > 0                                   |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 11:09:40 +0200
-->