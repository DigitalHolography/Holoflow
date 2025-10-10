<!-- # BatchQueueSettings -->

<!-- **Title:** BatchQueueSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the BatchQueue async task.
 -->

**Example:**

```json
{
    "target_capacity": 64,
    "output_size": 8,
    "output_stride": 16
}
```

| Property                               | Pattern | Type    | Deprecated | Definition | Title/Description                                                                                                                         |
| -------------------------------------- | ------- | ------- | ---------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| + [target_capacity](#target_capacity ) | No      | integer | No         | -          | Target capacity of the circular buffer measured in input batches. Must satisfy:<br />- target_capacity > 0                                |
| + [output_size](#output_size )         | No      | integer | No         | -          | Number of batches collected into each dequeued block. Must satisfy:<br />- output_size > 0                                                |
| + [output_stride](#output_stride )     | No      | integer | No         | -          | Distance in batches between two dequeued blocks. Must satisfy:<br />- output_stride > 0<br />- output_stride is a multiple of output_size |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 13:11:10 +0200
-->