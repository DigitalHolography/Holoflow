<!-- # DisplayTensorSettings -->

<!-- **Title:** DisplayTensorSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the DisplayTensor sink task.
 -->

**Example:**

```json
{
    "refresh_rate_hz": 30.0
}
```

| Property                               | Pattern | Type   | Deprecated | Definition | Title/Description                                                                  |
| -------------------------------------- | ------- | ------ | ---------- | ---------- | ---------------------------------------------------------------------------------- |
| + [refresh_rate_hz](#refresh_rate_hz ) | No      | number | No         | -          | Target refresh rate for the UI, in Hertz. Must satisfy:<br />- refresh_rate_hz > 0 |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 12:08:06 +0200
-->