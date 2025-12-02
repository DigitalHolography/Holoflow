<!-- # PcaSettings -->

<!-- **Title:** PcaSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the PCA sync task.
 -->

**Example:**

```json
{
    "begin": 0,
    "end": 8
}
```

| Property           | Pattern | Type    | Deprecated | Definition | Title/Description                                                                                                           |
| ------------------ | ------- | ------- | ---------- | ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| + [begin](#begin ) | No      | integer | No         | -          | Inclusive index of the first eigencomponent to keep. Must satisfy:<br />- begin >= 0<br />- begin < end                     |
| + [end](#end )     | No      | integer | No         | -          | Exclusive index of the last eigencomponent to keep. Must satisfy:<br />- end > begin<br />- end <= number of input features |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 13:11:10 +0200
-->