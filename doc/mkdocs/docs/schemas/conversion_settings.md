# ConversionSettings

<!-- **Title:** ConversionSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Conversion sync task.
 -->

**Example:**

```json
{
    "target": "F32",
    "strategy": "Modulus"
}
```

| Property                 | Pattern | Type             | Deprecated | Definition | Title/Description                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------ | ------- | ---------------- | ---------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [target](#target )     | No      | enum (of string) | No         | -          | Desired output data type. Accepted values:<br />- U8: 8-bit unsigned integer<br />- U16: 16-bit unsigned integer<br />- F32: 32-bit float<br />- CF32: 32-bit complex float                                                                                                                                                                                                                                           |
| + [strategy](#strategy ) | No      | enum (of string) | No         | -          | Conversion strategy to apply. Accepted values:<br />- Real: Extract real component (valid when target is CF32 and the input is U8/U16)<br />- Scaled: Normalize float input to integer range (valid when target is U8 or U16 and the input is F32)<br />- Modulus: Emit magnitude (valid when target is F32 and the input is CF32)<br />- Argument: Emit phase angle (valid when target is F32 and the input is CF32) |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 11:09:40 +0200
-->