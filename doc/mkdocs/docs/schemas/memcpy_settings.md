# MemcpySettings

<!-- **Title:** MemcpySettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Memcpy sync task.
 -->

**Example:**

```json
{
    "target": "Device"
}
```

| Property             | Pattern | Type             | Deprecated | Definition | Title/Description                                                                                                             |
| -------------------- | ------- | ---------------- | ---------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| + [target](#target ) | No      | enum (of string) | No         | -          | Destination memory space for the copy. Accepted values:<br />- Host: Copy into CPU memory<br />- Device: Copy into GPU memory |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 11:09:40 +0200
-->