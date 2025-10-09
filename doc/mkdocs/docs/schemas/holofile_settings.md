# HolofileSettings

<!-- **Title:** HolofileSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Holofile reader task.
 -->

**Example:**

```json
{
    "path": "C:/data/run01.holo",
    "load_kind": "Live",
    "start_frame": 0,
    "end_frame": 1024,
    "batch_size": 8
}
```

| Property                       | Pattern | Type             | Deprecated | Definition | Title/Description                   |
| ------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------------------------- |
| + [path](#path )               | No      | string           | No         | -          | Path to the HoloFile.               |
| + [load_kind](#load_kind )     | No      | enum (of string) | No         | -          | Loading strategy.                   |
| + [start_frame](#start_frame ) | No      | integer          | No         | -          | First frame to read (inclusive).    |
| + [end_frame](#end_frame )     | No      | integer          | No         | -          | Last frame to read (exclusive).     |
| + [batch_size](#batch_size )   | No      | integer          | No         | -          | Number of frames per output tensor. |

## <a name="path"></a>1. Property `HolofileSettings > path`

<!--  -->

<!-- |              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |
 -->

<!-- **Description:** Path to the HoloFile.
 -->

| Restrictions   |   |
| -------------- | - |
| **Min length** | 1 |

## <a name="load_kind"></a>2. Property `HolofileSettings > load_kind`

<!--  -->

<!-- |              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |
 -->

<!-- **Description:** Loading strategy.
 -->

Must be one of:

* "Live"
* "CPUCached"
* "GPUCached"

## <a name="start_frame"></a>3. Property `HolofileSettings > start_frame`

<!--  -->

<!-- |              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |
 -->

<!-- **Description:** First frame to read (inclusive).
 -->

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

## <a name="end_frame"></a>4. Property `HolofileSettings > end_frame`

<!--  -->

<!-- |              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |
 -->

<!-- **Description:** Last frame to read (exclusive).
 -->

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

## <a name="batch_size"></a>5. Property `HolofileSettings > batch_size`

<!--  -->

<!-- |              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |
 -->

<!-- **Description:** Number of frames per output tensor.
 -->

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 1 |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-09 at 16:35:01 +0200
-->