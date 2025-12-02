<!-- # HolofileSettings

- [1. Property `HolofileSettings > path`](#path)
- [2. Property `HolofileSettings > count`](#count)
- [3. Property `HolofileSettings > pipeline_settings`](#pipeline_settings)

**Title:** HolofileSettings

|                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |

**Description:** Settings for the Holofile sink task. -->

**Example:**

```json
{
    "path": "recordings/session_001.hf",
    "count": 250,
    "pipeline_settings": {
        "exposure_ms": 5.0,
        "gain": 1.2,
        "pipeline_version": "v1.3.0"
    }
}
```

| Property                                   | Pattern | Type                                           | Deprecated | Definition | Title/Description                                                                                          |
| ------------------------------------------ | ------- | ---------------------------------------------- | ---------- | ---------- | ---------------------------------------------------------------------------------------------------------- |
| + [path](#path )                           | No      | string                                         | No         | -          | Filesystem path where the HoloFile will be written. Must be a non-empty string.                            |
| + [count](#count )                         | No      | integer                                        | No         | -          | Number of frames to write. Must be a positive integer.                                                     |
| + [pipeline_settings](#pipeline_settings ) | No      | object, array, string, number, boolean or null | No         | -          | Arbitrary pipeline settings JSON that will be stored in the file footer. Any valid JSON value is accepted. |

<!-- ## <a name="path"></a>1. Property `HolofileSettings > path`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Filesystem path where the HoloFile will be written. Must be a non-empty string.

| Restrictions   |   |
| -------------- | - |
| **Min length** | 1 |

## <a name="count"></a>2. Property `HolofileSettings > count`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** Number of frames to write. Must be a positive integer.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 1 |

## <a name="pipeline_settings"></a>3. Property `HolofileSettings > pipeline_settings`

|              |                                                  |
| ------------ | ------------------------------------------------ |
| **Type**     | `object, array, string, number, boolean or null` |
| **Required** | Yes                                              |

**Description:** Arbitrary pipeline settings JSON that will be stored in the file footer. Any valid JSON value is accepted.

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-12-02 at 14:50:10 +0100 --> -->
