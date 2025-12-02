<!-- # AmetekS710EuresysCoaxlinkOctoSettings

- [1. Property `AmetekS710EuresysCoaxlinkOctoSettings > cfg_path`](#cfg_path)

**Title:** AmetekS710EuresysCoaxlinkOctoSettings

|                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |

**Description:** Settings for the Ametek S710 Euresys Coaxlink Octo camera source. -->

**Example:**

```json
{
    "cfg_path": "configs/phantom_s710_config.json"
}
```

| Property                 | Pattern | Type   | Deprecated | Definition | Title/Description                                                                                                                                                                                          |
| ------------------------ | ------- | ------ | ---------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [cfg_path](#cfg_path ) | No      | string | No         | -          | Path to the JSON configuration file for the Phantom S710 camera. The file must contain a top-level 's710' object with camera parameters (Width, Height, PixelFormat, BufferPartCount, ExposureTime, etc.). |

<!-- ## <a name="cfg_path"></a>1. Property `AmetekS710EuresysCoaxlinkOctoSettings > cfg_path`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Path to the JSON configuration file for the Phantom S710 camera. The file must contain a top-level 's710' object with camera parameters (Width, Height, PixelFormat, BufferPartCount, ExposureTime, etc.).

| Restrictions   |   |
| -------------- | - |
| **Min length** | 1 |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-12-02 at 16:10:49 +0100 -->
