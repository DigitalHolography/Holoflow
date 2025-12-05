<!-- # AmetekS711EuresysCoaxlinkQSFPSettings

- [1. Property `AmetekS711EuresysCoaxlinkQSFPSettings > cfg_path`](#cfg_path)

**Title:** AmetekS711EuresysCoaxlinkQSFPSettings

|                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |

**Description:** Settings for the Ametek S711 Euresys Coaxlink QSFP camera source. -->

**Example:**

```json
{
    "cfg_path": "configs/phantom_s711_config.json"
}
```

| Property                 | Pattern | Type   | Deprecated | Definition | Title/Description                                                                                                                                                                                          |
| ------------------------ | ------- | ------ | ---------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [cfg_path](#cfg_path ) | No      | string | No         | -          | Path to the JSON configuration file for the Phantom S711 camera. The file must contain a top-level 's711' object with camera parameters (Width, Height, PixelFormat, BufferPartCount, ExposureTime, etc.). |

<!-- ## <a name="cfg_path"></a>1. Property `AmetekS711EuresysCoaxlinkQSFPSettings > cfg_path`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Path to the JSON configuration file for the Phantom S711 camera. The file must contain a top-level 's711' object with camera parameters (Width, Height, PixelFormat, BufferPartCount, ExposureTime, etc.).

| Restrictions   |   |
| -------------- | - |
| **Min length** | 1 |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-12-02 at 17:37:51 +0100 -->
