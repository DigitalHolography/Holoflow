<!-- # AmetekS711EuresysCoaxlinkQSFPSettings

- [1. Property `AmetekS711EuresysCoaxlinkQSFPSettings > cfg_path`](#cfg_path)

**Title:** AmetekS711EuresysCoaxlinkQSFPSettings

|                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |

**Description:** Settings for the camera source. -->

**Example:**

With record settings:
```json
{
    "cfg_path": "configs/phantom_s711_config.json",
    "name": "AmetekS711CoaxlinkQSFP+",
    "record_settings": {
       "file_path": "record.holo",
       "recording_count": 4096,
       "record_queue_count": 3,
       "pipeline_config": { ... }
    }
}
```

With record settings set to `null`:
```json
{
    "cfg_path": "configs/phantom_s711_config.json",
    "name": "AmetekS711CoaxlinkQSFP+",
    "record_settings": null
}
```

Without record settings:
```json
{
    "cfg_path": "configs/phantom_s711_config.json",
    "name": "AmetekS711CoaxlinkQSFP+",
}

```
| Property                 | Pattern | Type   | Deprecated | Definition | Title/Description                                                                                                                                                                                          |
| ------------------------ | ------- | ------ | ---------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [cfg_path](#cfg_path ) | No      | string | No         | -          | Path to the JSON configuration file for the camera. The file must contain a top-level the short name of the camera ("AmetekS711CoaxlinkQSFP+" -> "s711") object with camera parameters (Width, Height, PixelFormat, BufferPartCount, ExposureTime, etc.). |
| + [name](#name ) | No      | string | No         | -          | Name of the camera (ex: "AmetekS711CoaxlinkQSFP+")|
| + [record_settings](#record_settings ) | No      | object | No         | -          | Record settings of the camera. It is nullable and optional. If specified it must contains the following fields: filepath (string), recording_count (int > 0), recording_queue_count (int > 0), pipeline_settings (object) |
