# Camera Info

`holocam::camera_info` describes the identifying and pixel-size metadata for a supported camera. The header also provides three predefined camera records and an inline lookup registry keyed by the full camera name.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `name` | `const char *` | Full camera identifier, used as the registry key. |
| `short_name` | `const char *` | Short camera identifier; also used as a concise model/config identifier. |
| `model_name` | `const char *` | Human-readable camera model name. |
| `pixel_size` | `float` | Pixel size in micrometers. |

## Predefined entries

| Entry | Full name (`name`) | Short name | Model name | Pixel size |
| --- | --- | --- | --- | ---: |
| `s710_info` | `AmetekS710EuresysCoaxlinkOcto` | `s710` | `Phantom S710` | `20.0f` |
| `s711_info` | `AmetekS711EuresysCoaxlinkQSFP+` | `s711` | `Phantom S711` | `20.0f` |
| `s991_info` | `AmetekS991EuresysCoaxlinkQSFP+` | `s991` | `Phantom S991` | `6.75f` |

## Registry

`name_to_info` is an inline `std::map<std::string_view, camera_info>` containing the three predefined entries. Each entry is keyed by that record's full `name` (not its `short_name`).

**Example:**

```cpp
const auto it = holocam::name_to_info.find("AmetekS710EuresysCoaxlinkOcto");
if (it != holocam::name_to_info.end()) {
    const holocam::camera_info &info = it->second;
    // info.short_name == "s710"
}
```
