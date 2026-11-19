# Camera Factory

`holocam::CameraFactory` creates a camera instance and returns it through the abstract `CameraBase` interface.

## `create`

| Signature | Description |
| --- | --- |
| `std::unique_ptr<CameraBase> create(std::string_view name, const nlohmann::json &config, const std::optional<RecordOptions> &record_options) const` | Creates a camera selected by `name`, using the supplied JSON configuration and optional recording options. Name should only be one of the name from the `camera_info.hh` header. |

## Example

```cpp
holocam::CameraFactory factory;
nlohmann::json config = nlohmann::json::object();
std::optional<holocam::RecordOptions> record_options;

auto camera = factory.create("AmetekS710EuresysCoaxlinkOcto",
                             config,
                             record_options);
```
