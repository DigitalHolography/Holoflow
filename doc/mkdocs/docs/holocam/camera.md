# Camera Base

`holocam::CameraBase` is the abstract camera interface. Implementations provide camera identity and hardware metadata; the base class exposes recording controls, frame access, and configuration.

## Public interface

| Method | Return type | Description |
| --- | --- | --- |
| `virtual ~CameraBase()` | — | Virtual destructor. |
| `name() const noexcept` | `const char *` | Pure virtual; returns the full camera name. The header specifies string-literal identity strings. |
| `short_name() const noexcept` | `const char *` | Pure virtual; returns the short camera name. |
| `model_name() const noexcept` | `const char *` | Pure virtual; returns the model name. |
| `grabber_count() const noexcept` | `size_t` | Pure virtual; returns the grabber count. |
| `pixel_size() const noexcept` | `float` | Pure virtual; returns pixel size in micrometers. |
| `set_record_options(std::optional<RecordOptions> o)` | `void` | Sets or clears the optional recording options. |
| `update_record_path(const std::string &path)` | `void` | Updates the recording path. |
| `start_acquisition()` | `void` | Starts acquisition. |
| `stop_acquisition()` | `void` | Stops acquisition. |
| `start_raw_record()` | `bool` | Starts raw recording; reports a boolean result. |
| `stop_raw_record()` | `void` | Stops raw recording. |
| `cancel_all_records()` | `void` | Cancels all records. |
| `current_recording_session_id()` | `RecordingSessionId` | Returns the current recording session ID. |
| `recording()` | `bool` | Reports whether the camera is recording. |
| `get_frame(std::atomic<bool> *cancelled)` | `std::byte *` | Returns the next frame in the buffer queue and makes it the pending frame; returns `nullptr` if cancelled. Any existing pending frame is released before replacement. |
| `release_frame()` | `void` | Releases the pending frame, if any. |
| `configuration() const noexcept` | `const nlohmann::json &` | Returns the camera configuration. |

## Protected members

| Member | Type | Description |
| --- | --- | --- |
| `ConfigurationCallback` | `std::function<void(Impl *, const nlohmann::json &cfg)>` | Callback type used to configure the implementation from JSON. |
| `CameraBase(...)` | Protected constructor | Takes JSON configuration, optional record options, short name, model name, grabber count, and a configuration callback. |
| `impl_` | `std::unique_ptr<Impl>` | Owned pointer to the implementation object. `Impl` is declared as a nested class. |

## Example

A derived camera supplies the pure virtual identity and hardware-information methods; the remaining public operations are inherited from `CameraBase`.

```cpp
class ExampleCamera : public holocam::CameraBase {
public:
    using CameraBase::CameraBase;

    const char *name() const noexcept override { return "ExampleCamera"; }
    const char *short_name() const noexcept override { return "example"; }
    const char *model_name() const noexcept override { return "Example Model"; }
    size_t grabber_count() const noexcept override { return 1; }
    float pixel_size() const noexcept override { return 10.0f; }
};
```
