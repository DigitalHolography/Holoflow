# Record Settings

This header defines recording-related callback types, the settings and callback bundles used by the camera API, and JSON conversion declarations for `holocam::RecordSettings`.

## Type aliases

| Alias | Type | Description |
| --- | --- | --- |
| `RecordingSessionId` | `uint32_t` | Identifier for a recording session. |
| `RecordBufferFlushingFinishedCallback` | `std::function<void(RecordingSessionId)>` | Callback invoked with a session ID when buffer flushing finishes. |
| `RecordPollMetricsCallback` | `std::function<void(RecordingSessionId, size_t, size_t, size_t)>` | Callback receiving a session ID and three `size_t` metrics; the header comment labels these frame written, writing speed, and elapsed time (seconds). |
| `RecordAcquisitionMetricsCallback` | `std::function<void(RecordingSessionId, size_t)>` | Callback receiving a session ID and one `size_t` metric. |
| `RecordSuccessCallback` | `std::function<void(RecordingSessionId, size_t)>` | Success callback receiving a session ID and a `size_t` value. |
| `RecordFailureCallback` | `std::function<void(RecordingSessionId, const std::string &)>` | Failure callback receiving a session ID and an error message. |
| `RecordOptions` | `std::pair<RecordSettings, RecordCallbacks>` | Bundles recording settings with their callbacks. |

## `RecordSettings`

| Field / method | Type | Description |
| --- | --- | --- |
| `file_path` | `std::string` | Recording file path. |
| `recording_count` | `int` | Recording count setting. |
| `pipeline_config` | `nlohmann::json` | Pipeline configuration. |
| `record_queue_count` | `size_t` | Number of records in the record queue. |
| `operator==(const RecordSettings &other)` | `bool` | Compares the four fields above for equality. |
| `requires_rebuild(const RecordSettings &old)` | `bool` | Declared predicate indicating whether the settings require a rebuild. |

The header declares `to_json(nlohmann::json &, const RecordSettings &)` and `from_json(const nlohmann::json &, RecordSettings &)`. It also defines a `fmt` formatter that prints `file_path` and `recording_count`.

## `RecordCallbacks`

| Field | Callback alias | Description |
| --- | --- | --- |
| `flushing_finished` | `RecordBufferFlushingFinishedCallback` | Buffer flushing completion callback. |
| `poll_metrics` | `RecordPollMetricsCallback` | Polling metrics callback. |
| `acquisition_metrics` | `RecordAcquisitionMetricsCallback` | Acquisition metrics callback. |
| `success_callback` | `RecordSuccessCallback` | Recording success callback. |
| `failure_callback` | `RecordFailureCallback` | Recording failure callback. |

**Example:**

```cpp
holocam::RecordSettings settings{
    .file_path = "run.holo",
    .recording_count = 100,
    .pipeline_config = nlohmann::json::object(),
    .record_queue_count = 4,
};
holocam::RecordCallbacks callbacks{};
holocam::RecordOptions options{settings, callbacks};
```
