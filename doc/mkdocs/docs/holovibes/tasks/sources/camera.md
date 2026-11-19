# Camera Sync Task

The **Camera** source node acquires frames from supported Ametek cameras using the `holocam` camera abstraction and publishes them as host-memory tensors.

The node is configured through a JSON camera configuration (`cfg_path`) and a camera name (`name`). The camera name determines which configuration section is used and which camera implementation is instantiated.

Supported camera configurations include:
- `AmetekS710EuresysCoaxlinkOcto` → `"s710"`
- `AmetekS711EuresysCoaxlinkQSFP+` → `"s711"`
- `AmetekS991EuresysCoaxlinkQSFP+` → `"s991"`

The node also supports optional recording settings, allowing recording to be started, stopped and monitored through runtime events.

Supported behaviors and guarantees:

- Reads camera configuration from the provided `cfg_path` (JSON) and selects the appropriate configuration section according to `name`.
- Determines the output tensor dimensions from the camera configuration:
  - `Width`: output width.
  - `FinalHeight`: output height, falling back to `Height` when `FinalHeight` is not provided.
  - `BufferPartCount`: number of parts per output buffer.
- Maps supported pixel formats (`Mono8` → `uint8`, `Mono16` → `uint16`) to the output tensor dtype.
- Creates the appropriate camera implementation through `holocam::CameraFactory`.
- Starts camera acquisition when the task begins executing and stops acquisition when the task is cancelled or destroyed.
- Publishes acquired frames as host-memory tensors.
- Requeues the acquired camera frame when the scheduler releases the output tensor, allowing the camera to reuse its buffer.
- Supports optional recording through `record_settings`, including runtime recording control and progress notifications.
- Supports updating recording paths without rebuilding the camera.
- Supports camera configuration updates. When the camera configuration changes, the camera is rebuilt. Otherwise, settings can be updated in place when the recording configuration does not require a rebuild.

Recording is controlled through the following events:

- `start_recording`: Starts a recording using the provided `record_path`.
- `stop_recording`: Stops the active recording and removes the incomplete recording file if applicable.
- `update_recording_path`: Updates the destination path of the current recording.

The node can emit the following events to the UI:

- `recording_update`: Reports the number of frames written, writing speed and elapsed time.
- `acquisition_update`: Reports the number of acquired frames.
- `recording_finished`: Reports the recording session ID and number of frames written.
- `recording_failed`: Reports the recording session ID and failure message.
- `record_flushing_finished`: Indicates that the recording buffer flushing process has completed.

The recording callbacks are only connected when recording settings are configured.

!!! warning
    The configuration file at `cfg_path` must exist and be readable by the process. The selected camera name must be supported and its corresponding configuration section must contain the required fields.

    Camera acquisition requires the library to be built with EGrabber support. Otherwise, camera factory operations will throw an exception.

## Inputs

This task consumes **no** input tensors (it is a source).

The factory enforces:

- No input descriptors are expected.

## Outputs

This task produces a single output tensor described by the camera configuration:

- Shape: `(BufferPartCount, FinalHeight, Width)`, where `FinalHeight` falls back to `Height` if not specified.
- DType: mapped from `PixelFormat` in the selected configuration section (`Mono8` → `uint8`, `Mono16` → `uint16`).
- Memory location: Host memory.

The output tensor contains the acquired camera frame data.

## Inplace

This task does not support inplace operation.

## Ownership

This task owns its output buffers.

The scheduler releases each output after downstream use. When the output is released, the task calls `camera_->release_frame()` to return the acquired frame buffer to the camera for reuse.

## Settings

The task accepts the following settings:

- `cfg_path`: Path to the JSON camera configuration file.
- `name`: Camera implementation identifier. Determines which camera configuration section is used.
- `record_settings` (optional): Recording configuration. When provided, recording callbacks are configured to report acquisition progress, writing progress, completion and failure events to the UI.

For the complete settings definition:

--8<-- "docs/schemas/sources/camera_settings.md"