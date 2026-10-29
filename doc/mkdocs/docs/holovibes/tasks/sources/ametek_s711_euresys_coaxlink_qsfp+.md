# AmetekS711EuresysCoaxlinkQSFP Sync Task
The **AmetekS711EuresysCoaxlinkQSFP** source node acquires frames from a Phantom S711 camera using the Euresys GenTL / EGrabber stack and publishes them as host-memory tensors. The node reads a JSON camera configuration (`cfg_path`) at creation time to determine pixel format, frame geometry and buffer counts, configures the camera and grabber accordingly, and announces host-side buffers for zero-copy delivery.

Supported behaviors and guarantees:

- Reads camera configuration from the provided `cfg_path` (JSON) and validates required fields under the `"s711"` key (e.g. `PixelFormat`, `Width`, `Height`, `BufferPartCount`).
- Maps supported pixel formats (`Mono8` → `uint8`, `Mono16` → `uint16`) to output tensor dtype.
- Configures the camera (device, remote, stream modules) using GenApi calls and sets up announced host buffers followed by a streaming grabber.
- Produces an output tensor shape `(BufferPartCount, Height, Width)` in host memory; the factory validates the config file and the available camera before creating the task.
- Publishes a two-bank buffer only when both banks report the same host buffer address and each reports `BufferPartCount` delivered parts. Rejected buffers are requeued; accepted-pair metadata and rejection warnings are logged at most once per second.

Matching buffer addresses identify a shared memory slot, not necessarily matching camera frames. The task also reports signed bank A minus bank B buffer and per-part timestamp offsets, plus the first pair after each pipeline update. Once per second it reports each grabber's raw `BUFFER_INFO_FRAMEID`, largest observed ID step, rejected-frame and broken-frame event counts, lost-buffer (queue-underrun) count, and queued/awaiting buffer counts. Frame IDs are read at buffer level; their steps are **not** interpreted as a count of missed individual frames in high-frame-rate mode. Counter changes since the previous sample and, on the first resumed report, since the factory's update-time snapshot are reported without resetting the hardware counters. If a diagnostic is unavailable, acquisition continues and the warning is limited to once per second.

!!! warning
    The configuration file at `cfg_path` must exist and be readable by the process. The node will throw if the file cannot be opened or does not contain a supported `PixelFormat`. Also ensure the camera hardware is present and accessible; missing hardware or insufficient privileges will cause creation to fail.

## Inputs
This task consumes **no** input tensors (it is a source).

The factory enforces:

- No input descriptors are expected.

## Outputs
This task produces a single output tensor described by the camera configuration:

- Shape: `(BufferPartCount, Height, Width)`.
- DType: mapped from the `PixelFormat` in the config (`Mono8` → `uint8`, `Mono16` → `uint16`).
- Memory location: Host memory.

The output tensor bytes reflect raw camera frame parts as configured by `BufferPartCount`.

## Inplace
This task does not support inplace operation.

## Ownership
This task owns its output buffers. The scheduler releases each output after downstream use, at which point the task requeues the buffer to both grabbers.

---
## Settings
--8<-- "docs/schemas/sources/ametek_s711_euresys_coaxlink_qsfp+_settings.md"
