# Ametek S710 Euresys Coaxlink Octo Sync Task
The **Ametek S710 Euresys Coaxlink Octo** source node acquires frames from a Phantom S710 camera using the Euresys GenTL / EGrabber stack and publishes them as host-memory tensors. The node reads a JSON camera configuration (`cfg_path`) at creation time to determine pixel format, frame geometry and buffer counts, configures the camera and grabber accordingly, and announces host-side buffers for zero-copy delivery.

Supported behaviors and guarantees:

- Reads camera configuration from the provided `cfg_path` (JSON) and validates required fields (e.g. `PixelFormat`, `Width`, `Height`, `BufferPartCount`).
- Maps supported pixel formats (`Mono8` → `uint8`, `Mono16` → `uint16`) to output tensor dtype.
- Configures the camera (device, remote, stream modules) using GenApi calls and sets up announced host buffers followed by a streaming grabber.
- Produces an output tensor shape `(BufferPartCount, Height, Width)` in host memory; the factory validates the config file and the available camera before creating the task.

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
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\sources\ametek_s710_euresys_coaxlink_octo_settings.md"
