# Holofile Sync Task
The **Holofile** sink task writes frames to a HoloFile on the host filesystem. Recording is controlled by the configured **path** and **count** settings and may also be started or stopped at runtime via events (`start_recording` / `stop_recording`) provided to the node. The HoloFile stores a header (geometry + frame count), raw frame data, and a footer containing the serialized `pipeline_settings`.

Supported behaviors and guarantees:

- Writes frames from a host-memory tensor to a HoloFile at the configured `path`.
- Accepts 8-bit unsigned (`U8`) and 16-bit unsigned (`U16`) image data and records the corresponding bits-per-pixel in the file header.
- Respects the configured `count`: once `count` frames have been written the task finalizes the file and emits a `recording_finished` event.
- If recording is stopped before `count` frames are written (either by a `stop_recording` event or task destructor), the incomplete file is removed to avoid partial recordings on disk.

!!! warning
    Ensure the configured `path` is writable and `count` is positive. If recording is interrupted before completion the partial file will be deleted. Also ensure sufficient disk space is available for `count * frame_size` bytes.

## Inputs
This task consumes a single input tensor (frames to write).

The input requirements (as enforced by `HolofileFactory::infer`) are:

- The tensor **must** have rank 3 (shape typically `(B, H, W)` where `B` is batch size).
- The tensor dtype **must** be either `uint8` (`U8`) or `uint16` (`U16`).
- The tensor **must** reside in Host memory.

## Outputs
This task produces no output tensors — it is a sink that writes frame data to disk.

The recording process writes raw frame bytes and a footer containing `pipeline_settings` to the configured file.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\sinks\holofile_settings.md"
