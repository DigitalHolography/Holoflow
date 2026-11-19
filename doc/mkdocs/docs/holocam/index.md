# Holocam

Holocam is the C++ camera abstraction layer used by Holovibes for high-speed image acquisition. It defines a common interface for supported high-speed cameras, a factory for instantiating them from configuration, and the recording settings and callbacks used during acquisition.

Its API is organized into four headers:

- **Camera registry** — [camera info](camera_info.md) describes the supported camera models (Phantom S710, S711, S991) and the `name_to_info` lookup table.
- **Recording configuration** — [record settings](record_settings.md) defines `RecordSettings`, `RecordCallbacks`, `RecordOptions`, and the session/metrics callback types.
- **Camera interface** — [camera base](camera.md) documents `CameraBase`, the abstract interface exposing acquisition control, recording, and frame access.
- **Instantiation** — [camera factory](factory.md) shows how `CameraFactory::create` builds a camera from a name, JSON config, and optional record options.

Use the navigation to browse the full API reference and the members of each header.