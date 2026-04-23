<!-- # HolofileSettings -->

<!-- **Title:** HolofileSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Holofile reader task.
 -->

**Example:**

```json
{
    "path": "C:/data/run01.holo",
    "load_kind": "Live",
    "start_frame": 0,
    "end_frame": 1024,
    "batch_size": 8,
    "max_fps": 120
}
```

| Property                       | Pattern | Type             | Deprecated | Definition | Title/Description                                                                                                                                                                          |
| ------------------------------ | ------- | ---------------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| + [path](#path )               | No      | string           | No         | -          | Path to the HoloFile. Must be a non-empty string.                                                                                                                                          |
| + [load_kind](#load_kind )     | No      | enum (of string) | No         | -          | Loading strategy. Accepted values:<br />- Live: Read on demand from disk.<br />- CPUCached: Preload all frames into CPU memory.<br />- GPUCached: Preload all frames into GPU memory.      |
| + [start_frame](#start_frame ) | No      | integer          | No         | -          | First frame to read (inclusive). Must satisfy:<br />- start_frame >= 0<br />- start_frame < number of frames in file                                                                        |
| + [end_frame](#end_frame )     | No      | integer          | No         | -          | Last frame to read (exclusive). Must satisfy:<br />- end_frame >= start_frame<br />- end_frame - start_frame >= batch_size<br />- end_frame <= number of frames in file                      |
| + [batch_size](#batch_size )   | No      | integer          | No         | -          | Number of frames per output tensor. Must satisfy:<br />- batch_size > 0<br />- batch_size <= (end_frame - start_frame)                                                                      |
| - [max_fps](#max_fps )         | No      | integer          | No         | -          | Optional playback cap in frames per second. When omitted, the holofile is read as fast as possible.                                                                                         |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2026-04-23 at 00:00:00 +0200
-->
