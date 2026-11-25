**Example (spatial and temporal contrast):**

```json
{
    "window_size": 5,
    "temporal_window_size": 16,
    "method": "kernel"
}
```

| Property | Type | Description |
| --- | --- | --- |
| `window_size` | integer | `1` for temporal-only contrast, or an odd spatial side length from 3 to 255 pixels. Defaults to `7` when omitted. |
| `temporal_window_size` | integer | `1` for spatial-only contrast, or at least `3` frames for a temporal window. Defaults to `1` when omitted. |
| `method` | `"kernel"` or `"fft"` | Direct CUDA kernels or FFT-based spatial box convolution. Defaults to `"kernel"`. |
