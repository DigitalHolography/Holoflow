# Dual Reader Batch Queue

The **Dual Reader Batch Queue** is an asynchronous circular-buffer bridge for centered streaming
corrections. Each consumer pop exposes:

1. the current frame `I[n]`;
2. a delayed frame `I[n-h]`, with `h = floor((window_size - 1) / 2)`;
3. a host `uint8` validity scalar.

The validity scalar becomes nonzero after `window_size` current frames have been observed
(`n >= window_size - 1`). Before the delayed reader reaches its first stored frame, its output
aliases zero-filled scratch storage. Even windows use the newer of the two frames surrounding the
half-frame window center.

The current and delayed outputs have independent read cursors. A circular-buffer slot remains
reserved until both readers have released it, so the producer cannot overwrite data still needed
by the delayed path.

The input may contain multiple frames on its leading axis. Consumer outputs always have a leading
dimension of one, preserving one-frame downstream cadence while allowing batched producer writes.

## Settings

- `target_capacity`: producer/consumer headroom beyond the retained delay; must be positive.
- `window_size`: averaging window used to derive the fixed lag and validity warm-up; must be
  positive.
