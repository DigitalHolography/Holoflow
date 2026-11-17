"""Small deterministic HoloFile fixture without NumPy dependencies."""

import math
from pathlib import Path
import struct


def write_holo(path, *, frames=1024, width=8, height=8):
    # Four identical 256-frame windows make recording start timing irrelevant.
    values = [int(2048 + 500 * math.sin(2 * math.pi * (t % 256) / 19 + x * .2 + y * .3)
                  + 300 * math.cos(2 * math.pi * (t % 256) / 31 + x * .4 - y * .1))
              for t in range(frames) for y in range(height) for x in range(width)]
    data = struct.pack(f"<{len(values)}H", *values)
    header = struct.pack("<IHHIIIQB35s", 0x4F4C4F48, 66, 16, width, height, frames,
                         len(data), 0, bytes(35))
    Path(path).write_bytes(header + data)
