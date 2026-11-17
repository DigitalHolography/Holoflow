"""Streaming sink graph operations; commands are sent through executable events."""

from os import fspath
from ._graph import Tensor, operation


def holofile(image: Tensor, path, *, count: int, node_id: str | None = None) -> None:
    operation("HolofileWriter", (image,), {
        "path": fspath(path), "count": count, "pipeline_settings": {}, "use_buffer": True,
    }, node_id)


def void(x: Tensor, *, node_id: str | None = None) -> None:
    operation("Void", (x,), {}, node_id)
