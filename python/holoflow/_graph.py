"""Symbolic graph construction and generic runtime orchestration."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
from threading import Lock
from typing import Any, Callable

from . import _native

_active: ContextVar[_native.Builder | None] = ContextVar("holoflow_builder", default=None)


@dataclass(frozen=True)
class Tensor:
    """Immutable tensor metadata and producer provenance, without data storage."""

    _builder: Any
    _vertex: int
    _port: int
    _desc: Any

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._desc.shape)

    @property
    def dtype(self) -> _native.DType:
        return self._desc.dtype

    @property
    def device(self) -> str:
        return self._desc.device

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(self._desc.strides)

    @property
    def offset(self) -> int:
        return self._desc.offset

    def to(self, device: str, *, node_id: str | None = None) -> Tensor:
        if device not in ("host", "device"):
            raise ValueError("device must be 'host' or 'device'")
        return operation("Memcpy", (self,), {"target": device.title()}, node_id)

    def astype(self, dtype: _native.DType, strategy: str = "real",
               *, node_id: str | None = None) -> Tensor:
        targets = {_native.DType.uint8: "U8", _native.DType.uint16: "U16",
                   _native.DType.float32: "F32", _native.DType.complex64: "CF32"}
        strategies = {s.lower(): s for s in ("Real", "Scaled", "Modulus", "Argument")}
        if dtype not in targets or strategy not in strategies:
            raise ValueError("Unsupported conversion dtype or strategy")
        return operation("Conversion", (self,),
                         {"target": targets[dtype], "strategy": strategies[strategy]}, node_id)

    def reshape(self, shape: tuple[int, ...], copy: bool | None = None,
                *, node_id: str | None = None) -> Tensor:
        return operation("Reshape", (self,), {"shape": list(shape), "copy": copy}, node_id)

    def __add__(self, other):
        return operation("Add", (self, other), {})

    def __mul__(self, other):
        return operation("Multiply", (self, other), {})

    def __bool__(self) -> bool:
        raise TypeError("Symbolic tensor values are unavailable during tracing")

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Symbolic tensors cannot be converted to arrays")

    def item(self):
        raise TypeError("Symbolic tensor values are unavailable during tracing")


def operation(kind: str, inputs: tuple[Tensor, ...], settings: dict,
              node_id: str | None = None):
    builder = _active.get()
    if builder is None:
        raise RuntimeError("Holoflow operations require hf.trace()")
    if node_id is not None and (not isinstance(node_id, str) or not node_id):
        raise ValueError("node_id must be a nonempty string")
    for tensor in inputs:
        if not isinstance(tensor, Tensor) or tensor._builder is not builder:
            raise ValueError("Inputs must be symbolic tensors from the active trace")
    vertex, outputs = builder.add(
        kind, [(t._vertex, t._port) for t in inputs],
        json.dumps(settings, allow_nan=False), node_id,
    )
    tensors = tuple(Tensor(builder, vertex, i, desc) for i, desc in enumerate(outputs))
    if not tensors:
        return None
    return tensors[0] if len(tensors) == 1 else tensors


@dataclass(frozen=True)
class Event:
    node_id: str
    data: dict[str, Any]


class Executable:
    """Own a native executable. Lifecycle mutations must not overlap."""

    def __init__(self, native):
        self._native = native
        self._lifecycle = Lock()
        self._started = False
        self._joined = False

    def _get(self):
        if self._native is None:
            raise RuntimeError("Executable is closed")
        return self._native

    def _acquire(self):
        if not self._lifecycle.acquire(blocking=False):
            raise RuntimeError("Another executable lifecycle operation is in progress")

    def start(self) -> None:
        self._acquire()
        try:
            native = self._get()
            native.check_failure()
            if self._started:
                raise RuntimeError("Compile a fresh executable to run again")
            native.start()
            self._started = True
        finally:
            self._lifecycle.release()

    def stop(self) -> None:
        self._get().stop()

    def wait(self) -> None:
        self._acquire()
        try:
            native = self._get()
            if self._started and not self._joined:
                try:
                    native.wait()
                finally:
                    self._joined = True
            native.check_failure()
        finally:
            self._lifecycle.release()

    def close(self) -> None:
        self._acquire()
        try:
            if self._native is None:
                return
            native = self._native
            native.stop()
            try:
                if self._started and not self._joined:
                    native.wait()
                native.check_failure()
            finally:
                self._joined = True
                self._native = None
        finally:
            self._lifecycle.release()

    def send_event(self, node_id: str, data: dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            raise TypeError("Event data must be a JSON object")
        return self._get().send_event(node_id, json.dumps(data, allow_nan=False))

    def receive_event(self) -> Event | None:
        event = self._get().receive_event()
        return None if event is None else Event(event[0], json.loads(event[1]))

    def metrics(self) -> dict:
        return json.loads(self._get().metrics())

    def __enter__(self) -> Executable:
        self._get()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Graph:
    def __init__(self, native):
        self._native = native

    def to_json(self) -> dict:
        return json.loads(self._native.to_json())

    def to_dot(self) -> str:
        return self._native.to_dot()

    def compile(self) -> Executable:
        return Executable(self._native.compile())


def trace(fn: Callable, *args, **kwargs) -> Graph:
    """Run fn once to record native tasks; ordinary Python side effects run once."""
    if _active.get() is not None:
        raise RuntimeError("Nested hf.trace() calls are not supported")
    builder = _native.Builder()
    token = _active.set(builder)
    try:
        fn(*args, **kwargs)
        return Graph(builder.finalize())
    finally:
        _active.reset(token)
