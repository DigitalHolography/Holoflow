# Copyright 2026 Digital Holography Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Symbolic graph construction and generic runtime orchestration."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
from threading import Lock
from typing import Any, Callable

from . import _native
from ._diagnostics import Artifacts

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

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        items = []
        for item in keys:
            if isinstance(item, slice):
                items.append({"start": item.start, "stop": item.stop,
                              "step": 1 if item.step is None else item.step})
            elif isinstance(item, int):
                items.append(item)
            else:
                raise TypeError("Symbolic slicing supports integer indices and slices")
        return operation("Slice", (self,), {"slices": items})

    def _binary(self, other, kind, *, reflected=False):
        if isinstance(other, Tensor):
            left, right = (other, self) if reflected else (self, other)
            return operation(kind, (left, right), {})
        if not isinstance(other, (int, float, complex)):
            return NotImplemented

        from ._ops import asarray

        tensor, scalar = self, asarray(other)
        if kind == "Add" and tensor.dtype != scalar.dtype:
            dtype = (
                _native.DType.complex64
                if _native.DType.complex64 in (tensor.dtype, scalar.dtype)
                else _native.DType.float32
            )
            if tensor.dtype != dtype:
                tensor = tensor.astype(dtype)
            if scalar.dtype != dtype:
                scalar = scalar.astype(dtype)
        left, right = (scalar, tensor) if reflected else (tensor, scalar)
        return operation(kind, (left, right), {})

    def __add__(self, other):
        return self._binary(other, "Add")

    def __radd__(self, other):
        return self._binary(other, "Add", reflected=True)

    def __mul__(self, other):
        return self._binary(other, "Multiply")

    def __rmul__(self, other):
        return self._binary(other, "Multiply", reflected=True)

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


class CancelledError(RuntimeError):
    """Execution stopped before the graph reached normal EOF."""


@dataclass(frozen=True)
class ArrayResult:
    """Symbolic collection output, resolved by Executable.result()."""

    _builder: Any
    _vertex: int


def _map_results(value, resolve):
    if isinstance(value, ArrayResult):
        return resolve(value)
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_map_results(v, resolve) for v in value)
    if isinstance(value, list):
        return [_map_results(v, resolve) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _map_results(v, resolve) for k, v in value.items()}
    raise TypeError("Traced returns must be ArrayResult handles, None, or tuples/lists/string-keyed dictionaries of them")


@dataclass(frozen=True)
class Event:
    node_id: str
    data: dict[str, Any]


class Executable:
    """Own a native executable. Lifecycle mutations must not overlap."""

    def __init__(self, native, outputs=None, artifacts=None):
        self._native = native
        self.artifacts = artifacts
        self._outputs = outputs
        self._arrays = {}
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

    def close(self, *, _finalize_diagnostics=True) -> None:
        self._acquire()
        try:
            if self._native is None:
                return
            native = self._native
            try:
                native.close(_finalize_diagnostics)
            finally:
                self._joined = True
                self._native = None
        finally:
            self._lifecycle.release()

    def result(self):
        """Wait for the entire graph and resolve the traced collector return structure."""
        self._acquire()
        try:
            native = self._get()
            if not self._started:
                raise RuntimeError("Executable has not been started")
            if not self._joined:
                try:
                    native.wait()
                finally:
                    self._joined = True
            native.check_failure()
            if not native.completed():
                raise CancelledError("Execution stopped before normal EOF")
            def resolve(handle):
                if handle._vertex not in self._arrays:
                    self._arrays[handle._vertex] = native.array_result(handle._vertex)
                return self._arrays[handle._vertex]
            return _map_results(self._outputs, resolve)
        finally:
            self._lifecycle.release()

    def send_event(self, node_id: str, data: dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            raise TypeError("Event data must be a JSON object")
        return self._get().send_event(node_id, json.dumps(data, allow_nan=False))

    def receive_event(self) -> Event | None:
        event = self._get().receive_event()
        return None if event is None else Event(event[0], json.loads(event[1]))

    def completed(self) -> bool:
        """Return whether the finite graph reached normal EOF."""
        return self._get().completed()

    def metrics(self) -> dict:
        return json.loads(self._get().metrics())

    def latency(self, node_id: str) -> dict:
        """Return cumulative latency statistics for the named output probe."""
        return json.loads(self._get().latency(node_id))

    def progress(self) -> dict:
        """Cumulative successful emissions and a native monotonic timestamp."""
        return json.loads(self._get().progress())

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
    def __init__(self, native, outputs=None):
        self._native = native
        self._outputs = outputs

    def to_json(self) -> dict:
        return json.loads(self._native.to_json())

    def to_dot(self) -> str:
        return self._native.to_dot()

    def compile(self, log_dir=None, *, capture=True) -> Executable:
        """Compile with optional diagnostics and Perfetto capture in a unique directory."""
        artifacts = None if log_dir is None else Artifacts.create(log_dir)
        return self._compile(artifacts, capture=capture)

    def _compile(self, artifacts=None, *, capture=True, previous=None):
        if artifacts is not None:
            artifacts.descriptor_json.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
            artifacts.descriptor_dot.write_text(self.to_dot(), encoding="utf-8")
        try:
            log_dir = "" if artifacts is None else str(artifacts.directory)
            if previous is None:
                native = self._native.compile(log_dir, capture)
            else:
                previous._acquire()
                try:
                    old = previous._get()
                    if not previous._joined or not old.completed():
                        raise ValueError("Graph update requires a completed and joined finite run")
                    try:
                        native = self._native.compile(log_dir, capture, old)
                    finally:
                        previous._native = None
                finally:
                    previous._lifecycle.release()
            return Executable(native, self._outputs, artifacts)
        except BaseException as error:
            if artifacts is not None:
                error.add_note(f"Holoflow diagnostics: {artifacts.directory}")
                try:
                    _native.flush_diagnostics()
                except Exception as cleanup_error:
                    error.add_note(f"Diagnostic finalization failed: {cleanup_error}")
            raise


def trace(fn: Callable, *args, **kwargs) -> Graph:
    """Run fn once to record native tasks; ordinary Python side effects run once."""
    if _active.get() is not None:
        raise RuntimeError("Nested hf.trace() calls are not supported")
    builder = _native.Builder()
    token = _active.set(builder)
    try:
        outputs = fn(*args, **kwargs)
        def validate(handle):
            if handle._builder is not builder:
                raise ValueError("Returned collector must belong to the active trace")
            return handle
        _map_results(outputs, validate)
        return Graph(builder.finalize(), outputs)
    finally:
        _active.reset(token)
